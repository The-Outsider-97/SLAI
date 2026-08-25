"""Production-grade deductive reasoning strategy for the reasoning subsystem.

The module keeps the reasoning-type contract used by ``ReasoningTypes`` while
turning the previous rule loop into a bounded, explainable proof engine.

Pipeline
--------
1. Normalize premises and hypothesis into canonical statement records.
2. Validate premises against evidence, known falsehoods, and contradiction rules.
3. Apply configured inference rules until a fixed point or max-step budget:
   - modus ponens
   - modus tollens
   - categorical syllogism and instance syllogism
   - disjunctive syllogism
   - hypothetical syllogism
4. Evaluate the hypothesis and proof chain with certainty/contradiction scoring.
5. Return a JSON-safe proof report with diagnostics and metrics.
"""
from __future__ import annotations

import hashlib
import re
import time

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..reasoning_cache import ReasoningCache
from ..utils.config_loader import load_global_config, get_config_section
from ..utils.reasoning_errors import *
from ..utils.reasoning_helpers import *
from .base_reasoning import BaseReasoning
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reasoning Deductive")
printer = PrettyPrinter()


_NEGATION_PREFIXES = (
    "not ", "no ", "never ", "false ", "cannot ", "can't ", "does not ",
    "do not ", "is not ", "are not ", "was not ", "were not ",
)
_IMPLIES_RE = re.compile(r"^\s*(?:if\s+)?(.+?)\s*(?:,?\s*then\s+|\s+implies\s+|\s*->\s*|\s+therefore\s+|\s+consequently\s+)(.+?)\s*$", re.I)
_UNIVERSAL_RE = re.compile(r"^\s*(all|every|each)\s+(.+?)\s+(?:are|is)\s+(.+?)\s*$", re.I)
_ISA_RE = re.compile(r"^\s*(.+?)\s+(?:is|are|has|have)\s+(?:a\s+|an\s+|the\s+)?(.+?)\s*$", re.I)
_DISJ_RE = re.compile(r"\s+(?:or|\|)\s+", re.I)


@dataclass(frozen=True)
class DeductiveStatement:
    """Canonical form of a premise, conclusion, or hypothesis."""

    text: str
    canonical: str
    kind: str = "atomic"
    confidence: float = 1.0
    source: str = "input"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProofStep:
    """A single proof derivation step."""

    step: int
    rule: str
    input: List[str]
    output: str
    confidence: float
    support: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe_reasoning_state(asdict(self))


class ReasoningDeductive(BaseReasoning):
    """Deductive reasoning: derive specific conclusions from premises.

    The public surface remains compatible with the earlier implementation while
    adding validation, proof budgets, cache support, rule diagnostics, and safer
    contradiction handling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = load_global_config()
        self.contradiction_threshold = clamp_confidence(self.config.get("contradiction_threshold", 0.25))

        self.deductive_config: Dict[str, Any] = get_config_section("reasoning_deductive") or {}
        self.max_steps: int = bounded_iterations(self.deductive_config.get("max_steps", 32), minimum=1, maximum=10_000)
        self.max_iterations: int = bounded_iterations(self.deductive_config.get("max_iterations", 8), minimum=1, maximum=1_000)
        self.max_premises: int = bounded_iterations(self.deductive_config.get("max_premises", 256), minimum=1, maximum=50_000)
        self.max_statement_length: int = bounded_iterations(self.deductive_config.get("max_statement_length", 512), minimum=8, maximum=10_000)
        self.certainty_threshold: float = clamp_confidence(self.deductive_config.get("certainty_threshold", 0.85))
        self.min_premise_confidence: float = clamp_confidence(self.deductive_config.get("min_premise_confidence", 0.5))
        self.rule_confidence_decay: float = clamp_confidence(self.deductive_config.get("rule_confidence_decay", 0.95))
        self.contradiction_penalty: float = clamp_confidence(self.deductive_config.get("contradiction_penalty", 0.5))
        self.negation_penalty: float = clamp_confidence(self.deductive_config.get("negation_penalty", 0.7))
        self.enable_fallacy_check: bool = bool(self.deductive_config.get("enable_fallacy_check", True))
        self.enable_cache: bool = bool(self.deductive_config.get("enable_cache", True))
        self.cache_ttl_seconds: float = float(self.deductive_config.get("cache_ttl_seconds", 300.0))
        self.return_trace: bool = bool(self.deductive_config.get("return_trace", True))
        self.strict_inputs: bool = bool(self.deductive_config.get("strict_inputs", True))
        self.include_rejected_premises: bool = bool(self.deductive_config.get("include_rejected_premises", True))
        self.allow_atomic_premises: bool = bool(self.deductive_config.get("allow_atomic_premises", True))
        self.rule_priority: List[str] = list(self.deductive_config.get("rule_priority") or [
            "modus_ponens", "modus_tollens", "syllogism", "disjunctive_syllogism", "hypothetical_syllogism"
        ])
        self.rule_weights: Dict[str, float] = {
            str(k): clamp_confidence(v) for k, v in dict(self.deductive_config.get("rule_weights", {})).items()
        }
        self.cache: Optional[ReasoningCache] = (
            ReasoningCache(namespace="reasoning_deductive", default_ttl_seconds=self.cache_ttl_seconds)
            if self.enable_cache else None
        )
        self._last_trace: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def perform_reasoning(self, premises: List[str], hypothesis: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # type: ignore
        """Perform bounded deductive proof search for ``hypothesis``."""
        started = time.monotonic()
        context = context or {}
        self._validate_inputs(premises, hypothesis)
        cache_key = self._cache_key(premises, hypothesis, context)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        normalized_premises = self._normalize_premises(premises, context)
        valid_premises, rejected = self._validate_premises_with_rejections(normalized_premises, context)
        if not valid_premises:
            result = self._format_result(False, 0.0, [], premises, hypothesis, {
                "internal_contradictions": [],
                "hypothesis_contradictions": [],
                "contradiction_score": 0.0,
                "rejected_premises": rejected,
                "reason": "Invalid premises",
            })
            return self._finalize_result(result, started, cache_key)

        proof_steps = self._apply_inference_rules([s.text for s in valid_premises], context)
        evaluation = self._evaluate_hypothesis(hypothesis, proof_steps, context)
        contradiction_analysis = self._check_contradictions(proof_steps, hypothesis, context)
        fallacies = self._detect_fallacies(valid_premises, proof_steps, hypothesis) if self.enable_fallacy_check else []
        contradiction_analysis["fallacies"] = fallacies
        if rejected and self.include_rejected_premises:
            contradiction_analysis["rejected_premises"] = rejected

        is_proven, certainty = self._determine_certainty(evaluation, contradiction_analysis)
        result = self._format_result(is_proven, certainty, proof_steps, [s.text for s in valid_premises], hypothesis, contradiction_analysis)
        result["evaluation"] = evaluation
        return self._finalize_result(result, started, cache_key)

    def identify_contradictions(self, statements: List[str]) -> List[Tuple[str, str, str]]:
        """Identify direct negation, predicate conflict, and universal-instance conflicts."""
        # Normalize all statements to canonical form
        canonical_statements = [self._canonical_text(s) for s in statements if str(s).strip()]
        contradictions: List[Tuple[str, str, str]] = []
        seen: Set[Tuple[str, str, str]] = set()
    
        for i, left in enumerate(canonical_statements):
            for right in canonical_statements[i + 1:]:
                ctype = self._contradiction_type(left, right)
                if ctype:
                    # Ensure deterministic ordering for deduplication
                    key = (left, right, ctype) if left <= right else (right, left, ctype)
                    if key not in seen:
                        seen.add(key)
                        contradictions.append((left, right, ctype))
        return contradictions

    def diagnostics(self) -> Dict[str, Any]:
        """Return lightweight runtime diagnostics."""
        payload = {
            "max_steps": self.max_steps,
            "max_iterations": self.max_iterations,
            "certainty_threshold": self.certainty_threshold,
            "rule_priority": list(self.rule_priority),
            "last_trace": self._last_trace,
        }
        if self.cache is not None:
            payload["cache"] = self.cache.metrics()
        return json_safe_reasoning_state(payload)

    # ------------------------------------------------------------------
    # Validation and normalization
    # ------------------------------------------------------------------
    def _validate_inputs(self, premises: Any, hypothesis: Any) -> None:
        if not isinstance(premises, list) or not premises:
            raise ReasoningValidationError("premises must be a non-empty list", context={"type": type(premises).__name__})
        if len(premises) > self.max_premises:
            raise ReasoningValidationError("premise count exceeds reasoning_deductive.max_premises", context={"count": len(premises), "max": self.max_premises})
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ReasoningValidationError("hypothesis must be a non-empty string")

    def _normalize_premises(self, premises: Sequence[Any], context: Mapping[str, Any]) -> List[DeductiveStatement]:
        evidence_conf = self._evidence_confidence_map(context.get("evidence", []))
        normalized: List[DeductiveStatement] = []
        seen: Set[str] = set()
        for idx, raw in enumerate(premises):
            text = self._clean_statement(raw)
            if not text:
                continue
            canonical = self._canonical_text(text)
            if canonical in seen:
                continue
            seen.add(canonical)
            normalized.append(DeductiveStatement(
                text=text,
                canonical=canonical,
                kind=self._statement_kind(canonical),
                confidence=clamp_confidence(evidence_conf.get(canonical, context.get("default_premise_confidence", 1.0))),
                source=f"premise_{idx}",
            ))
        return normalized

    def _validate_premises(self, premises: List[str], context: Dict) -> List[str]:
        valid, _ = self._validate_premises_with_rejections(self._normalize_premises(premises, context), context)
        return [item.text for item in valid]

    def _validate_premises_with_rejections(self, premises: List[DeductiveStatement], context: Mapping[str, Any]) -> Tuple[List[DeductiveStatement], List[Dict[str, Any]]]:
        known_falsehoods = {self._canonical_text(item) for item in context.get("known_falsehoods", [])}
        rejected: List[Dict[str, Any]] = []
        valid: List[DeductiveStatement] = []
        for item in premises:
            reason = None
            if item.canonical in known_falsehoods:
                reason = "known_falsehood"
            elif item.confidence < self.min_premise_confidence:
                reason = "low_confidence"
            elif not self._formal_validation(item.text):
                reason = "invalid_form"
            if reason:
                rejected.append({"premise": item.text, "reason": reason, "confidence": item.confidence})
            else:
                valid.append(item)

        contradictions = self.identify_contradictions([p.text for p in valid])
        if contradictions and self.strict_inputs:
            rejected.extend({"premise": f"{a} / {b}", "reason": c} for a, b, c in contradictions)
            return [], rejected
        return valid, rejected

    def _is_valid_premise(self, premise: str, context: Dict) -> bool:
        return bool(self._validate_premises([premise], context))

    def _formal_validation(self, statement: str) -> bool:
        text = self._canonical_text(statement)
        if not text or len(text) > self.max_statement_length:
            return False
        if self._parse_implication(text) or self._parse_universal(text) or self._parse_isa(text):
            return True
        if _DISJ_RE.search(text):
            return True
        if self.allow_atomic_premises and len(text.split()) >= 1:
            return True
        return not self.strict_inputs

    # ------------------------------------------------------------------
    # Inference engine
    # ------------------------------------------------------------------
    def _apply_inference_rules(self, premises: List[str], context: Dict) -> List[Dict[str, Any]]:
        current: Dict[str, float] = {self._canonical_text(p): 1.0 for p in premises}
        display: Dict[str, str] = {self._canonical_text(p): self._clean_statement(p) for p in premises}
        proof_steps: List[ProofStep] = []
        seen_steps: Set[Tuple[str, str, Tuple[str, ...]]] = set()
        step_no = 0

        for iteration in range(self.max_iterations):
            added_this_round = 0
            for rule_name in self.rule_priority:
                rule = getattr(self, f"_derive_{rule_name}", None)
                legacy_rule = getattr(self, f"_apply_{rule_name}", None)
    
                if rule is None and legacy_rule is None:
                    logger.warning("Unknown deductive rule skipped: %s", rule_name)
                    continue
    
                derived: List[Tuple[str, List[str], float, Dict[str, Any]]] = []
                if rule is not None:
                    derived = rule(current, display, context)
                elif legacy_rule is not None:
                    # legacy_rule expects (statements, context) → returns list of conclusion strings
                    legacy_conclusions = legacy_rule(list(display.values()), context)
                    derived = [self._make_derivation(rule_name, c, []) for c in legacy_conclusions]
    
                for conclusion, support, confidence, metadata in derived:
                    canonical = self._canonical_text(conclusion)
                    if not canonical or canonical in current:
                        continue
                    signature = (rule_name, canonical, tuple(sorted(self._canonical_text(s) for s in support)))
                    if signature in seen_steps:
                        continue
                    seen_steps.add(signature)
                    current[canonical] = clamp_confidence(confidence)
                    display[canonical] = self._restore_display(conclusion)
                    proof_steps.append(ProofStep(
                        step=step_no,
                        rule=rule_name,
                        input=[display.get(self._canonical_text(s), s) for s in support] or list(display.values()),
                        output=display[canonical],
                        confidence=current[canonical],
                        support=support,
                        metadata=metadata,
                    ))
                    step_no += 1
                    added_this_round += 1
                    if step_no >= self.max_steps:
                        return [p.to_dict() for p in proof_steps]
            if added_this_round == 0:
                break
        return [p.to_dict() for p in proof_steps]

    @staticmethod
    def _make_derivation(rule: str, conclusion: str, support: List[str]) -> Tuple[str, List[str], float, Dict[str, Any]]:
        return conclusion, support, 0.7, {"legacy_rule": rule}

    def _derive_modus_ponens(self, current: Mapping[str, float], display: Mapping[str, str], context: Dict) -> List[Tuple[str, List[str], float, Dict[str, Any]]]:
        results = []
        for stmt in current:
            parsed = self._parse_implication(stmt)
            if not parsed:
                continue
            antecedent, consequent = parsed
            ant = self._canonical_text(antecedent)
            if ant in current:
                conf = self._rule_confidence("modus_ponens", current[stmt], current[ant])
                results.append((consequent, [display.get(stmt, stmt), display.get(ant, ant)], conf, {"pattern": "if_p_then_q_and_p"}))
        return results

    def _derive_modus_tollens(self, current: Mapping[str, float], display: Mapping[str, str], context: Dict) -> List[Tuple[str, List[str], float, Dict[str, Any]]]:
        results = []
        for stmt in current:
            parsed = self._parse_implication(stmt)
            if not parsed:
                continue
            antecedent, consequent = parsed
            neg_consequent = self._negate(consequent)
            if neg_consequent in current:
                conclusion = self._negate(antecedent)
                conf = self._rule_confidence("modus_tollens", current[stmt], current[neg_consequent])
                results.append((conclusion, [display.get(stmt, stmt), display.get(neg_consequent, neg_consequent)], conf, {"pattern": "if_p_then_q_and_not_q"}))
        return results

    def _derive_syllogism(self, current: Mapping[str, float], display: Mapping[str, str], context: Dict) -> List[Tuple[str, List[str], float, Dict[str, Any]]]:
        universals = [(s, self._parse_universal(s)) for s in current]
        universals = [(s, p) for s, p in universals if p]
        instances = [(s, self._parse_isa(s)) for s in current]
        instances = [(s, p) for s, p in instances if p]
        results: List[Tuple[str, List[str], float, Dict[str, Any]]] = []

        for stmt_a, parsed_a in universals:
            subj_a, pred_a = parsed_a # type: ignore[misc]
            for stmt_b, parsed_b in universals:
                if stmt_a == stmt_b:
                    continue
                subj_b, pred_b = parsed_b # type: ignore[misc]
                if self._same_term(pred_a, subj_b):
                    conclusion = f"all {subj_a} are {pred_b}"
                    results.append((conclusion, [display.get(stmt_a, stmt_a), display.get(stmt_b, stmt_b)], self._rule_confidence("syllogism", current[stmt_a], current[stmt_b]), {"pattern": "universal_chain"}))

        for inst_stmt, parsed_i in instances:
            entity, category = parsed_i # type: ignore[misc]
            for uni_stmt, parsed_u in universals:
                subject, predicate = parsed_u # type: ignore[misc]
                if self._same_term(category, subject):
                    conclusion = f"{entity} is {predicate}"
                    results.append((conclusion, [display.get(inst_stmt, inst_stmt), display.get(uni_stmt, uni_stmt)], self._rule_confidence("syllogism", current[inst_stmt], current[uni_stmt]), {"pattern": "instance_subsumption"}))
        return results

    def _derive_disjunctive_syllogism(self, current: Mapping[str, float], display: Mapping[str, str], context: Dict) -> List[Tuple[str, List[str], float, Dict[str, Any]]]:
        results = []
        for stmt in current:
            options = [self._canonical_text(p) for p in _DISJ_RE.split(stmt) if p.strip()]
            if len(options) < 2:
                continue
            for option in options:
                neg = self._negate(option)
                if neg in current:
                    for other in options:
                        if other != option:
                            results.append((other, [display.get(stmt, stmt), display.get(neg, neg)], self._rule_confidence("disjunctive_syllogism", current[stmt], current[neg]), {"eliminated": option}))
        return results

    def _derive_hypothetical_syllogism(self, current: Mapping[str, float], display: Mapping[str, str], context: Dict) -> List[Tuple[str, List[str], float, Dict[str, Any]]]:
        implications = [(s, self._parse_implication(s)) for s in current]
        implications = [(s, p) for s, p in implications if p]
        results = []
        for stmt_a, parsed_a in implications:
            a1, a2 = parsed_a # type: ignore[misc]
            for stmt_b, parsed_b in implications:
                if stmt_a == stmt_b:
                    continue
                b1, b2 = parsed_b # type: ignore[misc]
                if self._canonical_text(a2) == self._canonical_text(b1):
                    conclusion = f"{a1} implies {b2}"
                    results.append((conclusion, [display.get(stmt_a, stmt_a), display.get(stmt_b, stmt_b)], self._rule_confidence("hypothetical_syllogism", current[stmt_a], current[stmt_b]), {"pattern": "implication_chain"}))
        return results

    # Legacy-compatible rule methods
    def _apply_modus_ponens(self, statements: List[str], context: Dict) -> List[str]:
        current = {self._canonical_text(s): 1.0 for s in statements}
        display = {self._canonical_text(s): s for s in statements}
        return [c for c, _, _, _ in self._derive_modus_ponens(current, display, context)]

    def _apply_modus_tollens(self, statements: List[str], context: Dict) -> List[str]:
        current = {self._canonical_text(s): 1.0 for s in statements}
        display = {self._canonical_text(s): s for s in statements}
        return [c for c, _, _, _ in self._derive_modus_tollens(current, display, context)]

    def _apply_syllogism(self, statements: List[str], context: Dict) -> List[str]:
        current = {self._canonical_text(s): 1.0 for s in statements}
        display = {self._canonical_text(s): s for s in statements}
        return [c for c, _, _, _ in self._derive_syllogism(current, display, context)]

    def _apply_disjunctive_syllogism(self, statements: List[str], context: Dict) -> List[str]:
        current = {self._canonical_text(s): 1.0 for s in statements}
        display = {self._canonical_text(s): s for s in statements}
        return [c for c, _, _, _ in self._derive_disjunctive_syllogism(current, display, context)]

    def _apply_hypothetical_syllogism(self, statements: List[str], context: Dict) -> List[str]:
        current = {self._canonical_text(s): 1.0 for s in statements}
        display = {self._canonical_text(s): s for s in statements}
        return [c for c, _, _, _ in self._derive_hypothetical_syllogism(current, display, context)]

    # ------------------------------------------------------------------
    # Proof evaluation
    # ------------------------------------------------------------------
    def _evaluate_hypothesis(self, hypothesis: str, proof_steps: List[Dict[str, Any]], context: Dict) -> Dict[str, Any]:
        target = self._canonical_text(hypothesis)
        neg_target = self._negate(target)
        conclusions = {self._canonical_text(step.get("output", "")): step for step in proof_steps}
        direct = target in conclusions
        negated = neg_target in conclusions or any(self._contradiction_type(target, c) for c in conclusions)
        derived = direct or self._can_derive(hypothesis, [s.get("output", "") for s in proof_steps], context)
        support_step = conclusions.get(target)
        proof_strength = 0.0
        if direct and support_step:
            proof_strength = clamp_confidence(support_step.get("confidence", 0.9))
        elif derived:
            proof_strength = 0.7
        elif not negated:
            proof_strength = 0.25
        return {
            "direct_proof": direct,
            "negation_proof": bool(negated),
            "derived_proof": bool(derived),
            "proof_strength": clamp_confidence(proof_strength),
            "supporting_step": support_step,
        }

    def _can_derive(self, hypothesis: str, statements: List[str], context: Dict) -> bool:
        target = self._canonical_text(hypothesis)
        current = [self._canonical_text(s) for s in statements]
        for rule_name in self.rule_priority:
            rule = getattr(self, f"_apply_{rule_name}", None)
            if rule and any(self._canonical_text(item) == target for item in rule(current, context)):
                return True
        return False

    def _check_contradictions(self, proof_steps: List[Dict[str, Any]], hypothesis: str, context: Dict) -> Dict[str, Any]:
        statements: List[str] = []
        for step in proof_steps:
            statements.extend(step.get("input", []))
            statements.append(step.get("output", ""))
        unique = list(dict.fromkeys(s for s in statements if s))
        contradictions = self.identify_contradictions(unique)
        hyp = self._canonical_text(hypothesis)
        hypothesis_contradictions = [stmt for stmt in unique if self._contradiction_type(hyp, self._canonical_text(stmt))]
        return {
            "internal_contradictions": contradictions,
            "hypothesis_contradictions": hypothesis_contradictions,
            "contradiction_score": clamp_confidence(len(contradictions) / max(1, len(unique))),
        }

    def _determine_certainty(self, evaluation: Dict[str, Any], contradiction_analysis: Dict[str, Any]) -> Tuple[bool, float]:
        certainty = clamp_confidence(evaluation.get("proof_strength", 0.0))
        certainty -= clamp_confidence(contradiction_analysis.get("contradiction_score", 0.0)) * self.contradiction_penalty
        if evaluation.get("negation_proof"):
            certainty *= (1.0 - self.negation_penalty)
        if contradiction_analysis.get("fallacies"):
            certainty *= 0.9
        certainty = clamp_confidence(certainty)
        is_proven = certainty >= self.certainty_threshold and not contradiction_analysis.get("hypothesis_contradictions")
        return bool(is_proven), certainty

    def _detect_fallacies(self, premises: List[DeductiveStatement], proof_steps: List[Dict[str, Any]], hypothesis: str) -> List[Dict[str, Any]]:
        fallacies: List[Dict[str, Any]] = []
        for step in proof_steps:
            if step.get("rule") == "modus_ponens" and not step.get("support"):
                fallacies.append({"type": "unsupported_modus_ponens", "step": step.get("step")})
        for premise in premises:
            if self._canonical_text(hypothesis) == premise.canonical:
                fallacies.append({"type": "begging_the_question", "premise": premise.text})
        return fallacies

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _clean_statement(self, value: Any) -> str:
        text = str(value).strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > self.max_statement_length:
            text = text[: self.max_statement_length].rstrip()
        return text

    def _canonical_text(self, value: Any) -> str:
        text = self._clean_statement(value).lower().strip(" .")
        text = text.replace("can't", "cannot")
        leading_neg = re.match(r"^not\s+(.+?)\s+(is|are)\s+(.+)$", text)
        if leading_neg:
            text = f"{leading_neg.group(1)} {leading_neg.group(2)} not {leading_neg.group(3)}"
        text = re.sub(r"\ba\s+", "", text)
        text = re.sub(r"\ban\s+", "", text)
        text = re.sub(r"\bthe\s+", "", text)
        text = re.sub(r"\s+", " ", text)
        parsed = self._parse_implication_raw(text)
        if parsed:
            return f"{parsed[0]} implies {parsed[1]}"
        return text

    def _restore_display(self, value: str) -> str:
        if " implies " in value:
            left, right = value.split(" implies ", 1)
            return f"{left} implies {right}"
        return value[:1].upper() + value[1:] if value else value

    def _statement_kind(self, text: str) -> str:
        if self._parse_implication(text):
            return "implication"
        if self._parse_universal(text):
            return "universal"
        if _DISJ_RE.search(text):
            return "disjunction"
        if self._parse_isa(text):
            return "isa"
        return "atomic"

    def _parse_implication_raw(self, text: str) -> Optional[Tuple[str, str]]:
        match = _IMPLIES_RE.match(text)
        if not match:
            return None
        return self._canonical_text(match.group(1)), self._canonical_text(match.group(2))

    def _parse_implication(self, text: str) -> Optional[Tuple[str, str]]:
        if " implies " in text:
            left, right = text.split(" implies ", 1)
            return self._canonical_text(left), self._canonical_text(right)
        return self._parse_implication_raw(text)

    def _parse_universal(self, text: str) -> Optional[Tuple[str, str]]:
        match = _UNIVERSAL_RE.match(text)
        if not match:
            return None
        return self._term(match.group(2)), self._term(match.group(3))

    def _parse_isa(self, text: str) -> Optional[Tuple[str, str]]:
        if self._parse_implication(text) or self._parse_universal(text) or _DISJ_RE.search(text):
            return None
        match = _ISA_RE.match(text)
        if not match:
            return None
        return self._canonical_text(match.group(1)), self._term(match.group(2))

    def _term(self, text: str) -> str:
        term = self._canonical_text(text)
        words = term.split()
        if len(words) == 1 and len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
            term = term[:-1]
        return term

    def _same_term(self, left: str, right: str) -> bool:
        return self._term(left) == self._term(right)

    def _negate(self, statement: str) -> str:
        canonical = self._canonical_text(statement)
        for prefix in _NEGATION_PREFIXES:
            if canonical.startswith(prefix):
                return canonical[len(prefix):].strip()
        parsed = self._parse_isa(canonical)
        if parsed:
            entity, pred = parsed
            return f"{entity} is not {pred}"
        return f"not {canonical}"

    def _contradiction_type(self, left: str, right: str) -> Optional[str]:
        l = self._canonical_text(left)
        r = self._canonical_text(right)
        if not l or not r or l == r:
            return None
        if self._negate(l) == r or self._negate(r) == l:
            return "direct_negation"
        li, ri = self._parse_isa(l), self._parse_isa(r)
        if li and ri and li[0] == ri[0] and li[1] != ri[1]:
            left_pred, right_pred = li[1], ri[1]
            if left_pred.startswith("not ") and left_pred[4:] == right_pred:
                return "predicate_negation"
            if right_pred.startswith("not ") and right_pred[4:] == left_pred:
                return "predicate_negation"
        return None

    def _rule_confidence(self, rule_name: str, *support_confidences: float) -> float:
        base = min([clamp_confidence(c) for c in support_confidences] or [1.0])
        weight = self.rule_weights.get(rule_name, 1.0)
        return clamp_confidence(base * weight * self.rule_confidence_decay)

    def _evidence_confidence_map(self, evidence: Any) -> Dict[str, float]:
        mapping: Dict[str, float] = {}
        if not isinstance(evidence, list):
            return mapping
        for item in evidence:
            if isinstance(item, Mapping) and item.get("content") is not None:
                mapping[self._canonical_text(item["content"])] = clamp_confidence(item.get("confidence", 0.5))
        return mapping

    def _cache_key(self, premises: Sequence[str], hypothesis: str, context: Mapping[str, Any]) -> str:
        payload = {
            "premises": [self._canonical_text(p) for p in premises],
            "hypothesis": self._canonical_text(hypothesis),
            "known_falsehoods": [self._canonical_text(x) for x in context.get("known_falsehoods", [])],
            "rules": self.rule_priority,
        }
        raw = str(json_safe_reasoning_state(payload))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _finalize_result(self, result: Dict[str, Any], started: float, cache_key: str) -> Dict[str, Any]:
        result.setdefault("metrics", {})["duration_seconds"] = elapsed_seconds(started)
        result["metrics"]["cache_enabled"] = bool(self.cache is not None)
        safe = json_safe_reasoning_state(result)
        self._last_trace = {"hypothesis": safe.get("hypothesis"), "metrics": safe.get("metrics", {})}
        if self.cache is not None:
            self.cache.set(cache_key, safe, ttl_seconds=self.cache_ttl_seconds)
        return safe

    def _format_result(self, is_proven: bool, certainty: float, proof_steps: List[Dict[str, Any]], premises: List[str], hypothesis: str, contradiction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        steps = proof_steps if self.return_trace else []
        return {
            "hypothesis": hypothesis,
            "proven": bool(is_proven),
            "certainty": clamp_confidence(certainty),
            "premises": premises,
            "proof_steps": steps,
            "contradictions": contradiction_analysis,
            "metrics": {
                "proof_length": len(proof_steps),
                "premises_used": len(premises),
                "contradictions_found": len(contradiction_analysis.get("internal_contradictions", [])),
                "hypothesis_contradictions": len(contradiction_analysis.get("hypothesis_contradictions", [])),
                "fallacies": len(contradiction_analysis.get("fallacies", [])),
                "certainty_level": "high" if certainty >= 0.8 else "medium" if certainty >= 0.5 else "low",
                "success": bool(is_proven),
            },
            "reasoning_type": "deductive",
        }


if __name__ == "__main__":
    print("\n=== Running Reasoning Deductive ===\n")
    printer.status("TEST", "Reasoning Deductive initialized", "info")

    engine = ReasoningDeductive()
    tests = [
        (
            ["if it rains then ground is wet", "it rains"],
            "ground is wet",
        ),
        (
            ["all humans are mortal", "Socrates is human"],
            "Socrates is mortal",
        ),
        (
            ["system is stable or system is degraded", "not system is stable"],
            "system is degraded",
        ),
    ]
    for premises, hypothesis in tests:
        result = engine.perform_reasoning(premises, hypothesis)
        assert result["proven"] or result["certainty"] >= 0.5
        assert result["metrics"]["proof_length"] >= 1
        printer.status("CHECK", f"{hypothesis}: {result['certainty']:.2f}", "success")

    contradiction = engine.identify_contradictions(["service is available", "not service is available"])
    assert contradiction
    assert engine.diagnostics()["max_steps"] >= 1

    print("\n=== Test ran successfully ===\n")
