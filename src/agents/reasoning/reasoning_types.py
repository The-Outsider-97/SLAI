"""
Factory/policy layer for single and combined reasoning strategies.

Provides:
- Dynamic discovery and creation of reasoning strategies.
- Instance caching (optional) to avoid re‑instantiating common strategies.
- Combined reasoning with sequential execution and context passing.
- Keyword‑based reasoning strategy suggestion.
- Full integration with ReasoningMemory and ReasoningCache.
"""
from __future__ import annotations

import copy
import inspect
import re
import threading

from collections import ChainMap
from typing import Any, Dict, List, Mapping, Optional, Type, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.reasoning_errors import *
from .utils.reasoning_helpers import *
from .types import *
from .reasoning_memory import ReasoningMemory
from .strategy_selector import ReasoningStrategySelector, problem_text
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Reasoning Types")
printer = PrettyPrinter()

class ReasoningTypes:
    """
    Factory and policy layer for reasoning strategies.

    Features:
    - Create single reasoning strategies (e.g., "abduction").
    - Create combined strategies using "+" syntax (e.g., "abduction+induction+deduction").
    - Optional caching of created instances to reduce overhead.
    - Keyword‑based suggestion of suitable reasoning strategy for a problem.
    - Shared memory and cache for all reasoning instances.
    """

    # Core mapping – can be extended by discovery or configuration
    _TASK_TYPES: Dict[str, Type[BaseReasoning]] = {
        "abduction": ReasoningAbduction,
        "deduction": ReasoningDeductive,
        "induction": ReasoningInductive,
        "analogical": ReasoningAnalogical,
        "decompositional": ReasoningDecompositional,
        "cause_effect": ReasoningCauseAndEffect,
    }

    # Alias -> canonical strategy name. Aliases only RESOLVE to a canonical key;
    # they are never registry keys, so the registry (and the learned selector's
    # frozen label space) stays exactly the six canonical names.
    _ALIASES: Dict[str, str] = {
        "abductive": "abduction",
        "abductive_reasoning": "abduction",
        "deductive": "deduction",
        "deductive_reasoning": "deduction",
        "inductive": "induction",
        "inductive_reasoning": "induction",
        "analogy": "analogical",
        "analogic": "analogical",
        "analogical_reasoning": "analogical",
        "decomposition": "decompositional",
        "decompose": "decompositional",
        "decompositional_reasoning": "decompositional",
        "causal": "cause_effect",
        "causality": "cause_effect",
        "cause_and_effect": "cause_effect",
        "causeandeffect": "cause_effect",
        "cause_effect_reasoning": "cause_effect",
    }

    #: Outcome vocabulary of the normalized result contract.
    OUTCOMES = (
        "supported", "unsupported", "disproven", "indeterminate",
        "partial", "degraded", "failed", "completed",
    )

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        default_strategy: str = "deduction",
        memory: Optional[ReasoningMemory] = None,
    ) -> None:
        # Instance-scoped registry: register()/discovery write to this instance's own
        # layer and never leak into other ReasoningTypes instances or the class dict,
        # which remains the (live, read-through) default layer.
        self._TASK_TYPES = ChainMap({}, type(self)._TASK_TYPES)  # type: ignore[assignment]
        self.config: Dict[str, Any] = dict(config or load_global_config())
        self.types_cfg: Dict[str, Any] = dict(get_config_section("reasoning_types", self.config, default={}) or {})
        self.max_combined_types = bounded_iterations(self.types_cfg.get("max_combined_types", 3), minimum=1, maximum=10)
        # Untrusted problem/context nesting beyond this depth is rejected with a typed error
        # instead of surfacing as a RecursionError from deep inside a strategy.
        self.max_input_depth = bounded_iterations(self.types_cfg.get("max_input_depth", 64), minimum=1, maximum=10_000)
        # Phase 0 established that strategies contain mutable run state.
        # Therefore strategy instances remain request/session scoped.
        self.enable_instance_cache = False
        self._instance_cache = None
        self.default_strategy = (str(default_strategy).strip() or "deduction")
        self.strategy_keywords = dict(
            self.types_cfg.get(
                "strategy_keywords",
                {
                    "abduction": ["explain", "why", "hypothesis", "plausible", "most likely"],
                    "deduction": ["prove", "derive", "therefore", "premise", "must be"],
                    "induction": ["pattern", "trend", "generalize", "predict", "extrapolate"],
                    "analogical": ["analogy", "similar", "compare", "resembles"],
                    "decompositional": ["break down", "component", "decompose", "subsystem"],
                    "cause_effect": ["effect", "impact", "results in", "causal", "cause"],
                },
            )
        )

        # Optional learned selector (train_rss.py). Off unless reasoning_types.learned_selector.enabled.
        self._learned_selector: Optional[ReasoningStrategySelector] = None
        self._learned_selector_status = "disabled"
        selector_cfg = dict(self.types_cfg.get("learned_selector") or {})
        if bool(selector_cfg.get("enabled", False)):
            self._load_learned_selector(selector_cfg)

        # Borrowed dependency; ReasoningTypes does not own its lifetime.
        self.reasoning_memory = memory
        self._lock = threading.RLock()
        self._stats: Dict[str, int] = {"single": 0, "combined": 0, "executions": 0, "step_failures": 0}

        logger.info(
            "ReasoningTypes initialized | "
            "mutable_instance_cache=False | "
            "max_combined=%s | default=%s",
            self.max_combined_types,
            self.default_strategy,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create(self, task_type: str) -> BaseReasoning:
        """Create a fresh reasoning strategy instance."""
        normalized = self.resolve_strategy_name(task_type)

        if not normalized:
            raise ReasoningTypeError("reasoning type is required", context={"task_type": task_type},)

        if "+" in normalized:
            instance = self._create_combined_reasoning(normalized)

            with self._lock:
                self._stats["combined"] += 1

            return instance

        instance = self._create_single_reasoning(normalized)

        with self._lock:
            self._stats["single"] += 1

        return instance

    def execute(self, task_type: str, problem: Any, context: Optional[Mapping[str, Any]] = None) -> Any:
        """Execute one or more reasoning strategies through one stable contract.

        ``problem`` is the top-level Agent input. ``context`` may provide the
        strategy-specific structured fields required by a concrete reasoner.
        """
        normalized = self.resolve_strategy_name(task_type)

        if not normalized:
            raise ReasoningTypeError(
                "reasoning type is required",
                context={
                    "task_type": task_type,
                },
            )

        context_dict = dict(context or {})
        self._check_input_depth(problem, context_dict)
        with self._lock:
            self._stats["executions"] += 1

        if "+" in normalized:
            names = self._parse_combined_types(normalized)
            return self._execute_combined(names, problem, context_dict)

        component = self._create_single_reasoning(normalized)
        return self._invoke_component(normalized, component, problem, context_dict)


    def _check_input_depth(self, problem: Any, context: Mapping[str, Any]) -> None:
        for label, value in (("problem", problem), ("context", context)):
            depth = nesting_depth(value, limit=self.max_input_depth)
            if depth > self.max_input_depth:
                raise ReasoningValidationError(
                    f"{label} nesting exceeds reasoning_types.max_input_depth",
                    context={"input": label, "depth": depth, "max_input_depth": self.max_input_depth},
                )

    def invoke_instance(
        self,
        reasoning_engine: BaseReasoning,
        problem: Any,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Invoke an already-created strategy without signature reflection.

        Kept primarily for compatibility with existing ReasoningAgent internals
        and external callers that still create strategies explicitly.
        """
        context_dict = dict(context or {})
        self._check_input_depth(problem, context_dict)
        component_names = getattr(reasoning_engine, "component_names", None)
        if component_names:
            return self._execute_combined(list(component_names), problem, context_dict)

        for strategy_name, strategy_cls in self._TASK_TYPES.items():
            if isinstance(reasoning_engine, strategy_cls):
                return self._invoke_component(
                    strategy_name,
                    reasoning_engine,
                    problem,
                    context_dict,
                )

        # Registered extension strategies are required to respect BaseReasoning's
        # declared input_data/context contract.
        performer = getattr(reasoning_engine, "perform_reasoning", None)
        if not callable(performer):
            raise ReasoningTypeError(
                "Reasoning engine does not expose perform_reasoning",
                context={
                    "engine":
                        type(reasoning_engine).__name__,
                },
            )

        return performer(problem, context_dict)

    def select_reasoning_strategy(self, problem: str, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Choose reasoning strategies for a problem.

        With ``reasoning_types.learned_selector.enabled`` and a loadable checkpoint the
        learned ReasoningStrategySelector decides (``method`` is ``learned_selector`` or
        ``learned_selector_fallback``). It is restricted to strategies registered here, capped
        at ``max_combined_types``, and defers to the keyword policy when it is unsure and the
        keyword policy has evidence. Otherwise, and on any selector error, the deterministic
        keyword policy decides exactly as before. The result keeps the ``strategy`` /
        ``method`` / ``matches`` contract ReasoningAgent consumes.
        """
        selector = self._learned_selector
        if selector is not None:
            try:
                return selector.select(
                    problem,
                    context,
                    allowed=self._TASK_TYPES,
                    max_strategies=self.max_combined_types,
                    fallback=self._select_by_keywords,
                )
            except Exception as exc:  # selection must never take reasoning down
                logger.warning("Learned selector failed (%s: %s); using keyword policy", type(exc).__name__, exc)
        return self._select_by_keywords(problem)

    def _load_learned_selector(self, cfg: Mapping[str, Any]) -> None:
        """Load the configured checkpoint. Failure is logged and reported by get_stats(), never fatal."""
        try:
            model = ReasoningStrategySelector.try_load(
                cfg.get("model_path") or None,
                allow_experimental=bool(cfg.get("allow_experimental", False)),
            )
        except Exception as exc:  # unapproved, corrupt or incompatible checkpoint
            self._learned_selector_status = f"error: {type(exc).__name__}: {exc}"
            logger.error("Learned reasoning selector disabled, keyword policy in use: %s", exc)
            return
        if model is None:
            self._learned_selector_status = "missing_checkpoint"
            logger.warning("Learned reasoning selector enabled but no checkpoint found; keyword policy in use")
            return
        # The checkpoint's frozen label space must be dispatchable by this registry.
        # (Never "fix" a mismatch by inventing outputs: refuse the selector instead.)
        from .strategy_selector import STRATEGIES as _SELECTOR_LABELS
        missing = [name for name in _SELECTOR_LABELS if name not in self._TASK_TYPES]
        if missing:
            self._learned_selector_status = f"incompatible_registry: missing {missing}"
            logger.error("Learned reasoning selector disabled: strategies not registered: %s", missing)
            return
        self._learned_selector = model
        self._learned_selector_status = "active"
        logger.info("Learned reasoning selector active | version=%s promotion=%s", model.version, model.promotion)

    def _select_by_keywords(self, problem: str) -> Dict[str, Any]:
        """Return the existing keyword-policy decision plus observable evidence.

        This remains a deterministic heuristic.  It deliberately does not claim
        to be a learned selector; the match evidence is exposed so Evaluation /
        Tuning can assess the policy empirically later without coupling the hot
        runtime path to those systems.
        """
        # Structured problems are matched on their text fields only; str(dict) would also match
        # key names and numbers ("causes" as a key is not the user asking about causes).
        text = (problem if isinstance(problem, str) else problem_text(problem)).lower()
        matches: Dict[str, List[str]] = {}
        order: Dict[str, int] = {}

        for position, (strategy, keywords) in enumerate(self.strategy_keywords.items()):
            canonical = self.resolve_strategy_name(strategy)
            if canonical not in self._TASK_TYPES:
                continue  # never route to a strategy the registry cannot dispatch
            matched_keywords = [
                str(keyword)
                for keyword in (keywords or [])
                if self._keyword_in_text(keyword, text)
            ]
            if matched_keywords:
                matches[canonical] = matches.get(canonical, []) + matched_keywords
                order.setdefault(canonical, position)

        if not matches:
            return {
                "strategy": self.default_strategy,
                "method": "keyword_policy_default",
                "matches": {},
            }

        # Deterministic: strongest evidence (most distinct keywords) first, ties by
        # configuration order; the chosen set is then emitted in sorted order, which
        # is identical to the v2.3 policy whenever the cap is not hit.
        ranked = sorted(matches, key=lambda name: (-len(matches[name]), order[name], name))
        names = sorted(ranked[: self.max_combined_types])

        return {
            "strategy": "+".join(names),
            "method": "keyword_policy",
            "matches": {name: matches[name] for name in names},
        }

    @staticmethod
    def _keyword_in_text(keyword: Any, text: str) -> bool:
        """Whole-word(-stem) keyword match.

        Plain substring matching produced false positives ("cause" in "because",
        "similar" in "dissimilar", "why" in "somewhere"). A keyword now matches only
        at word boundaries, with a simple inflection suffix allowed ("effect" ->
        "effects", "compare" -> "compared"). Multi-word keywords tolerate any whitespace.
        """
        token = str(keyword or "").strip().lower()
        if not token:
            return False
        body = r"\s+".join(re.escape(part) for part in token.split())
        pattern = r"(?<![a-z0-9_])" + body + r"(?:s|es|ed|d|ing)?(?![a-z0-9_])"
        return re.search(pattern, text) is not None

    def resolve_strategy_name(self, name: Any) -> str:
        """Return the canonical registry name for ``name`` (handles aliases and ``a+b`` chains).

        Unknown names are returned normalized but unchanged so the caller's own
        validation can report them.
        """
        text = str(name or "").strip().lower()
        if not text:
            return ""

        def one(part: str) -> str:
            key = re.sub(r"[\s\-]+", "_", part.strip())
            return self._ALIASES.get(key, key)

        if "+" in text:
            return "+".join(one(part) for part in text.split("+") if part.strip())
        return one(text)

    def determine_reasoning_strategy(self, problem: str) -> str:
        """Backward-compatible strategy string API."""
        return str(self.select_reasoning_strategy(problem)["strategy"])

    def reason(self, task_type: str, problem: Any, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Execute and normalize one reasoning request into the Phase-3 contract.

        ``execute`` intentionally remains unchanged and returns the concrete
        strategy's native result.  ``reason`` is the canonical semantic boundary
        used by ReasoningAgent.
        """
        normalized = self.resolve_strategy_name(task_type)
        if not normalized:
            raise ReasoningTypeError(
                "reasoning type is required",
                context={"task_type": task_type},
            )

        context_dict = dict(context or {})
        raw = self.execute(normalized, problem, context_dict)
        return self.normalize_result(
            normalized,
            raw,
            problem=problem,
            context=context_dict,
        )

    def normalize_result(
        self,
        task_type: str,
        raw_result: Any,
        *,
        problem: Any = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Normalize native strategy output without erasing strategy detail."""
        strategy = self.resolve_strategy_name(task_type)
        context_dict = dict(context or {})

        if "+" in strategy:
            return self._normalize_combined_result(
                strategy,
                raw_result,
                problem=problem,
                context=context_dict,
            )

        raw: Dict[str, Any] = (
            dict(raw_result)
            if isinstance(raw_result, Mapping)
            else {"value": raw_result}
        )

        conclusion = self._extract_conclusion(strategy, raw)
        confidence = self._extract_confidence(strategy, raw)
        evidence = self._extract_evidence(strategy, raw, context_dict)
        assumptions = self._extract_assumptions(raw, context_dict)
        contradictions = self._extract_contradictions(raw)
        validation = self._extract_validation(strategy, raw)
        steps = self._extract_steps(strategy, raw)
        outcome = self._infer_outcome(strategy, raw, conclusion)
        fallback = self._extract_fallback(strategy, raw)

        validation_status = str(
            validation.get(
                "validation_status",
                validation.get("status", ""),
            )
        ).strip().lower() if validation else ""
        degraded = bool(raw.get("degraded", False)) or validation_status in {
            "failed",
            "partial",
            "unavailable",
        }

        result = {
            "schema": "slai.reasoning.result.v1",
            "strategy": strategy,
            "status": "success",
            "outcome": outcome,
            "conclusion": conclusion,
            "confidence": confidence,
            "evidence": evidence,
            "assumptions": assumptions,
            "contradictions": contradictions,
            "validation": validation,
            "errors": [],
            "steps": steps,
            "step_count": len(steps),
            "stop_reason": self._infer_stop_reason(
                strategy,
                raw,
                outcome,
                fallback,
            ),
            "fallback": fallback,
            "degraded": degraded,
            "provenance": {
                "strategy": strategy,
                "input_type": type(problem).__name__,
                "explicit_context_keys": sorted(
                    str(key)
                    for key in context_dict
                    if not str(key).startswith("_")
                ),
                "evidence_count": len(evidence),
                "assumption_count": len(assumptions),
                "native_reasoning_type": raw.get("reasoning_type"),
            },
            # Never discard the concrete reasoner's richer native report.
            "output": raw,
        }
        return json_safe_reasoning_state(result)

    def _normalize_combined_result(
        self,
        strategy: str,
        raw_result: Any,
        *,
        problem: Any,
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        raw = dict(raw_result) if isinstance(raw_result, Mapping) else {}
        names = self._parse_combined_types(strategy)
        native_steps = raw.get("combined_result", {})
        raw_step_errors = raw.get("step_errors")
        step_errors: Mapping[str, Any] = raw_step_errors if isinstance(raw_step_errors, Mapping) else {}
        component_results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for index, name in enumerate(names, start=1):
            step_key = f"step_{index}_{name}"
            if step_key in step_errors:
                errors.append(dict(step_errors[step_key], step=index))
                continue
            native = None
            if isinstance(native_steps, Mapping):
                # Current v2.3 combined execution keys are intentionally not
                # assumed to be the only possible representation.
                for key in (
                    step_key,
                    f"step_{index}",
                    name,
                ):
                    if key in native_steps:
                        native = native_steps[key]
                        break
            if native is None and index == len(names) and not step_errors:
                native = raw.get("final_output")
            if native is None:
                continue
            component_results.append(
                self.normalize_result(
                    name,
                    native,
                    problem=problem,
                    context=context,
                )
            )

        final = component_results[-1] if component_results else None
        if final is None:
            final_native = raw.get("final_output")
            final = {
                "conclusion": final_native,
                "confidence": {
                    "value": None,
                    "type": "unavailable",
                    "calibrated": False,
                    "calibration_method": None,
                    "source": None,
                },
                "evidence": [],
                "assumptions": self._extract_assumptions(raw, context),
                "contradictions": [],
                "validation": {},
                "outcome": "failed" if errors else "completed",
                "fallback": {"used": False, "reason": None, "source": None},
                "degraded": False,
            }

        step_outcomes = [str(item.get("outcome", "completed")) for item in component_results]
        outcome = self._aggregate_combined_outcome(step_outcomes, has_errors=bool(errors))
        contradictions: List[Any] = []
        seen_contradictions = set()
        for item in component_results:
            for entry in item.get("contradictions", []) or []:
                marker = repr(entry)
                if marker not in seen_contradictions:
                    seen_contradictions.add(marker)
                    contradictions.append(entry)

        degraded = bool(errors) or any(bool(item.get("degraded", False)) for item in component_results)
        status = "success"
        if errors:
            status = "partial" if component_results else "failed"

        return json_safe_reasoning_state(
            {
                "schema": "slai.reasoning.result.v1",
                "strategy": strategy,
                "status": status,
                "outcome": outcome,
                "conclusion": final.get("conclusion"),
                "confidence": final.get("confidence"),
                "evidence": final.get("evidence", []),
                "assumptions": final.get("assumptions", []),
                "contradictions": contradictions,
                "validation": final.get("validation", {}),
                "errors": errors,
                "steps": component_results,
                "step_count": len(component_results),
                "stop_reason": (
                    "combined_sequence_partial" if errors else "combined_sequence_complete"
                ),
                "fallback": final.get(
                    "fallback",
                    {"used": False, "reason": None, "source": None},
                ),
                "degraded": degraded,
                "provenance": {
                    "strategy": strategy,
                    "component_strategies": names,
                    "step_outcomes": step_outcomes,
                    "input_type": type(problem).__name__,
                    "explicit_context_keys": sorted(
                        str(key)
                        for key in context
                        if not str(key).startswith("_")
                    ),
                },
                "output": raw,
            }
        )

    @staticmethod
    def _aggregate_combined_outcome(step_outcomes: List[str], *, has_errors: bool) -> str:
        """Combine per-step outcomes without letting the last step hide the others.

        * No completed step -> ``failed``.
        * ``supported`` and ``disproven`` both present -> ``indeterminate`` (the steps
          disagree; neither verdict is safe to report).
        * Any failed step -> ``partial``.
        * Otherwise the last step's verdict stands, except that a final ``supported``
          is downgraded to ``partial`` when an earlier step was unsupported/indeterminate.
        """
        if not step_outcomes:
            return "failed"
        if "supported" in step_outcomes and "disproven" in step_outcomes:
            return "indeterminate"
        if has_errors:
            return "partial"
        last = step_outcomes[-1]
        if last == "supported" and any(
            item in {"unsupported", "indeterminate", "partial", "failed", "disproven"}
            for item in step_outcomes[:-1]
        ):
            return "partial"
        return last

    @staticmethod
    def _as_mapping(value: Any) -> Dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _valid_relationships(raw: Mapping[str, Any]) -> List[Any]:
        """Causal relationships a cause_effect run accepted.

        ReasoningCauseAndEffect reports them as ``valid_relationships``; the
        normalizer previously read ``validated_relationships`` (never produced), so
        every successful causal run was reported as indeterminate with no confidence.
        Both spellings are accepted.
        """
        for key in ("valid_relationships", "validated_relationships"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
        return []

    @staticmethod
    def _safe_confidence(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return clamp_confidence(value)
        except ReasoningError:
            return None
        except (TypeError, ValueError):
            return None

    def _extract_confidence(self, strategy: str, raw: Mapping[str, Any]) -> Dict[str, Any]:
        value: Optional[float] = None
        semantic_type = "heuristic_confidence"
        source: Optional[str] = None

        if strategy == "deduction":
            value = self._safe_confidence(raw.get("certainty"))
            semantic_type = "deductive_certainty"
            source = "certainty"

        elif strategy == "abduction":
            best = self._as_mapping(raw.get("best_explanation"))
            value = self._safe_confidence(
                best.get("composite_score", best.get("confidence"))
            )
            semantic_type = "abductive_hypothesis_score"
            source = (
                "best_explanation.composite_score"
                if best.get("composite_score") is not None
                else "best_explanation.confidence"
            )

        elif strategy == "induction":
            metrics = self._as_mapping(raw.get("metrics"))
            value = self._safe_confidence(metrics.get("theory_confidence"))
            semantic_type = "inductive_support"
            source = "metrics.theory_confidence"

        elif strategy == "analogical":
            best = self._as_mapping(raw.get("best_transfer"))
            value = self._safe_confidence(
                best.get("transfer_score", best.get("confidence"))
            )
            semantic_type = "analogical_transfer_score"
            source = (
                "best_transfer.transfer_score"
                if best.get("transfer_score") is not None
                else "best_transfer.confidence"
            )

        elif strategy == "cause_effect":
            relationships = self._valid_relationships(raw)
            scores: List[float] = []
            if isinstance(relationships, list):
                for item in relationships:
                    if not isinstance(item, Mapping):
                        continue
                    candidate = self._safe_confidence(item.get("confidence", item.get("causal_score", item.get("score"))))
                    if candidate is not None:
                        scores.append(candidate)
            if scores:
                value = sum(scores) / len(scores)
                source = "valid_relationships.mean_confidence"
            semantic_type = "heuristic_causal_support"

        elif strategy == "decompositional":
            metrics = self._as_mapping(raw.get("metrics"))
            understanding = self._as_mapping(raw.get("system_understanding"))
            value = self._safe_confidence(metrics.get("confidence", understanding.get("confidence")))
            semantic_type = "decomposition_quality_score"
            source = (
                "metrics.confidence"
                if metrics.get("confidence") is not None
                else "system_understanding.confidence"
            )
            # ReasoningDecompositional emits no confidence value; report "unavailable"
            # rather than inventing one.

        if value is None:
            for key in ("confidence", "certainty", "score", "probability"):
                value = self._safe_confidence(raw.get(key))
                if value is not None:
                    source = key
                    break

        if value is None:
            source = None

        # Phase 4 safety: a bounded score is NOT empirically calibrated merely
        # because it lies in [0, 1].
        return {
            "value": value,
            "type": semantic_type if value is not None else "unavailable",
            "calibrated": False,
            "calibration_method": None,
            "source": source,
        }

    def _extract_conclusion(self, strategy: str, raw: Mapping[str, Any]) -> Any:
        if strategy == "deduction":
            return raw.get("hypothesis")
        if strategy == "abduction":
            return self._as_mapping(raw.get("best_explanation")).get("hypothesis")
        if strategy == "induction":
            theory = self._as_mapping(raw.get("theory"))
            return theory.get("theory", theory or None)
        if strategy == "analogical":
            return raw.get("best_transfer")
        if strategy == "cause_effect":
            return raw.get("predictions") or raw.get("causal_model")
        if strategy == "decompositional":
            return raw.get("system_understanding") or raw.get("decomposition_tree")
        for key in ("conclusion", "result", "final_output", "answer", "value"):
            if key in raw:
                return raw.get(key)
        return None

    def _extract_evidence(self, strategy: str, raw: Mapping[str, Any], context: Mapping[str, Any]) -> List[Any]:
        evidence: List[Any] = []

        for key in ("evidence", "evidence_sources"):
            value = context.get(key)
            if isinstance(value, list):
                evidence.extend(value)

        if strategy == "abduction" and isinstance(raw.get("evidence_used"), list):
            evidence.extend(raw["evidence_used"])
        elif strategy == "deduction" and isinstance(raw.get("premises"), list):
            evidence.extend(raw["premises"])
        elif strategy == "induction":
            supporting = self._as_mapping(raw.get("supporting_data"))
            observations = supporting.get("observations_used")
            if isinstance(observations, list):
                evidence.extend(observations)
        elif strategy == "analogical":
            analogies = raw.get("alternative_analogies")
            if isinstance(analogies, list):
                evidence.extend(analogies)
        elif strategy == "cause_effect":
            evidence.extend(self._valid_relationships(raw))

        # Preserve order while deduplicating JSON-safe representations.
        #
        # json_safe_reasoning_state() intentionally accepts a Mapping as its
        # root payload. Evidence items, however, may legitimately be scalar
        # values such as strings (e.g. deductive premises), tuples, sets, or
        # mappings. Wrap each item in a temporary mapping before serialization
        # so the helper's contract is respected.
        unique: List[Any] = []
        seen = set()

        for item in evidence:
            safe_item = json_safe_reasoning_state({"value": item})["value"]
            marker = repr(safe_item)

            if marker in seen:
                continue

            seen.add(marker)
            unique.append(item)

        return unique

    def _extract_assumptions(self, raw: Mapping[str, Any], context: Mapping[str, Any]) -> List[Any]:
        # Only explicit assumptions are reported.  The normalizer never invents
        # hidden premises on behalf of a concrete strategy.
        assumptions: List[Any] = []
        for source in (context.get("assumptions"), raw.get("assumptions")):
            if source is None:
                continue
            if isinstance(source, list):
                assumptions.extend(source)
            else:
                assumptions.append(source)
        return assumptions

    def _extract_contradictions(self, raw: Mapping[str, Any]) -> List[Any]:
        source = raw.get("contradictions", [])
        contradictions: List[Any] = []
        if isinstance(source, list):
            contradictions.extend(source)
        elif isinstance(source, Mapping):
            for key in (
                "internal_contradictions",
                "hypothesis_contradictions",
                "conflicts",
            ):
                value = source.get(key)
                if isinstance(value, list):
                    contradictions.extend(value)

        # Abduction reports contradictions per hypothesis; surface those of the
        # selected explanation (previously dropped by the normalizer).
        if not contradictions:
            best = self._as_mapping(raw.get("best_explanation"))
            if isinstance(best.get("contradictions"), list):
                contradictions.extend(best["contradictions"])
        return contradictions

    def _extract_validation(self, strategy: str, raw: Mapping[str, Any]) -> Dict[str, Any]:
        if isinstance(raw.get("validation"), Mapping):
            return dict(raw["validation"])

        if strategy == "abduction":
            best = self._as_mapping(raw.get("best_explanation"))
            if best:
                return {
                    "is_supported": bool(best.get("is_supported", False)),
                    "explanatory_power": best.get("explanatory_power"),
                    "origin": best.get("origin"),
                    "selection_mode": best.get("selection_mode"),
                }

        if strategy == "analogical":
            best = self._as_mapping(raw.get("best_transfer"))
            if isinstance(best.get("validation"), Mapping):
                return dict(best["validation"])

        if strategy == "deduction":
            return {
                "proven": bool(raw.get("proven", False)),
                "certainty": raw.get("certainty"),
            }

        return {}

    def _extract_steps(self, strategy: str, raw: Mapping[str, Any]) -> List[Any]:
        if strategy == "deduction" and isinstance(raw.get("proof_steps"), list):
            return list(raw["proof_steps"])
        for key in ("steps", "trace", "reasoning_steps"):
            value = raw.get(key)
            if isinstance(value, list):
                return list(value)
        return []

    def _infer_outcome(self, strategy: str, raw: Mapping[str, Any], conclusion: Any) -> str:
        """Map a native report to the shared outcome vocabulary (see ``OUTCOMES``).

        supported      the strategy established its conclusion under its own criteria
        disproven      the premises/evidence entail the negation of the conclusion
        unsupported    the strategy ran but produced nothing evidence-grounded
        indeterminate  the available information does not settle the question
        partial        a result exists but a quality gate was relaxed or a step failed
        "Supported" never means calibrated truth; see ``confidence.calibrated``.
        """
        if bool(raw.get("failed", False)):
            return "failed"
        if strategy == "deduction":
            if bool(raw.get("proven", False)):
                return "supported"
            contradictions = self._as_mapping(raw.get("contradictions"))
            if contradictions.get("hypothesis_contradictions"):
                return "disproven"
            return "indeterminate"
        if strategy == "abduction":
            best = self._as_mapping(raw.get("best_explanation"))
            if not best:
                return "unsupported" if raw.get("contradictions") else "indeterminate"
            if str(best.get("origin", "seed")) != "seed":
                # A hypothesis generated from a template that merely restates the
                # observation is not an explanation, however well it "covers" it.
                return "unsupported"
            if str(best.get("selection_mode", "strict")) != "strict":
                return "partial"
            return "supported"
        if strategy == "induction":
            validation = self._as_mapping(raw.get("validation"))
            return "supported" if bool(validation.get("is_valid", False)) else "indeterminate"
        if strategy == "analogical":
            best = self._as_mapping(raw.get("best_transfer"))
            if not best:
                return "indeterminate"
            validation = self._as_mapping(best.get("validation"))
            if validation and not bool(validation.get("valid", False)):
                return "partial"
            return "supported"
        if strategy == "cause_effect":
            relationships = self._valid_relationships(raw)
            if not relationships:
                return "indeterminate"
            # Temporal precedence / co-occurrence alone is association, not causation
            # (post hoc ergo propter hoc): "supported" needs a declared/explicit mechanism or
            # a mechanism score; otherwise the finding is only "partial".
            for item in relationships:
                if not isinstance(item, Mapping):
                    continue
                if item.get("type") in {"explicit", "declared_mechanism"}:
                    return "supported"
                score = self._safe_confidence(item.get("mechanism_score"))
                if score is not None and score >= 0.5:
                    return "supported"
            return "partial"
        if strategy == "decompositional":
            metrics = self._as_mapping(raw.get("metrics"))
            if conclusion is None or metrics.get("success") is False:
                return "indeterminate"
            # A lone root node means nothing was decomposed (e.g. free text): there is no
            # analysis to report, so do not call it "completed".
            if metrics.get("decomposed") is False or int(metrics.get("total_components", 2) or 0) <= 1:
                return "indeterminate"
            return "completed"

        metrics = self._as_mapping(raw.get("metrics"))
        if metrics.get("success") is True:
            return "supported"
        return "completed" if conclusion is not None else "indeterminate"

    def _extract_fallback(self, strategy: str, raw: Mapping[str, Any]) -> Dict[str, Any]:
        existing = raw.get("fallback")
        if isinstance(existing, Mapping):
            return {
                "used": bool(existing.get("used", False)),
                "reason": existing.get("reason"),
                "source": existing.get("source"),
            }

        if strategy == "abduction":
            best = self._as_mapping(raw.get("best_explanation"))
            mode = str(best.get("selection_mode", "strict"))
            if mode != "strict":
                return {
                    "used": True,
                    "reason": mode,
                    "source": "abduction_hypothesis_selection",
                }

        return {"used": False, "reason": None, "source": None}

    @staticmethod
    def _infer_stop_reason(
        strategy: str,
        raw: Mapping[str, Any],
        outcome: str,
        fallback: Mapping[str, Any],
    ) -> str:
        explicit = raw.get("stop_reason")
        if explicit:
            return str(explicit)
        if strategy == "deduction":
            if bool(raw.get("proven", False)):
                return "proof_established"
            return "hypothesis_refuted" if outcome == "disproven" else "proof_not_established"
        if strategy == "abduction":
            if raw.get("best_explanation") is None:
                return "no_hypothesis_accepted"
            if outcome == "unsupported":
                return "no_grounded_hypothesis"
            return "fallback_hypothesis_selected" if fallback.get("used") else "hypothesis_selected"
        if strategy == "cause_effect" and outcome == "partial":
            return "association_only_no_mechanism"
        if strategy == "decompositional" and outcome == "indeterminate":
            return "not_decomposable"
        if outcome in {"indeterminate", "unsupported"}:
            return "insufficient_support"
        return "strategy_complete"

    def get_memory(self) -> Optional[ReasoningMemory]:
        """Return the shared reasoning memory instance."""
        return self.reasoning_memory

    def get_cache(self) -> Optional[Any]:
        """Return the optional instance cache, if enabled."""
        return self._instance_cache
    
    def register(self, name: str, cls: Type[BaseReasoning]) -> None:
        """Register an extension strategy on THIS instance's registry.

        The registry is instance-scoped; registering never changes other
        ``ReasoningTypes`` instances or the class defaults. Built-in canonical names
        cannot be shadowed accidentally (use ``allow`` semantics via a new name).
        """
        normalized = self.resolve_strategy_name(name)
        if (
            not normalized
            or "+" in normalized
            or not inspect.isclass(cls)
            or not issubclass(cls, BaseReasoning)
        ):
            raise ReasoningTypeError(
                "Invalid reasoning type registration",
                context={"name": name, "cls": getattr(cls, "__name__", repr(cls))},
            )
        with self._lock:
            self._TASK_TYPES[normalized] = cls

    def discover_task_types(self) -> None:
        """Register BaseReasoning subclasses exported by the ``types`` package.

        Only classes that are not already registered (by class identity) are added,
        so discovery can no longer create duplicate pseudo-strategies such as
        ``deductive`` or ``causeandeffect`` next to the canonical names.
        """
        from . import types as reasoning_types_module

        with self._lock:
            registered = set(self._TASK_TYPES.values())
            for _, obj in inspect.getmembers(reasoning_types_module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, BaseReasoning)
                    and obj is not BaseReasoning
                    and obj not in registered
                ):
                    name = self.resolve_strategy_name(obj.__name__.replace("Reasoning", ""))
                    if name and name not in self._TASK_TYPES:
                        self._TASK_TYPES[name] = obj
                        registered.add(obj)
                        logger.debug(f"Discovered reasoning type: {name} -> {obj.__name__}")

    # ------------------------------------------------------------------
    # Internal factories
    # ------------------------------------------------------------------
    def _create_single_reasoning(self, normalized: str) -> BaseReasoning:
        """Instantiate a single reasoning strategy."""
        normalized = self.resolve_strategy_name(normalized)
        with self._lock:
            reasoning_cls = self._TASK_TYPES.get(normalized)
            if reasoning_cls is None:
                self.discover_task_types()
                reasoning_cls = self._TASK_TYPES.get(normalized)

        if reasoning_cls is None:
            raise ReasoningTypeError(f"Unknown reasoning type: {normalized}", context={"available": sorted(self._TASK_TYPES.keys())})
        return reasoning_cls()

    def _create_combined_reasoning(self, combined_type: str) -> BaseReasoning:
        """Create a compatibility wrapper for combined strategies."""
        names = self._parse_combined_types(combined_type)
        factory = self

        class CombinedReasoning(BaseReasoning):
            def __init__(self, component_names: List[str]) -> None:
                super().__init__()

                self.component_names = list(component_names)
                self.name = "+".join(self.component_names)

            def perform_reasoning(
                self,
                input_data: Any = None,
                context: Optional[Dict[str, Any]] = None,
                **legacy_kwargs: Any,
            ) -> Dict[str, Any]:
                merged_context = dict(context or {})

                for key, value in legacy_kwargs.items():
                    merged_context.setdefault(key, value)

                problem = input_data

                if problem is None:
                    for key in (
                        "problem",
                        "observations",
                        "events",
                        "system",
                        "target",
                        "hypothesis",
                    ):
                        if key in merged_context:
                            problem = merged_context[key]
                            break

                return factory._execute_combined(self.component_names, problem, merged_context)
        return CombinedReasoning(names)
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            cache_hits = 0
            cache_misses = 0
            if self._instance_cache is not None:
                metrics = self._instance_cache.metrics()
                cache_hits = metrics.get("hits", 0)
                cache_misses = metrics.get("misses", 0)
            return {
                "total_single_creations": self._stats.get("single", 0),
                "total_combined_creations": self._stats.get("combined", 0),
                "total_executions": self._stats.get("executions", 0),
                "combined_step_failures": self._stats.get("step_failures", 0),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "registered_types": list(self._TASK_TYPES.keys()),
                "learned_selector": self._learned_selector_status,
                "learned_selector_version": getattr(self._learned_selector, "version", None),
            }
    
    def shutdown(self, checkpoint_memory: bool = True) -> None:
        """
        Release ReasoningTypes-owned resources.

        ``checkpoint_memory`` remains in the signature for v2.3 caller
        compatibility but ReasoningTypes no longer owns memory persistence.
        """
        if checkpoint_memory:
            logger.debug("Ignoring checkpoint_memory=True: Reasoning memory durability is owned by BaseAgent/checkpointing.")

        logger.info("ReasoningTypes shut down")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        """Convert value to float or None if value is None/empty string."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ReasoningConfigurationError("reasoning_types value must be numeric or None", cause=exc, context={"value": value}) from exc

    # Backward‑compatible private method (used by legacy code)
    def _determine_reasoning_strategy(self, problem: str) -> str:
        return self.determine_reasoning_strategy(problem)

    def _parse_combined_types(self, combined_type: str) -> List[str]:
        names = [
            name
            for name in self.resolve_strategy_name(combined_type).split("+")
            if name
        ]

        if not (
            1
            <= len(names)
            <= self.max_combined_types
        ):
            raise ReasoningTypeError(
                (
                    "Combined reasoning must include between "
                    f"1 and {self.max_combined_types} strategies"
                ),
                context={
                    "requested": len(names),
                    "max": self.max_combined_types,
                },
            )

        unknown = [
            name
            for name in names
            if name not in self._TASK_TYPES
        ]

        if unknown:
            self.discover_task_types()

            unknown = [
                name
                for name in names
                if name not in self._TASK_TYPES
            ]

        if unknown:
            raise ReasoningTypeError(
                "Unknown reasoning type(s) in combined expression",
                context={
                    "unknown": unknown,
                    "available": sorted(
                        self._TASK_TYPES.keys()
                    ),
                },
            )

        return names


    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value is None:
            return []

        if isinstance(value, list):
            return list(value)

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, set):
            return list(value)

        return [value]


    def _invoke_component(
        self,
        strategy_name: str,
        component: BaseReasoning,
        problem: Any,
        context: Mapping[str, Any],
    ) -> Any:
        """Adapt the canonical Reasoning request to one concrete strategy."""
        strategy = str(strategy_name).strip().lower()
        context_dict = dict(context or {})
        problem_payload: Dict[str, Any] = (
            dict(problem)
            if isinstance(problem, Mapping)
            else {}
        )

        if strategy == "abduction":
            observations = context_dict.get("observations", problem_payload.get("observations", problem))
            return getattr(component, "perform_reasoning")(observations=observations, context=context_dict)

        if strategy == "induction":
            observations = context_dict.get("observations", problem_payload.get("observations", problem))
            return getattr(component, "perform_reasoning")(observations=self._as_list(observations), context=context_dict)

        if strategy == "deduction":
            premises = context_dict.get("premises", problem_payload.get("premises"))
            hypothesis = context_dict.get("hypothesis", problem_payload.get("hypothesis"))
            if premises is None:
                if isinstance(problem, (list, tuple, set)):
                    premises = list(problem)
                else:
                    premises = [str(problem)]

            premises = [str(item) for item in self._as_list(premises)]

            if hypothesis is None:
                hypothesis = (
                    problem_payload.get("goal")
                    or problem_payload.get("query")
                    or str(problem)
                )

            hypothesis = str(hypothesis).strip()

            if not hypothesis:
                raise ReasoningValidationError(
                    "Deductive reasoning requires a hypothesis.",
                    context={
                        "strategy": strategy,
                    },
                )

            return getattr(component, "perform_reasoning")(  # type: ignore[call-arg]
                premises=premises,
                hypothesis=hypothesis,
                context=context_dict,
            )

        if strategy == "analogical":
            target = context_dict.get("target", problem_payload.get("target", problem))
            source_domain = context_dict.get("source_domain", problem_payload.get("source_domain", []))
            return getattr(component, "perform_reasoning")(target=target, source_domain=self._as_list(source_domain), context=context_dict)

        if strategy == "cause_effect":
            events = context_dict.get("events", problem_payload.get("events", problem))
            conditions = context_dict.get("conditions", problem_payload.get("conditions", {}))
            return getattr(component, "perform_reasoning")(events=self._as_list(events), conditions=dict(conditions or {}), context=context_dict)

        if strategy == "decompositional":
            system = context_dict.get("system", problem_payload.get("system", problem))
            return getattr(component, "perform_reasoning")(system=system, context=context_dict)

        # Extensions registered outside the built-in strategy set must implement
        # BaseReasoning's canonical input_data/context interface.
        return component.perform_reasoning(problem, context_dict)


    def _execute_combined(self, names: List[str], problem: Any, context: Mapping[str, Any]) -> Dict[str, Any]:
        """Execute combined strategies with fresh per-step instances.

        Each step runs on its own strategy instance. A failing step is recorded in
        ``step_errors`` and the chain continues (set ``combined_fail_fast`` in the
        context to stop at the first failure); only when *every* step fails is an
        ``InferenceExecutionError`` raised. Earlier results are handed to later steps
        as deep copies, so a step can never mutate another step's reported result.

        NOTE: the built-in strategies read their own task fields (premises, events,
        ...) and do not consume ``prev_step_result``; a chain is therefore a set of
        independent analyses over the same input, not a data-flow pipeline.
        """
        working_context = dict(context or {})
        working_context.setdefault("original_input", problem)
        fail_fast = bool(working_context.get("combined_fail_fast", False))

        step_results: Dict[str, Any] = {}
        step_errors: Dict[str, Dict[str, Any]] = {}
        previous_result: Any = None
        previous_name: Optional[str] = None

        for index, strategy_name in enumerate(names, start=1):
            step_key = f"step_{index}_{strategy_name}"
            step_context = dict(working_context)

            if previous_result is not None:
                step_context["prev_step_result"] = previous_result
                step_context[f"prev_{previous_name}_result"] = previous_result

            try:
                component = self._create_single_reasoning(strategy_name)
                result = self._invoke_component(
                    strategy_name,
                    component,
                    problem,
                    step_context,
                )
            except Exception as exc:  # one failing step must not erase the others
                with self._lock:
                    self._stats["step_failures"] += 1
                step_errors[step_key] = {
                    "strategy": strategy_name,
                    "error_type": type(exc).__name__,
                    "code": getattr(exc, "code", None),
                    "message": str(exc)[:500],
                }
                logger.warning("Combined reasoning step %s failed: %s: %s", step_key, type(exc).__name__, exc)
                if fail_fast:
                    break
                continue

            step_results[step_key] = result
            try:
                shared = copy.deepcopy(result)
            except Exception:  # un-copyable native result: hand over a fresh summary instead
                shared = {"unavailable": True}
            working_context[f"step_{index}_result"] = shared
            working_context["prev_step_result"] = shared
            previous_result = shared
            previous_name = strategy_name

        if step_errors and not step_results:
            raise InferenceExecutionError(
                "Every step of the combined reasoning failed",
                context={"reasoning_types": "+".join(names), "step_errors": step_errors},
            )

        return {
            "combined_result": step_results,
            "reasoning_types": "+".join(names),
            "final_output": (
                step_results[list(step_results)[-1]] if step_results else None
            ),
            "step_errors": step_errors,
            "partial": bool(step_errors),
        }


__all__ = ["ReasoningTypes"]


if __name__ == "__main__":
    print("\n=== Running Reasoning Types ===\n")
    printer.status("TEST", "Reasoning Types initialized", "info")

    factory = ReasoningTypes()

    # 1. Single strategy creation
    abduction = factory.create("abduction")
    assert abduction is not None
    printer.status("PASS", "Created single abduction instance", "success")

    # 2. Combined strategy creation
    combined = factory.create("abduction+induction")
    assert combined is not None
    assert "combined_result" in combined.perform_reasoning(observations=["test"]) # type: ignore
    printer.status("PASS", "Created combined reasoning (3 types)", "success")

    # 3. Combined with invalid count
    try:
        factory.create("abduction+induction+deduction+cause_effect")
        assert False, "Should reject > max_combined_types"
    except ReasoningTypeError:
        printer.status("PASS", "Rejected too many combined types", "success")

    # 4. Strategy suggestion
    prob1 = "Explain why the grass is wet"
    assert factory.determine_reasoning_strategy(prob1) == "abduction"
    prob2 = "Prove that Socrates is mortal given premises"
    assert factory.determine_reasoning_strategy(prob2) == "deduction"
    prob3 = "Find patterns in temperature data"
    assert "induction" in factory.determine_reasoning_strategy(prob3)
    printer.status("PASS", "Keyword‑based strategy suggestion works", "success")

    # 5. Instance caching (if enabled)
    if factory.enable_instance_cache:
        first = factory.create("deduction")
        second = factory.create("deduction")
        assert first is second, "Cached instance should be reused"
        printer.status("PASS", "Instance cache working", "success")
    else:
        printer.status("SKIP", "Instance cache disabled", "warning")

    # 6. Memory integration: memory is borrowed (owned by the caller), never created here
    assert factory.get_memory() is None
    memory = ReasoningMemory()
    with_memory = ReasoningTypes(memory=memory)
    assert with_memory.get_memory() is memory
    memory.add({"type": "test", "content": "reasoning_types_test"}, priority=0.8)
    assert memory.size() >= 1
    printer.status("PASS", "Memory integration OK", "success")

    # 7. Combined reasoning context passing
    class DummyReasoning(BaseReasoning):
        def perform_reasoning(self, input_data=None, context=None, **kwargs):
            ctx = context if context is not None else kwargs.get("context") or {}
            prev = ctx.get("prev_step_result", None)
            return {"prev": prev, "step_info": "done"}

    # Register dummy for test (simulate component)
    factory._TASK_TYPES["dummy"] = DummyReasoning
    combo = factory.create("dummy+dummy")
    res = combo.perform_reasoning(input_data=42)
    assert "combined_result" in res
    # Check that second component received prev_step_result from first
    steps = res["combined_result"]
    assert steps["step_2_dummy"]["prev"] == steps["step_1_dummy"]
    printer.status("PASS", "Sequential context passing verified", "success")

    print("\n=== Test ran successfully ===\n")

