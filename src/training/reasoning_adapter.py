"""Thin, fail-closed adapter around ReasoningAgent's public validation API."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

from .enrichment_contracts import (
    CurriculumCompatibilityError,
    CurriculumConfig,
    KnowledgeFact,
    ReasoningValidation,
)


class ReasoningAdapter:
    """Use ReasoningAgent as a validation/consistency gate, not a text generator."""

    def __init__(self, agent: Any, config: CurriculumConfig) -> None:
        self.agent = agent
        self.config = config
        for method in ("load_knowledge", "validate_fact"):
            if not callable(getattr(agent, method, None)):
                raise CurriculumCompatibilityError(
                    f"ReasoningAgent does not expose required method {method}()."
                )
        knowledge_base = getattr(agent, "knowledge_base", None)
        if knowledge_base is None:
            raise CurriculumCompatibilityError(
                "ReasoningAgent does not expose its canonical knowledge_base compatibility view."
            )

    @contextmanager
    def _temporary_knowledge(self, facts: Sequence[KnowledgeFact]) -> Iterator[None]:
        previous = dict(getattr(self.agent, "knowledge_base", {}) or {})
        replacement = {fact.tuple: float(fact.confidence) for fact in facts}
        self.agent.load_knowledge(replacement)
        try:
            yield
        finally:
            self.agent.load_knowledge(previous)

    def validate_fact_set(
        self,
        facts: Sequence[KnowledgeFact],
    ) -> Dict[Tuple[str, str, str], ReasoningValidation]:
        if not facts:
            return {}
        output: Dict[Tuple[str, str, str], ReasoningValidation] = {}
        try:
            with self._temporary_knowledge(facts):
                for fact in facts:
                    validation = self._validate_active_fact(fact)
                    output[fact.tuple] = validation
        except Exception:
            if self.config.fail_on_agent_error:
                raise
        return output

    def classify_candidate(
        self,
        candidate: KnowledgeFact,
        support_facts: Sequence[KnowledgeFact],
    ) -> ReasoningValidation:
        """Classify a candidate against a bounded support KB.

        ``valid`` means supported by the supplied structured knowledge and fully
        validated. ``invalid`` is treated by the curriculum as *unsupported by
        the supplied facts*, never as a universal contradiction claim.
        """

        try:
            with self._temporary_knowledge(support_facts):
                return self._validate_active_fact(candidate)
        except Exception:
            if self.config.fail_on_agent_error:
                raise
            return ReasoningValidation(
                fact=candidate,
                decision="indeterminate",
                combined_valid=False,
                validation_status="failed",
                validation_complete=False,
                has_conflict=False,
                kb_confidence=None,
                probabilistic_confidence=None,
                details={"error": "reasoning_agent_failure"},
            )

    def infer_validated_facts(
        self,
        support_facts: Sequence[KnowledgeFact],
        *,
        max_iterations: int = 3,
    ) -> Tuple[Tuple[KnowledgeFact, ReasoningValidation], ...]:
        """Run configured public forward chaining and keep only fully validated additions.

        No reasoning trace is promoted to a target. Only the structured premises
        and the validated inferred fact are exposed to LANTRA. If the active
        ReasoningAgent has no applicable rules, this method simply returns no
        curriculum examples.
        """

        forward = getattr(self.agent, "forward_chaining_report", None)
        if not callable(forward) or not support_facts:
            return ()
        accepted = []
        try:
            with self._temporary_knowledge(support_facts):
                report = forward(max_iterations=max(1, int(max_iterations)))
                additions = getattr(report, "added", None)
                if additions is None and isinstance(report, Mapping):
                    additions = report.get("added", {})
                if not isinstance(additions, Mapping):
                    return ()
                for raw_fact, raw_confidence in additions.items():
                    if not isinstance(raw_fact, Sequence) or isinstance(raw_fact, (str, bytes)) or len(raw_fact) != 3:
                        continue
                    try:
                        fact = KnowledgeFact(
                            str(raw_fact[0]),
                            str(raw_fact[1]),
                            str(raw_fact[2]),
                            confidence=float(raw_confidence),
                            source="reasoning_forward_chaining",
                        )
                    except Exception:
                        continue
                    validation = self._validate_active_fact(fact)
                    if validation.accepted_gold:
                        accepted.append((fact, validation))
        except Exception:
            if self.config.fail_on_agent_error:
                raise
            return ()
        return tuple(sorted(accepted, key=lambda item: item[0].tuple))

    def _validate_active_fact(self, fact: KnowledgeFact) -> ReasoningValidation:
        raw = self.agent.validate_fact(
            fact.tuple,
            threshold=float(self.config.reasoning_threshold),
        )
        if not isinstance(raw, Mapping):
            raise CurriculumCompatibilityError(
                "ReasoningAgent.validate_fact() must return a mapping."
            )
        return ReasoningValidation(
            fact=fact,
            decision=str(raw.get("decision", "indeterminate")).strip().lower(),
            combined_valid=bool(raw.get("combined_valid", False)),
            validation_status=str(raw.get("validation_status", "unknown")).strip().lower(),
            validation_complete=bool(raw.get("validation_complete", False)),
            has_conflict=bool(raw.get("has_conflict", False)),
            kb_confidence=self._optional_float(raw.get("kb_confidence")),
            probabilistic_confidence=self._optional_float(raw.get("probabilistic_confidence")),
            details=dict(raw),
        )

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


__all__ = ["ReasoningAdapter"]
