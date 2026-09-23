"""
Public runtime facade for SLAI's alignment subsystem.

The facade is the boundary the AlignmentAgent should depend on.  It owns
alignment-private configuration and state construction, including AlignmentMemory,
so the agent does not directly import or manipulate ``alignment_memory.py`` or
``alignment_config.yaml``.

The facade deliberately does not become a second planner, safety engine,
privacy engine, evaluator, or learning system.  It composes existing alignment
modules, exposes focused constraint/value evidence, and delegates persistent
evidence storage to AlignmentMemory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from .alignment_memory import AlignmentMemory
from .assessment import AlignmentAssessment, AlignmentAssessor
from .bias_detection import BiasDetector
from .counterfactual_auditor import CounterfactualAuditor
from .ethical_constraints import EthicalConstraints
from .fairness_evaluator import FairnessEvaluator
from .value_embedding_model import ValueEmbeddingModel
from .utils.alignment_errors import (
    AlignmentMemoryError,
    ConfigurationError,
    ValueEmbeddingError,
    wrap_alignment_exception,
)
from .utils.alignment_helpers import json_safe, normalize_context
from .utils.config_loader import get_config_section
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Alignment Runtime")


@dataclass(frozen=True)
class AlignmentRuntimeHealth:
    """Compact health snapshot for the alignment subsystem."""

    status: str
    components: Dict[str, bool]
    memory_available: bool
    value_model_ready: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "components": dict(self.components),
            "memory_available": self.memory_available,
            "value_model_ready": self.value_model_ready,
        }


class AlignmentSubsystem:
    """
    Encapsulated alignment subsystem facade.

    The facade shares one AlignmentMemory instance across modules for an
    AlignmentAgent runtime.  Existing component instances can be injected for
    tests or specialist deployments; where they expose the current SLAI memory
    attributes, the facade rebinds them to the shared store.
    """

    def __init__(
        self,
        *,
        bias_detector: Optional[BiasDetector] = None,
        fairness_evaluator: Optional[FairnessEvaluator] = None,
        ethical_constraints: Optional[EthicalConstraints] = None,
        value_embedding_model: Optional[ValueEmbeddingModel] = None,
        counterfactual_auditor: Optional[CounterfactualAuditor] = None,
        alignment_memory: Optional[AlignmentMemory] = None,
    ) -> None:
        self._runtime_config = get_config_section("alignment_runtime") or {}
        self._assessment_config = get_config_section("alignment_assessment") or {}
        self._validate_config()

        self._memory = alignment_memory or AlignmentMemory()

        self.bias_detector = bias_detector or BiasDetector(
            alignment_memory=self._memory
        )
        self.fairness_evaluator = fairness_evaluator or FairnessEvaluator()
        self.ethical_constraints = ethical_constraints or EthicalConstraints()
        self.value_embedding_model = value_embedding_model or ValueEmbeddingModel()
        self.counterfactual_auditor = counterfactual_auditor or CounterfactualAuditor(
            alignment_memory=self._memory
        )

        self._bind_shared_memory(self.bias_detector)
        self._bind_shared_memory(self.fairness_evaluator)
        self._bind_shared_memory(self.ethical_constraints)
        self._bind_shared_memory(self.counterfactual_auditor)
        self._bind_shared_memory(self.value_embedding_model)

        self.assessor = AlignmentAssessor(self._assessment_config)

        self.memory_logging_enabled = bool(
            self._runtime_config.get("memory_logging_enabled", True)
        )
        self.strict_memory_integration = bool(
            self._runtime_config.get("strict_memory_integration", False)
        )
        self.value_model_requires_ready_marker = bool(
            self._runtime_config.get("value_model_requires_ready_marker", True)
        )
        self.value_model_ready_attribute = str(
            self._runtime_config.get(
                "value_model_ready_attribute",
                "alignment_evidence_ready",
            )
        ).strip() or "alignment_evidence_ready"
        self._explicit_value_model_ready = bool(
            self._runtime_config.get("value_model_ready", False)
        )
        self._value_model_provenance = self._runtime_config.get(
            "value_model_provenance"
        )

    # ------------------------------------------------------------------
    # Public alignment facade
    # ------------------------------------------------------------------
    @property
    def value_model_ready(self) -> bool:
        """
        Whether ValueEmbeddingModel may contribute authoritative runtime evidence.

        Randomly initialized neural scores must not be treated as alignment
        evidence.  Readiness therefore requires either an explicit subsystem
        configuration marker or a runtime marker placed on the model by the
        code that loaded/validated its trained parameters.
        """
        if self._explicit_value_model_ready:
            return True
        marker = getattr(
            self.value_embedding_model,
            self.value_model_ready_attribute,
            False,
        )
        return bool(marker)

    def mark_value_model_ready(
        self,
        *,
        provenance: Optional[str] = None,
    ) -> None:
        """
        Mark the already-loaded value model as validated for runtime evidence.

        This method does not train or load the model.  Training/loading remains
        the responsibility of the owning model/training infrastructure.
        """
        setattr(
            self.value_embedding_model,
            self.value_model_ready_attribute,
            True,
        )
        self._explicit_value_model_ready = True
        if provenance:
            self._value_model_provenance = str(provenance)

    def score_value_alignment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Score value/preference alignment only when model readiness is explicit.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            return {
                "available": False,
                "score": None,
                "confidence": 0.0,
                "source": "value_embedding_model",
                "reason": "invalid_or_empty_value_input",
                "model_ready": self.value_model_ready,
            }

        if self.value_model_requires_ready_marker and not self.value_model_ready:
            return {
                "available": False,
                "score": None,
                "confidence": 0.0,
                "source": "value_embedding_model",
                "reason": "value_model_not_marked_ready",
                "model_ready": False,
                "provenance": self._value_model_provenance,
            }

        try:
            score = float(self.value_embedding_model.score_trajectory(data))
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"ValueEmbeddingModel returned score outside [0, 1]: {score}"
                )
            return {
                "available": True,
                "score": score,
                "confidence": 1.0,
                "source": "value_embedding_model",
                "reason": None,
                "model_ready": True,
                "provenance": self._value_model_provenance,
            }
        except Exception as exc:
            wrapped = wrap_alignment_exception(
                exc,
                target_cls=ValueEmbeddingError,
                message="Value alignment scoring failed.",
                context={"model_ready": self.value_model_ready},
            )
            logger.warning("%s", wrapped)
            return {
                "available": False,
                "score": None,
                "confidence": 0.0,
                "source": "value_embedding_model",
                "reason": str(wrapped),
                "model_ready": self.value_model_ready,
                "provenance": self._value_model_provenance,
            }

    def evaluate_constraints(
        self,
        action_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate only Alignment-owned runtime constraints.

        Safety, privacy, fairness, and general policy enforcement are deliberately
        not re-run here.  Their assessments are accepted separately as delegated
        evidence by ``AlignmentAssessor``.  This method preserves support for
        human/runtime alignment constraints already represented by
        ``EthicalConstraints`` without turning Alignment into a second Safety or
        Privacy engine.
        """
        context = normalize_context(action_context or {}, drop_none=False)
        evaluator = self.ethical_constraints

        dynamic_fn = getattr(
            evaluator,
            "_check_dynamic_constraints",
            None,
        )
        if not callable(dynamic_fn):
            return {
                "approved": None,
                "violations": [],
                "corrective_actions": [],
                "explanations": [],
                "subreports": {},
                "summary": {
                    "active_alignment_constraints": 0,
                    "evidence_sufficient": False,
                    "delegated_layers": [
                        "safety",
                        "privacy",
                        "fairness",
                        "general_policy",
                    ],
                },
                "scope": "alignment_runtime_constraints_only",
                "available": False,
            }

        dynamic = dynamic_fn(context)
        if not isinstance(dynamic, Mapping):
            raise ConfigurationError(
                "EthicalConstraints._check_dynamic_constraints() must return a mapping."
            )

        details = dynamic.get("details", [])
        active_count = (
            len(details)
            if isinstance(details, Sequence)
            and not isinstance(details, (str, bytes, bytearray))
            else 0
        )
        raw_violations = dynamic.get("violations", [])
        violations = (
            [str(item) for item in raw_violations]
            if isinstance(raw_violations, Sequence)
            and not isinstance(raw_violations, (str, bytes, bytearray))
            else []
        )
        raw_corrections = dynamic.get("corrections", [])
        corrective_actions = (
            [
                dict(item) if isinstance(item, Mapping) else {"action": str(item)}
                for item in raw_corrections
            ]
            if isinstance(raw_corrections, Sequence)
            and not isinstance(raw_corrections, (str, bytes, bytearray))
            else []
        )
        raw_explanations = dynamic.get("explanations", [])
        explanations = (
            [str(item) for item in raw_explanations]
            if isinstance(raw_explanations, Sequence)
            and not isinstance(raw_explanations, (str, bytes, bytearray))
            else []
        )

        # An empty runtime-constraint registry is absence of evidence, not proof
        # that all applicable system policies are satisfied.
        evidence_sufficient = active_count > 0
        approved: Optional[bool]
        if evidence_sufficient:
            approved = not violations
        else:
            approved = None

        return {
            "approved": approved,
            "violations": violations,
            "corrective_actions": corrective_actions,
            "explanations": explanations,
            "subreports": {"dynamic": dict(dynamic)},
            "summary": {
                "active_alignment_constraints": active_count,
                "violation_count": len(violations),
                "evidence_sufficient": evidence_sufficient,
                "delegated_layers": [
                    "safety",
                    "privacy",
                    "fairness",
                    "general_policy",
                ],
            },
            "scope": "alignment_runtime_constraints_only",
            "available": evidence_sufficient,
        }

    def assess(
        self,
        *,
        task_context: Mapping[str, Any],
        component_report: Mapping[str, Any],
        assessment_id: Optional[str] = None,
        record_trajectory: bool = True,
    ) -> AlignmentAssessment:
        assessment = self.assessor.assess(
            task_context=task_context,
            component_report=component_report,
            assessment_id=assessment_id,
            record_trajectory=record_trajectory,
        )
        if record_trajectory and self.memory_logging_enabled:
            self._record_assessment(
                task_context=task_context,
                assessment=assessment,
            )
        return assessment

    def memory_report(self) -> Dict[str, Any]:
        """Return memory diagnostics without exposing the memory object."""
        try:
            return self._memory.get_memory_report()
        except Exception as exc:
            wrapped = wrap_alignment_exception(
                exc,
                target_cls=AlignmentMemoryError,
                message="Failed to generate alignment runtime memory report.",
            )
            if self.strict_memory_integration:
                raise wrapped
            logger.warning("%s", wrapped)
            return {
                "status": "degraded",
                "error": str(wrapped),
            }

    def health(self) -> AlignmentRuntimeHealth:
        components = {
            "bias_detector": self.bias_detector is not None,
            "fairness_evaluator": self.fairness_evaluator is not None,
            "ethical_constraints": self.ethical_constraints is not None,
            "value_embedding_model": self.value_embedding_model is not None,
            "counterfactual_auditor": self.counterfactual_auditor is not None,
            "alignment_assessor": self.assessor is not None,
        }
        healthy = all(components.values()) and self._memory is not None
        return AlignmentRuntimeHealth(
            status="healthy" if healthy else "degraded",
            components=components,
            memory_available=self._memory is not None,
            value_model_ready=self.value_model_ready,
        )

    def export_state(self) -> Dict[str, Any]:
        """
        Export facade-owned lightweight state.

        Persistent AlignmentMemory remains owned by the memory subsystem.
        """
        return {
            "trajectory": self.assessor.tracker.export_state(),
            "value_model_ready": self.value_model_ready,
            "value_model_provenance": self._value_model_provenance,
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("AlignmentSubsystem state must be a mapping.")
        trajectory = state.get("trajectory")
        if isinstance(trajectory, Mapping):
            self.assessor.tracker.import_state(trajectory)
        if bool(state.get("value_model_ready", False)):
            self.mark_value_model_ready(
                provenance=state.get("value_model_provenance")
            )

    # ------------------------------------------------------------------
    # Internal construction / persistence
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        if not isinstance(self._runtime_config, Mapping):
            raise ConfigurationError(
                "alignment_runtime configuration must be a mapping."
            )
        if not isinstance(self._assessment_config, Mapping):
            raise ConfigurationError(
                "alignment_assessment configuration must be a mapping."
            )

    def _bind_shared_memory(self, component: Any) -> None:
        """
        Rebind known alignment-module memory attributes to one shared store.

        This is deliberately contained here rather than in AlignmentAgent.
        """
        if component is None:
            return

        if hasattr(component, "alignment_memory"):
            try:
                setattr(component, "alignment_memory", self._memory)
            except Exception:
                logger.debug(
                    "Could not rebind alignment_memory on %s.",
                    type(component).__name__,
                    exc_info=True,
                )

        if hasattr(component, "memory"):
            current = getattr(component, "memory", None)
            if isinstance(current, AlignmentMemory):
                try:
                    setattr(component, "memory", self._memory)
                except Exception:
                    logger.debug(
                        "Could not rebind memory on %s.",
                        type(component).__name__,
                        exc_info=True,
                    )

        # Counterfactual fairness helper can retain a memory reference of its own.
        fairness_assessor = getattr(component, "fairness_assessor", None)
        if fairness_assessor is not None and hasattr(fairness_assessor, "memory"):
            try:
                setattr(fairness_assessor, "memory", self._memory)
            except Exception:
                logger.debug(
                    "Could not rebind counterfactual fairness memory.",
                    exc_info=True,
                )

    def _record_assessment(
        self,
        *,
        task_context: Mapping[str, Any],
        assessment: AlignmentAssessment,
    ) -> None:
        context = {
            "task_id": task_context.get("task_id"),
            "audit_id": task_context.get("audit_id"),
            "domain": (
                task_context.get("context", {}).get("domain")
                if isinstance(task_context.get("context"), Mapping)
                else None
            ),
        }
        context = {
            key: value
            for key, value in context.items()
            if value is not None
        }
        if not context:
            context = {"scope": "alignment_runtime"}

        violation = assessment.status.value in {
            "misaligned",
            "conflicting",
        }
        ethics_violations = sum(
            1
            for summary in assessment.dimensions.values()
            if summary.status.value == "misaligned"
        )

        try:
            self._memory.record_outcome(
                context=context,
                outcome={
                    "alignment_score": assessment.aggregate_score,
                    "bias_rate": 0.0,
                    "ethics_violations": ethics_violations,
                    "violation": violation,
                },
                source="alignment_assessment",
                tags=["agent_alignment", assessment.status.value],
                metadata={
                    "assessment": json_safe(assessment.to_dict()),
                },
            )
        except Exception as exc:
            wrapped = wrap_alignment_exception(
                exc,
                target_cls=AlignmentMemoryError,
                message="Failed to persist structured alignment assessment.",
                context={
                    "task_id": task_context.get("task_id"),
                    "audit_id": task_context.get("audit_id"),
                },
            )
            if self.strict_memory_integration:
                raise wrapped
            logger.warning("%s", wrapped)


__all__ = [
    "AlignmentRuntimeHealth",
    "AlignmentSubsystem",
]
