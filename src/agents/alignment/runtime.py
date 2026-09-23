"""Public facade for the SLAI alignment subsystem.

The AlignmentAgent depends on this facade rather than on private subsystem
configuration or AlignmentMemory. The facade owns alignment-private construction,
evidence normalization, bounded trajectory assessment, and persistence.

It intentionally does not enforce system safety/privacy policy, plan actions,
perform generic reasoning, or learn a reward/policy model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .assessment import (
    AlignmentAssessment,
    AlignmentAssessor,
    AlignmentDimension,
    AlignmentEvidence,
)
from .alignment_memory import AlignmentMemory
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Alignment Runtime")


@dataclass(frozen=True)
class AlignmentRuntimeHealth:
    status: str
    memory_available: bool
    value_model_ready: bool
    initialized_components: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "memory_available": self.memory_available,
            "value_model_ready": self.value_model_ready,
            "initialized_components": list(self.initialized_components),
        }


def _finite_probability(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


class AlignmentSubsystem:
    """Encapsulated alignment runtime used by AlignmentAgent."""

    SCORE_KEYS: Dict[AlignmentDimension, Tuple[str, ...]] = {
        AlignmentDimension.USER_INTENT: ("intent_alignment_score", "user_intent_alignment_score"),
        AlignmentDimension.TASK_OBJECTIVE: ("task_alignment_score", "task_objective_alignment_score"),
        AlignmentDimension.CONSTRAINT: ("constraint_alignment_score", "constraint_compliance_score"),
        AlignmentDimension.ENVIRONMENT: ("environment_alignment_score", "environment_consistency_score"),
        AlignmentDimension.PREFERENCE: ("preference_alignment_score", "value_alignment_score"),
        AlignmentDimension.TRAJECTORY: ("trajectory_alignment_score", "behavioral_alignment_score"),
        AlignmentDimension.SELF_CONSTRAINT: ("self_constraint_alignment_score", "resource_alignment_score"),
        AlignmentDimension.MULTI_AGENT: ("multi_agent_alignment_score", "collaboration_alignment_score"),
    }

    def __init__(
        self,
        *,
        alignment_memory: Optional[AlignmentMemory] = None,
        assessment_config: Optional[Mapping[str, Any]] = None,
        runtime_config: Optional[Mapping[str, Any]] = None,
        components: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # Agent-facing assessment/runtime policy is injected by AlignmentAgent
        # from src/agents/base/configs/agents_config.yaml.  This subsystem does
        # not read the agent configuration file itself.
        self._assessment_config = dict(assessment_config or {})
        self._runtime_config = dict(runtime_config or {})
        self._memory = alignment_memory or AlignmentMemory()
        self.assessor = AlignmentAssessor(self._assessment_config)
        self._components: Dict[str, Any] = dict(components or {})
        self._predict_func: Optional[Callable[[Any], Any]] = None

        # Components supplied by AlignmentAgent may have constructed their own
        # local AlignmentMemory instances. Rebind their public memory hooks to
        # the one private subsystem memory so runtime evidence/history does not
        # fragment across parallel stores. AlignmentAgent itself never sees the
        # private memory object.
        for component_name, component in tuple(self._components.items()):
            self._bind_component_memory(component_name, component)

        self.memory_logging_enabled = bool(self._runtime_config.get("memory_logging_enabled", True))
        self.strict_memory_integration = bool(self._runtime_config.get("strict_memory_integration", False))
        self.run_statistical_diagnostics = bool(self._runtime_config.get("run_statistical_diagnostics", True))
        self.use_statistical_diagnostics_as_alignment_evidence = bool(
            self._runtime_config.get("use_statistical_diagnostics_as_alignment_evidence", False)
        )
        self.value_model_requires_ready_marker = bool(
            self._runtime_config.get("value_model_requires_ready_marker", True)
        )
        self._explicit_value_model_ready = bool(self._runtime_config.get("value_model_ready", False))
        self._value_model_provenance = self._runtime_config.get("value_model_provenance")

    # ------------------------------------------------------------------
    # Private component wiring / lazy specialist components
    # ------------------------------------------------------------------
    def _bind_component_memory(self, name: str, component: Any) -> Any:
        """Bind a specialist's public memory hook to subsystem-private memory."""
        if component is None:
            return component
        if hasattr(component, "alignment_memory"):
            try:
                component.alignment_memory = self._memory
            except Exception as exc:
                logger.debug("Could not bind %s.alignment_memory: %s", name, exc)
        if name == "value_embedding_model" and hasattr(component, "memory"):
            try:
                component.memory = self._memory
            except Exception as exc:
                logger.debug("Could not bind value_embedding_model.memory: %s", exc)
        return component

    def _component(self, name: str) -> Any:
        if name in self._components:
            return self._components[name]
        if name == "bias_detector":
            from .bias_detection import BiasDetector
            component = BiasDetector(alignment_memory=self._memory)
        elif name == "fairness_evaluator":
            from .fairness_evaluator import FairnessEvaluator
            component = FairnessEvaluator()
            if hasattr(component, "alignment_memory"):
                component.alignment_memory = self._memory
        elif name == "ethical_constraints":
            from .ethical_constraints import EthicalConstraints
            component = EthicalConstraints()
            if hasattr(component, "alignment_memory"):
                component.alignment_memory = self._memory
        elif name == "value_embedding_model":
            from .value_embedding_model import ValueEmbeddingModel
            component = ValueEmbeddingModel()
            if hasattr(component, "memory"):
                component.memory = self._memory
        elif name == "counterfactual_auditor":
            from .counterfactual_auditor import CounterfactualAuditor
            component = CounterfactualAuditor(
                model_predict_func=self._predict_func,
                alignment_memory=self._memory,
            )
        else:
            raise KeyError(f"unknown alignment component: {name}")
        component = self._bind_component_memory(name, component)
        self._components[name] = component
        return component

    @property
    def predict_func(self) -> Optional[Callable[[Any], Any]]:
        return self._predict_func

    @predict_func.setter
    def predict_func(self, func: Optional[Callable[[Any], Any]]) -> None:
        if func is not None and not callable(func):
            raise TypeError("predict_func must be callable or None")
        self._predict_func = func
        auditor = self._components.get("counterfactual_auditor")
        if auditor is not None and func is not None:
            setter = getattr(auditor, "set_model_predict_func", None)
            if callable(setter):
                setter(func)

    @property
    def value_model_ready(self) -> bool:
        if self._explicit_value_model_ready:
            return True
        model = self._components.get("value_embedding_model")
        return bool(model is not None and getattr(model, "alignment_evidence_ready", False))

    def mark_value_model_ready(self, *, provenance: Optional[str] = None) -> None:
        model = self._component("value_embedding_model")
        setattr(model, "alignment_evidence_ready", True)
        self._explicit_value_model_ready = True
        if provenance:
            self._value_model_provenance = str(provenance)

    # ------------------------------------------------------------------
    # Evidence collection
    # ------------------------------------------------------------------
    def _explicit_evidence(self, payload: Mapping[str, Any]) -> List[AlignmentEvidence]:
        output: List[AlignmentEvidence] = []
        raw = payload.get("alignment_evidence", ())
        if isinstance(raw, Mapping):
            raw = [raw]
        for item in _sequence(raw):
            if not isinstance(item, Mapping):
                continue
            try:
                output.append(AlignmentEvidence.from_mapping(item, default_source="task_context"))
            except (TypeError, ValueError) as exc:
                logger.warning("Ignoring malformed alignment evidence: %s", exc)

        default_confidence = _finite_probability(payload.get("alignment_signal_confidence")) or 0.75
        for dimension, keys in self.SCORE_KEYS.items():
            for key in keys:
                if key not in payload:
                    continue
                score = _finite_probability(payload.get(key))
                output.append(AlignmentEvidence(
                    dimension=dimension,
                    source=f"task_context:{key}",
                    score=score,
                    confidence=default_confidence if score is not None else 0.0,
                    required=dimension.value in self.assessor.required_dimensions,
                    kind="explicit_score",
                    details={"input_key": key},
                    evidence_id=key,
                ))
                break
        return output

    def _delegated_evidence(self, payload: Mapping[str, Any]) -> List[AlignmentEvidence]:
        """Normalize evidence already produced by other SLAI owners.

        Alignment does not call Safety, Privacy, Reasoning, Planning, Evaluation,
        or Collaborative agents here. The orchestrator may pass their outputs as
        delegated evidence with explicit alignment dimensions.
        """
        output: List[AlignmentEvidence] = []
        raw = payload.get("delegated_evidence", ())
        if isinstance(raw, Mapping):
            raw = [raw]
        for item in _sequence(raw):
            if not isinstance(item, Mapping):
                continue
            try:
                output.append(AlignmentEvidence.from_mapping(
                    item,
                    default_source=str(item.get("source") or "delegated_component"),
                    default_confidence=0.75,
                ))
            except (TypeError, ValueError) as exc:
                logger.warning("Ignoring malformed delegated alignment evidence: %s", exc)
        return output

    def _runtime_constraint_evidence(self, payload: Mapping[str, Any]) -> Tuple[List[AlignmentEvidence], Dict[str, Any]]:
        """Evaluate Alignment-owned dynamic constraints only.

        The full EthicalConstraints.enforce() surface includes safety/privacy/
        fairness policy concerns that are owned elsewhere. We therefore use only
        its dynamic runtime constraint registry when available.
        """
        if not bool(payload.get("enable_ethics_check", True)):
            return [], {"available": False, "reason": "ethics_check_disabled"}
        if not bool(payload.get("evaluate_alignment_constraints", True)):
            return [], {"available": False, "reason": "disabled"}
        try:
            evaluator = self._component("ethical_constraints")
            dynamic_fn = getattr(evaluator, "_check_dynamic_constraints", None)
            if not callable(dynamic_fn):
                return [], {"available": False, "reason": "dynamic_constraint_api_unavailable"}
            context = _mapping(payload.get("action_context")) or _mapping(payload.get("context"))
            result = dynamic_fn(context)
            if not isinstance(result, Mapping):
                return [], {"available": False, "reason": "invalid_dynamic_constraint_result"}
            details = list(_sequence(result.get("details", ())))
            violations = list(_sequence(result.get("violations", ())))
            if not details and not violations:
                return [], {"available": False, "reason": "no_active_alignment_constraints", "result": dict(result)}
            score = 0.0 if violations else 1.0
            evidence = AlignmentEvidence(
                dimension=AlignmentDimension.CONSTRAINT,
                source="alignment_dynamic_constraints",
                score=score,
                confidence=1.0,
                required=True,
                kind="runtime_constraint_check",
                details={"violation_count": len(violations), "violations": [str(v) for v in violations]},
                evidence_id="alignment_dynamic_constraints",
            )
            return [evidence], {"available": True, "result": dict(result)}
        except Exception as exc:
            logger.warning("Alignment-specific constraint assessment failed: %s", exc)
            return [AlignmentEvidence(
                dimension=AlignmentDimension.CONSTRAINT,
                source="alignment_dynamic_constraints",
                score=None,
                confidence=0.0,
                required=True,
                kind="runtime_constraint_check",
                details={"error": str(exc)},
                evidence_id="alignment_dynamic_constraints",
            )], {"available": False, "error": str(exc)}

    def _value_evidence(self, payload: Mapping[str, Any]) -> Tuple[List[AlignmentEvidence], Dict[str, Any]]:
        if not bool(payload.get("enable_value_alignment", True)):
            return [], {"available": False, "reason": "value_alignment_disabled"}
        value_data = payload.get("value_data")
        if value_data is None:
            return [], {"available": False, "reason": "value_data_not_supplied"}
        if self.value_model_requires_ready_marker and not self.value_model_ready:
            return [AlignmentEvidence(
                dimension=AlignmentDimension.PREFERENCE,
                source="value_embedding_model",
                score=None,
                confidence=0.0,
                kind="model_score",
                details={"reason": "value_model_not_marked_ready", "provenance": self._value_model_provenance},
                evidence_id="value_embedding_model",
            )], {"available": False, "reason": "value_model_not_marked_ready"}
        try:
            model = self._component("value_embedding_model")
            score = _finite_probability(model.score_trajectory(value_data))
            if score is None:
                raise ValueError("value model returned a non-finite or out-of-range score")
            return [AlignmentEvidence(
                dimension=AlignmentDimension.PREFERENCE,
                source="value_embedding_model",
                score=score,
                confidence=1.0,
                kind="model_score",
                details={"provenance": self._value_model_provenance},
                evidence_id="value_embedding_model",
            )], {"available": True, "score": score, "provenance": self._value_model_provenance}
        except Exception as exc:
            logger.warning("Value alignment scoring unavailable: %s", exc)
            return [AlignmentEvidence(
                dimension=AlignmentDimension.PREFERENCE,
                source="value_embedding_model",
                score=None,
                confidence=0.0,
                kind="model_score",
                details={"error": str(exc)},
                evidence_id="value_embedding_model",
            )], {"available": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Existing statistical modules remain diagnostics, not a universal scalar.
    # ------------------------------------------------------------------
    def run_diagnostics(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if not self.run_statistical_diagnostics:
            return {"enabled": False}
        data = payload.get("input_data")
        predictions = payload.get("predictions")
        if data is None or predictions is None:
            return {"enabled": True, "available": False, "reason": "input_data_and_predictions_required"}
        try:
            import numpy as np
            import pandas as pd
            frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            pred = np.asarray(predictions).reshape(-1).astype(float)
            if frame.empty or len(frame) != len(pred):
                return {"enabled": True, "available": False, "reason": "invalid_data_or_prediction_length"}
        except Exception as exc:
            return {"enabled": True, "available": False, "reason": f"input_normalization_failed: {exc}"}

        labels = payload.get("labels")
        label_array = None
        if labels is not None:
            try:
                import numpy as np
                label_array = np.asarray(labels).reshape(-1).astype(float)
                if len(label_array) != len(frame):
                    label_array = None
            except Exception:
                label_array = None

        report: Dict[str, Any] = {"enabled": True, "available": True, "errors": []}
        if bool(payload.get("enable_bias_detection", True)):
            try:
                report["bias"] = self._component("bias_detector").compute_metrics(frame, pred, label_array)
            except Exception as exc:
                report["errors"].append({"component": "bias_detector", "error": str(exc)})
        else:
            report["bias"] = {"available": False, "reason": "bias_detection_disabled"}

        # Never fabricate labels. Label-dependent fairness is unavailable when
        # ground truth is absent rather than being computed against zeros.
        if label_array is not None:
            try:
                report["group_fairness"] = self._component("fairness_evaluator").evaluate_group_fairness(
                    frame, pred, label_array
                )
            except Exception as exc:
                report["errors"].append({"component": "fairness_evaluator.group", "error": str(exc)})
        else:
            report["group_fairness"] = {"available": False, "reason": "labels_required"}

        if bool(payload.get("enable_individual_fairness", True)):
            try:
                numeric = frame.select_dtypes(include=["number"])
                if len(numeric) > 1 and not numeric.empty:
                    report["individual_fairness"] = self._component("fairness_evaluator").evaluate_individual_fairness(
                        numeric, pred
                    )
            except Exception as exc:
                report["errors"].append({"component": "fairness_evaluator.individual", "error": str(exc)})

        if bool(payload.get("enable_counterfactual_audit", False)):
            sensitive = payload.get("sensitive_attributes") or []
            if self._predict_func is None:
                report["counterfactual"] = {"available": False, "reason": "predict_func_required"}
            elif not sensitive:
                report["counterfactual"] = {"available": False, "reason": "sensitive_attributes_required"}
            else:
                try:
                    auditor = self._component("counterfactual_auditor")
                    y_col = None
                    audit_frame = frame.copy()
                    if label_array is not None:
                        y_col = str(payload.get("label_column") or "__alignment_labels__")
                        audit_frame[y_col] = label_array
                    report["counterfactual"] = auditor.audit(
                        audit_frame,
                        sensitive_attrs=list(sensitive),
                        y_true_col=y_col,
                        context=_mapping(payload.get("context")),
                    )
                except Exception as exc:
                    report["errors"].append({"component": "counterfactual_auditor", "error": str(exc)})
        return report

    def _diagnostic_evidence(self, diagnostics: Mapping[str, Any]) -> List[AlignmentEvidence]:
        if not self.use_statistical_diagnostics_as_alignment_evidence:
            return []
        # Only consume explicit normalized overall scores exposed by specialist
        # reports. Unknown metric dictionaries are not collapsed heuristically.
        output: List[AlignmentEvidence] = []
        for key in ("bias", "group_fairness", "individual_fairness", "counterfactual"):
            report = diagnostics.get(key)
            if not isinstance(report, Mapping):
                continue
            raw = report.get("alignment_score")
            if raw is None and key == "counterfactual" and "overall_bias" in report:
                bias = _finite_probability(report.get("overall_bias"))
                raw = None if bias is None else 1.0 - bias
            score = _finite_probability(raw)
            if score is None:
                continue
            output.append(AlignmentEvidence(
                dimension=AlignmentDimension.CONSTRAINT,
                source=f"alignment_diagnostic:{key}",
                score=score,
                confidence=0.75,
                kind="statistical_diagnostic",
                required=False,
                evidence_id=f"diagnostic:{key}",
            ))
        return output

    def collect_evidence(self, payload: Mapping[str, Any]) -> Tuple[List[AlignmentEvidence], Dict[str, Any]]:
        evidence = self._explicit_evidence(payload)
        evidence.extend(self._delegated_evidence(payload))
        constraint_evidence, constraint_report = self._runtime_constraint_evidence(payload)
        evidence.extend(constraint_evidence)
        value_evidence, value_report = self._value_evidence(payload)
        evidence.extend(value_evidence)
        diagnostics = self.run_diagnostics(payload)
        evidence.extend(self._diagnostic_evidence(diagnostics))
        return evidence, {
            "constraints": constraint_report,
            "value_alignment": value_report,
            "diagnostics": diagnostics,
        }

    def assess(
        self,
        payload: Mapping[str, Any],
        *,
        assessment_id: Optional[str] = None,
        record_trajectory: bool = True,
    ) -> Tuple[AlignmentAssessment, Dict[str, Any], List[AlignmentEvidence]]:
        evidence, component_report = self.collect_evidence(payload)
        assessment = self.assessor.assess(
            evidence,
            assessment_id=assessment_id,
            record_trajectory=record_trajectory,
            metadata={"task_id": payload.get("task_id"), "audit_id": payload.get("audit_id")},
        )
        if record_trajectory and self.memory_logging_enabled:
            self._record_assessment(payload, assessment)
        return assessment, component_report, evidence

    def record_feedback(self, feedback: Mapping[str, Any]) -> Dict[str, Any]:
        """Record alignment-specific feedback without rewriting governing policy."""
        context = _mapping(feedback.get("context"))
        if not context:
            context = {"feedback_id": feedback.get("feedback_id") or feedback.get("task_id") or "alignment_feedback"}
        score = _finite_probability(feedback.get("alignment_score"))
        outcome = {
            "alignment_score": score,
            "bias_rate": _finite_probability(feedback.get("bias_rate")) or 0.0,
            "ethics_violations": int(feedback.get("ethics_violations", 0) or 0),
            "violation": bool(feedback.get("violation", False)),
        }
        return self._memory.record_outcome(
            context=context,
            outcome=outcome,
            source=str(feedback.get("source") or "human_feedback"),
            tags=["feedback", "alignment"],
            metadata=_mapping(feedback.get("metadata")),
        )

    def _record_assessment(self, payload: Mapping[str, Any], assessment: AlignmentAssessment) -> None:
        context = {
            "task_id": payload.get("task_id"),
            "audit_id": payload.get("audit_id"),
            "domain": _mapping(payload.get("context")).get("domain"),
            "alignment_status": assessment.status.value,
        }
        try:
            self._memory.record_outcome(
                context=context,
                outcome={
                    "alignment_score": assessment.aggregate_score,
                    "bias_rate": 0.0,
                    "ethics_violations": sum(
                        1 for item in assessment.dimensions.values()
                        if item.status.value == "misaligned"
                    ),
                    "violation": assessment.requires_review,
                },
                source="alignment_runtime",
                tags=["assessment", assessment.status.value],
                metadata={
                    "confidence": assessment.confidence,
                    "coverage": assessment.coverage,
                    "uncertainty": assessment.uncertainty,
                    "conflict_count": len(assessment.conflicts),
                    "drift": assessment.drift.to_dict(),
                },
            )
        except Exception as exc:
            if self.strict_memory_integration:
                raise
            logger.warning("Failed to persist alignment assessment: %s", exc)

    # ------------------------------------------------------------------
    # Facade diagnostics/state -- memory object itself remains private.
    # ------------------------------------------------------------------
    def memory_report(self) -> Dict[str, Any]:
        method = getattr(self._memory, "get_memory_report", None)
        if not callable(method):
            return {"status": "available", "detail": "report_api_unavailable"}
        try:
            result = method()
            return dict(result) if isinstance(result, Mapping) else {"status": "available", "report": result}
        except Exception as exc:
            if self.strict_memory_integration:
                raise
            return {"status": "degraded", "error": str(exc)}

    def health(self) -> AlignmentRuntimeHealth:
        return AlignmentRuntimeHealth(
            status="healthy" if self._memory is not None else "degraded",
            memory_available=self._memory is not None,
            value_model_ready=self.value_model_ready,
            initialized_components=tuple(sorted(self._components.keys())),
        )

    def export_state(self) -> Dict[str, Any]:
        return {
            "trajectory": self.assessor.tracker.export_state(),
            "value_model_ready": self.value_model_ready,
            "value_model_provenance": self._value_model_provenance,
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("AlignmentSubsystem state must be a mapping")
        trajectory = state.get("trajectory")
        if isinstance(trajectory, Mapping):
            self.assessor.tracker.import_state(trajectory)
        if bool(state.get("value_model_ready", False)):
            self._explicit_value_model_ready = True
            self._value_model_provenance = state.get("value_model_provenance")

__all__ = [
    # Dataclass
    "AlignmentRuntimeHealth",
    # Facade class
    "AlignmentSubsystem",
    # Re-exported from .assessment (used by AlignmentAgent and callers)
    "AlignmentAssessment",
    "AlignmentAssessor",
    "AlignmentDimension",
    "AlignmentEvidence",
    # Re-exported from .alignment_memory (used by AlignmentAgent and callers)
    "AlignmentMemory",
]