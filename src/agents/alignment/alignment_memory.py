"""
Continual Alignment Memory System
Implements:
- Causal outcome tracing (Goyal et al., 2019)
- Experience replay for alignment (Parisi et al., 2019)
- Concept drift detection
- Intervention effect tracking
"""

from __future__ import annotations

import json
import pickle
import joblib
import hashlib
import numpy as np
import pandas as pd

from pyexpat import features
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from datetime import datetime, timedelta
from collections import deque
from scipy.stats import entropy  # type: ignore
from sklearn.linear_model import SGDRegressor, BayesianRidge  # type: ignore

from .utils.config_loader import load_global_config, get_config_section
from .utils.alignment_errors import *
from logs.logger import get_logger, PrettyPrinter # type: ignore

logger = get_logger("Alignment Memory")
printer = PrettyPrinter

class AlignmentMemory:
    """
    Persistent alignment memory module with:
    - Temporal outcome logging
    - Causal intervention analysis
    - Experience replay for training
    - Concept drift detection

    Memory Structure:
    1. Alignment Logs: Raw evaluation metrics over time
    2. Context Registry: Domain-specific outcome statistics
    3. Intervention Graph: Corrections and their effects
    4. Causal Model: Learned relationships between actions/outcomes
    """

    ALIGNMENT_LOG_COLUMNS = [
        "timestamp",
        "metric",
        "value",
        "threshold",
        "violation",
        "context_hash",
        "context_snapshot",
        "source",
        "audit_id",
        "tags",
        "metadata",
    ]
    OUTCOME_HISTORY_COLUMNS = [
        "timestamp",
        "context_hash",
        "context_snapshot",
        "alignment_score",
        "bias_rate",
        "ethics_violations",
        "violation",
        "source",
        "tags",
        "metadata",
    ]
    CONTEXT_REGISTRY_COLUMNS = [
        "context_hash",
        "first_seen",
        "last_updated",
        "total_events",
        "violation_count",
        "violation_rate",
        "mean_metric_value",
        "mean_bias_rate",
        "total_ethics_violations",
        "outcome_count",
        "last_metric",
        "last_value",
        "sample_context",
    ]

    def __init__(
        self,
        config_section_name: str = "alignment_memory",
        config_file_path: Optional[str] = None,
    ):
        self.config = load_global_config()
        self.config_section_name = ensure_non_empty_string(
            config_section_name,
            "config_section_name",
            error_cls=ConfigurationError,
        )
        self.config_file_path = config_file_path
        self.memory_config = get_config_section(self.config_section_name)
        self._validate_memory_config()

        self.replay_buffer_size = int(self.memory_config["replay_buffer_size"])
        self.causal_window = int(self.memory_config["causal_window"])
        self.drift_threshold = float(self.memory_config["drift_threshold"])
        self.retention_period = int(self.memory_config["retention_period"])
        self.regressor_type = str(self.memory_config["regressor_type"]).strip().lower()
        self.min_samples_for_drift = int(self.memory_config["min_samples_for_drift"])
        self.histogram_bins = int(self.memory_config["histogram_bins"])
        self.histogram_range = tuple(self.memory_config["histogram_range"])
        self.context_decay = float(self.memory_config["context_decay"])
        self.violation_decay = float(self.memory_config["violation_decay"])
        self.bayesian_fit_min_samples = int(self.memory_config["bayesian_fit_min_samples"])
        self.model_learning_rate = float(self.memory_config["model_learning_rate"])
        self.max_context_registry_size = int(self.memory_config["max_context_registry_size"])
        self.default_outcome_bias_rate = float(self.memory_config["default_outcome_bias_rate"])
        self.default_ethics_violations = int(self.memory_config["default_ethics_violations"])
        self.retention_cleanup_on_write = bool(self.memory_config["retention_cleanup_on_write"])
        self.checkpoint_protocol = int(self.memory_config["checkpoint_protocol"])
        self.checkpoint_model_suffix = str(self.memory_config["checkpoint_model_suffix"])
        self.intervention_feature_keys = list(self.memory_config["intervention_feature_keys"])

        self.alignment_logs = pd.DataFrame(columns=self.ALIGNMENT_LOG_COLUMNS)
        self.outcome_history = pd.DataFrame(columns=self.OUTCOME_HISTORY_COLUMNS)
        self.context_registry = pd.DataFrame(columns=self.CONTEXT_REGISTRY_COLUMNS)
        self.intervention_graph: List[Dict[str, Any]] = []

        self.concept_drift_scores: List[Dict[str, Any]] = []
        self.replay_buffer: deque = deque(maxlen=self.replay_buffer_size)
        self.model_history: List[Dict[str, Any]] = []
        self.intervention_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self.causal_model = self._create_regressor(self.regressor_type)

        if config_file_path:
            logger.debug(
                "AlignmentMemory received config_file_path=%s but retained global config loader handling.",
                config_file_path,
            )

        logger.info(
            "AlignmentMemory initialized | regressor=%s replay_buffer_size=%s retention_days=%s",
            self.regressor_type,
            self.replay_buffer_size,
            self.retention_period,
        )

    # ------------------------------------------------------------------
    # Configuration and validation
    # ------------------------------------------------------------------
    def _validate_memory_config(self) -> None:
        try:
            ensure_mapping(
                self.memory_config,
                self.config_section_name,
                allow_empty=False,
                error_cls=ConfigurationError,
            )
            ensure_keys_present(
                self.memory_config,
                [
                    "replay_buffer_size",
                    "causal_window",
                    "drift_threshold",
                    "retention_period",
                    "regressor_type",
                    "min_samples_for_drift",
                    "histogram_bins",
                    "histogram_range",
                    "context_decay",
                    "violation_decay",
                    "bayesian_fit_min_samples",
                    "model_learning_rate",
                    "max_context_registry_size",
                    "default_outcome_bias_rate",
                    "default_ethics_violations",
                    "retention_cleanup_on_write",
                    "checkpoint_protocol",
                    "checkpoint_model_suffix",
                    "intervention_feature_keys",
                ],
                field_name=self.config_section_name,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.memory_config["replay_buffer_size"],
                "replay_buffer_size",
                min_value=1,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.memory_config["causal_window"],
                "causal_window",
                min_value=1,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.memory_config["drift_threshold"],
                "drift_threshold",
                min_value=0.0,
                max_value=10.0,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.memory_config["retention_period"],
                "retention_period",
                min_value=1,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.memory_config["min_samples_for_drift"],
                "min_samples_for_drift",
                min_value=2,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.memory_config["histogram_bins"],
                "histogram_bins",
                min_value=2,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.memory_config["context_decay"],
                "context_decay",
                min_value=0.0,
                max_value=1.0,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.memory_config["violation_decay"],
                "violation_decay",
                min_value=0.0,
                max_value=1.0,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.memory_config["bayesian_fit_min_samples"],
                "bayesian_fit_min_samples",
                min_value=1,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.memory_config["model_learning_rate"],
                "model_learning_rate",
                min_value=0.0,
                max_value=10.0,
                inclusive=False,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.memory_config["max_context_registry_size"],
                "max_context_registry_size",
                min_value=1,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.memory_config["default_outcome_bias_rate"],
                "default_outcome_bias_rate",
                min_value=0.0,
                max_value=1.0,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.memory_config["default_ethics_violations"],
                "default_ethics_violations",
                min_value=0,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.memory_config["checkpoint_protocol"],
                "checkpoint_protocol",
                min_value=0,
                error_cls=ConfigurationError,
            )
            ensure_non_empty_string(
                self.memory_config["checkpoint_model_suffix"],
                "checkpoint_model_suffix",
                error_cls=ConfigurationError,
            )
            histogram_range = self.memory_config["histogram_range"]
            if not isinstance(histogram_range, (list, tuple)) or len(histogram_range) != 2:
                raise ConfigurationError(
                    "'histogram_range' must be a two-element sequence [min, max].",
                    context={"histogram_range": histogram_range},
                )
            min_hist, max_hist = float(histogram_range[0]), float(histogram_range[1])
            if min_hist >= max_hist:
                raise ConfigurationError(
                    "'histogram_range' must contain an increasing numeric range.",
                    context={"histogram_range": histogram_range},
                )
            regressor_type = str(self.memory_config["regressor_type"]).strip().lower()
            if regressor_type not in {"sgd", "bayesian"}:
                raise ConfigurationError(
                    "'regressor_type' must be either 'sgd' or 'bayesian'.",
                    context={"regressor_type": self.memory_config["regressor_type"]},
                )
            feature_keys = self.memory_config["intervention_feature_keys"]
            if not isinstance(feature_keys, list) or not feature_keys:
                raise ConfigurationError(
                    "'intervention_feature_keys' must be a non-empty list.",
                    context={"intervention_feature_keys": feature_keys},
                )
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=ConfigurationError,
                message="AlignmentMemory configuration validation failed.",
                context={
                    "config_section": self.config_section_name,
                    "config_path": self.config.get("__config_path__"),
                },
            )

    def _create_regressor(self, regressor_type: str) -> Union[SGDRegressor, BayesianRidge]:
        if regressor_type == "bayesian":
            return BayesianRidge()
        return SGDRegressor(eta0=self.model_learning_rate, learning_rate="constant", random_state=42)

    def _normalise_context(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        ensure_mapping(context, "context", allow_empty=False, error_cls=DataValidationError)
        return self._json_safe_mapping(context)

    def _json_safe_mapping(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        return json.loads(json.dumps(value, default=self._json_default, sort_keys=True))

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (set, tuple)):
            return list(value)
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    def _hash_context(self, context: Mapping[str, Any]) -> str:
        context_json = json.dumps(
            self._json_safe_mapping(context),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(context_json.encode("utf-8")).hexdigest()

    def _coerce_tags(self, tags: Optional[Iterable[Any]]) -> List[str]:
        if not tags:
            return []
        result: List[str] = []
        for tag in tags:
            text = str(tag).strip()
            if text and text not in result:
                result.append(text)
        return result

    def _current_timestamp(self) -> datetime:
        return datetime.now()

    def _coerce_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if hasattr(value, "to_pydatetime"):
            return value.to_pydatetime()
        return datetime.fromisoformat(str(value).replace("Z", ""))

    def _normalise_datetime_columns(self) -> None:
        for frame, columns in [
            (self.alignment_logs, ["timestamp"]),
            (self.outcome_history, ["timestamp"]),
            (self.context_registry, ["first_seen", "last_updated"]),
        ]:
            if frame.empty:
                continue
            for column in columns:
                if column in frame.columns:
                    frame[column] = frame[column].apply(lambda value: self._coerce_datetime(value) if pd.notna(value) else value)

    def _append_dataframe_row(self, frame: pd.DataFrame, row: Dict[str, Any]) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame([row])
        return pd.concat([frame, pd.DataFrame([row])], ignore_index=True)

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------
    def log_evaluation(
        self,
        metric: str,
        value: float,
        threshold: float,
        context: Dict[str, Any],
        *,
        source: str = "alignment",
        tags: Optional[Iterable[Any]] = None,
        audit_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a single alignment evaluation outcome and update memory state."""
        try:
            metric_name = ensure_non_empty_string(metric, "metric", error_cls=DataValidationError)
            metric_value = ensure_numeric_range(value, "value", error_cls=DataValidationError)
            metric_threshold = ensure_numeric_range(threshold, "threshold", error_cls=DataValidationError)
            source_name = ensure_non_empty_string(source, "source", error_cls=DataValidationError)
            context_snapshot = self._normalise_context(context)
            metadata = self._json_safe_mapping(metadata or {})
            context_hash = self._hash_context(context_snapshot)
            timestamp = self._current_timestamp()
            violation = metric_value > metric_threshold

            entry = {
                "timestamp": timestamp,
                "metric": metric_name,
                "value": metric_value,
                "threshold": metric_threshold,
                "violation": bool(violation),
                "context_hash": context_hash,
                "context_snapshot": context_snapshot,
                "source": source_name,
                "audit_id": audit_id,
                "tags": self._coerce_tags(tags),
                "metadata": metadata,
            }

            self.alignment_logs = self._append_dataframe_row(self.alignment_logs, entry)
            self._update_replay_buffer(entry)
            self._update_context_registry_from_evaluation(entry)
            if self.retention_cleanup_on_write:
                self.enforce_retention()
            return entry
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=AlignmentMemoryError,
                message="Failed to log alignment evaluation.",
                context={"metric": metric, "source": source, "audit_id": audit_id},
            )

    def record_outcome(
        self,
        context: Dict[str, Any],
        outcome: Dict[str, Any],
        *,
        source: str = "outcome",
        tags: Optional[Iterable[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store context-specific operational outcomes for downstream analysis."""
        try:
            context_snapshot = self._normalise_context(context)
            outcome_mapping = ensure_mapping(outcome, "outcome", allow_empty=False, error_cls=DataValidationError)
            source_name = ensure_non_empty_string(source, "source", error_cls=DataValidationError)
            metadata = self._json_safe_mapping(metadata or {})

            bias_rate = float(outcome_mapping.get("bias_rate", self.default_outcome_bias_rate))
            ethics_violations = int(outcome_mapping.get("ethics_violations", self.default_ethics_violations))
            alignment_score = outcome_mapping.get("alignment_score")
            alignment_score = float(alignment_score) if alignment_score is not None else None
            explicit_violation = outcome_mapping.get("violation")
            violation_flag = bool(explicit_violation) if explicit_violation is not None else ethics_violations > 0

            ensure_numeric_range(bias_rate, "bias_rate", min_value=0.0, max_value=1.0, error_cls=DataValidationError)
            ensure_numeric_range(ethics_violations, "ethics_violations", min_value=0, error_cls=DataValidationError)
            if alignment_score is not None:
                ensure_numeric_range(alignment_score, "alignment_score", error_cls=DataValidationError)

            timestamp = self._current_timestamp()
            context_hash = self._hash_context(context_snapshot)
            record = {
                "timestamp": timestamp,
                "context_hash": context_hash,
                "context_snapshot": context_snapshot,
                "alignment_score": alignment_score,
                "bias_rate": bias_rate,
                "ethics_violations": ethics_violations,
                "violation": violation_flag,
                "source": source_name,
                "tags": self._coerce_tags(tags),
                "metadata": metadata,
            }
            self.outcome_history = self._append_dataframe_row(self.outcome_history, record)
            self._update_context_registry_from_outcome(record)
            if self.retention_cleanup_on_write:
                self.enforce_retention()
            return record
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=AlignmentMemoryError,
                message="Failed to record alignment outcome.",
                context={"source": source},
            )

    def apply_correction(
        self,
        correction: Dict[str, Any],
        effect: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
        source: str = "correction",
        tags: Optional[Iterable[Any]] = None,
    ) -> Dict[str, Any]:
        """Log an intervention and update the causal impact model."""
        try:
            correction_mapping = ensure_mapping(correction, "correction", allow_empty=False, error_cls=DataValidationError)
            effect_mapping = ensure_mapping(effect, "effect", allow_empty=False, error_cls=DataValidationError)
            ensure_keys_present(correction_mapping, ["type", "magnitude", "target"], field_name="correction", error_cls=MissingFieldError)
            ensure_keys_present(effect_mapping, ["alignment_score", "violation_rate"], field_name="effect", error_cls=MissingFieldError)

            correction_type = ensure_non_empty_string(correction_mapping["type"], "correction.type", error_cls=DataValidationError)
            correction_magnitude = ensure_numeric_range(
                correction_mapping["magnitude"],
                "correction.magnitude",
                min_value=0.0,
                error_cls=DataValidationError,
            )
            target = correction_mapping["target"]
            if isinstance(target, str):
                targets = [target]
            elif isinstance(target, Sequence):
                targets = [str(item) for item in target]
            else:
                raise DataValidationError(
                    "'correction.target' must be a string or a sequence of target identifiers.",
                    context={"target": target},
                )
            if not targets:
                raise DataValidationError("'correction.target' must not be empty.")

            alignment_score = ensure_numeric_range(
                effect_mapping["alignment_score"],
                "effect.alignment_score",
                error_cls=DataValidationError,
            )
            violation_rate = ensure_numeric_range(
                effect_mapping["violation_rate"],
                "effect.violation_rate",
                min_value=0.0,
                max_value=1.0,
                error_cls=DataValidationError,
            )

            pre_state = self._get_current_state()
            timestamp = self._current_timestamp()
            context_snapshot = self._normalise_context(context) if context else None
            context_hash = self._hash_context(context_snapshot) if context_snapshot else None

            intervention = {
                "intervention_id": f"INT-{len(self.intervention_graph) + 1:06d}",
                "timestamp": timestamp,
                "type": correction_type,
                "magnitude": correction_magnitude,
                "target": targets,
                "source": ensure_non_empty_string(source, "source", error_cls=DataValidationError),
                "tags": self._coerce_tags(tags),
                "context_hash": context_hash,
                "context_snapshot": context_snapshot,
                "correction": self._json_safe_mapping(correction_mapping),
                "pre_state": pre_state,
                "post_state": {
                    "alignment_score": alignment_score,
                    "violation_rate": violation_rate,
                    **self._json_safe_mapping(effect_mapping),
                },
                "causal_impact": None,
            }
            self.intervention_graph.append(intervention)
            self._update_causal_model(intervention)
            if self.retention_cleanup_on_write:
                self.enforce_retention()
            return intervention
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=AlignmentMemoryError,
                message="Failed to apply correction to alignment memory.",
                context={"source": source},
            )

    def analyze_causes(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        """Analyze the recent causal impact of interventions using the fitted regressor."""
        try:
            effective_window = int(window_size or self.causal_window)
            ensure_numeric_range(effective_window, "window_size", min_value=1, error_cls=DataValidationError)

            if not self.intervention_graph:
                return {
                    "status": "no_interventions",
                    "window_size": effective_window,
                    "n_interventions": 0,
                    "feature_names": self.intervention_feature_keys,
                }

            recent = self.intervention_graph[-effective_window:]
            if len(recent) < 2 or not hasattr(self.causal_model, "coef_"):
                return {
                    "status": "insufficient_model_state",
                    "window_size": effective_window,
                    "n_interventions": len(recent),
                    "feature_names": self.intervention_feature_keys,
                }

            features = np.array([self._encode_intervention(intervention) for intervention in recent])
            coef_array: np.ndarray = np.asarray(self.causal_model.coef_)
            impacts = features @ coef_array
            deltas = np.array([
                intervention["post_state"]["alignment_score"] - intervention["pre_state"]["alignment_score"]
                for intervention in recent
            ])

            for idx, intervention in enumerate(recent):
                intervention["causal_impact"] = float(impacts[idx])

            summary = {
                "status": "ok",
                "window_size": effective_window,
                "n_interventions": len(recent),
                "feature_names": self.intervention_feature_keys,
                "max_impact": float(np.max(impacts)),
                "min_impact": float(np.min(impacts)),
                "mean_impact": float(np.mean(impacts)),
                "median_impact": float(np.median(impacts)),
                "mean_effect": float(np.mean(deltas)),
                "impact_std": float(np.std(impacts)) if len(impacts) > 1 else 0.0,
                "coefficient_map": {
                    feature: float(coef)
                    for feature, coef in zip(self.intervention_feature_keys, np.asarray(self.causal_model.coef_).tolist())
                },
            }
            return summary
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=AlignmentMemoryError,
                message="Failed to analyse causal intervention effects.",
                context={"window_size": window_size},
            )

    def detect_drift(
        self,
        window_size: Optional[int] = None,
        *,
        metric: Optional[str] = None,
        return_details: bool = False,
    ) -> Union[bool, Dict[str, Any]]:
        """KL-divergence based concept drift detection over replay-buffered metric values."""
        try:
            effective_window = int(window_size or max(30, self.min_samples_for_drift // 2))
            ensure_numeric_range(effective_window, "window_size", min_value=2, error_cls=DataValidationError)

            replay_entries = list(self.replay_buffer)
            if metric:
                metric_name = ensure_non_empty_string(metric, "metric", error_cls=DataValidationError)
                replay_entries = [entry for entry in replay_entries if entry.get("metric") == metric_name]
            else:
                metric_name = None

            minimum_required = max(2 * effective_window, self.min_samples_for_drift)
            if len(replay_entries) < minimum_required:
                details = {
                    "status": "insufficient_data",
                    "detected": False,
                    "metric": metric_name,
                    "window_size": effective_window,
                    "required_samples": minimum_required,
                    "available_samples": len(replay_entries),
                }
                return details if return_details else False

            recent = replay_entries[-effective_window:]
            historical = replay_entries[-2 * effective_window:-effective_window]
            low, high = float(self.histogram_range[0]), float(self.histogram_range[1])
            bins = np.linspace(low, high, self.histogram_bins + 1)

            p = np.histogram([float(item["value"]) for item in recent], bins=bins)[0].astype(float)
            q = np.histogram([float(item["value"]) for item in historical], bins=bins)[0].astype(float)
            p = (p + 1e-9) / (np.sum(p) + 1e-9 * len(p))
            q = (q + 1e-9) / (np.sum(q) + 1e-9 * len(q))
            kl_div = float(entropy(p, q))
            detected = kl_div > self.drift_threshold

            details = {
                "status": "ok",
                "detected": detected,
                "metric": metric_name,
                "window_size": effective_window,
                "drift_score": kl_div,
                "threshold": self.drift_threshold,
                "evaluated_at": self._current_timestamp().isoformat(),
            }
            self.concept_drift_scores.append(details)
            return details if return_details else detected
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=ConceptDriftError,
                message="Failed to evaluate concept drift in alignment memory.",
                context={"window_size": window_size, "metric": metric},
            )

    def get_memory_report(self) -> Dict[str, Any]:
        """Generate a comprehensive operational report over memory state."""
        try:
            return {
                "generated_at": self._current_timestamp().isoformat(),
                "state": self._get_current_state(),
                "temporal_summary": self._temporal_analysis(),
                "context_analysis": self._context_statistics(),
                "intervention_effects": self._intervention_impact(),
                "model_diagnostics": self.get_model_diagnostics(),
                "drift_status": self.detect_drift(return_details=True),
            }
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=AlignmentMemoryError,
                message="Failed to generate alignment memory report.",
            )

    def get_logs_by_tag(self, tag_value: str, tag_key: str = "audit_id") -> pd.DataFrame:
        """Retrieve alignment log entries by audit identifier or tag membership."""
        try:
            lookup_value = ensure_non_empty_string(tag_value, "tag_value", error_cls=DataValidationError)
            lookup_key = ensure_non_empty_string(tag_key, "tag_key", error_cls=DataValidationError)
            if self.alignment_logs.empty:
                return self.alignment_logs.copy()

            logs = self.alignment_logs.copy()
            if lookup_key == "tags":
                return logs[logs["tags"].apply(lambda tags: lookup_value in (tags or []))].copy()
            if lookup_key not in logs.columns:
                raise DataValidationError(
                    f"Unsupported log lookup key '{lookup_key}'.",
                    context={"available_keys": list(logs.columns)},
                )
            return logs[logs[lookup_key] == lookup_value].copy()
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=AlignmentMemoryError,
                message="Failed to query alignment logs by tag.",
                context={"tag_value": tag_value, "tag_key": tag_key},
            )

    def query_logs(
        self,
        *,
        metric: Optional[str] = None,
        violation_only: bool = False,
        context_hash: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Flexible retrieval interface over stored alignment logs."""
        logs = self.alignment_logs.copy()
        if metric:
            logs = logs[logs["metric"] == metric]
        if violation_only:
            logs = logs[logs["violation"]]
        if context_hash:
            logs = logs[logs["context_hash"] == context_hash]
        if since is not None:
            logs = logs[logs["timestamp"] >= since]
        logs = logs.sort_values("timestamp", ascending=False)
        if limit is not None:
            ensure_numeric_range(limit, "limit", min_value=1, error_cls=DataValidationError)
            logs = logs.head(int(limit))
        return logs.reset_index(drop=True)

    def get_violation_history(self, metric: Optional[str] = None, *,
                              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return historical violation records from the alignment log store."""
        violations = self.query_logs(metric=metric, violation_only=True, limit=limit)
        records = violations.to_dict(orient="records")
        # Convert any non‑string keys to strings
        return [{str(k): v for k, v in record.items()} for record in records]

    def switch_regressor(self, new_type: str, *, reset_training_history: bool = False) -> None:
        """Switch between supported causal regressors while preserving memory state."""
        try:
            regressor_type = ensure_non_empty_string(new_type, "new_type", error_cls=ConfigurationError).lower()
            if regressor_type not in {"sgd", "bayesian"}:
                raise ConfigurationError(
                    "'new_type' must be either 'sgd' or 'bayesian'.",
                    context={"new_type": new_type},
                )
            self.regressor_type = regressor_type
            self.causal_model = self._create_regressor(self.regressor_type)
            if reset_training_history:
                self.model_history = []
                self.intervention_data = []
            logger.info("Switched AlignmentMemory regressor to %s", self.regressor_type)
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=AlignmentMemoryError,
                message="Failed to switch AlignmentMemory regressor.",
                context={"new_type": new_type},
            )

    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Return causal-model training history and current parameter state."""
        diagnostics: Dict[str, Any] = {
            "history": self.model_history,
            "current_model": {
                "type": self.regressor_type,
                "trained": hasattr(self.causal_model, "coef_"),
                "n_samples": len(self.intervention_data) if self.regressor_type == "bayesian" else len(self.model_history),
                "feature_names": self.intervention_feature_keys,
            },
        }
        if hasattr(self.causal_model, "coef_"):
            diagnostics["current_model"]["coef"] = np.asarray(self.causal_model.coef_).tolist()
        if hasattr(self.causal_model, "intercept_"):
            intercept = self.causal_model.intercept_
            # Convert intercept to a serialisable form
            if isinstance(intercept, np.ndarray):
                diagnostics["current_model"]["intercept"] = intercept.tolist()
            else:
                diagnostics["current_model"]["intercept"] = float(intercept)
        if self.model_history:
            diagnostics["current_model"]["last_loss"] = self.model_history[-1].get("loss")
        return diagnostics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self, path: str) -> Path:
        """Persist the causal model and its training state to disk."""
        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "model": self.causal_model,
                    "regressor_type": self.regressor_type,
                    "history": self.model_history,
                    "intervention_data": self.intervention_data,
                    "feature_names": self.intervention_feature_keys,
                    "saved_at": self._current_timestamp().isoformat(),
                },
                output_path,
            )
            logger.info("Saved AlignmentMemory causal model to %s", output_path)
            return output_path
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=PersistenceError,
                message="Failed to save AlignmentMemory causal model.",
                context={"path": path},
            )

    def load_model(self, path: str) -> Path:
        """Load the persisted causal model and training history from disk."""
        try:
            model_path = Path(path)
            if not model_path.exists():
                raise PersistenceError(
                    "Causal model file does not exist.",
                    context={"path": str(model_path)},
                )
            model_data = joblib.load(model_path)
            ensure_mapping(model_data, "model_data", allow_empty=False, error_cls=PersistenceError)
            self.causal_model = model_data["model"]
            self.regressor_type = str(model_data.get("regressor_type", self.regressor_type)).lower()
            self.model_history = list(model_data.get("history", []))
            self.intervention_data = list(model_data.get("intervention_data", []))
            logger.info("Loaded AlignmentMemory causal model from %s", model_path)
            return model_path
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=PersistenceError,
                message="Failed to load AlignmentMemory causal model.",
                context={"path": path},
            )

    def save_checkpoint(self, path: str) -> Path:
        """Persist the full memory state, excluding the causal model binary itself."""
        try:
            checkpoint_path = Path(path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model_path = checkpoint_path.with_name(checkpoint_path.stem + self.checkpoint_model_suffix)

            payload = {
                "config_section_name": self.config_section_name,
                "config_path": self.config.get("__config_path__"),
                "saved_at": self._current_timestamp().isoformat(),
                "regressor_type": self.regressor_type,
                "alignment_logs": self.alignment_logs.to_dict(orient="records"),
                "outcome_history": self.outcome_history.to_dict(orient="records"),
                "context_registry": self.context_registry.to_dict(orient="records"),
                "intervention_graph": self.intervention_graph,
                "concept_drift_scores": self.concept_drift_scores,
                "replay_buffer": list(self.replay_buffer),
                "model_history": self.model_history,
                "intervention_data": self.intervention_data,
                "model_path": str(model_path),
            }
            with checkpoint_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=self.checkpoint_protocol)
            self.save_model(str(model_path))
            logger.info("Saved AlignmentMemory checkpoint to %s", checkpoint_path)
            return checkpoint_path
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=PersistenceError,
                message="Failed to save AlignmentMemory checkpoint.",
                context={"path": path},
            )

    def load_checkpoint(self, path: str) -> Path:
        """Restore the full memory state and associated causal model from a checkpoint."""
        try:
            checkpoint_path = Path(path)
            if not checkpoint_path.exists():
                raise PersistenceError(
                    "AlignmentMemory checkpoint file does not exist.",
                    context={"path": str(checkpoint_path)},
                )
            with checkpoint_path.open("rb") as handle:
                state = pickle.load(handle)
            ensure_mapping(state, "checkpoint_state", allow_empty=False, error_cls=PersistenceError)

            self.regressor_type = str(state.get("regressor_type", self.regressor_type)).lower()
            self.alignment_logs = pd.DataFrame(state.get("alignment_logs", []), columns=self.ALIGNMENT_LOG_COLUMNS)
            self.outcome_history = pd.DataFrame(state.get("outcome_history", []), columns=self.OUTCOME_HISTORY_COLUMNS)
            self.context_registry = pd.DataFrame(state.get("context_registry", []), columns=self.CONTEXT_REGISTRY_COLUMNS)
            self._normalise_datetime_columns()
            self.intervention_graph = list(state.get("intervention_graph", []))
            self.concept_drift_scores = list(state.get("concept_drift_scores", []))
            self.replay_buffer = deque(state.get("replay_buffer", []), maxlen=self.replay_buffer_size)
            self.model_history = list(state.get("model_history", []))
            self.intervention_data = list(state.get("intervention_data", []))
            self.causal_model = self._create_regressor(self.regressor_type)

            model_path = state.get("model_path")
            if model_path:
                self.load_model(model_path)
            logger.info("Loaded AlignmentMemory checkpoint from %s", checkpoint_path)
            return checkpoint_path
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=PersistenceError,
                message="Failed to load AlignmentMemory checkpoint.",
                context={"path": path},
            )

    # ------------------------------------------------------------------
    # State management and statistics
    # ------------------------------------------------------------------
    def enforce_retention(self) -> Dict[str, int]:
        """Apply retention policy across logs, outcomes, interventions, and context registry."""
        try:
            cutoff = self._current_timestamp() - timedelta(days=self.retention_period)
            before = {
                "alignment_logs": len(self.alignment_logs),
                "outcome_history": len(self.outcome_history),
                "intervention_graph": len(self.intervention_graph),
                "context_registry": len(self.context_registry),
                "model_history": len(self.model_history),
            }

            if not self.alignment_logs.empty:
                self.alignment_logs = self.alignment_logs[self.alignment_logs["timestamp"] >= cutoff].reset_index(drop=True)
            if not self.outcome_history.empty:
                self.outcome_history = self.outcome_history[self.outcome_history["timestamp"] >= cutoff].reset_index(drop=True)
            self.intervention_graph = [
                intervention for intervention in self.intervention_graph if intervention["timestamp"] >= cutoff
            ]
            self.model_history = [
                entry for entry in self.model_history
                if entry.get("timestamp") and self._coerce_datetime(entry["timestamp"]) >= cutoff
            ] if self.model_history else []

            if not self.context_registry.empty:
                self.context_registry = self.context_registry[
                    self.context_registry["last_updated"] >= cutoff
                ].sort_values("last_updated", ascending=False).head(self.max_context_registry_size).reset_index(drop=True)

            after = {
                "alignment_logs": len(self.alignment_logs),
                "outcome_history": len(self.outcome_history),
                "intervention_graph": len(self.intervention_graph),
                "context_registry": len(self.context_registry),
                "model_history": len(self.model_history),
            }
            return {key: before[key] - after[key] for key in before}
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=AlignmentMemoryError,
                message="Failed to enforce AlignmentMemory retention policy.",
            )

    def _update_replay_buffer(self, entry: Dict[str, Any]) -> None:
        self.replay_buffer.append(entry)

    def _update_context_registry_from_evaluation(self, entry: Dict[str, Any]) -> None:
        context_hash = entry["context_hash"]
        now = entry["timestamp"]
        metric_value = float(entry["value"])
        violation = bool(entry["violation"])
    
        if self.context_registry.empty or context_hash not in self.context_registry["context_hash"].values:
            new_entry = {
                "context_hash": context_hash,
                "first_seen": now,
                "last_updated": now,
                "total_events": 1,
                "violation_count": int(violation),
                "violation_rate": float(violation),
                "mean_metric_value": metric_value,
                "mean_bias_rate": self.default_outcome_bias_rate,
                "total_ethics_violations": self.default_ethics_violations,
                "outcome_count": 0,
                "last_metric": entry["metric"],
                "last_value": metric_value,
                "sample_context": entry["context_snapshot"],
            }
            self.context_registry = self._append_dataframe_row(self.context_registry, new_entry)
            return

        idx = self.context_registry[self.context_registry["context_hash"] == context_hash].index[0]
        # Convert pandas scalars to native int/float
        total_events = int(self._to_native_scalar(self.context_registry.at[idx, "total_events"])) + 1 # pyright: ignore[reportArgumentType]
        previous_mean = float(self._to_native_scalar(self.context_registry.at[idx, "mean_metric_value"])) # pyright: ignore[reportArgumentType]
        previous_violation_rate = float(self._to_native_scalar(self.context_registry.at[idx, "violation_rate"])) # pyright: ignore[reportArgumentType]
        previous_violation_count = int(self._to_native_scalar(self.context_registry.at[idx, "violation_count"])) # pyright: ignore[reportArgumentType]
    
        self.context_registry.at[idx, "last_updated"] = now
        self.context_registry.at[idx, "total_events"] = total_events
        self.context_registry.at[idx, "violation_count"] = previous_violation_count + int(violation)
        self.context_registry.at[idx, "violation_rate"] = (
            previous_violation_rate * self.violation_decay + float(violation) * (1.0 - self.violation_decay)
        )
        self.context_registry.at[idx, "mean_metric_value"] = (
            previous_mean * (total_events - 1) + metric_value
        ) / total_events
        self.context_registry.at[idx, "last_metric"] = entry["metric"]
        self.context_registry.at[idx, "last_value"] = metric_value
        self.context_registry.at[idx, "sample_context"] = entry["context_snapshot"]

    def _update_context_registry_from_outcome(self, record: Dict[str, Any]) -> None:
        context_hash = record["context_hash"]
        now = record["timestamp"]
        bias_rate = float(record["bias_rate"])
        ethics_violations = int(record["ethics_violations"])
    
        if self.context_registry.empty or context_hash not in self.context_registry["context_hash"].values:
            new_entry = {
                "context_hash": context_hash,
                "first_seen": now,
                "last_updated": now,
                "total_events": 0,
                "violation_count": 0,
                "violation_rate": 0.0,
                "mean_metric_value": 0.0,
                "mean_bias_rate": bias_rate,
                "total_ethics_violations": ethics_violations,
                "outcome_count": 1,
                "last_metric": None,
                "last_value": None,
                "sample_context": record["context_snapshot"],
            }
            self.context_registry = self._append_dataframe_row(self.context_registry, new_entry)
            return
    
        idx = self.context_registry[self.context_registry["context_hash"] == context_hash].index[0]
        outcome_count = int(self._to_native_scalar(self.context_registry.at[idx, "outcome_count"])) + 1 # pyright: ignore[reportArgumentType]
        previous_bias_rate = float(self._to_native_scalar(self.context_registry.at[idx, "mean_bias_rate"])) # pyright: ignore[reportArgumentType]
        previous_ethics = int(self._to_native_scalar(self.context_registry.at[idx, "total_ethics_violations"])) # pyright: ignore[reportArgumentType]
    
        self.context_registry.at[idx, "last_updated"] = now
        self.context_registry.at[idx, "outcome_count"] = outcome_count
        self.context_registry.at[idx, "mean_bias_rate"] = (
            previous_bias_rate * self.context_decay + bias_rate * (1.0 - self.context_decay)
        )
        self.context_registry.at[idx, "total_ethics_violations"] = previous_ethics + ethics_violations
        self.context_registry.at[idx, "sample_context"] = record["context_snapshot"]

    def _get_current_state(self) -> Dict[str, Any]:
        """Snapshot the current aggregate memory state."""
        if self.alignment_logs.empty:
            alignment_score = 0.0
            violation_rate = 0.0
            threshold_mean = 0.0
        else:
            alignment_score = float(self.alignment_logs["value"].mean())
            violation_rate = float(self.alignment_logs["violation"].mean())
            threshold_mean = float(self.alignment_logs["threshold"].mean())
        return {
            "alignment_score": alignment_score,
            "violation_rate": violation_rate,
            "threshold_mean": threshold_mean,
            "active_contexts": int(len(self.context_registry)),
            "logged_events": int(len(self.alignment_logs)),
            "logged_outcomes": int(len(self.outcome_history)),
            "interventions": int(len(self.intervention_graph)),
        }

    def _encode_intervention(self, intervention: Dict[str, Any]) -> np.ndarray:
        pre_state = intervention.get("pre_state", {})
        post_state = intervention.get("post_state", {})
        values = {
            "magnitude": float(intervention.get("magnitude", 0.0) or 0.0),
            "target_count": float(len(intervention.get("target", []) or [])),
            "pre_alignment_score": float(pre_state.get("alignment_score", 0.0) or 0.0),
            "pre_violation_rate": float(pre_state.get("violation_rate", 0.0) or 0.0),
            "post_alignment_score": float(post_state.get("alignment_score", 0.0) or 0.0),
            "post_violation_rate": float(post_state.get("violation_rate", 0.0) or 0.0),
            "delta_alignment_score": float(post_state.get("alignment_score", 0.0) or 0.0) - float(pre_state.get("alignment_score", 0.0) or 0.0),
            "delta_violation_rate": float(post_state.get("violation_rate", 0.0) or 0.0) - float(pre_state.get("violation_rate", 0.0) or 0.0),
        }
        return np.array([float(values.get(feature, 0.0)) for feature in self.intervention_feature_keys], dtype=float)

    def _update_causal_model(self, intervention: Dict[str, Any]) -> None:
        try:
            features = self._encode_intervention(intervention).reshape(1, -1)
            target = np.array([float(intervention["post_state"]["alignment_score"])], dtype=float)
            timestamp = self._current_timestamp().isoformat()
    
            if self.regressor_type == "bayesian":
                self.intervention_data.append((features, target))
                self.intervention_data = self.intervention_data[-self.causal_window:]
                if len(self.intervention_data) < self.bayesian_fit_min_samples:
                    self.model_history.append(
                        {
                            "timestamp": timestamp,
                            "intervention_id": intervention["intervention_id"],
                            "loss": None,
                            "regressor_type": self.regressor_type,
                            "n_samples": len(self.intervention_data),
                            "status": "buffering",
                        }
                    )
                    return
                full_x = np.vstack([item[0] for item in self.intervention_data])
                full_y = np.concatenate([item[1] for item in self.intervention_data])
                self.causal_model.fit(full_x, full_y)
                predictions = self.causal_model.predict(full_x)
                loss = float(np.mean((full_y - predictions) ** 2))
                intervention["causal_impact"] = float(np.dot(features.flatten(), np.asarray(self.causal_model.coef_)))
                n_samples = len(self.intervention_data)
    
            elif isinstance(self.causal_model, SGDRegressor):
                self.causal_model.partial_fit(features, target)
                prediction = self.causal_model.predict(features)
                loss = float(np.mean((target - prediction) ** 2))
                intervention["causal_impact"] = float(np.dot(features.flatten(), np.asarray(self.causal_model.coef_)))
                n_samples = len(self.model_history) + 1
    
            else:
                raise ConfigurationError(f"Unsupported regressor type: {type(self.causal_model)}")
    
            self.model_history.append(
                {
                    "timestamp": timestamp,
                    "intervention_id": intervention["intervention_id"],
                    "loss": loss,
                    "regressor_type": self.regressor_type,
                    "n_samples": n_samples,
                    "status": "trained",
                }
            )
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=AlignmentMemoryError,
                message="Failed to update AlignmentMemory causal model.",
                context={"intervention_id": intervention.get("intervention_id")},
            )

    def _temporal_analysis(self) -> Dict[str, Any]:
        if self.alignment_logs.empty:
            return {"status": "empty", "metrics": {}}

        grouped = self.alignment_logs.groupby("metric").agg(
            value_mean=("value", "mean"),
            value_std=("value", "std"),
            value_min=("value", "min"),
            value_max=("value", "max"),
            threshold_mean=("threshold", "mean"),
            event_count=("metric", "count"),
            violation_rate=("violation", "mean"),
        )
        grouped = grouped.fillna(0.0)
        return {
            "status": "ok",
            "metrics": grouped.to_dict(orient="index"),
            "window": {
                "first_timestamp": self.alignment_logs["timestamp"].min().isoformat(),
                "last_timestamp": self.alignment_logs["timestamp"].max().isoformat(),
            },
        }

    def _context_statistics(self) -> Dict[str, Any]:
        if self.context_registry.empty:
            return {"status": "empty", "summary": {}, "recent_contexts": [], "violation_extremes": {}}

        registry = self.context_registry.copy().sort_values("last_updated", ascending=False)
        summary = registry[["total_events", "violation_count", "violation_rate", "mean_metric_value", "mean_bias_rate", "total_ethics_violations", "outcome_count"]].describe().fillna(0.0).to_dict()
        most_violations = registry.loc[registry["violation_count"].idxmax()].to_dict()
        highest_bias = registry.loc[registry["mean_bias_rate"].idxmax()].to_dict()

        return {
            "status": "ok",
            "summary": summary,
            "recent_contexts": registry.head(5)[[
                "context_hash",
                "last_updated",
                "total_events",
                "violation_rate",
                "mean_bias_rate",
                "total_ethics_violations",
            ]].to_dict(orient="records"),
            "violation_extremes": {
                "most_violations": {
                    "context_hash": most_violations["context_hash"],
                    "violation_count": int(most_violations["violation_count"]),
                    "violation_rate": float(most_violations["violation_rate"]),
                },
                "highest_bias": {
                    "context_hash": highest_bias["context_hash"],
                    "mean_bias_rate": float(highest_bias["mean_bias_rate"]),
                    "total_ethics_violations": int(highest_bias["total_ethics_violations"]),
                },
            },
        }

    def _intervention_impact(self) -> Dict[str, Any]:
        if not self.intervention_graph:
            return {"status": "empty", "types": {}, "top_interventions": [], "correlation": None}

        interventions = pd.DataFrame(self.intervention_graph)
        if interventions.empty:
            return {"status": "empty", "types": {}, "top_interventions": [], "correlation": None}

        if "causal_impact" not in interventions.columns or interventions["causal_impact"].isna().all():
            return {
                "status": "pending_model_signal",
                "types": interventions["type"].value_counts().to_dict(),
                "top_interventions": interventions[["intervention_id", "type", "magnitude", "target"]].tail(3).to_dict(orient="records"),
                "correlation": None,
            }

        modeled = interventions[interventions["causal_impact"].notna()].copy()
        grouped = modeled.groupby("type").agg(
            mean_impact=("causal_impact", "mean"),
            std_impact=("causal_impact", "std"),
            max_impact=("causal_impact", "max"),
            min_impact=("causal_impact", "min"),
            mean_magnitude=("magnitude", "mean"),
            median_magnitude=("magnitude", "median"),
            count=("type", "count"),
        ).fillna(0.0)

        correlation = None
        if len(modeled) > 1 and modeled["magnitude"].nunique() > 1:
            correlation = float(np.corrcoef(modeled["magnitude"], modeled["causal_impact"])[0, 1])

        top_interventions = modeled.sort_values("causal_impact", ascending=False).head(3)[[
            "intervention_id",
            "type",
            "magnitude",
            "causal_impact",
            "target",
        ]].to_dict(orient="records")

        return {
            "status": "ok",
            "types": grouped.to_dict(orient="index"),
            "top_interventions": top_interventions,
            "correlation": correlation,
        }
    
    def _to_native_scalar(self, value: Any) -> Union[int, float, str, None]:
        """Convert pandas/numpy scalars to native Python types."""
        if value is None:
            return None
        if hasattr(value, "item"):          # numpy or pandas scalar
            value = value.item()
        if isinstance(value, (int, float, str, bool)):
            return value
        try:
            return float(value)             # fallback
        except (TypeError, ValueError):
            return str(value)


if __name__ == "__main__":
    print("\n=== Running Alignment Memory ===\n")
    printer.status("TEST", "Alignment Memory initialized", "info")

    import random
    import tempfile

    rng = np.random.default_rng(42)

    def generate_context(index: int) -> Dict[str, Any]:
        domains = ["medical", "legal", "finance", "education"]
        task_types = ["classification", "generation", "prediction"]
        return {
            "audit_id": f"AUD-{index:05d}",
            "domain": domains[index % len(domains)],
            "task_type": task_types[index % len(task_types)],
            "user": {
                "segment": f"segment_{index % 7}",
                "region": f"region_{index % 5}",
            },
        }

    memory = AlignmentMemory()
    printer.status("TEST", "Alignment Memory config loaded", "success")

    print("\n--- Phase 1: Baseline logging and outcomes ---")
    for idx in range(80):
        context = generate_context(idx)
        metric = random.choice(["toxicity", "factuality", "fairness", "hallucination_risk"])
        value = float(np.clip(rng.normal(loc=0.18, scale=0.08), 0.0, 1.0))
        threshold = 0.30
        memory.log_evaluation(
            metric=metric,
            value=value,
            threshold=threshold,
            context=context,
            source="baseline_monitor",
            tags=["baseline", metric],
            audit_id=context["audit_id"],
            metadata={"phase": 1},
        )
        if idx % 3 == 0:
            memory.record_outcome(
                context=context,
                outcome={
                    "bias_rate": float(np.clip(rng.beta(2, 5), 0.0, 1.0)),
                    "ethics_violations": int(idx % 2),
                    "alignment_score": float(np.clip(rng.normal(loc=0.72, scale=0.07), 0.0, 1.0)),
                },
                source="baseline_outcome",
                tags=["baseline_outcome"],
            )

    printer.pretty("TEST", memory._get_current_state(), "success")

    print("\n--- Phase 2: Interventions and causal updates ---")
    for idx in range(18):
        current_state = memory._get_current_state()
        correction = {
            "type": random.choice(["reinforcement", "constraint", "reweighting", "threshold_adjustment"]),
            "magnitude": float(rng.uniform(0.05, 0.9)),
            "target": random.sample(["toxicity", "fairness", "factuality", "hallucination_risk"], k=2),
        }
        effect = {
            "alignment_score": float(max(0.0, current_state["alignment_score"] + rng.uniform(-0.03, 0.12))),
            "violation_rate": float(np.clip(current_state["violation_rate"] - rng.uniform(0.0, 0.08), 0.0, 1.0)),
        }
        intervention = memory.apply_correction(
            correction=correction,
            effect=effect,
            context=generate_context(1000 + idx),
            source="alignment_agent",
            tags=["intervention"],
        )
        printer.pretty("INTERVENTION", intervention["intervention_id"], "success")

    printer.pretty("MODEL", memory.get_model_diagnostics(), "success")
    printer.pretty("CAUSES", memory.analyze_causes(window_size=12), "success")

    print("\n--- Phase 3: Drift simulation ---")
    for idx in range(100):
        context = generate_context(2000 + idx)
        metric = random.choice(["toxicity", "factuality", "fairness"])
        value = float(np.clip(rng.normal(loc=0.46, scale=0.12), 0.0, 1.0))
        memory.log_evaluation(
            metric=metric,
            value=value,
            threshold=0.30,
            context=context,
            source="post_shift_monitor",
            tags=["shift", metric],
            audit_id=context["audit_id"],
            metadata={"phase": 3},
        )

    drift_report = memory.detect_drift(return_details=True)
    printer.pretty("DRIFT", drift_report, "success")

    print("\n--- Phase 4: Query and retention interfaces ---")
    printer.pretty("QUERY", memory.get_logs_by_tag("baseline", tag_key="tags").head(3).to_dict(orient="records"), "success")
    printer.pretty("VIOLATIONS", memory.get_violation_history(limit=5), "success")
    printer.pretty("RETENTION", memory.enforce_retention(), "success")

    print("\n--- Phase 5: Persistence round-trip ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = str(Path(temp_dir) / "alignment_memory_checkpoint.pkl")
        memory.save_checkpoint(checkpoint_path)
        restored = AlignmentMemory()
        restored.load_checkpoint(checkpoint_path)
        printer.pretty("RESTORED_STATE", restored._get_current_state(), "success")
        printer.pretty("RESTORED_REPORT", restored.get_memory_report(), "success")

    print("\n=== Test ran successfully ===\n")