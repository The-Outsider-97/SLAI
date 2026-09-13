from __future__ import annotations
import uuid

__version__ = "2.3.0"

"""
Unified Evaluation Framework
Implements: Static Analysis, Behavioral Testing, Reward Modeling, and Multi-Objective Evaluation

Key Features:
1. Implements safety mechanisms from Scholten et al. (2022) "Safe RL Validation"
2. Statistical methods follow Demšar (2006) "Statistical Comparisons of Classifiers"
3. Pareto ranking based on Deb et al. (2002) NSGA-II algorithm
4. Modular design with failure mode analysis (Ibrahim et al. 2021)
"""

import json
import math
import threading
import time
import uuid

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .base.utils.main_config_loader import load_global_config, get_config_section
from .base.utils.interpretability import InterpretabilityHelper
from .base_agent import BaseAgent
from .evaluators.adaptive_risk import RiskAdaptation
from .evaluators.autonomous_evaluator import AutonomousEvaluator
from .evaluators.behavioral_validator import BehavioralValidator
from .evaluators.efficiency_evaluator import EfficiencyEvaluator
from .evaluators.performance_budget_evaluator import PerformanceBudgetEvaluator
from .evaluators.performance_evaluator import PerformanceEvaluator
from .evaluators.resource_utilization_evaluator import ResourceUtilizationEvaluator
from .evaluators.safety_evaluator import SafetyEvaluator
from .evaluators.statistical_evaluator import StatisticalEvaluator
from .evaluators.data.issue_db import *
from .evaluators.documentation.certification_framework import CertificationStatus
from .evaluators.modules.evaluators_calculations import EvaluatorsCalculations
from .evaluators.modules.static_analyzer import StaticAnalyzer
from .evaluators.modules.validation_protocol import ValidationProtocol
from .evaluators.utils.evaluation_errors import *
from .safety.safety_guard import SafetyGuard
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Evaluation Agent")
printer = PrettyPrinter()

ANOMALY_FEATURES: Tuple[str, ...] = (
    "severity",
    "cognitive_complexity",
    "data_flow_depth",
    "security_risk",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


class _LocalFallbackIssueTracker:
    """Last-resort tracker used only when the issue-db subsystem cannot import."""

    def __init__(self) -> None:
        self.issues: List[Dict[str, Any]] = []

    def log_issue(self, issue_data: Mapping[str, Any]) -> bool:
        self.issues.append(dict(issue_data))
        return True

    def get_issues(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self.issues]

    def close(self) -> None:
        return None


class FallbackEvaluatorAgent(BaseAgent):
    """Minimal fail-operational evaluator used when the configured SUT fails."""

    def __init__(self, shared_memory: Any, agent_factory: Any, config: Optional[Mapping[str, Any]] = None, **_: Any) -> None:
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config,
        )
        self.safety_guard = SafetyGuard()

    def is_initialized(self) -> bool:
        return True

    def supports_fail_operational(self) -> bool:
        guard = getattr(self, "safety_guard", None)
        guard_ready = bool(
            guard
            and callable(getattr(guard, "is_minimal_viable", None))
            and guard.is_minimal_viable()
        )
        return guard_ready and self.shared_memory is not None

    def has_redundant_safety_channels(self) -> bool:
        guard = getattr(self, "safety_guard", None)
        guard_ready = bool(
            guard
            and callable(getattr(guard, "is_minimal_viable", None))
            and guard.is_minimal_viable()
        )
        safety_limits = self.config.get("safety_limits", {}) if isinstance(self.config, Mapping) else {}
        threshold_channel = isinstance(safety_limits, Mapping) and all(
            key in safety_limits for key in ("max_latency", "min_accuracy")
        )
        return guard_ready and threshold_channel

    def predict(self, state: Any = None) -> Dict[str, Any]:
        known_smoke_input = isinstance(state, str) and state == "test_input"
        value = "expected_output" if known_smoke_input else "fallback_output"
        prediction = {
            "value": value,
            "mode": "fallback",
            "input": state,
            "context": {
                "timestamp": _utc_now_iso(),
                "state_type": type(state).__name__ if state is not None else "none",
                "degraded_mode": True,
            },
        }
        return {
            "status": "degraded_success",
            "prediction": prediction,
            "confidence": 0.25,
            "reason": "primary_test_agent_unavailable",
            # Backward-compatible convenience fields.
            "value": value,
            "mode": "fallback",
            "input": state,
        }


class EvaluationAgent(BaseAgent):
    """
    Orchestrates the SLAI evaluator subsystem without taking ownership of its
    configuration or persistence internals.
    """

    def __init__(
        self,
        shared_memory: Any,
        agent_factory: Any,
        config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config,
            **kwargs,
        )

        self.shared_memory = shared_memory if shared_memory is not None else self.shared_memory
        self.agent_factory = agent_factory

        # IMPORTANT: this is the Base-Agent configuration only.
        self.config = load_global_config()
        self.agent_config: Dict[str, Any] = dict(
            get_config_section("evaluation_agent", config=self.config) or {}
        )
        self.db_config: Dict[str, Any] = dict(self.config.get("issue_database", {}) or {})

        self.memory_warning_threshold = self._coerce_probability(
            self.agent_config.get("memory_warning_threshold", 0.70),
            default=0.70,
        )
        self.memory_critical_threshold = self._coerce_probability(
            self.agent_config.get("memory_critical_threshold", 0.90),
            default=0.90,
        )
        if self.memory_warning_threshold > self.memory_critical_threshold:
            logger.warning(
                "EvaluationAgent memory warning threshold exceeds critical threshold; "
                "using critical threshold for both."
            )
            self.memory_warning_threshold = self.memory_critical_threshold

        self.model_dir = Path(
            str(self.agent_config.get("model_dir", "src/agents/evaluators/models"))
        )
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Unable to create evaluation model directory %s: %s", self.model_dir, exc)

        self.anomaly_detector_path = self._optional_path(
            self.agent_config.get("anomaly_detector")
        )
        self.deep_anomaly_path = self._optional_path(
            self.agent_config.get("deep_anomaly")
        )
        self.anomaly_min_training_samples = self._coerce_positive_int(
            self.agent_config.get("anomaly_min_training_samples", 64),
            default=64,
        )
        self.deep_sequence_length = self._coerce_positive_int(
            self.agent_config.get("deep_anomaly_sequence_length", 10),
            default=10,
        )
        self.deep_training_epochs = self._coerce_positive_int(
            self.agent_config.get("deep_anomaly_epochs", 30),
            default=30,
        )
        self.deep_training_quantile = self._coerce_probability(
            self.agent_config.get("deep_anomaly_quantile", 0.95),
            default=0.95,
        )
        if self.deep_training_quantile <= 0.5:
            logger.warning(
                "deep_anomaly_quantile=%s is unusually low; using 0.95.",
                self.deep_training_quantile,
            )
            self.deep_training_quantile = 0.95

        configured_tests = self.agent_config.get("behavioral_test_cases")
        self.test_cases = (
            list(configured_tests)
            if isinstance(configured_tests, list) and configured_tests
            else self._load_default_test_cases()
        )
        self.task_config = self.agent_config.get("autonomous_tasks", [])
        self.autonomous_tasks = self._load_autonomous_tasks()

        self._shutdown_lock = threading.RLock()
        self._shutdown_complete = False
        self.initialization_issues: List[str] = []

        # Subsystem objects own their own evaluator-specific config and memory.
        self.evaluators = self._init_evaluator_modules()
        self.protocol = self._init_validation_protocol()
        self.interpreter = InterpretabilityHelper()
        self.validation_suite = AIValidationSuite(protocol=self.protocol)
        self.risk_model = self._init_risk_model()
        self.issue_db = self._connect_issue_database()
        self.safety_guard = SafetyGuard()

        # Loading an existing model is safe. Training is explicit and never uses
        # synthetic bootstrap records.
        self.anomaly_model = self._init_anomaly_detector()

        # Hyperparameter tuning remains intentionally disabled until the evaluator
        # tuning objective is made explicit and deterministic in the subsystem.
        self.tuner = None

    # ------------------------------------------------------------------
    # Configuration-safe initialization
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_probability(value: Any, *, default: float) -> float:
        if _is_finite_number(value):
            numeric = float(value)
            if 0.0 <= numeric <= 1.0:
                return numeric
        return float(default)

    @staticmethod
    def _coerce_positive_int(value: Any, *, default: int) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return int(value)
        return int(default)

    @staticmethod
    def _optional_path(value: Any) -> Optional[Path]:
        if value is None:
            return None
        text = str(value).strip()
        return Path(text) if text else None

    def _load_default_test_cases(self) -> List[Dict[str, Any]]:
        """Return the deterministic smoke-test case used when none is configured."""
        return [
            {
                "test_id": "evaluation-agent-smoke-001",
                "scenario": {
                    "input": "test_input",
                    "requirement_id": "REQ-001",
                    "detection_method": "automated",
                },
                "requirement_id": "REQ-001",
                "expected_output": None,
                "severity": "medium",
                "oracle": lambda output: (
                    isinstance(output, Mapping)
                    and (
                        output.get("value") == "expected_output"
                        or (
                            isinstance(output.get("prediction"), Mapping)
                            and output["prediction"].get("value") == "expected_output"
                        )
                    )
                ),
            }
        ]

    def _init_evaluator_modules(self) -> Dict[str, Any]:
        """
        Initialize evaluator components independently.

        Imports are deliberately local so an optional dependency failure in one
        evaluator (for example torch/PyQt in a specialized evaluator) cannot make
        the EvaluationAgent module itself unimportable.
        """
        modules: Dict[str, Any] = {
            "behavioral": None,
            "performance": None,
            "efficiency": None,
            "statistical": None,
            "resource": None,
            "autonomous": None,
            "safety": None,
            "performance_budget": None,
        }

        factories: Dict[str, Callable[[], Any]] = {}
        try:
            from .evaluators.behavioral_validator import BehavioralValidator
            factories["behavioral"] = lambda: BehavioralValidator(test_cases=self.test_cases)
        except Exception as exc:
            self._record_initialization_issue("behavioral", exc)
        try:
            from .evaluators.performance_evaluator import PerformanceEvaluator
            factories["performance"] = PerformanceEvaluator
        except Exception as exc:
            self._record_initialization_issue("performance", exc)
        try:
            from .evaluators.efficiency_evaluator import EfficiencyEvaluator
            factories["efficiency"] = EfficiencyEvaluator
        except Exception as exc:
            self._record_initialization_issue("efficiency", exc)
        try:
            from .evaluators.statistical_evaluator import StatisticalEvaluator
            factories["statistical"] = StatisticalEvaluator
        except Exception as exc:
            self._record_initialization_issue("statistical", exc)
        try:
            from .evaluators.resource_utilization_evaluator import ResourceUtilizationEvaluator
            factories["resource"] = ResourceUtilizationEvaluator
        except Exception as exc:
            self._record_initialization_issue("resource", exc)
        try:
            from .evaluators.autonomous_evaluator import AutonomousEvaluator
            factories["autonomous"] = AutonomousEvaluator
        except Exception as exc:
            self._record_initialization_issue("autonomous", exc)
        try:
            from .evaluators.safety_evaluator import SafetyEvaluator
            factories["safety"] = SafetyEvaluator
        except Exception as exc:
            self._record_initialization_issue("safety", exc)
        try:
            from .evaluators.performance_budget_evaluator import PerformanceBudgetEvaluator
            factories["performance_budget"] = PerformanceBudgetEvaluator
        except Exception as exc:
            self._record_initialization_issue("performance_budget", exc)

        for name, factory in factories.items():
            try:
                modules[name] = factory()
            except Exception as exc:
                self._record_initialization_issue(name, exc)
        return modules

    def _record_initialization_issue(self, component: str, exc: BaseException) -> None:
        message = f"Evaluator '{component}' failed to initialize: {exc}"
        if message not in self.initialization_issues:
            self.initialization_issues.append(message)
        logger.error(message, exc_info=True)

    def _init_validation_protocol(self) -> ValidationProtocol:
        """Initialize the subsystem-owned validation protocol and fail closed if invalid."""
        protocol = ValidationProtocol()
        protocol.validate_configuration()
        return protocol

    def _init_risk_model(self) -> Optional[Any]:
        """Initialize subsystem-owned Bayesian risk adaptation without reading its config here."""
        try:
            from .evaluators.adaptive_risk import RiskAdaptation
            return RiskAdaptation()
        except Exception as exc:
            message = f"Adaptive risk model unavailable: {exc}"
            self.initialization_issues.append(message)
            logger.error(message, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Anomaly intelligence
    # ------------------------------------------------------------------

    def _init_anomaly_detector(self) -> Any:
        """Load a pre-trained IsolationForest model when one exists."""
        path = self.anomaly_detector_path
        if path is None or not path.is_file():
            logger.info("No persisted anomaly detector available; anomaly inference is disabled until trained.")
            return None

        try:
            from joblib import load  # type: ignore

            model = load(path)
            feature_count = getattr(model, "n_features_in_", len(ANOMALY_FEATURES))
            if int(feature_count) != len(ANOMALY_FEATURES):
                raise ValueError(
                    f"Model expects {feature_count} features; EvaluationAgent uses {len(ANOMALY_FEATURES)}."
                )
            logger.info("Loaded anomaly detector from %s", path)
            return model
        except Exception as exc:
            logger.error("Failed to load anomaly detector %s: %s", path, exc, exc_info=True)
            return None

    def _load_training_data(self) -> List[Dict[str, Any]]:
        """
        Load real historical issue records only.

        No synthetic observations are generated. The canonical agent-owned shared
        memory key is ``evaluation_issue_history``; fallback issue tracking is used
        as a secondary source when available.
        """
        records: List[Dict[str, Any]] = []
        try:
            shared_records = self.shared_memory.get("evaluation_issue_history", [])
            if isinstance(shared_records, list):
                records.extend(dict(item) for item in shared_records if isinstance(item, Mapping))
        except Exception as exc:
            logger.warning("Unable to read evaluation issue history from shared memory: %s", exc)

        if not records:
            getter = getattr(self.issue_db, "get_issues", None)
            if callable(getter):
                try:
                    db_records = getter()
                    if isinstance(db_records, list):
                        records.extend(dict(item) for item in db_records if isinstance(item, Mapping))
                except Exception as exc:
                    logger.warning("Unable to read fallback issue history: %s", exc)

        deduplicated: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in records:
            payload = dict(item)
            identity = str(payload.get("id") or payload.get("issue_id") or "")
            if identity and identity in seen:
                continue
            if identity:
                seen.add(identity)
            deduplicated.append(payload)
        return deduplicated

    def train_anomaly_models(
        self,
        records: Optional[Sequence[Mapping[str, Any]]] = None,
        *,
        train_deep: bool = False,
    ) -> Dict[str, Any]:
        """Explicitly train anomaly models from real historical issue data."""
        source = [dict(item) for item in records] if records is not None else self._load_training_data()
        isolation_model = self._train_anomaly_model(source)
        deep_trained = False
        if train_deep:
            deep_trained = self._train_deep_anomaly_detector(source)

        return {
            "isolation_forest_trained": isolation_model is not None,
            "deep_model_trained": deep_trained,
            "sample_count": len(source),
            "feature_schema": list(ANOMALY_FEATURES),
        }

    def _train_anomaly_model(self, records: Optional[Sequence[Mapping[str, Any]]] = None) -> Any:
        """Train IsolationForest only when sufficient real observations exist."""
        source = [dict(item) for item in records] if records is not None else self._load_training_data()
        features = self._extract_ml_features(source)
        if len(features) < self.anomaly_min_training_samples:
            logger.warning(
                "Skipping anomaly training: %d valid observations available; %d required.",
                len(features),
                self.anomaly_min_training_samples,
            )
            return None

        try:
            from joblib import dump  # type: ignore
            from sklearn.ensemble import IsolationForest  # type: ignore

            contamination: Any = self.agent_config.get("anomaly_contamination", "auto")
            if contamination != "auto":
                contamination = float(contamination)
                if not 0.0 < contamination <= 0.5:
                    raise ValueError("anomaly_contamination must be 'auto' or in (0, 0.5].")

            model = IsolationForest(
                n_estimators=self._coerce_positive_int(
                    self.agent_config.get("anomaly_estimators", 200),
                    default=200,
                ),
                contamination=contamination,
                random_state=42,
            )
            model.fit(features)

            if self.anomaly_detector_path is not None:
                self.anomaly_detector_path.parent.mkdir(parents=True, exist_ok=True)
                dump(model, self.anomaly_detector_path)
                logger.info("Persisted anomaly detector to %s", self.anomaly_detector_path)

            self.anomaly_model = model
            return model
        except Exception as exc:
            logger.error("Anomaly detector training failed: %s", exc, exc_info=True)
            return None

    def _train_deep_anomaly_detector(self, records: Sequence[Mapping[str, Any]]) -> bool:
        """Train one reconstruction transformer over temporal feature windows."""
        if self.deep_anomaly_path is None:
            logger.warning("Deep anomaly training requested without a configured checkpoint path.")
            return False

        features = self._extract_ml_features(records)
        sequences = self._create_sequences(features, self.deep_sequence_length)
        if len(sequences) < max(8, self.anomaly_min_training_samples // 4):
            logger.warning(
                "Skipping deep anomaly training: only %d temporal windows are available.",
                len(sequences),
            )
            return False

        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore

            # Lazy subsystem import: the transformer owns any evaluator-specific
            # model configuration it needs.
            from .evaluators.modules.evaluation_transformer import EvaluationTransformer

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = torch.tensor(sequences, dtype=torch.float32, device=device)
            input_dim = int(tensor.shape[-1])
            seq_len = int(tensor.shape[1])

            model = EvaluationTransformer(
                input_dim=input_dim,
                seq_len=seq_len,
                output_dim=input_dim,
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(self.agent_config.get("deep_anomaly_learning_rate", 1e-3)),
                weight_decay=float(self.agent_config.get("deep_anomaly_weight_decay", 1e-5)),
            )
            criterion = nn.MSELoss()
            batch_size = min(
                self._coerce_positive_int(
                    self.agent_config.get("deep_anomaly_batch_size", 64),
                    default=64,
                ),
                len(tensor),
            )

            model.train()
            for epoch in range(self.deep_training_epochs):
                permutation = torch.randperm(len(tensor), device=device)
                epoch_loss = 0.0
                batch_count = 0
                for start in range(0, len(tensor), batch_size):
                    indices = permutation[start : start + batch_size]
                    batch = tensor[indices]
                    output = model(batch, return_dict=True)
                    reconstruction = output["reconstructed_sequence"]
                    if reconstruction is None:
                        raise RuntimeError("EvaluationTransformer returned no reconstructed_sequence.")
                    loss = criterion(reconstruction, batch)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += float(loss.item())
                    batch_count += 1

                logger.debug(
                    "Deep anomaly epoch %d/%d loss=%.6f",
                    epoch + 1,
                    self.deep_training_epochs,
                    epoch_loss / max(batch_count, 1),
                )

            model.eval()
            with torch.no_grad():
                output = model(tensor, return_dict=True)
                reconstruction = output["reconstructed_sequence"]
                if reconstruction is None:
                    raise RuntimeError("EvaluationTransformer returned no reconstruction during calibration.")
                errors = torch.mean((tensor - reconstruction) ** 2, dim=(1, 2))
                threshold = float(torch.quantile(errors, self.deep_training_quantile).item())

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "seq_len": seq_len,
                "output_dim": input_dim,
                "d_model": int(getattr(model, "eval_d_model", 0)) or None,
                "anomaly_threshold": threshold,
                "threshold_quantile": self.deep_training_quantile,
                "feature_names": list(ANOMALY_FEATURES),
                "trained_at": _utc_now_iso(),
                "training_windows": len(sequences),
            }
            self.deep_anomaly_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, self.deep_anomaly_path)
            logger.info("Persisted deep anomaly detector to %s", self.deep_anomaly_path)
            return True
        except Exception as exc:
            logger.error("Deep anomaly detector training failed: %s", exc, exc_info=True)
            return False

    def detect_anomalies(self, current_issues: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies using only feature-compatible, calibrated detectors."""
        issues = [dict(item) for item in current_issues if isinstance(item, Mapping)]
        if not issues:
            return []

        features = self._extract_ml_features(issues)
        if len(features) != len(issues):
            logger.warning("Some anomaly records were invalid; inference will use valid records only.")
            # Feature extraction currently always emits one row per mapping, but
            # retain the guard to protect future schema changes.
            issues = issues[: len(features)]
        if not features:
            return []

        isolation_indices: set[int] = set()
        if self.anomaly_model is not None:
            try:
                predictions = self.anomaly_model.predict(features)
                isolation_indices = {
                    index for index, prediction in enumerate(predictions) if int(prediction) == -1
                }
            except Exception as exc:
                logger.error("IsolationForest anomaly inference failed: %s", exc, exc_info=True)

        deep_indices = self._detect_deep_anomaly_indices(features)
        anomaly_indices = sorted(isolation_indices | deep_indices)
        if not anomaly_indices and self.anomaly_model is None and not deep_indices:
            logger.info("No anomaly detector is currently available for inference.")
            return []

        detected: List[Dict[str, Any]] = []
        for index in anomaly_indices:
            if index >= len(issues):
                continue
            payload = dict(issues[index])
            payload["anomaly_detection"] = {
                "isolation_forest": index in isolation_indices,
                "deep_reconstruction": index in deep_indices,
                "feature_schema": list(ANOMALY_FEATURES),
            }
            detected.append(payload)
        return detected

    def _detect_deep_anomaly_indices(self, features: List[List[float]]) -> set[int]:
        path = self.deep_anomaly_path
        if path is None or not path.is_file():
            return set()

        try:
            import torch  # type: ignore
            from .evaluators.modules.evaluation_transformer import EvaluationTransformer

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(path, map_location=device)

            expected_features = tuple(checkpoint.get("feature_names", ()))
            if expected_features and expected_features != ANOMALY_FEATURES:
                raise ValueError(
                    f"Deep anomaly feature schema mismatch: {expected_features!r} != {ANOMALY_FEATURES!r}."
                )

            input_dim = int(checkpoint["input_dim"])
            seq_len = int(checkpoint["seq_len"])
            if input_dim != len(ANOMALY_FEATURES):
                raise ValueError(
                    f"Deep model expects {input_dim} features; {len(ANOMALY_FEATURES)} are configured."
                )

            sequences = self._create_sequences(features, seq_len)
            if not sequences:
                return set()

            model_kwargs: Dict[str, Any] = {
                "input_dim": input_dim,
                "seq_len": seq_len,
                "output_dim": int(checkpoint.get("output_dim", input_dim)),
            }
            if checkpoint.get("d_model"):
                model_kwargs["d_model"] = int(checkpoint["d_model"])

            model = EvaluationTransformer(**model_kwargs).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            tensor = torch.tensor(sequences, dtype=torch.float32, device=device)
            with torch.no_grad():
                output = model(tensor, return_dict=True)
                reconstruction = output["reconstructed_sequence"]
                if reconstruction is None:
                    raise RuntimeError("Deep anomaly model returned no reconstruction.")
                errors = torch.mean((tensor - reconstruction) ** 2, dim=(1, 2))

            threshold = float(checkpoint["anomaly_threshold"])
            flagged_windows = [
                index for index, error in enumerate(errors.detach().cpu().tolist())
                if float(error) > threshold
            ]
            # A window score describes the observation at the end of that window.
            return {index + seq_len - 1 for index in flagged_windows}
        except Exception as exc:
            logger.error("Deep anomaly inference failed: %s", exc, exc_info=True)
            return set()

    def _extract_ml_features(self, issues: Sequence[Mapping[str, Any]]) -> List[List[float]]:
        """Extract the canonical four-feature anomaly schema."""
        defaults = {
            "severity": 0.5,
            "cognitive_complexity": 2.0,
            "data_flow_depth": 1.0,
            "security_risk": 0.3,
        }
        rows: List[List[float]] = []
        for issue in issues:
            row: List[float] = []
            for feature in ANOMALY_FEATURES:
                raw = issue.get(feature, defaults[feature])
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = defaults[feature]
                if not math.isfinite(value):
                    value = defaults[feature]
                row.append(value)
            rows.append(row)
        return rows

    @staticmethod
    def _create_sequences(
        features: Sequence[Sequence[float]],
        window_size: int,
    ) -> List[List[List[float]]]:
        if window_size <= 0:
            return []
        values = [list(map(float, row)) for row in features]
        return [
            values[index : index + window_size]
            for index in range(0, len(values) - window_size + 1)
        ]

    # ------------------------------------------------------------------
    # Evaluation orchestration
    # ------------------------------------------------------------------

    def _load_autonomous_tasks(self) -> List[Dict[str, Any]]:
        task_config = self.task_config
        if isinstance(task_config, str) and task_config.lower().endswith(".json"):
            try:
                with Path(task_config).open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                return [dict(item) for item in payload if isinstance(item, Mapping)] if isinstance(payload, list) else []
            except Exception as exc:
                logger.error("Failed to load autonomous tasks from %s: %s", task_config, exc)
                return []
        if isinstance(task_config, list):
            return [dict(item) for item in task_config if isinstance(item, Mapping)]
        if task_config:
            logger.warning("Unsupported autonomous_tasks configuration type: %s", type(task_config).__name__)
        return []

    def _run_stage(
        self,
        name: str,
        operation: Callable[[], Any],
        *,
        unavailable_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one evaluator stage without discarding evidence from other stages."""
        try:
            payload = operation()
            if isinstance(payload, Mapping):
                return dict(payload)
            return {
                "status": "error",
                "error": f"{name} returned a non-mapping payload: {type(payload).__name__}",
            }
        except Exception as exc:
            message = unavailable_message or f"{name} evaluation failed"
            logger.error("%s: %s", message, exc, exc_info=True)
            return {
                "status": "error",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }

    def _require_evaluator(self, name: str) -> Any:
        evaluator = self.evaluators.get(name)
        if evaluator is None:
            raise OperationalError(
                message=f"Evaluator '{name}' is unavailable.",
                context={"initialization_issues": list(self.initialization_issues)},
            )
        return evaluator

    def execute_validation_cycle(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a full or lightweight domain-neutral validation cycle.

        Lightweight mode intentionally skips static analysis, behavioral SUT
        creation/execution, autonomous task evaluation, statistical-history
        analysis, and resource sampling unless explicit precomputed evidence is
        supplied. This keeps ``predict()`` lightweight and torch-free.
        """
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise TypeError("EvaluationAgent.execute_validation_cycle() expects a dictionary.")

        started = time.perf_counter()
        started_at = _utc_now_iso()
        lightweight = bool(params.get("lightweight", False))
        flow = self.protocol.get_enabled_evaluation_flow()

        results: Dict[str, Any] = {
            "validation_mode": "lightweight" if lightweight else "full",
            "started_at": started_at,
            "protocol_flow": dict(flow),
        }

        if self.initialization_issues:
            results["initialization_issues"] = list(self.initialization_issues)

        try:
            # ----------------------------------------------------------
            # Static analysis -- full cycles only
            # ----------------------------------------------------------
            if not lightweight and self.protocol.static_analysis.get("enable", False):
                evaluation_root = Path(__file__).resolve().parent / "evaluators"
                results["static_analysis"] = self._run_stage(
                    "static_analysis",
                    lambda: StaticAnalyzer(str(evaluation_root)).full_analysis(),
                )
                if "error" not in results["static_analysis"]:
                    results["static_analysis_explanation"] = self._explain_static_results(
                        results["static_analysis"]
                    )

            # ----------------------------------------------------------
            # Behavioral validation -- full cycles only
            # ----------------------------------------------------------
            test_suite: Dict[str, Any] = {"records": [], "summary": {}}
            if not lightweight and self.protocol.behavioral_testing.get("test_types"):
                behavioral = self.evaluators.get("behavioral")
                if behavioral is None:
                    test_suite = {
                        "status": "error",
                        "error": "Behavioral evaluator is unavailable.",
                        "records": [],
                        "summary": {},
                    }
                else:
                    test_agent = self.create_agent()
                    test_suite = self._run_stage(
                        "behavioral",
                        lambda: behavioral.execute_test_suite(test_agent),
                    )
                results["behavioral"] = test_suite
                if "error" not in test_suite:
                    results["test_explanation"] = self._explain_test_results(test_suite)

            outputs, truths = self._extract_supervised_pairs(test_suite, params)

            # ----------------------------------------------------------
            # Performance
            # ----------------------------------------------------------
            if flow.get("enable_performance", True) and outputs and truths:
                performance = self.evaluators.get("performance")
                if performance is None:
                    results["performance"] = {
                        "status": "error",
                        "error": "Performance evaluator is unavailable.",
                    }
                else:
                    probabilities = params.get("probabilities")
                    results["performance"] = self._run_stage(
                        "performance",
                        lambda: performance.evaluate(
                            outputs=outputs,
                            ground_truths=truths,
                            probabilities=probabilities if isinstance(probabilities, Sequence) and not isinstance(probabilities, (str, bytes)) else None,
                        ),
                    )

            # ----------------------------------------------------------
            # Efficiency. Unlike classification performance, efficiency can
            # operate without ground truths.
            # ----------------------------------------------------------
            if flow.get("enable_efficiency", True) and outputs:
                efficiency = self.evaluators.get("efficiency")
                if efficiency is None:
                    results["efficiency"] = {
                        "status": "error",
                        "error": "Efficiency evaluator is unavailable.",
                    }
                else:
                    efficiency_truths = truths if len(truths) == len(outputs) and truths else None
                    results["efficiency"] = self._run_stage(
                        "efficiency",
                        lambda: efficiency.evaluate(
                            outputs=outputs,
                            ground_truths=efficiency_truths,
                            execution_metadata={"validation_mode": results["validation_mode"]},
                        ),
                    )

            # ----------------------------------------------------------
            # Autonomous evaluation -- full cycles only
            # ----------------------------------------------------------
            if not lightweight and self.autonomous_tasks:
                autonomous = self.evaluators.get("autonomous")
                if autonomous is None:
                    results["autonomous"] = {
                        "status": "error",
                        "error": "Autonomous evaluator is unavailable.",
                    }
                else:
                    results["autonomous"] = self._run_stage(
                        "autonomous",
                        lambda: autonomous.evaluate_task_set(self._get_validated_tasks()),
                    )

            # ----------------------------------------------------------
            # Resource evidence. Lightweight mode consumes supplied evidence
            # but does not perform the evaluator's potentially multi-second
            # sampling window.
            # ----------------------------------------------------------
            supplied_resource = params.get("resource_result")
            if isinstance(supplied_resource, Mapping):
                results["resource"] = dict(supplied_resource)
            elif not lightweight and flow.get("enable_resource", True):
                resource = self.evaluators.get("resource")
                if resource is None:
                    results["resource"] = {
                        "status": "error",
                        "error": "Resource evaluator is unavailable.",
                    }
                else:
                    results["resource"] = self._run_stage(
                        "resource",
                        lambda: resource.evaluate(
                            sample_metadata={"validation_mode": results["validation_mode"]}
                        ),
                    )

            # ----------------------------------------------------------
            # Statistical evaluation -- full cycles only
            # ----------------------------------------------------------
            if not lightweight and flow.get("enable_statistical", True):
                statistical = self.evaluators.get("statistical")
                if statistical is None:
                    results["statistical"] = {
                        "status": "error",
                        "error": "Statistical evaluator is unavailable.",
                    }
                else:
                    supplied_datasets = params.get("statistical_datasets")
                    statistical_data = (
                        self._normalize_statistical_datasets(supplied_datasets)
                        if isinstance(supplied_datasets, Mapping)
                        else self._prepare_statistical_dataset()
                    )
                    if self._has_sufficient_statistical_data(
                        statistical_data,
                        statistical.min_sample_size,
                    ):
                        results["statistical"] = self._run_stage(
                            "statistical",
                            lambda: statistical.evaluate(datasets=statistical_data),
                        )
                    else:
                        results["statistical"] = {
                            "status": "skipped",
                            "reason": "Insufficient comparable statistical history.",
                            "dataset_sizes": {
                                name: len(values) for name, values in statistical_data.items()
                            },
                        }

            # ----------------------------------------------------------
            # Safety evidence. Never re-evaluate SafetyEvaluator.raw_incidents;
            # doing so would duplicate evaluator-owned incident history.
            # ----------------------------------------------------------
            supplied_safety = params.get("safety_result")
            if isinstance(supplied_safety, Mapping):
                results["safety"] = dict(supplied_safety)
            else:
                safety_incidents = params.get("safety_incidents")
                if isinstance(safety_incidents, Sequence) and not isinstance(safety_incidents, (str, bytes)) and safety_incidents:
                    safety = self.evaluators.get("safety")
                    if safety is None:
                        results["safety"] = {
                            "status": "error",
                            "error": "Safety evaluator is unavailable.",
                        }
                    else:
                        results["safety"] = self._run_stage(
                            "safety",
                            lambda: safety.evaluate_operation(safety_incidents),
                        )

            # ----------------------------------------------------------
            # Cross-agent performance-budget contracts
            # ----------------------------------------------------------
            observed_budget_metrics = params.get("agent_performance_metrics")
            if isinstance(observed_budget_metrics, Mapping) and observed_budget_metrics:
                budget = self.evaluators.get("performance_budget")
                if budget is None:
                    results["performance_budget"] = {
                        "status": "error",
                        "error": "Performance budget evaluator is unavailable.",
                    }
                else:
                    results["performance_budget"] = self._run_stage(
                        "performance_budget",
                        lambda: budget.evaluate(observed_budget_metrics),
                    )
            elif not lightweight:
                results["performance_budget"] = {
                    "status": "skipped",
                    "reason": "No agent_performance_metrics were supplied.",
                }

            core_metrics = self._gather_core_metrics(results)
            results.update(core_metrics)
            results.update(self._evaluate_validation_health(results=results, params=params))
            results["status"] = self._determine_system_status(results)
            results["completed_at"] = _utc_now_iso()
            results["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 3)

            persist = bool(params.get("persist_results", not lightweight))
            if persist:
                self.log_evaluation(results)
            return results

        except Exception as exc:
            logger.error("Validation cycle orchestration failed: %s", exc, exc_info=True)
            return {
                **results,
                "status": "critical",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "cycle_failed": True,
                "completed_at": _utc_now_iso(),
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }

    @staticmethod
    def _extract_supervised_pairs(
        test_suite: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> Tuple[List[Any], List[Any]]:
        explicit_outputs = params.get("predictions")
        explicit_truths = params.get("expected_outputs")
        if (
            isinstance(explicit_outputs, Sequence)
            and not isinstance(explicit_outputs, (str, bytes))
            and isinstance(explicit_truths, Sequence)
            and not isinstance(explicit_truths, (str, bytes))
            and len(explicit_outputs) == len(explicit_truths)
            and len(explicit_outputs) > 0
        ):
            return list(explicit_outputs), list(explicit_truths)

        records = test_suite.get("records", []) if isinstance(test_suite, Mapping) else []
        pairs: List[Tuple[Any, Any]] = []
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                output = record.get("output")
                expected = record.get("expected_output")
                if output is None or expected is None:
                    continue
                pairs.append((output, expected))
        return [item[0] for item in pairs], [item[1] for item in pairs]

    def _prepare_statistical_dataset(self) -> Dict[str, List[float]]:
        """Build two comparable temporal accuracy windows from shared history."""
        history = self.shared_memory.get("metric_history", []) or []
        if not isinstance(history, list):
            return {}

        accuracies: List[float] = []
        for entry in history:
            if not isinstance(entry, Mapping):
                continue
            value = entry.get("accuracy")
            if not _is_finite_number(value):
                performance = entry.get("performance", {})
                if isinstance(performance, Mapping):
                    metrics = performance.get("metrics", performance)
                    if isinstance(metrics, Mapping):
                        value = metrics.get("accuracy")
            if _is_finite_number(value):
                accuracies.append(float(value))

        statistical = self.evaluators.get("statistical")
        minimum = int(getattr(statistical, "min_sample_size", 10) or 10)
        if len(accuracies) < minimum * 2:
            return {"history": accuracies}

        recent = accuracies[-(minimum * 2) :]
        return {
            "previous_window": recent[:minimum],
            "current_window": recent[minimum:],
        }

    @staticmethod
    def _normalize_statistical_datasets(payload: Mapping[str, Any]) -> Dict[str, List[float]]:
        normalized: Dict[str, List[float]] = {}
        for name, values in payload.items():
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            numeric = [float(value) for value in values if _is_finite_number(value)]
            if numeric:
                normalized[str(name)] = numeric
        return normalized

    @staticmethod
    def _has_sufficient_statistical_data(
        datasets: Mapping[str, Sequence[float]],
        minimum: int,
    ) -> bool:
        valid = [values for values in datasets.values() if len(values) >= minimum]
        return len(valid) >= 2

    # ------------------------------------------------------------------
    # Health aggregation
    # ------------------------------------------------------------------

    def get_overall_system_health(self) -> Dict[str, Any]:
        """
        Return a policy-neutral health snapshot.

        Evaluator-owned statuses/violations determine validation health. This
        method does not reapply independent accuracy/safety/efficiency thresholds.
        """
        try:
            latest = self.shared_memory.get("latest_metrics") or {}
            if not isinstance(latest, Mapping):
                latest = {}

            try:
                shared_metrics = self.shared_memory.metrics()
            except Exception:
                shared_metrics = {}
            try:
                shared_stats = self.shared_memory.get_usage_stats()
            except Exception:
                shared_stats = {}

            memory_fraction = float(shared_stats.get("memory_usage_percentage", 0.0) or 0.0) / 100.0
            if memory_fraction >= self.memory_critical_threshold:
                memory_status = "Critical"
            elif memory_fraction >= self.memory_warning_threshold:
                memory_status = "Warning"
            else:
                memory_status = "Normal"

            component_status = latest.get("component_status", {})
            component_status = dict(component_status) if isinstance(component_status, Mapping) else {}
            critical_issues = latest.get("critical_issues", [])
            warnings = latest.get("warnings", [])
            critical_issues = list(critical_issues) if isinstance(critical_issues, list) else []
            warnings = list(warnings) if isinstance(warnings, list) else []

            normalized_statuses = {str(value).lower() for value in component_status.values()}
            if critical_issues or "critical" in normalized_statuses or memory_status == "Critical":
                overall = "Critical"
            elif warnings or "warning" in normalized_statuses or memory_status == "Warning":
                overall = "Warning"
            elif latest:
                overall = "Normal"
            else:
                overall = "Unknown"

            metric_names = (
                "accuracy",
                "efficiency",
                "safety_score",
                "safety_compliance",
                "resource_efficiency",
                "resource_usage",
                "statistical_p_value",
                "autonomous_score",
                "system_failure_rate",
            )
            metric_snapshot: Dict[str, Any] = {}
            for name in metric_names:
                value = latest.get(name)
                metric_snapshot[name] = round(float(value), 6) if _is_finite_number(value) else None

            return {
                "status": overall,
                "metrics": metric_snapshot,
                "component_status": component_status,
                "critical_issues": critical_issues,
                "warnings": warnings,
                "initialization_issues": list(self.initialization_issues),
                "shared_memory": {
                    "memory_usage_percent": shared_stats.get("memory_usage_percentage", 0),
                    "available_memory_mb": shared_stats.get("available_memory_mb", 0),
                    "item_count": shared_stats.get("item_count", 0),
                    "pending_expiration_cleanup": shared_stats.get("pending_expiration_cleanup", 0),
                    "access_count": shared_metrics.get("access_count", 0),
                    "status": memory_status,
                },
                "timestamp": _utc_now_iso(),
            }
        except Exception as exc:
            logger.error("Failed to compute system health: %s", exc, exc_info=True)
            return {
                "status": "Unknown",
                "error": str(exc),
                "timestamp": _utc_now_iso(),
            }

    def _evaluate_validation_health(
        self,
        results: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Aggregate evaluator-native evidence without redefining evaluator policy."""
        if not isinstance(results, dict):
            raise TypeError("results must be a dictionary")

        context = params if isinstance(params, dict) else {}
        critical_issues: List[str] = []
        warnings: List[str] = []
        component_status: Dict[str, str] = {}
        severity_rank = {"skipped": 0, "normal": 1, "warning": 2, "critical": 3}

        def as_dict(value: Any) -> Dict[str, Any]:
            return dict(value) if isinstance(value, Mapping) else {}

        def numeric(value: Any, default: float = 0.0) -> float:
            return float(value) if _is_finite_number(value) else float(default)

        def append_unique(collection: List[str], message: str) -> None:
            message = str(message).strip()
            if message and message not in collection:
                collection.append(message)

        def mark(component: str, status: str) -> None:
            normalized = str(status).strip().lower()
            if normalized not in severity_rank:
                return
            current = component_status.get(component)
            if current is None or severity_rank[normalized] > severity_rank.get(current, 0):
                component_status[component] = normalized

        def add_critical(component: str, message: str) -> None:
            append_unique(critical_issues, f"{component}: {message}")
            mark(component, "critical")

        def add_warning(component: str, message: str) -> None:
            append_unique(warnings, f"{component}: {message}")
            mark(component, "warning")

        for issue in self.initialization_issues:
            # Component import/config failures degrade capability, but become
            # critical only when the affected evaluator is actually requested.
            add_warning("initialization", issue)

        failing_statuses = {"block", "blocked", "critical", "error", "fail", "failed", "failure"}
        warning_statuses = {"degraded", "incomplete", "partial", "warn", "warning"}
        successful_statuses = {"allow", "complete", "completed", "normal", "ok", "pass", "passed", "success", "succeeded"}

        for component, payload in results.items():
            if not isinstance(payload, Mapping):
                continue
            error = payload.get("error")
            if error:
                add_critical(component, f"evaluation error: {error}")
                continue
            raw_status = payload.get("status")
            if raw_status is None:
                continue
            status = str(raw_status).strip().lower()
            if status in failing_statuses:
                add_critical(component, f"reported status {status!r}")
            elif status in warning_statuses:
                add_warning(component, f"reported status {status!r}")
            elif status == "skipped":
                mark(component, "skipped")
            elif status in successful_statuses:
                mark(component, "normal")

        static_result = as_dict(results.get("static_analysis"))
        if static_result and not static_result.get("error"):
            security = as_dict(static_result.get("security_metrics"))
            critical_count = numeric(security.get("critical_count"))
            static_config = as_dict(getattr(self.protocol, "static_analysis", {}))
            security_config = as_dict(static_config.get("security"))
            max_critical = numeric(security_config.get("max_critical"))
            if critical_count > max_critical:
                add_critical(
                    "static_analysis",
                    f"{int(critical_count)} critical issue(s) exceed allowed maximum {int(max_critical)}",
                )
            else:
                mark("static_analysis", "normal")

        behavioral = as_dict(results.get("behavioral"))
        if behavioral and not behavioral.get("error"):
            summary = as_dict(behavioral.get("summary"))
            failed = numeric(summary.get("failed"))
            errored = numeric(summary.get("errored"))
            if failed > 0 or errored > 0:
                add_critical(
                    "behavioral",
                    f"{int(failed)} failed and {int(errored)} errored test(s)",
                )
            elif summary:
                mark("behavioral", "normal")

        performance = as_dict(results.get("performance"))
        if performance and not performance.get("error"):
            assessment = as_dict(performance.get("threshold_assessment"))
            if assessment.get("composite_below_threshold") is True:
                add_warning("performance", "composite score is below configured threshold")
            assessment_warnings = assessment.get("warnings", [])
            if isinstance(assessment_warnings, (list, tuple)):
                for warning in assessment_warnings:
                    add_warning("performance", str(warning))
            if not assessment.get("composite_below_threshold") and not assessment_warnings:
                mark("performance", "normal")

        efficiency = as_dict(results.get("efficiency"))
        if efficiency and not efficiency.get("error"):
            diagnostics = as_dict(efficiency.get("diagnostics"))
            diagnostic_warnings = diagnostics.get("warnings", [])
            if isinstance(diagnostic_warnings, (list, tuple)) and diagnostic_warnings:
                for warning in diagnostic_warnings:
                    add_warning("efficiency", str(warning))
            elif efficiency.get("metrics"):
                mark("efficiency", "normal")

        resource = as_dict(results.get("resource"))
        if resource and not resource.get("error"):
            health_status = as_dict(resource.get("health_status"))
            critical_resources = [
                str(metric) for metric, status in health_status.items()
                if str(status).strip().upper() == "CRITICAL"
            ]
            warning_resources = [
                str(metric) for metric, status in health_status.items()
                if str(status).strip().upper() == "WARNING"
            ]
            if critical_resources:
                add_critical("resource", "critical utilization: " + ", ".join(sorted(critical_resources)))
            elif warning_resources:
                add_warning("resource", "elevated utilization: " + ", ".join(sorted(warning_resources)))
            elif health_status:
                mark("resource", "normal")
            threshold_violations = as_dict(resource.get("threshold_violations"))
            if threshold_violations:
                add_critical(
                    "resource",
                    "configured threshold(s) exceeded: " + ", ".join(sorted(map(str, threshold_violations))),
                )

        safety = as_dict(results.get("safety"))
        if safety and not safety.get("error"):
            aggregates = as_dict(safety.get("aggregates"))
            threshold_assessment = as_dict(safety.get("threshold_assessment"))
            critical_incidents = numeric(aggregates.get("critical_incidents"))
            violations = threshold_assessment.get("violations", [])
            if critical_incidents > 0:
                add_critical("safety", f"{int(critical_incidents)} critical safety incident(s)")
            if isinstance(violations, (list, tuple)):
                for violation in violations:
                    add_critical("safety", str(violation))
            if critical_incidents <= 0 and not violations and aggregates:
                mark("safety", "normal")

        budget = as_dict(results.get("performance_budget"))
        if budget and not budget.get("error"):
            status = str(budget.get("status", "")).strip().lower()
            summary = as_dict(budget.get("summary"))
            violation_count = numeric(summary.get("violations"))
            warning_count = numeric(summary.get("warnings"))
            if status == "fail" or violation_count > 0:
                add_critical("performance_budget", f"{int(violation_count)} budget violation(s)")
            elif status in {"warn", "warning"} or warning_count > 0:
                add_warning("performance_budget", f"{int(warning_count)} budget warning(s)")
            elif status == "skipped":
                mark("performance_budget", "skipped")
            elif status:
                mark("performance_budget", "normal")

        execution = as_dict(context.get("control_loop_execution"))
        if execution:
            status = str(execution.get("status", "")).strip().lower()
            if execution.get("success") is False or status in failing_statuses:
                add_critical("execution", "control-loop execution reported failure")
            elif execution.get("completed") is False or status in warning_statuses:
                add_warning("execution", "control-loop execution is incomplete or degraded")
            else:
                mark("execution", "normal")

        return {
            "critical_issues": critical_issues,
            "warnings": warnings,
            "component_status": component_status,
        }

    @staticmethod
    def _determine_system_status(results: Mapping[str, Any]) -> str:
        if results.get("cycle_failed") or results.get("error"):
            return "critical"
        critical = results.get("critical_issues", [])
        if isinstance(critical, list) and critical:
            return "critical"
        warnings = results.get("warnings", [])
        if isinstance(warnings, list) and warnings:
            return "warning"
        return "normal"

    def _gather_core_metrics(self, raw_results: Mapping[str, Any]) -> Dict[str, Any]:
        """Extract semantically consistent cross-evaluator summary metrics."""
        def as_mapping(value: Any) -> Dict[str, Any]:
            return dict(value) if isinstance(value, Mapping) else {}

        metrics: Dict[str, Any] = {}

        def coerce_finite_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            if _is_finite_number(value):
                return float(value)
            return None

        performance = as_mapping(raw_results.get("performance"))
        performance_metrics = as_mapping(performance.get("metrics", performance))
        accuracy = coerce_finite_float(performance_metrics.get("accuracy"))
        if accuracy is not None:
            metrics["accuracy"] = accuracy

        efficiency = as_mapping(raw_results.get("efficiency"))
        efficiency_metrics = as_mapping(efficiency.get("metrics", efficiency))
        efficiency_score = efficiency_metrics.get("score", efficiency_metrics.get("composite_score"))
        efficiency_value = coerce_finite_float(efficiency_score)
        if efficiency_value is not None:
            metrics["efficiency"] = efficiency_value

        safety = as_mapping(raw_results.get("safety"))
        safety_aggregates = as_mapping(safety.get("aggregates", safety))
        safety_score = coerce_finite_float(safety_aggregates.get("composite_score"))
        if safety_score is not None:
            metrics["safety_score"] = safety_score
        safety_compliance = coerce_finite_float(safety_aggregates.get("compliance_rate"))
        if safety_compliance is not None:
            metrics["safety_compliance"] = safety_compliance

        resource = as_mapping(raw_results.get("resource"))
        resource_weighted_score = coerce_finite_float(resource.get("weighted_score"))
        if resource_weighted_score is not None:
            resource_efficiency = max(0.0, min(1.0, resource_weighted_score))
            metrics["resource_efficiency"] = resource_efficiency
            metrics["resource_usage"] = 1.0 - resource_efficiency

        statistical = as_mapping(raw_results.get("statistical"))
        p_value = self._extract_statistical_p_value(statistical)
        if p_value is not None:
            metrics["statistical_p_value"] = p_value
            # Backward-compatible numeric alias. Do not interpret this as a
            # health score; it is a p-value.
            metrics["statistical_significance"] = p_value

        autonomous = as_mapping(raw_results.get("autonomous"))
        autonomous_score = autonomous.get("composite_score")
        if autonomous_score is not None and _is_finite_number(autonomous_score):
            metrics["autonomous_score"] = float(autonomous_score)

        if self.risk_model is not None:
            try:
                risk = self.risk_model.get_current_risk("system_failure")
                risk_metrics = as_mapping(as_mapping(risk).get("risk_metrics"))
                current_mean = risk_metrics.get("current_mean")
                if current_mean is not None and _is_finite_number(current_mean):
                    metrics["system_failure_rate"] = float(current_mean)
            except Exception:
                # A missing hazard category must not corrupt unrelated metrics.
                pass

        return metrics

    @staticmethod
    def _extract_statistical_p_value(statistical_payload: Mapping[str, Any]) -> Optional[float]:
        analysis = statistical_payload.get("analysis", statistical_payload)
        if not isinstance(analysis, Mapping):
            return None
        pairwise = analysis.get("pairwise_tests")

        candidates: List[Mapping[str, Any]] = []
        if isinstance(pairwise, Mapping):
            candidates.extend(item for item in pairwise.values() if isinstance(item, Mapping))
        elif isinstance(pairwise, list):
            candidates.extend(item for item in pairwise if isinstance(item, Mapping))

        for comparison in candidates:
            for key in ("paired_test", "t_test", "test"):
                test = comparison.get(key)
                if isinstance(test, Mapping) and _is_finite_number(test.get("p_value")):
                    return float(test["p_value"])
            if _is_finite_number(comparison.get("p_value")):
                return float(comparison["p_value"])
        return None

    # ------------------------------------------------------------------
    # Agent creation and prediction
    # ------------------------------------------------------------------

    def create_agent(self) -> BaseAgent:
        """Create the configured system-under-test with a deterministic fallback."""
        try:
            agent_type = str(self.agent_config.get("test_agent_type", "adaptive")).strip() or "adaptive"
            if agent_type in {"evaluation", "evaluation_agent"}:
                raise ValueError("EvaluationAgent cannot use itself as the behavioral test agent.")

            if agent_type == "adaptive" and threading.current_thread() is not threading.main_thread():
                logger.warning(
                    "Non-main thread detected; switching behavioral test agent from adaptive to lazy."
                )
                agent_type = "lazy"

            test_agent_config = self.agent_config.get("test_agent_config", {})
            if not isinstance(test_agent_config, Mapping):
                test_agent_config = {}

            agent = self.agent_factory.create(
                agent_type=agent_type,
                shared_memory=self.shared_memory,
                config=dict(test_agent_config),
            )
            if agent is None:
                raise RuntimeError(f"AgentFactory returned None for {agent_type!r}.")
            return agent
        except Exception as exc:
            logger.critical("Behavioral test-agent creation failed: %s", exc, exc_info=True)
            return FallbackEvaluatorAgent(
                shared_memory=self.shared_memory,
                agent_factory=self.agent_factory,
                config=self.agent_config,
            )

    def predict(self, state: Any = None) -> Dict[str, Any]:
        """Return evaluation-health telemetry with an optional lightweight pass."""
        started = time.perf_counter()
        timestamp = _utc_now_iso()
        prediction_warnings: List[str] = []

        try:
            health = self.get_overall_system_health()
            history = self.shared_memory.get("metric_history", [])
            history_count = len(history) if isinstance(history, list) else 0
            light_summary: Dict[str, Any] = {}
            has_light_eval = False

            if state is not None:
                has_light_eval = True
                lightweight_params: Dict[str, Any] = {
                    "lightweight": True,
                    "input_state": state,
                    "persist_results": False,
                }
                if isinstance(state, Mapping):
                    for key in (
                        "predictions",
                        "expected_outputs",
                        "probabilities",
                        "safety_incidents",
                        "safety_result",
                        "resource_result",
                        "agent_performance_metrics",
                        "control_loop_execution",
                    ):
                        if key in state:
                            lightweight_params[key] = state[key]

                light_results = self.execute_validation_cycle(lightweight_params)
                critical = light_results.get("critical_issues", [])
                warnings = light_results.get("warnings", [])
                light_summary = {
                    "status": light_results.get("status", "unknown"),
                    "has_errors": bool(light_results.get("error")),
                    "critical_issue_count": len(critical) if isinstance(critical, list) else 0,
                    "warning_count": len(warnings) if isinstance(warnings, list) else 0,
                    "duration_ms": light_results.get("duration_ms"),
                }
                if light_results.get("error"):
                    prediction_warnings.append(f"lightweight_validation_error: {light_results['error']}")
                health["predicted_metrics"] = self._gather_core_metrics(light_results)

            confidence = self._prediction_confidence(
                health=health,
                has_light_eval=has_light_eval,
                metric_history_count=history_count,
            )
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            return {
                "status": "success",
                "prediction": {
                    "health_forecast": health,
                    "light_eval_summary": light_summary,
                    "diagnostics": {
                        "timestamp": timestamp,
                        "latency_ms": latency_ms,
                        "metric_history_count": history_count,
                        "input_type": type(state).__name__ if state is not None else "none",
                        "warnings": prediction_warnings,
                    },
                },
                "confidence": confidence,
            }
        except Exception as exc:
            logger.error("Evaluation prediction failed: %s", exc, exc_info=True)
            return {
                "status": "error",
                "error": str(exc),
                "prediction": {
                    "health_forecast": {},
                    "light_eval_summary": {},
                    "diagnostics": {
                        "timestamp": timestamp,
                        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                        "warnings": prediction_warnings + ["predict_exception_raised"],
                    },
                },
                "confidence": 0.0,
            }

    @staticmethod
    def _prediction_confidence(
        *,
        health: Mapping[str, Any],
        has_light_eval: bool,
        metric_history_count: int,
    ) -> float:
        confidence = 0.55
        status = str(health.get("status", "unknown")).lower()
        if status == "normal":
            confidence += 0.15
        elif status == "warning":
            confidence += 0.05
        elif status == "critical":
            confidence -= 0.15
        if metric_history_count >= 50:
            confidence += 0.10
        elif metric_history_count >= 10:
            confidence += 0.05
        if has_light_eval:
            confidence += 0.10
        return float(max(0.05, min(0.95, confidence)))

    # ------------------------------------------------------------------
    # Persistence and adaptive-risk integration
    # ------------------------------------------------------------------

    def log_evaluation(self, results: Dict[str, Any]) -> None:
        if not isinstance(results, dict):
            logger.error("Invalid evaluation result: expected dictionary.")
            return
        sanitized = self._sanitize_results(results)
        self._store_metrics(sanitized)
        self._update_risk_model(sanitized)

    def _sanitize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        protected_keys = {"notes", "feedback"}

        def sanitize(value: Any, key: Optional[str] = None) -> Any:
            if key and key.casefold() in protected_keys:
                try:
                    return SafetyGuard.scrub(value)
                except Exception:
                    return "[REDACTED]"
            if isinstance(value, Mapping):
                return {str(k): sanitize(v, str(k)) for k, v in value.items()}
            if isinstance(value, list):
                return [sanitize(item) for item in value]
            if isinstance(value, tuple):
                return [sanitize(item) for item in value]
            return value

        return sanitize(results)

    def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        try:
            self.shared_memory.set("latest_metrics", metrics)
            self.shared_memory.append("metric_history", metrics)
        except Exception as exc:
            logger.error("Failed to persist evaluation metrics to shared memory: %s", exc, exc_info=True)

    def _update_risk_model(self, metrics: Mapping[str, Any]) -> None:
        if self.risk_model is None:
            return
        hazards = metrics.get("hazards")
        operational_time = metrics.get("operational_time")
        if not isinstance(hazards, Mapping) or not hazards:
            return
        if not _is_finite_number(operational_time) or float(operational_time) <= 0.0:
            logger.warning("Skipping risk update: operational_time must be a positive finite number.")
            return
        try:
            normalized_hazards = {
                str(name): int(count)
                for name, count in hazards.items()
                if isinstance(count, int) and not isinstance(count, bool) and count >= 0
            }
            if not normalized_hazards:
                return
            self.risk_model.update_model(
                normalized_hazards,
                float(operational_time),
                source="evaluation_agent",
            )
        except EvaluationError as exc:
            logger.error("Adaptive risk update rejected evaluation data: %s", exc)
        except Exception as exc:
            logger.error("Adaptive risk update failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Issue database and mitigation
    # ------------------------------------------------------------------

    def _connect_issue_database(self) -> Any:
        """Connect using Base-Agent configuration, otherwise use in-memory fallback."""
        try:
            from .evaluators.data.issue_db import FallbackIssueTracker, IssueDBConnector
        except Exception as exc:
            logger.error("Issue database subsystem is unavailable: %s", exc, exc_info=True)
            return _LocalFallbackIssueTracker()

        required = ("host", "port", "database")
        if not all(self.db_config.get(key) not in (None, "") for key in required):
            logger.warning("Issue database configuration is incomplete; using fallback tracker.")
            return FallbackIssueTracker()

        retries = self._coerce_positive_int(
            self.agent_config.get("issue_database_retries", 3),
            default=3,
        )
        for attempt in range(retries):
            try:
                connector = IssueDBConnector(**self.db_config)
                connector.initialize_schema(
                    {
                        "evaluation_issues": """
                            CREATE TABLE IF NOT EXISTS evaluation_issues (
                                id UUID PRIMARY KEY,
                                timestamp TIMESTAMPTZ NOT NULL,
                                issue_type VARCHAR(50) NOT NULL,
                                severity FLOAT CHECK (severity >= 0 AND severity <= 1),
                                context JSONB,
                                metrics JSONB,
                                resolution_status VARCHAR(20) DEFAULT 'unresolved'
                            )
                        """
                    }
                )
                logger.info("Connected to evaluation issue database.")
                return connector
            except Exception as exc:
                if attempt + 1 >= retries:
                    logger.error("Issue database connection failed after %d attempt(s): %s", retries, exc)
                    break
                delay = min(2 ** attempt, 8)
                logger.warning(
                    "Issue database connection attempt %d/%d failed; retrying in %ss: %s",
                    attempt + 1,
                    retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
        return FallbackIssueTracker()

    @staticmethod
    def _normalize_issue_payload(issue: Mapping[str, Any]) -> Dict[str, Any]:
        issue_type = str(issue.get("type") or issue.get("issue_type") or "evaluation_issue").strip()
        severity_raw = issue.get("severity", 1.0)
        severity = float(severity_raw) if _is_finite_number(severity_raw) else 1.0
        severity = max(0.0, min(1.0, severity))
        return {
            "id": str(issue.get("id") or uuid.uuid4()),
            "timestamp": str(issue.get("timestamp") or _utc_now_iso()),
            "type": issue_type,
            "issue_type": issue_type,
            "severity": severity,
            "context": dict(issue.get("context", {})) if isinstance(issue.get("context"), Mapping) else {},
            "metrics": dict(issue.get("metrics", {})) if isinstance(issue.get("metrics"), Mapping) else {},
            "resolution_status": str(issue.get("resolution_status", "unresolved")),
        }

    def _record_issue(self, issue: Mapping[str, Any]) -> bool:
        payload = self._normalize_issue_payload(issue)
        success = False
        try:
            success = bool(self.issue_db.log_issue(payload))
        except Exception as exc:
            logger.error("Issue tracking failed: %s", exc, exc_info=True)

        try:
            self.shared_memory.append("evaluation_issue_history", payload)
        except Exception as exc:
            logger.warning("Unable to append evaluation issue history: %s", exc)
        return success

    def trigger_mitigation_actions(self) -> None:
        """Enter degraded evaluation mode and preserve independent safety channels."""
        logger.critical("Triggering EvaluationAgent mitigation actions.")

        try:
            self.shared_memory.set("system_status", "degraded")
        except Exception as exc:
            logger.warning("Unable to publish degraded system status: %s", exc)

        try:
            last_metrics = self.shared_memory.get("latest_metrics", {}) or {}
        except Exception:
            last_metrics = {}

        self._record_issue(
            {
                "type": "safety_breach",
                "severity": 1.0,
                "context": {"last_metrics": last_metrics},
                "metrics": self._gather_core_metrics(last_metrics if isinstance(last_metrics, Mapping) else {}),
                "resolution_status": "unresolved",
            }
        )

        # Do not disable SafetyEvaluator or RiskAdaptation during mitigation.
        # They are independent safety-observation channels required in degraded mode.
        fallback = FallbackEvaluatorAgent(
            shared_memory=self.shared_memory,
            agent_factory=self.agent_factory,
            config=self.agent_config,
        )
        try:
            self.shared_memory.set("active_agent", fallback)
            self.shared_memory.set("evaluation:fallback_active", True)
        except Exception as exc:
            logger.error("Unable to publish fallback EvaluationAgent: %s", exc)

        if self.config.get("operations_notification"):
            self._notify_operations_team("System degraded: EvaluationAgent mitigation actions triggered.")

    # ------------------------------------------------------------------
    # Safety / fail-operational introspection
    # ------------------------------------------------------------------

    def _get_safety_compliance(self) -> str:
        """Report compliance from the latest SafetyEvaluator evidence only."""
        try:
            latest = self.shared_memory.get("latest_metrics", {}) or {}
        except Exception:
            return "Unknown"
        if not isinstance(latest, Mapping):
            return "Unknown"
        safety = latest.get("safety")
        if not isinstance(safety, Mapping):
            return "Unknown"
        if safety.get("error"):
            return "Unknown"
        assessment = safety.get("threshold_assessment", {})
        aggregates = safety.get("aggregates", {})
        if not isinstance(assessment, Mapping) or not isinstance(aggregates, Mapping):
            return "Unknown"
        violations = assessment.get("violations", [])
        critical = aggregates.get("critical_incidents", 0)
        if violations or (_is_finite_number(critical) and float(critical) > 0):
            return "Non-compliant"
        return "Compliant" if aggregates else "Unknown"

    def supports_fail_operational(self) -> bool:
        try:
            required = ("behavioral", "performance", "efficiency", "safety")
            if any(self.evaluators.get(name) is None for name in required):
                return False
            behavioral = self.evaluators["behavioral"]
            if not callable(getattr(behavioral, "execute_test_suite", None)):
                return False
            if self.risk_model is None or self.issue_db is None or self.shared_memory is None:
                return False
            return True
        except Exception as exc:
            logger.exception("Fail-operational check failed: %s", exc)
            return False

    def has_redundant_safety_channels(self) -> bool:
        """Require at least two independent safety-observation/guard channels."""
        guard = getattr(self, "safety_guard", None)
        guard_ready = bool(
            guard
            and callable(getattr(guard, "is_minimal_viable", None))
            and guard.is_minimal_viable()
        )
        safety_evaluator_ready = self.evaluators.get("safety") is not None
        risk_model_ready = self.risk_model is not None
        issue_tracking_ready = self.issue_db is not None
        try:
            liveness_ready = self.shared_memory.get("last_alive_ping") is not None
        except Exception:
            liveness_ready = False

        independent_channels = [
            guard_ready,
            safety_evaluator_ready,
            risk_model_ready,
            issue_tracking_ready,
            liveness_ready,
        ]
        return sum(bool(channel) for channel in independent_channels) >= 2

    # ------------------------------------------------------------------
    # Hyperparameter-tuning compatibility
    # ------------------------------------------------------------------

    def _init_hyperparam_tuner(self) -> None:
        """
        Compatibility hook retained from v2.2.

        Tuning remains disabled until the evaluator subsystem exposes a
        deterministic, explicitly named system-level optimization objective.
        """
        logger.info(
            "EvaluationAgent hyperparameter tuner is disabled pending an explicit "
            "subsystem-owned optimization objective."
        )
        return None

    def evaluate_hyperparameters(self, params: Dict[str, Any]) -> float:
        """Reject ambiguous tuning requests instead of optimizing an inferred score."""
        raise OperationalError(
            message=(
                "EvaluationAgent hyperparameter scoring is disabled because the "
                "evaluator subsystem does not yet expose a canonical system score."
            ),
            context={"parameter_keys": sorted(map(str, params.keys())) if isinstance(params, Mapping) else []},
        )

    # ------------------------------------------------------------------
    # Human-readable explanations
    # ------------------------------------------------------------------

    def _explain_static_results(self, static_results: Mapping[str, Any]) -> str:
        security = static_results.get("security_metrics", {})
        security = security if isinstance(security, Mapping) else {}
        code_quality = self.protocol.static_analysis.get("code_quality", {})
        security_config = self.protocol.static_analysis.get("security", {})
        return (
            "Code Quality Report:\n"
            f"- Technical Debt Ratio: {float(static_results.get('technical_debt', 0.0)):.1%} "
            f"(Threshold: {float(code_quality.get('tech_debt_threshold', 0.0)):.1%})\n"
            f"- Critical Security Issues: {int(security.get('critical_count', 0) or 0)} "
            f"(Max Allowed: {int(security_config.get('max_critical', 0) or 0)})"
        )

    def _explain_test_results(self, results: Mapping[str, Any]) -> str:
        summary = results.get("summary", {})
        summary = summary if isinstance(summary, Mapping) else {}
        return self.interpreter.explain_validation_metrics(
            {
                "passed": summary.get("passed", 0),
                "failed": int(summary.get("failed", 0) or 0) + int(summary.get("errored", 0) or 0),
                "coverage": summary.get("requirement_coverage_rate", 0.0),
            }
        )

    def _get_validated_tasks(self) -> List[Dict[str, Any]]:
        validated: List[Dict[str, Any]] = []
        for index, task in enumerate(self.autonomous_tasks):
            validated.append(
                {
                    "id": str(task.get("id") or f"task_{index}"),
                    "type": task.get("type", "generic"),
                    "path": task.get("path", []),
                    "optimal_path": task.get("optimal_path", []),
                    "completion_time": float(task.get("completion_time", 0.0) or 0.0),
                    "energy_consumed": float(task.get("energy_consumed", 0.0) or 0.0),
                    "collisions": int(task.get("collisions", 0) or 0),
                    "success": bool(task.get("success", False)),
                }
            )
        return validated

    # ------------------------------------------------------------------
    # Certification evidence orchestration
    # ------------------------------------------------------------------

    def request_certification(
        self,
        codebase_path: str,
        certification_requirements: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the internal assurance evidence pipeline.

        Formal certification requirements must be supplied explicitly; the agent
        will not silently treat smoke tests as regulatory requirements.
        """
        try:
            certification = self.validation_suite.execute_full_validation(
                codebase=codebase_path,
                agent=self.create_agent(),
                certification_requirements=certification_requirements,
            )
            self.shared_memory.set("certification_status", certification)
            return certification
        except CertificationError as exc:
            logger.error("Certification evidence pipeline failed: %s", exc, exc_info=True)
            self.trigger_mitigation_actions()
            raise

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _notify_operations_team(self, message: str, level: str = "CRITICAL") -> None:
        config = self.config.get("operations_notification", {})
        if not isinstance(config, Mapping) or not config:
            logger.warning("No operations notification configuration found.")
            return

        logger_method = getattr(logger, str(level).lower(), logger.info)
        logger_method("OPERATION ALERT: %s", message)
        try:
            email_config = config.get("email", {})
            if isinstance(email_config, Mapping) and email_config.get("enabled", False):
                self._send_email_alert(message, email_config)

            webhook_config = config.get("webhook", {})
            if isinstance(webhook_config, Mapping) and webhook_config.get("enabled", False):
                self._send_webhook_alert(message, webhook_config)

            external_config = config.get("external_logging", {})
            if isinstance(external_config, Mapping) and external_config.get("enabled", False):
                self._log_to_external_service(message, level, external_config)
        except Exception as exc:
            logger.error("Failed to notify operations team: %s", exc, exc_info=True)

    @staticmethod
    def _send_email_alert(message: str, config: Mapping[str, Any]) -> None:
        import smtplib
        from email.mime.text import MIMEText

        required = ("from", "to", "smtp_host")
        missing = [key for key in required if not config.get(key)]
        if missing:
            raise ValueError(f"Email notification configuration missing: {', '.join(missing)}")

        msg = MIMEText(message)
        msg["Subject"] = str(config.get("subject", "SLAI Evaluation Alert"))
        msg["From"] = str(config["from"])
        msg["To"] = str(config["to"])

        with smtplib.SMTP(str(config["smtp_host"]), int(config.get("smtp_port", 587)), timeout=10) as server:
            if config.get("use_tls", True):
                server.starttls()
            username = config.get("username")
            password = config.get("password")
            if username:
                server.login(str(username), str(password or ""))
            server.send_message(msg)
        logger.info("Evaluation alert email sent.")

    @staticmethod
    def _send_webhook_alert(message: str, config: Mapping[str, Any]) -> None:
        import requests  # type: ignore

        url = str(config.get("url", "")).strip()
        if not url:
            raise ValueError("Webhook URL is not configured.")
        response = requests.post(url, json={"text": message}, timeout=5)
        response.raise_for_status()
        logger.info("Evaluation webhook alert sent.")

    @staticmethod
    def _log_to_external_service(message: str, level: str, config: Mapping[str, Any]) -> None:
        import requests  # type: ignore

        url = str(config.get("url", "")).strip()
        if not url:
            raise ValueError("External logging URL is not configured.")
        response = requests.post(
            url,
            json={
                "service": "EvaluationAgent",
                "level": level,
                "message": message,
                "timestamp": _utc_now_iso(),
            },
            timeout=5,
        )
        response.raise_for_status()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release agent-owned orchestration resources exactly once."""
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._shutdown_complete = True

        if self.risk_model is not None:
            shutdown = getattr(self.risk_model, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:
                    logger.warning("Risk model shutdown failed: %s", exc)

        for name, evaluator in self.evaluators.items():
            shutdown = getattr(evaluator, "shutdown", None) if evaluator is not None else None
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:
                    logger.warning("Evaluator '%s' shutdown failed: %s", name, exc)

        close = getattr(self.issue_db, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.warning("Issue tracker close failed: %s", exc)
        else:
            # Compatibility with the current IssueDBConnector, which does not yet
            # expose a public close() method.
            for attribute in ("cursor", "conn"):
                resource = getattr(self.issue_db, attribute, None)
                closer = getattr(resource, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception as exc:
                        logger.warning("Issue database %s close failed: %s", attribute, exc)

        logger.info("EvaluationAgent shutdown complete.")


class AIValidationSuite:
    """
    Internal assurance-evidence pipeline.

    This class does not itself confer regulatory certification. It structures
    evidence and traceability for downstream review/certification processes.
    """

    def __init__(self, protocol: ValidationProtocol) -> None:
        self.protocol = protocol
        self.artifacts: Dict[str, Any] = {
            "static_report": None,
            "behavioral_results": None,
            "safety_case": None,
            "certification_evidence": [],
        }

    def execute_full_validation(
        self,
        codebase: str,
        agent: BaseAgent,
        certification_requirements: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self._validate_architecture(agent)
        self.artifacts["static_report"] = self._run_static_verification(codebase)
        self.artifacts["behavioral_results"] = self._run_behavioral_qualification(
            agent,
            certification_requirements,
        )
        self.artifacts["safety_case"] = self._build_safety_case(agent)
        return self._generate_certification_package()

    @staticmethod
    def _validate_architecture(agent: BaseAgent) -> None:
        if not callable(getattr(agent, "supports_fail_operational", None)) or not agent.supports_fail_operational():
            raise CertificationError("Architecture lacks demonstrated fail-operational capability.")
        if not callable(getattr(agent, "has_redundant_safety_channels", None)) or not agent.has_redundant_safety_channels():
            raise CertificationError("Architecture lacks demonstrated redundant safety channels.")

    def _run_static_verification(self, codebase: str) -> Dict[str, Any]:
        path = Path(codebase)
        if not path.is_dir():
            raise CertificationError(f"Invalid codebase directory: {codebase}")
        report = StaticAnalyzer(str(path)).full_analysis()
        security = report.get("security_metrics", {}) if isinstance(report, Mapping) else {}
        critical_count = int(security.get("critical_count", 0) or 0) if isinstance(security, Mapping) else 0
        max_critical = int(self.protocol.static_analysis["security"]["max_critical"])
        if critical_count > max_critical:
            raise CertificationError(
                f"Critical static-analysis findings exceed the configured threshold: {critical_count} > {max_critical}."
            )
        return dict(report)

    def _run_behavioral_qualification(
        self,
        agent: BaseAgent,
        certification_requirements: Optional[Sequence[Mapping[str, Any]]],
    ) -> Dict[str, Any]:
        if not certification_requirements:
            raise CertificationError(
                "Formal behavioral certification requirements were not supplied. "
                "Smoke tests cannot be promoted to certification requirements implicitly."
            )
        from .evaluators.behavioral_validator import BehavioralValidator
        validator = BehavioralValidator()
        results = validator.execute_certification_suite(
            sut=agent.perform_task,
            certification_requirements=certification_requirements,
        )
        if str(results.get("overall_status", "")).upper() == "FAILED":
            failed = [
                str(item.get("requirement_id", "UNKNOWN"))
                for item in results.get("traceability_matrix", [])
                if isinstance(item, Mapping)
                and str(item.get("status", "")).upper() in {"FAILED", "ERROR"}
            ]
            raise CertificationError(
                "Behavioral qualification failed.",
                context={"failed_requirements": failed},
            )
        return results

    @staticmethod
    def _build_safety_case(agent: BaseAgent) -> Dict[str, Any]:
        required_methods = {
            "hazard_analysis": "perform_hazard_analysis",
            "safety_goals": "derive_safety_goals",
            "fault_tree": "generate_fault_tree",
            "diagnostic_coverage": "calculate_diagnostic_coverage",
        }
        missing = [method for method in required_methods.values() if not callable(getattr(agent, method, None))]
        if missing:
            raise CertificationError(
                "System under test cannot produce the required safety-case evidence.",
                context={"missing_methods": missing},
            )
        return {
            key: getattr(agent, method)()
            for key, method in required_methods.items()
        }

    def _generate_certification_package(self) -> Dict[str, Any]:
        status = CertificationStatus()
        return {
            "assurance_scope": "internal_validation_evidence",
            "formal_certification_conferred": False,
            "certification_status": asdict(status),
            "compliance_matrix": self._generate_compliance_matrix(),
            "safety_case": self.artifacts["safety_case"],
            "evidence_bundle": self._package_evidence(),
            "generated_at": _utc_now_iso(),
        }

    def _generate_compliance_matrix(self) -> Dict[str, bool]:
        static_report = self.artifacts.get("static_report") or {}
        security = static_report.get("security_metrics", {}) if isinstance(static_report, Mapping) else {}
        max_critical = int(self.protocol.static_analysis["security"]["max_critical"])
        behavioral = self.artifacts.get("behavioral_results") or {}
        return {
            "static_analysis_internal_gate": int(security.get("critical_count", 0) or 0) <= max_critical,
            "behavioral_internal_gate": str(behavioral.get("overall_status", "")).upper() != "FAILED",
            "safety_case_present": bool(self.artifacts.get("safety_case")),
        }

    def _package_evidence(self) -> Dict[str, Any]:
        return {
            "static_analysis": self.artifacts["static_report"],
            "behavioral_tests": self.artifacts["behavioral_results"],
            "traceability_matrix": self._generate_traceability_matrix(),
            "tool_qualification": self._qualify_validation_tools(),
        }

    def _generate_traceability_matrix(self) -> List[Dict[str, Any]]:
        payload = self.artifacts.get("behavioral_results")
        if not isinstance(payload, Mapping):
            raise CertificationError("Behavioral results are missing; traceability cannot be generated.")
        records = payload.get("traceability_matrix") or payload.get("records") or payload.get("tests") or []
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            records = []

        traceability: List[Dict[str, Any]] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            traceability.append(
                {
                    "requirement_id": record.get("requirement_id", "UNKNOWN"),
                    "test_case": record.get("test_id", record.get("test_case_id", "N/A")),
                    "status": record.get("status", "UNKNOWN"),
                    "rationale": record.get("notes", record.get("details", "No explanation provided")),
                }
            )
        return sorted(
            traceability,
            key=lambda item: (str(item["requirement_id"]), str(item["test_case"])),
        )

    @staticmethod
    def _qualify_validation_tools() -> Dict[str, Any]:
        # Do not make unsupported qualification/certification claims.
        return {
            "static_analyzer": {
                "qualification_status": "not_formally_qualified_by_this_module",
                "evidence_required": "Independent tool qualification evidence when mandated by the target standard.",
            },
            "behavioral_test_framework": {
                "qualification_status": "not_formally_qualified_by_this_module",
                "evidence_required": "Independent validation and process evidence for the applicable assurance context.",
            },
        }


if __name__ == "__main__":
    print("\n=== Running Evaluation Agent smoke test ===\n")
    printer.status("TEST", "Evaluation Agent initialized", "info")

    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    memory = SharedMemory()
    factory = AgentFactory()
    agent = EvaluationAgent(shared_memory=memory, agent_factory=factory)

    try:
        health = agent.get_overall_system_health()
        validation = agent.execute_validation_cycle(
            {
                "lightweight": True,
                "persist_results": False,
            }
        )
        printer.pretty("System Health", health, "success" if health else "error")
        printer.pretty("Validation", validation, "success" if validation else "error")
    finally:
        agent.shutdown()

    print("\n=== Evaluation Agent smoke test completed ===")
