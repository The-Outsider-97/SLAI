"""
Heuristic Selector – production-grade orchestration for planning heuristics.

This module selects, invokes, monitors, and persists performance data for the
planning heuristics exposed by ``src/agents/planning/heuristics``.  It keeps the
existing public API compatible while adding validated configuration, lazy
heuristic construction, robust diagnostics, performance smoothing, fallback
selection, and structured error handling.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import joblib  # type: ignore

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.planning_errors import *
from .utils.planning_helpers import *
from .heuristics import *
from .planning_memory import PlanningMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Heuristic Selector")
printer = PrettyPrinter()

StatsMap = Dict[Any, Dict[str, Any]]
SelectionResult = Tuple[Optional[str], float]


@dataclass
class HeuristicPerformance:
    """Runtime performance statistics for one heuristic."""

    speed: float = 0.05
    accuracy: float = 0.50
    confidence: float = 0.50
    invocations: int = 0
    failures: int = 0
    selections: int = 0
    predictions: int = 0
    last_used: float = 0.0
    last_error: str = ""

    @property
    def reliability(self) -> float:
        if self.invocations <= 0:
            return 1.0
        return clamp(1.0 - (self.failures / max(self.invocations, 1)), 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speed": self.speed,
            "accuracy": self.accuracy,
            "confidence": self.confidence,
            "invocations": self.invocations,
            "failures": self.failures,
            "selections": self.selections,
            "predictions": self.predictions,
            "last_used": self.last_used,
            "last_error": self.last_error,
            "reliability": self.reliability,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], defaults: Dict[str, float]) -> "HeuristicPerformance":
        require_type(data, dict, "heuristic_performance")
        return cls(
            speed=float(data.get("speed", defaults.get("speed", 0.05)) or defaults.get("speed", 0.05)),
            accuracy=clamp(float(data.get("accuracy", defaults.get("accuracy", 0.5)) or 0.5), 0.0, 1.0),
            confidence=clamp(float(data.get("confidence", defaults.get("confidence", 0.5)) or 0.5), 0.0, 1.0),
            invocations=int(data.get("invocations", 0) or 0),
            failures=int(data.get("failures", 0) or 0),
            selections=int(data.get("selections", 0) or 0),
            predictions=int(data.get("predictions", 0) or 0),
            last_used=float(data.get("last_used", 0.0) or 0.0),
            last_error=str(data.get("last_error", "") or ""),
        )


@dataclass
class SelectorDecision:
    """Inspectable trace entry for routing decisions."""

    timestamp: float
    task_name: str
    selected_heuristic: str
    reason: str
    candidates: List[str]
    time_budget: float
    score: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "task_name": self.task_name,
            "selected_heuristic": self.selected_heuristic,
            "reason": self.reason,
            "candidates": list(self.candidates),
            "time_budget": self.time_budget,
            "score": self.score,
            "diagnostics": dict(self.diagnostics),
        }


class HeuristicSelector:
    """
    Selects and supervises concrete planning heuristics.

    Public compatibility surface
    ----------------------------
    - ``select_heuristic(task, world_state, candidate_methods, time_budget)``
    - ``select_best_method(task, world_state, candidate_methods, method_stats, time_budget)``
    - ``predict_success_prob(task, world_state, method_stats, method_id, time_budget)``
    - ``update_performance(name, speed, accuracy)``
    - ``load_state()``, ``save_state()``, ``load_performance_stats()``,
      ``save_performance_stats()``
    """

    _FACTORIES: Dict[str, Callable[[], Any]] = {
        "DT": DecisionTreeHeuristic,
        "GB": GradientBoostingHeuristic,
        "RL": ReinforcementLearningHeuristic,
        "UA": UncertaintyAwareHeuristic,
        "CBR": CaseBasedReasoningHeuristic,
    }

    def __init__(self, memory: Optional[PlanningMemory] = None) -> None:
        self.config = load_global_config()
        self.selector_config = get_config_section("heuristic_selector", config=self.config, default={})
        self._validate_config()

        self.performance_log_path = str(self.selector_config.get("performance_log_path"))
        self.speed_weight = float(self.selector_config.get("speed_weight", 0.30))
        self.accuracy_weight = float(self.selector_config.get("accuracy_weight", 0.70))
        self.reliability_weight = float(self.selector_config.get("reliability_weight", 0.10))
        self.confidence_weight = float(self.selector_config.get("confidence_weight", 0.10))
        self.speed_threshold = float(self.selector_config.get("speed_threshold", 0.10))
        self.max_dt_depth = int(self.selector_config.get("max_dt_depth", 5))
        self.min_rl_sequence_length = int(self.selector_config.get("min_rl_sequence_length", 3))
        self.heuristic_priority = [str(h).upper() for h in self.selector_config.get("heuristic_priority", [])]
        self.time_budget = float(self.selector_config.get("time_budget", 0.5))
        self.smoothing_alpha = float(self.selector_config.get("smoothing_alpha", 0.30))
        self.memory_state_key = str(self.selector_config.get("memory_state_key", "heuristic_selector"))

        self.memory = memory if memory is not None else PlanningMemory()
        self.heuristics: Dict[str, Any] = {}
        self.unavailable_heuristics: Dict[str, str] = {}
        self.heuristic_performance: Dict[str, Dict[str, float]] = {}
        self._performance_state: Dict[str, HeuristicPerformance] = {}
        self.last_used: Dict[str, float] = {}
        self.selection_trace: List[Dict[str, Any]] = []
        self.last_decision: Optional[Dict[str, Any]] = None
        self._lock = threading.RLock()

        self._init_runtime_state()
        self.load_state()
        self.load_performance_stats()

        if not bool(self.selector_config.get("lazy_initialization", True)):
            self._initialize_all_heuristics()

        logger.info("Heuristic Selector initialized with priority=%s", self.heuristic_priority)

    # ------------------------------------------------------------------
    # State setup and config validation
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        cfg = self.selector_config
        require_type(cfg.get("heuristic_priority"), list, "heuristic_selector.heuristic_priority")
        require_positive(float(cfg.get("time_budget", 0.5)), "heuristic_selector.time_budget")
        require_non_negative(float(cfg.get("speed_threshold", 0.1)), "heuristic_selector.speed_threshold")
        validate_probability(float(cfg.get("accuracy_weight", 0.7)), "heuristic_selector.accuracy_weight")
        validate_probability(float(cfg.get("speed_weight", 0.3)), "heuristic_selector.speed_weight")
        validate_probability(float(cfg.get("smoothing_alpha", 0.3)), "heuristic_selector.smoothing_alpha")
        for name in cfg.get("heuristic_priority", []):
            if str(name).upper() not in self._FACTORIES:
                raise PlanningConfigError(
                    f"Unknown heuristic in priority list: {name!r}",
                    config_key="heuristic_priority",
                    config_section="heuristic_selector",
                    expected_type=f"one of {sorted(self._FACTORIES)}",
                )

    def _init_runtime_state(self) -> None:
        defaults = {
            "speed": float(self.selector_config.get("default_speed", 0.05)),
            "accuracy": float(self.selector_config.get("default_accuracy", 0.5)),
            "confidence": float(self.selector_config.get("default_confidence", 0.5)),
        }
        for name in self._FACTORIES:
            perf = HeuristicPerformance(
                speed=defaults["speed"],
                accuracy=defaults["accuracy"],
                confidence=defaults["confidence"],
            )
            self._performance_state[name] = perf
            self.heuristic_performance[name] = {"speed": perf.speed, "accuracy": perf.accuracy}
            self.last_used[name] = 0.0

    def _sync_public_performance(self) -> None:
        self.heuristic_performance = {
            name: {"speed": perf.speed, "accuracy": perf.accuracy}
            for name, perf in self._performance_state.items()
        }
        self.last_used = {name: perf.last_used for name, perf in self._performance_state.items()}

    # ------------------------------------------------------------------
    # Heuristic construction and availability
    # ------------------------------------------------------------------
    def _initialize_all_heuristics(self) -> None:
        for name in self.heuristic_priority:
            self._get_heuristic(name)

    def _get_heuristic(self, heuristic_name: str) -> Optional[Any]:
        name = str(heuristic_name).upper()
        if name in self.heuristics:
            return self.heuristics[name]
        if name in self.unavailable_heuristics:
            return None
        factory = self._FACTORIES.get(name)
        if factory is None:
            self.unavailable_heuristics[name] = "unknown_heuristic"
            return None
        try:
            heuristic = factory()
            self.heuristics[name] = heuristic
            return heuristic
        except Exception as exc:
            self.unavailable_heuristics[name] = str(exc)
            self._record_failure(name, exc)
            if bool(self.selector_config.get("fail_fast_on_init", False)):
                raise
            logger.error("Failed to initialize heuristic %s: %s", name, exc)
            return None

    def _heuristic_available(self, heuristic_name: str, time_budget: float) -> bool:
        name = str(heuristic_name).upper()
        heuristic = self._get_heuristic(name)
        if heuristic is None:
            return False
        perf = self._performance_state[name]
        trained = bool(getattr(heuristic, "trained", False))
        if not trained and not bool(self.selector_config.get("allow_untrained_fallback", True)):
            return False
        budget = float(time_budget or self.time_budget)
        speed_buffer = float(self.selector_config.get("availability_speed_buffer", 1.2))
        if budget <= 0:
            return True
        return perf.speed * speed_buffer <= budget or perf.speed <= self.speed_threshold

    def available_heuristics(self, time_budget: Optional[float] = None) -> List[str]:
        budget = self.time_budget if time_budget is None else float(time_budget)
        return [name for name in self.heuristic_priority if self._heuristic_available(name, budget)]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load_state(self) -> None:
        """Load selector state from PlanningMemory's base state when present."""
        with self._lock:
            base_state = getattr(self.memory, "_base_state", {}) or {}
            state = base_state.get(self.memory_state_key, {}) if isinstance(base_state, dict) else {}
            if not isinstance(state, dict):
                return

            perf_data = state.get("performance", {})
            defaults = {
                "speed": float(self.selector_config.get("default_speed", 0.05)),
                "accuracy": float(self.selector_config.get("default_accuracy", 0.5)),
                "confidence": float(self.selector_config.get("default_confidence", 0.5)),
            }
            if isinstance(perf_data, dict):
                for name, value in perf_data.items():
                    key = str(name).upper()
                    if key in self._performance_state and isinstance(value, dict):
                        self._performance_state[key] = HeuristicPerformance.from_dict(value, defaults)

            last_used = state.get("last_used", {})
            if isinstance(last_used, dict):
                for name, value in last_used.items():
                    key = str(name).upper()
                    if key in self._performance_state:
                        self._performance_state[key].last_used = float(value or 0.0)

            trace = state.get("selection_trace", [])
            if isinstance(trace, list):
                limit = int(self.selector_config.get("max_trace_entries", 250))
                self.selection_trace = [x for x in trace[-limit:] if isinstance(x, dict)]

            self._sync_public_performance()

    def save_state(self) -> None:
        """Persist selector state into PlanningMemory's base state."""
        with self._lock:
            if getattr(self.memory, "_base_state", None) is None:
                return
            payload = {
                "performance": {name: perf.to_dict() for name, perf in self._performance_state.items()},
                "last_used": {name: perf.last_used for name, perf in self._performance_state.items()},
                "unavailable": dict(self.unavailable_heuristics),
                "selection_trace": list(self.selection_trace[-int(self.selector_config.get("max_trace_entries", 250)):]),
                "saved_at": time.time(),
            }
            self.memory._base_state[self.memory_state_key] = payload
            if getattr(self.memory, "agent", None):
                self.memory.save_checkpoint(label="heuristic_update", metadata={"component": "heuristic_selector"})

    def load_performance_stats(self) -> None:
        """Load persisted performance statistics from disk if configured."""
        if not bool(self.selector_config.get("persist_performance", True)):
            return
        path = self.performance_log_path
        if not path or not os.path.exists(path):
            return
        try:
            loaded = joblib.load(path)
            if not isinstance(loaded, dict):
                raise PlanningConfigError(
                    "Heuristic performance file must contain a dictionary.",
                    config_key="performance_log_path",
                    config_section="heuristic_selector",
                    expected_type="joblib dict",
                )
            defaults = {
                "speed": float(self.selector_config.get("default_speed", 0.05)),
                "accuracy": float(self.selector_config.get("default_accuracy", 0.5)),
                "confidence": float(self.selector_config.get("default_confidence", 0.5)),
            }
            for name, value in loaded.items():
                key = str(name).upper()
                if key in self._performance_state and isinstance(value, dict):
                    self._performance_state[key] = HeuristicPerformance.from_dict(value, defaults)
            self._sync_public_performance()
        except Exception as exc:
            logger.error("Failed to load heuristic performance stats: %s", exc)

    def save_performance_stats(self) -> None:
        """Atomically save performance statistics to disk."""
        if not bool(self.selector_config.get("persist_performance", True)):
            return
        path = self.performance_log_path
        if not path:
            return
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        payload = {name: perf.to_dict() for name, perf in self._performance_state.items()}
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(dir=directory, prefix="heuristic_perf_", suffix=".tmp", delete=False) as fh:
                tmp_path = fh.name
            joblib.dump(payload, tmp_path)
            os.replace(tmp_path, path)
        except Exception as exc:
            logger.error("Failed to save heuristic performance stats: %s", exc)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Performance updates
    # ------------------------------------------------------------------
    def update_performance(self, heuristic_name: str, speed: float, accuracy: float, *, confidence: Optional[float] = None) -> None:
        """Update smoothed performance metrics for a heuristic."""
        name = str(heuristic_name).upper()
        if name not in self._performance_state:
            raise PlanningConfigError(
                f"Unknown heuristic performance key: {heuristic_name!r}",
                config_key="heuristic_name",
                config_section="heuristic_selector",
                expected_type=f"one of {sorted(self._performance_state)}",
            )
        require_non_negative(float(speed), f"{name}.speed")
        validate_probability(float(accuracy), f"{name}.accuracy")
        if confidence is not None:
            validate_probability(float(confidence), f"{name}.confidence")

        alpha = self.smoothing_alpha
        with self._lock:
            perf = self._performance_state[name]
            perf.speed = (1.0 - alpha) * perf.speed + alpha * float(speed)
            perf.accuracy = clamp((1.0 - alpha) * perf.accuracy + alpha * float(accuracy), 0.0, 1.0)
            if confidence is not None:
                perf.confidence = clamp((1.0 - alpha) * perf.confidence + alpha * float(confidence), 0.0, 1.0)
            perf.invocations += 1
            perf.last_used = time.time()
            self._sync_public_performance()
            self.save_performance_stats()
            self.save_state()

    def _record_failure(self, heuristic_name: str, error: Exception) -> None:
        name = str(heuristic_name).upper()
        if name not in self._performance_state:
            return
        perf = self._performance_state[name]
        perf.failures += 1
        perf.invocations += 1
        perf.last_error = truncate_for_logging(str(error), 256)
        perf.last_used = time.time()
        self._sync_public_performance()

    # ------------------------------------------------------------------
    # Selection routing
    # ------------------------------------------------------------------
    def select_heuristic(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        candidate_methods: List[str],
        time_budget: float = 0.5,
    ) -> Tuple[str, Any]:
        """Select the best concrete heuristic for the current planning context."""
        require_type(task, dict, "task")
        require_type(world_state, dict, "world_state")
        require_type(candidate_methods, list, "candidate_methods")
        if bool(self.selector_config.get("require_candidates", True)) and not candidate_methods:
            raise MethodSelectionError(
                "HeuristicSelector.select_heuristic requires at least one candidate method.",
                task_name=str(task.get("name", "")),
                task_id=str(task.get("id", "")),
                candidate_methods=[],
            )

        budget = float(time_budget if time_budget is not None else self.time_budget)
        require_positive(budget, "time_budget")
        route = self._route_by_context(task, world_state, candidate_methods, budget)
        if route is not None:
            name, heuristic, reason, score = route
            self._record_decision(task, name, reason, candidate_methods, budget, score)
            return name, heuristic

        name, heuristic, score = self._select_by_performance(budget)
        self._record_decision(task, name, "performance_fallback", candidate_methods, budget, score)
        return name, heuristic

    def _route_by_context(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        candidate_methods: List[str],
        time_budget: float,
    ) -> Optional[Tuple[str, Any, str, float]]:
        thresholds = self.selector_config.get("routing_thresholds", {}) or {}
        ordered_checks: List[Tuple[str, Callable[[], bool], str]] = [
            ("RL", lambda: self._is_sequential_task(task), "sequential_task"),
            ("DT", lambda: self._calculate_task_depth(task) > self.max_dt_depth, "deep_task_hierarchy"),
            (
                "GB",
                lambda: self._world_float(world_state, "cpu_available", 1.0) < float(thresholds.get("resource_cpu_low", 0.3))
                or self._world_float(world_state, "memory_available", 1.0) < float(thresholds.get("resource_memory_low", 0.3)),
                "resource_constrained_context",
            ),
            (
                "UA",
                lambda: self._task_float(task, "priority", 0.0) >= float(thresholds.get("safety_priority", 0.8))
                or bool(task.get("safety_critical", False))
                or self._task_float(task, "risk_score", 0.0) >= float(thresholds.get("high_uncertainty_risk", 0.7)),
                "safety_or_high_risk_task",
            ),
            ("CBR", lambda: self._cbr_is_applicable(task), "sufficient_case_history"),
        ]

        for name, predicate, reason in ordered_checks:
            if name not in self.heuristic_priority:
                continue
            if predicate() and self._heuristic_available(name, time_budget):
                heuristic = self._get_heuristic(name)
                if heuristic is not None:
                    return name, heuristic, reason, self._performance_score(name, time_budget)
        return None

    def _select_by_performance(self, time_budget: float) -> Tuple[str, Any, float]:
        candidates: List[Tuple[float, str, Any]] = []
        for name in self.heuristic_priority:
            heuristic = self._get_heuristic(name)
            if heuristic is None:
                continue
            if not self._heuristic_available(name, time_budget):
                continue
            candidates.append((self._performance_score(name, time_budget), name, heuristic))

        if not candidates:
            fallback = str(self.selector_config.get("fallback_heuristic", "DT")).upper()
            heuristic = self._get_heuristic(fallback)
            if heuristic is None:
                # Last resort: any successfully initialized heuristic.
                for name in self.heuristic_priority:
                    heuristic = self._get_heuristic(name)
                    if heuristic is not None:
                        return name, heuristic, 0.0
                raise MethodSelectionError(
                    "No heuristics are available for selection.",
                    candidate_methods=[],
                    selection_scores={},
                    context={"unavailable": dict(self.unavailable_heuristics)},
                )
            return fallback, heuristic, 0.0

        candidates.sort(key=lambda item: (item[0], -self.heuristic_priority.index(item[1])), reverse=True)
        score, name, heuristic = candidates[0]
        return name, heuristic, score

    def _performance_score(self, heuristic_name: str, time_budget: float) -> float:
        perf = self._performance_state[heuristic_name]
        budget = max(float(time_budget), 1e-9)
        speed_score = clamp(1.0 - (perf.speed / budget), 0.0, 1.0)
        score = (
            self.accuracy_weight * perf.accuracy
            + self.speed_weight * speed_score
            + self.reliability_weight * perf.reliability
            + self.confidence_weight * perf.confidence
        )
        return clamp(score / max(self.accuracy_weight + self.speed_weight + self.reliability_weight + self.confidence_weight, 1e-9), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Primary prediction and method-selection API
    # ------------------------------------------------------------------
    def select_best_method(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        candidate_methods: List[str],
        method_stats: StatsMap,
        time_budget: float = 0.5,
    ) -> SelectionResult:
        """Select the best method by routing to the most suitable heuristic."""
        require_type(method_stats, dict, "method_stats")
        start = time.time()
        heuristic_name, heuristic = self.select_heuristic(task, world_state, candidate_methods, time_budget)
        try:
            method, confidence = heuristic.select_best_method(task, world_state, candidate_methods, method_stats)
            elapsed = time.time() - start
            confidence = clamp(float(confidence or 0.0), 0.0, 1.0)
            accuracy = self._infer_accuracy(task, method, confidence)
            self._performance_state[heuristic_name].selections += 1
            self.update_performance(heuristic_name, elapsed, accuracy, confidence=confidence)
            return method, confidence
        except PlanningError:
            self._record_failure(heuristic_name, MethodSelectionError("Planning heuristic selection failed", candidate_methods=candidate_methods))
            raise
        except Exception as exc:
            self._record_failure(heuristic_name, exc)
            logger.error("Heuristic %s failed during method selection: %s", heuristic_name, exc)
            return self._fallback_select_best_method(task, world_state, candidate_methods, method_stats, time_budget, exc)

    def predict_success_prob(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        method_stats: StatsMap,
        method_id: str,
        time_budget: float = 0.5,
    ) -> float:
        """Predict method success probability using the selected heuristic."""
        require_type(method_stats, dict, "method_stats")
        method = str(method_id).strip()
        require_non_empty(method, "method_id")

        start = time.time()
        heuristic_name, heuristic = self.select_heuristic(task, world_state, [method], time_budget)
        try:
            prob = float(heuristic.predict_success_prob(task, world_state, method_stats, method))
            validate_probability(prob, "predicted_success_probability")
            elapsed = time.time() - start
            accuracy = self._infer_accuracy(task, None, prob)
            self._performance_state[heuristic_name].predictions += 1
            self.update_performance(heuristic_name, elapsed, accuracy, confidence=max(prob, 1.0 - prob))
            return clamp(prob, 0.0, 1.0)
        except PlanningError:
            self._record_failure(heuristic_name, MethodSelectionError("Planning heuristic prediction failed", candidate_methods=[method]))
            raise
        except Exception as exc:
            self._record_failure(heuristic_name, exc)
            logger.error("Heuristic %s failed during prediction: %s", heuristic_name, exc)
            return self._fallback_success_rate(task, method_stats, method)

    def _fallback_select_best_method(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        candidate_methods: List[str],
        method_stats: StatsMap,
        time_budget: float,
        cause: Exception,
    ) -> SelectionResult:
        scores: Dict[str, float] = {}
        for method in candidate_methods:
            method_id = str(method)
            scores[method_id] = self._fallback_success_rate(task, method_stats, method_id)
        if not scores:
            raise MethodSelectionError(
                f"Heuristic fallback failed after: {cause}",
                task_name=str(task.get("name", "")),
                task_id=str(task.get("id", "")),
                candidate_methods=list(candidate_methods),
                selection_scores=scores,
            ) from cause
        best = max(scores, key=scores.__getitem__)
        self._record_decision(task, "FALLBACK", "statistics_fallback", candidate_methods, time_budget, scores[best])
        return best, scores[best]

    # ------------------------------------------------------------------
    # Outcome feedback
    # ------------------------------------------------------------------
    def record_outcome(self, heuristic_name: str, task: Dict[str, Any], world_state: Dict[str, Any],
                       method_used: str, outcome: Any, *, extra_context: Optional[Dict[str, Any]] = None) -> None:
        """Record an execution outcome in memory and forward it to capable heuristics."""
        name = str(heuristic_name).upper()
        success = self._normalise_outcome(outcome)
        accuracy = 1.0 if success else 0.0
        if name in self._performance_state:
            self.update_performance(name, self._performance_state[name].speed, accuracy)

        if hasattr(self.memory, "record_task_outcome"):
            self.memory.record_task_outcome(
                task_id=str(task.get("id", task.get("name", "unknown"))),
                status="success" if success else "failure",
                metadata={
                    "method": method_used,
                    "heuristic": name,
                    "duration": float(extra_context.get("duration", 0.0)) if extra_context else 0.0,
                },
            )
        if hasattr(self.memory, "update_method_stats"):
            self.memory.update_method_stats(str(task.get("name", "task")), method_used, success) # type: ignore

        heuristic = self._get_heuristic(name)
        if heuristic is not None and hasattr(heuristic, "record_outcome"):
            try:
                heuristic.record_outcome(
                    task,
                    world_state,
                    method_used,
                    "success" if success else "failure",
                    extra_context=extra_context,
                )
            except TypeError:
                heuristic.record_outcome(task, world_state, method_used, "success" if success else "failure")

    # ------------------------------------------------------------------
    # Diagnostics and helpers
    # ------------------------------------------------------------------
    def explain_last_decision(self) -> Dict[str, Any]:
        """Return the latest routing/selection diagnostic."""
        return dict(self.last_decision or {})

    def get_performance_report(self) -> Dict[str, Any]:
        """Return a serialisable performance report."""
        return {
            "heuristics": {name: perf.to_dict() for name, perf in self._performance_state.items()},
            "available": self.available_heuristics(self.time_budget),
            "unavailable": dict(self.unavailable_heuristics),
            "last_decision": self.explain_last_decision(),
        }

    def _record_decision(self, task: Dict[str, Any], heuristic_name: str, reason: str,
                         candidate_methods: List[str], time_budget: float, score: float) -> None:
        if not bool(self.selector_config.get("record_selection_trace", True)):
            return
        decision = SelectorDecision(
            timestamp=time.time(),
            task_name=str(task.get("name", task.get("id", "unknown"))),
            selected_heuristic=str(heuristic_name),
            reason=reason,
            candidates=[str(c) for c in candidate_methods],
            time_budget=float(time_budget),
            score=float(score),
            diagnostics={
                "performance": self._performance_state.get(str(heuristic_name).upper(), HeuristicPerformance()).to_dict(),
                "unavailable": dict(self.unavailable_heuristics),
            },
        ).to_dict()
        limit = int(self.selector_config.get("max_trace_entries", 250))
        with self._lock:
            self.selection_trace.append(decision)
            if len(self.selection_trace) > limit:
                self.selection_trace = self.selection_trace[-limit:]
            self.last_decision = decision

    def _is_sequential_task(self, task: Dict[str, Any]) -> bool:
        if hasattr(self.memory, "is_sequential_task"):
            try:
                if self.memory.is_sequential_task(task, self.min_rl_sequence_length):
                    return True
            except Exception:
                pass
        if len(task.get("dependencies", []) or []) >= self.min_rl_sequence_length - 1:
            return True
        current: Any = task
        depth = 1
        while isinstance(current, dict) and current.get("parent") is not None:
            depth += 1
            current = current.get("parent")
            if depth >= self.min_rl_sequence_length:
                return True
        return False

    def _calculate_task_depth(self, task: Dict[str, Any]) -> int:
        depth = 0
        current: Any = task
        visited: set[int] = set()
        while isinstance(current, dict) and current.get("parent") is not None:
            ident = id(current)
            if ident in visited:
                break
            visited.add(ident)
            depth += 1
            current = current.get("parent")
        return depth

    def _cbr_is_applicable(self, task: Dict[str, Any]) -> bool:
        cbr = self._get_heuristic("CBR")
        case_count = len(getattr(cbr, "case_base", []) or []) if cbr is not None else 0
        if case_count >= int(self.selector_config.get("min_cases_for_cbr", 10)):
            return True
        task_type = str(task.get("type", task.get("category", ""))).lower()
        return task_type in {str(x).lower() for x in self.selector_config.get("cbr_task_types", [])}

    def _fallback_success_rate(self, task: Dict[str, Any], method_stats: StatsMap, method_id: str) -> float:
        task_name = str(task.get("name", task.get("id", "")))
        keys = [(task_name, method_id), f"{task_name}:{method_id}", method_id]
        for key in keys:
            stats = method_stats.get(key)
            if isinstance(stats, dict):
                total = float(stats.get("total", stats.get("attempts", 0)) or 0.0)
                success = float(stats.get("success", stats.get("successes", 0)) or 0.0)
                failures = float(stats.get("failures", stats.get("failure", 0)) or 0.0)
                if total <= 0 and success + failures > 0:
                    total = success + failures
                if total > 0:
                    return clamp(success / total, 0.0, 1.0)
        return float(self.selector_config.get("default_accuracy", 0.5))

    def _infer_accuracy(self, task: Dict[str, Any], method: Optional[str], confidence: float) -> float:
        try:
            outcome = self.memory.get_task_outcome(str(task.get("id", task.get("name", ""))))
        except Exception:
            outcome = None
        if outcome is None:
            return clamp(float(confidence), 0.0, 1.0)
        return 1.0 if (float(confidence) >= 0.5) == bool(outcome) else 0.0

    @staticmethod
    def _normalise_outcome(outcome: Any) -> bool:
        if isinstance(outcome, bool):
            return outcome
        if isinstance(outcome, str):
            return outcome.strip().lower() in {"success", "succeeded", "true", "1", "ok"}
        return bool(outcome)

    @staticmethod
    def _task_float(task: Dict[str, Any], key: str, default: float) -> float:
        try:
            return float(task.get(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _world_float(world_state: Dict[str, Any], key: str, default: float) -> float:
        try:
            return float(world_state.get(key, default))
        except (TypeError, ValueError):
            return default


if __name__ == "__main__":
    print("\n=== Running Heuristic Selector ===\n")
    printer.status("TEST", "Heuristic Selector initialized", "info")

    selector = HeuristicSelector()
    task = {
        "id": "nav_1",
        "name": "navigation",
        "priority": 0.9,
        "goal_state": {"position": "target"},
        "parent": {"name": "mission", "parent": {"name": "root", "parent": None}},
        "deadline": time.time() + 900,
        "risk_score": 0.2,
    }
    world_state = {"position": "start", "cpu_available": 0.7, "memory_available": 0.8}
    methods = ["A*", "RRT", "D*"]
    stats = {
        ("navigation", "A*"): {"success": 15, "total": 20},
        ("navigation", "RRT"): {"success": 12, "total": 18},
        ("navigation", "D*"): {"success": 8, "total": 15},
    }

    heuristic_name, _ = selector.select_heuristic(task, world_state, methods, time_budget=0.5)
    printer.status("SELECT", f"Selected heuristic: {heuristic_name}", "success")
    method, confidence = selector.select_best_method(task, world_state, methods, stats, time_budget=0.5)
    printer.status("METHOD", f"Selected method: {method} ({confidence:.3f})", "success")
    for method_id in methods:
        prob = selector.predict_success_prob(task, world_state, stats, method_id, time_budget=0.5)
        printer.status("PREDICT", f"{method_id}: {prob:.3f}", "info")

    print("\n=== Test ran successfully ===\n")
