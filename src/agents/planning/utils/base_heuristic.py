"""
Base Heuristics – production-grade foundation for planning method selection.

This module centralises the shared feature extraction, reliability estimation,
scoring, validation, and selection behaviour used by concrete planning
heuristics.  Concrete heuristics only need to implement ``predict_success_prob``;
the base class provides a deterministic, thread-safe ``select_best_method`` that
uses the shared planning helper and error modules instead of duplicating utility
logic.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import threading
import time

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .config_loader import get_config_section, load_global_config
from .planning_calculations import PlanningCalculations
from .planning_errors import *
from .planning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Base Heuristics")
printer = PrettyPrinter()

StatsMap = Dict[Any, Dict[str, Any]]
FeatureMap = Dict[str, float]
ScoreMap = Dict[str, float]


class BaseHeuristics(ABC):
    """
    Abstract base class for production planning heuristics.

    Responsibilities
    ----------------
    - Load and validate the ``base_heuristics`` section from ``planning_config.yaml``.
    - Extract stable, bounded feature vectors from dict-based or object-based tasks.
    - Reuse shared planning helpers for state distance, JSON handling, validation,
      clamping, and structured exceptions.
    - Provide a reusable method-selection implementation that concrete heuristics
      can inherit instead of duplicating selection boilerplate.
    """

    def __init__(self, calculations: Optional[Any] = None) -> None:
        super().__init__()
        self.config = load_global_config()
        self.bh_cfg = get_config_section("base_heuristics", config=self.config, default={})
        self._validate_config()

        if calculations is None:
            self.calc = PlanningCalculations()
        elif isinstance(calculations, type):
            self.calc = calculations()
        else:
            self.calc = calculations

        self._feature_cache: Dict[str, Tuple[float, FeatureMap]] = {}
        self._cache_lock = threading.RLock()
        logger.info("Base Heuristics successfully initialized")

    # ------------------------------------------------------------------
    # Abstract prediction API
    # ------------------------------------------------------------------
    @abstractmethod
    def predict_success_prob(
        self,
        task: Any,
        world_state: Dict[str, Any],
        method_stats: StatsMap,
        method_id: str,
    ) -> float:
        """Return a success probability for ``method_id`` in [0, 1]."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Default selection API – concrete heuristics may override if needed
    # ------------------------------------------------------------------
    def select_best_method(
        self,
        task: Any,
        world_state: Dict[str, Any],
        candidate_methods: List[str],
        method_stats: StatsMap,
    ) -> Tuple[Optional[str], float]:
        """
        Select the highest-scoring method from ``candidate_methods``.

        The score combines the concrete heuristic's predicted success
        probability with base reliability and safety/alignment features.  Invalid
        candidates are rejected with structured diagnostics; if no candidate can
        be selected, a ``MethodSelectionError`` is raised unless configured
        otherwise.
        """
        scores, diagnostics = self.score_candidate_methods(
            task=task,
            world_state=world_state,
            candidate_methods=candidate_methods,
            method_stats=method_stats,
        )

        if not scores:
            if not self._selection_cfg().get("raise_on_empty_candidates", True):
                return None, 0.0
            raise MethodSelectionError(
                "No valid heuristic candidate methods were available for selection.",
                task_name=str(self._task_value(task, "name", "")),
                task_id=str(self._task_value(task, "id", "")),
                candidate_methods=list(candidate_methods or []),
                selection_scores=diagnostics,
            )

        best_method = max(
            scores,
            key=lambda method_id: self._selection_sort_key(method_id, scores, diagnostics),
        )
        best_score = scores[best_method]
        min_score = float(self._selection_cfg().get("min_score", 0.0))

        if best_score < min_score and self._selection_cfg().get("raise_below_min_score", False):
            raise MethodSelectionError(
                f"Best heuristic method score {best_score:.3f} is below the minimum {min_score:.3f}.",
                task_name=str(self._task_value(task, "name", "")),
                task_id=str(self._task_value(task, "id", "")),
                candidate_methods=list(candidate_methods or []),
                selection_scores=scores,
            )

        logger.debug(
            "Selected method %s with score %.4f for task %s",
            best_method,
            best_score,
            truncate_for_logging(self._task_value(task, "name", task), 96),
        )
        return best_method, best_score

    def score_candidate_methods(
        self,
        task: Any,
        world_state: Dict[str, Any],
        candidate_methods: List[str],
        method_stats: StatsMap,
    ) -> Tuple[ScoreMap, ScoreMap]:
        """
        Score all candidates and return ``(scores, diagnostics)``.

        ``diagnostics`` currently stores method-confidence values for stable
        tie-breaking and external observability.
        """
        require_type(world_state, dict, "world_state")
        require_type(method_stats, dict, "method_stats")
        require_type(candidate_methods, list, "candidate_methods")

        if not candidate_methods:
            return {}, {}

        scores: ScoreMap = {}
        diagnostics: ScoreMap = {}
        seen: set[str] = set()

        for raw_method_id in candidate_methods:
            method_id = str(raw_method_id).strip()
            if not method_id or method_id in seen:
                continue
            seen.add(method_id)

            try:
                predicted = float(
                    self.predict_success_prob(task, world_state, method_stats, method_id)
                )
                require_in_range(predicted, 0.0, 1.0, f"prediction[{method_id}]")
                features = self.extract_base_features(task, world_state, method_stats, method_id)
                scores[method_id] = self._combine_score(predicted, features)
                diagnostics[method_id] = features.get("method_confidence", 0.0)
            except PlanningError:
                raise
            except (TypeError, ValueError, AttributeError, KeyError) as exc:
                raise MethodSelectionError(
                    f"Failed to score heuristic method '{method_id}': {exc}",
                    task_name=str(self._task_value(task, "name", "")),
                    task_id=str(self._task_value(task, "id", "")),
                    candidate_methods=list(candidate_methods),
                    selection_scores=scores,
                    context={"method_id": method_id, "error": repr(exc)},
                ) from exc

        return scores, diagnostics

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def extract_base_features(
        self,
        task: Any,
        world_state: Dict[str, Any],
        method_stats: StatsMap,
        method_id: str,
    ) -> FeatureMap:
        """Extract a bounded, deterministic feature vector for a task-method pair."""
        require_type(world_state, dict, "world_state")
        require_type(method_stats, dict, "method_stats")
        require_non_empty(method_id, "method_id")

        cache_key = self._feature_cache_key(task, world_state, method_stats, method_id)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return dict(cached)

        goal_state = self._extract_goal_state(task)
        stats = self._resolve_method_stats(task, method_stats, method_id)
        temporal = self.extract_temporal_features(task)

        features: FeatureMap = {
            "task_depth": self._calculate_task_depth(task),
            "goal_overlap": self._calculate_goal_overlap(goal_state, world_state),
            "goal_distance": compute_state_distance(world_state, goal_state) if goal_state else 1.0,
            "goal_satisfied": 1.0 if goal_state and state_satisfies_goal(world_state, goal_state) else 0.0,
            "method_success_rate": stats["success_rate"],
            "method_failure_rate": stats["failure_rate"],
            "method_confidence": stats["confidence"],
            "state_diversity": self._calculate_state_diversity(world_state),
            "priority": self._calculate_priority(task),
            "dependency_load": self._calculate_dependency_load(task),
            "task_complexity": self._calculate_task_complexity(task),
            "risk_score": self._calculate_risk_score(task),
            "resource_margin": self._calculate_resource_margin(task),
            "temporal_margin": self._calculate_temporal_margin(task),
            **temporal,
        }

        normalised = self._normalise_features(features)
        self._cache_set(cache_key, normalised)
        return normalised

    def extract_temporal_features(self, task: Any) -> FeatureMap:
        """Return age and deadline-pressure features in [0, 1]."""
        created_at = self._coerce_datetime(self._task_value(task, "creation_time", None))
        deadline = self._coerce_datetime(self._task_value(task, "deadline", None))
        now = datetime.now(timezone.utc)

        max_age_hours = float(self.bh_cfg.get("max_age_hours", 168.0))
        age_hours = 0.0
        if created_at is not None:
            age_hours = max(0.0, (now - created_at).total_seconds() / 3600.0)

        proximity = 0.0
        urgency = 0.0
        if deadline is not None:
            horizon = float(self.bh_cfg.get("deadline_horizon_seconds", 86400.0))
            seconds_left = (deadline - now).total_seconds()
            if seconds_left <= 0.0:
                proximity = 1.0
                urgency = 1.0
            else:
                urgency = clamp(1.0 - (seconds_left / max(horizon, 1.0)), 0.0, 1.0)
                if created_at is not None:
                    total = max((deadline - created_at).total_seconds(), 1.0)
                    elapsed = max((now - created_at).total_seconds(), 0.0)
                    proximity = clamp(elapsed / total, 0.0, 1.0)
                else:
                    proximity = urgency

        return {
            "time_since_creation": clamp(age_hours / max(max_age_hours, 1.0), 0.0, 1.0),
            "deadline_proximity": proximity,
            "urgency": urgency,
        }

    # ------------------------------------------------------------------
    # Feature calculators
    # ------------------------------------------------------------------
    def _calculate_task_depth(self, task: Any) -> float:
        max_depth = int(self.bh_cfg.get("max_task_depth", 20))
        current = task
        visited: set[int] = set()
        depth = 0

        while current is not None and id(current) not in visited:
            visited.add(id(current))
            parent = self._task_value(current, "parent", None)
            if parent is None:
                break
            depth += 1
            if depth >= max_depth:
                break
            current = parent

        return clamp(depth / max(max_depth, 1), 0.0, 1.0)

    @staticmethod
    def _calculate_goal_overlap(goal_state: Dict[str, Any], world_state: Dict[str, Any]) -> float:
        if not goal_state:
            return 0.0
        return clamp(1.0 - compute_state_distance(world_state, goal_state), 0.0, 1.0)

    def _calculate_state_diversity(self, world_state: Dict[str, Any]) -> float:
        normalizer = float(self.bh_cfg.get("state_diversity_normalizer", 100.0))
        values = [float(v) for v in world_state.values() if isinstance(v, (int, float))]
        if len(values) <= 1:
            return 0.0
        return clamp(statistics.pstdev(values) / max(normalizer, 1.0), 0.0, 1.0)

    def _calculate_priority(self, task: Any) -> float:
        priority = self._to_float(self._task_value(task, "priority", 0.0), 0.0)
        max_priority = float(self.bh_cfg.get("max_priority", 10.0))
        return clamp(priority / max(max_priority, 1.0), 0.0, 1.0)

    def _calculate_dependency_load(self, task: Any) -> float:
        dependencies = self._task_value(task, "dependencies", []) or []
        if not isinstance(dependencies, Sequence) or isinstance(dependencies, (str, bytes)):
            dependencies = [dependencies]
        max_deps = int(self.bh_cfg.get("max_dependency_count", 25))
        return clamp(len(dependencies) / max(max_deps, 1), 0.0, 1.0)

    def _calculate_task_complexity(self, task: Any) -> float:
        requirements = self._task_value(task, "requirements", []) or []
        preconditions = self._task_value(task, "preconditions", []) or []
        dependencies = self._task_value(task, "dependencies", []) or []
        subtasks = self._task_value(task, "subtasks", []) or []
        cost = self._to_float(self._task_value(task, "cost", 0.0), 0.0)
        duration = self._to_float(self._task_value(task, "duration", 0.0), 0.0)

        collection_load = sum(
            len(v) if isinstance(v, Sequence) and not isinstance(v, (str, bytes)) else 0
            for v in (requirements, preconditions, dependencies, subtasks)
        )
        raw = collection_load + math.log1p(max(cost, 0.0)) + math.log1p(max(duration, 0.0) / 60.0)
        return clamp(raw / 25.0, 0.0, 1.0)

    def _calculate_risk_score(self, task: Any) -> float:
        explicit = self._task_value(task, "risk_score", None)
        if explicit is not None:
            return clamp(self._to_float(explicit, 0.0), 0.0, 1.0)
        if self.bh_cfg.get("use_planning_calculations", True) and self.calc is not None:
            try:
                if not isinstance(task, Mapping) and hasattr(self.calc, "estimate_risk_score"):
                    return clamp(float(self.calc.estimate_risk_score(task)), 0.0, 1.0)
            except (PlanningError, TypeError, AttributeError, ValueError) as exc:
                logger.debug("Risk-score calculation skipped: %s", exc)
        return 0.0

    def _calculate_resource_margin(self, task: Any) -> float:
        explicit = self._task_value(task, "resource_margin", None)
        if explicit is not None:
            return clamp(self._to_float(explicit, 1.0), 0.0, 1.0)
        if self.bh_cfg.get("use_planning_calculations", True) and self.calc is not None:
            try:
                if not isinstance(task, Mapping) and hasattr(self.calc, "calculate_resource_margin"):
                    return clamp(float(self.calc.calculate_resource_margin(task)), 0.0, 1.0)
            except (PlanningError, TypeError, AttributeError, ValueError) as exc:
                logger.debug("Resource-margin calculation skipped: %s", exc)
        return 1.0

    def _calculate_temporal_margin(self, task: Any) -> float:
        explicit = self._task_value(task, "temporal_margin", None)
        if explicit is not None:
            return clamp(self._to_float(explicit, 1.0), 0.0, 1.0)
        if self.bh_cfg.get("use_planning_calculations", True) and self.calc is not None:
            try:
                if not isinstance(task, Mapping) and hasattr(self.calc, "calculate_temporal_margin"):
                    return clamp(float(self.calc.calculate_temporal_margin(task)), 0.0, 1.0)
            except (PlanningError, TypeError, AttributeError, ValueError) as exc:
                logger.debug("Temporal-margin calculation skipped: %s", exc)
        return 1.0

    # ------------------------------------------------------------------
    # Method statistics
    # ------------------------------------------------------------------
    def _resolve_method_stats(self, task: Any, method_stats: StatsMap, method_id: str) -> FeatureMap:
        task_name = self._task_value(task, "name", None)
        task_id = self._task_value(task, "id", None)
        candidate_keys = [
            (task_name, method_id),
            (task_id, method_id),
            f"{task_name}:{method_id}" if task_name else None,
            f"{task_id}:{method_id}" if task_id else None,
            method_id,
        ]

        raw_stats: Any = None
        for key in candidate_keys:
            if key is not None and key in method_stats:
                raw_stats = method_stats[key]
                break

        stats_cfg = self.bh_cfg.get("method_stats", {})
        success = float(stats_cfg.get("default_success", 1.0))
        total = float(stats_cfg.get("default_total", 2.0))

        if isinstance(raw_stats, Mapping):
            success = self._extract_numeric_from_keys(raw_stats, stats_cfg.get("success_keys", []), success)
            total = self._extract_numeric_from_keys(raw_stats, stats_cfg.get("total_keys", []), total)
            failures = self._extract_numeric_from_keys(raw_stats, stats_cfg.get("failure_keys", []), None)
            if failures is not None and ("total" not in raw_stats or total <= 0.0):
                total = success + max(failures, 0.0)
        elif isinstance(raw_stats, Sequence) and not isinstance(raw_stats, (str, bytes)):
            total = float(len(raw_stats))
            success = float(sum(1 for item in raw_stats if self._is_success_outcome(item)))

        total = max(total, 0.0)
        success = clamp(success, 0.0, total) if total > 0.0 else 0.0
        success_rate = success / total if total > 0.0 else 0.0
        failure_rate = 1.0 - success_rate
        scale = float(stats_cfg.get("confidence_sample_scale", 20.0))
        confidence = total / (total + max(scale, 1.0))

        return {
            "method_success_rate": success_rate,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "method_failure_rate": failure_rate,
            "confidence": clamp(confidence, 0.0, 1.0),
            "total": total,
        }

    @staticmethod
    def _extract_numeric_from_keys(stats: Mapping[str, Any], keys: Sequence[str], default: Any) -> Any:
        for key in keys:
            if key in stats:
                try:
                    return float(stats[key])
                except (TypeError, ValueError):
                    return default
        return default

    @staticmethod
    def _is_success_outcome(item: Any) -> bool:
        if isinstance(item, Mapping):
            value = item.get("outcome", item.get("status", item.get("success", False)))
        else:
            value = item
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"success", "succeeded", "ok", "true", "1", "win"}

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _combine_score(self, predicted_success: float, features: FeatureMap) -> float:
        weights = self.bh_cfg.get("selection_weights", {})
        penalties = self.bh_cfg.get("penalties", {})

        score = float(weights.get("predicted_success", 0.70)) * predicted_success
        for feature_name, weight in weights.items():
            if feature_name == "predicted_success":
                continue
            score += float(weight) * features.get(feature_name, 0.0)

        for feature_name, weight in penalties.items():
            score -= float(weight) * features.get(feature_name, 0.0)

        return clamp(score, 0.0, 1.0)

    def _selection_sort_key(self, method_id: str, scores: ScoreMap, diagnostics: ScoreMap) -> Tuple[float, float, str]:
        tie_breaker = str(self._selection_cfg().get("tie_breaker", "score_then_confidence_then_name"))
        if tie_breaker == "score_then_name":
            return scores[method_id], 0.0, method_id
        return scores[method_id], diagnostics.get(method_id, 0.0), method_id

    # ------------------------------------------------------------------
    # Goal/state utilities
    # ------------------------------------------------------------------
    def _extract_goal_state(self, task: Any) -> Dict[str, Any]:
        raw_goal = self._task_value(task, "goal_state", {})
        if raw_goal is None:
            return {}
        if isinstance(raw_goal, str):
            parsed = safe_json_loads(raw_goal, default={})
            if not isinstance(parsed, dict):
                raise PlanningConfigError(
                    "Task goal_state JSON must decode to a dictionary.",
                    config_key="goal_state",
                    config_section="base_heuristics",
                    expected_type="dict or JSON object string",
                )
            return parsed
        if not isinstance(raw_goal, dict):
            raise PlanningConfigError(
                f"Task goal_state must be a dictionary, got {type(raw_goal).__name__}.",
                config_key="goal_state",
                config_section="base_heuristics",
                expected_type="dict or JSON object string",
            )
        return dict(raw_goal)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def clear_feature_cache(self) -> None:
        """Clear the in-memory feature cache."""
        with self._cache_lock:
            self._feature_cache.clear()

    def _cache_get(self, key: str) -> Optional[FeatureMap]:
        if not self.bh_cfg.get("enable_feature_cache", True):
            return None
        ttl = float(self.bh_cfg.get("feature_cache_ttl_seconds", 5.0))
        if ttl <= 0.0:
            return None
        now = time.time()
        with self._cache_lock:
            entry = self._feature_cache.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if now >= expires_at:
                self._feature_cache.pop(key, None)
                return None
            return dict(value)

    def _cache_set(self, key: str, value: FeatureMap) -> None:
        if not self.bh_cfg.get("enable_feature_cache", True):
            return
        ttl = float(self.bh_cfg.get("feature_cache_ttl_seconds", 5.0))
        if ttl <= 0.0:
            return
        with self._cache_lock:
            self._feature_cache[key] = (time.time() + ttl, dict(value))

    def _feature_cache_key(self, task: Any, world_state: Dict[str, Any], method_stats: StatsMap, method_id: str) -> str:
        payload = safe_json_dumps(
            {
                "task": self._task_fingerprint(task),
                "world_state": world_state,
                "method_id": method_id,
                "stats": self._resolve_method_stats(task, method_stats, method_id),
            },
            fallback_str=str((self._task_fingerprint(task), world_state, method_id)),
        )
        return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _task_fingerprint(self, task: Any) -> Dict[str, Any]:
        keys = (
            "id",
            "name",
            "goal_state",
            "priority",
            "deadline",
            "creation_time",
            "dependencies",
            "requirements",
            "risk_score",
            "resource_margin",
            "temporal_margin",
        )
        return {key: self._serialisable_value(self._task_value(task, key, None)) for key in keys}

    # ------------------------------------------------------------------
    # Validation and coercion helpers
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        require_positive(float(self.bh_cfg.get("max_task_depth", 20)), "base_heuristics.max_task_depth")
        require_positive(float(self.bh_cfg.get("max_priority", 10.0)), "base_heuristics.max_priority")
        require_positive(float(self.bh_cfg.get("max_age_hours", 168.0)), "base_heuristics.max_age_hours")
        require_positive(
            float(self.bh_cfg.get("deadline_horizon_seconds", 86400.0)),
            "base_heuristics.deadline_horizon_seconds",
        )
        require_type(self.bh_cfg.get("selection_weights", {}), dict, "base_heuristics.selection_weights")
        require_type(self.bh_cfg.get("penalties", {}), dict, "base_heuristics.penalties")

    def _normalise_features(self, features: FeatureMap) -> FeatureMap:
        normalised: FeatureMap = {}
        for key, value in features.items():
            numeric = self._to_float(value, 0.0)
            if math.isnan(numeric) or math.isinf(numeric):
                numeric = 0.0
            normalised[key] = clamp(numeric, 0.0, 1.0)
        return normalised

    @staticmethod
    def _task_value(task: Any, key: str, default: Any = None) -> Any:
        if isinstance(task, Mapping):
            return task.get(key, default)
        return getattr(task, key, default)

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if isinstance(value, bool):
                return float(value)
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_datetime(value: Any) -> Optional[datetime]:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        elif isinstance(value, str):
            text = value.strip().replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(text)
            except ValueError as exc:
                raise PlanningConfigError(
                    f"Invalid datetime value for heuristic temporal feature: {value!r}",
                    config_key="creation_time/deadline",
                    config_section="base_heuristics",
                    expected_type="ISO-8601 datetime or POSIX timestamp",
                ) from exc
        else:
            return None

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _serialisable_value(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
        return value

    def _selection_cfg(self) -> Dict[str, Any]:
        return dict(self.bh_cfg.get("selection", {}))


if __name__ == "__main__":
    print("\n=== Running Base Heuristics ===\n")
    printer.status("TEST", "Base Heuristics initialized", "info")

    class SmokeHeuristics(BaseHeuristics):
        def predict_success_prob(self, task, world_state, method_stats, method_id):
            features = self.extract_base_features(task, world_state, method_stats, method_id)
            return clamp(
                0.55 + 0.25 * features["goal_overlap"]
                + 0.15 * features["method_success_rate"]
                - 0.10 * features["method_failure_rate"],
                0.0,
                1.0,
            )

    task = {
        "id": "task_A",
        "name": "assemble_plan",
        "goal_state": {"assembled": True, "validated": True},
        "creation_time": datetime.now(timezone.utc).isoformat(),
        "deadline": (datetime.now(timezone.utc).timestamp() + 3600),
        "priority": 8,
        "dependencies": ["scan", "prepare"],
    }
    world_state = {"assembled": True, "validated": False, "battery": 82.0, "load": 0.34}
    stats = {
        ("assemble_plan", "safe_method"): {"success": 18, "total": 20},
        ("assemble_plan", "risky_method"): {"success": 4, "total": 10},
    }

    heuristic = SmokeHeuristics(calculations=None)
    features = heuristic.extract_base_features(task, world_state, stats, "safe_method")
    printer.status("TEST", f"Features extracted: {features}", "info")
    assert all(0.0 <= value <= 1.0 for value in features.values())

    best, score = heuristic.select_best_method(
        task, world_state, ["risky_method", "safe_method"], stats
    )
    printer.status("TEST", f"Best method: {best} ({score:.3f})", "success")
    assert best == "safe_method"
    assert 0.0 <= score <= 1.0

    heuristic.clear_feature_cache()
    assert heuristic.extract_temporal_features(task)["deadline_proximity"] >= 0.0

    print("\n=== Test ran successfully ===\n")
