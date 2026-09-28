"""
Feasibility Constraint Heuristic for Explainable Planning

This module implements a *symbolic / deductive* planning heuristic. Unlike the
other SLAI planning heuristics (DecisionTree, GradientBoosting, Reinforcement
Learning, UncertaintyAware, CaseBasedReasoning), it does not primarily learn a
statistical mapping from historical outcomes.

Instead, it evaluates each (task, method) pair against a **structured set of
planning constraints**, produces a decomposed feasibility score, and exposes a
ranked explanation for that score.  The prediction can therefore answer:

    "Will this method succeed on this task?"

and also:

    "WHY is this method predicted to succeed (or fail) — which constraint
     dominates, and by how much would it need to change?"

The constraint set currently evaluated is:

    1. precondition_satisfaction  – world state vs. task preconditions
    2. resource_feasibility       – task requirements vs. world resources
    3. temporal_feasibility       – estimated duration vs. deadline window
    4. dependency_readiness       – are task dependencies satisfied?
    5. capability_match           – method's declared capabilities vs. task
    6. goal_alignment             – current state vs. goal state
    7. historical_reliability     – method success rate (from method_stats)
    8. method_confidence          – sample size / confidence in (7)

The final probability is a logistic function of a weighted combination of
these normalised features.  Weights are hand-tuned by default (pure rule
mode), but can be learned from labeled execution history via a small
pure-Python logistic-regression trainer that mirrors the dependency policy
established by GradientBoostingHeuristic.

Real-world use cases
--------------------
* Robotics: auditable explanation of *why* a manipulation method is unsafe
  before committing, instead of a black-box probability.
* Long-horizon planning: as a symbolic second opinion when statistical
  heuristics disagree, or when training data is sparse.
* Safety-critical systems: hard constraints (preconditions, resources) can
  be configured to *veto* a prediction rather than merely down-weight it.
"""

from __future__ import annotations

import json
import math
import os
import threading

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils.config_loader import get_config_section
from ..utils.base_heuristic import BaseHeuristics
from ..utils.planning_errors import *
from ..utils.planning_helpers import (
    _clamp,
    _logit,
    _sigmoid,
    clamp,
    compute_resource_margin,
    compute_state_distance,
    compute_temporal_margin,
    require_non_empty,
    require_type,
    state_satisfies_goal,
)
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Feasibility Constraint Heuristic")
printer = PrettyPrinter()

MODEL_FILENAME = "fc_heuristic_model.json"
MODEL_SCHEMA = "slai.planning.feasibility_constraint.v1"
_EPS = 1e-12


# Ordering is part of the persisted model contract.  Do not reorder without
# bumping MODEL_SCHEMA and providing a migration path.
FEATURE_NAMES: Tuple[str, ...] = (
    "precondition_satisfaction",
    "resource_feasibility",
    "temporal_feasibility",
    "dependency_readiness",
    "capability_match",
    "goal_alignment",
    "historical_reliability",
    "method_confidence",
)

# Hand-tuned default weights used when the model has not been trained.
# Rationale (see module docstring):
#   - hard constraints (preconditions/resources) dominate
#   - temporal and dependency feasibility matter but are recoverable
#   - historical reliability and method confidence smooth cold-start
DEFAULT_WEIGHTS: Dict[str, float] = {
    "precondition_satisfaction": 2.40,
    "resource_feasibility": 1.80,
    "temporal_feasibility": 1.20,
    "dependency_readiness": 1.60,
    "capability_match": 1.10,
    "goal_alignment": 0.90,
    "historical_reliability": 1.50,
    "method_confidence": 0.70,
}
DEFAULT_BIAS: float = -2.20  # keeps untrained predictions centred near 0.35


class FeasibilityConstraintHeuristic(BaseHeuristics):
    """
    Symbolic feasibility heuristic.

    Public surface
    --------------
    - ``predict_success_prob(...)``              – BaseHeuristics contract
    - ``explain_prediction(...)``                – structured per-constraint report
    - ``train_from_planning_db(...)``            – learn weights from labeled history
    - ``report_feature_importance()``            – learned weights, sorted
    """

    def __init__(
        self,
        *,
        model_path: Optional[str | Path] = None,
        auto_load: bool = True,
    ) -> None:
        super().__init__()

        self.heuristics_config = get_config_section(
            "global_heuristic", config=self.config, default={}
        )
        self.fc_config = get_config_section(
            "feasibility_constraint_heuristic", config=self.config, default={}
        )

        self.random_state = int(self.heuristics_config.get("random_state", 42))
        self.feature_names: List[str] = list(FEATURE_NAMES)

        # Learned parameters (weights + bias).  Initialised to hand-tuned defaults.
        self.weights: Dict[str, float] = {
            name: float(self.fc_config.get("weights", {}).get(name, DEFAULT_WEIGHTS[name]))
            for name in self.feature_names
        }
        self.bias: float = float(self.fc_config.get("bias", DEFAULT_BIAS))

        # Hard-constraint handling.  When enabled, a hard constraint scored
        # below ``hard_constraint_floor`` vetoes the prediction outright.
        self.hard_constraints: List[str] = list(
            self.fc_config.get(
                "hard_constraints",
                ["precondition_satisfaction", "resource_feasibility"],
            )
        )
        self.hard_constraint_floor: float = float(
            self.fc_config.get("hard_constraint_floor", 0.05)
        )
        self.veto_probability: float = float(
            self.fc_config.get("veto_probability", 0.02)
        )

        # Resource buffer used when evaluating resource feasibility.
        self.resource_safety_buffer: float = float(
            self.fc_config.get("resource_safety_buffer", 0.0)
        )
        self.temporal_time_buffer: float = float(
            self.fc_config.get("temporal_time_buffer", 0.0)
        )

        # Persistence.
        self.model_path = self._resolve_model_path(model_path)
        self.trained = False
        self._lock = threading.RLock()

        if auto_load:
            self._load_model()

        logger.info(
            "Feasibility Constraint Heuristic initialized trained=%s features=%d path=%s",
            self.trained,
            len(self.feature_names),
            self.model_path,
        )

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _repo_root() -> Path:
        # .../src/agents/planning/heuristics/feasibility_constraint_heuristic.py
        return Path(__file__).resolve().parents[4]

    def _resolve_model_path(self, override: Optional[str | Path]) -> Path:
        if override is not None:
            return Path(override).expanduser().resolve()
        configured = Path(
            str(self.heuristics_config.get(
                "heuristic_model_path", "src/agents/planning/models/"
            ))
        )
        directory = configured if configured.is_absolute() else self._repo_root() / configured
        directory.mkdir(parents=True, exist_ok=True)
        return (directory / MODEL_FILENAME).resolve()

    # ------------------------------------------------------------------
    # Feature extraction — symbolic, not statistical
    # ------------------------------------------------------------------
    def extract_feasibility_features(
        self,
        task: Any,
        world_state: Dict[str, Any],
        method_stats: Dict[Any, Dict[str, Any]],
        method_id: str,
    ) -> Dict[str, float]:
        """
        Compute the eight symbolic feasibility features in ``[0, 1]``.

        Every feature is derived from *declared* task/method/world semantics,
        not from a learned embedding.  Historical_reliability and
        method_confidence are the only features that touch method statistics,
        and they delegate to ``BaseHeuristics._resolve_method_stats``.
        """
        require_type(world_state, dict, "world_state")
        require_type(method_stats, dict, "method_stats")
        require_non_empty(method_id, "method_id")

        goal_state = self._extract_goal_state(task)

        features: Dict[str, float] = {
            "precondition_satisfaction": self._score_preconditions(task, world_state),
            "resource_feasibility": self._score_resources(task, world_state),
            "temporal_feasibility": self._score_temporal(task),
            "dependency_readiness": self._score_dependencies(task, world_state),
            "capability_match": self._score_capability(task, method_id),
            "goal_alignment": self._score_goal_alignment(goal_state, world_state),
        }

        # Historical reliability and confidence come from the shared resolver.
        stats = self._resolve_method_stats(task, method_stats, method_id)
        features["historical_reliability"] = clamp(
            float(stats.get("success_rate", 0.5)), 0.0, 1.0
        )
        features["method_confidence"] = clamp(
            float(stats.get("confidence", 0.0)), 0.0, 1.0
        )

        # Final safety net: NaN/inf → 0.0, clamp to [0, 1].
        for name, value in list(features.items()):
            numeric = float(value) if math.isfinite(float(value)) else 0.0
            features[name] = clamp(numeric, 0.0, 1.0)
        return features

    # ----- individual symbolic evaluators -----------------------------

    def _score_preconditions(self, task: Any, world_state: Dict[str, Any]) -> float:
        """
        Fraction of declared preconditions satisfied by ``world_state``.

        Supports two precondition formats:

        * ``preconditions: {"key": expected_value, ...}``  — checked against
          the world state using the shared ``state_satisfies_goal`` helper.
        * ``preconditions: [callable, ...]``                — evaluated in a
          sandboxed try/except; each returned truthy value counts as satisfied.

        Returns 1.0 when the task declares no preconditions.
        """
        raw = self._task_value(task, "preconditions", None)
        if raw is None:
            return 1.0
        if isinstance(raw, Mapping):
            if not raw:
                return 1.0
            satisfied = 1.0 if state_satisfies_goal(world_state, dict(raw)) else 0.0
            return satisfied
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if not raw:
                return 1.0
            hits = 0
            for condition in raw:
                if callable(condition):
                    try:
                        if condition(world_state):
                            hits += 1
                    except Exception as exc:
                        logger.debug("Precondition callable raised: %s", exc)
                else:
                    logger.debug("Ignoring non-callable precondition: %r", condition)
            return hits / float(len(raw))
        logger.debug("Unsupported precondition format: %r", type(raw).__name__)
        return 1.0

    def _score_resources(self, task: Any, world_state: Dict[str, Any]) -> float:
        """
        Minimum per-resource margin across declared task requirements.

        Task requirements may be given as a dict under ``requirements`` or
        ``resource_requirements``.  Both ``needed`` (from the task) and
        ``available`` (from the world state) are matched by key.
        """
        raw = (
            self._task_value(task, "resource_requirements", None)
            or self._task_value(task, "requirements", None)
        )
        if not isinstance(raw, Mapping) or not raw:
            return 1.0

        margins: List[float] = []
        for resource, needed in raw.items():
            try:
                need = float(needed)
            except (TypeError, ValueError):
                continue
            available = self._world_numeric(world_state, resource, 1.0)
            margin = compute_resource_margin(
                need, available, safety_buffer=self.resource_safety_buffer
            )
            margins.append(margin)
        if not margins:
            return 1.0
        return clamp(min(margins), 0.0, 1.0)

    def _score_temporal(self, task: Any) -> float:
        """
        Temporal feasibility = margin of estimated duration within the window.

        Uses ``compute_temporal_margin`` from ``planning_helpers`` so the
        semantics stay identical to the rest of the planner.  Returns 1.0 when
        no deadline is declared.
        """
        duration = self._task_numeric(task, "estimated_duration", None)
        if duration is None:
            duration = self._task_numeric(task, "duration", None)
        if duration is None or duration <= 0:
            return 1.0

        deadline = self._coerce_datetime(self._task_value(task, "deadline", None))
        if deadline is None:
            return 1.0

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        seconds_left = (deadline - now).total_seconds()
        if seconds_left <= 0:
            return 0.0
        return clamp(
            compute_temporal_margin(
                total_duration=duration,
                available_time=seconds_left,
                time_buffer=self.temporal_time_buffer,
            ),
            0.0,
            1.0,
        )

    def _score_dependencies(self, task: Any, world_state: Dict[str, Any]) -> float:
        """
        Fraction of declared dependencies already satisfied in ``world_state``.

        A dependency is considered satisfied when the world state contains a
        truthy entry for the dependency's id/name, or when a
        ``completed_dependencies`` list in the world state includes it.
        """
        raw_deps = self._task_value(task, "dependencies", None)
        if not raw_deps:
            return 1.0
        if isinstance(raw_deps, (str, bytes)):
            raw_deps = [raw_deps]
        if not isinstance(raw_deps, Sequence):
            return 1.0

        completed = world_state.get("completed_dependencies") or world_state.get("completed_tasks")
        completed_set = set()
        if isinstance(completed, Sequence) and not isinstance(completed, (str, bytes)):
            completed_set = {str(x) for x in completed}

        hits = 0
        for dep in raw_deps:
            key = str(dep)
            if key in completed_set:
                hits += 1
                continue
            if bool(world_state.get(key, False)):
                hits += 1
                continue
            if bool(world_state.get(f"{key}_completed", False)):
                hits += 1
        return hits / float(len(raw_deps))

    def _score_capability(self, task: Any, method_id: str) -> float:
        """
        Match method capabilities against declared task requirements.

        This is a *symbolic* method-semantics check.  It is intentionally
        conservative: if neither the method nor the task declares capability
        metadata, the feature returns a neutral 0.5 so it neither boosts nor
        penalises the prediction.

        Recognised metadata:

        * ``task["required_capabilities"] : List[str]``
        * ``task["method_capabilities"]   : {method_id: List[str]}``
        * ``task["methods"][method_id]["capabilities"] : List[str]``
        """
        required = self._task_value(task, "required_capabilities", None)
        if not required or not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            return 0.5

        declared: Optional[Sequence[str]] = None
        cap_map = self._task_value(task, "method_capabilities", None)
        if isinstance(cap_map, Mapping):
            candidate = cap_map.get(method_id)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                declared = candidate
        if declared is None:
            methods = self._task_value(task, "methods", None)
            if isinstance(methods, Mapping):
                entry = methods.get(method_id)
                if isinstance(entry, Mapping):
                    candidate = entry.get("capabilities")
                    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                        declared = candidate
        if declared is None:
            return 0.5

        required_set = {str(x) for x in required}
        declared_set = {str(x) for x in declared}
        if not required_set:
            return 1.0
        return len(required_set & declared_set) / float(len(required_set))

    @staticmethod
    def _score_goal_alignment(goal_state: Dict[str, Any], world_state: Dict[str, Any]) -> float:
        if not goal_state:
            return 1.0
        return clamp(1.0 - compute_state_distance(world_state, goal_state), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_success_prob(
        self,
        task: Any,
        world_state: Dict[str, Any],
        method_stats: Dict[Any, Dict[str, Any]],
        method_id: str,
    ) -> float:
        try:
            features = self.extract_feasibility_features(task, world_state, method_stats, method_id)
        except PlanningError:
            raise
        except Exception as exc:
            logger.warning(
                "FC feature extraction failed for method=%s: %s — returning neutral fallback",
                method_id,
                exc,
            )
            return 0.5

        # Hard-constraint veto: if a hard constraint is below the configured
        # floor, predict ``veto_probability`` and skip the sigmoid entirely.
        for name in self.hard_constraints:
            if name in features and features[name] < self.hard_constraint_floor:
                logger.debug("FC veto triggered by %s=%.3f", name, features[name])
                return self.veto_probability

        logit = self.bias + sum(
            self.weights[name] * features[name] for name in self.feature_names
        )
        return clamp(_sigmoid(logit), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def explain_prediction(
        self,
        task: Any,
        world_state: Dict[str, Any],
        method_stats: Dict[Any, Dict[str, Any]],
        method_id: str,
    ) -> Dict[str, Any]:
        """
        Return a structured, JSON-serialisable explanation of the prediction.

        Shape::

            {
              "method_id": ...,
              "probability": ...,
              "hard_constraint_veto": bool,
              "contributions": [
                 {"feature": ..., "value": ..., "weight": ..., "contribution": ...},
                 ...                       # sorted by |contribution|, descending
              ],
              "dominant_positive": {...} or None,
              "dominant_negative": {...} or None,
            }
        """
        features = self.extract_feasibility_features(task, world_state, method_stats, method_id)

        veto = next(
            (
                name for name in self.hard_constraints
                if name in features and features[name] < self.hard_constraint_floor
            ),
            None,
        )

        contributions: List[Dict[str, float]] = []
        for name in self.feature_names:
            value = float(features[name])
            weight = float(self.weights[name])
            contributions.append(
                {
                    "feature": name,
                    "value": value,
                    "weight": weight,
                    "contribution": value * weight,
                }
            )
        contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

        positives = [c for c in contributions if c["contribution"] > 0]
        negatives = [c for c in contributions if c["contribution"] < 0]

        return {
            "method_id": str(method_id),
            "probability": (
                self.veto_probability
                if veto is not None
                else clamp(
                    _sigmoid(
                        self.bias
                        + sum(c["contribution"] for c in contributions)
                    ),
                    0.0,
                    1.0,
                )
            ),
            "bias": self.bias,
            "hard_constraint_veto": veto is not None,
            "veto_constraint": veto,
            "features": dict(features),
            "contributions": contributions,
            "dominant_positive": positives[0] if positives else None,
            "dominant_negative": negatives[0] if negatives else None,
        }

    # ------------------------------------------------------------------
    # Training (pure-Python logistic regression with L2)
    # ------------------------------------------------------------------
    def train(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        *,
        epochs: int = 400,
        learning_rate: float = 0.05,
        l2: float = 1e-3,
        class_weight_balanced: bool = True,
    ) -> Dict[str, Any]:
        """
        Learn feature weights + bias from labeled ``(features, label)`` pairs.

        Uses batch gradient descent on the standard logistic loss with L2
        regularisation.  This is deliberately small and deterministic — the
        model has 8 weights + 1 bias, so training converges quickly and does
        not require the GBDT machinery used by GradientBoostingHeuristic.
        """
        if not X or not y or len(X) != len(y):
            raise FeasibilityConstraintError(
                "Training X/y must be non-empty and have equal length."
            )
        width = len(self.feature_names)
        for row in X:
            if len(row) != width:
                raise FeasibilityConstraintError(
                    f"Training row width {len(row)} != expected {width}."
                )
        labels = {int(v) for v in y}
        if labels != {0, 1}:
            raise FeasibilityConstraintError(
                "Training requires both success (1) and failure (0) labels."
            )

        weights = [self.weights[name] for name in self.feature_names]
        bias = self.bias

        # Optional class balancing.
        sample_weight = [1.0] * len(y)
        if class_weight_balanced:
            negatives = sum(1 for v in y if int(v) == 0)
            positives = len(y) - negatives
            if negatives > 0 and positives > 0:
                w0 = len(y) / (2.0 * negatives)
                w1 = len(y) / (2.0 * positives)
                sample_weight = [w1 if int(v) == 1 else w0 for v in y]

        n = float(len(y))
        for epoch in range(max(1, epochs)):
            grad_w = [0.0] * width
            grad_b = 0.0
            for row, label, w in zip(X, y, sample_weight):
                logit = bias + sum(weights[i] * float(row[i]) for i in range(width))
                prob = _sigmoid(logit)
                error = (prob - float(label)) * w
                for i in range(width):
                    grad_w[i] += error * float(row[i])
                grad_b += error
            for i in range(width):
                grad_w[i] = grad_w[i] / n + l2 * weights[i]
                weights[i] -= learning_rate * grad_w[i]
            bias -= learning_rate * (grad_b / n)
            if epoch % 100 == 0:
                logger.debug("FC training epoch=%d bias=%.4f w0=%.4f", epoch, bias, weights[0])

        self.weights = {name: float(w) for name, w in zip(self.feature_names, weights)}
        self.bias = float(bias)
        self.trained = True
        self._save_model()
        return {"weights": dict(self.weights), "bias": self.bias}

    def train_from_planning_db(
        self,
        *,
        db_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
        minimum_total_samples: int = 30,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper that builds features directly from a planning DB.

        Delegates DB loading and label normalisation to the shared base
        helpers, and reuses the same causal-prior method-stat strategy as
        GradientBoostingHeuristic so the two heuristics see consistent inputs.
        """
        with self._lock:
            if output_path is not None:
                self.model_path = Path(output_path).expanduser().resolve()

            resolved_db = (
                Path(db_path).expanduser().resolve()
                if db_path is not None
                else self._resolve_planning_db_path()
            )
            data = self._load_planning_db(resolved_db)
            X, y = self._build_training_dataset(data)
            if len(y) < minimum_total_samples:
                raise FeasibilityConstraintError(
                    f"Only {len(y)} labeled records available; "
                    f"at least {minimum_total_samples} required to train FC."
                )
            result = self.train(X, y)
            logger.info(
                "FC training complete records=%d bias=%.4f model=%s",
                len(y),
                self.bias,
                self.model_path,
            )
            return {
                "model_path": str(self.model_path),
                "records": len(y),
                "weights": result["weights"],
                "bias": result["bias"],
                "feature_importance": self.report_feature_importance(),
            }

    def _resolve_planning_db_path(self) -> Path:
        raw = str(self.heuristics_config.get("planning_db_path", "templates/planning_db.json"))
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate.resolve()
        return (Path(__file__).resolve().parents[1] / candidate).resolve()

    def _load_planning_db(self, path: Path) -> Dict[str, Any]:
        if not path.is_file():
            raise FeasibilityConstraintError(f"Planning database does not exist: {path}")
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise FeasibilityConstraintError(
                f"Failed to read planning database {path}: {exc}"
            ) from exc

    def _build_training_dataset(
        self, data: Mapping[str, Any]
    ) -> Tuple[List[List[float]], List[int]]:
        tasks = data.get("tasks") or []
        states = data.get("world_states") or []
        if not isinstance(tasks, list) or not isinstance(states, list):
            raise FeasibilityConstraintError("planning_db tasks/world_states must be lists.")
        if len(tasks) != len(states):
            raise FeasibilityConstraintError(
                "planning_db tasks/world_states length mismatch."
            )

        running_stats: Dict[Tuple[str, str], Dict[str, int]] = {}
        X: List[List[float]] = []
        y: List[int] = []

        for task, state in zip(tasks, states):
            if not isinstance(task, Mapping) or not isinstance(state, Mapping):
                continue
            method_id = str(task.get("selected_method") or "").strip()
            label = self._outcome_to_label(task.get("outcome"))
            if not method_id or label is None:
                continue

            try:
                features = self.extract_feasibility_features(
                    task, dict(state), running_stats, method_id
                )
            except PlanningError as exc:
                logger.debug("Skipping record during FC training: %s", exc)
                continue

            X.append([features[name] for name in self.feature_names])
            y.append(label)

            # Causal prior update: only after the row has been consumed.
            key = (str(task.get("name") or ""), method_id)
            stat = running_stats.setdefault(key, {"success": 0, "total": 0})
            stat["total"] += 1
            stat["success"] += int(label == 1)

        return X, y

    @staticmethod
    def _outcome_to_label(outcome: Any) -> Optional[int]:
        value = str(outcome or "").strip().lower()
        if value in {"success", "successful", "succeeded", "completed", "ok"}:
            return 1
        if value in {"failure", "failed", "error", "cancelled", "timeout"}:
            return 0
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_model(self) -> None:
        payload = {
            "schema": MODEL_SCHEMA,
            "feature_names": list(self.feature_names),
            "weights": dict(self.weights),
            "bias": self.bias,
            "hard_constraints": list(self.hard_constraints),
            "hard_constraint_floor": self.hard_constraint_floor,
            "veto_probability": self.veto_probability,
        }
        tmp = self.model_path.with_suffix(self.model_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.model_path)
        logger.info("FC model saved to %s", self.model_path)

    def _load_model(self) -> None:
        with self._lock:
            if not self.model_path.is_file():
                logger.info("No trained FC model at %s — using default weights", self.model_path)
                return
            try:
                with self.model_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                if payload.get("schema") != MODEL_SCHEMA:
                    raise FeasibilityConstraintError(
                        f"Unsupported FC model schema: {payload.get('schema')!r}"
                    )
                if list(payload.get("feature_names", [])) != self.feature_names:
                    raise FeasibilityConstraintError(
                        "FC model feature contract mismatch with current code."
                    )
                self.weights = {name: float(payload["weights"][name]) for name in self.feature_names}
                self.bias = float(payload["bias"])
                self.hard_constraints = list(payload.get("hard_constraints", self.hard_constraints))
                self.hard_constraint_floor = float(
                    payload.get("hard_constraint_floor", self.hard_constraint_floor)
                )
                self.veto_probability = float(
                    payload.get("veto_probability", self.veto_probability)
                )
                self.trained = True
                logger.info("Loaded FC model from %s", self.model_path)
            except Exception as exc:
                logger.error("Failed to load FC model: %s", exc)
                self.trained = False

    def report_feature_importance(self) -> List[Tuple[str, float]]:
        """Return learned weights sorted by absolute magnitude."""
        return sorted(
            self.weights.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Small internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _world_numeric(world_state: Mapping[str, Any], key: str, default: float = 1.0) -> float:
        try:
            return float(world_state.get(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _task_numeric(task: Any, key: str, default: Optional[float] = None) -> Optional[float]:
        value = BaseHeuristics._task_value(task, key, default)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


# ===========================================================================
# Structured error for this heuristic
# ===========================================================================

class FeasibilityConstraintError(PlanningError):
    """Raised when the FC heuristic cannot build or train its model."""

    _default_recovery_hints = [
        "verify_planning_db_schema",
        "enable_rule_mode_fallback",
        "inspect_fc_feature_contract",
    ]


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == "__main__":
    printer.status("INIT", "Feasibility Constraint Heuristic loaded", "success")

    heuristic = FeasibilityConstraintHeuristic(auto_load=True)

    task = {
        "id": "nav_1",
        "name": "navigate_to_room",
        "priority": 0.8,
        "goal_state": {"location": "room_1"},
        "preconditions": {"battery_level": 0.9},
        "dependencies": ["power_on"],
        "required_capabilities": ["navigation", "obstacle_avoidance"],
        "method_capabilities": {"A*": ["navigation", "obstacle_avoidance"]},
        "estimated_duration": 120.0,
    }
    world_state = {
        "location": "corridor",
        "battery_level": 0.9,
        "cpu_available": 0.7,
        "memory_available": 0.6,
        "completed_dependencies": ["power_on"],
    }
    stats = {
        ("navigate_to_room", "A*"): {"success": 8, "total": 10},
        ("navigate_to_room", "RRT"): {"success": 5, "total": 10},
    }

    print("\n* * * * * Phase 1 — Probability Predictions * * * * *\n")
    for method_id in ("A*", "RRT"):
        prob = heuristic.predict_success_prob(task, world_state, stats, method_id)
        printer.pretty(f"{method_id}: P(success) = {prob:.3f}", "", "info")

    print("\n* * * * * Phase 2 — Explanation for A* * * * *\n")
    explanation = heuristic.explain_prediction(task, world_state, stats, "A*")
    printer.pretty("Probability:", f"{explanation['probability']:.3f}", "info")
    printer.pretty("Hard veto:", explanation["hard_constraint_veto"], "info")
    for c in explanation["contributions"]:
        printer.pretty(
            f"  {c['feature']:<26} value={c['value']:.3f} w={c['weight']:+.3f} "
            f"→ contribution={c['contribution']:+.3f}",
            "",
            "info",
        )
    if explanation["dominant_positive"]:
        printer.pretty("Dominant positive:", explanation["dominant_positive"]["feature"], "success")
    if explanation["dominant_negative"]:
        printer.pretty("Dominant negative:", explanation["dominant_negative"]["feature"], "warning")

    print("\n* * * * * Phase 3 — Veto Demonstration * * * * *\n")
    bad_state = dict(world_state)
    bad_state["battery_level"] = 0.05  # below precondition threshold
    prob = heuristic.predict_success_prob(task, bad_state, stats, "A*")
    printer.pretty(f"A* with violated precondition: {prob:.3f}", "", "warning")

    print("\n=== Feasibility Constraint Heuristic smoke test complete ===\n")