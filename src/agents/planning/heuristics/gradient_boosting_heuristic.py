"""
SLAI Gradient-Boosted Planning Success Heuristic.

Dependency policy
-----------------
This module uses only the Python standard library plus SLAI's own modules.
It deliberately removes the previous direct NumPy/scikit-learn/joblib runtime
requirements.  SLAI's planning configuration loader may itself use PyYAML; that
is an existing SLAI dependency, not a dependency introduced by this heuristic.

Model objective
---------------
Estimate P(success | task, method, world_state, historical_method_statistics).
The model is a binary gradient-boosted decision-tree ensemble trained with the
logistic objective.  It consumes the canonical, bounded feature map produced by
BaseHeuristics.extract_base_features(), so feature semantics remain centralized
in SLAI rather than being duplicated here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import threading

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils.config_loader import get_config_section
from ..utils.base_heuristic import BaseHeuristics
from ..utils.planning_errors import *
from ..utils.planning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Gradient Boosting Heuristic")
printer = PrettyPrinter()

MODEL_SCHEMA = "slai.planning.gradient_boosted_success.v1"
MODEL_FILENAME = "gb_planning_success_v1.json"
_EPS = 1e-12

# BaseHeuristics is the source of truth for these feature definitions.
FEATURE_NAMES: Tuple[str, ...] = (
    "task_depth",
    "goal_overlap",
    "goal_distance",
    "goal_satisfied",
    "method_success_rate",
    "method_failure_rate",
    "method_confidence",
    "state_diversity",
    "priority",
    "dependency_load",
    "task_complexity",
    "risk_score",
    "resource_margin",
    "temporal_margin",
    "time_since_creation",
    "deadline_proximity",
    "urgency",
)



@dataclass
class TreeNode:
    """A compact binary regression-tree node used by the boosted ensemble."""

    value: float
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    gain: float = 0.0
    samples: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None

    def predict(self, row: Sequence[float]) -> float:
        node: TreeNode = self
        while not node.is_leaf:
            assert node.feature_index is not None
            assert node.threshold is not None
            child = node.left if float(row[node.feature_index]) <= node.threshold else node.right
            if child is None:
                return node.value
            node = child
        return node.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "feature_index": self.feature_index,
            "threshold": self.threshold,
            "gain": self.gain,
            "samples": self.samples,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TreeNode":
        node = cls(
            value=float(raw["value"]),
            feature_index=(None if raw.get("feature_index") is None else int(raw["feature_index"])),
            threshold=(None if raw.get("threshold") is None else float(raw["threshold"])),
            gain=float(raw.get("gain", 0.0)),
            samples=int(raw.get("samples", 0)),
        )
        if raw.get("left") is not None:
            node.left = cls.from_dict(raw["left"])
        if raw.get("right") is not None:
            node.right = cls.from_dict(raw["right"])
        return node


class GradientBoostedBinaryClassifier:
    """
    Pure-Python binary gradient-boosted decision-tree classifier.

    The implementation uses first/second derivatives of binary logistic loss.
    Each leaf uses a regularized Newton step:

        leaf_value = sum(gradient) / (sum(hessian) + lambda)

    Candidate split gain follows the same second-order objective and is evaluated
    only on the current training subsample.  This is a compact GBDT implementation,
    not a reimplementation of scikit-learn and not intended for massive datasets.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        min_samples_split: int = 15,
        min_samples_leaf: int = 5,
        subsample: float = 0.8,
        l2_leaf_regularization: float = 1.0,
        min_split_gain: float = 1e-8,
        max_bins: int = 32,
        random_state: int = 42,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if not 0.0 < subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        if l2_leaf_regularization < 0.0:
            raise ValueError("l2_leaf_regularization must be >= 0")
        if max_bins < 2:
            raise ValueError("max_bins must be >= 2")

        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.l2_leaf_regularization = float(l2_leaf_regularization)
        self.min_split_gain = float(min_split_gain)
        self.max_bins = int(max_bins)
        self.random_state = int(random_state)

        self.base_score: float = 0.0
        self.trees: List[TreeNode] = []
        self.feature_importances_: List[float] = []
        self.best_iteration_: int = 0
        self.best_validation_loss_: Optional[float] = None
        self.training_history_: List[Dict[str, float]] = []
        self._rng = random.Random(self.random_state)

    @staticmethod
    def _validate_matrix(X: Sequence[Sequence[float]], y: Sequence[int]) -> int:
        if not X or not y or len(X) != len(y):
            raise GradientBoostingModelError("Training X/y must be non-empty and have equal length.")
        width = len(X[0])
        if width <= 0:
            raise GradientBoostingModelError("Training matrix has zero features.")
        for row in X:
            if len(row) != width:
                raise GradientBoostingModelError("Training matrix contains inconsistent row widths.")
            if any(not math.isfinite(float(value)) for value in row):
                raise GradientBoostingModelError("Training matrix contains a non-finite value.")
        labels = {int(v) for v in y}
        if not labels.issubset({0, 1}) or labels != {0, 1}:
            raise GradientBoostingModelError("Binary training requires both labels 0 and 1.")
        return width

    @staticmethod
    def _class_weights(y: Sequence[int], balanced: bool) -> List[float]:
        if not balanced:
            return [1.0, 1.0]
        negative = sum(1 for value in y if int(value) == 0)
        positive = len(y) - negative
        if negative == 0 or positive == 0:
            raise GradientBoostingModelError("Both classes are required for balanced weighting.")
        total = float(len(y))
        return [total / (2.0 * negative), total / (2.0 * positive)]

    def _raw_predict_row(self, row: Sequence[float], *, trees: Optional[Sequence[TreeNode]] = None) -> float:
        ensemble = self.trees if trees is None else trees
        return self.base_score + self.learning_rate * sum(tree.predict(row) for tree in ensemble)

    def predict_success_probability(self, row: Sequence[float]) -> float:
        return _sigmoid(self._raw_predict_row(row))

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        result: List[List[float]] = []
        for row in X:
            success = self.predict_success_probability(row)
            result.append([1.0 - success, success])
        return result

    def predict(self, X: Sequence[Sequence[float]], threshold: float = 0.5) -> List[int]:
        return [1 if self.predict_success_probability(row) >= threshold else 0 for row in X]

    @staticmethod
    def _binary_log_loss(y: Sequence[int], probabilities: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
        if len(y) != len(probabilities) or not y:
            raise GradientBoostingModelError("Cannot compute log loss on empty/mismatched inputs.")
        total = 0.0
        weight_sum = 0.0
        for index, (target, probability) in enumerate(zip(y, probabilities)):
            weight = 1.0 if weights is None else float(weights[index])
            p = _clamp(probability, 1e-15, 1.0 - 1e-15)
            total += weight * (-(target * math.log(p) + (1 - target) * math.log(1.0 - p)))
            weight_sum += weight
        return total / max(weight_sum, _EPS)

    def _candidate_thresholds(self, values: Sequence[float]) -> List[float]:
        unique = sorted(set(float(v) for v in values))
        if len(unique) <= 1:
            return []
        if len(unique) - 1 <= self.max_bins:
            return [(unique[i] + unique[i + 1]) * 0.5 for i in range(len(unique) - 1)]

        thresholds: List[float] = []
        step = (len(unique) - 1) / float(self.max_bins)
        seen: set[float] = set()
        for bin_index in range(1, self.max_bins + 1):
            left_index = min(len(unique) - 2, max(0, int(round(bin_index * step)) - 1))
            threshold = (unique[left_index] + unique[left_index + 1]) * 0.5
            if threshold not in seen:
                thresholds.append(threshold)
                seen.add(threshold)
        return thresholds

    def _score_term(self, gradient_sum: float, hessian_sum: float) -> float:
        return (gradient_sum * gradient_sum) / (hessian_sum + self.l2_leaf_regularization + _EPS)

    def _build_tree(
        self,
        X: Sequence[Sequence[float]],
        gradients: Sequence[float],
        hessians: Sequence[float],
        indices: Sequence[int],
        depth: int,
        feature_gain: List[float],
    ) -> TreeNode:
        gradient_sum = sum(gradients[i] for i in indices)
        hessian_sum = sum(hessians[i] for i in indices)
        leaf_value = gradient_sum / (hessian_sum + self.l2_leaf_regularization + _EPS)
        node = TreeNode(value=leaf_value, samples=len(indices))

        if depth >= self.max_depth or len(indices) < self.min_samples_split:
            return node

        parent_score = self._score_term(gradient_sum, hessian_sum)
        best_gain = self.min_split_gain
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None
        best_left: Optional[List[int]] = None
        best_right: Optional[List[int]] = None
        width = len(X[0])

        for feature_index in range(width):
            thresholds = self._candidate_thresholds([X[i][feature_index] for i in indices])
            for threshold in thresholds:
                left = [i for i in indices if float(X[i][feature_index]) <= threshold]
                right = [i for i in indices if float(X[i][feature_index]) > threshold]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue

                gl = sum(gradients[i] for i in left)
                hl = sum(hessians[i] for i in left)
                gr = gradient_sum - gl
                hr = hessian_sum - hl
                gain = 0.5 * (self._score_term(gl, hl) + self._score_term(gr, hr) - parent_score)

                if gain > best_gain + 1e-15:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
                    best_left = left
                    best_right = right

        if best_feature is None or best_left is None or best_right is None or best_threshold is None:
            return node

        node.feature_index = best_feature
        node.threshold = best_threshold
        node.gain = best_gain
        feature_gain[best_feature] += best_gain
        node.left = self._build_tree(X, gradients, hessians, best_left, depth + 1, feature_gain)
        node.right = self._build_tree(X, gradients, hessians, best_right, depth + 1, feature_gain)
        return node

    def fit(
        self,
        X_train: Sequence[Sequence[float]],
        y_train: Sequence[int],
        *,
        X_validation: Optional[Sequence[Sequence[float]]] = None,
        y_validation: Optional[Sequence[int]] = None,
        balanced_class_weight: bool = True,
        early_stopping_rounds: int = 20,
        min_delta: float = 1e-5,
    ) -> "GradientBoostedBinaryClassifier":
        width = self._validate_matrix(X_train, y_train)
        if X_validation is not None or y_validation is not None:
            if X_validation is None or y_validation is None:
                raise GradientBoostingModelError("Validation features and labels must be supplied together.")
            if not X_validation or len(X_validation) != len(y_validation):
                raise GradientBoostingModelError("Validation X/y must be non-empty and equal length.")
            if any(len(row) != width for row in X_validation):
                raise GradientBoostingModelError("Validation feature width differs from training data.")

        class_weight = self._class_weights(y_train, balanced_class_weight)
        sample_weights = [class_weight[int(label)] for label in y_train]

        # Base prior is deliberately unweighted so probability calibration starts
        # from the observed empirical success frequency.
        positive_rate = sum(int(value) for value in y_train) / float(len(y_train))
        self.base_score = _logit(positive_rate)
        raw_scores = [self.base_score for _ in y_train]
        validation_raw = (
            [self.base_score for _ in y_validation]
            if X_validation is not None and y_validation is not None
            else None
        )

        self.trees = []
        self.training_history_ = []
        feature_gain = [0.0 for _ in range(width)]
        best_trees: List[TreeNode] = []
        best_feature_gain = list(feature_gain)
        best_loss = float("inf")
        best_iteration = 0
        stale_rounds = 0

        for iteration in range(1, self.n_estimators + 1):
            probabilities = [_sigmoid(raw) for raw in raw_scores]
            gradients = [
                sample_weights[i] * (int(y_train[i]) - probabilities[i])
                for i in range(len(y_train))
            ]
            hessians = [
                sample_weights[i] * max(probabilities[i] * (1.0 - probabilities[i]), 1e-6)
                for i in range(len(y_train))
            ]

            population = list(range(len(y_train)))
            sample_count = max(self.min_samples_split, int(round(len(population) * self.subsample)))
            sample_count = min(len(population), sample_count)
            indices = self._rng.sample(population, sample_count) if sample_count < len(population) else population

            tree = self._build_tree(X_train, gradients, hessians, indices, 0, feature_gain)
            self.trees.append(tree)

            for i, row in enumerate(X_train):
                raw_scores[i] += self.learning_rate * tree.predict(row)

            train_prob = [_sigmoid(raw) for raw in raw_scores]
            train_loss = self._binary_log_loss(y_train, train_prob)
            record: Dict[str, float] = {
                "iteration": float(iteration),
                "train_log_loss": train_loss,
            }

            monitored_loss = train_loss
            if X_validation is not None and y_validation is not None and validation_raw is not None:
                for i, row in enumerate(X_validation):
                    validation_raw[i] += self.learning_rate * tree.predict(row)
                validation_prob = [_sigmoid(raw) for raw in validation_raw]
                validation_loss = self._binary_log_loss(y_validation, validation_prob)
                record["validation_log_loss"] = validation_loss
                monitored_loss = validation_loss

            self.training_history_.append(record)

            if iteration == 1 or iteration % 10 == 0:
                logger.info(
                    "GB training iteration=%d/%d train_log_loss=%.6f monitored_log_loss=%.6f",
                    iteration,
                    self.n_estimators,
                    train_loss,
                    monitored_loss,
                )

            if monitored_loss < best_loss - min_delta:
                best_loss = monitored_loss
                best_iteration = iteration
                best_trees = list(self.trees)
                best_feature_gain = list(feature_gain)
                stale_rounds = 0
            else:
                stale_rounds += 1
                if early_stopping_rounds > 0 and stale_rounds >= early_stopping_rounds:
                    logger.info(
                        "GB early stopping at iteration=%d; best_iteration=%d best_loss=%.6f",
                        iteration,
                        best_iteration,
                        best_loss,
                    )
                    break

        if best_trees:
            self.trees = best_trees
            feature_gain = best_feature_gain
        self.best_iteration_ = best_iteration or len(self.trees)
        self.best_validation_loss_ = best_loss if math.isfinite(best_loss) else None

        gain_sum = sum(feature_gain)
        self.feature_importances_ = (
            [gain / gain_sum for gain in feature_gain]
            if gain_sum > 0.0
            else [0.0 for _ in feature_gain]
        )
        return self

    def to_model_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": "binary_logistic_gradient_boosted_trees",
            "base_score": self.base_score,
            "learning_rate": self.learning_rate,
            "best_iteration": self.best_iteration_,
            "best_validation_loss": self.best_validation_loss_,
            "feature_importances": list(self.feature_importances_),
            "parameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "subsample": self.subsample,
                "l2_leaf_regularization": self.l2_leaf_regularization,
                "min_split_gain": self.min_split_gain,
                "max_bins": self.max_bins,
                "random_state": self.random_state,
            },
            "trees": [tree.to_dict() for tree in self.trees],
        }

    @classmethod
    def from_model_dict(cls, raw: Mapping[str, Any]) -> "GradientBoostedBinaryClassifier":
        params = dict(raw.get("parameters", {}))
        model = cls(
            n_estimators=int(params.get("n_estimators", max(1, len(raw.get("trees", []))))),
            learning_rate=float(raw.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 4)),
            min_samples_split=int(params.get("min_samples_split", 15)),
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            subsample=float(params.get("subsample", 0.8)),
            l2_leaf_regularization=float(params.get("l2_leaf_regularization", 1.0)),
            min_split_gain=float(params.get("min_split_gain", 1e-8)),
            max_bins=int(params.get("max_bins", 32)),
            random_state=int(params.get("random_state", 42)),
        )
        model.base_score = float(raw["base_score"])
        model.trees = [TreeNode.from_dict(item) for item in raw.get("trees", [])]
        model.best_iteration_ = int(raw.get("best_iteration", len(model.trees)))
        best_loss = raw.get("best_validation_loss")
        model.best_validation_loss_ = None if best_loss is None else float(best_loss)
        model.feature_importances_ = [float(v) for v in raw.get("feature_importances", [])]
        return model


class GradientBoostingHeuristic(BaseHeuristics):
    """SLAI planning heuristic backed by a pure-Python GBDT probability model."""

    def __init__(self, *, model_path: Optional[str | Path] = None, auto_load: bool = True) -> None:
        super().__init__()

        self.heuristics_config = get_config_section(
            "global_heuristic", config=self.config, default={}
        )
        self.gb_config = get_config_section(
            "gradient_boosting_heuristic", config=self.config, default={}
        )

        self.n_estimators = int(self.gb_config.get("n_estimators", 200))
        self.learning_rate = float(self.gb_config.get("learning_rate", 0.05))
        self.subsample = float(self.gb_config.get("subsample", 0.8))
        self.early_stopping_rounds = int(self.gb_config.get("early_stopping_rounds", 20) or 0)
        self.validation_fraction = float(self.gb_config.get("validation_fraction", 0.2))
        self.max_depth = int(self.heuristics_config.get("max_depth", 8))
        self.min_samples_split = int(self.heuristics_config.get("min_samples_split", 15))
        self.random_state = int(self.heuristics_config.get("random_state", 42))
        self.class_weight = str(self.heuristics_config.get("class_weight", "balanced"))

        self.feature_names: List[str] = list(FEATURE_NAMES)
        self.model: Optional[GradientBoostedBinaryClassifier] = None
        self.trained = False
        self.feature_importances_: List[float] = [0.0 for _ in self.feature_names]
        self._lock = threading.RLock()

        self.planning_db_path = self._resolve_planning_db_path(
            self.heuristics_config.get("planning_db_path", "templates/planning_db.json")
        )
        self.model_path = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else self._resolve_model_path()
        )

        if auto_load:
            self._load_model()

        logger.info(
            "Gradient Boosting Heuristic initialized model_path=%s trained=%s features=%d",
            self.model_path,
            self.trained,
            len(self.feature_names),
        )

    @staticmethod
    def _repo_root() -> Path:
        # .../src/agents/planning/heuristics/gradient_boosting_heuristic.py -> repository root
        return Path(__file__).resolve().parents[4]

    def _resolve_planning_db_path(self, configured: Any) -> Path:
        raw = Path(str(configured or "templates/planning_db.json"))
        if raw.is_absolute():
            return raw.resolve()
        # Existing planning config paths are relative to src/agents/planning/.
        return (Path(__file__).resolve().parents[1] / raw).resolve()

    def _resolve_model_path(self) -> Path:
        configured = Path(str(self.heuristics_config.get("heuristic_model_path", "src/agents/planning/models/")))
        directory = configured if configured.is_absolute() else self._repo_root() / configured
        directory.mkdir(parents=True, exist_ok=True)
        return (directory / MODEL_FILENAME).resolve()

    def extract_features(
        self,
        task: Any,
        world_state: Dict[str, Any],
        method_stats: Dict[Any, Dict[str, Any]],
        method_id: Optional[str] = None,
    ) -> List[float]:
        resolved_method = str(
            method_id if method_id is not None else self._task_value(task, "selected_method", "")
        ).strip()
        if not resolved_method:
            raise GradientBoostingModelError("Cannot extract planning features without a method_id.")

        feature_map = self.extract_base_features(
            task=task,
            world_state=world_state,
            method_stats=method_stats,
            method_id=resolved_method,
        )
        missing = [name for name in self.feature_names if name not in feature_map]
        if missing:
            raise GradientBoostingModelError(
                f"BaseHeuristics feature contract is missing required features: {missing}"
            )
        vector = [float(feature_map[name]) for name in self.feature_names]
        if any(not math.isfinite(value) for value in vector):
            raise GradientBoostingModelError("Feature extraction produced a non-finite value.")
        return vector

    def predict_success_prob(
        self,
        task: Any,
        world_state: Dict[str, Any],
        method_stats: Dict[Any, Dict[str, Any]],
        method_id: str,
    ) -> float:
        if not self._ensure_model_ready():
            return self._fallback_probability(task, method_stats, method_id)

        assert self.model is not None
        try:
            features = self.extract_features(task, world_state, method_stats, method_id)
            probability = self.model.predict_success_probability(features)
            return _clamp(probability, 0.0, 1.0)
        except Exception as exc:
            logger.exception("Gradient Boosting prediction failed method_id=%s: %s", method_id, exc)
            return self._fallback_probability(task, method_stats, method_id)

    def _fallback_probability(
        self,
        task: Any,
        method_stats: Dict[Any, Dict[str, Any]],
        method_id: str,
    ) -> float:
        try:
            stats = self._resolve_method_stats(task, method_stats, method_id)
            return _clamp(float(stats.get("success_rate", 0.5)), 0.0, 1.0)
        except Exception:
            return 0.5

    def _ensure_model_ready(self) -> bool:
        if self.trained and self.model is not None:
            return True
        with self._lock:
            if self.trained and self.model is not None:
                return True
            if self.model_path.is_file():
                self._load_model()
            return self.trained and self.model is not None

    def _load_model(self) -> None:
        with self._lock:
            if not self.model_path.is_file():
                logger.info("No trained Gradient Boosting model found at %s", self.model_path)
                return
            try:
                with self.model_path.open("r", encoding="utf-8") as handle:
                    artifact = json.load(handle)
                self._verify_artifact(artifact)
                artifact_features = list(artifact.get("feature_names", []))
                if artifact_features != self.feature_names:
                    raise GradientBoostingModelError(
                        "Model feature contract differs from current BaseHeuristics feature contract."
                    )
                self.model = GradientBoostedBinaryClassifier.from_model_dict(artifact["model"])
                self.feature_importances_ = list(self.model.feature_importances_)
                self.trained = True
                logger.info(
                    "Loaded Gradient Boosting planning-success model trees=%d schema=%s",
                    len(self.model.trees),
                    artifact.get("schema"),
                )
            except Exception as exc:
                self.model = None
                self.trained = False
                logger.exception("Failed to load Gradient Boosting model %s: %s", self.model_path, exc)

    @staticmethod
    def _verify_artifact(artifact: Mapping[str, Any]) -> None:
        if artifact.get("schema") != MODEL_SCHEMA:
            raise GradientBoostingModelError(
                f"Unsupported Gradient Boosting model schema: {artifact.get('schema')!r}"
            )
        integrity = artifact.get("integrity")
        if not isinstance(integrity, Mapping) or not integrity.get("sha256"):
            raise GradientBoostingModelError("Model artifact is missing SHA-256 integrity metadata.")
        core = dict(artifact)
        core.pop("integrity", None)
        actual = _sha256_payload(core)
        if actual != str(integrity["sha256"]):
            raise GradientBoostingModelError("Model artifact integrity verification failed.")

    def save_model(
        self,
        *,
        source_sha256: Optional[str] = None,
        training_metadata: Optional[Mapping[str, Any]] = None,
        evaluation: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        if self.model is None or not self.trained:
            raise GradientBoostingModelError("Cannot save an untrained Gradient Boosting model.")

        core: Dict[str, Any] = {
            "schema": MODEL_SCHEMA,
            "created_at": _utc_now_iso(),
            "model_type": "planning_task_method_success_probability",
            "target": "outcome == success",
            "feature_source": "BaseHeuristics.extract_base_features",
            "feature_names": list(self.feature_names),
            "feature_schema_sha256": _sha256_payload(list(self.feature_names)),
            "source_sha256": source_sha256,
            "training": dict(training_metadata or {}),
            "evaluation": dict(evaluation or {}),
            "model": self.model.to_model_dict(),
        }
        artifact = dict(core)
        artifact["integrity"] = {"sha256": _sha256_payload(core)}
        _atomic_write_json(self.model_path, artifact)
        self._verify_artifact(artifact)
        logger.info("Saved Gradient Boosting planning-success model to %s", self.model_path)
        return self.model_path

    def load_planning_db(self, path: Optional[str | Path] = None) -> Dict[str, Any]:
        db_path = Path(path).expanduser().resolve() if path is not None else self.planning_db_path
        if not db_path.is_file():
            raise GradientBoostingModelError(f"Planning database does not exist: {db_path}")
        try:
            with db_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise GradientBoostingModelError(f"Failed to read planning database {db_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise GradientBoostingModelError("Planning database root must be a JSON object.")
        tasks = data.get("tasks")
        states = data.get("world_states")
        if not isinstance(tasks, list) or not isinstance(states, list):
            raise GradientBoostingModelError("Planning database requires list fields 'tasks' and 'world_states'.")
        if len(tasks) != len(states):
            raise GradientBoostingModelError(
                f"Planning database task/state misalignment: tasks={len(tasks)} world_states={len(states)}"
            )
        return data

    @staticmethod
    def _outcome_to_label(outcome: Any) -> Optional[int]:
        value = str(outcome or "").strip().lower()
        if value in {"success", "successful", "succeeded", "completed", "complete", "ok"}:
            return 1
        if value in {"failure", "failed", "error", "cancelled", "canceled", "timeout"}:
            return 0
        return None

    @staticmethod
    def _timestamp_key(task: Mapping[str, Any], original_index: int) -> Tuple[str, int]:
        raw = task.get("timestamp") or task.get("completed_at") or task.get("creation_time") or ""
        return (str(raw), original_index)

    def build_training_dataset(
        self,
        data: Mapping[str, Any],
    ) -> Tuple[List[List[float]], List[int], List[Dict[str, Any]]]:
        """
        Build leakage-resistant examples from the planning database.

        Method reliability features are generated causally.  Each example sees
        only outcomes from earlier records for the same task/method pair; the
        current row and later rows are excluded from its method statistics.
        """
        tasks = list(data.get("tasks", []))
        states = list(data.get("world_states", []))
        ordered = sorted(range(len(tasks)), key=lambda i: self._timestamp_key(tasks[i], i))

        running_stats: Dict[Tuple[str, str], Dict[str, int]] = {}
        X: List[List[float]] = []
        y: List[int] = []
        records: List[Dict[str, Any]] = []

        for original_index in ordered:
            task = tasks[original_index]
            state = states[original_index]
            if not isinstance(task, dict) or not isinstance(state, dict):
                logger.warning("Skipping non-mapping planning record index=%d", original_index)
                continue

            method_id = str(task.get("selected_method") or "").strip()
            label = self._outcome_to_label(task.get("outcome"))
            task_name = str(task.get("name") or "").strip()
            if not method_id or not task_name or label is None:
                logger.debug(
                    "Skipping untrainable planning record index=%d task=%r method=%r outcome=%r",
                    original_index,
                    task_name,
                    method_id,
                    task.get("outcome"),
                )
                continue

            # BaseHeuristics accepts tuple keys such as (task_name, method_id).
            features = self.extract_features(task, state, running_stats, method_id)
            X.append(features)
            y.append(label)
            records.append(
                {
                    "original_index": original_index,
                    "task_id": task.get("id"),
                    "task_name": task_name,
                    "method_id": method_id,
                    "timestamp": task.get("timestamp"),
                    "label": label,
                }
            )

            key = (task_name, method_id)
            stat = running_stats.setdefault(key, {"success": 0, "total": 0})
            stat["total"] += 1
            stat["success"] += int(label == 1)

        return X, y, records

    @staticmethod
    def _temporal_split(
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        records: Sequence[Mapping[str, Any]],
        *,
        validation_fraction: float,
        test_fraction: float,
        minimum_total_samples: int = 30,
    ) -> Dict[str, Any]:
        n = len(y)
        if n < minimum_total_samples:
            raise GradientBoostingModelError(
                f"Only {n} labelled planning records are available; at least {minimum_total_samples} "
                "are required for a train/validation/test model. The bundled planning_db template "
                "is demonstration data, not sufficient training evidence."
            )
        if not 0.0 < validation_fraction < 0.5 or not 0.0 < test_fraction < 0.5:
            raise GradientBoostingModelError("Validation and test fractions must each be in (0, 0.5).")
        if validation_fraction + test_fraction >= 0.5:
            raise GradientBoostingModelError("Validation + test fractions must be < 0.5.")

        n_test = max(1, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * validation_fraction)))
        n_train = n - n_val - n_test
        if n_train < 10:
            raise GradientBoostingModelError("Training partition is too small after temporal splitting.")

        train_slice = slice(0, n_train)
        val_slice = slice(n_train, n_train + n_val)
        test_slice = slice(n_train + n_val, n)

        parts = {
            "X_train": [list(row) for row in X[train_slice]],
            "y_train": [int(v) for v in y[train_slice]],
            "X_validation": [list(row) for row in X[val_slice]],
            "y_validation": [int(v) for v in y[val_slice]],
            "X_test": [list(row) for row in X[test_slice]],
            "y_test": [int(v) for v in y[test_slice]],
            "records_train": list(records[train_slice]),
            "records_validation": list(records[val_slice]),
            "records_test": list(records[test_slice]),
        }

        for name in ("y_train", "y_validation", "y_test"):
            labels = set(parts[name])
            if labels != {0, 1}:
                raise GradientBoostingModelError(
                    f"Temporal partition '{name}' does not contain both success and failure classes. "
                    "Collect more representative planning history before training."
                )
        return parts

    @staticmethod
    def _metrics(y_true: Sequence[int], probabilities: Sequence[float]) -> Dict[str, Any]:
        if not y_true or len(y_true) != len(probabilities):
            raise GradientBoostingModelError("Evaluation inputs are empty or mismatched.")
        predictions = [1 if p >= 0.5 else 0 for p in probabilities]
        tp = sum(1 for y, pred in zip(y_true, predictions) if y == 1 and pred == 1)
        tn = sum(1 for y, pred in zip(y_true, predictions) if y == 0 and pred == 0)
        fp = sum(1 for y, pred in zip(y_true, predictions) if y == 0 and pred == 1)
        fn = sum(1 for y, pred in zip(y_true, predictions) if y == 1 and pred == 0)
        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        balanced_accuracy = 0.5 * (recall + specificity)
        log_loss = GradientBoostedBinaryClassifier._binary_log_loss(y_true, probabilities)
        brier = sum((p - y) ** 2 for y, p in zip(y_true, probabilities)) / len(y_true)

        positives = [p for y, p in zip(y_true, probabilities) if y == 1]
        negatives = [p for y, p in zip(y_true, probabilities) if y == 0]
        comparisons = len(positives) * len(negatives)
        auc = 0.5
        if comparisons:
            wins = 0.0
            for pos in positives:
                for neg in negatives:
                    if pos > neg:
                        wins += 1.0
                    elif pos == neg:
                        wins += 0.5
            auc = wins / comparisons

        return {
            "samples": len(y_true),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "roc_auc": auc,
            "log_loss": log_loss,
            "brier_score": brier,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    def train_from_planning_db(
        self,
        *,
        db_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
        test_fraction: float = 0.20,
        minimum_total_samples: int = 30,
        min_delta: float = 1e-5,
    ) -> Dict[str, Any]:
        with self._lock:
            if output_path is not None:
                self.model_path = Path(output_path).expanduser().resolve()

            resolved_db = Path(db_path).expanduser().resolve() if db_path is not None else self.planning_db_path
            data = self.load_planning_db(resolved_db)
            X, y, records = self.build_training_dataset(data)
            split = self._temporal_split(
                X,
                y,
                records,
                validation_fraction=self.validation_fraction,
                test_fraction=test_fraction,
                minimum_total_samples=minimum_total_samples,
            )

            min_samples_leaf = max(2, min(10, self.min_samples_split // 3))
            model = GradientBoostedBinaryClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=min_samples_leaf,
                subsample=self.subsample,
                l2_leaf_regularization=float(self.gb_config.get("l2_leaf_regularization", 1.0)),
                min_split_gain=float(self.gb_config.get("min_split_gain", 1e-8)),
                max_bins=int(self.gb_config.get("max_bins", 32)),
                random_state=self.random_state,
            )
            model.fit(
                split["X_train"],
                split["y_train"],
                X_validation=split["X_validation"],
                y_validation=split["y_validation"],
                balanced_class_weight=self.class_weight.lower() == "balanced",
                early_stopping_rounds=self.early_stopping_rounds,
                min_delta=min_delta,
            )

            validation_prob = [model.predict_success_probability(row) for row in split["X_validation"]]
            test_prob = [model.predict_success_probability(row) for row in split["X_test"]]
            validation_metrics = self._metrics(split["y_validation"], validation_prob)
            test_metrics = self._metrics(split["y_test"], test_prob)

            self.model = model
            self.trained = True
            self.feature_importances_ = list(model.feature_importances_)

            training_metadata = {
                "database": str(resolved_db),
                "database_schema_version": data.get("schema_version"),
                "records_total": len(y),
                "split_sizes": {
                    "train": len(split["y_train"]),
                    "validation": len(split["y_validation"]),
                    "test": len(split["y_test"]),
                },
                "split_strategy": "chronological_holdout",
                "method_stat_feature_strategy": "causal_prior_only",
                "random_state": self.random_state,
                "early_stopping_rounds": self.early_stopping_rounds,
                "best_iteration": model.best_iteration_,
                "class_weight": self.class_weight,
            }
            evaluation = {
                "validation": validation_metrics,
                "test": test_metrics,
            }
            saved = self.save_model(
                source_sha256=_sha256_file(resolved_db),
                training_metadata=training_metadata,
                evaluation=evaluation,
            )

            logger.info(
                "Gradient Boosting training complete records=%d trees=%d test_auc=%.4f test_f1=%.4f model=%s",
                len(y),
                len(model.trees),
                test_metrics["roc_auc"],
                test_metrics["f1"],
                saved,
            )
            return {
                "model_path": str(saved),
                "training": training_metadata,
                "evaluation": evaluation,
                "feature_importance": self.report_feature_importance(),
            }

    def report_feature_importance(self) -> List[Tuple[str, float]]:
        values = self.feature_importances_
        if len(values) != len(self.feature_names):
            values = [0.0 for _ in self.feature_names]
        return sorted(zip(self.feature_names, values), key=lambda item: item[1], reverse=True)

    def update_model(self, task: Mapping[str, Any], world_state: Mapping[str, Any], outcome: str) -> None:
        """
        Persist a new execution result to the planning database.

        Retraining is deliberately not triggered for every single result.  Batch
        retraining should be invoked by the trainer/automation after sufficient
        new evidence accumulates; this avoids expensive and statistically noisy
        single-sample rebuilds during planning inference.
        """
        if not self.planning_db_path.is_file():
            raise GradientBoostingModelError(f"Planning database does not exist: {self.planning_db_path}")
        label = self._outcome_to_label(outcome)
        if label is None:
            raise GradientBoostingModelError(f"Unsupported planning outcome for training: {outcome!r}")

        with self._lock:
            with self.planning_db_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            tasks = data.setdefault("tasks", [])
            states = data.setdefault("world_states", [])
            if not isinstance(tasks, list) or not isinstance(states, list):
                raise GradientBoostingModelError("planning_db tasks/world_states must be lists.")
            new_task = dict(task)
            new_task["outcome"] = "success" if label == 1 else "failure"
            new_task.setdefault("timestamp", _utc_now_iso())
            tasks.append(new_task)
            states.append(dict(world_state))
            _atomic_write_json(self.planning_db_path, data)
            logger.info(
                "Appended planning training evidence task=%s method=%s outcome=%s",
                new_task.get("name"),
                new_task.get("selected_method"),
                new_task["outcome"],
            )


if __name__ == "__main__":
    printer.status("INIT", "Gradient Boosting Heuristic module loaded", "success")
    heuristic = GradientBoostingHeuristic(auto_load=True)
    printer.pretty(
        "Gradient Boosting Heuristic",
        {
            "trained": heuristic.trained,
            "model_path": str(heuristic.model_path),
            "features": heuristic.feature_names,
        },
        "Info",
    )
