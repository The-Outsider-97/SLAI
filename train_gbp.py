#!/usr/bin/env python3
"""SLAI Gradient-Boosted Planning Success Model trainer.

Run from the SLAI repository root with:

    py -m train_gbp

The default training source and output paths are resolved from
``src/agents/planning/configs/planning_config.yaml`` through SLAI's own
planning config loader.  The trainer uses SLAI's logger and the current
``BaseHeuristics`` feature semantics, while the boosting implementation itself
uses only the Python standard library.

Purpose
-------
Train a binary gradient-boosted decision-tree model estimating

    P(task-method execution succeeds | task, state, prior method history)

for SLAI's Planning Agent.

Dependency policy
-----------------
This module directly imports only Python standard-library modules.  SLAI
components are imported lazily at runtime.  SLAI's planning config loader
currently uses PyYAML internally; that is an existing SLAI dependency rather
than a dependency introduced by this trainer.  NumPy, scikit-learn and joblib
are not imported by this file.

Persistence compatibility
-------------------------
The primary model is written to the path currently expected by
``GradientBoostingHeuristic`` (normally
``src/agents/planning/models/gb_heuristic_model.pkl``) as a standard pickle
containing ``(model, scaler)``.  ``joblib.load`` can read ordinary pickle
streams, so the existing loader can deserialize this bundle as long as this
``train_gbp.py`` module remains importable from the repository root.

The persisted model exposes the runtime attributes/methods used by SLAI:
``predict_proba``, ``predict``, ``classes_`` and ``feature_importances_``.

Academic / validation note
--------------------------
Training features that depend on historical method success/failure are built
causally.  A record never sees its own outcome, nor any future outcome, when
``method_failure_rate`` is constructed.  The aggregate ``method_stats`` field
inside the planning database is therefore not used as a training label source.
This prevents a direct target-leakage path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


# When executed through ``python -m train_gbp`` the running module is named
# ``__main__``.  Register the canonical module alias before model classes are
# pickled so a later runtime can resolve them as ``train_gbp.<Class>``.
if __name__ == "__main__":
    sys.modules.setdefault("train_gbp", sys.modules[__name__])


MODEL_SCHEMA = "slai.planning.gbp.stdlib.v1"
TRAINER_VERSION = "1.0.0"
LOGGER_NAME = "SLAI GBP Trainer"
DEFAULT_TEST_FRACTION = 0.20
MIN_HOLDOUT_SAMPLES = 20
MIN_CLASS_COUNT_FOR_HOLDOUT = 3
EPS = 1e-12

SUCCESS_LABELS = {"success", "succeeded", "ok", "true", "1", "win", "completed"}
FAILURE_LABELS = {
    "failure",
    "failed",
    "error",
    "false",
    "0",
    "loss",
    "cancelled",
    "canceled",
    "timeout",
    "timed_out",
}


class GBPTrainingError(RuntimeError):
    """Raised for deterministic, user-actionable GBP training failures."""


@dataclass(frozen=True)
class TrainingRecord:
    """One causally-featurized planning execution."""

    order: int
    task_id: str
    task_name: str
    method_id: str
    timestamp: Optional[str]
    features: Tuple[float, ...]
    label: int


@dataclass(frozen=True)
class DatasetBundle:
    feature_names: Tuple[str, ...]
    records: Tuple[TrainingRecord, ...]
    source_path: str
    source_sha256: str
    skipped_records: int
    skipped_reasons: Mapping[str, int]

    @property
    def X(self) -> List[List[float]]:
        return [list(record.features) for record in self.records]

    @property
    def y(self) -> List[int]:
        return [int(record.label) for record in self.records]


@dataclass(frozen=True)
class DataSplit:
    mode: str
    train_indices: Tuple[int, ...]
    validation_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]
    note: str = ""


@dataclass
class TreeNode:
    """One node in a pure-Python Newton regression tree."""

    value: float
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    gain: float = 0.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    sample_count: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None

    def predict_row(self, row: Sequence[float]) -> float:
        node: TreeNode = self
        while not node.is_leaf:
            assert node.feature_index is not None
            assert node.threshold is not None
            if float(row[node.feature_index]) <= node.threshold:
                if node.left is None:
                    break
                node = node.left
            else:
                if node.right is None:
                    break
                node = node.right
        return float(node.value)

    def accumulate_feature_gains(self, totals: List[float]) -> None:
        if self.is_leaf:
            return
        assert self.feature_index is not None
        totals[self.feature_index] += max(0.0, float(self.gain))
        if self.left is not None:
            self.left.accumulate_feature_gains(totals)
        if self.right is not None:
            self.right.accumulate_feature_gains(totals)


class IdentityScaler:
    """scikit-learn-like identity transformer used for runtime compatibility.

    Gradient-boosted trees are invariant to monotonic feature scaling and do not
    require standardization.  SLAI's existing runtime calls ``scaler.transform``
    before ``predict_proba``; this adapter preserves that contract without
    introducing NumPy or scikit-learn.
    """

    def __init__(self) -> None:
        self.n_features_in_: Optional[int] = None

    def fit(self, X: Sequence[Sequence[float]], y: Any = None) -> "IdentityScaler":
        rows = _coerce_matrix(X)
        if not rows:
            raise GBPTrainingError("Cannot fit IdentityScaler on an empty matrix.")
        self.n_features_in_ = len(rows[0])
        return self

    def transform(self, X: Any) -> List[List[float]]:
        rows = _coerce_matrix(X)
        if self.n_features_in_ is not None:
            for row in rows:
                if len(row) != self.n_features_in_:
                    raise ValueError(
                        f"Feature dimension mismatch: expected {self.n_features_in_}, got {len(row)}."
                    )
        return rows

    def fit_transform(self, X: Any, y: Any = None) -> List[List[float]]:
        self.fit(X, y)
        return self.transform(X)


class NewtonRegressionTree:
    """Small binary tree fit to logistic gradients/hessians.

    The tree uses the second-order approximation to Bernoulli log-loss.  For a
    node containing samples ``I``, the optimal leaf value under L2 regularization
    is ``-G/(H + lambda)`` where ``G = sum(g_i)`` and ``H = sum(h_i)``.
    Candidate splits maximize the corresponding reduction in the quadratic
    objective.
    """

    def __init__(
        self,
        *,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        l2_regularization: float,
        min_gain: float,
        max_leaf_value: float,
    ) -> None:
        self.max_depth = max(1, int(max_depth))
        self.min_samples_split = max(2, int(min_samples_split))
        self.min_samples_leaf = max(1, int(min_samples_leaf))
        self.l2_regularization = max(0.0, float(l2_regularization))
        self.min_gain = max(0.0, float(min_gain))
        self.max_leaf_value = max(0.1, float(max_leaf_value))
        self.root: Optional[TreeNode] = None
        self.n_features_in_: Optional[int] = None

    def fit(
        self,
        X: Sequence[Sequence[float]],
        gradients: Sequence[float],
        hessians: Sequence[float],
        sample_indices: Optional[Sequence[int]] = None,
    ) -> "NewtonRegressionTree":
        rows = _coerce_matrix(X)
        if not rows:
            raise GBPTrainingError("Cannot fit a tree on an empty matrix.")
        if len(rows) != len(gradients) or len(rows) != len(hessians):
            raise GBPTrainingError("Tree X/gradient/hessian lengths do not match.")
        self.n_features_in_ = len(rows[0])
        indices = list(sample_indices) if sample_indices is not None else list(range(len(rows)))
        if not indices:
            raise GBPTrainingError("Tree sample subset is empty.")
        self.root = self._build(rows, gradients, hessians, indices, depth=0)
        return self

    def predict_row(self, row: Sequence[float]) -> float:
        if self.root is None:
            raise GBPTrainingError("Regression tree has not been fitted.")
        return self.root.predict_row(row)

    def predict(self, X: Any) -> List[float]:
        return [self.predict_row(row) for row in _coerce_matrix(X)]

    def feature_gains(self, n_features: int) -> List[float]:
        totals = [0.0] * n_features
        if self.root is not None:
            self.root.accumulate_feature_gains(totals)
        return totals

    def _build(
        self,
        X: List[List[float]],
        gradients: Sequence[float],
        hessians: Sequence[float],
        indices: List[int],
        *,
        depth: int,
    ) -> TreeNode:
        total_g = sum(float(gradients[i]) for i in indices)
        total_h = sum(max(float(hessians[i]), EPS) for i in indices)
        leaf_value = self._leaf_weight(total_g, total_h)
        node = TreeNode(value=leaf_value, sample_count=len(indices))

        if depth >= self.max_depth or len(indices) < self.min_samples_split:
            return node
        if len(indices) < 2 * self.min_samples_leaf:
            return node

        best = self._best_split(X, gradients, hessians, indices, total_g, total_h)
        if best is None:
            return node

        feature_index, threshold, gain, left_indices, right_indices = best
        if gain <= self.min_gain:
            return node

        node.feature_index = feature_index
        node.threshold = threshold
        node.gain = gain
        node.left = self._build(
            X, gradients, hessians, left_indices, depth=depth + 1
        )
        node.right = self._build(
            X, gradients, hessians, right_indices, depth=depth + 1
        )
        return node

    def _best_split(
        self,
        X: List[List[float]],
        gradients: Sequence[float],
        hessians: Sequence[float],
        indices: List[int],
        total_g: float,
        total_h: float,
    ) -> Optional[Tuple[int, float, float, List[int], List[int]]]:
        n_features = len(X[0])
        parent_score = self._node_score(total_g, total_h)
        best_gain = -math.inf
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None

        for feature_index in range(n_features):
            ordered = sorted(indices, key=lambda i: (X[i][feature_index], i))
            left_g = 0.0
            left_h = 0.0

            for pos in range(len(ordered) - 1):
                idx = ordered[pos]
                left_g += float(gradients[idx])
                left_h += max(float(hessians[idx]), EPS)

                left_count = pos + 1
                right_count = len(ordered) - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                current_value = float(X[idx][feature_index])
                next_value = float(X[ordered[pos + 1]][feature_index])
                if current_value == next_value:
                    continue

                right_g = total_g - left_g
                right_h = total_h - left_h
                gain = 0.5 * (
                    self._node_score(left_g, left_h)
                    + self._node_score(right_g, right_h)
                    - parent_score
                )

                if gain > best_gain + 1e-15:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = (current_value + next_value) / 2.0

        if best_feature is None or best_threshold is None or not math.isfinite(best_gain):
            return None

        left_indices = [i for i in indices if X[i][best_feature] <= best_threshold]
        right_indices = [i for i in indices if X[i][best_feature] > best_threshold]
        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            return None

        return best_feature, best_threshold, best_gain, left_indices, right_indices

    def _leaf_weight(self, gradient_sum: float, hessian_sum: float) -> float:
        denominator = hessian_sum + self.l2_regularization
        if denominator <= EPS:
            return 0.0
        value = -gradient_sum / denominator
        return max(-self.max_leaf_value, min(self.max_leaf_value, value))

    def _node_score(self, gradient_sum: float, hessian_sum: float) -> float:
        denominator = hessian_sum + self.l2_regularization
        if denominator <= EPS:
            return 0.0
        return (gradient_sum * gradient_sum) / denominator


class PurePythonGradientBoostingClassifier:
    """Binary gradient-boosted tree classifier with a sklearn-like inference API.

    This implementation minimizes Bernoulli log-loss using Newton updates:

      g_i = p_i - y_i
      h_i = p_i (1 - p_i)

    Each tree approximates the negative Newton step.  The model supports row
    subsampling, early stopping, feature-importance aggregation and deterministic
    fitting through an isolated ``random.Random`` instance.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 8,
        subsample: float = 0.8,
        min_samples_split: int = 15,
        min_samples_leaf: Optional[int] = None,
        random_state: int = 42,
        n_iter_no_change: Optional[int] = 20,
        l2_regularization: float = 1.0,
        min_gain: float = 1e-12,
        max_leaf_value: float = 5.0,
    ) -> None:
        if int(n_estimators) <= 0:
            raise ValueError("n_estimators must be > 0")
        if float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if not 0.0 < float(subsample) <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        if int(max_depth) <= 0:
            raise ValueError("max_depth must be > 0")
        if int(min_samples_split) < 2:
            raise ValueError("min_samples_split must be >= 2")

        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.subsample = float(subsample)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = (
            max(1, int(min_samples_leaf))
            if min_samples_leaf is not None
            else max(1, int(math.ceil(self.min_samples_split * 0.25)))
        )
        self.random_state = int(random_state)
        self.n_iter_no_change = (
            None if n_iter_no_change in (None, 0) else max(1, int(n_iter_no_change))
        )
        self.l2_regularization = max(0.0, float(l2_regularization))
        self.min_gain = max(0.0, float(min_gain))
        self.max_leaf_value = max(0.1, float(max_leaf_value))

        self.classes_ = [0, 1]
        self.trees_: List[NewtonRegressionTree] = []
        self.estimators_: List[List[NewtonRegressionTree]] = []
        self.feature_importances_: List[float] = []
        self.n_features_in_: Optional[int] = None
        self.init_score_: float = 0.0
        self.best_iteration_: int = 0
        self.n_estimators_: int = 0
        self.train_score_: List[float] = []
        self.validation_score_: List[float] = []
        self.training_history_: List[Dict[str, float]] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        X: Any,
        y: Sequence[int],
        *,
        X_val: Optional[Any] = None,
        y_val: Optional[Sequence[int]] = None,
        logger: Any = None,
    ) -> "PurePythonGradientBoostingClassifier":
        rows = _coerce_matrix(X)
        labels = [int(v) for v in y]
        self._validate_training_data(rows, labels)

        val_rows: Optional[List[List[float]]] = None
        val_labels: Optional[List[int]] = None
        if X_val is not None and y_val is not None:
            val_rows = _coerce_matrix(X_val)
            val_labels = [int(v) for v in y_val]
            if len(val_rows) != len(val_labels):
                raise GBPTrainingError("Validation X/y lengths do not match.")
            if val_rows and len(val_rows[0]) != len(rows[0]):
                raise GBPTrainingError("Validation feature count differs from training feature count.")
            if not val_rows:
                val_rows = None
                val_labels = None

        self.n_features_in_ = len(rows[0])
        self.trees_ = []
        self.estimators_ = []
        self.train_score_ = []
        self.validation_score_ = []
        self.training_history_ = []
        self.is_fitted_ = False

        positives = sum(labels)
        # Jeffreys-style 0.5 smoothing keeps finite log-odds for very small data.
        prior = (positives + 0.5) / (len(labels) + 1.0)
        prior = _clamp(prior, 1e-6, 1.0 - 1e-6)
        self.init_score_ = math.log(prior / (1.0 - prior))

        train_scores = [self.init_score_] * len(rows)
        val_scores = [self.init_score_] * len(val_rows or [])
        rng = random.Random(self.random_state)

        best_validation_loss = math.inf
        best_tree_count = 0
        rounds_without_improvement = 0

        for iteration in range(1, self.n_estimators + 1):
            probabilities = [_sigmoid(score) for score in train_scores]
            gradients = [p - y_i for p, y_i in zip(probabilities, labels)]
            hessians = [max(p * (1.0 - p), 1e-6) for p in probabilities]
            sample_indices = self._subsample_indices(len(rows), rng)

            tree = NewtonRegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
                min_gain=self.min_gain,
                max_leaf_value=self.max_leaf_value,
            ).fit(rows, gradients, hessians, sample_indices)

            self.trees_.append(tree)
            self.estimators_.append([tree])
            for idx, row in enumerate(rows):
                train_scores[idx] += self.learning_rate * tree.predict_row(row)

            train_loss = _log_loss_from_scores(labels, train_scores)
            self.train_score_.append(train_loss)
            record: Dict[str, float] = {
                "iteration": float(iteration),
                "train_log_loss": train_loss,
            }

            validation_loss: Optional[float] = None
            if val_rows is not None and val_labels is not None:
                for idx, row in enumerate(val_rows):
                    val_scores[idx] += self.learning_rate * tree.predict_row(row)
                validation_loss = _log_loss_from_scores(val_labels, val_scores)
                self.validation_score_.append(validation_loss)
                record["validation_log_loss"] = validation_loss

                if validation_loss < best_validation_loss - 1e-10:
                    best_validation_loss = validation_loss
                    best_tree_count = iteration
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

            self.training_history_.append(record)

            if logger is not None and (iteration == 1 or iteration % 10 == 0):
                if validation_loss is None:
                    logger.info(
                        "GBP iteration=%d/%d train_log_loss=%.6f",
                        iteration,
                        self.n_estimators,
                        train_loss,
                    )
                else:
                    logger.info(
                        "GBP iteration=%d/%d train_log_loss=%.6f validation_log_loss=%.6f",
                        iteration,
                        self.n_estimators,
                        train_loss,
                        validation_loss,
                    )

            if (
                val_rows is not None
                and self.n_iter_no_change is not None
                and rounds_without_improvement >= self.n_iter_no_change
            ):
                if logger is not None:
                    logger.info(
                        "GBP early stopping at iteration=%d best_iteration=%d",
                        iteration,
                        best_tree_count,
                    )
                break

        if val_rows is not None and best_tree_count > 0 and best_tree_count < len(self.trees_):
            self.trees_ = self.trees_[:best_tree_count]
            self.estimators_ = self.estimators_[:best_tree_count]
            self.train_score_ = self.train_score_[:best_tree_count]
            self.validation_score_ = self.validation_score_[:best_tree_count]
            self.training_history_ = self.training_history_[:best_tree_count]

        self.n_estimators_ = len(self.trees_)
        self.best_iteration_ = self.n_estimators_
        self.feature_importances_ = self._calculate_feature_importances()
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: Any) -> List[List[float]]:
        self._require_fitted()
        rows = _coerce_matrix(X)
        probabilities: List[List[float]] = []
        for row in rows:
            if self.n_features_in_ is not None and len(row) != self.n_features_in_:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.n_features_in_}, got {len(row)}."
                )
            score = self.init_score_
            for tree in self.trees_:
                score += self.learning_rate * tree.predict_row(row)
            p1 = _sigmoid(score)
            probabilities.append([1.0 - p1, p1])
        return probabilities

    def predict(self, X: Any) -> List[int]:
        return [1 if probs[1] >= 0.5 else 0 for probs in self.predict_proba(X)]

    def decision_function(self, X: Any) -> List[float]:
        self._require_fitted()
        rows = _coerce_matrix(X)
        scores: List[float] = []
        for row in rows:
            score = self.init_score_
            for tree in self.trees_:
                score += self.learning_rate * tree.predict_row(row)
            scores.append(score)
        return scores

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "n_iter_no_change": self.n_iter_no_change,
            "l2_regularization": self.l2_regularization,
            "min_gain": self.min_gain,
            "max_leaf_value": self.max_leaf_value,
        }

    def _validate_training_data(self, X: List[List[float]], y: List[int]) -> None:
        if not X:
            raise GBPTrainingError("Training matrix is empty.")
        if len(X) != len(y):
            raise GBPTrainingError("Training X/y lengths do not match.")
        width = len(X[0])
        if width == 0:
            raise GBPTrainingError("Training matrix has zero features.")
        if any(len(row) != width for row in X):
            raise GBPTrainingError("Training matrix is ragged.")
        unique = set(y)
        if not unique.issubset({0, 1}):
            raise GBPTrainingError(f"Binary labels must be 0/1, got {sorted(unique)}.")
        if len(unique) < 2:
            raise GBPTrainingError("Gradient-boosted success training requires both success and failure examples.")
        for row in X:
            for value in row:
                if not math.isfinite(float(value)):
                    raise GBPTrainingError("Training matrix contains a non-finite feature value.")

    def _subsample_indices(self, n_samples: int, rng: random.Random) -> List[int]:
        if self.subsample >= 1.0 or n_samples <= 1:
            return list(range(n_samples))
        sample_size = max(1, int(round(n_samples * self.subsample)))
        # A split is impossible with fewer than min_samples_split samples.  If the
        # dataset itself is large enough, keep at least that many in each boosting
        # stage; otherwise use every available sample rather than inventing data.
        if n_samples >= self.min_samples_split:
            sample_size = max(sample_size, self.min_samples_split)
        sample_size = min(sample_size, n_samples)
        return sorted(rng.sample(range(n_samples), sample_size))

    def _calculate_feature_importances(self) -> List[float]:
        if self.n_features_in_ is None:
            return []
        totals = [0.0] * self.n_features_in_
        for tree in self.trees_:
            gains = tree.feature_gains(self.n_features_in_)
            totals = [a + b for a, b in zip(totals, gains)]
        total_gain = sum(totals)
        if total_gain <= EPS:
            return [0.0] * self.n_features_in_
        return [value / total_gain for value in totals]

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise GBPTrainingError("Gradient-boosted classifier is not fitted.")


# Force canonical pickle module names whether imported normally or run as -m.
for _pickle_class in (
    TreeNode,
    IdentityScaler,
    NewtonRegressionTree,
    PurePythonGradientBoostingClassifier,
):
    _pickle_class.__module__ = "train_gbp"


@dataclass(frozen=True)
class SLAIRuntime:
    logger: Any
    printer: Any
    config: Mapping[str, Any]
    global_heuristic: Mapping[str, Any]
    gb_config: Mapping[str, Any]
    feature_extractor: Any


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _sigmoid(score: float) -> float:
    if score >= 0.0:
        z = math.exp(-min(score, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(score, -60.0))
    return z / (1.0 + z)


def _coerce_matrix(X: Any) -> List[List[float]]:
    """Convert common 1D/2D iterable inputs to a finite Python float matrix."""
    if X is None:
        return []

    # ndarray-like objects expose ``tolist``; invoking it does not import NumPy.
    if hasattr(X, "tolist"):
        X = X.tolist()

    if isinstance(X, (str, bytes)):
        raise TypeError("Feature matrix cannot be a string.")

    try:
        outer = list(X)
    except TypeError as exc:
        raise TypeError("Feature matrix must be iterable.") from exc

    if not outer:
        return []

    first = outer[0]
    if isinstance(first, (int, float, bool)):
        outer = [outer]

    matrix: List[List[float]] = []
    for raw_row in outer:
        if hasattr(raw_row, "tolist"):
            raw_row = raw_row.tolist()
        try:
            row = [float(value) for value in raw_row]
        except TypeError as exc:
            raise TypeError("Every feature row must be iterable.") from exc
        if not row:
            raise ValueError("Feature rows cannot be empty.")
        if any(not math.isfinite(value) for value in row):
            raise ValueError("Features must be finite numeric values.")
        matrix.append(row)
    return matrix


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_stable_json_bytes(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        commit = completed.stdout.strip()
        return commit or None
    except (OSError, subprocess.SubprocessError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _outcome_to_label(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in SUCCESS_LABELS:
        return 1
    if text in FAILURE_LABELS:
        return 0
    return None


def _class_counts(labels: Sequence[int]) -> Dict[str, int]:
    return {
        "failure_0": sum(1 for value in labels if int(value) == 0),
        "success_1": sum(1 for value in labels if int(value) == 1),
    }


def _select_rows(X: Sequence[Sequence[float]], indices: Sequence[int]) -> List[List[float]]:
    return [list(X[index]) for index in indices]


def _select_labels(y: Sequence[int], indices: Sequence[int]) -> List[int]:
    return [int(y[index]) for index in indices]


# ---------------------------------------------------------------------------
# SLAI integration and feature construction
# ---------------------------------------------------------------------------


def _load_slai_runtime() -> SLAIRuntime:
    """Load SLAI internals lazily so importing a persisted model stays lightweight."""
    try:
        from logs.logger import PrettyPrinter, configure_logging, get_logger
        from src.agents.planning.utils.base_heuristic import BaseHeuristics
        from src.agents.planning.utils.config_loader import get_config_section, load_global_config
    except ImportError as exc:
        raise GBPTrainingError(
            "SLAI planning modules could not be imported. Run this command from the SLAI repository root "
            "with the SLAI virtual environment active."
        ) from exc

    configure_logging()
    logger = get_logger(LOGGER_NAME)
    printer = PrettyPrinter()
    config = load_global_config()
    global_heuristic = get_config_section("global_heuristic", config=config, default={})
    gb_config = get_config_section("gradient_boosting_heuristic", config=config, default={})

    class _TrainingFeatureExtractor(BaseHeuristics):
        def __init__(self) -> None:
            # Pass a non-None sentinel to avoid constructing PlanningCalculations;
            # GBP training only needs BaseHeuristics' deterministic feature logic.
            super().__init__(calculations=False)

        def predict_success_prob(
            self,
            task: Any,
            world_state: Dict[str, Any],
            method_stats: Dict[Any, Dict[str, Any]],
            method_id: str,
        ) -> float:
            raise NotImplementedError("Training feature extractor does not perform inference.")

    feature_extractor = _TrainingFeatureExtractor()
    return SLAIRuntime(
        logger=logger,
        printer=printer,
        config=config,
        global_heuristic=global_heuristic,
        gb_config=gb_config,
        feature_extractor=feature_extractor,
    )


def _feature_names(gb_config: Mapping[str, Any]) -> Tuple[str, ...]:
    feature_config = gb_config.get("feature_config", {})
    if not isinstance(feature_config, Mapping):
        feature_config = {}
    names = [
        "task_depth",
        "goal_overlap",
        "method_failure_rate",
        "state_diversity",
    ]
    if bool(feature_config.get("use_priority")):
        names.append("task_priority")
    if bool(feature_config.get("use_resource_check")):
        names.extend(["cpu_available", "memory_available"])
    return tuple(names)


def _feature_vector(
    runtime: SLAIRuntime,
    task: Mapping[str, Any],
    world_state: Mapping[str, Any],
    prior_method_stats: MutableMapping[Any, Dict[str, Any]],
    method_id: str,
) -> Tuple[float, ...]:
    state = dict(world_state)
    base = runtime.feature_extractor.extract_base_features(
        dict(task), state, dict(prior_method_stats), str(method_id)
    )
    feature_config = runtime.gb_config.get("feature_config", {})
    if not isinstance(feature_config, Mapping):
        feature_config = {}

    vector = [
        float(base["task_depth"]),
        float(base["goal_overlap"]),
        float(base["method_failure_rate"]),
        float(base["state_diversity"]),
    ]
    if bool(feature_config.get("use_priority")):
        # Preserve the historical GBP feature contract: ``task_priority`` is the
        # explicit task value, not BaseHeuristics' max-priority normalization.
        vector.append(_safe_float(task.get("priority", 0.5), 0.5))
    if bool(feature_config.get("use_resource_check")):
        vector.append(_safe_float(state.get("cpu_available", 0.0), 0.0))
        vector.append(_safe_float(state.get("memory_available", 0.0), 0.0))

    if any(not math.isfinite(value) for value in vector):
        raise GBPTrainingError("Feature extraction produced a non-finite value.")
    return tuple(vector)


def _update_prior_method_stats(
    prior_method_stats: MutableMapping[Any, Dict[str, Any]],
    task: Mapping[str, Any],
    method_id: str,
    label: int,
) -> None:
    task_name = str(task.get("name") or task.get("id") or "unknown")
    key = (task_name, str(method_id))
    stats = prior_method_stats.setdefault(key, {"success": 0, "total": 0})
    stats["total"] = int(stats.get("total", 0)) + 1
    stats["success"] = int(stats.get("success", 0)) + (1 if label == 1 else 0)


def _record_timestamp(task: Mapping[str, Any], fallback_order: int) -> Tuple[int, float, int]:
    raw = task.get("timestamp", task.get("completed_at", task.get("creation_time")))
    parsed = _parse_timestamp(raw)
    if parsed is None:
        return (1, float(fallback_order), fallback_order)
    return (0, parsed.timestamp(), fallback_order)


def _load_dataset(runtime: SLAIRuntime, db_path: Path) -> DatasetBundle:
    if not db_path.is_file():
        raise GBPTrainingError(f"Planning database does not exist: {db_path}")
    try:
        with db_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise GBPTrainingError(f"Failed to read planning database {db_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise GBPTrainingError("Planning database root must be a JSON object.")

    tasks = payload.get("tasks")
    world_states = payload.get("world_states")
    if not isinstance(tasks, list) or not isinstance(world_states, list):
        raise GBPTrainingError("Planning database must contain list fields 'tasks' and 'world_states'.")
    if len(tasks) != len(world_states):
        raise GBPTrainingError(
            f"Planning database alignment error: tasks={len(tasks)} world_states={len(world_states)}. "
            "Refusing silent zip truncation."
        )

    paired = list(enumerate(zip(tasks, world_states)))
    paired.sort(
        key=lambda pair: _record_timestamp(
            pair[1][0] if isinstance(pair[1][0], Mapping) else {}, pair[0]
        )
    )

    prior_stats: Dict[Any, Dict[str, Any]] = {}
    feature_names = _feature_names(runtime.gb_config)
    records: List[TrainingRecord] = []
    skipped_reasons: Dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

    for chronological_order, (original_index, pair) in enumerate(paired):
        task_raw, state_raw = pair
        if not isinstance(task_raw, Mapping):
            skip("task_not_mapping")
            continue
        if not isinstance(state_raw, Mapping):
            skip("world_state_not_mapping")
            continue

        task = dict(task_raw)
        state = dict(state_raw)
        method_id = task.get("selected_method")
        if method_id in (None, ""):
            skip("missing_selected_method")
            continue

        label = _outcome_to_label(task.get("outcome"))
        if label is None:
            skip("unknown_or_missing_outcome")
            continue

        try:
            features = _feature_vector(runtime, task, state, prior_stats, str(method_id))
        except Exception as exc:
            runtime.logger.warning(
                "Skipping planning record index=%d due to feature extraction failure: %s",
                original_index,
                exc,
            )
            skip("feature_extraction_failure")
            continue

        if len(features) != len(feature_names):
            raise GBPTrainingError(
                f"Feature schema mismatch: names={len(feature_names)} vector={len(features)}."
            )

        records.append(
            TrainingRecord(
                order=chronological_order,
                task_id=str(task.get("id", "")),
                task_name=str(task.get("name", "")),
                method_id=str(method_id),
                timestamp=(
                    str(task.get("timestamp"))
                    if task.get("timestamp") is not None
                    else None
                ),
                features=features,
                label=label,
            )
        )

        # Crucially update history only after extracting this row's features.
        _update_prior_method_stats(prior_stats, task, str(method_id), label)
        runtime.feature_extractor.clear_feature_cache()

    if not records:
        raise GBPTrainingError("No valid labelled planning executions were available for training.")

    return DatasetBundle(
        feature_names=feature_names,
        records=tuple(records),
        source_path=str(db_path),
        source_sha256=_sha256_file(db_path),
        skipped_records=sum(skipped_reasons.values()),
        skipped_reasons=dict(sorted(skipped_reasons.items())),
    )


# ---------------------------------------------------------------------------
# Split / evaluation
# ---------------------------------------------------------------------------


def _make_split(
    labels: Sequence[int],
    *,
    validation_fraction: float,
    test_fraction: float,
    random_state: int,
) -> DataSplit:
    n = len(labels)
    counts = {0: labels.count(0), 1: labels.count(1)}

    if (
        n < MIN_HOLDOUT_SAMPLES
        or min(counts.values()) < MIN_CLASS_COUNT_FOR_HOLDOUT
    ):
        return DataSplit(
            mode="all_data_no_holdout",
            train_indices=tuple(range(n)),
            validation_indices=tuple(),
            test_indices=tuple(),
            note=(
                f"Holdout disabled: samples={n}, class_counts={counts}. "
                f"Need >= {MIN_HOLDOUT_SAMPLES} samples and >= "
                f"{MIN_CLASS_COUNT_FOR_HOLDOUT} examples per class."
            ),
        )

    validation_fraction = _clamp(validation_fraction, 0.05, 0.30)
    test_fraction = _clamp(test_fraction, 0.05, 0.30)
    if validation_fraction + test_fraction >= 0.50:
        validation_fraction = 0.15
        test_fraction = 0.20

    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * validation_fraction)))
    n_train = n - n_val - n_test

    # Prefer chronological holdout: earlier observations train the model; newer
    # observations estimate forward-looking generalization.
    if n_train >= 2:
        train = tuple(range(0, n_train))
        validation = tuple(range(n_train, n_train + n_val))
        test = tuple(range(n_train + n_val, n))
        if len({labels[i] for i in train}) == 2:
            return DataSplit(
                mode="chronological_holdout",
                train_indices=train,
                validation_indices=validation,
                test_indices=test,
                note="Chronological split preserves the temporal direction of planning executions.",
            )

    # If early history has only one class, use a deterministic stratified split
    # rather than producing an unfittable training partition.  The metadata makes
    # this fallback explicit because it is weaker for temporal generalization.
    rng = random.Random(random_state)
    class_indices: Dict[int, List[int]] = {0: [], 1: []}
    for idx, label in enumerate(labels):
        class_indices[int(label)].append(idx)

    train_list: List[int] = []
    val_list: List[int] = []
    test_list: List[int] = []
    for label in (0, 1):
        indices = list(class_indices[label])
        rng.shuffle(indices)
        class_n = len(indices)
        c_test = max(1, int(round(class_n * test_fraction)))
        c_val = max(1, int(round(class_n * validation_fraction)))
        while class_n - c_test - c_val < 1:
            if c_val > 1:
                c_val -= 1
            elif c_test > 1:
                c_test -= 1
            else:
                break
        test_list.extend(indices[:c_test])
        val_list.extend(indices[c_test : c_test + c_val])
        train_list.extend(indices[c_test + c_val :])

    return DataSplit(
        mode="deterministic_stratified_fallback",
        train_indices=tuple(sorted(train_list)),
        validation_indices=tuple(sorted(val_list)),
        test_indices=tuple(sorted(test_list)),
        note=(
            "Chronological training prefix contained only one class; deterministic stratification "
            "was used so the classifier could be fitted."
        ),
    )


def _log_loss(y_true: Sequence[int], probabilities: Sequence[float]) -> float:
    if len(y_true) != len(probabilities) or not y_true:
        return math.nan
    total = 0.0
    for label, probability in zip(y_true, probabilities):
        p = _clamp(probability, 1e-15, 1.0 - 1e-15)
        total += -(label * math.log(p) + (1 - label) * math.log(1.0 - p))
    return total / len(y_true)


def _log_loss_from_scores(y_true: Sequence[int], scores: Sequence[float]) -> float:
    return _log_loss(y_true, [_sigmoid(score) for score in scores])


def _roc_auc(y_true: Sequence[int], probabilities: Sequence[float]) -> Optional[float]:
    positives = sum(1 for y in y_true if y == 1)
    negatives = sum(1 for y in y_true if y == 0)
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(zip(probabilities, y_true), key=lambda item: item[0])
    rank = 1
    positive_rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        average_rank = (rank + (rank + (j - i) - 1)) / 2.0
        positive_rank_sum += average_rank * sum(1 for _, y in ordered[i:j] if y == 1)
        rank += j - i
        i = j

    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _metrics(y_true: Sequence[int], probabilities: Sequence[float]) -> Dict[str, Any]:
    if len(y_true) != len(probabilities) or not y_true:
        return {"samples": 0, "available": False}
    predictions = [1 if p >= 0.5 else 0 for p in probabilities]
    tp = sum(1 for y, pred in zip(y_true, predictions) if y == 1 and pred == 1)
    tn = sum(1 for y, pred in zip(y_true, predictions) if y == 0 and pred == 0)
    fp = sum(1 for y, pred in zip(y_true, predictions) if y == 0 and pred == 1)
    fn = sum(1 for y, pred in zip(y_true, predictions) if y == 1 and pred == 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true)
    balanced_accuracy = (recall + specificity) / 2.0
    brier = sum((p - y) ** 2 for y, p in zip(y_true, probabilities)) / len(y_true)
    base_rate = sum(y_true) / len(y_true)
    baseline_brier = sum((base_rate - y) ** 2 for y in y_true) / len(y_true)
    brier_skill = 1.0 - (brier / baseline_brier) if baseline_brier > EPS else None

    return {
        "available": True,
        "samples": len(y_true),
        "class_counts": _class_counts(y_true),
        "accuracy": accuracy,
        "precision_success": precision,
        "recall_success": recall,
        "specificity_failure": specificity,
        "f1_success": f1,
        "balanced_accuracy": balanced_accuracy,
        "log_loss": _log_loss(y_true, probabilities),
        "brier_score": brier,
        "brier_skill_score": brier_skill,
        "roc_auc": _roc_auc(y_true, probabilities),
        "observed_success_rate": base_rate,
        "mean_predicted_success": statistics.fmean(probabilities),
        "confusion_matrix": {
            "true_failure_pred_failure": tn,
            "true_failure_pred_success": fp,
            "true_success_pred_failure": fn,
            "true_success_pred_success": tp,
        },
    }


def _predict_positive(model: PurePythonGradientBoostingClassifier, X: Any) -> List[float]:
    return [float(probabilities[1]) for probabilities in model.predict_proba(X)]


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def _atomic_pickle_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _atomic_json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _load_pickle_bundle(path: Path) -> Tuple[Any, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, tuple) or len(value) != 2:
        raise GBPTrainingError("Persisted GBP model must contain the tuple (model, scaler).")
    return value[0], value[1]


def _backup_existing(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.stem}.{timestamp}.bak{path.suffix}")
    shutil.copy2(path, backup)
    return backup


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------


def _resolve_paths(
    repo_root: Path,
    runtime: SLAIRuntime,
    db_override: Optional[str],
    model_override: Optional[str],
) -> Tuple[Path, Path, Path]:
    planning_root = repo_root / "src" / "agents" / "planning"

    if db_override:
        db_path = Path(db_override)
        if not db_path.is_absolute():
            db_path = repo_root / db_path
    else:
        configured_db = str(runtime.global_heuristic.get("planning_db_path") or "templates/planning_db.json")
        raw = Path(configured_db)
        if raw.is_absolute():
            db_path = raw
        else:
            root_candidate = repo_root / raw
            planning_candidate = planning_root / raw
            db_path = root_candidate if root_candidate.exists() else planning_candidate

    if model_override:
        model_path = Path(model_override)
        if not model_path.is_absolute():
            model_path = repo_root / model_path
    else:
        configured_model_dir = str(
            runtime.global_heuristic.get("heuristic_model_path")
            or "src/agents/planning/models/"
        )
        model_dir = Path(configured_model_dir)
        if not model_dir.is_absolute():
            model_dir = repo_root / model_dir
        model_path = model_dir / "gb_heuristic_model.pkl"

    metadata_path = model_path.with_name(model_path.stem + ".metadata.json")
    return db_path.resolve(), model_path.resolve(), metadata_path.resolve()


def _model_hyperparameters(runtime: SLAIRuntime, n_estimators_override: Optional[int]) -> Dict[str, Any]:
    g = runtime.global_heuristic
    c = runtime.gb_config
    n_estimators = int(n_estimators_override or c.get("n_estimators", 200))
    min_samples_split = int(g.get("min_samples_split", 15))
    return {
        "n_estimators": max(1, n_estimators),
        "learning_rate": float(c.get("learning_rate", 0.05)),
        "max_depth": max(1, int(g.get("max_depth", 8))),
        "subsample": float(c.get("subsample", 0.8)),
        "min_samples_split": max(2, min_samples_split),
        "min_samples_leaf": max(1, int(math.ceil(max(2, min_samples_split) * 0.25))),
        "random_state": int(g.get("random_state", 42)),
        "n_iter_no_change": (
            None
            if not c.get("early_stopping_rounds")
            else max(1, int(c.get("early_stopping_rounds")))
        ),
        "l2_regularization": 1.0,
        "min_gain": 1e-12,
        "max_leaf_value": 5.0,
    }


def _fit_evaluation_model(
    dataset: DatasetBundle,
    split: DataSplit,
    hyperparameters: Mapping[str, Any],
    runtime: SLAIRuntime,
) -> Tuple[PurePythonGradientBoostingClassifier, Dict[str, Any]]:
    X = dataset.X
    y = dataset.y
    train_X = _select_rows(X, split.train_indices)
    train_y = _select_labels(y, split.train_indices)
    val_X = _select_rows(X, split.validation_indices)
    val_y = _select_labels(y, split.validation_indices)
    test_X = _select_rows(X, split.test_indices)
    test_y = _select_labels(y, split.test_indices)

    model = PurePythonGradientBoostingClassifier(**dict(hyperparameters))
    model.fit(
        train_X,
        train_y,
        X_val=val_X if val_X else None,
        y_val=val_y if val_y else None,
        logger=runtime.logger,
    )

    evaluation: Dict[str, Any] = {
        "scope": (
            "held_out_generalization"
            if test_X
            else "training_resubstitution_not_generalization"
        ),
        "split_mode": split.mode,
        "split_note": split.note,
        "train": _metrics(train_y, _predict_positive(model, train_X)),
        "validation": (
            _metrics(val_y, _predict_positive(model, val_X))
            if val_X
            else {"samples": 0, "available": False}
        ),
        "test": (
            _metrics(test_y, _predict_positive(model, test_X))
            if test_X
            else {"samples": 0, "available": False}
        ),
        "best_iteration": model.best_iteration_,
    }
    return model, evaluation


def _fit_final_model(
    dataset: DatasetBundle,
    evaluation_model: PurePythonGradientBoostingClassifier,
    split: DataSplit,
    hyperparameters: Mapping[str, Any],
    runtime: SLAIRuntime,
) -> PurePythonGradientBoostingClassifier:
    final_parameters = dict(hyperparameters)
    if split.test_indices and evaluation_model.best_iteration_ > 0:
        # Refit on all valid evidence using the number of estimators selected by
        # the validation history.  Final fitting does not reserve another hidden
        # validation subset; the independent test result remains the reported
        # generalization estimate.
        final_parameters["n_estimators"] = evaluation_model.best_iteration_
        final_parameters["n_iter_no_change"] = None

    final_model = PurePythonGradientBoostingClassifier(**final_parameters)
    final_model.fit(dataset.X, dataset.y, logger=runtime.logger)
    return final_model


def _quality_assessment(dataset: DatasetBundle, split: DataSplit, evaluation: Mapping[str, Any]) -> Dict[str, Any]:
    labels = dataset.y
    n = len(labels)
    min_class = min(labels.count(0), labels.count(1))
    test = evaluation.get("test", {}) if isinstance(evaluation, Mapping) else {}

    if not split.test_indices:
        return {
            "deployment_recommended": False,
            "level": "insufficient_external_validation",
            "reason": (
                "A model was trained, but no independent test partition was statistically supportable. "
                "Do not treat resubstitution metrics as generalization evidence."
            ),
            "minimum_recommended_history": max(MIN_HOLDOUT_SAMPLES, 30),
            "observed_samples": n,
            "minority_class_samples": min_class,
        }

    balanced = test.get("balanced_accuracy") if isinstance(test, Mapping) else None
    brier_skill = test.get("brier_skill_score") if isinstance(test, Mapping) else None
    recommended = bool(
        isinstance(balanced, (int, float))
        and balanced >= 0.55
        and (brier_skill is None or brier_skill >= 0.0)
    )
    return {
        "deployment_recommended": recommended,
        "level": "validated" if recommended else "holdout_below_quality_gate",
        "reason": (
            "Independent test metrics meet the conservative default gate."
            if recommended
            else "Independent test metrics do not yet meet the conservative default gate."
        ),
        "quality_gate": {
            "balanced_accuracy_min": 0.55,
            "brier_skill_score_min": 0.0,
        },
        "observed_samples": n,
        "minority_class_samples": min_class,
    }


def _build_metadata(
    *,
    repo_root: Path,
    runtime: SLAIRuntime,
    dataset: DatasetBundle,
    model: PurePythonGradientBoostingClassifier,
    model_path: Path,
    split: DataSplit,
    evaluation: Mapping[str, Any],
    hyperparameters: Mapping[str, Any],
    quality: Mapping[str, Any],
    elapsed_seconds: float,
    backup_path: Optional[Path],
) -> Dict[str, Any]:
    config_path_raw = runtime.config.get("__config_path__")
    config_path = Path(str(config_path_raw)) if config_path_raw else None
    feature_importance = {
        name: model.feature_importances_[index]
        for index, name in enumerate(dataset.feature_names)
    }

    return {
        "schema": MODEL_SCHEMA,
        "trainer_version": TRAINER_VERSION,
        "created_at": _utc_now(),
        "objective": "Predict probability that a SLAI planning task/method execution succeeds.",
        "model": {
            "class": "train_gbp.PurePythonGradientBoostingClassifier",
            "path": str(model_path),
            "sha256": _sha256_file(model_path),
            "classes": list(model.classes_),
            "n_estimators_fitted": model.n_estimators_,
            "best_iteration": model.best_iteration_,
            "feature_names": list(dataset.feature_names),
            "feature_schema_sha256": _sha256_value(list(dataset.feature_names)),
            "feature_importances": feature_importance,
            "persistence": "stdlib_pickle_tuple_model_scaler",
            "runtime_api": ["predict_proba", "predict", "classes_", "feature_importances_"],
        },
        "data": {
            "planning_db_path": dataset.source_path,
            "planning_db_sha256": dataset.source_sha256,
            "valid_records": len(dataset.records),
            "skipped_records": dataset.skipped_records,
            "skipped_reasons": dict(dataset.skipped_reasons),
            "class_counts": _class_counts(dataset.y),
            "causal_method_history": True,
            "aggregate_method_stats_used_as_training_history": False,
        },
        "split": {
            "mode": split.mode,
            "note": split.note,
            "train_samples": len(split.train_indices),
            "validation_samples": len(split.validation_indices),
            "test_samples": len(split.test_indices),
        },
        "training": {
            "hyperparameters": dict(hyperparameters),
            "final_hyperparameters": model.get_params(),
            "elapsed_seconds": elapsed_seconds,
            "stdlib_boosting_implementation": True,
            "direct_third_party_imports": [],
        },
        "evaluation": dict(evaluation),
        "quality": dict(quality),
        "provenance": {
            "git_commit": _git_commit(repo_root),
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "planning_config_path": str(config_path) if config_path else None,
            "planning_config_sha256": (
                _sha256_file(config_path)
                if config_path is not None and config_path.is_file()
                else None
            ),
            "backup_of_previous_model": str(backup_path) if backup_path else None,
        },
    }


def train(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parent
    runtime = _load_slai_runtime()
    logger = runtime.logger
    printer = runtime.printer

    printer.section_header("SLAI Gradient-Boosted Planning Success Training")
    printer.status("INIT", "SLAI planning runtime and logger initialized", "success")

    db_path, model_path, metadata_path = _resolve_paths(
        repo_root, runtime, args.db, args.model
    )
    logger.info("Planning DB: %s", db_path)
    logger.info("GBP model output: %s", model_path)
    logger.info("GBP metadata output: %s", metadata_path)

    dataset = _load_dataset(runtime, db_path)
    labels = dataset.y
    counts = _class_counts(labels)
    printer.status(
        "DATA",
        f"Loaded {len(dataset.records)} valid executions; class distribution={counts}; "
        f"skipped={dataset.skipped_records}",
        "info",
    )
    logger.info("Feature schema: %s", list(dataset.feature_names))

    if len(set(labels)) < 2:
        raise GBPTrainingError(
            "Training data contains only one outcome class. At least one success and one failure are required."
        )

    hyperparameters = _model_hyperparameters(runtime, args.n_estimators)
    validation_fraction = float(runtime.gb_config.get("validation_fraction", 0.20))
    split = _make_split(
        labels,
        validation_fraction=validation_fraction,
        test_fraction=float(args.test_fraction),
        random_state=int(hyperparameters["random_state"]),
    )
    logger.info(
        "Split mode=%s train=%d validation=%d test=%d note=%s",
        split.mode,
        len(split.train_indices),
        len(split.validation_indices),
        len(split.test_indices),
        split.note,
    )

    if len(dataset.records) < int(hyperparameters["min_samples_split"]):
        logger.warning(
            "Valid records (%d) are fewer than configured min_samples_split (%d). "
            "Trees may remain unsplit and the model may behave like a calibrated prior until more history exists.",
            len(dataset.records),
            hyperparameters["min_samples_split"],
        )

    evaluation_model, evaluation = _fit_evaluation_model(
        dataset, split, hyperparameters, runtime
    )
    final_model = _fit_final_model(
        dataset, evaluation_model, split, hyperparameters, runtime
    )
    scaler = IdentityScaler().fit(dataset.X)

    quality = _quality_assessment(dataset, split, evaluation)
    if not quality["deployment_recommended"]:
        printer.status("QUALITY", str(quality["reason"]), "warning")

    backup_path: Optional[Path] = None
    if model_path.exists() and not args.no_backup:
        backup_path = _backup_existing(model_path)
        logger.info("Backed up previous GBP model to %s", backup_path)

    _atomic_pickle_dump(model_path, (final_model, scaler))

    # Immediate round-trip verification catches persistence/module-name errors
    # before metadata announces a model as successfully written.
    loaded_model, loaded_scaler = _load_pickle_bundle(model_path)
    smoke_features = [list(dataset.records[-1].features)]
    smoke_scaled = loaded_scaler.transform(smoke_features)
    smoke_probability = float(loaded_model.predict_proba(smoke_scaled)[0][1])
    if not 0.0 <= smoke_probability <= 1.0 or not math.isfinite(smoke_probability):
        raise GBPTrainingError("Persisted model failed probability smoke verification.")

    elapsed = time.perf_counter() - started
    metadata = _build_metadata(
        repo_root=repo_root,
        runtime=runtime,
        dataset=dataset,
        model=final_model,
        model_path=model_path,
        split=split,
        evaluation=evaluation,
        hyperparameters=hyperparameters,
        quality=quality,
        elapsed_seconds=elapsed,
        backup_path=backup_path,
    )
    _atomic_json_dump(metadata_path, metadata)

    test_metrics = evaluation.get("test", {})
    if isinstance(test_metrics, Mapping) and test_metrics.get("available"):
        printer.status(
            "EVAL",
            "Held-out test: "
            f"balanced_accuracy={test_metrics.get('balanced_accuracy', 0.0):.4f}, "
            f"F1={test_metrics.get('f1_success', 0.0):.4f}, "
            f"log_loss={test_metrics.get('log_loss', 0.0):.6f}, "
            f"ROC_AUC={test_metrics.get('roc_auc')}",
            "success" if quality["deployment_recommended"] else "warning",
        )
    else:
        train_metrics = evaluation.get("train", {})
        printer.status(
            "EVAL",
            "No independent holdout available; training-only diagnostics: "
            f"balanced_accuracy={train_metrics.get('balanced_accuracy', 0.0):.4f}, "
            f"log_loss={train_metrics.get('log_loss', 0.0):.6f}",
            "warning",
        )

    importance_rows = [
        (name, f"{importance:.6f}")
        for name, importance in sorted(
            zip(dataset.feature_names, final_model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    if importance_rows:
        printer.table(["Feature", "Importance"], importance_rows, title="GBP Feature Importance")

    printer.status("SAVE", f"Model saved: {model_path}", "success")
    printer.status("SAVE", f"Metadata saved: {metadata_path}", "success")
    printer.status(
        "DONE",
        f"Training completed in {elapsed:.3f}s; smoke P(success)={smoke_probability:.6f}",
        "success",
    )
    logger.info(
        "GBP training complete model=%s metadata=%s deployment_recommended=%s",
        model_path,
        metadata_path,
        quality["deployment_recommended"],
    )
    return 0


# ---------------------------------------------------------------------------
# Self-test independent of SLAI imports
# ---------------------------------------------------------------------------


def self_test() -> int:
    rng = random.Random(7)
    X: List[List[float]] = []
    y: List[int] = []
    for _ in range(500):
        depth = rng.random()
        overlap = rng.random()
        failure = rng.random()
        diversity = rng.random()
        priority = rng.random()
        cpu = rng.uniform(10.0, 100.0)
        memory = rng.uniform(256.0, 8192.0)
        latent = (
            2.2 * overlap
            - 2.0 * failure
            - 0.7 * depth
            + 0.8 * priority
            + 0.004 * cpu
            + 0.00005 * memory
            - 0.25 * diversity
            + rng.gauss(0.0, 0.35)
            - 0.5
        )
        probability = _sigmoid(latent)
        label = 1 if rng.random() < probability else 0
        X.append([depth, overlap, failure, diversity, priority, cpu, memory])
        y.append(label)

    indices = list(range(len(X)))
    rng.shuffle(indices)
    train_indices = indices[:350]
    val_indices = indices[350:425]
    test_indices = indices[425:]

    model = PurePythonGradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=3,
        subsample=0.85,
        min_samples_split=12,
        random_state=11,
        n_iter_no_change=12,
    )
    model.fit(
        _select_rows(X, train_indices),
        _select_labels(y, train_indices),
        X_val=_select_rows(X, val_indices),
        y_val=_select_labels(y, val_indices),
    )
    test_y = _select_labels(y, test_indices)
    metrics = _metrics(
        test_y,
        _predict_positive(model, _select_rows(X, test_indices)),
    )
    if not metrics.get("available"):
        raise GBPTrainingError("Self-test metrics unavailable.")
    if float(metrics["balanced_accuracy"]) < 0.58:
        raise GBPTrainingError(
            f"Self-test balanced accuracy unexpectedly low: {metrics['balanced_accuracy']:.4f}"
        )

    with tempfile.TemporaryDirectory(prefix="slai_gbp_selftest_") as temp_dir:
        path = Path(temp_dir) / "gb.pkl"
        scaler = IdentityScaler().fit(X)
        _atomic_pickle_dump(path, (model, scaler))
        restored_model, restored_scaler = _load_pickle_bundle(path)
        sample = [X[0]]
        before = model.predict_proba(sample)[0][1]
        after = restored_model.predict_proba(restored_scaler.transform(sample))[0][1]
        if abs(before - after) > 1e-12:
            raise GBPTrainingError("Self-test persistence round trip changed prediction.")

    print(
        json.dumps(
            {
                "status": "ok",
                "schema": MODEL_SCHEMA,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "roc_auc": metrics["roc_auc"],
                "log_loss": metrics["log_loss"],
                "estimators": model.n_estimators_,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m train_gbp",
        description=(
            "Train SLAI's Gradient-Boosted Planning Success Model from the planning execution database."
        ),
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Optional planning_db.json override. Default: SLAI planning_config.yaml.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model output override. Default: SLAI heuristic_model_path/gb_heuristic_model.pkl.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help=f"Independent test fraction when enough evidence exists (default {DEFAULT_TEST_FRACTION:.2f}).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Optional override for configured gradient_boosting_heuristic.n_estimators.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not backup an existing model before atomic replacement.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the dependency-free synthetic implementation/persistence self-test instead of SLAI training.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            return self_test()
        if not 0.0 < float(args.test_fraction) < 0.5:
            raise GBPTrainingError("--test-fraction must be between 0 and 0.5.")
        if args.n_estimators is not None and args.n_estimators <= 0:
            raise GBPTrainingError("--n-estimators must be > 0.")
        return train(args)
    except GBPTrainingError as exc:
        # If SLAI logger is available use it; otherwise keep failure reporting
        # dependency-free and suitable for bootstrap/import errors.
        try:
            from logs.logger import configure_logging, get_logger

            configure_logging()
            get_logger(LOGGER_NAME).error("%s", exc)
        except Exception:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Training interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        try:
            from logs.logger import configure_logging, get_logger

            configure_logging()
            get_logger(LOGGER_NAME).exception("Unexpected GBP trainer failure: %s", exc)
        except Exception:
            print(f"Unexpected GBP trainer failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
