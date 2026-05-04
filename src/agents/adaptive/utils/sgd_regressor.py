from __future__ import annotations

import pickle
import numpy as np

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .config_loader import load_global_config, get_config_section
from .adaptive_errors import *
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("SGD Regressor")
printer = PrettyPrinter


class SGDRegressor:
    """
    Online linear regression with SGD and adaptive learning rates.

    Production-oriented features
    ----------------------------
    - Incremental `partial_fit` updates for online learning.
    - Full `fit` workflow built on top of the online optimizer.
    - Config-driven learning-rate schedules with safe validation.
    - L2 regularization, optional intercept fitting, and gradient clipping.
    - Deterministic RNG support for reproducibility.
    - Stable checkpoint save/load helpers.
    - Structured diagnostics and error handling integrated with `adaptive_errors`.

    Public compatibility preserved
    -----------------------------
    This implementation intentionally preserves the core interface already used by
    the adaptive stack:
        - `partial_fit(X, y, sample_weight=None)`
        - `predict(X)`
        - `score(X, y)`
        - `get_feature_importance()`
        - `reset()`
        - `coef_`, `intercept_`, `n_samples_seen`, `t_`, `loss_history`

    That compatibility matters because `MultiModalMemory` uses this regressor for
    online causal/parameter impact analysis and expects the model to expose SGD-style
    coefficients after incremental updates.
    """

    SUPPORTED_LEARNING_RATES = {"constant", "invscaling", "adaptive"}

    def __init__(self) -> None:
        self.config = load_global_config()
        self.sgd_config = get_config_section("sgd_regressor")
        self._load_config()
        self._initialize_runtime_state()
        logger.info("SGD Regressor successfully initialized")

    def _load_config(self) -> None:
        """Load and validate configuration from adaptive_config.yaml."""
        try:
            self.eta0 = float(self.sgd_config.get("eta0", 0.01))
            self.learning_rate = str(self.sgd_config.get("learning_rate", "constant")).lower()
            self.alpha = float(self.sgd_config.get("alpha", 0.0001))
            self.power_t = float(self.sgd_config.get("power_t", 0.25))
            self.max_iter = int(self.sgd_config.get("max_iter", 100))
            self.tol = float(self.sgd_config.get("tol", 0.0001))
            self.fit_intercept = bool(self.sgd_config.get("fit_intercept", True))
            self.shuffle = bool(self.sgd_config.get("shuffle", True))
            self.random_state = self.sgd_config.get("random_state")
            self.min_learning_rate = float(self.sgd_config.get("min_learning_rate", 1e-6))
            self.adaptive_decay = float(self.sgd_config.get("adaptive_decay", 0.5))
            self.adaptive_patience = int(self.sgd_config.get("adaptive_patience", 5))
            self.adaptive_tolerance = float(self.sgd_config.get("adaptive_tolerance", 1e-4))
            self.lr_decay_frequency = int(self.sgd_config.get("lr_decay_frequency", 100))
            self.lr_decay_factor = float(self.sgd_config.get("lr_decay_factor", 0.9))
            self.clip_gradient = self.sgd_config.get("clip_gradient", None)
            self.track_history = bool(self.sgd_config.get("track_history", True))
            self.max_history = int(self.sgd_config.get("max_history", 1000))
            self.weight_init_scale = float(self.sgd_config.get("weight_init_scale", 0.01))
            self.convergence_check = str(self.sgd_config.get("convergence_check", "parameter_change")).lower()
            self.epsilon = float(self.sgd_config.get("epsilon", 1e-12))
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationValueError(
                "Failed to parse SGD regressor configuration values.",
                component="sgd_regressor",
                details={"section": "sgd_regressor"},
                remediation="Ensure all SGD regressor configuration values are valid scalars.",
                cause=exc,
            ) from exc

        ensure_positive(self.eta0, "eta0", component="sgd_regressor")
        ensure_positive(self.max_iter, "max_iter", component="sgd_regressor")
        ensure_positive(self.power_t, "power_t", allow_zero=True, component="sgd_regressor")
        ensure_in_range(self.alpha, "alpha", minimum=0.0, component="sgd_regressor")
        ensure_in_range(self.tol, "tol", minimum=0.0, component="sgd_regressor")
        ensure_positive(self.min_learning_rate, "min_learning_rate", component="sgd_regressor")
        ensure_in_range(self.adaptive_decay, "adaptive_decay", minimum=0.0, maximum=1.0, component="sgd_regressor")
        ensure_positive(self.adaptive_patience, "adaptive_patience", component="sgd_regressor")
        ensure_in_range(self.adaptive_tolerance, "adaptive_tolerance", minimum=0.0, component="sgd_regressor")
        ensure_positive(self.lr_decay_frequency, "lr_decay_frequency", component="sgd_regressor")
        ensure_in_range(self.lr_decay_factor, "lr_decay_factor", minimum=0.0, maximum=1.0, component="sgd_regressor")
        ensure_positive(self.weight_init_scale, "weight_init_scale", component="sgd_regressor")
        ensure_in_range(self.epsilon, "epsilon", minimum=0.0, component="sgd_regressor")

        if self.learning_rate not in self.SUPPORTED_LEARNING_RATES:
            raise UnsupportedLearningRateScheduleError(
                f"Unsupported learning rate schedule: {self.learning_rate}",
                component="sgd_regressor",
                details={
                    "supported": sorted(self.SUPPORTED_LEARNING_RATES),
                    "received": self.learning_rate,
                },
                remediation="Use one of: constant, invscaling, adaptive.",
            )

        if self.clip_gradient is not None:
            self.clip_gradient = float(self.clip_gradient)
            ensure_positive(self.clip_gradient, "clip_gradient", component="sgd_regressor")

        if self.convergence_check not in {"parameter_change", "loss", "either"}:
            raise InvalidConfigurationValueError(
                f"Unsupported convergence_check mode: {self.convergence_check}",
                component="sgd_regressor",
                details={"convergence_check": self.convergence_check},
                remediation="Use parameter_change, loss, or either.",
            )

    def _initialize_runtime_state(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        self.initial_eta0 = self.eta0
        self._current_eta = self.eta0
        self._best_loss = np.inf
        self._epochs_without_improvement = 0
        self._is_fitted = False
        self._last_epoch_loss: Optional[float] = None
        self._last_gradient_norm: Optional[float] = None
        self._last_update_norm: Optional[float] = None
        self._converged = False

        self.coef_: Optional[np.ndarray] = None
        self.intercept_ = 0.0
        self.n_features_in_: Optional[int] = None
        self.n_samples_seen = 0
        self.t_ = 0
        self.loss_history: List[float] = []
        self.learning_rate_history: List[float] = []
        self.gradient_norm_history: List[float] = []
        self.update_norm_history: List[float] = []

    def _initialize_parameters(self, n_features: int) -> None:
        ensure_positive(n_features, "n_features", component="sgd_regressor")
        self.n_features_in_ = int(n_features)
        self.coef_ = self.rng.normal(loc=0.0, scale=self.weight_init_scale, size=self.n_features_in_)
        self.intercept_ = 0.0
        logger.debug("SGD Regressor parameters initialized")

    @property
    def eta(self) -> float:
        """Current learning rate after schedule application."""
        if self.learning_rate == "constant":
            return max(self.min_learning_rate, self.eta0)
        if self.learning_rate == "invscaling":
            return max(self.min_learning_rate, self.eta0 / np.power(self.t_ + 1, self.power_t))
        if self.learning_rate == "adaptive":
            return max(self.min_learning_rate, self._current_eta)
        raise UnsupportedLearningRateScheduleError(
            f"Unsupported learning rate schedule: {self.learning_rate}",
            component="sgd_regressor",
            details={"received": self.learning_rate},
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted and self.coef_ is not None and self.n_features_in_ is not None

    def _trim_history(self) -> None:
        if not self.track_history:
            self.loss_history.clear()
            self.learning_rate_history.clear()
            self.gradient_norm_history.clear()
            self.update_norm_history.clear()
            return

        for history in (
            self.loss_history,
            self.learning_rate_history,
            self.gradient_norm_history,
            self.update_norm_history,
        ):
            if len(history) > self.max_history:
                del history[: len(history) - self.max_history]

    def _validate_and_prepare_X_y(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]],
        y: Union[np.ndarray, Sequence[float], float, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            X_arr = np.asarray(X, dtype=np.float64)
            y_arr = np.asarray(y, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidTypeError(
                "X and y must be numeric and convertible to numpy arrays.",
                component="sgd_regressor",
                details={"X_type": type(X).__name__, "y_type": type(y).__name__},
                remediation="Provide numeric array-like inputs for X and y.",
                cause=exc,
            ) from exc

        if X_arr.ndim == 0:
            raise InvalidValueError(
                "X must be one-dimensional or two-dimensional.",
                component="sgd_regressor",
                details={"X_shape": X_arr.shape},
            )
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        elif X_arr.ndim != 2:
            raise InvalidValueError(
                "X must be one-dimensional or two-dimensional.",
                component="sgd_regressor",
                details={"X_shape": X_arr.shape},
            )

        if y_arr.ndim == 0:
            y_arr = y_arr.reshape(1)
        elif y_arr.ndim > 1:
            y_arr = y_arr.reshape(-1)

        if X_arr.shape[0] != y_arr.shape[0]:
            raise InvalidValueError(
                "X and y must contain the same number of samples.",
                component="sgd_regressor",
                details={"X_samples": X_arr.shape[0], "y_samples": y_arr.shape[0]},
                remediation="Ensure each input row has a corresponding target value.",
            )

        if X_arr.shape[0] == 0:
            raise EmptyCollectionError(
                "Training data must not be empty.",
                component="sgd_regressor",
                details={"X_shape": X_arr.shape},
            )

        if not np.all(np.isfinite(X_arr)):
            raise InvalidValueError(
                "X contains NaN or infinite values.",
                component="sgd_regressor",
            )
        if not np.all(np.isfinite(y_arr)):
            raise InvalidValueError(
                "y contains NaN or infinite values.",
                component="sgd_regressor",
            )

        return X_arr, y_arr

    def _prepare_sample_weight(self, sample_weight: Optional[Union[np.ndarray, Sequence[float]]], n_samples: int) -> np.ndarray:
        if sample_weight is None:
            return np.ones(n_samples, dtype=np.float64)

        try:
            weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise InvalidTypeError(
                "sample_weight must be numeric and array-like.",
                component="sgd_regressor",
                cause=exc,
            ) from exc

        if weights.shape[0] != n_samples:
            raise InvalidValueError(
                "sample_weight must match the number of samples.",
                component="sgd_regressor",
                details={"expected": n_samples, "received": weights.shape[0]},
            )

        if np.any(weights < 0):
            raise RangeValidationError(
                "sample_weight must be non-negative.",
                component="sgd_regressor",
                details={"min_weight": float(np.min(weights))},
            )

        return weights

    def _ensure_feature_compatibility(self, X: np.ndarray) -> None:
        if self.coef_ is None or self.n_features_in_ is None:
            self._initialize_parameters(X.shape[1])
            return

        if X.shape[1] != self.n_features_in_:
            raise InvalidValueError(
                "Input feature dimension does not match the fitted model.",
                component="sgd_regressor",
                details={"expected": self.n_features_in_, "received": X.shape[1]},
                remediation="Use inputs with the same feature count used during initial fitting.",
            )

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.coef_) + self.intercept_

    def _compute_sample_loss(self, error: float) -> float:
        coef_penalty = 0.5 * self.alpha * float(np.dot(self.coef_, self.coef_)) if self.coef_ is not None else 0.0
        return 0.5 * float(error ** 2) + coef_penalty

    def _compute_gradients(self, x_i: np.ndarray, y_i: float) -> Tuple[float, np.ndarray, float]:
        pred = float(np.dot(x_i, self.coef_) + self.intercept_)
        error = pred - float(y_i)
        grad_coef = (error * x_i) + (self.alpha * self.coef_)
        grad_intercept = error if self.fit_intercept else 0.0
        return error, grad_coef, grad_intercept

    def _apply_gradient_clipping(self, grad_coef: np.ndarray, grad_intercept: float) -> Tuple[np.ndarray, float, float]:
        grad_norm = float(np.linalg.norm(grad_coef))
        if self.fit_intercept:
            grad_norm = float(np.sqrt(grad_norm ** 2 + grad_intercept ** 2))

        if self.clip_gradient is not None and grad_norm > self.clip_gradient and grad_norm > self.epsilon:
            scale = self.clip_gradient / grad_norm
            grad_coef = grad_coef * scale
            grad_intercept = grad_intercept * scale
            grad_norm = float(self.clip_gradient)

        return grad_coef, grad_intercept, grad_norm

    def _sgd_epoch(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> Dict[str, float]:
        indices = np.arange(X.shape[0])
        if self.shuffle:
            self.rng.shuffle(indices)

        epoch_loss = 0.0
        gradient_norms: List[float] = []
        max_update_norm = 0.0

        for idx in indices:
            x_i = X[idx]
            y_i = y[idx]
            weight = float(sample_weight[idx])

            error, grad_coef, grad_intercept = self._compute_gradients(x_i, y_i)
            grad_coef, grad_intercept, grad_norm = self._apply_gradient_clipping(grad_coef, grad_intercept)
            gradient_norms.append(grad_norm)

            effective_eta = self.eta * weight
            coef_before = self.coef_.copy()
            intercept_before = self.intercept_

            self.coef_ -= effective_eta * grad_coef
            if self.fit_intercept:
                self.intercept_ -= effective_eta * grad_intercept

            coef_update_norm = float(np.linalg.norm(self.coef_ - coef_before))
            if self.fit_intercept:
                coef_update_norm = float(np.sqrt(coef_update_norm ** 2 + (self.intercept_ - intercept_before) ** 2))
            max_update_norm = max(max_update_norm, coef_update_norm)

            epoch_loss += self._compute_sample_loss(error)

        epoch_loss /= max(1, X.shape[0])
        mean_gradient_norm = float(np.mean(gradient_norms)) if gradient_norms else 0.0
        return {
            "epoch_loss": float(epoch_loss),
            "mean_gradient_norm": mean_gradient_norm,
            "max_update_norm": float(max_update_norm),
        }

    def _has_converged(self, epoch_loss: float, max_update_norm: float) -> bool:
        if self._last_epoch_loss is None:
            return False

        loss_delta = abs(self._last_epoch_loss - epoch_loss)
        parameter_converged = max_update_norm <= self.tol
        loss_converged = loss_delta <= self.tol

        if self.convergence_check == "parameter_change":
            return parameter_converged
        if self.convergence_check == "loss":
            return loss_converged
        return parameter_converged or loss_converged

    def _update_learning_rate(self, epoch_loss: float) -> None:
        self.t_ += 1

        if self.learning_rate != "adaptive":
            return

        if epoch_loss + self.adaptive_tolerance < self._best_loss:
            self._best_loss = epoch_loss
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        if self._epochs_without_improvement >= self.adaptive_patience:
            self._current_eta = max(self.min_learning_rate, self._current_eta * self.adaptive_decay)
            self._epochs_without_improvement = 0

        if self.lr_decay_frequency > 0 and self.t_ % self.lr_decay_frequency == 0:
            self._current_eta = max(self.min_learning_rate, self._current_eta * self.lr_decay_factor)

    def fit(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]],
        y: Union[np.ndarray, Sequence[float], float, int],
        sample_weight: Optional[Union[np.ndarray, Sequence[float]]] = None,
        epochs: Optional[int] = None,
        reset: bool = True,
    ) -> "SGDRegressor":
        """Full-batch training entrypoint built on top of incremental updates."""
        if reset:
            self.reset()

        if epochs is not None:
            epochs = int(epochs)
            ensure_positive(epochs, "epochs", component="sgd_regressor")
            original_max_iter = self.max_iter
            self.max_iter = epochs
            try:
                return self.partial_fit(X, y, sample_weight=sample_weight)
            finally:
                self.max_iter = original_max_iter

        return self.partial_fit(X, y, sample_weight=sample_weight)

    def partial_fit(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]],
        y: Union[np.ndarray, Sequence[float], float, int],
        sample_weight: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> "SGDRegressor":
        """Incremental fit on a batch of samples."""
        printer.status("INIT", "Partial fit successfully initialized", "info")

        try:
            X_arr, y_arr = self._validate_and_prepare_X_y(X, y)
            weights = self._prepare_sample_weight(sample_weight, X_arr.shape[0])
            self._ensure_feature_compatibility(X_arr)

            for _ in range(self.max_iter):
                metrics = self._sgd_epoch(X_arr, y_arr, weights)
                epoch_loss = metrics["epoch_loss"]
                self._last_epoch_loss = epoch_loss
                self._last_gradient_norm = metrics["mean_gradient_norm"]
                self._last_update_norm = metrics["max_update_norm"]
                self.loss_history.append(epoch_loss)
                self.learning_rate_history.append(self.eta)
                self.gradient_norm_history.append(self._last_gradient_norm)
                self.update_norm_history.append(self._last_update_norm)
                self._trim_history()

                self._converged = self._has_converged(epoch_loss, self._last_update_norm)
                self._update_learning_rate(epoch_loss)

                if self._converged:
                    break

            self.n_samples_seen += X_arr.shape[0]
            self._is_fitted = True
            return self

        except AdaptiveError:
            raise
        except Exception as exc:
            raise wrap_exception(
                exc,
                AdaptiveRegressorError,
                "Unexpected failure during partial_fit.",
                component="sgd_regressor",
                details={"operation": "partial_fit"},
                remediation="Inspect the input arrays and regressor configuration for incompatible values.",
            ) from exc

    def predict(self, X: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]) -> np.ndarray:
        """Make predictions with the fitted regressor."""
        printer.status("INIT", "Predictor successfully initialized", "info")

        if not self.is_fitted:
            raise PredictionBeforeFitError(
                "Model must be fitted before calling predict().",
                component="sgd_regressor",
                remediation="Call fit() or partial_fit() before prediction.",
            )

        try:
            X_arr = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidTypeError(
                "X must be numeric and array-like.",
                component="sgd_regressor",
                cause=exc,
            ) from exc

        was_1d = False
        if X_arr.ndim == 1:
            was_1d = True
            X_arr = X_arr.reshape(1, -1)
        elif X_arr.ndim != 2:
            raise InvalidValueError(
                "X must be one-dimensional or two-dimensional.",
                component="sgd_regressor",
                details={"X_shape": X_arr.shape},
            )

        if X_arr.shape[1] != self.n_features_in_:
            raise InvalidValueError(
                "Prediction input dimension does not match the fitted model.",
                component="sgd_regressor",
                details={"expected": self.n_features_in_, "received": X_arr.shape[1]},
            )

        predictions = self._predict_internal(X_arr)
        return predictions.reshape(1) if was_1d else predictions

    def score(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]],
        y: Union[np.ndarray, Sequence[float], float, int],
    ) -> float:
        """Compute the coefficient of determination (R^2 score)."""
        printer.status("INIT", "Scorer successfully initialized", "info")

        X_arr, y_arr = self._validate_and_prepare_X_y(X, y)
        y_pred = self.predict(X_arr)

        residual_sum = float(np.sum((y_arr - y_pred) ** 2))
        total_sum = float(np.sum((y_arr - np.mean(y_arr)) ** 2))

        if total_sum <= self.epsilon:
            return 1.0 if residual_sum <= self.epsilon else 0.0

        return float(1.0 - (residual_sum / total_sum))

    def mean_squared_error(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]],
        y: Union[np.ndarray, Sequence[float], float, int],
    ) -> float:
        X_arr, y_arr = self._validate_and_prepare_X_y(X, y)
        y_pred = self.predict(X_arr)
        return float(np.mean((y_arr - y_pred) ** 2))

    def mean_absolute_error(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]],
        y: Union[np.ndarray, Sequence[float], float, int],
    ) -> float:
        X_arr, y_arr = self._validate_and_prepare_X_y(X, y)
        y_pred = self.predict(X_arr)
        return float(np.mean(np.abs(y_arr - y_pred)))

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Return normalized feature importance based on absolute coefficient magnitude."""
        printer.status("INIT", "Feature importance initialized", "info")

        if not self.is_fitted:
            return None

        importance = np.abs(self.coef_)
        total = float(np.sum(importance))
        if total <= self.epsilon:
            return np.zeros_like(importance)
        return importance / total

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return structured diagnostics for debugging, monitoring, and observability."""
        return {
            "is_fitted": self.is_fitted,
            "n_features_in": self.n_features_in_,
            "n_samples_seen": self.n_samples_seen,
            "learning_rate_schedule": self.learning_rate,
            "current_learning_rate": self.eta,
            "base_learning_rate": self.eta0,
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "shuffle": self.shuffle,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "converged": self._converged,
            "last_loss": self._last_epoch_loss,
            "last_gradient_norm": self._last_gradient_norm,
            "last_update_norm": self._last_update_norm,
            "coef_norm": float(np.linalg.norm(self.coef_)) if self.coef_ is not None else None,
            "intercept": float(self.intercept_),
            "history_lengths": {
                "loss": len(self.loss_history),
                "learning_rate": len(self.learning_rate_history),
                "gradient_norm": len(self.gradient_norm_history),
                "update_norm": len(self.update_norm_history),
            },
        }

    def save_model(self, filepath: Union[str, Path]) -> Path:
        """Persist the fitted regressor state to disk."""
        path = Path(filepath)
        payload = {
            "config": {
                "eta0": self.eta0,
                "learning_rate": self.learning_rate,
                "alpha": self.alpha,
                "power_t": self.power_t,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "fit_intercept": self.fit_intercept,
                "shuffle": self.shuffle,
                "random_state": self.random_state,
                "min_learning_rate": self.min_learning_rate,
                "adaptive_decay": self.adaptive_decay,
                "adaptive_patience": self.adaptive_patience,
                "adaptive_tolerance": self.adaptive_tolerance,
                "lr_decay_frequency": self.lr_decay_frequency,
                "lr_decay_factor": self.lr_decay_factor,
                "clip_gradient": self.clip_gradient,
                "track_history": self.track_history,
                "max_history": self.max_history,
                "weight_init_scale": self.weight_init_scale,
                "convergence_check": self.convergence_check,
                "epsilon": self.epsilon,
            },
            "state": {
                "coef_": self.coef_,
                "intercept_": self.intercept_,
                "n_features_in_": self.n_features_in_,
                "n_samples_seen": self.n_samples_seen,
                "t_": self.t_,
                "loss_history": self.loss_history,
                "learning_rate_history": self.learning_rate_history,
                "gradient_norm_history": self.gradient_norm_history,
                "update_norm_history": self.update_norm_history,
                "current_eta": self._current_eta,
                "best_loss": self._best_loss,
                "epochs_without_improvement": self._epochs_without_improvement,
                "is_fitted": self._is_fitted,
                "last_epoch_loss": self._last_epoch_loss,
                "last_gradient_norm": self._last_gradient_norm,
                "last_update_norm": self._last_update_norm,
                "converged": self._converged,
            },
        }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                pickle.dump(payload, handle)
            logger.info("SGD Regressor saved to %s", path)
            return path
        except Exception as exc:
            raise CheckpointSaveError(
                f"Failed to save SGD regressor checkpoint to {path}.",
                component="sgd_regressor",
                details={"filepath": str(path)},
                remediation="Verify the destination path and filesystem permissions.",
                cause=exc,
            ) from exc

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> "SGDRegressor":
        """Load a persisted regressor instance from disk."""
        path = Path(filepath)
        if not path.exists():
            raise CheckpointNotFoundError(
                f"SGD regressor checkpoint not found: {path}",
                component="sgd_regressor",
                details={"filepath": str(path)},
                remediation="Confirm the checkpoint path before loading.",
            )

        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            raise CheckpointLoadError(
                f"Failed to read SGD regressor checkpoint from {path}.",
                component="sgd_regressor",
                details={"filepath": str(path)},
                remediation="Ensure the checkpoint file is valid and readable.",
                cause=exc,
            ) from exc

        try:
            model = cls()
            state = payload["state"]
            model.coef_ = None if state["coef_"] is None else np.asarray(state["coef_"], dtype=np.float64)
            model.intercept_ = float(state["intercept_"])
            model.n_features_in_ = state["n_features_in_"]
            model.n_samples_seen = int(state["n_samples_seen"])
            model.t_ = int(state["t_"])
            model.loss_history = [float(v) for v in state.get("loss_history", [])]
            model.learning_rate_history = [float(v) for v in state.get("learning_rate_history", [])]
            model.gradient_norm_history = [float(v) for v in state.get("gradient_norm_history", [])]
            model.update_norm_history = [float(v) for v in state.get("update_norm_history", [])]
            model._current_eta = float(state.get("current_eta", model.eta0))
            model._best_loss = float(state.get("best_loss", np.inf))
            model._epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
            model._is_fitted = bool(state.get("is_fitted", model.coef_ is not None))
            model._last_epoch_loss = state.get("last_epoch_loss")
            model._last_gradient_norm = state.get("last_gradient_norm")
            model._last_update_norm = state.get("last_update_norm")
            model._converged = bool(state.get("converged", False))
            logger.info("SGD Regressor loaded from %s", path)
            return model
        except Exception as exc:
            raise CheckpointLoadError(
                f"Failed to restore SGD regressor state from {path}.",
                component="sgd_regressor",
                details={"filepath": str(path)},
                remediation="Ensure the checkpoint file matches the expected SGD regressor schema.",
                cause=exc,
            ) from exc

    def reset(self) -> None:
        """Reset model parameters and optimizer state for fresh training."""
        printer.status("INIT", "Reset initialized", "info")
        self._initialize_runtime_state()

    def __repr__(self) -> str:
        return (
            "SGDRegressor("
            f"learning_rate='{self.learning_rate}', "
            f"eta0={self.eta0}, "
            f"alpha={self.alpha}, "
            f"max_iter={self.max_iter}, "
            f"tol={self.tol}, "
            f"fit_intercept={self.fit_intercept}, "
            f"shuffle={self.shuffle}, "
            f"is_fitted={self.is_fitted}, "
            f"n_features_in_={self.n_features_in_})"
        )


if __name__ == "__main__":
    print("\n=== Running SGD Regressor ===\n")
    printer.status("TEST", "SGD Regressor initialized", "info")

    rng = np.random.default_rng(7)

    # Synthetic linear regression problem: y = 2.5*x0 - 1.2*x1 + 0.7*x2 + 0.5 + noise
    X_train = rng.normal(0.0, 1.0, size=(256, 3))
    true_coef = np.array([2.5, -1.2, 0.7], dtype=np.float64)
    true_intercept = 0.5
    noise = rng.normal(0.0, 0.05, size=256)
    y_train = X_train @ true_coef + true_intercept + noise

    X_eval = rng.normal(0.0, 1.0, size=(64, 3))
    y_eval = X_eval @ true_coef + true_intercept + rng.normal(0.0, 0.05, size=64)

    regressor = SGDRegressor()
    printer.status("TEST", str(regressor), "info")

    print("\n* * * * * Phase 2: Fit * * * * *\n")
    regressor.fit(X_train, y_train, epochs=200)
    printer.status("FIT", f"Current diagnostics: {regressor.get_diagnostics()}", "success")

    print("\n* * * * * Phase 3: Predict / Score * * * * *\n")
    predictions = regressor.predict(X_eval)
    r2 = regressor.score(X_eval, y_eval)
    mse = regressor.mean_squared_error(X_eval, y_eval)
    mae = regressor.mean_absolute_error(X_eval, y_eval)
    feature_importance = regressor.get_feature_importance()
    printer.status("PREDICT", f"Prediction sample: {predictions[:5]}", "success")
    printer.status("SCORE", f"R2: {r2:.6f} | MSE: {mse:.6f} | MAE: {mae:.6f}", "success")
    printer.status("FEATURE", f"Feature importance: {feature_importance}", "success")

    print("\n* * * * * Phase 4: Partial Fit * * * * *\n")
    X_incremental = rng.normal(0.0, 1.0, size=(32, 3))
    y_incremental = X_incremental @ true_coef + true_intercept + rng.normal(0.0, 0.05, size=32)
    sample_weight = np.linspace(0.5, 1.5, num=32)
    regressor.partial_fit(X_incremental, y_incremental, sample_weight=sample_weight)
    printer.status("PARTIAL_FIT", f"Updated diagnostics: {regressor.get_diagnostics()}", "success")

    print("\n* * * * * Phase 5: Save / Load * * * * *\n")
    checkpoint_path = Path("/tmp/sgd_regressor_checkpoint.pkl")
    regressor.save_model(checkpoint_path)
    restored = SGDRegressor.load_model(checkpoint_path)
    restored_predictions = restored.predict(X_eval[:5])
    printer.status("SAVE", f"Checkpoint saved to: {checkpoint_path}", "success")
    printer.status("LOAD", f"Restored prediction sample: {restored_predictions}", "success")

    print("\n* * * * * Phase 6: Reset * * * * *\n")
    regressor.reset()
    printer.status("RESET", str(regressor), "success")

    print("\n=== Test ran successfully ===\n")
