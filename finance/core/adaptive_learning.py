from __future__ import annotations

import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple
from scipy.stats import ttest_ind

from .utils.config_loader import load_global_config, get_config_section
from .utils.financial_errors import (InvalidConfigurationError, ModelTrainingError,
                                                 AdaptiveLearningError, ErrorContext, log_error,
                                                 ModelDriftError, ModelInferenceError,
                                                 PersistenceError, ValidationError)
from .finance_memory import FinanceMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Adaptive Learning")
printer = PrettyPrinter


DEFAULT_DRIFT_WINDOW = 1000
DEFAULT_HISTORY_SIZE = 1000
DEFAULT_UNCERTAINTY_HISTORY = 500
DEFAULT_MC_DROPOUT_PASSES = 64
DEFAULT_BASE_MODEL_UPDATE_INTERVAL = 25
DEFAULT_SIGNIFICANCE_LEVEL = 0.01
DEFAULT_META_HIDDEN_SIZE = 32
DEFAULT_MIN_LEARNING_RATE = 1e-5
DEFAULT_WEIGHT_FLOOR = 1e-6
DEFAULT_BASE_MODEL_LR_DIVISOR = 10.0
DEFAULT_GRAD_CLIP_NORM = 5.0


@dataclass(slots=True)
class DriftDiagnostics:
    detected: bool
    p_value: float
    t_statistic: float
    recent_mean: float
    historical_mean: float
    effect_size: float
    samples: int

    def to_dict(self) -> Dict[str, float | bool | int]:
        return {
            "detected": bool(self.detected),
            "p_value": float(self.p_value),
            "t_statistic": float(self.t_statistic),
            "recent_mean": float(self.recent_mean),
            "historical_mean": float(self.historical_mean),
            "effect_size": float(self.effect_size),
            "samples": int(self.samples),
        }


@dataclass(slots=True)
class PredictionSummary:
    prediction: float
    ensemble_prediction: float
    meta_prediction: float
    confidence: float
    uncertainty: float
    lower_bound: float
    upper_bound: float
    component_contributions: Dict[str, float]
    component_predictions: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": float(self.prediction),
            "ensemble_prediction": float(self.ensemble_prediction),
            "meta_prediction": float(self.meta_prediction),
            "confidence": float(self.confidence),
            "uncertainty": float(self.uncertainty),
            "lower_bound": float(self.lower_bound),
            "upper_bound": float(self.upper_bound),
            "component_contributions": dict(self.component_contributions),
            "component_predictions": dict(self.component_predictions),
        }


class MetaPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = DEFAULT_META_HIDDEN_SIZE) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 1:
            x = x.view(1, 1, -1)
        sequence_output, _ = self.lstm(x)
        features = self.dropout(sequence_output[:, -1, :])
        return self.fc(features)


class AdaptiveLearningSystem(nn.Module):
    """Autonomous adaptive learning system for financial prediction.

    Production-oriented upgrades:
    - robust validation and typed error handling
    - stable ensemble/meta-model training path
    - stateful drift detection over residual windows
    - memory-backed diagnostics and checkpointing
    - modular update and inference helpers
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = load_global_config()
        self.als_config = get_config_section("adaptive_learning_system")
        self.finance_memory = FinanceMemory()

        self.n_models = int(self.als_config.get("n_models", 3))
        self.market_features = int(self.als_config.get("market_features", 64))
        self.learning_rate = float(self.als_config.get("learning_rate", 0.01))
        self.threshold = float(self.als_config.get("drift_threshold", 2.5))
        self.diversity_strength = float(self.als_config.get("diversity_strength", 0.1))
        self.lr_decay = float(self.als_config.get("lr_decay", 0.9995))
        self.momentum = float(self.als_config.get("momentum", 0.9))
        self.regularization = float(self.als_config.get("regularization", 1e-4))

        self.drift_window_size = int(self.als_config.get("drift_window", DEFAULT_DRIFT_WINDOW))
        self.history_size = int(self.als_config.get("history_size", DEFAULT_HISTORY_SIZE))
        self.uncertainty_history_size = int(self.als_config.get("uncertainty_history_size", DEFAULT_UNCERTAINTY_HISTORY))
        self.mc_dropout_passes = int(self.als_config.get("mc_dropout_passes", DEFAULT_MC_DROPOUT_PASSES))
        self.base_model_update_interval = int(self.als_config.get("base_model_update_interval", DEFAULT_BASE_MODEL_UPDATE_INTERVAL))
        self.significance_level = float(self.als_config.get("significance_level", DEFAULT_SIGNIFICANCE_LEVEL))
        self.meta_hidden_size = int(self.als_config.get("meta_hidden_size", DEFAULT_META_HIDDEN_SIZE))
        self.min_learning_rate = float(self.als_config.get("min_learning_rate", DEFAULT_MIN_LEARNING_RATE))
        self.weight_floor = float(self.als_config.get("weight_floor", DEFAULT_WEIGHT_FLOOR))
        self.base_model_lr_divisor = float(self.als_config.get("base_model_lr_divisor", DEFAULT_BASE_MODEL_LR_DIVISOR))
        self.grad_clip_norm = float(self.als_config.get("grad_clip_norm", DEFAULT_GRAD_CLIP_NORM))

        self._validate_configuration()

        self.ensemble_models = self._create_ensemble()
        self.prediction_weights = nn.Parameter(torch.ones(self.n_models, dtype=torch.float32) / self.n_models)
        self.meta_model = self._build_meta_model()

        self.weight_optimizer = torch.optim.AdamW(
            [self.prediction_weights],
            lr=self.learning_rate,
            weight_decay=self.regularization,
        )
        self.meta_optimizer = torch.optim.AdamW(
            self.meta_model.parameters(),
            lr=max(self.learning_rate / 2.0, self.min_learning_rate),
            weight_decay=self.regularization,
        )
        self.base_model_optimizers = [
            torch.optim.AdamW(
                model.parameters(),
                lr=max(self.learning_rate / self.base_model_lr_divisor, self.min_learning_rate),
                weight_decay=self.regularization,
            )
            for model in self.ensemble_models
        ]

        self.window: Deque[float] = deque(maxlen=self.drift_window_size)
        self.error_history: Deque[float] = deque(maxlen=self.history_size)
        self.prediction_history: Deque[List[float]] = deque(maxlen=self.history_size)
        self.uncertainty_bounds: Deque[Tuple[float, float]] = deque(maxlen=self.uncertainty_history_size)
        self.target_history: Deque[float] = deque(maxlen=self.history_size)
        self.residual_history: Deque[float] = deque(maxlen=self.history_size)
        self.learning_rate_history: Deque[float] = deque(maxlen=self.history_size)
        self.velocity = np.zeros(self.n_models, dtype=np.float64)
        self.shadow_weights = self.prediction_weights.detach().cpu().numpy().copy()

        self.last_p_value = 1.0
        self.last_drift_detected = False
        self.last_drift_t_statistic = 0.0
        self.update_counter = 0
        self.last_update_timestamp: Optional[float] = None

        self._init_memory()
        logger.info(
            "Adaptive Learning initialized | n_models=%s market_features=%s learning_rate=%.6f drift_threshold=%.4f",
            self.n_models,
            self.market_features,
            self.learning_rate,
            self.threshold,
        )

    # ------------------------------------------------------------------
    # Configuration and context helpers
    # ------------------------------------------------------------------

    def _context(self, operation: str, **metadata: Any) -> ErrorContext:
        return ErrorContext(
            component="adaptive_learning",
            operation=operation,
            metadata=metadata or {},
        )

    def _validate_configuration(self) -> None:
        if self.n_models <= 0:
            raise InvalidConfigurationError(
                "adaptive_learning_system.n_models must be a positive integer.",
                context=self._context("validate_config", n_models=self.n_models),
            )
        if self.market_features <= 0:
            raise InvalidConfigurationError(
                "adaptive_learning_system.market_features must be a positive integer.",
                context=self._context("validate_config", market_features=self.market_features),
            )
        if self.learning_rate <= 0:
            raise InvalidConfigurationError(
                "adaptive_learning_system.learning_rate must be positive.",
                context=self._context("validate_config", learning_rate=self.learning_rate),
            )
        if self.threshold <= 0:
            raise InvalidConfigurationError(
                "adaptive_learning_system.drift_threshold must be positive.",
                context=self._context("validate_config", drift_threshold=self.threshold),
            )
        if self.significance_level <= 0 or self.significance_level >= 1:
            raise InvalidConfigurationError(
                "adaptive_learning_system.significance_level must be in (0, 1).",
                context=self._context("validate_config", significance_level=self.significance_level),
            )

    def _validate_input_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValidationError(
                "Input x must be a torch.Tensor.",
                context=self._context("validate_input"),
            )
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValidationError(
                "Input x must have shape [batch, features] or [features].",
                context=self._context("validate_input", shape=tuple(x.shape)),
            )
        if x.shape[-1] != self.market_features:
            raise ValidationError(
                "Input feature dimension does not match adaptive_learning_system.market_features.",
                context=self._context(
                    "validate_input",
                    shape=tuple(x.shape),
                    expected_features=self.market_features,
                ),
            )
        return x.float()

    def _validate_target(self, y_true: float | int | torch.Tensor) -> torch.Tensor:
        if isinstance(y_true, torch.Tensor):
            target = y_true.float().reshape(-1)
            if target.numel() != 1:
                raise ValidationError(
                    "Target tensor must contain exactly one scalar value.",
                    context=self._context("validate_target", shape=tuple(y_true.shape)),
                )
            return target.view(1, 1)
        if not isinstance(y_true, (float, int)):
            raise ValidationError(
                "y_true must be numeric.",
                context=self._context("validate_target", y_true_type=type(y_true).__name__),
            )
        return torch.tensor([[float(y_true)]], dtype=torch.float32)

    def _init_memory(self) -> None:
        system_records = (
            ("ensemble_weights", {}, ["system_state", "adaptive_learning"], "critical"),
            ("drift_detection", {}, ["system_state", "adaptive_learning"], "high"),
            ("error_history", [], ["system_state", "adaptive_learning"], "medium"),
        )
        for data_type, data, tags, priority in system_records:
            try:
                self.finance_memory.add_financial_data(
                    data=data,
                    data_type=data_type,
                    tags=tags,
                    priority=priority,
                    metadata={"component": "adaptive_learning", "bootstrap": True},
                )
            except Exception as exc:
                handled = PersistenceError(
                    "Failed to initialize adaptive learning memory records.",
                    context=self._context("init_memory", data_type=data_type),
                    cause=exc,
                )
                log_error(handled, logger_=logger)
                raise handled from exc

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    def update_model(self, new_model: nn.Module) -> str:
        if not isinstance(new_model, nn.Module):
            raise ValidationError(
                "new_model must be an nn.Module.",
                context=self._context("update_model", model_type=type(new_model).__name__),
            )
        try:
            return self.finance_memory.add_financial_data(
                data={
                    "timestamp": time.time(),
                    "state_dict": {k: v.detach().cpu().tolist() for k, v in new_model.state_dict().items()},
                },
                data_type="model_state",
                tags=["production_model", "adaptive_learning"],
                priority="high",
                metadata={"component": "adaptive_learning"},
            )
        except Exception as exc:
            handled = PersistenceError(
                "Failed to persist model state to finance memory.",
                context=self._context("update_model"),
                cause=exc,
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def _create_ensemble(self) -> nn.ModuleList:
        models = nn.ModuleList()
        for i in range(self.n_models):
            hidden_size = 32 + i * 8
            activation: nn.Module = nn.ReLU() if i % 2 == 0 else nn.Tanh()
            model = nn.Sequential(
                nn.Linear(self.market_features, hidden_size),
                activation,
                nn.LayerNorm(hidden_size) if i % 2 == 0 else nn.Identity(),
                nn.Dropout(p=min(0.1 + i * 0.03, 0.25)),
                nn.Linear(hidden_size, max(hidden_size // 2, 8)),
                activation,
                nn.Linear(max(hidden_size // 2, 8), 1),
            )
            models.append(model)
        return models

    def _build_meta_model(self) -> nn.Module:
        return MetaPredictor(input_size=self.n_models, hidden_size=self.meta_hidden_size)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self._validate_input_tensor(x)
        predictions: Dict[str, torch.Tensor] = {}
        for i, model in enumerate(self.ensemble_models):
            predictions[f"model_{i}"] = model(x)

        normalized_weights = torch.clamp(self.prediction_weights, min=self.weight_floor)
        normalized_weights = normalized_weights / normalized_weights.sum()
        weighted_pred = sum(
            normalized_weights[i] * predictions[f"model_{i}"]
            for i in range(self.n_models)
        )
        return weighted_pred, predictions

    def robust_predict(self, x: torch.Tensor) -> Dict[str, float | Dict[str, float]]:
        try:
            x = self._validate_input_tensor(x)
            prior_training_state = self.training
            prior_model_states = [model.training for model in self.ensemble_models]
            for model in self.ensemble_models:
                model.eval()

            with torch.no_grad():
                weighted_pred, component_preds = self.forward(x)
                base_pred = float(weighted_pred.reshape(-1)[0].item())
                component_values = torch.stack(
                    [component_preds[f"model_{i}"].reshape(-1) for i in range(self.n_models)],
                    dim=-1,
                )
                component_scalar_map = {
                    f"model_{i}": float(component_preds[f"model_{i}"].reshape(-1)[0].item())
                    for i in range(self.n_models)
                }
                meta_pred, meta_std = self._monte_carlo_uncertainty(component_values)

            hist_vol = float(np.std(list(self.error_history)[-50:] or [0.0]) * 1.96)
            combined_std = float(np.sqrt(max(meta_std, 0.0) ** 2 + hist_vol ** 2))
            final_prediction = float(0.7 * base_pred + 0.3 * meta_pred)
            lower_bound = float(base_pred - 2.0 * combined_std)
            upper_bound = float(base_pred + 2.0 * combined_std)
            confidence = float(np.exp(-combined_std))

            summary = PredictionSummary(
                prediction=final_prediction,
                ensemble_prediction=base_pred,
                meta_prediction=float(meta_pred),
                confidence=confidence,
                uncertainty=combined_std,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                component_contributions={
                    f"model_{i}": float(self.prediction_weights[i].detach().item())
                    for i in range(self.n_models)
                },
                component_predictions=component_scalar_map,
            )
            self.uncertainty_bounds.append((lower_bound, upper_bound))
            return summary.to_dict()
        except ValidationError:
            raise
        except Exception as exc:
            handled = ModelInferenceError(
                "Robust prediction failed.",
                context=self._context("robust_predict"),
                cause=exc,
            )
            log_error(handled, logger_=logger)
            raise handled from exc
        finally:
            if "prior_model_states" in locals():
                for model, state in zip(self.ensemble_models, prior_model_states):
                    model.train(state)
            if "prior_training_state" in locals():
                self.train(prior_training_state)

    def _monte_carlo_uncertainty(self, component_values: torch.Tensor) -> Tuple[float, float]:
        prior_state = self.meta_model.training
        self.meta_model.train(True)
        with torch.no_grad():
            mc_predictions = [
                float(self.meta_model(component_values).reshape(-1)[0].item())
                for _ in range(max(1, self.mc_dropout_passes))
            ]
        self.meta_model.train(prior_state)
        return float(np.mean(mc_predictions)), float(np.std(mc_predictions))

    # ------------------------------------------------------------------
    # Training and adaptation
    # ------------------------------------------------------------------

    def update(self, x: torch.Tensor, y_true: float) -> Dict[str, Any]:
        try:
            x = self._validate_input_tensor(x)
            target = self._validate_target(y_true)

            weighted_pred, component_preds = self.forward(x)
            predicted_value = float(weighted_pred.detach().reshape(-1)[0].item())
            residual = predicted_value - float(target.item())
            error = float(residual ** 2)

            self.window.append(float(residual))
            self.error_history.append(error)
            self.target_history.append(float(target.item()))
            self.residual_history.append(float(residual))
            component_vector = [float(component_preds[f"model_{i}"].detach().reshape(-1)[0].item()) for i in range(self.n_models)]
            self.prediction_history.append(component_vector)

            drift_info = self._detect_drift()
            self.last_p_value = drift_info.p_value
            self.last_drift_t_statistic = drift_info.t_statistic
            self.last_drift_detected = drift_info.detected
            if drift_info.detected:
                logger.warning(
                    "Concept drift detected | p_value=%.6f t_stat=%.6f recent_mean=%.6f historical_mean=%.6f",
                    drift_info.p_value,
                    drift_info.t_statistic,
                    drift_info.recent_mean,
                    drift_info.historical_mean,
                )
                self._reset_weights(reason="concept_drift")

            recent_volatility = float(np.std(list(self.error_history)[-20:] or [0.0]))
            self.learning_rate = max(
                self.min_learning_rate,
                float(self.learning_rate * self.lr_decay / (1.0 + recent_volatility)),
            )
            self.learning_rate_history.append(self.learning_rate)
            self._update_optimizer_learning_rates()

            weight_updates = self._update_weights(component_preds, target)
            meta_loss = self._update_meta_model(component_vector, target)

            self.update_counter += 1
            if self.update_counter % self.base_model_update_interval == 0:
                self._update_base_models(x, target)

            self.last_update_timestamp = time.time()
            self._update_memory(error, drift_info, weight_updates, meta_loss)

            return {
                "error": error,
                "learning_rate": float(self.learning_rate),
                "drift_detected": bool(drift_info.detected),
                "drift_p_value": float(drift_info.p_value),
                "drift_t_statistic": float(drift_info.t_statistic),
                "weight_updates": weight_updates.tolist(),
                "meta_loss": float(meta_loss),
            }
        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = ModelTrainingError(
                "Adaptive learning update failed.",
                context=self._context("update"),
                cause=exc,
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def _update_memory(
        self,
        error: float,
        drift_info: DriftDiagnostics,
        weight_updates: np.ndarray,
        meta_loss: float,
    ) -> None:
        timestamp = time.time()
        memory_payloads = (
            (
                "ensemble_weights",
                {
                    "weights": self.prediction_weights.detach().cpu().numpy().tolist(),
                    "updates": weight_updates.tolist(),
                    "timestamp": timestamp,
                    "shadow_weights": self.shadow_weights.tolist(),
                },
            ),
            (
                "drift_detection",
                {
                    "detected": drift_info.detected,
                    "threshold": self.threshold,
                    "window": list(self.window),
                    "timestamp": timestamp,
                    "p_value": drift_info.p_value,
                    "t_statistic": drift_info.t_statistic,
                    "effect_size": drift_info.effect_size,
                },
            ),
            (
                "error_history",
                {
                    "error": error,
                    "history": list(self.error_history),
                    "timestamp": timestamp,
                    "meta_loss": meta_loss,
                },
            ),
        )

        for data_type, data in memory_payloads:
            try:
                self.finance_memory.add_financial_data(
                    data=data,
                    data_type=data_type,
                    tags=["adaptive_learning", "system_state"],
                    priority="high" if data_type != "error_history" else "medium",
                    metadata={"component": "adaptive_learning"},
                )
            except Exception as exc:
                handled = PersistenceError(
                    "Failed to write adaptive learning state to finance memory.",
                    context=self._context("update_memory", data_type=data_type),
                    cause=exc,
                )
                log_error(handled, logger_=logger)
                raise handled from exc

    def _update_weights(self, component_preds: Dict[str, torch.Tensor], target: torch.Tensor) -> np.ndarray:
        self.weight_optimizer.zero_grad(set_to_none=True)
        normalized_weights = torch.clamp(self.prediction_weights, min=self.weight_floor)
        normalized_weights = normalized_weights / normalized_weights.sum()
        weighted_pred = sum(
            normalized_weights[i] * component_preds[f"model_{i}"]
            for i in range(self.n_models)
        )
        error_loss = F.mse_loss(weighted_pred, target)
        diversity_loss = self._diversity_regularization(component_preds)
        entropy_bonus = -torch.sum(normalized_weights * torch.log(normalized_weights + 1e-12))
        total_loss = error_loss + self.diversity_strength * diversity_loss - 0.01 * entropy_bonus
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.prediction_weights], max_norm=self.grad_clip_norm)
        self.weight_optimizer.step()

        with torch.no_grad():
            self.prediction_weights.data = torch.clamp(self.prediction_weights.data, min=self.weight_floor)
            self.prediction_weights.data /= self.prediction_weights.data.sum()
            current_weights = self.prediction_weights.detach().cpu().numpy()
            self.velocity = self.momentum * self.velocity + (1.0 - self.momentum) * (current_weights - self.shadow_weights)
            self.shadow_weights = 0.9 * self.shadow_weights + 0.1 * current_weights

        return self.prediction_weights.detach().cpu().numpy().copy()

    def _diversity_regularization(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred_tensor = torch.stack([predictions[f"model_{i}"].reshape(-1) for i in range(self.n_models)], dim=0)
        centered = pred_tensor - pred_tensor.mean(dim=0, keepdim=True)
        covariance = centered @ centered.T / max(1, centered.shape[1])
        off_diagonal_penalty = covariance.sum() - torch.trace(covariance)
        disagreement_bonus = -torch.trace(covariance)
        return off_diagonal_penalty + disagreement_bonus

    def _update_meta_model(self, component_vector: Sequence[float], target: torch.Tensor) -> float:
        self.meta_optimizer.zero_grad(set_to_none=True)
        meta_input = torch.tensor(component_vector, dtype=torch.float32).view(1, 1, -1)
        meta_prediction = self.meta_model(meta_input)
        meta_loss = F.mse_loss(meta_prediction, target)
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_model.parameters(), max_norm=self.grad_clip_norm)
        self.meta_optimizer.step()
        return float(meta_loss.detach().item())

    def _update_base_models(self, x: torch.Tensor, target: torch.Tensor) -> None:
        for model, optimizer in zip(self.ensemble_models, self.base_model_optimizers):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = F.mse_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip_norm)
            optimizer.step()

    def _update_optimizer_learning_rates(self) -> None:
        for param_group in self.weight_optimizer.param_groups:
            param_group["lr"] = self.learning_rate
        for param_group in self.meta_optimizer.param_groups:
            param_group["lr"] = max(self.learning_rate / 2.0, self.min_learning_rate)
        base_lr = max(self.learning_rate / self.base_model_lr_divisor, self.min_learning_rate)
        for optimizer in self.base_model_optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = base_lr

    # ------------------------------------------------------------------
    # Diagnostics and drift
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Any]:
        recent_errors = list(self.error_history)[-50:]
        uncertainty_widths = [upper - lower for lower, upper in self.uncertainty_bounds]
        diagnostics = {
            "current_weights": self.prediction_weights.detach().cpu().numpy().tolist(),
            "shadow_weights": self.shadow_weights.tolist(),
            "learning_rate": float(self.learning_rate),
            "recent_error": float(np.mean(recent_errors)) if recent_errors else 0.0,
            "error_volatility": float(np.std(recent_errors)) if recent_errors else 0.0,
            "drift_detected": bool(self.last_drift_detected),
            "model_diversity": float(self._calculate_diversity()),
            "p_value": float(self.last_p_value),
            "t_statistic": float(self.last_drift_t_statistic),
            "significance_level": float(self.significance_level),
            "prediction_count": int(len(self.prediction_history)),
            "update_counter": int(self.update_counter),
            "mean_uncertainty_band_width": float(np.mean(uncertainty_widths)) if uncertainty_widths else 0.0,
            "last_update_timestamp": float(self.last_update_timestamp) if self.last_update_timestamp else None,
        }

        p_val = diagnostics["p_value"]
        if p_val < 0.01:
            diagnostics["p_interpretation"] = "Very strong evidence against model stability"
        elif p_val < 0.05:
            diagnostics["p_interpretation"] = "Strong evidence against model stability"
        elif p_val < 0.1:
            diagnostics["p_interpretation"] = "Weak evidence against model stability"
        else:
            diagnostics["p_interpretation"] = "No significant evidence against model stability"
        return diagnostics

    def _detect_drift(self) -> DriftDiagnostics:
        if len(self.window) < max(20, self.drift_window_size // 5):
            return DriftDiagnostics(False, 1.0, 0.0, 0.0, 0.0, 0.0, len(self.window))

        values = np.array(self.window, dtype=np.float64)
        split = len(values) // 2
        historical = values[:split]
        recent = values[split:]
        if len(historical) < 10 or len(recent) < 10:
            return DriftDiagnostics(False, 1.0, 0.0, 0.0, 0.0, 0.0, len(values))

        try:
            t_statistic, p_value = ttest_ind(recent, historical, equal_var=False, nan_policy="omit")
            t_statistic = float(0.0 if np.isnan(t_statistic) else t_statistic)
            p_value = float(1.0 if np.isnan(p_value) else p_value)
        except Exception as exc:
            handled = ModelDriftError(
                "Drift detection failed.",
                context=self._context("detect_drift", samples=len(values)),
                cause=exc,
            )
            log_error(handled, logger_=logger)
            raise handled from exc

        recent_mean = float(np.mean(recent)) if len(recent) else 0.0
        historical_mean = float(np.mean(historical)) if len(historical) else 0.0
        denom = float(np.std(historical) + 1e-8)
        effect_size = float(abs(recent_mean - historical_mean) / denom)
        detected = bool(p_value < self.significance_level and abs(t_statistic) > self.threshold)
        return DriftDiagnostics(
            detected=detected,
            p_value=p_value,
            t_statistic=t_statistic,
            recent_mean=recent_mean,
            historical_mean=historical_mean,
            effect_size=effect_size,
            samples=len(values),
        )

    def _calculate_diversity(self) -> float:
        if not self.prediction_history:
            return 0.0
        recent_predictions = np.array(list(self.prediction_history)[-100:], dtype=np.float64)
        if recent_predictions.ndim != 2 or recent_predictions.shape[1] <= 1:
            return 0.0
        return float(np.mean(np.var(recent_predictions, axis=1)))

    def _reset_weights(self, reason: str = "manual") -> None:
        with torch.no_grad():
            self.prediction_weights.data = torch.ones_like(self.prediction_weights) / self.n_models
        self.velocity = np.zeros_like(self.velocity)
        self.window.clear()
        logger.info("Adaptive learning weights reset | reason=%s", reason)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _serializable_state(self) -> Dict[str, Any]:
        return {
            "prediction_weights": self.prediction_weights.detach().cpu(),
            "ensemble_state": [model.state_dict() for model in self.ensemble_models],
            "meta_model_state": self.meta_model.state_dict(),
            "error_history": list(self.error_history),
            "prediction_history": list(self.prediction_history),
            "target_history": list(self.target_history),
            "residual_history": list(self.residual_history),
            "uncertainty_bounds": list(self.uncertainty_bounds),
            "window": list(self.window),
            "learning_rate": float(self.learning_rate),
            "shadow_weights": self.shadow_weights.tolist(),
            "velocity": self.velocity.tolist(),
            "last_p_value": float(self.last_p_value),
            "last_drift_detected": bool(self.last_drift_detected),
            "last_drift_t_statistic": float(self.last_drift_t_statistic),
            "update_counter": int(self.update_counter),
            "config": self.config,
        }

    def save_state(self, path: str) -> str:
        if not path or not str(path).strip():
            raise ValidationError(
                "path is required to save adaptive learning state.",
                context=self._context("save_state"),
            )
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            state = self._serializable_state()
            torch.save(state, path)
            self.finance_memory.add_financial_data(
                data={"path": path, "saved_at": time.time(), "state": self._serializable_state()},
                data_type="system_checkpoint",
                tags=["adaptive_learning", "checkpoint"],
                priority="critical",
                metadata={"component": "adaptive_learning", "path": path},
            )
            logger.info("Saved adaptive learning state to %s", path)
            return path
        except Exception as exc:
            handled = PersistenceError(
                "Failed to save adaptive learning state.",
                context=self._context("save_state", path=path),
                cause=exc,
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def load_state(self, path: str) -> bool:
        if not path or not str(path).strip():
            raise ValidationError(
                "path is required to load adaptive learning state.",
                context=self._context("load_state"),
            )
        try:
            state: Optional[Mapping[str, Any]] = None
            records = self.finance_memory.query(
                data_type="system_checkpoint",
                metadata_filters={"path": path},
                limit=1,
            )
            if records:
                state = records[0].get("data", {}).get("state")
            if state is None:
                state = torch.load(path, map_location="cpu", weights_only=False)

            prediction_weights = state["prediction_weights"]
            if isinstance(prediction_weights, np.ndarray):
                prediction_weights = torch.tensor(prediction_weights, dtype=torch.float32)
            self.prediction_weights.data = prediction_weights.detach().clone().float()
            self.prediction_weights.data /= self.prediction_weights.data.sum()

            for model, model_state in zip(self.ensemble_models, state["ensemble_state"]):
                model.load_state_dict(model_state)
            self.meta_model.load_state_dict(state["meta_model_state"])

            self.error_history = deque(state.get("error_history", []), maxlen=self.history_size)
            self.prediction_history = deque(state.get("prediction_history", []), maxlen=self.history_size)
            self.target_history = deque(state.get("target_history", []), maxlen=self.history_size)
            self.residual_history = deque(state.get("residual_history", []), maxlen=self.history_size)
            self.uncertainty_bounds = deque(state.get("uncertainty_bounds", []), maxlen=self.uncertainty_history_size)
            self.window = deque(state.get("window", []), maxlen=self.drift_window_size)
            self.learning_rate = float(state.get("learning_rate", self.learning_rate))
            self.shadow_weights = np.array(state.get("shadow_weights", self.shadow_weights), dtype=np.float64)
            self.velocity = np.array(state.get("velocity", self.velocity), dtype=np.float64)
            self.last_p_value = float(state.get("last_p_value", self.last_p_value))
            self.last_drift_detected = bool(state.get("last_drift_detected", self.last_drift_detected))
            self.last_drift_t_statistic = float(state.get("last_drift_t_statistic", self.last_drift_t_statistic))
            self.update_counter = int(state.get("update_counter", self.update_counter))
            self._update_optimizer_learning_rates()
            logger.info("Loaded adaptive learning state from %s", path)
            return True
        except Exception as exc:
            handled = PersistenceError(
                "Failed to load adaptive learning state.",
                context=self._context("load_state", path=path),
                cause=exc,
            )
            log_error(handled, logger_=logger)
            raise handled from exc


if __name__ == "__main__":  # pragma: no cover
    printer.status("INIT", "Testing Adaptive Learning System", "info")
    als = AdaptiveLearningSystem()
    als.eval()

    x = torch.randn(1, als.market_features)
    y_true = 0.0
    path = "finance/state/adaptive_learning_state.pt"

    printer.pretty("forward", als.forward(x), "success")
    printer.pretty("prediction", als.robust_predict(x=x), "success")
    printer.pretty("update", als.update(x=x, y_true=y_true), "success")
    printer.pretty("diagnostics", als.get_diagnostics(), "info")
    printer.pretty("save", als.save_state(path=path), "success")
    printer.pretty("load", als.load_state(path=path), "success")
    printer.status("RESULT", "Adaptive Learning System test completed", "success")
