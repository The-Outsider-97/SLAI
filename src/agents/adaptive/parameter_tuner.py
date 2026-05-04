from __future__ import annotations

import pickle
import numpy as np

from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union
from collections import deque

from src.tuning.tuner import HyperparamTuner
from src.tuning.utils.config_loader import get_config_section as get_tuner_config
from .utils.config_loader import load_global_config, get_config_section as get_param_config
from .utils.adaptive_errors import *
from .adaptive_memory import MultiModalMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Parameter Tuner")
printer = PrettyPrinter


class LearningParameterTuner:
    """
    Adaptive hyperparameter tuner for exploration/learning dynamics.

    This component is responsible for online parameter adaptation during learning,
    bounded updates for safety and stability, exploration decay scheduling, and
    optional integration with the broader hyperparameter-search stack.

    Production-oriented features
    ----------------------------
    - Config-driven parameter bounds and adaptation dynamics.
    - Structured error handling integrated with `adaptive_errors`.
    - Safe online updates for learning rate, exploration rate, discount factor,
      and temperature.
    - Rolling performance diagnostics and adaptation history tracking.
    - Memory integration for parameter logging and intervention analysis.
    - Worker/application helpers for propagating tuned parameters into learners.
    - Checkpoint save/load helpers for reproducibility and recovery.

    Public compatibility preserved
    ------------------------------
    The core public methods expected by the adaptive stack are preserved:
        - adapt(recent_rewards)
        - run_hyperparameter_tuning(evaluation_function, ...)
        - decay_exploration(decay_factor=...)
        - update_performance(reward)
        - get_params(include_metadata=False)
        - adaptive_discount_factor(state_visits)
        - temperature_schedule(episode)
        - reset(params_to_reset=None)
    """

    TRACKED_PARAMS: Tuple[str, ...] = (
        "learning_rate",
        "exploration_rate",
        "discount_factor",
        "temperature",
    )

    def __init__(
        self,
        initial_params: Optional[Mapping[str, Any]] = None,
        memory: Optional[MultiModalMemory] = None,
        tuner_cls: Type[HyperparamTuner] = HyperparamTuner,
    ) -> None:
        self.config = load_global_config()
        self.tuner_config = get_param_config("parameter_tuner")
        self.tuning_section = get_tuner_config("tuning")

        self._load_config()
        self.memory = memory if memory is not None else MultiModalMemory()
        self.tuner_cls = tuner_cls

        if memory is not None:
            ensure_instance(memory, MultiModalMemory, "memory", component="parameter_tuner")
        ensure_not_none(self.tuner_cls, "tuner_cls", component="parameter_tuner")

        self.base_params = {
            "learning_rate": self.base_learning_rate,
            "exploration_rate": self.base_exploration_rate,
            "discount_factor": self.base_discount_factor,
            "temperature": self.base_temperature,
        }

        self.initial_params = self._normalize_candidate_params(initial_params or {}, allow_partial=True)
        self.params: Dict[str, float] = {**self.base_params, **self.initial_params}
        self._apply_bounds()

        self.performance_history: Deque[float] = deque(maxlen=self.history_size)
        self.adaptation_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_adaptation_history)
        self.tuning_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_tuning_history)
        self._episode_counter = 0
        self._step_counter = 0
        self._last_adaptation: Optional[Dict[str, Any]] = None
        self._last_tuning_result: Optional[Dict[str, Any]] = None

        logger.info("Learning Parameter Tuner successfully initialized with base params: %s", self.base_params)

    def _load_config(self) -> None:
        """Load and validate parameter-tuner configuration from adaptive_config.yaml."""
        try:
            self.base_learning_rate = float(self.tuner_config.get("base_learning_rate", 0.01))
            self.base_exploration_rate = float(self.tuner_config.get("base_exploration_rate", 0.30))
            self.base_discount_factor = float(self.tuner_config.get("base_discount_factor", 0.95))
            self.base_temperature = float(self.tuner_config.get("base_temperature", 1.00))

            self._min_learning_rate = float(self.tuner_config.get("min_learning_rate", 1e-4))
            self._max_learning_rate = float(self.tuner_config.get("max_learning_rate", 0.1))
            self._min_exploration = float(self.tuner_config.get("min_exploration", 0.01))
            self._max_exploration = float(self.tuner_config.get("max_exploration", 1.0))
            self._min_discount_factor = float(self.tuner_config.get("min_discount_factor", 0.80))
            self._max_discount_factor = float(self.tuner_config.get("max_discount_factor", 0.999))
            self._min_temperature = float(self.tuner_config.get("min_temperature", 0.05))
            self._max_temperature = float(self.tuner_config.get("max_temperature", 5.0))

            self.history_size = int(self.tuner_config.get("history_size", 100))
            self.max_adaptation_history = int(self.tuner_config.get("max_adaptation_history", 500))
            self.max_tuning_history = int(self.tuner_config.get("max_tuning_history", 200))
            self.performance_baseline_window = int(self.tuner_config.get("performance_baseline_window", 10))
            self.stability_window = int(self.tuner_config.get("stability_window", 10))
            self.trend_window = int(self.tuner_config.get("trend_window", 5))

            self.stable_variance_threshold = float(self.tuner_config.get("stable_variance_threshold", 0.10))
            self.high_variance_threshold = float(self.tuner_config.get("high_variance_threshold", 1.00))
            self.negative_reward_threshold = float(self.tuner_config.get("negative_reward_threshold", 0.0))

            self.stable_lr_decay = float(self.tuner_config.get("stable_lr_decay", 0.995))
            self.volatile_lr_growth = float(self.tuner_config.get("volatile_lr_growth", 1.01))
            self.lr_trend_bonus = float(self.tuner_config.get("lr_trend_bonus", 0.0025))

            self.negative_reward_exploration_boost = float(
                self.tuner_config.get("negative_reward_exploration_boost", 1.10)
            )
            self.positive_reward_exploration_decay = float(
                self.tuner_config.get("positive_reward_exploration_decay", 0.995)
            )
            self.exploration_decay = float(self.tuner_config.get("exploration_decay", 0.9995))

            self.discount_familiar_growth = float(self.tuner_config.get("discount_familiar_growth", 1.01))
            self.discount_novelty_decay = float(self.tuner_config.get("discount_novelty_decay", 0.99))

            self.temperature_growth = float(self.tuner_config.get("temperature_growth", 1.02))
            self.temperature_decay = float(self.tuner_config.get("temperature_decay", 0.995))

            self.state_visit_high_threshold = int(self.tuner_config.get("state_visit_high_threshold", 100))
            self.state_visit_low_threshold = int(self.tuner_config.get("state_visit_low_threshold", 5))

            self.schedule_initial_temperature = float(self.tuner_config.get("schedule_initial_temperature", 1.0))
            self.schedule_final_temperature = float(self.tuner_config.get("schedule_final_temperature", 0.1))
            self.schedule_decay_episodes = int(self.tuner_config.get("schedule_decay_episodes", 1000))

            self.default_apply_best_params = bool(self.tuner_config.get("default_apply_best_params", True))
            self.estimate_delta_from_memory = bool(self.tuner_config.get("estimate_delta_from_memory", True))
            self.log_every_update = bool(self.tuner_config.get("log_every_update", True))
            self.tuning_model_type = self.tuner_config.get("tuning_model_type", None)
            self.checkpoint_protocol = int(self.tuner_config.get("checkpoint_protocol", pickle.HIGHEST_PROTOCOL))
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationValueError(
                "Failed to parse parameter_tuner configuration values.",
                component="parameter_tuner",
                details={"section": "parameter_tuner"},
                remediation="Ensure all parameter_tuner values are valid scalars.",
                cause=exc,
            ) from exc

        ensure_positive(self.base_learning_rate, "base_learning_rate", component="parameter_tuner")
        ensure_probability(self.base_exploration_rate, "base_exploration_rate", component="parameter_tuner")
        ensure_in_range(self.base_discount_factor, "base_discount_factor", minimum=0.0, maximum=1.0, component="parameter_tuner")
        ensure_positive(self.base_temperature, "base_temperature", component="parameter_tuner")

        ensure_positive(self._min_learning_rate, "min_learning_rate", component="parameter_tuner")
        ensure_positive(self._max_learning_rate, "max_learning_rate", component="parameter_tuner")
        ensure_probability(self._min_exploration, "min_exploration", component="parameter_tuner")
        ensure_probability(self._max_exploration, "max_exploration", component="parameter_tuner")
        ensure_in_range(self._min_discount_factor, "min_discount_factor", minimum=0.0, maximum=1.0, component="parameter_tuner")
        ensure_in_range(self._max_discount_factor, "max_discount_factor", minimum=0.0, maximum=1.0, component="parameter_tuner")
        ensure_positive(self._min_temperature, "min_temperature", component="parameter_tuner")
        ensure_positive(self._max_temperature, "max_temperature", component="parameter_tuner")

        ensure_positive(self.history_size, "history_size", component="parameter_tuner")
        ensure_positive(self.max_adaptation_history, "max_adaptation_history", component="parameter_tuner")
        ensure_positive(self.max_tuning_history, "max_tuning_history", component="parameter_tuner")
        ensure_positive(self.performance_baseline_window, "performance_baseline_window", component="parameter_tuner")
        ensure_positive(self.stability_window, "stability_window", component="parameter_tuner")
        ensure_positive(self.trend_window, "trend_window", component="parameter_tuner")

        ensure_in_range(self.stable_variance_threshold, "stable_variance_threshold", minimum=0.0, component="parameter_tuner")
        ensure_in_range(self.high_variance_threshold, "high_variance_threshold", minimum=0.0, component="parameter_tuner")
        ensure_positive(self.stable_lr_decay, "stable_lr_decay", component="parameter_tuner")
        ensure_positive(self.volatile_lr_growth, "volatile_lr_growth", component="parameter_tuner")
        ensure_in_range(self.lr_trend_bonus, "lr_trend_bonus", minimum=0.0, component="parameter_tuner")
        ensure_positive(self.negative_reward_exploration_boost, "negative_reward_exploration_boost", component="parameter_tuner")
        ensure_positive(self.positive_reward_exploration_decay, "positive_reward_exploration_decay", component="parameter_tuner")
        ensure_positive(self.exploration_decay, "exploration_decay", component="parameter_tuner")
        ensure_positive(self.discount_familiar_growth, "discount_familiar_growth", component="parameter_tuner")
        ensure_positive(self.discount_novelty_decay, "discount_novelty_decay", component="parameter_tuner")
        ensure_positive(self.temperature_growth, "temperature_growth", component="parameter_tuner")
        ensure_positive(self.temperature_decay, "temperature_decay", component="parameter_tuner")
        ensure_positive(self.state_visit_high_threshold, "state_visit_high_threshold", component="parameter_tuner")
        ensure_positive(self.state_visit_low_threshold, "state_visit_low_threshold", component="parameter_tuner")
        ensure_positive(self.schedule_initial_temperature, "schedule_initial_temperature", component="parameter_tuner")
        ensure_positive(self.schedule_final_temperature, "schedule_final_temperature", component="parameter_tuner")
        ensure_positive(self.schedule_decay_episodes, "schedule_decay_episodes", component="parameter_tuner")

        if self._min_learning_rate > self._max_learning_rate:
            raise InvalidConfigurationValueError("min_learning_rate cannot exceed max_learning_rate.", component="parameter_tuner")
        if self._min_exploration > self._max_exploration:
            raise InvalidConfigurationValueError("min_exploration cannot exceed max_exploration.", component="parameter_tuner")
        if self._min_discount_factor > self._max_discount_factor:
            raise InvalidConfigurationValueError("min_discount_factor cannot exceed max_discount_factor.", component="parameter_tuner")
        if self._min_temperature > self._max_temperature:
            raise InvalidConfigurationValueError("min_temperature cannot exceed max_temperature.", component="parameter_tuner")
        if self.state_visit_low_threshold > self.state_visit_high_threshold:
            raise InvalidConfigurationValueError("state_visit_low_threshold cannot exceed state_visit_high_threshold.", component="parameter_tuner")

    def _coerce_numeric_sequence(self, rewards: Sequence[Any], name: str = "recent_rewards") -> List[float]:
        ensure_non_empty(rewards, name, component="parameter_tuner")
        values: List[float] = []
        for idx, reward in enumerate(rewards):
            if not isinstance(reward, (int, float, np.number)) or not np.isfinite(reward):
                raise InvalidValueError(
                    f"All entries in '{name}' must be finite numeric values.",
                    component="parameter_tuner",
                    details={"index": idx, "value": reward},
                )
            values.append(float(reward))
        return values

    def _coerce_param_value(self, name: str, value: Any) -> float:
        if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
            raise InvalidValueError(
                f"Parameter '{name}' must be a finite numeric value.",
                component="parameter_tuner",
                details={"name": name, "value": value},
            )
        return float(value)

    def _normalize_candidate_params(self, params: Mapping[str, Any], allow_partial: bool = True) -> Dict[str, float]:
        ensure_instance(params, Mapping, "params", component="parameter_tuner")
        normalized: Dict[str, float] = {}
        for key, value in params.items():
            if key not in self.TRACKED_PARAMS:
                continue
            normalized[key] = self._coerce_param_value(key, value)
        if not allow_partial:
            missing = [name for name in self.TRACKED_PARAMS if name not in normalized]
            if missing:
                raise MissingFieldError(
                    "Parameter set is missing required tracked parameters.",
                    component="parameter_tuner",
                    details={"missing": missing},
                )
        return normalized

    def _apply_bounds(self) -> None:
        printer.status("INIT", "Bounds successfully initialized", "info")
        self.params["learning_rate"] = float(np.clip(self.params["learning_rate"], self._min_learning_rate, self._max_learning_rate))
        self.params["exploration_rate"] = float(np.clip(self.params["exploration_rate"], self._min_exploration, self._max_exploration))
        self.params["discount_factor"] = float(np.clip(self.params["discount_factor"], self._min_discount_factor, self._max_discount_factor))
        self.params["temperature"] = float(np.clip(self.params["temperature"], self._min_temperature, self._max_temperature))

    def _get_recent_reward_stats(self, recent_rewards: Sequence[float]) -> Dict[str, float]:
        rewards = np.asarray(recent_rewards, dtype=np.float64)
        mean_reward = float(np.mean(rewards))
        reward_var = float(np.var(rewards))
        reward_std = float(np.std(rewards))
        reward_min = float(np.min(rewards))
        reward_max = float(np.max(rewards))
        if rewards.size == 1:
            trend = 0.0
        else:
            window = min(self.trend_window, rewards.size)
            earlier = float(np.mean(rewards[:window]))
            later = float(np.mean(rewards[-window:]))
            trend = later - earlier
        return {
            "mean_reward": mean_reward,
            "reward_variance": reward_var,
            "reward_std": reward_std,
            "reward_min": reward_min,
            "reward_max": reward_max,
            "trend": float(trend),
        }

    def _get_performance_baseline(self, window: Optional[int] = None) -> float:
        if not self.performance_history:
            return 0.0
        k = self.performance_baseline_window if window is None else int(window)
        ensure_positive(k, "window", component="parameter_tuner")
        values = list(self.performance_history)[-k:]
        return float(np.mean(values)) if values else 0.0

    def _record_performance_samples(self, rewards: Iterable[float]) -> None:
        for reward in rewards:
            self.performance_history.append(float(reward))

    def _record_parameter_state(self, performance: float) -> None:
        if self.log_every_update:
            self.memory.log_parameters(performance=performance, params=self.params.copy())

    def adapt(self, recent_rewards: Sequence[Any]) -> Dict[str, Any]:
        printer.status("INIT", "Adapter successfully initialized", "info")
        rewards = self._coerce_numeric_sequence(recent_rewards, "recent_rewards")
        stats = self._get_recent_reward_stats(rewards)
        params_before = self.params.copy()

        if stats["reward_variance"] < self.stable_variance_threshold:
            self.params["learning_rate"] *= self.stable_lr_decay
        elif stats["reward_variance"] > self.high_variance_threshold:
            self.params["learning_rate"] *= self.volatile_lr_growth

        if stats["trend"] > 0:
            self.params["learning_rate"] *= (1.0 + self.lr_trend_bonus)
        elif stats["trend"] < 0:
            self.params["learning_rate"] *= max(1e-6, 1.0 - self.lr_trend_bonus)

        if stats["mean_reward"] < self.negative_reward_threshold:
            self.params["exploration_rate"] *= self.negative_reward_exploration_boost
        else:
            self.params["exploration_rate"] *= self.positive_reward_exploration_decay

        if stats["trend"] > 0 and stats["reward_variance"] < self.high_variance_threshold:
            self.params["discount_factor"] *= self.discount_familiar_growth
        elif stats["trend"] < 0:
            self.params["discount_factor"] *= self.discount_novelty_decay

        if stats["reward_variance"] > self.high_variance_threshold or stats["mean_reward"] < self.negative_reward_threshold:
            self.params["temperature"] *= self.temperature_growth
        else:
            self.params["temperature"] *= self.temperature_decay

        self._apply_bounds()
        self._record_performance_samples(rewards)
        self._record_parameter_state(performance=stats["mean_reward"])
        self._step_counter += 1

        report = {
            "type": "adaptive_update",
            "timestamp": self._step_counter,
            "stats": stats,
            "params_before": params_before,
            "params_after": self.params.copy(),
            "deltas": {key: float(self.params[key] - params_before[key]) for key in self.TRACKED_PARAMS},
        }
        self.adaptation_history.append(report)
        self._last_adaptation = report
        return report

    def apply_params(
        self,
        new_params: Mapping[str, Any],
        *,
        record_memory: bool = True,
        performance: Optional[float] = None,
        source: str = "manual",
    ) -> Dict[str, Any]:
        normalized = self._normalize_candidate_params(new_params, allow_partial=True)
        params_before = self.params.copy()
        self.params.update(normalized)
        self._apply_bounds()
        if record_memory:
            perf = self._get_performance_baseline() if performance is None else float(performance)
            self.memory.log_parameters(performance=perf, params=self.params.copy())
        return {
            "source": source,
            "params_before": params_before,
            "params_after": self.params.copy(),
            "updated_keys": sorted(normalized.keys()),
        }

    def set_params(self, **kwargs: Any) -> Dict[str, Any]:
        return self.apply_params(kwargs, record_memory=False, source="set_params")

    def estimate_performance_delta(
        self,
        candidate_params: Mapping[str, Any],
        baseline_params: Optional[Mapping[str, Any]] = None,
    ) -> float:
        candidate = {**self.params, **self._normalize_candidate_params(candidate_params, allow_partial=True)}
        baseline = self.params if baseline_params is None else {**self.params, **self._normalize_candidate_params(baseline_params, allow_partial=True)}

        if not self.estimate_delta_from_memory:
            return 0.0

        try:
            impacts = self.memory.analyze_parameter_impact()
        except Exception:
            impacts = {}

        if not impacts:
            return 0.0

        impact_map = {
            "learning_rate": float(impacts.get("learning_rate_impact", 0.0)),
            "exploration_rate": float(impacts.get("exploration_impact", 0.0)),
            "discount_factor": float(impacts.get("discount_impact", 0.0)),
            "temperature": float(impacts.get("temperature_impact", 0.0)),
        }

        delta = 0.0
        for key in self.TRACKED_PARAMS:
            delta += (float(candidate[key]) - float(baseline[key])) * impact_map[key]
        return float(delta)

    def run_hyperparameter_tuning(
        self,
        evaluation_function: Callable[..., Any],
        X_data: Optional[Any] = None,
        y_data: Optional[Any] = None,
        model_type: Optional[str] = None,
        apply_best: Optional[bool] = None,
    ) -> Dict[str, Any]:
        printer.status("INIT", "Tuning successfully initialized", "info")

        if not callable(evaluation_function):
            raise InvalidTypeError(
                "evaluation_function must be callable.",
                component="parameter_tuner",
                details={"received_type": type(evaluation_function).__name__},
            )

        apply_best_flag = self.default_apply_best_params if apply_best is None else bool(apply_best)
        model_type = model_type or self.tuning_model_type

        params_before = self.params.copy()
        baseline_performance = self._get_performance_baseline()

        try:
            tuner = self.tuner_cls(model_type=model_type, evaluation_function=evaluation_function)
        except AdaptiveError:
            raise
        except Exception as exc:
            raise HyperparameterOptimizationError(
                "Failed to initialize HyperparamTuner.",
                component="parameter_tuner",
                details={"model_type": model_type},
                remediation="Inspect tuning configuration and the injected tuner class.",
                cause=exc,
            ) from exc

        if not hasattr(tuner, "run_tuning_pipeline"):
            raise InvalidLifecycleStateError(
                "Injected tuner does not implement run_tuning_pipeline().",
                component="parameter_tuner",
                details={"tuner_type": type(tuner).__name__},
            )

        try:
            best_params = tuner.run_tuning_pipeline(X_data=X_data, y_data=y_data)
        except AdaptiveError:
            raise
        except Exception as exc:
            raise HyperparameterOptimizationError(
                "Hyperparameter tuning pipeline failed.",
                component="parameter_tuner",
                remediation="Inspect evaluation_function and the configured search strategy.",
                cause=exc,
            ) from exc

        ensure_instance(best_params, Mapping, "best_params", component="parameter_tuner")
        normalized_best = self._normalize_candidate_params(best_params, allow_partial=True)
        if not normalized_best:
            raise HyperparameterOptimizationError(
                "Tuning pipeline did not return any tracked parameter values.",
                component="parameter_tuner",
                details={"returned_keys": list(best_params.keys())},
            )

        estimated_delta = self.estimate_performance_delta(normalized_best, baseline_params=params_before)
        params_after_candidate = {**self.params, **normalized_best}
        params_after_candidate = self._bounded_snapshot(params_after_candidate)

        if apply_best_flag:
            self.params.update(normalized_best)
            self._apply_bounds()
            performance_for_logging = baseline_performance + estimated_delta
            self.memory.log_parameters(performance=performance_for_logging, params=self.params.copy())
        else:
            performance_for_logging = baseline_performance

        intervention = {
            "type": "hyperparameter_tuning",
            "params_before": params_before,
            "params_after": self.params.copy() if apply_best_flag else params_after_candidate,
            "strategy": getattr(tuner, "strategy", None),
        }
        effect = {
            "performance_delta": float(estimated_delta),
            "baseline_performance": float(baseline_performance),
            "estimated_performance": float(baseline_performance + estimated_delta),
        }
        self.memory.apply_policy_intervention(intervention, effect)

        report = {
            "best_params": params_after_candidate,
            "applied": apply_best_flag,
            "params_before": params_before,
            "params_after": self.params.copy() if apply_best_flag else params_after_candidate,
            "estimated_performance_delta": float(estimated_delta),
            "strategy": getattr(tuner, "strategy", None),
            "model_type": model_type,
        }
        self.tuning_history.append(report)
        self._last_tuning_result = report
        return report

    def _bounded_snapshot(self, params: Mapping[str, Any]) -> Dict[str, float]:
        snapshot = {**self.params, **self._normalize_candidate_params(params, allow_partial=True)}
        snapshot["learning_rate"] = float(np.clip(snapshot["learning_rate"], self._min_learning_rate, self._max_learning_rate))
        snapshot["exploration_rate"] = float(np.clip(snapshot["exploration_rate"], self._min_exploration, self._max_exploration))
        snapshot["discount_factor"] = float(np.clip(snapshot["discount_factor"], self._min_discount_factor, self._max_discount_factor))
        snapshot["temperature"] = float(np.clip(snapshot["temperature"], self._min_temperature, self._max_temperature))
        return snapshot

    def decay_exploration(self, decay_factor: Optional[float] = None) -> float:
        printer.status("INIT", "Decay explorer successfully initialized", "info")
        base_decay = self.exploration_decay if decay_factor is None else float(decay_factor)
        ensure_positive(base_decay, "decay_factor", component="parameter_tuner")

        if len(self.performance_history) >= self.stability_window:
            recent = np.array(list(self.performance_history)[-self.stability_window :], dtype=np.float64)
            recent_avg = float(np.mean(recent))
            global_best = float(np.max(self.performance_history)) if self.performance_history else recent_avg
            if global_best > 0 and recent_avg > 0.7 * global_best:
                base_decay = base_decay ** 2

        self.params["exploration_rate"] = max(float(self.params["exploration_rate"]) * base_decay, self._min_exploration)
        self._apply_bounds()
        return float(self.params["exploration_rate"])

    def update_performance(self, reward: Any) -> float:
        if not isinstance(reward, (int, float, np.number)) or not np.isfinite(reward):
            raise InvalidValueError(
                "reward must be a finite numeric value.",
                component="parameter_tuner",
                details={"reward": reward},
            )

        reward_value = float(reward)
        self.performance_history.append(reward_value)
        self.memory.log_parameters(performance=reward_value, params=self.params.copy())
        printer.status("UPDATE", f"Recorded performance: {reward_value:.3f}", "info")
        return reward_value

    def get_params(self, include_metadata: bool = False) -> Dict[str, Any]:
        params_snapshot: Dict[str, Any] = self.params.copy()
        if include_metadata:
            params_snapshot["_metadata"] = {
                "bounds": self.get_bounds(),
                "history": {
                    "performance_history_length": len(self.performance_history),
                    "adaptation_history_length": len(self.adaptation_history),
                    "tuning_history_length": len(self.tuning_history),
                    "last_performance": self.performance_history[-1] if self.performance_history else None,
                    "average_performance": float(np.mean(self.performance_history)) if self.performance_history else None,
                    "baseline_performance": self._get_performance_baseline(),
                },
                "steps": {
                    "episode_counter": self._episode_counter,
                    "step_counter": self._step_counter,
                },
            }
        return params_snapshot

    def get_bounds(self) -> Dict[str, Dict[str, float]]:
        return {
            "learning_rate": {"min": self._min_learning_rate, "max": self._max_learning_rate},
            "exploration_rate": {"min": self._min_exploration, "max": self._max_exploration},
            "discount_factor": {"min": self._min_discount_factor, "max": self._max_discount_factor},
            "temperature": {"min": self._min_temperature, "max": self._max_temperature},
        }

    def adaptive_discount_factor(self, state_visits: int) -> float:
        ensure_positive(state_visits + 1, "state_visits_plus_one", component="parameter_tuner")
        base_gamma = self.params["discount_factor"]
        if state_visits > self.state_visit_high_threshold:
            return float(min(base_gamma * self.discount_familiar_growth, self._max_discount_factor))
        if state_visits < self.state_visit_low_threshold:
            return float(max(base_gamma * self.discount_novelty_decay, self._min_discount_factor))
        return float(base_gamma)

    def temperature_schedule(self, episode: int) -> float:
        if not isinstance(episode, int) or episode < 0:
            raise InvalidValueError(
                "episode must be a non-negative integer.",
                component="parameter_tuner",
                details={"episode": episode},
            )

        fraction = min(1.0, episode / max(1, self.schedule_decay_episodes))
        temperature = max(
            self.schedule_final_temperature,
            self.schedule_initial_temperature * (1.0 - fraction),
        )
        return float(np.clip(temperature, self._min_temperature, self._max_temperature))

    def apply_to_worker(self, worker: Any, update_optimizers: bool = True) -> Dict[str, Any]:
        ensure_not_none(worker, "worker", component="parameter_tuner")
        summary = {"worker_type": type(worker).__name__, "updated_attributes": [], "updated_optimizers": []}

        attribute_map = {
            "learning_rate": "learning_rate",
            "exploration_rate": "exploration_rate",
            "discount_factor": ("gamma", "discount_factor"),
            "temperature": "temperature",
        }

        for param_name, attr_names in attribute_map.items():
            if isinstance(attr_names, tuple):
                for attr_name in attr_names:
                    if hasattr(worker, attr_name):
                        setattr(worker, attr_name, self.params[param_name])
                        summary["updated_attributes"].append(attr_name)
            else:
                if hasattr(worker, attr_names):
                    setattr(worker, attr_names, self.params[param_name])
                    summary["updated_attributes"].append(attr_names)

        if update_optimizers:
            for optimizer_name in ("optimizer", "actor_optimizer", "critic_optimizer"):
                optimizer = getattr(worker, optimizer_name, None)
                if optimizer is not None and hasattr(optimizer, "param_groups"):
                    for group in optimizer.param_groups:
                        group["lr"] = self.params["learning_rate"]
                    summary["updated_optimizers"].append(optimizer_name)
        return summary

    def apply_to_workers(self, workers: Iterable[Any], update_optimizers: bool = True) -> List[Dict[str, Any]]:
        ensure_not_none(workers, "workers", component="parameter_tuner")
        results = []
        for worker in workers:
            results.append(self.apply_to_worker(worker, update_optimizers=update_optimizers))
        return results

    def reset(self, params_to_reset: Optional[Sequence[str]] = None) -> Dict[str, float]:
        reset_targets = list(self.TRACKED_PARAMS if params_to_reset is None else params_to_reset)
        ensure_non_empty(reset_targets, "params_to_reset", component="parameter_tuner")

        for param in reset_targets:
            if param not in self.TRACKED_PARAMS:
                raise InvalidValueError(
                    f"Unknown parameter requested for reset: {param}",
                    component="parameter_tuner",
                    details={"supported": list(self.TRACKED_PARAMS)},
                )
            self.params[param] = self._get_default(param)

        self._apply_bounds()
        return self.params.copy()

    def _get_default(self, param_name: str) -> float:
        if param_name not in self.base_params:
            raise InvalidValueError(f"Unknown parameter default requested: {param_name}", component="parameter_tuner")
        return float(self.base_params[param_name])

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "params": self.params.copy(),
            "base_params": self.base_params.copy(),
            "bounds": self.get_bounds(),
            "performance_history_length": len(self.performance_history),
            "adaptation_history_length": len(self.adaptation_history),
            "tuning_history_length": len(self.tuning_history),
            "baseline_performance": self._get_performance_baseline(),
            "last_adaptation": self._last_adaptation,
            "last_tuning_result": self._last_tuning_result,
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "params": self.params.copy(),
            "base_params": self.base_params.copy(),
            "initial_params": self.initial_params.copy(),
            "performance_history": list(self.performance_history),
            "adaptation_history": list(self.adaptation_history),
            "tuning_history": list(self.tuning_history),
            "episode_counter": self._episode_counter,
            "step_counter": self._step_counter,
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        ensure_instance(state, Mapping, "state", component="parameter_tuner")
        params = state.get("params", {})
        self.params = {**self.base_params, **self._normalize_candidate_params(params, allow_partial=True)}
        self._apply_bounds()

        perf_history = state.get("performance_history", [])
        self.performance_history = deque(
            [float(v) for v in perf_history if isinstance(v, (int, float, np.number)) and np.isfinite(v)],
            maxlen=self.history_size,
        )

        self.adaptation_history = deque(list(state.get("adaptation_history", [])), maxlen=self.max_adaptation_history)
        self.tuning_history = deque(list(state.get("tuning_history", [])), maxlen=self.max_tuning_history)
        self._episode_counter = int(state.get("episode_counter", 0))
        self._step_counter = int(state.get("step_counter", 0))

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        checkpoint_path = Path(path)
        payload = {
            "state": self.export_state(),
            "config": {"parameter_tuner": self.tuner_config, "tuning": self.tuning_section},
        }

        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=self.checkpoint_protocol)
            logger.info("Parameter Tuner checkpoint saved to %s", checkpoint_path)
            return checkpoint_path
        except Exception as exc:
            raise CheckpointSaveError(
                f"Failed to save Parameter Tuner checkpoint to {checkpoint_path}.",
                component="parameter_tuner",
                details={"path": str(checkpoint_path)},
                cause=exc,
            ) from exc

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(
                f"Parameter Tuner checkpoint not found: {checkpoint_path}",
                component="parameter_tuner",
                details={"path": str(checkpoint_path)},
            )

        try:
            with checkpoint_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            raise CheckpointLoadError(
                f"Failed to load Parameter Tuner checkpoint from {checkpoint_path}.",
                component="parameter_tuner",
                details={"path": str(checkpoint_path)},
                cause=exc,
            ) from exc

        ensure_instance(payload, Mapping, "checkpoint_payload", component="parameter_tuner")
        self.import_state(payload.get("state", {}))
        logger.info("Parameter Tuner checkpoint loaded from %s", checkpoint_path)
        return self.export_state()


if __name__ == "__main__":
    print("\n=== Running Parameter Tuner ===\n")
    printer.status("TEST", "Parameter Tuner initialized", "info")

    class _TestHyperparamTuner:
        """Lightweight injected tuner used only for the script self-test."""

        def __init__(self, model_type: Optional[str] = None, evaluation_function: Optional[Callable] = None):
            self.model_type = model_type
            self.evaluation_function = evaluation_function
            self.strategy = "test_search"

        def run_tuning_pipeline(self, X_data: Optional[Any] = None, y_data: Optional[Any] = None) -> Dict[str, float]:
            candidates = [
                {"learning_rate": 0.008, "exploration_rate": 0.25, "discount_factor": 0.95, "temperature": 0.90},
                {"learning_rate": 0.012, "exploration_rate": 0.20, "discount_factor": 0.97, "temperature": 0.80},
                {"learning_rate": 0.006, "exploration_rate": 0.30, "discount_factor": 0.94, "temperature": 1.00},
            ]
            scored = [(float(self.evaluation_function(candidate)), candidate) for candidate in candidates]
            scored.sort(key=lambda item: item[0], reverse=True)
            return scored[0][1]

    tuner = LearningParameterTuner(tuner_cls=_TestHyperparamTuner)
    printer.pretty("BASE_PARAMS", tuner.get_params(include_metadata=True), "success")

    rewards = [0.2, 0.5, 0.3, 0.65, 0.7, 0.6, 0.8, 0.75]
    adapt_report = tuner.adapt(recent_rewards=rewards)
    printer.pretty("ADAPT_REPORT", adapt_report, "success")

    tuner.update_performance(0.82)
    tuner.update_performance(0.88)
    decayed_exploration = tuner.decay_exploration()
    printer.status("TEST", f"Exploration after decay: {decayed_exploration:.6f}", "success")

    scheduled_temp = tuner.temperature_schedule(episode=250)
    adaptive_gamma = tuner.adaptive_discount_factor(state_visits=120)
    printer.status("TEST", f"Scheduled temperature: {scheduled_temp:.6f}", "success")
    printer.status("TEST", f"Adaptive discount factor: {adaptive_gamma:.6f}", "success")

    def evaluation_fn(params: Dict[str, float]) -> float:
        target = {"learning_rate": 0.010, "exploration_rate": 0.18, "discount_factor": 0.97, "temperature": 0.85}
        score = 1.0
        for key, target_value in target.items():
            score -= abs(float(params[key]) - target_value)
        return float(score)

    tuning_report = tuner.run_hyperparameter_tuning(evaluation_function=evaluation_fn, apply_best=True)
    printer.pretty("TUNING_REPORT", tuning_report, "success")

    class _DummyOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": 0.001}]

    class _DummyWorker:
        def __init__(self):
            self.learning_rate = 0.001
            self.exploration_rate = 0.4
            self.gamma = 0.95
            self.temperature = 1.0
            self.actor_optimizer = _DummyOptimizer()
            self.critic_optimizer = _DummyOptimizer()

    worker = _DummyWorker()
    apply_report = tuner.apply_to_worker(worker)
    printer.pretty("APPLY_REPORT", apply_report, "success")

    checkpoint_path = tuner.save_checkpoint("parameter_tuner_test.pkl")
    restored = LearningParameterTuner(tuner_cls=_TestHyperparamTuner)
    restored.load_checkpoint(checkpoint_path)
    printer.pretty("RESTORED_PARAMS", restored.get_params(include_metadata=True), "success")
    printer.pretty("DIAGNOSTICS", restored.get_diagnostics(), "success")

    restored.reset(params_to_reset=["learning_rate", "exploration_rate"])
    printer.pretty("RESET_PARAMS", restored.get_params(include_metadata=False), "success")

    print("\n=== Test ran successfully ===\n")
