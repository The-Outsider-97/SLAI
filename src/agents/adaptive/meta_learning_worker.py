
from __future__ import annotations

import pickle
import random
import numpy as np
import torch

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.adaptive_errors import *
from .adaptive_memory import MultiModalMemory
from src.tuning.networks.bayesian_neural_network import BayesianNeuralNetwork
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Meta Learning Worker")
printer = PrettyPrinter


class MetaLearningWorker:
    """
    Meta-Learning Worker for hyperparameter optimization across skills
    - Uses Bayesian Neural Network for uncertainty-aware hyperparameter tuning
    - Optimizes hyperparameters using expected improvement acquisition
    - Maintains memory of hyperparameter-performance relationships
    - Coordinates with SkillWorkers for parameter updates

    Production-oriented extensions
    ------------------------------
    - Structured validation and exception handling integrated with `adaptive_errors`
    - Config-driven hyperparameter spaces and training behavior
    - Robust worker registry management and metric collection
    - Safe Bayesian optimization loop with expected improvement
    - Checkpoint/export helpers aligned with the current BayesianNeuralNetwork API
    - Improved diagnostics, history tracking, and integration reporting
    """

    DEFAULT_HYPERPARAMETER_SPACE = {
        "learning_rate": {"min": 1e-5, "max": 1e-2},
        "exploration_rate": {"min": 1e-2, "max": 0.3},
        "entropy_coef": {"min": 1e-3, "max": 0.1},
        "discount_factor": {"min": 0.9, "max": 0.999},
    }

    REQUIRED_WORKER_FIELDS = ("learning_rate", "entropy_coef")
    METRIC_KEYS = ("recent_reward", "avg_reward", "success_rate", "episode_count")

    def __init__(
        self,
        memory: Optional[MultiModalMemory] = None,
        skill_worker_registry: Optional[Mapping[Any, Any]] = None,
        bnn: Optional[BayesianNeuralNetwork] = None,
    ) -> None:
        super().__init__()
        self.config = load_global_config()
        self.meta_config = get_config_section("meta_learning")

        self._load_config()

        self.memory = memory if memory is not None else MultiModalMemory()
        if memory is not None:
            ensure_instance(memory, MultiModalMemory, "memory", component="meta_learning")

        self.skill_worker_registry: Dict[Any, Any] = dict(skill_worker_registry or {})
        self.performance_history: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []
        self.optimization_step_count = 0
        self.last_training_summary: Optional[Dict[str, Any]] = None
        self.last_suggestion: Optional[Dict[str, float]] = None

        self.rng = np.random.default_rng(self.random_state)
        if self.random_state is not None:
            random.seed(int(self.random_state))

        self.hyperparameter_space = self._load_hyperparameter_space()
        self.hyperparam_names = list(self.hyperparameter_space.keys())
        self.num_hyperparams = len(self.hyperparam_names)

        self.bnn = bnn if bnn is not None else self._build_bnn()
        if bnn is not None:
            ensure_instance(bnn, BayesianNeuralNetwork, "bnn", component="meta_learning")

        logger.info("Meta Learning Worker initialized")
        logger.info("Optimizing hyperparameters: %s", ", ".join(self.hyperparam_names))
        logger.info("Using worker registry: %s", self.worker_registry_name)

    # ------------------------------------------------------------------
    # Configuration and initialization
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        try:
            self.training_epochs = int(self.meta_config.get("training_epochs", 10))
            self.worker_registry_name = str(self.meta_config.get("skill_workers", "SkillWorkerRegistry"))
            self.hidden_layers = list(self.meta_config.get("hidden_layers", [64, 64]))
            self.exploration_factor = float(self.meta_config.get("exploration_factor", 0.1))
            self.update_frequency = int(self.meta_config.get("update_frequency", 100))
            self.batch_size = int(self.meta_config.get("batch_size", 32))
            self.num_candidates = int(self.meta_config.get("num_candidates", 100))
            self.prediction_samples = int(self.meta_config.get("prediction_samples", 100))
            self.min_history_for_model = int(self.meta_config.get("min_history_for_model", 5))
            self.max_history = int(self.meta_config.get("max_history", 1000))
            self.normalize_targets = bool(self.meta_config.get("normalize_targets", True))
            self.validation_fraction = float(self.meta_config.get("validation_fraction", 0.2))
            self.shuffle_training = bool(self.meta_config.get("shuffle_training", True))
            self.early_stopping_patience = self.meta_config.get("early_stopping_patience", 10)
            self.min_delta = float(self.meta_config.get("min_delta", 1e-4))
            self.random_state = self.meta_config.get("random_state", 42)
            self.checkpoint_protocol = int(self.meta_config.get("checkpoint_protocol", pickle.HIGHEST_PROTOCOL))
            self.apply_optimizer_updates = bool(self.meta_config.get("apply_optimizer_updates", True))
            self.store_registry_in_memory = bool(self.meta_config.get("store_registry_in_memory", False))
            self.candidate_strategy = str(self.meta_config.get("candidate_strategy", "random")).lower()
            self.acquisition = str(self.meta_config.get("acquisition", "expected_improvement")).lower()
            self.reward_aggregation = str(self.meta_config.get("reward_aggregation", "mean")).lower()
            self.fallback_to_random_when_unfitted = bool(
                self.meta_config.get("fallback_to_random_when_unfitted", True)
            )
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationValueError(
                "Failed to parse meta_learning configuration values.",
                component="meta_learning",
                details={"section": "meta_learning"},
                remediation="Ensure all meta_learning configuration values are valid scalars.",
                cause=exc,
            ) from exc

        ensure_positive(self.training_epochs, "training_epochs", component="meta_learning")
        ensure_positive(self.update_frequency, "update_frequency", component="meta_learning")
        ensure_positive(self.batch_size, "batch_size", component="meta_learning")
        ensure_positive(self.num_candidates, "num_candidates", component="meta_learning")
        ensure_positive(self.prediction_samples, "prediction_samples", component="meta_learning")
        ensure_positive(self.min_history_for_model, "min_history_for_model", component="meta_learning")
        ensure_positive(self.max_history, "max_history", component="meta_learning")
        ensure_positive(self.exploration_factor, "exploration_factor", allow_zero=True, component="meta_learning")
        ensure_in_range(self.validation_fraction, "validation_fraction", minimum=0.0, maximum=0.49, component="meta_learning")
        ensure_in_range(self.min_delta, "min_delta", minimum=0.0, component="meta_learning")

        if self.early_stopping_patience is not None:
            self.early_stopping_patience = int(self.early_stopping_patience)
            ensure_positive(self.early_stopping_patience, "early_stopping_patience", component="meta_learning")

        ensure_non_empty(self.hidden_layers, "hidden_layers", component="meta_learning")
        for idx, size in enumerate(self.hidden_layers):
            ensure_positive(int(size), f"hidden_layers[{idx}]", component="meta_learning")

        if self.candidate_strategy not in {"random", "best_perturbation", "hybrid"}:
            raise InvalidConfigurationValueError(
                f"Unsupported candidate_strategy: {self.candidate_strategy}",
                component="meta_learning",
                details={"candidate_strategy": self.candidate_strategy},
                remediation="Use random, best_perturbation, or hybrid.",
            )

        if self.acquisition not in {"expected_improvement"}:
            raise InvalidConfigurationValueError(
                f"Unsupported acquisition: {self.acquisition}",
                component="meta_learning",
                details={"acquisition": self.acquisition},
                remediation="Use expected_improvement.",
            )

        if self.reward_aggregation not in {"mean", "median"}:
            raise InvalidConfigurationValueError(
                f"Unsupported reward_aggregation: {self.reward_aggregation}",
                component="meta_learning",
                details={"reward_aggregation": self.reward_aggregation},
                remediation="Use mean or median.",
            )

        if self.random_state is not None and not isinstance(self.random_state, (int, np.integer)):
            raise InvalidConfigurationValueError(
                "random_state must be an integer or null.",
                component="meta_learning",
                details={"random_state": self.random_state},
            )

    def _load_hyperparameter_space(self) -> Dict[str, Dict[str, float]]:
        raw_space = self.meta_config.get("hyperparameter_space", self.DEFAULT_HYPERPARAMETER_SPACE)
        ensure_instance(raw_space, Mapping, "hyperparameter_space", component="meta_learning")

        parsed: Dict[str, Dict[str, float]] = {}
        for name, bounds in raw_space.items():
            ensure_instance(bounds, Mapping, f"hyperparameter_space.{name}", component="meta_learning")
            if "min" not in bounds or "max" not in bounds:
                raise MissingFieldError(
                    f"hyperparameter_space.{name} must include 'min' and 'max'.",
                    component="meta_learning",
                    details={"name": name, "bounds": dict(bounds)},
                )
            low = float(bounds["min"])
            high = float(bounds["max"])
            ensure_in_range(low, f"{name}.min", component="meta_learning")
            ensure_in_range(high, f"{name}.max", component="meta_learning")
            if low >= high:
                raise InvalidConfigurationValueError(
                    f"Invalid range for hyperparameter '{name}': min must be < max.",
                    component="meta_learning",
                    details={"name": name, "min": low, "max": high},
                )
            parsed[str(name)] = {"min": low, "max": high}

        ensure_non_empty(parsed, "parsed_hyperparameter_space", component="meta_learning")
        return parsed

    def _build_bnn(self) -> BayesianNeuralNetwork:
        layer_sizes = [self.num_hyperparams] + [int(v) for v in self.hidden_layers] + [1]
        return BayesianNeuralNetwork(layer_sizes=layer_sizes)

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------
    def register_skill_worker(self, worker_id: Any, worker: Any) -> None:
        ensure_not_none(worker_id, "worker_id", component="meta_learning")
        ensure_not_none(worker, "worker", component="meta_learning")
        self.skill_worker_registry[worker_id] = worker
        if self.store_registry_in_memory:
            self._store_registry_snapshot()

    def register_skill_workers(self, workers: Mapping[Any, Any]) -> None:
        ensure_instance(workers, Mapping, "workers", component="meta_learning")
        for worker_id, worker in workers.items():
            self.register_skill_worker(worker_id, worker)

    def unregister_skill_worker(self, worker_id: Any) -> None:
        self.skill_worker_registry.pop(worker_id, None)
        if self.store_registry_in_memory:
            self._store_registry_snapshot()

    def clear_worker_registry(self) -> None:
        self.skill_worker_registry.clear()
        if self.store_registry_in_memory:
            self._store_registry_snapshot()

    def _store_registry_snapshot(self) -> None:
        try:
            snapshot = {
                str(worker_id): {"type": type(worker).__name__, "has_metrics": hasattr(worker, "get_performance_metrics")}
                for worker_id, worker in self.skill_worker_registry.items()
            }
            self.memory.semantic[f"registry_{self.worker_registry_name}"] = {
                "strength": 1.0,
                "last_accessed": datetime.now(),
                "data": snapshot,
                "context_hash": self.worker_registry_name,
                "count": len(snapshot),
            }
        except Exception as exc:
            logger.warning("Failed to store registry snapshot in memory: %s", exc)

    def get_worker_registry(self) -> Dict[Any, Any]:
        """Retrieve worker registry from local registry first, then a best-effort memory fallback."""
        if self.skill_worker_registry:
            return self.skill_worker_registry

        fallback_key = f"registry_{self.worker_registry_name}"
        if fallback_key in getattr(self.memory, "semantic", {}):
            registry_data = self.memory.semantic[fallback_key].get("data")
            if isinstance(registry_data, dict):
                logger.warning(
                    "Registry snapshot found in memory, but only metadata is available; runtime workers must be re-registered."
                )
                return {}

        legacy_key = f"ctx_{self.worker_registry_name[:6]}"
        if legacy_key in getattr(self.memory, "semantic", {}):
            registry_data = self.memory.semantic[legacy_key].get("data")
            if isinstance(registry_data, dict):
                return registry_data

        return {}

    # ------------------------------------------------------------------
    # Metrics and experience tracking
    # ------------------------------------------------------------------
    def collect_performance_metrics(self) -> Dict[Any, Dict[str, Any]]:
        """Collect performance metrics from all registered skill workers."""
        registry = self.get_worker_registry()
        if not registry:
            return {}

        metrics: Dict[Any, Dict[str, Any]] = {}
        for worker_id, worker in registry.items():
            if not hasattr(worker, "get_performance_metrics"):
                logger.warning("Worker %s missing performance metrics method", worker_id)
                continue

            try:
                worker_metrics = worker.get_performance_metrics()
            except Exception as exc:
                logger.warning("Error collecting metrics from worker %s: %s", worker_id, exc)
                continue

            if not isinstance(worker_metrics, Mapping):
                logger.warning("Worker %s returned non-mapping metrics: %s", worker_id, type(worker_metrics).__name__)
                continue

            metrics[worker_id] = dict(worker_metrics)

        return metrics

    def store_hyperparameter_experience(self, hyperparams: Mapping[str, Any], performance: float) -> Dict[str, Any]:
        """Store hyperparameter configuration and its performance."""
        normalized_hyperparams = self._normalize_hyperparams(hyperparams)

        if not isinstance(performance, (int, float, np.number)) or not np.isfinite(performance):
            raise InvalidValueError(
                "performance must be a finite numeric value.",
                component="meta_learning",
                details={"performance": performance},
            )

        context = {
            "type": "hyperparameter_config",
            "params": normalized_hyperparams,
        }

        self.memory.store_experience(
            state=None,
            action=None,
            reward=float(performance),
            context=context,
            params=normalized_hyperparams,
        )

        experience = {
            "hyperparams": normalized_hyperparams,
            "performance": float(performance),
            "timestamp": datetime.now(),
        }
        self.performance_history.append(experience)
        if len(self.performance_history) > self.max_history:
            self.performance_history = self.performance_history[-self.max_history :]
        return experience

    # ------------------------------------------------------------------
    # Hyperparameter suggestion and optimization
    # ------------------------------------------------------------------
    def suggest_hyperparameters(self) -> Dict[str, float]:
        """
        Suggest new hyperparameters using Bayesian optimization with
        Expected Improvement acquisition.
        """
        if len(self.performance_history) < self.min_history_for_model:
            suggestion = self._random_hyperparameters()
            self.last_suggestion = suggestion
            return suggestion

        X, y, normalization = self._prepare_training_data()
        self._train_bnn(X, y)

        candidates = self._generate_candidates()
        candidate_array = np.vstack([self._hyperparams_to_array(c) for c in candidates])

        means, stds = self.bnn.predict(candidate_array, num_samples=self.prediction_samples)
        means = np.asarray(means, dtype=np.float64).reshape(-1)
        stds = np.asarray(stds, dtype=np.float64).reshape(-1)

        y_reference = self._denormalize_targets(y, normalization) if self.normalize_targets else y
        means_reference = self._denormalize_targets(means, normalization) if self.normalize_targets else means
        stds_reference = stds * normalization["std"] if self.normalize_targets else stds

        acquisition_scores = self._expected_improvement(
            means_reference,
            stds_reference,
            best_observed=float(np.max(y_reference)),
        )

        best_idx = int(np.argmax(acquisition_scores))
        suggestion = candidates[best_idx]
        self.last_suggestion = suggestion
        return suggestion

    def _expected_improvement(self, means: np.ndarray, stds: np.ndarray, best_observed: float) -> np.ndarray:
        stds = np.maximum(np.asarray(stds, dtype=np.float64), 1e-12)
        means = np.asarray(means, dtype=np.float64)
        improvement = means - best_observed - self.exploration_factor

        from scipy.stats import norm  # aligned with the original module pattern

        z = improvement / stds
        ei = improvement * norm.cdf(z) + stds * norm.pdf(z)
        ei[stds <= 1e-12] = 0.0
        if not np.all(np.isfinite(ei)):
            ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
        return ei

    def update_skill_hyperparameters(self, hyperparams: Mapping[str, Any]) -> Dict[Any, Dict[str, Any]]:
        """Update hyperparameters for all registered skill workers."""
        registry = self.get_worker_registry()
        if not registry:
            raise EmptyRegistryError(
                "No registered skill workers found.",
                component="meta_learning",
                remediation="Register one or more skill workers before applying hyperparameter updates.",
            )

        normalized_hyperparams = self._normalize_hyperparams(hyperparams)
        updates: Dict[Any, Dict[str, Any]] = {}

        for worker_id, worker in registry.items():
            applied = {"updated_attributes": [], "updated_optimizers": []}

            if hasattr(worker, "learning_rate"):
                worker.learning_rate = normalized_hyperparams["learning_rate"]
                applied["updated_attributes"].append("learning_rate")
            if hasattr(worker, "exploration_rate"):
                worker.exploration_rate = normalized_hyperparams["exploration_rate"]
                applied["updated_attributes"].append("exploration_rate")
            if hasattr(worker, "entropy_coef"):
                worker.entropy_coef = normalized_hyperparams["entropy_coef"]
                applied["updated_attributes"].append("entropy_coef")
            if hasattr(worker, "gamma"):
                worker.gamma = normalized_hyperparams["discount_factor"]
                applied["updated_attributes"].append("gamma")
            elif hasattr(worker, "discount_factor"):
                worker.discount_factor = normalized_hyperparams["discount_factor"]
                applied["updated_attributes"].append("discount_factor")

            if self.apply_optimizer_updates:
                for optimizer_name in ("optimizer", "actor_optimizer", "critic_optimizer"):
                    optimizer = getattr(worker, optimizer_name, None)
                    if optimizer is not None and hasattr(optimizer, "param_groups"):
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = normalized_hyperparams["learning_rate"]
                        applied["updated_optimizers"].append(optimizer_name)

            updates[worker_id] = applied

        logger.info("Updated skill hyperparameters")
        return updates

    def optimization_step(self) -> Dict[str, Any]:
        """Perform one meta-optimization cycle."""
        metrics = self.collect_performance_metrics()
        if not metrics:
            raise WorkerMetricsError(
                "No metrics available for optimization step.",
                component="meta_learning",
                remediation="Ensure registered workers expose get_performance_metrics() and have accumulated experience.",
            )

        valid_rewards = []
        for metric in metrics.values():
            value = metric.get("recent_reward")
            if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                valid_rewards.append(float(value))

        if not valid_rewards:
            raise WorkerMetricsError(
                "Metrics missing valid 'recent_reward' values.",
                component="meta_learning",
                details={"available_metric_keys": sorted({k for metric in metrics.values() for k in metric.keys()})},
            )

        avg_performance = float(np.mean(valid_rewards)) if self.reward_aggregation == "mean" else float(np.median(valid_rewards))

        registry = self.get_worker_registry()
        if not registry:
            raise EmptyRegistryError(
                "No registered skill workers found for optimization step.",
                component="meta_learning",
            )

        representative_worker = next(iter(registry.values()))
        current_hyperparams = self._extract_worker_hyperparams(representative_worker)

        stored_experience = self.store_hyperparameter_experience(current_hyperparams, avg_performance)
        suggested_hyperparams = self.suggest_hyperparameters()
        update_summary = self.update_skill_hyperparameters(suggested_hyperparams)

        intervention = {
            "type": "meta_learning_optimization",
            "params_before": current_hyperparams,
            "params_after": suggested_hyperparams,
        }
        effect = {
            "performance_delta": self._estimate_performance_delta(
                previous_params=current_hyperparams,
                new_params=suggested_hyperparams,
                reference_performance=avg_performance,
            )
        }
        self.memory.apply_policy_intervention(intervention, effect)

        self.optimization_step_count += 1
        result = {
            "step": self.optimization_step_count,
            "avg_performance": avg_performance,
            "stored_experience": stored_experience,
            "suggested_hyperparams": suggested_hyperparams,
            "update_summary": update_summary,
        }
        self.optimization_history.append(result)
        if len(self.optimization_history) > self.max_history:
            self.optimization_history = self.optimization_history[-self.max_history :]
        logger.info("Meta-optimization step completed | Performance: %.4f", avg_performance)
        return result

    def _estimate_performance_delta(
        self,
        previous_params: Mapping[str, float],
        new_params: Mapping[str, float],
        reference_performance: float,
    ) -> float:
        impacts = self.memory.analyze_parameter_impact() if hasattr(self.memory, "analyze_parameter_impact") else {}
        if not impacts:
            return 0.0

        impact_map = {
            "learning_rate": float(impacts.get("learning_rate_impact", 0.0)),
            "exploration_rate": float(impacts.get("exploration_impact", 0.0)),
            "discount_factor": float(impacts.get("discount_impact", 0.0)),
            "temperature": float(impacts.get("temperature_impact", 0.0)),
        }

        delta = 0.0
        for name in ("learning_rate", "exploration_rate", "discount_factor"):
            delta += (float(new_params.get(name, 0.0)) - float(previous_params.get(name, 0.0))) * impact_map.get(name, 0.0)

        return float(delta)

    # ------------------------------------------------------------------
    # Data preparation and BNN training
    # ------------------------------------------------------------------
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        X: List[np.ndarray] = []
        y: List[float] = []

        for experience in self.performance_history:
            hyperparams = experience["hyperparams"]
            X.append(self._hyperparams_to_array(hyperparams))
            y.append(float(experience["performance"]))

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        if X_arr.ndim != 2 or X_arr.shape[1] != self.num_hyperparams:
            raise HyperparameterOptimizationError(
                "Prepared training features have invalid shape.",
                component="meta_learning",
                details={"shape": tuple(X_arr.shape)},
            )

        normalization = {"mean": 0.0, "std": 1.0}
        if self.normalize_targets:
            normalization["mean"] = float(np.mean(y_arr))
            normalization["std"] = float(np.std(y_arr) + 1e-8)
            y_arr = (y_arr - normalization["mean"]) / normalization["std"]

        return X_arr, y_arr, normalization

    def _train_bnn(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        validation_data = None
        if len(X) >= 5 and self.validation_fraction > 0.0:
            val_count = max(1, int(round(len(X) * self.validation_fraction)))
            if val_count < len(X):
                indices = np.arange(len(X))
                if self.shuffle_training:
                    self.rng.shuffle(indices)
                val_idx = indices[:val_count]
                train_idx = indices[val_count:]
                validation_data = (X[val_idx], y[val_idx].reshape(-1, 1))
                X_train = X[train_idx]
                y_train = y[train_idx]
            else:
                X_train = X
                y_train = y
        else:
            X_train = X
            y_train = y

        history = self.bnn.fit(
            X_train,
            y_train.reshape(-1, 1),
            epochs=self.training_epochs,
            batch_size=min(self.batch_size, max(1, len(X_train))),
            num_samples=5,
            shuffle=self.shuffle_training,
            validation_data=validation_data,
            early_stopping_patience=self.early_stopping_patience,
            min_delta=self.min_delta,
            verbose=False,
        )
        self.last_training_summary = history
        return history

    # ------------------------------------------------------------------
    # Hyperparameter conversions and candidate generation
    # ------------------------------------------------------------------
    def _normalize_hyperparams(self, hyperparams: Mapping[str, Any]) -> Dict[str, float]:
        ensure_instance(hyperparams, Mapping, "hyperparams", component="meta_learning")
        normalized: Dict[str, float] = {}

        for name, bounds in self.hyperparameter_space.items():
            if name not in hyperparams:
                raise MissingFieldError(
                    f"Hyperparameters are missing required field '{name}'.",
                    component="meta_learning",
                    details={"required": self.hyperparam_names},
                )
            value = hyperparams[name]
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
                raise InvalidValueError(
                    f"Hyperparameter '{name}' must be a finite numeric value.",
                    component="meta_learning",
                    details={"name": name, "value": value},
                )

            value = float(value)
            value = float(np.clip(value, bounds["min"], bounds["max"]))
            normalized[name] = value

        return normalized

    def _extract_worker_hyperparams(self, worker: Any) -> Dict[str, float]:
        learning_rate = getattr(worker, "learning_rate", None)
        exploration_rate = getattr(worker, "exploration_rate", None)
        entropy_coef = getattr(worker, "entropy_coef", None)
        discount_factor = getattr(worker, "gamma", getattr(worker, "discount_factor", None))

        candidate = {
            "learning_rate": learning_rate,
            "exploration_rate": exploration_rate if exploration_rate is not None else self.hyperparameter_space["exploration_rate"]["min"],
            "entropy_coef": entropy_coef,
            "discount_factor": discount_factor,
        }
        return self._normalize_hyperparams(candidate)

    def _hyperparams_to_array(self, hyperparams: Mapping[str, Any]) -> np.ndarray:
        normalized_hyperparams = self._normalize_hyperparams(hyperparams)
        array = []
        for param in self.hyperparam_names:
            low = self.hyperparameter_space[param]["min"]
            high = self.hyperparameter_space[param]["max"]
            value = normalized_hyperparams[param]
            denom = max(high - low, 1e-12)
            array.append((value - low) / denom)
        return np.asarray(array, dtype=np.float64)

    def _array_to_hyperparams(self, array: Sequence[float]) -> Dict[str, float]:
        values = np.asarray(array, dtype=np.float64).reshape(-1)
        ensure_dimension(len(values), self.num_hyperparams, name="hyperparameter_array", component="meta_learning")
        hyperparams: Dict[str, float] = {}
        for i, param in enumerate(self.hyperparam_names):
            low = self.hyperparameter_space[param]["min"]
            high = self.hyperparameter_space[param]["max"]
            value = float(values[i] * (high - low) + low)
            hyperparams[param] = float(np.clip(value, low, high))
        return hyperparams

    def _random_hyperparameters(self) -> Dict[str, float]:
        return {
            param: float(self.rng.uniform(bounds["min"], bounds["max"]))
            for param, bounds in self.hyperparameter_space.items()
        }

    def _best_historical_hyperparams(self) -> Optional[Dict[str, float]]:
        if not self.performance_history:
            return None
        best_idx = int(np.argmax([float(e["performance"]) for e in self.performance_history]))
        return dict(self.performance_history[best_idx]["hyperparams"])

    def _perturb_hyperparameters(self, center: Mapping[str, Any], scale: float = 0.10) -> Dict[str, float]:
        center_norm = self._normalize_hyperparams(center)
        candidate: Dict[str, float] = {}
        for name, bounds in self.hyperparameter_space.items():
            span = bounds["max"] - bounds["min"]
            noise = float(self.rng.normal(loc=0.0, scale=scale * span))
            candidate[name] = float(np.clip(center_norm[name] + noise, bounds["min"], bounds["max"]))
        return candidate

    def _generate_candidates(self) -> List[Dict[str, float]]:
        candidates: List[Dict[str, float]] = []

        if self.candidate_strategy in {"best_perturbation", "hybrid"}:
            best = self._best_historical_hyperparams()
            if best is not None:
                half = self.num_candidates if self.candidate_strategy == "best_perturbation" else max(1, self.num_candidates // 2)
                for _ in range(half):
                    candidates.append(self._perturb_hyperparameters(best, scale=0.10))

        remaining = self.num_candidates - len(candidates)
        for _ in range(max(0, remaining)):
            candidates.append(self._random_hyperparameters())

        if not candidates:
            candidates.append(self._random_hyperparameters())

        return candidates

    def _denormalize_targets(self, values: np.ndarray, normalization: Mapping[str, float]) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) * float(normalization["std"])) + float(normalization["mean"])

    # ------------------------------------------------------------------
    # Reporting and persistence
    # ------------------------------------------------------------------
    def get_optimization_report(self) -> Dict[str, Any]:
        if not self.performance_history:
            return {}

        performances = [float(entry["performance"]) for entry in self.performance_history]
        best_idx = int(np.argmax(performances))
        return {
            "total_configurations": len(self.performance_history),
            "best_performance": performances[best_idx],
            "best_hyperparams": self.performance_history[best_idx]["hyperparams"],
            "recent_performance": performances[-1] if performances else 0.0,
            "performance_history": performances,
            "optimization_steps": self.optimization_step_count,
            "last_suggestion": self.last_suggestion,
            "bnn_summary": self.bnn.summary() if hasattr(self.bnn, "summary") else {},
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "performance_history": self.performance_history,
            "optimization_history": self.optimization_history,
            "optimization_step_count": self.optimization_step_count,
            "worker_registry_name": self.worker_registry_name,
            "hyperparameter_space": self.hyperparameter_space,
            "last_training_summary": self.last_training_summary,
            "last_suggestion": self.last_suggestion,
            "bnn_state": self.bnn.to_serializable_dict(include_history=True),
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        ensure_instance(state, Mapping, "state", component="meta_learning")
        self.performance_history = list(state.get("performance_history", []))
        self.optimization_history = list(state.get("optimization_history", []))
        self.optimization_step_count = int(state.get("optimization_step_count", 0))
        self.worker_registry_name = str(state.get("worker_registry_name", self.worker_registry_name))
        self.last_training_summary = state.get("last_training_summary")
        self.last_suggestion = state.get("last_suggestion")

        bnn_state = state.get("bnn_state")
        if isinstance(bnn_state, Mapping):
            self.bnn = BayesianNeuralNetwork(
                layer_sizes=bnn_state["layer_sizes"],
                learning_rate=float(bnn_state.get("learning_rate", self.bnn.learning_rate)),
                prior_mu=float(bnn_state.get("prior_mu", self.bnn.prior_mu)),
                prior_logvar=float(bnn_state.get("prior_logvar", self.bnn.prior_logvar)),
                random_state=bnn_state.get("random_state", self.random_state),
                logvar_clip_range=tuple(bnn_state.get("logvar_clip_range", [-8.0, 4.0])),
                gradient_clip_norm=bnn_state.get("gradient_clip_norm"),
                weight_init_scale=float(bnn_state.get("weight_init_scale", 1.0)),
                hidden_activation=str(bnn_state.get("hidden_activation", "relu")),
                likelihood_std=float(bnn_state.get("likelihood_std", 1.0)),
                min_variance=float(bnn_state.get("min_variance", 1e-6)),
                stability_epsilon=float(bnn_state.get("stability_epsilon", 1e-8)),
                leaky_relu_slope=float(bnn_state.get("leaky_relu_slope", 0.01)),
            )
            self.bnn._load_from_payload(bnn_state, validate_shapes=True)

    def save_checkpoint(self, path: str) -> Path:
        output_path = Path(path)
        checkpoint = self.export_state()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("wb") as handle:
                pickle.dump(checkpoint, handle, protocol=self.checkpoint_protocol)
            logger.info("Meta-learning checkpoint saved to %s", output_path)
            return output_path
        except Exception as exc:
            raise CheckpointSaveError(
                f"Failed to save meta-learning checkpoint to {output_path}.",
                component="meta_learning",
                details={"path": str(output_path)},
                cause=exc,
            ) from exc

    def load_checkpoint(self, path: str) -> None:
        input_path = Path(path)
        if not input_path.exists():
            raise CheckpointNotFoundError(
                f"Meta-learning checkpoint not found: {input_path}",
                component="meta_learning",
                details={"path": str(input_path)},
            )

        try:
            with input_path.open("rb") as handle:
                checkpoint = pickle.load(handle)
            self.import_state(checkpoint)
            logger.info("Meta-learning checkpoint loaded from %s", input_path)
        except AdaptiveError:
            raise
        except Exception as exc:
            raise CheckpointLoadError(
                f"Failed to load meta-learning checkpoint from {input_path}.",
                component="meta_learning",
                details={"path": str(input_path)},
                cause=exc,
            ) from exc


if __name__ == "__main__":
    print("\n=== Running Meta Learning Worker ===\n")
    printer.status("TEST", "Meta Learning Worker initialized", "info")

    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)

    class _DummyWorker:
        def __init__(self, skill_id: int):
            self.skill_id = skill_id
            self.name = f"skill_{skill_id}"
            self.learning_rate = 0.001
            self.exploration_rate = 0.10
            self.entropy_coef = 0.01
            self.gamma = 0.95
            self.actor_param = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
            self.critic_param = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float32))
            self.actor_optimizer = torch.optim.Adam([self.actor_param], lr=self.learning_rate)
            self.critic_optimizer = torch.optim.Adam([self.critic_param], lr=self.learning_rate)
            self._recent_rewards = [0.1, 0.2, 0.3, 0.4, 0.5]

        def get_performance_metrics(self) -> Dict[str, float]:
            rewards = list(self._recent_rewards)
            avg_reward = float(np.mean(rewards))
            return {
                "skill_id": self.skill_id,
                "name": self.name,
                "episode_count": len(rewards),
                "avg_reward": avg_reward,
                "success_rate": float(np.mean([1.0 if r > 0 else 0.0 for r in rewards])),
                "recent_reward": float(np.mean(rewards[-3:])),
            }

    worker = MetaLearningWorker()
    worker.register_skill_workers({
        0: _DummyWorker(skill_id=0),
        1: _DummyWorker(skill_id=1),
    })

    collected = worker.collect_performance_metrics()
    printer.pretty("Collect", collected, "success" if collected else "error")

    print("\n* * * * * Phase 2 - Experience * * * * *\n")
    hyperparams = {
        "learning_rate": 0.001,
        "exploration_rate": 0.1,
        "entropy_coef": 0.01,
        "discount_factor": 0.95,
    }
    performance = 0.78
    stored = worker.store_hyperparameter_experience(hyperparams=hyperparams, performance=performance)
    printer.pretty("Store", stored, "success")

    for idx in range(8):
        worker.store_hyperparameter_experience(
            hyperparams={
                "learning_rate": float(np.clip(0.001 + idx * 0.0002, 1e-5, 0.01)),
                "exploration_rate": float(np.clip(0.08 + idx * 0.01, 0.01, 0.3)),
                "entropy_coef": float(np.clip(0.01 + idx * 0.002, 0.001, 0.1)),
                "discount_factor": float(np.clip(0.93 + idx * 0.005, 0.9, 0.999)),
            },
            performance=float(0.65 + idx * 0.03),
        )

    suggest = worker.suggest_hyperparameters()
    printer.pretty("Suggest", suggest, "success" if suggest else "error")

    updates = worker.update_skill_hyperparameters(hyperparams=suggest)
    printer.pretty("Update", updates, "success")

    print("\n* * * * * Phase 3 - Optimization Step * * * * *\n")
    result = worker.optimization_step()
    printer.pretty("Optimize", result, "success")

    print("\n* * * * * Phase 4 - Report / Checkpoint * * * * *\n")
    report = worker.get_optimization_report()
    printer.pretty("Report", {
        "total_configurations": report.get("total_configurations"),
        "best_performance": report.get("best_performance"),
        "optimization_steps": report.get("optimization_steps"),
    }, "success")

    ckpt_path = Path("meta_learning_worker_test.pkl")
    worker.save_checkpoint(str(ckpt_path))

    restored = MetaLearningWorker()
    restored.load_checkpoint(str(ckpt_path))
    restored_report = restored.get_optimization_report()
    printer.pretty("Restore", {
        "total_configurations": restored_report.get("total_configurations"),
        "best_performance": restored_report.get("best_performance"),
        "optimization_steps": restored_report.get("optimization_steps"),
    }, "success")

    if ckpt_path.exists():
        ckpt_path.unlink()

    print("\n=== Test ran successfully ===\n")
