
from __future__ import annotations

import json
import pickle
import hashlib
import signal
import sys
import threading
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from scipy.stats import entropy
from collections import defaultdict, deque
from datetime import datetime, timedelta

from src.utils.buffer.distributed_replay_buffer import DistributedReplayBuffer
from .utils.config_loader import load_global_config, get_config_section
from .utils.sgd_regressor import SGDRegressor
from .utils.adaptive_errors import *
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Adaptive Memory")
printer = PrettyPrinter

sys.stdout.flush()


class MultiModalMemory:
    """Reinforcement Learning Optimized Memory System with:
    - Policy parameter evolution tracking
    - Experience replay with self-tuning prioritization
    - Causal analysis of policy changes
    - Automated parameter tuning memory

    Production-oriented extensions
    ------------------------------
    - Structured error handling integrated with `adaptive_errors`
    - Robust context hashing for dictionaries, numpy arrays, and nested payloads
    - Separate regressors for parameter impact and intervention causal analysis
    - Replay-buffer integration with configurable sampling strategies
    - State export/import and checkpoint persistence helpers
    - Safer forgetting, retrieval, consolidation, and diagnostics workflows
    """

    PARAMETER_COLUMNS = [
        "timestamp",
        "learning_rate",
        "exploration_rate",
        "discount_factor",
        "temperature",
        "performance",
    ]
    TRACKED_PARAMS = ("learning_rate", "exploration_rate", "discount_factor", "temperature")

    def __init__(self) -> None:
        self.config = load_global_config()
        self.memory_config = get_config_section("adaptive_memory")
        self.sgd_config = get_config_section("sgd_regressor")
        self.rl_config = get_config_section("rl")
        self.parameter_tuner_config = get_config_section("parameter_tuner")

        self._load_config()
        self._initialize_runtime_state()
        self._register_signal_handlers()

        logger.info("Multi Modal Memory successfully initialized")

    # ------------------------------------------------------------------
    # Initialization and configuration
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load and validate configuration from adaptive_config.yaml."""
        try:
            self.episodic_capacity = int(self.memory_config.get("episodic_capacity", 1000))
            self.experience_staleness_days = float(self.memory_config.get("experience_staleness_days", 7))
            self.semantic_decay_rate = float(self.memory_config.get("semantic_decay_rate", 0.9))
            self.semantic_threshold = float(self.memory_config.get("semantic_threshold", 0.5))
            self.min_memory_strength = float(self.memory_config.get("min_memory_strength", 0.1))
            self.replay_capacity = int(self.memory_config.get("replay_capacity", 50000))
            self.replay_staleness_days = float(
                self.memory_config.get("replay_staleness_days", self.experience_staleness_days)
            )
            self.drift_threshold = float(self.memory_config.get("drift_threshold", 0.4))
            self.priority_alpha = float(self.memory_config.get("priority_alpha", 0.6))
            self.retrieval_limit = int(self.memory_config.get("retrieval_limit", 5))
            self.enable_goals = bool(self.memory_config.get("enable_goals", False))
            self.goal_dim = int(self.memory_config.get("goal_dim", 0))
            self.max_size = int(self.memory_config.get("max_size", 5000))
            self.goal_capacity = int(self.memory_config.get("goal_capacity", 1000))
            self.enable_policy_grad = bool(self.memory_config.get("enable_policy_grad", True))
            self.uncertainty_dropout = float(self.memory_config.get("uncertainty_dropout", 0.2))

            self.semantic_max_strength = float(self.memory_config.get("semantic_max_strength", 1.0))
            self.reinforcement_boost_factor = float(self.memory_config.get("reinforcement_boost_factor", 1.2))
            self.replay_sample_strategy = str(
                self.memory_config.get("replay_sample_strategy", "uniform")
            ).lower()
            self.replay_sample_beta = float(self.memory_config.get("replay_sample_beta", 0.4))
            self.replay_default_agent_id = str(self.memory_config.get("replay_default_agent_id", "default"))
            self.allow_partial_replay_failures = bool(
                self.memory_config.get("allow_partial_replay_failures", True)
            )
            self.context_match_boost = float(self.memory_config.get("context_match_boost", 1.25))
            self.recency_weight = float(self.memory_config.get("recency_weight", 0.15))
            self.reward_weight = float(self.memory_config.get("reward_weight", 0.15))
            self.strength_weight = float(self.memory_config.get("strength_weight", 0.20))
            self.parameter_weight = float(self.memory_config.get("parameter_weight", 0.50))
            self.max_parameter_history = int(self.memory_config.get("max_parameter_history", 5000))
            self.parameter_impact_window = int(self.memory_config.get("parameter_impact_window", 100))
            self.drift_bins = int(self.memory_config.get("drift_bins", 10))
            self.drift_min_window = int(self.memory_config.get("drift_min_window", 30))
            self.memory_bias_temperature = float(self.memory_config.get("memory_bias_temperature", 1.0))
            self.memory_bias_scale = float(self.memory_config.get("memory_bias_scale", 2.0))
            self.checkpoint_protocol = int(self.memory_config.get("checkpoint_protocol", pickle.HIGHEST_PROTOCOL))
            self.track_access_on_retrieval = bool(self.memory_config.get("track_access_on_retrieval", True))
            self.default_context_on_missing = bool(self.memory_config.get("default_context_on_missing", True))

            self.action_dim = int(self.rl_config.get("action_dim", 1))
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationValueError(
                "Failed to parse adaptive memory configuration values.",
                component="adaptive_memory",
                details={"section": "adaptive_memory"},
                remediation="Ensure adaptive memory configuration values are valid scalars.",
                cause=exc,
            ) from exc

        ensure_positive(self.episodic_capacity, "episodic_capacity", component="adaptive_memory")
        ensure_positive(self.experience_staleness_days, "experience_staleness_days", component="adaptive_memory")
        ensure_in_range(self.semantic_decay_rate, "semantic_decay_rate", minimum=0.0, maximum=1.0, component="adaptive_memory")
        ensure_in_range(self.semantic_threshold, "semantic_threshold", minimum=0.0, component="adaptive_memory")
        ensure_in_range(self.min_memory_strength, "min_memory_strength", minimum=0.0, maximum=1.0, component="adaptive_memory")
        ensure_positive(self.replay_capacity, "replay_capacity", component="adaptive_memory")
        ensure_positive(self.replay_staleness_days, "replay_staleness_days", component="adaptive_memory")
        ensure_in_range(self.drift_threshold, "drift_threshold", minimum=0.0, component="adaptive_memory")
        ensure_in_range(self.priority_alpha, "priority_alpha", minimum=0.0, maximum=3.0, component="adaptive_memory")
        ensure_positive(self.retrieval_limit, "retrieval_limit", component="adaptive_memory")
        ensure_positive(self.max_size, "max_size", component="adaptive_memory")
        ensure_positive(self.goal_capacity, "goal_capacity", component="adaptive_memory")
        ensure_in_range(self.uncertainty_dropout, "uncertainty_dropout", minimum=0.0, maximum=1.0, component="adaptive_memory")
        ensure_in_range(self.semantic_max_strength, "semantic_max_strength", minimum=0.0, maximum=10.0, component="adaptive_memory")
        ensure_positive(self.reinforcement_boost_factor, "reinforcement_boost_factor", component="adaptive_memory")
        ensure_in_range(self.replay_sample_beta, "replay_sample_beta", minimum=0.0, maximum=1.0, component="adaptive_memory")
        ensure_positive(self.context_match_boost, "context_match_boost", component="adaptive_memory")
        ensure_in_range(self.recency_weight, "recency_weight", minimum=0.0, maximum=1.0, component="adaptive_memory")
        ensure_in_range(self.reward_weight, "reward_weight", minimum=0.0, maximum=1.0, component="adaptive_memory")
        ensure_in_range(self.strength_weight, "strength_weight", minimum=0.0, maximum=1.0, component="adaptive_memory")
        ensure_in_range(self.parameter_weight, "parameter_weight", minimum=0.0, maximum=1.0, component="adaptive_memory")
        ensure_positive(self.max_parameter_history, "max_parameter_history", component="adaptive_memory")
        ensure_positive(self.parameter_impact_window, "parameter_impact_window", component="adaptive_memory")
        ensure_positive(self.drift_bins, "drift_bins", component="adaptive_memory")
        ensure_positive(self.drift_min_window, "drift_min_window", component="adaptive_memory")
        ensure_positive(self.memory_bias_temperature, "memory_bias_temperature", component="adaptive_memory")
        ensure_positive(self.memory_bias_scale, "memory_bias_scale", component="adaptive_memory")
        ensure_positive(self.action_dim, "action_dim", component="adaptive_memory")

        if self.replay_sample_strategy not in {"uniform", "prioritized", "reward", "agent_balanced"}:
            raise InvalidConfigurationValueError(
                f"Unsupported replay_sample_strategy: {self.replay_sample_strategy}",
                component="adaptive_memory",
                details={"replay_sample_strategy": self.replay_sample_strategy},
                remediation="Use one of: uniform, prioritized, reward, agent_balanced.",
            )

        total_weight = self.recency_weight + self.reward_weight + self.strength_weight + self.parameter_weight
        if total_weight <= 0.0:
            raise InvalidConfigurationValueError(
                "Retrieval scoring weights must sum to a positive value.",
                component="adaptive_memory",
                details={
                    "recency_weight": self.recency_weight,
                    "reward_weight": self.reward_weight,
                    "strength_weight": self.strength_weight,
                    "parameter_weight": self.parameter_weight,
                },
            )
        self.recency_weight /= total_weight
        self.reward_weight /= total_weight
        self.strength_weight /= total_weight
        self.parameter_weight /= total_weight

        self.staleness_threshold = timedelta(days=self.experience_staleness_days)

    def _initialize_runtime_state(self) -> None:
        self.param_names = list(self.TRACKED_PARAMS)
        self.episodic: deque = deque(maxlen=self.episodic_capacity)
        self.parameter_evolution = pd.DataFrame(columns=self.PARAMETER_COLUMNS)
        self.policy_interventions: List[Dict[str, Any]] = []

        self.semantic = defaultdict(self._default_semantic_entry)
        self.parameter_impact_model = SGDRegressor()
        self.intervention_model = SGDRegressor()
        self.causal_model = self.intervention_model  # Backward-compatible alias.

        self._state_lock = threading.RLock()
        self._last_drift_score: Optional[float] = None
        self._last_drift_detected_at: Optional[datetime] = None
        self._experience_counter = 0

        self.concept_drift_scores: List[float] = []
        self.reward_sum = 0.0
        self.reward_count = 0
        self.max_abs_reward = 0.0

        self._init_drb()

    def _default_semantic_entry(self) -> Dict[str, Any]:
        return {
            "strength": 1.0,
            "last_accessed": datetime.now(),
            "data": None,
            "context_hash": "",
            "count": 0,
        }

    def _init_drb(self) -> None:
        user_config = {
            "distributed": {
                "capacity": self.replay_capacity,
                "staleness_threshold_days": self.replay_staleness_days,
                "prioritization_alpha": self.priority_alpha,
                "seed": self.sgd_config.get("random_state"),
            }
        }
        self.replay_buffer = DistributedReplayBuffer(user_config=user_config)

    def _reset_replay_buffer_state(self) -> None:
        """Clear replay-buffer state more thoroughly than the upstream clear() helper."""
        if hasattr(self.replay_buffer, "clear"):
            self.replay_buffer.clear()

        if hasattr(self.replay_buffer, "timestamps"):
            self.replay_buffer.timestamps.clear()
        if hasattr(self.replay_buffer, "priorities"):
            self.replay_buffer.priorities.clear()
        if hasattr(self.replay_buffer, "agent_experience_map"):
            self.replay_buffer.agent_experience_map.clear()
        if hasattr(self.replay_buffer, "reward_stats"):
            self.replay_buffer.reward_stats = {"sum": 0.0, "max": -np.inf, "min": np.inf}
        if hasattr(self.replay_buffer, "agent_rewards"):
            self.replay_buffer.agent_rewards.clear()

    def _register_signal_handlers(self) -> None:
        """Register signal handlers only when running on main interpreter thread."""
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, self._handle_emergency_exit)
            else:
                logger.debug("Skipping SIGINT handler registration outside main thread.")
        except (ValueError, RuntimeError) as exc:
            logger.warning("Unable to register Adaptive Memory signal handlers: %s", exc)

    def _handle_emergency_exit(self, signum: int, frame: Any) -> None:
        logger.info("Emergency exit triggered. Cleaning up Adaptive Memory...")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _sanitize_for_serialization(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat()
        if isinstance(value, timedelta):
            return value.total_seconds()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize_for_serialization(v) for v in value]
        if isinstance(value, deque):
            return [self._sanitize_for_serialization(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._sanitize_for_serialization(v) for k, v in value.items()}
        if isinstance(value, pd.DataFrame):
            return value.to_dict("records")
        return repr(value)

    def _restore_datetime(self, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return value
        return value

    def _normalize_experience(self, experience: Mapping[str, Any]) -> Dict[str, Any]:
        exp = dict(experience)
        if "timestamp" in exp:
            exp["timestamp"] = self._restore_datetime(exp["timestamp"])
        if "strength" in exp:
            exp["strength"] = float(exp["strength"])
        if "reward" in exp and exp["reward"] is not None:
            exp["reward"] = float(exp["reward"])
        if "done" in exp:
            exp["done"] = bool(exp["done"])
        if "params" not in exp or exp["params"] is None:
            exp["params"] = {}
        return exp

    def _serialize_semantic_store(self) -> Dict[str, Dict[str, Any]]:
        payload: Dict[str, Dict[str, Any]] = {}
        for key, value in self.semantic.items():
            payload[key] = {
                "strength": float(value.get("strength", 1.0)),
                "last_accessed": self._sanitize_for_serialization(value.get("last_accessed")),
                "data": self._sanitize_for_serialization(value.get("data")),
                "context_hash": str(value.get("context_hash", "")),
                "count": int(value.get("count", 0)),
            }
        return payload

    def _restore_semantic_store(self, semantic_data: Mapping[str, Mapping[str, Any]]) -> defaultdict:
        semantic_store: defaultdict = defaultdict(self._default_semantic_entry)
        for key, value in semantic_data.items():
            entry = self._default_semantic_entry()
            entry.update(dict(value))
            entry["last_accessed"] = self._restore_datetime(entry.get("last_accessed"))
            entry["strength"] = float(entry.get("strength", 1.0))
            entry["count"] = int(entry.get("count", 0))
            semantic_store[str(key)] = entry
        return semantic_store

    # ------------------------------------------------------------------
    # Core memory operations
    # ------------------------------------------------------------------

    def store_experience(
        self,
        state: Any,
        action: Any,
        reward: Union[int, float],
        next_state: Any = None,
        done: bool = False,
        context: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Store experience with timestamp, strength, semantic update, and replay insertion."""
        printer.status("INIT", "Experience storage successfully initialized", "info")

        if reward is None or not isinstance(reward, (int, float, np.number)) or not np.isfinite(reward):
            raise ExperienceValidationError(
                "Reward must be a finite numeric value.",
                component="adaptive_memory",
                details={"reward": self._sanitize_for_serialization(reward)},
                remediation="Ensure all experiences provide a scalar finite reward.",
            )

        if context is not None and not isinstance(context, dict):
            raise InvalidTypeError(
                "context must be a dictionary when provided.",
                component="adaptive_memory",
                details={"context_type": type(context).__name__},
            )
        if params is not None and not isinstance(params, dict):
            raise InvalidTypeError(
                "params must be a dictionary when provided.",
                component="adaptive_memory",
                details={"params_type": type(params).__name__},
            )

        with self._state_lock:
            context_hash = (
                self._hash_context(context)
                if context
                else self._fallback_context_hash(state, action)
                if self.default_context_on_missing
                else ""
            )
            timestamp = datetime.now()
            tracked_params = self._extract_experience_params(params)

            experience = {
                "id": self._experience_counter,
                "state": state,
                "action": action,
                "reward": float(reward),
                "next_state": next_state,
                "done": bool(done),
                "timestamp": timestamp,
                "strength": 1.0,
                "context_hash": context_hash,
                "params": tracked_params,
                "metadata": dict(kwargs) if kwargs else {},
            }
            self._experience_counter += 1

            self.episodic.append(experience)
            self._update_reward_statistics(float(reward))
            self._update_semantic_memory(experience)

            try:
                priority = self._calculate_priority(float(reward))
                self.replay_buffer.push(
                    agent_id=str(kwargs.get("agent_id", self.replay_default_agent_id)),
                    state=state,
                    action=action,
                    reward=float(reward),
                    next_state=next_state,
                    done=bool(done),
                    priority=priority,
                )
            except Exception as exc:
                if self.allow_partial_replay_failures:
                    logger.warning("Replay buffer push failed; episodic memory retained: %s", exc)
                else:
                    raise ReplayBufferError(
                        "Replay buffer push failed during experience storage.",
                        component="adaptive_memory",
                        details={"experience_id": experience["id"]},
                        remediation="Inspect replay buffer availability and input payload compatibility.",
                        cause=exc,
                    ) from exc

            return experience

    def _extract_experience_params(self, params: Optional[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
        payload: Dict[str, Optional[float]] = {name: None for name in self.TRACKED_PARAMS}
        if not params:
            return payload

        for name in self.TRACKED_PARAMS:
            value = params.get(name)
            if value is None:
                payload[name] = None
            elif isinstance(value, (int, float, np.number)) and np.isfinite(value):
                payload[name] = float(value)
            else:
                raise ExperienceValidationError(
                    f"Experience parameter '{name}' must be numeric when provided.",
                    component="adaptive_memory",
                    details={"name": name, "value": self._sanitize_for_serialization(value)},
                )
        return payload

    def clear_episodic(self) -> None:
        """Clear only episodic memory."""
        with self._state_lock:
            self.episodic.clear()

    def clear(self, clear_replay_buffer: bool = True) -> None:
        """Reset all memory stores and optionally the replay buffer."""
        with self._state_lock:
            self.episodic = deque(maxlen=self.episodic_capacity)
            self.semantic = defaultdict(self._default_semantic_entry)
            self.parameter_evolution = pd.DataFrame(columns=self.PARAMETER_COLUMNS)
            self.policy_interventions = []
            self.concept_drift_scores = []
            self.reward_sum = 0.0
            self.reward_count = 0
            self.max_abs_reward = 0.0
            self._last_drift_score = None
            self._last_drift_detected_at = None
            self._experience_counter = 0

            self.parameter_impact_model.reset()
            self.intervention_model.reset()
            self.causal_model = self.intervention_model

            if clear_replay_buffer:
                self._reset_replay_buffer_state()

    # ------------------------------------------------------------------
    # Hashing and semantic memory
    # ------------------------------------------------------------------

    def _fallback_context_hash(self, state: Any, action: Any) -> str:
        payload = {"state": self._sanitize_for_serialization(state), "action": self._sanitize_for_serialization(action)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _hash_context(self, context: Mapping[str, Any]) -> str:
        """Create a deterministic, numpy-safe, collision-resistant hash from context."""
        ensure_instance(context, Mapping, "context", component="adaptive_memory")
        sanitized = self._sanitize_for_serialization(dict(context))
        context_blob = json.dumps(sanitized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        context_hash = hashlib.sha256(context_blob.encode("utf-8")).hexdigest()
        return context_hash

    def _generate_context_hash(self, context: dict) -> str:
        """Backward-compatible context-hash helper."""
        return self._hash_context(context)

    def _update_reward_statistics(self, reward: float) -> None:
        abs_reward = abs(reward)
        self.reward_sum += abs_reward
        self.reward_count += 1
        self.max_abs_reward = max(self.max_abs_reward, abs_reward)

    def _update_semantic_memory(self, experience: Mapping[str, Any]) -> None:
        """Convert high-impact experiences into semantic knowledge entries."""
        printer.status("INIT", "Semantic memory update started", "info")

        reward = float(experience.get("reward", 0.0))
        if abs(reward) < self.semantic_threshold:
            return

        context_hash = str(experience.get("context_hash", ""))
        if not context_hash:
            return

        context_key = f"ctx_{context_hash[:12]}"
        action = experience.get("action")
        entry = self.semantic[context_key]

        if entry["data"] is None:
            entry["data"] = {
                "action": action,
                "reward": reward,
                "avg_reward": reward,
                "last_reward": reward,
                "count": 1,
            }
            entry["count"] = 1
        else:
            data = dict(entry["data"])
            count = int(entry.get("count", data.get("count", 0))) + 1
            avg_reward = ((data.get("avg_reward", reward) * (count - 1)) + reward) / max(1, count)
            data.update(
                {
                    "action": action if action is not None else data.get("action"),
                    "reward": avg_reward,
                    "avg_reward": avg_reward,
                    "last_reward": reward,
                    "count": count,
                }
            )
            entry["data"] = data
            entry["count"] = count

        entry["strength"] = min(self.semantic_max_strength, float(entry.get("strength", 1.0)) + (abs(reward) * 0.01))
        entry["last_accessed"] = datetime.now()
        entry["context_hash"] = context_hash

    def reinforce_memory(self, key: str, boost_factor: Optional[float] = None) -> None:
        """Strengthen frequently accessed semantic memories."""
        if key not in self.semantic:
            return

        factor = self.reinforcement_boost_factor if boost_factor is None else float(boost_factor)
        ensure_positive(factor, "boost_factor", component="adaptive_memory")

        self.semantic[key]["strength"] = min(
            self.semantic_max_strength,
            float(self.semantic[key]["strength"]) * factor,
        )
        self.semantic[key]["last_accessed"] = datetime.now()

    # ------------------------------------------------------------------
    # Priority, parameters, and drift
    # ------------------------------------------------------------------

    def _calculate_priority(self, reward: float) -> float:
        """
        Self-tuning priority calculation using reward normalization and adaptive scaling.
        """
        if reward is None or not isinstance(reward, (int, float, np.number)) or not np.isfinite(reward):
            raise ExperienceValidationError(
                "Reward must be a finite numeric value for priority calculation.",
                component="adaptive_memory",
                details={"reward": self._sanitize_for_serialization(reward)},
            )

        abs_reward = abs(float(reward))
        norm_denom = self.max_abs_reward if self.max_abs_reward > 1e-12 else 1e-12
        count_denom = self.reward_count if self.reward_count > 0 else 1

        normalized_reward = abs_reward / norm_denom
        avg_reward = self.reward_sum / count_denom if count_denom > 0 else 0.0
        reward_deviation = abs_reward - avg_reward
        deviation_factor = 1.0 + np.tanh(reward_deviation / (avg_reward + 1e-12))

        raw_priority = (normalized_reward * deviation_factor + 0.01) ** self.priority_alpha
        clipped_priority = float(np.clip(raw_priority, 1e-6, 1.0))
        return clipped_priority

    def log_parameters(self, performance: float, params: Mapping[str, Any]) -> None:
        """Track evolution of learning parameters."""
        printer.status("INIT", "Parameter logger successfully initialized", "info")

        if performance is None or not isinstance(performance, (int, float, np.number)) or not np.isfinite(performance):
            raise InvalidValueError(
                "performance must be a finite numeric value.",
                component="adaptive_memory",
                details={"performance": self._sanitize_for_serialization(performance)},
            )
        ensure_instance(params, Mapping, "params", component="adaptive_memory")

        entry = {
            "timestamp": datetime.now(),
            "learning_rate": self._coerce_optional_float(params.get("learning_rate")),
            "exploration_rate": self._coerce_optional_float(params.get("exploration_rate")),
            "discount_factor": self._coerce_optional_float(params.get("discount_factor")),
            "temperature": self._coerce_optional_float(params.get("temperature")),
            "performance": float(performance),
        }

        with self._state_lock:
            self.parameter_evolution = pd.concat(
                [self.parameter_evolution, pd.DataFrame([entry])],
                ignore_index=True,
            )
            if len(self.parameter_evolution) > self.max_parameter_history:
                self.parameter_evolution = self.parameter_evolution.iloc[-self.max_parameter_history :].reset_index(drop=True)

    def _coerce_optional_float(self, value: Any) -> float:
        if value is None:
            return float("nan")
        if isinstance(value, (int, float, np.number)):
            # Accept nan/inf but convert to float("nan") to avoid breaking logging
            try:
                fval = float(value)
                if np.isfinite(fval):
                    return fval
                else:
                    return float("nan")
            except (TypeError, ValueError):
                pass
        raise InvalidValueError(
            "Parameter values must be numeric when provided.",
            component="adaptive_memory",
            details={"value": value},
        )

    def analyze_parameter_impact(self, window_size: Optional[int] = None) -> Dict[str, float]:
        """Analyze relationships between parameter changes and performance."""
        printer.status("INIT", "Parameter impact analysis successfully initialized", "info")

        window = self.parameter_impact_window if window_size is None else int(window_size)
        ensure_positive(window, "window_size", component="adaptive_memory")

        if len(self.parameter_evolution) < max(2, window):
            return {}

        recent = self.parameter_evolution.iloc[-window:].copy()
        recent = recent.dropna(subset=["learning_rate", "exploration_rate", "discount_factor", "temperature", "performance"])
        if len(recent) < 2:
            return {}

        X = recent[["learning_rate", "exploration_rate", "discount_factor", "temperature"]].astype(float).values
        y = recent["performance"].astype(float).values

        try:
            self.parameter_impact_model.partial_fit(X, y)
        except AdaptiveError:
            raise
        except Exception as exc:
            raise wrap_exception(
                exc,
                AdaptiveMemoryError,
                "Failed to fit parameter impact model.",
                component="adaptive_memory",
                details={"window_size": window},
                remediation="Inspect logged parameter history for invalid or degenerate values.",
            ) from exc

        coef = np.asarray(self.parameter_impact_model.coef_, dtype=np.float64)
        if coef.shape[0] != 4:
            raise DimensionMismatchError(
                "Parameter impact model returned an unexpected coefficient shape.",
                component="adaptive_memory",
                details={"coef_shape": tuple(coef.shape)},
            )

        return {
            "learning_rate_impact": float(coef[0]),
            "exploration_impact": float(coef[1]),
            "discount_impact": float(coef[2]),
            "temperature_impact": float(coef[3]),
        }

    def detect_drift(self, window_size: Optional[int] = None, track: bool = True) -> bool:
        """Performance-based concept drift detection using KL divergence."""
        window = self.drift_min_window if window_size is None else int(window_size)
        ensure_positive(window, "window_size", component="adaptive_memory")

        if len(self.parameter_evolution) < 2 * window:
            return False

        try:
            recent = self.parameter_evolution["performance"].iloc[-window:].astype(float).values
            historical = self.parameter_evolution["performance"].iloc[-2 * window : -window].astype(float).values

            p = np.histogram(recent, bins=self.drift_bins)[0].astype(np.float64) + 1e-12
            q = np.histogram(historical, bins=self.drift_bins)[0].astype(np.float64) + 1e-12
            p /= p.sum()
            q /= q.sum()

            kl_div = float(entropy(p, q))
            if track:
                self.concept_drift_scores.append(kl_div)
            self._last_drift_score = kl_div

            is_drift = kl_div > self.drift_threshold
            if is_drift:
                self._last_drift_detected_at = datetime.now()
            return is_drift

        except Exception as exc:
            raise DriftDetectionError(
                "Failed to compute concept drift score.",
                component="adaptive_memory",
                details={"window_size": window},
                remediation="Inspect parameter evolution history for non-numeric values.",
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Reporting and intervention analysis
    # ------------------------------------------------------------------

    def _analyze_parameters(self) -> Dict[str, Any]:
        if self.parameter_evolution.empty:
            return {}
        numeric_df = self.parameter_evolution.copy()
        numeric_df = numeric_df.drop(columns=["timestamp"], errors="ignore")
        return numeric_df.describe(include="all").to_dict()

    def _intervention_statistics(self) -> Dict[str, Any]:
        if not self.policy_interventions:
            return {}

        df = pd.DataFrame(self.policy_interventions)
        if df.empty or "type" not in df.columns:
            return {}

        if "causal_impact" not in df.columns:
            df["causal_impact"] = np.nan

        stats = df.groupby("type", dropna=False).agg(
            effect_size_mean=("effect_size", "mean"),
            effect_size_std=("effect_size", "std"),
            effect_size_count=("effect_size", "count"),
            causal_impact_median=("causal_impact", "median"),
        )
        return stats.fillna(0.0).to_dict(orient="index")

    def _semantic_analysis(self) -> Dict[str, Any]:
        if not self.semantic:
            return {"total_concepts": 0, "avg_strength": 0.0, "active_contexts": 0}

        strengths = [float(v["strength"]) for v in self.semantic.values()]
        return {
            "total_concepts": len(self.semantic),
            "avg_strength": float(np.mean(strengths)) if strengths else 0.0,
            "active_contexts": len({v["context_hash"] for v in self.semantic.values() if v.get("context_hash")}),
        }

    def get_memory_report(self) -> Dict[str, Any]:
        """Generate a unified memory analysis report."""
        printer.status("INIT", "Memory report initialized", "info")

        replay_stats = self.replay_buffer.stats() if hasattr(self.replay_buffer, "stats") else {}
        return {
            "sizes": self.size(),
            "parameter_analysis": self._analyze_parameters(),
            "parameter_impact": self.analyze_parameter_impact(window_size=min(len(self.parameter_evolution), self.parameter_impact_window))
            if len(self.parameter_evolution) >= 2
            else {},
            "intervention_impact": self._intervention_statistics(),
            "drift_status": {
                "detected": self.detect_drift(track=False),
                "latest_score": self._last_drift_score,
                "threshold": self.drift_threshold,
                "num_scores": len(self.concept_drift_scores),
                "last_detected_at": self._sanitize_for_serialization(self._last_drift_detected_at),
            },
            "replay_stats": replay_stats,
            "semantic_summary": self._semantic_analysis(),
            "avg_strength": float(np.mean([v["strength"] for v in self.semantic.values()])) if self.semantic else 0.0,
        }

    def apply_policy_intervention(self, intervention: Mapping[str, Any], effect: Mapping[str, Any]) -> None:
        """Log policy changes and their measured effects."""
        printer.status("INIT", "Policy intervention logging initialized", "info")

        ensure_instance(intervention, Mapping, "intervention", component="adaptive_memory")
        ensure_instance(effect, Mapping, "effect", component="adaptive_memory")

        params_before = intervention.get("params_before")
        params_after = intervention.get("params_after")
        if params_before is None:
            raise MissingFieldError(
                "intervention.params_before is required.",
                component="adaptive_memory",
            )
        if params_after is None:
            raise MissingFieldError(
                "intervention.params_after is required.",
                component="adaptive_memory",
            )

        record = {
            "timestamp": datetime.now(),
            "type": intervention.get("type", "unknown"),
            "params_before": dict(params_before),
            "params_after": dict(params_after),
            "effect_size": float(effect.get("performance_delta", 0.0)),
            "causal_impact": None,
        }
        self.policy_interventions.append(record)
        self._update_causal_model(record, effect)

    def _update_causal_model(self, intervention: Mapping[str, Any], effect: Mapping[str, Any]) -> None:
        """Update causal relationships between policy changes and outcomes."""
        before = dict(intervention.get("params_before", {}))
        after = dict(intervention.get("params_after", {}))
        performance_delta = effect.get("performance_delta", 0.0)

        X = np.array(
            [
                [
                    float(before.get("learning_rate", 0.0) or 0.0),
                    float(before.get("exploration_rate", 0.0) or 0.0),
                    float(after.get("learning_rate", before.get("learning_rate", 0.0)) or 0.0)
                    - float(before.get("learning_rate", 0.0) or 0.0),
                    float(after.get("exploration_rate", before.get("exploration_rate", 0.0)) or 0.0)
                    - float(before.get("exploration_rate", 0.0) or 0.0),
                ]
            ],
            dtype=np.float64,
        )
        y = np.array([float(performance_delta)], dtype=np.float64)
        self.intervention_model.partial_fit(X, y)

        coef = getattr(self.intervention_model, "coef_", None)
        if coef is not None and len(self.policy_interventions) > 0:
            self.policy_interventions[-1]["causal_impact"] = float(np.mean(np.abs(np.asarray(coef, dtype=np.float64))))

    # ------------------------------------------------------------------
    # Forgetting, consolidation, and retrieval
    # ------------------------------------------------------------------

    def consolidate(self) -> None:
        """Apply forgetting and pruning mechanisms to all memory systems."""
        self._forget_old_episodes()
        self._decay_semantic_memory()
        if hasattr(self.replay_buffer, "_remove_stale_experiences"):
            self.replay_buffer._remove_stale_experiences()
        self._prune_parameter_history()

    def _forget_old_episodes(self) -> None:
        if not self.episodic:
            return

        now = datetime.now()
        retained = deque(maxlen=self.episodic_capacity)
        for experience in self.episodic:
            age = now - experience["timestamp"]
            age_ratio = age.total_seconds() / max(self.staleness_threshold.total_seconds(), 1.0)
            age_factor = max(0.0, 1.0 - age_ratio)
            experience["strength"] = float(experience.get("strength", 1.0)) * self.semantic_decay_rate * age_factor

            if experience["strength"] > self.min_memory_strength and age <= self.staleness_threshold:
                retained.append(experience)

        self.episodic = retained

    def _decay_semantic_memory(self) -> None:
        for key in list(self.semantic.keys()):
            self.semantic[key]["strength"] = float(self.semantic[key]["strength"]) * self.semantic_decay_rate
            if self.semantic[key]["strength"] < self.min_memory_strength:
                del self.semantic[key]

    def _prune_parameter_history(self) -> None:
        if self.parameter_evolution.empty:
            return

        cutoff = datetime.now() - self.staleness_threshold
        if "timestamp" in self.parameter_evolution.columns:
            timestamps = pd.to_datetime(self.parameter_evolution["timestamp"], errors="coerce")
            self.parameter_evolution = self.parameter_evolution.loc[timestamps > cutoff].reset_index(drop=True)

        if len(self.parameter_evolution) > self.max_parameter_history:
            self.parameter_evolution = self.parameter_evolution.iloc[-self.max_parameter_history :].reset_index(drop=True)

    def retrieve(
        self,
        query: Any,
        context: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Context-aware retrieval with semantic and episodic ranking."""
        limit = self.retrieval_limit if limit is None else int(limit)
        ensure_positive(limit, "limit", component="adaptive_memory")

        results: List[Dict[str, Any]] = []
        target_features = self._extract_parameter_features(query)
        query_context_hash = self._hash_context(context) if context else ""

        if query_context_hash:
            semantic_key = f"ctx_{query_context_hash[:12]}"
            if semantic_key in self.semantic:
                entry = self.semantic[semantic_key]
                score = float(np.clip(entry.get("strength", 1.0), 0.0, self.semantic_max_strength))
                results.append(
                    {
                        "data": entry["data"],
                        "score": score,
                        "type": "semantic",
                        "context_hash": semantic_key,
                    }
                )
                if self.track_access_on_retrieval:
                    self.reinforce_memory(semantic_key, boost_factor=1.01)

        now = datetime.now()
        max_reward = self.max_abs_reward if self.max_abs_reward > 1e-12 else 1.0

        for exp in reversed(self.episodic):
            param_score = self._calculate_parameter_similarity(exp, target_features)
            context_score = self.context_match_boost if query_context_hash and exp.get("context_hash") == query_context_hash else 1.0

            age = now - exp["timestamp"]
            recency = max(0.0, 1.0 - (age.total_seconds() / max(self.staleness_threshold.total_seconds(), 1.0)))
            reward_salience = min(1.0, abs(float(exp.get("reward", 0.0))) / max_reward)
            strength = float(np.clip(exp.get("strength", 1.0), 0.0, self.semantic_max_strength)) / max(self.semantic_max_strength, 1e-12)

            combined = (
                self.parameter_weight * param_score
                + self.recency_weight * recency
                + self.reward_weight * reward_salience
                + self.strength_weight * strength
            ) * context_score

            results.append(
                {
                    "data": exp,
                    "score": float(combined),
                    "type": "episodic",
                    "context_hash": exp.get("context_hash", ""),
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def _extract_parameter_features(self, query: Any) -> Dict[str, float]:
        """Convert query text or mapping into parameter-space features."""
        features = {
            "learning_rate": float(self.parameter_tuner_config.get("base_learning_rate", 0.001)),
            "exploration_rate": float(self.parameter_tuner_config.get("base_exploration_rate", 0.1)),
            "discount_factor": float(self.parameter_tuner_config.get("base_discount_factor", 0.95)),
            "temperature": float(self.parameter_tuner_config.get("base_temperature", 1.0)),
        }

        if not self.parameter_evolution.empty:
            recent = self.parameter_evolution.iloc[-1]
            for key in features:
                value = recent.get(key)
                if value is not None and not pd.isnull(value):
                    features[key] = float(value)

        if isinstance(query, Mapping):
            for key in features:
                value = query.get(key)
                if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                    features[key] = float(value)
            return features

        query_text = str(query).lower()
        modifiers = {
            "high": 1.5,
            "increase": 1.3,
            "boost": 1.2,
            "low": 0.5,
            "reduce": 0.7,
            "decrease": 0.8,
        }

        param_ranges = {
            "learning_rate": (
                float(self.parameter_tuner_config.get("min_learning_rate", 1e-4)),
                float(self.parameter_tuner_config.get("max_learning_rate", 1.0)),
            ),
            "exploration_rate": (
                float(self.parameter_tuner_config.get("min_exploration", 0.0)),
                float(self.parameter_tuner_config.get("max_exploration", 1.0)),
            ),
            "discount_factor": (
                float(self.parameter_tuner_config.get("min_discount_factor", 0.0)),
                float(self.parameter_tuner_config.get("max_discount_factor", 1.0)),
            ),
            "temperature": (
                float(self.parameter_tuner_config.get("min_temperature", 0.0)),
                float(self.parameter_tuner_config.get("max_temperature", 10.0)),
            ),
        }

        for param in features:
            aliases = {param, param.replace("_", " ")}
            if any(alias in query_text for alias in aliases):
                for modifier, factor in modifiers.items():
                    if modifier in query_text:
                        lo, hi = param_ranges[param]
                        features[param] = float(np.clip(features[param] * factor, lo, hi))
                        break

        return features

    def _calculate_parameter_similarity(self, experience: Mapping[str, Any], target: Mapping[str, float]) -> float:
        """Compute bounded similarity between experience parameters and target parameters."""
        params = experience.get("params", {}) or {}
        if not isinstance(params, Mapping):
            return 0.0

        similarity_scores: List[float] = []
        for name in self.TRACKED_PARAMS:
            target_val = target.get(name)
            exp_val = params.get(name)

            if target_val is None or exp_val is None or (isinstance(exp_val, float) and np.isnan(exp_val)):
                similarity_scores.append(0.5)
                continue

            scale = max(abs(float(target_val)), 1e-6)
            diff = abs(float(exp_val) - float(target_val)) / scale
            similarity_scores.append(1.0 / (1.0 + diff))

        return float(np.mean(similarity_scores)) if similarity_scores else 0.0

    def _extract_action_reward_from_memory(self, memory: Mapping[str, Any]) -> Tuple[Optional[int], float]:
        data = memory.get("data")
        memory_type = memory.get("type")

        if memory_type == "semantic":
            if isinstance(data, dict):
                action = data.get("action")
                reward = data.get("reward", data.get("avg_reward", data.get("last_reward", 0.0)))
            elif isinstance(data, (list, tuple)) and len(data) >= 2:
                action, reward = data[0], data[1]
            else:
                return None, 0.0
        else:
            if not isinstance(data, Mapping):
                return None, 0.0
            action = data.get("action")
            reward = data.get("reward", 0.0)

        if not isinstance(action, (int, np.integer)):
            return None, 0.0
        if not isinstance(reward, (int, float, np.number)) or not np.isfinite(reward):
            return int(action), 0.0
        return int(action), float(reward)

    def _generate_memory_bias(self, memories: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate action bias vector from retrieved memories.
        Combines semantic and episodic memories into a unified bias signal.
        """
        bias = np.zeros(self.action_dim, dtype=np.float64)
        if not memories:
            return bias

        total_weight = 0.0
        for memory in memories:
            action, reward = self._extract_action_reward_from_memory(memory)
            score = float(memory.get("score", 0.0))
            if action is None or action < 0 or action >= self.action_dim:
                continue

            weight = abs(reward) * max(score, 1e-12)
            bias[action] += reward * score
            total_weight += weight

        if total_weight <= 1e-12 or not np.any(np.isfinite(bias)):
            return np.zeros(self.action_dim, dtype=np.float64)

        bias = bias / total_weight
        temperature = max(self.memory_bias_temperature, 1e-6)
        scaled = bias / temperature
        scaled -= np.max(scaled)
        exp_bias = np.exp(scaled)
        exp_sum = float(np.sum(exp_bias))
        if exp_sum <= 1e-12:
            return np.zeros(self.action_dim, dtype=np.float64)

        softmax_bias = exp_bias / exp_sum
        centered = softmax_bias - np.mean(softmax_bias)
        return self.memory_bias_scale * centered

    # ------------------------------------------------------------------
    # Replay buffer and sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        strategy: Optional[str] = None,
        beta: Optional[float] = None,
        agent_distribution: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Sample a batch from the replay buffer and return it in a structured format.
        """
        ensure_positive(batch_size, "batch_size", component="adaptive_memory")
        strategy = self.replay_sample_strategy if strategy is None else str(strategy).lower()
        beta = self.replay_sample_beta if beta is None else float(beta)

        try:
            raw_batch = self.replay_buffer.sample(
                batch_size=batch_size,
                strategy=strategy,
                beta=beta,
                agent_distribution=agent_distribution,
            )
        except ValueError:
            return {
                "agent_ids": np.array([], dtype=object),
                "states": np.array([], dtype=np.float32),
                "actions": np.array([], dtype=np.int64),
                "rewards": np.array([], dtype=np.float32),
                "next_states": np.array([], dtype=np.float32),
                "dones": np.array([], dtype=np.bool_),
                "advantages": np.array([], dtype=np.float32),
            }
        except Exception as exc:
            raise ReplayBufferError(
                "Replay buffer sampling failed.",
                component="adaptive_memory",
                details={"batch_size": batch_size, "strategy": strategy},
                remediation="Inspect replay buffer state and sampling configuration.",
                cause=exc,
            ) from exc

        return self._format_sampled_batch(raw_batch)

    def _format_sampled_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Normalize replay buffer output into a training-friendly structure.
        Supports tuple-of-arrays batches produced by DistributedReplayBuffer.
        """
        if batch is None:
            return {
                "agent_ids": np.array([], dtype=object),
                "states": np.array([], dtype=np.float32),
                "actions": np.array([], dtype=np.int64),
                "rewards": np.array([], dtype=np.float32),
                "next_states": np.array([], dtype=np.float32),
                "dones": np.array([], dtype=np.bool_),
                "advantages": np.array([], dtype=np.float32),
            }

        if isinstance(batch, tuple) and len(batch) >= 6:
            agent_ids, states, actions, rewards, next_states, dones = batch[:6]
        elif isinstance(batch, list) and batch and isinstance(batch[0], (list, tuple)):
            agent_ids, states, actions, rewards, next_states, dones = zip(*batch)
        else:
            raise ReplayBufferError(
                "Unsupported replay batch structure received from replay buffer.",
                component="adaptive_memory",
                details={"batch_type": type(batch).__name__},
            )

        rewards_arr = np.asarray(rewards, dtype=np.float32)
        if rewards_arr.size == 0:
            advantages = rewards_arr
        elif rewards_arr.size == 1:
            advantages = rewards_arr.copy()
        else:
            advantages = (rewards_arr - rewards_arr.mean()) / (rewards_arr.std() + 1e-8)

        return {
            "agent_ids": np.asarray(agent_ids),
            "states": np.asarray(states, dtype=np.float32),
            "actions": np.asarray(actions),
            "rewards": rewards_arr,
            "next_states": np.asarray(next_states, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.bool_),
            "advantages": np.asarray(advantages, dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Diagnostics and sizing
    # ------------------------------------------------------------------

    def size(self) -> Dict[str, int]:
        """Get comprehensive memory size report."""
        replay_size = len(self.replay_buffer) if hasattr(self.replay_buffer, "__len__") else 0
        return {
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
            "parameter_history": len(self.parameter_evolution),
            "policy_interventions": len(self.policy_interventions),
            "replay_buffer": replay_size,
            "concept_drift_scores": len(self.concept_drift_scores),
            "total": len(self.episodic) + len(self.semantic) + len(self.parameter_evolution) + len(self.policy_interventions),
        }

    def get_recent_experiences(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = self.retrieval_limit if limit is None else int(limit)
        ensure_positive(limit, "limit", component="adaptive_memory")
        return list(self.episodic)[-limit:]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def export_state(self, include_replay_buffer: bool = True) -> Dict[str, Any]:
        """Export a serializable memory state for checkpointing."""
        with self._state_lock:
            replay_entries: List[Dict[str, Any]] = []
            if include_replay_buffer and hasattr(self.replay_buffer, "get_all"):
                agent_ids, states, actions, rewards, next_states, dones = self.replay_buffer.get_all()
                for agent_id, state, action, reward, next_state, done in zip(
                    agent_ids, states, actions, rewards, next_states, dones
                ):
                    replay_entries.append(
                        {
                            "agent_id": self._sanitize_for_serialization(agent_id),
                            "state": self._sanitize_for_serialization(state),
                            "action": self._sanitize_for_serialization(action),
                            "reward": self._sanitize_for_serialization(reward),
                            "next_state": self._sanitize_for_serialization(next_state),
                            "done": self._sanitize_for_serialization(done),
                        }
                    )

            payload = {
                "config_snapshot": {
                    "episodic_capacity": self.episodic_capacity,
                    "replay_capacity": self.replay_capacity,
                    "retrieval_limit": self.retrieval_limit,
                    "action_dim": self.action_dim,
                },
                "episodic": [self._sanitize_for_serialization(exp) for exp in self.episodic],
                "semantic": self._serialize_semantic_store(),
                "parameter_evolution": self._sanitize_for_serialization(self.parameter_evolution),
                "policy_interventions": self._sanitize_for_serialization(self.policy_interventions),
                "concept_drift_scores": self._sanitize_for_serialization(self.concept_drift_scores),
                "reward_sum": float(self.reward_sum),
                "reward_count": int(self.reward_count),
                "max_abs_reward": float(self.max_abs_reward),
                "last_drift_score": self._sanitize_for_serialization(self._last_drift_score),
                "last_drift_detected_at": self._sanitize_for_serialization(self._last_drift_detected_at),
                "experience_counter": int(self._experience_counter),
                "replay_buffer_entries": replay_entries,
            }
            return payload

    def import_state(self, state: Mapping[str, Any], restore_replay_buffer: bool = True) -> None:
        """Restore memory state from a serialized checkpoint payload."""
        ensure_instance(state, Mapping, "state", component="adaptive_memory")

        try:
            with self._state_lock:
                episodic = [self._normalize_experience(exp) for exp in state.get("episodic", [])]
                self.episodic = deque(episodic, maxlen=self.episodic_capacity)

                semantic_data = state.get("semantic", {})
                self.semantic = self._restore_semantic_store(semantic_data if isinstance(semantic_data, Mapping) else {})

                parameter_records = state.get("parameter_evolution", [])
                self.parameter_evolution = pd.DataFrame(parameter_records, columns=self.PARAMETER_COLUMNS)
                if "timestamp" in self.parameter_evolution.columns:
                    self.parameter_evolution["timestamp"] = pd.to_datetime(
                        self.parameter_evolution["timestamp"], errors="coerce"
                    ).dt.to_pydatetime()

                self.policy_interventions = list(state.get("policy_interventions", []))
                self.concept_drift_scores = [float(x) for x in state.get("concept_drift_scores", [])]
                self.reward_sum = float(state.get("reward_sum", 0.0))
                self.reward_count = int(state.get("reward_count", 0))
                self.max_abs_reward = float(state.get("max_abs_reward", 0.0))
                self._last_drift_score = state.get("last_drift_score")
                if self._last_drift_score is not None:
                    self._last_drift_score = float(self._last_drift_score)
                self._last_drift_detected_at = self._restore_datetime(state.get("last_drift_detected_at"))
                self._experience_counter = int(state.get("experience_counter", len(self.episodic)))

                if restore_replay_buffer:
                    self._reset_replay_buffer_state()
                    for item in state.get("replay_buffer_entries", []):
                        try:
                            reward = float(item.get("reward", 0.0))
                            priority = self._calculate_priority(reward) if np.isfinite(reward) else 0.01
                            self.replay_buffer.push(
                                agent_id=str(item.get("agent_id", self.replay_default_agent_id)),
                                state=item.get("state"),
                                action=item.get("action"),
                                reward=reward,
                                next_state=item.get("next_state"),
                                done=bool(item.get("done", False)),
                                priority=priority,
                            )
                        except Exception as exc:
                            if self.allow_partial_replay_failures:
                                logger.warning("Failed to restore replay entry during import_state: %s", exc)
                            else:
                                raise
        except AdaptiveError:
            raise
        except Exception as exc:
            raise MemorySerializationError(
                "Failed to import adaptive memory state.",
                component="adaptive_memory",
                remediation="Inspect the checkpoint payload for incompatible or missing fields.",
                cause=exc,
            ) from exc

    def save(self, filepath: Union[str, Path], include_replay_buffer: bool = True) -> Path:
        """Save adaptive memory state to a checkpoint file."""
        path = Path(filepath)
        payload = self.export_state(include_replay_buffer=include_replay_buffer)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=self.checkpoint_protocol)
            logger.info("Adaptive Memory saved to %s", path)
            return path
        except Exception as exc:
            raise CheckpointSaveError(
                f"Failed to save Adaptive Memory checkpoint to {path}.",
                component="adaptive_memory",
                details={"filepath": str(path)},
                remediation="Ensure the target path is writable and the state payload is serializable.",
                cause=exc,
            ) from exc

    def load(self, filepath: Union[str, Path], restore_replay_buffer: bool = True) -> "MultiModalMemory":
        """Load adaptive memory state from a checkpoint file into the current instance."""
        path = Path(filepath)
        if not path.exists():
            raise CheckpointLoadError(
                f"Adaptive Memory checkpoint does not exist: {path}",
                component="adaptive_memory",
                details={"filepath": str(path)},
            )

        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            self.import_state(payload, restore_replay_buffer=restore_replay_buffer)
            logger.info("Adaptive Memory loaded from %s", path)
            return self
        except AdaptiveError:
            raise
        except Exception as exc:
            raise CheckpointLoadError(
                f"Failed to load Adaptive Memory checkpoint from {path}.",
                component="adaptive_memory",
                details={"filepath": str(path)},
                remediation="Inspect the checkpoint file for corruption or incompatible schema.",
                cause=exc,
            ) from exc


if __name__ == "__main__":
    print("\n=== Running Adaptive Memory ===\n")
    printer.status("TEST", "Adaptive Memory initialized", "info")

    try:
        memory = MultiModalMemory()
        rng = np.random.default_rng(42)

        print("\n* * * * * Phase 1: Parameter Logging * * * * *\n")
        for i in range(80):
            params = {
                "learning_rate": 0.01 * (1 - (i / 120.0)),
                "exploration_rate": 0.30 * (0.97 ** i),
                "discount_factor": 0.95 + (0.01 * np.sin(i / 8.0)),
                "temperature": max(0.5, 1.0 - (i * 0.004)),
            }
            performance = float(0.75 + 0.1 * np.sin(i / 6.0) + rng.normal(0.0, 0.02))
            memory.log_parameters(performance=performance, params=params)

        parameter_impact = memory.analyze_parameter_impact(window_size=40)
        printer.pretty("PARAMETER IMPACT", parameter_impact, "success")

        print("\n* * * * * Phase 2: Experience Storage * * * * *\n")
        for i in range(24):
            state = rng.normal(size=memory.action_dim + 2).astype(np.float32)
            next_state = state + rng.normal(scale=0.05, size=state.shape).astype(np.float32)
            action = int(i % memory.action_dim)
            reward = float(rng.normal(loc=1.0 if i % 3 == 0 else 0.2, scale=0.35))
            context = {
                "state_bucket": int(i % 4),
                "task_type": "control",
                "state_preview": state[: min(3, len(state))],
            }
            params = {
                "learning_rate": 0.005 + (0.0001 * i),
                "exploration_rate": max(0.01, 0.30 - (0.005 * i)),
                "discount_factor": 0.95,
                "temperature": 1.0,
            }
            stored = memory.store_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=bool(i % 5 == 0),
                context=context,
                params=params,
                agent_id=f"agent_{i % 3}",
                episode=i // 4,
            )
            ensure_instance(stored, Mapping, "stored_experience", component="adaptive_memory")

        printer.pretty("SIZE", memory.size(), "success")

        print("\n* * * * * Phase 3: Retrieval and Bias * * * * *\n")
        retrieval_context = {
            "state_bucket": 1,
            "task_type": "control",
            "state_preview": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        }
        memories = memory.retrieve(
            query="increase learning rate and reduce exploration rate",
            context=retrieval_context,
            limit=10,
        )
        bias = memory._generate_memory_bias(memories)
        printer.pretty("RETRIEVED MEMORIES", memories[:3], "success")
        printer.pretty("MEMORY BIAS", bias.tolist(), "success")

        print("\n* * * * * Phase 4: Replay Sampling * * * * *\n")
        sample = memory.sample(batch_size=8)
        printer.pretty(
            "SAMPLE SUMMARY",
            {
                "states_shape": list(sample["states"].shape),
                "actions_shape": list(sample["actions"].shape),
                "rewards_shape": list(sample["rewards"].shape),
                "advantages_shape": list(sample["advantages"].shape),
            },
            "success",
        )

        print("\n* * * * * Phase 5: Intervention Tracking and Consolidation * * * * *\n")
        memory.apply_policy_intervention(
            intervention={
                "type": "parameter_update",
                "params_before": {"learning_rate": 0.01, "exploration_rate": 0.30},
                "params_after": {"learning_rate": 0.008, "exploration_rate": 0.24},
            },
            effect={"performance_delta": 0.12},
        )
        memory.consolidate()
        report = memory.get_memory_report()
        printer.pretty("MEMORY REPORT", report, "success")

        print("\n* * * * * Phase 6: Export / Import / Checkpoint * * * * *\n")
        exported = memory.export_state(include_replay_buffer=True)
        printer.pretty("EXPORTED KEYS", sorted(exported.keys()), "success")

        checkpoint_path = Path("/tmp/adaptive_memory_test.pkl")
        memory.save(checkpoint_path)

        restored = MultiModalMemory()
        restored.load(checkpoint_path)
        restored_report = restored.get_memory_report()
        printer.pretty("RESTORED SIZE", restored.size(), "success")
        printer.pretty(
            "RESTORED REPORT SNAPSHOT",
            {
                "semantic_total": restored_report["semantic_summary"]["total_concepts"],
                "episodic_total": restored_report["sizes"]["episodic"],
                "drift_scores": restored_report["drift_status"]["num_scores"],
            },
            "success",
        )

        print("\n=== Test ran successfully ===\n")

    except Exception as exc:
        logger.exception("Adaptive Memory test run failed")
        raise
