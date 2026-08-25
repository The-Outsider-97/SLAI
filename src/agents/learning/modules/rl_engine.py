"""Production-ready recursive RL utility engine for SLAI learning agents.

This module keeps the original public surface used by ``RLAgent`` while
hardening the three core utilities:
- ``TabularStateProcessor`` for numpy state normalization, feature construction, and tile coding.
- ``ExplorationStrategies`` for discrete-action exploration policies.
- ``QTableOptimizer`` for cached, compact, momentum-aware Q-table updates.
"""

from __future__ import annotations

import math
import pickle
import random
import time
import numpy as np  # type: ignore

from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.learning_error import *
from ..utils.learning_calculations import *
from ..utils.learning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # type: ignore

logger = get_logger("Recursive Learning Engine")
printer = PrettyPrinter()

StateKey = Tuple[Any, ...]
QUpdate = Tuple[Any, Any, float]

__all__ = ["TabularStateProcessor", "ExplorationStrategies", "QTableOptimizer"]


def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite_array(values: Any, *, name: str, expected_size: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ObservationShapeError(expected_shape=(expected_size or "non-empty",), actual_shape=tuple(arr.shape))
    if expected_size is not None and arr.size != int(expected_size):
        raise ObservationShapeError(expected_shape=(int(expected_size),), actual_shape=tuple(arr.shape))
    if np.isnan(arr).any():
        raise NaNException(f"NaN detected in {name}", location=name)
    if np.isinf(arr).any():
        raise InfException(f"Inf detected in {name}", location=name)
    return arr


def _normalise_state_key(state: Any, precision: Optional[int] = None) -> StateKey:
    if isinstance(state, tuple):
        seq = state
    elif isinstance(state, np.ndarray):
        seq = tuple(state.reshape(-1).tolist())
    elif isinstance(state, list):
        seq = tuple(state)
    else:
        seq = (state,)
    if precision is None:
        return tuple(seq)
    normalised = []
    for item in seq:
        if isinstance(item, (float, int, np.floating, np.integer)):
            normalised.append(round(float(item), int(precision)))
        else:
            normalised.append(item)
    return tuple(normalised)


class TabularStateProcessor:
    """Numpy-based RL state processor with validation, normalization, and tile coding."""

    def __init__(self, state_size: int, tiling_resolution: Optional[float] = None, num_tilings: Optional[int] = None,
                 feature_engineering: Optional[bool] = None, low: Optional[Sequence[float]] = None,
                 high: Optional[Sequence[float]] = None) -> None:
        self.config = load_global_config()
        self.engine_config = get_config_section("rl_engine") or {}
        self.processor_config = _as_mapping(self.engine_config.get("state_processor")) or get_config_section("state_processor") or {}
        self.calculations = LearningCalculations()
        self.feature_stats = RunningStats()
        self.processing_stats = RunningStats()

        validate_positive(state_size, "state_size", strict=True)
        self.state_size = int(state_size)
        self.tiling_resolution = coerce_float(
            self.processor_config.get("tiling_resolution", 0.1) if tiling_resolution is None else tiling_resolution,
            default=0.1,
            minimum=1e-12,
        )
        self.num_tilings = coerce_int(
            self.processor_config.get("num_tilings", 8) if num_tilings is None else num_tilings,
            default=8,
            minimum=1,
        )
        self.feature_engineering = coerce_bool(
            self.processor_config.get("feature_engineering", True) if feature_engineering is None else feature_engineering,
            default=True,
        )
        self.normalize_states = coerce_bool(self.processor_config.get("normalize_states", low is not None and high is not None), default=False)
        self.clip_normalized = coerce_bool(self.processor_config.get("clip_normalized", True), default=True)
        self.feature_clip = self.processor_config.get("feature_clip")
        self.rbf_sigma = coerce_float(self.processor_config.get("rbf_sigma", 0.5), default=0.5, minimum=1e-12)
        self.include_cross_terms = coerce_bool(self.processor_config.get("include_cross_terms", True), default=True)
        self.include_rbf = coerce_bool(self.processor_config.get("include_rbf", True), default=True)

        self.low = None if low is None else _finite_array(low, name="state_low", expected_size=self.state_size)
        self.high = None if high is None else _finite_array(high, name="state_high", expected_size=self.state_size)
        if self.low is not None and self.high is not None:
            if np.any(self.high <= self.low):
                raise InvalidConfigError("state high bounds must be greater than low bounds", config_key="rl_engine.state_processor.bounds")
        elif self.normalize_states:
            logger.warning("State normalization requested without low/high bounds; raw states will be used.")
            self.normalize_states = False

        rng = np.random.default_rng(coerce_int(self.processor_config.get("seed", 0), default=0, minimum=0))
        self.feature_weights = rng.normal(0.0, 1.0, self.state_size).astype(np.float32)
        logger.info("RL StateProcessor initialized for state size %s.", self.state_size)

    def _prepare_state(self, state: Any, *, expected_size: Optional[int] = None) -> np.ndarray:
        return _finite_array(state, name="state", expected_size=expected_size or self.state_size)

    def normalize(self, raw_state: Any) -> np.ndarray:
        """Normalize a raw state to roughly [-1, 1] when bounds are available."""
        state = self._prepare_state(raw_state)
        if not self.normalize_states or self.low is None or self.high is None:
            return state.astype(np.float32)
        midpoint = (self.high + self.low) * 0.5
        half_range = np.maximum((self.high - self.low) * 0.5, 1e-12)
        normalised = (state - midpoint) / half_range
        if self.clip_normalized:
            normalised = np.clip(normalised, -1.0, 1.0)
        return normalised.astype(np.float32)

    def discretize(self, continuous_state: np.ndarray, num_tilings: Optional[int] = None) -> tuple:
        """Tile-code a continuous vector into a deterministic tuple of tile indices."""
        state = _finite_array(continuous_state, name="continuous_state")
        tilings = coerce_int(self.num_tilings if num_tilings is None else num_tilings, default=self.num_tilings, minimum=1)
        resolution = max(float(self.tiling_resolution), 1e-12)
        offsets = np.linspace(0.0, resolution, tilings, endpoint=False, dtype=np.float64)
        tile_indices: List[int] = []
        for tiling_idx, offset in enumerate(offsets):
            shifted = state + offset + (tiling_idx / max(tilings, 1)) * resolution
            tile_indices.extend(np.floor(shifted / resolution).astype(np.int64).tolist())
        return tuple(int(idx) for idx in tile_indices)

    def extract_features(self, raw_state: np.ndarray) -> np.ndarray:
        """Construct linear, quadratic, optional cross-term, and optional RBF features."""
        state = _finite_array(raw_state, name="raw_state")
        n = int(state.shape[0])
        features: List[float] = []
        features.extend(state.tolist())
        features.extend(np.square(state).tolist())
        if self.include_cross_terms:
            for i in range(n):
                for j in range(i + 1, n):
                    features.append(float(state[i] * state[j]))
        if self.include_rbf:
            centers = (np.zeros(n), np.ones(n), -np.ones(n))
            denom = 2.0 * self.rbf_sigma * self.rbf_sigma
            for center in centers:
                features.append(float(np.exp(-np.linalg.norm(state - center) ** 2 / denom)))
        out = np.asarray(features, dtype=np.float32)
        if self.feature_clip is not None:
            limit = coerce_float(self.feature_clip, default=0.0, minimum=0.0)
            if limit > 0.0:
                out = np.clip(out, -limit, limit)
        self.feature_stats.update(float(np.linalg.norm(out)))
        return out

    def process(self, raw_state: Any, *, discretize: bool = True, engineer_features: Optional[bool] = None) -> Union[np.ndarray, tuple]:
        """Normalize, optionally engineer features, then optionally discretize."""
        start = time.perf_counter()
        state = self.normalize(raw_state)
        if self.feature_engineering if engineer_features is None else bool(engineer_features):
            state = self.extract_features(state)
        self.processing_stats.update(time.perf_counter() - start)
        return self.discretize(state, self.num_tilings) if discretize else state

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "state_size": self.state_size,
            "tiling_resolution": self.tiling_resolution,
            "num_tilings": self.num_tilings,
            "feature_engineering": self.feature_engineering,
            "normalize_states": self.normalize_states,
            "feature_norm_stats": self.feature_stats.snapshot().to_dict() if hasattr(self.feature_stats.snapshot(), "to_dict") else to_json_safe(self.feature_stats.snapshot()), # type: ignore
            "processing_time_stats": to_json_safe(self.processing_stats.snapshot()),
        }


class ExplorationStrategies:
    """Discrete-action exploration strategies for tabular and approximated RL."""

    VALID_STRATEGIES = {"epsilon_greedy", "greedy", "random", "boltzmann", "softmax", "ucb", "thompson", "curiosity"}

    def __init__(self, action_space: Sequence[Any], strategy: Optional[str] = None,
                 temperature: Optional[float] = None, ucb_c: Optional[float] = None) -> None:
        self.config = load_global_config()
        self.engine_config = get_config_section("rl_engine") or {}
        self.strategies_config = _as_mapping(self.engine_config.get("exploration_strategies")) or get_config_section("exploration_strategies") or {}
        self.calculations = LearningCalculations()
        self.selection_stats = RunningStats()

        validate_non_empty_sequence(action_space, "action_space")
        self.action_space = list(action_space)
        self.possible_actions = self.action_space
        self.strategy = str(strategy or self.strategies_config.get("strategy", "epsilon_greedy")).lower()
        if self.strategy not in self.VALID_STRATEGIES:
            raise UnknownStrategyError(self.strategy, self.VALID_STRATEGIES)
        self.temperature = coerce_float(self.strategies_config.get("temperature", 1.0) if temperature is None else temperature, default=1.0, minimum=1e-12)
        self.ucb_c = coerce_float(self.strategies_config.get("ucb_c", 2.0) if ucb_c is None else ucb_c, default=2.0, minimum=0.0)
        self.epsilon = coerce_float(self.strategies_config.get("epsilon", 0.1), default=0.1, minimum=0.0, maximum=1.0)
        self.epsilon_decay = coerce_float(self.strategies_config.get("epsilon_decay", 0.995), default=0.995, minimum=0.0, maximum=1.0)
        self.min_epsilon = coerce_float(self.strategies_config.get("min_epsilon", 0.01), default=0.01, minimum=0.0, maximum=1.0)
        self.curiosity_weight = coerce_float(self.strategies_config.get("curiosity_weight", 0.1), default=0.1, minimum=0.0)
        self.seed = self.strategies_config.get("seed")
        self.rng = random.Random(None if self.seed is None else int(self.seed))
        self.state_history: List[tuple] = []
        self.action_history: Deque[Any] = deque(maxlen=coerce_int(self.strategies_config.get("history_size", 1000), default=1000, minimum=1))
        self.q_table: Dict[Tuple[tuple, Any], float] = {}
        self.selection_count = 0

    def _coerce_q_values(self, q_values: Union[Sequence[float], Mapping[Any, float]]) -> np.ndarray:
        if isinstance(q_values, Mapping):
            values = [float(q_values.get(action, 0.0)) for action in self.action_space]
        else:
            values = [float(v) for v in q_values]
        if len(values) != len(self.action_space):
            raise StrategySelectionError(
                "q_values length must match action_space",
                context={"q_values": len(values), "actions": len(self.action_space)},
            )
        arr = np.asarray(values, dtype=np.float64)
        if np.isnan(arr).any():
            raise NaNException("NaN detected in q_values", location="q_values")
        if np.isinf(arr).any():
            raise InfException("Inf detected in q_values", location="q_values")
        return arr

    def _choose_by_index(self, index: int) -> Any:
        action = self.action_space[int(index)]
        self.action_history.append(action)
        self.selection_count += 1
        return action

    def greedy(self, q_values: Union[Sequence[float], Mapping[Any, float]]) -> Any:
        values = self._coerce_q_values(q_values)
        return self._choose_by_index(argmax(values.tolist()))

    def epsilon_greedy(self, q_values: Union[Sequence[float], Mapping[Any, float]], epsilon: Optional[float] = None) -> Any:
        eps = clamp01(self.epsilon if epsilon is None else float(epsilon))
        if self.rng.random() < eps:
            return self.random_action()
        return self.greedy(q_values)

    def random_action(self) -> Any:
        return self._choose_by_index(self.rng.randrange(len(self.action_space)))

    def boltzmann(self, q_values: Union[Sequence[float], Mapping[Any, float]]) -> Any:
        values = self._coerce_q_values(q_values)
        shifted = values - float(np.max(values))
        logits = shifted / max(float(self.temperature), 1e-12)
        probs = np.exp(np.clip(logits, -700.0, 700.0))
        probabilities = normalize_probabilities(probs.tolist())
        if not probabilities:
            raise StrategySelectionError("empty probability vector")
        uniform = np.full(len(probabilities), 1.0 / len(probabilities), dtype=np.float64)
        self.selection_stats.update(self.calculations.calculate_js_divergence(np.asarray(probabilities), uniform))
        return self.rng.choices(self.action_space, weights=probabilities, k=1)[0]

    def ucb(self, state_action_counts: Dict[Tuple[tuple, Any], int], c: Optional[float] = None) -> Any:
        state = self.state_history[-1] if self.state_history else tuple()
        exploration_c = self.ucb_c if c is None else float(c)
        total_state_visits = sum(max(0, int(count)) for (s, _), count in state_action_counts.items() if s == state)
        if total_state_visits <= 0:
            return self.random_action()
        scores: List[float] = []
        for action in self.action_space:
            action_count = max(1, int(state_action_counts.get((state, action), 0)))
            q_value = self._get_q_value(state, action)
            bonus = exploration_c * math.sqrt(max(0.0, math.log(total_state_visits + 1.0)) / action_count)
            scores.append(q_value + bonus)
        return self._choose_by_index(argmax(scores))

    def thompson(self, q_values: Union[Sequence[float], Mapping[Any, float]], uncertainty: Optional[Sequence[float]] = None) -> Any:
        values = self._coerce_q_values(q_values)
        if uncertainty is None:
            std = np.ones_like(values)
        else:
            std = np.maximum(_finite_array(uncertainty, name="thompson_uncertainty", expected_size=len(self.action_space)), 1e-12)
        samples = np.asarray([self.rng.gauss(float(mu), float(sig)) for mu, sig in zip(values, std)], dtype=np.float64)
        return self._choose_by_index(argmax(samples.tolist()))

    def curiosity_driven(self, q_values: Union[Sequence[float], Mapping[Any, float]], novelty_scores: Optional[Sequence[float]] = None) -> Any:
        values = self._coerce_q_values(q_values)
        if novelty_scores is None:
            novelty = np.zeros_like(values)
        else:
            novelty = _finite_array(novelty_scores, name="novelty_scores", expected_size=len(self.action_space))
        adjusted = values + self.curiosity_weight * novelty
        return self._choose_by_index(argmax(adjusted.tolist()))

    def select_action(
        self,
        q_values: Union[Sequence[float], Mapping[Any, float]],
        state: Optional[Any] = None,
        strategy: Optional[str] = None,
        epsilon: Optional[float] = None,
        state_action_counts: Optional[Dict[Tuple[tuple, Any], int]] = None,
        novelty_scores: Optional[Sequence[float]] = None,
        uncertainty: Optional[Sequence[float]] = None,
    ) -> Any:
        if state is not None:
            self.state_history.append(_normalise_state_key(state))
        chosen = str(strategy or self.strategy).lower()
        if chosen in {"greedy"}:
            action = self.greedy(q_values)
        elif chosen == "epsilon_greedy":
            action = self.epsilon_greedy(q_values, epsilon=epsilon)
        elif chosen == "random":
            action = self.random_action()
        elif chosen in {"boltzmann", "softmax"}:
            action = self.boltzmann(q_values)
        elif chosen == "ucb":
            action = self.ucb(state_action_counts or {}, c=self.ucb_c)
        elif chosen == "thompson":
            action = self.thompson(q_values, uncertainty=uncertainty)
        elif chosen == "curiosity":
            action = self.curiosity_driven(q_values, novelty_scores=novelty_scores)
        else:
            raise UnknownStrategyError(chosen, self.VALID_STRATEGIES)
        return action

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def update_q_table(self, q_table: Mapping[Tuple[tuple, Any], float]) -> None:
        self.q_table = {key: float(value) for key, value in q_table.items()}

    def _get_q_value(self, state: Any, action: Any) -> float:
        key = (_normalise_state_key(state), action)
        return float(self.q_table.get(key, 0.0))

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "actions": len(self.action_space),
            "epsilon": round_float(self.epsilon),
            "temperature": round_float(self.temperature),
            "ucb_c": round_float(self.ucb_c),
            "selection_count": self.selection_count,
            "recent_actions": to_json_safe(list(self.action_history)[-10:]),
            "probability_drift_stats": to_json_safe(self.selection_stats.snapshot()),
        }


class QTableOptimizer:
    """Sparse/cached Q-table storage with momentum batch updates and diagnostics."""

    def __init__(
        self,
        batch_size: Optional[int] = None,
        momentum: Optional[float] = None,
        cache_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        default_value: Optional[float] = None,
    ) -> None:
        self.config = load_global_config()
        self.engine_config = get_config_section("rl_engine") or {}
        self.optimizer_config = _as_mapping(self.engine_config.get("q_table_optimizer")) or get_config_section("q_table_optimizer") or {}
        self.calculations = LearningCalculations()
        self.update_stats = RunningStats()
        self.reward_stats = RunningStats()

        self.batch_size = coerce_int(self.optimizer_config.get("batch_size", 32) if batch_size is None else batch_size, default=32, minimum=1)
        self.momentum = coerce_float(self.optimizer_config.get("momentum", 0.9) if momentum is None else momentum, default=0.9, minimum=0.0, maximum=0.999999)
        self.cache_size = coerce_int(self.optimizer_config.get("cache_size", 1000) if cache_size is None else cache_size, default=1000, minimum=1)
        self.learning_rate = coerce_float(self.optimizer_config.get("learning_rate", 0.1) if learning_rate is None else learning_rate, default=0.1, minimum=0.0)
        self.compression = coerce_bool(self.optimizer_config.get("compression", True), default=True)
        self.compaction_interval = coerce_int(self.optimizer_config.get("compaction_interval", 1000), default=1000, minimum=1)
        self.delta_precision = coerce_int(self.optimizer_config.get("delta_precision", 6), default=6, minimum=0)
        self.max_sparse_entries = coerce_int(self.optimizer_config.get("max_sparse_entries", 100000), default=100000, minimum=1)
        self.DEFAULT_VALUE = coerce_float(self.optimizer_config.get("default_value", 0.0) if default_value is None else default_value, default=0.0)

        self.state_index: Dict[StateKey, int] = defaultdict(int)
        self.action_index: Dict[Hashable, int] = defaultdict(int)
        self.next_state_idx = 0
        self.next_action_idx = 0
        self.state_action_matrix: Dict[StateKey, Dict[Any, float]] = defaultdict(dict)
        self.lru_cache: "OrderedDict[Tuple[StateKey, Any], float]" = OrderedDict()
        self.sparse_matrix: List[Tuple[int, int, Union[int, float]]] = []
        self.update_momentum: Dict[Tuple[StateKey, Any], float] = defaultdict(float)
        self.huffman_codes: Dict[float, int] = {}
        self.inverse_huffman: Dict[int, float] = {}
        self.code_counter = 0
        self.total_updates = 0

    def _state_to_index(self, state: Any) -> int:
        key = _normalise_state_key(state, precision=self.delta_precision)
        if key not in self.state_index:
            self.state_index[key] = self.next_state_idx
            self.next_state_idx += 1
        return self.state_index[key]

    def _action_to_index(self, action: Any) -> int:
        if action not in self.action_index:
            self.action_index[action] = self.next_action_idx
            self.next_action_idx += 1
        return self.action_index[action]

    def compressed_store(self, state: tuple, action: Any, value: float) -> None:
        """Store an update in compressed sparse form and mirror it to the live Q-table."""
        if not isinstance(state, tuple):
            raise ObservationShapeError(expected_shape="tuple", actual_shape=type(state).__name__)
        validate_finite(value, "q_value")
        stored_value = float(value)
        self._set_q_value(state, action, stored_value)
        if not self.compression:
            return
        delta = round(stored_value - self.DEFAULT_VALUE, self.delta_precision)
        if abs(delta) <= 10 ** (-self.delta_precision if self.delta_precision > 0 else 0):
            return
        encoded = self.huffman_codes.get(delta, delta)
        self.sparse_matrix.append((self._state_to_index(state), self._action_to_index(action), encoded))
        if len(self.sparse_matrix) % self.compaction_interval == 0 or len(self.sparse_matrix) > self.max_sparse_entries:
            self._compact_storage()

    def _compact_storage(self) -> None:
        decoded_buffer: List[Tuple[int, int, float]] = []
        for s_idx, a_idx, encoded in self.sparse_matrix:
            if isinstance(encoded, int) and encoded in self.inverse_huffman:
                delta = float(self.inverse_huffman[encoded])
            else:
                delta = float(encoded)
            if math.isfinite(delta):
                decoded_buffer.append((int(s_idx), int(a_idx), delta))
        latest_updates: Dict[Tuple[int, int], float] = defaultdict(float)
        for s_idx, a_idx, delta in decoded_buffer:
            latest_updates[(s_idx, a_idx)] += delta
        final_updates = {key: round(value, self.delta_precision) for key, value in latest_updates.items() if not math.isclose(value, 0.0)}
        delta_freq: Dict[float, int] = defaultdict(int)
        for delta in final_updates.values():
            delta_freq[delta] += 1
        self.huffman_codes.clear()
        self.inverse_huffman.clear()
        for code, (delta, _) in enumerate(sorted(delta_freq.items(), key=lambda item: -item[1])[:100]):
            self.huffman_codes[delta] = code
            self.inverse_huffman[code] = delta
        self.sparse_matrix = []
        for (s_idx, a_idx), delta in final_updates.items():
            self.sparse_matrix.append((s_idx, a_idx, self.huffman_codes.get(delta, delta)))
        self._prune_mappings()

    def _prune_mappings(self) -> None:
        used_states = {s_idx for s_idx, _, _ in self.sparse_matrix}
        used_actions = {a_idx for _, a_idx, _ in self.sparse_matrix}
        if used_states:
            self.state_index = defaultdict(int, {state: idx for state, idx in self.state_index.items() if idx in used_states})
        if used_actions:
            self.action_index = defaultdict(int, {action: idx for action, idx in self.action_index.items() if idx in used_actions})

    def batch_update(
        self,
        updates: List[QUpdate],
        batch_size: Optional[int] = None,
        momentum: Optional[float] = None,
    ) -> Dict[str, float]:
        """Apply Q-value deltas in mini-batches with momentum smoothing."""
        if not updates:
            return {"updated": 0.0, "mean_delta": 0.0, "max_delta": 0.0}
        current_batch_size = coerce_int(self.batch_size if batch_size is None else batch_size, default=self.batch_size, minimum=1)
        current_momentum = coerce_float(self.momentum if momentum is None else momentum, default=self.momentum, minimum=0.0, maximum=0.999999)
        applied: List[float] = []
        for batch in chunked(updates, current_batch_size):
            batch_delta: Dict[Tuple[StateKey, Any], float] = defaultdict(float)
            for state, action, delta in batch:
                validate_finite(delta, "q_delta")
                key = (_normalise_state_key(state, precision=self.delta_precision), action)
                batch_delta[key] += float(delta) / max(len(batch), 1)
            for (state, action), delta in batch_delta.items():
                previous = float(self.update_momentum.get((state, action), 0.0))
                smoothed_delta = (1.0 - current_momentum) * float(delta) + current_momentum * previous
                current_q = self._get_q_value(state, action)
                new_value = current_q + self.learning_rate * smoothed_delta
                self._set_q_value(state, action, new_value)
                self.update_momentum[(state, action)] = smoothed_delta
                self.update_stats.update(abs(smoothed_delta))
                self.reward_stats.update(new_value)
                applied.append(smoothed_delta)
        self.total_updates += len(applied)
        summary = self.calculations.summarize_rewards(applied)
        return {"updated": float(len(applied)), "mean_delta": summary["mean"], "max_delta": summary["max"]}

    def update(self, state: Any, action: Any, target: float, alpha: Optional[float] = None) -> float:
        """Move one Q-value toward a target and return the updated value."""
        validate_finite(target, "q_target")
        lr = self.learning_rate if alpha is None else coerce_float(alpha, default=self.learning_rate, minimum=0.0)
        current = self._get_q_value(state, action)
        new_value = current + lr * (float(target) - current)
        self._set_q_value(state, action, new_value)
        self.total_updates += 1
        self.update_stats.update(abs(new_value - current))
        return new_value

    def best_action(self, state: Any, actions: Sequence[Any]) -> Any:
        validate_non_empty_sequence(actions, "actions")
        values = [self._get_q_value(state, action) for action in actions]
        return list(actions)[argmax(values)]

    def bulk_get(self, state: Any, actions: Sequence[Any]) -> Dict[Any, float]:
        return {action: self._get_q_value(state, action) for action in actions}

    def _get_q_value(self, state: Any, action: Any) -> float:
        key = (_normalise_state_key(state, precision=self.delta_precision), action)
        if key in self.lru_cache:
            value = self.lru_cache.pop(key)
            self.lru_cache[key] = value
            return float(value)
        value = float(self.state_action_matrix.get(key[0], {}).get(action, self.DEFAULT_VALUE))
        self.lru_cache[key] = value
        self._enforce_cache_limit()
        return value

    def _set_q_value(self, state: Any, action: Any, value: float) -> None:
        validate_finite(value, "q_value")
        key = _normalise_state_key(state, precision=self.delta_precision)
        val = float(value)
        self.state_action_matrix[key][action] = val
        cache_key = (key, action)
        if cache_key in self.lru_cache:
            self.lru_cache.pop(cache_key)
        self.lru_cache[cache_key] = val
        self._enforce_cache_limit()

    def _enforce_cache_limit(self) -> None:
        while len(self.lru_cache) > self.cache_size:
            self.lru_cache.popitem(last=False)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "momentum": self.momentum,
            "cache_size": self.cache_size,
            "learning_rate": self.learning_rate,
            "state_index": dict(self.state_index),
            "action_index": dict(self.action_index),
            "next_state_idx": self.next_state_idx,
            "next_action_idx": self.next_action_idx,
            "state_action_matrix": {state: dict(actions) for state, actions in self.state_action_matrix.items()},
            "lru_cache": list(self.lru_cache.items()),
            "sparse_matrix": list(self.sparse_matrix),
            "update_momentum": dict(self.update_momentum),
            "default_value": self.DEFAULT_VALUE,
            "huffman_codes": dict(self.huffman_codes),
            "inverse_huffman": dict(self.inverse_huffman),
            "code_counter": self.code_counter,
            "total_updates": self.total_updates,
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        validate_required_keys(state, ["state_action_matrix"], name="q_optimizer_snapshot")
        self.batch_size = coerce_int(state.get("batch_size", self.batch_size), default=self.batch_size, minimum=1)
        self.momentum = coerce_float(state.get("momentum", self.momentum), default=self.momentum, minimum=0.0, maximum=0.999999)
        self.cache_size = coerce_int(state.get("cache_size", self.cache_size), default=self.cache_size, minimum=1)
        self.learning_rate = coerce_float(state.get("learning_rate", self.learning_rate), default=self.learning_rate, minimum=0.0)
        self.state_index = defaultdict(int, {_normalise_state_key(k): int(v) for k, v in dict(state.get("state_index", {})).items()})
        self.action_index = defaultdict(int, dict(state.get("action_index", {})))
        self.next_state_idx = int(state.get("next_state_idx", 0))
        self.next_action_idx = int(state.get("next_action_idx", 0))
        self.state_action_matrix = defaultdict(dict, {_normalise_state_key(k): dict(v) for k, v in dict(state.get("state_action_matrix", {})).items()})
        self.lru_cache = OrderedDict()
        for cache_key, value in state.get("lru_cache", []):
            if isinstance(cache_key, tuple) and len(cache_key) == 2:
                self.lru_cache[(_normalise_state_key(cache_key[0]), cache_key[1])] = float(value)
        self.sparse_matrix = list(state.get("sparse_matrix", []))
        self.update_momentum = defaultdict(float, {(_normalise_state_key(k[0]), k[1]): float(v) for k, v in dict(state.get("update_momentum", {})).items()})
        self.DEFAULT_VALUE = float(state.get("default_value", self.DEFAULT_VALUE))
        self.huffman_codes = dict(state.get("huffman_codes", {}))
        self.inverse_huffman = dict(state.get("inverse_huffman", {}))
        self.code_counter = int(state.get("code_counter", 0))
        self.total_updates = int(state.get("total_updates", 0))
        self._enforce_cache_limit()

    def save(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as fh:
                pickle.dump(self.snapshot(), fh)
            return target
        except Exception as exc:
            raise CheckpointError(str(target), operation="save", cause=exc) from exc

    def load(self, path: Union[str, Path]) -> None:
        source = Path(path)
        try:
            with source.open("rb") as fh:
                payload = pickle.load(fh)
            if not isinstance(payload, Mapping):
                raise CheckpointError(str(source), operation="load", message="QTableOptimizer checkpoint is not a mapping")
            self.restore(payload)
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(str(source), operation="load", cause=exc) from exc

    def diagnostics(self) -> Dict[str, Any]:
        values = [value for action_values in self.state_action_matrix.values() for value in action_values.values()]
        rewards_summary = self.calculations.summarize_rewards(values)
        return {
            "states": len(self.state_action_matrix),
            "cached_values": len(self.lru_cache),
            "sparse_entries": len(self.sparse_matrix),
            "total_updates": self.total_updates,
            "batch_size": self.batch_size,
            "momentum": round_float(self.momentum),
            "learning_rate": round_float(self.learning_rate),
            "value_summary": rewards_summary,
            "update_stats": to_json_safe(self.update_stats.snapshot()),
        }


if __name__ == "__main__":
    print("\n=== Running Recursive Learning Engine ===\n")
    printer.status("TEST", "Recursive Learning Engine initialized", "info")
    random.seed(7); np.random.seed(7)
    p = TabularStateProcessor(state_size=4, low=[-2]*4, high=[2]*4)
    state = np.array([0.2, -0.4, 1.1, 0.0], dtype=np.float32)
    features = p.process(state, discretize=False)
    tiles = p.discretize(features)
    assert isinstance(tiles, tuple) and len(tiles) > 0
    exp = ExplorationStrategies([0, 1, 2], strategy="boltzmann", temperature=0.8)
    action = exp.select_action([1.0, 0.5, -0.2], state=tiles)
    assert action in [0, 1, 2]
    opt = QTableOptimizer(batch_size=2, momentum=0.5, cache_size=8, learning_rate=0.2)
    opt.compressed_store(tiles, action, 1.25)
    summary = opt.batch_update([(tiles, 0, 0.5), (tiles, 1, -0.2), (tiles, 2, 0.1)])
    assert summary["updated"] > 0 and opt.best_action(tiles, [0, 1, 2]) in [0, 1, 2]
    ckpt = Path("rl_engine_qtable_test.pkl")
    opt.save(ckpt)
    restored = QTableOptimizer(batch_size=2, cache_size=8)
    restored.load(ckpt)
    assert restored._get_q_value(tiles, action) == opt._get_q_value(tiles, action)
    ckpt.unlink(missing_ok=True)
    printer.status("TEST", "Processor, exploration, and Q-table verified", "success")
    print("\n=== Test ran successfully ===\n")
