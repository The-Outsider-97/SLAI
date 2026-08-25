"""Production-ready SLAI learning environment.

This module provides the Gymnasium-compatible learning environment used by the
SLAI learning stack. It keeps the original responsibilities intact:
- optional wrapping of an external Gym/Gymnasium environment
- internal physics-based fallback dynamics
- reward shaping and terminal zones
- novelty bonus integration through ``NoveltyDetector``
- transition storage through ``LearningMemory``

The implementation is hardened for production use while retaining the existing
configuration flow through ``learning_config.yaml`` and ``get_config_section``.
"""

from __future__ import annotations

import cv2 # type: ignore
import math
import time
import torch # type: ignore
import numpy as np # type: ignore
import gymnasium as gym # type: ignore
from gymnasium.envs.registration import EnvSpec # type: ignore

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..base.modules.physics_constraints import PhysicsEngine
from .utils.config_loader import load_global_config, get_config_section
from .utils.learning_error import *
from .utils.learning_calculations import *
from .utils.learning_helpers import *
from .modules.policy_network import NoveltyDetector
from .learning_memory import LearningMemory, Transition
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Learning Environment")
printer = PrettyPrinter()

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]


class SLAIEnv(gym.Env):
    """Physics-aware Gymnasium environment with novelty and replay integration."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
        "progress": 10.0,
        "action_cost": -0.02,
        "stability": -0.1,
        "smoothness": -0.01,
        "goal": 100.0,
        "hazard": -10.0,
        "alive": 0.0,
        "boundary": -1.0,
    }

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        env: Optional[gym.Env] = None,
        max_steps: int = 500,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.config = load_global_config()
        if config is not None:
            self.env_config = config
        else:
            self.env_config = get_config_section("learning_env") or {}
        if not self.env_config:
            self.env_config = {}

        self.calculations = LearningCalculations()
        self.reward_stats = RunningStats()
        self.novelty_stats = RunningStats()
        self.step_time_stats = RunningStats()
        self.state_norm_stats = RunningStats()
        self.episode_returns: Deque[float] = deque(
            maxlen=coerce_int(self.env_config.get("episode_history", 100), default=100, minimum=1)
        )

        validate_positive(state_dim, "state_dim", strict=True)
        validate_positive(action_dim, "action_dim", strict=True)
        validate_positive(max_steps, "max_steps", strict=True)

        self.env = env
        self.render_mode = render_mode
        self.spec = EnvSpec(id="SLAIEnv-v0", entry_point=None, reward_threshold=200)
        self.max_steps = int(self.env_config.get("max_steps", max_steps))
        validate_positive(self.max_steps, "learning_env.max_steps", strict=True)

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.seed_value = self.env_config.get("seed")
        self._rng = np.random.default_rng(None if self.seed_value in (None, "", "none", "None") else int(self.seed_value))

        self.observation_space = self._create_observation_space(self.state_dim)
        self.action_space = self._create_action_space(self.action_dim)
        self.state_dim = self._infer_state_dim(self.observation_space, fallback=self.state_dim)
        self.action_dim = self._infer_action_dim(self.action_space, fallback=self.action_dim)

        self.memory_enabled = coerce_bool(self.env_config.get("memory_enabled", True), default=True)
        self.memory_priority_mode = str(self.env_config.get("memory_priority_mode", "abs_reward")).lower()
        self.train_novelty_detector = coerce_bool(self.env_config.get("train_novelty_detector", False), default=False)
        self.novelty_weight = coerce_float(self.env_config.get("novelty_weight", 1.0), default=1.0, minimum=0.0)
        self.novelty_clip = self.env_config.get("novelty_clip")
        self.strict_novelty_errors = coerce_bool(self.env_config.get("strict_novelty_errors", False), default=False)
        self.auto_reset_on_step = coerce_bool(self.env_config.get("auto_reset_on_step", False), default=False)
        self.quantum_jitter_probability = coerce_float(self.env_config.get("quantum_jitter_probability", 0.01), default=0.01, minimum=0.0, maximum=1.0)
        self.reward_clip = self.env_config.get("reward_clip")
        self.initial_state_low = coerce_float(self.env_config.get("initial_state_low", -0.5), default=-0.5)
        self.initial_state_high = coerce_float(self.env_config.get("initial_state_high", 0.5), default=0.5)

        if self.initial_state_high <= self.initial_state_low:
            raise InvalidConfigError(
                "learning_env.initial_state_high must be greater than initial_state_low",
                config_key="learning_env.initial_state_range",
                context={"low": self.initial_state_low, "high": self.initial_state_high},
            )

        self.learning_memory = LearningMemory()
        self.novelty_detector = NoveltyDetector(self.state_dim)
        self._current_state: Optional[np.ndarray] = None
        self._step_count = 0
        self._episode_count = 0
        self._last_action: Optional[int] = None
        self._episode_reward = 0.0
        self._last_info: Dict[str, Any] = {}
        self._task_context: Dict[str, Any] = {}

        self.physics = PhysicsEngine(self._build_physics_config())
        self._sync_physics_shortcuts()

        self.zones = self._load_zones()
        self.A = self._create_state_transition_matrix()
        self.B = self._create_action_effect_matrix()
        self.reward_weights = self._load_reward_weights()

        logger.info(
            "SLAIEnv initialized | state_dim=%s action_dim=%s max_steps=%s wrapped_env=%s",
            self.state_dim,
            self.action_dim,
            self.max_steps,
            self.env is not None,
        )

    # ------------------------------------------------------------------
    # Configuration / setup
    # ------------------------------------------------------------------
    def _sync_physics_shortcuts(self) -> None:
        self.dt = float(self.physics.dt)
        self.gravity = float(self.physics.gravity)
        self.friction_coeff = float(self.physics.friction_coeff)
        self.elasticity = float(self.physics.elasticity)
        self.wind_strength = float(self.physics.wind_strength)
        self.wind_direction = float(self.physics.wind_direction)
        self.drag_coeff = float(self.physics.drag_coeff)
        self.terminal_velocity = float(self.physics.terminal_velocity)
        self.rotational_friction = float(self.physics.rotational_friction)
        self.mass = float(self.physics.default_mass)

    def _build_physics_config(self) -> Dict[str, Any]:
        cfg = self.env_config.get("physics", {})
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, Mapping):
            raise InvalidConfigError("learning_env.physics must be a mapping", config_key="learning_env.physics")
    
        def get_float(name: str, default: float, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
            return coerce_float(cfg.get(name, default), default=default, minimum=minimum, maximum=maximum)
    
        # --- electric_field pre-processing ---
        raw_efield = ensure_list(cfg.get("electric_field", [0.0, 10.0]))
        if len(raw_efield) < 2:
            raw_efield = raw_efield + [0.0] * (2 - len(raw_efield))
        elif len(raw_efield) > 2:
            raw_efield = raw_efield[:2]
        electric_field = (coerce_float(raw_efield[0], default=0.0),
                          coerce_float(raw_efield[1], default=0.0))
    
        return {
            "gravity": get_float("gravity", 9.80665, minimum=0.0),
            "friction_coeff": get_float("friction_coeff", 0.02, minimum=0.0),
            "rotational_friction": get_float("rotational_friction", 0.01, minimum=0.0),
            "wind_strength": get_float("wind_strength", 0.0, minimum=0.0),
            "wind_direction": get_float("wind_direction", 0.0),
            "wind_turbulence_ratio": get_float("wind_turbulence_ratio", 0.1, minimum=0.0),
            "drag_coeff": get_float("drag_coeff", 0.01, minimum=0.0),
            "terminal_velocity": get_float("terminal_velocity", 50.0, minimum=0.0),
            "min_speed_for_drag": get_float("min_speed_for_drag", 0.01, minimum=0.0),
            "elasticity": get_float("elasticity", 0.8, minimum=0.0, maximum=1.0),
            "tangential_damping": get_float("tangential_damping", 0.2, minimum=0.0, maximum=1.0),
            "boundary_margin": get_float("boundary_margin", 0.01, minimum=0.0),
            "corner_threshold": get_float("corner_threshold", 0.05, minimum=0.0),
            "max_angular_velocity": get_float("max_angular_velocity", 5.0, minimum=0.0),
            "default_mass": get_float("default_mass", 1.0, minimum=1.0e-12),
            "default_charge": get_float("default_charge", 0.0),
            "enable_tunneling": coerce_bool(cfg.get("enable_tunneling", False), default=False),
            "tunneling_probability": get_float("tunneling_probability", 0.05, minimum=0.0, maximum=1.0),
            "barrier_positions": tuple(
                coerce_float(v, default=0.0)
                for v in ensure_list(cfg.get("barrier_positions", [-8.0, 8.0]))
            ),
            "barrier_width": get_float("barrier_width", 0.1, minimum=0.0),
            "enable_relativistic": coerce_bool(cfg.get("enable_relativistic", True), default=True),
            "relativistic_threshold": get_float("relativistic_threshold", 0.1, minimum=0.0, maximum=1.0),
            "relativistic_safety_factor": get_float(
                "relativistic_safety_factor",
                0.999999,
                minimum=0.0,
                maximum=1.0,
            ),
            "enable_electromagnetic": coerce_bool(cfg.get("enable_electromagnetic", False), default=False),
            "electric_field": electric_field,
            "magnetic_field": get_float("magnetic_field", 0.5),
            "dt": get_float("dt", 0.05, minimum=1.0e-12),
            "enable_history": coerce_bool(cfg.get("enable_history", True), default=True),
            "history_limit": coerce_int(cfg.get("history_limit", 200), default=200, minimum=1),
            "random_seed": (
                None
                if cfg.get("random_seed") in (None, "", "none", "None")
                else coerce_int(cfg.get("random_seed"), default=0)
            ),
        }

    def _load_reward_weights(self) -> Dict[str, float]:
        raw = dict(self.DEFAULT_REWARD_WEIGHTS)
        cfg = self.env_config.get("reward_weights", {})
        if cfg is not None and not isinstance(cfg, Mapping):
            raise InvalidConfigError("learning_env.reward_weights must be a mapping")
        raw.update(dict(cfg or {}))
        weights = {key: coerce_float(value, default=self.DEFAULT_REWARD_WEIGHTS.get(key, 0.0)) for key, value in raw.items()}
        return weights

    def _load_zones(self) -> List[Dict[str, Any]]:
        zones = self.env_config.get("zones") or [
            {"type": "goal", "bounds": [8.0, 9.0, 8.0, 9.0], "reward": 100.0, "terminal": True},
            {"type": "hazard", "bounds": [-5.0, -4.0, -5.0, -4.0], "reward": -10.0, "terminal": True},
        ]
        if not isinstance(zones, Sequence) or isinstance(zones, (str, bytes)):
            raise InvalidConfigError("learning_env.zones must be a sequence of mappings")
        validated: List[Dict[str, Any]] = []
        for idx, zone in enumerate(zones):
            if not isinstance(zone, Mapping):
                raise InvalidConfigError("Each zone must be a mapping", context={"zone_index": idx})
            bounds = ensure_list(zone.get("bounds"))
            if len(bounds) != 4:
                raise InvalidConfigError("Zone bounds must contain [xmin, xmax, ymin, ymax]", context={"zone_index": idx})
            xmin, xmax, ymin, ymax = [coerce_float(v, default=0.0) for v in bounds]
            if xmax < xmin or ymax < ymin:
                raise InvalidConfigError("Zone bounds are invalid", context={"zone_index": idx, "bounds": bounds})
            validated.append({
                "type": str(zone.get("type", "zone")),
                "bounds": [xmin, xmax, ymin, ymax],
                "reward": coerce_float(zone.get("reward", 0.0), default=0.0),
                "terminal": coerce_bool(zone.get("terminal", False), default=False),
            })
        return validated

    def _create_observation_space(self, state_dim: int) -> gym.Space:
        if self.env is not None:
            return self.env.observation_space
        low = coerce_float(self.env_config.get("state_low", -10.0), default=-10.0)
        high = coerce_float(self.env_config.get("state_high", 10.0), default=10.0)
        if high <= low:
            raise InvalidConfigError("learning_env.state_high must be greater than state_low")
        return gym.spaces.Box(low=low, high=high, shape=(int(state_dim),), dtype=np.float32)

    def _create_action_space(self, action_dim: int) -> gym.Space:
        if self.env is not None:
            return self.env.action_space
        return gym.spaces.Discrete(int(action_dim))

    @staticmethod
    def _infer_state_dim(space: Any, fallback: int) -> int:
        shape = getattr(space, "shape", None)
        if shape:
            size = int(np.prod(shape))
            return size if size > 0 else int(fallback)
        return int(fallback)

    @staticmethod
    def _infer_action_dim(space: Any, fallback: int) -> int:
        n = getattr(space, "n", None)
        if n is not None:
            return int(n)
        shape = getattr(space, "shape", None)
        if shape:
            size = int(np.prod(shape))
            return size if size > 0 else int(fallback)
        return int(fallback)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            super().reset(seed=seed)
            if seed is not None:
                self._rng = np.random.default_rng(int(seed))

            if self.env is not None:
                reset_output = self.env.reset(seed=seed, options=options)
                state, info = self._normalise_reset_output(reset_output)
            else:
                state = self._rng.uniform(self.initial_state_low, self.initial_state_high, self.state_dim).astype(np.float32)
                info = {
                    "episode": self._episode_count + 1,
                    "initial_state": state.tolist(),
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "max_steps": self.max_steps,
                    "reset_seed": seed,
                    "using_wrapped_env": False,
                }

            state = self._coerce_state(state, expected_dim=None if self.env is not None else self.state_dim)
            self._ensure_novelty_dim(len(state))
            self._current_state = state
            self._step_count = 0
            self._episode_count += 1
            self._last_action = None
            self._episode_reward = 0.0
            self.state_norm_stats.update(float(np.linalg.norm(state)))
            info = dict(info or {})
            info.update({"episode": self._episode_count, "state_norm": float(np.linalg.norm(state))})
            self._last_info = info
            return state, info
        except LearningError:
            raise
        except Exception as exc:
            raise EnvironmentResetError(env_name=type(self.env).__name__ if self.env is not None else "SLAIEnv", cause=exc) from exc

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        start = time.perf_counter()
        if self._current_state is None:
            if self.auto_reset_on_step:
                self.reset()
            else:
                raise EnvironmentStepError("step() called before reset()", action=action)
        
        # After reset, _current_state is guaranteed to be an ndarray.
        assert self._current_state is not None
        curr_state: np.ndarray = self._current_state
        previous_state = curr_state.copy()
        previous_action = self._last_action
        action_value = self._validate_action(action)

        try:
            self._step_count += 1
            if self.env is not None:
                raw_step = self.env.step(action_value)
                next_state, raw_reward, terminated, truncated, info = self._normalise_step_output(raw_step, action_value)
                next_state = self._coerce_state(next_state)
            else:
                next_state = self._simulate_dynamics(previous_state, action_value)
                raw_reward = self._calculate_reward(previous_state, action_value, next_state, previous_action)
                terminated = self._check_termination(next_state)
                truncated = self._step_count >= self.max_steps
                info = {
                    "step_count": self._step_count,
                    "episode": self._episode_count,
                    "terminated_due_to": "step_limit" if truncated else ("zone" if terminated else None),
                    "state": next_state.tolist(),
                    "action_taken": action_value,
                    "using_wrapped_env": False,
                }

            novelty_bonus = self._compute_novelty_bonus(next_state)
            total_reward = self._postprocess_reward(float(raw_reward) + novelty_bonus)
            done = bool(terminated or truncated)

            self._store_transition(previous_state, action_value, total_reward, next_state, done)
            self._current_state = next_state
            self._last_action = action_value
            self._episode_reward += total_reward

            self.reward_stats.update(total_reward)
            self.novelty_stats.update(novelty_bonus)
            self.state_norm_stats.update(float(np.linalg.norm(next_state)))
            self.step_time_stats.update(time.perf_counter() - start)
            self.calculations.update_performance(total_reward)
            if done:
                self.episode_returns.append(float(self._episode_reward))

            info = dict(info or {})
            info.update({
                "novelty_bonus": float(novelty_bonus),
                "raw_reward": float(raw_reward),
                "total_reward": float(total_reward),
                "done": done,
                "episode_return": float(self._episode_reward),
                "state_norm": float(np.linalg.norm(next_state)),
            })
            self._last_info = info
            return next_state, float(total_reward), bool(terminated), bool(truncated), info
        except LearningError:
            raise
        except Exception as exc:
            raise EnvironmentStepError("SLAIEnv step failed", action=action_value, cause=exc) from exc

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        render_mode = mode or self.render_mode or "human"
        if self.env is not None:
            rendered = self.env.render()
            return None if rendered is None else np.asarray(rendered)

        if render_mode == "human":
            print(f"State: {self._current_state} | Steps: {self._step_count}/{self.max_steps}")
            return None
        if render_mode == "rgb_array":
            return self._render_to_image()
        raise InvalidConfigError("Unsupported render mode", received_value=render_mode, context={"supported": self.metadata["render_modes"]})

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        logger.info("SLAI Environment closed")

    # ------------------------------------------------------------------
    # Environment internals
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_reset_output(reset_output: Any) -> Tuple[Any, Dict[str, Any]]:
        if isinstance(reset_output, tuple) and len(reset_output) == 2 and isinstance(reset_output[1], dict):
            return reset_output[0], dict(reset_output[1])
        return reset_output, {}

    @staticmethod
    def _normalise_step_output(step_output: Any, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        if not isinstance(step_output, tuple):
            raise EnvironmentStepError("env.step() must return a tuple", action=action)
        if len(step_output) == 5:
            next_state, reward, terminated, truncated, info = step_output
            return next_state, float(reward), bool(terminated), bool(truncated), dict(info or {})
        if len(step_output) == 4:
            next_state, reward, done, info = step_output
            return next_state, float(reward), bool(done), False, dict(info or {})
        raise EnvironmentStepError("env.step() returned unsupported tuple length", step_output_length=len(step_output), action=action)

    def _coerce_state(self, state: Any, expected_dim: Optional[int] = None) -> np.ndarray:
        # Unwrap Gymnasium (obs, info) tuple
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            state = state[0]
    
        # Convert plain sequences to numpy array early, before tensor/array checks
        if isinstance(state, (tuple, list)) and not torch.is_tensor(state):
            arr = np.asarray(state, dtype=np.float32)
        elif torch.is_tensor(state):
            arr = state.detach().cpu().numpy() # type: ignore
        elif hasattr(state, "__array__"):
            arr = np.asarray(state)
        elif isinstance(state, Mapping):
            arr = np.asarray([state[k] for k in sorted(state.keys())], dtype=np.float32)
        else:
            arr = np.asarray(state, dtype=np.float32)
        arr = arr.astype(np.float32, copy=False).reshape(-1)
        if arr.size == 0:
            raise ObservationShapeError(expected_shape=(expected_dim or "non-empty",), actual_shape=tuple(arr.shape))
        if expected_dim is not None and arr.size != int(expected_dim):
            raise ObservationShapeError(expected_shape=(int(expected_dim),), actual_shape=tuple(arr.shape))
        if np.isnan(arr).any():
            raise NaNException("NaN detected in environment state", location="environment_state")
        if np.isinf(arr).any():
            raise InfException("Inf detected in environment state", location="environment_state")
        return arr

    def _validate_action(self, action: Any) -> int:
        if torch.is_tensor(action):
            if action.numel() != 1:
                raise InvalidActionError(action=repr(action), reason="tensor action must contain exactly one value")
            action = int(action.detach().cpu().item())
        elif isinstance(action, np.generic):
            action = int(action.item())
        elif isinstance(action, (list, tuple, np.ndarray)) and np.asarray(action).size == 1:
            action = int(np.asarray(action).reshape(-1)[0])
        else:
            try:
                action = int(action)
            except (TypeError, ValueError) as exc:
                raise InvalidActionError(action=action, reason="action must be integer-like", cause=exc) from exc

        contains = getattr(self.action_space, "contains", None)
        if callable(contains):
            try:
                if not contains(action):
                    raise ActionSpaceMismatchError(action, expected_space=repr(self.action_space))
            except TypeError:
                if not (0 <= action < self.action_dim):
                    raise ActionSpaceMismatchError(action, expected_space=f"Discrete({self.action_dim})")
        elif not (0 <= action < self.action_dim):
            raise ActionSpaceMismatchError(action, expected_space=f"Discrete({self.action_dim})")
        return int(action)

    def _create_state_transition_matrix(self) -> np.ndarray:
        damping = coerce_float(self.env_config.get("state_damping", 0.95), default=0.95, minimum=0.0, maximum=1.0)
        velocity_damping = coerce_float(self.env_config.get("velocity_damping", 0.90), default=0.90, minimum=0.0, maximum=1.0)
        A = np.eye(self.state_dim, dtype=np.float32) * damping
        if self.state_dim >= 4:
            A[0, 2] = self.dt
            A[1, 3] = self.dt
            A[2, 2] = velocity_damping
            A[3, 3] = velocity_damping
        return A

    def _create_action_effect_matrix(self) -> np.ndarray:
        B = np.zeros((self.state_dim, self.action_dim), dtype=np.float32)
        action_scale = coerce_float(self.env_config.get("action_scale", 0.2), default=0.2)
        velocity_action_scale = coerce_float(self.env_config.get("velocity_action_scale", 0.1), default=0.1)
        if self.action_dim >= 2 and self.state_dim >= 2:
            B[0, 0] = -action_scale
            B[0, 1] = action_scale
        if self.action_dim >= 4 and self.state_dim >= 4:
            B[1, 2] = -action_scale
            B[1, 3] = action_scale
        if self.action_dim >= 6 and self.state_dim >= 4:
            B[2, 4] = -velocity_action_scale
            B[2, 5] = velocity_action_scale
        return B

    def _space_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        low = getattr(self.observation_space, "low", None)
        high = getattr(self.observation_space, "high", None)
        if low is None or high is None:
            low = np.full(self.state_dim, -10.0, dtype=np.float32)
            high = np.full(self.state_dim, 10.0, dtype=np.float32)
        low_arr = np.asarray(low, dtype=np.float32).reshape(-1)
        high_arr = np.asarray(high, dtype=np.float32).reshape(-1)
        if low_arr.size != self.state_dim:
            low_arr = np.resize(low_arr, self.state_dim).astype(np.float32)
        if high_arr.size != self.state_dim:
            high_arr = np.resize(high_arr, self.state_dim).astype(np.float32)
        low_arr = np.where(np.isfinite(low_arr), low_arr, -10.0)
        high_arr = np.where(np.isfinite(high_arr), high_arr, 10.0)
        high_arr = np.maximum(high_arr, low_arr + 1.0e-6)
        return low_arr, high_arr

    def _simulate_dynamics(self, state: np.ndarray, action: int) -> np.ndarray:
        state = self._coerce_state(state, expected_dim=self.state_dim)
        next_state = self.A @ state
        if 0 <= action < self.action_dim:
            next_state = next_state + self.B[:, action]
        low, high = self._space_bounds()
        next_state = self.physics.apply_all(next_state.astype(np.float64), self.dt, low, high, self.mass)
        if self.quantum_jitter_probability > 0 and self._rng.random() < self.quantum_jitter_probability:
            hbar = float(getattr(self.physics, "constants", {}).get("hbar", 1.054571817e-34))
            jitter_scale = math.sqrt(max(hbar, 0.0) / 2.0)
            next_state = next_state + self._rng.normal(0.0, jitter_scale, self.state_dim)
        return np.clip(next_state, low, high).astype(np.float32)

    def _calculate_reward(self, state: np.ndarray, action: int, next_state: np.ndarray, previous_action: Optional[int] = None) -> float:
        w = self.reward_weights
        dist_curr = float(np.linalg.norm(state[:2])) if len(state) >= 2 else 0.0
        dist_next = float(np.linalg.norm(next_state[:2])) if len(next_state) >= 2 else 0.0
        progress = (dist_curr - dist_next) * w.get("progress", 10.0)
        action_cost = w.get("action_cost", -0.02) * (1.0 + float(action) ** 2)
        velocity = float(np.linalg.norm(next_state[2:4])) if len(next_state) >= 4 else 0.0
        stability = w.get("stability", -0.1) * velocity * velocity
        smoothness = 0.0 if previous_action is None else w.get("smoothness", -0.01) * abs(float(action) - float(previous_action))
        zone_reward = self._check_zone_rewards(next_state)
        boundary_penalty = self._boundary_penalty(next_state) * w.get("boundary", -1.0)
        reward = progress + action_cost + stability + smoothness + zone_reward + boundary_penalty + w.get("alive", 0.0)
        validate_finite(reward, "environment_reward")
        return float(reward)

    def _boundary_penalty(self, state: np.ndarray) -> float:
        low, high = self._space_bounds()
        if len(state) < 2:
            return 0.0
        near_low = np.isclose(state[:2], low[:2], atol=max(self.physics.boundary_margin, 1.0e-6))
        near_high = np.isclose(state[:2], high[:2], atol=max(self.physics.boundary_margin, 1.0e-6))
        return float(np.any(near_low | near_high))

    def _check_zone_rewards(self, state: np.ndarray) -> float:
        if len(state) < 2:
            return 0.0
        x, y = float(state[0]), float(state[1])
        for zone in self.zones:
            xmin, xmax, ymin, ymax = zone["bounds"]
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return float(zone["reward"])
        return 0.0

    def _check_termination(self, state: np.ndarray) -> bool:
        if len(state) < 2:
            return False
        x, y = float(state[0]), float(state[1])
        for zone in self.zones:
            if zone.get("terminal", False):
                xmin, xmax, ymin, ymax = zone["bounds"]
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    return True
        return False

    def _ensure_novelty_dim(self, state_dim: int) -> None:
        if getattr(self.novelty_detector, "input_dim", state_dim) != int(state_dim):
            self.novelty_detector = NoveltyDetector(int(state_dim))

    def _compute_novelty_bonus(self, next_state: np.ndarray) -> float:
        try:
            self._ensure_novelty_dim(len(next_state))
            state_tensor = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                score = float(self.novelty_detector(state_tensor).mean().item())
            if self.train_novelty_detector and hasattr(self.novelty_detector, "train_step"):
                self.novelty_detector.train_step(state_tensor)
            bonus = self.novelty_weight * score
            if self.novelty_clip is not None:
                limit = coerce_float(self.novelty_clip, default=0.0, minimum=0.0)
                if limit > 0:
                    bonus = clamp(bonus, -limit, limit)
            validate_finite(bonus, "novelty_bonus")
            return float(bonus)
        except LearningError:
            if self.strict_novelty_errors:
                raise
            logger.warning("Novelty bonus failed; using 0.0", exc_info=True)
            return 0.0
        except Exception as exc:
            if self.strict_novelty_errors:
                raise NumericalInstabilityError("Novelty detector failed", metric_name="novelty_bonus", cause=exc) from exc
            logger.warning("Novelty bonus failed; using 0.0: %s", exc)
            return 0.0

    def _postprocess_reward(self, reward: float) -> float:
        validate_finite(reward, "total_reward")
        if self.reward_clip is not None:
            limit = coerce_float(self.reward_clip, default=0.0, minimum=0.0)
            if limit > 0.0:
                reward = clamp(reward, -limit, limit)
        return float(reward)

    def _store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        if not self.memory_enabled:
            return
        exp = Transition(state=state.copy(), action=action, reward=float(reward), next_state=next_state.copy(), done=bool(done))
        if self.memory_priority_mode == "abs_reward":
            priority = abs(float(reward))
        elif self.memory_priority_mode == "terminal":
            priority = abs(float(reward)) + (1.0 if done else 0.0)
        else:
            priority = None
        self.learning_memory.add(exp, priority=priority, tag=f"episode_{self._episode_count}")

    # ------------------------------------------------------------------
    # Rendering / embeddings / metrics
    # ------------------------------------------------------------------
    def _render_to_image(self) -> np.ndarray:
        size = coerce_int(self.env_config.get("render_size", 400), default=400, minimum=64)
        img = np.zeros((size, size, 3), dtype=np.uint8)
        cx, cy = size // 2, size // 2
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), -1)

        for zone in self.zones:
            if len(zone["bounds"]) == 4:
                xmin, xmax, ymin, ymax = zone["bounds"]
                p1 = (int(np.clip(cx + xmin * 10, 0, size - 1)), int(np.clip(cy - ymax * 10, 0, size - 1)))
                p2 = (int(np.clip(cx + xmax * 10, 0, size - 1)), int(np.clip(cy - ymin * 10, 0, size - 1)))
                color = (0, 120, 0) if zone["type"] == "goal" else (0, 0, 120)
                cv2.rectangle(img, p1, p2, color, 1)

        if self._current_state is not None and len(self._current_state) >= 2:
            x, y = float(self._current_state[0]), float(self._current_state[1])
            px = int(np.clip(cx + x * 20, 0, size - 1))
            py = int(np.clip(cy - y * 20, 0, size - 1))
            cv2.circle(img, (px, py), 8, (0, 0, 255), -1)
            if len(self._current_state) >= 4:
                vx, vy = float(self._current_state[2]), float(self._current_state[3])
                end_x = int(np.clip(px + vx * 30, 0, size - 1))
                end_y = int(np.clip(py - vy * 30, 0, size - 1))
                cv2.arrowedLine(img, (px, py), (end_x, end_y), (255, 255, 0), 2)
        return img

    def get_state_embedding(self, state: Union[np.ndarray, torch.Tensor, List[float], str]) -> torch.Tensor:
        if isinstance(state, str):
            dim = int(getattr(self.novelty_detector, "input_dim", self.state_dim))
            hashed = stable_hash(state, digest_size=8)
            vals = [int(ch, 16) / 15.0 for ch in hashed]
            state_tensor = torch.as_tensor((vals + [0.0] * dim)[:dim], dtype=torch.float32).unsqueeze(0)
        else:
            arr = self._coerce_state(state)
            self._ensure_novelty_dim(len(arr))
            state_tensor = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            embedding = self.novelty_detector.predictor(state_tensor)
        return embedding

    def get_metrics(self) -> Dict[str, Any]:
        rewards = list(self.episode_returns)
        current_reward_summary = self.calculations.summarize_rewards(rewards) if rewards else self.calculations.summarize_rewards([self._episode_reward])
        memory_metrics = self.learning_memory.metrics() if hasattr(self.learning_memory, "metrics") else {"size": self.learning_memory.size()}
        physics_stats = self.physics.stats() if hasattr(self.physics, "stats") else {}
        return {
            "episode": self._episode_count,
            "steps": self._step_count,
            "max_steps": self.max_steps,
            "memory_size": self.learning_memory.size(),
            "memory": to_json_safe(memory_metrics),
            "reward_summary": current_reward_summary,
            "reward_stats": to_json_safe(self.reward_stats.snapshot()),
            "novelty_stats": to_json_safe(self.novelty_stats.snapshot()),
            "step_time_stats": to_json_safe(self.step_time_stats.snapshot()),
            "state_norm_stats": to_json_safe(self.state_norm_stats.snapshot()),
            "performance_trend": self.calculations.calculate_performance_trend(window=10),
            "physics": to_json_safe(physics_stats),
            "last_info": to_json_safe(self._last_info),
        }

    def diagnostics(self) -> Dict[str, Any]:
        return self.get_metrics()

    def set_task_context(self, context: Mapping[str, Any]) -> None:
        """Replace the environment context associated with the current task.
    
        Task context is metadata, not environment configuration. Consequently,
        this method does not implicitly modify dynamics, reward weights, zones,
        observation spaces, or action spaces.
    
        When SLAIEnv wraps another environment that implements the same protocol,
        the validated context is forwarded to it.
    
        Parameters
        ----------
        context:
            Mapping containing task-specific metadata. The mapping is copied so
            later top-level mutations by the caller do not alter environment state.
    
        Raises
        ------
        InvalidConfigError
            If ``context`` is not a mapping.
        """
        if not isinstance(context, Mapping):
            raise InvalidConfigError(
                "task_context must be a mapping",
                config_key="task_context",
                received_value=type(context).__name__,
            )
    
        normalized_context = dict(context)
    
        # Preserve compositional behaviour when wrapping another environment.
        if self.env is not None and self.env is not self:
            wrapped_setter = getattr(self.env, "set_task_context", None)
            if callable(wrapped_setter):
                wrapped_setter(dict(normalized_context))
            elif wrapped_setter is not None:
                logger.warning(
                    "Wrapped environment %s exposes a non-callable "
                    "set_task_context attribute; context was not forwarded.",
                    type(self.env).__name__,
                )
    
        # Replacement prevents metadata from a previous task leaking into the next.
        self._task_context = normalized_context

if __name__ == "__main__":
    print("\n=== Running  SLAIEnv ===\n")
    printer.status("TEST", " SLAIEnv initialized", "info")
    env = SLAIEnv(state_dim=4, action_dim=4, max_steps=6, config={
        "max_steps": 6,
        "memory_enabled": True,
        "novelty_weight": 0.01,
        "reward_clip": 100.0,
        "quantum_jitter_probability": 0.0,
        "physics": {"dt": 0.02, "gravity": 0.0, "enable_relativistic": False},
        "zones": [{"type": "goal", "bounds": [9, 10, 9, 10], "reward": 5, "terminal": True}],
    })
    obs, info = env.reset(seed=7)
    assert obs.shape == (4,) and info["episode"] == 1
    total = 0.0
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        assert np.isfinite(obs).all() and math.isfinite(reward)
        if terminated or truncated:
            break
    assert env.learning_memory.size() >= 1
    img = env.render("rgb_array")
    assert img.ndim == 3 and img.shape[-1] == 3 # type: ignore
    emb = env.get_state_embedding(obs)
    assert emb.ndim == 2
    metrics = env.get_metrics()
    assert metrics["memory_size"] >= 1 and "reward_summary" in metrics
    try:
        env.step(999)
        raise AssertionError("invalid action was not rejected")
    except LearningError:
        pass
    env.close()
    printer.status("TEST", "SLAIEnv dynamics, memory, rendering, metrics, and errors verified", "success")
    print("\n=== Test ran successfully ===\n")
