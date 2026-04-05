import cv2
import copy
import torch
import random
import numpy as np
import gymnasium as gym

from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union

from ..base.utils.physics_constraints import PhysicsEngine, PhysicsConfig
from src.agents.learning.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.utils.policy_network import NoveltyDetector
from src.agents.learning.learning_memory import LearningMemory, Transition
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("SLAI Learning Environment")
printer = PrettyPrinter

class SLAIEnv(gym.Env):
    """
    Enhanced environment for SLAI operations with comprehensive dynamics,
    novelty bonus, and prioritized experience replay.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        env: Optional[gym.Env] = None,
        max_steps: int = 500,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the SLAI environment.

        Args:
            state_dim: Dimensionality of the state vector (ignored if env is provided).
            action_dim: Number of discrete actions (ignored if env is provided).
            env: Optional wrapped Gym environment.
            max_steps: Maximum steps per episode before truncation.
            config: Optional override for environment configuration.
        """
        super().__init__()
        self.config = load_global_config()
        if config is not None:
            self.env_config = config
        else:
            self.env_config = get_config_section('learning_env')
        if not self.env_config:
            self.env_config = {}  # use defaults

        self.env = env
        self.max_steps = max_steps
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize spaces (may be overridden by wrapped env)
        self.observation_space = self._create_observation_space(state_dim)
        self.action_space = self._create_action_space(action_dim)

        # Learning components
        self.learning_memory = LearningMemory()
        self.novelty_detector = NoveltyDetector(state_dim)
        self._current_state: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._episode_count: int = 0
        self._last_action: Optional[int] = None

        # Physics engine (will be configured from env_config)
        physics_cfg = self._build_physics_config()
        self.physics = PhysicsEngine(physics_cfg)
        # Store commonly used physics parameters for quick access
        self.dt = self.physics.config.dt
        self.gravity = self.physics.config.gravity
        self.friction_coeff = self.physics.config.friction_coeff
        self.elasticity = self.physics.config.elasticity
        self.wind_strength = self.physics.config.wind_strength
        self.wind_direction = self.physics.config.wind_direction
        self.drag_coeff = self.physics.config.drag_coeff
        self.terminal_velocity = self.physics.config.terminal_velocity
        self.rotational_friction = self.physics.config.rotational_friction
        self.mass = self.physics.config.default_mass

        # Zones (goal, hazard, etc.)
        self.zones = self._load_zones()

        # Dynamics matrices (used for linear approximation)
        self.A = self._create_state_transition_matrix()
        self.B = self._create_action_effect_matrix()

        # Reward shaping coefficients
        self.reward_weights = self.env_config.get('reward_weights', {
            'progress': 10.0,
            'action_cost': -0.02,
            'stability': -0.1,
            'smoothness': -0.01,
            'goal': 100.0,
            'hazard': -10.0,
        })

        logger.info(
            f"SLAI Environment initialized | state_dim={state_dim} | action_dim={action_dim} | "
            f"max_steps={max_steps} | wrapped_env={env is not None}"
        )

    def _build_physics_config(self) -> PhysicsConfig:
        """Build PhysicsConfig from environment configuration."""
        cfg_dict = self.env_config.get('physics', {})
        return PhysicsConfig(
            gravity=cfg_dict.get('gravity', 9.80665),
            friction_coeff=cfg_dict.get('friction_coeff', 0.02),
            rotational_friction=cfg_dict.get('rotational_friction', 0.01),
            wind_strength=cfg_dict.get('wind_strength', 0.0),
            wind_direction=cfg_dict.get('wind_direction', 0.0),
            drag_coeff=cfg_dict.get('drag_coeff', 0.01),
            terminal_velocity=cfg_dict.get('terminal_velocity', 50.0),
            elasticity=cfg_dict.get('elasticity', 0.8),
            default_mass=cfg_dict.get('default_mass', 1.0),
            default_charge=cfg_dict.get('default_charge', 0.0),
            enable_tunneling=cfg_dict.get('enable_tunneling', False),
            tunneling_probability=cfg_dict.get('tunneling_probability', 0.05),
            barrier_positions=tuple(cfg_dict.get('barrier_positions', [-8.0, 8.0])),
            barrier_width=cfg_dict.get('barrier_width', 0.1),
            enable_relativistic=cfg_dict.get('enable_relativistic', True),
            relativistic_threshold=cfg_dict.get('relativistic_threshold', 0.1),
            enable_electromagnetic=cfg_dict.get('enable_electromagnetic', False),
            electric_field=tuple(cfg_dict.get('electric_field', [0.0, 10.0])),
            magnetic_field=cfg_dict.get('magnetic_field', 0.5),
            dt=cfg_dict.get('dt', 0.05),
            boundary_margin=cfg_dict.get('boundary_margin', 0.01),
            corner_threshold=cfg_dict.get('corner_threshold', 0.05),
        )

    def _load_zones(self) -> List[Dict[str, Any]]:
        """Load goal/hazard zones from config or use defaults."""
        zones = self.env_config.get('zones', [])
        if not zones:
            zones = [
                {
                    'type': 'goal',
                    'bounds': [8.0, 9.0, 8.0, 9.0],  # x_min, x_max, y_min, y_max
                    'reward': 100,
                    'terminal': True
                },
                {
                    'type': 'hazard',
                    'bounds': [-5.0, -4.0, -5.0, -4.0],
                    'reward': -10,
                    'terminal': True
                },
            ]
        return zones

    def _create_observation_space(self, state_dim: int) -> gym.Space:
        """Create observation space (either from wrapped env or default Box)."""
        if self.env is not None:
            return self.env.observation_space
        return gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(state_dim,),
            dtype=np.float32
        )

    def _create_action_space(self, action_dim: int) -> gym.Space:
        """Create action space (either from wrapped env or default Discrete)."""
        if self.env is not None:
            return self.env.action_space
        return gym.spaces.Discrete(action_dim)

    def _create_state_transition_matrix(self) -> np.ndarray:
        """Create physics-based linear state transition matrix A."""
        A = np.eye(self.state_dim, dtype=np.float32) * 0.95  # damping
        dt = self.dt

        if self.state_dim >= 4:
            A[0, 2] = dt      # x += vx * dt
            A[1, 3] = dt      # y += vy * dt
            A[2, 2] = 0.9     # friction for vx
            A[3, 3] = 0.9     # friction for vy
            A[3, 3] -= self.gravity * dt * 0.1  # gravity effect

        return A

    def _create_action_effect_matrix(self) -> np.ndarray:
        """Create action effect matrix B (mapping discrete actions to state deltas)."""
        B = np.zeros((self.state_dim, self.action_dim), dtype=np.float32)

        if self.action_dim >= 2 and self.state_dim >= 2:
            B[0, 0] = -0.2   # left
            B[0, 1] = 0.2    # right

        if self.action_dim >= 4 and self.state_dim >= 4:
            B[1, 2] = -0.2   # down
            B[1, 3] = 0.2    # up

        if self.action_dim >= 6 and self.state_dim >= 4:
            B[2, 4] = -0.1   # decelerate x
            B[2, 5] = 0.1    # accelerate x

        return B

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to an initial state."""
        super().reset(seed=seed)

        if self.env is not None:
            state, info = self.env.reset(seed=seed, options=options)
        else:
            # Generate random initial state near origin
            state = np.random.uniform(-0.5, 0.5, self.state_dim).astype(np.float32)
            info = {
                "episode": self._episode_count + 1,
                "initial_state": state.tolist(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "max_steps": self.max_steps,
                "reset_seed": seed,
                "using_wrapped_env": False
            }

        self._current_state = state
        self._step_count = 0
        self._episode_count += 1
        self._last_action = None

        # Reset novelty detector (optional, but we recreate to avoid stale state)
        self.novelty_detector = NoveltyDetector(len(state))

        logger.debug(f"Environment reset | State: {state[:5]}... | Episode: {self._episode_count}")
        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take an action and step the environment."""
        self._step_count += 1
        self._last_action = action if self._last_action is None else action

        if self.env is not None:
            next_state, reward, terminated, truncated, info = self.env.step(action)
        else:
            next_state = self._simulate_dynamics(self._current_state, action)
            reward = self._calculate_reward(self._current_state, action, next_state)
            terminated = self._check_termination(next_state)
            truncated = self._step_count >= self.max_steps
            info = {
                "step_count": self._step_count,
                "episode": self._episode_count,
                "terminated_due_to": "step_limit" if truncated else ("zone" if terminated else None),
                "state": next_state.tolist(),
                "action_taken": action
            }

        # Compute novelty bonus
        with torch.no_grad():
            state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            novelty_bonus = self.novelty_detector(state_tensor).item()
        total_reward = reward + novelty_bonus

        info.update({
            "novelty_bonus": novelty_bonus,
            "raw_reward": reward,
            "total_reward": total_reward
        })

        # Store experience (use the Transition from learning_memory)
        exp = Transition(
            state=self._current_state.copy(),
            action=action,
            reward=total_reward,
            next_state=next_state.copy(),
            done=terminated or truncated  # 'done' field expected by LearningMemory
        )
        self.learning_memory.add(exp, tag=f"episode_{self._episode_count}")

        self._current_state = next_state
        logger.debug(f"Step {self._step_count} | Action: {action} | Reward: {total_reward:.4f}")
        return next_state, total_reward, terminated, truncated, info

    def _simulate_dynamics(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Simulate next state using linear dynamics, action effect, and physics engine.
        """
        # Linear core dynamics
        next_state = self.A @ state

        # Add action effect
        if 0 <= action < self.action_dim:
            next_state += self.B[:, action]

        # Apply environmental effects using the physics engine
        low = self.observation_space.low
        high = self.observation_space.high
        next_state = self.physics.apply_all(next_state, self.dt, low, high, self.mass)

        # Optional quantum fluctuations (legacy, can be moved to physics engine)
        if np.random.random() < 0.01:  # 1% chance
            hbar = self.physics.constants["hbar"]
            quantum_jitter = np.random.normal(0, np.sqrt(hbar / 2), self.state_dim)
            next_state[:len(quantum_jitter)] += quantum_jitter

        # Final safety clip
        next_state = np.clip(next_state, low, high)

        return next_state.astype(np.float32)

    def _calculate_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """
        Multi-component reward function.
        """
        w = self.reward_weights

        # Progress towards origin (negative distance change)
        dist_curr = np.linalg.norm(state[:2]) if len(state) >= 2 else 0.0
        dist_next = np.linalg.norm(next_state[:2]) if len(next_state) >= 2 else 0.0
        progress = (dist_curr - dist_next) * w.get('progress', 10.0)

        # Action cost (small penalty to encourage minimal actions)
        action_cost = w.get('action_cost', -0.02) * (1 + action ** 2)

        # Stability penalty (discourage high velocities)
        vel = np.linalg.norm(next_state[2:4]) if len(next_state) >= 4 else 0.0
        stability = w.get('stability', -0.1) * (vel ** 2)

        # Smoothness penalty (action change)
        smoothness = 0.0
        if hasattr(self, '_last_action') and self._last_action is not None:
            smoothness = w.get('smoothness', -0.01) * abs(action - self._last_action)
        self._last_action = action

        # Zone rewards
        zone_reward = self._check_zone_rewards(next_state)

        return progress + action_cost + stability + smoothness + zone_reward

    def _check_zone_rewards(self, state: np.ndarray) -> float:
        """Check if the agent is inside any goal/hazard zone."""
        if len(state) < 2:
            return 0.0
        x, y = state[0], state[1]
        for zone in self.zones:
            xmin, xmax, ymin, ymax = zone['bounds']
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return float(zone['reward'])
        return 0.0

    def _check_termination(self, state: np.ndarray) -> bool:
        """Check if the episode should terminate due to zone entry."""
        if len(state) < 2:
            return False
        x, y = state[0], state[1]
        for zone in self.zones:
            if zone.get('terminal', False):
                xmin, xmax, ymin, ymax = zone['bounds']
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    return True
        return False

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        if self.env is not None:
            return self.env.render(mode)

        if mode == 'human':
            print(f"State: {self._current_state} | Steps: {self._step_count}/{self.max_steps}")
            return None
        elif mode == 'rgb_array':
            return self._render_to_image()
        return None

    def _render_to_image(self) -> np.ndarray:
        """Generate a simple 2D visualization as an RGB image."""
        size = 400
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Draw origin target
        cx, cy = size // 2, size // 2
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), -1)

        if self._current_state is not None and len(self._current_state) >= 2:
            x, y = self._current_state[0], self._current_state[1]
            px = int(cx + x * 100)
            py = int(cy - y * 100)  # flip y
            px = np.clip(px, 0, size - 1)
            py = np.clip(py, 0, size - 1)
            cv2.circle(img, (px, py), 15, (0, 0, 255), -1)

            if len(self._current_state) >= 4:
                vx, vy = self._current_state[2], self._current_state[3]
                end_x = int(px + vx * 50)
                end_y = int(py - vy * 50)
                cv2.arrowedLine(img, (px, py), (end_x, end_y), (255, 255, 0), 2)

        return img

    def close(self) -> None:
        """Clean up resources."""
        if self.env is not None:
            self.env.close()
        logger.info("SLAI Environment closed")

    def get_state_embedding(self, state: Union[np.ndarray, torch.Tensor, List[float]]) -> torch.Tensor:
        """
        Generate a state embedding using the novelty detector's predictor network.
        """
        if isinstance(state, str):
            # Fallback for string inputs (unlikely, but kept for compatibility)
            emb_dim = self.novelty_detector.predictor[0].in_features
            hashed = hash(state) % (10 ** 8)
            digits = [int(x) for x in str(abs(hashed))]
            padded = (digits + [0] * emb_dim)[:emb_dim]
            state_vec = np.array(padded, dtype=np.float32) / 10.0
            state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        elif isinstance(state, (list, np.ndarray)):
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        elif isinstance(state, torch.Tensor):
            state_tensor = state.float()
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")

        with torch.no_grad():
            embedding = self.novelty_detector.predictor(state_tensor)
        return embedding

    def get_metrics(self) -> Dict[str, Any]:
        """Return current environment metrics."""
        return {
            'episode': self._episode_count,
            'steps': self._step_count,
            'memory_size': self.learning_memory.size(),
            'novelty_score': self.novelty_detector.metrics() if hasattr(self.novelty_detector, 'metrics') else {},
        }

    @property
    def spec(self):
        """Return a dummy spec object for Gymnasium compatibility."""
        class Spec:
            reward_threshold = 200
        return Spec()


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Testing SLAI Learning Environment ===\n")

    env = SLAIEnv(state_dim=4, action_dim=2, max_steps=500)
    obs, info = env.reset()
    print(f"Initial observation: {obs}")

    total_reward = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {step+1}: action={action}, reward={reward:.3f}, total={total_reward:.3f}")
        if terminated or truncated:
            print("Episode ended")
            break

    # Test physics constraints with out-of-bound state
    test_state = np.array([-11.0, 0.2, 0.3, 0.4])
    print(f"\nBefore constraint: {test_state}")
    constrained = env.physics.enforce_boundary_constraints(
        test_state, env.observation_space.low, env.observation_space.high
    )
    print(f"After constraint: {constrained}")

    # Test embedding
    emb = env.get_state_embedding(obs)
    print(f"State embedding shape: {emb.shape}")

    print("\n=== Successfully Ran SLAI Learning Environment ===")
