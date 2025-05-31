
import cv2
import copy
import torch
import random
import numpy as np
import gymnasium as gym

from collections import namedtuple

from src.agents.learning.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.utils.physics_constraints import apply_environmental_effects, enforce_physics_constraints
from src.agents.learning.utils.policy_network import NoveltyDetector
from src.agents.learning.learning_memory import LearningMemory
from logs.logger import get_logger

logger = get_logger("SLAI Learning Environment")

@property
class spec:
    def reward_threshold(self):
        return type('Spec', (), {'reward_threshold': 200})

class SLAIEnv(gym.Env):
    """Enhanced environment for SLAI operations with comprehensive dynamics"""
    
    def __init__(self, state_dim=4, action_dim=2, env=None, max_steps=500):
        super().__init__()
        self.config = load_global_config()
        self.env_config = get_config_section('learning_env')
        self.env = env
        self.max_steps = max_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize spaces
        self.observation_space = self._create_observation_space(state_dim)
        self.action_space = self._create_action_space(action_dim)
        
        # Learning components
        self.learning_memory = LearningMemory()
        self.novelty_detector = NoveltyDetector(state_dim)
        self._current_state = None
        self._step_count = 0
        self._episode_count = 0
        
        # Environment parameters
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {'render_modes': ['human', 'rgb_array']}
        
        # Physics parameters
        self.dt = 0.05  # Time step
        self.gravity = 9.8
        self.friction_coeff = 0.1
        self.elasticity = 0.8

        goal_zone_bounds = [8.0, 9.0, 8.0, 9.0]  # x_min, x_max, y_min, y_max
        hazard_zone_bounds = [-5.0, -4.0, -5.0, -4.0]
        
        self.zones = [
            {'type': 'goal', 'bounds': goal_zone_bounds, 'reward': 100, 'terminal': True},
            {'type': 'hazard', 'bounds': hazard_zone_bounds, 'reward': -10, 'terminal': True},
        ]
        
        # Dynamics matrices
        self.A = self._create_state_transition_matrix()
        self.B = self._create_action_effect_matrix()
        
        #self.spec.id = type('Spec', (), {'reward_threshold': 200})
        logger.info(f"SLAI Environment successfully initialized with:\n{self.zones}")

    def _create_observation_space(self, state_dim):
        if self.env:
            return self.env.observation_space
        
        # Bounded continuous observation space
        return gym.spaces.Box(
            low=-10.0, 
            high=10.0, 
            shape=(state_dim,),
            dtype=np.float32
        )

    def _create_action_space(self, action_dim):
        if self.env:
            return self.env.action_space
        
        # Discrete action space with meaningful actions
        return gym.spaces.Discrete(action_dim)

    def _create_state_transition_matrix(self):
        """Create physics-based state transition matrix"""
        # Identity matrix with damping for stable dynamics
        A = np.eye(self.state_dim) * 0.95
        
        # Position-velocity relationships for first 4 states
        if self.state_dim >= 4:
            A[0, 2] = self.dt  # x += vx * dt
            A[1, 3] = self.dt  # y += vy * dt
            A[2, 2] = 0.9  # Friction for vx
            A[3, 3] = 0.9  # Friction for vy
            
        # Add gravity effect if state includes vertical velocity
        if self.state_dim >= 4:
            A[3, 3] -= self.gravity * self.dt * 0.1
            
        return A

    def _create_action_effect_matrix(self):
        """Create action effect matrix mapping actions to state changes"""
        B = np.zeros((self.state_dim, self.action_dim))
        
        # Map actions to state effects
        if self.action_dim >= 2 and self.state_dim >= 2:
            B[0, 0] = -0.2  # Left: decrease x-position
            B[0, 1] = 0.2   # Right: increase x-position
            
        if self.action_dim >= 4 and self.state_dim >= 4:
            B[1, 2] = -0.2  # Down: decrease y-position
            B[1, 3] = 0.2   # Up: increase y-position
            
        if self.action_dim >= 6 and self.state_dim >= 4:
            B[2, 4] = -0.1  # Decrease x-velocity
            B[2, 5] = 0.1   # Increase x-velocity
            
        return B

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.env:
            state, info = self.env.reset(seed=seed, options=options)
        else:
            # Generate random initial state near origin
            state = np.random.uniform(-0.5, 0.5, self.state_dim)
            info = {
                "episode": self._episode_count + 1,
                "initial_state": state.tolist(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "max_steps": self.max_steps,
                "reset_seed": seed,
                "using_wrapped_env": self.env is not None
            }
        
        self._current_state = state
        self._step_count = 0
        self._episode_count += 1
        self.novelty_detector = NoveltyDetector(len(state))
        
        logger.debug(f"Environment reset | State: {state[:5]}...")
        return state, info

    def step(self, action):
        self._step_count += 1
        
        if self.env:
            next_state, reward, terminated, truncated, original_info = self.env.step(action)
        else:
            next_state = self._simulate_dynamics(self._current_state, action)
            reward = self._calculate_reward(self._current_state, action, next_state)
            terminated = False
            truncated = self._step_count >= self.max_steps
            original_info = {
                "step_count": self._step_count,
                "episode": self._episode_count,
                "terminated_due_to": "step_limit" if truncated else None,
                "state": next_state.tolist(),
                "action_taken": action
            }
    
        # Calculate novelty bonus
        with torch.no_grad():
            state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            novelty_bonus = self.novelty_detector(state_tensor)
        total_reward = reward + novelty_bonus.item()
        
        # Build the info dictionary
        info = {
            **original_info,
            "novelty_bonus": novelty_bonus.item(),
            "raw_reward": reward,
            "total_reward": total_reward
        }
        
        # Store experience
        experience = Transition(
            state=self._current_state,
            action=action,
            reward=total_reward,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated
        )
        self.learning_memory.add(experience, tag=f"episode_{self._episode_count}")
        
        self._current_state = next_state
        logger.debug(f"Step {self._step_count} | Action: {action} | Reward: {total_reward:.2f}")
        return next_state, total_reward, terminated, truncated, info

    def _simulate_dynamics(self, state, action):
        """Enhanced physics-based dynamics simulation"""
        # Linear dynamics core
        next_state = self.A @ state  # 'state' here is self._current_state
        
        # Add action effect
        if action < self.action_dim:
            next_state += self.B[:, action]
        
        # Environmental forces - Call imported function
        next_state = apply_environmental_effects(self, next_state) # Pass self (as env_instance) and next_state
        
        # Physics constraints - Call imported function
        next_state = enforce_physics_constraints(self, next_state) # Pass self (as env_instance) and next_state
        
        # Random perturbations
        next_state += np.random.normal(0, 0.02, self.state_dim)
        
        # Boundary constraints (final clipping)
        next_state = np.clip(
            next_state, 
            self.observation_space.low, 
            self.observation_space.high
        )
            
        return next_state

    def _create_physics_variation(self):
        """Create physics-based task variation"""
        env_variant = copy.deepcopy(self.env)
        if hasattr(env_variant.unwrapped, 'dynamics_config'):
            dynamics = env_variant.unwrapped.dynamics_config
            env_variant.unwrapped.configure(
                mass=dynamics.mass * np.random.uniform(0.8, 1.2),
                damping=dynamics.damping * np.random.uniform(0.7, 1.3),
                gravity=dynamics.gravity * np.random.uniform(0.9, 1.1)
            )
        return env_variant

    def _enforce_physics_constraints(self, state):
        """Apply physical constraints to state"""
        # Ground collision
        if len(state) > 1 and state[1] < self.observation_space.low[1] + 0.01:
            state[1] = self.observation_space.low[1] + 0.01
            if len(state) > 3:
                state[3] = -state[3] * self.elasticity  # Bounce
                
        # Ceiling collision
        if len(state) > 1 and state[1] > self.observation_space.high[1] - 0.01:
            state[1] = self.observation_space.high[1] - 0.01
            if len(state) > 3:
                state[3] = -state[3] * self.elasticity
                
        # Wall collisions
        if len(state) > 0 and state[0] < self.observation_space.low[0] + 0.01:
            state[0] = self.observation_space.low[0] + 0.01
            if len(state) > 2:
                state[2] = -state[2] * self.elasticity
                
        if len(state) > 0 and state[0] > self.observation_space.high[0] - 0.01:
            state[0] = self.observation_space.high[0] - 0.01
            if len(state) > 2:
                state[2] = -state[2] * self.elasticity
                
        return state

    def _calculate_reward(self, state, action, next_state):
        """Multi-component reward function"""
        # Distance to target (origin)
        distance_current = np.linalg.norm(state[:2]) if len(state) >= 2 else 0
        distance_next = np.linalg.norm(next_state[:2]) if len(next_state) >= 2 else 0
        progress_reward = (distance_current - distance_next) * 10
        
        # Action cost penalty
        action_cost = -0.02 * (1 + action**2)
        
        # Stability bonus
        velocity = np.linalg.norm(state[2:4]) if len(state) >= 4 else 0
        stability_penalty = -0.1 * velocity**2
        
        # Control smoothness
        action_change_penalty = -0.01 * abs(action - self._last_action) if hasattr(self, '_last_action') else 0
        self._last_action = action
        
        return progress_reward + action_cost + stability_penalty + action_change_penalty

    def render(self, mode='human'):
        if self.env:
            return self.env.render(mode)
        
        if mode == 'human':
            print(f"State: {self._current_state} | Steps: {self._step_count}/{self.max_steps}")
        elif mode == 'rgb_array':
            return self._render_to_image()
        return None

    def _render_to_image(self):
        """Generate visualization of environment state"""
        size = 400
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Draw origin target
        cv2.circle(img, (size//2, size//2), 10, (0, 255, 0), -1)
        
        # Draw agent position if state has positional components
        if len(self._current_state) >= 2:
            x, y = self._current_state[:2]
            x_pos = int(size//2 + x * 100)
            y_pos = int(size//2 - y * 100)  # Invert y for display
            
            # Draw agent
            color = (0, 0, 255)  # Red
            cv2.circle(img, (x_pos, y_pos), 15, color, -1)
            
            # Draw velocity vector if available
            if len(self._current_state) >= 4:
                vx, vy = self._current_state[2:4]
                cv2.arrowedLine(
                    img, 
                    (x_pos, y_pos),
                    (int(x_pos + vx * 50), int(y_pos - vy * 50)),
                    (255, 255, 0),
                    2
                )
        
        return img

    def close(self):
        if self.env:
            self.env.close()
        logger.info("Environment closed")

    def get_state_embedding(self, state):
        return self.novelty_detector.predictor(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        )
    
    def get_metrics(self):
        return {
            'episode': self._episode_count,
            'steps': self._step_count,
            'memory_size': len(self.learning_memory),
            'novelty_score': self.novelty_detector.metrics()
        }

# Define Transition structure
Transition = namedtuple('Transition', 
    ['state', 'action', 'reward', 'next_state', 'terminated', 'truncated']
)


if __name__ == "__main__":
    print("\n=== Testing SLAI Learning Environment ===\n")
    state_dim = 4
    action_dim = 2
    env = None
    max_steps = 500

    state = np.array([0.1, 0.2, 0.3, 0.4])
    action = 0

    env = SLAIEnv(state_dim=state_dim, action_dim=action_dim, max_steps=max_steps, env=None)
    logger.info(f"{env}")
    print(env._simulate_dynamics(state, action))
    print("\n* * * * * Phase 2 * * * * *")
    mode='human'

    print(env._enforce_physics_constraints(state))
    print(env.render(mode=mode))
    print("\n* * * * * Phase 3 * * * * *")

    print(env.get_state_embedding(state))
    print("\n=== Succesfully Ran SLAI Learning Environment ===")
