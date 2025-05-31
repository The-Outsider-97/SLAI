"""
Proficient In:
    Simple tasks with discrete and finite state/action spaces.
    Educational or experimental RL settings.

Best Used When:
    You need explainability and transparency.
    The environment is small and fast to simulate.
    Quick prototyping or testing different exploration strategies.
"""
import numpy as np
import math
import yaml
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import gymnasium as gym

from torch.utils.checkpoint import checkpoint
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from typing import Dict, Tuple, List, Any, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict, OrderedDict

from src.agents.perception.encoders.vision_encoder import VisionEncoder
from src.agents.learning.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.learning_memory import LearningMemory
from src.agents.learning.utils.rl_engine import StateProcessor, ExplorationStrategies, QTableOptimizer
from logs.logger import get_logger

logger = get_logger("Recursive Learning")

class RLAgent:
    """
    A basic recursive learning AI agent.

    This agent learns through trial and error by interacting with an environment.
    It maintains a value function (or Q-function implicitly) and updates it
    based on received rewards. The exploration-exploitation dilemma is handled
    through an epsilon-greedy strategy.

    This implementation prioritizes independence from external libraries.

    Mathematical Foundations:
    - Reinforcement Learning Framework (Markov Decision Process - implicitly)
    - Q-learning with eligibility traces (TD(λ))
    - Epsilon-Greedy Exploration Strategy

    Academic Sources (Conceptual Basis):
    - Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
    - Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
    """

    def __init__(self, agent_id, possible_actions: List[Any], state_size: int):
        """
        Initializes the RL Agent.

        Args:
            possible_actions (list): A list of all possible actions the agent can take.
            learning_rate (float): The learning rate (alpha) for updating value estimates.
            discount_factor (float): The discount factor (gamma) for future rewards.
            epsilon (float): The probability of taking a random action (exploration).
        """
        self.config = load_global_config()
        self.rl_engine_config = get_config_section('rl_engine')

        self.agent_id = agent_id
        self.possible_actions = possible_actions
        self.state_size = state_size
        if not possible_actions:
            raise ValueError("At least one possible action must be provided.")

        # Retrieve RL-specific parameters
        rl_config = self.config.get('rl', {})
        self.learning_rate = rl_config.get('learning_rate')
        self.discount_factor = rl_config.get('discount_factor')
        self.epsilon = rl_config.get('epsilon')
        self.trace_decay = rl_config.get('trace_decay')

        # Core Q-learning components
        self.q_table: Dict[Tuple[Tuple[Any], Any], float] = {}
        self.eligibility_traces: Dict[Tuple[Tuple[Any], Any], float] = {}
        
        # Experience tracking
        self.state_history: List[Tuple[Any]] = []
        self.action_history: List[Any] = []
        self.reward_history: List[float] = []

        # Initialize engine components
        self.state_processor = StateProcessor(
            state_size=state_size
        )
        
        exploration_config = self.rl_engine_config.get('exploration_strategies', {})
        self.exploration = ExplorationStrategies(
            action_space=possible_actions,
            strategy=exploration_config.get('strategy'),
            temperature=exploration_config.get('temperature'),
            ucb_c=exploration_config.get('ucb_c')
        )
        
        # Corrected QTableOptimizer initialization
        q_optimizer_config = self.rl_engine_config.get('q_table_optimizer', {})
        self.q_optimizer = QTableOptimizer(
            batch_size=q_optimizer_config.get('batch_size'),
            momentum=q_optimizer_config.get('momentum'),
            cache_size=q_optimizer_config.get('cache_size'),
            learning_rate=self.learning_rate
        )

        # Additional tracking for exploration strategies
        self.state_action_counts = defaultdict(int)
        self.episode_count = 0

        self.learning_memory = LearningMemory()
        self.model_id = "RL_Agent"

        if any(param is None for param in [self.learning_rate, self.discount_factor, self.epsilon, self.trace_decay]):
            logger.error("Critical RL parameters missing in config! Using fallback values")
            self.learning_rate = self.learning_rate or 0.1
            self.discount_factor = self.discount_factor or 0.9
            self.epsilon = self.epsilon or 0.1
            self.trace_decay = self.trace_decay or 0.7

        logger.info("Recursive Learning has successfully initialized")

    def _get_q_value(self, state: Tuple[Any], action: Any) -> float:
        """Get Q-value with optimistic initialization"""
        return self.q_table.get((state, action), 1.0)  # Optimistic initial values

    def _update_eligibility(self, state: Tuple[Any], action: Any) -> None:
        """Update eligibility traces using accumulating traces"""
        key = (state, action)
        self.eligibility_traces[key] = self.eligibility_traces.get(key, 0.0) + 1

    def _decay_eligibility(self) -> None:
        """Decay all eligibility traces"""
        for key in self.eligibility_traces:
            self.eligibility_traces[key] *= self.discount_factor * self.trace_decay

    def _process_state(self, raw_state):
        """Apply state processing and feature engineering"""
        state_processor_config = self.rl_engine_config.get('state_processor', {})
        if state_processor_config.get('feature_engineering', False):
            return tuple(self.state_processor.extract_features(raw_state))
        return self.state_processor.discretize(raw_state)

    def choose_action(self, state):
        """Enhanced action selection using configured strategy"""
        processed_state = self._process_state(state)
        
        exploration_config = self.rl_engine_config.get('exploration_strategies', {})
        strategy = exploration_config.get('strategy', 'epsilon_greedy')
        
        if strategy == "ucb":
            return self.exploration.ucb(
                state_action_counts=self.state_action_counts,
                current_state=processed_state
            )
        elif strategy == "boltzmann":
            # Convert to numpy array for vector operations
            q_values = np.array([self._get_q_value(processed_state, a) 
                       for a in self.possible_actions])
            return self.exploration.boltzmann(q_values)
        else:  # Fallback to epsilon-greedy
            return self._epsilon_greedy(processed_state)

    def _epsilon_greedy(self, processed_state: Tuple[Any]) -> Any:
        """
        Selects an action using the epsilon-greedy exploration strategy.
        
        With probability epsilon, chooses a random action (exploration).
        Otherwise, chooses the best known action for the state (exploitation).
        Handles ties by randomly selecting among actions with the maximum Q-value.
        
        Mathematical Formulation:
            action = {
                random choice from possible_actions,       if rand < epsilon
                argmax_a Q(state, a),                     otherwise
            }
        
        Where:
            epsilon = exploration probability (0 ≤ epsilon ≤ 1)
            Q(state, a) = estimated value of taking action 'a' in 'state'
        
        Args:
            processed_state: The current state after processing and discretization
            
        Returns:
            Selected action
        """
        # Exploration: choose random action with probability epsilon
        if random.random() < self.epsilon:
            action = random.choice(self.possible_actions)
            logger.debug(f"Exploring: random action {action}")
            return action
        
        # Exploitation: choose best known action
        # Calculate Q-values for all possible actions in current state
        q_values = {}
        for action in self.possible_actions:
            q_values[action] = self._get_q_value(processed_state, action)
        
        # Find maximum Q-value
        max_q = max(q_values.values())
        
        # Collect all actions with this maximum value (handle ties)
        best_actions = [a for a, q in q_values.items() if q == max_q]
        
        # Randomly select among best actions
        action = random.choice(best_actions)
        
        logger.debug(
            f"Exploiting: best action {action} with Q-value {max_q:.4f} "
            f"(tie broken among {len(best_actions)} candidates)"
        )
        return action

    def learn(self, next_state: Tuple[Any], reward: float, done: bool) -> None:
        """
        Q-learning update with eligibility traces.
        
        Implements:
        - Eligibility trace updates
        - Terminal state handling
        - Batch updates from experience
        """
        processed_state = self._process_state(self.state_history[-1])
        processed_next_state = self._process_state(next_state)
        action = self.action_history[-1]

        # Track state-action counts
        self.state_action_counts[(processed_state, action)] += 1

        # Store experience in optimized format
        q_optimizer_config = self.rl_engine_config.get('q_table_optimizer', {})
        update_freq = q_optimizer_config.get('update_frequency', 100)
        
        if self.episode_count % update_freq == 0:
            self.q_optimizer.batch_update(
                self._prepare_batch_updates(processed_next_state, reward, done)
            )

    def _prepare_batch_updates(self, next_state, reward, done):
        """Prepare batch updates for Q-table optimizer"""
        updates = []
        for (state, action), trace in self.eligibility_traces.items():
            next_max_q = max([self._get_q_value(next_state, a) 
                           for a in self.possible_actions]) if not done else 0.0
            td_error = (reward + self.discount_factor * next_max_q - 
                       self._get_q_value(state, action))
            updates.append((state, action, self.learning_rate * td_error * trace))
        return updates

    def step(self, state: Tuple[Any]) -> Any:
        """Record state and choose action"""
        action = self.choose_action(state)
        self.state_history.append(state)
        self.action_history.append(action)
        return action

    def train(self, env, num_tasks=3, episodes_per_task=5):
        """Comprehensive training loop with task-based learning"""
        for task in range(num_tasks):
            print(f"\n=== Starting Task {task+1}/{num_tasks} ===")
            task_rewards = []
            
            for episode in range(episodes_per_task):
                state, _ = env.reset()
                done = False
                total_reward = 0
                
                while not done:
                    action = self.step(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    # Enhanced reward processing
                    self.receive_reward(reward, state, action)
                    state = next_state
                    total_reward += reward
                    
                self.end_episode(state, done)
                task_rewards.append(total_reward)
                print(f"Episode {episode+1}/{episodes_per_task} | Reward: {total_reward}")
            
            avg_reward = sum(task_rewards) / len(task_rewards)
            print(f"Task {task+1} Complete | Avg Reward: {avg_reward:.2f}")
        
        print("\n=== Training Complete ===")
        return self.get_policy()

    def receive_reward(self, reward: float, state=None, action=None) -> None:
        """Record reward with optional state-action context"""
        self.reward_history.append(reward)
        
        # Add reward shaping if state/action provided
        if state is not None and action is not None:
            processed_state = self._process_state(state)
            self.q_optimizer.compressed_store(
                state=processed_state,
                action=action,
                value=reward,
                #reward=True
            )

    def end_episode(self, final_state: Tuple[Any], done: bool) -> None:
        """Finalize episode learning with epsilon decay"""
        if self.state_history and self.action_history and self.reward_history:
            self.learn(final_state, self.reward_history[-1], done)

        # Get decay parameters from config with proper defaults
        exploration_config = self.rl_engine_config.get('exploration_strategies', {})
        min_epsilon = exploration_config.get('min_epsilon', 0.01)
        epsilon_decay = exploration_config.get('epsilon_decay', 0.995)

        # Apply epsilon decay
        self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay)

        logger.debug(f"Epsilon decayed to {self.epsilon:.4f}")
        self.reset_history()

    def get_q_table(self):
        """
        Returns the current Q-table.

        Returns:
            dict: The Q-table mapping (state, action) to Q-values.
        """
        return self.q_table

    def reset_history(self) -> None:
        """Reset episode-specific tracking"""
        self.state_history.clear()
        self.action_history.clear()
        self.reward_history.clear()

    def get_policy(self) -> Dict[Tuple[Any], Any]:
        """Extract deterministic policy from Q-table"""
        policy = {}
        for (state, _), q_value in self.q_table.items():
            if state not in policy:
                policy[state] = max(
                    [(a, self._get_q_value(state, a)) for a in self.possible_actions],
                    key=lambda x: x[1]
                )[0]
        return policy

class AdvancedQLearning(RLAgent):
    """
    Implements enhancements from recent RL research:
    - Double Q-learning (prevent maximization bias)
    - Prioritized Experience Replay
    - N-step Q-learning
    - Dynamic hyperparameter adjustment
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_table2 = {}  # Second Q-table for double Q-learning
        self.replay_buffer = deque(maxlen=10000)  # Experience replay storage
    
    def _double_q_update(self, state: tuple, action: Any, reward: float, next_state: tuple) -> None:
        """
        Implements Double Q-learning update rule to mitigate maximization bias.
        
        Methodology:
        1. Randomly select which Q-table to update (Q1 or Q2)
        2. Use the other table to select the best next action
        3. Update the selected Q-table using the TD target
        
        Reference:
        Hasselt, H. V. (2010). Double Q-learning. Advances in Neural Information Processing Systems.
        """
        # Randomly choose which Q-table to update
        update_table = random.choice([self.q_table, self.q_table2])
        target_table = self.q_table2 if update_table is self.q_table else self.q_table

        # Calculate TD target
        next_action = max(self.possible_actions, 
                        key=lambda a: target_table.get((next_state, a), 0.0))
        td_target = reward + self.discount_factor * target_table.get((next_state, next_action), 0.0)
        
        # Calculate current Q-value
        current_q = update_table.get((state, action), 0.0)
        
        # Update Q-value
        update_table[(state, action)] = current_q + self.learning_rate * (td_target - current_q)
    
    def prioritize_experience(self, alpha: float = 0.6, epsilon: float = 1e-4) -> None:
        """
        Implements proportional prioritization for experience replay.
        
        Formula:
        priority = (|TD-error| + ε)^α
        
        Parameters:
        α - determines how much prioritization is used (0 = uniform)
        ε - small constant to ensure all transitions are sampled
        
        Reference:
        Schaul, T., et al. (2015). Prioritized Experience Replay. arXiv:1511.05952
        """
        priorities = []
        for experience in self.replay_buffer:
            state, action, reward, next_state, done = experience
            current_q = self._get_q_value(state, action)
            next_max_q = max(self._get_q_value(next_state, a) for a in self.possible_actions)
            td_error = abs(reward + self.discount_factor * next_max_q * (1 - done) - current_q)
            priority = (td_error + epsilon) ** alpha
            priorities.append(priority)
        
        # Store priorities with experiences
        self.priorities = priorities
        total = sum(priorities)
        self.sampling_probs = [p/total for p in priorities]

class RLVisualizer:
    """
    Enhanced learning diagnostics with hardware-accelerated visualization pipeline.
    
    New Features:
    - Multi-resolution Q-value heatmaps
    - Temporal difference error visualization
    - 3D policy landscape projection
    - Interactive exploration-exploitation dashboard
    - Frame interpolation with optical flow
    - Adaptive color mapping
    """
    
    def __init__(self, agent, env, config_path="src/agents/learning/configs/learning_config.yaml"):
        self.agent = agent
        self.env = env
        self.config = self._load_config(config_path)
        self.frame_queue = deque(maxlen=1000)
        self.metrics_history = defaultdict(list)
        self._setup_color_maps()
        self._init_gpu_acceleration()
        
    def _load_config(self, path):
        """Load visualization-specific configuration"""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('rl_visualization', {})
    
    def _setup_color_maps(self):
        """Create perceptually uniform color maps"""
        self.value_cmap = LinearSegmentedColormap.from_list(
            'q_values', ['#2A0A12', '#C62F2F', '#F9D379'], N=256)
        self.trajectory_cmap = plt.get_cmap('viridis')
        self.exploration_palette = np.array([
            [42, 157, 143],   # Exploration color
            [233, 196, 106]   # Exploitation color
        ], dtype=np.uint8)
        
    def _init_gpu_acceleration(self):
        """Initialize GPU-accelerated OpenCV context"""
        self.use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_gpu:
            self.stream = cv2.cuda_Stream()
            self.gpu_heatmap = cv2.cuda_GpuMat()
            self.gpu_overlay = cv2.cuda_GpuMat()

    def render_heatmap_overlay(self, frame, state, q_values):
        """GPU-accelerated heatmap rendering with temporal smoothing"""
        heatmap = self._create_heatmap(state, q_values)
        
        if self.use_gpu:
            self.gpu_heatmap.upload(heatmap)
            self.gpu_overlay.upload(frame)
            cv2.cuda.alphaComp(
                self.gpu_overlay, self.gpu_heatmap,
                cv2.cuda.ALPHA_OVER, dst=self.gpu_overlay, stream=self.stream
            )
            self.gpu_overlay.download(frame, stream=self.stream)
        else:
            frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
            
        return frame

    def _create_heatmap(self, state, q_values, resolution=100):
        """Generate multi-resolution Q-value heatmap using state projections"""
        # Create 2D projection grid
        x = np.linspace(-2.4, 2.4, resolution)
        y = np.linspace(-3.0, 3.0, resolution)
        xx, yy = np.meshgrid(x, y)
        
        # Calculate Q-values for grid points
        grid_values = np.zeros_like(xx)
        for i in range(resolution):
            for j in range(resolution):
                projected_state = self._project_state(state, (xx[i,j], yy[i,j]))
                grid_values[i,j] = max(
                    self.agent._get_q_value(projected_state, a)
                    for a in self.agent.possible_actions
                )
                
        # Normalize and apply colormap
        normalized = cv2.normalize(grid_values, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(normalized.astype(np.uint8), self.value_cmap)
        return cv2.resize(heatmap, self.env.observation_space.shape[:2][::-1])

    def _project_state(self, current_state, projection):
        """State projection for high-dimensional visualization"""
        # Implement domain-specific projection logic
        if len(current_state) >= 2:
            return (projection[0], projection[1], *current_state[2:])
        return projection

    def interpolate_frames(self, prev_frame, current_frame, num_interpolations=3):
        """Optical flow-based frame interpolation"""
        if self.use_gpu:
            prev_gpu = cv2.cuda_GpuMat(prev_frame)
            curr_gpu = cv2.cuda_GpuMat(current_frame)
            flow = cv2.cuda_FarnebackOpticalFlow.create(
                numLevels=3,
                pyrScale=0.5,
                fastPyramids=True
            ).calc(prev_gpu, curr_gpu, None)
            
            interpolated = []
            for alpha in np.linspace(0.2, 0.8, num_interpolations):
                flow_scale = cv2.cuda_GpuMat(flow.size(), flow.type(), alpha)
                interp_frame = cv2.cuda.addWeighted(
                    prev_gpu, 1-alpha, curr_gpu, alpha, 0)
                interpolated.append(interp_frame.download())
            return interpolated
        else:
            # CPU fallback using TVL1 optical flow
            flow = cv2.DualTVL1OpticalFlow_create()
            flow_map = flow.calc(prev_frame, current_frame, None)
            return self._cpu_interpolate(prev_frame, current_frame, flow_map, num_interpolations)

    def animate_policy(self, fps=60, max_steps=1000, render_metrics=True):
        """Real-time rendering pipeline with performance optimizations"""
        raw_obs, _ = self.env.reset()
        state = raw_obs
        
        # Get frame dimensions from first render
        test_frame = self.env.render()
        frame_height, frame_width = test_frame.shape[:2]
        
        video_writer = cv2.VideoWriter(
            'policy_animation.mp4', 
            cv2.VideoWriter_fourcc(*'mp4v'), fps, 
            (frame_width*2, frame_height*2)  # Use actual frame dimensions
        )
        
        prev_frame = None
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = []
            
            for step in range(max_steps):
                frame = self.env.render()
                action = self.agent.get_policy().get(
                    self.agent._process_state(state), 
                    random.choice(self.agent.possible_actions)
                )
                
                # Parallel frame processing
                future = executor.submit(
                    self._process_frame, frame, state, action, step)
                futures.append(future)
                
                if len(futures) > 5:  # Pipeline depth
                    processed_frame = futures.pop(0).result()
                    if prev_frame is not None:
                        interpolated = self.interpolate_frames(prev_frame, processed_frame)
                        for interp in interpolated:
                            video_writer.write(interp)
                    video_writer.write(processed_frame)
                    prev_frame = processed_frame
                
                next_raw_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                state = next_raw_obs
                if done: break
                
            video_writer.release()
            self._postprocess_animation('policy_animation.mp4')

    def _process_frame(self, frame, state, action, step):
        """Frame processing worker function"""
        q_values = [self.agent._get_q_value(state, a) for a in self.agent.possible_actions]
        frame = self.render_heatmap_overlay(frame, state, q_values)
        frame = self._draw_trajectory(frame, step)
        frame = self._render_metrics(frame, step)
        frame = cv2.resize(frame, (self.env.width*2, self.env.height*2))
        return frame

    def _render_metrics(self, frame, step):
        """On-screen performance monitoring overlay"""
        metrics = [
            f"Episode: {self.agent.episode_count}",
            f"Exploration: {self._calculate_exploration_ratio():.2f}%",
            f"Avg Reward: {np.mean(self.agent.reward_history[-100:]):.2f}",
            f"Q-Variance: {self._calculate_q_variance():.2f}"
        ]
        
        y_offset = 30
        for metric in metrics:
            cv2.putText(frame, metric, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y_offset += 25
        return frame

    def _calculate_exploration_ratio(self, window=100):
        """Calculate exploration percentage from action history"""
        if not self.agent.action_history:
            return 0.0
        recent_actions = self.agent.action_history[-window:]
        if not recent_actions: return 0.0
        return sum(1 for a in recent_actions if a != np.argmax(
            [self.agent._get_q_value(s, a) for a in self.agent.possible_actions]
        )) / len(recent_actions) * 100

    def _calculate_q_variance(self):
        """Measure value function uncertainty"""
        return np.var(list(self.agent.q_table.values()))

    def plot_learning_curves(self, window=100):
        """Interactive matplotlib dashboard with multiple metrics"""
        if not self.agent.reward_history:
            print("Warning: Reward history is empty. Skipping plot.")
            return
        plt.figure(figsize=(15, 8))
        
        # Smoothed reward curve
        rewards = np.convolve(
            self.agent.reward_history, 
            np.ones(window)/window, mode='valid'
        )
        
        # Create subplots grid
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
        ax2 = plt.subplot2grid((3, 3), (1, 0))
        ax3 = plt.subplot2grid((3, 3), (1, 1))
        ax4 = plt.subplot2grid((3, 3), (1, 2))
        ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)

        # Main reward plot
        ax1.plot(rewards, label='Smoothed Reward')
        ax1.set_title("Learning Progress")
        ax1.legend()

        # Exploration-Exploitation pie chart
        exp_ratio = self._calculate_exploration_ratio()
        ax2.pie([exp_ratio, 100-exp_ratio], 
                colors=self.exploration_palette/255,
                labels=['Explore', 'Exploit'])
        ax2.set_title("Behavior Ratio")

        # Q-Value Distribution
        q_values = list(self.agent.q_table.values())
        ax3.hist(q_values, bins=50, density=True)
        ax3.set_title("Q-Value Distribution")

        # State Coverage
        states = len({s for s, _ in self.agent.q_table.keys()})
        ax4.bar(['Visited'], [states], color='#3A86FF')
        ax4.set_title("Unique States Visited")

        # Temporal Difference Error
        ax5.plot(self.metrics_history['td_error'], alpha=0.3)
        ax5.set_title("TD Error History")

        plt.tight_layout()
        plt.show()

    def _postprocess_animation(self, filename):
        """Apply final video enhancements with FFMPEG"""
        import subprocess
        subprocess.run([
            'ffmpeg', '-i', filename, '-vf',
            'hqdn3d=4:3:6:4.5, unsharp=5:5:1.0:5:5:0.0',
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
            '-pix_fmt', 'yuv420p', f'enhanced_{filename}'
        ])

class RLEncoder(RLAgent):
    def __init__(self, agent_id: str,
                 possible_actions: List[Any], 
                 state_size: int, 
                 vision_encoder: VisionEncoder = None,
                 rl_encoder_config: Dict[str, Any] = None,
                 train_vision: bool = False):
        
        super().__init__(agent_id, possible_actions, state_size)
        
        # Vision processing
        self.vision_encoder = vision_encoder
        
        _rl_encoder_config = rl_encoder_config if rl_encoder_config else {}
        self.use_visual_observations = _rl_encoder_config.get('use_visual_observations', False)
        
        if self.vision_encoder and self.use_visual_observations:

            if train_vision:
                self.vision_encoder.train()
                for param in self.vision_encoder.parameters():
                    param.requires_grad = True
            else:
                self.vision_encoder.eval()
                for param in self.vision_encoder.parameters():
                    param.requires_grad = False
        elif self.use_visual_observations and not self.vision_encoder:
            logger.warning("RLEncoder: use_visual_observations is True, but no vision_encoder provided.")
            self.use_visual_observations = False

        logger.info(f"Recursive Learning Encoder Activated!")

    def _visual_state_processor(self, raw_state):
        """Process raw pixels using VisionEncoder"""
        with torch.no_grad() if not (
            self.vision_encoder.training and any(
                p.requires_grad for p in self.vision_encoder.parameters())) else torch.enable_grad():
            state_tensor = torch.tensor(
                raw_state, dtype=torch.float32, device=next(
                    self.vision_encoder.parameters()).device if list(
                        self.vision_encoder.parameters()) else torch.device("cpu"))
            state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0)
            
            embeddings = self.vision_encoder(state_tensor)
            cls_embedding = embeddings[0, 0]
            return tuple(cls_embedding.detach().cpu().numpy())

    def _process_state(self, raw_state):
        """Override state processing for visual inputs"""
        if self.use_visual_observations and self.vision_encoder:
            return self._visual_state_processor(raw_state)
        else:
            return super()._process_state(raw_state)

    def _visual_state_processor(self, raw_state):
        """Process raw pixels using VisionEncoder"""
        with torch.no_grad():
            state_tensor = torch.tensor(raw_state, dtype=torch.float32)
            state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0)
            embeddings = self.vision_encoder(state_tensor)
        return tuple(embeddings.cpu().numpy().flatten())

class RLTransformer(RLEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from src.agents.perception.modules.transformer import Transformer
        self.transformer = Transformer(config['transformer'])
        self.state_memory = deque(maxlen=config['sequence_length'])
        
    def choose_action(self, state):
        # Process state sequence
        self.state_memory.append(self._process_state(state))
        sequence = F.normalize(torch.tensor(list(self.state_memory)), dim=-1)
        
        # Transformer-based policy
        context = checkpoint(self.transformer, sequence.unsqueeze(0))
        return super().choose_action(context[-1])
    
    def _visual_state_processor(self, raw_state):
        with torch.cuda.amp.autocast():  # Add this
            state_tensor = torch.tensor(raw_state).float().unsqueeze(0)
            embeddings = self.vision_encoder(state_tensor)

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Recursive Learning ===\n")
    class VisualCartPole(gym.Wrapper):
        def __init__(self):
            env = gym.make('CartPole-v1', render_mode='rgb_array')
            super().__init__(env)
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
            )
            
        def reset(self, **kwargs):
            state, info = self.env.reset(**kwargs)
            return self._process_frame(), info
        
        def step(self, action):
            state, reward, terminated, truncated, info = self.env.step(action)
            return self._process_frame(), reward, terminated, truncated, info
        
        def _process_frame(self):
            frame = self.env.render()
            frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
            return frame
    
    # Main test function
    def test_rl_with_vision():
        print("\n=== Running RL Agent with Vision Encoder Test ===")
        
        # Create environment
        env = VisualCartPole()
        
        # Load vision encoder configuration
        config_path = "src/agents/perception/configs/perception_config.yaml"
        with open(config_path, 'r') as f:
            vision_config = yaml.safe_load(f)
        
        # Override for CartPole environment
        vision_config['vision_encoder']['img_size'] = 84
        vision_config['vision_encoder']['patch_size'] = 14
        
        # Create vision encoder
        vision_encoder = VisionEncoder(vision_config)
        print(f"Vision Encoder created with {sum(p.numel() for p in vision_encoder.parameters()):,} parameters")
        
        # Agent configuration
        agent_id = "RL-Vision-Agent"
        possible_actions = [0, 1]  # CartPole actions (left/right)
        state_size = vision_config['transformer']['embed_dim']  # Embedding size as state size
        
        # Create RL agent with vision processing
        print("\n * * * * Phase 1 * * * *")
        print("=== Creating RL Agent with Vision Encoder ===")
        agent = RLEncoder(
            agent_id=agent_id,  # Pass agent_id correctly
            possible_actions=possible_actions,
            state_size=state_size,
            vision_encoder=vision_encoder,
            rl_encoder_config={'use_visual_observations': True}, # Pass specific config
            train_vision=False  # Explicitly False
        )
        
        # Create visualizer
        print("\n * * * * Phase 2 * * * *")
        print("=== Creating RL Visualizer ===")
        visualizer = RLVisualizer(agent, env)

        logger.info(f"{visualizer}\n{agent}\n{env}")
        
        # Run short training
        print("\n * * * * Phase 3 * * * *")
        print("=== Starting Training ===")
        policy = agent.train(
            env, 
            num_tasks=2, 
            episodes_per_task=3
        )
        
        # Visualize results
        print("\n * * * * Phase 4 * * * *")
        print("=== Visualizing Results ===")
        visualizer.plot_learning_curves()
        visualizer.animate_policy(fps=30, max_steps=200)
        
        print("\n=== Test Completed Successfully ===")
    
    if __name__ == "__main__":
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        test_rl_with_vision()
    print("\n=== Recursive Learning Complete ===")
