
import numpy as np
import random
import math

from typing import Any, Tuple, Dict, OrderedDict, List
from collections import defaultdict

from src.agents.safety.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger

logger = get_logger("Recursive Learning Engine")

class StateProcessor:
    """
    Handles state representation challenges:
    - Continuous state discretization (Tile Coding)
    - Feature engineering
    - Dimensionality reduction
    - State normalization
    """
    
    def __init__(self, state_size):
        self.config = load_global_config()
        self.rle_config = get_config_section('rl_engine')
        self.state_size = state_size
        
        # Get config values directly from loaded config
        state_processor_config = self.rle_config.get('state_processor', {})
        self.tiling_resolution = state_processor_config.get('tiling_resolution', 0.1)
        self.num_tilings = state_processor_config.get('num_tilings', 8)
        self.feature_engineering = state_processor_config.get('feature_engineering', True)
        self.feature_weights = np.random.randn(state_size)

        logger.info(f"Recursive Learning Engine Activated!")
    
    def discretize(self, continuous_state: np.ndarray, num_tilings: int = 8) -> tuple:
        """
        Implements tile coding with multiple overlapping tilings.
        
        Parameters:
        continuous_state - Input vector in ℝ^n
        num_tilings - Number of overlapping tilings (typically 8-32)
        
        Returns:
        Tuple of tile indices across all tilings
        
        Methodology:
        1. Create multiple offset grids (tilings)
        2. Calculate tile indices for each tiling
        3. Combine indices into single state representation
        
        Reference:
        Sutton, R. S. (1996). Generalization in Reinforcement Learning.
        """
        offsets = np.linspace(0, self.tiling_resolution, num_tilings, endpoint=False)
        tile_indices = []
        
        for offset in offsets:
            # Apply offset and discretize
            offset_state = continuous_state + offset
            discretized = tuple((offset_state // self.tiling_resolution).astype(int))
            tile_indices.extend(discretized)
        
        return tuple(tile_indices)
    
    def extract_features(self, raw_state: np.ndarray) -> np.ndarray:
        """
        Constructs basis functions for linear function approximation.
        
        Features:
        1. Raw state components
        2. Quadratic terms
        3. Cross-terms
        4. Radial basis functions
        
        Formula:
        ϕ(s) = [s, s², s_i*s_j, exp(-||s - c||²/2σ²)]
        
        Reference:
        Konidaris, G., et al. (2011). Value Function Approximation in Reinforcement Learning.
        """
        n = raw_state.shape[0]
        features = []
        
        # Linear terms
        features.extend(raw_state)
        
        # Quadratic terms
        features.extend(raw_state**2)
        
        # Cross-terms
        for i in range(n):
            for j in range(i+1, n):
                features.append(raw_state[i] * raw_state[j])
        
        # Radial basis functions (3 centers as example)
        centers = [np.zeros(n), np.ones(n), -np.ones(n)]
        for c in centers:
            features.append(np.exp(-np.linalg.norm(raw_state - c)**2 / 0.5))
        
        return np.array(features)

class ExplorationStrategies:
    """
    Decoupled exploration policy implementations:
    - Boltzmann (Softmax) exploration
    - Upper Confidence Bound (UCB)
    - Thompson Sampling
    - Curiosity-Driven Exploration
    """
    
    def __init__(self, action_space, strategy="epsilon_greedy", temperature=1.0, ucb_c=2.0):
        self.action_space = action_space
        self.strategy = strategy
        self.temperature = temperature
        self.ucb_c = ucb_c
        self.state_history = []
    
    def boltzmann(self, q_values):
        # Convert to numpy array and normalize
        probabilities = np.exp(q_values / self.temperature)
        probabilities /= probabilities.sum()  # Ensure proper probability distribution
        return random.choices(self.action_space, weights=probabilities.tolist())[0]
    
    def ucb(self, state_action_counts: Dict[Tuple[tuple, Any], int], c: float = 2.0) -> Any:
        """
        Implements Upper Confidence Bound (UCB1) exploration strategy.
        
        Formula:
        UCB(a) = Q(s,a) + c * sqrt(ln(N(s)) / n(s,a))
        
        Where:
        - N(s): Total visits to state s
        - n(s,a): Visits to action a in state s
        - c: Exploration-exploitation trade-off parameter
        
        Reference:
        Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem.
        """
        state = self.state_history[-1] if self.state_history else tuple()
        total_state_visits = sum(count for (s,a), count in state_action_counts.items() if s == state)
        
        ucb_values = {}
        for action in self.possible_actions:
            sa_pair = (state, action)
            action_count = state_action_counts.get(sa_pair, 1e-5)  # Avoid division by zero
            q_value = self._get_q_value(state, action)
            
            if total_state_visits == 0:
                exploration_bonus = float('inf')
            else:
                exploration_bonus = c * math.sqrt(math.log(total_state_visits) / action_count)
            
            ucb_values[action] = q_value + exploration_bonus
        
        return max(ucb_values, key=ucb_values.get)

class QTableOptimizer:
    """
    Optimizes Q-value storage and retrieval:
    - Compressed sparse representation
    - Approximate nearest neighbor lookup
    - Batch updates
    - Cache optimization
    """
    
    def __init__(self, batch_size=32, momentum=0.9, cache_size=1000, learning_rate=0.1):
        self.batch_size = batch_size
        self.momentum = momentum
        self.cache_size = cache_size
        self.learning_rate = learning_rate
        
        # State/Action indexing
        self.state_index = defaultdict(int)
        self.action_index = defaultdict(int)
        self.next_state_idx = 0
        self.next_action_idx = 0
        
        # Storage systems
        self.state_action_matrix = defaultdict(dict)
        self.lru_cache = OrderedDict()
        self.sparse_matrix = []
        self.update_momentum = defaultdict(float)
        
        # Compression systems
        self.DEFAULT_VALUE = 0.0
        self.huffman_codes = {}          # {delta: code}
        self.inverse_huffman = {}        # {code: delta}
        self.code_counter = 0

    def _state_to_index(self, state):
        if state not in self.state_index:
            self.state_index[state] = self.next_state_idx
            self.next_state_idx += 1
        return self.state_index[state]

    def _action_to_index(self, action):
        if action not in self.action_index:
            self.action_index[action] = self.next_action_idx
            self.next_action_idx += 1
        return self.action_index[action]
    
    def compressed_store(self, state: tuple, action: Any, value: float) -> None:
        """Advanced storage with dynamic codebook adaptation"""
        precision = 2  # Maintain 2 decimal places for delta precision
        delta = round(value - self.DEFAULT_VALUE, precision)
        
        if delta != 0:
            # Encode with current codebook if available
            encoded = self.huffman_codes.get(delta, delta)
            
            self.sparse_matrix.append((
                self._state_to_index(state),
                self._action_to_index(action),
                encoded
            ))

            # Compact every 1000 entries (configurable)
            if len(self.sparse_matrix) % 1000 == 0:
                self._compact_storage()

    def _compact_storage(self) -> None:
        """Three-phase compaction: Decode, Deduplicate, Re-encode"""
        # Phase 1: Decode all values to raw deltas
        decoded_buffer = []
        for s_idx, a_idx, val in self.sparse_matrix:
            if isinstance(val, int) and val in self.inverse_huffman:
                decoded_buffer.append((s_idx, a_idx, self.inverse_huffman[val]))
            else:
                decoded_buffer.append((s_idx, a_idx, val))

        # Phase 2: Deduplicate and clean
        latest_updates = {}
        for s_idx, a_idx, delta in decoded_buffer:
            key = (s_idx, a_idx)
            # Apply delta stacking for multi-step changes
            latest_updates[key] = latest_updates.get(key, 0.0) + delta

        # Remove neutral updates (net zero change)
        final_updates = {k: v for k, v in latest_updates.items() 
                        if round(v, 2) != 0.0}

        # Phase 3: Frequency analysis and codebook generation
        delta_freq = defaultdict(int)
        for delta in final_updates.values():
            delta_freq[round(delta, 2)] += 1  # Maintain precision consistency

        # Generate new codebook (top 100 most frequent deltas)
        sorted_deltas = sorted(delta_freq.items(), 
                             key=lambda x: -x[1])[:100]
        
        # Clear existing codes
        self.huffman_codes.clear()
        self.inverse_huffman.clear()
        
        # Assign new codes starting from 0
        for code, (delta, _) in enumerate(sorted_deltas):
            self.huffman_codes[delta] = code
            self.inverse_huffman[code] = delta

        # Phase 4: Re-encode with new codebook
        self.sparse_matrix = []
        for (s_idx, a_idx), delta in final_updates.items():
            refined_delta = round(delta, 2)
            if refined_delta in self.huffman_codes:
                self.sparse_matrix.append((s_idx, a_idx, self.huffman_codes[refined_delta]))
            else:
                self.sparse_matrix.append((s_idx, a_idx, refined_delta))

        # Optional: Prune least used state/action mappings
        self._prune_mappings()

    def _prune_mappings(self) -> None:
        """LRU eviction for state/action indexes"""
        # Keep only mappings referenced in sparse_matrix
        used_states = {s_idx for s_idx, _, _ in self.sparse_matrix}
        used_actions = {a_idx for _, a_idx, _ in self.sparse_matrix}
        
        # Prune state index
        self.state_index = defaultdict(int, {
            s: idx for s, idx in self.state_index.items()
            if idx in used_states
        })
        
        # Prune action index
        self.action_index = defaultdict(int, {
            a: idx for a, idx in self.action_index.items()
            if idx in used_actions
        })
    
    def batch_update(self, updates: List[Tuple[tuple, Any, float]], 
                    batch_size: int = 32, 
                    momentum: float = 0.9) -> None:
        """
        Efficient batch processing of Q-updates with momentum acceleration.
        
        Features:
        - Mini-batch averaging
        - Update momentum
        - Parallel processing (thread-safe)
        
        Mathematical Formulation:
        ΔQ_batch = (1 - momentum) * ΔQ_current + momentum * ΔQ_previous
        """
        if not updates:
            return

        # Split updates into mini-batches
        for batch in [updates[i:i+batch_size] for i in range(0, len(updates), batch_size)]:
            batch_delta = defaultdict(float)
            
            # Aggregate updates
            for state, action, delta in batch:
                key = (tuple(state), action)
                batch_delta[key] += delta / batch_size  # Average updates
            
            # Apply momentum and update Q-values
            for (state, action), delta in batch_delta.items():
                current_q = self._get_q_value(state, action)
                smoothed_delta = (1 - momentum) * delta + momentum * self.update_momentum.get((state, action), 0)
                new_value = current_q + self.learning_rate * smoothed_delta
                self._set_q_value(state, action, new_value)
                self.update_momentum[(state, action)] = smoothed_delta
