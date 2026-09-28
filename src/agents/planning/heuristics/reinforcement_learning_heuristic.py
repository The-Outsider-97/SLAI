"""
Reinforcement Learning Heuristic for Task Planning

This module implements a reinforcement learning–based heuristic that learns task-method selection policies over time 
using policy gradient algorithms. Unlike supervised learning heuristics, this module optimizes action choices based on 
long-term rewards gathered through interaction with the planning environment.

It maps task-method features to a probability distribution over available methods and updates its policy based on observed 
outcomes (e.g., task success/failure, time efficiency, or other reward signals).

Real-World Use Case:
1. Autonomous Robots: Learning which navigation or manipulation strategy yields the highest long-term success in dynamic environments.
2. Smart Assistants: Adapting method choices based on evolving user preferences and delayed feedback (e.g., task scheduling success).
3. Mission Planning: Optimizing sequences of actions in search-and-rescue or logistics under uncertainty, where immediate success 
   doesn't always reflect long-term effectiveness.
"""

import os
import json
import threading
import numpy as np # type: ignore
import joblib # type: ignore
import torch # type: ignore

from typing import List, Any, Dict, Tuple, Optional

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.base_heuristic import BaseHeuristics
from ..utils.planning_errors import *
from ..utils.planning_helpers import clamp, require_type
from ...learning.modules.policy_network import PolicyNetwork
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reinforcement Learning Heuristic")
printer = PrettyPrinter()

class DotDict(dict):
    def __getattr__(self, item):
        return self.get(item)

class ReinforcementLearningHeuristic(BaseHeuristics):
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.heuristics_config = get_config_section('global_heuristic')
        self.rl_config = get_config_section('reinforcement_learning_heuristic')

        # RL parameters
        self.hidden_size = self.rl_config.get('hidden_size', 64)
        self.learning_rate = self.rl_config.get('learning_rate', 0.01)
        self.exploration_rate = self.rl_config.get('initial_exploration', 0.3)
        self.min_exploration = self.rl_config.get('min_exploration', 0.05)
        self.exploration_decay = self.rl_config.get('exploration_decay', 0.995)
        self.gamma = self.rl_config.get('discount_factor', 0.99)

        # Paths
        self.planning_db_path = self.heuristics_config.get('planning_db_path')
        self.heuristic_model_path = self.heuristics_config.get('heuristic_model_path') or '.'
        os.makedirs(self.heuristic_model_path, exist_ok=True)
        self.model_path = os.path.join(self.heuristic_model_path, 'rl_heuristic_model.pkl')

        # Features
        self.feature_config = self.rl_config.get('feature_config', {})
        self.feature_names = self._get_feature_names()

        # Model state
        self.policy_net = None
        self.method_to_index = {}
        self.candidate_methods = None
        self.trained = False
        self._lock = threading.RLock()
        self.episode_cache = []
        self.current_state = None
        self.current_method_idx = None

        self._load_model()
        
        logger.info(f"Reinforcement Learning Heuristic initialized with {len(self.feature_names)} features")

    def _get_feature_names(self) -> List[str]:
        base = ['task_depth', 'goal_overlap', 'method_failure_rate', 'state_diversity']
        if self.feature_config.get("use_priority"):
            base.append('task_priority')
        if self.feature_config.get("use_resource_check"):
            base.extend(['cpu_available', 'memory_available'])
        if self.feature_config.get("use_temporal_features"):
            base.extend(['time_since_creation', 'deadline_proximity'])
        return base

    def _load_model(self):
        with self._lock:
            if os.path.exists(self.model_path):
                try:
                    data = joblib.load(self.model_path)
                    self.feature_names = data['feature_names']
                    self.method_to_index = data['method_to_index']
                    self.candidate_methods = sorted(self.method_to_index, key=self.method_to_index.get)
                    self.exploration_rate = data['exploration_rate']
                    state_size = len(self.feature_names)
                    action_size = len(self.candidate_methods)
                    self.policy_net = PolicyNetwork(
                        input_dim=state_size,
                        output_dim=action_size,
                        hidden_sizes=[self.hidden_size, self.hidden_size//2] if self.hidden_size else None,
                        hidden_activation="relu",
                        output_activation="softmax",
                    )
                    # Load weights
                    self.policy_net.load_state_dict(data['weights'])
                    self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
                    self.trained = True
                    logger.info("Loaded pre-trained RL model")
                except Exception as e:
                    logger.error(f"Failed to load RL model: {e}")
                    # Clean up incomplete state
                    self.policy_net = None
                    self.optimizer = None
                    self.method_to_index = {}
                    self.candidate_methods = None
                    self.trained = False

    def _ensure_network_ready(self, candidate_methods: List[str]) -> bool:
        with self._lock:
            if (self.policy_net is not None and self.optimizer is not None and 
                self.candidate_methods == candidate_methods):
                return True
            # Need to initialize or reinitialize
            state_size = len(self.feature_names)
            action_size = len(candidate_methods)
            self.policy_net = PolicyNetwork(
                input_dim=state_size,
                output_dim=action_size,
                hidden_sizes=[self.hidden_size, self.hidden_size//2] if self.hidden_size else None,
                hidden_activation="relu",
                output_activation="softmax",
            )
            self.method_to_index = {m: i for i, m in enumerate(candidate_methods)}
            self.candidate_methods = candidate_methods
            # Create optimizer for the new network
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
            self.trained = False
            logger.info(f"Initialised new RL network for {action_size} methods")
            return False  # not trained

    def select_best_method(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        candidate_methods: List[str],
        method_stats: Dict[Tuple[str, str], Dict[str, int]]
    ) -> Tuple[Optional[str], float]:
        self._ensure_network_ready(candidate_methods)

        # Extract features for current state (without method)
        original_method = task.get("selected_method")
        # For state representation we don't need a specific method
        state_features = self.extract_state_features(task, world_state, method_stats)

        with self._lock:
            # Exploration
            if np.random.rand() < self.exploration_rate:
                idx = np.random.choice(len(candidate_methods))
                prob = 1.0 / len(candidate_methods)
            else:
                state_tensor = torch.tensor(state_features.reshape(1, -1), dtype=torch.float32)
                assert self.policy_net is not None
                probs = self.policy_net.forward(state_tensor).squeeze(0)
                dist = torch.distributions.Categorical(probs)
                idx = dist.sample().item()
                prob = probs[idx].item()

            # Decay exploration rate
            self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)

            selected_method = candidate_methods[idx]
            self.current_state = state_features
            self.current_method_idx = idx
            return selected_method, prob

    def extract_state_features(self, task: Dict[str, Any], world_state: Dict[str, Any],
                               method_stats: Dict) -> np.ndarray:
        # Extract features that do not depend on method
        features = np.zeros(len(self.feature_names))
        idx = 0

        features[idx] = self._calculate_task_depth(task)
        idx += 1
        features[idx] = self._calculate_goal_overlap(task, world_state)
        idx += 1
        # For method_failure_rate we need a method – we can use a placeholder or average
        # In RL we often don't include method‑specific features in the state.
        # We'll set it to 0.5 for now.
        features[idx] = 0.5
        idx += 1
        features[idx] = self._calculate_state_diversity(world_state)
        idx += 1

        if self.feature_config.get("use_priority"):
            features[idx] = task.get("priority", 0.5)
            idx += 1

        if self.feature_config.get("use_resource_check"):
            features[idx] = world_state.get("cpu_available", 0.0)
            idx += 1
            features[idx] = world_state.get("memory_available", 0.0)
            idx += 1

        if self.feature_config.get("use_temporal_features"):
            temporal = self.extract_temporal_features(task)
            features[idx] = clamp(float(temporal.get("time_since_creation", 0.0)), 0.0, 1.0)
            idx += 1
            features[idx] = clamp(float(temporal.get("deadline_proximity", 0.0)), 0.0, 1.0)
            idx += 1

        return features

    def predict_success_prob(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        method_stats: Dict[Tuple[str, str], Dict[str, int]],
        method_id: str
    ) -> float:
        if not self.trained or self.policy_net is None:
            return 0.5
        state_features = self.extract_state_features(task, world_state, method_stats)
        state_tensor = torch.tensor(state_features.reshape(1, -1), dtype=torch.float32)
        probs = self.policy_net.forward(state_tensor).squeeze(0)
        idx = self.method_to_index.get(method_id)
        if idx is None:
            return 0.5
        return probs[idx].item()

    def record_outcome(self, reward: float):
        """Store experience for later policy update."""
        with self._lock:
            if self.current_state is not None and self.current_method_idx is not None:
                self.episode_cache.append((self.current_state, self.current_method_idx, reward))
                self.current_state = None
                self.current_method_idx = None

    def update_policy(self):
        """REINFORCE update using cached episode."""
        with self._lock:
            if not self.episode_cache:
                return
            # Compute discounted returns (advantages)
            discounted = []
            cumulative = 0.0
            for _, _, r in reversed(self.episode_cache):
                cumulative = r + self.gamma * cumulative
                discounted.insert(0, cumulative)
            discounted = torch.tensor(discounted, dtype=torch.float32)
    
            # Normalize advantages for stability
            if discounted.numel() > 1:
                discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)
            else:
                discounted = discounted - discounted.mean()  # single element, std would be 0
    
            # Compute policy loss
            assert self.policy_net is not None
            loss = torch.zeros((), dtype=torch.float32)
            for (state, action_idx, _), adv in zip(self.episode_cache, discounted):
                state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
                probs = self.policy_net(state_tensor).squeeze(0)  # shape (action_size,)
                # Use torch.distributions for numerical stability
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(torch.tensor(action_idx))
                loss -= log_prob * adv
    
            # Backward pass and optimizer step
            assert self.optimizer is not None
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            # Clear cache
            self.episode_cache = []
            self.trained = True
            self._save_model()

    def save_model(self):
        """Public wrapper to save the model to disk."""
        self._save_model()

    def _save_model(self):
        with self._lock:
            if self.policy_net is None:
                return
            data = {
                'weights': self.policy_net.state_dict(),  # save as state dict
                'exploration_rate': self.exploration_rate,
                'feature_names': self.feature_names,
                'method_to_index': self.method_to_index,
            }
            joblib.dump(data, self.model_path)
            logger.info("RL model saved")

    def load_planning_db(self, path: str | None = None):
        """Load task, world state, and method statistics from the planning database."""
        if path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            planning_db_path = self.planning_db_path
            if planning_db_path is None:
                raise ValueError("planning_db_path is not configured")
            path = os.path.join(base_dir, planning_db_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tasks = data["tasks"]
        method_stats = {tuple(k.split(":")): v for k, v in data["method_stats"].items()}
        world_states = data["world_states"]
        return tasks, method_stats, world_states

    def _calculate_method_failure_rate(
        self,
        task: Any,
        method_stats: Dict[Any, Dict[str, Any]],
        method_id: Any,
    ) -> float:
        """
        Return the normalised failure rate for ``method_id`` on ``task``.

        Delegates to ``BaseHeuristics._resolve_method_stats``, which already
        resolves tuple keys like ``(task_name, method_id)``, string keys like
        ``"task_name:method_id"``, and plain ``method_id`` keys.  It also
        understands explicit ``success`` / ``total`` / ``failure`` fields and
        list‑based outcome histories.

        The result is clamped to ``[0.0, 1.0]``.

        .. note::
            ``ReinforcementLearningHeuristic.extract_state_features`` is
            deliberately method‑agnostic — it feeds the policy network only
            with state features, not with per‑method reliability.  This method
            is provided so that callers can still query method reliability from
            the shared base class when they need it (e.g. for diagnostics or
            reward shaping), and so the public API stays consistent with the
            other heuristics.

        Raises
        ------
        PlanningConfigError
            If ``method_stats`` is not a mapping, ``method_id`` is missing, or
            an unexpected type/value is encountered while computing the rate.
        """
        require_type(method_stats, dict, "method_stats")

        if method_id is None or method_id == "":
            raise PlanningConfigError(
                "method_id must be provided and non-empty.",
                config_key="method_id",
                config_section="reinforcement_learning_heuristic",
                expected_type="non-empty string or numeric identifier",
            )

        try:
            stats = self._resolve_method_stats(task, method_stats, method_id)
            failure_rate = float(stats.get("failure_rate", 0.0))
            return clamp(failure_rate, 0.0, 1.0)
        except PlanningError:
            raise
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            raise PlanningConfigError(
                f"Failed to compute method failure rate for '{method_id}': {exc}",
                config_key="method_stats",
                config_section="reinforcement_learning_heuristic",
                expected_type="mapping with success/total or failure keys",
                context={"method_id": method_id, "error": repr(exc)},
            ) from exc

    def _time_since_creation(self, task: Any) -> float:
        """
        Return the normalised age of ``task`` since its creation in ``[0, 1]``.

        Delegates to ``BaseHeuristics.extract_temporal_features``, which already
        handles ISO‑8601 strings, POSIX timestamps, timezone normalisation, and
        configuration‑driven normalisation (``base_heuristics.max_age_hours``).

        ``0.0`` means the task was just created (or has no creation time);
        ``1.0`` means the task is at or beyond the configured maximum age.

        Raises
        ------
        PlanningConfigError
            If ``creation_time`` is present but cannot be parsed, or an
            unexpected type/value is encountered while computing the feature.
        """
        try:
            temporal = self.extract_temporal_features(task)
            return clamp(float(temporal.get("time_since_creation", 0.0)), 0.0, 1.0)
        except PlanningError:
            raise
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            raise PlanningConfigError(
                f"Failed to compute time since creation: {exc}",
                config_key="creation_time",
                config_section="reinforcement_learning_heuristic",
                expected_type="ISO-8601 datetime or POSIX timestamp",
                context={"error": repr(exc)},
            ) from exc

    def _deadline_proximity(self, task: Any) -> float:
        """
        Return the normalised deadline proximity of ``task`` in ``[0, 1]``.

        ``0.0`` means the deadline is far away or not set; ``1.0`` means the
        deadline has already passed (or is imminent relative to the configured
        horizon).  Delegates to ``BaseHeuristics.extract_temporal_features`` so
        that the deadline horizon configuration
        (``base_heuristics.deadline_horizon_seconds``) stays consistent across
        every heuristic in the planner.

        Raises
        ------
        PlanningConfigError
            If ``deadline`` is present but cannot be parsed, or an unexpected
            type/value is encountered while computing the feature.
        """
        try:
            temporal = self.extract_temporal_features(task)
            return clamp(float(temporal.get("deadline_proximity", 0.0)), 0.0, 1.0)
        except PlanningError:
            raise
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            raise PlanningConfigError(
                f"Failed to compute deadline proximity: {exc}",
                config_key="deadline",
                config_section="reinforcement_learning_heuristic",
                expected_type="ISO-8601 datetime or POSIX timestamp",
                context={"error": repr(exc)},
            ) from exc


if __name__ == "__main__":
    print("\n=== Running Reinforcement Learning Heuristic Test ===\n")
    printer.status("Init", "Reinforcement Learning Heuristic initialized", "success")
    from datetime import timedelta, datetime

    # Initialize heuristic
    rl_heuristic = ReinforcementLearningHeuristic()
    
    # Create test scenario
    task = {
        "name": "navigate_to_room",
        "priority": 0.8,
        "goal_state": {"location": "room_1"},
        "parent": {"name": "root_task", "parent": None},
        "creation_time": datetime.now().isoformat(),
        "deadline": (datetime.now() + timedelta(hours=2)).isoformat()
    }
    state = {
        "location": "corridor",
        "cpu_available": 0.7,
        "memory_available": 0.6,
        "battery_level": 0.9
    }
    stats = {
        ("navigate_to_room", "A*"): {"success": 8, "total": 10},
        ("navigate_to_room", "RRT"): {"success": 6, "total": 10}
    }
    methods = ["A*", "RRT", "D*"]

    # Test method selection
    print("\n* * * * * Phase 1 - Method Selection * * * * *\n")
    selected_method, prob = rl_heuristic.select_best_method(task, state, methods, stats)  # Fixed method name
    printer.pretty(f"Selected method: {selected_method} (prob: {prob:.3f})", "", "Success")
    
    # Simulate execution outcome
    success = selected_method == "A*"  # Assume A* always succeeds in this test
    rl_heuristic.record_outcome(1.0 if success else -0.5)
    
    # Update policy
    print("\n* * * * * Phase 2 - Policy Update * * * * *\n")
    rl_heuristic.update_policy()
    
    # Test probability prediction
    print("\n* * * * * Phase 3 - Success Prediction * * * * *\n")
    for method in methods:
        prob = rl_heuristic.predict_success_prob(task, state, stats, method)
        printer.pretty(f"Success probability for {method}: {prob:.3f}", "", "Info")
    
    # Save model
    print("\n* * * * * Phase 4 - Model Saving * * * * *\n")
    rl_heuristic.save_model()
    
    print("\n=== Successfully Ran Reinforcement Learning Heuristic ===\n")
