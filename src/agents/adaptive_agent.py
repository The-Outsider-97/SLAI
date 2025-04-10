"""
Adaptive Agent with Reinforcement Learning Capabilities

This agent combines:
1. Reinforcement learning for self-improvement through experience
2. Memory systems for retaining knowledge
3. Adaptive routing for task delegation
4. Continuous learning from feedback and demonstrations

Key Features:
- Self-tuning learning parameters
- Multi-modal memory system
- Flexible task routing
- Integrated learning from various sources
- Minimal external dependencies

Academic References:
- Sutton & Barto (2018) - Reinforcement Learning: An Introduction
- Silver et al. (2014) - Deterministic Policy Gradient Algorithms
- Mnih et al. (2015) - Human-level control through deep reinforcement learning
- Schmidhuber (2015) - On Learning to Think
"""

import random
import math
import numpy as np
from collections import defaultdict, deque
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AdaptiveAgent:
    """
    An adaptive agent that combines reinforcement learning with memory and routing capabilities.
    Continuously improves from feedback, success/failure, and demonstrations.
    """
    
    def __init__(self, shared_memory, learning_params=None):
        """
        Initialize the adaptive agent with learning and memory systems.
        """
        self.shared_memory = shared_memory
        self.episodic_memory = deque(maxlen=1000)  # Recent experiences
        self.semantic_memory = defaultdict(dict)   # Conceptual knowledge
        self.learning_params = {
            'learning_rate': 0.01,
            'exploration_rate': 0.3,
            'discount_factor': 0.95,
            'temperature': 1.0,  # For softmax decision making
            'memory_capacity': 1000,
            **({} if learning_params is None else learning_params)
        }

        # Policy and value function representations
        self.policy_weights = self._initialize_weights()
        self.value_estimates = defaultdict(float)
        
        # Performance tracking
        self.performance_history = []
        self.last_reward = 0
        self.total_steps = 0
        
        # Skill library
        self.skills = {
            'basic_rl': self._basic_rl_skill,
            'memory_retrieval': self._memory_skill,
            'message_routing': self._routing_skill
        }
        
        logger.info("AdaptiveAgent initialized with parameters: %s", self.learning_params)
    
    def _initialize_weights(self):
        """Initialize policy weights with random values"""
        return {
            'input_layer': np.random.randn(10, 10) * 0.1,
            'hidden_layer': np.random.randn(10, 5) * 0.1,
            'output_layer': np.random.randn(5, 2) * 0.1
        }
    
    def execute(self, task_data):
        """
        Main execution interface for handling tasks.
        
        Args:
            task_data (dict): Task specification containing:
                - 'type': Task type (train, evaluate, route, etc.)
                - 'content': Task-specific data
        
        Returns:
            dict: Results of task execution
        """
        failures = self.shared_memory.get("agent_stats", {}).get(self.name, {}).get("errors", [])
        for err in failures:
            if self.is_similar(task_data, err["data"]):
                self.logger.info("Recognized a known problematic case, applying workaround.")
                return self.alternative_execute(task_data)
            
        errors = self.shared_memory.get(f"errors:{self.name}", [])
        for error in errors:
            if self.is_similar(task_data, error['task_data']):
                self.handle_known_issue(task_data, error)
                return
            
        # Proceed with normal execution
        try:
            result = self.perform_task(task_data)
            self.shared_memory.set(f"results:{self.name}", result)
        except Exception as e:
            # Log the failure in shared memory
            error_entry = {'task_data': task_data, 'error': str(e)}
            errors.append(error_entry)
            self.shared_memory.set(f"errors:{self.name}", errors)
            raise
    
        task_type = task_data.get('type', 'evaluate')
        
        if task_type == 'train':
            episodes = task_data.get('episodes', 10)
            self.train(episodes)
            return {"status": "training_complete", "episodes": episodes}
            
        elif task_type == 'evaluate':
            eval_episodes = task_data.get('eval_episodes', 5)
            avg_reward = self.evaluate(eval_episodes)
            return {"status": "evaluation_complete", "avg_reward": avg_reward}
            
        elif task_type == 'route':
            message = task_data.get('message', '')
            routing_table = task_data.get('routing_table', {})
            return self.route_message(message, routing_table)
            
        elif task_type == 'learn_from_demo':
            demonstration = task_data['demonstration']
            self.learn_from_demonstration(demonstration)
            return {"status": "demonstration_learned"}
            
        else:
            return {"status": "error", "message": "Unknown task type"}

    def is_similar(self, task_data, past_task_data):
        """
        Compares current task with past task to detect similarity.
        Uses key overlap and value resemblance heuristics.
        """
        if type(task_data) != type(past_task_data):
            return False

        # Handle simple text-based tasks
        if isinstance(task_data, str) and isinstance(past_task_data, str):
            return task_data.strip().lower() == past_task_data.strip().lower()

        # Handle dict-based structured tasks
        if isinstance(task_data, dict) and isinstance(past_task_data, dict):
            shared_keys = set(task_data.keys()) & set(past_task_data.keys())
            similarity_score = 0
            for key in shared_keys:
                if isinstance(task_data[key], str) and isinstance(past_task_data[key], str):
                    if task_data[key].strip().lower() == past_task_data[key].strip().lower():
                        similarity_score += 1
            # Consider similar if 50% or more keys match closely
            return similarity_score >= (len(shared_keys) / 2)

        return False
    
    def alternative_execute(self, task_data):
        """
        Fallback logic when normal execution fails or matches a known failure pattern.
        Attempts to simplify, sanitize, or reroute the input for safer processing.
        """
        try:
            # Step 1: Sanitize task data (remove noise, normalize casing, trim tokens)
            if isinstance(task_data, str):
                clean_data = task_data.strip().lower().replace('\n', ' ')
            elif isinstance(task_data, dict) and "text" in task_data:
                clean_data = task_data["text"].strip().lower()
            else:
                clean_data = str(task_data).strip()

            # Step 2: Apply a safer, simplified prompt or fallback logic
            fallback_prompt = f"Can you try again with simplified input:\n{clean_data}"
            if hasattr(self, "llm") and callable(getattr(self.llm, "generate", None)):
                return self.llm.generate(fallback_prompt)

            # Step 3: If the agent wraps another processor (e.g. GrammarProcessor, LLM), reroute
            if hasattr(self, "grammar") and callable(getattr(self.grammar, "compose_sentence", None)):
                facts = {"event": "fallback", "value": clean_data}
                return self.grammar.compose_sentence(facts)

            # Step 4: Otherwise just echo the cleaned input as confirmation
            return f"[Fallback response] I rephrased your input: {clean_data}"

        except Exception as e:
            # Final fallback — very safe and generic
            return "[Fallback failure] Unable to process your request at this time."
    
    def handle_known_issue(self, task_data, error):
        """
        Attempt to recover from known failure patterns.
        Could apply input transformation or fallback logic.
        """
        self.logger.warning(f"Handling known issue from error: {error.get('error')}")

        # Fallback strategy #1: remove problematic characters
        if isinstance(task_data, str):
            cleaned = task_data.replace("🧠", "").replace("🔥", "")
            self.logger.info(f"Retrying with cleaned input: {cleaned}")
            return self.perform_task(cleaned)

        # Fallback strategy #2: modify specific fields in structured input
        if isinstance(task_data, dict):
            cleaned_data = task_data.copy()
            for key, val in cleaned_data.items():
                if isinstance(val, str) and "emoji" in error.get("error", ""):
                    cleaned_data[key] = val.encode("ascii", "ignore").decode()
            self.logger.info("Retrying task with cleaned structured data.")
            return self.perform_task(cleaned_data)

        # Fallback strategy #3: return a graceful degradation response
        self.logger.warning("Returning fallback response for unresolvable input.")
        return {"status": "failed", "reason": "Repeated known issue", "fallback": True}

    def perform_task(self, task_data):
        """
        Simulated execution method — replace with actual agent logic.
        This is where core functionality would happen.
        """
        self.logger.info(f"Executing task with data: {task_data}")

        if isinstance(task_data, str) and "fail" in task_data.lower():
            raise ValueError("Simulated failure due to blacklisted word.")

        if isinstance(task_data, dict):
            # Simulate failure on missing required keys
            required_keys = ["input", "context"]
            for key in required_keys:
                if key not in task_data:
                    raise KeyError(f"Missing required key: {key}")

        # Simulate result
        return {"status": "success", "result": f"Processed: {task_data}"}

    def train(self, episodes=100):
        """
        Train the agent through interaction with the environment.
        
        Args:
            episodes (int): Number of training episodes
        """
        logger.info("Starting training for %d episodes", episodes)
        
        for episode in range(1, episodes + 1):
            total_reward = 0
            state = self._get_initial_state()
            
            while not self._is_terminal_state(state):
                action = self._select_action(state)
                next_state, reward = self._take_action(state, action)
                
                # Update the agent
                self._update_policy(state, action, reward, next_state)
                self._update_value_estimates(state, reward, next_state)
                
                # Adjust learning parameters
                self._adapt_parameters(reward)
                
                total_reward += reward
                state = next_state
            
            # Episode complete
            self.performance_history.append(total_reward)
            self._consolidate_memory()
            
            if episode % 10 == 0:
                logger.info("Episode %d - Total reward: %.2f", episode, total_reward)
        
        logger.info("Training completed. Average reward: %.2f", 
                   np.mean(self.performance_history[-10:]))
    
    def evaluate(self, eval_episodes=10):
        """
        Evaluate the agent's performance without learning.
        
        Args:
            eval_episodes (int): Number of evaluation episodes
            
        Returns:
            float: Average reward across episodes
        """
        logger.info("Evaluating agent over %d episodes", eval_episodes)
        rewards = []
        
        for _ in range(eval_episodes):
            state = self._get_initial_state()
            episode_reward = 0
            
            while not self._is_terminal_state(state):
                action = self._select_action(state, explore=False)
                state, reward = self._take_action(state, action)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        avg_reward = np.mean(rewards)
        logger.info("Evaluation complete. Average reward: %.2f", avg_reward)
        return avg_reward
    
    def learn_from_demonstration(self, demonstration):
        """
        Learn from a provided demonstration.
        
        Args:
            demonstration (list): Sequence of (state, action, reward) tuples
        """
        logger.info("Learning from demonstration with %d steps", len(demonstration))
        
        for state, action, reward in demonstration:
            # Update policy towards demonstrated actions
            self._update_policy_from_demo(state, action)
            
            # Update value estimates
            self.value_estimates[state] = (1 - self.learning_params['learning_rate']) * \
                                         self.value_estimates.get(state, 0) + \
                                         self.learning_params['learning_rate'] * reward
            
            # Store in memory
            self._store_experience(state, action, reward)
        
        logger.info("Demonstration learning complete")
    
    def _select_action(self, state, explore=True):
        """
        Select an action based on current policy.
        
        Args:
            state: Current state representation
            explore (bool): Whether to include exploration
            
        Returns:
            Action selected by the agent
        """
        if explore and random.random() < self.learning_params['exploration_rate']:
            # Exploration: random action
            action = random.choice([0, 1])
        else:
            # Exploitation: policy-based action
            policy_probs = self._compute_policy(state)
            action = 0 if random.random() < policy_probs[0] else 1
        
        return action
    
    def _compute_policy(self, state):
        """
        Compute action probabilities for a given state.
        
        Args:
            state: Current state representation
            
        Returns:
            list: Probability distribution over actions
        """
        # Simple linear policy computation (could be replaced with neural network)
        state_features = self._extract_features(state)
        hidden = np.tanh(np.dot(state_features, self.policy_weights['input_layer']))
        hidden = np.tanh(np.dot(hidden, self.policy_weights['hidden_layer']))
        logits = np.dot(hidden, self.policy_weights['output_layer'])
        
        # Softmax to get probabilities
        exp_logits = np.exp(logits / self.learning_params['temperature'])
        probs = exp_logits / np.sum(exp_logits)
        return probs
    
    def _update_policy(self, state, action, reward, next_state):
        """
        Update the policy based on experience.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
        """
        # Compute TD error
        td_error = reward + self.learning_params['discount_factor'] * \
                  self.value_estimates.get(next_state, 0) - \
                  self.value_estimates.get(state, 0)
        
        # Get policy gradient
        state_features = self._extract_features(state)
        hidden = np.tanh(np.dot(state_features, self.policy_weights['input_layer']))
        hidden = np.tanh(np.dot(hidden, self.policy_weights['hidden_layer']))
        grad = hidden * (1 if action == 0 else -1)  # Simplified gradient
        
        # Update weights
        update = self.learning_params['learning_rate'] * td_error * grad
        self.policy_weights['output_layer'] += update
        
        # Store experience
        self._store_experience(state, action, reward)
    
    def _update_value_estimates(self, state, reward, next_state):
        """
        Update value estimates using TD learning.
        
        Args:
            state: Previous state
            reward: Reward received
            next_state: Resulting state
        """
        current_value = self.value_estimates.get(state, 0)
        next_value = self.value_estimates.get(next_state, 0)
        
        td_target = reward + self.learning_params['discount_factor'] * next_value
        self.value_estimates[state] = current_value + \
                                    self.learning_params['learning_rate'] * \
                                    (td_target - current_value)
    
    def _update_policy_from_demo(self, state, action):
        """
        Update policy towards demonstrated action.
        
        Args:
            state: Demonstrated state
            action: Demonstrated action
        """
        state_features = self._extract_features(state)
        hidden = np.tanh(np.dot(state_features, self.policy_weights['input_layer']))
        hidden = np.tanh(np.dot(hidden, self.policy_weights['hidden_layer']))
        
        # Update towards demonstrated action
        target = np.zeros(2)
        target[action] = 1.0
        current = self._compute_policy(state)
        
        # Cross-entropy gradient
        grad = hidden[:, None] * (current - target)
        self.policy_weights['output_layer'] -= self.learning_params['learning_rate'] * grad
    
    def _adapt_parameters(self, reward):
        """
        Adapt learning parameters based on recent performance.
        """
        # Track recent rewards
        self.last_reward = reward
        self.total_steps += 1
        
        # Decay exploration
        self.learning_params['exploration_rate'] *= 0.9995
        self.learning_params['exploration_rate'] = max(0.01, self.learning_params['exploration_rate'])
        
        # Adjust learning rate based on performance variance
        if len(self.performance_history) > 10:
            recent_perf = self.performance_history[-10:]
            perf_variance = np.var(recent_perf)
            
            if perf_variance < 0.1:  # Low variance, decrease learning rate
                self.learning_params['learning_rate'] *= 0.995
            elif perf_variance > 1.0:  # High variance, increase learning rate
                self.learning_params['learning_rate'] *= 1.01
            
            # Keep within bounds
            self.learning_params['learning_rate'] = np.clip(
                self.learning_params['learning_rate'], 1e-4, 0.1)
    
    def _store_experience(self, state, action, reward):
        """
        Store experience in memory systems.
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'timestamp': time.time()
        }
        
        self.episodic_memory.append(experience)
        
        # Also store in semantic memory if significant
        if abs(reward) > 1.0:
            key = f"state_{hash(state) % 1000}"
            self.semantic_memory[key][action] = reward
    
    def _consolidate_memory(self):
        """
        Consolidate recent experiences into long-term memory.
        """
        if len(self.episodic_memory) > 0:
            recent_exp = self.episodic_memory[-1]  # Most recent experience
            state_key = f"state_{hash(recent_exp['state']) % 1000}"
            
            # Update semantic memory with important experiences
            if abs(recent_exp['reward']) > 0.8:
                self.semantic_memory[state_key][recent_exp['action']] = \
                    self.semantic_memory[state_key].get(recent_exp['action'], 0) * 0.9 + \
                    recent_exp['reward'] * 0.1
    
    def update_memory(self, key: str, value):
        """
        Update shared memory with a key-value pair.
        """
        self.shared_memory[key] = value
    
    def retrieve_memory(self, key: str):
        """
        Retrieve value from shared memory.
        """
        return self.shared_memory.get(key, None)
    
    def route_message(self, message: str, routing_table: dict):
        """
        Route a message to the appropriate handler based on routing table.
        
        Args:
            message (str): Message to route
            routing_table (dict): Mapping of conditions to handlers
            
        Returns:
            Response from the handler or error message
        """
        for condition, handler in routing_table.items():
            if condition in message:
                return handler(message)
        return "No suitable handler found for message"
    
    def _get_initial_state(self):
        """Generate an initial state for an episode"""
        return tuple(np.random.randint(0, 10, size=3))
    
    def _is_terminal_state(self, state):
        """Check if a state is terminal"""
        return sum(state) > 25  # Simple terminal condition
    
    def _take_action(self, state, action):
        """
        Simulate taking an action in the environment.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            tuple: (next_state, reward)
        """
        # Simple environment dynamics
        if action == 0:  # Action 0
            next_state = tuple(s + 1 for s in state)
            reward = -0.1 + random.random() * 0.2
        else:  # Action 1
            next_state = tuple(s * 1.5 for s in state)
            reward = sum(state) / 10 + random.random() * 0.5
        
        return next_state, reward
    
    def _extract_features(self, state):
        """Convert state into feature vector"""
        return np.array([
            state[0] / 10.0,
            state[1] / 10.0,
            state[2] / 10.0,
            sum(state) / 30.0,
            min(state) / 10.0,
            max(state) / 10.0,
            len([s for s in state if s > 5]) / 3.0,
            math.sqrt(sum(s**2 for s in state)) / 17.32,  # sqrt(300)
            (state[0] * state[1]) / 100.0,
            (state[1] * state[2]) / 100.0
        ])
    
    def _basic_rl_skill(self, state):
        """Basic RL skill implementation"""
        action = self._select_action(state)
        next_state, reward = self._take_action(state, action)
        self._update_policy(state, action, reward, next_state)
        return next_state, reward
    
    def _memory_skill(self, query):
        """Memory retrieval skill"""
        if query in self.shared_memory:
            return self.shared_memory[query]
        elif query in self.semantic_memory:
            return self.semantic_memory[query]
        else:
            # Search episodic memory
            for exp in reversed(self.episodic_memory):
                if query in str(exp['state']):
                    return exp
            return None
    
    def _routing_skill(self, message):
        """Message routing skill"""
        # Simplified routing for demonstration
        if "train" in message:
            return self.execute({'type': 'train', 'episodes': 5})
        elif "evaluate" in message:
            return self.execute({'type': 'evaluate'})
        else:
            return {"status": "unrecognized_message"}

# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = AdaptiveAgent({
        'learning_rate': 0.02,
        'exploration_rate': 0.2,
        'memory_capacity': 500
    })
    
    # Training example
    print("Starting training...")
    agent.execute({'type': 'train', 'episodes': 20})
    
    # Evaluation example
    print("\nEvaluating...")
    result = agent.execute({'type': 'evaluate', 'eval_episodes': 5})
    print("Evaluation result:", result)
    
    # Memory example
    print("\nTesting memory...")
    agent.update_memory("important_value", 42)
    print("Retrieved value:", agent.retrieve_memory("important_value"))
    
    # Routing example
    print("\nTesting routing...")
    routing_table = {
        "train": agent.execute,
        "eval": agent.execute
    }
    print("Routing result:", agent.route_message("train now", routing_table))
