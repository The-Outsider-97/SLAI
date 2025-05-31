"""
Proficient In:
    Rapid adaptation to new environments/tasks.
    Meta-RL setups (learning across many tasks).
    Communication-driven or multi-agent learning with structured policies.

Best Used When:
    The agent must generalize quickly to new, unseen tasks.
    Tasks are highly variable but share underlying structure.
    You need fast learning or transfer in changing environments.
"""
import gym
import yaml
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from collections import namedtuple, defaultdict

from src.agents.learning.utils.config_loader import load_global_config
from src.agents.learning.learning_memory import LearningMemory
from src.agents.learning.utils.policy_network import PolicyNetwork, NoveltyDetector
from logs.logger import get_logger

logger = get_logger("Model-Agnostic Meta-Learning")

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'log_prob', 'message'])

class MAMLAgent:
    def __init__(self, agent_id, state_size, action_size):
        self.config = load_global_config()

        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size

        # Retrieve MAML-specific parameters
        maml_config = self.config.get('maml', {})
        self.max_trajectory_steps = maml_config.get('max_trajectory_steps', 100)  # Default 100 steps
        self.gamma = maml_config.get('gamma', 0.99)
        self.meta_lr = maml_config.get('meta_lr', 0.001)
        self.inner_lr = maml_config.get('inner_lr', 0.01)

        # Initialize PolicyNetwork with merged config
        self.policy = PolicyNetwork(
            state_size=self.state_size,
            action_size=self.action_size
        )
        self.meta_optimizer = optim.Adam(self.policy.parameters(), lr=self.meta_lr)

        self.learning_memory = LearningMemory()
        self.model_id = "MAML_Agent"
        self._init_nlp(action_size)

        logger.info("Model-Agnostic Meta-Learning has successfully initialized")

    def _init_nlp(self, action_size):
        try:
            from src.agents.language.nlp_engine import NLPEngine
            NLP_ENGINE_AVAILABLE = True
        except ImportError:
            NLP_ENGINE_AVAILABLE = False
        maml_config = self.config.get('maml', {})
        self.vocab_size = maml_config.get('vocab_size', 50)
        self.max_message_length = maml_config.get('max_message_length', 10)

        # Initialize NLPEngine
        self.nlp_engine = None
        if NLP_ENGINE_AVAILABLE:
            nlp_config_path = maml_config.get('nlp_engine_config_path', "src/agents/language/configs/language_config.yaml")
            try:
                nlp_config_data = load_global_config(nlp_config_path)
                if nlp_config_data: # Ensure config was loaded
                    self.nlp_engine = NLPEngine(config=nlp_config_data)
                    logger.info(f"Agent {self.agent_id}: NLPEngine initialized from {nlp_config_path}")
                else:
                    logger.warning(f"Agent {self.agent_id}: NLPEngine config from {nlp_config_path} was empty. NLPEngine not initialized.")
            except FileNotFoundError:
                logger.warning(f"Agent {self.agent_id}: NLPEngine config not found at {nlp_config_path}. NLPEngine not initialized.")
            except Exception as e:
                logger.error(f"Agent {self.agent_id}: Error initializing NLPEngine: {e}. NLPEngine not initialized.")
        else:
            logger.warning(f"Agent {self.agent_id}: NLPEngine module not available. Communication features will be symbolic.")

        logger.info(f"MAMLAgent {self.agent_id} successfully initialized. Policy output size: {action_size}")

    def clone_policy(self, policy_to_clone=None):
        if policy_to_clone is None:
            policy_to_clone = self.policy
        
        cloned_policy = PolicyNetwork(
            state_size=self.state_size,
            action_size=self.action_size
        )
        cloned_policy.load_state_dict(policy_to_clone.state_dict())
        return cloned_policy

    def get_action(self, state, current_policy=None, is_speaking_task=False):
        if current_policy is None:
            current_policy = self.policy
        
        # Ensure state is a FloatTensor
        if not isinstance(state, torch.Tensor):
            state_array = np.array(state, dtype=np.float32)
            state_tensor = torch.FloatTensor(state_array)
        else:
            state_tensor = state.float()

        if state_tensor.ndim == 1: # If single state, unsqueeze to make it a batch of 1
            state_tensor = state_tensor.unsqueeze(0)
        
        probs = current_policy(state_tensor) # PolicyNetwork's forward method
        
        # If PolicyNetwork outputs logits for a Categorical distribution
        dist = torch.distributions.Categorical(logits=probs) # Use logits if output is not softmaxed
        # If PolicyNetwork output IS ALREADY probabilities (e.g. softmaxed)
        # dist = torch.distributions.Categorical(probs=probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        action_item = action.item()
        
        # Conceptual: if it's a speaking task, action_item is a token_id.
        # We might form a message from a sequence of such actions.
        # For now, action is just the item. Message part of trajectory can store generated message.
        message_token = action_item if is_speaking_task else None

        return action_item, log_prob, message_token

    def collect_trajectory(self, env, current_policy=None, is_speaking_task=False, partner_agent=None):
        # This is a simplified trajectory collection.
        # Real multi-agent communication would be more complex.
        if current_policy is None:
            current_policy = self.policy
            
        trajectory = []
        state, _ = env.reset() # Assuming standard Gym env
        
        # Conceptual message passing
        current_message_sequence = []

        for step in range(self.max_trajectory_steps):
            # If listener, state might need to incorporate message from speaker (partner_agent)
            # For simplicity, we assume env handles this or task is non-communicative for now.
            
            action, log_prob, msg_token = self.get_action(state, current_policy, is_speaking_task)
            
            if is_speaking_task and msg_token is not None:
                current_message_sequence.append(msg_token)
                # Speaker's "reward" might be delayed until listener acts, or based on message quality.
                # For simplicity, let's assume speaker gets a reward from env after forming full message.
                # Or, the environment step implicitly handles one token of speaking.
                # env.speak_token(msg_token) # Hypothetical
                if len(current_message_sequence) == self.max_message_length:
                    # Message complete, environment might transition or give reward
                    next_state, reward, done, truncated, _ = env.step(action) # action might be special "end_message" action
                    full_message = copy.deepcopy(current_message_sequence)
                    current_message_sequence = [] # Reset for next potential message
                else: # Mid-message
                    next_state, reward, done, truncated, _ = env.step_speaker(action) # Hypothetical step for speaker
                    full_message = None
            else: # Standard RL step or listener step
                next_state, reward, done, truncated, _ = env.step(action)
                full_message = None

            trajectory.append(Transition(state, action, reward, log_prob, full_message))
            state = next_state
            if done or truncated:
                break
        return trajectory

    def compute_loss_from_trajectory(self, trajectory, policy_to_evaluate):
        # policy_to_evaluate is used if we need to re-calculate log_probs for an adapted policy.
        # However, MAML typically calculates loss using log_probs stored during trajectory collection
        # by the policy *that generated them*. Here, we use stored log_probs.
        
        rewards = [t.reward for t in trajectory]
        log_probs = [t.log_prob for t in trajectory] # Use stored log_probs

        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted_rewards.insert(0, G)
        
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        # Normalize rewards (optional, but often helpful)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_loss = []
        for log_prob, G_t in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * G_t)
        
        return torch.stack(policy_loss).sum()

    def inner_update(self, env, current_meta_policy, is_speaking_task=False, partner_agent=None):
        """
        Performs the inner loop update of MAML (task-specific adaptation).
        Returns an adapted_policy.
        Inspired by Finn et al. (MAML) and conceptually by OML's PLN adaptation.
        """
        # Create a temporary policy for adaptation (clone of current_meta_policy)
        task_specific_policy = self.clone_policy(current_meta_policy)
        # task_specific_policy.train() # Set to train mode

        # Collect a trajectory with this task_specific_policy (could be current_meta_policy if adapting from scratch)
        trajectory = self.collect_trajectory(env, task_specific_policy, is_speaking_task, partner_agent)
        loss = self.compute_loss_from_trajectory(trajectory, task_specific_policy)

        # Compute gradients for the task_specific_policy's parameters
        # Gradients are computed w.r.t. task_specific_policy.parameters()
        # create_graph=True is essential for MAML's meta-optimization (differentiating through the update)
        grads = torch.autograd.grad(loss, task_specific_policy.parameters(), create_graph=True, retain_graph=True)
        
        # Perform gradient descent step to get adapted parameters
        # This is θ' = θ - α * ∇_θ L_task(θ)
        adapted_params = []
        for param, grad in zip(task_specific_policy.parameters(), grads):
            adapted_params.append(param - self.inner_lr * grad)

        # Create a new policy with these adapted parameters
        adapted_policy = self.clone_policy(current_meta_policy) # Start from meta-policy structure
        
        # Load adapted_params into adapted_policy.
        # This is tricky with state_dict if params are just a list.
        # Need to assign them back carefully or rebuild the policy.
        # For PolicyNetwork, Ws and bs are in nn.ParameterList.
        
        # Simpler: update the task_specific_policy with these new params and return it.
        # This requires task_specific_policy to have its parameters updated MANUALLY without an optimizer.
        # Let's re-do how adapted_policy is formed.
        
        final_adapted_policy = self.clone_policy(current_meta_policy)
        
        # Correct way to update parameters for the final_adapted_policy
        # We need to assign the computed `adapted_params` (which are tensors)
        # to the `final_adapted_policy.parameters()`.
        # Since `adapted_params` were derived from `task_specific_policy.parameters()`,
        # and `final_adapted_policy` is a clone of `current_meta_policy`,
        # the parameter correspondence should hold.

        idx = 0
        for param_group in final_adapted_policy.parameters(): # Ws then bs
            # param_group is a Parameter object
            # We need to assign data from the list adapted_params
            # This is somewhat risky if param order changes, but typical for PolicyNetwork
            # Ensure adapted_params is correctly ordered (Ws first, then bs, matching .parameters())
            param_group.data = adapted_params[idx].data # Assign data to avoid issues with graph
            idx += 1
            if idx >= len(adapted_params): break # Should match perfectly

        return final_adapted_policy

    def compute_meta_gradient_contribution(self, tasks_for_agent):
        """
        Computes the meta-loss for this agent and accumulates gradients on its self.policy.
        This is one part of the original meta_update, preparing for gradient diffusion or optimizer step.
        """
        meta_loss_for_agent = 0.0
        
        for (env, task_info) in tasks_for_agent: # task_info could specify if speaking/listening, partner
            is_speaking = task_info.get('is_speaking', False) if task_info else False
            partner = task_info.get('partner_agent', None) if task_info else None # Partner for communication

            # 1. Adapt: Perform inner update to get an adapted policy for this task
            #    The inner update should use self.policy (the current meta-parameters)
            adapted_policy = self.inner_update(env, self.policy, is_speaking, partner)
            # adapted_policy now contains parameters that are functions of self.policy's parameters (due to create_graph=True)

            # 2. Evaluate: Collect a new trajectory (query set) with the adapted_policy
            query_trajectory = self.collect_trajectory(env, adapted_policy, is_speaking, partner)
            
            # 3. Compute loss on the query set using the adapted_policy
            task_meta_loss = self.compute_loss_from_trajectory(query_trajectory, adapted_policy)
            meta_loss_for_agent += task_meta_loss
        
        if not tasks_for_agent: # Avoid division by zero if no tasks
             return 0.0

        meta_loss_for_agent /= len(tasks_for_agent)
        
        # Gradients are computed w.r.t. self.policy.parameters() because adapted_policy's parameters
        # trace back to self.policy.parameters() through the inner_lr update step.
        self.meta_optimizer.zero_grad()
        meta_loss_for_agent.backward() # Accumulates gradients in self.policy.parameters().grad
        
        # Gradient clipping (optional but good practice)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

        return meta_loss_for_agent.item()

    def meta_update(self, tasks, inner_steps=1):
        meta_loss = 0.0
        for env, _ in tasks:
            adapted_policy = self.inner_update(env)
            trajectory = self.collect_trajectory(env, adapted_policy)
            task_loss = self.compute_loss(trajectory, adapted_policy)
            meta_loss += task_loss

        meta_loss /= len(tasks)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.meta_optimizer.step()
        return meta_loss.item()

    def execute(self, task_data):
        tasks = task_data.get("tasks", [])
        loss = self.meta_update(tasks)
        return {"status": "success", "agent": "MAMLAgent", "meta_loss": loss}

    def train(self, num_meta_epochs=100, tasks_per_epoch=5, adaptation_steps=1):
        """
        Enhanced training with comprehensive reward system
        """
        logger.info(f"Starting meta-training with {num_meta_epochs} epochs")
        self.training_metrics = {
            'extrinsic_rewards': [],
            'intrinsic_rewards': [],
            'communication_success': [],
            'task_success_rate': [],
            'meta_loss': []
        }

        for epoch in range(num_meta_epochs):
            epoch_extrinsic = 0
            epoch_intrinsic = 0
            epoch_comm_success = 0
            epoch_task_success = 0
            total_meta_loss = torch.tensor(0.0, requires_grad=True)

            for _ in range(tasks_per_epoch):
                # Sample a new task/environment
                env = self._sample_training_task()
                task_type = getattr(env, 'task_type', 'default')
                
                # Inner adaptation loop
                adapted_policy = self.policy
                for step in range(adaptation_steps):
                    trajectory = self.collect_trajectory(env, adapted_policy)
                    loss = self.compute_loss_from_trajectory(trajectory, adapted_policy)
                    adapted_policy = self.inner_update(env, adapted_policy)

                # Collect post-adaptation trajectory for meta-update
                meta_trajectory = self.collect_trajectory(env, adapted_policy)
                task_metrics = self._compute_task_metrics(meta_trajectory, env)
                
                # Accumulate rewards and metrics
                epoch_extrinsic += task_metrics['extrinsic_reward']
                epoch_intrinsic += task_metrics['intrinsic_reward']
                epoch_comm_success += task_metrics['communication_success']
                epoch_task_success += task_metrics['task_success']
                
                # Compute and accumulate meta-loss
                meta_loss = self.compute_loss_from_trajectory(meta_trajectory, adapted_policy)
                total_meta_loss = total_meta_loss + meta_loss

            # Update meta-policy
            self.meta_optimizer.zero_grad()
            total_meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.meta_optimizer.step()

            # Log epoch metrics
            self.training_metrics['extrinsic_rewards'].append(epoch_extrinsic/tasks_per_epoch)
            self.training_metrics['intrinsic_rewards'].append(epoch_intrinsic/tasks_per_epoch)
            self.training_metrics['communication_success'].append(epoch_comm_success/tasks_per_epoch)
            self.training_metrics['task_success_rate'].append(epoch_task_success/tasks_per_epoch)
            self.training_metrics['meta_loss'].append(total_meta_loss/tasks_per_epoch)

            logger.info(f"Epoch {epoch+1}/{num_meta_epochs} | "
                       f"Extrinsic: {self.training_metrics['extrinsic_rewards'][-1]:.2f} | "
                       f"Intrinsic: {self.training_metrics['intrinsic_rewards'][-1]:.2f} | "
                       f"Comm Success: {self.training_metrics['communication_success'][-1]:.2f}")

        logger.info("Meta-training complete")
        return self.training_metrics

    def evaluate(self, num_eval_tasks=20, adaptation_steps=3):
        """
        Comprehensive evaluation with multiple performance metrics
        """
        logger.info(f"Starting evaluation on {num_eval_tasks} tasks")
        evaluation_metrics = {
            'average_return': 0,
            'success_rate': 0,
            'communication_accuracy': 0,
            'adaptation_speed': [],
            'reward_components': {
                'extrinsic': [],
                'intrinsic': [],
                'task_completion': [],
                'communication': []
            }
        }

        for _ in range(num_eval_tasks):
            env = self._sample_evaluation_task()
            task_type = getattr(env, 'task_type', 'default')
            adaptation_rewards = []

            # Initial policy performance
            baseline_trajectory = self.collect_trajectory(env, self.policy)
            baseline_return = sum(t.reward for t in baseline_trajectory)
            
            # Adaptation process
            adapted_policy = self.policy
            for step in range(adaptation_steps):
                trajectory = self.collect_trajectory(env, adapted_policy)
                loss = self.compute_loss_from_trajectory(trajectory, adapted_policy)
                adapted_policy = self.inner_update(env, adapted_policy)
                adaptation_rewards.append(sum(t.reward for t in trajectory))

            # Post-adaptation performance
            final_trajectory = self.collect_trajectory(env, adapted_policy)
            final_metrics = self._compute_task_metrics(final_trajectory, env)
            
            # Update metrics
            evaluation_metrics['average_return'] += sum(t.reward for t in final_trajectory)
            evaluation_metrics['success_rate'] += final_metrics['task_success']
            evaluation_metrics['communication_accuracy'] += final_metrics['communication_success']
            evaluation_metrics['adaptation_speed'].append(
                (sum(adaptation_rewards) - baseline_return) / adaptation_steps
            )
            
            # Record reward components
            evaluation_metrics['reward_components']['extrinsic'].append(final_metrics['extrinsic_reward'])
            evaluation_metrics['reward_components']['intrinsic'].append(final_metrics['intrinsic_reward'])
            evaluation_metrics['reward_components']['task_completion'].append(final_metrics['task_completion_bonus'])
            evaluation_metrics['reward_components']['communication'].append(final_metrics['communication_bonus'])

        # Normalize metrics
        evaluation_metrics['average_return'] /= num_eval_tasks
        evaluation_metrics['success_rate'] /= num_eval_tasks
        evaluation_metrics['communication_accuracy'] /= num_eval_tasks
        evaluation_metrics['adaptation_speed'] = np.mean(evaluation_metrics['adaptation_speed'])
        
        for component in evaluation_metrics['reward_components']:
            evaluation_metrics['reward_components'][component] = np.mean(
                evaluation_metrics['reward_components'][component]
            )

        logger.info(f"Evaluation complete. Success rate: {evaluation_metrics['success_rate']:.2f}")
        return evaluation_metrics

    def _compute_task_metrics(self, trajectory, env):
        """
        Calculate comprehensive reward components and task success
        """
        metrics = {
            'extrinsic_reward': sum(t.reward for t in trajectory),
            'intrinsic_reward': 0,
            'communication_success': 0,
            'task_success': 0,
            'task_completion_bonus': 0,
            'communication_bonus': 0
        }

        # Intrinsic reward (curiosity-based)
        states = [t.state for t in trajectory]
        novelty_bonus = self._calculate_novelty(states)
        metrics['intrinsic_reward'] = 0.1 * novelty_bonus  # Scale intrinsic reward

        # Communication success (if applicable)
        if hasattr(env, 'communication_success'):
            comm_success = env.communication_success
            metrics['communication_success'] = comm_success
            metrics['communication_bonus'] = 2.0 * comm_success

        # Task success and completion bonus
        if hasattr(env, "task_completed") and env.task_completed():
            metrics['task_success'] = 1
            metrics['task_completion_bonus'] = 5.0

        # Combine all reward components
        adjusted_trajectory = []
        for t in trajectory:
            adjusted_reward = t.reward + (
                novelty_bonus / len(trajectory) +
                metrics['communication_bonus'] / len(trajectory) +
                metrics['task_completion_bonus'] / len(trajectory)
            )
            adjusted_trajectory.append(Transition(t.state, t.action, adjusted_reward, t.log_prob, t.message))
        trajectory[:] = adjusted_trajectory 

        return metrics

    def _calculate_novelty(self, states):
        """
        Calculate novelty bonus using random network distillation
        """
        if not hasattr(self, 'nd_network'):
            self.nd_network = NoveltyDetector(self.state_size)
        
        states_tensor = torch.stack([torch.FloatTensor(s) for s in states])
        return self.nd_network(states_tensor).mean().item()

    def _sample_training_task(self):
        """Override with task sampling logic"""
        return gym.make("CartPole-v1")  # Placeholder

    def _sample_evaluation_task(self):
        """Override with evaluation task sampling"""
        return gym.make("CartPole-v1")  # Placeholder

class DecentralizedMAMLFleet:
    def __init__(self, num_agents, global_config, env_creator_fn, state_size, action_size):
        self.num_agents = num_agents
        self.global_config = global_config
        self.agents = [MAMLAgent(i, state_size, action_size) for i in range(num_agents)]
        self.env_creator_fn = env_creator_fn # Function to create/sample environments/tasks

        maml_config = global_config.get('maml', {})
        self.diffusion_type = maml_config.get('diffusion_type', 'average') # e.g., 'average', 'weighted'
        self.meta_epochs = maml_config.get('meta_epochs', 100)
        self.tasks_per_agent_meta_batch = maml_config.get('tasks_per_agent_meta_batch', 5)

        # Adjacency matrix for diffusion (simple: fully connected for averaging)
        # For 'average' diffusion_type, can compute on the fly. For 'weighted', this would be pre-defined.
        self.adj_matrix = self._setup_adjacency(maml_config.get('adjacency_matrix', None))
        
        self.fleet_rewards_log = []
        logger.info(f"DecentralizedMAMLFleet initialized with {num_agents} agents.")

    def _setup_adjacency(self, adj_matrix_config):
        if adj_matrix_config == "fully_connected" or adj_matrix_config is None:
            # For simple average, each agent gets 1/N from everyone including self
            adj = np.ones((self.num_agents, self.num_agents)) / self.num_agents
        elif isinstance(adj_matrix_config, list): # Expected to be a list of lists
            adj = np.array(adj_matrix_config)
            # Normalize rows to sum to 1 (for weighted averaging - Metropolis-Hastings like)
            # Ensure adj is stochastic (e.g., row-stochastic for diffusion)
            # This needs proper definition based on desired diffusion (e.g. Dif-MAML paper eq. 7b's `a_lk`)
            # For now, let's assume if provided, it's correctly defined.
            # For Dif-MAML like updates, combination weights should be doubly stochastic or satisfy specific properties.
        else: # Default to fully connected if config is unclear
             adj = np.ones((self.num_agents, self.num_agents)) / self.num_agents
        
        # Ensure square matrix
        if adj.shape != (self.num_agents, self.num_agents):
            raise ValueError("Adjacency matrix shape mismatch.")
        return torch.FloatTensor(adj)


    def train_fleet(self):
        logger.info("Starting decentralized meta-training...")

        for epoch in range(self.meta_epochs):
            candidate_policy_params_list = [None] * self.num_agents

            # --- Local Meta-Update Phase (Simulating Dif-MAML's Adapt step to get φ_k) ---
            total_meta_loss_epoch = 0
            for i, agent in enumerate(self.agents):
                # Sample tasks for this agent
                # For communication tasks, task_info might specify partners
                # Example: env_creator_fn could yield (env, {'is_speaking': True, 'partner_idx': (i+1)%self.num_agents})
                agent_tasks = [self.env_creator_fn() for _ in range(self.tasks_per_agent_meta_batch)]
                
                # 1. Compute meta-gradient contribution for agent i
                # This populates agent.policy.parameters().grad
                meta_loss_val = agent.compute_meta_gradient_contribution(agent_tasks)
                total_meta_loss_epoch += meta_loss_val
                
                # 2. Simulate one step of the agent's meta-optimizer to get candidate parameters (φ_k)
                # These are the parameters agent_i *would* have if it updated independently.
                temp_policy_for_candidate = agent.clone_policy(agent.policy)
                temp_optimizer_for_candidate = optim.Adam(temp_policy_for_candidate.parameters(), lr=agent.meta_lr)
                
                # Copy gradients from agent.policy to temp_policy_for_candidate
                for p_orig, p_temp in zip(agent.policy.parameters(), temp_policy_for_candidate.parameters()):
                    if p_orig.grad is not None:
                        p_temp.grad = p_orig.grad.clone().detach() # Detach crucial
                
                temp_optimizer_for_candidate.step() # Updates temp_policy_for_candidate's params
                candidate_policy_params_list[i] = temp_policy_for_candidate.state_dict()

            avg_meta_loss = total_meta_loss_epoch / self.num_agents
            logger.info(f"Epoch {epoch+1}/{self.meta_epochs}, Avg Meta-Loss: {avg_meta_loss:.4f}")
            self.fleet_rewards_log.append(avg_meta_loss) # Using meta-loss as a proxy for progress

            # --- Diffusion/Combination Phase (Dif-MAML's Combine step) ---
            new_agent_policies_params = [{} for _ in range(self.num_agents)]

            # Aggregate parameters based on diffusion_type and adj_matrix
            # Example: Simple averaging if fully_connected
            # More sophisticated: Use self.adj_matrix for weighted averaging (a_lk * φ_l)
            
            # Get keys from the first state_dict (assuming all are same structure)
            if not candidate_policy_params_list or candidate_policy_params_list[0] is None:
                logger.error("No candidate policy parameters generated. Skipping diffusion.")
                continue
            
            param_keys = candidate_policy_params_list[0].keys()

            for k_idx in range(self.num_agents): # For each agent k that is updating
                current_agent_new_params = {}
                for key in param_keys: # For each parameter tensor (e.g., 'Ws.0.weight', 'bs.0.bias')
                    
                    # Weighted sum for parameter key: Σ_l a_kl * φ_l[key]
                    # where a_kl is self.adj_matrix[k_idx, l_idx]
                    # and φ_l[key] is candidate_policy_params_list[l_idx][key]
                    
                    sum_weighted_param_tensor = torch.zeros_like(candidate_policy_params_list[0][key])
                    for l_idx in range(self.num_agents):
                        weight_kl = self.adj_matrix[k_idx, l_idx]
                        param_tensor_l = candidate_policy_params_list[l_idx][key]
                        sum_weighted_param_tensor += weight_kl * param_tensor_l
                    
                    current_agent_new_params[key] = sum_weighted_param_tensor
                new_agent_policies_params[k_idx] = current_agent_new_params

            # Update each agent's policy and reset its optimizer
            for i, agent in enumerate(self.agents):
                agent.policy.load_state_dict(new_agent_policies_params[i])
                # Optimizer state is now inconsistent with the new parameters.
                # Re-initialize optimizer (simplest) or explore optimizer state averaging (complex).
                agent.meta_optimizer = optim.Adam(agent.policy.parameters(), lr=agent.meta_lr)
        
        logger.info("Decentralized meta-training complete.")
        return self.fleet_rewards_log

    def evaluate_fleet(self, num_eval_tasks_per_agent=10):
        logger.info("Evaluating decentralized fleet...")
        all_agent_avg_rewards = []
        for agent in self.agents:
            agent_total_reward = 0
            for _ in range(num_eval_tasks_per_agent):
                env, task_info = self.env_creator_fn()
                is_speaking = task_info.get('is_speaking', False) if task_info else False
                partner = task_info.get('partner_agent', None) if task_info else None

                # For evaluation, agent adapts its current meta-policy (self.policy)
                adapted_policy = agent.inner_update(env, agent.policy, is_speaking, partner)
                
                # Collect trajectory with adapted policy
                eval_trajectory = agent.collect_trajectory(env, adapted_policy, is_speaking, partner)
                task_reward = sum(t.reward for t in eval_trajectory)
                agent_total_reward += task_reward
            
            avg_reward = agent_total_reward / num_eval_tasks_per_agent
            all_agent_avg_rewards.append(avg_reward)
            logger.info(f"Agent {agent.agent_id} - Avg Eval Reward: {avg_reward:.2f}")
            
        overall_avg_reward = np.mean(all_agent_avg_rewards)
        logger.info(f"Fleet Overall Avg Eval Reward: {overall_avg_reward:.2f}")
        return {"overall_average_reward": overall_avg_reward, "agent_rewards": all_agent_avg_rewards}

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Model-Agnostic Meta-Learning ===\n")

    config = load_global_config()
    agent_id = None

    agent = MAMLAgent(
        state_size=4,
        action_size=2,
        agent_id=agent_id
    )
    training_metrics = agent.train(num_meta_epochs=20)
    evaluation_metrics = agent.evaluate(num_eval_tasks=50)
    print(f"\n{agent}\n")
    print("\n=== Successfully Ran Model-Agnostic Meta-Learning ===\n")

if __name__ == "__main__":
    print("\n * * * * Phase 2 * * * *\n=== Running Model-Agnostic Meta-Learning (Decentralized) ===\n")
    try:
        NLP_ENGINE_AVAILABLE = True
    except ImportError:
        NLP_ENGINE_AVAILABLE = False
    # Dummy environment creator for testing
    def simple_env_creator():
        # In a real scenario, this would create or sample a specific task environment
        # For now, let's use CartPole as a placeholder
        # Task_info can be used to specify communication roles, etc.
        try:
            env = gym.make("CartPole-v1")
        except gym.error.NameNotFound: # Fallback if gym is not fully installed or specific env missing
            class DummyEnv:
                def __init__(self):
                    self.action_space = gym.spaces.Discrete(2)
                    self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
                def reset(self): return self.observation_space.sample(), {}
                def step(self, action): return self.observation_space.sample(), np.random.rand(), np.random.choice([True, False]), False, {}
                def step_speaker(self,action): return self.step(action) # Placeholder
            env = DummyEnv()

        task_info = {'type': 'non_communicative'} # Example task_info
        return env, task_info

    # Load main configuration
    config = load_global_config()

    # MAML specific parameters from config
    maml_config = config.get('maml', {})
    num_agents_config = maml_config.get('num_agents', 5) # Default to 3 agents
    
    # For CartPole example
    example_state_size = 4 
    example_action_size = 2

    # If using NLPEngine for communication, ensure language_config.yaml exists.
    # Create a dummy language_config.yaml if NLPEngine is used and it's missing
    if NLP_ENGINE_AVAILABLE:
        nlp_config_path_main = maml_config.get('nlp_engine_config_path', "src/agents/language/configs/language_config.yaml")
        import os
        nlp_conf_dir = os.path.dirname(nlp_config_path_main)
        if not os.path.exists(nlp_conf_dir):
            os.makedirs(nlp_conf_dir, exist_ok=True)
        if not os.path.exists(nlp_config_path_main):
            dummy_nlp_conf = {
                "nlu": {
                    "sentiment_lexicon_path": "src/agents/language/templates/sentiment_lexicon.json" # Needs actual dummy file or handle missing
                }
            }
            with open(nlp_config_path_main, 'w') as f:
                yaml.dump(dummy_nlp_conf, f)
            logger.info(f"Created dummy NLP config at {nlp_config_path_main}")
            # Create dummy lexicon if path is specified
            dummy_lex_path = dummy_nlp_conf["nlu"]["sentiment_lexicon_path"]
            dummy_lex_dir = os.path.dirname(dummy_lex_path)
            if not os.path.exists(dummy_lex_dir) and dummy_lex_dir: os.makedirs(dummy_lex_dir, exist_ok=True)
            if not os.path.exists(dummy_lex_path) and dummy_lex_dir:
                 with open(dummy_lex_path, 'w') as f_lex:
                    import json
                    json.dump({"negators": ["not"], "intensifiers": {"very": 1.5}, "positive": {"good": 1}, "negative": {"bad": -1}}, f_lex)

    # Initialize the fleet
    fleet = DecentralizedMAMLFleet(
        num_agents=num_agents_config,
        global_config=config,
        env_creator_fn=simple_env_creator,
        state_size=example_state_size,
        action_size=example_action_size 
    )
    
    # Train the fleet
    training_log = fleet.train_fleet()
    print(f"\nTraining meta_loss log: {training_log}")

    # Evaluate the fleet
    eval_results = fleet.evaluate_fleet()
    print(f"\nEvaluation results: {eval_results}")

    # Example of how one agent might use NLPEngine (if available and initialized)
    if fleet.agents and fleet.agents[0].nlp_engine:
        sample_text = "This is a test sentence for the NLP engine."
        tokens = fleet.agents[0].nlp_engine.process_text(sample_text)
        print(f"\nNLPEngine test on agent 0 for text: '{sample_text}'")
        for token in tokens:
            print(f"  Token: {token.text}, POS: {token.pos}, Lemma: {token.lemma}")
    else:
        print("\nNLPEngine not available or not initialized in agent 0 for testing.")

    print("\n=== Successfully Ran Model-Agnostic Meta-Learning ===\n")
