"""Production-ready MAML agent for meta-reinforcement learning.

Key reference:
- Finn et al. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.

This module provides:
- A policy-gradient MAML agent for discrete-action meta-RL tasks.
- Differentiable inner-loop adaptation with optional first-order approximation.
- Robust trajectory collection across Gym and Gymnasium-style APIs.
- Evaluation, checkpointing, novelty-based intrinsic reward analysis, and task sampling.
- A decentralized fleet wrapper for diffusion-style multi-agent meta-learning.
"""

from __future__ import annotations

import copy
import inspect
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import OrderedDict, defaultdict, namedtuple
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple, Union
from torch.func import functional_call

try:  # pragma: no cover - runtime dependency may vary by deployment
    import gymnasium as gym
except Exception:  # pragma: no cover - defensive fallback for legacy stacks
    gym = None

# NumPy 2.x compatibility for legacy Gym internals that still reference np.bool8.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from src.agents.learning.learning_memory import LearningMemory
from src.agents.learning.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.utils.policy_network import (
    NoveltyDetector,
    PolicyNetwork,
    create_policy_network,
    create_policy_optimizer,
)
from src.agents.learning.utils.error_calls import (
        GradientExplosionError,
        InvalidActionError,
        InvalidConfigError,
        NaNException,
    )
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Model-Agnostic Meta-Learning")
printer = PrettyPrinter

TensorLike = Union[torch.Tensor, np.ndarray, Sequence[float]]
TaskSpec = Union[Any, Tuple[Any, Dict[str, Any]], Dict[str, Any]]
Transition = namedtuple("Transition", ["state", "action", "reward", "log_prob", "message"])


def _state_to_tensor(state: TensorLike, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert an arbitrary state-like input into a 1D float tensor."""
    if isinstance(state, torch.Tensor):
        tensor = state.detach().clone().to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(state, dtype=torch.float32)

    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim > 1:
        tensor = tensor.reshape(-1)

    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def _safe_mean(values: Sequence[Union[int, float]]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: Sequence[Union[int, float]]) -> float:
    return float(np.std(values)) if values else 0.0


def _coerce_bool(value: Any) -> bool:
    return bool(value() if callable(value) else value)


class MAMLAgent:
    """Model-Agnostic Meta-Learning agent using a policy-gradient objective.

    The implementation targets discrete-action environments, which matches the
    provided ``PolicyNetwork`` interface and the original module's action flow.
    """

    def __init__(
        self,
        agent_id: Optional[Union[str, int]],
        state_size: int,
        action_size: int,
        config: Optional[Dict[str, Any]] = None,
        task_sampler: Optional[Callable[..., TaskSpec]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if int(state_size) <= 0 or int(action_size) <= 0:
            raise InvalidConfigError("state_size and action_size must be positive integers.")

        self.agent_id = str(agent_id) if agent_id is not None else "MAML"
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.config = load_global_config()
        self.maml_config = get_config_section("maml")
        if config is not None and not isinstance(config, dict):
            raise InvalidConfigError("config must be a dictionary when provided.")

        self.config_override = config or {}
        self.maml_override = self.config_override.get("maml", {})
        if self.maml_override and not isinstance(self.maml_override, dict):
            raise InvalidConfigError("config['maml'] must be a dictionary when provided.")

        self.model_id = "MAML_Agent"
        self.task_sampler = task_sampler
        self.policy_config = self._build_policy_config()

        maml_get = lambda key, default: self.maml_override.get(key, self.maml_config.get(key, default))

        self.gamma = float(maml_get("gamma", 0.99))
        self.meta_lr = float(maml_get("meta_lr", 1e-3))
        self.inner_lr = float(maml_get("inner_lr", 1e-2))
        self.inner_steps = int(maml_get("inner_steps", 1))
        self.support_episodes = int(maml_get("support_episodes", 1))
        self.query_episodes = int(maml_get("query_episodes", 1))
        self.max_trajectory_steps = int(maml_get("max_trajectory_steps", 500))
        self.entropy_coef = float(maml_get("entropy_coef", 0.0))
        self.normalize_returns = bool(maml_get("normalize_returns", True))
        self.first_order = bool(maml_get("first_order", False))
        self.grad_clip_norm = float(maml_get("grad_clip_norm", 1.0))
        self.gradient_explosion_threshold = float(maml_get("gradient_explosion_threshold", 1e3))
        self.eval_deterministic = bool(maml_get("eval_deterministic", False))
        self.use_reward_shaping = bool(maml_get("use_reward_shaping", False))
        self.novelty_coef = float(maml_get("novelty_coef", 0.1))
        self.communication_bonus_coef = float(maml_get("communication_bonus_coef", 2.0))
        self.task_completion_bonus_coef = float(maml_get("task_completion_bonus_coef", 5.0))
        self.train_novelty_detector = bool(maml_get("train_novelty_detector", True))
        self.novelty_feature_dim = int(maml_get("novelty_feature_dim", max(16, int(maml_get("hidden_size", 64)))))
        self.checkpoint_dir = Path(maml_get("checkpoint_dir", "src/agents/learning/checkpoints/maml"))
        self.training_metrics: Dict[str, List[float]] = defaultdict(list)
        self.last_meta_metrics: Optional[Dict[str, Any]] = None

        if not 0.0 < self.gamma <= 1.0:
            raise InvalidConfigError("maml.gamma must be in (0, 1].")
        if self.meta_lr <= 0.0 or self.inner_lr <= 0.0:
            raise InvalidConfigError("meta_lr and inner_lr must be positive.")
        if self.inner_steps <= 0:
            raise InvalidConfigError("inner_steps must be positive.")
        if self.support_episodes <= 0 or self.query_episodes <= 0:
            raise InvalidConfigError("support_episodes and query_episodes must be positive.")
        if self.max_trajectory_steps <= 0:
            raise InvalidConfigError("max_trajectory_steps must be positive.")
        if self.grad_clip_norm <= 0.0:
            raise InvalidConfigError("grad_clip_norm must be positive.")

        self.policy = create_policy_network(
            input_dim=self.state_size,
            output_dim=self.action_size,
            config=self.policy_config,
        ).to(self.device)
        self.meta_optimizer = self._build_meta_optimizer(self.policy)
        self.learning_memory = LearningMemory()
        self.nd_network = NoveltyDetector(
            input_dim=self.state_size,
            feature_dim=self.novelty_feature_dim,
            learning_rate=float(self.maml_config.get("novelty_lr", 1e-3)),
            hidden_sizes=self.maml_config.get("novelty_hidden_sizes"),
            activation=str(self.maml_config.get("novelty_activation", "relu")),
            gradient_clip_norm=self.maml_config.get("novelty_grad_clip_norm"),
        ).to(self.device)

        self._init_nlp(self.action_size)
        logger.info(
            "Model-Agnostic Meta-Learning agent initialised | id=%s state=%s actions=%s meta_lr=%s inner_lr=%s",
            self.agent_id,
            self.state_size,
            self.action_size,
            self.meta_lr,
            self.inner_lr,
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _build_policy_config(self) -> Dict[str, Any]:
        base_policy_cfg = self.config.get("policy_network", {})
        policy_cfg = dict(base_policy_cfg) if isinstance(base_policy_cfg, dict) else {}

        policy_override = self.config_override.get("policy_network", {})
        if policy_override:
            if not isinstance(policy_override, dict):
                raise InvalidConfigError("config['policy_network'] must be a dictionary when provided.")
            for key, value in policy_override.items():
                if key == "optimizer_config" and isinstance(value, dict):
                    optimizer_cfg = dict(policy_cfg.get("optimizer_config", {}))
                    optimizer_cfg.update(value)
                    policy_cfg["optimizer_config"] = optimizer_cfg
                else:
                    policy_cfg[key] = value

        hidden_size = self.maml_override.get("hidden_size", self.maml_config.get("hidden_size", 64))
        if "hidden_layer_sizes" not in policy_cfg:
            if isinstance(hidden_size, int):
                policy_cfg["hidden_layer_sizes"] = [int(hidden_size), int(hidden_size)]
            elif isinstance(hidden_size, (list, tuple)):
                policy_cfg["hidden_layer_sizes"] = [int(dim) for dim in hidden_size]
            else:
                raise InvalidConfigError("maml.hidden_size must be an int or a sequence of ints.")
        policy_cfg["output_activation"] = "softmax"
        optimizer_cfg = dict(policy_cfg.get("optimizer_config", {}))
        optimizer_cfg["learning_rate"] = self.maml_override.get("meta_lr", self.maml_config.get("meta_lr", 1e-3))
        policy_cfg["optimizer_config"] = optimizer_cfg
        return policy_cfg

    def _build_meta_optimizer(self, model: nn.Module) -> optim.Optimizer:
        optimizer_cfg = dict(self.policy_config)
        nested_optimizer_cfg = dict(optimizer_cfg.get("optimizer_config", {}))
        nested_optimizer_cfg["learning_rate"] = self.meta_lr
        optimizer_cfg["optimizer_config"] = nested_optimizer_cfg
        return create_policy_optimizer(model, optimizer_cfg)

    def _init_nlp(self, action_size: int) -> None:
        try:  # pragma: no cover - optional subsystem
            from src.agents.language.nlp_engine import NLPEngine

            self.nlp_engine = NLPEngine()
        except Exception:
            self.nlp_engine = None

        self.vocab_size = int(self.maml_config.get("vocab_size", max(action_size, 50)))
        self.max_message_length = int(self.maml_config.get("max_message_length", 10))
        logger.info(
            "MAMLAgent %s initialised. Policy output size=%s NLP=%s",
            self.agent_id,
            action_size,
            self.nlp_engine is not None,
        )

    # ------------------------------------------------------------------
    # Environment / task handling
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_reset(env: Any) -> Any:
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            return reset_result[0]
        return reset_result

    @staticmethod
    def _safe_step(env: Any, action: int, step_method: str = "step") -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        step_fn = getattr(env, step_method)
        step_result = step_fn(action)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
            return next_state, float(reward), bool(terminated), bool(truncated), info or {}
        if isinstance(step_result, tuple) and len(step_result) == 4:
            next_state, reward, done, info = step_result
            return next_state, float(reward), bool(done), False, info or {}
        raise ValueError(
            f"Unexpected step output from '{step_method}': expected tuple length 4 or 5, got {type(step_result)}"
        )

    def _resolve_task_spec(
        self,
        task: Optional[TaskSpec] = None,
        *,
        split: str = "train",
        sampler: Optional[Callable[..., TaskSpec]] = None,
    ) -> Tuple[Any, Dict[str, Any], bool]:
        """Resolve a task specification into ``(env, task_info, was_sampled)``."""
        if task is None:
            sampler = sampler or self.task_sampler
            if sampler is None:
                raise InvalidConfigError(
                    f"No task provided and no task sampler configured for split='{split}'."
                )

            try:
                signature = inspect.signature(sampler)
                task = sampler(split=split) if "split" in signature.parameters else sampler()
            except (TypeError, ValueError):
                task = sampler()
            sampled = True
        else:
            sampled = False

        if isinstance(task, dict) and "env" in task:
            env = task["env"]
            task_info = copy.deepcopy(task.get("task_info", {}))
            return env, task_info, sampled

        if isinstance(task, tuple) and len(task) == 2 and isinstance(task[1], dict):
            env, task_info = task
            return env, copy.deepcopy(task_info), sampled

        return task, {}, sampled

    @staticmethod
    def _close_env_if_possible(env: Any) -> None:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:  # pragma: no cover - cleanup should never fail loudly
                pass

    def _sample_training_task(self) -> Tuple[Any, Dict[str, Any], bool]:
        return self._resolve_task_spec(None, split="train")

    def _sample_evaluation_task(self) -> Tuple[Any, Dict[str, Any], bool]:
        return self._resolve_task_spec(None, split="eval")

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------
    def clone_policy(self, policy_to_clone: Optional[PolicyNetwork] = None) -> PolicyNetwork:
        if policy_to_clone is None:
            policy_to_clone = self.policy
        cloned_policy = create_policy_network(
            input_dim=self.state_size,
            output_dim=self.action_size,
            config=self.policy_config,
        ).to(self.device)
        cloned_policy.load_state_dict(copy.deepcopy(policy_to_clone.state_dict()))
        return cloned_policy

    def _named_parameters_dict(self, policy: Optional[PolicyNetwork] = None) -> OrderedDict[str, torch.Tensor]:
        policy = policy or self.policy
        return OrderedDict((name, param) for name, param in policy.named_parameters())

    def _regularization_penalty(
        self,
        policy: Optional[PolicyNetwork] = None,
        params: Optional[MutableMapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        policy = policy or self.policy
        l1_lambda = float(getattr(policy, "l1_lambda", 0.0))
        l2_lambda = float(getattr(policy, "l2_lambda", 0.0))
        if params is None:
            return policy.regularization_penalty()

        penalty = torch.zeros((), device=self.device)
        if l1_lambda > 0.0:
            penalty = penalty + l1_lambda * sum(param.abs().sum() for param in params.values())
        if l2_lambda > 0.0:
            penalty = penalty + l2_lambda * sum(param.pow(2).sum() for param in params.values())
        return penalty

    def _policy_forward(
        self,
        state_batch: torch.Tensor,
        *,
        current_policy: Optional[PolicyNetwork] = None,
        params: Optional[MutableMapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        current_policy = current_policy or self.policy
        current_policy.train()
        if params is None:
            return current_policy(state_batch)
        return functional_call(current_policy, params, (state_batch,))

    def _policy_distribution(
        self,
        state_batch: torch.Tensor,
        *,
        current_policy: Optional[PolicyNetwork] = None,
        params: Optional[MutableMapping[str, torch.Tensor]] = None,
    ) -> torch.distributions.Categorical:
        probs = self._policy_forward(state_batch, current_policy=current_policy, params=params)
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.distributions.Categorical(probs=probs)

    def get_action(
        self,
        state: TensorLike,
        current_policy: Optional[PolicyNetwork] = None,
        is_speaking_task: bool = False,
        deterministic: bool = False,
        params: Optional[MutableMapping[str, torch.Tensor]] = None,
    ) -> Tuple[int, torch.Tensor, Optional[int]]:
        current_policy = current_policy or self.policy
        state_tensor = _state_to_tensor(state, device=self.device).unsqueeze(0)
        with torch.set_grad_enabled(params is not None or current_policy.training):
            dist = self._policy_distribution(
                state_tensor,
                current_policy=current_policy,
                params=params,
            )
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            if action.numel() != 1:
                raise InvalidActionError(action)
            log_prob = dist.log_prob(action).squeeze(0)

        action_item = int(action.item())
        message_token = action_item if is_speaking_task else None
        return action_item, log_prob, message_token

    def select_action(
        self,
        processed_state: TensorLike,
        deterministic: bool = False,
    ) -> int:
        action, _, _ = self.get_action(processed_state, deterministic=deterministic)
        return action

    # ------------------------------------------------------------------
    # Trajectory collection and loss computation
    # ------------------------------------------------------------------
    def collect_trajectory(
        self,
        env: Any,
        current_policy: Optional[PolicyNetwork] = None,
        is_speaking_task: bool = False,
        partner_agent: Optional[Any] = None,
        deterministic: bool = False,
        params: Optional[MutableMapping[str, torch.Tensor]] = None,
        tag: Optional[str] = None,
        store_in_memory: bool = False,
    ) -> List[Transition]:
        del partner_agent  # The environment is expected to encode any partner-agent coupling.
        current_policy = current_policy or self.policy
        trajectory: List[Transition] = []
        state = self._safe_reset(env)
        current_message_sequence: List[int] = []

        episode_limit = min(
            self.max_trajectory_steps,
            int(getattr(getattr(env, "spec", None), "max_episode_steps", self.max_trajectory_steps)),
        )

        for _ in range(episode_limit):
            state_tensor = _state_to_tensor(state, device=self.device)
            action, log_prob, msg_token = self.get_action(
                state_tensor,
                current_policy=current_policy,
                is_speaking_task=is_speaking_task,
                deterministic=deterministic,
                params=params,
            )

            if is_speaking_task and msg_token is not None:
                current_message_sequence.append(msg_token)
                if len(current_message_sequence) == self.max_message_length:
                    next_state, reward, done, truncated, _ = self._safe_step(env, action)
                    full_message = tuple(current_message_sequence)
                    current_message_sequence = []
                else:
                    if hasattr(env, "step_speaker"):
                        next_state, reward, done, truncated, _ = self._safe_step(
                            env,
                            action,
                            step_method="step_speaker",
                        )
                    else:
                        next_state, reward, done, truncated, _ = self._safe_step(env, action)
                    full_message = None
            else:
                next_state, reward, done, truncated, _ = self._safe_step(env, action)
                full_message = None

            transition = Transition(
                state=state_tensor.detach().cpu(),
                action=int(action),
                reward=float(reward),
                log_prob=log_prob,
                message=full_message,
            )
            trajectory.append(transition)
            if store_in_memory:
                self.learning_memory.add(
                    {
                        "tag": tag or "trajectory",
                        "agent_id": self.agent_id,
                        "transition": transition,
                    },
                    tag=tag or "trajectory",
                )

            state = next_state
            if done or truncated:
                break

        return trajectory

    def collect_rollouts(
        self,
        env: Any,
        *,
        current_policy: Optional[PolicyNetwork] = None,
        params: Optional[MutableMapping[str, torch.Tensor]] = None,
        episodes: int = 1,
        is_speaking_task: bool = False,
        partner_agent: Optional[Any] = None,
        deterministic: bool = False,
        tag: Optional[str] = None,
        store_in_memory: bool = False,
    ) -> List[List[Transition]]:
        if episodes <= 0:
            raise InvalidConfigError("collect_rollouts episodes must be positive.")

        trajectories: List[List[Transition]] = []
        for _ in range(episodes):
            trajectory = self.collect_trajectory(
                env,
                current_policy=current_policy,
                is_speaking_task=is_speaking_task,
                partner_agent=partner_agent,
                deterministic=deterministic,
                params=params,
                tag=tag,
                store_in_memory=store_in_memory,
            )
            if trajectory:
                trajectories.append(trajectory)
        return trajectories

    def _discounted_returns(self, rewards: Sequence[float]) -> torch.Tensor:
        returns: List[float] = []
        running_return = 0.0
        for reward in reversed(rewards):
            running_return = float(reward) + self.gamma * running_return
            returns.insert(0, running_return)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if self.normalize_returns and returns_tensor.numel() > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-8)
        return returns_tensor

    def compute_loss_from_trajectory(
        self,
        trajectory: Sequence[Transition],
        policy_to_evaluate: Optional[PolicyNetwork] = None,
        params: Optional[MutableMapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        policy_to_evaluate = policy_to_evaluate or self.policy
        if not trajectory:
            return torch.zeros((), device=self.device, requires_grad=True)

        states = torch.stack([_state_to_tensor(t.state, device=self.device) for t in trajectory])
        actions = torch.tensor([int(t.action) for t in trajectory], dtype=torch.long, device=self.device)
        returns = self._discounted_returns([float(t.reward) for t in trajectory])

        if params is not None:
            dist = self._policy_distribution(states, current_policy=policy_to_evaluate, params=params)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
        else:
            stored_log_probs = [t.log_prob for t in trajectory]
            if all(torch.is_tensor(log_prob) for log_prob in stored_log_probs):
                log_probs = torch.stack([log_prob.to(self.device) for log_prob in stored_log_probs])
                dist = self._policy_distribution(states, current_policy=policy_to_evaluate, params=None)
                entropy = dist.entropy().mean()
            else:
                dist = self._policy_distribution(states, current_policy=policy_to_evaluate, params=None)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

        loss = -(log_probs * returns).mean()
        if self.entropy_coef > 0.0:
            loss = loss - self.entropy_coef * entropy
        loss = loss + self._regularization_penalty(policy=policy_to_evaluate, params=params)
        return loss

    def compute_loss_from_trajectories(
        self,
        trajectories: Sequence[Sequence[Transition]],
        policy_to_evaluate: Optional[PolicyNetwork] = None,
        params: Optional[MutableMapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if not trajectories:
            return torch.zeros((), device=self.device, requires_grad=True)
        losses = [
            self.compute_loss_from_trajectory(trajectory, policy_to_evaluate=policy_to_evaluate, params=params)
            for trajectory in trajectories
            if trajectory
        ]
        if not losses:
            return torch.zeros((), device=self.device, requires_grad=True)
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Reward shaping and metrics
    # ------------------------------------------------------------------
    def _calculate_novelty(self, states: Sequence[TensorLike]) -> float:
        if not states:
            return 0.0
        states_tensor = torch.stack([_state_to_tensor(state, device=self.device) for state in states])
        if self.train_novelty_detector:
            self.nd_network.train_step(states_tensor)
        with torch.no_grad():
            novelty_scores = self.nd_network(states_tensor)
        return float(novelty_scores.mean().item())

    def _communication_success(self, env: Any) -> float:
        if hasattr(env, "communication_success"):
            value = getattr(env, "communication_success")
            try:
                return float(value() if callable(value) else value)
            except Exception:
                return 0.0
        return 0.0

    def _task_completed(self, env: Any) -> bool:
        if hasattr(env, "task_completed"):
            try:
                return _coerce_bool(getattr(env, "task_completed"))
            except Exception:
                return False
        return False

    def _apply_task_reward_adjustments(
        self,
        trajectory: Sequence[Transition],
        env: Any,
    ) -> Tuple[List[Transition], Dict[str, float]]:
        if not trajectory:
            metrics = {
                "extrinsic_reward": 0.0,
                "intrinsic_reward": 0.0,
                "communication_success": 0.0,
                "task_success": 0.0,
                "task_completion_bonus": 0.0,
                "communication_bonus": 0.0,
                "adjusted_reward": 0.0,
            }
            return [], metrics

        extrinsic_reward = float(sum(t.reward for t in trajectory))
        novelty_bonus = self._calculate_novelty([t.state for t in trajectory])
        intrinsic_reward = self.novelty_coef * novelty_bonus
        communication_success = self._communication_success(env)
        communication_bonus = self.communication_bonus_coef * communication_success
        task_success = 1.0 if self._task_completed(env) else 0.0
        task_completion_bonus = self.task_completion_bonus_coef * task_success

        total_bonus = intrinsic_reward + communication_bonus + task_completion_bonus
        per_step_bonus = total_bonus / max(len(trajectory), 1)
        adjusted_trajectory = [
            Transition(
                state=t.state,
                action=t.action,
                reward=float(t.reward) + per_step_bonus,
                log_prob=t.log_prob,
                message=t.message,
            )
            for t in trajectory
        ]

        metrics = {
            "extrinsic_reward": extrinsic_reward,
            "intrinsic_reward": intrinsic_reward,
            "communication_success": communication_success,
            "task_success": task_success,
            "task_completion_bonus": task_completion_bonus,
            "communication_bonus": communication_bonus,
            "adjusted_reward": float(sum(t.reward for t in adjusted_trajectory)),
        }
        return adjusted_trajectory, metrics

    def _compute_task_metrics(
        self,
        trajectory: Sequence[Transition],
        env: Any,
        *,
        apply_reward_adjustment: bool = False,
    ) -> Dict[str, Any]:
        adjusted_trajectory, metrics = self._apply_task_reward_adjustments(trajectory, env)
        metrics.update(
            {
                "episode_length": len(trajectory),
                "raw_reward_mean": _safe_mean([t.reward for t in trajectory]),
                "reward_std": _safe_std([t.reward for t in trajectory]),
            }
        )
        if apply_reward_adjustment:
            metrics["adjusted_trajectory"] = adjusted_trajectory
        return metrics

    # ------------------------------------------------------------------
    # Inner-loop adaptation and meta-objective
    # ------------------------------------------------------------------
    def inner_update(
        self,
        env: Any,
        current_meta_policy: Optional[PolicyNetwork] = None,
        is_speaking_task: bool = False,
        partner_agent: Optional[Any] = None,
        *,
        inner_steps: Optional[int] = None,
        support_episodes: Optional[int] = None,
        create_graph: Optional[bool] = None,
        return_params: bool = False,
        deterministic: bool = False,
    ) -> Union[PolicyNetwork, OrderedDict[str, torch.Tensor]]:
        current_meta_policy = current_meta_policy or self.policy
        inner_steps = int(inner_steps or self.inner_steps)
        support_episodes = int(support_episodes or self.support_episodes)
        create_graph = (not self.first_order) if create_graph is None else bool(create_graph)

        if inner_steps <= 0:
            raise InvalidConfigError("inner_steps must be positive.")
        if support_episodes <= 0:
            raise InvalidConfigError("support_episodes must be positive.")

        fast_params: OrderedDict[str, torch.Tensor] = self._named_parameters_dict(current_meta_policy)
        for step_idx in range(inner_steps):
            support_trajectories = self.collect_rollouts(
                env,
                current_policy=current_meta_policy,
                params=fast_params,
                episodes=support_episodes,
                is_speaking_task=is_speaking_task,
                partner_agent=partner_agent,
                deterministic=deterministic,
                tag=f"support_step_{step_idx}",
                store_in_memory=True,
            )
            if self.use_reward_shaping:
                shaped_support_trajectories = []
                for trajectory in support_trajectories:
                    metrics = self._compute_task_metrics(trajectory, env, apply_reward_adjustment=True)
                    shaped_support_trajectories.append(metrics["adjusted_trajectory"])
                support_trajectories = shaped_support_trajectories

            support_loss = self.compute_loss_from_trajectories(
                support_trajectories,
                policy_to_evaluate=current_meta_policy,
                params=fast_params,
            )

            grads = torch.autograd.grad(
                support_loss,
                tuple(fast_params.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=False,
            )

            fast_params = OrderedDict(
                (
                    name,
                    param - self.inner_lr * grad,
                )
                for (name, param), grad in zip(fast_params.items(), grads)
            )

        if return_params:
            return fast_params

        adapted_policy = self.clone_policy(current_meta_policy)
        with torch.no_grad():
            named_params = dict(adapted_policy.named_parameters())
            for name, fast_param in fast_params.items():
                named_params[name].copy_(fast_param.detach())
        return adapted_policy

    def _meta_objective(
        self,
        tasks: Sequence[TaskSpec],
        *,
        inner_steps: Optional[int] = None,
        create_graph: Optional[bool] = None,
        apply_reward_shaping: Optional[bool] = None,
        sampler: Optional[Callable[..., TaskSpec]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not tasks:
            raise InvalidConfigError("meta_update requires at least one task.")

        inner_steps = int(inner_steps or self.inner_steps)
        create_graph = (not self.first_order) if create_graph is None else bool(create_graph)
        apply_reward_shaping = self.use_reward_shaping if apply_reward_shaping is None else bool(apply_reward_shaping)

        task_losses: List[torch.Tensor] = []
        support_rewards: List[float] = []
        query_rewards: List[float] = []
        intrinsic_rewards: List[float] = []
        communication_scores: List[float] = []
        task_success_scores: List[float] = []
        episode_lengths: List[int] = []

        for task in tasks:
            env, task_info, sampled = self._resolve_task_spec(task, split="train", sampler=sampler)
            is_speaking = bool(task_info.get("is_speaking", False))
            partner = task_info.get("partner_agent")
            task_support_episodes = int(task_info.get("support_episodes", self.support_episodes))
            task_query_episodes = int(task_info.get("query_episodes", self.query_episodes))

            try:
                fast_params = self.inner_update(
                    env,
                    self.policy,
                    is_speaking_task=is_speaking,
                    partner_agent=partner,
                    inner_steps=inner_steps,
                    support_episodes=task_support_episodes,
                    create_graph=create_graph,
                    return_params=True,
                )
                query_trajectories = self.collect_rollouts(
                    env,
                    current_policy=self.policy,
                    params=fast_params,
                    episodes=task_query_episodes,
                    is_speaking_task=is_speaking,
                    partner_agent=partner,
                    deterministic=False,
                    tag="query",
                    store_in_memory=True,
                )
                processed_query_trajectories: List[List[Transition]] = []
                query_metric_records: List[Dict[str, Any]] = []
                for trajectory in query_trajectories:
                    if apply_reward_shaping:
                        metrics = self._compute_task_metrics(trajectory, env, apply_reward_adjustment=True)
                        processed_query_trajectories.append(metrics["adjusted_trajectory"])
                    else:
                        metrics = self._compute_task_metrics(trajectory, env, apply_reward_adjustment=False)
                        processed_query_trajectories.append(list(trajectory))
                    query_metric_records.append(metrics)

                support_reward = 0.0
                if query_trajectories:
                    support_reward = float(np.mean([sum(t.reward for t in traj) for traj in query_trajectories]))

                query_loss = self.compute_loss_from_trajectories(
                    processed_query_trajectories,
                    policy_to_evaluate=self.policy,
                    params=fast_params,
                )
                task_losses.append(query_loss)
                support_rewards.append(support_reward)
                query_rewards.extend([record.get("adjusted_reward", record.get("extrinsic_reward", 0.0)) for record in query_metric_records])
                intrinsic_rewards.extend([record.get("intrinsic_reward", 0.0) for record in query_metric_records])
                communication_scores.extend([record.get("communication_success", 0.0) for record in query_metric_records])
                task_success_scores.extend([record.get("task_success", 0.0) for record in query_metric_records])
                episode_lengths.extend([record.get("episode_length", 0) for record in query_metric_records])
            finally:
                if sampled:
                    self._close_env_if_possible(env)

        meta_loss = torch.stack(task_losses).mean() if task_losses else torch.zeros((), device=self.device)
        diagnostics = {
            "support_reward": _safe_mean(support_rewards),
            "query_reward": _safe_mean(query_rewards),
            "intrinsic_reward": _safe_mean(intrinsic_rewards),
            "communication_success": _safe_mean(communication_scores),
            "task_success_rate": _safe_mean(task_success_scores),
            "avg_episode_length": _safe_mean(episode_lengths),
            "task_count": len(task_losses),
        }
        return meta_loss, diagnostics

    def _post_backward_checks(self) -> float:
        grad_sq_sum = 0.0
        max_abs_grad = 0.0
        for param in self.policy.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if not torch.isfinite(grad).all():
                raise NaNException("Non-finite gradient detected during meta-update.")
            grad_norm = float(torch.norm(grad).item())
            grad_sq_sum += grad_norm ** 2
            max_abs_grad = max(max_abs_grad, float(grad.abs().max().item()))
        total_grad_norm = grad_sq_sum ** 0.5
        if total_grad_norm > self.gradient_explosion_threshold:
            raise GradientExplosionError(total_grad_norm, self.gradient_explosion_threshold)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.grad_clip_norm)
        return total_grad_norm

    def compute_meta_gradient_contribution(
        self,
        tasks_for_agent: Sequence[TaskSpec],
        inner_steps: Optional[int] = None,
    ) -> float:
        """Compute meta-gradients without applying an optimiser step.

        This is used by the decentralized fleet before parameter diffusion.
        """
        self.meta_optimizer.zero_grad(set_to_none=True)
        meta_loss, diagnostics = self._meta_objective(
            tasks_for_agent,
            inner_steps=inner_steps,
            create_graph=not self.first_order,
        )
        if not torch.isfinite(meta_loss):
            raise NaNException("Meta-loss became non-finite while computing gradient contribution.")
        meta_loss.backward()
        diagnostics["grad_norm"] = self._post_backward_checks()
        diagnostics["meta_loss"] = float(meta_loss.item())
        self.last_meta_metrics = diagnostics
        return float(meta_loss.item())

    def meta_update(
        self,
        tasks: Sequence[TaskSpec],
        inner_steps: Optional[int] = None,
    ) -> float:
        self.meta_optimizer.zero_grad(set_to_none=True)
        meta_loss, diagnostics = self._meta_objective(
            tasks,
            inner_steps=inner_steps,
            create_graph=not self.first_order,
        )
        if not torch.isfinite(meta_loss):
            raise NaNException("Meta-loss became non-finite during optimisation.")
        meta_loss.backward()
        diagnostics["grad_norm"] = self._post_backward_checks()
        self.meta_optimizer.step()
        diagnostics["meta_loss"] = float(meta_loss.item())
        self.last_meta_metrics = diagnostics
        return float(meta_loss.item())

    # ------------------------------------------------------------------
    # Public training / evaluation API
    # ------------------------------------------------------------------
    def learn_step(self, trajectory: Sequence[Transition]) -> float:
        if not trajectory:
            raise InvalidConfigError("learn_step requires a non-empty trajectory.")
        self.meta_optimizer.zero_grad(set_to_none=True)
        loss = self.compute_loss_from_trajectory(trajectory, self.policy)
        if not torch.isfinite(loss):
            raise NaNException("Non-finite loss encountered in learn_step.")
        loss.backward()
        self._post_backward_checks()
        self.meta_optimizer.step()
        return float(loss.item())

    def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        tasks = task_data.get("tasks")
        if not tasks:
            num_tasks = int(task_data.get("num_tasks", task_data.get("tasks_per_batch", 1)))
            tasks = [self._sample_training_task()[0:2] for _ in range(num_tasks)]
        loss = self.meta_update(tasks, inner_steps=task_data.get("inner_steps"))
        return {
            "status": "success",
            "agent": "MAMLAgent",
            "meta_loss": loss,
            "metrics": copy.deepcopy(self.last_meta_metrics or {}),
        }

    def train(
        self,
        num_meta_epochs: int = 50,
        tasks_per_epoch: int = 5,
        adaptation_steps: Optional[int] = None,
        validation_freq: Optional[int] = None,
        validation_tasks: int = 5,
        checkpoint_dir: Optional[Union[str, os.PathLike[str]]] = None,
        early_stop_patience: Optional[int] = None,
        target_reward: Optional[float] = None,
        task_sampler: Optional[Callable[..., TaskSpec]] = None,
    ) -> Dict[str, List[float]]:
        if num_meta_epochs <= 0 or tasks_per_epoch <= 0:
            raise InvalidConfigError("num_meta_epochs and tasks_per_epoch must be positive integers.")

        task_sampler = task_sampler or self.task_sampler
        if task_sampler is None:
            raise InvalidConfigError("train() requires a task_sampler or agent-level task_sampler.")

        adaptation_steps = int(adaptation_steps or self.inner_steps)
        validation_freq = int(validation_freq or self.maml_config.get("validation_freq", 0) or 0)
        validation_tasks = int(validation_tasks)
        early_stop_patience = (
            int(early_stop_patience)
            if early_stop_patience is not None
            else self.config.get("unified", {}).get("early_stop_patience")
        )

        checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else self.checkpoint_dir
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        best_validation_reward = float("-inf")
        early_stop_counter = 0

        self.training_metrics = defaultdict(list)
        logger.info(
            "Starting meta-training | epochs=%s tasks_per_epoch=%s adaptation_steps=%s",
            num_meta_epochs,
            tasks_per_epoch,
            adaptation_steps,
        )

        for epoch in range(num_meta_epochs):
            tasks = [self._resolve_task_spec(None, split="train", sampler=task_sampler)[:2] for _ in range(tasks_per_epoch)]
            meta_loss = self.meta_update(tasks, inner_steps=adaptation_steps)
            epoch_metrics = copy.deepcopy(self.last_meta_metrics or {})
            epoch_metrics["meta_loss"] = meta_loss

            for key, value in epoch_metrics.items():
                if isinstance(value, (int, float)):
                    self.training_metrics[key].append(float(value))

            logger.info(
                "Epoch %s/%s | meta_loss=%.4f query_reward=%.3f support_reward=%.3f grad_norm=%.3f",
                epoch + 1,
                num_meta_epochs,
                meta_loss,
                epoch_metrics.get("query_reward", 0.0),
                epoch_metrics.get("support_reward", 0.0),
                epoch_metrics.get("grad_norm", 0.0),
            )

            should_validate = validation_freq > 0 and ((epoch + 1) % validation_freq == 0)
            if should_validate:
                validation_metrics = self.evaluate(
                    env=None,
                    num_eval_tasks=validation_tasks,
                    adaptation_steps=adaptation_steps,
                    meta_eval=False,
                    task_sampler=task_sampler,
                )
                validation_reward = float(validation_metrics.get("adapted_performance", validation_metrics.get("baseline_performance", 0.0)))
                self.training_metrics["validation_reward"].append(validation_reward)

                if validation_reward > best_validation_reward:
                    best_validation_reward = validation_reward
                    early_stop_counter = 0
                    best_path = checkpoint_root / f"best_epoch_{epoch + 1}.pt"
                    self.save(best_path)
                else:
                    early_stop_counter += 1

                if early_stop_patience is not None and early_stop_counter >= int(early_stop_patience):
                    logger.info("Early stopping triggered at epoch %s", epoch + 1)
                    break

                if target_reward is not None and validation_reward >= float(target_reward):
                    logger.info("Target reward %.3f reached at epoch %s", float(target_reward), epoch + 1)
                    break

        logger.info("Meta-training complete")
        return {key: list(values) for key, values in self.training_metrics.items()}

    def evaluate(
        self,
        env: Optional[Any],
        num_eval_tasks: int = 20,
        adaptation_steps: int = 3,
        meta_eval: bool = False,
        task_sampler: Optional[Callable[..., TaskSpec]] = None,
    ) -> Dict[str, Any]:
        if num_eval_tasks <= 0:
            raise InvalidConfigError("num_eval_tasks must be positive.")
        if adaptation_steps <= 0:
            raise InvalidConfigError("adaptation_steps must be positive.")

        logger.info("Evaluating MAML Agent %s on %s tasks", self.agent_id, num_eval_tasks)

        baseline_returns: List[float] = []
        adapted_returns: List[float] = []
        adaptation_gains: List[float] = []
        adaptation_speed: List[float] = []
        communication_scores: List[float] = []
        task_success_scores: List[float] = []
        q_lengths: List[int] = []
        reward_components: Dict[str, List[float]] = defaultdict(list)

        for _ in range(num_eval_tasks):
            if env is None:
                task_env, task_info, sampled = (
                    self._sample_evaluation_task()
                    if task_sampler is None
                    else self._resolve_task_spec(None, split="eval", sampler=task_sampler)
                )
            else:
                task_env, task_info, sampled = self._resolve_task_spec(env, split="eval")
            if not isinstance(task_info, dict):
                task_info = {}

            is_speaking = bool(task_info.get("is_speaking", False))
            partner = task_info.get("partner_agent")

            try:
                baseline_trajectories = self.collect_rollouts(
                    task_env,
                    current_policy=self.policy,
                    params=None,
                    episodes=self.query_episodes,
                    is_speaking_task=is_speaking,
                    partner_agent=partner,
                    deterministic=self.eval_deterministic,
                )
                baseline_return = _safe_mean([sum(t.reward for t in trajectory) for trajectory in baseline_trajectories])
                baseline_returns.append(baseline_return)

                if not meta_eval:
                    adaptation_rewards: List[float] = []
                    fast_params = self.inner_update(
                        task_env,
                        self.policy,
                        is_speaking_task=is_speaking,
                        partner_agent=partner,
                        inner_steps=adaptation_steps,
                        support_episodes=self.support_episodes,
                        create_graph=False,
                        return_params=True,
                        deterministic=False,
                    )
                    adapted_trajectories = self.collect_rollouts(
                        task_env,
                        current_policy=self.policy,
                        params=fast_params,
                        episodes=self.query_episodes,
                        is_speaking_task=is_speaking,
                        partner_agent=partner,
                        deterministic=self.eval_deterministic,
                    )
                    for trajectory in adapted_trajectories:
                        adaptation_rewards.append(sum(t.reward for t in trajectory))
                        metrics = self._compute_task_metrics(trajectory, task_env, apply_reward_adjustment=False)
                        communication_scores.append(metrics.get("communication_success", 0.0))
                        task_success_scores.append(metrics.get("task_success", 0.0))
                        q_lengths.append(metrics.get("episode_length", 0))
                        for key, value in metrics.items():
                            if key.endswith("reward") or key.endswith("bonus"):
                                reward_components[key].append(float(value))

                    final_return = _safe_mean(adaptation_rewards)
                    adapted_returns.append(final_return)
                    adaptation_gains.append(final_return - baseline_return)
                    adaptation_speed.append(final_return - baseline_return)
            finally:
                if sampled:
                    self._close_env_if_possible(task_env)

        total_params = int(sum(p.numel() for p in self.policy.parameters()))
        trainable_params = int(sum(p.numel() for p in self.policy.parameters() if p.requires_grad))

        metrics: Dict[str, Any] = {
            "baseline_performance": _safe_mean(baseline_returns),
            "baseline_std": _safe_std(baseline_returns),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "adaptation_steps": adaptation_steps,
            "meta_evaluation": bool(meta_eval),
        }

        if not meta_eval:
            metrics.update(
                {
                    "adapted_performance": _safe_mean(adapted_returns),
                    "adapted_std": _safe_std(adapted_returns),
                    "adaptation_gain": _safe_mean(adaptation_gains),
                    "adaptation_speed": _safe_mean(adaptation_speed),
                    "communication_accuracy": _safe_mean(communication_scores),
                    "task_success_rate": _safe_mean(task_success_scores),
                    "avg_episode_length": _safe_mean(q_lengths),
                    "reward_components": {key: _safe_mean(values) for key, values in reward_components.items()},
                }
            )

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Union[str, os.PathLike[str]]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "agent_id": self.agent_id,
            "model_id": self.model_id,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "config": copy.deepcopy(self.config),
            "policy_state_dict": self.policy.state_dict(),
            "meta_optimizer_state_dict": self.meta_optimizer.state_dict(),
            "novelty_detector_state_dict": self.nd_network.state_dict(),
            "training_metrics": {key: list(values) for key, values in self.training_metrics.items()},
        }
        torch.save(checkpoint, path)
        logger.info("Saved MAML model checkpoint to %s", path)

    def load(self, path: Union[str, os.PathLike[str]], strict: bool = True) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"], strict=strict)
        self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer_state_dict"])
        nd_state = checkpoint.get("novelty_detector_state_dict")
        if nd_state is not None:
            self.nd_network.load_state_dict(nd_state, strict=False)
        self.training_metrics = defaultdict(list, checkpoint.get("training_metrics", {}))
        logger.info("Loaded MAML model checkpoint from %s", path)

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, os.PathLike[str]],
        *,
        task_sampler: Optional[Callable[..., TaskSpec]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "MAMLAgent":
        checkpoint = torch.load(Path(path), map_location=device or "cpu")
        agent = cls(
            agent_id=checkpoint.get("agent_id", "MAML"),
            state_size=int(checkpoint["state_size"]),
            action_size=int(checkpoint["action_size"]),
            config=checkpoint.get("config"),
            task_sampler=task_sampler,
            device=device,
        )
        agent.load(path)
        return agent


class DecentralizedMAMLFleet:
    """Decentralized MAML fleet with configurable parameter diffusion."""

    def __init__(
        self,
        num_agents: int,
        global_config: Optional[Dict[str, Any]],
        env_creator_fn: Callable[..., TaskSpec],
        state_size: int,
        action_size: int,
        agent_config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if int(num_agents) <= 0:
            raise InvalidConfigError("num_agents must be positive.")
        if env_creator_fn is None or not callable(env_creator_fn):
            raise InvalidConfigError("env_creator_fn must be callable.")

        self.num_agents = int(num_agents)
        if global_config is not None and not isinstance(global_config, dict):
            raise InvalidConfigError("global_config must be a dictionary when provided.")
        if agent_config is not None and not isinstance(agent_config, dict):
            raise InvalidConfigError("agent_config must be a dictionary when provided.")

        self.global_config = global_config if global_config is not None else load_global_config()
        self.agent_config = agent_config or {}
        self.env_creator_fn = env_creator_fn
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.agents = [
            MAMLAgent(
                agent_id=i,
                state_size=self.state_size,
                action_size=self.action_size,
                config=self.agent_config,
                task_sampler=self.env_creator_fn,
                device=self.device,
            )
            for i in range(self.num_agents)
        ]

        maml_config = get_config_section.get("maml", {})
        self.diffusion_type = str(maml_config.get("diffusion_type", "average")).lower()
        self.meta_epochs = int(maml_config.get("meta_epochs", 100))
        self.tasks_per_agent_meta_batch = int(maml_config.get("tasks_per_agent_meta_batch", 5))
        self.inner_steps = int(maml_config.get("inner_steps", 1))
        self.adj_matrix = self._setup_adjacency(maml_config.get("adjacency_matrix"))
        self.fleet_metrics: Dict[str, List[float]] = defaultdict(list)

        logger.info(
            "DecentralizedMAMLFleet initialised | agents=%s diffusion=%s tasks_per_agent=%s",
            self.num_agents,
            self.diffusion_type,
            self.tasks_per_agent_meta_batch,
        )

    def _setup_adjacency(self, adj_matrix_config: Optional[Any]) -> torch.Tensor:
        if adj_matrix_config in {None, "fully_connected", "average"}:
            adj = np.ones((self.num_agents, self.num_agents), dtype=np.float32)
        elif isinstance(adj_matrix_config, list):
            adj = np.asarray(adj_matrix_config, dtype=np.float32)
        else:
            raise InvalidConfigError("adjacency_matrix must be None, 'fully_connected', or a numeric square matrix.")

        if adj.shape != (self.num_agents, self.num_agents):
            raise InvalidConfigError("Adjacency matrix shape mismatch.")
        if np.any(adj < 0.0):
            raise InvalidConfigError("Adjacency weights must be non-negative.")

        row_sums = adj.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise InvalidConfigError("Each adjacency row must contain at least one positive weight.")
        adj = adj / row_sums
        return torch.as_tensor(adj, dtype=torch.float32, device=self.device)

    def _clone_optimizer_for_policy(self, agent: MAMLAgent, policy: PolicyNetwork) -> optim.Optimizer:
        temp_optimizer = agent._build_meta_optimizer(policy)
        try:
            temp_optimizer.load_state_dict(copy.deepcopy(agent.meta_optimizer.state_dict()))
        except Exception:
            # If state shapes no longer match, fall back to a fresh optimizer.
            temp_optimizer = agent._build_meta_optimizer(policy)
        return temp_optimizer

    def _candidate_state_dicts(self) -> Tuple[List[Dict[str, torch.Tensor]], List[float], List[float]]:
        candidate_policy_params_list: List[Dict[str, torch.Tensor]] = []
        per_agent_meta_losses: List[float] = []
        per_agent_grad_norms: List[float] = []

        for agent in self.agents:
            tasks = [self.env_creator_fn() for _ in range(self.tasks_per_agent_meta_batch)]
            meta_loss_val = agent.compute_meta_gradient_contribution(tasks, inner_steps=self.inner_steps)
            per_agent_meta_losses.append(float(meta_loss_val))
            per_agent_grad_norms.append(float((agent.last_meta_metrics or {}).get("grad_norm", 0.0)))

            temp_policy = agent.clone_policy(agent.policy)
            temp_optimizer = self._clone_optimizer_for_policy(agent, temp_policy)
            temp_optimizer.zero_grad(set_to_none=True)
            for source_param, temp_param in zip(agent.policy.parameters(), temp_policy.parameters()):
                if source_param.grad is not None:
                    temp_param.grad = source_param.grad.detach().clone()
            temp_optimizer.step()
            candidate_policy_params_list.append(copy.deepcopy(temp_policy.state_dict()))

        return candidate_policy_params_list, per_agent_meta_losses, per_agent_grad_norms

    def _diffuse_parameters(self, candidate_policy_params_list: Sequence[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        if not candidate_policy_params_list:
            raise InvalidConfigError("No candidate policies available for diffusion.")

        param_keys = list(candidate_policy_params_list[0].keys())
        new_agent_params: List[Dict[str, torch.Tensor]] = []
        for target_idx in range(self.num_agents):
            blended_state: Dict[str, torch.Tensor] = {}
            for key in param_keys:
                template = candidate_policy_params_list[0][key]
                blended_tensor = torch.zeros_like(template)
                for source_idx in range(self.num_agents):
                    weight = float(self.adj_matrix[target_idx, source_idx].item())
                    blended_tensor = blended_tensor + weight * candidate_policy_params_list[source_idx][key]
                blended_state[key] = blended_tensor
            new_agent_params.append(blended_state)
        return new_agent_params

    def train_fleet(self) -> Dict[str, List[float]]:
        logger.info("Starting decentralized meta-training")
        self.fleet_metrics = defaultdict(list)

        for epoch in range(self.meta_epochs):
            candidate_policy_params_list, per_agent_meta_losses, per_agent_grad_norms = self._candidate_state_dicts()
            new_agent_policies_params = self._diffuse_parameters(candidate_policy_params_list)

            for agent, new_params in zip(self.agents, new_agent_policies_params):
                agent.policy.load_state_dict(new_params)
                agent.meta_optimizer = agent._build_meta_optimizer(agent.policy)

            avg_meta_loss = _safe_mean(per_agent_meta_losses)
            avg_grad_norm = _safe_mean(per_agent_grad_norms)
            self.fleet_metrics["meta_loss"].append(avg_meta_loss)
            self.fleet_metrics["grad_norm"].append(avg_grad_norm)
            logger.info(
                "Fleet epoch %s/%s | avg_meta_loss=%.4f avg_grad_norm=%.4f",
                epoch + 1,
                self.meta_epochs,
                avg_meta_loss,
                avg_grad_norm,
            )

        logger.info("Decentralized meta-training complete")
        return {key: list(values) for key, values in self.fleet_metrics.items()}

    def evaluate_fleet(self, num_eval_tasks_per_agent: int = 10, adaptation_steps: Optional[int] = None) -> Dict[str, Any]:
        if num_eval_tasks_per_agent <= 0:
            raise InvalidConfigError("num_eval_tasks_per_agent must be positive.")

        adaptation_steps = int(adaptation_steps or self.inner_steps)
        logger.info("Evaluating decentralized fleet")
        all_agent_avg_rewards: List[float] = []
        all_agent_gains: List[float] = []

        for agent in self.agents:
            metrics = agent.evaluate(
                env=None,
                num_eval_tasks=num_eval_tasks_per_agent,
                adaptation_steps=adaptation_steps,
                meta_eval=False,
                task_sampler=self.env_creator_fn,
            )
            all_agent_avg_rewards.append(float(metrics.get("adapted_performance", metrics.get("baseline_performance", 0.0))))
            all_agent_gains.append(float(metrics.get("adaptation_gain", 0.0)))
            logger.info(
                "Agent %s | avg_eval_reward=%.3f adaptation_gain=%.3f",
                agent.agent_id,
                all_agent_avg_rewards[-1],
                all_agent_gains[-1],
            )

        overall_avg_reward = _safe_mean(all_agent_avg_rewards)
        overall_avg_gain = _safe_mean(all_agent_gains)
        logger.info(
            "Fleet evaluation complete | overall_avg_reward=%.3f overall_avg_gain=%.3f",
            overall_avg_reward,
            overall_avg_gain,
        )
        return {
            "overall_average_reward": overall_avg_reward,
            "overall_adaptation_gain": overall_avg_gain,
            "agent_rewards": all_agent_avg_rewards,
            "agent_adaptation_gains": all_agent_gains,
        }


def _infer_env_dimensions(env: Any) -> Tuple[int, int]:
    observation_space = getattr(env, "observation_space", None)
    action_space = getattr(env, "action_space", None)

    if observation_space is None or action_space is None:
        raise InvalidConfigError(
            "Unable to infer state/action dimensions from environment: observation_space or action_space missing."
        )

    obs_shape = getattr(observation_space, "shape", None)
    if obs_shape is None:
        raise InvalidConfigError("Environment observation_space does not expose a shape attribute.")
    if len(obs_shape) == 0:
        state_size = 1
    else:
        state_size = int(np.prod(obs_shape))

    if hasattr(action_space, "n"):
        action_size = int(action_space.n)
    else:
        raise InvalidConfigError("MAMLAgent currently supports discrete action spaces with an 'n' attribute.")

    if state_size <= 0 or action_size <= 0:
        raise InvalidConfigError("Inferred state/action dimensions must be positive.")
    return state_size, action_size


if __name__ == "__main__":
    def project_env_task_sampler(split: str = "train") -> Tuple[Any, Dict[str, Any]]:
        del split
        import importlib

        import_candidates = [
            ("src.agents.learnig.slaienv", "SLAIEnv"),
            ("src.agents.learning.slaienv", "SLAIEnv"),
        ]

        last_error = None
        for module_name, attr_name in import_candidates:
            try:
                module = importlib.import_module(module_name)
                env_factory = getattr(module, attr_name)
                env_obj = env_factory()
                if callable(env_obj) and not hasattr(env_obj, "reset"):
                    task = env_obj()
                else:
                    task = env_obj

                if isinstance(task, tuple) and len(task) == 2 and isinstance(task[1], dict):
                    return task
                return task, {}
            except Exception as exc:  # pragma: no cover - smoke-test integration path
                last_error = exc

        raise RuntimeError(
            "Failed to import the project environment for the MAML smoke test. "
            "Tried: src.agents.learnig.slaienv.SLAIEnv and src.agents.learning.slaienv.SLAIEnv"
        ) from last_error

    sampled_task = project_env_task_sampler(split="train")
    if isinstance(sampled_task, tuple) and len(sampled_task) == 2 and isinstance(sampled_task[1], dict):
        smoke_env, _task_info = sampled_task
    else:
        smoke_env, _task_info = sampled_task, {}

    try:
        inferred_state_size, inferred_action_size = _infer_env_dimensions(smoke_env)
    finally:
        MAMLAgent._close_env_if_possible(smoke_env)

    agent = MAMLAgent(
        agent_id="maml_smoke",
        state_size=inferred_state_size,
        action_size=inferred_action_size,
        task_sampler=project_env_task_sampler,
    )
    train_metrics = agent.train(num_meta_epochs=2, tasks_per_epoch=2, adaptation_steps=1, validation_freq=0)
    eval_metrics = agent.evaluate(env=None, num_eval_tasks=2, adaptation_steps=1, task_sampler=project_env_task_sampler)
    print("Train metrics keys:", sorted(train_metrics.keys()))
    print("Evaluation:", eval_metrics)
