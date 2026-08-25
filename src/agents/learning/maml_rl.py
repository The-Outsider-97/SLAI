"""
Production-ready Model-Agnostic Meta-Learning utilities for SLAI.

This module provides a policy-gradient MAML agent for discrete-action meta-RL
and a decentralized fleet wrapper for diffusion-style multi-agent meta-learning.
It keeps the existing SLAI learning subsystem integration points intact:
configuration is loaded through ``learning_config.yaml`` via ``load_global_config``
and ``get_config_section``, policies come from ``modules.policy_network``, and
experience traces can be written to ``LearningMemory``.
"""

from __future__ import annotations

import copy
import inspect
import math
import os
import time
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore

from collections import OrderedDict, defaultdict, namedtuple, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from torch.func import functional_call  # type: ignore

# NumPy 2.x compatibility for older Gym internals that still reference np.bool8.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from .utils.config_loader import load_global_config, get_config_section
from .utils.learning_error import *
from .utils.learning_calculations import *
from .utils.learning_helpers import *
from .modules.policy_network import *
from .learning_memory import LearningMemory
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Model-Agnostic Meta-Learning")
printer = PrettyPrinter()

TensorLike = Union[torch.Tensor, np.ndarray, Sequence[float]]
TaskSpec = Union[Any, Tuple[Any, Dict[str, Any]], Dict[str, Any]]
Transition = namedtuple("Transition", ["state", "action", "reward", "log_prob", "message"])


# ---------------------------------------------------------------------------
# Shared module helpers
# ---------------------------------------------------------------------------
def _safe_mean(values: Sequence[Union[int, float]]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: Sequence[Union[int, float]]) -> float:
    return float(np.std(values)) if values else 0.0


def _call_or_value(value: Any) -> Any:
    return value() if callable(value) else value


def _finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if torch.isnan(tensor).any():
        raise NaNException(f"NaN detected in {name}", location=name)
    if torch.isinf(tensor).any():
        raise InfException(f"Inf detected in {name}", location=name)


def _state_to_tensor(
    state: Union[torch.Tensor, np.ndarray, Sequence[float]],
    device: Optional[torch.device] = None,
    *,
    expected_dim: Optional[int] = None,
    name: str = "state",
) -> torch.Tensor:
    """Convert an arbitrary state-like input into a finite 1D float tensor."""
    if isinstance(state, torch.Tensor):
        tensor = state.detach().clone().to(dtype=torch.float32) # type: ignore
    else:
        try:
            tensor = torch.as_tensor(state, dtype=torch.float32)
        except Exception as exc:
            raise ObservationShapeError(expected_shape=expected_dim or "tensor-like", actual_shape=type(state).__name__, cause=exc) from exc

    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim > 1:
        tensor = tensor.reshape(-1)

    if expected_dim is not None and tensor.numel() != int(expected_dim):
        raise ObservationShapeError(expected_shape=(int(expected_dim),), actual_shape=tuple(tensor.shape))

    if device is not None:
        tensor = tensor.to(device=device)
    _finite_tensor(tensor, name)
    return tensor


def _normalise_task_info(task_info: Any) -> Dict[str, Any]:
    return copy.deepcopy(task_info) if isinstance(task_info, dict) else {}


class MAMLAgent:
    """Policy-gradient MAML agent for discrete-action meta-reinforcement learning."""

    CHECKPOINT_VERSION = 2

    def __init__(
        self,
        agent_id: Optional[Union[str, int]],
        state_size: int,
        action_size: int,
        config: Optional[Dict[str, Any]] = None,
        task_sampler: Optional[Callable[..., TaskSpec]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        validate_positive(state_size, "state_size", strict=True)
        validate_positive(action_size, "action_size", strict=True)
        if config is not None and not isinstance(config, dict):
            raise InvalidConfigError("config must be a dictionary when provided.", received_value=type(config).__name__)

        self.agent_id = str(agent_id) if agent_id is not None else "MAML"
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        # Keep the existing config pattern: global config + section lookup.
        self.config = load_global_config()
        self.maml_config = get_config_section("maml") or {}
        self.config_override = config or {}
        self.maml_override = self.config_override.get("maml", {})
        if self.maml_override and not isinstance(self.maml_override, dict):
            raise InvalidConfigError("config['maml'] must be a dictionary when provided.")

        self.model_id = "MAML_Agent"
        self.task_sampler = task_sampler
        self.calculations = LearningCalculations()
        self.reward_stats = RunningStats()
        self.loss_stats = RunningStats()
        self.grad_stats = RunningStats()
        self.episode_length_stats = RunningStats()
        self.recent_rewards: Deque[float] = deque(maxlen=coerce_int(self._cfg("reward_history", 250), 250, minimum=1))
        self.recent_losses: Deque[float] = deque(maxlen=coerce_int(self._cfg("loss_history", 250), 250, minimum=1))
        self.total_rollouts = 0
        self.total_meta_updates = 0
        self.total_inner_updates = 0

        self.gamma = coerce_float(self._cfg("gamma", 0.99), 0.99, minimum=1e-12, maximum=1.0)
        self.meta_lr = coerce_float(self._cfg("meta_lr", 1e-3), 1e-3, minimum=1e-12)
        self.inner_lr = coerce_float(self._cfg("inner_lr", 1e-2), 1e-2, minimum=1e-12)
        self.inner_steps = coerce_int(self._cfg("inner_steps", 1), 1, minimum=1)
        self.support_episodes = coerce_int(self._cfg("support_episodes", 1), 1, minimum=1)
        self.query_episodes = coerce_int(self._cfg("query_episodes", 1), 1, minimum=1)
        self.max_trajectory_steps = coerce_int(self._cfg("max_trajectory_steps", 500), 500, minimum=1)
        self.entropy_coef = coerce_float(self._cfg("entropy_coef", 0.0), 0.0, minimum=0.0)
        self.normalize_returns = coerce_bool(self._cfg("normalize_returns", True), True)
        self.first_order = coerce_bool(self._cfg("first_order", False), False)
        self.grad_clip_norm = coerce_float(self._cfg("grad_clip_norm", 1.0), 1.0, minimum=1e-12)
        self.gradient_explosion_threshold = coerce_float(self._cfg("gradient_explosion_threshold", 1e3), 1e3, minimum=1e-12)
        self.eval_deterministic = coerce_bool(self._cfg("eval_deterministic", False), False)
        self.use_reward_shaping = coerce_bool(self._cfg("use_reward_shaping", False), False)
        self.novelty_coef = coerce_float(self._cfg("novelty_coef", 0.1), 0.1, minimum=0.0)
        self.communication_bonus_coef = coerce_float(self._cfg("communication_bonus_coef", 2.0), 2.0)
        self.task_completion_bonus_coef = coerce_float(self._cfg("task_completion_bonus_coef", 5.0), 5.0)
        self.train_novelty_detector = coerce_bool(self._cfg("train_novelty_detector", True), True)
        self.novelty_feature_dim = coerce_int(self._cfg("novelty_feature_dim", max(16, int(self._cfg("hidden_size", 64)))), 32, minimum=1)
        self.reward_clip = self._cfg("reward_clip", None)
        self.fail_fast_task_errors = coerce_bool(self._cfg("fail_fast_task_errors", True), True)
        self.checkpoint_dir = Path(str(self._cfg("checkpoint_dir", "src/agents/learning/checkpoints/maml")))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.training_metrics: Dict[str, List[float]] = defaultdict(list)
        self.last_meta_metrics: Optional[Dict[str, Any]] = None
        self.policy_config = self._build_policy_config()
        self.policy = create_policy_network(self.state_size, self.action_size, config=self.policy_config).to(self.device)
        self.meta_optimizer = self._build_meta_optimizer(self.policy)
        self.learning_memory = LearningMemory()
        self.nd_network = NoveltyDetector(
            input_dim=self.state_size,
            feature_dim=self.novelty_feature_dim,
            learning_rate=coerce_float(self._cfg("novelty_lr", 1e-3), 1e-3, minimum=1e-12),
            hidden_sizes=self._cfg("novelty_hidden_sizes", None),
            activation=str(self._cfg("novelty_activation", "relu")),
            gradient_clip_norm=self._cfg("novelty_grad_clip_norm", self.grad_clip_norm),
        ).to(self.device)
        self._init_nlp(self.action_size)
        self._validate_runtime_config()
        logger.info(
            "MAMLAgent initialised | id=%s state=%s actions=%s meta_lr=%.6f inner_lr=%.6f",
            self.agent_id, self.state_size, self.action_size, self.meta_lr, self.inner_lr,
        )

    # ------------------------------------------------------------------
    # Configuration and initialisation
    # ------------------------------------------------------------------
    def _cfg(self, key: str, default: Any = None) -> Any:
        return self.maml_override.get(key, self.maml_config.get(key, default))

    def _validate_runtime_config(self) -> None:
        validate_probability(self.gamma, "maml.gamma")
        validate_positive(self.meta_lr, "maml.meta_lr", strict=True)
        validate_positive(self.inner_lr, "maml.inner_lr", strict=True)
        validate_positive(self.inner_steps, "maml.inner_steps", strict=True)
        validate_positive(self.support_episodes, "maml.support_episodes", strict=True)
        validate_positive(self.query_episodes, "maml.query_episodes", strict=True)
        validate_positive(self.max_trajectory_steps, "maml.max_trajectory_steps", strict=True)
        validate_non_negative(self.entropy_coef, "maml.entropy_coef")
        validate_positive(self.grad_clip_norm, "maml.grad_clip_norm", strict=True)
        validate_positive(self.gradient_explosion_threshold, "maml.gradient_explosion_threshold", strict=True)
        if self.reward_clip is not None:
            validate_positive(self.reward_clip, "maml.reward_clip", strict=True)

    def _build_policy_config(self) -> Dict[str, Any]:
        base_policy_cfg = self.config.get("policy_network", {})
        policy_cfg = dict(base_policy_cfg) if isinstance(base_policy_cfg, dict) else {}
        policy_override = self.config_override.get("policy_network", {})
        if policy_override:
            if not isinstance(policy_override, dict):
                raise InvalidConfigError("config['policy_network'] must be a dictionary when provided.")
            for key, value in policy_override.items():
                if key == "optimizer_config" and isinstance(value, dict):
                    merged_optim = dict(policy_cfg.get("optimizer_config", {}))
                    merged_optim.update(value)
                    policy_cfg["optimizer_config"] = merged_optim
                else:
                    policy_cfg[key] = value

        hidden_size = self._cfg("hidden_size", 64)
        if "hidden_layer_sizes" not in policy_cfg:
            if isinstance(hidden_size, int):
                policy_cfg["hidden_layer_sizes"] = [int(hidden_size), int(hidden_size)]
            elif isinstance(hidden_size, (list, tuple)) and hidden_size:
                policy_cfg["hidden_layer_sizes"] = [int(dim) for dim in hidden_size]
            else:
                raise InvalidConfigError("maml.hidden_size must be an int or a non-empty sequence of ints.")
        policy_cfg["output_activation"] = "softmax"
        optimizer_cfg = dict(policy_cfg.get("optimizer_config", {}))
        optimizer_cfg["learning_rate"] = self.meta_lr
        policy_cfg["optimizer_config"] = optimizer_cfg
        return policy_cfg

    def _build_meta_optimizer(self, model: nn.Module) -> optim.Optimizer:
        optimizer_cfg = dict(self.policy_config)
        nested = dict(optimizer_cfg.get("optimizer_config", {}))
        nested["learning_rate"] = self.meta_lr
        optimizer_cfg["optimizer_config"] = nested
        return create_policy_optimizer(model, optimizer_cfg)

    def _init_nlp(self, action_size: int) -> None:
        # No local optional imports here; language integration can be injected by callers.
        self.nlp_engine = None
        self.vocab_size = coerce_int(self._cfg("vocab_size", max(action_size, 50)), max(action_size, 50), minimum=1)
        self.max_message_length = coerce_int(self._cfg("max_message_length", 10), 10, minimum=1)

    # ------------------------------------------------------------------
    # Environment / task handling
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_reset(env: Any) -> Any:
        reset = getattr(env, "reset", None)
        if not callable(reset):
            raise EnvironmentResetError(env_name=type(env).__name__, message="Task environment does not expose reset().")
        try:
            reset_result = reset()
        except Exception as exc:
            raise EnvironmentResetError(env_name=type(env).__name__, cause=exc) from exc
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            return reset_result[0]
        return reset_result

    @staticmethod
    def _safe_step(env: Any, action: int, step_method: str = "step") -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        step_fn = getattr(env, step_method, None)
        if not callable(step_fn):
            raise EnvironmentStepError(message=f"Task environment does not expose {step_method}().", action=action)
        try:
            step_result = step_fn(action)
        except Exception as exc:
            raise EnvironmentStepError(message=f"Environment {step_method} failed.", action=action, cause=exc) from exc
        if isinstance(step_result, tuple) and len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
            validate_finite(reward, "env_reward")
            return next_state, float(reward), bool(terminated), bool(truncated), info or {}
        if isinstance(step_result, tuple) and len(step_result) == 4:
            next_state, reward, done, info = step_result
            validate_finite(reward, "env_reward")
            return next_state, float(reward), bool(done), False, info or {}
        raise EnvironmentStepError(
            step_output_length=len(step_result) if isinstance(step_result, tuple) else None,
            action=action,
            context={"step_method": step_method, "received_type": type(step_result).__name__},
        )

    def _resolve_task_spec(
        self,
        task: Optional[TaskSpec] = None,
        *,
        split: str = "train",
        sampler: Optional[Callable[..., TaskSpec]] = None,
    ) -> Tuple[Any, Dict[str, Any], bool]:
        if task is None:
            sampler = sampler or self.task_sampler
            if sampler is None:
                raise InvalidConfigError(f"No task provided and no task sampler configured for split={split!r}.")
            try:
                signature = inspect.signature(sampler)
                task = sampler(split=split) if "split" in signature.parameters else sampler()
            except TypeError:
                task = sampler()
            sampled = True
        else:
            sampled = False

        if isinstance(task, dict):
            if "env" not in task:
                raise InvalidConfigError("Task dictionary must contain an 'env' key.", context={"keys": list(task.keys())})
            return task["env"], _normalise_task_info(task.get("task_info", {})), sampled
        if isinstance(task, tuple) and len(task) == 2 and isinstance(task[1], dict):
            return task[0], _normalise_task_info(task[1]), sampled
        return task, {}, sampled

    @staticmethod
    def _close_env_if_possible(env: Any) -> None:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    def _sample_training_task(self) -> Tuple[Any, Dict[str, Any], bool]:
        return self._resolve_task_spec(None, split="train")

    def _sample_evaluation_task(self) -> Tuple[Any, Dict[str, Any], bool]:
        return self._resolve_task_spec(None, split="eval")

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------
    def clone_policy(self, policy_to_clone: Optional[PolicyNetwork] = None) -> PolicyNetwork:
        policy_to_clone = policy_to_clone or self.policy
        cloned = create_policy_network(self.state_size, self.action_size, config=self.policy_config).to(self.device)
        cloned.load_state_dict(copy.deepcopy(policy_to_clone.state_dict()))
        return cloned

    def _named_parameters_dict(self, policy: Optional[PolicyNetwork] = None) -> OrderedDict[str, torch.Tensor]:
        policy = policy or self.policy
        return OrderedDict((name, param) for name, param in policy.named_parameters())

    def _regularization_penalty(
        self,
        policy: Optional[PolicyNetwork] = None,
        params: Optional[MutableMapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        policy = policy or self.policy
        if params is None and hasattr(policy, "regularization_penalty"):
            return policy.regularization_penalty()
        l1_lambda = float(getattr(policy, "l1_lambda", 0.0))
        l2_lambda = float(getattr(policy, "l2_lambda", 0.0))
        penalty = torch.zeros((), device=self.device)
        if params is not None:
            tensors = list(params.values())
        else:
            tensors = list(policy.parameters())
        if l1_lambda > 0.0:
            penalty = penalty + l1_lambda * sum(param.abs().sum() for param in tensors)
        if l2_lambda > 0.0:
            penalty = penalty + l2_lambda * sum(param.pow(2).sum() for param in tensors)
        return penalty

    def _policy_forward(
        self,
        state_batch: torch.Tensor,
        *,
        current_policy: Optional[PolicyNetwork] = None,
        params: Optional[MutableMapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        current_policy = current_policy or self.policy
        if params is None:
            output = current_policy(state_batch)
        else:
            output = functional_call(current_policy, params, (state_batch,))
        _finite_tensor(output, "policy_output")
        return output

    def _policy_distribution(
        self,
        state_batch: torch.Tensor,
        *,
        current_policy: Optional[PolicyNetwork] = None,
        params: Optional[MutableMapping[str, torch.Tensor]] = None,
    ) -> torch.distributions.Categorical:
        probs = self._policy_forward(state_batch, current_policy=current_policy, params=params)
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
        if probs.shape[-1] != self.action_size:
            raise ActionSpaceMismatchError(action="policy_output", expected_space=f"{self.action_size} discrete actions")
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        _finite_tensor(probs, "policy_probabilities")
        return torch.distributions.Categorical(probs=probs)

    def get_action(
        self,
        state: TensorLike, # type: ignore
        current_policy: Optional[PolicyNetwork] = None,
        is_speaking_task: bool = False,
        deterministic: bool = False,
        params: Optional[MutableMapping[str, torch.Tensor]] = None,
    ) -> Tuple[int, torch.Tensor, Optional[int]]:
        current_policy = current_policy or self.policy
        state_tensor = _state_to_tensor(state, self.device, expected_dim=self.state_size).unsqueeze(0)
        with torch.set_grad_enabled(params is not None or current_policy.training):
            dist = self._policy_distribution(state_tensor, current_policy=current_policy, params=params)
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            if action.numel() != 1:
                raise InvalidActionError(action=action.detach().cpu().tolist(), reason="expected scalar discrete action")
            log_prob = dist.log_prob(action).squeeze(0)
        action_item = int(action.item())
        if not 0 <= action_item < self.action_size:
            raise ActionSpaceMismatchError(action=action_item, expected_space=f"[0, {self.action_size - 1}]")
        return action_item, log_prob, action_item if is_speaking_task else None

    def select_action(self, processed_state: TensorLike, deterministic: bool = False) -> int: # type: ignore
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
        del partner_agent
        current_policy = current_policy or self.policy
        trajectory: List[Transition] = []
        state = self._safe_reset(env)
        current_message_sequence: List[int] = []
        env_limit = getattr(getattr(env, "spec", None), "max_episode_steps", self.max_trajectory_steps)
        episode_limit = min(self.max_trajectory_steps, coerce_int(env_limit, self.max_trajectory_steps, minimum=1))
        previous_mode = current_policy.training
        current_policy.train(not deterministic)
        try:
            for _ in range(episode_limit):
                state_tensor = _state_to_tensor(state, self.device, expected_dim=self.state_size)
                action, log_prob, msg_token = self.get_action(
                    state_tensor,
                    current_policy=current_policy,
                    is_speaking_task=is_speaking_task,
                    deterministic=deterministic,
                    params=params,
                )
                if is_speaking_task and msg_token is not None:
                    current_message_sequence.append(msg_token)
                    step_method = "step_speaker" if hasattr(env, "step_speaker") and len(current_message_sequence) < self.max_message_length else "step"
                    next_state, reward, done, truncated, _ = self._safe_step(env, action, step_method=step_method)
                    full_message = tuple(current_message_sequence) if len(current_message_sequence) >= self.max_message_length else None
                    if full_message is not None:
                        current_message_sequence = []
                else:
                    next_state, reward, done, truncated, _ = self._safe_step(env, action)
                    full_message = None

                reward = self._process_reward(reward)
                transition = Transition(
                    state=state_tensor.detach().cpu(),
                    action=int(action),
                    reward=float(reward),
                    log_prob=log_prob,
                    message=full_message,
                )
                trajectory.append(transition)
                self.reward_stats.update(float(reward))
                self.recent_rewards.append(float(reward))
                self.calculations.update_performance(float(reward))
                if store_in_memory:
                    self.learning_memory.add(
                        {"tag": tag or "trajectory", "agent_id": self.agent_id, "transition": transition},
                        priority=abs(float(reward)) + 1e-6,
                        tag=tag or "trajectory",
                    )
                state = next_state
                if done or truncated:
                    break
        finally:
            current_policy.train(previous_mode)
        self.total_rollouts += 1
        self.episode_length_stats.update(len(trajectory))
        return trajectory

    def _process_reward(self, reward: float) -> float:
        validate_finite(reward, "reward")
        value = float(reward)
        if self.reward_clip is not None:
            clip = float(self.reward_clip)
            value = clamp(value, -clip, clip)
        return value

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
        validate_positive(episodes, "collect_rollouts.episodes", strict=True)
        trajectories: List[List[Transition]] = []
        for _ in range(int(episodes)):
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
        returns = self.calculations.discounted_returns(rewards, gamma=self.gamma)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        if self.normalize_returns and returns_tensor.numel() > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-8)
        _finite_tensor(returns_tensor, "discounted_returns")
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
        states = torch.stack([_state_to_tensor(t.state, self.device, expected_dim=self.state_size) for t in trajectory])
        actions = torch.as_tensor([int(t.action) for t in trajectory], dtype=torch.long, device=self.device)
        returns = self._discounted_returns([float(t.reward) for t in trajectory])
        dist = self._policy_distribution(states, current_policy=policy_to_evaluate, params=params)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        loss = -(log_probs * returns.detach()).mean()
        if self.entropy_coef > 0.0:
            loss = loss - self.entropy_coef * entropy
        loss = loss + self._regularization_penalty(policy=policy_to_evaluate, params=params)
        _finite_tensor(loss, "policy_loss")
        return loss

    def compute_loss_from_trajectories(
        self,
        trajectories: Sequence[Sequence[Transition]],
        policy_to_evaluate: Optional[PolicyNetwork] = None,
        params: Optional[MutableMapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        losses = [
            self.compute_loss_from_trajectory(t, policy_to_evaluate=policy_to_evaluate, params=params)
            for t in trajectories
            if t
        ]
        if not losses:
            return torch.zeros((), device=self.device, requires_grad=True)
        loss = torch.stack(losses).mean()
        _finite_tensor(loss, "trajectory_batch_loss")
        return loss

    # ------------------------------------------------------------------
    # Reward shaping and metrics
    # ------------------------------------------------------------------
    def _calculate_novelty(self, states: Sequence[TensorLike]) -> float: # type: ignore
        if not states:
            return 0.0
        states_tensor = torch.stack([_state_to_tensor(s, self.device, expected_dim=self.state_size) for s in states])
        if self.train_novelty_detector:
            self.nd_network.train_step(states_tensor)
        with torch.no_grad():
            scores = self.nd_network(states_tensor)
        _finite_tensor(scores, "novelty_scores")
        return float(scores.mean().item())

    @staticmethod
    def _communication_success(env: Any) -> float:
        value = getattr(env, "communication_success", 0.0)
        try:
            return clamp01(float(_call_or_value(value)))
        except Exception:
            return 0.0

    @staticmethod
    def _task_completed(env: Any) -> bool:
        value = getattr(env, "task_completed", False)
        try:
            return bool(_call_or_value(value))
        except Exception:
            return False

    def _apply_task_reward_adjustments(self, trajectory: Sequence[Transition], env: Any) -> Tuple[List[Transition], Dict[str, Any]]:
        if not trajectory:
            return [], {
                "extrinsic_reward": 0.0,
                "intrinsic_reward": 0.0,
                "communication_success": 0.0,
                "task_success": 0.0,
                "task_completion_bonus": 0.0,
                "communication_bonus": 0.0,
                "adjusted_reward": 0.0,
            }
        extrinsic_reward = float(sum(t.reward for t in trajectory))
        novelty_bonus = self._calculate_novelty([t.state for t in trajectory])
        intrinsic_reward = self.novelty_coef * novelty_bonus
        communication_success = self._communication_success(env)
        communication_bonus = self.communication_bonus_coef * communication_success
        task_success = 1.0 if self._task_completed(env) else 0.0
        task_completion_bonus = self.task_completion_bonus_coef * task_success
        total_bonus = intrinsic_reward + communication_bonus + task_completion_bonus
        per_step_bonus = safe_divide(total_bonus, len(trajectory), default=0.0)
        adjusted = [Transition(t.state, t.action, self._process_reward(float(t.reward) + per_step_bonus), t.log_prob, t.message) for t in trajectory]
        metrics: Dict[str, Any] = {
            "extrinsic_reward": extrinsic_reward,
            "intrinsic_reward": float(intrinsic_reward),
            "communication_success": float(communication_success),
            "task_success": float(task_success),
            "task_completion_bonus": float(task_completion_bonus),
            "communication_bonus": float(communication_bonus),
            "adjusted_reward": float(sum(t.reward for t in adjusted)),
        }
        return adjusted, metrics

    def _compute_task_metrics(self, trajectory: Sequence[Transition], env: Any, *, apply_reward_adjustment: bool = False) -> Dict[str, Any]:
        adjusted_trajectory, metrics = self._apply_task_reward_adjustments(trajectory, env)
        rewards = [float(t.reward) for t in trajectory]
        reward_summary = self.calculations.summarize_rewards(rewards)
        metrics.update(
            {
                "episode_length": len(trajectory),
                "raw_reward_mean": reward_summary["mean"],
                "reward_std": reward_summary["std"],
                "reward_sum": reward_summary["sum"],
            }
        )
        if apply_reward_adjustment:
            metrics["adjusted_trajectory"] = adjusted_trajectory
        return metrics

    # ------------------------------------------------------------------
    # Adaptation and meta-objective
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
        inner_steps = coerce_int(inner_steps or self.inner_steps, self.inner_steps, minimum=1)
        support_episodes = coerce_int(support_episodes or self.support_episodes, self.support_episodes, minimum=1)
        create_graph = (not self.first_order) if create_graph is None else bool(create_graph)
        fast_params: OrderedDict[str, torch.Tensor] = self._named_parameters_dict(current_meta_policy)
        for step_idx in range(inner_steps):
            support = self.collect_rollouts(
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
                support = [self._compute_task_metrics(t, env, apply_reward_adjustment=True)["adjusted_trajectory"] for t in support]
            support_loss = self.compute_loss_from_trajectories(support, policy_to_evaluate=current_meta_policy, params=fast_params)
            grads = torch.autograd.grad(
                support_loss,
                tuple(fast_params.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=True,
            )
            new_params: OrderedDict[str, torch.Tensor] = OrderedDict()
            for (name, param), grad in zip(fast_params.items(), grads):
                if grad is None:
                    new_params[name] = param
                    continue
                _finite_tensor(grad, f"inner_grad.{name}")
                new_params[name] = param - self.inner_lr * grad
            fast_params = new_params
            self.total_inner_updates += 1
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
        validate_non_empty_sequence(tasks, "tasks")
        inner_steps = coerce_int(inner_steps or self.inner_steps, self.inner_steps, minimum=1)
        create_graph = (not self.first_order) if create_graph is None else bool(create_graph)
        apply_reward_shaping = self.use_reward_shaping if apply_reward_shaping is None else bool(apply_reward_shaping)
        task_losses: List[torch.Tensor] = []
        query_rewards: List[float] = []
        intrinsic_rewards: List[float] = []
        communication_scores: List[float] = []
        task_success_scores: List[float] = []
        episode_lengths: List[int] = []
        failures: List[str] = []

        for task in tasks:
            env, task_info, sampled = self._resolve_task_spec(task, split="train", sampler=sampler)
            is_speaking = bool(task_info.get("is_speaking", False))
            partner = task_info.get("partner_agent")
            try:
                fast_params = self.inner_update(
                    env,
                    self.policy,
                    is_speaking_task=is_speaking,
                    partner_agent=partner,
                    inner_steps=inner_steps,
                    support_episodes=coerce_int(task_info.get("support_episodes", self.support_episodes), self.support_episodes, minimum=1),
                    create_graph=create_graph,
                    return_params=True,
                )
                query_trajectories = self.collect_rollouts(
                    env,
                    current_policy=self.policy,
                    params=fast_params,
                    episodes=coerce_int(task_info.get("query_episodes", self.query_episodes), self.query_episodes, minimum=1),
                    is_speaking_task=is_speaking,
                    partner_agent=partner,
                    deterministic=False,
                    tag="query",
                    store_in_memory=True,
                )
                processed: List[List[Transition]] = []
                for trajectory in query_trajectories:
                    metrics = self._compute_task_metrics(trajectory, env, apply_reward_adjustment=apply_reward_shaping)
                    processed.append(metrics["adjusted_trajectory"] if apply_reward_shaping else list(trajectory))
                    query_rewards.append(float(metrics.get("adjusted_reward", metrics.get("reward_sum", 0.0))))
                    intrinsic_rewards.append(float(metrics.get("intrinsic_reward", 0.0)))
                    communication_scores.append(float(metrics.get("communication_success", 0.0)))
                    task_success_scores.append(float(metrics.get("task_success", 0.0)))
                    episode_lengths.append(int(metrics.get("episode_length", 0)))
                query_loss = self.compute_loss_from_trajectories(processed, policy_to_evaluate=self.policy, params=fast_params)
                task_losses.append(query_loss)
            except LearningError as exc:
                failures.append(type(exc).__name__)
                if self.fail_fast_task_errors:
                    raise
                logger.warning("Skipping failed MAML task: %s", exc)
            finally:
                if sampled:
                    self._close_env_if_possible(env)

        if not task_losses:
            raise TrainingError("No valid task losses were produced during meta-objective.", context={"failures": failures})
        meta_loss = torch.stack(task_losses).mean()
        _finite_tensor(meta_loss, "meta_loss")
        reward_summary = self.calculations.summarize_rewards(query_rewards)
        diagnostics = {
            "query_reward": reward_summary["mean"],
            "query_reward_std": reward_summary["std"],
            "intrinsic_reward": _safe_mean(intrinsic_rewards),
            "communication_success": _safe_mean(communication_scores),
            "task_success_rate": _safe_mean(task_success_scores),
            "avg_episode_length": _safe_mean(episode_lengths),
            "task_count": len(task_losses),
            "failed_tasks": len(failures),
        }
        return meta_loss, diagnostics

    def _post_backward_checks(self) -> float:
        gradients = [param.grad for param in self.policy.parameters() if param.grad is not None]
        for grad in gradients:
            _finite_tensor(grad, "meta_gradient")
        total_grad_norm = self.calculations.gradient_global_norm(gradients)
        validate_finite(total_grad_norm, "meta_gradient_norm")
        if total_grad_norm > self.gradient_explosion_threshold:
            raise GradientExplosionError(total_grad_norm, self.gradient_explosion_threshold)
        self.calculations.clip_gradients_by_global_norm(gradients, self.grad_clip_norm)
        self.grad_stats.update(total_grad_norm)
        return float(total_grad_norm)

    def compute_meta_gradient_contribution(self, tasks_for_agent: Sequence[TaskSpec], inner_steps: Optional[int] = None) -> float:
        self.meta_optimizer.zero_grad(set_to_none=True)
        meta_loss, diagnostics = self._meta_objective(tasks_for_agent, inner_steps=inner_steps, create_graph=not self.first_order)
        meta_loss.backward()
        diagnostics["grad_norm"] = self._post_backward_checks()
        diagnostics["meta_loss"] = float(meta_loss.detach().item())
        self.last_meta_metrics = diagnostics
        return float(meta_loss.detach().item())

    def meta_update(self, tasks: Sequence[TaskSpec], inner_steps: Optional[int] = None) -> float:
        self.meta_optimizer.zero_grad(set_to_none=True)
        meta_loss, diagnostics = self._meta_objective(tasks, inner_steps=inner_steps, create_graph=not self.first_order)
        meta_loss.backward()
        diagnostics["grad_norm"] = self._post_backward_checks()
        self.meta_optimizer.step()
        value = float(meta_loss.detach().item())
        self.loss_stats.update(value)
        self.recent_losses.append(value)
        self.total_meta_updates += 1
        diagnostics["meta_loss"] = value
        diagnostics["reward_trend"] = self.calculations.calculate_performance_trend()
        self.last_meta_metrics = diagnostics
        return value

    # ------------------------------------------------------------------
    # Public training / evaluation API
    # ------------------------------------------------------------------
    def learn_step(self, trajectory: Sequence[Transition]) -> float:
        validate_non_empty_sequence(trajectory, "trajectory")
        self.meta_optimizer.zero_grad(set_to_none=True)
        loss = self.compute_loss_from_trajectory(trajectory, self.policy)
        loss.backward()
        self._post_backward_checks()
        self.meta_optimizer.step()
        value = float(loss.detach().item())
        self.loss_stats.update(value)
        return value

    def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(task_data, dict):
            raise InvalidConfigError("execute expects a task_data dictionary.")
        tasks = task_data.get("tasks")
        if not tasks:
            num_tasks = coerce_int(task_data.get("num_tasks", task_data.get("tasks_per_batch", 1)), 1, minimum=1)
            tasks = [self._sample_training_task()[0:2] for _ in range(num_tasks)]
        loss = self.meta_update(tasks, inner_steps=task_data.get("inner_steps"))
        return {"status": "success", "agent": "MAMLAgent", "meta_loss": loss, "metrics": copy.deepcopy(self.last_meta_metrics or {})}

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
        validate_positive(num_meta_epochs, "num_meta_epochs", strict=True)
        validate_positive(tasks_per_epoch, "tasks_per_epoch", strict=True)
        task_sampler = task_sampler or self.task_sampler
        if task_sampler is None:
            raise InvalidConfigError("train() requires a task_sampler or agent-level task_sampler.")
        adaptation_steps = coerce_int(adaptation_steps or self.inner_steps, self.inner_steps, minimum=1)
        validation_freq = coerce_int(validation_freq if validation_freq is not None else self._cfg("validation_freq", 0), 0, minimum=0)
        validation_tasks = coerce_int(validation_tasks, 5, minimum=1)
        early_stop_patience = early_stop_patience if early_stop_patience is not None else self.config.get("unified", {}).get("early_stop_patience")
        early_stop_patience = None if early_stop_patience in (None, "", "None", "none") else int(early_stop_patience)
        checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else self.checkpoint_dir
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        best_validation_reward = float("-inf")
        early_stop_counter = 0
        self.training_metrics = defaultdict(list)

        for epoch in range(int(num_meta_epochs)):
            tasks = [self._resolve_task_spec(None, split="train", sampler=task_sampler)[:2] for _ in range(int(tasks_per_epoch))]
            meta_loss = self.meta_update(tasks, inner_steps=adaptation_steps)
            epoch_metrics = copy.deepcopy(self.last_meta_metrics or {})
            epoch_metrics["meta_loss"] = meta_loss
            for key, value in epoch_metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    self.training_metrics[key].append(float(value))
            logger.info("MAML epoch %s/%s | meta_loss=%.4f query_reward=%.3f grad_norm=%.3f", epoch + 1, num_meta_epochs, meta_loss, epoch_metrics.get("query_reward", 0.0), epoch_metrics.get("grad_norm", 0.0))
            if validation_freq > 0 and ((epoch + 1) % validation_freq == 0):
                val_metrics = self.evaluate(None, validation_tasks, adaptation_steps, meta_eval=False, task_sampler=task_sampler)
                validation_reward = float(val_metrics.get("adapted_performance", val_metrics.get("baseline_performance", 0.0)))
                self.training_metrics["validation_reward"].append(validation_reward)
                if validation_reward > best_validation_reward:
                    best_validation_reward = validation_reward
                    early_stop_counter = 0
                    self.save(checkpoint_root / f"best_epoch_{epoch + 1}.pt")
                else:
                    early_stop_counter += 1
                if early_stop_patience is not None and early_stop_counter >= early_stop_patience:
                    break
                if target_reward is not None and validation_reward >= float(target_reward):
                    break
        return {key: list(values) for key, values in self.training_metrics.items()}

    def evaluate(
        self,
        env: Optional[Any],
        num_eval_tasks: int = 20,
        adaptation_steps: int = 3,
        meta_eval: bool = False,
        task_sampler: Optional[Callable[..., TaskSpec]] = None,
    ) -> Dict[str, Any]:
        validate_positive(num_eval_tasks, "num_eval_tasks", strict=True)
        validate_positive(adaptation_steps, "adaptation_steps", strict=True)
        baseline_returns: List[float] = []
        adapted_returns: List[float] = []
        gains: List[float] = []
        communication_scores: List[float] = []
        task_success_scores: List[float] = []
        episode_lengths: List[int] = []
        reward_components: Dict[str, List[float]] = defaultdict(list)
        for _ in range(int(num_eval_tasks)):
            if env is None:
                task_env, task_info, sampled = self._resolve_task_spec(None, split="eval", sampler=task_sampler or self.task_sampler)
            else:
                task_env, task_info, sampled = self._resolve_task_spec(env, split="eval")
            is_speaking = bool(task_info.get("is_speaking", False))
            partner = task_info.get("partner_agent")
            try:
                baseline = self.collect_rollouts(task_env, current_policy=self.policy, episodes=self.query_episodes, is_speaking_task=is_speaking, partner_agent=partner, deterministic=self.eval_deterministic)
                baseline_return = _safe_mean([sum(t.reward for t in traj) for traj in baseline])
                baseline_returns.append(baseline_return)
                if not meta_eval:
                    fast_params = self.inner_update(task_env, self.policy, is_speaking_task=is_speaking, partner_agent=partner, inner_steps=adaptation_steps, support_episodes=self.support_episodes, create_graph=False, return_params=True, deterministic=False)
                    adapted = self.collect_rollouts(task_env, current_policy=self.policy, params=fast_params, episodes=self.query_episodes, is_speaking_task=is_speaking, partner_agent=partner, deterministic=self.eval_deterministic)
                    returns = [sum(t.reward for t in traj) for traj in adapted]
                    adapted_return = _safe_mean(returns)
                    adapted_returns.append(adapted_return)
                    gains.append(adapted_return - baseline_return)
                    for trajectory in adapted:
                        metrics = self._compute_task_metrics(trajectory, task_env, apply_reward_adjustment=False)
                        communication_scores.append(float(metrics.get("communication_success", 0.0)))
                        task_success_scores.append(float(metrics.get("task_success", 0.0)))
                        episode_lengths.append(int(metrics.get("episode_length", 0)))
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)) and (key.endswith("reward") or key.endswith("bonus")):
                                reward_components[key].append(float(value))
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
            "adaptation_steps": int(adaptation_steps),
            "meta_evaluation": bool(meta_eval),
        }
        if not meta_eval:
            metrics.update({
                "adapted_performance": _safe_mean(adapted_returns),
                "adapted_std": _safe_std(adapted_returns),
                "adaptation_gain": _safe_mean(gains),
                "communication_accuracy": _safe_mean(communication_scores),
                "task_success_rate": _safe_mean(task_success_scores),
                "avg_episode_length": _safe_mean(episode_lengths),
                "reward_components": {key: _safe_mean(values) for key, values in reward_components.items()},
            })
        return metrics

    # ------------------------------------------------------------------
    # Diagnostics and persistence
    # ------------------------------------------------------------------
    def diagnostics(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "total_rollouts": self.total_rollouts,
            "total_meta_updates": self.total_meta_updates,
            "total_inner_updates": self.total_inner_updates,
            "last_meta_metrics": to_json_safe(self.last_meta_metrics or {}),
            "reward_stats": to_json_safe(self.reward_stats.snapshot()),
            "loss_stats": to_json_safe(self.loss_stats.snapshot()),
            "grad_stats": to_json_safe(self.grad_stats.snapshot()),
            "episode_length_stats": to_json_safe(self.episode_length_stats.snapshot()),
            "recent_reward_summary": self.calculations.summarize_rewards(list(self.recent_rewards)),
            "recent_loss_summary": self.calculations.summarize_rewards(list(self.recent_losses)),
        }

    def get_checkpoint(self) -> Dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "agent_id": self.agent_id,
            "model_id": self.model_id,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "config": copy.deepcopy(self.config),
            "maml_config": copy.deepcopy(self.maml_config),
            "policy_state_dict": self.policy.state_dict(),
            "meta_optimizer_state_dict": self.meta_optimizer.state_dict(),
            "novelty_detector_state_dict": self.nd_network.state_dict(),
            "training_metrics": {key: list(values) for key, values in self.training_metrics.items()},
            "last_meta_metrics": copy.deepcopy(self.last_meta_metrics),
            "diagnostics": self.diagnostics(),
        }

    def save(self, path: Union[str, os.PathLike[str]]) -> None:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.get_checkpoint(), target)
            logger.info("Saved MAML checkpoint to %s", target)
        except Exception as exc:
            raise CheckpointError(str(target), operation="save", cause=exc) from exc

    def load(self, path: Union[str, os.PathLike[str]], strict: bool = True) -> None:
        source = Path(path)
        try:
            checkpoint = torch.load(source, map_location=self.device)
            if not isinstance(checkpoint, Mapping):
                raise CheckpointError(str(source), operation="load", message="MAML checkpoint is not a mapping")
            self.policy.load_state_dict(checkpoint["policy_state_dict"], strict=strict)
            self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer_state_dict"])
            nd_state = checkpoint.get("novelty_detector_state_dict")
            if nd_state is not None:
                self.nd_network.load_state_dict(nd_state, strict=False)
            self.training_metrics = defaultdict(list, checkpoint.get("training_metrics", {}))
            self.last_meta_metrics = checkpoint.get("last_meta_metrics")
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(str(source), operation="load", cause=exc) from exc

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, os.PathLike[str]],
        *,
        task_sampler: Optional[Callable[..., TaskSpec]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "MAMLAgent":
        checkpoint = torch.load(Path(path), map_location=device or "cpu")
        if not isinstance(checkpoint, Mapping):
            raise CheckpointError(str(path), operation="load", message="MAML checkpoint is not a mapping")
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
    ) -> None:
        validate_positive(num_agents, "num_agents", strict=True)
        if env_creator_fn is None or not callable(env_creator_fn):
            raise InvalidConfigError("env_creator_fn must be callable.")
        self.num_agents = int(num_agents)
        self.global_config = global_config if global_config is not None else load_global_config()
        if not isinstance(self.global_config, dict):
            raise InvalidConfigError("global_config must be a dictionary.")
        self.agent_config = agent_config or {}
        if not isinstance(self.agent_config, dict):
            raise InvalidConfigError("agent_config must be a dictionary when provided.")
        self.env_creator_fn = env_creator_fn
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.maml_config = get_config_section("maml", config=self.global_config) or {}
        self.diffusion_type = str(self.maml_config.get("diffusion_type", "average")).lower()
        self.meta_epochs = coerce_int(self.maml_config.get("meta_epochs", 100), 100, minimum=1)
        self.tasks_per_agent_meta_batch = coerce_int(self.maml_config.get("tasks_per_agent_meta_batch", 5), 5, minimum=1)
        self.inner_steps = coerce_int(self.maml_config.get("inner_steps", 1), 1, minimum=1)
        self.adj_matrix = self._setup_adjacency(self.maml_config.get("adjacency_matrix"))
        self.calculations = LearningCalculations()
        self.fleet_metrics: Dict[str, List[float]] = defaultdict(list)
        self.agents = [
            MAMLAgent(i, self.state_size, self.action_size, config=self.agent_config, task_sampler=self.env_creator_fn, device=self.device)
            for i in range(self.num_agents)
        ]
        logger.info("DecentralizedMAMLFleet initialised | agents=%s diffusion=%s", self.num_agents, self.diffusion_type)

    def _setup_adjacency(self, adj_matrix_config: Optional[Any]) -> torch.Tensor:
        if adj_matrix_config in {None, "fully_connected", "average"}:
            adj = np.ones((self.num_agents, self.num_agents), dtype=np.float32)
        elif adj_matrix_config == "ring":
            adj = np.eye(self.num_agents, dtype=np.float32)
            for idx in range(self.num_agents):
                adj[idx, (idx - 1) % self.num_agents] = 1.0
                adj[idx, (idx + 1) % self.num_agents] = 1.0
        elif isinstance(adj_matrix_config, list):
            adj = np.asarray(adj_matrix_config, dtype=np.float32)
        else:
            raise InvalidConfigError("adjacency_matrix must be None, 'fully_connected', 'ring', or a numeric square matrix.")
        if adj.shape != (self.num_agents, self.num_agents):
            raise InvalidConfigError("Adjacency matrix shape mismatch.", context={"expected": (self.num_agents, self.num_agents), "actual": adj.shape})
        if np.any(adj < 0.0):
            raise InvalidConfigError("Adjacency weights must be non-negative.")
        row_sums = adj.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise InvalidConfigError("Each adjacency row must contain at least one positive weight.")
        return torch.as_tensor(adj / row_sums, dtype=torch.float32, device=self.device)

    def _clone_optimizer_for_policy(self, agent: MAMLAgent, policy: PolicyNetwork) -> optim.Optimizer:
        optimizer = agent._build_meta_optimizer(policy)
        return optimizer

    def _candidate_state_dicts(self) -> Tuple[List[Dict[str, torch.Tensor]], List[float], List[float]]:
        candidates: List[Dict[str, torch.Tensor]] = []
        losses: List[float] = []
        grad_norms: List[float] = []
        for agent in self.agents:
            tasks = [self.env_creator_fn() for _ in range(self.tasks_per_agent_meta_batch)]
            meta_loss = agent.compute_meta_gradient_contribution(tasks, inner_steps=self.inner_steps)
            losses.append(float(meta_loss))
            grad_norms.append(float((agent.last_meta_metrics or {}).get("grad_norm", 0.0)))
            temp_policy = agent.clone_policy(agent.policy)
            temp_optimizer = self._clone_optimizer_for_policy(agent, temp_policy)
            temp_optimizer.zero_grad(set_to_none=True)
            for src, dst in zip(agent.policy.parameters(), temp_policy.parameters()):
                if src.grad is not None:
                    dst.grad = src.grad.detach().clone()
            temp_optimizer.step()
            candidates.append(copy.deepcopy(temp_policy.state_dict()))
        return candidates, losses, grad_norms

    def _diffuse_parameters(self, candidate_policy_params_list: Sequence[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        validate_non_empty_sequence(candidate_policy_params_list, "candidate_policy_params_list")
        keys = list(candidate_policy_params_list[0].keys())
        blended_states: List[Dict[str, torch.Tensor]] = []
        for target_idx in range(self.num_agents):
            blended: Dict[str, torch.Tensor] = {}
            for key in keys:
                template = candidate_policy_params_list[0][key]
                acc = torch.zeros_like(template)
                for source_idx in range(self.num_agents):
                    acc = acc + float(self.adj_matrix[target_idx, source_idx].item()) * candidate_policy_params_list[source_idx][key]
                blended[key] = acc
            blended_states.append(blended)
        return blended_states

    def train_fleet(self) -> Dict[str, List[float]]:
        self.fleet_metrics = defaultdict(list)
        for epoch in range(self.meta_epochs):
            candidates, losses, grad_norms = self._candidate_state_dicts()
            for agent, new_params in zip(self.agents, self._diffuse_parameters(candidates)):
                agent.policy.load_state_dict(new_params)
                agent.meta_optimizer = agent._build_meta_optimizer(agent.policy)
            avg_loss = _safe_mean(losses)
            avg_grad = _safe_mean(grad_norms)
            self.fleet_metrics["meta_loss"].append(avg_loss)
            self.fleet_metrics["grad_norm"].append(avg_grad)
            logger.info("Fleet epoch %s/%s | avg_meta_loss=%.4f avg_grad_norm=%.4f", epoch + 1, self.meta_epochs, avg_loss, avg_grad)
        return {key: list(values) for key, values in self.fleet_metrics.items()}

    def evaluate_fleet(self, num_eval_tasks_per_agent: int = 10, adaptation_steps: Optional[int] = None) -> Dict[str, Any]:
        validate_positive(num_eval_tasks_per_agent, "num_eval_tasks_per_agent", strict=True)
        adaptation_steps = coerce_int(adaptation_steps or self.inner_steps, self.inner_steps, minimum=1)
        rewards: List[float] = []
        gains: List[float] = []
        for agent in self.agents:
            metrics = agent.evaluate(None, num_eval_tasks_per_agent, adaptation_steps, meta_eval=False, task_sampler=self.env_creator_fn)
            rewards.append(float(metrics.get("adapted_performance", metrics.get("baseline_performance", 0.0))))
            gains.append(float(metrics.get("adaptation_gain", 0.0)))
        return {
            "overall_average_reward": _safe_mean(rewards),
            "overall_adaptation_gain": _safe_mean(gains),
            "agent_rewards": rewards,
            "agent_adaptation_gains": gains,
        }


def _infer_env_dimensions(env: Any) -> Tuple[int, int]:
    observation_space = getattr(env, "observation_space", None)
    action_space = getattr(env, "action_space", None)
    if observation_space is None or action_space is None:
        raise InvalidConfigError("Unable to infer state/action dimensions: observation_space or action_space missing.")
    obs_shape = getattr(observation_space, "shape", None)
    if obs_shape is None:
        raise InvalidConfigError("Environment observation_space does not expose a shape attribute.")
    state_size = int(np.prod(obs_shape)) if len(obs_shape) > 0 else 1
    if not hasattr(action_space, "n"):
        raise InvalidConfigError("MAMLAgent supports discrete action spaces with an 'n' attribute.")
    action_size = int(action_space.n)
    validate_positive(state_size, "state_size", strict=True)
    validate_positive(action_size, "action_size", strict=True)
    return state_size, action_size


if __name__ == "__main__":
    print("\n=== Running  Model-Agnostic Meta-Learning ===\n")
    printer.status("TEST", " Model-Agnostic Meta-Learning initialized", "info")
    torch.manual_seed(7); np.random.seed(7)

    class _Space:
        def __init__(self, shape=None, n=None): self.shape, self.n = shape, n
    class _MiniEnv:
        observation_space = _Space(shape=(4,)); action_space = _Space(n=2)
        def __init__(self): self.step_i = 0; self.state = np.zeros(4, dtype=np.float32)
        def reset(self): self.step_i = 0; self.state = np.array([0.2, -0.1, 0.0, 0.1], dtype=np.float32); return self.state, {}
        def step(self, action):
            self.step_i += 1; self.state = self.state + (0.05 if int(action) == 1 else -0.03)
            reward = 1.0 - float(np.linalg.norm(self.state[:2])); done = self.step_i >= 3
            return self.state.astype(np.float32), reward, done, False, {}
        def close(self): pass
    def sampler(split="train"):
        return _MiniEnv(), {"support_episodes": 1, "query_episodes": 1}

    ss, aa = _infer_env_dimensions(_MiniEnv())
    agent = MAMLAgent("maml_test", ss, aa, task_sampler=sampler, config={"maml": {"max_trajectory_steps": 3, "inner_steps": 1, "support_episodes": 1, "query_episodes": 1, "first_order": True, "train_novelty_detector": False}})
    traj = agent.collect_trajectory(_MiniEnv())
    assert len(traj) > 0 and isinstance(agent.select_action(np.zeros(4, dtype=np.float32)), int)
    loss = agent.meta_update([sampler()], inner_steps=1)
    assert math.isfinite(loss) and agent.last_meta_metrics is not None
    metrics = agent.evaluate(None, num_eval_tasks=1, adaptation_steps=1, task_sampler=sampler)
    assert "baseline_performance" in metrics
    p = Path("maml_agent_test.pt"); agent.save(p); agent.load(p); p.unlink(missing_ok=True)
    printer.status("TEST", "MAML rollout, update, evaluation, and checkpoint verified", "success")
    print("\n=== Test ran successfully ===\n")
