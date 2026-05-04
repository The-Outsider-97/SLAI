from __future__ import annotations

import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, ClassVar, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from collections import deque

from ..learning.learning_memory import LearningMemory
from .adaptive_memory import MultiModalMemory
from .imitation_learning_worker import ImitationLearningWorker
from .meta_learning_worker import MetaLearningWorker
from .utils.neural_network import ActorCriticNetwork
from .utils.config_loader import load_global_config, get_config_section
from .utils.adaptive_errors import *
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Skill Worker")
printer = PrettyPrinter


@dataclass
class Transition:
    """Structured on-policy transition for skill-level updates."""

    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    log_prob: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    next_value: Optional[torch.Tensor] = None
    source: str = "rl"


@dataclass
class PolicyUpdateResult:
    """Structured policy-update summary for diagnostics and logging."""

    total_loss: float
    actor_loss: float
    critic_loss: float
    entropy_bonus: float
    mean_return: float
    mean_advantage: float
    num_samples: int
    update_steps: int
    imitation_mixed: bool = False
    imitation_loss: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SkillWorker:
    """
    Reinforcement-learning worker for skill execution.

    Core responsibilities
    ---------------------
    - Learn primitive actions that implement a single skill.
    - Perform actor-critic updates from on-policy trajectory memory.
    - Support optional goal conditioning.
    - Integrate cleanly with Adaptive Memory, Learning Memory, Imitation Learning,
      and Meta Learning workers.
    - Persist and restore operational state safely.

    Production-oriented extensions
    ------------------------------
    - Structured validation and error handling using ``adaptive_errors``.
    - Config-driven actor/critic architecture, optimizers, schedulers, and reward handling.
    - Safer state, goal, action, and reward normalization pipelines.
    - Generalized Advantage Estimation (GAE) and advantage normalization.
    - Full checkpoint/export helpers plus registry management.
    - Rich diagnostics, metrics, and self-test coverage.
    """

    SUPPORTED_OPTIMIZERS: ClassVar[set[str]] = {"adam", "adamw", "sgd", "rmsprop"}
    SUPPORTED_SCHEDULERS: ClassVar[set[str]] = {"none", "step", "cosine", "reduce_on_plateau"}
    _worker_registry: ClassVar[Dict[int, "SkillWorker"]] = {}

    def __init__(
        self,
        skill_id: Optional[int] = None,
        skill_metadata: Optional[Mapping[str, Any]] = None,
        *,
        local_memory: Optional[MultiModalMemory] = None,
        learner_memory: Optional[LearningMemory] = None,
    ) -> None:
        super().__init__()
        self.skill_id: Optional[int] = None
        self.name: Optional[str] = None
        self.skill_metadata: Dict[str, Any] = {}

        self.config = load_global_config()
        section = get_config_section("skill_worker")
        self.worker_config = section if isinstance(section, dict) else {}
        self.actor_critic_config = get_config_section("actor_critic")
        self.rl_config = get_config_section("rl")

        self._load_config()

        self.state_dim: Optional[int] = None
        self.action_dim: Optional[int] = None
        self.goal_dim: int = self.configured_goal_dim if self.enable_goals else 0
        self.enable_goals: bool = self.enable_goals
        self.current_goal: Optional[np.ndarray] = None
        self.input_dim: Optional[int] = None
        self.continuous_actions: bool = self.configured_continuous_action

        self.local_memory = local_memory if local_memory is not None else MultiModalMemory()
        if local_memory is not None:
            ensure_instance(local_memory, MultiModalMemory, "local_memory", component="skill_worker")

        self.learner_memory = learner_memory if learner_memory is not None else LearningMemory()
        if learner_memory is not None:
            ensure_instance(learner_memory, LearningMemory, "learner_memory", component="skill_worker")

        self.actor_critic: Optional[ActorCriticNetwork] = None
        self.actor_optimizer: Optional[torch.optim.Optimizer] = None
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None
        self.actor_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.critic_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        self.meta_learning: Optional[MetaLearningWorker] = None
        self.imitation_learning: Optional[ImitationLearningWorker] = None

        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_std = 1.0
        self.reward_count = 1e-4

        self.exploration_rate = float(self.initial_exploration_rate)
        self.update_counter = 0
        self.episode_counter = 0
        self.step_counter = 0
        self.last_update: Optional[PolicyUpdateResult] = None
        self.last_action_info: Optional[Dict[str, Any]] = None
        self.recent_episode_returns: Deque[float] = deque(maxlen=self.performance_window)
        self.recent_episode_lengths: Deque[int] = deque(maxlen=self.performance_window)
        self.success_history: Deque[int] = deque(maxlen=self.performance_window)
        self.training_history: Dict[str, List[float]] = {
            "actor_loss": [],
            "critic_loss": [],
            "total_loss": [],
            "entropy_bonus": [],
            "mean_return": [],
            "mean_advantage": [],
            "actor_lr": [],
            "critic_lr": [],
        }

        if skill_id is not None or skill_metadata is not None:
            ensure_not_none(skill_id, "skill_id", component="skill_worker")
            ensure_not_none(skill_metadata, "skill_metadata", component="skill_worker")
            self.initialize(int(skill_id), skill_metadata)

    @classmethod
    def create_worker(cls, skill_id: int, skill_metadata: Mapping[str, Any]) -> "SkillWorker":
        worker = cls(skill_id=skill_id, skill_metadata=skill_metadata)
        cls._worker_registry[int(skill_id)] = worker
        return worker

    @classmethod
    def get_worker(cls, skill_id: int) -> Optional["SkillWorker"]:
        return cls._worker_registry.get(int(skill_id))

    @classmethod
    def unregister_worker(cls, skill_id: int) -> None:
        cls._worker_registry.pop(int(skill_id), None)

    @classmethod
    def clear_registry(cls) -> None:
        cls._worker_registry.clear()

    @classmethod
    def registry_snapshot(cls) -> Dict[int, Dict[str, Any]]:
        return {
            worker_id: {
                "name": worker.name,
                "state_dim": worker.state_dim,
                "action_dim": worker.action_dim,
                "input_dim": worker.input_dim,
            }
            for worker_id, worker in cls._worker_registry.items()
        }

    def _load_config(self) -> None:
        try:
            self.enable_goals = bool(self.worker_config.get("enable_goals", False))
            self.configured_goal_dim = int(self.worker_config.get("goal_dim", self.config.get("goal_dim", 0) or 0))
            self.default_actor_layers = list(self.worker_config.get("actor_layers", [64, 64]))
            self.default_critic_layers = list(self.worker_config.get("critic_layers", [64, 32]))

            self.gamma = float(self.worker_config.get("discount_factor", 0.99))
            self.gae_lambda = float(self.worker_config.get("gae_lambda", 0.95))
            self.learning_rate = float(self.worker_config.get("learning_rate", 0.001))
            self.entropy_coef = float(self.worker_config.get("entropy_coef", 0.01))
            self.value_coef = float(self.worker_config.get("value_coef", 0.5))
            self.max_grad_norm = float(self.worker_config.get("max_grad_norm", 0.5))
            self.update_epochs = int(self.worker_config.get("update_epochs", 1))
            self.normalize_advantages = bool(self.worker_config.get("normalize_advantages", True))
            self.advantage_epsilon = float(self.worker_config.get("advantage_epsilon", 1e-8))
            self.min_update_batch = int(self.worker_config.get("min_update_batch", 1))

            self.initial_exploration_rate = float(self.worker_config.get("exploration_rate", 0.1))
            self.imitation_usage_prob = float(self.worker_config.get("imitation_usage_prob", 0.3))
            self.auto_meta_update = bool(self.worker_config.get("auto_meta_update", False))
            self.meta_update_frequency = int(self.worker_config.get("meta_update_frequency", 25))
            self.performance_window = int(self.worker_config.get("performance_window", 100))
            self.success_reward_threshold = float(self.worker_config.get("success_reward_threshold", 0.0))

            self.reward_normalization = bool(self.worker_config.get("reward_normalization", True))
            self.reward_clip_range = tuple(self.worker_config.get("reward_clip_range", self.config.get("reward_clip_range", (-10.0, 10.0))))
            self.reward_scale = float(self.worker_config.get("reward_scale", self.config.get("reward_scale", 1.0)))
            self.reward_bias = float(self.worker_config.get("reward_bias", self.config.get("reward_bias", 0.0)))
            self.reward_momentum = float(self.worker_config.get("reward_momentum", self.config.get("reward_momentum", 0.99)))

            self.optimizer_name = str(self.worker_config.get("optimizer_name", "adamw")).lower()
            self.weight_decay = float(self.worker_config.get("weight_decay", 0.0))
            self.adam_beta1 = float(self.worker_config.get("adam_beta1", 0.9))
            self.adam_beta2 = float(self.worker_config.get("adam_beta2", 0.999))
            self.adam_epsilon = float(self.worker_config.get("adam_epsilon", 1e-8))
            self.sgd_momentum = float(self.worker_config.get("sgd_momentum", 0.9))
            self.rmsprop_alpha = float(self.worker_config.get("rmsprop_alpha", 0.99))

            self.scheduler_config = self.worker_config.get("scheduler", {}) or {}
            self.scheduler_name = str(self.scheduler_config.get("name", "none")).lower()

            self.device_preference = str(self.worker_config.get("device", "auto")).lower()
            self.checkpoint_protocol = int(self.worker_config.get("checkpoint_protocol", pickle.HIGHEST_PROTOCOL))
            self.continuous_action_std_init = float(self.worker_config.get("continuous_action_std_init", 0.5))
            self.configured_continuous_action = bool(
                self.worker_config.get("continuous_actions", self.actor_critic_config.get("continuous_action", False))
            )
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationValueError(
                "Failed to parse skill_worker configuration values.",
                component="skill_worker",
                details={"section": "skill_worker"},
                remediation="Ensure all skill_worker configuration values are valid scalars.",
                cause=exc,
            ) from exc

        ensure_positive(self.configured_goal_dim + 1, "goal_dim_plus_one", component="skill_worker")
        ensure_in_range(self.gamma, "discount_factor", minimum=0.0, maximum=1.0, component="skill_worker")
        ensure_in_range(self.gae_lambda, "gae_lambda", minimum=0.0, maximum=1.0, component="skill_worker")
        ensure_positive(self.learning_rate, "learning_rate", component="skill_worker")
        ensure_in_range(self.entropy_coef, "entropy_coef", minimum=0.0, component="skill_worker")
        ensure_in_range(self.value_coef, "value_coef", minimum=0.0, component="skill_worker")
        ensure_positive(self.max_grad_norm, "max_grad_norm", allow_zero=True, component="skill_worker")
        ensure_positive(self.update_epochs, "update_epochs", component="skill_worker")
        ensure_positive(self.min_update_batch, "min_update_batch", component="skill_worker")
        ensure_in_range(self.initial_exploration_rate, "exploration_rate", minimum=0.0, maximum=1.0, component="skill_worker")
        ensure_in_range(self.imitation_usage_prob, "imitation_usage_prob", minimum=0.0, maximum=1.0, component="skill_worker")
        ensure_positive(self.meta_update_frequency, "meta_update_frequency", component="skill_worker")
        ensure_positive(self.performance_window, "performance_window", component="skill_worker")
        ensure_positive(self.reward_scale, "reward_scale", allow_zero=True, component="skill_worker")
        ensure_in_range(self.reward_momentum, "reward_momentum", minimum=0.0, maximum=1.0, component="skill_worker")
        ensure_in_range(self.weight_decay, "weight_decay", minimum=0.0, component="skill_worker")
        ensure_positive(self.continuous_action_std_init, "continuous_action_std_init", component="skill_worker")

        ensure_non_empty(self.default_actor_layers, "actor_layers", component="skill_worker")
        ensure_non_empty(self.default_critic_layers, "critic_layers", component="skill_worker")

        if self.optimizer_name not in self.SUPPORTED_OPTIMIZERS:
            raise InvalidConfigurationValueError(
                f"Unsupported optimizer_name: {self.optimizer_name}",
                component="skill_worker",
                details={"supported": sorted(self.SUPPORTED_OPTIMIZERS)},
            )
        if self.scheduler_name not in self.SUPPORTED_SCHEDULERS:
            raise InvalidConfigurationValueError(
                f"Unsupported scheduler name: {self.scheduler_name}",
                component="skill_worker",
                details={"supported": sorted(self.SUPPORTED_SCHEDULERS)},
            )

        if not isinstance(self.reward_clip_range, (list, tuple)) or len(self.reward_clip_range) != 2:
            raise InvalidConfigurationValueError(
                "reward_clip_range must be a two-item list or tuple.",
                component="skill_worker",
                details={"reward_clip_range": self.reward_clip_range},
            )
        clip_low, clip_high = float(self.reward_clip_range[0]), float(self.reward_clip_range[1])
        if clip_low >= clip_high:
            raise InvalidConfigurationValueError(
                "reward_clip_range must satisfy low < high.",
                component="skill_worker",
                details={"reward_clip_range": self.reward_clip_range},
            )
        self.reward_clip_range = (clip_low, clip_high)

    def _resolve_device(self) -> torch.device:
        if self.device_preference == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device_preference == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self, skill_id: int, skill_metadata: Mapping[str, Any]) -> "SkillWorker":
        ensure_positive(int(skill_id), "skill_id", component="skill_worker")
        ensure_instance(skill_metadata, Mapping, "skill_metadata", component="skill_worker")

        if "state_dim" not in skill_metadata:
            raise MissingFieldError(
                "skill_metadata must define 'state_dim'.",
                component="skill_worker",
                details={"skill_metadata": dict(skill_metadata)},
            )
        if "action_dim" not in skill_metadata:
            raise MissingFieldError(
                "skill_metadata must define 'action_dim'.",
                component="skill_worker",
                details={"skill_metadata": dict(skill_metadata)},
            )

        self.skill_id = int(skill_id)
        self.skill_metadata = dict(skill_metadata)
        self.name = str(skill_metadata.get("name", f"skill_{self.skill_id}"))
        self.state_dim = int(skill_metadata["state_dim"])
        self.action_dim = int(skill_metadata["action_dim"])
        ensure_positive(self.state_dim, "state_dim", component="skill_worker")
        ensure_positive(self.action_dim, "action_dim", component="skill_worker")

        if "enable_goals" in skill_metadata:
            self.enable_goals = bool(skill_metadata["enable_goals"])
        if "goal_dim" in skill_metadata and self.enable_goals:
            self.goal_dim = int(skill_metadata["goal_dim"])
        elif not self.enable_goals:
            self.goal_dim = 0
        ensure_positive(self.goal_dim + 1, "goal_dim_plus_one", component="skill_worker")

        self.current_goal = np.zeros(self.goal_dim, dtype=np.float32) if self.enable_goals else None
        self.input_dim = int(self.state_dim + (self.goal_dim if self.enable_goals else 0))
        self.continuous_actions = bool(skill_metadata.get("continuous_actions", self.continuous_actions))
        self.device = self._resolve_device()

        actor_layers = self._adjust_output_layer(self.default_actor_layers, self.action_dim)
        critic_layers = self._adjust_output_layer(self.default_critic_layers, 1)

        try:
            acn_override = {
                "continuous_action": self.continuous_actions,
                "initial_std": self.continuous_action_std_init,
                "min_std": getattr(self, "min_std", 1e-4),
                "max_std": getattr(self, "max_std", 10.0),
            }
            self.actor_critic = ActorCriticNetwork(
                state_dim=self.input_dim,
                action_dim=self.action_dim,
                actor_layers=actor_layers,
                critic_layers=critic_layers,
                acn_config_override=acn_override,
            ).to(self.device)

            self.actor_optimizer = self._configure_optimizer(self.actor_critic.get_actor_parameters(), role="actor")
            self.critic_optimizer = self._configure_optimizer(self.actor_critic.get_critic_parameters(), role="critic")
            self.actor_scheduler = self._configure_scheduler(self.actor_optimizer, role="actor")
            self.critic_scheduler = self._configure_scheduler(self.critic_optimizer, role="critic")

        except Exception as exc:
            raise wrap_exception(
                exc,
                ReinforcementLearningError,
                "Failed to initialize ActorCriticNetwork or optimizers for SkillWorker.",
                component="skill_worker",
                details={
                    "skill_id": self.skill_id,
                    "input_dim": self.input_dim,
                    "action_dim": self.action_dim,
                },
            ) from exc

        self.actor_optimizer = self._configure_optimizer(self.actor_critic.get_actor_parameters(), role="actor")
        self.critic_optimizer = self._configure_optimizer(self.actor_critic.get_critic_parameters(), role="critic")
        self.actor_scheduler = self._configure_scheduler(self.actor_optimizer, role="actor")
        self.critic_scheduler = self._configure_scheduler(self.critic_optimizer, role="critic")

        logger.info(
            "Skill Worker '%s' initialized | skill_id=%s state_dim=%s action_dim=%s input_dim=%s goals=%s continuous=%s",
            self.name,
            self.skill_id,
            self.state_dim,
            self.action_dim,
            self.input_dim,
            self.enable_goals,
            self.continuous_actions,
        )
        return self
    
    def _ensure_initialized(self) -> None:
        if self.actor_critic is None:
            raise AdaptiveLearningError(
                "SkillWorker has not been initialized. Call initialize() or use create_worker().",
                component="skill_worker",
            )
        if self.actor_optimizer is None or self.critic_optimizer is None:
            raise AdaptiveLearningError(
                "SkillWorker optimizers are not initialized.",
                component="skill_worker",
            )
    def _adjust_output_layer(self, layers: Sequence[Union[int, Mapping[str, Any]]], output_dim: int) -> List[Union[int, Dict[str, Any]]]:
        if not layers:
            raise InvalidConfigurationValueError(
                "Layer configuration cannot be empty when adjusting output dimension.",
                component="skill_worker",
            )
        ensure_non_empty(list(layers), "layers", component="skill_worker")
        ensure_positive(output_dim, "output_dim", component="skill_worker")

        adjusted = list(layers)
        last = adjusted[-1]
        if isinstance(last, Mapping):
            copied = dict(last)
            copied["neurons"] = int(output_dim)
            adjusted[-1] = copied
        else:
            adjusted[-1] = int(output_dim)
        return adjusted

    def _configure_optimizer(self, parameters: Iterable[torch.nn.Parameter], *, role: str) -> torch.optim.Optimizer:
        params = list(parameters)
        ensure_non_empty(params, f"{role}_parameters", component="skill_worker")

        if self.optimizer_name == "adam":
            return torch.optim.Adam(
                params,
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=self.sgd_momentum,
                weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "rmsprop":
            return torch.optim.RMSprop(
                params,
                lr=self.learning_rate,
                alpha=self.rmsprop_alpha,
                momentum=self.sgd_momentum,
                weight_decay=self.weight_decay,
            )
        raise InvalidConfigurationValueError(
            f"Unsupported optimizer_name: {self.optimizer_name}",
            component="skill_worker",
        )

    def _configure_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        role: str,
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if self.scheduler_name == "none":
            return None

        if self.scheduler_name == "step":
            step_size = int(self.scheduler_config.get("step_size", 10))
            gamma = float(self.scheduler_config.get("gamma", 0.5))
            ensure_positive(step_size, f"{role}_scheduler.step_size", component="skill_worker")
            ensure_in_range(gamma, f"{role}_scheduler.gamma", minimum=0.0, maximum=1.0, component="skill_worker")
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        if self.scheduler_name == "cosine":
            t_max = int(self.scheduler_config.get("t_max", 20))
            eta_min = float(self.scheduler_config.get("min_lr", 0.0))
            ensure_positive(t_max, f"{role}_scheduler.t_max", component="skill_worker")
            ensure_in_range(eta_min, f"{role}_scheduler.min_lr", minimum=0.0, component="skill_worker")
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

        if self.scheduler_name == "reduce_on_plateau":
            factor = float(self.scheduler_config.get("factor", 0.5))
            patience = int(self.scheduler_config.get("patience", 5))
            min_lr = float(self.scheduler_config.get("min_lr", 1e-6))
            ensure_in_range(factor, f"{role}_scheduler.factor", minimum=0.0, maximum=1.0, component="skill_worker")
            ensure_positive(patience, f"{role}_scheduler.patience", component="skill_worker")
            ensure_in_range(min_lr, f"{role}_scheduler.min_lr", minimum=0.0, component="skill_worker")
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=str(self.scheduler_config.get("mode", "min")).lower(),
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )

        raise InvalidConfigurationValueError(
            f"Unsupported scheduler name: {self.scheduler_name}",
            component="skill_worker",
            details={"role": role},
        )

    def attach_meta_learning(self, meta_worker: MetaLearningWorker) -> None:
        ensure_instance(meta_worker, MetaLearningWorker, "meta_worker", component="skill_worker")
        self.meta_learning = meta_worker
        if hasattr(meta_worker, "register_skill_worker") and self.skill_id is not None:
            meta_worker.register_skill_worker(self.skill_id, self)
        logger.info("Meta Learning Worker attached to SkillWorker '%s'", self.name)

    def attach_imitation_learning(self, imitation_worker: ImitationLearningWorker) -> None:
        ensure_instance(imitation_worker, ImitationLearningWorker, "imitation_worker", component="skill_worker")
        self.imitation_learning = imitation_worker
        logger.info("Imitation Learning Worker attached to SkillWorker '%s'", self.name)

    def apply_hyperparameters(self, hyperparams: Mapping[str, Any]) -> Dict[str, Any]:
        ensure_instance(hyperparams, Mapping, "hyperparams", component="skill_worker")
        applied: Dict[str, Any] = {}

        if "learning_rate" in hyperparams:
            self.learning_rate = float(hyperparams["learning_rate"])
            # FIX: guard against None optimizers
            for optimizer in (self.actor_optimizer, self.critic_optimizer):
                if optimizer is not None:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
            applied["learning_rate"] = self.learning_rate

        if "exploration_rate" in hyperparams:
            self.exploration_rate = float(hyperparams["exploration_rate"])
            applied["exploration_rate"] = self.exploration_rate

        if "entropy_coef" in hyperparams:
            self.entropy_coef = float(hyperparams["entropy_coef"])
            applied["entropy_coef"] = self.entropy_coef

        if "discount_factor" in hyperparams:
            self.gamma = float(hyperparams["discount_factor"])
            applied["discount_factor"] = self.gamma

        return applied

    def set_goal(self, goal: np.ndarray) -> None:
        if not self.enable_goals:
            return

        goal_array = self._sanitize_goal(goal)
        self.current_goal = goal_array
        logger.debug("Worker %s set goal: %s", self.skill_id, goal_array[: min(4, len(goal_array))])

    def clear_goal(self) -> None:
        if self.enable_goals:
            self.current_goal = np.zeros(self.goal_dim, dtype=np.float32)

    def _sanitize_goal(self, goal: Any) -> np.ndarray:
        if not self.enable_goals:
            return np.zeros(0, dtype=np.float32)

        if goal is None:
            raise InvalidValueError(
                "goal cannot be None when goal conditioning is enabled.",
                component="skill_worker",
            )

        try:
            arr = np.asarray(goal, dtype=np.float32).reshape(-1)
        except Exception as exc:
            raise wrap_exception(
                exc,
                InvalidTypeError,
                "Failed to coerce goal into a float32 vector.",
                component="skill_worker",
                details={"goal_type": type(goal).__name__},
            ) from exc

        if arr.size != self.goal_dim:
            raise InvalidValueError(
                "Goal dimension mismatch.",
                component="skill_worker",
                details={"expected": self.goal_dim, "received": int(arr.size)},
                remediation="Provide a goal vector with the configured goal_dim.",
            )
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def _sanitize_state(self, value: Any, *, label: str = "state") -> np.ndarray:
        if self.state_dim is None:
            raise AdaptiveLearningError(
                "SkillWorker must be initialized before processing states.",
                component="skill_worker",
            )

        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
        elif value is None:
            raise InvalidValueError(
                f"{label} cannot be None.",
                component="skill_worker",
                details={"label": label},
            )
        else:
            try:
                arr = np.asarray(value, dtype=np.float32)
            except Exception as exc:
                raise wrap_exception(
                    exc,
                    InvalidTypeError,
                    f"Failed to convert {label} into a float32 array.",
                    component="skill_worker",
                    details={"label": label, "type": type(value).__name__},
                ) from exc

        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.size != self.state_dim:
            raise InvalidValueError(
                f"{label} dimension mismatch.",
                component="skill_worker",
                details={"label": label, "expected": self.state_dim, "received": int(arr.size)},
            )
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return arr

    def _process_state(self, state: Any) -> np.ndarray:
        state_arr = self._sanitize_state(state, label="state")
        if self.enable_goals:
            goal = self.current_goal if self.current_goal is not None else np.zeros(self.goal_dim, dtype=np.float32)
            return np.concatenate([state_arr, goal.astype(np.float32)], axis=0)
        return state_arr

    def _state_tensor(self, state: Any) -> torch.Tensor:
        processed = self._process_state(state)
        return torch.as_tensor(processed, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _select_action_from_policy(self, state_tensor: torch.Tensor, *, explore: bool) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_initialized()
        ensure_not_none(self.actor_critic, "actor_critic", component="skill_worker")

        actor_output = self.actor_critic.forward_actor(state_tensor)
        value = self.actor_critic.forward_critic(state_tensor).view(-1)[0]

        if self.continuous_actions:
            if hasattr(self.actor_critic, "action_std") and self.actor_critic.action_std is not None:
                std = self.actor_critic.action_std.expand_as(actor_output)
            else:
                std = torch.ones_like(actor_output) * self.continuous_action_std_init
            dist = torch.distributions.Normal(actor_output, std)
            action = dist.rsample() if explore else actor_output
            log_prob = dist.log_prob(action).sum(dim=-1)[0]
            entropy = dist.entropy().sum(dim=-1)[0]
            return action.squeeze(0).detach().cpu().numpy(), log_prob, entropy, value

        dist = torch.distributions.Categorical(logits=actor_output)
        action = dist.sample() if explore else torch.argmax(actor_output, dim=-1)
        log_prob = dist.log_prob(action).view(-1)[0]
        entropy = dist.entropy().view(-1)[0]
        return int(action.view(-1)[0].item()), log_prob, entropy, value

    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[Any, float, float]:
        self._ensure_initialized()
        ensure_not_none(self.actor_critic, "actor_critic", component="skill_worker")

        if self.imitation_learning and explore and random.random() < self.imitation_usage_prob:
            try:
                imitation_state = self._process_state(state)
                imitation_action = self.imitation_learning.get_action(imitation_state)
                self.last_action_info = {
                    "source": "imitation",
                    "explore": bool(explore),
                    "log_prob": 0.0,
                    "entropy": 0.0,
                }
                return imitation_action, 0.0, 0.0
            except AdaptiveError:
                raise
            except Exception as exc:
                logger.warning("Imitation action selection failed; falling back to RL: %s", exc)

        try:
            state_tensor = self._state_tensor(state)
            action, log_prob, entropy, value = self._select_action_from_policy(state_tensor, explore=bool(explore))
            self.last_action_info = {
                "source": "rl",
                "explore": bool(explore),
                "log_prob": float(log_prob.item()),
                "entropy": float(entropy.item()),
                "value": float(value.item()),
            }
            return action, float(log_prob.item()), float(entropy.item())
        except AdaptiveError:
            raise
        except Exception as exc:
            raise wrap_exception(
                exc,
                ReinforcementLearningError,
                "Action selection failed.",
                component="skill_worker",
                details={"skill_id": self.skill_id, "name": self.name},
            ) from exc

    def _normalize_reward(self, reward: float) -> float:
        if not np.isfinite(reward):
            raise RewardNormalizationError(
                "Reward must be finite.",
                component="skill_worker",
                details={"reward": reward},
            )

        adjusted = (float(reward) * self.reward_scale) + self.reward_bias
        if not self.reward_normalization:
            return float(np.clip(adjusted, *self.reward_clip_range))

        try:
            self.reward_count += 1.0
            delta = adjusted - self.reward_mean
            self.reward_mean += (1.0 - self.reward_momentum) * delta
            delta2 = adjusted - self.reward_mean
            self.reward_var = self.reward_momentum * self.reward_var + (1.0 - self.reward_momentum) * (delta * delta2)
            self.reward_var = float(max(self.reward_var, 1e-8))
            self.reward_std = float(np.sqrt(self.reward_var))
            normalized = (adjusted - self.reward_mean) / (self.reward_std + 1e-8)
            return float(np.clip(normalized, *self.reward_clip_range))
        except Exception as exc:
            raise RewardNormalizationError(
                "Reward normalization failed.",
                component="skill_worker",
                details={"reward": reward},
                cause=exc,
            ) from exc

    def store_experience(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float = 0.0,
        entropy: float = 0.0,
    ) -> Transition:
        self._ensure_initialized() 
        ensure_not_none(self.skill_id, "skill_id", component="skill_worker")

        normalized_reward = self._normalize_reward(float(reward))
        state_arr = self._sanitize_state(state, label="state")
        next_state_arr = self._sanitize_state(next_state, label="next_state")

        if self.continuous_actions:
            action_tensor = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
            action_value: Any = action_tensor.detach().cpu().numpy()
        else:
            if torch.is_tensor(action):
                action_value = int(action.detach().cpu().view(-1)[0].item())
            else:
                try:
                    action_value = int(action)
                except Exception as exc:
                    raise wrap_exception(
                        exc,
                        InvalidValueError,
                        "Discrete action must be convertible to int.",
                        component="skill_worker",
                        details={"action": action},
                    ) from exc
            if action_value < 0 or action_value >= self.action_dim:
                raise RangeValidationError(
                    "Discrete action is outside the valid action range.",
                    component="skill_worker",
                    details={"action": action_value, "action_dim": self.action_dim},
                )
            action_tensor = torch.tensor(action_value, dtype=torch.long)

        transition = Transition(
            state=torch.as_tensor(state_arr, dtype=torch.float32),
            action=action_tensor,
            reward=float(normalized_reward),
            next_state=torch.as_tensor(next_state_arr, dtype=torch.float32),
            done=bool(done),
            log_prob=torch.tensor(float(log_prob), dtype=torch.float32),
            entropy=torch.tensor(float(entropy), dtype=torch.float32),
            source="rl",
        )

        try:
            self.local_memory.store_experience(
                state=state_arr,
                action=action_value if not self.continuous_actions else action_value.tolist(),
                reward=float(normalized_reward),
                next_state=next_state_arr,
                done=bool(done),
                context={"source": "skill_worker", "skill_id": self.skill_id, "skill_name": self.name},
                params={
                    "learning_rate": self.learning_rate,
                    "exploration_rate": self.exploration_rate,
                    "discount_factor": self.gamma,
                },
                log_prob=float(log_prob),
                entropy=float(entropy),
            )
            self.learner_memory.add(transition, tag=f"skill_{self.skill_id}")
            self.step_counter += 1
            return transition
        except AdaptiveError:
            raise
        except Exception as exc:
            raise wrap_exception(
                exc,
                ExperienceValidationError,
                "Failed to store skill-worker experience.",
                component="skill_worker",
                details={"skill_id": self.skill_id},
            ) from exc

    def compute_returns(self, rewards: Sequence[float], dones: Sequence[bool]) -> List[float]:
        ensure_instance(rewards, (list, tuple), "rewards", component="skill_worker")
        ensure_instance(dones, (list, tuple), "dones", component="skill_worker")
        if len(rewards) != len(dones):
            raise ReturnComputationError(
                "rewards and dones must have the same length.",
                component="skill_worker",
                details={"rewards_len": len(rewards), "dones_len": len(dones)},
            )
        if len(rewards) == 0:
            raise EmptyCollectionError(
                "Cannot compute returns for an empty trajectory.",
                component="skill_worker",
            )

        returns: List[float] = []
        running_return = 0.0
        for reward, done in zip(reversed(list(rewards)), reversed(list(dones))):
            running_return = float(reward) + self.gamma * running_return * (1.0 - float(bool(done)))
            returns.insert(0, running_return)
        return returns

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            advantages = torch.zeros_like(rewards, dtype=torch.float32, device=self.device)
            gae = torch.zeros((), dtype=torch.float32, device=self.device)
            for step in reversed(range(rewards.shape[0])):
                mask = 1.0 - dones[step]
                delta = rewards[step] + self.gamma * next_values[step] * mask - values[step]
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                advantages[step] = gae
            returns = advantages + values
            return advantages, returns
        except Exception as exc:
            raise ReturnComputationError(
                "Failed to compute generalized advantages.",
                component="skill_worker",
                cause=exc,
            ) from exc

    def _prepare_update_batch(self) -> Dict[str, torch.Tensor]:
        self._ensure_initialized()
        experiences = list(self.local_memory.episodic)
        if len(experiences) < self.min_update_batch:
            raise EmptyCollectionError(
                "Not enough on-policy experiences to update the skill worker.",
                component="skill_worker",
                details={"available": len(experiences), "required": self.min_update_batch},
            )

        states: List[np.ndarray] = []
        next_states: List[np.ndarray] = []
        rewards: List[float] = []
        dones: List[float] = []
        actions: List[Any] = []
        dropped_experiences = 0

        for exp in experiences:
            try:
                state_arr = self._sanitize_state(exp.get("state"), label="state")
                next_state_arr = self._sanitize_state(exp.get("next_state"), label="next_state")
                reward_value = float(np.asarray(exp.get("reward"), dtype=np.float32).reshape(-1)[0])
                done_value = float(bool(exp.get("done", False)))

                if self.continuous_actions:
                    action_value = np.asarray(exp.get("action"), dtype=np.float32).reshape(-1)
                else:
                    action_value = int(np.asarray(exp.get("action")).reshape(-1)[0])
                    if action_value < 0 or action_value >= self.action_dim:
                        raise RangeValidationError(
                            "Discrete action is outside the valid action range.",
                            component="skill_worker",
                            details={"action": action_value, "action_dim": self.action_dim},
                        )

                states.append(state_arr)
                next_states.append(next_state_arr)
                rewards.append(reward_value)
                dones.append(done_value)
                actions.append(action_value)
            except Exception:
                dropped_experiences += 1

        if len(states) < self.min_update_batch:
            raise EmptyCollectionError(
                "Not enough valid on-policy experiences to update the skill worker.",
                component="skill_worker",
                details={
                    "available": len(experiences),
                    "valid": len(states),
                    "dropped": dropped_experiences,
                    "required": self.min_update_batch,
                },
            )

        rewards_arr = np.asarray(rewards, dtype=np.float32)
        dones_arr = np.asarray(dones, dtype=np.float32)
        if self.continuous_actions:
            actions_arr = np.asarray(actions, dtype=np.float32)
        else:
            actions_arr = np.asarray(actions, dtype=np.int64)

        processed_states = np.vstack([self._process_state(state) for state in states]).astype(np.float32)
        processed_next_states = np.vstack([self._process_state(state) for state in next_states]).astype(np.float32)

        states_tensor = torch.as_tensor(processed_states, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.as_tensor(processed_next_states, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.as_tensor(rewards_arr, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones_arr, dtype=torch.float32, device=self.device)
        if self.continuous_actions:
            actions_tensor = torch.as_tensor(actions_arr, dtype=torch.float32, device=self.device)
        else:
            actions_tensor = torch.as_tensor(actions_arr, dtype=torch.long, device=self.device)

        values = self.actor_critic.forward_critic(states_tensor).view(-1)
        with torch.no_grad():
            next_values = self.actor_critic.forward_critic(next_states_tensor).view(-1)

        advantages, returns = self._compute_gae(rewards_tensor, dones_tensor, values.detach(), next_values.detach())
        if self.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + self.advantage_epsilon)

        return {
            "states": states_tensor,
            "actions": actions_tensor,
            "rewards": rewards_tensor,
            "dones": dones_tensor,
            "values": values,
            "next_values": next_values,
            "advantages": advantages,
            "returns": returns.detach(),
        }

    def update_policy(self) -> Optional[float]:
        self._ensure_initialized()
        if len(self.local_memory.episodic) == 0:
            return None

        batch = self._prepare_update_batch()
        states = batch["states"]
        actions = batch["actions"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_bonus = 0.0

        for _ in range(self.update_epochs):
            actor_output = self.actor_critic.forward_actor(states)
            if self.continuous_actions:
                if hasattr(self.actor_critic, "action_std") and self.actor_critic.action_std is not None:
                    std = self.actor_critic.action_std.expand_as(actor_output)
                else:
                    std = torch.ones_like(actor_output) * self.continuous_action_std_init
                dist = torch.distributions.Normal(actor_output, std)
                log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
            else:
                dist = torch.distributions.Categorical(logits=actor_output)
                log_probs = dist.log_prob(actions.view(-1))
                entropy = dist.entropy().mean()

            values = self.actor_critic.forward_critic(states).view(-1)

            actor_loss = -(log_probs * advantages.detach()).mean() - (self.entropy_coef * entropy)
            critic_loss = self.value_coef * F.mse_loss(values, returns)

            # FIX: get parameters safely
            actor_params = list(self.actor_critic.get_actor_parameters())
            critic_params = list(self.actor_critic.get_critic_parameters())

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm)
            self.critic_optimizer.step()

            total_actor_loss += float(actor_loss.item())
            total_critic_loss += float(critic_loss.item())
            total_entropy_bonus += float(entropy.item())

        mean_actor_loss = total_actor_loss / self.update_epochs
        mean_critic_loss = total_critic_loss / self.update_epochs
        mean_entropy = total_entropy_bonus / self.update_epochs
        total_loss = mean_actor_loss + mean_critic_loss

        imitation_mixed = False
        imitation_loss = None
        if self.imitation_learning is not None and self.learner_memory.size() >= max(100, self.min_update_batch):
            try:
                rl_loss = torch.tensor(total_loss, dtype=torch.float32, device=self.device, requires_grad=True)
                imitation_loss = self.imitation_learning.mixed_objective_update(
                    states=states.detach(),
                    actions=actions.detach(),
                    advantages=advantages.detach(),
                    rl_loss=rl_loss,
                )
                imitation_mixed = True
            except AdaptiveError:
                raise
            except Exception as exc:
                logger.warning("Mixed imitation update failed for SkillWorker '%s': %s", self.name, exc)

        self.update_counter += 1
        result = PolicyUpdateResult(
            total_loss=float(total_loss),
            actor_loss=float(mean_actor_loss),
            critic_loss=float(mean_critic_loss),
            entropy_bonus=float(mean_entropy),
            mean_return=float(returns.mean().item()),
            mean_advantage=float(advantages.mean().item()),
            num_samples=int(states.shape[0]),
            update_steps=int(self.update_epochs),
            imitation_mixed=imitation_mixed,
            imitation_loss=float(imitation_loss) if imitation_loss is not None else None,
        )
        self.last_update = result

        self.training_history["actor_loss"].append(result.actor_loss)
        self.training_history["critic_loss"].append(result.critic_loss)
        self.training_history["total_loss"].append(result.total_loss)
        self.training_history["entropy_bonus"].append(result.entropy_bonus)
        self.training_history["mean_return"].append(result.mean_return)
        self.training_history["mean_advantage"].append(result.mean_advantage)
        self.training_history["actor_lr"].append(float(self.actor_optimizer.param_groups[0]["lr"]))
        self.training_history["critic_lr"].append(float(self.critic_optimizer.param_groups[0]["lr"]))

        self._step_scheduler(self.actor_scheduler, result.total_loss)
        self._step_scheduler(self.critic_scheduler, result.total_loss)

        self.local_memory.clear_episodic()

        if self.auto_meta_update and self.meta_learning is not None and self.update_counter % self.meta_update_frequency == 0:
            try:
                self.meta_learning.optimization_step()
            except AdaptiveError:
                raise
            except Exception as exc:
                logger.warning("Auto meta-learning update failed for SkillWorker '%s': %s", self.name, exc)

        logger.info(
            "Skill %s updated | total=%.4f actor=%.4f critic=%.4f entropy=%.4f samples=%s",
            self.skill_id,
            result.total_loss,
            result.actor_loss,
            result.critic_loss,
            result.entropy_bonus,
            result.num_samples,
        )
        return result.total_loss

    def _step_scheduler(self, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], loss: float) -> None:
        if scheduler is None:
            return
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss)
        else:
            scheduler.step()

    def record_episode_outcome(self, episode_return: float, episode_length: int, success: Optional[bool] = None) -> None:
        self.episode_counter += 1
        self.recent_episode_returns.append(float(episode_return))
        self.recent_episode_lengths.append(int(episode_length))
        if success is None:
            success = float(episode_return) > self.success_reward_threshold
        self.success_history.append(int(bool(success)))

    def _iter_valid_learner_transitions(self) -> List[Transition]:
        tag = f"skill_{self.skill_id}" if self.skill_id is not None else None
        candidates: List[Any] = []

        try:
            if tag and hasattr(self.learner_memory, "get_by_tag"):
                candidates = list(self.learner_memory.get_by_tag(tag))
            else:
                raw = self.learner_memory.get()
                # ensure raw is iterable
                if raw is None:
                    candidates = []
                elif isinstance(raw, (list, tuple)):
                    candidates = list(raw)
                else:
                    candidates = [raw]
        except Exception as exc:
            logger.warning("Failed to retrieve learner-memory transitions for skill %s: %s", self.skill_id, exc)
            candidates = []

        transitions: List[Transition] = []
        for item in candidates:
            if isinstance(item, Transition):
                transitions.append(item)
                continue
            if hasattr(item, "reward") and hasattr(item, "state") and hasattr(item, "action"):
                try:
                    transitions.append(
                        Transition(
                            state=torch.as_tensor(item.state, dtype=torch.float32),
                            action=torch.as_tensor(item.action),
                            reward=float(item.reward),
                            next_state=torch.as_tensor(item.next_state, dtype=torch.float32),
                            done=bool(item.done),
                            log_prob=getattr(item, "log_prob", None),
                            entropy=getattr(item, "entropy", None),
                            value=getattr(item, "value", None),
                            next_value=getattr(item, "next_value", None),
                            source=getattr(item, "source", "rl"),
                        )
                    )
                except Exception:
                    logger.debug("Skipping malformed learner-memory transition-like item for skill %s", self.skill_id)
                continue
            if isinstance(item, Mapping) and {"state", "action", "reward", "next_state", "done"}.issubset(item.keys()):
                try:
                    transitions.append(
                        Transition(
                            state=torch.as_tensor(item["state"], dtype=torch.float32),
                            action=torch.as_tensor(item["action"]),
                            reward=float(item["reward"]),
                            next_state=torch.as_tensor(item["next_state"], dtype=torch.float32),
                            done=bool(item["done"]),
                            log_prob=torch.as_tensor(item["log_prob"], dtype=torch.float32) if item.get("log_prob") is not None else None,
                            entropy=torch.as_tensor(item["entropy"], dtype=torch.float32) if item.get("entropy") is not None else None,
                            value=torch.as_tensor(item["value"], dtype=torch.float32) if item.get("value") is not None else None,
                            next_value=torch.as_tensor(item["next_value"], dtype=torch.float32) if item.get("next_value") is not None else None,
                            source=str(item.get("source", "rl")),
                        )
                    )
                except Exception:
                    logger.debug("Skipping malformed learner-memory mapping item for skill %s", self.skill_id)
                continue
            logger.debug("Ignoring non-transition item from learner memory for skill %s: %s", self.skill_id, type(item).__name__)
        return transitions

    def get_performance_metrics(self) -> Dict[str, Any]:
        self._ensure_initialized()
        transitions = self._iter_valid_learner_transitions()
        if not transitions:
            base = {
                "skill_id": self.skill_id,
                "name": self.name,
                "episode_count": 0,
                "avg_reward": 0.0,
                "success_rate": 0.0,
                "recent_reward": 0.0,
                "memory_size": int(self.learner_memory.size()),
                "local_episodic_size": int(len(self.local_memory.episodic)),
                "learning_rate": float(self.learning_rate),
                "exploration_rate": float(self.exploration_rate),
                "discount_factor": float(self.gamma),
                "entropy_coef": float(self.entropy_coef),
            }
            if self.last_update is not None:
                base["last_total_loss"] = self.last_update.total_loss
            return base

        rewards = [float(t.reward) for t in transitions if hasattr(t, "reward")]
        successes = [1 if reward > self.success_reward_threshold else 0 for reward in rewards]
        recent_rewards = rewards[-10:]

        metrics = {
            "skill_id": self.skill_id,
            "name": self.name,
            "episode_count": len(transitions),
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "recent_reward": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
            "memory_size": int(self.learner_memory.size()),
            "local_episodic_size": int(len(self.local_memory.episodic)),
            "learning_rate": float(self.learning_rate),
            "exploration_rate": float(self.exploration_rate),
            "discount_factor": float(self.gamma),
            "entropy_coef": float(self.entropy_coef),
        }
        if self.last_update is not None:
            metrics.update({
                "last_total_loss": self.last_update.total_loss,
                "last_actor_loss": self.last_update.actor_loss,
                "last_critic_loss": self.last_update.critic_loss,
                "last_entropy_bonus": self.last_update.entropy_bonus,
            })
        if self.recent_episode_returns:
            metrics["recent_episode_return"] = float(np.mean(list(self.recent_episode_returns)))
        if self.recent_episode_lengths:
            metrics["recent_episode_length"] = float(np.mean(list(self.recent_episode_lengths)))
        return metrics

    def get_training_report(self) -> Dict[str, Any]:
        registry_snapshot = SkillWorker.registry_snapshot()   # class method
        registry_entry = registry_snapshot.get(self.skill_id, {}) if self.skill_id is not None else {}
    
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "registry_snapshot": registry_entry,
            "performance_metrics": self.get_performance_metrics(),
            "last_action_info": self.last_action_info,
            "last_update": self.last_update.to_dict() if self.last_update is not None else None,
            "training_history_lengths": {key: len(values) for key, values in self.training_history.items()},
            "goal_enabled": self.enable_goals,
            "current_goal": self.current_goal.tolist() if isinstance(self.current_goal, np.ndarray) else None,
            "meta_learning_attached": self.meta_learning is not None,
            "imitation_learning_attached": self.imitation_learning is not None,
        }

    def export_state(self) -> Dict[str, Any]:
        self._ensure_initialized()
        ensure_not_none(self.actor_critic, "actor_critic", component="skill_worker")

        local_memory_state = self.local_memory.export_state() if hasattr(self.local_memory, "export_state") else None
        state = {
            "format_version": "1.0.0",
            "skill_id": self.skill_id,
            "name": self.name,
            "skill_metadata": dict(self.skill_metadata),
            "config": {
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "learning_rate": self.learning_rate,
                "entropy_coef": self.entropy_coef,
                "value_coef": self.value_coef,
                "max_grad_norm": self.max_grad_norm,
                "update_epochs": self.update_epochs,
                "normalize_advantages": self.normalize_advantages,
                "exploration_rate": self.exploration_rate,
                "reward_normalization": self.reward_normalization,
                "reward_clip_range": list(self.reward_clip_range),
                "reward_scale": self.reward_scale,
                "reward_bias": self.reward_bias,
                "reward_momentum": self.reward_momentum,
                "continuous_actions": self.continuous_actions,
                "enable_goals": self.enable_goals,
                "goal_dim": self.goal_dim,
            },
            "model_state": {
                "actor_critic": self.actor_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict() if self.actor_optimizer is not None else None,
                "critic_optimizer": self.critic_optimizer.state_dict() if self.critic_optimizer is not None else None,
                "actor_scheduler": self.actor_scheduler.state_dict() if self.actor_scheduler is not None else None,
                "critic_scheduler": self.critic_scheduler.state_dict() if self.critic_scheduler is not None else None,
            },
            "reward_stats": {
                "mean": self.reward_mean,
                "var": self.reward_var,
                "std": self.reward_std,
                "count": self.reward_count,
            },
            "runtime": {
                "update_counter": self.update_counter,
                "episode_counter": self.episode_counter,
                "step_counter": self.step_counter,
                "recent_episode_returns": list(self.recent_episode_returns),
                "recent_episode_lengths": list(self.recent_episode_lengths),
                "success_history": list(self.success_history),
                "training_history": self.training_history,
                "last_action_info": self.last_action_info,
                "last_update": self.last_update.to_dict() if self.last_update is not None else None,
            },
            "goal_state": self.current_goal.tolist() if isinstance(self.current_goal, np.ndarray) else None,
            "local_memory_state": local_memory_state,
        }
        return state

    def import_state(self, payload: Mapping[str, Any]) -> None:
        self._ensure_initialized()
        ensure_instance(payload, Mapping, "payload", component="skill_worker")
        ensure_not_none(self.skill_id, "skill_id", component="skill_worker")
        ensure_not_none(self.actor_critic, "actor_critic", component="skill_worker")

        model_state = payload.get("model_state", {})
        if not isinstance(model_state, Mapping):
            raise InvalidValueError(
                "Checkpoint payload is missing a valid model_state.",
                component="skill_worker",
            )

        actor_critic_state = model_state.get("actor_critic")
        if actor_critic_state is None:
            raise MissingFieldError(
                "Checkpoint payload is missing model_state.actor_critic.",
                component="skill_worker",
            )
        self.actor_critic.load_state_dict(actor_critic_state)

        if self.actor_optimizer is not None and model_state.get("actor_optimizer") is not None:
            self.actor_optimizer.load_state_dict(model_state["actor_optimizer"])
        if self.critic_optimizer is not None and model_state.get("critic_optimizer") is not None:
            self.critic_optimizer.load_state_dict(model_state["critic_optimizer"])
        if self.actor_scheduler is not None and model_state.get("actor_scheduler") is not None:
            self.actor_scheduler.load_state_dict(model_state["actor_scheduler"])
        if self.critic_scheduler is not None and model_state.get("critic_scheduler") is not None:
            self.critic_scheduler.load_state_dict(model_state["critic_scheduler"])

        reward_stats = payload.get("reward_stats", {})
        if isinstance(reward_stats, Mapping):
            self.reward_mean = float(reward_stats.get("mean", self.reward_mean))
            self.reward_var = float(reward_stats.get("var", self.reward_var))
            self.reward_std = float(reward_stats.get("std", self.reward_std))
            self.reward_count = float(reward_stats.get("count", self.reward_count))

        runtime = payload.get("runtime", {})
        if isinstance(runtime, Mapping):
            self.update_counter = int(runtime.get("update_counter", self.update_counter))
            self.episode_counter = int(runtime.get("episode_counter", self.episode_counter))
            self.step_counter = int(runtime.get("step_counter", self.step_counter))
            self.recent_episode_returns = deque(runtime.get("recent_episode_returns", []), maxlen=self.performance_window)
            self.recent_episode_lengths = deque(runtime.get("recent_episode_lengths", []), maxlen=self.performance_window)
            self.success_history = deque(runtime.get("success_history", []), maxlen=self.performance_window)
            history = runtime.get("training_history", {})
            if isinstance(history, Mapping):
                self.training_history = {str(k): list(v) for k, v in history.items()}
            self.last_action_info = runtime.get("last_action_info", self.last_action_info)
            last_update = runtime.get("last_update")
            if isinstance(last_update, Mapping):
                self.last_update = PolicyUpdateResult(**last_update)

        goal_state = payload.get("goal_state")
        if goal_state is not None and self.enable_goals:
            self.current_goal = self._sanitize_goal(goal_state)

        local_memory_state = payload.get("local_memory_state")
        if local_memory_state is not None and hasattr(self.local_memory, "import_state"):
            self.local_memory.import_state(local_memory_state)

    def save_checkpoint(self, path: str) -> Path:
        self._ensure_initialized()
        output_path = Path(path)
        checkpoint = self.export_state()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("wb") as handle:
                pickle.dump(checkpoint, handle, protocol=self.checkpoint_protocol)
            logger.info("Skill %s checkpoint saved to %s", self.skill_id, output_path)
            return output_path
        except Exception as exc:
            raise CheckpointSaveError(
                f"Failed to save SkillWorker checkpoint to {output_path}.",
                component="skill_worker",
                details={"path": str(output_path)},
                cause=exc,
            ) from exc

    def load_checkpoint(self, path: str) -> None:
        self._ensure_initialized()
        input_path = Path(path)
        if not input_path.exists():
            raise CheckpointNotFoundError(
                f"SkillWorker checkpoint not found: {input_path}",
                component="skill_worker",
                details={"path": str(input_path)},
            )

        try:
            with input_path.open("rb") as handle:
                checkpoint = pickle.load(handle)
            self.import_state(checkpoint)
            logger.info("Skill %s checkpoint loaded from %s", self.skill_id, input_path)
        except AdaptiveError:
            raise
        except Exception as exc:
            raise CheckpointLoadError(
                f"Failed to load SkillWorker checkpoint from {input_path}.",
                component="skill_worker",
                details={"path": str(input_path)},
                cause=exc,
            ) from exc


if __name__ == "__main__":
    print("\n=== Running Skill Worker ===\n")
    printer.status("TEST", "Skill Worker initialized", "info")

    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)

    skill_metadata = {
        "name": "navigate_to_A",
        "state_dim": 10,
        "action_dim": 2,
        "enable_goals": False,
    }

    worker = SkillWorker.create_worker(skill_id=1, skill_metadata=skill_metadata)

    state = np.random.randn(skill_metadata["state_dim"]).astype(np.float32)
    next_state = np.random.randn(skill_metadata["state_dim"]).astype(np.float32)

    action, log_prob, entropy = worker.select_action(state, explore=True)
    printer.pretty("Action", {"action": action, "log_prob": log_prob, "entropy": entropy}, "success")

    for step in range(32):
        done = step % 8 == 7
        reward = float(np.random.randn())
        next_state = np.random.randn(skill_metadata["state_dim"]).astype(np.float32)
        worker.store_experience(
            state=state,
            action=action if not isinstance(action, np.ndarray) else 0,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            entropy=entropy,
        )
        state = next_state
        action, log_prob, entropy = worker.select_action(state, explore=True)

    update_loss = worker.update_policy()
    printer.pretty("UpdateLoss", update_loss, "success")

    worker.record_episode_outcome(episode_return=1.5, episode_length=8, success=True)
    metrics = worker.get_performance_metrics()
    report = worker.get_training_report()
    printer.pretty("Metrics", metrics, "success")
    printer.pretty("Report", {"skill_id": report["skill_id"], "last_update": report["last_update"]}, "success")

    checkpoint_path = Path("/tmp/skill_worker_test_checkpoint.pkl")
    worker.save_checkpoint(str(checkpoint_path))

    restored = SkillWorker.create_worker(
        skill_id=2,
        skill_metadata=skill_metadata,
    )
    restored.load_checkpoint(str(checkpoint_path))
    restored_metrics = restored.get_performance_metrics()
    printer.pretty("RestoredMetrics", restored_metrics, "success")

    print("\n=== Test ran successfully ===\n")
