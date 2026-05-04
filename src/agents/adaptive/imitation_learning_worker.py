from __future__ import annotations

import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from collections import deque

from .adaptive_memory import MultiModalMemory
from .utils.config_loader import load_global_config, get_config_section
from .utils.adaptive_errors import *
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Imitation Learning")
printer = PrettyPrinter


@dataclass(frozen=True)
class DemonstrationSample:
    """Structured demonstration sample used for persistence and diagnostics."""

    state: torch.Tensor
    action: torch.Tensor
    source: str
    timestamp: datetime


class ImitationLearningWorker:
    """
    Imitation Learning Worker with Behavior Cloning and DAgger.

    Core responsibilities
    ---------------------
    - Learn from offline demonstrations via behavior cloning.
    - Aggregate online expert feedback with DAgger.
    - Blend imitation and reinforcement-learning losses for policy refinement.
    - Persist demonstration buffers and training state safely.
    - Log demonstration-driven experience into adaptive memory.

    Production-oriented extensions
    ------------------------------
    - Structured error handling integrated with ``adaptive_errors``.
    - Config-driven optimizer, scheduler, persistence, and query behavior.
    - Support for discrete and continuous action spaces.
    - Safe handling of policy networks that output logits or probabilities.
    - Validation-aware persistence and full checkpoint save/load helpers.
    - Diagnostics, counts, and training reports for observability.
    """

    SUPPORTED_OPTIMIZERS = {"adam", "adamw", "sgd", "rmsprop"}
    SUPPORTED_SCHEDULERS = {"none", "step", "cosine", "reduce_on_plateau"}
    SUPPORTED_QUERY_STRATEGIES = {"probabilistic", "uncertainty", "hybrid"}

    def __init__(self, action_dim: int, state_dim: int, policy_network: nn.Module,
                 memory: Optional[MultiModalMemory] = None) -> None:
        self.config = load_global_config()
        self.il_config = get_config_section("imitation_learning")
        self._load_config()

        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        ensure_positive(self.action_dim, "action_dim", component="imitation_learning")
        ensure_positive(self.state_dim, "state_dim", component="imitation_learning")
        ensure_instance(policy_network, nn.Module, "policy_network", component="imitation_learning")

        self.policy_net = policy_network
        self.memory = memory if memory is not None else MultiModalMemory()
        if memory is not None:
            ensure_instance(memory, MultiModalMemory, "memory", component="imitation_learning")

        self.device = self._resolve_device()
        self.policy_net = self.policy_net.to(self.device)

        self.optimizer = self._configure_optimizer()
        self.scheduler = self._configure_scheduler()

        # Demonstration buffers
        self.demo_memory: Deque[DemonstrationSample] = deque(maxlen=self.demo_capacity)
        self.dagger_memory: Deque[DemonstrationSample] = deque(maxlen=self.dagger_capacity)

        # Expert / DAgger state
        self.expert_policy: Optional[Callable[[np.ndarray], Any]] = None
        self.query_prob = float(self.initial_query_prob)
        self.total_queries = 0
        self.total_expert_queries = 0
        self.update_count = 0
        self.behavior_cloning_steps = 0
        self.dagger_updates = 0
        self.last_bc_loss: Optional[float] = None
        self.last_dagger_loss: Optional[float] = None
        self.last_combined_loss: Optional[float] = None
        self.last_il_loss: Optional[float] = None
        self.last_rl_loss: Optional[float] = None
        self.training_history: Dict[str, List[float]] = {
            "bc_loss": [],
            "dagger_loss": [],
            "combined_loss": [],
            "query_probability": [],
        }

        adaptive_root = Path(__file__).resolve().parent
        self.demo_dir = adaptive_root / self.demo_dirname
        self.demo_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Imitation Learning Worker initialized | action_dim=%s state_dim=%s continuous=%s device=%s",
            self.action_dim,
            self.state_dim,
            self.continuous_actions,
            self.device,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        try:
            self.batch_size = int(self.il_config.get("batch_size", self.config.get("batch_size", 32)))
            self.lr = float(self.il_config.get("learning_rate", 0.001))
            self.clip_value = float(self.il_config.get("grad_clip", 1.0))
            self.mix_ratio = float(self.il_config.get("rl_mix_ratio", 0.7))
            self.entropy_threshold = float(self.il_config.get("entropy_threshold", 0.5))
            self.continuous_actions = bool(self.il_config.get("continuous_actions", False))
            self.demo_capacity = int(self.il_config.get("demo_capacity", 10000))
            self.dagger_capacity = int(self.il_config.get("dagger_capacity", 5000))
            self.dagger_frequency = int(self.il_config.get("dagger_frequency", 5))
            self.initial_query_prob = float(self.il_config.get("initial_query_prob", 0.8))
            self.min_query_prob = float(self.il_config.get("min_query_prob", 0.05))
            self.query_decay = float(self.il_config.get("query_decay", 0.99))
            self.query_strategy = str(self.il_config.get("query_strategy", "hybrid")).lower()
            self.use_uncertainty_query = bool(self.il_config.get("use_uncertainty_query", True))
            self.optimizer_name = str(self.il_config.get("optimizer_name", "adam")).lower()
            self.weight_decay = float(self.il_config.get("weight_decay", 0.0))
            self.adam_beta1 = float(self.il_config.get("adam_beta1", 0.9))
            self.adam_beta2 = float(self.il_config.get("adam_beta2", 0.999))
            self.adam_epsilon = float(self.il_config.get("adam_epsilon", 1.0e-8))
            self.sgd_momentum = float(self.il_config.get("sgd_momentum", 0.9))
            self.rmsprop_alpha = float(self.il_config.get("rmsprop_alpha", 0.99))
            self.label_smoothing = float(self.il_config.get("label_smoothing", 0.0))
            self.imitation_reward = float(self.il_config.get("imitation_reward", 2.0))
            self.advantage_weighted_imitation = bool(self.il_config.get("advantage_weighted_imitation", False))
            self.normalize_advantages_for_il = bool(self.il_config.get("normalize_advantages_for_il", True))
            self.log_to_memory = bool(self.il_config.get("log_to_memory", True))
            self.store_dagger_in_memory = bool(self.il_config.get("store_dagger_in_memory", True))
            self.memory_context_source = str(self.il_config.get("memory_context_source", "demonstration"))
            self.device_preference = str(self.il_config.get("device", "auto")).lower()
            self.demo_dirname = str(self.il_config.get("demo_dirname", "demo"))
            self.checkpoint_protocol = int(self.il_config.get("checkpoint_protocol", pickle.HIGHEST_PROTOCOL))
            self.save_optimizer_state = bool(self.il_config.get("save_optimizer_state", True))
            self.scheduler_config = self.il_config.get("scheduler", {}) or {}
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationValueError(
                "Failed to parse imitation_learning configuration values.",
                component="imitation_learning",
                details={"section": "imitation_learning"},
                remediation="Ensure imitation_learning configuration values are valid scalars.",
                cause=exc,
            ) from exc

        ensure_positive(self.batch_size, "batch_size", component="imitation_learning")
        ensure_positive(self.lr, "learning_rate", component="imitation_learning")
        ensure_positive(self.clip_value, "grad_clip", component="imitation_learning")
        ensure_in_range(self.mix_ratio, "rl_mix_ratio", minimum=0.0, maximum=1.0, component="imitation_learning")
        ensure_in_range(self.entropy_threshold, "entropy_threshold", minimum=0.0, component="imitation_learning")
        ensure_positive(self.demo_capacity, "demo_capacity", component="imitation_learning")
        ensure_positive(self.dagger_capacity, "dagger_capacity", component="imitation_learning")
        ensure_positive(self.dagger_frequency, "dagger_frequency", component="imitation_learning")
        ensure_in_range(self.initial_query_prob, "initial_query_prob", minimum=0.0, maximum=1.0, component="imitation_learning")
        ensure_in_range(self.min_query_prob, "min_query_prob", minimum=0.0, maximum=1.0, component="imitation_learning")
        ensure_in_range(self.query_decay, "query_decay", minimum=0.0, maximum=1.0, component="imitation_learning")
        ensure_in_range(self.label_smoothing, "label_smoothing", minimum=0.0, maximum=1.0, component="imitation_learning")
        ensure_positive(self.imitation_reward, "imitation_reward", component="imitation_learning")
        ensure_positive(self.weight_decay + 1.0, "weight_decay_plus_one", component="imitation_learning")

        if self.min_query_prob > self.initial_query_prob:
            raise InvalidConfigurationValueError(
                "min_query_prob cannot exceed initial_query_prob.",
                component="imitation_learning",
            )
        if self.optimizer_name not in self.SUPPORTED_OPTIMIZERS:
            raise InvalidConfigurationValueError(
                f"Unsupported optimizer_name: {self.optimizer_name}",
                component="imitation_learning",
                details={"supported": sorted(self.SUPPORTED_OPTIMIZERS)},
            )
        if self.query_strategy not in self.SUPPORTED_QUERY_STRATEGIES:
            raise InvalidConfigurationValueError(
                f"Unsupported query_strategy: {self.query_strategy}",
                component="imitation_learning",
                details={"supported": sorted(self.SUPPORTED_QUERY_STRATEGIES)},
            )

        scheduler_name = str(self.scheduler_config.get("name", "none")).lower()
        if scheduler_name not in self.SUPPORTED_SCHEDULERS:
            raise InvalidConfigurationValueError(
                f"Unsupported imitation scheduler name: {scheduler_name}",
                component="imitation_learning",
                details={"supported": sorted(self.SUPPORTED_SCHEDULERS)},
            )
        self.scheduler_name = scheduler_name

    def _resolve_device(self) -> torch.device:
        if self.device_preference == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device_preference == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        params = self.policy_net.parameters()
        if self.optimizer_name == "adam":
            return torch.optim.Adam(
                params,
                lr=self.lr,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.lr,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.lr,
                momentum=self.sgd_momentum,
                weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "rmsprop":
            return torch.optim.RMSprop(
                params,
                lr=self.lr,
                alpha=self.rmsprop_alpha,
                momentum=self.sgd_momentum,
                weight_decay=self.weight_decay,
            )
        raise InvalidConfigurationValueError(
            f"Unsupported optimizer_name: {self.optimizer_name}",
            component="imitation_learning",
        )

    def _configure_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if self.scheduler_name == "none":
            return None

        if self.scheduler_name == "step":
            step_size = int(self.scheduler_config.get("step_size", 10))
            gamma = float(self.scheduler_config.get("gamma", 0.5))
            ensure_positive(step_size, "scheduler.step_size", component="imitation_learning")
            ensure_in_range(gamma, "scheduler.gamma", minimum=0.0, maximum=1.0, component="imitation_learning")
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        if self.scheduler_name == "cosine":
            t_max = int(self.scheduler_config.get("t_max", 10))
            eta_min = float(self.scheduler_config.get("min_lr", 0.0))
            ensure_positive(t_max, "scheduler.t_max", component="imitation_learning")
            ensure_in_range(eta_min, "scheduler.min_lr", minimum=0.0, component="imitation_learning")
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)

        if self.scheduler_name == "reduce_on_plateau":
            factor = float(self.scheduler_config.get("factor", 0.5))
            patience = int(self.scheduler_config.get("patience", 5))
            min_lr = float(self.scheduler_config.get("min_lr", 1.0e-6))
            ensure_in_range(factor, "scheduler.factor", minimum=0.0, maximum=1.0, component="imitation_learning")
            ensure_positive(patience, "scheduler.patience", component="imitation_learning")
            ensure_in_range(min_lr, "scheduler.min_lr", minimum=0.0, component="imitation_learning")
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=str(self.scheduler_config.get("mode", "min")).lower(),
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )

        return None

    # ------------------------------------------------------------------
    # Demonstration preprocessing and validation
    # ------------------------------------------------------------------
    def _prepare_state_tensor(self, state: Any) -> torch.Tensor:
        try:
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
        except Exception as exc:
            raise InvalidTypeError(
                "state must be numeric and convertible to a float tensor.",
                component="imitation_learning",
                details={"state_type": type(state).__name__},
                cause=exc,
            ) from exc

        state_tensor = state_tensor.reshape(-1)
        if state_tensor.numel() != self.state_dim:
            raise InvalidValueError(
                "state dimension does not match state_dim.",
                component="imitation_learning",
                details={"expected": self.state_dim, "received": int(state_tensor.numel())},
            )
        if not torch.isfinite(state_tensor).all():
            raise InvalidValueError(
                "state contains NaN or infinite values.",
                component="imitation_learning",
            )
        return state_tensor

    def _prepare_action_tensor(self, action: Any) -> torch.Tensor:
        if self.continuous_actions:
            try:
                action_tensor = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
            except Exception as exc:
                raise InvalidTypeError(
                    "continuous action must be numeric and convertible to a float tensor.",
                    component="imitation_learning",
                    details={"action_type": type(action).__name__},
                    cause=exc,
                ) from exc
            if action_tensor.numel() != self.action_dim:
                raise InvalidValueError(
                    "continuous action dimension does not match action_dim.",
                    component="imitation_learning",
                    details={"expected": self.action_dim, "received": int(action_tensor.numel())},
                )
            if not torch.isfinite(action_tensor).all():
                raise InvalidValueError("action contains NaN or infinite values.", component="imitation_learning")
            return action_tensor

        try:
            action_tensor = torch.as_tensor(action, dtype=torch.long).reshape(-1)
        except Exception as exc:
            raise InvalidTypeError(
                "discrete action must be integer-like and convertible to a long tensor.",
                component="imitation_learning",
                details={"action_type": type(action).__name__},
                cause=exc,
            ) from exc

        if action_tensor.numel() == 0:
            raise InvalidValueError("discrete action cannot be empty.", component="imitation_learning")
        action_value = int(action_tensor[0].item())
        if action_value < 0 or action_value >= self.action_dim:
            raise InvalidValueError(
                "discrete action is out of range.",
                component="imitation_learning",
                details={"action": action_value, "action_dim": self.action_dim},
            )
        return torch.tensor(action_value, dtype=torch.long)

    def _build_demo_sample(
        self,
        state: Any,
        action: Any,
        *,
        source: str,
        timestamp: Optional[datetime] = None,
    ) -> DemonstrationSample:
        state_tensor = self._prepare_state_tensor(state)
        action_tensor = self._prepare_action_tensor(action)
        return DemonstrationSample(
            state=state_tensor,
            action=action_tensor,
            source=str(source),
            timestamp=datetime.now() if timestamp is None else timestamp,
        )

    def _sample_batch(
        self,
        dataset: Sequence[DemonstrationSample],
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ensure_positive(batch_size, "batch_size", component="imitation_learning")
        if len(dataset) == 0:
            raise DemonstrationError(
                "Cannot sample from an empty demonstration dataset.",
                component="imitation_learning",
            )

        sample_size = min(batch_size, len(dataset))
        batch = random.sample(list(dataset), sample_size)
        states = torch.stack([sample.state for sample in batch]).to(self.device)
        actions = torch.stack([sample.action for sample in batch]).to(self.device)
        return states, actions

    def _resolve_training_outputs(self, states: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Return policy output and whether it represents probabilities for discrete control."""
        if hasattr(self.policy_net, "forward_logits") and callable(getattr(self.policy_net, "forward_logits")):
            outputs = self.policy_net.forward_logits(states)
            return outputs, False

        outputs = self.policy_net(states)
        if self.continuous_actions:
            return outputs, False

        probs_like = (
            torch.isfinite(outputs).all()
            and torch.all(outputs >= 0)
            and outputs.ndim == 2
            and torch.allclose(outputs.sum(dim=-1), torch.ones(outputs.shape[0], device=outputs.device), atol=1e-4, rtol=1e-4)
        )
        return outputs, bool(probs_like)

    def _discrete_loss(self, outputs: torch.Tensor, actions: torch.Tensor, probs_like: bool) -> torch.Tensor:
        if probs_like:
            log_probs = torch.log(outputs.clamp_min(1.0e-8))
            return F.nll_loss(log_probs, actions)
        return F.cross_entropy(outputs, actions, label_smoothing=self.label_smoothing)

    def _per_sample_il_loss(self, outputs: torch.Tensor, actions: torch.Tensor, probs_like: bool) -> torch.Tensor:
        if self.continuous_actions:
            per_sample = F.mse_loss(outputs, actions, reduction="none")
            return per_sample.mean(dim=-1)
        if probs_like:
            log_probs = torch.log(outputs.clamp_min(1.0e-8))
            return F.nll_loss(log_probs, actions, reduction="none")
        return F.cross_entropy(outputs, actions, reduction="none", label_smoothing=self.label_smoothing)

    def _compute_il_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        *,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        outputs, probs_like = self._resolve_training_outputs(states)
        if sample_weights is None:
            if self.continuous_actions:
                loss = F.mse_loss(outputs, actions)
            else:
                loss = self._discrete_loss(outputs, actions, probs_like)
            return loss, outputs, probs_like

        per_sample = self._per_sample_il_loss(outputs, actions, probs_like)
        weights = sample_weights.to(device=per_sample.device, dtype=per_sample.dtype)
        weights = weights / weights.sum().clamp_min(1.0e-8)
        loss = torch.sum(per_sample * weights)
        return loss, outputs, probs_like

    def _optimizer_step(self, loss: torch.Tensor) -> float:
        if not torch.isfinite(loss):
            raise ImitationLearningError(
                "Imitation-learning loss became non-finite.",
                component="imitation_learning",
            )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_value)
        self.optimizer.step()
        return float(loss.item())

    def _scheduler_step(self, metric: Optional[float] = None) -> None:
        if self.scheduler is None:
            return
        if self.scheduler_name == "reduce_on_plateau":
            if metric is None:
                return
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------
    def get_action(self, state: np.ndarray, deterministic: bool = True) -> Union[int, np.ndarray]:
        """Get action from imitation-learning policy."""
        state_tensor = self._prepare_state_tensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if self.continuous_actions:
                output = self.policy_net(state_tensor)
                return output.squeeze(0).detach().cpu().numpy()

            if hasattr(self.policy_net, "forward_logits") and callable(getattr(self.policy_net, "forward_logits")):
                logits = self.policy_net.forward_logits(state_tensor)
                probs = torch.softmax(logits, dim=-1)
            else:
                output = self.policy_net(state_tensor)
                if output.ndim != 2 or output.shape[-1] != self.action_dim:
                    raise ImitationLearningError(
                        "Policy network returned an unexpected discrete action shape.",
                        component="imitation_learning",
                        details={"shape": tuple(output.shape)},
                    )
                probs = output
                if torch.any(probs < 0) or not torch.allclose(probs.sum(dim=-1), torch.ones(1, device=probs.device), atol=1e-4, rtol=1e-4):
                    probs = torch.softmax(output, dim=-1)

            if deterministic:
                return int(torch.argmax(probs, dim=-1).item())
            dist = torch.distributions.Categorical(probs=probs)
            return int(dist.sample().item())

    def register_expert(self, expert: Callable[[np.ndarray], Any]) -> None:
        """Register expert policy function used by DAgger."""
        if not callable(expert):
            raise ExpertPolicyError(
                "expert must be callable.",
                component="imitation_learning",
                details={"expert_type": type(expert).__name__},
            )
        self.expert_policy = expert
        logger.info("Expert policy registered")

    def load_demonstrations(self, demonstrations: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
        """Load offline demonstration dataset from an in-memory iterable."""
        ensure_instance(demonstrations, Sequence, "demonstrations", component="imitation_learning")

        loaded = 0
        for idx, demo in enumerate(demonstrations):
            ensure_instance(demo, Mapping, f"demonstration[{idx}]", component="imitation_learning")
            if "state" not in demo:
                raise MissingFieldError(
                    "Demonstration sample is missing 'state'.",
                    component="imitation_learning",
                    details={"index": idx},
                )
            if "action" not in demo:
                raise MissingFieldError(
                    "Demonstration sample is missing 'action'.",
                    component="imitation_learning",
                    details={"index": idx},
                )
            sample = self._build_demo_sample(
                demo["state"],
                demo["action"],
                source=str(demo.get("source", "offline")),
                timestamp=demo.get("timestamp", None),
            )
            self.demo_memory.append(sample)
            loaded += 1

        logger.info("Loaded %d offline demonstrations", loaded)
        return self.get_demonstration_count()

    def add_demonstration(self, state: Any, action: Any, *, source: str = "offline") -> DemonstrationSample:
        """Add a single demonstration sample and optionally log it to adaptive memory."""
        sample = self._build_demo_sample(state, action, source=source)
        self.demo_memory.append(sample)

        if self.log_to_memory:
            self.memory.store_experience(
                state=sample.state.cpu().numpy(),
                action=sample.action.cpu().numpy() if self.continuous_actions else int(sample.action.item()),
                reward=self.imitation_reward,
                next_state=None,
                done=True,
                context={"source": self.memory_context_source, "demo_source": source},
            )
        return sample

    def clear_demonstrations(self, clear_dagger: bool = True) -> None:
        self.demo_memory.clear()
        if clear_dagger:
            self.dagger_memory.clear()

    def behavior_cloning(
        self,
        epochs: int = 10,
        validation_data: Optional[Sequence[Mapping[str, Any]]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train using pure behavior cloning."""
        ensure_positive(epochs, "epochs", component="imitation_learning")
        if len(self.demo_memory) == 0:
            raise DemonstrationError(
                "Behavior cloning requires at least one offline demonstration.",
                component="imitation_learning",
            )

        dataset = list(self.demo_memory)
        val_samples: List[DemonstrationSample] = []
        if validation_data is not None:
            for idx, sample in enumerate(validation_data):
                ensure_instance(sample, Mapping, f"validation_data[{idx}]", component="imitation_learning")
                val_samples.append(
                    self._build_demo_sample(
                        sample["state"],
                        sample["action"],
                        source=str(sample.get("source", "validation")),
                        timestamp=sample.get("timestamp", None),
                    )
                )

        history = {"epoch_losses": [], "validation_losses": []}
        logger.info("Starting Behavior Cloning with %d samples", len(dataset))

        for epoch in range(epochs):
            self.policy_net.train()
            epoch_loss = 0.0
            num_batches = 0
            random.shuffle(dataset)

            for start in range(0, len(dataset), self.batch_size):
                batch = dataset[start : start + self.batch_size]
                states = torch.stack([sample.state for sample in batch]).to(self.device)
                actions = torch.stack([sample.action for sample in batch]).to(self.device)
                loss, _, _ = self._compute_il_loss(states, actions)
                epoch_loss += self._optimizer_step(loss)
                num_batches += 1
                self.behavior_cloning_steps += 1

            avg_loss = epoch_loss / max(1, num_batches)
            self.last_bc_loss = avg_loss
            self.training_history["bc_loss"].append(avg_loss)
            history["epoch_losses"].append(avg_loss)

            validation_loss = None
            if val_samples:
                validation_loss = self.evaluate_demonstrations(val_samples)
                history["validation_losses"].append(validation_loss)
                self._scheduler_step(validation_loss)
            else:
                self._scheduler_step(avg_loss)

            if verbose:
                if validation_loss is None:
                    logger.info("BC Epoch %d/%d | Loss: %.6f", epoch + 1, epochs, avg_loss)
                else:
                    logger.info(
                        "BC Epoch %d/%d | Loss: %.6f | Val Loss: %.6f",
                        epoch + 1,
                        epochs,
                        avg_loss,
                        validation_loss,
                    )

        return {
            "epochs": epochs,
            "num_samples": len(dataset),
            "last_loss": self.last_bc_loss,
            "history": history,
        }

    def _should_query_expert(self, state: np.ndarray) -> bool:
        probabilistic = np.random.rand() < self.query_prob
        uncertainty = self.uncertainty_query(state) if self.use_uncertainty_query else False

        if self.query_strategy == "probabilistic":
            return probabilistic
        if self.query_strategy == "uncertainty":
            return uncertainty
        return probabilistic or uncertainty

    def dagger_query(self, state: np.ndarray, agent_action: Any) -> Any:
        """
        DAgger interaction: decide whether to query expert and aggregate data.

        Returns the expert action when queried, otherwise the agent action.
        """
        if self.expert_policy is None:
            return agent_action

        state_tensor = self._prepare_state_tensor(state)
        self.total_queries += 1

        if self._should_query_expert(state_tensor.cpu().numpy()):
            try:
                expert_action = self.expert_policy(state_tensor.cpu().numpy())
            except Exception as exc:
                raise ExpertPolicyError(
                    "Expert policy failed during DAgger query.",
                    component="imitation_learning",
                    details={"state_dim": self.state_dim},
                    cause=exc,
                ) from exc

            sample = self._build_demo_sample(state_tensor.cpu().numpy(), expert_action, source="dagger")
            self.dagger_memory.append(sample)
            self.total_expert_queries += 1
            self.query_prob = max(self.min_query_prob, self.query_prob * self.query_decay)
            self.training_history["query_probability"].append(self.query_prob)

            if self.log_to_memory and self.store_dagger_in_memory:
                self.memory.store_experience(
                    state=sample.state.cpu().numpy(),
                    action=sample.action.cpu().numpy() if self.continuous_actions else int(sample.action.item()),
                    reward=self.imitation_reward,
                    next_state=None,
                    done=True,
                    context={"source": self.memory_context_source, "demo_source": "dagger"},
                )
            return expert_action

        self.training_history["query_probability"].append(self.query_prob)
        return agent_action

    def uncertainty_query(self, state: np.ndarray) -> bool:
        """Entropy- or dispersion-based expert query decision."""
        state_tensor = self._prepare_state_tensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if self.continuous_actions:
                outputs = self.policy_net(state_tensor)
                uncertainty = float(outputs.std(dim=-1).mean().item())
                return bool(uncertainty > self.entropy_threshold)

            if hasattr(self.policy_net, "forward_logits") and callable(getattr(self.policy_net, "forward_logits")):
                logits = self.policy_net.forward_logits(state_tensor)
                probs = torch.softmax(logits, dim=-1)
            else:
                outputs = self.policy_net(state_tensor)
                if torch.any(outputs < 0) or not torch.allclose(outputs.sum(dim=-1), torch.ones(outputs.shape[0], device=outputs.device), atol=1e-4, rtol=1e-4):
                    probs = torch.softmax(outputs, dim=-1)
                else:
                    probs = outputs

            entropy = float((-torch.sum(probs * torch.log(probs.clamp_min(1.0e-8)), dim=-1)).item())
            return bool(entropy > self.entropy_threshold)

    def dagger_update(self, force: bool = True) -> Dict[str, Any]:
        """Perform DAgger update using aggregated demonstrations."""
        combined_data = list(self.demo_memory) + list(self.dagger_memory)
        if len(combined_data) == 0:
            raise DemonstrationError(
                "DAgger update requires at least one demonstration.",
                component="imitation_learning",
            )
        if len(self.dagger_memory) == 0:
            return {"skipped": True, "reason": "no_dagger_samples", "loss": None}
        if not force and (self.total_expert_queries % self.dagger_frequency != 0):
            return {"skipped": True, "reason": "frequency_gate", "loss": None}

        states, actions = self._sample_batch(combined_data, self.batch_size)
        self.policy_net.train()
        loss, _, _ = self._compute_il_loss(states, actions)
        loss_value = self._optimizer_step(loss)
        self._scheduler_step(loss_value)

        self.last_dagger_loss = loss_value
        self.training_history["dagger_loss"].append(loss_value)
        self.update_count += 1
        self.dagger_updates += 1

        logger.info(
            "DAgger Update | Loss: %.6f | Samples: %d | Expert queries: %d",
            loss_value,
            len(combined_data),
            self.total_expert_queries,
        )
        return {
            "skipped": False,
            "loss": loss_value,
            "samples": len(combined_data),
            "expert_queries": self.total_expert_queries,
        }

    def mixed_objective_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        rl_loss: torch.Tensor,
    ) -> float:
        """
        Combined RL and imitation-learning update.

        The imitation component can optionally be weighted by advantage magnitude,
        which lets the IL objective focus more strongly on higher-signal samples.
        """
        ensure_instance(states, torch.Tensor, "states", component="imitation_learning")
        ensure_instance(actions, torch.Tensor, "actions", component="imitation_learning")
        ensure_instance(advantages, torch.Tensor, "advantages", component="imitation_learning")
        ensure_instance(rl_loss, torch.Tensor, "rl_loss", component="imitation_learning")

        states = states.to(self.device)
        actions = actions.to(self.device)
        advantages = advantages.to(self.device)
        rl_loss = rl_loss.to(self.device)

        sample_weights = None
        if self.advantage_weighted_imitation:
            adv = advantages.detach().reshape(-1)
            if adv.shape[0] != states.shape[0]:
                raise InvalidValueError(
                    "advantages batch size must match states batch size.",
                    component="imitation_learning",
                    details={"advantages": int(adv.shape[0]), "states": int(states.shape[0])},
                )
            weights = adv.abs()
            if self.normalize_advantages_for_il:
                weights = weights / weights.mean().clamp_min(1.0e-8)
            sample_weights = weights

        self.policy_net.train()
        il_loss, _, _ = self._compute_il_loss(states, actions, sample_weights=sample_weights)
        combined_loss = (self.mix_ratio * rl_loss) + ((1.0 - self.mix_ratio) * il_loss)
        loss_value = self._optimizer_step(combined_loss)

        self.last_combined_loss = loss_value
        self.last_il_loss = float(il_loss.item())
        self.last_rl_loss = float(rl_loss.detach().item())
        self.training_history["combined_loss"].append(loss_value)
        self.update_count += 1
        self._scheduler_step(loss_value)
        return loss_value

    # ------------------------------------------------------------------
    # Evaluation and reporting
    # ------------------------------------------------------------------
    def evaluate_demonstrations(self, dataset: Sequence[DemonstrationSample]) -> float:
        if len(dataset) == 0:
            raise DemonstrationError(
                "Cannot evaluate an empty demonstration dataset.",
                component="imitation_learning",
            )

        self.policy_net.eval()
        losses: List[float] = []
        with torch.no_grad():
            for start in range(0, len(dataset), self.batch_size):
                batch = dataset[start : start + self.batch_size]
                states = torch.stack([sample.state for sample in batch]).to(self.device)
                actions = torch.stack([sample.action for sample in batch]).to(self.device)
                loss, _, _ = self._compute_il_loss(states, actions)
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def get_demonstration_count(self) -> Dict[str, int]:
        return {
            "offline_demos": len(self.demo_memory),
            "dagger_demos": len(self.dagger_memory),
            "total_demos": len(self.demo_memory) + len(self.dagger_memory),
        }

    def get_training_report(self) -> Dict[str, Any]:
        return {
            "counts": self.get_demonstration_count(),
            "device": str(self.device),
            "continuous_actions": self.continuous_actions,
            "optimizer": self.optimizer_name,
            "scheduler": self.scheduler_name,
            "query_prob": self.query_prob,
            "total_queries": self.total_queries,
            "expert_queries": self.total_expert_queries,
            "update_count": self.update_count,
            "behavior_cloning_steps": self.behavior_cloning_steps,
            "dagger_updates": self.dagger_updates,
            "last_bc_loss": self.last_bc_loss,
            "last_dagger_loss": self.last_dagger_loss,
            "last_combined_loss": self.last_combined_loss,
            "last_il_loss": self.last_il_loss,
            "last_rl_loss": self.last_rl_loss,
            "current_lr": float(self.optimizer.param_groups[0]["lr"]),
            "history_lengths": {key: len(value) for key, value in self.training_history.items()},
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _resolve_demo_path(self, filepath: Union[str, Path]) -> Path:
        path = Path(filepath)
        if not path.is_absolute():
            path = self.demo_dir / path
        return path

    def _serialize_sample(self, sample: DemonstrationSample) -> Dict[str, Any]:
        return {
            "state": sample.state.detach().cpu().numpy(),
            "action": sample.action.detach().cpu().numpy(),
            "source": sample.source,
            "timestamp": sample.timestamp.isoformat(),
        }

    def save_demonstrations(self, filepath: Union[str, Path]) -> Path:
        """Save offline and DAgger demonstrations to disk."""
        target_path = self._resolve_demo_path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "metadata": {
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "continuous_actions": self.continuous_actions,
                "offline_count": len(self.demo_memory),
                "dagger_count": len(self.dagger_memory),
                "query_prob": self.query_prob,
                "saved_at": datetime.now().isoformat(),
            },
            "offline": [self._serialize_sample(sample) for sample in self.demo_memory],
            "dagger": [self._serialize_sample(sample) for sample in self.dagger_memory],
        }

        try:
            torch.save(payload, target_path, pickle_protocol=self.checkpoint_protocol)
        except Exception as exc:
            raise DemonstrationPersistenceError(
                f"Failed to save demonstrations to {target_path}.",
                component="imitation_learning",
                details={"filepath": str(target_path)},
                remediation="Verify the target path is writable and the payload is serializable.",
                cause=exc,
            ) from exc

        logger.info("Saved demonstrations to %s", target_path)
        return target_path

    def load_demonstrations_from_file(self, filepath: Union[str, Path]) -> Dict[str, int]:
        """Load offline and DAgger demonstrations from disk."""
        source_path = self._resolve_demo_path(filepath)
        if not source_path.exists():
            raise DemonstrationNotFoundError(
                f"Demonstration file not found: {source_path}",
                component="imitation_learning",
                details={"filepath": str(source_path)},
            )

        try:
            data = torch.load(source_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise DemonstrationPersistenceError(
                f"Failed to load demonstrations from {source_path}.",
                component="imitation_learning",
                details={"filepath": str(source_path)},
                cause=exc,
            ) from exc

        if not isinstance(data, dict) or "offline" not in data:
            raise DemonstrationFormatError(
                f"Unsupported demonstration format in {source_path}.",
                component="imitation_learning",
                details={"keys": sorted(data.keys()) if isinstance(data, dict) else None},
            )

        self.demo_memory.clear()
        self.dagger_memory.clear()

        try:
            for sample in data.get("offline", []):
                ensure_instance(sample, Mapping, "offline_sample", component="imitation_learning")
                self.demo_memory.append(
                    self._build_demo_sample(
                        sample["state"],
                        sample["action"],
                        source=str(sample.get("source", "offline")),
                        timestamp=datetime.fromisoformat(sample["timestamp"]) if sample.get("timestamp") else None,
                    )
                )
            for sample in data.get("dagger", []):
                ensure_instance(sample, Mapping, "dagger_sample", component="imitation_learning")
                self.dagger_memory.append(
                    self._build_demo_sample(
                        sample["state"],
                        sample["action"],
                        source=str(sample.get("source", "dagger")),
                        timestamp=datetime.fromisoformat(sample["timestamp"]) if sample.get("timestamp") else None,
                    )
                )
        except KeyError as exc:
            raise DemonstrationFormatError(
                "Demonstration file is missing required fields.",
                component="imitation_learning",
                details={"missing_key": str(exc)},
            ) from exc
        except AdaptiveError:
            raise
        except Exception as exc:
            raise wrap_exception(
                exc,
                DemonstrationFormatError,
                "Failed to reconstruct demonstrations from file.",
                component="imitation_learning",
                details={"filepath": str(source_path)},
            ) from exc

        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        if isinstance(metadata, Mapping):
            maybe_query_prob = metadata.get("query_prob")
            if isinstance(maybe_query_prob, (int, float)) and np.isfinite(maybe_query_prob):
                self.query_prob = float(np.clip(maybe_query_prob, self.min_query_prob, 1.0))

        logger.info(
            "Loaded demonstrations from %s | offline=%d dagger=%d",
            source_path,
            len(self.demo_memory),
            len(self.dagger_memory),
        )
        return self.get_demonstration_count()

    def save_checkpoint(self, filepath: Union[str, Path]) -> Path:
        path = self._resolve_demo_path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "continuous_actions": self.continuous_actions,
                "query_prob": self.query_prob,
                "update_count": self.update_count,
                "behavior_cloning_steps": self.behavior_cloning_steps,
                "dagger_updates": self.dagger_updates,
                "total_queries": self.total_queries,
                "total_expert_queries": self.total_expert_queries,
                "optimizer_name": self.optimizer_name,
                "scheduler_name": self.scheduler_name,
            },
            "policy_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.save_optimizer_state else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "demo_memory": [self._serialize_sample(sample) for sample in self.demo_memory],
            "dagger_memory": [self._serialize_sample(sample) for sample in self.dagger_memory],
            "training_history": self.training_history,
        }
        try:
            torch.save(payload, path, pickle_protocol=self.checkpoint_protocol)
        except Exception as exc:
            raise DemonstrationPersistenceError(
                f"Failed to save imitation-learning checkpoint to {path}.",
                component="imitation_learning",
                details={"filepath": str(path)},
                cause=exc,
            ) from exc
        logger.info("Imitation-learning checkpoint saved to %s", path)
        return path

    def load_checkpoint(self, filepath: Union[str, Path], load_optimizer_state: bool = True) -> Dict[str, Any]:
        path = self._resolve_demo_path(filepath)
        if not path.exists():
            raise DemonstrationNotFoundError(
                f"Imitation-learning checkpoint not found: {path}",
                component="imitation_learning",
                details={"filepath": str(path)},
            )

        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as exc:
            raise DemonstrationPersistenceError(
                f"Failed to load imitation-learning checkpoint from {path}.",
                component="imitation_learning",
                details={"filepath": str(path)},
                cause=exc,
            ) from exc

        if not isinstance(payload, dict) or "policy_state_dict" not in payload:
            raise DemonstrationFormatError(
                "Checkpoint payload is missing required keys.",
                component="imitation_learning",
                details={"filepath": str(path)},
            )

        self.policy_net.load_state_dict(payload["policy_state_dict"])
        if load_optimizer_state and self.save_optimizer_state and payload.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if self.scheduler is not None and payload.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])

        self.demo_memory.clear()
        self.dagger_memory.clear()
        for sample in payload.get("demo_memory", []):
            self.demo_memory.append(
                self._build_demo_sample(
                    sample["state"],
                    sample["action"],
                    source=str(sample.get("source", "offline")),
                    timestamp=datetime.fromisoformat(sample["timestamp"]) if sample.get("timestamp") else None,
                )
            )
        for sample in payload.get("dagger_memory", []):
            self.dagger_memory.append(
                self._build_demo_sample(
                    sample["state"],
                    sample["action"],
                    source=str(sample.get("source", "dagger")),
                    timestamp=datetime.fromisoformat(sample["timestamp"]) if sample.get("timestamp") else None,
                )
            )

        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        self.query_prob = float(np.clip(config.get("query_prob", self.query_prob), self.min_query_prob, 1.0))
        self.update_count = int(config.get("update_count", self.update_count))
        self.behavior_cloning_steps = int(config.get("behavior_cloning_steps", self.behavior_cloning_steps))
        self.dagger_updates = int(config.get("dagger_updates", self.dagger_updates))
        self.total_queries = int(config.get("total_queries", self.total_queries))
        self.total_expert_queries = int(config.get("total_expert_queries", self.total_expert_queries))
        history = payload.get("training_history", {})
        if isinstance(history, Mapping):
            self.training_history = {
                "bc_loss": list(history.get("bc_loss", [])),
                "dagger_loss": list(history.get("dagger_loss", [])),
                "combined_loss": list(history.get("combined_loss", [])),
                "query_probability": list(history.get("query_probability", [])),
            }

        return self.get_training_report()


if __name__ == "__main__":
    print("\n=== Running Imitation Learning ===\n")
    printer.status("TEST", "Imitation Learning initialized", "info")

    class SimpleDiscretePolicy(nn.Module):
        def __init__(self, state_dim: int, action_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

        def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    np.random.seed(7)
    random.seed(7)
    torch.manual_seed(7)

    state_dim = 10
    action_dim = 2
    policy_net = SimpleDiscretePolicy(state_dim=state_dim, action_dim=action_dim)

    worker = ImitationLearningWorker(
        action_dim=action_dim,
        state_dim=state_dim,
        policy_network=policy_net,
    )

    def dummy_expert(state: np.ndarray) -> int:
        return int(np.sum(state) > 0)

    worker.register_expert(dummy_expert)
    assert worker.get_demonstration_count()["total_demos"] == 0

    demonstrations = [
        {"state": np.random.randn(state_dim), "action": np.random.randint(0, action_dim), "source": "offline"}
        for _ in range(max(worker.batch_size, 64))
    ]
    counts = worker.load_demonstrations(demonstrations)
    assert counts["offline_demos"] == len(demonstrations)

    added = worker.add_demonstration(np.random.randn(state_dim), 1, source="manual")
    assert isinstance(added, DemonstrationSample)
    assert worker.get_demonstration_count()["offline_demos"] == len(demonstrations) + 1

    bc_report = worker.behavior_cloning(epochs=3, verbose=True)
    printer.pretty("BC", bc_report, "success")
    assert bc_report["last_loss"] is not None

    test_state = np.random.randn(state_dim)
    predicted_action = worker.get_action(test_state)
    assert isinstance(predicted_action, int)
    assert 0 <= predicted_action < action_dim

    previous_query_prob = worker.query_prob
    returned_action = worker.dagger_query(test_state, agent_action=predicted_action)
    assert returned_action in [0, 1]
    assert worker.query_prob <= previous_query_prob

    for _ in range(worker.batch_size + 8):
        state = np.random.randn(state_dim)
        worker.dagger_query(state, agent_action=np.random.randint(0, action_dim))

    previous_updates = worker.update_count
    dagger_report = worker.dagger_update(force=True)
    printer.pretty("DAgger", dagger_report, "success")
    assert worker.update_count >= previous_updates

    uncertainty_flag = worker.uncertainty_query(np.random.randn(state_dim))
    assert isinstance(uncertainty_flag, bool)

    states = torch.randn(worker.batch_size, state_dim)
    actions = torch.randint(0, action_dim, (worker.batch_size,), dtype=torch.long)
    advantages = torch.randn(worker.batch_size)
    rl_loss = torch.tensor(0.5, requires_grad=True)
    combined_loss = worker.mixed_objective_update(states, actions, advantages, rl_loss)
    assert isinstance(combined_loss, float)

    save_file = "test_demos.pt"
    save_path = worker.save_demonstrations(save_file)
    assert save_path.exists()

    restore_worker = ImitationLearningWorker(
        action_dim=action_dim,
        state_dim=state_dim,
        policy_network=SimpleDiscretePolicy(state_dim=state_dim, action_dim=action_dim),
    )
    restore_counts = restore_worker.load_demonstrations_from_file(save_file)
    assert restore_counts["offline_demos"] == worker.get_demonstration_count()["offline_demos"]
    assert restore_counts["dagger_demos"] == worker.get_demonstration_count()["dagger_demos"]

    checkpoint_file = "imitation_checkpoint.pt"
    checkpoint_path = worker.save_checkpoint(checkpoint_file)
    assert checkpoint_path.exists()
    restored_report = restore_worker.load_checkpoint(checkpoint_file)
    printer.pretty("Restore", restored_report, "success")

    training_report = worker.get_training_report()
    printer.pretty("Report", training_report, "success")
    assert training_report["counts"]["total_demos"] >= worker.batch_size

    print("\n=== Test ran successfully ===\n")
