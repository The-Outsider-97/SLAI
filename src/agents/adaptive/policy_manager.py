from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .utils.config_loader import load_global_config, get_config_section
from .utils.adaptive_errors import *
from .adaptive_memory import MultiModalMemory
from ..learning.utils.policy_network import PolicyNetwork
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Policy Manager")
printer = PrettyPrinter


class PolicyManager:
    """
    Manager for hierarchical skill selection.

    Production-oriented extensions
    ------------------------------
    - Structured error handling integrated with `adaptive_errors`
    - Robust skill registry validation and internal skill-index mapping
    - Stable integration with the upgraded `PolicyNetwork` interface
    - Memory-aware skill selection with bounded probability adjustment
    - Configurable policy-gradient updates with entropy regularisation
    - Checkpoint save/load helpers using memory export/import
    - Expanded diagnostics, reporting, and end-to-end test coverage
    """

    _manager_instance: Optional["PolicyManager"] = None

    def __init__(self) -> None:
        super().__init__()
        self.config = load_global_config()
        self.manager_config = get_config_section("policy_manager")

        self._load_config()
        self._initialize_runtime_state()

        logger.info("Policy Manager base initialized")
        logger.info("State Dim: %s", self.state_dim)

    @classmethod
    def get_instance(cls) -> "PolicyManager":
        if cls._manager_instance is None:
            cls._manager_instance = cls()
        return cls._manager_instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._manager_instance = None

    def _load_config(self) -> None:
        try:
            raw_state_dim = self.manager_config.get("state_dim")
            self.state_dim = None if raw_state_dim is None else int(raw_state_dim)
            self.hidden_layers = [int(x) for x in self.manager_config.get("hidden_layers", [64, 32])]
            self.activation = str(self.manager_config.get("activation", "tanh")).lower()
            self.output_activation = str(self.manager_config.get("output_activation", "softmax")).lower()
            self.explore = bool(self.manager_config.get("explore", True))
            self.skill_gamma = float(self.manager_config.get("skill_discount_factor", 0.99))
            self.learning_rate = float(self.manager_config.get("manager_learning_rate", 0.001))
            self.weight_decay = float(self.manager_config.get("weight_decay", 0.0))
            self.batch_size = int(self.manager_config.get("batch_size", 32))
            self.min_batch_size = int(self.manager_config.get("min_batch_size", 16))
            self.update_frequency = int(self.manager_config.get("update_frequency", 10))
            self.entropy_coef = float(self.manager_config.get("entropy_coef", 0.0))
            self.grad_clip_norm = self.manager_config.get("grad_clip_norm", 1.0)
            self.normalize_advantages = bool(self.manager_config.get("normalize_advantages", True))
            self.advantage_epsilon = float(self.manager_config.get("advantage_epsilon", 1e-8))
            self.use_memory_bias = bool(self.manager_config.get("use_memory_bias", True))
            self.memory_query = str(self.manager_config.get("memory_query", "manager")).strip() or "manager"
            self.memory_context_type = str(self.manager_config.get("memory_context_type", "skill_selection"))
            self.memory_bias_blend = float(self.manager_config.get("memory_bias_blend", 1.0))
            self.policy_temperature = float(self.manager_config.get("policy_temperature", 1.0))
            self.pad_states = bool(self.manager_config.get("pad_states", True))
            self.strict_state_dim = bool(self.manager_config.get("strict_state_dim", False))
            self.device_name = str(self.manager_config.get("device", "cpu"))
            self.use_batch_norm = bool(self.manager_config.get("use_batch_norm", False))
            self.dropout_rate = float(self.manager_config.get("dropout_rate", 0.0))
            self.l1_lambda = float(self.manager_config.get("l1_lambda", 0.0))
            self.l2_lambda = float(self.manager_config.get("l2_lambda", 0.0))
            self.weight_init = str(self.manager_config.get("weight_init", "auto")).lower()
            self.optimizer_name = str(self.manager_config.get("optimizer_name", "adam")).lower()
            self.adam_beta1 = float(self.manager_config.get("adam_beta1", 0.9))
            self.adam_beta2 = float(self.manager_config.get("adam_beta2", 0.999))
            self.adam_epsilon = float(self.manager_config.get("adam_epsilon", 1e-8))
            self.checkpoint_protocol = int(self.manager_config.get("checkpoint_protocol", 5))
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationValueError(
                "Failed to parse Policy Manager configuration values.",
                component="policy_manager",
                details={"section": "policy_manager"},
                remediation="Ensure all Policy Manager configuration values are valid scalars.",
                cause=exc,
            ) from exc

        if self.state_dim is not None:
            ensure_positive(self.state_dim, "state_dim", component="policy_manager")
        if not self.hidden_layers or any(int(h) <= 0 for h in self.hidden_layers):
            raise InvalidConfigurationValueError(
                "hidden_layers must contain positive integers.",
                component="policy_manager",
                details={"hidden_layers": self.hidden_layers},
            )

        ensure_in_range(self.skill_gamma, "skill_discount_factor", minimum=0.0, maximum=1.0, component="policy_manager")
        ensure_positive(self.learning_rate, "manager_learning_rate", component="policy_manager")
        ensure_in_range(self.weight_decay, "weight_decay", minimum=0.0, component="policy_manager")
        ensure_positive(self.batch_size, "batch_size", component="policy_manager")
        ensure_positive(self.min_batch_size, "min_batch_size", component="policy_manager")
        ensure_positive(self.update_frequency, "update_frequency", component="policy_manager")
        ensure_in_range(self.entropy_coef, "entropy_coef", minimum=0.0, component="policy_manager")
        ensure_positive(self.advantage_epsilon, "advantage_epsilon", component="policy_manager")
        ensure_in_range(self.memory_bias_blend, "memory_bias_blend", minimum=0.0, maximum=1.0, component="policy_manager")
        ensure_positive(self.policy_temperature, "policy_temperature", component="policy_manager")
        ensure_in_range(self.dropout_rate, "dropout_rate", minimum=0.0, maximum=1.0, component="policy_manager")
        ensure_in_range(self.l1_lambda, "l1_lambda", minimum=0.0, component="policy_manager")
        ensure_in_range(self.l2_lambda, "l2_lambda", minimum=0.0, component="policy_manager")
        ensure_in_range(self.adam_beta1, "adam_beta1", minimum=0.0, maximum=1.0, component="policy_manager")
        ensure_in_range(self.adam_beta2, "adam_beta2", minimum=0.0, maximum=1.0, component="policy_manager")
        ensure_positive(self.adam_epsilon, "adam_epsilon", component="policy_manager")

        if self.grad_clip_norm is not None:
            self.grad_clip_norm = float(self.grad_clip_norm)
            ensure_positive(self.grad_clip_norm, "grad_clip_norm", component="policy_manager")

        if self.optimizer_name not in {"adam", "adamw", "sgd"}:
            raise InvalidConfigurationValueError(
                f"Unsupported optimizer_name: {self.optimizer_name}",
                component="policy_manager",
                details={"optimizer_name": self.optimizer_name},
                remediation="Use one of: adam, adamw, sgd.",
            )

        resolved_device = self.device_name
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved_device)

    def _initialize_runtime_state(self) -> None:
        self.skills: Optional[Dict[int, Any]] = None
        self.skill_ids: List[int] = []
        self.skill_to_index: Dict[int, int] = {}
        self.index_to_skill: Dict[int, int] = {}
        self.num_skills = 0

        self.memory = MultiModalMemory()
        self.active_skill: Optional[int] = None
        self.active_skill_index: Optional[int] = None
        self.skill_start_state: Optional[np.ndarray] = None
        self.skill_history: List[Dict[str, Any]] = []
        self.last_policy_update: Optional[Dict[str, Any]] = None

        self.policy_network: Optional[PolicyNetwork] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.current_goal: Optional[str] = None
        self.current_task_type: Optional[str] = None
        self._steps = 0

    def set_task_goal(self, goal: str) -> None:
        self.current_goal = goal
        logger.info("Task goal set to: %s", goal)

    def set_task_type(self, task_type: str) -> None:
        self.current_task_type = task_type
        logger.info("Task type set to: %s", task_type)

    def _validate_skills(self, skills: Mapping[int, Any]) -> Dict[int, Any]:
        ensure_instance(skills, Mapping, "skills", component="policy_manager")
        if not skills:
            raise EmptyRegistryError(
                "Cannot initialize PolicyManager with an empty skills dictionary.",
                component="policy_manager",
            )

        normalized: Dict[int, Any] = {}
        inferred_state_dim: Optional[int] = self.state_dim

        for raw_skill_id, skill in skills.items():
            if not isinstance(raw_skill_id, (int, np.integer)):
                raise InvalidSkillSpecificationError(
                    "Skill identifiers must be integers.",
                    component="policy_manager",
                    details={"skill_id_type": type(raw_skill_id).__name__},
                )
            skill_id = int(raw_skill_id)

            if isinstance(skill, Mapping):
                if "state_dim" not in skill and inferred_state_dim is None:
                    raise MissingFieldError(
                        "Skill mapping must include 'state_dim' when PolicyManager.state_dim is not configured.",
                        component="policy_manager",
                        details={"skill_id": skill_id},
                    )
                skill_state_dim = int(skill.get("state_dim", inferred_state_dim))
                ensure_positive(skill_state_dim, "skill.state_dim", component="policy_manager")
                normalized[skill_id] = dict(skill)
            else:
                if not hasattr(skill, "state_dim"):
                    raise InvalidSkillSpecificationError(
                        "Skill objects must expose a 'state_dim' attribute.",
                        component="policy_manager",
                        details={"skill_id": skill_id, "skill_type": type(skill).__name__},
                    )
                skill_state_dim = int(getattr(skill, "state_dim"))
                ensure_positive(skill_state_dim, "skill.state_dim", component="policy_manager")
                normalized[skill_id] = skill

            if inferred_state_dim is None:
                inferred_state_dim = skill_state_dim

        if inferred_state_dim is None:
            raise SkillInitializationError(
                "Failed to infer Policy Manager state_dim from the provided skills.",
                component="policy_manager",
            )

        ensure_positive(inferred_state_dim, "state_dim", component="policy_manager")
        self.state_dim = int(inferred_state_dim)
        return normalized

    def _build_policy_network(self) -> None:
        if self.state_dim is None or self.num_skills <= 0:
            raise SkillInitializationError(
                "Policy network cannot be built before skills and state_dim are initialized.",
                component="policy_manager",
                details={"state_dim": self.state_dim, "num_skills": self.num_skills},
            )

        self.policy_network = PolicyNetwork(
            input_dim=self.state_dim,
            output_dim=self.num_skills,
            hidden_sizes=self.hidden_layers,
            hidden_activation=self.activation,
            output_activation=self.output_activation,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            l1_lambda=self.l1_lambda,
            l2_lambda=self.l2_lambda,
            weight_init=self.weight_init,
        ).to(self.device)

        self.optimizer = self._create_optimizer(self.policy_network)

    def _create_optimizer(self, model: PolicyNetwork) -> torch.optim.Optimizer:
        if self.optimizer_name == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        raise InvalidConfigurationValueError(
            f"Unsupported optimizer_name: {self.optimizer_name}",
            component="policy_manager",
            details={"optimizer_name": self.optimizer_name},
        )

    def initialize_skills(self, skills: Mapping[int, Any]) -> None:
        """Initialize the manager with specific skills and rebuild the policy network."""
        normalized = self._validate_skills(skills)

        self.skills = normalized
        self.skill_ids = sorted(normalized.keys())
        self.num_skills = len(self.skill_ids)
        self.skill_to_index = {skill_id: idx for idx, skill_id in enumerate(self.skill_ids)}
        self.index_to_skill = {idx: skill_id for skill_id, idx in self.skill_to_index.items()}

        self._build_policy_network()

        if hasattr(self.memory, "action_dim"):
            self.memory.action_dim = self.num_skills

        logger.info("Policy Manager skills initialized with %s skills", self.num_skills)
        logger.info("Skill Space: %s", self.skill_ids)

    def _require_initialized_policy(self) -> None:
        if self.policy_network is None or self.optimizer is None or self.skills is None:
            raise PolicyNotInitializedError(
                "Policy network is not initialized. Call `initialize_skills()` first.",
                component="policy_manager",
            )

    def _skill_name(self, skill_id: int) -> str:
        if self.skills is None:
            return f"skill_{skill_id}"
        skill = self.skills.get(skill_id)
        if isinstance(skill, Mapping):
            return str(skill.get("name", f"skill_{skill_id}"))
        if hasattr(skill, "name"):
            return str(getattr(skill, "name"))
        return f"skill_{skill_id}"

    def _prepare_state(self, state: Any) -> np.ndarray:
        if self.state_dim is None:
            raise SkillInitializationError(
                "state_dim is not available. Initialize skills before preparing states.",
                component="policy_manager",
            )

        if isinstance(state, str):
            if self.strict_state_dim:
                raise InvalidTypeError(
                    "State must be numeric; string states are not allowed when strict_state_dim is enabled.",
                    component="policy_manager",
                    details={"state": state},
                )
            logger.warning("Invalid string state received: '%s'. Using zero state.", state)
            return np.zeros(self.state_dim, dtype=np.float32)

        try:
            array = np.asarray(state, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            if self.strict_state_dim:
                raise InvalidTypeError(
                    "State must be numeric and convertible to a numpy array.",
                    component="policy_manager",
                    details={"state_type": type(state).__name__},
                    cause=exc,
                ) from exc
            logger.warning("Failed to convert state to a numeric array. Using zero state.")
            return np.zeros(self.state_dim, dtype=np.float32)

        if array.ndim == 0:
            array = array.reshape(1)
        elif array.ndim > 1:
            array = array.reshape(-1)

        if array.shape[0] != self.state_dim:
            if self.strict_state_dim:
                raise DimensionMismatchError(
                    f"State dimension mismatch: expected {self.state_dim}, got {array.shape[0]}",
                    component="policy_manager",
                    details={"expected": self.state_dim, "received": int(array.shape[0])},
                )
            if self.pad_states and array.shape[0] < self.state_dim:
                array = np.pad(array, (0, self.state_dim - array.shape[0]), mode="constant")
            else:
                array = array[: self.state_dim]
                if array.shape[0] < self.state_dim:
                    array = np.pad(array, (0, self.state_dim - array.shape[0]), mode="constant")

        if not np.all(np.isfinite(array)):
            raise InvalidValueError(
                "State contains NaN or infinite values.",
                component="policy_manager",
                details={"state": array.tolist()},
            )

        return array.astype(np.float32, copy=False)

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _build_memory_context(self, state: np.ndarray, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        context_payload: Dict[str, Any] = {
            "state": state.tolist(),
            "type": self.memory_context_type,
            "goal": self.current_goal,
            "task_type": self.current_task_type,
        }
        if context:
            for key, value in context.items():
                if isinstance(value, np.ndarray):
                    context_payload[key] = value.tolist()
                else:
                    context_payload[key] = value
        return context_payload

    def _policy_forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._require_initialized_policy()

        if hasattr(self.policy_network, "forward_logits"):
            logits = self.policy_network.forward_logits(states)
        else:
            probs = self.policy_network(states)
            logits = torch.log(probs.clamp_min(1e-8))

        logits = logits / max(self.policy_temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def _compute_policy_distribution(
        self,
        state: np.ndarray,
        context: Optional[Mapping[str, Any]] = None,
        explore: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self._require_initialized_policy()
        prepared_state = self._prepare_state(state)
        state_tensor = self._state_tensor(prepared_state)
        memory_context = self._build_memory_context(prepared_state, context)
        context_hash = self.memory._generate_context_hash(memory_context)

        with torch.no_grad():
            _, probs_tensor = self._policy_forward(state_tensor)
            raw_probs = probs_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64)

        adjusted_probs = self._normalize_probs(raw_probs.copy())
        retrieved: List[Dict[str, Any]] = []
        memory_bias = np.zeros(self.num_skills, dtype=np.float64)

        use_explore = self.explore if explore is None else bool(explore)
        if self.use_memory_bias:
            retrieved = self.memory.retrieve(query=self.memory_query, context=memory_context, limit=self.batch_size)
            if retrieved:
                memory_bias = self.memory._generate_memory_bias(memories=retrieved)
                adjusted_probs = self._adjust_probs(raw_probs, memory_bias)

        if use_explore:
            selected_index = int(np.random.choice(self.num_skills, p=adjusted_probs))
        else:
            selected_index = int(np.argmax(adjusted_probs))

        skill_id = self.index_to_skill[selected_index]
        skill_name = self._skill_name(skill_id)

        self.active_skill = skill_id
        self.active_skill_index = selected_index
        self.skill_start_state = prepared_state.copy()

        return {
            "skill_id": skill_id,
            "skill_index": selected_index,
            "skill_name": skill_name,
            "probabilities": adjusted_probs.tolist(),
            "raw_probabilities": raw_probs.tolist(),
            "memory_bias": memory_bias.tolist(),
            "context_hash": context_hash,
            "raw_state": prepared_state,
            "retrieved_memories": retrieved,
        }

    def select_skill(self, state: np.ndarray, explore: bool = True) -> int:
        """Select a skill based on the current high-level state."""
        try:
            selection = self._compute_policy_distribution(state, context=None, explore=explore)
            logger.debug(
                "Selected skill %s (%s) with probs: %s",
                selection["skill_id"],
                selection["skill_name"],
                np.round(np.asarray(selection["probabilities"]), 4).tolist(),
            )
            return int(selection["skill_id"])
        except AdaptiveError:
            raise
        except Exception as exc:
            raise SkillSelectionError(
                "Unexpected failure during skill selection.",
                component="policy_manager",
                details={"state_type": type(state).__name__},
                remediation="Inspect the current state payload and policy-manager initialization.",
                cause=exc,
            ) from exc

    def get_action(self, state: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        ensure_instance(context, Mapping, "context", component="policy_manager")
        return self._compute_policy_distribution(state, context=context, explore=self.explore)

    def finalize_skill(self, final_reward: float, success: bool = False, params: Optional[Dict[str, Any]] = None) -> None:
        """Complete skill execution and update manager policy."""
        if self.active_skill is None or self.active_skill_index is None or self.skill_start_state is None:
            logger.warning("Skill finalized with no active skill")
            return

        if not isinstance(final_reward, (int, float, np.number)) or not np.isfinite(final_reward):
            raise InvalidValueError(
                "final_reward must be a finite numeric value.",
                component="policy_manager",
                details={"final_reward": final_reward},
            )

        manager_reward = 1.0 if bool(success) else -0.1
        self.store_experience(
            state=self.skill_start_state,
            action=self.active_skill,
            reward=manager_reward,
            next_state=None,
            done=True,
            context={"success": bool(success), "type": self.memory_context_type},
            params=params,
        )

        self.skill_history.append(
            {
                "timestamp": datetime.now(),
                "skill": int(self.active_skill),
                "skill_index": int(self.active_skill_index),
                "skill_name": self._skill_name(int(self.active_skill)),
                "reward": float(final_reward),
                "manager_reward": float(manager_reward),
                "success": bool(success),
                "steps": int(self._steps),
            }
        )

        self.active_skill = None
        self.active_skill_index = None
        self.skill_start_state = None
        self._steps += 1

        if self.update_frequency > 0 and len(self.skill_history) % self.update_frequency == 0:
            self.update_policy()

    def update_policy(self) -> Dict[str, float]:
        """Update manager policy based on stored experiences."""
        self._require_initialized_policy()

        memory_sizes = self.memory.size() if hasattr(self.memory, "size") else {"episodic": 0}
        episodic_size = int(memory_sizes.get("episodic", 0))
        if episodic_size < self.min_batch_size:
            return {}

        batch_limit = min(self.batch_size, episodic_size)
        experiences = self.memory.retrieve(query=self.memory_query, context=None, limit=batch_limit)

        states: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []
        for item in experiences:
            if item.get("type") != "episodic":
                continue
            data = item.get("data", {})
            if not isinstance(data, Mapping):
                continue
            if "state" not in data or "action" not in data or "reward" not in data:
                continue

            state = self._prepare_state(data["state"])
            action_index = int(data["action"])
            if action_index < 0 or action_index >= self.num_skills:
                continue

            states.append(state)
            actions.append(action_index)
            rewards.append(float(data["reward"]))

        if not states:
            return {}

        states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        advantages_np = self.advantage_calculation(rewards, gamma=self.skill_gamma, normalize=self.normalize_advantages)
        advantages_t = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)

        try:
            self.policy_network.train()
            self.optimizer.zero_grad(set_to_none=True)

            logits, probs = self._policy_forward(states_t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            policy_loss = -(selected_log_probs * advantages_t).mean()
            regularization_penalty = (
                self.policy_network.regularization_penalty()
                if hasattr(self.policy_network, "regularization_penalty")
                else torch.zeros((), device=self.device)
            )
            loss = policy_loss - (self.entropy_coef * entropy) + regularization_penalty

            if not torch.isfinite(loss):
                raise PolicyUpdateError(
                    "Policy loss became non-finite.",
                    component="policy_manager",
                    details={"loss": float(loss.detach().cpu().item())},
                )

            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            metrics = {
                "loss": float(loss.detach().cpu().item()),
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "entropy": float(entropy.detach().cpu().item()),
                "batch_size": float(len(states)),
                "avg_reward": float(np.mean(rewards)),
                "avg_advantage": float(np.mean(advantages_np)),
            }
            self.last_policy_update = metrics
            logger.info("Manager policy updated | Loss: %.4f | Entropy: %.4f", metrics["loss"], metrics["entropy"])
            return metrics
        except AdaptiveError:
            raise
        except Exception as exc:
            raise PolicyUpdateError(
                "Unexpected failure during manager policy update.",
                component="policy_manager",
                details={"batch_size": len(states)},
                remediation="Inspect memory contents, network outputs, and optimizer configuration.",
                cause=exc,
            ) from exc
        finally:
            self.policy_network.eval()

    def advantage_calculation(
        self,
        rewards: Sequence[Union[int, float]],
        gamma: Optional[float] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """Calculate discounted returns / advantages from a reward sequence."""
        if not rewards:
            raise EmptyCollectionError(
                "rewards must not be empty.",
                component="policy_manager",
            )

        gamma_value = self.skill_gamma if gamma is None else float(gamma)
        normalize_value = self.normalize_advantages if normalize is None else bool(normalize)
        ensure_in_range(gamma_value, "gamma", minimum=0.0, maximum=1.0, component="policy_manager")

        returns: List[float] = []
        running_return = 0.0
        for reward in reversed(rewards):
            reward_value = float(reward)
            running_return = reward_value + gamma_value * running_return
            returns.insert(0, running_return)

        advantages = np.asarray(returns, dtype=np.float32)
        if normalize_value and advantages.size > 1:
            mean = float(advantages.mean())
            std = float(advantages.std())
            advantages = (advantages - mean) / (std + self.advantage_epsilon)
        return advantages

    def _adjust_probs(self, probs: np.ndarray, memory_bias: np.ndarray) -> np.ndarray:
        """Adjust skill probabilities using memory-based bias."""
        probs = np.asarray(probs, dtype=np.float64).flatten()
        memory_bias = np.asarray(memory_bias, dtype=np.float64).flatten()

        if probs.size != self.num_skills:
            raise DimensionMismatchError(
                "Policy probability vector has unexpected dimensionality.",
                component="policy_manager",
                details={"expected": self.num_skills, "received": int(probs.size)},
            )

        if memory_bias.size != probs.size:
            logger.warning("Memory bias dimension mismatch: expected %s, got %s", probs.size, memory_bias.size)
            return self._normalize_probs(probs)

        base_probs = self._normalize_probs(probs)
        if not np.any(np.isfinite(memory_bias)):
            return base_probs

        safe_bias = np.nan_to_num(memory_bias, nan=0.0, posinf=0.0, neginf=0.0)
        adjustment = np.exp(self.memory_bias_blend * safe_bias)
        adjusted = base_probs * adjustment
        return self._normalize_probs(adjusted)

    def _normalize_probs(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64).flatten()
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, None)

        total = float(np.sum(probs))
        if total <= 1e-12:
            uniform = np.ones(self.num_skills, dtype=np.float64) / max(self.num_skills, 1)
            return uniform / uniform.sum()

        normalized = probs / total
        normalized = normalized / max(float(normalized.sum()), 1e-12)
        if normalized.size > 0:
            normalized[-1] = max(0.0, 1.0 - float(np.sum(normalized[:-1])))
            normalized = normalized / max(float(normalized.sum()), 1e-12)
        return normalized.astype(np.float64)

    def store_experience(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any = None,
        done: bool = True,
        context: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store manager-level experience in memory."""
        self._require_initialized_policy()

        if action not in self.skill_to_index:
            raise InvalidValueError(
                "Unknown skill_id passed to store_experience.",
                component="policy_manager",
                details={"action": action, "known_skill_ids": self.skill_ids},
            )
        if not isinstance(reward, (int, float, np.number)) or not np.isfinite(reward):
            raise InvalidValueError(
                "reward must be a finite numeric value.",
                component="policy_manager",
                details={"reward": reward},
            )

        state_array = self._prepare_state(state)
        next_state_array = None if next_state is None else self._prepare_state(next_state)
        action_index = self.skill_to_index[int(action)]
        enriched_context = self._build_memory_context(state_array, context)

        experience = self.memory.store_experience(
            state=state_array,
            action=action_index,
            reward=float(reward),
            next_state=next_state_array,
            done=bool(done),
            context=enriched_context,
            params=params,
            skill_id=int(action),
            skill_name=self._skill_name(int(action)),
            manager_step=int(self._steps),
        )

        log_params = {
            "learning_rate": params.get("learning_rate"),
            "exploration_rate": params.get("exploration_rate"),
            "discount_factor": params.get("discount_factor"),
            "temperature": params.get("temperature"),
        }
        self.memory.log_parameters(float(reward), log_params)

        return experience

    def get_skill_report(self) -> Dict[str, Any]:
        """Generate performance report for skills."""
        if not self.skill_history:
            return {}

        skill_stats: Dict[int, Dict[str, Any]] = {}
        for skill_id in self.skill_ids:
            skill_data = [entry for entry in self.skill_history if entry["skill"] == skill_id]
            if not skill_data:
                continue

            rewards = [float(entry["reward"]) for entry in skill_data]
            successes = [bool(entry["success"]) for entry in skill_data]
            skill_stats[int(skill_id)] = {
                "skill_name": self._skill_name(int(skill_id)),
                "usage_count": len(skill_data),
                "success_rate": float(np.mean(successes)),
                "avg_reward": float(np.mean(rewards)),
                "best_reward": float(np.max(rewards)),
                "last_used": int(max(entry["steps"] for entry in skill_data)),
            }

        recent_window = self.skill_history[-10:]
        return {
            "total_skill_invocations": len(self.skill_history),
            "num_skills": self.num_skills,
            "skill_stats": skill_stats,
            "recent_success_rate": float(np.mean([bool(entry["success"]) for entry in recent_window])),
            "active_skill": self.active_skill,
            "steps": self._steps,
        }

    def get_manager_report(self) -> Dict[str, Any]:
        """Return a unified operational report for policy control and memory status."""
        self._require_initialized_policy()
        return {
            "policy": {
                "state_dim": self.state_dim,
                "num_skills": self.num_skills,
                "skill_ids": self.skill_ids,
                "explore": self.explore,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "update_frequency": self.update_frequency,
                "last_policy_update": self.last_policy_update,
            },
            "task": {
                "goal": self.current_goal,
                "task_type": self.current_task_type,
            },
            "skills": self.get_skill_report(),
            "memory": self.memory.get_memory_report() if hasattr(self.memory, "get_memory_report") else {},
        }

    def consolidate_memory(self) -> None:
        self.memory.consolidate()

    def _skill_snapshot(self) -> Dict[int, Dict[str, Any]]:
        snapshot: Dict[int, Dict[str, Any]] = {}
        if not self.skills:
            return snapshot

        for skill_id, skill in self.skills.items():
            if isinstance(skill, Mapping):
                snapshot[int(skill_id)] = dict(skill)
            else:
                payload = {
                    "name": getattr(skill, "name", f"skill_{skill_id}"),
                    "state_dim": getattr(skill, "state_dim", self.state_dim),
                    "action_dim": getattr(skill, "action_dim", None),
                    "skill_id": getattr(skill, "skill_id", skill_id),
                    "class_name": type(skill).__name__,
                }
                snapshot[int(skill_id)] = payload
        return snapshot

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        self._require_initialized_policy()
        checkpoint_path = Path(path)
        payload = {
            "policy_state_dict": self.policy_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "memory_state": self.memory.export_state() if hasattr(self.memory, "export_state") else {},
            "skill_history": self.skill_history,
            "steps": self._steps,
            "skills_snapshot": self._skill_snapshot(),
            "skill_ids": self.skill_ids,
            "state_dim": self.state_dim,
            "current_goal": self.current_goal,
            "current_task_type": self.current_task_type,
            "last_policy_update": self.last_policy_update,
            "config_snapshot": {
                "hidden_layers": self.hidden_layers,
                "activation": self.activation,
                "output_activation": self.output_activation,
                "explore": self.explore,
            },
        }

        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, checkpoint_path, pickle_protocol=self.checkpoint_protocol)
            logger.info("Policy Manager checkpoint saved to %s", checkpoint_path)
            return checkpoint_path
        except Exception as exc:
            raise CheckpointSaveError(
                f"Failed to save Policy Manager checkpoint to {checkpoint_path}.",
                component="policy_manager",
                details={"filepath": str(checkpoint_path)},
                remediation="Ensure the target path is writable and checkpoint contents are serializable.",
                cause=exc,
            ) from exc

    def load_checkpoint(self, path: Union[str, Path]) -> "PolicyManager":
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise CheckpointLoadError(
                f"Policy Manager checkpoint does not exist: {checkpoint_path}",
                component="policy_manager",
                details={"filepath": str(checkpoint_path)},
            )

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            if self.policy_network is None:
                skills_snapshot = checkpoint.get("skills_snapshot", {})
                if not skills_snapshot:
                    raise SkillInitializationError(
                        "Checkpoint did not contain a skills_snapshot and the current manager is not initialized.",
                        component="policy_manager",
                    )
                self.initialize_skills(skills_snapshot)

            self.policy_network.load_state_dict(checkpoint["policy_state_dict"])
            if self.optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if hasattr(self.memory, "import_state"):
                self.memory.import_state(checkpoint.get("memory_state", {}))

            self.skill_history = list(checkpoint.get("skill_history", []))
            self._steps = int(checkpoint.get("steps", 0))
            self.current_goal = checkpoint.get("current_goal")
            self.current_task_type = checkpoint.get("current_task_type")
            self.last_policy_update = checkpoint.get("last_policy_update")
            logger.info("Policy Manager checkpoint loaded from %s", checkpoint_path)
            return self
        except AdaptiveError:
            raise
        except Exception as exc:
            raise CheckpointLoadError(
                f"Failed to load Policy Manager checkpoint from {checkpoint_path}.",
                component="policy_manager",
                details={"filepath": str(checkpoint_path)},
                remediation="Inspect the checkpoint file for corruption or incompatible schema.",
                cause=exc,
            ) from exc


if __name__ == "__main__":
    print("\n=== Running Policy Manager ===\n")
    printer.status("TEST", "Policy Manager initialized", "info")

    skills = {
        10: {"name": "navigate_to_A", "state_dim": 8, "action_dim": 4},
        20: {"name": "collect_item_B", "state_dim": 8, "action_dim": 3},
        30: {"name": "avoid_obstacles", "state_dim": 8, "action_dim": 2},
    }

    manager = PolicyManager()
    manager.initialize_skills(skills)
    printer.status("TEST", f"Policy Manager initialized with skill_ids={manager.skill_ids}", "success")

    state = np.random.randn(manager.state_dim).astype(np.float32)
    selected_skill = manager.select_skill(state, explore=True)
    printer.status("TEST", f"Selected skill_id={selected_skill}", "success")

    action_payload = manager.get_action(
        state,
        context={"goal": "reach_target", "type": "navigation", "episode": 1},
    )
    printer.status("TEST", f"Action payload generated for skill={action_payload['skill_name']}", "success")

    for step in range(max(manager.update_frequency, manager.min_batch_size) + 2):
        reward = 1.0 if step % 2 == 0 else -0.25
        chosen_skill = manager.skill_ids[step % manager.num_skills]
        manager.store_experience(
            state=np.random.randn(manager.state_dim).astype(np.float32),
            action=chosen_skill,
            reward=reward,
            next_state=np.random.randn(manager.state_dim).astype(np.float32),
            done=(step % 3 == 0),
            context={"episode": step, "type": "manager_training"},
            params={
                "learning_rate": manager.learning_rate,
                "exploration_rate": 0.1,
                "discount_factor": manager.skill_gamma,
                "temperature": 1.0,
            },
        )

    manager.finalize_skill(final_reward=2.5, success=True, params={"learning_rate": manager.learning_rate, "temperature": 1.0})
    update_metrics = manager.update_policy()
    printer.status("TEST", f"Policy update metrics: {update_metrics}", "success")

    report = manager.get_manager_report()
    printer.status("TEST", f"Manager report keys: {list(report.keys())}", "success")

    checkpoint_path = Path("/tmp/policy_manager_test_checkpoint.pt")
    manager.save_checkpoint(checkpoint_path)
    restored = PolicyManager()
    restored.load_checkpoint(checkpoint_path)
    printer.status("TEST", f"Restored manager steps={restored._steps}", "success")

    print("\n=== Test ran successfully ===\n")