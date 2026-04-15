__version__ = "2.0.0"

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

import inspect
import pickle
import random
import numpy as np
import torch

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Callable
from collections import defaultdict, deque

from .base_agent import BaseAgent
from .base.utils.main_config_loader import load_global_config, get_config_section
from .adaptive import PolicyManager, LearningParameterTuner, ImitationLearningWorker, MetaLearningWorker, SkillWorker
from .adaptive.utils.adaptive_errors import *
from .learning.slaienv import SLAIEnv
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Adaptive Agent")
printer = PrettyPrinter

@dataclass
class EpisodeSummary:
    """Structured per-episode execution summary for observability and persistence."""

    episode: int
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    selected_skills: List[int]
    unique_skills: int
    success: bool
    task_type: Optional[str]
    goal: Any
    env_metrics: Dict[str, Any]
    tuner_params: Dict[str, Any]
    skill_metrics: Dict[str, Any]
    policy_metrics: Dict[str, Any]
    meta_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdaptiveAgent(BaseAgent):
    """
    Adaptive agent that coordinates the upgraded adaptive subsystem.

    Responsibilities
    ----------------
    - orchestrate hierarchical task execution through the PolicyManager
    - manage skill workers, imitation learning, meta learning, and parameter tuning
    - maintain operational state in shared memory for warm restart and recovery
    - learn from task execution, demonstrations, and explicit feedback
    - provide route selection, recovery strategies, reporting, and checkpoint support

    Production-oriented extensions
    ------------------------------
    - config-driven initialization via agents_config.yaml
    - structured validation and exception handling integrated with adaptive_errors
    - robust subsystem wiring against the upgraded adaptive module interfaces
    - shared-memory backed state, recovery, routing, and reporting
    - compatibility with both older and upgraded subsystem method variants
    - end-to-end test block covering initialization, task execution, learning, routing, and recovery
    """

    STATE_VERSION = "2.0.0"
    DEFAULT_RECOVERY_TTL = int(timedelta(days=7).total_seconds())
    DEFAULT_STATE_TTL = int(timedelta(days=30).total_seconds())
    RESERVED_GLOBAL_SKILL_ID = 1_000_000

    def __init__(
        self,
        shared_memory: Any,
        agent_factory: Any,
        config: Optional[Mapping[str, Any]] = None,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)

        self.global_config = load_global_config()
        self.agent_config = dict(get_config_section("adaptive_agent") or {})
        if config is not None:
            ensure_instance(config, Mapping, "config", component="adaptive_agent")
            self.agent_config.update(dict(config))

        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self._load_config()

        self.current_state: Optional[np.ndarray] = None
        self.current_task: Dict[str, Any] = {}
        self.current_task_context: Dict[str, Any] = {}
        self.episode = 0
        self.total_steps = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.last_reward = 0.0
        self.last_episode_summary: Optional[Dict[str, Any]] = None
        self.last_route_decision: Optional[Dict[str, Any]] = None
        self.last_recovery_strategy: Optional[str] = None
        self.last_checkpoint_path: Optional[str] = None
        self.task_history: deque = deque(maxlen=self.max_task_history)
        self.route_history: deque = deque(maxlen=self.max_route_history)
        self.feedback_history: deque = deque(maxlen=self.max_feedback_history)
        self.recent_episode_returns: Deque[float] = deque(maxlen=self.performance_window)
        self.recent_episode_lengths: Deque[int] = deque(maxlen=self.performance_window)
        self.success_history: Deque[int] = deque(maxlen=self.performance_window)

        self._initialize_subsystems()
        self._initialize_runtime_state()
        self._warm_start_if_available()
        self._load_agent_state()
        self._load_recovery_history()

        self.recovery_strategies = [
            self._recover_soft_reset,
            self._recover_lr_adjustment,
            self._recover_full_reset,
        ]

        logger.info(
            "Adaptive Agent initialized | state_dim=%s num_actions=%s skills=%s max_episode_steps=%s",
            self.state_dim,
            self.num_actions,
            len(self.skills),
            self.max_episode_steps,
        )

    # ------------------------------------------------------------------
    # Initialization and configuration
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        try:
            self.state_dim = int(self.agent_config.get("state_dim", 10))
            self.num_actions = int(self.agent_config.get("num_actions", 2))
            self.num_handlers = int(self.agent_config.get("num_handlers", 3))
            self.max_episode_steps = int(self.agent_config.get("max_episode_steps", 100))
            self.base_learning_interval = int(self.agent_config.get("base_learning_interval", 10))
            self.skill_max_steps = int(self.agent_config.get("skill_max_steps", 10))
            self.skill_batch_size = int(self.agent_config.get("skill_batch_size", 32))
            self.manager_batch_size = int(self.agent_config.get("manager_batch_size", 32))
            self.tuning_window = int(self.agent_config.get("tuning_window", 100))
            self.performance_window = int(self.agent_config.get("performance_window", 50))
            self.max_task_history = int(self.agent_config.get("max_task_history", 200))
            self.max_route_history = int(self.agent_config.get("max_route_history", 200))
            self.max_feedback_history = int(self.agent_config.get("max_feedback_history", 200))
            self.random_seed = self.agent_config.get("random_seed", 42)

            self.explore_skills = bool(self.agent_config.get("explore_skills", True))
            self.explore_actions = bool(self.agent_config.get("explore_actions", True))
            self.learn_after_task = bool(self.agent_config.get("learn_after_task", True))
            self.auto_behavior_cloning = bool(self.agent_config.get("auto_behavior_cloning", True))
            self.auto_dagger_update = bool(self.agent_config.get("auto_dagger_update", True))
            self.auto_meta_optimize = bool(self.agent_config.get("auto_meta_optimize", True))
            self.auto_save_state = bool(self.agent_config.get("auto_save_state", True))
            self.auto_log_interventions = bool(self.agent_config.get("auto_log_interventions", True))
            self.sync_tuner_to_skills = bool(self.agent_config.get("sync_tuner_to_skills", True))
            self.use_global_rl_engine = bool(self.agent_config.get("use_global_rl_engine", True))
            self.share_meta_registry = bool(self.agent_config.get("share_meta_registry", True))

            self.recovery_reward_threshold = float(self.agent_config.get("recovery_reward_threshold", -10.0))
            self.success_reward_threshold = float(self.agent_config.get("success_reward_threshold", 0.0))
            self.correction_bonus = float(self.agent_config.get("correction_bonus", 1.0))
            self.demonstration_bonus = float(self.agent_config.get("demonstration_bonus", 2.0))
            self.feedback_bonus = float(self.agent_config.get("feedback_bonus", 0.5))
            self.route_similarity_threshold = float(self.agent_config.get("route_similarity_threshold", 0.25))
            self.shared_memory_state_ttl = int(self.agent_config.get("shared_memory_state_ttl", self.DEFAULT_STATE_TTL))
            self.shared_memory_recovery_ttl = int(self.agent_config.get("shared_memory_recovery_ttl", self.DEFAULT_RECOVERY_TTL))
            self.shared_memory_report_ttl = int(self.agent_config.get("shared_memory_report_ttl", self.DEFAULT_STATE_TTL))
            self.global_rl_skill_id = int(self.agent_config.get("global_rl_skill_id", self.RESERVED_GLOBAL_SKILL_ID))
            self.default_task_type = self.agent_config.get("default_task_type", "generic")
            self.task_embedding_strategy = str(self.agent_config.get("task_embedding_strategy", "hash_bow")).lower()
            self.report_detail_level = str(self.agent_config.get("report_detail_level", "full")).lower()
            self.skills_config = dict(self.agent_config.get("skills", {}))
            self.handlers_config = dict(self.agent_config.get("handlers", {}))
            self.env_config = dict(self.agent_config.get("env", {}))
            self.checkpoint_protocol = int(self.agent_config.get("checkpoint_protocol", pickle.HIGHEST_PROTOCOL))
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationValueError(
                "Failed to parse adaptive_agent configuration values.",
                component="adaptive_agent",
                details={"section": "adaptive_agent"},
                remediation="Ensure adaptive_agent values in agents_config.yaml are valid scalars and mappings.",
                cause=exc,
            ) from exc

        ensure_positive(self.state_dim, "state_dim", component="adaptive_agent")
        ensure_positive(self.num_actions, "num_actions", component="adaptive_agent")
        ensure_positive(self.num_handlers, "num_handlers", component="adaptive_agent")
        ensure_positive(self.max_episode_steps, "max_episode_steps", component="adaptive_agent")
        ensure_positive(self.base_learning_interval, "base_learning_interval", component="adaptive_agent")
        ensure_positive(self.skill_max_steps, "skill_max_steps", component="adaptive_agent")
        ensure_positive(self.skill_batch_size, "skill_batch_size", component="adaptive_agent")
        ensure_positive(self.manager_batch_size, "manager_batch_size", component="adaptive_agent")
        ensure_positive(self.tuning_window, "tuning_window", component="adaptive_agent")
        ensure_positive(self.performance_window, "performance_window", component="adaptive_agent")
        ensure_positive(self.max_task_history, "max_task_history", component="adaptive_agent")
        ensure_positive(self.max_route_history, "max_route_history", component="adaptive_agent")
        ensure_positive(self.max_feedback_history, "max_feedback_history", component="adaptive_agent")
        ensure_in_range(self.route_similarity_threshold, "route_similarity_threshold", minimum=0.0, maximum=1.0, component="adaptive_agent")
        ensure_positive(self.shared_memory_state_ttl, "shared_memory_state_ttl", component="adaptive_agent")
        ensure_positive(self.shared_memory_recovery_ttl, "shared_memory_recovery_ttl", component="adaptive_agent")
        ensure_positive(self.shared_memory_report_ttl, "shared_memory_report_ttl", component="adaptive_agent")
        ensure_non_empty(self.default_task_type, "default_task_type", component="adaptive_agent")
        ensure_instance(self.skills_config, Mapping, "skills", component="adaptive_agent")
        ensure_instance(self.handlers_config, Mapping, "handlers", component="adaptive_agent")
        ensure_instance(self.env_config, Mapping, "env", component="adaptive_agent")

        if self.random_seed is not None:
            if not isinstance(self.random_seed, (int, np.integer)):
                raise InvalidConfigurationValueError(
                    "random_seed must be an integer or null.",
                    component="adaptive_agent",
                    details={"random_seed": self.random_seed},
                )
            self._set_seed(int(self.random_seed))

        if self.report_detail_level not in {"compact", "standard", "full"}:
            raise InvalidConfigurationValueError(
                f"Unsupported report_detail_level: {self.report_detail_level}",
                component="adaptive_agent",
                details={"report_detail_level": self.report_detail_level},
                remediation="Use compact, standard, or full.",
            )

        if self.task_embedding_strategy not in {"hash_bow", "random_projection"}:
            raise InvalidConfigurationValueError(
                f"Unsupported task_embedding_strategy: {self.task_embedding_strategy}",
                component="adaptive_agent",
                details={"task_embedding_strategy": self.task_embedding_strategy},
                remediation="Use hash_bow or random_projection.",
            )

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _initialize_subsystems(self) -> None:
        self.skills = self._initialize_skills()
        if not self.skills and not self.use_global_rl_engine:
            raise EmptyRegistryError(
                "AdaptiveAgent requires at least one configured skill when use_global_rl_engine is disabled.",
                component="adaptive_agent",
                remediation=(
                    "Either define adaptive_agent.skills in agents_config.yaml or set "
                    "adaptive_agent.use_global_rl_engine=true to register the global RL engine as a managed skill."
                ),
            )

        self.rl_engine = self._build_global_rl_engine()
        if self.use_global_rl_engine:
            self.skills[self.global_rl_skill_id] = self.rl_engine
            logger.info(
                "Registered global RL engine as managed skill %s (configured skills=%s)",
                self.global_rl_skill_id,
                max(0, len(self.skills) - 1),
            )
        self.policy_manager = PolicyManager()
        self.policy_manager.initialize_skills(self.skills)

        shared_memory_for_learning = getattr(self.policy_manager, "memory", None)
        self.tuner = LearningParameterTuner(memory=shared_memory_for_learning)

        if self.share_meta_registry:
            self.meta_learning = MetaLearningWorker(skill_worker_registry=self.skills)
        else:
            self.meta_learning = MetaLearningWorker()

        imitation_policy = self._resolve_imitation_policy_network()
        self.imitation_worker = ImitationLearningWorker(
            action_dim=self.num_actions,
            state_dim=self.state_dim,
            policy_network=imitation_policy,
            memory=getattr(self.rl_engine, "local_memory", None),
        )

        self._connect_imitation_learning()
        self._connect_meta_learning()

        self.env = SLAIEnv(
            state_dim=self.state_dim,
            action_dim=self.num_actions,
            max_steps=self.max_episode_steps,
            config=self.env_config if self.env_config else None,
        )

    def _initialize_runtime_state(self) -> None:
        self.performance_metrics = defaultdict(lambda: deque(maxlen=self.performance_window))
        self.recovery_history = defaultdict(lambda: {"success": 0, "fail": 0})

        # Episode-level rolling observability buffers
        self.recent_episode_returns = deque(
            getattr(self, "recent_episode_returns", []),
            maxlen=self.performance_window,
        )
        self.recent_episode_lengths = deque(
            getattr(self, "recent_episode_lengths", []),
            maxlen=self.performance_window,
        )
        self.success_history = deque(
            getattr(self, "success_history", []),
            maxlen=self.performance_window,
        )

        self.current_goal = None
        self.current_task_type = None

    def _initialize_skills(self) -> Dict[int, SkillWorker]:
        skills: Dict[int, SkillWorker] = {}
        for skill_id_raw, skill_meta in self.skills_config.items():
            skill_idx = int(skill_id_raw)
            ensure_instance(skill_meta, Mapping, f"skills[{skill_id_raw}]", component="adaptive_agent")
            merged_meta = dict(skill_meta)
            merged_meta.setdefault("name", f"skill_{skill_idx}")
            merged_meta.setdefault("state_dim", self.state_dim)
            merged_meta.setdefault("action_dim", self.num_actions)
            worker = SkillWorker.create_worker(skill_idx, merged_meta)
            skills[skill_idx] = worker
            logger.debug("Initialized skill %s with metadata=%s", skill_idx, merged_meta)
        return skills

    def _create_fallback_skill(self) -> Dict[int, SkillWorker]:
        fallback_skill_id = self._allocate_valid_skill_id(preferred=1, minimum=1)
        fallback = SkillWorker.create_worker(
            fallback_skill_id,
            {
                "name": "fallback_navigation",
                "state_dim": self.state_dim,
                "action_dim": self.num_actions,
            },
        )
        return {fallback_skill_id: fallback}

    def _build_global_rl_engine(self) -> SkillWorker:
        engine_skill_id = self._allocate_valid_skill_id(
            preferred=self.global_rl_skill_id,
            minimum=1,
            include_global_engine=False,
        )

        if hasattr(SkillWorker, "unregister_worker"):
            try:
                if getattr(self, "global_rl_skill_id", None) is not None:
                    SkillWorker.unregister_worker(int(self.global_rl_skill_id))
            except (TypeError, ValueError):
                pass

        self.global_rl_skill_id = int(engine_skill_id)

        return SkillWorker.create_worker(
            skill_id=self.global_rl_skill_id,
            skill_metadata={
                "name": "global_rl_engine",
                "state_dim": self.state_dim,
                "action_dim": self.num_actions,
            },
        )

    def _resolve_imitation_policy_network(self) -> torch.nn.Module:
        if self.use_global_rl_engine and getattr(self.rl_engine, "actor_critic", None) is not None:
            actor_critic = self.rl_engine.actor_critic
            if hasattr(actor_critic, "actor"):
                return actor_critic.actor
            return actor_critic

        first_skill = next(iter(self.skills.values()))
        ensure_not_none(getattr(first_skill, "actor_critic", None), "first_skill.actor_critic", component="adaptive_agent")
        actor_critic = first_skill.actor_critic
        return actor_critic.actor if hasattr(actor_critic, "actor") else actor_critic

    def _connect_imitation_learning(self) -> None:
        for worker in self.skills.values():
            if hasattr(worker, "attach_imitation_learning"):
                worker.attach_imitation_learning(self.imitation_worker)
        if hasattr(self.rl_engine, "attach_imitation_learning"):
            self.rl_engine.attach_imitation_learning(self.imitation_worker)

    def _connect_meta_learning(self) -> None:
        if hasattr(self.meta_learning, "register_skill_workers"):
            self.meta_learning.register_skill_workers(self.skills)
        else:
            self.meta_learning.skill_worker_registry = dict(self.skills)

        for worker in self.skills.values():
            if hasattr(worker, "attach_meta_learning"):
                worker.attach_meta_learning(self.meta_learning)

    def _iter_unique_workers(self) -> Iterable[SkillWorker]:
        seen_workers: set[int] = set()
        for worker in list(self.skills.values()) + [getattr(self, "rl_engine", None)]:
            if worker is None:
                continue
            worker_key = id(worker)
            if worker_key in seen_workers:
                continue
            seen_workers.add(worker_key)
            yield worker

    # ------------------------------------------------------------------
    # Shared-memory backed state and recovery persistence
    # ------------------------------------------------------------------
    @property
    def _state_key(self) -> str:
        return f"agent_state:{self.name}"

    @property
    def _recovery_key(self) -> str:
        return f"recovery_history:{self.name}"

    @property
    def _report_key(self) -> str:
        return f"agent_report:{self.name}"

    @property
    def _warm_state_key(self) -> str:
        return f"warm_state:{self.name}"

    def _load_agent_state(self) -> None:
        cached_state = self.shared_memory.get(self._state_key)
        if not isinstance(cached_state, Mapping):
            self.current_state = None
            self.episode = 0
            self.total_steps = 0
            self.episode_reward = 0.0
            self.episode_length = 0
            self.last_reward = 0.0
            return

        self.current_state = None if cached_state.get("current_state") is None else self._validate_state(cached_state.get("current_state"))
        self.episode = int(cached_state.get("episode", 0))
        self.total_steps = int(cached_state.get("total_steps", 0))
        self.episode_reward = float(cached_state.get("episode_reward", 0.0))
        self.episode_length = int(cached_state.get("episode_length", 0))
        self.last_reward = float(cached_state.get("last_reward", 0.0))
        self.current_goal = cached_state.get("current_goal")
        self.current_task_type = cached_state.get("current_task_type")
        self.last_episode_summary = cached_state.get("last_episode_summary")
        logger.info("Loaded adaptive agent state from shared memory")

    def _save_agent_state(self) -> Dict[str, Any]:
        state_data = {
            "version": self.STATE_VERSION,
            "episode": int(self.episode),
            "total_steps": int(self.total_steps),
            "episode_reward": float(self.episode_reward),
            "episode_length": int(self.episode_length),
            "last_reward": float(self.last_reward),
            "current_state": None if self.current_state is None else np.asarray(self.current_state, dtype=np.float32).tolist(),
            "current_goal": self.current_goal,
            "current_task_type": self.current_task_type,
            "last_episode_summary": self.last_episode_summary,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.shared_memory.set(self._state_key, state_data, ttl=self.shared_memory_state_ttl)
        self.shared_memory.set(self._warm_state_key, state_data, ttl=self.shared_memory_state_ttl)
        return state_data

    def _load_recovery_history(self) -> None:
        history_data = self.shared_memory.get(self._recovery_key)
        self.recovery_history = defaultdict(lambda: {"success": 0, "fail": 0})
        if not isinstance(history_data, Mapping):
            return
        for strategy_name, counts in history_data.items():
            if isinstance(counts, Mapping):
                self.recovery_history[str(strategy_name)] = {
                    "success": int(counts.get("success", 0)),
                    "fail": int(counts.get("fail", 0)),
                }
        logger.info("Loaded recovery history from shared memory")

    def _save_recovery_history(self) -> Dict[str, Dict[str, int]]:
        history_data = {key: {"success": int(value["success"]), "fail": int(value["fail"])} for key, value in self.recovery_history.items()}
        self.shared_memory.set(self._recovery_key, history_data, ttl=self.shared_memory_recovery_ttl)
        return history_data

    def _publish_report(self, report: Mapping[str, Any]) -> None:
        payload = dict(report)
        payload["timestamp"] = datetime.utcnow().isoformat()
        self.shared_memory.set(self._report_key, payload, ttl=self.shared_memory_report_ttl)
        if hasattr(self.shared_memory, "publish"):
            self.shared_memory.publish("adaptive_agent/reports", payload)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def is_initialized(self) -> bool:
        return all(
            [
                self.policy_manager is not None,
                self.tuner is not None,
                self.meta_learning is not None,
                self.imitation_worker is not None,
                self.rl_engine is not None,
                self.env is not None,
                bool(self.skills),
            ]
        )

    def perform_task(self, task_data: Any) -> Any:
        task_payload = self._normalize_task_payload(task_data)
        self.current_task = task_payload
        self.current_task_context = dict(task_payload.get("context", {}))
        self.current_task_type = task_payload.get("type")
        self.current_goal = task_payload.get("goal")

        self._prepare_task_context(task_payload)

        state, info = self.env.reset(seed=self.random_seed if self.random_seed is not None else None)
        state = self._validate_state(state)
        self.current_state = state

        if getattr(self.policy_manager, "state_dim", None) != self.state_dim:
            logger.info("Aligning PolicyManager state_dim from %s to %s", getattr(self.policy_manager, "state_dim", None), self.state_dim)
            self.policy_manager.state_dim = self.state_dim
            self.policy_manager.initialize_skills(self.skills)

        self.episode_reward = 0.0
        self.episode_length = 0
        terminated = False
        truncated = False
        selected_skills: List[int] = []
        per_skill_rewards: Dict[int, float] = defaultdict(float)

        while not (terminated or truncated):
            skill_id = self._select_skill(state)
            skill_worker = self.skills[skill_id]
            selected_skills.append(int(skill_id))

            skill_reward, state, terminated, truncated = self._execute_skill_rollout(
                skill_id=skill_id,
                skill_worker=skill_worker,
                state=state,
                task_payload=task_payload,
            )
            per_skill_rewards[int(skill_id)] += float(skill_reward)

            success = float(skill_reward) >= self.success_reward_threshold
            self._finalize_skill(skill_id=skill_id, state=state, reward=skill_reward, done=terminated or truncated, success=success)

        self._end_episode(task_payload=task_payload)
        summary = self._generate_performance_report(
            task_payload=task_payload,
            terminated=terminated,
            truncated=truncated,
            selected_skills=selected_skills,
            per_skill_rewards=per_skill_rewards,
        )
        self.last_episode_summary = summary
        self.task_history.append(summary)
        self._publish_report(summary)
        return summary

    def _normalize_task_payload(self, task_data: Any) -> Dict[str, Any]:
        if task_data is None:
            return {"type": self.default_task_type, "goal": None, "context": {}, "description": ""}
        if isinstance(task_data, Mapping):
            payload = dict(task_data)
            payload.setdefault("type", self.default_task_type)
            payload.setdefault("goal", None)
            payload.setdefault("context", {})
            payload.setdefault("description", str(payload.get("type", self.default_task_type)))
            ensure_instance(payload["context"], Mapping, "task_payload.context", component="adaptive_agent")
            return payload
        return {
            "type": self.default_task_type,
            "goal": None,
            "context": {},
            "description": str(task_data),
        }

    def _prepare_task_context(self, task_payload: Mapping[str, Any]) -> None:
        context = dict(task_payload.get("context", {}))
        if hasattr(self.env, "set_task_context"):
            try:
                self.env.set_task_context(context)
            except Exception as exc:
                logger.warning("Environment task-context injection failed: %s", exc)

        if hasattr(self.policy_manager, "set_task_goal"):
            try:
                self.policy_manager.set_task_goal(task_payload.get("goal"))
            except Exception as exc:
                logger.warning("PolicyManager set_task_goal failed: %s", exc)

        if hasattr(self.policy_manager, "set_task_type"):
            try:
                self.policy_manager.set_task_type(task_payload.get("type"))
            except Exception as exc:
                logger.warning("PolicyManager set_task_type failed: %s", exc)

        goal_vector = self._goal_to_vector(task_payload.get("goal"), context)
        for worker in self.skills.values():
            if hasattr(worker, "set_goal") and getattr(worker, "enable_goals", False):
                try:
                    worker.set_goal(goal_vector)
                except Exception as exc:
                    logger.warning("Failed to set goal on worker %s: %s", getattr(worker, "skill_id", None), exc)

    def _select_skill(self, state: np.ndarray) -> int:
        try:
            selected = self.policy_manager.select_skill(state, explore=self.explore_skills)
            return int(selected)
        except Exception as exc:
            logger.error("Skill selection failed, falling back to first skill: %s", exc)
            return int(next(iter(self.skills.keys())))

    def _execute_skill_rollout(
        self,
        *,
        skill_id: int,
        skill_worker: SkillWorker,
        state: np.ndarray,
        task_payload: Mapping[str, Any],
    ) -> Tuple[float, np.ndarray, bool, bool]:
        skill_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.skill_max_steps):
            action, log_prob, entropy = self._select_primitive_action(skill_worker, state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            next_state = self._validate_state(next_state)
            normalized_reward = self._coerce_reward(reward)

            self._store_skill_experience(
                skill_worker=skill_worker,
                state=state,
                action=action,
                reward=normalized_reward,
                next_state=next_state,
                done=terminated or truncated,
                log_prob=log_prob,
                entropy=entropy,
            )
            self._store_global_experience(
                state=state,
                action=action,
                reward=normalized_reward,
                next_state=next_state,
                done=terminated or truncated,
                skill_id=skill_id,
            )

            state = next_state
            skill_reward += float(normalized_reward)
            self.episode_reward += float(normalized_reward)
            self.episode_length += 1
            self.total_steps += 1
            self.last_reward = float(normalized_reward)
            self.current_state = state

            if terminated or truncated:
                break

        return float(skill_reward), state, terminated, truncated

    def _select_primitive_action(self, skill_worker: SkillWorker, state: np.ndarray) -> Tuple[Any, float, float]:
        try:
            action, log_prob, entropy = skill_worker.select_action(state, explore=self.explore_actions)
            return action, float(log_prob), float(entropy)
        except Exception as exc:
            logger.error("Primitive action selection failed for skill %s: %s", getattr(skill_worker, "skill_id", None), exc)
            fallback = random.randint(0, self.num_actions - 1)
            return fallback, 0.0, 0.0

    def _finalize_skill(
        self,
        *,
        skill_id: int,
        state: np.ndarray,
        reward: float,
        done: bool,
        success: bool,
    ) -> None:
        params = self.tuner.get_params(include_metadata=False)
        try:
            if hasattr(self.policy_manager, "store_experience"):
                self.policy_manager.store_experience(
                    state=np.asarray(state, dtype=np.float32),
                    action=int(skill_id),
                    reward=float(reward),
                    next_state=np.asarray(state, dtype=np.float32),
                    done=bool(done),
                    context={"episode": self.episode, "type": self.current_task_type or self.default_task_type},
                    params=params,
                )
            self._call_finalize_skill(final_reward=float(reward), success=success, params=params)
        except Exception as exc:
            logger.warning("Skill finalization failed for skill_id=%s: %s", skill_id, exc)

        if hasattr(self.skills[skill_id], "learner_memory") and self.skills[skill_id].learner_memory.size() >= self.skill_batch_size:
            try:
                self.skills[skill_id].update_policy()
            except Exception as exc:
                logger.warning("Skill policy update failed for skill_id=%s: %s", skill_id, exc)

    def _call_finalize_skill(self, *, final_reward: float, success: bool, params: Mapping[str, Any]) -> None:
        finalize = getattr(self.policy_manager, "finalize_skill")
        try:
            signature = inspect.signature(finalize)
            if "params" in signature.parameters:
                finalize(final_reward=final_reward, success=success, params=dict(params))
            else:
                finalize(final_reward=final_reward, success=success)
        except (TypeError, ValueError):
            finalize(final_reward, success)

    def _store_skill_experience(
        self,
        *,
        skill_worker: SkillWorker,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        entropy: float,
    ) -> None:
        skill_worker.store_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            entropy=entropy,
        )

    def _store_global_experience(
        self,
        *,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        skill_id: int,
    ) -> None:
        if not self.use_global_rl_engine:
            return
        self.rl_engine.store_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=0.0,
            entropy=0.0,
        )
        if hasattr(self.rl_engine, "local_memory") and hasattr(self.rl_engine.local_memory, "store_experience"):
            try:
                self.rl_engine.local_memory.store_experience(
                    state=state,
                    action=skill_id,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    context={"source": "adaptive_agent", "task_type": self.current_task_type or self.default_task_type},
                    params=self.tuner.get_params(include_metadata=False),
                )
            except Exception:
                pass

    def _end_episode(self, *, task_payload: Mapping[str, Any]) -> None:
        self.tuner.update_performance(self.episode_reward)
        if self.episode_reward < self.recovery_reward_threshold:
            self.trigger_recovery(reason="reward_threshold", task_payload=task_payload)

        self.tuner.decay_exploration()
        if self.learn_after_task and (self.total_steps % self.base_learning_interval == 0 or self.episode_length > 0):
            self._learn()

        if hasattr(self.rl_engine, "local_memory") and hasattr(self.rl_engine.local_memory, "consolidate"):
            self.rl_engine.local_memory.consolidate()

        self._log_episode_metrics()
        if self.auto_save_state:
            self._save_agent_state()
            self._save_recovery_history()
        self.episode += 1

    # ------------------------------------------------------------------
    # Learning and adaptation
    # ------------------------------------------------------------------

    def _learn(self) -> Dict[str, Any]:
        rewards = self._recent_rewards()
        if rewards:
            self.tuner.adapt(rewards[-self.tuning_window :])

        tuned_params = self.tuner.get_params(include_metadata=False)
        if self.sync_tuner_to_skills:
            self._apply_tuner_params_to_workers(tuned_params)

        skill_updates: Dict[int, Any] = {}
        for skill_id, worker in self.skills.items():
            if hasattr(worker, "learner_memory") and worker.learner_memory.size() >= self.skill_batch_size:
                try:
                    skill_updates[int(skill_id)] = worker.update_policy()
                except Exception as exc:
                    logger.warning("Skill update failed for skill %s during adaptive learning: %s", skill_id, exc)

        manager_update = None
        try:
            manager_update = self.policy_manager.update_policy()
        except Exception as exc:
            logger.warning("PolicyManager update_policy failed during adaptive learning: %s", exc)

        imitation_update = None
        if self.auto_behavior_cloning and hasattr(self.imitation_worker, "get_demonstration_count"):
            demo_counts = self.imitation_worker.get_demonstration_count()
            if demo_counts.get("offline_demos", 0) >= max(1, getattr(self.imitation_worker, "batch_size", 1)):
                try:
                    imitation_update = self.imitation_worker.behavior_cloning(epochs=self.agent_config.get("auto_behavior_cloning_epochs", 1))
                except Exception as exc:
                    logger.warning("Behavior cloning update failed: %s", exc)

        dagger_update = None
        if self.auto_dagger_update and hasattr(self.imitation_worker, "dagger_update"):
            demo_counts: Dict[str, int] = {}
            if hasattr(self.imitation_worker, "get_demonstration_count"):
                try:
                    demo_counts = self.imitation_worker.get_demonstration_count()
                except Exception:
                    demo_counts = {}

            if int(demo_counts.get("total_demos", 0)) > 0:
                try:
                    dagger_update = self.imitation_worker.dagger_update()
                except Exception as exc:
                    logger.warning("DAgger update failed: %s", exc)
            else:
                dagger_update = {"skipped": True, "reason": "no_demonstrations", "loss": None}

        meta_update = None
        if self.auto_meta_optimize and self.episode > 0 and self.episode % max(1, self.agent_config.get("meta_update_frequency", 5)) == 0:
            try:
                meta_update = self.meta_learning.optimization_step()
            except Exception as exc:
                logger.warning("Meta-learning optimization step failed: %s", exc)

        result = {
            "tuned_params": tuned_params,
            "skill_updates": skill_updates,
            "manager_update": manager_update,
            "imitation_update": imitation_update,
            "dagger_update": dagger_update,
            "meta_update": meta_update,
        }
        return result

    def _apply_tuner_params_to_workers(self, tuned_params: Mapping[str, Any]) -> None:
        for worker in self._iter_unique_workers():
            if hasattr(worker, "apply_hyperparameters"):
                try:
                    worker.apply_hyperparameters(tuned_params)
                except Exception as exc:
                    logger.warning("Failed to apply tuned params to worker %s: %s", getattr(worker, "skill_id", None), exc)

    def _recent_rewards(self) -> List[float]:
        rewards: List[float] = []

        episodic = getattr(getattr(self.rl_engine, "local_memory", None), "episodic", None)
        if episodic:
            for exp in episodic:
                try:
                    if isinstance(exp, Mapping):
                        rewards.append(float(exp.get("reward", 0.0)))
                    elif hasattr(exp, "reward"):
                        rewards.append(float(exp.reward))
                except Exception:
                    continue

        recent_episode_returns = getattr(self, "recent_episode_returns", None)
        if recent_episode_returns:
            rewards.extend(float(v) for v in recent_episode_returns)

        return rewards

    def learn_from_demonstration(self, demo_data: Mapping[str, Any]) -> Dict[str, Any]:
        ensure_instance(demo_data, Mapping, "demo_data", component="adaptive_agent")
        if "state" not in demo_data or "action" not in demo_data:
            raise MissingFieldError(
                "Demonstration data must include 'state' and 'action'.",
                component="adaptive_agent",
                details={"demo_data_keys": list(demo_data.keys())},
            )

        state = self._validate_state(demo_data["state"])
        action = demo_data["action"]
        self.imitation_worker.add_demonstration(state, action)

        reward = float(demo_data.get("reward", self.demonstration_bonus))
        self._store_global_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=state,
            done=True,
            skill_id=self.global_rl_skill_id,
        )

        result = {
            "status": "ok",
            "type": "demonstration",
            "reward": reward,
            "demonstrations": self.imitation_worker.get_demonstration_count(),
        }
        self.feedback_history.append(result)
        return result

    def learn_from_feedback(self, feedback: Mapping[str, Any]) -> Dict[str, Any]:
        ensure_instance(feedback, Mapping, "feedback", component="adaptive_agent")
        state = self._validate_state(feedback.get("state", np.zeros(self.state_dim, dtype=np.float32)))
        action = feedback.get("action", 0)
        reward = self._coerce_reward(feedback.get("reward", 0.0))
        feedback_type = str(feedback.get("type", "correction")).lower()

        if feedback_type == "correction":
            reward += self.correction_bonus
        elif feedback_type == "demonstration":
            reward += self.demonstration_bonus
        else:
            reward += self.feedback_bonus

        self._store_global_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=state,
            done=True,
            skill_id=self.global_rl_skill_id,
        )
        if feedback_type == "demonstration":
            self.imitation_worker.add_demonstration(state, action)

        learn_result = self._learn() if self.learn_after_task else {}
        result = {
            "status": "ok",
            "type": feedback_type,
            "reward": reward,
            "learn_result": learn_result,
        }
        self.feedback_history.append(result)
        return result

    # ------------------------------------------------------------------
    # Routing, embeddings, and delegation
    # ------------------------------------------------------------------

    def route_task(self, task_description: str) -> Dict[str, Any]:
        ensure_instance(task_description, str, "task_description", component="adaptive_agent")
        ensure_non_empty(task_description.strip(), "task_description", component="adaptive_agent")

        embedding = self._get_task_embedding(task_description)
        skill_candidates = self._score_skills_for_task(task_description, embedding)
        best_skill_id, best_skill_score = max(skill_candidates.items(), key=lambda item: item[1])

        handler_candidates = self._score_handlers_for_task(task_description, embedding)
        selected_handler = None
        if handler_candidates:
            selected_handler, handler_score = max(handler_candidates.items(), key=lambda item: item[1])
        else:
            handler_score = 0.0

        decision = {
            "task_description": task_description,
            "embedding": embedding.tolist(),
            "selected_skill_id": int(best_skill_id),
            "selected_skill_name": getattr(self.skills[best_skill_id], "name", f"skill_{best_skill_id}"),
            "skill_score": float(best_skill_score),
            "selected_handler": selected_handler,
            "handler_score": float(handler_score),
            "routed": float(best_skill_score) >= self.route_similarity_threshold,
        }
        self.last_route_decision = decision
        self.route_history.append(decision)
        return decision

    def _score_skills_for_task(self, task_description: str, embedding: np.ndarray) -> Dict[int, float]:
        task_tokens = set(self._tokenize(task_description))
        scores: Dict[int, float] = {}
        for skill_id, worker in self.skills.items():
            skill_name = getattr(worker, "name", f"skill_{skill_id}")
            meta = getattr(worker, "skill_metadata", {}) or {}
            description = str(meta.get("description", skill_name))
            skill_tokens = set(self._tokenize(description))
            overlap = len(task_tokens & skill_tokens) / max(1, len(task_tokens | skill_tokens))
            skill_vector = self._get_task_embedding(description)
            cosine = self._cosine_similarity(embedding, skill_vector)
            scores[int(skill_id)] = float(0.6 * overlap + 0.4 * cosine)
        return scores

    def _score_handlers_for_task(self, task_description: str, embedding: np.ndarray) -> Dict[str, float]:
        if not self.handlers_config:
            return {}

        task_tokens = set(self._tokenize(task_description))
        scores: Dict[str, float] = {}
        for handler_name, handler_meta in self.handlers_config.items():
            if not isinstance(handler_meta, Mapping):
                continue
            descriptor = str(handler_meta.get("description", handler_name))
            handler_tokens = set(self._tokenize(descriptor))
            overlap = len(task_tokens & handler_tokens) / max(1, len(task_tokens | handler_tokens))
            handler_vector = self._get_task_embedding(descriptor)
            cosine = self._cosine_similarity(embedding, handler_vector)
            scores[str(handler_name)] = float(0.5 * overlap + 0.5 * cosine)
        return scores

    def _get_task_embedding(self, task_description: str) -> np.ndarray:
        tokens = self._tokenize(task_description)
        vector = np.zeros(self.state_dim, dtype=np.float32)
        if not tokens:
            return vector

        if self.task_embedding_strategy == "hash_bow":
            for token in tokens:
                idx = hash(token) % self.state_dim
                vector[idx] += 1.0
        else:
            rng = np.random.default_rng(abs(hash(task_description)) % (2**32))
            vector = rng.normal(loc=0.0, scale=1.0, size=self.state_dim).astype(np.float32)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32, copy=False)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token.strip().lower() for token in str(text).replace("_", " ").replace("-", " ").split() if token.strip()]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    # ------------------------------------------------------------------
    # Recovery and resilience
    # ------------------------------------------------------------------

    def update_recovery_success(self, strategy_name: str, success: bool = True) -> None:
        ensure_non_empty(strategy_name, "strategy_name", component="adaptive_agent")
        record = self.recovery_history[strategy_name]
        if success:
            record["success"] += 1
        else:
            record["fail"] += 1
        self._save_recovery_history()

    def rank_recovery_strategies(self) -> List[Callable[[], bool]]:
        return sorted(
            self.recovery_strategies,
            key=lambda strategy: self.recovery_history[strategy.__name__]["success"]
            / max(1, self.recovery_history[strategy.__name__]["success"] + self.recovery_history[strategy.__name__]["fail"] + 1),
            reverse=True,
        )

    def trigger_recovery(
        self,
        *,
        reason: str = "manual",
        task_payload: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        logger.warning("AdaptiveAgent recovery triggered | reason=%s", reason)
        strategies = self.rank_recovery_strategies()

        for strategy in strategies:
            strategy_name = strategy.__name__
            try:
                success = bool(strategy())
                self.update_recovery_success(strategy_name, success)
                if success:
                    self.last_recovery_strategy = strategy_name
                    if self.auto_log_interventions and hasattr(self.shared_memory, "log_intervention"):
                        self.shared_memory.log_intervention(
                            report={
                                "agent": self.name,
                                "strategy": strategy_name,
                                "reason": reason,
                                "episode": self.episode,
                                "task_type": None if task_payload is None else task_payload.get("type"),
                            }
                        )
                    return True
            except Exception as exc:
                logger.error("Recovery strategy %s failed: %s", strategy_name, exc)
                self.update_recovery_success(strategy_name, False)
        return False

    def _recover_soft_reset(self) -> bool:
        if hasattr(self.rl_engine, "local_memory") and hasattr(self.rl_engine.local_memory, "clear_episodic"):
            self.rl_engine.local_memory.clear_episodic()
        self.last_reward = 0.0
        self.episode_length = 0
        self.current_state = None
        return True

    def _recover_lr_adjustment(self) -> bool:
        rewards = self._recent_rewards()[-10:]
        if not rewards:
            return False

        avg_reward = float(np.mean(rewards))
        current_params = self.tuner.get_params(include_metadata=False)
        current_lr = float(current_params["learning_rate"])
        new_lr = current_lr * (1.2 if avg_reward < 0.0 else 0.8)
        new_lr = float(np.clip(new_lr, self.tuner._min_learning_rate, self.tuner._max_learning_rate))
        self.tuner.params["learning_rate"] = new_lr
        if self.sync_tuner_to_skills:
            self._apply_tuner_params_to_workers(self.tuner.get_params(include_metadata=False))
        return True

    def _recover_full_reset(self) -> bool:
        try:
            for worker in self._iter_unique_workers():
                if hasattr(worker, "reset"):
                    worker.reset()
            self.tuner.reset()
            self.policy_manager = PolicyManager()
            self.policy_manager.initialize_skills(self.skills)
            self.episode = 0
            self.total_steps = 0
            self.episode_reward = 0.0
            self.episode_length = 0
            self.last_reward = 0.0
            return True
        except Exception as exc:
            logger.error("Full reset failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Reporting, evaluation, and compatibility helpers
    # ------------------------------------------------------------------

    def _generate_performance_report(
        self,
        *,
        task_payload: Mapping[str, Any],
        terminated: bool,
        truncated: bool,
        selected_skills: Sequence[int],
        per_skill_rewards: Mapping[int, float],
    ) -> Dict[str, Any]:
        skill_metrics = self._collect_skill_metrics()
        policy_metrics = self._get_policy_report()
        meta_metrics = self._get_meta_report()
        env_metrics = self.env.get_metrics() if hasattr(self.env, "get_metrics") else {}
        summary = EpisodeSummary(
            episode=int(self.episode),
            total_reward=float(self.episode_reward),
            steps=int(self.episode_length),
            terminated=bool(terminated),
            truncated=bool(truncated),
            selected_skills=[int(v) for v in selected_skills],
            unique_skills=len(set(int(v) for v in selected_skills)),
            success=bool(self.episode_reward >= self.success_reward_threshold),
            task_type=task_payload.get("type"),
            goal=task_payload.get("goal"),
            env_metrics=dict(env_metrics),
            tuner_params=self.tuner.get_params(include_metadata=True),
            skill_metrics=skill_metrics,
            policy_metrics=policy_metrics,
            meta_metrics=meta_metrics,
        ).to_dict()
        summary["per_skill_rewards"] = {int(k): float(v) for k, v in per_skill_rewards.items()}

        if self.report_detail_level == "compact":
            return {
                "episode": summary["episode"],
                "total_reward": summary["total_reward"],
                "steps": summary["steps"],
                "success": summary["success"],
                "task_type": summary["task_type"],
            }
        if self.report_detail_level == "standard":
            return {
                "episode": summary["episode"],
                "total_reward": summary["total_reward"],
                "steps": summary["steps"],
                "success": summary["success"],
                "task_type": summary["task_type"],
                "selected_skills": summary["selected_skills"],
                "policy_metrics": summary["policy_metrics"],
                "skill_metrics": summary["skill_metrics"],
            }
        return summary

    def _log_episode_metrics(self) -> None:
        metrics = {
            "episode": int(self.episode),
            "reward": float(self.episode_reward),
            "length": int(self.episode_length),
            "steps": int(self.total_steps),
            "last_reward": float(self.last_reward),
        }
        self.evaluate_performance(metrics)

        if not hasattr(self, "recent_episode_returns"):
            self.recent_episode_returns = deque(maxlen=self.performance_window)
        if not hasattr(self, "recent_episode_lengths"):
            self.recent_episode_lengths = deque(maxlen=self.performance_window)
        if not hasattr(self, "success_history"):
            self.success_history = deque(maxlen=self.performance_window)

        self.recent_episode_returns.append(float(self.episode_reward))
        self.recent_episode_lengths.append(int(self.episode_length))
        self.success_history.append(1 if self.episode_reward >= self.success_reward_threshold else 0)

    def _collect_skill_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for skill_id, worker in self.skills.items():
            try:
                if hasattr(worker, "get_performance_metrics"):
                    metrics[str(skill_id)] = worker.get_performance_metrics()
            except Exception as exc:
                metrics[str(skill_id)] = {"error": str(exc)}
        return metrics

    def _get_policy_report(self) -> Dict[str, Any]:
        if hasattr(self.policy_manager, "get_manager_report"):
            try:
                return self.policy_manager.get_manager_report()
            except Exception:
                pass
        if hasattr(self.policy_manager, "get_skill_report"):
            try:
                return self.policy_manager.get_skill_report()
            except Exception:
                pass
        return {}

    def _get_meta_report(self) -> Dict[str, Any]:
        if hasattr(self.meta_learning, "get_optimization_report"):
            try:
                return self.meta_learning.get_optimization_report()
            except Exception:
                pass
        return {}

    def _validate_state(self, state: Any) -> np.ndarray:
        if torch.is_tensor(state):
            arr = state.detach().cpu().numpy()
        elif isinstance(state, np.ndarray):
            arr = state
        else:
            try:
                arr = np.asarray(state, dtype=np.float32)
            except Exception:
                logger.warning("Failed to coerce state to numpy array. Using zeros.")
                return np.zeros(self.state_dim, dtype=np.float32)

        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size > self.state_dim:
            return arr[: self.state_dim].astype(np.float32, copy=False)
        if arr.size < self.state_dim:
            return np.pad(arr, (0, self.state_dim - arr.size), mode="constant").astype(np.float32, copy=False)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _coerce_reward(value: Any) -> float:
        try:
            reward = float(value)
        except Exception as exc:
            raise wrap_exception(
                exc,
                InvalidValueError,
                "Reward could not be converted to float.",
                component="adaptive_agent",
                details={"reward": value},
            ) from exc
        if not np.isfinite(reward):
            raise InvalidValueError(
                "Reward must be finite.",
                component="adaptive_agent",
                details={"reward": value},
            )
        return reward

    def _goal_to_vector(self, goal: Any, context: Mapping[str, Any]) -> np.ndarray:
        goal_dim = 0
        for worker in self.skills.values():
            if getattr(worker, "enable_goals", False):
                goal_dim = int(getattr(worker, "goal_dim", 0))
                break
        if goal_dim <= 0:
            return np.zeros(0, dtype=np.float32)

        if goal is None and not context:
            return np.zeros(goal_dim, dtype=np.float32)

        if isinstance(goal, (list, tuple, np.ndarray)):
            arr = np.asarray(goal, dtype=np.float32).reshape(-1)
            if arr.size >= goal_dim:
                return arr[:goal_dim].astype(np.float32, copy=False)
            return np.pad(arr, (0, goal_dim - arr.size), mode="constant").astype(np.float32, copy=False)

        goal_blob = f"{goal}|{dict(context)}"
        embedding = self._get_task_embedding(goal_blob)
        if embedding.size >= goal_dim:
            return embedding[:goal_dim].astype(np.float32, copy=False)
        return np.pad(embedding, (0, goal_dim - embedding.size), mode="constant").astype(np.float32, copy=False)

    def supports_fail_operational(self) -> bool:
        try:
            has_recovery = bool(self.recovery_strategies)
            policy_intact = self.policy_manager is not None and hasattr(self.policy_manager, "select_skill")
            env_intact = self.env is not None and hasattr(self.env, "reset") and hasattr(self.env, "step")
            memory_intact = self.shared_memory is not None and hasattr(self.shared_memory, "get") and hasattr(self.shared_memory, "set")
            return all([has_recovery, policy_intact, env_intact, memory_intact])
        except Exception:
            return False

    def has_redundant_safety_channels(self) -> bool:
        channels = [
            hasattr(self.shared_memory, "log_intervention"),
            hasattr(self.policy_manager, "memory"),
            hasattr(self, "trigger_recovery") and callable(self.trigger_recovery),
        ]
        return sum(1 for channel in channels if channel) >= 2

    def extract_performance_metrics(self, result: Any) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if isinstance(result, Mapping):
            for key in ("total_reward", "steps", "episode", "success"):
                if key in result and isinstance(result[key], (int, float, bool)):
                    metrics[key] = result[key]

        recent_episode_returns = getattr(self, "recent_episode_returns", None)
        success_history = getattr(self, "success_history", None)

        if recent_episode_returns:
            metrics["recent_reward_mean"] = float(np.mean(recent_episode_returns))
        if success_history:
            metrics["recent_success_rate"] = float(np.mean(success_history))

        return metrics

    def alternative_execute(self, task_data: Any, original_error: Optional[BaseException] = None) -> Dict[str, Any]:
        logger.warning("Entering adaptive fallback execution path | error=%s", original_error)
        task_payload = self._normalize_task_payload(task_data)
        route = self.route_task(task_payload.get("description") or task_payload.get("type") or self.default_task_type)
        fallback_skill_id = int(route.get("selected_skill_id", next(iter(self.skills.keys()))))
        fallback_skill = self.skills[fallback_skill_id]

        state, _ = self.env.reset(seed=self.random_seed if self.random_seed is not None else None)
        state = self._validate_state(state)
        action, _, _ = self._select_primitive_action(fallback_skill, state)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state = self._validate_state(next_state)
        reward = self._coerce_reward(reward)
        self._store_skill_experience(
            skill_worker=fallback_skill,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=terminated or truncated,
            log_prob=0.0,
            entropy=0.0,
        )
        return {
            "status": "fallback_executed",
            "selected_skill_id": fallback_skill_id,
            "action": action,
            "reward": reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "route": route,
            "original_error": None if original_error is None else str(original_error),
        }
    
    def _allocate_valid_skill_id(self, *, preferred: Optional[int] = None, minimum: int = 1,
                                 include_global_engine: bool = False) -> int:
        """
        Allocate a strictly positive, non-conflicting skill id.

        SkillWorker enforces skill_id > 0, so adaptive-agent-owned workers
        must never use 0 or negative ids.
        """
        used_ids = set()

        if hasattr(self, "skills") and isinstance(getattr(self, "skills", None), Mapping):
            used_ids.update(int(skill_id) for skill_id in self.skills.keys())

        if include_global_engine and getattr(self, "global_rl_skill_id", None) is not None:
            try:
                used_ids.add(int(self.global_rl_skill_id))
            except (TypeError, ValueError):
                pass

        candidate: Optional[int] = None
        if preferred is not None:
            try:
                candidate = int(preferred)
            except (TypeError, ValueError):
                candidate = None

        if candidate is None or candidate < minimum or candidate in used_ids:
            highest = max(used_ids) if used_ids else (minimum - 1)
            candidate = max(minimum, highest + 1)

        while candidate in used_ids or candidate < minimum:
            candidate += 1

        return int(candidate)

    def save_checkpoint(self, path: str | Path) -> str:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.STATE_VERSION,
            "agent_state": self._save_agent_state(),
            "recovery_history": self._save_recovery_history(),
            "task_history": list(self.task_history),
            "route_history": list(self.route_history),
            "feedback_history": list(self.feedback_history),
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            with checkpoint_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=self.checkpoint_protocol)
        except Exception as exc:
            raise wrap_exception(
                exc,
                CheckpointSaveError,
                "Failed to save AdaptiveAgent checkpoint.",
                component="adaptive_agent",
                details={"path": str(checkpoint_path)},
            ) from exc
        self.last_checkpoint_path = str(checkpoint_path)
        return str(checkpoint_path)

    def load_checkpoint(self, path: str | Path) -> "AdaptiveAgent":
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(
                f"AdaptiveAgent checkpoint not found: {checkpoint_path}",
                component="adaptive_agent",
                details={"path": str(checkpoint_path)},
            )
        try:
            with checkpoint_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            raise wrap_exception(
                exc,
                CheckpointLoadError,
                "Failed to load AdaptiveAgent checkpoint.",
                component="adaptive_agent",
                details={"path": str(checkpoint_path)},
            ) from exc

        ensure_instance(payload, Mapping, "checkpoint_payload", component="adaptive_agent")
        state = payload.get("agent_state", {})
        if isinstance(state, Mapping):
            self.shared_memory.set(self._state_key, dict(state), ttl=self.shared_memory_state_ttl)
            self._load_agent_state()
        history = payload.get("recovery_history", {})
        if isinstance(history, Mapping):
            self.shared_memory.set(self._recovery_key, dict(history), ttl=self.shared_memory_recovery_ttl)
            self._load_recovery_history()
        self.task_history = deque(payload.get("task_history", []), maxlen=self.max_task_history)
        self.route_history = deque(payload.get("route_history", []), maxlen=self.max_route_history)
        self.feedback_history = deque(payload.get("feedback_history", []), maxlen=self.max_feedback_history)
        self.last_checkpoint_path = str(checkpoint_path)
        return self

    def retrain(self) -> Dict[str, Any]:
        learn_result = self._learn()
        return {
            "status": "retrained",
            "episode": self.episode,
            "learn_result": learn_result,
        }

    def log_evaluation_result(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        super().log_evaluation_result(metrics)
        return metrics


if __name__ == "__main__":
    print("\n=== Running Adaptive Agent ===\n")
    printer.status("TEST", "Adaptive Agent initialized", "info")

    class _TestAgentFactory:
        def create(self, module: str, shared_memory: Any = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"module": module, "status": "created"}

    from src.agents.collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    agent_factory = _TestAgentFactory()
    agent = AdaptiveAgent(shared_memory=shared_memory, agent_factory=agent_factory)

    printer.status("TEST", f"Initialized with skills={list(agent.skills.keys())}", "success")
    state_snapshot = agent._save_agent_state()
    recovery_snapshot = agent._save_recovery_history()
    printer.pretty("STATE", state_snapshot, "success")
    printer.pretty("RECOVERY", recovery_snapshot, "success")

    task = {
        "type": "navigation",
        "goal": [1.0, 0.0, 0.0, 0.0],
        "context": {
            "target_position": [5.0, 3.0],
            "obstacles": [[1.0, 1.0], [2.0, 2.0]],
            "start_position": [0.0, 0.0],
        },
        "description": "Navigate safely to the target while avoiding obstacles.",
    }

    report = agent.perform_task(task)
    printer.pretty("TASK_REPORT", report, "success")

    sample_state = np.random.randn(agent.state_dim).astype(np.float32)
    route = agent.route_task("Find the safest route to the target avoiding all obstacles.")
    feedback = agent.learn_from_feedback(
        {
            "state": sample_state,
            "action": 0,
            "reward": 1.0,
            "type": "correction",
        }
    )
    demonstration = agent.learn_from_demonstration(
        {
            "state": sample_state,
            "action": 1,
            "reward": 2.0,
        }
    )
    fallback = agent.alternative_execute(task, original_error=None)
    retrain_result = agent.retrain()

    printer.pretty("ROUTE", route, "success")
    printer.pretty("FEEDBACK", feedback, "success")
    printer.pretty("DEMONSTRATION", demonstration, "success")
    printer.pretty("FALLBACK", fallback, "success")
    printer.pretty("RETRAIN", retrain_result, "success")

    checkpoint_path = Path("/tmp/adaptive_agent_test_checkpoint.pkl")
    saved_path = agent.save_checkpoint(checkpoint_path)
    restored = AdaptiveAgent(shared_memory=shared_memory, agent_factory=agent_factory)
    restored.load_checkpoint(saved_path)
    printer.status("TEST", f"Restored checkpoint from {saved_path}", "success")
    printer.pretty("RESTORED_STATE", restored._save_agent_state(), "success")

    printer.pretty("FAIL_OPERATIONAL", agent.supports_fail_operational(), "success")
    printer.pretty("SAFETY_CHANNELS", agent.has_redundant_safety_channels(), "success")

    print("\n=== Test ran successfully ===\n")
