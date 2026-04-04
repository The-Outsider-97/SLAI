"""
Production-ready learning factory for coordinating core SLAI learners.

This module initialises and coordinates the project's core learners:
- DQNAgent
- MAMLAgent
- RSIAgent
- RLAgent

It provides:
- robust agent initialisation against interface drift across sub-agents
- task-aware agent selection using metadata, recent performance, and checkpoint quality
- temporary/permanent agent pools with promotion and garbage collection
- evolutionary mutation/crossover utilities for generating new candidate learners
- checkpoint orchestration, architecture monitoring, and historical replay hooks
"""

from __future__ import annotations

import inspect
import os
import random
import time
import hashlib
import numpy as np
import torch

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from src.agents.learning.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.dqn import DQNAgent
from src.agents.learning.maml_rl import MAMLAgent
from src.agents.learning.rsi import RSIAgent
from src.agents.learning.rl_agent import RLAgent
from src.agents.learning.learning_memory import LearningMemory, Transition
from src.agents.learning.utils.neural_network import NeuralNetwork
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Learning Factory")
printer = PrettyPrinter

AgentName = str
TaskMetadata = Dict[str, Any]


class LearningFactory:
    """Coordinate multiple learning agents and route tasks to the best learner.

    The factory keeps permanent base agents for the core learning paradigms and
    can create temporary variants when task metadata suggests that none of the
    existing agents are a sufficiently strong fit.
    """

    def __init__(self, env: Any, performance_metrics: Optional[Dict[str, Any]] = None):
        if env is None or not hasattr(env, "observation_space") or not hasattr(env, "action_space"):
            raise ValueError("LearningFactory requires a valid environment with observation_space and action_space")

        self.env = env
        self.config = load_global_config()
        self.factory_config = get_config_section("evolutionary") or {}
        self.memory_config = get_config_section("learning_memory") or {}
        self.performance_metrics = performance_metrics or {}

        self.state_dim = self._infer_state_dim(env)
        self.action_dim = self._infer_action_dim(env)

        self.mutation_rate = float(self.factory_config.get("mutation_rate", 0.2))
        self.top_k = int(self.factory_config.get("top_k", 2))
        self.population_size = int(self.factory_config.get("population_size", 10))
        self.generations = int(self.factory_config.get("generations", 20))
        self.evaluation_episodes = int(self.factory_config.get("evaluation_episodes", 3))
        self.elite_ratio = float(self.factory_config.get("elite_ratio", 0.3))
        self.creation_threshold = float(self.factory_config.get("creation_threshold", 0.4))
        self.promotion_threshold = int(self.factory_config.get("promotion_threshold", 3))
        self.max_temporary_agents = int(self.factory_config.get("max_temporary_agents", 10))
        self.garbage_collect_interval = int(self.factory_config.get("garbage_collect_interval", 1000))
        self.step_size_adaptation = bool(self.factory_config.get("step_size_adaptation", True))
        self.checkpoint_dir = Path(self.memory_config.get("checkpoint_dir", "src/agents/learning/checkpoints/memory"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.top_k <= 0:
            raise ValueError("evolutionary.top_k must be positive")
        if self.max_temporary_agents <= 0:
            raise ValueError("evolutionary.max_temporary_agents must be positive")
        if self.creation_threshold < 0.0:
            raise ValueError("evolutionary.creation_threshold must be non-negative")
        if self.promotion_threshold <= 0:
            raise ValueError("evolutionary.promotion_threshold must be positive")

        self.param_bounds: Dict[AgentName, Dict[str, Tuple[float, float]]] = {
            "dqn": {
                "hidden_size": (64, 512),
                "learning_rate": (1e-5, 0.1),
                "batch_size": (16, 1024),
                "gamma": (0.85, 0.999),
                "epsilon_decay": (0.90, 0.9999),
            },
            "maml": {
                "meta_lr": (1e-5, 0.01),
                "inner_lr": (1e-4, 0.1),
                "inner_steps": (1, 10),
                "hidden_size": (32, 256),
            },
            "rsi": {
                "learning_rate": (1e-5, 0.1),
                "gamma": (0.80, 0.999),
                "epsilon": (0.01, 1.0),
                "param_mutation_rate": (0.01, 0.5),
                "performance_history": (10, 500),
            },
            "rl": {
                "learning_rate": (1e-4, 0.5),
                "discount_factor": (0.80, 0.999),
                "epsilon": (0.0, 1.0),
                "trace_decay": (0.0, 1.0),
                "epsilon_decay": (0.90, 0.9999),
            },
        }
        self._step_sizes = {agent: 0.10 for agent in self.param_bounds}
        self._mutation_q_values = {agent: 1.0 for agent in self.param_bounds}

        self.learning_memory = LearningMemory()
        self.model_id = "Learning_Factory"
        self.memory: Deque[Dict[str, Any]] = deque(maxlen=10000)
        self.training_steps = 0

        self.permanent_agents: List[str] = ["dqn", "maml", "rsi", "rl"]
        self.temporary_agents: Dict[str, Dict[str, Any]] = {}
        self.task_registry: Dict[str, int] = defaultdict(int)
        self.selection_history: Deque[Dict[str, Any]] = deque(maxlen=500)
        self.performance_tracker: Dict[str, Deque[float]] = {
            "dqn": deque(maxlen=100),
            "maml": deque(maxlen=100),
            "rsi": deque(maxlen=100),
            "rl": deque(maxlen=100),
        }
        self.architecture_snapshot = {
            "hidden_layers": defaultdict(int),
            "activation_functions": Counter(),
            "agent_types": Counter(),
        }

        self._init_component()
        self.agent_pool: Dict[str, Any] = {name: getattr(self, name) for name in self.permanent_agents}
        self.agents: Dict[str, Any] = dict(self.agent_pool)

        logger.info(
            "LearningFactory initialised | state_dim=%s action_dim=%s permanent_agents=%s",
            self.state_dim,
            self.action_dim,
            self.permanent_agents,
        )

    # ------------------------------------------------------------------
    # Initialisation and interface adaptation
    # ------------------------------------------------------------------
    def _init_component(self) -> None:
        self.dqn = self._create_agent_by_type("dqn", {"agent_id": "dqn_agent"})
        self.maml = self._create_agent_by_type("maml", {"agent_id": "maml_agent"})
        self.rsi = self._create_agent_by_type("rsi", {"agent_id": "rsi_agent"})
        self.rl = self._create_agent_by_type("rl", {"agent_id": "rl_agent"})

        for agent_name in self.permanent_agents:
            self._load_agent_checkpoint(agent_name, getattr(self, agent_name))

        logger.info("Sub-agents initialized with state_dim=%s action_dim=%s", self.state_dim, self.action_dim)

    @staticmethod
    def _infer_state_dim(env: Any) -> int:
        observation_space = getattr(env, "observation_space", None)
        if observation_space is None:
            raise ValueError("Environment observation_space is required")

        shape = getattr(observation_space, "shape", None)
        if shape is None:
            if hasattr(observation_space, "n"):
                return int(observation_space.n)
            raise ValueError("Unable to infer state dimension from environment observation_space")

        if len(shape) == 0:
            return 1
        state_dim = int(np.prod(shape))
        if state_dim <= 0:
            raise ValueError("Inferred state dimension must be positive")
        return state_dim

    @staticmethod
    def _infer_action_dim(env: Any) -> int:
        action_space = getattr(env, "action_space", None)
        if action_space is None:
            raise ValueError("Environment action_space is required")

        if hasattr(action_space, "n"):
            action_dim = int(action_space.n)
        else:
            shape = getattr(action_space, "shape", None)
            if shape is None:
                raise ValueError("Unable to infer action dimension from environment action_space")
            action_dim = int(np.prod(shape))

        if action_dim <= 0:
            raise ValueError("Inferred action dimension must be positive")
        return action_dim

    def _task_sampler(self, split: str = "train") -> Tuple[Any, Dict[str, Any]]:
        del split
        return self.env, {}

    def _constructor_accepts(self, cls: Any, param_name: str) -> bool:
        try:
            return param_name in inspect.signature(cls).parameters
        except (TypeError, ValueError):
            return False

    def _build_override_config(self, agent_type: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        params = params or {}
        override: Dict[str, Any] = {}

        if agent_type == "dqn":
            dqn_cfg = {k: v for k, v in params.items() if k not in {"agent_id"}}
            if dqn_cfg:
                override["dqn"] = dqn_cfg
        elif agent_type == "maml":
            maml_cfg = {k: v for k, v in params.items() if k not in {"agent_id"}}
            if maml_cfg:
                override["maml"] = maml_cfg
        elif agent_type == "rsi":
            rsi_cfg = {k: v for k, v in params.items() if k not in {"agent_id"}}
            if rsi_cfg:
                override["rsi"] = rsi_cfg
        elif agent_type == "rl":
            rl_cfg = {k: v for k, v in params.items() if k not in {"agent_id"}}
            if rl_cfg:
                override["rl"] = rl_cfg
        return override

    def _canonical_agent_type(self, agent_type: str) -> str:
        agent_type = str(agent_type).strip().lower()
        if agent_type in self.param_bounds:
            return agent_type
        if agent_type.endswith("_hybrid"):
            base = agent_type.split("_", 1)[0]
            if base in self.param_bounds:
                return base
        if "dqn" in agent_type:
            return "dqn"
        if "maml" in agent_type:
            return "maml"
        if "rsi" in agent_type:
            return "rsi"
        if "rl" in agent_type:
            return "rl"
        logger.warning("Unknown agent type '%s', defaulting to dqn", agent_type)
        return "dqn"

    def _instantiate_agent(self, agent_type: str, params: Optional[Dict[str, Any]] = None) -> Any:
        agent_type = self._canonical_agent_type(agent_type)
        params = dict(params or {})
        agent_id = str(params.pop("agent_id", f"{agent_type}_{int(time.time() * 1000)}"))
        override_config = self._build_override_config(agent_type, params)

        if agent_type == "dqn":
            kwargs: Dict[str, Any] = {"agent_id": agent_id, "state_dim": self.state_dim, "action_dim": self.action_dim}
            if self._constructor_accepts(DQNAgent, "config"):
                kwargs["config"] = override_config
            if self._constructor_accepts(DQNAgent, "device"):
                kwargs["device"] = None
            agent = DQNAgent(**kwargs)
            self._apply_runtime_overrides("dqn", agent, params)
            return agent

        if agent_type == "maml":
            kwargs = {"agent_id": agent_id, "state_size": self.state_dim, "action_size": self.action_dim}
            if self._constructor_accepts(MAMLAgent, "config"):
                kwargs["config"] = override_config
            if self._constructor_accepts(MAMLAgent, "task_sampler"):
                kwargs["task_sampler"] = self._task_sampler
            if self._constructor_accepts(MAMLAgent, "device"):
                kwargs["device"] = None
            agent = MAMLAgent(**kwargs)
            self._apply_runtime_overrides("maml", agent, params)
            return agent

        if agent_type == "rsi":
            kwargs = {"state_size": self.state_dim, "action_size": self.action_dim, "agent_id": agent_id}
            if self._constructor_accepts(RSIAgent, "env"):
                kwargs["env"] = self.env
            agent = RSIAgent(**kwargs)
            self._apply_runtime_overrides("rsi", agent, params)
            return agent

        kwargs = {"agent_id": agent_id, "possible_actions": list(range(self.action_dim)), "state_size": self.state_dim}
        agent = RLAgent(**kwargs)
        self._apply_runtime_overrides("rl", agent, params)
        return agent

    def _rebuild_value_networks(self, agent: Any, learning_rate: Optional[float] = None, hidden_size: Any = None) -> None:
        learning_rate = float(learning_rate if learning_rate is not None else getattr(agent, "learning_rate", getattr(agent, "lr", 0.001)))
        hidden_size = hidden_size if hidden_size is not None else getattr(agent, "hidden_dim", 64)

        if isinstance(hidden_size, int):
            hidden_layers = [int(hidden_size), int(hidden_size)]
        elif isinstance(hidden_size, (list, tuple)):
            hidden_layers = [int(dim) for dim in hidden_size]
        else:
            hidden_layers = [64, 64]

        network_config = {
            "layer_dims": [int(getattr(agent, "state_dim", getattr(agent, "state_size", self.state_dim))), *hidden_layers, int(getattr(agent, "action_dim", getattr(agent, "action_size", self.action_dim)))],
            "hidden_activation": "relu",
            "output_activation": "linear",
            "loss_function": "mse",
            "optimizer": "adam",
            "learning_rate": learning_rate,
        }

        if hasattr(agent, "policy_net") and hasattr(agent, "target_net"):
            device = getattr(agent, "device", None)
            agent.policy_net = NeuralNetwork(
                input_dim=int(getattr(agent, "state_dim", self.state_dim)),
                output_dim=int(getattr(agent, "action_dim", self.action_dim)),
                config=network_config,
                device=device,
            )
            agent.target_net = NeuralNetwork(
                input_dim=int(getattr(agent, "state_dim", self.state_dim)),
                output_dim=int(getattr(agent, "action_dim", self.action_dim)),
                config=network_config,
                device=device,
            )
            if hasattr(agent.target_net, "set_weights") and hasattr(agent.policy_net, "get_weights"):
                agent.target_net.set_weights(agent.policy_net.get_weights())
        elif hasattr(agent, "q_network") and hasattr(agent, "target_network"):
            agent.q_network = NeuralNetwork(
                input_dim=int(getattr(agent, "state_size", self.state_dim)),
                output_dim=int(getattr(agent, "action_size", self.action_dim)),
                config=network_config,
            )
            agent.target_network = NeuralNetwork(
                input_dim=int(getattr(agent, "state_size", self.state_dim)),
                output_dim=int(getattr(agent, "action_size", self.action_dim)),
                config=network_config,
            )
            if hasattr(agent.target_network, "set_weights") and hasattr(agent.q_network, "get_weights"):
                agent.target_network.set_weights(agent.q_network.get_weights())
            agent.policy_net = getattr(agent, "q_network", None)

    def _apply_runtime_overrides(self, agent_type: str, agent: Any, params: Dict[str, Any]) -> None:
        if not params:
            return

        if agent_type == "dqn":
            if "gamma" in params:
                agent.gamma = float(params["gamma"])
            if "learning_rate" in params:
                agent.lr = float(params["learning_rate"])
            if "batch_size" in params:
                agent.batch_size = int(params["batch_size"])
            if "epsilon" in params:
                agent.epsilon = float(params["epsilon"])
            if "epsilon_decay" in params:
                agent.epsilon_decay = float(params["epsilon_decay"])
            if "hidden_size" in params:
                agent.hidden_dim = params["hidden_size"]
                self._rebuild_value_networks(agent, learning_rate=getattr(agent, "lr", None), hidden_size=params["hidden_size"])
        elif agent_type == "maml":
            if "meta_lr" in params:
                agent.meta_lr = float(params["meta_lr"])
                if hasattr(agent, "meta_optimizer"):
                    for group in agent.meta_optimizer.param_groups:
                        group["lr"] = agent.meta_lr
            if "inner_lr" in params:
                agent.inner_lr = float(params["inner_lr"])
            if "inner_steps" in params:
                agent.inner_steps = int(params["inner_steps"])
            if "adaptation_steps" in params:
                agent.inner_steps = int(params["adaptation_steps"])
        elif agent_type == "rsi":
            if "gamma" in params:
                agent.gamma = float(params["gamma"])
            if "learning_rate" in params:
                agent.learning_rate = float(params["learning_rate"])
            if "epsilon" in params:
                agent.epsilon = float(params["epsilon"])
            if "performance_history" in params:
                window = max(1, int(params["performance_history"]))
                current_values = list(getattr(agent, "performance_history", []))[-window:]
                agent.performance_history = deque(current_values, maxlen=window)
            if "param_mutation_rate" in params:
                agent.param_mutation_rate = float(params["param_mutation_rate"])
            if "hidden_size" in params or "learning_rate" in params:
                self._rebuild_value_networks(agent, learning_rate=getattr(agent, "learning_rate", None), hidden_size=params.get("hidden_size", 64))
        elif agent_type == "rl":
            if "learning_rate" in params:
                agent.learning_rate = float(params["learning_rate"])
            if "discount_factor" in params:
                agent.discount_factor = float(params["discount_factor"])
            if "epsilon" in params:
                agent.epsilon = float(params["epsilon"])
            if "trace_decay" in params:
                agent.trace_decay = float(params["trace_decay"])
            if "epsilon_decay" in params:
                agent.epsilon_decay = float(params["epsilon_decay"])

    # ------------------------------------------------------------------
    # Core selection and monitoring
    # ------------------------------------------------------------------
    @property
    def state_history(self) -> List[Any]:
        return self.learning_memory.get_recent_states()

    def _normalise_task_metadata(self, task_metadata: Optional[Dict[str, Any]]) -> TaskMetadata:
        task_metadata = dict(task_metadata or {})
        normalised = {
            "novelty": float(task_metadata.get("novelty", 0.5)),
            "complexity": float(task_metadata.get("complexity", 0.5)),
            "volatility": float(task_metadata.get("volatility", 0.5)),
            "compute_budget": float(task_metadata.get("compute_budget", 0.5)),
            "training_budget": float(task_metadata.get("training_budget", task_metadata.get("compute_budget", 0.5))),
            "stagnation": float(task_metadata.get("stagnation", 0.0)),
            "long_horizon": float(task_metadata.get("long_horizon", 0.5)),
            "preferred_agent": task_metadata.get("preferred_agent"),
            "task_type": str(task_metadata.get("task_type", "generic")),
        }
        for key in ("novelty", "complexity", "volatility", "compute_budget", "training_budget", "stagnation", "long_horizon"):
            normalised[key] = float(np.clip(normalised[key], 0.0, 1.0))
        self.task_registry[normalised["task_type"]] += 1
        return normalised

    def _heuristic_agent_scores(self, task_metadata: TaskMetadata) -> Dict[str, float]:
        novelty = task_metadata["novelty"]
        complexity = task_metadata["complexity"]
        volatility = task_metadata["volatility"]
        compute_budget = task_metadata["compute_budget"]
        training_budget = task_metadata["training_budget"]
        stagnation = task_metadata["stagnation"]
        long_horizon = task_metadata["long_horizon"]
        small_discrete_bonus = 1.0 if self.state_dim <= 128 and self.action_dim <= 64 else 0.0

        scores = {
            "dqn": 0.45 * complexity + 0.20 * compute_budget + 0.20 * training_budget + 0.15 * (1.0 - novelty),
            "maml": 0.50 * novelty + 0.20 * compute_budget + 0.15 * volatility + 0.15 * (1.0 - stagnation),
            "rsi": 0.40 * volatility + 0.25 * stagnation + 0.20 * long_horizon + 0.15 * training_budget,
            "rl": 0.35 * (1.0 - complexity) + 0.25 * (1.0 - compute_budget) + 0.20 * small_discrete_bonus + 0.20 * (1.0 - novelty),
        }

        preferred = task_metadata.get("preferred_agent")
        if preferred in scores:
            scores[preferred] += 0.25
        return {k: float(np.clip(v, 0.0, 1.25)) for k, v in scores.items()}

    def _recent_performance_scores(self) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for agent_name, history in self.performance_tracker.items():
            if not history:
                scores[agent_name] = 0.0
                continue
            recent_mean = float(np.mean(list(history)[-10:]))
            scores[agent_name] = float(np.tanh(recent_mean / 100.0))
        return scores

    def record_performance(self, agent_name: str, reward: float) -> None:
        if agent_name not in self.performance_tracker:
            self.performance_tracker[agent_name] = deque(maxlen=100)
        self.performance_tracker[agent_name].append(float(reward))

    def update_performance_metrics(self, agent_name: str, metrics: Dict[str, Any]) -> None:
        self.performance_metrics[agent_name] = metrics
        reward_candidates = [
            metrics.get("avg_reward"),
            metrics.get("baseline_performance"),
            metrics.get("adapted_performance"),
            metrics.get("global_avg_reward"),
        ]
        reward_candidates = [value for value in reward_candidates if isinstance(value, (int, float))]
        if reward_candidates:
            self.record_performance(agent_name, float(reward_candidates[0]))

    def select_agent(self, task_metadata: Optional[Dict[str, Any]]) -> Any:
        task = self._normalise_task_metadata(task_metadata)
        recent_performance = self._recent_performance_scores()
        checkpoint_scores = {name: self._get_checkpoint_quality(name) for name in self.permanent_agents}
        heuristic_scores = self._heuristic_agent_scores(task)

        combined_scores = {}
        for agent_name in self.permanent_agents:
            combined_scores[agent_name] = (
                0.45 * heuristic_scores.get(agent_name, 0.0)
                + 0.35 * recent_performance.get(agent_name, 0.0)
                + 0.20 * checkpoint_scores.get(agent_name, 0.0)
            )

        selected_name = max(combined_scores, key=combined_scores.get)
        selection_record = {
            "timestamp": time.time(),
            "task": task,
            "scores": combined_scores,
            "selected": selected_name,
        }
        self.selection_history.append(selection_record)
        self.memory.append(selection_record)
        self.learning_memory.set("last_selection", selection_record)

        if max(combined_scores.values()) < self.creation_threshold:
            selected_agent = self._create_agent(task)
        else:
            selected_agent = getattr(self, selected_name)

        self.training_steps += 1
        if self.training_steps % max(1, self.garbage_collect_interval) == 0:
            self._garbage_collect_temporary_agents()
        return selected_agent

    def _checkpoint_candidates(self, agent_name: str) -> List[Path]:
        names = [
            self.checkpoint_dir / f"{agent_name}_checkpoint.pt",
            self.checkpoint_dir / f"{agent_name}.pt",
            self.checkpoint_dir / f"{agent_name}_checkpoint.pkl",
            self.checkpoint_dir / f"{agent_name}.pkl",
        ]
        return names

    def _load_agent_checkpoint(self, agent_name: str, agent: Any) -> bool:
        loaders = [
            getattr(agent, "load_checkpoint", None),
            getattr(agent, "load", None),
        ]
        loaders = [loader for loader in loaders if callable(loader)]
        if not loaders:
            return False

        for path in self._checkpoint_candidates(agent_name):
            if not path.exists():
                continue
            for loader in loaders:
                try:
                    loader(str(path))
                    logger.info("Loaded checkpoint for %s from %s", agent_name, path)
                    return True
                except Exception as exc:
                    logger.warning("Failed loading checkpoint for %s from %s: %s", agent_name, path, exc)
        return False

    def _save_agent_checkpoint(self, agent_name: str, agent: Any) -> Optional[str]:
        save_path = self.checkpoint_dir / f"{agent_name}_checkpoint.pt"
        savers = [
            getattr(agent, "save_checkpoint", None),
            getattr(agent, "save", None),
        ]
        savers = [saver for saver in savers if callable(saver)]
        if not savers:
            return None

        for saver in savers:
            try:
                result = saver(str(save_path))
                if isinstance(result, dict) and not result.get("success", True):
                    continue
                return str(save_path)
            except Exception as exc:
                logger.warning("Failed saving checkpoint for %s via %s: %s", agent_name, saver.__name__, exc)
        return None

    def _get_checkpoint_quality(self, agent_name: str) -> float:
        candidate_paths = [path for path in self._checkpoint_candidates(agent_name) if path.exists()]
        if not candidate_paths:
            return 0.0

        checkpoint_path = max(candidate_paths, key=lambda p: p.stat().st_mtime)
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
            except Exception:
                return 0.05

        score = 0.0
        if isinstance(checkpoint, dict):
            metrics = [
                checkpoint.get("validation_accuracy"),
                checkpoint.get("success_rate"),
                checkpoint.get("best_validation_reward"),
                checkpoint.get("avg_reward"),
                checkpoint.get("avg_return"),
                checkpoint.get("overall_average_reward"),
            ]
            numeric_metrics = [float(x) for x in metrics if isinstance(x, (int, float))]
            if numeric_metrics:
                score += float(np.mean([np.tanh(metric / 100.0) if abs(metric) > 1.0 else np.clip(metric, -1.0, 1.0) for metric in numeric_metrics]))
            if checkpoint.get("policy_net") is not None or checkpoint.get("q_network") is not None:
                score += 0.20
            if checkpoint.get("target_net") is not None or checkpoint.get("target_network") is not None:
                score += 0.10
            if checkpoint.get("episodes_completed") or checkpoint.get("episode_count") or checkpoint.get("total_episodes"):
                score += 0.10
            if checkpoint.get("version") is not None:
                score += 0.05
        else:
            score += 0.05

        age_hours = max((time.time() - checkpoint_path.stat().st_mtime) / 3600.0, 0.0)
        freshness_bonus = 0.10 * np.exp(-age_hours / 72.0)
        score += float(freshness_bonus)
        return float(np.clip(score, 0.0, 1.0))

    def _classify_task(self, task_metadata: Optional[Dict[str, Any]]) -> TaskMetadata:
        return self._normalise_task_metadata(task_metadata)

    def monitor_architecture(self) -> Dict[str, Any]:
        snapshot = {
            "hidden_layers": defaultdict(int),
            "activation_functions": Counter(),
            "agent_types": Counter(),
        }

        for agent_name in list(self.permanent_agents) + list(self.temporary_agents.keys()):
            agent = self.agents.get(agent_name) if agent_name in self.agents else self.temporary_agents[agent_name]["agent"]
            snapshot["agent_types"][type(agent).__name__] += 1
            network = getattr(agent, "policy_net", None)
            if network is None:
                network = getattr(agent, "q_network", None)
            if network is None:
                network = getattr(agent, "policy", None)
            if network is None:
                continue

            if hasattr(network, "num_layers") and hasattr(network, "layer_dims"):
                for i in range(max(int(network.num_layers) - 1, 0)):
                    layer_type = f"Linear({network.layer_dims[i]}→{network.layer_dims[i+1]})"
                    snapshot["hidden_layers"][layer_type] += 1
                hidden_acts = getattr(network, "hidden_activations", [])
                for act in hidden_acts:
                    snapshot["activation_functions"][act.__class__.__name__] += 1
                output_activation = getattr(network, "output_activation", None)
                if output_activation is not None:
                    snapshot["activation_functions"][output_activation.__class__.__name__] += 1
            elif hasattr(network, "hidden_sizes"):
                dims = [getattr(network, "input_dim", self.state_dim), *list(getattr(network, "hidden_sizes", [])), getattr(network, "output_dim", self.action_dim)]
                for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
                    snapshot["hidden_layers"][f"Linear({in_dim}→{out_dim})"] += 1
                hidden_name = getattr(network, "hidden_activation_name", None)
                output_name = getattr(network, "output_activation_name", None)
                if hidden_name:
                    snapshot["activation_functions"][str(hidden_name)] += 1
                if output_name:
                    snapshot["activation_functions"][str(output_name)] += 1

        self.architecture_snapshot = snapshot
        return {
            "hidden_layers": dict(snapshot["hidden_layers"]),
            "activation_functions": dict(snapshot["activation_functions"]),
            "agent_types": dict(snapshot["agent_types"]),
        }

    def save_checkpoints(self) -> Dict[str, Optional[str]]:
        saved_paths: Dict[str, Optional[str]] = {}
        for agent_name in self.permanent_agents:
            saved_paths[agent_name] = self._save_agent_checkpoint(agent_name, getattr(self, agent_name))
        self.learning_memory.set("learning_factory_saved_paths", saved_paths)
        return saved_paths

    # ------------------------------------------------------------------
    # Evolutionary search helpers
    # ------------------------------------------------------------------
    def generate_new_strategies(self) -> List[Any]:
        optimized_agents: List[Any] = []
        metric_scores = self._extract_agent_scores(self.performance_metrics)
        if not metric_scores:
            metric_scores = {name: float(np.mean(history)) if history else 0.0 for name, history in self.performance_tracker.items()}

        if not metric_scores:
            logger.warning("No usable performance metrics for evolutionary selection.")
            return optimized_agents

        sorted_agents = sorted(metric_scores.items(), key=lambda item: item[1], reverse=True)[: max(1, self.top_k)]

        for agent_id, _score in sorted_agents:
            for _ in range(2):
                mutated_params = self._mutate_parameters(agent_id)
                candidate = self._create_agent_by_type(agent_id, {**mutated_params, "agent_id": f"{agent_id}_mutant_{len(optimized_agents)}"})
                optimized_agents.append(candidate)

        if len(sorted_agents) >= 2:
            hybrid_params = self._crossover(sorted_agents[0][0], sorted_agents[1][0])
            candidate = self._create_agent_by_type(sorted_agents[0][0], {**hybrid_params, "agent_id": f"hybrid_{sorted_agents[0][0]}_{sorted_agents[1][0]}"})
            optimized_agents.append(candidate)

        return optimized_agents

    def _extract_agent_scores(self, metrics_source: Any) -> Dict[str, float]:
        agent_names = ["dqn", "maml", "rsi", "rl"]
        scores: Dict[str, float] = {}

        def _composite_score(agent: str, metric_dict: Dict[str, Any]) -> float:
            reward = float(metric_dict.get(f"{agent}_reward", metric_dict.get("avg_reward", 0.0)) or 0.0)
            length = float(metric_dict.get(f"{agent}_length", metric_dict.get("avg_episode_length", 0.0)) or 0.0)
            stability = float(metric_dict.get(f"{agent}_stability", metric_dict.get("std_reward", 0.0)) or 0.0)
            success = float(metric_dict.get("success_rate", 0.0) or 0.0)
            length_score = 1.0 / (1.0 + max(length, 0.0))
            stability_score = 1.0 / (1.0 + max(stability, 0.0))
            return float(reward * (0.70 + 0.10 * length_score + 0.10 * stability_score) + 0.10 * success)

        if hasattr(metrics_source, "items"):
            for agent in agent_names:
                if agent not in metrics_source:
                    continue
                raw = metrics_source.get(agent)
                if isinstance(raw, (int, float)):
                    scores[agent] = float(raw)
                elif isinstance(raw, (list, tuple, deque, np.ndarray)) and len(raw) > 0:
                    scores[agent] = float(np.mean(list(raw)[-10:]))
                elif isinstance(raw, dict):
                    scores[agent] = _composite_score(agent, raw)
            return scores

        if hasattr(metrics_source, "get_metrics_summary"):
            try:
                summary = metrics_source.get_metrics_summary()
            except Exception as exc:
                logger.warning("Failed to read metric summary: %s", exc)
                return {}

            timings = summary.get("timings_avg_s", {}) if isinstance(summary, dict) else {}
            for agent in agent_names:
                reward_key = f"{agent}_reward"
                if reward_key in timings:
                    scores[agent] = float(timings[reward_key])
                    continue
                train_key = f"{agent}_train"
                if train_key in timings:
                    scores[agent] = 1.0 / (float(timings[train_key]) + 1e-9)
            return scores

        return scores

    def _get_numeric_base_value(self, agent_id: str, param: str, min_val: float, max_val: float) -> float:
        base_cfg = self._get_base_config(agent_id)
        if isinstance(base_cfg, dict) and param in base_cfg:
            value = base_cfg[param]
            if isinstance(value, (int, float)):
                return float(value)
        return (min_val + max_val) / 2.0

    def _mutate_parameters(self, agent_id: str) -> Dict[str, Any]:
        agent_id = self._canonical_agent_type(agent_id)
        params: Dict[str, Any] = {}
        step_size = self._step_sizes.get(agent_id, 0.1)
        q_value = self._mutation_q_values.get(agent_id, 1.0)
        tau = 1.0 / np.sqrt(max(len(self.param_bounds[agent_id]), 1))

        for param, (min_val, max_val) in self.param_bounds[agent_id].items():
            base_val = self._get_numeric_base_value(agent_id, param, min_val, max_val)
            u1 = np.clip(np.random.uniform(1e-8, 1.0), 1e-8, 1.0)
            u2 = np.random.uniform(0.0, 1.0)
            if abs(q_value - 1.0) < 1e-8:
                noise = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            else:
                beta = 1.0 / (3.0 - q_value) if q_value < 3.0 else 0.5
                noise = np.sqrt(max(-beta * np.log(1.0 - u1), 0.0)) * np.cos(2.0 * np.pi * u2)

            param_range = max_val - min_val
            mutated = float(np.clip(base_val + step_size * param_range * noise, min_val, max_val))
            if param in {"hidden_size", "batch_size", "performance_history", "inner_steps"}:
                params[param] = int(round(mutated))
            else:
                params[param] = mutated

            q_value = float(np.clip(q_value * np.exp((1.0 / np.sqrt(max(len(self.param_bounds[agent_id]), 1))) * np.random.randn()), 0.9, 2.5))

        if self.step_size_adaptation:
            self._step_sizes[agent_id] = float(np.clip(step_size * np.exp(tau * np.random.randn()), 1e-6, 1.0))
        self._mutation_q_values[agent_id] = q_value
        return params

    def _determine_agent_architecture(self, task_signature: Optional[Dict[str, Any]]) -> str:
        task = self._normalise_task_metadata(task_signature)
        novelty = task["novelty"]
        complexity = task["complexity"]
        volatility = task["volatility"]
        compute_budget = task["compute_budget"]

        if novelty > 0.8 and compute_budget > 0.7:
            return "maml_hybrid"
        if volatility > 0.7 and complexity > 0.6:
            return "rsi_hybrid"
        if complexity < 0.4 and compute_budget < 0.6:
            return "rl_hybrid"
        if complexity > 0.7 and compute_budget > 0.6:
            return "dqn_hybrid"
        if complexity > 0.7 and volatility > 0.6:
            return "dqn_rsi_hybrid"
        if novelty > 0.7 and volatility > 0.6:
            return "maml_rsi_hybrid"
        if complexity < 0.3 and novelty > 0.6:
            return "rl_maml_hybrid"
        if novelty > 0.6:
            return "maml_hybrid"
        if volatility > 0.6:
            return "rsi_hybrid"
        return "dqn_hybrid"

    def _create_agent_by_type(self, agent_type: str, params: Optional[Dict[str, Any]] = None) -> Any:
        agent_type = self._canonical_agent_type(agent_type)
        return self._instantiate_agent(agent_type, params=params)

    def _task_signature_hash(self, task_signature: Optional[Dict[str, Any]]) -> str:
        task = self._normalise_task_metadata(task_signature)
        fingerprint = repr(sorted(task.items())).encode("utf-8")
        return hashlib.sha1(fingerprint).hexdigest()[:16]

    def _create_agent(self, task_signature: Optional[Dict[str, Any]], evolved_params: Optional[Dict[str, Any]] = None) -> Any:
        signature_hash = self._task_signature_hash(task_signature)
        if signature_hash in self.temporary_agents:
            temp_data = self.temporary_agents[signature_hash]
            temp_data["use_count"] += 1
            temp_data["last_used"] = time.time()
            if temp_data["use_count"] >= self.promotion_threshold:
                self._promote_agent(signature_hash, temp_data["agent"])
            return temp_data["agent"]

        agent_type = self._determine_agent_architecture(task_signature)
        params = dict(evolved_params or {})
        params.setdefault("agent_id", f"temp_{agent_type}_{signature_hash}")
        new_agent = self._create_agent_by_type(agent_type, params)

        if len(self.temporary_agents) >= self.max_temporary_agents:
            self._garbage_collect_temporary_agents(force=True)

        self.temporary_agents[signature_hash] = {
            "agent": new_agent,
            "agent_type": agent_type,
            "use_count": 1,
            "created_at": time.time(),
            "last_used": time.time(),
            "task_signature": self._normalise_task_metadata(task_signature),
        }
        return new_agent

    def _evolve_new_agent(self, agent_type: str, task_signature: Optional[Dict[str, Any]], evolved_params: Optional[Dict[str, Any]] = None) -> Any:
        params = dict(evolved_params or self._get_base_config(agent_type) or {})
        params.setdefault("agent_id", f"temp_{self._canonical_agent_type(agent_type)}_{self._task_signature_hash(task_signature)}")
        return self._create_agent_by_type(agent_type, params)

    def _get_base_config(self, agent_type: str) -> Dict[str, Any]:
        base_agent = self._canonical_agent_type(agent_type)
        return get_config_section(base_agent) or {}

    def _promote_agent(self, agent_hash: str, agent: Any) -> str:
        temp_meta = self.temporary_agents.pop(agent_hash, {"agent_type": type(agent).__name__.lower()})
        base_type = self._canonical_agent_type(temp_meta.get("agent_type", type(agent).__name__.lower()))
        agent_name = f"perm_{base_type}_{agent_hash}"

        setattr(self, agent_name, agent)
        self.permanent_agents.append(agent_name)
        self.agent_pool[agent_name] = agent
        self.agents[agent_name] = agent
        self.performance_tracker[agent_name] = deque(maxlen=100)

        agent_config = self._safe_agent_config(agent)
        self.learning_memory.set(f"promoted_agents/{agent_name}", {
            "agent_type": base_type,
            "config": agent_config,
            "promoted_at": time.time(),
        })
        logger.info("Promoted temporary agent %s to permanent pool as %s", agent_hash, agent_name)
        return agent_name

    def _safe_agent_config(self, agent: Any) -> Dict[str, Any]:
        config_attr_names = ["config", "dqn_config", "maml_config", "rsi_config"]
        extracted: Dict[str, Any] = {"agent_type": type(agent).__name__}
        for name in config_attr_names:
            value = getattr(agent, name, None)
            if isinstance(value, dict):
                extracted[name] = dict(value)
        for field in ("state_dim", "action_dim", "state_size", "action_size", "agent_id"):
            if hasattr(agent, field):
                extracted[field] = getattr(agent, field)
        return extracted

    def _crossover(self, agent_id1: str, agent_id2: str) -> Dict[str, Any]:
        agent_id1 = self._canonical_agent_type(agent_id1)
        agent_id2 = self._canonical_agent_type(agent_id2)
        if agent_id1 not in self.param_bounds or agent_id2 not in self.param_bounds:
            logger.warning("Invalid agent IDs for crossover. Returning empty configuration.")
            return {}

        common_params = set(self.param_bounds[agent_id1]) & set(self.param_bounds[agent_id2])
        hybrid_params: Dict[str, Any] = {}
        for param in common_params:
            base1 = self._get_numeric_base_value(agent_id1, param, *self.param_bounds[agent_id1][param])
            base2 = self._get_numeric_base_value(agent_id2, param, *self.param_bounds[agent_id2][param])
            if all(isinstance(v, (int, float)) for v in (base1, base2)):
                mixed = 0.5 * base1 + 0.5 * base2 if random.random() < 0.5 else random.choice([base1, base2])
                min_val = max(self.param_bounds[agent_id1][param][0], self.param_bounds[agent_id2][param][0])
                max_val = min(self.param_bounds[agent_id1][param][1], self.param_bounds[agent_id2][param][1])
                mixed = float(np.clip(mixed, min_val, max_val))
                if param in {"hidden_size", "batch_size", "performance_history", "inner_steps"}:
                    hybrid_params[param] = int(round(mixed))
                else:
                    hybrid_params[param] = mixed
        return hybrid_params

    @staticmethod
    def compute_intrinsic_reward(state: Any, action: Any, next_state: Any) -> float:
        try:
            state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
            next_arr = np.asarray(next_state, dtype=np.float32).reshape(-1)
            transition_novelty = float(np.linalg.norm(next_arr - state_arr))
        except Exception:
            transition_novelty = 0.0

        if isinstance(action, (list, tuple, np.ndarray)):
            action_bonus = float(np.linalg.norm(np.asarray(action, dtype=np.float32).reshape(-1)))
        else:
            action_bonus = float(abs(action)) if isinstance(action, (int, float, np.integer, np.floating)) else 0.0
        return transition_novelty + 0.01 * action_bonus

    def _detect_new_data(self) -> bool:
        return bool(self.learning_memory.get("new_data_flag", False))

    def get_similar_states(self, state_embedding: Any, k: int = 5) -> List[Dict[str, Any]]:
        if k <= 0:
            raise ValueError("k must be positive")
        recent_states = self.learning_memory.get_recent_states(max(k * 10, 20))
        if not recent_states:
            return []

        query = np.asarray(state_embedding, dtype=np.float32).reshape(-1)
        similarities: List[Tuple[float, Any]] = []
        for state in recent_states:
            try:
                candidate = np.asarray(state, dtype=np.float32).reshape(-1)
                if candidate.shape != query.shape:
                    min_dim = min(candidate.size, query.size)
                    candidate = candidate[:min_dim]
                    local_query = query[:min_dim]
                else:
                    local_query = query
                denom = np.linalg.norm(local_query) * np.linalg.norm(candidate)
                cosine = float(np.dot(local_query, candidate) / denom) if denom > 0 else 0.0
                distance = float(np.linalg.norm(local_query - candidate))
                score = cosine - 0.01 * distance
                similarities.append((score, state))
            except Exception:
                continue

        similarities.sort(key=lambda item: item[0], reverse=True)
        return [
            {"score": float(score), "state": state}
            for score, state in similarities[:k]
        ]

    def _replay_transition_for_rl(self, agent: Any, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        if hasattr(agent, "_process_state") and hasattr(agent, "_get_q_value"):
            processed_state = agent._process_state(state)
            processed_next_state = agent._process_state(next_state)
            current_q = agent._get_q_value(processed_state, action)
            next_best = 0.0 if done else max(agent._get_q_value(processed_next_state, a) for a in agent.possible_actions)
            td_error = float(reward) + float(getattr(agent, "discount_factor", 0.9)) * next_best - current_q
            if hasattr(agent, "learning_memory"):
                agent.learning_memory.add(
                    Transition(processed_state, action, float(reward), processed_next_state, bool(done)),
                    priority=abs(td_error) + 1e-6,
                    tag=getattr(agent, "agent_id", "rl"),
                )
            if hasattr(agent, "_update_eligibility"):
                agent._update_eligibility(processed_state, action)
            traces = getattr(agent, "eligibility_traces", {})
            for key, eligibility in list(traces.items()):
                agent.q_table[key] = agent.q_table.get(key, getattr(agent, "default_q_value", 0.0)) + agent.learning_rate * td_error * eligibility
            if hasattr(agent, "_decay_eligibility"):
                agent._decay_eligibility()
            if hasattr(agent, "state_action_counts"):
                agent.state_action_counts[(processed_state, action)] = agent.state_action_counts.get((processed_state, action), 0) + 1
            if done and hasattr(agent, "end_episode"):
                agent.end_episode(processed_next_state, done=True)

    def _agent_replay_size(self, agent: Any) -> int:
        if hasattr(agent, "replay_size") and callable(agent.replay_size):
            try:
                return int(agent.replay_size())
            except Exception:
                pass
        memory = getattr(agent, "memory", None)
        if memory is None:
            return 0
        if hasattr(memory, "size") and callable(memory.size):
            return int(memory.size())
        try:
            return int(len(memory))
        except Exception:
            return 0

    def _replay_historical_data(self) -> Dict[str, Any]:
        historical_data = self.learning_memory.get("historical_episodes")
        if historical_data is None:
            historical_data = self.learning_memory.get_by_tag("historical_episodes") or []
        if not historical_data:
            return {"replayed_episodes": 0, "updated_agents": []}

        if isinstance(historical_data, dict):
            historical_episodes = [historical_data]
        else:
            historical_episodes = list(historical_data)

        replay_strategy = "prioritized" if len(historical_episodes) > 100 else "uniform"
        if replay_strategy == "prioritized":
            replay_data = sorted(
                historical_episodes,
                key=lambda item: item.get("timestamp", 0.0) if isinstance(item, dict) else 0.0,
                reverse=True,
            )[:100]
        else:
            replay_data = random.sample(historical_episodes, min(len(historical_episodes), 100))

        updated_agents = set()
        for episode in replay_data:
            if not isinstance(episode, dict):
                continue
            states = episode.get("states", [])
            actions = episode.get("actions", [])
            rewards = episode.get("rewards", [])
            dones = episode.get("dones", [])
            if len(states) < 2 or not actions or not rewards:
                continue

            next_states = episode.get("next_states")
            if next_states is None:
                next_states = list(states[1:]) + [states[-1]]
            if not dones:
                dones = [False] * (len(actions) - 1) + [True]

            transition_count = min(len(actions), len(rewards), len(states) - 1, len(next_states), len(dones))
            if transition_count <= 0:
                continue

            for agent_id, agent in self.agents.items():
                if agent_id.startswith("dqn") and hasattr(agent, "store_transition"):
                    for i in range(transition_count):
                        agent.store_transition(states[i], actions[i], rewards[i], next_states[i], bool(dones[i]))
                    replay_size = self._agent_replay_size(agent)
                    batch_size = int(getattr(agent, "batch_size", 1))
                    if replay_size >= batch_size and hasattr(agent, "train"):
                        try:
                            agent.train()
                        except TypeError:
                            pass
                    updated_agents.add(agent_id)
                elif agent_id.startswith("rl"):
                    for i in range(transition_count):
                        self._replay_transition_for_rl(agent, states[i], actions[i], rewards[i], next_states[i], bool(dones[i]))
                    updated_agents.add(agent_id)
                elif agent_id.startswith("rsi") and hasattr(agent, "remember"):
                    for i in range(transition_count):
                        agent.remember(states[i], actions[i], rewards[i], next_states[i], bool(dones[i]))
                    if hasattr(agent, "train_episode"):
                        try:
                            agent.train_episode()
                        except TypeError:
                            pass
                    updated_agents.add(agent_id)

        if len(historical_episodes) > 1000:
            self.learning_memory.set("historical_episodes", historical_episodes[-1000:])

        return {
            "replayed_episodes": len(replay_data),
            "updated_agents": sorted(updated_agents),
            "strategy": replay_strategy,
        }

    def _garbage_collect_temporary_agents(self, force: bool = False) -> int:
        if not self.temporary_agents:
            return 0
        if not force and len(self.temporary_agents) <= self.max_temporary_agents:
            return 0

        removable = sorted(
            self.temporary_agents.items(),
            key=lambda item: (item[1].get("use_count", 0), item[1].get("last_used", 0.0)),
        )
        remove_count = max(0, len(self.temporary_agents) - self.max_temporary_agents + (1 if force else 0))
        removed = 0
        for agent_hash, _meta in removable[:remove_count]:
            del self.temporary_agents[agent_hash]
            removed += 1
        if removed:
            logger.info("Garbage-collected %s temporary agents", removed)
        return removed


if __name__ == "__main__":
    print("\n=== Running Learning Factory ===\n")
    from src.agents.learning.slaienv import SLAIEnv

    env = SLAIEnv()
    factory = LearningFactory(env, performance_metrics={})

    print(factory.monitor_architecture())
    task_signature = {
        "novelty": 0.85,
        "complexity": 0.75,
        "volatility": 0.80,
        "compute_budget": 0.90,
    }
    selected_agent = factory.select_agent(task_signature)
    print(f"Selected agent: {type(selected_agent).__name__}")
    print(factory.generate_new_strategies())
    print(factory._replay_historical_data())
    print("\n=== Learning Factory smoke test complete ===\n")
