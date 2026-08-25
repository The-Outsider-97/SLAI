from __future__ import annotations

__version__ = "2.2.1"

"""
SLAI Learning Agent — Orchestration Layer for Lifelong Autonomous Learning
==========================================================================

Responsibilities
----------------
- Coordinate DQN, MAML, RSI, and RL agents through LearningFactory
- Maintain a trainable meta-controller (StrategySelector + PolicyNetwork)
  that routes each episode to the most appropriate learning paradigm
- Run training / evaluation / auto / health modes via perform_task
- Track performance, novelty, uncertainty, and concept drift
- Delegate recovery to RecoverySystem when error thresholds are crossed
- Persist episode traces, metrics, and strategy history to SharedMemory
- Never directly manage src/agents/learning/modules/ (avoids re-init)

Academic References
-------------------
1. DQN & RL  : Mnih et al. (2015). Human-level control via deep RL. Nature.
2. MAML      : Finn et al. (2017). Model-Agnostic Meta-Learning. PMLR.
3. RSI       : Schmidhuber (2013). PowerPlay: Training General Problem Solvers.
4. EWC       : Kirkpatrick et al. (2017). Overcoming Catastrophic Forgetting.
5. Drift     : Gama et al. (2014). Survey on Concept Drift Adaptation.
"""

import hashlib
import time
import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from .base.utils.main_config_loader import get_config_section, load_global_config
from .base_agent import BaseAgent
from .learning.learning_factory import LearningFactory
from .learning.slaienv import SLAIEnv
from .learning.strategy_selector import StrategySelector
from .learning.utils.learning_error import *
from .learning.utils.learning_helpers import *
from .learning.utils.learning_calculations import LearningCalculations
from .learning.modules.multi_task_learner import MultiTaskLearner
from .learning.modules.policy_network import PolicyNetwork, create_policy_optimizer
from .learning.modules.recovery_system import RecoverySystem
from .learning.modules.state_processor import StateProcessor
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Learning Agent")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Internal dataclass — lightweight, avoids importing external dataclass libs
# ---------------------------------------------------------------------------
class _StrategyTrace(dict):
    """Typed alias for strategy-selection audit records (dict subclass for JSON compat)."""


class LearningAgent(BaseAgent):
    """Production orchestration layer for SLAI's lifelong learning subsystem.

    The agent never instantiates learning modules directly; it obtains them
    through LearningFactory and StrategySelector so each module is
    initialized exactly once across the process lifetime.

    Mode dispatch (via ``perform_task``):
    - ``"train"``    — Run N training episodes, trigger recovery if needed.
    - ``"evaluate"`` — Run M evaluation episodes (no gradient updates).
    - ``"auto"``     — Train if retraining interval has elapsed, else evaluate.
    - ``"health"``   — Return a health/diagnostics snapshot.
    """

    DEFAULT_TASK_IDS: Sequence[str] = ("dqn", "maml", "rsi", "rl")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, shared_memory, agent_factory, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self._lock = RLock()

        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.config = load_global_config()
        self.learning_config: Dict[str, Any] = dict(get_config_section("learning_agent") or {})
        if isinstance(config, dict):
            self.learning_config.update(config)
        self._validate_config()

        self.task_ids: List[str] = list(self.learning_config.get("task_ids", self.DEFAULT_TASK_IDS))
        validate_non_empty_sequence(self.task_ids, "task_ids")

        self.strategy_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.task_ids)
        }

        self.batch_size = coerce_int(self.learning_config.get("batch_size", 32), default=32, minimum=1)
        self.max_episode_steps = coerce_int(self.learning_config.get("max_episode_steps", 128), default=128, minimum=1)
        self.max_eval_episodes = coerce_int(self.learning_config.get("max_eval_episodes", 3), default=3, minimum=1)
        self.recovery_trigger_threshold = coerce_int(self.learning_config.get("recovery_trigger_threshold", 3), default=3, minimum=1)
        self.task_embedding_dim = coerce_int(self.learning_config.get("task_embedding_dim", 256), default=256, minimum=1)
        self.maml_task_pool_size = coerce_int(self.learning_config.get("maml_task_pool_size", 100), default=100, minimum=1)
        self.rsi_improvement_cycle = coerce_int(self.learning_config.get("rsi_improvement_cycle", 50), default=50, minimum=1)
        self.performance_threshold = coerce_float(self.learning_config.get("performance_threshold", 0.7), default=0.7)
        self.data_change_threshold = coerce_float(self.learning_config.get("data_change_threshold", 0.15), default=0.15)
        self.novelty_threshold = coerce_float(self.learning_config.get("novelty_threshold", 0.3), default=0.3)
        self.uncertainty_threshold = coerce_float(self.learning_config.get("uncertainty_threshold", 0.25), default=0.25)
        self.retraining_interval = timedelta(
            hours=coerce_int(
                self.learning_config.get("retraining_interval_hours", 24), default=24, minimum=0
            )
        )
        self.heuristic_perf_weight = coerce_float(self.learning_config.get("heuristic_perf_weight", 0.45), default=0.45)
        self.heuristic_exploration_bonus = coerce_float(self.learning_config.get("heuristic_exploration_bonus", 0.20), default=0.20)
        self.heuristic_rsi_bonus = coerce_float(self.learning_config.get("heuristic_rsi_bonus", 0.15), default=0.15)
        self.heuristic_trend_penalty_bonus = coerce_float(self.learning_config.get("heuristic_trend_penalty_bonus", 0.10), default=0.10)
        self.heuristic_preferred_bonus = coerce_float(self.learning_config.get("heuristic_preferred_bonus", 0.20), default=0.20)
        self.trace_history_limit = coerce_int(self.learning_config.get("trace_history_limit", 500), default=500, minimum=1)

        # strategy_weights — one weight per task_id (padded/trimmed if mismatch)
        raw_weights: List[float] = list(self.learning_config.get("strategy_weights", [0.25] * len(self.task_ids))
        )
        self.strategy_weights = np.array(_pad_or_trim(raw_weights, len(self.task_ids), fill=0.25), dtype=np.float32
        )

        # ---- Ring buffers -------------------------------------------------
        self.embedding_buffer: Deque[Tuple[torch.Tensor, int]] = deque(
            maxlen=coerce_int(self.learning_config.get("embedding_buffer_size", 512), default=512, minimum=1)
        )
        self.performance_history: Deque[float] = deque(
            maxlen=coerce_int(self.learning_config.get("performance_history_size", 1000), default=1000, minimum=1)
        )
        self.state_recency: Deque[np.ndarray] = deque(
            maxlen=coerce_int(self.learning_config.get("state_recency_size", 1000), default=1000, minimum=1)
        )
        self.architecture_history: Deque[Dict[str, Any]] = deque(
            maxlen=coerce_int(self.learning_config.get("architecture_history_size", 10), default=10, minimum=1)
        )
        self.error_history: Deque[Dict[str, Any]] = deque(
            maxlen=coerce_int(self.learning_config.get("error_history_size", 100), default=100, minimum=1)
        )

        # ---- Online reward stats (per-strategy) ---------------------------
        self._reward_stats: Dict[str, RunningStats] = {
            task_id: RunningStats() for task_id in self.task_ids
        }
        self._global_reward_stats = RunningStats()

        # ---- Performance metrics dict (passed to LearningFactory) ---------
        # Keys that are lambdas are evaluated lazily when health_report is
        # called; all other keys hold defaultdict/dict containers.
        self.performance_metrics: Dict[str, Any] = {
            "scenario_rewards": defaultdict(float),
            "success_rate": defaultdict(float),
            "episode_length": defaultdict(list),
            "q_value_mean": defaultdict(list),
            "q_value_std": defaultdict(list),
            "strategy_selection_count": defaultdict(int),
            "strategy_accuracy": defaultdict(float),
            "strategy_loss": defaultdict(float),
            "novelty_score": defaultdict(float),
            "uncertainty_estimate": defaultdict(float),
            "catastrophic_forgetting": defaultdict(float),
            "concept_drift_detected": defaultdict(bool),
            "replay_buffer_usage": defaultdict(int),
            "param_mutation_rate": {},
            "checkpoint_quality": defaultdict(float),
            "agent_fitness_score": defaultdict(float),
            # Lazy lambdas — evaluated only when accessed explicitly
            "embedding_buffer_size": lambda: len(self.embedding_buffer),
            "performance_history_stats": lambda: self._global_reward_stats.snapshot().__dict__,
            "plot_tags": [
                "average_reward",
                "success_rate",
                "strategy_selection_count",
                "novelty_score",
            ],
        }

        # ---- Core subsystems ----------------------------------------------
        env = kwargs.get("env") or SLAIEnv()
        self.env: SLAIEnv = env

        self.state_processor = StateProcessor(env=self.env)
        self.learning_calculations = LearningCalculations()
        self.learning_factory = LearningFactory(
            env=self.env,
            performance_metrics=self.performance_metrics,
        )
        self.agents: Dict[str, Any] = self.learning_factory.agents

        self.multi_task_learner = MultiTaskLearner(task_ids=self.task_ids)
        self.strategy_selector = StrategySelector()
        self._initialize_strategy_selector()

        self.recovery_system = RecoverySystem(learning_agent=self)

        # ---- Runtime counters --------------------------------------------
        self.observation_count: int = 0
        self.training_iterations: int = 0
        self.last_training_time: datetime = (
            utc_now() - self.retraining_interval
        )

        self._init_shared_memory_keys()
        logger.info(
            "LearningAgent initialized | task_ids=%s | batch_size=%d | "
            "max_episode_steps=%d | embedding_dim=%d",
            self.task_ids,
            self.batch_size,
            self.max_episode_steps,
            self.task_embedding_dim,
        )
        printer.status("INIT", f"LearningAgent ready — strategies: {self.task_ids}", "success")

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        """Validate all mandatory positive-integer config parameters."""
        int_positives = (
            "task_embedding_dim",
            "batch_size",
            "max_episode_steps",
            "recovery_trigger_threshold",
        )
        for key in int_positives:
            raw = self.learning_config.get(key, 1)
            validate_positive(raw, f"learning_agent.{key}")

        for prob_key in ("performance_threshold", "data_change_threshold",
                         "novelty_threshold", "uncertainty_threshold"):
            raw = self.learning_config.get(prob_key)
            if raw is not None:
                validate_in_range(raw, f"learning_agent.{prob_key}", 0.0, 1.0)

    # ------------------------------------------------------------------
    # Shared memory
    # ------------------------------------------------------------------
    def _init_shared_memory_keys(self) -> None:
        self.sm_keys: Dict[str, str] = {
            "decision_trace": f"learning:decision_trace:{self.name}",
            "metrics":        f"learning:metrics:{self.name}",
            "strategies":     f"learning:strategies:{self.name}",
            "episodes":       f"learning:episodes:{self.name}",
            "health":         f"learning:health:{self.name}",
        }

    # ------------------------------------------------------------------
    # Strategy selector bootstrap
    # ------------------------------------------------------------------
    def _initialize_strategy_selector(self) -> None:
        """Wire the StrategySelector with an embedder and policy network.

        The embedder maps an arbitrary-length flattened state array
        into a fixed ``task_embedding_dim``-dimensional representation.
        The policy network maps that embedding to per-strategy logits.
        """
        self.strategy_selector.set_agent_strategies_map(self.strategy_index)

        embedder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.task_embedding_dim),
        )
        self.strategy_selector.set_state_embedder(embedder)

        policy_network = PolicyNetwork(
            input_dim=self.task_embedding_dim,
            output_dim=len(self.task_ids),
            hidden_sizes=[128, 64],
            hidden_activation="relu",
            output_activation="linear",
        )
        optimizer = create_policy_optimizer(policy_network)
        self.strategy_selector.set_policy_network(
            policy_net=policy_network,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )
        logger.debug("StrategySelector initialized with %d strategies", len(self.task_ids))

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_state(reset_output: Any) -> Any:
        """Unpack Gym-style (obs, info) or plain obs from env.reset()."""
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            return reset_output[0]
        return reset_output

    @staticmethod
    def _safe_step(env: Any, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Normalise env.step() output to (next_state, reward, done, info).

        Supports both Gym ≥0.26 (5-tuple) and legacy 4-tuple APIs.
        """
        result = env.step(action)
        if not isinstance(result, tuple):
            raise RuntimeError("env.step() must return a tuple")
        if len(result) == 5:
            next_s, reward, terminated, truncated, info = result
            return next_s, float(reward), bool(terminated or truncated), info or {}
        if len(result) == 4:
            next_s, reward, done, info = result
            return next_s, float(reward), bool(done), info or {}
        raise RuntimeError(f"Unsupported env.step() output length: {len(result)}")

    @staticmethod
    def _align_vectors(
        a: np.ndarray, b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Zero-pad both arrays to the longer dimension."""
        max_dim = max(len(a), len(b))
        if len(a) < max_dim:
            a = np.pad(a, (0, max_dim - len(a)))
        if len(b) < max_dim:
            b = np.pad(b, (0, max_dim - len(b)))
        return a, b

    def _prepare_state_array(self, state: Any) -> np.ndarray:
        """Run StateProcessor on *state* and return a finite float32 vector."""
        processed = self.state_processor.process(state)
        if processed.numel() == 0:
            return np.zeros(1, dtype=np.float32)
        array = processed.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise NaNException(
                "State contains non-finite values after processing",
                location="state_processor",
            )
        return array

    # ------------------------------------------------------------------
    # Novelty & uncertainty
    # ------------------------------------------------------------------
    def _compute_novelty(self, state: np.ndarray) -> float:
        """Relative L2 distance between the current and most-recent state."""
        if not self.state_recency:
            return 1.0
        prev = self.state_recency[-1]
        prev, state = self._align_vectors(prev, state)
        denom = max(1e-8, float(np.linalg.norm(prev)) + float(np.linalg.norm(state)))
        return clamp01(float(np.linalg.norm(state - prev) / denom))

    def _estimate_uncertainty(self, embedding: torch.Tensor) -> float:
        """Normalised entropy of the policy distribution over strategies."""
        if self.strategy_selector.policy_net is None:
            # No policy network → maximum uncertainty
            return 1.0
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        with torch.no_grad():
            logits = self.strategy_selector.policy_net(embedding)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
        max_entropy = float(np.log(max(2, len(self.task_ids))))
        return clamp01(safe_divide(float(entropy), max_entropy))

    # ------------------------------------------------------------------
    # Strategy scoring & selection
    # ------------------------------------------------------------------
    def _normalized_strategy_weights(self) -> Dict[str, float]:
        """Return per-strategy prior weights, normalised to sum to 1."""
        weights = self.strategy_weights
        if len(weights) != len(self.task_ids):
            weights = np.ones(len(self.task_ids), dtype=np.float32)
        weights = np.clip(weights, 1e-8, None)
        weights = weights / float(weights.sum())
        return {task_id: float(weights[idx]) for idx, task_id in enumerate(self.task_ids)}

    def _strategy_scores(self, uncertainty: float, novelty: float,
                         task_metadata: Mapping[str, Any]) -> Dict[str, float]:
        """Compute a composite heuristic score for each strategy.

        Score components (all additive):
        1. Prior weight from config (normalised).
        2. Mean historical reward for that strategy (weighted).
        3. Exploration bonuses for MAML/RSI when uncertainty or novelty is
           above threshold.
        4. RSI bonus when overall performance trend is declining.
        5. Preferred-strategy boost from task metadata.
        """
        priors = self._normalized_strategy_weights()
        scores: Dict[str, float] = {task: priors.get(task, 0.0) for task in self.task_ids}

        # Reward-based component
        for task_id in self.task_ids:
            stats = self._reward_stats.get(task_id)
            mean_reward = stats.snapshot().mean if (stats and stats.count > 0) else 0.0
            scores[task_id] += self.heuristic_perf_weight * mean_reward

        # Exploration regime
        if uncertainty > self.uncertainty_threshold or novelty > self.novelty_threshold:
            if "maml" in scores:
                scores["maml"] += self.heuristic_exploration_bonus
            if "rsi" in scores:
                scores["rsi"] += self.heuristic_rsi_bonus

        # Stagnation / declining trend
        if self._performance_trend() < 0.0 and "rsi" in scores:
            scores["rsi"] += self.heuristic_trend_penalty_bonus

        # Explicit caller preference
        preferred = str(task_metadata.get("preferred_strategy", "")).strip().lower()
        if preferred in scores:
            scores[preferred] += self.heuristic_preferred_bonus

        return scores

    def _performance_trend(self) -> float:
        """Relative change in mean reward: recent window vs. earliest window.

        Delegates to LearningCalculations for numerical consistency.
        """
        history = list(self.performance_history)
        if len(history) < 10:
            return 0.0
        self.learning_calculations.performance_history.extend(history)
        return self.learning_calculations.calculate_performance_trend(window=10)

    def _resolve_strategy_name(self, agent: Any) -> str:
        """Map an agent object back to its registered factory name."""
        for name, known_agent in self.learning_factory.agents.items():
            if known_agent is agent:
                return name
        return getattr(agent, "agent_id", "dqn").split("_")[0].lower()

    def _select_strategy(self, state: Any,
                         task_metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, Any, _StrategyTrace]:
        """Select the best learning strategy for the given state.

        Decision pipeline:
        1. Embed state → policy network → policy_pick (learnt meta-controller).
        2. Compute heuristic scores → heuristic_pick.
        3. Ask LearningFactory based on task metadata → factory_pick.
        4. Combine: prefer heuristic when uncertainty is high; otherwise trust
           policy. Override with factory if caller sets selector_mode=factory.

        Returns
        -------
        strategy : str
            Name of the selected strategy (matches a task_id).
        agent : object
            Concrete learning agent object from LearningFactory.
        trace : _StrategyTrace
            Full audit record of the selection decision.
        """
        metadata = task_metadata or {}
        state_array = self._prepare_state_array(state)
        embedding = self.strategy_selector.generate_task_embedding(state_array)

        novelty = self._compute_novelty(state_array)
        uncertainty = self._estimate_uncertainty(embedding)
        policy_pick_raw = self.strategy_selector.select_strategy(embedding, return_details=False)
        policy_pick: str = str(policy_pick_raw)  # safe coercion
        heuristic_scores = self._strategy_scores(uncertainty, novelty, metadata)
        heuristic_pick: str = max(heuristic_scores, key=heuristic_scores.__getitem__)

        factory_agent = self.learning_factory.select_agent(
            {
                "novelty": novelty,
                "volatility": uncertainty,
                "preferred_agent": metadata.get("preferred_strategy"),
                "complexity": coerce_float(metadata.get("complexity", 0.5), default=0.5),
                "compute_budget": coerce_float(metadata.get("compute_budget", 0.5), default=0.5),
                "training_budget": coerce_float(metadata.get("training_budget", 0.5), default=0.5),
            }
        )
        factory_pick: str = self._resolve_strategy_name(factory_agent)

        # Final resolution
        selected: str = heuristic_pick if uncertainty > self.uncertainty_threshold else policy_pick
        if metadata.get("selector_mode") == "factory":
            selected = factory_pick

        selected_agent = self.learning_factory.agents.get(selected, factory_agent)

        trace = _StrategyTrace(
            policy_pick=policy_pick,
            heuristic_pick=heuristic_pick,
            factory_pick=factory_pick,
            selected=selected,
            uncertainty=round(uncertainty, 6),
            novelty=round(novelty, 6),
            scores={k: round(v, 6) for k, v in heuristic_scores.items()},
            state_hash=hashlib.sha256(state_array.tobytes()).hexdigest()[:16],
            timestamp=utc_now().isoformat(),
        )

        self.state_recency.append(state_array)
        self.performance_metrics["strategy_selection_count"][selected] += 1
        self.performance_metrics["novelty_score"][selected] = novelty
        self.performance_metrics["uncertainty_estimate"][selected] = uncertainty

        return selected, selected_agent, trace

    # ------------------------------------------------------------------
    # Action & learning-step adapters
    # ------------------------------------------------------------------
    def _agent_action(self, agent: Any, state: np.ndarray, explore: bool = True) -> int:
        """Dispatch to whichever action API the agent exposes."""
        for method_name in ("act", "select_action"):
            method = getattr(agent, method_name, None)
            if callable(method):
                try:
                    raw = method(state, explore=explore)
                except TypeError:
                    raw = method(state)
                # Convert to int safely
                try:
                    return int(raw) # type: ignore
                except (TypeError, ValueError):
                    raise InvalidActionError(
                        f"Agent method {method_name} returned non-integer: {raw!r}",
                        action=raw, # type: ignore
                    )
    
        get_action = getattr(agent, "get_action", None)
        if callable(get_action):
            result = get_action(state)
            raw_act = result[0] if isinstance(result, tuple) else result
            try:
                return int(raw_act) # type: ignore
            except (TypeError, ValueError):
                raise InvalidActionError(
                    f"Agent get_action returned non-integer: {raw_act!r}",
                    action=raw_act, # type: ignore
                )
    
        raise InvalidActionError(
            f"Agent of type '{type(agent).__name__}' exposes no compatible action API "
            "(expected act / select_action / get_action)",
            action=None, # type: ignore
        )

    def _agent_learn_step(self, strategy: str, agent: Any,
                          transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> Optional[float]:
        """Store a transition and trigger one agent training step.

        Returns the scalar loss if the agent exposes one, otherwise None.
        Errors are logged and appended to error_history rather than re-raised,
        so a single bad step never aborts an episode.
        """
        try:
            # Store transition
            if hasattr(agent, "store_transition"):
                agent.store_transition(*transition)
            elif hasattr(agent, "remember"):
                agent.remember(*transition)

            # Training step
            if not hasattr(agent, "train") or not callable(agent.train):
                return None

            train_result = agent.train()

            loss: Optional[float] = None
            if isinstance(train_result, (int, float)) and np.isfinite(train_result):
                loss = float(train_result)
            elif isinstance(train_result, dict):
                raw_loss = train_result.get("loss") or train_result.get("avg_loss")
                if isinstance(raw_loss, (int, float)) and np.isfinite(raw_loss):
                    loss = float(raw_loss)

            if loss is not None:
                self.multi_task_learner.update_loss(strategy, loss)
                self.performance_metrics["strategy_loss"][strategy] = loss
                return loss

        except LearningError as exc:
            logger.warning(
                "Learning step raised LearningError | strategy=%s | %s", strategy, exc
            )
            self.error_history.append(
                {"strategy": strategy, "error": str(exc), "error_type": type(exc).__name__, "timestamp": time.time()}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Learning step failed unexpectedly | strategy=%s | %s: %s",
                strategy, type(exc).__name__, exc,
            )
            self.error_history.append(
                {"strategy": strategy, "error": str(exc), "error_type": type(exc).__name__, "timestamp": time.time()}
            )
        return None

    # ------------------------------------------------------------------
    # Episode runner
    # ------------------------------------------------------------------
    def _run_episode(self, strategy: str, agent: Any, max_steps: int,
                     seed: Optional[int], train: bool = True) -> Dict[str, Any]:
        """Execute one full episode and return a structured report.

        During training:
        - Each transition is forwarded to the agent via _agent_learn_step.
        - MultiTaskLearner rebalances weights after the episode.
        - StrategySelector observes the final embedding and trains.

        Non-finite rewards abort the episode immediately (NaNException).
        """
        reset_output = self.env.reset(seed=seed)
        state = self._extract_state(reset_output)

        episode_id = make_learning_id(f"ep_{strategy}")
        total_reward = 0.0
        losses: List[float] = []
        step_idx = 0
        t_start = monotonic_seconds()

        for step_idx in range(max_steps):
            state_arr = self._prepare_state_array(state)
            action = self._agent_action(agent, state_arr, explore=train)
            if action < 0:
                raise InvalidActionError(
                    f"Agent returned invalid action {action!r}", action=action # type: ignore
                )

            next_state, reward, done, info = self._safe_step(self.env, action)

            if not np.isfinite(reward):
                raise NaNException(
                    f"Non-finite reward at step {step_idx}: {reward}",
                    location=f"env.step/{strategy}",
                )

            next_arr = self._prepare_state_array(next_state)

            if train:
                loss = self._agent_learn_step(
                    strategy, agent, (state_arr, action, reward, next_arr, done)
                )
                if loss is not None:
                    losses.append(loss)

            total_reward += reward
            self.observation_count += 1
            state = next_state

            if done:
                break

        elapsed_ms = int((monotonic_seconds() - t_start) * 1000)

        # Update reward tracking
        self.performance_history.append(total_reward)
        self._global_reward_stats.update(total_reward)
        if strategy in self._reward_stats:
            self._reward_stats[strategy].update(total_reward)
        else:
            self._reward_stats[strategy] = RunningStats()
            self._reward_stats[strategy].update(total_reward)

        # Per-strategy ring buffer in performance_metrics (for health_report)
        if strategy not in self.performance_metrics or not isinstance(
            self.performance_metrics.get(strategy), deque
        ):
            self.performance_metrics[strategy] = deque(maxlen=100)
        self.performance_metrics[strategy].append(total_reward)
        self.learning_factory.record_performance(strategy, total_reward)
        self.learning_calculations.update_performance(total_reward)

        # Post-episode bookkeeping (train mode only)
        if train:
            self.multi_task_learner.rebalance()
            final_arr = self._prepare_state_array(state)
            final_embedding = self.strategy_selector.generate_task_embedding(final_arr)
            self.strategy_selector.observe(final_embedding, strategy)
            self.strategy_selector.train_from_embeddings()

        avg_loss = float(np.mean(losses)) if losses else 0.0
        steps_taken = step_idx + 1

        return {
            "status": "ok",
            "episode_id": episode_id,
            "strategy": strategy,
            "episode_reward": total_reward,
            "steps": steps_taken,
            "avg_loss": avg_loss,
            "loss_samples": len(losses),
            "train_mode": train,
            "elapsed_ms": elapsed_ms,
            "timestamp": utc_now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Shared memory persistence
    # ------------------------------------------------------------------
    def _record_shared_memory(self, episode_result: Dict[str, Any], trace: _StrategyTrace) -> None:
        """Persist episode result and selection trace to shared memory."""
        # --- Rolling metric snapshot ---
        metrics = dict(self.shared_memory.get(self.sm_keys["metrics"]) or {})
        metrics.update(
            {
                "last_reward": episode_result["episode_reward"],
                "last_steps": episode_result["steps"],
                "last_strategy": episode_result["strategy"],
                "last_avg_loss": episode_result["avg_loss"],
                "train_mode": episode_result["train_mode"],
                "observations": self.observation_count,
                "training_iterations": self.training_iterations,
                "updated_at": utc_now().isoformat(),
            }
        )
        self.shared_memory.set(self.sm_keys["metrics"], metrics)

        # --- Strategy history (bounded ring) ---
        strategy_history: List[Dict[str, Any]] = list(
            self.shared_memory.get(self.sm_keys["strategies"]) or []
        )
        strategy_history.append(
            {
                "strategy": episode_result["strategy"],
                "reward": episode_result["episode_reward"],
                "train_mode": episode_result["train_mode"],
                "timestamp": time.time(),
            }
        )
        self.shared_memory.set(
            self.sm_keys["strategies"], strategy_history[-self.trace_history_limit:]
        )

        # --- Episode log ---
        episodes: List[Dict[str, Any]] = list(
            self.shared_memory.get(self.sm_keys["episodes"]) or []
        )
        episodes.append(episode_result)
        self.shared_memory.set(
            self.sm_keys["episodes"], episodes[-self.trace_history_limit:]
        )

        # --- Decision trace ---
        traces: List[Dict[str, Any]] = list(
            self.shared_memory.get(self.sm_keys["decision_trace"]) or []
        )
        traces.append(dict(trace))
        self.shared_memory.set(
            self.sm_keys["decision_trace"], traces[-self.trace_history_limit:]
        )

    # ------------------------------------------------------------------
    # Retraining gate
    # ------------------------------------------------------------------
    def _should_retrain(self) -> bool:
        return utc_now() - self.last_training_time >= self.retraining_interval

    # ------------------------------------------------------------------
    # Core modes
    # ------------------------------------------------------------------
    def _train_cycles(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run *episodes* training episodes and return an aggregate report."""
        episodes = coerce_int(task_data.get("episodes", 1), default=1, minimum=1)
        max_steps = coerce_int(
            task_data.get("max_steps", self.max_episode_steps),
            default=self.max_episode_steps,
            minimum=1,
        )
        seed: Optional[int] = task_data.get("seed")
        metadata: Dict[str, Any] = dict(task_data.get("task_metadata") or {})

        episode_reports: List[Dict[str, Any]] = []
        per_episode_rewards: List[float] = []
        t_start = monotonic_seconds()

        for episode_idx in range(episodes):
            probe_state = self._extract_state(self.env.reset(seed=seed))
            strategy, agent, trace = self._select_strategy(probe_state, metadata)
            report = self._run_episode(
                strategy=strategy,
                agent=agent,
                max_steps=max_steps,
                seed=seed,
                train=True,
            )
            report["episode_index"] = episode_idx
            report["selection_trace"] = dict(trace)
            episode_reports.append(report)
            per_episode_rewards.append(report["episode_reward"])
            self._record_shared_memory(report, trace)

        self.training_iterations += episodes
        self.last_training_time = utc_now()

        reward_summary = summarize_rewards(per_episode_rewards)
        elapsed_ms = int((monotonic_seconds() - t_start) * 1000)

        result: Dict[str, Any] = {
            "status": "ok",
            "mode": "train",
            "episodes": episodes,
            "avg_reward": reward_summary["mean"],
            "reward_summary": reward_summary,
            "reports": episode_reports,
            "trend": self._performance_trend(),
            "task_weights": self.multi_task_learner.get_weights(),
            "elapsed_ms": elapsed_ms,
        }

        # Trigger recovery if error accumulation exceeds threshold
        if len(self.error_history) >= self.recovery_trigger_threshold:
            logger.warning(
                "Error threshold reached (%d errors) — triggering recovery",
                len(self.error_history),
            )
            result["recovery"] = self.recovery_system.execute_recovery()

        logger.info(
            "Training complete | episodes=%d | avg_reward=%.4f | trend=%.4f",
            episodes,
            reward_summary["mean"],
            result["trend"],
        )
        return result

    def _evaluate(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation episodes (no learning updates) and return metrics."""
        episodes = coerce_int(
            task_data.get("episodes", self.max_eval_episodes),
            default=self.max_eval_episodes,
            minimum=1,
        )
        max_steps = coerce_int(
            task_data.get("max_steps", self.max_episode_steps),
            default=self.max_episode_steps,
            minimum=1,
        )
        metadata: Dict[str, Any] = dict(task_data.get("task_metadata") or {})

        rewards: List[float] = []
        steps_list: List[int] = []
        strategies_used: List[str] = []
        traces: List[Dict[str, Any]] = []

        for _ in range(episodes):
            state = self._extract_state(self.env.reset())
            strategy, agent, trace = self._select_strategy(state, metadata)
            report = self._run_episode(
                strategy=strategy,
                agent=agent,
                max_steps=max_steps,
                seed=None,
                train=False,
            )
            rewards.append(report["episode_reward"])
            steps_list.append(report["steps"])
            strategies_used.append(strategy)
            traces.append(dict(trace))

        reward_summary = summarize_rewards(rewards)
        result: Dict[str, Any] = {
            "status": "ok",
            "mode": "evaluate",
            "episodes": episodes,
            "avg_reward": reward_summary["mean"],
            "avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
            "reward_summary": reward_summary,
            "strategy_distribution": {
                k: strategies_used.count(k) for k in sorted(set(strategies_used))
            },
            "traces": traces,
        }

        existing = dict(self.shared_memory.get(self.sm_keys["metrics"]) or {})
        existing["last_evaluation"] = result
        self.shared_memory.set(self.sm_keys["metrics"], existing)

        logger.info(
            "Evaluation complete | episodes=%d | avg_reward=%.4f",
            episodes,
            reward_summary["mean"],
        )
        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def perform_task(self, task_data: Any) -> Dict[str, Any]:
        """Top-level dispatcher — entry point for all agent calls.

        Supported modes
        ---------------
        ``"train"``    — Training loop with optional recovery.
        ``"evaluate"`` — Evaluation loop (no gradient updates).
        ``"auto"``     — Train if stale, else evaluate.
        ``"health"``   — Health/diagnostics snapshot.

        Parameters
        ----------
        task_data : dict | Any
            When dict, recognised keys are:
            - ``"mode"``          : str — one of the above modes.
            - ``"episodes"``      : int — number of episodes to run.
            - ``"max_steps"``     : int — step cap per episode.
            - ``"seed"``          : int | None — env reset seed.
            - ``"task_metadata"`` : dict — metadata passed to strategy selector.
        """
        if task_data is None:
            task_data = {}
        if not isinstance(task_data, dict):
            task_data = {"payload": task_data}

        mode = str(task_data.get("mode", "train")).strip().lower()

        if mode == "health":
            report = self.health_report()
            self.shared_memory.set(self.sm_keys["health"], report)
            return {"status": "ok", "mode": "health", "report": report}

        if mode == "evaluate":
            return self._evaluate(task_data)

        if mode == "train":
            return self._train_cycles(task_data)

        if mode == "auto":
            if self._should_retrain():
                logger.info("Auto mode: retraining interval elapsed — training")
                return self._train_cycles(task_data)
            logger.info("Auto mode: interval not elapsed — evaluating")
            return self._evaluate(task_data)

        raise InvalidConfigError(
            f"Unsupported perform_task mode: {mode!r}. "
            "Expected one of: 'train', 'evaluate', 'auto', 'health'.",
            config_key="mode",
            received_value=mode,
        )

    def alternative_execute(self, task_data: Any, original_error: Optional[BaseException] = None) -> Dict[str, Any]:
        """Fallback execution path invoked by BaseAgent on repeated failure.

        Triggers a recovery cycle, then runs a minimal evaluation to
        confirm the system is still functional.
        """
        recovery_result = self.recovery_system.execute_recovery(error=original_error)
        safe_task: Dict[str, Any] = dict(task_data) if isinstance(task_data, dict) else {"payload": task_data}
        safe_task["mode"] = "evaluate"
        safe_task["episodes"] = min(2, coerce_int(safe_task.get("episodes", 1), default=1, minimum=1))
        safe_task["max_steps"] = min(32, coerce_int(safe_task.get("max_steps", 32), default=32, minimum=1))

        try:
            eval_result = self._evaluate(safe_task)
        except Exception:  # noqa: BLE001
            eval_result = super().alternative_execute(safe_task, original_error=original_error)

        return {
            "status": "recovered_fallback",
            "recovery": recovery_result,
            "fallback_result": eval_result,
        }

    def extract_performance_metrics(self, result: Any) -> Dict[str, Any]:
        """Normalise a task result dict into a flat metrics snapshot."""
        if not isinstance(result, dict):
            return {}
        return {
            "avg_reward": coerce_float(
                result.get("avg_reward", result.get("episode_reward", 0.0))
            ),
            "episodes": coerce_int(result.get("episodes", 1), default=1),
            "steps": coerce_float(result.get("avg_steps", result.get("steps", 0))),
            "trend": coerce_float(result.get("trend", self._performance_trend())),
        }

    # ------------------------------------------------------------------
    # Health & diagnostics
    # ------------------------------------------------------------------
    def health_report(self) -> Dict[str, Any]:
        """Return a comprehensive snapshot of agent state and performance.

        Covers per-strategy reward statistics, global counters, task
        weights, error history summary, and running stats from the
        LearningCalculations subsystem.
        """
        strategy_health: Dict[str, Any] = {}
        for strategy in self.task_ids:
            stats = self._reward_stats.get(strategy)
            snap = stats.snapshot() if (stats and stats.count > 0) else None
            strategy_health[strategy] = {
                "samples": snap.count if snap else 0,
                "mean_reward": snap.mean if snap else 0.0,
                "std_reward": snap.std if snap else 0.0,
                "min_reward": snap.minimum if snap else 0.0,
                "max_reward": snap.maximum if snap else 0.0,
                "selection_count": self.performance_metrics["strategy_selection_count"].get(strategy, 0),
                "last_loss": self.performance_metrics["strategy_loss"].get(strategy, 0.0),
            }

        global_snap = self._global_reward_stats.snapshot()

        return {
            "name": self.name,
            "agent_id": self.agent_id,
            "version": __version__,
            "task_ids": list(self.task_ids),
            "observations": self.observation_count,
            "training_iterations": self.training_iterations,
            "trend": self._performance_trend(),
            "last_training_time": self.last_training_time.isoformat(),
            "retraining_due": self._should_retrain(),
            "recovery_error_count": self.recovery_system.error_count,
            "error_history_size": len(self.error_history),
            "task_weights": self.multi_task_learner.get_weights(),
            "global_reward_stats": {
                "count": global_snap.count,
                "mean": global_snap.mean,
                "std": global_snap.std,
                "min": global_snap.minimum,
                "max": global_snap.maximum,
            },
            "strategy_health": strategy_health,
        }


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------
def _pad_or_trim(values: List[float], target_len: int, fill: float = 0.0) -> List[float]:
    """Return *values* padded with *fill* or truncated to *target_len*."""
    if len(values) >= target_len:
        return values[:target_len]
    return values + [fill] * (target_len - len(values))


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Learning Agent ===\n")
    printer.status("TEST", "Learning Agent self-test starting", "info")
    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory
    from .learning.slaienv import SLAIEnv

    # ------------------------------------------------------------------
    # 1. Config helpers (tested in isolation)
    # ------------------------------------------------------------------
    printer.status("TEST", "1/7 — Validating config helpers", "info")
    assert coerce_int("5", default=0) == 5
    assert coerce_float("3.14", default=0.0) == 3.14
    assert coerce_bool("yes") is True
    assert clamp(1.5, 0.0, 1.0) == 1.0
    assert clamp01(-0.1) == 0.0
    assert safe_divide(1.0, 0.0) == 0.0
    printer.status("PASS", "Config helpers OK", "success")

    # ------------------------------------------------------------------
    # 2. _pad_or_trim utility
    # ------------------------------------------------------------------
    printer.status("TEST", "2/7 — _pad_or_trim", "info")
    assert _pad_or_trim([0.1, 0.2], 4, fill=0.25) == [0.1, 0.2, 0.25, 0.25]
    assert _pad_or_trim([0.1, 0.2, 0.3, 0.4, 0.5], 4) == [0.1, 0.2, 0.3, 0.4]
    printer.status("PASS", "_pad_or_trim OK", "success")

    # ------------------------------------------------------------------
    # 3. RunningStats (used for reward tracking)
    # ------------------------------------------------------------------
    printer.status("TEST", "3/7 — RunningStats", "info")
    rs = RunningStats()
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        rs.update(v)
    snap = rs.snapshot()
    assert snap.count == 5
    assert abs(snap.mean - 3.0) < 1e-6, f"Expected mean=3.0 got {snap.mean}"
    assert snap.minimum == 1.0
    assert snap.maximum == 5.0
    printer.status("PASS", "RunningStats OK", "success")

    # ------------------------------------------------------------------
    # 4. LearningCalculations integration
    # ------------------------------------------------------------------
    printer.status("TEST", "4/7 — LearningCalculations", "info")
    lc = LearningCalculations()
    returns = lc.discounted_returns([1.0, 1.0, 1.0], gamma=0.9)
    assert len(returns) == 3
    assert abs(returns[0] - (1 + 0.9 + 0.81)) < 1e-5, f"Unexpected returns: {returns}"
    p = np.array([0.4, 0.6])
    q = np.array([0.5, 0.5])
    kl = lc.calculate_kl_divergence(p, q)
    assert kl >= 0.0, f"KL should be non-negative, got {kl}"
    printer.status("PASS", "LearningCalculations OK", "success")

    # ------------------------------------------------------------------
    # 5. Episode metric helpers
    # ------------------------------------------------------------------
    printer.status("TEST", "5/7 — Episode metric helpers", "info")
    em = make_episode_metrics(0, [1.0, 2.0, 3.0], moving_average_reward=2.0, completion_rate=1.0)
    assert em.total_reward == 6.0
    assert em.length == 3
    assert abs(em.average_reward - 2.0) < 1e-6
    rs_ep = summarize_rewards([1.0, 2.0, 3.0])
    assert rs_ep["count"] == 3
    assert abs(rs_ep["mean"] - 2.0) < 1e-6
    printer.status("PASS", "Episode metric helpers OK", "success")

    # ------------------------------------------------------------------
    # 6. LearningAgent instantiation (patched env)
    # ------------------------------------------------------------------
    printer.status("TEST", "6/7 — LearningAgent instantiation", "info")
    memory = SharedMemory()
    factory = AgentFactory()
    fake_env = SLAIEnv(state_dim=4, action_dim=3)

    # Inject stub overrides for subsystems that require real GPU/files
    la_config = get_config_section("learning_agent") or {}

    agent = LearningAgent(
        shared_memory=memory,
        agent_factory=factory,
        config=la_config,
        env=fake_env,
    )
    assert agent.name == "LearningAgent"
    assert len(agent.task_ids) >= 1
    assert isinstance(agent.learning_factory, LearningFactory)
    assert isinstance(agent.strategy_selector, StrategySelector)
    assert isinstance(agent.recovery_system, RecoverySystem)
    printer.status("PASS", f"LearningAgent instantiated | task_ids={agent.task_ids}", "success")

    # ------------------------------------------------------------------
    # 7. perform_task modes
    # ------------------------------------------------------------------
    printer.status("TEST", "7/7 — perform_task modes", "info")

    # Health
    h = agent.perform_task({"mode": "health"})
    assert h["status"] == "ok"
    assert h["mode"] == "health"
    assert "strategy_health" in h["report"]
    printer.status("PASS", f"health mode OK | observations={h['report']['observations']}", "success")

    # Evaluate
    ev = agent.perform_task({"mode": "evaluate", "episodes": 1, "max_steps": 5})
    assert ev["status"] == "ok"
    assert ev["mode"] == "evaluate"
    assert "avg_reward" in ev
    printer.status("PASS", f"evaluate mode OK | avg_reward={ev['avg_reward']:.4f}", "success")

    # Train
    tr = agent.perform_task({"mode": "train", "episodes": 2, "max_steps": 5})
    assert tr["status"] == "ok"
    assert tr["mode"] == "train"
    assert tr["episodes"] == 2
    printer.status("PASS", f"train mode OK | avg_reward={tr['avg_reward']:.4f} | trend={tr['trend']:.4f}", "success")

    # Auto (should evaluate since interval not elapsed)
    auto = agent.perform_task({"mode": "auto", "episodes": 1, "max_steps": 5})
    assert auto["status"] == "ok"
    printer.status("PASS", f"auto mode OK | routed to '{auto['mode']}'", "success")

    # extract_performance_metrics
    ep_metrics = agent.extract_performance_metrics(tr)
    assert "avg_reward" in ep_metrics
    assert "trend" in ep_metrics
    printer.status("PASS", "extract_performance_metrics OK", "success")

    # Invalid mode guard
    try:
        agent.perform_task({"mode": "invalid_xyz"})
        assert False, "Should have raised InvalidConfigError"
    except InvalidConfigError:
        printer.status("PASS", "Invalid mode correctly rejected", "success")

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    final_health = agent.health_report()
    printer.status(
        "INFO",
        f"Final state | observations={final_health['observations']} | "
        f"training_iterations={final_health['training_iterations']} | "
        f"global_mean_reward={final_health['global_reward_stats']['mean']:.4f}",
        "info",
    )

    print("\n=== Test ran successfully ===\n")
