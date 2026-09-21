from __future__ import annotations

__version__ = "2.3.0"

"""SLAI v2.3 Reasoning Agent.

The agent is deliberately a thin application-facing facade over
``src.agents.reasoning``.  Agent-level policy comes only from the central
``agents_config.yaml`` contract inherited from the BaseAgent configuration
stack.  Algorithmic parameters, model resources, symbolic inference policy,
validation policy, and probabilistic backend settings remain subsystem-owned.

State ownership
---------------
- BaseAgent: lifecycle, checkpoint orchestration, shared-memory ownership,
  telemetry infrastructure, retry/recovery envelope.
- ReasoningAgent: request routing, bounded refinement, transient working facts,
  result publication, agent-level diagnostics.
- RuleEngine: symbolic facts/rules, inference, adaptive rule weights.
- ReasoningTypes: strategy selection/invocation and canonical result semantics.
- ValidationEngine: reasoning validation pipeline.
- ProbabilisticModels / HybridProbabilisticModels: probabilistic semantics.

The agent does not perform direct filesystem persistence for knowledge and does
not own subsystem configuration or subsystem-local experience storage.
"""

import json
import time
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from threading import Event, RLock, Thread
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .base_agent import BaseAgent
from .base.utils.config_contract import assert_valid_config_contract
from .base.utils.main_config_loader import get_config_section, load_global_config
from .reasoning.reasoning_types import ReasoningTypes
from .reasoning.rule_engine import RuleEngine
from .reasoning.validation import ValidationEngine
from .reasoning.probabilistic_models import ProbabilisticModels
from .reasoning.hybrid_probabilistic_models import HybridProbabilisticModels
from .reasoning.utils.reasoning_errors import *
from .reasoning.utils.reasoning_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Reasoning Agent")
printer = PrettyPrinter()


@dataclass(frozen=True)
class AgentReasoningTrace:
    """Compact audit record for one ReasoningAgent operation."""

    operation: str
    started_at: float
    finished_at: float
    status: str
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe_reasoning_state(
            {
                "operation": self.operation,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "duration_ms": int((self.finished_at - self.started_at) * 1000),
                "status": self.status,
                "summary": self.summary,
            }
        )


@dataclass(frozen=True)
class ForwardChainReport:
    """Structured symbolic-inference report exposed by the Agent."""

    added: Dict[Fact, float]
    iterations: int
    conflicts: List[Any]
    redundancies: List[Any]
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return json_safe_reasoning_state(
            {
                "added": self.added,
                "iterations": self.iterations,
                "conflicts": self.conflicts,
                "redundancies": self.redundancies,
                "duration_seconds": self.duration_seconds,
            }
        )


class DistributedLock:
    """CAS-backed renewable lease over SLAI SharedMemory.

    The lease duration and acquisition timeout are deliberately separate.  The
    expiry is calculated when the CAS is attempted, never before a wait loop,
    and a heartbeat renews long-running inference safely.
    """

    def __init__(
        self,
        shared_memory: Any,
        lock_key: str,
        timeout_seconds: float = 30.0,
        *,
        acquire_timeout_seconds: Optional[float] = None,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        if shared_memory is None:
            raise ReasoningConfigurationError(
                "DistributedLock requires shared memory.",
                context={"lock_key": lock_key},
            )

        getter = getattr(shared_memory, "get", None)
        cas = getattr(shared_memory, "compare_and_swap", None)
        if not callable(getter) or not callable(cas):
            raise ReasoningConfigurationError(
                "Shared memory does not satisfy the distributed-lock contract.",
                context={
                    "lock_key": lock_key,
                    "has_get": callable(getter),
                    "has_compare_and_swap": callable(cas),
                },
            )

        self.sm = shared_memory
        self.lock_key = str(lock_key)
        self.lease_seconds = max(0.5, float(timeout_seconds))
        self.acquire_timeout_seconds = max(
            0.0,
            float(
                self.lease_seconds
                if acquire_timeout_seconds is None
                else acquire_timeout_seconds
            ),
        )
        self.poll_interval_seconds = max(0.01, float(poll_interval_seconds))

        self._owner_token = uuid.uuid4().hex
        self._held = False
        self._lost = False
        self._lease_record: Optional[Dict[str, Any]] = None
        self._state_lock = RLock()
        self._heartbeat_stop = Event()
        self._heartbeat_thread: Optional[Thread] = None

    @staticmethod
    def _expiry_of(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, Mapping):
            try:
                return float(value.get("expires_at", 0.0))
            except (TypeError, ValueError):
                return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _is_owned_by_me(self, value: Any) -> bool:
        return isinstance(value, Mapping) and value.get("owner") == self._owner_token

    def _new_lease_record(self) -> Dict[str, Any]:
        return {
            "owner": self._owner_token,
            "expires_at": time.time() + self.lease_seconds,
        }

    def acquire(self, blocking: bool = True) -> bool:
        deadline = (
            time.monotonic() + self.acquire_timeout_seconds
            if blocking
            else time.monotonic()
        )

        while True:
            current = self.sm.get(self.lock_key)
            if current is None or self._expiry_of(current) <= time.time():
                candidate = self._new_lease_record()
                if self.sm.compare_and_swap(self.lock_key, current, candidate):
                    with self._state_lock:
                        self._held = True
                        self._lost = False
                        self._lease_record = candidate
                    self._start_heartbeat()
                    return True

            if not blocking:
                return False

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            time.sleep(min(self.poll_interval_seconds, remaining))

    def renew(self) -> bool:
        with self._state_lock:
            if not self._held or self._lost:
                return False

        current = self.sm.get(self.lock_key)
        if not self._is_owned_by_me(current):
            with self._state_lock:
                self._held = False
                self._lost = True
            return False

        updated = self._new_lease_record()
        if not self.sm.compare_and_swap(self.lock_key, current, updated):
            with self._state_lock:
                self._held = False
                self._lost = True
            return False

        with self._state_lock:
            self._lease_record = updated
        return True

    def assert_held(self) -> None:
        with self._state_lock:
            if not self._held or self._lost:
                raise InferenceExecutionError(
                    "Distributed reasoning lock was lost.",
                    context={"lock_key": self.lock_key},
                )

        current = self.sm.get(self.lock_key)
        if (
            not self._is_owned_by_me(current)
            or self._expiry_of(current) <= time.time()
        ):
            with self._state_lock:
                self._held = False
                self._lost = True
            raise InferenceExecutionError(
                "Distributed reasoning lock expired or changed ownership.",
                context={"lock_key": self.lock_key},
            )

    def _start_heartbeat(self) -> None:
        self._heartbeat_stop.clear()
        interval = max(0.1, min(self.lease_seconds / 3.0, self.lease_seconds * 0.5))
        self._heartbeat_thread = Thread(
            target=self._heartbeat_loop,
            args=(interval,),
            name=f"reasoning-lock:{self.lock_key}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self, interval: float) -> None:
        while not self._heartbeat_stop.wait(interval):
            try:
                if not self.renew():
                    logger.error("Reasoning distributed lock lease lost | key=%s", self.lock_key)
                    return
            except Exception as exc:
                with self._state_lock:
                    self._held = False
                    self._lost = True
                logger.error( "Reasoning distributed lock renewal failed | key=%s | error=%s", self.lock_key, exc)
                return

    def release(self) -> None:
        self._heartbeat_stop.set()
        heartbeat = self._heartbeat_thread
        if heartbeat is not None and heartbeat.is_alive():
            heartbeat.join(timeout=max(0.2, min(self.lease_seconds, 1.0)))

        current = self.sm.get(self.lock_key)
        if self._is_owned_by_me(current):
            self.sm.compare_and_swap(self.lock_key, current, None)

        with self._state_lock:
            self._held = False
            self._lost = False
            self._lease_record = None
            self._heartbeat_thread = None

    def __enter__(self) -> "DistributedLock":
        if not self.acquire(blocking=True):
            raise ReasoningTimeoutError(
                "Timed out acquiring distributed reasoning lock.",
                context={
                    "lock_key": self.lock_key,
                    "acquire_timeout_seconds": self.acquire_timeout_seconds,
                },
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.release()
        return False


class ReasoningAgent(BaseAgent):
    """Application-facing Reasoning facade for SLAI v2.3."""

    AGENT_KEY = "reasoning_agent"
    WORKING_KNOWLEDGE_KEY_DEFAULT = "reasoning_agent:working_knowledge"
    KNOWLEDGE_CANDIDATE_TOPIC_DEFAULT = "reasoning_agent:knowledge_candidates"
    STATE_UPDATED_TOPIC = "reasoning_agent:state_updated"

    CHECKPOINTING_SUPPORTED = True
    CHECKPOINT_SCHEMA = "slai.reasoning-agent.state.v2"

    _ALLOWED_CONFIG_KEYS = {
        "large_kb_threshold",
        "low_confidence_threshold",
        "max_action_results",
        "max_chain_depth",
        "max_react_steps",
        "max_trace_items",
        "enable_shared_memory_publish",
        "enable_memory_logging",
        "enable_probabilistic_fallback",
        "strict_fact_validation",
        "human_intervention_key",
        "working_knowledge_key",
        "knowledge_candidate_topic",
        "last_validation_key",
        "reasoning_trace_topic",
        "memory_event_tag",
        "memory_event_priority",
        "strategy_default",
        "reasoning_type_aliases",
    }

    def __init__(
        self,
        shared_memory: Any,
        agent_factory: Any,
        config: Optional[Mapping[str, Any]] = None,
        *,
        checkpoint_manager: Any = None,
    ) -> None:
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config,
            checkpoint_manager=checkpoint_manager,
        )

        self._reasoning_lock = RLock()

        self.config = load_global_config()
        self.agent_config: Dict[str, Any] = dict(
            get_config_section(self.AGENT_KEY) or {}
        )
        if config:
            self.agent_config.update(dict(config))

        assert_valid_config_contract(
            global_config=self.config,
            agent_key=self.AGENT_KEY,
            agent_config=self.agent_config,
            logger=logger,
            agent_allowed_keys=self._ALLOWED_CONFIG_KEYS,
            require_global_keys=False,
            require_agent_section=False,
            warn_unknown_global_keys=False,
        )

        self._load_runtime_config()
        self._validate_runtime_config()

        # One transient working-fact mapping is shared by all stateful reasoning
        # components.  The Agent never creates a parallel symbolic KB.
        self.working_knowledge: Dict[Fact, float] = self._load_working_knowledge()

        self.rule_engine = RuleEngine(knowledge_base=self.working_knowledge)
        self.validation_engine = ValidationEngine(knowledge_base=self.working_knowledge)
        self.probabilistic_models = ProbabilisticModels(
            knowledge_base=self.working_knowledge
        )
        self.hybrid_models = HybridProbabilisticModels()
        self.types = ReasoningTypes(default_strategy=self.strategy_default)

        self.conflict_count = 0
        self.last_forward_chaining_duration_seconds = 0.0
        self.operation_counts: Counter[str] = Counter()
        self.reasoning_history: Deque[Dict[str, Any]] = deque(
            maxlen=self.max_trace_items
        )

        self._publish_working_state(reason="initialized", publish=False)
        logger.info(
            "ReasoningAgent initialized | working_facts=%s | rules=%s",
            len(self.working_knowledge),
            len(self.rule_engine.list_rules()),
        )

    # ------------------------------------------------------------------
    # Compatibility read-only views
    # ------------------------------------------------------------------
    @property
    def knowledge_base(self) -> MutableMapping[Fact, float]:
        """Compatibility view of the canonical RuleEngine working facts."""
        return self.working_knowledge

    @property
    def rules(self) -> List[Any]:
        """Compatibility view of registered executable rule entries."""
        getter = getattr(self.rule_engine, "rule_entries", None)
        return list(getter()) if callable(getter) else [] # type: ignore

    @property
    def rule_weights(self) -> Dict[str, float]:
        """Compatibility snapshot; callers cannot mutate engine weights through it."""
        return self._rule_weight_snapshot()

    @property
    def forward_chaining_speed(self) -> float:
        """Legacy alias retained for v2.3 callers; value is a duration in seconds."""
        return self.last_forward_chaining_duration_seconds

    # ------------------------------------------------------------------
    # Agent-level configuration only
    # ------------------------------------------------------------------
    def _load_runtime_config(self) -> None:
        cfg = self.agent_config

        self.large_kb_threshold = bounded_iterations(
            cfg.get("large_kb_threshold", 500),
            minimum=1,
            maximum=50_000_000,
        )
        self.max_action_results = bounded_iterations(
            cfg.get("max_action_results", 20),
            minimum=1,
            maximum=100_000,
        )
        self.max_chain_depth = bounded_iterations(
            cfg.get("max_chain_depth", 4),
            minimum=1,
            maximum=256,
        )
        self.max_react_steps = bounded_iterations(
            cfg.get("max_react_steps", 5),
            minimum=1,
            maximum=256,
        )
        self.max_trace_items = bounded_iterations(
            cfg.get("max_trace_items", 250),
            minimum=1,
            maximum=100_000,
        )
        self.low_confidence_threshold = clamp_confidence(
            cfg.get("low_confidence_threshold", 0.4)
        )

        self.enable_shared_memory_publish = bool(
            cfg.get("enable_shared_memory_publish", True)
        )
        self.enable_memory_logging = bool(cfg.get("enable_memory_logging", True))
        self.enable_probabilistic_fallback = bool(
            cfg.get("enable_probabilistic_fallback", True)
        )
        self.strict_fact_validation = bool(cfg.get("strict_fact_validation", True))

        self.working_knowledge_key = str(
            cfg.get("working_knowledge_key", self.WORKING_KNOWLEDGE_KEY_DEFAULT)
        ).strip()
        self.knowledge_candidate_topic = str(
            cfg.get(
                "knowledge_candidate_topic",
                self.KNOWLEDGE_CANDIDATE_TOPIC_DEFAULT,
            )
        ).strip()
        self.last_validation_key = str(
            cfg.get("last_validation_key", "reasoning_agent:last_validated_fact")
        ).strip()
        self.reasoning_trace_topic = str(
            cfg.get("reasoning_trace_topic", "reasoning_trace")
        ).strip()
        self.human_intervention_key = str(
            cfg.get("human_intervention_key", "human_intervention_requests")
        ).strip()
        self.memory_event_tag = str(
            cfg.get("memory_event_tag", "reasoning_agent")
        ).strip()
        self.memory_event_priority = clamp_confidence(
            cfg.get("memory_event_priority", 0.75)
        )

        self.strategy_default = str(
            cfg.get("strategy_default", "deduction")
        ).strip().lower() or "deduction"
        self.reasoning_type_aliases = {
            str(key).strip().lower(): str(value).strip().lower()
            for key, value in dict(cfg.get("reasoning_type_aliases", {})).items()
            if str(key).strip() and str(value).strip()
        }

    def _validate_runtime_config(self) -> None:
        required_strings = {
            "working_knowledge_key": self.working_knowledge_key,
            "knowledge_candidate_topic": self.knowledge_candidate_topic,
            "last_validation_key": self.last_validation_key,
            "reasoning_trace_topic": self.reasoning_trace_topic,
            "human_intervention_key": self.human_intervention_key,
            "memory_event_tag": self.memory_event_tag,
            "strategy_default": self.strategy_default,
        }
        empty = sorted(key for key, value in required_strings.items() if not value)
        if empty:
            raise ReasoningConfigurationError(
                "ReasoningAgent central configuration contains empty required values.",
                context={"empty_keys": empty},
            )

    # ------------------------------------------------------------------
    # Shared runtime state
    # ------------------------------------------------------------------
    def _shared_get(self, key: str, default: Any = None) -> Any:
        getter = getattr(self.shared_memory, "get", None)
        if not callable(getter):
            return default
        try:
            return getter(key, default=default)
        except TypeError:
            value = getter(key)
            return default if value is None else value

    def _shared_set(self, key: str, value: Any) -> None:
        setter = getattr(self.shared_memory, "set", None)
        if callable(setter):
            setter(key, value)

    def _shared_publish(self, topic: str, payload: Any) -> None:
        if not self.enable_shared_memory_publish:
            return
        publisher = getattr(self.shared_memory, "publish", None)
        if callable(publisher):
            publisher(topic, payload)

    def _shared_append(self, key: str, payload: Any) -> None:
        appender = getattr(self.shared_memory, "append", None)
        if callable(appender):
            appender(key, payload)
            return

        # CAS fallback preserves cross-agent atomicity when append is not part
        # of the concrete SharedMemory implementation.
        cas = getattr(self.shared_memory, "compare_and_swap", None)
        if callable(cas):
            for _ in range(16):
                current = self._shared_get(key, default=[])
                current_list = list(current) if isinstance(current, list) else [current]
                updated = current_list + [payload]
                if cas(key, current, updated):
                    return
            raise InferenceExecutionError(
                "Unable to append shared-memory item after bounded CAS retries.",
                context={"key": key},
            )

        with self._reasoning_lock:
            current = self._shared_get(key, default=[])
            current_list = list(current) if isinstance(current, list) else [current]
            current_list.append(payload)
            self._shared_set(key, current_list)

    def _record_runtime_event(
        self,
        payload: Mapping[str, Any],
        *,
        tag: Optional[str] = None,
        priority: Optional[float] = None,
    ) -> None:
        if not self.enable_memory_logging:
            return
        add = getattr(self.shared_memory, "add", None)
        if not callable(add):
            return
        try:
            add(
                experience=json_safe_reasoning_state(dict(payload)),
                tag=tag or self.memory_event_tag,
                priority=(
                    self.memory_event_priority
                    if priority is None
                    else clamp_confidence(priority)
                ),
            )
        except Exception as exc:
            logger.warning("Reasoning runtime-event logging failed: %s", exc)

    def _record_trace(self, trace_record: AgentReasoningTrace) -> None:
        payload = trace_record.to_dict()
        self.reasoning_history.append(payload)
        self._shared_publish(
            self.reasoning_trace_topic,
            {"agent": self.name, "trace": payload},
        )
        self._record_runtime_event(payload)

    def _load_working_knowledge(self) -> Dict[Fact, float]:
        raw = self._shared_get(self.working_knowledge_key, default={})
        return self._normalize_knowledge_payload(raw)

    def _normalize_knowledge_payload(self, raw: Any) -> Dict[Fact, float]:
        normalized: Dict[Fact, float] = {}
        if not raw:
            return normalized

        if isinstance(raw, Mapping) and "knowledge" in raw:
            items: Iterable[Any] = raw.get("knowledge", [])
        elif isinstance(raw, Mapping):
            items = raw.items()
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            items = raw
        else:
            raise ReasoningValidationError(
                "Working knowledge must be a mapping or sequence.",
                context={"actual_type": type(raw).__name__},
            )

        for item in items:
            try:
                if isinstance(item, Mapping):
                    fact = (
                        item.get("subject", ""),
                        item.get("predicate", ""),
                        item.get("object", ""),
                    )
                    confidence = item.get("confidence", item.get("weight", 0.0))
                elif (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], (tuple, list, str))
                ):
                    fact, confidence = item
                elif isinstance(item, (tuple, list)) and len(item) >= 4:
                    fact = (item[0], item[1], item[2])
                    confidence = item[3]
                else:
                    continue
                normalized[normalize_fact(fact)] = clamp_confidence(confidence)
            except ReasoningError:
                if self.strict_fact_validation:
                    raise
            except Exception as exc:
                if self.strict_fact_validation:
                    raise ReasoningValidationError(
                        "Invalid working-knowledge item.",
                        cause=exc,
                        context={"item": repr(item)},
                    ) from exc
        return normalized

    def _serialize_working_state(self, *, reason: str) -> Dict[str, Any]:
        return json_safe_reasoning_state(
            {
                "knowledge": [
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "confidence": confidence,
                    }
                    for (subject, predicate, obj), confidence
                    in sorted(self.working_knowledge.items(), key=lambda entry: entry[0])
                ],
                "rules": self.rule_engine.list_rules(),
                "reason": reason,
                "updated_at": time.time(),
            }
        )

    def _publish_working_state(self, *, reason: str, publish: bool = True) -> None:
        # Shared state is runtime coordination, not canonical knowledge storage.
        self._shared_set(self.working_knowledge_key, dict(self.working_knowledge))
        if publish:
            self._shared_publish(
                self.STATE_UPDATED_TOPIC,
                self._serialize_working_state(reason=reason),
            )

    def _publish_knowledge_candidate(
        self,
        operation: str,
        fact: Fact,
        confidence: float,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._shared_publish(
            self.knowledge_candidate_topic,
            json_safe_reasoning_state(
                {
                    "agent": self.name,
                    "agent_id": self.agent_id,
                    "operation": operation,
                    "fact": fact,
                    "confidence": clamp_confidence(confidence),
                    "metadata": dict(metadata or {}),
                    "timestamp": time.time(),
                }
            ),
        )

    # ------------------------------------------------------------------
    # Base metric-store helpers
    # ------------------------------------------------------------------
    def _metric_value(
        self,
        metric_name: str,
        value: float,
        *,
        unit: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        recorder = getattr(getattr(self, "metric_store", None), "record_value", None)
        if not callable(recorder):
            return
        try:
            recorder(
                metric_name,
                float(value),
                category="reasoning",
                unit=unit,
                metadata=dict(metadata or {}),
            )
        except Exception as exc:
            logger.debug("Reasoning metric recording failed | %s | %s", metric_name, exc)

    def _metric_increment(
        self,
        metric_name: str,
        amount: float = 1.0,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        store = getattr(self, "metric_store", None)
        incrementer = getattr(store, "increment_counter", None)
        if callable(incrementer):
            try:
                incrementer(
                    metric_name,
                    float(amount),
                    category="reasoning",
                    unit="count",
                    metadata=dict(metadata or {}),
                )
                return
            except Exception as exc:
                logger.debug("Reasoning counter failed | %s | %s", metric_name, exc)
        self._metric_value(metric_name, amount, unit="count", metadata=metadata)

    def _metric_gauge(
        self,
        metric_name: str,
        value: float,
        *,
        unit: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        store = getattr(self, "metric_store", None)
        setter = getattr(store, "set_gauge", None)
        if callable(setter):
            try:
                setter(
                    metric_name,
                    float(value),
                    category="reasoning",
                    unit=unit,
                    metadata=dict(metadata or {}),
                )
                return
            except Exception as exc:
                logger.debug("Reasoning gauge failed | %s | %s", metric_name, exc)
        self._metric_value(metric_name, value, unit=unit, metadata=metadata)

    # ------------------------------------------------------------------
    # RuleEngine contract helpers
    # ------------------------------------------------------------------
    def _rule_entries(self) -> List[Any]:
        getter = getattr(self.rule_engine, "rule_entries", None)
        if callable(getter):
            value = getter()
            if isinstance(value, Iterable):
                return list(value)
        # Fact-level validation remains valid without executable-rule inspection.
        return []

    def _rule_weight_snapshot(self) -> Dict[str, float]:
        getter = getattr(self.rule_engine, "rule_weight_snapshot", None)
        if callable(getter):
            raw_snapshot = getter()
            if isinstance(raw_snapshot, Mapping):
                return {
                    str(name): clamp_confidence(weight)
                    for name, weight in raw_snapshot.items()
                }
            if isinstance(raw_snapshot, Iterable):
                return {
                    str(name): clamp_confidence(weight)
                    for name, weight in raw_snapshot
                }
            return {}
        return {
            str(item["name"]): clamp_confidence(item.get("weight", 0.0))
            for item in self.rule_engine.list_rules()
            if isinstance(item, Mapping) and item.get("name")
        }

    def _replace_working_knowledge(
        self,
        knowledge: Mapping[Fact, float],
        *,
        allow_contradiction: bool,
    ) -> None:
        replacer = getattr(self.rule_engine, "replace_knowledge", None)
        if callable(replacer):
            replacer(knowledge, allow_contradiction=allow_contradiction)
            return

        # Compatibility fallback uses public mutation APIs only.
        for fact in list(self.working_knowledge):
            self.rule_engine.retract_fact(fact)
        self.rule_engine.bulk_assert(
            knowledge,
            allow_contradiction=allow_contradiction,
        )

    def _restore_rule_weights(self, weights: Mapping[str, Any]) -> None:
        restorer = getattr(self.rule_engine, "restore_rule_weights", None)
        if not callable(restorer):
            raise KnowledgePersistenceError(
                "Active RuleEngine does not expose restore_rule_weights; "
                "apply the SLAI v2.3 Reasoning RuleEngine contract patch first."
            )
        restorer(weights)

    # ------------------------------------------------------------------
    # Checkpoint specialization
    # ------------------------------------------------------------------
    def _export_checkpoint_state(self) -> Mapping[str, Any]:
        return json_safe_reasoning_state(
            {
                "schema": self.CHECKPOINT_SCHEMA,
                "agent_version": __version__,
                "knowledge": [
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "confidence": confidence,
                    }
                    for (subject, predicate, obj), confidence
                    in sorted(self.working_knowledge.items(), key=lambda entry: entry[0])
                ],
                "rule_weights": self._rule_weight_snapshot(),
                "registered_rules": [
                    str(item.get("name"))
                    for item in self.rule_engine.list_rules()
                    if isinstance(item, Mapping) and item.get("name")
                ],
                "conflict_count": int(self.conflict_count),
                "last_forward_chaining_duration_seconds": float(
                    self.last_forward_chaining_duration_seconds
                ),
                # v2 checkpoint compatibility field.
                "forward_chaining_speed": float(
                    self.last_forward_chaining_duration_seconds
                ),
                "operation_counts": dict(self.operation_counts),
                "reasoning_history": list(self.reasoning_history),
            }
        )

    def _import_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != self.CHECKPOINT_SCHEMA:
            raise KnowledgePersistenceError(
                "Unsupported ReasoningAgent checkpoint schema.",
                context={
                    "expected": self.CHECKPOINT_SCHEMA,
                    "actual": state.get("schema"),
                },
            )

        restored_knowledge = self._normalize_knowledge_payload(
            state.get("knowledge", [])
        )
        raw_rule_weights = state.get("rule_weights", {})
        if not isinstance(raw_rule_weights, Mapping):
            raise KnowledgePersistenceError(
                "ReasoningAgent checkpoint rule_weights must be a mapping."
            )

        saved_rule_names = state.get("registered_rules", [])
        if (
            isinstance(saved_rule_names, Sequence)
            and not isinstance(saved_rule_names, (str, bytes))
        ):
            active_rule_names = {
                str(item.get("name"))
                for item in self.rule_engine.list_rules()
                if isinstance(item, Mapping) and item.get("name")
            }
            missing = {str(name) for name in saved_rule_names} - active_rule_names
            if missing:
                raise KnowledgePersistenceError(
                    "Reasoning checkpoint requires rules absent from the active runtime.",
                    context={"missing_rules": sorted(missing)},
                )

        # Restore subsystem-owned mutable state through subsystem APIs.  No
        # external publication/persistence is triggered from this hook.
        self._replace_working_knowledge(
            restored_knowledge,
            allow_contradiction=True,
        )
        self._restore_rule_weights(raw_rule_weights)

        self.conflict_count = int(state.get("conflict_count", 0))
        self.last_forward_chaining_duration_seconds = float(
            state.get(
                "last_forward_chaining_duration_seconds",
                state.get("forward_chaining_speed", 0.0),
            )
        )

        raw_counts = state.get("operation_counts", {})
        if not isinstance(raw_counts, Mapping):
            raise KnowledgePersistenceError(
                "ReasoningAgent checkpoint operation_counts must be a mapping."
            )
        self.operation_counts = Counter(
            {str(name): int(count) for name, count in raw_counts.items()}
        )

        raw_history = state.get("reasoning_history", [])
        if (
            not isinstance(raw_history, Sequence)
            or isinstance(raw_history, (str, bytes))
        ):
            raise KnowledgePersistenceError(
                "ReasoningAgent checkpoint reasoning_history must be a sequence."
            )
        self.reasoning_history = deque(
            (
                dict(item)
                for item in raw_history
                if isinstance(item, Mapping)
            ),
            maxlen=self.max_trace_items,
        )

    # ------------------------------------------------------------------
    # Public working-fact / rule API
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_fact(fact: Union[str, Sequence[Any]]) -> Fact:
        return normalize_fact(fact)

    def add_fact(
        self,
        fact: Union[str, Sequence[Any]],
        confidence: float = 1.0,
        *,
        publish: bool = True,
    ) -> bool:
        normalized = normalize_fact(fact)
        safe_confidence = clamp_confidence(confidence)

        try:
            self.rule_engine.assert_fact(normalized, safe_confidence)
        except ContradictionError:
            logger.warning("Rejected contradictory working fact: %s", normalized)
            return False

        self._publish_working_state(reason="add_fact", publish=publish)
        if publish:
            self._publish_knowledge_candidate(
                "assert",
                normalized,
                safe_confidence,
            )
        return True

    def add_rule(
        self,
        rule: Union[Callable[[Mapping[Fact, float]], Mapping[Fact, float]], List[Callable]],
        rule_name: Optional[str] = None,
        weight: float = 1.0,
    ) -> str:
        if isinstance(rule, list):
            last_name = ""
            for item in rule:
                last_name = self.add_rule(item, None, weight)
            return last_name

        resolved_name, safe_weight = validate_rule_registration(
            rule,
            rule_name,
            weight,
        )
        self.rule_engine.add_rule(
            rule,
            resolved_name,
            safe_weight,
            antecedents=[],
            consequents=[],
        )
        self._publish_working_state(reason="add_rule", publish=False)
        return resolved_name

    def forget_fact(self, fact: Union[str, Sequence[Any]]) -> bool:
        normalized = normalize_fact(fact)
        removed = self.rule_engine.retract_fact(normalized)
        if removed:
            self._publish_working_state(reason="forget_fact")
            self._shared_publish(
                self.knowledge_candidate_topic,
                {
                    "agent": self.name,
                    "agent_id": self.agent_id,
                    "operation": "retract",
                    "fact": normalized,
                    "timestamp": time.time(),
                },
            )
        return removed

    def forget_by_subject(self, subject: str) -> int:
        targets = list(self.rule_engine.query_subject(str(subject).strip()).keys())
        removed = 0
        for fact in targets:
            removed += int(self.rule_engine.retract_fact(fact))
        if removed:
            self._publish_working_state(reason="forget_by_subject")
        return removed

    def load_knowledge(self, knowledge: Mapping[Any, Any]) -> None:
        normalized = self._normalize_knowledge_payload(knowledge)
        self._replace_working_knowledge(
            normalized,
            allow_contradiction=False,
        )
        self._publish_working_state(reason="load_knowledge")

    def learn_from_interaction(
        self,
        fact_tuple: Union[str, Sequence[Any]],
        feedback: Mapping[Any, bool],
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """Record reasoning feedback without duplicating subsystem learning policy.

        The Agent may assert the interaction fact and publish explicit feedback,
        but it does not reinterpret boolean feedback as an algorithm-specific
        weight update.  Learning/tuning consumers can subscribe to the candidate
        stream and apply their own evidence-backed update policy.
        """
        start = time.time()
        added = self.add_fact(fact_tuple, confidence=confidence)
        records: List[Dict[str, Any]] = []

        for raw_fact, is_correct in feedback.items():
            normalized = normalize_fact(raw_fact)
            record = {
                "fact": normalized,
                "is_correct": bool(is_correct),
            }
            records.append(record)
            self._shared_publish(
                self.knowledge_candidate_topic,
                json_safe_reasoning_state(
                    {
                        "agent": self.name,
                        "agent_id": self.agent_id,
                        "operation": "feedback",
                        **record,
                        "timestamp": time.time(),
                    }
                ),
            )

        self._record_trace(
            AgentReasoningTrace(
                "learn_from_interaction",
                start,
                time.time(),
                "success",
                {"fact_added": added, "feedback_records": len(records)},
            )
        )
        return {
            "status": "success",
            "fact_added": added,
            "feedback_recorded": records,
            # Retained compatibility key; the Agent itself changed no learned
            # numeric weights.
            "updated": {},
        }

    # ------------------------------------------------------------------
    # Validation / probabilistic access
    # ------------------------------------------------------------------
    @staticmethod
    def _conflict_contains_fact(conflict: Any, fact: Fact) -> bool:
        if conflict == fact:
            return True
        if isinstance(conflict, Mapping):
            return any(
                ReasoningAgent._conflict_contains_fact(value, fact)
                for value in conflict.values()
            )
        if isinstance(conflict, (tuple, list, set)):
            return any(
                ReasoningAgent._conflict_contains_fact(value, fact)
                for value in conflict
            )
        return False

    def validate_fact(
        self,
        fact: Union[str, Sequence[Any]],
        threshold: float = 0.75,
    ) -> Dict[str, Any]:
        start = time.time()
        normalized = normalize_fact(fact)
        safe_threshold = clamp_confidence(threshold)
        confidence = clamp_confidence(
            self.rule_engine.query_fact(normalized) or 0.0
        )

        try:
            raw_validation = self.validation_engine.validate_all(
                rules=self._rule_entries(),
                new_facts={normalized: confidence},
            )
            validation_details = (
                dict(raw_validation)
                if isinstance(raw_validation, Mapping)
                else {
                    "validation_status": "failed",
                    "validation_error": (
                        "ValidationEngine returned unsupported result type: "
                        f"{type(raw_validation).__name__}"
                    ),
                }
            )
        except Exception as exc:
            logger.warning("ValidationEngine failed for %s: %s", normalized, exc)
            validation_details = {
                "validation_status": "failed",
                "validation_error": f"{type(exc).__name__}: {exc}",
            }

        validation_status = str(
            validation_details.get("validation_status", "unknown")
        ).strip().lower()
        if (
            "validation_error" in validation_details
            and validation_status not in {"failed", "partial"}
        ):
            validation_status = "failed"
        validation_complete = validation_status == "success"

        conflicts = validation_details.get("conflicts", [])
        has_conflict = any(
            self._conflict_contains_fact(item, normalized)
            for item in conflicts
        ) if isinstance(conflicts, list) else False

        probability = (
            self.probabilistic_query(normalized)
            if self.enable_probabilistic_fallback
            else confidence
        )

        base_valid = confidence >= safe_threshold and not has_conflict
        probability_valid = probability >= safe_threshold

        if not base_valid or not probability_valid:
            decision = "invalid"
        elif self.strict_fact_validation and not validation_complete:
            decision = "indeterminate"
        else:
            decision = "valid"

        payload = json_safe_reasoning_state(
            {
                "fact": normalized,
                "kb_confidence": confidence,
                "probabilistic_confidence": probability,
                "has_conflict": has_conflict,
                # v2.3 compatibility field.
                "is_valid": base_valid,
                "combined_valid": decision == "valid",
                "decision": decision,
                "validation_status": validation_status,
                "validation_complete": validation_complete,
                "strict_validation": self.strict_fact_validation,
                "validation_details": validation_details,
            }
        )

        if payload["combined_valid"]:
            self._shared_set(self.last_validation_key, payload)
        if has_conflict:
            self._metric_increment("conflicts_total")
        self._metric_value(
            "validation_complete",
            1.0 if validation_complete else 0.0,
        )
        self._record_trace(
            AgentReasoningTrace(
                "validate_fact",
                start,
                time.time(),
                "success",
                {
                    "decision": decision,
                    "validation_status": validation_status,
                },
            )
        )
        return payload

    def check_consistency(
        self,
        fact: Optional[Union[str, Sequence[Any]]] = None,
    ) -> bool:
        if fact is not None:
            return bool(
                self.validate_fact(
                    fact,
                    threshold=max(0.5, self.low_confidence_threshold),
                ).get("combined_valid", False)
            )
        conflicts = list(self.rule_engine.detect_fact_conflicts())
        self.conflict_count = len(conflicts)
        return not conflicts

    def probabilistic_query(
        self,
        fact: Union[str, Sequence[Any]],
        evidence: Optional[Mapping[Any, Any]] = None,
    ) -> float:
        normalized = normalize_fact(fact)
        try:
            return clamp_confidence(
                self.probabilistic_models.probabilistic_query(normalized, evidence)
            )
        except Exception as exc:
            if not self.enable_probabilistic_fallback:
                raise ModelInferenceError(
                    "ReasoningAgent probabilistic query failed.",
                    cause=exc,
                    context={"fact": normalized},
                ) from exc
            logger.debug("Probabilistic fallback used for %s: %s", normalized, exc)
            return clamp_confidence(
                self.rule_engine.query_fact(normalized) or 0.0
            )

    def multi_hop_reasoning(
        self,
        query: Union[str, Sequence[Any]],
        max_depth: int = 3,
    ) -> float:
        normalized = normalize_fact(query)
        depth = bounded_iterations(
            max_depth,
            minimum=1,
            maximum=self.max_chain_depth,
        )
        try:
            return clamp_confidence(
                self.probabilistic_models.multi_hop_reasoning(
                    normalized,
                    max_depth=depth,
                )
            )
        except Exception as exc:
            if not self.enable_probabilistic_fallback:
                raise ModelInferenceError(
                    "Multi-hop probabilistic reasoning failed.",
                    cause=exc,
                    context={"query": normalized, "max_depth": depth},
                ) from exc
            # Symbolic fallback delegates symbolic semantics to RuleEngine.
            self.forward_chaining(max_iterations=depth)
            return clamp_confidence(
                self.rule_engine.query_fact(normalized) or 0.0
            )

    # ------------------------------------------------------------------
    # Symbolic inference
    # ------------------------------------------------------------------
    def forward_chaining(self, max_iterations: Optional[int] = None) -> Dict[Fact, float]:
        return self.forward_chaining_report(max_iterations=max_iterations).added

    def forward_chaining_report(self, max_iterations: Optional[int] = None) -> ForwardChainReport:
        start = time.time()
        lock_key = f"locks:forward_chaining:{self.name}"
        inferred: Dict[Fact, float] = {}

        with DistributedLock(
            self.shared_memory,
            lock_key,
            timeout_seconds=30.0,
            acquire_timeout_seconds=30.0,
        ) as distributed_lock:
            inferred = self.rule_engine.run_inference(max_rounds=max_iterations, lease_guard=distributed_lock.assert_held)

        conflicts = list(self.rule_engine.detect_fact_conflicts())
        redundancies = list(self.rule_engine.redundant_fact_check())
        elapsed = time.time() - start

        self.conflict_count = len(conflicts)
        self.last_forward_chaining_duration_seconds = elapsed
        self.operation_counts["forward_chaining"] += 1
        self._publish_working_state(reason="forward_chaining")

        self._metric_value("forward_chaining_duration_seconds",elapsed, unit="seconds")
        self._metric_gauge("knowledge_base_size", len(self.working_knowledge), unit="facts")
        for name, weight in self._rule_weight_snapshot().items():
            self._metric_gauge("rule_weight",weight, metadata={"rule_name": name})

        report = ForwardChainReport(
            added=dict(inferred),
            iterations=int(getattr(self.rule_engine, "last_inference_rounds", 0)),
            conflicts=conflicts,
            redundancies=redundancies,
            duration_seconds=elapsed,
        )
        self._record_trace(
            AgentReasoningTrace(
                "forward_chaining",
                start,
                time.time(),
                "success",
                {
                    "added": len(inferred),
                    "iterations": report.iterations,
                    "conflicts": len(conflicts),
                },
            )
        )
        return report

    # ------------------------------------------------------------------
    # Typed reasoning / Phase 3 intelligence contract
    # ------------------------------------------------------------------
    def _resolve_reasoning_selection(
        self,
        reasoning_type: Optional[str],
        problem: Any = None,
    ) -> Tuple[str, Dict[str, Any]]:
        requested = str(reasoning_type or "auto").strip().lower()
        requested = self.reasoning_type_aliases.get(requested, requested)

        if requested not in {"", "auto", "default"}:
            return requested, {
                "requested": reasoning_type,
                "resolved": requested,
                "method": "explicit",
                "evidence": {},
            }

        selector = getattr(self.types, "select_reasoning_strategy", None)
        if callable(selector) and problem is not None:
            selection = selector(str(problem))
            if isinstance(selection, Mapping) and selection.get("strategy"):
                resolved = str(selection["strategy"]).strip().lower()
                return resolved, {
                    "requested": reasoning_type or "auto",
                    "resolved": resolved,
                    "method": str(selection.get("method", "subsystem_policy")),
                    "evidence": dict(selection.get("matches", {})),
                }

        determiner = getattr(self.types, "determine_reasoning_strategy", None)
        if callable(determiner) and problem is not None:
            resolved = str(determiner(str(problem))).strip().lower()
            return resolved, {
                "requested": reasoning_type or "auto",
                "resolved": resolved,
                "method": "subsystem_policy",
                "evidence": {},
            }

        return self.strategy_default, {
            "requested": reasoning_type or "auto",
            "resolved": self.strategy_default,
            "method": "agent_default",
            "evidence": {},
        }

    def _resolve_reasoning_type(
        self,
        reasoning_type: Optional[str],
        problem: Any = None,
    ) -> str:
        """v2.3 compatibility helper returning only the resolved strategy."""
        return self._resolve_reasoning_selection(reasoning_type, problem)[0]

    def _invoke_reasoning_engine(
        self,
        reasoning_engine: Any,
        problem: Any,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Compatibility wrapper; argument adaptation remains subsystem-owned."""
        return self.types.invoke_instance(reasoning_engine, problem, context)

    def reason(
        self,
        problem: Any,
        reasoning_type: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        start = time.time()
        resolved_type, selection = self._resolve_reasoning_selection(
            reasoning_type,
            problem,
        )

        canonical_reason = getattr(self.types, "reason", None)
        if not callable(canonical_reason):
            raise InferenceExecutionError(
                "ReasoningTypes does not expose the Phase 3 canonical reason() contract. "
                "Apply the SLAI v2.3 ReasoningTypes Phase 3 patch first."
            )

        try:
            result = canonical_reason(
                resolved_type,
                problem,
                context,
            )
            if not isinstance(result, Mapping):
                raise InferenceExecutionError(
                    "ReasoningTypes.reason() must return a mapping.",
                    context={"actual_type": type(result).__name__},
                )

            payload = dict(result)
            payload["selection"] = selection
            # Compatibility fields used by existing v2.3 callers.
            payload.setdefault("reasoning_type", resolved_type)
            payload.setdefault("result", payload.get("output"))
            payload.setdefault("status", "success")

            self.operation_counts["reason"] += 1
            self._metric_increment(
                "strategy_selected_total",
                metadata={
                    "strategy": resolved_type,
                    "selection_method": selection["method"],
                },
            )

            confidence = payload.get("confidence", {})
            if isinstance(confidence, Mapping) and confidence.get("value") is not None:
                self._metric_value(
                    "reasoning_confidence",
                    float(confidence["value"]),
                    metadata={
                        "strategy": resolved_type,
                        "confidence_type": str(confidence.get("type", "unknown")),
                        "calibrated": bool(confidence.get("calibrated", False)),
                    },
                )

            self._record_trace(
                AgentReasoningTrace(
                    "reason",
                    start,
                    time.time(),
                    "success",
                    {
                        "strategy": resolved_type,
                        "selection_method": selection["method"],
                        "outcome": payload.get("outcome"),
                        "degraded": bool(payload.get("degraded", False)),
                        "stop_reason": payload.get("stop_reason"),
                    },
                )
            )
            return json_safe_reasoning_state(payload)

        except ReasoningError as exc:
            self._record_trace(
                AgentReasoningTrace(
                    "reason",
                    start,
                    time.time(),
                    "failed",
                    {
                        "strategy": resolved_type,
                        "error_type": type(exc).__name__,
                    },
                )
            )
            raise
        except Exception as exc:
            self._record_trace(
                AgentReasoningTrace(
                    "reason",
                    start,
                    time.time(),
                    "failed",
                    {
                        "strategy": resolved_type,
                        "error_type": type(exc).__name__,
                    },
                )
            )
            raise InferenceExecutionError(
                "Typed reasoning execution failed.",
                cause=exc,
                context={"strategy": resolved_type},
            ) from exc

    def _refinement_reasons(self, result: Mapping[str, Any]) -> List[str]:
        reasons: List[str] = []

        if bool(result.get("degraded", False)):
            reasons.append("degraded_execution")

        if str(result.get("outcome", "")).lower() == "indeterminate":
            reasons.append("indeterminate_outcome")

        confidence = result.get("confidence")
        if isinstance(confidence, Mapping) and confidence.get("value") is not None:
            try:
                value = clamp_confidence(confidence["value"])
                if value < self.low_confidence_threshold:
                    reasons.append("low_confidence")
            except ReasoningError:
                reasons.append("invalid_confidence")

        contradictions = result.get("contradictions")
        if isinstance(contradictions, Sequence) and not isinstance(
            contradictions, (str, bytes)
        ) and contradictions:
            reasons.append("unresolved_contradictions")

        validation = result.get("validation")
        if isinstance(validation, Mapping):
            status = str(
                validation.get(
                    "validation_status",
                    validation.get("status", ""),
                )
            ).strip().lower()
            if status in {"failed", "partial", "indeterminate", "unavailable"}:
                reasons.append(f"validation_{status}")

        return list(dict.fromkeys(reasons))

    @staticmethod
    def _refinement_fingerprint(result: Mapping[str, Any]) -> str:
        confidence = result.get("confidence", {})
        compact = {
            "strategy": result.get("strategy"),
            "outcome": result.get("outcome"),
            "conclusion": result.get("conclusion"),
            "confidence": (
                confidence.get("value")
                if isinstance(confidence, Mapping)
                else confidence
            ),
            "contradictions": result.get("contradictions", []),
            "degraded": bool(result.get("degraded", False)),
        }
        return json.dumps(
            json_safe_reasoning_state(compact),
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

    def react_loop(
        self,
        problem: str,
        max_steps: Optional[int] = None,
        *,
        reasoning_type: Optional[str] = "auto",
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run bounded iterative refinement with explicit stop conditions.

        This is not a hidden chain-of-thought interface.  Each step is a normal
        canonical reasoning result.  Refinement continues only for observable
        uncertainty, degraded execution, unresolved contradiction, or incomplete
        validation, and terminates on resolution, no progress, or budget.
        """
        start = time.time()
        steps_limit = bounded_iterations(
            max_steps or self.max_react_steps,
            minimum=1,
            maximum=self.max_react_steps,
        )
        strategy = self._resolve_reasoning_type(reasoning_type, problem)
        working_context: Dict[str, Any] = dict(context or {})
        steps: List[Dict[str, Any]] = []
        previous_fingerprint: Optional[str] = None
        stop_reason = "budget_exhausted"
        resolved = False

        for index in range(1, steps_limit + 1):
            result = self.reason(
                problem,
                strategy,
                context=working_context,
            )
            refinement_reasons = self._refinement_reasons(result)
            fingerprint = self._refinement_fingerprint(result)

            steps.append(
                {
                    "step": index,
                    "result": result,
                    "refinement_reasons": refinement_reasons,
                }
            )

            if not refinement_reasons:
                stop_reason = "resolved"
                resolved = True
                break

            if previous_fingerprint is not None and fingerprint == previous_fingerprint:
                stop_reason = "no_progress"
                break

            previous_fingerprint = fingerprint
            working_context["previous_reasoning_result"] = result
            working_context["refinement"] = {
                "step": index,
                "reasons": refinement_reasons,
            }

        response = json_safe_reasoning_state(
            {
                "strategy": strategy,
                "steps": steps,
                "resolved": resolved,
                "stop_reason": stop_reason,
                "step_count": len(steps),
            }
        )
        self.operation_counts["react_loop"] += 1
        self._record_trace(
            AgentReasoningTrace(
                "react_loop",
                start,
                time.time(),
                "success",
                {
                    "steps": len(steps),
                    "strategy": strategy,
                    "resolved": resolved,
                    "stop_reason": stop_reason,
                },
            )
        )
        return response

    # ------------------------------------------------------------------
    # Actions / task execution
    # ------------------------------------------------------------------
    def _support_lookup(self, goal: Union[str, Sequence[Any]]) -> Dict[str, Any]:
        goal_fact = normalize_fact(goal)
        supporting = [
            fact
            for fact in self.working_knowledge
            if fact == goal_fact
            or fact[2] == goal_fact[0]
            or fact[0] == goal_fact[0]
            or fact[2] == goal_fact[2]
        ]
        return {
            "success": True,
            "mode": "support_lookup",
            "goal": goal_fact,
            "supporting_facts": supporting[: self.max_action_results],
        }

    def execute_action(self, action: str, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        action_key = str(action or "").strip().lower()
        data = dict(payload or {})

        if action_key == "query_knowledge_base":
            key = data.get("key")
            if key is None:
                return {
                    "success": True,
                    "results": list(self.working_knowledge.items())[
                        : self.max_action_results
                    ],
                }
            fact = normalize_fact(key)
            return {
                "success": True,
                "results": {fact: self.rule_engine.query_fact(fact) or 0.0},
            }

        if action_key == "run_consistency_check":
            return {
                "success": True,
                "consistent": self.check_consistency(data.get("fact")),
            }

        if action_key == "forward_chaining":
            report = self.forward_chaining_report(data.get("max_iterations"))
            return {"success": True, "report": report.to_dict()}

        if action_key in {"support_lookup", "backward_chaining"}:
            goal = data.get("goal")
            if goal is None:
                return {"success": False, "error": "goal is required"}
            result = self._support_lookup(goal)
            if action_key == "backward_chaining":
                result["compatibility_alias"] = "backward_chaining"
                result["note"] = (
                    "This v2.3 compatibility alias performs support lookup; "
                    "it does not claim goal-directed backward rule expansion."
                )
            return result

        if action_key == "probabilistic_query":
            if "fact" not in data:
                return {"success": False, "error": "fact is required"}
            return {
                "success": True,
                "probability": self.probabilistic_query(
                    data["fact"],
                    data.get("evidence"),
                ),
            }

        if action_key == "reason":
            if data.get("problem") is None:
                return {"success": False, "error": "problem is required"}
            return {
                "success": True,
                "result": self.reason(
                    data["problem"],
                    data.get("reasoning_type"),
                    data.get("context"),
                ),
            }

        if action_key == "react_loop":
            if data.get("problem") is None:
                return {"success": False, "error": "problem is required"}
            return {
                "success": True,
                "result": self.react_loop(
                    str(data["problem"]),
                    data.get("max_steps"),
                    reasoning_type=data.get("reasoning_type", "auto"),
                    context=data.get("context"),
                ),
            }

        if action_key == "request_human_input":
            request = {
                "timestamp": time.time(),
                "reason": data.get("reason", "low_confidence_reasoning"),
                "context": data.get("context", {}),
            }
            self._shared_append(self.human_intervention_key, request)
            return {"success": True, "request": request}

        return {"success": False, "error": f"Unknown action: {action}"}

    def perform_task(self, task_data: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(task_data, Mapping):
            raise ReasoningValidationError(
                "ReasoningAgent task_data must be a mapping.",
                context={"received_type": type(task_data).__name__},
            )

        payload = dict(task_data)
        task_type = str(payload.get("task_type", "forward_chaining")).strip().lower()
        supported = {
            "add_fact",
            "validate_fact",
            "probabilistic_query",
            "multi_hop_reasoning",
            "reason",
            "react_loop",
            "support_lookup",
            "execute_action",
            "forward_chaining",
        }
        if task_type not in supported:
            raise ReasoningValidationError(
                f"Unsupported reasoning task type: {task_type!r}.",
                context={"supported_task_types": sorted(supported)},
            )

        def require(field_name: str) -> Any:
            if field_name not in payload or payload[field_name] is None:
                raise ReasoningValidationError(
                    f"Reasoning task {task_type!r} requires field {field_name!r}.",
                    context={"task_type": task_type, "missing_field": field_name},
                )
            return payload[field_name]

        if task_type == "add_fact":
            return {
                "status": "success",
                "task_type": task_type,
                "added": self.add_fact(
                    require("fact"),
                    payload.get("confidence", 1.0),
                ),
            }

        if task_type == "validate_fact":
            return self.validate_fact(
                require("fact"),
                payload.get("threshold", 0.75),
            )

        if task_type == "probabilistic_query":
            return {
                "status": "success",
                "task_type": task_type,
                "probability": self.probabilistic_query(
                    require("fact"),
                    payload.get("evidence"),
                ),
            }

        if task_type == "multi_hop_reasoning":
            return {
                "status": "success",
                "task_type": task_type,
                "score": self.multi_hop_reasoning(
                    require("query"),
                    payload.get("max_depth", 3),
                ),
            }

        if task_type == "reason":
            return self.reason(
                require("problem"),
                payload.get("reasoning_type"),
                payload.get("context"),
            )

        if task_type == "react_loop":
            return self.react_loop(
                str(require("problem")),
                payload.get("max_steps"),
                reasoning_type=payload.get("reasoning_type", "auto"),
                context=payload.get("context"),
            )

        if task_type == "support_lookup":
            return self._support_lookup(require("goal"))

        if task_type == "execute_action":
            return self.execute_action(
                payload.get("action", "query_knowledge_base"),
                payload.get("payload"),
            )

        report = self.forward_chaining_report(payload.get("max_iterations"))
        return {
            "status": "success",
            "task_type": "forward_chaining",
            "report": report.to_dict(),
        }

    def stream_update(
        self,
        new_facts: Iterable[Union[str, Sequence[Any]]],
        confidence: float = 1.0,
        *,
        max_inference_rounds: Optional[int] = 2,
    ) -> Dict[str, Any]:
        safe_confidence = clamp_confidence(confidence)
        batch: Dict[Fact, float] = {}
        for raw_fact in new_facts:
            fact = normalize_fact(raw_fact)
            batch[fact] = max(batch.get(fact, 0.0), safe_confidence)

        if not batch:
            return {"added": 0, "skipped": 0, "inferred": 0}

        accepted, skipped = self.rule_engine.bulk_assert(batch)
        inferred = self.forward_chaining(max_iterations=max_inference_rounds)
        self._publish_working_state(reason="stream_update")

        for fact in batch:
            if fact in self.working_knowledge:
                self._publish_knowledge_candidate(
                    "stream_assert",
                    fact,
                    self.working_knowledge[fact],
                )

        return {
            "added": accepted,
            "skipped": skipped,
            "inferred": len(inferred),
        }

    # ------------------------------------------------------------------
    # Probabilistic / hybrid compatibility surface
    # ------------------------------------------------------------------
    def run_bayesian_learning(self, observations: List[Any]) -> Any:
        runner = getattr(self.probabilistic_models, "run_bayesian_learning_cycle", None)
        if not callable(runner):
            raise ModelInferenceError("ProbabilisticModels does not expose run_bayesian_learning_cycle.")
        return runner(observations)

    def get_probability_grid(self, agent_pos: Any = None, target_pos: Any = None) -> Any:
        getter = getattr(self.probabilistic_models, "get_probability_grid", None)
        return (
            getter(agent_pos=agent_pos, target_pos=target_pos)
            if callable(getter)
            else []
        )

    # ------------------------------------------------------------------
    # Explainable evidence-path helpers
    # ------------------------------------------------------------------
    def generate_reasoning_path(self, query: Union[str, Sequence[Any]], depth: int = 3) -> List[Dict[str, Any]]:
        """Return a bounded path through explicit working facts.

        This is an evidence/provenance path, not hidden model chain-of-thought.
        """
        fact = normalize_fact(query)
        max_depth = bounded_iterations(depth, minimum=1, maximum=self.max_chain_depth)
        path: List[Dict[str, Any]] = []
        current = [fact]
        visited: set[Fact] = set()

        for step_index in range(max_depth):
            if not current:
                break
            next_facts: List[Fact] = []
            for current_fact in current:
                if current_fact in visited:
                    continue
                visited.add(current_fact)
                path.append(
                    {
                        "step": step_index + 1,
                        "fact": current_fact,
                        "confidence": clamp_confidence(
                            self.rule_engine.query_fact(current_fact) or 0.0
                        ),
                    }
                )
                for candidate in self.working_knowledge:
                    if (
                        candidate[0] == current_fact[2]
                        or candidate[2] == current_fact[0]
                    ):
                        next_facts.append(candidate)
            current = next_facts[: self.max_action_results]

        return [json_safe_reasoning_state(item) for item in path]

    def generate_chain_of_thought(self, query: Union[str, Sequence[Any]], depth: int = 3) -> List[str]:
        """Legacy alias for explicit evidence-path display.

        The name is retained for v2.3 caller compatibility only; the returned
        content is derived from stored facts and is not private chain-of-thought.
        """
        path = self.generate_reasoning_path(query, depth)
        return [
            f"Step {item['step']}: {tuple(item['fact'])} @ {float(item['confidence']):.3f}"
            for item in path
        ]

    # ------------------------------------------------------------------
    # Context / diagnostics
    # ------------------------------------------------------------------
    def parse_goal(self, goal_description: str) -> Dict[str, Any]:
        text = str(goal_description or "")
        tokens = text.lower().split()
        return {
            "raw": text,
            "reasoning_type": self._resolve_reasoning_type("auto", text),
            "contains_uncertainty": any(
                word in tokens
                for word in ("maybe", "likely", "uncertain", "probable")
            ),
            "contains_constraint": any(
                word in tokens
                for word in ("must", "should", "cannot", "never")
            ),
        }

    def get_current_context(self) -> List[str]:
        context: List[str] = []
        if len(self.working_knowledge) > self.large_kb_threshold:
            context.append("large_knowledge_base")
        if any(
            confidence < self.low_confidence_threshold
            for confidence in self.working_knowledge.values()
        ):
            context.append("low_confidence_environment")
        if self.conflict_count > 0:
            context.append("conflict_detected")
        if not self.rule_engine.list_rules():
            context.append("no_symbolic_rules_registered")
        return context

    def predict(self, state: Any = None) -> Dict[str, Any]:
        confidences = list(self.working_knowledge.values())
        return {
            "knowledge_size": len(self.working_knowledge),
            "rule_count": len(self.rule_engine.list_rules()),
            "context": self.get_current_context(),
            "confidence_mean": (
                sum(confidences) / len(confidences)
                if confidences
                else 0.0
            ),
            "conflict_count": self.conflict_count,
            "state": state,
        }

    @staticmethod
    def _safe_component_diagnostics(component: Any) -> Optional[Dict[str, Any]]:
        diagnostic = getattr(component, "diagnostics", None)
        if not callable(diagnostic):
            return None
        try:
            value = diagnostic()
            return dict(value) if isinstance(value, Mapping) else {"value": value}
        except Exception as exc:
            return {"status": "degraded", "error": f"{type(exc).__name__}: {exc}"}

    def diagnostics(self) -> Dict[str, Any]:
        return json_safe_reasoning_state(
            {
                "agent": self.name,
                "version": __version__,
                "knowledge_size": len(self.working_knowledge),
                "rule_count": len(self.rule_engine.list_rules()),
                "rule_weights": self._rule_weight_snapshot(),
                "conflict_count": self.conflict_count,
                "last_forward_chaining_duration_seconds": (
                    self.last_forward_chaining_duration_seconds
                ),
                "operation_counts": dict(self.operation_counts),
                "history_size": len(self.reasoning_history),
                "reasoning_types": self.types.get_stats(),
                "components": {
                    "types": type(self.types).__name__,
                    "hybrid_models": type(self.hybrid_models).__name__,
                    "probabilistic_models": type(self.probabilistic_models).__name__,
                    "rule_engine": type(self.rule_engine).__name__,
                    "validation_engine": type(self.validation_engine).__name__,
                },
                "component_diagnostics": {
                    "rule_engine": self._safe_component_diagnostics(self.rule_engine),
                    "probabilistic_models": self._safe_component_diagnostics(
                        self.probabilistic_models
                    ),
                },
            }
        )

    def health_check(self) -> Dict[str, Any]:
        diagnostics = self.diagnostics()
        healthy = all(
            component is not None
            for component in (
                self.types,
                self.rule_engine,
                self.validation_engine,
                self.probabilistic_models,
                self.hybrid_models,
            )
        )
        return {"healthy": healthy, "diagnostics": diagnostics}

    def __repr__(self) -> str:
        return (
            f"ReasoningAgent(kb={len(self.working_knowledge)}, "
            f"rules={len(self.rule_engine.list_rules())}, "
            f"conflicts={self.conflict_count})"
        )


if __name__ == "__main__":
    print("\n=== Running Reasoning Agent ===\n")
    printer.status("TEST", "Reasoning Agent initialized", "info")

    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    agent = ReasoningAgent(
        shared_memory=shared_memory,
        agent_factory=agent_factory,
    )

    assert agent.add_fact(("Apple", "is", "Fruit"), 0.9)
    assert agent.add_fact(("Fruit", "is", "Healthy"), 0.85)

    inferred = agent.forward_chaining(max_iterations=3)
    validation = agent.validate_fact(("Apple", "is", "Fruit"), threshold=0.5)
    reasoned = agent.reason(
        "Socrates is mortal",
        "deduction",
        {
            "premises": ["all humans are mortal", "Socrates is human"],
            "hypothesis": "Socrates is mortal",
        },
    )
    action = agent.execute_action(
        "query_knowledge_base",
        {"key": ("Apple", "is", "Fruit")},
    )
    stream = agent.stream_update(
        [("Banana", "is", "Fruit")],
        confidence=0.8,
    )
    health = agent.health_check()

    assert isinstance(inferred, dict)
    assert validation.get("decision") in {"valid", "invalid", "indeterminate"}
    assert reasoned.get("status") == "success"
    assert reasoned.get("schema") == "slai.reasoning.result.v1"
    assert action.get("success") is True
    assert stream["added"] >= 0
    assert health["healthy"] is True

    print("\n=== Test ran successfully ===\n")
