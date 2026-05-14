from __future__ import annotations

__version__ = "2.2.0"

"""
Production Reasoning Agent for SLAI.

The agent is a thin, production-oriented facade over the reasoning subsystem. It
owns only agent-level runtime policy from ``agents_config.yaml`` and delegates
specialized reasoning work to ``src/agents/reasoning`` components:

- ``ReasoningTypes`` for single/combined reasoning strategies.
- ``HybridProbabilisticModels`` for hybrid Bayesian/grid construction.
- ``ProbabilisticModels`` for Bayesian/neural probabilistic inference.
- ``RuleEngine`` for symbolic resources and linguistic rule support.
- ``ValidationEngine`` for consistency and confidence validation.

Design constraints honored here:
- agent config is loaded only through ``main_config_loader`` from
  ``agents_config.yaml``;
- subsystem-specific configuration remains inside ``src/agents/reasoning``;
- local imports are direct and are not wrapped in ``try/except``;
- shared reasoning helpers/errors are reused instead of copied;
- BaseAgent remains responsible for generic execution envelope behavior.
"""

import inspect
import json
import os
import tempfile
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from .base_agent import BaseAgent
from .base.utils.config_contract import assert_valid_config_contract
from .base.utils.main_config_loader import get_config_section, load_global_config
from .reasoning import *
from .reasoning.utils.reasoning_errors import *
from .reasoning.utils.reasoning_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reasoning Agent")
printer = PrettyPrinter()


@dataclass(frozen=True)
class AgentReasoningTrace:
    """Compact audit payload for one reasoning-agent operation."""

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
    """Structured result from symbolic forward chaining."""

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


class ReasoningAgent(BaseAgent):
    """Production facade for symbolic, probabilistic, and typed reasoning.

    The agent deliberately does not duplicate the reasoning subsystem. It keeps a
    coordinated agent-level knowledge view, publishes lifecycle state to shared
    memory, and calls the subsystem components through their public APIs.
    """

    AGENT_KEY = "reasoning_agent"
    KNOWLEDGE_MEMORY_KEY_DEFAULT = "reasoning_agent:knowledge_base"

    _ALLOWED_CONFIG_KEYS = {
        "learning_rate", "decay", "exploration_rate", "max_iterations",
        "contradiction_threshold", "redundancy_margin", "knowledge_db",
        "large_kb_threshold", "low_confidence_threshold", "max_action_results",
        "max_chain_depth", "max_react_steps", "max_trace_items",
        "enable_shared_memory_publish", "enable_knowledge_persistence",
        "enable_memory_logging", "enable_probabilistic_fallback",
        "auto_register_builtin_rules", "strict_fact_validation",
        "human_intervention_key", "knowledge_memory_key", "last_validation_key",
        "reasoning_trace_topic", "memory_event_tag", "memory_event_priority",
        "strategy_default", "reasoning_type_aliases", "builtin_rule_weights",
        "tuple_key", "hypothesis_graph", "glove_path", "ner_tag", "embedding",
    }

    def __init__(self, shared_memory: Any, agent_factory: Any, config: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self._reasoning_lock = RLock()
        self.config: Dict[str, Any] = load_global_config()
        self.agent_config: Dict[str, Any] = dict(get_config_section(self.AGENT_KEY) or {})
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

        # Subsystem components loaded from ``.reasoning`` as requested. Their
        # own configuration stays inside the reasoning subsystem.
        self.types = ReasoningTypes()
        self.hybrid_models = HybridProbabilisticModels()
        self.probabilistic_models = ProbabilisticModels()
        self.rule_engine = RuleEngine()
        self.validation_engine = ValidationEngine()

        self.reasoning_strategies = self.types
        self.hybrid_probabilistic_models = self.hybrid_models

        self._link_components()

        self.rules: List[RuleEntry] = []
        self.rule_weights: RuleWeightMap = {}
        self.knowledge_base: Dict[Fact, float] = self._load_initial_knowledge()
        self.conflict_count = 0
        self.forward_chaining_speed = 0.0
        self.operation_counts: Counter[str] = Counter()
        self.reasoning_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_trace_items)

        if self.auto_register_builtin_rules:
            weights = self.builtin_rule_weights
            self.add_rule(self.identity_rule, "identity_rule", weights.get("identity_rule", 1.0))
            self.add_rule(self.transitive_rule, "transitive_rule", weights.get("transitive_rule", 0.8))

        self._sync_component_state()
        self._persist_state(reason="initialized", publish=False)
        logger.info(
            "ReasoningAgent initialized | kb=%s | rules=%s | max_iterations=%s",
            len(self.knowledge_base),
            len(self.rules),
            self.max_iterations,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _load_runtime_config(self) -> None:
        cfg = self.agent_config
        self.learning_rate = clamp_confidence(cfg.get("learning_rate", 0.05))
        self.decay = clamp_confidence(cfg.get("decay", 0.95))
        self.exploration_rate = clamp_confidence(cfg.get("exploration_rate", 0.1))
        self.max_iterations = bounded_iterations(cfg.get("max_iterations", 20), minimum=1, maximum=100_000)
        self.contradiction_threshold = clamp_confidence(cfg.get("contradiction_threshold", 0.25))
        self.redundancy_margin = clamp_confidence(cfg.get("redundancy_margin", 0.05))
        self.knowledge_db = str(cfg.get("knowledge_db", "src/agents/knowledge/templates/knowledge_db.json"))
        self.large_kb_threshold = bounded_iterations(cfg.get("large_kb_threshold", 500), minimum=1, maximum=50_000_000)
        self.max_action_results = bounded_iterations(cfg.get("max_action_results", 20), minimum=1, maximum=100_000)
        self.max_chain_depth = bounded_iterations(cfg.get("max_chain_depth", 4), minimum=1, maximum=256)
        self.max_react_steps = bounded_iterations(cfg.get("max_react_steps", 5), minimum=1, maximum=256)
        self.max_trace_items = bounded_iterations(cfg.get("max_trace_items", 250), minimum=1, maximum=100_000)
        self.low_confidence_threshold = clamp_confidence(cfg.get("low_confidence_threshold", 0.4))
        self.enable_shared_memory_publish = bool(cfg.get("enable_shared_memory_publish", True))
        self.enable_knowledge_persistence = bool(cfg.get("enable_knowledge_persistence", True))
        self.enable_memory_logging = bool(cfg.get("enable_memory_logging", True))
        self.enable_probabilistic_fallback = bool(cfg.get("enable_probabilistic_fallback", True))
        self.auto_register_builtin_rules = bool(cfg.get("auto_register_builtin_rules", True))
        self.strict_fact_validation = bool(cfg.get("strict_fact_validation", True))
        self.knowledge_memory_key = str(cfg.get("knowledge_memory_key", self.KNOWLEDGE_MEMORY_KEY_DEFAULT))
        self.last_validation_key = str(cfg.get("last_validation_key", "reasoning_agent:last_validated_fact"))
        self.reasoning_trace_topic = str(cfg.get("reasoning_trace_topic", "reasoning_trace"))
        self.human_intervention_key = str(cfg.get("human_intervention_key", "human_intervention_requests"))
        self.memory_event_tag = str(cfg.get("memory_event_tag", "reasoning_agent"))
        self.memory_event_priority = clamp_confidence(cfg.get("memory_event_priority", 0.75))
        self.strategy_default = str(cfg.get("strategy_default", "deduction")).strip() or "deduction"
        self.reasoning_type_aliases = {str(k).strip(): str(v).strip() for k, v in dict(cfg.get("reasoning_type_aliases", {})).items()}
        self.builtin_rule_weights = {
            str(k).strip(): clamp_confidence(v)
            for k, v in dict(cfg.get("builtin_rule_weights", {"identity_rule": 1.0, "transitive_rule": 0.8})).items()
        }

    def _validate_runtime_config(self) -> None:
        if self.learning_rate <= 0.0:
            raise ReasoningConfigurationError("reasoning_agent.learning_rate must be positive")
        if not self.knowledge_memory_key:
            raise ReasoningConfigurationError("reasoning_agent.knowledge_memory_key cannot be empty")
        if not self.reasoning_trace_topic:
            raise ReasoningConfigurationError("reasoning_agent.reasoning_trace_topic cannot be empty")

    # ------------------------------------------------------------------
    # Shared-memory and component integration
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
        current = self._shared_get(key, default=[])
        if not isinstance(current, list):
            current = [current]
        current.append(payload)
        self._shared_set(key, current)

    def _link_components(self) -> None:
        for component in (self.probabilistic_models, self.hybrid_models, self.rule_engine, self.validation_engine, self.types):
            linker = getattr(component, "link_agent", None)
            if callable(linker):
                linker(self)

    def _sync_component_state(self) -> None:
        for component in (self.rule_engine, self.validation_engine, self.probabilistic_models):
            if hasattr(component, "knowledge_base"):
                setattr(component, "knowledge_base", self.knowledge_base)
        if hasattr(self.rule_engine, "rule_weights"):
            self.rule_engine.rule_weights.update(self.rule_weights)

    def _remember(self, payload: Mapping[str, Any], *, tag: Optional[str] = None, priority: Optional[float] = None) -> None:
        if not self.enable_memory_logging or self.shared_memory is None:
            return
        add = getattr(self.shared_memory, "add", None)
        if callable(add):
            try:
                add(
                    experience=json_safe_reasoning_state(dict(payload)),
                    tag=tag or self.memory_event_tag,
                    priority=self.memory_event_priority if priority is None else clamp_confidence(priority),
                )
            except Exception as exc:
                logger.warning("Reasoning memory logging failed: %s", exc)

    def _record_trace(self, trace: AgentReasoningTrace) -> None:
        payload = trace.to_dict()
        self.reasoning_history.append(payload)
        self._shared_publish(self.reasoning_trace_topic, {"agent": self.name, "trace": payload})
        self._remember(payload, tag=self.memory_event_tag, priority=self.memory_event_priority)

    # ------------------------------------------------------------------
    # Knowledge loading and persistence
    # ------------------------------------------------------------------
    def _load_initial_knowledge(self) -> Dict[Fact, float]:
        kb = self._normalize_knowledge_payload(self._shared_get(self.knowledge_memory_key, default={}))
        if kb:
            return kb
        path = Path(self.knowledge_db)
        if path.exists():
            try:
                return self._normalize_knowledge_payload(json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:
                if self.strict_fact_validation:
                    raise KnowledgePersistenceError("Failed to load reasoning-agent knowledge DB", cause=exc, context={"path": str(path)}) from exc
                logger.warning("Skipping unreadable reasoning-agent knowledge DB %s: %s", path, exc)
        return {}

    def _normalize_knowledge_payload(self, raw: Any) -> Dict[Fact, float]:
        normalized: Dict[Fact, float] = {}
        if not raw:
            return normalized
        items: Iterable[Any]
        if isinstance(raw, Mapping):
            items = raw.get("knowledge", raw.items()) if "knowledge" in raw else raw.items()
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            items = raw
        else:
            return normalized

        for item in items:
            try:
                if isinstance(item, Mapping):
                    fact = (item.get("subject", ""), item.get("predicate", ""), item.get("object", ""))
                    conf = item.get("confidence", item.get("weight", 0.0))
                elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (tuple, list, str)):
                    fact, conf = item
                elif isinstance(item, (tuple, list)) and len(item) >= 4:
                    fact, conf = (item[0], item[1], item[2]), item[3]
                else:
                    continue
                normalized[normalize_fact(fact)] = clamp_confidence(conf)
            except ReasoningError:
                if self.strict_fact_validation:
                    raise
            except Exception as exc:
                if self.strict_fact_validation:
                    raise FactNormalizationError("Invalid knowledge item", cause=exc, context={"item": item}) from exc
        return normalized

    def _serialize_knowledge_payload(self, *, reason: str) -> Dict[str, Any]:
        return json_safe_reasoning_state(
            {
                "knowledge": [
                    {"subject": s, "predicate": p, "object": o, "confidence": c, "weight": c}
                    for (s, p, o), c in sorted(self.knowledge_base.items(), key=lambda entry: entry[0])
                ],
                "rules": [
                    {"name": name, "callable": getattr(rule_fn, "__name__", "anonymous"), "weight": self.rule_weights.get(name, weight)}
                    for name, rule_fn, weight in self.rules
                ],
                "reason": reason,
                "updated_at": time.time(),
            }
        )

    def _persist_state(self, *, reason: str = "state_update", publish: bool = True) -> None:
        payload = self._serialize_knowledge_payload(reason=reason)
        self._shared_set(self.knowledge_memory_key, self.knowledge_base)
        if publish:
            self._shared_publish("reasoning_agent:state_updated", payload)

        if not self.enable_knowledge_persistence or not self.knowledge_db:
            return
        path = Path(self.knowledge_db)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
                json.dump(payload, tmp, indent=2, ensure_ascii=False)
                temp_path = tmp.name
            os.replace(temp_path, path)
        except Exception as exc:
            raise KnowledgePersistenceError("Failed to persist reasoning-agent knowledge", cause=exc, context={"path": str(path)}) from exc

    # ------------------------------------------------------------------
    # Built-in symbolic rules
    # ------------------------------------------------------------------
    @staticmethod
    def identity_rule(kb: Dict[Fact, float]) -> Dict[Fact, float]:
        """Return identity/is-a facts as stable baseline inferences."""
        return {(s, p, o): conf for (s, p, o), conf in kb.items() if p == "is"}

    @staticmethod
    def transitive_rule(kb: Dict[Fact, float]) -> Dict[Fact, float]:
        """Infer A is C from A is B and B is C."""
        inferred: Dict[Fact, float] = {}
        facts = list(kb.items())
        for (a, p1, b1), c1 in facts:
            if p1 != "is":
                continue
            for (b2, p2, c), c2 in facts:
                if p2 == "is" and b1 == b2 and a != c:
                    inferred[(a, "is", c)] = max(inferred.get((a, "is", c), 0.0), min(float(c1), float(c2)))
        return inferred

    # ------------------------------------------------------------------
    # Public knowledge and rule API
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_fact(fact: Union[str, Sequence[Any]]) -> Fact:
        return normalize_fact(fact)

    def add_fact(self, fact: Union[str, Sequence[Any]], confidence: float = 1.0, *, publish: bool = True) -> bool:
        with self._reasoning_lock:
            normalized = normalize_fact(fact)
            safe_confidence = clamp_confidence(confidence)
            try:
                ensure_non_contradictory(
                    normalized,
                    self.knowledge_base,
                    threshold=self.contradiction_threshold,
                    source="reasoning_agent",
                )
            except ContradictionError:
                logger.warning("Rejected contradictory fact: %s", normalized)
                return False

            existing = self.knowledge_base.get(normalized, 0.0)
            self.knowledge_base[normalized] = merge_confidence(existing, safe_confidence)
            self._sync_component_state()
            self._persist_state(reason="add_fact", publish=publish)
            if publish:
                self._shared_publish("new_facts", (normalized, safe_confidence))
            return True

    def add_rule(self, rule: Union[Callable[[Dict[Fact, float]], Dict[Fact, float]], List[Callable]], rule_name: Optional[str] = None, weight: float = 1.0) -> str:
        if isinstance(rule, list):
            registered = ""
            for item in rule:
                registered = self.add_rule(item, None, weight)
            return registered

        resolved_name, safe_weight = validate_rule_registration(rule, rule_name, weight)
        with self._reasoning_lock:
            self.rules = [entry for entry in self.rules if entry[0] != resolved_name]
            self.rules.append((resolved_name, rule, safe_weight))
            self.rule_weights[resolved_name] = safe_weight
            self._register_rule_with_component(resolved_name, rule, safe_weight)
            self._persist_state(reason="add_rule", publish=False)
            return resolved_name

    def _register_rule_with_component(self, name: str, rule: Callable, weight: float) -> None:
        add_rule = getattr(self.rule_engine, "add_rule", None)
        if not callable(add_rule):
            return
        try:
            params = inspect.signature(add_rule).parameters
            if "antecedents" in params and "consequents" in params:
                add_rule(rule, name, weight, [], [])
            else:
                add_rule(rule, name, weight)
        except Exception as exc:
            logger.debug("RuleEngine registration skipped for %s: %s", name, exc)

    def forget_fact(self, fact: Union[str, Sequence[Any]]) -> bool:
        with self._reasoning_lock:
            normalized = normalize_fact(fact)
            if normalized not in self.knowledge_base:
                return False
            del self.knowledge_base[normalized]
            self._sync_component_state()
            self._persist_state(reason="forget_fact")
            return True

    def forget_by_subject(self, subject: str) -> int:
        subject_key = str(subject).strip()
        with self._reasoning_lock:
            targets = [fact for fact in self.knowledge_base if fact[0] == subject_key]
            for fact in targets:
                del self.knowledge_base[fact]
            if targets:
                self._sync_component_state()
                self._persist_state(reason="forget_by_subject")
            return len(targets)

    def load_knowledge(self, knowledge: Mapping[Any, Any]) -> None:
        with self._reasoning_lock:
            self.knowledge_base = self._normalize_knowledge_payload(knowledge)
            self._sync_component_state()
            self._persist_state(reason="load_knowledge")

    def learn_from_interaction(self, fact_tuple: Union[str, Sequence[Any]], feedback: Mapping[Any, bool], confidence: float = 1.0) -> Dict[str, Any]:
        start = time.time()
        self.add_fact(fact_tuple, confidence=confidence)
        updated: Dict[Fact, float] = {}
        with self._reasoning_lock:
            for fact, is_correct in feedback.items():
                normalized = normalize_fact(fact)
                current = self.knowledge_base.get(normalized, 0.0)
                updated_conf = update_rule_weight(current, success=bool(is_correct), learning_rate=self.learning_rate, decay=self.decay)
                self.knowledge_base[normalized] = updated_conf
                updated[normalized] = updated_conf
            self._sync_component_state()
            self._persist_state(reason="learn_from_interaction")
        trace = AgentReasoningTrace("learn_from_interaction", start, time.time(), "success", {"updated": len(updated)})
        self._record_trace(trace)
        return {"status": "success", "updated": updated}

    # ------------------------------------------------------------------
    # Validation and inference
    # ------------------------------------------------------------------
    def validate_fact(self, fact: Union[str, Sequence[Any]], threshold: float = 0.75) -> Dict[str, Any]:
        start = time.time()
        normalized = normalize_fact(fact)
        safe_threshold = clamp_confidence(threshold)
        confidence = self.knowledge_base.get(normalized, 0.0)
        validation_details: Dict[str, Any] = {}
        try:
            validator = getattr(self.validation_engine, "validate_all", None)
            if callable(validator):
                validation_details = validator(rules=self.rules, new_facts={normalized: confidence or safe_threshold})
        except Exception as exc:
            validation_details = {"validation_error": f"{type(exc).__name__}: {exc}"}
            logger.warning("ValidationEngine failed for %s: %s", normalized, exc)

        conflicts = validation_details.get("conflicts", []) if isinstance(validation_details, Mapping) else []
        has_conflict = any(normalized in pair for pair in conflicts if isinstance(pair, (tuple, list, set)))
        probability = self.probabilistic_query(normalized) if self.enable_probabilistic_fallback else confidence
        is_valid = confidence >= safe_threshold and not has_conflict
        payload = json_safe_reasoning_state(
            {
                "fact": normalized,
                "kb_confidence": confidence,
                "probabilistic_confidence": probability,
                "has_conflict": has_conflict,
                "is_valid": is_valid,
                "combined_valid": is_valid and probability >= safe_threshold,
                "validation_details": validation_details,
            }
        )
        if payload["combined_valid"]:
            self._shared_set(self.last_validation_key, payload)
        self._record_trace(AgentReasoningTrace("validate_fact", start, time.time(), "success", {"valid": payload["combined_valid"]}))
        return payload

    def check_consistency(self, fact: Optional[Union[str, Sequence[Any]]] = None) -> bool:
        if fact is not None:
            return bool(self.validate_fact(fact, threshold=max(0.5, self.low_confidence_threshold)).get("combined_valid", False))
        self._sync_component_state()
        try:
            conflicts = self.rule_engine.detect_fact_conflicts(self.contradiction_threshold)
        except Exception:
            conflicts = conflict_pairs(self.knowledge_base)
        self.conflict_count = len(conflicts)
        return not bool(conflicts)

    def probabilistic_query(self, fact: Union[str, Sequence[Any]], evidence: Optional[Mapping[Any, Any]] = None) -> float:
        normalized = normalize_fact(fact)
        try:
            return clamp_confidence(self.probabilistic_models.probabilistic_query(normalized, evidence))
        except Exception as exc:
            if not self.enable_probabilistic_fallback:
                raise ModelInferenceError("ReasoningAgent probabilistic query failed", cause=exc, context={"fact": normalized}) from exc
            logger.debug("Probabilistic fallback used for %s: %s", normalized, exc)
            return clamp_confidence(self.knowledge_base.get(normalized, 0.0))

    def multi_hop_reasoning(self, query: Union[str, Sequence[Any]], max_depth: int = 3) -> float:
        normalized = normalize_fact(query)
        depth = bounded_iterations(max_depth, minimum=1, maximum=self.max_chain_depth)
        try:
            return clamp_confidence(self.probabilistic_models.multi_hop_reasoning(normalized, max_depth=depth))
        except Exception:
            return self._multi_hop_symbolic_score(normalized, depth)

    def _multi_hop_symbolic_score(self, query: Fact, max_depth: int) -> float:
        target_s, target_p, target_o = query
        frontier: List[Tuple[str, float, int]] = [(target_s, 1.0, 0)]
        visited = {target_s}
        best = self.knowledge_base.get(query, 0.0)
        while frontier:
            subject, confidence, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for (s, p, o), fact_conf in self.knowledge_base.items():
                if s != subject or p != target_p:
                    continue
                score = confidence * fact_conf * (self.decay ** depth)
                if o == target_o:
                    best = max(best, score)
                if o not in visited:
                    visited.add(o)
                    frontier.append((o, score, depth + 1))
        return clamp_confidence(best)

    def forward_chaining(self, max_iterations: Optional[int] = None) -> Dict[Fact, float]:
        report = self.forward_chaining_report(max_iterations=max_iterations)
        return report.added

    def forward_chaining_report(self, max_iterations: Optional[int] = None) -> ForwardChainReport:
        start = time.time()
        limit = bounded_iterations(max_iterations or self.max_iterations, minimum=1, maximum=max(1, self.max_iterations))
        added: Dict[Fact, float] = {}
        iterations = 0
        with self._reasoning_lock:
            for _ in range(limit):
                iterations += 1
                current_new: Dict[Fact, float] = {}
                ranked_rules = rank_rules_by_weight(self.rules, self.rule_weights)
                if self.exploration_rate > 0.0 and len(ranked_rules) > 1:
                    sampled = sample_rules(ranked_rules, self.rule_weights, k=max(1, min(len(ranked_rules), 3)))
                    ranked_rules = sampled + [rule for rule in ranked_rules if rule[0] not in {r[0] for r in sampled}]

                for name, rule_fn, default_weight in ranked_rules:
                    try:
                        inferred = rule_fn(dict(self.knowledge_base)) or {}
                        if not isinstance(inferred, Mapping):
                            raise RuleExecutionError("Rule must return a mapping", context={"rule": name})
                    except ReasoningError:
                        self._update_rule_weights(name, success=False)
                        raise
                    except Exception as exc:
                        self._update_rule_weights(name, success=False)
                        raise RuleExecutionError("Symbolic rule execution failed", cause=exc, context={"rule": name}) from exc

                    effective_weight = clamp_confidence(self.rule_weights.get(name, default_weight))
                    rule_added = False
                    for inferred_fact, inferred_conf in inferred.items():
                        normalized = normalize_fact(inferred_fact)
                        try:
                            ensure_non_contradictory(normalized, self.knowledge_base, threshold=self.contradiction_threshold, source=name)
                        except ContradictionError:
                            continue
                        weighted = clamp_confidence(inferred_conf) * effective_weight
                        previous = self.knowledge_base.get(normalized, 0.0)
                        if weighted > previous:
                            current_new[normalized] = max(current_new.get(normalized, 0.0), weighted)
                            rule_added = True
                    self._update_rule_weights(name, success=rule_added)

                if not current_new:
                    break
                self.knowledge_base.update(current_new)
                added.update(current_new)

            self._sync_component_state()
            self._persist_state(reason="forward_chaining")

        conflicts = self._detect_conflicts()
        redundancies = self._detect_redundancies()
        elapsed = time.time() - start
        self.conflict_count = len(conflicts)
        self.forward_chaining_speed = elapsed
        report = ForwardChainReport(added=added, iterations=iterations, conflicts=conflicts, redundancies=redundancies, duration_seconds=elapsed)
        self._record_trace(AgentReasoningTrace("forward_chaining", start, time.time(), "success", {"added": len(added), "iterations": iterations}))
        return report

    def _update_rule_weights(self, rule_name: str, success: bool) -> None:
        if rule_name not in self.rule_weights:
            return
        self.rule_weights[rule_name] = update_rule_weight(self.rule_weights[rule_name], success=success, learning_rate=self.learning_rate, decay=self.decay)

    def _detect_conflicts(self) -> List[Any]:
        try:
            return list(self.rule_engine.detect_fact_conflicts(self.contradiction_threshold))
        except Exception:
            return list(conflict_pairs(self.knowledge_base))

    def _detect_redundancies(self) -> List[Any]:
        try:
            return list(self.rule_engine.redundant_fact_check(self.redundancy_margin))
        except Exception:
            return list(redundancy_groups(self.knowledge_base, margin=self.redundancy_margin).values())

    # ------------------------------------------------------------------
    # Typed reasoning facade
    # ------------------------------------------------------------------
    def _resolve_reasoning_type(self, reasoning_type: Optional[str], problem: Any = None) -> str:
        requested = (reasoning_type or "auto").strip().lower()
        requested = self.reasoning_type_aliases.get(requested, requested)
        if requested in {"", "auto", "default"}:
            determiner = getattr(self.types, "determine_reasoning_strategy", None)
            if callable(determiner) and problem is not None:
                return str(determiner(str(problem)))
            return self.strategy_default
        return requested

    def _invoke_reasoning_engine(self, reasoning_engine: Any, problem: Any, context: Optional[Mapping[str, Any]] = None) -> Any:
        context_dict = dict(context or {})
        performer = getattr(reasoning_engine, "perform_reasoning", None)
        if not callable(performer):
            raise ReasoningTypeError("Reasoning engine does not expose perform_reasoning", context={"engine": type(reasoning_engine).__name__})
        sig = inspect.signature(performer)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        names = [p.name for p in params]
        kwargs: Dict[str, Any] = {}
        if "context" in names:
            kwargs["context"] = context_dict
        if "premises" in names and "hypothesis" in names:
            return performer(
                premises=context_dict.get("premises", [str(problem)]),
                hypothesis=context_dict.get("hypothesis", str(problem)),
                **kwargs,
            )
        if "events" in names:
            if "conditions" in names:
                kwargs["conditions"] = context_dict.get("conditions", {})
            return performer(events=context_dict.get("events", problem if isinstance(problem, list) else [problem]), **kwargs)
        if "observations" in names:
            return performer(observations=context_dict.get("observations", problem), **kwargs)
        if "target" in names and "source_domain" in names:
            return performer(target=context_dict.get("target", problem), source_domain=context_dict.get("source_domain", []), **kwargs)
        if "system" in names:
            return performer(system=context_dict.get("system", problem), **kwargs)
        if "input_data" in names:
            return performer(input_data=problem, **kwargs)
        if params and params[0].name != "context":
            kwargs[params[0].name] = problem
        return performer(**kwargs)

    def reason(self, problem: Any, reasoning_type: Optional[str] = None, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        start = time.time()
        resolved_type = self._resolve_reasoning_type(reasoning_type, problem)
        engine = self.types.create(resolved_type)
        result = self._invoke_reasoning_engine(engine, problem, context)
        payload = {"reasoning_type": resolved_type, "result": result, "status": "success"}
        self.operation_counts["reason"] += 1
        self._record_trace(AgentReasoningTrace("reason", start, time.time(), "success", {"type": resolved_type}))
        return payload

    def react_loop(self, problem: str, max_steps: Optional[int] = None) -> Dict[str, Any]:
        start = time.time()
        steps_limit = bounded_iterations(max_steps or self.max_react_steps, minimum=1, maximum=self.max_react_steps)
        strategy = self._resolve_reasoning_type("auto", problem)
        state: Dict[str, Any] = {"problem": problem, "strategy": strategy}
        steps: List[Dict[str, Any]] = []
        for idx in range(steps_limit):
            trace = self.generate_chain_of_thought((problem, "related_to", "goal"), depth=min(2, self.max_chain_depth))
            result = self.reason(problem, strategy, context=state)
            steps.append({"step": idx + 1, "reasoning_trace": trace, "result": result})
            state[f"step_{idx + 1}"] = result
            if result.get("status") == "success":
                break
        response = {"strategy": strategy, "steps": steps, "resolved": bool(steps)}
        self._record_trace(AgentReasoningTrace("react_loop", start, time.time(), "success", {"steps": len(steps), "strategy": strategy}))
        return response

    # ------------------------------------------------------------------
    # Actions, task execution, and diagnostics
    # ------------------------------------------------------------------
    def execute_action(self, action: str, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        action_key = str(action or "").strip().lower()
        data = dict(payload or {})
        if action_key == "query_knowledge_base":
            key = data.get("key")
            if key is None:
                return {"success": True, "results": list(self.knowledge_base.items())[: self.max_action_results]}
            fact = normalize_fact(key)
            return {"success": True, "results": {fact: self.knowledge_base.get(fact, 0.0)}}
        if action_key == "run_consistency_check":
            return {"success": True, "consistent": self.check_consistency(data.get("fact"))}
        if action_key == "forward_chaining":
            report = self.forward_chaining_report(data.get("max_iterations"))
            return {"success": True, "report": report.to_dict()}
        if action_key == "backward_chaining":
            goal = data.get("goal")
            if goal is None:
                return {"success": False, "error": "goal is required"}
            goal_fact = normalize_fact(goal)
            supporting = [fact for fact in self.knowledge_base if fact[2] == goal_fact[0] or fact == goal_fact]
            return {"success": True, "goal": goal_fact, "supporting_facts": supporting[: self.max_action_results]}
        if action_key == "probabilistic_query":
            return {"success": True, "probability": self.probabilistic_query(data["fact"], data.get("evidence"))}
        if action_key == "reason":
            return {"success": True, "result": self.reason(data.get("problem"), data.get("reasoning_type"), data.get("context"))}
        if action_key == "request_human_input":
            request = {"timestamp": time.time(), "reason": data.get("reason", "low_confidence_reasoning"), "context": data.get("context", {})}
            self._shared_append(self.human_intervention_key, request)
            return {"success": True, "request": request}
        return {"success": False, "error": f"Unknown action: {action}"}

    def perform_task(self, task_data: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(task_data or {})
        task_type = str(payload.get("task_type", "forward_chaining")).strip().lower()
        if task_type == "add_fact":
            return {"status": "success", "added": self.add_fact(payload["fact"], payload.get("confidence", 1.0))}
        if task_type == "validate_fact":
            return self.validate_fact(payload["fact"], payload.get("threshold", 0.75))
        if task_type == "probabilistic_query":
            return {"status": "success", "probability": self.probabilistic_query(payload["fact"], payload.get("evidence"))}
        if task_type == "multi_hop_reasoning":
            return {"status": "success", "score": self.multi_hop_reasoning(payload["query"], payload.get("max_depth", 3))}
        if task_type == "reason":
            return self.reason(payload.get("problem"), payload.get("reasoning_type"), payload.get("context"))
        if task_type == "execute_action":
            return self.execute_action(payload.get("action", "query_knowledge_base"), payload.get("payload"))
        report = self.forward_chaining_report(payload.get("max_iterations"))
        return {"status": "success", "task_type": "forward_chaining", "report": report.to_dict()}

    def stream_update(self, new_facts: Iterable[Union[str, Sequence[Any]]], confidence: float = 1.0) -> Dict[str, Any]:
        added = 0
        for fact in new_facts:
            if self.add_fact(fact, confidence=confidence, publish=False):
                added += 1
        inferred = self.forward_chaining(max_iterations=min(2, self.max_iterations))
        return {"added": added, "inferred": len(inferred)}

    def run_bayesian_learning(self, observations: List[Any]) -> Any:
        runner = getattr(self.probabilistic_models, "run_bayesian_learning_cycle", None)
        if not callable(runner):
            raise ModelInferenceError("ProbabilisticModels does not expose run_bayesian_learning_cycle")
        return runner(observations)

    def get_probability_grid(self, agent_pos: Any = None, target_pos: Any = None) -> Any:
        getter = getattr(self.probabilistic_models, "get_probability_grid", None)
        return getter(agent_pos=agent_pos, target_pos=target_pos) if callable(getter) else []

    def generate_chain_of_thought(self, query: Union[str, Sequence[Any]], depth: int = 3) -> List[str]:
        fact = normalize_fact(query)
        max_depth = bounded_iterations(depth, minimum=1, maximum=self.max_chain_depth)
        trace: List[str] = []
        current = [fact]
        visited: set[Fact] = set()
        for step_idx in range(max_depth):
            if not current:
                break
            next_facts: List[Fact] = []
            for current_fact in current:
                if current_fact in visited:
                    continue
                visited.add(current_fact)
                conf = self.knowledge_base.get(current_fact, 0.0)
                trace.append(f"Step {step_idx + 1}: {current_fact} @ {conf:.3f}")
                for candidate in self.knowledge_base:
                    if candidate[0] == current_fact[2] or candidate[2] == current_fact[0]:
                        next_facts.append(candidate)
            current = next_facts[: self.max_action_results]
        return trace

    def parse_goal(self, goal_description: str) -> Dict[str, Any]:
        text = str(goal_description or "")
        tokens = text.lower().split()
        return {
            "raw": text,
            "reasoning_type": self._resolve_reasoning_type("auto", text),
            "contains_uncertainty": any(w in tokens for w in ["maybe", "likely", "uncertain", "probable"]),
            "contains_constraint": any(w in tokens for w in ["must", "should", "cannot", "never"]),
        }

    def get_current_context(self) -> List[str]:
        context: List[str] = []
        if len(self.knowledge_base) > self.large_kb_threshold:
            context.append("large_knowledge_base")
        if any(conf < self.low_confidence_threshold for conf in self.knowledge_base.values()):
            context.append("low_confidence_environment")
        if self.conflict_count > 0:
            context.append("conflict_detected")
        if not self.rules:
            context.append("no_symbolic_rules_registered")
        return context

    def predict(self, state: Any = None) -> Dict[str, Any]:
        confidence_values = list(self.knowledge_base.values())
        return {
            "knowledge_size": len(self.knowledge_base),
            "rule_count": len(self.rules),
            "context": self.get_current_context(),
            "confidence_mean": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
            "conflict_count": self.conflict_count,
            "state": state,
        }

    def diagnostics(self) -> Dict[str, Any]:
        return json_safe_reasoning_state(
            {
                "agent": self.name,
                "version": __version__,
                "knowledge_size": len(self.knowledge_base),
                "rule_count": len(self.rules),
                "rule_weights": dict(self.rule_weights),
                "conflict_count": self.conflict_count,
                "forward_chaining_speed": self.forward_chaining_speed,
                "operation_counts": dict(self.operation_counts),
                "history_size": len(self.reasoning_history),
                "components": {
                    "types": type(self.types).__name__,
                    "hybrid_models": type(self.hybrid_models).__name__,
                    "probabilistic_models": type(self.probabilistic_models).__name__,
                    "rule_engine": type(self.rule_engine).__name__,
                    "validation_engine": type(self.validation_engine).__name__,
                },
            }
        )

    def health_check(self) -> Dict[str, Any]:
        diagnostics = self.diagnostics()
        healthy = bool(self.types and self.rule_engine and self.validation_engine and self.probabilistic_models and self.hybrid_models)
        return {"healthy": healthy, "diagnostics": diagnostics}

    def __repr__(self) -> str:
        return f"ReasoningAgent(kb={len(self.knowledge_base)}, rules={len(self.rules)}, conflicts={self.conflict_count})"


if __name__ == "__main__":
    print("\n=== Running Reasoning Agent ===\n")
    printer.status("TEST", "Reasoning Agent initialized", "info")

    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    agent = ReasoningAgent(shared_memory=shared_memory, agent_factory=agent_factory)

    assert agent.add_fact(("Apple", "is", "Fruit"), 0.9)
    assert agent.add_fact(("Fruit", "is", "Healthy"), 0.85)
    inferred = agent.forward_chaining(max_iterations=3)
    validation = agent.validate_fact(("Apple", "is", "Fruit"), threshold=0.5)
    reasoned = agent.reason(
        "Socrates is mortal",
        "deduction",
        {"premises": ["all humans are mortal", "Socrates is human"], "hypothesis": "Socrates is mortal"},
    )
    action = agent.execute_action("query_knowledge_base", {"key": ("Apple", "is", "Fruit")})
    stream = agent.stream_update([("Banana", "is", "Fruit")], confidence=0.8)
    health = agent.health_check()

    assert isinstance(inferred, dict)
    assert validation.get("combined_valid") in {True, False}
    assert reasoned.get("status") == "success"
    assert action.get("success") is True
    assert stream["added"] >= 0
    assert health["healthy"] is True

    print("\n=== Test ran successfully ===\n")
