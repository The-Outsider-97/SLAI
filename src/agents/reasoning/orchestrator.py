from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from .utils.reasoning_errors import *
from .utils.reasoning_helpers import *
from .probabilistic_models import ProbabilisticModels
from .hybrid_probabilistic_models import HybridProbabilisticModels
from .reasoning_memory import ReasoningMemory
from .reasoning_types import ReasoningTypes
from .rule_engine import RuleEngine
from .validation import ValidationEngine
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reasoning Orchestrator")
printer = PrettyPrinter()

Fact = Tuple[str, str, str]
RuleEntry = Tuple[str, Callable[[Dict[Fact, float]], Dict[Fact, float]], float]


@dataclass
class InferenceOutcome:
    added: Dict[Fact, float]
    iterations: int
    conflicts: List[Any]
    redundancies: List[Any]


class ReasoningOrchestrator:
    """Coordinates symbolic, probabilistic, validation, and memory components."""

    def __init__(
        self,
        *,
        shared_memory: Any,
        agent_factory: Any,
        config: Dict[str, Any],
        storage_path: str,
        contradiction_threshold: float,
    ) -> None:
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.config = config
        self.storage_path = storage_path
        self.contradiction_threshold = contradiction_threshold

        self.rule_engine = RuleEngine()
        self.validation_engine = ValidationEngine()
        self.reasoning_types = ReasoningTypes()
        self.probabilistic_models = ProbabilisticModels()
        self.hybrid_models = HybridProbabilisticModels()
        self.probabilistic_models.link_agent(self)
        self.reasoning_memory = ReasoningMemory()

        self.knowledge_base: Dict[Fact, float] = self._load_kb_from_shared_memory()
        self.rules: List[RuleEntry] = []
        self.rule_weights: Dict[str, float] = {}

    def _load_kb_from_shared_memory(self) -> Dict[Fact, float]:
        kb = self.shared_memory.get("reasoning_agent:knowledge_base", default={}) or {}
        normalized: Dict[Fact, float] = {}
        for fact, confidence in kb.items():
            if isinstance(fact, tuple) and len(fact) == 3:
                normalized[(str(fact[0]), str(fact[1]), str(fact[2]))] = float(confidence)
        return normalized

    @staticmethod
    def normalize_fact(fact: Union[str, Fact]) -> Fact:
        if isinstance(fact, tuple) and len(fact) == 3:
            return (str(fact[0]).strip(), str(fact[1]).strip(), str(fact[2]).strip())
        if not isinstance(fact, str):
            raise ValueError(f"Fact must be tuple[str,str,str] or string; got {type(fact).__name__}")

        text = fact.strip()
        if not text:
            raise ValueError("Fact string cannot be empty")

        if "->" in text:
            left, right = [part.strip() for part in text.split("->", 1)]
            return (left, "implies", right)
        if ":" in text:
            left, right = [part.strip() for part in text.split(":", 1)]
            return (left, "is", right)

        tokens = text.split()
        if len(tokens) >= 3:
            return (tokens[0], tokens[1], " ".join(tokens[2:]))
        if len(tokens) == 2:
            return (tokens[0], "related_to", tokens[1])
        raise ValueError(f"Unable to parse fact: {fact}")

    def add_rule(self, rule: Callable[[Dict[Fact, float]], Dict[Fact, float]], name: Optional[str], weight: float) -> str:
        if not callable(rule):
            raise ValueError("Rule must be callable")
        rule_name = (name or rule.__name__).strip()
        if not rule_name:
            raise ValueError("Rule name cannot be empty")
        safe_weight = min(max(float(weight), 0.0), 1.0)

        self.rules = [entry for entry in self.rules if entry[0] != rule_name]
        self.rules.append((rule_name, rule, safe_weight))
        self.rule_weights[rule_name] = safe_weight
        self._persist_state()
        return rule_name

    def add_fact(self, fact: Union[str, Fact], confidence: float, *, publish: bool = True) -> bool:
        normalized = self.normalize_fact(fact)
        confidence = min(max(float(confidence), 0.0), 1.0)

        if self._is_contradictory(normalized):
            logger.warning("Rejected contradictory fact: %s", normalized)
            return False

        existing = self.knowledge_base.get(normalized, 0.0)
        self.knowledge_base[normalized] = 1 - (1 - existing) * (1 - confidence)
        self._persist_state()

        if publish:
            self.shared_memory.publish("new_facts", (normalized, confidence))
        return True

    def _is_contradictory(self, fact: Fact) -> bool:
        validator = getattr(self.agent_factory, "validate_with_azr", None)
        if callable(validator):
            try:
                score = float(validator(fact))
                return score > self.contradiction_threshold
            except Exception:
                logger.exception("External contradiction validator failed")

        subject, predicate, obj = fact
        inverse = (subject, predicate, f"not_{obj}")
        return self.knowledge_base.get(inverse, 0.0) > self.contradiction_threshold

    def validate_fact(self, fact: Fact, threshold: float) -> Dict[str, Any]:
        normalized = self.normalize_fact(fact)
        confidence = self.knowledge_base.get(normalized, 0.0)
        new_facts = {normalized: confidence or 1.0}
        results = self.validation_engine.validate_all(rules=self.rules, new_facts=new_facts)
        probabilistic_confidence = float(self.probabilistic_models.probabilistic_query(normalized))

        conflict_pairs = results.get("conflicts", [])
        has_conflict = any(normalized in pair for pair in conflict_pairs if isinstance(pair, (tuple, list)))
        is_valid = confidence >= threshold and not has_conflict

        payload = {
            "fact": normalized,
            "kb_confidence": confidence,
            "probabilistic_confidence": probabilistic_confidence,
            "has_conflict": has_conflict,
            "is_redundant": normalized in results.get("redundancies", []),
            "is_valid": is_valid,
            "combined_valid": is_valid and probabilistic_confidence >= threshold,
            "validation_details": results,
        }
        if is_valid:
            self.shared_memory.set("reasoning_agent:last_validated_fact", payload)
        return payload

    def forward_chain(self, *, max_iterations: int, exploration_rate: float) -> InferenceOutcome:
        added: Dict[Fact, float] = {}
        iterations = 0

        for _ in range(max_iterations):
            iterations += 1
            current_new: Dict[Fact, float] = {}
            for name, rule_fn, default_weight in sorted(
                self.rules,
                key=lambda item: self.rule_weights.get(item[0], item[2]),
                reverse=True,
            ):
                try:
                    inferred = rule_fn(self.knowledge_base) or {}
                except Exception:
                    logger.exception("Rule execution failed: %s", name)
                    self._update_rule_weight(name, success=False)
                    continue

                weight = self.rule_weights.get(name, default_weight)
                for inferred_fact, inferred_conf in inferred.items():
                    normalized = self.normalize_fact(inferred_fact)
                    weighted_conf = min(max(float(inferred_conf) * weight, 0.0), 1.0)
                    prev = self.knowledge_base.get(normalized, 0.0)
                    if weighted_conf > prev:
                        current_new[normalized] = max(current_new.get(normalized, 0.0), weighted_conf)
                        self._update_rule_weight(name, success=True)
                    else:
                        self._update_rule_weight(name, success=False)

            if not current_new:
                break

            for new_fact, confidence in current_new.items():
                self.knowledge_base[new_fact] = confidence
            added.update(current_new)

            if exploration_rate > 0 and len(added) > 0:
                break

        self._persist_state()
        conflicts = self.rule_engine.detect_fact_conflicts(self.contradiction_threshold)
        redundancies = self.rule_engine.redundant_fact_check(self.config.get("redundancy_margin", 0.05))

        return InferenceOutcome(
            added=added,
            iterations=iterations,
            conflicts=conflicts,
            redundancies=redundancies,
        )

    def _update_rule_weight(self, rule_name: str, success: bool) -> None:
        if rule_name not in self.rule_weights:
            return
        lr = float(self.config.get("learning_rate", 0.1))
        decay = float(self.config.get("decay", 0.95))
        current = self.rule_weights[rule_name]
        self.rule_weights[rule_name] = (
            min(1.0, current + lr * (1.0 - current)) if success else max(0.01, current * decay)
        )

    def remember(self, payload: Dict[str, Any], tag: str, priority: float = 0.5) -> None:
        try:
            self.reasoning_memory.add(experience=payload, tag=tag, priority=priority)
        except Exception:
            logger.exception("Failed to persist reasoning memory payload")


    def load_knowledge(self, knowledge: Dict[Fact, float]) -> None:
        normalized: Dict[Fact, float] = {}
        for fact, conf in (knowledge or {}).items():
            try:
                normalized[self.normalize_fact(fact)] = min(max(float(conf), 0.0), 1.0)
            except Exception:
                logger.warning("Skipping invalid knowledge item during load: %s", fact)
        self.knowledge_base = normalized
        self._persist_state()

    def _persist_state(self) -> None:
        self.shared_memory.set("reasoning_agent:knowledge_base", self.knowledge_base)
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

        data = {
            "knowledge": [
                {
                    "subject": s,
                    "predicate": p,
                    "object": o,
                    "confidence": c,
                    "weight": c,
                }
                for (s, p, o), c in self.knowledge_base.items()
            ],
            "rules": [
                {"name": name, "callable": getattr(rule_fn, "__name__", "anonymous"), "weight": self.rule_weights.get(name, weight)}
                for name, rule_fn, weight in self.rules
            ],
            "updated_at": time.time(),
        }

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=os.path.dirname(self.storage_path)) as tmp:
            json.dump(data, tmp, indent=2)
            temp_path = tmp.name
        os.replace(temp_path, self.storage_path)


if __name__ == "__main__":
    print("\n=== Running Reasoning Orchistrator ===\n")
    printer.status("Init", "Reasoning Orchistrator initialized", "success")
    from src.agents.collaborative.shared_memory import SharedMemory
    from src.agents.agent_factory import AgentFactory
    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    path = "src/agents/reasoning/"
    threshold=None

    orch = ReasoningOrchestrator(
        shared_memory=shared_memory,
        agent_factory=agent_factory,
        storage_path=path,
        contradiction_threshold=threshold,
        config=None
        )
    print(orch)

    print("\n* * * * * Phase 2 * * * * *\n")

    print("\n=== Successfully ran the Knopwledge Agent ===\n")