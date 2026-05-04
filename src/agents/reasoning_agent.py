from __future__ import annotations

__version__ = "2.1.0"

"""
Reasoning Agent for Scalable Autonomous Intelligence
Features:
- Knowledge representation with probabilistic confidence
- Rule learning and adaptation
- Advanced NLP capabilities
- Probabilistic reasoning
- Multiple inference methods

Real-World Usage:
1. Healthcare Decision Support: Detect conflicting treatment plans (e.g., drug interactions) using contradiction thresholds.
2. Legal Tech: Audit contracts for logical inconsistencies or unenforceable clauses.
3. Content Moderation: Identify contradictory claims in user-generated content.
4. Financial Fraud Detection: Flag transactional contradictions (e.g., "purchases" in two countries simultaneously).
5. AI Tutoring Systems: Check student answers against domain knowledge (e.g., physics/math rules).

"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.agents.base.utils.main_config_loader import get_config_section, load_global_config
from src.agents.base_agent import BaseAgent
from src.agents.reasoning.orchestrator import Fact, ReasoningOrchestrator
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reasoning Agent")
printer = PrettyPrinter


def identity_rule(kb: Dict[Fact, float]) -> Dict[Fact, float]:
    return {(s, p, o): conf for (s, p, o), conf in kb.items() if p == "is"}


def transitive_rule(kb: Dict[Fact, float]) -> Dict[Fact, float]:
    inferred: Dict[Fact, float] = {}
    for (a, p1, b1), c1 in kb.items():
        if p1 != "is":
            continue
        for (b2, p2, c), c2 in kb.items():
            if p2 == "is" and b1 == b2:
                inferred[(a, "is", c)] = max(inferred.get((a, "is", c), 0.0), min(c1, c2))
    return inferred


class ReasoningAgent(BaseAgent):
    """Production façade that keeps orchestration thin and delegates deep logic."""

    def __init__(self, shared_memory, agent_factory, config=None):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.config = load_global_config()
        self.reasoning_config = get_config_section("reasoning_agent")
        self.inference_config = get_config_section("inference")
        self.validation_config = get_config_section("validation")
        self.storage_config = get_config_section("storage")

        self.learning_rate = float(self.reasoning_config.get("learning_rate", self.config.get("learning_rate", 0.05)))
        self.decay = float(self.reasoning_config.get("decay", 0.95))
        self.exploration_rate = float(self.inference_config.get("exploration_rate", 0.1))
        self.max_iterations = int(self.inference_config.get("default_chain_length", 5))
        self.contradiction_threshold = float(self.config.get("contradiction_threshold", 0.25))
        self.knowledge_db = self.storage_config.get("knowledge_db", "src/agents/knowledge/templates/knowledge_db.json")

        orchestrator_config = {
            "learning_rate": self.learning_rate,
            "decay": self.decay,
            "redundancy_margin": self.validation_config.get("redundancy_margin", 0.05),
        }
        self.orchestrator = ReasoningOrchestrator(
            shared_memory=self.shared_memory,
            agent_factory=self.agent_factory,
            config=orchestrator_config,
            storage_path=self.knowledge_db,
            contradiction_threshold=self.contradiction_threshold,
        )

        self.rules: List[Tuple[str, Callable, float]] = self.orchestrator.rules
        self.rule_weights: Dict[str, float] = self.orchestrator.rule_weights
        self.knowledge_base: Dict[Fact, float] = self.orchestrator.knowledge_base
        self.validation_engine = self.orchestrator.validation_engine
        self.rule_engine = self.orchestrator.rule_engine
        self.probabilistic_models = self.orchestrator.probabilistic_models
        self.hybrid_probabilistic_models = self.orchestrator.hybrid_models
        self.reasoning_strategies = self.orchestrator.reasoning_types

        self.conflict_count = 0
        self.forward_chaining_speed = 0.0

        self.add_rule(identity_rule, "identity_rule", weight=1.0)
        self.add_rule(transitive_rule, "transitive_rule", weight=0.8)
        logger.info("ReasoningAgent initialized with orchestrator-backed architecture")

    def _update_rule_weights(self, rule_name: str, success: bool):
        self.orchestrator._update_rule_weight(rule_name, success)

    def add_fact(self, fact: Union[Tuple[str, str, str], str], confidence: float = 1.0, publish: bool = True) -> bool:
        return self.orchestrator.add_fact(fact, confidence, publish=publish)

    def add_rule(self, rule: Callable, rule_name: Optional[str] = None, weight: float = 1.0) -> None:
        if isinstance(rule, list):
            for item in rule:
                if callable(item):
                    self.orchestrator.add_rule(item, None, weight)
            return
        self.orchestrator.add_rule(rule, rule_name, weight)

    def learn_from_interaction(self, fact_tuple, feedback: Dict[Tuple, bool], confidence: float = 1.0):
        self.add_fact(fact_tuple, confidence=confidence)
        for fact, is_correct in feedback.items():
            normalized = self.orchestrator.normalize_fact(fact)
            current = self.knowledge_base.get(normalized, 0.0)
            updated = (
                min(1.0, current + self.learning_rate * (1 - current))
                if is_correct
                else max(0.0, current * self.decay)
            )
            self.knowledge_base[normalized] = updated
        self.orchestrator._persist_state()

    def forget_fact(self, fact: Union[str, Tuple[str, str, str]]) -> bool:
        try:
            normalized = self.orchestrator.normalize_fact(fact)
        except Exception:
            return False
        if normalized not in self.knowledge_base:
            return False
        del self.knowledge_base[normalized]
        self.orchestrator._persist_state()
        return True

    def forget_by_subject(self, subject: str) -> int:
        targets = [fact for fact in self.knowledge_base if fact[0] == subject]
        for fact in targets:
            del self.knowledge_base[fact]
        if targets:
            self.orchestrator._persist_state()
        return len(targets)

    def validate_fact(self, fact: Tuple[str, str, str], threshold: float = 0.75) -> Dict[str, Any]:
        return self.orchestrator.validate_fact(fact, threshold)

    def check_consistency(self, fact: Optional[Tuple[str, str, str]] = None) -> bool:
        if fact is None:
            return not bool(self.rule_engine.detect_fact_conflicts(self.contradiction_threshold))
        return bool(self.validate_fact(fact, threshold=0.5).get("combined_valid", False))

    def probabilistic_query(self, fact: Tuple[str, str, str], evidence: Optional[Dict[Tuple[str, str, str], bool]] = None) -> float:
        return float(self.probabilistic_models.probabilistic_query(fact, evidence))

    def multi_hop_reasoning(self, query: Tuple[str, str, str], max_depth: int = 3) -> float:
        return float(self.probabilistic_models.multi_hop_reasoning(query, max_depth=max_depth))

    def run_bayesian_learning(self, observations: List[Any]) -> None:
        self.probabilistic_models.run_bayesian_learning_cycle(observations)


    def _invoke_reasoning_engine(self, reasoning_engine: Any, problem: Any, context: Optional[dict] = None) -> Any:
        """Invoke heterogeneous reasoning implementations with signature-aware argument mapping."""
        import inspect

        context = context or {}
        try:
            sig = inspect.signature(reasoning_engine.perform_reasoning)
            params = [p for p in sig.parameters.values() if p.name != "self"]
            param_names = [p.name for p in params]

            if "input_data" in param_names:
                return reasoning_engine.perform_reasoning(input_data=problem, context=context)

            kwargs: Dict[str, Any] = {}
            # Common context parameter
            if "context" in param_names:
                kwargs["context"] = context

            if "premises" in param_names and "hypothesis" in param_names:
                premises = context.get("premises", [str(problem)])
                hypothesis = context.get("hypothesis", str(problem))
                return reasoning_engine.perform_reasoning(premises=premises, hypothesis=hypothesis, **kwargs)

            if "events" in param_names:
                events = context.get("events", problem if isinstance(problem, list) else [problem])
                if "conditions" in param_names:
                    kwargs["conditions"] = context.get("conditions", {})
                return reasoning_engine.perform_reasoning(events=events, **kwargs)

            if "observations" in param_names:
                observations = context.get("observations", problem)
                return reasoning_engine.perform_reasoning(observations=observations, **kwargs)

            if "system" in param_names:
                system = context.get("system", problem)
                return reasoning_engine.perform_reasoning(system=system, **kwargs)

            # Generic fallback: first positional semantic arg gets `problem`
            if params:
                first_name = params[0].name
                if first_name != "context":
                    kwargs[first_name] = problem
            return reasoning_engine.perform_reasoning(**kwargs)
        except Exception as e:
            logger.warning("Signature-aware invocation failed (%s); using basic fallback", e)
            try:
                return reasoning_engine.perform_reasoning(problem, context=context)
            except TypeError:
                return reasoning_engine.perform_reasoning(problem)

    def reason(self, problem: Any, reasoning_type: str, context: Optional[dict] = None) -> Any:
        reasoning_engine = self.reasoning_strategies.create(reasoning_type)
        return self._invoke_reasoning_engine(reasoning_engine, problem, context)

    def execute_action(self, action: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        if action == "query_knowledge_base":
            key = payload.get("key")
            if key is None:
                return {"success": True, "results": list(self.knowledge_base.items())[:20]}
            fact = self.orchestrator.normalize_fact(key)
            return {"success": True, "results": {fact: self.knowledge_base.get(fact, 0.0)}}

        if action == "run_consistency_check":
            fact = payload.get("fact")
            return {"success": True, "consistent": self.check_consistency(fact)}

        if action == "forward_chaining":
            inferred = self.forward_chaining(payload.get("max_iterations"))
            return {"success": True, "new_facts": inferred}

        if action == "backward_chaining":
            goal = payload.get("goal")
            if goal is None:
                return {"success": False, "error": "goal is required"}
            goal_fact = self.orchestrator.normalize_fact(goal)
            supporting = [fact for fact in self.knowledge_base if fact[2] == goal_fact[0]]
            return {"success": True, "supporting_facts": supporting}

        if action == "request_human_input":
            request = {
                "timestamp": time.time(),
                "reason": payload.get("reason", "low_confidence_reasoning"),
                "context": payload.get("context", {}),
            }
            self.shared_memory.append("human_intervention_requests", request)
            return {"success": True, "request": request}

        return {"success": False, "error": f"Unknown action: {action}"}

    def generate_chain_of_thought(self, query: Union[str, Tuple[str, str, str]], depth: int = 3) -> List[str]:
        fact = self.orchestrator.normalize_fact(query)
        chain = []
        current = [fact]
        visited = set()

        for i in range(max(1, depth)):
            if not current:
                break
            next_facts: List[Fact] = []
            for current_fact in current:
                if current_fact in visited:
                    continue
                visited.add(current_fact)
                conf = self.knowledge_base.get(current_fact, 0.0)
                chain.append(f"Step {i + 1}: {current_fact} @ {conf:.3f}")

                for candidate in self.knowledge_base:
                    if candidate[0] == current_fact[2] or candidate[2] == current_fact[0]:
                        next_facts.append(candidate)
            current = next_facts
        return chain

    def react_loop(self, problem: str, max_steps: int = 5) -> Dict[str, Any]:
        strategy = self.reasoning_strategies.determine_reasoning_strategy(problem)
        reasoning_engine = self.reasoning_strategies.create(strategy)

        actions: List[Dict[str, Any]] = []
        state: Dict[str, Any] = {"problem": problem, "strategy": strategy}
        for _ in range(max(1, max_steps)):
            thoughts = self.generate_chain_of_thought((problem, "related_to", "goal"), depth=2)
            result = self._invoke_reasoning_engine(reasoning_engine, problem, state)
            actions.append({"thoughts": thoughts, "result": result})
            state["last_result"] = result
            if result:
                break

        response = {"strategy": strategy, "steps": actions, "resolved": bool(actions)}
        self.shared_memory.publish("reasoning_trace", {"agent": self.name, "response": response})
        return response

    def forward_chaining(self, max_iterations: Optional[int] = None) -> Dict[Tuple[str, str, str], float]:
        start = time.time()
        outcome = self.orchestrator.forward_chain(
            max_iterations=max_iterations or self.max_iterations,
            exploration_rate=self.exploration_rate,
        )
        elapsed = time.time() - start
        self.forward_chaining_speed = elapsed
        self.conflict_count = len(outcome.conflicts)

        self.orchestrator.remember(
            {
                "type": "forward_chaining",
                "iterations": outcome.iterations,
                "added_count": len(outcome.added),
                "conflicts": outcome.conflicts,
                "redundancies": outcome.redundancies,
                "duration_sec": elapsed,
            },
            tag="inference",
            priority=0.8,
        )
        return outcome.added

    def stream_update(self, new_facts: List[Tuple[str, str, str]], confidence: float = 1.0):
        added = 0
        for fact in new_facts:
            if self.add_fact(fact, confidence=confidence, publish=False):
                added += 1
        inferred = self.forward_chaining(max_iterations=2)
        return {"added": added, "inferred": len(inferred)}

    def load_knowledge(self, knowledge: Dict[Tuple[str, str, str], float]) -> None:
        self.orchestrator.load_knowledge(knowledge)
        self.knowledge_base = self.orchestrator.knowledge_base

    def throttle_inference(self, factor: float = 0.5):
        factor = min(max(float(factor), 0.1), 1.0)
        self.max_iterations = max(1, int(self.max_iterations * factor))
        return {"max_iterations": self.max_iterations}

    def get_probability_grid(self, agent_pos=None, target_pos=None):
        try:
            return self.probabilistic_models.get_probability_grid(agent_pos=agent_pos, target_pos=target_pos)
        except Exception:
            return []

    def parse_goal(self, goal_description: str) -> Dict[str, Any]:
        tokens = goal_description.lower().split()
        return {
            "raw": goal_description,
            "reasoning_type": self.reasoning_strategies.determine_reasoning_strategy(goal_description),
            "contains_uncertainty": any(w in tokens for w in ["maybe", "likely", "uncertain"]),
            "contains_constraint": any(w in tokens for w in ["must", "should", "cannot"]),
        }

    def get_current_context(self) -> List[str]:
        context: List[str] = []
        if len(self.knowledge_base) > 500:
            context.append("large_knowledge_base")
        if any(v < 0.4 for v in self.knowledge_base.values()):
            context.append("low_confidence_environment")
        if self.conflict_count > 0:
            context.append("conflict_detected")
        return context

    def predict(self, state: Any = None) -> Dict[str, Any]:
        return {
            "knowledge_size": len(self.knowledge_base),
            "context": self.get_current_context(),
            "confidence_mean": (sum(self.knowledge_base.values()) / len(self.knowledge_base)) if self.knowledge_base else 0.0,
            "state": state,
        }

    def perform_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        task_type = (task_data or {}).get("task_type", "forward_chaining")
        if task_type == "add_fact":
            return {
                "status": "success",
                "added": self.add_fact(task_data["fact"], task_data.get("confidence", 1.0)),
            }
        if task_type == "validate_fact":
            return self.validate_fact(task_data["fact"], task_data.get("threshold", 0.75))
        if task_type == "probabilistic_query":
            return {"probability": self.probabilistic_query(task_data["fact"], task_data.get("evidence"))}
        if task_type == "reason":
            return {"result": self.reason(task_data.get("problem"), task_data.get("reasoning_type", "deduction"), task_data.get("context"))}
        if task_type == "execute_action":
            return self.execute_action(task_data.get("action", "query_knowledge_base"), task_data.get("payload"))

        inferred = self.forward_chaining(task_data.get("max_iterations"))
        return {"status": "success", "task_type": "forward_chaining", "new_facts": inferred}

    def __repr__(self) -> str:
        return f"ReasoningAgent(kb={len(self.knowledge_base)}, rules={len(self.rules)})"


if __name__ == "__main__":
    from pprint import pprint

    from src.agents.agent_factory import AgentFactory
    from src.agents.collaborative.shared_memory import SharedMemory

    print("\n=== Running Reasoning Agent Test ===\n")
    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    agent = ReasoningAgent(shared_memory=shared_memory, agent_factory=agent_factory)
    print(agent)

    print("\n* * * * * Phase 1: Fact Ingestion + Validation * * * * *")
    sample_facts = [
        (("Apple", "is", "Fruit"), 0.9),
        (("Fruit", "is", "Healthy"), 0.85),
        (("Banana", "is", "Fruit"), 0.88),
    ]
    for fact, conf in sample_facts:
        added = agent.add_fact(fact, confidence=conf)
        print(f"Added {fact}: {added}")

    validation = agent.validate_fact(("Apple", "is", "Fruit"), threshold=0.5)
    print("Validation result for ('Apple','is','Fruit'):")
    pprint(validation)

    print("\n* * * * * Phase 2: Rule Registration + Forward Chaining * * * * *")

    def healthy_transitive_rule(kb):
        inferred = {}
        for (s1, p1, o1), c1 in kb.items():
            if p1 != "is":
                continue
            for (s2, p2, o2), c2 in kb.items():
                if p2 == "is" and o1 == s2:
                    inferred[(s1, "is", o2)] = min(c1, c2)
        return inferred

    agent.add_rule(healthy_transitive_rule, rule_name="healthy_transitive_rule", weight=0.9)
    inferred = agent.forward_chaining(max_iterations=3)
    print(f"Inferred facts count: {len(inferred)}")
    pprint(dict(list(inferred.items())[:5]))

    print("\n* * * * * Phase 3: Probabilistic + Multi-hop Queries * * * * *")
    probability = agent.probabilistic_query(("Apple", "is", "Healthy"))
    multi_hop = agent.multi_hop_reasoning(("Apple", "is", "Healthy"), max_depth=3)
    print(f"Probabilistic query P(Apple is Healthy): {probability:.4f}")
    print(f"Multi-hop score (Apple is Healthy): {multi_hop:.4f}")

    print("\n* * * * * Phase 4: Strategy + Action Endpoints * * * * *")
    strategy = agent.reasoning_strategies.determine_reasoning_strategy(
        "Explain why apples are considered healthy."
    )
    print(f"Selected strategy: {strategy}")
    action_result = agent.execute_action(
        "query_knowledge_base", payload={"key": ("Apple", "is", "Fruit")}
    )
    print("Action result:")
    pprint(action_result)

    print("\n* * * * * Phase 5: ReAct Loop + Stream Update * * * * *")
    react_result = agent.react_loop("Why is Apple healthy?", max_steps=2)
    print("ReAct summary keys:", list(react_result.keys()))
    stream_result = agent.stream_update(
        new_facts=[("Carrot", "is", "Vegetable"), ("Vegetable", "is", "Healthy")],
        confidence=0.8,
    )
    print("Stream update result:")
    pprint(stream_result)

    print("\n* * * * * Phase 6: Notification Handling / Shared Memory * * * * *")
    latest_kb = shared_memory.get("reasoning_agent:knowledge_base", default={})
    print(f"Shared-memory KB entries: {len(latest_kb)}")
    print("Current context:", agent.get_current_context())
    print("Predict snapshot:")
    pprint(agent.predict())

    print("\n=== Successfully ran the Reasoning Agent Test ===\n")
