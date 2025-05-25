__version__ = "1.8.0"

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

import json
import re
import yaml
import math
import time
import itertools
import random
import hashlib
import numpy as np

from collections import defaultdict, OrderedDict
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional
from dataclasses import dataclass
from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.agents.reasoning.rule_engine import RuleEngine, load_config
from src.agents.reasoning.probabilistic_models import ProbabilisticModels
from src.agents.reasoning.validation import ValidationEngine
from logs.logger import get_logger

logger = get_logger("Reasoning Agent")

LOCAL_CONFIG_PATH = "src/agents/reasoning/configs/reasoning_config.yaml"

def identity_rule(kb):
    return {(s, p, o): 1.0 for (s, p, o) in kb if p == 'is'}

def transitive_rule(kb):
    new_facts = {}
    for (a, _, b1), conf1 in kb.items():
        for (b2, _, c), conf2 in kb.items():
            if b1 == b2:
                new_facts[(a, 'is', c)] = min(conf1, conf2)
    return new_facts

@dataclass
class Token:
    text: str
    lemma: str
    pos: str
    index: int # Original index in the sentence
    # Add other attributes as needed, e.g., is_stop, is_punct, shape, etc.
    is_stop: bool = False
    is_punct: bool = False
    # For more advanced features, you might add:
    # ner_tag: Optional[str] = None
    # embedding: Optional[List[float]] = None

class ReasoningAgent(BaseAgent):
    """
    Initialize the Reasoning Agent with learning capabilities.
    """
    def __init__(self, shared_memory, agent_factory,
                 config=None,
                 tuple_key: Tuple = ("subject", "predicate", "object"),
                 contradiction_threshold=0.25,
                 rule_validation: Dict = None,
                 nlp_integration: Dict = None,
                 inference: Dict = None,
                 llm: Any = None,
                 language_agent: Any = None):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config)
        
        self.tuple_key = tuple_key
        self.hypothesis_graph = defaultdict(
            lambda: {
                'nodes': set(),
                'edges': defaultdict(float),
                'confidence': 0.0
            })
        self.reasoning_agent = []
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.inference_settings = config.get("inference", {"exploration_rate": 0.1})
        storage_path = config.get("storage", {}).get("knowledge_db")
        self.storage_path = storage_path

        self.rule_engine = RuleEngine(config=load_config(LOCAL_CONFIG_PATH))
        self.validation_engine = ValidationEngine(config=load_config(LOCAL_CONFIG_PATH))
        self.probabilistic_models = ProbabilisticModels(config=load_config(LOCAL_CONFIG_PATH))



        # Initialize components
        self._load_knowledge()

        # Initialize Language components
        language_config_path = "src/agents/language/configs/language_config.yaml"
        self.language_config = self._load_language_config(language_config_path)
        glove_path = self.language_config.get("nlu", {}).get("glove_path")
        self._initialize_nlp(glove_path, tokens=[])

        logger.info(f"\nReasoning Agent Initialized...")

    def _save_knowledge(self):
        """Save knowledge and learned models to storage."""
        data = {
            'knowledge': {list(k): v for k, v in self.knowledge_base.items()},
            'rules': [(r[0], r[1].__name__, r[2]) for r in self.rules],
            'rule_weights': self.rule_weights,
            'bayesian_network': self.bayesian_network
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f)

    def _load_knowledge(self, storage_path: str = None):
        """Load knowledge and learned models from storage."""
        self.knowledge_base = {}
        self.rule_weights = {}
        self.bayesian_network = {}

        # Use the instance storage_path if not overridden
        path = storage_path or self.storage_path
        knowledge = []

        if Path(path).exists():
            with open(path, 'r') as f:
                data = json.load(f)
                knowledge = data.get('knowledge', [])

                if isinstance(knowledge, list):
                    clean_knowledge = {}
                    for k in knowledge:
                        try:
                            key = tuple(k) if isinstance(k, (list, tuple)) else (k,)
                            clean_knowledge[key] = 1.0
                        except TypeError:
                            print(f"[Warning] Skipped invalid knowledge item: {k}")
                    knowledge = clean_knowledge

                self.knowledge_base = {tuple(k): v for k, v in knowledge.items()}
                self.rule_weights = data.get('rule_weights', {})
                self.bayesian_network = data.get('bayesian_network', {})

    def _load_language_config(self, path: str) -> Dict:
        """Load language configuration file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load language config: {e}")
            return {}

    def _initialize_nlp(self, path: str, tokens: List['Token']):
        """
        Advanced NLP initialization with multi-layered linguistic features.
        """
        from src.agents.language.nlp_engine import NLPEngine
        from src.agents.language.nlu_engine import Wordlist
    
        # Initialize NLPEngine instance
        nlp_engine = NLPEngine(config=self.language_config)
    
        # Core linguistic resources
        self.stop_words = nlp_engine.STOPWORDS
    
        # Embedding spaces
        wordlist_instance = Wordlist(config=self.language_config)
        self.word_vectors = wordlist_instance._load_glove(path)
        
        # Initialize POS tagger with valid tokens
        if not tokens:
            # Create minimal default tokens if empty
            tokens = [Token(text="", lemma="", pos="", index=0)]
        self.pos_tagger = nlp_engine.apply_dependency_rules(tokens)
    
        # Rest of the initialization code remains unchanged
        self.dependency_grammar = self.rule_engine._create_dependency_rules()
        self.morphology = self._initialize_morphology
        semantic_frames = self.config.get("semantic_frames_path")
        self.semantic_frames = semantic_frames
        self.vocab = self._build_core_lexicon
    
        # Discourse features
        self.cohesion_metrics = {'entity_grid': {}, 'coref_chains': {}}
        self.pragmatic_rules = self.rule_engine._load_pragmatic_heuristics()

    def _build_core_lexicon(self):
        # This accesses the NLP Engine
        pass

    def _initialize_morphology(self):
        # this access the morphologies in morphology_rules.json
        pass

    def _initialize_probabilistic_models(self):
        """Initialize probabilistic reasoning structures."""
        self.bayesian_network = {
            'nodes': set(),
            'edges': set(),
            'cpt': {}  # Conditional Probability Tables
        }

    def add_fact(self, fact_tuple, fact: Union[Tuple, str], confidence: float = 1.0) -> bool:
        """
        Add a fact with confidence score.
        
        Args:
            fact: Either a tuple (subject, predicate, object) or string statement
            confidence: Probability/confidence score (0.0 to 1.0)
            
        Returns:
            True if fact was added/modified
        """
        contradiction_score = self.agent_factory.validate_with_azr(fact_tuple)
        if contradiction_score > 0.3:
            logger.warning(f"Fact rejected due to contradiction: {fact_tuple} (score={contradiction_score})")
        else:
            self.knowledge_base[fact_tuple] = confidence

        try:
            if isinstance(fact, str):
                fact = self._parse_statement(fact)
            
            if len(fact) != 3:
                raise ValueError("Fact must be a 3-tuple")
                
            # Update confidence using noisy-OR combination
            current_conf = self.knowledge_base.get(fact, 0.0)
            self.knowledge_base[fact] = 1 - (1 - current_conf) * (1 - confidence)
            
            # Update vocabulary for NLP
            for element in fact:
                self._update_vocabulary(str(element))
            
            self._save_knowledge()
            return True
        except Exception as e:
            print(f"Error adding fact: {e}")
            return False

    def add_rule(self, rule: Callable, rule_name: str = None, weight: float = 1.0) -> None:
        """
        Add an inference rule with initial weight.
        
        Args:
            rule: Function that takes KB and returns new facts with confidence
            rule_name: Optional identifier for the rule
            weight: Initial weight/importance of the rule
        """
        if not hasattr(self, "knowledge_base"):
            raise AttributeError("ReasoningAgent: knowledge_base not initialized before rule addition.")
        if not callable(rule):
            raise ValueError("Rule must be callable")
            
        name = rule_name or rule.__name__
        self.rules.append((name, rule, weight))
        self.rule_weights[name] = weight
        self._save_knowledge()

    def learn_from_interaction(self, fact_tuple, feedback: Dict[Tuple, bool], confidence: float = 1.0):
        """
        Learn from user feedback about facts.
        
        Args:
            feedback: Dictionary of facts and whether they were correct
        """
        contradiction_score = self.agent_factory.validate_with_azr(fact_tuple)
        if contradiction_score > 0.3:
            logger.warning(f"Fact rejected due to contradiction: {fact_tuple} (score={contradiction_score})")
        else:
            self.knowledge_base[fact_tuple] = confidence

        for fact, is_correct in feedback.items():
            current_conf = self.knowledge_base.get(fact, 0.0)
            if is_correct:
                new_conf = current_conf + self.learning_rate * (1 - current_conf)
            else:
                new_conf = current_conf * self.decay_factor
            self.knowledge_base[fact] = new_conf
        
        self._save_knowledge()

    def _update_rule_weights(self, rule_name: str, success: bool):
        """
        Reinforcement learning for rule weights.
        
        Args:
            rule_name: Name of the rule to update
            success: Whether the rule application was successful
        """
        if rule_name in self.rule_weights:
            current_weight = self.rule_weights[rule_name]
            if success:
                # Positive reinforcement
                new_weight = current_weight + self.learning_rate * (1 - current_weight)
            else:
                # Negative reinforcement
                new_weight = current_weight * self.decay_factor
                
            self.rule_weights[rule_name] = new_weight
            
            # Update the rule in the rules list
            for i, (name, rule, weight) in enumerate(self.rules):
                if name == rule_name:
                    self.rules[i] = (name, rule, new_weight)
                    break

    def check_consistency(self, fact: Tuple) -> bool:
        """
        Self-consistency via paraphrased queries.
        """
        paraphrases = [
            f"Is {fact} true?",
            f"Does {fact} hold in all cases?",
            f"Confirm the validity of {fact}"
        ]
        results = [self.ProbabilisticModels.probabilistic_query(fact) > 0.8 for _ in paraphrases]
        return sum(results)/len(results) >= 0.75

    def validate_fact(self, fact: Tuple[str, str, str], threshold: float = 0.75) -> Dict[str, Any]:
        subject, predicate, obj = fact
    
        symbolic_confidence = 1.0 if self.knowledge and self.knowledge.has_fact(fact) else 0.0
        semantic_similarity_score = self.llm.semantic_similarity(f"{subject} {predicate}", obj)
    
        combined_score = (symbolic_confidence + semantic_similarity_score) / 2
    
        validation_result = {
            "fact": fact,
            "symbolic_confidence": symbolic_confidence,
            "semantic_similarity": semantic_similarity_score,
            "combined_confidence": combined_score,
            "valid": combined_score >= threshold
        }
    
        return validation_result

    def react_loop(self, problem: str, max_steps: int = 5) -> dict:
        """
        ReAct-style problem solving with interleaved thoughts and actions.
        """
        solution = {}
        for _ in range(max_steps):
            # Thought Phase
            thoughts = self.generate_chain_of_thought(problem)
            print(f"Thought: {thoughts[-1]}")
            
            # Action Phase
            action = self._select_action(thoughts)
            result = self.execute_action(action)
            
            # Update state
            solution.update(result)
            if self._is_goal_reached(solution):
                break
        return solution

    def generate_chain_of_thought(self, query: Union[str, Tuple], depth: int = 3) -> List[str]:
        """
        Generate step-by-step inference trace for the given query.
        Returns a list of reasoning steps ("thoughts").
        """
        chain = []
        visited = set()
        current = [query] if isinstance(query, tuple) else [self._parse_statement(query)]

        for step in range(depth):
            next_facts = []
            for fact in current:
                if fact in visited:
                    continue
                visited.add(fact)

                confidence = self.knowledge_base.get(fact, 0.0)
                trace = f"Step {step+1}: {fact} with confidence {confidence:.2f}"
                chain.append(trace)

                for (subj, pred, obj), conf in self.knowledge_base.items():
                    if subj == fact[2] or obj == fact[2]:
                        next_facts.append((subj, pred, obj))

            if not next_facts:
                break
            current = next_facts

        return chain

    def _select_action(self, thoughts: List[str]) -> str:
        """
        Rule-based action selection inspired by ReAct paper.
        """
        last_thought = thoughts[-1].lower()
        if "retriev" in last_thought:
            return "query_knowledge_base"
        elif "verify" in last_thought:
            return "run_consistency_check"
        return "forward_chaining"

    def execute_action(self, action: List[str]) -> str:
        pass

    def _is_goal_reached(self, context: Dict[str, Any]) -> bool:
        """
        Determines whether the reasoning process has reached its goal.
        Based on:
        - Confidence threshold        - Fact convergence        - Contradiction avoidance
        """
        target = context.get("target_fact")
        if not target:
            return False

        confidence = self.knowledge_base.get(target, 0.0)
        contradiction = any(
            k for k in self.knowledge_base
            if k[0] == target[0] and k[1] == target[1] and k != target and self.knowledge_base[k] > self.contradiction_threshold
        )
    
        return confidence >= 0.9 and not contradiction
        
    def forward_chaining(self, max_iterations: int = 100) -> Dict[Tuple, float]:
        """
        Probabilistic forward chaining inference.

        Args:
            max_iterations: Maximum number of inference cycles

        Returns:
            New facts with their confidence scores
        """
        new_facts = {}
        for _ in range(max_iterations):
            current_new = {}

            # Explore new rules occasionally
            if random.random() < self.inference_settings.get("exploration_rate", 0.1):
                self.rule_engine._discover_new_rules()

            for name, rule_func, weight in self.rules:
                try:
                    inferred = rule_func(self.knowledge_base)
                    for fact, confidence in inferred.items():
                        weighted_conf = confidence * weight
                        if fact not in self.knowledge_base or weighted_conf > self.knowledge_base[fact]:
                            current_new[fact] = weighted_conf
                            self._update_rule_weights(name, True)
                        else:
                            self._update_rule_weights(name, False)
                except Exception as e:
                    self.logger.warning(f"Rule {name} failed: {e}")

            if not current_new:
                break

            new_facts.update(current_new)
            self.knowledge_base.update(current_new)

        # --- Validation Phase ---
        rule_engine = RuleEngine(config=load_config(LOCAL_CONFIG_PATH), knowledge_base=self.knowledge_base)
        circular = rule_engine.detect_circular_rules()
        conflicts = rule_engine.detect_fact_conflicts()
        redundant = rule_engine.redundant_fact_check(confidence_margin=0.05)

        if circular:
            self.logger.warning(f"Circular rules detected: {circular}")
        if conflicts:
            self.logger.warning(f"Conflicting facts found: {conflicts}")
        if redundant:
            self.logger.info(f"Redundant facts skipped: {redundant}")

        return {k: v for k, v in new_facts.items() if k not in redundant}

    def _incremental_forward_chain(self, seed_facts: Dict[Tuple, float], max_iterations: int = 10):
        """
        Run a light inference pass starting from seed_facts.
        """
        new_facts = {}
        for _ in range(max_iterations):
            current_new = {}
            for name, rule_func, weight in self.rules:
                try:
                    inferred = rule_func(self.knowledge_base)
                    for fact, conf in inferred.items():
                        weighted_conf = conf * weight
                        if fact not in self.knowledge_base or weighted_conf > self.knowledge_base[fact]:
                            current_new[fact] = weighted_conf
                            self._update_rule_weights(name, True)
                        else:
                            self._update_rule_weights(name, False)
                except Exception as e:
                    self.logger.warning(f"Rule {name} failed during incremental chaining: {e}")

            if not current_new:
                break
            new_facts.update(current_new)
            self.knowledge_base.update(current_new)

        self.logger.info(f"[IncrementalInference] Added {len(new_facts)} new facts.")

    def _parse_statement(self, statement: str) -> Tuple:
        from src.agents.language.nlu_engine import NLUEngine
        """
        Advanced statement parsing with entity recognition.
        
        Args:
            statement: Natural language statement
            
        Returns:
            Structured fact tuple
        """
        # Enhanced parsing with simple entity recognition
        parsed = NLUEngine.get_entities

        # Try multiple parsing patterns
        patterns = [
            r'(.+)\s+(is|are)\s+(.+)',  # "X is Y"
            r'(.+)\s+->\s+(.+)',        # "X -> Y"
            r'(.+):\s+(.+)'             # "X: Y"
        ]

        for pattern in patterns:
            match = re.match(pattern, parsed)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return (groups[0].strip(), 'is', groups[1].strip())
                elif len(groups) == 3:
                    return (groups[0].strip(), groups[1].strip(), groups[2].strip())

        raise ValueError(f"Could not parse statement: {statement}")

    def _build_hypothesis_graph(self, root_node: Tuple):
        """Constructs multi-layered hypothesis graph with confidence scoring"""
        from src.agents.language.nlu_engine import NLUEngine
        root_key = hash(root_node)
        
        # Initialize root node
        self.hypothesis_graph[root_key]['nodes'].add(root_node)
        self.hypothesis_graph[root_key]['confidence'] = \
            self.knowledge_base.get(root_node, 0.5)
        
        # Semantic expansion with decay factor
        decay = 0.8  # Confidence reduction per hop
        visited = set()
        
        def expand_node(node, current_confidence, depth=0):
            if depth > 5 or node in visited:  # Limit recursion depth
                return
                
            visited.add(node)
            
            for fact in self.knowledge_base:
                similarity = NLUEngine.semantic_similarity(str(node), str(fact))
                if similarity > 0.65:
                    edge_conf = current_confidence * decay * similarity
                    
                    # Add node and edge
                    self.hypothesis_graph[root_key]['nodes'].add(fact)
                    self.hypothesis_graph[root_key]['edges'][(node, fact)] = \
                        max(edge_conf, 
                            self.hypothesis_graph[root_key]['edges'].get((node, fact), 0))
                    
                    # Recursive expansion
                    expand_node(fact, edge_conf, depth+1)
        
        expand_node(root_node, self.hypothesis_graph[root_key]['confidence'])
        
        # Normalize confidence scores
        max_conf = max([c for _, c in self.hypothesis_graph[root_key]['edges'].values()] 
                       or [0])
        if max_conf > 0:
            for edge in self.hypothesis_graph[root_key]['edges']:
                self.hypothesis_graph[root_key]['edges'][edge] /= max_conf

    def stream_update(self, new_facts: List[Tuple[str, str, str]], confidence: float = 1.0):
        """
        Incrementally update the knowledge base with new facts and run inference only on those.
        """
        updated = False
        affected_facts = {}

        for fact in new_facts:
            if fact not in self.knowledge_base or self.knowledge_base[fact] < confidence:
                self.knowledge_base[fact] = confidence
                affected_facts[fact] = confidence
                updated = True

        if updated:
            self._incremental_forward_chain(affected_facts)
            self._save_knowledge()

        return updated

    def forget_fact(self, fact: Union[str, Tuple[str, str, str]]) -> bool:
        """
        Remove a fact from the knowledge base and delete it from storage.
        Also clears derived facts and rules influenced by it if applicable.
        """
        try:
            if isinstance(fact, str):
                fact = self._parse_statement(fact)
    
            # Directly delete the fact
            if fact in self.knowledge_base:
                del self.knowledge_base[fact]
    
            # Optionally: remove any facts inferred from this
            derived_facts = [k for k in self.knowledge_base if fact[0] in k or fact[2] in k]
            for df in derived_facts:
                del self.knowledge_base[df]
    
            # Clean any rule references if applicable
            if hasattr(self, 'rules'):
                self.rules = [(name, fn, wt) for (name, fn, wt) in self.rules if fact not in fn(self.knowledge_base)]
    
            self._save_knowledge()
            self.logger.info(f"[GDPR] Forgotten fact: {fact}")
            return True
    
        except Exception as e:
            self.logger.error(f"[GDPR] Failed to forget fact {fact}: {e}")
            return False
        
    def forget_by_subject(self, subject: str) -> int:
        """
        Remove all facts related to a given subject/entity (e.g., 'user123').
        Returns the number of facts removed.
        """
        to_remove = [fact for fact in self.knowledge_base if subject in fact]
        for fact in to_remove:
            del self.knowledge_base[fact]
        self._save_knowledge()
        self.logger.info(f"[GDPR] Removed {len(to_remove)} facts related to subject: {subject}")
        return len(to_remove)
    
    def forget_memory_keys(self, key_prefix: str):
        """
        Remove all shared memory keys starting with a specific prefix.
        """
        keys_to_clear = [key for key in self.shared_memory.keys() if key.startswith(key_prefix)]
        for key in keys_to_clear:
            self.shared_memory.set(key, None)
        self.logger.info(f"[GDPR] Cleared {len(keys_to_clear)} shared memory keys with prefix '{key_prefix}'.")

    def log_gdpr_request(self, request_type: str, target: str):
        with open("logs/gdpr_audit_log.jsonl", "a") as log:
            log.write(json.dumps({
                "timestamp": time.time(),
                "agent": self.name,
                "action": request_type,
                "target": target
            }) + "\n")

if __name__ == "__main__":
    print("\n=== Running Reasoning Agent ===\n")

    config = {
        "storage": {
            "knowledge_db": "knowledge_db.json"
        },
        "semantic_frames_path": "semantic_frames.json"
    }
    shared_memory = {}
    agent_factory = lambda: None

    agent = ReasoningAgent(shared_memory, agent_factory, config=config)
    print(agent)

    # Prepopulate some basic knowledge
    agent.knowledge_base = {
        ("cat", "is", "animal"): 1.0,
        ("animal", "is", "living"): 1.0,
        ("penguin", "is", "bird"): 1.0,
        ("bird", "can", "fly"): 0.9,
        ("penguin", "cannot", "fly"): 0.99
    }

    # Register sample rule manually
    def identity_rule(kb):
        return {(s, p, o): 1.0 for (s, p, o) in kb if p == 'is'}

    agent.rules = [
        ("IdentityRule", identity_rule, 1.0)
    ]

    # Test generate_chain_of_thought
    print("\n--- Chain of Thought ---")
    thoughts = agent.generate_chain_of_thought(("penguin", "is", "bird"))
    for t in thoughts:
        print(t)

    # Test _select_action
    action = agent._select_action(thoughts)
    print(f"\nSelected Action: {action}")

    # Stub execute_action
    result = agent.execute_action(action)
    print(f"\nExecuted Action Result (stub): {result}")

    # Mock context and test _is_goal_reached
    test_context = {"target_fact": ("penguin", "is", "bird")}
    goal_reached = agent._is_goal_reached(test_context)
    print(f"\nGoal Reached? {goal_reached}")

    # Test forward chaining
    print("\n--- Forward Chaining ---")
    inferred_facts = agent.forward_chaining()
    for fact, conf in inferred_facts.items():
        print(f"Inferred: {fact} => {conf:.2f}")

    print("\n=== Successfully Ran Reasoning Agent ===\n")
