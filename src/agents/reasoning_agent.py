__version__ = "1.9.0"

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
import queue
import re, os
import yaml
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict, OrderedDict
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional
from dataclasses import dataclass
from pathlib import Path

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.reasoning.types.base_reasoning import BaseReasoning
from src.agents.reasoning.rule_engine import RuleEngine
from src.agents.reasoning.probabilistic_models import ProbabilisticModels
from src.agents.reasoning.reasoning_types import ReasoningTypes
from src.agents.reasoning.validation import ValidationEngine
from src.agents.base_agent import BaseAgent
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reasoning Agent")
printer = PrettyPrinter

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
    index: int
    is_stop: bool = False
    is_punct: bool = False
    ner_tag: Optional[str] = None
    embedding: Optional[List[float]] = None

class ReasoningAgent(BaseAgent, nn.Module):
    """
    Initialize the Reasoning Agent with learning capabilities.
    """
    def __init__(self, shared_memory, agent_factory, config=None):
        BaseAgent.__init__(self, shared_memory, agent_factory, config=config)
        nn.Module.__init__(self)
        self.rules = []
        self.rule_weights = {}
        self.reasoning_agent = []
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.config = load_global_config()
        self.reasoning_config = get_config_section('reasoning_agent')

        # Extract specific parameters from config
        self.language_config_path = self.reasoning_config.get('language_config_path')
        self.glove_path = self.reasoning_config.get('glove_path')
        self.ner_tag = self.reasoning_config.get('ner_tag')
        self.embedding = self.reasoning_config.get('embedding')
        self.learning_rate = self.reasoning_config.get('learning_rate')
        self.exploration_rate = self.reasoning_config.get('exploration_rate')
        self.decay = self.reasoning_config.get('decay')
        self.knowledge_db = self.reasoning_config.get('knowledge_db')
        self.max_iterations = self.reasoning_config.get('max_iterations')

        self.rule_engine = RuleEngine()
        self.validation_engine = ValidationEngine()
        self.reasoning_strategies = ReasoningTypes()
        self.probabilistic_models = ProbabilisticModels()
        self.probabilistic_models.link_agent(self)  # Create bidirectional connection

        # Set tuple_key from parameter or config
        self.tuple_key = self.reasoning_config.get(
            'tuple_key', ("subject", "predicate", "object"))

        self.hypothesis_graph = defaultdict(
            lambda: {
                'nodes': set(),
                'edges': defaultdict(float),
                'confidence': 0.0
            })
        self.knowledge_base = self.shared_memory.get(
            "reasoning_agent:knowledge_base", 
            default={}
        )
        #self.shared_memory.subscribe(
        #    "new_facts", 
        #    self._handle_new_fact
        #)
        self.storage_path = None
        self._init_lang_engines()

        logger.info(f"\nReasoning Agent Initialized...")

    def _handle_new_fact(self, message):
        """Callback for new fact notifications"""
        try:
            fact, confidence = message
            self.add_fact(fact, confidence, publish=False)
        except Exception as e:
            logger.error(f"Error processing new fact: {str(e)}")

    def _initialize_probabilistic_models(self):
        """Initialize probabilistic reasoning structures."""
        self.bayesian_network = {
            'nodes': set(),
            'edges': set(),
            'cpt': {}  # Conditional Probability Tables
        }

    def _update_vocabulary(self, term: str):
        """Update the vocabulary with a new term."""
        if hasattr(self, 'wordlist') and self.wordlist is not None:
            # Add term to vocabulary using dictionary-like access
            term_lower = term.lower()
            if term_lower not in self.wordlist.data:
                self.wordlist.data[term_lower] = {}

    # Modify this method to initialize properly
    def _init_lang_engines(self):
        from src.agents.language.nlu_engine import Wordlist, NLUEngine
        self.wordlist = Wordlist()  # Initialize first
        self.nlu_engine = NLUEngine(self.wordlist)

    def process_notifications(self, input_queue, output_queue, termination_queue):
        """Process notifications from shared memory using queues"""
        while True:
            try:
                # Check termination signal first
                try:
                    if termination_queue.get(block=False) == "STOP":
                        logger.info("Received termination signal, exiting notification loop")
                        return
                except queue.Empty:
                    pass
                
                # Get queues from shared memory once
                if not hasattr(self, '_input_queue'):
                    self._input_queue = self.shared_memory.get('reasoning_input')
                    self._output_queue = self.shared_memory.get('reasoning_output')
                
                # Process messages with timeout
                try:
                    message = self._input_queue.get(block=True, timeout=0.5)
                    if message.get('channel') == 'new_facts':
                        self._handle_new_fact(message['data'])
                except queue.Empty:
                    continue
                except (ConnectionResetError, OSError) as e:
                    if "10054" in str(e) or "232" in str(e):  # Handle pipe errors
                        logger.info("Connection closed remotely, exiting notification loop")
                        return
            except Exception as e:
                logger.error(f"Notification processing failed: {str(e)}")
                time.sleep(1)  # Prevent tight error loop

    def _save_knowledge(self):
        path = self.knowledge_db

        # Get knowledge base and rule set
        kb = self.shared_memory.get("reasoning_agent:knowledge_base", default=[])
        rules = []
        rule_weights = {}

        for rule_name, rule_fn, weight in self.rules:
            rules.append([rule_name, rule_fn.__name__, weight])
            rule_weights[rule_name] = weight
    
        data = {
            "knowledge": kb,
            "rules": rules,
            "rule_weights": rule_weights,
            "bayesian_network": self.config.get("bayesian_network", None)
        }
    
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def add_fact(self, fact_tuple, fact: Union[Tuple, str], confidence: float = 1.0, publish=True) -> bool:
        """
        Add a fact with confidence score.
        
        Args:
            fact: Either a tuple (subject, predicate, object) or string statement
            confidence: Probability/confidence score (0.0 to 1.0)
            
        Returns:
            True if fact was added/modified
        """
        # Validate input
        if fact is None:
            logger.error("Attempted to add None fact")
            return False
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

            # Publish to shared memory after addition
            if publish:
                self.shared_memory.publish(
                    "new_facts", 
                    (fact, confidence)
                )
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
                new_conf = current_conf * self.decay
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
                new_weight = current_weight * self.decay
                
            self.rule_weights[rule_name] = new_weight
            
            # Update the rule in the rules list
            for i, (name, rule, weight) in enumerate(self.rules):
                if name == rule_name:
                    self.rules[i] = (name, rule, new_weight)
                    break

    def validate_fact(self, fact: Tuple[str, str, str], threshold: float = 0.75) -> Dict[str, Any]:
        subject, predicate, obj = fact
        
        # Check KB confidence
        confidence_in_kb = self.knowledge_base.get(fact, 0.0)
        
        # Create temporary fact dictionary for validation
        temp_facts = {fact: 1.0}  # Assume max confidence for validation
        
        # Run comprehensive validation
        validation_results = self.validation_engine.validate_all(
            rules=self.rules,
            new_facts=temp_facts
        )
        
        # Check for conflicts involving this fact
        has_conflict = any(
            fact in conflict_pair 
            for conflict_pair in validation_results['conflicts']
        )
        
        # Check if fact is redundant
        is_redundant = fact in validation_results['redundancies']
        
        # Determine validity - must meet confidence threshold and have no conflicts
        is_valid = (confidence_in_kb >= threshold) and not has_conflict
        
        validation_result = {
            "fact": fact,
            "kb_confidence": confidence_in_kb,
            "has_conflict": has_conflict,
            "is_redundant": is_redundant,
            "is_valid": is_valid,
            "validation_details": {
                "sound_rules": validation_results['sound_rules'],
                "conflicts": validation_results['conflicts'],
                "redundancies": validation_results['redundancies']
            }
        }

        prob_confidence = self.probabilistic_models.probabilistic_query(fact)
        prob_consistent = self.check_consistency(fact)
        
        validation_result.update({
            "probabilistic_confidence": prob_confidence,
            "probabilistic_consistent": prob_consistent,
            "combined_valid": is_valid and prob_consistent and (prob_confidence > threshold)
        })

        # Atomic update if validation passes
        if validation_result["is_valid"]:
            self.shared_memory.compare_and_swap(
                key="validated_facts",
                expected_value=None,
                new_value=fact
            )

        return validation_result

    def check_consistency(self, fact: Tuple) -> bool:
        """
        Self-consistency via paraphrased queries.
        """
        paraphrases = [
            f"Is {fact} true?",
            f"Does {fact} hold in all cases?",
            f"Confirm the validity of {fact}"
        ]
        results = [self.probabilistic_models.probabilistic_query(fact) > 0.8 for _ in paraphrases]
        return sum(results)/len(results) >= 0.75
    
    def probabilistic_query(self, fact: Tuple, evidence: Dict[Tuple, bool] = None) -> float:
        """Enhanced probabilistic query with agent context"""
        return self.probabilistic_models.probabilistic_query(
            fact, 
            evidence,
            context=self.get_current_context()
        )

    def multi_hop_reasoning(self, query: Tuple, max_depth: int = 3) -> Union[float, list]:
        """Context-aware multi-hop reasoning with fallback"""
        try:
            return self.probabilistic_models.multi_hop_reasoning(
                query,
                context=self.get_current_context(),
                max_depth=max_depth
            )
        except Exception as e:
            logger.error(f"Multi-hop reasoning failed: {str(e)}")
            # Return confidence score as fallback
            return self.probabilistic_models.probabilistic_query(query)

    def run_bayesian_learning(self, observations: list):
        """Run Bayesian learning with agent-specific context"""
        self.probabilistic_models.run_bayesian_learning_cycle(
            observations,
            # context=self.get_current_context()
        )

    def get_current_context(self) -> List[str]:
        """Get current reasoning context for probabilistic models"""
        context = []
        if len(self.knowledge_base) > 1000:
            context.append("large_knowledge_base")
        if any(conf < 0.3 for conf in self.knowledge_base.values()):
            context.append("low_confidence_environment")
        return context

    def react_loop(self, problem: str, max_steps: int = 5) -> dict:
        """
        ReAct-style problem solving with integrated reasoning strategies
        """
        solution = {}
        # Determine the best reasoning strategy based on the problem
        reasoning_type = self.reasoning_strategies._determine_reasoning_strategy(problem)
        reasoning_engine = self.reasoning_strategies.create(reasoning_type)
        
        self.current_thoughts = []  # Initialize thought storage
        for step in range(max_steps):
            # Thought Phase - use reasoning engine to enhance thought generation
            thoughts = self._generate_enhanced_chain(problem, solution, reasoning_engine)
            self.current_thoughts = thoughts
            print(f"Thought: {thoughts[-1]}")
            
            # Action Phase
            action = self._select_action(thoughts)
            result = self.execute_action(action)
            
            # Update state
            solution.update(result)
            if self._is_goal_reached(solution):
                break
    
        # Probabilistic guidance to action selection
        last_thought = self.current_thoughts[-1].lower()
        if "uncertain" in last_thought:
            prob_score = self.probabilistic_models.probabilistic_query(problem)
            if prob_score < 0.5:
                self.shared_memory.append(
                    key="human_intervention_requests",
                    value={
                        "timestamp": time.time(),
                        "thoughts": self.current_thoughts,
                        "problem": problem,
                        "reason": "Low confidence in solution"
                    }
                )
        return solution

    def _generate_enhanced_chain(self, problem: str, context: dict, 
                                reasoning_engine: BaseReasoning) -> List[str]:
        """
        Generate enhanced chain of thought using reasoning engine
        """
        # First generate the base chain of thought
        base_chain = self.generate_chain_of_thought(problem)
        
        # Enhance with reasoning engine
        reasoning_result = reasoning_engine.perform_reasoning(
            input_data=problem,
            context=context
        )
        
        # Extract insights from reasoning result
        if 'final_output' in reasoning_result:
            insight = reasoning_result['final_output']
        elif 'conclusion' in reasoning_result:
            insight = reasoning_result['conclusion']
        else:
            insight = str(reasoning_result)
        
        # Add reasoning insights to the chain
        enhanced_chain = base_chain + [
            f"Reasoning Insight: {insight}",
            f"Reasoning Strategy: {reasoning_engine.name}"
        ]
        return enhanced_chain

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
        Enhanced action selection with context-aware prioritization.
        Uses a weighted scoring system based on thought content, context, and probabilistic confidence.
        
        Args:
            thoughts: List of reasoning steps from generate_chain_of_thought
            
        Returns:
            Selected action string
        """
        # Base weights for different actions
        action_weights = {
            "query_knowledge_base": 0.3,
            "run_consistency_check": 0.4,
            "forward_chaining": 0.5,
            "backward_chaining": 0.2,
            "request_human_input": 0.1
        }
        
        last_thought = thoughts[-1].lower()
        
        # Contextual boosting based on problem state
        context = self.get_current_context()
        if "low_confidence_environment" in context:
            action_weights["run_consistency_check"] += 0.3
            action_weights["request_human_input"] += 0.2
        
        # Keyword-based scoring
        keyword_scores = {
            "retriev": ("query_knowledge_base", 0.7),
            "verify": ("run_consistency_check", 0.8),
            "check": ("run_consistency_check", 0.6),
            "infer": ("forward_chaining", 0.7),
            "derive": ("forward_chaining", 0.6),
            "uncertain": ("request_human_input", 0.9),
            "conflict": ("run_consistency_check", 0.9),
            "goal": ("backward_chaining", 0.7)
        }
        
        # Score all thoughts, not just the last one
        for thought in thoughts:
            thought_lower = thought.lower()
            for keyword, (action, score) in keyword_scores.items():
                if keyword in thought_lower:
                    action_weights[action] += score
        
        # Probabilistic adjustment based on solution confidence
        if self.knowledge_base:
            avg_confidence = sum(self.knowledge_base.values()) / len(self.knowledge_base)
            if avg_confidence < 0.6:
                action_weights["run_consistency_check"] += 0.4
        
        # Select action with highest weight
        selected_action = max(action_weights, key=action_weights.get)
        
        # Fallback to forward chaining if below threshold
        if action_weights[selected_action] < 0.5:
            return "forward_chaining"
        
        return selected_action

    def execute_action(self, action: str) -> Dict[str, Any]:
        """
        Executes selected actions with enhanced reasoning and inter-agent communication.
        Integrates reasoning types and shares results via shared memory.
        """
        result = {"action": action, "success": False, "details": None}
        
        try:
            # Get current reasoning engine from ReAct loop
            reasoning_engine = getattr(self, 'current_reasoning_engine', None)
            
            if action == "query_knowledge_base":
                # Enhanced query using reasoning engine
                if reasoning_engine:
                    structured_query = reasoning_engine.structure_query(self.current_thoughts)
                else:
                    query_terms = " ".join(self.current_thoughts[-5:])
                    structured_query = {"terms": query_terms.split(), "context": self.get_current_context()}
                
                # Publish query to shared memory
                self.shared_memory.publish(
                    "knowledge_queries", 
                    {
                        "query": structured_query,
                        "source": self.name,
                        "timestamp": time.time()
                    }
                )
                
                # Get responses from other agents
                responses = self.shared_memory.get("knowledge_responses", [])
                relevant_facts = {}
                
                # Combine local and external knowledge
                for fact, conf in self.knowledge_base.items():
                    if any(term in str(fact) for term in structured_query.get("terms", [])):
                        relevant_facts[fact] = conf
                
                for response in responses:
                    if response["query"] == structured_query:
                        relevant_facts.update(response["facts"])
                
                result.update({
                    "success": True,
                    "details": {
                        "found_facts": len(relevant_facts),
                        "sample_facts": list(relevant_facts.items())[:3],
                        "reasoning_type": reasoning_engine.name if reasoning_engine else "default"
                    }
                })
                
            elif action == "run_consistency_check":
                # Validate with reasoning-enhanced consistency check
                recent_fact = None
                for thought in reversed(self.current_thoughts):
                    try:
                        recent_fact = self._parse_statement(thought.split(":")[-1])
                        break
                    except ValueError:
                        continue
                
                if recent_fact:
                    # Use reasoning engine for enhanced validation
                    if reasoning_engine:
                        validation = reasoning_engine.validate_fact(recent_fact, self.validation_engine)
                    else:
                        validation = self.validate_fact(recent_fact)
                    
                    # Publish validation request
                    self.shared_memory.publish(
                        "consistency_checks", 
                        {
                            "fact": recent_fact,
                            "source": self.name,
                            "timestamp": time.time()
                        }
                    )
                    
                    result.update({
                        "success": True,
                        "details": validation,
                        "reasoning_type": reasoning_engine.name if reasoning_engine else "default"
                    })
            
            elif action == "forward_chaining":
                # Reasoning-guided forward chaining
                if reasoning_engine:
                    new_facts = reasoning_engine.forward_chaining(
                        self.knowledge_base, 
                        self.rules,
                        max_iterations=10
                    )
                else:
                    new_facts = self.forward_chaining(max_iterations=10)
                
                # Publish new facts to shared memory
                if new_facts:
                    self.shared_memory.publish(
                        "new_inferred_facts", 
                        (list(new_facts.keys()), "forward_chaining")
                    )
                
                result.update({
                    "success": bool(new_facts),
                    "details": {
                        "new_facts_count": len(new_facts),
                        "sample_facts": list(new_facts.items())[:3],
                        "reasoning_type": reasoning_engine.name if reasoning_engine else "default"
                    }
                })
            
            elif action == "backward_chaining":
                # Enhanced backward chaining with reasoning
                target_fact = self._parse_statement(self.current_thoughts[-1].split(":")[-1])
                supporting_facts = []
                
                if reasoning_engine:
                    # Decompose goal using reasoning engine
                    subgoals = reasoning_engine.decompose_goal(target_fact)
                    for subgoal in subgoals:
                        # Publish subgoals to shared memory
                        self.shared_memory.publish(
                            "subgoal_requests",
                            {
                                "subgoal": subgoal,
                                "parent_goal": target_fact,
                                "source": self.name
                            }
                        )
                        # Collect supporting facts
                        supporting_facts.extend(self._find_supporting_facts(subgoal))
                else:
                    supporting_facts = self._find_supporting_facts(target_fact)
                
                result.update({
                    "success": bool(supporting_facts),
                    "details": {
                        "supporting_facts": supporting_facts[:5],
                        "reasoning_type": reasoning_engine.name if reasoning_engine else "default"
                    }
                })
            
            elif action == "request_human_input":
                # Generate focused question using reasoning engine
                if reasoning_engine:
                    question = reasoning_engine.generate_question(
                        self.current_thoughts,
                        self.get_current_context()
                    )
                else:
                    question = " ".join(self.current_thoughts[-3:])
                
                # Enhanced request with reasoning context
                request_data = {
                    "timestamp": time.time(),
                    "thoughts": self.current_thoughts,
                    "problem_context": self.get_current_context(),
                    "reasoning_type": reasoning_engine.name if reasoning_engine else "default",
                    "specific_question": question
                }
                
                self.shared_memory.append("human_intervention_requests", request_data)
                
                result.update({
                    "success": True,
                    "details": {
                        "question": question,
                        "reasoning_type": reasoning_engine.name if reasoning_engine else "default"
                    }
                })
            
            # Update rule weights based on action success
            self._update_rule_weights(action, result["success"])
            
        except Exception as e:
            logger.error(f"Action {action} failed: {str(e)}")
            result["details"] = str(e)
            self._update_rule_weights(action, False)
        
        return result
    
    # Helper method for backward chaining
    def _find_supporting_facts(self, target_fact):
        """Find facts supporting a target fact"""
        supporting_facts = []
        for fact in self.knowledge_base:
            if fact[2] == target_fact[0]:  # Object matches subject
                supporting_facts.append(fact)
        return supporting_facts

    def _is_goal_reached(self, context: Dict[str, Any]) -> bool:
        """
        Determines whether the reasoning process has reached its goal.
        Based on:
        - Confidence threshold
        - Fact convergence
        - Contradiction avoidance
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

    def forward_chaining(self) -> Dict[Tuple, float]:
        """
        Probabilistic forward chaining inference.

        Args:
            max_iterations: Maximum number of inference cycles

        Returns:
            New facts with their confidence scores
        """
        new_facts = {}
        applied_rules = defaultdict(int)  # Track rule applications
        for _ in range(self.max_iterations):
            current_new = {}

            # Explore new rules occasionally
            if random.random() < self.exploration_rate:
                self.rule_engine._discover_new_rules()

            for name, rule_func, weight in self.rules:
                if applied_rules[name] > self.rule_engine.max_circular_depth:
                    continue
                try:
                    inferred = rule_func(self.knowledge_base)
                    for fact, confidence in inferred.items():
                        weighted_conf = confidence * weight
                        if fact not in self.knowledge_base or weighted_conf > self.knowledge_base[fact]:
                            current_new[fact] = weighted_conf
                            self._update_rule_weights(name, True)
                        else:
                            self._update_rule_weights(name, False)
                        applied_rules[name] += 1
                except Exception as e:
                    logger.warning(f"Rule {name} failed: {e}")

            if not current_new:
                break

            new_facts.update(current_new)
            self.knowledge_base.update(current_new)

        # Probabilistic validation to inferred facts
        for fact, conf in current_new.items():
            self.add_fact(fact, conf)
            prob_conf = self.probabilistic_models.probabilistic_query(fact)
            # Combine rule confidence with probabilistic confidence
            weighted_conf = (conf * 0.6) + (prob_conf * 0.4)
            current_new[fact] = weighted_conf

        # --- Validation Phase ---
        rule_engine = self.rule_engine
        circular = rule_engine.detect_circular_rules()
        conflicts = rule_engine.detect_fact_conflicts()
        redundant = rule_engine.redundant_fact_check(confidence_margin=0.05)

        if circular:
            logger.warning(f"Circular rules detected: {circular}")
            rule_engine.adjust_circular_rule_weights(circular)
        if conflicts:
            logger.warning(f"Conflicting facts found: {conflicts}")
        if redundant:
            logger.info(f"Redundant facts skipped: {redundant}")

        # Save after batch updates
        self.shared_memory.set(
            "reasoning_agent:knowledge_base",
            self.knowledge_base
        )
        return new_facts # {k: v for k, v in new_facts.items() if k not in redundant}

    def _incremental_forward_chain(self, seed_facts: Dict[Tuple, float], max_iterations: int = 10):
        """Run a light inference pass starting from seed_facts."""
        new_facts = {}

        for _ in range(max_iterations):
            current_new = {}
            # Prioritize rules using agent's rule weights instead of probabilistic_models
            prioritized_rules = sorted(
                self.rules,
                key=lambda r: self.rule_weights.get(r[0], 1.0),  # Use agent's rule weights
                reverse=True
            )
            for name, rule_func, weight in prioritized_rules:
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
                    logger.warning(f"Rule {name} failed during incremental chaining: {e}")

            if not current_new:
                break
            new_facts.update(current_new)
            self.knowledge_base.update(current_new)

        logger.info(f"[IncrementalInference] Added {len(new_facts)} new facts.")

    def _parse_statement(self, statement: str) -> Tuple:
        """
        Advanced statement parsing using NLU engine for entity recognition.
        
        Args:
            statement: Natural language statement
            
        Returns:
            Structured fact tuple
        """
        frame = self.nlu_engine.parse(statement)
        entities = []
        for entity_type, entity_list in frame.entities.items():
            entities.extend(entity_list)

        if len(entities) >= 3:
            return (entities[0], entities[1], entities[2])
        elif len(entities) == 2:
            return (entities[0], "related_to", entities[1])

        # Enhanced parsing with simple entity recognition
        parsed = self.nlu_engine.get_entities

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
        # Check if embeddings exist in shared memory
        embedding = self.shared_memory.get(f"embedding:{root_node}")
        if not embedding:
            embedding = self._generate_embedding(root_node)
            self.shared_memory.put(
                key=f"embedding:{root_node}",
                value=embedding,
                priority=10  # High priority for frequent access
            )

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
                similarity = self.nlu_engine.wordlist.semantic_similarity(
                    node[2],  # Use the object part of the fact tuple
                    fact[2]   # Compare object-to-object
                )
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

    def _generate_embedding(self, root_node: Tuple) -> List[float]:
        """Generate embedding for a fact tuple using transformer or GloVe fallback"""
        # Convert fact tuple to natural language string
        fact_str = f"{root_node[0]} {root_node[1]} {root_node[2]}"
        
        # First try: Use transformer embedding from NLU engine
        try:
            if hasattr(self.nlu_engine.wordlist, '_get_transformer_embedding'):
                embedding = self.nlu_engine.wordlist._get_transformer_embedding(fact_str)
                if embedding:
                    return embedding
        except Exception as e:
            logger.warning(f"Transformer embedding failed: {str(e)}")
        
        # Fallback: Use GloVe vectors
        glove_vectors = getattr(self.nlu_engine.wordlist, 'glove_vectors', {})
        if not glove_vectors:
            logger.warning("GloVe vectors not available - using random embedding")
            return [random.gauss(0, 0.1) for _ in range(300)]  # Random fallback
        
        # Get embedding dimension from first vector
        sample_vec = next(iter(glove_vectors.values()), None)
        dim = len(sample_vec) if sample_vec else 300
        
        # Process each part of the fact tuple
        vectors = []
        for element in root_node:
            element_str = str(element)
            words = element_str.split()
            word_vecs = []
            
            for word in words:
                vec = glove_vectors.get(word.lower())
                if vec:
                    word_vecs.append(vec)
            
            if word_vecs:
                # Average word vectors in this element
                avg_vec = [sum(x) / len(word_vecs) for x in zip(*word_vecs)]
                vectors.append(avg_vec)
        
        if vectors:
            # Average vectors across all elements
            final_embedding = [sum(x) / len(vectors) for x in zip(*vectors)]
            return final_embedding
        
        # Ultimate fallback: zero vector
        return [0.0] * dim

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
    
    def detect_risk_factors(self, task: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect risk factors for a given task and state using:
        - Probabilistic inference
        - Knowledge base patterns
        - Contextual analysis
        """
        risk_factors = []
        context = self.get_current_context()
        evidence = self._extract_evidence_from_state(state)
        
        # 1. Check knowledge base for known risk patterns
        for fact, conf in self.knowledge_base.items():
            # Add null check and tuple validation
            if fact is None or not isinstance(fact, tuple) or len(fact) != 3:
                logger.warning(f"Skipping invalid fact in knowledge base: {fact}")
                continue
                
            if "risk" in fact[1].lower() and conf > 0.7:
                # Check relevance to current task/state
                similarity = self.nlu_engine.wordlist.semantic_similarity(task, fact[0])
                if similarity > 0.6:
                    risk_factors.append({
                        "type": fact[2],
                        "description": f"Known risk pattern: {fact}",
                        "severity": int(conf * 10),
                        "mitigation": self._get_mitigation_strategy(fact[2]),
                        "source": "Knowledge Base",
                        "confidence": conf
                    })
        
        # 2. Bayesian inference for hidden risks
        risk_nodes = self.probabilistic_models.pgmpy_bn.get_risk_nodes()
        for node in risk_nodes:
            prob = self.probabilistic_models.bayesian_inference(
                query=node,
                evidence=evidence
            )
            if prob > 0.65:
                risk_factors.append({
                    "type": node,
                    "description": f"Probabilistic risk detected in {node}",
                    "severity": int(prob * 8),  # Scale to 1-8
                    "mitigation": "Review system constraints",
                    "source": "Bayesian Network",
                    "confidence": prob
                })
        
        # 3. Rule-based risk detection
        risk_rules = {}
        try:
            risk_rules = self.rule_engine.get_rules_by_category("risk_detection")
        except AttributeError:
            logger.error("RuleEngine does not support get_rules_by_category")

        for rule_name, rule_func in risk_rules.items():
            try:
                rule_result = rule_func(self.knowledge_base, task, state)
                if rule_result.get("is_risky", False):
                    risk_factors.append({
                        "type": rule_result.get("risk_type", "RuleBasedRisk"),
                        "description": rule_result.get("description", f"Rule {rule_name} triggered"),
                        "severity": rule_result.get("severity", 5),
                        "mitigation": rule_result.get("mitigation", "Apply standard risk protocol"),
                        "source": f"Rule: {rule_name}",
                        "confidence": rule_result.get("confidence", 0.75)
                    })
            except Exception as e:
                logger.error(f"Risk rule {rule_name} failed: {str(e)}")
        
        # Fallback to default if no risks detected
        if not risk_factors and "high_uncertainty" in context:
            risk_factors.append({
                "type": "UncertaintyRisk",
                "description": "High uncertainty environment risk",
                "severity": 7,
                "mitigation": "Gather additional information",
                "source": "System Default",
                "confidence": 0.8
            })
        
        # Remove duplicates and low-confidence entries
        return self._deduplicate_risks(risk_factors)
    
    def detect_opportunity_factors(self, task: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect opportunity factors using:
        - Multi-hop reasoning
        - Resource availability analysis
        - Pattern recognition
        """
        opportunities = []
        context = self.get_current_context()
        
        # 1. Check for resource-based opportunities
        resource_availability = state.get("available_resources", {})
        for resource, amount in resource_availability.items():
            if amount > state.get("required_resources", {}).get(resource, 0) * 1.5:
                opportunities.append({
                    "type": f"Excess{resource.capitalize()}",
                    "description": f"Excess {resource} available for utilization",
                    "potential": min(10, int(amount/10)),  # Scale based on amount
                    "action": f"Reallocate {resource} to high-priority tasks",
                    "source": "Resource Analysis",
                    "confidence": min(0.9, amount/100)
                })
        
        # 2. Multi-hop reasoning for indirect opportunities
        #query = (task.id, "has_opportunity", None) 
        query = (task[0], "has_opportunity", None)
        opportunity_paths = self.probabilistic_models.multi_hop_reasoning(query)
        
        # Handle float returns (confidence scores)
        if isinstance(opportunity_paths, float):
            if opportunity_paths > 0.6:
                opportunities.append({
                    "type": "InferredOpportunity",
                    "description": f"Opportunity confidence: {opportunity_paths:.2f}",
                    "potential": int(opportunity_paths * 8),
                    "action": "Investigate opportunity connections",
                    "source": "Multi-hop Reasoning",
                    "confidence": opportunity_paths
                })
        elif isinstance(opportunity_paths, list):
            for path in opportunity_paths[:5]:  # Limit to top 5
                if path.get("confidence", 0) > 0.6:  # Use get with default
                    opportunities.append({
                        "type": path.get("concept", "InferredOpportunity"),
                        "description": f"Opportunity path: {' -> '.join(path.get('nodes', ['Unknown']))}",
                        "potential": int(path.get("confidence", 0) * 8),
                        "action": f"Explore connections with {path.get('nodes', ['Unknown'])[-1]}",
                        "source": "Multi-hop Reasoning",
                        "confidence": path.get("confidence", 0)
                    })
        
        # 3. Temporal opportunity detection
        if "time_sensitive" in context:
            time_based = self._detect_temporal_opportunities(task, state)
            opportunities.extend(time_based)
        
        # 4. Rule-based opportunities
        opportunity_rules = {}
        try:
            opportunity_rules = self.rule_engine.get_rules_by_category("opportunity_detection") or {}
        except AttributeError:
            logger.error("RuleEngine does not support get_rules_by_category")

        for rule_name, rule_func in opportunity_rules.items():
            try:
                rule_result = rule_func(self.knowledge_base, task, state)
                if rule_result.get("is_opportunity", False):
                    opportunities.append({
                        "type": rule_result.get("opportunity_type", "RuleBasedOpportunity"),
                        "description": rule_result.get("description", f"Rule {rule_name} triggered"),
                        "potential": rule_result.get("potential", 6),
                        "action": rule_result.get("action", "Execute opportunity protocol"),
                        "source": f"Rule: {rule_name}",
                        "confidence": rule_result.get("confidence", 0.7)
                    })
            except Exception as e:
                logger.error(f"Opportunity rule {rule_name} failed: {str(e)}")
        
        # Fallback to default if no opportunities detected
        if not opportunities:
            opportunities.append({
                "type": "DefaultOpportunity",
                "description": "General improvement potential",
                "potential": 5,
                "action": "Analyze task parameters for optimization",
                "source": "System Default",
                "confidence": 0.65
            })
        
        return self._prioritize_opportunities(opportunities)

    def _extract_evidence_from_state(self, state: Dict[str, Any]) -> Dict[str, bool]:
        """Convert state to Bayesian evidence format"""
        evidence = {}
        for key, value in state.items():
            if isinstance(value, bool):
                evidence[key] = value
            elif isinstance(value, (int, float)):
                evidence[key] = value > state.get(f"{key}_threshold", 0.5)
        return evidence
    
    def _get_mitigation_strategy(self, risk_type: str) -> str:
        """Retrieve mitigation strategy from knowledge base"""
        strategy = self.knowledge_base.get(
            (risk_type, "has_mitigation_strategy", None), 
            "Implement containment protocol"
        )
        return strategy
    
    def _deduplicate_risks(self, risks: List) -> List:
        """Remove duplicate risks and filter low-confidence"""
        seen = set()
        unique_risks = []
        for risk in risks:
            identifier = (risk["type"], risk["source"])
            if identifier not in seen and risk["confidence"] > 0.55:
                seen.add(identifier)
                unique_risks.append(risk)
        return sorted(unique_risks, key=lambda x: x["severity"], reverse=True)
    
    def _prioritize_opportunities(self, opportunities: List) -> List:
        """Sort opportunities by potential and confidence"""
        return sorted(
            opportunities, 
            key=lambda x: x["potential"] * x["confidence"], 
            reverse=True
        )[:5]  # Return top 5 opportunities
    
    def _detect_temporal_opportunities(self, task: str, state: Dict) -> List:
        """Detect time-sensitive opportunities"""
        opportunities = []
        if "deadline" in state and "completion_estimate" in state:
            time_window = state["deadline"] - state["completion_estimate"]
            if time_window > 0:
                opportunities.append({
                    "type": "TimeBuffer",
                    "description": f"Available time buffer: {time_window} units",
                    "potential": min(8, int(time_window/5)),
                    "action": "Allocate time to quality improvement",
                    "source": "Temporal Analysis",
                    "confidence": min(0.85, time_window/50)
                })
        return opportunities

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
            self.shared_memory.delete(f"fact:{fact}")
            logger.info(f"[GDPR] Forgotten fact: {fact}")
            return True
    
        except Exception as e:
            logger.error(f"[GDPR] Failed to forget fact {fact}: {e}")
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
        logger.info(f"[GDPR] Removed {len(to_remove)} facts related to subject: {subject}")
        return len(to_remove)
    
    def get_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text using NLU capabilities
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of key terms (nouns and named entities)
        """
        try:
            # Parse text with NLU engine
            frame = self.nlu_engine.parse(text)
            
            # Extract nouns and named entities
            key_terms = []
            for token in frame.tokens:
                if token.ner_tag or token.pos in ["NOUN", "PROPN"]:
                    key_terms.append(token.lemma)
            
            return list(set(key_terms))
        except Exception as e:
            logger.error(f"Key term extraction failed: {str(e)}")
            # Fallback: simple word extraction
            return [word for word in re.findall(r'\w+', text) if len(word) > 3]
    
    def forget_memory_keys(self, key_prefix: str):
        """
        Remove all shared memory keys starting with a specific prefix.
        """
        keys_to_clear = [key for key in self.shared_memory.keys() if key.startswith(key_prefix)]
        for key in keys_to_clear:
            self.shared_memory.set(key, None)
        logger.info(f"[GDPR] Cleared {len(keys_to_clear)} shared memory keys with prefix '{key_prefix}'.")

    def log_gdpr_request(self, request_type: str, target: str):
        with open("logs/gdpr_audit_log.jsonl", "a") as log:
            log.write(json.dumps({
                "timestamp": time.time(),
                "agent": self.name,
                "action": request_type,
                "target": target
            }) + "\n")

    def __repr__(self):
        """Safe representation to avoid recursion errors"""
        return (f"<Reasoning Agent v{__version__} "
                f"| rules: {len(self.rules)}, "
                f"facts: {len(self.knowledge_base)}>")
    
    def parse_goal(self, goal_description: str) -> Dict[str, Any]:
        """
        Parse a natural language goal description into a structured planning task.
        
        Args:
            goal_description (str): Natural language goal, e.g., "Increase market share"
        
        Returns:
            dict: Parsed components such as action, target_state, and optional deadline
        """
        try:
            # Basic verb extraction using simple heuristics
            tokens = goal_description.lower().split()
            if len(tokens) < 2:
                raise ValueError("Insufficient information to parse goal.")
    
            # Assume structure: [verb/action] + [target concept]
            action_verb = tokens[0]
            target_concept = "_".join(tokens[1:])  # e.g., "market share" → "market_share"
    
            # Create structured response
            return {
                "action": action_verb + "_" + target_concept,   # e.g., "increase_market_share"
                "target_state": {
                    target_concept: "increased"  # Default symbolic goal
                },
                "deadline": None
            }
        except Exception as e:
            logger.error(f"[Reasoning Agent] Failed to parse goal '{goal_description}': {e}")
            raise
    
    def predict(self, state: Any = None) -> Dict[str, Any]:
        """
        Predicts the confidence level for a given fact or query.
        
        Args:
            state (Any, optional): Can be:
                - A fact tuple (subject, predicate, object)
                - A natural language query string
                - None (returns random fact)
                
        Returns:
            Dict[str, Any]: Structured prediction containing:
                - fact: The fact being evaluated
                - confidence: Confidence score (0.0-1.0)
                - type: Prediction type (random_fact, direct_query, parsed_query)
        """
        # Handle None state - return random fact
        if state is None:
            if not self.knowledge_base:
                return {"fact": None, "confidence": 0.0, "type": "empty_knowledge_base"}
            
            fact = random.choice(list(self.knowledge_base.keys()))
            return {
                "fact": fact,
                "confidence": self.knowledge_base[fact],
                "type": "random_fact"
            }
        
        # Handle tuple input - direct fact lookup
        if isinstance(state, tuple) and len(state) == 3:
            confidence = self.knowledge_base.get(state, 0.0)
            return {
                "fact": state,
                "confidence": confidence,
                "type": "direct_query"
            }
        
        # Handle string input - parse and evaluate
        if isinstance(state, str):
            try:
                # Try to parse the string into a fact
                parsed_fact = self._parse_statement(state)
                confidence = self.knowledge_base.get(parsed_fact, 0.0)
                return {
                    "fact": parsed_fact,
                    "confidence": confidence,
                    "type": "parsed_query"
                }
            except ValueError:
                # Fallback to keyword-based confidence
                matching_facts = [
                    (fact, conf) 
                    for fact, conf in self.knowledge_base.items()
                    if any(term in str(fact) for term in state.split())
                ]
                if matching_facts:
                    best_fact, confidence = max(matching_facts, key=lambda x: x[1])
                    return {
                        "fact": best_fact,
                        "confidence": confidence,
                        "type": "keyword_match"
                    }
                return {
                    "fact": None,
                    "confidence": 0.0,
                    "type": "no_matches"
                }
        
        # Unsupported input type
        return {
            "error": "Unsupported input type",
            "type": "input_error"
        }
    
    def get_probability_grid(self, agent_pos=None, target_pos=None):
        """
        Get probability grid from probabilistic models
        :param agent_pos: Current agent position
        :param target_pos: Target position
        :return: Probability grid as numpy array
        """
        return self.probabilistic_models.get_grid_state(agent_pos, target_pos)
    
    def reason(self, problem: Any, reasoning_type: str, context: dict = None) -> Any:
        """
        Perform reasoning using specified reasoning type or combination
        Args:
            problem: Input data for reasoning
            reasoning_type: Type of reasoning (e.g., 'abduction', 'deduction+induction')
            context: Additional context information
        Returns:
            Result of the reasoning process
        """
        try:
            reasoning_engine = self.reasoning_strategies.create(reasoning_type)
            return reasoning_engine.perform_reasoning(problem, context or {})
        except Exception as e:
            self.log_step(f"Reasoning failed: {str(e)}", "error")
            return None
