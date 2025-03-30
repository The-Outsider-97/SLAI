"""
Advanced Reasoning Agent for Scalable Autonomous Intelligence
Features:
- Knowledge representation with probabilistic confidence
- Rule learning and adaptation
- Advanced NLP capabilities
- Probabilistic reasoning
- Multiple inference methods
"""

import json
import re
import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Set, Tuple, Union
from pathlib import Path
import numpy as np

class ReasoningAgent:
    def __init__(self, storage_path: str = "knowledge_db.json"):
        """
        Initialize the Reasoning Agent with learning capabilities.
        
        Args:
            storage_path: Path to persist knowledge base and learned models
        """
        # Knowledge representation with confidence scores
        self.knowledge_base: Dict[Tuple, float] = defaultdict(float)
        self.rules: List[Tuple[str, Callable, float]] = []  # (name, rule, weight)
        self.rule_weights: Dict[str, float] = defaultdict(float)
        self.storage_path = Path(storage_path)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.decay_factor = 0.99
        
        # NLP components
        self.vocab = set()
        self.word_vectors = {}
        self.entity_recognition = {}
        
        # Probabilistic reasoning
        self.bayesian_network = {}
        
        self._load_knowledge()
        self._initialize_nlp()
        self._initialize_probabilistic_models()

    def _initialize_nlp(self):
        """Initialize basic NLP components."""
        # Placeholder for more advanced NLP initialization
        self.vocab = {
            'is', 'are', 'a', 'an', 'the', 'of', 'in', 'on', 'at', 
            'and', 'or', 'not', 'all', 'some', 'none', 'true', 'false'
        }
        
    def _initialize_probabilistic_models(self):
        """Initialize probabilistic reasoning structures."""
        self.bayesian_network = {
            'nodes': set(),
            'edges': set(),
            'cpt': {}  # Conditional Probability Tables
        }

    def _load_knowledge(self):
        """Load knowledge and learned models from storage."""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.knowledge_base = {tuple(k): v for k, v in data.get('knowledge', {}).items()}
                self.rules = [(r[0], eval(r[1]), r[2]) for r in data.get('rules', [])]  # Note: eval is unsafe here - use proper serialization in production
                self.rule_weights = data.get('rule_weights', {})
                self.bayesian_network = data.get('bayesian_network', {})

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

    def add_fact(self, fact: Union[Tuple, str], confidence: float = 1.0) -> bool:
        """
        Add a fact with confidence score.
        
        Args:
            fact: Either a tuple (subject, predicate, object) or string statement
            confidence: Probability/confidence score (0.0 to 1.0)
            
        Returns:
            True if fact was added/modified
        """
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

    def _update_vocabulary(self, text: str):
        """Update NLP vocabulary and simple word vectors."""
        words = re.findall(r'\w+', text.lower())
        for word in words:
            if word not in self.vocab:
                self.vocab.add(word)
                # Simple binary word vector (could be enhanced)
                self.word_vectors[word] = {w: 1 if w == word else 0 for w in self.vocab}

    def _parse_statement(self, statement: str) -> Tuple:
        """
        Advanced statement parsing with entity recognition.
        
        Args:
            statement: Natural language statement
            
        Returns:
            Structured fact tuple
        """
        # Enhanced parsing with simple entity recognition
        parsed = self._recognize_entities(statement)
        
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

    def _recognize_entities(self, text: str) -> str:
        """
        Simple entity recognition and normalization.
        
        Args:
            text: Input text
            
        Returns:
            Text with recognized entities marked
        """
        # Placeholder for more advanced NER
        # Currently just does simple normalization
        normalized = text.lower()
        
        # Cache recognized entities
        if text not in self.entity_recognition:
            self.entity_recognition[text] = normalized
            
        return self.entity_recognition[text]

    def add_rule(self, rule: Callable, rule_name: str = None, weight: float = 1.0) -> None:
        """
        Add an inference rule with initial weight.
        
        Args:
            rule: Function that takes KB and returns new facts with confidence
            rule_name: Optional identifier for the rule
            weight: Initial weight/importance of the rule
        """
        if not callable(rule):
            raise ValueError("Rule must be callable")
            
        name = rule_name or rule.__name__
        self.rules.append((name, rule, weight))
        self.rule_weights[name] = weight
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
            if random.random() < self.exploration_rate:
                self._discover_new_rules()
            
            for name, rule, weight in self.rules:
                try:
                    # Rules now return (fact, confidence) pairs
                    inferred = rule(self.knowledge_base)
                    for fact, confidence in inferred.items():
                        weighted_conf = confidence * weight
                        
                        if fact not in self.knowledge_base or weighted_conf > self.knowledge_base[fact]:
                            current_new[fact] = weighted_conf
                            self._update_rule_weights(name, True)
                        else:
                            self._update_rule_weights(name, False)
                except Exception as e:
                    print(f"Rule {name} failed: {e}")
                    self._update_rule_weights(name, False)
            
            if not current_new:
                break
                
            new_facts.update(current_new)
            self.knowledge_base.update(current_new)
        
        if new_facts:
            self._save_knowledge()
        return new_facts

    def _discover_new_rules(self):
        """
        Machine learning method to discover new rules from knowledge patterns.
        """
        # Placeholder for actual rule learning algorithm
        # This could use association rule mining, inductive logic programming, etc.
        pass

    def probabilistic_query(self, fact: Tuple, evidence: Dict[Tuple, bool] = None) -> float:
        """
        Probabilistic reasoning about a fact given evidence.
        
        Args:
            fact: Fact to query
            evidence: Observed facts with their truth values
            
        Returns:
            Probability estimate (0.0 to 1.0)
        """
        # Simple Bayesian inference
        if evidence is None:
            return self.knowledge_base.get(fact, 0.0)
        
        # Placeholder for proper Bayesian network inference
        # Currently just does simple weighted combination
        total_weight = 0.0
        weighted_sum = 0.0
        
        for e_fact, e_value in evidence.items():
            if e_fact in self.knowledge_base:
                weight = self.knowledge_base[e_fact]
                total_weight += weight
                if e_value:
                    weighted_sum += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return self.knowledge_base.get(fact, 0.0)

    def bayesian_inference(self, query: str, evidence: Dict[str, bool]) -> float:
        """
        Perform Bayesian network inference.
        
        Args:
            query: Node to query
            evidence: Observed nodes and their states
            
        Returns:
            Probability of query given evidence
        """
        # Placeholder for proper Bayesian network inference
        # Currently returns simple marginal probability
        return self.knowledge_base.get((query, 'probability', None), 0.5)

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple cosine similarity using word vectors
        vec1 = self._text_to_vector(text1)
        vec2 = self._text_to_vector(text2)
        
        dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in set(vec1) | set(vec2))
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _text_to_vector(self, text: str) -> Dict[str, float]:
        """Convert text to simple word vector."""
        words = re.findall(r'\w+', text.lower())
        vec = defaultdict(float)
        for word in words:
            if word in self.word_vectors:
                for w, val in self.word_vectors[word].items():
                    vec[w] += val
        return dict(vec)

    def learn_from_interaction(self, feedback: Dict[Tuple, bool]):
        """
        Learn from user feedback about facts.
        
        Args:
            feedback: Dictionary of facts and whether they were correct
        """
        for fact, is_correct in feedback.items():
            current_conf = self.knowledge_base.get(fact, 0.0)
            if is_correct:
                new_conf = current_conf + self.learning_rate * (1 - current_conf)
            else:
                new_conf = current_conf * self.decay_factor
            self.knowledge_base[fact] = new_conf
        
        self._save_knowledge()
