"""
Reasoning Agent for Scalable Autonomous Intelligence
Features:
- Knowledge representation with probabilistic confidence
- Rule learning and adaptation
- Advanced NLP capabilities
- Probabilistic reasoning
- Multiple inference methods
"""

import json
import re
import yaml
import math
import logging as logger
import itertools
import random
import hashlib
import numpy as np

from collections import defaultdict, OrderedDict
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional
from pathlib import Path
from src.agents.base_agent import BaseAgent
from src.agents.language.resource_loader import ResourceLoader
from src.agents.reasoning.rule_engine import RuleEngine
from src.agents.reasoning.probabilistic_models import ProbabilisticModels
from src.agents.reasoning.validation import (
        detect_circular_rules,
        detect_fact_conflicts,
        redundant_fact_check
    )


def identity_rule(kb):
    return {(s, p, o): 1.0 for (s, p, o) in kb if p == 'is'}

def transitive_rule(kb):
    new_facts = {}
    for (a, _, b1), conf1 in kb.items():
        for (b2, _, c), conf2 in kb.items():
            if b1 == b2:
                new_facts[(a, 'is', c)] = min(conf1, conf2)
    return new_facts

class ReasoningAgent(BaseAgent):
    """
    Initialize the Reasoning Agent with learning capabilities.
    """
    def __init__(self, shared_memory, agent_factory, tuple_key,
                 storage_path: str = "src/agents/knowledge/knowledge_db.json",
                 contradiction_threshold=0.25,
                 rule_validation: Dict = None,
                 nlp_integration: Dict = None,
                 inference: Dict = None,
                 llm: Any = None,
                 language_agent: Any = None,
                 args=(), kwargs={}):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory
        )
        self.hypothesis_graph = defaultdict(
            lambda: {
                'nodes': set(),
                'edges': defaultdict(float),
                'confidence': 0.0
            }
        )

        # Core components
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.llm = llm
        self.language_agent = language_agent
        self.knowledge_base = {}
        self.bayesian_network = {}
        self.rule_engine = RuleEngine()
        self.rules = []
        self.rule_weights = {}

        # Ensure storage_path is set before it's used
        self.storage_path = storage_path

        # Add default rules
        self.add_rule(identity_rule, rule_name="IdentityRule", weight=1.0)
        self.add_rule(transitive_rule, rule_name="TransitiveIsRule", weight=0.8)

        # Load YAML config if available
        config_path = "src/agents/reasoning/templates/reasoning_config.yaml"
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                external_config = yaml.safe_load(f)
                self.inference_settings = external_config.get('inference', {})
                self.rule_validation = external_config.get('rules', {})
                self.nlp_config = external_config.get('nlp', {})
                self.storage_path = external_config.get('storage', {}).get('knowledge_db', self.storage_path)

        # Fallback configurations if not set
        self.contradiction_threshold = contradiction_threshold
        self.rule_validation = self.rule_validation or {
            'enable': True,
            'min_soundness_score': 0.7,
            'max_circular_depth': 3
        }
        self.nlp_config = nlp_integration or self.nlp_config or {
            'sentence_transformer': 'your-internal-model',
            'tokenizer': self.language_agent.tokenizer if language_agent else None
        }
        self.inference_settings = inference or self.inference_settings or {
            'default_chain_length': 5,
            'neuro_symbolic_weight': 0.4,
            'max_hypotheses': 100,
            'exploration_rate': 0.1,
            'llm_fallback': {
                'enable': True,
                'temperature': 0.3,
                'max_tokens': 100
            }
        }

        # Initialize components
        self._load_knowledge()
        self._initialize_nlp()
        self._initialize_probabilistic_models()

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

    def _initialize_nlp(self):
        """
        Advanced NLP initialization with multi-layered linguistic features.
        Incorporates concepts from:
        - Mikolov et al. (2013) Word2Vec embeddings
        - Manning et al. (2014) Stanford CoreNLP architecture
        - Bird et al. (2009) NLTK design patterns
        """
        # Core linguistic resources
        self.vocab = self._build_core_lexicon()
        self.stop_words = self._load_stop_words()
        self.morphology = self._initialize_morphology()
        self.semantic_frames = self._create_semantic_frames()
        
        # Embedding spaces
        self.word_vectors = self._init_embeddings()
        self.dependency_grammar = self.rule_engine._create_dependency_rules()
        self.pos_tagger = self._build_rule_based_tagger()
        
        # Discourse features
        self.cohesion_metrics = {'entity_grid': {}, 'coref_chains': {}}
        self.pragmatic_rules = self.rule_engine._load_pragmatic_heuristics()

    def _build_core_lexicon(self) -> Set[str]:
        """Construct foundational lexicon with psycholinguistic priors"""
        base_words = {
            # Closed-class words
            'is', 'are', 'the', 'a', 'an', 'and', 'or', 'not', 'all', 'some', 'none',
            'true', 'false', 'in', 'on', 'at', 'of', 'to', 'for', 'with', 'by',
            
            # Semantic primes (Wierzbicka, 1996)
            'I', 'you', 'someone', 'something', 'do', 'happen', 'know', 'think',
            'want', 'feel', 'see', 'hear', 'say', 'word', 'true', 'good', 'bad',
            
            # Basic ontological categories
            'time', 'space', 'object', 'event', 'action', 'property', 'quantity'
        }
        return base_words

    def _load_stop_words(self) -> Set[str]:
        """Curated stop words list with domain adaptation"""
        return {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'of',
            'at', 'by', 'for', 'with', 'to', 'from', 'in', 'on', 'that', 'this',
            'these', 'those', 'is', 'are', 'was', 'were', 'have', 'has', 'had'
        }

    def _initialize_morphology(self) -> Dict[str, dict]:
        """Morphological analyzer with stemming and lemmaization"""
        return {
            'stemmer': self._porter_style_stemmer(),
            'lemmatizer': self._lemmatization_rules(),
            'inflection_rules': self._inflection_patterns()
        }

    def _init_embeddings(self) -> Dict[str, np.ndarray]:
        """Distributional semantics with simplified word embeddings"""
        embed_dim = 50  # Reduced dimension for efficiency
        return {word: np.random.normal(scale=0.1, size=embed_dim)
                for word in self.vocab}

    def _build_rule_based_tagger(self) -> Dict[str, str]:
        """Regex-based POS tagger with context rules"""
        patterns = [
            (r'.*ing$', 'VBG'),               # gerunds
            (r'.*ed$', 'VBD'),                 # past tense
            (r'.*es$', 'VBZ'),                 # 3rd singular
            (r'.*ould$', 'MD'),                # modals
            (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # numbers
            (r'.*', 'NN')                      # default noun
        ]
        return {'patterns': patterns, 'context_rules': self._context_sensitive_rules()}

    def _create_semantic_frames(self) -> Dict[str, dict]:
        """Frame semantics inventory based on FrameNet concepts"""
        return {
            'Motion': {'roles': ['Agent', 'Theme', 'Path'], 'verbs': ['move', 'go', 'travel']},
            'Transfer': {'roles': ['Donor', 'Recipient', 'Theme'], 'verbs': ['give', 'send', 'receive']},
            'Cognitive': {'roles': ['Cognizer', 'Content'], 'verbs': ['think', 'believe', 'consider']}
        }

    def _porter_style_stemmer(self) -> Callable:
        """Implementation of Porter algorithm key steps"""
        step1_suffixes = {
            'sses': 'ss',
            'ies': 'i',
            'ss': 'ss',
            's': ''
        }
        return lambda word: next(
            (word[:-len(suf)] + repl for suf, repl in step1_suffixes.items()
             if word.endswith(suf)), word)

    def _lemmatization_rules(self) -> Dict[str, str]:
        """Exception lists and transformation rules"""
        return {
            'is': 'be',
            'are': 'be',
            'were': 'be',
            'went': 'go',
            'better': 'good',
            'best': 'good'
        }

    def _inflection_patterns(self) -> Dict[str, Union[List[Tuple], Dict[str, str]]]:
        """Return regex patterns and irregular mappings for inflection handling."""
        return {
            # Ordered regex patterns (most specific first)
            'regular': [
                # --- Noun Plurals ---
                (r'(?i)([aeiou]y)s$', r'\1y'),         # toys -> toy
                (r'(?i)([^aeiou]y)s$', r'\1y'),        # babies -> baby (but catches "boys" -> "boy")
                (r'(?i)(ss|sh|ch|x|z)es$', r'\1'),     # buses -> bus, dishes -> dish
                (r'(?i)(m|l)ice$', r'\1ouse'),         # mice -> mouse, lice -> louse
                (r'(?i)([ft]eeth)$', r'\1ooth'),       # teeth -> tooth, feet -> foot
                (r'(?i)([a-z]+[^aeiou])ies$', r'\1y'), # cities -> city
                (r'(?i)([a-z]+[aeiou])s$', r'\1'),     # radios -> radio
                
                # --- Verb Conjugations ---
                (r'(?i)(\w+)(ed)$', r'\1'),            # walked -> walk
                (r'(?i)(\w+)(ing)$', r'\1'),           # running -> run
                (r'(?i)(\w+)(s)$', r'\1'),             # walks -> walk
                
                # --- Adjectives/Adverbs ---
                (r'(?i)(\w+)(er)$', r'\1'),            # bigger -> big
                (r'(?i)(\w+)(est)$', r'\1'),           # biggest -> big
                (r'(?i)(\w+)(ly)$', r'\1'),            # quickly -> quick
            ],
            
            # Irregular forms (inflected -> base)
            'irregular': {
                # Nouns
                'children': 'child',
                'men': 'man',
                'women': 'woman',
                'people': 'person',
                'geese': 'goose',
                'mice': 'mouse',
                
                # Verbs
                'went': 'go',
                'were': 'be',
                'ate': 'eat',
                'ran': 'run',
                'spoke': 'speak',
                
                # Adjectives
                'better': 'good',
                'best': 'good',
                'worse': 'bad',
                'worst': 'bad'
            }
        }

    def _context_sensitive_rules(self) -> List[tuple]:
        """Brill-style contextual tagging rules"""
        return [
            (('NN', 'VB'), ('PREVTAG', 'DT'), 'NN'),
            (('VB', 'NN'), ('NEXTTAG', 'NN'), 'VB'),
            (('JJ', 'NN'), ('PREVTAG', 'RB'), 'JJ')
        ]
        
    def _initialize_probabilistic_models(self):
        """Initialize probabilistic reasoning structures."""
        self.bayesian_network = {
            'nodes': set(),
            'edges': set(),
            'cpt': {}  # Conditional Probability Tables
        }

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
        circular = detect_circular_rules(self.rules)
        conflicts = detect_fact_conflicts(self.knowledge_base)
        redundant = redundant_fact_check(new_facts, self.knowledge_base)

        if circular:
            self.logger.warning(f"Circular rules detected: {circular}")
        if conflicts:
            self.logger.warning(f"Conflicting facts found: {conflicts}")
        if redundant:
            self.logger.info(f"Redundant facts skipped: {redundant}")

        return {k: v for k, v in new_facts.items() if k not in redundant}

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
        Hybrid rule-based/statistical NER with semantic normalization.
        Implements features from:
        - Lample et al. (2016) Neural Architectures for Named Entity Recognition
        - Regular expression patterns from Cunningham et al. (2002) GATE system
        """
        if text in self.entity_recognition:
            return self.entity_recognition[text]

        # Semantic normalization pipeline
        normalized = text.lower()
        entities = []

        # Rule-based patterns with priority
        patterns = OrderedDict([
            ('DATE', r'\b(\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?,? \d{4})\b'),
            ('TIME', r'\b\d{1,2}:\d{2}(?::\d{2})?\b(?: [APap][Mm])?'),
            ('PERSON', r'\b([A-Z][a-z]+ [A-Z][a-z]+-?[A-Za-z]*)\b'),
            ('GPE', r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b(?=\s+(?:County|State|Republic|Province|City))'),
            ('ORG', r'\b([A-Z][a-z]+(?: [A-Za-z0-9&]+){1,3} (?:Inc|Ltd|LLC|Corp|University))\b')
        ])

        # Multi-pass recognition with confidence scoring
        entity_conf = defaultdict(float)
        for ent_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                span = match.span()
                raw_entity = text[span[0]:span[1]]

                # Conflict resolution: prefer longer matches and higher priority types
                existing = next((e for e in entities if e['start'] <= span[0] and e['end'] >= span[1]), None)
                if not existing:
                    entities.append({
                        'text': raw_entity,
                        'type': ent_type,
                        'start': span[0],
                        'end': span[1],
                        'confidence': 0.9  # Base confidence for rule-based matches
                    })
                    entity_conf[raw_entity] = max(entity_conf[raw_entity], 0.9)
                elif existing['confidence'] < 0.9:
                    entities.remove(existing)
                    entities.append({
                        'text': raw_entity,
                        'type': ent_type,
                        'start': span[0],
                        'end': span[1],
                        'confidence': 0.9
                    })

        # Statistical disambiguation using knowledge base
        for ent in entities:
            # Check against known entities in KB
            kb_matches = [k for k in self.knowledge_base 
                          if k[1] == 'is' and str(k[0]).lower() == ent['text'].lower()]
            
            if kb_matches:
                ent['confidence'] = min(ent['confidence'] + 0.1 * len(kb_matches), 1.0)
                ent_type = kb_matches[0][2]
                if ent_type in patterns:
                    ent['type'] = ent_type

        # Build normalized text with entity markers
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        result = []
        last_pos = 0
        
        for ent in sorted_entities:
            result.append(normalized[last_pos:ent['start']])
            result.append(f"[{ent['text']}:{ent['type']}({ent['confidence']:.2f})]")
            last_pos = ent['end']
            
        result.append(normalized[last_pos:])
        marked_text = ''.join(result)
        
        self.entity_recognition[text] = marked_text
        return marked_text

    def _tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with linguistic features"""
        # Basic tokenization with regex
        tokens = re.findall(r'''
            \b\w+(?:['’]\w+)?        # Words with contractions
            | \d+\.?\d*              # Numbers
            | \S\W+                  # Special characters
            ''', text, re.X)
        
        # Normalization pipeline
        processed = []
        for token in tokens:
            # Case normalization
            if self.config.get('lowercase', True):
                token = token.lower()
                
            # Stemming
            if self.config.get('stemming', False):
                token = self.morphology['stemmer'](token)
                
            # Remove residual punctuation
            token = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', token)
            
            if token:
                processed.append(token)
                
        return processed

    def _update_vocabulary(self, text: str):
        if not hasattr(self, "vocab_vectors"):
            self.vocab_vectors = {}
        
        tokens = self._tokenize(text)
        for token in tokens:
            self.vocab_vectors[token.lower()] = self._enhanced_binary_encode(token)

    def _enhanced_binary_encode(self, word: str, dim: int = 64) -> np.ndarray:
        """
        Enhanced binary encoding for discrete semantic hashing.
        """
        hash_val = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        bin_hash = bin(hash_val)[2:].zfill(dim)
        return np.array([int(b) for b in bin_hash[-dim:]], dtype=np.uint8)

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity using mean-pooled word vectors and cosine similarity.
        Falls back to token overlap if embeddings are missing.
        """
        def embed(text: str) -> Optional[np.ndarray]:
            tokens = self._tokenize(text)
            vectors = [self.word_vectors[t] for t in tokens if t in self.word_vectors]
            if not vectors:
                return None
            return np.mean(vectors, axis=0)

        vec1 = embed(text1)
        vec2 = embed(text2)

        if vec1 is None or vec2 is None:
            # Fallback: Jaccard token overlap
            t1, t2 = set(self._tokenize(text1)), set(self._tokenize(text2))
            return len(t1 & t2) / max(len(t1 | t2), 1)

        numerator = np.dot(vec1, vec2)
        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return float(numerator / denominator) if denominator != 0 else 0.0

    def _text_to_vector(self, text: str) -> Dict[str, float]:
        """Convert text to simple word vector."""
        words = re.findall(r'\w+', text.lower())
        vec = defaultdict(float)
        for word in words:
            if word in self.word_vectors:
                for w, val in self.word_vectors[word].items():
                    vec[w] += val
        return dict(vec)
    
    def _build_hypothesis_graph(self, root_node: Tuple):
        """Constructs multi-layered hypothesis graph with confidence scoring"""
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
                similarity = self.semantic_similarity(str(node), str(fact))
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

# if __name__ == "__main__":
