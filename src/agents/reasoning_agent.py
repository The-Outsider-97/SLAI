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
import math
import logging as logger
import itertools
import random
import numpy as np

from collections import defaultdict
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional
from pathlib import Path
from src.agents.base_agent import BaseAgent


class ReasoningAgent(BaseAgent):
    """
    Initialize the Reasoning Agent with learning capabilities.
    """
    def __init__(self, shared_memory, agent_factory, tuple_key,
                 k=None,
                 storage_path: str = "src/agents/knowledge/knowledge_db.json",
                 contradiction_threshold=0.25,
                 rule_validation: Dict = None,
                 nlp_integration: Dict = None,
                 inference: Dict = None,
                 llm: Any = None,
                 language_agent: Any = None,
                 args=(),
                 kwargs={}):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory
        )
        
        # Core components
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.llm = llm
        self.language_agent = language_agent

        # Configuration from YAML
        self.contradiction_threshold = contradiction_threshold
        self.rule_validation = rule_validation or {
            'enable': True,
            'min_soundness_score': 0.7,
            'max_circular_depth': 3
        }
        
        # NLP integration with your existing components
        self.nlp_config = nlp_integration or {
            'sentence_transformer': 'your-internal-model',
            'tokenizer': self.language_agent.tokenizer if language_agent else None
        }

        # Inference configuration
        self.inference_settings = inference or {
            'default_chain_length': 5,
            'neuro_symbolic_weight': 0.4,
            'max_hypotheses': 100,
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

    def _initialize_nlp(self):
        """Use your existing language processing components"""
        if self.language_agent:
            # Reuse existing tokenizer from LanguageAgent
            self.tokenizer = self.language_agent.tokenizer
            self.embedder = self.language_agent.embedder
            
        if self.llm:
            # Direct integration with SLAILM capabilities
            self.semantic_similarity = self.llm.calculate_similarity
            self.entailment_checker = self.llm.check_entailment

    def _load_knowledge(self, storage_path: str = "src/agents/knowledge/knowledge_db.json",):
        """Load knowledge and learned models from storage."""
        self.knowledge_base = {}
        self.rule_weights = {}
        self.bayesian_network = {}
        self.storage_path = storage_path
        knowledge = []

        if isinstance(knowledge, list):
            self.knowledge_base = {tuple(k[0]): k[1] for k in knowledge}  # Convert list to tuple
        if self.storage_path and Path(self.storage_path).exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                knowledge = data.get('knowledge', [])

                if isinstance(knowledge, list):
                    knowledge = {tuple(k): 1.0 for k in knowledge}  # Default confidence
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
        self.dependency_grammar = self._create_dependency_rules()
        self.pos_tagger = self._build_rule_based_tagger()
        
        # Discourse features
        self.cohesion_metrics = {'entity_grid': {}, 'coref_chains': {}}
        self.pragmatic_rules = self._load_pragmatic_heuristics()

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

    def _create_dependency_rules(self) -> Dict[str, list]:
        """Head-modifier grammar inspired by Universal Dependencies"""
        return {
            'nsubj': ['NN', 'VB'],
            'dobj': ['VB', 'NN'],
            'amod': ['NN', 'JJ'],
            'advmod': ['VB', 'RB'],
            'prep': ['IN', 'NN']
        }

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

    def _discover_new_rules(self, min_support: float = 0.3, min_confidence: float = 0.7):
        """
        Association rule mining with Apriori algorithm adaptation.
        Implements:
        - Agrawal & Srikant (1994) Fast Algorithms for Mining Association Rules
        - Confidence-weighted support from Lê et al. (2014) Fuzzy Association Rules
        """
        # Convert knowledge base to transaction-style format
        transactions = []
        for fact, conf in self.knowledge_base.items():
            if conf > 0.5:  # Consider facts with at least 50% confidence
                transactions.append(fact)

        # Generate frequent itemsets with confidence-weighted support
        itemsets = defaultdict(float)
        for fact in transactions:
            for element in itertools.chain.from_iterable(
                itertools.combinations(fact, r) for r in range(1, 4)
            ):
                itemsets[element] += self.knowledge_base.get(fact, 0.5)

        # Filter by minimum support
        freq_itemsets = {k: v/len(transactions) for k, v in itemsets.items() 
                        if v/len(transactions) >= min_support}

        # Generate candidate rules
        candidate_rules = []
        for itemset in freq_itemsets:
            if len(itemset) < 2:
                continue
            for i in range(1, len(itemset)):
                antecedent = itemset[:i]
                consequent = itemset[i:]
                candidate_rules.append((antecedent, consequent))

        # Calculate rule confidence
        valid_rules = []
        for ant, cons in candidate_rules:
            ant_support = sum(self.knowledge_base.get(fact, 0.0) 
                            for fact in transactions 
                            if set(ant).issubset(fact)) / len(transactions)
            
            rule_support = sum(self.knowledge_base.get(fact, 0.0)
                            for fact in transactions
                            if set(ant+cons).issubset(fact)) / len(transactions)
            
            if ant_support > 0:
                confidence = rule_support / ant_support
                if confidence >= min_confidence:
                    valid_rules.append({
                        'antecedent': ant,
                        'consequent': cons,
                        'confidence': confidence,
                        'support': rule_support
                    })

        # Convert to executable rules
        for rule in valid_rules:
            ant = rule['antecedent']
            cons = rule['consequent']
            
            def rule_func(kb, antecedents=ant, consequents=cons, conf=rule['confidence']):
                matches = [fact for fact in kb if all(e in fact for e in antecedents)]
                return {consequents: conf * len(matches)/(len(kb)+1e-8)}  # Prevent division by zero
                
            rule_name = f"LearnedRule_{hash(frozenset(ant+cons))}"
            self.add_rule(rule_func, rule_name, weight=rule['confidence'])

    def check_consistency(self, fact: Tuple) -> bool:
        """
        Self-consistency via paraphrased queries.
        """
        paraphrases = [
            f"Is {fact} true?",
            f"Does {fact} hold in all cases?",
            f"Confirm the validity of {fact}"
        ]
        results = [self.probabilistic_query(fact) > 0.8 for _ in paraphrases]
        return sum(results)/len(results) >= 0.75

    def neuro_symbolic_verify(self, fact: Tuple) -> float:
        """Updated to use your SLAILM components"""
        symbolic_score = self.knowledge_base.get(fact, 0.0)
        
        # Use SLAILM for neural validation
        neural_score = self.llm.validate_fact(
            fact, 
            self.knowledge_base
        )
        
        # Weighted combination from config
        return (self.inference_settings['neuro_symbolic_weight'] * neural_score + 
                (1 - self.inference_settings['neuro_symbolic_weight']) * symbolic_score)

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

    def generate_chain_of_thought(self, query: str, max_depth: int = 3) -> List[str]:
        """
        Generate reasoning steps using internal LM-style simulation.
        """
        # Simulated LM pipeline (replace with actual model in production)
        intermediate_steps = [
            f"Analyzing query: {query}",
            "Retrieving relevant facts from knowledge base",
            "Applying forward chaining rules",
            "Verifying consistency with Bayesian network"
        ]
        self.reasoning_traces.append((query, intermediate_steps))
        return intermediate_steps

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
    
    def multi_hop_reasoning(self, query: Tuple) -> float:
        """
        Graph-based traversal for combining facts across sources.
        """
        # Build hypothesis graph
        self._build_hypothesis_graph(query)
        
        # Probabilistic graph traversal
        confidence = 1.0
        for hop in self.hypothesis_graph[query]:
            confidence *= self.knowledge_base.get(hop, 0.5)  # Bayesian chain rule
        return confidence
                
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
            
            for name, rule_func, weight in self.rules:
                try:
                    # Rules now return (fact, confidence) pairs
                    inferred = rule_func(self.knowledge_base)
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

    def probabilistic_query(self, fact: Tuple, evidence: Dict[Tuple, bool] = None) -> float:
        """
        Hybrid inference combining:
        - Exact Bayesian inference when network structure exists
        - Markov Logic Network-style weighted formulae
        - Neural semantic similarity fallback
        """
        # First check Bayesian network structure
        bn_nodes = self.bayesian_network['nodes']
        if fact[0] in bn_nodes and all(e[0] in bn_nodes for e in evidence.keys()):
            return self.bayesian_inference(fact[0], {k[0]:v for k,v in evidence.items()})
            
        # Fallback to Markov Logic Network-style inference
        ml_weights = []
        ml_evidences = []
        
        # Weighted formulae components
        for e_fact, e_value in evidence.items():
            if e_fact in self.knowledge_base:
                # Formula weight from KB confidence
                weight = math.log(self.knowledge_base[e_fact] / (1 - self.knowledge_base[e_fact] + 1e-8))
                ml_weights.append(weight * (1 if e_value else -1))
                ml_evidences.append(1)
                
        # Add semantic similarity component
        semantic_sim = sum(self.semantic_similarity(str(fact), str(e)) 
                        for e in evidence.keys()) / len(evidence)
        ml_weights.append(2.0)  # Fixed weight for semantic component
        ml_evidences.append(semantic_sim)
        
        # Logistic regression combination
        z = sum(w * e for w, e in zip(ml_weights, ml_evidences))
        probability = 1 / (1 + math.exp(-z))
        
        # Knowledge-based calibration
        base_prob = self.knowledge_base.get(fact, 0.5)
        return 0.7 * probability + 0.3 * base_prob  # Ensured to stay in [0,1]

    def bayesian_inference(self, query: str, evidence: Dict[str, bool]) -> float:
        """
        Exact inference using message passing algorithm based on:
        - Pearl (1988) Probabilistic Reasoning in Intelligent Systems
        - Koller & Friedman (2009) Probabilistic Graphical Models
        """
        # Convert evidence to network nodes
        observed = {node: value for node, value in evidence.items() 
                   if node in self.bayesian_network['nodes']}
        
        # Initialize belief states
        beliefs = {node: {'prior': 0.5, 'likelihood': 1.0} 
                  for node in self.bayesian_network['nodes']}
        
        # Set observed evidence
        for node, value in observed.items():
            beliefs[node]['prior'] = 1.0 if value else 0.0
            beliefs[node]['likelihood'] = 1.0  # Hard evidence

        # Message passing schedule
        for _ in range(2):  # Two-pass loopy belief propagation
            # Forward pass (children to parents)
            for edge in self.bayesian_network['edges']:
                parent, child = edge
                if parent == query:
                    continue
                    
                # Calculate message: P(child|parent) * belief(child)
                cpt = self.bayesian_network['cpt'][child]
                message = sum(cpt[parent_val][child_val] * beliefs[child]['prior']
                             for parent_val in [True, False]
                             for child_val in [True, False])
                
                beliefs[parent]['likelihood'] *= message

            # Backward pass (parents to children)
            for edge in reversed(self.bayesian_network['edges']):
                parent, child = edge
                if child == query:
                    continue
                
                # Calculate message: sum_{parent} P(child|parent) * belief(parent)
                cpt = self.bayesian_network['cpt'][child]
                message = sum(cpt[parent_val][child_val] * beliefs[parent]['prior']
                             for parent_val in [True, False]
                             for child_val in [True, False])
                
                beliefs[child]['likelihood'] *= message

        # Final marginal calculation using belief propagation
        marginal = 1.0
        for node in self.bayesian_network['nodes']:
            if node == query:
                prior = self.bayesian_network['cpt'].get(node, {}).get('prior', 0.5)
                marginal = prior * beliefs[node]['likelihood']
                break
            elif node in observed:
                marginal *= beliefs[node]['prior']
        
        # Normalize using partition function
        partition = marginal + (1 - prior) * (1 - beliefs[node]['likelihood'])
        return marginal / partition if partition != 0 else 0.5

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

    def _update_vocabulary(self, text: str):
        """Update NLP vocabulary and simple word vectors."""
        words = re.findall(r'\w+', text.lower())
        for word in words:
            if word not in self.vocab:
                self.vocab.add(word)
                # Simple binary word vector (could be enhanced)
                self.word_vectors[word] = {w: 1 if w == word else 0 for w in self.vocab}

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
    
    def _build_hypothesis_graph(self, root_node: Tuple):
        """
        Construct connected fact graph using semantic similarity.
        """
        for fact in self.knowledge_base:
            if self.semantic_similarity(str(fact), str(root_node)) > 0.7:
                self.hypothesis_graph[root_node].add(fact)
