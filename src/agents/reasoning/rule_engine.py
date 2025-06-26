
import yaml, json
import itertools
import re

from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable
from collections import defaultdict

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from src.agents.reasoning.reasoning_memory import ReasoningMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Rule Engine")
printer = PrettyPrinter

class RuleEngine:
    def __init__(self):
        self.config = load_global_config()
        self.rules_config = get_config_section('rules')
        self.enable_learning = self.rules_config.get('enable_learning')
        self.min_support = self.rules_config.get('min_support')
        self.min_confidence = self.rules_config.get('min_confidence')
        self.auto_weight_adjustment = self.rules_config.get('auto_weight_adjustment')
        self._max_circular_depth = self.rules_config.get('max_circular_depth')
        self.max_utterance_length = self.rules_config.get('max_utterance_length')
        self.discourse_markers_path = self.rules_config.get('discourse_markers_path')
        self.politeness_strategies_path = self.rules_config.get('politeness_strategies_path')

        self.storage_config = get_config_section('storage')
        self.knowledge_db_path = self.storage_config.get('knowledge_db')
        self.lexicon_path = self.storage_config.get('lexicon_path')
        self.dependency_rules_path = self.storage_config.get('dependency_rules_path')

        # Initialize core components
        self.knowledge_base = self._load_knowledge_base(Path(self.knowledge_db_path))
        self.sentiment_lexicon = self._load_sentiment_lexicon(Path(self.lexicon_path))
        self.pos_patterns  = self._load_pos_patterns(Path(self.dependency_rules_path))
        self.dependency_rules = self._create_dependency_rules()
        self.pragmatic_heuristics = self._load_pragmatic_heuristics()

        self.reasoning_memory = ReasoningMemory()
        # Initialize rule storage
        self.learned_rules = defaultdict(float)
        self.rule_weights = {}

        self.rule_antecedents = {}  # rule_name → list of antecedents
        self.rule_consequents = {}  # rule_name → list of consequents

        logger.info("\n\nRule Engine initialized with:")
        logger.info(f" - {len(self.knowledge_base)} knowledge base entries")
        logger.info(f" - {len(self.sentiment_lexicon['positive'])} sentiment terms")
        logger.info(f" - {len(self.dependency_rules)} dependency grammar rules")

    @property
    def max_circular_depth(self) -> int:
        return self._max_circular_depth

    def _load_knowledge_base(self, kb_path: Path) -> Dict[Tuple, Any]:
        """
        Load knowledge base with fact confidences.
        Handles multiple formats:
        - "knowledge" key with dictionary of "s||p||o": confidence
        - "knowledge" key with list of [s, p, o, confidence]
        - "knowledge" key with list of dictionaries
        - Fallback to root-level dictionary
        """
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")
        logger.info(f"Loading knowledge base from {kb_path}")
        parsed_kb: Dict[Tuple, float] = {}
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                full_kb_data = json.load(f)

            # Primary expectation: facts are under a "knowledge" key
            facts_data_source = full_kb_data.get('knowledge')

            if isinstance(facts_data_source, dict):
                # Format: {"s||p||o": confidence}
                logger.info(f"Found 'knowledge' dictionary in {kb_path}. Parsing facts...")
                for k, v in facts_data_source.items():
                    if not isinstance(k, str) or "||" not in k:
                        logger.warning(f"Skipping invalid fact key '{k}' in 'knowledge' dict. Expected 's||p||o' format.")
                        continue
                    try:
                        parts = k.split('||')
                        if len(parts) == 3:
                            obj_str = parts[2]
                            if obj_str.lower() == 'true':
                                obj_val = True
                            elif obj_str.lower() == 'false':
                                obj_val = False
                            else:
                                obj_val = obj_str
                            fact_tuple = (parts[0], parts[1], obj_val)
                            parsed_kb[fact_tuple] = float(v)
                        else:
                            logger.warning(f"Fact key '{k}' in 'knowledge' dict does not have 3 parts. Skipping.")
                    except Exception as e:
                        logger.error(f"Error parsing fact '{k}': {e}")

            elif isinstance(facts_data_source, list):
                logger.info(f"Found 'knowledge' list in {kb_path}. Parsing items...")
                for item in facts_data_source:
                    if isinstance(item, list) and len(item) == 4:
                        # Format: [s, p, o, confidence]
                        try:
                            s, p, o, conf = item
                            if isinstance(o, str):
                                if o.lower() == 'true':
                                    o_val = True
                                elif o.lower() == 'false':
                                    o_val = False
                                else:
                                    o_val = o
                            else:
                                o_val = o
                            fact_tuple = (s, p, o_val)
                            parsed_kb[fact_tuple] = float(conf)
                        except Exception as e:
                            logger.error(f"Error parsing fact from list: {item}: {e}")
                    elif isinstance(item, dict):
                        # Format: {"subject": s, "predicate": p, "object": o, "confidence": c}
                        try:
                            s = item.get("subject")
                            p = item.get("predicate")
                            o = item.get("object")
                            conf = item.get("confidence", 0.5)
                            if s is None or p is None or o is None:
                                logger.warning(f"Skipping incomplete fact: {item}")
                                continue
                            if isinstance(o, str):
                                if o.lower() == 'true':
                                    o_val = True
                                elif o.lower() == 'false':
                                    o_val = False
                                else:
                                    o_val = o
                            else:
                                o_val = o
                            fact_tuple = (s, p, o_val)
                            parsed_kb[fact_tuple] = float(conf)
                        except Exception as e:
                            logger.error(f"Error parsing fact from dict: {item}: {e}")
                    else:
                        logger.warning(f"Unsupported item type in knowledge list: {type(item)}")

            elif facts_data_source is None:
                logger.warning(f"'knowledge' key not found in {kb_path}. Falling back to root.")
                # Fallback: Check if root is a dictionary of facts
                if isinstance(full_kb_data, dict):
                    for k, v in full_kb_data.items():
                        if not isinstance(k, str) or "||" not in k:
                            continue
                        try:
                            parts = k.split('||')
                            if len(parts) == 3:
                                obj_str = parts[2]
                                if obj_str.lower() == 'true':
                                    obj_val = True
                                elif obj_str.lower() == 'false':
                                    obj_val = False
                                else:
                                    obj_val = obj_str
                                fact_tuple = (parts[0], parts[1], obj_val)
                                parsed_kb[fact_tuple] = float(v)
                        except Exception as e:
                            logger.error(f"Error parsing root-level fact '{k}': {e}")

            if not parsed_kb:
                logger.info(f"No facts were successfully parsed from {kb_path} into the knowledge base.")
            else:
                logger.info(f"Successfully loaded {len(parsed_kb)} facts into the knowledge base.")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {kb_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading knowledge base from {kb_path}: {e}")
            return {}

        return parsed_kb

    def _load_sentiment_lexicon(self, lexicon_path: Path) -> Dict[Tuple, Any]:
        """Load sentiment analysis resources"""
        if not lexicon_path.exists():
            raise FileNotFoundError(f"Sentiment lexicon file not found: {lexicon_path}")
        
        logger.info(f"Loading sentiment lexicon from {lexicon_path}")
        with open(lexicon_path, 'r') as f:
            lexicon = json.load(f)

        # Log summary instead of entire lexicon
        num_positive = len(lexicon.get('positive', {}))
        num_negative = len(lexicon.get('negative', {}))
        num_intensifiers = len(lexicon.get('intensifiers', {}))
        num_negators = len(lexicon.get('negators', []))
        
        logger.info(f"Loaded lexicon with {num_positive} positive, {num_negative} negative, "
                    f"{num_intensifiers} intensifiers, and {num_negators} negators")
                    
        return lexicon

    def _load_pos_patterns(self, patterns_path: Path) -> Dict[str, list]:
        """Load POS patterns from JSON file"""
        if not patterns_path.exists():
            raise FileNotFoundError(f"POS patterns file not found: {patterns_path}")
        
        logger.info(f"Loading POS patterns from {patterns_path}")
        with open(patterns_path, 'r') as f:
            patterns_data = json.load(f)
        
        # Convert to pattern_name -> pattern mapping
        return {pattern["name"]: pattern["pattern"] for pattern in patterns_data}

    def _create_dependency_rules(self) -> Dict[str, list]:
        """Create dependency rules from loaded POS patterns"""
        # Filter for patterns that represent dependencies
        dependency_patterns = {}
        for name, pattern in self.pos_patterns.items():
            # Filter for patterns that represent grammatical relationships
            if "->" in name or any(term in name.lower() for term in ["dependency", "relation", "modifier"]):
                dependency_patterns[name] = pattern
        return dependency_patterns

    def _load_pragmatic_heuristics(self) -> Dict[str, Any]:
        """Initialize pragmatic analysis components"""
        discourse_markers = self._load_discourse_markers()
        return {
            "gricean_maxims": self._create_gricean_maxims(discourse_markers),
            "sentiment_analysis": self._create_sentiment_rules(),
            "politeness_strategies": self._load_politeness_strategies(),
            "discourse_markers": discourse_markers
        }

    def _create_gricean_maxims(self, discourse_markers: List[str]) -> Dict[str, callable]:
        """Grice's conversational maxims as executable rules"""
        return {
            "quantity": lambda utterance: len(utterance.split()) < 50,
            "quality": lambda utterance: "probably" not in utterance.lower(),
            "relation": lambda utterance: any(
                marker in utterance.lower() 
                for marker in discourse_markers
            ),
            "manner": lambda utterance: not any(
                filler in utterance 
                for filler in ["uh", "um", "like"]
            )
        }

    def _create_sentiment_rules(self) -> Dict[str, List[str]]:
        """Sentiment analysis rules with intensifier handling"""
        return {
            "positive": self.sentiment_lexicon["positive"],
            "negative": self.sentiment_lexicon["negative"],
            "intensifiers": self.sentiment_lexicon["intensifiers"],
            "negators": set(self.sentiment_lexicon["negators"])
        }

    def _load_discourse_markers(self) -> List[str]:
        """Load discourse markers from configured JSON file"""
        markers_path = Path(self.discourse_markers_path)
        if not markers_path.exists():
            raise FileNotFoundError(f"Discourse markers file not found: {markers_path}")

        logger.info(f"Loading discourse markers from {markers_path}")
        with open(markers_path, 'r') as f:
            data = json.load(f)

        return data.get('discourse_markers', [])

    def _load_politeness_strategies(self) -> Dict[str, List[str]]:
        """Load politeness strategies from configured YAML file"""
        strategies_path = Path(self.politeness_strategies_path)
        if not strategies_path.exists():
            raise FileNotFoundError(f"Politeness strategies file not found: {strategies_path}")

        logger.info(f"Loading politeness strategies from {strategies_path}")
        with open(strategies_path, 'r') as f:
            data = yaml.safe_load(f)

        return data.get('strategies', {})
    
    def get_rules_by_category(self, category: str) -> Dict[str, Callable]:
        """
        Get rules filtered by category. 
        Currently returns all rules since categories aren't implemented yet.
        """
        logger.warning(f"Rule categories not implemented yet. Returning all rules for category '{category}'")
        return self.learned_rules.copy()

    def _discover_new_rules(self):
        """
        Association rule mining with Apriori algorithm adaptation.
        Implements:
        - Agrawal & Srikant (1994) Fast Algorithms for Mining Association Rules
        - Confidence-weighted support from Lê et al. (2014) Fuzzy Association Rules
        """
        # Convert knowledge base to transaction-style format
        transactions = []
        for fact, conf in self.knowledge_base.items():
            if isinstance(conf, dict):  # patch for incorrect value format
                conf = conf.get("confidence", 0.0)
            if conf > 0.5:
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
                        if v/len(transactions) >= self.min_support}

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
                if confidence >= self.min_confidence:
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
            conf = rule['confidence']
            rule_name = f"LearnedRule_{hash(frozenset(ant+cons))}"
            
            # Define the rule function FIRST
            def rule_func(kb, antecedents=ant, consequents=cons, conf=conf):
                matches = [fact for fact in kb if all(e in fact for e in antecedents)]
                return {consequents: conf * len(matches)/(len(kb)+1e-8)}  # Prevent division by zero
            
            # THEN add the rule
            self.add_rule(rule_func, rule_name, weight=conf, antecedents=ant, consequents=cons)
            
            # Remove the redundant rule addition below this
            self.reasoning_memory.add(
                experience={
                    "type": "rule_discovery",
                    "rule_name": rule_name,
                    "antecedents": ant,
                    "consequents": cons,
                    "confidence": conf
                },
            )
            # Check for circular dependencies before adding
            if not self._would_create_circular_dependency(ant, cons):
                self.add_rule(rule_func, rule_name, weight=conf, antecedents=ant, consequents=cons)
            else:
                logger.warning(f"Skipped circular rule: {ant} => {cons}")

    def add_rule(self, rule_func: Callable, name: str, weight: float, antecedents: List, consequents: List):
        self.learned_rules[name] = rule_func
        self.rule_weights[name] = weight
        self.rule_antecedents[name] = antecedents
        self.rule_consequents[name] = consequents

    def _would_create_circular_dependency(self, antecedents, consequents):
        """Check if new rule would create circular dependency"""
        # Check if any consequent appears in existing antecedents
        for cons in consequents:
            if any(cons in self.rule_antecedents.get(r, []) for r in self.learned_rules):
                return True
        return False
    
    def adjust_circular_rule_weights(self, circular_chains: List[List[str]]):
        """Reduce weights of rules in circular chains"""
        for chain in circular_chains:
            for rule_name in chain:
                current_weight = self.rule_weights.get(rule_name, 1.0)
                new_weight = max(0.1, current_weight * 0.5)  # Decay weight
                self.rule_weights[rule_name] = new_weight
                logger.info(f"Reduced weight for circular rule {rule_name}: {current_weight} -> {new_weight}")

    def detect_circular_rules(self, max_depth: int = 3) -> List[List[str]]:
        """
        Detect circular dependencies in rule applications using depth-limited search.
        
        Args:
            max_depth: Maximum allowed chain length before considering it circular
            
        Returns:
            List of circular rule chains (e.g., [["RuleA", "RuleB"], ["RuleC"]])
        """
        circular_chains = []
        rule_deps = {rule: self._get_rule_dependencies(rule) for rule in self.learned_rules}
        
        for rule in self.learned_rules:
            visited = []
            stack = [(rule, [rule])]
            
            while stack:
                current_rule, path = stack.pop()
                if current_rule in visited:
                    if current_rule == path[0] and len(path) > 1:
                        circular_chains.append(path)
                    continue
                
                visited.append(current_rule)
                if len(path) >= max_depth:
                    continue
                    
                for dep_rule in rule_deps.get(current_rule, []):
                    stack.append((dep_rule, path + [dep_rule]))
        
        return circular_chains

    def _get_rule_dependencies(self, rule_name: str) -> List[str]:
        """Extract rules that trigger/are triggered by the given rule"""
        consequents = self.rule_consequents.get(rule_name, [])
        dependent_rules = [
            r for r in self.learned_rules 
            if any(c in self.rule_antecedents[r] for c in consequents)
        ]
        return dependent_rules

    def detect_fact_conflicts(self, contradiction_threshold: float = 0.25) -> List[Tuple]:
        """
        Identify contradictory facts using semantic opposition checks.
        
        Returns:
            List of conflicting fact pairs and their confidence scores
        """
        conflicts = []
        fact_index = defaultdict(list)  # {(s, p): [(o, confidence)]}
        
        for (s, p, o), conf in self.knowledge_base.items():
            fact_index[(s, p)].append((o, conf))
        
        for (s, p), objects in fact_index.items():
            for i, (o1, c1) in enumerate(objects):
                for o2, c2 in objects[i+1:]:
                    if self._are_contradictory(o1, o2) and (c1 + c2)/2 > contradiction_threshold:
                        conflicts.append(((s, p, o1, c1), (s, p, o2, c2)))
        
        return conflicts

    def _are_contradictory(self, obj1: str, obj2: str) -> bool:
        """Check if two objects are antonyms using Wordlist or sentiment lexicon"""
        if obj1 in self.sentiment_lexicon.get('antonyms', {}).get(obj2, []):
            return True
        if hasattr(self, 'wordlist') and self.wordlist:
            return self.wordlist.synonym_path(obj1, obj2) is None and \
                self.wordlist.semantic_similarity(obj1, obj2) < 0.2
        return False

    def redundant_fact_check(self, confidence_margin: float = 0.05) -> List[Tuple]:
        """
        Find facts that can be inferred from existing rules/knowledge.
        
        Args:
            confidence_margin: Minimum confidence difference to consider redundant
            
        Returns:
            List of redundant facts with their inferred confidence
        """
        redundancies = []
        inferred_facts = self._apply_all_rules()
        
        for (s, p, o), kb_conf in self.knowledge_base.items():
            inf_conf = inferred_facts.get((s, p, o), 0.0)
            if inf_conf >= (kb_conf - confidence_margin):
                redundancies.append(((s, p, o), kb_conf, inf_conf))
        
        return redundancies

    def _apply_all_rules(self) -> Dict[Tuple, float]:
        """Apply all learned rules to infer new facts"""
        inferred = defaultdict(float)
        for rule_func in self.learned_rules.values():
            new_facts = rule_func(self.knowledge_base)
            for fact, conf in new_facts.items():
                inferred[fact] = max(inferred[fact], conf)
        return inferred

    def analyze_utterance(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of natural language utterances containing:
        - Gricean maxim violations
        - Politeness strategy detection
        - Discourse relation identification
        - Sentiment analysis
        
        Args:
            text: Input utterance to analyze
            
        Returns:
            Dictionary with detailed analysis results
        """
        result = {
            'gricean_violations': self._check_gricean_maxims(text),
            'politeness_strategies': self._detect_politeness(text),
            'discourse_relations': self._identify_discourse_relations(text),
            'sentiment': self._analyze_sentiment(text),
            'dependency_analysis': self._analyze_dependencies(text)
        }
        self.reasoning_memory.add(
            experience={
                "type": "utterance_analysis",
                "text": text,
                "result": result
            },
            tag="pragmatic_analysis"
        )
        return result

    def _check_gricean_maxims(self, text: str) -> Dict[str, bool]:
        """Evaluate utterance against Grice's conversational maxims"""
        maxims = self.pragmatic_heuristics['gricean_maxims']
        return {
            'quantity': not maxims['quantity'](text),
            'quality': not maxims['quality'](text),
            'relation': not maxims['relation'](text),
            'manner': not maxims['manner'](text)
        }

    def _detect_politeness(self, text: str) -> Dict[str, List[str]]:
        """Identify politeness strategies in utterance"""
        strategies = self.pragmatic_heuristics['politeness_strategies']
        detected = defaultdict(list)
        text_lower = text.lower()
        
        for strategy, phrases in strategies.items():
            for phrase in phrases:
                if phrase.lower() in text_lower:
                    detected[strategy].append(phrase)
                    
        return dict(detected)

    def _identify_discourse_relations(self, text: str) -> List[str]:
        """Find discourse markers in utterance"""
        markers = self.pragmatic_heuristics['discourse_markers']
        text_lower = text.lower()
        return [marker for marker in markers if marker.lower() in text_lower]

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate sentiment score with intensifier/negator handling"""
        sentiment = 0.0
        words = re.findall(r'\b\w+\b', text.lower())
        lexicon = self.pragmatic_heuristics['sentiment_analysis']
        
        # Track sentiment modifiers
        current_multiplier = 1.0
        negation = False
        
        for i, word in enumerate(words):
            # Handle intensifiers
            if word in lexicon['intensifiers']:
                current_multiplier = lexicon['intensifiers'][word]
                continue
                
            # Handle negators
            if word in lexicon['negators']:
                negation = not negation
                continue
                
            # Base sentiment scoring
            if word in lexicon['positive']:
                base = lexicon['positive'][word]
            elif word in lexicon['negative']:
                base = lexicon['negative'][word]
            else:
                continue
                
            # Apply modifiers
            score = base * current_multiplier
            if negation:
                score *= -1
                
            sentiment += score
            # Reset modifiers after use
            current_multiplier = 1.0
            negation = False
            
        # Normalize sentiment score
        return {'score': sentiment / len(words) if len(words) > 0 else 0.0}

    def _analyze_dependencies(self, text: str) -> Dict[str, list]:
        """Analyze grammatical dependencies using POS patterns and custom rules"""
        # Initialize NLP Engine for tokenization and POS tagging
        try:
            from src.agents.language.nlp_engine import NLPEngine
        except ImportError:
            logger.error("NLPEngine could not be imported for dependency analysis")
            return {"error": "NLPEngine unavailable"}
    
        nlp_engine = NLPEngine()
        tokens = nlp_engine.process_text(text)
        
        # Convert tokens to simplified format for pattern matching
        token_data = [
            {"text": token.text, "pos": token.pos, "index": token.index}
            for token in tokens
        ]
        
        # Get sentence structure by grouping tokens
        sentence_boundaries = self._detect_sentence_boundaries(tokens)
        analysis_results = defaultdict(list)
        
        # Apply dependency patterns
        for rule_name, pattern in self.dependency_rules.items():
            matches = self._match_pos_pattern(token_data, pattern)
            analysis_results[rule_name] = matches
            
        # Add basic clause detection
        analysis_results["clauses"] = self._detect_clauses(token_data, sentence_boundaries)
        
        # Add core grammatical relationships
        analysis_results["relations"] = self._extract_grammatical_relations(token_data)
        
        return dict(analysis_results)
    
    def _detect_sentence_boundaries(self, tokens: list) -> List[Tuple[int, int]]:
        """Identify sentence boundaries based on punctuation"""
        boundaries = []
        start_idx = 0
        for i, token in enumerate(tokens):
            if token.text in {'.', '!', '?'}:
                boundaries.append((start_idx, i))
                start_idx = i + 1
        if start_idx < len(tokens):
            boundaries.append((start_idx, len(tokens)-1))
        return boundaries
    
    def _match_pos_pattern(self, tokens: List[dict], pattern: list) -> List[dict]:
        """Match POS patterns in token sequence with wildcard support"""
        matches = []
        pattern_length = len(pattern)
        
        for i in range(len(tokens) - pattern_length + 1):
            window = tokens[i:i+pattern_length]
            window_pos = [t["pos"] for t in window]
            
            # Check for exact match
            if window_pos == pattern:
                matches.append({
                    "text": " ".join(t["text"] for t in window),
                    "start_index": i,
                    "end_index": i + pattern_length - 1
                })
                
            # Wildcard matching (e.g., ["DET", "*", "NOUN"])
            elif self._wildcard_match(window_pos, pattern):
                matches.append({
                    "text": " ".join(t["text"] for t in window),
                    "start_index": i,
                    "end_index": i + pattern_length - 1,
                    "pattern": pattern,
                    "actual": window_pos
                })
                
        return matches
    
    def _wildcard_match(self, actual: List[str], pattern: List[str]) -> bool:
        """Flexible pattern matching with wildcard support"""
        if len(actual) != len(pattern):
            return False
            
        for a, p in zip(actual, pattern):
            if p == "*":
                continue
            if "|" in p:  # Handle alternates (e.g., "NOUN|PROPN")
                if a not in p.split("|"):
                    return False
            elif a != p:
                return False
                
        return True
    
    def _detect_clauses(self, tokens: List[dict], boundaries: List[Tuple[int, int]]) -> List[dict]:
        """Detect basic clause structures in sentences"""
        clauses = []
        
        for start, end in boundaries:
            sent_tokens = tokens[start:end+1]
            verbs = [i for i, t in enumerate(sent_tokens) if t["pos"] in {"VERB", "AUX"}]
            
            for verb_idx in verbs:
                # Find subject (left of verb)
                subject = None
                for i in range(verb_idx-1, start-1, -1):
                    if sent_tokens[i]["pos"] in {"NOUN", "PROPN", "PRON"}:
                        subject = sent_tokens[i]
                        break
                        
                # Find object (right of verb)
                obj = None
                for i in range(verb_idx+1, end+1):
                    if sent_tokens[i]["pos"] in {"NOUN", "PROPN", "PRON"}:
                        obj = sent_tokens[i]
                        break
                        
                if subject or obj:
                    clauses.append({
                        "subject": subject["text"] if subject else None,
                        "verb": sent_tokens[verb_idx]["text"],
                        "object": obj["text"] if obj else None,
                        "sentence_range": (start, end),
                        "verb_index": verb_idx
                    })
                    
        return clauses
    
    def _extract_grammatical_relations(self, tokens: List[dict]) -> List[dict]:
        """Extract core grammatical relationships between tokens"""
        relations = []
        
        # Subject-Verb-Object relationships
        for i, token in enumerate(tokens):
            if token["pos"] in {"VERB", "AUX"}:
                # Look for subjects (left of verb)
                subj = self._find_previous(tokens, i, {"NOUN", "PROPN", "PRON"})
                # Look for objects (right of verb)
                obj = self._find_next(tokens, i, {"NOUN", "PROPN", "PRON"})
                
                if subj or obj:
                    relations.append({
                        "relation": "SVO",
                        "subject": subj["text"] if subj else None,
                        "verb": token["text"],
                        "object": obj["text"] if obj else None
                    })
        
        # Adjective-Noun relationships
        for i, token in enumerate(tokens):
            if token["pos"] == "ADJ":
                # Look for modified noun (right of adjective)
                noun = self._find_next(tokens, i, {"NOUN", "PROPN"})
                if noun:
                    relations.append({
                        "relation": "AMOD",
                        "adjective": token["text"],
                        "noun": noun["text"]
                    })
        
        # Prepositional phrases
        for i, token in enumerate(tokens):
            if token["pos"] == "ADP":  # Preposition
                obj = self._find_next(tokens, i, {"NOUN", "PROPN", "PRON"})
                if obj:
                    relations.append({
                        "relation": "PREP",
                        "preposition": token["text"],
                        "object": obj["text"]
                    })
        
        return relations
    
    def _find_previous(self, tokens, index, target_pos):
        """Find nearest matching token to the left"""
        for i in range(index-1, -1, -1):
            if tokens[i]["pos"] in target_pos:
                return tokens[i]
        return None
    
    def _find_next(self, tokens, index, target_pos):
        """Find nearest matching token to the right"""
        for i in range(index+1, len(tokens)):
            if tokens[i]["pos"] in target_pos:
                return tokens[i]
        return None

if __name__ == "__main__":
    print("\n=== Running Rule Engine ===\n")
    engine = RuleEngine()
    print(engine)

    # Detect issues
    circular = engine.detect_circular_rules()
    conflicts = engine.detect_fact_conflicts()
    redundant = engine.redundant_fact_check()

    # Create a dummy knowledge_db.json for testing
    test_kb_content_scenario1 = { # Scenario 1: "knowledge" key with a dict of facts
        "knowledge": {
            "cat||is_animal||True": 0.9,
            "dog||is_animal||True": 0.95,
            "cat||has_fur||True": 0.8,
            "sky||is_blue||False": 0.1 # e.g. at night
        },
        "rules": [],
        "rule_weights": {}
    }
    test_kb_content_scenario2 = { # Scenario 2: "knowledge" key with an empty list
        "knowledge": [],
        "rules": [],
        "rule_weights": {}
    }
    test_kb_content_scenario3 = { # Scenario 3: Flat fact dictionary (no "knowledge" key)
        "bird||can_fly||True": 0.8,
        "penguin||is_bird||True": 1.0,
        "penguin||can_fly||False": 0.99
    }
    with open("knowledge_db.json", "w") as f:
        json.dump(test_kb_content_scenario1, f) # Change this to test other scenarios

    print("--- Testing RuleEngine with corrected _load_knowledge_base ---")

    utterances = [
    "I think you might perhaps want to consider this option, however it's not absolutely perfect."
    "Um, well, like, I don't really know what to say.",
    "The absolute hate in her heart is unlike anything I've seen before. Yet, she is loved by so many",
    "I hate working at this store!!",
    "Did you buy the 3 books I've suggested?",
    "I love the new piano you bought for me!!",
    "You must read the book before tomorrow."
    ]
    #analysis = engine.analyze_utterance(utterances)
    
    for i, utterance in enumerate(utterances):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Utterance: {utterance}")
        result = engine.analyze_utterance(utterance)

        print("Gricean Violations:")
        for maxim, violated in result['gricean_violations'].items():
            print(f"  {maxim.capitalize()}: {'Yes' if violated else 'No'}")

        print("Politeness Strategies:")
        for strat, phrases in result['politeness_strategies'].items():
            print(f"  {strat}: {phrases}")

        print("Discourse Relations:", result['discourse_relations'])
        print("Sentiment Score:", result['sentiment']['score'])
        print("Dependency Analysis:")
        for rel, links in result['dependency_analysis'].items():
            print(f"  {rel}: {links}")

    print("\n=== Successfully Ran Rule Engine ===\n")
