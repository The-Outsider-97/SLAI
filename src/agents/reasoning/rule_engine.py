
import yaml, json
import itertools
import re

from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable
from collections import defaultdict

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from src.agents.reasoning.reasoning_memory import ReasoningMemory
from logs.logger import get_logger

logger = get_logger("Rule Engine")

class RuleEngine:
    def __init__(self, knowledge_base: Dict[Tuple, float] = None):
        self.config = load_global_config()
        self.validation_config = get_config_section('rules')

        # Initialize core components
        self.knowledge_base = knowledge_base or self._load_knowledge_base()
        self.sentiment_lexicon = self._load_sentiment_lexicon()
        self.pragmatic_heuristics = self._load_pragmatic_heuristics()
        self.dependency_rules = self._create_dependency_rules()

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

    def _load_knowledge_base(self) -> Dict[Tuple, float]:
        """
        Load knowledge base with fact confidences.
        This version attempts to read facts primarily from a "knowledge" key 
        within the JSON file, expecting a dictionary of "s||p||o": confidence.
        It includes a fallback to read from the root if "knowledge" is not found
        or not in the expected dictionary format.
        """
        # Path to knowledge_db.json is taken from the global config
        kb_path_str = self.config.get('storage', {}).get('knowledge_db')
        if not kb_path_str:
            logger.error("Knowledge base path not found in configuration (storage.knowledge_db).")
            return {}
        
        kb_path = Path(kb_path_str)

        if not kb_path.exists():
            logger.error(f"Knowledge base file not found: {kb_path}")
            # Depending on strictness, you might raise FileNotFoundError here
            # raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")
            return {}
            
        logger.info(f"Loading knowledge base from {kb_path}")
        parsed_kb: Dict[Tuple, float] = {}
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                full_kb_data = json.load(f)
            
            # Primary expectation: facts are under a "knowledge" key as a dictionary.
            # e.g., "knowledge": {"cat||is_animal||True": 0.9, ...}
            facts_data_source = full_kb_data.get('knowledge')

            if isinstance(facts_data_source, dict):
                logger.info(f"Found 'knowledge' dictionary in {kb_path}. Parsing facts...")
                for k, v in facts_data_source.items():
                    if not isinstance(k, str) or "||" not in k:
                        logger.warning(f"Skipping invalid fact key '{k}' in 'knowledge' dict. Expected 's||p||o' format.")
                        continue
                    try:
                        parts = k.split('||')
                        if len(parts) == 3: # s, p, o
                            obj_str = parts[2]
                            # Convert common boolean strings to actual booleans
                            if obj_str.lower() == 'true':
                                obj_val = True
                            elif obj_str.lower() == 'false':
                                obj_val = False
                            else:
                                obj_val = obj_str # Keep as string if not 'true'/'false'
                            
                            fact_tuple = (parts[0], parts[1], obj_val)
                            parsed_kb[fact_tuple] = float(v)
                        else:
                            logger.warning(f"Fact key '{k}' in 'knowledge' dict does not have 3 parts. Skipping.")
                    except ValueError:
                        logger.error(f"Error converting confidence for fact '{k}': '{v}' to float. Skipping.")
                    except Exception as e:
                        logger.error(f"Error parsing fact '{k}': {v} from 'knowledge' dict: {e}")
            
            elif isinstance(facts_data_source, list):
                # This handles the case where "knowledge": [] (empty list)
                if not facts_data_source:
                    logger.info(f"'knowledge' field in {kb_path} is an empty list. No facts loaded from this field.")
                else:
                    # If it's a list of {"fact_string": "s||p||o", "confidence": 0.9}
                    # or list of {"subject":s, "predicate":p, "object":o, "confidence":0.9}
                    # This part would need to be more specific based on the actual list structure.
                    # For now, we log a warning if it's a non-empty list and not a dict.
                    logger.warning(f"'knowledge' field in {kb_path} is a list, but a dictionary was primarily expected. "
                                   f"Attempting to parse list items if structured specifically (not fully generic).")
                    # Example: parsing list of {"s||p||o": conf} dicts (less common)
                    for item_dict in facts_data_source:
                        if isinstance(item_dict, dict):
                            for k, v in item_dict.items(): # Assuming one fact per dict in list
                                if isinstance(k, str) and "||" in k:
                                    parts = k.split('||')
                                    if len(parts) == 3:
                                        obj_str = parts[2]
                                        if obj_str.lower() == 'true': obj_val = True
                                        elif obj_str.lower() == 'false': obj_val = False
                                        else: obj_val = obj_str
                                        parsed_kb[(parts[0], parts[1], obj_val)] = float(v)
                                    break # Process first valid key-value pair
            
            elif facts_data_source is None:
                logger.warning(f"'knowledge' key not found in {kb_path}.")
                # Fallback: Check if the root of the JSON is the fact dictionary itself
                # This maintains compatibility if knowledge_db.json is flat like {"s||p||o": conf, ...}
                if isinstance(full_kb_data, dict):
                    is_flat_fact_dict = True
                    # Heuristic: check if most keys look like facts
                    fact_like_keys = 0
                    if len(full_kb_data) > 0: # Avoid division by zero
                        for k_root in full_kb_data.keys():
                            if isinstance(k_root, str) and "||" in k_root:
                                fact_like_keys +=1
                        if (fact_like_keys / len(full_kb_data)) < 0.5: # If less than half keys are fact-like, probably not it
                            is_flat_fact_dict = False
                    else: # Empty dict
                        is_flat_fact_dict = False


                    if is_flat_fact_dict:
                        logger.info(f"Treating root of {kb_path} as facts dictionary (fallback).")
                        for k, v in full_kb_data.items():
                            if not isinstance(k, str) or "||" not in k:
                                # This case should ideally not be hit if is_flat_fact_dict was true
                                logger.warning(f"Skipping invalid fact key '{k}' at root. Expected 's||p||o' format.")
                                continue
                            try:
                                parts = k.split('||')
                                if len(parts) == 3:
                                    obj_str = parts[2]
                                    if obj_str.lower() == 'true': obj_val = True
                                    elif obj_str.lower() == 'false': obj_val = False
                                    else: obj_val = obj_str
                                    parsed_kb[(parts[0], parts[1], obj_val)] = float(v)
                                else:
                                    logger.warning(f"Fact key '{k}' at root does not have 3 parts. Skipping.")
                            except ValueError:
                                logger.error(f"Error converting confidence for fact '{k}': '{v}' (root) to float. Skipping.")
                            except Exception as e:
                                logger.error(f"Error parsing fact '{k}': {v} (root): {e}")
                    else:
                        logger.info(f"Root of {kb_path} is not a flat facts dictionary. No facts loaded from root.")
                else:
                    logger.info(f"Root of {kb_path} is not a dictionary. No facts loaded from root.")


            if not parsed_kb:
                logger.info(f"No facts were successfully parsed from {kb_path} into the knowledge base.")
            else:
                logger.info(f"Successfully loaded {len(parsed_kb)} facts into the knowledge base.")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {kb_path}: {e}")
            # raise ValueError(f"Invalid JSON format in {kb_path}") from e # Or return {}
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading knowledge base from {kb_path}: {e}")
            return {} # Return empty on other errors to prevent crash

        return parsed_kb

    def _load_sentiment_lexicon(self) -> Dict[str, Any]:
        """Load sentiment analysis resources"""
        lexicon_path = Path(self.config['storage']['lexicon_path'])
        if not lexicon_path.exists():
            raise FileNotFoundError(f"Sentiment lexicon not found: {lexicon_path}")
            
        logger.info(f"Loading sentiment lexicon from {lexicon_path}")
        with open(lexicon_path, 'r') as f:
            return json.load(f)

    def _load_pragmatic_heuristics(self) -> Dict[str, Any]:
        """Initialize pragmatic analysis components"""
        return {
            "gricean_maxims": self._create_gricean_maxims(),
            "sentiment_analysis": self._create_sentiment_rules(),
            "discourse_markers": self._load_discourse_markers(),
            "politeness_strategies": self._load_politeness_strategies()
        }

    def _create_gricean_maxims(self) -> Dict[str, callable]:
        """Grice's conversational maxims as executable rules"""
        return {
            "quantity": lambda utterance: len(utterance.split()) < 50,
            "quality": lambda utterance: "probably" not in utterance.lower(),
            "relation": lambda utterance: any(
                marker in utterance.lower() 
                for marker in self.pragmatic_heuristics['discourse_markers']
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
        markers_path = Path(self.config['rules']['discourse_markers_path'])
        if not markers_path.exists():
            raise FileNotFoundError(f"Discourse markers file not found: {markers_path}")

        logger.info(f"Loading discourse markers from {markers_path}")
        with open(markers_path, 'r') as f:
            data = json.load(f)

        return data.get('discourse_markers', [])

    def _load_politeness_strategies(self) -> Dict[str, List[str]]:
        """Load politeness strategies from configured YAML file"""
        strategies_path = Path(self.config['rules']['politeness_strategies_path'])
        if not strategies_path.exists():
            raise FileNotFoundError(f"Politeness strategies file not found: {strategies_path}")

        logger.info(f"Loading politeness strategies from {strategies_path}")
        with open(strategies_path, 'r') as f:
            data = yaml.safe_load(f)

        return data.get('strategies', {})

    def _create_dependency_rules(self) -> Dict[str, list]:
        """Head-modifier grammar inspired by Universal Dependencies"""
        return {
            'nsubj': ['NN', 'VB'],
            'dobj': ['VB', 'NN'],
            'amod': ['NN', 'JJ'],
            'advmod': ['VB', 'RB'],
            'prep': ['IN', 'NN']
        }

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
            rule_name = f"LearnedRule_{hash(frozenset(ant+cons))}"
            self.add_rule(rule_func, rule_name, weight=rule['confidence'], antecedents=ant, consequents=cons)
            
            def rule_func(kb, antecedents=ant, consequents=cons, conf=rule['confidence']):
                matches = [fact for fact in kb if all(e in fact for e in antecedents)]
                return {consequents: conf * len(matches)/(len(kb)+1e-8)}  # Prevent division by zero
                
            rule_name = f"LearnedRule_{hash(frozenset(ant+cons))}"
            self.add_rule(rule_func, rule_name, weight=rule['confidence'])
            self.reasoning_memory.add(
                experience={
                    "type": "rule_discovery",
                    "rule_name": rule_name,
                    "antecedents": ant,
                    "consequents": cons,
                    "confidence": rule['confidence']
                },
            )

    def add_rule(self, rule_func: Callable, name: str, weight: float, antecedents: List, consequents: List):
        self.learned_rules[name] = rule_func
        self.rule_weights[name] = weight
        self.rule_antecedents[name] = antecedents
        self.rule_consequents[name] = consequents

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
        """Parse grammatical dependencies using rule-based patterns"""
        from src.agents.language.grammar_processor import GrammarProcessor
        dependencies = defaultdict(list)
        words = text.split()
        upos_map = GrammarProcessor._UPOS_MAP
        
        # Reverse the mapping to go from UPOS → tag-like shortcut
        reverse_upos = {v: k.upper()[:2] for k, v in upos_map.items()}  # 'NOUN' -> 'NO'
        pos_tags = [reverse_upos.get(upos_map.get(word.lower(), 'noun').upper(), 'NN') for word in words]
        
        for relation, patterns in self.dependency_rules.items():
            for i in range(len(words)-1):
                head_tag = pos_tags[i]
                modifier_tag = pos_tags[i+1]
                if [head_tag, modifier_tag] == patterns:
                    dependencies[relation].append(f"{words[i]}->{words[i+1]}")
                    
        return dict(dependencies)

if __name__ == "__main__":
    print("\n=== Running Rule Engine ===\n")
    engine = RuleEngine()
    print(engine)

    # Detect issues
    circular = engine.detect_circular_rules()
    conflicts = engine.detect_fact_conflicts()
    redundant = engine.redundant_fact_check()

    def mock_load_global_config():
        return {
            "storage": {
                "knowledge_db": "knowledge_db.json" # Path to your test DB
            },
            "rules": { # From reasoning_config.yaml
                "min_confidence_to_propagate": 0.1 # Example value
            }
        }
    
    # Override the actual loader with the mock for this test
    import src.agents.reasoning.utils.config_loader as cl
    original_loader = cl.load_global_config
    cl.load_global_config = mock_load_global_config

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
    try:
        engine = RuleEngine() # This will call _load_knowledge_base
        print(f"Loaded Knowledge Base ({len(engine.knowledge_base)} facts):")
        for fact, conf in engine.knowledge_base.items():
            print(f"  {fact} -> {conf}")
        
        if not engine.knowledge_base and "knowledge_db.json": # contains actual facts
             print("\nWARNING: Knowledge base is empty but knowledge_db.json might have facts. Check parsing logic and file structure.")

    except FileNotFoundError as e:
        print(f"ERROR: {e}. Ensure knowledge_db.json exists.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up dummy file
        if Path("knowledge_db.json").exists():
            Path("knowledge_db.json").unlink()
        # Restore original loader
        cl.load_global_config = original_loader

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
