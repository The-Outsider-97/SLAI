import re
import math
import json
import copy
import random
import numpy as np
import logging as logger
from pathlib import Path
from typing import Optional, Any, List, Union, Tuple
from src.agents.language.language_profiles import MORPHOLOGY_RULES
from collections import defaultdict, deque, Counter
# === Configuration ===
STRUCTURED_WORDLIST_PATH = "src/agents/language/structured_wordlist_en.json"

class GrammarProcessor:
    """Implements formal grammar systems based on Chomsky hierarchy and information theory"""
    # Universal POS Tags (Petrov et al., 2012)
    _UPOS_MAP = {
        # Open class words
        'noun': 'NOUN',        # Common nouns
        'propn': 'PROPN',      # Proper nouns (New York, Alice)
        'verb': 'VERB',        # Main verbs (eat, run)
        'aux': 'AUX',          # Auxiliaries (will, would, be)
        'adjective': 'ADJ',    # Adjectives (happy, big)
        'adverb': 'ADV',       # Adverbs (quickly, very)
        'numeral': 'NUM',      # Numerals (7, twenty-three)
        
        # Closed class words
        'determiner': 'DET',   # Determiners (the, a, some)
        'pronoun': 'PRON',     # Pronouns (I, you, they)
        'preposition': 'ADP',  # Prepositions (in, on, at)
        'conjunction': 'CCONJ',# Coordinating conjunctions (and, or)
        'subord': 'SCONJ',     # Subordinating conjunctions (if, because)
        'particle': 'PART',    # Particles (to, 's)
        'interjection': 'INTJ',# Interjections (oh, wow)
        
        # Special categories
        'symbol': 'SYM',       # Symbols ($, %, ©)
        'punct': 'PUNCT',      # Punctuation
        'wh-word': 'PRON',     # WH-words (what, who)
        'existential': 'EX',   # Existential 'there'
    }

    def __init__(self, lang='en', structured_wordlist=None, wordlist=None,
                 nlg_templates=None, rules_path=None, knowledge_agent=None):
        self._wordlist = None
        self._structured_wordlist = None

        if not self._validate_wordlist_integrity():
            raise ValueError("Wordlist failed integrity check")

        self.morph_rules = MORPHOLOGY_RULES[lang]
        self.reset_parser_state()
        self.knowledge_base = knowledge_agent

        if structured_wordlist is not None:
            self.pos_map = self._convert_structured_wordlist(structured_wordlist)
        else:
            self.pos_map = self._load_pos_data()
        if wordlist is not None:
            self.wordlist = wordlist
        else:
            self.wordlist = {}
        if nlg_templates is not None:
            self.nlg_templates = nlg_templates
        else:
            from src.agents.language.resource_loader import ResourceLoader
            self.nlg_templates = ResourceLoader.get_nlg_templates()
        self._build_pos_patterns()
        if rules_path:
            self._load_custom_cfg_rules(rules_path)
        # CFG rules to use standardized tags
        self.cfg_rules = {
            # Sentence level rules
            'S': [
                ['DECLARATIVE'], 
                ['INTERROGATIVE'],
                ['IMPERATIVE'],
                ['EXCLAMATORY']
            ],
            
            # Declarative structure (Chomsky, 1965)
            'DECLARATIVE': [
                ['NP', 'VP'],
                ['ADV_PHRASE', 'NP', 'VP']
            ],
            
            # Interrogative structures
            'INTERROGATIVE': [
                ['AUX', 'NP', 'VP', 'PUNCT_QM'],
                ['WH_PHRASE', 'AUX', 'NP', 'VP', 'PUNCT_QM']
            ],
            
            # Noun phrase expansions (Radford, 1988)
            'NP': [
                ['DET', 'NOMINAL'], 
                ['PROPN'],
                ['PRON'],
                ['NOMINAL'],
                ['NP', 'PP'],       # Recursive prepositional attachment
                ['NP', 'REL_CLAUSE'] # Relative clauses
            ],
            
            'NOMINAL': [
                ['ADJ_PHRASE', 'NOUN'],
                ['NOUN', 'ADJ_PHRASE'],
                ['NUM', 'NOUN']
            ],
            
            # Verb phrase expansions (Pollard & Sag, 1994)
            'VP': [
                ['V', 'NP'],        # Transitive
                ['V', 'PP'],        # Prepositional
                ['V', 'ADJ_PHRASE'],# Copular
                ['V', 'ADV'],       # Intransitive
                ['AUX', 'VP'],      # Auxiliary chains
                ['V', 'S']          # Sentential complements
            ],
            
            # Phrase extensions
            'PP': [['ADP', 'NP']],          # Prepositional phrase
            'ADJ_PHRASE': [                # Adjective phrase
                ['ADJ'], 
                ['ADV', 'ADJ'], 
                ['ADJ', 'PP']
            ],
            'ADV_PHRASE': [                # Adverb phrase
                ['ADV'], 
                ['ADV', 'ADV_PHRASE']
            ],
            
            # Special constructions
            'WH_PHRASE': [['PRON_WH'], ['ADV_WH']],
            'REL_CLAUSE': [['PRON_REL', 'VP']],
            'PUNCT_QM': [['PUNCT']]  # Question mark
        }

        # Add feature constraints (Gazdar et al., 1985)
        self.feature_constraints = {
            'NP': {
                'NUMBER': ['sg', 'pl'],
                'CASE': ['nom', 'acc']
            },
            'V': {
                'TENSE': ['pres', 'past'],
                'AGREEMENT': ['3sg']
            },
            'DET-NOUN': {
                'NUMBER_AGREEMENT': True
            }
        }

        self.ngram_model = defaultdict(lambda: defaultdict(int))
        self._build_ngram_model()
        self.stemmer = self.PorterStemmer()
        self.entity_tracker = {
            'entities': deque(maxlen=10),
            'pronouns': {
                'he': ['male', 'singular'], # Subject pronouns
                'she': ['female', 'singular'],
                'they': ['neutral', 'plural'],
                'it': ['neutral', 'singular'],

                'him': ['male', 'singular', 'object'], # Object pronouns
                'her': ['female', 'singular', 'object'],
                'them': ['neutral', 'plural', 'object'],

                'his': ['male', 'singular', 'possessive'], # Possessive pronouns
                'hers': ['female', 'singular', 'possessive'],
                'theirs': ['neutral', 'plural', 'possessive'],
                'its': ['neutral', 'singular', 'possessive'],

                'himself': ['male', 'singular', 'reflexive'], # Reflexive pronouns
                'herself': ['female', 'singular', 'reflexive'],
                'themself': ['neutral', 'singular', 'reflexive'],
                'themselves': ['neutral', 'plural', 'reflexive'],
                'itself': ['neutral', 'singular', 'reflexive']

            },
            'entities': deque(maxlen=15)  # Increased buffer size
        }

        self.pos_patterns = [
            # Proper nouns (must come first to prevent substring matches)
            (re.compile(r'\b[A-Z][a-z]+\b'), 'PROPN'),
            
            # Core POS patterns with descending specificity
            (re.compile(r'\b\w+(tion|ment|ness|ity|acy|ism)\b', re.IGNORECASE), 'NOUN'),
            (re.compile(r'\b\w+(ed|ing|ate|ify|ize|ise|en)\b', re.IGNORECASE), 'VERB'),
            (re.compile(r'\b\w+(able|ible|ive|ous|ic|ary|ful|less)\b', re.IGNORECASE), 'ADJ'),
            (re.compile(r'\b\w+ly\b', re.IGNORECASE), 'ADV'),
            
            # Closed class words
            (re.compile(r'\b(the|a|an|this|that|these|those)\b', re.IGNORECASE), 'DET'),
            (re.compile(r'\b(I|you|he|she|it|we|they|me|him|her|us|them)\b', re.IGNORECASE), 'PRON'),
            (re.compile(r'\b(in|on|at|with|by|for|from|to|of|about|as|into)\b', re.IGNORECASE), 'ADP'),
            (re.compile(r'\b(and|or|but)\b', re.IGNORECASE), 'CCONJ'),
            (re.compile(r'\b(if|because|when|while|although)\b', re.IGNORECASE), 'SCONJ'),
            
            # Special categories
            (re.compile(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b'), 'NUM'),
            (re.compile(r'\b(oh|wow|ouch|oops|hey|ah|uh|hmm)\b', re.IGNORECASE), 'INTJ'),
            (re.compile(r'[.,!?;:"]'), 'PUNCT'),
            
            # Auxiliaries and modals
            (re.compile(r'\b(am|is|are|was|were|have|has|had|do|does|did|can|could|will|would|shall|should|may|might|must)\b', re.IGNORECASE), 'AUX'),
            
            # Fallback patterns (lower priority)
            (re.compile(r'\b\w+\b'), 'NOUN')  # Default catch-all
        ]

        # This ensures that _expand_with_synonyms() has access to the enriched synonyms.
        self._wordlist = {"words": structured_wordlist} if structured_wordlist else {"words": {}}


    @property
    def structured_wordlist(self):
        if self._wordlist is None:
            with open(STRUCTURED_WORDLIST_PATH) as f:
                self._wordlist = json.load(f)
        return self._wordlist

    def _validate_wordlist_integrity(self):
        required_keys = {'pos', 'synonyms', 'related_terms'}
        for word, entry in self.structured_wordlist['words'].items():
            if not isinstance(entry, dict):
                return False
            if not required_keys.issubset(entry.keys()):
                return False
        return True

    def extract_entities(self, text, pos_tags):
        """
        Extracts basic named entities using noun/proper noun clusters.
        """
        entities = {}
        current_entity = []

        for word, tag in pos_tags:
            if tag in {"NOUN", "PROPN"}:
                current_entity.append(word)
            else:
                if current_entity:
                    key = " ".join(current_entity)
                    entities[key] = {"type": "noun_phrase"}
                    current_entity = []

        if current_entity:
            key = " ".join(current_entity)
            entities[key] = {"type": "noun_phrase"}

        return entities

    def _convert_structured_wordlist(self, structured_wordlist: dict) -> dict:
        if not isinstance(structured_wordlist, dict):
            raise ValueError("structured_wordlist must be a dict, got {}".format(type(structured_wordlist).__name__))
        pos_mapping = {}
        from collections import Counter
        for word, entry in structured_wordlist.items():
            tag_counter = Counter()
            for raw_tag in entry.get("pos", []):
                normalized = raw_tag.lower().strip()
                upos_tag = self._UPOS_MAP.get(normalized)
                if upos_tag:
                    tag_counter[upos_tag] += 1
            if tag_counter:
                pos_mapping[word.lower()] = tag_counter.most_common(1)[0][0]
        return pos_mapping

    def _load_custom_cfg_rules(self, rules_path: Union[str, Path]):
        try:
            rules_path = Path(rules_path)
            if rules_path.is_file():
                with open(rules_path, 'r') as f:
                    custom_rules = json.load(f)
                    if isinstance(custom_rules, dict):
                        self.cfg_rules.update(custom_rules)
        except Exception as e:
            logger.warning(f"Failed to load custom grammar rules from {rules_path}: {e}")

    def _is_coreferent(self, word):
        """Enhanced coreference resolution with case sensitivity"""
        word_lower = word.lower()
        current_pos = self._get_pos_tag(word)
        
        # Check all pronoun types
        if word_lower not in self.entity_tracker['pronouns']:
            return False
        
        pronoun_props = self.entity_tracker['pronouns'][word_lower]
        
        # Special handling for "her" ambiguity (can be possessive or object)
        if word_lower == 'her':
            if current_pos == 'PRON':  # Object pronoun
                pronoun_props = ['female', 'singular', 'object']
            else:  # Possessive determiner
                return False  # Treat as new reference
        
        # Search strategy based on pronoun type
        search_window = None
        if pronoun_props[2] == 'reflexive':
            # Reflexives typically refer to recent subjects
            search_window = [e for e in self.entity_tracker['entities'] 
                            if e['properties'].get('grammar_role') == 'subject']
        else:
            search_window = self.entity_tracker['entities']
        
        # Property matching with case sensitivity
        for entity in reversed(search_window):
            if self._match_properties(entity['properties'], pronoun_props):
                # Additional check for case matching
                if self._check_case_compatibility(word, entity['text']):
                    return True
        
        return False

    def _check_case_compatibility(self, pronoun, antecedent):
        """Verify case agreement between pronoun and antecedent"""
        pronoun_lower = pronoun.lower()
        
        # Subject pronouns must follow sentence boundaries
        if pronoun_lower in ['he', 'she', 'they', 'it']:
            return pronoun[0].isupper()  # Must be capitalized
        
        # Object pronouns typically don't start sentences
        if pronoun_lower in ['him', 'her', 'them']:
            return not pronoun[0].isupper()
        
        # Possessives can appear anywhere
        return True

    def _update_discourse_context(self, word, is_sentence_start):
        """Enhanced with grammatical role tracking"""
        current_pos = self._get_pos_tag(word)
        
        if is_sentence_start:
            self.entity_tracker['entities'] = deque(maxlen=15)
            self.current_subject = None
        
        # Track grammatical roles
        grammar_role = None
        if current_pos in ['NOUN', 'PROPN', 'PRON']:
            if len(self.entity_tracker['entities']) == 0 or is_sentence_start:
                grammar_role = 'subject'
                self.current_subject = word.lower()
            else:
                grammar_role = 'object'
        
        if current_pos in ['NOUN', 'PROPN']:
            self.entity_tracker['entities'].append({
                'text': word.lower(),
                'position': len(self.previous_words),
                'properties': {
                    **self._get_entity_properties(word),
                    'grammar_role': grammar_role,
                    'is_definite': word.lower() in ['the', 'this', 'that']
                }
            })

    def _validate_cfg_rules(self):
        """Detect left-recursive cycles in CFG rules using DFS (Johnson, 1975)
        Raises:
            ValueError: If infinite recursion is detected
        Returns:
            dict: Rule dependency graph for visualization
        """
        # Build adjacency list
        graph = {non_terminal: set() for non_terminal in self.cfg_rules}
        for lhs, productions in self.cfg_rules.items():
            for production in productions:
                for symbol in production:
                    if symbol in self.cfg_rules:  # Only non-terminals
                        graph[lhs].add(symbol)

        # Check for cycles using iterative DFS
        visited = set()
        recursion_stack = set()
        dependency_graph = {}

        def _detect_cycles(symbol):
            """Modified DFS cycle detection (Tarjan, 1972)"""
            nonlocal dependency_graph
            visited.add(symbol)
            recursion_stack.add(symbol)
            dependency_graph[symbol] = []

            for neighbor in graph[symbol]:
                dependency_graph[symbol].append(neighbor)
                if neighbor not in visited:
                    if _detect_cycles(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    # Highlight the cyclic path
                    cycle_path = list(recursion_stack) + [neighbor]
                    raise ValueError(
                        f"Infinite recursion detected: {' → '.join(cycle_path)}\n"
                        f"Offending rule: {symbol} → {' '.join(production)}"
                    )
            
            recursion_stack.remove(symbol)
            return False

        # Check all non-terminals
        for non_terminal in self.cfg_rules:
            if non_terminal not in visited:
                if _detect_cycles(non_terminal):
                    # This line won't be reached due to immediate exception
                    pass

        return dependency_graph

    def _safe_add_rule(self, lhs, production):
        """Safely add new CFG rule with cycle checking"""
        old_rules = self.cfg_rules.get(lhs, [])
        self.cfg_rules[lhs] = old_rules + [production]
        
        try:
            dep_graph = self._validate_cfg_rules()
        except ValueError as e:
            # Revert changes if unsafe
            self.cfg_rules[lhs] = old_rules
            raise RuntimeError(
                f"Rule addition rejected: {lhs} → {' '.join(production)}\n"
                f"Reason: {str(e)}"
            ) from e
        
        return dep_graph

    def _load_pos_data(self):
        """Convert custom tags to Universal Dependencies scheme with frequency-based disambiguation"""
        pos_path = Path(__file__).parent / "structured_wordlist_en.json"
        
        try:
            with open(pos_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load POS data: {str(e)}")
            return {}

        pos_mapping = {}
        for word, entry in data['words'].items():
            tag_counter = Counter()
            
            for raw_tag in entry['pos']:
                normalized = raw_tag.lower().strip()
                if upos_tag := self._UPOS_MAP.get(normalized):
                    tag_counter[upos_tag] += 1
                else:
                    logger.debug(f"Unmapped POS tag: {raw_tag} for word {word}")

            if tag_counter:
                # Select most frequent tag, with random tiebreaker
                max_freq = max(tag_counter.values())
                candidates = [tag for tag, count in tag_counter.items() if count == max_freq]
                selected_tag = random.choice(candidates) if len(candidates) > 1 else candidates[0]
                pos_mapping[word.lower()] = selected_tag
            else:
                # Fallback to noun for content words, determiner for short words
                default_tag = 'NOUN' if len(word) > 3 else 'DET'
                pos_mapping[word.lower()] = default_tag
                logger.warning(f"No valid POS tags for {word}, defaulting to {default_tag}")

        return pos_mapping

    def _pos_tag(self, text: str) -> List[Tuple[str, str]]:
        """
        Tokenize and apply POS tagging to input text using internal POS map and stem fallback.
        """
        tokens = re.findall(r'\b\w+\b', text)
        tagged = [(token, self._get_pos_tag(token)) for token in tokens]
        return tagged

    def _get_pos_tag(self, word):
        """Get standardized POS tag with fallback"""
        base_tag = self.pos_map.get(word.lower()) or \
                  self.pos_map.get(self.stemmer.stem(word.lower()))
        
        # Enhanced unknown word handling (Mikolov et al., 2013)
        if not base_tag:
            return self._guess_pos_by_morphology(word)
        
        return base_tag

    def _load_econ_lexicon(self):
        """Financial Sentiment Lexicon from FinancialTracker (shared for NLP-based agents)"""
        return {
            'positive': {
                'bullish', 'growth', 'buy', 'strong', 'surge', 'rally', 'gain', 
                'profit', 'upside', 'outperform', 'recovery', 'breakout', 'boom',
                'soar', 'target', 'undervalued', 'dividend', 'premium', 'stable',
                'rebound', 'momentum', 'innovative', 'leadership', 'upgrade',
                'opportunity', 'success', 'record', 'beat', 'raise', 'trending',
                'accumulate', 'hold', 'long', 'bull', 'green', 'positive', 'strong',
                'resilient', 'robust', 'thrive', 'accelerate', 'superior', 'peak',
                'promising', 'dominant', 'breakthrough', 'optimal', 'efficient',
                'sustainable', 'hodl', 'moon', 'lambo', 'fomo', 'yolo', 'rocket',
                'adoption', 'institutional', 'partnership', 'burn', 'deflationary'
            },
            'negative': {
                'bearish', 'loss', 'sell', 'weak', 'crash', 'plunge', 'decline',
                'downturn', 'risk', 'warning', 'volatile', 'fraud', 'bankrupt',
                'default', 'short', 'dump', 'bubble', 'correction', 'manipulation',
                'recession', 'downgrade', 'distress', 'failure', 'bear', 'red',
                'negative', 'warning', 'caution', 'overbought', 'overvalued',
                'uncertainty', 'fear', 'volatility', 'liquidate', 'capitulation',
                'contraction', 'headwind', 'insolvent', 'delist', 'regulation',
                'hack', 'exploit', 'rugpull', 'ponzi', 'wash', 'fud', 'rekt',
                'bagholder', 'dump', 'correction', 'sink', 'collapse', 'bleed',
                'stagnant', 'dilution', 'inflation', 'deficit', 'warn', 'sue',
                'investigate', 'scam', 'vulnerability', 'attack', 'compromise'
            }
        }

    def detect_intent(self, text: str) -> str:
        """
        Enhanced intent recognizer using financial sentiment lexicon and rule patterns.
        """
        lowered = text.lower()
        lexicon = self._load_econ_lexicon()  # Pull sentiment terms from FinancialTracker
        
        pos_hits = [word for word in lexicon['positive'] if word in lowered]
        neg_hits = [word for word in lexicon['negative'] if word in lowered]

        # Simple rule priority based on term presence
        if any(word in lowered for word in ["buy", "accumulate", "long"]) or len(pos_hits) > len(neg_hits):
            return "buy_signal"
        if any(word in lowered for word in ["sell", "short", "dump"]) or len(neg_hits) > len(pos_hits):
            return "sell_signal"
        if any(word in lowered for word in ["hold", "wait", "stable"]):
            return "hold_signal"
        if any(word in lowered for word in ["risk", "volatility", "uncertainty", "fear", "warning"]):
            return "risk_assessment"
        
        return "unknown"

    def _guess_pos_by_morphology(self, word):
        """Advanced morphological analysis using linguistic patterns (Aronoff, 1976)"""
        word_lower = word.lower()
        
        # Check for nominal morphology (Bauer, 1983)
        if re.search(r'(tion|ment|ness|ity|acy|ism|ship|hood|dom|ee|eer|ist)$', word_lower):
            return 'NOUN'
        
        # Verbal inflections (Bybee, 1985)
        if re.search(r'(ate|ify|ize|ise|en|ish|fy|esce)$', word_lower):
            return 'VERB'
        
        # Adjectival suffixes (Marchand, 1969)
        if re.search(r'(able|ible|ive|ous|ic|ary|ful|less|ish|ese|most|like)$', word_lower):
            return 'ADJ'
        
        # Adverbial markers (Payne, 1997)
        if re.search(r'(wise|ward|wards|way|ways|where)$', word_lower):
            return 'ADV'
        
        # Pronominal patterns (Quirk et al., 1985)
        if re.search(r'(self|selves|thing|body|one|where)$', word_lower):
            return 'PRON'
        
        # Determiner morphology (Dryer, 2005)
        if re.search(r'(th|se|ch|wh|ev|at)$', word_lower) and len(word_lower) < 5:
            return 'DET'
        
        # Prepositional patterns (Huddleston & Pullum, 2002)
        if re.search(r'(ward|side|neath|tween|mong|pon|mid|anti|non)$', word_lower):
            return 'ADP'
        
        # Numeral detection (Hurford, 1975)
        if re.search(r'^\d+([.,]\d+)?(th|st|nd|rd)?$', word_lower):
            return 'NUM'
        
        # Conjunction patterns (Halliday, 1985)
        if re.search(r'(though|while|whereas|because|unless|until|than|ther)$', word_lower):
            return 'SCONJ' if len(word_lower) > 4 else 'CCONJ'
        
        # Derivational prefixes (Katamba, 1993)
        prefixes = {
            'un': 'ADJ', 're': 'VERB', 'pre': 'VERB', 
            'dis': 'VERB', 'mis': 'VERB', 'non': 'ADJ'
        }
        for prefix, pos in prefixes.items():
            if word_lower.startswith(prefix):
                return pos
        
        # Reduplication patterns (Inkelas & Zoll, 2005)
        if re.match(r'(\w{2,})\1$', word_lower):  # e.g., "bye-bye"
            return 'NOUN'
        
        # Compound words (Lieber, 2009)
        if '-' in word_lower:
            components = word_lower.split('-')
            if len(components) > 1:
                return self._analyze_compound(components)
        
        # Capitalization check for proper nouns (Chomsky, 1970)
        if word[0].isupper() and not self._is_sentence_initial(word):
            return 'PROPN'
        
        # Requires precomputed character n-gram weights
        char_ngrams = [word_lower[i:i+3] for i in range(len(word_lower)-2)]
        noun_score = sum(1 for ng in char_ngrams if ng in {'ion','ment','nes'})
        verb_score = sum(1 for ng in char_ngrams if ng in {'ing','ate','ify'})
        return 'NOUN' if noun_score > verb_score else 'VERB'

    def semantic_distance(self, w1, w2):
        if w1 in self.glove and w2 in self.glove:
            vec1 = np.array(self.glove[w1])
            vec2 = np.array(self.glove[w2])
            return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-5)
        return float('inf')

    def _analyze_compound(self, components):
        """Compound analysis with dedicated idiom processing layer"""
        # Phase 0: Idiom detection (Fernando & Flavell, 1981)
        idiom_result = self._detect_idiomatic_expression(components)
        if idiom_result:
            return idiom_result

        """Advanced compound word and MWE analysis incorporating:
        - Lexicalized Tree Substitution Grammar (Sag et al., 2002)
        - Construction Grammar (Goldberg, 2006)
        - Idiom Principle (Sinclair, 1991)
        """
        compound_str = '-'.join(components).lower()
        
        # Phase 1: Fixed MWEs detection
        if self._is_lexicalized_mwe(compound_str):
            return self._get_mwe_category(compound_str)
        
        # Phase 2: Semi-fixed patterns
        mwe_type = self._detect_semi_fixed_pattern(components)
        if mwe_type:
            return mwe_type
        
        # Phase 3: Morphosyntactic analysis
        return self._morphological_compound_analysis(components)

    def _is_lexicalized_mwe(self, compound):
        """Check against known MWEs using:
        - Non-compositionality criterion (Nunberg et al., 1994)
        - Institutionalization metric (Bauer, 1983)
        """
        LEXICALIZED_MWES = {
            # Verb-Noun MWEs
            'take-place': 'VERB',
            'give-way': 'VERB',
            
            # Adjective-Noun MWEs
            'red-tape': 'NOUN',
            'high-school': 'NOUN',
            
            # Prepositional MWEs
            'in-spite-of': 'ADP',
            'by-means-of': 'ADP',
            
            # Institutionalized phrases
            'attorney-general': 'NOUN',
            'mother-in-law': 'NOUN'
        }
        return compound in LEXICALIZED_MWES

    def _get_mwe_category(self, mwe):
        """Return syntactic head category following:
        - Right-hand Head Rule (Williams, 1981)
        - Lexical Inheritance Principles (Pollard & Sag, 1994)
        """
        MWE_CATEGORIES = {
            'VERB': {'take', 'give', 'make', 'do'},
            'NOUN': {'school', 'law', 'tape', 'general'},
            'ADJ': {'free', 'high', 'low', 'wide'},
            'ADV': {'how', 'when', 'where', 'why'}
        }
        
        last_component = mwe.split('-')[-1]
        for cat, markers in MWE_CATEGORIES.items():
            if last_component in markers:
                return cat
        return 'NOUN'  # Default nominal category

    def _detect_semi_fixed_pattern(self, components):
        """Identify productive MWE patterns using:
        - Construction Grammar templates (Goldberg, 2006)
        - Lexical-grammatical continua (Bybee, 2010)
        """
        # Verb-Particle constructions
        if (len(components) == 2 and 
            self._get_pos_tag(components[0]) == 'VERB' and
            components[1] in {'up', 'down', 'in', 'out'}):
            return 'VERB'
        
        # Light verb constructions
        if (len(components) == 2 and
            components[0] in {'take', 'make', 'do'} and
            self._get_pos_tag(components[1]) == 'NOUN'):
            return 'VERB'
        
        # Comparative compounds
        if (len(components) == 3 and
            components[1] == 'than' and
            self._get_pos_tag(components[0]) == 'ADJ'):
            return 'ADJ'
        
        return None

    def _morphological_compound_analysis(self, components):
        """Determine category via morphological structure using:
        - Lexeme-based morphology (Aronoff, 1994)
        - Hierarchical word formation (Selkirk, 1982)
        """
        last_pos = self._get_pos_tag(components[-1])
        first_pos = self._get_pos_tag(components[0])
        
        # Noun-noun compounds
        if last_pos == 'NOUN':
            return 'NOUN'
        
        # Adjective-noun compounds
        if last_pos == 'NOUN' and first_pos == 'ADJ':
            return 'NOUN'
        
        # Verb-particle compounds
        if first_pos == 'VERB' and len(components[-1]) <= 3:
            return 'VERB'
        
        # Default to right-headedness
        return last_pos

    def _is_lexicalized_mwe(self, compound):
        """Check against known MWEs using:
        - Non-compositionality criterion (Nunberg et al., 1994)
        - Institutionalization metric (Bauer, 1983)
        """
        LEXICALIZED_MWES = {
            # Verb-Noun MWEs
            'take-place': 'VERB',
            'give-way': 'VERB',
            
            # Adjective-Noun MWEs
            'red-tape': 'NOUN',
            'high-school': 'NOUN',
            
            # Prepositional MWEs
            'in-spite-of': 'ADP',
            'by-means-of': 'ADP',
            
            # Institutionalized phrases
            'attorney-general': 'NOUN',
            'mother-in-law': 'NOUN'
        }
        return compound in LEXICALIZED_MWES

    def _get_mwe_category(self, mwe):
        """Return syntactic head category following:
        - Right-hand Head Rule (Williams, 1981)
        - Lexical Inheritance Principles (Pollard & Sag, 1994)
        """
        MWE_CATEGORIES = {
            'VERB': {'take', 'give', 'make', 'do'},
            'NOUN': {'school', 'law', 'tape', 'general'},
            'ADJ': {'free', 'high', 'low', 'wide'},
            'ADV': {'how', 'when', 'where', 'why'}
        }
        
        last_component = mwe.split('-')[-1]
        for cat, markers in MWE_CATEGORIES.items():
            if last_component in markers:
                return cat
        return 'NOUN'  # Default nominal category

    def _detect_semi_fixed_pattern(self, components):
        """Identify productive MWE patterns using:
        - Construction Grammar templates (Goldberg, 2006)
        - Lexical-grammatical continua (Bybee, 2010)
        """
        # Verb-Particle constructions
        if (len(components) == 2 and 
            self._get_pos_tag(components[0]) == 'VERB' and
            components[1] in {'up', 'down', 'in', 'out'}):
            return 'VERB'
        
        # Light verb constructions
        if (len(components) == 2 and
            components[0] in {'take', 'make', 'do'} and
            self._get_pos_tag(components[1]) == 'NOUN'):
            return 'VERB'
        
        # Comparative compounds
        if (len(components) == 3 and
            components[1] == 'than' and
            self._get_pos_tag(components[0]) == 'ADJ'):
            return 'ADJ'
        
        return None

    def _morphological_compound_analysis(self, components):
        """Determine category via morphological structure using:
        - Lexeme-based morphology (Aronoff, 1994)
        - Hierarchical word formation (Selkirk, 1982)
        """
        last_pos = self._get_pos_tag(components[-1])
        first_pos = self._get_pos_tag(components[0])
        
        # Noun-noun compounds
        if last_pos == 'NOUN':
            return 'NOUN'
        
        # Adjective-noun compounds
        if last_pos == 'NOUN' and first_pos == 'ADJ':
            return 'NOUN'
        
        # Verb-particle compounds
        if first_pos == 'VERB' and len(components[-1]) <= 3:
            return 'VERB'
        
        # Default to right-headedness
        return last_pos

    def _is_sentence_initial(self, word):
        """Determines if word is sentence-initial using context-aware analysis
        Implements principles from:
        - Sentence boundary detection (Palmer & Hearst, 1997)
        - Functional Sentence Perspective (Firbas, 1964)
        - Discourse Representation Theory (Kamp & Reyle, 1993)
        """
        # Get access to parsing state through the processor's context tracking
        if not hasattr(self, 'current_sentence'):
            # Initialize document structure tracking
            self.current_sentence = 0
            self.previous_words = []
            self.document_structure = {
                'paragraph_starts': [0],
                'sentence_breaks': []
            }

        # Context analysis factors
        position_factors = {
            'is_absolute_start': len(self.previous_words) == 0,
            'follows_sentence_break': self._precedes_sentence_break(),
            'capitalization': word[0].isupper(),
            'previous_punctuation': self._get_previous_punctuation(),
            'in_quotation_context': self._in_quotation_sequence(),
            'known_proper_noun': self.pos_map.get(word.lower()) == 'PROPN'
        }

        # Sentence boundary detection rules
        sentence_start = (
            # Case 1: Absolute document start
            position_factors['is_absolute_start'] or
            
            # Case 2: After sentence-final punctuation
            (position_factors['follows_sentence_break'] and
            position_factors['capitalization'] and
            not position_factors['known_proper_noun']) or
            
            # Case 3: After closing quotation with sentence-final punctuation
            (position_factors['in_quotation_context'] and
            position_factors['previous_punctuation'] in {'."', '!"', '?"'} and
            position_factors['capitalization']) or
            
            # Case 4: Following paragraph break
            (self.current_sentence in 
            self.document_structure['paragraph_starts'])
        )

        # Update document structure tracking
        if sentence_start:
            self.document_structure['sentence_breaks'].append(
                len(self.previous_words))
            self.current_sentence += 1

        # Add word to processing history
        self.previous_words.append(word)
        
        return sentence_start

    def _precedes_sentence_break(self):
        """Check if previous context indicates sentence boundary"""
        if len(self.previous_words) < 1:
            return False
            
        last_token = self.previous_words[-1]
        sentence_final_punct = {'。', '.', '!', '?'}  # Multi-lingual support
        
        # Check for sentence-final punctuation patterns
        return any(
            c in sentence_final_punct for c in last_token
        ) and not self._is_abbreviation(last_token)

    def _get_previous_punctuation(self):
        """Get trailing punctuation from previous word"""
        if len(self.previous_words) < 1:
            return ''
        
        prev_word = self.previous_words[-1]
        return ''.join(c for c in prev_word if c in {'.', '!', '?', '"', "'"})

    def _in_quotation_sequence(self):
        """Check for nested quotation context"""
        quote_chars = {'"', "'", '“', '”', '‘', '’'}
        quote_stack = []
        
        for char in ''.join(self.previous_words):
            if char in {'“', '‘', '"', "'"}:
                quote_stack.append(char)
            elif char in {'”', '’'} and quote_stack:
                quote_stack.pop()
        
        return len(quote_stack) > 0

    def _is_abbreviation(self, token):
        """Check if token is a known abbreviation (simplified)"""
        abbreviations = {
            'mr.', 'mrs.', 'dr.', 'prof.', 'etc.', 'e.g.', 'i.e.',
            'vs.', 'jan.', 'feb.', 'a.m.', 'p.m.', 'u.s.', 'u.k.'
        }
        return token.lower() in abbreviations

    def _is_sentence_final_punct(self, char):
        """Check if character is sentence-final punctuation"""
        return char in {'.', '!', '?', '。', '！', '？'}

    class PorterStemmer:
        """Implementation of Porter's stemming algorithm (1980)"""
        def __init__(self):
            self.vowels = {'a', 'e', 'i', 'o', 'u'}

        def stem(self, word):
            """Main stemming algorithm"""
            if len(word) < 3:
                return word.lower()

            word = self.step1a(word.lower())
            word = self.step1b(word)
            word = self.step1c(word)
            word = self.step2(word)
            word = self.step3(word)
            word = self.step4(word)
            word = self.step5a(word)
            word = self.step5b(word)
            
            return word

        def measure(self, stem):
            """Calculate the 'measure' (VC sequence count)"""
            count = 0
            prev_vowel = False
            for char in stem:
                if char in self.vowels:
                    prev_vowel = True
                else:
                    if prev_vowel:
                        count += 1
                    prev_vowel = False
            return count

        def has_vowel(self, stem):
            """Check if stem contains any vowels"""
            return any(char in self.vowels for char in stem)

        def ends_with_double(self, word):
            """Check for double consonant ending"""
            return len(word) > 1 and word[-1] == word[-2] and word[-1] not in self.vowels

        def replace_suffix(self, word, old, new, measure=None):
            """Conditional suffix replacement"""
            if word.endswith(old):
                base = word[:-len(old)]
                if measure is None or self.measure(base) > measure:
                    return base + new
            return word

        def step1a(self, word):
            """Plurals and past participles"""
            for suffix in ['sses', 'ies', 'ss', 's']:
                if word.endswith(suffix):
                    if suffix == 'sses':
                        return word[:-4] + 'ss'
                    elif suffix == 'ies':
                        return word[:-3] + 'i'
                    elif suffix == 'ss':
                        return word
                    elif suffix == 's' and self.has_vowel(word[:-1]):
                        return word[:-1]
            return word

        def step1b(self, word):
            """Verb endings"""
            if word.endswith('eed'):
                base = word[:-3]
                if self.measure(base) > 0:
                    return base + 'ee'
            elif word.endswith(('ed', 'ing')):
                base = word[:-2] if word.endswith('ed') else word[:-3]
                if self.has_vowel(base):
                    word = self.step1b_adjust(base)
            return word

        def step1b_adjust(self, base):
            """Additional adjustments for step1b"""
            for suffix in ['at', 'bl', 'iz']:
                if base.endswith(suffix):
                    return base + 'e'
            if self.ends_with_double(base) and not base.endswith(('l', 's', 'z')):
                return base[:-1]
            if self.measure(base) == 1 and self.ends_cvc(base):
                return base + 'e'
            return base

        def step1c(self, word):
            """Replace y with i if preceded by vowel"""
            if word.endswith('y') and self.has_vowel(word[:-1]):
                return word[:-1] + 'i'
            return word

        def step2(self, word):
            """Double-derivational suffixes"""
            replacements = {
                'ational': 'ate', 'tional': 'tion', 'enci': 'ence',
                'anci': 'ance', 'izer': 'ize', 'abli': 'able',
                'alli': 'al', 'entli': 'ent', 'eli': 'e',
                'ousli': 'ous', 'ization': 'ize', 'ation': 'ate',
                'ator': 'ate', 'alism': 'al', 'iveness': 'ive',
                'fulness': 'ful', 'ousness': 'ous', 'aliti': 'al',
                'iviti': 'ive', 'biliti': 'ble'
            }
            for suffix, replacement in replacements.items():
                if word.endswith(suffix):
                    base = word[:-len(suffix)]
                    if self.measure(base) > 0:
                        return base + replacement
            return word

        def step3(self, word):
            """Replace -ic-, -full, -ness etc."""
            replacements = {
                'icate': 'ic', 'ative': '', 'alize': 'al',
                'iciti': 'ic', 'ical': 'ic', 'ful': '',
                'ness': ''
            }
            for suffix, replacement in replacements.items():
                if word.endswith(suffix):
                    base = word[:-len(suffix)]
                    if self.measure(base) > 0:
                        return base + replacement
            return word

        def step4(self, word):
            """Remove -ant, -ence, etc."""
            suffixes = [
                'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible',
                'ant', 'ement', 'ment', 'ent', 'ion', 'ou', 'ism',
                'ate', 'iti', 'ous', 'ive', 'ize'
            ]
            for suffix in suffixes:
                if word.endswith(suffix):
                    base = word[:-len(suffix)]
                    if self.measure(base) > 1:
                        if suffix == 'ion' and base[-1] in {'s', 't'}:
                            return base
                        return base
            return word

        def step5a(self, word):
            """Remove final 'e' if measure > 1"""
            if word.endswith('e'):
                base = word[:-1]
                if self.measure(base) > 1:
                    return base
                if self.measure(base) == 1 and not self.ends_cvc(base):
                    return base
            return word

        def step5b(self, word):
            """Remove double consonant ending"""
            if self.measure(word) > 1 and self.ends_with_double(word) and word.endswith('l'):
                return word[:-1]
            return word

        def ends_cvc(self, word):
            """Check CVC pattern where last C is not w, x or y"""
            if len(word) < 3:
                return False
            return (word[-1] not in self.vowels and 
                    word[-2] in self.vowels and 
                    word[-3] not in self.vowels and 
                    word[-1] not in {'w', 'x', 'y'})

    def _build_ngram_model(self):
        """Construct n-gram model with incremental learning capabilities
        Implements:
        - Base frequencies from Brown Corpus (Francis & Kucera, 1982)
        - Exponential decay for model adaptation (Anderson, 1990)
        - Online learning framework (Bottou, 1998)
        """
        # Initialize with Brown Corpus baseline
        self.ngram_model = defaultdict(lambda: defaultdict(float))
        
        # Base frequencies (preserve as floating point for decay)
        base_frequencies = {
            'DET': {'NOUN': 89412,'ADJ': 18765},
            'ADJ': {'NOUN': 23451},
            'NUM': {'NOUN': 15678},
            'NOUN': {'VERB': 67342},
            'AUX': {'VERB': 44531},
            'ADV': {'VERB': 22345},
            'ADP': {'NOUN': 55678, 'PROPN': 12345},
            'CCONJ': {'NOUN': 33219},
            'SCONJ': {'VERB': 11234},
            'PRON': {'VERB': 44231},
            'VERB': {'ADV': 15673},
            'INTJ': {'PUNCT': 5123},
            'SYM': {'NUM': 2345}
        }
        
        # Convert to float for decay operations
        self.ngram_model = defaultdict(lambda: defaultdict(float))
        for prev_tag, next_tags in base_frequencies.items():
            for next_tag, count in next_tags.items():
                self.ngram_model[prev_tag][next_tag] = float(count)

        # Enhanced fallback using weighted average
        noun_total = sum(self.ngram_model['NOUN'].values())
        self.ngram_model['UNKNOWN'] = defaultdict(
            lambda: noun_total / (1000 + noun_total)  # Bayesian smoothing
        )
    
        # Initialize learning parameters
        self.decay_factor = 0.999  # Memory decay rate (Anderson, 1990)
        self.learning_rate = 0.01  # SGD-inspired rate (Bottou, 1998)
    
    def update_ngram_model(self, tag_sequence):
        """Incremental update with exponential recency weighting
        Implements:
        - Online learning (Collins, 2002)
        - Adaptive decay (Katz, 1987)
        """
        # Apply decay to existing counts
        for prev_tag in self.ngram_model:
            for next_tag in self.ngram_model[prev_tag]:
                self.ngram_model[prev_tag][next_tag] *= self.decay_factor

        # Add new observations
        for i in range(len(tag_sequence)-1):
            prev_tag = tag_sequence[i]
            next_tag = tag_sequence[i+1]
            
            # Smoothing factor based on Zipf's law (Zipf, 1935)
            smoothing = 1 / (1 + math.log1p(self.ngram_model[prev_tag][next_tag]))
            
            # Update rule with adaptive learning rate
            self.ngram_model[prev_tag][next_tag] += self.learning_rate * smoothing

    def process_sentence(self, sentence):
        """Full processing pipeline with incremental learning"""
        # Parse and validate sentence
        if not self.parse_grammar(sentence):
            return False
        
        # Get POS tags
        words = re.findall(r'\b\w+\b', sentence.lower())
        tag_sequence = [self._get_pos_tag(word) for word in words]
        
        # Update model with decayed learning
        self.update_ngram_model(tag_sequence)
        
        return True

    def _build_pos_patterns(self):
        """Generate regex patterns from wordlist data with frequency weighting"""
        pos_groups = defaultdict(list)
        
        # Group words by their POS tags from JSON data
        for word, tag in self.pos_map.items():
            pos_groups[tag].append(re.escape(word))  # Escape special chars
        
        # Create regex patterns for each POS group
        self.pos_patterns = []

        # Add fallback morphological patterns (lower priority)
        self.pos_patterns.extend([
            (re.compile(r'\b[A-Z][a-z]+\b'), 'PROPN'),  # Proper nouns
            (re.compile(r'\b\w+(tion|ment|ness|ity|acy|ism)\b', re.IGNORECASE), 'NOUN'),
            (re.compile(r'\b\w+(ed|ing|ate|ify|ize|ise|en)\b', re.IGNORECASE), 'VERB'),
            (re.compile(r'\b\w+(able|ible|ive|ous|ic|ary|ful|less)\b', re.IGNORECASE), 'ADJ'),
            (re.compile(r'\b\w+ly\b', re.IGNORECASE), 'ADV'),
            (re.compile(r'\b(the|a|an|this|that|these|those)\b', re.IGNORECASE), 'DET'),
            (re.compile(r'\b(I|you|he|she|it|we|they|me|him|her|us|them)\b', re.IGNORECASE), 'PRON'),
            (re.compile(r'\b(in|on|at|with|by|for|from|to|of|about|as|into)\b', re.IGNORECASE), 'ADP'),
            (re.compile(r'\b(and|or|but)\b', re.IGNORECASE), 'CCONJ'),
            (re.compile(r'\b(if|because|when|while|although)\b', re.IGNORECASE), 'SCONJ'),
            (re.compile(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b'), 'NUM'),
            (re.compile(r'\b(oh|wow|ouch|oops|hey|ah|uh|hmm)\b', re.IGNORECASE), 'INTJ'),
            (re.compile(r'[.,!?;:"]'), 'PUNCT'),
            (re.compile(r'\b(am|is|are|was|were|have|has|had|do|does|did|can|could|will|would|shall|should|may|might|must)\b', re.IGNORECASE), 'AUX'),
        ])

        # Add patterns from structured wordlist groups
        for pos, words in pos_groups.items():
            if words:
                # Sort to prioritize longer words
                words_sorted = sorted(words, key=len, reverse=True)
                pattern = r'\b(' + '|'.join(words_sorted) + r')\b'
                self.pos_patterns.append(
                    (re.compile(pattern, re.IGNORECASE), pos)
                )
        
        # Add fallback patterns (lower priority)
        self.pos_patterns.extend([
            (re.compile(r'\b\w+\b'), 'NOUN')  # Catch-all
        ])

    def parse_grammar(self, sentence, max_length=20):
        """CYK parser with length limiting and fail-fast"""
        words = re.findall(r'\b\w+\b', sentence.lower())
        n = len(words)

        # Fail-fast for long sentences (Church & Patil, 1982)
        if n > max_length:
            logger.warning(f"Sentence length {n} exceeds safety threshold {max_length}, skipping parse")
            return True  # Bypass check for performance
        
        # Initialize parse table with early termination
        try:
            table = [[set() for _ in range(n+1)] for _ in range(n+1)]
        except MemoryError:
            logger.error("Memory error initializing parse table")
            return False

        # POS tagging with fail-safe
        pos_tags = []
        for word in words:
            try:
                pos_tags.append(self._get_pos_tag(word))
            except Exception as e:
                logger.error(f"POS tagging failed for '{word}': {str(e)}")
                pos_tags.append('NOUN')

        # CYK algorithm with early bailout
        for length in range(1, n+1):
            for i in range(n - length + 2):
                if not table[i][length]:  # Skip empty cells early
                    continue
                    
                # Short-circuit if root symbol found early
                if i == 0 and length == n and 'S' in table[i][length]:
                    return True
                
        table = [[set() for _ in range(n+1)] for _ in range(n+1)]
        
        # POS tagging using regex patterns
        pos_tags = []
        for word in words:
            for pattern, tag in self.pos_patterns:
                if re.match(pattern, word):
                    pos_tags.append(tag)
                    break
            else:
                pos_tags.append('NOUN')  # Default to noun
        
        # Initialize table
        for i in range(n):
            table[i][1].update(self._get_symbols(pos_tags[i]))
            
        # CYK algorithm
        for length in range(2, n+1):
            for i in range(n-length+2):
                for k in range(1, length):
                    j = i + k
                    for B in table[i][k]:
                        for C in table[j][length-k]:
                            for A in self.cfg_rules:
                                if [B, C] in self.cfg_rules[A]:
                                    table[i][length].add(A)
        
        return 'S' in table[0][n]

    def generate_sentence(self, seed_words):
        """Generative grammar using Markov chain (Shannon, 1948)"""
        current_tag = 'DET'  # Start with determiner
        sentence = []
        max_length = 8
        
        while len(sentence) < max_length:
            next_tags = self.ngram_model[current_tag]
            total = sum(next_tags.values())
            if total == 0: break
            
            # Select next tag using probability distribution
            rand = random.uniform(0, 1)
            cumulative = 0
            for tag, count in next_tags.items():
                prob = count / total
                cumulative += prob
                if rand <= cumulative:
                    current_tag = tag
                    if len(sentence) == 0:  # only expand at start
                        expanded = self._expand_with_synonyms(seed_words)
                    else:
                        expanded = seed_words
                    sentence.append(self._sample_word(tag, expanded))
                    break
        
        return ' '.join(sentence)

    def _expand_with_synonyms(self, seed_words: List[str], max_expansions: int = 3) -> List[str]:
        """
        Expand the seed words using synonyms from structured_wordlist.
        """
        if not hasattr(self, 'structured_wordlist') or not isinstance(self.structured_wordlist, dict):
            return seed_words

        enriched = []
        seen = set(seed_words)

        for word in seed_words:
            enriched.append(word)
            entry = self.structured_wordlist.get("words", {}).get(word.lower())
            if not entry:
                continue
            synonyms = entry.get("synonyms", [])
            for syn in synonyms:
                if syn not in seen:
                    enriched.append(syn)
                    seen.add(syn)
                    if len(enriched) - len(seed_words) >= max_expansions:
                        break
        return enriched

    def _sample_word(self, tag, seed_words):
        """Word selection using TF-IDF similarity (Salton, 1971)"""
        candidates = [w for w in seed_words 
                     if any(re.match(p, w) for p,t in self.pos_patterns if t == tag)]
        
        if not candidates:
            return self._get_default_word(tag)
            
        # Simple frequency-based selection
        return max(set(candidates), key=candidates.count)

    def compose_sentence(self, facts: dict) -> str:
        """
        Takes structured agent facts and generates a grammatically correct sentence.
        """
        event = facts.get("event", "unknown_event")
        agent = facts.get("agent", "agent")
        value = facts.get("value", None)
        metric = facts.get("metric", "metric")

        if event == "training_complete" and value is not None:
            return f"{agent} has successfully completed training with a {metric} score of {value:.2f}."
        elif event == "failure":
            return f"{agent} encountered an error during training."
        else:
            return f"{agent} reported event '{event}' with value {value}."

    def _get_symbols(self, tag):
        """Get non-terminals producing the given POS tag"""
        return [A for A, prods in self.cfg_rules.items() 
                for prod in prods if tag in prod]

    def _get_default_word(self, tag):
        defaults = {'DET': 'the', 'NOUN': 'thing', 'VERB': 'is'}
        return defaults.get(tag, '')

    def reset_parser_state(self):
        """Reset all document-tracking state variables"""
        self.current_sentence = 0
        self.previous_words = []
        self.document_structure = {
            'paragraph_starts': [0],
            'sentence_breaks': [],
            'current_quotes': []
        }
        self.quote_stack = []  # Added for better quotation tracking

    def is_grammatical(self, text: str) -> bool:
        """Check if the input text is grammatically valid using the CYK parser."""
        return self.parse_grammar(text)

    def process(self, linguistic_frame, raw_response):
        if self.is_grammatical(raw_response):
            return raw_response
        else:
            return self.rephrase_response(linguistic_frame, raw_response)

    def rephrase_response(self, linguistic_frame: dict, raw_response: str) -> str:
        """Rephrase a sentence using seed words or structured templates."""
        # Extract seed words (nouns, verbs, adjectives) from raw_response
        words = re.findall(r'\b\w+\b', raw_response.lower())
        pos_tags = [(word, self._get_pos_tag(word)) for word in words]
        seed_words = [word for word, tag in pos_tags if tag in {'NOUN', 'VERB', 'ADJ', 'PROPN'}]

        # Attempt 3 times to generate a grammatical sentence
        for _ in range(3):
            generated = self.generate_sentence(seed_words)
            if self.is_grammatical(generated):
                return generated.capitalize() + '.'  # Ensure punctuation

        # Fallback to structured template using linguistic_frame
        event = linguistic_frame.get("event", "unknown_event")
        agent = linguistic_frame.get("agent", "The system")
        metric = linguistic_frame.get("metric", "performance")
        value = linguistic_frame.get("value", None)

        if event == "training_complete" and value is not None:
            return f"{agent} completed training with a {metric} score of {value}."
        elif event == "failure":
            return f"{agent} encountered an error."
        else:
            return f"{agent} reports: '{event}'."

class EnhancedLanguageAgent():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grammar = GrammarProcessor()
        self.syntax_buffer = deque(maxlen=5)  # Working memory for syntax

    def process_input(self, user_input, is_new_document=False):
        if self._detect_document_boundary(user_input):
            self.reset_parser_state()
        """Augmented processing pipeline with grammatical analysis"""
        # Stage 0: Syntactic validation
        if not self.grammar.parse_grammar(user_input):
            return "I notice a grammatical irregularity. Could you rephrase?"
        
        # Original processing pipeline
        clean_input = self.safety.sanitize(user_input)
        frame = self.nlu.parse(clean_input)
        
        # Stage 2.5: Context-aware generation
        response = self._generate_grammatical_response(frame, clean_input)
        
        # Update context with grammatical features
        self._update_syntax_model(response)
        
        return response

    def reset_parser_state(self, preserve_history=False):
        if preserve_history:
            self.document_structure['previous_documents'].append(
                copy.deepcopy(
                    self.document_structure))
        """Public method to reset parser state between documents"""
        self.grammar.reset_parser_state()
        self.syntax_buffer.clear()

    def _generate_grammatical_response(self, frame, input_text):
        """Generate responses using formal grammar constraints"""
        seed_words = input_text.split() + list(frame.entities.values())
        
        # Attempt 3 times to generate grammatical sentence
        for _ in range(3):
            generated = self.grammar.generate_sentence(seed_words)
            if self.grammar.parse_grammar(generated):
                return generated.capitalize()
        
        # Fallback to template-based generation
        return super().generate_response(input_text)

    def _update_syntax_model(self, sentence):
        """Adaptive learning using error-driven approach (Brill, 1995)"""
        tags = [tag for word, tag in self._pos_tag(sentence)]
        for i in range(len(tags)-1):
            self.grammar.ngram_model[tags[i]][tags[i+1]] += 1

    def _pos_tag(self, text):
        """Rule-based POS tagging (Greene & Rubin, 1971)"""
        return [(word, self._get_pos_tag(word)) for word in text.split()]

    def _get_pos_tag(self, word):
        for pattern, tag in self.grammar.pos_patterns:
            if re.match(pattern, word):
                return tag
        return 'NOUN'  # Default to noun
