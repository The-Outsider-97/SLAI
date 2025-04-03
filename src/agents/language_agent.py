import re
import json
import time
import pickle
import hashlib
import ply.lex as lex

from textstat import textstat
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any, OrderedDict, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict, OrderedDict

@dataclass
class DialogueContext:
    """Stores conversation history and environment state. 
    Inspired by Adiwardana et al. (2020) for dialogue coherence."""
    history: deque = field(default_factory=lambda: deque(maxlen=10))
    environment_state: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    attention_weights: Dict[str, float] = field(default_factory=dict)

    def compress_history(self, threshold: float = 0.7) -> None:
        """Remove less important context entries using attention weights"""
        self.history = deque(
            [entry for entry, weight in zip(self.history, self.attention_weights.values()) 
             if weight > threshold],
            maxlen=self.history.maxlen
        )

@dataclass 
class LinguisticFrame:
    """Structured representation of language acts (inspired by Speech Act Theory)"""
    intent: str
    entities: Dict[str, str]
    sentiment: float  # Range [-1, 1]
    modality: str  # e.g., "query", "command", "clarification"
    confidence: float  # [0, 1]

# --------------------------
# Independent Modules
# --------------------------
class Wordlist:
    """Advanced linguistic processor with phonetics, morphology, and semantic analysis"""
    
    def __init__(self, path: Union[str, Path] = "learning/structured_wordlist_en.json"):
        self.path = Path(path)
        self.data = {}
        self.metadata = {}
        self._load()
        
        # Advanced caching systems
        self.lru_cache = OrderedDict()
        self.lfu_cache = defaultdict(int)
        self.max_cache_size = 10_000
        
        # Precomputed linguistic data
        self.phonetic_index = defaultdict(set)
        self.ngram_index = defaultdict(set)
        self._precompute_linguistic_data()
        
        # Language model parameters
        self.ngram_model = defaultdict(lambda: defaultdict(int))
        self._build_ngram_model()
        
        # Keyboard proximity costs
        self.keyboard_layout = {
            'q': {'w': 0.5, 'a': 0.7}, 'w': {'e': 0.5, 's': 0.7},
            # ... complete keyboard proximity mapping
        }

    def _load(self) -> None:
        """Robust data loading with validation"""
        if not self.path.exists():
            raise FileNotFoundError(f"Wordlist missing: {self.path}")
        
        with open(self.path, 'r') as f:
            raw = json.load(f)
        
        required_keys = {'words', 'metadata', 'version'}
        if not required_keys.issubset(raw.keys()):
            raise ValueError("Invalid wordlist format - missing required keys")
        
        self.data = raw['words']
        self.metadata = raw['metadata']
        self._validate_word_entries()

    def _validate_word_entries(self) -> None:
        """Ensure all entries have valid structure"""
        for word, entry in self.data.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid entry format for word: {word}")
            if 'pos' not in entry or 'synonyms' not in entry:
                raise ValueError(f"Missing required fields in entry: {word}")

    def _precompute_linguistic_data(self) -> None:
        """Precompute phonetic and n-gram indexes"""
        for word in self.data:
            # Phonetic representations
            self.phonetic_index[self._soundex(word)].add(word)
            self.phonetic_index[self._metaphone(word)].add(word)
            
            # N-gram profiles (tri-grams)
            ngrams = self._generate_ngrams(word, 3)
            for ng in ngrams:
                self.ngram_index[ng].add(word)

    def _build_ngram_model(self) -> None:
        """Build basic n-gram frequency model"""
        for word in self.data:
            for i in range(len(word)-1):
                self.ngram_model[word[i]][word[i+1]] += 1

    # PHONETIC ALGORITHMS ------------------------------------------------------
    
    def _soundex(self, word: str) -> str:
        """Soundex phonetic encoding implementation"""
        # Implementation details...
    
    def _metaphone(self, word: str) -> str:
        """Metaphone phonetic encoding implementation"""
        # Implementation details...

    # MORPHOLOGICAL ANALYSIS ---------------------------------------------------
    
    def stem(self, word: str) -> str:
        """Porter Stemmer implementation for morphological reduction"""
        # Implementation of stemming algorithm...
    
    # ADVANCED SPELLING CORRECTION ----------------------------------------------
    
    def weighted_edit_distance(self, a: str, b: str) -> float:
        """Keyboard-aware weighted edit distance"""
        # Implementation with dynamic programming and keyboard cost matrix...
    
    def phonetic_candidates(self, word: str) -> List[str]:
        """Get phonetically similar candidates"""
        return list(self.phonetic_index.get(self._soundex(word), set()) |
                    self.phonetic_index.get(self._metaphone(word), set()))
    
    # SEMANTIC ANALYSIS ---------------------------------------------------------
    
    def semantic_similarity(self, word1: str, word2: str) -> float:
        """Vector space similarity using co-occurrence statistics"""
        vec1 = self._word_vector(word1)
        vec2 = self._word_vector(word2)
        return self._cosine_similarity(vec1, vec2)
    
    def _word_vector(self, word: str) -> Dict[str, int]:
        """Build co-occurrence vector for a word"""
        # Implementation using n-gram model...
    
    def _cosine_similarity(self, vec1: Dict, vec2: Dict) -> float:
        """Calculate cosine similarity between vectors"""
        # Mathematical implementation...
    
    # LANGUAGE MODELING ---------------------------------------------------------
    
    def word_probability(self, word: str) -> float:
        """Calculate relative frequency probability"""
        total_words = self.metadata.get('word_count', 1)
        return self.data[word].get('frequency', 1) / total_words
    
    def context_suggestions(self, previous_words: List[str], limit: int = 5) -> List[str]:
        """Predict next word using n-gram model"""
        # Implementation using n-gram probabilities...
    
    # SYLLABLE ANALYSIS ---------------------------------------------------------
    
    def syllable_count(self, word: str) -> int:
        """Mathematical syllable estimation algorithm"""
        # Implementation based on vowel counting and exceptions...
    
    # CACHE MANAGEMENT ----------------------------------------------------------
    
    def query(self, word: str) -> Optional[Dict]:
        """Intelligent caching with combined LRU/LFU strategy"""
        word = word.lower()
        
        if word in self.lru_cache:
            self.lru_cache.move_to_end(word)
            self.lfu_cache[word] += 1
            return self.lru_cache[word]
        
        entry = self.data.get(word)
        if entry:
            self._update_cache(word, entry)
        
        return entry
    
    def _update_cache(self, word: str, entry: Dict) -> None:
        """Hybrid cache update strategy"""
        # Combined LRU/LFU eviction logic...
    
    # GRAPH-BASED RELATIONSHIPS -------------------------------------------------
    
    def build_synonym_graph(self) -> None:
        """Construct synonym relationship graph"""
        self.graph = defaultdict(set)
        for word, entry in self.data.items():
            for syn in entry['synonyms']:
                self.graph[word].add(syn.lower())
                self.graph[syn.lower()].add(word)
    
    def synonym_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find shortest path through synonym relationships"""
        # BFS implementation for graph traversal...
    
    # VALIDATION AND ERROR HANDLING ---------------------------------------------
    
    def validate_word(self, word: str) -> bool:
        """Comprehensive word validation"""
        return (
            self._check_orthography(word) and
            self._check_morphology(word) and
            self._check_phonotactics(word)
        )
    
    def _check_phonotactics(self, word: str) -> bool:
        """Validate word structure against language phonotactic rules"""
        # Implementation of phonotactic constraints...

class NLUEngine:
    """Rule-based semantic parser with fallback patterns"""
    def __init__(self, wordlist_path: str = "learning/wordlist_en.json"):
        self.intent_patterns = {
            'information_request': [
                r'(what|where|when|how)\s+is\s+the',
                r'explain\s+.*',
            ],
            'action_request': [
                r'(please|kindly)\s+(run|execute|perform)',
                r'^(start|stop)\s+',
            ],
            'clarification': [
                r'^do\s+you\s+mean',
                r'^(wait|hold\s+on)',
            ]
        }
        
        self.entity_patterns = {
            'date': r'\b(\d{4}-\d{2}-\d{2}|tomorrow|today)\b',
            'url': r'https?://\S+',
            'number': r'\b\d+\b'
        }
        self.wordlist = Wordlist(wordlist_path)

    def parse(self, text: str) -> LinguisticFrame:
        """Hybrid parsing using rules and simple statistics"""
        frame = LinguisticFrame(
            intent='unknown',
            entities={},
            sentiment=self._calculate_sentiment(text),
            modality='statement',
            confidence=0.0
        )

        # Intent detection
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    frame.intent = intent
                    frame.confidence += 0.3  # Simple confidence scoring

        # Entity extraction
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[entity_type] = matches
        frame.entities = entities

        self._validate_words(frame)
        return frame

    def _validate_words(self, frame: LinguisticFrame) -> None:
        """Check if recognized entities exist in the wordlist"""
        valid_entities = {}
        for entity_type, values in frame.entities.items():
            valid = [v for v in values if self.wordlist.query(v)]
            if valid:
                valid_entities[entity_type] = valid
        frame.entities = valid_entities

    def _calculate_sentiment(self, text: str) -> float:
        """Basic sentiment analysis using lexicon approach"""
        positive = len(re.findall(r'\b(good|great|excellent)\b', text, re.IGNORECASE))
        negative = len(re.findall(r'\b(bad|terrible|horrible)\b', text, re.IGNORECASE))
        return (positive - negative) / max(len(text.split()), 1)  # Prevent division by zero

# --------------------------
# Enhanced NLU Components
# --------------------------
class EnhancedNLU(NLUEngine):
    """Implements advanced NLU techniques from recent research"""
    def __init__(self, wordlist_path: str):
        super().__init__(wordlist_path)
        self.coref_resolver = CoreferenceResolver()
        self.dependency_parser = ShallowDependencyParser()
        
        # Add psycholinguistic features
        self.lexical_diversity = lex()
        self.readability = textstat()

    def parse(self, text: str) -> LinguisticFrame:
        """Enhanced parsing pipeline with multiple processing stages"""
        # Stage 0: Coreference resolution
        resolved_text = self.coref_resolver.resolve(text, self.context.history)
        
        # Stage 1: Dependency parsing
        parse_tree = self.dependency_parser.parse(resolved_text)
        
        # Stage 2: Enhanced sentiment analysis
        frame = super().parse(resolved_text)
        frame.sentiment = self._enhanced_sentiment(resolved_text, parse_tree)
        
        # Stage 3: Pragmatic analysis
        frame.modality = self._detect_modality(parse_tree)
        
        return frame

    def _enhanced_sentiment(self, text: str, parse_tree: Dict) -> float:
        """Sentiment analysis considering negation and intensity"""
        # Implementation based on Socher et al. (2013) Recursive Neural Networks
        sentiment = 0.0
        for node in parse_tree['nodes']:
            if node['relation'] == 'neg':
                sentiment -= 0.5
            else:
                word_sentiment = self.wordlist.query(node['word']).get('sentiment', 0)
                sentiment += word_sentiment * node['intensity']
        return tanh(sentiment)  # Squash to [-1, 1]

    def _detect_modality(self, parse_tree: Dict) -> str:
        """Detect speech modality using verb patterns"""
        # Inspired by Austin's Speech Act Theory (1975)
        main_verb = parse_tree['root']['lemma']
        modality_map = {
            'ask': 'query',
            'request': 'command',
            'suggest': 'proposal'
        }
        return modality_map.get(main_verb, 'statement')

class CoreferenceResolver:
    """Simple rule-based coreference resolution"""
    def resolve(self, text: str, history: deque) -> str:
        """Replace pronouns with recent entities"""
        # Implementation inspired by Hobbs algorithm (1978)
        last_entities = self._extract_last_entities(history)
        return re.sub(r'\b(he|she|it|they)\b', lambda m: last_entities.get(m.group().lower(), m.group()), text)

class ShallowDependencyParser:
    """Lightweight dependency parser using regex patterns"""
    def parse(self, text: str) -> Dict:
        """Extract basic dependency relations"""
        # Simplified version of Marneffe et al. (2014) Universal Dependencies
        patterns = {
            'nsubj': r'(\w+)\s+is',  # Simplified subject detection
            'dobj': r'(\w+)\s+',      # Direct object placeholder
        }
        return {'nodes': [...]}  # Simplified output

# --------------------------
# Enhanced NLG Components
# --------------------------
class NLGEngine:
    """Controlled text generation with style management"""
    def __init__(self, templates_path: str = "learning/nlg_templates.json"):
        self.templates = self._load_templates(templates_path)
        self.style = {'formality': 0.5, 'verbosity': 1.0}
        self.coherence_checker = ResponseCoherence()

    def generate(self, frame: LinguisticFrame, context: DialogueContext) -> str:
        """Generate response using hybrid template-neural approach"""
        # Step 1: Template selection
        if template := self._match_template(frame):
            response = self._instantiate_template(template, context)
        else:
            response = self._neural_generation(frame, context)
            
        # Step 2: Style adaptation
        response = self._adapt_style(response)
        
        # Step 3: Coherence check
        if not self.coherence_checker.validate(response, context):
            response = self._fallback_generation(frame)
            
        return response

    def _match_template(self, frame: LinguisticFrame) -> Optional[str]:
        """Match against handcrafted templates for common intents"""
        template_map = {
            'information_request': "The {entity} is {value}",
            'action_request': "Executing command: {command}"
        }
        return template_map.get(frame.intent)

    def _neural_generation(self, frame: LinguisticFrame, context: DialogueContext) -> str:
        """Generate using LLM with controlled prompting"""
        prompt = f"Generate response with intent {frame.intent} and entities {frame.entities}"
        return self.llm.generate(prompt)

    def _adapt_style(self, text: str) -> str:
        """Adjust formality and verbosity"""
        if self.style['formality'] > 0.7:
            text = re.sub(r"\b(can't|don't)\b", "cannot do not", text)
        if self.style['verbosity'] < 0.5:
            text = ' '.join(text.split()[:15]) + '...'
        return text

class ResponseCoherence:
    """Ensure generated responses stay on-topic"""
    def validate(self, response: str, context: DialogueContext) -> bool:
        """Check lexical overlap with conversation history"""
        # Implementation inspired by Centering Theory (Grosz et al. 1995)
        history_words = set(word for turn in context.history for word in turn[0].split())
        response_words = set(response.split())
        return len(history_words & response_words) / len(response_words) > 0.3

class KnowledgeCache:
    """Lightweight cache with LRU eviction policy"""
    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def hash_query(self, query: str) -> str:
        """Semantic hashing for similar queries (simplified)"""
        return hashlib.md5(query.encode()).hexdigest()

class SafetyGuard:
    """Multi-layered content safety system"""
    def __init__(self):
        self.redact_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]'),
            (r'(?i)\b(credit card|password)\b', '[REDACTED_PII]')
        ]
        
        self.toxicity_patterns = [
            r'\b(kill|harm|attack)\b',
            r'(racial|ethnic)\s+slur'
        ]

    def sanitize(self, text: str) -> str:
        """Apply redaction and toxicity filtering"""
        # Redaction layer
        for pattern, replacement in self.redact_patterns:
            text = re.sub(pattern, replacement, text)
            
        # Toxicity check
        for pattern in self.toxicity_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "[SAFETY_BLOCK] Content violates safety policy"
            
        return text

class LanguageAgent:
    def __init__(self, llm, config: Dict):
        self.llm = llm
        self.nlu = NLUEngine()
        self.cache = KnowledgeCache(max_size=config.get('cache_size', 1000))
        self.safety = SafetyGuard()
        self.context = DialogueContext()
        self.dialogue_policy = self._load_dialogue_policy()
        self.wordlist = Wordlist(config.get('wordlist_path'))
        self.nlu = NLUEngine(config.get('wordlist_path'))
        self.nlg = NLGEngine(config.get('nlg_templates'))

    def preprocess_input(self, text: str) -> str:
        words = text.split()
        corrected = [w if self.wordlist.query(w) else self._guess_spelling(w) 
                     for w in words]
        return " ".join(corrected)

    def process_input(self, user_input: str) -> Tuple[str, LinguisticFrame]:
        """Full processing pipeline with academic-inspired components"""
        # Stage 1: Input sanitization
        clean_input = self.safety.sanitize(user_input)
        frame = self.nlu.parse(clean_input)
        
        # Stage 2: Semantic parsing
        frame = self.nlu.parse(clean_input)
        
        # Stage 3: Context-aware response generation
        cache_key = self.cache.hash_query(clean_input)
        if cached := self.cache.get(cache_key):
            return safe_response, frame
            
        prompt = self._construct_prompt(clean_input, frame)
        raw_response = self.llm.generate(prompt)
        refined_response = self.nlg.generate(frame, self.context)
        safe_response = self.safety.sanitize(refined_response)
        
        # Stage 4: Context update
        self._update_context(clean_input, safe_response, frame)
        self.cache.set(cache_key, safe_response)

        if cached := self.cache.get(self.cache.hash_query(clean_input)):
            return safe_response, frame

    def expand_query(self, query: str) -> str:
        words = query.split()
        expanded = []
        for word in words:
            details = self.wordlist.query(word)
            if details and 'synonyms' in details:
                expanded.append(f"{word} ({'|'.join(details['synonyms'])})")
            else:
                expanded.append(word)
        return " ".join(expanded)

    def save_context(self, file_path: Union[str, Path]) -> None:
        """
        Save dialogue context using Python's pickle serialization.
        Security Note: Pickle can execute arbitrary code. Only load trusted files.
    
        Args:
            file_path: Path to save context (supports both str and pathlib.Path).
    
        Raises:
            PermissionError: On write permission issues.
            pickle.PicklingError: If serialization fails.

        Example:
            >>> agent.save_context("chat_context.pkl")
            >>> agent.save_context(Path("/data/context.pkl"))

        Reference:
            Python Software Foundation. (2023). `pickle` module documentation.
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.context, f)
        except (PermissionError, IsADirectoryError) as e:
            raise PermissionError(f"Failed to save context: {str(e)}") from e
        except (pickle.PicklingError, AttributeError) as e:
            raise pickle.PicklingError(f"Serialization error: {str(e)}") from e

    def load_context(self, file_path: Union[str, Path]) -> None:
        """
        Load dialogue context from a pickle file.
        
        Args:
            file_path: Path to the saved context file.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            pickle.UnpicklingError: If the file is corrupted.
        
        Example:
            >>> agent.load_context("chat_context.pkl")
        """
        try:
            with open(file_path, 'rb') as f:
                self.context = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Context file not found: {file_path}")
        except pickle.UnpicklingError:
            raise pickle.UnpicklingError(f"Failed to load context from {file_path}")

    def _validate_llm(self) -> None:
        """Ensure the LLM has the required `generate` method."""
        if not hasattr(self.llm, 'generate') or not callable(self.llm.generate):
            raise AttributeError("LLM must implement a 'generate' method.")

    def _construct_prompt(self, text: str, frame: LinguisticFrame) -> str:
        """Build prompt using T5-style text-to-text approach"""
        components = [
            f"User: {text}",
            f"Intent: {frame.intent}",
            f"Context: {json.dumps(self.context.environment_state)}",
            "History:"
        ]
        
        for idx, (user, bot) in enumerate(self.context.history):
            components.append(f"Turn {idx+1}:")
            components.append(f"User: {user}")
            components.append(f"Bot: {bot}")
            
        components.append("Assistant Response:")
        return "\n".join(components)

    def _load_dialogue_policy(self) -> Dict:
        """Load conversation rules from embedded config"""
        return {
            'clarification_triggers': ['unknown', 'low_confidence'],
            'reprompt_limit': 2,
            'fallback_responses': [
                "Could you rephrase that?",
                "I need more context to help effectively."
            ]
        }

    def validate_response(self, response: str) -> bool:
        """
        Check if the LLM response violates safety constraints (e.g., toxicity, PII).
        
        Args:
            response: LLM-generated text.
        
        Returns:
            True if safe, False if unsafe.
        
        Notes:
            Extend with ML-based classifiers (e.g., Hugging Face's `detoxify`) for production.
        """
        unsafe_patterns = [
            r"(?i)\b(kill|harm|hurt|attack|hate)\b",
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN-like patterns
            r"(?i)\b(password|credit\s*card|social\s*security)\b",
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, response):
                self.benchmark_data["safety_violations"] += 1
                return False
        return True

    def update_context(self, user_input: str, llm_response: str) -> None:
        """Update dialogue history and environment state."""
        self.context.history.append((user_input, llm_response))

    def evaluate_parsing_accuracy(self, test_cases: List[Tuple[str, Dict]]) -> float:
        """
        Benchmark parsing accuracy against labeled test cases.
        
        Args:
            test_cases: List of (input_text, expected_parsed_output).
        
        Returns:
            Accuracy score (0.0 to 1.0).
        """
        correct = 0
        for text, expected in test_cases:
            parsed = self.translate_user_input(text)
            if parsed == expected:
                correct += 1
            self.benchmark_data["parsing_accuracy"].append(parsed == expected)
        return correct / len(test_cases)

    def generate_prompt(self, user_input: str) -> str:
        """Generate a prompt with context and history."""
        prompt = f"User: {user_input}\nContext: {json.dumps(self.context.environment_state)}\n"
        if self.context.history:
            prompt += "Dialogue History:\n" + "\n".join([f"User: {u}\nBot: {r}" for u, r in self.context.history])
        prompt += "\nAssistant:"
        return prompt

    def process_input(self, user_input: str) -> Tuple[str, Dict]:
        """
        End-to-end processing pipeline.
        
        Returns:
            Tuple of (LLM response, structured command).
        
        Raises:
            RuntimeError: If LLM fails or response is unsafe.
        """
        start_time = time.time()
        
        # Step 1: Generate prompt
        prompt = self.generate_prompt(user_input)
        
        # Step 2: Get LLM response
        llm_response = self.interface_llm(prompt)
        
        # Step 3: Validate safety
        if not self.validate_response(llm_response):
            llm_response = "[SAFETY FILTER] I cannot comply with this request."
        
        # Step 4: Parse and update context
        structured_input = self.translate_user_input(user_input)
        self.update_context(user_input, llm_response)
        
        # Step 5: Record performance
        self.benchmark_data["response_time"].append(time.time() - start_time)
        
        return llm_response, structured_input

    def interface_llm(self, prompt: str) -> str:
        """Robust LLM interface with retries and timeout."""
        for attempt in range(self.max_retries):
            try:
                response = self.llm.generate(prompt, timeout=self.timeout)
                return response.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM failed after {self.max_retries} attempts: {str(e)}")
                continue

    def translate_user_input(self, user_input: str) -> Dict:
        """Parse user input into structured commands."""
        patterns = {
            'search': r'(?:search|find)\s+(?P<query>.+?)(?:\s+(?:about|for)\s+(?P<topic>.+))?',
            'create': r'(?:create|make)\s+(?P<object>\w+)(?:\s+with\s+(?P<params>.+))?',
            'default': r'(?P<command>\w+)(?:\s+(?P<args>.+))?'
        }

        for intent, pattern in patterns.items():
            match = re.match(pattern, user_input.strip(), re.IGNORECASE)
            if match:
                groups = match.groupdict()
                return {
                    'intent': intent,
                    'entities': {k: v for k, v in groups.items() if v},
                    'args': groups.get('args', '').split() if 'args' in groups else []
                }
        return {'intent': 'unknown', 'entities': {}, 'args': []}

# --------------------------
# Utility Functions
# --------------------------
def load_config(config_path: Union[str, Path]) -> Dict:
    """Load configuration with integrity checking"""
    # Implementation omitted for brevity
    return {}

def initialize_agent(llm, config_path: str = "config.json") -> LanguageAgent:
    config = load_config(config_path)
    return LanguageAgent(llm, config)
