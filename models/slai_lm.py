"""
SLAILM is the Base Language Model of SLAI
"""

import os
import sys
import random
import re
import json
import time
import warnings
import logging
import datetime
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, Dict, Any, Union, Set, Tuple

from src.agents.language_agent import DialogueContext
from src.agents.language.grammar_processor import GrammarProcessor
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.language.resource_loader import ResourceLoader

logging.basicConfig(level=logging.WARNING)

shared_slailm = None

def get_shared_slailm(shared_memory, agent_factory=None):
    global shared_slailm
    if shared_slailm is None:
        shared_slailm = SLAILM(shared_memory, agent_factory=agent_factory)
    return shared_slailm

class SLAILM:
    """
    Self-Contained Language and Interaction Logic Module that aims to minimize external dependencies, relying on
    custom GrammarProcessor, wordlists, and context management.
    """
    def __init__(self, shared_memory,
                 agent_factory=None,
                 structured_wordlist_path="src/agents/language/structured_wordlist_en.json",
                 simple_wordlist_path="src/agents/language/wordlist_en.json",
                 grammar_rules_path: Optional[Union[str, Path]] = None,
                 knowledge_agent_path: Optional[Union[str, Path]] = None,

                 # --- Component Instances (for dependency injection) ---
                 grammar_processor_instance: Optional[GrammarProcessor] = None,
                 dialogue_context_instance: Optional[DialogueContext] = None,
                 knowledge_agent_instance: Optional[KnowledgeAgent] = None,

                 # --- Operational Configurations ---
                 node_id: Optional[str] = None,
                 log_level: int = logging.INFO,
                 log_file: Optional[Union[str, Path]] = None,
                 context_memory_limit: int = 10, # Max items for simple context deque
                 dialogue_history_limit: int = 100,
                 enable_summarization: bool = False,

                 # --- Custom Configuration Dictionary ---
                 custom_config: Optional[Dict[str, Any]] = None
                 ):
        start = time.time()
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self._knowledge_agent = None
        self.conversation_history = []
        self.sentiment_lexicon = self.load_sentiment_lexicon()
        """
        Initializes the SLAILM with broader configuration options.

        Args:
            structured_wordlist_path: Path to the JSON structured wordlist file.
            simple_wordlist_path: Path to the simple wordlist file (one word per line).
            grammar_rules_path: Optional path to load additional grammar rules for GrammarProcessor.
            knowledge_agent_path: Optional path to load data for the KnowledgeAgent.
            grammar_processor_instance: Optional pre-initialized GrammarProcessor instance.
            dialogue_context_instance: Optional pre-initialized DialogueContext instance.
            knowledge_agent_instance: Optional pre-initialized KnowledgeAgent instance.
            node_id: Optional unique identifier for this instance.
            log_level: Logging level (e.g., logging.DEBUG, logging.INFO).
            log_file: Optional file path to write logs to.
            context_memory_limit: Max size for the internal context_memory deque.
            dialogue_history_limit: Max history size for the main DialogueContext.
            enable_summarization: Whether DialogueContext should summarize long histories.
            custom_config: Dictionary for any other custom configurations.
        """
        self.node_id = node_id or hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
        self._setup_logging(log_level, log_file)
        logging.info(f"Initializing SLAILM instance {self.node_id}...")
        self.sentiment_lexicon = {
            "positive": {}, "negative": {},
            "intensifiers": {}, "negators": []
        }
        try:
            with open("src/agents/language/sentiment_lexicon.json", "r") as f:
                self.sentiment_lexicon = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load sentiment lexicon: {e}")

        self.custom_config = custom_config or {}
        self.context_memory = deque(maxlen=context_memory_limit)

        # --- Load Resources (Wordlist) ---
        self.structured_wordlist = ResourceLoader.get_structured_wordlist(structured_wordlist_path)
        self.wordlist = ResourceLoader.get_simple_wordlist(simple_wordlist_path)
        self.sentiment_lexicon = ResourceLoader.get_sentiment_lexicon()
        self.responses = ResourceLoader.get_nlg_templates()

        # --- Initialize Components (using instances or loading) ---
        # Knowledge Agent (Initialize first if other components depend on it)
        if knowledge_agent_instance:
            self.knowledge = knowledge_agent_instance
            logging.info("Using provided KnowledgeAgent instance.")
        else:
            # Initialize KB, potentially loading from path
            try:
                self.knowledge = KnowledgeAgent(
                    shared_memory=self.shared_memory,
                    agent_factory=agent_factory
                    )
                logging.info(f"Initialized KnowledgeAgent (path: {knowledge_agent_path}).")
            except Exception as e:
                logging.error(f"Failed to initialize KnowledgeAgent: {e}")
                self.knowledge = None # Fallback

        # Grammar Processor
        if grammar_processor_instance:
            self.grammar_processor = grammar_processor_instance
            logging.info("Using provided GrammarProcessor instance.")
        else:
            # Initialize GrammarProcessor, passing loaded resources and config
            # NOTE: GrammarProcessor.__init__ needs to be adapted to accept these args
            try:
                self.grammar_processor = GrammarProcessor(
                    structured_wordlist=self.structured_wordlist,
                    wordlist=self.wordlist,
                    rules_path=grammar_rules_path, # Pass rules path if needed
                    knowledge_agent=self.knowledge # Pass KB if needed
                    # Add other relevant configs from self.custom_config if necessary
                )
                logging.info("Initialized GrammarProcessor.")
            except Exception as e:
                logging.error(f"Failed to initialize GrammarProcessor: {e}")
                # Decide on fallback: raise error or use a dummy processor?
                # For now, setting to None and checking later might be safer
                self.grammar_processor = None
                # raise RuntimeError("Critical component GrammarProcessor failed to initialize.") from e

        # Dialogue Context
        if dialogue_context_instance:
            self.dialogue_context = dialogue_context_instance
            logging.info("Using provided DialogueContext instance.")
        else:
            # Initialize DialogueContext, passing config
            # NOTE: DialogueContext.__init__ needs to accept these args
            try:
                self.dialogue_context = DialogueContext(
                    llm=self, # Pass self if context needs to call back to LM (use carefully)
                    history=[], # Start with empty history
                    memory_limit=dialogue_history_limit,
                    enable_summarization=enable_summarization
                    # Pass summarizer function if needed/available
                )
                logging.info("Initialized DialogueContext.")
            except Exception as e:
                logging.error(f"Failed to initialize DialogueContext: {e}")
                self.dialogue_context = None # Fallback

        # self.responses = {}

        self.custom_config = custom_config or {}
        self.context_memory = deque(maxlen=context_memory_limit)

        # --- Final Checks and Setup ---
        if self.grammar_processor is None or self.dialogue_context is None:
            logging.critical("One or more critical components (GrammarProcessor, DialogueContext) failed to initialize. SLAILM may not function correctly.")
            raise RuntimeError("Failed to initialize critical SLAILM components.")

        # Predefined Responses (can be loaded from config too)
        self.responses = self.custom_config.get("responses", {
            "default": [
                "I am processing your input using my internal linguistic rules.",
                "Let me analyze that based on my grammar model.",
                "That's an interesting point. Let me construct a response.",
            ]
        })

        logging.info(f"[SLAILM INIT] Finished in {time.time() - start:.2f}s")

    def _setup_logging(self, level: int, log_file: Optional[Union[str, Path]]):
        """Configures logging for the SLAILM instance."""
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        logger = logging.getLogger(f"SLAILM_{self.node_id}") # Logger specific to this instance
        logger.setLevel(level)

        # Prevent adding multiple handlers if re-initialized
        if not logger.handlers:
            # Console Handler
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(log_formatter)
            logger.addHandler(stdout_handler)

            # File Handler (Optional)
            if log_file:
                try:
                    file_handler = logging.FileHandler(log_file, encoding='utf-8')
                    file_handler.setFormatter(log_formatter)
                    logger.addHandler(file_handler)
                    logger.info(f"Logging to file: {log_file}")
                except Exception as e:
                    logger.error(f"Failed to set up log file at {log_file}: {e}")
        logging.getLogger(f"SLAILM_{self.node_id}").info(...)
        # Or assign self.logger = logger

    def _load_json_resource(self, file_path: Union[str, Path], resource_name: str) -> Dict:
        """Loads a JSON resource file with error handling."""
        path = Path(file_path)
        data = {}
        if not path.is_file():
            logging.error(f"{resource_name} file not found at {path}")
            return data # Return empty dict

        try:
            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Successfully loaded {resource_name} from {path}")
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from {resource_name} file: {path}")
        except Exception as e:
            logging.error(f"An error occurred loading {resource_name} from {path}: {e}")

        return data

    def _load_simple_wordlist(self, file_path: Union[str, Path]) -> Set[str]:
        """Loads a simple wordlist (one word per line) into a set."""
        path = Path(file_path)
        word_set = set()
        if not path.is_file():
            logging.error(f"Simple wordlist file not found at {path}")
            return word_set # Return empty set

        try:
            with path.open('r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        word_set.add(word)
            logging.info(f"Successfully loaded {len(word_set)} words from Simple Wordlist: {path}")
        except Exception as e:
            logging.error(f"An error occurred loading simple wordlist from {path}: {e}")

        return word_set
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenization using GrammarProcessor's POS patterns and linguistic rules."""
        for i, pattern_item in enumerate(pos_patterns):
            try:
                pattern, name = pattern_item
            except (ValueError, TypeError) as e:
                logging.error(f"Invalid POS pattern format: {pattern_item}")
                continue

        if self.grammar_processor and hasattr(self.grammar_processor, 'pos_patterns'):
            tokens = []
            pos_patterns = sorted(
                [(p, name) for name, p in self.grammar_processor.pos_patterns.items()],  # Convert dict items to tuples
                key=lambda x: len(x[0].pattern),
                reverse=True  # Match longer patterns first
            )
            
            # Build combined regex pattern with UNIQUE group names
            combined_pattern = '|'.join(
                f'(?P<{name}_{i}>{pattern.pattern})'  # Unique group names: NOUN_0, VERB_1, etc.
                for i, (pattern, name) in enumerate(pos_patterns)
            )
            combined_re = re.compile(combined_pattern, re.IGNORECASE)

            # Scan text using POS-aware tokenization
            pos = 0
            while pos < len(text):
                match = combined_re.match(text, pos)
                if match:
                    token = match.group()
                    # Determine which group matched and extract the POS tag
                    for i, (pattern, name) in enumerate(pos_patterns):
                        group_name = f"{name}_{i}"
                        if match.group(group_name):  # Check if this group matched
                            pos_tag = name
                            break
                    tokens.append(token)
                    pos = match.end()
                else:
                    # Fallback: Split unknown characters
                    tokens.append(text[pos])
                    pos += 1
            
            # Post-process hyphenated words and contractions
            refined_tokens = []
            for token in tokens:
                if '-' in token and token not in self.structured_wordlist:
                    refined_tokens.extend(token.split('-'))
                elif "'" in token:
                    parts = re.split(r"('(?:t|s|d|ll|re|ve|m))$", token)
                    refined_tokens.extend(filter(None, parts))
                else:
                    refined_tokens.append(token)
            return refined_tokens

        # Fallback to enhanced regex-based tokenization
        text = re.sub(r"([^'\w\s-]|(?<!\d)\.(?!\d))", r' \1 ', text)  # Handle punctuation
        tokens = re.findall(
            r"\w+(?:[-']\w+)*|['\".,!?;:()\-–—/]",  # Match words, contractions, punctuation
            text,
            re.UNICODE
        )
        
        # Validate against wordlist and morphology rules
        validated = []
        for token in tokens:
            if (token not in self.wordlist and 
                not self.grammar_processor._guess_pos_by_morphology(token)):
                stems = self.grammar_processor.stemmer.stem(token)
                if '-' in stems:
                    validated.extend(stems.split('-'))
                else:
                    validated.append(token)
            else:
                validated.append(token)
        return validated
    
    def parse_intent(self, prompt: str) -> dict:
        try:
            # ... your logic here ...
            return {"type": detected_type, "confidence": score}
        except Exception as e:
            logging.warning(f"[SLAILM] parse_intent failed: {e}")

            return {"type": "unknown", "confidence": 0.0}
    
    def process_input(self, prompt, text: str) -> dict:
        """
        Processes input text with advanced linguistic steps: tokenization, POS tagging,
        intent recognition, entity extraction, sentiment scoring, and concept identification.
        """

        if not isinstance(text, str) or not text.strip():
            return {"error": "Input must be a non-empty string."}

        logging.debug(f"[SLAILM] Processing input: {text}")
        tokens = self._tokenize(text)
        analysis = {
            "raw_text": text,
            "tokens": tokens,
            "timestamp": time.time()
        }

        # POS tagging
        try:
            pos_tags = self.grammar_processor._pos_tag(text)
            analysis["pos_tags"] = pos_tags
        except Exception as e:
            logging.error(f"POS tagging failed: {e}")
            pos_tags = [(token, "UNK") for token in tokens]
            analysis["pos_tags"] = pos_tags

        # Extract concepts based on Noun/Proper Noun
        analysis["concepts"] = [
            word for word, tag in pos_tags if tag in ("NOUN", "PROPN")
        ]

        # Intent recognition using regex patterns or Wordlist
        try:
            intent = self.grammar_processor.detect_intent(text)
            #analysis["intent"] = intent if isinstance(intent, dict) else {"type": "unknown", "confidence": 0.0}
            if not isinstance(intent, dict):
                intent = {"type": "unknown", "confidence": 0.0}
        except Exception as e:
            logging.warning(f"Intent recognition failed: {e}")
            analysis["intent"] = "unknown"
            

        # If question detected by NLU:
        if intent.get("type") == "question":
            query = prompt.strip()
            try:
                results = self.knowledge.retrieve(query) if self.knowledge else []
                context = "\n".join(results[:2]) if results else ""
                return self.generate_response(f"Q: {query}\nContext: {context}\nA:")
            except Exception as e:
                logging.warning(f"[SLAILM] KnowledgeAgent retrieval failed: {e}")
                return random.choice(self.responses["default"])

        # Entity recognition via pattern or POS chunks
        try:
            entities = self.grammar_processor.extract_entities(text, pos_tags)
            analysis["entities"] = entities
        except Exception as e:
            logging.warning(f"Entity extraction failed: {e}")
            analysis["entities"] = {}

        # Enhanced Sentiment Scoring
            valence_dict = {**self.sentiment_lexicon["positive"], **self.sentiment_lexicon["negative"]}
            intensifiers = self.sentiment_lexicon.get("intensifiers", {})
            negators = set(self.sentiment_lexicon.get("negators", []))

            tokens = self._tokenize(text)
            sentiment_score = 0
            weight_sum = 0
            negate = False
            intensity = 1.0

            for word in tokens:
                w = word.lower()
                if w in negators:
                    negate = True
                    continue
                elif w in intensifiers:
                    intensity *= intensifiers[w]
                    continue

                if w in valence_dict:
                    score = valence_dict[w]
                    if negate:
                        score *= -1
                        negate = False
                    sentiment_score += score * intensity
                    weight_sum += intensity
                    intensity = 1.0

            sentiment = sentiment_score / weight_sum if weight_sum else 0.0

            analysis["sentiment"] = round(max(-1.0, min(1.0, sentiment)), 3)

        except Exception as e:
            logging.warning(f"Sentiment analysis failed: {e}")
            analysis["sentiment"] = 0.0

        # Update context
        if self.dialogue_context:
            self.dialogue_context.add(text)

        self.context_memory.append(analysis)
        return analysis

    def load_sentiment_lexicon(self, path: str = "src/agents/language/sentiment_lexicon.json") -> dict:
        """
        Loads a structured sentiment lexicon for valence scoring.
        Returns a dictionary with 'positive', 'negative', 'intensifiers', and 'negators'.
        """
    
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            required_keys = {"positive", "negative", "intensifiers", "negators"}
            if not required_keys.issubset(data.keys()):
                raise ValueError(f"Sentiment lexicon missing required keys: {required_keys - set(data.keys())}")
            
            return data

        except Exception as e:
            logging.warning(f"[SLAILM] Failed to load sentiment lexicon: {e}")
            return {
                "positive": {},
                "negative": {},
                "intensifiers": {},
                "negators": []
            }

    def forward_pass(self, processed_input: dict) -> str:
        """
        Generates a response based on the processed input using GrammarProcessor.
        Replaces the GPT-2 based generation.
        """
        input_text = processed_input.get("raw_text", "")
        concepts = processed_input.get("concepts", [])
        logging.debug(f"Generating response for concepts: {concepts}")

        # Option 1: Use GrammarProcessor's constrained generation
        try:
            # Create a dummy frame or pass relevant info if needed by _generate_grammatical_response
            # The original method expected 'frame' and 'input_text'
            # We might need to adapt GrammarProcessor or how we call it.
            # Passing concepts as potential seed words.
            # NOTE: The signature and requirements of _generate_grammatical_response
            # need to match what's available here.
            # Let's assume it can work with just the input text for now,
            # or adapt it if necessary.
            # frame_stub = type('Frame', (object,), {'entities': {}}) # Minimal stub if needed
            # generated_text = self.grammar_processor._generate_grammatical_response(frame_stub, input_text)

            # Simpler approach: Maybe GrammarProcessor has a direct generate method?
            # Or we use a simpler template + concept approach if grammar generation is complex.

            if hasattr(self.grammar_processor, 'grammar') and hasattr(self.grammar_processor.grammar, 'generate_sentence'):
                seed_words = processed_input.get("tokens", []) + concepts
                for _ in range(3): # Try a few times
                     generated = self.grammar_processor.grammar.generate_sentence(seed_words)
                     # Check basic validity (non-empty)
                     if generated and isinstance(generated, str) and generated.strip():
                         # Optional: Validate with parse_grammar if available
                         if hasattr(self.grammar_processor, 'parse_grammar') and self.grammar_processor.parse_grammar(generated):
                             return generated.capitalize()
                         elif not hasattr(self.grammar_processor, 'parse_grammar'):
                             return generated.capitalize() # Return if no parser available
                logging.warning("Grammatical generation failed after retries.")
                generated_text = None
            else:
                logging.warning("Grammar generation method not found.")
                generated_text = None

        except Exception as e:
            logging.error(f"Error during grammatical response generation: {e}")
            generated_text = None

        # Fallback Strategy: Conceptual response or template
        if not generated_text:
            if concepts and self.knowledge:
                 generated_text = self._conceptual_response(concepts[0] if concepts else "information")
            else:
                 generated_text = random.choice(self.responses["default"]) + f" Your input mentioned: {', '.join(concepts)}."

        # Post-processing (optional)
        generated_text = self._refine_generated_text(generated_text)
        return {
            "text": generated_text,
            "raw": generated_text,
            "intent": "response",
            "confidence": 1.0,
            "meta": {
                "source": "SLAILM",
                "id": self.instance_id
            }
        }
    
    def summarize_text(self, input_text: str) -> str:
        summarization_prompt = f"Summarize the following text clearly and concisely:\n\n{input_text}\n\nSummary:"
        return self.generate_response(summarization_prompt)

    def _refine_generated_text(self, text: str) -> str:
        """
        Cleans and enhances generated text for grammatical, stylistic, and contextual quality.
        """
        if not text or not isinstance(text, str):
            return ""

        # Strip unnecessary whitespace
        text = text.strip()

        # Fix sentence termination
        if text and text[-1] not in {'.', '?', '!'}:
            text += '.'

        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        # Remove repeated punctuation
        text = re.sub(r'([.?!])\1+', r'\1', text)

        # Collapse excessive spacing
        text = re.sub(r'\s{2,}', ' ', text)

        # Normalize quote characters
        text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")

        # Censor potentially offensive words (optional)
        if self.custom_config.get("safe_mode", True):
            censor_words = {"damn", "fuck", "shit",}  # Expand as needed
            for word in censor_words:
                pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
                text = pattern.sub("*" * len(word), text)

        # Apply grammar-aware polishing (if enabled)
        if self.grammar_processor and hasattr(self.grammar_processor, "polish_response"):
            try:
                text = self.grammar_processor.polish_response(text)
            except Exception as e:
                logging.warning(f"Failed to polish text via grammar processor: {e}")

        return text

    def _conceptual_response(self, concept: str) -> str:
        """Generates a response based on knowledge about a concept (Placeholder)."""
        # This method originally used N-grams and academic closures.
        if not self.knowledge:
            return f"I understand you're asking about {concept}, but my knowledge agent is currently unavailable."

        # Original N-gram logic (requires self.knowledge structure)
        # Replace with actual access to your KnowledgeAgent data
        ngram_models = {
            "concept_definition": [f"{concept} is generally understood as", f"Regarding {concept}, the primary definition involves"],
            "research_area": [f"In the context of {concept}, current research focuses on", f"Studies involving {concept} often explore"],
            "methodology": ["Key methodologies related to", "Common approaches include"],
            "uncertainty": ["There is ongoing debate about", "Uncertainty measurement through"]
        }
        # Simulated knowledge lookup
        related_topic = random.choice(list(ngram_models.keys()))
        return random.choice(ngram_models.get(related_topic, [f"Considering {concept},"])) + " " + self._academic_closure()


    def _academic_closure(self) -> str:
        """Academic phrase completion using lexical patterns."""
        closures = [
            "significant improvements in model performance.",
            "novel approaches to computational problems.",
            "fundamental breakthroughs in theoretical understanding.",
            "substantial implications for future research directions.",
            "further investigation in this domain.",
            "the development of new frameworks.",
        ]
        return random.choice(closures)

    def validate_fact(self, fact: Tuple, context: Dict) -> float:
        """LLM-based factual validation"""
        prompt = f"Validate this statement (true/false): {fact}. Context: {context}"
        response = self.generate_response(prompt)
        return self._parse_validation_response(response)

    def generate_response(self, prompt: str) -> str:
        """
        Full response generation pipeline using internal components.
        """
        if not isinstance(prompt, str) or len(prompt.strip()) < 3:
            return "[SLAILM] Input too short to generate meaningful output."

        start_time = time.time()
        logging.info(f"Received prompt: {prompt}")

        # Add context from dialogue history (if managed)
        prompt_with_context = self.dialogue_context.generate_prompt(prompt) # If using context manager's prompt feature

        # Process the input using internal tools
        processed = self.process_input(prompt)
        if "error" in processed:
            return f"[SLAILM] Error processing input: {processed['error']}"

        # Generate response using internal logic (forward_pass)
        response_text = self.forward_pass(processed)

        # Update dialogue history
        self.dialogue_context.add_entry(user_input=prompt, bot_response=response_text)
        self.conversation_history.append({"user": prompt, "bot": response_text}) # Keep simple history too

        # Academic formatting
        final_response = f"RESPONSE:\n{response_text}"

        # Citation generation (Requires self.knowledge)
        if self.knowledge:
             citations = self._generate_citations(processed.get("concepts", []))
             if citations:
                 final_response += f"\n\nReferences:\n{citations}"

        end_time = time.time()
        logging.info(f"Generated response in {end_time - start_time:.2f} seconds.")
        return final_response
    
    def polish_response(self, text: str) -> str:
        """
        Optional final polish using rule-based grammar optimization.
        """
        # Example: Avoid "It is..." passive openings
        text = re.sub(r"\bIt is (important|likely|clear|possible)\b", r"This is \1", text)

        # Optional contraction fixups or simplification
        text = text.replace("do not", "don't").replace("cannot", "can't")

        return text

    def _generate_citations(self, concepts: list) -> str:
        """citation implementation"""
        if not self.knowledge:
            return ""
        
        try:
            refs = set()
            for concept in concepts[:3]:  # Limit to top 3
                results = self.knowledge.get_references_for_concepts([concept], k=1)
                if results:
                    refs.add(f"- {concept}: {results[0][:100]}...")
            return "\n".join(refs) if refs else ""
        except Exception as e:
            logging.error(f"Citation generation failed: {e}")
            return ""
            
    def handle_general_prompt(self, prompt: str) -> str:
        return self.generate_response(prompt)

    @property
    def knowledge_agent(self):
        if self._knowledge_agent is None:
            self._knowledge_agent = KnowledgeAgent(
                shared_memory=self.shared_memory,
                agent_factory=self.agent_factory
            )
        return self._knowledge_agent

class SLAILMValueModel:
    def __init__(self, slai_lm, memory=None, ethics_checker=None):
        self.slai_lm = slai_lm
        self.memory = memory  # Injected AlignmentMemory
        self.ethics_checker = ethics_checker  # Injected EthicalConstraints

        self.preference_weights = {
            "helpfulness": 0.4,
            "harmlessness": 0.3,
            "honesty": 0.3
        }

    def score_trajectory(self, data: pd.DataFrame) -> float:
        """Trajectory evaluation using SLAILM + RLHF feedback-aware scoring"""
        scores = []

        for _, row in data.iterrows():
            input_text = row.get("input", "")
            response = row.get("output", "")

            # Use SLAILM for grounded understanding
            result = self.slai_lm.process_input(prompt=input_text, text=response)

            helpfulness = result.get("helpfulness", 0.5)
            harmlessness = 1.0 - result.get("toxicity", 0.5)
            honesty = result.get("factuality", 0.5)

            # Aggregate weighted preference
            composite = (
                helpfulness * self.preference_weights["helpfulness"] +
                harmlessness * self.preference_weights["harmlessness"] +
                honesty * self.preference_weights["honesty"]
            )
            scores.append(composite)

            # Log to memory
            if self.memory:
                self.memory.log_evaluation(
                    metric="value_alignment",
                    value=composite,
                    threshold=0.3,
                    context={"input": input_text, "output": response}
                )

            # Ethical check (optional)
            if self.ethics_checker:
                ethics_result = self.ethics_checker.enforce({
                    "input": input_text,
                    "output": response,
                    "score": composite
                })
                if not ethics_result.get("approved"):
                    composite *= 0.5  # Penalize score

        return float(np.mean(scores))

    def update_preferences(self, feedback: Dict[str, float]):
        """Online update of RLHF weights"""
        for key, value in feedback.items():
            if key in self.preference_weights:
                self.preference_weights[key] = 0.9 * self.preference_weights[key] + 0.1 * value
