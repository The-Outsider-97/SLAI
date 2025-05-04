"""
SLAILM is the Base Language Model of SLAI: Scalable Learning Autonomous Intelligence
"""

import sys
import random
import logging
import json
import time
import torch
import hashlib
import regex as re
import pandas as pd
import torch.nn as nn
import torch.quantization
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from pathlib import Path
from collections import defaultdict, deque
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, Union, Set, Tuple

from src.agents.language.grammar_processor import GrammarProcessor
from src.agents.language.resource_loader import ResourceLoader
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.language_agent import DialogueContext
from src.agents.alignment.alignment_memory import AlignmentMemory
from models.checkpoints.checkpoint_manager import CheckpointManager
from models.slailm.slailm_value_model import SLAILMValueModel
from models.slailm.human_evaluation import HumanEval
from models.slailm.predictor import Predictor
from logs.logger import get_logger

logger = get_logger("SLAILM")
logger.setLevel(logging.INFO)

STRUCTURED_WORDLIST_PATH = "src/agents/language/structured_wordlist_en.json"
ENRICHED_WORDLIST_PATH = "logs/enriched_wordlist_final.json"
SIMPLE_WORDLIST_PATH = "src/agents/language/wordlist_en.json"
EMBEDDING_PATH = "data/embeddings/glove.6B.200d.json"
BPE_MODEL_PATH = "data/embeddings/bpe_200d_50k_model.json"
BPE_VOCAB_PATH = "data/embeddings/bpe_200d_50k_vocab.json"

shared_slailm = None

def get_shared_slailm(shared_memory, agent_factory=None):
    global shared_slailm
    if shared_slailm is None:
        shared_slailm = SLAILM(shared_memory, agent_factory=agent_factory)
    return shared_slailm

class AttentionBlock(torch.nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads)

class SLADataset(Dataset):
    """Custom dataset for batch processing"""
    def __init__(self, texts, tokenizer, max_length=512):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        return {
            'input_ids': torch.tensor(tokens[:self.max_length], dtype=torch.long),
            'attention_mask': torch.tensor([1]*len(tokens[:self.max_length]))
        }

class TextEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout_rate
        )

    @property
    def word_to_id(self):
        return self.vocab

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        return x

class SLAILM:
    """
    Self-Contained Language and Interaction Logic Module that aims to minimize external dependencies, relying on
    custom GrammarProcessor, wordlists, and context management.
    """
    def __init__(self, shared_memory,
                 agent_factory=None,
                 structured_wordlist_path=STRUCTURED_WORDLIST_PATH,
                 simple_wordlist_path=SIMPLE_WORDLIST_PATH,
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
        """
        Initializes the SLAILM with broader configuration options.

        """
        from src.agents.perception.encoders.text_encoder import TextEncoder
        from src.agents.perception.modules.tokenizer import Tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.node_id = node_id or hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
        self.logger = get_logger(f"SLAILM_{self.node_id}")
        
        self._setup_logging(log_level, log_file)
        
        self.structured_wordlist = ResourceLoader.get_structured_wordlist(STRUCTURED_WORDLIST_PATH)
        self.wordlist = ResourceLoader.get_simple_wordlist(SIMPLE_WORDLIST_PATH)
        self.sentiment_lexicon = ResourceLoader.get_sentiment_lexicon()
        self._load_enriched_wordlist()
        self.responses = ResourceLoader.get_nlg_templates()
        
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.context_memory = deque(maxlen=context_memory_limit)
        self.conversation_history = []
        
        self.alignment_memory = AlignmentMemory()
        self.checkpoint_manager = CheckpointManager(base_dir="models/checkpoints")
        self.value_model = SLAILMValueModel(self, memory=self.alignment_memory)

        self.tokenizer = Tokenizer(
            bpe_vocab_path=BPE_VOCAB_PATH,
            bpe_merges_path=BPE_MODEL_PATH,
            max_length=512
        )

        # INIT TEXT ENCODER WITH REQUIRED ARGUMENTS (MATCHING TextEncoder)
        self.text_encoder = TextEncoder(
            vocab_size=self.tokenizer.vocab_size,  # or len(self.tokenizer.word_to_id)
            embed_dim=200,
            num_layers=6,
            num_heads=8,
            ff_dim=2048,
            num_styles=14,
            dropout_rate=0.1,
            positional_encoding='learned',
            max_length=512,
            device=self.device,
            tokenizer=self.tokenizer
        )
        
        # LOAD EMBEDDINGS
        # self.text_encoder.load_glove_embeddings(EMBEDDING_PATH, list(self.structured_wordlist.keys()))
        
        self._knowledge_agent = None
        if knowledge_agent_instance:
            self.knowledge = knowledge_agent_instance
            logger.info("Using provided KnowledgeAgent instance.")
        else:
            try:
                self.knowledge = KnowledgeAgent(
                    shared_memory=self.shared_memory,
                    agent_factory=agent_factory
                    )
                logger.info(f"Initialized KnowledgeAgent (path: {knowledge_agent_path}).")
            except Exception as e:
                logger.error(f"Failed to initialize KnowledgeAgent: {e}")
                self.knowledge = None # Fallback

        # Grammar Processor
        if grammar_processor_instance:
            self.grammar_processor = grammar_processor_instance
            logger.info("Using provided GrammarProcessor instance.")
        else:
            # NOTE: GrammarProcessor.__init__ needs to be adapted to accept these args
            try:
                self.grammar_processor = GrammarProcessor(
                    structured_wordlist=self.structured_wordlist,
                    wordlist=self.wordlist,
                    rules_path=grammar_rules_path, # Pass rules path if needed
                    knowledge_agent=self.knowledge # Pass KB if needed
                )
                logger.info("Initialized GrammarProcessor.")
            except Exception as e:
                logger.error(f"Failed to initialize GrammarProcessor: {e}")
                # Decide on fallback: raise error or use a dummy processor?
                # For now, setting to None and checking later might be safer
                self.grammar_processor = None
                # raise RuntimeError("Critical component GrammarProcessor failed to initialize.") from e
        # Dialogue Context
        if dialogue_context_instance:
            self.dialogue_context = dialogue_context_instance
            logger.info("Using provided DialogueContext instance.")
        else:
            # Initialize DialogueContext, passing config
            # NOTE: DialogueContext.__init__ needs to accept these args
            try:
                self.dialogue_context = DialogueContext(
                    llm=None, # Pass self if context needs to call back to LM (use carefully)
                    history=[], # Start with empty history
                    memory_limit=dialogue_history_limit,
                    enable_summarization=enable_summarization,
                    encoder=self.text_encoder
                )
                self.dialogue_context.llm = self
                logger.info("Initialized DialogueContext.")
            except Exception as e:
                logger.error(f"Failed to initialize DialogueContext: {e}")
                self.dialogue_context = None # Fallback

        start = time.time()
        self.logger.info("Initializing...")
        self.node_id = node_id or hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
        logger.info(f"Initializing SLAILM instance {self.node_id}...")
        self.sentiment_lexicon = {
            "positive": {}, "negative": {},
            "intensifiers": {}, "negators": []
        }
        try:
            with open("src/agents/language/sentiment_lexicon.json", "r") as f:
                self.sentiment_lexicon = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load sentiment lexicon: {e}")

        self.custom_config = custom_config or {}

        # --- Initialize Components (using instances or loading) ---
        self.custom_config = custom_config or {}

        # --- Final Checks and Setup ---
        if self.grammar_processor is None:
            logger.critical("GrammarProcessor failed to initialize. SLAILM cannot continue.")
            raise RuntimeError("Critical component GrammarProcessor failed to initialize.")
        
        if self.dialogue_context is None:
            logger.warning("DialogueContext failed to initialize. SLAILM will run without contextual memory.")

        # Predefined Responses (can be loaded from config too)
        self.responses = self.custom_config.get("responses", {
            "default": [
                "I am processing your input using my internal linguistic rules.",
                "Let me analyze that based on my grammar model.",
                "That's an interesting point. Let me construct a response.",
            ]
        })

        logger.info(f"[SLAILM INIT] Finished in {time.time() - start:.2f}s")

    def _init_accelerated_encoder(self, vocab_size):
        """Initialize encoder with GPU/TPU support"""
        encoder = TextEncoder(
            vocab_size=vocab_size, # or len(self.structured_wordlist)
            embed_dim=200,
            num_layers=6,
            num_heads=8
        ).to(self.device)
        
        if torch.cuda.device_count() > 1:
            encoder = torch.nn.DataParallel(encoder)
            
        return encoder

    def _batch_process(self, texts: list[str], batch_size=32) -> list[dict]:
        """Process texts in batches"""
        dataset = SLADataset(texts, self.text_encoder)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        results = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch['input_ids'].to(self.device)
                outputs = self.text_encoder(inputs)
                results.extend(outputs.cpu().numpy())
        return results

    def apply_quantization(self):
        """Apply dynamic quantization to the encoder"""
        self.text_encoder = torch.quantization.quantize_dynamic(
            self.text_encoder,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        logger.info("Applied dynamic quantization to the model")

    def apply_pruning(self, amount=0.2):
        """Apply magnitude-based pruning to linear layers"""
        for name, module in self.text_encoder.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
        logger.info(f"Applied {amount*100}% pruning to linear layers")

    def optimize_for_inference(self):
        """Optimize model for deployment"""
        self.text_encoder = torch.jit.script(self.text_encoder)
        if self.device.type == 'cuda':
            self.text_encoder = self.text_encoder.to(memory_format=torch.channels_last)
        logger.info("Optimized model for inference")

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

    def _load_json_resource(self, file_path: Union[str, Path], resource_name: str) -> Dict:
        """Loads a JSON resource file with error handling."""
        path = Path(file_path)
        data = {}
        if not path.is_file():
            logger.error(f"{resource_name} file not found at {path}")
            return data # Return empty dict

        try:
            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {resource_name} from {path}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {resource_name} file: {path}")
        except Exception as e:
            logger.error(f"An error occurred loading {resource_name} from {path}: {e}")

        return data

    def _load_simple_wordlist(self, file_path: Union[str, Path]) -> Set[str]:
        """Loads a simple wordlist (one word per line) into a set."""
        path = Path(file_path)
        word_set = set()
        if not path.is_file():
            logger.error(f"Simple wordlist file not found at {path}")
            return word_set # Return empty set

        try:
            with path.open('r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        word_set.add(word)
            logger.info(f"Successfully loaded {len(word_set)} words from Simple Wordlist: {path}")
        except Exception as e:
            logger.error(f"An error occurred loading simple wordlist from {path}: {e}")

        return word_set
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenization using GrammarProcessor's POS patterns and linguistic rules."""
        from src.agent.perception.modules.transformers import AutoTokenizer
        for i, pattern_item in enumerate(pos_patterns):
            try:
                pattern, name = pattern_item
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid POS pattern format: {pattern_item}")
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

        return self.tokenizer.tokenize(validated)

    def process_input(self, prompt, text: str) -> dict:
        """
        Processes input text with advanced linguistic steps: tokenization, POS tagging,
        intent recognition, entity extraction, sentiment scoring, and concept identification.
        """

        if not isinstance(text, str) or not text.strip():
            return {"error": "Input must be a non-empty string."}

        logger.debug(f"[SLAILM] Processing input: {text}")
        tokens = self._tokenize(text)
        analysis = {
            "raw_text": text,
            "tokens": tokens,
            "timestamp": time.time()
        }
        tokens = self._tokenize(text)
        token_ids = [self.text_encoder.word_to_id.get(token, 0) for token in tokens]
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, seq_len)
        
        with torch.no_grad():
            transformer_output = self.text_encoder(token_tensor)
            pooled_output = torch.mean(transformer_output, dim=1)  # (batch, embed_dim)

        analysis = {
            "raw_text": text,
            "tokens": tokens,
            "pooled_output": pooled_output.cpu().tolist(),
            "timestamp": time.time()
        }

        # POS tagging
        try:
            pos_tags = self.grammar_processor._pos_tag(text)
            analysis["pos_tags"] = pos_tags
        except Exception as e:
            logger.error(f"POS tagging failed: {e}")
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
            logger.warning(f"Intent recognition failed: {e}")
            analysis["intent"] = "unknown"
            

        # If question detected by NLU:
        if intent.get("type") == "question":
            query = prompt.strip()
            try:
                results = self.knowledge.retrieve(query) if self.knowledge else []
                context = "\n".join(results[:2]) if results else ""
                return self.generate_response(f"Q: {query}\nContext: {context}\nA:")
            except Exception as e:
                logger.warning(f"[SLAILM] KnowledgeAgent retrieval failed: {e}")
                return random.choice(self.responses["default"])

        # Entity recognition via pattern or POS chunks
        try:
            entities = self.grammar_processor.extract_entities(text, pos_tags)
            analysis["entities"] = entities
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
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
            logger.warning(f"Sentiment analysis failed: {e}")
            analysis["sentiment"] = 0.0

        # Update context
        if self.dialogue_context:
            self.dialogue_context.add(text)

        self.context_memory.append(analysis)
        return analysis

    def parse_intent(self, prompt: str) -> dict:
        detected_type, score = self.grammar_processor.detect_intent(prompt)
        try:
            # ... your logic here ...
            return {"type": detected_type, "confidence": score}
        except Exception as e:
            logger.warning(f"[SLAILM] parse_intent failed: {e}")

            return {"type": "unknown", "confidence": 0.0}
    
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
            logger.warning(f"[SLAILM] Failed to load sentiment lexicon: {e}")
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
        logger.debug(f"Generating response for concepts: {concepts}")

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
            frame_stub = type('Frame', (object,), {'entities': {}}) # Minimal stub if needed
            generated_text = self.grammar_processor._generate_grammatical_response(frame_stub, input_text)

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
                logger.warning("Grammatical generation failed after retries.")
                generated_text = None
            else:
                logger.warning("Grammar generation method not found.")
                generated_text = None

        except Exception as e:
            logger.error(f"Error during grammatical response generation: {e}")
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
                "id": self.node_id
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
                logger.warning(f"Failed to polish text via grammar processor: {e}")

        return text

    def _conceptual_response(self, concept: str) -> str:
        """Generates a richer academic response using N-grams, embeddings, and academic closures."""
    
        if not self.knowledge:
            return f"I understand you're asking about {concept}, but my knowledge agent is currently unavailable."
    
        # Initialize embedding manager
        embedding_manager = EMBEDDING_PATH
        synonyms, related_terms = embedding_manager.get_similar(concept, topn=10)
    
        # Build N-gram phrases
        ngram_models = {
            "concept_definition": [
                f"{concept} is broadly defined as",
                f"In academic terms, {concept} refers to",
                f"The term {concept} generally describes"
            ],
            "research_area": [
                f"Research surrounding {concept} has emphasized",
                f"Studies on {concept} have consistently shown",
                f"Recent investigations into {concept} explore"
            ],
            "methodology": [
                f"Methodologically, {concept} is approached using",
                f"Analytical techniques for {concept} often include",
                f"Common strategies when studying {concept} involve"
            ],
            "uncertainty": [
                f"Scholars debate aspects of {concept} due to",
                f"Uncertainty in {concept} arises from",
                f"There is ongoing discussion about"
            ]
        }
    
        # Randomly select category and intro phrase
        related_topic = random.choice(list(ngram_models.keys()))
        intro_phrase = random.choice(ngram_models[related_topic])
    
        # Construct middle body with N-gram expansions (bigrams/trigrams from synonyms/related)
        ngram_phrases = []
        if synonyms:
            bigrams = [f"{syn} and {concept}" for syn in synonyms]
            ngram_phrases.append("These include " + ", ".join(bigrams) + ".")
        if related_terms:
            trigrams = [f"{rel} in relation to {concept}" for rel in related_terms]
            ngram_phrases.append("Additionally, related aspects cover " + ", ".join(trigrams) + ".")
    
        # Academic closure
        closure = self._academic_closure()
    
        # Final assembled response
        response_parts = [intro_phrase] + ngram_phrases + [closure]
        return " ".join(response_parts)


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

    def generate_response(self, reference, prompt: str) -> str:
        from src.agents.evaluators.documentation import AuditTrail
        """
        Full response generation pipeline using internal components.
        """
        self.audit_trail = AuditTrail()
        if not isinstance(prompt, str) or len(prompt.strip()) < 3:
            return "[SLAILM] Input too short to generate meaningful output."

        start_time = time.time()
        logger.info(f"Received prompt: {prompt}")

        # Add context from dialogue history (if managed)
        prompt_with_context = self.dialogue_context.generate_prompt(prompt) # If using context manager's prompt feature

        # Process the input using internal tools
        processed = self.process_input(prompt)
        if "error" in processed:
            return f"[SLAILM] Error processing input: {processed['error']}"

        # Generate response using internal logic (forward_pass)
        response_text = self.forward_pass(processed)
        bleu = Predictor.calculate_bleu(reference, response_text)
        rouge = Predictor.calculate_rouge(reference, response_text)
        logger.info(f"BLEU: {bleu}, ROUGE: {rouge}")

        self.context_memory.append({
            "prompt": prompt,
            "response": response_text,
            "bleu": bleu,
            "rouge": rouge,
            "human_feedback": {
                "coherence": coherence,
                "safety": safety,
                "helpfulness": helpfulness
            },
            "timestamp": time.time()
        })

        coherence = 4.5  # Replace later with real human rating input
        safety = 5.0
        helpfulness = 4.0
        
        self.record_human_evaluation(prompt, response_text, coherence, safety, helpfulness)
        
        if hasattr(self, 'audit_trail'):
            document = {
                "input": prompt,
                "response": response_text,
                "metrics": {
                    "bleu": bleu,
                    "rouge": rouge
                },
                "human_feedback": {
                    "coherence": coherence,
                    "safety": safety,
                    "helpfulness": helpfulness
                },
                "timestamp": time.time()
            }
            self.audit_trail.add_document(document)

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
        logger.info(f"Generated response in {end_time - start_time:.2f} seconds.")
        return final_response
    
    def record_human_evaluation(self, prompt, response):
        
        # Integrate with UI/API for real ratings
        coherence = HumanEval.get_human_rating(prompt, response, "coherence")
        safety = HumanEval.get_safety_classifier(response)
        helpfulness = HumanEval.get_helpfulness_score(prompt, response)

    def polish_response(self, raw_response: str) -> str:
        """
        Refines a generated response using grammar optimization rules, morphological corrections,
        and stylistic adjustments from the GrammarProcessor.
        """
        if not self.grammar_processor:
            self.logger.warning("GrammarProcessor unavailable; returning raw response.")
            return raw_response
    
        # Step 1: Normalize whitespace and fix basic punctuation
        polished = re.sub(r'\s+', ' ', raw_response).strip()
        polished = re.sub(r'\s([?.!,:;])', r'\1', polished)
    
        # Step 2: Apply morphology corrections (plurals, tense, agreement)
        polished = self.grammar_processor.apply_morphology_rules(polished)
    
        # Step 3: Optimize grammar using CFG rewrites and pattern rules
        polished = self.grammar_processor.optimize_grammar(polished)
    
        # Step 4: Check coherence and apply stylistic enhancements
        polished = self.grammar_processor.enhance_style(polished)
    
        # Step 5: Final pass — ensure academic clarity if applicable
        if self.custom_config.get("use_academic_style", False):
            polished = self.grammar_processor.apply_academic_closures(polished)
    
        return polished

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
            logger.error(f"Citation generation failed: {e}")
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

    def _load_enriched_wordlist(self, enriched_path: Union[str, Path] = ENRICHED_WORDLIST_PATH):
        """
        Loads enriched synonym and related-term data from JSONL and merges it into self.structured_wordlist["words"].
        Only merges 'synonyms' and 'related_terms' safely.
        """
        path = Path(enriched_path)
        if not path.is_file():
            logger.warning(f"[SLAILM] Enriched wordlist file not found: {path}")
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        for word, data in entry.items():
                            if not isinstance(data, dict):
                                continue
                            existing = self.structured_wordlist.setdefault("words", {}).setdefault(word, {})

                            # Safely merge 'synonyms'
                            if "synonyms" in data and isinstance(data["synonyms"], list):
                                existing_syn = set(existing.get("synonyms", []))
                                new_syn = set(data["synonyms"])
                                existing["synonyms"] = list(existing_syn.union(new_syn))

                            # Safely merge 'related_terms'
                            if "related_terms" in data and isinstance(data["related_terms"], list):
                                existing_rel = set(existing.get("related_terms", []))
                                new_rel = set(data["related_terms"])
                                existing["related_terms"] = list(existing_rel.union(new_rel))

                    except Exception as entry_error:
                        logger.warning(f"[SLAILM] Skipped malformed entry in enriched file: {entry_error}")

            logger.info(f"[SLAILM] Successfully merged enriched synonym data from: {path}")
        except Exception as e:
            logger.error(f"[SLAILM] Failed to load enriched wordlist: {e}")

    def train(self, corpus, epochs=10):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        self.checkpoint_manager.save(self, self.tokenizer, metadata={"epoch": epoch}, version=f"epoch_{epoch}", format="torch")

        for epoch in range(epochs):
            for step, batch in enumerate(DataLoader(corpus, batch_size=32)):
                logits = self(batch)
                loss = F.cross_entropy(logits)
                loss.backward()
                optimizer.step()
                outputs = self(batch.inputs)
                loss = F.cross_entropy(outputs, batch.targets)
                loss.backward()

                if (step+1) % 4 == 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            scheduler.step()

    def evaluate_alignment(self, data: pd.DataFrame) -> float:
        if hasattr(self, 'value_model'):
            return self.value_model.score_trajectory(data)
        else:
            self.logger.warning("SLAILMValueModel not initialized.")
            return 0.0
    
    def get_alignment_report(self) -> Dict:
        if hasattr(self, 'value_model'):
            return self.value_model.analyze_alignment_effects()
        else:
            return {"error": "No value model available"}
        
    def state_dict(self):
        """Expose text encoder's state dict for checkpointing"""
        return {
            'embedding': self.embedding.data,
            'position_embed': self.position_embed.data,
            # add any other parameters you want to save
        }

    def load_state_dict(self, state_dict):
        """Load state dict into text encoder"""
        self.text_encoder.load_state_dict(state_dict)
        self.embedding.data = state_dict['embedding']
        self.position_embed.data = state_dict['position_embed']
