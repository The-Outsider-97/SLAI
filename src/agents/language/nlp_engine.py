"""
Core Function:
Extracts linguistic features from normalized input to support downstream parsing, grammar checking, and semantic understanding.

Responsibilities:
Tokenization (breaking text into words, phrases)
Lemmatization (converting words to base forms)
Part-of-Speech (POS) tagging
Generating embeddings (vector representations of text)
Dependency and constituency parsing (if applicable)

Why it matters:
These annotations form the linguistic backbone for grammar checks, intent classification, and entity extraction. Almost every other processor relies on this.
"""

import re
import yaml, json

from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

from src.agents.language.utils.config_loader import load_global_config, get_config_section
from src.agents.language.utils.language_tokenizer import LanguageTokenizer
from src.agents.language.utils.rules import Rules, DependencyRelation
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("NLP Engine")
printer = PrettyPrinter

# --- Coreference Resolution ---
@dataclass
class Entity:
    text: str
    type: str
    gender: str
    number: str
    sentence_index: int
    token_indices: Tuple[int, ...]
    mentions: List[Tuple[int, int]] = field(default_factory=list)  #%20(sentence_idx,%20token_idx)
    coref_id: int = -1  # Group ID for coreference chain

@dataclass
class Token:
    text: str
    lemma: str
    pos: str
    index: int # Original index in the sentence
    # Add other attributes as needed, e.g., is_stop, is_punct, shape, etc.
    is_stop: bool = False
    is_punct: bool = False
    # For more advanced features, you might add:
    # ner_tag: Optional[str] = None
    # embedding: Optional[List[float]] = None

class NLPEngine:
    def __init__(self):
        """
        Initializes the NLP Engine.
        """
        self.config = load_global_config()
        self.wordlist_path = self.config.get('main_wordlist_path')
        
        self.nlp_config = get_config_section('nlp')
        self.irregular_verbs_path = self.nlp_config.get('irregular_verbs_path')
        self.irregular_nouns_path = self.nlp_config.get('irregular_nouns_path')
        self.pos_patterns_path = self.nlp_config.get('pos_patterns_path')
        self.stopwords_list_path = self.nlp_config.get('stopwords_list_path')

        self._stopwords = self._load_stopwords(self.stopwords_list_path)

        # Load resources
        self.wordlist = self._load_wordlist(self.wordlist_path)
        self._stopwords = self._load_stopwords(self.stopwords_list_path)
        self.irregular_verbs = self._load_json(self.irregular_verbs_path)
        self.irregular_nouns = self._load_json(self.irregular_nouns_path)
        self._pos_patterns = self._load_pos_patterns(self.pos_patterns_path)

        logger.info("NLP Engine initialized...")

    @property
    def stopwords(self):
        return self._stopwords
    
    @stopwords.setter
    def stopwords(self, value):
        self._stopwords = value

    @property
    def pos_patterns(self):
        return self._pos_patterns
    
    @pos_patterns.setter
    def pos_patterns(self, value):
        self._pos_patterns = value

    def process_text(self, text: str) -> List[Token]:
        """
        Processes a raw text string into a list of Token objects using the Perception Agent's Tokenizer.
        """
        printer.status("INIT", "Text processor initialized", "info")

        self.tokenizer = LanguageTokenizer()
        
        # Tokenize text using the Tokenizer's internal method
        raw_tokens = self.tokenizer.tokenize(text)
        
        processed_tokens: List[Token] = []
        for i, word_text in enumerate(raw_tokens):
            pos_tag = self._get_pos_tag(word_text)
            lemma = self._get_lemma(word_text, pos_tag)
            is_stop = word_text.lower() in self.stopwords
            is_punct_flag = pos_tag == "PUNCT"
    
            token_obj = Token(
                text=word_text,
                lemma=lemma,
                pos=pos_tag,
                index=i,
                is_stop=is_stop,
                is_punct=is_punct_flag
            )
            processed_tokens.append(token_obj)
    
        logger.debug(f"Processed text into {len(processed_tokens)} tokens.")
        return processed_tokens

    def _get_pos_tag(self, word: str) -> str:
        """
        Assigns a Part-of-Speech tag to a word using wordlist and regex patterns.
        Tags should ideally conform to a standard set (e.g., Universal Dependencies).
        """
        printer.status("INIT", "Part-of-Speech tag initialized", "info")

        word_lower = word.lower()

        # 1. Check wordlist (highest priority)
        if self.wordlist and word_lower in self.wordlist:
            entry = self.wordlist[word_lower]
            if isinstance(entry, dict):
                pos_tags = entry.get("pos", [])
            else:
                pos_tags = []
            if pos_tags:
                # Prioritize more specific tags if multiple exist, or just take the first one.
                # Example: if 'NOUN' and 'VERB' are present, context might be needed.
                # For simplicity, take the first. Could be more sophisticated.
                return pos_tags[0].upper() # Ensure consistent case, e.g., NOUN, VERB

        # 2. Rule-based POS tagging using regex patterns (fallback)
        for pattern, tag in self.pos_patterns:
            if pattern.fullmatch(word): # Use fullmatch for patterns designed for whole words
                return tag.upper()

        # 3. Default/heuristic POS tagging (very basic)
        if re.fullmatch(r'[.,!?;:()"\[\]{}]', word): return "PUNCT"
        if word.isnumeric(): return "NUM"
        if word_lower in {"i", "you", "he", "she", "they", "we", "it", "me", "him", "her", "us", "them"}: return "PRON"
        if word_lower in {"is", "are", "was", "were", "be", "am", "been", "being", "has", "have", "had", "do", "does", "did"}: return "AUX" # Or VERB
        if word_lower.endswith("ing"): return "VERB" # Gerund or present participle
        if word_lower.endswith("ed"): return "VERB" # Past tense or past participle
        if word_lower.endswith("ly"): return "ADV"
        if word_lower.endswith("tion") or word_lower.endswith("sion") or word_lower.endswith("ment") or \
           word_lower.endswith("ness") or word_lower.endswith("ity"): return "NOUN"
        if word_lower.endswith("al") or word_lower.endswith("ous") or word_lower.endswith("ful") or \
           word_lower.endswith("less") or word_lower.endswith("able") or word_lower.endswith("ible"): return "ADJ"
        if word[0].isupper() and word_lower not in self.stopwords_list_path: return "PROPN" # Proper Noun (heuristic)

        return "X" # Unknown or default to NOUN if it's a common fallback

    def _get_lemma(self, word: str, pos: str) -> str:
        """
        Enhanced rule-based lemmatization without external libraries.
        Uses POS hints and pattern matching for verbs, nouns, adjectives, and adverbs.
        """
        printer.status("INIT", "Rule-based lemmatization initialized", "info")

        if pos == "PUNCT":
            return word

        word_lower = word.lower()
        
        # 1. Check wordlist first (highest priority)
        if word_lower in self.wordlist:
            entry = self.wordlist[word_lower]
            if isinstance(entry, dict):
                lemma_from_wordlist = entry.get('lemma')
            else:
                lemma_from_wordlist = entry
                
            if lemma_from_wordlist:
                return lemma_from_wordlist
    
        # 2. Linguistic pattern matching based on POS tag
        # Common irregular verbs (base form -> past/past participle)
        self.irregular_verbs
    
        # Irregular plurals (plural -> singular)
        self.irregular_nouns
    
        # Handle possessives first
        if word_lower.endswith(("'s", "s'", "’s", "s’")):
            word = re.sub(r"['’]s?$", "", word)
            word_lower = word.lower()
    
        # Verb handling
        if pos.startswith('VB'):
            # Check irregular verbs first
            if word_lower in self.irregular_verbs:
                return self.irregular_verbs[word_lower]
            
            # Present participle (running -> run)
            if word_lower.endswith('ing'):
                base = word_lower[:-3]
                # Handle double consonants (running -> run)
                if len(base) > 2 and base[-1] == base[-2]:
                    return base[:-1]
                return base
                
            # Past tense (walked -> walk)
            if word_lower.endswith('ed'):
                base = word_lower[:-2]
                # Handle verbs ending with 'e' (liked -> like)
                if len(base) > 0 and base[-1] == 'e':
                    return base
                # Handle double consonants (stopped -> stop)
                if len(base) > 1 and base[-1] == base[-2]:
                    return base[:-1]
                return base
                
            # Third person singular (walks -> walk)
            if word_lower.endswith('s') and not word_lower.endswith('ss'):
                return word_lower[:-1]
    
        # Noun handling
        elif pos.startswith('NN'):
            # Check irregular plurals
            if word_lower in self.irregular_nouns:
                return self.irregular_nouns[word_lower]
                
            # Standard plural endings
            if word_lower.endswith('ies') and len(word_lower) > 3:  # cities -> city
                return word_lower[:-3] + 'y'
            if word_lower.endswith('ves') and len(word_lower) > 3:  # wives -> wife
                return word_lower[:-3] + 'fe'
            if word_lower.endswith('s') and not word_lower.endswith(('ss', 'us', 'is')):
                return word_lower[:-1]
    
        # Adjective handling
        elif pos == 'ADJ':
            # Comparative/Superlative (bigger -> big, biggest -> big)
            if word_lower.endswith('er'):
                return word_lower[:-2]
            if word_lower.endswith('est'):
                return word_lower[:-3]
            # Other adjective suffixes
            for suffix in ['able', 'ible', 'ful', 'less', 'ous', 'ish']:
                if word_lower.endswith(suffix) and len(word_lower) > len(suffix):
                    return word_lower[:-len(suffix)]
    
        # Adverb handling (quickly -> quick)
        elif pos == 'ADV' and word_lower.endswith('ly') and len(word_lower) > 2:
            return word_lower[:-2]
    
        # Default noun plural handling (after other checks)
        if pos.startswith('NN') and word_lower.endswith('s') and len(word_lower) > 1:
            return word_lower[:-1]
    
        # Final fallback: lowercase with basic stemming
        return word_lower

    def apply_dependency_rules(self, tokens: List['Token']) -> List[DependencyRelation]:
        """
        Applies handcrafted grammatical rules to extract syntactic dependencies
        from a list of processed Token objects.
        """
        printer.status("INIT", "Syntactic dependencies initialized", "info")

        rule_engine = Rules()

        # Convert Token objects to dict format expected by Rules
        token_dicts = [{
            'text': t.text,
            'lemma': t.lemma,
            'upos': t.pos,  # Universal POS
            'id': t.index
        } for t in tokens]

        dependencies = rule_engine._apply_rules(token_dicts)
        return dependencies

    def resolve_coreferences(self, sentences: List[List[Token]]) -> List[Entity]:
        """Coreference resolution with multiple matching strategies."""
        printer.status("INIT", "Coreference resolution initialized", "info")
    
        entity_clusters = defaultdict(list)
        all_entities = []
        coref_id_counter = 0
    
        # Flat list of all entities for candidate matching
        prior_entities: List[Entity] = []
    
        for sent_idx, sentence in enumerate(sentences):
            entities = self._extract_entities(sentence, sent_idx)
    
            for entity in entities:
                matched = False
    
                # Try exact text match (if it's a non-pronoun and named entity)
                if entity.type == "PERSON" and entity.text.lower() not in {"he", "she", "they", "it", "them", "him", "her", "us", "we"}:
                    for prev in reversed(prior_entities):
                        if prev.text.lower() == entity.text.lower():
                            entity.coref_id = prev.coref_id
                            entity_clusters[prev.coref_id].append(entity)
                            matched = True
                            break
    
                # Try pronoun resolution by gender, number, and distance
                if not matched and entity.text.lower() in {"he", "she", "they", "it", "them", "him", "her", "us", "we"}:
                    for prev in reversed(prior_entities):
                        if (
                            entity.gender == prev.gender and
                            entity.number == prev.number and
                            abs(sent_idx - prev.sentence_index) <= 2
                        ):
                            entity.coref_id = prev.coref_id
                            entity_clusters[prev.coref_id].append(entity)
                            matched = True
                            break
    
                # If still unmatched, create a new coreference cluster
                if not matched:
                    entity.coref_id = coref_id_counter
                    entity_clusters[coref_id_counter].append(entity)
                    coref_id_counter += 1
    
                prior_entities.append(entity)
                all_entities.append(entity)
    
        return all_entities

    def _extract_entities(self, tokens: List[Token], sentence_idx: int) -> List[Entity]:
        """Identify entities in a sentence using POS and patterns."""
        entities = []
        current_entity = []
        
        for i, token in enumerate(tokens):
            if token.pos in ['PROPN', 'NOUN', 'PRON']:
                current_entity.append((i, token))
            elif current_entity:
                # Create entity from accumulated tokens
                indices = tuple(idx for idx, _ in current_entity)
                texts = [t.text for _, t in current_entity]
                entity_text = ' '.join(texts)
                
                # Simple gender/number heuristics
                gender = "NEUT"
                if entity_text.lower() in {'he', 'him'}:
                    gender = "MASC"
                elif entity_text.lower() in {'she', 'her'}:
                    gender = "FEM"
                    
                number = "SING" if token.text.lower() in {'he', 'she', 'it'} else "PLUR"
                
                entities.append(Entity(
                    text=entity_text,
                    type="PERSON" if token.pos == 'PROPN' else "OBJECT",
                    gender=gender,
                    number=number,
                    sentence_index=sentence_idx,
                    token_indices=indices
                ))
                current_entity = []
                
        return entities

    # --- Sarcasm/Irony Detection Additions ---
    def detect_sarcasm(self, tokens: List[Token]) -> float:
        """Returns a sarcasm confidence score between 0-1 using lexicon features."""
        printer.status("INIT", "Entity extractor initialized", "info")

        self.nlu_config = get_config_section('nlu')
        lexicon = self.nlu_config.get('sentiment_lexicon_path')
        try:
            with open(Path(lexicon), 'r', encoding='utf-8') as f:
                lexicon = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load sentiment lexicon: {e}")
            return 0.0
        #if not self.sentiment_lexicon_path:
        #    logger.error("Sentiment lexicon path not configured!")
        #    return 0.0
            
        #try:
        #    with open(Path(self.sentiment_lexicon_path), 'r', encoding='utf-8') as f:
        #        lexicon = json.load(f)
        #except Exception as e:
        #    logger.error(f"Failed to load sentiment lexicon: {e}")
        #    return 0.0

        text = ' '.join([t.text.lower() for t in tokens])
        score = 0.0
        
        # 1. Sentiment analysis with negation handling
        base_sentiment = 0.0
        negation_active = False
        current_intensifier = 1.0
        
        for i, token in enumerate(tokens):
            word = token.text.lower()
            lemma = token.lemma
            
            # Handle negators
            if word in lexicon["negators"]:
                negation_active = not negation_active
                continue
                
            # Handle intensifiers
            if word in lexicon["intensifiers"]:
                current_intensifier = lexicon["intensifiers"][word]
                continue
                
            # Calculate sentiment contribution
            sentiment = 0.0
            if lemma in lexicon["positive"]:
                sentiment = lexicon["positive"][lemma] * current_intensifier
            elif lemma in lexicon["negative"]:
                sentiment = lexicon["negative"][lemma] * current_intensifier
                
            if negation_active:
                sentiment *= -1.0  # Flip sentiment
                
            base_sentiment += sentiment
            
            # Reset states after processing
            current_intensifier = 1.0
            negation_active = False  # Negation typically affects next word only
    
        # 2. Sentiment-based sarcasm patterns
        if base_sentiment > 0.5:  # Strong positive sentiment
            # Check for negative context clues
            negative_clues = sum(1 for t in tokens if t.lemma in lexicon["negative"])
            if negative_clues > 2:
                score += min(0.4, negative_clues * 0.15)
                
        elif base_sentiment < -0.5:  # Strong negative sentiment
            # Check for excessive positive words (ironic praise)
            positive_clues = sum(1 for t in tokens if t.lemma in lexicon["positive"])
            if positive_clues > 2:
                score += min(0.4, positive_clues * 0.15)
    
        # 3. Contradiction detection
        positive_words = [t for t in tokens if t.lemma in lexicon["positive"]]
        negative_words = [t for t in tokens if t.lemma in lexicon["negative"]]
        
        if positive_words and negative_words:
            # Check ordering patterns (positive followed by negative)
            first_positive = next((i for i, t in enumerate(tokens) if t.lemma in lexicon["positive"]), None)
            first_negative = next((i for i, t in enumerate(tokens) if t.lemma in lexicon["negative"]), None)
            
            if first_positive is not None and first_negative is not None:
                if first_negative > first_positive:
                    score += 0.3  # Positive setup followed by negative
                else:
                    score += 0.2  # Negative setup followed by positive
    
        # 4. Lexicon-boosted pattern matching
        sarcastic_phrases = {
            'oh great', 'big surprise', 'what a joy', 'as if', 'yeah yeah'
            'yeah right', 'tell me more', 'perfect, just perfect', 'what a shocker'
        }
        
        # Add common sarcastic combinations from lexicon
        for pos_word in lexicon["positive"]:
            sarcastic_phrases.add(f"what a {pos_word}")  # "What a genius idea"
            sarcastic_phrases.add(f"so {pos_word}")      # "So helpful, thanks"
        
        for phrase in sarcastic_phrases:
            if phrase in text:
                score += 0.25
    
        # 5. Punctuation and structural features
        exaggerated_punct = re.findall(r'(!|\?){2,}', text)
        if len(exaggerated_punct) > 0:
            score += 0.15 * len(exaggerated_punct)
            
        # 6. Adverb patterns from lexicon
        ironic_adverbs = {'totally', 'completely', 'absolutely', 'utterly'}
        if any(t.lemma in ironic_adverbs and t.pos == 'ADV' for t in tokens):
            score += 0.2
    
        # 7. Capitalization emphasis
        if re.search(r'\b[A-Z]{3,}\b', text):
            score += 0.1
    
        # Normalize and clamp score
        return min(max(score, 0.0), 1.0)
    
    def _load_wordlist(self, path: str) -> Dict:
        """Load structured wordlist dictionary"""
        if not path:
            return {}
        try:
            with open(Path(path), 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('words', {})
        except Exception as e:
            logger.error(f"Failed to load wordlist: {e}")
            return {}
    
    def _load_stopwords(self, path: str) -> Set:
        """Load stopwords as a set"""
        if not path:
            return set()
        try:
            with open(Path(path), 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load stopwords: {e}")
            return set()
        
    def _load_json(self, path: str) -> Dict:
        if not path:
            return {}
        try:
            with open(Path(path), 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            return {}
        
    def _load_pos_patterns(self, path: str) -> List[Tuple[re.Pattern, str]]:
        """Load and compile POS regex patterns from JSON file."""
        if not path:
            return []
        
        try:
            with open(Path(path), 'r', encoding='utf-8') as f:
                patterns = json.load(f)
            compiled = []
            for p in patterns:
                # Handle both string and list patterns
                pattern_str = p.get('pattern')
                if isinstance(pattern_str, list):
                    # Skip grammar patterns (not word-level)
                    continue
                if pattern_str:
                    regex = re.compile(p['pattern'])
                    # Use first tag from example if available
                    tag = p.get('example_tags', [''])[0]
                    compiled.append((regex, tag))
            return compiled
        except Exception as e:
            logger.error(f"Failed to load POS patterns: {e}")
            return []

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("\n=== Running Natural Language Processor Engine (NLP Engine) ===\n")
    printer.status("Init", "NLP Engine initialized", "success")

    engine = NLPEngine()
    print(engine)

    print("\n* * * * * Phase 2 * * * * *\n")
    text1="There aren't any resources where we are going, so get packing friend."
    tokens = engine.process_text(text=text1)
    batch = [
        engine.process_text("We can start training as soon as I'm done eating"),
        engine.process_text("Where are the kids going? I'm not even heading there!")
    ]

    printer.pretty("Init", tokens, "success")
    printer.pretty("rules", engine.apply_dependency_rules(tokens=tokens), "success")
    printer.pretty("resolve", engine.resolve_coreferences(sentences=batch), "success")

    print("\n* * * * * Phase 3 * * * * *\n")
    text2="Eric: 'Did you know sounds are waves? Me: 'You don't say, what a shocker!'"
    tokens2=engine.process_text(text=text2)

    printer.pretty("resolve", engine.detect_sarcasm(tokens=tokens2), "success")

    print("\n=== Finished Running NLP Engine ===\n")
