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
from typing import List, Dict, Tuple, Optional, Any

from src.agents.language.utils.rules import Rules, DependencyRelation
from logs.logger import get_logger

logger = get_logger("NLP Engine")

CONFIG_PATH = "src/agents/language/configs/language_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return

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

    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "should",
        "can", "could", "may", "might", "must", "and", "but", "or", "nor",
        "for", "so", "yet", "in", "on", "at", "by", "from", "to", "with",
        "about", "above", "after", "again", "against", "all", "am", "as",
        "because", "before", "below", "between", "both", "each", "few",
        "further", "he", "her", "here", "hers", "herself", "him", "himself",
        "his", "how", "i", "if", "into", "it", "its", "itself", "just",
        "me", "more", "most", "my", "myself", "no", "not", "now", "of",
        "off", "once", "only", "other", "our", "ours", "ourselves", "out",
        "over", "own", "same", "she", "since", "some", "still", "such",
        "than", "that", "their", "theirs", "them", "themselves", "then",
        "there", "these", "they", "this", "those", "through", "too",
        "under", "until", "up", "very", "we", "what", "when", "where",
        "which", "while", "who", "whom", "why", "won", "you", "your",
        "yours", "yourself", "yourselves"
    }

    def __init__(self, config,
                 wordlist_data: Optional[Dict[str, Any]] = None,
                 pos_patterns: Optional[List[Tuple[re.Pattern, str]]] = None,
                 stopwords: Optional[set] = None):
        """
        Initializes the NLP Engine.

        Args:
            wordlist_data (Optional[Dict[str, Any]]): Pre-loaded structured wordlist
                                                      for POS tagging and lemmatization hints.
                                                      Format: {'word': {'pos': ['NOUN'], 'lemma': 'word', ...}}
            pos_patterns (Optional[List[Tuple[re.Pattern, str]]]):
                                                      List of (regex_pattern, POS_tag) for rule-based tagging.
            stopwords (Optional[set]): A set of stopwords for the language.
        """
        self.config = config
        self.wordlist_data = wordlist_data if wordlist_data else {}
        self.pos_patterns = pos_patterns if pos_patterns else []
        self.stopwords = stopwords if stopwords else set()

        # Basic English stopwords if none provided
        if not self.stopwords:
            self.stopwords = {
                "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "will", "would", "should",
                "can", "could", "may", "might", "must", "and", "but", "or", "nor",
                "for", "so", "yet", "in", "on", "at", "by", "from", "to", "with",
                "about", "above", "after", "again", "against", "all", "am", "as",
                "because", "before", "below", "between", "both", "each", "few",
                "further", "he", "her", "here", "hers", "herself", "him", "himself",
                "his", "how", "i", "if", "into", "it", "its", "itself", "just",
                "me", "more", "most", "my", "myself", "no", "not", "now", "of",
                "off", "once", "only", "other", "our", "ours", "ourselves", "out",
                "over", "own", "same", "she", "since", "some", "still", "such",
                "than", "that", "their", "theirs", "them", "themselves", "then",
                "there", "these", "they", "this", "those", "through", "too",
                "under", "until", "up", "very", "we", "what", "when", "where",
                "which", "while", "who", "whom", "why", "won", "you", "your",
                "yours", "yourself", "yourselves"
            }
        logger.info("NLP Engine initialized...")

    def _get_lemma(self, word: str, pos: str) -> str:
        """
        Enhanced rule-based lemmatization without external libraries.
        Uses POS hints and pattern matching for verbs, nouns, adjectives, and adverbs.
        """
        word_lower = word.lower()
        
        # 1. Check wordlist first (highest priority)
        if self.wordlist_data and word_lower in self.wordlist_data:
            lemma_from_wordlist = self.wordlist_data[word_lower].get("lemma")
            if lemma_from_wordlist:
                return lemma_from_wordlist
    
        # 2. Linguistic pattern matching based on POS tag
        # Common irregular verbs (base form -> past/past participle)
        irregular_verbs = {
            'ran': 'run', 'ate': 'eat', 'came': 'come', 'went': 'go',
            'was': 'be', 'were': 'be', 'had': 'have', 'did': 'do',
            'saw': 'see', 'found': 'find', 'spoke': 'speak', 'took': 'take'
        }
    
        # Irregular plurals (plural -> singular)
        irregular_nouns = {
            'children': 'child', 'geese': 'goose', 'mice': 'mouse',
            'teeth': 'tooth', 'feet': 'foot', 'oxen': 'ox',
            'men': 'man', 'women': 'woman', 'people': 'person'
        }
    
        # Handle possessives first
        if word_lower.endswith(("'s", "s'", "’s", "s’")):
            word = re.sub(r"['’]s?$", "", word)
            word_lower = word.lower()
    
        # Verb handling
        if pos.startswith('VB'):
            # Check irregular verbs first
            if word_lower in irregular_verbs:
                return irregular_verbs[word_lower]
            
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
            if word_lower in irregular_nouns:
                return irregular_nouns[word_lower]
                
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

    def _get_pos_tag(self, word: str) -> str:
        """
        Assigns a Part-of-Speech tag to a word using wordlist and regex patterns.
        Tags should ideally conform to a standard set (e.g., Universal Dependencies).
        """
        word_lower = word.lower()

        # 1. Check wordlist (highest priority)
        if self.wordlist_data and word_lower in self.wordlist_data:
            pos_tags = self.wordlist_data[word_lower].get("pos", [])
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
        if word[0].isupper() and word_lower not in self.stopwords: return "PROPN" # Proper Noun (heuristic)

        return "X" # Unknown or default to NOUN if it's a common fallback

    def process_text(self, text: str) -> List[Token]:
        """
        Processes a raw text string into a list of Token objects using the Perception Agent's Tokenizer.
        """
        from src.agents.perception.modules.tokenizer import Tokenizer, load_config as load_perception_config
        
        # Load config and initialize tokenizer
        config = load_perception_config()
        tokenizer = Tokenizer(config)
        
        # Tokenize text using the Tokenizer's internal method
        raw_tokens = tokenizer._tokenize(text)
        
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

    def apply_dependency_rules(self, tokens: List['Token']) -> List[DependencyRelation]:
        """
        Applies handcrafted grammatical rules to extract syntactic dependencies
        from a list of processed Token objects.
        """
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

    def resolve_coreferences(self, sentences: List[List[Token]]) -> List[Entity]:
        """Main coreference resolution method."""
        entity_clusters = defaultdict(list)
        all_entities = []
        coref_id = 0
        
        for sent_idx, sentence in enumerate(sentences):
            entities = self._extract_entities(sentence, sent_idx)
            
            for entity in entities:
                # Check for matching antecedents
                matched = False
                for cluster in entity_clusters.values():
                    last_ent = cluster[-1]
                    
                    # Match based on gender/number/proximity
                    if (entity.gender == last_ent.gender and 
                        entity.number == last_ent.number and
                        sent_idx - last_ent.sentence_index <= 2):
                        cluster.append(entity)
                        entity.coref_id = last_ent.coref_id
                        matched = True
                        break
                        
                if not matched:
                    entity.coref_id = coref_id
                    entity_clusters[coref_id].append(entity)
                    coref_id += 1
                
                all_entities.append(entity)
        
        return all_entities

    # --- Sarcasm/Irony Detection Additions ---
    def detect_sarcasm(self, tokens: List[Token]) -> float:
        """Returns a sarcasm confidence score between 0-1 using lexicon features."""
        text = ' '.join([t.text.lower() for t in tokens])
        lexicon_path = self.config.get("nlu", {}).get("sentiment_lexicon_path")
        if not lexicon_path:
            logger.error("Sentiment lexicon path not configured!")
            return 0.0
            
        try:
            with open(Path(lexicon_path), 'r', encoding='utf-8') as f:
                lexicon = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load sentiment lexicon: {e}")
            return 0.0

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
            'oh great', 'big surprise', 'what a joy', 'as if',
            'yeah right', 'tell me more', 'perfect just perfect'
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

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("\n=== Running NLP Engine ===\n")

    config = load_config()
    engine = NLPEngine(config)
    print(engine)
    print("\n=== Finished Running NLP Engine ===\n")

# --- Interactive Usage ---
if __name__ == "__main__":
    from pathlib import Path
    
    print("""←[36m
    #############################################
    #   NLP Engine Interactive Mode            #
    #   Commands:                              #
    #   /tokens - Show tokenization            #
    #   /deps   - Show dependencies            #
    #   /sarcasm - Analyze sarcasm             #
    #   /coref  - Multi-sentence coreference   #
    #   /exit   - Quit                         #
    #############################################←[0m
    """)
    
    config = load_config()
    engine = NLPEngine(config)
    session_history = []
    
    while True:
        try:
            user_input = input("\n←[33mEnter text or command:←[0m ").strip()
            
            if user_input.lower() in ('/exit', '/quit'):
                print("Exiting...")
                break
                
            elif user_input == '/tokens':
                if not session_history:
                    print("←[31mNo text in session history!←[0m")
                    continue
                last_text = ' '.join(session_history[-1])
                tokens = engine.process_text(last_text)
                print(f"\n←[34mToken analysis for: '{last_text}':←[0m")
                for token in tokens:
                    print(f"{token.text:<15} {token.pos:<5} {token.lemma}")
                    
            elif user_input == '/deps':
                if not session_history:
                    print("←[31mNo text in session history!←[0m")
                    continue
                last_text = ' '.join(session_history[-1])
                tokens = engine.process_text(last_text)
                deps = engine.apply_dependency_rules(tokens)
                print(f"\n←[34mDependencies for: '{last_text}':←[0m")
                for dep in deps:
                    print(f"{dep.relation.upper():<15} {dep.head} → {dep.dependent}")
                    
            elif user_input == '/sarcasm':
                if not session_history:
                    print("←[31mNo text in session history!←[0m")
                    continue
                last_text = ' '.join(session_history[-1])
                tokens = engine.process_text(last_text)
                score = engine.detect_sarcasm(tokens)
                print(f"\n←[34mSarcasm analysis for: '{last_text}':←[0m")
                print(f"Confidence score: ←[35m{score:.2f}/1.0←[0m")
                print("Interpretation:")
                if score > 0.75: print("←[31mStrong sarcasm detected←[0m")
                elif score > 0.5: print("←[33mLikely sarcastic←[0m")
                else: print("←[32mNo strong sarcasm indicators←[0m")
                
            elif user_input == '/coref':
                if len(session_history) < 2:
                    print("←[31mNeed at least 2 sentences for coreference!←[0m")
                    continue
                all_tokens = [engine.process_text(text) for text in session_history]
                entities = engine.resolve_coreferences(all_tokens)
                print("\n←[34mCoreference chains:←[0m")
                clusters = defaultdict(list)
                for ent in entities:
                    clusters[ent.coref_id].append(ent)
                for cid, ents in clusters.items():
                    mentions = [f"'{e.text}' (sentence {e.sentence_index+1})" for e in ents]
                    print(f"Group {cid}: {', '.join(mentions)}")
                    
            else:
                # Store raw text input
                session_history.append(user_input)
                tokens = engine.process_text(user_input)
                print(f"←[32mProcessed {len(tokens)} tokens←[0m")
                
        except Exception as e:
            print(f"←[31mError: {str(e)}←[0m")
            logger.error(f"Interactive session error: {str(e)}")
