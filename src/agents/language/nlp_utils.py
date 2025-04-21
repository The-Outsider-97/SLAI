import re
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import namedtuple, deque, defaultdict
from typing import List, Dict, Tuple, Optional, Any

from src.agents.language.grammar_processor import GrammarProcessor
from src.agents.language.resource_loader import ResourceLoader

logger = logging.getLogger(__name__)

# --- Coreference Resolution ---
@dataclass
class Entity:
    text: str
    type: str
    gender: str
    number: str
    sentence_index: int
    token_indices: Tuple[int, ...]

@dataclass
class DependencyRelation:
    head: str
    head_index: int
    relation: str
    dependent: str
    dependent_index: int


class CoreferenceResolver:
    """
    Attempts rule-based coreference resolution based on Hobbs algorithm principles.

    Limitations:
    - Requires accurate sentence splitting, tokenization, POS tagging, and ideally
      syntactic parsing (NP chunking) which are not fully implemented here.
    - Uses simplified heuristics for searching antecedents.
    - Gender/number matching is basic.
    - Does not handle complex cases like cataphora, pleonastic 'it', etc.
    """

    PRONOUNS = {
        'he': {'gender': 'male', 'number': 'singular'},
        'him': {'gender': 'male', 'number': 'singular'},
        'his': {'gender': 'male', 'number': 'singular', 'possessive': True},
        'she': {'gender': 'female', 'number': 'singular'},
        'her': {'gender': 'female', 'number': 'singular'},
        'hers': {'gender': 'female', 'number': 'singular', 'possessive': True},
        'it': {'gender': 'neuter', 'number': 'singular'},
        'its': {'gender': 'neuter', 'number': 'singular', 'possessive': True},
        'they': {'gender': 'unknown', 'number': 'plural'}, # Can be singular in modern usage
        'them': {'gender': 'unknown', 'number': 'plural'},
        'their': {'gender': 'unknown', 'number': 'plural', 'possessive': True},
        'theirs': {'gender': 'unknown', 'number': 'plural', 'possessive': True},
    }

    def __init__(self):
        try:
            self.gender_lexicon = ResourceLoader.get_gender_list()
        except Exception as e:
            logger.warning(f"[CoreferenceResolver] Failed to load gender list: {e}")
            self.gender_lexicon = {}

    def _preprocess(self, text: str) -> List[List[Tuple[str, str]]]:
        """
        Performs sentence splitting, tokenization, and POS tagging using GrammarProcessor-compatible logic.
        Returns a list of sentences, where each sentence is a list of (token, pos_tag) tuples.
        """

        structured_wordlist = ResourceLoader.get_structured_wordlist()
        grammar = GrammarProcessor(structured_wordlist=structured_wordlist)

        # Sentence splitting using basic punctuation
        sentence_boundaries = re.compile(r'(?<=[.!?])\s+')
        raw_sentences = [s.strip() for s in sentence_boundaries.split(text) if s.strip()]
        tokenized_sentences = []

        for sentence in raw_sentences:
            tokens = re.findall(r'\w+|[^\w\s]', sentence, re.UNICODE)  # Keep punctuation as tokens
            tagged_tokens = []

            for token in tokens:
                token_lower = token.lower()

                # POS tag from structured wordlist if available
                if token_lower in grammar.pos_map:
                    tag = grammar.pos_map[token_lower]
                else:
                    # Apply fallback POS pattern matching
                    tag = 'X'  # Unknown by default
                    for pattern, pos_tag in grammar.pos_patterns:
                        if pattern.fullmatch(token):
                            tag = pos_tag
                            break

                tagged_tokens.append((token, tag))

            tokenized_sentences.append(tagged_tokens)

        return tokenized_sentences

    def _find_noun_phrases(self, sentence: List[Tuple[str, str]], sentence_index: int = -1) -> List[Entity]:
        """
        Extracts Noun Phrases (NPs) using grammar rules and structured wordlist.
        Assigns gender and number based on structured lexicon or heuristics.
        """
        structured_wordlist = ResourceLoader.get_structured_wordlist()
        grammar = GrammarProcessor(structured_wordlist=structured_wordlist)

        nps = []
        current_np = []
        start_idx = None

        def flush_np():
            nonlocal current_np, start_idx
            if current_np:
                phrase = " ".join([tok for tok, _ in current_np])
                # Heuristic gender/number from structure
                gender, number, ent_type = self._infer_entity_attributes(current_np, structured_wordlist)
                token_indices = tuple(range(start_idx, start_idx + len(current_np)))
                nps.append(Entity(
                    text=phrase,
                    type=ent_type,
                    gender=gender,
                    number=number,
                    sentence_index=sentence_index,
                    token_indices=token_indices
                ))
                current_np = []
                start_idx = None

        for i, (token, pos) in enumerate(sentence):
            if pos in {'DET', 'ADJ', 'ADP', 'PROPN', 'NOUN', 'PRON', 'NUM'}:
                if start_idx is None:
                    start_idx = i
                current_np.append((token, pos))
            else:
                flush_np()
        flush_np()
        return nps

    def _infer_entity_attributes(self, np_tokens: List[Tuple[str, str]], structured_wordlist: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        Returns (gender, number, entity_type) for a given NP using structured wordlist and heuristics.
        """
        words = [tok.lower() for tok, _ in np_tokens]
        gender = 'unknown'
        number = 'singular'
        ent_type = 'OTHER'

        # Try last token as head noun
        head = words[-1]
        if head in structured_wordlist:
            pos_tags = structured_wordlist[head].get('pos', [])
            if 'noun' in pos_tags or 'proper_noun' in pos_tags:
                ent_type = 'PERSON' if head.istitle() else 'ORG/LOC'

        # Attempt better gender lookup
        if head in self.gender_lexicon:
            gender = self.gender_lexicon[head].get("gender", "unknown").lower()

        # Fallback: name-based guess
        if head.endswith("s") and not head.endswith("ss"):
            number = 'plural'

        if any(tok in {'he', 'him', 'his'} for tok in words):
            gender = 'male'
            ent_type = 'PERSON'
        elif any(tok in {'she', 'her', 'hers'} for tok in words):
            gender = 'female'
            ent_type = 'PERSON'
        elif any(tok in {'they', 'them', 'their', 'theirs'} for tok in words):
            gender = 'unknown'
            number = 'plural'
            ent_type = 'PERSON'

        return gender, number, ent_type

    def _hobbs_search(self,
                      pronoun_info: Dict,
                      pronoun_sentence_index: int,
                      pronoun_token_index: int,
                      sentences: List[List[Tuple[str, str]]]) -> Optional[Entity]:
        """
        Simplified search for an antecedent based on Hobbs algorithm steps.
        Requires parse trees for proper implementation. This uses linear search.
        """
        # Step 1-3 (Simplified): Search current and preceding sentences
        for i in range(pronoun_sentence_index, -1, -1):
            current_sentence_tokens = sentences[i]
            # Find potential antecedents (NPs) in the sentence
            # In real Hobbs, you'd traverse the parse tree here
            potential_nps = self._find_noun_phrases(current_sentence_tokens)

            # Search backwards from the pronoun position (or end of sentence if previous sentence)
            start_search_index = pronoun_token_index -1 if i == pronoun_sentence_index else len(current_sentence_tokens) - 1

            for j in range(start_search_index, -1, -1):
                 # Check if token at index j is part of a potential NP found earlier
                 for np_entity in reversed(potential_nps): # Search closest first
                    if j in np_entity.token_indices:
                        # Step 4 (Simplified): Check constraints (agreement)
                        if self._check_agreement(pronoun_info, np_entity):
                             # Found a potential match
                             # Add more checks here (binding constraints, etc.)
                            return np_entity # Return the first plausible match found searching backwards

        # Step 5+ (Simplified): If no match in preceding sentences, search wider context (not implemented here)
        return None

    def _check_agreement(self, pronoun_info: Dict, entity: Entity) -> bool:
        """Check basic gender and number agreement."""
        if pronoun_info['number'] != entity.number:
            # Basic handling for singular 'they' - could match singular non-binary or unknown gender
            if not (pronoun_info['number'] == 'plural' and pronoun_info['text'] == 'they' and entity.number == 'singular'):
                 return False
        # Allow unknown gender pronoun ('they') to match any gender entity
        if pronoun_info['gender'] != 'unknown' and entity.gender != 'unknown':
             if pronoun_info['gender'] != entity.gender:
                 return False
        return True


    def resolve(self, text: str, history: Optional[deque] = None) -> str:
        """
        Replace pronouns with the text of their most likely antecedent.

        Args:
            text: The input text potentially containing pronouns.
            history: Optional deque of previous texts/context (not fully utilized in this simplified version).

        Returns:
            Text with resolved pronouns (or original text if no resolution found).
        """
        processed_sentences = self._preprocess(text)
        resolved_text = text
        offset = 0 # Track changes in string length due to replacements

        original_tokens_flat = [token for sentence in processed_sentences for token, pos in sentence]
        current_resolved_tokens = list(original_tokens_flat) # Keep track of resolved tokens

        token_global_index = 0
        for i, sentence in enumerate(processed_sentences):
            for j, (token, pos) in enumerate(sentence):
                lower_token = token.lower()
                if lower_token in self.PRONOUNS and pos == "PRON": # Check POS tag if available
                    pronoun_info = self.PRONOUNS[lower_token]
                    pronoun_info['text'] = lower_token # Add original text for checks

                    # Search for antecedent using simplified Hobbs logic
                    antecedent = self._hobbs_search(pronoun_info, i, j, processed_sentences)

                    if antecedent:
                        # Simple replacement: Replace the pronoun token with the antecedent text
                        # A more sophisticated approach would handle possessives ('his' -> 'John's')
                        replacement_text = antecedent.text
                        if pronoun_info.get('possessive'):
                            # Basic possessive handling
                            replacement_text += "'s" if not replacement_text.endswith('s') else "'"

                        # Replace in our tracked list of tokens
                        current_resolved_tokens[token_global_index] = replacement_text
                        print(f"Resolved '{token}' to '{replacement_text}' based on antecedent '{antecedent.text}'") # Debugging

                token_global_index += 1

        # Reconstruct the text from resolved tokens (simple space join)
        # This loses original whitespace and punctuation handling - needs improvement
        resolved_text = " ".join(current_resolved_tokens)
        # Crude re-punctuation - needs significant improvement
        resolved_text = resolved_text.replace(" .", ".").replace(" ,", ",") # etc.

        return resolved_text


# --- Dependency Parsing ---

DependencyRelation = namedtuple("DependencyRelation", ["head", "head_index", "relation", "dependent", "dependent_index"])

class ShallowDependencyParser:
    """
    Outlines a shallow dependency parser based on Universal Dependencies principles.

    Limitations:
    - This is a structural outline, not a functional parser.
    - Relies heavily on placeholders for tokenization, POS tagging, and feature extraction.
    - Rule-based pattern matching is extremely simplified.
    - Real UD parsing requires complex models or extensive linguistic rules.
    """
    # Common Universal Dependencies relation types (subset)
    UD_RELATIONS = [
        'nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp', # Core arguments
        'obl', 'vocative', 'expl', 'dislocated',          # Non-core dependents
        'advcl', 'advmod', 'discourse',                   # Adverbial dependents
        'aux', 'cop', 'mark',                             # Auxiliary words
        'nmod', 'appos', 'nummod',                        # Noun dependents
        'acl', 'amod', 'det',                             # Adjective/determiner dependents
        'clf', 'case',                                    # Case marking
        'conj', 'cc',                                     # Coordination
        'fixed', 'flat', 'compound',                      # Multi-word expressions
        'list', 'parataxis',                              # Loose joining relations
        'orphan', 'goeswith', 'reparandum',               # Special relations
        'punct', 'root', 'dep'                            # Other
    ]

    def __init__(self, lang='en'):
        # Initialize grammar and linguistic resources
        self.lang = lang
        self.structured_wordlist = ResourceLoader.get_structured_wordlist()
        self.sentiment_lexicon = ResourceLoader.get_sentiment_lexicon()
        self.gender_lexicon = ResourceLoader.get_gender_list()
        self.modality_markers = ResourceLoader.get_modality_markers()
        self.grammar = GrammarProcessor(lang=self.lang, structured_wordlist=self.structured_wordlist)
        self.coref_resolver = CoreferenceResolver()

        # Initialize dependency patterns (subject → verb → object)
        self.dependency_rules = [
            ('NOUN', 'VERB', 'NOUN'),
            ('PRON', 'VERB', 'NOUN'),
            ('PROPN', 'VERB', 'NOUN'),
            ('NOUN', 'AUX', 'VERB'),
            ('PRON', 'AUX', 'VERB'),
            ('DET', 'NOUN', 'VERB'),
        ]

        # Part-of-speech clusters for simplified dependency roles
        self.pos_clusters = {
            'SUBJECT': {'PRON', 'NOUN', 'PROPN'},
            'VERB': {'VERB', 'AUX'},
            'OBJECT': {'NOUN', 'PROPN', 'PRON'},
            'MODALITY': set(self.modality_markers['epistemic'] + 
                            self.modality_markers['deontic'] +
                            self.modality_markers['dynamic']),
            'SENTIMENT': set(self.sentiment_lexicon['positive'].keys()) |
                         set(self.sentiment_lexicon['negative'].keys()) |
                         set(self.sentiment_lexicon['moderate'].keys()),
        }

        # Syntactic relation types (used for dependency labeling)
        self.relations = ['nsubj', 'dobj', 'aux', 'mod', 'det', 'advmod', 'neg', 'prep']

        # Entity buffer for context-aware parsing (used in coherence and resolution)
        self.entity_buffer = defaultdict(list)

        # Diagnostic flags
        self.enable_logging = False
        self.strict_mode = False

    def _tokenize_pos_tag(self, text: str) -> List[Dict[str, Any]]:
        """
        Tokenize and assign UPOS tags using a rule-based approach from grammar_processor.

        Returns:
            List of token dictionaries with keys:
            - id
            - text
            - lemma
            - upos
            - xpos (optional, left as default)
            - feats (optional morphological features)
        """

        grammar = GrammarProcessor()
        words = re.findall(r"\\w+|[.,!?;]", text)
        tokens = []

        for i, word in enumerate(words, start=1):
            lower = word.lower()
            upos = "X"
            lemma = lower
            feats = {}

            # POS heuristics based on suffixes and word shape
            if re.fullmatch(r"[.,!?;]", word): upos = "PUNCT"
            elif lower in {"i", "you", "he", "she", "they", "we", "it"}: upos = "PRON"
            elif lower in {"is", "are", "was", "were", "be", "am"}: upos = "AUX"
            elif lower.endswith("ing") or lower in {"have", "has", "had"}: upos = "VERB"
            elif lower.endswith("ly"): upos = "ADV"
            elif lower in {"the", "a", "an", "some"}: upos = "DET"
            elif lower in grammar._UPOS_MAP: upos = grammar._UPOS_MAP[lower]
            elif lower[0].isupper() and i == 1: upos = "PROPN"
            elif lower.endswith("ed"): upos = "VERB"
            elif lower in {"and", "or", "but"}: upos = "CCONJ"
            elif lower in {"in", "on", "at", "with", "from", "to"}: upos = "ADP"
            elif lower.isdigit(): upos = "NUM"
            else: upos = "NOUN"  # default fallback

            tokens.append({"id": i,"text": word,"lemma": lemma,"upos": upos,"xpos": None,"feats": feats})

        return tokens

    def _apply_rules(self, tokens: List[Dict[str, Any]]) -> List[DependencyRelation]:
        """
        Placeholder for applying grammatical rules or patterns to find dependencies.
        This is the core logic and the most complex part.
        Example Simplified Rules (highly inadequate for real parsing):
        """
        relations = []
        # 1. Find a potential root (often the main verb)
        root_candidate = -1
        for i, token in enumerate(tokens):
            if token['upos'] == 'VERB':
                 root_candidate = i
                 relations.append(DependencyRelation(head="ROOT", head_index=0,
                                                     relation="root",
                                                     dependent=token['text'],
                                                     dependent_index=token['id']))
                 break # Assume first verb is root

        if root_candidate != -1:
            root = tokens[root_candidate]

            # 2. Find nominal subject (nsubj) - often a NOUN before the root verb
            for i in range(root_candidate):
                 if tokens[i]['upos'] in ['NOUN', 'PROPN', 'PRON']:
                      relations.append(DependencyRelation(head=tokens[root_candidate]['text'],
                                                          head_index=tokens[root_candidate]['id'],
                                                          relation="nsubj", dependent=tokens[i]['text'],
                                                          dependent_index=tokens[i]['id']))
                      break # Assume first preceding noun/pronoun is subject

            # 3. Find direct object (obj) - often a NOUN after the root verb
            for i in range(root_candidate + 1, len(tokens)):
                 if tokens[i]['upos'] in ['NOUN', 'PROPN']:
                     relations.append(DependencyRelation(head=tokens[root_candidate]['text'],
                                                         head_index=tokens[root_candidate]['id'],
                                                         relation="obj", dependent=tokens[i]['text'],
                                                         dependent_index=tokens[i]['id']))
                     break # Assume first following noun is object

            # 4. Punctuation (attach to preceding word or root) - simplified
            for i, token in enumerate(tokens):
                if token['upos'] == 'PUNCT':
                    attach_to = root_candidate if root_candidate != -1 else len(tokens) - 2 # Fallback
                    if i > 0: attach_to = i - 1 # Attach to previous token usually
                    if attach_to >= 0:
                        relations.append(DependencyRelation(head=tokens[attach_to]['text'],
                                                            head_index=tokens[attach_to]['id'],
                                                            relation="punct", dependent=token['text'],
                                                            dependent_index=token['id']))
                            
            # 5. amod - adjectival modifier
            for i in range(1, len(tokens)):
                if tokens[i]['upos'] in ['NOUN', 'PROPN'] and tokens[i-1]['upos'] == 'ADJ':
                    relations.append(DependencyRelation(head=tokens[i]['text'],
                                                        head_index=tokens[i]['id'],
                                                        relation="amod", dependent=tokens[i-1]['text'],
                                                        dependent_index=tokens[i-1]['id']))

            # 6. advmod - adverb modifying verb
            for i in range(1, len(tokens)):
                if tokens[i]['upos'] == 'VERB' and tokens[i-1]['upos'] == 'ADV':
                    relations.append(DependencyRelation(head=tokens[i]['text'],
                                                        head_index=tokens[i]['id'],
                                                        relation="advmod", dependent=tokens[i-1]['text'],
                                                        dependent_index=tokens[i-1]['id']))

            # 7. aux - auxiliary verb before main verb
            for i in range(1, len(tokens)):
                if tokens[i]['upos'] == 'VERB' and tokens[i-1]['upos'] == 'AUX':
                    relations.append(DependencyRelation(head=tokens[i]['text'],
                                                        head_index=tokens[i]['id'],
                                                        relation="aux", dependent=tokens[i-1]['text'],
                                                        dependent_index=tokens[i-1]['id']))

            # 8. case - prepositions before nouns
            for i in range(1, len(tokens)):
                if tokens[i]['upos'] == 'NOUN' and tokens[i-1]['upos'] == 'ADP':
                    relations.append(DependencyRelation(head=tokens[i]['text'],
                                                        head_index=tokens[i]['id'],
                                                        relation="case", dependent=tokens[i-1]['text'],
                                                        dependent_index=tokens[i-1]['id']))

            # 9. punct - punctuation
            for i, token in enumerate(tokens):
                if token['upos'] == 'PUNCT':
                    attach_to = i - 1 if i > 0 else root_candidate
                    if attach_to >= 0:
                        relations.append(DependencyRelation(head=tokens[attach_to]['text'],
                                                            head_index=tokens[attach_to]['id'],
                                                            relation="punct", dependent=token['text'],
                                                            dependent_index=token['id']))

            # 10. conj - coordinated noun/verb/adjective phrases
            for i in range(1, len(tokens) - 1):
                if tokens[i]['upos'] == 'CCONJ':
                    if tokens[i-1]['upos'] == tokens[i+1]['upos'] and tokens[i-1]['upos'] in ['NOUN', 'VERB', 'ADJ']:
                        relations.append(DependencyRelation(head=tokens[i-1]['text'],
                                                            head_index=tokens[i-1]['id'],
                                                            relation="conj", dependent=tokens[i+1]['text'],
                                                            dependent_index=tokens[i+1]['id']))
            
            # 11. nmod - noun modifying another noun
            for i in range(1, len(tokens)):
                if tokens[i-1]['upos'] == 'NOUN' and tokens[i]['upos'] == 'NOUN':
                    relations.append(DependencyRelation(head=tokens[i]['text'],
                                                        head_index=tokens[i]['id'],
                                                        relation="nmod", dependent=tokens[i-1]['text'],
                                                        dependent_index=tokens[i-1]['id']))
            
            # 12. compound - compound noun modifier
            for i in range(1, len(tokens)):
                if tokens[i]['upos'] == 'NOUN' and tokens[i-1]['upos'] in ['NOUN', 'PROPN']:
                    relations.append(DependencyRelation(head=tokens[i]['text'],
                                                        head_index=tokens[i]['id'],
                                                        relation="compound", dependent=tokens[i-1]['text'],
                                                        dependent_index=tokens[i-1]['id']))
                    
            # 13. mark - subordinating conjunction
            for i in range(1, len(tokens)):
                if tokens[i]['upos'] == 'VERB' and tokens[i-1]['upos'] == 'SCONJ':
                    relations.append(DependencyRelation(head=tokens[i]['text'],
                                                        head_index=tokens[i]['id'],
                                                        relation="mark", dependent=tokens[i-1]['text'],
                                                        dependent_index=tokens[i-1]['id']))
                    
            # 14. expl - expletive 'there' or 'it'
            for i in range(len(tokens)):
                if tokens[i]['text'].lower() in ['there', 'it'] and tokens[i]['upos'] == 'PRON':
                    for j in range(i+1, len(tokens)):
                        if tokens[j]['upos'] == 'VERB':
                            relations.append(DependencyRelation(head=tokens[j]['text'],
                                                                head_index=tokens[j]['id'],
                                                                relation="expl", dependent=tokens[i]['text'],
                                                                dependent_index=tokens[i]['id']))
                            break
                            
            # 15. xcomp - open clausal complement
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] == 'VERB' and tokens[i+1]['upos'] == 'VERB':
                    if tokens[i+1]['text'].lower() in ['to', 'be', 'go', 'do', 'have'] or tokens[i+1]['upos'] == 'VERB':
                        relations.append(DependencyRelation(head=tokens[i]['text'],
                                                            head_index=tokens[i]['id'],
                                                            relation="xcomp", dependent=tokens[i+1]['text'],
                                                            dependent_index=tokens[i+1]['id']))
                        
            # 16. ccomp - clausal complement
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] == 'VERB' and tokens[i+1]['text'].lower() in ['that']:
                    for j in range(i+2, len(tokens)):
                        if tokens[j]['upos'] == 'VERB':
                            relations.append(DependencyRelation(head=tokens[i]['text'],
                                                                ead_index=tokens[i]['id'],
                                                                relation="ccomp", dependent=tokens[j]['text'],
                                                                dependent_index=tokens[j]['id']))
                            break

            # 17. discourse - discourse markers
            discourse_words = {'well', 'so', 'however', 'anyway', 'actually'}
            for i, token in enumerate(tokens):
                if token['text'].lower() in discourse_words:
                    if root_candidate != -1:
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'],
                                                            head_index=tokens[root_candidate]['id'],
                                                            relation="discourse", dependent=token['text'],
                                                            dependent_index=token['id']))
                        
            # 18. vocative - direct address (PROPN or NOUN at sentence start)
            if len(tokens) > 1 and tokens[0]['upos'] in ['PROPN', 'NOUN']:
                for i in range(1, len(tokens)):
                    if tokens[i]['upos'] == 'VERB':
                        relations.append(DependencyRelation(head=tokens[i]['text'],
                                                            head_index=tokens[i]['id'],
                                                            relation="vocative", dependent=tokens[0]['text'],
                                                            dependent_index=tokens[0]['id']))
                        break
            
            # 19. advcl - adverbial clause introduced by subordinator
            for i in range(len(tokens) - 2):
                if tokens[i]['upos'] == 'SCONJ' and tokens[i+1]['upos'] == 'PRON' and tokens[i+2]['upos'] == 'VERB':
                    relations.append(DependencyRelation(head=tokens[i+2]['text'],
                                                        head_index=tokens[i+2]['id'],
                                                        relation="advcl", dependent=tokens[i]['text'],
                                                        dependent_index=tokens[i]['id']))

            # 20. obl - nominal used with preposition
            for i in range(1, len(tokens) - 1):
                if tokens[i]['upos'] == 'ADP' and tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                    for j in range(i-1, -1, -1):
                        if tokens[j]['upos'] == 'VERB':
                            relations.append(DependencyRelation(head=tokens[j]['text'],
                                                                head_index=tokens[j]['id'],
                                                                relation="obl", dependent=tokens[i+1]['text'],
                                                                dependent_index=tokens[i+1]['id']))
                            break

            # 21. nummod - numeric modifier
            for i in range(1, len(tokens)):
                if tokens[i]['upos'] in ['NOUN', 'PROPN'] and tokens[i-1]['upos'] == 'NUM':
                    relations.append(DependencyRelation(head=tokens[i]['text'],
                                                        head_index=tokens[i]['id'],
                                                        relation="nummod", dependent=tokens[i-1]['text'],
                                                        dependent_index=tokens[i-1]['id']))

        # ... many more rules needed for other relation types (amod, advmod, case, det, etc.) ...
        # A real system uses machine learning or hundreds/thousands of rules.
        return relations

    def _detect_modality(self, tokens: List[Dict[str, Any]]) -> Optional[str]:
        """
        Detects linguistic modality of the sentence based on lexical markers.

        Returns:
            The modality label (e.g., 'epistemic', 'imperative', 'deontic', etc.) or None.
        """
        token_texts = [t["text"].lower() for t in tokens]

        for modality_type, markers in self.modality_markers.items():
            for marker in markers:
                if marker in token_texts:
                    return modality_type
        return None

    def extract_entities(self, text: str, pos_tags: List[Tuple[str, str]]) -> Dict[str, Dict[str, str]]:
        """
        Extracts basic named or noun phrase entities using POS-tag patterns.

        Returns:
            A dictionary of extracted entities with basic types (noun phrase).
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

        # Flush last entity
        if current_entity:
            key = " ".join(current_entity)
            entities[key] = {"type": "noun_phrase"}

        return entities

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parses input text and extracts shallow dependency relations.
        
        Returns:
            {
                'tokens': List of POS-tagged token dictionaries,
                'dependencies': List of DependencyRelation tuples,
                'modality': Optional detected modality (e.g., 'epistemic', 'imperative', etc.),
                'entities': Detected named or noun-phrase entities,
                'sentiment_terms': Tokens contributing to sentiment,
            }
        """
        # 1. Coreference resolution
        resolved_text = self.coref_resolver.resolve(text)

        # 2. Tokenization & POS tagging
        tokens = self._tokenize_pos_tag(resolved_text)

        # 3. Dependency parsing rules
        dependencies = self._apply_rules(tokens)

        # 4. Modality detection
        modality = self._detect_modality(tokens)

        # 5. Named/noun-phrase entity recognition
        tagged = [(t["text"], t["upos"]) for t in tokens]
        entities = self.grammar.extract_entities(resolved_text, tagged)

        # 6. Sentiment terms
        sentiment_terms = [
            t["text"] for t in tokens if t["text"].lower() in self.pos_clusters["SENTIMENT"]
        ]

        return {
            "tokens": tokens,
            "dependencies": dependencies,
            "modality": modality,
            "entities": entities,
            "sentiment_terms": sentiment_terms,
        }


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("--- Coreference Resolution Example ---")
    resolver = CoreferenceResolver()
    text1 = "John went to the park. He played fetch."
    resolved1 = resolver.resolve(text1)
    print(f"Original: {text1}")
    print(f"Resolved: {resolved1}") # Expected (simplified): John went to the park. John played fetch.

    text2 = "Sarah likes her new car. It is red."
    resolved2 = resolver.resolve(text2)
    print(f"Original: {text2}")
    print(f"Resolved: {resolved2}") # Expected (simplified): Sarah likes Sarah's new car. car is red. (Shows limitations)

    text3 = "The team celebrated their victory. They were happy."
    resolved3 = resolver.resolve(text3)
    print(f"Original: {text3}")
    print(f"Resolved: {resolved3}") # Expected (simplified): The team celebrated team's victory. team were happy.


    print("\n--- Dependency Parsing Example ---")
    parser = ShallowDependencyParser()
    text_dep = "Mary loves her cat."
    parse_result = parser.parse(text_dep)
    print(f"Parsing: '{text_dep}'")
    print("Tokens:")
    for token in parse_result['tokens']:
        print(f"  {token}")
    print("Dependencies:")
    for dep in parse_result['dependencies']:
        print(f"  {dep}")

    # Expected Output (Simplified):
    # Tokens: [{'id': 1, 'text': 'Mary', 'lemma': 'mary', 'upos': 'PROPN', ...}, {'id': 2, 'text': 'loves', 'lemma': 'loves', 'upos': 'VERB', ...}, ...]
    # Dependencies: [DependencyRelation(head='loves', head_index=2, relation='nsubj', dependent='Mary', dependent_index=1), DependencyRelation(head='loves', head_index=2, relation='obj', dependent='cat', dependent_index=4), ...]
