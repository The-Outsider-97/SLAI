import re
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import namedtuple, deque
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

    def __init__(self):
        # Load linguistic resources if needed (e.g., lexicons, rules)
        pass

    def _tokenize_pos_tag(self, text: str) -> List[Dict[str, Any]]:
        """
        Placeholder for tokenization and POS tagging.
        Should return a list of tokens, each represented as a dictionary
        with keys like 'id', 'text', 'lemma', 'upos', 'xpos', 'feats'.
        """
        # Simplified: Split by space, assign basic info
        tokens = []
        words = text.split()
        for i, word in enumerate(words):
            tokens.append({
                'id': i + 1,
                'text': word,
                'lemma': word.lower(), # Simplistic lemmatization
                'upos': 'NOUN' if word[0].isupper() else 'VERB' if i==1 else 'PROPN' if i==0 else 'PUNCT' if word=='.' else 'PART', # Very basic POS
                'xpos': None,
                'feats': None,
                'head': 0, # Placeholder, to be filled
                'deprel': '_', # Placeholder, to be filled
            })
        return tokens

    def _apply_rules(self, tokens: List[Dict[str, Any]]) -> List[DependencyRelation]:
        """
        Placeholder for applying grammatical rules or patterns to find dependencies.
        This is the core logic and the most complex part.
        """
        relations = []
        # Example Simplified Rules (highly inadequate for real parsing):
        # 1. Find a potential root (often the main verb)
        root_candidate = -1
        for i, token in enumerate(tokens):
            if token['upos'] == 'VERB':
                 root_candidate = i
                 relations.append(DependencyRelation(head="ROOT",
                                                     head_index=0,
                                                     relation="root",
                                                     dependent=token['text'],
                                                     dependent_index=token['id']))
                 break # Assume first verb is root

        if root_candidate != -1:
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

        # ... many more rules needed for other relation types (amod, advmod, case, det, etc.) ...
        # A real system uses machine learning or hundreds/thousands of rules.

        return relations

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parses the text to extract shallow dependency relations.

        Args:
            text: The input text.

        Returns:
            A dictionary containing the list of tokens (with POS/features)
            and a list of identified dependency relations.
        """
        # 1. Tokenize and POS Tag (using placeholder)
        tokens = self._tokenize_pos_tag(text)

        # 2. Apply rules/patterns (using placeholder)
        dependencies = self._apply_rules(tokens)

        # Update token dictionaries with head/relation info (optional)
        # ... logic to map dependencies back to token['head'] and token['deprel'] ...

        return {
            'tokens': tokens, # List of token dictionaries
            'dependencies': dependencies # List of DependencyRelation namedtuples
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
