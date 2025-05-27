"""
Core Function:
Analyzes syntax and structure of the input to validate grammar, detect mistakes, or help with reformulations.

Responsibilities:
Perform syntax parsing (with help from NLPEngine)
Apply grammar rule checks (e.g., subject-verb agreement)
Suggest corrections or reformulate malformed input
Classify sentence type (declarative, interrogative, etc.)

Why it matters:
Adds grammatical intelligence, improves clarity, and can assist both interpretation (for ambiguous input) and generation (polished output).
"""
import yaml

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

from src.agents.language.utils.rules import Rules #, DependencyRelation
from logs.logger import get_logger

logger = get_logger("Grammar Processor")

CONFIG_PATH = "src/agents/language/configs/language_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

@dataclass
class InputToken:
    """
    Expected structure for tokens passed to GrammarProcessor.
    This should be constructed by the main agent pipeline from NLPEngine's output.
    """
    text: str
    lemma: str
    pos: str      # Universal POS tag (e.g., 'VERB', 'NOUN', 'PRON')
    index: int    # Index of the token within its sentence (0-based)
    head: int     # Index of the head token in the sentence (-1 or self.index for root)
    dep: str      # Dependency relation to the head (e.g., 'nsubj', 'obj', 'aux')
    start_char_abs: int # Absolute start char offset in the original full text
    end_char_abs: int   # Absolute end char offset in the original full text (inclusive)

@dataclass
class GrammarIssue:
    description: str
    # Character indices in the original full text
    source_text_char_span: Tuple[int, int]
    # Inclusive start and end indices of relevant tokens within their sentence
    source_sentence_token_indices_span: Tuple[int, int]
    severity: str = "warning" # e.g., "warning", "error"
    suggestion: Optional[str] = None

@dataclass
class GrammarAnalysisResult:
    original_text_snippet: str # Could be the full text or a relevant snippet
    is_grammatical: bool
    sentence_analyses: List[Dict[str, Any]]
    # Each dict in sentence_analyses:
    #   - text: str (the sentence text, reconstructed from input tokens)
    #   - type: str (declarative, interrogative, etc.)
    #   - issues: List[GrammarIssue]
    #   - original_sentence_token_count: int

class GrammarProcessor:
    # _UPOS_MAP could be used for internal normalization if NLPEngine POS tags vary
    _UPOS_MAP = {
        'noun': 'NOUN', 'propn': 'PROPN', 'verb': 'VERB', 'aux': 'AUX',
        'adjective': 'ADJ', 'adverb': 'ADV', 'numeral': 'NUM',
        'determiner': 'DET', 'pronoun': 'PRON', 'preposition': 'ADP',
        'conjunction': 'CCONJ', 'subord': 'SCONJ', 'particle': 'PART',
        'interjection': 'INTJ', 'symbol': 'SYM', 'punct': 'PUNCT',
    }

    def __init__(self, config):

        self.config = config
        self.rule_engine = Rules()
        self.rule_engine._verb_rules()
        logger.info("Grammar Processor initialized...")

    def _reconstruct_sentence_text(self, sentence_tokens: List[InputToken]) -> str:
        """Reconstructs sentence text from InputToken list for display."""
        if not sentence_tokens:
            return ""
        return " ".join(token.text for token in sentence_tokens)


    def analyze_text(self, sentences: List[List[InputToken]], full_text_snippet: Optional[str] = None) -> GrammarAnalysisResult:
        """
        Analyzes a list of sentences, where each sentence is a list of InputTokens.
        `full_text_snippet` is optional, used for context in GrammarAnalysisResult.
        """
        if not sentences:
            logger.warning("Received empty sentence list for analysis.")
            return GrammarAnalysisResult(original_text_snippet=full_text_snippet or "",
                                         is_grammatical=True, sentence_analyses=[])

        all_issues_overall: List[GrammarIssue] = []
        sentence_analyses_results: List[Dict[str, Any]] = []

        for sentence_tokens in sentences:
            if not sentence_tokens:
                continue

            current_sentence_issues: List[GrammarIssue] = []
            sentence_text_reconstructed = self._reconstruct_sentence_text(sentence_tokens)
            
            sentence_type = self._classify_sentence_type(sentence_tokens)
            
            sv_agreement_issues = self._check_subject_verb_agreement(sentence_tokens)
            current_sentence_issues.extend(sv_agreement_issues)
            
            # TODO: Add calls to other grammar check methods here using sentence_tokens

            all_issues_overall.extend(current_sentence_issues)
            sentence_analyses_results.append({
                "text": sentence_text_reconstructed,
                "type": sentence_type,
                "issues": current_sentence_issues,
                "original_sentence_token_count": len(sentence_tokens)
            })

        is_overall_grammatical = not any(issue.severity == "error" for issue_list in sentence_analyses_results for issue in issue_list["issues"])
        
        display_text = full_text_snippet
        if not display_text and sentences and sentences[0]:
             display_text = self._reconstruct_sentence_text(sentences[0])
             if len(sentences) > 1: display_text += "..."


        return GrammarAnalysisResult(
            original_text_snippet=display_text or "N/A",
            is_grammatical=is_overall_grammatical,
            sentence_analyses=sentence_analyses_results
        )

    def _classify_sentence_type(self, sentence_tokens: List[InputToken]) -> str:
        if not sentence_tokens:
            return "EMPTY"

        last_token = sentence_tokens[-1]
        if last_token.text == '?':
            return 'INTERROGATIVE'
        if last_token.text == '!':
            return 'EXCLAMATORY'

        first_token = sentence_tokens[0]
        # Imperative: often starts with a base form verb, subject is implied 'you'
        if first_token.pos == 'VERB' and first_token.lemma == first_token.text: # Heuristic for base form
            # Ensure it's not part of a question ("Do you...")
            is_aux_question_start = (first_token.pos == 'AUX' and len(sentence_tokens) > 1 and
                                     any(tok.dep == 'nsubj' and tok.head == first_token.index for tok in sentence_tokens[1:]))
            if not is_aux_question_start:
                # Check if there's an explicit subject
                has_explicit_subject = any(tok.dep == 'nsubj' for tok in sentence_tokens)
                if not has_explicit_subject:
                    return 'IMPERATIVE'
        
        # WH-questions (heuristic, might misclassify WH-clauses in declaratives)
        wh_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which'}
        if first_token.lemma.lower() in wh_words and last_token.text != '.': # Avoid "Who knows."
             # A more robust check would involve looking at the main verb structure
             return 'INTERROGATIVE'


        return 'DECLARATIVE'

    def _get_token_number_and_person(self, token: InputToken) -> Tuple[Optional[str], Optional[str]]:
        """Infers number and person from token's POS and text. More heuristic than spaCy's morph."""
        number: Optional[str] = None
        person: Optional[str] = None
        
        # Pronoun checks (most reliable for person/number without full morphology)
        if token.pos == 'PRON' or token.pos.startswith('PRP'): # PRP is Penn Treebank for Pronoun
            text_lower = token.text.lower()
            if text_lower in ["i", "me", "myself"]:
                number, person = "singular", "1st"
            elif text_lower in ["we", "us", "ourselves"]:
                number, person = "plural", "1st"
            elif text_lower in ["you", "yourself", "yourselves"]:
                person = "2nd"
                number = "singular_or_plural" # 'you' is ambiguous
            elif text_lower in ["he", "him", "himself", "she", "her", "herself", "it", "itself"]:
                number, person = "singular", "3rd"
            elif text_lower in ["they", "them", "themselves"]:
                number, person = "plural", "3rd"

        # Noun checks for number
        elif token.pos == 'NOUN' or token.pos == 'PROPN':
            if token.pos == 'NNS' or token.pos == 'NNPS': # Penn Treebank plural tags
                number = "plural"
            elif token.text.lower().endswith('s') and not token.lemma.lower().endswith('s') and token.text.lower() != token.lemma.lower() + 's':
                # Heuristic: ends in 's', lemma doesn't, not just lemma+'s' (e.g. "bus" vs "buses")
                # This is imperfect (e.g., "series", "news")
                if token.text.lower() not in ["series", "news", "mathematics", "physics"]: # common exceptions
                     number = "plural"
                else:
                     number = "singular" # assume singular for exceptions
            else:
                number = "singular"
            person = "3rd" # Nouns are typically 3rd person

        # Default to 3rd person singular if unknown, a common case
        if number is None: number = "singular"
        if person is None: person = "3rd"
        
        return number, person

    def _suggest_verb_form(self, verb_lemma: str, subject_number: Optional[str], subject_person: Optional[str], verb_tense: str = "present") -> str:
        if verb_tense.lower() == "present":
            if subject_number == "singular" and subject_person == "3rd":
                if verb_lemma in self.rule_engine.irregular_verbs_present_singular:
                    return self.rule_engine.irregular_verbs_present_singular[verb_lemma]
                if verb_lemma.endswith('y') and len(verb_lemma) > 1 and verb_lemma[-2].lower() not in 'aeiou':
                    return verb_lemma[:-1] + 'ies'
                if verb_lemma.endswith(('s', 'x', 'z', 'ch', 'sh')):
                    return verb_lemma + 'es'
                return verb_lemma + 's'
            else:
                if verb_lemma in self.rule_engine.irregular_verbs_present_plural:
                    return self.rule_engine.irregular_verbs_present_plural[verb_lemma]
                return verb_lemma
        return verb_lemma

    def _check_subject_verb_agreement(self, sentence_tokens: List[InputToken]) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []
        
        for token_idx, current_token in enumerate(sentence_tokens):
            # Consider VERB and AUX as potential main verbs of a clause
            if current_token.pos in ('VERB', 'AUX'):
                verb = current_token
                
                # Find subjects of this verb using dependency relations
                # A subject's head will be the verb, and its dep relation 'nsubj' or 'nsubjpass'
                subjects = [
                    s_tok for s_tok in sentence_tokens 
                    if s_tok.head == verb.index and s_tok.dep in ("nsubj", "nsubjpass")
                ]

                for subject in subjects:
                    # Basic check for present tense verbs (more complex for past/perfect etc.)
                    # This assumes NLPEngine provides good POS and Lemma.
                    # A simple heuristic: if the verb is not lemmatized to 'be'/'have' and is in base form or ends with 's'
                    is_present_tense_candidate = True # Assume present unless clear indicators otherwise
                    if verb.pos == 'VERB':
                        if verb.lemma == verb.text and not verb.text.endswith('s'): # Base form, e.g. "go"
                            pass
                        elif verb.text.endswith('s') and verb.lemma + 's' == verb.text: # e.g. "goes"
                            pass
                        elif verb.lemma == "be" and verb.text.lower() not in ["is", "are", "am"]: # past forms of be
                            is_present_tense_candidate = False
                        elif verb.lemma == "have" and verb.text.lower() not in ["has", "have"]: # past form had
                            is_present_tense_candidate = False
                        # Add more heuristics if needed, or rely on more detailed tense info from NLPEngine
                        # if NLPEngine provided `token.tense == 'PAST'`, this would be easier.

                    if not is_present_tense_candidate:
                        continue

                    subj_number, subj_person = self._get_token_number_and_person(subject)
                    verb_text_lower = verb.text.lower()
                    
                    agreement_error = False
                    suggested_verb = verb.text # Default to current form

                    if subj_number == "singular" and subj_person == "3rd":
                        expected_form = self._suggest_verb_form(verb.lemma, "singular", "3rd")
                        if verb_text_lower != expected_form.lower():
                            # Specific check for "be": "he are" -> "he is"
                            if verb.lemma == "be" and verb_text_lower != "is":
                                agreement_error = True
                            # Check for common regular verb errors: "he go" -> "he goes"
                            elif verb.lemma != "be" and verb.text == verb.lemma: # base form used
                                agreement_error = True
                            # Check for "he don't"
                            elif verb.lemma == "do" and verb_text_lower == "don't":
                                agreement_error = True

                    elif subj_number == "plural" or \
                         (subj_number == "singular_or_plural" and subj_person == "2nd") or \
                         (subj_number == "singular" and subj_person == "1st"):
                        expected_form = self._suggest_verb_form(verb.lemma, subj_number, subj_person)
                        if verb_text_lower != expected_form.lower():
                            # Specific check for "be": "they is" -> "they are", "I is" -> "I am"
                            if verb.lemma == "be":
                                if subj_person == "1st" and subj_number == "singular" and verb_text_lower != "am":
                                    agreement_error = True
                                elif not (subj_person == "1st" and subj_number == "singular") and verb_text_lower != "are":
                                    agreement_error = True
                            # Check for common regular verb errors: "they goes" -> "they go"
                            elif verb.lemma != "be" and verb_text_lower.endswith('s') and verb.lemma + 's' == verb_text_lower:
                                agreement_error = True
                    
                    if agreement_error:
                        suggested_verb = self._suggest_verb_form(verb.lemma, subj_number, subj_person)
                        issues.append(GrammarIssue(
                            description=f"Potential subject-verb agreement error: Subject '{subject.text}' "
                                        f"({subj_number or 'unknown'}/{subj_person or 'unknown'}) "
                                        f"with verb '{verb.text}'.",
                            source_text_char_span=(verb.start_char_abs, verb.end_char_abs),
                            source_sentence_token_indices_span=(subject.index, verb.index), # span from subj to verb
                            severity="error",
                            suggestion=f"Consider using '{suggested_verb}'."
                        ))
        return issues

# Example usage (requires manual setup of InputToken list):
if __name__ == "__main__":
    logger.info("Running GrammarProcessor (spaCy-free) standalone example...")
    
    test_config_data = load_config() # Load from actual config file
    gp = GrammarProcessor(config=test_config_data)

    # --- Helper to simulate NLPEngine output for testing ---
    # In a real scenario, NLPEngine would produce this.
    # This is a very basic mock dependency "parser".
    def mock_tokenize_and_parse(text: str) -> List[InputToken]:
        words = text.split()
        tokens = []
        char_offset = 0
        # Simple POS and lemma, basic head assignment (verb is root, others attach to previous or verb)
        # THIS IS HIGHLY SIMPLIFIED AND NOT A REAL PARSER.
        root_index = -1
        for i, word_text in enumerate(words):
            # Rudimentary POS tagging for testing
            pos = "NOUN"
            lemma = word_text.lower()
            if word_text.lower() in ["is", "are", "was", "were", "am", "has", "have", "do", "does", "goes"]:
                pos = "VERB" # Actually AUX or VERB
                if root_index == -1: root_index = i
            elif word_text.lower() in ["cat", "cats", "fish", "apples", "park", "system", "data", "committee", "friends", "concert"]:
                pos = "NOUN"
            elif word_text.lower() in ["i", "she", "they", "he", "my"]:
                pos = "PRON"
            elif word_text.endswith("?") or word_text.endswith(".") or word_text.endswith("!"):
                pos = "PUNCT"
            
            # Lemma for common irregulars
            if word_text.lower() == "eats": lemma = "eat"
            if word_text.lower() == "goes": lemma = "go"
            if word_text.lower() == "is": lemma = "be"
            if word_text.lower() == "are": lemma = "be"
            if word_text.lower() == "don't": lemma = "do" # and "not" implicitly


            # Rudimentary dependency head assignment (highly simplified)
            head_idx = -1
            dep_rel = "dep" # generic dependency
            if pos == "VERB" and i == root_index : # Main verb is root
                 head_idx = i # root points to self or -1
                 dep_rel = "root"
            elif pos == "NOUN" or pos == "PRON":
                if root_index != -1:
                    head_idx = root_index
                    dep_rel = "nsubj" if i < root_index else "obj" # very naive
                else: # No verb found yet, attach to previous if exists
                    head_idx = i-1 if i > 0 else i
            elif pos == "PUNCT":
                head_idx = i-1 if i > 0 else i
                dep_rel = "punct"
            else: # Other tags
                head_idx = i-1 if i > 0 else i # attach to previous

            if i == 0 and dep_rel != "root": # first word cannot attach to -1
                if root_index == -1 or root_index == 0 : # if it is the root or no root found yet
                    head_idx = i
                    if dep_rel != "root": dep_rel = "root" # make it root if not already
                elif root_index > 0 :
                    head_idx = root_index


            start_char = char_offset
            end_char = char_offset + len(word_text) - 1
            tokens.append(InputToken(
                text=word_text, lemma=lemma, pos=pos, index=i,
                head=head_idx, dep=dep_rel,
                start_char_abs=start_char, end_char_abs=end_char
            ))
            char_offset += len(word_text) + 1 # Add 1 for space
        return tokens
    # --- End of mock helper ---

    test_sentences_text = [
        "The quick brown fox jumps over the lazy dog.",
        "Cats eats fish.", # Error
        "What time is it?",
        "Go home now!",
        "She like apples.", # Error
        "They goes to the park.", # Error
        "I am happy.",
        "An error occur in the system.", # Error
        "Data were processed.",
        "The committee decide.", # Error
        "My friends and I is going to the concert.", # Error (complex subject, simplified mock won't catch well)
        "He don't know." # Error
    ]

    for i, text_example in enumerate(test_sentences_text):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Input Text: {text_example}")
        
        # Simulate NLPEngine providing a list of sentences (here, each test is one sentence)
        sentence_as_input_tokens = [mock_tokenize_and_parse(text_example)]
        
        analysis_result = gp.analyze_text(sentence_as_input_tokens, full_text_snippet=text_example)
        
        if analysis_result:
            print(f"Overall Grammatical: {analysis_result.is_grammatical}")
            for sent_analysis in analysis_result.sentence_analyses:
                print(f"  Sentence: '{sent_analysis['text']}'")
                print(f"  Type: {sent_analysis['type']}")
                if sent_analysis['issues']:
                    print("  Issues Found:")
                    for issue in sent_analysis['issues']:
                        issue_text_snippet = text_example[issue.source_text_char_span[0] : issue.source_text_char_span[1]+1]
                        print(f"    - Desc: {issue.description}")
                        print(f"      Affected Text: '{issue_text_snippet}', Severity: {issue.severity}")
                        if issue.suggestion:
                            print(f"      Suggestion: {issue.suggestion}")
                else:
                    print("  No issues found in this sentence.")
        else:
            print("Analysis failed.")

    print("\nStandalone example finished.")
