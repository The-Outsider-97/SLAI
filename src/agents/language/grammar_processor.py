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

from src.agents.language.utils.config_loader import load_global_config, get_config_section
from src.agents.language.utils.rules import Rules
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Grammar Processor")
printer = PrettyPrinter

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
    def __init__(self):
        self.config = load_global_config()
        self.wordlist_path = self.config.get('main_wordlist_path')

        self.grammar_config = get_config_section('grammar_processor')
        self.pos_map = self.grammar_config.get('pos_map_path')

        self.rule_engine = Rules()
        #self.rule_engine._verb_rules()
        logger.info("Grammar Processor initialized...")

    def analyze_text(self, sentences: List[List[InputToken]], full_text_snippet: Optional[str] = None) -> GrammarAnalysisResult:
        """
        Analyzes a list of sentences, where each sentence is a list of InputTokens.
        `full_text_snippet` is optional, used for context in GrammarAnalysisResult.
        """
        printer.status("INIT", "Text analyzer initialized", "info")

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

            article_issues = self._check_article_usage(sentence_tokens)
            current_sentence_issues.extend(article_issues)

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

    def _reconstruct_sentence_text(self, sentence_tokens: List[InputToken]) -> str:
        """Reconstructs sentence text from InputToken list for display."""
        printer.status("INIT", "Sentence reconstruction initialized", "info")

        if not sentence_tokens:
            return ""
        return " ".join(token.text for token in sentence_tokens)

    def _classify_sentence_type(self, sentence_tokens: List[InputToken]) -> str:
        printer.status("INIT", "Sentence classifier initialized", "info")

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

    def _check_subject_verb_agreement(self, sentence_tokens: List[InputToken]) -> List[GrammarIssue]:
        printer.status("INIT", "Verb agreement initialized", "info")

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

    def _get_token_number_and_person(self, token: InputToken) -> Tuple[Optional[str], Optional[str]]:
        """Infers number and person from token's POS and text. More heuristic than spaCy's morph."""
        printer.status("INIT", "Token number initialized", "info")

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
                if verb_lemma in self.rule_engine.irregular_singular_forms: # self.rule_engine.irregular_verbs_present_singular:
                    return self.rule_engine.irregular_singular_forms[verb_lemma] # self.rule_engine.irregular_verbs_present_singular[verb_lemma]
                if verb_lemma.endswith('y') and len(verb_lemma) > 1 and verb_lemma[-2].lower() not in 'aeiou':
                    return verb_lemma[:-1] + 'ies'
                if verb_lemma.endswith(('s', 'x', 'z', 'ch', 'sh')):
                    return verb_lemma + 'es'
                return verb_lemma + 's'
            else:
                if verb_lemma in self.rule_engine.irregular_plural_forms: # self.rule_engine.irregular_verbs_present_plural:
                    return self.rule_engine.irregular_plural_forms[verb_lemma] # self.rule_engine.irregular_verbs_present_plural[verb_lemma]
                return verb_lemma
        return verb_lemma

    def _check_article_usage(self, sentence_tokens: List[InputToken]) -> List[GrammarIssue]:
        issues = []
        words_using_a = {"university", "union", "user", "euler", "one", "once"}
        words_using_an = {"hour", "heir", "honest", "honor"}
    
        for token in sentence_tokens:
            if token.pos == 'DET' and token.text.lower() in ['a', 'an']:
                head_noun = next((t for t in sentence_tokens if t.index == token.head), None)
                if not head_noun:
                    continue
                    
                head_text = head_noun.text.lower()
                if not head_text:
                    continue
                    
                first_char = head_text[0]
                
                if token.text.lower() == 'a':
                    if first_char in 'aeiou' and head_text not in words_using_a:
                        issues.append(GrammarIssue(
                            description=f"Incorrect article: 'a' used before vowel sound '{head_text}'",
                            source_text_char_span=(token.start_char_abs, token.end_char_abs),
                            source_sentence_token_indices_span=(token.index, token.index),
                            severity="error",
                            suggestion="Use 'an' instead"
                        ))
                    elif head_text in words_using_an:
                        issues.append(GrammarIssue(
                            description=f"Article exception: 'a' used before '{head_text}'",
                            source_text_char_span=(token.start_char_abs, token.end_char_abs),
                            source_sentence_token_indices_span=(token.index, token.index),
                            severity="error",
                            suggestion="Use 'an' instead"
                        ))
                        
                elif token.text.lower() == 'an':
                    if first_char not in 'aeiou' and head_text not in words_using_an:
                        issues.append(GrammarIssue(
                            description=f"Incorrect article: 'an' used before consonant sound '{head_text}'",
                            source_text_char_span=(token.start_char_abs, token.end_char_abs),
                            source_sentence_token_indices_span=(token.index, token.index),
                            severity="error",
                            suggestion="Use 'a' instead"
                        ))
        return issues

if __name__ == "__main__":
    print("\n=== Running Grammar Processor ===\n")
    printer.status("Init", "Grammar Processor initialized", "success")

    processor = GrammarProcessor()

    print(processor)

    print("\n* * * * * Phase 2 * * * * *\n")
    # text="I don't want this to become normal, I want this to be stop!"
    # text=[[InputToken(text="I don't want this to become normal, I want this to be stop!")]]
    sentences = [[
        InputToken(text="I", lemma="I", pos="PRON", index=0, head=1, dep="nsubj", start_char_abs=0, end_char_abs=0),
        InputToken(text="do", lemma="do", pos="VERB", index=11, head=8, dep="xcomp", start_char_abs=45, end_char_abs=46),
        InputToken(text="not", lemma="not", pos="VERB", index=11, head=8, dep="xcomp", start_char_abs=45, end_char_abs=46),
        InputToken(text="want", lemma="want", pos="VERB", index=1, head=1, dep="ROOT", start_char_abs=2, end_char_abs=5),
        InputToken(text="this", lemma="this", pos="DET", index=2, head=3, dep="det", start_char_abs=7, end_char_abs=10),
        InputToken(text="to", lemma="to", pos="PART", index=3, head=4, dep="aux", start_char_abs=12, end_char_abs=13),
        InputToken(text="become", lemma="become", pos="VERB", index=4, head=1, dep="xcomp", start_char_abs=15, end_char_abs=20),
        InputToken(text="normal", lemma="normal", pos="ADJ", index=5, head=4, dep="attr", start_char_abs=22, end_char_abs=27),
        InputToken(text=",", lemma=",", pos="PUNCT", index=6, head=1, dep="punct", start_char_abs=28, end_char_abs=28),
        InputToken(text="I", lemma="I", pos="PRON", index=7, head=8, dep="nsubj", start_char_abs=30, end_char_abs=30),
        InputToken(text="want", lemma="want", pos="VERB", index=8, head=1, dep="conj", start_char_abs=32, end_char_abs=35),
        InputToken(text="this", lemma="this", pos="DET", index=9, head=10, dep="det", start_char_abs=37, end_char_abs=40),
        InputToken(text="to", lemma="to", pos="PART", index=10, head=11, dep="aux", start_char_abs=42, end_char_abs=43),
        InputToken(text="stop", lemma="stop", pos="VERB", index=12, head=11, dep="xcomp", start_char_abs=48, end_char_abs=51),
        InputToken(text="?", lemma="?", pos="PUNCT", index=13, head=8, dep="punct", start_char_abs=52, end_char_abs=52),
    ]]
    full_text_snippet= None
    sentence_tokens=[]
    printer.pretty("ANALYZE", processor.analyze_text(sentences=sentences, full_text_snippet=full_text_snippet), "success")
    printer.status("RECON", processor._reconstruct_sentence_text(sentence_tokens=sentence_tokens), "success")

    print("\n* * * * * Phase 3 * * * * *\n")
    print("\n=== Grammar Processor Tests Complete ===\n")
