
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class DependencyRelation:
    head: str
    head_index: int
    relation: str
    dependent: str
    dependent_index: int

class Rules():
    def __init__(self):
        self._verb_rules()

    def _verb_rules(self):
        self.irregular_verbs_present_singular = {
            "be": "is", "have": "has", "do": "does", "go": "goes", "say": "says",
            "undo": "undoes", "misdo": "misdoes", "overdo": "overdoes",
            "fly": "flies", "try": "tries", "cry": "cries",
            "echo": "echoes", "veto": "vetoes", "torpedo": "torpedoes",
            "forgo": "forgoes", "outdo": "outdoes", "undergo": "undergoes",
            "knife": "knifes", "life": "lives", "strife": "strives"  # Edge cases with 'f' to 'ves'
        }
        
        self.irregular_verbs_present_plural = {
            "be": "are", "have": "have", "do": "do", "go": "go", "say": "say",
            "undo": "undo", "misdo": "misdo", "overdo": "overdo",
            "fly": "fly", "try": "try", "cry": "cry",
            "echo": "echo", "veto": "veto", "torpedo": "torpedo",
            "forgo": "forgo", "outdo": "outdo", "undergo": "undergo",
            "knife": "knife", "life": "life", "strife": "strife"
        }

    @property
    def irregular_singular_forms(self):
        return self.irregular_verbs_present_singular

    @property
    def irregular_plural_forms(self):
        return self.irregular_verbs_present_plural

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

            # Placeholder for the new rules. Assume 'relations' list and 'tokens' list are available.
            # 'root_candidate' may or may not be -1. Rules should handle this if they depend on it.
            # Most local rules (e.g., within a noun phrase) can operate independently of the main root.

            # --- Determiners and Possessives ---
            # Rule 22: det - Determiner (DET) preceding a NOUN/PROPN.
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] == 'DET' and tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                    relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                        relation="det", dependent=tokens[i]['text'],
                                                        dependent_index=tokens[i]['id']))

            # Rule 23: det:poss - Possessive pronoun (PRON with Poss=Yes feature, approximated here by common possessives) as determiner.
            possessive_pronouns = {'my', 'your', 'his', 'her', 'its', 'our', 'their'}
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] == 'PRON' and tokens[i]['text'].lower() in possessive_pronouns and \
                   tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                    relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                        relation="det:poss", dependent=tokens[i]['text'],
                                                        dependent_index=tokens[i]['id']))
            
            # Rule 24: nmod:poss - Possessive marking ('s or of NOUN). Simplified for 's.
            # Assumes 's is a separate token with POS 'PART' or similar (e.g. from some tokenizers). Or attaches to preceding NOUN.
            # If 's is PART after a NOUN, and followed by another NOUN. head is the second NOUN.
            for i in range(len(tokens) - 2):
                if tokens[i]['upos'] in ['NOUN', 'PROPN'] and \
                   tokens[i+1]['text'] == "'s" and tokens[i+1]['upos'] == 'PART' and \
                   tokens[i+2]['upos'] in ['NOUN', 'PROPN']:
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'], # owner is head
                                                        relation="nmod:poss", dependent=tokens[i+2]['text'], # owned is dependent
                                                        dependent_index=tokens[i+2]['id']))
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'], # 's marks possession on owner
                                                        relation="case", dependent=tokens[i+1]['text'],
                                                        dependent_index=tokens[i+1]['id']))


            # --- Copula and Auxiliaries ---
            # Rule 25: cop - Copula verb (AUX 'be', 'seem') with a non-verbal predicate. Copula is dependent on predicate.
            copula_verbs = {'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'seem', 'seems', 'seemed', 'appear', 'appears', 'appeared'}
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] == 'AUX' and tokens[i]['text'].lower() in copula_verbs and \
                   tokens[i+1]['upos'] in ['NOUN', 'ADJ', 'PRON', 'PROPN', 'ADV', 'NUM']:
                    relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                        relation="cop", dependent=tokens[i]['text'],
                                                        dependent_index=tokens[i]['id']))

            # Rule 26: aux:pass - Passive auxiliary ('be' forms) with a VERB (typically past participle).
            passive_aux_verbs = {'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being'}
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] == 'AUX' and tokens[i]['text'].lower() in passive_aux_verbs and \
                   tokens[i+1]['upos'] == 'VERB':
                    # Simple check, could be improved by checking verb form if available (e.g. 'VBN')
                    relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                        relation="aux:pass", dependent=tokens[i]['text'],
                                                        dependent_index=tokens[i]['id']))

            # Rule 27: nsubj:pass - Nominal subject of a passive verb.
            for i in range(len(tokens) - 2):
                if tokens[i]['upos'] in ['NOUN', 'PROPN', 'PRON'] and \
                   tokens[i+1]['upos'] == 'AUX' and tokens[i+1]['text'].lower() in passive_aux_verbs and \
                   tokens[i+2]['upos'] == 'VERB':
                    relations.append(DependencyRelation(head=tokens[i+2]['text'], head_index=tokens[i+2]['id'],
                                                        relation="nsubj:pass", dependent=tokens[i]['text'],
                                                        dependent_index=tokens[i]['id']))

            # Rule 28: aux - Modal auxiliaries (e.g., 'can', 'will', 'should') before a VERB.
            modal_aux_verbs = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'}
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] == 'AUX' and tokens[i]['text'].lower() in modal_aux_verbs and \
                   tokens[i+1]['upos'] == 'VERB':
                    relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                        relation="aux", dependent=tokens[i]['text'],
                                                        dependent_index=tokens[i]['id']))
            
            # Rule 29: aux - 'do/does/did' as auxiliary.
            do_aux_verbs = {'do', 'does', 'did'}
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] == 'AUX' and tokens[i]['text'].lower() in do_aux_verbs and \
                   tokens[i+1]['upos'] == 'VERB':
                    relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                        relation="aux", dependent=tokens[i]['text'],
                                                        dependent_index=tokens[i]['id']))

            # --- Subjects and Objects Variants ---
            # Rule 30: iobj - Indirect object (NOUN/PRON between VERB and another NOUN/PRON (direct object)).
            if root_candidate != -1:
                for i in range(root_candidate + 1, len(tokens) - 1):
                    if tokens[i]['upos'] in ['NOUN', 'PRON'] and tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                        # This is a simplification: VERB iobj dobj
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="iobj", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))
                        # Assuming the original obj rule (rule 3) would then pick tokens[i+1] as obj. This might conflict.
                        # A better iobj would be VERB dobj PREP iobj (e.g. "gave the book to Mary") handled by obl.
                        # Or "gave Mary the book" -> iobj:Mary, obj:book
                        break # Assume first one is iobj

            # Rule 31: csubj - Clausal subject (SCONJ + VERB phrase acting as subject, before main verb).
            if root_candidate > 1: # Need at least SCONJ + VERB before main verb
                if tokens[0]['upos'] == 'SCONJ' and tokens[1]['upos'] == 'VERB':
                     relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                         relation="csubj", dependent=tokens[1]['text'], # The verb of the clause
                                                         dependent_index=tokens[1]['id']))
                     relations.append(DependencyRelation(head=tokens[1]['text'], head_index=tokens[1]['id'],
                                                         relation="mark", dependent=tokens[0]['text'], # The SCONJ marking the clause
                                                         dependent_index=tokens[0]['id']))

            # Rule 32: obj - Pronoun as direct object.
            if root_candidate != -1:
                for i in range(root_candidate + 1, len(tokens)):
                    if tokens[i]['upos'] == 'PRON':
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="obj", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))
                        break # Assume first following pronoun is object (if no NOUN object found by rule 3)
            
            # Rule 33: nsubj - Subject appearing after the verb (e.g., in questions or VSO structures).
            if root_candidate != -1 and root_candidate < len(tokens) - 1:
                # Example: Is (VERB) he (PRON) ...?  or  Said (VERB) the man (NOUN)...
                if tokens[root_candidate+1]['upos'] in ['NOUN', 'PROPN', 'PRON']:
                    # Avoid if an nsubj before verb is already found by rule 2
                    is_subj_already_found = any(r.relation == "nsubj" and r.head_index == tokens[root_candidate]['id'] for r in relations)
                    if not is_subj_already_found:
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="nsubj", dependent=tokens[root_candidate+1]['text'],
                                                            dependent_index=tokens[root_candidate+1]['id']))

            # --- Modifiers (Nominal, Adjectival, Adverbial) ---
            # Rule 34: amod - Adjective after NOUN.
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] in ['NOUN', 'PROPN'] and tokens[i+1]['upos'] == 'ADJ':
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="amod", dependent=tokens[i+1]['text'],
                                                        dependent_index=tokens[i+1]['id']))

            # Rule 35: advmod - Adverb modifying ADJ.
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] == 'ADV' and tokens[i+1]['upos'] == 'ADJ':
                    relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                        relation="advmod", dependent=tokens[i]['text'],
                                                        dependent_index=tokens[i]['id']))

            # Rule 36: advmod - Adverb modifying ADV.
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] == 'ADV' and tokens[i+1]['upos'] == 'ADV':
                    relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                        relation="advmod", dependent=tokens[i]['text'],
                                                        dependent_index=tokens[i]['id']))
            
            # Rule 37: advmod:neg - Negation particle 'not' or "n't" (as PART).
            negation_particles = {'not', "n't"}
            for i in range(len(tokens) -1):
                # if 'not' (ADV) or "n't" (PART) modifies a VERB or AUX or ADJ
                if (tokens[i]['text'].lower() in negation_particles and tokens[i]['upos'] in ['ADV', 'PART']) and \
                    tokens[i+1]['upos'] in ['VERB', 'AUX', 'ADJ']:
                     relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                        relation="advmod", dependent=tokens[i]['text'], # UD uses advmod for not
                                                        dependent_index=tokens[i]['id']))
                # if 'not' or "n't" follows aux/verb e.g. "is not", "can't"
                elif i > 0 and (tokens[i]['text'].lower() in negation_particles and tokens[i]['upos'] in ['ADV', 'PART']) and \
                    tokens[i-1]['upos'] in ['VERB', 'AUX']:
                     relations.append(DependencyRelation(head=tokens[i-1]['text'], head_index=tokens[i-1]['id'],
                                                        relation="advmod", dependent=tokens[i]['text'],
                                                        dependent_index=tokens[i]['id']))


            # Rule 38: nummod - Numeric modifier after NOUN (e.g., "chapter 7").
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] in ['NOUN', 'PROPN'] and tokens[i+1]['upos'] == 'NUM':
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="nummod", dependent=tokens[i+1]['text'],
                                                        dependent_index=tokens[i+1]['id']))

            # Rule 39: nmod:tmod - Temporal noun phrase as modifier (simplified: common temporal nouns modifying root).
            if root_candidate != -1:
                temporal_nouns = {"today", "yesterday", "tomorrow", "morning", "afternoon", "evening", "night", "week", "month", "year"}
                for i, token in enumerate(tokens):
                    if token['upos'] == 'NOUN' and token['text'].lower() in temporal_nouns:
                        # Attach to root if it's a plausible modification context (e.g. not subject/object)
                        is_subj_obj = any(r.dependent_index == token['id'] and r.relation in ("nsubj", "obj") for r in relations)
                        if not is_subj_obj:
                            relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                                relation="obl:tmod", # nmod:tmod if modifying noun, obl:tmod if verb
                                                                dependent=token['text'], dependent_index=token['id']))
            
            # Rule 40: compound:prt - Verb particle (PART, often ADP like 'up', 'out', 'off') with a VERB.
            # Example: "look up", "take off". Particle often follows verb.
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] == 'VERB' and tokens[i+1]['upos'] in ['PART', 'ADP'] : # ADP used for particles too
                    # Heuristic: check if ADP is a common particle.
                    common_particles = {'up', 'down', 'in', 'out', 'on', 'off', 'away', 'back'}
                    if tokens[i+1]['text'].lower() in common_particles:
                        relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                            relation="compound:prt", dependent=tokens[i+1]['text'],
                                                            dependent_index=tokens[i+1]['id']))

            # --- Clauses and Complements Variants ---
            # Rule 41: acl - Adjectival clause modifying a NOUN (e.g., VERB participle "cat *sleeping*", "letter *written*").
            for i in range(len(tokens) -1):
                if tokens[i]['upos'] in ['NOUN', 'PROPN'] and tokens[i+1]['upos'] == 'VERB': # simplified: any VERB form
                     relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="acl", dependent=tokens[i+1]['text'],
                                                        dependent_index=tokens[i+1]['id']))
            
            # Rule 42: acl:relcl - Relative clause introduced by a relative pronoun (who, which, that as PRON).
            relative_pronouns = {'who', 'whom', 'whose', 'which', 'that'} # 'that' can be SCONJ or PRON
            for i in range(1, len(tokens) -1): # Need a noun before, and a verb after pronoun
                if tokens[i]['upos'] == 'PRON' and tokens[i]['text'].lower() in relative_pronouns and \
                   tokens[i-1]['upos'] in ['NOUN', 'PROPN'] and \
                   tokens[i+1]['upos'] == 'VERB': # Simplification: rel pronoun then verb
                        relations.append(DependencyRelation(head=tokens[i-1]['text'], head_index=tokens[i-1]['id'], # Modifies preceding noun
                                                            relation="acl:relcl", dependent=tokens[i+1]['text'], # Verb of the rel clause
                                                            dependent_index=tokens[i+1]['id'])) 
                        relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'], # Pronoun is subj/obj of its clause verb
                                                            relation="nsubj", # Or "obj" depending on context
                                                            dependent=tokens[i]['text'], dependent_index=tokens[i]['id']))

            # Rule 43: xcomp - Verb + "to" + Verb (infinitive), where "to" is PART.
            for i in range(len(tokens) - 2):
                if tokens[i]['upos'] == 'VERB' and tokens[i+1]['text'].lower() == 'to' and tokens[i+1]['upos'] == 'PART' \
                   and tokens[i+2]['upos'] == 'VERB':
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="xcomp", dependent=tokens[i+2]['text'],
                                                        dependent_index=tokens[i+2]['id']))
                    relations.append(DependencyRelation(head=tokens[i+2]['text'], head_index=tokens[i+2]['id'],
                                                        relation="mark", dependent=tokens[i+1]['text'], # 'to' is marker for xcomp
                                                        dependent_index=tokens[i+1]['id']))

            # Rule 44: ccomp - Verb followed by SCONJ ('if', 'whether') and clause with VERB.
            if root_candidate != -1:
                sconjs_for_ccomp = {'if', 'whether'}
                for i in range(root_candidate + 1, len(tokens) - 1):
                    if tokens[i]['upos'] == 'SCONJ' and tokens[i]['text'].lower() in sconjs_for_ccomp and \
                       tokens[i+1]['upos'] == 'VERB': # Simplified: SCONJ VERB
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="ccomp", dependent=tokens[i+1]['text'],
                                                            dependent_index=tokens[i+1]['id']))
                        relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                            relation="mark", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))
                        break
            
            # Rule 45: advcl - Adverbial clause (SCONJ + ... + VERB) modifying main verb (root_candidate).
            # SCONJs like 'when', 'while', 'before', 'after', 'since', 'until', 'because', 'if', 'unless', 'although'
            adv_sconjs = {'when', 'while', 'before', 'after', 'since', 'until', 'because', 'if', 'unless', 'although', 'as', 'though'}
            if root_candidate != -1:
                for i in range(len(tokens) - 1):
                    if tokens[i]['upos'] == 'SCONJ' and tokens[i]['text'].lower() in adv_sconjs and \
                       tokens[i+1]['upos'] == 'VERB': # Simplified: SCONJ VERB form an advcl
                        # Ensure this SCONJ VERB is not part of csubj or ccomp already handled
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="advcl", dependent=tokens[i+1]['text'], # Verb of the advcl
                                                            dependent_index=tokens[i+1]['id']))
                        relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                            relation="mark", dependent=tokens[i]['text'], # SCONJ marks the advcl verb
                                                            dependent_index=tokens[i]['id']))
            
            # Rule 46: acl:toinf - Infinitive with "to" (PART) modifying a NOUN ("a chance to win").
            for i in range(len(tokens) - 2):
                if tokens[i]['upos'] in ['NOUN', 'PROPN'] and \
                   tokens[i+1]['text'].lower() == 'to' and tokens[i+1]['upos'] == 'PART' and \
                   tokens[i+2]['upos'] == 'VERB':
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="acl", dependent=tokens[i+2]['text'], # UD uses acl for this
                                                        dependent_index=tokens[i+2]['id']))
                    relations.append(DependencyRelation(head=tokens[i+2]['text'], head_index=tokens[i+2]['id'],
                                                        relation="mark", dependent=tokens[i+1]['text'],
                                                        dependent_index=tokens[i+1]['id']))

            # Rule 47: advcl:toinf - Infinitive with "to" (PART) as adverbial clause ("He came to win").
            if root_candidate != -1:
                for i in range(root_candidate + 1, len(tokens) - 1): # after main verb
                    if tokens[i]['text'].lower() == 'to' and tokens[i]['upos'] == 'PART' and \
                       tokens[i+1]['upos'] == 'VERB':
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="advcl", dependent=tokens[i+1]['text'],
                                                            dependent_index=tokens[i+1]['id']))
                        relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                            relation="mark", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))
                        break # Assume first one

            # --- Conjunctions and Structure ---
            # Rule 48: cc - Coordinating conjunction (CCONJ) linking to the preceding conjunct head.
            # Rule 10 (conj) links the conjuncts. This rule links the CCONJ itself.
            for i in range(1, len(tokens) -1):
                if tokens[i]['upos'] == 'CCONJ' and \
                   tokens[i-1]['upos'] == tokens[i+1]['upos'] and tokens[i-1]['upos'] in ['NOUN', 'VERB', 'ADJ', 'PROPN', 'ADV']:
                     relations.append(DependencyRelation(head=tokens[i-1]['text'], head_index=tokens[i-1]['id'],
                                                         relation="cc", dependent=tokens[i]['text'],
                                                         dependent_index=tokens[i]['id']))

            # Rule 49: conj - Conjunction of adverbs (ADV CCONJ ADV).
            for i in range(1, len(tokens) - 1):
                if tokens[i]['upos'] == 'CCONJ' and \
                   tokens[i-1]['upos'] == 'ADV' and tokens[i+1]['upos'] == 'ADV':
                    relations.append(DependencyRelation(head=tokens[i-1]['text'], head_index=tokens[i-1]['id'],
                                                        relation="conj", dependent=tokens[i+1]['text'],
                                                        dependent_index=tokens[i+1]['id']))
            
            # Rule 50: conj - Conjunction of proper nouns (PROPN CCONJ PROPN).
            for i in range(1, len(tokens) - 1):
                if tokens[i]['upos'] == 'CCONJ' and \
                   tokens[i-1]['upos'] == 'PROPN' and tokens[i+1]['upos'] == 'PROPN':
                    relations.append(DependencyRelation(head=tokens[i-1]['text'], head_index=tokens[i-1]['id'],
                                                        relation="conj", dependent=tokens[i+1]['text'],
                                                        dependent_index=tokens[i+1]['id']))

            # --- Punctuation Variants (Original rules 4 and 9 cover general punct) ---
            # Rule 51: punct - Comma separating items in a list or clauses, attached to preceding item.
            # This is similar to existing punct rules but emphasizes list/clause separation.
            for i, token in enumerate(tokens):
                if token['text'] == ',' and token['upos'] == 'PUNCT' and i > 0:
                    # Heuristic: if comma is between two same-POS words (likely list) or before CCONJ
                    if (i < len(tokens) - 1 and tokens[i-1]['upos'] == tokens[i+1]['upos']) or \
                       (i < len(tokens) - 1 and tokens[i+1]['upos'] == 'CCONJ'):
                        relations.append(DependencyRelation(head=tokens[i-1]['text'], head_index=tokens[i-1]['id'],
                                                            relation="punct", dependent=token['text'],
                                                            dependent_index=token['id']))

            # Rule 52: punct - Sentence final punctuation ('.', '?', '!') attached to root or last word.
            if root_candidate != -1 and len(tokens) > 0:
                last_token = tokens[-1]
                if last_token['upos'] == 'PUNCT' and last_token['text'] in ['.', '?', '!']:
                     relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                         relation="punct", dependent=last_token['text'],
                                                         dependent_index=last_token['id']))
            elif len(tokens) > 1 and tokens[-1]['upos'] == 'PUNCT' and tokens[-1]['text'] in ['.', '?', '!']: # No root verb found
                 relations.append(DependencyRelation(head=tokens[-2]['text'], head_index=tokens[-2]['id'], # Attach to word before punct
                                                     relation="punct", dependent=tokens[-1]['text'],
                                                     dependent_index=tokens[-1]['id']))


            # --- Multiword Expressions ---
            # Rule 53: fixed - Common fixed multi-word expressions. E.g. "of course", "at least".
            # This requires a list of MWEs.
            mwe_list = {("of", "course"), ("at", "least"), ("in", "fact"), ("such", "as"), ("for", "example")}
            for i in range(len(tokens) - 1):
                current_pair = (tokens[i]['text'].lower(), tokens[i+1]['text'].lower())
                if current_pair in mwe_list:
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="fixed", dependent=tokens[i+1]['text'],
                                                        dependent_index=tokens[i+1]['id']))
            
            # Rule 54: flat - Proper Nouns forming a single name entity ("New York", "Joe Biden").
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] == 'PROPN' and tokens[i+1]['upos'] == 'PROPN':
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="flat", dependent=tokens[i+1]['text'],
                                                        dependent_index=tokens[i+1]['id']))

            # --- Apposition and Parataxis ---
            # Rule 55: appos - Appositional modifier (NOUN/PROPN phrase next to another, e.g. "my friend, Sue").
            # Simplified: NOUN/PROPN, (PUNCT) NOUN/PROPN.
            for i in range(len(tokens) - 2): # NP1, NP2
                if tokens[i]['upos'] in ['NOUN', 'PROPN'] and \
                   tokens[i+1]['text'] == ',' and tokens[i+1]['upos'] == 'PUNCT' and \
                   tokens[i+2]['upos'] in ['NOUN', 'PROPN']:
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="appos", dependent=tokens[i+2]['text'],
                                                        dependent_index=tokens[i+2]['id']))
            for i in range(len(tokens) - 1): # NP1 NP2 (no comma, harder to distinguish from compound)
                if tokens[i]['upos'] in ['NOUN', 'PROPN'] and tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                    # A very weak heuristic: if the second noun is PROPN and first is common NOUN (e.g. "president Biden")
                    if tokens[i]['upos'] == 'NOUN' and tokens[i+1]['upos'] == 'PROPN':
                         # check not already compound
                        is_compound = any(r.head_index == tokens[i+1]['id'] and r.dependent_index == tokens[i]['id'] and r.relation == "compound" for r in relations)
                        if not is_compound:
                            relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                                relation="appos", dependent=tokens[i+1]['text'],
                                                                dependent_index=tokens[i+1]['id']))


            # Rule 56: parataxis - Juxtaposed clauses/phrases (e.g. linked by semicolon or colon, or just comma).
            if root_candidate != -1:
                for i in range(root_candidate + 1, len(tokens) - 1):
                    # MainVerb ... PUNCT(: or ;) Verb ...
                    if tokens[i]['upos'] == 'PUNCT' and tokens[i]['text'] in [';', ':'] and \
                       tokens[i+1]['upos'] == 'VERB':
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="parataxis", dependent=tokens[i+1]['text'],
                                                            dependent_index=tokens[i+1]['id']))
                        break # Assume first one for simplicity


            # --- Discourse Elements & Vocatives ---
            # Rule 57: discourse - Interjections (INTJ) linked to root or nearby phrase.
            if root_candidate != -1:
                for i, token in enumerate(tokens):
                    if token['upos'] == 'INTJ':
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="discourse", dependent=token['text'],
                                                            dependent_index=token['id']))
            elif len(tokens) > 0: # No root, attach to first token if INTJ is not first.
                 for i, token in enumerate(tokens):
                    if token['upos'] == 'INTJ' and i > 0:
                        relations.append(DependencyRelation(head=tokens[0]['text'], head_index=tokens[0]['id'],
                                                            relation="discourse", dependent=token['text'],
                                                            dependent_index=token['id']))


            # Rule 58: vocative - Noun/PROPN used in address, often set off by commas. (Extends rule 18)
            # Example: "John, come here." or "Come here, John."
            if root_candidate != -1:
                # Vocative at end: ... Verb, Vocative.
                if len(tokens) > root_candidate + 2 and \
                   tokens[-1]['upos'] == 'PUNCT' and tokens[-1]['text'] == '.' and \
                   tokens[-2]['upos'] in ['NOUN', 'PROPN'] and \
                   tokens[-3]['text'] == ',' and tokens[-3]['upos'] == 'PUNCT':
                    relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                        relation="vocative", dependent=tokens[-2]['text'],
                                                        dependent_index=tokens[-2]['id']))


            # --- Specific Prepositional Phrase Attachments (obl, nmod) ---
            # Rule 59: obl:agent - Agent in passive sentences ("by" + NOUN/PROPN).
            for i in range(len(tokens) - 2):
                # Look for Verb_passive ... by Agent
                if tokens[i]['upos'] == 'VERB' and \
                   tokens[i+1]['text'].lower() == 'by' and tokens[i+1]['upos'] == 'ADP' and \
                   tokens[i+2]['upos'] in ['NOUN', 'PROPN', 'PRON']:
                    # Check if tokens[i] is passive (has aux:pass)
                    is_passive = any(r.head_index == tokens[i]['id'] and r.relation == "aux:pass" for r in relations)
                    if is_passive:
                        relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                            relation="obl:agent", dependent=tokens[i+2]['text'],
                                                            dependent_index=tokens[i+2]['id']))
                        relations.append(DependencyRelation(head=tokens[i+2]['text'], head_index=tokens[i+2]['id'],
                                                            relation="case", dependent=tokens[i+1]['text'],
                                                            dependent_index=tokens[i+1]['id']))

            # Rule 60: obl:tmod - Temporal oblique with specific prepositions (on Monday, in April).
            # Extends original rule 20 (obl) and rule 39 (nmod:tmod).
            temporal_preps = {'on', 'in', 'at', 'during', 'before', 'after', 'since', 'until'}
            if root_candidate != -1:
                for i in range(len(tokens) -1):
                    if tokens[i]['upos'] == 'ADP' and tokens[i]['text'].lower() in temporal_preps and \
                       tokens[i+1]['upos'] == 'NOUN': # Could check if NOUN is e.g. month, day name
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="obl", dependent=tokens[i+1]['text'], # Using generic "obl"
                                                            dependent_index=tokens[i+1]['id']))
                        relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                            relation="case", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))

            # Rule 61: obl:lmod - Locative oblique with specific prepositions (in London, at the park).
            locative_preps = {'in', 'on', 'at', 'near', 'under', 'over', 'beside', 'by'}
            if root_candidate != -1:
                for i in range(len(tokens) -1):
                    if tokens[i]['upos'] == 'ADP' and tokens[i]['text'].lower() in locative_preps and \
                       tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="obl", dependent=tokens[i+1]['text'], # Using generic "obl"
                                                            dependent_index=tokens[i+1]['id']))
                        relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                            relation="case", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))

            # Rule 62: nmod - Noun modifying noun via "of" preposition (e.g. "member of the team").
            for i in range(len(tokens) - 2):
                if tokens[i]['upos'] in ['NOUN', 'PROPN'] and \
                   tokens[i+1]['text'].lower() == 'of' and tokens[i+1]['upos'] == 'ADP' and \
                   tokens[i+2]['upos'] in ['NOUN', 'PROPN']:
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="nmod", dependent=tokens[i+2]['text'],
                                                        dependent_index=tokens[i+2]['id']))
                    relations.append(DependencyRelation(head=tokens[i+2]['text'], head_index=tokens[i+2]['id'],
                                                        relation="case", dependent=tokens[i+1]['text'],
                                                        dependent_index=tokens[i+1]['id']))
            
            # Rule 63: det:predet - Predeterminers like 'all', 'both', 'half' before DET/NOUN.
            predeterminers = {'all', 'both', 'half', 'such'}
            for i in range(len(tokens) -1):
                if tokens[i]['text'].lower() in predeterminers and tokens[i]['upos'] == 'DET' : # some are DET, some ADV
                    if tokens[i+1]['upos'] in ['DET', 'NOUN', 'PROPN', 'PRON']: # e.g. "all the people", "both boys"
                        relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                            relation="det", dependent=tokens[i]['text'], # UD often uses det for these
                                                            dependent_index=tokens[i]['id']))

            # Rule 64: list - Items in a list (NOUN, NOUN, CCONJ NOUN).
            # Connects subsequent NOUNs to the first NOUN in a list.
            for i in range(len(tokens) - 2):
                if tokens[i]['upos'] == 'NOUN' and \
                   tokens[i+1]['text'] == ',' and tokens[i+1]['upos'] == 'PUNCT' and \
                   tokens[i+2]['upos'] == 'NOUN':
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="list", dependent=tokens[i+2]['text'], # Using "list" relation. Could be "conj".
                                                        dependent_index=tokens[i+2]['id']))

            # Rule 65: reparandum - Simple repetition of a word.
            for i in range(len(tokens) - 1):
                if tokens[i]['text'].lower() == tokens[i+1]['text'].lower() and tokens[i]['upos'] == tokens[i+1]['upos']:
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="reparandum", dependent=tokens[i+1]['text'],
                                                        dependent_index=tokens[i+1]['id']))

            # --- Fallback and Unspecified ---
            # Rule 66: dep - Generic dependency for unattached tokens (e.g. orphan ADV to root). (Very heuristic)
            if root_candidate != -1:
                for i, token in enumerate(tokens):
                    is_attached = any(r.dependent_index == token['id'] or r.head_index == token['id'] for r in relations)
                    # Don't attach root itself as dependent via 'dep'
                    if token['id'] == tokens[root_candidate]['id']: is_attached = True 
                    
                    if not is_attached and token['upos'] == 'ADV': # Attach loose ADV to root
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="dep", dependent=token['text'],
                                                            dependent_index=token['id']))
            
            # Rule 67: nmod:npmod - Noun phrase as adverbial modifier (e.g., "5 miles long", "three days ago").
            # "three days ago": NOUN NOUN ADV -> "days" nmod "ago", then "ago" advmod verb. Or "days" obl verb.
            # Simplified: NUM NOUN (like "5 miles") modifying an ADJ ("long") or ADV ("ago")
            for i in range(len(tokens) - 2):
                if tokens[i]['upos'] == 'NUM' and tokens[i+1]['upos'] == 'NOUN':
                    if tokens[i+2]['upos'] == 'ADJ': # "5 miles long" -> "miles" nmod "long"
                        relations.append(DependencyRelation(head=tokens[i+2]['text'], head_index=tokens[i+2]['id'],
                                                            relation="nmod:npmod", dependent=tokens[i+1]['text'],
                                                            dependent_index=tokens[i+1]['id']))
                        relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                            relation="nummod", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))
                    elif tokens[i+2]['upos'] == 'ADV': # "three days ago" -> "days" nmod "ago"
                         relations.append(DependencyRelation(head=tokens[i+2]['text'], head_index=tokens[i+2]['id'],
                                                            relation="nmod:npmod", dependent=tokens[i+1]['text'],
                                                            dependent_index=tokens[i+1]['id']))
                         relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                            relation="nummod", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))


            # Rule 68: det:q - Quantifiers (some, any, many, few) as determiners.
            quantifiers = {'some', 'any', 'many', 'few', 'several', 'much', 'little', 'no', 'each', 'every'}
            for i in range(len(tokens) - 1):
                if (tokens[i]['upos'] == 'DET' or tokens[i]['upos'] == 'PRON') and \
                   tokens[i]['text'].lower() in quantifiers and tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                    relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                        relation="det", dependent=tokens[i]['text'],
                                                        dependent_index=tokens[i]['id']))
            
            # Rule 69: mark - Specific SCONJ 'that' introducing VERB (alternative to existing ccomp/mark).
            if root_candidate != -1:
                for i in range(root_candidate + 1, len(tokens) -1):
                    if tokens[i]['text'].lower() == 'that' and tokens[i]['upos'] == 'SCONJ' and tokens[i+1]['upos'] == 'VERB':
                        # Check if already ccomp from rule 16
                        is_ccomp = any(r.head_index == tokens[root_candidate]['id'] and r.dependent_index == tokens[i+1]['id'] and r.relation == "ccomp" for r in relations)
                        if not is_ccomp:
                             relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                                relation="mark", dependent=tokens[i]['text'],
                                                                dependent_index=tokens[i]['id']))
                        break # Assuming one main 'that' clause after verb

            # Rule 70: case - Preposition 'to' indicating recipient/goal for a NOUN.
            for i in range(len(tokens) - 1):
                if tokens[i]['text'].lower() == 'to' and tokens[i]['upos'] == 'ADP' and \
                   tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                    # Check if already handled by obl (rule 20) or other specific rules
                    is_handled = any(r.dependent_index == tokens[i]['id'] and r.head_index == tokens[i+1]['id'] and r.relation == "case" for r in relations)
                    if not is_handled:
                        relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                            relation="case", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))

            # Rule 71: case - Preposition 'with' indicating accompaniment/instrument for a NOUN.
            for i in range(len(tokens) - 1):
                if tokens[i]['text'].lower() == 'with' and tokens[i]['upos'] == 'ADP' and \
                   tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                    is_handled = any(r.dependent_index == tokens[i]['id'] and r.head_index == tokens[i+1]['id'] and r.relation == "case" for r in relations)
                    if not is_handled:
                        relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                            relation="case", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))

            # Rule 72: case - Preposition 'for' indicating beneficiary/purpose for a NOUN.
            for i in range(len(tokens) - 1):
                if tokens[i]['text'].lower() == 'for' and tokens[i]['upos'] == 'ADP' and \
                   tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                    is_handled = any(r.dependent_index == tokens[i]['id'] and r.head_index == tokens[i+1]['id'] and r.relation == "case" for r in relations)
                    if not is_handled:
                        relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                            relation="case", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))
            
            # Rule 73: obl - Nominal with 'to' modifying VERB (extends rule 20).
            if root_candidate != -1:
                 for i in range(len(tokens) - 1):
                    if tokens[i]['text'].lower() == 'to' and tokens[i]['upos'] == 'ADP' and \
                       tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                        # Check if this obl relation is already added by Rule 20 or others
                        already_added = False
                        for rel in relations:
                            if rel.relation == "obl" and rel.dependent_index == tokens[i+1]['id'] and rel.head_index == tokens[root_candidate]['id']:
                                already_added = True; break
                        if not already_added:
                             relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                                relation="obl", dependent=tokens[i+1]['text'],
                                                                dependent_index=tokens[i+1]['id']))

            # Rule 74: obl - Nominal with 'with' modifying VERB.
            if root_candidate != -1:
                 for i in range(len(tokens) - 1):
                    if tokens[i]['text'].lower() == 'with' and tokens[i]['upos'] == 'ADP' and \
                       tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                        already_added = False
                        for rel in relations:
                            if rel.relation == "obl" and rel.dependent_index == tokens[i+1]['id'] and rel.head_index == tokens[root_candidate]['id']:
                                already_added = True; break
                        if not already_added:
                            relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                                relation="obl", dependent=tokens[i+1]['text'],
                                                                dependent_index=tokens[i+1]['id']))

            # Rule 75: obl - Nominal with 'for' modifying VERB.
            if root_candidate != -1:
                 for i in range(len(tokens) - 1):
                    if tokens[i]['text'].lower() == 'for' and tokens[i]['upos'] == 'ADP' and \
                       tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                        already_added = False
                        for rel in relations:
                            if rel.relation == "obl" and rel.dependent_index == tokens[i+1]['id'] and rel.head_index == tokens[root_candidate]['id']:
                                already_added = True; break
                        if not already_added:
                            relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                                relation="obl", dependent=tokens[i+1]['text'],
                                                                dependent_index=tokens[i+1]['id']))
            
            # Rule 76: csubj:pass - Clausal subject of a passive verb.
            if root_candidate > 1: # Main verb (passive)
                # Check if root_candidate is passive
                is_passive_root = any(r.head_index == tokens[root_candidate]['id'] and r.relation == "aux:pass" for r in relations)
                if is_passive_root:
                    # Clausal subject: SCONJ VERB ... PassiveRootVerb
                    if tokens[0]['upos'] == 'SCONJ' and tokens[1]['upos'] == 'VERB':
                         relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                             relation="csubj:pass", dependent=tokens[1]['text'],
                                                             dependent_index=tokens[1]['id']))

            # Rule 77: advmod:emph - Emphasizing adverbial modifier (e.g., 'very', 'really', 'so' modifying ADJ/ADV).
            emph_adverbs = {'very', 'really', 'so', 'too', 'extremely', 'quite', 'rather'}
            for i in range(len(tokens) -1):
                if tokens[i]['upos'] == 'ADV' and tokens[i]['text'].lower() in emph_adverbs and \
                   tokens[i+1]['upos'] in ['ADJ', 'ADV']:
                    relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                        relation="advmod", dependent=tokens[i]['text'], # UD uses advmod
                                                        dependent_index=tokens[i]['id']))

            # Rule 78: xcomp - Verb followed by ADJ/NOUN where subject is shared (e.g. "He made her happy", "They elected him president").
            # This is complex. Simplified: VERB PRON/NOUN ADJ/NOUN
            if root_candidate != -1 and root_candidate < len(tokens) - 2:
                # V obj xcomp_pred: He (subj) made (V) her (obj) happy (xcomp_pred ADJ)
                # V obj xcomp_pred: They (subj) elected (V) him (obj) president (xcomp_pred NOUN)
                # Here, we link the predicate (ADJ/NOUN) to the object. Or to the verb as xcomp.
                # Let's assume obj is tokens[root_candidate+1]
                obj_token_idx = -1
                for rel in relations: # Find object of the root verb
                    if rel.head_index == tokens[root_candidate]['id'] and rel.relation == "obj":
                        # Find the token corresponding to rel.dependent_index
                        for k_idx, t_k in enumerate(tokens):
                            if t_k['id'] == rel.dependent_index:
                                obj_token_idx = k_idx
                                break
                        break
                
                if obj_token_idx != -1 and obj_token_idx < len(tokens) -1:
                    if tokens[obj_token_idx+1]['upos'] in ['ADJ', 'NOUN']: # Predicate after object
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="xcomp", dependent=tokens[obj_token_idx+1]['text'],
                                                            dependent_index=tokens[obj_token_idx+1]['id']))

            # Rule 79: root - Fallback root: if no VERB, first NOUN is root.
            if root_candidate == -1 and len(tokens) > 0:
                for i, token in enumerate(tokens):
                    if token['upos'] == 'NOUN':
                        relations.append(DependencyRelation(head="ROOT", head_index=0,
                                                            relation="root",
                                                            dependent=token['text'],
                                                            dependent_index=token['id']))
                        # Set this as a pseudo root_candidate for other rules that might need some anchor
                        # This is a bit of a hack for this example structure
                        # If you do this, ensure `root` variable is also updated or rules handle `root_candidate` pointing to non-verb
                        # For simplicity, not setting pseudo root_candidate here.
                        break
            
            # Rule 80: punct - Hyphenated compounds (if hyphen is separate token). word1 - word2
            for i in range(len(tokens) - 2):
                 if tokens[i+1]['text'] == '-' and tokens[i+1]['upos'] == 'PUNCT' and \
                    tokens[i]['upos'] not in ['PUNCT', 'SPACE'] and tokens[i+2]['upos'] not in ['PUNCT', 'SPACE']:
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="punct", dependent=tokens[i+1]['text'], # Or 'compound' if they form one unit
                                                        dependent_index=tokens[i+1]['id']))
            
            # Rule 81: compound - Two NOUNs where the first modifies the second (extends rule 12 to ensure not already nmod).
            for i in range(len(tokens) -1):
                if tokens[i]['upos'] == 'NOUN' and tokens[i+1]['upos'] == 'NOUN':
                    # Check not already nmod (rule 11) or other relations
                    is_nmod = any(r.head_index == tokens[i+1]['id'] and r.dependent_index == tokens[i]['id'] and r.relation == 'nmod' for r in relations)
                    if not is_nmod:
                         relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                            relation="compound", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))
            
            # Rule 82: mark - 'to' as PART marking an infinitive VERB (complement of NOUN/ADJ).
            # E.g. "easy to learn", "time to go" (already covered by acl:toinf / advcl:toinf for VERB heads)
            # This is for ADJ head: "easy to learn" -> learn (xcomp) easy, to (mark) learn
            for i in range(len(tokens) - 2):
                if tokens[i]['upos'] == 'ADJ' and \
                   tokens[i+1]['text'].lower() == 'to' and tokens[i+1]['upos'] == 'PART' and \
                   tokens[i+2]['upos'] == 'VERB':
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],      # ADJ is head
                                                        relation="xcomp", dependent=tokens[i+2]['text'],    # Verb is xcomp
                                                        dependent_index=tokens[i+2]['id']))
                    relations.append(DependencyRelation(head=tokens[i+2]['text'], head_index=tokens[i+2]['id'],  # Verb is head of 'to'
                                                        relation="mark", dependent=tokens[i+1]['text'],     # 'to' is mark
                                                        dependent_index=tokens[i+1]['id']))

            # Rule 83: discourse - "please" modifying a verb (often root).
            if root_candidate != -1:
                for i, token in enumerate(tokens):
                    if token['text'].lower() == 'please' and token['upos'] == 'INTJ': # or ADV sometimes
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="discourse", dependent=token['text'],
                                                            dependent_index=token['id']))

            # Rule 84: obl - Numeral as oblique modifier of verb (e.g. "cost 5 dollars" - 5 dollars is obl of cost).
            # Simplified: VERB NUM NOUN -> NOUN is obl of VERB.
            if root_candidate != -1 and root_candidate < len(tokens) - 2:
                if tokens[root_candidate+1]['upos'] == 'NUM' and tokens[root_candidate+2]['upos'] == 'NOUN':
                    relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                        relation="obl", dependent=tokens[root_candidate+2]['text'],
                                                        dependent_index=tokens[root_candidate+2]['id']))
                    # nummod for NUM to NOUN is covered by rule 21.

            # Rule 85: advcl - Comparative clause "than..."
            # E.g. "older than me". "than" is SCONJ (or ADP), "me" is PRON. Clause modifies "older" (ADJ).
            for i in range(len(tokens) - 2):
                if tokens[i]['upos'] == 'ADJ' and \
                   tokens[i+1]['text'].lower() == 'than' and tokens[i+1]['upos'] in ['SCONJ', 'ADP'] and \
                   tokens[i+2]['upos'] in ['NOUN', 'PRON', 'PROPN', 'NUM']: # "than I (am)", "than him"
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="advcl", dependent=tokens[i+2]['text'], # The head of the comparative phrase
                                                        dependent_index=tokens[i+2]['id']))
                    relations.append(DependencyRelation(head=tokens[i+2]['text'], head_index=tokens[i+2]['id'],
                                                        relation="mark", dependent=tokens[i+1]['text'], # 'than' is mark
                                                        dependent_index=tokens[i+1]['id']))

            # Rule 86: expl - 'it' as expletive in "it seems that..." or "it is important to..."
            # Extends rule 14.
            if root_candidate != -1: # root_candidate is 'seems' or 'is'
                 if tokens[root_candidate-1]['text'].lower() == 'it' and tokens[root_candidate-1]['upos'] == 'PRON':
                    # Check if 'it' is not already nsubj (e.g. if root_candidate is not main verb)
                    is_expl_already = any(r.relation == "expl" and r.dependent_index == tokens[root_candidate-1]['id'] for r in relations)
                    if not is_expl_already:
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="expl", dependent=tokens[root_candidate-1]['text'],
                                                            dependent_index=tokens[root_candidate-1]['id']))

            # Rule 87: case - Postposition (NOUN + ADP, rare in English, common in other languages).
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] in ['NOUN', 'PROPN'] and tokens[i+1]['upos'] == 'ADP':
                    # This might be e.g. "the whole year round" where 'round' is ADP.
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="case", dependent=tokens[i+1]['text'],
                                                        dependent_index=tokens[i+1]['id']))

            # Rule 88: advmod - Adverbial 'there' (locative, not expletive). "He is there."
            if root_candidate != -1: # e.g. root is 'is'
                for i in range(root_candidate + 1, len(tokens)):
                    if tokens[i]['text'].lower() == 'there' and tokens[i]['upos'] == 'ADV':
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="advmod", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))
                        break

            # Rule 89: flat:foreign - Foreign words used in sequence, not part of MWEs.
            # Requires 'X' (other) POS tag or language annotation. Assuming 'X' is foreign material.
            for i in range(len(tokens) - 1):
                if tokens[i]['upos'] == 'X' and tokens[i+1]['upos'] == 'X':
                     relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="flat:foreign", dependent=tokens[i+1]['text'],
                                                        dependent_index=tokens[i+1]['id']))

            # Rule 90: goeswith - Part of a word separated by tokenization error. (Very hard to detect simply)
            # Heuristic: if a token ends with hyphen and next token starts lowercase? Or specific patterns.
            # Example: "state - of - the - art" as "state-", "of-", "the-", "art"
            # This is too complex for this framework without more info. Simple version: token + hyphen token.
            for i in range(len(tokens) -1):
                if tokens[i]['text'].endswith('-') and tokens[i+1]['upos'] != 'PUNCT': # e.g. "state-" "of"
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="goeswith", dependent=tokens[i+1]['text'],
                                                        dependent_index=tokens[i+1]['id']))

            # Rule 91: orphan - If a CCONJ is left without a second conjunct (e.g. sentence ends "apples and...").
            # Attach CCONJ to its first conjunct. (Rule 48 `cc` already does this more generally)
            # This rule tries to find an orphan CCONJ at the end of a sequence.
            if len(tokens) > 1 and tokens[-1]['upos'] == 'CCONJ':
                # Attach to the token before it if it's a potential conjunct
                if tokens[-2]['upos'] in ['NOUN', 'VERB', 'ADJ', 'PROPN', 'ADV']:
                    relations.append(DependencyRelation(head=tokens[-2]['text'], head_index=tokens[-2]['id'],
                                                        relation="orphan", dependent=tokens[-1]['text'], # or 'cc'
                                                        dependent_index=tokens[-1]['id']))
            
            # Rule 92: det - Article 'a'/'an'/'the'. (Specific version of rule 22)
            articles = {'a', 'an', 'the'}
            for i in range(len(tokens) - 1):
                if tokens[i]['text'].lower() in articles and tokens[i]['upos'] == 'DET' and \
                   tokens[i+1]['upos'] in ['NOUN', 'PROPN', 'ADJ']: # ADJ if like "the big dog"
                    # If ADJ, attach DET to ADJ, and ADJ to NOUN (amod)
                    # Simplified: attach to following NOUN/PROPN, or ADJ if NOUN follows ADJ
                    if tokens[i+1]['upos'] == 'ADJ' and i < len(tokens) - 2 and tokens[i+2]['upos'] in ['NOUN', 'PROPN']:
                         relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'], # Det attaches to ADJ
                                                            relation="det", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))
                    elif tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                         relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                            relation="det", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))
            
            # Rule 93: obl - "ago" with preceding temporal noun (e.g. "three days ago").
            # "days" is obl of verb (via "ago"), "ago" is case for "days" or "ago" is advmod and "days" nmod "ago".
            # Simpler: "days ago" -> "ago" modifies verb, "days" modifies "ago".
            if root_candidate != -1:
                for i in range(1, len(tokens)):
                    if tokens[i]['text'].lower() == 'ago' and tokens[i]['upos'] == 'ADV' and \
                       tokens[i-1]['upos'] in ['NOUN', 'NUM']: # "days ago" or "week ago" or "2 years ago"
                        # Attach "ago" to the root verb
                        relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                            relation="advmod", dependent=tokens[i]['text'],
                                                            dependent_index=tokens[i]['id']))
                        # Attach the preceding noun/num to "ago"
                        relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                            relation="nmod:npmod", dependent=tokens[i-1]['text'], # UD uses nmod:npmod or obl:tmod for the noun
                                                            dependent_index=tokens[i-1]['id']))
            
            # Rule 94: xcomp - Infinitival verb complement of certain verbs like 'try', 'begin', 'fail' + to + VERB.
            # (Extends rule 43, which handles V to V generally). This targets specific matrix verbs.
            xcomp_trigger_verbs = {'try', 'tried', 'tries', 'begin', 'began', 'begins', 'fail', 'failed', 'fails', 'want', 'wants', 'wanted'}
            for i in range(len(tokens) - 2):
                if tokens[i]['text'].lower() in xcomp_trigger_verbs and tokens[i]['upos'] == 'VERB' and \
                   tokens[i+1]['text'].lower() == 'to' and tokens[i+1]['upos'] == 'PART' and \
                   tokens[i+2]['upos'] == 'VERB':
                    # Avoid re-adding if Rule 43 already did.
                    is_handled = any(r.head_index == tokens[i]['id'] and r.dependent_index == tokens[i+2]['id'] and r.relation == 'xcomp' for r in relations)
                    if not is_handled:
                        relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                            relation="xcomp", dependent=tokens[i+2]['text'],
                                                            dependent_index=tokens[i+2]['id']))
                        relations.append(DependencyRelation(head=tokens[i+2]['text'], head_index=tokens[i+2]['id'],
                                                            relation="mark", dependent=tokens[i+1]['text'],
                                                            dependent_index=tokens[i+1]['id']))
            
            # Rule 95: compound - Adjective-Noun compound where it's lexicalized (e.g. "blackboard", "greenhouse").
            # This is hard to distinguish from amod. Usually, these are single words or hyphenated.
            # If tokenized as ADJ NOUN, it's typically amod. If a rule for "compound" is needed for ADJ NOUN:
            # This rule is speculative and might conflict with 'amod' (rule 5).
            # A common signal would be if they are not separable or ADJ isn't gradable.
            # For simplicity, let's say if ADJ + NOUN and there's no DET for the NOUN.
            # E.g. "She has blackboard." vs "She has a black board."
            # No, this is too ambiguous. Stick to existing amod. Perhaps for specific lexical items?
            # compound_adj_noun = {("black", "board"), ("green", "house"), ("high", "school")}
            # for i in range(len(tokens) - 1):
            #    if tokens[i]['upos'] == 'ADJ' and tokens[i+1]['upos'] == 'NOUN' and \
            #       (tokens[i]['text'].lower(), tokens[i+1]['text'].lower()) in compound_adj_noun:
            #        relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
            #                                            relation="compound", dependent=tokens[i]['text'],
            #                                            dependent_index=tokens[i]['id']))

            # Rule 96: mark - Subordinating conjunction 'that' when it's optional and omitted but implied. (Impossible for this system)
            # This rule is a conceptual placeholder; detection of omitted 'that' is beyond simple patterns.

            # Rule 97: parataxis - Direct speech without explicit reporting verb. e.g. "Okay," she said. (Rule 56 is for punctuation-based)
            # "Quote" (VERB), Person (NOUN) said (VERB_reporting). -> parataxis "Quote" to "said", nsubj Person to "said".
            # If "She said: I am happy." -> "said" ccomp "happy". "I am happy" is the ccomp.
            # "I am happy, she said." -> "happy" parataxis "said".
            if root_candidate != -1: # Assume root is main verb of quote for "I am happy, she said."
                # Look for Verb_quote PUNCT(,) Noun_speaker Verb_reporting
                if len(tokens) > root_candidate + 3 and tokens[root_candidate+1]['text'] == ',' and \
                   tokens[root_candidate+2]['upos'] in ['NOUN', 'PRON', 'PROPN'] and \
                   tokens[root_candidate+3]['upos'] == 'VERB':
                    relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                        relation="parataxis", dependent=tokens[root_candidate+3]['text'],
                                                        dependent_index=tokens[root_candidate+3]['id']))
                    relations.append(DependencyRelation(head=tokens[root_candidate+3]['text'], head_index=tokens[root_candidate+3]['id'],
                                                        relation="nsubj", dependent=tokens[root_candidate+2]['text'],
                                                        dependent_index=tokens[root_candidate+2]['id']))


            # Rule 98: advmod:lmod - Locative adverbs like 'here', 'there', 'abroad', 'upstairs'.
            locative_adverbs = {'here', 'there', 'abroad', 'upstairs', 'downstairs', 'everywhere', 'somewhere', 'nowhere', 'anywhere'}
            if root_candidate != -1:
                for i, token in enumerate(tokens):
                    if token['text'].lower() in locative_adverbs and token['upos'] == 'ADV':
                        # Avoid if 'there' is expletive or already handled
                        is_expl = any(r.dependent_index == token['id'] and r.relation == "expl" for r in relations)
                        if not is_expl:
                             relations.append(DependencyRelation(head=tokens[root_candidate]['text'], head_index=tokens[root_candidate]['id'],
                                                                relation="advmod", dependent=token['text'], # UD uses advmod for these
                                                                dependent_index=token['id']))

            # Rule 99: det - 'no' as a determiner. e.g. "no dogs allowed"
            for i in range(len(tokens) - 1):
                if tokens[i]['text'].lower() == 'no' and tokens[i]['upos'] == 'DET' and \
                   tokens[i+1]['upos'] in ['NOUN', 'PROPN']:
                    relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                        relation="det", dependent=tokens[i]['text'],
                                                        dependent_index=tokens[i]['id']))

            # Rule 100: ccomp - Clause as complement of NOUN or ADJ. ("the fact that S", "happy that S")
            # "fact that she came" -> that_clause ccomp of fact. "happy that she came" -> that_clause ccomp of happy.
            for i in range(len(tokens) - 2):
                if tokens[i]['upos'] in ['NOUN', 'ADJ'] and \
                   tokens[i+1]['text'].lower() == 'that' and tokens[i+1]['upos'] == 'SCONJ' and \
                   tokens[i+2]['upos'] == 'VERB': # Simplified: N/ADJ that VERB
                    relations.append(DependencyRelation(head=tokens[i]['text'], head_index=tokens[i]['id'],
                                                        relation="ccomp", dependent=tokens[i+2]['text'], # Verb of the clause
                                                        dependent_index=tokens[i+2]['id']))
                    relations.append(DependencyRelation(head=tokens[i+2]['text'], head_index=tokens[i+2]['id'],
                                                        relation="mark", dependent=tokens[i+1]['text'], # 'that'
                                                        dependent_index=tokens[i+1]['id']))

            # Rule 101: punct - Opening/Closing quotes/parentheses. Attach to first/last word of content.
            # This is complex. Simplified: opening punct attaches to following word, closing to preceding.
            opening_punct = {'(', '[', '{', '“', '‘', '"'}
            closing_punct = {')', ']', '}', '”', '’', '"'} # Double quote is ambiguous
            for i, token in enumerate(tokens):
                if token['text'] in opening_punct and token['upos'] == 'PUNCT' and i < len(tokens) -1:
                    if tokens[i+1]['upos'] != 'PUNCT': # Don't attach to another punctuation
                        relations.append(DependencyRelation(head=tokens[i+1]['text'], head_index=tokens[i+1]['id'],
                                                            relation="punct", dependent=token['text'],
                                                            dependent_index=token['id']))
                elif token['text'] in closing_punct and token['upos'] == 'PUNCT' and i > 0:
                    if tokens[i-1]['upos'] != 'PUNCT':
                         relations.append(DependencyRelation(head=tokens[i-1]['text'], head_index=tokens[i-1]['id'],
                                                            relation="punct", dependent=token['text'],
                                                            dependent_index=token['id']))

        # ... many more rules needed for other relation types (amod, advmod, case, det, etc.) ...
        # A real system uses machine learning or hundreds/thousands of rules.
        return relations
