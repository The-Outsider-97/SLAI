
import re
import json
from pathlib import Path

# ===== Load full NLP configuration (intents + entities) =====
def load_config(filepath="src/agents/language/nlp_config.json"):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

# ===== Provide accessors =====
class NLPProfiles:
    def __init__(self, config_path="src/agents/language/nlp_config.json"):
        self.config = load_config(config_path)
        self.intent_patterns = self.config.get("intent_weights", {})
        self.entity_patterns = self.config.get("entity_patterns", {})

    def get_intents(self):
        return self.intent_patterns

    def get_entities(self):
        return self.entity_patterns

# ===== Morphology rules =====
MORPHOLOGY_RULES = {
    'en': {
        'allowed_affixes': {
            'pre': ['un','re','dis','mis'],
            'suf': ['ing','ed','s','ly','able']
        },
        'compound_patterns': [
            r'\w+-\w+',
            r'\w+-\d+'
        ],
        'max_syllables': 7,
        'valid_chars': r'^[a-z-\']+$'
    },
        # ... other language rules
}

# ===== BPE token merger =====
def merge_bpe_tokens(tokens, suffix='</w>', unknown_token='<unk>', skip_special=True, join_marker=''):
    """
    Merge BPE tokens back into full words.

    Args:
        tokens (list of str): Token list from BPE tokenizer.
        suffix (str): Suffix marking end of word (e.g., '</w>' or '@@').
        unknown_token (str): Token used for unknown words.
        skip_special (bool): Whether to skip special tokens like [PAD], [CLS].
        join_marker (str): Optional marker to join merged subwords (e.g., '' or '-').

    Returns:
        list of str: Reconstructed full words.
    """
    special_tokens = {unknown_token, '[PAD]', '[CLS]', '[SEP]'}
    words = []
    current_word = ''

    for token in tokens:
        if skip_special and token in special_tokens:
            continue

        if token.endswith(suffix):
            part = token[:-len(suffix)]
            current_word += part
            words.append(current_word)
            current_word = ''
        else:
            current_word += token

    if current_word:
        words.append(current_word)

    return words
