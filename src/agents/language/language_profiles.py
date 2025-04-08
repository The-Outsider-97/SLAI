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
    'nl': {
        'allowed_affixes': {
            'pre': ['on','her'],
            'suf': ['en','tie']
        },
        # ... other language rules
    }
}
