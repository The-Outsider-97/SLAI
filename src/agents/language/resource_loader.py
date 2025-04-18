import json
from pathlib import Path
from typing import Dict, Optional


class ResourceLoader:
    _structured_wordlist: Optional[Dict] = None
    _simple_wordlist: Optional[Dict] = None
    _sentiment_lexicon: Optional[Dict] = None
    _nlg_templates: Optional[Dict] = None
    _loughran_mcdonald_lexicon: Optional[Dict] = None

    @classmethod
    def get_structured_wordlist(cls, path="src/agents/language/structured_wordlist_en.json") -> Dict:
        if cls._structured_wordlist is None:
            with open(Path(path), "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "words" in data:
                    cls._structured_wordlist = data["words"]
                elif isinstance(data, dict):
                    cls._structured_wordlist = data
                else:
                    raise TypeError("Expected a dict in structured_wordlist, got {}".format(type(data).__name__))
        return cls._structured_wordlist

    @classmethod
    def get_simple_wordlist(cls, path="src/agents/language/wordlist_en.json") -> Dict:
        if cls._simple_wordlist is None:
            with open(Path(path), "r", encoding="utf-8") as f:
                cls._simple_wordlist = json.load(f)
        return cls._simple_wordlist

    @classmethod
    def get_sentiment_lexicon(cls, path="src/agents/language/sentiment_lexicon.json") -> Dict:
        if cls._sentiment_lexicon is None:
            try:
                with open(Path(path), "r", encoding="utf-8") as f:
                    cls._sentiment_lexicon = json.load(f)
            except Exception:
                cls._sentiment_lexicon = {
                    "positive": {}, "negative": {},
                    "intensifiers": {}, "negators": []
                }
        return cls._sentiment_lexicon

    @classmethod
    def get_nlg_templates(cls, path="src/agents/language/nlg_templates_en.json") -> Dict:
        if cls._nlg_templates is None:
            try:
                with open(Path(path), "r", encoding="utf-8") as f:
                    cls._nlg_templates = json.load(f)
            except Exception:
                cls._nlg_templates = {}
        return cls._nlg_templates
    
    @classmethod
    def get_loughran_mcdonald_lexicon(cls, path="src/agents/language/loughran_mcdonald_lexicon.json") -> Dict:
        if cls._loughran_mcdonald_lexicon is None:
            try:
                with open(Path(path), "r", encoding="utf-8") as f:
                    cls._loughran_mcdonald_lexicon = json.load(f)
            except Exception:
                cls._loughran_mcdonald_lexicon = {
                    "positive": {}, "negative": {},
                    "intensifiers": {}, "negators": []
                }
        return cls._loughran_mcdonald_lexicon
