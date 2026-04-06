import json

from pathlib import Path
from typing import Dict, Optional

class ResourceLoader:
    _structured_wordlist: Optional[Dict] = None
    _simple_wordlist: Optional[Dict] = None
    _bpe_model: Optional[Dict] = None
    _bpe_vocab: Optional[Dict] = None
    _glove_embeddings: Optional[Dict] = None
    _modality_markers: Optional[Dict] = None
    _loughran_mcdonald_lexicon: Optional[Dict] = None

    @classmethod
    def get_structured_wordlist(cls, path="src/agents/language/library/structured_wordlist_en.json") -> Dict:
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
    def get_simple_wordlist(cls, path="src/agents/language/library/wordlist_en.json") -> Dict:
        if cls._simple_wordlist is None:
            with open(Path(path), "r", encoding="utf-8") as f:
                cls._simple_wordlist = json.load(f)
        return cls._simple_wordlist

    @classmethod
    def get_bpe_model(cls, path="data/bpe_200d_20k_model.json") -> Dict:
        if cls._bpe_model is None:
            try:
                with open(Path(path), "r", encoding="utf-8") as f:
                    cls._bpe_model = json.load(f)
            except Exception:
                cls._bpe_model = {}
        return cls._bpe_model

    @classmethod
    def get_bpe_vocab(cls, path="data/bpe_200d_20k_vocab.json") -> Dict:
        if cls._bpe_vocab is None:
            try:
                with open(Path(path), "r", encoding="utf-8") as f:
                    cls._bpe_vocab = json.load(f)
            except Exception:
                cls._bpe_vocab = {}
        return cls._bpe_vocab

    @classmethod
    def get_glove_embeddings(cls, path="data/embedings/glove.6B.200d.json") -> Dict:
        if cls._glove_embeddings is None:
            try:
                with open(Path(path), "r", encoding="utf-8") as f:
                    cls._glove_embeddings = json.load(f)
            except Exception:
                cls._glove_embeddings = {}
        return cls._glove_embeddings

    @classmethod
    def get_modality_markers(cls, path="src/agents/language/templates/modality_markers.json") -> Dict:
        if cls._modality_markers is None:
            try:
                with open(Path(path), "r", encoding="utf-8") as f:
                    cls._modality_markers = json.load(f)
            except Exception:
                cls._modality_markers = {
                    "epistemic": {}, "deontic": {}, "dynamic": {},
                    "alethic": {}, "interrogative": {}, "imperative": {},
                    "conditional": {}
                }
        return cls._modality_markers

    @classmethod
    def get_loughran_mcdonald_lexicon(cls, path="src/agents/language/templates/loughran_mcdonald_lexicon.json") -> Dict:
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
