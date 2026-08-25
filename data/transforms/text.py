"""
Text transforms: cleaning, tokenization, and truncation.
"""
from __future__ import annotations
 
import re
 
from typing import Any, Dict, List, Optional
 
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import _CONTROL_CHAR_RE
from .base_transform import Transform
from .registry import register_transform
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]
 
logger = get_logger("Text Transform")
printer = PrettyPrinter()
 
# Optional HuggingFace tokenizer — guarded so the module loads without it.
try:
    from transformers import AutoTokenizer  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
 
_WHITESPACE_RE = re.compile(r"\s+")
 
 
# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
# @register_transform("clean_text")
class CleanText(Transform):
    """Basic text cleaning: strip, optional lowercasing, control-char removal,
    extra-whitespace collapse, and character-level truncation.
 
    Config keys (``transforms.text``):
 
    * ``lowercase`` (bool, default ``True``) — fold to lowercase.
    * ``strip_control_chars`` (bool, default ``True``) — remove ASCII control chars.
    * ``remove_extra_whitespace`` (bool, default ``True``) — collapse runs of
      whitespace to a single space.
    * ``max_chars`` (int, default ``None``) — character truncation ceiling; used
      when ``max_length`` is not supplied to the constructor.
    """
 
    def __init__(
        self,
        max_length: Optional[int] = None,
        remove_extra_whitespace: bool = True,
    ) -> None:
        super().__init__()
        self.text_cfg: Dict[str, Any] = get_config_section("transforms").get("text", {})
        self.lowercase: bool = bool(self.text_cfg.get("lowercase", True))
        self.strip_control_chars: bool = bool(self.text_cfg.get("strip_control_chars", True))
        self.remove_extra_whitespace: bool = remove_extra_whitespace
 
        # max_length: explicit constructor arg wins over config value.
        _cfg_max: Optional[int] = self.text_cfg.get("max_chars")
        self.max_length: Optional[int] = (
            max_length if max_length is not None
            else (int(_cfg_max) if _cfg_max is not None else None)
        )
 
    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        if modality != "text":
            return record
 
        raw = record.get("text")
        if raw is None:
            return record
        if not isinstance(raw, str):
            raise DataTransformError(
                "CleanText expected record['text'] to be a str",
                context={"modality": modality, "got": type(raw).__name__},
            )
 
        cleaned = raw.strip()
        if self.strip_control_chars:
            cleaned = _CONTROL_CHAR_RE.sub("", cleaned)
        if self.lowercase:
            cleaned = cleaned.lower()
        if self.remove_extra_whitespace:
            cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
        if self.max_length is not None and len(cleaned) > self.max_length:
            cleaned = cleaned[:self.max_length]
 
        record["text"] = cleaned
        logger.debug({
            "event": "clean_text",
            "orig_len": len(raw),
            "cleaned_len": len(cleaned),
        })
        return record
 
    def _get_params(self) -> Dict[str, Any]:
        return {
            "lowercase": self.lowercase,
            "strip_control_chars": self.strip_control_chars,
            "remove_extra_whitespace": self.remove_extra_whitespace,
            "max_length": self.max_length,
        }
 
 
# @register_transform("tokenize_text")
class TokenizeText(Transform):
    """Tokenize text using a HuggingFace ``AutoTokenizer``.
 
    Stores ``input_ids`` and ``attention_mask`` (plain Python lists) back
    into the record.  The tokenizer is loaded lazily on first call and
    cached on the instance to avoid repeated disk I/O.
 
    Config keys (``transforms.token``):
 
    * ``model_name`` (str, default ``"bert-base-uncased"``)
    * ``max_length`` (int, default ``128``)
    * ``truncation`` (bool, default ``True``)
    * ``padding`` (bool, default ``False``)
    """
 
    def __init__(self, return_tensors: Optional[str] = None) -> None:
        super().__init__()
        self.token_cfg: Dict[str, Any] = get_config_section("transforms").get("token", {})
        self.model_name: str = str(self.token_cfg.get("model_name", "bert-base-uncased"))
        self.max_length: int = int(self.token_cfg.get("max_length", 128))
        self.truncation: bool = bool(self.token_cfg.get("truncation", True))
        self.padding: bool = bool(self.token_cfg.get("padding", False))
        self.return_tensors: Optional[str] = return_tensors
        self._tokenizer = None  # lazy init
 
    def _get_tokenizer(self):
        if self._tokenizer is None:
            if not TRANSFORMERS_AVAILABLE:
                raise DataConfigError(
                    "TokenizeText requires the transformers library: pip install transformers",
                    context={"model_name": self.model_name},
                )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # type: ignore
            logger.info({"event": "tokenizer_loaded", "model_name": self.model_name})
        return self._tokenizer
 
    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        if modality != "text":
            return record
 
        text = record.get("text")
        if text is None or text == "":
            record["input_ids"] = []
            record["attention_mask"] = []
            return record
        if not isinstance(text, str):
            raise DataTransformError(
                "TokenizeText expected record['text'] to be a str",
                context={"modality": modality, "got": type(text).__name__},
            )
 
        try:
            tokenizer = self._get_tokenizer()
            encoding = tokenizer(
                text,
                max_length=self.max_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors=self.return_tensors,
            )
        except (DataConfigError, DataTransformError):
            raise
        except Exception as exc:
            raise DataTransformError(
                "TokenizeText failed",
                context={"modality": modality, "model_name": self.model_name},
                cause=exc,
            ) from exc
 
        # Convert to plain lists regardless of return_tensors setting.
        ids = encoding["input_ids"]
        mask = encoding.get("attention_mask")
        record["input_ids"] = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        if mask is not None:
            record["attention_mask"] = mask.tolist() if hasattr(mask, "tolist") else list(mask)
        logger.debug({
            "event": "text_tokenized",
            "model_name": self.model_name,
            "num_tokens": len(record["input_ids"]),
        })
        return record
 
    def _get_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "truncation": self.truncation,
            "padding": self.padding,
            "return_tensors": self.return_tensors,
        }
 
 
# @register_transform("truncate_text")
class TruncateText(Transform):
    """Hard character-level truncation without tokenization.
 
    Useful as a lightweight safety guard before heavier transforms.
 
    Config key: ``transforms.truncate.max_chars`` (default ``1000``).
    """
 
    def __init__(self) -> None:
        super().__init__()
        self.tt_cfg: Dict[str, Any] = get_config_section("transforms").get("truncate", {})
        self.max_chars: int = int(self.tt_cfg.get("max_chars", 1000))
 
    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        if modality != "text":
            return record
        text = record.get("text")
        if isinstance(text, str) and len(text) > self.max_chars:
            record["text"] = text[:self.max_chars]
        return record
 
    def _get_params(self) -> Dict[str, Any]:
        return {"max_chars": self.max_chars}


if __name__ == "__main__":
    print("\n=== Running text ===\n")
    printer.status("TEST", "text initialized", "info")
 
    # CleanText — basic pipeline
    ct = CleanText(max_length=20)
    rec = {"text": "  Hello\x00 WORLD\t!  "}
    out = ct(rec, "text")
    assert out["text"] == "hello world !", f"got: {out['text']!r}"
    printer.status("PASS", "CleanText strips, lowercases, removes control chars", "success")
 
    # CleanText — truncation
    rec_long = {"text": "a" * 50}
    out_long = ct(rec_long, "text")
    assert len(out_long["text"]) == 20
    printer.status("PASS", "CleanText truncates at max_length", "success")
 
    # CleanText — skips non-text modality
    out_skip = ct({"text": "ABC"}, "audio")
    assert out_skip["text"] == "ABC"
    printer.status("PASS", "CleanText skips non-text modality", "success")
 
    # CleanText — raises on non-string field
    try:
        ct({"text": 42}, "text")
        assert False
    except DataTransformError:
        printer.status("PASS", "CleanText raises DataTransformError on non-str", "success")
 
    # CleanText — None text is passed through
    out_none = ct({"text": None}, "text")
    assert out_none["text"] is None
    printer.status("PASS", "CleanText passthrough on None text", "success")
 
    # TruncateText
    tt = TruncateText()
    rec_t = {"text": "x" * 2000}
    out_t = tt(rec_t, "text")
    assert len(out_t["text"]) == tt.max_chars
    printer.status("PASS", "TruncateText truncates correctly", "success")
 
    out_t2 = tt({"text": "short"}, "text")
    assert out_t2["text"] == "short"
    printer.status("PASS", "TruncateText leaves short text unchanged", "success")
 
    # TokenizeText — empty string produces empty lists
    if not TRANSFORMERS_AVAILABLE:
        printer.status("SKIP", "TokenizeText test skipped (transformers not installed)", "warning")
    else:
        tkt = TokenizeText()
        out_e = tkt({"text": ""}, "text")
        assert out_e["input_ids"] == [] and out_e["attention_mask"] == []
        printer.status("PASS", "TokenizeText returns empty lists for empty text", "success")
 
    # _get_params coverage
    assert "max_length" in CleanText()._get_params()
    assert "max_chars" in TruncateText()._get_params()
    assert "model_name" in TokenizeText()._get_params()
    printer.status("PASS", "_get_params returns expected keys", "success")
 
    print("\n=== Test ran successfully ===\n")