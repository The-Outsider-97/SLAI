from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Dict

from src.agents.reader.utils.reader_error import RecoveryFailureError
from src.agents.reader.semantic_recovery import SemanticRecovery
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Recovery Engine")
printer = PrettyPrinter


class RecoveryEngine:
    """
    Multi-pass recovery engine:
      1) Fast low-level cleanup for common corruption artifacts.
      2) Heuristic quality check on cleaned output.
      3) Semantic recovery fallback when quality is poor.

    Efficiency notes:
    - Caches recovered outputs by source+content hash.
    - Uses inexpensive heuristics before semantic pass.
    """

    def __init__(self, min_quality_score: float = 0.55, semantic_threshold: float = 0.40, cache_size: int = 256):
        self.semantic = SemanticRecovery()
        self.min_quality_score = max(0.0, min(1.0, min_quality_score))
        self.semantic_threshold = max(0.0, min(1.0, semantic_threshold))
        self.cache_size = max(16, cache_size)
        self._recovery_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        logger.info(f"Recovery Engine successfully Initialized")

    def _cache_key(self, parsed_doc: Dict[str, Any]) -> str:
        source = str(parsed_doc.get("source", "unknown"))
        content = str(parsed_doc.get("content", ""))
        digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        return f"{source}:{digest}:{self.min_quality_score}:{self.semantic_threshold}"

    def _set_cache(self, key: str, value: Dict[str, Any]) -> None:
        self._recovery_cache[key] = value
        self._recovery_cache.move_to_end(key)
        while len(self._recovery_cache) > self.cache_size:
            self._recovery_cache.popitem(last=False)

    def _quality_score(self, text: str) -> float:
        if not text:
            return 0.0
        printable = sum(ch.isprintable() for ch in text) / len(text)
        alnum = sum(ch.isalnum() for ch in text) / len(text)
        control_chars = sum(ord(ch) < 32 and ch not in "\n\r\t" for ch in text) / len(text)
        return max(0.0, min(1.0, (0.60 * printable) + (0.40 * alnum) - (0.80 * control_chars)))

    def _low_level_repair(self, content: str) -> str:
        repaired = content.replace("\x00", "")
        repaired = repaired.replace("\r\n", "\n").replace("\r", "\n")
        repaired = repaired.replace("\uFFFD", "")
        return repaired

    def recover_document(self, parsed_doc: Dict[str, Any]) -> Dict[str, Any]:
        key = self._cache_key(parsed_doc)
        if key in self._recovery_cache:
            cached = dict(self._recovery_cache[key])
            cached["cached"] = True
            return cached

        source = str(parsed_doc.get("source", "unknown"))
        try:
            content = str(parsed_doc.get("content", ""))
            low_level = self._low_level_repair(content)
            quality_before = self._quality_score(content)
            quality_after = self._quality_score(low_level)

            use_semantic = (
                not low_level.strip() or quality_after < self.min_quality_score or quality_before < self.semantic_threshold
            )

            if not use_semantic:
                result = {
                    "status": "ok",
                    "strategy": "low_level",
                    "content": low_level,
                    "confidence": round(quality_after, 3),
                    "quality_before": round(quality_before, 3),
                    "quality_after": round(quality_after, 3),
                    "cached": False,
                    "metadata": {"used_semantic": False},
                }
                self._set_cache(key, result)
                return result

            semantic_result = self.semantic.recover(low_level)
            result = {
                "status": "ok",
                "strategy": "semantic",
                "content": semantic_result["recovered_text"],
                "confidence": semantic_result["confidence"],
                "quality_before": round(quality_before, 3),
                "quality_after": round(self._quality_score(semantic_result["recovered_text"]), 3),
                "cached": False,
                "metadata": {
                    "used_semantic": True,
                    "token_count": semantic_result.get("token_count", 0),
                    "corruption_ratio": semantic_result.get("corruption_ratio", 0.0),
                    "chunk_count": semantic_result.get("chunk_count", 0),
                },
            }
            self._set_cache(key, result)
            return result
        except Exception as exc:
            raise RecoveryFailureError(source, f"Failed recovering document '{source}': {exc}", cause=exc) from exc
