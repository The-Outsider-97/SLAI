import hashlib

from typing import Dict, Any

from src.agents.reader.semantic_recovery import SemanticRecovery


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

    def __init__(self, min_quality_score: float = 0.55):
        self.semantic = SemanticRecovery()
        self.min_quality_score = min_quality_score
        self._recovery_cache: dict[str, Dict[str, Any]] = {}

    def _cache_key(self, parsed_doc: Dict[str, Any]) -> str:
        source = parsed_doc.get("source", "unknown")
        content = parsed_doc.get("content", "")
        digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        return f"{source}:{digest}"

    def _quality_score(self, text: str) -> float:
        if not text:
            return 0.0

        printable_chars = sum(ch.isprintable() for ch in text)
        printable_ratio = printable_chars / max(1, len(text))
        alpha_num_ratio = sum(ch.isalnum() for ch in text) / max(1, len(text))
        null_penalty = text.count("\x00") / max(1, len(text))

        score = (0.6 * printable_ratio) + (0.4 * alpha_num_ratio) - (0.75 * null_penalty)
        return max(0.0, min(1.0, score))

    def _low_level_repair(self, content: str) -> str:
        repaired = content.replace("\x00", "")
        repaired = repaired.replace("\r\n", "\n").replace("\r", "\n")
        return repaired

    def recover_document(self, parsed_doc: Dict[str, Any]) -> Dict[str, Any]:
        key = self._cache_key(parsed_doc)
        if key in self._recovery_cache:
            cached = dict(self._recovery_cache[key])
            cached["cached"] = True
            return cached

        content = parsed_doc.get("content", "")
        low_level = self._low_level_repair(content)
        quality = self._quality_score(low_level)

        if quality >= self.min_quality_score and low_level.strip():
            result = {
                "strategy": "low_level",
                "content": low_level,
                "confidence": round(quality, 3),
                "quality_score": round(quality, 3),
                "cached": False,
            }
            self._recovery_cache[key] = result
            return result

        semantic_result = self.semantic.recover(content)
        result = {
            "strategy": "semantic",
            "content": semantic_result["recovered_text"],
            "confidence": semantic_result["confidence"],
            "quality_score": round(quality, 3),
            "cached": False,
            "semantic_stats": {
                "token_count": semantic_result.get("token_count", 0),
                "corruption_ratio": semantic_result.get("corruption_ratio", 0.0),
            },
        }
        self._recovery_cache[key] = result
        return result
