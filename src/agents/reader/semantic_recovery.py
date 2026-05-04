from __future__ import annotations

import re
from typing import Dict, Iterator


class SemanticRecovery:
    """Conservative non-fabricating salvage of readable tokens from corrupted text."""

    _TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/.,:;!?()'\"]*")
    _CORRUPTION_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uFFFD]")

    def __init__(self, chunk_size: int = 20_000, max_output_chars: int = 2_000_000):
        self.chunk_size = max(1_024, chunk_size)
        self.max_output_chars = max(32_768, max_output_chars)

    def _iter_chunks(self, text: str) -> Iterator[str]:
        for index in range(0, len(text), self.chunk_size):
            yield text[index : index + self.chunk_size]

    def recover(self, raw_text: str) -> Dict[str, float | str | int]:
        if not raw_text:
            return {
                "recovered_text": "[CORRUPTED_DATA]",
                "confidence": 0.0,
                "token_count": 0,
                "corruption_ratio": 1.0,
                "chunk_count": 0,
            }

        recovered_parts: list[str] = []
        token_count = 0
        corruption_hits = 0
        chars_emitted = 0
        chunk_count = 0

        for chunk in self._iter_chunks(raw_text):
            chunk_count += 1
            corruption_hits += len(self._CORRUPTION_PATTERN.findall(chunk))
            tokens = self._TOKEN_PATTERN.findall(chunk)
            if not tokens:
                continue
            recovered_chunk = " ".join(tokens)
            projected = chars_emitted + len(recovered_chunk)
            if projected > self.max_output_chars:
                remaining = max(0, self.max_output_chars - chars_emitted)
                if remaining > 0:
                    recovered_parts.append(recovered_chunk[:remaining])
                    chars_emitted += remaining
                break
            recovered_parts.append(recovered_chunk)
            token_count += len(tokens)
            chars_emitted += len(recovered_chunk)

        cleaned = " ".join(recovered_parts).strip()
        corruption_ratio = corruption_hits / max(1, len(raw_text))
        lexical_density = token_count / max(1, len(raw_text.split()))

        if not cleaned:
            return {
                "recovered_text": "[CORRUPTED_DATA]",
                "confidence": 0.0,
                "token_count": 0,
                "corruption_ratio": round(corruption_ratio, 5),
                "chunk_count": chunk_count,
            }

        confidence = lexical_density * (1.0 - corruption_ratio)
        confidence = min(0.95, max(0.05, confidence))

        return {
            "recovered_text": cleaned,
            "confidence": round(confidence, 3),
            "token_count": token_count,
            "corruption_ratio": round(corruption_ratio, 5),
            "chunk_count": chunk_count,
        }
