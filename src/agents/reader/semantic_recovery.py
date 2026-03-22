import re

from typing import Dict


class SemanticRecovery:
    """
    Conservative semantic recovery focused on preserving human-readable tokens
    without inventing new text.

    Design for efficiency:
    - Precompiled regex patterns.
    - Chunked processing for large documents.
    - Lightweight confidence scoring based on salvage ratio and corruption signals.
    """

    _TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/.,:;!?()'\"]*")
    _CORRUPTION_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uFFFD]")

    def __init__(self, chunk_size: int = 20000, max_output_chars: int = 2_000_000):
        self.chunk_size = max(1024, chunk_size)
        self.max_output_chars = max_output_chars

    def _iter_chunks(self, text: str):
        for i in range(0, len(text), self.chunk_size):
            yield text[i : i + self.chunk_size]

    def recover(self, raw_text: str) -> Dict[str, float | str | int]:
        if not raw_text:
            return {
                "recovered_text": "[CORRUPTED_DATA]",
                "confidence": 0.0,
                "token_count": 0,
                "corruption_ratio": 1.0,
            }

        recovered_parts: list[str] = []
        token_count = 0
        corruption_hits = 0

        for chunk in self._iter_chunks(raw_text):
            tokens = self._TOKEN_PATTERN.findall(chunk)
            if tokens:
                recovered_parts.append(" ".join(tokens))
                token_count += len(tokens)
            corruption_hits += len(self._CORRUPTION_PATTERN.findall(chunk))

            # Guardrail against runaway memory usage on huge files.
            if sum(len(part) for part in recovered_parts) >= self.max_output_chars:
                break

        cleaned = " ".join(recovered_parts).strip()
        corruption_ratio = corruption_hits / max(1, len(raw_text))

        if not cleaned:
            return {
                "recovered_text": "[CORRUPTED_DATA]",
                "confidence": 0.0,
                "token_count": 0,
                "corruption_ratio": round(corruption_ratio, 5),
            }

        lexical_density = token_count / max(1, len(raw_text.split()))
        confidence = min(0.95, max(0.05, lexical_density * (1.0 - corruption_ratio)))

        return {
            "recovered_text": cleaned,
            "confidence": round(confidence, 3),
            "token_count": token_count,
            "corruption_ratio": round(corruption_ratio, 5),
        }
