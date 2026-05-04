from __future__ import annotations

import hashlib
import os

from pathlib import Path
from typing import Any, Dict, List

from src.agents.reader.utils.config_loader import get_config_section
from src.agents.reader.utils.reader_error import (
    FileMissingError,
    ParseFailureError,
    UnsupportedFormatError,
)


class ParserEngine:
    """Normalizes supported file inputs into deterministic parse results."""

    _TEXT_EXTENSIONS = {".txt", ".md", ".html", ".xml", ".json", ".csv"}

    def __init__(self) -> None:
        reader_config = get_config_section("reader")
        allowed = reader_config.get("allowed_input_extensions") or sorted(self._TEXT_EXTENSIONS)
        self.allowed_extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in allowed}
        self.max_file_size_bytes = int(reader_config.get("max_file_size_bytes", 50 * 1024 * 1024))
        self.encoding_candidates: List[str] = list(reader_config.get("encoding_candidates", ["utf-8", "utf-8-sig", "latin-1"]))

    def _validate_input(self, file_path: str) -> Path:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileMissingError(str(path))
        if not path.is_file():
            raise ParseFailureError(str(path), "Input path is not a regular file")

        extension = path.suffix.lower()
        if extension not in self.allowed_extensions:
            raise UnsupportedFormatError(str(path), extension, self.allowed_extensions)

        size = path.stat().st_size
        if size > self.max_file_size_bytes:
            raise ParseFailureError(
                str(path),
                f"File exceeds max allowed size of {self.max_file_size_bytes} bytes",
            )
        return path

    def _decode_bytes(self, raw_bytes: bytes) -> tuple[str, str, bool]:
        for encoding in self.encoding_candidates:
            try:
                return raw_bytes.decode(encoding), encoding, False
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="replace"), "utf-8-replace", True

    def parse(self, file_path: str) -> Dict[str, Any]:
        path = self._validate_input(file_path)
        extension = path.suffix.lower()

        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            raise ParseFailureError(str(path), f"Failed to read file: {exc}", cause=exc) from exc

        content, encoding, had_decode_fallback = self._decode_bytes(raw_bytes)
        is_probably_binary = b"\x00" in raw_bytes[:4096]

        warnings: List[str] = []
        if had_decode_fallback:
            warnings.append("lossy_decode_fallback_used")
        if is_probably_binary and extension in self._TEXT_EXTENSIONS:
            warnings.append("binary_signature_detected")
        if not content.strip():
            warnings.append("empty_or_whitespace_content")

        return {
            "status": "ok",
            "source": str(path),
            "extension": extension,
            "content": content,
            "metadata": {
                "size": len(raw_bytes),
                "mtime": path.stat().st_mtime,
                "encoding": encoding,
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "line_count": content.count("\n") + (1 if content else 0),
                "char_count": len(content),
                "is_binary_signature": is_probably_binary,
            },
            "warnings": warnings,
        }
