from __future__ import annotations

"""Production-grade semantic recovery for the Reader subsystem.

``SemanticRecovery`` is the Reader subsystem's conservative semantic salvage
layer. It is intentionally **non-fabricating**: it never reconstructs missing
meaning, never invents words, and never expands partial content with guessed
context. It only keeps readable spans and tokens that are already present in the
input, then reports enough metrics for ``RecoveryEngine`` to decide whether the
salvage is acceptable.

Integration contract
--------------------
The legacy ``recover(raw_text)`` keys are preserved because ``RecoveryEngine``
currently consumes them directly:

- ``recovered_text``
- ``confidence``
- ``token_count``
- ``corruption_ratio``
- ``chunk_count``

The method also returns additional production metadata, warnings, hashes, and a
``content`` alias for newer helper-based flows.
"""

import re

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.reader_error import *
from ..utils.reader_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Semantic Recovery")
printer = PrettyPrinter()


@dataclass(frozen=True)
class SemanticRecoveryProfile:
    """Quality and observability profile for one semantic recovery pass."""

    input_char_count: int
    input_line_count: int
    output_char_count: int
    output_line_count: int
    token_count: int
    unique_token_count: int
    corruption_hits: int
    corruption_ratio: float
    control_char_ratio: float
    lexical_density: float
    output_ratio: float
    quality_before: float
    quality_after: float
    chunk_count: int
    truncated: bool
    placeholder_used: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


class SemanticRecovery:
    """Conservative non-fabricating salvage of readable text from corruption.

    The class does not perform low-level filesystem or parser work. It focuses
    on text-level recovery only and delegates shared concerns to
    ``reader_helpers.py``: chunking, hashing, redaction-safe JSON conversion,
    quality scoring, truncation, de-duplication, and parsed-document validation.
    """

    _TOKEN_PATTERN = re.compile(r"[\w][\w\-_/.,:;!?()'\"@#%&+=]*", re.UNICODE)
    _ALNUM_PATTERN = re.compile(r"[\w]", re.UNICODE)
    _READABLE_SPAN_PATTERN = re.compile(r"[^\x00-\x08\x0B\x0C\x0E-\x1F\uFFFD]+", re.UNICODE)
    _CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uFFFD]")
    _SPACE_PATTERN = re.compile(r"[ \t\f\v]+")
    _LINEBREAK_PATTERN = re.compile(r"\n+")

    def __init__(self, chunk_size: Optional[int] = None, max_output_chars: Optional[int] = None) -> None:
        self.config = load_global_config()
        self.recovery_config = get_config_section("semantic_recovery") or {}

        self.chunk_size = self._cfg_int("chunk_size", chunk_size if chunk_size is not None else 20_000, minimum=1_024)
        self.chunk_overlap = self._cfg_int("chunk_overlap", 0, minimum=0)
        self.chunk_overlap = min(self.chunk_overlap, max(0, self.chunk_size - 1))
        self.max_chunks = self._cfg_optional_int("max_chunks", minimum=1)

        self.max_output_chars = self._cfg_int(
            "max_output_chars",
            max_output_chars if max_output_chars is not None else 2_000_000,
            minimum=32_768,
        )
        self.min_span_chars = self._cfg_int("min_span_chars", 1, minimum=1)
        self.max_span_chars = self._cfg_int("max_span_chars", 8_000, minimum=self.min_span_chars)
        self.min_token_length = self._cfg_int("min_token_length", 1, minimum=1)
        self.max_token_length = self._cfg_int("max_token_length", 256, minimum=self.min_token_length)
        self.min_tokens_per_span = self._cfg_int("min_tokens_per_span", 1, minimum=0)
        self.min_span_quality = self._cfg_float("min_span_quality", 0.20, minimum=0.0, maximum=1.0)

        self.min_confidence = self._cfg_float("min_confidence", 0.05, minimum=0.0, maximum=1.0)
        self.max_confidence = self._cfg_float("max_confidence", 0.95, minimum=0.0, maximum=1.0)
        if self.max_confidence < self.min_confidence:
            self.max_confidence = self.min_confidence
        self.min_success_confidence = self._cfg_float("min_success_confidence", 0.10, minimum=0.0, maximum=1.0)
        self.max_corruption_ratio = self._cfg_float("max_corruption_ratio", 0.85, minimum=0.0, maximum=1.0)
        self.max_control_char_ratio = self._cfg_float("max_control_char_ratio", 0.35, minimum=0.0, maximum=1.0)

        self.preserve_line_breaks = self._cfg_bool("preserve_line_breaks", True)
        self.normalize_whitespace = self._cfg_bool("normalize_whitespace", True)
        self.drop_numeric_only_spans = self._cfg_bool("drop_numeric_only_spans", False)
        self.remove_duplicate_lines_window = self._cfg_int("remove_duplicate_lines_window", 32, minimum=0, maximum=1_024)
        self.remove_duplicate_tokens_window = self._cfg_int("remove_duplicate_tokens_window", 0, minimum=0, maximum=1_024)

        self.corrupted_placeholder = self._cfg_str("corrupted_placeholder", "[CORRUPTED_DATA]")
        self.low_confidence_placeholder = self._cfg_str("low_confidence_placeholder", "[CORRUPTED_DATA]")
        self.include_debug_preview = self._cfg_bool("include_debug_preview", False)
        self.debug_preview_chars = self._cfg_int("debug_preview_chars", 240, minimum=0, maximum=5_000)

        self._validate_config()
        logger.info(
            "Semantic Recovery initialized | chunk_size=%s overlap=%s max_output_chars=%s",
            self.chunk_size,
            self.chunk_overlap,
            self.max_output_chars,
        )

    # ------------------------------------------------------------------
    # Config access. This intentionally uses the existing config loader as-is.
    # ------------------------------------------------------------------

    def _cfg_value(self, key: str, default: Any) -> Any:
        return self.recovery_config.get(key, default)

    def _cfg_int(
        self,
        key: str,
        default: int,
        *,
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
    ) -> int:
        raw = self._cfg_value(key, default)
        try:
            return coerce_int(raw, default, minimum=minimum, maximum=maximum)
        except Exception as exc:
            raise self._configuration_error(
                f"semantic_recovery.{key} must be an integer",
                {"key": key, "value": raw, "default": default},
                exc,
            ) from exc

    def _cfg_optional_int(self, key: str, *, minimum: int = 1) -> Optional[int]:
        raw = self.recovery_config.get(key)
        if raw in (None, "", 0, "0"):
            return None
        try:
            return coerce_int(raw, minimum, minimum=minimum)
        except Exception as exc:
            raise self._configuration_error(
                f"semantic_recovery.{key} must be an integer or null",
                {"key": key, "value": raw},
                exc,
            ) from exc

    def _cfg_float(
        self,
        key: str,
        default: float,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        raw = self._cfg_value(key, default)
        try:
            return coerce_float(raw, default, minimum=minimum, maximum=maximum)
        except Exception as exc:
            raise self._configuration_error(
                f"semantic_recovery.{key} must be a number",
                {"key": key, "value": raw, "default": default},
                exc,
            ) from exc

    def _cfg_bool(self, key: str, default: bool) -> bool:
        raw = self._cfg_value(key, default)
        return coerce_bool(raw, default)

    def _cfg_str(self, key: str, default: str) -> str:
        value = str(self._cfg_value(key, default) or default)
        return value

    @staticmethod
    def _configuration_error(
        message: str,
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> ReaderError:
        error_cls = globals().get("ReaderConfigurationError")
        if isinstance(error_cls, type):
            return error_cls(message, context=dict(context or {}), cause=cause)
        return ReaderValidationError(message, dict(context or {}))

    def _validate_config(self) -> None:
        if self.min_token_length > self.max_token_length:
            raise self._configuration_error(
                "semantic_recovery.min_token_length cannot exceed semantic_recovery.max_token_length",
                {"min_token_length": self.min_token_length, "max_token_length": self.max_token_length},
            )
        if self.min_span_chars > self.max_span_chars:
            raise self._configuration_error(
                "semantic_recovery.min_span_chars cannot exceed semantic_recovery.max_span_chars",
                {"min_span_chars": self.min_span_chars, "max_span_chars": self.max_span_chars},
            )
        if not self.corrupted_placeholder.strip():
            raise self._configuration_error("semantic_recovery.corrupted_placeholder cannot be empty")
        if not self.low_confidence_placeholder.strip():
            raise self._configuration_error("semantic_recovery.low_confidence_placeholder cannot be empty")

    def _settings_snapshot(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_chunks": self.max_chunks,
            "max_output_chars": self.max_output_chars,
            "min_span_chars": self.min_span_chars,
            "max_span_chars": self.max_span_chars,
            "min_token_length": self.min_token_length,
            "max_token_length": self.max_token_length,
            "min_tokens_per_span": self.min_tokens_per_span,
            "min_span_quality": self.min_span_quality,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "min_success_confidence": self.min_success_confidence,
            "max_corruption_ratio": self.max_corruption_ratio,
            "max_control_char_ratio": self.max_control_char_ratio,
            "preserve_line_breaks": self.preserve_line_breaks,
            "normalize_whitespace": self.normalize_whitespace,
            "drop_numeric_only_spans": self.drop_numeric_only_spans,
            "remove_duplicate_lines_window": self.remove_duplicate_lines_window,
            "remove_duplicate_tokens_window": self.remove_duplicate_tokens_window,
            "include_debug_preview": self.include_debug_preview,
            "debug_preview_chars": self.debug_preview_chars,
        }

    # ------------------------------------------------------------------
    # Salvage primitives.
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for token in self._TOKEN_PATTERN.findall(text):
            if self.min_token_length <= len(token) <= self.max_token_length:
                tokens.append(token)
        if self.remove_duplicate_tokens_window <= 0:
            return tokens

        output: List[str] = []
        recent: List[str] = []
        for token in tokens:
            marker = token.lower()
            if marker in recent:
                continue
            output.append(token)
            recent.append(marker)
            if len(recent) > self.remove_duplicate_tokens_window:
                recent.pop(0)
        return output

    def _normalize_span(self, span: str) -> str:
        value = low_level_repair_text(span)
        value = self._SPACE_PATTERN.sub(" ", value)
        value = self._LINEBREAK_PATTERN.sub("\n", value)
        value = value.strip()
        if len(value) > self.max_span_chars:
            value = truncate_text(value, self.max_span_chars, suffix="")
        return value.strip()

    def _span_is_readable(self, span: str) -> Tuple[bool, List[str]]:
        warnings: List[str] = []
        if len(span) < self.min_span_chars:
            return False, ["semantic_recovery_span_too_short"]
        if not self._ALNUM_PATTERN.search(span):
            return False, ["semantic_recovery_span_without_alnum"]

        tokens = self._tokenize(span)
        if len(tokens) < self.min_tokens_per_span:
            return False, ["semantic_recovery_span_below_token_floor"]
        if self.drop_numeric_only_spans and tokens and all(token.replace(".", "", 1).isdigit() for token in tokens):
            return False, ["semantic_recovery_numeric_only_span_dropped"]

        quality = text_quality_report(span)
        if quality.quality_score < self.min_span_quality:
            warnings.append("semantic_recovery_low_quality_span_dropped")
            return False, warnings
        return True, list(quality.warnings)

    def _iter_readable_spans(self, chunk: str) -> Iterable[str]:
        for raw_span in self._READABLE_SPAN_PATTERN.findall(chunk):
            span = self._normalize_span(raw_span)
            if not span:
                continue
            is_readable, _warnings = self._span_is_readable(span)
            if is_readable:
                yield span

    def _dedupe_spans(self, spans: Sequence[str]) -> List[str]:
        if self.remove_duplicate_lines_window <= 0:
            return list(spans)
        output: List[str] = []
        recent: List[str] = []
        for span in spans:
            marker = self._SPACE_PATTERN.sub(" ", span.lower()).strip()
            if marker in recent:
                continue
            output.append(span)
            recent.append(marker)
            if len(recent) > self.remove_duplicate_lines_window:
                recent.pop(0)
        return output

    def _salvage_chunk(self, chunk: str) -> Tuple[str, int, List[str]]:
        spans = self._dedupe_spans(list(self._iter_readable_spans(chunk)))
        if not spans:
            return "", 0, ["semantic_recovery_no_readable_span_in_chunk"]

        joiner = "\n" if self.preserve_line_breaks else " "
        recovered = joiner.join(spans).strip()
        if self.normalize_whitespace:
            if self.preserve_line_breaks:
                recovered = "\n".join(self._SPACE_PATTERN.sub(" ", line).strip() for line in recovered.splitlines() if line.strip())
            else:
                recovered = self._SPACE_PATTERN.sub(" ", recovered).strip()
        tokens = self._tokenize(recovered)
        warnings = detect_text_warnings(recovered)
        return recovered, len(tokens), warnings

    def _append_with_limit(self, parts: List[str], text: str, current_chars: int) -> Tuple[int, bool]:
        if not text:
            return current_chars, False
        separator_len = 1 if parts else 0
        projected = current_chars + separator_len + len(text)
        if projected <= self.max_output_chars:
            parts.append(text)
            return projected, False

        remaining = self.max_output_chars - current_chars - separator_len
        if remaining > 0:
            parts.append(truncate_text(text, remaining, suffix=""))
            return self.max_output_chars, True
        return current_chars, True

    # ------------------------------------------------------------------
    # Scoring and result shaping.
    # ------------------------------------------------------------------

    def _build_profile(
        self,
        raw_text: str,
        recovered_text: str,
        *,
        token_count: int,
        unique_tokens: Iterable[str],
        corruption_hits: int,
        chunk_count: int,
        truncated: bool,
        placeholder_used: bool,
        warnings: Iterable[str],
    ) -> SemanticRecoveryProfile:
        raw_report = text_quality_report(raw_text)
        recovered_report = text_quality_report(recovered_text)
        input_len = len(raw_text)
        output_len = len(recovered_text)
        corruption_ratio = round(corruption_hits / max(1, input_len), 5)
        lexical_density = round(token_count / max(1, len(str(raw_text).split())), 5)
        output_ratio = round(output_len / max(1, input_len), 5)

        combined_warnings = dedupe_preserve_order(
            [
                *warnings,
                *raw_report.warnings,
                *recovered_report.warnings,
            ]
        )
        if corruption_ratio > self.max_corruption_ratio:
            combined_warnings.append("semantic_recovery_corruption_ratio_above_limit")
        if raw_report.control_char_ratio > self.max_control_char_ratio:
            combined_warnings.append("semantic_recovery_control_ratio_above_limit")
        if truncated:
            combined_warnings.append("semantic_recovery_output_truncated")
        if placeholder_used:
            combined_warnings.append("semantic_recovery_placeholder_used")

        return SemanticRecoveryProfile(
            input_char_count=input_len,
            input_line_count=raw_report.line_count,
            output_char_count=output_len,
            output_line_count=recovered_report.line_count,
            token_count=token_count,
            unique_token_count=len({token.lower() for token in unique_tokens}),
            corruption_hits=corruption_hits,
            corruption_ratio=corruption_ratio,
            control_char_ratio=raw_report.control_char_ratio,
            lexical_density=lexical_density,
            output_ratio=output_ratio,
            quality_before=raw_report.quality_score,
            quality_after=recovered_report.quality_score,
            chunk_count=chunk_count,
            truncated=truncated,
            placeholder_used=placeholder_used,
            warnings=tuple(dedupe_preserve_order(combined_warnings)),
        )

    def _confidence(self, profile: SemanticRecoveryProfile) -> float:
        if profile.placeholder_used or profile.output_char_count == 0:
            return 0.0

        quality_component = profile.quality_after * 0.42
        corruption_component = max(0.0, 1.0 - profile.corruption_ratio) * 0.28
        density_component = min(1.0, profile.lexical_density) * 0.15
        retention_component = min(1.0, profile.output_ratio * 2.5) * 0.10
        uniqueness_component = min(1.0, profile.unique_token_count / max(1, profile.token_count)) * 0.05

        penalty = 0.0
        if profile.corruption_ratio > self.max_corruption_ratio:
            penalty += 0.30
        if profile.control_char_ratio > self.max_control_char_ratio:
            penalty += 0.20
        if profile.quality_after < self.min_span_quality:
            penalty += 0.15
        if profile.token_count == 0:
            penalty += 0.25
        if profile.truncated:
            penalty += 0.03

        confidence = quality_component + corruption_component + density_component + retention_component + uniqueness_component - penalty
        return round(clamp(confidence, self.min_confidence, self.max_confidence), 3)

    def _failure_result(
        self,
        raw_text: str,
        *,
        reason: str,
        chunk_count: int = 0,
        corruption_hits: Optional[int] = None,
        warnings: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        text = str(raw_text or "")
        hits = corruption_hits if corruption_hits is not None else len(self._CONTROL_PATTERN.findall(text))
        profile = self._build_profile(
            text,
            self.corrupted_placeholder,
            token_count=0,
            unique_tokens=[],
            corruption_hits=hits,
            chunk_count=chunk_count,
            truncated=False,
            placeholder_used=True,
            warnings=[reason, *(warnings or [])],
        )
        result: Dict[str, Any] = {
            "status": "failed",
            "strategy": "semantic_span_salvage",
            "recovered_text": self.corrupted_placeholder,
            "content": self.corrupted_placeholder,
            "confidence": 0.0,
            "token_count": 0,
            "corruption_ratio": profile.corruption_ratio,
            "chunk_count": chunk_count,
            "warnings": list(profile.warnings),
            "metadata": {
                "reason": reason,
                "profile": profile.to_dict(),
                "settings": self._settings_snapshot(),
                "input_sha256": sha256_text(text),
                "output_sha256": sha256_text(self.corrupted_placeholder),
            },
        }
        if self.include_debug_preview and self.debug_preview_chars > 0:
            result["metadata"]["input_preview"] = truncate_text(text, self.debug_preview_chars)
        return result

    def _result(
        self,
        *,
        status: str,
        raw_text: str,
        recovered_text: str,
        confidence: float,
        token_count: int,
        unique_tokens: Iterable[str],
        corruption_hits: int,
        chunk_count: int,
        truncated: bool,
        placeholder_used: bool,
        warnings: Iterable[str],
    ) -> Dict[str, Any]:
        profile = self._build_profile(
            raw_text,
            recovered_text,
            token_count=token_count,
            unique_tokens=unique_tokens,
            corruption_hits=corruption_hits,
            chunk_count=chunk_count,
            truncated=truncated,
            placeholder_used=placeholder_used,
            warnings=warnings,
        )
        result: Dict[str, Any] = {
            "status": status,
            "strategy": "semantic_span_salvage",
            "recovered_text": recovered_text,
            "content": recovered_text,
            "confidence": confidence,
            "token_count": token_count,
            "corruption_ratio": profile.corruption_ratio,
            "chunk_count": chunk_count,
            "warnings": list(profile.warnings),
            "metadata": {
                "profile": profile.to_dict(),
                "settings": self._settings_snapshot(),
                "input_sha256": sha256_text(raw_text),
                "output_sha256": sha256_text(recovered_text),
                "truncated": truncated,
                "placeholder_used": placeholder_used,
            },
        }
        if self.include_debug_preview and self.debug_preview_chars > 0:
            result["metadata"]["input_preview"] = truncate_text(raw_text, self.debug_preview_chars)
            result["metadata"]["output_preview"] = truncate_text(recovered_text, self.debug_preview_chars)
        return json_safe(result)

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def recover(self, raw_text: str) -> Dict[str, Any]:
        """Recover readable text from a possibly corrupted string.

        The operation is bounded by ``chunk_size``, ``max_chunks`` and
        ``max_output_chars``. It returns placeholders instead of questionable
        text when no readable spans are detected or confidence falls below the
        configured success threshold.
        """

        try:
            if raw_text is None:
                return self._failure_result("", reason="semantic_recovery_empty_input")

            text = str(raw_text)
            if not text:
                return self._failure_result("", reason="semantic_recovery_empty_input")

            chunk_spec = ChunkSpec(
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
                max_chunks=self.max_chunks,
            )
            recovered_parts: List[str] = []
            warnings: List[str] = []
            unique_tokens: List[str] = []
            token_count = 0
            corruption_hits = 0
            chars_emitted = 0
            chunk_count = 0
            truncated = False

            for chunk in iter_text_chunks(text, chunk_spec):
                chunk_count += 1
                corruption_hits += len(self._CONTROL_PATTERN.findall(chunk))
                recovered_chunk, recovered_token_count, chunk_warnings = self._salvage_chunk(chunk)
                warnings.extend(chunk_warnings)

                if not recovered_chunk:
                    continue

                tokens = self._tokenize(recovered_chunk)
                unique_tokens.extend(tokens)
                token_count += recovered_token_count
                chars_emitted, was_truncated = self._append_with_limit(recovered_parts, recovered_chunk, chars_emitted)
                truncated = truncated or was_truncated
                if truncated:
                    break

            joiner = "\n" if self.preserve_line_breaks else " "
            recovered_text = joiner.join(part for part in recovered_parts if part).strip()
            if self.normalize_whitespace:
                if self.preserve_line_breaks:
                    recovered_text = "\n".join(
                        self._SPACE_PATTERN.sub(" ", line).strip()
                        for line in recovered_text.splitlines()
                        if line.strip()
                    )
                else:
                    recovered_text = self._SPACE_PATTERN.sub(" ", recovered_text).strip()

            if not recovered_text:
                return self._failure_result(
                    text,
                    reason="semantic_recovery_no_readable_content",
                    chunk_count=chunk_count,
                    corruption_hits=corruption_hits,
                    warnings=warnings,
                )

            preliminary_profile = self._build_profile(
                text,
                recovered_text,
                token_count=token_count,
                unique_tokens=unique_tokens,
                corruption_hits=corruption_hits,
                chunk_count=chunk_count,
                truncated=truncated,
                placeholder_used=False,
                warnings=warnings,
            )
            confidence = self._confidence(preliminary_profile)

            if confidence < self.min_success_confidence:
                final_text = self.low_confidence_placeholder
                return self._result(
                    status="low_confidence",
                    raw_text=text,
                    recovered_text=final_text,
                    confidence=0.0,
                    token_count=token_count,
                    unique_tokens=unique_tokens,
                    corruption_hits=corruption_hits,
                    chunk_count=chunk_count,
                    truncated=truncated,
                    placeholder_used=True,
                    warnings=[*warnings, "semantic_recovery_confidence_below_success_threshold"],
                )

            return self._result(
                status="ok",
                raw_text=text,
                recovered_text=recovered_text,
                confidence=confidence,
                token_count=token_count,
                unique_tokens=unique_tokens,
                corruption_hits=corruption_hits,
                chunk_count=chunk_count,
                truncated=truncated,
                placeholder_used=False,
                warnings=warnings,
            )
        except ReaderError:
            raise
        except Exception as exc:
            raise RecoveryFailureError(
                "semantic_recovery",
                f"Semantic recovery failed: {exc}",
                cause=exc,
            ) from exc

    def recover_document(self, parsed_doc: Mapping[str, Any]) -> Dict[str, Any]:
        """Recover the ``content`` field of a parsed Reader document."""

        try:
            doc = validate_parsed_document(parsed_doc)
            source = str(doc.get("source", "unknown"))
            result = self.recover(str(doc.get("content", "")))
            updated = dict(doc)
            updated["content"] = result["recovered_text"]
            updated["semantic_recovery"] = result
            recovery_history = list(updated.get("recovery_history", []) or [])
            recovery_history.append(
                {
                    "strategy": result.get("strategy"),
                    "status": result.get("status"),
                    "confidence": result.get("confidence"),
                    "corruption_ratio": result.get("corruption_ratio"),
                    "source": source,
                    "input_sha256": result.get("metadata", {}).get("input_sha256"),
                    "output_sha256": result.get("metadata", {}).get("output_sha256"),
                }
            )
            updated["recovery_history"] = recovery_history
            return json_safe(updated)
        except ReaderError:
            raise
        except Exception as exc:
            source = "unknown"
            if isinstance(parsed_doc, Mapping):
                source = str(parsed_doc.get("source", "unknown"))
            raise RecoveryFailureError(source, f"Failed applying semantic recovery to parsed document: {exc}", cause=exc) from exc

    def can_recover(self, raw_text: str) -> bool:
        """Return whether semantic recovery is likely worth attempting."""

        if raw_text is None or not str(raw_text):
            return False
        report = text_quality_report(str(raw_text))
        return bool(
            report.quality_score < 0.75
            or report.control_char_ratio > 0.0
            or report.replacement_char_count > 0
            or report.null_byte_count > 0
        )

    def summarize(self, raw_text: str) -> Dict[str, Any]:
        """Return a cheap pre-recovery profile without emitting recovered text."""

        text = str(raw_text or "")
        report = text_quality_report(text)
        return {
            "can_recover": self.can_recover(text),
            "char_count": report.char_count,
            "line_count": report.line_count,
            "quality_score": report.quality_score,
            "control_char_ratio": report.control_char_ratio,
            "replacement_char_count": report.replacement_char_count,
            "null_byte_count": report.null_byte_count,
            "warnings": list(report.warnings),
            "input_sha256": sha256_text(text),
            "settings": self._settings_snapshot(),
        }


if __name__ == "__main__":
    print("\n=== Running Semantic Recovery ===\n")
    printer.status("TEST", "Semantic Recovery initialized", "info")

    recovery = SemanticRecovery(chunk_size=1024, max_output_chars=32768)

    clean = "Architecture creates memory through rhythm, threshold, proportion, and light."
    clean_summary = recovery.summarize(clean)
    assert clean_summary["char_count"] == len(clean)
    clean_result = recovery.recover(clean)
    assert clean_result["status"] in {"ok", "low_confidence"}
    assert clean_result["chunk_count"] >= 1
    assert "recovered_text" in clean_result
    assert "content" in clean_result

    corrupted = "\x00Plan\uFFFD section A\nwall-grid::12 broken\x05 threshold light datum"
    corrupted_result = recovery.recover(corrupted)
    assert corrupted_result["status"] in {"ok", "low_confidence"}
    assert corrupted_result["corruption_ratio"] > 0
    assert corrupted_result["metadata"]["profile"]["corruption_hits"] >= 2
    assert "Plan" in corrupted_result["recovered_text"] or corrupted_result["recovered_text"] == recovery.low_confidence_placeholder

    empty_result = recovery.recover("")
    assert empty_result["status"] == "failed"
    assert empty_result["recovered_text"] == recovery.corrupted_placeholder

    parsed = {
        "source": "sample.txt",
        "extension": ".txt",
        "content": corrupted,
        "metadata": {"size": len(corrupted)},
    }
    updated = recovery.recover_document(parsed)
    assert updated["source"] == "sample.txt"
    assert "semantic_recovery" in updated
    assert "recovery_history" in updated

    assert recovery.can_recover(corrupted) is True
    assert isinstance(recovery.can_recover(clean), bool)

    printer.pretty("SEMANTIC_RECOVERY_RESULT", corrupted_result, "success")
    print("\n=== Test ran successfully ===\n")
