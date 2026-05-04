"""
Input sanitization and validation utilities for the base subsystem.

This module provides the production-grade input sanitization layer used across
base-adjacent modules before values are persisted, logged, routed into models,
or passed into tool-like execution paths. It focuses on configurable text
normalization, HTML escaping, suspicious-pattern detection, recursive container
sanitization, bounded history, and lightweight validation helpers.

The implementation is intentionally helper-first. Shared coercion,
serialization, normalization, redaction, identifier generation, and validation
primitives are delegated to the base helper and base error layers rather than
being duplicated locally.

Key design goals:
- deterministic, config-driven sanitization behavior
- safe recursive handling of mappings and sequences
- strong validation and structured base-error integration
- optional strict/paranoid mode for higher-risk contexts
- bounded operational history and lightweight introspection
- backward-compatible utility-style entry points
"""

from __future__ import annotations

import html
import re
import unicodedata

from dataclasses import dataclass, field
from threading import RLock
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any, Deque, Dict, List, Optional, Pattern, Tuple, Union

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.base_errors import *
from ..utils.base_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Input Sanitizer")
printer = PrettyPrinter()


@dataclass(frozen=True)
class SanitizationRecord:
    """Bounded audit record for one sanitization or validation action."""

    timestamp: str
    operation: str
    input_type: str
    changed: bool
    suspicious: bool
    paranoid_mode: bool
    input_preview: str
    output_preview: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "input_type": self.input_type,
            "changed": self.changed,
            "suspicious": self.suspicious,
            "paranoid_mode": self.paranoid_mode,
            "input_preview": self.input_preview,
            "output_preview": self.output_preview,
            "metadata": to_json_safe(self.metadata),
        }


@dataclass(frozen=True)
class InputSanitizerStats:
    """Summary counters for sanitizer activity."""

    total_operations: int
    sanitize_operations: int
    validate_operations: int
    suspicious_hits: int
    history_length: int
    paranoid_mode: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_operations": self.total_operations,
            "sanitize_operations": self.sanitize_operations,
            "validate_operations": self.validate_operations,
            "suspicious_hits": self.suspicious_hits,
            "history_length": self.history_length,
            "paranoid_mode": self.paranoid_mode,
        }


class InputSanitizer:
    """Configuration-driven sanitizer and validator for text and containers."""

    _default_instance: Optional["InputSanitizer"] = None
    _paranoid_mode: bool = False
    _VALID_UNICODE_FORMS: Tuple[str, ...] = ("NFC", "NFD", "NFKC", "NFKD")

    def __init__(self) -> None:
        self.config = load_global_config()
        self.is_config = get_config_section("input_sanitizer") or {}
        self._lock = RLock()

        self.unicode_normalization_form = ensure_one_of(
            self.is_config.get("unicode_normalization_form", "NFKC"),
            self._VALID_UNICODE_FORMS,
            "unicode_normalization_form",
            config=self.is_config,
            error_cls=BaseConfigurationError,
        )
        self.strip_control_characters = coerce_bool(
            self.is_config.get("strip_control_characters", True),
            default=True,
        )
        self.strip_null_bytes = coerce_bool(
            self.is_config.get("strip_null_bytes", True),
            default=True,
        )
        self.squeeze_whitespace = coerce_bool(
            self.is_config.get("squeeze_whitespace", True),
            default=True,
        )
        self.strip_edges = coerce_bool(
            self.is_config.get("strip_edges", True),
            default=True,
        )
        self.escape_html = coerce_bool(
            self.is_config.get("escape_html", True),
            default=True,
        )
        self.lowercase = coerce_bool(
            self.is_config.get("lowercase", False),
            default=False,
        )
        self.allow_newlines = coerce_bool(
            self.is_config.get("allow_newlines", False),
            default=False,
        )
        self.preserve_tabs = coerce_bool(
            self.is_config.get("preserve_tabs", False),
            default=False,
        )
        self.max_text_length = coerce_int(
            self.is_config.get("max_text_length", 4096),
            default=4096,
            minimum=1,
        )
        self.filename_max_length = coerce_int(
            self.is_config.get("filename_max_length", 128),
            default=128,
            minimum=8,
        )
        self.max_container_items = coerce_int(
            self.is_config.get("max_container_items", 1000),
            default=1000,
            minimum=1,
        )
        self.max_recursion_depth = coerce_int(
            self.is_config.get("max_recursion_depth", 8),
            default=8,
            minimum=1,
        )
        self.record_history = coerce_bool(
            self.is_config.get("record_history", True),
            default=True,
        )
        self.history_limit = coerce_int(
            self.is_config.get("history_limit", 200),
            default=200,
            minimum=1,
        )
        self.reject_on_suspicious_patterns = coerce_bool(
            self.is_config.get("reject_on_suspicious_patterns", False),
            default=False,
        )
        self.default_allow_null = coerce_bool(
            self.is_config.get("default_allow_null", False),
            default=False,
        )
        self.sanitize_keys = coerce_bool(
            self.is_config.get("sanitize_keys", True),
            default=True,
        )
        self.default_paranoid_mode = coerce_bool(
            self.is_config.get("default_paranoid_mode", False),
            default=False,
        )
        self.paranoid_escape_quotes = coerce_bool(
            self.is_config.get("paranoid_escape_quotes", True),
            default=True,
        )

        replacement_map_raw = self.is_config.get("replacement_map", {}) or {}
        ensure_mapping(
            replacement_map_raw,
            "replacement_map",
            config=self.is_config,
            error_cls=BaseConfigurationError,
        )
        self.replacement_map = {ensure_text(k): ensure_text(v) for k, v in replacement_map_raw.items()}

        allowed_pattern_raw = self.is_config.get("allowed_text_pattern")
        self.allowed_text_pattern: Optional[Pattern[str]] = None
        if allowed_pattern_raw not in (None, "", "none", "None"):
            self.allowed_text_pattern = re.compile(ensure_non_empty_string(
                allowed_pattern_raw,
                "allowed_text_pattern",
                config=self.is_config,
                error_cls=BaseConfigurationError,
            ))

        suspicious_patterns_raw = self.is_config.get("suspicious_patterns", [])
        self.suspicious_patterns: List[Pattern[str]] = [
            re.compile(pattern)
            for pattern in parse_delimited_text(suspicious_patterns_raw)
        ]

        self._history: Deque[SanitizationRecord] = deque(maxlen=self.history_limit)
        self._stats = defaultdict(int)

        logger.info("Input Sanitizer successfully initialized")

    # ------------------------------------------------------------------
    # Singleton / mode control
    # ------------------------------------------------------------------
    @classmethod
    def get_default(cls) -> "InputSanitizer":
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    @classmethod
    def enable_paranoid_mode(cls) -> None:
        cls._paranoid_mode = True
        logger.warning("InputSanitizer is now running in paranoid mode.")

    @classmethod
    def disable_paranoid_mode(cls) -> None:
        cls._paranoid_mode = False
        logger.info("InputSanitizer paranoid mode disabled.")

    @classmethod
    def paranoid_mode_enabled(cls) -> bool:
        return bool(cls._paranoid_mode)

    def _paranoid_enabled(self, paranoid: Optional[bool] = None) -> bool:
        if paranoid is not None:
            return bool(paranoid)
        return bool(self.default_paranoid_mode or self.__class__._paranoid_mode)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_operation(
        self,
        *,
        operation: str,
        input_value: Any,
        output_value: Any,
        changed: bool,
        suspicious: bool,
        paranoid_mode: bool,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._stats["total_operations"] += 1
        if operation.startswith("sanitize"):
            self._stats["sanitize_operations"] += 1
        if operation.startswith("validate"):
            self._stats["validate_operations"] += 1
        if suspicious:
            self._stats["suspicious_hits"] += 1

        if not self.record_history:
            return

        record = SanitizationRecord(
            timestamp=utc_now_iso(),
            operation=operation,
            input_type=type(input_value).__name__,
            changed=changed,
            suspicious=suspicious,
            paranoid_mode=paranoid_mode,
            input_preview=truncate_string(safe_repr(input_value), 220),
            output_preview=truncate_string(safe_repr(output_value), 220),
            metadata=dict(metadata or {}),
        )
        self._history.append(record)

    def _check_container_bounds(self, value: Union[Mapping[str, Any], Sequence[Any]], *, name: str) -> None:
        size = len(value)
        if size > self.max_container_items:
            raise BaseValidationError(
                f"'{name}' exceeds the configured container item limit.",
                self.is_config,
                component="InputSanitizer",
                operation="container_bounds",
                context={
                    "name": name,
                    "size": size,
                    "max_container_items": self.max_container_items,
                },
            )

    def _sanitize_text_impl(
        self,
        text: Any,
        *,
        paranoid: Optional[bool] = None,
        allow_null: Optional[bool] = None,
        max_length: Optional[int] = None,
    ) -> Optional[str]:
        local_allow_null = self.default_allow_null if allow_null is None else bool(allow_null)
        paranoid_mode = self._paranoid_enabled(paranoid)

        if text is None:
            if local_allow_null:
                self._record_operation(
                    operation="sanitize_text",
                    input_value=text,
                    output_value=None,
                    changed=False,
                    suspicious=False,
                    paranoid_mode=paranoid_mode,
                    metadata={"allow_null": True},
                )
                return None
            raise BaseValidationError(
                "'text' must not be None.",
                self.is_config,
                component="InputSanitizer",
                operation="sanitize_text",
            )

        if isinstance(text, bytes):
            text = ensure_text(text)
        else:
            ensure_type(
                text,
                str,
                "text",
                config=self.is_config,
                error_cls=BaseValidationError,
                component="InputSanitizer",
                operation="sanitize_text",
            )

        original = text
        output = text

        if self.strip_null_bytes:
            output = output.replace("\x00", "")

        if self.unicode_normalization_form:
            output = unicodedata.normalize(self.unicode_normalization_form, output)

        if self.strip_control_characters:
            if self.allow_newlines and self.preserve_tabs:
                output = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", output)
            elif self.allow_newlines:
                output = re.sub(r"[\x00-\x08\x09\x0B\x0C\x0E-\x1F\x7F]", "", output)
            else:
                output = re.sub(r"[\x00-\x1F\x7F]", "", output)

        for source, replacement in self.replacement_map.items():
            output = output.replace(source, replacement)

        suspicious_matches = self._find_suspicious_matches(output)
        suspicious = bool(suspicious_matches)

        if self.reject_on_suspicious_patterns and suspicious:
            raise BaseValidationError(
                "Input text matched one or more suspicious patterns.",
                self.is_config,
                component="InputSanitizer",
                operation="sanitize_text",
                context={"matches": suspicious_matches},
            )

        for pattern in self.suspicious_patterns:
            output = pattern.sub("", output)

        if self.allow_newlines:
            if self.squeeze_whitespace:
                output = re.sub(r"[ \t\f\v]+", " ", output)
                output = re.sub(r"\n{3,}", "\n\n", output)
        else:
            output = normalize_text(
                output,
                lowercase=self.lowercase,
                strip=self.strip_edges,
                collapse_whitespace=self.squeeze_whitespace,
                unicode_form=None,
            )

        if self.allow_newlines and self.strip_edges:
            output = output.strip()
        if self.allow_newlines and self.lowercase:
            output = output.lower()
        if not self.preserve_tabs:
            output = output.replace("\t", " ")

        if self.escape_html:
            output = html.escape(output, quote=paranoid_mode or self.paranoid_escape_quotes)

        if self.allowed_text_pattern is not None and output:
            filtered_chars: List[str] = []
            for char in output:
                if char.isspace() or self.allowed_text_pattern.fullmatch(char):
                    filtered_chars.append(char)
            output = "".join(filtered_chars)
            if not self.allow_newlines and self.squeeze_whitespace:
                output = normalize_text(output, lowercase=False, strip=self.strip_edges, collapse_whitespace=True, unicode_form=None)

        effective_max_length = coerce_int(max_length, self.max_text_length, minimum=1) if max_length is not None else self.max_text_length
        if effective_max_length > 0:
            output = truncate_string(output, max_length=effective_max_length)

        changed = output != original
        self._record_operation(
            operation="sanitize_text",
            input_value=original,
            output_value=output,
            changed=changed,
            suspicious=suspicious,
            paranoid_mode=paranoid_mode,
            metadata={
                "max_length": effective_max_length,
                "suspicious_matches": suspicious_matches,
                "fingerprint": stable_fingerprint(output or "", length=16),
            },
        )
        return output

    def _sanitize_mapping_impl(
        self,
        data: Mapping[str, Any],
        *,
        paranoid: Optional[bool] = None,
        depth: int = 0,
    ) -> Dict[str, Any]:
        ensure_mapping(
            data,
            "data",
            config=self.is_config,
            error_cls=BaseValidationError,
            component="InputSanitizer",
            operation="sanitize_mapping",
        )
        if depth > self.max_recursion_depth:
            raise BaseValidationError(
                "Maximum sanitization recursion depth exceeded.",
                self.is_config,
                component="InputSanitizer",
                operation="sanitize_mapping",
                context={"depth": depth, "max_recursion_depth": self.max_recursion_depth},
            )

        self._check_container_bounds(data, name="data")
        sanitized: Dict[str, Any] = {}
        changed = False
        suspicious = False

        for raw_key, raw_value in data.items():
            key_text = ensure_text(raw_key)
            sanitized_key = self._sanitize_text_impl(key_text, paranoid=paranoid, allow_null=False)
            if self.sanitize_keys:
                sanitized_key = normalize_identifier(sanitized_key or key_text)
            if sanitized_key != key_text:
                changed = True

            sanitized_value = self._sanitize_value_impl(raw_value, paranoid=paranoid, depth=depth + 1)
            changed = changed or (sanitized_value != raw_value)
            if isinstance(raw_value, str) and self.contains_suspicious_content(raw_value):
                suspicious = True
            sanitized[sanitized_key] = sanitized_value

        self._record_operation(
            operation="sanitize_mapping",
            input_value=data,
            output_value=sanitized,
            changed=changed,
            suspicious=suspicious,
            paranoid_mode=self._paranoid_enabled(paranoid),
            metadata={"depth": depth, "size": len(data)},
        )
        return sanitized

    def _sanitize_sequence_impl(
        self,
        data: Sequence[Any],
        *,
        paranoid: Optional[bool] = None,
        depth: int = 0,
    ) -> Sequence[Any]:
        ensure_sequence(
            data,
            "data",
            config=self.is_config,
            error_cls=BaseValidationError,
            allow_str=False,
            component="InputSanitizer",
            operation="sanitize_sequence",
        )
        if depth > self.max_recursion_depth:
            raise BaseValidationError(
                "Maximum sanitization recursion depth exceeded.",
                self.is_config,
                component="InputSanitizer",
                operation="sanitize_sequence",
                context={"depth": depth, "max_recursion_depth": self.max_recursion_depth},
            )

        self._check_container_bounds(data, name="data")
        sanitized_items = [
            self._sanitize_value_impl(item, paranoid=paranoid, depth=depth + 1)
            for item in data
        ]
        changed = list(data) != sanitized_items
        suspicious = any(isinstance(item, str) and self.contains_suspicious_content(item) for item in data)

        if isinstance(data, tuple):
            output: Sequence[Any] = tuple(sanitized_items)
        elif isinstance(data, set):
            output = set(sanitized_items)
        else:
            output = list(sanitized_items)

        self._record_operation(
            operation="sanitize_sequence",
            input_value=data,
            output_value=output,
            changed=changed,
            suspicious=suspicious,
            paranoid_mode=self._paranoid_enabled(paranoid),
            metadata={"depth": depth, "size": len(data)},
        )
        return output

    def _sanitize_value_impl(
        self,
        value: Any,
        *,
        paranoid: Optional[bool] = None,
        depth: int = 0,
    ) -> Any:
        if depth > self.max_recursion_depth:
            raise BaseValidationError(
                "Maximum sanitization recursion depth exceeded.",
                self.is_config,
                component="InputSanitizer",
                operation="sanitize_value",
                context={"depth": depth, "max_recursion_depth": self.max_recursion_depth},
            )

        if value is None:
            return None
        if isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, (str, bytes)):
            return self._sanitize_text_impl(value, paranoid=paranoid, allow_null=True)
        if isinstance(value, Mapping):
            return self._sanitize_mapping_impl(value, paranoid=paranoid, depth=depth)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return self._sanitize_sequence_impl(value, paranoid=paranoid, depth=depth)
        return value

    def _validate_text_impl(self, value: Any, *, allow_null: Optional[bool] = None) -> bool:
        local_allow_null = self.default_allow_null if allow_null is None else bool(allow_null)
        if value is None:
            result = local_allow_null
            self._record_operation(
                operation="validate_text",
                input_value=value,
                output_value=result,
                changed=False,
                suspicious=False,
                paranoid_mode=self._paranoid_enabled(None),
                metadata={"allow_null": local_allow_null},
            )
            return result

        if isinstance(value, bytes):
            value = ensure_text(value)
        if not isinstance(value, str):
            self._record_operation(
                operation="validate_text",
                input_value=value,
                output_value=False,
                changed=False,
                suspicious=False,
                paranoid_mode=self._paranoid_enabled(None),
                metadata={"reason": "type_mismatch"},
            )
            return False

        normalized = value
        if self.unicode_normalization_form:
            normalized = unicodedata.normalize(self.unicode_normalization_form, normalized)
        suspicious_matches = self._find_suspicious_matches(normalized)
        suspicious = bool(suspicious_matches)

        valid = True
        if len(normalized) > self.max_text_length:
            valid = False
        if self.allowed_text_pattern is not None:
            valid = valid and all(char.isspace() or bool(self.allowed_text_pattern.fullmatch(char)) for char in normalized)
        if self.reject_on_suspicious_patterns and suspicious:
            valid = False

        self._record_operation(
            operation="validate_text",
            input_value=value,
            output_value=valid,
            changed=False,
            suspicious=suspicious,
            paranoid_mode=self._paranoid_enabled(None),
            metadata={"matches": suspicious_matches},
        )
        return valid

    def _validate_value_impl(self, value: Any, *, allow_null: Optional[bool] = None, depth: int = 0) -> bool:
        if depth > self.max_recursion_depth:
            return False
        local_allow_null = self.default_allow_null if allow_null is None else bool(allow_null)
        if value is None:
            return local_allow_null
        if isinstance(value, (bool, int, float)):
            return True
        if isinstance(value, (str, bytes)):
            return self._validate_text_impl(value, allow_null=local_allow_null)
        if isinstance(value, Mapping):
            if len(value) > self.max_container_items:
                return False
            return all(self._validate_value_impl(item, allow_null=local_allow_null, depth=depth + 1) for item in value.values())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if len(value) > self.max_container_items:
                return False
            return all(self._validate_value_impl(item, allow_null=local_allow_null, depth=depth + 1) for item in value)
        return True

    def _find_suspicious_matches(self, value: Any) -> List[str]:
        if value is None:
            return []
        text = ensure_text(value)
        matches: List[str] = []
        for pattern in self.suspicious_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)
        return matches

    # ------------------------------------------------------------------
    # Public singleton-backed API
    # ------------------------------------------------------------------
    @classmethod
    def sanitize_text(
        cls,
        text: Any,
        *,
        paranoid: Optional[bool] = None,
        allow_null: Optional[bool] = None,
        max_length: Optional[int] = None,
    ) -> Optional[str]:
        return cls.get_default()._sanitize_text_impl(
            text,
            paranoid=paranoid,
            allow_null=allow_null,
            max_length=max_length,
        )

    @classmethod
    def sanitize_dict(cls, data: Mapping[str, Any], *, paranoid: Optional[bool] = None) -> Dict[str, Any]:
        return cls.get_default()._sanitize_mapping_impl(data, paranoid=paranoid)

    @classmethod
    def sanitize_sequence(cls, data: Sequence[Any], *, paranoid: Optional[bool] = None) -> Sequence[Any]:
        return cls.get_default()._sanitize_sequence_impl(data, paranoid=paranoid)

    @classmethod
    def sanitize_value(cls, value: Any, *, paranoid: Optional[bool] = None) -> Any:
        return cls.get_default()._sanitize_value_impl(value, paranoid=paranoid)

    @classmethod
    def sanitize_identifier(cls, value: Any, *, default: str = "item") -> str:
        text = cls.sanitize_text(value, allow_null=False) or default
        return normalize_identifier(text or default, lowercase=True, separator="_", max_length=120)

    @classmethod
    def sanitize_filename(cls, filename: Any, *, default_stem: str = "file") -> str:
        text = cls.sanitize_text(filename, allow_null=False) or default_stem
        text = text.replace("/", " ").replace("\\", " ")
        pieces = text.rsplit(".", 1)
        if len(pieces) == 2 and pieces[1]:
            stem = slugify(pieces[0], default=default_stem, max_length=cls.get_default().filename_max_length)
            ext = slugify(pieces[1], default="txt", max_length=12)
            return f"{stem}.{ext}"
        return slugify(text, default=default_stem, max_length=cls.get_default().filename_max_length)

    @classmethod
    def contains_suspicious_content(cls, value: Any) -> List[str]:
        return cls.get_default()._find_suspicious_matches(value)

    @classmethod
    def validate_input(cls, value: Any, allow_null: bool = False) -> bool:
        return cls.get_default()._validate_value_impl(value, allow_null=allow_null)

    @classmethod
    def assert_valid_input(cls, value: Any, *, allow_null: bool = False, name: str = "value") -> None:
        is_valid = cls.validate_input(value, allow_null=allow_null)
        if not is_valid:
            raise BaseValidationError(
                f"'{name}' failed input validation.",
                cls.get_default().is_config,
                component="InputSanitizer",
                operation="assert_valid_input",
                context={
                    "name": name,
                    "value_preview": truncate_string(safe_repr(value), 220),
                    "allow_null": allow_null,
                },
            )

    @classmethod
    def recent_history(cls, limit: int = 20) -> List[Dict[str, Any]]:
        instance = cls.get_default()
        count = coerce_int(limit, default=20, minimum=1)
        return [record.to_dict() for record in list(instance._history)[-count:]]

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        instance = cls.get_default()
        summary = InputSanitizerStats(
            total_operations=int(instance._stats.get("total_operations", 0)),
            sanitize_operations=int(instance._stats.get("sanitize_operations", 0)),
            validate_operations=int(instance._stats.get("validate_operations", 0)),
            suspicious_hits=int(instance._stats.get("suspicious_hits", 0)),
            history_length=len(instance._history),
            paranoid_mode=instance._paranoid_enabled(None),
        )
        return summary.to_dict()

    @classmethod
    def export_history_json(cls, *, pretty: bool = True) -> str:
        return json_dumps(cls.recent_history(cls.get_default().history_limit), pretty=pretty)


if __name__ == "__main__":
    print("\n=== Running Input Sanitizer ===\n")
    printer.status("TEST", "Input Sanitizer initialized", "info")

    sanitizer = InputSanitizer.get_default()

    raw_text = "  <script>alert('x')</script>  Hello\tWorld\n\n🙂  "
    raw_mapping = {
        "User Name": "  Alice <b>Admin</b>  ",
        "notes": [
            "Hello\nWorld",
            "javascript:alert(1)",
            {"nested key": "<img src=x onerror=alert(1)>"},
        ],
        "count": 3,
    }

    sanitized_text = sanitizer.sanitize_text(raw_text)
    sanitized_mapping = sanitizer.sanitize_dict(raw_mapping)
    sanitized_identifier = sanitizer.sanitize_identifier("  Project: Alpha/Beta  ")
    sanitized_filename = sanitizer.sanitize_filename("../../Quarterly Report (Final).pdf")
    validation_ok = sanitizer.validate_input(raw_mapping)
    suspicious_matches = sanitizer.contains_suspicious_content(raw_text)

    printer.pretty("SANITIZED_TEXT", {"value": sanitized_text}, "success")
    printer.pretty("SANITIZED_MAPPING", sanitized_mapping, "success")
    printer.pretty("SANITIZED_IDENTIFIER", {"value": sanitized_identifier}, "success")
    printer.pretty("SANITIZED_FILENAME", {"value": sanitized_filename}, "success")
    printer.pretty("VALIDATION", {"valid": validation_ok}, "success")
    printer.pretty("SUSPICIOUS_MATCHES", suspicious_matches, "success")
    printer.pretty("STATS", sanitizer.stats(), "success")
    printer.pretty("RECENT_HISTORY", sanitizer.recent_history(), "success")

    print("\n=== Test ran successfully ===\n")
