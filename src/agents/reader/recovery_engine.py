from __future__ import annotations

"""Production-grade recovery orchestration for the Reader subsystem.

``RecoveryEngine`` coordinates deterministic, non-fabricating document recovery
for parsed Reader documents. It intentionally does not own generic helper
behavior: path handling, hashing, text quality scoring, JSON safety, low-level
text cleanup, LRU caching, and parsed-document validation live in
``reader_helpers.py``. It also does not own semantic salvage logic; expensive
semantic/token recovery remains delegated to ``SemanticRecovery``.

Pipeline
--------
1. Validate the parsed Reader document shape.
2. Profile the original content using shared helper quality metrics.
3. Apply conservative low-level cleanup for detectable corruption artifacts.
4. Decide whether semantic recovery is warranted using configured thresholds.
5. Accept, reject, or fallback based on quality/confidence without fabricating.
6. Return a backward-compatible recovery result for ``ReaderAgent`` and newer
   metadata-rich payloads for observability, checkpoints, and audits.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .utils.config_loader import get_config_section, load_reader_config
from .utils.reader_error import *
from .utils.reader_helpers import *
from .modules.semantic_recovery import SemanticRecovery
from .reader_memory import ReaderMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Recovery Engine")
printer = PrettyPrinter()


@dataclass(frozen=True)
class RecoveryDecision:
    """Decision made after low-level cleanup and before optional semantic pass."""

    use_semantic: bool
    reason: str
    quality_before: float
    quality_after_low_level: float
    content_changed_by_low_level: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


@dataclass(frozen=True)
class RecoveryEngineProfile:
    """Operational profile for one document recovery pass."""

    source: str
    original_chars: int
    recovered_chars: int
    original_sha256: str
    recovered_sha256: str
    quality_before: float
    quality_after_low_level: float
    quality_after: float
    low_level_changed_content: bool
    used_semantic: bool
    semantic_status: Optional[str]
    semantic_confidence: Optional[float]
    cache_key: str
    warnings: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


class RecoveryEngine:
    """Multi-pass recovery engine for parsed Reader documents.

    Public compatibility is preserved:
    - ``recover_document(parsed_doc)`` still returns ``status``, ``strategy``,
      ``content``, ``confidence``, ``quality_before``, ``quality_after``,
      ``cached``, and ``metadata``.
    - Constructor arguments remain accepted for existing ``ReaderAgent`` usage,
      but config values from ``reader_config.yaml`` are the default source of
      behavior.
    """

    def __init__(
        self,
        min_quality_score: Optional[float] = None,
        semantic_threshold: Optional[float] = None,
        cache_size: Optional[int] = None,
        memory: Optional[ReaderMemory] = None,
    ) -> None:
        self.config = load_reader_config()
        self.recovery_config = get_config_section("recovery_engine") or {}

        self.min_quality_score = self._cfg_float(
            "min_quality_score",
            0.55 if min_quality_score is None else min_quality_score,
            minimum=0.0,
            maximum=1.0,
        )
        self.semantic_threshold = self._cfg_float(
            "semantic_threshold",
            0.40 if semantic_threshold is None else semantic_threshold,
            minimum=0.0,
            maximum=1.0,
        )
        self.cache_size = self._cfg_int("cache_size", 256 if cache_size is None else cache_size, minimum=16)

        self.enable_cache = self._cfg_bool("enable_cache", True)
        self.enable_low_level_repair = self._cfg_bool("enable_low_level_repair", True)
        self.enable_semantic_recovery = self._cfg_bool("enable_semantic_recovery", True)
        self.semantic_if_empty = self._cfg_bool("semantic_if_empty", True)
        self.semantic_if_low_quality = self._cfg_bool("semantic_if_low_quality", True)
        self.semantic_if_corruption_detected = self._cfg_bool("semantic_if_corruption_detected", True)
        self.accept_semantic_when_quality_improves = self._cfg_bool("accept_semantic_when_quality_improves", True)
        self.fallback_to_low_level_on_semantic_failure = self._cfg_bool("fallback_to_low_level_on_semantic_failure", True)
        self.preserve_original_on_noop = self._cfg_bool("preserve_original_on_noop", True)
        self.fail_on_unrecoverable = self._cfg_bool("fail_on_unrecoverable", False)
        self.mark_unrecoverable_placeholder = self._cfg_bool("mark_unrecoverable_placeholder", True)

        self.min_semantic_confidence = self._cfg_float("min_semantic_confidence", 0.10, minimum=0.0, maximum=1.0)
        self.max_input_chars = self._cfg_int("max_input_chars", 0, minimum=0)
        self.recovery_history_limit = self._cfg_int("recovery_history_limit", 10, minimum=1, maximum=250)
        self.warnings_limit = self._cfg_int("warnings_limit", 200, minimum=1, maximum=5_000)
        self.include_debug_preview = self._cfg_bool("include_debug_preview", False)
        self.debug_preview_chars = self._cfg_int("debug_preview_chars", 240, minimum=0, maximum=10_000)
        self.corrupted_placeholder = self._cfg_str("corrupted_placeholder", "[CORRUPTED_DATA]")

        self.semantic = SemanticRecovery()
        self.memory = memory
        self._recovery_cache: BoundedLRUCache[Dict[str, Any]] = BoundedLRUCache(max_size=self.cache_size) # Keep the in‑memory LRU as L1

        self._validate_config()
        logger.info(
            "Recovery Engine initialized | min_quality=%.3f semantic_threshold=%.3f cache_size=%s",
            self.min_quality_score,
            self.semantic_threshold,
            self.cache_size,
        )

    # ------------------------------------------------------------------
    # Config access. Uses the existing config loader as-is.
    # ------------------------------------------------------------------

    def _cfg_value(self, key: str, default: Any) -> Any:
        return self.recovery_config.get(key, default)

    def _cfg_bool(self, key: str, default: bool) -> bool:
        return coerce_bool(self._cfg_value(key, default), default)

    def _cfg_int(
        self,
        key: str,
        default: int | float,
        *,
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
    ) -> int:
        try:
            return coerce_int(self._cfg_value(key, default), int(default), minimum=minimum, maximum=maximum)
        except Exception as exc:
            raise self._configuration_error(
                f"recovery_engine.{key} must be an integer",
                {"key": key, "value": self.recovery_config.get(key), "default": default},
                exc,
            ) from exc

    def _cfg_float(
        self,
        key: str,
        default: int | float,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        try:
            return coerce_float(self._cfg_value(key, default), float(default), minimum=minimum, maximum=maximum)
        except Exception as exc:
            raise self._configuration_error(
                f"recovery_engine.{key} must be a number",
                {"key": key, "value": self.recovery_config.get(key), "default": default},
                exc,
            ) from exc

    def _cfg_str(self, key: str, default: str) -> str:
        value = self._cfg_value(key, default)
        return str(default if value is None else value)

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
        if self.semantic_threshold > self.min_quality_score:
            logger.warning(
                "Recovery semantic_threshold %.3f is above min_quality_score %.3f; semantic fallback may run aggressively.",
                self.semantic_threshold,
                self.min_quality_score,
            )
        if not self.corrupted_placeholder.strip():
            raise self._configuration_error("recovery_engine.corrupted_placeholder cannot be empty")

    def _settings_snapshot(self) -> Dict[str, Any]:
        return {
            "min_quality_score": self.min_quality_score,
            "semantic_threshold": self.semantic_threshold,
            "cache_size": self.cache_size,
            "enable_cache": self.enable_cache,
            "enable_low_level_repair": self.enable_low_level_repair,
            "enable_semantic_recovery": self.enable_semantic_recovery,
            "semantic_if_empty": self.semantic_if_empty,
            "semantic_if_low_quality": self.semantic_if_low_quality,
            "semantic_if_corruption_detected": self.semantic_if_corruption_detected,
            "min_semantic_confidence": self.min_semantic_confidence,
            "accept_semantic_when_quality_improves": self.accept_semantic_when_quality_improves,
            "fallback_to_low_level_on_semantic_failure": self.fallback_to_low_level_on_semantic_failure,
            "preserve_original_on_noop": self.preserve_original_on_noop,
            "fail_on_unrecoverable": self.fail_on_unrecoverable,
            "mark_unrecoverable_placeholder": self.mark_unrecoverable_placeholder,
            "max_input_chars": self.max_input_chars,
            "recovery_history_limit": self.recovery_history_limit,
            "warnings_limit": self.warnings_limit,
            "include_debug_preview": self.include_debug_preview,
            "debug_preview_chars": self.debug_preview_chars,
        }

    # ------------------------------------------------------------------
    # Cache and profiling helpers.
    # ------------------------------------------------------------------

    def _cache_key(self, parsed_doc: Mapping[str, Any]) -> str:
        doc = validate_parsed_document(parsed_doc)
        return recovery_cache_key(
            str(doc.get("source", "unknown")),
            str(doc.get("content", "")),
            policy=self._settings_snapshot(),
        )

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        # 1. Check L1 (in‑memory)
        if not self.enable_cache:
            return None
        cached = self._recovery_cache.get_or_default(key)
        if cached is not None:
            output = dict(cached)
            output["cached"] = True
            output.setdefault("metadata", {})["cache_level"] = "memory"
            return output
    
        # 2. Check L2 (persistent memory)
        if self.memory is not None:
            # The persistent cache key must be stable – we already use recovery_cache_key
            # We'll store under the same key but with a dedicated namespace.
            persistent_key_payload = {"recovery_key": key}
            persistent_value = self.memory.get_cache(persistent_key_payload, namespace="recovery")
            if persistent_value is not None:
                # Load into L1 for future hits
                self._recovery_cache[key] = persistent_value
                output = dict(persistent_value)
                output["cached"] = True
                output.setdefault("metadata", {})["cache_level"] = "persistent"
                return output
    
        return None
    
    def _set_cache(self, key: str, value: Mapping[str, Any]) -> None:
        if not self.enable_cache:
            return
        # Store in L1
        self._recovery_cache[key] = json_safe(dict(value))
        # Store in L2 (persistent)
        if self.memory is not None:
            persistent_key_payload = {"recovery_key": key}
            # We can optionally set a TTL based on recovery_engine.cache_ttl (if we add config)
            self.memory.set_cache(persistent_key_payload, dict(value), namespace="recovery")

    def clear_cache(self) -> None:
        self._recovery_cache.clear()

    def cache_info(self) -> Dict[str, Any]:
        info = self._recovery_cache.to_dict()
        info["enabled"] = self.enable_cache
        return info

    @staticmethod
    def _quality_score(text: str) -> float:
        return text_quality_report(text).quality_score

    def _repair_low_level(self, content: str) -> str:
        if not self.enable_low_level_repair:
            return str(content)
        return low_level_repair_text(str(content))

    def _limited_warnings(self, warnings: Iterable[str]) -> List[str]:
        return dedupe_preserve_order(warnings)[: self.warnings_limit]

    def _debug_preview(self, text: str) -> Optional[str]:
        if not self.include_debug_preview or self.debug_preview_chars <= 0:
            return None
        return truncate_text(text, self.debug_preview_chars)

    # ------------------------------------------------------------------
    # Recovery decision and result shaping.
    # ------------------------------------------------------------------

    def _decide(self, original: str, low_level: str) -> RecoveryDecision:
        before = text_quality_report(original)
        after = text_quality_report(low_level)
        content_changed = original != low_level
        warnings: List[str] = [*before.warnings, *after.warnings]

        use_semantic = False
        reason = "quality_acceptable_after_low_level_repair"

        if not self.enable_semantic_recovery:
            reason = "semantic_recovery_disabled"
        elif self.semantic_if_empty and not low_level.strip():
            use_semantic = True
            reason = "empty_after_low_level_repair"
        elif self.semantic_if_low_quality and after.quality_score < self.min_quality_score:
            use_semantic = True
            reason = "quality_after_low_level_below_minimum"
        elif self.semantic_if_low_quality and before.quality_score < self.semantic_threshold:
            use_semantic = True
            reason = "quality_before_below_semantic_threshold"
        elif self.semantic_if_corruption_detected and any(
            warning in set(warnings)
            for warning in (
                "null_bytes_detected",
                "replacement_chars_detected",
                "high_control_char_ratio",
                "low_text_quality",
            )
        ):
            use_semantic = True
            reason = "corruption_warnings_detected"
        elif content_changed:
            reason = "low_level_repair_sufficient"

        return RecoveryDecision(
            use_semantic=use_semantic,
            reason=reason,
            quality_before=before.quality_score,
            quality_after_low_level=after.quality_score,
            content_changed_by_low_level=content_changed,
            warnings=tuple(self._limited_warnings(warnings)),
        )

    def _profile(
        self,
        *,
        source: str,
        original: str,
        recovered: str,
        decision: RecoveryDecision,
        used_semantic: bool,
        semantic_result: Optional[Mapping[str, Any]],
        cache_key: str,
        warnings: Iterable[str],
    ) -> RecoveryEngineProfile:
        semantic_status = None
        semantic_confidence = None
        if semantic_result:
            semantic_status = str(semantic_result.get("status", "unknown"))
            try:
                semantic_confidence = float(semantic_result.get("confidence", 0.0))
            except (TypeError, ValueError):
                semantic_confidence = 0.0
        return RecoveryEngineProfile(
            source=source,
            original_chars=len(original),
            recovered_chars=len(recovered),
            original_sha256=sha256_text(original),
            recovered_sha256=sha256_text(recovered),
            quality_before=decision.quality_before,
            quality_after_low_level=decision.quality_after_low_level,
            quality_after=self._quality_score(recovered),
            low_level_changed_content=decision.content_changed_by_low_level,
            used_semantic=used_semantic,
            semantic_status=semantic_status,
            semantic_confidence=semantic_confidence,
            cache_key=cache_key,
            warnings=tuple(self._limited_warnings(warnings)),
        )

    def _build_result(
        self,
        *,
        source: str,
        strategy: str,
        original: str,
        recovered: str,
        confidence: float,
        decision: RecoveryDecision,
        semantic_result: Optional[Mapping[str, Any]],
        cache_key: str,
        timer: Optional[OperationTimer],
        warnings: Iterable[str],
    ) -> Dict[str, Any]:
        used_semantic = strategy in {"semantic", "semantic_low_confidence", "unrecoverable"}
        profile = self._profile(
            source=source,
            original=original,
            recovered=recovered,
            decision=decision,
            used_semantic=used_semantic,
            semantic_result=semantic_result,
            cache_key=cache_key,
            warnings=warnings,
        )
        metadata: Dict[str, Any] = {
            "source": source,
            "used_semantic": used_semantic,
            "decision": decision.to_dict(),
            "profile": profile.to_dict(),
            "settings": self._settings_snapshot(),
            "semantic": json_safe(semantic_result or {}),
            "cache_hit": False,
        }
        preview = self._debug_preview(recovered)
        if preview is not None:
            metadata["debug_preview"] = preview
        if timer is not None:
            metadata["timing"] = timer.stop().to_dict()

        return {
            "status": "ok" if strategy != "unrecoverable" else "unrecoverable",
            "strategy": strategy,
            "content": recovered,
            "confidence": round(clamp(float(confidence), 0.0, 1.0), 3),
            "quality_before": round(decision.quality_before, 3),
            "quality_after": round(profile.quality_after, 3),
            "cached": False,
            "warnings": self._limited_warnings(warnings),
            "metadata": metadata,
        }

    def _unrecoverable_result(
        self,
        *,
        source: str,
        original: str,
        low_level: str,
        decision: RecoveryDecision,
        semantic_result: Optional[Mapping[str, Any]],
        cache_key: str,
        timer: Optional[OperationTimer],
        warnings: Iterable[str],
    ) -> Dict[str, Any]:
        if self.fail_on_unrecoverable:
            raise CorruptContentError(
                source,
                "Reader recovery could not produce acceptable content",
                confidence=float(semantic_result.get("confidence", 0.0)) if semantic_result else 0.0,
                corruption_ratio=float(semantic_result.get("corruption_ratio", 0.0)) if semantic_result else None,
            )
        recovered = self.corrupted_placeholder if self.mark_unrecoverable_placeholder else low_level
        return self._build_result(
            source=source,
            strategy="unrecoverable",
            original=original,
            recovered=recovered,
            confidence=0.0,
            decision=decision,
            semantic_result=semantic_result,
            cache_key=cache_key,
            timer=timer,
            warnings=[*warnings, "recovery_unrecoverable_content"],
        )

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def can_recover(self, parsed_doc: Mapping[str, Any]) -> bool:
        doc = validate_parsed_document(parsed_doc)
        content = str(doc.get("content", ""))
        low_level = self._repair_low_level(content)
        decision = self._decide(content, low_level)
        return bool(decision.use_semantic or decision.content_changed_by_low_level)

    def recover_document(self, parsed_doc: Mapping[str, Any]) -> Dict[str, Any]:
        """Recover a parsed Reader document's content.

        The method returns a recovery payload rather than mutating the document.
        Use ``recover_and_apply`` when a full parsed-document copy with updated
        content is needed.
        """

        source = "unknown"
        timer = OperationTimer(operation="recover_document")
        try:
            doc = validate_parsed_document(parsed_doc)
            source = str(doc.get("source", "unknown"))
            content = str(doc.get("content", ""))

            if self.max_input_chars > 0 and len(content) > self.max_input_chars:
                raise RecoveryFailureError(
                    source,
                    f"Reader recovery input exceeds max_input_chars={self.max_input_chars}",
                )

            key = self._cache_key(doc)
            cached = self._get_cached(key)
            if cached is not None:
                return cached

            low_level = self._repair_low_level(content)
            decision = self._decide(content, low_level)
            warnings: List[str] = list(decision.warnings)

            if not decision.use_semantic:
                recovered = low_level if decision.content_changed_by_low_level or not self.preserve_original_on_noop else content
                confidence = self._quality_score(recovered)
                strategy = "low_level" if decision.content_changed_by_low_level else "noop"
                result = self._build_result(
                    source=source,
                    strategy=strategy,
                    original=content,
                    recovered=recovered,
                    confidence=confidence,
                    decision=decision,
                    semantic_result=None,
                    cache_key=key,
                    timer=timer,
                    warnings=warnings,
                )
                self._set_cache(key, result)
                return result

            semantic_result = self.semantic.recover(low_level)
            semantic_text = str(semantic_result.get("recovered_text", semantic_result.get("content", "")))
            semantic_confidence = float(semantic_result.get("confidence", 0.0) or 0.0)
            semantic_quality = self._quality_score(semantic_text)
            low_level_quality = self._quality_score(low_level)
            semantic_status = str(semantic_result.get("status", "unknown"))
            warnings.extend(list(semantic_result.get("warnings", []) or []))

            semantic_is_acceptable = (
                semantic_text.strip()
                and semantic_text != self.corrupted_placeholder
                and semantic_confidence >= self.min_semantic_confidence
                and semantic_status not in {"failed", "low_confidence"}
            )
            if self.accept_semantic_when_quality_improves:
                semantic_is_acceptable = bool(
                    semantic_is_acceptable and semantic_quality >= min(self.min_quality_score, low_level_quality)
                )

            if semantic_is_acceptable:
                result = self._build_result(
                    source=source,
                    strategy="semantic",
                    original=content,
                    recovered=semantic_text,
                    confidence=semantic_confidence,
                    decision=decision,
                    semantic_result=semantic_result,
                    cache_key=key,
                    timer=timer,
                    warnings=warnings,
                )
                self._set_cache(key, result)
                return result

            fallback_allowed = self.fallback_to_low_level_on_semantic_failure and bool(low_level.strip())
            fallback_quality_ok = low_level_quality >= self.min_quality_score or low_level_quality > decision.quality_before
            if fallback_allowed and fallback_quality_ok:
                result = self._build_result(
                    source=source,
                    strategy="low_level_fallback",
                    original=content,
                    recovered=low_level,
                    confidence=low_level_quality,
                    decision=decision,
                    semantic_result=semantic_result,
                    cache_key=key,
                    timer=timer,
                    warnings=[*warnings, "semantic_recovery_rejected_low_level_fallback_used"],
                )
                self._set_cache(key, result)
                return result

            result = self._unrecoverable_result(
                source=source,
                original=content,
                low_level=low_level,
                decision=decision,
                semantic_result=semantic_result,
                cache_key=key,
                timer=timer,
                warnings=[*warnings, "semantic_recovery_rejected"],
            )
            self._set_cache(key, result)
            return result
        except ReaderError:
            raise
        except Exception as exc:
            raise RecoveryFailureError(source, f"Failed recovering document '{source}': {exc}", cause=exc) from exc

    def recover_and_apply(self, parsed_doc: Mapping[str, Any]) -> Dict[str, Any]:
        """Return a parsed-document copy with recovered content and history."""

        doc = validate_parsed_document(parsed_doc)
        result = self.recover_document(doc)
        updated = apply_recovered_content(doc, result)
        history = list(updated.get("recovery_history", []) or [])
        history.append(
            {
                "strategy": result.get("strategy"),
                "status": result.get("status"),
                "confidence": result.get("confidence"),
                "quality_before": result.get("quality_before"),
                "quality_after": result.get("quality_after"),
                "source": doc.get("source", "unknown"),
                "metadata": {
                    "used_semantic": result.get("metadata", {}).get("used_semantic"),
                    "profile": result.get("metadata", {}).get("profile"),
                },
            }
        )
        updated["recovery_history"] = history[-self.recovery_history_limit :]
        return json_safe(updated)

    def recover_documents(self, parsed_docs: Sequence[Mapping[str, Any]], *, apply: bool = False) -> List[Dict[str, Any]]:
        """Recover multiple parsed documents sequentially with per-item errors."""

        results: List[Dict[str, Any]] = []
        for index, doc in enumerate(parsed_docs):
            try:
                results.append(self.recover_and_apply(doc) if apply else self.recover_document(doc))
            except Exception as exc:
                error = normalize_reader_error(
                    exc,
                    message="Reader document recovery failed",
                    context={"index": index},
                    stage=ReaderErrorStage.RECOVERY,
                )
                results.append(build_error_result(error, operation="recover_document"))
        return results

    def summarize(self, parsed_doc: Mapping[str, Any]) -> Dict[str, Any]:
        doc = validate_parsed_document(parsed_doc)
        content = str(doc.get("content", ""))
        low_level = self._repair_low_level(content)
        decision = self._decide(content, low_level)
        summary = summarize_parsed_document(doc)
        summary.update(
            {
                "can_recover": self.can_recover(doc),
                "decision": decision.to_dict(),
                "settings": self._settings_snapshot(),
                "cache_key": self._cache_key(doc),
            }
        )
        return json_safe(summary)

    def capabilities(self) -> Dict[str, Any]:
        return {
            "engine": "RecoveryEngine",
            "strategies": ["noop", "low_level", "semantic", "low_level_fallback", "unrecoverable"],
            "semantic_recovery_enabled": self.enable_semantic_recovery,
            "low_level_repair_enabled": self.enable_low_level_repair,
            "cache": self.cache_info(),
            "settings": self._settings_snapshot(),
        }


if __name__ == "__main__":
    print("\n=== Running Recovery Engine ===\n")
    printer.status("TEST", "Recovery Engine initialized", "info")

    engine = RecoveryEngine(min_quality_score=0.55, semantic_threshold=0.40, cache_size=32)

    clean_doc = {
        "status": "ok",
        "source": "clean.txt",
        "extension": ".txt",
        "content": "Architecture uses rhythm, proportion, threshold, and light to organize memory.",
        "metadata": {"size": 74},
        "warnings": [],
    }
    clean_result = engine.recover_document(clean_doc)
    assert clean_result["status"] == "ok"
    assert clean_result["strategy"] in {"noop", "low_level"}
    assert clean_result["content"] == clean_doc["content"]
    assert clean_result["metadata"]["used_semantic"] is False

    corrupted_doc = {
        "status": "ok",
        "source": "corrupted.txt",
        "extension": ".txt",
        "content": "Architecture\x00\x00 creates\r\nmeaning through \uFFFDthresholds and light.",
        "metadata": {"size": 64},
        "warnings": ["binary_signature_detected"],
    }
    recovered = engine.recover_document(corrupted_doc)
    assert recovered["status"] in {"ok", "unrecoverable"}
    assert "\x00" not in recovered["content"]
    assert "\uFFFD" not in recovered["content"]
    assert recovered["quality_after"] >= 0.0

    cached = engine.recover_document(corrupted_doc)
    assert cached["cached"] is True

    applied = engine.recover_and_apply(corrupted_doc)
    assert applied["content"] == recovered["content"]
    assert applied["recovery"]["strategy"] == recovered["strategy"]
    assert applied["recovery_history"]

    batch = engine.recover_documents([clean_doc, corrupted_doc])
    assert len(batch) == 2
    assert all(item.get("status") in {"ok", "unrecoverable"} for item in batch)

    summary = engine.summarize(corrupted_doc)
    assert summary["source"] == "corrupted.txt"
    assert "decision" in summary

    capabilities = engine.capabilities()
    assert capabilities["engine"] == "RecoveryEngine"
    assert capabilities["cache"]["enabled"] is True

    print("\n=== Test ran successfully ===\n")
