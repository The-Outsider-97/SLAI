from __future__ import annotations

import copy
import time

from collections import Counter, deque
from threading import RLock
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.handler_error import *
from .utils.handler_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Handler Memory")
printer = PrettyPrinter()


class HandlerMemory:
    """
    Production checkpoint, telemetry, and postmortem memory for HandlerAgent decisions.

    Scope:
    - stores bounded in-memory checkpoints for rollback/recovery
    - stores bounded telemetry and postmortem streams for adaptation/evaluation
    - normalizes and redacts stored payloads through HandlerErrorPolicy + handler_helpers
    - provides lookup, filtering, pruning, export, and operational health APIs

    This class intentionally remains an in-process memory layer. It can optionally mirror
    bounded buffers to SharedMemory-like objects, but persistence/backing stores should be
    implemented outside this module by higher-level orchestration.
    """

    CHECKPOINT_KEY = "handler:memory:checkpoints"
    TELEMETRY_KEY = "handler:telemetry"
    POSTMORTEM_KEY = "handler:postmortems"

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        shared_memory: Any = None,
        error_policy: Optional[HandlerErrorPolicy] = None,
    ):
        self.config = load_global_config()
        self.memory_config = get_config_section("memory")

        self.max_checkpoints = coerce_int(self.memory_config.get("max_checkpoints"), 100, minimum=1, maximum=100_000)
        self.max_telemetry_events = coerce_int(self.memory_config.get("max_telemetry_events"), 1000, minimum=1, maximum=1_000_000)
        self.max_postmortems = coerce_int(self.memory_config.get("max_postmortems"), 1000, minimum=1, maximum=1_000_000)
        self.max_query_results = coerce_int(self.memory_config.get("max_query_results"), 250, minimum=1, maximum=100_000)
        self.max_checkpoint_state_chars = coerce_int(self.memory_config.get("max_checkpoint_state_chars"), 200_000, minimum=1024)
        self.max_checkpoint_metadata_chars = coerce_int(self.memory_config.get("max_checkpoint_metadata_chars"), 20_000, minimum=256)
        self.max_event_chars = coerce_int(self.memory_config.get("max_event_chars"), 80_000, minimum=512)
        self.default_checkpoint_ttl_seconds = self._optional_float(self.memory_config.get("default_checkpoint_ttl_seconds"), default=600.0)
        self.default_telemetry_ttl_seconds = self._optional_float(self.memory_config.get("default_telemetry_ttl_seconds"), default=None)
        self.default_postmortem_ttl_seconds = self._optional_float(self.memory_config.get("default_postmortem_ttl_seconds"), default=None)
        self.copy_on_read = coerce_bool(self.memory_config.get("copy_on_read"), default=True)
        self.sanitize_payloads = coerce_bool(self.memory_config.get("sanitize_payloads"), default=True)
        self.mirror_to_shared_memory = coerce_bool(self.memory_config.get("mirror_to_shared_memory"), default=False)
        self.mirror_checkpoints_to_shared_memory = coerce_bool(
            self.memory_config.get("mirror_checkpoints_to_shared_memory"),
            default=self.mirror_to_shared_memory,
        )
        self.mirror_telemetry_to_shared_memory = coerce_bool(
            self.memory_config.get("mirror_telemetry_to_shared_memory"),
            default=self.mirror_to_shared_memory,
        )
        self.mirror_postmortems_to_shared_memory = coerce_bool(
            self.memory_config.get("mirror_postmortems_to_shared_memory"),
            default=self.mirror_to_shared_memory,
        )
        self.shared_memory_checkpoint_ttl_seconds = self._optional_int(self.memory_config.get("shared_memory_checkpoint_ttl_seconds"), default=600)
        self.shared_memory_telemetry_ttl_seconds = self._optional_int(self.memory_config.get("shared_memory_telemetry_ttl_seconds"), default=None)
        self.shared_memory_postmortem_ttl_seconds = self._optional_int(self.memory_config.get("shared_memory_postmortem_ttl_seconds"), default=None)
        self.require_shared_memory_when_mirroring = coerce_bool(
            self.memory_config.get("require_shared_memory_when_mirroring"),
            default=True,
        )

        policy_cfg = self.memory_config.get("error_policy")
        self.error_policy = error_policy or HandlerErrorPolicy.from_mapping(policy_cfg if isinstance(policy_cfg, Mapping) else None)
        self.shared_memory = shared_memory

        self._lock = RLock()
        self._checkpoints: Deque[Dict[str, Any]] = deque(maxlen=self.max_checkpoints)
        self._telemetry: Deque[Dict[str, Any]] = deque(maxlen=self.max_telemetry_events)
        self._postmortems: Deque[Dict[str, Any]] = deque(maxlen=self.max_postmortems)

        self._checkpoint_index: Dict[str, Dict[str, Any]] = {}
        self._label_index: Dict[str, set[str]] = {}
        self._correlation_index: Dict[str, set[str]] = {}
        self._task_index: Dict[str, set[str]] = {}

        self._validate_mirroring_wiring()

        logger.info(
            "Handler memory successfully initialized | checkpoints=%s telemetry=%s postmortems=%s",
            self.max_checkpoints,
            self.max_telemetry_events,
            self.max_postmortems,
        )

    def _validate_mirroring_wiring(self) -> None:
        mirroring_enabled = any(
            (
                self.mirror_to_shared_memory,
                self.mirror_checkpoints_to_shared_memory,
                self.mirror_telemetry_to_shared_memory,
                self.mirror_postmortems_to_shared_memory,
            )
        )
        if not mirroring_enabled or self.shared_memory is not None:
            return

        message = (
            "HandlerMemory shared-memory mirroring is enabled but no shared_memory instance "
            "was provided. Pass shared_memory=SharedMemory() (or call attach_shared_memory) "
            "to activate mirroring."
        )
        if self.require_shared_memory_when_mirroring:
            raise ConfigurationError(
                message,
                code="HANDLER_MEMORY_SHARED_MEMORY_REQUIRED",
                context={
                    "mirror_to_shared_memory": self.mirror_to_shared_memory,
                    "mirror_checkpoints_to_shared_memory": self.mirror_checkpoints_to_shared_memory,
                    "mirror_telemetry_to_shared_memory": self.mirror_telemetry_to_shared_memory,
                    "mirror_postmortems_to_shared_memory": self.mirror_postmortems_to_shared_memory,
                },
                policy=self.error_policy,
            )
        logger.warning(message)

    def save_checkpoint(
        self,
        label: str,
        state: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        checkpoint_id: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Save a sanitized checkpoint and return its checkpoint id."""
        self._validate_mapping(state, source="checkpoint state")
        metadata_map = coerce_mapping(metadata)
        safe_label = normalize_identifier(label, default="checkpoint")
        now = utc_timestamp()
        ttl = self.default_checkpoint_ttl_seconds if ttl_seconds is None else self._optional_float(ttl_seconds, default=None)
        expires_at = (now + ttl) if ttl is not None and ttl > 0 else None
        resolved_correlation_id = correlation_id or metadata_map.get("correlation_id") or metadata_map.get("task_correlation_id")
        resolved_checkpoint_id = checkpoint_id or generate_checkpoint_id(safe_label)

        try:
            checkpoint = build_checkpoint_payload(
                label=safe_label,
                state=state,
                metadata=compact_dict(
                    {
                        **metadata_map,
                        "correlation_id": resolved_correlation_id,
                        "expires_at": expires_at,
                    },
                    drop_none=True,
                ),
                checkpoint_id=resolved_checkpoint_id,
                timestamp=now,
                policy=self.error_policy if self.sanitize_payloads else None,
            )
            self._enforce_payload_size(checkpoint.get("state", {}), self.max_checkpoint_state_chars, source="checkpoint state")
            self._enforce_payload_size(checkpoint.get("metadata", {}), self.max_checkpoint_metadata_chars, source="checkpoint metadata")
            checkpoint["expires_at"] = expires_at
            checkpoint["updated"] = now
            checkpoint["version"] = 1
            checkpoint["summary"] = self._checkpoint_summary(checkpoint)
            checkpoint["state_hash"] = stable_hash(checkpoint.get("state", {}), length=16)
        except HandlerError:
            raise
        except Exception as exc:
            raise SerializationError(
                "Unable to serialize handler checkpoint",
                cause=exc,
                context={"label": safe_label, "metadata_keys": list(metadata_map.keys())},
                code="HANDLER_MEMORY_CHECKPOINT_SERIALIZATION_FAILED",
                policy=self.error_policy,
            ) from exc

        with self._lock:
            self._append_checkpoint(checkpoint)
            self._mirror_checkpoint(checkpoint)
            self._prune_expired_locked(now=now)

        logger.debug("Handler checkpoint saved | id=%s label=%s", checkpoint["id"], safe_label)
        return str(checkpoint["id"])

    def get_checkpoint(self, checkpoint_id: str, *, include_expired: bool = False) -> Optional[Dict[str, Any]]:
        """Return a checkpoint payload by id, or None when missing/expired."""
        if not checkpoint_id:
            return None
        with self._lock:
            checkpoint = self._checkpoint_index.get(str(checkpoint_id))
            if checkpoint is None:
                return None
            if not include_expired and self._is_expired(checkpoint):
                return None
            return self._copy_payload(checkpoint)

    def restore_checkpoint(self, checkpoint_id: str, *, include_expired: bool = False) -> Optional[Dict[str, Any]]:
        """Return a checkpoint's saved state, preserving the legacy API shape."""
        checkpoint = self.get_checkpoint(checkpoint_id, include_expired=include_expired)
        if checkpoint is None:
            return None
        state = checkpoint.get("state", {})
        return self._copy_payload(state) if isinstance(state, Mapping) else {}

    def find_checkpoints(
        self,
        label: Optional[str] = None,
        max_age: Optional[float] = None,
        *,
        task_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata_filters: Optional[Mapping[str, Any]] = None,
        include_expired: bool = False,
        limit: Optional[int] = None,
        newest_first: bool = True,
    ) -> List[Dict[str, Any]]:
        """Find checkpoint payloads by label, age, task id, correlation id, and metadata filters."""
        now = utc_timestamp()
        safe_label = normalize_identifier(label, default="") if label else None
        safe_limit = coerce_int(limit, self.max_query_results, minimum=1, maximum=self.max_query_results)
        metadata_filters = coerce_mapping(metadata_filters)

        with self._lock:
            candidates = list(self._checkpoints)
            if newest_first:
                candidates = list(reversed(candidates))

            results: List[Dict[str, Any]] = []
            for checkpoint in candidates:
                if not include_expired and self._is_expired(checkpoint, now=now):
                    continue
                if safe_label and checkpoint.get("label") != safe_label:
                    continue
                if max_age is not None and now - coerce_float(checkpoint.get("created"), now) > coerce_float(max_age, 0.0, minimum=0.0):
                    continue
                metadata = coerce_mapping(checkpoint.get("metadata"))
                if task_id and str(metadata.get("task_id") or checkpoint.get("state", {}).get("task_id")) != str(task_id):
                    continue
                if correlation_id and str(metadata.get("correlation_id") or checkpoint.get("correlation_id")) != str(correlation_id):
                    continue
                if metadata_filters and not self._metadata_matches(metadata, metadata_filters):
                    continue
                results.append(self._copy_payload(checkpoint))
                if len(results) >= safe_limit:
                    break
            return results

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint from the in-memory buffers and indexes."""
        with self._lock:
            if checkpoint_id not in self._checkpoint_index:
                return False
            self._checkpoint_index.pop(checkpoint_id, None)
            self._checkpoints = deque(
                [checkpoint for checkpoint in self._checkpoints if checkpoint.get("id") != checkpoint_id],
                maxlen=self.max_checkpoints,
            )
            self._rebuild_indexes_locked()
            return True

    def clear_checkpoints(self, *, label: Optional[str] = None) -> int:
        """Clear all checkpoints or only checkpoints for one label."""
        with self._lock:
            if label is None:
                count = len(self._checkpoints)
                self._checkpoints.clear()
                self._checkpoint_index.clear()
                self._label_index.clear()
                self._correlation_index.clear()
                self._task_index.clear()
                return count

            safe_label = normalize_identifier(label, default="checkpoint")
            before = len(self._checkpoints)
            self._checkpoints = deque(
                [checkpoint for checkpoint in self._checkpoints if checkpoint.get("label") != safe_label],
                maxlen=self.max_checkpoints,
            )
            self._rebuild_indexes_locked()
            return before - len(self._checkpoints)

    def append_telemetry(
        self,
        event: Mapping[str, Any],
        *,
        ttl_seconds: Optional[float] = None,
        mirror: Optional[bool] = None,
    ) -> None:
        """Append a sanitized telemetry event to the bounded telemetry stream."""
        self._validate_mapping(event, source="telemetry event")
        telemetry_event = self._prepare_event(
            event,
            default_event_type="handler_telemetry",
            ttl_seconds=self.default_telemetry_ttl_seconds if ttl_seconds is None else ttl_seconds,
        )

        with self._lock:
            self._telemetry.append(telemetry_event)
            if self._should_mirror("telemetry", mirror):
                append_shared_memory_list(
                    self.shared_memory,
                    self.TELEMETRY_KEY,
                    telemetry_event,
                    max_items=self.max_telemetry_events,
                    ttl=self.shared_memory_telemetry_ttl_seconds,
                )
            self._prune_expired_locked()

    def append_recovery_telemetry(
        self,
        *,
        failure: Mapping[str, Any],
        recovery: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
        insight: Optional[Mapping[str, Any]] = None,
        sla: Optional[Mapping[str, Any]] = None,
        strategy_distribution: Optional[Mapping[str, Any]] = None,
        event_type: str = "handler_recovery",
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build and append a normalized Handler recovery telemetry event."""
        try:
            event = build_telemetry_event(
                event_type=event_type,
                failure=failure,
                recovery=recovery,
                context=context,
                insight=insight,
                sla=sla,
                strategy_distribution=strategy_distribution,
                correlation_id=correlation_id,
            )
            self.append_telemetry(event)
            return event
        except HandlerError:
            raise
        except Exception as exc:
            raise TelemetryError(
                "Unable to append recovery telemetry",
                cause=exc,
                context={"event_type": event_type},
                code="HANDLER_MEMORY_RECOVERY_TELEMETRY_FAILED",
                policy=self.error_policy,
            ) from exc

    def recent_telemetry(
        self,
        limit: int = 100,
        *,
        event_type: Optional[str] = None,
        agent: Optional[str] = None,
        task_id: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        context_hash: Optional[str] = None,
        since: Optional[float] = None,
        include_expired: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return recent telemetry events with optional operational filters."""
        return self.query_telemetry(
            limit=limit,
            event_type=event_type,
            agent=agent,
            task_id=task_id,
            severity=severity,
            status=status,
            context_hash=context_hash,
            since=since,
            include_expired=include_expired,
        )

    def query_telemetry(
        self,
        *,
        limit: int = 100,
        event_type: Optional[str] = None,
        agent: Optional[str] = None,
        task_id: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        context_hash: Optional[str] = None,
        since: Optional[float] = None,
        include_expired: bool = False,
        newest_first: bool = False,
    ) -> List[Dict[str, Any]]:
        """Query telemetry without exposing the internal deque."""
        safe_limit = coerce_int(limit, 100, minimum=1, maximum=self.max_query_results)
        now = utc_timestamp()
        normalized_event_type = normalize_identifier(event_type, default="") if event_type else None
        normalized_severity = normalize_severity(severity) if severity else None
        normalized_status = str(status).lower() if status else None

        with self._lock:
            candidates = list(self._telemetry)
            if newest_first:
                candidates = list(reversed(candidates))

            results: List[Dict[str, Any]] = []
            for event in candidates:
                if not include_expired and self._is_expired(event, now=now):
                    continue
                if since is not None and coerce_float(event.get("timestamp"), 0.0) < coerce_float(since, 0.0):
                    continue
                if normalized_event_type and event.get("event_type") != normalized_event_type:
                    continue
                context = coerce_mapping(event.get("context"))
                failure = coerce_mapping(event.get("failure"))
                recovery = coerce_mapping(event.get("recovery"))
                if agent and str(context.get("agent")) != str(agent):
                    continue
                if task_id and str(context.get("task_id")) != str(task_id):
                    continue
                if normalized_severity and failure.get("severity") != normalized_severity:
                    continue
                if normalized_status and str(recovery.get("status", "")).lower() != normalized_status:
                    continue
                if context_hash and failure.get("context_hash") != context_hash:
                    continue
                results.append(self._copy_payload(event))
                if len(results) >= safe_limit:
                    break
            return results

    def append_postmortem(
        self,
        postmortem: Optional[Mapping[str, Any]] = None,
        *,
        normalized_failure: Optional[Mapping[str, Any]] = None,
        recovery_result: Optional[Mapping[str, Any]] = None,
        telemetry: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
        ttl_seconds: Optional[float] = None,
        mirror: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Append either a prebuilt postmortem or build one from failure/recovery/telemetry."""
        try:
            if postmortem is None:
                require_keys({"normalized_failure": normalized_failure, "recovery_result": recovery_result}, ("normalized_failure", "recovery_result"), source="postmortem")
                event = build_postmortem(
                    normalized_failure=normalized_failure or {},
                    recovery_result=recovery_result or {},
                    telemetry=telemetry,
                    context=context,
                )
            else:
                event = coerce_mapping(postmortem)

            prepared = self._prepare_event(
                event,
                default_event_type="handler_postmortem",
                ttl_seconds=self.default_postmortem_ttl_seconds if ttl_seconds is None else ttl_seconds,
            )

            with self._lock:
                self._postmortems.append(prepared)
                if self._should_mirror("postmortems", mirror):
                    append_shared_memory_list(
                        self.shared_memory,
                        self.POSTMORTEM_KEY,
                        prepared,
                        max_items=self.max_postmortems,
                        ttl=self.shared_memory_postmortem_ttl_seconds,
                    )
                self._prune_expired_locked()
            return self._copy_payload(prepared)
        except HandlerError:
            raise
        except Exception as exc:
            raise TelemetryError(
                "Unable to append handler postmortem",
                cause=exc,
                context={"has_prebuilt_postmortem": postmortem is not None},
                code="HANDLER_MEMORY_POSTMORTEM_APPEND_FAILED",
                policy=self.error_policy,
            ) from exc

    def recent_postmortems(
        self,
        limit: int = 100,
        *,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        task_id: Optional[str] = None,
        include_expired: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return recent postmortem records with optional filters."""
        safe_limit = coerce_int(limit, 100, minimum=1, maximum=self.max_query_results)
        normalized_severity = normalize_severity(severity) if severity else None
        normalized_status = str(status).lower() if status else None
        now = utc_timestamp()

        with self._lock:
            results: List[Dict[str, Any]] = []
            for event in reversed(self._postmortems):
                if not include_expired and self._is_expired(event, now=now):
                    continue
                if normalized_severity and event.get("severity") != normalized_severity:
                    continue
                if normalized_status and str(event.get("recovery_status", "")).lower() != normalized_status:
                    continue
                if task_id and str(event.get("task_id")) != str(task_id):
                    continue
                results.append(self._copy_payload(event))
                if len(results) >= safe_limit:
                    break
            return list(reversed(results))

    def telemetry_stats(self, *, limit: Optional[int] = None, since: Optional[float] = None) -> Dict[str, Any]:
        """Summarize telemetry volume, recovery rates, severity distribution, and strategy distribution."""
        events = self.query_telemetry(limit=limit or self.max_query_results, since=since, include_expired=False)
        stats = success_rate_for_events(events)
        severity_counts: Counter[str] = Counter()
        strategy_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()

        for event in events:
            failure = coerce_mapping(event.get("failure"))
            recovery = coerce_mapping(event.get("recovery"))
            severity_counts[str(failure.get("severity", "unknown"))] += 1
            category_counts[str(failure.get("category", "unknown"))] += 1
            strategy_counts[strategy_base_name(recovery.get("strategy"))] += 1

        return {
            **stats,
            "severity_counts": dict(severity_counts),
            "category_counts": dict(category_counts),
            "strategy_counts": dict(strategy_counts),
            "window_size": len(events),
        }

    def failure_history(
        self,
        *,
        context_hash: Optional[str] = None,
        signature: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return telemetry events matching a failure context hash or insight signature."""
        safe_limit = coerce_int(limit, 100, minimum=1, maximum=self.max_query_results)
        with self._lock:
            matches = [
                self._copy_payload(event)
                for event in reversed(self._telemetry)
                if event_matches_failure(event, context_hash=context_hash, signature=signature)
            ]
        return list(reversed(matches[:safe_limit]))

    def prune_expired(self) -> Dict[str, int]:
        """Prune expired checkpoints, telemetry events, and postmortems."""
        with self._lock:
            return self._prune_expired_locked()

    def clear(self) -> None:
        """Clear all in-memory handler memory streams."""
        with self._lock:
            self._checkpoints.clear()
            self._telemetry.clear()
            self._postmortems.clear()
            self._checkpoint_index.clear()
            self._label_index.clear()
            self._correlation_index.clear()
            self._task_index.clear()

    def snapshot(self, *, include_payloads: bool = False) -> Dict[str, Any]:
        """Return operational state for diagnostics or tests."""
        with self._lock:
            snapshot = {
                "schema": "handler.memory.snapshot.v2",
                "timestamp": utc_timestamp(),
                "config": {
                    "max_checkpoints": self.max_checkpoints,
                    "max_telemetry_events": self.max_telemetry_events,
                    "max_postmortems": self.max_postmortems,
                    "max_checkpoint_state_chars": self.max_checkpoint_state_chars,
                    "max_checkpoint_metadata_chars": self.max_checkpoint_metadata_chars,
                    "max_event_chars": self.max_event_chars,
                    "copy_on_read": self.copy_on_read,
                    "sanitize_payloads": self.sanitize_payloads,
                    "mirror_to_shared_memory": self.mirror_to_shared_memory,
                },
                "counts": {
                    "checkpoints": len(self._checkpoints),
                    "telemetry": len(self._telemetry),
                    "postmortems": len(self._postmortems),
                    "checkpoint_index": len(self._checkpoint_index),
                },
                "labels": {label: len(ids) for label, ids in self._label_index.items()},
            }
            if include_payloads:
                snapshot["checkpoints"] = [self._copy_payload(item) for item in self._checkpoints]
                snapshot["telemetry"] = [self._copy_payload(item) for item in self._telemetry]
                snapshot["postmortems"] = [self._copy_payload(item) for item in self._postmortems]
            return snapshot

    def export_state(self, *, include_expired: bool = False) -> Dict[str, Any]:
        """Export a sanitized, JSON-safe copy of all bounded memory state."""
        now = utc_timestamp()
        with self._lock:
            payload = {
                "schema": "handler.memory.export.v2",
                "timestamp": now,
                "checkpoints": [
                    self._copy_payload(checkpoint)
                    for checkpoint in self._checkpoints
                    if include_expired or not self._is_expired(checkpoint, now=now)
                ],
                "telemetry": [
                    self._copy_payload(event)
                    for event in self._telemetry
                    if include_expired or not self._is_expired(event, now=now)
                ],
                "postmortems": [
                    self._copy_payload(event)
                    for event in self._postmortems
                    if include_expired or not self._is_expired(event, now=now)
                ],
            }
            return make_json_safe(payload)  # type: ignore[return-value]

    def import_state(self, payload: Mapping[str, Any], *, replace: bool = False) -> Dict[str, int]:
        """Import exported HandlerMemory state into the current bounded buffers."""
        self._validate_mapping(payload, source="memory import payload")
        imported = {"checkpoints": 0, "telemetry": 0, "postmortems": 0}

        with self._lock:
            if replace:
                self.clear()

            for checkpoint in coerce_list(payload.get("checkpoints")):
                if isinstance(checkpoint, Mapping) and checkpoint.get("id"):
                    self._append_checkpoint(dict(checkpoint))
                    imported["checkpoints"] += 1

            for event in coerce_list(payload.get("telemetry")):
                if isinstance(event, Mapping):
                    self._telemetry.append(self._prepare_event(event, default_event_type="handler_telemetry", ttl_seconds=None))
                    imported["telemetry"] += 1

            for postmortem in coerce_list(payload.get("postmortems")):
                if isinstance(postmortem, Mapping):
                    self._postmortems.append(self._prepare_event(postmortem, default_event_type="handler_postmortem", ttl_seconds=None))
                    imported["postmortems"] += 1

            self._rebuild_indexes_locked()
        return imported

    def health(self) -> Dict[str, Any]:
        """Return a compact memory health payload for dashboards and smoke tests."""
        with self._lock:
            telemetry_stats = success_rate_for_events(self._telemetry)
            return {
                "status": "ok",
                "timestamp": utc_timestamp(),
                "capacity": {
                    "checkpoints": {"used": len(self._checkpoints), "max": self.max_checkpoints},
                    "telemetry": {"used": len(self._telemetry), "max": self.max_telemetry_events},
                    "postmortems": {"used": len(self._postmortems), "max": self.max_postmortems},
                },
                "telemetry": telemetry_stats,
                "expired": {
                    "checkpoints": sum(1 for item in self._checkpoints if self._is_expired(item)),
                    "telemetry": sum(1 for item in self._telemetry if self._is_expired(item)),
                    "postmortems": sum(1 for item in self._postmortems if self._is_expired(item)),
                },
            }

    def attach_shared_memory(self, shared_memory: Any, *, mirror_existing: bool = False) -> None:
        """Attach a SharedMemory-like object and optionally mirror current buffers."""
        self.shared_memory = shared_memory
        if not mirror_existing:
            return
        with self._lock:
            for checkpoint in self._checkpoints:
                self._mirror_checkpoint(checkpoint)
            for event in self._telemetry:
                append_shared_memory_list(self.shared_memory, self.TELEMETRY_KEY, event, max_items=self.max_telemetry_events, ttl=self.shared_memory_telemetry_ttl_seconds)
            for event in self._postmortems:
                append_shared_memory_list(self.shared_memory, self.POSTMORTEM_KEY, event, max_items=self.max_postmortems, ttl=self.shared_memory_postmortem_ttl_seconds)

    @staticmethod
    def _optional_float(value: Any, *, default: Optional[float]) -> Optional[float]:
        if value is None:
            return default
        parsed = coerce_float(value, default if default is not None else 0.0, minimum=0.0)
        return parsed

    @staticmethod
    def _optional_int(value: Any, *, default: Optional[int]) -> Optional[int]:
        if value is None:
            return default
        return coerce_int(value, default if default is not None else 0, minimum=0)

    def _validate_mapping(self, value: Any, *, source: str) -> None:
        if not isinstance(value, Mapping):
            raise ValidationError(
                f"Handler memory expected mapping for {source}",
                context={"source": source, "actual_type": type(value).__name__},
                code="HANDLER_MEMORY_MAPPING_REQUIRED",
                policy=self.error_policy,
            )

    def _copy_payload(self, payload: Any) -> Any:
        if not self.copy_on_read:
            return payload
        return copy.deepcopy(payload)

    def _enforce_payload_size(self, payload: Any, max_chars: int, *, source: str) -> None:
        serialized_length = len(stable_json_dumps(payload))
        if serialized_length > max_chars:
            raise SerializationError(
                f"Handler memory {source} exceeds configured size limit",
                context={"source": source, "serialized_chars": serialized_length, "max_chars": max_chars},
                code="HANDLER_MEMORY_PAYLOAD_TOO_LARGE",
                policy=self.error_policy,
            )

    def _checkpoint_summary(self, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
        state = coerce_mapping(checkpoint.get("state"))
        metadata = coerce_mapping(checkpoint.get("metadata"))
        return compact_dict(
            {
                "task_id": metadata.get("task_id") or state.get("task_id"),
                "agent": metadata.get("agent") or state.get("agent"),
                "route": metadata.get("route") or state.get("route"),
                "correlation_id": metadata.get("correlation_id") or state.get("correlation_id"),
                "state_keys": sorted(str(key) for key in state.keys())[:25],
                "metadata_keys": sorted(str(key) for key in metadata.keys())[:25],
            },
            drop_none=True,
            drop_empty=True,
        )

    def _append_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if not checkpoint.get("id"):
            raise ValidationError(
                "Checkpoint payload missing id",
                context={"checkpoint_keys": list(checkpoint.keys())},
                code="HANDLER_MEMORY_CHECKPOINT_ID_MISSING",
                policy=self.error_policy,
            )

        if len(self._checkpoints) == self.max_checkpoints and self._checkpoints:
            evicted = self._checkpoints[0]
            self._remove_checkpoint_indexes(evicted)

        checkpoint_id = str(checkpoint["id"])
        existing = self._checkpoint_index.get(checkpoint_id)
        if existing is not None:
            self._checkpoints = deque(
                [item for item in self._checkpoints if item.get("id") != checkpoint_id],
                maxlen=self.max_checkpoints,
            )
            self._remove_checkpoint_indexes(existing)

        self._checkpoints.append(checkpoint)
        self._checkpoint_index[checkpoint_id] = checkpoint
        self._add_checkpoint_indexes(checkpoint)

    def _add_checkpoint_indexes(self, checkpoint: Mapping[str, Any]) -> None:
        checkpoint_id = str(checkpoint.get("id"))
        label = str(checkpoint.get("label") or "checkpoint")
        metadata = coerce_mapping(checkpoint.get("metadata"))
        state = coerce_mapping(checkpoint.get("state"))
        correlation_id = metadata.get("correlation_id") or state.get("correlation_id")
        task_id = metadata.get("task_id") or state.get("task_id")

        self._label_index.setdefault(label, set()).add(checkpoint_id)
        if correlation_id:
            self._correlation_index.setdefault(str(correlation_id), set()).add(checkpoint_id)
        if task_id:
            self._task_index.setdefault(str(task_id), set()).add(checkpoint_id)

    def _remove_checkpoint_indexes(self, checkpoint: Mapping[str, Any]) -> None:
        checkpoint_id = str(checkpoint.get("id"))
        self._checkpoint_index.pop(checkpoint_id, None)
        for index in (self._label_index, self._correlation_index, self._task_index):
            for key in list(index.keys()):
                index[key].discard(checkpoint_id)
                if not index[key]:
                    index.pop(key, None)

    def _rebuild_indexes_locked(self) -> None:
        self._checkpoint_index.clear()
        self._label_index.clear()
        self._correlation_index.clear()
        self._task_index.clear()
        for checkpoint in self._checkpoints:
            if checkpoint.get("id"):
                self._checkpoint_index[str(checkpoint["id"])] = checkpoint
                self._add_checkpoint_indexes(checkpoint)

    def _mirror_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        if not self._should_mirror("checkpoints", None):
            return
        checkpoint_id = str(checkpoint.get("id"))
        shared_memory_set(self.shared_memory, checkpoint_id, self._copy_payload(checkpoint), ttl=self.shared_memory_checkpoint_ttl_seconds)
        append_shared_memory_list(
            self.shared_memory,
            self.CHECKPOINT_KEY,
            self._checkpoint_summary(checkpoint) | {"id": checkpoint_id, "created": checkpoint.get("created")},
            max_items=self.max_checkpoints,
            ttl=self.shared_memory_checkpoint_ttl_seconds,
        )

    def _prepare_event(self, event: Mapping[str, Any], *, default_event_type: str, ttl_seconds: Optional[float]) -> Dict[str, Any]:
        now = utc_timestamp()
        ttl = self._optional_float(ttl_seconds, default=None)
        expires_at = (now + ttl) if ttl is not None and ttl > 0 else event.get("expires_at")
        event_type = event.get("event_type") or default_event_type

        payload = make_json_safe(self.error_policy.sanitize_context(event) if self.sanitize_payloads else event)
        if not isinstance(payload, dict):
            raise SerializationError(
                "Handler memory event could not be converted to a mapping",
                context={"event_type": event_type, "payload_type": type(payload).__name__},
                code="HANDLER_MEMORY_EVENT_MAPPING_REQUIRED",
                policy=self.error_policy,
            )

        payload["event_type"] = normalize_identifier(event_type, default=default_event_type)
        payload["timestamp"] = coerce_float(payload.get("timestamp"), now)
        payload["correlation_id"] = payload.get("correlation_id") or self._correlation_from_event(payload) or generate_correlation_id("handler")
        payload["expires_at"] = expires_at
        prepared = compact_dict(payload, drop_none=True)
        self._enforce_payload_size(prepared, self.max_event_chars, source=default_event_type)
        return prepared

    @staticmethod
    def _correlation_from_event(event: Mapping[str, Any]) -> Optional[str]:
        failure = coerce_mapping(event.get("failure"))
        context = coerce_mapping(event.get("context"))
        return failure.get("correlation_id") or context.get("correlation_id")

    def _should_mirror(self, stream: str, mirror: Optional[bool]) -> bool:
        if mirror is not None:
            return bool(mirror)
        if self.shared_memory is None:
            return False
        stream_key = normalize_identifier(stream, default="")
        if stream_key == "checkpoints":
            return self.mirror_checkpoints_to_shared_memory
        if stream_key == "telemetry":
            return self.mirror_telemetry_to_shared_memory
        if stream_key == "postmortems":
            return self.mirror_postmortems_to_shared_memory
        return self.mirror_to_shared_memory

    @staticmethod
    def _is_expired(payload: Mapping[str, Any], *, now: Optional[float] = None) -> bool:
        expires_at = payload.get("expires_at")
        if expires_at is None:
            return False
        return coerce_float(expires_at, 0.0) <= (utc_timestamp() if now is None else now)

    @staticmethod
    def _metadata_matches(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
        for key, expected in filters.items():
            if metadata.get(key) != expected:
                return False
        return True

    def _prune_expired_locked(self, *, now: Optional[float] = None) -> Dict[str, int]:
        current_time = utc_timestamp() if now is None else now
        before = {
            "checkpoints": len(self._checkpoints),
            "telemetry": len(self._telemetry),
            "postmortems": len(self._postmortems),
        }
        self._checkpoints = deque(
            [item for item in self._checkpoints if not self._is_expired(item, now=current_time)],
            maxlen=self.max_checkpoints,
        )
        self._telemetry = deque(
            [item for item in self._telemetry if not self._is_expired(item, now=current_time)],
            maxlen=self.max_telemetry_events,
        )
        self._postmortems = deque(
            [item for item in self._postmortems if not self._is_expired(item, now=current_time)],
            maxlen=self.max_postmortems,
        )
        self._rebuild_indexes_locked()
        after = {
            "checkpoints": len(self._checkpoints),
            "telemetry": len(self._telemetry),
            "postmortems": len(self._postmortems),
        }
        return {key: before[key] - after[key] for key in before}


if __name__ == "__main__":
    print("\n=== Running Handler Memory ===\n")
    printer.status("TEST", "Handler Memory initialized", "info")
    from ..collaborative.shared_memory import SharedMemory

    strict_policy = HandlerErrorPolicy(
        name="handler_memory.strict_test",
        expose_internal_messages=False,
        include_context_in_public=False,
        include_context_in_telemetry=True,
        max_message_chars=240,
        max_string_chars=160,
    )

    shared_memory = SharedMemory()
    memory = HandlerMemory(
        config={
            "max_checkpoints": 3,
            "max_telemetry_events": 5,
            "max_postmortems": 5,
            "default_checkpoint_ttl_seconds": 60,
            "mirror_to_shared_memory": True,
            "copy_on_read": True,
            "sanitize_payloads": True,
        },
        shared_memory=shared_memory,
        error_policy=strict_policy,
    )

    sensitive_state = {
        "task_id": "handler-memory-smoke-001",
        "route": "handler.recovery",
        "agent": "demo_agent",
        "step": "pre_recovery",
        "password": "SuperSecret123",
        "nested": {"api_key": "sk-test-123", "safe": "visible metadata"},
    }

    checkpoint_id = memory.save_checkpoint(
        label="pre_recovery",
        state=sensitive_state,
        metadata={
            "task_id": "handler-memory-smoke-001",
            "agent": "demo_agent",
            "route": "handler.recovery",
            "strategy": "timeout",
            "correlation_id": "corr-handler-memory-test",
        },
    )
    checkpoint = memory.get_checkpoint(checkpoint_id)
    restored = memory.restore_checkpoint(checkpoint_id)
    found = memory.find_checkpoints(label="pre_recovery", task_id="handler-memory-smoke-001")

    failure = build_normalized_failure(
        error=TimeoutError("Upstream timed out with Authorization: Bearer token-123"),
        context={"task_id": "handler-memory-smoke-001", "agent": "demo_agent", "correlation_id": "corr-handler-memory-test"},
        policy=strict_policy,
        source="handler.memory.__main__",
        correlation_id="corr-handler-memory-test",
    )
    recovery = {
        "status": "recovered",
        "strategy": "timeout",
        "attempts": 1,
        "sla": {"remaining_seconds": 12.5, "mode": "standard"},
        "strategy_distribution": {"timeout": 0.75, "runtime": 0.25},
    }
    telemetry = memory.append_recovery_telemetry(
        failure=failure,
        recovery=recovery,
        context={"task_id": "handler-memory-smoke-001", "agent": "demo_agent", "route": "handler.recovery", "correlation_id": "corr-handler-memory-test"},
        insight={"signature": "timeout:abc", "recommendation": "retry_with_backoff"},
    )
    postmortem = memory.append_postmortem(
        normalized_failure=failure,
        recovery_result=recovery,
        telemetry=telemetry,
        context={"task_id": "handler-memory-smoke-001", "agent": "demo_agent", "correlation_id": "corr-handler-memory-test"},
    )

    stats = memory.telemetry_stats()
    history = memory.failure_history(context_hash=failure["context_hash"])
    snapshot = memory.snapshot(include_payloads=False)
    exported = memory.export_state()
    imported_memory = HandlerMemory(config={"max_checkpoints": 3, "max_telemetry_events": 5, "max_postmortems": 5}, error_policy=strict_policy)
    imported = imported_memory.import_state(exported, replace=True)
    health = memory.health()

    serialized = stable_json_dumps(
        {
            "checkpoint": checkpoint,
            "restored": restored,
            "found": found,
            "telemetry": telemetry,
            "postmortem": postmortem,
            "stats": stats,
            "history": history,
            "snapshot": snapshot,
            "exported": exported,
            "imported": imported,
            "health": health,
            "shared_memory": shared_memory.get_all_keys(),
        }
    )

    assert checkpoint is not None
    assert restored is not None
    assert len(found) == 1
    assert telemetry["failure"]["type"] == "TimeoutError"
    assert postmortem["recovery_status"] == "recovered"
    assert stats["total"] == 1
    assert stats["recovered"] == 1
    assert len(history) == 1
    assert snapshot["counts"]["checkpoints"] == 1
    assert imported["checkpoints"] == 1
    assert imported["telemetry"] == 1
    assert imported["postmortems"] == 1
    assert health["status"] == "ok"
    assert "SuperSecret123" not in serialized
    assert "sk-test-123" not in serialized
    assert "token-123" not in serialized

    printer.pretty("Checkpoint", checkpoint, "success")
    printer.pretty("Telemetry stats", stats, "success")
    printer.pretty("Memory health", health, "success")
    print("\n=== Test ran successfully ===\n")
