"""Bounded transient memory for formal verification results.

VerificationMemory is intentionally not a persistence or checkpoint system.
Durable agent recovery remains the responsibility of BaseAgent/checkpointing.
This class owns only Verification-specific transient result history and query
semantics.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Callable, Generic, Iterable, TypeVar


from src.tuning.utils.tuning_helpers import utc_iso, utc_now # type: ignore
from .utils.verification_errors import MalformedSpecificationError
from .utils.verification_helpers import normalize_tags, result_fingerprint
from .verification_result import VerificationResult
from .verification_types import MemorySettings, VerificationMethod, VerificationStatus
from logs.logger import get_logger, configure_logging, PrettyPrinter


TState = TypeVar("TState")


@dataclass(frozen=True, slots=True)
class VerificationRecord(Generic[TState]):
    """One immutable transient memory record."""

    record_id: str
    created_at: str
    result: VerificationResult[TState]
    fingerprint: str
    request_id: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        record_id = str(self.record_id).strip()
        created_at = str(self.created_at).strip()
        fingerprint = str(self.fingerprint).strip()
        if not record_id:
            raise MalformedSpecificationError("verification record_id must be non-empty")
        if not created_at:
            raise MalformedSpecificationError("verification created_at must be non-empty")
        if not isinstance(self.result, VerificationResult):
            raise MalformedSpecificationError("verification memory records require VerificationResult")
        if not fingerprint:
            raise MalformedSpecificationError("verification fingerprint must be non-empty")
        request_id = None if self.request_id is None else str(self.request_id).strip()
        if self.request_id is not None and not request_id:
            raise MalformedSpecificationError("request_id must be non-empty when provided")
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "tags", normalize_tags(self.tags))


class VerificationMemory(Generic[TState]):
    """Thread-safe FIFO history of immutable formal verification results."""

    def __init__(
        self,
        settings: MemorySettings | None = None,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._settings = settings or MemorySettings()
        self._clock = clock
        self._lock = RLock()
        self._records: "OrderedDict[str, VerificationRecord[TState]]" = OrderedDict()
        self._sequence = 0
        self._evictions = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def capacity(self) -> int:
        return self._settings.max_entries

    def record(
        self,
        result: VerificationResult[TState],
        *,
        request_id: str | None = None,
        tags: Iterable[str] | str | None = None,
        record_id: str | None = None,
    ) -> VerificationRecord[TState]:
        """Append one result, evicting the oldest record at capacity."""

        if not isinstance(result, VerificationResult):
            raise MalformedSpecificationError("VerificationMemory.record requires VerificationResult")
        normalized_tags = normalize_tags(tags)
        normalized_request = None if request_id is None else str(request_id).strip()
        if request_id is not None and not normalized_request:
            raise MalformedSpecificationError("request_id must be non-empty when provided")

        with self._lock:
            self._sequence += 1
            resolved_id = (
                str(record_id).strip()
                if record_id is not None
                else f"verification-{self._sequence:08d}"
            )
            if not resolved_id:
                raise MalformedSpecificationError("record_id must be non-empty when provided")
            if resolved_id in self._records:
                raise MalformedSpecificationError(
                    "verification record_id already exists",
                    context={"record_id": resolved_id},
                )

            record = VerificationRecord(
                record_id=resolved_id,
                created_at=utc_iso(self._clock()),
                result=result,
                fingerprint=result_fingerprint(result),
                request_id=normalized_request,
                tags=normalized_tags,
            )
            while len(self._records) >= self._settings.max_entries:
                self._records.popitem(last=False)
                self._evictions += 1
            self._records[resolved_id] = record
            return record

    def get(self, record_id: str) -> VerificationRecord[TState] | None:
        key = str(record_id).strip()
        if not key:
            raise MalformedSpecificationError("record_id must be non-empty")
        with self._lock:
            return self._records.get(key)

    def recent(self, limit: int | None = None) -> tuple[VerificationRecord[TState], ...]:
        """Return newest-first immutable records."""

        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise MalformedSpecificationError("memory recent limit must be a positive integer")
        with self._lock:
            values = tuple(reversed(tuple(self._records.values())))
        return values if limit is None else values[:limit]

    def query(
        self,
        *,
        status: VerificationStatus | None = None,
        method: VerificationMethod | None = None,
        backend: str | None = None,
        property_name: str | None = None,
        tags: Iterable[str] | str | None = None,
        limit: int | None = None,
    ) -> tuple[VerificationRecord[TState], ...]:
        """Filter newest-first by formal result metadata."""

        if status is not None and not isinstance(status, VerificationStatus):
            status = VerificationStatus(status)
        if method is not None and not isinstance(method, VerificationMethod):
            method = VerificationMethod(method)
        normalized_backend = None if backend is None else str(backend).strip()
        if backend is not None and not normalized_backend:
            raise MalformedSpecificationError("backend filter must be non-empty")
        normalized_property = None if property_name is None else str(property_name).strip()
        if property_name is not None and not normalized_property:
            raise MalformedSpecificationError("property_name filter must be non-empty")
        required_tags = frozenset(normalize_tags(tags))
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise MalformedSpecificationError("memory query limit must be a positive integer")

        matches: list[VerificationRecord[TState]] = []
        for record in self.recent():
            result = record.result
            if status is not None and result.status is not status:
                continue
            if method is not None and result.provenance.method is not method:
                continue
            if normalized_backend is not None and result.provenance.backend != normalized_backend:
                continue
            if normalized_property is not None and result.property_name != normalized_property:
                continue
            if required_tags and not required_tags.issubset(record.tags):
                continue
            matches.append(record)
            if limit is not None and len(matches) >= limit:
                break
        return tuple(matches)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def statistics(self) -> dict[str, object]:
        """Return a snapshot of bounded-memory operational counts."""

        with self._lock:
            records = tuple(self._records.values())
            evictions = self._evictions
            sequence = self._sequence
        by_status = Counter(record.result.status.value for record in records)
        by_method = Counter(record.result.provenance.method.value for record in records)
        return {
            "size": len(records),
            "capacity": self._settings.max_entries,
            "evictions": evictions,
            "records_created": sequence,
            "by_status": dict(sorted(by_status.items())),
            "by_method": dict(sorted(by_method.items())),
        }


__all__ = ["VerificationMemory", "VerificationRecord"]
