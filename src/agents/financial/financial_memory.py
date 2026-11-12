"""Bounded, checkpoint-safe local analysis history for the SLAI Financial agent.

Only FinancialResult objects are stored. This is not a ledger, bank account
repository, payment idempotency store, shared-memory bus or model artifact DB.
Serialized state is integrity-checked for accidental corruption, not signed or
encrypted: store snapshots only through the application's protected checkpoint
infrastructure. Expiry is UTC-based so state survives restart without reset.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Mapping

from .financial_types import FinancialResult
from .utils.config_loader import load_global_config, get_config_section
from .utils.financial_errors import *
from .utils.financial_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Memory")
printer = PrettyPrinter()

_SCHEMA = "slai.financial.memory.v1"


class FinancialMemory:
    """Thread-safe recent-results cache with bounded, validated state restore."""

    def __init__(self):
        self.config = load_global_config()
        self.financial_memory_config = get_config_section("financial_memory", config=self.config)
        try:
            self.max_records = require_positive_int(self.financial_memory_config["max_records"], "max_records", maximum=100000)
            self.ttl_seconds = require_positive_int(self.financial_memory_config["ttl_seconds"], "ttl_seconds", maximum=315360000)
        except KeyError as exc:
            raise FinancialConfigurationError("Missing financial_memory configuration key", context={"key": str(exc)}) from exc
        self._lock = RLock()
        self._records: OrderedDict[str, tuple[FinancialResult, datetime]] = OrderedDict()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _prune(self, now: datetime) -> None:
        expired = [key for key, (_, deadline) in self._records.items() if deadline <= now]
        for key in expired:
            self._records.pop(key, None)
        while len(self._records) > self.max_records:
            self._records.popitem(last=False)

    def remember(self, result: FinancialResult, *, key: str | None = None) -> str:
        """Remember an immutable result by its request id or a scoped public key.

        Re-registering a key updates the latest snapshot. This is an analysis
        cache, not a durable/audited journal and not an idempotent payment store.
        """
        if not isinstance(result, FinancialResult):
            raise FinancialValidationError("FinancialMemory only accepts FinancialResult")
        identity = normalize_identifier(result.request_id if key is None else key, "memory_key")
        now = self._now()
        deadline = now + timedelta(seconds=self.ttl_seconds)
        with self._lock:
            self._prune(now)
            self._records.pop(identity, None)
            self._records[identity] = (result, deadline)
            self._prune(now)
        logger.debug("Financial analysis result memorized | status=%s", result.status)
        return identity

    def get(self, key: str) -> FinancialResult | None:
        identity = normalize_identifier(key, "memory_key")
        with self._lock:
            self._prune(self._now())
            entry = self._records.get(identity)
            return None if entry is None else entry[0]

    def recent(self, *, limit: int | None = None) -> tuple[FinancialResult, ...]:
        count = self.max_records if limit is None else require_positive_int(limit, "limit", maximum=self.max_records)
        with self._lock:
            self._prune(self._now())
            return tuple(record for record, _ in list(self._records.values())[-count:][::-1])

    def remove(self, key: str) -> bool:
        identity = normalize_identifier(key, "memory_key")
        with self._lock:
            return self._records.pop(identity, None) is not None

    def add(self, result: FinancialResult, *, key: str | None = None) -> str:
        """Alias for remember() to match common cache terminology."""
        return self.remember(result, key=key)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            self._prune(self._now())
            return len(self._records)

    def snapshot(self) -> dict[str, Any]:
        """Return versioned, JSON-ready state for SLAI's checkpoint manager.

        SHA-256 detects accidental/unsanctioned changes but is NOT an HMAC or
        authentication substitute. Snapshot storage may contain financial data.
        """
        with self._lock:
            self._prune(self._now())
            rows = [
                {"key": key, "result": result.to_dict(), "expires_at": iso_utc(expiry)}
                for key, (result, expiry) in self._records.items()
            ]
        payload = {"schema": _SCHEMA, "entries": rows}
        return {**payload, "checksum_sha256": stable_fingerprint(payload)}

    def restore(self, state: Mapping[str, Any]) -> int:
        """Validate full snapshot before atomically replacing in-memory state."""
        if not isinstance(state, Mapping) or set(state) != {"schema", "entries", "checksum_sha256"}:
            raise FinancialValidationError("Invalid FinancialMemory snapshot envelope")
        if state["schema"] != _SCHEMA:
            raise FinancialIntegrityError("Unsupported financial-memory checkpoint schema")
        entries = state["entries"]
        if not isinstance(entries, list) or len(entries) > self.max_records:
            raise FinancialValidationError("FinancialMemory snapshot entries exceed configured limit")
        actual = stable_fingerprint({"schema": state["schema"], "entries": entries})
        if not isinstance(state["checksum_sha256"], str) or state["checksum_sha256"] != actual:
            raise FinancialIntegrityError("FinancialMemory snapshot checksum mismatch")
        now = self._now()
        candidate: OrderedDict[str, tuple[FinancialResult, datetime]] = OrderedDict()
        for row in entries:
            if not isinstance(row, Mapping) or set(row) != {"key", "result", "expires_at"}:
                raise FinancialValidationError("Invalid financial-memory entry")
            key = normalize_identifier(row["key"], "memory_key")
            if key in candidate:
                raise FinancialIntegrityError("Duplicate financial-memory checkpoint key")
            result = FinancialResult.from_dict(row["result"])
            expiry = to_utc_datetime(row["expires_at"], "expires_at")
            if expiry > now:
                candidate[key] = (result, expiry)
        with self._lock:
            self._records = candidate
        logger.info("Financial analysis memory restored | records=%s", len(candidate))
        return len(candidate)

    def repair(self) -> int:
        """Prune expired entries and return the number of remaining records."""
        with self._lock:
            self._prune(self._now())
            return len(self._records)

    def is_empty(self) -> bool:
        with self._lock:
            self._prune(self._now())
            return len(self._records) == 0

    def __contains__(self, key: str) -> bool:
        identity = normalize_identifier(key, "memory_key")
        with self._lock:
            self._prune(self._now())
            return identity in self._records

    def __iter__(self):
        with self._lock:
            self._prune(self._now())
            return iter(self._records.keys())

    def __repr__(self) -> str:
        with self._lock:
            self._prune(self._now())
            return f"FinancialMemory(records={len(self._records)})"


__all__ = ["FinancialMemory"]

if __name__ == "__main__":
    print("\n=== Running Financial Memory ===\n")
    printer.status("TEST", "Financial Memory initialized", "info")
    memory = FinancialMemory()
    result = FinancialResult(request_id="memory-check", domain="test", operation="sample",
                             status="success", result={"value": "20.00"})
    assert memory.remember(result) == "memory-check"
    assert memory.get("memory-check") is result
    assert len(memory.recent(limit=1)) == 1
    saved = memory.snapshot()
    memory.clear()
    assert len(memory) == 0
    assert memory.restore(saved) == 1
    restored = memory.get("memory-check")
    assert restored is not None and restored.result["value"] == "20.00"
    invalid = dict(saved)
    invalid["checksum_sha256"] = "0" * 64
    try:
        memory.restore(invalid)
        raise AssertionError("Corrupt memory checkpoint accepted")
    except FinancialIntegrityError:
        pass
    assert memory.remove("memory-check")
    print("\n=== Test ran successfully ===\n")
