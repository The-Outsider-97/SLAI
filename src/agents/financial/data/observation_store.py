"""Bitemporal (valid-time x knowledge-time) store for sourced financial facts.

Every stored fact carries two instants: ``observed_at`` (valid time: when the
fact describes the world) and ``source.retrieved_at`` (knowledge time: when the
system could first have known it). A read "as of" T therefore sees only facts
with BOTH instants <= T, which is what prevents look-ahead bias in backtests.
Revisions are separate facts with their own knowledge time; nothing is
overwritten. For revised macroeconomic series this is the "real-time data set"
design of Croushore & Stark (2001), "A real-time data set for macroeconomists",
Journal of Econometrics 105(1), 111-130: ``vintage="latest"`` returns, per valid
time, the newest vintage known at the as-of instant, while ``vintage="all"``
returns every vintage for audit.

Generic by construction: it is used for ``MarketObservation`` series and
``ExchangeRate`` pairs by FinancialData, and is the single owner of ingest
conflict detection, retention and as-of visibility rules. No I/O.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Callable, Hashable, Iterable, Sequence
from datetime import datetime
from threading import RLock
from typing import Any, Generic, TypeVar

from ..utils.financial_errors import *
from ..utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Observation Store")
printer = PrettyPrinter()

T = TypeVar("T")
VINTAGES = ("all", "latest")


def knowledge_order(item: Any) -> tuple[Any, ...]:
    """Total order: valid time, then knowledge time, then source and revision."""
    s = item.source
    return (item.observed_at, s.retrieved_at, s.source_id, s.revision or "")


def _vintage_order(item: Any) -> tuple[Any, ...]:
    s = item.source
    return (s.retrieved_at, s.source_id, s.revision or "")


class BitemporalStore(Generic[T]):
    """Thread-safe, bounded, append-only store of one fact type.

    ``identity`` of a fact is (bucket, observed_at, source_id, revision,
    retrieved_at). Replaying an identical fact is idempotent; a different
    payload under the same identity is an integrity violation and rejects the
    whole batch. Retention keeps the newest ``max_per_bucket`` facts per bucket.
    """

    def __init__(self, fact_type: type[T], bucket_of: Callable[[T], Hashable], *, max_per_bucket: int):
        if not isinstance(fact_type, type):
            raise FinancialValidationError("fact_type must be a class")
        if not callable(bucket_of):
            raise FinancialValidationError("bucket_of must be callable")
        self._type = fact_type
        self._bucket_of = bucket_of
        self.max_per_bucket = require_positive_int(max_per_bucket, "max_per_bucket", maximum=1_000_000)
        self._buckets: dict[Hashable, list[T]] = {}
        self._lock = RLock()

    # ------------------------------------------------------------------ write
    def _identity(self, bucket: Hashable, item: Any) -> tuple[Any, ...]:
        s = item.source
        return (bucket, item.observed_at, s.source_id, s.revision, s.retrieved_at)

    def ingest(self, items: Sequence[T]) -> int:
        """Atomically add facts; return the number newly stored."""
        if not isinstance(items, (list, tuple)) or any(not isinstance(x, self._type) for x in items):
            raise FinancialValidationError(f"Expected a list/tuple of {self._type.__name__} objects")
        if not items:
            return 0
        with self._lock:
            staged: dict[Hashable, list[T]] = {}
            changed = 0
            for item in items:
                bucket = self._bucket_of(item)
                current = staged.get(bucket)
                if current is None:
                    current = staged[bucket] = list(self._buckets.get(bucket, ()))
                identity = self._identity(bucket, item)
                duplicate = next((x for x in current if self._identity(bucket, x) == identity), None)
                if duplicate is not None:
                    if stable_fingerprint(duplicate.to_dict()) != stable_fingerprint(item.to_dict()):  # type: ignore[attr-defined]
                        raise FinancialIntegrityError(
                            "Conflicting financial observation for the same source/retrieval/revision key")
                    continue
                current.append(item)
                current.sort(key=knowledge_order)
                if len(current) > self.max_per_bucket:
                    del current[: len(current) - self.max_per_bucket]
                changed += 1
            self._buckets.update(staged)
        logger.debug("Bitemporal store ingest | type=%s | stored=%d", self._type.__name__, changed)
        return changed

    # ------------------------------------------------------------------- read
    @staticmethod
    def _visible(values: Iterable[Any], as_of: datetime, source_id: str | None) -> list[Any]:
        return [v for v in values
                if v.observed_at <= as_of and v.source.retrieved_at <= as_of
                and (source_id is None or v.source.source_id == source_id)]

    def known(self, bucket: Hashable, *, as_of: datetime, source_id: str | None = None) -> tuple[T, ...]:
        """All facts for ``bucket`` knowable at ``as_of`` (every vintage)."""
        as_of = to_utc_datetime(as_of, "as_of")
        with self._lock:
            values = tuple(self._buckets.get(bucket, ()))
        return tuple(self._visible(values, as_of, source_id))

    def latest(self, bucket: Hashable, *, as_of: datetime, source_id: str | None = None) -> T | None:
        """Fact with the greatest valid time known at ``as_of`` (newest vintage on ties)."""
        visible = self.known(bucket, as_of=as_of, source_id=source_id)
        return max(visible, key=knowledge_order) if visible else None

    def history(self, bucket: Hashable, *, start: datetime, end: datetime, as_of: datetime,
                source_id: str | None = None, vintage: str = "all") -> tuple[T, ...]:
        """Facts with ``start <= observed_at < end`` knowable at ``as_of``.

        ``vintage="latest"`` keeps one fact per valid time: the one with the
        newest knowledge time known at ``as_of`` (ties broken by source id and
        revision). Output is ordered by valid time.
        """
        if vintage not in VINTAGES:
            raise UnsupportedConventionError("vintage must be 'all' or 'latest'", context={"vintage": str(vintage)[:32]})
        begin, finish = to_utc_datetime(start, "from_time"), to_utc_datetime(end, "until_time")
        if begin >= finish:
            raise FinancialValidationError("Financial history requires from_time < until_time")
        window = [v for v in self.known(bucket, as_of=as_of, source_id=source_id)
                  if begin <= v.observed_at < finish]  # type: ignore[attr-defined]
        if vintage == "latest":
            newest: dict[datetime, Any] = {}
            for v in window:
                held = newest.get(v.observed_at)  # type: ignore[attr-defined]
                if held is None or _vintage_order(v) > _vintage_order(held):
                    newest[v.observed_at] = v  # type: ignore[attr-defined]
            window = list(newest.values())
        return tuple(sorted(window, key=knowledge_order))  # type: ignore[arg-type]

    # ------------------------------------------------------------ introspection
    def buckets(self) -> tuple[Hashable, ...]:
        with self._lock:
            return tuple(sorted(self._buckets, key=repr))

    def items(self) -> tuple[T, ...]:
        """Every stored fact (for checkpointing and audit), bucket order then knowledge order."""
        with self._lock:
            return tuple(v for key in sorted(self._buckets, key=repr) for v in self._buckets[key])

    def counts(self) -> tuple[int, int]:
        """(number of buckets, number of facts)."""
        with self._lock:
            return len(self._buckets), sum(map(len, self._buckets.values()))

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()


__all__ = ["BitemporalStore", "knowledge_order", "VINTAGES"]

if __name__ == "__main__":
    print("\n=== Running Financial Observation Store ===\n")
    printer.status("TEST", "Financial Observation Store initialized", "info")
    from datetime import timedelta, timezone
    from decimal import Decimal
    from ..financial_types import FinancialDataSource, MarketObservation
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def obs(value: str, observed: datetime, retrieved: datetime, revision: str | None = None):
        return MarketObservation("CPI", "index", Decimal(value), observed, FinancialDataSource("stat-office", retrieved, revision=revision), unit="index")

    store = BitemporalStore(MarketObservation, lambda x: (x.instrument_id, x.field_name), max_per_bucket=10)
    first, revised = obs("100", t0, t0 + timedelta(days=10)), obs("101", t0, t0 + timedelta(days=40), "r2")
    assert store.ingest([first, revised]) == 2 and store.ingest([first]) == 0
    key = ("CPI", "index")
    end = t0 + timedelta(days=1)
    assert store.latest(key, as_of=t0 + timedelta(days=5)) is None, "unreleased data must be invisible"
    assert store.latest(key, as_of=t0 + timedelta(days=20)).value == Decimal("100")
    assert store.latest(key, as_of=t0 + timedelta(days=50)).value == Decimal("101")
    assert len(store.history(key, start=t0, end=end, as_of=t0 + timedelta(days=50))) == 2
    only = store.history(key, start=t0, end=end, as_of=t0 + timedelta(days=50), vintage="latest")
    assert [x.value for x in only] == [Decimal("101")]
    assert [x.value for x in store.history(key, start=t0, end=end, as_of=t0 + timedelta(days=20), vintage="latest")] == [Decimal("100")]
    clash = obs("999", t0, t0 + timedelta(days=10))
    try:
        store.ingest([clash])
    except FinancialIntegrityError:
        pass
    else:
        raise AssertionError("conflicting payload must be rejected")
    assert store.counts() == (1, 2), "rejected batch must leave the store unchanged"
    small = BitemporalStore(MarketObservation, lambda x: (x.instrument_id, x.field_name), max_per_bucket=1)
    small.ingest([first, revised])
    assert small.counts() == (1, 1) and small.items()[0].value == Decimal("101")
    print("\n=== Test ran successfully ===\n")
