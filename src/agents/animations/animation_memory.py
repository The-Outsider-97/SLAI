"""Bounded, thread-safe operational history of committed animation edits.

This is *not* the authoritative project history, undo stack, or persistent
checkpoint store.  Those responsibilities remain with the scene provider,
Provenance, and the existing SLAI infrastructure.  Only immutable edit
summaries are retained; no scene snapshots, keyframe arrays, or spatial
data are duplicated.

Purpose
-------
``AnimationMemory`` exists to make the *operational* behaviour of the
agent observable:

* per-scene edit throughput,
* recent-edit replay for diagnostics,
* cross-process correlation of "which edits touched scene X?",
* and a bounded audit trail for post-hoc inspection.

It is deliberately **ephemeral**: it holds at most ``capacity`` records
and never spills to disk.  Anything that needs durability belongs in the
provider or in a downstream telemetry system.

Design invariants
-----------------
* **Bounded.**  At most ``capacity`` records are retained; the oldest
  are evicted (FIFO) as new ones arrive.  Capacity is fixed at
  construction time and never changes.
* **Thread-safe.**  All mutating and reading operations acquire a
  single reentrant lock.  Iteration snapshots the internal deque before
  yielding, so callers never observe a deque-changed-during-iteration
  error even under concurrent :meth:`record` calls.
* **Atomic batch writes.**  :meth:`record_many` validates every record
  *before* inserting any, so a single malformed record cannot leave the
  caller's batch half-committed.
* **No hidden I/O.**  Records are constructed from already-validated
  inputs; the memory never re-reads the scene, never re-parses
  configuration, and never calls a provider.
* **Wall-clock-independent timestamps.**  ``recorded_at`` uses
  :func:`time.monotonic` so interval measurements survive system clock
  adjustments.  Absolute values are only meaningful *within* the
  process; for cross-process correlation use the ``request_id`` from
  :class:`~.animation_types.AnimationRequest` (which is not stored here
  to keep records minimal — the caller can join on ``scene_id`` plus
  :meth:`AnimationMemory.recent` ordering).

Non-goals
---------
* **Redo / undo.**  Edit inversion is a domain concern; the memory
  records what happened, not what the caller *could* re-apply.
* **Deduplication.**  Two identical requests produce two records.  If
  the caller needs idempotency, they should key on ``request_id``
  upstream; the memory does not implement a deduplication layer.
* **Compaction.**  When full, the deque evicts the oldest record — it
  does not try to coalesce compatible edits.  Coalescing would require
  reading keyframe data, which the memory deliberately does not retain.
"""

from __future__ import annotations

import time

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

from .utils.animation_errors import *
from .utils.animation_helpers import *
from logs.logger import get_logger  # type: ignore


logger = get_logger("Animation Memory")


@dataclass(frozen=True, slots=True)
class AnimationEditRecord:
    """One successfully committed edit and its source/target revisions.

    Parameters
    ----------
    operation
        Name of the committed edit (``"set_animation"``, ``"add_keyframe"``, …).  Leading and trailing whitespace is
        stripped before storage so records produced via the request envelope and records produced directly from
        :class:`AnimationEditRecord` are stored consistently.
    scene_id
        Identifier of the target scene.  Stripped for the same reason.
    revision_before
        Revision of the scene *before* the commit.  Must be a positive integer.
    revision_after
        Revision of the scene *after* the commit.  Must be strictly greater than ``revision_before`` — a committed edit that did not
        advance the revision is a contract violation, not a valid record.
    recorded_at
        Monotonic timestamp (seconds since an unspecified process-local epoch) captured at record construction.  Used for interval and
        throughput measurements; **not** comparable across processes and **not** a wall-clock time.  Callers needing wall-clock
        correlation should join on ``(scene_id, recorded_at)`` against structured logs that carry ``time.time()`` alongside.

    Notes
    -----
    Instances are frozen and hashable, so records can be used as dictionary keys, set members, and equality comparisons without
    defensive copying.  Validation happens in ``__post_init__`` — an :class:`AnimationEditRecord` is a *witness* that the record is
    structurally valid.
    """

    operation: str
    scene_id: str
    revision_before: int
    revision_after: int
    recorded_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        # Normalise operation and scene_id so records produced via the
        # request envelope and records built directly are byte-identical.
        ensure_non_empty_string(self.operation, "operation", strip=True)
        object.__setattr__(self, "operation", self.operation.strip())
        ensure_non_empty_string(self.scene_id, "scene_id", strip=True)
        object.__setattr__(self, "scene_id", self.scene_id.strip())

        ensure_positive_integer(self.revision_before, "revision_before")
        ensure_positive_integer(self.revision_after, "revision_after")
        if self.revision_after <= self.revision_before:
            raise AnimationValidationError(
                "revision_after must be greater than revision_before",
                context={
                    "revision_before": self.revision_before,
                    "revision_after": self.revision_after,
                },
            )

        ensure_finite_non_negative(self.recorded_at, "recorded_at")

    @property
    def revision_delta(self) -> int:
        """Net revision advance produced by this edit.

        Always ``>= 1`` by construction; provided for convenience when
        computing aggregate metrics without re-deriving the subtraction
        at every call site.
        """
        return self.revision_after - self.revision_before

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-``dict`` representation for serialisation.

        Used by :meth:`AnimationMemory.to_dicts` and available for
        callers who want to embed individual records in structured log
        payloads or audit trails.  The mapping is fresh on every call —
        mutations to the returned dict cannot affect this record.
        """
        return {
            "operation": self.operation,
            "scene_id": self.scene_id,
            "revision_before": self.revision_before,
            "revision_after": self.revision_after,
            "recorded_at": self.recorded_at,
        }


# _CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "animation_config.yaml" # config loader already has DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "animation_config.yaml"


def _capacity_from_config() -> int:
    """Resolve ``animation.memory.max_recent_edits`` via the existing loader.

    The loader is imported lazily so that constructing
    ``AnimationMemory(capacity=n)`` with an explicit capacity never
    touches the filesystem.  Fail-closed: if the section is missing,
    malformed, or non-positive, this raises
    :class:`AnimationConfigurationError` rather than silently choosing
    a default — a missing memory bound is a configuration error, not a
    state the caller should have to guess about.
    """
    from .utils.config_loader import get_config_section

    section = get_config_section("animation")
    memory_config = section.get("memory")
    if not isinstance(memory_config, dict) or "max_recent_edits" not in memory_config:
        raise AnimationConfigurationError(
            "animation.memory.max_recent_edits is required in animation_config.yaml",
            context={"section": "animation.memory"},
        )
    capacity = memory_config["max_recent_edits"]
    # ``type(...) is int`` also rejects bool, which is not a valid capacity.
    if type(capacity) is not int or capacity <= 0:
        raise AnimationConfigurationError(
            "animation.memory.max_recent_edits must be a positive integer",
            context={
                "received_type": type(capacity).__name__,
                "received": capacity,
            },
        )
    return capacity


class AnimationMemory:
    """Bounded, thread-safe recent-edit history.

    Parameters
    ----------
    capacity
        Maximum number of records retained.  When ``None`` (default),
        the value is read from ``animation.memory.max_recent_edits`` in
        ``animation_config.yaml``.  Passing an explicit ``capacity``
        bypasses configuration loading entirely, which keeps tests
        filesystem-free.

    Raises
    ------
    AnimationConfigurationError
        If ``capacity`` is ``None`` and the configuration is missing,
        malformed, or non-positive — or if an explicit ``capacity`` is
        not a positive integer.

    Threading
    ---------
    A single :class:`threading.RLock` guards the internal deque.  All
    public methods acquire it; :meth:`recent` and :meth:`__iter__`
    snapshot under the lock and yield outside it, so iterating while
    another thread records never blocks for the duration of iteration
    and never raises "deque mutated during iteration".

    Mutation semantics
    ------------------
    * :meth:`record` appends one record.
    * :meth:`record_many` appends a batch atomically: either every
      record in the batch is inserted, or none is.
    * :meth:`clear` drops every record.  There is no partial deletion;
      callers who need scene-scoped cleanup should construct a fresh
      memory or query via :meth:`recent` with a ``scene_id`` filter.
    """

    def __init__(self, capacity: int | None = None) -> None:
        if capacity is None:
            capacity = _capacity_from_config()
        if type(capacity) is not int or capacity <= 0:
            raise AnimationConfigurationError(
                "animation memory capacity must be a positive integer",
                context={
                    "received_type": type(capacity).__name__,
                    "received": capacity,
                },
            )
        self._capacity: int = capacity
        self._records: deque[AnimationEditRecord] = deque(maxlen=capacity)
        self._lock = RLock()
        logger.debug("animation memory initialized | capacity=%d", capacity)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def capacity(self) -> int:
        """Maximum number of records retained.  Fixed at construction."""
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, AnimationEditRecord):
            return False
        with self._lock:
            return item in self._records

    def __iter__(self) -> Iterator[AnimationEditRecord]:
        """Iterate over a snapshot of the records in chronological order.

        The snapshot is taken under the lock; iteration itself does not
        hold the lock, so a slow consumer does not block producers.
        Because records are immutable, the snapshot is safe to retain.
        """
        with self._lock:
            snapshot = tuple(self._records)
        return iter(snapshot)

    def __repr__(self) -> str:
        return (
            f"AnimationMemory(capacity={self._capacity}, "
            f"size={len(self)})"
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(self, item: AnimationEditRecord) -> None:
        """Append one committed edit record.

        Raises
        ------
        AnimationValidationError
            If ``item`` is not an :class:`AnimationEditRecord`.  String
            identifiers, dicts, and namedtuples are rejected — the
            caller must construct a validated record explicitly.
        """
        if not isinstance(item, AnimationEditRecord):
            raise AnimationValidationError(
                "animation memory accepts only AnimationEditRecord",
                context={"received_type": type(item).__name__},
            )
        with self._lock:
            self._records.append(item)
        logger.debug(
            "animation edit recorded | operation=%s scene=%s revision=%d",
            item.operation,
            item.scene_id,
            item.revision_after,
        )

    def record_many(self, items: "list[AnimationEditRecord] | tuple[AnimationEditRecord, ...]") -> None:
        """Append a batch atomically.

        The *entire* batch is validated before *any* record is inserted:
        if one item is not an :class:`AnimationEditRecord`, none is
        appended.  This preserves the invariant that the memory never
        holds a partially-applied batch, which matters for callers who
        treat ``record_many`` as an all-or-nothing operation (e.g.
        bulk-importing an audit trail).

        The deque's ``maxlen`` still governs eviction during the insert,
        so a batch larger than ``capacity`` will leave the *last*
        ``capacity`` items in place, matching sequential :meth:`record`
        semantics.
        """
        # Materialise once — the caller might pass a generator and we
        # need to inspect it twice (validate then insert).
        materialised = tuple(items)
        for index, item in enumerate(materialised):
            if not isinstance(item, AnimationEditRecord):
                raise AnimationValidationError(
                    "animation memory accepts only AnimationEditRecord",
                    context={
                        "index": index,
                        "received_type": type(item).__name__,
                    },
                )
        if not materialised:
            return
        with self._lock:
            self._records.extend(materialised)
        logger.debug(
            "animation edit batch recorded | count=%d last_operation=%s",
            len(materialised),
            materialised[-1].operation,
        )

    def clear(self) -> None:
        """Drop every record without touching authoritative scene state."""
        with self._lock:
            self._records.clear()
        logger.debug("animation memory cleared")

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def recent(
        self,
        *,
        scene_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[AnimationEditRecord, ...]:
        """Return records in chronological order, newest last.

        Parameters
        ----------
        scene_id
            When supplied, restrict to records for that scene.  Must be
            a non-empty string (stripped for the comparison, matching
            the storage normalisation applied by
            :class:`AnimationEditRecord`).
        limit
            When supplied, return at most the *last* ``limit`` records
            (i.e. the most recent ones).  Must be a positive integer;
            pass ``None`` for "all matching records".

        Returns
        -------
        tuple[AnimationEditRecord, ...]
            A fresh tuple; safe to retain across subsequent
            :meth:`record` calls without the caller observing mid-flight
            mutations.
        """
        if scene_id is not None:
            ensure_non_empty_string(scene_id, "scene_id", strip=True)
            scene_id = scene_id.strip()
        if limit is not None:
            ensure_positive_integer(limit, "limit")
        with self._lock:
            items = tuple(self._records)
        if scene_id is not None:
            items = tuple(record for record in items if record.scene_id == scene_id)
        if limit is not None:
            items = items[-limit:]
        return items

    def latest(self) -> AnimationEditRecord | None:
        """Return the most recent record, or ``None`` when the memory is empty."""
        with self._lock:
            return self._records[-1] if self._records else None

    def operation_counts(self) -> Counter[str]:
        """Return a :class:`collections.Counter` of operation names.

        Intended for operational dashboards and tests: the counter can
        be diffed against expectations, emitted as a metric, or compared
        across processes without re-parsing the record log.
        """
        with self._lock:
            snapshot = tuple(self._records)
        return Counter(record.operation for record in snapshot)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return all records as plain dicts, in chronological order.

        The result is a fresh list of fresh dicts, safe to serialise
        with :mod:`json` or attach to a structured log entry.  Ordering
        matches :meth:`recent` with no arguments (oldest first).
        """
        with self._lock:
            snapshot = tuple(self._records)
        return [record.to_dict() for record in snapshot]


__all__ = [
    "AnimationEditRecord",
    "AnimationMemory",
]