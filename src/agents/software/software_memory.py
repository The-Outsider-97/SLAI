"""Private working memory of the Software subsystem.

Holds only *derived, recomputable* state: bounded LRU caches of parsed models,
snapshots, repository models and graph/index bundles, plus a ledger of the
patch proposals the subsystem has issued (with lineage, so a refactoring can be
built on the result of an earlier one).  Everything here is process-local and
can be dropped at any time without losing correctness; cross-agent state lives
in SharedMemory, which only the agent talks to.
"""

from __future__ import annotations

__version__ = "2.4.0"

import threading

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from .models.repository_model import RepositoryModel, RepositorySnapshot
from .software_types import PatchProposal, RefactoringKind, Serializable
from .utils.software_errors import SoftwareMemoryError
from .utils.software_helpers import opt_int, utc_now_iso
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Memory")

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(MutableMapping[K, V], Generic[K, V]):
    """Thread-safe least-recently-used mapping with hit/miss accounting."""

    def __init__(self, capacity: int, name: str = "cache") -> None:
        if capacity < 1:
            raise SoftwareMemoryError("cache capacity must be positive", context={"cache": name, "capacity": capacity})
        self.capacity = capacity
        self.name = name
        self._data: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = self.misses = self.evictions = 0

    def __getitem__(self, key: K) -> V:
        with self._lock:
            value = self._data[key]
            self._data.move_to_end(key)
            return value

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)
                self.evictions += 1

    def __delitem__(self, key: K) -> None:
        with self._lock:
            del self._data[key]

    def __iter__(self) -> Iterator[K]:
        with self._lock:
            return iter(list(self._data))

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: K, default: Any = None) -> Any:  # type: ignore[override]
        with self._lock:
            if key in self._data:
                self.hits += 1
                self._data.move_to_end(key)
                return self._data[key]
            self.misses += 1
            return default

    def get_or_build(self, key: K, factory: Callable[[], V]) -> V:
        """Return the cached value or build, store and return it (builds are not held under the lock)."""

        found = self.get(key)
        if found is not None:
            return found
        value = factory()
        self[key] = value
        return value

    def stats(self) -> dict[str, Any]:
        return {"entries": len(self._data), "capacity": self.capacity, "hits": self.hits, "misses": self.misses, "evictions": self.evictions}


@dataclass(frozen=True, slots=True)
class LedgerEntry(Serializable):
    patch_id: str
    kind: RefactoringKind
    intent: str
    base_snapshot_digest: str
    result_snapshot_digest: str
    parent_patch_id: Optional[str]
    recorded_at: str
    confidence: float
    files: int
    additions: int
    deletions: int


class SoftwareMemory:
    """Caches and the proposal ledger shared by the Software handlers."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        data = config or {}
        self.models: LRUCache[tuple, Any] = LRUCache(opt_int(data, "model_cache_entries", 8192, minimum=1), "models")
        self.snapshots: LRUCache[str, RepositorySnapshot] = LRUCache(opt_int(data, "snapshot_entries", 16, minimum=1), "snapshots")
        self.repositories: LRUCache[tuple, RepositoryModel] = LRUCache(opt_int(data, "repository_entries", 8, minimum=1), "repositories")
        self.derived: LRUCache[tuple, Any] = LRUCache(opt_int(data, "derived_entries", 24, minimum=1), "derived")
        self._ledger_capacity = opt_int(data, "ledger_entries", 500, minimum=1)
        self._ledger: OrderedDict[str, tuple[PatchProposal, LedgerEntry]] = OrderedDict()
        self._children: dict[str, list[str]] = {}
        self._lock = threading.RLock()
        self.recorded = 0

    # -- snapshots ----------------------------------------------------------------------
    def remember_snapshot(self, snapshot: RepositorySnapshot) -> RepositorySnapshot:
        self.snapshots[snapshot.digest] = snapshot
        return snapshot

    def snapshot(self, digest: str) -> RepositorySnapshot:
        found = self.snapshots.get(digest)
        if found is None:
            raise SoftwareMemoryError("snapshot is not available (evicted or never seen); supply the files again", context={"digest": digest})
        return found

    # -- proposal ledger -------------------------------------------------------------------
    def record(self, proposal: PatchProposal, result_snapshot: RepositorySnapshot) -> LedgerEntry:
        with self._lock:
            self.remember_snapshot(result_snapshot)
            entry = LedgerEntry(
                patch_id=proposal.patch_id,
                kind=proposal.kind,
                intent=proposal.intent,
                base_snapshot_digest=proposal.base_snapshot_digest,
                result_snapshot_digest=proposal.result_snapshot_digest,
                parent_patch_id=proposal.parent_patch_id,
                recorded_at=utc_now_iso(),
                confidence=proposal.confidence,
                files=len(proposal.files),
                additions=int(proposal.stats.get("additions", 0)),
                deletions=int(proposal.stats.get("deletions", 0)),
            )
            self._ledger[proposal.patch_id] = (proposal, entry)
            self._ledger.move_to_end(proposal.patch_id)
            if proposal.parent_patch_id:
                siblings = self._children.setdefault(proposal.parent_patch_id, [])
                if proposal.patch_id not in siblings:
                    siblings.append(proposal.patch_id)
            while len(self._ledger) > self._ledger_capacity:
                dropped, _ = self._ledger.popitem(last=False)
                self._children.pop(dropped, None)
            self.recorded += 1
            return entry

    def proposal(self, patch_id: str) -> PatchProposal:
        with self._lock:
            item = self._ledger.get(patch_id)
        if item is None:
            raise SoftwareMemoryError("patch proposal is not in the ledger", context={"patch_id": patch_id})
        return item[0]

    def has_proposal(self, patch_id: str) -> bool:
        return patch_id in self._ledger

    def result_snapshot(self, patch_id: str) -> RepositorySnapshot:
        """The snapshot a proposal *would* produce (still unapplied in the real world)."""

        return self.snapshot(self.proposal(patch_id).result_snapshot_digest)

    def lineage(self, patch_id: str) -> tuple[LedgerEntry, ...]:
        """Ancestors of ``patch_id`` (oldest first) ending with the proposal itself."""

        chain: list[LedgerEntry] = []
        seen: set[str] = set()
        current: Optional[str] = patch_id
        with self._lock:
            while current and current not in seen and current in self._ledger:
                seen.add(current)
                entry = self._ledger[current][1]
                chain.append(entry)
                current = entry.parent_patch_id
        return tuple(reversed(chain))

    def children(self, patch_id: str) -> tuple[str, ...]:
        return tuple(self._children.get(patch_id, ()))

    def recent(self, limit: int = 20) -> tuple[LedgerEntry, ...]:
        with self._lock:
            entries = [item[1] for item in self._ledger.values()]
        return tuple(reversed(entries[-max(1, limit):]))

    # -- housekeeping ------------------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "models": self.models.stats(),
            "snapshots": self.snapshots.stats(),
            "repositories": self.repositories.stats(),
            "derived": self.derived.stats(),
            "ledger": {"entries": len(self._ledger), "capacity": self._ledger_capacity, "recorded": self.recorded},
        }

    def clear(self, *, ledger: bool = False) -> None:
        for cache in (self.models, self.snapshots, self.repositories, self.derived):
            cache.clear()
        if ledger:
            with self._lock:
                self._ledger.clear()
                self._children.clear()


__all__ = ["LRUCache", "LedgerEntry", "SoftwareMemory"]
