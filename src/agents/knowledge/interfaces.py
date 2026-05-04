"""Typed interfaces for the knowledge subsystem.

These protocols provide stable contracts between orchestration code and
concrete implementations (cache, memory, sync, governance, monitoring, actions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable


@runtime_checkable
class MemoryStore(Protocol):
    """Contract for long-lived or session memory stores."""

    def update(
        self,
        key: str,
        value: Any,
        metadata: Optional[dict] = None,
        context: Optional[dict] = None,
        ttl: Optional[int] = None,
    ) -> None:
        ...

    def recall(
        self,
        key: Optional[str] = None,
        filters: Optional[dict] = None,
        sort_by: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        ...

    def delete(self, key: str) -> None:
        ...

    def clear(self) -> None:
        ...

    def keys(self) -> List[str]:
        ...

    def get_statistics(self) -> Dict[str, Any]:
        ...


@runtime_checkable
class CacheStore(Protocol):
    """Contract for low-latency cache storage."""

    def get(self, key: str) -> Optional[Any]:
        ...

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        ...

    def cleanup_expired(self) -> int:
        ...

    def hash_query(self, query: str) -> str:
        ...


@runtime_checkable
class RuleService(Protocol):
    """Contract for rule loading and inference services."""

    def load_all_sectors(self) -> None:
        ...

    def apply(self, knowledge_base: Dict[str, Any], verbose: bool = False) -> Any:
        ...

    def smart_apply(self, knowledge_base: Dict[str, Any], verbose: bool = False) -> Any:
        ...


@runtime_checkable
class ComplianceService(Protocol):
    """Contract for governance and safety checks."""

    def full_audit(self) -> Dict[str, Any]:
        ...

    def audit_retrieval(self, query: str, results: list, context: dict) -> Dict[str, Any]:
        ...

    def generate_report(self, format_type: str = "json") -> Any:
        ...


@runtime_checkable
class SyncService(Protocol):
    """Contract for synchronization components."""

    def full_sync(self, components: Optional[List[str]] = None) -> Dict[str, int]:
        ...

    def stop_sync(self) -> None:
        ...


@runtime_checkable
class MonitorService(Protocol):
    """Contract for monitoring components."""

    def check_academic_compliance(self) -> None:
        ...

    def verify_data_integrity(self) -> None:
        ...

    def generate_academic_report(self) -> Dict[str, Any]:
        ...

    def stop_monitoring(self) -> None:
        ...


@runtime_checkable
class ActionExecutor(Protocol):
    """Contract for action execution subsystem."""

    def from_knowledge(self, knowledge_batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ...


@dataclass
class OrchestratorHealth:
    """Basic runtime health envelope for subsystem state."""

    orchestrator_started: bool
    components: Dict[str, bool] = field(default_factory=dict)
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    cache_entries: Optional[int] = None
    last_sync_stats: Dict[str, int] = field(default_factory=dict)
    last_audit_summary: Dict[str, Any] = field(default_factory=dict)
