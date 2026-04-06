"""Knowledge subsystem orchestrator.

Centralizes component creation, lifecycle operations, and cross-component calls.
This module is intentionally thin and relies on protocol contracts defined in
`src.agents.knowledge.interfaces`.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence

from src.agents.knowledge.governor import Governor
from src.agents.knowledge.interfaces import (
    ActionExecutor,
    CacheStore,
    ComplianceService,
    MemoryStore,
    MonitorService,
    OrchestratorHealth,
    RuleService,
    SyncService,
)
from src.agents.knowledge.knowledge_cache import KnowledgeCache
from src.agents.knowledge.knowledge_memory import KnowledgeMemory
from src.agents.knowledge.knowledge_monitor import KnowledgeMonitor
from src.agents.knowledge.knowledge_sync import KnowledgeSynchronizer
from src.agents.knowledge.perform_action import PerformAction
from src.agents.knowledge.utils.rule_engine import RuleEngine
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("KnowledgeOrchestrator")
printer = PrettyPrinter


class KnowledgeOrchestrator:
    """Production-oriented orchestrator for knowledge subsystem components."""

    def __init__(
        self,
        agent: Optional[Any] = None,
        memory: Optional[MemoryStore] = None,
        cache: Optional[CacheStore] = None,
        rule_engine: Optional[RuleService] = None,
        governor: Optional[ComplianceService] = None,
        synchronizer: Optional[SyncService] = None,
        monitor: Optional[MonitorService] = None,
        action_executor: Optional[ActionExecutor] = None,
        lazy_start: bool = True,
    ) -> None:
        self.agent = agent

        # Dependency injection with practical defaults
        self.memory: MemoryStore = memory or KnowledgeMemory()
        self.cache: CacheStore = cache or KnowledgeCache()
        self.rule_engine: RuleService = rule_engine or RuleEngine()
        self.governor: ComplianceService = governor or Governor(knowledge_agent=agent)
        self.synchronizer: SyncService = synchronizer or KnowledgeSynchronizer()
        self.monitor: MonitorService = monitor or KnowledgeMonitor(agent=agent)
        self.action_executor: ActionExecutor = action_executor or PerformAction()

        self._started = False
        self._started_at: Optional[float] = None
        self._last_sync_stats: Dict[str, int] = {}
        self._last_audit: Dict[str, Any] = {}

        if not lazy_start:
            self.start()

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        """Mark orchestrator as started.

        Note: current subsystem components self-start internal background threads
        during their own initialization (based on config). This method is still
        useful as an explicit lifecycle signal for callers.
        """
        if self._started:
            logger.debug("KnowledgeOrchestrator.start() called while already started")
            return

        self._started = True
        self._started_at = time.time()
        logger.info("KnowledgeOrchestrator started")

    def stop(self) -> None:
        """Gracefully stop components that expose lifecycle shutdown hooks."""
        if not self._started:
            return

        # Stop monitor thread if available
        if hasattr(self.monitor, "stop_monitoring"):
            self.monitor.stop_monitoring()

        # Stop synchronizer thread if available
        if hasattr(self.synchronizer, "stop_sync"):
            self.synchronizer.stop_sync()

        # Persist memory if implementation exposes shutdown
        if hasattr(self.memory, "shutdown"):
            self.memory.shutdown()

        self._started = False
        logger.info("KnowledgeOrchestrator stopped")

    def sync(self, components: Optional[Sequence[str]] = None) -> Dict[str, int]:
        """Run an explicit synchronization pass."""
        component_list = list(components) if components else None
        stats = self.synchronizer.full_sync(component_list)
        self._last_sync_stats = stats
        return stats

    def audit(self) -> Dict[str, Any]:
        """Run an explicit governance audit."""
        report = self.governor.full_audit()
        self._last_audit = report
        return report

    def monitor_once(self) -> Dict[str, Any]:
        """Run one monitoring cycle without waiting on background intervals."""
        self.monitor.check_academic_compliance()
        self.monitor.verify_data_integrity()
        return self.monitor.generate_academic_report()

    def execute_actions(self, docs: Sequence[Dict[str, Any]]) -> Any:
        """Execute action pipeline for extracted action directives."""
        return self.action_executor.from_knowledge(docs)

    def health(self) -> Dict[str, Any]:
        """Return a normalized health/status payload."""
        components = {
            "memory": self.memory is not None,
            "cache": self.cache is not None,
            "rule_engine": self.rule_engine is not None,
            "governor": self.governor is not None,
            "synchronizer": self.synchronizer is not None,
            "monitor": self.monitor is not None,
            "action_executor": self.action_executor is not None,
        }

        memory_stats = {}
        if hasattr(self.memory, "get_statistics"):
            try:
                memory_stats = self.memory.get_statistics()
            except Exception as exc:
                memory_stats = {"error": str(exc)}

        cache_entries = None
        if hasattr(self.cache, "__len__"):
            try:
                cache_entries = len(self.cache)  # type: ignore[arg-type]
            except Exception:
                cache_entries = None

        last_audit_summary: Dict[str, Any] = {}
        if self._last_audit:
            last_audit_summary = {
                "timestamp": self._last_audit.get("timestamp"),
                "violations": len(self._last_audit.get("violations", [])),
            }

        health = OrchestratorHealth(
            orchestrator_started=self._started,
            components=components,
            memory_stats=memory_stats,
            cache_entries=cache_entries,
            last_sync_stats=self._last_sync_stats,
            last_audit_summary=last_audit_summary,
        )
        payload = asdict(health)
        if self._started_at:
            payload["uptime_seconds"] = round(time.time() - self._started_at, 2)
        else:
            payload["uptime_seconds"] = 0.0
        return payload


__all__ = ["KnowledgeOrchestrator"]


if __name__ == "__main__":
    print("\n=== Running Knowledge Orchestrator ===\n")
    printer.status("Init", "Knowledge Orchestrator initialized", "success")

    orchestrator = KnowledgeOrchestrator(lazy_start=False)
    printer.pretty("Health", orchestrator.health(), "info")

    orchestrator.stop()
    print("\n=== Knowledge Orchestrator Test Completed ===\n")