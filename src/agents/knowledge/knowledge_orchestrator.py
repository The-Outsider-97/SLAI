"""Knowledge subsystem orchestrator.

Centralizes component creation, lifecycle operations, and cross-component calls.
This module is intentionally thin and relies on protocol contracts defined in
`src.agents.knowledge.interfaces`.
"""

from __future__ import annotations

import time

from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .governor import Governor
from .knowledge_cache import KnowledgeCache
from .knowledge_memory import KnowledgeMemory
from .knowledge_monitor import KnowledgeMonitor
from .knowledge_sync import KnowledgeSynchronizer
from .perform_action import PerformAction
from .utils.knowledge_errors import *
from .utils.knowledge_helpers import *
from .modules.rule_engine import RuleEngine
from .modules.interfaces import *  # type: ignore
from .modules.inference_result import InferenceResult, InferenceTrace
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("KnowledgeOrchestrator")
printer = PrettyPrinter()


class KnowledgeOrchestrator:
    """Production-oriented orchestrator for knowledge subsystem components."""

    def __init__(self, agent: Optional[Any] = None,
                memory: Optional[MemoryStore] = None,
                cache: Optional[CacheStore] = None,
                rule_engine: Optional[RuleService] = None,
                governor: Optional[ComplianceService] = None,
                synchronizer: Optional[SyncService] = None,
                monitor: Optional[MonitorService] = None,
                action_executor: Optional[ActionExecutor] = None,
                lazy_start: bool = True,
                create_governor= None,
                manage_memory=None,
                manage_synchronizer=None,
                manage_monitor=None,) -> None:
        self.agent = agent

        # Dependency injection with practical defaults
        self.memory = memory or KnowledgeMemory()
        self.cache = cache or KnowledgeCache()
        self.rule_engine = rule_engine or RuleEngine()
        
        self.governor = governor or Governor(
            knowledge_agent=agent,
            knowledge_memory=self.memory,
            rule_engine=self.rule_engine,
        )
        
        self.synchronizer = synchronizer or KnowledgeSynchronizer(
            knowledge_memory=self.memory,
            rule_engine=self.rule_engine,
        )
        
        self.action_executor = action_executor or PerformAction(
            knowledge_memory=self.memory,
        )
        self.monitor = monitor or KnowledgeMonitor(
            agent=agent,
            cache=self.cache,
            rule_engine=self.rule_engine,
            governor=self.governor,
            action_executor=self.action_executor,
        )

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
            self.memory.shutdown()  # type: ignore[attr-defined]

        self._started = False
        logger.info("KnowledgeOrchestrator stopped")

    def infer(
        self,
        knowledge_base: Mapping[Any, Any],
        *,
        sector: Optional[str] = None,
        smart: bool = True,
        trace: bool = False,
        persist: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        """Run Knowledge inference through the configured RuleService.
    
        Persistence is optional and remains an orchestration responsibility.
    
        When persistence is requested, tracing is automatically enabled so
        persisted inferred facts retain rule provenance.
        """
        result = self.rule_engine.infer(
            knowledge_base,
            sector=sector,
            smart=smart,
            trace=trace or persist,
        )
    
        if persist:
            self._persist_inference_result(
                result,
                context=context,
            )
    
        return result

    def _persist_inference_result(
        self,
        result: InferenceResult,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist inferred facts using the configured MemoryStore."""
    
        traces_by_fact: Dict[Any, List[Dict[str, Any]]] = {}
    
        for trace in result.traces:
            traces_by_fact.setdefault(
                trace.fact,
                [],
            ).append(
                {
                    "confidence": trace.confidence,
                    "rule": trace.rule,
                    "source": trace.source,
                    "sector": trace.sector,
                }
            )
    
        for fact, confidence in result.facts.items():
            metadata: Dict[str, Any] = {
                "type": "inferred_fact",
                "confidence": float(confidence),
                "source": "rule_engine",
                "sector": result.sector,
            }
    
            provenance = traces_by_fact.get(fact)
    
            if provenance:
                metadata["provenance"] = provenance
    
            self.memory.update(
                key=f"inferred:{safe_hash(fact)}",
                value=fact,
                metadata=metadata,
                context=context,
            )

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
        return self.action_executor.from_knowledge(list(docs))

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
