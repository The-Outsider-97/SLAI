"""
Standard health endpoints/checks for knowledge agent components.
Provides liveness, readiness, and detailed component health checks.
"""

import time
import threading

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..utils.knowledge_errors import RuntimeHealthError
from ..utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Runtime Health")
printer = PrettyPrinter()


class HealthStatus(Enum):
    """Health status of a component."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health report for a single component."""
    component: str
    status: HealthStatus
    message: str = ""
    last_check: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


class RTHealth:
    """
    Runtime Health manager for knowledge agent components.
    Provides liveness, readiness, and component‑specific health checks.
    Thread‑safe for concurrent probe calls.
    """

    def __init__(self):
        self.config = load_global_config()
        self.health_config = get_config_section('runtime_health')
        self._lock = threading.RLock()
        self._component_health: Dict[str, ComponentHealth] = {}
        self._component_checks: Dict[str, Callable[[], ComponentHealth]] = {}
        self._bound_components: Dict[str, Any] = {}  # explicit live references
        self._liveness_check = None
        self._readiness_check = None

        # Configure thresholds
        self.degraded_threshold_sec = self.health_config.get('degraded_after_seconds', 30)
        self.unhealthy_threshold_sec = self.health_config.get('unhealthy_after_seconds', 120)
        self.enable_periodic_checks = self.health_config.get('enable_periodic_checks', True)
        self.check_interval_sec = self.health_config.get('check_interval_seconds', 60)

        # Register built‑in checks
        self._register_default_checks()

        # Start periodic checker if enabled
        self._stop_checker = threading.Event()
        self._checker_thread = None
        if self.enable_periodic_checks:
            self._start_periodic_checker()

    def _register_default_checks(self):
        """Register health checks for core components."""
        self.register_component_check("memory", self._check_memory)
        self.register_component_check("cache", self._check_cache)
        self.register_component_check("rule_engine", self._check_rule_engine)
        self.register_component_check("governor", self._check_governor)
        self.register_component_check("sync", self._check_sync)
        self.register_component_check("action_executor", self._check_action_executor)
        self.register_component_check("ontology", self._check_ontology)

    def register_component_check(self, component: str, check_fn: Callable[[], ComponentHealth]) -> None:
        """Register a health check function for a component."""
        with self._lock:
            self._component_checks[component] = check_fn
            # Initialize with unknown health
            self._component_health[component] = ComponentHealth(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message="Not yet checked"
            )

    def _start_periodic_checker(self):
        """Start background thread to periodically run all checks."""
        def checker_loop():
            while not self._stop_checker.wait(self.check_interval_sec):
                try:
                    self.run_all_checks()
                except Exception as e:
                    logger.error(f"Periodic health check failed: {e}")

        self._checker_thread = threading.Thread(target=checker_loop, daemon=True, name="HealthChecker")
        self._checker_thread.start()

    def run_all_checks(self) -> Dict[str, ComponentHealth]:
        """Run all registered component checks and update internal state."""
        results = {}
        with self._lock:
            for component, check_fn in self._component_checks.items():
                try:
                    health = check_fn()
                    self._component_health[component] = health
                    results[component] = health
                except Exception as e:
                    logger.warning(f"Health check for {component} raised exception: {e}")
                    unhealthy = ComponentHealth(
                        component=component,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check exception: {e}",
                        details={"error": str(e)}
                    )
                    self._component_health[component] = unhealthy
                    results[component] = unhealthy
        return results

    def get_component_health(self, component: str) -> Optional[ComponentHealth]:
        """Get the latest health status of a specific component."""
        with self._lock:
            return self._component_health.get(component)

    def get_all_health(self) -> Dict[str, ComponentHealth]:
        """Get all component health statuses (runs checks if stale)."""
        # Optionally refresh if last check is too old
        now = time.time()
        with self._lock:
            for comp, health in list(self._component_health.items()):
                if now - health.last_check > self.degraded_threshold_sec:
                    # Run check again if stale
                    if comp in self._component_checks:
                        try:
                            new_health = self._component_checks[comp]()
                            self._component_health[comp] = new_health
                        except Exception as e:
                            logger.warning(f"Stale check refresh failed for {comp}: {e}")
            return dict(self._component_health)

    def liveness(self) -> bool:
        """
        Liveness probe: returns True if the knowledge agent is alive.
        Checks that critical components are responsive.
        """
        critical = ["memory", "cache", "rule_engine"]
        for comp in critical:
            health = self.get_component_health(comp)
            if health is None or health.status == HealthStatus.UNHEALTHY:
                logger.warning(f"Liveness failed: {comp} unhealthy")
                return False
        return True

    def readiness(self) -> bool:
        """
        Readiness probe: returns True if the agent is ready to serve requests.
        Requires all core components to be healthy or degraded (not unhealthy).
        """
        core = ["memory", "cache", "rule_engine", "governor"]
        for comp in core:
            health = self.get_component_health(comp)
            if health is None or health.status != HealthStatus.HEALTHY:
                logger.warning(f"Readiness failed: {comp} unhealthy")
                return False
        return True

    def shutdown(self) -> None:
        """Stop the periodic health checker thread."""
        self._stop_checker.set()
        if self._checker_thread and self._checker_thread.is_alive():
            self._checker_thread.join(timeout=2.0)

    # -------------------------------------------------------------------------
    # Built‑in component checks (to be implemented by caller or injected)
    # These are stubs that assume external components are accessible via
    # a global registry or dependency injection. In production, you would
    # inject references to the actual components.
    # -------------------------------------------------------------------------
    def bind_component(self, name: str, instance: Any) -> None:
        """Bind a real live instance; default import-based checks are forbidden."""
        if name not in self._component_checks:
            raise KeyError(f"Unknown health component: {name}")
        with self._lock:
            self._bound_components[name] = instance

    def _probe_bound(self, name: str) -> ComponentHealth:
        with self._lock:
            instance = self._bound_components.get(name)
        if instance is None:
            return ComponentHealth(component=name, status=HealthStatus.UNHEALTHY,
                                   message="No live instance bound; health not established")
        try:
            if name == "memory":
                stats = instance.get_statistics()
                if not isinstance(stats, dict):
                    raise RuntimeError("No valid live memory statistics")
                return ComponentHealth(name, HealthStatus.HEALTHY, "Live memory probe successful",
                                       details={"total_entries": stats.get("total_entries", 0)})
            if name == "cache":
                count = len(instance)
                return ComponentHealth(name, HealthStatus.HEALTHY, "Live cache probe successful",
                                       details={"entries": count})
            if name == "ontology":
                with instance._connect() as conn:
                    conn.execute("SELECT count(*) FROM ontology").fetchone()
                return ComponentHealth(name, HealthStatus.HEALTHY, "Live ontology database probe successful")
            if name == "rule_engine":
                if not hasattr(instance, "get_failure_stats"):
                    raise RuntimeError("No live rule failure metrics")
                instance.get_failure_stats()
                return ComponentHealth(name, HealthStatus.DEGRADED,
                                       "Instance bound, but safety/evaluation readiness not independently proven")
            return ComponentHealth(name, HealthStatus.DEGRADED,
                                   "Instance bound; no side-effect-free independent readiness probe")
        except Exception as exc:
            return ComponentHealth(name, HealthStatus.UNHEALTHY,
                                   message=f"Live probe failed: {type(exc).__name__}: {exc}")

    def _check_memory(self) -> ComponentHealth:
        return self._probe_bound("memory")

    def _check_cache(self) -> ComponentHealth:
        return self._probe_bound("cache")

    def _check_rule_engine(self) -> ComponentHealth:
        return self._probe_bound("rule_engine")

    def _check_governor(self) -> ComponentHealth:
        return self._probe_bound("governor")

    def _check_sync(self) -> ComponentHealth:
        return self._probe_bound("sync")

    def _check_action_executor(self) -> ComponentHealth:
        return self._probe_bound("action_executor")

    def _check_ontology(self) -> ComponentHealth:
        return self._probe_bound("ontology")


if __name__ == "__main__":
    print("\n=== Running Runtime Health ===\n")
    printer.status("Init", "Runtime Health initialized", "success")

    health = RTHealth()
    print(f"{health}")

    print("\n=== Successfully ran the Runtime Health ===\n")