"""
Resource Monitor – production-grade resource discovery, telemetry, and reservation.

This module provides the resource-facing runtime contract used by SafetyPlanning,
PlanningCalculations, and PlanningAgent.  It separates observed cluster capacity
from planner-owned reservations, keeps discovery resilient, and exposes typed
resource acquisition/release methods with structured PlanningError subclasses.

Design notes
------------
- The monitor does not mutate total cluster capacity when tasks reserve resources.
  Reservations are tracked separately in ``self.allocations`` and subtracted only
  when callers ask for available resources.
- Node-level telemetry allocations are stored separately from planner task
  reservations to avoid double-counting.
- All mutable state is guarded by an ``RLock`` because acquisition calls query
  availability while already holding the monitor lock.
- Local package imports are kept direct and unwrapped.  Runtime failures are
  surfaced through structured planning errors or conservative fallbacks.
"""

from __future__ import annotations

import copy
import socket
import threading
import time
import psutil  # type: ignore
import requests  # type: ignore

try:
    import GPUtil  # type: ignore
except ImportError:  # Optional GPU telemetry; CPU/RAM monitoring still works.
    GPUtil = None  # type: ignore

from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple
from requests.exceptions import RequestException  # type: ignore

from .config_loader import get_config_section, load_global_config
from .planning_errors import *
from .planning_helpers import *
from ..planning_types import ClusterResources, ResourceProfile
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Resource Monitor")
printer = PrettyPrinter()

ResourceMetrics = Dict[str, Dict[str, float]]
NodeResourceData = Dict[str, Any]

def _cluster_resources_cls():
    return ClusterResources

def _resource_profile_cls():
    return ResourceProfile

class ResourceMonitor:
    """
    Real-time cluster resource monitor with explicit reservation accounting.

    Public compatibility surface
    ----------------------------
    - ``get_available_resources()`` returns a ``ClusterResources`` snapshot after
      subtracting planner-owned allocations.
    - ``acquire_resources(...)`` reserves resources for a task or raises
      ``ResourceAcquisitionError``.
    - ``release_resources(...)`` releases a task reservation.
    - ``allocate_resources(...)`` remains as a backward-compatible alias for
      legacy code, but callers should prefer ``acquire_resources`` with task IDs.
    - ``_update_resource_map()`` is preserved for existing internal call sites.
    """

    _global_last_logged_cluster_signature: Optional[Tuple[Any, ...]] = None
    _global_last_log_ts: float = 0.0

    def __init__(self) -> None:
        self.config = load_global_config()
        self.resource_config = get_config_section("service_discovery", config=self.config, default={})
        self.monitor_config = get_config_section("cluster_monitoring", config=self.config, default={})
        self.safety_config = get_config_section("safety_margins", config=self.config, default={})
        self._validate_config()

        # Service discovery config – kept as attributes for compatibility with
        # the existing implementation and external callers.
        self.skip_localhost_http = bool(self.resource_config.get("skip_localhost_http", True))
        self.static_nodes = list(self.resource_config.get("static_nodes") or ["localhost"])
        self.consul_url = str(self.resource_config.get("consul_url", "http://localhost:8500")).rstrip("/")
        self.k8s_token = str(self.resource_config.get("k8s_token", ""))
        self.node_port = int(self.resource_config.get("node_port", 9100))
        self.k8s_api = str(self.resource_config.get("k8s_api", "https://kubernetes.default.svc")).rstrip("/")
        self.mode = str(self.resource_config.get("mode", "static")).lower()

        # Monitoring config – numeric values are normalised once.
        self.update_interval = float(self.monitor_config.get("update_interval", 5.0))
        self.node_query_timeout = float(self.monitor_config.get("node_query_timeout", 2.0))
        self.node_cache_ttl = float(self.monitor_config.get("node_cache_ttl", self.update_interval * 2.0))
        self.discovery_poll_interval = float(self.monitor_config.get("discovery_poll_interval", 300.0))
        self.history_limit = int(self.monitor_config.get("history_limit", 1000))
        self.resource_log_interval = float(self.monitor_config.get("resource_log_interval", 60.0))

        resource_buffers = dict(self.safety_config.get("resource_buffers", {}))
        self.safety_buffers: Dict[str, float] = {
            "gpu": float(resource_buffers.get("gpu", 0.15)),
            "ram": float(resource_buffers.get("ram", 0.20)),
        }
        self.reserved_hardware = set(resource_buffers.get("specialized_hardware", []) or [])

        # Observed cluster capacity.  Task reservations are not stored here.
        self.cluster_resources = _cluster_resources_cls()()
        self.node_allocations: Dict[str, ResourceProfile] = {}
        self.allocations: Dict[str, ResourceProfile] = {}
        self.resource_graph: Dict[str, NodeResourceData] = {}
        self._node_cache: Dict[str, Dict[str, Any]] = {}
        self._last_discovery_ts = 0.0
        self._discovered_nodes: List[str] = []

        # Metric state used by PlanningMonitor / SafetyPlanning style callers.
        self.gpu_utilization: Dict[str, float] = {}
        self.ram_utilization: Dict[str, float] = {}
        self.cpu_utilization: Dict[str, float] = {}
        self.network_utilization: Dict[str, float] = {}
        self.storage_utilization: Dict[str, float] = {}
        self.thermal_readings: Dict[str, float] = {}
        self.power_consumption: Dict[str, float] = {}
        self.hardware_status: Dict[str, str] = {}
        self.resource_history: Deque[Dict[str, Any]] = deque(maxlen=self.history_limit)
        self.alert_thresholds: Dict[str, float] = self._load_thresholds()
        self.last_update: float = 0.0

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_error: Optional[Dict[str, Any]] = None

        # Prime the map once so get_available_resources has a meaningful result
        # even when auto-start is disabled.
        self.refresh_once()
        if bool(self.monitor_config.get("auto_start", True)):
            self.start_monitoring()

        logger.info("Resource Monitor successfully initialized")

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def start_monitoring(self) -> None:
        """Start the background refresh loop if it is not already running."""
        with self._lock:
            if self._monitor_thread and self._monitor_thread.is_alive():
                return
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="ResourceMonitorLoop",
                daemon=True,
            )
            self._monitor_thread.start()
            logger.debug("Resource monitor background loop started")

    def stop_monitoring(self, timeout: float = 2.0) -> None:
        """Stop the background refresh loop."""
        self._stop_event.set()
        thread = self._monitor_thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        logger.debug("Resource monitor background loop stopped")

    def _init_monitoring_thread(self) -> None:
        """Backward-compatible alias for legacy call sites."""
        self.start_monitoring()

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._update_resource_map()
            except Exception as exc:
                self._record_error("background_refresh", exc)
                logger.error("Resource monitor refresh failed: %s", exc)
            self._stop_event.wait(self.update_interval)

    def refresh_once(self) -> ClusterResources:
        """Synchronously refresh and return the latest available resources."""
        self._update_resource_map()
        return self.get_available_resources()

    # ------------------------------------------------------------------
    # Reservation API
    # ------------------------------------------------------------------
    def acquire_resources(self, requirements: ResourceProfile, task_id: Optional[str] = None) -> bool:
        """
        Reserve resources for ``task_id``.

        The method validates RAM/GPU through ``check_resource_feasibility`` from
        planning_helpers and performs hardware checks here because hardware is a
        set-membership constraint rather than a numeric margin.
        """
        self._validate_resource_profile(requirements, "requirements")
        reservation_id = self._normalise_task_id(task_id)

        with self._lock:
            if reservation_id in self.allocations:
                raise ResourceAcquisitionError(
                    f"Resources already acquired for task {reservation_id!r}",
                    resource_type="reservation",
                    requested=reservation_id,
                    available=list(self.allocations.keys()),
                    task_id=reservation_id,
                    retry_after_seconds=self.update_interval,
                )

            available = self._available_locked(include_allocations=False)
            numeric_requirements = {
                "gpu": float(requirements.gpu),
                "ram": float(requirements.ram),
            }
            numeric_available = {
                "gpu": float(available.gpu_total),
                "ram": float(available.ram_total),
            }
            try:
                check_resource_feasibility(
                    numeric_requirements,
                    numeric_available,
                    safety_buffers=self.safety_buffers,
                    task_id=reservation_id,
                )
            except ResourceViolation as exc:
                raise ResourceAcquisitionError(
                    f"Insufficient resources for task {reservation_id}: {exc}",
                    resource_type=exc.resource_type,
                    requested=exc.requested,
                    available=exc.available,
                    task_id=reservation_id,
                    retry_after_seconds=self.update_interval,
                ) from exc

            required_hw = set(requirements.specialized_hardware or [])
            available_hw = set(available.specialized_hardware_available or []) - self.reserved_hardware
            missing_hw = sorted(required_hw - available_hw)
            if missing_hw:
                raise ResourceAcquisitionError(
                    f"Missing specialized hardware for task {reservation_id}: {missing_hw}",
                    resource_type="specialized_hardware",
                    requested=sorted(required_hw),
                    available=sorted(available_hw),
                    task_id=reservation_id,
                    retry_after_seconds=self.update_interval,
                )

            self.allocations[reservation_id] = self._copy_profile(requirements)
            logger.info(
                "Reserved resources for %s: gpu=%s ram=%s hw=%s",
                reservation_id,
                requirements.gpu,
                requirements.ram,
                requirements.specialized_hardware,
            )
            return True

    def release_resources(self, task_id: str) -> bool:
        """Release resources reserved for ``task_id``; return True when found."""
        require_type(task_id, str, "task_id")
        with self._lock:
            if task_id not in self.allocations:
                logger.debug("No resource allocation found for task %s", task_id)
                return False
            released = self.allocations.pop(task_id)
            logger.info(
                "Released resources for %s: gpu=%s ram=%s hw=%s",
                task_id,
                released.gpu,
                released.ram,
                released.specialized_hardware,
            )
            return True

    def release_all_resources(self) -> int:
        """Release all planner-owned reservations and return the count."""
        with self._lock:
            count = len(self.allocations)
            self.allocations.clear()
            logger.info("Released all planner-owned resource reservations (%d)", count)
            return count

    def allocate_resources(self, requirements: ResourceProfile, task_id: Optional[str] = None) -> bool:
        """
        Backward-compatible alias for legacy code.

        Older code called ``allocate_resources(requirements)`` and directly
        mutated capacity.  The production implementation reserves instead.  When
        no task ID is supplied, a deterministic internal ID is created.
        """
        return self.acquire_resources(requirements, task_id=task_id)

    # ------------------------------------------------------------------
    # Resource state API
    # ------------------------------------------------------------------
    def get_available_resources(self) -> ClusterResources:
        """Return current cluster resources after subtracting planner allocations."""
        with self._lock:
            return self._available_locked(include_allocations=True)

    def get_cluster_resources(self) -> ClusterResources:
        """Return observed raw cluster capacity before planner reservations."""
        with self._lock:
            snapshot = copy.deepcopy(self.cluster_resources)
            snapshot.current_allocations = dict(self.node_allocations)
            return snapshot

    def get_allocation_snapshot(self) -> Dict[str, ResourceProfile]:
        """Return a copy of current planner-owned reservations."""
        with self._lock:
            return {task_id: self._copy_profile(profile) for task_id, profile in self.allocations.items()}

    def get_resource_report(self) -> Dict[str, Any]:
        """Return a serialisable diagnostic snapshot for logs, APIs, and tests."""
        with self._lock:
            raw = self.get_cluster_resources()
            available = self.get_available_resources()
            return {
                "timestamp": time.time(),
                "mode": self.mode,
                "nodes": list(self._discovered_nodes),
                "raw_capacity": self._cluster_to_dict(raw),
                "available": self._cluster_to_dict(available),
                "planner_allocations": {
                    task_id: self._profile_to_dict(profile)
                    for task_id, profile in self.allocations.items()
                },
                "node_allocations": {
                    node_id: self._profile_to_dict(profile)
                    for node_id, profile in self.node_allocations.items()
                },
                "violations": self.check_violations(),
                "last_error": dict(self._last_error or {}),
            }

    def _available_locked(self, *, include_allocations: bool) -> ClusterResources:
        gpu_total = float(getattr(self.cluster_resources, "gpu_total", 0.0) or 0.0)
        ram_total = float(getattr(self.cluster_resources, "ram_total", 0.0) or 0.0)
        hardware = list(getattr(self.cluster_resources, "specialized_hardware_available", []) or [])

        allocations = self.allocations if include_allocations else {}
        allocated_gpu = sum(float(profile.gpu) for profile in allocations.values())
        allocated_ram = sum(float(profile.ram) for profile in allocations.values())
        allocated_hw = {
            hw
            for profile in allocations.values()
            for hw in (profile.specialized_hardware or [])
        }

        return _cluster_resources_cls()(
            gpu_total=int(round(max(0.0, gpu_total - allocated_gpu))),
            ram_total=int(round(max(0.0, ram_total - allocated_ram))),
            specialized_hardware_available=[hw for hw in hardware if hw not in allocated_hw],
            current_allocations={task_id: self._copy_profile(profile) for task_id, profile in allocations.items()},
        )

    # ------------------------------------------------------------------
    # Metrics and threshold checks
    # ------------------------------------------------------------------
    def update_metrics(self, metrics: Dict[str, Any], *, force: bool = False) -> None:
        """Update resource metrics from an external monitoring system."""
        require_type(metrics, dict, "metrics")
        now = time.time()
        if not force and self.last_update and (now - self.last_update) < self.update_interval:
            return

        with self._lock:
            self.gpu_utilization = self._coerce_metric_map(metrics.get("gpu", {}))
            self.ram_utilization = self._coerce_metric_map(metrics.get("ram", metrics.get("memory", {})))
            self.cpu_utilization = self._coerce_metric_map(metrics.get("cpu", {}))
            self.network_utilization = self._coerce_metric_map(metrics.get("network", {}))
            self.storage_utilization = self._coerce_metric_map(metrics.get("storage", {}))
            self.thermal_readings = self._coerce_metric_map(metrics.get("temperature", {}))
            self.power_consumption = self._coerce_metric_map(metrics.get("power", {}))
            self.hardware_status = dict(metrics.get("status", {}) or {})
            self.last_update = now
            self.resource_history.append({"timestamp": now, "metrics": copy.deepcopy(metrics)})

    def check_violations(self) -> List[str]:
        """Return human-readable threshold violations for current metric maps."""
        violations: List[str] = []
        metric_sources = {
            "gpu": self.gpu_utilization,
            "ram": self.ram_utilization,
            "memory": self.ram_utilization,
            "cpu": self.cpu_utilization,
            "storage": self.storage_utilization,
            "temperature": self.thermal_readings,
        }
        for resource, threshold in self.alert_thresholds.items():
            current = metric_sources.get(resource, {})
            for device, usage in current.items():
                if usage > threshold:
                    violations.append(
                        f"{resource} violation on {device}: {usage:.3f} > {threshold:.3f}"
                    )
        return violations

    # ------------------------------------------------------------------
    # Discovery and node querying
    # ------------------------------------------------------------------
    def _discover_cluster_nodes(self) -> List[str]:
        """Discover nodes through the configured service discovery backend."""
        now = time.time()
        if self._discovered_nodes and (now - self._last_discovery_ts) < self.discovery_poll_interval:
            return list(self._discovered_nodes)

        try:
            if self.mode == "consul":
                nodes = self._query_consul_cluster()
            elif self.mode in {"k8s", "kubernetes"}:
                nodes = self._query_kubernetes_cluster()
            elif self.mode in {"local", "localhost"}:
                nodes = ["localhost"]
            else:
                nodes = list(self.static_nodes or ["localhost"])
        except Exception as exc:
            self._record_error("discovery", exc)
            logger.error("Discovery failed: %s", exc)
            nodes = ["localhost"]

        clean_nodes = self._normalise_nodes(nodes)
        self._discovered_nodes = clean_nodes or ["localhost"]
        self._last_discovery_ts = now
        return list(self._discovered_nodes)

    def _query_consul_cluster(self) -> List[str]:
        """Query Consul service discovery for node names."""
        response = requests.get(
            f"{self.consul_url}/v1/catalog/nodes",
            timeout=self.node_query_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [str(node.get("Node")) for node in payload if node.get("Node")]

    def _query_kubernetes_cluster(self) -> List[str]:
        """Query the Kubernetes API for schedulable node names."""
        headers = {"Authorization": f"Bearer {self.k8s_token}"} if self.k8s_token else {}
        response = requests.get(
            f"{self.k8s_api}/api/v1/nodes",
            headers=headers,
            timeout=self.node_query_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [
            str(item.get("metadata", {}).get("name"))
            for item in payload.get("items", [])
            if item.get("metadata", {}).get("name")
        ]

    def _query_node_resources(self, node_id: str) -> Optional[NodeResourceData]:
        """Query a node resource endpoint or local metrics with cache fallback."""
        require_type(node_id, str, "node_id")
        now = time.time()
        cached = self._node_cache.get(node_id)
        if cached and (now - float(cached.get("timestamp", 0.0))) < self.node_cache_ttl:
            return dict(cached.get("data", {}))

        try:
            if self._is_local_node(node_id) and self.skip_localhost_http:
                resource_data = self._get_local_resources()
            else:
                response = requests.get(
                    f"http://{node_id}:{self.node_port}/metrics",
                    timeout=self.node_query_timeout,
                )
                response.raise_for_status()
                resource_data = self._normalise_node_metrics(response.json())

            self._node_cache[node_id] = {"data": dict(resource_data), "timestamp": now}
            return resource_data
        except RequestException as exc:
            if self._is_local_node(node_id):
                logger.debug("Using local resource fallback for %s", node_id)
                return self._get_local_resources()
            self._record_error(f"query_node:{node_id}", exc)
            logger.warning("Resource query failed for %s: %s", node_id, exc)
            return dict(cached.get("data", {})) if cached else None
        except Exception as exc:
            self._record_error(f"query_node:{node_id}", exc)
            logger.warning("Resource parsing failed for %s: %s", node_id, exc)
            return dict(cached.get("data", {})) if cached else None

    def _get_local_resources(self) -> NodeResourceData:
        """Read local RAM, CPU, storage, and GPU availability."""
        try:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            gpu_total = 0
            gpu_available = 0
            gpu_utilisation: Dict[str, float] = {}
            gpu_temperature: Dict[str, float] = {}
            local_cfg = dict(self.monitor_config.get("local_monitoring", {}) or {})
            max_gpu_load = float(local_cfg.get("max_gpu_load", 0.85))
            max_gpu_mem = float(local_cfg.get("max_gpu_memory_utilization", 0.90))

            gpus = GPUtil.getGPUs() if GPUtil is not None else []
            for gpu in gpus:
                gpu_total += 1
                gpu_utilisation[str(gpu.id)] = float(getattr(gpu, "load", 0.0) or 0.0)
                gpu_temperature[str(gpu.id)] = float(getattr(gpu, "temperature", 0.0) or 0.0)
                memory_util = 0.0
                if getattr(gpu, "memoryTotal", 0.0):
                    memory_util = float(getattr(gpu, "memoryUsed", 0.0) or 0.0) / float(gpu.memoryTotal)
                if gpu_utilisation[str(gpu.id)] <= max_gpu_load and memory_util <= max_gpu_mem:
                    gpu_available += 1

            hostname = socket.gethostname() if local_cfg.get("include_hostname_as_node", True) else "localhost"
            local_metrics = {
                "cpu": {hostname: psutil.cpu_percent(interval=None) / 100.0},
                "memory": {hostname: mem.percent / 100.0},
                "storage": {hostname: disk.percent / 100.0},
                "gpu": gpu_utilisation,
                "temperature": gpu_temperature,
            }
            self.update_metrics(local_metrics, force=True)

            return {
                "gpu_available": int(round(float(gpu_available))),
                "gpu_total": int(round(float(gpu_total))),
                "ram_available": int(round(float(mem.available) / (1024 ** 3))),
                "ram_total": int(round(float(mem.total) / (1024 ** 3))),
                "specialized_hw": [],
                "gpu_allocated": int(round(float(max(0, gpu_total - gpu_available)))),
                "ram_allocated": int(round(float(mem.used) / (1024 ** 3))),
                "specialized_allocated": [],
            }
        except Exception as exc:
            self._record_error("local_resources", exc)
            logger.error("Local resource check failed: %s", exc)
            fallback = dict(self.monitor_config.get("fallback_profile", {}) or {})
            return {
                "gpu_available": int(round(float(fallback.get("gpu_total", 0.0)))),
                "gpu_total": int(round(float(fallback.get("gpu_total", 0.0)))),
                "ram_available": int(round(float(fallback.get("ram_total", 16.0)))),
                "ram_total": int(round(float(fallback.get("ram_total", 16.0)))),
                "specialized_hw": list(fallback.get("specialized_hardware_available", []) or []),
                "gpu_allocated": 0.0,
                "ram_allocated": 0.0,
                "specialized_allocated": [],
            }

    def _update_resource_map(self) -> None:
        """Refresh the observed cluster resource map in a thread-safe way."""
        nodes = self._discover_cluster_nodes()
        observed = _cluster_resources_cls()(
            gpu_total=0,
            ram_total=0,
            specialized_hardware_available=[],
            current_allocations={},
        )
        observed_allocations: Dict[str, ResourceProfile] = {}
        resource_graph: Dict[str, NodeResourceData] = {}
        seen_hardware = set()

        for node_id in nodes:
            node_data = self._query_node_resources(node_id)
            if not node_data:
                continue

            resource_graph[node_id] = dict(node_data)
            gpu_add = float(node_data.get("gpu_available", node_data.get("gpu_total", 0.0)) or 0.0)
            ram_add = float(node_data.get("ram_available", node_data.get("ram_total", 0.0)) or 0.0)
            observed.gpu_total += int(round(gpu_add))
            observed.ram_total += int(round(ram_add))

            for hw in node_data.get("specialized_hw", []) or []:
                if hw not in seen_hardware:
                    observed.specialized_hardware_available.append(hw)
                    seen_hardware.add(hw)

            observed_allocations[node_id] = _resource_profile_cls()(
                gpu=int(round(float(node_data.get("gpu_allocated", 0.0) or 0.0))),
                ram=int(round(float(node_data.get("ram_allocated", 0.0) or 0.0))),
                specialized_hardware=list(node_data.get("specialized_allocated", []) or []),
            )

        if not resource_graph:
            logger.warning("No resource telemetry available; using fallback profile")
            fallback = dict(self.monitor_config.get("fallback_profile", {}) or {})
            observed = _cluster_resources_cls()(
                gpu_total=int(round(float(fallback.get("gpu_total", 0.0)))),
                ram_total=int(round(float(fallback.get("ram_total", 16.0)))),
                specialized_hardware_available=list(fallback.get("specialized_hardware_available", []) or []),
                current_allocations={},
            )

        with self._lock:
            previous_signature = self._cluster_signature(self.cluster_resources)
            self.cluster_resources = observed
            self.node_allocations = observed_allocations
            self.resource_graph = resource_graph
            self.cluster_resources.current_allocations = dict(self.allocations)
            current_signature = self._cluster_signature(observed)

        if current_signature != previous_signature:
            self._log_resource_map_if_needed(current_signature)

    # ------------------------------------------------------------------
    # Normalisation and validation helpers
    # ------------------------------------------------------------------
    def _normalise_node_metrics(self, metrics: Dict[str, Any]) -> NodeResourceData:
        """Convert heterogeneous node metrics into the monitor's resource schema."""
        require_type(metrics, dict, "metrics")
        gpu = dict(metrics.get("gpu", {}) or {})
        memory = dict(metrics.get("memory", metrics.get("ram", {})) or {})

        gpu_total = float(gpu.get("total", gpu.get("count", gpu.get("available", 0.0))) or 0.0)
        gpu_free = float(gpu.get("free", gpu.get("available", gpu_total)) or 0.0)
        ram_total = float(memory.get("total", memory.get("ram_total", memory.get("available", 0.0))) or 0.0)
        ram_free = float(memory.get("free", memory.get("available", ram_total)) or 0.0)

        return {
            "gpu_available": int(round(max(0.0, gpu_free))),
            "gpu_total": int(round(max(0.0, gpu_total))),
            "ram_available": int(round(max(0.0, ram_free))),
            "ram_total": int(round(max(0.0, ram_total))),
            "specialized_hw": list(metrics.get("specialized_hw", metrics.get("specialized_hardware", [])) or []),
            "gpu_allocated": int(round(max(0.0, gpu_total - gpu_free))),
            "ram_allocated": int(round(max(0.0, ram_total - ram_free))),
            "specialized_allocated": list(metrics.get("specialized_allocated", []) or []),
        }

    def _validate_config(self) -> None:
        mode = str(self.resource_config.get("mode", "static")).lower()
        if mode not in {"static", "consul", "k8s", "kubernetes", "local", "localhost"}:
            raise PlanningConfigError(
                f"Unsupported service discovery mode: {mode!r}",
                config_key="service_discovery.mode",
                expected_type="static|consul|k8s|local",
            )
        require_positive(float(self.monitor_config.get("update_interval", 5.0)), "cluster_monitoring.update_interval")
        require_positive(float(self.monitor_config.get("node_query_timeout", 2.0)), "cluster_monitoring.node_query_timeout")
        require_positive(float(self.monitor_config.get("node_cache_ttl", 10.0)), "cluster_monitoring.node_cache_ttl")
        require_positive(float(self.monitor_config.get("discovery_poll_interval", 300.0)), "cluster_monitoring.discovery_poll_interval")
        require_positive(int(self.monitor_config.get("history_limit", 1000)), "cluster_monitoring.history_limit")

        buffers = dict(self.safety_config.get("resource_buffers", {}) or {})
        for key in ("gpu", "ram"):
            value = float(buffers.get(key, 0.0))
            if not 0.0 <= value <= 1.0:
                raise PlanningConfigError(
                    f"Safety buffer for {key!r} must be in [0, 1], got {value}",
                    config_key=f"safety_margins.resource_buffers.{key}",
                    expected_type="fraction in [0, 1]",
                )

    def _validate_resource_profile(self, profile: ResourceProfile, name: str) -> None:
        require_type(profile, _resource_profile_cls(), name)
        require_non_negative(float(profile.gpu), f"{name}.gpu")
        require_non_negative(float(profile.ram), f"{name}.ram")
        if profile.specialized_hardware is None:
            profile.specialized_hardware = []
        require_type(profile.specialized_hardware, list, f"{name}.specialized_hardware")

    def _load_thresholds(self) -> Dict[str, float]:
        thresholds = dict(self.monitor_config.get("load_thresholds", {}) or {})
        normalised: Dict[str, float] = {}
        for key, value in thresholds.items():
            try:
                val = float(value)
                # Historical configs use 85/90 percentages; runtime metrics use
                # fractions for CPU/RAM/storage/GPU. Temperatures stay absolute.
                if key not in {"temperature"} and val > 1.0:
                    val = val / 100.0
                normalised[str(key)] = val
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid resource threshold %s=%r", key, value)
        return normalised

    @staticmethod
    def _normalise_nodes(nodes: Iterable[Any]) -> List[str]:
        clean: List[str] = []
        for node in nodes:
            if node is None:
                continue
            value = str(node).strip()
            if value and value not in clean:
                clean.append(value)
        return clean

    @staticmethod
    def _is_local_node(node_id: str) -> bool:
        return node_id in {"localhost", "127.0.0.1", "::1", socket.gethostname()}

    @staticmethod
    def _normalise_task_id(task_id: Optional[str]) -> str:
        if task_id is None or not str(task_id).strip():
            return f"anonymous-{int(time.time() * 1_000_000)}"
        return str(task_id).strip()

    @staticmethod
    def _copy_profile(profile: ResourceProfile) -> ResourceProfile:
        return _resource_profile_cls()(
            gpu=int(profile.gpu),
            ram=int(profile.ram),
            specialized_hardware=list(profile.specialized_hardware or []),
        )

    @staticmethod
    def _coerce_metric_map(value: Any) -> Dict[str, float]:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return {"default": float(value)}
        if not isinstance(value, dict):
            return {}
        result: Dict[str, float] = {}
        for key, raw in value.items():
            try:
                result[str(key)] = float(raw)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _profile_to_dict(profile: ResourceProfile) -> Dict[str, Any]:
        return {
            "gpu": float(profile.gpu),
            "ram": float(profile.ram),
            "specialized_hardware": list(profile.specialized_hardware or []),
        }

    @classmethod
    def _cluster_to_dict(cls, resources: ClusterResources) -> Dict[str, Any]:
        return {
            "gpu_total": float(getattr(resources, "gpu_total", 0.0) or 0.0),
            "ram_total": float(getattr(resources, "ram_total", 0.0) or 0.0),
            "specialized_hardware_available": list(
                getattr(resources, "specialized_hardware_available", []) or []
            ),
            "current_allocations": {
                key: cls._profile_to_dict(value)
                for key, value in dict(getattr(resources, "current_allocations", {}) or {}).items()
            },
        }

    @staticmethod
    def _cluster_signature(resources: ClusterResources) -> Tuple[Any, ...]:
        return (
            round(float(getattr(resources, "gpu_total", 0.0) or 0.0), 4),
            round(float(getattr(resources, "ram_total", 0.0) or 0.0), 4),
            tuple(sorted(getattr(resources, "specialized_hardware_available", []) or [])),
        )

    def _log_resource_map_if_needed(self, signature: Tuple[Any, ...]) -> None:
        now = time.time()
        should_log = (
            signature != ResourceMonitor._global_last_logged_cluster_signature
            or (now - ResourceMonitor._global_last_log_ts) >= self.resource_log_interval
        )
        if should_log:
            logger.info(
                "Cluster resource map updated: gpu=%s ram=%.2fGB hw=%s",
                signature[0],
                signature[1],
                list(signature[2]),
            )
            ResourceMonitor._global_last_logged_cluster_signature = signature
            ResourceMonitor._global_last_log_ts = now

    def _record_error(self, stage: str, exc: Exception) -> None:
        self._last_error = {
            "stage": stage,
            "error_type": type(exc).__name__,
            "message": truncate_for_logging(str(exc), 256),
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> "ResourceMonitor":
        self.start_monitoring()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_monitoring()


if __name__ == "__main__":
    print("\n=== Running Resource Monitor ===\n")
    printer.status("TEST", "Resource Monitor initialized", "info")

    monitor = ResourceMonitor()
    monitor.stop_monitoring()

    # Use deterministic test capacity independent of host hardware.
    monitor.cluster_resources = _cluster_resources_cls()(
        gpu_total=2,
        ram_total=32,
        specialized_hardware_available=["tensor_core", "npu"],
        current_allocations={},
    )

    req = _resource_profile_cls()(gpu=1, ram=4, specialized_hardware=["tensor_core"])
    assert monitor.acquire_resources(req, task_id="task_a") is True
    available = monitor.get_available_resources()
    assert available.gpu_total == 1
    assert available.ram_total == 28
    assert "tensor_core" not in available.specialized_hardware_available

    monitor.update_metrics({"cpu": {"local": 0.25}, "memory": {"local": 0.30}}, force=True)
    report = monitor.get_resource_report()
    assert report["planner_allocations"]["task_a"]["gpu"] == 1
    assert monitor.release_resources("task_a") is True
    assert monitor.get_available_resources().gpu_total == 2

    print("\n=== Test ran successfully ===\n")
