"""
monitoring/metrics_collector.py
────────────────────────────────
Collects structured system metrics (CPU, memory, disk, network, processes).

Key improvements over v1
─────────────────────────
• Interval-based collection loop via `MetricsCollector.start()` / `.stop()`
• Bounded in-memory history with `get_history()` and `get_aggregates()`
• Prometheus text-format export via `to_prometheus()`
• Robust partial-failure handling (disk_io_counters / net_io_counters can be None)
• All psutil calls isolated so a single failure never drops the whole snapshot
• Thread-safe history buffer
"""

from __future__ import annotations

import threading
import time
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None          # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False

from collections import deque
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any

from .config_loader import get_config_section, load_global_config
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Metrics Collector")
printer = PrettyPrinter()

# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────
@dataclass
class CPUStats:
    percent_total: float
    percent_per_core: list[float]
    load_average: list[float] | None


@dataclass
class MemoryStats:
    percent_used: float
    used_bytes: int
    available_bytes: int
    total_bytes: int


@dataclass
class DiskStats:
    percent_used: float
    used_bytes: int
    free_bytes: int
    read_bytes: int
    write_bytes: int


@dataclass
class NetworkStats:
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int


@dataclass
class ProcessStats:
    process_count: int
    running_count: int


@dataclass
class MetricSnapshot:
    timestamp_utc: str
    cpu: CPUStats
    memory: MemoryStats
    disk: DiskStats
    network: NetworkStats
    process: ProcessStats
    # True when psutil is missing or a sub-collector failed
    partial: bool = False
    collection_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_pretty_string(self) -> str:
        lines = [
            f"CPU total: {self.cpu.percent_total:.1f}% | "
            f"per-core: {', '.join(f'{v:.1f}%' for v in self.cpu.percent_per_core)}",
            f"Memory: {self.memory.percent_used:.1f}% | "
            f"used {self.memory.used_bytes:,} / {self.memory.total_bytes:,}",
            f"Disk: {self.disk.percent_used:.1f}% | "
            f"read {self.disk.read_bytes:,} | write {self.disk.write_bytes:,}",
            f"Network: sent {self.network.bytes_sent:,} | recv {self.network.bytes_recv:,}",
            f"Processes: {self.process.running_count} running / {self.process.process_count} total",
        ]
        if self.partial:
            lines.append(f"[PARTIAL] errors: {'; '.join(self.collection_errors)}")
        return "\n".join(lines)

    def to_prometheus(self, prefix: str = "slai") -> str:
        """
        Return a Prometheus text-format string suitable for a /metrics endpoint.
        """
        ts_ms = int(
            datetime.fromisoformat(self.timestamp_utc).timestamp() * 1000
        )
        lines: list[str] = []

        def gauge(name: str, value: float, labels: str = "") -> None:
            metric_name = f"{prefix}_{name}"
            label_str = f"{{{labels}}}" if labels else ""
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name}{label_str} {value} {ts_ms}")

        gauge("cpu_percent_total", self.cpu.percent_total)
        for i, v in enumerate(self.cpu.percent_per_core):
            gauge("cpu_percent_core", v, f'core="{i}"')
        if self.cpu.load_average:
            for window, v in zip(("1m", "5m", "15m"), self.cpu.load_average):
                gauge("load_average", v, f'window="{window}"')

        gauge("memory_percent_used", self.memory.percent_used)
        gauge("memory_used_bytes", float(self.memory.used_bytes))
        gauge("memory_available_bytes", float(self.memory.available_bytes))
        gauge("memory_total_bytes", float(self.memory.total_bytes))

        gauge("disk_percent_used", self.disk.percent_used)
        gauge("disk_used_bytes", float(self.disk.used_bytes))
        gauge("disk_free_bytes", float(self.disk.free_bytes))
        gauge("disk_read_bytes_total", float(self.disk.read_bytes))
        gauge("disk_write_bytes_total", float(self.disk.write_bytes))

        gauge("network_bytes_sent_total", float(self.network.bytes_sent))
        gauge("network_bytes_recv_total", float(self.network.bytes_recv))
        gauge("network_packets_sent_total", float(self.network.packets_sent))
        gauge("network_packets_recv_total", float(self.network.packets_recv))

        gauge("process_count", float(self.process.process_count))
        gauge("process_running_count", float(self.process.running_count))

        return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────
# Aggregates
# ──────────────────────────────────────────────
@dataclass
class MetricAggregates:
    """Simple min/max/avg aggregates over a snapshot window."""
    window_size: int
    cpu_percent_avg: float
    cpu_percent_max: float
    memory_percent_avg: float
    memory_percent_max: float
    disk_percent_avg: float
    disk_percent_max: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────
# Collector
# ──────────────────────────────────────────────
class MetricsCollector:
    """
    Collect structured system metrics with optional background loop.

    Parameters
    ----------
    disk_path:
        Mount point to measure disk usage for.
    history_max_snapshots:
        Maximum snapshots held in the ring buffer (default 240 = 1 h at 15 s intervals).
    """

    def __init__(
        self,
        disk_path: str = "/",
        history_max_snapshots: int = 240,
    ) -> None:
        self.config = load_global_config()
        self.collector_config = get_config_section("metrics_collector")

        self.disk_path = disk_path
        self._history: deque[MetricSnapshot] = deque(maxlen=history_max_snapshots)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Public API ───────────────────────────────

    def collect_snapshot(self) -> MetricSnapshot:
        """Collect and return a single MetricSnapshot right now."""
        now = datetime.now(timezone.utc).isoformat()
        if not _PSUTIL_AVAILABLE:
            logger.warning("psutil not available; returning zero-value snapshot.")
            snap = self._fallback_snapshot(now)
            snap.collection_errors.append("psutil not installed")
            snap.partial = True
            return snap
        return self._collect_live_snapshot(now)

    def start(self, interval_seconds: float = 15.0) -> None:
        """Start a background thread that collects snapshots every *interval_seconds*."""
        if self._thread and self._thread.is_alive():
            logger.warning("MetricsCollector background loop already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            args=(interval_seconds,),
            daemon=True,
            name="slai-metrics-collector",
        )
        self._thread.start()
        logger.info("MetricsCollector started.", interval=interval_seconds)

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the background loop to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("MetricsCollector stopped.")

    def get_history(self) -> list[MetricSnapshot]:
        """Return a copy of the snapshot history (oldest → newest)."""
        with self._lock:
            return list(self._history)

    def get_aggregates(self) -> MetricAggregates | None:
        """Return simple aggregates over all buffered snapshots, or None if empty."""
        with self._lock:
            snaps = list(self._history)
        if not snaps:
            return None

        def _avg(vals: list[float]) -> float:
            return sum(vals) / len(vals)

        cpu = [s.cpu.percent_total for s in snaps]
        mem = [s.memory.percent_used for s in snaps]
        disk = [s.disk.percent_used for s in snaps]

        return MetricAggregates(
            window_size=len(snaps),
            cpu_percent_avg=_avg(cpu),
            cpu_percent_max=max(cpu),
            memory_percent_avg=_avg(mem),
            memory_percent_max=max(mem),
            disk_percent_avg=_avg(disk),
            disk_percent_max=max(disk),
        )

    # ── Internal ─────────────────────────────────

    def _loop(self, interval: float) -> None:
        while not self._stop_event.is_set():
            try:
                snap = self.collect_snapshot()
                with self._lock:
                    self._history.append(snap)
            except Exception as exc:
                logger.error("Unexpected error in metrics loop.", error=str(exc))
            self._stop_event.wait(timeout=interval)

    def _collect_live_snapshot(self, now: str) -> MetricSnapshot:
        # Help the type checker: this method is only called when psutil is available
        assert psutil is not None
    
        errors: list[str] = []
    
        # ── CPU ──────────────────────────────────
        try:
            cpu_total = float(psutil.cpu_percent(interval=0.15))
            cpu_per_core = [float(v) for v in psutil.cpu_percent(interval=0.0, percpu=True)]
            load_avg: list[float] | None = None
            if hasattr(psutil, "getloadavg"):
                try:
                    load_avg = [float(v) for v in psutil.getloadavg()]
                except OSError:
                    pass
            cpu = CPUStats(percent_total=cpu_total, percent_per_core=cpu_per_core, load_average=load_avg)
        except Exception as exc:
            errors.append(f"cpu: {exc}")
            cpu = CPUStats(percent_total=0.0, percent_per_core=[], load_average=None)
    
        # ── Memory ───────────────────────────────
        try:
            mem = psutil.virtual_memory()
            memory = MemoryStats(
                percent_used=float(mem.percent),
                used_bytes=int(mem.used),
                available_bytes=int(mem.available),
                total_bytes=int(mem.total),
            )
        except Exception as exc:
            errors.append(f"memory: {exc}")
            memory = MemoryStats(percent_used=0.0, used_bytes=0, available_bytes=0, total_bytes=0)
    
        # ── Disk ─────────────────────────────────
        try:
            disk_usage = psutil.disk_usage(self.disk_path)
            disk_io = None
            try:
                disk_io = psutil.disk_io_counters()
            except (OSError, AttributeError):
                errors.append("disk_io: unavailable on this platform")
            disk = DiskStats(
                percent_used=float(disk_usage.percent),
                used_bytes=int(disk_usage.used),
                free_bytes=int(disk_usage.free),
                read_bytes=int(getattr(disk_io, "read_bytes", 0)) if disk_io else 0,
                write_bytes=int(getattr(disk_io, "write_bytes", 0)) if disk_io else 0,
            )
        except Exception as exc:
            errors.append(f"disk: {exc}")
            disk = DiskStats(percent_used=0.0, used_bytes=0, free_bytes=0, read_bytes=0, write_bytes=0)
    
        # ── Network ──────────────────────────────
        try:
            net_io = psutil.net_io_counters()
            network = NetworkStats(
                bytes_sent=int(net_io.bytes_sent) if net_io else 0,
                bytes_recv=int(net_io.bytes_recv) if net_io else 0,
                packets_sent=int(net_io.packets_sent) if net_io else 0,
                packets_recv=int(net_io.packets_recv) if net_io else 0,
            )
        except Exception as exc:
            errors.append(f"network: {exc}")
            network = NetworkStats(bytes_sent=0, bytes_recv=0, packets_sent=0, packets_recv=0)
    
        # ── Processes ────────────────────────────
        try:
            process_count = 0
            running_count = 0
            for proc in psutil.process_iter(attrs=["status"]):
                process_count += 1
                try:
                    if proc.info.get("status") == psutil.STATUS_RUNNING:
                        running_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            process = ProcessStats(process_count=process_count, running_count=running_count)
        except Exception as exc:
            errors.append(f"processes: {exc}")
            process = ProcessStats(process_count=0, running_count=0)
    
        return MetricSnapshot(
            timestamp_utc=now,
            cpu=cpu,
            memory=memory,
            disk=disk,
            network=network,
            process=process,
            partial=bool(errors),
            collection_errors=errors,
        )

    @staticmethod
    def _fallback_snapshot(timestamp: str) -> MetricSnapshot:
        return MetricSnapshot(
            timestamp_utc=timestamp,
            cpu=CPUStats(percent_total=0.0, percent_per_core=[], load_average=None),
            memory=MemoryStats(percent_used=0.0, used_bytes=0, available_bytes=0, total_bytes=0),
            disk=DiskStats(percent_used=0.0, used_bytes=0, free_bytes=0, read_bytes=0, write_bytes=0),
            network=NetworkStats(bytes_sent=0, bytes_recv=0, packets_sent=0, packets_recv=0),
            process=ProcessStats(process_count=0, running_count=0),
            partial=True,
        )


# ── Compatibility helper ─────────────────────────────────────────────────────

def collect_system_metrics() -> dict[str, Any]:
    """Compatibility shim: returns a dict of the current metric snapshot."""
    return MetricsCollector().collect_snapshot().to_dict()


if __name__ == "__main__":
    collector = MetricsCollector()
    snap = collector.collect_snapshot()
    print(snap.to_pretty_string())
    print()
    print(snap.to_prometheus())