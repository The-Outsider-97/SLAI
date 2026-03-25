from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None


@dataclass
class CPUStats:
    percent_total: float
    percent_per_core: List[float]
    load_average: Optional[List[float]]


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_pretty_string(self) -> str:
        return (
            f"CPU total: {self.cpu.percent_total:.1f}% | per-core: {', '.join(f'{v:.1f}%' for v in self.cpu.percent_per_core)}\n"
            f"Memory: {self.memory.percent_used:.1f}% | used {self.memory.used_bytes:,} / {self.memory.total_bytes:,}\n"
            f"Disk: {self.disk.percent_used:.1f}% | read {self.disk.read_bytes:,} | write {self.disk.write_bytes:,}\n"
            f"Network: sent {self.network.bytes_sent:,} | recv {self.network.bytes_recv:,}\n"
            f"Processes: {self.process.running_count} running / {self.process.process_count} total"
        )


class MetricsCollector:
    """Collect structured system metrics with fallback for missing psutil."""

    def __init__(self, disk_path: str = "/") -> None:
        self.disk_path = disk_path

    def collect_snapshot(self) -> MetricSnapshot:
        """Return a MetricSnapshot of current system metrics."""
        now = datetime.now(timezone.utc).isoformat()
        if psutil is None:
            return self._collect_fallback_snapshot(now)

        # CPU
        cpu_total = float(psutil.cpu_percent(interval=0.15))
        cpu_per_core = [float(v) for v in psutil.cpu_percent(interval=0.15, percpu=True)]
        load_avg = None
        if hasattr(psutil, "getloadavg"):
            try:
                load_avg = [float(v) for v in psutil.getloadavg()]
            except (OSError, AttributeError):
                pass

        # Memory
        mem = psutil.virtual_memory()
        # Disk
        disk = psutil.disk_usage(self.disk_path)
        disk_io = psutil.disk_io_counters()
        # Network
        net_io = psutil.net_io_counters()

        # Processes (iterating over all processes)
        process_count = 0
        running_count = 0
        for proc in psutil.process_iter(attrs=["status"]):
            process_count += 1
            if proc.info.get("status") == psutil.STATUS_RUNNING:
                running_count += 1

        return MetricSnapshot(
            timestamp_utc=now,
            cpu=CPUStats(
                percent_total=cpu_total,
                percent_per_core=cpu_per_core,
                load_average=load_avg,
            ),
            memory=MemoryStats(
                percent_used=float(mem.percent),
                used_bytes=int(mem.used),
                available_bytes=int(mem.available),
                total_bytes=int(mem.total),
            ),
            disk=DiskStats(
                percent_used=float(disk.percent),
                used_bytes=int(disk.used),
                free_bytes=int(disk.free),
                read_bytes=int(getattr(disk_io, "read_bytes", 0)) if disk_io else 0,
                write_bytes=int(getattr(disk_io, "write_bytes", 0)) if disk_io else 0,
            ),
            network=NetworkStats(
                bytes_sent=int(net_io.bytes_sent) if net_io else 0,
                bytes_recv=int(net_io.bytes_recv) if net_io else 0,
                packets_sent=int(net_io.packets_sent) if net_io else 0,
                packets_recv=int(net_io.packets_recv) if net_io else 0,
            ),
            process=ProcessStats(
                process_count=process_count,
                running_count=running_count,
            ),
        )

    def _collect_fallback_snapshot(self, timestamp: str) -> MetricSnapshot:
        """Fallback when psutil is unavailable."""
        return MetricSnapshot(
            timestamp_utc=timestamp,
            cpu=CPUStats(percent_total=0.0, percent_per_core=[], load_average=None),
            memory=MemoryStats(percent_used=0.0, used_bytes=0, available_bytes=0, total_bytes=0),
            disk=DiskStats(percent_used=0.0, used_bytes=0, free_bytes=0, read_bytes=0, write_bytes=0),
            network=NetworkStats(bytes_sent=0, bytes_recv=0, packets_sent=0, packets_recv=0),
            process=ProcessStats(process_count=0, running_count=0),
        )


def collect_system_metrics() -> Dict[str, Any]:
    """Compatibility helper: returns a dict of a metrics snapshot."""
    return MetricsCollector().collect_snapshot().to_dict()


if __name__ == "__main__":
    collector = MetricsCollector()
    print(collector.collect_snapshot().to_pretty_string())
