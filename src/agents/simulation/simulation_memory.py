"""Bounded, simulation-specific run memory.

This memory preserves run lineage and replay metadata rather than replacing
SLAI shared memory or checkpointing.  Its reproducibility contract follows
Sandve et al. (2013): inputs, model identity, random seed/state, parameters,
and scenario lineage remain inspectable after execution.
"""

from __future__ import annotations

__version__ = "2.3.0"

import threading

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from .simulation_types import *
from .utils.config_loader import get_config_section, load_global_config
from .utils.simulation_errors import *
from .utils.simulation_helpers import *
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Simulation Memory")


@dataclass(frozen=True, slots=True)
class SimulationMemoryRecord:
    run_id: str
    scenario_id: Optional[str]
    parent_run_id: Optional[str]
    model_id: str
    model_version: str
    seed: int
    termination_reason: str
    initial_state_fingerprint: str
    actions_fingerprint: str
    parameters_fingerprint: str
    trajectory_id: str
    transition_count: int
    rng_state_initial: Mapping[str, Any]
    rng_state_final: Mapping[str, Any]
    created_at: str
    request_payload: Optional[Mapping[str, Any]] = None
    result_payload: Optional[Mapping[str, Any]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe({
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "parent_run_id": self.parent_run_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "seed": self.seed,
            "termination_reason": self.termination_reason,
            "initial_state_fingerprint": self.initial_state_fingerprint,
            "actions_fingerprint": self.actions_fingerprint,
            "parameters_fingerprint": self.parameters_fingerprint,
            "trajectory_id": self.trajectory_id,
            "transition_count": self.transition_count,
            "rng_state_initial": dict(self.rng_state_initial),
            "rng_state_final": dict(self.rng_state_final),
            "created_at": self.created_at,
            "request_payload": self.request_payload,
            "result_payload": self.result_payload,
            "metadata": dict(self.metadata),
        })


class SimulationMemory:
    """Thread-safe bounded index of recent Simulation runs."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("simulation_memory", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.max_records = int(section.get("max_records", 256))
        if self.max_records <= 0:
            raise SimulationValidationError("simulation_memory.max_records must be > 0")
        self.store_request_payloads = bool(section.get("store_request_payloads", True))
        self.store_result_payloads = bool(section.get("store_result_payloads", False))
        self._records: deque[SimulationMemoryRecord] = deque(maxlen=self.max_records)
        self._index: dict[str, SimulationMemoryRecord] = {}
        self._lock = threading.RLock()
        logger.info("Simulation memory initialized | max_records=%s", self.max_records)

    def record(
        self,
        result: SimulationResult,
        *,
        request: Optional[SimulationRequest] = None,
        parent_run_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SimulationMemoryRecord:
        if not isinstance(result, SimulationResult):
            raise SimulationMemoryError("result must be a SimulationResult")
        request_payload = request.to_dict() if request is not None and self.store_request_payloads else None
        result_payload = result.to_dict() if self.store_result_payloads else None
        record = SimulationMemoryRecord(
            run_id=result.run_id,
            scenario_id=result.scenario_id,
            parent_run_id=parent_run_id,
            model_id=result.model_id,
            model_version=result.model_version,
            seed=result.seed,
            termination_reason=result.termination_reason.value,
            initial_state_fingerprint=stable_fingerprint(result.initial_state, length=32),
            actions_fingerprint=stable_fingerprint(result.actions_supplied, length=32),
            parameters_fingerprint=stable_fingerprint(result.parameters, length=32),
            trajectory_id=result.trajectory.trajectory_id,
            transition_count=result.trajectory.transition_count,
            rng_state_initial=to_json_safe(dict(result.rng_state_initial)),
            rng_state_final=to_json_safe(dict(result.rng_state_final)),
            created_at=utc_now_iso(),
            request_payload=request_payload,
            result_payload=result_payload,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            if self._records.maxlen and len(self._records) == self._records.maxlen:
                evicted = self._records[0]
                self._index.pop(evicted.run_id, None)
            self._records.append(record)
            self._index[record.run_id] = record
        return record

    def get(self, run_id: str) -> Optional[SimulationMemoryRecord]:
        with self._lock:
            return self._index.get(str(run_id))

    def require(self, run_id: str) -> SimulationMemoryRecord:
        record = self.get(run_id)
        if record is None:
            raise SimulationMemoryError("simulation run is not present in memory", context={"run_id": run_id})
        return record

    def recent(self, limit: Optional[int] = None) -> tuple[SimulationMemoryRecord, ...]:
        with self._lock:
            records = tuple(self._records)
        if limit is None:
            return records
        if limit < 0:
            raise SimulationValidationError("limit must be >= 0")
        return records[-limit:] if limit else ()

    def scenario_runs(self, scenario_id: str) -> tuple[SimulationMemoryRecord, ...]:
        sid = str(scenario_id)
        with self._lock:
            return tuple(record for record in self._records if record.scenario_id == sid)

    def lineage(self, run_id: str) -> tuple[SimulationMemoryRecord, ...]:
        chain: list[SimulationMemoryRecord] = []
        visited: set[str] = set()
        current = self.get(run_id)
        while current is not None:
            if current.run_id in visited:
                raise SimulationMemoryError("cycle detected in run lineage", context={"run_id": current.run_id})
            visited.add(current.run_id)
            chain.append(current)
            current = self.get(current.parent_run_id) if current.parent_run_id else None
        chain.reverse()
        return tuple(chain)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._index.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            scenario_count = len({record.scenario_id for record in self._records if record.scenario_id is not None})
            return {
                "record_count": len(self._records),
                "capacity": self.max_records,
                "scenario_count": scenario_count,
                "store_request_payloads": self.store_request_payloads,
                "store_result_payloads": self.store_result_payloads,
            }


__all__ = ["SimulationMemory", "SimulationMemoryRecord"]
