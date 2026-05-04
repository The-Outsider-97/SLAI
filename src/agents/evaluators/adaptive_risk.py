
"""Production-ready adaptive Bayesian risk assessment engine.

This module implements a real-time risk assessment workflow aligned with:
- STPA / STAMP-style safety reasoning for unsafe control actions
- ISO 21448 (SOTIF) style treatment of known and unknown performance limitations
- Dynamic Bayesian adaptation of hazard-rate estimates using conjugate updates

The implementation is designed for continuous evaluator integration, checkpoint-
friendly reporting, and memory-backed auditability.
"""

from __future__ import annotations

import json
import math
import time
import threading
import jsonschema
import numpy as np

from pathlib import Path
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .evaluators_memory import EvaluatorsMemory
from .utils.evaluation_errors import (ConfigLoadError, OperationalError, MemoryAccessError,
                                      StatisticalAnalysisError, ValidationFailureError,
                                      SerializationError, DocumentationProcessingError)
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Adaptive Risk")
printer = PrettyPrinter


SAFETY_CASE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "metadata": {"type": "object"},
        "system_description": {"type": "object"},
        "control_structure": {"type": "object"},
        "safety_requirements": {"type": "object"},
        "hazard_analysis": {"type": "object"},
        "evidence_base": {"type": "object"},
        "validation": {"type": "object"},
        "framework_alignment": {"type": "object"},
    },
    "required": [
        "metadata",
        "hazard_analysis",
        "safety_requirements",
        "validation",
        "framework_alignment",
    ],
}


@dataclass(slots=True)
class HazardBayesianState:
    """Posterior state for a single hazard-rate model."""

    hazard_id: str
    prior_rate: float
    base_alpha: float
    base_beta: float
    alpha: float
    beta: float
    total_occurrences: int = 0
    total_operational_time: float = 0.0
    update_count: int = 0
    last_updated: str = field(default_factory=lambda: _utcnow().isoformat())

    @property
    def mean_rate(self) -> float:
        return self.alpha / self.beta if self.beta > 0 else 0.0

    @property
    def variance(self) -> float:
        return self.alpha / (self.beta ** 2) if self.beta > 0 else 0.0

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    @property
    def coefficient_of_variation(self) -> float:
        mean = self.mean_rate
        return self.standard_deviation / mean if mean > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["mean_rate"] = self.mean_rate
        payload["variance"] = self.variance
        payload["standard_deviation"] = self.standard_deviation
        payload["coefficient_of_variation"] = self.coefficient_of_variation
        return payload


@dataclass(slots=True)
class RiskObservation:
    """Single observation batch used to update the adaptive risk model."""

    timestamp: str
    hazard_counts: Dict[str, int]
    operational_time: float
    source: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = _coerce_timestamp(self.timestamp).isoformat()

        if not isinstance(self.hazard_counts, dict) or not self.hazard_counts:
            raise ValidationFailureError(
                "risk_observation.hazard_counts",
                type(self.hazard_counts).__name__,
                "non-empty mapping",
            )
        normalized: Dict[str, int] = {}
        for hazard, count in self.hazard_counts.items():
            name = _normalize_non_empty_string(hazard, "hazard_counts.key")
            if not isinstance(count, int) or count < 0:
                raise ValidationFailureError(
                    f"risk_observation.hazard_counts[{name}]",
                    count,
                    "non-negative integer",
                )
            normalized[name] = count
        self.hazard_counts = normalized

        self.operational_time = _require_positive_float(
            self.operational_time,
            "risk_observation.operational_time",
        )

        if self.source is not None:
            self.source = _normalize_non_empty_string(self.source, "risk_observation.source")

        if not isinstance(self.context, dict):
            raise ValidationFailureError(
                "risk_observation.context",
                type(self.context).__name__,
                "mapping",
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SchedulerJob:
    """Simple interval-based background job."""

    name: str
    func: Callable[[], Any]
    interval: timedelta
    next_run: datetime
    run_count: int = 0
    last_run: Optional[str] = None

    def should_run(self, now: datetime) -> bool:
        return now >= self.next_run

    def mark_run(self, now: datetime) -> None:
        self.run_count += 1
        self.last_run = now.isoformat()
        self.next_run = now + self.interval


class BackgroundScheduler:
    """Minimal thread-based scheduler for periodic risk tasks."""

    def __init__(self, poll_interval_seconds: int = 30) -> None:
        self.poll_interval_seconds = _require_positive_int(
            poll_interval_seconds,
            "BackgroundScheduler.poll_interval_seconds",
        )
        self.jobs: List[SchedulerJob] = []
        self.thread: Optional[threading.Thread] = None
        self.active = False
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

    def add_job(
        self,
        func: Callable[[], Any],
        trigger: str,
        *,
        name: Optional[str] = None,
        hours: Optional[float] = None,
        seconds: Optional[int] = None,
        next_run_time: Optional[datetime] = None,
    ) -> SchedulerJob:
        if trigger != "interval":
            raise ValidationFailureError("scheduler.trigger", trigger, "interval")

        if hours is None and seconds is None:
            raise ValidationFailureError("scheduler.interval", None, "hours or seconds")

        interval_seconds = 0.0
        if hours is not None:
            interval_seconds += _require_positive_float(hours, "scheduler.hours") * 3600.0
        if seconds is not None:
            interval_seconds += _require_positive_int(seconds, "scheduler.seconds")

        job = SchedulerJob(
            name=_normalize_non_empty_string(name or getattr(func, "__name__", "scheduled_job"), "scheduler.job_name"),
            func=func,
            interval=timedelta(seconds=interval_seconds),
            next_run=next_run_time or _utcnow(),
        )
        with self._lock:
            self.jobs.append(job)
        return job

    def start(self) -> None:
        with self._lock:
            if self.active:
                return
            self.active = True
            self._stop_event.clear()
            self.thread = threading.Thread(target=self._run, daemon=True, name="AdaptiveRiskScheduler")
            self.thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            now = _utcnow()
            with self._lock:
                jobs_snapshot = list(self.jobs)

            for job in jobs_snapshot:
                if not job.should_run(now):
                    continue
                try:
                    job.func()
                except Exception as exc:  # noqa: BLE001
                    logger.error("Scheduled job '%s' failed: %s", job.name, exc, exc_info=True)
                finally:
                    with self._lock:
                        job.mark_run(now)

            self._stop_event.wait(self.poll_interval_seconds)

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            if not self.active:
                return
            self.active = False
            self._stop_event.set()
            thread = self.thread
        if thread is not None:
            thread.join(timeout=timeout)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": job.name,
                    "interval_seconds": job.interval.total_seconds(),
                    "next_run": job.next_run.isoformat(),
                    "run_count": job.run_count,
                    "last_run": job.last_run,
                }
                for job in self.jobs
            ]


class RiskAdaptation:
    """
    Real-time Bayesian risk adaptation engine.

    Core responsibilities
    ---------------------
    - Maintain posterior hazard-rate estimates with adaptive forgetting
    - Accept streaming operational observations
    - Produce per-hazard and system-level risk summaries
    - Generate STPA/SOTIF-aligned safety-case and reporting artifacts
    - Persist reports and state transitions into evaluator memory
    """

    def __init__(self, *, auto_start_scheduler: bool = True) -> None:
        self.config = load_global_config()
        self.config_path = str(self.config.get("__config_path__", "<config>"))

        self.risk_config = get_config_section("risk_adaptation")
        self.reporting_config = get_config_section("automated_reporting")
        self.documentation_config = get_config_section("documentation")
        self.safety_config = get_config_section("safety_evaluator")
        self.hazard_config = get_config_section("initial_hazard_rates")

        self.learning_rate = _coerce_probability(
            self.risk_config.get("learning_rate", 0.05),
            "risk_adaptation.learning_rate",
            inclusive_zero=False,
            inclusive_one=True,
        )
        self.uncertainty_window = _require_positive_float(
            self.risk_config.get("uncertainty_window", 1000),
            "risk_adaptation.uncertainty_window",
        )
        self.report_interval_hours = _require_positive_float(
            self.reporting_config.get("interval_hours", 24),
            "automated_reporting.interval_hours",
        )
        self.max_safety_case_versions = _require_positive_int(
            (self.documentation_config.get("versioning") or {}).get("max_versions", 7),
            "documentation.versioning.max_versions",
        )
        self.retention_days = _require_positive_int(
            self.reporting_config.get("retention_days", 30),
            "automated_reporting.retention_days",
        )
        self.max_risk_level = _coerce_probability(
            (self.safety_config.get("thresholds") or {}).get("max_risk_level", 0.3),
            "safety_evaluator.thresholds.max_risk_level",
            inclusive_zero=True,
            inclusive_one=True,
        )
        self.min_safety_margin = _coerce_probability(
            (self.safety_config.get("thresholds") or {}).get("min_safety_margin", 0.5),
            "safety_evaluator.thresholds.min_safety_margin",
            inclusive_zero=True,
            inclusive_one=True,
        )
        self.risk_categories = _normalize_string_list(
            self.safety_config.get(
                "risk_categories",
                ["collision", "pinch_point", "crush_hazard", "electrical", "environmental", "control_failure"],
            ),
            "safety_evaluator.risk_categories",
        )

        if not isinstance(self.hazard_config, dict) or not self.hazard_config:
            raise ConfigLoadError(
                self.config_path,
                "initial_hazard_rates",
                "Expected a non-empty mapping of hazard priors.",
            )

        self.initial_hazard_rates = {
            _normalize_non_empty_string(hazard, "initial_hazard_rates.key"): _require_non_negative_float(
                rate,
                f"initial_hazard_rates.{hazard}",
            )
            for hazard, rate in self.hazard_config.items()
        }

        self.memory = EvaluatorsMemory()
        self._lock = threading.RLock()
        self.hazard_states: Dict[str, HazardBayesianState] = self._initialize_model()
        self.observation_history: List[RiskObservation] = []
        self.safety_case_versions: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.scheduler = BackgroundScheduler()
        self._report_directory = self._resolve_output_dir("src/agents/evaluators/reports")
        self._docs_directory = self._resolve_output_dir("src/agents/evaluators/docs/safety_cases")
        self._state_directory = self._resolve_output_dir("src/agents/evaluators/checkpoints/risk")

        self._report_directory.mkdir(parents=True, exist_ok=True)
        self._docs_directory.mkdir(parents=True, exist_ok=True)
        self._state_directory.mkdir(parents=True, exist_ok=True)

        if auto_start_scheduler:
            self._init_report_scheduler()

        logger.info(
            "Adaptive Risk successfully initialized: hazards=%s learning_rate=%.4f uncertainty_window=%.2f",
            list(self.hazard_states.keys()),
            self.learning_rate,
            self.uncertainty_window,
        )

    # ------------------------------------------------------------------
    # Initialization and lifecycle
    # ------------------------------------------------------------------
    def _initialize_model(self) -> Dict[str, HazardBayesianState]:
        """Create prior Gamma posteriors for each configured hazard."""
        states: Dict[str, HazardBayesianState] = {}
        for hazard, rate in self.initial_hazard_rates.items():
            base_beta = max(self.uncertainty_window, 1.0)
            base_alpha = max(rate * base_beta, 1e-9)
            states[hazard] = HazardBayesianState(
                hazard_id=hazard,
                prior_rate=rate,
                base_alpha=base_alpha,
                base_beta=base_beta,
                alpha=base_alpha,
                beta=base_beta,
            )
        return states

    def _init_report_scheduler(self) -> None:
        self.scheduler.add_job(
            self.generate_automated_report,
            trigger="interval",
            hours=self.report_interval_hours,
            next_run_time=_utcnow(),
            name="adaptive_risk_daily_report",
        )
        self.scheduler.start()
        logger.info("Background scheduler started for adaptive-risk reports")

    def shutdown(self) -> None:
        """Gracefully stop background scheduling."""
        self.scheduler.stop()

    # ------------------------------------------------------------------
    # Observation ingestion and Bayesian updating
    # ------------------------------------------------------------------
    def update_model(
        self,
        observations: Mapping[str, int],
        operational_time: float,
        *,
        timestamp: Optional[str] = None,
        source: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        store_event: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Update the Bayesian hazard model using observed event counts.

        The model uses Gamma-Poisson conjugacy with adaptive forgetting toward
        the configured base prior. This makes the risk model responsive to new
        evidence without discarding accumulated operational experience.
        """
        if not isinstance(observations, Mapping):
            raise ValidationFailureError(
                "adaptive_risk.observations",
                type(observations).__name__,
                "mapping of hazard->count",
            )

        observation = RiskObservation(
            timestamp=timestamp or _utcnow().isoformat(),
            hazard_counts=dict(observations),
            operational_time=operational_time,
            source=source,
            context=dict(context or {}),
        )

        updates: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for hazard in observation.hazard_counts:
                if hazard not in self.hazard_states:
                    raise ValidationFailureError(
                        "adaptive_risk.hazard_name",
                        hazard,
                        f"one of {sorted(self.hazard_states.keys())}",
                    )

            for hazard, count in observation.hazard_counts.items():
                state = self.hazard_states[hazard]

                adaptive_alpha = state.base_alpha + (1.0 - self.learning_rate) * (state.alpha - state.base_alpha)
                adaptive_beta = state.base_beta + (1.0 - self.learning_rate) * (state.beta - state.base_beta)

                posterior_alpha = adaptive_alpha + float(count)
                posterior_beta = adaptive_beta + observation.operational_time

                if posterior_beta <= 0:
                    raise StatisticalAnalysisError(
                        "hazard_rate_update",
                        {"hazard": hazard, "beta": posterior_beta},
                        "posterior beta must remain positive",
                    )

                state.alpha = posterior_alpha
                state.beta = posterior_beta
                state.total_occurrences += int(count)
                state.total_operational_time += observation.operational_time
                state.update_count += 1
                state.last_updated = observation.timestamp

                updates[hazard] = {
                    "hazard_id": hazard,
                    "posterior_alpha": posterior_alpha,
                    "posterior_beta": posterior_beta,
                    "mean_rate": state.mean_rate,
                    "variance": state.variance,
                    "event_count": int(count),
                    "operational_time": observation.operational_time,
                }

            self.observation_history.append(observation)
            self._prune_retained_history_locked()

        if store_event:
            self._store_memory_event(
                {
                    "event": "risk_model_update",
                    "timestamp": observation.timestamp,
                    "source": observation.source,
                    "operational_time": observation.operational_time,
                    "hazard_counts": dict(observation.hazard_counts),
                    "context": dict(observation.context),
                    "posterior_updates": updates,
                },
                tags=["risk_update", "bayesian_update"],
                priority="high",
                category="risk_updates",
                source=source or "adaptive_risk",
            )

        return updates

    def process_runtime_snapshot(self, snapshot: Mapping[str, Any], *, source: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Convenience ingestion helper for evaluator/runtime snapshots.

        Expected keys:
        - hazards: mapping hazard->count
        - operational_time: positive float
        - timestamp: optional ISO-8601 timestamp
        - context: optional mapping
        """
        if not isinstance(snapshot, Mapping):
            raise ValidationFailureError(
                "adaptive_risk.snapshot",
                type(snapshot).__name__,
                "mapping",
            )
        if "hazards" not in snapshot:
            raise ValidationFailureError("adaptive_risk.snapshot.hazards", None, "required")
        if "operational_time" not in snapshot:
            raise ValidationFailureError("adaptive_risk.snapshot.operational_time", None, "required")

        return self.update_model(
            observations=dict(snapshot["hazards"]),
            operational_time=float(snapshot["operational_time"]),
            timestamp=str(snapshot["timestamp"]) if snapshot.get("timestamp") is not None else None,
            source=source or (str(snapshot["source"]) if snapshot.get("source") is not None else None),
            context=dict(snapshot.get("context", {})),
            store_event=True,
        )


    def simulate_candidate_parameters(self, *,
        learning_rate: Optional[float] = None,
        uncertainty_window: Optional[float] = None,
        observations: Optional[Sequence[Mapping[str, Any] | RiskObservation]] = None,
    ) -> Dict[str, Any]:
        """
        Simulate adaptive-risk behavior under candidate hyperparameters.

        This method is designed as an integration point for external optimizers
        such as Bayesian search without coupling this module directly to a
        specific tuning implementation.
        """
        candidate_learning_rate = (
            self.learning_rate if learning_rate is None else _coerce_probability(
                learning_rate,
                "candidate.learning_rate",
                inclusive_zero=False,
                inclusive_one=True,
            )
        )
        candidate_window = (
            self.uncertainty_window if uncertainty_window is None else _require_positive_float(
                uncertainty_window,
                "candidate.uncertainty_window",
            )
        )

        source_history = observations if observations is not None else list(self.observation_history)
        candidate_states = self._initialize_candidate_states(candidate_window)

        replay_count = 0
        for raw_item in source_history:
            item = raw_item if isinstance(raw_item, RiskObservation) else RiskObservation(
                timestamp=str(raw_item.get("timestamp", _utcnow().isoformat())),
                hazard_counts=dict(raw_item.get("hazard_counts", raw_item.get("hazards", {}))),
                operational_time=float(raw_item.get("operational_time", 0.0)),
                source=str(raw_item["source"]) if isinstance(raw_item, Mapping) and raw_item.get("source") is not None else None,
                context=dict(raw_item.get("context", {})) if isinstance(raw_item, Mapping) else {},
            )
            for hazard, count in item.hazard_counts.items():
                if hazard not in candidate_states:
                    raise ValidationFailureError(
                        "adaptive_risk.simulation.hazard_name",
                        hazard,
                        f"one of {sorted(candidate_states.keys())}",
                    )
                state = candidate_states[hazard]
                adaptive_alpha = state.base_alpha + (1.0 - candidate_learning_rate) * (state.alpha - state.base_alpha)
                adaptive_beta = state.base_beta + (1.0 - candidate_learning_rate) * (state.beta - state.base_beta)
                state.alpha = adaptive_alpha + float(count)
                state.beta = adaptive_beta + item.operational_time
                state.total_occurrences += int(count)
                state.total_operational_time += item.operational_time
                state.update_count += 1
                state.last_updated = item.timestamp
            replay_count += 1

        hazard_summaries = []
        for hazard, state in candidate_states.items():
            incident_probability_24h = 1.0 - math.exp(-state.mean_rate * 24.0)
            hazard_summaries.append(
                {
                    "hazard_id": hazard,
                    "mean_rate": float(state.mean_rate),
                    "variance": float(state.variance),
                    "incident_probability_24h": float(incident_probability_24h),
                    "coefficient_of_variation": float(state.coefficient_of_variation),
                }
            )

        mean_probabilities = [item["incident_probability_24h"] for item in hazard_summaries]
        return {
            "learning_rate": candidate_learning_rate,
            "uncertainty_window": candidate_window,
            "replayed_observations": replay_count,
            "hazards": hazard_summaries,
            "aggregate_mean_rate": float(sum(item["mean_rate"] for item in hazard_summaries)),
            "max_incident_probability_24h": float(max(mean_probabilities)) if mean_probabilities else 0.0,
            "average_incident_probability_24h": float(np.mean(mean_probabilities)) if mean_probabilities else 0.0,
        }

    def build_adaptation_objective(self, *, target: str = "max_incident_probability_24h",
                                   observations: Optional[Sequence[Mapping[str, Any] | RiskObservation]] = None,
                                   ) -> Callable[[float, float], float]:
        """
        Return a callable objective for external optimizers.

        The returned function accepts ``learning_rate`` and
        ``uncertainty_window`` and produces a scalar score suitable for
        minimization.
        """
        normalized_target = _normalize_non_empty_string(target, "target")

        def objective(learning_rate: float, uncertainty_window: float) -> float:
            simulation = self.simulate_candidate_parameters(
                learning_rate=learning_rate,
                uncertainty_window=uncertainty_window,
                observations=observations,
            )
            if normalized_target not in simulation:
                raise ValidationFailureError(
                    "adaptive_risk.objective.target",
                    normalized_target,
                    "simulation output key",
                )
            value = simulation[normalized_target]
            if not isinstance(value, (int, float, np.floating)):
                raise ValidationFailureError(
                    "adaptive_risk.objective.value_type",
                    type(value).__name__,
                    "numeric",
                )
            return float(value)

        return objective

    # ------------------------------------------------------------------
    # Risk inspection
    # ------------------------------------------------------------------
    def get_current_risk(self, hazard: str) -> Dict[str, Any]:
        """Return a structured risk summary for one hazard."""
        hazard_name = _normalize_non_empty_string(hazard, "hazard")
        with self._lock:
            state = self.hazard_states.get(hazard_name)
            if state is None:
                raise ValidationFailureError(
                    "adaptive_risk.hazard_name",
                    hazard_name,
                    f"one of {sorted(self.hazard_states.keys())}",
                )
            history_rates = self._get_historical_rates_locked(hazard_name)
            interval_payload = self._build_interval_payload(state.mean_rate, state.standard_deviation)
            trend = self._calculate_trend(history_rates)

            incident_probability_1h = 1.0 - math.exp(-state.mean_rate * 1.0)
            incident_probability_24h = 1.0 - math.exp(-state.mean_rate * 24.0)
            uncertainty_score = min(1.0, state.coefficient_of_variation)
            risk_band = self._classify_risk_band(state.mean_rate, incident_probability_24h, uncertainty_score)

            return {
                "hazard_id": hazard_name,
                "posterior": {
                    "alpha": float(state.alpha),
                    "beta": float(state.beta),
                },
                "risk_metrics": {
                    "prior_rate": float(state.prior_rate),
                    "current_mean": float(state.mean_rate),
                    "variance": float(state.variance),
                    "standard_deviation": float(state.standard_deviation),
                    "coefficient_of_variation": float(state.coefficient_of_variation),
                    "expected_events_per_1000h": float(state.mean_rate * 1000.0),
                    "incident_probability_1h": float(incident_probability_1h),
                    "incident_probability_24h": float(incident_probability_24h),
                    "credible_intervals": interval_payload,
                },
                "trend_analysis": {
                    "historical_data_points": len(history_rates),
                    "slope": float(trend),
                    "stability": self._classify_trend_stability(trend, state.mean_rate),
                },
                "sotif": self._build_sotif_hazard_summary(state),
                "risk_band": risk_band,
                "total_occurrences": int(state.total_occurrences),
                "total_operational_time": float(state.total_operational_time),
                "last_updated": state.last_updated,
            }

    def get_all_current_risks(self) -> List[Dict[str, Any]]:
        with self._lock:
            hazard_ids = list(self.hazard_states.keys())
        return [self.get_current_risk(hazard) for hazard in hazard_ids]

    def get_system_risk_summary(self) -> Dict[str, Any]:
        risks = self.get_all_current_risks()
        if not risks:
            raise OperationalError("No hazard states are available for system-risk summarization.")

        mean_rates = [item["risk_metrics"]["current_mean"] for item in risks]
        incident_probabilities = [item["risk_metrics"]["incident_probability_24h"] for item in risks]
        degrading_count = sum(1 for item in risks if item["trend_analysis"]["stability"] == "degrading")
        high_risk_count = sum(1 for item in risks if item["risk_band"] in {"high", "critical"})
        uncertainty_scores = [item["sotif"]["uncertainty_score"] for item in risks]

        return {
            "timestamp": _utcnow().isoformat(),
            "hazard_count": len(risks),
            "aggregate_mean_rate": float(sum(mean_rates)),
            "max_hazard_mean_rate": float(max(mean_rates)),
            "max_incident_probability_24h": float(max(incident_probabilities)),
            "average_uncertainty": float(np.mean(uncertainty_scores)) if uncertainty_scores else 0.0,
            "degrading_hazards": int(degrading_count),
            "high_risk_hazards": int(high_risk_count),
            "system_risk_band": self._classify_system_risk(max(incident_probabilities), high_risk_count),
            "operational_hours": float(sum(item.operational_time for item in self.observation_history)),
        }

    # ------------------------------------------------------------------
    # Safety-case and reporting
    # ------------------------------------------------------------------
    def generate_safety_case(self, *, system_name: str = "Autonomous Decision System",
                             include_report_reference: bool = True) -> Dict[str, Any]:
        """Generate an STPA/SOTIF-aligned safety case for the current model state."""
        risks = self.get_all_current_risks()
        system_summary = self.get_system_risk_summary()
        timestamp = _utcnow().isoformat()

        safety_case = {
            "metadata": {
                "system": system_name,
                "version": self._next_safety_case_version(),
                "generation_date": timestamp,
                "standard_compliance": ["STPA", "ISO 21448"],
                "generated_by": "RiskAdaptation",
            },
            "system_description": {
                "purpose": "AI-driven operational decision system",
                "boundary_conditions": "Configured operational design domain",
                "operational_domain": "Dynamic evaluator runtime",
            },
            "control_structure": {
                "controllers": ["Risk Assessment Engine", "Planning Module", "Supervisory Policy Layer"],
                "actuators": ["Decision Interface", "Mitigation Control Hooks"],
                "sensors": ["Operational metrics", "Hazard telemetry", "Runtime event stream"],
                "controlled_processes": ["Hazard-rate adaptation", "Operational risk gating"],
            },
            "safety_requirements": {
                "goals": [
                    {
                        "id": f"SG-{risk['hazard_id'].upper()}",
                        "description": f"Maintain {risk['hazard_id']} within tolerated operational bounds.",
                        "verification_method": "Bayesian operational monitoring and safety-case review",
                        "target_value": min(self.max_risk_level, risk["risk_metrics"]["current_mean"] * 2.0),
                    }
                    for risk in risks
                ],
                "system_constraints": [
                    "Risk adaptation shall remain observable and auditable for all configured hazards.",
                    "Mitigation recommendations shall be generated when hazard risk bands reach high or critical.",
                    "Posterior updates shall be based only on validated operational evidence.",
                ],
            },
            "hazard_analysis": {
                "identified_hazards": [
                    {
                        "id": f"HAZ-{risk['hazard_id'].upper()}",
                        "description": risk["hazard_id"],
                        "current_risk": risk,
                        "mitigation_strategy": self._build_mitigation_strategy(risk),
                        "safety_constraints": [
                            f"Maintain {risk['hazard_id']} mean hazard rate below {max(self.max_risk_level, risk['risk_metrics']['current_mean'] * 1.5):.6g}/hour.",
                            f"Trigger mitigation escalation when incident probability over 24h exceeds {self.max_risk_level:.3f}.",
                        ],
                    }
                    for risk in risks
                ],
            },
            "evidence_base": {
                "operational_history": {
                    "total_operational_hours": float(sum(item.operational_time for item in self.observation_history)),
                    "observation_batches": len(self.observation_history),
                    "observed_events": [
                        {
                            "hazard": risk["hazard_id"],
                            "total_occurrences": int(self.hazard_states[risk["hazard_id"]].total_occurrences),
                            "last_occurrence": self._last_occurrence_timestamp(risk["hazard_id"]),
                        }
                        for risk in risks
                    ],
                },
                "model_parameters": {
                    "learning_rate": self.learning_rate,
                    "uncertainty_window": self.uncertainty_window,
                    "model_version": "bayesian-gamma-poisson-2.0",
                },
            },
            "validation": {
                "assumptions": [
                    "Hazard counts are captured consistently and mapped to configured hazard identifiers.",
                    "Operational exposure time is measured in comparable units across updates.",
                    "Adaptive forgetting reflects the configured learning-rate policy.",
                ],
                "limitations": [
                    "Unknown-unknown failure modes are only indirectly represented through uncertainty.",
                    "Credible intervals use a normal approximation for reporting simplicity.",
                    "Causal discovery is not inferred automatically from observational risk data.",
                ],
            },
            "framework_alignment": {
                "stpa": self._build_stpa_alignment(risks),
                "sotif": self._build_sotif_alignment(risks),
                "system_summary": system_summary,
            },
        }

        try:
            jsonschema.validate(instance=safety_case, schema=SAFETY_CASE_SCHEMA)
        except jsonschema.ValidationError as exc:
            raise DocumentationProcessingError(
                "Safety case validation failed.",
                context={"schema_error": exc.message},
                cause=exc,
            ) from exc

        version_key = f"safety_case_{timestamp.replace(':', '').replace('-', '')}"
        with self._lock:
            self.safety_case_versions[version_key] = safety_case
            self._prune_retained_history_locked()
            while len(self.safety_case_versions) > self.max_safety_case_versions:
                self.safety_case_versions.popitem(last=False)

        self._store_memory_event(
            safety_case,
            tags=["safety_case", "adaptive_risk", "stpa", "sotif"],
            priority="high",
            category="safety_case",
            source="adaptive_risk",
        )

        documentation_path = self._generate_documentation(safety_case)
        if include_report_reference:
            safety_case["metadata"]["documentation_path"] = str(documentation_path)

        return safety_case

    def _generate_documentation(self, safety_case: Mapping[str, Any]) -> Path:
        """Generate Markdown documentation for a safety case."""
        if not isinstance(safety_case, Mapping):
            raise ValidationFailureError(
                "adaptive_risk.safety_case",
                type(safety_case).__name__,
                "mapping",
            )

        version = (
            str((safety_case.get("metadata") or {}).get("version", "unknown"))
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )
        filename = self._docs_directory / f"case_{version}.md"

        hazards = ((safety_case.get("hazard_analysis") or {}).get("identified_hazards") or [])
        content_lines = [
            "# Safety Case Documentation",
            "",
            f"Version: {version}",
            f"Generated: {(safety_case.get('metadata') or {}).get('generation_date', 'unknown')}",
            "",
            "## System Description",
            f"- Purpose: {(safety_case.get('system_description') or {}).get('purpose', 'N/A')}",
            f"- Operational Domain: {(safety_case.get('system_description') or {}).get('operational_domain', 'N/A')}",
            "",
            "## Hazard Analysis",
        ]
        for hazard in hazards:
            current_risk = (hazard.get("current_risk") or {}).get("risk_metrics", {})
            content_lines.extend(
                [
                    f"### {hazard.get('description', 'Unknown Hazard')}",
                    f"- Mean Rate: {current_risk.get('current_mean', 'N/A')}",
                    f"- Incident Probability (24h): {current_risk.get('incident_probability_24h', 'N/A')}",
                    f"- Risk Band: {(hazard.get('current_risk') or {}).get('risk_band', 'N/A')}",
                    "",
                ]
            )

        try:
            filename.write_text("\n".join(content_lines), encoding="utf-8")
        except OSError as exc:
            raise DocumentationProcessingError(
                "Failed to write safety-case documentation.",
                context={"path": str(filename)},
                cause=exc,
            ) from exc

        logger.info("Generated adaptive-risk documentation at %s", filename)
        return filename

    def generate_automated_report(self) -> str:
        """Generate and persist a comprehensive adaptive-risk report."""
        risks = self.get_all_current_risks()
        report = {
            "timestamp": _utcnow().isoformat(),
            "system_summary": self.get_system_risk_summary(),
            "current_risks": risks,
            "recent_safety_cases": self.get_safety_case_history(limit=3),
            "scheduler": self.scheduler.snapshot(),
            "memory_usage": self.memory.get_statistics(),
            "stpa": self._build_stpa_alignment(risks),
            "sotif": self._build_sotif_alignment(risks),
        }

        filename = self._report_directory / f"risk_report_{_utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            filename.write_text(json.dumps(report, indent=2, sort_keys=False, default=str), encoding="utf-8")
        except OSError as exc:
            raise SerializationError("adaptive_risk_report", str(exc), report) from exc

        self._store_memory_event(
            report,
            tags=["automated_report", "adaptive_risk"],
            priority="medium",
            category="risk_report",
            source="adaptive_risk",
        )
        logger.info("Generated adaptive-risk report: %s", filename)
        return str(filename)

    # ------------------------------------------------------------------
    # Persistence and reset
    # ------------------------------------------------------------------
    def export_state(self, path: Optional[str] = None) -> str:
        """Persist the current adaptive-risk state to disk."""
        filename = Path(path) if path else self._state_directory / f"adaptive_risk_state_{_utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "timestamp": _utcnow().isoformat(),
            "learning_rate": self.learning_rate,
            "uncertainty_window": self.uncertainty_window,
            "hazard_states": {hazard: state.to_dict() for hazard, state in self.hazard_states.items()},
            "observation_history": [item.to_dict() for item in self.observation_history],
            "safety_case_versions": list(self.safety_case_versions.values()),
        }
        try:
            filename.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
        except OSError as exc:
            raise SerializationError("adaptive_risk_state", str(exc), payload) from exc
        return str(filename)

    def reset_model(self, retain_last_n: int = 0,
        override_hazard_rates: Optional[Mapping[str, float]] = None,
        reset_reason: str = "manual reset") -> None:
        """
        Reset the Bayesian risk model while optionally retaining recent evidence.
        """
        retain_count = _require_non_negative_int(retain_last_n, "retain_last_n")
        reason = _normalize_non_empty_string(reset_reason, "reset_reason")
        override_rates: Optional[Dict[str, float]] = None
        if override_hazard_rates is not None:
            if not isinstance(override_hazard_rates, Mapping) or not override_hazard_rates:
                raise ValidationFailureError(
                    "adaptive_risk.override_hazard_rates",
                    type(override_hazard_rates).__name__,
                    "non-empty mapping",
                )
            override_rates = {
                _normalize_non_empty_string(hazard, "override_hazard_rates.key"): _require_non_negative_float(
                    value,
                    f"override_hazard_rates.{hazard}",
                )
                for hazard, value in override_hazard_rates.items()
            }

        with self._lock:
            retained_history = list(self.observation_history[-retain_count:]) if retain_count > 0 else []
            if override_rates is not None:
                self.initial_hazard_rates = override_rates
            self.hazard_states = self._initialize_model()
            self.observation_history = []
            if retained_history:
                for item in retained_history:
                    self.update_model(
                        item.hazard_counts,
                        item.operational_time,
                        timestamp=item.timestamp,
                        source=item.source,
                        context=item.context,
                        store_event=False,
                    )

        self._store_memory_event(
            {
                "event": "risk_model_reset",
                "timestamp": _utcnow().isoformat(),
                "reason": reason,
                "retained_observations": len(retained_history),
                "initial_hazard_rates": dict(self.initial_hazard_rates),
            },
            tags=["risk_reset", "system_event"],
            priority="medium",
            category="risk_reset",
            source="adaptive_risk",
        )
        logger.info("Risk model reset completed: reason=%s retained=%d", reason, len(retained_history))

    # ------------------------------------------------------------------
    # History and summaries
    # ------------------------------------------------------------------
    def get_safety_case_history(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        versions = list(self.safety_case_versions.values())
        if limit is None:
            return versions[-self.max_safety_case_versions :]
        limit_value = _require_positive_int(limit, "limit")
        return versions[-limit_value:]

    def get_observation_history(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        records = [item.to_dict() for item in self.observation_history]
        if limit is None:
            return records
        limit_value = _require_positive_int(limit, "limit")
        return records[-limit_value:]

    # ------------------------------------------------------------------
    # Internal framework logic
    # ------------------------------------------------------------------
    def _prune_retained_history_locked(self) -> None:
        cutoff = _utcnow() - timedelta(days=self.retention_days)
        self.observation_history = [
            item for item in self.observation_history
            if _coerce_timestamp(item.timestamp) >= cutoff
        ]
        retained_cases: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        for key, value in self.safety_case_versions.items():
            generated = str((value.get("metadata") or {}).get("generation_date", ""))
            try:
                timestamp = _coerce_timestamp(generated)
            except Exception:
                retained_cases[key] = value
                continue
            if timestamp >= cutoff:
                retained_cases[key] = value
        self.safety_case_versions = retained_cases


    def _build_stpa_alignment(self, risks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        unsafe_control_actions: List[Dict[str, Any]] = []
        for risk in risks:
            hazard = str(risk["hazard_id"])
            mean_rate = float((risk.get("risk_metrics") or {}).get("current_mean", 0.0))
            incident_probability = float((risk.get("risk_metrics") or {}).get("incident_probability_24h", 0.0))
            if mean_rate > self.max_risk_level or incident_probability > self.max_risk_level:
                unsafe_control_actions.append(
                    {
                        "hazard": hazard,
                        "controller": "Risk Assessment Engine",
                        "unsafe_control_action": "Risk escalation not issued despite elevated posterior hazard rate.",
                        "loss_scenario": f"Operational decisions continue while {hazard} risk remains elevated.",
                    }
                )

        return {
            "unsafe_control_actions": unsafe_control_actions,
            "control_constraints": [
                "Controllers shall escalate mitigation when posterior risk exceeds configured tolerance.",
                "Controllers shall preserve traceability for each adaptive update.",
                "Controllers shall prevent stale hazard priors from masking new operational evidence.",
            ],
            "status": "attention_required" if unsafe_control_actions else "nominal",
        }

    def _build_sotif_alignment(self, risks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        known_performance_limitations: List[Dict[str, Any]] = []
        emerging_uncertainties: List[Dict[str, Any]] = []

        for risk in risks:
            hazard = str(risk["hazard_id"])
            uncertainty_score = float((risk.get("sotif") or {}).get("uncertainty_score", 0.0))
            trend = str((risk.get("trend_analysis") or {}).get("stability", "stable"))
            incident_probability = float((risk.get("risk_metrics") or {}).get("incident_probability_24h", 0.0))

            if incident_probability >= self.max_risk_level:
                known_performance_limitations.append(
                    {
                        "hazard": hazard,
                        "issue": "Observed hazard likelihood exceeds operational tolerance.",
                        "incident_probability_24h": incident_probability,
                    }
                )
            if uncertainty_score >= (1.0 - self.min_safety_margin) or trend == "degrading":
                emerging_uncertainties.append(
                    {
                        "hazard": hazard,
                        "issue": "Model uncertainty or trend suggests an emerging SOTIF concern.",
                        "uncertainty_score": uncertainty_score,
                        "trend": trend,
                    }
                )

        return {
            "known_performance_limitations": known_performance_limitations,
            "emerging_uncertainties": emerging_uncertainties,
            "status": "attention_required" if (known_performance_limitations or emerging_uncertainties) else "nominal",
        }

    def _build_sotif_hazard_summary(self, state: HazardBayesianState) -> Dict[str, Any]:
        uncertainty_score = min(1.0, state.coefficient_of_variation)
        monitored_limitation = state.mean_rate >= self.max_risk_level
        return {
            "uncertainty_score": float(uncertainty_score),
            "known_performance_limitation": bool(monitored_limitation),
            "safety_margin": float(max(0.0, 1.0 - uncertainty_score)),
            "attention_required": bool(monitored_limitation or uncertainty_score >= (1.0 - self.min_safety_margin)),
        }

    def _build_mitigation_strategy(self, risk: Mapping[str, Any]) -> Dict[str, Any]:
        incident_probability = float((risk.get("risk_metrics") or {}).get("incident_probability_24h", 0.0))
        trend_status = str((risk.get("trend_analysis") or {}).get("stability", "stable"))
        actions = [
            "Continue operational monitoring with posterior update logging.",
            "Preserve hazard evidence in evaluator memory for auditability.",
        ]
        if incident_probability >= self.max_risk_level:
            actions.append("Escalate operational mitigation and route to supervisory review.")
        if trend_status == "degrading":
            actions.append("Increase observation review cadence and investigate causal drift.")
        return {
            "status": "escalated" if len(actions) > 2 else "monitoring",
            "actions": actions,
        }

    def _build_interval_payload(self, mean: float, std_dev: float) -> Dict[str, Tuple[float, float]]:
        z_scores = {"90%": 1.645, "95%": 1.96, "99%": 2.576}
        intervals: Dict[str, Tuple[float, float]] = {}
        for label, z_value in z_scores.items():
            lower = max(0.0, mean - z_value * std_dev)
            upper = max(lower, mean + z_value * std_dev)
            intervals[label] = (float(lower), float(upper))
        return intervals

    def _calculate_trend(self, historical_rates: Sequence[float]) -> float:
        if len(historical_rates) < 2:
            return 0.0
        try:
            slope = np.polyfit(range(len(historical_rates)), historical_rates, 1)[0]
        except np.linalg.LinAlgError as exc:
            raise StatisticalAnalysisError(
                "risk_trend_estimation",
                list(historical_rates),
                f"trend estimation failed: {exc}",
            ) from exc
        return float(slope)

    def _classify_trend_stability(self, trend: float, current_mean: float) -> str:
        tolerance = max(1e-9, current_mean * 0.02)
        if trend < -tolerance:
            return "improving"
        if trend > tolerance:
            return "degrading"
        return "stable"

    def _classify_risk_band(self, mean_rate: float, incident_probability_24h: float, uncertainty_score: float) -> str:
        combined = max(mean_rate, incident_probability_24h, uncertainty_score * 0.5)
        if combined >= 0.75:
            return "critical"
        if combined >= 0.35:
            return "high"
        if combined >= 0.10:
            return "moderate"
        return "low"

    def _classify_system_risk(self, max_incident_probability: float, high_risk_hazards: int) -> str:
        if max_incident_probability >= 0.75 or high_risk_hazards >= 2:
            return "critical"
        if max_incident_probability >= self.max_risk_level or high_risk_hazards >= 1:
            return "high"
        if max_incident_probability >= 0.10:
            return "moderate"
        return "low"

    def _get_historical_rates_locked(self, hazard: str) -> List[float]:
        rates: List[float] = []
        for item in self.observation_history:
            count = item.hazard_counts.get(hazard, 0)
            rate = float(count) / item.operational_time if item.operational_time > 0 else 0.0
            rates.append(rate)
        return rates

    def _last_occurrence_timestamp(self, hazard: str) -> str:
        for item in reversed(self.observation_history):
            if item.hazard_counts.get(hazard, 0) > 0:
                return item.timestamp
        return "Never"

    def _next_safety_case_version(self) -> str:
        major = 2
        minor = len(self.safety_case_versions) + 1
        return f"{major}.{minor}"

    def _store_memory_event(self, payload: Mapping[str, Any], *,
                            tags: Sequence[str], priority: str, category: str, source: str,
                            ) -> None:
        try:
            self.memory.add(
                dict(payload),
                tags=list(tags),
                priority=priority,
                category=category,
                source=source,
            )
        except MemoryAccessError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MemoryAccessError("add", category, str(exc)) from exc


    def _initialize_candidate_states(self, uncertainty_window: float) -> Dict[str, HazardBayesianState]:
        states: Dict[str, HazardBayesianState] = {}
        for hazard, rate in self.initial_hazard_rates.items():
            base_beta = max(uncertainty_window, 1.0)
            base_alpha = max(rate * base_beta, 1e-9)
            states[hazard] = HazardBayesianState(
                hazard_id=hazard,
                prior_rate=rate,
                base_alpha=base_alpha,
                base_beta=base_beta,
                alpha=base_alpha,
                beta=base_beta,
            )
        return states

    def _resolve_output_dir(self, configured_path: str) -> Path:
        raw = _normalize_non_empty_string(configured_path, "configured_path")
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
        config_file = self.config.get("__config_path__")
        if not config_file:
            return candidate
        return Path(config_file).resolve().parent / candidate

    # ------------------------------------------------------------------
    # Compatibility / convenience
    # ------------------------------------------------------------------
    @property
    def risk_model(self) -> Dict[str, Tuple[float, float]]:
        """
        Compatibility view exposing the current risk model as hazard -> (mean, variance).
        """
        return {
            hazard: (state.mean_rate, state.variance)
            for hazard, state in self.hazard_states.items()
        }


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValidationFailureError("timestamp", value, "non-empty ISO-8601 string")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValidationFailureError("timestamp", value, "valid ISO-8601 string") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationFailureError(field_name, value, "non-empty string")
    return value.strip()


def _normalize_string_list(value: Any, field_name: str) -> List[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValidationFailureError(field_name, type(value).__name__, "sequence of strings")
    normalized: List[str] = []
    seen: set[str] = set()
    for item in value:
        text = _normalize_non_empty_string(item, field_name)
        key = text.casefold()
        if key not in seen:
            normalized.append(text)
            seen.add(key)
    if not normalized:
        raise ValidationFailureError(field_name, value, "non-empty sequence of strings")
    return normalized


def _require_positive_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigLoadError("<config>", field_name, f"Expected a positive float, got {value!r}") from exc
    if number <= 0:
        raise ConfigLoadError("<config>", field_name, f"Expected a positive float, got {number!r}")
    return number


def _require_non_negative_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigLoadError("<config>", field_name, f"Expected a non-negative float, got {value!r}") from exc
    if number < 0:
        raise ConfigLoadError("<config>", field_name, f"Expected a non-negative float, got {number!r}")
    return number


def _require_positive_int(value: Any, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigLoadError("<config>", field_name, f"Expected a positive integer, got {value!r}") from exc
    if number <= 0:
        raise ConfigLoadError("<config>", field_name, f"Expected a positive integer, got {number!r}")
    return number


def _require_non_negative_int(value: Any, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationFailureError(field_name, value, "non-negative integer") from exc
    if number < 0:
        raise ValidationFailureError(field_name, value, "non-negative integer")
    return number


def _coerce_probability(value: Any, field_name: str, *, inclusive_zero: bool, inclusive_one: bool) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigLoadError("<config>", field_name, f"Expected a probability, got {value!r}") from exc

    lower_ok = number > 0.0 or (inclusive_zero and number == 0.0)
    upper_ok = number < 1.0 or (inclusive_one and number == 1.0)
    if not (lower_ok and upper_ok):
        comparator = (
            "[0, 1]" if inclusive_zero and inclusive_one else
            "(0, 1]" if not inclusive_zero and inclusive_one else
            "[0, 1)" if inclusive_zero and not inclusive_one else
            "(0, 1)"
        )
        raise ConfigLoadError("<config>", field_name, f"Expected probability in {comparator}, got {number!r}")
    return number


if __name__ == "__main__":
    print("\n=== Running Adaptive Risk ===\n")
    risk = RiskAdaptation(auto_start_scheduler=False)

    risk.update_model(
        {"system_failure": 1, "sensor_failure": 0, "unexpected_behavior": 2},
        operational_time=120.0,
        source="demo",
        context={"mode": "test"},
    )
    safety_case = risk.generate_safety_case()
    report_path = risk.generate_automated_report()

    print(json.dumps(risk.get_system_risk_summary(), indent=2))
    print(json.dumps(safety_case, indent=2))
    print(f"report_path={report_path}")

    exported = risk.export_state()
    print(f"state_path={exported}")

    risk.shutdown()
    print("\n=== Successfully Ran Adaptive Risk ===\n")
