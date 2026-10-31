"""Pareto dominance, front ranking, archive management, and front metrics.

The semantics follow Miettinen (1998): feasibility is handled before objective
comparison, and feasible Pareto dominance means no worse in every objective and
strictly better in at least one.  Crowding-distance ideas follow Deb et al.
(2002), but the abstraction is algorithm-independent and does not equate
multi-objective optimization with NSGA-II.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import threading
import numpy as np # type: ignore

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import*
from ..utils.optimization_helpers import *
from ..optimization_types import Objective, ObjectiveSense, ParetoFront, ParetoPoint
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Pareto Efficiency")
printer = PrettyPrinter()

SELECTION_STRATEGIES = ("compromise", "weighted_sum", "chebyshev", "knee", "hypervolume")
TRUNCATION_STRATEGIES = ("crowding", "hypervolume")


@dataclass(frozen=True, slots=True)
class ParetoSettings:
    """Validated runtime settings for Pareto analysis and archive management."""

    dominance_tolerance: float = 1e-9
    feasibility_tolerance: float = 1e-8
    archive_limit: int = 512
    archive_truncation: str = "crowding"
    hypervolume_samples: int = 20_000
    hypervolume_exact_max_points: int = 60
    metrics_max_front: int = 1_000
    max_candidates: int = 20_000
    seed: int = 0
    default_strategy: str = "compromise"

    def __post_init__(self) -> None:
        for name in ("dominance_tolerance", "feasibility_tolerance"):
            value = require_finite(getattr(self, name), name)
            if value < 0:
                raise OptimizationConfigurationError(f"{name} must be non-negative", context={"value": value})
            object.__setattr__(self, name, value)
        for name in ("archive_limit", "hypervolume_samples", "hypervolume_exact_max_points", "metrics_max_front", "max_candidates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise OptimizationConfigurationError(f"{name} must be a positive integer", context={"value": value})
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise OptimizationConfigurationError("seed must be a non-negative integer", context={"value": self.seed})
        if self.archive_truncation not in TRUNCATION_STRATEGIES:
            raise OptimizationConfigurationError(
                "archive_truncation is unsupported",
                context={"value": self.archive_truncation, "allowed": TRUNCATION_STRATEGIES},
            )
        if self.default_strategy not in SELECTION_STRATEGIES:
            raise OptimizationConfigurationError(
                "default_strategy is unsupported",
                context={"value": self.default_strategy, "allowed": SELECTION_STRATEGIES},
            )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "ParetoSettings":
        raw = dict(config or {})
        aliases = {
            "tolerance": "dominance_tolerance",
            "default_tolerance": "dominance_tolerance",
            "max_archive_size": "archive_limit",
            "hv_samples": "hypervolume_samples",
            "hv_exact_max_points": "hypervolume_exact_max_points",
        }
        normalized = {aliases.get(key, key): value for key, value in raw.items()}
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in normalized.items() if key in known})


@dataclass(frozen=True, slots=True)
class Candidate:
    values: Mapping[str, Any]
    objectives: Mapping[str, float]
    feasible: bool = True
    violation: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        violation = require_finite(self.violation, "violation")
        if violation < 0:
            raise OptimizationValidationError("candidate violation must be non-negative")
        normalized: dict[str, float] = {}
        for name, value in self.objectives.items():
            if not isinstance(name, str) or not name:
                raise OptimizationValidationError("candidate objective names must be non-empty")
            normalized[name] = require_finite(value, f"objective[{name}]")
        object.__setattr__(self, "objectives", normalized)
        object.__setattr__(self, "violation", violation)
        if self.candidate_id is None:
            object.__setattr__(self, "candidate_id", stable_fingerprint({"values": dict(self.values), "objectives": normalized})[:20])

    @property
    def id(self) -> str:
        """Backward-compatible candidate identifier."""
        return str(self.candidate_id)

    @property
    def variables(self) -> Mapping[str, Any]:
        """Backward-compatible alias for decision-variable values."""
        return self.values

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "values": dict(self.values),
            "objectives": dict(self.objectives),
            "feasible": self.feasible,
            "violation": self.violation,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ParetoMetrics:
    candidate_count: int
    feasible_count: int
    front_size: int
    front_count: int
    spacing: float | None = None
    hypervolume: float | None = None
    hypervolume_method: str | None = None
    reference_point: tuple[float, ...] | None = None

    @property
    def n_candidates(self) -> int:
        return self.candidate_count

    @property
    def n_feasible(self) -> int:
        return self.feasible_count

    @property
    def n_fronts(self) -> int:
        return self.front_count

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True, slots=True)
class ParetoResult:
    candidates: tuple[Candidate, ...]
    fronts: tuple[tuple[Candidate, ...], ...]
    objective_names: tuple[str, ...]
    metrics: ParetoMetrics
    approximate: bool = False
    objective_senses: tuple[ObjectiveSense, ...] = ()

    @property
    def front(self) -> tuple[Candidate, ...]:
        return self.fronts[0] if self.fronts else ()

    @property
    def all_candidates(self) -> tuple[Candidate, ...]:
        return tuple(self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "fronts": [[candidate.to_dict() for candidate in front] for front in self.fronts],
            "objective_names": list(self.objective_names),
            "objective_senses": [sense.value for sense in self.objective_senses],
            "metrics": self.metrics.to_dict(),
            "approximate": self.approximate,
        }

    def to_front(self, *, algorithm: str | None = None) -> ParetoFront:
        return ParetoFront(
            points=tuple(
                ParetoPoint(c.values, c.objectives, c.feasible, c.violation, c.metadata)
                for c in self.front
            ),
            objective_names=self.objective_names,
            approximate=self.approximate,
            algorithm=algorithm,
            metadata={
                "candidate_count": self.metrics.candidate_count,
                "front_count": self.metrics.front_count,
                "hypervolume": self.metrics.hypervolume,
                "spacing": self.metrics.spacing,
            },
        )


@dataclass(frozen=True, slots=True)
class ArchiveUpdate:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    archive: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class FrontComparison:
    previous_size: int
    current_size: int
    current_dominates_previous_fraction: float
    previous_dominates_current_fraction: float


class ParetoEfficiency:
    def __init__(self, config: Mapping[str, Any] | None = None, template: Mapping[str, Any] | None = None) -> None:
        global_config = load_global_config()
        cfg = dict(get_config_section("multiobjective", config=global_config) or {})
        if template:
            # Templates are data only.  Pareto settings may be supplied by a caller
            # through ``config``; template values only seed data-level defaults.
            for key in ("reference_point", "limits"):
                if key in template:
                    cfg[key] = template[key]
        if config:
            cfg.update(dict(config))
        self.settings = ParetoSettings.from_mapping(cfg)
        self.tolerance = self.settings.dominance_tolerance
        self.feasibility_tolerance = self.settings.feasibility_tolerance
        self.archive_limit = self.settings.archive_limit
        self.hypervolume_samples = self.settings.hypervolume_samples
        self.seed = self.settings.seed
        self._lock = threading.RLock()
        self._archive: tuple[Candidate, ...] = ()
        self._archive_objectives: tuple[Objective, ...] = ()

    @staticmethod
    def _coerce_objectives(objectives: Sequence[Objective] | Mapping[str, Any]) -> tuple[Objective, ...]:
        if isinstance(objectives, Mapping):
            result_list: list[Objective] = []
            for name, raw in objectives.items():
                if isinstance(raw, Objective):
                    result_list.append(raw)
                    continue
                if isinstance(raw, Mapping):
                    sense = raw.get("sense", raw.get("direction", "min"))
                else:
                    sense = raw
                # A zero callback is only a structural carrier for Pareto
                # direction metadata; it is never executed by this module.
                result_list.append(Objective(name=str(name), sense=ObjectiveSense.coerce(sense), function=lambda _: 0.0))
            return tuple(result_list)
        result = tuple(objectives)
        if not result or any(not isinstance(item, Objective) for item in result):
            raise OptimizationValidationError("objectives must contain Objective instances")
        return result

    @staticmethod
    def _coerce_candidate(raw: Candidate | ParetoPoint | Mapping[str, Any]) -> Candidate:
        if isinstance(raw, Candidate):
            return raw
        if isinstance(raw, ParetoPoint):
            return Candidate(raw.decision, raw.objective_values, raw.feasible, raw.violation, raw.metadata)
        if not isinstance(raw, Mapping):
            raise OptimizationValidationError("candidate must be Candidate, ParetoPoint, or mapping")
        objectives = raw.get("objectives", raw.get("objective_values"))
        values = raw.get("values", raw.get("decision", {}))
        if not isinstance(objectives, Mapping) or not isinstance(values, Mapping):
            raise OptimizationValidationError("candidate mapping requires values/decision and objectives/objective_values")
        violation = require_finite(raw.get("violation", 0.0), "violation")
        return Candidate(
            dict(values),
            {str(k): require_finite(v, str(k)) for k, v in objectives.items()},
            bool(raw.get("feasible", violation <= 0.0)),
            violation,
            dict(raw.get("metadata", {})) if isinstance(raw.get("metadata", {}), Mapping) else {},
            raw.get("candidate_id"),
        )

    def _min_vector(self, candidate: Candidate, objectives: Sequence[Objective]) -> tuple[float, ...]:
        values: list[float] = []
        for objective in objectives:
            if objective.name not in candidate.objectives:
                raise MultiObjectiveError("candidate is missing an objective", context={"objective": objective.name})
            value = candidate.objectives[objective.name]
            values.append(value if objective.sense is ObjectiveSense.MINIMIZE else -value)
        return tuple(values)

    def weakly_dominates(
        self,
        a: Candidate | ParetoPoint | Mapping[str, Any],
        b: Candidate | ParetoPoint | Mapping[str, Any],
        objectives: Sequence[Objective] | Mapping[str, Any],
    ) -> bool:
        """Return weak Pareto dominance (no objective worse, equality allowed)."""
        left = self._coerce_candidate(a)
        right = self._coerce_candidate(b)
        specs = self._coerce_objectives(objectives)
        if left.feasible and not right.feasible:
            return True
        if right.feasible and not left.feasible:
            return False
        if not left.feasible and not right.feasible:
            return left.violation <= right.violation + self.tolerance
        lv = self._min_vector(left, specs)
        rv = self._min_vector(right, specs)
        return all(x <= y + self.tolerance for x, y in zip(lv, rv))

    def strictly_dominates(
        self,
        a: Candidate | ParetoPoint | Mapping[str, Any],
        b: Candidate | ParetoPoint | Mapping[str, Any],
        objectives: Sequence[Objective] | Mapping[str, Any],
    ) -> bool:
        """Return strict Pareto dominance (strictly better in every objective)."""
        left = self._coerce_candidate(a)
        right = self._coerce_candidate(b)
        specs = self._coerce_objectives(objectives)
        if left.feasible and not right.feasible:
            return True
        if right.feasible and not left.feasible:
            return False
        if not left.feasible and not right.feasible:
            return left.violation < right.violation - self.tolerance
        lv = self._min_vector(left, specs)
        rv = self._min_vector(right, specs)
        return all(x < y - self.tolerance for x, y in zip(lv, rv))

    def dominates(
        self,
        a: Candidate | ParetoPoint | Mapping[str, Any],
        b: Candidate | ParetoPoint | Mapping[str, Any],
        objectives: Sequence[Objective] | Mapping[str, Any],
    ) -> bool:
        """Return ordinary Pareto dominance: weakly better and better in at least one objective."""
        left = self._coerce_candidate(a)
        right = self._coerce_candidate(b)
        specs = self._coerce_objectives(objectives)
        if left.feasible and not right.feasible:
            return True
        if right.feasible and not left.feasible:
            return False
        if not left.feasible and not right.feasible:
            return left.violation < right.violation - self.tolerance
        lv = self._min_vector(left, specs)
        rv = self._min_vector(right, specs)
        weak = all(x <= y + self.tolerance for x, y in zip(lv, rv))
        better_somewhere = any(x < y - self.tolerance for x, y in zip(lv, rv))
        return weak and better_somewhere

    def equivalent(
        self,
        a: Candidate,
        b: Candidate,
        objectives: Sequence[Objective],
    ) -> bool:
        if a.feasible != b.feasible:
            return False
        if abs(a.violation - b.violation) > self.tolerance:
            return False
        av = self._min_vector(a, objectives)
        bv = self._min_vector(b, objectives)
        return all(abs(x - y) <= self.tolerance for x, y in zip(av, bv))

    def _deduplicate(self, candidates: Sequence[Candidate], objectives: Sequence[Objective]) -> list[Candidate]:
        unique: list[Candidate] = []
        for candidate in candidates:
            if not any(self.equivalent(candidate, existing, objectives) for existing in unique):
                unique.append(candidate)
        return unique

    def non_dominated(
        self,
        candidates: Sequence[Candidate | ParetoPoint | Mapping[str, Any]],
        objectives: Sequence[Objective] | Mapping[str, Any],
    ) -> list[Candidate]:
        specs = self._coerce_objectives(objectives)
        items = self._deduplicate([self._coerce_candidate(item) for item in candidates], specs)
        result: list[Candidate] = []
        for i, candidate in enumerate(items):
            if any(j != i and self.dominates(other, candidate, specs) for j, other in enumerate(items)):
                continue
            result.append(candidate)
        return result

    def pareto_improvements(
        self,
        candidate: Candidate | ParetoPoint | Mapping[str, Any],
        pool: Sequence[Candidate | ParetoPoint | Mapping[str, Any]],
        objectives: Sequence[Objective] | Mapping[str, Any],
        *,
        limit: int | None = None,
    ) -> list[Candidate]:
        """Return members of ``pool`` that dominate ``candidate``.

        Results are ordered by normalized Euclidean distance from the target,
        making the smallest available Pareto improvement appear first.
        """
        specs = self._coerce_objectives(objectives)
        target = self._coerce_candidate(candidate)
        target_vector = np.asarray(self._min_vector(target, specs), dtype=float)
        improvements = [
            self._coerce_candidate(item)
            for item in pool
            if self.dominates(self._coerce_candidate(item), target, specs)
        ]
        if not improvements:
            return []
        vectors = np.asarray([self._min_vector(item, specs) for item in improvements] + [tuple(target_vector)], dtype=float)
        lo = np.min(vectors, axis=0)
        hi = np.max(vectors, axis=0)
        span = np.where(np.abs(hi - lo) <= self.tolerance, 1.0, hi - lo)
        target_norm = (target_vector - lo) / span
        improvements.sort(
            key=lambda item: float(np.linalg.norm((np.asarray(self._min_vector(item, specs)) - lo) / span - target_norm))
        )
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise OptimizationValidationError("limit must be a non-negative integer")
            improvements = improvements[:limit]
        return improvements

    def sort_fronts(
        self,
        candidates: Sequence[Candidate | ParetoPoint | Mapping[str, Any]],
        objectives: Sequence[Objective] | Mapping[str, Any],
    ) -> tuple[tuple[Candidate, ...], ...]:
        specs = self._coerce_objectives(objectives)
        items = self._deduplicate([self._coerce_candidate(item) for item in candidates], specs)
        dominates_set: list[list[int]] = [[] for _ in items]
        domination_count = [0] * len(items)
        first: list[int] = []
        for i, left in enumerate(items):
            for j, right in enumerate(items):
                if i == j:
                    continue
                if self.dominates(left, right, specs):
                    dominates_set[i].append(j)
                elif self.dominates(right, left, specs):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                first.append(i)
        fronts: list[list[int]] = []
        current = first
        while current:
            fronts.append(current)
            nxt: list[int] = []
            for i in current:
                for j in dominates_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        nxt.append(j)
            current = nxt
        return tuple(tuple(items[i] for i in front) for front in fronts)

    def crowding_distance(self, front: Sequence[Candidate], objectives: Sequence[Objective]) -> dict[str, float]:
        if not front:
            return {}
        distances = {candidate.candidate_id or str(i): 0.0 for i, candidate in enumerate(front)}
        if len(front) <= 2:
            return {key: math.inf for key in distances}
        for objective in objectives:
            ordered = sorted(front, key=lambda c: self._min_vector(c, (objective,))[0])
            keys = [candidate.candidate_id or str(i) for i, candidate in enumerate(ordered)]
            distances[keys[0]] = math.inf
            distances[keys[-1]] = math.inf
            values = [self._min_vector(candidate, (objective,))[0] for candidate in ordered]
            span = values[-1] - values[0]
            if abs(span) <= self.tolerance:
                continue
            for i in range(1, len(ordered) - 1):
                key = ordered[i].candidate_id or str(i)
                if not math.isinf(distances[key]):
                    distances[key] += (values[i + 1] - values[i - 1]) / span
        return distances

    def spacing(self, front: Sequence[Candidate], objectives: Sequence[Objective]) -> float | None:
        if len(front) < 2:
            return None
        vectors = [np.asarray(self._min_vector(candidate, objectives), dtype=float) for candidate in front]
        nearest: list[float] = []
        for i, vector in enumerate(vectors):
            distances = [float(np.linalg.norm(vector - other, ord=1)) for j, other in enumerate(vectors) if i != j]
            nearest.append(min(distances))
        mean = sum(nearest) / len(nearest)
        if len(nearest) == 1:
            return 0.0
        return math.sqrt(sum((distance - mean) ** 2 for distance in nearest) / (len(nearest) - 1))

    def hypervolume(
        self,
        front: Sequence[Candidate],
        objectives: Sequence[Objective],
        *,
        reference_point: Sequence[float] | None = None,
    ) -> tuple[float | None, str | None, tuple[float, ...] | None]:
        feasible = [candidate for candidate in front if candidate.feasible]
        if not feasible:
            return None, None, None
        vectors = np.asarray([self._min_vector(candidate, objectives) for candidate in feasible], dtype=float)
        if reference_point is None:
            worst = np.max(vectors, axis=0)
            best = np.min(vectors, axis=0)
            margin = np.maximum(np.abs(worst - best) * 0.1, 1.0)
            reference = worst + margin
        else:
            reference = np.asarray(reference_point, dtype=float)
            if reference.shape != (len(objectives),) or not np.all(np.isfinite(reference)):
                raise OptimizationValidationError("hypervolume reference point has invalid dimension/value")
            if np.any(reference < np.max(vectors, axis=0) - self.tolerance):
                raise OptimizationValidationError("hypervolume reference point must be weakly worse than all front points")
        if vectors.shape[1] == 2:
            ordered = vectors[np.argsort(vectors[:, 0])]
            hv = 0.0
            best_second = reference[1]
            for x, y in ordered:
                if y < best_second:
                    hv += max(0.0, reference[0] - x) * max(0.0, best_second - y)
                    best_second = y
            return float(hv), "exact_2d", tuple(float(x) for x in reference)

        lower = np.min(vectors, axis=0)
        box = reference - lower
        if np.any(box <= 0):
            return 0.0, "monte_carlo", tuple(float(x) for x in reference)
        rng = np.random.default_rng(self.seed)
        samples = rng.uniform(lower, reference, size=(self.hypervolume_samples, vectors.shape[1]))
        dominated = np.zeros(self.hypervolume_samples, dtype=bool)
        for vector in vectors:
            dominated |= np.all(vector <= samples + self.tolerance, axis=1)
        volume = float(np.prod(box))
        estimate = volume * float(np.mean(dominated))
        return estimate, "monte_carlo", tuple(float(x) for x in reference)

    def analyze(
        self,
        candidates: Sequence[Candidate | ParetoPoint | Mapping[str, Any]],
        objectives: Sequence[Objective] | Mapping[str, Any],
        *,
        approximate: bool = False,
        reference_point: Sequence[float] | None = None,
    ) -> ParetoResult:
        specs = self._coerce_objectives(objectives)
        coerced = tuple(self._coerce_candidate(item) for item in candidates)
        fronts = self.sort_fronts(coerced, specs)
        first = fronts[0] if fronts else ()
        hypervolume, hv_method, reference = self.hypervolume(first, specs, reference_point=reference_point)
        metrics = ParetoMetrics(
            candidate_count=len(coerced),
            feasible_count=sum(1 for candidate in coerced if candidate.feasible),
            front_size=len(first),
            front_count=len(fronts),
            spacing=self.spacing(first, specs),
            hypervolume=hypervolume,
            hypervolume_method=hv_method,
            reference_point=reference,
        )
        return ParetoResult(
            coerced,
            fronts,
            tuple(objective.name for objective in specs),
            metrics,
            approximate,
            tuple(objective.sense for objective in specs),
        )

    @staticmethod
    def _selection_weights(objectives: Sequence[Objective], override: Mapping[str, float] | None) -> np.ndarray:
        raw = []
        for objective in objectives:
            value = objective.weight if override is None else override.get(objective.name, objective.weight)
            weight = require_finite(value, f"weight[{objective.name}]")
            if weight < 0:
                raise OptimizationValidationError("selection weights must be non-negative")
            raw.append(weight)
        weights = np.asarray(raw, dtype=float)
        total = float(np.sum(weights))
        if total <= 0:
            raise OptimizationValidationError("at least one selection weight must be positive")
        return weights / total

    def _eligible_front(
        self,
        candidates: Sequence[Candidate],
        objectives: Sequence[Objective],
        limits: Mapping[str, float] | None,
    ) -> list[Candidate]:
        if not limits:
            return self.non_dominated(candidates, objectives)
        known = {objective.name for objective in objectives}
        unknown = set(limits) - known
        if unknown:
            raise OptimizationValidationError("limits reference unknown objectives", context={"unknown": sorted(unknown)})
        eligible: list[Candidate] = []
        by_name = {objective.name: objective for objective in objectives}
        for candidate in candidates:
            if not candidate.feasible:
                continue
            accepted = True
            for name, raw_limit in limits.items():
                limit_value = require_finite(raw_limit, f"limit[{name}]")
                value = candidate.objectives[name]
                objective = by_name[name]
                if objective.sense is ObjectiveSense.MINIMIZE and value > limit_value + self.tolerance:
                    accepted = False
                    break
                if objective.sense is ObjectiveSense.MAXIMIZE and value < limit_value - self.tolerance:
                    accepted = False
                    break
            if accepted:
                eligible.append(candidate)
        return self.non_dominated(eligible, objectives) if eligible else []

    def score(
        self,
        source: ParetoResult | Sequence[Candidate | ParetoPoint | Mapping[str, Any]],
        objectives: Sequence[Objective] | Mapping[str, Any] | None = None,
        *,
        strategy: str | None = None,
        weights: Mapping[str, float] | None = None,
        limits: Mapping[str, float] | None = None,
        reference_point: Sequence[float] | None = None,
    ) -> list[tuple[Candidate, float]]:
        """Rank eligible non-dominated candidates for decision support.

        Pareto dominance remains the primary semantics.  These scalar scores are
        only tie-breaking/decision-support views over an already non-dominated
        set; they never redefine Pareto optimality.
        """
        if isinstance(source, ParetoResult):
            raw_candidates = source.candidates
            if objectives is None:
                if len(source.objective_senses) != len(source.objective_names):
                    raise OptimizationValidationError(
                        "objectives are required for a ParetoResult that does not retain objective directions"
                    )
                specs = tuple(
                    Objective(name, sense, function=lambda _: 0.0)
                    for name, sense in zip(source.objective_names, source.objective_senses)
                )
            else:
                specs = self._coerce_objectives(objectives)
        else:
            if objectives is None:
                raise OptimizationValidationError("objectives are required when scoring raw candidates")
            specs = self._coerce_objectives(objectives)
            raw_candidates = tuple(self._coerce_candidate(item) for item in source)

        selected_strategy = (strategy or self.settings.default_strategy).strip().lower()
        if selected_strategy not in SELECTION_STRATEGIES:
            raise OptimizationValidationError(
                "unknown Pareto selection strategy",
                context={"strategy": selected_strategy, "allowed": SELECTION_STRATEGIES},
            )
        front = self._eligible_front(list(raw_candidates), specs, limits)
        if not front:
            return []
        vectors = np.asarray([self._min_vector(candidate, specs) for candidate in front], dtype=float)
        ideal = np.min(vectors, axis=0)
        nadir = np.max(vectors, axis=0)
        span = np.where(np.abs(nadir - ideal) <= self.tolerance, 1.0, nadir - ideal)
        normalized = (vectors - ideal) / span
        normalized_weights = self._selection_weights(specs, weights)

        if selected_strategy == "weighted_sum":
            scores = normalized @ normalized_weights
        elif selected_strategy == "chebyshev":
            weighted = normalized * normalized_weights
            scores = np.max(weighted, axis=1) + 1e-4 * np.sum(weighted, axis=1)
        elif selected_strategy == "compromise":
            scores = np.sqrt(np.sum((normalized * normalized_weights) ** 2, axis=1))
        elif selected_strategy == "knee":
            # A balanced knee is near the ideal while avoiding a solution that
            # sacrifices one objective disproportionately.  The max component
            # captures the worst normalized regret; distance resolves ties.
            weighted = normalized * normalized_weights
            scores = np.max(weighted, axis=1) + 0.05 * np.sqrt(np.sum(weighted**2, axis=1))
        else:  # hypervolume contribution; larger contribution is better.
            _, _, ref = self.hypervolume(front, specs, reference_point=reference_point)
            if ref is None:
                return []
            total = self.hypervolume(front, specs, reference_point=ref)[0] or 0.0
            contributions = []
            for index in range(len(front)):
                reduced = front[:index] + front[index + 1 :]
                value = self.hypervolume(reduced, specs, reference_point=ref)[0] or 0.0
                contributions.append(max(0.0, total - value))
            scores = -np.asarray(contributions, dtype=float)

        ranked = sorted(
            ((candidate, float(score_value)) for candidate, score_value in zip(front, scores)),
            key=lambda item: (item[1], item[0].candidate_id or ""),
        )
        return ranked

    def select(
        self,
        source: ParetoResult | Sequence[Candidate | ParetoPoint | Mapping[str, Any]],
        objectives: Sequence[Objective] | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Candidate | None:
        ranked = self.score(source, objectives, **kwargs)
        return ranked[0][0] if ranked else None

    def archive(self) -> tuple[Candidate, ...]:
        with self._lock:
            return tuple(self._archive)

    def update_archive(
        self,
        candidates: Sequence[Candidate | ParetoPoint | Mapping[str, Any]],
        objectives: Sequence[Objective] | Mapping[str, Any],
    ) -> ArchiveUpdate:
        specs = self._coerce_objectives(objectives)
        signature = tuple((objective.name, objective.sense.value) for objective in specs)
        with self._lock:
            existing_signature = tuple((objective.name, objective.sense.value) for objective in self._archive_objectives)
            if self._archive and existing_signature != signature:
                raise OptimizationValidationError(
                    "archive already contains a different objective system; reset it before reuse",
                    context={"stored": existing_signature, "requested": signature},
                )
            before = {candidate.candidate_id for candidate in self._archive}
            incoming = [self._coerce_candidate(candidate) for candidate in candidates]
            incoming_ids = [candidate.candidate_id for candidate in incoming]
            if len(incoming_ids) != len(set(incoming_ids)):
                raise OptimizationValidationError("incoming archive update contains duplicate candidate identifiers")
            # Incoming identifiers replace earlier snapshots of the same decision.
            replacement_ids = set(incoming_ids)
            retained = [candidate for candidate in self._archive if candidate.candidate_id not in replacement_ids]
            merged = self.non_dominated((*retained, *incoming), specs)
            if len(merged) > self.archive_limit:
                if self.settings.archive_truncation == "hypervolume":
                    base_hv, _, reference = self.hypervolume(merged, specs)
                    if reference is None:
                        contributions = {candidate.candidate_id: 0.0 for candidate in merged}
                    else:
                        contributions = {}
                        total = base_hv or 0.0
                        for index, candidate in enumerate(merged):
                            reduced = merged[:index] + merged[index + 1 :]
                            hv_without = self.hypervolume(reduced, specs, reference_point=reference)[0] or 0.0
                            contributions[candidate.candidate_id] = max(0.0, total - hv_without)
                    merged = sorted(
                        merged,
                        key=lambda candidate: (contributions.get(candidate.candidate_id, 0.0), candidate.candidate_id or ""),
                        reverse=True,
                    )[: self.archive_limit]
                else:
                    distances = self.crowding_distance(merged, specs)
                    merged = sorted(
                        merged,
                        key=lambda candidate: (
                            math.isinf(distances.get(candidate.candidate_id or "", 0.0)),
                            distances.get(candidate.candidate_id or "", 0.0),
                            candidate.candidate_id or "",
                        ),
                        reverse=True,
                    )[: self.archive_limit]
            self._archive = tuple(merged)
            self._archive_objectives = tuple(specs)
            after = {candidate.candidate_id for candidate in self._archive}
            return ArchiveUpdate(
                added=tuple(sorted(str(item) for item in after - before)),
                removed=tuple(sorted(str(item) for item in before - after)),
                archive=self._archive,
            )

    def compare_fronts(
        self,
        previous: Sequence[Candidate | ParetoPoint | Mapping[str, Any]],
        current: Sequence[Candidate | ParetoPoint | Mapping[str, Any]],
        objectives: Sequence[Objective] | Mapping[str, Any],
    ) -> FrontComparison:
        specs = self._coerce_objectives(objectives)
        prev = self.non_dominated(previous, specs)
        curr = self.non_dominated(current, specs)

        def coverage(by: Sequence[Candidate], target: Sequence[Candidate]) -> float:
            if not target:
                return 0.0
            covered = sum(1 for candidate in target if any(self.dominates(other, candidate, specs) for other in by))
            return covered / len(target)

        return FrontComparison(len(prev), len(curr), coverage(curr, prev), coverage(prev, curr))

    def reset_archive(self) -> None:
        with self._lock:
            self._archive = ()
            self._archive_objectives = ()

    def export_archive(self) -> dict[str, Any]:
        """Return a JSON-safe archive snapshot for persistence or agent memory."""
        with self._lock:
            return {
                "version": __version__,
                "objectives": [
                    {
                        "name": objective.name,
                        "sense": objective.sense.value,
                        "weight": objective.weight,
                        "priority": objective.priority,
                    }
                    for objective in self._archive_objectives
                ],
                "candidates": [candidate.to_dict() for candidate in self._archive],
            }

    def import_archive(
        self,
        data: Mapping[str, Any],
        objectives: Sequence[Objective] | Mapping[str, Any] | None = None,
    ) -> ArchiveUpdate:
        """Replace the in-memory archive from ``export_archive`` output."""
        if not isinstance(data, Mapping):
            raise OptimizationValidationError("archive snapshot must be a mapping")
        raw_candidates = data.get("candidates")
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
            raise OptimizationValidationError("archive snapshot requires a candidate sequence")
        if objectives is None:
            raw_objectives = data.get("objectives")
            if not isinstance(raw_objectives, Sequence) or isinstance(raw_objectives, (str, bytes)) or not raw_objectives:
                raise OptimizationValidationError("archive snapshot requires objective metadata")
            specs = tuple(
                Objective(
                    str(item["name"]),
                    ObjectiveSense.coerce(item.get("sense", "minimize")),
                    function=lambda _: 0.0,
                    weight=item.get("weight", 1.0),
                    priority=item.get("priority", 0),
                )
                for item in raw_objectives
                if isinstance(item, Mapping) and "name" in item
            )
            if len(specs) != len(raw_objectives):
                raise OptimizationValidationError("archive objective metadata is malformed")
        else:
            specs = self._coerce_objectives(objectives)
        candidates: list[Candidate] = []
        for item in raw_candidates:
            if not isinstance(item, Mapping):
                raise OptimizationValidationError("archive candidate record must be a mapping")
            candidates.append(
                Candidate(
                    values=dict(item.get("values", item.get("variables", {}))),
                    objectives=dict(item.get("objectives", {})),
                    feasible=bool(item.get("feasible", True)),
                    violation=item.get("violation", 0.0),
                    metadata=dict(item.get("metadata", {})),
                    candidate_id=item.get("candidate_id", item.get("id")),
                )
            )
        self.reset_archive()
        return self.update_archive(candidates, specs)

    def report(self, result: ParetoResult, *, max_rows: int = 20) -> None:
        """Log a bounded human-readable summary without changing structured output."""
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
            raise OptimizationValidationError("max_rows must be a positive integer")
        preview = [candidate.to_dict() for candidate in result.front[:max_rows]]
        logger.info(
            "Pareto result | candidates=%d | feasible=%d | front=%d | fronts=%d | hv=%s | preview=%s",
            result.metrics.candidate_count,
            result.metrics.feasible_count,
            result.metrics.front_size,
            result.metrics.front_count,
            result.metrics.hypervolume,
            printer.pformat(preview),
        )


__all__ = [
    "ArchiveUpdate",
    "Candidate",
    "FrontComparison",
    "ParetoEfficiency",
    "ParetoMetrics",
    "ParetoResult",
    "ParetoSettings",
    "SELECTION_STRATEGIES",
    "TRUNCATION_STRATEGIES",
]
