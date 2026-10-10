"""
Pareto efficiency engine for the Optimization agent.

sources:
- Pareto Optimal

Core concepts:
- Optimal Solution: A choice where no other available option can improve one criterion without harming another.
- Pareto Frontier (Pareto Front): The set or curve of all Pareto-optimal trade-off solutions plotted in objective space.
- Dominance: Solution A "dominates" solution B if A is strictly better than B in at least one objective and no worse in all others.
- Pareto Improvement: A change that makes at least one objective better off without making any other objective worse off.

Real-World Applications
- Engineering & Design: Balancing conflicting factors like minimizing manufacturing cost while maximizing product strength or fuel efficiency.
- Economics & Welfare: Distributing goods or resources so that no individual can be made better off without hurting someone else.
- Power Systems & Logistics: Optimizing delivery routes, energy production costs, or emissions reduction using tools like evolutionary algorithms.

What this module provides
- Dominance (Deb's constrained dominance: feasible beats infeasible, smaller violation beats larger).
- Non-dominated filtering and full non-dominated sorting into ranked fronts.
- Crowding distance for diversity within a front.
- Front quality metrics: hypervolume (exact, Monte Carlo fallback), normalised hypervolume, spacing.
- Front comparison between optimisation iterations (coverage + hypervolume + verdict).
- Decision support: pick a solution from the front (compromise, weighted sum, Chebyshev, knee, hypervolume
  contribution), optionally under per-objective limits (epsilon-constraint).
- Pareto improvements: which candidates would make a given candidate strictly better off.
- A bounded, thread-safe, non-dominated archive that persists across agent iterations.

Conventions
- Internally every objective is converted to *minimisation space* (maximised objectives are negated).
  All public inputs and outputs are in the caller's original units.
- ``tolerance`` (per objective or ``default_tolerance``) bins values onto a grid of that width before
  dominance is evaluated. Binning keeps dominance transitive, which a pairwise "within epsilon" test does not.
- Errors come from ``optimization_errors`` and numeric/validation primitives from ``optimization_helpers``;
  this module only consumes them.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import random
import threading
import time
import uuid

from dataclasses import dataclass, field, fields
from itertools import chain
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import *
from ..utils.optimization_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Pareto Efficiency")
printer = PrettyPrinter()

MINIMIZE = "min"
MAXIMIZE = "max"

SELECTION_STRATEGIES: Tuple[str, ...] = ("compromise", "weighted_sum", "chebyshev", "knee", "hypervolume")
TRUNCATION_STRATEGIES: Tuple[str, ...] = ("crowding", "hypervolume")

_DIRECTION_ALIASES: Dict[str, str] = {
    "min": MINIMIZE, "minimize": MINIMIZE, "minimise": MINIMIZE, "minimum": MINIMIZE, "lower": MINIMIZE,
    "max": MAXIMIZE, "maximize": MAXIMIZE, "maximise": MAXIMIZE, "maximum": MAXIMIZE, "higher": MAXIMIZE,
}
_RESERVED_KEYS = frozenset({"id", "objectives", "variables", "violation", "metadata"})
_TEMPLATE_KEYS = frozenset({"objectives", "limits", "reference_point"})
_CHEBYSHEV_RHO = 1e-4  # augmentation term: keeps Chebyshev picks Pareto-optimal rather than weakly optimal

CandidateLike = Union["Candidate", Mapping[str, Any], Sequence[float]]
ObjectivesLike = Union[None, Mapping[str, Any], Sequence[Any]]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Objective:
    """One optimisation criterion.

    ``lower``/``upper`` are optional natural bounds of the objective. They anchor normalisation
    (ideal/nadir) and, for hypervolume, the worst bound doubles as the reference point.
    ``tolerance`` overrides ``default_tolerance`` for this objective.
    """

    name: str
    direction: str = MINIMIZE
    weight: float = 1.0
    lower: Optional[float] = None
    upper: Optional[float] = None
    tolerance: Optional[float] = None

    def __post_init__(self) -> None:
        name = self.name.strip() if isinstance(self.name, str) else ""
        if not name:
            raise OptimizationValidationError("Objective name must be a non-empty string", context={"name": self.name})
        direction = _DIRECTION_ALIASES.get(str(self.direction).strip().lower())
        if direction is None:
            raise OptimizationValidationError(
                f"Objective '{name}' has unknown direction {self.direction!r}",
                context={"allowed": sorted(_DIRECTION_ALIASES)},
            )
        weight = to_finite_float(self.weight)
        if weight is None or weight <= 0:
            raise OptimizationValidationError(f"Objective '{name}' weight must be a finite number > 0", context={"weight": self.weight})
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "weight", weight)
        for attr in ("lower", "upper", "tolerance"):
            raw = getattr(self, attr)
            if raw is None:
                continue
            value = to_finite_float(raw)
            if value is None or (attr == "tolerance" and value < 0):
                raise OptimizationValidationError(f"Objective '{name}' {attr} must be a finite number" + (" >= 0" if attr == "tolerance" else ""), context={attr: raw})
            object.__setattr__(self, attr, value)
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise OptimizationValidationError(f"Objective '{name}' requires lower < upper", context={"lower": self.lower, "upper": self.upper})

    @property
    def sign(self) -> float:
        """Multiplier that maps original units into minimisation space."""
        return 1.0 if self.direction == MINIMIZE else -1.0

    @property
    def best_bound(self) -> Optional[float]:
        """Best natural bound, in minimisation space."""
        if self.direction == MINIMIZE:
            return self.lower
        return None if self.upper is None else -self.upper

    @property
    def worst_bound(self) -> Optional[float]:
        """Worst natural bound, in minimisation space."""
        if self.direction == MINIMIZE:
            return self.upper
        return None if self.lower is None else -self.lower

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "direction": self.direction, "weight": self.weight,
                "lower": self.lower, "upper": self.upper, "tolerance": self.tolerance}


@dataclass(frozen=True, eq=False)
class Candidate:
    """A solution evaluated on the objectives.

    ``violation`` is the aggregate constraint violation (0 = feasible). ``variables`` carries the decision
    variables and ``metadata`` anything the caller wants to round-trip; neither influences dominance.
    """

    id: str
    objectives: Mapping[str, float]
    variables: Mapping[str, Any] = field(default_factory=dict)
    violation: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "objectives": dict(self.objectives), "variables": dict(self.variables),
                "violation": self.violation, "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class ParetoSettings:
    """Runtime settings, read from the ``pareto_efficiency`` config section."""

    default_tolerance: float = 0.0          # grid width used to bin objective values (0 = exact)
    feasibility_tolerance: float = 1e-9     # violations at or below this count as feasible
    max_fronts: Optional[int] = None        # stop non-dominated sorting after this many fronts
    deduplicate: bool = False               # drop candidates whose binned objectives + violation repeat
    max_candidates: int = 20000             # guard for the O(N * front) dominance work
    max_archive_size: int = 200
    archive_truncation: str = "crowding"
    reference_margin: float = 0.1           # fraction of the observed range added beyond the nadir
    hv_exact_max_points: int = 60           # above this (and > 2 objectives) hypervolume is Monte Carlo
    hv_samples: int = 20000
    metrics_max_front: int = 1000           # skip hypervolume/spacing for larger fronts
    seed: int = 0
    default_strategy: str = "compromise"

    def __post_init__(self) -> None:
        def is_int(v: Any, minimum: int) -> bool:
            return isinstance(v, int) and not isinstance(v, bool) and v >= minimum

        def is_num(v: Any, minimum: float = 0.0) -> bool:
            number = to_finite_float(v)
            return number is not None and number >= minimum

        checks = (
            ("default_tolerance", is_num(self.default_tolerance), "a number >= 0"),
            ("feasibility_tolerance", is_num(self.feasibility_tolerance), "a number >= 0"),
            ("max_fronts", self.max_fronts is None or is_int(self.max_fronts, 1), "null or an integer >= 1"),
            ("deduplicate", isinstance(self.deduplicate, bool), "a boolean"),
            ("max_candidates", is_int(self.max_candidates, 1), "an integer >= 1"),
            ("max_archive_size", is_int(self.max_archive_size, 1), "an integer >= 1"),
            ("archive_truncation", self.archive_truncation in TRUNCATION_STRATEGIES, f"one of {TRUNCATION_STRATEGIES}"),
            ("reference_margin", is_num(self.reference_margin), "a number >= 0"),
            ("hv_exact_max_points", is_int(self.hv_exact_max_points, 1), "an integer >= 1"),
            ("hv_samples", is_int(self.hv_samples, 1), "an integer >= 1"),
            ("metrics_max_front", is_int(self.metrics_max_front, 1), "an integer >= 1"),
            ("seed", is_int(self.seed, 0), "an integer >= 0"),
            ("default_strategy", self.default_strategy in SELECTION_STRATEGIES, f"one of {SELECTION_STRATEGIES}"),
        )
        for name, ok, expected in checks:
            if not ok:
                raise OptimizationConfigurationError(
                    f"pareto_efficiency.{name} must be {expected}", context={"value": getattr(self, name)})

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ParetoSettings":
        known = {f.name for f in fields(cls)}
        unknown = set(config) - known - _TEMPLATE_KEYS
        if unknown:
            logger.warning("Ignoring unknown pareto_efficiency settings: %s", sorted(unknown))
        return cls(**{key: config[key] for key in known if key in config})


@dataclass(frozen=True)
class ParetoMetrics:
    n_candidates: int
    n_feasible: int
    front_size: int
    n_fronts: int
    front_feasible: bool                          # False only when no candidate satisfies the constraints
    ideal: Mapping[str, float]                    # best value per objective on the first front (original units)
    nadir: Mapping[str, float]                    # worst value per objective on the first front
    reference_point: Optional[Mapping[str, float]]
    hypervolume: Optional[float]
    hypervolume_normalised: Optional[float]       # hypervolume / volume of the box [ideal, reference]
    hypervolume_method: Optional[str]             # "exact" or "monte_carlo"
    spacing: Optional[float]                      # Schott spacing; 0 = perfectly even

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class ParetoResult:
    run_id: str
    objectives: Tuple[Objective, ...]
    fronts: Tuple[Tuple[Candidate, ...], ...]     # fronts[0] is the Pareto front, each ordered by objective 1, best first
    unranked: Tuple[Candidate, ...]               # candidates beyond ``max_fronts``
    duplicates: Tuple[Candidate, ...]             # removed by ``deduplicate``
    ranks: Mapping[str, Optional[int]]            # id -> front index (None when unranked)
    crowding: Mapping[str, float]                 # id -> crowding distance within its front (inf = boundary)
    metrics: ParetoMetrics
    strategy: str
    recommended: Optional[Candidate]
    recommended_score: Optional[float]            # lower is better
    elapsed_ms: float

    @property
    def front(self) -> Tuple[Candidate, ...]:
        return self.fronts[0] if self.fronts else ()

    @property
    def candidates(self) -> Tuple[Candidate, ...]:
        """Every analysed candidate (ranked and unranked), excluding removed duplicates."""
        return tuple(chain.from_iterable(self.fronts)) + self.unranked

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view (non-finite floats become None)."""
        return {
            "run_id": self.run_id,
            "objectives": [o.to_dict() for o in self.objectives],
            "fronts": [[c.to_dict() for c in front] for front in self.fronts],
            "unranked": [c.to_dict() for c in self.unranked],
            "duplicates": [c.to_dict() for c in self.duplicates],
            "ranks": dict(self.ranks),
            "crowding": {k: to_finite_float(v) for k, v in self.crowding.items()},
            "metrics": self.metrics.to_dict(),
            "strategy": self.strategy,
            "recommended": self.recommended.to_dict() if self.recommended else None,
            "recommended_score": self.recommended_score,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class ArchiveUpdate:
    accepted: Tuple[str, ...]                     # incoming ids now in the archive
    rejected: Mapping[str, str]                   # incoming id -> "dominated" | "duplicate" | "truncated"
    evicted: Tuple[str, ...]                      # previous archive ids that were removed
    size: int


@dataclass(frozen=True)
class FrontComparison:
    previous_size: int
    current_size: int
    previous_covers_current: float                # share of current points weakly dominated by the previous front
    current_covers_previous: float
    previous_hypervolume: Optional[float]
    current_hypervolume: Optional[float]
    hypervolume_delta: Optional[float]            # current - previous, on a shared reference point
    reference_point: Optional[Mapping[str, float]]
    verdict: str                                  # "improved" | "regressed" | "equivalent" | "incomparable"


@dataclass
class _Frame:
    """Prepared analysis input; all vectors share the order of ``objectives``."""

    candidates: List[Candidate]
    objectives: Tuple[Objective, ...]
    vectors: List[Tuple[float, ...]]              # minimisation space, exact values
    keys: List[Tuple[float, ...]]                 # minimisation space, binned values used for dominance
    violations: List[float]                       # 0.0 means feasible


@dataclass(frozen=True)
class _Hypervolume:
    value: float
    normalised: Optional[float]
    method: str
    reference: Tuple[float, ...]                  # minimisation space


# ---------------------------------------------------------------------------
# Input coercion (module-level: pure, no engine state)
# ---------------------------------------------------------------------------
def _build_objective(spec: Mapping[str, Any]) -> Objective:
    try:
        return Objective(**dict(spec))
    except TypeError as exc:
        raise OptimizationValidationError(
            f"Invalid objective specification: {exc}", context={"spec": dict(spec), "allowed": [f.name for f in fields(Objective)]}) from exc


def _coerce_objectives(raw: ObjectivesLike) -> Tuple[Objective, ...]:
    """Accepts Objective items, ``{name: "min"|"max"}``, ``{name: {...spec}}``, ``[{name, ...}]`` or ``[(name, dir)]``."""
    if raw is None:
        return ()
    items: List[Objective] = []
    if isinstance(raw, Mapping):
        for name, spec in raw.items():
            if isinstance(spec, Objective):
                items.append(spec)
            elif isinstance(spec, str):
                items.append(Objective(name=name, direction=spec))
            elif isinstance(spec, Mapping):
                items.append(_build_objective({**spec, "name": name}))
            else:
                raise OptimizationValidationError(f"Objective '{name}' must be a direction string or a mapping", context={"spec": spec})
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for spec in raw:
            if isinstance(spec, Objective):
                items.append(spec)
            elif isinstance(spec, Mapping):
                items.append(_build_objective(spec))
            elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and len(spec) == 2:
                items.append(Objective(name=spec[0], direction=spec[1]))
            else:
                raise OptimizationValidationError("Each objective must be an Objective, a mapping or a (name, direction) pair", context={"spec": spec})
    else:
        raise OptimizationValidationError("objectives must be a mapping or a sequence", context={"type": type(raw).__name__})
    names = [o.name for o in items]
    if len(set(names)) != len(names):
        raise OptimizationValidationError("Objective names must be unique", context={"names": names})
    return tuple(items)


def _coerce_value_map(raw: Optional[Mapping[str, Any]], label: str) -> Dict[str, float]:
    """Validate a ``{objective_name: number}`` mapping (limits, reference point)."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise OptimizationValidationError(f"{label} must be a mapping of objective name to number", context={"type": type(raw).__name__})
    out: Dict[str, float] = {}
    for name, value in raw.items():
        number = to_finite_float(value)
        if number is None:
            raise OptimizationValidationError(f"{label}['{name}'] must be a finite number", context={"value": value})
        out[str(name)] = number
    return out


def _new_id() -> str:
    return f"cand-{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class ParetoEfficiency:
    """Pareto dominance, ranking, metrics, decision support and archiving for multi-objective optimisation.

    Configuration precedence (lowest to highest): global ``pareto_efficiency`` section, ``template``, ``config``.
    ``template`` may only contain ``objectives``, ``limits`` and ``reference_point`` (the problem definition);
    anything else raises ``OptimizationTemplateError``. ``config`` may additionally override any
    ``ParetoSettings`` field.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None, template: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.pe_config = dict(get_config_section("pareto_efficiency", config=self.config) or {})
        if template:
            self.pe_config.update(self._parse_template(template))
        if config:
            self.pe_config.update(dict(config))

        self.settings = ParetoSettings.from_config(self.pe_config)
        self.objectives: Tuple[Objective, ...] = _coerce_objectives(self.pe_config.get("objectives"))
        self.limits: Dict[str, float] = _coerce_value_map(self.pe_config.get("limits"), "limits")
        self.reference_point: Dict[str, float] = _coerce_value_map(self.pe_config.get("reference_point"), "reference_point")

        self._lock = threading.RLock()
        self._archive: List[Candidate] = []
        self._archive_objectives: Tuple[Objective, ...] = ()
        logger.debug("ParetoEfficiency initialised (%d default objectives)", len(self.objectives))

    # ------------------------------------------------------------------ public: dominance
    def dominates(self, a: CandidateLike, b: CandidateLike, objectives: ObjectivesLike = None) -> bool:
        """True if ``a`` dominates ``b`` (constrained dominance, honouring tolerances)."""
        objs = self._resolve_objectives(objectives)
        _, key_a, viol_a = self._encode(self._coerce_candidate(a, objs), objs)
        _, key_b, viol_b = self._encode(self._coerce_candidate(b, objs), objs)
        return self._dominates_key(key_a, viol_a, key_b, viol_b)

    def non_dominated(self, candidates: Sequence[CandidateLike], objectives: ObjectivesLike = None) -> List[Candidate]:
        """The first Pareto front only (cheaper than ``analyze``). Ordered by the first objective, best first."""
        frame, _ = self._prepare(candidates, self._resolve_objectives(objectives))
        return [frame.candidates[i] for i in self._nondominated(frame.keys, None, frame.violations)]

    def pareto_improvements(self, candidate: CandidateLike, pool: Sequence[CandidateLike],
                            objectives: ObjectivesLike = None, *, limit: Optional[int] = None) -> List[Candidate]:
        """Members of ``pool`` that dominate ``candidate`` (i.e. available Pareto improvements).

        Sorted by normalised distance, so the smallest change comes first. An empty list means ``candidate``
        is not dominated by anything in ``pool``.
        """
        objs = self._resolve_objectives(objectives)
        target = self._coerce_candidate(candidate, objs)
        others = [c for c in (self._coerce_candidate(r, objs) for r in self._as_list(pool)) if c.id != target.id]
        frame, _ = self._prepare([target] + others, objs, deduplicate=False)
        dominating = [i for i in range(1, len(frame.candidates))
                      if self._dominates_key(frame.keys[i], frame.violations[i], frame.keys[0], frame.violations[0])]
        ideal, nadir = self._extent(frame.vectors, objs)
        origin = self._normalise(frame.vectors[0], ideal, nadir)
        dominating.sort(key=lambda i: math.dist(self._normalise(frame.vectors[i], ideal, nadir), origin))
        if limit is not None:
            dominating = dominating[:max(0, limit)]
        return [frame.candidates[i] for i in dominating]

    # ------------------------------------------------------------------ public: full analysis
    def analyze(self, candidates: Sequence[CandidateLike], objectives: ObjectivesLike = None, *,
                reference_point: Optional[Mapping[str, float]] = None, strategy: Optional[str] = None,
                weights: Optional[Mapping[str, float]] = None, limits: Optional[Mapping[str, float]] = None,
                max_fronts: Optional[int] = None, compute_metrics: bool = True) -> ParetoResult:
        """Rank candidates into fronts, compute diversity/quality metrics and recommend one solution.

        ``limits`` ({objective: worst acceptable value}) restricts the *recommendation* only; ranking and
        metrics always cover every candidate. ``weights`` overrides ``Objective.weight`` for the recommendation.
        """
        started = time.perf_counter()
        objs = self._resolve_objectives(objectives)
        strategy = self._resolve_strategy(strategy)
        depth = max_fronts if max_fronts is not None else self.settings.max_fronts
        if depth is not None and (not isinstance(depth, int) or isinstance(depth, bool) or depth < 1):
            raise OptimizationValidationError("max_fronts must be an integer >= 1", context={"max_fronts": depth})
        frame, duplicates = self._prepare(candidates, objs)
        run_id = uuid.uuid4().hex[:12]

        if not frame.candidates:
            logger.debug("analyze(%s): no candidates", run_id)
            return self._build_result(run_id, objs, frame, [], [], duplicates, {}, strategy, [], started,
                                      self._empty_metrics(0, 0))

        fronts, unranked = self._sort_fronts(frame, depth)
        crowding: Dict[int, float] = {}
        for members in fronts:
            crowding.update(self._crowding(frame.vectors, members))

        front0 = fronts[0]
        front_feasible = frame.violations[front0[0]] == 0.0
        n_feasible = sum(1 for v in frame.violations if v == 0.0)
        if not front_feasible:
            logger.warning("analyze(%s): no candidate satisfies the constraints; front 0 holds the least-violating candidates", run_id)

        metrics = self._build_metrics(frame, fronts, front_feasible, n_feasible, reference_point, compute_metrics)
        ranked = self._score_eligible(frame, strategy, weights, limits, reference_point)
        result = self._build_result(run_id, objs, frame, fronts, unranked, duplicates, crowding, strategy, ranked, started, metrics)
        logger.debug("analyze(%s): n=%d fronts=%d front0=%d hv=%s %.1f ms", run_id, len(frame.candidates), len(fronts),
                     len(front0), metrics.hypervolume, result.elapsed_ms)
        return result

    # ------------------------------------------------------------------ public: decision support
    def score(self, source: Union[ParetoResult, Sequence[CandidateLike]], objectives: ObjectivesLike = None, *,
              strategy: Optional[str] = None, weights: Optional[Mapping[str, float]] = None,
              limits: Optional[Mapping[str, float]] = None,
              reference_point: Optional[Mapping[str, float]] = None) -> List[Tuple[Candidate, float]]:
        """Eligible Pareto-optimal candidates ordered best first as ``(candidate, score)`` (lower is better).

        ``source`` is a candidate collection or a previous ``ParetoResult``. Limits are applied *before* the
        front is taken, so a dominated candidate can surface when the limits exclude everything that dominates it.
        """
        if isinstance(source, ParetoResult):
            objs, raw = (self._resolve_objectives(objectives) if objectives is not None else source.objectives), source.candidates
        else:
            objs, raw = self._resolve_objectives(objectives), source
        frame, _ = self._prepare(raw, objs)
        ranked = self._score_eligible(frame, self._resolve_strategy(strategy), weights, limits, reference_point)
        return [(frame.candidates[i], s) for i, s in ranked]

    def select(self, source: Union[ParetoResult, Sequence[CandidateLike]], objectives: ObjectivesLike = None, **kwargs: Any) -> Optional[Candidate]:
        """Best candidate under ``strategy``/``weights``/``limits`` (see ``score``); None when nothing is eligible."""
        ranked = self.score(source, objectives, **kwargs)
        if not ranked:
            logger.warning("select(): no feasible candidate satisfies the requested limits")
            return None
        return ranked[0][0]

    # ------------------------------------------------------------------ public: metrics
    def hypervolume(self, candidates: Sequence[CandidateLike], objectives: ObjectivesLike = None, *,
                    reference_point: Optional[Mapping[str, float]] = None) -> float:
        """Hypervolume dominated by the feasible Pareto front (0.0 when no candidate is feasible)."""
        frame, _ = self._prepare(candidates, self._resolve_objectives(objectives))
        members = self._feasible_front(frame)
        if not members:
            return 0.0
        return self._measure_hypervolume(frame.objectives, [frame.vectors[i] for i in members],
                                         [v for v, viol in zip(frame.vectors, frame.violations) if viol == 0.0],
                                         reference_point).value

    def compare_fronts(self, previous: Sequence[CandidateLike], current: Sequence[CandidateLike],
                       objectives: ObjectivesLike = None, *,
                       reference_point: Optional[Mapping[str, float]] = None) -> FrontComparison:
        """Did ``current`` make progress over ``previous``? Uses coverage and hypervolume on a shared reference.

        Verdict: ``improved`` (current covers all of previous but not vice versa), ``regressed`` (the reverse),
        ``equivalent`` (mutual coverage) or ``incomparable`` (trade-offs on both sides; see the hypervolume delta).
        Candidate ids may overlap between the two sets.
        """
        objs = self._resolve_objectives(objectives)
        sides = []
        for raw in (previous, current):
            frame, _ = self._prepare(raw, objs, deduplicate=False)
            members = self._feasible_front(frame)
            sides.append(([frame.vectors[i] for i in members], [frame.keys[i] for i in members]))
        (vec_p, key_p), (vec_c, key_c) = sides

        def coverage(by: List[Tuple[float, ...]], target: List[Tuple[float, ...]]) -> float:
            if not target:
                return 0.0
            covered = sum(1 for t in target if any(all(x <= y for x, y in zip(b, t)) for b in by))
            return covered / len(target)

        p_over_c, c_over_p = coverage(key_p, key_c), coverage(key_c, key_p)
        if (not key_p and not key_c) or (p_over_c == 1.0 and c_over_p == 1.0):
            verdict = "equivalent"
        elif c_over_p == 1.0:
            verdict = "improved"
        elif p_over_c == 1.0:
            verdict = "regressed"
        else:
            verdict = "incomparable"

        hv_p = hv_c = delta = reference = None
        union = vec_p + vec_c
        if union and max(len(vec_p), len(vec_c)) <= self.settings.metrics_max_front:
            shared = self._reference(union, objs, reference_point)
            hv_p = self._hypervolume(vec_p, shared)[0]
            hv_c = self._hypervolume(vec_c, shared)[0]
            delta = hv_c - hv_p
            reference = self._to_original(shared, objs)
        return FrontComparison(len(vec_p), len(vec_c), p_over_c, c_over_p, hv_p, hv_c, delta, reference, verdict)

    # ------------------------------------------------------------------ public: archive
    @property
    def archive(self) -> Tuple[Candidate, ...]:
        with self._lock:
            return tuple(self._archive)

    def update_archive(self, candidates: Sequence[CandidateLike], objectives: ObjectivesLike = None) -> ArchiveUpdate:
        """Merge candidates into the bounded non-dominated archive.

        An incoming id that already exists replaces the stored candidate. When the archive exceeds
        ``max_archive_size`` the most crowded (or lowest hypervolume contribution) members are removed one by one.
        """
        with self._lock:
            objs = self._resolve_objectives(objectives)
            if self._archive and tuple((o.name, o.direction) for o in objs) != tuple((o.name, o.direction) for o in self._archive_objectives):
                raise OptimizationValidationError("Archive holds candidates for different objectives; call reset_archive() first",
                                                  context={"archive": [o.name for o in self._archive_objectives], "requested": [o.name for o in objs]})
            incoming = [self._coerce_candidate(raw, objs) for raw in self._as_list(candidates)]
            incoming_ids = {c.id for c in incoming}
            if len(incoming_ids) != len(incoming):
                raise OptimizationValidationError("Incoming candidates contain duplicate ids")

            retained = [c for c in self._archive if c.id not in incoming_ids]
            frame, duplicates = self._prepare(retained + incoming, objs)
            front = self._nondominated(frame.keys, None, frame.violations)
            kept = self._truncate(frame, front, self.settings.max_archive_size)

            kept_ids = {frame.candidates[i].id for i in kept}
            front_ids = {frame.candidates[i].id for i in front}
            duplicate_ids = {c.id for c in duplicates}
            rejected = {c.id: ("duplicate" if c.id in duplicate_ids else "truncated" if c.id in front_ids else "dominated")
                        for c in incoming if c.id not in kept_ids}
            update = ArchiveUpdate(
                accepted=tuple(c.id for c in incoming if c.id in kept_ids),
                rejected=rejected,
                evicted=tuple(c.id for c in retained if c.id not in kept_ids),
                size=len(kept),
            )
            self._archive = [frame.candidates[i] for i in kept]
            self._archive_objectives = objs
            logger.debug("archive: +%d -%d rejected=%d size=%d", len(update.accepted), len(update.evicted), len(rejected), update.size)
            return update

    def reset_archive(self) -> None:
        with self._lock:
            self._archive = []
            self._archive_objectives = ()

    def export_archive(self) -> Dict[str, Any]:
        """Serialisable snapshot (objectives + candidates) for agent memory or persistence."""
        with self._lock:
            return {"version": __version__, "objectives": [o.to_dict() for o in self._archive_objectives],
                    "candidates": [c.to_dict() for c in self._archive]}

    def import_archive(self, data: Mapping[str, Any]) -> ArchiveUpdate:
        """Replace the archive with a snapshot produced by ``export_archive``."""
        if not isinstance(data, Mapping) or "candidates" not in data:
            raise OptimizationValidationError("Archive snapshot must be a mapping with a 'candidates' entry")
        objs = _coerce_objectives(data.get("objectives")) or self._resolve_objectives(None)
        with self._lock:
            self.reset_archive()
            return self.update_archive(data["candidates"], objs)

    # ------------------------------------------------------------------ public: reporting
    def report(self, result: ParetoResult, *, max_rows: int = 20) -> None:
        """Print the first front and metrics for humans (the structured result stays the agent-facing output)."""
        names = [o.name for o in result.objectives]
        rows = [[c.id, *(f"{c.objectives[n]:.6g}" for n in names)] for c in result.front[:max_rows]]
        printer.section_header("Pareto efficiency")
        title = f"Pareto front ({len(result.front)} of {result.metrics.n_candidates})"
        printer.table(["id", *names], rows, title=title)
        if len(result.front) > max_rows:
            printer.status("Pareto", f"... {len(result.front) - max_rows} more front members not shown", "info")
        printer.pretty("Pareto metrics", result.metrics.to_dict())
        if result.recommended is not None:
            printer.status("Pareto", f"Recommended ({result.strategy}): {result.recommended.id}", "success")

    # ------------------------------------------------------------------ setup / resolution
    @staticmethod
    def _parse_template(template: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(template, Mapping):
            raise OptimizationTemplateError("Template must be a mapping", context={"type": type(template).__name__})
        unknown = set(template) - _TEMPLATE_KEYS
        if unknown:
            raise OptimizationTemplateError("Template contains unsupported keys", context={"unknown": sorted(unknown), "allowed": sorted(_TEMPLATE_KEYS)})
        try:
            _coerce_objectives(template.get("objectives"))
            _coerce_value_map(template.get("limits"), "limits")
            _coerce_value_map(template.get("reference_point"), "reference_point")
        except OptimizationValidationError as exc:
            raise OptimizationTemplateError(f"Invalid template: {exc}") from exc
        return dict(template)

    def _resolve_objectives(self, objectives: ObjectivesLike) -> Tuple[Objective, ...]:
        objs = _coerce_objectives(objectives) if objectives is not None else self.objectives
        if not objs:
            raise OptimizationValidationError("No objectives available: pass `objectives`, or define them in the pareto_efficiency config or template")
        return objs

    def _resolve_strategy(self, strategy: Optional[str]) -> str:
        strategy = strategy or self.settings.default_strategy
        if strategy not in SELECTION_STRATEGIES:
            raise OptimizationValidationError(f"Unknown strategy '{strategy}'", context={"allowed": SELECTION_STRATEGIES})
        return strategy

    @staticmethod
    def _as_list(raw: Any) -> List[Any]:
        if raw is None:
            return []
        if isinstance(raw, (Mapping, str, bytes)):
            raise OptimizationValidationError("Expected a collection of candidates, not a single record", context={"type": type(raw).__name__})
        return list(raw)

    # ------------------------------------------------------------------ candidate preparation
    @staticmethod
    def _coerce_candidate(raw: CandidateLike, objectives: Tuple[Objective, ...]) -> Candidate:
        """Candidate | mapping (nested ``objectives`` or flat objective keys) | sequence in objective order."""
        if isinstance(raw, Candidate):
            return raw
        if isinstance(raw, Mapping):
            cid = raw.get("id")
            cid = str(cid) if cid not in (None, "") else _new_id()
            if "objectives" in raw:
                values = raw["objectives"]
                if not isinstance(values, Mapping):
                    raise OptimizationValidationError(f"Candidate '{cid}': 'objectives' must be a mapping", context={"type": type(values).__name__})
                values, variables = dict(values), dict(raw.get("variables") or {})
            else:
                names = {o.name for o in objectives}
                values = {n: raw[n] for n in names if n in raw}
                variables = {k: v for k, v in raw.items() if k not in names and k not in _RESERVED_KEYS}
                variables.update(raw.get("variables") or {})
            return Candidate(id=cid, objectives=values, variables=variables, violation=raw.get("violation", 0.0),
                             metadata=dict(raw.get("metadata") or {}))
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if len(raw) != len(objectives):
                raise OptimizationValidationError("Positional candidate must provide one value per objective",
                                                  context={"expected": len(objectives), "received": len(raw)})
            return Candidate(id=_new_id(), objectives={o.name: v for o, v in zip(objectives, raw)})
        raise OptimizationValidationError("Candidate must be a Candidate, a mapping or a sequence of numbers", context={"type": type(raw).__name__})

    def _encode(self, candidate: Candidate, objectives: Tuple[Objective, ...]) -> Tuple[Tuple[float, ...], Tuple[float, ...], float]:
        """Return ``(vector, dominance_key, violation)`` in minimisation space; validates every value."""
        vector: List[float] = []
        key: List[float] = []
        for obj in objectives:
            value = to_finite_float(candidate.objectives.get(obj.name))
            if value is None:
                raise OptimizationValidationError(f"Candidate '{candidate.id}' has no finite value for objective '{obj.name}'",
                                                  context={"candidate": candidate.id, "objective": obj.name})
            signed = obj.sign * value
            tolerance = obj.tolerance if obj.tolerance is not None else self.settings.default_tolerance
            if tolerance > 0:
                bin_index = signed / tolerance
                if not math.isfinite(bin_index):
                    raise OptimizationValidationError(f"Objective '{obj.name}' tolerance is too small for value {value}", context={"tolerance": tolerance})
                key.append(math.floor(bin_index + 0.5))
            else:
                key.append(signed)
            vector.append(signed)
        violation = to_finite_float(candidate.violation)
        if violation is None or violation < 0:
            raise OptimizationValidationError(f"Candidate '{candidate.id}' violation must be a finite number >= 0", context={"violation": candidate.violation})
        if violation <= self.settings.feasibility_tolerance:
            violation = 0.0
        return tuple(vector), tuple(key), violation

    def _prepare(self, raw: Sequence[CandidateLike], objectives: Tuple[Objective, ...], *,
                 deduplicate: Optional[bool] = None) -> Tuple[_Frame, List[Candidate]]:
        items = self._as_list(raw)
        if len(items) > self.settings.max_candidates:
            raise OptimizationValidationError("Too many candidates for one analysis",
                                              context={"received": len(items), "max_candidates": self.settings.max_candidates})
        candidates = [self._coerce_candidate(item, objectives) for item in items]
        seen_ids: set = set()
        for c in candidates:
            if c.id in seen_ids:
                raise OptimizationValidationError(f"Duplicate candidate id '{c.id}'", context={"id": c.id})
            seen_ids.add(c.id)

        frame = _Frame([], objectives, [], [], [])
        duplicates: List[Candidate] = []
        seen_signatures: set = set()
        drop_repeats = self.settings.deduplicate if deduplicate is None else deduplicate
        for c in candidates:
            vector, key, violation = self._encode(c, objectives)
            if drop_repeats:
                signature = (key, violation)
                if signature in seen_signatures:
                    duplicates.append(c)
                    continue
                seen_signatures.add(signature)
            frame.candidates.append(c)
            frame.vectors.append(vector)
            frame.keys.append(key)
            frame.violations.append(violation)
        return frame, duplicates

    # ------------------------------------------------------------------ dominance and sorting
    @staticmethod
    def _dominates_key(ka: Sequence[float], va: float, kb: Sequence[float], vb: float) -> bool:
        """Deb's constrained dominance on binned keys."""
        if va != vb:
            return va < vb                      # lower violation wins, feasible (0.0) beats infeasible
        if va > 0.0:
            return False                        # equal, non-zero violation: incomparable
        strictly_better = False
        for x, y in zip(ka, kb):
            if x > y:
                return False
            if x < y:
                strictly_better = True
        return strictly_better

    @classmethod
    def _nondominated(cls, keys: Sequence[Sequence[float]], indices: Optional[Sequence[int]] = None,
                      violations: Optional[Sequence[float]] = None) -> List[int]:
        """Indices of non-dominated entries, ordered by (violation, key).

        Sorting lexicographically means a later entry can never dominate an earlier one, so each entry only
        has to be checked against the front built so far: O(N * |front|).
        """
        pool = range(len(keys)) if indices is None else indices
        viol = violations if violations is not None else [0.0] * len(keys)
        front: List[int] = []
        for i in sorted(pool, key=lambda idx: (viol[idx], keys[idx])):
            if not any(cls._dominates_key(keys[j], viol[j], keys[i], viol[i]) for j in front):
                front.append(i)
        return front

    def _sort_fronts(self, frame: _Frame, max_fronts: Optional[int]) -> Tuple[List[List[int]], List[int]]:
        """Non-dominated sorting by repeated peeling of the first front."""
        remaining = list(range(len(frame.candidates)))
        fronts: List[List[int]] = []
        while remaining and (max_fronts is None or len(fronts) < max_fronts):
            front = self._nondominated(frame.keys, remaining, frame.violations)
            members = set(front)
            fronts.append(front)
            remaining = [i for i in remaining if i not in members]
        return fronts, remaining

    def _feasible_front(self, frame: _Frame) -> List[int]:
        feasible = [i for i, v in enumerate(frame.violations) if v == 0.0]
        return self._nondominated(frame.keys, feasible, frame.violations)

    @staticmethod
    def _crowding(vectors: Sequence[Sequence[float]], members: Sequence[int]) -> Dict[int, float]:
        """NSGA-II crowding distance within one front; boundary points (and fronts of <= 2) get infinity."""
        if len(members) <= 2:
            return {i: math.inf for i in members}
        distance = {i: 0.0 for i in members}
        for k in range(len(vectors[members[0]])):
            order = sorted(members, key=lambda i: vectors[i][k])
            span = vectors[order[-1]][k] - vectors[order[0]][k]
            if span <= 0:
                continue
            distance[order[0]] = distance[order[-1]] = math.inf
            for pos in range(1, len(order) - 1):
                distance[order[pos]] += (vectors[order[pos + 1]][k] - vectors[order[pos - 1]][k]) / span
        return distance

    # ------------------------------------------------------------------ normalisation and reference points
    @staticmethod
    def _extent(vectors: Sequence[Sequence[float]], objectives: Tuple[Objective, ...]) -> Tuple[List[float], List[float]]:
        """Ideal and nadir in minimisation space; natural bounds win over observed values."""
        ideal: List[float] = []
        nadir: List[float] = []
        for k, obj in enumerate(objectives):
            column = [v[k] for v in vectors]
            best, worst = obj.best_bound, obj.worst_bound
            ideal.append(best if best is not None else (min(column) if column else 0.0))
            nadir.append(worst if worst is not None else (max(column) if column else 0.0))
        return ideal, nadir

    @staticmethod
    def _normalise(vector: Sequence[float], ideal: Sequence[float], nadir: Sequence[float]) -> Tuple[float, ...]:
        return tuple((x - lo) / (hi - lo) if hi > lo else 0.0 for x, lo, hi in zip(vector, ideal, nadir))

    @staticmethod
    def _to_original(vector: Sequence[float], objectives: Tuple[Objective, ...]) -> Dict[str, float]:
        return {o.name: o.sign * x for o, x in zip(objectives, vector)}

    def _reference(self, vectors: Sequence[Sequence[float]], objectives: Tuple[Objective, ...],
                   reference_point: Optional[Mapping[str, float]] = None) -> Tuple[float, ...]:
        """Hypervolume reference point in minimisation space.

        Priority: explicit/configured point, then each objective's worst natural bound, then the observed
        nadir pushed outward by ``reference_margin`` of the observed range.
        """
        given = _coerce_value_map(reference_point, "reference_point") if reference_point is not None else self.reference_point
        if given:
            reference = []
            for obj in objectives:
                if obj.name not in given:
                    raise OptimizationValidationError(f"reference_point is missing objective '{obj.name}'", context={"provided": sorted(given)})
                reference.append(obj.sign * given[obj.name])
            return tuple(reference)
        reference = []
        for k, obj in enumerate(objectives):
            if obj.worst_bound is not None:
                reference.append(obj.worst_bound)
                continue
            column = [v[k] for v in vectors]
            hi, lo = max(column), min(column)
            reference.append(hi + self.settings.reference_margin * ((hi - lo) if hi > lo else max(abs(hi), 1.0)))
        return tuple(reference)

    # ------------------------------------------------------------------ hypervolume
    @classmethod
    def _hv_exact(cls, points: Sequence[Tuple[float, ...]], ref: Sequence[float]) -> float:
        """Exact hypervolume: sweep for 1-2 objectives, slicing on the last objective above that."""
        if not points:
            return 0.0
        dim = len(ref)
        if dim == 1:
            return ref[0] - min(p[0] for p in points)
        if dim == 2:
            volume, previous_y = 0.0, ref[1]
            for x, y in sorted(points):
                if y < previous_y:
                    volume += (ref[0] - x) * (previous_y - y)
                    previous_y = y
            return volume
        ordered = sorted(points, key=lambda p: p[-1])
        volume = 0.0
        for k, point in enumerate(ordered):
            upper = ordered[k + 1][-1] if k + 1 < len(ordered) else ref[-1]
            depth = upper - point[-1]
            if depth <= 0:
                continue
            slab = [q[:-1] for q in ordered[:k + 1]]
            slab = [slab[i] for i in cls._nondominated(slab)]
            volume += depth * cls._hv_exact(slab, ref[:-1])
        return volume

    def _hv_monte_carlo(self, points: Sequence[Tuple[float, ...]], ref: Sequence[float]) -> float:
        dim = len(ref)
        lower = [min(p[k] for p in points) for k in range(dim)]
        box = math.prod(ref[k] - lower[k] for k in range(dim))
        rng = random.Random(self.settings.seed)  # fixed seed: repeated calls (and leave-one-out) share samples
        hits = 0
        for _ in range(self.settings.hv_samples):
            sample = [rng.uniform(lower[k], ref[k]) for k in range(dim)]
            if any(all(a <= b for a, b in zip(p, sample)) for p in points):
                hits += 1
        return box * hits / self.settings.hv_samples

    def _hypervolume(self, points: Sequence[Tuple[float, ...]], ref: Sequence[float]) -> Tuple[float, str]:
        """Hypervolume of ``points`` (minimisation space) w.r.t. ``ref``; points not beating ``ref`` are ignored."""
        inside = [p for p in points if all(a < b for a, b in zip(p, ref))]
        if not inside:
            return 0.0, "exact"
        front = [inside[i] for i in self._nondominated(inside)]
        if len(ref) <= 2 or len(front) <= self.settings.hv_exact_max_points:
            return self._hv_exact(front, ref), "exact"
        return self._hv_monte_carlo(front, ref), "monte_carlo"

    def _hv_contributions(self, points: Sequence[Tuple[float, ...]], ref: Sequence[float]) -> List[float]:
        total = self._hypervolume(points, ref)[0]
        return [max(0.0, total - self._hypervolume(list(points[:i]) + list(points[i + 1:]), ref)[0]) for i in range(len(points))]

    def _measure_hypervolume(self, objectives: Tuple[Objective, ...], front_vectors: Sequence[Tuple[float, ...]],
                             reference_vectors: Sequence[Tuple[float, ...]],
                             reference_point: Optional[Mapping[str, float]]) -> _Hypervolume:
        reference = self._reference(reference_vectors, objectives, reference_point)
        value, method = self._hypervolume(front_vectors, reference)
        ideal, _ = self._extent(reference_vectors, objectives)
        box = math.prod(r - i for r, i in zip(reference, ideal))
        return _Hypervolume(value, value / box if box > 0 else None, method, reference)

    @staticmethod
    def _spacing(normalised: Sequence[Sequence[float]]) -> float:
        """Schott spacing: standard deviation of nearest-neighbour (L1) distances."""
        n = len(normalised)
        if n < 2:
            return 0.0
        nearest = [min(sum(abs(a - b) for a, b in zip(p, q)) for j, q in enumerate(normalised) if j != i)
                   for i, p in enumerate(normalised)]
        mean = sum(nearest) / n
        return math.sqrt(sum((mean - d) ** 2 for d in nearest) / (n - 1))

    # ------------------------------------------------------------------ recommendation
    def _weights(self, objectives: Tuple[Objective, ...], override: Optional[Mapping[str, float]]) -> List[float]:
        raw = {o.name: o.weight for o in objectives}
        for name, value in (override or {}).items():
            number = to_finite_float(value)
            if name not in raw or number is None or number < 0:
                raise OptimizationValidationError(f"Invalid weight for '{name}'", context={"value": value, "objectives": sorted(raw)})
            raw[name] = number
        total = sum(raw.values())
        if total <= 0:
            raise OptimizationValidationError("At least one objective weight must be > 0")
        return [raw[o.name] / total for o in objectives]

    def _eligible(self, frame: _Frame, limits: Optional[Mapping[str, float]]) -> List[int]:
        """Feasible candidates inside the per-objective limits (each limit is the worst acceptable value)."""
        active = _coerce_value_map(limits, "limits") if limits is not None else self.limits
        checks: List[Tuple[int, float]] = []
        for name, bound in active.items():
            position = next((k for k, o in enumerate(frame.objectives) if o.name == name), None)
            if position is None:
                raise OptimizationValidationError(f"Limit refers to unknown objective '{name}'", context={"objectives": [o.name for o in frame.objectives]})
            checks.append((position, frame.objectives[position].sign * bound))
        return [i for i, v in enumerate(frame.violations)
                if v == 0.0 and all(frame.vectors[i][k] <= bound for k, bound in checks)]

    def _score_eligible(self, frame: _Frame, strategy: str, weights: Optional[Mapping[str, float]],
                        limits: Optional[Mapping[str, float]], reference_point: Optional[Mapping[str, float]]) -> List[Tuple[int, float]]:
        eligible = self._eligible(frame, limits)
        members = self._nondominated(frame.keys, eligible, frame.violations)
        if not members:
            return []
        objectives = frame.objectives
        vectors = [frame.vectors[i] for i in members]
        w = self._weights(objectives, weights)
        ideal, nadir = self._extent(vectors, objectives)
        norm = [self._normalise(v, ideal, nadir) for v in vectors]

        if strategy == "weighted_sum":
            scores = [sum(wk * x for wk, x in zip(w, n)) for n in norm]
        elif strategy == "chebyshev":
            scores = [max(wk * x for wk, x in zip(w, n)) + _CHEBYSHEV_RHO * sum(n) for n in norm]
        elif strategy == "knee":
            scores = self._knee_scores(norm, w)
        elif strategy == "hypervolume":
            reference = self._reference([frame.vectors[i] for i in eligible], objectives, reference_point)
            scores = [-c for c in self._hv_contributions(vectors, reference)]
        else:  # compromise
            scores = self._compromise_scores(norm, w)
        return sorted(zip(members, scores), key=lambda pair: (pair[1], pair[0]))

    @staticmethod
    def _compromise_scores(norm: Sequence[Sequence[float]], weights: Sequence[float]) -> List[float]:
        return [math.sqrt(sum(wk * x * x for wk, x in zip(weights, n))) for n in norm]

    def _knee_scores(self, norm: Sequence[Sequence[float]], weights: Sequence[float]) -> List[float]:
        """Negative distance to the hyperplane through the extreme points (more negative = sharper knee).

        A concave front has no knee (every point lies beyond the hyperplane), so an extreme point wins there.
        """
        m = len(norm[0])
        plane = None
        if m >= 2 and len(norm) >= m:
            extremes = [min(norm, key=lambda p: (p[k], sum(p))) for k in range(m)]
            plane = solve_linear_system([list(e) for e in extremes], [1.0] * m)
        length = math.sqrt(sum(a * a for a in plane)) if plane else 0.0
        if not plane or length <= 1e-12:
            logger.debug("knee: degenerate extreme-point hyperplane, falling back to compromise scoring")
            return self._compromise_scores(norm, weights)
        return [-(1.0 - sum(a * x for a, x in zip(plane, p))) / length for p in norm]

    # ------------------------------------------------------------------ archive truncation
    def _truncate(self, frame: _Frame, members: List[int], size: int) -> List[int]:
        """Drop members one at a time (re-evaluating after each drop) until ``size`` remain."""
        kept = list(members)
        if len(kept) <= size:
            return kept
        by_hypervolume = self.settings.archive_truncation == "hypervolume"
        while len(kept) > size:
            if by_hypervolume:
                reference = self._reference([frame.vectors[i] for i in kept], frame.objectives)
                scores = self._hv_contributions([frame.vectors[i] for i in kept], reference)
            else:
                crowd = self._crowding(frame.vectors, kept)
                scores = [crowd[i] for i in kept]
            del kept[min(range(len(kept)), key=scores.__getitem__)]
        return kept

    # ------------------------------------------------------------------ result assembly
    def _build_metrics(self, frame: _Frame, fronts: List[List[int]], front_feasible: bool, n_feasible: int,
                       reference_point: Optional[Mapping[str, float]], compute: bool) -> ParetoMetrics:
        objectives = frame.objectives
        front0 = fronts[0]
        front_feasible = bool(front_feasible)
        base = {
            "n_candidates": len(frame.candidates),
            "n_feasible": n_feasible,
            "front_size": len(front0),
            "n_fronts": len(fronts),
            "front_feasible": front_feasible,
        }
        if not front_feasible:
            return ParetoMetrics(
                n_candidates=base["n_candidates"], n_feasible=base["n_feasible"], front_size=base["front_size"],
                n_fronts=base["n_fronts"], front_feasible=base["front_feasible"], ideal={}, nadir={},
                reference_point=None, hypervolume=None, hypervolume_normalised=None, hypervolume_method=None, spacing=None,
            )
        vectors = [frame.vectors[i] for i in front0]
        observed_ideal = [min(v[k] for v in vectors) for k in range(len(objectives))]
        observed_nadir = [max(v[k] for v in vectors) for k in range(len(objectives))]
        hv: Optional[_Hypervolume] = None
        spacing: Optional[float] = None
        if compute and len(front0) <= self.settings.metrics_max_front:
            feasible_vectors = [v for v, viol in zip(frame.vectors, frame.violations) if viol == 0.0]
            hv = self._measure_hypervolume(objectives, vectors, feasible_vectors, reference_point)
            ideal, nadir = self._extent(vectors, objectives)
            spacing = self._spacing([self._normalise(v, ideal, nadir) for v in vectors])
        elif compute:
            logger.warning("Front of %d exceeds metrics_max_front=%d; skipping hypervolume and spacing", len(front0), self.settings.metrics_max_front)
        return ParetoMetrics(
            n_candidates=base["n_candidates"], n_feasible=base["n_feasible"], front_size=base["front_size"],
            n_fronts=base["n_fronts"], front_feasible=base["front_feasible"],
            ideal=self._to_original(observed_ideal, objectives), nadir=self._to_original(observed_nadir, objectives),
            reference_point=self._to_original(hv.reference, objectives) if hv else None,
            hypervolume=hv.value if hv else None, hypervolume_normalised=hv.normalised if hv else None,
            hypervolume_method=hv.method if hv else None, spacing=spacing,
        )

    @staticmethod
    def _empty_metrics(n_candidates: int, n_feasible: int) -> ParetoMetrics:
        return ParetoMetrics(
            n_candidates=n_candidates,
            n_feasible=n_feasible,
            front_size=0,
            n_fronts=0,
            front_feasible=True,
            ideal={},
            nadir={},
            reference_point=None,
            hypervolume=None,
            hypervolume_normalised=None,
            hypervolume_method=None,
            spacing=None,
        )

    @staticmethod
    def _build_result(run_id: str, objectives: Tuple[Objective, ...], frame: _Frame, fronts: List[List[int]],
                      unranked: List[int], duplicates: List[Candidate], crowding: Dict[int, float], strategy: str,
                      ranked: List[Tuple[int, float]], started: float, metrics: ParetoMetrics) -> ParetoResult:
        ranks: Dict[str, Optional[int]] = {}
        for rank, members in enumerate(fronts):
            for i in members:
                ranks[frame.candidates[i].id] = rank
        for i in unranked:
            ranks[frame.candidates[i].id] = None
        best = ranked[0] if ranked else None
        return ParetoResult(
            run_id=run_id,
            objectives=objectives,
            fronts=tuple(tuple(frame.candidates[i] for i in members) for members in fronts),
            unranked=tuple(frame.candidates[i] for i in unranked),
            duplicates=tuple(duplicates),
            ranks=ranks,
            crowding={frame.candidates[i].id: d for i, d in crowding.items()},
            metrics=metrics,
            strategy=strategy,
            recommended=frame.candidates[best[0]] if best else None,
            recommended_score=best[1] if best else None,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )


__all__ = [
    "_coerce_objectives",
    "_coerce_value_map",
    "ParetoEfficiency",
    "ParetoSettings",
    "Objective",
    "Candidate",
    "ParetoResult",
    "ParetoMetrics",
    "ArchiveUpdate",
    "FrontComparison",
    "SELECTION_STRATEGIES",
    "TRUNCATION_STRATEGIES",
]