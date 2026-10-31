"""Version-gated state adapter for SciPy's existing differential evolution engine.

This module does NOT implement another DE algorithm. It only advances SciPy's
DifferentialEvolutionSolver by completed generations and snapshots the state
required for exact replay across interrupted runs.

The engine is a *private SciPy API*. Only versions independently tested below
are allowed, with explicit fail-closed behavior for other versions.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import scipy
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

from .evaluation_runtime import EvaluationJournal
from .utils.optimization_errors import OptimizationValidationError, UnsupportedProblemError

SUPPORTED_SCIPY_DE_VERSIONS = frozenset({"1.17.0"})


def require_supported_scipy() -> None:
    if scipy.__version__ not in SUPPORTED_SCIPY_DE_VERSIONS:
        raise UnsupportedProblemError(
            "Exact DE resume requires a verified SciPy private API version",
            context={"installed": scipy.__version__,
                     "tested": sorted(SUPPORTED_SCIPY_DE_VERSIONS)},
        )


def _array(raw: Any, *, shape: tuple[int, ...], label: str,
           dtype: Any = float, finite: bool = True) -> np.ndarray:
    try:
        arr = np.asarray(raw, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizationValidationError(f"invalid DE state: {label}") from exc
    if arr.shape != shape or (finite and not np.all(np.isfinite(arr))):
        raise OptimizationValidationError(f"invalid DE state dimensions/values: {label}")
    return arr


def _save_state(engine: DifferentialEvolutionSolver, generation: int) -> dict[str, Any]:
    energies: list[float | None] = [
        None if math.isinf(float(v)) else float(v) for v in engine.population_energies
    ]
    if any(v is not None and not math.isfinite(v) for v in energies):
        raise OptimizationValidationError("DE energies must be finite or +/-infinity")
    if not isinstance(engine.random_number_generator, np.random.Generator):
        raise UnsupportedProblemError("DE checkpoint requires NumPy Generator RNG")
    if engine.random_number_generator.bit_generator.__class__.__name__ != "PCG64":
        raise UnsupportedProblemError("DE checkpoint currently supports PCG64 only")
    return {
        "kind": "scipy_differential_evolution",
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "generation": generation,
        "population": engine.population.tolist(),
        "energies": energies,
        "feasible": engine.feasible.tolist(),
        "constraint_violation": engine.constraint_violation.tolist(),
        "random_index": engine._random_population_index.tolist(),
        "rng_state": engine.random_number_generator.bit_generator.state,
        "scale": list(engine.scale) if isinstance(engine.scale, tuple) else float(engine.scale),
        "nfev": int(engine._nfev),
        "converged": bool(engine.converged()),
    }


def _restore_state(engine: DifferentialEvolutionSolver, raw: Mapping[str, Any]) -> int:
    if raw.get("kind") != "scipy_differential_evolution":
        raise OptimizationValidationError("checkpoint is not a SciPy DE generation state")
    if raw.get("scipy_version") != scipy.__version__ or raw.get("numpy_version") != np.__version__:
        raise OptimizationValidationError("DE checkpoint backend versions differ from installed libraries")
    generation = raw.get("generation")
    evaluations = raw.get("nfev")
    if (isinstance(generation, bool) or not isinstance(generation, int) or generation < 0
        or isinstance(evaluations, bool) or not isinstance(evaluations, int) or evaluations < 0):
        raise OptimizationValidationError("invalid DE generation/evaluation counters")
    shape = tuple(engine.population.shape)
    size, dimension = shape
    population = _array(raw.get("population"), shape=shape, label="population")
    if np.any(population < 0.0) or np.any(population > 1.0):
        raise OptimizationValidationError("DE population values must lie in normalized [0,1] bounds")
    energy_raw = raw.get("energies")
    if not isinstance(energy_raw, list) or len(energy_raw) != size:
        raise OptimizationValidationError("invalid DE energy vector")
    energies = _array(
        [np.inf if v is None else v for v in energy_raw],
        shape=(size,), label="energies", finite=False,
    )
    if np.any(np.isnan(energies)) or np.any(energies == -np.inf):
        raise OptimizationValidationError("invalid DE energies")
    feasible = _array(raw.get("feasible"), shape=(size,), label="feasible", dtype=bool)
    violations = _array(raw.get("constraint_violation"),
                        shape=tuple(engine.constraint_violation.shape),
                        label="constraint_violation")
    if np.any(violations < 0):
        raise OptimizationValidationError("DE constraint violations must be non-negative")
    indices = _array(raw.get("random_index"), shape=(size,), label="random_index", dtype=int)
    if sorted(indices.tolist()) != list(range(size)):
        raise OptimizationValidationError("invalid DE random permutation")
    scale_raw = raw.get("scale")
    if isinstance(scale_raw, list):
        if len(scale_raw) != 2 or any(isinstance(v, bool) for v in scale_raw):
            raise OptimizationValidationError("invalid DE mutation dither")
        scale: Any = tuple(float(x) for x in scale_raw)
        if not all(math.isfinite(x) and x > 0 for x in scale):
            raise OptimizationValidationError("invalid DE mutation dither")
    else:
        if isinstance(scale_raw, bool) or not isinstance(scale_raw, (int, float)) or not math.isfinite(scale_raw) or scale_raw <= 0:
            raise OptimizationValidationError("invalid DE mutation scale")
        scale = float(scale_raw)
    rng_state = raw.get("rng_state")
    if not isinstance(rng_state, Mapping) or rng_state.get("bit_generator") != "PCG64":
        raise OptimizationValidationError("invalid DE RNG state type")
    # Apply state only after validation; the journal itself provides the
    # atomic and checksummed persistence contract.
    try:
        rng = np.random.default_rng()
        rng.bit_generator.state = dict(rng_state)
    except (TypeError, ValueError, KeyError) as exc:
        raise OptimizationValidationError("corrupt DE RNG state") from exc
    engine.population = population
    engine.population_energies = energies
    engine.feasible = feasible
    engine.constraint_violation = violations
    engine._random_population_index = indices
    engine.random_number_generator = rng
    engine.scale = scale
    engine._nfev = evaluations
    return generation


def run_checkpointed_de(
    merit: Callable[[Sequence[float]], float],
    bounds: Sequence[tuple[float, float]],
    integrality: Sequence[bool],
    *,
    options: Mapping[str, Any],
    journal: EvaluationJournal,
    seed: int,
    popsize: int,
    tolerance: float,
    maximum_generations: int,
    cancel_event: Any = None,
    deadline: Any = None,
) -> tuple[int, bool]:
    """Advance SciPy DE and commit only complete generations.

    If an exception or cancellation interrupts a generation, individual
    evaluations remain in the journal but its last complete generation state
    stays unchanged. A later resume replays that generation using the cache.
    """
    require_supported_scipy()
    if isinstance(maximum_generations, bool) or maximum_generations < 1:
        raise OptimizationValidationError("max_generations must be positive")
    if any(key in options for key in ("workers", "updating", "vectorized", "mutation",
                                    "recombination", "init", "callback", "constraints")):
        raise UnsupportedProblemError("checkpointed DE supports the fixed serial SciPy engine configuration only")
    if bool(options.get("polish", False)):
        raise UnsupportedProblemError("checkpointed DE does not support unchecked polishing after resume")
    with DifferentialEvolutionSolver(
        merit, bounds,
        strategy=str(options.get("strategy", "best1bin")),
        maxiter=maximum_generations,
        popsize=int(options.get("popsize", popsize)),
        tol=float(options.get("tolerance", tolerance)),
        atol=float(options.get("absolute_tolerance", 0.0)),
        rng=np.random.default_rng(int(options.get("seed", seed))),
        polish=False, updating="immediate", workers=1,
        integrality=np.asarray(integrality, dtype=bool),
        x0=np.asarray(options["x0"], dtype=float) if "x0" in options else None,
    ) as engine:
        if journal.progress is not None:
            generation = _restore_state(engine, journal.progress)
        else:
            generation = 0
            journal.set_progress(_save_state(engine, 0))
        if generation > maximum_generations:
            raise OptimizationValidationError("checkpoint already exceeds requested maximum generations")
        while generation < maximum_generations:
            if cancel_event is not None and cancel_event.is_set():
                return generation, False
            if deadline is not None and deadline.expired():
                return generation, False
            if engine.converged():
                return generation, True
            try:
                next(engine)
            except StopIteration:
                return generation, bool(engine.converged())
            generation += 1
            journal.set_progress(_save_state(engine, generation))
        return generation, bool(engine.converged())


__all__ = ["SUPPORTED_SCIPY_DE_VERSIONS", "require_supported_scipy", "run_checkpointed_de"]
