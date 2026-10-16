"""Monte Carlo rollout and experiment-design utilities.

Repeated sampling follows Robert & Casella (2004) and JCGM 101:2008.  Latin
hypercube sampling follows McKay, Beckman & Conover (1979).  The implementation
creates independent explicit RNG streams and returns samples; it does not rank,
select, or optimize the simulated actions or parameters.
"""

from __future__ import annotations

__version__ = "2.3.0"

import math
import time
import uuid
import numpy as np

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from collections.abc import Callable, Mapping
from typing import Any, Optional

from ..simulation_types import *
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.simulation_errors import *
from ..utils.simulation_helpers import *
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Monte Carlo")


class MonteCarlo:
    """Bounded Monte Carlo executor with schedule-independent child seeds."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("monte_carlo", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.default_sample_count = int(section.get("sample_count", 100))
        self.max_samples = int(section.get("max_samples", 10_000))
        self.default_concurrency = int(section.get("concurrency", 1))
        self.continue_on_error = bool(section.get("continue_on_error", False))
        timeout = section.get("timeout_seconds", None)
        self.default_timeout = None if timeout in (None, "", "none", "None") else float(timeout)
        if self.default_sample_count <= 0 or self.max_samples <= 0:
            raise SimulationValidationError("Monte Carlo sample counts must be > 0")
        if self.default_sample_count > self.max_samples:
            raise SimulationValidationError("monte_carlo.sample_count exceeds max_samples")
        if self.default_concurrency <= 0:
            raise SimulationValidationError("monte_carlo.concurrency must be > 0")
        if self.default_timeout is not None and self.default_timeout <= 0.0:
            raise SimulationValidationError("monte_carlo.timeout_seconds must be > 0")

    def run(
        self,
        rollout_fn: Callable[..., SimulationResult],
        *,
        sample_count: Optional[int] = None,
        seed: Optional[int] = None,
        concurrency: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        continue_on_error: Optional[bool] = None,
        cancellation_check: Optional[Callable[[], bool]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MonteCarloResult:
        if not callable(rollout_fn):
            raise SimulationValidationError("rollout_fn must be callable")
        count = self.default_sample_count if sample_count is None else int(sample_count)
        workers = self.default_concurrency if concurrency is None else int(concurrency)
        timeout = self.default_timeout if timeout_seconds is None else float(timeout_seconds)
        keep_going = self.continue_on_error if continue_on_error is None else bool(continue_on_error)
        if count <= 0 or count > self.max_samples:
            raise SimulationValidationError(
                "sample_count must be within configured bounds",
                context={"sample_count": count, "max_samples": self.max_samples},
            )
        if workers <= 0:
            raise SimulationValidationError("concurrency must be > 0")
        workers = min(workers, count)
        root_seed, child_seeds = spawn_child_seeds(seed, count)
        started_at = time.monotonic()
        logger.info(
            "Monte Carlo batch start | samples=%s concurrency=%s root_seed=%s",
            count,
            workers,
            root_seed,
        )

        def execute(index: int) -> SimulationResult:
            if cancellation_check is not None:
                try:
                    cancelled = bool(cancellation_check())
                except SimulationError:
                    raise
                except Exception as exc:
                    raise SimulationCallbackError(
                        "Monte Carlo cancellation_check failed", cause=exc, operation="monte_carlo"
                    ) from exc
                if cancelled:
                    raise SimulationError("Monte Carlo batch cancelled", code="SIM-1401", operation="monte_carlo")
            check_timeout(started_at, timeout, operation="monte_carlo")
            child_seed = child_seeds[index]
            rng, _ = make_rng(child_seed)
            result = invoke_with_supported_kwargs(
                rollout_fn,
                sample_index=index,
                seed=child_seed,
                rng=rng,
            )
            check_timeout(started_at, timeout, operation="monte_carlo")
            if not isinstance(result, SimulationResult):
                raise SimulationValidationError(
                    "rollout_fn must return SimulationResult",
                    context={"sample_index": index, "returned_type": type(result).__name__},
                )
            return result

        completed: dict[int, SimulationResult] = {}
        failures: list[Mapping[str, Any]] = []

        def handle_failure(index: int, exc: BaseException) -> None:
            payload = {
                "sample_index": index,
                "seed": child_seeds[index],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(payload)
            logger.warning("Monte Carlo sample failed | index=%s error=%s", index, exc)
            if not keep_going:
                raise exc

        if workers == 1:
            for index in range(count):
                try:
                    completed[index] = execute(index)
                except BaseException as exc:
                    handle_failure(index, exc)
        else:
            executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="slai-sim-mc")
            futures: dict[Future[SimulationResult], int] = {}
            try:
                for index in range(count):
                    futures[executor.submit(execute, index)] = index
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        completed[index] = future.result()
                    except BaseException as exc:
                        if not keep_going:
                            for pending in futures:
                                pending.cancel()
                        handle_failure(index, exc)
                    check_timeout(started_at, timeout, operation="monte_carlo")
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

        ordered_results = tuple(completed[index] for index in sorted(completed))
        batch = MonteCarloResult(
            batch_id=f"mc-{uuid.uuid4().hex[:20]}",
            root_seed=root_seed,
            sample_seeds=child_seeds,
            results=ordered_results,
            failures=tuple(failures),
            metadata={
                "requested_samples": count,
                "completed_samples": len(ordered_results),
                "failed_samples": len(failures),
                "concurrency": workers,
                "duration_seconds": time.monotonic() - started_at,
                **dict(metadata or {}),
            },
        )
        logger.info(
            "Monte Carlo batch end | completed=%s failed=%s",
            len(batch.results),
            len(batch.failures),
        )
        return batch

    @staticmethod
    def latin_hypercube(
        bounds: Mapping[str, tuple[float, float]],
        sample_count: int,
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[tuple[dict[str, float], ...], int]:
        """Generate a classic randomized Latin hypercube over finite bounds."""
        if not bounds:
            raise SimulationValidationError("Latin hypercube bounds must be non-empty")
        if sample_count <= 0:
            raise SimulationValidationError("sample_count must be > 0")
        if rng is None:
            generator, concrete_seed = make_rng(seed=seed)
        else:
            if seed is None:
                raise SimulationValidationError("an injected RNG requires its originating seed")
            generator, concrete_seed = rng, normalize_seed(seed)
        names = tuple(str(name) for name in bounds)
        lows = np.empty(len(names), dtype=np.float64)
        highs = np.empty(len(names), dtype=np.float64)
        for index, name in enumerate(names):
            pair = bounds[name]
            if len(pair) != 2:
                raise SimulationValidationError("each Latin hypercube bound must be (low, high)", context={"parameter": name})
            low, high = float(pair[0]), float(pair[1])
            if not math.isfinite(low) or not math.isfinite(high) or not high > low:
                raise SimulationValidationError("Latin hypercube bounds must be finite with high > low", context={"parameter": name})
            lows[index], highs[index] = low, high
        unit = np.empty((sample_count, len(names)), dtype=np.float64)
        for column in range(len(names)):
            permutation = generator.permutation(sample_count)
            jitter = generator.random(sample_count)
            unit[:, column] = (permutation + jitter) / sample_count
        scaled = lows + unit * (highs - lows)
        rows = tuple(
            {name: float(scaled[row, column]) for column, name in enumerate(names)}
            for row in range(sample_count)
        )
        return rows, concrete_seed

    def sample_parameter_space(
        self,
        bounds: Mapping[str, tuple[float, float]],
        *,
        sample_count: Optional[int] = None,
        seed: Optional[int] = None,
        strategy: SamplingStrategy | str = SamplingStrategy.LATIN_HYPERCUBE,
    ) -> tuple[tuple[dict[str, float], ...], int]:
        count = self.default_sample_count if sample_count is None else int(sample_count)
        if count <= 0 or count > self.max_samples:
            raise SimulationValidationError("sample_count is outside configured bounds")
        selected = SamplingStrategy(strategy)
        if selected is SamplingStrategy.LATIN_HYPERCUBE:
            return self.latin_hypercube(bounds, count, seed=seed)
        generator, concrete_seed = make_rng(seed)
        names = tuple(str(name) for name in bounds)
        rows: list[dict[str, float]] = []
        for _ in range(count):
            row: dict[str, float] = {}
            for name in names:
                low, high = (float(value) for value in bounds[name])
                if not math.isfinite(low) or not math.isfinite(high) or not high > low:
                    raise SimulationValidationError("random-sampling bounds must be finite with high > low", context={"parameter": name})
                row[name] = float(generator.uniform(low, high))
            rows.append(row)
        return tuple(rows), concrete_seed


__all__ = ["MonteCarlo"]
