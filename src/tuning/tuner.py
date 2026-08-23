"""Strategy-independent orchestration for SLAI hyperparameter tuning.

``HyperparamTuner`` owns configuration snapshotting, request construction,
search dispatch, result-contract verification, optional promotion-policy
execution, and optional artifact dispatch.  It deliberately does *not* own
model evaluation or live agent state:

* search modules expose ``run_search(request, evaluation_context)``;
* evaluators own dataset or scenario execution and transactional state restore;
* promotion occurs only when a caller supplies an explicit policy;
* artifact writes occur only when a caller supplies an artifact writer.

This boundary is essential for mutating QNN and adaptive meta-learning
evaluators: a successful trial record is accepted only when its state record
shows that candidate mutations were restored or discarded.
"""

from __future__ import annotations

import asyncio
import copy
import inspect

from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, TypeAlias, cast

from .tuning_types import *
from .tuning_validation import *
from .utils.config_loader import *
from .utils.tuning_errors import *
from .utils.tuning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("HyperparameterTuner")
printer = PrettyPrinter()

SearchRunner: TypeAlias = Callable[
    [TuningRunRequest, Any], SearchResult | Awaitable[SearchResult]
]
PromotionPolicy: TypeAlias = Callable[
    [TuningRunRequest, SearchResult, Any],
    PromotionRecord | None | Awaitable[PromotionRecord | None],
]
ArtifactWriter: TypeAlias = Callable[
    [TuningResult], Sequence[ArtifactRecord] | Awaitable[Sequence[ArtifactRecord]]
]

_DEFAULT_RUNNER_SYMBOLS: Mapping[TuningStrategy, str] = {
    TuningStrategy.BAYESIAN: "src.tuning.search.bayesian:run_search",
    TuningStrategy.GRID: "src.tuning.search.grid:run_search",
}


class HyperparamTuner:
    """Coordinate one immutable tuning configuration snapshot.

    Parameters
    ----------
    model_type:
        Explicit model-type override.  If omitted, configuration is used.
    evaluation_function:
        Compatibility bridge for v2.2 supervised callers.  New integrations
        should pass an explicit ``evaluation_context`` to ``run``.
    strategy:
        Optional explicit strategy override.
    config / config_path:
        Supply either an already-loaded mapping or a YAML path, never both.
    strategy_runners:
        Instance-local runner overrides.  Each runner receives exactly
        ``(TuningRunRequest, evaluation_context)`` and returns ``SearchResult``.
    promotion_policy:
        Optional explicit policy.  With no policy, no promotion is attempted.
    artifact_writer:
        Optional persistence callback.  With no writer, the tuner performs no
        filesystem writes.

    A tuner retains a deep-copied config snapshot for reproducibility.  Create
    a new tuner (or reload the config before constructing one) to observe file
    changes between experiments.
    """

    def __init__(
        self,
        model_type: str | None = None,
        evaluation_function: Callable[..., float] | None = None,
        *,
        strategy: TuningStrategy | str | None = None,
        config: Mapping[str, Any] | None = None,
        config_path: str | Path | None = None,
        strategy_runners: Mapping[TuningStrategy | str, SearchRunner] | None = None,
        promotion_policy: PromotionPolicy | None = None,
        artifact_writer: ArtifactWriter | None = None,
    ) -> None:
        if config is not None and config_path is not None:
            raise TuningConfigError(
                "Provide either config or config_path, not both.",
                context=TuningErrorContext(
                    component=self.__class__.__name__, operation="initialize"
                ),
            )
        if config is not None and not isinstance(config, Mapping):
            raise TuningConfigError(
                "config must be a mapping.",
                context=TuningErrorContext(
                    component=self.__class__.__name__, operation="initialize"
                ),
                details={"actual_type": type(config).__name__},
            )
        self._config: dict[str, Any] = copy.deepcopy(
            dict(config) if config is not None else load_global_config(config_path)
        )
        validation = validate_tuning_config(self._config)
        validation.raise_if_invalid(
            message="Invalid tuning configuration.",
            error_cls=TuningConfigError,
            context=self._error_context("validate_config"),
        )
        self._tuning_config = get_config_section("tuning", self._config, required=True)
        self._strategy_override = strategy
        self._model_type_override = model_type
        for name, callback in (
            ("evaluation_function", evaluation_function),
            ("promotion_policy", promotion_policy),
            ("artifact_writer", artifact_writer),
        ):
            if callback is not None and not callable(callback):
                raise TuningContractError(
                    f"{name} must be callable when provided.",
                    context=self._error_context("initialize"),
                    details={"actual_type": type(callback).__name__},
                )
        self.evaluation_function = evaluation_function
        self._promotion_policy = promotion_policy
        self._artifact_writer = artifact_writer

        self.settings = self._load_settings()
        self._strategy_config = get_config_section(
            f"{self.settings.strategy.value}_search",
            self._config,
            required=True,
        )
        self._strategy_runners: dict[TuningStrategy, SearchRunner] = {}
        self._registry_lock = RLock()
        for runner_strategy, runner in (strategy_runners or {}).items():
            self.register_strategy(runner_strategy, runner)

        logger.info(
            "Tuner initialized: strategy=%s model_type=%s config=%s",
            self.settings.strategy.value,
            self.settings.model_type,
            self.settings.config_path,
        )

    @property
    def config(self) -> dict[str, Any]:
        """Return a defensive copy of the immutable run configuration source."""

        return copy.deepcopy(self._config)

    @property
    def config_fingerprint(self) -> str:
        """Fingerprint semantic config content, excluding repository path metadata."""

        semantic = {
            key: value
            for key, value in self._config.items()
            if not str(key).startswith("__")
        }
        return stable_fingerprint(semantic)

    @property
    def promotion_policy(self) -> PromotionPolicy | None:
        return self._promotion_policy

    @property
    def artifact_writer(self) -> ArtifactWriter | None:
        return self._artifact_writer

    def register_strategy(
        self,
        strategy: TuningStrategy | str,
        runner: SearchRunner,
        *,
        replace_existing: bool = False,
    ) -> None:
        """Register an instance-local search runner with explicit replacement."""

        try:
            parsed = TuningStrategy.parse(strategy)
        except ValueError as exc:
            raise TuningStrategyError(
                str(exc),
                context=self._error_context("register_strategy"),
                cause=exc,
            ) from exc
        if not callable(runner):
            raise TuningContractError(
                "A search runner must be callable.",
                context=self._error_context("register_strategy"),
                details={"strategy": parsed.value, "runner_type": type(runner).__name__},
            )
        with self._registry_lock:
            if parsed in self._strategy_runners and not replace_existing:
                raise TuningStrategyError(
                    f"A runner for {parsed.value!r} is already registered.",
                    context=self._error_context("register_strategy"),
                )
            self._strategy_runners[parsed] = runner

    def prepare_request(
        self,
        *,
        run_id: str | None = None,
        seeds: Sequence[int] | None = None,
        scenario_ids: Sequence[str] | None = None,
        objective: MetricSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TuningRunRequest:
        """Build the exact immutable request that will be handed to search."""

        if objective is not None and not isinstance(objective, MetricSpec):
            raise TuningValidationError(
                "objective must be a MetricSpec.",
                context=self._error_context("prepare_request"),
                details={"actual_type": type(objective).__name__},
            )
        resolved_objective = objective or self._objective_from_config()
        search_space = self._extract_search_space()
        if not search_space and not self.settings.allow_generate:
            raise TuningSearchSpaceError(
                f"No search space is configured for model_type {self.settings.model_type!r}.",
                context=self._error_context("prepare_request"),
                details={"allow_generate": False},
            )
        try:
            return TuningRunRequest(
                run_id=run_id or generate_run_id("tuning"),
                settings=self.settings,
                config=self._config,
                strategy_config=self._strategy_config,
                search_space=search_space,
                config_fingerprint=self.config_fingerprint,
                objective=resolved_objective,
                seeds=tuple(seeds or ()),
                scenario_ids=tuple(scenario_ids or ()),
                metadata=dict(metadata or {}),
            )
        except (TypeError, ValueError) as exc:
            raise TuningValidationError(
                "Invalid tuning run request.",
                context=self._error_context("prepare_request"),
                details={"validation_error": str(exc)},
                cause=exc,
            ) from exc

    def run(
        self,
        *,
        evaluation_context: Any = None,
        X_data: Any = None,
        y_data: Any = None,
        run_id: str | None = None,
        seeds: Sequence[int] | None = None,
        scenario_ids: Sequence[str] | None = None,
        objective: MetricSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TuningResult:
        """Execute a synchronous tuning lifecycle.

        Async runners, promotion policies, or artifact writers are rejected
        here with a contract error; use ``run_async`` for those integrations.
        """

        started_at = utc_now()
        request = self.prepare_request(
            run_id=run_id,
            seeds=seeds,
            scenario_ids=scenario_ids,
            objective=objective,
            metadata=metadata,
        )
        context = self._build_evaluation_context(evaluation_context, X_data, y_data)
        runner = self._resolve_runner(request.settings.strategy)
        search_result = self._call_search_sync(runner, request, context)
        promotion = self._call_promotion_sync(request, search_result, context)
        preliminary = self._compose_result(
            request=request,
            search_result=search_result,
            promotion=promotion,
            artifacts=(),
            started_at=started_at,
            completed_at=utc_now(),
        )
        artifacts = self._call_artifact_writer_sync(preliminary)
        result = self._compose_result(
            request=request,
            search_result=search_result,
            promotion=promotion,
            artifacts=artifacts,
            started_at=started_at,
            completed_at=utc_now(),
        )
        self._log_completion(result)
        return result

    async def run_async(
        self,
        *,
        evaluation_context: Any = None,
        X_data: Any = None,
        y_data: Any = None,
        run_id: str | None = None,
        seeds: Sequence[int] | None = None,
        scenario_ids: Sequence[str] | None = None,
        objective: MetricSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TuningResult:
        """Execute a tuning lifecycle without blocking the caller's event loop."""

        started_at = utc_now()
        request = self.prepare_request(
            run_id=run_id,
            seeds=seeds,
            scenario_ids=scenario_ids,
            objective=objective,
            metadata=metadata,
        )
        context = self._build_evaluation_context(evaluation_context, X_data, y_data)
        runner = self._resolve_runner(request.settings.strategy)
        search_result = await self._call_search_async(runner, request, context)
        promotion = await self._call_promotion_async(request, search_result, context)
        preliminary = self._compose_result(
            request=request,
            search_result=search_result,
            promotion=promotion,
            artifacts=(),
            started_at=started_at,
            completed_at=utc_now(),
        )
        artifacts = await self._call_artifact_writer_async(preliminary)
        result = self._compose_result(
            request=request,
            search_result=search_result,
            promotion=promotion,
            artifacts=artifacts,
            started_at=started_at,
            completed_at=utc_now(),
        )
        self._log_completion(result)
        return result

    def run_tuning_pipeline(
        self,
        X_data: Any = None,
        y_data: Any = None,
        **kwargs: Any,
    ) -> TuningResult:
        """Backward-compatible v2.2 entry point delegated to ``run``."""

        return self.run(X_data=X_data, y_data=y_data, **kwargs)

    def _load_settings(self) -> TunerSettings:
        raw_strategy = (
            self._strategy_override
            if self._strategy_override is not None
            else self._tuning_config.get("strategy")
        )
        if raw_strategy is None:
            raise TuningConfigError(
                "tuning.strategy is required.",
                context=self._error_context("load_settings"),
            )
        try:
            strategy = TuningStrategy.parse(raw_strategy)
        except ValueError as exc:
            raise TuningStrategyError(
                str(exc), context=self._error_context("load_settings"), cause=exc
            ) from exc

        strategy_section = get_config_section(
            f"{strategy.value}_search", self._config, required=True
        )
        explicit_model = self._model_type_override
        global_model = self._config.get("model_type")
        strategy_model = strategy_section.get("model_type")
        configured_models = [
            str(value).strip()
            for value in (global_model, strategy_model)
            if value is not None and str(value).strip()
        ]
        if explicit_model is None and len({item.casefold() for item in configured_models}) > 1:
            raise TuningConfigError(
                "Conflicting model_type values exist in global and strategy configuration; "
                "pass model_type explicitly or make the config consistent.",
                context=self._error_context("load_settings"),
                details={
                    "global_model_type": global_model,
                    "strategy_model_type": strategy_model,
                },
            )
        model_type = explicit_model or global_model or strategy_model
        if model_type is None or not str(model_type).strip():
            raise TuningConfigError(
                "model_type is required either explicitly or in configuration.",
                context=self._error_context("load_settings"),
            )
        try:
            allow_generate = coerce_bool(
                self._tuning_config.get("allow_generate", False),
                name="tuning.allow_generate",
            )
        except ValueError as exc:
            raise TuningConfigError(
                str(exc), context=self._error_context("load_settings"), cause=exc
            ) from exc
        raw_output = strategy_section.get("output_dir", self._tuning_config.get("output_dir"))
        output_dir = None if raw_output is None else Path(str(raw_output)).expanduser()

        raw_config_path = self._config.get("__config_path__")
        # An injected mapping intentionally has no source path.  A loaded
        # config receives __config_path__ from the shared repository.
        config_path = None if raw_config_path is None else Path(str(raw_config_path))
        return TunerSettings(
            strategy=strategy,
            model_type=str(model_type).strip(),
            allow_generate=allow_generate,
            output_dir=output_dir,
            config_path=config_path,
        )

    def _extract_search_space(self) -> tuple[dict[str, Any], ...]:
        hyperparameters = self._config.get("hyperparameters", {})
        if hyperparameters is None:
            return ()
        if not isinstance(hyperparameters, Mapping):
            raise TuningSearchSpaceError(
                "hyperparameters must be a mapping keyed by model_type.",
                context=self._error_context("extract_search_space"),
                details={"actual_type": type(hyperparameters).__name__},
            )
        matches = [
            key
            for key in hyperparameters
            if str(key).strip().casefold() == self.settings.model_type.casefold()
        ]
        if len(matches) > 1:
            raise TuningSearchSpaceError(
                f"Multiple case-insensitive search spaces match {self.settings.model_type!r}.",
                context=self._error_context("extract_search_space"),
                details={"matching_keys": [str(item) for item in matches]},
            )
        if not matches:
            return ()
        raw_space = hyperparameters[matches[0]]
        if not isinstance(raw_space, Sequence) or isinstance(raw_space, (str, bytes, bytearray)):
            raise TuningSearchSpaceError(
                f"Search space for {self.settings.model_type!r} must be a sequence of mappings.",
                context=self._error_context("extract_search_space"),
                details={"actual_type": type(raw_space).__name__},
            )
        copied: list[dict[str, Any]] = []
        for index, item in enumerate(raw_space):
            if not isinstance(item, Mapping):
                raise TuningSearchSpaceError(
                    f"Search-space entry {index} must be a mapping.",
                    context=self._error_context("extract_search_space"),
                    details={"index": index, "actual_type": type(item).__name__},
                )
            copied.append(copy.deepcopy(dict(item)))
        return tuple(copied)

    def _objective_from_config(self) -> MetricSpec | None:
        raw_objective = self._tuning_config.get(
            "objective", self._strategy_config.get("objective")
        )
        raw_metric = self._tuning_config.get(
            "scoring_metric", self._strategy_config.get("scoring_metric")
        )
        if isinstance(raw_objective, Mapping):
            raw_metric = raw_objective.get("name", raw_objective.get("metric", raw_metric))
            raw_direction = raw_objective.get("direction")
            raw_unit = raw_objective.get("unit")
            if raw_metric is None or raw_direction is None:
                raise TuningConfigError(
                    "Objective mapping requires both 'name' (or 'metric') and 'direction'.",
                    context=self._error_context("resolve_objective"),
                )
            try:
                return MetricSpec(
                    name=str(raw_metric),
                    direction=ObjectiveDirection.parse(raw_direction),
                    unit=None if raw_unit is None else str(raw_unit),
                )
            except (TypeError, ValueError) as exc:
                raise TuningConfigError(
                    "Invalid objective configuration.",
                    context=self._error_context("resolve_objective"),
                    details={"validation_error": str(exc)},
                    cause=exc,
                ) from exc

        if raw_objective is None or str(raw_objective).strip().casefold() == "auto":
            return None
        if raw_metric is None or not str(raw_metric).strip():
            raise TuningConfigError(
                "An explicit objective direction requires scoring_metric.",
                context=self._error_context("resolve_objective"),
                details={"objective": raw_objective},
            )
        try:
            return MetricSpec(
                name=str(raw_metric).strip(),
                direction=ObjectiveDirection.parse(str(raw_objective)),
            )
        except (TypeError, ValueError) as exc:
            raise TuningConfigError(
                "Invalid objective configuration.",
                context=self._error_context("resolve_objective"),
                details={"objective": raw_objective, "scoring_metric": raw_metric},
                cause=exc,
            ) from exc

    def _build_evaluation_context(
        self, evaluation_context: Any, X_data: Any, y_data: Any
    ) -> Any:
        legacy_supplied = (
            self.evaluation_function is not None or X_data is not None or y_data is not None
        )
        if evaluation_context is not None and legacy_supplied:
            raise TuningValidationError(
                "Do not combine evaluation_context with legacy evaluation_function/X_data/y_data.",
                context=self._error_context("build_evaluation_context"),
            )
        if evaluation_context is not None:
            return evaluation_context
        if (X_data is None) != (y_data is None):
            raise TuningValidationError(
                "Legacy supervised evaluation requires X_data and y_data together.",
                context=self._error_context("build_evaluation_context"),
                details={
                    "X_data_provided": X_data is not None,
                    "y_data_provided": y_data is not None,
                },
            )
        if not legacy_supplied:
            return None
        return {
            "evaluation_function": self.evaluation_function,
            "X_data": X_data,
            "y_data": y_data,
        }

    def _resolve_runner(self, strategy: TuningStrategy) -> SearchRunner:
        with self._registry_lock:
            registered = self._strategy_runners.get(strategy)
        if registered is not None:
            return registered
        symbol_path = _DEFAULT_RUNNER_SYMBOLS[strategy]
        try:
            runner = load_symbol(symbol_path)
        except Exception as exc:
            raise TuningDependencyError(
                f"Unable to load the {strategy.value!r} search runner.",
                context=self._error_context("resolve_runner"),
                details={"symbol": symbol_path},
                cause=exc,
            ) from exc
        if not callable(runner):
            raise TuningContractError(
                f"Default runner {symbol_path!r} is not callable.",
                context=self._error_context("resolve_runner"),
            )
        return cast(SearchRunner, runner)

    def _call_search_sync(
        self, runner: SearchRunner, request: TuningRunRequest, context: Any
    ) -> SearchResult:
        try:
            raw_result = runner(request, context)
        except TuningError:
            raise
        except Exception as exc:
            raise TuningOptimizationError(
                "Search runner failed.",
                context=self._error_context("run_search", run_id=request.run_id),
                details={"runner": qualified_name(runner)},
                cause=exc,
            ) from exc
        if inspect.isawaitable(raw_result):
            self._close_unawaited(raw_result)
            raise TuningContractError(
                "The search runner returned an awaitable during run(); use run_async().",
                context=self._error_context("run_search", run_id=request.run_id),
                details={"runner": qualified_name(runner)},
            )
        return self._validate_search_result(request, raw_result)

    async def _call_search_async(
        self, runner: SearchRunner, request: TuningRunRequest, context: Any
    ) -> SearchResult:
        try:
            raw_result = await self._invoke_async_capable(runner, request, context)
        except TuningError:
            raise
        except Exception as exc:
            raise TuningOptimizationError(
                "Search runner failed.",
                context=self._error_context("run_search", run_id=request.run_id),
                details={"runner": qualified_name(runner)},
                cause=exc,
            ) from exc
        return self._validate_search_result(request, raw_result)

    def _validate_search_result(
        self, request: TuningRunRequest, result: Any
    ) -> SearchResult:
        if not isinstance(result, SearchResult):
            raise TuningContractError(
                "Search runner must return SearchResult.",
                context=self._error_context("validate_search_result", run_id=request.run_id),
                details={"actual_type": type(result).__name__},
            )
        if result.run_id != request.run_id:
            raise TuningContractError(
                "SearchResult run_id does not match its request.",
                context=self._error_context("validate_search_result", run_id=request.run_id),
                details={"result_run_id": result.run_id},
            )
        if result.strategy is not request.settings.strategy:
            raise TuningContractError(
                "SearchResult strategy does not match its request.",
                context=self._error_context("validate_search_result", run_id=request.run_id),
                details={"result_strategy": result.strategy.value},
            )
        if request.objective is not None and result.objective != request.objective:
            raise TuningContractError(
                "Search runner changed an explicitly resolved objective.",
                context=self._error_context("validate_search_result", run_id=request.run_id),
                details={
                    "requested": request.objective.to_dict(),
                    "returned": result.objective.to_dict(),
                },
            )
        best = result.best_trial
        if best is not None:
            for trial in result.trials:
                if (
                    trial.status is TrialStatus.SUCCEEDED
                    and trial.constraints_passed
                    and trial.objective_value is not None
                    and result.objective.direction.better(
                        trial.objective_value, cast(float, best.objective_value)
                    )
                ):
                    raise TuningContractError(
                        "best_trial_id does not reference the optimal eligible trial.",
                        context=self._error_context(
                            "validate_search_result", run_id=request.run_id
                        ),
                        details={
                            "reported_best_trial_id": best.trial_id,
                            "better_trial_id": trial.trial_id,
                        },
                    )
        return result

    def _call_promotion_sync(
        self, request: TuningRunRequest, search_result: SearchResult, context: Any
    ) -> PromotionRecord | None:
        if self.promotion_policy is None or search_result.best_trial is None:
            return None
        try:
            promotion = self.promotion_policy(request, search_result, context)
        except TuningError:
            raise
        except Exception as exc:
            raise TuningPromotionError(
                "Promotion policy failed.",
                context=self._error_context("promote", run_id=request.run_id),
                details={"policy": qualified_name(self.promotion_policy)},
                cause=exc,
            ) from exc
        if inspect.isawaitable(promotion):
            self._close_unawaited(promotion)
            raise TuningContractError(
                "The promotion policy returned an awaitable during run(); use run_async().",
                context=self._error_context("promote", run_id=request.run_id),
            )
        return self._validate_promotion(search_result, promotion)

    async def _call_promotion_async(
        self, request: TuningRunRequest, search_result: SearchResult, context: Any
    ) -> PromotionRecord | None:
        if self.promotion_policy is None or search_result.best_trial is None:
            return None
        try:
            promotion = await self._invoke_async_capable(
                self.promotion_policy, request, search_result, context
            )
        except TuningError:
            raise
        except Exception as exc:
            raise TuningPromotionError(
                "Promotion policy failed.",
                context=self._error_context("promote", run_id=request.run_id),
                details={"policy": qualified_name(self.promotion_policy)},
                cause=exc,
            ) from exc
        return self._validate_promotion(search_result, promotion)

    def _validate_promotion(
        self, search_result: SearchResult, promotion: Any
    ) -> PromotionRecord | None:
        if promotion is None:
            return None
        if not isinstance(promotion, PromotionRecord):
            raise TuningContractError(
                "Promotion policy must return PromotionRecord or None.",
                context=self._error_context("validate_promotion", run_id=search_result.run_id),
                details={"actual_type": type(promotion).__name__},
            )
        candidate = next(
            (
                trial
                for trial in search_result.trials
                if trial.trial_id == promotion.candidate_trial_id
            ),
            None,
        )
        if candidate is None:
            raise TuningContractError(
                "Promotion policy referenced a trial outside the search result.",
                context=self._error_context("validate_promotion", run_id=search_result.run_id),
                details={"candidate_trial_id": promotion.candidate_trial_id},
            )
        if promotion.decision is PromotionDecision.PROMOTE and not candidate.eligible_for_promotion:
            raise TuningPromotionError(
                "Promotion policy attempted to promote an ineligible trial.",
                context=self._error_context("validate_promotion", run_id=search_result.run_id),
                details={"candidate_trial_id": candidate.trial_id},
            )
        return promotion

    def _call_artifact_writer_sync(
        self, result: TuningResult
    ) -> tuple[ArtifactRecord, ...]:
        if self.artifact_writer is None:
            return ()
        try:
            artifacts = self.artifact_writer(result)
        except TuningError:
            raise
        except Exception as exc:
            raise TuningPersistenceError(
                "Artifact writer failed.",
                context=self._error_context("write_artifacts", run_id=result.run_id),
                details={"writer": qualified_name(self.artifact_writer)},
                cause=exc,
            ) from exc
        if inspect.isawaitable(artifacts):
            self._close_unawaited(artifacts)
            raise TuningContractError(
                "The artifact writer returned an awaitable during run(); use run_async().",
                context=self._error_context("write_artifacts", run_id=result.run_id),
            )
        return self._validate_artifacts(result.run_id, artifacts)

    async def _call_artifact_writer_async(
        self, result: TuningResult
    ) -> tuple[ArtifactRecord, ...]:
        if self.artifact_writer is None:
            return ()
        try:
            artifacts = await self._invoke_async_capable(self.artifact_writer, result)
        except TuningError:
            raise
        except Exception as exc:
            raise TuningPersistenceError(
                "Artifact writer failed.",
                context=self._error_context("write_artifacts", run_id=result.run_id),
                details={"writer": qualified_name(self.artifact_writer)},
                cause=exc,
            ) from exc
        return self._validate_artifacts(result.run_id, artifacts)

    def _validate_artifacts(
        self, run_id: str, artifacts: Any
    ) -> tuple[ArtifactRecord, ...]:
        if not isinstance(artifacts, Sequence) or isinstance(
            artifacts, (str, bytes, bytearray)
        ):
            raise TuningContractError(
                "Artifact writer must return a sequence of ArtifactRecord objects.",
                context=self._error_context("validate_artifacts", run_id=run_id),
                details={"actual_type": type(artifacts).__name__},
            )
        result = tuple(artifacts)
        for index, artifact in enumerate(result):
            if not isinstance(artifact, ArtifactRecord):
                raise TuningContractError(
                    "Artifact writer returned an invalid item.",
                    context=self._error_context("validate_artifacts", run_id=run_id),
                    details={"index": index, "actual_type": type(artifact).__name__},
                )
        return result

    def _compose_result(
        self,
        *,
        request: TuningRunRequest,
        search_result: SearchResult,
        promotion: PromotionRecord | None,
        artifacts: Sequence[ArtifactRecord],
        started_at: datetime,
        completed_at: datetime,
    ) -> TuningResult:
        status = search_result.status
        warnings: list[str] = []
        if promotion is not None and promotion.decision is PromotionDecision.ROLLBACK:
            status = RunStatus.ROLLED_BACK
        failed_artifacts = [
            artifact for artifact in artifacts if artifact.status is ArtifactStatus.FAILED
        ]
        if failed_artifacts and status is RunStatus.SUCCEEDED:
            status = RunStatus.DEGRADED
        if failed_artifacts:
            warnings.append(
                f"{len(failed_artifacts)} artifact(s) failed; inspect artifact records."
            )
        try:
            return TuningResult(
                request=request,
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                search_result=search_result,
                promotion=promotion,
                artifacts=tuple(artifacts),
                warnings=tuple(warnings),
                metadata={
                    "promotion_policy": (
                        None
                        if self.promotion_policy is None
                        else qualified_name(self.promotion_policy)
                    ),
                    "artifact_writer": (
                        None
                        if self.artifact_writer is None
                        else qualified_name(self.artifact_writer)
                    ),
                },
            )
        except (TypeError, ValueError) as exc:
            raise TuningInternalError(
                "Unable to compose a valid TuningResult.",
                context=self._error_context("compose_result", run_id=request.run_id),
                details={"validation_error": str(exc)},
                cause=exc,
            ) from exc

    async def _invoke_async_capable(self, function: Callable[..., Any], *args: Any) -> Any:
        if inspect.iscoroutinefunction(function):
            return await function(*args)
        value = await asyncio.to_thread(function, *args)
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _close_unawaited(value: Any) -> None:
        close = getattr(value, "close", None)
        if inspect.iscoroutine(value) and callable(close):
            close()

    def _error_context(self, operation: str, **updates: Any) -> TuningErrorContext:
        settings = getattr(self, "settings", None)
        config_path = None
        if settings is not None and settings.config_path is not None:
            config_path = str(settings.config_path)
        elif hasattr(self, "_config"):
            raw = self._config.get("__config_path__")
            config_path = None if raw is None else str(raw)
        base: dict[str, Any] = {
            "component": self.__class__.__name__,
            "operation": operation,
            "strategy": None if settings is None else settings.strategy.value,
            "model_type": None if settings is None else settings.model_type,
            "config_path": config_path,
        }
        known = set(TuningErrorContext.__dataclass_fields__)
        metadata: dict[str, Any] = {}
        for key, value in updates.items():
            if key in known:
                base[key] = value
            else:
                metadata[key] = value
        base["metadata"] = metadata
        return TuningErrorContext.from_mapping(base)

    @staticmethod
    def _log_completion(result: TuningResult) -> None:
        logger.info(
            "Tuning completed: run_id=%s status=%s strategy=%s model_type=%s "
            "best_trial_id=%s best_score=%s duration_seconds=%.6f",
            result.run_id,
            result.status.value,
            result.request.settings.strategy.value,
            result.request.settings.model_type,
            None if result.best_trial is None else result.best_trial.trial_id,
            result.best_score,
            result.duration_seconds,
        )


__all__ = [
    "ArtifactWriter",
    "HyperparamTuner",
    "PromotionPolicy",
    "SearchRunner",
]

if __name__ == "__main__":
    print("\n=== Running HyperparamTuner Comprehensive Self-Test ===\n")
    printer.status("TEST", "Starting HyperparamTuner tests", "info")

    _failures: list[str] = []

    def _check(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def _run_test(name: str, test: Callable[[], None]) -> None:
        try:
            test()
            printer.status("TEST", name, "success")
        except Exception as exc:
            _failures.append(f"{name}: {type(exc).__name__}: {exc}")
            printer.status("TEST", _failures[-1], "error")

    _objective = MetricSpec("loss", ObjectiveDirection.MINIMIZE)
    _config = {
        "schema_version": 2,
        "model_type": "SelfTestModel",
        "tuning": {
            "strategy": "grid",
            "evaluation_mode": "supervised",
            "allow_generate": False,
            "objective": _objective.to_dict(),
        },
        "grid_search": {"fail_fast": False, "max_combinations": None},
        "hyperparameters": {
            "SelfTestModel": [
                {"name": "x", "type": "integer", "values": [0, 1]}
            ]
        },
        "supervised_evaluation": {
            "objective": _objective.to_dict(),
            "metrics": ["loss"],
            "seeds": [7],
            "split_strategy": "holdout",
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "n_splits": 2,
            "shuffle": True,
            "constraints": [],
        },
    }

    def _runner(
        request: TuningRunRequest, evaluation_context: Any
    ) -> SearchResult:
        _check(evaluation_context == {"token": "context"}, "context was changed")
        started = utc_now()
        trials = tuple(
            TrialRecord(
                trial_id=f"trial-{index}",
                run_id=request.run_id,
                status=TrialStatus.SUCCEEDED,
                parameters={"x": value},
                started_at=started,
                completed_at=utc_now(),
                metrics={"loss": float(value)},
                objective_value=float(value),
            )
            for index, value in enumerate((1, 0), start=1)
        )
        return SearchResult(
            run_id=request.run_id,
            strategy=TuningStrategy.GRID,
            status=RunStatus.SUCCEEDED,
            objective=cast(MetricSpec, request.objective),
            trials=trials,
            started_at=started,
            completed_at=utc_now(),
            best_trial_id="trial-2",
        )

    def _test_request_and_run() -> None:
        tuner = HyperparamTuner(
            config=_config,
            strategy_runners={TuningStrategy.GRID: _runner},
        )
        request = tuner.prepare_request(run_id="self-test", seeds=(7,))
        _check(request.run_id == "self-test", "run identity was not preserved")
        _check(request.objective == _objective, "objective was not resolved")
        _check(len(request.search_space) == 1, "search space was not selected")
        result = tuner.run(
            run_id="self-test-run",
            evaluation_context={"token": "context"},
            seeds=(7,),
        )
        _check(result.status is RunStatus.SUCCEEDED, "run did not succeed")
        _check(result.best_score == 0.0, "best score is incorrect")
        _check(result.promotion is None, "implicit promotion occurred")
        _check(not result.artifacts, "implicit artifact writes occurred")

    def _test_defensive_config_copy() -> None:
        tuner = HyperparamTuner(config=_config, strategy_runners={TuningStrategy.GRID: _runner})
        exposed = tuner.config
        exposed["model_type"] = "mutated"
        _check(
            tuner.settings.model_type == "SelfTestModel",
            "external mutation changed the tuner snapshot",
        )

    def _test_invalid_configuration_rejected() -> None:
        invalid = copy.deepcopy(_config)
        invalid["tuning"]["objective"] = {
            "name": "different",
            "direction": "minimize",
        }
        try:
            HyperparamTuner(config=invalid)
        except TuningConfigError:
            return
        raise AssertionError("inconsistent evaluator objective was accepted")

    _run_test("request construction and synchronous lifecycle", _test_request_and_run)
    _run_test("defensive configuration snapshot", _test_defensive_config_copy)
    _run_test("cross-section configuration rejection", _test_invalid_configuration_rejected)

    _all_passed = not _failures
    printer.status(
        "",
        f"{3 - len(_failures)}/3 HyperparamTuner tests passed",
        "success" if _all_passed else "error",
    )
    if not _all_passed:
        raise SystemExit(1)
    print("\n=== All HyperparamTuner tests passed ===\n")