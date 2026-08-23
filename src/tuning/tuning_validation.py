"""Cross-field validation for SLAI tuning configuration and records.

Dataclasses in ``tuning_types.py`` enforce local invariants.  This module owns
relationships that span fields or modules: strategy-specific search-space
semantics, evaluator configuration, objective/metric agreement, constraint
references, and best-trial correctness.
"""

from __future__ import annotations

import math

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

from .tuning_contracts import *
from .tuning_types import *
from .utils.tuning_errors import *
from .utils.tuning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Tuning Contracts")
printer = PrettyPrinter()

class ValidationLevel(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    level: ValidationLevel
    path: str
    message: str
    value: Any = None
    code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "path": self.path,
            "message": self.message,
            "value": to_json_safe(self.value),
            "code": self.code,
        }


@dataclass(slots=True)
class ValidationReport:
    """Accumulate independent validation findings before raising once."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(item for item in self.issues if item.level is ValidationLevel.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(item for item in self.issues if item.level is ValidationLevel.WARNING)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def error(
        self, path: str, message: str, *, value: Any = None, code: str | None = None
    ) -> None:
        self.issues.append(
            ValidationIssue(ValidationLevel.ERROR, path, message, value, code)
        )

    def warning(
        self, path: str, message: str, *, value: Any = None, code: str | None = None
    ) -> None:
        self.issues.append(
            ValidationIssue(ValidationLevel.WARNING, path, message, value, code)
        )

    def extend(self, other: "ValidationReport") -> "ValidationReport":
        self.issues.extend(other.issues)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [item.to_dict() for item in self.issues],
        }

    def raise_if_invalid(
        self,
        *,
        message: str = "Tuning validation failed.",
        error_cls: type[TuningError] = TuningValidationError,
        context: TuningErrorContext | None = None,
    ) -> None:
        if not issubclass(error_cls, TuningError):
            raise TypeError("error_cls must derive from TuningError")
        if self.errors:
            raise error_cls(
                message,
                context=context,
                details=self.to_dict(),
            )


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _validate_bool(report: ValidationReport, path: str, value: Any) -> None:
    try:
        coerce_bool(value, name=path)
    except ValueError as exc:
        report.error(path, str(exc), value=value, code="invalid_boolean")


def parse_metric_spec(raw: Any, *, path: str = "objective") -> MetricSpec:
    """Parse a concrete metric objective; ``auto`` is intentionally rejected."""

    if not isinstance(raw, Mapping):
        raise TuningConfigError(
            f"{path} must be a mapping with name and direction.",
            context=TuningErrorContext(component="TuningValidation", operation="parse_metric"),
            details={"path": path, "value": to_json_safe(raw)},
        )
    name = raw.get("name", raw.get("metric"))
    direction = raw.get("direction")
    if name is None or direction is None:
        raise TuningConfigError(
            f"{path} requires both name and direction.",
            context=TuningErrorContext(component="TuningValidation", operation="parse_metric"),
            details={"path": path},
        )
    if str(direction).strip().casefold() == "auto":
        raise TuningConfigError(
            f"{path}.direction must be resolved before evaluator execution.",
            context=TuningErrorContext(component="TuningValidation", operation="parse_metric"),
            details={"path": path},
        )
    try:
        return MetricSpec(
            name=str(name),
            direction=ObjectiveDirection.parse(direction),
            unit=None if raw.get("unit") is None else str(raw["unit"]),
        )
    except (TypeError, ValueError) as exc:
        raise TuningConfigError(
            f"Invalid metric specification at {path}.",
            context=TuningErrorContext(component="TuningValidation", operation="parse_metric"),
            details={"path": path, "validation_error": str(exc)},
            cause=exc,
        ) from exc


def parse_metric_constraints(
    raw: Any, *, path: str = "constraints"
) -> tuple[MetricConstraint, ...]:
    if raw is None:
        return ()
    if not _is_sequence(raw):
        raise TuningConfigError(
            f"{path} must be a sequence.",
            context=TuningErrorContext(
                component="TuningValidation", operation="parse_constraints"
            ),
        )
    constraints: list[MetricConstraint] = []
    names: set[str] = set()
    for index, item in enumerate(raw):
        item_path = f"{path}[{index}]"
        if not isinstance(item, Mapping):
            raise TuningConfigError(f"{item_path} must be a mapping.")
        threshold = item.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TuningConfigError(f"{item_path}.threshold must be a number.")
        try:
            constraint = MetricConstraint(
                name=str(item.get("name", "")),
                metric_name=str(item.get("metric", item.get("metric_name", ""))),
                operator=ConstraintOperator(str(item.get("operator", ""))),
                threshold=float(threshold),
            )
        except (TypeError, ValueError) as exc:
            raise TuningConfigError(
                f"Invalid metric constraint at {item_path}.",
                context=TuningErrorContext(
                    component="TuningValidation", operation="parse_constraints"
                ),
                details={"path": item_path, "validation_error": str(exc)},
                cause=exc,
            ) from exc
        normalized = constraint.name.casefold()
        if normalized in names:
            raise TuningConfigError(f"Duplicate constraint name {constraint.name!r}.")
        names.add(normalized)
        constraints.append(constraint)
    return tuple(constraints)


def validate_search_space(
    search_space: Any,
    strategy: TuningStrategy | str,
    *,
    path: str = "hyperparameters",
) -> ValidationReport:
    report = ValidationReport()
    try:
        active_strategy = TuningStrategy.parse(strategy)
    except ValueError as exc:
        report.error("tuning.strategy", str(exc), value=strategy, code="unsupported_strategy")
        return report
    if not _is_sequence(search_space):
        report.error(path, "Search space must be a sequence of parameter mappings.")
        return report
    if not search_space:
        report.error(path, "Search space must contain at least one parameter.")
        return report

    names: set[str] = set()
    for index, raw in enumerate(search_space):
        item_path = f"{path}[{index}]"
        if not isinstance(raw, Mapping):
            report.error(item_path, "Parameter definition must be a mapping.")
            continue
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            report.error(f"{item_path}.name", "Parameter name must be non-empty.")
            continue
        normalized_name = name.strip().casefold()
        if normalized_name in names:
            report.error(f"{item_path}.name", f"Duplicate parameter name {name!r}.")
        names.add(normalized_name)

        kind = str(raw.get("type", "")).strip().casefold()
        if kind not in {"real", "integer", "categorical"}:
            report.error(
                f"{item_path}.type",
                "Parameter type must be real, integer, or categorical.",
                value=raw.get("type"),
            )
            continue
        has_values = "values" in raw
        has_lower = "min" in raw
        has_upper = "max" in raw
        has_bounds = has_lower or has_upper
        if has_values and has_bounds:
            report.error(item_path, "Use either values or min/max bounds, not both.")

        if kind == "categorical":
            if not has_values:
                report.error(f"{item_path}.values", "Categorical parameters require values.")
            elif not _is_sequence(raw["values"]) or not raw["values"]:
                report.error(
                    f"{item_path}.values", "Categorical values must be a non-empty sequence."
                )
            else:
                fingerprints = [stable_fingerprint(value) for value in raw["values"]]
                if len(fingerprints) != len(set(fingerprints)):
                    report.error(f"{item_path}.values", "Categorical values contain duplicates.")
                if len(fingerprints) == 1:
                    report.warning(
                        f"{item_path}.values",
                        "A single categorical value does not define a tunable dimension.",
                    )
            if "prior" in raw or "step" in raw:
                report.error(item_path, "Categorical parameters cannot declare prior or step.")
            continue

        if has_values:
            values = raw.get("values")
            if not _is_sequence(values) or not values:
                report.error(f"{item_path}.values", "Numeric values must be non-empty.")
            else:
                normalized_values: list[float] = []
                for value_index, value in enumerate(values):
                    numeric = _finite_number(value)
                    if numeric is None:
                        report.error(
                            f"{item_path}.values[{value_index}]",
                            "Numeric search values must be finite numbers.",
                            value=value,
                        )
                    elif kind == "integer" and not isinstance(value, int):
                        report.error(
                            f"{item_path}.values[{value_index}]",
                            "Integer parameter values must be integers.",
                            value=value,
                        )
                    else:
                        normalized_values.append(numeric)
                if len(normalized_values) != len(set(normalized_values)):
                    report.error(f"{item_path}.values", "Numeric values contain duplicates.")
                if len(normalized_values) == 1:
                    report.warning(
                        f"{item_path}.values",
                        "A single numeric value does not define a tunable dimension.",
                    )
        else:
            if not has_lower or not has_upper:
                report.error(item_path, "Numeric parameters require both min and max bounds.")
                continue
            lower = _finite_number(raw.get("min"))
            upper = _finite_number(raw.get("max"))
            if lower is None:
                report.error(f"{item_path}.min", "min must be a finite number.")
            if upper is None:
                report.error(f"{item_path}.max", "max must be a finite number.")
            if lower is None or upper is None:
                continue
            if kind == "integer" and (
                not isinstance(raw.get("min"), int) or not isinstance(raw.get("max"), int)
            ):
                report.error(item_path, "Integer bounds must be integers.")
            if lower > upper:
                report.error(item_path, "min cannot exceed max.")
            elif lower == upper:
                report.warning(item_path, "Equal bounds do not define a tunable dimension.")
            prior = str(raw.get("prior", "uniform")).strip().casefold()
            if prior not in {"uniform", "log-uniform"}:
                report.error(
                    f"{item_path}.prior", "prior must be uniform or log-uniform."
                )
            if prior == "log-uniform" and lower <= 0:
                report.error(
                    f"{item_path}.min", "log-uniform parameters require min > 0."
                )
            if active_strategy is TuningStrategy.GRID:
                step = _finite_number(raw.get("step"))
                if step is None or step <= 0:
                    report.error(
                        f"{item_path}.step",
                        "Grid numeric bounds require a positive step or explicit values.",
                    )
                elif step > upper - lower and lower != upper:
                    report.warning(
                        f"{item_path}.step",
                        "step exceeds the parameter range and yields only one grid point.",
                    )
    return report


def validate_supervised_config(
    config: Any, *, path: str = "supervised_evaluation"
) -> ValidationReport:
    report = ValidationReport()
    if not isinstance(config, Mapping):
        report.error(path, "Supervised evaluation config must be a mapping.")
        return report
    try:
        objective = parse_metric_spec(config.get("objective"), path=f"{path}.objective")
    except TuningConfigError as exc:
        objective = None
        report.error(f"{path}.objective", exc.message)
    split_strategy = str(config.get("split_strategy", "")).strip().casefold()
    if split_strategy not in {"holdout", "nested_k_fold", "time_series"}:
        report.error(
            f"{path}.split_strategy",
            "split_strategy must be holdout, nested_k_fold, or time_series.",
        )
    validation_fraction = _finite_number(config.get("validation_fraction"))
    if validation_fraction is None or not 0.0 < validation_fraction < 0.5:
        report.error(
            f"{path}.validation_fraction",
            "validation_fraction must be finite and within (0, 0.5).",
        )
    if split_strategy == "holdout":
        test_fraction = _finite_number(config.get("test_fraction"))
        if test_fraction is None or not 0.0 < test_fraction < 1.0:
            report.error(
                f"{path}.test_fraction", "test_fraction must be finite and within (0, 1)."
            )
        elif validation_fraction is not None and validation_fraction + test_fraction >= 1:
            report.error(
                path, "validation_fraction + test_fraction must be less than 1."
            )
    if split_strategy in {"nested_k_fold", "time_series"}:
        n_splits = config.get("n_splits")
        if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
            report.error(f"{path}.n_splits", "n_splits must be an integer >= 2.")
    shuffle = config.get("shuffle")
    if shuffle is not None:
        _validate_bool(report, f"{path}.shuffle", shuffle)
        if split_strategy == "time_series":
            try:
                if coerce_bool(shuffle, name="shuffle"):
                    report.error(
                        f"{path}.shuffle", "Time-series evaluation cannot shuffle observations."
                    )
            except ValueError:
                pass
    metrics = config.get("metrics")
    if not _is_sequence(metrics) or not metrics:
        report.error(f"{path}.metrics", "metrics must be a non-empty sequence.")
    else:
        metric_names = [str(item).strip() for item in metrics]
        if any(not name for name in metric_names):
            report.error(f"{path}.metrics", "Metric names must be non-empty.")
        if len(metric_names) != len(set(metric_names)):
            report.error(f"{path}.metrics", "Metric names must not be duplicated.")
        if objective is not None and objective.name not in set(metric_names):
            report.error(
                f"{path}.metrics", "metrics must include the primary objective name."
            )
        try:
            constraints = parse_metric_constraints(
                config.get("constraints"), path=f"{path}.constraints"
            )
        except TuningConfigError as exc:
            report.error(f"{path}.constraints", exc.message)
        else:
            aggregated_names = set(metric_names)
            for name in metric_names:
                aggregated_names.update(
                    {f"{name}_std", f"{name}_seed_std", f"{name}_seed_sem"}
                )
            aggregated_names.add("evaluation_count")
            for constraint in constraints:
                if constraint.metric_name not in aggregated_names:
                    report.error(
                        f"{path}.constraints",
                        f"Constraint metric {constraint.metric_name!r} is not produced "
                        "by supervised aggregation.",
                    )
    _validate_seeds(report, config.get("seeds"), f"{path}.seeds", required=False)
    return report


def validate_agent_config(
    config: Any, *, path: str = "agent_evaluation"
) -> ValidationReport:
    report = ValidationReport()
    if not isinstance(config, Mapping):
        report.error(path, "Agent evaluation config must be a mapping.")
        return report
    try:
        objective = parse_metric_spec(config.get("objective"), path=f"{path}.objective")
    except TuningConfigError as exc:
        objective = None
        report.error(f"{path}.objective", exc.message)
    _validate_seeds(report, config.get("seeds"), f"{path}.seeds", required=True)
    state = config.get("state")
    if not isinstance(state, Mapping):
        report.error(f"{path}.state", "state must be a mapping.")
    else:
        source = str(state.get("source", "")).strip().casefold()
        if source not in {"fresh", "checkpoint"}:
            report.error(f"{path}.state.source", "source must be fresh or checkpoint.")
        checkpoint_id = state.get("checkpoint_id")
        if source == "checkpoint" and (
            not isinstance(checkpoint_id, str) or not checkpoint_id.strip()
        ):
            report.error(
                f"{path}.state.checkpoint_id",
                "checkpoint source requires a non-empty checkpoint_id.",
            )
        if source == "fresh" and checkpoint_id not in {None, ""}:
            report.error(
                f"{path}.state.checkpoint_id",
                "fresh source cannot declare checkpoint_id.",
            )
        cleanup = str(state.get("cleanup", "auto")).strip().casefold()
        if cleanup not in {"auto", "restore", "discard"}:
            report.error(
                f"{path}.state.cleanup", "cleanup must be auto, restore, or discard."
            )
    bins = config.get("calibration_bins", 10)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        report.error(f"{path}.calibration_bins", "calibration_bins must be an integer >= 2.")
    for field_name in ("require_calibration", "fail_fast"):
        if field_name in config:
            _validate_bool(report, f"{path}.{field_name}", config[field_name])
    try:
        constraints = parse_metric_constraints(
            config.get("constraints"), path=f"{path}.constraints"
        )
    except TuningConfigError as exc:
        constraints = ()
        report.error(f"{path}.constraints", exc.message)
    if objective is not None:
        known_metrics = {
            "task_utility",
            "success_rate",
            "safety_violation_count",
            "safety_violation_rate",
            "latency_mean_seconds",
            "latency_p95_seconds",
            "peak_memory_bytes",
            "utility_seed_std",
            "utility_seed_sem",
            "calibration_ece",
            "calibration_brier",
            "calibration_coverage",
            "evaluation_count",
        }
        known_metrics.add(objective.name)
        for constraint in constraints:
            if constraint.metric_name not in known_metrics:
                report.warning(
                    f"{path}.constraints",
                    f"Constraint metric {constraint.metric_name!r} must be supplied "
                    "by the scenario runner.",
                )
    return report


def validate_artifact_config(
    config: Any, *, path: str = "tuning_artifacts"
) -> ValidationReport:
    report = ValidationReport()
    if not isinstance(config, Mapping):
        report.error(path, "Artifact config must be a mapping.")
        return report
    output_dir = config.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        report.error(f"{path}.output_dir", "output_dir must be a non-empty path string.")
    for name in ("write_summary", "write_trials", "write_config_snapshot"):
        if name in config:
            _validate_bool(report, f"{path}.{name}", config[name])
    enabled: list[bool] = []
    for name in ("write_summary", "write_trials", "write_config_snapshot"):
        try:
            enabled.append(coerce_bool(config.get(name, True), name=name))
        except ValueError:
            pass
    if len(enabled) == 3 and not any(enabled):
        report.error(path, "At least one artifact payload must be enabled.")
    indent = config.get("indent", 2)
    if isinstance(indent, bool) or not isinstance(indent, int) or indent < 0:
        report.error(f"{path}.indent", "indent must be a non-negative integer.")
    return report


def _validate_seeds(
    report: ValidationReport, raw: Any, path: str, *, required: bool
) -> None:
    if raw is None:
        if required:
            report.error(path, "At least one seed is required.")
        return
    if not _is_sequence(raw) or not raw:
        report.error(path, "seeds must be a non-empty sequence of integers.")
        return
    seeds = list(raw)
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        report.error(path, "Every seed must be an integer.")
    elif len(seeds) != len(set(seeds)):
        report.error(path, "seeds must not contain duplicates.")


def validate_tuning_config(config: Any) -> ValidationReport:
    report = ValidationReport()
    if not isinstance(config, Mapping):
        report.error("$", "Configuration root must be a mapping.")
        return report

    tuning = config.get("tuning")
    if not isinstance(tuning, Mapping):
        report.error("tuning", "tuning must be a mapping.")
        return report

    try:
        strategy = TuningStrategy.parse(cast(TuningStrategy | str, tuning.get("strategy")))
    except ValueError as exc:
        report.error("tuning.strategy", str(exc), value=tuning.get("strategy"))
        return report

    evaluation_mode = str(tuning.get("evaluation_mode", "")).strip().casefold()
    if evaluation_mode == "supervised":
        active_evaluator_section = "supervised_evaluation"
    elif evaluation_mode == "agent":
        active_evaluator_section = "agent_evaluation"
    else:
        active_evaluator_section = None
        report.error(
            "tuning.evaluation_mode",
            "evaluation_mode must be supervised or agent.",
            value=tuning.get("evaluation_mode"),
        )

    try:
        primary_objective = parse_metric_spec(
            tuning.get("objective"), path="tuning.objective"
        )
    except TuningConfigError as exc:
        primary_objective = None
        report.error("tuning.objective", exc.message)

    section_name = f"{strategy.value}_search"
    if not isinstance(config.get(section_name), Mapping):
        report.error(section_name, f"{section_name} must be a mapping.")
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type.strip():
        report.error("model_type", "model_type must be a non-empty string.")
    hyperparameters = config.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        report.error("hyperparameters", "hyperparameters must be a mapping.")
    elif isinstance(model_type, str) and model_type.strip():
        matches = [
            key
            for key in hyperparameters
            if str(key).strip().casefold() == model_type.strip().casefold()
        ]
        if len(matches) > 1:
            report.error(
                "hyperparameters", "Multiple case-insensitive model_type keys match."
            )
        elif not matches:
            try:
                allow_generate = coerce_bool(
                    tuning.get("allow_generate", False), name="tuning.allow_generate"
                )
            except ValueError as exc:
                report.error("tuning.allow_generate", str(exc))
                allow_generate = False
            if not allow_generate:
                report.error(
                    "hyperparameters", f"No search space exists for {model_type!r}."
                )
        else:
            report.extend(
                validate_search_space(
                    hyperparameters[matches[0]],
                    strategy,
                    path=f"hyperparameters.{matches[0]}",
                )
            )
    if "supervised_evaluation" in config:
        report.extend(validate_supervised_config(config["supervised_evaluation"]))
    if "agent_evaluation" in config:
        report.extend(validate_agent_config(config["agent_evaluation"]))
    if "tuning_artifacts" in config:
        report.extend(validate_artifact_config(config["tuning_artifacts"]))
    # objective of the evaluator selected by tuning.evaluation_mode.
    if active_evaluator_section is not None:
        active_evaluator = config.get(active_evaluator_section)

        if active_evaluator_section not in config:
            report.error(
                active_evaluator_section,
                f"{active_evaluator_section} is required when "
                f"tuning.evaluation_mode is {evaluation_mode!r}.",
                code="missing_active_evaluator",
            )

        elif (
            isinstance(active_evaluator, Mapping)
            and primary_objective is not None
        ):
            try:
                evaluator_objective = parse_metric_spec(
                    active_evaluator.get("objective"),
                    path=f"{active_evaluator_section}.objective",
                )
            except TuningConfigError:
                # validate_supervised_config() or validate_agent_config()
                # already reports a malformed evaluator objective.
                pass
            else:
                if evaluator_objective != primary_objective:
                    report.error(
                        "tuning.objective",
                        "The primary objective must exactly match the objective "
                        f"of the active {evaluation_mode} evaluator.",
                        value={
                            "tuning": primary_objective.to_dict(),
                            active_evaluator_section: (
                                evaluator_objective.to_dict()
                            ),
                        },
                        code="objective_mismatch",
                    )

    return report


def validate_parameters_against_space(
    parameters: Mapping[str, Any],
    search_space: Sequence[Mapping[str, Any]],
) -> ValidationReport:
    report = ValidationReport()
    if not isinstance(parameters, Mapping):
        report.error("parameters", "parameters must be a mapping.")
        return report
    definitions = {
        str(item.get("name")): item
        for item in search_space
        if isinstance(item, Mapping) and item.get("name") is not None
    }
    missing = set(definitions) - set(parameters)
    unknown = set(parameters) - set(definitions)
    for name in sorted(missing):
        report.error(f"parameters.{name}", "Required search parameter is missing.")
    for name in sorted(unknown):
        report.error(f"parameters.{name}", "Parameter is not declared in the search space.")
    for name in sorted(set(parameters) & set(definitions)):
        value = parameters[name]
        definition = definitions[name]
        kind = str(definition.get("type", "")).strip().casefold()
        if "values" in definition:
            allowed = {
                stable_fingerprint(item): item for item in definition.get("values", ())
            }
            if stable_fingerprint(value) not in allowed:
                report.error(
                    f"parameters.{name}", "Value is not one of the declared choices.", value=value
                )
            continue
        numeric = _finite_number(value)
        if numeric is None:
            report.error(f"parameters.{name}", "Value must be a finite number.")
            continue
        if kind == "integer" and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            report.error(f"parameters.{name}", "Value must be an integer.")
        lower = _finite_number(definition.get("min"))
        upper = _finite_number(definition.get("max"))
        if lower is not None and numeric < lower:
            report.error(f"parameters.{name}", f"Value is below minimum {lower}.")
        if upper is not None and numeric > upper:
            report.error(f"parameters.{name}", f"Value exceeds maximum {upper}.")
    return report


def validate_trial_record(
    trial: TrialRecord, objective: MetricSpec | None = None
) -> ValidationReport:
    report = ValidationReport()
    if not isinstance(trial, TrialRecord):
        report.error("trial", "Expected TrialRecord.")
        return report
    if objective is not None and trial.objective_value is not None:
        observed = trial.metrics.get(objective.name)
        if observed is None:
            report.error(
                f"trial.metrics.{objective.name}",
                "Successful objective value must also appear in aggregated metrics.",
            )
        elif not math.isclose(
            float(observed), float(trial.objective_value), rel_tol=1e-12, abs_tol=1e-12
        ):
            report.error("trial.objective_value", "Objective value disagrees with metrics.")
    if trial.agent_state is not None:
        dispositions = (
            [trial.agent_state.disposition]
            if not isinstance(trial.agent_state, AgentStateAudit)
            else [item.disposition for item in trial.agent_state.transactions]
        )
        failed_dispositions = {
            CandidateStateDisposition.RESTORE_FAILED,
        }
        discard_failed = getattr(CandidateStateDisposition, "DISCARD_FAILED", None)
        if discard_failed is not None:
            failed_dispositions.add(discard_failed)
        if any(
            disposition in failed_dispositions
            for disposition in dispositions
        ) and trial.status is not TrialStatus.FAILED:
            report.error(
                "trial.agent_state.disposition",
                "Failed cleanup requires a failed trial.",
            )
    return report


def validate_search_result(result: SearchResult) -> ValidationReport:
    report = ValidationReport()
    if not isinstance(result, SearchResult):
        report.error("search_result", "Expected SearchResult.")
        return report
    for index, trial in enumerate(result.trials):
        child = validate_trial_record(trial, result.objective)
        for issue in child.issues:
            report.issues.append(
                ValidationIssue(
                    issue.level,
                    f"trials[{index}].{issue.path}",
                    issue.message,
                    issue.value,
                    issue.code,
                )
            )
    best = result.best_trial
    if best is not None and best.objective_value is not None:
        for trial in result.trials:
            if (
                trial.status is TrialStatus.SUCCEEDED
                and trial.constraints_passed
                and trial.objective_value is not None
                and result.objective.direction.better(
                    trial.objective_value, float(best.objective_value)
                )
            ):
                report.error(
                    "best_trial_id",
                    f"Trial {trial.trial_id!r} is better than the declared best trial.",
                )
    return report


__all__ = [
    "ValidationIssue",
    "ValidationLevel",
    "ValidationReport",
    "parse_metric_constraints",
    "parse_metric_spec",
    "validate_agent_config",
    "validate_artifact_config",
    "validate_parameters_against_space",
    "validate_search_result",
    "validate_search_space",
    "validate_supervised_config",
    "validate_trial_record",
    "validate_tuning_config",
]

if __name__ == "__main__":
    print("\n=== Running Tuning Validation Comprehensive Self-Test ===\n")
    import copy
    from collections.abc import Callable
    printer.status("TEST", "Starting tuning validation tests", "info")

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

    _objective_mapping = {"name": "loss", "direction": "minimize", "unit": None}
    _valid_config = {
        "schema_version": 2,
        "model_type": "SelfTestModel",
        "tuning": {
            "strategy": "grid",
            "evaluation_mode": "supervised",
            "allow_generate": False,
            "objective": _objective_mapping,
        },
        "grid_search": {"fail_fast": False, "max_combinations": None},
        "hyperparameters": {
            "SelfTestModel": [
                {"name": "x", "type": "real", "values": [0.0, 1.0]}
            ]
        },
        "supervised_evaluation": {
            "objective": _objective_mapping,
            "metrics": ["loss"],
            "seeds": [3, 5],
            "split_strategy": "holdout",
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "n_splits": 2,
            "shuffle": True,
            "constraints": [],
        },
        "tuning_artifacts": {
            "output_dir": "src/tuning/reports",
            "write_summary": True,
            "write_trials": True,
            "write_config_snapshot": True,
            "indent": 2,
        },
    }

    def _test_valid_config() -> None:
        report = validate_tuning_config(_valid_config)
        _check(report.is_valid, f"valid config rejected: {report.to_dict()}")
        _check(not report.warnings, "valid config produced unexpected warnings")

    def _test_cross_section_objective() -> None:
        invalid = copy.deepcopy(_valid_config)
        invalid["supervised_evaluation"]["objective"] = {
            "name": "accuracy",
            "direction": "maximize",
        }
        report = validate_tuning_config(invalid)
        _check(not report.is_valid, "objective mismatch was accepted")
        _check(
            any(issue.path == "tuning.objective" for issue in report.errors),
            "objective mismatch was not located precisely",
        )

    def _test_space_and_parameter_validation() -> None:
        space = _valid_config["hyperparameters"]["SelfTestModel"]
        _check(
            validate_search_space(space, TuningStrategy.GRID).is_valid,
            "valid grid space was rejected",
        )
        _check(
            validate_parameters_against_space({"x": 1.0}, space).is_valid,
            "valid candidate was rejected",
        )
        report = validate_parameters_against_space({"x": 2.0, "extra": 1}, space)
        _check(not report.is_valid, "invalid candidate was accepted")
        paths = {issue.path for issue in report.errors}
        _check("parameters.x" in paths, "out-of-space value was not reported")
        _check("parameters.extra" in paths, "unknown parameter was not reported")

    def _test_best_trial_validation() -> None:
        objective = parse_metric_spec(_objective_mapping)
        started = utc_now()
        trials = tuple(
            TrialRecord(
                trial_id=f"trial-{index}",
                run_id="validation-self-test",
                status=TrialStatus.SUCCEEDED,
                parameters={"x": value},
                started_at=started,
                completed_at=utc_now(),
                metrics={"loss": value},
                objective_value=value,
            )
            for index, value in enumerate((2.0, 1.0), start=1)
        )
        result = SearchResult(
            run_id="validation-self-test",
            strategy=TuningStrategy.GRID,
            status=RunStatus.SUCCEEDED,
            objective=objective,
            trials=trials,
            started_at=started,
            completed_at=utc_now(),
            best_trial_id="trial-1",
        )
        report = validate_search_result(result)
        _check(not report.is_valid, "suboptimal declared best trial was accepted")
        _check(
            any(issue.path == "best_trial_id" for issue in report.errors),
            "best-trial error was not reported at best_trial_id",
        )

    _run_test("complete centralized configuration", _test_valid_config)
    _run_test("active objective agreement", _test_cross_section_objective)
    _run_test("search-space and candidate validation", _test_space_and_parameter_validation)
    _run_test("best-trial correctness", _test_best_trial_validation)

    _all_passed = not _failures
    printer.status(
        "",
        f"{4 - len(_failures)}/4 tuning validation tests passed",
        "success" if _all_passed else "error",
    )
    if not _all_passed:
        raise SystemExit(1)
    print("\n=== All tuning validation tests passed ===\n")