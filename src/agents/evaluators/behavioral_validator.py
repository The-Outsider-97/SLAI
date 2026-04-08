
from __future__ import annotations

import yaml
import json
import inspect
import hashlib

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence
from PyQt5.QtCore import QSize, QBuffer

from .utils.config_loader import load_global_config, get_config_section
from .utils.evaluation_errors import (ConfigLoadError, ReportGenerationError, ValidationFailureError,
                                      MemoryAccessError, MetricCalculationError, OperationalError,
                                      ThresholdViolationError, SerializationError)
from .utils.evaluators_calculations import EvaluatorsCalculations
from .modules.report import get_visualizer
from .evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Behavioral Validator")
printer = PrettyPrinter

MODULE_VERSION = "2.0.0"


@dataclass(slots=True)
class BehavioralTestCase:
    """Normalized behavioral test case following the SUT testing model."""

    test_id: str
    scenario: Dict[str, Any]
    oracle: Callable[..., bool]
    severity: str = "medium"
    requirement_id: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    expected_output: Any = None
    detection_method: str = "oracle"
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_case_id: Optional[str] = None
    is_mutation: bool = False
    mutation_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["oracle"] = getattr(self.oracle, "__name__", "<callable>")
        return payload


@dataclass(slots=True)
class FailureMode:
    """Structured failure analysis result with FMEA-oriented scoring."""

    test_id: str
    requirement_id: Optional[str]
    severity: str
    severity_score: int
    occurrence_score: int
    detectability_score: int
    criticality_score: int
    risk_priority_number: int
    detection_mechanism: str
    category: str
    message: str
    timestamp: str
    observed_output: Any = None
    expected_output: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TestExecutionRecord:
    """Execution artifact for an individual behavioral test."""

    test_id: str
    requirement_id: Optional[str]
    passed: bool
    status: str
    severity: str
    scenario_summary: Dict[str, Any]
    output: Any
    expected_output: Any
    execution_time: float
    timestamp: str
    oracle_result: Optional[bool]
    oracle_message: Optional[str] = None
    exception: Optional[str] = None
    failure_mode: Optional[FailureMode] = None
    tags: List[str] = field(default_factory=list)
    mutation_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.failure_mode is not None:
            payload["failure_mode"] = self.failure_mode.to_dict()
        return payload


class BehavioralValidator:
    """
    Production-grade behavioral testing framework implementing the
    System Under Test (SUT) paradigm described by Ammann & Black.

    Responsibilities
    ----------------
    - Normalize and validate behavioral test cases
    - Execute a SUT against scenarios with oracle-based assessment
    - Track requirement coverage and failure modes
    - Support optional mutation-style robustness variants
    - Persist evaluation outcomes through evaluator memory
    - Integrate with the shared reporting/visualization stack
    """

    _SEVERITY_SCORES: Dict[str, int] = {
        "low": 1,
        "medium": 3,
        "high": 5,
        "critical": 8,
    }
    _DETECTABILITY_SCORES: Dict[str, int] = {
        "oracle": 2,
        "automated": 1,
        "manual": 3,
        "exception": 2,
        "monitoring": 2,
    }

    def __init__(
        self,
        test_cases: Optional[Sequence[Mapping[str, Any]]] = None,
        sut: Optional[Any] = None,
    ) -> None:
        self.config = load_global_config()
        self.config_path = str(self.config.get("__config_path__", "<config>"))

        self.validator_config = get_config_section("behavioral_validator")
        if not isinstance(self.validator_config, Mapping):
            raise ConfigLoadError(self.config_path, "behavioral_validator", "section must be a mapping")

        self.enable_historical = bool(
            self.validator_config.get("enable_historical", self.config.get("enable_historical", True))
        )
        self.store_results = bool(
            self.validator_config.get("store_results", self.config.get("store_results", True))
        )
        self.mutation_testing = bool(self.validator_config.get("mutation_testing", False))
        self.requirement_tags = self._normalize_string_list(
            self.validator_config.get("requirement_tags", []),
            "behavioral_validator.requirement_tags",
            allow_empty=True,
        )
        self.max_failure_modes = self._require_positive_integer(
            self.validator_config.get("max_failure_modes", 10),
            "behavioral_validator.max_failure_modes",
        )
        self.thresholds = self._load_thresholds(self.validator_config.get("thresholds", {}))
        self.weights = self._load_weights(self.validator_config.get("weights", {}))

        self.memory = EvaluatorsMemory()
        self.calculations = EvaluatorsCalculations()
        self.test_cases: List[BehavioralTestCase] = []
        self.failure_modes: List[FailureMode] = []
        self.historical_results: List[Dict[str, Any]] = []
        self.requirement_coverage: set[str] = set()

        self.sut = self._coerce_sut(sut) if sut is not None else self.default_sut()

        if test_cases:
            self.set_test_cases(test_cases)

        logger.info(
            "Behavioral Validator initialized: tests=%d mutation_testing=%s store_results=%s",
            len(self.test_cases),
            self.mutation_testing,
            self.store_results,
        )

    # ------------------------------------------------------------------
    # SUT lifecycle
    # ------------------------------------------------------------------

    def set_sut(self, sut: Any) -> Callable[[Mapping[str, Any]], Any]:
        self.sut = self._coerce_sut(sut)
        return self.sut

    def default_sut(self, model: Optional[Any] = None) -> Callable[[Mapping[str, Any]], Any]:
        """
        Return a default SUT wrapper.

        The wrapper expects a scenario mapping containing an ``input`` field and
        invokes ``model.predict(input_data)``.
        """
        model_instance = model if model is not None else self.load_default_model()
        if not hasattr(model_instance, "predict") or not callable(model_instance.predict):
            raise ValidationFailureError("default_sut_model", type(model_instance).__name__, "object with callable predict()")

        def sut(scenario: Mapping[str, Any]) -> Any:
            if not isinstance(scenario, Mapping):
                raise ValidationFailureError("scenario_mapping", type(scenario).__name__, "mapping")
            if "input" not in scenario:
                raise ValidationFailureError("scenario_input", scenario, "scenario containing 'input'")
            return model_instance.predict(scenario["input"])

        return sut

    def load_default_model(self) -> Any:
        """Return a deterministic default model used for smoke testing."""
        class DummyModel:
            def predict(self, input_data: Any) -> Any:
                if input_data == "test_input":
                    return "expected_output"
                return {"prediction": "unexpected_output", "input": input_data}

        logger.info("Using DummyModel as fallback SUT.")
        return DummyModel()

    def _coerce_sut(self, sut: Any) -> Callable[[Mapping[str, Any]], Any]:
        if sut is None:
            raise ValidationFailureError("sut_instance", sut, "callable or object with callable predict()")

        if hasattr(sut, "predict") and callable(sut.predict):
            def wrapped(scenario: Mapping[str, Any]) -> Any:
                if not isinstance(scenario, Mapping):
                    raise ValidationFailureError("scenario_mapping", type(scenario).__name__, "mapping")
                if "input" not in scenario:
                    raise ValidationFailureError("scenario_input", scenario, "scenario containing 'input'")
                return sut.predict(scenario["input"])

            return wrapped

        if callable(sut):
            def wrapped_callable(scenario: Mapping[str, Any]) -> Any:
                if not isinstance(scenario, Mapping):
                    raise ValidationFailureError("scenario_mapping", type(scenario).__name__, "mapping")
                return sut(scenario)

            return wrapped_callable

        raise ValidationFailureError("sut_instance", type(sut).__name__, "callable or object with callable predict()")

    # ------------------------------------------------------------------
    # Test-case lifecycle
    # ------------------------------------------------------------------

    def set_test_cases(self, test_cases: Sequence[Mapping[str, Any]]) -> List[BehavioralTestCase]:
        normalized = [self._normalize_test_case(case, index=index) for index, case in enumerate(test_cases, start=1)]
        self.test_cases = normalized
        return list(self.test_cases)

    def add_test_case(self, test_case: Mapping[str, Any]) -> BehavioralTestCase:
        normalized = self._normalize_test_case(test_case, index=len(self.test_cases) + 1)
        self.test_cases.append(normalized)
        return normalized

    def clear_test_cases(self) -> None:
        self.test_cases.clear()

    def _normalize_test_case(self, raw_case: Mapping[str, Any], index: int) -> BehavioralTestCase:
        if not isinstance(raw_case, Mapping):
            raise ValidationFailureError("behavioral_test_case", type(raw_case).__name__, "mapping")

        if "scenario" not in raw_case:
            raise ValidationFailureError("behavioral_test_case.scenario", raw_case, "test case with 'scenario'")
        if "oracle" not in raw_case:
            raise ValidationFailureError("behavioral_test_case.oracle", raw_case, "test case with 'oracle'")

        scenario_raw = raw_case["scenario"]
        if not isinstance(scenario_raw, Mapping):
            raise ValidationFailureError("behavioral_test_case.scenario_mapping", type(scenario_raw).__name__, "mapping")

        oracle = raw_case["oracle"]
        if not callable(oracle):
            raise ValidationFailureError("behavioral_test_case.oracle_callable", type(oracle).__name__, "callable")

        test_id = str(raw_case.get("id") or raw_case.get("test_id") or f"TEST-{index:05d}").strip()
        if not test_id:
            raise ValidationFailureError("behavioral_test_case.id", raw_case.get("id"), "non-empty test id")

        severity = self._normalize_severity(raw_case.get("severity", "medium"))
        requirement_id = raw_case.get("requirement_id") or scenario_raw.get("requirement_id")
        normalized_requirement_id = (
            self._normalize_non_empty_string(requirement_id, "requirement_id") if requirement_id is not None else None
        )

        description = str(raw_case.get("description", "")).strip()
        tags = self._normalize_string_list(raw_case.get("tags", []), "behavioral_test_case.tags", allow_empty=True)
        expected_output = raw_case.get("expected_output", scenario_raw.get("expected_output"))
        detection_method = self._normalize_non_empty_string(
            raw_case.get("detection_method", scenario_raw.get("detection_method", "oracle")),
            "detection_method",
        )
        metadata = dict(raw_case.get("metadata", {})) if isinstance(raw_case.get("metadata", {}), Mapping) else {}

        return BehavioralTestCase(
            test_id=test_id,
            scenario=dict(scenario_raw),
            oracle=oracle,
            severity=severity,
            requirement_id=normalized_requirement_id,
            description=description,
            tags=tags,
            expected_output=expected_output,
            detection_method=detection_method,
            metadata=metadata,
            source_case_id=raw_case.get("source_case_id"),
            is_mutation=bool(raw_case.get("is_mutation", False)),
            mutation_label=raw_case.get("mutation_label"),
        )

    def _expand_test_cases(self, cases: Sequence[BehavioralTestCase]) -> List[BehavioralTestCase]:
        expanded = list(cases)
        if not self.mutation_testing:
            return expanded

        mutated_cases: List[BehavioralTestCase] = []
        for case in cases:
            for mutation_label, mutated_scenario in self._generate_mutation_variants(case.scenario).items():
                mutated_cases.append(
                    BehavioralTestCase(
                        test_id=f"{case.test_id}::{mutation_label}",
                        scenario=mutated_scenario,
                        oracle=case.oracle,
                        severity=case.severity,
                        requirement_id=case.requirement_id,
                        description=f"{case.description} [mutation:{mutation_label}]".strip(),
                        tags=list(dict.fromkeys([*case.tags, "mutation"])),
                        expected_output=case.expected_output,
                        detection_method=case.detection_method,
                        metadata=dict(case.metadata),
                        source_case_id=case.test_id,
                        is_mutation=True,
                        mutation_label=mutation_label,
                    )
                )
        return expanded + mutated_cases

    def _generate_mutation_variants(self, scenario: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Generate lightweight mutation-style scenario variants.

        This is not source-code mutation testing; it is scenario mutation used to
        stress robustness at the SUT boundary.
        """
        mutations: Dict[str, Dict[str, Any]] = {}
        base = dict(scenario)

        if "input" in base:
            value = base["input"]
            if isinstance(value, bool):
                variant = dict(base)
                variant["input"] = not value
                mutations["flip_boolean_input"] = variant
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                variant = dict(base)
                variant["input"] = value * 1.1 if value != 0 else 1
                mutations["perturb_numeric_input"] = variant
            elif isinstance(value, str) and value:
                variant = dict(base)
                variant["input"] = value + "_mutated"
                mutations["mutate_string_input"] = variant
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                variant = dict(base)
                variant["input"] = list(value)[::-1]
                mutations["reverse_sequence_input"] = variant

        if "context" in base and isinstance(base["context"], Mapping):
            variant = dict(base)
            trimmed_context = dict(base["context"])
            if trimmed_context:
                trimmed_context.pop(next(iter(trimmed_context.keys())))
                variant["context"] = trimmed_context
                mutations["drop_context_field"] = variant

        return mutations

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_test_suite(
        self,
        agent: Optional[Any] = None,
        *,
        sut: Optional[Any] = None,
        test_cases: Optional[Sequence[Mapping[str, Any] | BehavioralTestCase]] = None,
        suite_name: str = "behavioral_validation",
    ) -> Dict[str, Any]:
        """
        Execute the behavioral test suite against the supplied SUT.

        Backward compatibility:
        - ``agent`` may be a callable or an object exposing ``predict``
        - ``sut`` may be provided explicitly
        """
        active_sut = self._coerce_sut(sut or agent or self.sut)
        selected_cases = self._prepare_execution_cases(test_cases or self.test_cases)
        if not selected_cases:
            return self._empty_results(suite_name=suite_name)

        self.requirement_coverage.clear()
        self.failure_modes.clear()

        execution_records: List[TestExecutionRecord] = []
        anomalies: List[Dict[str, Any]] = []
        start_ts = _utcnow()

        for case in selected_cases:
            try:
                record = self._execute_single_case(active_sut, case)
                execution_records.append(record)
                if record.passed and record.requirement_id:
                    self.requirement_coverage.add(record.requirement_id)
                if record.failure_mode is not None:
                    self.failure_modes.append(record.failure_mode)
            except Exception as exc:
                anomaly = {
                    "test_id": getattr(case, "test_id", "<unknown>"),
                    "error": str(exc),
                    "type": exc.__class__.__name__,
                    "timestamp": _utcnow().isoformat(),
                }
                anomalies.append(anomaly)
                logger.error("Behavioral test execution error for %s: %s", anomaly["test_id"], exc)
                self._store_error_if_possible(
                    OperationalError(
                        message=f"Behavioral test execution failed for {anomaly['test_id']}",
                        context=anomaly,
                        cause=exc,
                    )
                )

        results = self._build_results_payload(
            suite_name=suite_name,
            execution_records=execution_records,
            anomalies=anomalies,
            started_at=start_ts,
            completed_at=_utcnow(),
            source="behavioral_validation",
        )

        self._persist_results(results, tags=["behavioral_eval", suite_name], category="behavioral_validation")
        self.historical_results.append(self._historical_snapshot(results))
        self._update_visualizer(results)

        return results

    def execute_certification_suite(
        self,
        sut: Any,
        certification_requirements: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute a stricter, traceable certification-oriented behavioral suite.

        Critical failures halt further execution immediately.
        """
        active_sut = self._coerce_sut(sut)
        if not certification_requirements:
            raise ValidationFailureError(
                "certification_requirements",
                certification_requirements,
                "non-empty sequence of requirement test cases",
            )

        cases = self._prepare_execution_cases(certification_requirements)
        traceability_matrix: List[Dict[str, Any]] = []
        evidence_log: List[Dict[str, Any]] = []
        execution_records: List[TestExecutionRecord] = []
        anomalies: List[Dict[str, Any]] = []
        overall_status = "PASSED"
        start_ts = _utcnow()

        self.requirement_coverage.clear()
        self.failure_modes.clear()

        for case in cases:
            try:
                record = self._execute_single_case(active_sut, case)
                execution_records.append(record)
                if record.passed and record.requirement_id:
                    self.requirement_coverage.add(record.requirement_id)

                traceability_matrix.append(
                    {
                        "requirement_id": record.requirement_id or case.requirement_id or "UNSPECIFIED",
                        "test_id": record.test_id,
                        "status": record.status,
                        "passed": record.passed,
                        "severity": record.severity,
                    }
                )
                evidence_log.append(
                    {
                        "timestamp": record.timestamp,
                        "test_id": record.test_id,
                        "requirement_id": record.requirement_id,
                        "status": record.status,
                        "details": record.oracle_message or record.exception or "Validation successful.",
                    }
                )

                if record.failure_mode is not None:
                    self.failure_modes.append(record.failure_mode)

                if not record.passed and record.severity == "critical":
                    overall_status = "FAILED"
                    logger.critical(
                        "Critical certification requirement failed: test=%s requirement=%s",
                        record.test_id,
                        record.requirement_id,
                    )
                    break
            except Exception as exc:
                anomaly = {
                    "test_id": case.test_id,
                    "requirement_id": case.requirement_id,
                    "error": str(exc),
                    "type": exc.__class__.__name__,
                    "timestamp": _utcnow().isoformat(),
                }
                anomalies.append(anomaly)
                traceability_matrix.append(
                    {
                        "requirement_id": case.requirement_id or "UNSPECIFIED",
                        "test_id": case.test_id,
                        "status": "ERROR",
                        "passed": False,
                        "severity": case.severity,
                    }
                )
                evidence_log.append(
                    {
                        "timestamp": anomaly["timestamp"],
                        "test_id": case.test_id,
                        "requirement_id": case.requirement_id,
                        "status": "ERROR",
                        "details": anomaly["error"],
                    }
                )
                overall_status = "FAILED"
                logger.error("Certification execution failed for %s: %s", case.test_id, exc)
                if case.severity == "critical":
                    break

        summary = self._build_results_payload(
            suite_name="behavioral_certification",
            execution_records=execution_records,
            anomalies=anomalies,
            started_at=start_ts,
            completed_at=_utcnow(),
            source="behavioral_certification",
        )
        summary.update(
            {
                "overall_status": overall_status if not anomalies else "FAILED",
                "traceability_matrix": traceability_matrix,
                "evidence_log": evidence_log,
            }
        )

        self._persist_results(summary, tags=["behavioral_certification", "certification"], category="certification")
        self.historical_results.append(self._historical_snapshot(summary))
        self._update_visualizer(summary)

        return summary

    def _execute_single_case(
        self,
        sut: Callable[[Mapping[str, Any]], Any],
        case: BehavioralTestCase,
    ) -> TestExecutionRecord:
        start = datetime.now(timezone.utc).timestamp()
        output: Any = None
        exception_message: Optional[str] = None
        oracle_message: Optional[str] = None
        oracle_result: Optional[bool] = None
        status = "PASSED"

        try:
            output = sut(case.scenario)
            oracle_result = self._invoke_oracle(case.oracle, output, case.scenario, case)
            if not isinstance(oracle_result, bool):
                raise ValidationFailureError("oracle_return_type", type(oracle_result).__name__, "bool")
            if not oracle_result:
                status = "FAILED"
                oracle_message = "Oracle returned False."
        except Exception as exc:
            exception_message = str(exc)
            oracle_result = False
            status = "ERROR"
            output = None
            logger.warning("SUT/oracle execution failed for test %s: %s", case.test_id, exc)

        execution_time = datetime.now(timezone.utc).timestamp() - start
        passed = bool(oracle_result) and exception_message is None

        failure_mode = None
        if not passed:
            failure_mode = self._analyze_failure(case, output, exception_message or oracle_message or "Assertion failed")

        record = TestExecutionRecord(
            test_id=case.test_id,
            requirement_id=case.requirement_id,
            passed=passed,
            status=status,
            severity=case.severity,
            scenario_summary=self._summarize_scenario(case.scenario),
            output=output,
            expected_output=case.expected_output,
            execution_time=float(execution_time),
            timestamp=_utcnow().isoformat(),
            oracle_result=oracle_result,
            oracle_message=oracle_message,
            exception=exception_message,
            failure_mode=failure_mode,
            tags=list(case.tags),
            mutation_label=case.mutation_label,
        )
        return record

    def _invoke_oracle(
        self,
        oracle: Callable[..., bool],
        output: Any,
        scenario: Mapping[str, Any],
        case: BehavioralTestCase,
    ) -> bool:
        signature = inspect.signature(oracle)
        parameter_count = len(
            [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
        )

        if parameter_count <= 1:
            return bool(oracle(output))
        if parameter_count == 2:
            return bool(oracle(output, scenario))
        return bool(oracle(output, scenario, case))

    def _prepare_execution_cases(
        self,
        test_cases: Sequence[Mapping[str, Any] | BehavioralTestCase],
    ) -> List[BehavioralTestCase]:
        normalized: List[BehavioralTestCase] = []
        for index, case in enumerate(test_cases, start=1):
            if isinstance(case, BehavioralTestCase):
                normalized.append(case)
            else:
                normalized.append(self._normalize_test_case(case, index=index))
        return self._expand_test_cases(normalized)

    # ------------------------------------------------------------------
    # Analysis and scoring
    # ------------------------------------------------------------------

    def _analyze_failure(self, case: BehavioralTestCase, output: Any, message: str) -> FailureMode:
        severity_score = self._SEVERITY_SCORES[case.severity]
        occurrence_score = self._estimate_occurrence_score(case.test_id)
        detectability_score = self._DETECTABILITY_SCORES.get(case.detection_method.lower(), 2)
        criticality_score = severity_score * max(1, detectability_score)
        rpn = severity_score * occurrence_score * detectability_score
        category = self._infer_failure_category(case, output, message)

        return FailureMode(
            test_id=case.test_id,
            requirement_id=case.requirement_id,
            severity=case.severity,
            severity_score=severity_score,
            occurrence_score=occurrence_score,
            detectability_score=detectability_score,
            criticality_score=criticality_score,
            risk_priority_number=rpn,
            detection_mechanism=case.detection_method,
            category=category,
            message=message,
            timestamp=_utcnow().isoformat(),
            observed_output=output,
            expected_output=case.expected_output,
        )

    def _estimate_occurrence_score(self, test_id: str) -> int:
        history_count = sum(
            1
            for snapshot in self.historical_results
            for failure in snapshot.get("failure_modes", [])
            if failure.get("test_id") == test_id
        )
        return min(10, max(1, history_count + 1))

    def _infer_failure_category(self, case: BehavioralTestCase, output: Any, message: str) -> str:
        lowered = str(message).lower()
        if "exception" in lowered or "traceback" in lowered:
            return "execution_exception"
        if output is None:
            return "null_output"
        if case.expected_output is not None and output != case.expected_output:
            return "oracle_mismatch"
        if case.mutation_label:
            return "mutation_robustness_failure"
        return "behavioral_nonconformance"

    def _build_results_payload(
        self,
        *,
        suite_name: str,
        execution_records: Sequence[TestExecutionRecord],
        anomalies: Sequence[Mapping[str, Any]],
        started_at: datetime,
        completed_at: datetime,
        source: str,
    ) -> Dict[str, Any]:
        executed = len(execution_records)
        passed = sum(1 for record in execution_records if record.passed)
        failed = sum(1 for record in execution_records if record.status == "FAILED")
        errored = sum(1 for record in execution_records if record.status == "ERROR")
        requirements_encountered = {
            record.requirement_id for record in execution_records if record.requirement_id is not None
        }

        pass_rate = (passed / executed) if executed > 0 else 0.0
        failure_rate = ((failed + errored) / executed) if executed > 0 else 0.0
        requirement_coverage_rate = (
            len(self.requirement_coverage) / len(requirements_encountered)
            if requirements_encountered
            else 0.0
        )
        test_coverage = 1.0 if executed > 0 else 0.0
        composite_score = self._calculate_composite_score(
            pass_rate=pass_rate,
            test_coverage=test_coverage,
            requirement_coverage=requirement_coverage_rate,
        )

        execution_times = [record.execution_time for record in execution_records]
        statistical_summary = self._build_statistical_summary(execution_times, self.failure_modes)
        threshold_events = self._evaluate_thresholds(pass_rate=pass_rate, failure_rate=failure_rate)

        payload = {
            "metadata": {
                "timestamp": completed_at.isoformat(),
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "validator_version": MODULE_VERSION,
                "suite_name": suite_name,
                "source": source,
            },
            "summary": {
                "total_cases": executed,
                "passed": passed,
                "failed": failed,
                "errored": errored,
                "anomaly_count": len(anomalies),
                "pass_rate": pass_rate,
                "failure_rate": failure_rate,
                "test_coverage": test_coverage,
                "test_coverage_count": executed,
                "requirement_coverage": len(self.requirement_coverage),
                "requirement_coverage_rate": requirement_coverage_rate,
                "composite_score": composite_score,
                "mean_execution_time": mean(execution_times) if execution_times else 0.0,
                "median_execution_time": median(execution_times) if execution_times else 0.0,
            },
            "thresholds": {
                "configured": dict(self.thresholds),
                "events": threshold_events,
                "passed": len(threshold_events) == 0,
            },
            "records": [record.to_dict() for record in execution_records],
            "failure_modes": [failure.to_dict() for failure in self.failure_modes[: self.max_failure_modes]],
            "anomalies": [dict(item) for item in anomalies],
            "statistics": statistical_summary,
            "weights": dict(self.weights),
            "historical_context": self._build_history_summary(),
        }
        return payload

    def _calculate_composite_score(
        self,
        *,
        pass_rate: float,
        test_coverage: float,
        requirement_coverage: float,
    ) -> float:
        return (
            self.weights.get("pass_rate", 0.0) * pass_rate
            + self.weights.get("test_coverage", 0.0) * test_coverage
            + self.weights.get("requirement_coverage", 0.0) * requirement_coverage
        )

    def _build_statistical_summary(
        self,
        execution_times: Sequence[float],
        failure_modes: Sequence[FailureMode],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "execution_time": {
                "count": len(execution_times),
                "mean": mean(execution_times) if execution_times else 0.0,
                "median": median(execution_times) if execution_times else 0.0,
                "min": min(execution_times) if execution_times else 0.0,
                "max": max(execution_times) if execution_times else 0.0,
            },
            "failure_modes": {
                "count": len(failure_modes),
                "mean_rpn": mean([mode.risk_priority_number for mode in failure_modes]) if failure_modes else 0.0,
                "max_rpn": max([mode.risk_priority_number for mode in failure_modes], default=0),
            },
        }

        if execution_times:
            try:
                summary["inferential"] = self.calculations.calculate_statistical_analysis(
                    {"execution_time": list(execution_times)}
                )
            except Exception as exc:
                logger.info("Inferential behavioral statistics skipped: %s", exc)
                summary["inferential"] = {
                    "status": "skipped",
                    "reason": str(exc),
                }
        else:
            summary["inferential"] = {"status": "skipped", "reason": "no execution data"}

        return summary

    def _evaluate_thresholds(self, *, pass_rate: float, failure_rate: float) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        min_pass_rate = self.thresholds.get("pass_rate")
        max_failure_tolerance = self.thresholds.get("failure_tolerance")

        if pass_rate < min_pass_rate:
            events.append(
                ThresholdViolationError("behavioral_pass_rate", pass_rate, min_pass_rate, comparator="minimum").to_audit_dict()
            )
        if failure_rate > max_failure_tolerance:
            events.append(
                ThresholdViolationError(
                    "behavioral_failure_tolerance",
                    failure_rate,
                    max_failure_tolerance,
                    comparator="maximum",
                ).to_audit_dict()
            )
        return events

    def _build_history_summary(self) -> Dict[str, Any]:
        if not self.historical_results:
            return {
                "enabled": self.enable_historical,
                "runs_tracked": 0,
                "pass_rates": [],
                "average_pass_rate": 0.0,
            }

        pass_rates = [float(item["summary"]["pass_rate"]) for item in self.historical_results if "summary" in item]
        return {
            "enabled": self.enable_historical,
            "runs_tracked": len(self.historical_results),
            "pass_rates": pass_rates,
            "average_pass_rate": mean(pass_rates) if pass_rates else 0.0,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(
        self,
        results: Mapping[str, Any],
        *,
        format: str = "markdown",
        include_visualizations: bool = True,
    ) -> Any:
        """Generate a behavioral validation report in markdown, json, yaml, or dict form."""
        if not isinstance(results, Mapping):
            raise ReportGenerationError(
                report_type="Behavioral Validation",
                template="behavioral_report",
                error_details="results must be a mapping",
            )

        normalized_format = str(format).strip().lower()
        try:
            if normalized_format == "dict":
                return dict(results)

            enriched = dict(results)
            if include_visualizations:
                enriched["visualizations"] = self._generate_visualization_assets(results)

            if normalized_format == "json":
                return json.dumps(enriched, indent=2, sort_keys=False, default=str)
            if normalized_format == "yaml":
                return yaml.safe_dump(enriched, default_flow_style=False, sort_keys=False)
            if normalized_format == "markdown":
                return self._build_markdown_report(enriched)

            raise ReportGenerationError(
                report_type="Behavioral Validation",
                template=normalized_format,
                error_details="Unsupported report format. Use markdown, json, yaml, or dict.",
            )
        except ReportGenerationError:
            raise
        except Exception as exc:
            raise ReportGenerationError(
                report_type="Behavioral Validation",
                template=normalized_format,
                error_details=str(exc),
            ) from exc

    def _build_markdown_report(self, results: Mapping[str, Any]) -> str:
        summary = results.get("summary", {})
        thresholds = results.get("thresholds", {})
        failure_modes = results.get("failure_modes", [])
        anomalies = results.get("anomalies", [])
        statistics = results.get("statistics", {})
        historical = results.get("historical_context", {})
        visualizations = results.get("visualizations", {})

        lines: List[str] = []
        lines.append("# Behavioral Validation Report")
        lines.append("")
        lines.append(f"**Generated**: {results.get('metadata', {}).get('timestamp', _utcnow().isoformat())}")
        lines.append(f"**Validator Version**: {results.get('metadata', {}).get('validator_version', MODULE_VERSION)}")
        lines.append(f"**Suite**: {results.get('metadata', {}).get('suite_name', 'behavioral_validation')}")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append(f"- **Total Tests Executed**: {summary.get('test_coverage_count', 0)}")
        lines.append(f"- **Passed**: {summary.get('passed', 0)}")
        lines.append(f"- **Failed**: {summary.get('failed', 0)}")
        lines.append(f"- **Errored**: {summary.get('errored', 0)}")
        lines.append(f"- **Pass Rate**: {summary.get('pass_rate', 0.0):.2%}")
        lines.append(f"- **Requirement Coverage**: {summary.get('requirement_coverage', 0)} ({summary.get('requirement_coverage_rate', 0.0):.2%})")
        lines.append(f"- **Composite Score**: {summary.get('composite_score', 0.0):.4f}")
        lines.append("")

        lines.append("## Threshold Evaluation")
        if thresholds.get("events"):
            for event in thresholds["events"]:
                lines.append(
                    f"- **Violation**: {event.get('message', 'threshold violation')} "
                    f"(severity={event.get('severity', 'unknown')})"
                )
        else:
            lines.append("✅ All configured behavioral thresholds passed.")
        lines.append("")

        lines.append("## Failure Mode Analysis")
        if failure_modes:
            for failure in failure_modes[: self.max_failure_modes]:
                lines.append(
                    f"- **{failure.get('test_id', '<unknown>')}** | "
                    f"Severity: {failure.get('severity', 'unknown').title()} | "
                    f"Category: {failure.get('category', 'unknown')} | "
                    f"RPN: {failure.get('risk_priority_number', 0)} | "
                    f"{failure.get('message', '')}"
                )
        else:
            lines.append("✅ No failure modes were recorded.")
        lines.append("")

        lines.append("## Statistical Summary")
        exec_stats = statistics.get("execution_time", {})
        lines.append(
            f"- **Execution Time**: mean={exec_stats.get('mean', 0.0):.6f}s, "
            f"median={exec_stats.get('median', 0.0):.6f}s, "
            f"min={exec_stats.get('min', 0.0):.6f}s, "
            f"max={exec_stats.get('max', 0.0):.6f}s"
        )
        fail_stats = statistics.get("failure_modes", {})
        lines.append(
            f"- **Failure Criticality**: count={fail_stats.get('count', 0)}, "
            f"mean_rpn={fail_stats.get('mean_rpn', 0.0):.2f}, "
            f"max_rpn={fail_stats.get('max_rpn', 0)}"
        )
        if statistics.get("inferential", {}).get("status") == "skipped":
            lines.append(f"- **Inferential Statistics**: skipped ({statistics['inferential'].get('reason', 'n/a')})")
        lines.append("")

        if anomalies:
            lines.append("## Execution Anomalies")
            for anomaly in anomalies:
                lines.append(
                    f"- **{anomaly.get('test_id', '<unknown>')}**: {anomaly.get('type', 'Error')} - {anomaly.get('error', '')}"
                )
            lines.append("")

        if self.enable_historical:
            lines.append("## Historical Context")
            lines.append(f"- **Tracked Runs**: {historical.get('runs_tracked', 0)}")
            lines.append(f"- **Average Pass Rate**: {historical.get('average_pass_rate', 0.0):.2%}")
            lines.append("")

        if visualizations:
            lines.append("## Visualizations")
            for name, asset in visualizations.items():
                lines.append(f"### {name.replace('_', ' ').title()}")
                lines.append(f"![{name}](data:image/png;base64,{asset['image']})")
                lines.append("")

        lines.append("---")
        lines.append(f"*Report generated by {self.__class__.__name__}*")
        return "\n".join(lines)

    def _generate_visualization_assets(self, results: Mapping[str, Any]) -> Dict[str, Any]:
        visualizer = get_visualizer()
        assets: Dict[str, Any] = {}

        pass_rates = [
            item.get("summary", {}).get("pass_rate", 0.0)
            for item in self.historical_results
            if isinstance(item, Mapping)
        ]
        if results.get("summary"):
            pass_rates = [*pass_rates, results["summary"].get("pass_rate", 0.0)]

        if pass_rates:
            pixmap = visualizer.render_temporal_chart(QSize(600, 400), "pass_rate", data=pass_rates)
            assets["pass_rate_trend"] = self._pixmap_to_asset(pixmap, "pass_rate_trend")

        failure_counts = [
            item.get("summary", {}).get("failed", 0) + item.get("summary", {}).get("errored", 0)
            for item in self.historical_results
            if isinstance(item, Mapping)
        ]
        if results.get("summary"):
            failure_counts = [*failure_counts, results["summary"].get("failed", 0) + results["summary"].get("errored", 0)]
        if failure_counts:
            pixmap = visualizer.render_temporal_chart(QSize(600, 400), "risks", data=failure_counts)
            assets["failure_trend"] = self._pixmap_to_asset(pixmap, "failure_trend")

        return assets

    # ------------------------------------------------------------------
    # Persistence and integration
    # ------------------------------------------------------------------

    def _persist_results(self, results: Mapping[str, Any], *, tags: Sequence[str], category: str) -> None:
        if not self.store_results:
            return

        priority = "high" if results.get("thresholds", {}).get("events") else "medium"
        try:
            if hasattr(self.memory, "add_evaluation_result"):
                self.memory.add_evaluation_result(
                    evaluator_name=self.__class__.__name__,
                    result=dict(results),
                    tags=list(tags),
                    priority=priority,
                    metadata={"category": category},
                )
            else:
                self.memory.add(
                    entry=dict(results),
                    tags=list(tags),
                    priority=priority,
                    source=self.__class__.__name__,
                    category=category,
                    metadata={"validator_version": MODULE_VERSION},
                )
        except Exception as exc:
            raise MemoryAccessError("add", category, str(exc)) from exc

    def _store_error_if_possible(self, error: Exception) -> None:
        try:
            if hasattr(self.memory, "add_error") and hasattr(error, "to_audit_dict"):
                self.memory.add_error(error)  # type: ignore[arg-type]
        except Exception:
            logger.debug("Unable to persist behavioral validation error to memory", exc_info=True)

    def _update_visualizer(self, results: Mapping[str, Any]) -> None:
        try:
            visualizer = get_visualizer()
            summary = results.get("summary", {})
            visualizer.update_metrics(
                {
                    "successes": int(summary.get("passed", 0)),
                    "failures": int(summary.get("failed", 0)) + int(summary.get("errored", 0)),
                    "pass_rate": float(summary.get("pass_rate", 0.0)),
                    "risk": float(summary.get("failure_rate", 0.0)),
                    "reward": float(summary.get("composite_score", 0.0)),
                    "operational_time": float(summary.get("mean_execution_time", 0.0)),
                }
            )
        except Exception as exc:
            logger.info("Behavioral validator metrics were not pushed to shared visualizer: %s", exc)

    def _historical_snapshot(self, results: Mapping[str, Any]) -> Dict[str, Any]:
        snapshot = {
            "metadata": dict(results.get("metadata", {})),
            "summary": dict(results.get("summary", {})),
            "failure_modes": list(results.get("failure_modes", [])),
        }
        return snapshot

    # ------------------------------------------------------------------
    # Administrative controls
    # ------------------------------------------------------------------

    def disable_temporarily(self) -> None:
        """Temporarily disable behavioral testing by clearing active cases."""
        self.test_cases = []
        logger.warning("Behavioral Validator temporarily disabled.")

    def enable_with_test_cases(self, test_cases: Sequence[Mapping[str, Any]]) -> List[BehavioralTestCase]:
        """Re-enable behavioral validation with the supplied test suite."""
        return self.set_test_cases(test_cases)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_results(self, *, suite_name: str) -> Dict[str, Any]:
        now = _utcnow()
        return {
            "metadata": {
                "timestamp": now.isoformat(),
                "started_at": now.isoformat(),
                "completed_at": now.isoformat(),
                "validator_version": MODULE_VERSION,
                "suite_name": suite_name,
                "source": "behavioral_validation",
            },
            "summary": {
                "total_cases": 0,
                "passed": 0,
                "failed": 0,
                "errored": 0,
                "anomaly_count": 0,
                "pass_rate": 0.0,
                "failure_rate": 0.0,
                "test_coverage": 0.0,
                "test_coverage_count": 0,
                "requirement_coverage": 0,
                "requirement_coverage_rate": 0.0,
                "composite_score": 0.0,
                "mean_execution_time": 0.0,
                "median_execution_time": 0.0,
            },
            "thresholds": {
                "configured": dict(self.thresholds),
                "events": [],
                "passed": True,
            },
            "records": [],
            "failure_modes": [],
            "anomalies": [],
            "statistics": {
                "execution_time": {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
                "failure_modes": {"count": 0, "mean_rpn": 0.0, "max_rpn": 0},
                "inferential": {"status": "skipped", "reason": "no data"},
            },
            "weights": dict(self.weights),
            "historical_context": self._build_history_summary(),
        }

    def _summarize_scenario(self, scenario: Mapping[str, Any]) -> Dict[str, Any]:
        summary = {}
        for key, value in scenario.items():
            if key == "input":
                summary[key] = self._summarize_value(value)
            else:
                summary[str(key)] = self._summarize_value(value)
        return summary

    def _summarize_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value if len(value) <= 120 else value[:117] + "..."
        if isinstance(value, Mapping):
            return {str(k): self._summarize_value(v) for k, v in list(value.items())[:10]}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            materialized = list(value)
            preview = [self._summarize_value(item) for item in materialized[:10]]
            return {"type": type(value).__name__, "length": len(materialized), "preview": preview}
        return str(value)

    def _pixmap_to_asset(self, pixmap: Any, chart_type: str) -> Dict[str, Any]:
        if pixmap is None:
            raise SerializationError(chart_type, "Cannot serialize a null pixmap.")
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        if not pixmap.save(buffer, "PNG"):
            raise SerializationError(chart_type, "QPixmap.save returned False.")
        image = bytes(buffer.data().toBase64()).decode("utf-8")
        return {
            "chart_type": chart_type,
            "encoding": "base64/png",
            "width": int(pixmap.width()),
            "height": int(pixmap.height()),
            "image": image,
        }

    def _load_thresholds(self, raw: Mapping[str, Any]) -> Dict[str, float]:
        if not isinstance(raw, Mapping):
            raise ConfigLoadError(self.config_path, "behavioral_validator.thresholds", "must be a mapping")
        pass_rate = self._coerce_probability(raw.get("pass_rate", 0.95), "behavioral_validator.thresholds.pass_rate")
        failure_tolerance = self._coerce_probability(
            raw.get("failure_tolerance", 0.05),
            "behavioral_validator.thresholds.failure_tolerance",
        )
        return {
            "pass_rate": pass_rate,
            "failure_tolerance": failure_tolerance,
        }

    def _load_weights(self, raw: Mapping[str, Any]) -> Dict[str, float]:
        if not isinstance(raw, Mapping):
            raise ConfigLoadError(self.config_path, "behavioral_validator.weights", "must be a mapping")

        weights = {
            "test_coverage": self._coerce_non_negative_float(raw.get("test_coverage", 0.4), "behavioral_validator.weights.test_coverage"),
            "pass_rate": self._coerce_non_negative_float(raw.get("pass_rate", 0.3), "behavioral_validator.weights.pass_rate"),
            "requirement_coverage": self._coerce_non_negative_float(
                raw.get("requirement_coverage", 0.3),
                "behavioral_validator.weights.requirement_coverage",
            ),
        }
        total = sum(weights.values())
        if total <= 0:
            raise ConfigLoadError(self.config_path, "behavioral_validator.weights", "sum of weights must be > 0")
        return {key: value / total for key, value in weights.items()}

    def _normalize_non_empty_string(self, value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValidationFailureError(field_name, value, "non-empty string")
        return value.strip()

    def _normalize_string_list(self, values: Any, field_name: str, *, allow_empty: bool = False) -> List[str]:
        if values is None:
            return []
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise ValidationFailureError(field_name, type(values).__name__, "sequence of strings")
        normalized: List[str] = []
        seen: set[str] = set()
        for item in values:
            text = str(item).strip()
            if not text:
                continue
            key = text.casefold()
            if key not in seen:
                normalized.append(text)
                seen.add(key)
        if not normalized and not allow_empty:
            raise ValidationFailureError(field_name, values, "non-empty sequence of strings")
        return normalized

    def _normalize_severity(self, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in self._SEVERITY_SCORES:
            raise ValidationFailureError("severity", value, f"one of {sorted(self._SEVERITY_SCORES)}")
        return normalized

    def _require_positive_integer(self, value: Any, field_name: str) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigLoadError(self.config_path, field_name, "must be a positive integer") from exc
        if numeric <= 0:
            raise ConfigLoadError(self.config_path, field_name, "must be a positive integer")
        return numeric

    def _coerce_non_negative_float(self, value: Any, field_name: str) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigLoadError(self.config_path, field_name, "must be a non-negative number") from exc
        if numeric < 0:
            raise ConfigLoadError(self.config_path, field_name, "must be a non-negative number")
        return numeric

    def _coerce_probability(self, value: Any, field_name: str) -> float:
        numeric = self._coerce_non_negative_float(value, field_name)
        if numeric > 1.0:
            raise ConfigLoadError(self.config_path, field_name, "must be between 0 and 1")
        return numeric


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    print("\n=== Running Behavioral Validator ===\n")

    test_cases = [
        {
            "id": "REQ-001",
            "scenario": {
                "input": "test_input",
                "requirement_id": "REQ-001",
                "detection_method": "automated",
            },
            "oracle": lambda output: output == "expected_output",
            "severity": "critical",
            "description": "Default dummy model should satisfy the golden input case.",
            "tags": ["smoke", "critical"],
        }
    ]

    validator = BehavioralValidator(test_cases=test_cases)
    results = validator.execute_test_suite()
    print(validator.generate_report(results, format="markdown"))

    cert_results = validator.execute_certification_suite(
        sut=validator.default_sut(),
        certification_requirements=test_cases,
    )
    print(json.dumps(cert_results["summary"], indent=2))
    print("\n=== Successfully Ran Behavioral Validator ===\n")