from __future__ import annotations

import math

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from .evaluators_memory import EvaluatorsMemory
from .utils.config_loader import get_config_section, load_global_config
from .utils.evaluation_errors import *
from .utils.evaluation_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Performance Budget Evaluator")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"


@dataclass(slots=True)
class BudgetContract:
    agent: str
    max_latency_ms: float
    max_memory_mb: float
    soft_latency_ms: Optional[float] = None
    soft_memory_mb: Optional[float] = None
    severity: str = "error"
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContractCheckResult:
    agent: str
    contract: Dict[str, Any]
    observed: Dict[str, Optional[float]]
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceBudgetEvaluator:
    """Evaluates cross-agent latency/memory contracts against observed metrics."""

    def __init__(self) -> None:
        self.config = load_global_config()
        self.config_path = str(self.config.get("__config_path__", "<config>"))
        section = get_config_section("performance_budget_evaluator") or {}
        if not isinstance(section, Mapping):
            raise ConfigLoadError(self.config_path, "performance_budget_evaluator", "Section must be mapping")

        self.enabled = bool(section.get("enabled", True))
        self.strict_mode = bool(section.get("strict_mode", True))
        self.store_results = bool(section.get("store_results", True))
        self.contracts = self._load_contracts(section.get("contracts", []))
        self.memory = EvaluatorsMemory()

    def evaluate(self, observed_metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "metadata": {"enabled": False, "evaluated_at": self._utc_now()},
                "status": "skipped",
                "contracts": [],
                "summary": {"passed": None, "violations": 0, "warnings": 0, "missing_evidence": 0},
            }
        if not isinstance(observed_metrics, Mapping):
            raise ValidationFailureError(
                rule_name="performance_budget_observed_metrics_mapping",
                data=type(observed_metrics).__name__,
                expected="mapping of agent->metrics",
            )

        checks: List[ContractCheckResult] = []
        total_violations = 0
        total_warnings = 0
        total_missing = 0

        for contract in self.contracts.values():
            observed = observed_metrics.get(contract.agent, {}) if isinstance(observed_metrics.get(contract.agent, {}), Mapping) else {}
            latency = self._observed_value(observed.get("latency_ms"))
            memory = self._observed_value(observed.get("memory_mb"))

            check = ContractCheckResult(
                agent=contract.agent,
                contract=contract.to_dict(),
                observed={"latency_ms": latency, "memory_mb": memory},
            )

            if latency is None:
                check.passed = False
                total_missing += 1
                check.warnings.append("Missing or invalid latency_ms: budget unverified.")
            elif latency > contract.max_latency_ms:
                check.passed = False
                check.violations.append({
                    "metric": "latency_ms",
                    "limit": contract.max_latency_ms,
                    "observed": latency,
                    "severity": contract.severity,
                })
            elif contract.soft_latency_ms is not None and latency > contract.soft_latency_ms:
                check.warnings.append(
                    f"Latency approaching hard limit ({latency:.2f}ms > soft {contract.soft_latency_ms:.2f}ms)."
                )

            if memory is None:
                check.passed = False
                total_missing += 1
                check.warnings.append("Missing or invalid memory_mb: budget unverified.")
            elif memory > contract.max_memory_mb:
                check.passed = False
                check.violations.append({
                    "metric": "memory_mb",
                    "limit": contract.max_memory_mb,
                    "observed": memory,
                    "severity": contract.severity,
                })
            elif contract.soft_memory_mb is not None and memory > contract.soft_memory_mb:
                check.warnings.append(
                    f"Memory approaching hard limit ({memory:.2f}MB > soft {contract.soft_memory_mb:.2f}MB)."
                )

            total_violations += len(check.violations)
            total_warnings += len(check.warnings)
            checks.append(check)

        passed = total_violations == 0 and total_missing == 0 and bool(checks)
        status = (
            ("fail" if self.strict_mode else "warn") if total_violations
            else "incomplete" if total_missing
            else "no_contracts" if not checks
            else "pass"
        )
        payload = {
            "metadata": {
                "module_version": MODULE_VERSION,
                "config_path": self.config_path,
                "evaluated_at": self._utc_now(),
                "strict_mode": self.strict_mode,
                "contracts_count": len(self.contracts),
            },
            "status": status,
            "contracts": [c.to_dict() for c in checks],
            "summary": {
                "passed": passed,
                "violations": total_violations,
                "warnings": total_warnings,
                "missing_evidence": total_missing,
            },
        }
        if self.store_results:
            self.memory.add_evaluation_result("performance_budget", payload)
        return payload

    def _load_contracts(self, contracts_payload: Any) -> Dict[str, BudgetContract]:
        contracts: Dict[str, BudgetContract] = {}
        if not isinstance(contracts_payload, list):
            raise ValueError("Performance budget contracts must be a list")
        for item in contracts_payload:
            if not isinstance(item, Mapping):
                raise ValueError("Performance budget contract must be a mapping")
            agent = str(item.get("agent", "")).strip()
            if not agent:
                raise ValueError("Performance budget contract is missing agent")
            if agent in contracts:
                raise ValueError(f"Duplicate performance budget contract: {agent}")
            contract = BudgetContract(
                agent=agent,
                max_latency_ms=self._to_float(item.get("max_latency_ms"), default=1000.0),
                max_memory_mb=self._to_float(item.get("max_memory_mb"), default=1024.0),
                soft_latency_ms=self._to_optional_float(item.get("soft_latency_ms")),
                soft_memory_mb=self._to_optional_float(item.get("soft_memory_mb")),
                severity=str(item.get("severity", "error")),
                enabled=bool(item.get("enabled", True)),
            )
            if contract.enabled:
                if contract.soft_latency_ms is not None and contract.soft_latency_ms > contract.max_latency_ms:
                    raise ValueError(f"Soft latency exceeds hard budget for {agent}")
                if contract.soft_memory_mb is not None and contract.soft_memory_mb > contract.max_memory_mb:
                    raise ValueError(f"Soft memory exceeds hard budget for {agent}")
                contracts[agent] = contract
        return contracts

    @staticmethod
    def _to_float(value: Any, *, default: float) -> float:
        if value is None:
            value = default
        if isinstance(value, bool):
            raise ValueError("Budget limit cannot be boolean")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Budget limit must be a positive finite number") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError("Budget limit must be a positive finite number")
        return number

    @staticmethod
    def _observed_value(value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and number >= 0 else None

    @classmethod
    def _to_optional_float(cls, value: Any) -> Optional[float]:
        if value is None:
            return None
        return cls._to_float(value, default=0.0)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()


__all__ = [
    "BudgetContract",
    "ContractCheckResult",
    "PerformanceBudgetEvaluator",
]