from __future__ import annotations

import hashlib
import json
import time as time_module

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from .utils.config_loader import get_config_section, load_global_config
from .handler_memory import HandlerMemory
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Failure Intelligence")
printer = PrettyPrinter()


@dataclass(frozen=True)
class FailureInsight:
    signature: str
    confidence: float
    category: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "confidence": self.confidence,
            "category": self.category,
            "recommendation": self.recommendation,
        }


class FailureIntelligence:
    """
    Lightweight failure intelligence for HandlerAgent.

    Scope:
    - categorize failures with deterministic heuristics
    - compute stable signatures for de-duplication/routing
    - produce bounded confidence + recommendation hints
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config = load_global_config()
        self.intelligence_config = get_config_section("intelligence")

        merged: Dict[str, Any] = {}
        if isinstance(self.intelligence_config, Mapping):
            merged.update(self.intelligence_config)
        if isinstance(config, Mapping):
            merged.update(config)

        self.max_message_chars = int(merged.get("max_message_chars", 280))
        self.confidence_floor = float(merged.get("confidence_floor", 0.35))
        self.confidence_ceiling = float(merged.get("confidence_ceiling", 0.92))

    def analyze(
        self,
        normalized_failure: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
        telemetry_history: Optional[List[Dict[str, Any]]] = None,
    ) -> FailureInsight:
        context = context or {}
        telemetry_history = telemetry_history or []

        failure_type = str(normalized_failure.get("type", "UnknownError"))
        failure_message = str(normalized_failure.get("message", ""))
        severity = str(normalized_failure.get("severity", "low")).lower()
        retryable = bool(normalized_failure.get("retryable", False))

        compact_message = failure_message[: self.max_message_chars]
        category = self._categorize(failure_type=failure_type, failure_message=compact_message)
        signature = self._signature(
            failure_type=failure_type,
            failure_message=compact_message,
            category=category,
            context=context,
        )
        confidence = self._confidence(signature=signature, severity=severity, retryable=retryable, telemetry_history=telemetry_history)
        recommendation = self._recommend(category=category, severity=severity, retryable=retryable)

        return FailureInsight(
            signature=signature,
            confidence=confidence,
            category=category,
            recommendation=recommendation,
        )

    def _categorize(self, failure_type: str, failure_message: str) -> str:
        lowered = f"{failure_type} {failure_message}".lower()
        if "timeout" in lowered or "timed out" in lowered:
            return "timeout"
        if any(token in lowered for token in ("network", "connection", "socket", "dns", "http")):
            return "network"
        if any(token in lowered for token in ("memory", "oom", "outofmemory", "cuda")):
            return "memory"
        if any(token in lowered for token in ("import", "module", "dependency", "dll")):
            return "dependency"
        if "unicode" in lowered:
            return "unicode"
        return "runtime"

    def _signature(
        self,
        failure_type: str,
        failure_message: str,
        category: str,
        context: Mapping[str, Any],
    ) -> str:
        payload = {
            "type": failure_type.lower(),
            "message": failure_message.lower(),
            "category": category,
            "route": context.get("route"),
            "agent": context.get("agent"),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return f"{category}:{digest[:16]}"

    def _confidence(
        self,
        signature: str,
        severity: str,
        retryable: bool,
        telemetry_history: List[Dict[str, Any]],
    ) -> float:
        total = 0
        recovered = 0
        for event in telemetry_history:
            failure = event.get("failure", {}) if isinstance(event, dict) else {}
            if failure.get("signature") != signature:
                continue
            total += 1
            recovery = event.get("recovery", {}) if isinstance(event, dict) else {}
            if recovery.get("status") == "recovered":
                recovered += 1

        historical = (recovered / total) if total else 0.5
        severity_penalty = {"critical": -0.18, "high": -0.08, "medium": -0.03}.get(severity, 0.04)
        retry_bonus = 0.06 if retryable else -0.06
        value = historical + severity_penalty + retry_bonus
        return max(self.confidence_floor, min(self.confidence_ceiling, round(value, 3)))

    @staticmethod
    def _recommend(category: str, severity: str, retryable: bool) -> str:
        if severity == "critical" and not retryable:
            return "immediate_escalation"
        if category in {"timeout", "network"} and retryable:
            return "retry_with_backoff"
        if category == "dependency":
            return "validate_runtime_dependencies"
        if category == "memory":
            return "degrade_and_reduce_resource_pressure"
        if category == "unicode":
            return "sanitize_encoding_and_retry"
        return "collect_context_and_escalate"
