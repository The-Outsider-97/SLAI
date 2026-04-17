"""
Route policy evaluation for SLAI's Network Agent routing subsystem.

This module provides the production-grade route-policy layer used by the
Network Stream routing stack. It evaluates candidate transport routes and
endpoint paths against operational, security, and routing constraints to
produce policy-aware routing decisions.

The route-policy layer is intentionally scoped to route evaluation and route
eligibility. It is responsible for:
- evaluating candidate paths against configured constraints and caller intent,
- incorporating endpoint-health, circuit, and channel-metric signals,
- applying cost, latency, reliability, security, and affinity weighting,
- selecting primary and secondary route sets,
- emitting structured policy decisions to shared NetworkMemory.

It does not own endpoint inventory, candidate discovery, transport execution,
or retry/circuit-breaking mechanics. Those concerns belong to Endpoint
Registry, Channel Selector, adapters, and reliability components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils import *
from ..network_memory import NetworkMemory
from .endpoint_registry import EndpointRegistry
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Route Policy")
printer = PrettyPrinter()


_ROUTE_POLICY_LAST_KEY = "network.routing.route_policy.last"
_ROUTE_POLICY_HISTORY_KEY = "network.routing.route_policy.history"
_ROUTE_POLICY_SNAPSHOT_KEY = "network.routing.route_policy.snapshot"

_VALID_SELECTION_STRATEGIES = {"weighted", "priority", "first_viable"}


@dataclass(slots=True)
class RoutePolicyScore:
    """Detailed policy-scoring record for a single route candidate."""

    total_score: float = 0.0
    priority_score: float = 0.0
    health_score: float = 0.0
    reliability_score: float = 0.0
    latency_score: float = 0.0
    cost_score: float = 0.0
    security_score: float = 0.0
    affinity_score: float = 0.0
    policy_bonus: float = 0.0
    penalty_score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    disqualified: bool = False
    disqualification_reasons: List[str] = field(default_factory=list)

    def add_reason(self, reason: str) -> None:
        text = str(reason).strip()
        if text:
            self.reasons.append(text)

    def disqualify(self, reason: str) -> None:
        text = str(reason).strip()
        if text:
            self.disqualified = True
            self.disqualification_reasons.append(text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": round(float(self.total_score), 6),
            "priority_score": round(float(self.priority_score), 6),
            "health_score": round(float(self.health_score), 6),
            "reliability_score": round(float(self.reliability_score), 6),
            "latency_score": round(float(self.latency_score), 6),
            "cost_score": round(float(self.cost_score), 6),
            "security_score": round(float(self.security_score), 6),
            "affinity_score": round(float(self.affinity_score), 6),
            "policy_bonus": round(float(self.policy_bonus), 6),
            "penalty_score": round(float(self.penalty_score), 6),
            "reasons": list(self.reasons),
            "disqualified": self.disqualified,
            "disqualification_reasons": list(self.disqualification_reasons),
        }


@dataclass(slots=True)
class RouteEvaluation:
    """Serializable route-evaluation record."""

    candidate: Dict[str, Any]
    score: RoutePolicyScore
    allowed: bool
    evaluated_at: str
    constraints: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": json_safe(self.candidate),
            "score": self.score.to_dict(),
            "allowed": self.allowed,
            "evaluated_at": self.evaluated_at,
            "constraints": json_safe(self.constraints),
            "metadata": json_safe(self.metadata),
        }


class RoutePolicy:
    """
    Route-policy evaluator for runtime network routing.

    The evaluator takes candidate paths from Channel Selector or Endpoint
    Registry and decides which routes are eligible and preferred.
    """

    def __init__(
        self,
        memory: Optional[NetworkMemory] = None,
        endpoint_registry: Optional[EndpointRegistry] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = load_global_config()
        self.route_policy_config = merge_mappings(
            get_config_section("network_route_policy") or {},
            ensure_mapping(config, field_name="config", allow_none=True),
        )
        self.memory = memory or NetworkMemory()
        self.endpoint_registry = endpoint_registry or EndpointRegistry(memory=self.memory)
        self._lock = RLock()

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.record_policy_decisions = self._get_bool_config("record_policy_decisions", True)
        self.use_memory_health = self._get_bool_config("use_memory_health", True)
        self.use_memory_metrics = self._get_bool_config("use_memory_metrics", True)
        self.use_endpoint_registry = self._get_bool_config("use_endpoint_registry", True)
        self.allow_degraded_routes = self._get_bool_config("allow_degraded_routes", True)
        self.allow_unhealthy_routes = self._get_bool_config("allow_unhealthy_routes", False)
        self.disallow_open_circuits = self._get_bool_config("disallow_open_circuits", True)
        self.require_secure_for_tls_routes = self._get_bool_config("require_secure_for_tls_routes", True)
        self.disallow_private_hosts = self._get_bool_config("disallow_private_hosts", False)
        self.disallow_loopback_hosts = self._get_bool_config("disallow_loopback_hosts", False)
        self.fail_on_no_viable_routes = self._get_bool_config("fail_on_no_viable_routes", True)

        self.selection_strategy = self._get_strategy_config("selection_strategy", "weighted")
        self.default_primary_count = max(1, self._get_non_negative_int_config("default_primary_count", 1))
        self.default_secondary_count = self._get_non_negative_int_config("default_secondary_count", 1)
        self.snapshot_ttl_seconds = self._get_non_negative_int_config("snapshot_ttl_seconds", 1800)
        self.history_ttl_seconds = self._get_non_negative_int_config("history_ttl_seconds", 3600)
        self.max_history_size = max(1, self._get_non_negative_int_config("max_history_size", 1000))

        self.default_affinity_region = self._get_optional_string_config("default_affinity_region")
        self.default_max_cost = self._coerce_float(self.route_policy_config.get("default_max_cost"), 100.0, minimum=0.0)
        self.max_latency_ms = self._coerce_float(self.route_policy_config.get("max_latency_ms"), 2000.0, minimum=1.0)
        self.min_success_rate = self._coerce_float(self.route_policy_config.get("min_success_rate"), 0.80, minimum=0.0, maximum=1.0)
        self.max_error_rate = self._coerce_float(self.route_policy_config.get("max_error_rate"), 0.25, minimum=0.0, maximum=1.0)

        self.priority_weight = self._coerce_float(self.route_policy_config.get("priority_weight"), 1.0, minimum=0.0)
        self.health_weight = self._coerce_float(self.route_policy_config.get("health_weight"), 1.5, minimum=0.0)
        self.reliability_weight = self._coerce_float(self.route_policy_config.get("reliability_weight"), 1.25, minimum=0.0)
        self.latency_weight = self._coerce_float(self.route_policy_config.get("latency_weight"), 1.0, minimum=0.0)
        self.cost_weight = self._coerce_float(self.route_policy_config.get("cost_weight"), 0.75, minimum=0.0)
        self.security_weight = self._coerce_float(self.route_policy_config.get("security_weight"), 1.0, minimum=0.0)
        self.affinity_weight = self._coerce_float(self.route_policy_config.get("affinity_weight"), 0.75, minimum=0.0)
        self.degraded_penalty = self._coerce_float(self.route_policy_config.get("degraded_penalty"), 0.5, minimum=0.0)
        self.unhealthy_penalty = self._coerce_float(self.route_policy_config.get("unhealthy_penalty"), 2.0, minimum=0.0)
        self.insecure_penalty = self._coerce_float(self.route_policy_config.get("insecure_penalty"), 1.0, minimum=0.0)
        self.secure_bonus = self._coerce_float(self.route_policy_config.get("secure_bonus"), 0.5, minimum=0.0)
        self.open_circuit_penalty = self._coerce_float(self.route_policy_config.get("open_circuit_penalty"), 5.0, minimum=0.0)

        self._history: List[Dict[str, Any]] = []
        self._stats: Dict[str, int] = {
            "evaluations": 0,
            "candidate_evaluations": 0,
            "selections": 0,
            "rejections": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        constraints: Optional[Mapping[str, Any]] = None,
        request_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise ReliabilityError(
                "Route policy is disabled by configuration.",
                context={"operation": "evaluate_candidate"},
            )

        normalized_candidate = self._normalize_candidate(candidate)
        normalized_constraints = ensure_mapping(constraints, field_name="constraints", allow_none=True)
        request_context_map = ensure_mapping(request_context, field_name="request_context", allow_none=True)

        score = RoutePolicyScore()
        endpoint = normalized_candidate.get("endpoint")
        protocol = normalize_protocol_name(normalized_candidate.get("protocol") or "http")
        channel = normalize_channel_name(normalized_candidate.get("channel") or protocol)
        secure = bool(normalized_candidate.get("secure", is_secure_protocol(protocol)))
        status = str(normalized_candidate.get("status", "unknown")).lower()
        region = normalized_candidate.get("region")
        priority = int(normalized_candidate.get("priority", 0) or 0)
        cost = float(normalized_candidate.get("cost", normalized_candidate.get("weight", 1.0)) or 1.0)
        latency_ms = self._coerce_float(normalized_candidate.get("latency_ms"), normalized_candidate.get("metadata", {}).get("latency_ms") or 0.0, minimum=0.0)
        success_rate = self._maybe_float(normalized_candidate.get("success_rate"))
        error_rate = self._maybe_float(normalized_candidate.get("error_rate"))
        circuit_state = str(normalized_candidate.get("circuit_state", "") or "").lower() or None

        if self.use_endpoint_registry and endpoint:
            registry_view = self.endpoint_registry.get_endpoint(endpoint)
            if registry_view is not None:
                normalized_candidate = merge_mappings(registry_view, normalized_candidate)
                score.add_reason("enriched from endpoint registry")
                status = str(normalized_candidate.get("status", status)).lower()
                region = normalized_candidate.get("region") or region
                secure = bool(normalized_candidate.get("secure", secure))
                circuit_state = str(normalized_candidate.get("circuit_state", circuit_state or "") or "").lower() or None
                latency_ms = self._coerce_float(normalized_candidate.get("latency_ms"), latency_ms, minimum=0.0)
                success_rate = self._maybe_float(normalized_candidate.get("success_rate")) if normalized_candidate.get("success_rate") is not None else success_rate
                error_rate = self._maybe_float(normalized_candidate.get("error_rate")) if normalized_candidate.get("error_rate") is not None else error_rate

        memory_endpoint_health = self._get_memory_endpoint_health(endpoint)
        if memory_endpoint_health:
            normalized_candidate = merge_mappings(memory_endpoint_health, normalized_candidate)
            score.add_reason("enriched from memory endpoint health")
            status = str(normalized_candidate.get("status", status)).lower()
            circuit_state = str(normalized_candidate.get("circuit_state", circuit_state or "") or "").lower() or None
            if normalized_candidate.get("latency_ms") is not None:
                latency_ms = self._coerce_float(normalized_candidate.get("latency_ms"), latency_ms, minimum=0.0)
            if normalized_candidate.get("success_rate") is not None:
                success_rate = self._maybe_float(normalized_candidate.get("success_rate"))
            if normalized_candidate.get("error_rate") is not None:
                error_rate = self._maybe_float(normalized_candidate.get("error_rate"))

        channel_metrics = self._get_memory_channel_metrics(channel)
        if channel_metrics:
            score.add_reason("enriched from memory channel metrics")
            if success_rate is None and channel_metrics.get("success_rate") is not None:
                success_rate = self._maybe_float(channel_metrics.get("success_rate"))
            if error_rate is None and channel_metrics.get("error_rate") is not None:
                error_rate = self._maybe_float(channel_metrics.get("error_rate"))
            if not latency_ms and channel_metrics.get("p95_latency_ms") is not None:
                latency_ms = self._coerce_float(channel_metrics.get("p95_latency_ms"), latency_ms, minimum=0.0)

        required_protocol = normalized_constraints.get("required_protocol")
        required_channel = normalized_constraints.get("required_channel")
        preferred_region = normalized_constraints.get("preferred_region") or request_context_map.get("preferred_region") or self.default_affinity_region
        max_cost = self._coerce_float(normalized_constraints.get("max_cost"), self.default_max_cost, minimum=0.0)
        require_secure = bool(normalized_constraints.get("require_secure", False) or request_context_map.get("tls_required", False))
        require_streaming = bool(normalized_constraints.get("require_streaming", False))
        require_bidirectional_streaming = bool(normalized_constraints.get("require_bidirectional_streaming", False))

        capabilities = ensure_mapping(normalized_candidate.get("capabilities"), field_name="capabilities", allow_none=True)

        if required_protocol and protocol != normalize_protocol_name(required_protocol):
            score.disqualify("protocol requirement not satisfied")
        if required_channel and channel != normalize_channel_name(required_channel):
            score.disqualify("channel requirement not satisfied")
        if require_streaming and not bool(capabilities.get("supports_streaming", False)):
            score.disqualify("streaming capability required")
        if require_bidirectional_streaming and not bool(capabilities.get("supports_bidirectional_streaming", False)):
            score.disqualify("bidirectional streaming capability required")
        if cost > max_cost:
            score.disqualify("candidate cost exceeds allowed maximum")
        if self.disallow_open_circuits and circuit_state == "open":
            score.disqualify("endpoint circuit is open")
        if not self.allow_degraded_routes and status in {"degraded", "warning", "limited", "slow"}:
            score.disqualify("degraded routes are disabled")
        if not self.allow_unhealthy_routes and status in {"down", "failed", "unhealthy", "blocked", "closed"}:
            score.disqualify("unhealthy routes are disabled")
        if require_secure and self.require_secure_for_tls_routes and not secure:
            score.disqualify("secure route is required")
        if self.disallow_private_hosts and endpoint and self._endpoint_is_private(endpoint):
            score.disqualify("private-host route is disallowed")
        if self.disallow_loopback_hosts and endpoint and self._endpoint_is_loopback(endpoint):
            score.disqualify("loopback route is disallowed")
        if success_rate is not None and success_rate < self.min_success_rate:
            score.disqualify("success rate below policy threshold")
        if error_rate is not None and error_rate > self.max_error_rate:
            score.disqualify("error rate exceeds policy threshold")

        score.priority_score = round((max(0.0, priority) / 100.0) * self.priority_weight, 6)
        score.health_score = round(float(normalized_candidate.get("health_score", self._status_health_score(status))) * self.health_weight, 6)
        score.reliability_score = round(((success_rate if success_rate is not None else 0.75) - (error_rate if error_rate is not None else 0.0)) * self.reliability_weight, 6)
        latency_ratio = 1.0 - min(1.0, (latency_ms / self.max_latency_ms)) if self.max_latency_ms > 0 else 0.0
        score.latency_score = round(latency_ratio * self.latency_weight, 6)
        score.cost_score = round(max(0.0, 1.0 - min(1.0, cost / max_cost)) * self.cost_weight if max_cost > 0 else 0.0, 6)
        score.security_score = round((self.secure_bonus if secure else -self.insecure_penalty) * self.security_weight, 6)
        score.affinity_score = round((1.0 if preferred_region and region == preferred_region else 0.0) * self.affinity_weight, 6)

        if status in {"degraded", "warning", "limited", "slow"}:
            score.penalty_score += self.degraded_penalty
            score.add_reason("degraded-route penalty applied")
        if status in {"down", "failed", "unhealthy", "blocked", "closed"}:
            score.penalty_score += self.unhealthy_penalty
            score.add_reason("unhealthy-route penalty applied")
        if circuit_state == "open":
            score.penalty_score += self.open_circuit_penalty
            score.add_reason("open-circuit penalty applied")
        if secure:
            score.add_reason("secure route bonus applied")
        if preferred_region and region == preferred_region:
            score.add_reason("region affinity matched")

        score.total_score = round(
            score.priority_score
            + score.health_score
            + score.reliability_score
            + score.latency_score
            + score.cost_score
            + score.security_score
            + score.affinity_score
            + score.policy_bonus
            - score.penalty_score,
            6,
        )

        allowed = not score.disqualified
        evaluation = RouteEvaluation(
            candidate=normalized_candidate,
            score=score,
            allowed=allowed,
            evaluated_at=utc_timestamp(),
            constraints=normalize_metadata(normalized_constraints),
            metadata=normalize_metadata(request_context_map),
        )
        self._record_evaluation(evaluation)
        return evaluation.to_dict()

    def evaluate_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        constraints: Optional[Mapping[str, Any]] = None,
        request_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        evaluations = [
            self.evaluate_candidate(candidate, constraints=constraints, request_context=request_context)
            for candidate in ensure_sequence(candidates, field_name="candidates", allow_none=False, coerce_scalar=False)
        ]
        evaluations.sort(key=self._sort_evaluation, reverse=True)
        return evaluations

    def select_routes(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        constraints: Optional[Mapping[str, Any]] = None,
        request_context: Optional[Mapping[str, Any]] = None,
        primary_count: Optional[int] = None,
        secondary_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        normalized_primary_count = max(1, int(primary_count or self.default_primary_count))
        normalized_secondary_count = max(0, int(self.default_secondary_count if secondary_count is None else secondary_count))
        evaluations = self.evaluate_candidates(candidates, constraints=constraints, request_context=request_context)
        allowed = [item for item in evaluations if bool(item.get("allowed"))]

        if not allowed and self.fail_on_no_viable_routes:
            raise NoRouteAvailableError(
                "No viable routes satisfy current route policy.",
                context={"operation": "select_routes"},
                details={
                    "constraints": sanitize_for_logging(constraints) if self.sanitize_logs else json_safe(constraints),
                    "request_context": sanitize_for_logging(request_context) if self.sanitize_logs else json_safe(request_context),
                },
            )

        primaries = allowed[:normalized_primary_count]
        secondaries = allowed[normalized_primary_count:normalized_primary_count + normalized_secondary_count]

        selected_route = primaries[0]["candidate"] if primaries else None
        route_snapshot = {
            "selected": selected_route,
            "primary": [item["candidate"] for item in primaries],
            "secondary": [item["candidate"] for item in secondaries],
            "evaluations": evaluations,
            "strategy": self.selection_strategy,
            "selected_at": utc_timestamp(),
            "constraints": sanitize_for_logging(constraints) if self.sanitize_logs else json_safe(constraints),
            "request_context": sanitize_for_logging(request_context) if self.sanitize_logs else json_safe(request_context),
        }

        with self._lock:
            self._stats["selections"] += 1
            if selected_route is None:
                self._stats["rejections"] += 1

        if self.record_policy_decisions:
            self.memory.set_route_selection(
                selected_route=selected_route or {},
                candidate_routes=[item["candidate"] for item in allowed],
                route_id=str(selected_route.get("endpoint_id") or selected_route.get("endpoint") or "none") if selected_route else "none",
                reason="route_policy_selection" if selected_route else "no_viable_route",
                metadata={
                    "policy": "route_policy",
                    "primary_count": normalized_primary_count,
                    "secondary_count": normalized_secondary_count,
                },
            )
            self.memory.record_policy_decision(
                "route_policy",
                {
                    "allowed": bool(selected_route),
                    "primary": [item["candidate"] for item in primaries],
                    "secondary": [item["candidate"] for item in secondaries],
                },
                endpoint=selected_route.get("endpoint") if selected_route else None,
                protocol=selected_route.get("protocol") if selected_route else None,
                channel=selected_route.get("channel") if selected_route else None,
                reason="selected primary and secondary routes" if selected_route else "no viable routes",
                metadata={"evaluations": len(evaluations)},
            )
            self.memory.set(
                _ROUTE_POLICY_LAST_KEY,
                route_snapshot,
                ttl_seconds=self.snapshot_ttl_seconds,
                source="route_policy",
            )
            self.memory.append(
                _ROUTE_POLICY_HISTORY_KEY,
                route_snapshot,
                max_items=self.max_history_size,
                ttl_seconds=self.history_ttl_seconds,
                source="route_policy",
            )
            self.memory.set(
                _ROUTE_POLICY_SNAPSHOT_KEY,
                self.get_policy_snapshot(),
                ttl_seconds=self.snapshot_ttl_seconds,
                source="route_policy",
            )

        return route_snapshot

    def get_policy_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "generated_at": utc_timestamp(),
                "enabled": self.enabled,
                "selection_strategy": self.selection_strategy,
                "stats": dict(self._stats),
                "history_size": len(self._history),
                "config": {
                    "allow_degraded_routes": self.allow_degraded_routes,
                    "allow_unhealthy_routes": self.allow_unhealthy_routes,
                    "disallow_open_circuits": self.disallow_open_circuits,
                    "require_secure_for_tls_routes": self.require_secure_for_tls_routes,
                    "default_primary_count": self.default_primary_count,
                    "default_secondary_count": self.default_secondary_count,
                    "default_affinity_region": self.default_affinity_region,
                    "default_max_cost": self.default_max_cost,
                },
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_candidate(self, candidate: Mapping[str, Any]) -> Dict[str, Any]:
        payload = ensure_mapping(candidate, field_name="candidate")
        protocol = normalize_protocol_name(payload.get("protocol") or "http")
        channel = normalize_channel_name(payload.get("channel") or protocol)
        endpoint = payload.get("endpoint")
        secure = payload.get("secure")
        if endpoint and isinstance(endpoint, str) and "://" in endpoint:
            try:
                endpoint = normalize_endpoint(endpoint, protocol=protocol)
                if secure is None:
                    secure = bool(parse_endpoint(endpoint, default_scheme=protocol, protocol=protocol, require_host=False).secure)
            except Exception:
                endpoint = str(endpoint)
        return merge_mappings(
            payload,
            {
                "protocol": protocol,
                "channel": channel,
                "endpoint": endpoint,
                "secure": bool(secure) if secure is not None else is_secure_protocol(protocol),
                "priority": int(payload.get("priority", 0) or 0),
                "cost": float(payload.get("cost", payload.get("weight", 1.0)) or 1.0),
                "metadata": normalize_metadata(payload.get("metadata")),
                "capabilities": ensure_mapping(payload.get("capabilities"), field_name="capabilities", allow_none=True),
            },
        )

    def _get_memory_endpoint_health(self, endpoint: Optional[str]) -> Dict[str, Any]:
        if not self.use_memory_health or not endpoint:
            return {}
        health_map = self.memory.get("network.endpoint.health", default={})
        if isinstance(health_map, Mapping):
            payload = health_map.get(endpoint)
            if isinstance(payload, Mapping):
                return dict(payload)
        return {}

    def _get_memory_channel_metrics(self, channel: str) -> Dict[str, Any]:
        if not self.use_memory_metrics:
            return {}
        metrics_map = self.memory.get("network.telemetry.channel_metrics", default={})
        if isinstance(metrics_map, Mapping):
            payload = metrics_map.get(channel)
            if isinstance(payload, Mapping):
                return dict(payload)
        return {}

    def _status_health_score(self, status: str) -> float:
        normalized = str(status).strip().lower()
        if normalized in {"healthy", "available", "up", "connected", "idle"}:
            return 1.0
        if normalized in {"degraded", "warning", "limited", "slow"}:
            return 0.6
        if normalized in {"down", "failed", "unhealthy", "blocked", "closed"}:
            return 0.1
        return 0.5

    def _endpoint_is_private(self, endpoint: str) -> bool:
        if "://" not in endpoint:
            return False
        try:
            host = parse_endpoint(endpoint, require_host=False).host
        except Exception:
            return False
        return is_private_host(host)

    def _endpoint_is_loopback(self, endpoint: str) -> bool:
        if "://" not in endpoint:
            return False
        try:
            host = parse_endpoint(endpoint, require_host=False).host
        except Exception:
            return False
        return host in {"127.0.0.1", "::1", "localhost"}

    def _record_evaluation(self, evaluation: RouteEvaluation) -> None:
        with self._lock:
            self._stats["evaluations"] += 1
            self._stats["candidate_evaluations"] += 1
            serialized = evaluation.to_dict()
            self._history.append(serialized)
            if len(self._history) > self.max_history_size:
                self._history = self._history[-self.max_history_size:]

    def _sort_evaluation(self, payload: Mapping[str, Any]) -> Tuple[int, float, int]:
        score_map = ensure_mapping(payload.get("score"), field_name="score", allow_none=True)
        return (
            1 if bool(payload.get("allowed")) else 0,
            float(score_map.get("total_score", 0.0) or 0.0),
            int(ensure_mapping(payload.get("candidate"), field_name="candidate", allow_none=True).get("priority", 0) or 0),
        )

    def _get_bool_config(self, name: str, default: bool) -> bool:
        return self._coerce_bool(self.route_policy_config.get(name, default), default)

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        return self._coerce_int(self.route_policy_config.get(name, default), default, non_negative=True)

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.route_policy_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _get_strategy_config(self, name: str, default: str) -> str:
        value = str(self.route_policy_config.get(name, default)).strip().lower() or default
        if value not in _VALID_SELECTION_STRATEGIES:
            raise NetworkConfigurationError(
                "Invalid selection strategy in route policy configuration.",
                context={"operation": "route_policy_config"},
                details={"config_key": name, "config_value": value},
            )
        return value

    def _coerce_bool(self, value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
        raise NetworkConfigurationError(
            "Invalid boolean value in route policy configuration.",
            context={"operation": "route_policy_config"},
            details={"config_value": value},
        )

    def _coerce_int(self, value: Any, default: int, *, non_negative: bool = False) -> int:
        if value is None:
            value = default
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in route policy configuration.",
                context={"operation": "route_policy_config"},
                details={"config_value": value},
                cause=exc,
            ) from exc
        if non_negative and coerced < 0:
            raise NetworkConfigurationError(
                "Configuration integer value must be non-negative.",
                context={"operation": "route_policy_config"},
                details={"config_value": value},
            )
        return coerced

    def _coerce_float(self, value: Any, default: float, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
        if value is None:
            value = default
        try:
            coerced = float(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid float value in route policy configuration.",
                context={"operation": "route_policy_config"},
                details={"config_value": value},
                cause=exc,
            ) from exc
        if minimum is not None:
            coerced = max(float(minimum), coerced)
        if maximum is not None:
            coerced = min(float(maximum), coerced)
        return round(coerced, 6)

    def _maybe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        return self._coerce_float(value, 0.0)


class _printer_proxy:
    @staticmethod
    def status(label: str, message: str, level: str = "info") -> None:
        try:
            printer.status(label, message, level)
        except Exception:
            print(f"[{label}] {message}")


if __name__ == "__main__":
    print("\n=== Running Route Policy ===\n")
    _printer_proxy.status("TEST", "Route Policy initialized", "info")

    memory = NetworkMemory()
    registry = EndpointRegistry(memory=memory)
    policy = RoutePolicy(memory=memory, endpoint_registry=registry)

    registry.register_endpoint(
        "https://api.example.com/v1/jobs",
        endpoint_id="api_primary",
        protocol="https",
        channel="http",
        region="eu-west",
        priority=120,
        secure=True,
        capabilities={"supports_request_reply": True, "supports_headers": True},
        tags=["primary"],
    )
    registry.register_endpoint(
        "grpc://grpc.example.com:50051",
        endpoint_id="grpc_primary",
        protocol="grpc",
        channel="grpc",
        region="eu-west",
        priority=115,
        secure=False,
        capabilities={"supports_request_reply": True, "supports_streaming": True},
        tags=["rpc"],
    )
    registry.register_endpoint(
        "jobs.primary",
        endpoint_id="queue_fallback",
        protocol="queue",
        channel="queue",
        endpoint_type="logical",
        region="eu-west",
        priority=90,
        secure=False,
        capabilities={"supports_ack": True, "supports_nack": True, "supports_batch_send": True},
        tags=["fallback"],
    )
    _printer_proxy.status("TEST", "Endpoints registered for route policy", "info")

    registry.update_health("api_primary", status="healthy", latency_ms=44.0, success_rate=0.993, error_rate=0.007)
    registry.update_health("grpc_primary", status="degraded", latency_ms=82.0, success_rate=0.94, error_rate=0.06, circuit_state="half_open")
    registry.update_health("queue_fallback", status="healthy", latency_ms=8.0, success_rate=0.985, error_rate=0.015)
    memory.record_channel_metrics("http", {"success_rate": 0.992, "error_rate": 0.008, "p95_latency_ms": 47})
    memory.record_channel_metrics("grpc", {"success_rate": 0.945, "error_rate": 0.055, "p95_latency_ms": 88})
    memory.record_channel_metrics("queue", {"success_rate": 0.987, "error_rate": 0.013, "p95_latency_ms": 12})
    _printer_proxy.status("TEST", "Health and metrics seeded", "info")

    candidates = registry.get_candidates(include_unavailable=False, include_degraded=True, include_unhealthy=False)
    evaluations = policy.evaluate_candidates(
        candidates,
        constraints={
            "preferred_region": "eu-west",
            "max_cost": 2.0,
            "require_secure": True,
        },
        request_context={"tls_required": True},
    )
    _printer_proxy.status("TEST", "Candidates evaluated", "info")

    selection = policy.select_routes(
        candidates,
        constraints={"preferred_region": "eu-west", "require_secure": True},
        request_context={"tls_required": True},
        primary_count=1,
        secondary_count=1,
    )
    _printer_proxy.status("TEST", "Routes selected", "info")

    snapshot = policy.get_policy_snapshot()

    print("Candidates:", stable_json_dumps(candidates))
    print("Evaluations:", stable_json_dumps(evaluations))
    print("Selection:", stable_json_dumps(selection))
    print("Policy Snapshot:", stable_json_dumps(snapshot))

    assert evaluations[0]["allowed"] is True
    assert selection["selected"]["endpoint_id"] == "api_primary"
    assert len(selection["primary"]) == 1
    assert isinstance(selection["secondary"], list)
    assert memory.get("network.route.selected") is not None
    assert memory.get(_ROUTE_POLICY_LAST_KEY) is not None
    assert memory.get("network.policy.decision") is not None

    _printer_proxy.status("TEST", "All Route Policy checks passed", "info")
    print("\n=== Test ran successfully ===\n")
