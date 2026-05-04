"""
Channel selection logic for SLAI's Network Agent routing subsystem.

This module provides the production-grade channel-selection layer used by the
network routing stack. It is responsible for evaluating transport candidates
across the specialized adapter families (HTTP, WebSocket, gRPC, queue) and
choosing the most suitable channel for a given operation based on:

- requested protocol/channel intent,
- adapter capability requirements,
- endpoint health and circuit posture,
- channel metrics such as latency and reliability,
- TLS/security expectations,
- policy-aware constraints supplied by callers,
- configured priority/weighting rules.

The selector is intentionally scoped to selection and ranking. It does not own
endpoint discovery/registry management, route-policy arbitration, retry logic,
or transport execution. Those concerns belong to the adjacent routing,
reliability, and specialized adapter layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils import *
from ..network_memory import NetworkMemory
from ..network_adapters import NetworkAdapters
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Channel Selector")
printer = PrettyPrinter()

__all__ = [
    "ChannelScoreBreakdown",
    "ChannelCandidate",
    "ChannelSelectionDecision",
    "ChannelSelector",
]


_CHANNEL_SELECTOR_LAST_KEY = "network.routing.channel_selector.last"
_CHANNEL_SELECTOR_HISTORY_KEY = "network.routing.channel_selector.history"
_CHANNEL_SELECTOR_SNAPSHOT_KEY = "network.routing.channel_selector.snapshot"

_VALID_SELECTION_STRATEGIES = {"weighted", "priority", "first_viable"}
_DEFAULT_HEALTHY_STATUSES = ("healthy", "available", "up", "connected", "idle")
_DEFAULT_DEGRADED_STATUSES = ("degraded", "warning", "limited", "slow")
_DEFAULT_UNHEALTHY_STATUSES = ("down", "failed", "unhealthy", "blocked", "closed")
_CAPABILITY_KEYS = (
    "supports_streaming",
    "supports_bidirectional_streaming",
    "supports_ack",
    "supports_nack",
    "supports_batch_send",
    "supports_headers",
    "supports_tls",
    "supports_reconnect",
    "supports_receive",
    "supports_request_reply",
)


@dataclass(slots=True)
class ChannelScoreBreakdown:
    """Detailed per-candidate scoring record used for selection transparency."""

    total_score: float = 0.0
    adapter_priority_score: float = 0.0
    config_priority_score: float = 0.0
    protocol_match_score: float = 0.0
    channel_match_score: float = 0.0
    preferred_score: float = 0.0
    capability_score: float = 0.0
    health_score: float = 0.0
    reliability_score: float = 0.0
    latency_score: float = 0.0
    security_score: float = 0.0
    cost_score: float = 0.0
    affinity_score: float = 0.0
    penalty_score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    disqualified: bool = False
    disqualification_reasons: List[str] = field(default_factory=list)

    def add_reason(self, reason: str) -> None:
        text = str(reason).strip()
        if text:
            self.reasons.append(text)

    def add_disqualification(self, reason: str) -> None:
        text = str(reason).strip()
        if text:
            self.disqualified = True
            self.disqualification_reasons.append(text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": round(float(self.total_score), 6),
            "adapter_priority_score": round(float(self.adapter_priority_score), 6),
            "config_priority_score": round(float(self.config_priority_score), 6),
            "protocol_match_score": round(float(self.protocol_match_score), 6),
            "channel_match_score": round(float(self.channel_match_score), 6),
            "preferred_score": round(float(self.preferred_score), 6),
            "capability_score": round(float(self.capability_score), 6),
            "health_score": round(float(self.health_score), 6),
            "reliability_score": round(float(self.reliability_score), 6),
            "latency_score": round(float(self.latency_score), 6),
            "security_score": round(float(self.security_score), 6),
            "cost_score": round(float(self.cost_score), 6),
            "affinity_score": round(float(self.affinity_score), 6),
            "penalty_score": round(float(self.penalty_score), 6),
            "reasons": list(self.reasons),
            "disqualified": self.disqualified,
            "disqualification_reasons": list(self.disqualification_reasons),
        }


@dataclass(slots=True)
class ChannelCandidate:
    """Normalized candidate transport path considered by the selector."""

    adapter_name: str
    protocol: str
    channel: str
    endpoint: Optional[str] = None
    route: Optional[str] = None
    region: Optional[str] = None
    priority: int = 100
    cost: float = 1.0
    weight: float = 1.0
    secure: Optional[bool] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def identity_key(self) -> str:
        endpoint_token = self.endpoint or "none"
        route_token = self.route or "default"
        return f"{self.adapter_name}:{self.protocol}:{self.channel}:{endpoint_token}:{route_token}"

    def inferred_secure(self) -> bool:
        if self.secure is not None:
            return bool(self.secure)
        if self.endpoint:
            try:
                return bool(parse_endpoint(self.endpoint, default_scheme=self.protocol, protocol=self.protocol, require_host=False).secure)
            except Exception:
                return is_secure_protocol(self.protocol)
        return is_secure_protocol(self.protocol)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "protocol": self.protocol,
            "channel": self.channel,
            "endpoint": self.endpoint,
            "route": self.route,
            "region": self.region,
            "priority": int(self.priority),
            "cost": float(self.cost),
            "weight": float(self.weight),
            "secure": self.inferred_secure(),
            "capabilities": json_safe(self.capabilities),
            "metadata": json_safe(self.metadata),
            "identity_key": self.identity_key,
        }


@dataclass(slots=True)
class ChannelSelectionDecision:
    """Serializable result returned by a channel-selection run."""

    selected: ChannelCandidate
    score: ChannelScoreBreakdown
    candidates: List[Dict[str, Any]]
    strategy: str
    selected_at: str
    constraints: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": self.selected.to_dict(),
            "score": self.score.to_dict(),
            "candidates": json_safe(self.candidates),
            "strategy": self.strategy,
            "selected_at": self.selected_at,
            "constraints": json_safe(self.constraints),
            "metadata": json_safe(self.metadata),
        }


class ChannelSelector:
    """
    Runtime channel-selection logic for routing and stream orchestration.

    The selector consumes explicit candidate definitions or derives candidates
    from the registered adapter surface, then ranks them using config-backed
    weighting and in-memory operational signals.
    """

    def __init__(
        self,
        memory: Optional[NetworkMemory] = None,
        adapters: Optional[NetworkAdapters] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = load_global_config()
        self.selector_config = merge_mappings(
            get_config_section("network_channel_selector") or {},
            ensure_mapping(config, field_name="config", allow_none=True),
        )
        self.memory = memory or NetworkMemory()
        self.adapters = adapters or NetworkAdapters(memory=self.memory)
        self._lock = RLock()

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.record_selection_snapshots = self._get_bool_config("record_selection_snapshots", True)
        self.use_memory_health = self._get_bool_config("use_memory_health", True)
        self.use_memory_metrics = self._get_bool_config("use_memory_metrics", True)
        self.use_policy_memory = self._get_bool_config("use_policy_memory", True)
        self.fail_on_no_candidates = self._get_bool_config("fail_on_no_candidates", True)
        self.allow_degraded_channels = self._get_bool_config("allow_degraded_channels", True)
        self.allow_unhealthy_channels = self._get_bool_config("allow_unhealthy_channels", False)
        self.disallow_open_circuits = self._get_bool_config("disallow_open_circuits", True)
        self.prefer_secure_for_tls_endpoints = self._get_bool_config("prefer_secure_for_tls_endpoints", True)
        self.prefer_registered_adapters = self._get_bool_config("prefer_registered_adapters", True)

        self.default_protocol = normalize_protocol_name(self._get_optional_string_config("default_protocol") or "http")
        self.default_channel = normalize_channel_name(self._get_optional_string_config("default_channel") or self.default_protocol)
        self.selection_strategy = self._get_strategy_config("selection_strategy", "weighted")

        self.selection_ttl_seconds = self._get_non_negative_int_config("selection_ttl_seconds", 900)
        self.history_ttl_seconds = self._get_non_negative_int_config("history_ttl_seconds", 3600)
        self.max_history_size = max(1, self._get_non_negative_int_config("max_history_size", 250))
        self.max_candidates_considered = max(1, self._get_non_negative_int_config("max_candidates_considered", 128))
        self.default_cost = self._get_non_negative_float_config("default_cost", 1.0)
        self.default_weight = self._get_positive_float_config("default_weight", 1.0)
        self.reference_latency_ms = self._get_positive_float_config("reference_latency_ms", 500.0)

        self.healthy_statuses = self._get_status_set("healthy_statuses", _DEFAULT_HEALTHY_STATUSES)
        self.degraded_statuses = self._get_status_set("degraded_statuses", _DEFAULT_DEGRADED_STATUSES)
        self.unhealthy_statuses = self._get_status_set("unhealthy_statuses", _DEFAULT_UNHEALTHY_STATUSES)

        weights = ensure_mapping(self.selector_config.get("weights"), field_name="weights", allow_none=True)
        penalties = ensure_mapping(self.selector_config.get("penalties"), field_name="penalties", allow_none=True)

        self.weight_adapter_priority = self._coerce_float(weights.get("adapter_priority"), 12.0, non_negative=True)
        self.weight_config_priority = self._coerce_float(weights.get("config_priority"), 8.0, non_negative=True)
        self.weight_protocol_match = self._coerce_float(weights.get("protocol_match"), 20.0, non_negative=True)
        self.weight_channel_match = self._coerce_float(weights.get("channel_match"), 18.0, non_negative=True)
        self.weight_preferred = self._coerce_float(weights.get("preferred"), 8.0, non_negative=True)
        self.weight_capability = self._coerce_float(weights.get("capability_match"), 15.0, non_negative=True)
        self.weight_health = self._coerce_float(weights.get("health"), 20.0, non_negative=True)
        self.weight_reliability = self._coerce_float(weights.get("reliability"), 18.0, non_negative=True)
        self.weight_latency = self._coerce_float(weights.get("latency"), 12.0, non_negative=True)
        self.weight_security = self._coerce_float(weights.get("security"), 10.0, non_negative=True)
        self.weight_cost = self._coerce_float(weights.get("cost"), 6.0, non_negative=True)
        self.weight_affinity = self._coerce_float(weights.get("affinity"), 8.0, non_negative=True)

        self.penalty_degraded = self._coerce_float(penalties.get("degraded_status"), 15.0, non_negative=True)
        self.penalty_unhealthy = self._coerce_float(penalties.get("unhealthy_status"), 100.0, non_negative=True)
        self.penalty_open_circuit = self._coerce_float(penalties.get("open_circuit"), 100.0, non_negative=True)
        self.penalty_missing_metrics = self._coerce_float(penalties.get("missing_metrics"), 2.0, non_negative=True)
        self.penalty_missing_health = self._coerce_float(penalties.get("missing_health"), 2.0, non_negative=True)
        self.penalty_insecure_for_tls = self._coerce_float(penalties.get("insecure_for_tls"), 100.0, non_negative=True)
        self.penalty_high_cost = self._coerce_float(penalties.get("high_cost"), 25.0, non_negative=True)

        self._stats: Dict[str, int] = {
            "builds": 0,
            "selections": 0,
            "rankings": 0,
            "disqualifications": 0,
            "failures": 0,
        }
        self._selection_counts: Dict[str, int] = {}
        self._started_at = _utcnow()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_candidates(
        self,
        *,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        endpoint: Optional[str] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        include_registered: bool = True,
        constraints: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        built = self._build_candidates_internal(
            protocol=protocol,
            channel=channel,
            endpoint=endpoint,
            candidates=candidates,
            include_registered=include_registered,
            constraints=constraints,
        )
        return [candidate.to_dict() for candidate in built]

    def rank_candidates(
        self,
        *,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        endpoint: Optional[str] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        include_registered: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            normalized_constraints = self._normalize_constraints(constraints)
            normalized_metadata = normalize_metadata(metadata)
            built_candidates = self._build_candidates_internal(
                protocol=protocol,
                channel=channel,
                endpoint=endpoint,
                candidates=candidates,
                include_registered=include_registered,
                constraints=normalized_constraints,
            )
            ranked = self._rank_candidates_internal(
                built_candidates,
                protocol=protocol,
                channel=channel,
                endpoint=endpoint,
                constraints=normalized_constraints,
                metadata=normalized_metadata,
            )
            self._stats["rankings"] += 1
            return ranked

    def select_channel(
        self,
        *,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        endpoint: Optional[str] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        include_registered: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise RoutingError(
                "ChannelSelector is disabled by configuration.",
                context={"operation": "select_channel", "channel": channel, "protocol": protocol, "endpoint": endpoint},
            )

        with self._lock:
            normalized_constraints = self._normalize_constraints(constraints)
            normalized_metadata = normalize_metadata(metadata)
            built_candidates = self._build_candidates_internal(
                protocol=protocol,
                channel=channel,
                endpoint=endpoint,
                candidates=candidates,
                include_registered=include_registered,
                constraints=normalized_constraints,
            )
            ranked = self._rank_candidates_internal(
                built_candidates,
                protocol=protocol,
                channel=channel,
                endpoint=endpoint,
                constraints=normalized_constraints,
                metadata=normalized_metadata,
            )

            viable = [item for item in ranked if not item["score"].get("disqualified", False)]
            if not viable:
                self._stats["failures"] += 1
                if self.fail_on_no_candidates:
                    raise NoRouteAvailableError(
                        "No viable channel candidate was available for the requested routing constraints.",
                        context={"operation": "select_channel", "channel": channel, "protocol": protocol, "endpoint": endpoint},
                        details={
                            "constraints": sanitize_for_logging(normalized_constraints) if self.sanitize_logs else json_safe(normalized_constraints),
                            "candidates": sanitize_for_logging(ranked) if self.sanitize_logs else json_safe(ranked),
                        },
                    )
                return {
                    "selected": None,
                    "candidates": ranked,
                    "constraints": json_safe(normalized_constraints),
                    "selected_at": utc_timestamp(),
                    "strategy": self.selection_strategy,
                    "metadata": json_safe(normalized_metadata),
                }

            selected_payload = viable[0]
            selected_candidate = self._mapping_to_candidate(selected_payload["candidate"])
            selected_score = self._mapping_to_score(selected_payload["score"])
            decision = ChannelSelectionDecision(
                selected=selected_candidate,
                score=selected_score,
                candidates=ranked,
                strategy=self.selection_strategy,
                selected_at=utc_timestamp(),
                constraints=normalized_constraints,
                metadata=normalized_metadata,
            )

            self._stats["selections"] += 1
            self._selection_counts[selected_candidate.adapter_name] = self._selection_counts.get(selected_candidate.adapter_name, 0) + 1
            self._record_selection(decision)
            return decision.to_dict()

    def select_and_create_adapter(
        self,
        *,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        endpoint: Optional[str] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        include_registered: bool = True,
        config: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **adapter_kwargs: Any,
    ) -> Tuple[Any, Dict[str, Any]]:
        decision = self.select_channel(
            protocol=protocol,
            channel=channel,
            endpoint=endpoint,
            candidates=candidates,
            constraints=constraints,
            include_registered=include_registered,
            metadata=metadata,
        )
        selected = ensure_mapping(decision.get("selected"), field_name="selected")
        adapter = self.adapters.create_adapter(
            name=selected.get("adapter_name"),
            protocol=selected.get("protocol"),
            channel=selected.get("channel"),
            endpoint=selected.get("endpoint") or endpoint,
            constraints=constraints,
            config=config,
            **adapter_kwargs,
        )
        return adapter, decision

    def get_selector_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "generated_at": utc_timestamp(),
                "enabled": self.enabled,
                "selection_strategy": self.selection_strategy,
                "default_protocol": self.default_protocol,
                "default_channel": self.default_channel,
                "stats": dict(self._stats),
                "selection_counts": dict(self._selection_counts),
                "started_at": self._started_at.isoformat(),
                "last_selection": self.memory.get(_CHANNEL_SELECTOR_LAST_KEY, default=None),
            }

    # ------------------------------------------------------------------
    # Internal candidate handling
    # ------------------------------------------------------------------
    def _build_candidates_internal(
        self,
        *,
        protocol: Optional[str],
        channel: Optional[str],
        endpoint: Optional[str],
        candidates: Optional[Sequence[Mapping[str, Any]]],
        include_registered: bool,
        constraints: Optional[Mapping[str, Any]],
    ) -> List[ChannelCandidate]:
        normalized_protocol = normalize_protocol_name(protocol) if protocol is not None else None
        normalized_channel = normalize_channel_name(channel) if channel is not None else None
        built: List[ChannelCandidate] = []
        seen: set[str] = set()

        explicit_candidates = ensure_sequence(candidates, field_name="candidates", allow_none=True, coerce_scalar=False)
        for raw in explicit_candidates:
            if not isinstance(raw, Mapping):
                raise PayloadValidationError(
                    "Each candidate must be a mapping-like object.",
                    context={"operation": "build_channel_candidates"},
                    details={"received_type": type(raw).__name__},
                )
            candidate = self._mapping_to_candidate(raw, fallback_endpoint=endpoint)
            if normalized_protocol and candidate.protocol != normalized_protocol:
                continue
            if normalized_channel and candidate.channel != normalized_channel:
                continue
            if candidate.identity_key not in seen:
                built.append(candidate)
                seen.add(candidate.identity_key)

        if include_registered and self.prefer_registered_adapters:
            for spec in self.adapters.list_registered_adapters(enabled_only=True):
                adapter_name = str(spec["name"])
                spec_protocols = [normalize_protocol_name(item) for item in spec.get("protocols", [])]
                spec_channels = [normalize_channel_name(item) for item in spec.get("channels", [])]
                chosen_protocol = normalized_protocol or (spec_protocols[0] if spec_protocols else self.default_protocol)
                chosen_channel = normalized_channel or (spec_channels[0] if spec_channels else self.default_channel)
                if normalized_protocol and normalized_protocol not in spec_protocols:
                    continue
                if normalized_channel and normalized_channel not in spec_channels:
                    continue

                endpoint_hint = endpoint or self._endpoint_hint_for_adapter(adapter_name)
                candidate = ChannelCandidate(
                    adapter_name=adapter_name,
                    protocol=chosen_protocol,
                    channel=chosen_channel,
                    endpoint=self._safe_endpoint(endpoint_hint),
                    route=None,
                    region=self._region_hint(spec),
                    priority=int(spec.get("priority", 100) or 100),
                    cost=self.default_cost,
                    weight=self.default_weight,
                    secure=self._infer_candidate_secure(chosen_protocol, endpoint_hint),
                    capabilities=ensure_mapping(spec.get("capabilities"), field_name="capabilities", allow_none=True),
                    metadata=normalize_metadata(spec.get("metadata")),
                )
                if candidate.identity_key not in seen:
                    built.append(candidate)
                    seen.add(candidate.identity_key)

        built = built[: self.max_candidates_considered]
        self._stats["builds"] += 1

        if not built and self.fail_on_no_candidates:
            raise NoRouteAvailableError(
                "No channel candidates were available to score.",
                context={"operation": "build_channel_candidates", "channel": channel, "protocol": protocol, "endpoint": endpoint},
                details={"constraints": json_safe(constraints)},
            )
        return built

    def _rank_candidates_internal(
        self,
        candidates: Sequence[ChannelCandidate],
        *,
        protocol: Optional[str],
        channel: Optional[str],
        endpoint: Optional[str],
        constraints: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        for candidate in candidates:
            score = self._score_candidate(
                candidate,
                protocol=protocol,
                channel=channel,
                endpoint=endpoint,
                constraints=constraints,
                metadata=metadata,
            )
            if score.disqualified:
                self._stats["disqualifications"] += 1
            ranked.append({"candidate": candidate.to_dict(), "score": score.to_dict()})

        ranked.sort(key=self._rank_sort_key)
        return ranked

    def _score_candidate(
        self,
        candidate: ChannelCandidate,
        *,
        protocol: Optional[str],
        channel: Optional[str],
        endpoint: Optional[str],
        constraints: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> ChannelScoreBreakdown:
        breakdown = ChannelScoreBreakdown()
        requested_protocol = normalize_protocol_name(protocol) if protocol is not None else None
        requested_channel = normalize_channel_name(channel) if channel is not None else None
        requested_endpoint = self._safe_endpoint(endpoint)
        require_secure = bool(constraints.get("require_secure") or constraints.get("tls_required"))
        max_cost = constraints.get("max_cost")

        # Adapter/config priority.
        priority_rank = max(1, self._priority_rank(candidate.adapter_name))
        breakdown.adapter_priority_score += priority_rank * self.weight_adapter_priority
        breakdown.config_priority_score += max(1.0, float(candidate.priority)) * self.weight_config_priority / 100.0
        breakdown.add_reason(f"priority rank={priority_rank} adapter_priority={candidate.priority}")

        # Protocol and channel intent.
        if requested_protocol:
            if candidate.protocol == requested_protocol:
                breakdown.protocol_match_score += self.weight_protocol_match
                breakdown.add_reason(f"protocol matched {requested_protocol}")
            else:
                breakdown.add_disqualification(f"protocol mismatch: required {requested_protocol}, candidate {candidate.protocol}")

        if requested_channel:
            if candidate.channel == requested_channel:
                breakdown.channel_match_score += self.weight_channel_match
                breakdown.add_reason(f"channel matched {requested_channel}")
            else:
                breakdown.add_disqualification(f"channel mismatch: required {requested_channel}, candidate {candidate.channel}")

        # Preferred adapters/protocols/channels.
        preferred_protocols = self._normalized_value_set(constraints.get("preferred_protocols"), kind="protocol")
        preferred_channels = self._normalized_value_set(constraints.get("preferred_channels"), kind="channel")
        preferred_adapters = self._normalized_value_set(constraints.get("preferred_adapters"), kind="raw")
        preferred_regions = self._normalized_value_set(constraints.get("preferred_regions"), kind="raw")
        preferred_endpoint_patterns = tuple(
            str(item).strip() for item in ensure_sequence(constraints.get("preferred_endpoint_patterns"), field_name="preferred_endpoint_patterns", allow_none=True, coerce_scalar=True) if str(item).strip()
        )

        if candidate.protocol in preferred_protocols:
            breakdown.preferred_score += self.weight_preferred
            breakdown.add_reason(f"preferred protocol {candidate.protocol}")
        if candidate.channel in preferred_channels:
            breakdown.preferred_score += self.weight_preferred
            breakdown.add_reason(f"preferred channel {candidate.channel}")
        if candidate.adapter_name in preferred_adapters:
            breakdown.preferred_score += self.weight_preferred
            breakdown.add_reason(f"preferred adapter {candidate.adapter_name}")
        if candidate.region and candidate.region.lower() in preferred_regions:
            breakdown.affinity_score += self.weight_affinity
            breakdown.add_reason(f"preferred region {candidate.region}")
        if candidate.endpoint and any(pattern in candidate.endpoint for pattern in preferred_endpoint_patterns):
            breakdown.affinity_score += self.weight_affinity
            breakdown.add_reason("preferred endpoint pattern matched")
        if requested_endpoint and candidate.endpoint == requested_endpoint:
            breakdown.affinity_score += self.weight_affinity
            breakdown.add_reason("exact endpoint affinity matched")

        # Capability requirements.
        required_capabilities = ensure_mapping(constraints.get("required_capabilities"), field_name="required_capabilities", allow_none=True)
        preferred_capabilities = ensure_mapping(constraints.get("preferred_capabilities"), field_name="preferred_capabilities", allow_none=True)
        for key in _CAPABILITY_KEYS:
            if key in constraints:
                required_capabilities[key] = constraints[key]

        capability_matches = 0
        for key, expected in required_capabilities.items():
            expected_bool = bool(expected)
            actual_bool = bool(candidate.capabilities.get(key, False))
            if actual_bool != expected_bool:
                breakdown.add_disqualification(f"required capability {key}={expected_bool} not satisfied")
            else:
                capability_matches += 1
                breakdown.add_reason(f"required capability matched {key}={expected_bool}")

        for key, expected in preferred_capabilities.items():
            expected_bool = bool(expected)
            actual_bool = bool(candidate.capabilities.get(key, False))
            if actual_bool == expected_bool:
                capability_matches += 1
                breakdown.add_reason(f"preferred capability matched {key}={expected_bool}")

        if capability_matches:
            breakdown.capability_score += capability_matches * self.weight_capability

        # Security posture.
        secure_candidate = candidate.inferred_secure()
        if require_secure and not secure_candidate:
            breakdown.penalty_score += self.penalty_insecure_for_tls
            breakdown.add_disqualification("secure transport required but candidate is not secure")
        elif secure_candidate:
            breakdown.security_score += self.weight_security
            breakdown.add_reason("secure transport available")

        if requested_endpoint and self.prefer_secure_for_tls_endpoints:
            try:
                requested_secure = parse_endpoint(requested_endpoint, default_scheme=candidate.protocol, protocol=candidate.protocol, require_host=False).secure
            except Exception:
                requested_secure = is_secure_protocol(candidate.protocol)
            if requested_secure and secure_candidate:
                breakdown.security_score += self.weight_security
                breakdown.add_reason("secure endpoint alignment")

        # Cost posture.
        candidate_cost = max(0.0, float(candidate.cost or self.default_cost))
        if max_cost is not None:
            try:
                max_cost_value = float(max_cost)
            except (TypeError, ValueError) as exc:
                raise PayloadValidationError(
                    "max_cost constraint must be numeric.",
                    context={"operation": "score_channel_candidate"},
                    details={"max_cost": max_cost},
                    cause=exc,
                ) from exc
            if candidate_cost > max_cost_value:
                breakdown.penalty_score += self.penalty_high_cost
                breakdown.add_disqualification(f"candidate cost {candidate_cost} exceeds max_cost {max_cost_value}")
        breakdown.cost_score += max(0.0, self.weight_cost / max(candidate_cost, 1.0))
        breakdown.add_reason(f"cost={candidate_cost}")

        # Endpoint health and reliability.
        endpoint_health = self._endpoint_health(candidate)
        channel_metrics = self._channel_metrics(candidate)
        circuit_state = self._circuit_state(candidate)
        policy_penalty = self._policy_penalty(candidate, constraints)
        if policy_penalty > 0:
            breakdown.penalty_score += policy_penalty
            breakdown.add_reason(f"policy penalty {policy_penalty}")

        if circuit_state in {"open", "half_open"}:
            if self.disallow_open_circuits and circuit_state == "open":
                breakdown.penalty_score += self.penalty_open_circuit
                breakdown.add_disqualification("endpoint circuit is open")
            else:
                breakdown.penalty_score += self.penalty_open_circuit / 2.0
                breakdown.add_reason(f"circuit state {circuit_state}")

        if endpoint_health:
            status = str(endpoint_health.get("status", "")).strip().lower()
            latency_ms = self._coerce_optional_float(endpoint_health.get("latency_ms"))
            success_rate = self._coerce_optional_float(endpoint_health.get("success_rate"))
            error_rate = self._coerce_optional_float(endpoint_health.get("error_rate"))

            if status in self.healthy_statuses:
                breakdown.health_score += self.weight_health
                breakdown.add_reason(f"healthy endpoint status {status}")
            elif status in self.degraded_statuses:
                if not self.allow_degraded_channels and not bool(constraints.get("allow_degraded")):
                    breakdown.penalty_score += self.penalty_degraded
                    breakdown.add_disqualification(f"degraded endpoint status {status} rejected")
                else:
                    breakdown.health_score += self.weight_health * 0.35
                    breakdown.penalty_score += self.penalty_degraded
                    breakdown.add_reason(f"degraded endpoint status {status}")
            elif status in self.unhealthy_statuses:
                if not self.allow_unhealthy_channels and not bool(constraints.get("allow_unhealthy")):
                    breakdown.penalty_score += self.penalty_unhealthy
                    breakdown.add_disqualification(f"unhealthy endpoint status {status} rejected")
                else:
                    breakdown.penalty_score += self.penalty_unhealthy / 2.0
                    breakdown.add_reason(f"unhealthy endpoint status {status}")

            if success_rate is not None:
                breakdown.reliability_score += max(0.0, min(success_rate, 1.0)) * self.weight_reliability
                breakdown.add_reason(f"endpoint success_rate={success_rate}")
            if error_rate is not None:
                breakdown.reliability_score += max(0.0, 1.0 - min(error_rate, 1.0)) * (self.weight_reliability / 2.0)
                breakdown.add_reason(f"endpoint error_rate={error_rate}")
            if latency_ms is not None:
                breakdown.latency_score += self._latency_score(latency_ms)
                breakdown.add_reason(f"endpoint latency_ms={latency_ms}")
        else:
            breakdown.penalty_score += self.penalty_missing_health
            breakdown.add_reason("missing endpoint health snapshot")

        if channel_metrics:
            success_rate = self._coerce_optional_float(channel_metrics.get("success_rate"))
            retry_rate = self._coerce_optional_float(channel_metrics.get("retry_rate"))
            error_rate = self._coerce_optional_float(channel_metrics.get("error_rate"))
            latency_ms = (
                self._coerce_optional_float(channel_metrics.get("p95_latency_ms"))
                or self._coerce_optional_float(channel_metrics.get("latency_ms"))
                or self._coerce_optional_float(channel_metrics.get("avg_latency_ms"))
            )

            if success_rate is not None:
                breakdown.reliability_score += max(0.0, min(success_rate, 1.0)) * (self.weight_reliability / 1.5)
                breakdown.add_reason(f"channel success_rate={success_rate}")
            if error_rate is not None:
                breakdown.reliability_score += max(0.0, 1.0 - min(error_rate, 1.0)) * (self.weight_reliability / 2.5)
                breakdown.add_reason(f"channel error_rate={error_rate}")
            if retry_rate is not None:
                breakdown.reliability_score += max(0.0, 1.0 - min(retry_rate, 1.0)) * (self.weight_reliability / 2.5)
                breakdown.add_reason(f"channel retry_rate={retry_rate}")
            if latency_ms is not None:
                breakdown.latency_score += self._latency_score(latency_ms)
                breakdown.add_reason(f"channel latency_ms={latency_ms}")
        else:
            breakdown.penalty_score += self.penalty_missing_metrics
            breakdown.add_reason("missing channel metrics snapshot")

        total = (
            breakdown.adapter_priority_score
            + breakdown.config_priority_score
            + breakdown.protocol_match_score
            + breakdown.channel_match_score
            + breakdown.preferred_score
            + breakdown.capability_score
            + breakdown.health_score
            + breakdown.reliability_score
            + breakdown.latency_score
            + breakdown.security_score
            + breakdown.cost_score
            + breakdown.affinity_score
            - breakdown.penalty_score
        )
        breakdown.total_score = total * max(float(candidate.weight), 0.0)
        return breakdown

    # ------------------------------------------------------------------
    # Internal memory/config helpers
    # ------------------------------------------------------------------
    def _record_selection(self, decision: ChannelSelectionDecision) -> None:
        payload = decision.to_dict()
        if self.record_selection_snapshots:
            self.memory.set(
                _CHANNEL_SELECTOR_LAST_KEY,
                payload,
                ttl_seconds=self.selection_ttl_seconds,
                source="channel_selector",
            )
            self.memory.append(
                _CHANNEL_SELECTOR_HISTORY_KEY,
                payload,
                max_items=self.max_history_size,
                ttl_seconds=self.history_ttl_seconds,
                source="channel_selector",
            )
            self.memory.set(
                _CHANNEL_SELECTOR_SNAPSHOT_KEY,
                self.get_selector_snapshot(),
                ttl_seconds=self.selection_ttl_seconds,
                source="channel_selector",
            )
            self.memory.set_route_selection(
                selected_route={
                    "adapter_name": decision.selected.adapter_name,
                    "protocol": decision.selected.protocol,
                    "channel": decision.selected.channel,
                    "endpoint": decision.selected.endpoint,
                    "route": decision.selected.route,
                    "score": decision.score.total_score,
                    "secure": decision.selected.inferred_secure(),
                },
                candidate_routes=[
                    {
                        "adapter_name": item["candidate"].get("adapter_name"),
                        "protocol": item["candidate"].get("protocol"),
                        "channel": item["candidate"].get("channel"),
                        "endpoint": item["candidate"].get("endpoint"),
                        "score": item["score"].get("total_score"),
                        "disqualified": item["score"].get("disqualified"),
                    }
                    for item in decision.candidates
                ],
                route_id=decision.selected.identity_key,
                reason=", ".join(decision.score.reasons[:4]) if decision.score.reasons else "best weighted candidate",
                ttl_seconds=self.selection_ttl_seconds,
                metadata={"selector": "channel_selector", **normalize_metadata(decision.metadata)},
            )

    def _endpoint_health(self, candidate: ChannelCandidate) -> Dict[str, Any]:
        if not self.use_memory_health or not candidate.endpoint:
            return {}
        endpoint_health = self.memory.get("network.endpoint.health", default={})
        if isinstance(endpoint_health, Mapping):
            exact = endpoint_health.get(candidate.endpoint)
            if isinstance(exact, Mapping):
                return dict(exact)
            normalized = self._safe_endpoint(candidate.endpoint)
            if normalized and isinstance(endpoint_health.get(normalized), Mapping):
                return dict(endpoint_health[normalized])
        return {}

    def _channel_metrics(self, candidate: ChannelCandidate) -> Dict[str, Any]:
        if not self.use_memory_metrics:
            return {}
        channel_metrics = self.memory.get("network.telemetry.channel_metrics", default={})
        if isinstance(channel_metrics, Mapping):
            snapshot = channel_metrics.get(candidate.channel)
            if isinstance(snapshot, Mapping):
                return dict(snapshot)
        return {}

    def _circuit_state(self, candidate: ChannelCandidate) -> Optional[str]:
        if not candidate.endpoint:
            return None
        circuit_state = self.memory.get("network.endpoint.circuit_state", default={})
        if isinstance(circuit_state, Mapping):
            snapshot = circuit_state.get(candidate.endpoint)
            if isinstance(snapshot, Mapping):
                state = snapshot.get("circuit_state")
                return str(state).strip().lower() if state is not None else None
        return None

    def _policy_penalty(self, candidate: ChannelCandidate, constraints: Mapping[str, Any]) -> float:
        if not self.use_policy_memory:
            return 0.0
        required_policy_status = ensure_mapping(constraints.get("required_policy_status"), field_name="required_policy_status", allow_none=True)
        if not required_policy_status:
            return 0.0
        policy_memory = self.memory.get("network.policy.decision", default={})
        if not isinstance(policy_memory, Mapping):
            return 0.0

        penalty = 0.0
        for policy_name, expected_status in required_policy_status.items():
            snapshot = policy_memory.get(policy_name)
            actual_status = None
            if isinstance(snapshot, Mapping):
                decision = snapshot.get("decision")
                if isinstance(decision, str):
                    actual_status = decision.strip().lower()
            if actual_status is None:
                penalty += self.penalty_missing_health
            elif actual_status != str(expected_status).strip().lower():
                penalty += self.penalty_unhealthy / 2.0
        return penalty

    def _latency_score(self, latency_ms: float) -> float:
        normalized_latency = max(0.0, float(latency_ms))
        ratio = min(normalized_latency / max(self.reference_latency_ms, 1.0), 10.0)
        return max(0.0, (1.0 - min(ratio, 1.0))) * self.weight_latency

    def _mapping_to_candidate(self, value: Mapping[str, Any], *, fallback_endpoint: Optional[str] = None) -> ChannelCandidate:
        payload = ensure_mapping(value, field_name="candidate")
        adapter_name = ensure_non_empty_string(str(payload.get("adapter_name") or payload.get("name") or payload.get("adapter")), field_name="adapter_name").strip().lower()
        protocol = normalize_protocol_name(str(payload.get("protocol") or self.default_protocol))
        channel = normalize_channel_name(str(payload.get("channel") or protocol or self.default_channel))
        endpoint = self._safe_endpoint(payload.get("endpoint") or fallback_endpoint)
        region = str(payload.get("region")).strip() if payload.get("region") is not None and str(payload.get("region")).strip() else None
        route = str(payload.get("route")).strip() if payload.get("route") is not None and str(payload.get("route")).strip() else None
        priority = int(payload.get("priority", 100) or 100)
        cost = float(payload.get("cost", self.default_cost) or self.default_cost)
        weight = float(payload.get("weight", self.default_weight) or self.default_weight)
        secure = payload.get("secure")
        capabilities = ensure_mapping(payload.get("capabilities"), field_name="capabilities", allow_none=True)
        metadata = normalize_metadata(payload.get("metadata"))
        return ChannelCandidate(
            adapter_name=adapter_name,
            protocol=protocol,
            channel=channel,
            endpoint=endpoint,
            route=route,
            region=region,
            priority=priority,
            cost=cost,
            weight=weight,
            secure=bool(secure) if secure is not None else None,
            capabilities=capabilities,
            metadata=metadata,
        )

    def _mapping_to_score(self, value: Mapping[str, Any]) -> ChannelScoreBreakdown:
        payload = ensure_mapping(value, field_name="score")
        score = ChannelScoreBreakdown(
            total_score=float(payload.get("total_score", 0.0) or 0.0),
            adapter_priority_score=float(payload.get("adapter_priority_score", 0.0) or 0.0),
            config_priority_score=float(payload.get("config_priority_score", 0.0) or 0.0),
            protocol_match_score=float(payload.get("protocol_match_score", 0.0) or 0.0),
            channel_match_score=float(payload.get("channel_match_score", 0.0) or 0.0),
            preferred_score=float(payload.get("preferred_score", 0.0) or 0.0),
            capability_score=float(payload.get("capability_score", 0.0) or 0.0),
            health_score=float(payload.get("health_score", 0.0) or 0.0),
            reliability_score=float(payload.get("reliability_score", 0.0) or 0.0),
            latency_score=float(payload.get("latency_score", 0.0) or 0.0),
            security_score=float(payload.get("security_score", 0.0) or 0.0),
            cost_score=float(payload.get("cost_score", 0.0) or 0.0),
            affinity_score=float(payload.get("affinity_score", 0.0) or 0.0),
            penalty_score=float(payload.get("penalty_score", 0.0) or 0.0),
            reasons=list(payload.get("reasons") or []),
            disqualified=bool(payload.get("disqualified", False)),
            disqualification_reasons=list(payload.get("disqualification_reasons") or []),
        )
        return score

    def _rank_sort_key(self, value: Mapping[str, Any]) -> Tuple[int, float, str]:
        score = ensure_mapping(value.get("score"), field_name="score", allow_none=True)
        candidate = ensure_mapping(value.get("candidate"), field_name="candidate", allow_none=True)
        disqualified = bool(score.get("disqualified", False))
        total_score = float(score.get("total_score", 0.0) or 0.0)
        identity = str(candidate.get("identity_key") or candidate.get("adapter_name") or "")
        return (1 if disqualified else 0, -total_score, identity)

    def _priority_rank(self, adapter_name: str) -> int:
        try:
            index = list(self.adapters.priority_order).index(adapter_name)
            return max(1, len(self.adapters.priority_order) - index)
        except ValueError:
            return 1

    def _endpoint_hint_for_adapter(self, adapter_name: str) -> Optional[str]:
        mapping = {
            "http": get_config_section("network_http_adapter").get("endpoint") if isinstance(get_config_section("network_http_adapter"), Mapping) else None,
            "websocket": get_config_section("network_websocket_adapter").get("endpoint") if isinstance(get_config_section("network_websocket_adapter"), Mapping) else None,
            "grpc": get_config_section("network_grpc_adapter").get("endpoint") if isinstance(get_config_section("network_grpc_adapter"), Mapping) else None,
            "queue": get_config_section("network_queue_adapter").get("endpoint") if isinstance(get_config_section("network_queue_adapter"), Mapping) else None,
        }
        return self._safe_endpoint(mapping.get(adapter_name))

    def _region_hint(self, spec: Mapping[str, Any]) -> Optional[str]:
        metadata = ensure_mapping(spec.get("metadata"), field_name="metadata", allow_none=True)
        region = metadata.get("region")
        if region is None:
            return None
        text = str(region).strip()
        return text or None

    def _infer_candidate_secure(self, protocol: str, endpoint: Optional[str]) -> bool:
        if endpoint:
            try:
                return bool(parse_endpoint(endpoint, default_scheme=protocol, protocol=protocol, require_host=False).secure)
            except Exception:
                pass
        return is_secure_protocol(protocol)

    def _normalize_constraints(self, constraints: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        payload = ensure_mapping(constraints, field_name="constraints", allow_none=True)
        normalized = normalize_metadata(payload)
        for key in ("preferred_protocols", "required_protocols"):
            if key in normalized:
                normalized[key] = [normalize_protocol_name(str(item)) for item in ensure_sequence(normalized[key], field_name=key, allow_none=True, coerce_scalar=True)]
        for key in ("preferred_channels", "required_channels"):
            if key in normalized:
                normalized[key] = [normalize_channel_name(str(item)) for item in ensure_sequence(normalized[key], field_name=key, allow_none=True, coerce_scalar=True)]
        if "preferred_adapters" in normalized:
            normalized["preferred_adapters"] = [str(item).strip().lower() for item in ensure_sequence(normalized["preferred_adapters"], field_name="preferred_adapters", allow_none=True, coerce_scalar=True)]
        return normalized

    def _normalized_value_set(self, value: Any, *, kind: str) -> set[str]:
        values = ensure_sequence(value, field_name=kind, allow_none=True, coerce_scalar=True)
        normalized: set[str] = set()
        for item in values:
            text = str(item).strip().lower()
            if not text:
                continue
            if kind == "protocol":
                normalized.add(normalize_protocol_name(text))
            elif kind == "channel":
                normalized.add(normalize_channel_name(text))
            else:
                normalized.add(text)
        return normalized

    def _safe_endpoint(self, endpoint: Any) -> Optional[str]:
        if endpoint is None:
            return None
        text = str(endpoint).strip()
        if not text:
            return None
        try:
            if "://" in text:
                return normalize_endpoint(text)
        except Exception:
            return text
        return text

    def _coerce_optional_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def _get_bool_config(self, name: str, default: bool) -> bool:
        value = self.selector_config.get(name, default)
        return self._coerce_bool(value, default)

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        value = self.selector_config.get(name, default)
        return int(self._coerce_float(value, float(default), non_negative=True))

    def _get_non_negative_float_config(self, name: str, default: float) -> float:
        value = self.selector_config.get(name, default)
        return self._coerce_float(value, default, non_negative=True)

    def _get_positive_float_config(self, name: str, default: float) -> float:
        value = self.selector_config.get(name, default)
        coerced = self._coerce_float(value, default, non_negative=False)
        if coerced <= 0:
            raise NetworkConfigurationError(
                "Configuration float value must be > 0.",
                context={"operation": "channel_selector_config", "channel": self.default_channel, "protocol": self.default_protocol},
                details={"config_key": name, "config_value": value},
            )
        return coerced

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.selector_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _get_strategy_config(self, name: str, default: str) -> str:
        value = str(self.selector_config.get(name, default)).strip().lower() or default
        if value not in _VALID_SELECTION_STRATEGIES:
            raise NetworkConfigurationError(
                "Invalid selection strategy in channel selector configuration.",
                context={"operation": "channel_selector_config", "channel": self.default_channel, "protocol": self.default_protocol},
                details={"config_key": name, "config_value": value},
            )
        return value

    def _get_status_set(self, name: str, default: Sequence[str]) -> set[str]:
        values = self.selector_config.get(name, default)
        result: set[str] = set()
        for item in ensure_sequence(values, field_name=name, allow_none=True, coerce_scalar=True):
            text = str(item).strip().lower()
            if text:
                result.add(text)
        return result or set(default)

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
            "Invalid boolean value in channel selector configuration.",
            context={"operation": "channel_selector_config", "channel": self.default_channel, "protocol": self.default_protocol},
            details={"config_value": value},
        )

    def _coerce_float(self, value: Any, default: float, *, non_negative: bool = False) -> float:
        if value is None:
            value = default
        try:
            coerced = float(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid float value in channel selector configuration.",
                context={"operation": "channel_selector_config", "channel": self.default_channel, "protocol": self.default_protocol},
                details={"config_value": value},
                cause=exc,
            ) from exc
        if non_negative and coerced < 0:
            raise NetworkConfigurationError(
                "Configuration float value must be non-negative.",
                context={"operation": "channel_selector_config", "channel": self.default_channel, "protocol": self.default_protocol},
                details={"config_value": value},
            )
        return coerced

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    print("\n=== Running Channel Selector ===\n")
    printer.status("TEST", "Channel Selector initialized", "info")

    shared_memory = NetworkMemory()
    adapters = NetworkAdapters(memory=shared_memory)
    selector = ChannelSelector(memory=shared_memory, adapters=adapters)

    shared_memory.record_channel_metrics(
        "http",
        {
            "success_rate": 0.992,
            "retry_rate": 0.018,
            "error_rate": 0.008,
            "p95_latency_ms": 84,
            "requests": 1280,
        },
        metadata={"window": "5m"},
    )
    shared_memory.record_channel_metrics(
        "websocket",
        {
            "success_rate": 0.981,
            "retry_rate": 0.011,
            "error_rate": 0.019,
            "p95_latency_ms": 42,
            "requests": 744,
        },
        metadata={"window": "5m"},
    )
    shared_memory.record_channel_metrics(
        "grpc",
        {
            "success_rate": 0.975,
            "retry_rate": 0.025,
            "error_rate": 0.025,
            "p95_latency_ms": 56,
            "requests": 612,
        },
        metadata={"window": "5m"},
    )
    shared_memory.record_channel_metrics(
        "queue",
        {
            "success_rate": 0.999,
            "retry_rate": 0.002,
            "error_rate": 0.001,
            "p95_latency_ms": 220,
            "requests": 3392,
        },
        metadata={"window": "5m"},
    )
    printer.status("TEST", "Channel metrics primed", "info")

    shared_memory.update_endpoint_health(
        "https://api.example.com/v1/jobs",
        status="healthy",
        latency_ms=76,
        success_rate=0.996,
        error_rate=0.004,
        metadata={"region": "eu-west"},
    )
    shared_memory.update_endpoint_health(
        "wss://stream.example.com/events",
        status="healthy",
        latency_ms=38,
        success_rate=0.987,
        error_rate=0.013,
        metadata={"region": "eu-west"},
    )
    shared_memory.update_endpoint_health(
        "grpcs://rpc.example.com:443/RelayService",
        status="degraded",
        latency_ms=112,
        success_rate=0.962,
        error_rate=0.038,
        metadata={"region": "eu-west"},
    )
    shared_memory.update_endpoint_health(
        "amqps://broker.example.com:5671",
        status="healthy",
        latency_ms=210,
        success_rate=0.999,
        error_rate=0.001,
        metadata={"region": "eu-west"},
    )
    printer.status("TEST", "Endpoint health primed", "info")

    shared_memory.record_policy_decision(
        "destination_allowlist",
        "allowed",
        endpoint="https://api.example.com/v1/jobs",
        protocol="https",
        channel="http",
        metadata={"scope": "demo"},
    )
    printer.status("TEST", "Policy memory primed", "info")

    explicit_candidates = [
        {
            "adapter_name": "http",
            "protocol": "https",
            "channel": "http",
            "endpoint": "https://api.example.com/v1/jobs",
            "priority": 20,
            "cost": 1.0,
            "capabilities": {"supports_request_reply": True, "supports_receive": True, "supports_tls": True},
            "metadata": {"region": "eu-west"},
        },
        {
            "adapter_name": "websocket",
            "protocol": "websocket",
            "channel": "websocket",
            "endpoint": "wss://stream.example.com/events",
            "priority": 18,
            "cost": 1.2,
            "capabilities": {
                "supports_streaming": True,
                "supports_bidirectional_streaming": True,
                "supports_receive": True,
                "supports_tls": True,
            },
            "metadata": {"region": "eu-west"},
        },
        {
            "adapter_name": "grpc",
            "protocol": "grpc",
            "channel": "grpc",
            "endpoint": "grpcs://rpc.example.com:443/RelayService",
            "priority": 17,
            "cost": 1.1,
            "capabilities": {
                "supports_streaming": True,
                "supports_bidirectional_streaming": True,
                "supports_request_reply": True,
                "supports_receive": True,
                "supports_tls": True,
            },
            "metadata": {"region": "eu-west"},
        },
        {
            "adapter_name": "queue",
            "protocol": "queue",
            "channel": "queue",
            "endpoint": "amqps://broker.example.com:5671",
            "priority": 14,
            "cost": 0.6,
            "capabilities": {
                "supports_batch_send": True,
                "supports_ack": True,
                "supports_nack": True,
                "supports_receive": True,
                "supports_tls": True,
            },
            "metadata": {"region": "eu-west"},
        },
    ]

    request_reply_decision = selector.select_channel(
        protocol="https",
        channel="http",
        endpoint="https://api.example.com/v1/jobs",
        candidates=explicit_candidates,
        constraints={
            "require_secure": True,
            "supports_request_reply": True,
            "preferred_adapters": ["http"],
            "preferred_protocols": ["https", "http"],
            "preferred_regions": ["eu-west"],
        },
        metadata={"task_class": "request_reply"},
    )
    printer.status("TEST", "Request/reply channel selected", "info")

    streaming_decision = selector.select_channel(
        endpoint="wss://stream.example.com/events",
        candidates=explicit_candidates,
        constraints={
            "require_secure": True,
            "supports_streaming": True,
            "supports_bidirectional_streaming": True,
            "preferred_channels": ["websocket"],
            "preferred_regions": ["eu-west"],
            "allow_degraded": True,
        },
        metadata={"task_class": "streaming"},
    )
    printer.status("TEST", "Streaming channel selected", "info")

    ranked = selector.rank_candidates(
        endpoint="https://api.example.com/v1/jobs",
        candidates=explicit_candidates,
        constraints={"require_secure": True},
    )
    snapshot = selector.get_selector_snapshot()

    print("Request/Reply Decision:", stable_json_dumps(request_reply_decision))
    print("Streaming Decision:", stable_json_dumps(streaming_decision))
    print("Ranked Candidates:", stable_json_dumps(ranked))
    print("Selector Snapshot:", stable_json_dumps(snapshot))

    assert request_reply_decision["selected"]["adapter_name"] == "http"
    assert streaming_decision["selected"]["adapter_name"] in {"websocket", "grpc"}
    assert ranked[0]["score"]["disqualified"] is False
    assert shared_memory.get("network.route.selected") is not None
    assert shared_memory.get("network.routing.channel_selector.last") is not None
    assert shared_memory.get("network.routing.channel_selector.history") is not None

    printer.status("TEST", "All Channel Selector checks passed", "info")
    print("\n=== Test ran successfully ===\n")
