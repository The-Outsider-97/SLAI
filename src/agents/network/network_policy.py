"""
Network Policy Controls
- Outbound destination allowlist/denylist checks.
- Protocol and port policy checks.
- TLS/certificate posture checks.
- Rate limit coordination with global handler/compliance layers.

This module owns transport-facing policy enforcement for the network subsystem.
It is intentionally focused on communication guardrails rather than business
logic, routing strategy, or generic workflow arbitration. The policy layer
exists to make consistent, explainable decisions about whether a network action
is permitted, what posture requirements apply, and which network facts should be
published for the rest of the subsystem.

Its responsibilities are deliberately scoped to network-policy concerns:

- destination allowlist and denylist evaluation,
- protocol / channel / port gatekeeping,
- TLS posture and certificate validation checks,
- rate-limit coordination by policy class and request key,
- structured decision recording for memory and observability,
- and stable, JSON-safe policy snapshots for downstream agents.

The policy layer does not attempt to replace Safety, Privacy, or Compliance
agents. Instead, it enforces transport-native controls locally and emits
decision facts that other agents can consume.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import fnmatch
from threading import RLock
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .utils import *
from .network_memory import NetworkMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Network Policy")
printer = PrettyPrinter()


@dataclass(slots=True)
class PolicyCheckResult:
    """Structured result for a single policy check."""

    name: str
    allowed: bool
    decision: str
    reason: str
    severity: str = "info"
    details: Dict[str, Any] = field(default_factory=dict)
    error_snapshot: Optional[Dict[str, Any]] = None
    exception: Optional[NetworkError] = field(default=None, repr=False)

    def to_dict(self, *, sanitize_logs_enabled: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "allowed": self.allowed,
            "decision": self.decision,
            "reason": self.reason,
            "severity": self.severity,
            "details": json_safe(self.details),
            "error": self.error_snapshot,
        }
        payload = {key: value for key, value in payload.items() if value not in (None, {}, [])}
        if sanitize_logs_enabled:
            return sanitize_for_logging(payload)
        return json_safe(payload)


@dataclass(slots=True)
class RateLimitBucket:
    """Mutable rolling counter for rate-limit enforcement."""

    key: str
    rate_class: str
    limit: int
    window_seconds: int
    window_started_at: datetime
    count: int = 0
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def refresh(self, now: datetime) -> None:
        if now >= self.window_started_at + timedelta(seconds=self.window_seconds):
            self.window_started_at = now
            self.count = 0
        self.last_seen_at = now

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.count)

    def would_exceed(self) -> bool:
        return self.count >= self.limit

    def consume(self) -> None:
        self.count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "rate_class": self.rate_class,
            "limit": self.limit,
            "window_seconds": self.window_seconds,
            "window_started_at": self.window_started_at.isoformat(),
            "count": self.count,
            "remaining": self.remaining,
            "last_seen_at": self.last_seen_at.isoformat(),
        }


class NetworkPolicy:
    """
    Transport guardrails for destinations, protocols, TLS posture, and rate classes.

    The class exposes a primary `evaluate(...)` flow for outbound network actions
    and a set of smaller helpers for individual policy domains. Decisions are
    returned as structured snapshots and can optionally raise typed network
    policy exceptions for hard enforcement.
    """

    def __init__(
        self,
        memory: Optional[NetworkMemory] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = load_global_config()
        self.policy_config = get_config_section("network_policy") or {}
        if config:
            self.policy_config = merge_mappings(self.policy_config, config)

        self.memory = memory or NetworkMemory()
        self._lock = RLock()

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.record_policy_decisions = self._get_bool_config("record_policy_decisions", True)

        self.decision_ttl_seconds = self._get_non_negative_int_config("decision_ttl_seconds", 1800)
        self.max_decision_history_size = max(1, self._get_non_negative_int_config("max_decision_history_size", 500))
        self.max_rate_buckets = max(1, self._get_non_negative_int_config("max_rate_buckets", 2000))
        self.default_rate_window_seconds = max(1, self._get_non_negative_int_config("default_rate_window_seconds", 60))

        self.enforce_destination_policy = self._get_bool_config("enforce_destination_policy", True)
        self.enforce_protocol_policy = self._get_bool_config("enforce_protocol_policy", True)
        self.enforce_port_policy = self._get_bool_config("enforce_port_policy", True)
        self.enforce_tls_policy = self._get_bool_config("enforce_tls_policy", True)
        self.enforce_certificate_policy = self._get_bool_config("enforce_certificate_policy", True)
        self.enforce_rate_limits = self._get_bool_config("enforce_rate_limits", True)

        self.require_tls_by_default = self._get_bool_config("require_tls_by_default", True)
        self.allow_private_hosts = self._get_bool_config("allow_private_hosts", False)
        self.allow_loopback_hosts = self._get_bool_config("allow_loopback_hosts", False)
        self.allow_ip_hosts = self._get_bool_config("allow_ip_hosts", True)

        self.deny_self_signed_certificates = self._get_bool_config("deny_self_signed_certificates", True)
        self.deny_expired_certificates = self._get_bool_config("deny_expired_certificates", True)
        self.require_hostname_match = self._get_bool_config("require_hostname_match", True)
        self.require_certificate_info_for_tls = self._get_bool_config("require_certificate_info_for_tls", False)

        self.allowed_protocols = self._get_sequence_config("allowed_protocols", ["http", "https", "websocket", "grpc", "queue"])
        self.denied_protocols = self._get_sequence_config("denied_protocols", [])
        self.allowed_channels = self._get_sequence_config("allowed_channels", [])
        self.denied_channels = self._get_sequence_config("denied_channels", [])
        self.allowed_ports = self._get_port_set_config("allowed_ports", [])
        self.denied_ports = self._get_port_set_config("denied_ports", [])
        self.allowed_destination_patterns = self._get_sequence_config("allowed_destination_patterns", [])
        self.denied_destination_patterns = self._get_sequence_config("denied_destination_patterns", [])
        self.insecure_destination_patterns = self._get_sequence_config("insecure_destination_patterns", [])
        self.tls_required_protocols = self._get_sequence_config("tls_required_protocols", ["https", "grpc"])
        self.allowed_certificate_authorities = self._get_sequence_config("allowed_certificate_authorities", [])

        self.rate_limits = self._get_rate_limits_config(
            "rate_limits",
            {
                "default": {"limit": 1000, "window_seconds": 60},
                "strict": {"limit": 100, "window_seconds": 60},
                "burst": {"limit": 5000, "window_seconds": 60},
            },
        )

        self._decision_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_decision_history_size)
        self._rate_buckets: "OrderedDict[str, RateLimitBucket]" = OrderedDict()
        self._stats: Dict[str, int] = {
            "evaluations": 0,
            "allowed": 0,
            "denied": 0,
            "rate_limit_hits": 0,
            "tls_denials": 0,
            "certificate_denials": 0,
            "destination_denials": 0,
            "protocol_denials": 0,
            "port_denials": 0,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        envelope: Optional[Mapping[str, Any]] = None,
        *,
        endpoint: Optional[str] = None,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        headers: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        tls_required: Optional[bool] = None,
        certificate_info: Optional[Mapping[str, Any]] = None,
        rate_limit_key: Optional[str] = None,
        rate_class: str = "default",
        consume_rate_limit: bool = True,
        raise_on_violation: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate outbound transport policy for a network action.

        The method returns a rich decision snapshot regardless of allow/deny
        outcome. When `raise_on_violation=True`, the first violating check is
        raised as a typed network policy exception after the full decision is
        recorded.
        """
        base_envelope = ensure_mapping(envelope, field_name="envelope", allow_none=True)
        effective_constraints = ensure_mapping(constraints, field_name="constraints", allow_none=True)
        request = self._resolve_request(
            envelope=base_envelope,
            endpoint=endpoint,
            protocol=protocol,
            channel=channel,
            headers=headers,
            metadata=metadata,
            tls_required=tls_required,
            certificate_info=certificate_info,
            rate_limit_key=rate_limit_key,
            rate_class=rate_class,
            constraints=effective_constraints,
        )

        if not self.enabled:
            snapshot = {
                "decision_id": generate_message_id(prefix="policy"),
                "policy_name": "network_policy.final",
                "decision": "allowed",
                "allowed": True,
                "reason": "Network policy module is disabled.",
                "recorded_at": utc_timestamp(),
                "request": request["public_request"],
                "checks": [],
                "violations": [],
                "metadata": request["metadata"],
            }
            self._record_decision(snapshot, ttl_seconds=self.decision_ttl_seconds)
            return snapshot

        checks = [
            self._check_destination_policy(request),
            self._check_protocol_policy(request),
            self._check_port_policy(request),
            self._check_tls_policy(request),
            self._check_certificate_policy(request),
        ]

        prior_allowed = all(check.allowed for check in checks)
        checks.append(
            self._check_rate_limit_policy(
                request,
                rate_class=request["rate_class"],
                rate_limit_key=request["rate_limit_key"],
                consume=consume_rate_limit and prior_allowed,
            )
        )

        violations = [check.to_dict(sanitize_logs_enabled=self.sanitize_logs) for check in checks if not check.allowed]
        allowed = not violations
        final_reason = "All network policy checks passed." if allowed else checks[[idx for idx, chk in enumerate(checks) if not chk.allowed][0]].reason
        decision = "allowed" if allowed else "denied"

        snapshot = {
            "decision_id": generate_message_id(prefix="policy"),
            "policy_name": "network_policy.final",
            "decision": decision,
            "allowed": allowed,
            "reason": final_reason,
            "recorded_at": utc_timestamp(),
            "request": request["public_request"],
            "checks": [check.to_dict(sanitize_logs_enabled=self.sanitize_logs) for check in checks],
            "violations": violations,
            "metadata": request["metadata"],
        }

        self._stats["evaluations"] += 1
        if allowed:
            self._stats["allowed"] += 1
        else:
            self._stats["denied"] += 1

        self._record_decision(snapshot, ttl_seconds=self.decision_ttl_seconds)

        if raise_on_violation and not allowed:
            first_violation = next(check for check in checks if not check.allowed)
            if first_violation.exception is not None:
                raise first_violation.exception
            raise PolicyViolationError(
                first_violation.reason,
                context={
                    "operation": "policy_evaluate",
                    "endpoint": request["public_request"].get("endpoint"),
                    "protocol": request["public_request"].get("protocol"),
                    "channel": request["public_request"].get("channel"),
                    "policy_name": first_violation.name,
                },
                details=first_violation.details,
            )

        return snapshot

    def assert_allowed(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate policy and raise on the first violation."""
        kwargs["raise_on_violation"] = True
        return self.evaluate(*args, **kwargs)

    def preview(
        self,
        envelope: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate policy without consuming rate-limit counters."""
        kwargs["consume_rate_limit"] = False
        kwargs["raise_on_violation"] = False
        return self.evaluate(envelope=envelope, **kwargs)

    def evaluate_destination(
        self,
        endpoint: str,
        *,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        raise_on_violation: bool = False,
    ) -> Dict[str, Any]:
        """Convenience wrapper for destination-centric checks."""
        return self.evaluate(
            endpoint=endpoint,
            protocol=protocol,
            channel=channel,
            constraints=constraints,
            raise_on_violation=raise_on_violation,
            consume_rate_limit=False,
        )

    def reset_rate_limit(self, key: str) -> bool:
        """Reset a specific rate-limit bucket."""
        normalized_key = ensure_non_empty_string(key, field_name="rate_limit_key")
        with self._lock:
            removed = self._rate_buckets.pop(normalized_key, None)
            return removed is not None

    def clear_rate_limits(self) -> int:
        """Clear all rate-limit buckets."""
        with self._lock:
            count = len(self._rate_buckets)
            self._rate_buckets.clear()
            return count

    def get_rate_limit_state(self) -> Dict[str, Any]:
        """Return the current rate-limit bucket state."""
        with self._lock:
            return {
                "generated_at": utc_timestamp(),
                "bucket_count": len(self._rate_buckets),
                "buckets": [bucket.to_dict() for bucket in self._rate_buckets.values()],
            }

    def get_recent_decisions(self, *, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent policy decisions in most-recent-first order."""
        resolved_limit = max(1, int(limit))
        with self._lock:
            decisions = list(self._decision_history)[-resolved_limit:]
        decisions.reverse()
        return decisions

    def get_policy_state(self) -> Dict[str, Any]:
        """Return a coarse module state snapshot for debugging and observability."""
        with self._lock:
            return {
                "generated_at": utc_timestamp(),
                "enabled": self.enabled,
                "stats": dict(self._stats),
                "decision_history_size": len(self._decision_history),
                "rate_bucket_count": len(self._rate_buckets),
                "policy_config": {
                    "require_tls_by_default": self.require_tls_by_default,
                    "allow_private_hosts": self.allow_private_hosts,
                    "allow_loopback_hosts": self.allow_loopback_hosts,
                    "allow_ip_hosts": self.allow_ip_hosts,
                    "allowed_protocols": sorted(self.allowed_protocols),
                    "denied_protocols": sorted(self.denied_protocols),
                    "allowed_ports": sorted(self.allowed_ports),
                    "denied_ports": sorted(self.denied_ports),
                    "allowed_destination_patterns": list(self.allowed_destination_patterns),
                    "denied_destination_patterns": list(self.denied_destination_patterns),
                    "tls_required_protocols": list(self.tls_required_protocols),
                    "rate_limit_classes": sorted(self.rate_limits.keys()),
                },
            }

    # ------------------------------------------------------------------ #
    # Request normalization
    # ------------------------------------------------------------------ #
    def _resolve_request(
        self,
        *,
        envelope: Mapping[str, Any],
        endpoint: Optional[str],
        protocol: Optional[str],
        channel: Optional[str],
        headers: Optional[Mapping[str, Any]],
        metadata: Optional[Mapping[str, Any]],
        tls_required: Optional[bool],
        certificate_info: Optional[Mapping[str, Any]],
        rate_limit_key: Optional[str],
        rate_class: str,
        constraints: Mapping[str, Any],
    ) -> Dict[str, Any]:
        resolved_endpoint = endpoint or envelope.get("endpoint")
        resolved_protocol = normalize_protocol_name(protocol or envelope.get("protocol") or channel or envelope.get("channel"))
        resolved_headers = normalize_headers(merge_mappings(envelope.get("headers"), headers))
        resolved_metadata = normalize_metadata(merge_mappings(envelope.get("metadata"), metadata))
        resolved_certificate_info = normalize_metadata(certificate_info)

        parsed_endpoint: Optional[ParsedEndpoint] = None
        normalized_endpoint: Optional[str] = None
        host: Optional[str] = None
        port: Optional[int] = None
        scheme: Optional[str] = None
        secure = False

        if resolved_endpoint is not None:
            endpoint_text = ensure_non_empty_string(str(resolved_endpoint), field_name="endpoint")
            if resolved_protocol == "queue" and "://" not in endpoint_text:
                normalized_endpoint = endpoint_text
                host = endpoint_text
                port = None
                scheme = None
                secure = False
            else:
                if "://" in endpoint_text:
                    parsed_endpoint = parse_endpoint(endpoint_text)
                    resolved_protocol = parsed_endpoint.protocol
                else:
                    parsed_endpoint = parse_endpoint(endpoint_text, protocol=resolved_protocol)
                normalized_endpoint = parsed_endpoint.normalized
                host = parsed_endpoint.host
                port = parsed_endpoint.port
                scheme = parsed_endpoint.scheme
                secure = parsed_endpoint.secure

        resolved_channel = normalize_channel_name(channel or envelope.get("channel") or resolved_protocol)

        effective_tls_required = self._resolve_tls_requirement(
            protocol=resolved_protocol,
            secure=secure,
            endpoint=normalized_endpoint or resolved_endpoint,
            explicit_tls_required=tls_required,
            constraints=constraints,
        )

        public_request = {
            "endpoint": normalized_endpoint or resolved_endpoint,
            "protocol": resolved_protocol,
            "channel": resolved_channel,
            "host": host,
            "port": port,
            "scheme": scheme,
            "secure": secure,
            "tls_required": effective_tls_required,
            "headers": sanitize_for_logging(resolved_headers) if self.sanitize_logs else resolved_headers,
            "metadata": sanitize_for_logging(resolved_metadata) if self.sanitize_logs else resolved_metadata,
            "rate_class": rate_class,
            "rate_limit_key": rate_limit_key or self._build_rate_limit_key(
                endpoint=normalized_endpoint or resolved_endpoint,
                channel=resolved_channel,
                protocol=resolved_protocol,
                metadata=resolved_metadata,
            ),
        }

        return {
            "envelope": envelope,
            "endpoint": normalized_endpoint or resolved_endpoint,
            "parsed_endpoint": parsed_endpoint,
            "host": host,
            "port": port,
            "scheme": scheme,
            "secure": secure,
            "protocol": resolved_protocol,
            "channel": resolved_channel,
            "headers": resolved_headers,
            "metadata": resolved_metadata,
            "constraints": constraints,
            "tls_required": effective_tls_required,
            "certificate_info": resolved_certificate_info,
            "rate_class": ensure_non_empty_string(str(rate_class), field_name="rate_class").lower(),
            "rate_limit_key": public_request["rate_limit_key"],
            "public_request": public_request,
        }

    # ------------------------------------------------------------------ #
    # Individual checks
    # ------------------------------------------------------------------ #
    def _check_destination_policy(self, request: Mapping[str, Any]) -> PolicyCheckResult:
        if not self.enforce_destination_policy:
            return PolicyCheckResult(
                name="destination_policy",
                allowed=True,
                decision="skipped",
                reason="Destination policy enforcement is disabled.",
            )

        endpoint = request.get("endpoint")
        host = request.get("host")
        constraints = request.get("constraints") or {}

        if endpoint is None and host is None:
            return PolicyCheckResult(
                name="destination_policy",
                allowed=True,
                decision="allowed",
                reason="No endpoint was supplied for destination evaluation.",
            )

        allow_private_hosts = self._resolve_bool_override("allow_private_hosts", constraints, self.allow_private_hosts)
        allow_loopback_hosts = self._resolve_bool_override("allow_loopback_hosts", constraints, self.allow_loopback_hosts)
        allow_ip_hosts = self._resolve_bool_override("allow_ip_hosts", constraints, self.allow_ip_hosts)

        denied_patterns = self._resolve_sequence_override(
            "denied_destination_patterns",
            constraints,
            self.denied_destination_patterns,
        )
        allowed_patterns = self._resolve_sequence_override(
            "allowed_destination_patterns",
            constraints,
            self.allowed_destination_patterns,
        )

        if self._matches_any_pattern(endpoint, host, denied_patterns):
            return self._deny(
                "destination_policy",
                DestinationDeniedError,
                "Destination matches a denied destination policy pattern.",
                request,
                details={"endpoint": endpoint, "host": host, "denied_patterns": sorted(denied_patterns)},
            )

        if host and is_ip_address(host) and not allow_ip_hosts:
            return self._deny(
                "destination_policy",
                DestinationDeniedError,
                "IP address destinations are not permitted by policy.",
                request,
                details={"endpoint": endpoint, "host": host},
            )

        if host and is_private_host(host):
            is_loopback = str(host).lower() in {"localhost", "localhost.localdomain", "127.0.0.1", "::1"}
            if is_loopback and not allow_loopback_hosts:
                return self._deny(
                    "destination_policy",
                    DestinationDeniedError,
                    "Loopback destinations are not permitted by policy.",
                    request,
                    details={"endpoint": endpoint, "host": host},
                )
            if not is_loopback and not allow_private_hosts:
                return self._deny(
                    "destination_policy",
                    DestinationDeniedError,
                    "Private-network destinations are not permitted by policy.",
                    request,
                    details={"endpoint": endpoint, "host": host},
                )

        if allowed_patterns and not self._matches_any_pattern(endpoint, host, allowed_patterns):
            return self._deny(
                "destination_policy",
                DestinationDeniedError,
                "Destination is not present in the allowed destination set.",
                request,
                details={"endpoint": endpoint, "host": host, "allowed_patterns": sorted(allowed_patterns)},
            )

        return PolicyCheckResult(
            name="destination_policy",
            allowed=True,
            decision="allowed",
            reason="Destination passed destination policy evaluation.",
            details={"endpoint": endpoint, "host": host},
        )

    def _check_protocol_policy(self, request: Mapping[str, Any]) -> PolicyCheckResult:
        if not self.enforce_protocol_policy:
            return PolicyCheckResult(
                name="protocol_policy",
                allowed=True,
                decision="skipped",
                reason="Protocol policy enforcement is disabled.",
            )

        protocol = ensure_non_empty_string(str(request["protocol"]), field_name="protocol").lower()
        channel = ensure_non_empty_string(str(request["channel"]), field_name="channel").lower()
        constraints = request.get("constraints") or {}

        allowed_protocols = self._resolve_sequence_override("allowed_protocols", constraints, self.allowed_protocols)
        denied_protocols = self._resolve_sequence_override("denied_protocols", constraints, self.denied_protocols)
        allowed_channels = self._resolve_sequence_override("allowed_channels", constraints, self.allowed_channels)
        denied_channels = self._resolve_sequence_override("denied_channels", constraints, self.denied_channels)

        if protocol in denied_protocols:
            return self._deny(
                "protocol_policy",
                ProtocolDeniedError,
                "Protocol is explicitly denied by policy.",
                request,
                details={"protocol": protocol, "denied_protocols": sorted(denied_protocols)},
            )
        if channel in denied_channels:
            return self._deny(
                "protocol_policy",
                ProtocolDeniedError,
                "Channel is explicitly denied by policy.",
                request,
                details={"channel": channel, "denied_channels": sorted(denied_channels)},
            )
        if allowed_protocols and protocol not in allowed_protocols:
            return self._deny(
                "protocol_policy",
                ProtocolDeniedError,
                "Protocol is not present in the allowed protocol set.",
                request,
                details={"protocol": protocol, "allowed_protocols": sorted(allowed_protocols)},
            )
        if allowed_channels and channel not in allowed_channels:
            return self._deny(
                "protocol_policy",
                ProtocolDeniedError,
                "Channel is not present in the allowed channel set.",
                request,
                details={"channel": channel, "allowed_channels": sorted(allowed_channels)},
            )

        return PolicyCheckResult(
            name="protocol_policy",
            allowed=True,
            decision="allowed",
            reason="Protocol and channel passed policy evaluation.",
            details={"protocol": protocol, "channel": channel},
        )

    def _check_port_policy(self, request: Mapping[str, Any]) -> PolicyCheckResult:
        if not self.enforce_port_policy:
            return PolicyCheckResult(
                name="port_policy",
                allowed=True,
                decision="skipped",
                reason="Port policy enforcement is disabled.",
            )

        port = request.get("port")
        if port is None:
            return PolicyCheckResult(
                name="port_policy",
                allowed=True,
                decision="allowed",
                reason="No transport port applies to this target.",
            )

        constraints = request.get("constraints") or {}
        allowed_ports = self._resolve_port_override("allowed_ports", constraints, self.allowed_ports)
        denied_ports = self._resolve_port_override("denied_ports", constraints, self.denied_ports)

        if port in denied_ports:
            return self._deny(
                "port_policy",
                PortDeniedError,
                "Port is explicitly denied by policy.",
                request,
                details={"port": port, "denied_ports": sorted(denied_ports)},
            )
        if allowed_ports and port not in allowed_ports:
            return self._deny(
                "port_policy",
                PortDeniedError,
                "Port is not present in the allowed port set.",
                request,
                details={"port": port, "allowed_ports": sorted(allowed_ports)},
            )

        return PolicyCheckResult(
            name="port_policy",
            allowed=True,
            decision="allowed",
            reason="Port passed policy evaluation.",
            details={"port": port},
        )

    def _check_tls_policy(self, request: Mapping[str, Any]) -> PolicyCheckResult:
        if not self.enforce_tls_policy:
            return PolicyCheckResult(
                name="tls_policy",
                allowed=True,
                decision="skipped",
                reason="TLS policy enforcement is disabled.",
            )

        endpoint = request.get("endpoint")
        if request.get("protocol") == "queue" and request.get("tls_required") is not True:
            return PolicyCheckResult(
                name="tls_policy",
                allowed=True,
                decision="allowed",
                reason="TLS is not required for the current queue-target policy context.",
            )

        if not request.get("tls_required"):
            return PolicyCheckResult(
                name="tls_policy",
                allowed=True,
                decision="allowed",
                reason="TLS is not required for this request.",
            )

        insecure_patterns = self._resolve_sequence_override(
            "insecure_destination_patterns",
            request.get("constraints") or {},
            self.insecure_destination_patterns,
        )
        if endpoint is not None and self._matches_any_pattern(endpoint, request.get("host"), insecure_patterns):
            return PolicyCheckResult(
                name="tls_policy",
                allowed=True,
                decision="allowed",
                reason="Destination is explicitly exempt from TLS requirements.",
                details={"endpoint": endpoint},
            )

        if not request.get("secure", False):
            return self._deny(
                "tls_policy",
                TLSRequiredError,
                "TLS is required for this request but the target is not configured as secure.",
                request,
                details={
                    "endpoint": endpoint,
                    "protocol": request.get("protocol"),
                    "scheme": request.get("scheme"),
                },
            )

        return PolicyCheckResult(
            name="tls_policy",
            allowed=True,
            decision="allowed",
            reason="TLS posture satisfied policy requirements.",
            details={"endpoint": endpoint, "scheme": request.get("scheme")},
        )

    def _check_certificate_policy(self, request: Mapping[str, Any]) -> PolicyCheckResult:
        if not self.enforce_certificate_policy:
            return PolicyCheckResult(
                name="certificate_policy",
                allowed=True,
                decision="skipped",
                reason="Certificate policy enforcement is disabled.",
            )

        if not request.get("tls_required") and not request.get("secure"):
            return PolicyCheckResult(
                name="certificate_policy",
                allowed=True,
                decision="allowed",
                reason="Certificate posture is not applicable to this request.",
            )

        cert_info = ensure_mapping(request.get("certificate_info"), field_name="certificate_info", allow_none=True)
        if not cert_info:
            if self.require_certificate_info_for_tls:
                return self._deny(
                    "certificate_policy",
                    CertificateValidationError,
                    "Certificate validation data is required for TLS policy decisions.",
                    request,
                    details={"endpoint": request.get("endpoint")},
                )
            return PolicyCheckResult(
                name="certificate_policy",
                allowed=True,
                decision="allowed",
                reason="No certificate metadata was provided; certificate policy did not find a violation.",
            )

        valid = cert_info.get("valid")
        expired = bool(cert_info.get("expired", False))
        self_signed = bool(cert_info.get("self_signed", False))
        hostname_match = cert_info.get("hostname_match", True)
        authority = cert_info.get("authority") or cert_info.get("issuer") or cert_info.get("ca")

        if valid is False:
            return self._deny(
                "certificate_policy",
                CertificateValidationError,
                "Certificate metadata indicates an invalid certificate.",
                request,
                details={"certificate_info": cert_info},
            )
        if self.deny_expired_certificates and expired:
            return self._deny(
                "certificate_policy",
                CertificateValidationError,
                "Expired certificates are not permitted by policy.",
                request,
                details={"certificate_info": cert_info},
            )
        if self.deny_self_signed_certificates and self_signed:
            return self._deny(
                "certificate_policy",
                CertificateValidationError,
                "Self-signed certificates are not permitted by policy.",
                request,
                details={"certificate_info": cert_info},
            )
        if self.require_hostname_match and hostname_match is False:
            return self._deny(
                "certificate_policy",
                CertificateValidationError,
                "Certificate hostname validation failed policy requirements.",
                request,
                details={"certificate_info": cert_info},
            )
        if self.allowed_certificate_authorities:
            normalized_authority = str(authority).strip().lower() if authority is not None else ""
            allowed_authorities = {item.lower() for item in self.allowed_certificate_authorities}
            if normalized_authority not in allowed_authorities:
                return self._deny(
                    "certificate_policy",
                    CertificateValidationError,
                    "Certificate authority is not present in the allowed authority set.",
                    request,
                    details={
                        "certificate_authority": authority,
                        "allowed_certificate_authorities": sorted(allowed_authorities),
                    },
                )

        return PolicyCheckResult(
            name="certificate_policy",
            allowed=True,
            decision="allowed",
            reason="Certificate posture satisfied policy requirements.",
            details={"certificate_info": cert_info},
        )

    def _check_rate_limit_policy(
        self,
        request: Mapping[str, Any],
        *,
        rate_class: str,
        rate_limit_key: str,
        consume: bool,
    ) -> PolicyCheckResult:
        if not self.enforce_rate_limits:
            return PolicyCheckResult(
                name="rate_limit_policy",
                allowed=True,
                decision="skipped",
                reason="Rate-limit enforcement is disabled.",
            )

        spec = self.rate_limits.get(rate_class) or self.rate_limits.get("default")
        if not spec:
            return PolicyCheckResult(
                name="rate_limit_policy",
                allowed=True,
                decision="allowed",
                reason="No rate-limit specification applies to this request.",
            )

        limit = int(spec["limit"])
        window_seconds = int(spec["window_seconds"])
        if limit <= 0:
            return PolicyCheckResult(
                name="rate_limit_policy",
                allowed=True,
                decision="allowed",
                reason="Rate-limit specification is effectively unlimited.",
                details={"rate_class": rate_class},
            )

        with self._lock:
            bucket = self._get_or_create_rate_bucket(
                key=rate_limit_key,
                rate_class=rate_class,
                limit=limit,
                window_seconds=window_seconds,
            )
            now = _utcnow()
            bucket.refresh(now)

            if bucket.would_exceed():
                self._stats["rate_limit_hits"] += 1
                return self._deny(
                    "rate_limit_policy",
                    RateLimitExceededError,
                    "Rate limit exceeded for the current request key.",
                    request,
                    details={
                        "rate_limit_key": rate_limit_key,
                        "rate_class": rate_class,
                        "limit": limit,
                        "window_seconds": window_seconds,
                        "count": bucket.count,
                        "remaining": bucket.remaining,
                    },
                )

            if consume:
                bucket.consume()

            return PolicyCheckResult(
                name="rate_limit_policy",
                allowed=True,
                decision="allowed",
                reason="Rate-limit policy allows the request.",
                details={
                    "rate_limit_key": rate_limit_key,
                    "rate_class": rate_class,
                    "limit": limit,
                    "window_seconds": window_seconds,
                    "count": bucket.count,
                    "remaining": bucket.remaining,
                },
            )

    # ------------------------------------------------------------------ #
    # Internal decision / memory helpers
    # ------------------------------------------------------------------ #
    def _record_decision(self, snapshot: Mapping[str, Any], *, ttl_seconds: Optional[int]) -> None:
        sanitized = sanitize_for_logging(snapshot) if self.sanitize_logs else json_safe(snapshot)
        with self._lock:
            self._decision_history.append(json_safe(sanitized))

        if not self.record_policy_decisions:
            return

        try:
            request = ensure_mapping(snapshot.get("request"), field_name="request", allow_none=True)
            self.memory.record_policy_decision(
                ensure_non_empty_string(str(snapshot.get("policy_name", "network_policy.final")), field_name="policy_name"),
                {
                    "decision_id": snapshot.get("decision_id"),
                    "allowed": bool(snapshot.get("allowed", False)),
                    "decision": snapshot.get("decision"),
                    "reason": snapshot.get("reason"),
                    "checks": snapshot.get("checks", []),
                    "violations": snapshot.get("violations", []),
                    "recorded_at": snapshot.get("recorded_at"),
                    "request": request,
                },
                endpoint=request.get("endpoint"),
                protocol=request.get("protocol"),
                channel=request.get("channel"),
                reason=str(snapshot.get("reason") or ""),
                ttl_seconds=ttl_seconds or self.decision_ttl_seconds,
                metadata=ensure_mapping(snapshot.get("metadata"), field_name="metadata", allow_none=True),
            )
        except NetworkError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalization is intentional here.
            raise normalize_network_exception(
                exc,
                operation="policy_record_decision",
                metadata={"policy_name": snapshot.get("policy_name")},
            ) from exc

    def _deny(
        self,
        policy_name: str,
        exc_type: type[NetworkError],
        message: str,
        request: Mapping[str, Any],
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> PolicyCheckResult:
        request_public = ensure_mapping(request.get("public_request"), field_name="public_request", allow_none=True)
        error = exc_type(
            message,
            context={
                "operation": "policy_evaluate",
                "endpoint": request_public.get("endpoint"),
                "protocol": request_public.get("protocol"),
                "channel": request_public.get("channel"),
                "policy_name": policy_name,
                "tls_required": request.get("tls_required"),
            },
            details=details,
        )

        if policy_name == "destination_policy":
            self._stats["destination_denials"] += 1
        elif policy_name == "protocol_policy":
            self._stats["protocol_denials"] += 1
        elif policy_name == "port_policy":
            self._stats["port_denials"] += 1
        elif policy_name == "tls_policy":
            self._stats["tls_denials"] += 1
        elif policy_name == "certificate_policy":
            self._stats["certificate_denials"] += 1

        return PolicyCheckResult(
            name=policy_name,
            allowed=False,
            decision="denied",
            reason=message,
            severity="warning",
            details=json_safe(details or {}),
            error_snapshot=error.to_memory_snapshot(),
            exception=error,
        )

    def _resolve_tls_requirement(
        self,
        *,
        protocol: str,
        secure: bool,
        endpoint: Optional[str],
        explicit_tls_required: Optional[bool],
        constraints: Mapping[str, Any],
    ) -> bool:
        if explicit_tls_required is not None:
            return bool(explicit_tls_required)
        if "tls_required" in constraints:
            return bool(constraints["tls_required"])

        insecure_exemptions = self._resolve_sequence_override(
            "insecure_destination_patterns",
            constraints,
            self.insecure_destination_patterns,
        )
        if endpoint is not None and self._matches_any_pattern(endpoint, None, insecure_exemptions):
            return False

        if protocol in {item.lower() for item in self.tls_required_protocols}:
            return True
        if secure:
            return True
        if protocol == "queue":
            return False
        return self.require_tls_by_default

    def _build_rate_limit_key(
        self,
        *,
        endpoint: Optional[str],
        channel: str,
        protocol: str,
        metadata: Mapping[str, Any],
    ) -> str:
        if metadata.get("rate_limit_key"):
            return ensure_non_empty_string(str(metadata["rate_limit_key"]), field_name="rate_limit_key")
        if endpoint:
            return f"{channel}:{endpoint}"
        return f"{channel}:{protocol}"

    def _get_or_create_rate_bucket(
        self,
        *,
        key: str,
        rate_class: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitBucket:
        normalized_key = ensure_non_empty_string(key, field_name="rate_limit_key")
        existing = self._rate_buckets.get(normalized_key)
        if existing is not None:
            existing.limit = limit
            existing.window_seconds = window_seconds
            self._rate_buckets.move_to_end(normalized_key)
            return existing

        if len(self._rate_buckets) >= self.max_rate_buckets:
            self._rate_buckets.popitem(last=False)

        bucket = RateLimitBucket(
            key=normalized_key,
            rate_class=rate_class,
            limit=limit,
            window_seconds=window_seconds,
            window_started_at=_utcnow(),
        )
        self._rate_buckets[normalized_key] = bucket
        return bucket

    # ------------------------------------------------------------------ #
    # Config helpers
    # ------------------------------------------------------------------ #
    def _get_bool_config(self, name: str, default: bool) -> bool:
        value = self.policy_config.get(name, default)
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
            "Invalid boolean value in network policy configuration.",
            context={"operation": "network_policy_config"},
            details={"config_key": name, "config_value": value},
        )

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        value = self.policy_config.get(name, default)
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in network policy configuration.",
                context={"operation": "network_policy_config"},
                details={"config_key": name, "config_value": value},
                cause=exc,
            ) from exc
        if coerced < 0:
            raise NetworkConfigurationError(
                "Configuration value must be non-negative.",
                context={"operation": "network_policy_config"},
                details={"config_key": name, "config_value": value},
            )
        return coerced

    def _get_sequence_config(self, name: str, default: Sequence[Any]) -> Set[str]:
        raw = self.policy_config.get(name, default)
        if raw is None:
            return {str(item).strip().lower() for item in default if str(item).strip()}
        if isinstance(raw, str):
            items = [part.strip() for part in raw.split(",")]
        else:
            items = [str(item).strip() for item in ensure_sequence(raw, field_name=name, coerce_scalar=True)]
        return {item.lower() for item in items if item}

    def _get_port_set_config(self, name: str, default: Sequence[int]) -> Set[int]:
        raw = self.policy_config.get(name, default)
        if raw is None:
            raw = default
        values = ensure_sequence(raw, field_name=name, allow_none=True, coerce_scalar=True)
        ports: Set[int] = set()
        for value in values:
            try:
                port = int(value)
            except (TypeError, ValueError) as exc:
                raise NetworkConfigurationError(
                    "Invalid port value in network policy configuration.",
                    context={"operation": "network_policy_config"},
                    details={"config_key": name, "config_value": value},
                    cause=exc,
                ) from exc
            if port < 1 or port > 65535:
                raise NetworkConfigurationError(
                    "Configured ports must be between 1 and 65535.",
                    context={"operation": "network_policy_config"},
                    details={"config_key": name, "config_value": value},
                )
            ports.add(port)
        return ports

    def _get_rate_limits_config(self, name: str, default: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
        raw = self.policy_config.get(name, default)
        source = ensure_mapping(raw, field_name=name)
        normalized: Dict[str, Dict[str, int]] = {}
        for rate_class, spec in source.items():
            spec_map = ensure_mapping(spec, field_name=f"{name}.{rate_class}")
            try:
                limit = int(spec_map.get("limit", 0))
                window_seconds = int(spec_map.get("window_seconds", self.default_rate_window_seconds))
            except (TypeError, ValueError) as exc:
                raise NetworkConfigurationError(
                    "Invalid rate-limit specification in network policy configuration.",
                    context={"operation": "network_policy_config"},
                    details={"config_key": f"{name}.{rate_class}", "config_value": spec_map},
                    cause=exc,
                ) from exc
            if limit < 0 or window_seconds <= 0:
                raise NetworkConfigurationError(
                    "Rate-limit configuration values must be positive.",
                    context={"operation": "network_policy_config"},
                    details={"config_key": f"{name}.{rate_class}", "config_value": spec_map},
                )
            normalized[str(rate_class).strip().lower()] = {
                "limit": limit,
                "window_seconds": window_seconds,
            }
        return normalized

    # ------------------------------------------------------------------ #
    # Override / matching helpers
    # ------------------------------------------------------------------ #
    def _resolve_bool_override(self, name: str, constraints: Mapping[str, Any], default: bool) -> bool:
        if name not in constraints:
            return default
        value = constraints[name]
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
        raise PayloadValidationError(
            "Constraint override must be boolean.",
            context={"operation": "policy_constraints"},
            details={"constraint": name, "value": value},
        )

    def _resolve_sequence_override(self, name: str, constraints: Mapping[str, Any], default: Set[str]) -> Set[str]:
        if name not in constraints:
            return set(default)
        raw = constraints[name]
        if raw is None:
            return set()
        if isinstance(raw, str):
            values = [part.strip() for part in raw.split(",")]
        else:
            values = [str(item).strip() for item in ensure_sequence(raw, field_name=name, allow_none=True, coerce_scalar=True)]
        return {value.lower() for value in values if value}

    def _resolve_port_override(self, name: str, constraints: Mapping[str, Any], default: Set[int]) -> Set[int]:
        if name not in constraints:
            return set(default)
        raw = constraints[name]
        if raw is None:
            return set()
        values = ensure_sequence(raw, field_name=name, allow_none=True, coerce_scalar=True)
        ports: Set[int] = set()
        for value in values:
            try:
                port = int(value)
            except (TypeError, ValueError) as exc:
                raise PayloadValidationError(
                    "Constraint port values must be integers.",
                    context={"operation": "policy_constraints"},
                    details={"constraint": name, "value": value},
                    cause=exc,
                ) from exc
            if port < 1 or port > 65535:
                raise PayloadValidationError(
                    "Constraint ports must be between 1 and 65535.",
                    context={"operation": "policy_constraints"},
                    details={"constraint": name, "value": value},
                )
            ports.add(port)
        return ports

    def _matches_any_pattern(
        self,
        endpoint: Optional[str],
        host: Optional[str],
        patterns: Sequence[str] | Set[str],
    ) -> bool:
        if not patterns:
            return False
        normalized_endpoint = str(endpoint).lower() if endpoint is not None else ""
        normalized_host = str(host).lower() if host is not None else ""
        for pattern in patterns:
            normalized_pattern = str(pattern).strip().lower()
            if not normalized_pattern:
                continue
            if normalized_endpoint and fnmatch.fnmatch(normalized_endpoint, normalized_pattern):
                return True
            if normalized_host and fnmatch.fnmatch(normalized_host, normalized_pattern):
                return True
        return False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    print("\n=== Running Network Policy ===\n")
    printer.status("TEST", "Network Policy initialized", "info")

    memory = NetworkMemory()
    policy = NetworkPolicy(memory=memory)

    allowed_snapshot = policy.evaluate(
        endpoint="https://api.example.com/v1/jobs",
        protocol="https",
        channel="http",
        metadata={"task_class": "relay"},
        certificate_info={
            "valid": True,
            "expired": False,
            "self_signed": False,
            "hostname_match": True,
            "authority": "Example CA",
        },
        rate_class="default",
    )
    printer.status("TEST", "Evaluated allowed HTTPS request", "info")

    denied_insecure_snapshot = policy.evaluate(
        endpoint="http://api.example.com/v1/jobs",
        protocol="http",
        channel="http",
        metadata={"task_class": "relay"},
        rate_class="default",
    )
    printer.status("TEST", "Evaluated denied insecure HTTP request", "info")

    try:
        policy.assert_allowed(
            endpoint="https://api.example.com:25/v1/jobs",
            protocol="https",
            channel="http",
            constraints={"allowed_ports": [443]},
            certificate_info={
                "valid": True,
                "expired": False,
                "self_signed": False,
                "hostname_match": True,
                "authority": "Example CA",
            },
        )
    except PortDeniedError as exc:
        port_denial = exc.to_memory_snapshot()
        printer.status("TEST", "Caught expected port denial", "warning")
    else:
        raise AssertionError("Expected PortDeniedError for denied port policy.")

    tight_policy = NetworkPolicy(
        memory=memory,
        config={"rate_limits": {"default": {"limit": 2, "window_seconds": 60}}},
    )
    tight_policy.assert_allowed(
        endpoint="https://api.example.com/v1/jobs",
        protocol="https",
        channel="http",
        certificate_info={
            "valid": True,
            "expired": False,
            "self_signed": False,
            "hostname_match": True,
            "authority": "Example CA",
        },
        rate_class="default",
        rate_limit_key="demo-rate-key",
    )
    tight_policy.assert_allowed(
        endpoint="https://api.example.com/v1/jobs",
        protocol="https",
        channel="http",
        certificate_info={
            "valid": True,
            "expired": False,
            "self_signed": False,
            "hostname_match": True,
            "authority": "Example CA",
        },
        rate_class="default",
        rate_limit_key="demo-rate-key",
    )

    try:
        tight_policy.assert_allowed(
            endpoint="https://api.example.com/v1/jobs",
            protocol="https",
            channel="http",
            certificate_info={
                "valid": True,
                "expired": False,
                "self_signed": False,
                "hostname_match": True,
                "authority": "Example CA",
            },
            rate_class="default",
            rate_limit_key="demo-rate-key",
        )
    except RateLimitExceededError as exc:
        rate_limit_denial = exc.to_memory_snapshot()
        printer.status("TEST", "Caught expected rate-limit denial", "warning")
    else:
        raise AssertionError("Expected RateLimitExceededError after consuming the configured limit.")

    preview_snapshot = policy.preview(
        endpoint="https://api.example.com/v1/jobs",
        protocol="https",
        channel="http",
        certificate_info={
            "valid": True,
            "expired": False,
            "self_signed": False,
            "hostname_match": True,
            "authority": "Example CA",
        },
        rate_limit_key="preview-rate-key",
    )

    state_snapshot = policy.get_policy_state()
    recent_decisions = policy.get_recent_decisions(limit=5)
    rate_state = tight_policy.get_rate_limit_state()

    print("Allowed Snapshot:", stable_json_dumps(allowed_snapshot))
    print("Denied Insecure Snapshot:", stable_json_dumps(denied_insecure_snapshot))
    print("Port Denial:", stable_json_dumps(port_denial))
    print("Rate Limit Denial:", stable_json_dumps(rate_limit_denial))
    print("Preview Snapshot:", stable_json_dumps(preview_snapshot))
    print("Policy State:", stable_json_dumps(state_snapshot))
    print("Recent Decisions:", stable_json_dumps(recent_decisions))
    print("Rate Limit State:", stable_json_dumps(rate_state))
    print("Memory Policy Snapshot:", stable_json_dumps(memory.get("network.policy.decision")))

    assert allowed_snapshot["allowed"] is True
    assert denied_insecure_snapshot["allowed"] is False
    assert denied_insecure_snapshot["violations"][0]["name"] == "tls_policy"
    assert port_denial["error_code"] == "PORT_DENIED"
    assert rate_limit_denial["error_code"] == "RATE_LIMIT_EXCEEDED"
    assert preview_snapshot["allowed"] is True
    assert rate_state["bucket_count"] >= 1
    assert memory.get("network.policy.decision") is not None

    printer.status("TEST", "All Network Policy checks passed", "info")
    print("\n=== Test ran successfully ===\n")
