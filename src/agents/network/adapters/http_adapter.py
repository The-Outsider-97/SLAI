"""
HTTP adapter implementation for SLAI's Network Agent.

This module provides the production-grade HTTP(S) transport adapter that
inherits from the shared BaseAdapter contract. It keeps protocol-specific HTTP
behavior in one place while relying on the base adapter for lifecycle
orchestration, structured memory updates, delivery bookkeeping, health
snapshots, and network-native error semantics.

The adapter is intentionally scoped to HTTP request/reply transport concerns:
- connection/session establishment for HTTP and HTTPS endpoints,
- request construction and transmission,
- response normalization and cached receive support,
- redirect handling,
- TLS posture configuration,
- response status classification and structured failure conversion,
- adapter-local request/response state useful to routing and observability.

It does not own routing strategy, retry policy, circuit breaking, or policy
arbitration. Those belong to the higher-level network modules and agents.
"""

from __future__ import annotations

import base64
import http.client
import json
import ssl

from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from time import sleep
from typing import Any, Deque, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urljoin, urlparse, urlencode

from ..utils import *
from .base_adapter import BaseAdapter
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("HTTP Adapter")
printer = PrettyPrinter()


_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
_DEFAULT_HTTP_SUCCESS_RANGE = (200, 299)


class HTTPAdapter(BaseAdapter):
    """
    Production-grade HTTP(S) transport adapter.

    The adapter supports synchronous request/reply flows, optional cached
    receive semantics, redirect following, configurable TLS posture, and
    structured response normalization suitable for the Network Agent.
    """

    DEFAULT_HTTP_CONTENT_TYPES: Tuple[str, ...] = (
        "application/json",
        "text/plain",
        "application/octet-stream",
    )
    DEFAULT_HTTP_AUTH_MODES: Tuple[str, ...] = ("none", "basic", "bearer", "custom_header")

    def __init__(
        self,
        *,
        memory=None,
        config: Optional[Mapping[str, Any]] = None,
        endpoint: Optional[str] = None,
        adapter_name: str = "HTTP",
        protocol: Optional[str] = None,
    ) -> None:
        provided_config = ensure_mapping(config, field_name="config", allow_none=True)
        section_config = get_config_section("network_http_adapter") or {}
        merged_http_config = merge_mappings(section_config, provided_config)

        inferred_protocol = protocol or merged_http_config.get("protocol") or self._infer_protocol_from_endpoint(endpoint) or "http"

        super().__init__(
            adapter_name=adapter_name,
            protocol=inferred_protocol,
            channel="http",
            memory=memory,
            config=merged_http_config,
            endpoint=endpoint or merged_http_config.get("endpoint"),
        )

        self.http_adapter_config = merge_mappings(section_config, self.adapter_config)

        self.default_method = self._get_http_method_config("default_method", "POST")
        self.receive_method = self._get_http_method_config("receive_method", "GET")
        self.healthcheck_method = self._get_http_method_config("healthcheck_method", "HEAD")
        self.receive_path = self._get_optional_string_config("receive_path")
        self.healthcheck_path = self._get_optional_string_config("healthcheck_path") or "/"
        self.user_agent = self._get_optional_string_config("user_agent") or "SLAI-NetworkAgent/1.0"
        self.default_accept = self._get_optional_string_config("default_accept") or "application/json, text/plain, */*"

        self.verify_tls = self._get_bool_config("verify_tls", True)
        self.allow_insecure_tls = self._get_bool_config("allow_insecure_tls", False)
        self.follow_redirects = self._get_bool_config("follow_redirects", True)
        self.raise_for_status = self._get_bool_config("raise_for_status", True)
        self.strict_response_deserialization = self._get_bool_config("strict_response_deserialization", False)
        self.healthcheck_on_connect = self._get_bool_config("healthcheck_on_connect", False)
        self.persistent_connections = self._get_bool_config("persistent_connections", True)
        self.cache_responses_for_recv = self._get_bool_config("cache_responses_for_recv", True)
        self.consume_cached_responses_on_recv = self._get_bool_config("consume_cached_responses_on_recv", True)
        self.use_cookie_jar = self._get_bool_config("use_cookie_jar", True)
        self.allow_redirect_downgrade = self._get_bool_config("allow_redirect_downgrade", False)
        self.allow_body_on_get = self._get_bool_config("allow_body_on_get", False)

        self.max_redirects = max(0, self._get_non_negative_int_config("max_redirects", 3))
        self.max_response_history_size = max(1, self._get_non_negative_int_config("max_response_history_size", 100))
        self.max_stored_response_body_bytes = max(1024, self._get_non_negative_int_config("max_stored_response_body_bytes", 262144))
        self.success_status_min = self._get_non_negative_int_config("success_status_min", _DEFAULT_HTTP_SUCCESS_RANGE[0])
        self.success_status_max = self._get_non_negative_int_config("success_status_max", _DEFAULT_HTTP_SUCCESS_RANGE[1])

        self.ca_file = self._get_optional_string_config("ca_file")
        self.cert_file = self._get_optional_string_config("cert_file")
        self.key_file = self._get_optional_string_config("key_file")
        self.tls_minimum_version = self._get_optional_string_config("tls_minimum_version") or "TLSv1_2"

        self.default_headers = normalize_headers(
            ensure_mapping(self.http_adapter_config.get("default_headers"), field_name="default_headers", allow_none=True),
            lowercase=False,
        )
        self.default_query_params = ensure_mapping(
            self.http_adapter_config.get("default_query_params"),
            field_name="default_query_params",
            allow_none=True,
        )

        self._connection: Optional[http.client.HTTPConnection] = None
        self._parsed_endpoint: Optional[ParsedEndpoint] = None
        self._request_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_response_history_size)
        self._response_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_response_history_size)
        self._receive_queue: Deque[Dict[str, Any]] = deque(maxlen=self.max_response_history_size)
        self._cookie_jar: Dict[str, str] = {}
        self._last_request: Optional[Dict[str, Any]] = None
        self._last_response: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # BaseAdapter protocol hooks
    # ------------------------------------------------------------------
    def _connect_impl(self, *, endpoint: str, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        parsed = parse_endpoint(endpoint, default_scheme=self.protocol, protocol=self.protocol, require_host=True)
        if parsed.scheme not in {"http", "https"}:
            raise ProtocolNegotiationError(
                "HTTPAdapter only supports http:// and https:// endpoints.",
                context={"operation": "connect", "protocol": parsed.scheme, "endpoint": parsed.normalized},
            )

        self._update_protocol_from_endpoint(parsed)
        self._reset_connection()
        self._connection = self._create_connection(parsed, timeout_ms=timeout_ms)
        self._connection.connect()
        self._parsed_endpoint = parsed

        if self.healthcheck_on_connect:
            health_result = self._perform_request(
                method=self.healthcheck_method,
                target_endpoint=parsed,
                request_path=self.healthcheck_path,
                timeout_ms=timeout_ms,
                headers={},
                body=None,
                metadata=metadata,
                cache_response=False,
                is_receive=False,
            )
            self._maybe_raise_for_status(health_result["response"], operation="connect", endpoint=parsed.normalized)

        return {
            "endpoint": parsed.normalized,
            "host": parsed.host,
            "port": parsed.port,
            "scheme": parsed.scheme,
            "secure": parsed.secure,
            "persistent_connections": self.persistent_connections,
            "verify_tls": self.verify_tls,
            "session_id": self.session.session_id,
            "healthcheck_on_connect": self.healthcheck_on_connect,
            "metadata": normalize_metadata(metadata),
        }

    def _send_impl(
        self,
        *,
        payload: bytes,
        envelope: Mapping[str, Any],
        timeout_ms: int,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        target_endpoint = self._resolve_request_endpoint(envelope, metadata)
        request_method = self._resolve_request_method(envelope, metadata)
        request_path = self._resolve_request_path(target_endpoint, envelope, metadata)
        headers = self._build_request_headers(envelope, metadata)
        body = self._resolve_request_body(request_method, payload, envelope)

        result = self._perform_request(
            method=request_method,
            target_endpoint=target_endpoint,
            request_path=request_path,
            timeout_ms=timeout_ms,
            headers=headers,
            body=body,
            metadata=metadata,
            envelope=envelope,
            cache_response=self.cache_responses_for_recv,
            is_receive=False,
        )

        self._maybe_raise_for_status(
            result["response"],
            operation="send",
            endpoint=target_endpoint.normalized,
            correlation_id=str(envelope.get("correlation_id") or "") or None,
            message_id=str(envelope.get("message_id") or "") or None,
        )
        return result

    def _receive_impl(self, *, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        force_poll = bool(metadata.get("force_poll"))

        if self.cache_responses_for_recv and self._receive_queue and not force_poll:
            response_snapshot = self._receive_queue.popleft() if self.consume_cached_responses_on_recv else self._receive_queue[-1]
            return self._response_to_inbound_message(response_snapshot)

        if self.receive_path:
            target_endpoint = self._resolve_request_endpoint({}, metadata)
            headers = self._build_request_headers({}, metadata)
            result = self._perform_request(
                method=self.receive_method,
                target_endpoint=target_endpoint,
                request_path=self.receive_path,
                timeout_ms=timeout_ms,
                headers=headers,
                body=None,
                metadata=metadata,
                cache_response=False,
                is_receive=True,
            )
            self._maybe_raise_for_status(result["response"], operation="receive", endpoint=target_endpoint.normalized)
            return self._response_to_inbound_message(result["response"])

        raise ReceiveFailureError(
            "No HTTP response is available for receive().",
            context={
                "operation": "receive",
                "channel": self.channel,
                "protocol": self.protocol,
                "endpoint": self.session.endpoint,
                "session_id": self.session.session_id,
            },
            details={
                "cache_responses_for_recv": self.cache_responses_for_recv,
                "queued_responses": len(self._receive_queue),
                "receive_path": self.receive_path,
            },
        )

    def _ack_impl(
        self,
        *,
        message_id: str,
        correlation_id: Optional[str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        return {
            "mode": "synthetic_noop",
            "transport_ack": False,
            "message_id": message_id,
            "correlation_id": correlation_id,
            "metadata": normalize_metadata(metadata),
        }

    def _nack_impl(
        self,
        *,
        message_id: str,
        correlation_id: Optional[str],
        reason: Optional[str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        return {
            "mode": "synthetic_noop",
            "transport_nack": False,
            "message_id": message_id,
            "correlation_id": correlation_id,
            "reason": reason,
            "metadata": normalize_metadata(metadata),
        }

    def _close_impl(self, *, reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        self._reset_connection()
        return {
            "closed": True,
            "reason": reason,
            "session_id": self.session.session_id,
            "metadata": normalize_metadata(metadata),
        }

    # ------------------------------------------------------------------
    # HTTP-specific state
    # ------------------------------------------------------------------
    def get_state_snapshot(self) -> Dict[str, Any]:
        return merge_mappings(
            super().get_state_snapshot(),
            {
                "http": {
                    "default_method": self.default_method,
                    "receive_method": self.receive_method,
                    "receive_path": self.receive_path,
                    "healthcheck_path": self.healthcheck_path,
                    "verify_tls": self.verify_tls,
                    "follow_redirects": self.follow_redirects,
                    "raise_for_status": self.raise_for_status,
                    "cached_receive_queue_size": len(self._receive_queue),
                    "request_history_size": len(self._request_history),
                    "response_history_size": len(self._response_history),
                    "cookie_names": sorted(self._cookie_jar.keys()),
                    "last_request": sanitize_for_logging(self._last_request),
                    "last_response": sanitize_for_logging(self._last_response),
                }
            },
        )

    def get_last_response(self) -> Optional[Dict[str, Any]]:
        return json_safe(self._last_response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_protocol_from_endpoint(endpoint: Optional[str]) -> Optional[str]:
        if not endpoint or "://" not in endpoint:
            return None
        parsed = urlparse(endpoint)
        if parsed.scheme:
            return parsed.scheme.lower()
        return None

    def _get_http_method_config(self, name: str, default: str) -> str:
        candidate = self.http_adapter_config.get(name, default)
        return ensure_non_empty_string(str(candidate), field_name=name).upper()

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.http_adapter_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _update_protocol_from_endpoint(self, parsed: ParsedEndpoint) -> None:
        self.protocol = parsed.scheme
        self.channel = "http"
        self.session.protocol = parsed.scheme
        self.session.channel = "http"
        self.health.protocol = parsed.scheme
        self.health.channel = "http"

    def _create_connection(self, parsed: ParsedEndpoint, *, timeout_ms: int) -> http.client.HTTPConnection:
        timeout_seconds = max(0.001, float(timeout_ms) / 1000.0)
        if parsed.scheme == "https":
            context = self._build_ssl_context()
            return http.client.HTTPSConnection(parsed.host, parsed.port, timeout=timeout_seconds, context=context)
        return http.client.HTTPConnection(parsed.host, parsed.port, timeout=timeout_seconds)

    def _build_ssl_context(self) -> ssl.SSLContext:
        if self.allow_insecure_tls or not self.verify_tls:
            context = ssl._create_unverified_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        else:
            context = ssl.create_default_context(cafile=self.ca_file or None)
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED

        minimum_version = self._resolve_tls_version(self.tls_minimum_version)
        if minimum_version is not None and hasattr(context, "minimum_version"):
            context.minimum_version = minimum_version

        if self.cert_file:
            context.load_cert_chain(certfile=self.cert_file, keyfile=self.key_file or None)
        return context

    def _resolve_tls_version(self, value: Optional[str]) -> Optional[ssl.TLSVersion]:
        if not value:
            return None
        normalized = str(value).strip().upper().replace(".", "_")
        mapping = {
            "TLSV1": ssl.TLSVersion.TLSv1,
            "TLSV1_1": ssl.TLSVersion.TLSv1_1,
            "TLSV1_2": ssl.TLSVersion.TLSv1_2,
        }
        if hasattr(ssl.TLSVersion, "TLSv1_3"):
            mapping["TLSV1_3"] = ssl.TLSVersion.TLSv1_3
        if normalized not in mapping:
            raise NetworkConfigurationError(
                "Unsupported TLS minimum version configured for HTTP adapter.",
                context={"operation": "http_adapter_config", "protocol": self.protocol},
                details={"tls_minimum_version": value},
            )
        return mapping[normalized]

    def _resolve_request_endpoint(self, envelope: Mapping[str, Any], metadata: Mapping[str, Any]) -> ParsedEndpoint:
        endpoint_candidate = (
            envelope.get("endpoint")
            or metadata.get("endpoint")
            or self.session.endpoint
            or self.http_adapter_config.get("endpoint")
        )
        if endpoint_candidate is None:
            raise AdapterInitializationError(
                "HTTP request requires a target endpoint but none is configured.",
                context={"operation": "send", "channel": self.channel, "protocol": self.protocol},
            )
        parsed = parse_endpoint(str(endpoint_candidate), default_scheme=self.protocol, protocol=self.protocol, require_host=True)
        if parsed.scheme not in {"http", "https"}:
            raise ProtocolNegotiationError(
                "HTTPAdapter only supports http:// and https:// endpoints.",
                context={"operation": "send", "protocol": parsed.scheme, "endpoint": parsed.normalized},
            )
        return parsed

    def _resolve_request_method(self, envelope: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
        method = envelope.get("method") or metadata.get("method") or self.default_method
        return ensure_non_empty_string(str(method), field_name="http_method").upper()

    def _resolve_request_path(self, target_endpoint: ParsedEndpoint, envelope: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
        path_candidate = (
            envelope.get("path")
            or metadata.get("path")
            or target_endpoint.path
            or "/"
        )
        path_text = ensure_non_empty_string(str(path_candidate), field_name="request_path") if str(path_candidate).strip() else "/"
        if not path_text.startswith("/"):
            path_text = f"/{path_text}"

        query_values = merge_mappings(
            parse_qs(target_endpoint.query, keep_blank_values=True),
            self.default_query_params,
            ensure_mapping(envelope.get("query_params"), field_name="query_params", allow_none=True),
            ensure_mapping(metadata.get("query_params"), field_name="query_params", allow_none=True),
        )
        if query_values:
            query_string = urlencode(query_values, doseq=True)
            return f"{path_text}?{query_string}"
        return path_text

    def _resolve_request_body(self, method: str, payload: bytes, envelope: Mapping[str, Any]) -> Optional[bytes]:
        if method in {"GET", "HEAD"} and not self.allow_body_on_get and len(payload) == 0:
            return None
        if method in {"GET", "HEAD"} and not self.allow_body_on_get:
            return None
        return payload

    def _build_request_headers(self, envelope: Mapping[str, Any], metadata: Mapping[str, Any]) -> Dict[str, str]:
        headers = normalize_headers(
            merge_mappings(
                self.default_headers,
                ensure_mapping(envelope.get("headers"), field_name="headers", allow_none=True),
                ensure_mapping(metadata.get("headers"), field_name="headers", allow_none=True),
            ),
            lowercase=False,
        )
        headers.setdefault("User-Agent", self.user_agent)
        headers.setdefault("Accept", self.default_accept)

        content_type = envelope.get("content_type")
        if content_type and "Content-Type" not in headers and "content-type" not in {key.lower() for key in headers}:
            headers["Content-Type"] = str(content_type)

        if envelope.get("correlation_id"):
            headers.setdefault("X-Correlation-ID", str(envelope["correlation_id"]))
        if envelope.get("idempotency_key"):
            headers.setdefault("Idempotency-Key", str(envelope["idempotency_key"]))
        if envelope.get("message_id"):
            headers.setdefault("X-Message-ID", str(envelope["message_id"]))

        auth_mode = str(metadata.get("auth_mode") or self.http_adapter_config.get("auth_mode") or "none").strip().lower()
        if auth_mode == "basic":
            username = metadata.get("username") or self.http_adapter_config.get("username")
            password = metadata.get("password") or self.http_adapter_config.get("password")
            if username is None or password is None:
                raise AuthenticationFailedError(
                    "HTTP basic authentication requires username and password.",
                    context={"operation": "send", "channel": self.channel, "protocol": self.protocol},
                )
            token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        elif auth_mode == "bearer":
            token = metadata.get("token") or self.http_adapter_config.get("token")
            if token is None:
                raise AuthenticationFailedError(
                    "HTTP bearer authentication requires a token.",
                    context={"operation": "send", "channel": self.channel, "protocol": self.protocol},
                )
            headers["Authorization"] = f"Bearer {token}"
        elif auth_mode == "custom_header":
            auth_header_name = str(metadata.get("auth_header_name") or self.http_adapter_config.get("auth_header_name") or "Authorization")
            auth_header_value = metadata.get("auth_header_value") or self.http_adapter_config.get("auth_header_value")
            if auth_header_value is None:
                raise AuthenticationFailedError(
                    "Custom header authentication requires auth_header_value.",
                    context={"operation": "send", "channel": self.channel, "protocol": self.protocol},
                )
            headers[auth_header_name] = str(auth_header_value)

        if self.use_cookie_jar and self._cookie_jar and "Cookie" not in headers and "cookie" not in {key.lower() for key in headers}:
            headers["Cookie"] = "; ".join(f"{name}={value}" for name, value in sorted(self._cookie_jar.items()))
        return headers

    def _ensure_connection_for_target(self, target_endpoint: ParsedEndpoint, *, timeout_ms: int) -> http.client.HTTPConnection:
        """Return an active HTTPConnection, creating one if necessary."""
        connection_needs_refresh = (
            self._connection is None
            or self._parsed_endpoint is None
            or self._parsed_endpoint.host != target_endpoint.host
            or self._parsed_endpoint.port != target_endpoint.port
            or self._parsed_endpoint.scheme != target_endpoint.scheme
            or not self.persistent_connections
        )
        if connection_needs_refresh:
            self._reset_connection()
            self._update_protocol_from_endpoint(target_endpoint)
            self._connection = self._create_connection(target_endpoint, timeout_ms=timeout_ms)
            self._connection.connect()
            self._parsed_endpoint = target_endpoint
            self.session.endpoint = target_endpoint.normalized
            self.health.endpoint = target_endpoint.normalized
        # At this point self._connection is guaranteed to be non-None
        if self._connection is None:
            raise NetworkConnectionError(
                "Failed to establish HTTP connection.",
                context={"operation": "ensure_connection", "endpoint": target_endpoint.normalized},
            )
        # Update timeout on existing connection (if supported)
        if hasattr(self._connection, "timeout"):
            self._connection.timeout = max(0.001, float(timeout_ms) / 1000.0)
        return self._connection

    def _perform_request(self, *, method: str, target_endpoint: ParsedEndpoint, request_path: str,
                         timeout_ms: int, headers: Mapping[str, Any], body: Optional[bytes],
                         metadata: Mapping[str, Any], envelope: Optional[Mapping[str, Any]] = None,
                         redirect_count: int = 0, redirect_chain: Optional[Sequence[Mapping[str, Any]]] = None,
                         cache_response: bool = True, is_receive: bool = False) -> Dict[str, Any]:
        # Normalize headers and remove Host header so the connection can set it correctly
        normalized_headers = normalize_headers(headers, lowercase=False)
        # Remove any existing Host header – the connection will add the correct one
        normalized_headers = {k: v for k, v in normalized_headers.items() if k.lower() != "host"}

        connection = self._ensure_connection_for_target(target_endpoint, timeout_ms=timeout_ms)
        redirect_chain_list = list(redirect_chain or [])

        request_snapshot = {
            "adapter_name": self.adapter_name,
            "protocol": target_endpoint.scheme,
            "channel": self.channel,
            "endpoint": target_endpoint.normalized,
            "url": f"{target_endpoint.scheme}://{target_endpoint.host}:{target_endpoint.port}{request_path}",
            "method": method,
            "path": request_path,
            "headers": sanitize_for_logging(normalized_headers),
            "payload_size": len(body) if body is not None else 0,
            "timeout_ms": timeout_ms,
            "redirect_count": redirect_count,
            "created_at": utc_timestamp(),
            "metadata": sanitize_for_logging(metadata),
        }

        try:
            connection.request(method, request_path, body=body, headers=normalized_headers)
            response = connection.getresponse()
            raw_body = response.read()
            response_headers = self._normalize_response_headers(response.getheaders())
            retry_after_ms = extract_retry_after_ms(response_headers)
            content_type = self._normalize_response_content_type(response_headers.get("content-type"))
            parsed_body, parsing_issue = self._deserialize_response_body(raw_body, content_type)
            body_payload, body_truncated = self._prepare_body_for_storage(raw_body, parsed_body, content_type)

            if self.use_cookie_jar:
                self._update_cookie_jar(response.getheaders())

            response_snapshot = {
                "adapter_name": self.adapter_name,
                "protocol": target_endpoint.scheme,
                "channel": self.channel,
                "endpoint": target_endpoint.normalized,
                "url": request_snapshot["url"],
                "method": method,
                "status_code": int(response.status),
                "reason": str(response.reason or HTTPStatus(response.status).phrase if response.status in HTTPStatus._value2member_map_ else response.reason or ""),
                "ok": self._is_success_status(int(response.status)),
                "headers": response_headers,
                "content_type": content_type,
                "payload": body_payload,
                "payload_size": len(raw_body),
                "body_truncated": body_truncated,
                "retry_after_ms": retry_after_ms,
                "redirect_count": redirect_count,
                "redirect_chain": redirect_chain_list,
                "received_at": utc_timestamp(),
                "metadata": sanitize_for_logging(metadata),
            }
            if parsing_issue is not None:
                response_snapshot["parsing_issue"] = parsing_issue

            self._last_request = request_snapshot
            self._last_response = response_snapshot
            self._request_history.append(json_safe(request_snapshot))
            self._response_history.append(json_safe(response_snapshot))

            if cache_response:
                self._receive_queue.append(json_safe(response_snapshot))

            location = response_headers.get("location")
            if (
                self.follow_redirects
                and location
                and int(response.status) in _REDIRECT_STATUS_CODES
                and redirect_count < self.max_redirects
            ):
                redirected_endpoint, redirected_path = self._resolve_redirect_target(
                    current_endpoint=target_endpoint,
                    current_path=request_path,
                    location=location,
                )
                if redirected_endpoint.scheme == "http" and target_endpoint.scheme == "https" and not self.allow_redirect_downgrade:
                    raise TLSRequiredError(
                        "Refusing HTTPS-to-HTTP redirect because downgrade is disabled.",
                        context={
                            "operation": "send",
                            "endpoint": redirected_endpoint.normalized,
                            "protocol": redirected_endpoint.scheme,
                        },
                        details={"location": location},
                    )
                redirect_chain_list.append(
                    {
                        "status_code": int(response.status),
                        "location": location,
                        "from": request_snapshot["url"],
                        "to": f"{redirected_endpoint.scheme}://{redirected_endpoint.host}:{redirected_endpoint.port}{redirected_path}",
                    }
                )
                redirected_method = "GET" if int(response.status) == 303 else method
                redirected_body = None if redirected_method == "GET" else body
                return self._perform_request(
                    method=redirected_method,
                    target_endpoint=redirected_endpoint,
                    request_path=redirected_path,
                    timeout_ms=timeout_ms,
                    headers=normalized_headers,
                    body=redirected_body,
                    metadata=metadata,
                    envelope=envelope,
                    redirect_count=redirect_count + 1,
                    redirect_chain=redirect_chain_list,
                    cache_response=cache_response,
                    is_receive=is_receive,
                )

            return {"request": request_snapshot, "response": response_snapshot}
        except NetworkError:
            raise
        except Exception as exc:
            raise normalize_network_exception(
                exc,
                operation="receive" if is_receive else "send",
                endpoint=target_endpoint.normalized,
                channel=self.channel,
                protocol=target_endpoint.scheme,
                timeout_ms=timeout_ms,
                session_id=self.session.session_id,
                metadata={
                    "method": method,
                    "path": request_path,
                    "redirect_count": redirect_count,
                    **normalize_metadata(metadata),
                },
            ) from exc

    def _resolve_redirect_target(self, *, current_endpoint: ParsedEndpoint, current_path: str, location: str) -> Tuple[ParsedEndpoint, str]:
        base_url = f"{current_endpoint.scheme}://{current_endpoint.host}:{current_endpoint.port}{current_path}"
        absolute = urljoin(base_url, location)
        redirected_endpoint = parse_endpoint(absolute, default_scheme=current_endpoint.scheme, protocol=current_endpoint.scheme, require_host=True)
        redirected_path = redirected_endpoint.path or "/"
        if redirected_endpoint.query:
            redirected_path = f"{redirected_path}?{redirected_endpoint.query}"
        return redirected_endpoint, redirected_path

    def _normalize_response_headers(self, header_items: Sequence[Tuple[str, str]]) -> Dict[str, str]:
        merged: Dict[str, Any] = {}
        for key, value in header_items:
            key_text = str(key)
            if key_text in merged:
                existing = merged[key_text]
                if isinstance(existing, list):
                    existing.append(value)
                else:
                    merged[key_text] = [existing, value]
            else:
                merged[key_text] = value
        return normalize_headers(merged)

    def _normalize_response_content_type(self, content_type: Optional[str]) -> str:
        if not content_type:
            return "application/octet-stream"
        return infer_content_type(b"", explicit_content_type=str(content_type).split(";", 1)[0].strip())

    def _deserialize_response_body(self, raw_body: bytes, content_type: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
        if len(raw_body) == 0:
            return None, None
        try:
            return deserialize_payload(raw_body, content_type=content_type), None
        except NetworkError as exc:
            if self.strict_response_deserialization:
                raise
            fallback: Any
            if content_type == "application/octet-stream":
                fallback = raw_body
            else:
                try:
                    fallback = raw_body.decode("utf-8")
                except UnicodeDecodeError:
                    fallback = raw_body
            return fallback, exc.to_memory_snapshot() if isinstance(exc, NetworkError) else build_error_snapshot(exc, operation="deserialize")

    def _prepare_body_for_storage(self, raw_body: bytes, parsed_body: Any, content_type: str) -> Tuple[Any, bool]:
        if len(raw_body) <= self.max_stored_response_body_bytes:
            return parsed_body, False
        if isinstance(parsed_body, str):
            truncated = parsed_body[: self.max_stored_response_body_bytes]
        elif isinstance(parsed_body, bytes):
            truncated = parsed_body[: self.max_stored_response_body_bytes]
        else:
            truncated = {
                "notice": "response body truncated for adapter snapshot storage",
                "content_type": content_type,
                "payload_size": len(raw_body),
            }
        return truncated, True

    def _maybe_raise_for_status(
        self,
        response_snapshot: Mapping[str, Any],
        *,
        operation: str,
        endpoint: str,
        correlation_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> None:
        status_code = int(response_snapshot.get("status_code") or 0)
        if not self.raise_for_status or self._is_success_status(status_code):
            return
        raise network_error_from_http_status(
            status_code,
            message=f"HTTP request failed with status {status_code}.",
            endpoint=endpoint,
            operation=operation,
            channel=self.channel,
            protocol=self.protocol,
            retry_after_ms=response_snapshot.get("retry_after_ms"),
            details={
                "message_id": message_id,
                "correlation_id": correlation_id,
                "reason": response_snapshot.get("reason"),
                "headers": sanitize_for_logging(response_snapshot.get("headers")),
                "payload": sanitize_for_logging(response_snapshot.get("payload")),
            },
        )

    def _is_success_status(self, status_code: int) -> bool:
        return self.success_status_min <= int(status_code) <= self.success_status_max

    def _response_to_inbound_message(self, response_snapshot: Mapping[str, Any]) -> Dict[str, Any]:
        headers = ensure_mapping(response_snapshot.get("headers"), field_name="headers", allow_none=True)
        metadata = normalize_metadata(
            merge_mappings(
                response_snapshot.get("metadata"),
                {
                    "status_code": response_snapshot.get("status_code"),
                    "reason": response_snapshot.get("reason"),
                    "url": response_snapshot.get("url"),
                    "method": response_snapshot.get("method"),
                    "retry_after_ms": response_snapshot.get("retry_after_ms"),
                },
            )
        )
        payload = response_snapshot.get("payload")
        message_id = str(response_snapshot.get("response_message_id") or response_snapshot.get("message_id") or generate_message_id(prefix=f"recv_{self.adapter_name.lower()}"))
        correlation_id = response_snapshot.get("correlation_id")
        return {
            "message_id": message_id,
            "correlation_id": correlation_id,
            "payload": payload,
            "content_type": response_snapshot.get("content_type"),
            "headers": headers,
            "metadata": metadata,
        }

    def _update_cookie_jar(self, header_items: Sequence[Tuple[str, str]]) -> None:
        for key, value in header_items:
            if str(key).lower() != "set-cookie":
                continue
            cookie_pair = str(value).split(";", 1)[0].strip()
            if not cookie_pair or "=" not in cookie_pair:
                continue
            name, cookie_value = cookie_pair.split("=", 1)
            self._cookie_jar[name.strip()] = cookie_value.strip()

    def _reset_connection(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
        self._connection = None
        self._parsed_endpoint = None


class _HTTPAdapterTestHandler(BaseHTTPRequestHandler):
    server_version = "SLAIHTTPAdapterTest/1.0"

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        return self.rfile.read(length) if length > 0 else b""

    def _write_json(self, status: int, payload: Mapping[str, Any], *, headers: Optional[Mapping[str, str]] = None) -> None:
        body = stable_json_dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if headers:
            for key, value in headers.items():
                self.send_header(str(key), str(value))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self) -> None:
        if self.path.startswith("/health"):
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            self._write_json(200, {"ok": True, "path": self.path})
            return
        if self.path.startswith("/poll"):
            self._write_json(200, {"event": "polled", "path": self.path})
            return
        if self.path.startswith("/redirect"):
            self.send_response(302)
            self.send_header("Location", "/health")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.startswith("/error"):
            self._write_json(503, {"error": "temporarily unavailable"}, headers={"Retry-After": "1"})
            return
        self._write_json(200, {"method": "GET", "path": self.path})

    def do_POST(self) -> None:
        body = self._read_body()
        try:
            decoded = json.loads(body.decode("utf-8")) if body else None
        except Exception:
            decoded = body.decode("utf-8", errors="replace")
        self._write_json(200, {"method": "POST", "path": self.path, "body": decoded})

    def log_message(self, format: str, *args: Any) -> None:
        return


def _printer_status(label: str, message: str, level: str = "info") -> None:
    try:
        if hasattr(printer, "status"):
            printer.status(label, message, level)
        else:
            print(f"[{label}] {message}")
    except Exception:
        print(f"[{label}] {message}")


if __name__ == "__main__":
    print("\n=== Running HTTP Adapter ===\n")
    _printer_status("TEST", "HTTP Adapter initialized", "info")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _HTTPAdapterTestHandler)
    host, port = server.server_address
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sleep(0.05)

    base_url = f"http://{host}:{port}"
    adapter = HTTPAdapter(
        endpoint=base_url,
        config={
            "default_method": "POST",
            "receive_path": "/poll",
            "healthcheck_on_connect": True,
            "healthcheck_path": "/health",
            "raise_for_status": True,
            "follow_redirects": True,
            "cache_responses_for_recv": True,
            "consume_cached_responses_on_recv": True,
            "supports_ack": False,
            "supports_nack": False,
            "supports_receive": True,
            "supports_request_reply": True,
        },
    )

    connect_snapshot = adapter.connect()
    _printer_status("TEST", "HTTP connection established", "info")

    send_snapshot = adapter.send(
        {"hello": "world"},
        envelope={
            "path": "/echo",
            "headers": {"X-Test": "true"},
            "method": "POST",
        },
        metadata={"request_id": "demo_http_send"},
    )
    _printer_status("TEST", "HTTP request sent", "info")

    cached_receive_snapshot = adapter.recv()
    _printer_status("TEST", "Cached HTTP response received", "info")

    polled_receive_snapshot = adapter.recv(metadata={"force_poll": True})
    _printer_status("TEST", "Polled HTTP response received", "info")

    redirect_snapshot = adapter.send(
        b"",
        envelope={"path": "/redirect", "method": "GET", "content_type": "application/octet-stream"},
    )
    _printer_status("TEST", "Redirect handling verified", "info")

    ack_snapshot = adapter.ack("msg_http_demo")
    nack_snapshot = adapter.nack("msg_http_demo", reason="demonstration")
    _printer_status("TEST", "Synthetic ack/nack completed", "info")

    state_snapshot = adapter.get_state_snapshot()
    close_snapshot = adapter.close(reason="test_complete")
    _printer_status("TEST", "HTTP adapter closed", "info")

    print("Connect Snapshot:", stable_json_dumps(connect_snapshot))
    print("Send Snapshot:", stable_json_dumps(send_snapshot))
    print("Cached Receive Snapshot:", stable_json_dumps(cached_receive_snapshot))
    print("Polled Receive Snapshot:", stable_json_dumps(polled_receive_snapshot))
    print("Redirect Snapshot:", stable_json_dumps(redirect_snapshot))
    print("Ack Snapshot:", stable_json_dumps(ack_snapshot))
    print("Nack Snapshot:", stable_json_dumps(nack_snapshot))
    print("State Snapshot:", stable_json_dumps(state_snapshot))
    print("Close Snapshot:", stable_json_dumps(close_snapshot))

    assert connect_snapshot["connected"] is True
    assert send_snapshot["result"]["response"]["status_code"] == 200
    cached_body = cached_receive_snapshot["payload"]["body"]
    if isinstance(cached_body, Mapping):
        assert cached_body["hello"] == "world"
    else:
        assert "world" in str(cached_body)
    assert polled_receive_snapshot["payload"]["event"] == "polled"
    assert redirect_snapshot["result"]["response"]["status_code"] == 200
    assert ack_snapshot["acknowledged"] is True
    assert close_snapshot["closed"] is True

    server.shutdown()
    server.server_close()
    _printer_status("TEST", "All HTTP Adapter checks passed", "info")
    print("\n=== Test ran successfully ===\n")
