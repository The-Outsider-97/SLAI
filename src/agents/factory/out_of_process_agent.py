from __future__ import annotations

import json
import os
import subprocess
import sys
import time as time_module

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .utils.config_loader import get_config_section, load_global_config
from .utils.factory_errors import *
from .utils.factory_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Out Of Process Agent")
printer = PrettyPrinter()


@dataclass(slots=True)
class OutOfProcessCallRecord:
    """Compact runtime record for one isolated worker call."""

    method: str
    status: str
    duration_ms: float
    returncode: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "returncode": self.returncode,
            "error": self.error,
        }


@dataclass(slots=True)
class OutOfProcessAgentProxy:
    """
    Runs agent calls in an isolated Python subprocess.

    The proxy is intentionally part of the factory-isolation boundary: it does
    not own registry logic, routing decisions, learning loops, or governance
    policy. Its responsibility is to safely invoke an already-resolved agent
    implementation in a child Python process and return either the worker result
    or a structured degraded result when isolation fails.
    """

    agent_type: str
    module_path: str
    class_name: str
    init_error: str = ""

    implementation: str = "out_of_process_proxy"

    config: Dict[str, Any] = field(init=False, repr=False)
    oopa_config: Dict[str, Any] = field(init=False, repr=False)
    worker_module: str = field(init=False)
    python_executable: str = field(init=False)
    timeout_seconds: Optional[float] = field(init=False)
    max_retries: int = field(init=False)
    retry_backoff_seconds: float = field(init=False)
    max_stdout_chars: int = field(init=False)
    max_stderr_chars: int = field(init=False)
    max_history_size: int = field(init=False)
    return_envelope: bool = field(init=False)
    include_worker_stderr: bool = field(init=False)
    strict_worker_response: bool = field(init=False)
    allow_private_methods: bool = field(init=False)
    pass_through_env: bool = field(init=False)
    allowed_methods: Tuple[str, ...] = field(init=False)
    environment_overrides: Dict[str, str] = field(init=False)

    invocation_count: int = field(default=0, init=False)
    success_count: int = field(default=0, init=False)
    degraded_count: int = field(default=0, init=False)
    failure_count: int = field(default=0, init=False)
    last_error: Optional[str] = field(default=None, init=False)
    last_duration_ms: Optional[float] = field(default=None, init=False)
    _history: list[OutOfProcessCallRecord] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.config = load_global_config()
        self.oopa_config = get_config_section("oopa")

        self.agent_type = validate_agent_name(self.agent_type, max_length=128)
        allowed_prefixes = self._configured_allowed_module_prefixes()
        self.module_path = validate_module_path(self.module_path, allowed_prefixes=allowed_prefixes)
        self.class_name = validate_class_name(self.class_name)

        self.worker_module = require_non_empty_string(
            self.oopa_config.get("worker_module", "src.agents.factory.remote_worker"),
            "oopa.worker_module",
            max_length=512,
        )
        self.python_executable = str(self.oopa_config.get("python_executable") or sys.executable)

        timeout_value = self.oopa_config.get("timeout_seconds", 60.0)
        self.timeout_seconds = ensure_positive_optional_float(timeout_value, "oopa.timeout_seconds")

        self.max_retries = int(self.oopa_config.get("max_retries", 0))
        if self.max_retries < 0:
            raise InvalidFactoryConfigurationError(
                "oopa.max_retries must be >= 0",
                context={"max_retries": self.max_retries},
                component="out_of_process_agent",
            )

        self.retry_backoff_seconds = float(self.oopa_config.get("retry_backoff_seconds", 0.0))
        if self.retry_backoff_seconds < 0:
            raise InvalidFactoryConfigurationError(
                "oopa.retry_backoff_seconds must be >= 0",
                context={"retry_backoff_seconds": self.retry_backoff_seconds},
                component="out_of_process_agent",
            )

        self.max_stdout_chars = ensure_positive_int(int(self.oopa_config.get("max_stdout_chars", 20_000)), "oopa.max_stdout_chars")
        self.max_stderr_chars = ensure_positive_int(int(self.oopa_config.get("max_stderr_chars", 10_000)), "oopa.max_stderr_chars")
        self.max_history_size = ensure_positive_int(int(self.oopa_config.get("max_history_size", 50)), "oopa.max_history_size")

        self.return_envelope = bool(self.oopa_config.get("return_envelope", False))
        self.include_worker_stderr = bool(self.oopa_config.get("include_worker_stderr", True))
        self.strict_worker_response = bool(self.oopa_config.get("strict_worker_response", False))
        self.allow_private_methods = bool(self.oopa_config.get("allow_private_methods", False))
        self.pass_through_env = bool(self.oopa_config.get("pass_through_env", True))

        configured_methods = self.oopa_config.get("allowed_methods", ())
        self.allowed_methods = self._normalise_allowed_methods(configured_methods)
        self.environment_overrides = self._normalise_environment_overrides(
            self.oopa_config.get(
                "environment",
                {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUTF8": "1",
                },
            )
        )

        logger.info("Out Of Process Agent successfully initialized")

    # ------------------------------------------------------------------
    # Configuration and payload preparation
    # ------------------------------------------------------------------
    def _configured_allowed_module_prefixes(self) -> Tuple[str, ...]:
        prefixes = self.oopa_config.get("allowed_module_prefixes")
        if prefixes is None:
            meta_rules = get_config_section("agent_meta").get("validation_rules", {})
            prefixes = meta_rules.get("allowed_modules", ())
        if not prefixes:
            return tuple()
        if isinstance(prefixes, str):
            return (prefixes,)
        if not isinstance(prefixes, Sequence):
            raise InvalidFactoryConfigurationError(
                "oopa.allowed_module_prefixes must be a sequence of strings",
                context={"actual_type": type(prefixes).__name__},
                component="out_of_process_agent",
            )
        return tuple(require_non_empty_string(prefix, "oopa.allowed_module_prefixes[]") for prefix in prefixes)

    def _normalise_allowed_methods(self, methods: Any) -> Tuple[str, ...]:
        if methods in (None, "", ()):
            return tuple()
        if isinstance(methods, str):
            methods = [methods]
        if not isinstance(methods, Sequence):
            raise InvalidFactoryConfigurationError(
                "oopa.allowed_methods must be a sequence of method names",
                context={"actual_type": type(methods).__name__},
                component="out_of_process_agent",
            )
        return tuple(require_non_empty_string(method, "oopa.allowed_methods[]", max_length=256) for method in methods)

    def _normalise_environment_overrides(self, env: Any) -> Dict[str, str]:
        if env in (None, ""):
            return {}
        mapping = require_mapping(env, "oopa.environment", allow_empty=True)
        normalised: Dict[str, str] = {}
        for key, value in mapping.items():
            env_key = require_non_empty_string(str(key), "oopa.environment.key", max_length=128)
            normalised[env_key] = str(value)
        return normalised

    def _validate_method(self, method: str) -> str:
        method_name = require_non_empty_string(method, "method", max_length=256)
        if not self.allow_private_methods and method_name.startswith("_"):
            raise RemoteWorkerPayloadError(
                "Private or dunder methods cannot be invoked out-of-process",
                context={"agent": self.agent_type, "method": method_name},
                component="out_of_process_agent",
                operation="validate_method",
            )
        if self.allowed_methods and method_name not in self.allowed_methods:
            raise RemoteWorkerPayloadError(
                "Method is not allowed by oopa.allowed_methods",
                context={"agent": self.agent_type, "method": method_name, "allowed_methods": self.allowed_methods},
                component="out_of_process_agent",
                operation="validate_method",
            )
        return method_name

    def _json_default(self, value: Any) -> Any:
        return safe_serialize(value, redact=True)

    def _build_payload(self, method: str, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        method_name = self._validate_method(method)
        payload: Dict[str, Any] = {
            "module_path": self.module_path,
            "class_name": self.class_name,
            "method": method_name,
            "args": list(args),
            "kwargs": dict(kwargs),
        }
        return validate_remote_worker_payload(payload)

    def _serialise_payload(self, payload: Mapping[str, Any]) -> str:
        try:
            return json.dumps(payload, default=self._json_default)
        except Exception as exc:
            raise RemoteWorkerSerializationError(
                "Failed to serialise out-of-process worker payload",
                context={"agent": self.agent_type, "payload": safe_serialize(payload, redact=True)},
                component="out_of_process_agent",
                operation="serialise_payload",
                cause=exc,
            ) from exc

    def _build_child_environment(self) -> Dict[str, str]:
        child_env = os.environ.copy() if self.pass_through_env else {}
        # Ensure worker stdio uses UTF-8 to avoid Windows cp1252/charmap crashes
        # when dependencies/loggers emit non-ASCII symbols (e.g. ℹ).
        child_env.setdefault("PYTHONIOENCODING", "utf-8")
        child_env.setdefault("PYTHONUTF8", "1")
        child_env.update(self.environment_overrides)
        return {str(key): str(value) for key, value in child_env.items()}

    def _command(self) -> list[str]:
        return [self.python_executable, "-m", self.worker_module]

    # ------------------------------------------------------------------
    # Invocation and result handling
    # ------------------------------------------------------------------
    def _invoke(self, method: str, *args: Any, **kwargs: Any) -> Any:
        envelope = self.invoke_envelope(method, *args, **kwargs)
        if self.return_envelope:
            return envelope
        if envelope.get("status") == "ok":
            return envelope.get("result")
        return envelope

    def invoke_envelope(self, method: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Invoke a worker method and always return a structured envelope."""
        started_ms = monotonic_ms()
        method_name = str(method)
        self.invocation_count += 1

        try:
            payload = self._build_payload(method_name, args, kwargs)
            serialised_payload = self._serialise_payload(payload)
            response = self._run_with_retries(method_name, serialised_payload)
            duration_ms = monotonic_ms() - started_ms
            self.last_duration_ms = duration_ms
            self.success_count += 1
            response = dict(response)
            response.setdefault("status", "ok")
            response.setdefault("agent", self.agent_type)
            response.setdefault("method", method_name)
            response.setdefault("duration_ms", duration_ms)
            self._record(method_name, "ok", duration_ms, None, None)
            return response

        except FactoryError as exc:
            duration_ms = monotonic_ms() - started_ms
            self.last_duration_ms = duration_ms
            self.last_error = exc.message
            self.degraded_count += 1
            self.failure_count += 1
            self._record(method_name, "degraded", duration_ms, None, exc.message)
            exc.log()
            return self._degraded_result(exc, method_name, duration_ms)

        except Exception as exc:
            wrapped = wrap_factory_exception(
                exc,
                message="Out-of-process invocation failed",
                operation=method_name,
                component="out_of_process_agent",
                context={"agent": self.agent_type, "module_path": self.module_path, "class_name": self.class_name},
                default_error_cls=RemoteWorkerInvocationError,
                retryable=True,
            )
            duration_ms = monotonic_ms() - started_ms
            self.last_duration_ms = duration_ms
            self.last_error = wrapped.message
            self.degraded_count += 1
            self.failure_count += 1
            self._record(method_name, "degraded", duration_ms, None, wrapped.message)
            wrapped.log()
            return self._degraded_result(wrapped, method_name, duration_ms)

    def _run_with_retries(self, method: str, serialised_payload: str) -> Dict[str, Any]:
        attempts = self.max_retries + 1
        last_error: Optional[FactoryError] = None

        for attempt in range(1, attempts + 1):
            try:
                return self._run_once(method, serialised_payload, attempt=attempt)
            except FactoryError as exc:
                last_error = exc
                if attempt >= attempts or not exc.retryable:
                    raise
                if self.retry_backoff_seconds > 0:
                    time_module.sleep(self.retry_backoff_seconds * attempt)

        if last_error is not None:
            raise last_error
        raise RemoteWorkerInvocationError(
            "Out-of-process invocation failed without a captured error",
            context={"agent": self.agent_type, "method": method},
            component="out_of_process_agent",
            operation=method,
        )

    def _run_once(self, method: str, serialised_payload: str, *, attempt: int) -> Dict[str, Any]:
        try:
            completed = subprocess.run(
                self._command(),
                input=serialised_payload,
                text=True,
                capture_output=True,
                check=False,
                env=self._build_child_environment(),
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise FactoryTimeoutError(
                "Out-of-process worker timed out",
                context={
                    "agent": self.agent_type,
                    "method": method,
                    "attempt": attempt,
                    "timeout_seconds": self.timeout_seconds,
                },
                component="out_of_process_agent",
                operation=method,
                cause=exc,
                retryable=True,
            ) from exc
        except Exception as exc:
            raise SubprocessExecutionError(
                "Failed to start or execute out-of-process worker",
                context={"agent": self.agent_type, "method": method, "attempt": attempt, "command": self._command()},
                component="out_of_process_agent",
                operation=method,
                cause=exc,
                retryable=True,
            ) from exc

        stdout = self._truncate_output(completed.stdout or "", self.max_stdout_chars)
        stderr = self._truncate_output(completed.stderr or "", self.max_stderr_chars)

        if completed.returncode != 0:
            error_text = stderr or stdout or f"worker exited with code {completed.returncode}"
            raise SubprocessExecutionError(
                "Out-of-process worker returned a non-zero exit code",
                context={
                    "agent": self.agent_type,
                    "method": method,
                    "attempt": attempt,
                    "returncode": completed.returncode,
                    "stderr": stderr if self.include_worker_stderr else None,
                    "stdout": stdout,
                },
                component="out_of_process_agent",
                operation=method,
                cause=RuntimeError(error_text),
                retryable=True,
            )

        return self._parse_worker_output(method, stdout, stderr, attempt=attempt)

    def _parse_worker_output(self, method: str, stdout: str, stderr: str, *, attempt: int) -> Dict[str, Any]:
        raw = stdout.strip()
        if not raw:
            response = {"status": "ok", "agent": self.agent_type, "result": None}
            validate_worker_response(response)
            return response

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = self._parse_json_from_noisy_stdout(raw)

        if not isinstance(parsed, Mapping):
            raise RemoteWorkerResultError(
                "Out-of-process worker returned a non-object JSON response",
                context={"agent": self.agent_type, "method": method, "response_type": type(parsed).__name__},
                component="out_of_process_agent",
                operation=method,
                retryable=True,
            )

        response = dict(parsed)
        if "status" not in response:
            # Backward-compatible normalization for workers that return a bare
            # object even though remote_worker normally emits a status envelope.
            response = {"status": "ok", "agent": self.agent_type, "result": response}
        elif response.get("status") == "ok" and "result" not in response:
            response["result"] = None

        if self.strict_worker_response:
            validate_worker_response(response)
        else:
            try:
                validate_worker_response(response)
            except RemoteWorkerResultError:
                response = {"status": "ok", "agent": self.agent_type, "result": response}

        if response.get("status") in {"error", "degraded"}:
            message = str(response.get("error") or response.get("message") or "Worker returned degraded response")
            raise RemoteWorkerInvocationError(
                message,
                context={
                    "agent": self.agent_type,
                    "method": method,
                    "attempt": attempt,
                    "response": response,
                    "stderr": stderr if self.include_worker_stderr else None,
                },
                component="out_of_process_agent",
                operation=method,
                retryable=True,
            )

        response.setdefault("agent", self.agent_type)
        return response

    def _parse_json_from_noisy_stdout(self, raw: str) -> Mapping[str, Any]:
        for line in reversed(raw.splitlines()):
            candidate = line.strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, Mapping):
                        return parsed
                except json.JSONDecodeError:
                    continue

        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                parsed = json.loads(raw[first : last + 1])
                if isinstance(parsed, Mapping):
                    return parsed
            except json.JSONDecodeError:
                pass

        raise RemoteWorkerSerializationError(
            "Out-of-process worker stdout is not valid JSON",
            context={"agent": self.agent_type, "stdout": raw},
            component="out_of_process_agent",
            operation="parse_worker_stdout",
            retryable=False,
        )

    def _truncate_output(self, value: str, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        return f"{value[:max_chars]}…[truncated {len(value) - max_chars} chars]"

    def _degraded_result(self, error: FactoryError, method: str, duration_ms: float) -> Dict[str, Any]:
        return error.to_degraded_result(
            agent=self.agent_type,
            action=method,
            extra={
                "implementation": self.implementation,
                "module_path": self.module_path,
                "class_name": self.class_name,
                "duration_ms": duration_ms,
                "init_error": self.init_error,
            },
        )

    def _record(self, method: str, status: str, duration_ms: float, returncode: Optional[int], error: Optional[str] ) -> None:
        self._history.append(OutOfProcessCallRecord(method=method, status=status, duration_ms=duration_ms,
                                                    returncode=returncode, error=error))
        if len(self._history) > self.max_history_size:
            del self._history[: len(self._history) - self.max_history_size]

    # ------------------------------------------------------------------
    # Backward-compatible proxy methods
    # ------------------------------------------------------------------
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("execute", *args, **kwargs)

    def perform_task(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("perform_task", *args, **kwargs)

    def act(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("act", *args, **kwargs)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("predict", *args, **kwargs)

    def get_action(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("get_action", *args, **kwargs)

    def failure_normalization(self, *args: Any, **kwargs: Any) -> Any:
        # This is frequently used by SignalSentry; keep best-effort semantics.
        return self._invoke("failure_normalization", *args, **kwargs)

    def __getattr__(self, item: str):
        if item.startswith("__"):
            raise AttributeError(item)

        def _call(*args: Any, **kwargs: Any) -> Any:
            return self._invoke(item, *args, **kwargs)

        return _call

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return {
            "status": "degraded" if self.last_error else "ok",
            "agent": self.agent_type,
            "implementation": self.implementation,
            "module_path": self.module_path,
            "class_name": self.class_name,
            "worker_module": self.worker_module,
            "timeout_seconds": self.timeout_seconds,
            "invocation_count": self.invocation_count,
            "success_count": self.success_count,
            "degraded_count": self.degraded_count,
            "failure_count": self.failure_count,
            "last_error": self.last_error,
            "last_duration_ms": self.last_duration_ms,
        }

    def history(self, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        if limit is None:
            records = self._history
        else:
            limit_value = int(ensure_non_negative_number(limit, "history.limit"))
            records = self._history[-limit_value:] if limit_value else []
        return [record.to_dict() for record in records]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "module_path": self.module_path,
            "class_name": self.class_name,
            "init_error": self.init_error,
            "implementation": self.implementation,
            "worker_module": self.worker_module,
            "python_executable": self.python_executable,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "return_envelope": self.return_envelope,
            "allowed_methods": list(self.allowed_methods),
            "health": self.health(),
        }


if __name__ == "__main__":
    print("\n=== Running OOPA ===\n")
    printer.status("TEST", "Factory Observability initialized", "info")

    proxy = OutOfProcessAgentProxy(
        agent_type="test_agent",
        module_path="src.agents.factory.remote_worker",
        class_name="RemoteWorker",
        init_error="test-mode",
    )

    payload = proxy._build_payload("execute", tuple(), {})
    assert payload["method"] == "execute"
    assert payload["module_path"] == "src.agents.factory.remote_worker"

    health = proxy.health()
    assert health["implementation"] == "out_of_process_proxy"
    assert health["agent"] == "test_agent"

    degraded = proxy._invoke("_private_test_method")
    assert degraded["status"] == "degraded"

    print("\n=== Test ran successfully ===\n")
