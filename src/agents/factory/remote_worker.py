from __future__ import annotations

import importlib
import json
import sys

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, TextIO, Tuple, Type

from .utils.config_loader import get_config_section, load_global_config
from .utils.factory_errors import *
from .utils.factory_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Remote Worker")
printer = PrettyPrinter()


DEFAULT_ALLOWED_METHODS: Tuple[str, ...] = (
    "execute",
    "perform_task",
    "act",
    "predict",
    "get_action",
    "failure_normalization",
)

DEFAULT_ALLOWED_MODULE_PREFIXES: Tuple[str, ...] = ("src.agents",)

DEFAULT_CONSTRUCTOR_DEPENDENCIES: Dict[str, Dict[str, Any]] = {
    "shared_memory": {
        "enabled": True,
        "module_path": "src.agents.collaborative.shared_memory",
        "class_name": "SharedMemory",
        "args": [],
        "kwargs": {},
    },
    "agent_factory": {
        "enabled": True,
        "module_path": "src.agents.agent_factory",
        "class_name": "AgentFactory",
        "args": [],
        "kwargs": {},
    },
}


@dataclass(slots=True)
class RemoteWorkerConfig:
    """Runtime configuration for the subprocess-side remote worker."""

    enabled: bool = True
    allowed_module_prefixes: Tuple[str, ...] = DEFAULT_ALLOWED_MODULE_PREFIXES
    allowed_methods: Tuple[str, ...] = DEFAULT_ALLOWED_METHODS
    allow_private_methods: bool = False
    inject_constructor_dependencies: bool = True
    constructor_dependencies: Dict[str, Dict[str, Any]] = field(default_factory=lambda: dict(DEFAULT_CONSTRUCTOR_DEPENDENCIES))
    strict_dependency_injection: bool = False
    fallback_without_dependencies: bool = True
    max_stdin_chars: int = 1_000_000
    max_result_chars: int = 2_000_000
    include_diagnostics: bool = True
    include_error_payload: bool = True
    redact_payload: bool = True
    json_sort_keys: bool = False

    @classmethod
    def from_section(cls, section: Optional[Mapping[str, Any]]) -> "RemoteWorkerConfig":
        data = normalize_payload(section)
        raw_allowed_prefixes = data.get("allowed_module_prefixes", DEFAULT_ALLOWED_MODULE_PREFIXES)
        raw_allowed_methods = data.get("allowed_methods", DEFAULT_ALLOWED_METHODS)
        raw_dependencies = data.get("constructor_dependencies", DEFAULT_CONSTRUCTOR_DEPENDENCIES)

        allowed_prefixes = tuple(str(item).strip() for item in require_sequence(raw_allowed_prefixes, "remote_worker.allowed_module_prefixes", allow_empty=True) if str(item).strip())
        allowed_methods = tuple(str(item).strip() for item in require_sequence(raw_allowed_methods, "remote_worker.allowed_methods", allow_empty=True) if str(item).strip())
        dependencies = dict(require_mapping(raw_dependencies, "remote_worker.constructor_dependencies", allow_empty=True))

        return cls(
            enabled=bool(data.get("enabled", True)),
            allowed_module_prefixes=allowed_prefixes,
            allowed_methods=allowed_methods,
            allow_private_methods=bool(data.get("allow_private_methods", False)),
            inject_constructor_dependencies=bool(data.get("inject_constructor_dependencies", True)),
            constructor_dependencies=dependencies,
            strict_dependency_injection=bool(data.get("strict_dependency_injection", False)),
            fallback_without_dependencies=bool(data.get("fallback_without_dependencies", True)),
            max_stdin_chars=ensure_positive_int(int(data.get("max_stdin_chars", 1_000_000)), "remote_worker.max_stdin_chars"),
            max_result_chars=ensure_positive_int(int(data.get("max_result_chars", 2_000_000)), "remote_worker.max_result_chars"),
            include_diagnostics=bool(data.get("include_diagnostics", True)),
            include_error_payload=bool(data.get("include_error_payload", True)),
            redact_payload=bool(data.get("redact_payload", True)),
            json_sort_keys=bool(data.get("json_sort_keys", False)),
        )


@dataclass(slots=True)
class RemoteWorkerRequest:
    """Validated request envelope consumed by the worker."""

    module_path: str
    class_name: str
    method: str
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    agent_type: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], config: RemoteWorkerConfig) -> "RemoteWorkerRequest":
        data = validate_remote_worker_payload(payload)
        module_path = validate_module_path(data["module_path"], allowed_prefixes=config.allowed_module_prefixes)
        class_name = validate_class_name(data["class_name"])
        method = require_non_empty_string(data["method"], "method", max_length=256)

        if not config.allow_private_methods and method.startswith("_"):
            raise RemoteWorkerPayloadError(
                "Remote worker method is private and not allowed",
                context={"method": method, "allow_private_methods": config.allow_private_methods},
            )

        if config.allowed_methods and method not in config.allowed_methods:
            raise RemoteWorkerPayloadError(
                "Remote worker method is not allowed by policy",
                context={"method": method, "allowed_methods": config.allowed_methods},
            )

        args = tuple(data.get("args") or ())
        kwargs = dict(data.get("kwargs") or {})
        request_id = data.get("request_id")
        agent_type = data.get("agent_type")

        return cls(
            module_path=module_path,
            class_name=class_name,
            method=method,
            args=args,
            kwargs=kwargs,
            request_id=str(request_id) if request_id not in (None, "") else None,
            agent_type=str(agent_type) if agent_type not in (None, "") else None,
        )

    def context(self) -> Dict[str, Any]:
        return {
            "module_path": self.module_path,
            "class_name": self.class_name,
            "method": self.method,
            "request_id": self.request_id,
            "agent_type": self.agent_type,
        }


@dataclass(slots=True)
class RemoteWorkerInvocation:
    """Structured invocation result returned before JSON encoding."""

    status: str
    request: RemoteWorkerRequest
    result: Any = None
    error: Optional[FactoryError] = None
    duration_ms: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_response(self, *, include_diagnostics: bool = True, include_error_payload: bool = True, redact: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": self.status,
            "request_id": self.request.request_id,
            "agent": self.request.agent_type or self.request.class_name,
            "module_path": self.request.module_path,
            "class_name": self.request.class_name,
            "method": self.request.method,
            "duration_ms": self.duration_ms,
        }
        if self.status == "ok":
            payload["result"] = self.result
        elif self.error is not None:
            payload["message"] = self.error.message
            payload["code"] = self.error.code
            payload["error_type"] = self.error.error_type.value
            payload["retryable"] = self.error.retryable
            if include_error_payload:
                payload["error"] = self.error.to_dict(redact=redact, include_cause=True, include_traceback=False)
        else:
            payload["message"] = "Remote worker failed without an error payload"
        if include_diagnostics:
            payload["diagnostics"] = sanitize_context(self.diagnostics, redact=redact)
        return payload


class RemoteWorker:
    """Subprocess-side worker for isolated agent construction and invocation."""

    def __init__(self, config_override: Optional[Mapping[str, Any]] = None) -> None:
        self.config = load_global_config()
        self.worker_config = get_config_section("remote_worker")
        if not self.worker_config:
            # Backward-compatible alias for older factory_config.yaml files.
            self.worker_config = get_config_section("oopa_worker")
        if config_override:
            merged = dict(self.worker_config)
            merged.update(dict(config_override))
            self.worker_config = merged

        self.settings = RemoteWorkerConfig.from_section(self.worker_config)
        logger.info("Remote Worker successfully initialized")

    def parse_payload(self, raw_input: str) -> RemoteWorkerRequest:
        if len(raw_input) > self.settings.max_stdin_chars:
            raise RemoteWorkerPayloadError(
                "Remote worker stdin payload exceeds configured maximum size",
                context={"max_stdin_chars": self.settings.max_stdin_chars, "actual_chars": len(raw_input)},
            )

        try:
            payload = json.loads(raw_input or "{}")
        except json.JSONDecodeError as exc:
            raise RemoteWorkerSerializationError(
                "Remote worker stdin is not valid JSON",
                context={"stdin_chars": len(raw_input)},
                cause=exc,
            ) from exc

        if not isinstance(payload, Mapping):
            raise RemoteWorkerPayloadError(
                "Remote worker payload must be a JSON object",
                context={"payload_type": type(payload).__name__},
            )

        return RemoteWorkerRequest.from_payload(payload, self.settings)

    def _import_module(self, module_path: str) -> ModuleType:
        try:
            return importlib.import_module(module_path)
        except Exception as exc:
            raise AgentModuleImportError(
                "Remote worker failed to import target module",
                context={"module_path": module_path},
                cause=exc,
            ) from exc

    def _resolve_class(self, module: ModuleType, class_name: str) -> Type[Any]:
        try:
            resolved = getattr(module, class_name)
        except AttributeError as exc:
            raise AgentClassResolutionError(
                "Remote worker failed to resolve target class",
                context={"module_path": getattr(module, "__name__", None), "class_name": class_name},
                cause=exc,
            ) from exc
        if not isinstance(resolved, type):
            raise AgentClassResolutionError(
                "Resolved target is not a class",
                context={"class_name": class_name, "resolved_type": type(resolved).__name__},
            )
        return resolved

    def _resolve_provider(self, provider_name: str, provider_config: Mapping[str, Any]) -> Any:
        module_path = validate_module_path(provider_config.get("module_path"))
        class_name = validate_class_name(provider_config.get("class_name"))
        args = tuple(provider_config.get("args") or ())
        kwargs = dict(provider_config.get("kwargs") or {})

        module = self._import_module(module_path)
        provider_cls = self._resolve_class(module, class_name)
        try:
            return provider_cls(*args, **kwargs)
        except Exception as exc:
            raise AgentDependencyInjectionError(
                "Remote worker failed to instantiate constructor dependency",
                context={"provider": provider_name, "module_path": module_path, "class_name": class_name},
                cause=exc,
            ) from exc

    def _build_constructor_kwargs(self) -> Dict[str, Any]:
        if not self.settings.inject_constructor_dependencies:
            return {}

        constructor_kwargs: Dict[str, Any] = {}
        for dependency_name, raw_dependency_config in self.settings.constructor_dependencies.items():
            dependency_config = require_mapping(raw_dependency_config, f"constructor_dependencies[{dependency_name}]", allow_empty=False)
            if not bool(dependency_config.get("enabled", True)):
                continue
            try:
                constructor_kwargs[str(dependency_name)] = self._resolve_provider(str(dependency_name), dependency_config)
            except FactoryError:
                if self.settings.strict_dependency_injection:
                    raise
                logger.warning("Skipping optional remote-worker constructor dependency '%s'", dependency_name)
        return constructor_kwargs

    def build_agent(self, request: RemoteWorkerRequest) -> Any:
        module = self._import_module(request.module_path)
        agent_cls = self._resolve_class(module, request.class_name)
        constructor_kwargs = self._build_constructor_kwargs()

        try:
            return agent_cls(**constructor_kwargs)
        except TypeError as exc:
            if constructor_kwargs and self.settings.fallback_without_dependencies:
                logger.warning(
                    "Agent '%s' rejected injected constructor dependencies; retrying without dependencies",
                    request.class_name,
                )
                try:
                    return agent_cls()
                except Exception as retry_exc:
                    raise AgentConstructionError(
                        "Remote worker failed to construct target agent after dependency-free fallback",
                        context={**request.context(), "constructor_kwargs": tuple(constructor_kwargs.keys())},
                        cause=retry_exc,
                    ) from retry_exc
            raise AgentConstructionError(
                "Remote worker failed to construct target agent",
                context={**request.context(), "constructor_kwargs": tuple(constructor_kwargs.keys())},
                cause=exc,
            ) from exc
        except Exception as exc:
            raise AgentConstructionError(
                "Remote worker failed to construct target agent",
                context={**request.context(), "constructor_kwargs": tuple(constructor_kwargs.keys())},
                cause=exc,
            ) from exc

    def resolve_method(self, agent: Any, request: RemoteWorkerRequest) -> Any:
        try:
            method = getattr(agent, request.method)
        except AttributeError as exc:
            raise RemoteWorkerInvocationError(
                "Remote worker target method is not available",
                context=request.context(),
                cause=exc,
            ) from exc
        if not callable(method):
            raise RemoteWorkerInvocationError(
                "Remote worker target attribute is not callable",
                context={**request.context(), "attribute_type": type(method).__name__},
            )
        return method

    def invoke(self, request: RemoteWorkerRequest) -> RemoteWorkerInvocation:
        start_ms = monotonic_ms()
        diagnostics: Dict[str, Any] = {
            "worker_enabled": self.settings.enabled,
            "args_count": len(request.args),
            "kwargs_keys": tuple(request.kwargs.keys()),
        }
        try:
            if not self.settings.enabled:
                raise RemoteWorkerInvocationError("Remote worker is disabled by configuration", context=request.context())

            agent = self.build_agent(request)
            method = self.resolve_method(agent, request)
            result = method(*request.args, **request.kwargs)
            duration_ms = monotonic_ms() - start_ms
            diagnostics.update({"agent_type": type(agent).__name__, "result_type": type(result).__name__})
            return RemoteWorkerInvocation(
                status="ok",
                request=request,
                result=result,
                duration_ms=duration_ms,
                diagnostics=diagnostics,
            )
        except FactoryError as exc:
            duration_ms = monotonic_ms() - start_ms
            exc.with_context(**request.context())
            return RemoteWorkerInvocation(
                status="error",
                request=request,
                error=exc,
                duration_ms=duration_ms,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            duration_ms = monotonic_ms() - start_ms
            error = RemoteWorkerInvocationError(
                "Remote worker target invocation failed",
                context=request.context(),
                cause=exc,
            )
            return RemoteWorkerInvocation(
                status="error",
                request=request,
                error=error,
                duration_ms=duration_ms,
                diagnostics=diagnostics,
            )

    def encode_response(self, invocation: RemoteWorkerInvocation) -> str:
        response = invocation.to_response(
            include_diagnostics=self.settings.include_diagnostics,
            include_error_payload=self.settings.include_error_payload,
            redact=self.settings.redact_payload,
        )
        try:
            encoded = json.dumps(
                response,
                default=self._json_default,
                sort_keys=self.settings.json_sort_keys,
                ensure_ascii=False,
            )
        except Exception as exc:
            fallback_error = RemoteWorkerSerializationError(
                "Remote worker response serialization failed",
                context={"status": invocation.status, "method": invocation.request.method},
                cause=exc,
            )
            encoded = json.dumps(
                {
                    "status": "error",
                    "message": fallback_error.message,
                    "code": fallback_error.code,
                    "error_type": fallback_error.error_type.value,
                    "error": fallback_error.to_dict(redact=True, include_cause=True),
                },
                default=str,
                ensure_ascii=False,
            )

        if len(encoded) > self.settings.max_result_chars:
            truncated_error = RemoteWorkerSerializationError(
                "Remote worker encoded response exceeds configured maximum size",
                context={"max_result_chars": self.settings.max_result_chars, "actual_chars": len(encoded)},
            )
            return json.dumps(
                {
                    "status": "error",
                    "message": truncated_error.message,
                    "code": truncated_error.code,
                    "error_type": truncated_error.error_type.value,
                    "error": truncated_error.to_dict(redact=True, include_cause=False),
                },
                default=str,
                ensure_ascii=False,
            )
        return encoded

    @staticmethod
    def _json_default(value: Any) -> Any:
        return safe_serialize(value, redact=True)

    def handle(self, raw_input: str) -> Tuple[int, str, str]:
        """Process one JSON request and return ``(exit_code, stdout, stderr)``."""
        start_ms = monotonic_ms()
        try:
            request = self.parse_payload(raw_input)
        except FactoryError as exc:
            # Build a minimal request shell so the response remains consistently shaped.
            fallback_request = RemoteWorkerRequest(module_path="", class_name="", method="", args=(), kwargs={})
            invocation = RemoteWorkerInvocation(
                status="error",
                request=fallback_request,
                error=exc,
                duration_ms=monotonic_ms() - start_ms,
                diagnostics={"phase": "parse_payload"},
            )
            return 1, self.encode_response(invocation), f"{exc.__class__.__name__}: {exc.message}"
        except Exception as exc:
            error = RemoteWorkerError("Remote worker request handling failed", cause=exc)
            fallback_request = RemoteWorkerRequest(module_path="", class_name="", method="", args=(), kwargs={})
            invocation = RemoteWorkerInvocation(
                status="error",
                request=fallback_request,
                error=error,
                duration_ms=monotonic_ms() - start_ms,
                diagnostics={"phase": "parse_payload"},
            )
            return 1, self.encode_response(invocation), f"{error.__class__.__name__}: {error.message}"

        invocation = self.invoke(request)
        stdout = self.encode_response(invocation)
        stderr = "" if invocation.status == "ok" else f"{invocation.error.__class__.__name__}: {invocation.error.message}" if invocation.error else "RemoteWorkerError"
        exit_code = 0 if invocation.status == "ok" else 1
        return exit_code, stdout, stderr


def main(raw_input: Optional[str] = None, stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr) -> int:
    worker = RemoteWorker()
    input_text = sys.stdin.read() if raw_input is None else raw_input
    exit_code, out_text, err_text = worker.handle(input_text)
    if out_text:
        stdout.write(out_text)
    if err_text:
        stderr.write(err_text)
    return exit_code



if __name__ == "__main__":

    class _RemoteWorkerSelfTestAgent:
        def __init__(self, **_: Any) -> None:
            self.ready = True

        def execute(self, value: int = 1) -> Dict[str, Any]:
            return {"ready": self.ready, "value": value}

        def perform_task(self, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
            return {"payload": dict(payload or {})}


    def _run_self_test() -> int:
        print("\n=== Running Remote Worker ===\n")
        printer.status("TEST", "Remote Worker initialized", "info")

        worker = RemoteWorker(
            {
                "allowed_module_prefixes": ("__main__",),
                "allowed_methods": ("execute", "perform_task"),
                "inject_constructor_dependencies": False,
                "include_diagnostics": True,
                "max_result_chars": 100_000,
            }
        )
        payload = {
            "module_path": "__main__",
            "class_name": "_RemoteWorkerSelfTestAgent",
            "method": "execute",
            "args": [7],
            "kwargs": {},
            "request_id": "remote-worker-self-test",
            "agent_type": "self_test",
        }
        exit_code, out_text, err_text = worker.handle(json.dumps(payload))
        if exit_code != 0:
            raise RemoteWorkerInvocationError("Remote worker self-test failed", context={"stderr": err_text, "stdout": out_text})
        response = json.loads(out_text)
        validate_worker_response(response)
        if response.get("result", {}).get("value") != 7:
            raise RemoteWorkerResultError("Remote worker self-test returned an unexpected result", context={"response": response})

        denied_payload = dict(payload)
        denied_payload["method"] = "_private"
        denied_exit_code, denied_out_text, _ = worker.handle(json.dumps(denied_payload))
        if denied_exit_code == 0:
            raise RemoteWorkerPayloadError("Remote worker self-test expected private method denial")
        denied_response = json.loads(denied_out_text)
        if denied_response.get("status") != "error":
            raise RemoteWorkerResultError("Remote worker self-test expected structured error response", context={"response": denied_response})

        printer.status("TEST", "Remote Worker invocation checks passed", "success")
        print("\n=== Test ran successfully ===\n")
        return 0

    stdin_text = sys.stdin.read()
    if "--self-test" in sys.argv or not stdin_text.strip():
        raise SystemExit(_run_self_test())
    raise SystemExit(main(stdin_text))
