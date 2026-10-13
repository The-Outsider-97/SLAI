"""
Orchestrator Agent
==================

A lightweight, structurally-scoped orchestrator that fills micro-task and
coordination gaps in the SLAI runtime without duplicating the
responsibilities of heavyweight specialist agents.

Design responsibilities
-----------------------
1. **Inline micro-task execution.** Payload normalisation, lightweight
   sanitisation, and structural key-presence validation execute
   in-process, avoiding the registry, cache, and lifecycle overhead that
   full agent construction would impose on trivially-transformative work.

2. **Registry inspection and dynamic onboarding.** When a delegation
   target is absent from the factory registry, the orchestrator synthesises
   a validated :class:`AgentMetaData` record. For genuinely novel
   specialties it also materialises a functional ``BaseAgent`` subclass
   under ``src/agents/subagents/`` so that the factory can import and
   instantiate the target through its normal lifecycle. Registration is
   idempotent and re-validated against the registry on every call, so a
   peer's ``unregister`` cannot be masked by a stale cache entry.

3. **Collaborative telemetry.** Delegation start, delegation outcome, and
   dynamic registration events are published to ``SharedMemory``
   coordination channels so that peer agents and observability consumers
   retain a consistent view of runtime activity.

4. **Bounded performance caching.** A thread-safe LRU cache with optional
   TTL memoises micro-task results. Cache size and TTL are bounded by
   configuration, and the cache is explicitly cleared during
   ``shutdown`` to prevent stale entries from outliving the agent.

5. **Structured error management and resource lifecycle.** Low-level
   exceptions are projected into the factory error taxonomy, retryable
   failures are retried with bounded backoff, and ephemeral specialist
   instances are deterministically released according to the
   ``ttl_recycling`` policy.

Non-responsibilities
--------------------
- The orchestrator does **not** own global factory lifecycle, registry
  invariants, or specialist-internal logic. It is a *client* of those
  subsystems.
- It does **not** perform broad reasoning loops, model training, or
  heavyweight inference. Such work is delegated.
- It does **not** redefine the factory's canonical default agents. When a
  known default is missing from the registry, the orchestrator leaves the
  situation to the factory's own registration path rather than shadowing
  it with a generated module.

Configuration
-------------
Runtime policy is loaded from the ``orchestrator_agent`` section of
``agents_config.yaml`` through the shared configuration loader.
Constructor-supplied ``config`` overrides file values.

Dynamically-onboarded specialties are materialised as source modules under
``src/agents/subagents/``. Module generation is idempotent: existing files
are left untouched so that hand-written overrides are preserved across
orchestration passes.
"""

from __future__ import annotations

__version__ = "2.3.0"

import os
import tempfile
import time
import importlib
import re

from threading import RLock
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .base_agent import BaseAgent
from .base.utils.main_config_loader import get_config_section
from .factory.utils.factory_errors import FactoryError
from .factory.utils.factory_helpers import *
from .factory.agent_meta_data import AgentMetaData
from .factory.factory_cache import FactoryCache
from .factory.operations import validate_specialist_spec
from src.utils.configuration import bind_config # type: ignore
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Orchestrator Agent")
printer = PrettyPrinter()


_DYNAMIC_REGISTRATION_LOCK = RLock()
_SUBAGENT_SETTINGS = bind_config(
    Path(__file__).resolve().parent
    / "subagents"
    / "config"
    / "subagents_config.yaml"
)


# ---------------------------------------------------------------------------
# Auto-generated subagent module template
#
# The template is instantiated once per novel specialty. It produces a
# minimal but *functional* BaseAgent subclass: standard constructor
# contract, structural default ``perform_task`` implementation, and no
# optional dependencies. Users may replace the file in place; generation
# is guarded by file existence and never overwrites an existing module.
# ---------------------------------------------------------------------------
_SUBAGENT_MODULE_TEMPLATE = '''"""
SLAI generated specialist: {specialty}

A factory-managed specialized agent executing a validated operation.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..base_agent import BaseAgent
from ..factory.operations import execute_operation
from logs.logger import PrettyPrinter, get_logger


logger = get_logger("{class_name}")
printer = PrettyPrinter()


class {class_name}(BaseAgent):
    """Validated specialist for the {specialty} capability."""

    SPECIALTY = {specialty_repr}
    SPECIALIST_SPEC = {specialist_spec}
    CAPABILITIES = ({specialty_repr}, {operation_repr})

    def __init__(
        self,
        shared_memory: Any,
        agent_factory: Any,
        config: Optional[Mapping[str, Any]] = None,
        *,
        checkpoint_manager: Any = None,
    ) -> None:
        super().__init__(
            shared_memory,
            agent_factory,
            config,
            checkpoint_manager=checkpoint_manager,
        )

        logger.info(
            "{class_name} initialized | specialty=%s",
            self.SPECIALTY,
        )

    def perform_task(self, task_data: Any) -> dict[str, Any]:
        printer.status(
            "TASK",
            "{class_name} executing specialized operation",
            "info",
        )

        result = execute_operation(
            self.SPECIALIST_SPEC,
            task_data,
        )

        logger.info(
            "{class_name} completed operation: %s",
            self.SPECIALIST_SPEC["operation"],
        )

        return result


__all__ = ["{class_name}"]
'''



class OrchestratorAgent(BaseAgent):
    """Lightweight structural and micro-task orchestration for SLAI.

    The orchestrator combines three concerns that do not warrant a
    dedicated specialist:

    * micro-tasks too trivial to warrant a full agent instance;
    * delegation glue between callers and the factory-managed specialist
      registry, including dynamic onboarding of missing targets;
    * coordination telemetry that keeps peer agents and observability
      consumers aligned on runtime activity.

    See the module docstring for the explicit non-responsibility list.
    """

    # ------------------------------------------------------------------
    # Class-level defaults
    # ------------------------------------------------------------------
    DEFAULT_SPECIALTY: str = "reasoning"
    DEFAULT_CACHE_MAX_SIZE: int = 128
    DEFAULT_CACHE_TTL_SECONDS: float = 300.0
    DEFAULT_COORDINATION_PREFIX: str = "orchestrator"
    DEFAULT_DELEGATION_RETRIES: int = 2
    DEFAULT_RETRY_BACKOFF_SECONDS: float = 0.25

    #: Canonical module prefix for dynamically-onboarded subagents.
    SUBAGENT_MODULE_PREFIX: str = "src.agents.subagents"

    #: Filesystem directory (relative to this module) for subagent modules.
    SUBAGENT_DIR_NAME: str = "subagents"

    #: Task types that are executed inline without delegating.
    MICRO_TASK_TYPES: Tuple[str, ...] = (
        "normalize_payload",
        "sanitize_payload",
        "validate_schema",
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        shared_memory: Any,
        agent_factory: Any,
        config: Optional[Mapping[str, Any]] = None,
        *,
        checkpoint_manager: Any = None,
    ) -> None:
        super().__init__(
            shared_memory,
            agent_factory,
            config,
            checkpoint_manager=checkpoint_manager,
        )

        # ``orchestrator_agent`` is the canonical configuration section.
        # A constructor override is merged on top so that ephemeral
        # per-instance tuning remains possible without touching YAML.
        self.orchestrator_config: Dict[str, Any] = dict(
            get_config_section("orchestrator_agent") or {}
        )
        if config:
            self.orchestrator_config.update(dict(config))

        # ---- Policy ---------------------------------------------------
        self.default_specialty: str = validate_agent_name(self.orchestrator_config.get("default_specialty", self.DEFAULT_SPECIALTY))
        self.ttl_recycling: bool = bool(self.orchestrator_config.get("ttl_recycling", True))
        self.max_delegation_retries: int = max(
            0,
            int(
                self.orchestrator_config.get(
                    "max_delegation_retries", self.DEFAULT_DELEGATION_RETRIES
                )
            ),
        )
        self.retry_backoff_seconds: float = max(
            0.0,
            float(
                self.orchestrator_config.get(
                    "retry_backoff_seconds", self.DEFAULT_RETRY_BACKOFF_SECONDS
                )
            ),
        )
        self.coordination_prefix: str = (
            str(
                self.orchestrator_config.get(
                    "shared_memory_coordination_prefix",
                    self.DEFAULT_COORDINATION_PREFIX,
                )
            ).strip()
            or self.DEFAULT_COORDINATION_PREFIX
        )

        # ---- Cache parameters ----------------------------------------
        self.cache_max_size: int = max(
            1,
            int(
                self.orchestrator_config.get(
                    "cache_max_size", self.DEFAULT_CACHE_MAX_SIZE
                )
            ),
        )
        self.cache_ttl_seconds: float = max(
            0.0,
            float(
                self.orchestrator_config.get(
                    "cache_ttl_seconds", self.DEFAULT_CACHE_TTL_SECONDS
                )
            ),
        )

        # Thread-safe bounded LRU cache with TTL for micro-task
        # memoisation. Registration is not cached here — the registry is
        # the single source of truth for onboarding decisions.
        self.cache: FactoryCache[str, Any] = FactoryCache(
            max_size=self.cache_max_size,
            default_ttl_seconds=self.cache_ttl_seconds,
            name="orchestrator_cache",
        )

        # ---- Subagent module target ----------------------------------

        # Initialize the complete subagent workspace automatically.
        self._subagent_dir = (
            Path(__file__).resolve().parent
            / self.SUBAGENT_DIR_NAME
        )

        self._initialize_subagent_workspace()


        # Namespaces used for SharedMemory coordination channels.
        self._coordination_key: str = (f"{self.coordination_prefix}:task:{self.agent_id}")
        self._registration_prefix: str = (f"{self.coordination_prefix}:registration")

        self._publish_lifecycle_event(
            "orchestrator_initialized",
            {
                "default_specialty": self.default_specialty,
                "ttl_recycling": self.ttl_recycling,
                "max_delegation_retries": self.max_delegation_retries,
                "cache_max_size": self.cache_max_size,
                "cache_ttl_seconds": self.cache_ttl_seconds,
                "subagent_dir": str(self._subagent_dir),
            },
        )
        logger.info(
            "OrchestratorAgent initialized | specialty=%s ttl_recycling=%s "
            "retries=%s cache_max=%s cache_ttl=%ss subagent_dir=%s",
            self.default_specialty,
            self.ttl_recycling,
            self.max_delegation_retries,
            self.cache_max_size,
            self.cache_ttl_seconds,
            self._subagent_dir,
        )

    @staticmethod
    def _create_file_if_absent(path: Path, content: str) -> None:
        """Atomically create a file without replacing an existing file."""
        if path.is_symlink():
            raise ValueError(f"Symlink is not permitted: {path}")

        if path.exists():
            if not path.is_file():
                raise ValueError(f"Expected a regular file: {path}")
            return

        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())

            # Atomic, create-only publication.
            # A concurrent initializer cannot overwrite this file.
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass

        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Workspace file was not created safely: {path}")

    def _initialize_subagent_workspace(self) -> None:
        """Create and validate the canonical subagent workspace.

        Creates missing directories and initial configuration once.
        Existing valid configuration is preserved.
        """
        with _DYNAMIC_REGISTRATION_LOCK:
            root = Path(__file__).resolve().parent
            subagents_dir = root / self.SUBAGENT_DIR_NAME
            config_dir = subagents_dir / "config"

            for directory in (subagents_dir, config_dir):
                if directory.is_symlink():
                    raise ValueError(f"Workspace directory cannot be a symlink: {directory}")
                directory.mkdir(parents=True, exist_ok=True)

            init_file = subagents_dir / "__init__.py"
            config_file = config_dir / "subagents_config.yaml"

            self._create_file_if_absent(init_file, '"""SLAI dynamically managed specialist subagents."""\n')

            default_config = (
                '# SLAI v2.3 — Managed subagent configuration\n'
                'version: "2.3.0"\n'
                '\n'
                'generation:\n'
                '  enabled: true\n'
                '  max_definitions: 256\n'
            )

            self._create_file_if_absent(config_file, default_config)
            configuration = _SUBAGENT_SETTINGS.load(force_reload=True)
            generation = configuration.get("generation")
            if not isinstance(generation, Mapping):
                raise ValueError("subagents_config.yaml must define a generation mapping")

            if not isinstance(generation.get("enabled"), bool):
                raise ValueError("generation.enabled must be a boolean")

            maximum = generation.get("max_definitions")

            if (
                isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or maximum < 1
            ):
                raise ValueError("generation.max_definitions must be a positive integer")

            self._subagent_dir = subagents_dir
            self._subagent_config_dir = config_dir
            self._subagent_config_path = config_file

            logger.info(
                "ORCHESTRATOR | Subagent workspace ready | "
                "directory=%s | config=%s",
                subagents_dir,
                config_file,
            )

    # ==================================================================
    # Task dispatch
    # ==================================================================
    def perform_task(self, task_data: Any) -> Any:
        """Dispatch between inline micro-tasks and delegated execution.

        Dispatch is intentionally cheap: mapping payloads declare their
        intent via ``task_type``; non-mapping payloads fall through to
        delegation with the configured default specialty.
        """
        if isinstance(task_data, Mapping):
            task_type = str(task_data.get("task_type", "delegate")).strip().lower()
            if task_type in self.MICRO_TASK_TYPES:
                return self._execute_micro_task(task_type, task_data)
            return self._handle_delegation(task_data)

        return self._handle_delegation(
            {
                "specialty": self.default_specialty,
                "action": "execute",
                "task_data": {"input_data": task_data},
            }
        )

    # ==================================================================
    # Inline micro-tasks
    # ==================================================================
    def _execute_micro_task(self, task_type: str, task_data: Mapping[str, Any]) -> Dict[str, Any]:
        """Route inline micro-tasks to their dedicated handlers.

        Micro-tasks are pure transformations; their results are cached by
        input fingerprint to avoid recomputation on identical requests.
        Caching is safe because each handler is deterministic.
        """
        try:
            if task_type == "normalize_payload":
                payload = task_data.get("payload")
                cache_key = f"micro:normalize:{hash(repr(payload))}"
                return self.cache.get_or_set(cache_key, lambda: self._micro_normalize_payload(payload))
            if task_type == "sanitize_payload":
                payload = task_data.get("payload")
                cache_key = f"micro:sanitize:{hash(repr(payload))}"
                return self.cache.get_or_set(cache_key, lambda: self._micro_sanitize_payload(payload))
            if task_type == "validate_schema":
                return self._micro_validate_schema(task_data.get("schema"), task_data.get("data"))
        except Exception as exc:
            self._mark_runtime_degraded("execution", f"micro_task.{task_type}", exc)
            logger.warning("Micro-task '%s' failed: %s", task_type, exc)
            return {
                "status": "failed",
                "task_type": task_type,
                "error": f"{type(exc).__name__}: {exc}",
            }

        # Defensive: unreachable unless MICRO_TASK_TYPES is extended
        # without a matching handler.
        return {
            "status": "failed",
            "task_type": task_type,
            "error": "unsupported micro-task type",
        }

    def _micro_normalize_payload(self, payload: Any) -> Dict[str, Any]:
        """Lower-case, strip, and drop ``None`` values from a mapping.

        Intended for lightweight structural shaping ahead of specialist
        dispatch: it does not coerce types, does not recurse, and does
        not reorder keys.
        """
        raw = require_mapping(payload, "payload", allow_empty=True)
        return {
            "status": "success",
            "normalized_data": {
                str(k).strip().lower(): v for k, v in raw.items() if v is not None
            },
            "timestamp": time.time(),
        }

    def _micro_sanitize_payload(self, payload: Any) -> Dict[str, Any]:
        """Structural sanitisation of a mapping payload.

        Applies four guarantees that are safe and reversible at the
        orchestrator boundary:

        * non-mapping inputs degrade to an empty result;
        * string values are stripped of surrounding whitespace;
        * control characters are removed from string values;
        * a strict bound is applied to string length, using a
          deterministic marker so downstream consumers can detect the
          truncation.
        """
        _MAX_STRING_LEN = 4096
        if not isinstance(payload, Mapping):
            return {
                "status": "success",
                "sanitized_data": {},
                "dropped_fields": [],
                "timestamp": time.time(),
            }

        sanitized: Dict[str, Any] = {}
        dropped: list[str] = []
        for raw_key, value in payload.items():
            key = str(raw_key).strip()
            if not key:
                dropped.append(repr(raw_key))
                continue
            if isinstance(value, str):
                cleaned = "".join(ch for ch in value if ch >= " " or ch in "\t\n\r").strip()
                if len(cleaned) > _MAX_STRING_LEN:
                    cleaned = cleaned[:_MAX_STRING_LEN] + "…[truncated]"
                sanitized[key] = cleaned
            else:
                sanitized[key] = value

        return {
            "status": "success",
            "sanitized_data": sanitized,
            "dropped_fields": dropped,
            "timestamp": time.time(),
        }

    def _micro_validate_schema(self, schema: Any, data: Any) -> Dict[str, Any]:
        """Structural key-presence validation against a schema.

        The schema is interpreted as a sequence of required keys. This
        is deliberately weaker than a full JSON-schema validator: it is
        intended as a fast pre-flight gate, not as a normative
        contract checker.
        """
        required_keys = (list(schema) if isinstance(schema, (list, tuple, set)) else [])
        payload = dict(data) if isinstance(data, Mapping) else {}
        missing = [k for k in required_keys if k not in payload]
        return {
            "status": "success" if not missing else "failed",
            "valid": not missing,
            "missing_keys": missing,
            "checked_keys": list(required_keys),
        }

    # ==================================================================
    # Delegation
    # ==================================================================
    def _handle_delegation(self, task_data: Mapping[str, Any]) -> Any:
        """Drive the delegation lifecycle for a single request.

        The method owns the full lifecycle: registry inspection, dynamic
        onboarding, telemetry, retry, and resource release. All failure
        modes converge on a structured result so callers never see a
        raised exception from the delegation boundary.
        """
        specialty = validate_agent_name(task_data.get("specialty", self.default_specialty))
        action = str(task_data.get("action", "execute")).strip() or "execute"
        params = task_data.get("task_data", task_data.get("params", task_data))
        module_path = task_data.get("module_path")
        class_name = task_data.get("class_name")

        # 1. Ensure the target is registered. Onboarding failures are
        #    surfaced as structured results because delegation cannot
        #    succeed without a target.
        try:
            self._inspect_and_register_sub_agent(specialty, module_path, class_name, specialist_spec=task_data.get("specialist_spec"))
        except FactoryError as fe:
            fe.log()
            return fe.to_result(agent=self.name, action=action)
        except Exception as exc:
            wrapped = FactoryError.from_exception(
                exc,
                operation="register_sub_agent",
                component=self.name,
                context={"specialty": specialty},
            )
            wrapped.log()
            return wrapped.to_result(agent=self.name, action=action)

        self._broadcast_delegation_start( specialty, action)
        result = self._delegate_with_retry(specialty, action, params)

        # 4. Broadcast outcome.
        self._broadcast_delegation_result(specialty, action, result)
        return result


    def _delegate_once(self, specialty: str, action: str, params: Any) -> Any:
        """Acquire, invoke, and conditionally release one specialist."""
        if self.agent_factory is None:
            raise RuntimeError("AgentFactory is required for delegation")

        if not isinstance(params, Mapping):
            raise TypeError("Delegation params must be a mapping")

        normalized = self.agent_factory.normalize_agent_type(specialty)
        core_types = set(getattr(self.agent_factory, "DEFAULT_AGENT_SPECS", {}))

        # Core agents remain factory-owned and reusable.
        # Generated specialists may be explicitly ephemeral.
        ephemeral = self.ttl_recycling and normalized not in core_types
        agent = self.agent_factory.create(normalized, self.shared_memory, _ephemeral=ephemeral)

        try:
            method = getattr(agent, action, None)
            if not callable(method):
                raise AttributeError(
                    f"Specialist '{normalized}' has no "
                    f"callable action '{action}'"
                )

            return method(**dict(params))

        finally:
            if ephemeral:
                self._release_specialty(normalized, agent)

    def _delegate_with_retry(self, specialty: str, action: str, params: Any) -> Any:
        """Invoke ``call_external`` with bounded retry and structured errors.

        Retries are only attempted for :class:`FactoryError` instances
        that explicitly declare themselves retryable, or for
        non-``FactoryError`` exceptions that are wrapped and then
        re-checked. Backoff uses a linear ramp scaled by the configured
        ``retry_backoff_seconds``.
        """
        last_error: Optional[FactoryError] = None
        for attempt in range(self.max_delegation_retries + 1):
            try:
                return self._delegate_once(specialty, action, params)
            except FactoryError as fe:
                last_error = fe
                if not fe.retryable or attempt >= self.max_delegation_retries:
                    break
            except Exception as exc:
                last_error = FactoryError.from_exception(
                    exc,
                    operation=action,
                    component=self.name,
                    context={"specialty": specialty, "attempt": attempt + 1},
                )
                if attempt >= self.max_delegation_retries:
                    break

            # Linear backoff between attempts.
            delay = self.retry_backoff_seconds * (attempt + 1)
            if delay > 0:
                time.sleep(delay)

        assert last_error is not None  # guaranteed by loop construction
        last_error.log()
        printer.status(
            "ERROR",
            f"Delegation to '{specialty}' failed: {last_error.message}",
            "error",
        )
        return last_error.to_result(agent=self.name, action=action)


    def _release_specialty(self, specialty: str, instance: Any) -> None:
        """Release only an explicitly ephemeral acquired instance.

        Never release shared core agents or another scope's
        specialist through this delegation cleanup path.
        """
        if not (
            self.ttl_recycling
            and self.agent_factory is not None
        ):
            return

        try:
            released = self.agent_factory.release_instance(
                instance,
                ephemeral_only=True,
            )

        except Exception as exc:
            self._mark_runtime_degraded(
                "lifecycle",
                "release_specialty",
                exc,
                retryable=True,
            )

            logger.warning(
                "Failed to release specialist '%s': %s",
                specialty,
                exc,
            )
            return

        if released:
            logger.debug(
                "Released ephemeral specialist '%s'",
                specialty,
            )
        else:
            logger.warning(
                "Specialist '%s' was not released by "
                "the exact-instance ephemeral release path",
                specialty,
            )

    # ==================================================================
    # Registry inspection and dynamic onboarding
    # ==================================================================
    def _inspect_and_register_sub_agent(
        self,
        specialty: str,
        module_path: Optional[str] = None,
        class_name: Optional[str] = None,
        specialist_spec: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Register a validated existing or newly generated specialist.

        The factory remains authoritative for agent registration.
        Unknown specialties must have a valid executable specification.
        """
        if self.agent_factory is None:
            raise RuntimeError("AgentFactory is required for delegation")

        normalized_type = self.agent_factory.normalize_agent_type(specialty)
        default_specs = (getattr(self.agent_factory, "DEFAULT_AGENT_SPECS", {}) or {})
        registered = self.agent_factory.get_registered_agent_types()

        if normalized_type in default_specs:
            if specialist_spec is not None:
                raise ValueError("Core SLAI agents cannot be redefined as generated specialists")
            return

        if module_path is not None or class_name is not None:
            raise ValueError("Custom implementations must be registered directly through AgentFactory, not through an orchestration task payload")

        with _DYNAMIC_REGISTRATION_LOCK:
            registered = self.agent_factory.get_registered_agent_types()

            if normalized_type in registered:
                if specialist_spec is not None:
                    expected_spec = validate_specialist_spec(specialist_spec)

                    safe_segment = self._safe_module_segment(normalized_type)
                    resolved_module = (f"{self.SUBAGENT_MODULE_PREFIX}.{safe_segment}_agent")

                    module = importlib.import_module(resolved_module)
                    expected_class = self._class_name_for(safe_segment)
                    agent_class = getattr(module, expected_class)

                    if (
                        not issubclass(agent_class, BaseAgent)
                        or getattr(agent_class, "SPECIALIST_SPEC", None)
                        != expected_spec
                    ):
                        raise ValueError("Existing specialist has a conflicting definition")
                return

            if specialist_spec is None:
                raise ValueError("Unknown specialty requires an executable specialist_spec")

            validated_spec = validate_specialist_spec(specialist_spec)
            policy = _SUBAGENT_SETTINGS.load()
            generation = policy.get("generation", {})
            if not isinstance(generation, Mapping):
                raise ValueError("Invalid subagent generation configuration")

            if not generation.get("enabled", False):
                raise ValueError("Dynamic subagent generation is disabled")

            safe_segment = self._safe_module_segment(normalized_type)
            # A dotted or hyphenated specialty can collide with an underscored name.
            # Fail before importing or registering a different generated agent.
            for registered_name in registered:
                if registered_name != normalized_type and self._safe_module_segment(registered_name) == safe_segment:
                    raise ValueError(
                        f"Specialty '{normalized_type}' conflicts with registered specialty '{registered_name}'"
                    )
            if not re.fullmatch(r"[a-z][a-z0-9_]*", safe_segment):
                raise ValueError("Unsafe subagent module identifier")

            if len(safe_segment) > 64:
                raise ValueError("Subagent identifier is too long")

            resolved_module = validate_module_path(f"{self.SUBAGENT_MODULE_PREFIX}.{safe_segment}_agent")
            resolved_class = validate_class_name(self._class_name_for(safe_segment))

            max_definitions = int(generation.get("max_definitions", 256))
            if max_definitions < 1:
                raise ValueError("max_definitions must be positive")

            module_target = self._subagent_dir / f"{safe_segment}_agent.py"

            if not module_target.exists():
                existing_count = sum(1 for path in self._subagent_dir.glob("*_agent.py") if path.is_file()) if self._subagent_dir.exists() else 0
                if existing_count >= max_definitions:
                    raise ValueError("Generated specialist limit reached")

            module_file = self._ensure_subagent_module(safe_segment, resolved_class, validated_spec)

            importlib.invalidate_caches()
            module = importlib.import_module(resolved_module)
            agent_class = getattr(module, resolved_class, None)

            if (
                not isinstance(agent_class, type)
                or not issubclass(agent_class, BaseAgent)
            ):
                raise TypeError("Generated specialist is not a BaseAgent")

            if getattr(agent_class, "SPECIALIST_SPEC", None) != validated_spec:
                raise ValueError("Generated specialist specification mismatch")

            if not callable(getattr(agent_class, "perform_task", None)):
                raise TypeError("Generated specialist has no task handler")

            metadata = AgentMetaData(
                name=normalized_type,
                class_name=resolved_class,
                module_path=resolved_module,
                version=__version__,
                description=(f"Validated generated specialist: {normalized_type}"),
                capabilities=(normalized_type, validated_spec["operation"]),
                tags=("generated", "specialized"),
                lifecycle="registered",
                metadata={
                    "specialist_spec": validated_spec,
                    "source_file": str(module_file),
                },
            )

            self.agent_factory.register_agent(metadata, overwrite=False)

            logger.info(
                "ORCHESTRATOR | Registered executable specialist %s | %s",
                normalized_type,
                validated_spec["operation"],
            )

            self._broadcast_registration(
                normalized_type,
                resolved_module,
                resolved_class,
            )

    # ------------------------------------------------------------------
    # Subagent materialisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_module_segment(specialty: str) -> str:
        """Reduce a specialty identifier to an import-safe path segment.

        ``validate_agent_name`` is the authoritative validator, but the
        subagent directory layout additionally requires a conservative
        filesystem-safe segment. Dots and hyphens are normalised to
        underscores so that the generated filename always matches the
        module path registered with the factory.
        """
        return specialty.replace(".", "_").replace("-", "_")

    @staticmethod
    def _class_name_for(specialty: str) -> str:
        """Derive the canonical class name for a specialty identifier.

        ``foo``       -> ``FooAgent``
        ``foo_bar``   -> ``FooBarAgent``
        ``foo.bar``   -> ``FooBarAgent`` (after segment normalisation)
        """
        parts = [segment for segment in specialty.replace(".", "_").split("_") if segment]
        if not parts:
            return "DynamicAgent"
        return "".join(segment.capitalize() for segment in parts) + "Agent"



    def _ensure_subagent_module(
        self,
        specialty: str,
        class_name: str,
        specialist_spec: Mapping[str, Any],
    ) -> Path:
        """Generate and verify a specialized agent module.

        The workspace is initialized automatically.
        Existing implementations are never silently overwritten.
        """
        with _DYNAMIC_REGISTRATION_LOCK:
            self._initialize_subagent_workspace()
            validated_spec = validate_specialist_spec(specialist_spec)
            module_file = (
                self._subagent_dir
                / f"{specialty}_agent.py"
            )

            rendered = _SUBAGENT_MODULE_TEMPLATE.format(
                specialty=specialty,
                specialty_repr=repr(specialty),
                class_name=class_name,
                specialist_spec=repr(validated_spec),
                operation_repr=repr(
                    validated_spec["operation"]
                ),
            )

            compile(rendered, str(module_file), "exec")
            self._create_file_if_absent(module_file, rendered)
            existing = module_file.read_text(encoding="utf-8")

            if existing != rendered:
                raise ValueError(
                    "Existing specialist definition conflicts "
                    f"with requested implementation: {module_file}"
                )

            logger.info(
                "ORCHESTRATOR | Specialist ready | "
                "specialty=%s | module=%s",
                specialty,
                module_file,
            )

            return module_file

    # ==================================================================
    # Telemetry helpers
    # ==================================================================
    def _broadcast_delegation_start(self, specialty: str, action: str) -> None:
        """Publish the intent to delegate to a specialist."""
        self.broadcast(
            self._coordination_key,
            {
                "status": "delegating",
                "target_specialty": specialty,
                "action": action,
                "orchestrator_id": self.agent_id,
                "timestamp": time.time(),
            },
        )
        if hasattr(self.shared_memory, "publish"):
            try:
                self.shared_memory.publish(
                    "system_events",
                    {
                        "event_type": "task_delegation",
                        "orchestrator_id": self.agent_id,
                        "specialty": specialty,
                        "action": action,
                    },
                )
            except Exception as exc:
                self._mark_runtime_degraded(
                    "telemetry", "publish.delegation_start", exc
                )

    def _broadcast_delegation_result(self, specialty: str, action: str, result: Any) -> None:
        """Publish the outcome of a delegation attempt."""
        status = "success"
        if isinstance(result, Mapping):
            status = str(result.get("status", "success"))

        self.broadcast(
            f"{self._coordination_key}:result",
            {
                "status": status,
                "target_specialty": specialty,
                "action": action,
                "orchestrator_id": self.agent_id,
                "timestamp": time.time(),
            },
        )

    def _broadcast_registration(self, specialty: str, module_path: str, class_name: str) -> None:
        """Publish a dynamic onboarding event."""
        self.broadcast(
            f"{self._registration_prefix}:{specialty}",
            {
                "status": "dynamically_registered",
                "normalized_type": specialty,
                "module_path": module_path,
                "class_name": class_name,
                "orchestrator_id": self.agent_id,
                "timestamp": time.time(),
            },
        )

    # ==================================================================
    # Observability
    # ==================================================================
    def health_check(self) -> Dict[str, Any]:
        """Return a compact orchestrator-specific health payload.

        Extends the base runtime snapshot with configuration echo and
        cache statistics. Cache-stat retrieval is defensive: a failure
        in the cache subsystem must not itself report as a health
        failure of the orchestrator.
        """
        runtime = self.runtime_status()
        lifecycle = str(runtime.get("lifecycle", "unknown"))

        try:
            cache_stats: Dict[str, Any] = self.cache.stats()
        except Exception as exc:
            self._mark_runtime_degraded("persistence", "cache.stats", exc)
            cache_stats = {
                "status": "degraded",
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            self._mark_runtime_recovered("persistence", "cache.stats")

        healthy = (
            lifecycle in {"active", "degraded"}
            and runtime.get("health", "healthy") != "unavailable"
        )

        return {
            "status": "ok" if healthy else "degraded",
            "health": runtime.get("health", "healthy"),
            "agent": self.name,
            "version": __version__,
            "lifecycle": lifecycle,
            "runtime": runtime,
            "config": {
                "default_specialty": self.default_specialty,
                "ttl_recycling": self.ttl_recycling,
                "max_delegation_retries": self.max_delegation_retries,
                "retry_backoff_seconds": self.retry_backoff_seconds,
                "coordination_prefix": self.coordination_prefix,
                "cache_max_size": self.cache_max_size,
                "cache_ttl_seconds": self.cache_ttl_seconds,
                "subagent_dir": str(self._subagent_dir),
            },
            "cache": cache_stats,
            "factory_available": self.agent_factory is not None,
            "micro_task_types": list(self.MICRO_TASK_TYPES),
        }

    # ==================================================================
    # Lifecycle
    # ==================================================================
    def shutdown(self) -> None:
        """Release orchestrator-owned resources.

        Only resources *owned* by the orchestrator are released here: the
        memoisation cache and the lifecycle event. Factory-owned
        specialist instances remain under the factory's lifecycle control
        and are not forcibly torn down by a single orchestrator, because
        peer agents may still hold references to them.
        """
        try:
            self.cache.clear()
        except Exception as exc:
            self._mark_runtime_degraded("persistence", "cache.clear", exc, retryable=False)
            logger.warning("Failed to clear orchestrator cache: %s", exc)

        self._publish_lifecycle_event("orchestrator_shutdown", {"agent_id": self.agent_id})
        logger.info("OrchestratorAgent shutdown complete")


__all__ = ["OrchestratorAgent"]


if __name__ == "__main__":
    print("\n=== Running Orchestrator Agent ===\n")
    printer.status("TEST", "Orchestrator Agent initialized", "info")
    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    factory = AgentFactory()
    shared_memory = SharedMemory()
    orchestrator = OrchestratorAgent(
        shared_memory=shared_memory,
        agent_factory=factory,
        config={
            "cache_max_size": 32,
            "cache_ttl_seconds": 60.0,
            "max_delegation_retries": 1,
            "ttl_recycling": True,
        },
    )

    # ---- Micro-task: normalise ---------------------------------------
    norm = orchestrator.execute(
        {
            "task_type": "normalize_payload",
            "payload": {"  Alpha ": 1, "BETA": None, "gamma": 3},
        }
    )
    assert norm["status"] == "success", norm
    assert norm["normalized_data"] == {"alpha": 1, "gamma": 3}, norm

    # ---- Micro-task: sanitize ----------------------------------------
    san = orchestrator.execute(
        {
            "task_type": "sanitize_payload",
            "payload": {"key": "  hello\x00world  ", "  ": "ignored"},
        }
    )
    assert san["status"] == "success", san
    assert san["sanitized_data"]["key"] == "helloworld", san
    assert san["dropped_fields"], san

    # ---- Micro-task: validate_schema ---------------------------------
    val = orchestrator.execute(
        {
            "task_type": "validate_schema",
            "schema": ["alpha", "beta"],
            "data": {"alpha": 1},
        }
    )
    assert val["valid"] is False, val
    assert val["missing_keys"] == ["beta"], val

    # ---- Registry introspection --------------------------------------
    registered = factory.get_registered_agent_types()
    assert "orchestrator" in registered, registered

    # ---- Health check ------------------------------------------------
    health = orchestrator.health_check()
    assert health["status"] == "ok", health
    assert health["config"]["cache_max_size"] == 32, health

    # ---- Idempotent onboarding for a known default -------------------
    orchestrator._inspect_and_register_sub_agent("reasoning")
    orchestrator._inspect_and_register_sub_agent("reasoning")
    assert "reasoning" in factory.get_registered_agent_types()


    # ---- Dynamic specialist onboarding -------------------------------
    dynamic_specialty = "smoketest_dynamic_specialty"
    specialist_spec = {"operation": "numeric_summary"}
    dynamic_module_file = (
        orchestrator._subagent_dir
        / f"{dynamic_specialty}_agent.py"
    )

    dynamic_module_name = (
        f"{OrchestratorAgent.SUBAGENT_MODULE_PREFIX}"
        f".{dynamic_specialty}_agent"
    )

    try:
        # The Orchestrator creates and registers the specialist.
        orchestrator._inspect_and_register_sub_agent(
            dynamic_specialty,
            specialist_spec=specialist_spec,
        )

        assert dynamic_module_file.is_file()
        assert (orchestrator._subagent_config_path.is_file())
        assert (dynamic_specialty in factory.get_registered_agent_types())

        # Repeated registration must not duplicate the agent.
        orchestrator._inspect_and_register_sub_agent(dynamic_specialty, specialist_spec=specialist_spec)

        # Import the actual generated implementation.
        importlib.invalidate_caches()
        module = importlib.import_module(dynamic_module_name)
        expected_class = orchestrator._class_name_for(dynamic_specialty)
        specialist_class = getattr(module, expected_class)
        specialist = specialist_class(shared_memory=shared_memory, agent_factory=factory)

        # Verify that the specialist performs real work.
        result = specialist.perform_task({"values": [10, 20, 30, 40]})

        assert result["status"] == "success"
        assert result["result"]["count"] == 4
        assert result["result"]["sum"] == 100
        assert result["result"]["mean"] == 25

        printer.status("TEST", "Dynamic specialist execution passed", "success")

    finally:
        if dynamic_specialty in factory.get_registered_agent_types():
            factory.unregister_agent(dynamic_specialty)

        import sys
        sys.modules.pop(dynamic_module_name, None)
        dynamic_module_file.unlink(missing_ok=True)

    # ---- Shutdown ----------------------------------------------------
    orchestrator.shutdown()
    assert orchestrator.cache.is_empty()

    printer.status("TEST", "Orchestrator Agent checks passed", "success")
    print("\n=== Test ran successfully ===\n")