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

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .base_agent import BaseAgent
from .base.utils.main_config_loader import get_config_section
from .factory.utils.factory_errors import FactoryError
from .factory.utils.factory_helpers import *
from .factory.agent_meta_data import AgentMetaData
from .factory.factory_cache import FactoryCache
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Orchestrator Agent")
printer = PrettyPrinter()


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
Auto-generated specialist subagent for the ``{specialty}`` specialty.

Produced by :class:`OrchestratorAgent` on {timestamp}.

Generation is guarded by file existence: edits to this file are preserved
across future orchestration passes. Delete this file to force regeneration.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..base_agent import BaseAgent
from ..base.utils.main_config_loader import get_config_section
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("{class_name}")


class {class_name}(BaseAgent):
    """Auto-generated specialist for the ``{specialty}`` specialty.

    Provides a structural default implementation: it honours the standard
    :class:`BaseAgent` constructor contract and returns a structured
    acknowledgement for delegated tasks. Replace this module with a
    hand-written implementation to attach domain-specific behaviour.
    """

    SPECIALTY: str = "{specialty}"
    CAPABILITIES: tuple = ("{specialty}", "dynamic_execution")

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
        self.specialty_config: dict = dict(
            get_config_section("{specialty}_agent") or {{}}
        )
        if config:
            self.specialty_config.update(dict(config))
        logger.info("{class_name} initialized | specialty=%s", self.SPECIALTY)

    def perform_task(self, task_data: Any) -> Any:
        """Structural default handler for delegated work.

        Extracts ``context`` and a task payload from the incoming request
        and returns a structured acknowledgement. Override in a
        hand-written implementation to attach domain-specific behaviour.
        """
        context: Any = None
        payload: Any = task_data
        if isinstance(task_data, Mapping):
            context = task_data.get("context")
            payload = task_data.get(
                "task_data",
                task_data.get("input_data", task_data.get("payload", task_data)),
            )
        return {{
            "status": "success",
            "specialty": self.SPECIALTY,
            "capabilities": list(self.CAPABILITIES),
            "result": payload,
            "context": context,
        }}


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
        self._subagent_dir: Path = (Path(__file__).resolve().parent / self.SUBAGENT_DIR_NAME)

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
            self._inspect_and_register_sub_agent(specialty, module_path, class_name)
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

        # 2. Broadcast intent and delegate with retry.
        self._broadcast_delegation_start(specialty, action)
        try:
            result = self._delegate_with_retry(specialty, action, params)
        finally:
            # 3. Always recycle the specialist if TTL recycling is on.
            self._release_specialty(specialty)

        # 4. Broadcast outcome.
        self._broadcast_delegation_result(specialty, action, result)
        return result

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
                return self.call_external(specialty, action, params)
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

    def _release_specialty(self, specialty: str) -> None:
        """Release an ephemeral specialist instance if recycling is on.

        Release failures are logged but never mask the delegation
        outcome. The runtime status reflects the degradation via the
        standard ``lifecycle`` channel so that health checks observe it.
        """
        if not (self.ttl_recycling and self.agent_factory):
            return
        try:
            released = self.agent_factory.release(specialty)
        except Exception as exc:
            self._mark_runtime_degraded(
                "lifecycle", "release_specialty", exc, retryable=True
            )
            logger.warning("Failed to release specialty '%s': %s", specialty, exc)
            return
        if released:
            logger.debug("Released specialty '%s' after delegation", specialty)

    # ==================================================================
    # Registry inspection and dynamic onboarding
    # ==================================================================
    def _inspect_and_register_sub_agent(
        self,
        specialty: str,
        module_path: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> None:
        """Ensure ``specialty`` is registered with the factory.

        The registry is consulted on every call rather than relying on a
        cache entry, because a peer agent may have unregistered the
        target since the last orchestration pass.

        Decision matrix:

        * **Registered** — no-op.
        * **Known factory default, not registered** — leave to the
          factory. Default agents are owned by ``AgentFactory``; the
          orchestrator does not shadow them with generated modules.
        * **Unknown specialty, not registered** — materialise a
          subagent module under ``src/agents/subagents/`` (if not
          already present) and register metadata pointing at it.
        """
        if not self.agent_factory:
            return

        normalized_type = self.agent_factory.normalize_agent_type(specialty)

        if normalized_type in self.agent_factory.get_registered_agent_types():
            return

        default_specs = getattr(self.agent_factory, "DEFAULT_AGENT_SPECS", {}) or {}
        if normalized_type in default_specs:
            logger.info(
                "Specialty '%s' is a known factory default that is not "
                "currently registered; deferring to the factory's own "
                "registration path.",
                normalized_type,
            )
            return

        # ---- Dynamic onboarding of an unknown specialty --------------
        safe_segment = self._safe_module_segment(normalized_type)
        resolved_module = validate_module_path(module_path or f"{self.SUBAGENT_MODULE_PREFIX}.{safe_segment}_agent")
        resolved_class = validate_class_name(class_name or self._class_name_for(safe_segment))

        # Only materialise a file when we own the target directory. If
        # the caller supplied an explicit module path outside the
        # subagents package, we register metadata and trust the caller.
        if resolved_module.startswith(f"{self.SUBAGENT_MODULE_PREFIX}."):
            self._ensure_subagent_module(safe_segment, resolved_class)

        metadata = AgentMetaData(
            name=normalized_type,
            class_name=resolved_class,
            module_path=resolved_module,
            version=__version__,
            description=(
                f"Dynamically onboarded sub-agent for specialty "
                f"'{normalized_type}'."
            ),
            capabilities=(normalized_type, "dynamic_execution"),
            lifecycle="registered",
        )
        self.agent_factory.register_agent(metadata, overwrite=True)

        logger.info(
            "Dynamically registered sub-agent metadata for type '%s' at %s.%s",
            normalized_type,
            resolved_module,
            resolved_class,
        )
        self._broadcast_registration(normalized_type, resolved_module, resolved_class)

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

    def _ensure_subagent_module(self, specialty: str, class_name: str) -> Path:
        """Materialise the subagent module on disk if it does not exist.

        Returns the path of the module file. Writing is atomic (temp file
        followed by ``os.replace``) so that a partially-written module is
        never visible to the import system. Existing files are preserved
        unchanged, allowing hand-written implementations to survive
        subsequent orchestration passes.
        """
        module_dir = self._subagent_dir
        module_dir.mkdir(parents=True, exist_ok=True)

        init_file = module_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text(
                '"""Dynamically-onboarded SLAI specialist subagents."""\n',
                encoding="utf-8",
            )

        module_file = module_dir / f"{specialty}_agent.py"
        if module_file.exists():
            logger.debug(
                "Subagent module already present at %s; leaving untouched.",
                module_file,
            )
            return module_file

        rendered = _SUBAGENT_MODULE_TEMPLATE.format(
            specialty=specialty,
            class_name=class_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        fd, temp_path = tempfile.mkstemp(
            prefix=f".{specialty}_agent.",
            suffix=".py.tmp",
            dir=str(module_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, module_file)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        logger.info("Generated subagent module at %s", module_file)
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

    # ---- Dynamic onboarding of an unknown specialty ------------------
    dynamic_specialty = "smoketest_dynamic_specialty"
    dynamic_module_file = (
        orchestrator._subagent_dir / f"{dynamic_specialty}_agent.py"
    )
    try:
        orchestrator._inspect_and_register_sub_agent(dynamic_specialty)
        assert dynamic_module_file.exists(), dynamic_module_file
        assert dynamic_specialty in factory.get_registered_agent_types()

        # Verify the generated module imports cleanly and constructs.
        import importlib

        module = importlib.import_module(
            f"{OrchestratorAgent.SUBAGENT_MODULE_PREFIX}"
            f".{dynamic_specialty}_agent"
        )
        expected_class = orchestrator._class_name_for(dynamic_specialty)
        assert hasattr(module, expected_class), expected_class
    finally:
        # Clean up the generated module and its registration so the
        # repository is left unchanged.
        if dynamic_module_file.exists():
            dynamic_module_file.unlink()
        try:
            factory.unregister_agent(dynamic_specialty)
        except Exception:
            pass

    # ---- Shutdown ----------------------------------------------------
    orchestrator.shutdown()
    assert orchestrator.cache.is_empty()

    printer.status("TEST", "Orchestrator Agent checks passed", "success")
    print("\n=== Test ran successfully ===\n")