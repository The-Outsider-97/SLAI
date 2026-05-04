from __future__ import annotations

"""
Production-ready registry system for the collaborative agent subsystem.

This module owns the collaborative runtime's agent registration boundary. It
keeps discovery, registration validation, health metadata, instance creation,
capability indexing, shared-memory heartbeats, and registry audit events in one
place while leaving routing, reliability state transitions, and task execution
to their dedicated modules.

Responsibilities
----------------
- Dynamically discover concrete agent classes from configured packages.
- Validate and register agent metadata with version-aware replacement rules.
- Keep capability-based lookup fast and deterministic for TaskRouter.
- Lazily instantiate agents using constructor-aware dependency injection.
- Publish registry heartbeats and registration audit events to SharedMemory.
- Preserve degradation behavior: stale agents remain routable by default so the
  router and reliability manager can perform controlled fallback/retry logic.

Design principles
-----------------
1. Stable public API: existing methods are retained and not renamed.
2. Direct local imports: project-local config, error, helper, and BaseAgent
   imports remain explicit and unwrapped.
3. Helper/error integration: normalization, redaction, audit payloads, shared
   memory wrappers, and collaboration errors come from the collaborative utils.
4. Config-backed behavior: registry tuning belongs in collaborative_config.yaml.
5. Defensive runtime boundaries: import/initialization failures are recorded and
   surfaced through snapshots instead of silently corrupting registry state.
"""

import importlib
import inspect
import pkgutil
import threading
import time

from collections import defaultdict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Type

from .utils.config_loader import load_global_config, get_config_section
from .utils.collaboration_error import *
from .utils.collaborative_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Agent Registry")
printer = PrettyPrinter()


class RegistryEventType(str, Enum):
    """Normalized registry audit event labels."""

    DISCOVERY_STARTED = "discovery_started"
    DISCOVERY_COMPLETED = "discovery_completed"
    MODULE_LOADED = "module_loaded"
    MODULE_FAILED = "module_failed"
    AGENT_REGISTERED = "agent_registered"
    AGENT_REPLACED = "agent_replaced"
    AGENT_SKIPPED = "agent_skipped"
    AGENT_UNREGISTERED = "agent_unregistered"
    AGENT_INSTANTIATED = "agent_instantiated"
    AGENT_INIT_FAILED = "agent_init_failed"
    AGENT_RELOADED = "agent_reloaded"
    REGISTRY_CLEARED = "registry_cleared"


@dataclass(frozen=True)
class AgentRegistrationRecord:
    """Serializable registration metadata used in audit and snapshots."""

    name: str
    capabilities: Tuple[str, ...]
    version: Any
    module: Optional[str] = None
    class_name: Optional[str] = None
    has_instance: bool = False
    registered_at: float = field(default_factory=epoch_seconds)
    source: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = asdict(self)
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class ModuleDiscoveryRecord:
    """Serializable module discovery result."""

    module_name: str
    status: str
    loaded_agents: Tuple[str, ...] = ()
    reason: Optional[str] = None
    duration_ms: Optional[float] = None
    timestamp: float = field(default_factory=epoch_seconds)

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(asdict(self))


@dataclass(frozen=True)
class RegistryIntegrityReport:
    """Health/integrity summary for the registry state."""

    valid: bool
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    agent_count: int = 0
    capability_count: int = 0
    module_failure_count: int = 0
    init_failure_count: int = 0
    timestamp: float = field(default_factory=epoch_seconds)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentRegistry:
    """
    Enhanced registry system with extended capabilities including:
    - Dynamic agent discovery
    - Health monitoring
    - Capability-based routing
    - Versioned registrations

    The class keeps the existing public methods while adding production-grade
    lifecycle, validation, audit, and snapshot helpers around them.
    """

    _module_failures: Dict[str, str] = {}
    _torch_runtime_checked: bool = False
    _torch_available: bool = True

    def __init__(self, shared_memory: Optional[Any] = None, auto_discover: bool = True):
        self.config = load_global_config()
        self.registry_config = get_config_section("registry") or {}
        agent_discovery_config = self.registry_config.get("agent_discovery", {}) or {}
        instantiation_config = self.registry_config.get("instantiation", {}) or {}
        audit_config = self.registry_config.get("audit", {}) or {}

        self._lock = threading.RLock()
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._capability_index: Dict[str, set[str]] = defaultdict(set)
        self._registration_records: Dict[str, AgentRegistrationRecord] = {}
        self._module_discovery_records: Dict[str, ModuleDiscoveryRecord] = {}

        self.shared_memory = shared_memory
        self._version = coerce_float(self.registry_config.get("version"), default=1.8, minimum=0.0)
        self._health_check_interval = coerce_float(
            self.registry_config.get("health_check_interval"),
            default=300.0,
            minimum=1.0,
        )
        self.default_package = str(agent_discovery_config.get("default_package", "src.agents"))
        self.excluded_modules = tuple(str(item).lower() for item in agent_discovery_config.get("excluded_modules", []) or [])
        self.module_suffix = str(agent_discovery_config.get("module_suffix", "_agent"))
        self.skip_module_suffixes = tuple(
            str(item).lower() for item in agent_discovery_config.get("skip_module_suffixes", [".base_agent", ".collaborative_agent"]) or []
        )
        self.require_base_agent_subclass = coerce_bool(
            agent_discovery_config.get("require_base_agent_subclass"),
            default=True,
        )
        self.require_capabilities = coerce_bool(
            self.registry_config.get("require_capabilities"),
            default=True,
        )
        self.allow_duplicate_names = coerce_bool(
            self.registry_config.get("allow_duplicate_names"),
            default=False,
        )
        self.replace_older_versions = coerce_bool(
            self.registry_config.get("replace_older_versions"),
            default=True,
        )
        self.keep_stale_agents_routable = coerce_bool(
            self.registry_config.get("keep_stale_agents_routable"),
            default=True,
        )
        self.cache_instances = coerce_bool(instantiation_config.get("cache_instances"), default=True)
        self.pass_shared_memory = coerce_bool(instantiation_config.get("pass_shared_memory"), default=True)
        self.pass_agent_factory = coerce_bool(instantiation_config.get("pass_agent_factory"), default=True)
        self.agent_factory = instantiation_config.get("agent_factory")
        self.default_constructor_kwargs = normalize_metadata(instantiation_config.get("default_constructor_kwargs") or {}, drop_none=True)
        self.audit_enabled = coerce_bool(audit_config.get("enabled", self.registry_config.get("audit_enabled")), default=True)
        self.audit_key = str(audit_config.get("key", self.registry_config.get("audit_key", "collaboration:registry_events")))
        self.audit_max_events = coerce_int(audit_config.get("max_events", self.registry_config.get("audit_max_events")), default=1000, minimum=1)
        self.heartbeat_status = str(self.registry_config.get("default_heartbeat_status", AgentHealthStatus.ACTIVE.value))
        self._discovered_packages: set[str] = set()
        self._agent_init_failures: Dict[str, str] = {}
        self._agent_last_error: Dict[str, Dict[str, Any]] = {}
        self._agent_instantiation_count: Dict[str, int] = defaultdict(int)
        self._torch_sensitive_modules = set(
            self.registry_config.get(
                "torch_sensitive_modules",
                [
                    "src.agents.adaptive_agent",
                    "src.agents.alignment_agent",
                    "src.agents.evaluation_agent",
                    "src.agents.knowledge_agent",
                    "src.agents.language_agent",
                    "src.agents.learning_agent",
                    "src.agents.perception_agent",
                    "src.agents.planning_agent",
                    "src.agents.qnn_agent",
                    "src.agents.reasoning_agent",
                    "src.agents.safety_agent",
                ],
            )
            or []
        )

        if auto_discover:
            self.discover_agents(self.default_package)

        self._record_registry_event(
            RegistryEventType.DISCOVERY_COMPLETED.value if auto_discover else "registry_initialized",
            "Agent registry initialized.",
            severity="info",
            metadata={"version": self._version, "auto_discover": auto_discover, "agent_count": len(self._agents)},
        )
        logger.info("Agent Registry Version %s successfully initialized with %s agent(s)", self._version, len(self._agents))

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover_agents(self, agents_package: str = "src.agents") -> None:
        """
        Dynamically discover and register all concrete agent implementations.

        Args:
            agents_package: Python package path to search for agents.

        Raises:
            ImportError: If package cannot be imported.
        """
        package_name = require_non_empty_string(agents_package, "agents_package")
        with self._lock:
            if package_name in self._discovered_packages:
                logger.debug("Agent package %s already discovered for this registry instance; skipping re-scan.", package_name)
                return
            self._record_registry_event(
                RegistryEventType.DISCOVERY_STARTED.value,
                f"Starting agent discovery for package '{package_name}'.",
                metadata={"package": package_name},
            )

        started_ms = monotonic_ms()
        try:
            package = importlib.import_module(package_name)
            if not hasattr(package, "__path__"):
                raise ImportError(f"Package '{package_name}' does not expose __path__ for discovery.")

            for _, module_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
                if not self._should_scan_module(module_name):
                    continue
                if self._should_skip_torch_sensitive_module(module_name):
                    self._module_discovery_records[module_name] = ModuleDiscoveryRecord(
                        module_name=module_name,
                        status="skipped",
                        reason="torch runtime unavailable",
                    )
                    continue
                self._load_agent_module(module_name)

            with self._lock:
                self._discovered_packages.add(package_name)
                self._record_registry_event(
                    RegistryEventType.DISCOVERY_COMPLETED.value,
                    f"Agent discovery completed for package '{package_name}'.",
                    metadata={
                        "package": package_name,
                        "duration_ms": elapsed_ms(started_ms),
                        "agent_count": len(self._agents),
                    },
                )
        except ImportError as exc:
            self._module_failures[package_name] = f"ImportError: {exc}"
            self._record_registry_event(
                RegistryEventType.MODULE_FAILED.value,
                f"Failed to import agents package '{package_name}'.",
                severity="error",
                error=exc,
                metadata={"package": package_name},
            )
            logger.error("Failed to import agents package: %s", exc)
            raise

    def rediscover(self, agents_package: Optional[str] = None) -> None:
        """Clear discovery marker for a package and run discovery again."""
        package_name = agents_package or self.default_package
        with self._lock:
            self._discovered_packages.discard(package_name)
        self.discover_agents(package_name)

    def _should_scan_module(self, module_name: str) -> bool:
        lowered = module_name.lower()
        if any(excluded and excluded in lowered for excluded in self.excluded_modules):
            logger.warning("Skipping %s due to exclusion rule.", module_name)
            self._module_discovery_records[module_name] = ModuleDiscoveryRecord(
                module_name=module_name,
                status="skipped",
                reason="excluded_by_config",
            )
            return False
        if any(lowered.endswith(suffix) for suffix in self.skip_module_suffixes):
            logger.debug("Skipping %s due to registry skip suffix.", module_name)
            return False
        if self.module_suffix and not lowered.endswith(self.module_suffix):
            logger.debug("Skipping non-plugin module %s during discovery.", module_name)
            return False
        return True

    def _should_skip_torch_sensitive_module(self, module_name: str) -> bool:
        if module_name not in self._torch_sensitive_modules:
            return False
        if not self._torch_runtime_checked:
            self._torch_runtime_checked = True
            try:
                importlib.import_module("torch")
                self._torch_available = True
            except Exception as exc:  # optional third-party runtime check
                self._torch_available = False
                logger.warning(
                    "Torch runtime unavailable during collaborative discovery; skipping torch-sensitive modules. Cause: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        if self._torch_available:
            return False
        logger.debug("Skipping %s because torch runtime is unavailable.", module_name)
        self._module_failures[module_name] = "Skipped: torch runtime unavailable"
        return True

    def _load_agent_module(self, module_name: str) -> None:
        """Internal method to load and validate agent modules."""
        module_name = require_non_empty_string(module_name, "module_name")
        if module_name in self._module_failures:
            logger.debug("Skipping module %s after cached import failure: %s", module_name, self._module_failures[module_name])
            return

        started_ms = monotonic_ms()
        loaded: List[str] = []
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if not self._is_discoverable_agent_class(module, name, obj):
                    continue
                caps = normalize_capabilities(getattr(obj, "capabilities", []))
                if self.require_capabilities and not caps:
                    logger.debug("Skipping class %s from %s: no capabilities declared.", name, module_name)
                    continue
                meta = {
                    "class": obj,
                    "instance": None,
                    "capabilities": list(caps),
                    "version": self._version,
                    "metadata": {
                        "source": "discovery",
                        "module": module_name,
                        "class_name": name,
                    },
                }
                self._register_agent(obj.__name__, meta)
                loaded.append(obj.__name__)

            record = ModuleDiscoveryRecord(
                module_name=module_name,
                status="loaded",
                loaded_agents=tuple(loaded),
                duration_ms=elapsed_ms(started_ms),
            )
            self._module_discovery_records[module_name] = record
            self._record_registry_event(
                RegistryEventType.MODULE_LOADED.value,
                f"Loaded registry module '{module_name}'.",
                metadata=record.to_dict(),
            )
        except Exception as exc:
            self._module_failures[module_name] = f"{type(exc).__name__}: {exc}"
            record = ModuleDiscoveryRecord(
                module_name=module_name,
                status="failed",
                reason=str(exc),
                duration_ms=elapsed_ms(started_ms),
            )
            self._module_discovery_records[module_name] = record
            self._record_registry_event(
                RegistryEventType.MODULE_FAILED.value,
                f"Failed to load registry module '{module_name}'.",
                severity="error",
                error=exc,
                metadata=record.to_dict(),
            )
            logger.error("Failed to load module %s: %s", module_name, exc)

    def _is_discoverable_agent_class(self, module: ModuleType, name: str, obj: Type[Any]) -> bool:
        if obj.__module__ != module.__name__:
            return False
        if inspect.isabstract(obj):
            return False
        # Lazy import BaseAgent to avoid circular dependency
        from ..base_agent import BaseAgent
        if obj is BaseAgent or not name.endswith("Agent"):
            return False
        if self.require_base_agent_subclass and not issubclass(obj, BaseAgent):
            return False
        return True

    # ------------------------------------------------------------------
    # Registration and validation
    # ------------------------------------------------------------------
    def _register_agent(self, name: str, meta: Dict) -> None:
        """Validate and register an agent with version control."""
        started_ms = monotonic_ms()
        try:
            normalized_name = normalize_agent_name(name)
            normalized_meta = self._normalize_registration_meta(normalized_name, meta)
            incoming_version = normalized_meta.get("version", self._version)

            with self._lock:
                existing = self._agents.get(normalized_name)
                if existing is not None and not self._should_replace_existing(normalized_name, existing, incoming_version):
                    self._record_registry_event(
                        RegistryEventType.AGENT_SKIPPED.value,
                        f"Skipped registration for existing agent '{normalized_name}'.",
                        agent_name=normalized_name,
                        metadata={"incoming_version": incoming_version, "existing_version": existing.get("version")},
                    )
                    return

                action = RegistryEventType.AGENT_REPLACED.value if existing is not None else RegistryEventType.AGENT_REGISTERED.value
                self._agents[normalized_name] = normalized_meta
                self._agent_init_failures.pop(normalized_name, None)
                self._agent_last_error.pop(normalized_name, None)
                self._rebuild_capability_index_locked()
                self._registration_records[normalized_name] = self._build_registration_record(normalized_name, normalized_meta)

                self._publish_agent_heartbeat(normalized_name, normalized_meta, status=self.heartbeat_status)
                self._record_registry_event(
                    action,
                    f"Registered agent '{normalized_name}' with capabilities {normalized_meta['capabilities']}.",
                    agent_name=normalized_name,
                    metadata={
                        "duration_ms": elapsed_ms(started_ms),
                        "version": incoming_version,
                        "capabilities": normalized_meta.get("capabilities", []),
                    },
                )
                logger.info("Registered agent: %s with capabilities: %s", normalized_name, normalized_meta["capabilities"])
        except Exception as exc:
            error = self._registration_error(
                f"Failed to register agent '{name}': {exc}",
                context={"agent_name": name, "meta": sanitize_for_logging(meta)},
                cause=exc,
            )
            logger.error("%s", error)
            raise error from exc

    def register(self, agent_name: str, agent_instance: Any = None, capabilities: Optional[Iterable[Any]] = None, **metadata: Any) -> None:
        """Register an agent instance/class through the public registry API."""
        registration = build_agent_registration(
            agent_name,
            agent_instance=agent_instance,
            capabilities=capabilities,
            version=metadata.pop("version", self._version),
            metadata=metadata or None,
        )
        self._register_agent(registration["name"], registration["meta"])

    def register_instance(self, agent_name: str, instance: Any, capabilities: Optional[Iterable[Any]] = None, **metadata: Any) -> None:
        """Explicit alias for registering a ready-to-use agent instance."""
        self.register(agent_name, agent_instance=instance, capabilities=capabilities, **metadata)

    def batch_register(self, agents: List[Dict[str, Any]]) -> None:
        """Bulk register pre-configured agents."""
        registrations = build_agent_batch_registrations(agents)
        errors: List[str] = []
        for agent in registrations:
            try:
                self._register_agent(agent["name"], agent["meta"])
            except Exception as exc:
                errors.append(f"{agent.get('name', '<unknown>')}: {exc}")
        if errors:
            raise self._registration_error(
                "One or more agent registrations failed.",
                context={"errors": errors, "count": len(errors)},
            )

    def _normalize_registration_meta(self, name: str, meta: Mapping[str, Any]) -> Dict[str, Any]:
        source = ensure_mapping(meta, field_name="meta")
        agent_class = source.get("class")
        instance = source.get("instance")
        if agent_class is None and instance is not None:
            agent_class = type(instance)
        if agent_class is None:
            raise ValueError("agent meta must include 'class' or 'instance'.")
    
        if inspect.isclass(agent_class):
            # Lazy import BaseAgent to avoid circular dependency
            from ..base_agent import BaseAgent
            if self.require_base_agent_subclass and not issubclass(agent_class, BaseAgent):
                # Existing tests and manual batch registration sometimes use simple
                # objects. Allow explicit instances, but discovery remains strict.
                if instance is None:
                    raise TypeError(f"{agent_class.__name__} must inherit from BaseAgent.")
            if inspect.isabstract(agent_class) and instance is None:
                raise TypeError(f"{agent_class.__name__} is abstract and cannot be instantiated by the registry.")

        capabilities = normalize_capabilities(source.get("capabilities") or extract_agent_capabilities(source or instance or agent_class))
        if self.require_capabilities and not capabilities:
            raise ValueError(f"Agent '{name}' must declare at least one capability.")

        required_attrs = ("execute",)
        target = instance if instance is not None else agent_class
        missing = [attr for attr in required_attrs if not hasattr(target, attr)]
        if missing:
            raise AttributeError(f"Agent '{name}' missing required attribute(s): {', '.join(missing)}")

        normalized = normalize_agent_meta(
            name,
            source,
            capabilities=capabilities,
            version=source.get("version", self._version),
            instance=instance,
            agent_class=agent_class,
            metadata=merge_mappings(
                source.get("metadata"),
                {
                    "registered_by": "AgentRegistry",
                    "module": getattr(agent_class, "__module__", None),
                    "class_name": getattr(agent_class, "__name__", type(agent_class).__name__),
                },
                deep=True,
                drop_none=True,
            ),
        )
        normalized.setdefault("registered_at", epoch_seconds())
        normalized.setdefault("status", AgentHealthStatus.ACTIVE.value)
        return normalized

    def _should_replace_existing(self, name: str, existing: Mapping[str, Any], incoming_version: Any) -> bool:
        if self.allow_duplicate_names:
            return True
        if not self.replace_older_versions:
            logger.warning("Skipping duplicate registration for %s", name)
            return False
        existing_version = coerce_float(existing.get("version"), default=0.0)
        incoming = coerce_float(incoming_version, default=0.0)
        if existing_version > incoming:
            logger.warning("Skipping older version of %s", name)
            return False
        return True

    def _build_registration_record(self, name: str, meta: Mapping[str, Any]) -> AgentRegistrationRecord:
        agent_class = meta.get("class")
        metadata = ensure_mapping(meta.get("metadata"), field_name="metadata", allow_none=True)
        return AgentRegistrationRecord(
            name=name,
            capabilities=tuple(normalize_capabilities(meta.get("capabilities"))),
            version=meta.get("version"),
            module=metadata.get("module") or getattr(agent_class, "__module__", None),
            class_name=metadata.get("class_name") or getattr(agent_class, "__name__", None),
            has_instance=meta.get("instance") is not None,
            registered_at=coerce_float(meta.get("registered_at"), default=epoch_seconds(), minimum=0.0),
            source=str(metadata.get("source", "manual")),
            metadata=normalize_metadata(metadata, drop_none=True),
        )

    def unregister(self, name: str) -> None:
        """Safely remove an agent from the registry."""
        normalized_name = normalize_agent_name(name)
        with self._lock:
            if normalized_name not in self._agents:
                logger.warning("Attempted to unregister unknown agent: %s", normalized_name)
                return
            del self._agents[normalized_name]
            self._registration_records.pop(normalized_name, None)
            self._agent_init_failures.pop(normalized_name, None)
            self._agent_last_error.pop(normalized_name, None)
            self._agent_instantiation_count.pop(normalized_name, None)
            self._rebuild_capability_index_locked()
        memory_delete(self.shared_memory, agent_memory_key(normalized_name))
        self._record_registry_event(
            RegistryEventType.AGENT_UNREGISTERED.value,
            f"Unregistered agent '{normalized_name}'.",
            agent_name=normalized_name,
        )
        logger.info("Unregistered agent: %s", normalized_name)

    def clear(self) -> None:
        """Clear all registered agents and indexes from this registry instance."""
        with self._lock:
            names = list(self._agents)
            self._agents.clear()
            self._registration_records.clear()
            self._agent_init_failures.clear()
            self._agent_last_error.clear()
            self._agent_instantiation_count.clear()
            self._capability_index.clear()
        for name in names:
            memory_delete(self.shared_memory, agent_memory_key(name))
        self._record_registry_event(
            RegistryEventType.REGISTRY_CLEARED.value,
            "Registry cleared.",
            metadata={"agent_count": len(names)},
        )

    # ------------------------------------------------------------------
    # Lookup and instantiation
    # ------------------------------------------------------------------
    def get_agents_by_task(self, task_type: str) -> Dict[str, Dict]:
        """
        Find agents supporting a specific task type with health checks.

        Args:
            task_type: Task identifier to match against capabilities.

        Returns:
            Dictionary of qualified agents with their metadata.
        """
        task = normalize_task_type(task_type)
        qualified: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            candidate_names = set(self._capability_index.get(task, set()))
            if not candidate_names:
                # Preserve compatibility with case-sensitive custom capabilities
                # while allowing normalized lookups to still work.
                candidate_names = {
                    name for name, agent in self._agents.items()
                    if can_agent_handle_task(agent, task)
                }
            ordered_names = sorted(candidate_names)

        for name in ordered_names:
            with self._lock:
                agent = self._agents.get(name)
            if not agent:
                continue
            if not self._check_agent_health(name):
                continue
            instance = self._get_or_create_instance(name)
            if instance is None:
                continue
            qualified[name] = {
                "instance": instance,
                "capabilities": list(agent.get("capabilities", [])),
                "version": agent.get("version"),
                "metadata": dict(agent.get("metadata", {}) or {}),
            }
        return qualified

    def get_agent_meta(self, name: str, *, include_instance: bool = True) -> Optional[Dict[str, Any]]:
        """Return registry metadata for one agent."""
        normalized_name = normalize_agent_name(name)
        with self._lock:
            meta = self._agents.get(normalized_name)
            if meta is None:
                return None
            result = dict(meta)
        if not include_instance:
            result.pop("instance", None)
        return result

    def get_agent_class(self, name: str) -> Optional[Type[Any]]:
        meta = self.get_agent_meta(name, include_instance=False)
        cls = meta.get("class") if meta else None
        return cls if inspect.isclass(cls) else None

    def get_agent_instance(self, name: str) -> Any:
        """Public instance accessor that uses lazy creation when needed."""
        return self._get_or_create_instance(name)

    def _get_or_create_instance(self, name: str):
        normalized_name = normalize_agent_name(name)
        with self._lock:
            agent = self._agents.get(normalized_name)
            if not agent:
                return None
            if self.cache_instances and agent.get("instance") is not None:
                return agent["instance"]
            if normalized_name in self._agent_init_failures:
                logger.debug("Skipping agent %s after cached initialization failure: %s", normalized_name, self._agent_init_failures[normalized_name])
                return None
            cls = agent.get("class")

        if cls is None or not inspect.isclass(cls):
            self._agent_init_failures[normalized_name] = "Missing concrete class for instantiation."
            return None

        started_ms = monotonic_ms()
        try:
            instance = self._instantiate_agent(normalized_name, cls, agent)
            with self._lock:
                if self.cache_instances:
                    self._agents[normalized_name]["instance"] = instance
                self._agent_instantiation_count[normalized_name] += 1
            self._publish_agent_heartbeat(normalized_name, agent, status=AgentHealthStatus.ACTIVE.value)
            self._record_registry_event(
                RegistryEventType.AGENT_INSTANTIATED.value,
                f"Instantiated agent '{normalized_name}'.",
                agent_name=normalized_name,
                metadata={"duration_ms": elapsed_ms(started_ms), "class": cls.__name__},
            )
            return instance
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._agent_init_failures[normalized_name] = error
                self._agent_last_error[normalized_name] = exception_to_error_payload(exc, action="instantiate_agent").get("error", {})
            self._record_registry_event(
                RegistryEventType.AGENT_INIT_FAILED.value,
                f"Agent '{normalized_name}' is unavailable for execution.",
                severity="warning",
                agent_name=normalized_name,
                error=exc,
                metadata={"duration_ms": elapsed_ms(started_ms)},
            )
            logger.warning("Agent %s is unavailable for execution: %s", normalized_name, error)
            return None

    def _instantiate_agent(self, name: str, cls: Type[Any], meta: Mapping[str, Any]) -> Any:
        signature = inspect.signature(cls.__init__)
        parameters = signature.parameters
        kwargs: Dict[str, Any] = {}
        kwargs.update(dict(self.default_constructor_kwargs))
        constructor_kwargs = ensure_mapping(meta.get("constructor_kwargs"), field_name="constructor_kwargs", allow_none=True)
        kwargs.update(constructor_kwargs)

        if self.pass_shared_memory and "shared_memory" in parameters:
            kwargs.setdefault("shared_memory", self.shared_memory)
        if self.pass_agent_factory and "agent_factory" in parameters:
            kwargs.setdefault("agent_factory", self.agent_factory)
        if "config" in parameters and "config" not in kwargs:
            agent_config = ensure_mapping(meta.get("config"), field_name="config", allow_none=True)
            if agent_config:
                kwargs["config"] = agent_config

        # Drop kwargs unsupported by constructors that do not accept **kwargs.
        accepts_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
        if not accepts_var_kw:
            allowed = {
                key for key, param in parameters.items()
                if key != "self" and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            kwargs = {key: value for key, value in kwargs.items() if key in allowed}
        return cls(**kwargs)

    def _check_agent_health(self, name: str) -> bool:
        """Perform health check on registered agent."""
        normalized_name = normalize_agent_name(name)
        with self._lock:
            agent = self._agents.get(normalized_name)
        if not agent:
            return False

        heartbeat = memory_get(self.shared_memory, agent_memory_key(normalized_name), default={}) or {}
        if isinstance(heartbeat, Mapping):
            last_seen = coerce_float(heartbeat.get("last_seen"), default=0.0, minimum=0.0)
            if last_seen and (epoch_seconds() - last_seen) > self._health_check_interval:
                logger.warning("Agent %s appears unresponsive", normalized_name)
                self._publish_agent_heartbeat(
                    normalized_name,
                    agent,
                    status=AgentHealthStatus.STALE.value,
                    metadata={"stale_after_seconds": self._health_check_interval},
                )
                return bool(self.keep_stale_agents_routable)
        return True

    def update_agent_status(self, name: str, status: str, *, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Update one agent heartbeat/status record."""
        normalized_name = normalize_agent_name(name)
        with self._lock:
            meta = self._agents.get(normalized_name, {})
        heartbeat = self._publish_agent_heartbeat(normalized_name, meta, status=status, metadata=metadata)
        return heartbeat

    def mark_unavailable(self, name: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """Mark an agent heartbeat as unavailable without unregistering it."""
        return self.update_agent_status(name, AgentHealthStatus.UNAVAILABLE.value, metadata={"reason": reason})

    def reload_agent(self, name: str) -> bool:
        """Reload an agent module and update registration."""
        normalized_name = normalize_agent_name(name)
        with self._lock:
            agent = self._agents.get(normalized_name)
        if not agent:
            logger.error("Agent %s not found for reload", normalized_name)
            return False

        try:
            module = inspect.getmodule(agent["class"])
            if module:
                importlib.reload(module)
                with self._lock:
                    self._agent_init_failures.pop(normalized_name, None)
                    self._agents[normalized_name]["instance"] = None
                self._load_agent_module(module.__name__)
                self._record_registry_event(
                    RegistryEventType.AGENT_RELOADED.value,
                    f"Reloaded agent '{normalized_name}'.",
                    agent_name=normalized_name,
                    metadata={"module": module.__name__},
                )
                return True
        except Exception as exc:
            self._agent_last_error[normalized_name] = exception_to_error_payload(exc, action="reload_agent").get("error", {})
            self._record_registry_event(
                RegistryEventType.MODULE_FAILED.value,
                f"Failed to reload agent '{normalized_name}'.",
                severity="error",
                agent_name=normalized_name,
                error=exc,
            )
            logger.error("Failed to reload agent %s: %s", normalized_name, exc)
        return False

    # ------------------------------------------------------------------
    # Introspection and snapshots
    # ------------------------------------------------------------------
    def list_agents(self) -> Dict[str, List[str]]:
        """Get comprehensive agent list with capabilities."""
        with self._lock:
            return {name: list(info.get("capabilities", [])) for name, info in sorted(self._agents.items())}

    def list_agent_metadata(self, *, include_instances: bool = False, redact: bool = True) -> Dict[str, Dict[str, Any]]:
        """Return full registry metadata for all agents."""
        from typing import cast
        with self._lock:
            payload: Dict[str, Dict[str, Any]] = {}
            for name, info in sorted(self._agents.items()):
                row = dict(info)
                if not include_instances:
                    row.pop("instance", None)
                    if inspect.isclass(row.get("class")):
                        row["class"] = f"{row['class'].__module__}.{row['class'].__name__}"
                if redact:
                    safe_row = sanitize_for_logging(row)
                    if not isinstance(safe_row, dict):
                        safe_row = {"_error": "sanitization_failed", "original_type": type(row).__name__}
                    payload[name] = cast(Dict[str, Any], safe_row)
                else:
                    safe_row = json_safe(row)
                    if not isinstance(safe_row, dict):
                        safe_row = {"_error": "json_safe_failed", "original_type": type(row).__name__}
                    payload[name] = cast(Dict[str, Any], safe_row)
            return payload

    def get_capabilities_index(self) -> Dict[str, List[str]]:
        """Return capability -> agent names index."""
        with self._lock:
            return {capability: sorted(names) for capability, names in sorted(self._capability_index.items())}

    def list_capabilities(self) -> List[str]:
        """Return sorted unique capabilities known to the registry."""
        return sorted(self.get_capabilities_index())

    def get_module_failures(self) -> Dict[str, str]:
        return dict(self._module_failures)

    def get_init_failures(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._agent_init_failures)

    def get_registration_records(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {name: record.to_dict() for name, record in sorted(self._registration_records.items())}

    def get_discovery_records(self) -> Dict[str, Dict[str, Any]]:
        return {name: record.to_dict() for name, record in sorted(self._module_discovery_records.items())}

    def snapshot(self, *, include_instances: bool = False) -> Dict[str, Any]:
        """Return a JSON-safe registry snapshot."""
        with self._lock:
            agent_names = sorted(self._agents)
            stats = get_agent_stats(self.shared_memory) if self.shared_memory is not None else {}
            heartbeats = read_agent_heartbeats(self.shared_memory, agent_names) if self.shared_memory is not None else {}
            snapshots = build_agent_snapshots(self._agents, stats=stats, heartbeats=heartbeats)
            return {
                "version": self._version,
                "agent_count": len(self._agents),
                "agents": self.list_agent_metadata(include_instances=include_instances),
                "agent_snapshots": snapshots,
                "capabilities": self.get_capabilities_index(),
                "discovered_packages": sorted(self._discovered_packages),
                "module_failures": self.get_module_failures(),
                "init_failures": self.get_init_failures(),
                "registration_records": self.get_registration_records(),
                "discovery_records": self.get_discovery_records(),
                "timestamp": epoch_seconds(),
                "timestamp_utc": utc_timestamp(),
            }

    def health_report(self) -> Dict[str, Any]:
        """Return a collaborative health report for registry consumers."""
        integrity = self.validate_integrity()
        snapshot = self.snapshot(include_instances=False)
        base_report = build_health_report(
            agents=self._agents,
            stats=get_agent_stats(self.shared_memory) if self.shared_memory is not None else {},
            heartbeats=read_agent_heartbeats(self.shared_memory, snapshot["agents"].keys()) if self.shared_memory is not None else {},
            shared_memory=self.shared_memory,
        )
        base_report["component"] = "agent_registry"
        base_report["integrity"] = integrity.to_dict()
        base_report["registry"] = {
            "agent_count": snapshot["agent_count"],
            "capability_count": len(snapshot["capabilities"]),
            "module_failure_count": len(snapshot["module_failures"]),
            "init_failure_count": len(snapshot["init_failures"]),
        }
        if not integrity.valid:
            base_report["status"] = "degraded"
        return base_report

    def validate_integrity(self) -> RegistryIntegrityReport:
        """Validate internal registry invariants."""
        errors: List[str] = []
        warnings: List[str] = []
        with self._lock:
            for name, meta in self._agents.items():
                if not name:
                    errors.append("Encountered empty agent name.")
                capabilities = normalize_capabilities(meta.get("capabilities"))
                if self.require_capabilities and not capabilities:
                    errors.append(f"Agent '{name}' has no capabilities.")
                if meta.get("class") is None and meta.get("instance") is None:
                    errors.append(f"Agent '{name}' has neither class nor instance.")
                if name in self._agent_init_failures:
                    warnings.append(f"Agent '{name}' has cached initialization failure: {self._agent_init_failures[name]}")
            indexed_names = set().union(*self._capability_index.values()) if self._capability_index else set()
            missing_from_index = set(self._agents) - indexed_names if self._agents else set()
            if missing_from_index:
                warnings.append(f"Agents missing from capability index: {sorted(missing_from_index)}")
            return RegistryIntegrityReport(
                valid=not errors,
                errors=tuple(errors),
                warnings=tuple(warnings),
                agent_count=len(self._agents),
                capability_count=len(self._capability_index),
                module_failure_count=len(self._module_failures),
                init_failure_count=len(self._agent_init_failures),
            )

    def export_registry(self, filename: str) -> str:
        """Export a redacted registry snapshot to JSON."""
        return str(export_json_file(filename, self.snapshot(include_instances=False), pretty=True))

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------
    def _rebuild_capability_index_locked(self) -> None:
        self._capability_index.clear()
        for name, meta in self._agents.items():
            for capability in normalize_capabilities(meta.get("capabilities")):
                self._capability_index[capability].add(name)

    def _publish_agent_heartbeat(
        self,
        name: str,
        meta: Mapping[str, Any],
        *,
        status: str = AgentHealthStatus.ACTIVE.value,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.shared_memory is None:
            return {}
        heartbeat = touch_agent_heartbeat(
            self.shared_memory,
            name,
            status=status,
            capabilities=meta.get("capabilities"),
            version=meta.get("version"),
            metadata=merge_mappings(meta.get("metadata"), metadata, deep=True, drop_none=True),
        )
        return heartbeat

    def _record_registry_event(
        self,
        event_type: str,
        message: str,
        *,
        severity: str = "info",
        agent_name: Optional[str] = None,
        error: Optional[BaseException] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.audit_enabled or self.shared_memory is None:
            return None
        event = build_audit_event(
            event_type,
            message,
            severity=severity,
            component="agent_registry",
            agent_name=agent_name,
            error=error,
            metadata=metadata,
        )
        append_audit_event(self.shared_memory, event, key=self.audit_key, max_events=self.audit_max_events)
        return event

    def _registration_error(
        self,
        message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> Exception:
        state = {
            "agent_count": len(self._agents),
            "capabilities": self.get_capabilities_index(),
            "module_failures": self.get_module_failures(),
            "init_failures": self.get_init_failures(),
        }
        cls = globals().get("RegistrationFailureError")
        if isinstance(cls, type) and issubclass(cls, Exception):
            for build in (
                lambda: cls(message=message, context=context, collaborative_agent_state=state, cause=cause), # type: ignore
                lambda: cls(message, context=context, collaborative_agent_state=state), # type: ignore
                lambda: cls(message, context=context), # type: ignore
                lambda: cls(message),
            ):
                try:
                    return build()
                except TypeError:
                    continue
        return CollaborationError(
            CollaborationErrorType.REGISTRATION_FAILURE, # type: ignore
            message,
            severity="high",
            context=dict(context or {}),
            collaborative_agent_state=state,
            remediation_guidance="Validate agent metadata, capabilities, constructor signature, and registry configuration.",
        ) # type: ignore


if __name__ == "__main__":
    print("\n=== Running Registry ===\n")
    printer.status("TEST", "Registry initialized", "info")
    from .shared_memory import SharedMemory

    class TranslationAgent:
        capabilities = ["translation", "language"]

        def __init__(self, shared_memory: Any = None):
            self.shared_memory = shared_memory

        def execute(self, task_data: Dict[str, Any]) -> Any:
            return {"status": "success", "text": f"Translated: {task_data['text']}"}

    class AnalysisAgent:
        capabilities = ["analysis", "data"]

        def execute(self, task_data: Dict[str, Any]) -> Any:
            return {"status": "analyzed", "result": 42, "input": task_data}

    class FailingInitAgent:
        capabilities = ["failing"]

        def __init__(self):
            raise RuntimeError("expected init failure")

        def execute(self, task_data: Dict[str, Any]) -> Any:
            return task_data

    shared_memory = SharedMemory()
    registry = AgentRegistry(shared_memory, auto_discover=False)

    registry.batch_register([
        {
            "name": "Translator",
            "meta": {
                "class": TranslationAgent,
                "instance": TranslationAgent(shared_memory=shared_memory),
                "capabilities": ["translation"],
                "version": 1.0,
                "metadata": {"source": "test"},
            },
        },
        {
            "name": "Analyzer",
            "meta": {
                "class": AnalysisAgent,
                "instance": AnalysisAgent(),
                "capabilities": ["analysis", "data"],
                "version": 1.0,
            },
        },
    ])

    registry.require_base_agent_subclass = False
    registry._register_agent("LazyTranslator", {"class": TranslationAgent, "capabilities": ["translation", "lazy"], "version": 2.0})
    registry._register_agent("Failing", {"class": FailingInitAgent, "capabilities": ["failing"], "version": 1.0})

    agents = registry.list_agents()
    assert "Translator" in agents and "Analyzer" in agents and "LazyTranslator" in agents
    assert "translation" in registry.list_capabilities()

    translators = registry.get_agents_by_task("translation")
    assert "Translator" in translators and "LazyTranslator" in translators
    assert translators["Translator"]["instance"].execute({"text": "Hello"})["text"].startswith("Translated")

    lazy_instance = registry.get_agent_instance("LazyTranslator")
    assert isinstance(lazy_instance, TranslationAgent)
    assert lazy_instance.shared_memory is shared_memory

    failing_instance = registry.get_agent_instance("Failing")
    assert failing_instance is None
    assert "Failing" in registry.get_init_failures()

    heartbeat = registry.update_agent_status("Translator", "active", metadata={"test": True})
    assert heartbeat["status"] == "active"

    index = registry.get_capabilities_index()
    assert "translation" in index and "Translator" in index["translation"]

    integrity = registry.validate_integrity()
    assert integrity.valid, integrity.to_dict()

    snapshot = registry.snapshot()
    assert snapshot["agent_count"] == 4
    assert "agent_snapshots" in snapshot

    report = registry.health_report()
    assert report["component"] == "agent_registry"

    registry.unregister("Analyzer")
    assert "Analyzer" not in registry.list_agents()

    registry.clear()
    assert registry.list_agents() == {}

    printer.status("TEST", "Registry discovery/registration/lookup/health checks passed", "success")
    print("\n=== Test ran successfully ===\n")
