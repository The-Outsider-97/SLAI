from __future__ import annotations

__version__ = "2.2.0"

"""
SLAI Agent Factory

Production-ready factory/orchestrator for SLAI agent construction.

Responsibilities
----------------
- load agent-specific factory configuration from agents_config.yaml;
- register validated agent metadata through the factory-core registry;
- resolve aliases, versions, dependency order, and constructor requirements;
- create and cache agent instances with lifecycle/diagnostic tracking;
- integrate with factory cache, factory observability, metrics adaptation, and
  out-of-process isolation without owning their internal subsystem configs;
- keep broad research/reasoning loops outside the factory boundary.

The factory intentionally keeps local project imports direct. Agent
implementations are loaded dynamically from registered metadata because dynamic
resolution is the purpose of this module, not an optional import workaround.
"""

import importlib
import inspect

from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Deque, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Type

from .base.utils.main_config_loader import get_config_section, load_global_config
from .base_agent import BaseAgent
from .factory.agent_meta_data import AgentMetaData, AgentRegistry
from .factory.factory_cache import FactoryCache
from .factory.factory_obs import FactoryObservability
from .factory.metrics_adapter import MetricsAdapter
from .factory.out_of_process_agent import OutOfProcessAgentProxy
from .factory.utils.factory_errors import *
from .factory.utils.factory_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Agent Factory")
printer = PrettyPrinter()


@dataclass(slots=True)
class AgentCreationRecord:
    """Compact lifecycle/audit record for one factory creation attempt."""

    agent_type: str
    status: str
    duration_ms: float
    version: Optional[str] = None
    implementation: str = "in_process"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "version": self.version,
            "implementation": self.implementation,
            "error": self.error,
            "metadata": safe_serialize(self.metadata, redact=True),
        }


@dataclass(slots=True)
class AgentFactoryConfig:
    """Resolved AgentFactory policy from agents_config.yaml."""

    enabled: bool = True
    auto_register_default_agents: bool = True
    strict_base_agent_subclass: bool = False
    use_instance_cache: bool = True
    cache_agent_instances: bool = True
    cache_imports: bool = True
    cache_constructor_signatures: bool = True
    allow_runtime_config_override: bool = True
    allow_unknown_constructor_kwargs: bool = False
    create_dependencies: bool = True
    block_torch_required_when_unavailable: bool = True
    enable_out_of_process_fallback: bool = True
    publish_observability_events: bool = True
    record_creation_history: bool = True
    creation_history_size: int = 100
    diagnostics_import_check: bool = True
    diagnostics_constructor_check: bool = True
    discovery_enabled: bool = False
    discovery_packages: Tuple[str, ...] = ()
    required_action_methods: Tuple[str, ...] = ("predict", "get_action", "act", "perform_task", "execute")
    native_failure_patterns: Tuple[str, ...] = ("winerror 1114", "c10.dll", "torch_cuda.dll", "cuda error")
    out_of_process_fallback_agents: Tuple[str, ...] = ()
    default_version: str = __version__

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AgentFactoryConfig":
        payload = normalize_payload(data)
        history_size = ensure_positive_int(int(payload.get("creation_history_size", 100)), "agent_factory.creation_history_size")

        def _tuple_of_strings(key: str, default: Sequence[str] = ()) -> Tuple[str, ...]:
            raw = payload.get(key, default)
            if raw in (None, ""):
                return tuple()
            if isinstance(raw, str):
                return (raw,)
            if not isinstance(raw, Sequence):
                raise InvalidFactoryConfigurationError(
                    f"agent_factory.{key} must be a sequence of strings",
                    context={"actual_type": type(raw).__name__, "key": key},
                    component="agent_factory",
                    operation="load_config",
                )
            return tuple(require_non_empty_string(item, f"agent_factory.{key}[]") for item in raw)

        return cls(
            enabled=bool(payload.get("enabled", True)),
            auto_register_default_agents=bool(payload.get("auto_register_default_agents", True)),
            strict_base_agent_subclass=bool(payload.get("strict_base_agent_subclass", False)),
            use_instance_cache=bool(payload.get("use_instance_cache", True)),
            cache_agent_instances=bool(payload.get("cache_agent_instances", True)),
            cache_imports=bool(payload.get("cache_imports", True)),
            cache_constructor_signatures=bool(payload.get("cache_constructor_signatures", True)),
            allow_runtime_config_override=bool(payload.get("allow_runtime_config_override", True)),
            allow_unknown_constructor_kwargs=bool(payload.get("allow_unknown_constructor_kwargs", False)),
            create_dependencies=bool(payload.get("create_dependencies", True)),
            block_torch_required_when_unavailable=bool(payload.get("block_torch_required_when_unavailable", True)),
            enable_out_of_process_fallback=bool(payload.get("enable_out_of_process_fallback", True)),
            publish_observability_events=bool(payload.get("publish_observability_events", True)),
            record_creation_history=bool(payload.get("record_creation_history", True)),
            creation_history_size=history_size,
            diagnostics_import_check=bool(payload.get("diagnostics_import_check", True)),
            diagnostics_constructor_check=bool(payload.get("diagnostics_constructor_check", True)),
            discovery_enabled=bool(payload.get("discovery_enabled", False)),
            discovery_packages=_tuple_of_strings("discovery_packages"),
            required_action_methods=_tuple_of_strings("required_action_methods", ("predict", "get_action", "act", "perform_task", "execute")),
            native_failure_patterns=tuple(pattern.lower() for pattern in _tuple_of_strings("native_failure_patterns", ("winerror 1114", "c10.dll", "torch_cuda.dll", "cuda error"))),
            out_of_process_fallback_agents=_tuple_of_strings("out_of_process_fallback_agents"),
            default_version=str(payload.get("default_version", __version__)),
        )


class AgentFactory:
    """Production-ready dynamic factory for SLAI agent orchestration.

    The factory is the integration point for the factory-core registry,
    factory-runtime metrics adapter, factory-isolation out-of-process proxy,
    factory cache, and factory observability. It does not own the internal
    configuration of those subsystems; their settings remain in
    ``src/agents/factory/configs/factory_config.yaml``.
    """

    DEFAULT_AGENT_SPECS: Dict[str, Dict[str, Any]] = {
        "adaptive": {"module_path": "src.agents.adaptive_agent", "class_name": "AdaptiveAgent"},
        "alignment": {"module_path": "src.agents.alignment_agent", "class_name": "AlignmentAgent"},
        "browser": {"module_path": "src.agents.browser_agent", "class_name": "BrowserAgent"},
        "evaluation": {"module_path": "src.agents.evaluation_agent", "class_name": "EvaluationAgent"},
        "execution": {"module_path": "src.agents.execution_agent", "class_name": "ExecutionAgent"},
        "handler": {"module_path": "src.agents.handler_agent", "class_name": "HandlerAgent"},
        "knowledge": {"module_path": "src.agents.knowledge_agent", "class_name": "KnowledgeAgent"},
        "language": {"module_path": "src.agents.language_agent", "class_name": "LanguageAgent"},
        "learning": {"module_path": "src.agents.learning_agent", "class_name": "LearningAgent"},
        "network": {"module_path": "src.agents.network_agent", "class_name": "NetworkAgent"},
        "observability": {"module_path": "src.agents.observability_agent", "class_name": "ObservabilityAgent"},
        "perception": {"module_path": "src.agents.perception_agent", "class_name": "PerceptionAgent"},
        "planning": {"module_path": "src.agents.planning_agent", "class_name": "PlanningAgent"},
        "reader": {"module_path": "src.agents.reader_agent", "class_name": "ReaderAgent"},
        "reasoning": {"module_path": "src.agents.reasoning_agent", "class_name": "ReasoningAgent"},
        "safety": {"module_path": "src.agents.safety_agent", "class_name": "SafetyAgent"},
    }

    DEFAULT_ALIASES: Dict[str, str] = {
        "web": "browser",
        "obs": "observability",
        "net": "network",
    }

    DEFAULT_DEPENDENCY_PROFILES: Dict[str, Dict[str, Any]] = {
        "browser": {"torch_required": False, "notes": "Selenium/browser stack only."},
        "planning": {"torch_required": False, "notes": "Core planning is torch-free; perception heads are optional."},
        "knowledge": {"torch_required": False, "notes": "SBERT/transformers are optional; TF-IDF path can run without torch."},
        "reasoning": {"torch_required": False, "notes": "Rule/probabilistic reasoning core does not require torch."},
        "language": {"torch_required": False, "notes": "Lightweight language mode available without torch."},
        "evaluation": {"torch_required": False, "notes": "Core evaluation works without deep anomaly torch module."},
        "observability": {"torch_required": False, "notes": "Observability orchestration is telemetry-first and torch-free."},
        "network": {"torch_required": False, "notes": "Network transport/routing/reliability stack is torch-free."},
        "safety": {"torch_required": False, "notes": "Safety policy checks run without torch."},
        "learning": {"torch_required": True, "notes": "RL/meta-learning pipelines require torch."},
        "alignment": {"torch_required": True, "notes": "Value embedding model is torch-based."},
        "adaptive": {"torch_required": True, "notes": "Adaptive RL workers are torch-based."},
        "perception": {"torch_required": True, "notes": "Perception encoder/decoder stack is torch-based."},
    }

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.global_config = load_global_config()
        if not isinstance(self.global_config, MutableMapping):
            raise InvalidFactoryConfigurationError(
                "agents_config.yaml must load into a mapping",
                context={"actual_type": type(self.global_config).__name__},
                component="agent_factory",
                operation="load_global_config",
            )

        self.agent_factory_config: Dict[str, Any] = dict(get_config_section("agent_factory") or {})
        if config:
            overrides = normalize_payload(config)
            nested_overrides = normalize_payload(overrides.get("agent_factory")) if isinstance(overrides.get("agent_factory"), Mapping) else overrides
            self.agent_factory_config.update(nested_overrides)
            self.global_config.update(overrides)

        self.settings = AgentFactoryConfig.from_mapping(self.agent_factory_config)
        if not self.settings.enabled:
            raise FactoryConfigurationError(
                "AgentFactory is disabled by configuration",
                component="agent_factory",
                operation="initialize",
            )

        self._lock = RLock()
        self.registry = AgentRegistry()
        cache_signature = inspect.signature(FactoryCache)
        if "name" in cache_signature.parameters:
            self.cache: FactoryCache[str, Any] = FactoryCache(name="agent_factory")
        else:
            self.cache = FactoryCache()
        self.observability = FactoryObservability()
        self._metrics_adapter: Optional[MetricsAdapter] = None
        self.metrics_adapter_status = "not_initialized"

        self.active_agents: Dict[str, Any] = {}
        self.unavailable_agents: Dict[str, str] = {}
        self._import_cache: Dict[str, Any] = {}
        self._constructor_signature_cache: Dict[Type[Any], Dict[str, Any]] = {}
        self.creation_history: Deque[AgentCreationRecord] = deque(maxlen=self.settings.creation_history_size)
        self._torch_probe = self._probe_torch_runtime()

        self._agent_specs = self._load_agent_specs()
        self._agent_aliases = self._load_aliases()
        self._agent_dependency_profiles = self._load_dependency_profiles()
        self._out_of_process_fallback_agents = self._load_oopa_fallback_agents()

        if self.settings.auto_register_default_agents:
            self.register_default_agents()
        if self.settings.discovery_enabled:
            self.discover_agents()

        self._record_event("factory.initialized", {"registered_agents": len(self.registry.agents)})
        logger.info("Agent Factory initialized with %s registered agents", len(self.registry.agents))

    # ------------------------------------------------------------------
    # Config loading and registration
    # ------------------------------------------------------------------
    def _load_agent_specs(self) -> Dict[str, Dict[str, Any]]:
        configured = self.agent_factory_config.get("agent_specs") or self.agent_factory_config.get("default_agent_specs")
        specs = dict(self.DEFAULT_AGENT_SPECS)
        if configured:
            if not isinstance(configured, Mapping):
                raise InvalidFactoryConfigurationError(
                    "agent_factory.agent_specs must be a mapping",
                    context={"actual_type": type(configured).__name__},
                    component="agent_factory",
                    operation="load_agent_specs",
                )
            for name, spec in configured.items():
                agent_name = validate_agent_name(name)
                spec_payload = dict(require_mapping(spec, f"agent_factory.agent_specs[{agent_name}]", allow_empty=False))
                specs[agent_name] = spec_payload
        return specs

    def _load_aliases(self) -> Dict[str, str]:
        aliases = dict(self.DEFAULT_ALIASES)
        configured = self.agent_factory_config.get("aliases") or self.agent_factory_config.get("agent_aliases") or {}
        if not isinstance(configured, Mapping):
            raise InvalidFactoryConfigurationError(
                "agent_factory.aliases must be a mapping",
                context={"actual_type": type(configured).__name__},
                component="agent_factory",
                operation="load_aliases",
            )
        for alias, target in configured.items():
            aliases[validate_agent_name(str(alias))] = validate_agent_name(str(target))
        return aliases

    def _load_dependency_profiles(self) -> Dict[str, Dict[str, Any]]:
        profiles = {name: dict(profile) for name, profile in self.DEFAULT_DEPENDENCY_PROFILES.items()}
        configured = self.agent_factory_config.get("dependency_profiles") or {}
        if not isinstance(configured, Mapping):
            raise InvalidFactoryConfigurationError(
                "agent_factory.dependency_profiles must be a mapping",
                context={"actual_type": type(configured).__name__},
                component="agent_factory",
                operation="load_dependency_profiles",
            )
        for name, profile in configured.items():
            profiles[validate_agent_name(str(name))] = dict(require_mapping(profile, f"dependency_profiles[{name}]"))
        return profiles

    def _load_oopa_fallback_agents(self) -> Tuple[str, ...]:
        configured = self.settings.out_of_process_fallback_agents
        if configured:
            return tuple(validate_agent_name(name) for name in configured)
        defaults = (
            "adaptive",
            "alignment",
            "browser",
            "evaluation",
            "execution",
            "handler",
            "knowledge",
            "language",
            "learning",
            "network",
            "observability",
            "perception",
            "planning",
            "privacy",
            "quality",
            "reader",
            "reasoning",
            "safety",
        )
        return defaults

    def _metadata_constructor_kwargs(self, spec: Mapping[str, Any], name: str) -> Dict[str, Any]:
        require_required_keys(spec, ("module_path", "class_name"), payload_name=f"agent_specs[{name}]")
        version = str(spec.get("version") or self.settings.default_version or __version__)
        payload = {
            "name": name,
            "module_path": spec["module_path"],
            "class_name": spec["class_name"],
            "version": version,
            "dependencies": list(spec.get("dependencies", ()) or ()),
            "required_params": tuple(spec.get("required_params", ()) or ()),
            "description": spec.get("description", ""),
            "author": spec.get("author", "SLAI"),
        }
        # Newer AgentMetaData versions accept richer fields; older versions do not.
        for optional_key in ("tags", "status", "capabilities", "lifecycle", "metadata"):
            if optional_key in spec:
                payload[optional_key] = spec[optional_key]

        signature = inspect.signature(AgentMetaData)
        accepted = set(signature.parameters.keys())
        return {key: value for key, value in payload.items() if key in accepted}

    def register_default_agents(self) -> None:
        for name, spec in self._agent_specs.items():
            metadata = AgentMetaData(**self._metadata_constructor_kwargs(spec, name))
            self.register_agent(metadata, overwrite=True)

    def register_agent(self, metadata: AgentMetaData, *, overwrite: bool = True) -> AgentMetaData:
        if not isinstance(metadata, AgentMetaData):
            raise FactoryTypeError(
                "Can only register objects of type AgentMetaData",
                context={"actual_type": type(metadata).__name__},
                component="agent_factory",
                operation="register_agent",
            )
        with self._lock:
            try:
                signature = inspect.signature(self.registry.register)
                if "overwrite" in signature.parameters:
                    registered = self.registry.register(metadata, overwrite=overwrite)
                else:
                    registered = self.registry.register(metadata)
            except FactoryError:
                raise
            except Exception as exc:
                raise FactoryRegistryError.from_exception(
                    exc,
                    message=f"Failed to register agent '{getattr(metadata, 'name', 'unknown')}'",
                    component="agent_factory",
                    operation="register_agent",
                    context={"agent_name": getattr(metadata, "name", None)},
                ) from exc

            self.cache.set(f"metadata:{metadata.name}:{metadata.version}", metadata)
            self._record_event("agent.registered", {"agent_type": metadata.name, "version": metadata.version})
            return registered if isinstance(registered, AgentMetaData) else metadata

    def unregister_agent(self, agent_type: str, *, version: Optional[str] = None) -> AgentMetaData:
        normalized = self.normalize_agent_type(agent_type)
        with self._lock:
            if not hasattr(self.registry, "unregister"):
                raise FactoryRegistryError(
                    "Current AgentRegistry does not support unregister",
                    component="agent_factory",
                    operation="unregister_agent",
                    context={"agent_type": normalized},
                )
            removed = self.registry.unregister(normalized, version=version)  # type: ignore[attr-defined]
            self.active_agents.pop(normalized, None)
            self.unavailable_agents.pop(normalized, None)
            self.cache.delete(f"instance:{normalized}")
            self._record_event("agent.unregistered", {"agent_type": normalized, "version": version})
            return removed

    def discover_agents(self, packages: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Discover BaseAgent subclasses from configured packages.

        Discovery is intentionally opt-in because importing all agent modules may
        trigger optional dependencies. It registers discovered metadata only when
        imports succeed.
        """
        discovered: Dict[str, Dict[str, Any]] = {}
        package_names = tuple(packages or self.settings.discovery_packages)
        for package_name in package_names:
            package = importlib.import_module(package_name)
            for _, obj in inspect.getmembers(package):
                if inspect.isclass(obj) and issubclass(obj, BaseAgent) and obj is not BaseAgent:
                    name = obj.__name__.removesuffix("Agent").lower()
                    spec = {"module_path": obj.__module__, "class_name": obj.__name__, "version": self.settings.default_version}
                    metadata = AgentMetaData(**self._metadata_constructor_kwargs(spec, name))
                    self.register_agent(metadata, overwrite=True)
                    discovered[name] = spec
        self._record_event("agent.discovery.completed", {"count": len(discovered)})
        return discovered

    # ------------------------------------------------------------------
    # Agent resolution and construction
    # ------------------------------------------------------------------
    def normalize_agent_type(self, agent_type: str) -> str:
        name = validate_agent_name(agent_type, max_length=128)
        resolved = self._agent_aliases.get(name, name)
        if resolved != name:
            logger.info("Resolved agent alias '%s' -> '%s'", name, resolved)
        return resolved

    def _get_metadata(self, agent_type: str, version: Optional[str] = None) -> AgentMetaData:
        normalized = self.normalize_agent_type(agent_type)
        try:
            metadata = self.registry.get(normalized, version=version)
        except FactoryError:
            raise
        except KeyError as exc:
            raise AgentNotRegisteredError(
                f"Agent '{normalized}' is not registered",
                component="agent_factory",
                operation="get_metadata",
                context={"agent_type": normalized, "version": version, "registered_agents": self.get_registered_agent_types()},
                cause=exc,
            ) from exc
        except Exception as exc:
            raise AgentSelectionError.from_exception(
                exc,
                message=f"Failed to select agent metadata for '{normalized}'",
                component="agent_factory",
                operation="get_metadata",
                context={"agent_type": normalized, "version": version},
            ) from exc
        return metadata

    def _probe_torch_runtime(self) -> Dict[str, str]:
        if not bool(self.agent_factory_config.get("probe_torch_runtime", True)):
            return {"status": "disabled"}
        try:
            torch_module = importlib.import_module("torch")
            return {
                "status": "available",
                "version": str(getattr(torch_module, "__version__", "unknown")),
                "cuda": str(getattr(getattr(torch_module, "version", None), "cuda", None) or "cpu"),
            }
        except Exception as exc:
            return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    def get_agent_dependency_report(self, agent_type: str) -> Dict[str, Any]:
        normalized = self.normalize_agent_type(agent_type)
        profile = dict(self._agent_dependency_profiles.get(normalized, {}))
        profile.setdefault("torch_required", False)
        profile["agent_type"] = normalized
        profile["torch_probe"] = dict(self._torch_probe)
        return profile

    def _ensure_agent_available(self, agent_type: str) -> None:
        if agent_type in self.unavailable_agents:
            reason = self.unavailable_agents[agent_type]
            raise AgentInitializationError(
                f"Agent '{agent_type}' is marked unavailable: {reason}",
                component="agent_factory",
                operation="ensure_agent_available",
                context={"agent_type": agent_type, "reason": reason},
            )
        profile = self.get_agent_dependency_report(agent_type)
        if (
            self.settings.block_torch_required_when_unavailable
            and profile.get("torch_required")
            and self._torch_probe.get("status") != "available"
        ):
            reason = f"torch required by design; torch runtime unavailable ({self._torch_probe.get('error', 'unknown')})"
            self.unavailable_agents[agent_type] = reason
            raise AgentInitializationError(
                reason,
                component="agent_factory",
                operation="ensure_agent_available",
                context={"agent_type": agent_type, "profile": profile},
            )

    def _resolve_dependency_order(self, agent_type: str) -> Tuple[str, ...]:
        try:
            load_order = tuple(self.registry.resolve_dependency_tree(agent_type))
        except FactoryError:
            raise
        except Exception as exc:
            raise DependencyResolutionError.from_exception(
                exc,
                message=f"Failed to resolve dependencies for '{agent_type}'",
                component="agent_factory",
                operation="resolve_dependency_order",
                context={"agent_type": agent_type},
            ) from exc
        return load_order

    def _resolve_agent_class(self, metadata: AgentMetaData) -> Type[Any]:
        module_path = validate_module_path(metadata.module_path)
        class_name = validate_class_name(metadata.class_name)
        cache_key = f"module:{module_path}"
        try:
            module = self.cache.get(cache_key) if self.settings.cache_imports else None
            if module is None:
                module = self._import_cache.get(module_path)
            if module is None:
                module = importlib.import_module(module_path)
                self._import_cache[module_path] = module
                if self.settings.cache_imports:
                    self.cache.set(cache_key, module)
            agent_cls = getattr(module, class_name)
        except ModuleNotFoundError as exc:
            raise AgentModuleImportError(
                f"Agent module '{module_path}' could not be imported",
                component="agent_factory",
                operation="resolve_agent_class",
                context={"module_path": module_path, "class_name": class_name},
                cause=exc,
            ) from exc
        except ImportError as exc:
            raise AgentModuleImportError.from_exception(
                exc,
                message=f"Agent module import failed for '{module_path}'",
                component="agent_factory",
                operation="resolve_agent_class",
                context={"module_path": module_path, "class_name": class_name},
            ) from exc
        except AttributeError as exc:
            raise AgentClassResolutionError(
                f"Agent class '{class_name}' not found in '{module_path}'",
                component="agent_factory",
                operation="resolve_agent_class",
                context={"module_path": module_path, "class_name": class_name},
                cause=exc,
            ) from exc
        except Exception as exc:
            if self._is_native_dependency_failure(exc):
                raise
            raise FactoryResolutionError.from_exception(
                exc,
                message=f"Failed to resolve agent class '{class_name}' from '{module_path}'",
                component="agent_factory",
                operation="resolve_agent_class",
                context={"module_path": module_path, "class_name": class_name},
            ) from exc

        if self.settings.strict_base_agent_subclass and not issubclass(agent_cls, BaseAgent):
            raise AgentClassResolutionError(
                f"Agent class '{class_name}' must inherit BaseAgent",
                component="agent_factory",
                operation="resolve_agent_class",
                context={"module_path": module_path, "class_name": class_name, "actual_type": str(agent_cls)},
            )
        return agent_cls

    def _is_native_dependency_failure(self, exc: BaseException) -> bool:
        message = f"{type(exc).__name__}: {exc}".lower()
        return any(pattern in message for pattern in self.settings.native_failure_patterns)

    def _should_use_oopa_fallback(self, agent_type: str, exc: BaseException) -> bool:
        return (
            self.settings.enable_out_of_process_fallback
            and agent_type in set(self._out_of_process_fallback_agents)
            and self._is_native_dependency_failure(exc)
        )

    def _create_oopa_proxy(self, agent_type: str, metadata: AgentMetaData, exc: BaseException) -> OutOfProcessAgentProxy:
        proxy = OutOfProcessAgentProxy(
            agent_type=agent_type,
            module_path=metadata.module_path,
            class_name=metadata.class_name,
            init_error=f"{type(exc).__name__}: {exc}",
        )
        self.degraded_count = getattr(self, "degraded_count", 0) + 1
        self._record_event("agent.oopa_fallback", {"agent_type": agent_type, "error": str(exc)})
        return proxy

    def _get_constructor_signature(self, agent_cls: Type[Any]) -> Dict[str, Any]:
        cache_key = f"signature:{agent_cls.__module__}.{agent_cls.__name__}"
        cached = self.cache.get(cache_key) if self.settings.cache_constructor_signatures else None
        if cached is not None:
            return dict(cached)
        cached = self._constructor_signature_cache.get(agent_cls)
        if cached is not None:
            return cached

        params = inspect.signature(agent_cls.__init__).parameters
        accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
        signature_meta = {"params": params, "accepts_var_kwargs": accepts_var_kwargs}
        self._constructor_signature_cache[agent_cls] = signature_meta
        if self.settings.cache_constructor_signatures:
            self.cache.set(cache_key, signature_meta)
        return signature_meta

    def _agent_config_for(self, agent_type: str, runtime_override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        config_key = f"{agent_type}_agent"
        agent_config = normalize_payload(self.global_config.get(config_key))
        if runtime_override is not None:
            if not self.settings.allow_runtime_config_override:
                raise FactoryConfigurationError(
                    "Runtime config overrides are disabled for AgentFactory",
                    component="agent_factory",
                    operation="agent_config_for",
                    context={"agent_type": agent_type},
                )
            agent_config.update(normalize_payload(runtime_override))
        return agent_config

    def _build_constructor_args(
        self,
        agent_type: str,
        agent_cls: Type[Any],
        shared_memory: Any,
        agent_config: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        signature_meta = self._get_constructor_signature(agent_cls)
        constructor_params = signature_meta["params"]
        accepts_var_kwargs = bool(signature_meta["accepts_var_kwargs"])
        constructor_args: Dict[str, Any] = {}

        if "shared_memory" in constructor_params:
            constructor_args["shared_memory"] = shared_memory
        if "agent_factory" in constructor_params:
            constructor_args["agent_factory"] = self
        if "config" in constructor_params:
            constructor_args["config"] = dict(agent_config)

        unsupported = []
        for key, value in kwargs.items():
            if key in {"config", "version"}:
                continue
            if key in constructor_params or accepts_var_kwargs or self.settings.allow_unknown_constructor_kwargs:
                constructor_args[key] = value
            else:
                unsupported.append(key)

        if unsupported:
            logger.debug("Skipping unsupported constructor args for %s: %s", agent_type, unsupported)
        return constructor_args

    def _verify_action_surface(self, agent_type: str, instance: Any) -> None:
        if not self.settings.required_action_methods:
            return
        if not any(hasattr(instance, method) for method in self.settings.required_action_methods):
            logger.warning(
                "Agent '%s' does not expose standard action methods (%s).",
                agent_type,
                ", ".join(self.settings.required_action_methods),
            )

    def create(self, agent_type: str, shared_memory: Any = None, **kwargs: Any) -> Any:
        """Create or retrieve an agent instance by registered agent type.

        The method preserves backward compatibility with the original factory
        while using the improved registry, cache, error, observability, metrics,
        and isolation subsystems.
        """
        started_ms = monotonic_ms()
        normalized = self.normalize_agent_type(agent_type)
        version = kwargs.get("version")
        self._record_counter("create.requested")

        with self._lock:
            if self.settings.use_instance_cache and normalized in self.active_agents:
                cached = self.active_agents[normalized]
                self._record_counter("create.cache_hit")
                self._record_event("agent.cache_hit", {"agent_type": normalized})
                return cached

            cached_instance = self.cache.get(f"instance:{normalized}") if self.settings.cache_agent_instances else None
            if cached_instance is not None:
                self.active_agents[normalized] = cached_instance
                self._record_counter("create.cache_hit")
                return cached_instance

            self._ensure_agent_available(normalized)
            metadata = self._get_metadata(normalized, version=version)

            try:
                if self.settings.create_dependencies:
                    for dependency in self._resolve_dependency_order(normalized)[:-1]:
                        if dependency not in self.active_agents:
                            self.create(dependency, shared_memory)

                try:
                    agent_cls = self._resolve_agent_class(metadata)
                except Exception as exc:
                    if self._should_use_oopa_fallback(normalized, exc):
                        proxy = self._create_oopa_proxy(normalized, metadata, exc)
                        self._store_instance(normalized, proxy)
                        self._record_creation(normalized, "degraded", started_ms, metadata, implementation="out_of_process_proxy", error=str(exc))
                        return proxy
                    raise

                agent_config = self._agent_config_for(normalized, kwargs.get("config"))
                constructor_args = self._build_constructor_args(normalized, agent_cls, shared_memory, agent_config, kwargs)
                instance = agent_cls(**constructor_args)
                self._verify_action_surface(normalized, instance)
                self._store_instance(normalized, instance)
                self.unavailable_agents.pop(normalized, None)
                self._record_creation(normalized, "ok", started_ms, metadata)
                self._record_counter("create.succeeded")
                logger.info("Successfully created agent '%s'", normalized)
                return instance

            except FactoryError as exc:
                self.unavailable_agents[normalized] = exc.summary()
                self._record_counter("create.failed")
                self._record_creation(normalized, "error", started_ms, metadata, error=str(exc))
                exc.log()
                raise
            except TypeError as exc:
                error = AgentConstructionError.from_exception(
                    exc,
                    message=f"Constructor arguments failed for agent '{normalized}'",
                    component="agent_factory",
                    operation="create",
                    context={"agent_type": normalized, "class_name": metadata.class_name},
                )
                self.unavailable_agents[normalized] = error.summary()
                self._record_counter("create.failed")
                self._record_creation(normalized, "error", started_ms, metadata, error=str(error))
                raise error from exc
            except Exception as exc:
                error = AgentInitializationError.from_exception(
                    exc,
                    message=f"Failed to initialize agent '{normalized}'",
                    component="agent_factory",
                    operation="create",
                    context={"agent_type": normalized, "module_path": metadata.module_path, "class_name": metadata.class_name},
                )
                self.unavailable_agents[normalized] = error.summary()
                self._record_counter("create.failed")
                self._record_creation(normalized, "error", started_ms, metadata, error=str(error))
                raise error from exc

    def create_agent(self, agent_type: str, shared_memory: Any = None, **kwargs: Any) -> Any:
        return self.create(agent_type, shared_memory=shared_memory, **kwargs)

    def get_agent(self, agent_type: str, shared_memory: Any = None, **kwargs: Any) -> Any:
        return self.create(agent_type, shared_memory=shared_memory, **kwargs)

    def _store_instance(self, agent_type: str, instance: Any) -> None:
        self.active_agents[agent_type] = instance
        if self.settings.cache_agent_instances:
            self.cache.set(f"instance:{agent_type}", instance)
        try:
            if hasattr(instance, "with_lifecycle"):
                instance.with_lifecycle("active")
        except Exception:
            logger.debug("Unable to update lifecycle for %s", agent_type, exc_info=True)

    # ------------------------------------------------------------------
    # Metrics adaptation and diagnostics
    # ------------------------------------------------------------------
    def _get_metrics_adapter(self) -> MetricsAdapter:
        if self._metrics_adapter is not None:
            return self._metrics_adapter
        self._metrics_adapter = MetricsAdapter()
        self.metrics_adapter_status = "initialized"
        return self._metrics_adapter

    def run_adaptation_cycle(self, metrics: Mapping[str, Any], agent_types: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Process runtime metrics and apply factory-managed adjustments."""
        targets = tuple(self.normalize_agent_type(name) for name in (agent_types or self.get_registered_agent_types()))
        if not targets:
            raise MetricsValidationError(
                "At least one agent type is required for adaptation",
                component="agent_factory",
                operation="run_adaptation_cycle",
            )
        started_ms = monotonic_ms()
        try:
            adapter = self._get_metrics_adapter()
            if hasattr(adapter, "process_metrics_result"):
                result = adapter.process_metrics_result(dict(metrics), list(targets))
                adjustments = dict(result.bounded_adjustments) if hasattr(result, "bounded_adjustments") else result
            else:
                adjustments = adapter.process_metrics(dict(metrics), list(targets))

            if hasattr(adapter, "update_factory_config"):
                adapter.update_factory_config(self, adjustments)

            safe_adjustments = safe_serialize(adjustments, redact=True)
            self.cache.set("adaptation:latest", safe_adjustments)
            self._record_counter("adaptation.succeeded")
            self._record_timing("adaptation.duration_ms", started_ms)
            self._record_event("adaptation.completed", {"agent_types": targets, "adjustments": safe_adjustments})
            return {"status": "ok", "agent_types": list(targets), "adjustments": safe_adjustments}
        except FactoryError:
            self._record_counter("adaptation.failed")
            raise
        except Exception as exc:
            self._record_counter("adaptation.failed")
            raise MetricsAdapterError.from_exception(
                exc,
                message="AgentFactory adaptation cycle failed",
                component="agent_factory",
                operation="run_adaptation_cycle",
                context={"agent_types": targets},
            ) from exc

    def inspect_registered_agents(self, *, import_check: Optional[bool] = None, constructor_check: Optional[bool] = None) -> Dict[str, Dict[str, Any]]:
        import_check = self.settings.diagnostics_import_check if import_check is None else import_check
        constructor_check = self.settings.diagnostics_constructor_check if constructor_check is None else constructor_check
        diagnostics: Dict[str, Dict[str, Any]] = {}
        for name, metadata in sorted(self.registry.agents.items()):
            info: Dict[str, Any] = {
                "status": "ok",
                "module_path": metadata.module_path,
                "class_name": metadata.class_name,
                "version": metadata.version,
                "dependencies": list(getattr(metadata, "dependencies", []) or []),
                "issues": [],
            }
            if name in self.unavailable_agents:
                info["status"] = "unavailable"
                info["issues"].append(self.unavailable_agents[name])

            if import_check:
                try:
                    cls = self._resolve_agent_class(metadata)
                    if self.settings.strict_base_agent_subclass and not issubclass(cls, BaseAgent):
                        info["status"] = "warning"
                        info["issues"].append("Class is not a BaseAgent subclass.")
                    if constructor_check:
                        signature_meta = self._get_constructor_signature(cls)
                        params = set(signature_meta["params"].keys())
                        expected = {"shared_memory", "agent_factory", "config"}
                        missing = sorted(expected - params)
                        if missing and not signature_meta["accepts_var_kwargs"]:
                            info["status"] = "warning"
                            info["issues"].append(f"Constructor does not declare optional standard args: {missing}")
                except Exception as exc:
                    info["status"] = "error"
                    info["issues"].append(f"{type(exc).__name__}: {exc}")
            diagnostics[name] = info
        printer.pretty("Agent Diagnostics", diagnostics, "debug")
        return diagnostics

    def health_check(self) -> Dict[str, Any]:
        registry_size = len(getattr(self.registry, "agents", {}))
        active_size = len(self.active_agents)
        unavailable_size = len(self.unavailable_agents)
        status = "ok" if unavailable_size == 0 else "degraded"
        payload = {
            "status": status,
            "registered_agents": registry_size,
            "active_agents": active_size,
            "unavailable_agents": unavailable_size,
            "metrics_adapter_status": self.metrics_adapter_status,
            "torch_probe": dict(self._torch_probe),
            "cache": self.cache.stats() if hasattr(self.cache, "stats") else {},
            "observability": self.observability.snapshot() if hasattr(self.observability, "snapshot") else {},
        }
        self._record_event("factory.health_check", payload)
        return payload

    def snapshot(self, *, include_agents: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "version": __version__,
            "settings": safe_serialize(self.agent_factory_config, redact=True),
            "health": self.health_check(),
            "registered_agent_types": self.get_registered_agent_types(),
            "active_agent_types": sorted(self.active_agents.keys()),
            "unavailable_agents": dict(self.unavailable_agents),
            "creation_history": [record.to_dict() for record in self.creation_history],
        }
        if include_agents:
            payload["agents"] = self.inspect_registered_agents(import_check=False, constructor_check=False)
        return payload

    # ------------------------------------------------------------------
    # Lifecycle and utility methods
    # ------------------------------------------------------------------
    def get_registered_agent_types(self) -> list[str]:
        return sorted(str(name) for name in getattr(self.registry, "agents", {}).keys())

    def get_active_agent_types(self) -> list[str]:
        return sorted(self.active_agents.keys())

    def get(self, agent_type: str, default: Any = None) -> Any:
        normalized = self.normalize_agent_type(agent_type)
        return self.active_agents.get(normalized, default)

    def release(self, agent_type: str) -> bool:
        normalized = self.normalize_agent_type(agent_type)
        removed = self.active_agents.pop(normalized, None)
        self.cache.delete(f"instance:{normalized}")
        if removed is not None:
            self._record_event("agent.released", {"agent_type": normalized})
            return True
        return False

    def clear_active_agents(self) -> None:
        self.active_agents.clear()
        self._record_event("agent.active_cleared", {})

    def reset_unavailable(self, agent_type: Optional[str] = None) -> None:
        if agent_type is None:
            self.unavailable_agents.clear()
            self._record_event("agent.unavailable_cleared", {})
            return
        normalized = self.normalize_agent_type(agent_type)
        self.unavailable_agents.pop(normalized, None)
        self._record_event("agent.unavailable_reset", {"agent_type": normalized})

    def shutdown(self) -> None:
        for name, agent in list(self.active_agents.items()):
            for method_name in ("shutdown", "close", "stop"):
                method = getattr(agent, method_name, None)
                if callable(method):
                    method()
                    break
        self.active_agents.clear()
        self.cache.clear()
        self._record_event("factory.shutdown", {})
        logger.info("Agent Factory shutdown complete")

    def _record_creation(
        self,
        agent_type: str,
        status: str,
        started_ms: float,
        metadata: AgentMetaData,
        *,
        implementation: str = "in_process",
        error: Optional[str] = None,
    ) -> None:
        duration_ms = max(0.0, monotonic_ms() - started_ms)
        if self.settings.record_creation_history:
            self.creation_history.append(
                AgentCreationRecord(
                    agent_type=agent_type,
                    status=status,
                    duration_ms=duration_ms,
                    version=str(metadata.version) if metadata.version is not None else None,
                    implementation=implementation,
                    error=error,
                    metadata={"module_path": metadata.module_path, "class_name": metadata.class_name},
                )
            )
        self._record_timing("create.duration_ms", started_ms)
        self._record_event(
            "agent.creation",
            {"agent_type": agent_type, "status": status, "duration_ms": duration_ms, "implementation": implementation, "error": error},
        )

    def _record_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self.settings.publish_observability_events:
            self.observability.record_event(event_type, dict(payload))

    def _record_counter(self, name: str, value: int = 1) -> None:
        if self.settings.publish_observability_events:
            self.observability.inc(name, value)

    def _record_timing(self, name: str, started_ms: float) -> None:
        if self.settings.publish_observability_events:
            self.observability.observe_timing(name, max(0.0, monotonic_ms() - started_ms))


if __name__ == "__main__":
    print("\n=== Running  Agent factory ===\n")
    printer.status("TEST", " Agent factory initialized", "info")

    factory = AgentFactory()
    registered = factory.get_registered_agent_types()
    assert registered, "AgentFactory should register default agents"
    assert "browser" in registered, "Browser agent should be registered by default"

    alias_report = factory.get_agent_dependency_report("web")
    assert alias_report["agent_type"] == "browser"

    diagnostics = factory.inspect_registered_agents(import_check=False, constructor_check=False)
    assert "browser" in diagnostics

    health = factory.health_check()
    assert health["registered_agents"] == len(registered)

    snapshot = factory.snapshot(include_agents=True)
    assert snapshot["registered_agent_types"] == registered

    factory.reset_unavailable()
    factory.clear_active_agents()

    print("\n=== Test ran successfully ===\n")
