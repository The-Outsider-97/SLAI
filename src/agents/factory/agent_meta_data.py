"""Agent metadata and registry utilities for the factory-core layer.

This module owns the factory-core responsibilities only:

- describing agent implementations through validated metadata;
- registering, unregistering, and querying agent metadata;
- maintaining version indexes and dependency graphs;
- resolving dependency load order for factory lifecycle operations.

It intentionally does not instantiate agents, perform runtime adaptation, or
coordinate broad reasoning loops. Those concerns belong to factory isolation,
metrics/runtime adaptation, or higher-level orchestration modules.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from .utils.config_loader import get_config_section, load_global_config
from .utils.factory_errors import *
from .utils.factory_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Agent Meta Data")
printer = PrettyPrinter()


@dataclass(slots=True)
class AgentMetaData:
    """Validated descriptor for a factory-managed agent implementation.

    The metadata object is deliberately small enough to remain serialisable, but
    rich enough to support dependency resolution, version selection, lifecycle
    checks, and future factory expansion. Runtime instances are not stored here.
    """

    name: str
    class_name: str
    module_path: str
    required_params: Tuple[str, ...] = ()
    version: Optional[str] = None
    description: str = ""
    author: str = "Unknown"
    dependencies: Optional[Sequence[str]] = None
    tags: Tuple[str, ...] = ()
    status: str = "active"
    capabilities: Tuple[str, ...] = ()
    lifecycle: str = "registered"
    config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime-loaded config sections retained for compatibility with existing code.
    meta_config: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    required_fields: Optional[List[str]] = field(default=None, init=False, repr=False)
    validation_rules: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        self.config = load_global_config()
        self.meta_config = get_config_section("agent_meta")

        if not isinstance(self.meta_config, MutableMapping):
            raise InvalidFactoryConfigurationError(
                "agent_meta config section must be a mapping",
                context={"section_type": type(self.meta_config).__name__},
                component="agent_meta_data",
                operation="load_agent_meta_config",
            )

        if self.version is None:
            self.version = str(self.meta_config.get("default_version", "1.0.0"))

        self.required_fields = list(self.meta_config.get("required_fields", ["name", "class_name", "module_path", "version"]))
        self.validation_rules = dict(
            self.meta_config.get(
                "validation_rules",
                {
                    "max_name_length": 64,
                    "allowed_modules": ["agents.", "src.agents."],
                    "strict_semver": False,
                    "allow_dotted_names": True,
                },
            )
        )

        if not self.author or self.author == "Unknown":
            self.author = str(self.meta_config.get("default_author", self.author or "Unknown"))
        if not self.status:
            self.status = str(self.meta_config.get("default_status", "active"))

        self._normalise_and_validate()

    @property
    def identity(self) -> str:
        """Return a stable name/version identity key."""
        return f"{self.name}@{self.version}"

    @property
    def import_path(self) -> str:
        """Return the fully-qualified class import path."""
        return f"{self.module_path}.{self.class_name}"

    def _normalise_and_validate(self) -> None:
        rules = self.validation_rules or {}
        required_fields = self.required_fields or ("name", "class_name", "module_path", "version")
        max_name_length = int(rules.get("max_name_length", 64))
        allowed_modules = tuple(rules.get("allowed_modules", ()) or ())
        strict_semver = bool(rules.get("strict_semver", False))
        allow_dotted_names = bool(rules.get("allow_dotted_names", True))

        payload = validate_agent_metadata_dict(
            {
                "name": self.name,
                "class_name": self.class_name,
                "module_path": self.module_path,
                "version": self.version,
                "required_params": self.required_params,
                "dependencies": list(self.dependencies or ()),
            },
            required_fields=required_fields,
            allowed_module_prefixes=allowed_modules,
            max_name_length=max_name_length,
            strict_version=strict_semver,
        )

        self.name = validate_agent_name(payload["name"], max_length=max_name_length, allow_dotted=allow_dotted_names)
        self.class_name = str(payload["class_name"])
        self.module_path = str(payload["module_path"])
        self.version = validate_version(payload.get("version") or self.version or "1.0.0", strict_semver=strict_semver)
        self.required_params = validate_required_params(payload.get("required_params", ()))
        self.dependencies = list(validate_dependency_names(payload.get("dependencies", ()), agent_name=self.name))
        self.tags = tuple(str(tag).strip() for tag in self.tags if str(tag).strip())
        self.capabilities = tuple(str(capability).strip() for capability in self.capabilities if str(capability).strip())
        self.metadata = sanitize_context(normalize_payload(self.metadata), redact=False)
        self.description = str(self.description or "")
        self.author = str(self.author or "Unknown")
        self.status = str(self.status or "active")
        self.lifecycle = str(self.lifecycle or "registered")
        self.updated_at = datetime.now(timezone.utc).isoformat()

        # Validate module path again after normalisation, preserving explicit policy errors.
        validate_module_path(self.module_path, allowed_prefixes=allowed_modules)

    def has_dependency(self, dependency_name: str) -> bool:
        dependency = validate_agent_name(dependency_name)
        return dependency in set(self.dependencies or [])

    def with_lifecycle(self, lifecycle: str) -> "AgentMetaData":
        self.lifecycle = str(lifecycle or "registered")
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return self

    def to_dict(self, *, include_config: bool = False, include_runtime: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "class_name": self.class_name,
            "module_path": self.module_path,
            "import_path": self.import_path,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "dependencies": list(self.dependencies or []),
            "required_params": list(self.required_params),
            "tags": list(self.tags),
            "status": self.status,
            "capabilities": list(self.capabilities),
            "metadata": safe_serialize(self.metadata, redact=True),
        }
        if include_runtime:
            payload.update(
                {
                    "identity": self.identity,
                    "lifecycle": self.lifecycle,
                    "created_at": self.created_at,
                    "updated_at": self.updated_at,
                }
            )
        if include_config:
            payload["config"] = safe_serialize(self.config, redact=True)
            payload["meta_config"] = safe_serialize(self.meta_config, redact=True)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentMetaData":
        payload = dict(normalize_payload(data))
        payload.pop("identity", None)
        payload.pop("import_path", None)
        return cls(**payload)

    def clone(self, **overrides: Any) -> "AgentMetaData":
        payload = self.to_dict(include_config=False, include_runtime=False)
        payload.update(overrides)
        return AgentMetaData.from_dict(payload)

    def __hash__(self) -> int:
        return hash((self.name, self.version, self.module_path, self.class_name))


class AgentRegistry:
    """In-memory registry and dependency index for factory-managed agents."""

    def __init__(self) -> None:
        self.config = load_global_config()
        self.registry_config = get_config_section("agent_registry")
        if not isinstance(self.registry_config, MutableMapping):
            raise InvalidFactoryConfigurationError(
                "agent_registry config section must be a mapping",
                context={"section_type": type(self.registry_config).__name__},
                component="agent_registry",
                operation="load_registry_config",
            )

        self.agents: Dict[str, AgentMetaData] = {}
        self._versioned_agents: Dict[str, Dict[str, AgentMetaData]] = defaultdict(dict)
        self.version_map: Dict[str, List[str]] = defaultdict(list)
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.registration_order: List[str] = []
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at

        logger.info("Agent registry initialized")

    def __contains__(self, agent_name: object) -> bool:
        return isinstance(agent_name, str) and agent_name in self.agents

    def __len__(self) -> int:
        return len(self.agents)

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def _allow_overwrite(self, explicit: Optional[bool]) -> bool:
        if explicit is not None:
            return bool(explicit)
        return bool(self.registry_config.get("allow_overwrite", False))

    def _allow_missing_dependencies(self) -> bool:
        return bool(self.registry_config.get("allow_missing_dependencies", True))

    def _should_validate_graph_on_register(self) -> bool:
        return bool(self.registry_config.get("validate_graph_on_register", True))

    def register(self, metadata: AgentMetaData, *, overwrite: Optional[bool] = None, make_default: bool = True) -> AgentMetaData:
        """Register metadata and update version/dependency indexes."""
        if not isinstance(metadata, AgentMetaData):
            raise FactoryTypeError(
                "Only AgentMetaData objects can be registered",
                context={"actual_type": type(metadata).__name__},
                component="agent_registry",
                operation="register",
            )

        allow_overwrite = self._allow_overwrite(overwrite)
        name = validate_agent_name(metadata.name)
        version = validate_version(metadata.version or "1.0.0")

        if name in self._versioned_agents and version in self._versioned_agents[name] and not allow_overwrite:
            raise DuplicateAgentRegistrationError(
                f"Agent '{name}' version '{version}' is already registered",
                context={"agent_name": name, "version": version},
                component="agent_registry",
                operation="register",
            )

        self._versioned_agents[name][version] = metadata
        if make_default or name not in self.agents:
            self.agents[name] = metadata

        if name not in self.registration_order:
            self.registration_order.append(name)

        if name not in self.version_map[version]:
            self.version_map[version].append(name)

        self.dependency_graph[name] = set(validate_dependency_names(metadata.dependencies or (), agent_name=name))
        for dependency in self.dependency_graph[name]:
            self.reverse_dependency_graph[dependency].add(name)

        if self._should_validate_graph_on_register():
            self.validate_integrity(allow_missing_dependencies=self._allow_missing_dependencies())

        self._touch()
        logger.info("Registered agent: %s v%s", name, version)
        return metadata

    def register_many(self, metadata_items: Iterable[AgentMetaData], *, overwrite: Optional[bool] = None) -> List[AgentMetaData]:
        registered: List[AgentMetaData] = []
        for metadata in metadata_items:
            registered.append(self.register(metadata, overwrite=overwrite))
        return registered

    def unregister(self, agent_name: str, version: Optional[str] = None) -> AgentMetaData:
        """Remove an agent or one specific version from the registry."""
        name = validate_agent_name(agent_name)
        if name not in self._versioned_agents:
            raise AgentNotRegisteredError(
                f"Agent '{name}' is not registered",
                context={"agent_name": name},
                component="agent_registry",
                operation="unregister",
            )

        selected_version = validate_version(version) if version else (self.agents[name].version or "1.0.0")
        if selected_version not in self._versioned_agents[name]:
            raise AgentVersionUnavailableError(
                f"Version '{selected_version}' is not available for agent '{name}'",
                context={"agent_name": name, "version": selected_version, "available_versions": self.list_versions(name)},
                component="agent_registry",
                operation="unregister",
            )

        removed = self._versioned_agents[name].pop(selected_version)
        if selected_version in self.version_map and name in self.version_map[selected_version]:
            self.version_map[selected_version].remove(name)
            if not self.version_map[selected_version]:
                self.version_map.pop(selected_version, None)

        if not self._versioned_agents[name]:
            self._versioned_agents.pop(name, None)
            self.agents.pop(name, None)
            self.dependency_graph.pop(name, None)
            self.registration_order = [registered for registered in self.registration_order if registered != name]
            for dependents in self.reverse_dependency_graph.values():
                dependents.discard(name)
            self.reverse_dependency_graph.pop(name, None)
        elif self.agents.get(name) is removed:
            # Pick a deterministic fallback default version.
            fallback_version = sorted(self._versioned_agents[name].keys())[-1]
            self.agents[name] = self._versioned_agents[name][fallback_version]

        self._touch()
        logger.info("Unregistered agent: %s v%s", name, selected_version)
        return removed

    def get(self, agent_name: str, version: Optional[str] = None) -> AgentMetaData:
        name = validate_agent_name(agent_name)
        if name not in self.agents:
            raise AgentNotRegisteredError(
                f"Agent '{name}' is not registered",
                context={"agent_name": name, "registered_agents": self.names()},
                component="agent_registry",
                operation="get",
            )

        if version is None:
            return self.agents[name]

        selected_version = validate_version(version)
        candidate = self._versioned_agents.get(name, {}).get(selected_version)
        if candidate is None:
            raise AgentVersionUnavailableError(
                f"Version '{selected_version}' is not available for agent '{name}'",
                context={"agent_name": name, "version": selected_version, "available_versions": self.list_versions(name)},
                component="agent_registry",
                operation="get",
            )
        return candidate

    def set_default(self, agent_name: str, version: str) -> AgentMetaData:
        metadata = self.get(agent_name, version=version)
        self.agents[metadata.name] = metadata
        self._touch()
        return metadata

    def names(self) -> List[str]:
        return list(self.registration_order)

    def list_versions(self, agent_name: str) -> List[str]:
        name = validate_agent_name(agent_name)
        return sorted(self._versioned_agents.get(name, {}).keys())

    def list_agents(self, *, include_config: bool = False) -> List[Dict[str, Any]]:
        return [self.agents[name].to_dict(include_config=include_config) for name in self.names() if name in self.agents]

    def get_dependencies(self, agent_name: str, *, transitive: bool = False) -> List[str]:
        name = validate_agent_name(agent_name)
        if name not in self.agents:
            raise AgentNotRegisteredError(
                f"Agent '{name}' is not registered",
                context={"agent_name": name},
                component="agent_registry",
                operation="get_dependencies",
            )
        if not transitive:
            return sorted(self.dependency_graph.get(name, set()))
        order = self.resolve_dependency_tree(name)
        return [candidate for candidate in order if candidate != name]

    def get_dependents(self, agent_name: str, *, transitive: bool = False) -> List[str]:
        name = validate_agent_name(agent_name)
        direct = set(self.reverse_dependency_graph.get(name, set()))
        if not transitive:
            return sorted(direct)

        discovered: Set[str] = set()
        stack = list(direct)
        while stack:
            current = stack.pop()
            if current in discovered:
                continue
            discovered.add(current)
            stack.extend(self.reverse_dependency_graph.get(current, set()))
        return sorted(discovered)

    def resolve_dependency_tree(self, agent_name: str) -> List[str]:
        """Return dependency-first load order for one agent."""
        root = validate_agent_name(agent_name)
        if root not in self.agents:
            raise AgentNotRegisteredError(
                f"Agent '{root}' is not registered",
                context={"agent_name": root, "registered_agents": self.names()},
                component="agent_registry",
                operation="resolve_dependency_tree",
            )

        visited: Set[str] = set()
        visiting: Set[str] = set()
        load_order: List[str] = []
        path: List[str] = []

        def resolve(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                cycle_start = path.index(name) if name in path else 0
                cycle = path[cycle_start:] + [name]
                raise CircularDependencyError(
                    "Circular dependency detected while resolving agent dependencies",
                    context={"agent_name": root, "cycle": cycle},
                    component="agent_registry",
                    operation="resolve_dependency_tree",
                )
            if name not in self.agents:
                raise MissingDependencyError(
                    "Dependency is not registered",
                    context={"root_agent": root, "missing_dependency": name, "known_agents": self.names()},
                    component="agent_registry",
                    operation="resolve_dependency_tree",
                )

            visiting.add(name)
            path.append(name)
            for dependency in sorted(self.dependency_graph.get(name, set())):
                resolve(dependency)
            path.pop()
            visiting.remove(name)
            visited.add(name)
            load_order.append(name)

        try:
            resolve(root)
        except FactoryError:
            raise
        except Exception as exc:
            raise DependencyResolutionError.from_exception(
                exc,
                message="Dependency resolution failed",
                component="agent_registry",
                operation="resolve_dependency_tree",
                context={"agent_name": root},
            ) from exc
        return load_order

    def validate_integrity(self, *, allow_missing_dependencies: Optional[bool] = None) -> Dict[str, Any]:
        allow_missing = self._allow_missing_dependencies() if allow_missing_dependencies is None else bool(allow_missing_dependencies)
        known_agents = set(self.agents.keys())
        graph = {name: sorted(dependencies) for name, dependencies in self.dependency_graph.items()}

        if allow_missing:
            validate_dependency_graph(graph)
        else:
            validate_dependency_graph(graph, known_agents=known_agents)

        missing = sorted({dependency for dependencies in graph.values() for dependency in dependencies if dependency not in known_agents})
        if missing and not allow_missing:
            raise MissingDependencyError(
                "Registry contains dependencies that are not registered",
                context={"missing_dependencies": missing, "known_agents": sorted(known_agents)},
                component="agent_registry",
                operation="validate_integrity",
            )

        orphan_reverse_edges = sorted(edge for edge in self.reverse_dependency_graph.keys() if edge not in known_agents and edge not in missing)
        return {
            "status": "ok" if not missing and not orphan_reverse_edges else "warning",
            "agents": len(self.agents),
            "versions": sum(len(versions) for versions in self._versioned_agents.values()),
            "missing_dependencies": missing,
            "orphan_reverse_edges": orphan_reverse_edges,
            "updated_at": self.updated_at,
        }

    def snapshot(self, *, include_config: bool = False, include_agents: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "agent_count": len(self.agents),
            "registration_order": self.names(),
            "versions": {name: self.list_versions(name) for name in self._versioned_agents.keys()},
            "dependency_graph": {name: sorted(deps) for name, deps in self.dependency_graph.items()},
            "reverse_dependency_graph": {name: sorted(deps) for name, deps in self.reverse_dependency_graph.items()},
        }
        if include_agents:
            payload["agents"] = self.list_agents(include_config=include_config)
        if include_config:
            payload["config"] = safe_serialize(self.config, redact=True)
            payload["registry_config"] = safe_serialize(self.registry_config, redact=True)
        return payload

    def clear(self) -> None:
        self.agents.clear()
        self._versioned_agents.clear()
        self.version_map.clear()
        self.dependency_graph.clear()
        self.reverse_dependency_graph.clear()
        self.registration_order.clear()
        self._touch()


if __name__ == "__main__":
    print("\n=== Running Agent Meta Data ===\n")
    printer.status("TEST", "Agent Meta Data initialized", "info")

    registry = AgentRegistry()
    browser_metadata = AgentMetaData(
        name="browser",
        class_name="BrowserAgent",
        module_path="src.agents.browser_agent",
        version="1.0.0",
        description="Browser automation agent metadata used for registry testing.",
        capabilities=("navigation", "search"),
    )
    reasoning_metadata = AgentMetaData(
        name="reasoning",
        class_name="ReasoningAgent",
        module_path="src.agents.reasoning_agent",
        version="1.0.0",
        dependencies=["browser"],
        description="Reasoning agent metadata with a browser dependency.",
    )

    registry.register(browser_metadata)
    registry.register(reasoning_metadata)

    assert registry.get("browser").class_name == "BrowserAgent"
    assert registry.get("reasoning").has_dependency("browser")
    assert registry.resolve_dependency_tree("reasoning") == ["browser", "reasoning"]
    integrity = registry.validate_integrity(allow_missing_dependencies=False)
    assert integrity["status"] == "ok"
    assert registry.snapshot()["agent_count"] == 2

    print("\n=== Test ran successfully ===\n")
