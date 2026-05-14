"""
Production decompositional reasoning strategy for the reasoning subsystem.

Decompositional reasoning breaks a system into meaningful parts, analyses each
component, identifies interactions, and synthesises a system-level view.
"""
from __future__ import annotations

import hashlib
import inspect
import math
import re
import time

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from ..reasoning_cache import ReasoningCache
from ..utils.config_loader import load_global_config, get_config_section
from ..utils.reasoning_errors import *
from ..utils.reasoning_helpers import *
from .base_reasoning import BaseReasoning
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reasoning Decompositional")
printer = PrettyPrinter()


@dataclass
class ComponentNode:
    """Canonical component node used internally during decomposition."""

    id: str
    name: str
    type: str
    depth: int
    content: Any
    parent_id: Optional[str] = None
    path: Tuple[str, ...] = field(default_factory=tuple)
    children: List["ComponentNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "depth": self.depth,
            "content": self.content,
            "parent_id": self.parent_id,
            "path": list(self.path),
            "metadata": dict(self.metadata),
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class ComponentProfile:
    """Flattened, analysis-ready profile for one component."""

    component_id: str
    name: str
    depth: int
    kind: str
    size: int
    tokens: Set[str]
    functions: Set[str]
    dependencies: Set[str]
    complexity: float
    criticality: float


class ReasoningDecompositional(BaseReasoning):
    """Break systems into components and synthesize structural insight.

    The implementation supports dictionaries, lists/tuples/sets, objects with
    attributes, callables, and scalar leaves.  Dicts using a ``components`` key
    receive first-class treatment so application/system descriptions decompose
    into their declared parts instead of generic placeholders.
    """

    MODULE_VERSION = "2.2.0"

    def __init__(self) -> None:
        super().__init__()
        self.config: Dict[str, Any] = load_global_config()
        self.decompositional_config: Dict[str, Any] = get_config_section("reasoning_decompositional") or {}

        self.max_depth: int = bounded_iterations(
            self.decompositional_config.get("max_depth", 3), minimum=1, maximum=64
        )
        self.min_component_size: int = bounded_iterations(
            self.decompositional_config.get("min_component_size", 1), minimum=0, maximum=1_000_000
        )
        self.max_components: int = bounded_iterations(
            self.decompositional_config.get("max_components", 512), minimum=1, maximum=100_000
        )
        self.max_children_per_node: int = bounded_iterations(
            self.decompositional_config.get("max_children_per_node", 64), minimum=1, maximum=10_000
        )
        self.max_interactions: int = bounded_iterations(
            self.decompositional_config.get("max_interactions", 1000), minimum=1, maximum=1_000_000
        )
        self.max_name_length: int = bounded_iterations(
            self.decompositional_config.get("max_name_length", 96), minimum=8, maximum=512
        )
        self.content_preview_chars: int = bounded_iterations(
            self.decompositional_config.get("content_preview_chars", 240), minimum=16, maximum=10_000
        )

        self.structural_weight: float = clamp_confidence(self.decompositional_config.get("structural_weight", 0.5))
        self.functional_weight: float = clamp_confidence(self.decompositional_config.get("functional_weight", 0.5))
        self.dependency_weight: float = clamp_confidence(self.decompositional_config.get("dependency_weight", 0.25))
        self.criticality_weight: float = clamp_confidence(self.decompositional_config.get("criticality_weight", 0.25))
        self.interaction_threshold: float = clamp_confidence(self.decompositional_config.get("interaction_threshold", 0.55))
        self.criticality_threshold: float = clamp_confidence(self.decompositional_config.get("criticality_threshold", 0.7))
        self.vulnerability_threshold: float = clamp_confidence(self.decompositional_config.get("vulnerability_threshold", 0.8))
        self.high_complexity_threshold: float = clamp_confidence(self.decompositional_config.get("high_complexity_threshold", 0.8))
        self.low_cohesion_threshold: float = clamp_confidence(self.decompositional_config.get("low_cohesion_threshold", 0.3))

        self.enable_interaction_analysis: bool = bool(
            self.decompositional_config.get("enable_interaction_analysis", True)
        )
        self.strict_inputs: bool = bool(self.decompositional_config.get("strict_inputs", True))
        self.include_content_preview: bool = bool(self.decompositional_config.get("include_content_preview", True))
        self.include_context: bool = bool(self.decompositional_config.get("return_context", False))
        self.enable_cache: bool = bool(self.decompositional_config.get("enable_cache", True))
        self.default_strategy: str = str(
            self.decompositional_config.get("default_strategy", "structural")
        ).strip().lower() or "structural"
        self.cache_ttl_seconds: float = float(self.decompositional_config.get("cache_ttl_seconds", 300.0))

        self.cache: Optional[ReasoningCache] = None
        if self.enable_cache:
            self.cache = ReasoningCache(
                namespace="reasoning_decompositional",
                default_ttl_seconds=self.cache_ttl_seconds,
            )

        self._component_budget_used = 0
        logger.info("ReasoningDecompositional initialized | max_depth=%s | max_components=%s", self.max_depth, self.max_components)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def perform_reasoning(self, system: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: # type: ignore
        """Perform decompositional reasoning on a system."""
        start = time.monotonic()
        context = dict(context or {})
        self._validate_input(system, context)
        cache_key = self._cache_key(system, context)

        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        self._component_budget_used = 0
        decomposition_tree = self.decompose_system(system, context)
        component_analysis = self.analyze_components(decomposition_tree, context)
        interaction_analysis: Dict[str, List[Dict[str, Any]]] = self._empty_interactions()
        if self.enable_interaction_analysis:
            interaction_analysis = self.analyze_interactions(decomposition_tree, context)
        system_understanding = self.synthesize_understanding(
            decomposition_tree,
            component_analysis,
            interaction_analysis,
            context,
        )
        result = self._format_results(
            decomposition_tree,
            component_analysis,
            interaction_analysis,
            system_understanding,
        )
        result["metrics"]["elapsed_seconds"] = elapsed_seconds(start)
        result["metrics"]["cache_enabled"] = self.cache is not None
        if self.include_context:
            result["context_used"] = json_safe_reasoning_state(context)

        if self.cache is not None:
            self.cache.set(cache_key, result, metadata={"reasoning_type": "decompositional"})
        return result

    def decompose_system(self, system: Any, context: Dict[str, Any], depth: int = 0,
                         parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Recursively decompose a system into a bounded hierarchy."""
        strategy = str(context.get("decomposition_strategy", self.default_strategy)).strip().lower()
        node = self._build_node(system, context, depth=depth, parent_id=parent_id)
        if depth >= self.max_depth or not self._is_decomposable(system, depth):
            node.type = "atomic"
            return node.to_dict()

        children = self.decompose_whole(system, strategy)
        if not children:
            node.type = "atomic"
            return node.to_dict()

        for idx, component in enumerate(children[: self.max_children_per_node]):
            if self._component_budget_used >= self.max_components:
                node.metadata["truncated"] = True
                node.metadata["truncation_reason"] = "max_components"
                break
            child_parent = f"{node.id}-{idx}"
            child_dict = self.decompose_system(component, context, depth + 1, child_parent)
            node.children.append(self._node_from_dict(child_dict))
        node.type = "composite" if node.children else "atomic"
        return node.to_dict()

    def decompose_whole(self, whole: Any, decomposition_strategy: str = "functional") -> List[Any]:
        """Break a whole into child components using structural data first."""
        strategy = (decomposition_strategy or self.default_strategy).lower()
        if isinstance(whole, Mapping):
            if "components" in whole:
                return self._component_values(whole["components"])
            if strategy == "functional" and "functions" in whole:
                return list(whole.get("functions") or [])
            return [self._named_child(key, value) for key, value in whole.items() if key not in {"name", "id", "description"}]
        if isinstance(whole, (list, tuple)):
            return list(whole)
        if isinstance(whole, set):
            return sorted(list(whole), key=lambda item: str(item))
        if hasattr(whole, "__dict__") and not isinstance(whole, type):
            return [self._named_child(key, value) for key, value in vars(whole).items() if not key.startswith("_")]
        return []

    def analyze_components(self, decomposition_tree: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Analyze every component in a decomposition tree."""
        self._validate_tree(decomposition_tree)
        analysis: Dict[str, Dict[str, Any]] = {}
        for node in self._iter_nodes(decomposition_tree):
            analysis[node["id"]] = self._analyze_component(node, context)
        return analysis

    def analyze_interactions(self, decomposition_tree: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze structural, functional, data-flow and hierarchy interactions."""
        self._validate_tree(decomposition_tree)
        interactions = self._empty_interactions()
        components = self._flatten_decomposition(decomposition_tree)
        ids = list(components.keys())
        count = 0
        for i, left_id in enumerate(ids):
            for right_id in ids[i + 1:]:
                if count >= self.max_interactions:
                    interactions["metadata"].append({"truncated": True, "reason": "max_interactions"})
                    return interactions
                left = components[left_id]
                right = components[right_id]
                if left.get("depth") != right.get("depth"):
                    continue
                interaction = self._analyze_interaction(left, right, context)
                if interaction:
                    interactions.setdefault(interaction["type"], []).append(interaction)
                    count += 1

        interactions["hierarchical"] = self._analyze_hierarchical_interactions(decomposition_tree, context)
        interactions["dependency"] = self._analyze_dependency_interactions(components, context)
        return interactions

    def synthesize_understanding(self, decomposition_tree: Dict[str, Any],
                                 component_analysis: Dict[str, Dict[str, Any]],
                                 interaction_analysis: Dict[str, List[Dict[str, Any]]],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize component and interaction data into system understanding."""
        all_components = self._flatten_decomposition(decomposition_tree)
        component_count = len(all_components)
        max_depth = max((int(comp.get("depth", 0)) for comp in all_components.values()), default=0)
        avg_complexity = self._mean([a.get("complexity", 0.0) for a in component_analysis.values()])
        key_components = [
            cid for cid, analysis in component_analysis.items()
            if float(analysis.get("criticality", 0.0)) >= self.criticality_threshold
        ]
        system_properties = {
            "modularity": self._calculate_modularity(interaction_analysis),
            "cohesion": self._calculate_cohesion(component_analysis, interaction_analysis),
            "coupling": self._calculate_coupling(interaction_analysis),
            "decomposition_balance": self._calculate_balance(decomposition_tree),
        }
        insights = self._generate_insights(decomposition_tree, component_analysis, interaction_analysis, context)
        vulnerabilities = self._identify_vulnerabilities(component_analysis, interaction_analysis, context)
        return {
            "system_properties": system_properties,
            "key_components": key_components,
            "system_metrics": {
                "component_count": component_count,
                "max_depth": max_depth,
                "avg_complexity": avg_complexity,
                "interaction_density": self._interaction_density(interaction_analysis, component_count),
                "leaf_count": sum(1 for comp in all_components.values() if not comp.get("children")),
            },
            "insights": insights,
            "vulnerabilities": vulnerabilities,
        }

    # ------------------------------------------------------------------
    # Component analysis helpers
    # ------------------------------------------------------------------
    def _analyze_component(self, component_node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        component = component_node.get("content")
        structure = self._analyze_structure(component)
        function = self._analyze_function(component, context)
        dependencies = self._identify_dependencies(component, context)
        complexity = self._calculate_complexity(component_node)
        criticality = self._assess_criticality(component_node, context)
        profile = self._profile_component(component_node, context, complexity=complexity, criticality=criticality)
        return {
            "structure": structure,
            "function": function,
            "functions": sorted(profile.functions),
            "complexity": complexity,
            "criticality": criticality,
            "dependencies": sorted(dependencies),
            "tokens": sorted(profile.tokens)[: self.max_children_per_node],
            "size": profile.size,
            "health": self._component_health(complexity, criticality, dependencies),
        }

    def _analyze_structure(self, component: Any) -> Dict[str, Any]:
        """Analyze structural properties of a component."""
        if isinstance(component, Mapping):
            keys = [str(key) for key in component.keys()]
            return {
                "type": "dictionary",
                "key_count": len(component),
                "keys": keys[: self.max_children_per_node],
                "value_types": {str(k): type(v).__name__ for k, v in list(component.items())[: self.max_children_per_node]},
                "has_components": "components" in component,
                "has_dependencies": "dependencies" in component,
            }
        if isinstance(component, (list, tuple)):
            return {
                "type": type(component).__name__,
                "length": len(component),
                "element_types": [type(item).__name__ for item in list(component)[: self.max_children_per_node]],
                "unique_types": len({type(item).__name__ for item in component}),
            }
        if isinstance(component, set):
            values = sorted(component, key=lambda item: str(item))
            return {
                "type": "set",
                "size": len(values),
                "element_types": [type(item).__name__ for item in values[: self.max_children_per_node]],
                "unique_types": len({type(item).__name__ for item in values}),
            }
        if callable(component):
            signature = None
            try:
                signature = str(inspect.signature(component))
            except (TypeError, ValueError):
                signature = None
            return {"type": "callable", "name": getattr(component, "__name__", str(component)), "signature": signature}
        if hasattr(component, "__dict__") and not isinstance(component, type):
            attrs = [name for name in vars(component).keys() if not name.startswith("_")]
            return {"type": type(component).__name__, "attributes": attrs[: self.max_children_per_node], "attribute_count": len(attrs)}
        return {"type": type(component).__name__, "scalar": True, "preview": self._preview(component)}

    def _analyze_function(self, component: Any, context: Dict[str, Any]) -> str:
        """Determine the primary function/purpose of a component."""
        mapping = context.get("function_mapping", {}) or {}
        text = self._string_blob(component).lower()
        for pattern, function in mapping.items():
            if str(pattern).lower() in text:
                return str(function)
        functions = self._extract_functions(component)
        if functions:
            return ", ".join(sorted(functions)[:3])
        if isinstance(component, Mapping):
            return "Structured configuration or data container"
        if isinstance(component, (list, tuple, set)):
            return "Collection of related elements"
        if callable(component):
            return "Executable operation"
        return "Atomic data representation"

    def _calculate_complexity(self, component_node: Dict[str, Any]) -> float:
        """Calculate normalized component complexity."""
        content = component_node.get("content")
        child_count = len(component_node.get("children", []))
        depth = int(component_node.get("depth", 0))
        size = self._component_size(content)
        size_score = min(1.0, math.log1p(size) / math.log1p(max(2, self.max_children_per_node)))
        child_score = min(1.0, child_count / max(1, self.max_children_per_node))
        depth_score = min(1.0, depth / max(1, self.max_depth))
        return clamp_confidence((0.45 * size_score) + (0.35 * child_score) + (0.20 * depth_score))

    def _assess_criticality(self, component_node: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess how critical a component is to system operation."""
        if int(component_node.get("depth", 0)) == 0:
            return 1.0
        name = str(component_node.get("name", ""))
        critical_components = {str(item).lower() for item in context.get("critical_components", [])}
        if name.lower() in critical_components or component_node.get("id", "").lower() in critical_components:
            return 0.95
        content = component_node.get("content")
        dependencies = self._identify_dependencies(content, context)
        dependency_score = min(0.35, 0.08 * len(dependencies))
        child_score = min(0.30, 0.07 * len(component_node.get("children", [])))
        name_score = 0.20 if any(token in name.lower() for token in {"core", "database", "gateway", "auth", "engine"}) else 0.0
        return clamp_confidence(0.15 + dependency_score + child_score + name_score)

    def _identify_dependencies(self, component: Any, context: Dict[str, Any]) -> Set[str]:
        """Identify dependencies of the component from conventional fields and references."""
        deps: Set[str] = set()
        dependency_keys = set(context.get("dependency_keys", ["dependencies", "depends_on", "requires", "uses", "inputs"]))
        if isinstance(component, Mapping):
            for key, value in component.items():
                key_text = str(key)
                if key_text in dependency_keys:
                    deps.update(self._coerce_string_set(value))
                elif isinstance(value, str) and (value.startswith("comp_") or value.startswith("svc_") or value.startswith("module_")):
                    deps.add(value)
        elif hasattr(component, "dependencies"):
            deps.update(self._coerce_string_set(getattr(component, "dependencies")))
        return {dep for dep in deps if dep}

    # ------------------------------------------------------------------
    # Interaction analysis helpers
    # ------------------------------------------------------------------
    def _analyze_interaction(self, comp1: Dict[str, Any], comp2: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        structural_score = self._calculate_structural_similarity(comp1, comp2)
        functional_score = self._calculate_functional_dependency(comp1, comp2, context)
        dependency_score = self._dependency_overlap_score(comp1, comp2, context)
        score = self._weighted_score(
            [structural_score, functional_score, dependency_score],
            [self.structural_weight, self.functional_weight, self.dependency_weight],
        )
        if score < self.interaction_threshold:
            return None
        if dependency_score >= max(structural_score, functional_score):
            interaction_type = "data_flow"
        elif structural_score >= functional_score:
            interaction_type = "structural"
        else:
            interaction_type = "functional"
        return {
            "type": interaction_type,
            "components": [comp1["id"], comp2["id"]],
            "structural_score": structural_score,
            "functional_score": functional_score,
            "dependency_score": dependency_score,
            "score": score,
            "description": f"{interaction_type} relationship between {comp1['name']} and {comp2['name']}",
        }

    def _calculate_structural_similarity(self, comp1: Dict[str, Any], comp2: Dict[str, Any]) -> float:
        """Calculate structural similarity between components."""
        s1 = self._analyze_structure(comp1.get("content"))
        s2 = self._analyze_structure(comp2.get("content"))
        type_score = 0.6 if s1.get("type") == s2.get("type") else 0.0
        key_score = 0.0
        keys1 = set(map(str, s1.get("keys", [])))
        keys2 = set(map(str, s2.get("keys", [])))
        if keys1 or keys2:
            key_score = len(keys1 & keys2) / max(1, len(keys1 | keys2))
        size1 = int(s1.get("key_count", s1.get("length", s1.get("size", 1))) or 1)
        size2 = int(s2.get("key_count", s2.get("length", s2.get("size", 1))) or 1)
        size_score = 1.0 - (abs(size1 - size2) / max(size1, size2, 1))
        return clamp_confidence(type_score + 0.25 * key_score + 0.15 * size_score)

    def _calculate_functional_dependency(self, comp1: Dict[str, Any], comp2: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate functional dependency/similarity between components."""
        f1 = self._extract_functions(comp1.get("content")) or set(self._tokens(comp1.get("name", "")))
        f2 = self._extract_functions(comp2.get("content")) or set(self._tokens(comp2.get("name", "")))
        overlap = len(f1 & f2) / max(1, len(f1 | f2)) if (f1 or f2) else 0.0
        complementary_score = 0.0
        for pair in context.get("complementary_functions", []) or []:
            pair_set = {str(item).lower() for item in pair}
            if (f1 & pair_set) and (f2 & pair_set):
                complementary_score = max(complementary_score, 0.9)
        return clamp_confidence(max(overlap, complementary_score))

    def _analyze_hierarchical_interactions(self, node: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        interactions: List[Dict[str, Any]] = []
        for child in node.get("children", []):
            interactions.append({
                "type": "parent_child",
                "parent": node["id"],
                "child": child["id"],
                "direction": "parent_to_child",
                "score": 1.0,
                "description": f"{node['name']} contains {child['name']}",
            })
            interactions.extend(self._analyze_hierarchical_interactions(child, context))
        return interactions[: self.max_interactions]

    def _analyze_dependency_interactions(self, components: Dict[str, Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        by_name = {str(node.get("name", "")).lower(): cid for cid, node in components.items()}
        by_id = {cid.lower(): cid for cid in components.keys()}
        results: List[Dict[str, Any]] = []
        for cid, node in components.items():
            for dep in self._identify_dependencies(node.get("content"), context):
                dep_key = str(dep).lower()
                target = by_id.get(dep_key) or by_name.get(dep_key)
                if target and target != cid:
                    results.append({
                        "type": "dependency",
                        "source": cid,
                        "target": target,
                        "dependency": dep,
                        "score": 0.9,
                        "description": f"{node['name']} depends on {components[target]['name']}",
                    })
                if len(results) >= self.max_interactions:
                    return results
        return results

    # ------------------------------------------------------------------
    # Synthesis metrics and diagnostics
    # ------------------------------------------------------------------
    def _calculate_modularity(self, interactions: Dict[str, List[Dict[str, Any]]]) -> float:
        hierarchical = len(interactions.get("hierarchical", []))
        lateral = sum(len(interactions.get(key, [])) for key in ["functional", "structural", "data_flow", "control_flow", "dependency"])
        total = hierarchical + lateral
        return clamp_confidence(hierarchical / total) if total else 1.0

    def _calculate_cohesion(self, component_analysis: Dict[str, Dict[str, Any]], interactions: Dict[str, List[Dict[str, Any]]]) -> float:
        groups: Dict[str, List[str]] = defaultdict(list)
        for cid, analysis in component_analysis.items():
            groups[str(analysis.get("function", "unknown"))].append(cid)
        interaction_pairs = self._interaction_pairs(interactions)
        possible = 0
        actual = 0
        for members in groups.values():
            for i, left in enumerate(members):
                for right in members[i + 1:]:
                    possible += 1
                    if frozenset({left, right}) in interaction_pairs:
                        actual += 1
        return clamp_confidence(actual / possible) if possible else 0.0

    def _calculate_coupling(self, interactions: Dict[str, List[Dict[str, Any]]]) -> float:
        lateral = sum(len(interactions.get(key, [])) for key in ["functional", "structural", "data_flow", "control_flow", "dependency"])
        total = lateral + len(interactions.get("hierarchical", []))
        return clamp_confidence(lateral / total) if total else 0.0

    def _generate_insights(self, decomposition_tree: Dict[str, Any],
                           component_analysis: Dict[str, Dict[str, Any]],
                           interaction_analysis: Dict[str, List[Dict[str, Any]]],
                           context: Dict[str, Any]) -> List[str]:
        insights: List[str] = []
        complex_components = [cid for cid, a in component_analysis.items() if a.get("complexity", 0.0) >= self.high_complexity_threshold]
        if complex_components:
            insights.append(f"High-complexity components detected: {', '.join(complex_components[:5])}.")
        critical_paths = self._find_critical_paths(decomposition_tree, component_analysis)
        if critical_paths:
            insights.append(f"Critical dependency paths identified: {len(critical_paths)}.")
        cohesion = self._calculate_cohesion(component_analysis, interaction_analysis)
        coupling = self._calculate_coupling(interaction_analysis)
        if cohesion < self.low_cohesion_threshold:
            insights.append("Low cohesion detected; components may not be grouped around clear responsibilities.")
        if coupling > 0.65:
            insights.append("High coupling detected; changes may propagate across component boundaries.")
        if not insights:
            insights.append("System decomposition appears balanced under the configured thresholds.")
        return insights

    def _find_critical_paths(self, node: Dict[str, Any], component_analysis: Dict[str, Dict[str, Any]]) -> List[List[str]]:
        paths: List[List[str]] = []
        node_id = node.get("id", "")
        is_critical = component_analysis.get(node_id, {}).get("criticality", 0.0) >= self.criticality_threshold
        children = node.get("children", [])
        if not children:
            return [[node_id]] if is_critical else []
        for child in children:
            for path in self._find_critical_paths(child, component_analysis):
                if is_critical or any(component_analysis.get(cid, {}).get("criticality", 0.0) >= self.criticality_threshold for cid in path):
                    paths.append([node_id] + path)
        return paths[: self.max_interactions]

    def _identify_vulnerabilities(self, component_analysis: Dict[str, Dict[str, Any]],
                                  interaction_analysis: Dict[str, List[Dict[str, Any]]],
                                  context: Dict[str, Any]) -> List[Dict[str, Any]]:
        vulnerabilities: List[Dict[str, Any]] = []
        for cid, analysis in component_analysis.items():
            criticality = float(analysis.get("criticality", 0.0))
            if criticality >= self.vulnerability_threshold and not self._has_redundancy(cid, interaction_analysis):
                vulnerabilities.append({
                    "component": cid,
                    "type": "single_point_of_failure",
                    "severity": "high" if criticality >= 0.9 else "medium",
                    "score": criticality,
                    "description": f"Critical component {cid} has no detected redundancy.",
                })
        for interaction in interaction_analysis.get("dependency", []):
            source = interaction.get("source")
            target = interaction.get("target")
            if source in component_analysis and target in component_analysis:
                combined = self._mean([component_analysis[source].get("criticality", 0.0), component_analysis[target].get("criticality", 0.0)])
                if combined >= self.criticality_threshold:
                    vulnerabilities.append({
                        "components": [source, target],
                        "type": "critical_dependency",
                        "severity": "medium",
                        "score": combined,
                        "description": f"Critical dependency from {source} to {target}.",
                    })
        return vulnerabilities[: self.max_interactions]

    def _has_redundancy(self, comp_id: str, interactions: Dict[str, List[Dict[str, Any]]]) -> bool:
        for interaction in interactions.get("structural", []):
            if comp_id in interaction.get("components", []) and interaction.get("structural_score", 0.0) >= 0.8:
                return True
        return False

    def diagnostics(self) -> Dict[str, Any]:
        """Return runtime diagnostics for this reasoning strategy."""
        payload = {
            "module_version": self.MODULE_VERSION,
            "max_depth": self.max_depth,
            "max_components": self.max_components,
            "enable_interaction_analysis": self.enable_interaction_analysis,
            "cache_enabled": self.cache is not None,
        }
        if self.cache is not None:
            payload["cache"] = self.cache.metrics()
        return json_safe_reasoning_state(payload)

    def _format_results(self, decomposition_tree: Dict[str, Any],
                        component_analysis: Dict[str, Dict[str, Any]],
                        interaction_analysis: Dict[str, List[Dict[str, Any]]],
                        system_understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Format final results with metadata."""
        all_components = self._flatten_decomposition(decomposition_tree)
        metrics = {
            "total_components": len(all_components),
            "max_depth": max((int(comp.get("depth", 0)) for comp in all_components.values()), default=0),
            "interaction_count": sum(len(v) for v in interaction_analysis.values() if isinstance(v, list)),
            "critical_components": len(system_understanding.get("key_components", [])),
            "vulnerabilities": len(system_understanding.get("vulnerabilities", [])),
            "success": bool(all_components),
        }
        return {
            "decomposition_tree": decomposition_tree,
            "component_analysis": component_analysis,
            "interaction_analysis": interaction_analysis,
            "system_understanding": system_understanding,
            "metrics": metrics,
            "reasoning_type": "decompositional",
        }

    # ------------------------------------------------------------------
    # Low-level utility methods
    # ------------------------------------------------------------------
    def _validate_input(self, system: Any, context: Dict[str, Any]) -> None:
        if system is None:
            raise ReasoningValidationError("system cannot be None")
        if not isinstance(context, dict):
            raise ReasoningValidationError("context must be a dictionary", context={"type": type(context).__name__})
        if self.strict_inputs and isinstance(system, (str, bytes)) and not str(system).strip():
            raise ReasoningValidationError("system string cannot be empty")

    def _validate_tree(self, tree: Dict[str, Any]) -> None:
        required = {"id", "name", "type", "depth", "children"}
        if not isinstance(tree, dict) or not required.issubset(tree.keys()):
            raise ReasoningValidationError("Invalid decomposition tree", context={"required": sorted(required)})

    def _build_node(self, content: Any, context: Dict[str, Any], *, depth: int, parent_id: Optional[str]) -> ComponentNode:
        self._component_budget_used += 1
        name = self._get_component_name(content)
        path = tuple(str(parent_id or "root").split("-")) + (name,)
        node_id = self._generate_component_id(content, parent_id)
        return ComponentNode(
            id=node_id,
            name=name,
            type="composite",
            depth=depth,
            content=content,
            parent_id=parent_id,
            path=path,
            metadata=self._node_metadata(content),
        )

    def _node_from_dict(self, payload: Dict[str, Any]) -> ComponentNode:
        return ComponentNode(
            id=payload["id"],
            name=payload["name"],
            type=payload.get("type", "atomic"),
            depth=int(payload.get("depth", 0)),
            content=payload.get("content"),
            parent_id=payload.get("parent_id"),
            path=tuple(payload.get("path", [])),
            children=[self._node_from_dict(child) for child in payload.get("children", [])],
            metadata=dict(payload.get("metadata", {})),
        )

    def _is_decomposable(self, component: Any, depth: int) -> bool:
        if depth >= self.max_depth:
            return False
        return self._component_size(component) > self.min_component_size

    def _generate_component_id(self, component: Any, parent_id: Optional[str]) -> str:
        base_name = self._slug(self._get_component_name(component)) or "component"
        digest = hashlib.sha1(self._preview(component, limit=128).encode("utf-8", errors="ignore")).hexdigest()[:6]
        if parent_id:
            return f"{parent_id}_{base_name}_{digest}"
        return f"root_{base_name}_{digest}"

    def _get_component_name(self, component: Any) -> str:
        if isinstance(component, Mapping):
            for key in ("name", "id", "title", "component"):
                if component.get(key):
                    return self._truncate(str(component[key]))
        if hasattr(component, "name"):
            return self._truncate(str(getattr(component, "name")))
        if callable(component):
            return self._truncate(getattr(component, "__name__", type(component).__name__))
        if isinstance(component, str):
            return self._truncate(component)
        return self._truncate(type(component).__name__)

    def _flatten_decomposition(self, node: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        components = {node["id"]: node}
        for child in node.get("children", []):
            components.update(self._flatten_decomposition(child))
        return components

    def _iter_nodes(self, tree: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        queue: Deque[Dict[str, Any]] = deque([tree])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.get("children", []))

    def _component_values(self, components: Any) -> List[Any]:
        if isinstance(components, Mapping):
            result = []
            for key, value in components.items():
                if isinstance(value, Mapping) and "name" not in value:
                    merged = dict(value)
                    merged.setdefault("name", str(key))
                    result.append(merged)
                else:
                    result.append(self._named_child(key, value))
            return result
        if isinstance(components, (list, tuple, set)):
            return list(components)
        return [components]

    @staticmethod
    def _named_child(key: Any, value: Any) -> Any:
        if isinstance(value, Mapping):
            merged = dict(value)
            merged.setdefault("name", str(key))
            return merged
        return {"name": str(key), "value": value}

    def _component_size(self, component: Any) -> int:
        if isinstance(component, Mapping):
            if "components" in component:
                return len(component.get("components") or {})
            return len(component)
        if isinstance(component, (list, tuple, set)):
            return len(component)
        if hasattr(component, "__dict__") and not isinstance(component, type):
            return len(vars(component))
        return 1

    def _profile_component(self, node: Dict[str, Any], context: Dict[str, Any], *,
                           complexity: float, criticality: float) -> ComponentProfile:
        content = node.get("content")
        return ComponentProfile(
            component_id=node["id"],
            name=node["name"],
            depth=int(node.get("depth", 0)),
            kind=type(content).__name__,
            size=self._component_size(content),
            tokens=set(self._tokens(self._string_blob(content))),
            functions=self._extract_functions(content),
            dependencies=self._identify_dependencies(content, context),
            complexity=complexity,
            criticality=criticality,
        )

    def _extract_functions(self, component: Any) -> Set[str]:
        functions: Set[str] = set()
        if isinstance(component, Mapping):
            for key in ("function", "functions", "responsibility", "responsibilities", "capabilities"):
                if key in component:
                    functions.update(self._coerce_string_set(component[key]))
        elif callable(component):
            functions.add(getattr(component, "__name__", "callable"))
        return {item.lower() for item in functions if item}

    def _dependency_overlap_score(self, comp1: Dict[str, Any], comp2: Dict[str, Any], context: Dict[str, Any]) -> float:
        d1 = self._identify_dependencies(comp1.get("content"), context)
        d2 = self._identify_dependencies(comp2.get("content"), context)
        names = {str(comp1.get("name", "")).lower(), str(comp2.get("name", "")).lower(), comp1.get("id", "").lower(), comp2.get("id", "").lower()}
        direct = 1.0 if (str(comp1.get("name", "")).lower() in {d.lower() for d in d2} or str(comp2.get("name", "")).lower() in {d.lower() for d in d1}) else 0.0
        overlap = len(d1 & d2) / max(1, len(d1 | d2)) if (d1 or d2) else 0.0
        return clamp_confidence(max(direct, overlap))

    def _interaction_pairs(self, interactions: Dict[str, List[Dict[str, Any]]]) -> Set[frozenset]:
        pairs: Set[frozenset] = set()
        for records in interactions.values():
            if not isinstance(records, list):
                continue
            for record in records:
                comps = record.get("components")
                if comps and len(comps) == 2:
                    pairs.add(frozenset(comps))
        return pairs

    def _interaction_density(self, interactions: Dict[str, List[Dict[str, Any]]], component_count: int) -> float:
        if component_count <= 1:
            return 0.0
        lateral = sum(len(interactions.get(key, [])) for key in ["functional", "structural", "data_flow", "control_flow", "dependency"])
        possible = component_count * (component_count - 1) / 2
        return clamp_confidence(lateral / possible)

    def _calculate_balance(self, tree: Dict[str, Any]) -> float:
        depths = [int(node.get("depth", 0)) for node in self._iter_nodes(tree) if not node.get("children")]
        if not depths:
            return 1.0
        return clamp_confidence(1.0 - ((max(depths) - min(depths)) / max(1, self.max_depth)))

    def _component_health(self, complexity: float, criticality: float, dependencies: Set[str]) -> str:
        if criticality >= self.vulnerability_threshold and complexity >= self.high_complexity_threshold:
            return "high_risk"
        if complexity >= self.high_complexity_threshold or len(dependencies) >= self.max_children_per_node // 2:
            return "watch"
        return "stable"

    def _node_metadata(self, content: Any) -> Dict[str, Any]:
        metadata = {"content_type": type(content).__name__, "size": self._component_size(content)}
        if self.include_content_preview:
            metadata["preview"] = self._preview(content)
        return metadata

    def _empty_interactions(self) -> Dict[str, List[Dict[str, Any]]]:
        return {"structural": [], "functional": [], "data_flow": [], "control_flow": [], "hierarchical": [], "dependency": [], "metadata": []}

    def _cache_key(self, system: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"system": self._preview(system, limit=2000), "context": json_safe_reasoning_state(context), "version": self.MODULE_VERSION}

    def _weighted_score(self, values: Sequence[float], weights: Sequence[float]) -> float:
        safe_values = [clamp_confidence(v) for v in values]
        safe_weights = [max(0.0, float(w)) for w in weights]
        total = sum(safe_weights)
        if total <= 0.0:
            return self._mean(safe_values)
        return clamp_confidence(sum(v * w for v, w in zip(safe_values, safe_weights)) / total)

    @staticmethod
    def _mean(values: Iterable[Any]) -> float:
        vals = [float(v) for v in values]
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def _coerce_string_set(value: Any) -> Set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            return {value}
        if isinstance(value, Mapping):
            return {str(key) for key in value.keys()}
        if isinstance(value, Iterable):
            return {str(item) for item in value if item is not None}
        return {str(value)}

    def _tokens(self, value: Any) -> List[str]:
        return [token for token in re.split(r"[^A-Za-z0-9]+", str(value).lower()) if token][: self.max_property_tokens]

    @property
    def max_property_tokens(self) -> int:
        return bounded_iterations(self.decompositional_config.get("max_property_tokens", 256), minimum=8, maximum=100_000)

    @staticmethod
    def _string_blob(value: Any) -> str:
        if isinstance(value, Mapping):
            return " ".join([str(k) + " " + ReasoningDecompositional._string_blob(v) for k, v in value.items()])
        if isinstance(value, (list, tuple, set)):
            return " ".join(ReasoningDecompositional._string_blob(v) for v in value)
        return str(value)

    def _preview(self, value: Any, *, limit: Optional[int] = None) -> str:
        text = str(value)
        max_len = limit or self.content_preview_chars
        return text if len(text) <= max_len else f"{text[:max_len - 15]}...<truncated>"

    def _truncate(self, value: str) -> str:
        text = value.strip() or "component"
        return text if len(text) <= self.max_name_length else f"{text[:self.max_name_length - 3]}..."

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
        return slug[:80]


if __name__ == "__main__":
    print("\n=== Running Reasoning Decompositional ===\n")
    printer.status("TEST", "Reasoning Decompositional initialized", "info")
    engine = ReasoningDecompositional()
    system = {
        "name": "Commerce Platform",
        "components": {
            "auth": {"functions": ["login", "profile"], "dependencies": ["database"]},
            "catalog": {"functions": ["search", "inventory"], "dependencies": ["database"]},
            "orders": {"functions": ["checkout"], "dependencies": ["auth", "catalog", "payments"]},
            "payments": {"functions": ["authorization"], "dependencies": ["external_api"]},
            "database": {"functions": ["storage"]},
        },
    }
    ctx = {
        "decomposition_strategy": "structural",
        "critical_components": ["database", "payments"],
        "function_mapping": {"auth": "Identity control", "payment": "Financial processing"},
    }
    result = engine.perform_reasoning(system, ctx)
    assert result["metrics"]["total_components"] >= 5
    assert result["metrics"]["success"] is True
    assert result["system_understanding"]["system_metrics"]["component_count"] >= 5
    printer.pretty("Decompositional metrics", result["metrics"])
    print("\n=== Test ran successfully ===\n")
