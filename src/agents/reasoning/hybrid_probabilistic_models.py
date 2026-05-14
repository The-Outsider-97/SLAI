from __future__ import annotations

import copy
import itertools
import json
import math
import re

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.reasoning_errors import *
from .utils.reasoning_helpers import *
from .reasoning_memory import ReasoningMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Hybrid Models")
printer = PrettyPrinter()

NetworkDict = Dict[str, Any]
Edge = List[str]
Region = Tuple[int, int, int, int]

_RESERVED_CPT_KEYS: Set[str] = {"prior", "type", "parents", "threshold", "probabilities", "metadata", "description"}


@dataclass(frozen=True)
class HybridStrategySpec:
    """Resolved strategy plan used by ``select_hybrid_network``."""

    strategy: str
    base_bn_key: str
    base_grid_key: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = "task_profile"


@dataclass(frozen=True)
class HybridBuildReport:
    """Compact diagnostics for a generated hybrid network."""

    strategy: str
    node_count: int
    edge_count: int
    cpt_count: int
    cycle_free: bool
    functional_cpt_count: int
    warnings: Tuple[str, ...] = ()


class HybridProbabilisticModels:
    """
    Production hybrid Bayesian/Grid network builder.

    The class combines semantic Bayesian networks with spatial grid networks
    while keeping the public API expected by ``ReasoningOrchestrator`` and
    older callers:
    - ``select_hybrid_network(task_description)``
    - ``create_hybrid_network(base_bn_path, base_grid_path, connection_strategy, connection_params)``
    - helper methods such as ``_create_aggregator_cpt`` and ``_get_grid_nodes_in_region``

    The implementation removes duplicate strategy branches from the legacy
    module, validates network definitions, deduplicates edges, bounds explicit
    CPT expansion, and records explainable metadata for downstream reasoning.
    """

    SUPPORTED_STRATEGIES: Tuple[str, ...] = (
        "global_to_local",
        "local_to_global",
        "regional",
        "multi_regional",
        "hierarchical_aggregation",
        "feedback_loop",
        "pathway_influence",
        "robotics_nav_feedback",
        "player_influence_aoe",
        "pcg_biome_generation",
    )

    _STRATEGY_DEFAULTS: Dict[str, Dict[str, Any]] = {
        "global_to_local": {
            "base_bn_key": "bn2x2",
            "base_grid_key": "gn3x3",
            "params": {"global_node": "X"},
        },
        "local_to_global": {
            "base_bn_key": "bn2x2",
            "base_grid_key": "gn2x2",
            "params": {"aggregator_node_name": "Agg_Sensor", "target_node": "Y"},
        },
        "regional": {
            "base_bn_key": "bn2x2",
            "base_grid_key": "gn4x4",
            "params": {"source_node": "X", "region_coords": (0, 0, 1, 1)},
        },
        "multi_regional": {
            "base_bn_key": "bn6x6",
            "base_grid_key": "gn6x6",
            "params": {"source_nodes": ["A", "D"], "regions": [(0, 0, 2, 2), (3, 3, 5, 5)]},
        },
        "hierarchical_aggregation": {
            "base_bn_key": "bn7x7",
            "base_grid_key": "gn4x4",
            "params": {"leaf_nodes": ["D", "E", "F", "G"]},
        },
        "feedback_loop": {
            "base_bn_key": "bn3x3",
            "base_grid_key": "gn3x3",
            "params": {"initial_cause": "X", "aggregator_node_name": "Grid_State_Aggregator", "feedback_target": "Y"},
        },
        "pathway_influence": {
            "base_bn_key": "bn2x2",
            "base_grid_key": "gn4x4",
            "params": {"source_node": "X", "pathway_nodes": ["N00", "N11", "N22", "N33"]},
        },
        "robotics_nav_feedback": {
            "base_bn_key": "bn3x3",
            "base_grid_key": "gn4x4",
            "params": {
                "task_node": "X",
                "action_node": "Y",
                "aggregator_node_name": "LIDAR_Obstacle",
                "grid_sensor_prefix": "Sensor_",
            },
        },
        "player_influence_aoe": {
            "base_bn_key": "bn4x4",
            "base_grid_key": "gn6x6",
            "params": {"player_pos": (3, 3), "influence_radius": 2, "target_nodes": ["Y", "Z", "W"]},
        },
        "pcg_biome_generation": {
            "base_bn_key": "bn4x4",
            "base_grid_key": "gn5x5",
            "params": {"global_property_nodes": ["Y", "Z", "W"], "grid_tile_prefix": "Tile_"},
        },
    }

    def __init__(self, memory: Optional[ReasoningMemory] = None):
        super().__init__()
        self.config = load_global_config()
        self.net_config = get_config_section("networks") or {}
        self.hybrid_config = get_config_section("hybrid_models") or {}

        self.memory = memory or ReasoningMemory()
        self.hybrid_networks_cache: "OrderedDict[str, NetworkDict]" = OrderedDict()
        self._source_network_cache: "OrderedDict[str, NetworkDict]" = OrderedDict()
        self._refresh_runtime_config()
        logger.info("Hybrid Probabilistic Models initialized.")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _refresh_runtime_config(self) -> None:
        """Read and validate runtime knobs from ``reasoning_config.yaml``."""
        cfg = self.hybrid_config
        self.cache_enabled = bool(cfg.get("cache_enabled", True))
        self.cache_max_size = bounded_iterations(cfg.get("cache_max_size", 128), minimum=1, maximum=100_000)
        self.source_cache_max_size = bounded_iterations(cfg.get("source_cache_max_size", 64), minimum=1, maximum=100_000)
        self.validate_on_create = bool(cfg.get("validate_on_create", True))
        self.validate_acyclic = bool(cfg.get("validate_acyclic", True))
        self.deduplicate_edges = bool(cfg.get("deduplicate_edges", True))
        self.allow_missing_cpt = bool(cfg.get("allow_missing_cpt", True))
        self.strict_cpt_validation = bool(cfg.get("strict_cpt_validation", False))
        self.namespace_on_collision = bool(cfg.get("namespace_on_collision", False))
        self.functional_cpt_fallback = bool(cfg.get("functional_cpt_fallback", True))
        self.record_memory_events = bool(cfg.get("record_memory_events", True))
        self.default_bn_key = str(cfg.get("default_bn_key", "bn2x2"))
        self.default_grid_key = str(cfg.get("default_grid_key", "gn2x2"))
        self.max_strategy_candidates = bounded_iterations(cfg.get("max_strategy_candidates", 8), minimum=1, maximum=64)
        self.max_explicit_cpt_parents = bounded_iterations(cfg.get("max_explicit_cpt_parents", 12), minimum=0, maximum=32)

        self.default_probability = clamp_confidence(cfg.get("default_probability", 0.5))
        self.min_probability = clamp_confidence(cfg.get("min_probability", 0.01))
        self.max_probability = clamp_confidence(cfg.get("max_probability", 0.99))
        self.influence_step = float(cfg.get("influence_step", 0.1))
        self.aggregator_threshold = clamp_confidence(cfg.get("aggregator_threshold", 0.5))
        self.aggregator_true_probability = clamp_confidence(cfg.get("aggregator_true_probability", 0.95))
        self.aggregator_false_probability = clamp_confidence(cfg.get("aggregator_false_probability", 0.05))
        self.or_gate_true_probability = clamp_confidence(cfg.get("or_gate_true_probability", 0.95))
        self.or_gate_false_probability = clamp_confidence(cfg.get("or_gate_false_probability", 0.01))

        if self.min_probability > self.max_probability:
            raise ReasoningConfigurationError(
                "hybrid_models.min_probability must be <= max_probability",
                context={"min_probability": self.min_probability, "max_probability": self.max_probability},
            )
        if self.influence_step < 0.0:
            raise ReasoningConfigurationError("hybrid_models.influence_step must be non-negative")

        dims = cfg.get("preferred_grid_dimensions", [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 32, 64])
        if not isinstance(dims, Sequence) or isinstance(dims, (str, bytes)):
            raise ReasoningConfigurationError("hybrid_models.preferred_grid_dimensions must be a sequence")
        self.preferred_grid_dimensions = [bounded_iterations(dim, minimum=1, maximum=10_000) for dim in dims]

    # ------------------------------------------------------------------
    # High-level selection
    # ------------------------------------------------------------------
    def select_hybrid_network(self, task_description: str) -> NetworkDict:
        """
        Select and build a hybrid network for a natural-language task.

        The selector is intentionally deterministic: strategy keywords map to
        a resolved ``HybridStrategySpec`` and task dimensions such as ``4x4``
        are used only to refine the selected grid network key.
        """
        if not isinstance(task_description, str) or not task_description.strip():
            raise ReasoningValidationError("task_description must be a non-empty string")

        spec = self.resolve_strategy(task_description)
        logger.info(
            f"Selected hybrid strategy '{spec.strategy}' using BN '{spec.base_bn_key}' "
            f"and grid '{spec.base_grid_key}'."
        )
        return self.create_hybrid_network(
            base_bn_path=self._network_path(spec.base_bn_key),
            base_grid_path=self._network_path(spec.base_grid_key),
            connection_strategy=spec.strategy,
            connection_params=spec.params,
        )

    def resolve_strategy(self, task_description: str) -> HybridStrategySpec:
        """Resolve task text into a concrete hybrid strategy specification."""
        task_lower = task_description.lower().strip()
        explicit_dim = self._extract_square_dimension(task_lower)

        if ("robot" in task_lower or "robotic" in task_lower) and "navigation" in task_lower:
            return self._spec("robotics_nav_feedback", explicit_dim, "robot_navigation_keywords")
        if "procedural" in task_lower and ("generation" in task_lower or "generate" in task_lower):
            return self._spec("pcg_biome_generation", explicit_dim, "pcg_keywords")
        if "player" in task_lower and ("influence" in task_lower or "radius" in task_lower or "aoe" in task_lower):
            return self._spec("player_influence_aoe", explicit_dim, "player_influence_keywords")
        if "pathway" in task_lower or "conduction" in task_lower or "corridor" in task_lower:
            return self._spec("pathway_influence", explicit_dim, "pathway_keywords")
        if "feedback" in task_lower:
            return self._spec("feedback_loop", explicit_dim, "feedback_keyword")
        if "hierarchical" in task_lower and ("aggregation" in task_lower or "aggregate" in task_lower):
            return self._spec("hierarchical_aggregation", explicit_dim, "hierarchical_aggregation_keywords")
        if "multi-regional" in task_lower or "multiple zones" in task_lower or "multi regional" in task_lower:
            return self._spec("multi_regional", explicit_dim, "multi_regional_keywords")
        if "aggregate" in task_lower or "aggregation" in task_lower or "alarm" in task_lower:
            return self._spec("local_to_global", explicit_dim, "local_to_global_keywords")
        if "regional" in task_lower or "zone" in task_lower or "irrigation" in task_lower:
            return self._spec("regional", explicit_dim, "regional_keywords")
        if "global" in task_lower or "forecast" in task_lower or "affecting" in task_lower or "influencing" in task_lower:
            return self._spec("global_to_local", explicit_dim, "global_to_local_keywords")

        default_strategy = str(self.hybrid_config.get("default_strategy", "global_to_local"))
        if default_strategy not in self.SUPPORTED_STRATEGIES:
            default_strategy = "global_to_local"
        logger.warning(f"Could not determine hybrid strategy from task: '{task_description}'. Using default.")
        return self._spec(default_strategy, explicit_dim, "fallback_default")

    def _spec(self, strategy: str, explicit_dim: Optional[int], reason: str) -> HybridStrategySpec:
        defaults = copy.deepcopy(self._STRATEGY_DEFAULTS.get(strategy, {}))
        strategy_overrides = (self.hybrid_config.get("strategy_defaults") or {}).get(strategy, {})
        defaults.update(copy.deepcopy(strategy_overrides))

        bn_key = str(defaults.get("base_bn_key") or defaults.get("bn") or self.default_bn_key)
        grid_key = str(defaults.get("base_grid_key") or defaults.get("grid") or self.default_grid_key)
        params = dict(defaults.get("params", {}))
        if "connection_params" in defaults and isinstance(defaults["connection_params"], Mapping):
            params.update(dict(defaults["connection_params"]))

        if explicit_dim is not None:
            candidate = f"gn{explicit_dim}x{explicit_dim}"
            if candidate in self.net_config:
                grid_key = candidate
        return HybridStrategySpec(strategy=strategy, base_bn_key=bn_key, base_grid_key=grid_key, params=params, reason=reason)

    @staticmethod
    def _extract_square_dimension(task_lower: str) -> Optional[int]:
        match = re.search(r"\b(\d{1,3})\s*x\s*\1\b", task_lower)
        if match:
            return int(match.group(1))
        match = re.search(r"\b(\d{1,3})\s+by\s+\1\b", task_lower)
        if match:
            return int(match.group(1))
        return None

    # ------------------------------------------------------------------
    # Build orchestration
    # ------------------------------------------------------------------
    def create_hybrid_network(
        self,
        base_bn_path: str,
        base_grid_path: str,
        connection_strategy: str,
        connection_params: Optional[Dict[str, Any]] = None,
    ) -> NetworkDict:
        """
        Dynamically construct a hybrid network from a Bayesian and grid network.

        Args:
            base_bn_path: Path to a semantic/causal network JSON file.
            base_grid_path: Path to a grid/spatial network JSON file.
            connection_strategy: A strategy listed in ``SUPPORTED_STRATEGIES``.
            connection_params: Strategy-specific parameters.

        Returns:
            Complete hybrid network dictionary containing ``nodes``, ``edges``,
            ``cpt``, ``description`` and structured ``metadata``.
        """
        strategy = str(connection_strategy or "").strip()
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ReasoningValidationError(
                "Unsupported hybrid connection strategy",
                context={"strategy": strategy, "supported": list(self.SUPPORTED_STRATEGIES)},
            )
        params = dict(connection_params or {})

        cache_key = self._cache_key(base_bn_path, base_grid_path, strategy, params)
        if self.cache_enabled and cache_key in self.hybrid_networks_cache:
            self.hybrid_networks_cache.move_to_end(cache_key)
            logger.info("Returning cached hybrid network.")
            return copy.deepcopy(self.hybrid_networks_cache[cache_key])

        base_bn = self.load_network(base_bn_path, role="bayesian")
        base_grid = self.load_network(base_grid_path, role="grid")

        if strategy == "robotics_nav_feedback":
            base_grid = self._rename_grid_network(base_grid, prefix=str(params.get("grid_sensor_prefix", "Sensor_")))
        elif strategy == "pcg_biome_generation":
            base_grid = self._rename_grid_network(base_grid, prefix=str(params.get("grid_tile_prefix", "Tile_")))

        hybrid_network = self._merge_networks(base_bn, base_grid, base_bn_path, base_grid_path, strategy)
        self._apply_strategy(hybrid_network, base_bn, base_grid, strategy, params)
        self._finalize_hybrid(hybrid_network)

        if self.validate_on_create:
            diagnostics = self.validate_hybrid_network(hybrid_network)
            if not diagnostics["valid"]:
                raise ProbabilisticModelError(
                    "Generated hybrid network failed validation",
                    context={"strategy": strategy, "diagnostics": diagnostics},
                )
            hybrid_network["metadata"]["validation"] = diagnostics

        report = self._build_report(hybrid_network)
        hybrid_network["metadata"]["build_report"] = report.__dict__
        self._record_memory_event(hybrid_network)

        if self.cache_enabled:
            self.hybrid_networks_cache[cache_key] = copy.deepcopy(hybrid_network)
            self.hybrid_networks_cache.move_to_end(cache_key)
            self._trim_cache(self.hybrid_networks_cache, self.cache_max_size)
        return hybrid_network

    def create_hybrid_network_from_data(
        self,
        base_bn: Mapping[str, Any],
        base_grid: Mapping[str, Any],
        connection_strategy: str,
        connection_params: Optional[Dict[str, Any]] = None,
    ) -> NetworkDict:
        """Build a hybrid network from already-loaded network dictionaries."""
        self._validate_base_network(base_bn, role="bayesian")
        self._validate_base_network(base_grid, role="grid")
        strategy = str(connection_strategy or "").strip()
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ReasoningValidationError("Unsupported hybrid connection strategy", context={"strategy": strategy})
        params = dict(connection_params or {})
        bn_copy = copy.deepcopy(dict(base_bn))
        grid_copy = copy.deepcopy(dict(base_grid))
        if strategy == "robotics_nav_feedback":
            grid_copy = self._rename_grid_network(grid_copy, prefix=str(params.get("grid_sensor_prefix", "Sensor_")))
        elif strategy == "pcg_biome_generation":
            grid_copy = self._rename_grid_network(grid_copy, prefix=str(params.get("grid_tile_prefix", "Tile_")))
        hybrid_network = self._merge_networks(bn_copy, grid_copy, "<dict:bayesian>", "<dict:grid>", strategy)
        self._apply_strategy(hybrid_network, bn_copy, grid_copy, strategy, params)
        self._finalize_hybrid(hybrid_network)
        if self.validate_on_create:
            diagnostics = self.validate_hybrid_network(hybrid_network)
            if not diagnostics["valid"]:
                raise ProbabilisticModelError("Generated hybrid network failed validation", context={"diagnostics": diagnostics})
            hybrid_network["metadata"]["validation"] = diagnostics
        hybrid_network["metadata"]["build_report"] = self._build_report(hybrid_network).__dict__
        return hybrid_network

    def _apply_strategy(
        self,
        hybrid_network: NetworkDict,
        base_bn: NetworkDict,
        base_grid: NetworkDict,
        strategy: str,
        params: Dict[str, Any],
    ) -> None:
        handlers = {
            "global_to_local": self._apply_global_to_local,
            "local_to_global": self._apply_local_to_global,
            "regional": self._apply_regional,
            "multi_regional": self._apply_multi_regional,
            "hierarchical_aggregation": self._apply_hierarchical_aggregation,
            "feedback_loop": self._apply_feedback_loop,
            "pathway_influence": self._apply_pathway_influence,
            "robotics_nav_feedback": self._apply_robotics_nav_feedback,
            "player_influence_aoe": self._apply_player_influence_aoe,
            "pcg_biome_generation": self._apply_pcg_biome_generation,
        }
        handlers[strategy](hybrid_network, base_bn, base_grid, params)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------
    def _apply_global_to_local(self, hybrid: NetworkDict, base_bn: NetworkDict, base_grid: NetworkDict, params: Dict[str, Any]) -> None:
        global_node = str(params.get("global_node", "X"))
        self._require_node(hybrid, global_node, context="global_node")
        for grid_node in base_grid["nodes"]:
            self._add_edge_and_expand_cpt(hybrid, global_node, grid_node)
        hybrid["metadata"]["connection_summary"] = {"source": global_node, "targets": list(base_grid["nodes"])}

    def _apply_local_to_global(self, hybrid: NetworkDict, base_bn: NetworkDict, base_grid: NetworkDict, params: Dict[str, Any]) -> None:
        agg_node = str(params.get("aggregator_node_name", "Agg_Sensor"))
        target_node = str(params.get("target_node", "Y"))
        self._require_node(hybrid, target_node, context="target_node")
        self._add_node(hybrid, agg_node)
        parents = list(base_grid["nodes"])
        for grid_node in parents:
            self._add_edge(hybrid, grid_node, agg_node)
        hybrid["cpt"][agg_node] = self._create_aggregator_cpt(parents)
        self._add_edge_and_expand_cpt(hybrid, agg_node, target_node)
        hybrid["metadata"]["connection_summary"] = {"aggregator": agg_node, "parents": parents, "target": target_node}

    def _apply_regional(self, hybrid: NetworkDict, base_bn: NetworkDict, base_grid: NetworkDict, params: Dict[str, Any]) -> None:
        source_node = str(params.get("source_node", "X"))
        region = self._normalize_region(params.get("region_coords", (0, 0, 1, 1)))
        self._require_node(hybrid, source_node, context="source_node")
        region_nodes = self._get_grid_nodes_in_region(base_grid, region)
        for grid_node in region_nodes:
            if grid_node in hybrid["nodes"]:
                self._add_edge_and_expand_cpt(hybrid, source_node, grid_node)
        hybrid["metadata"]["connection_summary"] = {"source": source_node, "region": region, "targets": region_nodes}

    def _apply_multi_regional(self, hybrid: NetworkDict, base_bn: NetworkDict, base_grid: NetworkDict, params: Dict[str, Any]) -> None:
        source_nodes = [str(node) for node in params.get("source_nodes", ["A", "D"])]
        regions = [self._normalize_region(region) for region in params.get("regions", [(0, 0, 2, 2), (3, 3, 5, 5)])]
        if len(source_nodes) != len(regions):
            raise ReasoningValidationError(
                "The number of source nodes must match the number of regions",
                context={"source_nodes": source_nodes, "regions": regions},
            )
        summary: Dict[str, List[str]] = {}
        for source_node, region in zip(source_nodes, regions):
            self._require_node(hybrid, source_node, context="source_node")
            region_nodes = self._get_grid_nodes_in_region(base_grid, region)
            summary[source_node] = region_nodes
            for grid_node in region_nodes:
                if grid_node in hybrid["nodes"]:
                    self._add_edge_and_expand_cpt(hybrid, source_node, grid_node)
        hybrid["metadata"]["connection_summary"] = {"regions": summary}

    def _apply_hierarchical_aggregation(self, hybrid: NetworkDict, base_bn: NetworkDict, base_grid: NetworkDict, params: Dict[str, Any]) -> None:
        target_nodes = params.get("leaf_nodes") or params.get("hub_nodes") or ["D", "E", "F", "G"]
        target_nodes = [str(node) for node in target_nodes]
        available_targets = [node for node in target_nodes if node in hybrid["nodes"]]
        if not available_targets:
            raise ReasoningValidationError("hierarchical_aggregation requires at least one existing target node", context={"target_nodes": target_nodes})

        grid_dim = self._infer_grid_dimension(base_grid)
        regions = self._partition_grid_regions(grid_dim, len(available_targets))
        summary: Dict[str, List[str]] = {}
        for target_node, region in zip(available_targets, regions):
            region_nodes = self._get_grid_nodes_in_region(base_grid, region)
            summary[target_node] = region_nodes
            for grid_node in region_nodes:
                if grid_node in hybrid["nodes"]:
                    self._add_edge(hybrid, grid_node, target_node)
            if region_nodes:
                self._expand_cpt_for_new_parents(hybrid, target_node, region_nodes)
        hybrid["metadata"]["connection_summary"] = {"hierarchy": summary}

    def _apply_feedback_loop(self, hybrid: NetworkDict, base_bn: NetworkDict, base_grid: NetworkDict, params: Dict[str, Any]) -> None:
        initial_cause = str(params.get("initial_cause", "X"))
        agg_node = str(params.get("aggregator_node_name", "Grid_State_Aggregator"))
        feedback_target = str(params.get("feedback_target") or params.get("intermediary_effect") or "Y")
        self._require_node(hybrid, initial_cause, context="initial_cause")
        self._require_node(hybrid, feedback_target, context="feedback_target")

        for grid_node in base_grid["nodes"]:
            self._add_edge_and_expand_cpt(hybrid, initial_cause, grid_node)

        self._add_node(hybrid, agg_node)
        for grid_node in base_grid["nodes"]:
            self._add_edge(hybrid, grid_node, agg_node)
        hybrid["cpt"][agg_node] = self._create_aggregator_cpt(list(base_grid["nodes"]))

        if bool(params.get("remove_direct_cause_edge", True)):
            self._remove_edge(hybrid, initial_cause, feedback_target)
        self._add_edge_and_expand_cpt(hybrid, agg_node, feedback_target)
        hybrid["metadata"]["connection_summary"] = {
            "initial_cause": initial_cause,
            "aggregator": agg_node,
            "feedback_target": feedback_target,
        }

    def _apply_pathway_influence(self, hybrid: NetworkDict, base_bn: NetworkDict, base_grid: NetworkDict, params: Dict[str, Any]) -> None:
        source_node = str(params.get("source_node", "X"))
        pathway_nodes = [str(node) for node in params.get("pathway_nodes", [])]
        if not pathway_nodes:
            grid_dim = self._infer_grid_dimension(base_grid)
            pathway_nodes = [f"N{i}{i}" for i in range(grid_dim)]
        self._require_node(hybrid, source_node, context="source_node")
        connected: List[str] = []
        for grid_node in pathway_nodes:
            if grid_node in hybrid["nodes"]:
                self._add_edge_and_expand_cpt(hybrid, source_node, grid_node)
                connected.append(grid_node)
        hybrid["metadata"]["connection_summary"] = {"source": source_node, "pathway_nodes": connected}

    def _apply_robotics_nav_feedback(self, hybrid: NetworkDict, base_bn: NetworkDict, base_grid: NetworkDict, params: Dict[str, Any]) -> None:
        task_node = str(params.get("task_node", "X"))
        action_node = str(params.get("action_node", "Y"))
        agg_node = str(params.get("aggregator_node_name", "LIDAR_Obstacle"))
        sensor_nodes = list(base_grid["nodes"])
        self._require_node(hybrid, task_node, context="task_node")
        self._require_node(hybrid, action_node, context="action_node")
        self._add_node(hybrid, agg_node)
        for sensor_node in sensor_nodes:
            self._add_edge(hybrid, sensor_node, agg_node)
        hybrid["cpt"][agg_node] = self._create_or_gate_cpt(sensor_nodes)
        if bool(params.get("replace_task_action_edge", True)):
            self._remove_edge(hybrid, task_node, action_node)
        self._add_edge_and_expand_cpt(hybrid, agg_node, action_node)
        hybrid["metadata"]["connection_summary"] = {
            "task_node": task_node,
            "action_node": action_node,
            "aggregator": agg_node,
            "sensor_count": len(sensor_nodes),
        }

    def _apply_player_influence_aoe(self, hybrid: NetworkDict, base_bn: NetworkDict, base_grid: NetworkDict, params: Dict[str, Any]) -> None:
        player_pos = params.get("player_pos", (3, 3))
        radius = int(params.get("influence_radius", 2))
        target_nodes = [str(node) for node in params.get("target_nodes", ["Y", "Z", "W"])]
        if not isinstance(player_pos, Sequence) or len(player_pos) != 2:
            raise ReasoningValidationError("player_pos must be a two-item coordinate", context={"player_pos": player_pos})
        row, col = int(player_pos[0]), int(player_pos[1])
        region = (col - radius, row - radius, col + radius, row + radius)
        nodes_in_radius = self._get_grid_nodes_in_region(base_grid, region)
        summary: Dict[str, List[str]] = {}
        for target_node in target_nodes:
            if target_node not in hybrid["nodes"]:
                logger.warning(f"Skipping missing AOE target node '{target_node}'")
                continue
            summary[target_node] = []
            for grid_node in nodes_in_radius:
                if grid_node in hybrid["nodes"]:
                    self._add_edge(hybrid, grid_node, target_node)
                    summary[target_node].append(grid_node)
            hybrid["cpt"][target_node] = self._create_or_gate_cpt(summary[target_node])
        hybrid["metadata"]["connection_summary"] = {"player_pos": tuple(player_pos), "radius": radius, "targets": summary}

    def _apply_pcg_biome_generation(self, hybrid: NetworkDict, base_bn: NetworkDict, base_grid: NetworkDict, params: Dict[str, Any]) -> None:
        prop_nodes = [str(node) for node in params.get("global_property_nodes", ["Y", "Z", "W"])]
        for prop_node in prop_nodes:
            self._require_node(hybrid, prop_node, context="global_property_node")
        tile_nodes = list(base_grid["nodes"])
        for tile_node in tile_nodes:
            for prop_node in prop_nodes:
                self._add_edge(hybrid, prop_node, tile_node)
            self._expand_cpt_for_new_parents(hybrid, tile_node, prop_nodes)
        hybrid["metadata"]["connection_summary"] = {"global_properties": prop_nodes, "tile_count": len(tile_nodes)}

    # ------------------------------------------------------------------
    # Loading / validation
    # ------------------------------------------------------------------
    def load_network(self, network_path: Union[str, Path], *, role: str = "network") -> NetworkDict:
        """Load and validate a JSON network definition with bounded caching."""
        path = str(Path(network_path).expanduser())
        if self.cache_enabled and path in self._source_network_cache:
            self._source_network_cache.move_to_end(path)
            return copy.deepcopy(self._source_network_cache[path])

        resolved = Path(path)
        if not resolved.exists():
            raise ResourceLoadError(f"{role} network file not found", context={"path": path})
        try:
            with resolved.open("r", encoding="utf-8") as handle:
                network = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ResourceLoadError(f"Invalid JSON in {role} network", cause=exc, context={"path": path}) from exc
        except OSError as exc:
            raise ResourceLoadError(f"Failed to read {role} network", cause=exc, context={"path": path}) from exc

        self._validate_base_network(network, role=role, source=path)
        if self.cache_enabled:
            self._source_network_cache[path] = copy.deepcopy(network)
            self._source_network_cache.move_to_end(path)
            self._trim_cache(self._source_network_cache, self.source_cache_max_size)
        return network

    def validate_hybrid_network(self, network: Mapping[str, Any]) -> Dict[str, Any]:
        """Validate hybrid structure and return detailed diagnostics."""
        diagnostics: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "node_count": 0,
            "edge_count": 0,
            "cpt_count": 0,
            "cycle_free": True,
            "functional_cpt_count": 0,
        }
        try:
            self._validate_base_network(network, role="hybrid")
        except ReasoningError as exc:
            diagnostics["valid"] = False
            diagnostics["errors"].append(exc.to_payload() if hasattr(exc, "to_payload") else str(exc))
            return diagnostics

        nodes = list(network.get("nodes", []))
        edges = list(network.get("edges", []))
        cpt = dict(network.get("cpt", {}))
        diagnostics.update(
            {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "cpt_count": len(cpt),
                "functional_cpt_count": sum(1 for value in cpt.values() if isinstance(value, Mapping) and value.get("type")),
            }
        )

        missing_cpt = [node for node in nodes if node not in cpt]
        if missing_cpt and not self.allow_missing_cpt:
            diagnostics["valid"] = False
            diagnostics["errors"].append({"message": "Missing CPT entries", "nodes": missing_cpt})
        elif missing_cpt:
            diagnostics["warnings"].append({"message": "Missing CPT entries use implicit default", "nodes": missing_cpt[:20]})

        if self.validate_acyclic:
            cycle = self._find_cycle(nodes, edges)
            diagnostics["cycle_free"] = cycle is None
            if cycle is not None:
                diagnostics["valid"] = False
                diagnostics["errors"].append({"message": "Network contains a directed cycle", "cycle": cycle})

        cpt_warnings = self._validate_cpt_shapes(network)
        diagnostics["warnings"].extend(cpt_warnings)
        if cpt_warnings and self.strict_cpt_validation:
            diagnostics["valid"] = False
            diagnostics["errors"].extend(cpt_warnings)
        return diagnostics

    def _validate_base_network(self, network: Mapping[str, Any], *, role: str, source: Optional[str] = None) -> None:
        if not isinstance(network, Mapping):
            raise ModelInitializationError(f"{role} network must be a dictionary", context={"source": source})
        nodes = network.get("nodes")
        edges = network.get("edges", [])
        cpt = network.get("cpt", {})
        if not isinstance(nodes, list) or not nodes:
            raise ModelInitializationError(f"{role} network must contain a non-empty 'nodes' list", context={"source": source})
        if len(nodes) != len(set(map(str, nodes))):
            raise ModelInitializationError(f"{role} network contains duplicate nodes", context={"source": source})
        if not isinstance(edges, list):
            raise ModelInitializationError(f"{role} network 'edges' must be a list", context={"source": source})
        node_set = set(map(str, nodes))
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise ModelInitializationError(f"{role} network contains invalid edge", context={"edge": edge, "source": source})
            parent, child = str(edge[0]), str(edge[1])
            if parent == child:
                raise ModelInitializationError(f"{role} network contains a self-loop", context={"edge": edge, "source": source})
            if parent not in node_set or child not in node_set:
                raise ModelInitializationError(
                    f"{role} network edge references missing node",
                    context={"edge": edge, "source": source, "missing": [n for n in (parent, child) if n not in node_set]},
                )
        if cpt is not None and not isinstance(cpt, Mapping):
            raise ModelInitializationError(f"{role} network 'cpt' must be a mapping", context={"source": source})

    # ------------------------------------------------------------------
    # Network composition helpers
    # ------------------------------------------------------------------
    def _merge_networks(self, base_bn: NetworkDict, base_grid: NetworkDict, bn_path: str, grid_path: str, strategy: str) -> NetworkDict:
        bn_nodes = [str(node) for node in base_bn["nodes"]]
        grid_nodes = [str(node) for node in base_grid["nodes"]]
        collisions = sorted(set(bn_nodes) & set(grid_nodes))
        if collisions and not self.namespace_on_collision:
            raise ModelInitializationError(
                "Base BN and grid network share node names; enable namespace_on_collision or rename inputs",
                context={"collisions": collisions[:20]},
            )
        if collisions:
            base_grid = self._rename_grid_network(base_grid, prefix="Grid_")
            grid_nodes = [str(node) for node in base_grid["nodes"]]

        merged_nodes = self._unique_list(bn_nodes + grid_nodes)
        merged_edges = self._normalize_edges(list(base_bn.get("edges", [])) + list(base_grid.get("edges", [])))
        merged_cpt = {**copy.deepcopy(base_bn.get("cpt", {})), **copy.deepcopy(base_grid.get("cpt", {}))}
        return {
            "nodes": merged_nodes,
            "edges": merged_edges,
            "cpt": merged_cpt,
            "description": f"Hybrid of '{base_bn.get('description', 'BN')}' and '{base_grid.get('description', 'Grid')}'",
            "metadata": {
                "hybrid_strategy": strategy,
                "base_bn_path": str(bn_path),
                "base_grid_path": str(grid_path),
                "base_bn_nodes": len(bn_nodes),
                "base_grid_nodes": len(grid_nodes),
                "created_at_ms": monotonic_timestamp_ms(),
                "functional_cpts": [],
                "warnings": [],
            },
        }

    def _rename_grid_network(self, grid: NetworkDict, *, prefix: str) -> NetworkDict:
        """Return a grid network copy with nodes renamed by index using a prefix."""
        renamed = copy.deepcopy(grid)
        original_nodes = [str(node) for node in grid.get("nodes", [])]
        node_map = {node: f"{prefix}{idx}" for idx, node in enumerate(original_nodes)}
        renamed["nodes"] = [node_map[node] for node in original_nodes]
        renamed["edges"] = [[node_map[str(parent)], node_map[str(child)]] for parent, child in grid.get("edges", [])]
        renamed["cpt"] = {node_map.get(str(node), str(node)): copy.deepcopy(value) for node, value in dict(grid.get("cpt", {})).items()}
        renamed.setdefault("metadata", {})
        renamed["metadata"]["node_map"] = node_map
        return renamed

    def _finalize_hybrid(self, network: NetworkDict) -> None:
        if self.deduplicate_edges:
            network["edges"] = self._normalize_edges(network.get("edges", []))
        network["nodes"] = self._unique_list([str(node) for node in network.get("nodes", [])])
        network["metadata"]["node_count"] = len(network["nodes"])
        network["metadata"]["edge_count"] = len(network["edges"])
        network["metadata"]["cpt_count"] = len(network.get("cpt", {}))

    @staticmethod
    def _unique_list(values: Iterable[Any]) -> List[str]:
        result: List[str] = []
        seen: Set[str] = set()
        for value in values:
            item = str(value)
            if item not in seen:
                result.append(item)
                seen.add(item)
        return result

    @staticmethod
    def _normalize_edges(edges: Iterable[Sequence[Any]]) -> List[Edge]:
        result: List[Edge] = []
        seen: Set[Tuple[str, str]] = set()
        for edge in edges:
            parent, child = str(edge[0]), str(edge[1])
            key = (parent, child)
            if key in seen:
                continue
            result.append([parent, child])
            seen.add(key)
        return result

    # ------------------------------------------------------------------
    # CPT generation and mutation
    # ------------------------------------------------------------------
    def _add_node(self, network: NetworkDict, node: str) -> None:
        if node not in network["nodes"]:
            network["nodes"].append(node)
        network.setdefault("cpt", {}).setdefault(node, {"prior": self.default_probability})

    def _add_edge(self, network: NetworkDict, parent: str, child: str) -> bool:
        self._require_node(network, parent, context="edge_parent")
        self._require_node(network, child, context="edge_child")
        edge = [str(parent), str(child)]
        if edge not in network["edges"]:
            network["edges"].append(edge)
            return True
        return False

    def _remove_edge(self, network: NetworkDict, parent: str, child: str) -> bool:
        before = len(network["edges"])
        network["edges"] = [edge for edge in network["edges"] if not (edge[0] == parent and edge[1] == child)]
        return len(network["edges"]) != before

    def _add_edge_and_expand_cpt(self, network: NetworkDict, parent: str, child: str) -> None:
        added = self._add_edge(network, parent, child)
        if added:
            self._expand_cpt_for_new_parents(network, child, [parent])

    def _modify_cpt_for_new_parent(
        self,
        hybrid_network: NetworkDict,
        child: str,
        new_parents: Union[str, List[str]],
        base_bn: Optional[NetworkDict] = None,
        base_grid: Optional[NetworkDict] = None,
    ) -> None:
        """Backward-compatible wrapper around the production CPT expander."""
        del base_bn, base_grid  # kept in signature for older callers
        parents = [new_parents] if isinstance(new_parents, str) else list(new_parents)
        self._expand_cpt_for_new_parents(hybrid_network, child, parents)

    def _expand_cpt_for_new_parents(self, network: NetworkDict, child: str, new_parents: Sequence[str]) -> None:
        new_parent_list = self._unique_list(new_parents)
        if not new_parent_list:
            return
        self._require_node(network, child, context="cpt_child")
        for parent in new_parent_list:
            self._require_node(network, parent, context="cpt_parent")

        all_parents = self._get_parents(network, child)
        original_parents = [parent for parent in all_parents if parent not in new_parent_list]
        cpt_parent_order = original_parents + [parent for parent in new_parent_list if parent not in original_parents]
        original_cpt = copy.deepcopy(network.get("cpt", {}).get(child, {"prior": self.default_probability}))

        if len(cpt_parent_order) > self.max_explicit_cpt_parents:
            network["cpt"][child] = self._functional_cpt(
                cpt_type="parent_influence",
                parents=cpt_parent_order,
                child=child,
                metadata={"new_parents": new_parent_list, "base_probability": self._base_probability(original_cpt)},
            )
            network["metadata"].setdefault("functional_cpts", []).append(child)
            return

        new_cpt: Dict[str, Dict[str, float]] = {}
        for combo in itertools.product([True, False], repeat=len(cpt_parent_order)):
            original_combo = combo[: len(original_parents)]
            new_combo = combo[len(original_parents) :]
            base_prob = self._prob_true_from_cpt(original_cpt, original_combo)
            true_count = sum(1 for state in new_combo if bool(state))
            false_count = len(new_combo) - true_count
            prob_true = self._clamp_probability(base_prob + self.influence_step * true_count - self.influence_step * false_count)
            new_cpt[self._combo_key(combo)] = {"True": prob_true, "False": 1.0 - prob_true}
        network["cpt"][child] = new_cpt

    def _create_aggregator_cpt(self, parent_nodes: List[str], threshold: Optional[float] = None) -> Dict[str, Any]:
        """Create a threshold-aggregator CPT or compact functional CPT for large parent sets."""
        parents = [str(node) for node in parent_nodes]
        if not parents:
            return {"prior": self.aggregator_false_probability}
        safe_threshold = self.aggregator_threshold if threshold is None else clamp_confidence(threshold)
        if len(parents) > self.max_explicit_cpt_parents:
            logger.warning(f"Using functional aggregator CPT for {len(parents)} parents to avoid CPT explosion.")
            return self._functional_cpt(
                cpt_type="threshold_aggregator",
                parents=parents,
                threshold=safe_threshold,
                probabilities={
                    "above_threshold_true": self.aggregator_true_probability,
                    "below_threshold_true": self.aggregator_false_probability,
                },
            )

        new_cpt: Dict[str, Dict[str, float]] = {}
        for combo in itertools.product([True, False], repeat=len(parents)):
            proportion_true = sum(1 for state in combo if bool(state)) / max(1, len(combo))
            prob_true = self.aggregator_true_probability if proportion_true > safe_threshold else self.aggregator_false_probability
            prob_true = self._clamp_probability(prob_true)
            new_cpt[self._combo_key(combo)] = {"True": prob_true, "False": 1.0 - prob_true}
        return new_cpt

    def _create_or_gate_cpt(self, parent_nodes: List[str]) -> Dict[str, Any]:
        """Create a noisy-OR CPT or compact functional CPT for large parent sets."""
        parents = [str(node) for node in parent_nodes]
        if not parents:
            return {"prior": self.or_gate_false_probability}
        if len(parents) > self.max_explicit_cpt_parents:
            logger.warning(f"Using functional noisy-OR CPT for {len(parents)} parents to avoid CPT explosion.")
            return self._functional_cpt(
                cpt_type="noisy_or",
                parents=parents,
                probabilities={"any_true": self.or_gate_true_probability, "all_false": self.or_gate_false_probability},
            )

        new_cpt: Dict[str, Dict[str, float]] = {}
        for combo in itertools.product([True, False], repeat=len(parents)):
            prob_true = self.or_gate_true_probability if any(combo) else self.or_gate_false_probability
            prob_true = self._clamp_probability(prob_true)
            new_cpt[self._combo_key(combo)] = {"True": prob_true, "False": 1.0 - prob_true}
        return new_cpt

    def _functional_cpt(self, cpt_type: str, parents: Sequence[str], **kwargs: Any) -> Dict[str, Any]:
        if not self.functional_cpt_fallback:
            raise ProbabilisticModelError(
                "Explicit CPT would exceed max_explicit_cpt_parents and functional fallback is disabled",
                context={"cpt_type": cpt_type, "parents": list(parents), "limit": self.max_explicit_cpt_parents},
            )
        payload = {"type": cpt_type, "parents": list(parents), "metadata": {"functional": True}}
        payload.update(kwargs)
        return json_safe_reasoning_state(payload)

    def _prob_true_from_cpt(self, cpt: Any, parent_combo: Sequence[bool]) -> float:
        if not isinstance(cpt, Mapping):
            return self.default_probability

        if parent_combo:
            key_candidates = [
                self._combo_key(parent_combo),
                ",".join(str(bool(value)) for value in parent_combo),
                ",".join("1" if bool(value) else "0" for value in parent_combo),
            ]
            for key in key_candidates:
                entry = cpt.get(key)
                if isinstance(entry, Mapping):
                    return self._coerce_probability(entry.get("True", entry.get(True, entry.get("true"))), fallback=self.default_probability)
                if entry is not None:
                    return self._coerce_probability(entry, fallback=self.default_probability)

        if "prior" in cpt:
            return self._coerce_probability(cpt.get("prior"), fallback=self.default_probability)

        # Only treat top-level True/False as a probability row when True is not
        # itself a conditional row such as {"True": {"True": 0.8, ...}}.
        true_entry = cpt.get("True", cpt.get(True))
        if true_entry is not None and not isinstance(true_entry, Mapping):
            return self._coerce_probability(true_entry, fallback=self.default_probability)

        return self._base_probability(cpt)

    def _base_probability(self, cpt: Any) -> float:
        if not isinstance(cpt, Mapping):
            return self.default_probability
        if "prior" in cpt:
            return self._coerce_probability(cpt.get("prior"), fallback=self.default_probability)
        true_entry = cpt.get("True", cpt.get(True))
        if true_entry is not None and not isinstance(true_entry, Mapping):
            return self._coerce_probability(true_entry, fallback=self.default_probability)
        probabilities: List[float] = []
        for key, value in cpt.items():
            if str(key) in _RESERVED_CPT_KEYS:
                continue
            if isinstance(value, Mapping):
                probabilities.append(self._coerce_probability(value.get("True", value.get(True, value.get("true"))), fallback=self.default_probability))
            else:
                probabilities.append(self._coerce_probability(value, fallback=self.default_probability))
        return sum(probabilities) / len(probabilities) if probabilities else self.default_probability

    def _coerce_probability(self, value: Any, *, fallback: float) -> float:
        try:
            return self._clamp_probability(float(value))
        except (TypeError, ValueError):
            return self._clamp_probability(fallback)

    def _clamp_probability(self, value: float) -> float:
        if not math.isfinite(float(value)):
            raise ConfidenceBoundsError("Probability must be finite", context={"value": value})
        return min(self.max_probability, max(self.min_probability, float(value)))

    @staticmethod
    def _combo_key(combo: Sequence[bool]) -> str:
        return ",".join("True" if bool(value) else "False" for value in combo)

    # ------------------------------------------------------------------
    # Graph / grid helpers
    # ------------------------------------------------------------------
    def _get_parents(self, network: Mapping[str, Any], child_node: str) -> List[str]:
        """Return parent nodes for a child in edge order."""
        return [str(parent) for parent, child in network.get("edges", []) if str(child) == str(child_node)]

    def _get_children(self, network: Mapping[str, Any], parent_node: str) -> List[str]:
        return [str(child) for parent, child in network.get("edges", []) if str(parent) == str(parent_node)]

    def _get_grid_nodes_in_region(self, grid: Union[int, Mapping[str, Any]], region_coords: Tuple[int, int, int, int]) -> List[str]:
        """Return grid node names within a rectangular region.

        ``grid`` may be a grid dimension for backwards compatibility or a
        network dictionary containing ``nodes``.
        """
        region = self._normalize_region(region_coords)
        x_start, y_start, x_end, y_end = region
        if isinstance(grid, int):
            grid_dim = grid
            node_set = {f"N{row}{col}" for row in range(grid_dim) for col in range(grid_dim)}
        else:
            grid_dim = self._infer_grid_dimension(grid)
            node_set = set(map(str, grid.get("nodes", [])))

        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(grid_dim - 1, x_end)
        y_end = min(grid_dim - 1, y_end)
        nodes: List[str] = []
        for row in range(y_start, y_end + 1):
            for col in range(x_start, x_end + 1):
                candidates = [f"N{row}{col}", f"N{row}_{col}", f"Node_{row}_{col}"]
                found = next((candidate for candidate in candidates if candidate in node_set), None)
                if found is not None:
                    nodes.append(found)
                else:
                    # For renamed grid networks, fall back to row-major index.
                    ordinal = row * grid_dim + col
                    if not isinstance(grid, int):
                        grid_nodes = list(map(str, grid.get("nodes", [])))
                        if 0 <= ordinal < len(grid_nodes):
                            nodes.append(grid_nodes[ordinal])
        return self._unique_list(nodes)

    def _infer_grid_dimension(self, grid: Mapping[str, Any]) -> int:
        nodes = list(grid.get("nodes", []))
        if not nodes:
            raise ModelInitializationError("Grid network must have nodes")
        explicit = grid.get("metadata", {}).get("dimension") if isinstance(grid.get("metadata"), Mapping) else None
        if explicit:
            return bounded_iterations(explicit, minimum=1, maximum=10_000)
        root = int(round(math.sqrt(len(nodes))))
        if root * root == len(nodes):
            return root
        # Non-square grids are supported by treating width as ceil(sqrt(n)).
        return max(1, int(math.ceil(math.sqrt(len(nodes)))))

    @staticmethod
    def _normalize_region(region: Any) -> Region:
        if not isinstance(region, Sequence) or isinstance(region, (str, bytes)) or len(region) != 4:
            raise ReasoningValidationError("Region must be a four-item sequence", context={"region": region})
        x_start, y_start, x_end, y_end = (int(region[0]), int(region[1]), int(region[2]), int(region[3]))
        if x_start > x_end:
            x_start, x_end = x_end, x_start
        if y_start > y_end:
            y_start, y_end = y_end, y_start
        return (x_start, y_start, x_end, y_end)

    @staticmethod
    def _partition_grid_regions(grid_dim: int, count: int) -> List[Region]:
        count = max(1, int(count))
        if count == 1:
            return [(0, 0, grid_dim - 1, grid_dim - 1)]
        cols = int(math.ceil(math.sqrt(count)))
        rows = int(math.ceil(count / cols))
        width = int(math.ceil(grid_dim / cols))
        height = int(math.ceil(grid_dim / rows))
        regions: List[Region] = []
        for r in range(rows):
            for c in range(cols):
                if len(regions) >= count:
                    break
                x0 = c * width
                y0 = r * height
                x1 = min(grid_dim - 1, x0 + width - 1)
                y1 = min(grid_dim - 1, y0 + height - 1)
                regions.append((x0, y0, x1, y1))
        return regions

    def _require_node(self, network: Mapping[str, Any], node: str, *, context: str) -> None:
        if str(node) not in set(map(str, network.get("nodes", []))):
            raise ReasoningValidationError("Required node is missing from hybrid network", context={"node": node, "context": context})

    def _find_cycle(self, nodes: Sequence[str], edges: Sequence[Sequence[str]]) -> Optional[List[str]]:
        graph: Dict[str, List[str]] = defaultdict(list)
        for parent, child in edges:
            graph[str(parent)].append(str(child))
        visiting: Set[str] = set()
        visited: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            visiting.add(node)
            path.append(node)
            for nxt in graph.get(node, []):
                if nxt in visiting:
                    start = path.index(nxt) if nxt in path else 0
                    return path[start:] + [nxt]
                if nxt not in visited:
                    cycle = dfs(nxt)
                    if cycle:
                        return cycle
            visiting.discard(node)
            visited.add(node)
            path.pop()
            return None

        for node in nodes:
            if node not in visited:
                cycle = dfs(str(node))
                if cycle:
                    return cycle
        return None

    def _validate_cpt_shapes(self, network: Mapping[str, Any]) -> List[Dict[str, Any]]:
        warnings: List[Dict[str, Any]] = []
        for node, cpt in dict(network.get("cpt", {})).items():
            if not isinstance(cpt, Mapping):
                warnings.append({"message": "CPT entry is not a mapping", "node": node})
                continue
            if cpt.get("type"):
                continue
            parents = self._get_parents(network, str(node))
            for key, value in cpt.items():
                if str(key) in _RESERVED_CPT_KEYS:
                    continue
                if parents and isinstance(key, str) and key:
                    parts = [part.strip() for part in key.split(",")]
                    if len(parts) != len(parents):
                        warnings.append(
                            {
                                "message": "CPT key parent count does not match graph parents",
                                "node": node,
                                "key": key,
                                "parents": parents,
                            }
                        )
                if isinstance(value, Mapping):
                    true_prob = value.get("True", value.get(True, value.get("true")))
                    false_prob = value.get("False", value.get(False, value.get("false")))
                    if true_prob is not None and false_prob is not None:
                        total = float(true_prob) + float(false_prob)
                        if abs(total - 1.0) > 1e-6:
                            warnings.append({"message": "CPT row probabilities do not sum to 1", "node": node, "key": key, "sum": total})
        return warnings[:100]

    # ------------------------------------------------------------------
    # Diagnostics / persistence helpers
    # ------------------------------------------------------------------
    def explain_hybrid_network(self, network: Mapping[str, Any]) -> Dict[str, Any]:
        """Return JSON-safe explanation of the hybrid build."""
        nodes = list(network.get("nodes", []))
        edges = list(network.get("edges", []))
        metadata = dict(network.get("metadata", {}))
        return json_safe_reasoning_state(
            {
                "description": network.get("description"),
                "strategy": metadata.get("hybrid_strategy"),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "cpt_count": len(dict(network.get("cpt", {}))),
                "connection_summary": metadata.get("connection_summary", {}),
                "build_report": metadata.get("build_report", {}),
                "sample_edges": edges[: min(20, len(edges))],
            }
        )

    def save_hybrid_network(self, network: Mapping[str, Any], output_path: Union[str, Path]) -> str:
        """Persist a hybrid network JSON file with deterministic formatting."""
        path = Path(output_path).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(json_safe_reasoning_state(network), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            raise KnowledgePersistenceError("Failed to save hybrid network", cause=exc, context={"path": str(path)}) from exc
        return str(path)

    def clear_cache(self) -> None:
        self.hybrid_networks_cache.clear()
        self._source_network_cache.clear()

    def cache_metrics(self) -> Dict[str, Any]:
        return {
            "hybrid_cache_size": len(self.hybrid_networks_cache),
            "source_cache_size": len(self._source_network_cache),
            "cache_enabled": self.cache_enabled,
            "cache_max_size": self.cache_max_size,
        }

    def _build_report(self, network: Mapping[str, Any]) -> HybridBuildReport:
        diagnostics = network.get("metadata", {}).get("validation") or self.validate_hybrid_network(network)
        return HybridBuildReport(
            strategy=str(network.get("metadata", {}).get("hybrid_strategy")),
            node_count=len(list(network.get("nodes", []))),
            edge_count=len(list(network.get("edges", []))),
            cpt_count=len(dict(network.get("cpt", {}))),
            cycle_free=bool(diagnostics.get("cycle_free", True)),
            functional_cpt_count=int(diagnostics.get("functional_cpt_count", 0)),
            warnings=tuple(str(item) for item in diagnostics.get("warnings", [])[:10]),
        )

    def _record_memory_event(self, network: Mapping[str, Any]) -> None:
        if not self.record_memory_events or self.memory is None:
            return
        try:
            strategy = network.get("metadata", {}).get("hybrid_strategy", "unknown")
            self.memory.add(
                {
                    "type": "hybrid_network_build",
                    "strategy": strategy,
                    "node_count": len(list(network.get("nodes", []))),
                    "edge_count": len(list(network.get("edges", []))),
                    "created_at_ms": monotonic_timestamp_ms(),
                },
                priority=0.65,
                tag=["hybrid_models", str(strategy)],
            )
        except Exception as exc:
            logger.warning(f"Failed to record hybrid build in reasoning memory: {exc}")

    def _network_path(self, network_key: str) -> str:
        if network_key not in self.net_config:
            fallback = self.default_bn_key if network_key.startswith("bn") else self.default_grid_key
            if fallback in self.net_config:
                logger.warning(f"Network key '{network_key}' missing; using fallback '{fallback}'.")
                return str(self.net_config[fallback])
            raise ReasoningConfigurationError(
                "Network key missing from reasoning_config.yaml networks section",
                context={"network_key": network_key, "available": sorted(self.net_config.keys())[:50]},
            )
        return str(self.net_config[network_key])

    @staticmethod
    def _cache_key(base_bn_path: str, base_grid_path: str, strategy: str, params: Mapping[str, Any]) -> str:
        payload = {
            "base_bn_path": str(base_bn_path),
            "base_grid_path": str(base_grid_path),
            "strategy": strategy,
            "params": json_safe_reasoning_state(dict(params)),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    @staticmethod
    def _trim_cache(cache: MutableMapping[str, NetworkDict], max_size: int) -> None:
        while len(cache) > max_size:
            if isinstance(cache, OrderedDict):
                cache.popitem(last=False)
            else:
                cache.pop(next(iter(cache)))


if __name__ == "__main__":
    print("\n=== Running Hybrid Models ===\n")
    printer.status("TEST", "Hybrid Models initialized", "info")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        bn_path = tmp / "bayesian_network_2x2.json"
        grid_path = tmp / "grid_network_2x2.json"

        bn_path.write_text(
            json.dumps(
                {
                    "description": "Test causal BN",
                    "nodes": ["X", "Y"],
                    "edges": [["X", "Y"]],
                    "cpt": {
                        "X": {"prior": 0.6},
                        "Y": {
                            "True": {"True": 0.8, "False": 0.2},
                            "False": {"True": 0.3, "False": 0.7},
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        grid_path.write_text(
            json.dumps(
                {
                    "description": "Test 2x2 grid",
                    "nodes": ["N00", "N01", "N10", "N11"],
                    "edges": [["N00", "N01"], ["N00", "N10"], ["N01", "N11"], ["N10", "N11"]],
                    "cpt": {
                        "N00": {"prior": 0.5},
                        "N01": {"True": {"True": 0.7, "False": 0.3}, "False": {"True": 0.2, "False": 0.8}},
                        "N10": {"True": {"True": 0.7, "False": 0.3}, "False": {"True": 0.2, "False": 0.8}},
                        "N11": {
                            "True,True": {"True": 0.9, "False": 0.1},
                            "True,False": {"True": 0.7, "False": 0.3},
                            "False,True": {"True": 0.7, "False": 0.3},
                            "False,False": {"True": 0.1, "False": 0.9},
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

        builder = HybridProbabilisticModels()
        builder.hybrid_config.update(
            {
                "record_memory_events": False,
                "max_explicit_cpt_parents": 8,
                "cache_enabled": True,
                "validate_on_create": True,
            }
        )
        builder._refresh_runtime_config()

        global_hybrid = builder.create_hybrid_network(
            str(bn_path),
            str(grid_path),
            "global_to_local",
            {"global_node": "X"},
        )
        assert global_hybrid["metadata"]["hybrid_strategy"] == "global_to_local"
        assert ["X", "N00"] in global_hybrid["edges"]
        assert builder.validate_hybrid_network(global_hybrid)["valid"] is True

        local_hybrid = builder.create_hybrid_network(
            str(bn_path),
            str(grid_path),
            "local_to_global",
            {"aggregator_node_name": "Grid_Aggregator", "target_node": "Y"},
        )
        assert "Grid_Aggregator" in local_hybrid["nodes"]
        assert ["Grid_Aggregator", "Y"] in local_hybrid["edges"]
        assert builder.validate_hybrid_network(local_hybrid)["valid"] is True

        regional_hybrid = builder.create_hybrid_network(
            str(bn_path),
            str(grid_path),
            "regional",
            {"source_node": "X", "region_coords": (0, 0, 1, 0)},
        )
        assert ["X", "N00"] in regional_hybrid["edges"]
        assert ["X", "N01"] in regional_hybrid["edges"]

        explanation = builder.explain_hybrid_network(local_hybrid)
        assert explanation["strategy"] == "local_to_global"
        assert builder.cache_metrics()["hybrid_cache_size"] >= 3

    print("\n=== Test ran successfully ===\n")
