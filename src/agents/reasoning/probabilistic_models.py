"""Production probabilistic reasoning orchestration for the reasoning subsystem.

``ProbabilisticModels`` is the high-level probabilistic facade used by the
reasoning orchestrator.  It coordinates the exact Bayesian wrapper,
neural/adaptive circuit, low-level model compute layer, semantic frames,
knowledge-base confidences, learning updates, and multi-hop probabilistic graph
support without re-implementing work that belongs in those lower-level modules.

Configuration is intentionally kept in the existing loader flow:
``load_global_config()`` + ``get_config_section(...)``.  New knobs live under
``probabilistic_models`` in ``reasoning_config.yaml`` while legacy settings from
``inference``, ``storage``, and top-level config remain supported.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
import time
import numpy as np  # type: ignore
import torch.nn as nn  # type: ignore

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.reasoning_errors import *
from .utils.reasoning_helpers import *
from .modules.model_compute import ModelCompute
from .modules.adaptive_circuit import AdaptiveCircuit
from .modules.pgmpy_wrapper import PgmpyBayesianNetwork
from .reasoning_memory import ReasoningMemory
from .reasoning_cache import ReasoningCache
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Probabilistic Models")
printer = PrettyPrinter()

Fact = Tuple[str, str, str]
EvidenceLike = Optional[Mapping[Any, Any]]
NetworkDefinition = Dict[str, Any]

_TRUE_STRINGS = {"true", "t", "yes", "y", "1", "on", "positive", "present", "active"}
_FALSE_STRINGS = {"false", "f", "no", "n", "0", "off", "negative", "absent", "inactive"}


@dataclass(frozen=True)
class NetworkSelectionDecision:
    """Structured explanation of a network-selection decision."""

    network_key: str
    path: str
    family: str
    task_type: str
    complexity: str
    speed_requirement: str
    strategy: str
    confidence: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceTrace:
    """Compact telemetry for a probabilistic query."""

    query: Any
    evidence: Dict[str, Any]
    method: str
    result: float
    duration_seconds: float
    used_cache: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    supporting_facts: List[Tuple[Fact, float]] = field(default_factory=list)
    rule_contributions: Dict[str, float] = field(default_factory=dict)

    def to_memory_event(self) -> Dict[str, Any]:
        return json_safe_reasoning_state(
            {
                "type": "probabilistic_inference",
                "query": self.query,
                "evidence": self.evidence,
                "method": self.method,
                "result": self.result,
                "duration_seconds": self.duration_seconds,
                "used_cache": self.used_cache,
                "diagnostics": self.diagnostics,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )


@dataclass
class LearningCycleReport:
    """Result payload for Bayesian learning/update loops."""

    cycles_run: int
    observations_seen: int
    evidence_batches: int
    nodes_updated: int
    final_avg_delta: float
    converged: bool
    duration_seconds: float
    posterior_count: int

    def to_dict(self) -> Dict[str, Any]:
        return json_safe_reasoning_state(self.__dict__)


class ProbabilisticModels(nn.Module):
    """Production facade for Bayesian, neural, and KB-backed probabilities.

    Public methods from the previous module are intentionally retained:
    ``link_agent``, ``load_from_dict``, ``is_bayesian_task``,
    ``select_network``, ``bayesian_inference``,
    ``run_bayesian_learning_cycle``, ``probabilistic_query``,
    ``multi_hop_reasoning``, and the helper names used internally by older
    subsystem callers.
    """

    def __init__(
        self,
        network_data: Optional[NetworkDefinition] = None,
        semantic_frames: Optional[Mapping[str, Any]] = None,
        knowledge_base: Optional[Mapping[Any, Any]] = None,
        memory: Optional[ReasoningMemory] = None,
        *,
        initialize_components: bool = True,
    ) -> None:
        super().__init__()

        self.config: Dict[str, Any] = load_global_config()
        self.prob_config: Dict[str, Any] = get_config_section("probabilistic_models", default={})
        self.inference_config: Dict[str, Any] = get_config_section("inference")
        self.storage_config: Dict[str, Any] = get_config_section("storage")
        self.net_config: Dict[str, Any] = get_config_section("networks")

        self.semantic_frames_path: str = str(self.config.get("semantic_frames_path", ""))
        self.contradiction_threshold: float = self._cfg_confidence(
            "contradiction_threshold",
            self.config.get("contradiction_threshold", 0.25),
            source="top_level",
        )
        self.markov_logic_weight: float = self._cfg_confidence(
            "markov_logic_weight",
            self.config.get("markov_logic_weight", 0.7),
            source="top_level",
        )

        self.default_probability: float = self._cfg_confidence("default_probability", 0.5)
        self.epsilon: float = self._cfg_float("epsilon", 1e-9, minimum=1e-12, maximum=0.49)
        self.strict_resource_loading: bool = bool(self.prob_config.get("strict_resource_loading", False))
        self.strict_inference: bool = bool(self.prob_config.get("strict_inference", False))
        self.bayesian_first: bool = bool(self.prob_config.get("bayesian_first", True))
        self.neural_fallback: bool = bool(self.prob_config.get("neural_fallback", True))
        self.record_memory_events: bool = bool(self.prob_config.get("record_memory_events", True))
        self.enable_multi_hop: bool = bool(self.prob_config.get("enable_multi_hop", True))
        self.enable_semantic_similarity: bool = bool(self.prob_config.get("enable_semantic_similarity", True))
        self.update_knowledge_after_learning: bool = bool(self.prob_config.get("update_knowledge_after_learning", True))

        self.convergence_threshold: float = self._cfg_float(
            "convergence_threshold",
            self.inference_config.get("convergence_threshold", 1e-4),
            minimum=0.0,
            maximum=1.0,
        )
        self.max_learning_cycles: int = bounded_iterations(
            self.inference_config.get("max_learning_cycles", self.prob_config.get("max_learning_cycles", 100)),
            minimum=1,
            maximum=1_000_000,
        )
        self.learning_batch_size: int = bounded_iterations(
            self.prob_config.get("learning_batch_size", 100),
            minimum=1,
            maximum=1_000_000,
        )
        self.observation_buffer_size: int = bounded_iterations(
            self.prob_config.get("observation_buffer_size", 1000),
            minimum=1,
            maximum=5_000_000,
        )
        self.max_hop_depth: int = bounded_iterations(
            self.prob_config.get("max_hop_depth", 5),
            minimum=1,
            maximum=256,
        )
        self.hypothesis_depth_limit: int = bounded_iterations(
            self.prob_config.get("hypothesis_depth_limit", 3),
            minimum=1,
            maximum=128,
        )
        self.max_graph_neighbors: int = bounded_iterations(
            self.prob_config.get("max_graph_neighbors", 128),
            minimum=1,
            maximum=100_000,
        )
        self.posterior_cache_ttl_seconds: float = self._cfg_float(
            "posterior_cache_ttl_seconds", 300.0, minimum=0.0, maximum=86_400_000.0
        )
        self.posterior_cache_max_size: int = bounded_iterations(
            self.prob_config.get("posterior_cache_max_size", 2048),
            minimum=0,
            maximum=1_000_000,
        )
        self.similarity_cache_max_size: int = bounded_iterations(
            self.prob_config.get("similarity_cache_max_size", 4096),
            minimum=0,
            maximum=1_000_000,
        )
        self.selector_similarity_threshold: float = self._cfg_confidence("selector_similarity_threshold", 0.7)
        self.semantic_similarity_threshold: float = self._cfg_confidence("semantic_similarity_threshold", 0.7)
        self.similar_confidence_margin: float = self._cfg_confidence("similar_confidence_margin", 0.15)
        self.path_decay: float = self._cfg_confidence("path_decay", 0.95)
        self.evidence_confidence_default: float = self._cfg_confidence("evidence_confidence_default", 0.9)
        self.knowledge_conflict_threshold: float = self._cfg_confidence("knowledge_conflict_threshold", 0.3)
        self.knowledge_update_blend: float = self._cfg_confidence("knowledge_update_blend", 0.7)
        self.sensor_threshold: float = self._cfg_confidence("sensor_threshold", self.config.get("sensor_threshold", 0.7))
        self.temporal_halflife_days: float = self._cfg_float("temporal_halflife_days", 180.0, minimum=1.0, maximum=365_000.0)
        self.knowledge_halflife_days: float = self._cfg_float("knowledge_halflife_days", 90.0, minimum=1.0, maximum=365_000.0)

        self.query_weights: Dict[str, float] = self._normalize_weight_map(
            self.prob_config.get(
                "query_weights",
                {"base": 0.35, "evidence": 0.25, "semantic": 0.2, "multi_hop": 0.2},
            ),
            default={"base": 0.35, "evidence": 0.25, "semantic": 0.2, "multi_hop": 0.2},
        )
        self.similarity_weights: List[float] = self._normalize_similarity_weights(
            self.prob_config.get("similarity_weights", self.config.get("similarity_weights", [0.4, 0.4, 0.2]))
        )
        self.structural_weights: Dict[str, float] = self._load_structural_weights(
            self.inference_config.get("structural_weights", {})
        )
        self.sensor_node_mapping: Dict[str, str] = {
            str(k): str(v) for k, v in dict(self.config.get("sensor_node_mapping", {})).items()
        }

        self.bayesian_network_path = str(self.storage_config.get("bayesian_network", ""))
        self.knowledge_db_path = str(self.storage_config.get("knowledge_db", ""))
        for key, path in self.net_config.items():
            setattr(self, str(key), path)

        self.bn_selector_map, self.gn_selector_map = self._initialize_network_selectors()

        self.bayesian_network: NetworkDefinition = self._validate_network_definition(
            network_data if network_data is not None else self._load_bayesian_network(Path(self.bayesian_network_path)),
            source="provided" if network_data is not None else self.bayesian_network_path,
        )
        # --- Stable signature for cache invalidation ---
        self._definition_signature = self._compute_network_signature(self.bayesian_network)
        self.semantic_frames: Dict[str, Any] = self._normalize_semantic_frames(
            semantic_frames if semantic_frames is not None else self._load_semantic_frames(Path(self.semantic_frames_path))
        )
        self.knowledge_base: Dict[Fact, float] = self._normalize_knowledge_base(
            knowledge_base if knowledge_base is not None else self._load_knowledge_base(Path(self.knowledge_db_path))
        )
        self.knowledge_versions: Dict[Fact, List[Dict[str, Any]]] = defaultdict(list)

        self.reasoning_memory = memory or ReasoningMemory()
        self.pgmpy_bn: Optional[PgmpyBayesianNetwork] = None
        self.adaptive_circuit: Optional[AdaptiveCircuit] = None
        self.model_compute: Optional[ModelCompute] = None
        if initialize_components:
            self._initialize_components()

        self.agent: Any = None
        self.domain: Optional[str] = None
        self.posterior_cache: "OrderedDict[str, Tuple[float, float]]" = OrderedDict()
        self.similarity_cache: "OrderedDict[Tuple[str, str], float]" = OrderedDict()
        self.cache_enabled = self.prob_config.get("cache_enabled", True)
        if self.cache_enabled:
            try:
                self.cache = ReasoningCache(
                    namespace="probabilistic_models",
                    max_size=self.prob_config.get("cache_max_size", 2048),
                    default_ttl_seconds=self.prob_config.get("cache_ttl_seconds", 300.0),
                )
            except Exception as exc:
                logger.warning("Failed to initialize ReasoningCache for ProbabilisticModels: %s", exc)
                self.cache_enabled = False
                self.cache = None
        else:
            self.cache = None
        self.observation_buffer: Deque[Any] = deque(maxlen=self.observation_buffer_size)
        self.hypothesis_graph: Dict[Fact, List[Tuple[str, Fact]]] = defaultdict(list)
        self.last_selection: Optional[NetworkSelectionDecision] = None
        self.last_learning_report: Optional[LearningCycleReport] = None
        self.last_inference_trace: Optional[InferenceTrace] = None

        logger.info(
            "Probabilistic Models initialized | nodes=%s | edges=%s | kb_facts=%s | semantic_frames=%s",
            len(self.bayesian_network.get("nodes", [])),
            len(self.bayesian_network.get("edges", [])),
            len(self.knowledge_base),
            len(self.semantic_frames),
        )
        printer.status("INIT", "Probabilistic Models initialized", "success")

    def _compute_network_signature(self, network: NetworkDefinition) -> str:
        """Return a stable hash string for the Bayesian network structure."""
        try:
            payload = json.dumps(
                {
                    "nodes": sorted(network.get("nodes", [])),
                    "edges": sorted([sorted(e) for e in network.get("edges", [])]),
                },
                sort_keys=True,
                default=str
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        except Exception as exc:
            logger.warning("Could not compute network signature: %s", exc)
            return "fallback_signature"

    def _cache_key(self, prefix: str, *args: Any) -> str:
        """Generate a deterministic cache key for probabilistic results."""
        key_data = (prefix, self._definition_signature, args)
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _cache_get(self, key: str) -> Optional[Any]:
        if not self.cache_enabled or self.cache is None:
            return None
        return self.cache.get(key, default=None)

    def _cache_set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        if not self.cache_enabled or self.cache is None:
            return
        self.cache.set(key, value, ttl_seconds=ttl_seconds)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _cfg_float(self, key: str, default: Any, *, minimum: float, maximum: float) -> float:
        value = self.prob_config.get(key, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ReasoningConfigurationError(
                f"probabilistic_models.{key} must be numeric",
                cause=exc,
                context={"key": key, "value": value},
            ) from exc
        if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
            raise ReasoningConfigurationError(
                f"probabilistic_models.{key} outside allowed range",
                context={"key": key, "value": parsed, "minimum": minimum, "maximum": maximum},
            )
        return parsed

    def _cfg_confidence(self, key: str, default: Any, *, source: str = "probabilistic_models") -> float:
        value = default if source != "probabilistic_models" else self.prob_config.get(key, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ReasoningConfigurationError(
                f"{source}.{key} must be a confidence value",
                cause=exc,
                context={"key": key, "value": value, "source": source},
            ) from exc
        if not math.isfinite(parsed):
            raise ReasoningConfigurationError(
                f"{source}.{key} must be finite",
                context={"key": key, "value": value, "source": source},
            )
        return clamp_confidence(parsed)

    @staticmethod
    def _normalize_weight_map(raw: Any, *, default: Mapping[str, float]) -> Dict[str, float]:
        if not isinstance(raw, Mapping):
            raw = default
        weights = {str(k): max(float(v), 0.0) for k, v in dict(raw).items() if str(k).strip()}
        if not weights or sum(weights.values()) <= 0.0:
            weights = dict(default)
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    @staticmethod
    def _normalize_similarity_weights(raw: Any) -> List[float]:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raw = [0.4, 0.4, 0.2]
        values = [max(float(v), 0.0) for v in list(raw)[:3]]
        while len(values) < 3:
            values.append(0.0)
        total = sum(values)
        if total <= 0.0:
            return [0.4, 0.4, 0.2]
        return [v / total for v in values]

    def _load_structural_weights(self, raw: Any) -> Dict[str, float]:
        defaults = {
            "parent": 0.72,
            "child": 0.66,
            "causal": 0.74,
            "semantic": 0.62,
            "correlative": 0.58,
            "hierarchical": 0.67,
            "spouse": 0.55,
        }
        if isinstance(raw, Mapping):
            return {**defaults, **{str(k): clamp_confidence(v) for k, v in raw.items()}}
        if isinstance(raw, str) and raw.strip():
            path = Path(raw).expanduser()
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, Mapping):
                        return {**defaults, **{str(k): clamp_confidence(v) for k, v in data.items()}}
                except Exception as exc:
                    if self.strict_resource_loading:
                        raise ResourceLoadError(
                            "Failed to load structural weights",
                            cause=exc,
                            context={"path": str(path)},
                        ) from exc
                    logger.warning("Failed to load structural weights from %s: %s", path, exc)
        return defaults

    # ------------------------------------------------------------------
    # Initialization / lifecycle
    # ------------------------------------------------------------------
    def _initialize_components(self) -> None:
        """Initialize exact and neural probabilistic components."""
        try:
            self.pgmpy_bn = PgmpyBayesianNetwork(self.bayesian_network)
            self.adaptive_circuit = AdaptiveCircuit(
                network_structure=self.bayesian_network,
                knowledge_base=self.knowledge_base, # type: ignore
            )
            self.model_compute = ModelCompute(circuit=self.adaptive_circuit)
        except ReasoningError:
            raise
        except Exception as exc:
            raise ModelInitializationError(
                "Failed to initialize probabilistic model components",
                cause=exc,
                context={
                    "nodes": len(self.bayesian_network.get("nodes", [])),
                    "edges": len(self.bayesian_network.get("edges", [])),
                    "kb_facts": len(self.knowledge_base),
                },
            ) from exc

    def link_agent(self, agent: Any) -> None:
        """Connect to parent ReasoningAgent/Orchestrator."""
        self.agent = agent
        logger.info("Linked ProbabilisticModels to ReasoningAgent")

    def load_from_dict(self, network_data: Dict[str, Any]) -> None:
        """Replace the active Bayesian network and rebuild dependent components."""
        self.bayesian_network = self._validate_network_definition(network_data, source="load_from_dict")
        self.pgmpy_bn = PgmpyBayesianNetwork(self.bayesian_network)
        self.adaptive_circuit = AdaptiveCircuit(
            network_structure=self.bayesian_network,
            knowledge_base=self.knowledge_base, # type: ignore
        )
        if self.model_compute is None:
            self.model_compute = ModelCompute(circuit=self.adaptive_circuit)
        elif hasattr(self.model_compute, "set_circuit"):
            self.model_compute.set_circuit(self.adaptive_circuit)  # type: ignore[attr-defined]
        else:
            self.model_compute = ModelCompute(circuit=self.adaptive_circuit)
        self.clear_caches()

    # ------------------------------------------------------------------
    # Network selection
    # ------------------------------------------------------------------
    def _initialize_network_selectors(self) -> Tuple[Dict[Tuple[str, str], str], Dict[str, List[int]]]:
        """Initialize configurable task/complexity network selectors."""
        configured_bn = self.prob_config.get("bayesian_selector_map", {})
        bn_map: Dict[Tuple[str, str], str] = {}
        if isinstance(configured_bn, Mapping):
            for task, complexity_map in configured_bn.items():
                if isinstance(complexity_map, Mapping):
                    for complexity, network_key in complexity_map.items():
                        bn_map[(str(task).lower(), str(complexity).lower())] = str(network_key)

        if not bn_map:
            bn_map = {
                ("simple_check", "low"): "bn2x2",
                ("sequential_inference", "low"): "bn3x3",
                ("common_cause_analysis", "low"): "bn4x4",
                ("contextual_reasoning", "medium"): "bn5x5",
                ("multi_source_fusion", "medium"): "bn6x6",
                ("hierarchical_inference", "medium"): "bn7x7",
                ("dual_process_modeling", "high"): "bn8x8",
                ("modular_diagnostics", "high"): "bn9x9",
                ("hybrid_reasoning", "high"): "bn10x10",
                ("scalability_test", "very_high"): "bn20x20",
                ("large_context", "very_high"): "bn32x32",
                ("stress_test", "extreme"): "bn64x64",
            }

        configured_grid = self.prob_config.get("grid_complexity_map", {})
        gn_map: Dict[str, List[int]] = {}
        if isinstance(configured_grid, Mapping):
            for complexity, dims in configured_grid.items():
                if isinstance(dims, Sequence) and not isinstance(dims, (str, bytes)):
                    parsed = [int(x) for x in dims if int(x) > 0]
                    if parsed:
                        gn_map[str(complexity).lower()] = sorted(set(parsed))
        if not gn_map:
            gn_map = {
                "low": [2, 3, 4],
                "medium": [5, 6, 7, 8],
                "high": [9, 10, 20],
                "very_high": [32],
                "extreme": [64],
            }
        return bn_map, gn_map

    def is_bayesian_task(self, task_type: str) -> bool:
        """Determine whether a task should prefer Bayesian or grid/spatial networks."""
        task_lower = str(task_type or "").strip().lower()
        if not task_lower:
            return True
        grid_keywords = set(self.prob_config.get("grid_keywords", ["grid", "spatial", "image", "map", "layout", "pixel", "sensor", "topology", "navigation"]))
        bayesian_keywords = set(self.prob_config.get("bayesian_keywords", ["causal", "diagnostic", "logical", "inference", "reasoning", "probability", "belief"]))
        if any(keyword in task_lower for keyword in grid_keywords):
            return False
        if any(keyword in task_lower for keyword in bayesian_keywords):
            return True
        if "network" in task_lower or "graph" in task_lower:
            return "bayes" in task_lower or "pgm" in task_lower or "causal" in task_lower
        return not ("adjacency" in task_lower or "neighbor" in task_lower)

    def select_network(self, task_type: str, complexity: str, speed_requirement: str) -> str:
        """Return the selected network path while storing an explainable decision."""
        decision = self.select_network_decision(task_type, complexity, speed_requirement)
        return decision.path

    def select_network_decision(self, task_type: str, complexity: str, speed_requirement: str) -> NetworkSelectionDecision:
        task = str(task_type or "").strip().lower() or "simple_check"
        level = str(complexity or "medium").strip().lower()
        speed = str(speed_requirement or "balanced").strip().lower()
        try:
            if self.is_bayesian_task(task):
                decision = self._select_bayesian_network(task, level, speed)
            else:
                decision = self._select_grid_network(task, level, speed)
            self.last_selection = decision
            return decision
        except ReasoningError:
            raise
        except Exception as exc:
            fallback_key = str(self.prob_config.get("default_bn_key", "bn2x2"))
            fallback_path = str(getattr(self, fallback_key, self.bayesian_network_path))
            logger.error("Network selection failed for %s/%s/%s: %s", task, level, speed, exc)
            decision = NetworkSelectionDecision(
                network_key=fallback_key,
                path=fallback_path,
                family="bayesian",
                task_type=task,
                complexity=level,
                speed_requirement=speed,
                strategy="fallback_after_error",
                confidence=0.25,
                diagnostics={"error": f"{type(exc).__name__}: {exc}"},
            )
            self.last_selection = decision
            return decision

    def _select_bayesian_network(self, task_type: str, complexity: str, speed_requirement: str) -> NetworkSelectionDecision:
        lookup_key = (task_type, complexity)
        diagnostics: Dict[str, Any] = {}
        if lookup_key in self.bn_selector_map:
            network_key = self.bn_selector_map[lookup_key]
            strategy = "exact_task_complexity"
            confidence = 1.0
        else:
            best_match, similarity = self._closest_task_match(task_type, complexity)
            diagnostics["best_match"] = best_match
            diagnostics["similarity"] = similarity
            if best_match and similarity >= self.selector_similarity_threshold:
                network_key = self.bn_selector_map[(best_match, complexity)]
                strategy = "similar_task"
                confidence = similarity
            else:
                candidates = self._bayesian_candidates_for_complexity(complexity)
                network_key = self._select_by_speed(candidates, speed_requirement)
                strategy = "complexity_fallback"
                confidence = 0.55
                diagnostics["candidates"] = candidates
        if not hasattr(self, network_key):
            diagnostics["missing_network_key"] = network_key
            network_key = str(self.prob_config.get("default_bn_key", "bn2x2"))
        return NetworkSelectionDecision(
            network_key=network_key,
            path=str(getattr(self, network_key, self.bayesian_network_path)),
            family="bayesian",
            task_type=task_type,
            complexity=complexity,
            speed_requirement=speed_requirement,
            strategy=strategy,
            confidence=clamp_confidence(confidence),
            diagnostics=diagnostics,
        )

    def _select_grid_network(self, task_type: str, complexity: str, speed_requirement: str) -> NetworkSelectionDecision:
        dims = self.gn_selector_map.get(complexity)
        diagnostics: Dict[str, Any] = {"requested_complexity": complexity}
        if not dims:
            dims = self.gn_selector_map.get("high" if complexity in {"very_high", "extreme"} else "medium", [4, 5, 6])
            diagnostics["complexity_fallback_dims"] = dims
        if "high_res" in task_type or "detailed" in task_type:
            dim = max(dims)
            strategy = "task_high_resolution"
        elif "low_res" in task_type or "coarse" in task_type:
            dim = min(dims)
            strategy = "task_low_resolution"
        else:
            dim = self._select_by_speed(dims, speed_requirement)
            strategy = f"speed_{speed_requirement if speed_requirement in {'fast', 'accurate'} else 'balanced'}"
        network_key = f"gn{dim}x{dim}"
        if not hasattr(self, network_key):
            available_dims = self._available_grid_dimensions()
            closest_dim = min(available_dims or [2], key=lambda x: abs(x - dim))
            diagnostics["missing_grid_key"] = network_key
            diagnostics["closest_dim"] = closest_dim
            network_key = f"gn{closest_dim}x{closest_dim}"
        return NetworkSelectionDecision(
            network_key=network_key,
            path=str(getattr(self, network_key, self.bayesian_network_path)),
            family="grid",
            task_type=task_type,
            complexity=complexity,
            speed_requirement=speed_requirement,
            strategy=strategy,
            confidence=0.85 if hasattr(self, network_key) else 0.45,
            diagnostics=diagnostics,
        )

    def _closest_task_match(self, task_type: str, complexity: str) -> Tuple[Optional[str], float]:
        candidates = sorted({task for task, level in self.bn_selector_map if level == complexity})
        if not candidates:
            candidates = sorted({task for task, _ in self.bn_selector_map})
        if not candidates:
            return None, 0.0
        best = max(candidates, key=lambda item: SequenceMatcher(None, task_type, item).ratio())
        return best, SequenceMatcher(None, task_type, best).ratio()

    def _bayesian_candidates_for_complexity(self, complexity: str) -> List[str]:
        grouped: Dict[str, List[str]] = defaultdict(list)
        for (_, level), network_key in self.bn_selector_map.items():
            grouped[level].append(network_key)
        if complexity in grouped:
            return sorted(set(grouped[complexity]), key=self._network_size_sort_key)
        if complexity in {"very_high", "extreme"}:
            return sorted(set(grouped.get("high", ["bn8x8", "bn9x9", "bn10x10"])), key=self._network_size_sort_key)
        return sorted(set(grouped.get("medium", ["bn5x5", "bn6x6", "bn7x7"])), key=self._network_size_sort_key)

    @staticmethod
    def _select_by_speed(candidates: Sequence[Any], speed_requirement: str) -> Any:
        if not candidates:
            raise ReasoningValidationError("Network candidate list cannot be empty")
        ordered = list(candidates)
        if speed_requirement == "fast":
            return ordered[0]
        if speed_requirement == "accurate":
            return ordered[-1]
        return ordered[len(ordered) // 2]

    @staticmethod
    def _network_size_sort_key(network_key: str) -> Tuple[int, str]:
        match = re.search(r"(\d+)x\d+", str(network_key))
        return (int(match.group(1)) if match else 0, str(network_key))

    def _available_grid_dimensions(self) -> List[int]:
        dims: List[int] = []
        for key in self.net_config:
            match = re.fullmatch(r"gn(\d+)x\1", str(key))
            if match:
                dims.append(int(match.group(1)))
        return sorted(set(dims))

    # ------------------------------------------------------------------
    # Resource loading and normalization
    # ------------------------------------------------------------------
    def _load_bayesian_network(self, network_path: Path) -> NetworkDefinition:
        """Load Bayesian network structure from JSON with explicit recovery policy."""
        if not network_path.exists():
            message = f"Bayesian network file not found: {network_path}"
            if self.strict_resource_loading:
                raise ResourceLoadError(message, context={"path": str(network_path)})
            logger.warning("%s. Using safe fallback network.", message)
            return self._safe_default_network()
        try:
            return json.loads(network_path.read_text(encoding="utf-8"))
        except Exception as exc:
            if self.strict_resource_loading:
                raise ResourceLoadError(
                    "Error loading Bayesian network",
                    cause=exc,
                    context={"path": str(network_path)},
                ) from exc
            logger.error("Error loading Bayesian network from %s: %s. Using fallback.", network_path, exc)
            return self._safe_default_network()

    def _load_semantic_frames(self, path: Path) -> Dict[str, Any]:
        if not str(path):
            return {}
        if not path.exists():
            if self.strict_resource_loading:
                raise ResourceLoadError("Semantic frames file not found", context={"path": str(path)})
            logger.warning("Semantic frames file not found: %s", path)
            return {}
        try:
            logger.info("Loading semantic frames from %s", path)
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            if self.strict_resource_loading:
                raise ResourceLoadError("Failed to load semantic frames", cause=exc, context={"path": str(path)}) from exc
            logger.warning("Failed to load semantic frames from %s: %s", path, exc)
            return {}

    def _load_knowledge_base(self, kb_path: Path) -> Dict[Any, Any]:
        if not kb_path.exists():
            if self.strict_resource_loading:
                raise ResourceLoadError("Knowledge base file not found", context={"path": str(kb_path)})
            logger.warning("Knowledge base file not found: %s", kb_path)
            return {}
        logger.info("Loading knowledge base from %s", kb_path)
        try:
            return json.loads(kb_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            try:
                data = ast.literal_eval(kb_path.read_text(encoding="utf-8"))
                logger.warning("Recovered knowledge base using ast.literal_eval fallback")
                return data if isinstance(data, (dict, list)) else {} # type: ignore
            except Exception as fallback_exc:
                if self.strict_resource_loading:
                    raise ResourceLoadError(
                        "Invalid knowledge base JSON",
                        cause=fallback_exc,
                        context={"path": str(kb_path), "json_error": str(exc)},
                    ) from fallback_exc
                logger.error("Invalid knowledge base %s: %s", kb_path, exc)
                return {}

    def _validate_network_definition(self, network: Any, *, source: str) -> NetworkDefinition:
        if not isinstance(network, Mapping):
            raise ModelInitializationError(
                "Bayesian network must be a dictionary",
                context={"source": source, "type": type(network).__name__},
            )
        nodes = [str(node).strip() for node in network.get("nodes", []) if str(node).strip()]
        if not nodes:
            if self.strict_resource_loading:
                raise ModelInitializationError("Bayesian network contains no nodes", context={"source": source})
            return self._safe_default_network()
        if len(set(nodes)) != len(nodes):
            raise ModelInitializationError("Bayesian network contains duplicate nodes", context={"source": source, "nodes": nodes})
        node_set = set(nodes)
        edges: List[List[str]] = []
        for edge in network.get("edges", []) or []:
            if not isinstance(edge, Sequence) or isinstance(edge, (str, bytes)) or len(edge) != 2:
                raise CircuitConstraintError("Invalid Bayesian edge", context={"edge": edge, "source": source})
            parent, child = str(edge[0]).strip(), str(edge[1]).strip()
            if parent not in node_set or child not in node_set or parent == child:
                raise CircuitConstraintError("Bayesian edge references invalid node", context={"edge": edge, "source": source})
            if [parent, child] not in edges:
                edges.append([parent, child])
        cpt = network.get("cpt", {}) or {}
        if not isinstance(cpt, Mapping):
            raise ModelInitializationError("Bayesian network CPT must be a dictionary", context={"source": source})
        normalized = dict(network)
        normalized["nodes"] = nodes
        normalized["edges"] = edges
        normalized["cpt"] = dict(cpt)
        normalized.setdefault("description", f"Bayesian network loaded from {source}")
        return normalized

    @staticmethod
    def _safe_default_network() -> NetworkDefinition:
        return {
            "nodes": ["X", "Y"],
            "edges": [["X", "Y"]],
            "cpt": {
                "X": {"prior": 0.5},
                "Y": {
                    "True": {"True": 0.75, "False": 0.25},
                    "False": {"True": 0.25, "False": 0.75},
                },
            },
            "description": "Fallback 2-node Bayesian network",
            "metadata": {"generated_by": "ProbabilisticModels._safe_default_network"},
        }

    def _normalize_semantic_frames(self, frames: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not frames:
            return {}
        normalized: Dict[str, Any] = {}
        for name, frame in dict(frames).items():
            if not isinstance(frame, Mapping):
                continue
            normalized[str(name)] = {
                "description": str(frame.get("description", "")),
                "roles": list(frame.get("roles", []) or []),
                "verbs": [str(v).lower() for v in list(frame.get("verbs", []) or [])],
                "prepositions": list(frame.get("prepositions", []) or []),
            }
        return normalized

    def _normalize_knowledge_base(self, kb_data: Optional[Mapping[Any, Any]]) -> Dict[Fact, float]:
        if not kb_data:
            return {}
        if isinstance(kb_data, Mapping) and "knowledge" in kb_data:
            knowledge_items = kb_data.get("knowledge", [])
        elif isinstance(kb_data, Mapping):
            knowledge_items = kb_data.items()
        else:
            knowledge_items = kb_data

        processed: Dict[Fact, float] = {}
        for item in knowledge_items:
            try:
                fact, confidence = self._parse_knowledge_item(item)
                previous = processed.get(fact, 0.0)
                processed[fact] = merge_confidence(previous, confidence)
            except ReasoningError as exc:
                logger.warning("Skipping invalid knowledge item: %s", exc)
            except Exception as exc:
                logger.warning("Skipping invalid knowledge item %s: %s", item, exc)
        return processed

    def _parse_knowledge_item(self, item: Any) -> Tuple[Fact, float]:
        if isinstance(item, Mapping):
            s = item.get("subject")
            p = item.get("predicate")
            o = item.get("object")
            confidence = item.get("confidence", item.get("weight", 0.5))
            return normalize_fact((s, p, o)), clamp_confidence(confidence)
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) == 2:
                key, confidence = item
                if isinstance(key, str):
                    if "||" in key:
                        return normalize_fact(tuple(str(key).split("||", 2))), clamp_confidence(confidence)
                    return normalize_fact(key), clamp_confidence(confidence)
                if isinstance(key, Sequence) and not isinstance(key, (str, bytes)) and len(key) == 3:
                    return normalize_fact(key), clamp_confidence(confidence)
            if len(item) >= 3:
                confidence = item[3] if len(item) >= 4 else 0.5
                return normalize_fact(item[:3]), clamp_confidence(confidence)
        raise KnowledgeBaseError("Unsupported knowledge item format", context={"item": item})

    # ------------------------------------------------------------------
    # Inference APIs
    # ------------------------------------------------------------------
    def bayesian_inference(self, query: str, evidence: Optional[Dict[str, Any]] = None) -> float:
        """Exact Bayesian query with neural fallback when configured.

        Returns ``P(query=True | evidence)`` for binary networks.  Exact pgmpy
        inference is preferred; neural circuit inference is only used when exact
        inference is unavailable/fails and ``neural_fallback`` is enabled.
        """
        start = time.monotonic()
        query_node = str(query).strip()
        evidence_map = self._normalize_bn_evidence(evidence or {})
        nodes = set(self.bayesian_network.get("nodes", []))
        if query_node not in nodes:
            return self._handle_inference_failure(
                ModelInferenceError("Query node is not present in Bayesian network", context={"query": query_node}),
                query=query_node,
                evidence=evidence_map,
                start=start,
            )

        # ---- Cache check ----
        cache_key = self._cache_key("bayesian_inference", query_node, frozenset(evidence_map.items()))
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._record_inference_trace(InferenceTrace(
                query=query_node, evidence=evidence_map, method="cache_hit",
                result=cached, duration_seconds=0.0, used_cache=True
            ))
            return cached

        if self.bayesian_first and self.pgmpy_bn is not None:
            try:
                result = self._safe_probability(self.pgmpy_bn.query(query_node, evidence_map))
                self._record_inference_trace(
                    InferenceTrace(
                        query=query_node,
                        evidence=evidence_map,
                        method="pgmpy_exact",
                        result=result,
                        duration_seconds=elapsed_seconds(start),
                    )
                )
                return result
            except Exception as exc:
                if not self.neural_fallback:
                    return self._handle_inference_failure(exc, query=query_node, evidence=evidence_map, start=start)
                logger.warning("Exact Bayesian inference failed for %s; trying neural fallback: %s", query_node, exc)

        if self.neural_fallback and self.model_compute is not None:
            try:
                numeric_evidence = {key: 1.0 if self._coerce_bool_state(value) else 0.0 for key, value in evidence_map.items()}
                result = self._safe_probability(
                    self.model_compute.compute_marginal_probability((query_node,), numeric_evidence)
                )
                self._record_inference_trace(
                    InferenceTrace(
                        query=query_node,
                        evidence=evidence_map,
                        method="adaptive_circuit",
                        result=result,
                        duration_seconds=elapsed_seconds(start),
                    )
                )
                return result
            except Exception as exc:
                return self._handle_inference_failure(exc, query=query_node, evidence=evidence_map, start=start)

        result= self._handle_inference_failure(
            ModelInferenceError("No probabilistic inference backend is available"),
            query=query_node,
            evidence=evidence_map,
            start=start,
        )
        self._cache_set(cache_key, result, ttl_seconds=self.posterior_cache_ttl_seconds)
        return result

    def get_all_marginals(self, evidence: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
        """Return all Bayesian/network marginals using the best available backend."""
        evidence_map = self._normalize_bn_evidence(evidence or {})
        if self.pgmpy_bn is not None and hasattr(self.pgmpy_bn, "get_all_marginals"):
            try:
                return {str(k): self._safe_probability(v) for k, v in self.pgmpy_bn.get_all_marginals(evidence_map).items()}
            except Exception as exc:
                logger.warning("Exact all-marginals query failed: %s", exc)
        if self.model_compute is not None and hasattr(self.model_compute, "compute_all_marginals"):
            return {str(k): self._safe_probability(v) for k, v in self.model_compute.compute_all_marginals(evidence_map).items()}  # type: ignore[attr-defined]
        return {node: self.bayesian_inference(node, evidence_map) for node in self.bayesian_network.get("nodes", [])}

    def map_estimate(self, evidence: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Return a MAP assignment when supported by the active exact wrapper."""
        evidence_map = self._normalize_bn_evidence(evidence or {})
        if self.pgmpy_bn is not None and hasattr(self.pgmpy_bn, "map_query"):
            try:
                return dict(self.pgmpy_bn.map_query(evidence_map))
            except Exception as exc:
                logger.warning("MAP query failed: %s", exc)
        marginals = self.get_all_marginals(evidence_map)
        return {node: prob >= 0.5 for node, prob in marginals.items()}

    def sample_network(self, n_samples: int = 1000, *, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Sample the active Bayesian network through the wrapper if available."""
        count = bounded_iterations(n_samples, minimum=1, maximum=1_000_000)
        if self.pgmpy_bn is not None and hasattr(self.pgmpy_bn, "sample"):
            return list(self.pgmpy_bn.sample(n_samples=count, seed=seed))
        if self.model_compute is not None and hasattr(self.model_compute, "_sample_circuit"):
            return list(self.model_compute._sample_circuit(count))  # type: ignore[attr-defined]
        raise ModelInferenceError("No sampling backend available")

    def probabilistic_query(self, fact: Union[str, Sequence[Any]], evidence: EvidenceLike = None, context: Optional[List[Any]] = None) -> float:
        """Hybrid query over Bayesian variables and symbolic KB facts.

        If the query resolves to a Bayesian node, exact Bayesian inference is
        used.  Otherwise the method combines KB confidence, evidence support,
        semantic similarity, and optional multi-hop graph support using
        configured weights.
        """
        start = time.monotonic()
        evidence_map = evidence or {}
        bn_query = self._fact_to_bn_query(fact)
        if bn_query is not None:
            return self.bayesian_inference(bn_query, self._evidence_to_bn_map(evidence_map))

        query_fact = normalize_fact(fact)
        cache_key = self._cache_key("probabilistic_query", query_fact, frozenset(evidence_map.items()))
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._record_inference_trace(InferenceTrace(
                query=query_fact, evidence=evidence_map, method="hybrid_cache_hit", # type: ignore
                result=cached, duration_seconds=0.0, used_cache=True
            ))
            return cached

        query_fact = normalize_fact(fact)
        base_score = self._knowledge_confidence(query_fact)
        evidence_score = self._score_fact_evidence(query_fact, evidence_map)
        semantic_score = self._semantic_support(query_fact, context=context) if self.enable_semantic_similarity else self.default_probability
        multi_hop_score = self.multi_hop_reasoning(query_fact, context=context, max_depth=self.max_hop_depth) if self.enable_multi_hop else self.default_probability

        supporting = []
        if base_score > self.default_probability:
            supporting.append((query_fact, base_score))
        if evidence_score > self.default_probability:
            supporting.append((("evidence", "supports", str(query_fact)), evidence_score))
        if semantic_score > self.default_probability:
            supporting.append((("evidence", "supports", str(query_fact)), semantic_score))
        if multi_hop_score > self.default_probability:
            supporting.append((("evidence", "supports", str(query_fact)), multi_hop_score))

        rule_contrib = {
            "base": base_score,
            "evidence": evidence_score,
            "semantic": semantic_score,
            "multi_hop": multi_hop_score,
        }

        weighted_result = weighted_confidence(
            [base_score, evidence_score, semantic_score, multi_hop_score],
            [
                self.query_weights.get("base", 0.0),
                self.query_weights.get("evidence", 0.0),
                self.query_weights.get("semantic", 0.0),
                self.query_weights.get("multi_hop", 0.0),
            ],
        )
        result = self._safe_probability(weighted_result)
        self._cache_set(cache_key, result, ttl_seconds=self.posterior_cache_ttl_seconds)
        self._record_inference_trace(
            InferenceTrace(
                query=query_fact,
                evidence=json_safe_reasoning_state(dict(evidence_map)),
                method="hybrid_kb_semantic_multihop",
                result=result,
                duration_seconds=elapsed_seconds(start),
                diagnostics={
                    "base": base_score,
                    "evidence": evidence_score,
                    "semantic": semantic_score,
                    "multi_hop": multi_hop_score,
                    "weights": self.query_weights,
                },
                supporting_facts=supporting,
                rule_contributions=rule_contrib,
            )
        )
        return result

    def _handle_inference_failure(self, exc: BaseException, *, query: Any, evidence: Dict[str, Any], start: float) -> float:
        if self.strict_inference:
            if isinstance(exc, ReasoningError):
                raise exc
            raise ModelInferenceError(
                "Probabilistic inference failed",
                cause=exc,
                context={"query": query, "evidence": evidence},
            ) from exc
        logger.warning("Probabilistic inference fallback for %s: %s", query, exc)
        result = self.default_probability
        self._record_inference_trace(
            InferenceTrace(
                query=query,
                evidence=evidence,
                method="default_probability_fallback",
                result=result,
                duration_seconds=elapsed_seconds(start),
                diagnostics={"error": f"{type(exc).__name__}: {exc}"},
            )
        )
        return result

    def _record_inference_trace(self, trace: InferenceTrace) -> None:
        self.last_inference_trace = trace
        if self.record_memory_events and self.reasoning_memory is not None:
            try:
                self.reasoning_memory.add(trace.to_memory_event(), tag=["inference", trace.method], priority=trace.result)
            except TypeError:
                self.reasoning_memory.add(trace.to_memory_event(), tag="inference", priority=trace.result)
            except Exception as exc:
                logger.debug("Failed to record inference event in reasoning memory: %s", exc)

    # ------------------------------------------------------------------
    # Learning and posterior updates
    # ------------------------------------------------------------------
    def run_bayesian_learning_cycle(self, observations: Sequence[Any], context: Optional[List[Any]] = None) -> LearningCycleReport:
        """Refine posterior cache from observations and optionally update KB."""
        start = time.monotonic()
        if observations:
            self.observation_buffer.extend(observations)
        cycles_run = 0
        evidence_batches = 0
        nodes_updated = 0
        final_avg_delta = 0.0
        converged = False
        nodes = list(self.bayesian_network.get("nodes", []))

        for cycle in range(self.max_learning_cycles):
            cycles_run = cycle + 1
            if not self.observation_buffer:
                break
            batch_size = min(self.learning_batch_size, len(self.observation_buffer))
            batch = random.sample(list(self.observation_buffer), batch_size)
            delta_norm = 0.0
            updated_this_cycle = 0

            for obs in batch:
                evidence = self._extract_evidence_from_obs(obs)
                if not evidence:
                    continue
                evidence_batches += 1
                bn_evidence = {node: data["value"] for node, data in evidence.items() if isinstance(data, Mapping) and node in nodes}
                if not bn_evidence:
                    continue

                try:
                    marginals = self.get_all_marginals(bn_evidence)
                except Exception as exc:
                    logger.warning("Marginal extraction failed during learning cycle: %s", exc)
                    marginals = {node: self.bayesian_inference(node, bn_evidence) for node in nodes if node not in bn_evidence}

                revisions: Dict[str, float] = {}
                for node, posterior in marginals.items():
                    if node not in nodes:
                        continue
                    old = self._get_cached_posterior(node, default=self._node_prior(node))
                    observed_confidence = self._observation_confidence(evidence.get(node))
                    updated = self._bayesian_update(old, observed_confidence, posterior)
                    delta_norm += abs(updated - old)
                    updated_this_cycle += 1
                    nodes_updated += 1
                    revisions[node] = updated
                    self._set_cached_posterior(node, updated)

                if revisions and self.model_compute is not None:
                    try:
                        self.model_compute.dynamic_model_revision(revisions)
                    except Exception as exc:
                        logger.debug("ModelCompute revision skipped: %s", exc)

            if updated_this_cycle > 0:
                final_avg_delta = delta_norm / updated_this_cycle
            if cycle > 0 and final_avg_delta <= self.convergence_threshold:
                converged = True
                break

        if self.update_knowledge_after_learning and nodes_updated > 0:
            self._update_knowledge_base()

        report = LearningCycleReport(
            cycles_run=cycles_run,
            observations_seen=len(observations or []),
            evidence_batches=evidence_batches,
            nodes_updated=nodes_updated,
            final_avg_delta=final_avg_delta,
            converged=converged,
            duration_seconds=elapsed_seconds(start),
            posterior_count=len(self.posterior_cache),
        )
        self.last_learning_report = report
        if self.record_memory_events:
            self._add_memory_event({"type": "bayesian_learning_cycle_completed", **report.to_dict()}, tag="bayesian_learning", priority=1.0)
        return report

    def _bayesian_update(self, prior: float, likelihood: float, posterior: float) -> float:
        """Stable posterior blending/update used by the learning cycle."""
        prior = self._safe_probability(prior)
        likelihood = self._safe_probability(likelihood)
        posterior = self._safe_probability(posterior)
        denominator = likelihood * posterior + (1.0 - likelihood) * (1.0 - posterior)
        if denominator <= self.epsilon:
            return prior
        updated = (likelihood * posterior) / denominator
        blend = self.knowledge_update_blend
        return self._safe_probability((blend * updated) + ((1.0 - blend) * prior))

    def _calculate_likelihood(self, observed_value: Any) -> float:
        """Calculate likelihood from observation values."""
        if isinstance(observed_value, Mapping) and "value" in observed_value:
            return self._calculate_likelihood(observed_value.get("value"))
        if isinstance(observed_value, bool):
            return 0.95 if observed_value else 0.05
        if isinstance(observed_value, (int, float)) and math.isfinite(float(observed_value)):
            return self._safe_probability(1.0 - min(0.5, abs(float(observed_value) - 0.5)))
        if isinstance(observed_value, str):
            lowered = observed_value.strip().lower()
            if lowered in _TRUE_STRINGS:
                return 0.95
            if lowered in _FALSE_STRINGS:
                return 0.05
        return 0.75

    def _observation_confidence(self, evidence_entry: Any) -> float:
        if isinstance(evidence_entry, Mapping):
            return self._safe_probability(evidence_entry.get("confidence", self.evidence_confidence_default))
        return self.evidence_confidence_default

    def _get_cached_posterior(self, node: str, *, default: float) -> float:
        if self.cache_enabled and self.cache is not None:
            key = self._cache_key("posterior", node)
            val = self.cache.get(key)
            if val is not None:
                return self._safe_probability(val)
        # legacy fallback
        record = self.posterior_cache.get(node)
        if record is None:
            return self._safe_probability(default)
        value, stored_at = record
        if self.posterior_cache_ttl_seconds > 0 and (time.time() - stored_at) > self.posterior_cache_ttl_seconds:
            self.posterior_cache.pop(node, None)
            return self._safe_probability(default)
        self.posterior_cache.move_to_end(node)
        return self._safe_probability(value)

    def _set_cached_posterior(self, node: str, value: float) -> None:
        if self.cache_enabled and self.cache is not None:
            key = self._cache_key("posterior", node)
            self.cache.set(key, value, ttl_seconds=self.posterior_cache_ttl_seconds)
        # legacy
        if self.posterior_cache_max_size == 0:
            return
        self.posterior_cache[node] = (self._safe_probability(value), time.time())
        self.posterior_cache.move_to_end(node)
        while len(self.posterior_cache) > self.posterior_cache_max_size:
            self.posterior_cache.popitem(last=False)

    def _update_knowledge_base(self) -> None:
        """Persist learned posterior probabilities into the in-memory KB."""
        if not self.posterior_cache:
            return
        update_count = 0
        for node, (posterior, _) in list(self.posterior_cache.items()):
            fact = normalize_fact((node, "has_belief", "True"))
            current_value = self.knowledge_base.get(fact, self.default_probability)
            if abs(posterior - current_value) > self.knowledge_conflict_threshold:
                posterior = (posterior * self.knowledge_update_blend) + (current_value * (1.0 - self.knowledge_update_blend))
                logger.info("Resolved knowledge conflict for %s: %.3f -> %.3f", node, current_value, posterior)
            posterior = self._safe_probability(posterior)
            self.knowledge_base[fact] = posterior
            self.knowledge_versions[fact].append(
                {
                    "value": posterior,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "source": "bayesian_learning",
                }
            )
            update_count += 1

            # Optional integration with external ontology/knowledge graph, if the caller attached one.
            ontology = getattr(self, "ontology", None)
            if ontology is not None and hasattr(ontology, "get_related_nodes"):
                self._propagate_posterior_to_related_nodes(node, posterior, ontology)

        self._apply_temporal_decay_to_beliefs()
        if update_count > 0:
            logger.info("Updated %s knowledge facts from posterior cache", update_count)
        self.posterior_cache.clear()
        if self.record_memory_events:
            self._add_memory_event({"type": "knowledge_base_updated", "updated": update_count}, tag="knowledge_update", priority=1.0)

    def _propagate_posterior_to_related_nodes(self, node: str, posterior: float, ontology: Any) -> None:
        for rel_type in ["causal", "correlative", "hierarchical"]:
            try:
                related_nodes = ontology.get_related_nodes(node, rel_type)
            except Exception:
                related_nodes = []
            for related_node in related_nodes or []:
                rel_strength = 0.5
                if hasattr(ontology, "relationship_strength"):
                    try:
                        rel_strength = float(ontology.relationship_strength(node, related_node))
                    except Exception:
                        rel_strength = 0.5
                rel_fact = normalize_fact((related_node, "has_belief", "True"))
                current = self.knowledge_base.get(rel_fact, self.default_probability)
                updated = current + (posterior - 0.5) * clamp_confidence(rel_strength) * 0.3
                self.knowledge_base[rel_fact] = self._safe_probability(updated)

    def _apply_temporal_decay_to_beliefs(self) -> None:
        now = datetime.now()
        for fact, versions in list(self.knowledge_versions.items()):
            if not versions or fact not in self.knowledge_base:
                continue
            try:
                last_timestamp = versions[-1].get("timestamp")
                last_update = datetime.fromisoformat(str(last_timestamp))
                days_old = max(0, (now - last_update).days)
                decay_factor = math.exp(-days_old / self.knowledge_halflife_days)
                self.knowledge_base[fact] = self._safe_probability(self.knowledge_base[fact] * decay_factor)
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Multi-hop and semantic reasoning
    # ------------------------------------------------------------------
    def multi_hop_reasoning(self, query: Union[str, Sequence[Any]], context: Optional[List[Any]] = None, max_depth: int = 5) -> float:
        """Graph traversal over Bayesian structure and semantic KB associations."""
        try:
            query_fact = normalize_fact(query)
        except ReasoningError:
            return self.default_probability
        depth_limit = bounded_iterations(max_depth, minimum=1, maximum=max(1, self.max_hop_depth))
        self._build_hypothesis_graph(query_fact, max_depth=depth_limit)
        base_confidence = self._knowledge_confidence(query_fact)
        if query_fact not in self.hypothesis_graph:
            return base_confidence

        queue: Deque[Tuple[float, Fact, List[Fact]]] = deque([(base_confidence, query_fact, [query_fact])])
        best_confidence = base_confidence
        visited_paths: Set[Tuple[Fact, ...]] = set()

        while queue:
            current_conf, current_node, path = queue.popleft()
            path_signature = tuple(path)
            if path_signature in visited_paths:
                continue
            visited_paths.add(path_signature)
            path_penalty = self.path_decay ** max(0, len(path) - 1)
            best_confidence = max(best_confidence, self._safe_probability(current_conf * path_penalty))
            if len(path) > depth_limit:
                continue
            for hop_type, next_node in self.hypothesis_graph.get(current_node, [])[: self.max_graph_neighbors]:
                if next_node in path:
                    continue
                hop_conf = self._hop_confidence((hop_type, next_node), context=context)
                queue.append((self._safe_probability(current_conf * hop_conf), next_node, path + [next_node]))
        return self._safe_probability(best_confidence)

    def _build_hypothesis_graph(self, query: Fact, *, max_depth: Optional[int] = None) -> None:
        """Build graph traversal structure for multi-hop reasoning."""
        self.hypothesis_graph = defaultdict(list)
        depth_limit = max_depth if max_depth is not None else self.hypothesis_depth_limit
        queue: Deque[Tuple[Fact, int]] = deque([(query, 0)])
        visited: Set[Fact] = {query}
        while queue:
            current_fact, depth = queue.popleft()
            if depth >= depth_limit:
                continue
            self._add_network_connections(current_fact)
            self._add_semantic_associations(current_fact)
            self._prune_graph(current_fact)
            for _, next_node in self.hypothesis_graph[current_fact][: self.max_graph_neighbors]:
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, depth + 1))

    def _add_network_connections(self, fact: Fact) -> None:
        """Add Bayesian parent/child connections related to a fact subject."""
        root = fact[0]
        nodes = set(self.bayesian_network.get("nodes", []))
        if root not in nodes:
            return
        for parent, child in self.bayesian_network.get("edges", []):
            if child == root and parent in nodes:
                self.hypothesis_graph[fact].append(("parent", normalize_fact((parent, "is_parent_of", root))))
            if parent == root and child in nodes:
                self.hypothesis_graph[fact].append(("child", normalize_fact((child, "is_child_of", root))))

    def _add_semantic_associations(self, query: Fact) -> None:
        """Add semantically related KB facts to the hypothesis graph."""
        base_confidence = self._knowledge_confidence(query)
        candidates: List[Tuple[float, Fact]] = []
        for fact, confidence in self.knowledge_base.items():
            if fact == query:
                continue
            conf_value = self._safe_probability(confidence)
            similarity = self._semantic_similarity(query, fact)
            if similarity >= self.semantic_similarity_threshold or abs(conf_value - base_confidence) <= self.similar_confidence_margin:
                candidates.append((max(similarity, conf_value), fact))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, fact in candidates[: self.max_graph_neighbors]:
            self.hypothesis_graph[query].append(("semantic", fact))

    def _prune_graph(self, root: Fact) -> None:
        """Prune low-confidence hops from a single graph adjacency list."""
        self.hypothesis_graph[root] = [hop for hop in self.hypothesis_graph[root] if self._hop_confidence(hop) > 0.2]

    def _hop_confidence(self, hop: Tuple[str, Fact], context: Optional[List[Any]] = None) -> float:
        """Get confidence for a graph hop."""
        hop_type, hop_target = hop
        if hop_type == "semantic":
            base_conf = self.knowledge_base.get(hop_target, self.default_probability)
        else:
            base_conf = self.structural_weights.get(hop_type, self.default_probability)

        current_context = context or []
        if not current_context:
            try:
                current_context = self.reasoning_memory.get_current_context()
            except Exception:
                current_context = []
        if "high_uncertainty" in current_context:
            base_conf *= 0.8
        elif "crisis" in current_context:
            base_conf *= 1.2
        if hop_type == "causal":
            base_conf *= 1.1
        elif hop_type == "correlative":
            base_conf *= 0.9
        return self._safe_probability(base_conf)

    def _semantic_similarity(self, fact1: Any, fact2: Any) -> float:
        """Semantic similarity with frame-aware predicate comparison and lexical fallback."""
        s1, s2 = str(fact1), str(fact2)
        cache_key = self._cache_key("semantic_similarity", s1, s2)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if isinstance(fact1, tuple) and isinstance(fact2, tuple):
            components = [
                ("subject", fact1[0] if len(fact1) > 0 else "", fact2[0] if len(fact2) > 0 else ""),
                ("predicate", fact1[1] if len(fact1) > 1 else "", fact2[1] if len(fact2) > 1 else ""),
                ("object", fact1[2] if len(fact1) > 2 else "", fact2[2] if len(fact2) > 2 else ""),
            ]
            scores: List[float] = []
            for comp_type, val1, val2 in components:
                if comp_type == "predicate":
                    scores.append(self._predicate_similarity(str(val1), str(val2)))
                else:
                    scores.append(self._entity_similarity(str(val1), str(val2)))
            sim = sum(weight * score for weight, score in zip(self.similarity_weights, scores))
        else:
            sim = SequenceMatcher(None, s1, s2).ratio()

        sim = self._safe_probability(sim)
        self._cache_set(cache_key, sim, ttl_seconds=None)  # persist until eviction
        if len(self.similarity_cache) < self.similarity_cache_max_size:
            self.similarity_cache[cache_key] = sim # type: ignore
        return sim

    def _predicate_similarity(self, pred1: str, pred2: str) -> float:
        """Compute predicate similarity using semantic frames when available."""
        p1, p2 = str(pred1).strip().lower(), str(pred2).strip().lower()
        if not p1 or not p2:
            return 0.0
        if p1 == p2:
            return 1.0
        if not self.semantic_frames:
            return SequenceMatcher(None, p1, p2).ratio()
        frame1_keys = [k for k, v in self.semantic_frames.items() if p1 in v.get("verbs", [])]
        frame2_keys = [k for k, v in self.semantic_frames.items() if p2 in v.get("verbs", [])]
        if frame1_keys and frame2_keys:
            if frame1_keys[0] == frame2_keys[0]:
                return 1.0
            roles1 = set(self.semantic_frames[frame1_keys[0]].get("roles", []))
            roles2 = set(self.semantic_frames[frame2_keys[0]].get("roles", []))
            if roles1 or roles2:
                role_overlap = len(roles1 & roles2) / max(1, len(roles1 | roles2))
                return self._safe_probability(0.5 + 0.5 * role_overlap)
        return self._safe_probability(SequenceMatcher(None, p1, p2).ratio())

    def _entity_similarity(self, ent1: str, ent2: str) -> float:
        """Compute entity similarity via attached graph/ontology or lexical fallback."""
        e1, e2 = str(ent1), str(ent2)
        if not e1 or not e2:
            return 0.0
        if e1 == e2:
            return 1.0
        knowledge_graph = getattr(self, "knowledge_graph", None)
        ontology = getattr(self, "ontology", None)
        if knowledge_graph is not None and hasattr(knowledge_graph, "get_embedding"):
            try:
                emb1 = knowledge_graph.get_embedding(e1)
                emb2 = knowledge_graph.get_embedding(e2)
                if emb1 is not None and emb2 is not None:
                    denom = float(np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    if denom > self.epsilon:
                        return self._safe_probability((float(np.dot(emb1, emb2)) / denom + 1.0) / 2.0)
            except Exception:
                pass
        if ontology is not None and hasattr(ontology, "path_similarity"):
            try:
                sim = ontology.path_similarity(e1, e2)
                if sim is not None:
                    return self._safe_probability(sim)
            except Exception:
                pass
        sim = SequenceMatcher(None, e1.lower(), e2.lower()).ratio()
        if self.domain == "medical" and "symptom" in e1.lower() and "symptom" in e2.lower():
            sim *= 1.2
        return self._safe_probability(sim)

    def _cache_similarity(self, key: Tuple[str, str], value: float) -> None:
        if self.similarity_cache_max_size == 0:
            return
        self.similarity_cache[key] = value
        self.similarity_cache.move_to_end(key)
        while len(self.similarity_cache) > self.similarity_cache_max_size:
            self.similarity_cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Evidence and fact utilities
    # ------------------------------------------------------------------
    def _extract_evidence_from_obs(self, obs: Any) -> Dict[str, Dict[str, Any]]:
        """Extract structured Bayesian evidence from observations."""
        evidence: Dict[str, Any] = {}
        nodes = set(self.bayesian_network.get("nodes", []))
        if isinstance(obs, Mapping):
            for node in nodes:
                if node in obs:
                    evidence[node] = self._coerce_bool_state(obs[node])
            nested = obs.get("evidence") if isinstance(obs.get("evidence"), Mapping) else None
            if nested:
                for key, value in nested.items():
                    if str(key) in nodes:
                        evidence[str(key)] = self._coerce_bool_state(value)
        elif isinstance(obs, str):
            lowered = obs.lower()
            for frame in self.semantic_frames.values():
                if any(str(verb).lower() in lowered for verb in frame.get("verbs", [])):
                    for role in frame.get("roles", []):
                        node = str(role)
                        if node in nodes:
                            evidence[node] = True
        elif isinstance(obs, np.ndarray):
            evidence.update(self._extract_sensor_evidence(obs.tolist()))
        elif isinstance(obs, Sequence) and not isinstance(obs, (str, bytes)):
            if obs and all(isinstance(item, Mapping) for item in obs):
                for item in obs:
                    evidence.update({k: v["value"] for k, v in self._extract_evidence_from_obs(item).items()})
            else:
                evidence.update(self._extract_sensor_evidence(obs))

        return {
            node: {"value": self._coerce_bool_state(value), "confidence": self.evidence_confidence_default}
            for node, value in evidence.items()
            if node in nodes
        }

    def _extract_sensor_evidence(self, values: Any) -> Dict[str, bool]:
        if not self.sensor_node_mapping:
            return {}
        result: Dict[str, bool] = {}
        if isinstance(values, Mapping):
            iterator = values.items()
        else:
            iterator = enumerate(values or [])
        for sensor_id, raw_values in iterator:
            sensor_key = str(sensor_id)
            node = self.sensor_node_mapping.get(sensor_key)
            if not node:
                continue
            seq = list(raw_values) if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)) else [raw_values]
            numeric = [float(v) for v in seq if isinstance(v, (int, float)) and math.isfinite(float(v))]
            if not numeric:
                continue
            window_size = min(5, len(numeric))
            smoothed = float(np.mean(numeric[-window_size:]))
            result[node] = smoothed >= self.sensor_threshold
        return result

    def _normalize_bn_evidence(self, evidence: Mapping[Any, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        nodes = set(self.bayesian_network.get("nodes", []))
        for raw_key, raw_value in dict(evidence or {}).items():
            value = raw_value.get("value") if isinstance(raw_value, Mapping) and "value" in raw_value else raw_value
            key: Optional[str] = None
            if isinstance(raw_key, str):
                key = raw_key.strip()
            elif isinstance(raw_key, Sequence) and not isinstance(raw_key, (str, bytes)):
                if len(raw_key) == 1:
                    key = str(raw_key[0]).strip()
                elif len(raw_key) >= 3 and str(raw_key[0]).strip() in nodes:
                    key = str(raw_key[0]).strip()
            if key and key in nodes:
                result[key] = self._coerce_bool_state(value)
        return result

    def _evidence_to_bn_map(self, evidence: Mapping[Any, Any]) -> Dict[str, Any]:
        return self._normalize_bn_evidence(evidence)

    def _fact_to_bn_query(self, fact: Union[str, Sequence[Any]]) -> Optional[str]:
        nodes = set(self.bayesian_network.get("nodes", []))
        if isinstance(fact, str) and fact.strip() in nodes:
            return fact.strip()
        if isinstance(fact, Sequence) and not isinstance(fact, (str, bytes)):
            if len(fact) == 1 and str(fact[0]).strip() in nodes:
                return str(fact[0]).strip()
            if len(fact) >= 3:
                subj, pred = str(fact[0]).strip(), str(fact[1]).strip().lower()
                if subj in nodes and pred in {"has_belief", "is_active", "state", "is_true", "probability"}:
                    return subj
        return None

    @staticmethod
    def _coerce_bool_state(value: Any) -> bool:
        if isinstance(value, Mapping) and "value" in value:
            return ProbabilisticModels._coerce_bool_state(value.get("value"))
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value) >= 0.5
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in _TRUE_STRINGS:
                return True
            if lowered in _FALSE_STRINGS:
                return False
        return bool(value)

    def _knowledge_confidence(self, fact: Fact) -> float:
        return self._safe_probability(self.knowledge_base.get(normalize_fact(fact), self.default_probability))

    def _score_fact_evidence(self, query_fact: Fact, evidence: Mapping[Any, Any]) -> float:
        if not evidence:
            return self.default_probability
        scores: List[float] = []
        weights: List[float] = []
        for raw_fact, raw_value in evidence.items():
            try:
                ev_fact = normalize_fact(raw_fact)
            except ReasoningError:
                continue
            evidence_truth = raw_value.get("value") if isinstance(raw_value, Mapping) and "value" in raw_value else raw_value
            evidence_conf = raw_value.get("confidence", self.evidence_confidence_default) if isinstance(raw_value, Mapping) else self.evidence_confidence_default
            kb_conf = self._knowledge_confidence(ev_fact)
            sim = self._semantic_similarity(query_fact, ev_fact) if self.enable_semantic_similarity else 0.0
            signed = kb_conf if bool(evidence_truth) else 1.0 - kb_conf
            scores.append(self._safe_probability((signed + sim) / 2.0))
            weights.append(self._safe_probability(evidence_conf))
        if not scores:
            return self.default_probability
        return self._safe_probability(weighted_confidence(scores, weights))

    def _semantic_support(self, query_fact: Fact, context: Optional[List[Any]] = None) -> float:
        if not self.knowledge_base:
            return self.default_probability
        scored: List[Tuple[float, float]] = []
        for fact, confidence in self.knowledge_base.items():
            sim = self._semantic_similarity(query_fact, fact)
            if sim >= self.semantic_similarity_threshold:
                scored.append((self._safe_probability(confidence), sim))
        if not scored:
            return self.default_probability
        values = [value for value, _ in scored]
        weights = [weight for _, weight in scored]
        support = weighted_confidence(values, weights)
        if context and "high_uncertainty" in context:
            support *= 0.9
        return self._safe_probability(support)

    def _node_prior(self, node: str) -> float:
        cpt = self.bayesian_network.get("cpt", {}).get(node, {})
        if isinstance(cpt, Mapping) and "prior" in cpt:
            return self._safe_probability(cpt.get("prior"))
        return self.default_probability

    def _safe_probability(self, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return self.default_probability
        if not math.isfinite(parsed):
            return self.default_probability
        return min(1.0 - self.epsilon, max(self.epsilon, clamp_confidence(parsed)))

    def _add_memory_event(self, event: Mapping[str, Any], *, tag: Union[str, List[str]], priority: float) -> None:
        try:
            self.reasoning_memory.add(json_safe_reasoning_state(dict(event)), tag=tag, priority=priority)
        except TypeError:
            self.reasoning_memory.add(json_safe_reasoning_state(dict(event)), tag=str(tag), priority=priority)
        except Exception as exc:
            logger.debug("Reasoning memory event skipped: %s", exc)

    # ------------------------------------------------------------------
    # Diagnostics and maintenance
    # ------------------------------------------------------------------
    def clear_caches(self) -> None:
        self.posterior_cache.clear()
        self.similarity_cache.clear()
        if self.pgmpy_bn is not None and hasattr(self.pgmpy_bn, "clear_cache"):
            self.pgmpy_bn.clear_cache()

    def cache_metrics(self) -> Dict[str, Any]:
        pgmpy_summary = getattr(self.pgmpy_bn, "summary", {}) if self.pgmpy_bn is not None else {}
        return {
            "posterior_cache_size": len(self.posterior_cache),
            "posterior_cache_max_size": self.posterior_cache_max_size,
            "similarity_cache_size": len(self.similarity_cache),
            "similarity_cache_max_size": self.similarity_cache_max_size,
            "pgmpy_summary": pgmpy_summary,
        }

    def diagnostics(self) -> Dict[str, Any]:
        nodes = list(self.bayesian_network.get("nodes", []))
        edges = list(self.bayesian_network.get("edges", []))
        return json_safe_reasoning_state(
            {
                "nodes": len(nodes),
                "edges": len(edges),
                "knowledge_facts": len(self.knowledge_base),
                "semantic_frames": len(self.semantic_frames),
                "observation_buffer": len(self.observation_buffer),
                "components": {
                    "pgmpy_bn": self.pgmpy_bn is not None,
                    "adaptive_circuit": self.adaptive_circuit is not None,
                    "model_compute": self.model_compute is not None,
                    "reasoning_memory": self.reasoning_memory is not None,
                },
                "cache_metrics": self.cache_metrics(),
                "last_selection": self.last_selection.__dict__ if self.last_selection else None,
                "last_learning_report": self.last_learning_report.to_dict() if self.last_learning_report else None,
                "last_inference_trace": self.last_inference_trace.to_memory_event() if self.last_inference_trace else None,
            }
        )

    def health_check(self) -> Dict[str, Any]:
        issues: List[str] = []
        if not self.bayesian_network.get("nodes"):
            issues.append("empty_bayesian_network")
        if self.pgmpy_bn is None:
            issues.append("pgmpy_backend_unavailable")
        if self.model_compute is None:
            issues.append("model_compute_unavailable")
        if not self.knowledge_base:
            issues.append("empty_knowledge_base")
        return {"healthy": not issues, "issues": issues, "diagnostics": self.diagnostics()}


if __name__ == "__main__":
    print("\n=== Running Probabilistic Models ===\n")
    printer.status("TEST", "Probabilistic Models initialized", "info")

    test_network = {
        "nodes": ["A", "B", "C"],
        "edges": [["A", "B"], ["B", "C"]],
        "cpt": {
            "A": {"prior": 0.6},
            "B": {
                "True": {"True": 0.85, "False": 0.15},
                "False": {"True": 0.2, "False": 0.8},
            },
            "C": {
                "True": {"True": 0.8, "False": 0.2},
                "False": {"True": 0.1, "False": 0.9},
            },
        },
        "description": "Compact probabilistic_models smoke-test network",
    }
    test_frames = {
        "CauseEffect": {
            "description": "Causal relationship",
            "roles": ["A", "B", "C"],
            "verbs": ["cause", "trigger"],
            "prepositions": ["because"],
        }
    }
    test_kb = {
        "knowledge": [
            {"subject": "A", "predicate": "is", "object": "Active", "confidence": 0.8},
            {"subject": "B", "predicate": "is", "object": "Active", "confidence": 0.7},
            {"subject": "C", "predicate": "has_belief", "object": "True", "confidence": 0.65},
        ]
    }

    models = ProbabilisticModels(network_data=test_network, semantic_frames=test_frames, knowledge_base=test_kb)
    selected = models.select_network("simple_check", "low", "fast")
    print(f"Selected network: {selected}")

    p_c = models.bayesian_inference("C", {"A": True})
    print(f"P(C | A=True) = {p_c:.4f}")
    assert 0.0 <= p_c <= 1.0

    fact_prob = models.probabilistic_query(("C", "has_belief", "True"), {("A", "is", "Active"): True})
    print(f"Hybrid fact probability = {fact_prob:.4f}")
    assert 0.0 <= fact_prob <= 1.0

    hop_prob = models.multi_hop_reasoning(("A", "is", "Active"), max_depth=3)
    print(f"Multi-hop probability = {hop_prob:.4f}")
    assert 0.0 <= hop_prob <= 1.0

    report = models.run_bayesian_learning_cycle([{"A": True, "B": True}, {"A": False, "B": False}])
    print(f"Learning report: {report.to_dict()}")
    assert report.cycles_run >= 1

    health = models.health_check()
    print(f"Health: {health['healthy']} | Issues: {health['issues']}")

    print("\n=== Test ran successfully ===\n")
