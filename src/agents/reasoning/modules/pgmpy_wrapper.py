"""
Production Bayesian Network wrapper for the reasoning subsystem.

This module keeps pgmpy integration isolated behind a stable SLAI-facing API:
- robust network-definition validation
- binary and named-state CPT normalization
- exact inference with bounded cache support
- MAP queries, all-marginals, sampling, structural diagnostics
- JSON-safe serialization and atomic persistence

The wrapper intentionally keeps configuration loading unchanged:
``load_global_config()`` + ``get_config_section("pgmpy_wrapper")``.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import itertools
import json
import math
import os
import tempfile
import time
import numpy as np  # pyright: ignore[reportMissingImports]

from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union
from pgmpy.factors.discrete import TabularCPD # type: ignore
from pgmpy.inference import VariableElimination # type: ignore
from pgmpy.models import DiscreteBayesianNetwork # type: ignore
from pgmpy.sampling import BayesianModelSampling, GibbsSampling # type: ignore

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.reasoning_errors import *
from ..utils.reasoning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Pgmpy Wrapper")
printer = PrettyPrinter()

StateValue = Union[str, int, bool, float]
EvidenceMap = Dict[str, StateValue]
Distribution = Dict[str, float]
NetworkDefinition = Dict[str, Any]

_TRUE_ALIASES = {"true", "t", "yes", "y", "1", "on", "positive", "present"}
_FALSE_ALIASES = {"false", "f", "no", "n", "0", "off", "negative", "absent"}
_DEFAULT_BINARY_STATES: List[Any] = [False, True]


class PgmpyBayesianNetwork:
    """
    Production-ready Bayesian Network wrapper using pgmpy.

    Public compatibility surface retained:
    - ``query(query_variable, evidence=None, use_cache=True) -> float``
    - ``map_query(evidence=None) -> Dict[str, bool]`` for binary networks
    - ``get_all_marginals(evidence=None) -> Dict[str, float]``
    - ``conditional_probability_table(target, evidence) -> float``
    - ``sample(n_samples=1000, seed=None) -> List[Dict[str, bool]]``
    - structural helpers, JSON export/import, and ``summary`` property

    Network definition format:
    ```python
    {
        "nodes": ["A", "B"],
        "edges": [["A", "B"]],
        "states": {"A": [False, True], "B": [False, True]},  # optional
        "cpt": {
            "A": {"prior": 0.6},
            "B": {
                "False": 0.2,
                "True": 0.8,
            }
        }
    }
    ```
    """

    def __init__(self, network_definition: NetworkDefinition):
        """Initialise the Bayesian Network from a JSON-like dictionary."""
        self.config = load_global_config()
        self.wrapper_config = get_config_section("pgmpy_wrapper") or {}

        self.default_probability = clamp_confidence(
            self.wrapper_config.get("default_probability", 0.5)
        )
        self.epsilon = float(self.wrapper_config.get("epsilon", 1e-9))
        if self.epsilon <= 0:
            raise ReasoningConfigurationError(
                "pgmpy_wrapper.epsilon must be positive",
                context={"epsilon": self.epsilon},
            )

        self.normalize_cpds = bool(self.wrapper_config.get("normalize_cpds", True))
        self.strict_cpt = bool(self.wrapper_config.get("strict_cpt", False))
        self.validate_model_on_build = bool(self.wrapper_config.get("validate_model_on_build", True))
        self.validate_dag = bool(self.wrapper_config.get("validate_dag", True))
        self.cache_enabled = bool(self.wrapper_config.get("cache_enabled", True))
        self.cache_ttl_seconds = float(self.wrapper_config.get("cache_ttl_seconds", 300.0))
        self.max_cache_entries = bounded_iterations(
            self.wrapper_config.get("max_cache_entries", 2048),
            minimum=0,
            maximum=250_000,
        )
        self.ignore_unknown_evidence = bool(self.wrapper_config.get("ignore_unknown_evidence", False))
        self.missing_cpt_policy = str(
            self.wrapper_config.get("missing_cpt_policy", "uniform")
        ).strip().lower()
        self.sampling_backend = str(
            self.wrapper_config.get("sampling_backend", "bayesian")
        ).strip().lower()
        self.show_progress = bool(self.wrapper_config.get("show_progress", False))
        self.max_sample_size = bounded_iterations(
            self.wrapper_config.get("max_sample_size", 100_000),
            minimum=1,
            maximum=10_000_000,
        )
        self.default_seed = self.wrapper_config.get("seed")
        self.export_indent = bounded_iterations(
            self.wrapper_config.get("export_indent", 2),
            minimum=0,
            maximum=8,
        )
        self.max_diagnostics_cpds = bounded_iterations(
            self.wrapper_config.get("max_diagnostics_cpds", 32),
            minimum=1,
            maximum=10_000,
        )

        self.network_def: NetworkDefinition = self._copy_network_definition(network_definition)
        self.model: Optional[DiscreteBayesianNetwork] = None
        self.inference_engine: Optional[VariableElimination] = None
        self.sampler: Optional[Union[BayesianModelSampling, GibbsSampling]] = None

        self._inference_cache: "OrderedDict[Tuple[Any, ...], Tuple[float, float]]" = OrderedDict()
        self._query_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._built_at_ms = monotonic_timestamp_ms()
        self._state_names: Dict[str, List[Any]] = {}
        self._state_lookup: Dict[str, Dict[str, int]] = {}
        self._parent_map: Dict[str, List[str]] = {}
        self._topological_order: List[str] = []
        self._definition_signature = self._stable_definition_signature(self.network_def)

        self._build_model()
        logger.info(
            "PgmpyBayesianNetwork initialized successfully | nodes=%s | edges=%s | signature=%s",
            len(list(self._ensure_model().nodes)),
            len(list(self._ensure_model().edges)),
            self._definition_signature,
        )

    # ----------------------------------------------------------------------
    # Core construction and validation
    # ----------------------------------------------------------------------
    @staticmethod
    def _copy_network_definition(network_definition: NetworkDefinition) -> NetworkDefinition:
        if not isinstance(network_definition, dict):
            raise ModelInitializationError(
                "Network definition must be a dictionary",
                context={"type": type(network_definition).__name__},
            )
        try:
            return copy.deepcopy(network_definition)
        except Exception as exc:
            raise ModelInitializationError(
                "Failed to copy Bayesian network definition",
                cause=exc,
            ) from exc

    @staticmethod
    def _stable_definition_signature(network_definition: NetworkDefinition) -> str:
        try:
            payload = json.dumps(network_definition, sort_keys=True, default=str)
        except TypeError:
            payload = repr(network_definition)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _build_model(self) -> None:
        """Construct the pgmpy model from the definition."""
        nodes = self._normalize_nodes(self.network_def.get("nodes", []))
        edges = self._normalize_edges(self.network_def.get("edges", []), nodes)
        cpt_data = self.network_def.get("cpt", {}) or {}

        if not isinstance(cpt_data, Mapping):
            raise ModelInitializationError(
                "Network definition 'cpt' must be a dictionary",
                context={"type": type(cpt_data).__name__},
            )

        self._initialize_state_metadata(nodes)

        self.model = DiscreteBayesianNetwork()
        self.model.add_nodes_from(nodes) # type: ignore
        self.model.add_edges_from(edges) # type: ignore

        self._parent_map = self._build_parent_map()
        self._validate_network_structure(self._parent_map, cpt_data)
        self._topological_order = self._get_topological_order(self._parent_map)

        cpds: List[TabularCPD] = []
        missing_cpt_nodes: List[str] = []
        for node in self._topological_order:
            cpt_info = cpt_data.get(node)
            if cpt_info is None:
                missing_cpt_nodes.append(node)
                cpds.append(self._build_missing_cpd(node, self._parent_map.get(node, [])))
                continue

            try:
                parents = self._parent_map.get(node, [])
                if parents:
                    cpd = self._build_child_cpd(node, parents, cpt_info)
                else:
                    cpd = self._build_root_cpd(node, cpt_info)
                cpds.append(cpd)
            except ReasoningError:
                raise
            except Exception as exc:
                logger.error("Error building CPD for node %s: %s", node, exc)
                raise ModelInitializationError(
                    f"CPD construction failed for node '{node}'",
                    cause=exc,
                    context={"node": node, "parents": self._parent_map.get(node, [])},
                ) from exc

        if missing_cpt_nodes:
            logger.warning(
                "Missing CPT data for nodes %s. Applied policy '%s'.",
                missing_cpt_nodes,
                self.missing_cpt_policy,
            )

        model = self._ensure_model()
        model.add_cpds(*cpds)
        if self.validate_model_on_build and not self._validate_model():
            raise ModelInitializationError(
                "pgmpy model validation failed",
                context={"nodes": nodes, "edges": edges},
            )

        self.inference_engine = VariableElimination(model)
        self.sampler = self._build_sampler(model)

    def _build_sampler(self, model: DiscreteBayesianNetwork) -> Union[BayesianModelSampling, GibbsSampling]:
        if self.sampling_backend == "gibbs":
            return GibbsSampling(model)
        # default: BayesianModelSampling (which uses forward_sample)
        return BayesianModelSampling(model)

    def _ensure_model(self) -> DiscreteBayesianNetwork:
        """Raise a clear error if the model is not initialised."""
        if self.model is None:
            raise ModelInitializationError("Bayesian network model not built")
        return self.model

    def _ensure_inference_engine(self) -> VariableElimination:
        if self.inference_engine is None:
            raise ModelInferenceError("Inference engine not initialized")
        return self.inference_engine

    def _normalize_nodes(self, nodes: Any) -> List[str]:
        if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
            raise ModelInitializationError(
                "Network definition must contain a sequence of nodes",
                context={"type": type(nodes).__name__},
            )
        normalized: List[str] = []
        seen: Set[str] = set()
        for raw in nodes:
            node = str(raw).strip()
            if not node:
                raise ModelInitializationError("Node names must be non-empty", context={"node": raw})
            if node in seen:
                raise ModelInitializationError("Duplicate node in Bayesian network", context={"node": node})
            seen.add(node)
            normalized.append(node)
        if not normalized:
            raise ModelInitializationError("Network definition must contain at least one node")
        return normalized

    def _normalize_edges(self, edges: Any, nodes: Sequence[str]) -> List[Tuple[str, str]]:
        if edges is None:
            return []
        if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
            raise ModelInitializationError(
                "Network definition 'edges' must be a sequence",
                context={"type": type(edges).__name__},
            )
        node_set = set(nodes)
        normalized: List[Tuple[str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        for edge in edges:
            if not isinstance(edge, Sequence) or isinstance(edge, (str, bytes)) or len(edge) != 2:
                raise ModelInitializationError("Each edge must be a pair", context={"edge": edge})
            parent, child = str(edge[0]).strip(), str(edge[1]).strip()
            if parent not in node_set or child not in node_set:
                raise ModelInitializationError(
                    "Edge references unknown node",
                    context={"edge": (parent, child), "known_nodes": list(nodes)},
                )
            if parent == child:
                raise CircuitConstraintError("Self-loop detected in Bayesian network", context={"node": parent})
            key = (parent, child)
            if key not in seen:
                seen.add(key)
                normalized.append(key)
        return normalized

    def _initialize_state_metadata(self, nodes: Sequence[str]) -> None:
        raw_states = self.network_def.get("states", {}) or self.network_def.get("state_names", {}) or {}
        if raw_states and not isinstance(raw_states, Mapping):
            raise ModelInitializationError(
                "Network definition 'states' must be a dictionary when provided",
                context={"type": type(raw_states).__name__},
            )

        for node in nodes:
            states = raw_states.get(node, _DEFAULT_BINARY_STATES)
            if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
                raise ModelInitializationError(
                    "Node states must be a sequence",
                    context={"node": node, "states": states},
                )
            normalized_states = list(states)
            if len(normalized_states) < 2:
                raise ModelInitializationError(
                    "Each node must have at least two states",
                    context={"node": node, "states": normalized_states},
                )
            if len({self._state_key(s) for s in normalized_states}) != len(normalized_states):
                raise ModelInitializationError(
                    "Duplicate state labels for node",
                    context={"node": node, "states": normalized_states},
                )
            self._state_names[node] = normalized_states
            self._state_lookup[node] = {
                self._state_key(state): idx for idx, state in enumerate(normalized_states)
            }

            # Common binary aliases remain accepted even when the stored labels are booleans/ints/strings.
            if len(normalized_states) == 2:
                self._state_lookup[node].setdefault("0", 0)
                self._state_lookup[node].setdefault("false", 0)
                self._state_lookup[node].setdefault("1", 1)
                self._state_lookup[node].setdefault("true", 1)

    @staticmethod
    def _state_key(value: Any) -> str:
        return str(value).strip().lower()

    def _state_count(self, node: str) -> int:
        return len(self._state_names[node])

    def _true_state_index(self, node: str) -> int:
        states = self._state_names.get(node, _DEFAULT_BINARY_STATES)
        for idx, state in enumerate(states):
            if isinstance(state, bool) and state is True:
                return idx
            if self._state_key(state) in _TRUE_ALIASES:
                return idx
        return len(states) - 1

    def _false_state_index(self, node: str) -> int:
        states = self._state_names.get(node, _DEFAULT_BINARY_STATES)
        for idx, state in enumerate(states):
            if isinstance(state, bool) and state is False:
                return idx
            if self._state_key(state) in _FALSE_ALIASES:
                return idx
        return 0

    def _build_parent_map(self) -> Dict[str, List[str]]:
        """Map each node to its parents with optional cycle detection."""
        model = self._ensure_model()
        parent_map = {str(node): [str(parent) for parent in model.get_parents(node)] for node in model.nodes}
        if self.validate_dag:
            self._assert_acyclic(parent_map)
        return parent_map

    def _assert_acyclic(self, parent_map: Dict[str, List[str]]) -> None:
        visited: Set[str] = set()
        recursion_stack: Set[str] = set()

        def dfs(current: str) -> None:
            visited.add(current)
            recursion_stack.add(current)
            for parent in parent_map.get(current, []):
                if parent not in visited:
                    dfs(parent)
                elif parent in recursion_stack:
                    raise CircuitConstraintError(
                        "Cycle detected in Bayesian network",
                        context={"cycle_edge": (parent, current)},
                    )
            recursion_stack.remove(current)

        for node in parent_map:
            if node not in visited:
                dfs(node)

    def _validate_network_structure(self, parent_map: Dict[str, List[str]], cpt_data: Mapping[str, Any]) -> None:
        """Validate CPT compatibility with network structure without duplicating full pgmpy validation."""
        for cpt_node in cpt_data.keys():
            if cpt_node not in parent_map:
                message = f"CPT provided for unknown node '{cpt_node}'"
                if self.strict_cpt:
                    raise ReasoningValidationError(message, context={"node": cpt_node})
                logger.warning(message)

        for node, parents in parent_map.items():
            cpt_info = cpt_data.get(node)
            if cpt_info is None or not parents or not isinstance(cpt_info, Mapping):
                continue
            if "values" in cpt_info or "table" in cpt_info:
                continue
            expected_combinations = math.prod(self._state_count(parent) for parent in parents)
            actual_combinations = len(cpt_info)
            if actual_combinations != expected_combinations:
                message = (
                    f"Node {node} has {len(parents)} parents ({expected_combinations} parent-state "
                    f"combinations) but CPT mapping has {actual_combinations} entries"
                )
                if self.strict_cpt:
                    raise ReasoningValidationError(
                        message,
                        context={"node": node, "parents": parents, "expected": expected_combinations, "actual": actual_combinations},
                    )
                logger.warning(message)

    def _get_topological_order(self, parent_map: Dict[str, List[str]]) -> List[str]:
        """Kahn's algorithm for topological sorting."""
        model = self._ensure_model()
        in_degree = {str(node): 0 for node in model.nodes}
        child_map: Dict[str, List[str]] = {str(node): [] for node in model.nodes}
        for child, parents in parent_map.items():
            in_degree[child] = len(parents)
            for parent in parents:
                child_map[parent].append(child)

        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        order: List[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in child_map.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(in_degree):
            raise CircuitConstraintError("Network contains cycles; cannot topologically sort")
        return order

    # ----------------------------------------------------------------------
    # CPD construction
    # ----------------------------------------------------------------------
    def _build_root_cpd(self, node: str, cpt_info: Any) -> TabularCPD:
        """Build CPD for a node without parents."""
        cardinality = self._state_count(node)
        probabilities = self._root_probabilities(node, cpt_info, cardinality)
        values = [[prob] for prob in probabilities]
        return TabularCPD(
            variable=node,
            variable_card=cardinality,
            values=values,
            state_names={node: self._state_names[node]},
        )

    def _root_probabilities(self, node: str, cpt_info: Any, cardinality: int) -> List[float]:
        if isinstance(cpt_info, Mapping):
            if "values" in cpt_info:
                return self._probability_vector(node, cpt_info["values"], expected_size=cardinality)
            if "probabilities" in cpt_info:
                return self._probability_vector(node, cpt_info["probabilities"], expected_size=cardinality)
            if "prior" in cpt_info:
                prior_true = clamp_confidence(cpt_info["prior"])
                probabilities = [0.0] * cardinality
                true_idx = self._true_state_index(node)
                if cardinality == 2:
                    false_idx = 1 - true_idx
                    probabilities[false_idx] = 1.0 - prior_true
                    probabilities[true_idx] = prior_true
                    return self._normalize_probability_vector(probabilities, context={"node": node})
                remaining = (1.0 - prior_true) / max(1, cardinality - 1)
                for idx in range(cardinality):
                    probabilities[idx] = prior_true if idx == true_idx else remaining
                return self._normalize_probability_vector(probabilities, context={"node": node})
            if self._mapping_looks_like_distribution(node, cpt_info):
                return self._probability_vector(node, cpt_info, expected_size=cardinality)

        if isinstance(cpt_info, (int, float)):
            if cardinality != 2:
                raise ReasoningValidationError(
                    "Scalar root CPT is only supported for binary nodes",
                    context={"node": node, "cardinality": cardinality},
                )
            prior_true = clamp_confidence(cpt_info)
            probabilities = [0.0, 0.0]
            true_idx = self._true_state_index(node)
            false_idx = 1 - true_idx
            probabilities[false_idx] = 1.0 - prior_true
            probabilities[true_idx] = prior_true
            return probabilities

        if isinstance(cpt_info, Sequence) and not isinstance(cpt_info, (str, bytes)):
            return self._probability_vector(node, cpt_info, expected_size=cardinality)

        raise ReasoningValidationError(
            f"Invalid CPT format for root node '{node}'",
            context={"node": node, "cpt_type": type(cpt_info).__name__},
        )

    def _build_child_cpd(self, node: str, parents: List[str], cpt_info: Any) -> TabularCPD:
        """Build CPD for a node with parents."""
        cardinality = self._state_count(node)
        parent_cards = [self._state_count(parent) for parent in parents]
        num_parent_configs = math.prod(parent_cards)

        direct_values = self._extract_direct_cpd_values(cpt_info)
        if direct_values is not None:
            values = self._normalize_cpd_matrix(node, direct_values, cardinality, num_parent_configs)
        else:
            table = self._extract_cpt_table(cpt_info)
            values = [[] for _ in range(cardinality)]
            for parent_indices in itertools.product(*[range(card) for card in parent_cards]):
                cpt_entry = self._lookup_cpt_entry(table, parents, parent_indices)
                if cpt_entry is None:
                    vector = self._missing_probability_vector(node)
                    logger.warning(
                        "Missing CPT entry for node=%s parent_state=%s. Applied policy '%s'.",
                        node,
                        self._parent_state_key(parents, parent_indices, labels=True),
                        self.missing_cpt_policy,
                    )
                else:
                    vector = self._probability_vector(node, cpt_entry, expected_size=cardinality)
                for state_idx, prob in enumerate(vector):
                    values[state_idx].append(prob)

        return TabularCPD(
            variable=node,
            variable_card=cardinality,
            values=values,
            evidence=parents,
            evidence_card=parent_cards,
            state_names={node: self._state_names[node], **{parent: self._state_names[parent] for parent in parents}},
        )

    @staticmethod
    def _extract_direct_cpd_values(cpt_info: Any) -> Optional[Any]:
        if isinstance(cpt_info, Mapping):
            if "values" in cpt_info:
                return cpt_info["values"]
            if "cpd" in cpt_info:
                return cpt_info["cpd"]
        if isinstance(cpt_info, Sequence) and not isinstance(cpt_info, (str, bytes)):
            if cpt_info and all(isinstance(row, Sequence) and not isinstance(row, (str, bytes)) for row in cpt_info):
                return cpt_info
        return None

    @staticmethod
    def _extract_cpt_table(cpt_info: Any) -> Mapping[str, Any]:
        if isinstance(cpt_info, Mapping):
            if "table" in cpt_info and isinstance(cpt_info["table"], Mapping):
                return cpt_info["table"]
            if "conditional" in cpt_info and isinstance(cpt_info["conditional"], Mapping):
                return cpt_info["conditional"]
            return cpt_info
        raise ReasoningValidationError(
            "Conditional CPT must be a mapping or direct matrix",
            context={"cpt_type": type(cpt_info).__name__},
        )

    def _lookup_cpt_entry(self, table: Mapping[str, Any], parents: List[str], parent_indices: Tuple[int, ...]) -> Optional[Any]:
        candidate_keys = self._candidate_parent_keys(parents, parent_indices)
        for key in candidate_keys:
            if key in table:
                return table[key]
        return None

    def _candidate_parent_keys(self, parents: List[str], parent_indices: Tuple[int, ...]) -> List[Any]:
        labels = [self._state_names[parent][idx] for parent, idx in zip(parents, parent_indices)]
        label_text = [str(label) for label in labels]
        bool_text = [str(bool(idx)) for idx in parent_indices]
        int_text = [str(idx) for idx in parent_indices]

        candidates: List[Any] = []
        for values in (label_text, bool_text, int_text):
            candidates.extend(
                [
                    ",".join(values),
                    "|".join(values),
                    ";".join(values),
                    tuple(values),
                    str(tuple(values)),
                ]
            )
        named = ",".join(f"{parent}={value}" for parent, value in zip(parents, label_text))
        named_int = ",".join(f"{parent}={value}" for parent, value in zip(parents, int_text))
        candidates.extend([named, named_int])
        if len(parent_indices) == 1:
            candidates.extend([label_text[0], bool_text[0], int_text[0], labels[0], parent_indices[0]])
        return candidates

    def _parent_state_key(self, parents: List[str], parent_indices: Tuple[int, ...], *, labels: bool) -> str:
        values = [self._state_names[parent][idx] if labels else idx for parent, idx in zip(parents, parent_indices)]
        return ",".join(str(value) for value in values)

    def _mapping_looks_like_distribution(self, node: str, mapping: Mapping[str, Any]) -> bool:
        lookup = self._state_lookup.get(node, {})
        lowered_keys = {self._state_key(key) for key in mapping.keys()}
        return bool(lowered_keys & set(lookup.keys())) or bool(lowered_keys & (_TRUE_ALIASES | _FALSE_ALIASES))

    def _probability_vector(self, node: str, raw: Any, *, expected_size: int) -> List[float]:
        """Convert scalar/list/dict CPT value to a normalized probability vector."""
        if isinstance(raw, Mapping):
            vector = [0.0] * expected_size
            found = False
            for key, value in raw.items():
                key_norm = self._state_key(key)
                idx = self._state_lookup[node].get(key_norm)
                if idx is None and key_norm in _TRUE_ALIASES:
                    idx = self._true_state_index(node)
                if idx is None and key_norm in _FALSE_ALIASES:
                    idx = self._false_state_index(node)
                if idx is None:
                    continue
                vector[idx] = float(value)
                found = True
            if not found:
                raise ReasoningValidationError(
                    "Probability mapping does not contain known states",
                    context={"node": node, "states": self._state_names[node], "mapping_keys": list(raw.keys())},
                )
            return self._normalize_probability_vector(vector, context={"node": node, "raw": raw})

        if isinstance(raw, (int, float)):
            if expected_size != 2:
                raise ReasoningValidationError(
                    "Scalar probability is only supported for binary nodes",
                    context={"node": node, "cardinality": expected_size},
                )
            p_true = clamp_confidence(raw)
            vector = [0.0, 0.0]
            true_idx = self._true_state_index(node)
            false_idx = 1 - true_idx
            vector[false_idx] = 1.0 - p_true
            vector[true_idx] = p_true
            return vector

        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if len(raw) == expected_size and not any(isinstance(v, Sequence) and not isinstance(v, (str, bytes)) for v in raw):
                vector = [float(v) for v in raw]
                return self._normalize_probability_vector(vector, context={"node": node, "raw": raw})
            if expected_size == 2 and len(raw) == 2 and all(isinstance(v, Sequence) and not isinstance(v, (str, bytes)) for v in raw):
                # Root CPD matrix form [[p_false], [p_true]].
                flattened = [float(row[0]) for row in raw if len(row) > 0]
                if len(flattened) == expected_size:
                    return self._normalize_probability_vector(flattened, context={"node": node, "raw": raw})

        raise ReasoningValidationError(
            "Unsupported probability vector format",
            context={"node": node, "raw_type": type(raw).__name__, "raw": raw},
        )

    def _normalize_cpd_matrix(
        self,
        node: str,
        raw_values: Any,
        cardinality: int,
        num_parent_configs: int,
    ) -> List[List[float]]:
        try:
            matrix = np.asarray(raw_values, dtype=float)
        except Exception as exc:
            raise ReasoningValidationError(
                "Direct CPD values must be numeric",
                cause=exc,
                context={"node": node},
            ) from exc

        if matrix.ndim == 1:
            if cardinality != 2 or matrix.size != num_parent_configs:
                raise ReasoningValidationError(
                    "1D CPD vector is only valid as binary true-probability columns",
                    context={"node": node, "shape": list(matrix.shape)},
                )
            true_idx = self._true_state_index(node)
            false_idx = 1 - true_idx
            values = np.zeros((2, num_parent_configs), dtype=float)
            values[true_idx, :] = matrix
            values[false_idx, :] = 1.0 - matrix
            matrix = values

        if matrix.shape[0] != cardinality:
            raise ReasoningValidationError(
                "CPD matrix row count must match node cardinality",
                context={"node": node, "expected_rows": cardinality, "actual_shape": list(matrix.shape)},
            )

        matrix = matrix.reshape(cardinality, -1)
        if matrix.shape[1] != num_parent_configs:
            raise ReasoningValidationError(
                "CPD matrix column count must match parent-state combinations",
                context={"node": node, "expected_columns": num_parent_configs, "actual_shape": list(matrix.shape)},
            )

        normalized_columns = []
        for col_idx in range(num_parent_configs):
            vector = self._normalize_probability_vector(
                matrix[:, col_idx].tolist(),
                context={"node": node, "column": col_idx},
            )
            normalized_columns.append(vector)
        return np.asarray(normalized_columns, dtype=float).T.tolist()

    def _normalize_probability_vector(self, values: Sequence[float], *, context: Optional[Dict[str, Any]] = None) -> List[float]:
        context = context or {}
        vector = [float(v) for v in values]
        if any(math.isnan(v) or math.isinf(v) for v in vector):
            raise ReasoningValidationError("Probability vector contains non-finite values", context=context)
        if any(v < -self.epsilon for v in vector):
            raise ReasoningValidationError("Probability vector contains negative values", context={**context, "values": vector})

        vector = [max(0.0, v) for v in vector]
        total = sum(vector)
        if total <= self.epsilon:
            if self.strict_cpt:
                raise ReasoningValidationError("Probability vector sum is zero", context=context)
            size = max(1, len(vector))
            return [1.0 / size] * size

        if abs(total - 1.0) > self.epsilon:
            if not self.normalize_cpds and self.strict_cpt:
                raise ReasoningValidationError(
                    "Probability vector does not sum to 1",
                    context={**context, "sum": total, "values": vector},
                )
            vector = [v / total for v in vector]
        return vector

    def _build_missing_cpd(self, node: str, parents: List[str]) -> TabularCPD:
        """Build missing CPD according to configured policy."""
        if self.missing_cpt_policy == "error":
            raise ModelInitializationError("Missing CPT data", context={"node": node, "parents": parents})
        if parents:
            cardinality = self._state_count(node)
            parent_cards = [self._state_count(parent) for parent in parents]
            num_cols = math.prod(parent_cards)
            vector = self._missing_probability_vector(node)
            values = [[vector[state_idx]] * num_cols for state_idx in range(cardinality)]
            return TabularCPD(
                variable=node,
                variable_card=cardinality,
                values=values,
                evidence=parents,
                evidence_card=parent_cards,
                state_names={node: self._state_names[node], **{parent: self._state_names[parent] for parent in parents}},
            )
        return self._build_root_cpd(node, self._missing_probability_vector(node))

    def _missing_probability_vector(self, node: str) -> List[float]:
        cardinality = self._state_count(node)
        if self.missing_cpt_policy in {"uniform", "balanced"}:
            return [1.0 / cardinality] * cardinality
        if self.missing_cpt_policy in {"default", "default_probability"}:
            if cardinality != 2:
                return [1.0 / cardinality] * cardinality
            p_true = self.default_probability
            vector = [0.0, 0.0]
            true_idx = self._true_state_index(node)
            false_idx = 1 - true_idx
            vector[false_idx] = 1.0 - p_true
            vector[true_idx] = p_true
            return vector
        raise ReasoningConfigurationError(
            "Unsupported pgmpy_wrapper.missing_cpt_policy",
            context={"missing_cpt_policy": self.missing_cpt_policy},
        )

    def _validate_model(self) -> bool:
        """Comprehensive model validation with additional CPD diagnostics."""
        model = self._ensure_model()
        try:
            if not model.check_model():
                logger.error("pgmpy model.check_model() failed")
                return False
        except Exception as exc:
            logger.error("pgmpy model validation raised: %s", exc)
            return False

        for cpd in model.get_cpds():
            try:
                if not cpd.is_valid_cpd():
                    logger.error("Invalid CPD for %s", cpd.variable)
                    return False
                values = self._cpd_values(cpd)
                for idx in range(values.shape[1]):
                    prob_sum = float(np.sum(values[:, idx]))
                    if abs(prob_sum - 1.0) > max(self.epsilon, 1e-5):
                        logger.error(
                            "CPD for %s does not sum to 1 for parent config %s: sum=%.8f values=%s",
                            cpd.variable,
                            idx,
                            prob_sum,
                            values[:, idx].tolist(),
                        )
                        return False
            except Exception as exc:
                logger.error("CPD validation failed for %s: %s", getattr(cpd, "variable", "unknown"), exc)
                return False
        return True

    @staticmethod
    def _cpd_values(cpd: TabularCPD) -> np.ndarray:
        if hasattr(cpd, "get_values"):
            values = cpd.get_values()
        else:
            values = cpd.values
        matrix = np.asarray(values, dtype=float)
        return matrix.reshape(cpd.variable_card, -1)

    @staticmethod
    def _cpd_evidence(cpd: TabularCPD) -> List[str]:
        if hasattr(cpd, "get_evidence"):
            return list(cpd.get_evidence())
        return list(getattr(cpd, "evidence", []) or [])

    # ----------------------------------------------------------------------
    # Inference API
    # ----------------------------------------------------------------------
    def query(
        self,
        query_variable: str,
        evidence: Optional[EvidenceMap] = None,
        use_cache: bool = True,
        state: StateValue = True,
    ) -> float:
        """Return P(query_variable=state | evidence). Defaults to True for binary compatibility."""
        self._query_count += 1
        inference_engine = self._ensure_inference_engine()
        model = self._ensure_model()
        query_variable = self._require_node(query_variable, label="query_variable")
        target_state_idx = self._normalize_state_value(query_variable, state)
        pgmpy_evidence = self._normalize_evidence(evidence or {})

        if query_variable in pgmpy_evidence:
            return 1.0 if pgmpy_evidence[query_variable] == target_state_idx else 0.0

        cache_key = self._cache_key("query", query_variable, target_state_idx, pgmpy_evidence)
        if use_cache and self.cache_enabled:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        try:
            kwargs = {
                "variables": [query_variable],
                "evidence": pgmpy_evidence,
                "show_progress": self.show_progress,
            }
            result = inference_engine.query(**kwargs)
            result_values = np.asarray(result.values, dtype=float).reshape(-1)
            probability = float(result_values[target_state_idx])
            probability = clamp_confidence(probability)
            if use_cache and self.cache_enabled:
                self._set_cached(cache_key, probability)
            return probability
        except ReasoningError:
            raise
        except Exception as exc:
            logger.error("Query failed for %s with evidence=%s: %s", query_variable, evidence, exc)
            raise ModelInferenceError(
                "pgmpy inference query failed",
                cause=exc,
                context={"query_variable": query_variable, "evidence": evidence, "model_nodes": list(model.nodes)},
            ) from exc

    def query_distribution(
        self,
        query_variable: str,
        evidence: Optional[EvidenceMap] = None,
        use_cache: bool = True,
    ) -> Distribution:
        """Return full posterior distribution for one query variable."""
        query_variable = self._require_node(query_variable, label="query_variable")
        pgmpy_evidence = self._normalize_evidence(evidence or {})

        if query_variable in pgmpy_evidence:
            observed_idx = pgmpy_evidence[query_variable]
            return {
                str(state): (1.0 if idx == observed_idx else 0.0)
                for idx, state in enumerate(self._state_names[query_variable])
            }

        cache_key = self._cache_key("distribution", query_variable, None, pgmpy_evidence)
        if use_cache and self.cache_enabled:
            cached = self._get_cached(cache_key)
            if isinstance(cached, dict):
                return cached

        inference_engine = self._ensure_inference_engine()
        try:
            result = inference_engine.query(
                variables=[query_variable],
                evidence=pgmpy_evidence,
                show_progress=self.show_progress,
            )
            values = np.asarray(result.values, dtype=float).reshape(-1)
            distribution = {
                str(state): clamp_confidence(float(values[idx]))
                for idx, state in enumerate(self._state_names[query_variable])
            }
            if use_cache and self.cache_enabled:
                self._set_cached(cache_key, distribution)  # type: ignore[arg-type]
            return distribution
        except Exception as exc:
            raise ModelInferenceError(
                "pgmpy distribution query failed",
                cause=exc,
                context={"query_variable": query_variable, "evidence": evidence},
            ) from exc

    def batch_query(
        self,
        query_variables: Sequence[str],
        evidence: Optional[EvidenceMap] = None,
        state: StateValue = True,
        use_cache: bool = True,
    ) -> Dict[str, float]:
        """Compute same-state marginals for multiple variables."""
        if not query_variables:
            return {}
        return {
            self._require_node(variable, label="query_variable"): self.query(variable, evidence, use_cache=use_cache, state=state)
            for variable in query_variables
        }

    def map_query(
        self,
        evidence: Optional[EvidenceMap] = None,
        variables: Optional[Sequence[str]] = None,
        include_evidence: bool = False,
        as_state_names: bool = False,
    ) -> Dict[str, Any]:
        """Maximum a posteriori assignment for all or selected unobserved nodes."""
        inference_engine = self._ensure_inference_engine()
        model = self._ensure_model()
        pgmpy_evidence = self._normalize_evidence(evidence or {})

        if variables is None:
            query_variables = [str(node) for node in model.nodes if str(node) not in pgmpy_evidence]
        else:
            query_variables = [self._require_node(variable, label="map_variable") for variable in variables]
            query_variables = [variable for variable in query_variables if variable not in pgmpy_evidence]

        if not query_variables:
            return self._format_assignment(pgmpy_evidence if include_evidence else {}, as_state_names=as_state_names)

        try:
            result = inference_engine.map_query(
                variables=query_variables,
                evidence=pgmpy_evidence,
                show_progress=self.show_progress,
            )
            normalized_result = {str(node): int(value) for node, value in result.items()}
            if include_evidence:
                normalized_result.update(pgmpy_evidence)
            return self._format_assignment(normalized_result, as_state_names=as_state_names)
        except Exception as exc:
            raise ModelInferenceError(
                "MAP query failed",
                cause=exc,
                context={"variables": query_variables, "evidence": evidence},
            ) from exc

    def get_all_marginals(
        self,
        evidence: Optional[EvidenceMap] = None,
        state: StateValue = True,
        use_cache: bool = False,
    ) -> Dict[str, float]:
        """Compute P(node=state | evidence) for every node."""
        evidence = evidence or {}
        results: Dict[str, float] = {}
        for node in self._ensure_model().nodes:
            node_name = str(node)
            try:
                results[node_name] = self.query(node_name, evidence, use_cache=use_cache, state=state)
            except ModelInferenceError:
                logger.warning("Could not compute marginal for %s; using default_probability", node_name)
                results[node_name] = self.default_probability
        return results

    def conditional_probability_table(self, target: str, evidence: Optional[EvidenceMap] = None) -> float:
        """Compatibility alias for query()."""
        return self.query(target, evidence or {})

    def explain_query(
        self,
        query_variable: str,
        evidence: Optional[EvidenceMap] = None,
        state: StateValue = True,
    ) -> Dict[str, Any]:
        """Return query result with structural context for diagnostics/UI layers."""
        query_variable = self._require_node(query_variable, label="query_variable")
        probability = self.query(query_variable, evidence, state=state)
        return json_safe_reasoning_state(
            {
                "query_variable": query_variable,
                "state": state,
                "probability": probability,
                "evidence": evidence or {},
                "parents": self.get_parents(query_variable),
                "children": self.get_children(query_variable),
                "markov_blanket": sorted(self.get_markov_blanket(query_variable)),
                "cache_info": self.cache_info(),
            }
        )

    # ----------------------------------------------------------------------
    # Evidence and state normalization
    # ----------------------------------------------------------------------
    def _require_node(self, node: Any, *, label: str = "node") -> str:
        node_name = str(node).strip()
        if node_name not in set(str(n) for n in self._ensure_model().nodes):
            raise ReasoningValidationError(
                f"{label} '{node_name}' not in Bayesian network",
                context={"node": node_name, "known_nodes": list(self._ensure_model().nodes)},
            )
        return node_name

    def _normalize_evidence(self, evidence: Mapping[str, StateValue]) -> Dict[str, int]:
        if not isinstance(evidence, Mapping):
            raise ReasoningValidationError(
                "Evidence must be a dictionary",
                context={"type": type(evidence).__name__},
            )
        normalized: Dict[str, int] = {}
        known_nodes = set(str(node) for node in self._ensure_model().nodes)
        for raw_node, raw_value in evidence.items():
            node = str(raw_node).strip()
            if node not in known_nodes:
                if self.ignore_unknown_evidence:
                    logger.warning("Ignoring unknown evidence variable '%s'", node)
                    continue
                raise ReasoningValidationError(
                    "Evidence references unknown variable",
                    context={"node": node, "known_nodes": sorted(known_nodes)},
                )
            normalized[node] = self._normalize_state_value(node, raw_value)
        return normalized

    def _normalize_state_value(self, node: str, value: StateValue) -> int:
        states = self._state_names[node]
        lookup = self._state_lookup[node]

        if isinstance(value, bool) and len(states) == 2:
            return self._true_state_index(node) if value else self._false_state_index(node)
        if isinstance(value, int) and not isinstance(value, bool):
            if 0 <= value < len(states):
                return int(value)
            raise ReasoningValidationError(
                "State index out of range",
                context={"node": node, "value": value, "states": states},
            )
        if isinstance(value, float):
            if len(states) != 2:
                raise ReasoningValidationError(
                    "Float evidence is only supported for binary nodes",
                    context={"node": node, "value": value, "states": states},
                )
            if 0.0 <= value <= 1.0:
                return self._true_state_index(node) if value >= 0.5 else self._false_state_index(node)

        key = self._state_key(value)
        if key in lookup:
            return lookup[key]
        if key in _TRUE_ALIASES:
            return self._true_state_index(node)
        if key in _FALSE_ALIASES:
            return self._false_state_index(node)
        raise ReasoningValidationError(
            "Unknown state value for node",
            context={"node": node, "value": value, "allowed_states": states},
        )

    def _format_assignment(self, assignment: Mapping[str, int], *, as_state_names: bool) -> Dict[str, Any]:
        formatted: Dict[str, Any] = {}
        for node, state_idx in assignment.items():
            state_value = self._state_names[node][int(state_idx)]
            if as_state_names:
                formatted[node] = state_value
            elif len(self._state_names[node]) == 2:
                formatted[node] = int(state_idx) == self._true_state_index(node)
            else:
                formatted[node] = state_value
        return formatted

    # ----------------------------------------------------------------------
    # Cache helpers
    # ----------------------------------------------------------------------
    def _cache_key(self, kind: str, query_variable: str, state_idx: Optional[int], evidence: Mapping[str, int]) -> Tuple[Any, ...]:
        return (
            kind,
            self._definition_signature,
            query_variable,
            state_idx,
            tuple(sorted(evidence.items())),
        )

    def _get_cached(self, key: Tuple[Any, ...]) -> Optional[Any]:
        if self.max_cache_entries <= 0:
            return None
        entry = self._inference_cache.get(key)
        if entry is None:
            self._cache_misses += 1
            return None
        value, timestamp = entry
        if self.cache_ttl_seconds > 0 and (time.time() - timestamp) > self.cache_ttl_seconds:
            self._inference_cache.pop(key, None)
            self._cache_misses += 1
            return None
        self._cache_hits += 1
        self._inference_cache.move_to_end(key)
        return value

    def _set_cached(self, key: Tuple[Any, ...], value: Any) -> None:
        if self.max_cache_entries <= 0:
            return
        self._inference_cache[key] = (value, time.time())
        self._inference_cache.move_to_end(key)
        while len(self._inference_cache) > self.max_cache_entries:
            self._inference_cache.popitem(last=False)

    def clear_cache(self) -> None:
        self._inference_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def cache_info(self) -> Dict[str, Any]:
        return {
            "enabled": self.cache_enabled,
            "entries": len(self._inference_cache),
            "max_entries": self.max_cache_entries,
            "ttl_seconds": self.cache_ttl_seconds,
            "query_count": self._query_count,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

    # ----------------------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------------------
    def sample(self, n_samples: int = 1000, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate samples from the joint distribution."""
        if self.sampler is None:
            raise ModelInferenceError("Sampler not available")
        model = self._ensure_model()
        n = bounded_iterations(n_samples, minimum=1, maximum=self.max_sample_size)
        sample_seed = seed if seed is not None else self.default_seed
    
        if sample_seed is not None:
            np.random.seed(int(sample_seed))
    
        try:
            # Determine which sampling method to use
            if isinstance(self.sampler, GibbsSampling):
                # GibbsSampling has a sample() method
                sample_method = self.sampler.sample
                kwargs = {"size": n}
                sig_params = inspect.signature(sample_method).parameters
                if "seed" in sig_params and sample_seed is not None:
                    kwargs["seed"] = int(sample_seed)
                if "show_progress" in sig_params:
                    kwargs["show_progress"] = self.show_progress
                samples_df = sample_method(**kwargs)
            else:
                # BayesianModelSampling uses forward_sample()
                sample_method = self.sampler.forward_sample
                kwargs = {"size": n}
                sig_params = inspect.signature(sample_method).parameters
                if "seed" in sig_params and sample_seed is not None:
                    kwargs["seed"] = int(sample_seed)
                if "show_progress" in sig_params:
                    kwargs["show_progress"] = self.show_progress
                samples_df = sample_method(**kwargs)
    
            # Convert DataFrame to list of dicts
            samples: List[Dict[str, Any]] = []
            for _, row in samples_df.iterrows():
                sample_dict = {}
                for node in model.nodes:
                    node_name = str(node)
                    raw_value = row[node_name]
                    state_idx = self._normalize_state_value(node_name, raw_value)
                    sample_dict[node_name] = self._format_assignment({node_name: state_idx}, as_state_names=False)[node_name]
                samples.append(sample_dict)
            return samples
        except Exception as exc:
            raise ModelInferenceError(
                "Sampling failed",
                cause=exc,
                context={"n_samples": n, "seed": sample_seed, "backend": self.sampling_backend, "sampler_type": type(self.sampler).__name__},
            ) from exc

    def sample_with_evidence(
        self,
        evidence: EvidenceMap,
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate evidence-consistent samples by rejection over joint samples.

        This is intentionally conservative and deterministic enough for diagnostics;
        exact posterior inference should use ``query``/``map_query``.
        """
        normalized_evidence = self._normalize_evidence(evidence)
        target_n = bounded_iterations(n_samples, minimum=1, maximum=self.max_sample_size)
        batch_size = min(self.max_sample_size, max(target_n * 4, 128))
        accepted: List[Dict[str, Any]] = []
        attempts = 0
        max_attempts = bounded_iterations(
            self.wrapper_config.get("max_rejection_batches", 25),
            minimum=1,
            maximum=10_000,
        )

        while len(accepted) < target_n and attempts < max_attempts:
            attempts += 1
            batch = self.sample(batch_size, seed=(None if seed is None else int(seed) + attempts))
            for sample in batch:
                if self._sample_matches_evidence(sample, normalized_evidence):
                    accepted.append(sample)
                    if len(accepted) >= target_n:
                        break
        if len(accepted) < target_n:
            logger.warning(
                "Evidence sampling accepted %s/%s requested samples after %s batches",
                len(accepted),
                target_n,
                attempts,
            )
        return accepted[:target_n]

    def _sample_matches_evidence(self, sample: Mapping[str, Any], evidence: Mapping[str, int]) -> bool:
        for node, state_idx in evidence.items():
            sample_idx = self._normalize_state_value(node, sample[node])
            if sample_idx != state_idx:
                return False
        return True

    def estimate_marginals_from_samples(
        self,
        samples: Sequence[Mapping[str, Any]],
        state: StateValue = True,
    ) -> Dict[str, float]:
        """Estimate node marginals from samples using the wrapper's state normalization."""
        if not samples:
            return {}
        model = self._ensure_model()
        counts = {str(node): 0 for node in model.nodes}
        targets = {str(node): self._normalize_state_value(str(node), state) for node in model.nodes}
        for sample in samples:
            for node in counts:
                if node in sample and self._normalize_state_value(node, sample[node]) == targets[node]:
                    counts[node] += 1
        return {node: count / len(samples) for node, count in counts.items()}

    # ----------------------------------------------------------------------
    # Structural analysis
    # ----------------------------------------------------------------------
    def get_parents(self, node: str) -> List[str]:
        model = self._ensure_model()
        node = self._require_node(node)
        return [str(parent) for parent in model.get_parents(node)]

    def get_children(self, node: str) -> List[str]:
        model = self._ensure_model()
        node = self._require_node(node)
        return [str(child) for child in model.get_children(node)]

    def get_markov_blanket(self, node: str) -> Set[str]:
        model = self._ensure_model()
        node = self._require_node(node)
        blanket = set(str(parent) for parent in model.get_parents(node))
        children = [str(child) for child in model.get_children(node)]
        blanket.update(children)
        for child in children:
            blanket.update(str(parent) for parent in model.get_parents(child))
        blanket.discard(node)
        return blanket

    def is_d_separated(self, x: str, y: str, z: Set[str]) -> bool:
        model = self._ensure_model()
        x = self._require_node(x, label="x")
        y = self._require_node(y, label="y")
        observed = {self._require_node(node, label="observed") for node in z}
        try:
            if hasattr(model, "is_dconnected"):
                return bool(model.is_dconnected(x, y, observed=observed)) is False
            if hasattr(model, "is_dconnected"):
                return not bool(model.is_dconnected(x, y, observed=observed))
        except TypeError:
            # Some pgmpy/networkx versions require a list-like observed argument.
            return not bool(model.is_dconnected(x, y, observed=list(observed)))
        raise ModelInferenceError("Installed pgmpy version does not expose d-separation checks")

    def topological_order(self) -> List[str]:
        return list(self._topological_order)

    def structural_diagnostics(self) -> Dict[str, Any]:
        model = self._ensure_model()
        nodes = [str(node) for node in model.nodes]
        edges = [(str(parent), str(child)) for parent, child in model.edges]
        return json_safe_reasoning_state(
            {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "nodes": nodes,
                "edges": edges,
                "topological_order": self.topological_order(),
                "parents": {node: self.get_parents(node) for node in nodes},
                "children": {node: self.get_children(node) for node in nodes},
                "state_names": self._state_names,
            }
        )

    # ----------------------------------------------------------------------
    # Serialisation and persistence
    # ----------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        model = self._ensure_model()
        cpt_export: Dict[str, Any] = {}
        for cpd in model.get_cpds():
            node = str(cpd.variable)
            parents = self._cpd_evidence(cpd)
            values = self._cpd_values(cpd)
            if not parents:
                if self._state_count(node) == 2:
                    cpt_export[node] = {"prior": values[self._true_state_index(node), 0]}
                else:
                    cpt_export[node] = {
                        "probabilities": {
                            str(state): float(values[idx, 0])
                            for idx, state in enumerate(self._state_names[node])
                        }
                    }
                continue

            mapping: Dict[str, Any] = {}
            parent_cards = [self._state_count(parent) for parent in parents]
            for col_idx, parent_indices in enumerate(itertools.product(*[range(card) for card in parent_cards])):
                key = self._parent_state_key(parents, parent_indices, labels=True)
                if self._state_count(node) == 2:
                    mapping[key] = {
                        "false": float(values[self._false_state_index(node), col_idx]),
                        "true": float(values[self._true_state_index(node), col_idx]),
                    }
                else:
                    mapping[key] = {
                        str(state): float(values[state_idx, col_idx])
                        for state_idx, state in enumerate(self._state_names[node])
                    }
            cpt_export[node] = mapping
        return json_safe_reasoning_state(
            {
                "nodes": [str(node) for node in model.nodes],
                "edges": [(str(parent), str(child)) for parent, child in model.edges],
                "states": self._state_names,
                "cpt": cpt_export,
            }
        )

    def save_json(self, path: str, *, atomic: bool = True) -> None:
        """Persist network definition to JSON. Atomic write is enabled by default."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if not atomic:
            with target.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=self.export_indent)
            logger.info("Network saved to %s", target)
            return

        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=self.export_indent)
                handle.write("\n")
            os.replace(tmp_name, target)
            logger.info("Network saved atomically to %s", target)
        except Exception as exc:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise KnowledgePersistenceError(
                "Failed to persist Bayesian network JSON",
                cause=exc,
                context={"path": str(target)},
            ) from exc

    @classmethod
    def load_json(cls, path: str) -> "PgmpyBayesianNetwork":
        source = Path(path)
        if not source.exists():
            raise ResourceLoadError("Bayesian network JSON file not found", context={"path": str(source)})
        try:
            with source.open("r", encoding="utf-8") as handle:
                definition = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ResourceLoadError(
                "Invalid Bayesian network JSON",
                cause=exc,
                context={"path": str(source)},
            ) from exc
        return cls(definition)

    # ----------------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------------
    @property
    def summary(self) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "Model not built"}
        return json_safe_reasoning_state(
            {
                "nodes": [str(node) for node in self.model.nodes],
                "node_count": len(list(self.model.nodes)),
                "edge_count": len(list(self.model.edges)),
                "edges": [(str(parent), str(child)) for parent, child in self.model.edges],
                "has_inference": self.inference_engine is not None,
                "has_sampler": self.sampler is not None,
                "cached_queries": len(self._inference_cache),
                "cache": self.cache_info(),
                "signature": self._definition_signature,
                "built_at_ms": self._built_at_ms,
            }
        )

    def diagnostics(self) -> Dict[str, Any]:
        model = self._ensure_model()
        cpd_diagnostics = []
        for cpd in model.get_cpds()[: self.max_diagnostics_cpds]:
            values = self._cpd_values(cpd)
            cpd_diagnostics.append(
                {
                    "variable": str(cpd.variable),
                    "evidence": self._cpd_evidence(cpd),
                    "variable_card": int(cpd.variable_card),
                    "columns": int(values.shape[1]),
                    "min_probability": float(np.min(values)),
                    "max_probability": float(np.max(values)),
                    "column_sum_min": float(np.min(values.sum(axis=0))),
                    "column_sum_max": float(np.max(values.sum(axis=0))),
                }
            )
        return json_safe_reasoning_state(
            {
                "summary": self.summary,
                "structure": self.structural_diagnostics(),
                "cpds": cpd_diagnostics,
                "config": {
                    "normalize_cpds": self.normalize_cpds,
                    "strict_cpt": self.strict_cpt,
                    "missing_cpt_policy": self.missing_cpt_policy,
                    "sampling_backend": self.sampling_backend,
                    "validate_model_on_build": self.validate_model_on_build,
                    "validate_dag": self.validate_dag,
                },
            }
        )


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Pgmpy Wrapper ===\n")
    printer.status("TEST", "Pgmpy Wrapper initialized", "info")

    test_network = {
        "nodes": ["Rain", "Sprinkler", "WetGrass", "Slippery"],
        "edges": [
            ["Rain", "Sprinkler"],
            ["Rain", "WetGrass"],
            ["Sprinkler", "WetGrass"],
            ["WetGrass", "Slippery"],
        ],
        "cpt": {
            "Rain": {"prior": 0.35},
            "Sprinkler": {
                "False": {"false": 0.40, "true": 0.60},
                "True": {"false": 0.90, "true": 0.10},
            },
            "WetGrass": {
                "False,False": {"false": 0.98, "true": 0.02},
                "False,True": {"false": 0.25, "true": 0.75},
                "True,False": {"false": 0.20, "true": 0.80},
                "True,True": {"false": 0.02, "true": 0.98},
            },
            "Slippery": {
                "False": {"false": 0.95, "true": 0.05},
                "True": {"false": 0.30, "true": 0.70},
            },
        },
    }

    bn = PgmpyBayesianNetwork(test_network)

    p_rain = bn.query("Rain")
    printer.pretty("P(Rain=True)", p_rain, "success")

    p_slippery = bn.query("Slippery", {"WetGrass": True})
    printer.pretty("P(Slippery=True | WetGrass=True)", p_slippery, "success")

    distribution = bn.query_distribution("WetGrass", {"Rain": True})
    printer.pretty("P(WetGrass | Rain=True)", distribution, "success")

    map_assignment = bn.map_query({"WetGrass": True})
    printer.pretty("MAP assignment given WetGrass=True", map_assignment, "success")

    marginals = bn.get_all_marginals({"Rain": True})
    printer.pretty("Marginals given Rain=True", marginals, "success")

    samples = bn.sample(25, seed=7)
    estimated = bn.estimate_marginals_from_samples(samples)
    printer.pretty("Sample-estimated marginals", estimated, "success")

    assert 0.0 <= p_rain <= 1.0
    assert 0.0 <= p_slippery <= 1.0
    assert abs(sum(distribution.values()) - 1.0) < 1e-6
    assert "Rain" in map_assignment
    assert set(marginals.keys()) == set(test_network["nodes"])
    assert len(samples) == 25
    assert bn.get_parents("WetGrass") == ["Rain", "Sprinkler"]
    assert "Rain" in bn.get_markov_blanket("WetGrass")

    export_path = Path(tempfile.gettempdir()) / "slai_pgmpy_wrapper_test_network.json"
    bn.save_json(str(export_path))
    reloaded = PgmpyBayesianNetwork.load_json(str(export_path))
    assert abs(reloaded.query("Rain") - p_rain) < 1e-9
    bn.clear_cache()
    printer.pretty("Network summary", bn.summary, "success")

    print("\n=== Test ran successfully ===\n")
