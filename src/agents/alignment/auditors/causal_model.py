"""
Structural Causal Model Framework
Implements causal graph operations for counterfactual analysis through:
- Directed acyclic graph construction (Pearl, 2009)
- Backdoor/frontdoor adjustment sets (Shpitser et al., 2010)
- Doubly robust estimation (Bang & Robins, 2005)
"""

from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import tempfile
import networkx as nx
import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union
from scipy.stats import pearsonr, norm
from sklearn.covariance import GraphicalLasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from statsmodels.api import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.sandbox.regression.gmm import IV2SLS

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.alignment_errors import *
from ..utils.alignment_helpers import *
from ..alignment_memory import AlignmentMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Causal Model")
printer = PrettyPrinter

@dataclass
class CausalEffectEstimate:
    """
    Canonical container for causal effect estimation results.

    The object keeps the effect estimate machine-readable while also preserving
    contextual metadata required by audit logging, intervention reports, and
    downstream model diagnostics.
    """

    treatment: str
    outcome: str
    method: str
    effect: float
    std_error: Optional[float] = None
    p_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    n_obs: Optional[int] = None
    instrument: Optional[str] = None
    adjustment_set: List[str] = field(default_factory=list)
    mediator_set: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_series(self) -> pd.Series:
        return pd.Series(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "method": self.method,
            "effect": self.effect,
            "std_error": self.std_error,
            "p_value": self.p_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_obs": self.n_obs,
            "instrument": self.instrument,
            "adjustment_set": list(self.adjustment_set),
            "mediator_set": list(self.mediator_set),
            "metadata": dict(self.metadata),
        }


@dataclass
class StructuralEquation:
    """Container for a node-level structural equation."""

    node: str
    parents: List[str]
    equation_type: str
    model: Any = None
    mean: Optional[float] = None
    variance: Optional[float] = None
    predictors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "parents": list(self.parents),
            "equation_type": self.equation_type,
            "mean": self.mean,
            "variance": self.variance,
            "predictors": list(self.predictors),
        }


class CausalGraphBuilder:
    """
    Domain-aware causal structure learning implementing:
    - Constraint-based causal discovery (PC algorithm)
    - Score-based structure optimization (BIC scoring)
    - Confounder detection via latent variable analysis
    """

    DEFAULT_REQUIRED_CONFIG_KEYS = (
        "conditional_independence_test",
        "min_adjacency_confidence",
        "max_parents",
        "forbidden_edges",
        "required_edges",
        "structure_learning_method",
        "latent_confounder_detection",
        "tetrad_path",
        "fci_max_conditioning_set",
        "use_inverse_covariance",
        "significance_level",
    )

    def __init__(
        self,
        config_section_name: str = "causal_model",
        config_file_path: Optional[str] = None,
        alignment_memory: Optional[AlignmentMemory] = None,
    ):
        self.config = load_global_config()
        self.config_section_name = ensure_non_empty_string(
            config_section_name,
            "config_section_name",
            error_cls=ConfigurationError,
        )
        self.config_file_path = config_file_path
        self.causal_config = get_config_section(self.config_section_name)
        self._validate_builder_config()

        self.min_adjacency_confidence = coerce_float(
            self.causal_config.get("min_adjacency_confidence", 0.7),
            field_name="min_adjacency_confidence",
            minimum=0.0,
            maximum=1.0,
        )
        self.max_parents = coerce_positive_int(
            self.causal_config.get("max_parents", 3),
            field_name="max_parents",
        )
        self.significance_level = coerce_float(
            self.causal_config.get("significance_level", 0.05),
            field_name="significance_level",
            minimum=1e-8,
            maximum=0.5,
        )
        self.forbidden_edges = self._normalize_edge_constraints(
            self.causal_config.get("forbidden_edges", [])
        )
        self.required_edges = self._normalize_edge_constraints(
            self.causal_config.get("required_edges", [])
        )
        self.structure_learning_method = ensure_non_empty_string(
            self.causal_config.get("structure_learning_method", "pc"),
            "structure_learning_method",
            error_cls=ConfigurationError,
        ).lower()
        self.latent_confounder_detection = coerce_bool(
            self.causal_config.get("latent_confounder_detection", True),
            field_name="latent_confounder_detection",
        )
        self.tetrad_path = str(self.causal_config.get("tetrad_path", "") or "")
        self.fci_max_conditioning_set = coerce_positive_int(
            self.causal_config.get("fci_max_conditioning_set", 5),
            field_name="fci_max_conditioning_set",
        )
        self.use_inverse_covariance = coerce_bool(
            self.causal_config.get("use_inverse_covariance", False),
            field_name="use_inverse_covariance",
        )
        self.optimize_structure = coerce_bool(
            self.causal_config.get("optimize_structure", True),
            field_name="optimize_structure",
        )
        self.max_structure_iterations = coerce_positive_int(
            self.causal_config.get("max_structure_iterations", 100),
            field_name="max_structure_iterations",
        )
        self.ci_test_min_samples = coerce_positive_int(
            self.causal_config.get("ci_test_min_samples", 10),
            field_name="ci_test_min_samples",
        )
        self.enable_memory_logging = coerce_bool(
            self.causal_config.get("enable_memory_logging", True),
            field_name="enable_memory_logging",
        )
        self.allow_tetrad_fallback = coerce_bool(
            self.causal_config.get("allow_tetrad_fallback", True),
            field_name="allow_tetrad_fallback",
        )
        self.sensitive_attributes_as_roots = coerce_bool(
            self.causal_config.get("sensitive_attributes_as_roots", True),
            field_name="sensitive_attributes_as_roots",
        )
        self.random_seed = coerce_int(
            self.causal_config.get("random_seed", 42),
            field_name="random_seed",
        )
        self.graph_export_format = ensure_non_empty_string(
            self.causal_config.get("graph_export_format", "gml"),
            "graph_export_format",
            error_cls=ConfigurationError,
        ).lower()

        self.alignment_memory = alignment_memory or AlignmentMemory()
        self.random_state = np.random.default_rng(self.random_seed)

        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes: List[str] = []
        self.separating_sets: Dict[Tuple[str, str], Set[str]] = {}
        self.pag: Optional[nx.DiGraph] = None
        self.tetrad_result: Optional[Dict[str, Any]] = None
        self.potential_latent_confounders: Set[Tuple[str, str]] = set()

        if config_file_path:
            logger.debug(
                "CausalGraphBuilder received config_file_path=%s but retained global config loader handling.",
                config_file_path,
            )

        logger.info(
            "CausalGraphBuilder initialized | method=%s max_parents=%s significance_level=%.4f",
            self.structure_learning_method,
            self.max_parents,
            self.significance_level,
        )

    # ------------------------------------------------------------------
    # Configuration and validation
    # ------------------------------------------------------------------
    def _validate_builder_config(self) -> None:
        try:
            ensure_mapping(
                self.causal_config,
                self.config_section_name,
                allow_empty=False,
                error_cls=ConfigurationError,
            )
            for required_key in self.DEFAULT_REQUIRED_CONFIG_KEYS:
                if required_key not in self.causal_config:
                    raise ConfigurationError(
                        f"Missing required causal model configuration key: '{required_key}'.",
                        context={
                            "config_section": self.config_section_name,
                            "config_path": self.config.get("__config_path__"),
                        },
                    )
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=ConfigurationError,
                message="CausalGraphBuilder configuration validation failed.",
                context={
                    "config_section": self.config_section_name,
                    "config_path": self.config.get("__config_path__"),
                },
            ) from exc

    def _normalize_edge_constraints(
        self,
        edges: Optional[Sequence[Any]],
    ) -> List[Tuple[str, str]]:
        if edges is None:
            return []
        normalized: List[Tuple[str, str]] = []
        for edge in ensure_sequence(edges, "edges", allow_empty=True, error_cls=ConfigurationError):
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise ConfigurationError(
                    "Edge constraints must be two-item sequences [source, target].",
                    context={"edge": edge},
                )
            src = ensure_non_empty_string(edge[0], "edge_source", error_cls=ConfigurationError)
            dst = ensure_non_empty_string(edge[1], "edge_target", error_cls=ConfigurationError)
            normalized.append((src, dst))
        return normalized

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def construct_graph(
        self,
        data: pd.DataFrame,
        sensitive_attrs: Optional[Sequence[str]] = None,
    ) -> "CausalModel":
        """
        Builds domain-constrained causal DAG with:
        1. Feature causal ordering
        2. Confounder identification
        3. Sensitive attribute positioning
        """
        try:
            if not isinstance(data, pd.DataFrame):
                raise TypeMismatchError(
                    "Causal graph construction requires a pandas DataFrame.",
                    context={"actual_type": type(data).__name__},
                )
            if data.empty:
                raise DataValidationError("Input data for causal graph construction must not be empty.")

            sensitive_attributes = normalize_sensitive_attributes(
                sensitive_attrs or [],
                lowercase=False,
                allow_empty=True,
            )
            if sensitive_attributes:
                ensure_columns_present(
                    data,
                    sensitive_attributes,
                    field_name="data",
                    error_cls=DataValidationError,
                )

            learning_data = self._prepare_structure_learning_data(data)
            self.nodes = [str(column) for column in learning_data.columns]
            self.separating_sets = {}
            self.potential_latent_confounders = set()

            method = self.structure_learning_method
            if method == "pc":
                self._build_pc_graph(learning_data)
            elif method == "fci":
                self._run_fci_algorithm(learning_data)
            elif method == "tetrad":
                if not self.tetrad_path:
                    raise ExternalDependencyError(
                        "Tetrad structure learning requested but 'tetrad_path' is not configured.",
                        context={"structure_learning_method": method},
                    )
                self.graph = self._run_tetrad_fci(learning_data)
                if self.graph.number_of_nodes() == 0 and self.allow_tetrad_fallback:
                    logger.warning("Tetrad returned an empty graph. Falling back to PC structure learning.")
                    self._build_pc_graph(learning_data)
            else:
                raise ConfigurationError(
                    "Unsupported structure_learning_method configured for causal model.",
                    context={"structure_learning_method": method},
                )

            if self.sensitive_attributes_as_roots and sensitive_attributes:
                self._enforce_sensitive_attribute_rooting(sensitive_attributes)

            self._enforce_graph_constraints()

            if self.optimize_structure and self.graph.number_of_nodes() > 0:
                self._optimize_structure(learning_data)

            if not nx.is_directed_acyclic_graph(self.graph):
                self.graph = self._break_cycles(self.graph, learning_data)

            if not nx.is_directed_acyclic_graph(self.graph):
                raise CausalModelError(
                    "Final causal graph contains cycles after structure learning and constraint enforcement.",
                    context={"edges": list(self.graph.edges())},
                )

            if self.latent_confounder_detection:
                self._detect_confounders(learning_data, list(sensitive_attributes))

            # Store latent confounders in graph's metadata dictionary
            self.graph.graph["potential_latent_confounders"] = set(self.potential_latent_confounders)

            self._log_memory_metric(
                metric="causal_graph_edges",
                value=float(self.graph.number_of_edges()),
                threshold=float(max(1, len(self.nodes) * self.max_parents)),
                context={
                    "structure_learning_method": method,
                    "n_nodes": len(self.nodes),
                    "latent_confounders": len(self.potential_latent_confounders),
                },
                source="causal_graph_builder",
            )

            return CausalModel(
                graph=self.graph.copy(),
                data=data.copy(),
                config_section_name=self.config_section_name,
                config_file_path=self.config_file_path,
                alignment_memory=self.alignment_memory,
            )
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=CausalModelError,
                message="Failed to construct causal graph.",
                context={
                    "structure_learning_method": getattr(self, "structure_learning_method", None),
                    "sensitive_attrs": list(sensitive_attrs or []),
                },
            ) from exc

    def save_graph(self, path: Union[str, Path], *, include_metadata: bool = True) -> Path:
        """Persist the learned graph to disk using the configured export format."""
        try:
            target_path = Path(path)
            if self.graph_export_format == "gml":
                export_graph = self.graph.copy()
                if include_metadata:
                    # Retrieve confounders from graph metadata (or fallback to instance attribute)
                    confounders = self.graph.graph.get("potential_latent_confounders", set())
                    if isinstance(confounders, set):
                        # Convert each tuple to a string "a->b"
                        export_graph.graph["potential_latent_confounders"] = [
                            f"{a}->{b}" for a, b in confounders
                        ]
                    elif isinstance(confounders, list):
                        export_graph.graph["potential_latent_confounders"] = [
                            f"{item[0]}->{item[1]}" if isinstance(item, tuple) else str(item)
                            for item in confounders
                        ]
                    else:
                        export_graph.graph["potential_latent_confounders"] = str(confounders)
                nx.write_gml(export_graph, target_path)
            else:
                raise ConfigurationError(
                    "Unsupported graph_export_format configured for causal graph export.",
                    context={"graph_export_format": self.graph_export_format},
                )
            return target_path
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=CausalModelError,
                message="Failed to save causal graph.",
                context={"path": str(path)},
            ) from exc

    # ------------------------------------------------------------------
    # Core structure learning
    # ------------------------------------------------------------------
    def _prepare_structure_learning_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert heterogeneous alignment-domain features into a numeric view for
        structure learning while preserving column identity.
        """
        prepared = pd.DataFrame(index=data.index)
        for column in data.columns:
            series = data[column]
            if pd.api.types.is_bool_dtype(series):
                prepared[column] = series.astype(int)
            elif pd.api.types.is_numeric_dtype(series):
                prepared[column] = pd.to_numeric(series, errors="coerce")
            else:
                codes, _ = pd.factorize(series.astype(str), sort=True)
                prepared[column] = codes.astype(float)

        prepared = prepared.replace([np.inf, -np.inf], np.nan)
        prepared = prepared.dropna(axis=0, how="any")
        if prepared.empty:
            raise DataValidationError(
                "Structure learning data became empty after numeric preparation and NaN removal.",
                context={"original_rows": len(data)},
            )

        low_variance = [
            column for column in prepared.columns
            if prepared[column].nunique(dropna=True) <= 1 or float(prepared[column].var(ddof=0)) < 1e-12
        ]
        if low_variance:
            prepared = prepared.drop(columns=low_variance)
        if prepared.empty:
            raise DataValidationError(
                "All columns were removed during structure learning preparation due to low variance.",
                context={"dropped_columns": low_variance},
            )
        return prepared

    def _build_pc_graph(self, data: pd.DataFrame) -> None:
        # Temporary undirected graph; type checker expects DiGraph, we ignore this assignment
        self.graph = nx.Graph()  # type: ignore[assignment]
        self.graph.add_nodes_from(data.columns)
        self._run_pc_algorithm_skeleton(data)
        self._run_pc_algorithm_orientation(data)
        self.graph = self._orient_remaining_edges(self.graph)  # type: ignore[arg-type]

    def _estimate_inverse_covariance(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        try:
            model = GraphicalLasso(alpha=0.01, max_iter=500)   # increased from 200
            model.fit(data.values)
            return model.precision_
        except Exception as exc:
            logger.warning("Inverse covariance estimation failed: %s", exc)
            return None

    def _partial_correlation(
        self,
        data: pd.DataFrame,
        i: str,
        j: str,
        conditioning_set: Set[str],
    ) -> Tuple[float, float]:
        """
        Calculates partial correlation between variables i and j conditioned on
        the conditioning_set using regression residualisation, optionally backed
        by sparse inverse covariance when configured and feasible.
        """
        if i not in data.columns or j not in data.columns:
            raise DataValidationError(
                "Variables referenced in partial correlation were not found in data.",
                context={"i": i, "j": j},
            )

        # Case 1: empty conditioning set -> simple Pearson correlation
        if not conditioning_set:
            corr_val, p_val = pearsonr(data[i], data[j])
            return float(corr_val), float(p_val)

        # Case 2: use inverse covariance if requested and feasible
        if self.use_inverse_covariance and len(data) > max(50, len(conditioning_set) + 10):
            precision_matrix = self._estimate_inverse_covariance(data)
            if precision_matrix is not None:
                idx_i = data.columns.get_loc(i)
                idx_j = data.columns.get_loc(j)
                numerator = -precision_matrix[idx_i, idx_j]
                denominator = math.sqrt(
                    max(precision_matrix[idx_i, idx_i], 1e-12) *
                    max(precision_matrix[idx_j, idx_j], 1e-12)
                )
                partial_corr = float(np.clip(numerator / denominator, -0.999999, 0.999999))
                z_transform = 0.5 * math.log((1 + partial_corr) / (1 - partial_corr))
                se = 1.0 / math.sqrt(max(len(data) - len(conditioning_set) - 3, 1))
                z_score = abs(z_transform / se)
                p_val = 2 * (1 - norm.cdf(z_score))
                return partial_corr, float(p_val)

        # Case 3: fallback to regression residualisation
        cond_columns = [column for column in conditioning_set if column in data.columns]
        if not cond_columns:
            corr_val, p_val = pearsonr(data[i], data[j])
            return float(corr_val), float(p_val)

        x_model = OLS(data[i], add_constant(data[cond_columns], has_constant="add")).fit()
        y_model = OLS(data[j], add_constant(data[cond_columns], has_constant="add")).fit()
        corr_val, p_val = pearsonr(x_model.resid, y_model.resid)
        return float(corr_val), float(p_val)

    def _fisher_z_test(
        self,
        data: pd.DataFrame,
        i: str,
        j: str,
        conditioning_set: Set[str],
    ) -> bool:
        """
        Performs the Fisher-Z test for conditional independence.
        Returns True when independence is accepted and False otherwise.
        """
        n = len(data)
        k = len(conditioning_set)
        if n < max(self.ci_test_min_samples, k + 4):
            return False

        partial_corr, _ = self._partial_correlation(data, i, j, conditioning_set)
        if abs(partial_corr) >= 1.0:
            return False

        z_transform = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))
        test_statistic = abs(z_transform * np.sqrt(max(n - k - 3, 1)))
        critical_value = norm.ppf(1 - self.significance_level / 2)
        is_independent = bool(test_statistic < critical_value)
        if is_independent:
            self.separating_sets[(i, j)] = set(conditioning_set)
            self.separating_sets[(j, i)] = set(conditioning_set)
        return is_independent

    def _run_pc_algorithm_skeleton(self, data: pd.DataFrame) -> None:
        node_pairs = list(itertools.combinations(list(self.graph.nodes()), 2))
        for i, j in node_pairs:
            if not self.graph.has_edge(i, j):
                self.graph.add_edge(i, j)

        l = 0
        while True:
            edges_removed_in_iteration = False
            current_edges = list(self.graph.edges())
            for i, j in current_edges:
                neighbors_i = set(self.graph.neighbors(i)) - {j}
                if len(neighbors_i) < l:
                    continue
                for conditioning_tuple in itertools.combinations(neighbors_i, l):
                    conditioning_set = set(conditioning_tuple)
                    if self._fisher_z_test(data, i, j, conditioning_set):
                        if self.graph.has_edge(i, j):
                            self.graph.remove_edge(i, j)
                            edges_removed_in_iteration = True
                        break
            l += 1
            if not edges_removed_in_iteration or l > len(self.graph.nodes()) - 2:
                break

    def _run_pc_algorithm_orientation(self, data: pd.DataFrame) -> None:
        oriented_graph = nx.DiGraph()
        oriented_graph.add_nodes_from(self.graph.nodes())

        for k in self.graph.nodes():
            neighbors = list(self.graph.neighbors(k))
            if len(neighbors) < 2:
                continue
            for i, j in itertools.combinations(neighbors, 2):
                if self.graph.has_edge(i, j):
                    continue
                sep_set = self.separating_sets.get((i, j), set())
                if k not in sep_set:
                    if not oriented_graph.has_edge(k, i):
                        oriented_graph.add_edge(i, k)
                    if not oriented_graph.has_edge(k, j):
                        oriented_graph.add_edge(j, k)

        undirected_edges = {
            tuple(sorted((u, v)))
            for u, v in self.graph.edges()
            if not oriented_graph.has_edge(u, v) and not oriented_graph.has_edge(v, u)
        }

        changed = True
        while changed:
            changed = False
            for u, v in list(undirected_edges):
                if self._apply_orientation_rules(oriented_graph, u, v):
                    undirected_edges.discard(tuple(sorted((u, v))))
                    changed = True

        for u, v in undirected_edges:
            if not oriented_graph.has_edge(u, v) and not oriented_graph.has_edge(v, u):
                oriented_graph.add_edge(u, v)

        self.graph = oriented_graph

    def _apply_orientation_rules(self, graph: nx.DiGraph, u: str, v: str) -> bool:
        for predecessor in graph.predecessors(u):
            if predecessor == v:
                continue
            if not self.graph.has_edge(predecessor, v):
                graph.add_edge(u, v)
                return True

        for predecessor in graph.predecessors(v):
            if predecessor == u:
                continue
            if not self.graph.has_edge(predecessor, u):
                graph.add_edge(v, u)
                return True

        for intermediate in graph.successors(u):
            if graph.has_edge(intermediate, v):
                graph.add_edge(u, v)
                return True

        for intermediate in graph.successors(v):
            if graph.has_edge(intermediate, u):
                graph.add_edge(v, u)
                return True

        return False

    def _orient_remaining_edges(self, graph: Union[nx.Graph, nx.DiGraph]) -> nx.DiGraph:
        final_graph = nx.DiGraph()
        final_graph.add_nodes_from(graph.nodes())

        if isinstance(graph, nx.Graph) and not isinstance(graph, nx.DiGraph):
            edges = list(graph.edges())
        else:
            edges = list(graph.to_undirected().edges())

        for u, v in edges:
            if final_graph.has_edge(u, v) or final_graph.has_edge(v, u):
                continue
            final_graph.add_edge(u, v)
            if not nx.is_directed_acyclic_graph(final_graph):
                final_graph.remove_edge(u, v)
                final_graph.add_edge(v, u)
                if not nx.is_directed_acyclic_graph(final_graph):
                    final_graph.remove_edge(v, u)
        return final_graph

    def _enforce_sensitive_attribute_rooting(self, sensitive_attrs: Sequence[str]) -> None:
        if not isinstance(self.graph, nx.DiGraph):
            return
        for sensitive_attr in sensitive_attrs:
            if sensitive_attr not in self.graph:
                continue
            for predecessor in list(self.graph.predecessors(sensitive_attr)):
                self.graph.remove_edge(predecessor, sensitive_attr)

    def _enforce_graph_constraints(self) -> None:
        if not isinstance(self.graph, nx.DiGraph):
            converted_graph = nx.DiGraph()
            converted_graph.add_nodes_from(self.graph.nodes())
            converted_graph.add_edges_from(self.graph.edges())
            self.graph = converted_graph

        for source, target in self.forbidden_edges:
            if self.graph.has_edge(source, target):
                self.graph.remove_edge(source, target)

        for source, target in self.required_edges:
            if source not in self.graph:
                self.graph.add_node(source)
            if target not in self.graph:
                self.graph.add_node(target)
            self.graph.add_edge(source, target)
            if not nx.is_directed_acyclic_graph(self.graph):
                self.graph.remove_edge(source, target)
                raise CausalModelError(
                    "Adding a required edge would violate DAG constraints.",
                    context={"required_edge": (source, target)},
                )

        for node in list(self.graph.nodes()):
            parents = list(self.graph.predecessors(node))
            if len(parents) <= self.max_parents:
                continue
            parents_to_remove = parents[self.max_parents:]
            for parent in parents_to_remove:
                self.graph.remove_edge(parent, node)

    def _calculate_bic(self, data: pd.DataFrame, graph: nx.DiGraph) -> float:
        n_obs = len(data)
        if n_obs == 0:
            return float("-inf")

        total_bic = 0.0
        total_params = 0

        for node in graph.nodes():
            parents = list(graph.predecessors(node))
            y = data[node]
            if not parents:
                variance = max(float(y.var(ddof=0)), 1e-12)
                log_likelihood = -n_obs / 2 * np.log(2 * np.pi * variance) - n_obs / 2
                total_params += 2
            else:
                X = add_constant(data[parents], has_constant="add")
                model = OLS(y, X).fit()
                rss = float(np.sum(model.resid ** 2))
                variance = max(rss / n_obs, 1e-12)
                log_likelihood = -n_obs / 2 * np.log(variance) - n_obs / 2 * np.log(2 * np.pi) - n_obs / 2
                total_params += len(parents) + 2
            total_bic += log_likelihood

        return float(total_bic - 0.5 * np.log(max(n_obs, 1)) * total_params)

    def _optimize_structure(self, data: pd.DataFrame) -> None:
        if not nx.is_directed_acyclic_graph(self.graph):
            return

        current_score = self._calculate_bic(data, self.graph)
        nodes = list(self.graph.nodes())

        for _ in range(self.max_structure_iterations):
            best_graph = None
            best_score = current_score

            for source, target in itertools.permutations(nodes, 2):
                neighbor_graph = self.graph.copy()

                if not neighbor_graph.has_edge(source, target) and not neighbor_graph.has_edge(target, source):
                    neighbor_graph.add_edge(source, target)
                    if nx.is_directed_acyclic_graph(neighbor_graph):
                        score = self._calculate_bic(data, neighbor_graph)
                        if score > best_score:
                            best_graph = neighbor_graph.copy()
                            best_score = score

                if self.graph.has_edge(source, target):
                    neighbor_graph = self.graph.copy()
                    neighbor_graph.remove_edge(source, target)
                    score = self._calculate_bic(data, neighbor_graph)
                    if score > best_score:
                        best_graph = neighbor_graph.copy()
                        best_score = score

                    neighbor_graph = self.graph.copy()
                    neighbor_graph.remove_edge(source, target)
                    neighbor_graph.add_edge(target, source)
                    if nx.is_directed_acyclic_graph(neighbor_graph):
                        score = self._calculate_bic(data, neighbor_graph)
                        if score > best_score:
                            best_graph = neighbor_graph.copy()
                            best_score = score

            if best_graph is None:
                break
            self.graph = best_graph
            current_score = best_score

    def _break_cycles(self, graph: nx.DiGraph, data: pd.DataFrame) -> nx.DiGraph:
        candidate = graph.copy()
        while not nx.is_directed_acyclic_graph(candidate):
            cycle = nx.find_cycle(candidate, orientation="original")
            if not cycle:
                break

            best_edge_to_remove = None
            best_score = float("-inf")
            for source, target, _ in cycle:
                trial = candidate.copy()
                trial.remove_edge(source, target)
                if not nx.is_directed_acyclic_graph(trial):
                    continue
                score = self._calculate_bic(data, trial)
                if score > best_score:
                    best_score = score
                    best_edge_to_remove = (source, target)

            if best_edge_to_remove is None:
                source, target, _ = cycle[0]
                candidate.remove_edge(source, target)
            else:
                candidate.remove_edge(*best_edge_to_remove)
        return candidate

    # ------------------------------------------------------------------
    # Latent confounders and Tetrad / FCI
    # ------------------------------------------------------------------
    def _run_fci_algorithm(self, data: pd.DataFrame) -> None:
        # Temporary undirected graph; type checker expects DiGraph, we ignore this assignment
        self.graph = nx.Graph()  # type: ignore[assignment]
        self.graph.add_nodes_from(data.columns)
        self._run_fci_skeleton(data)
        self._run_fci_orientation()

    def _run_fci_skeleton(self, data: pd.DataFrame) -> None:
        self.graph = nx.complete_graph(list(data.columns), create_using=nx.Graph())
        l = 0
        while l <= self.fci_max_conditioning_set:
            edges_removed = False
            current_edges = list(self.graph.edges())
            for i, j in current_edges:
                neighbors_i = set(self.graph.neighbors(i)) - {j}
                neighbors_j = set(self.graph.neighbors(j)) - {i}
                candidate_sets: Set[Tuple[str, ...]] = set()
                for base_neighbors in (neighbors_i, neighbors_j):
                    if len(base_neighbors) < l:
                        continue
                    for conditioning_tuple in itertools.combinations(base_neighbors, l):
                        candidate_sets.add(tuple(conditioning_tuple))

                for conditioning_tuple in candidate_sets:
                    conditioning_set = set(conditioning_tuple)
                    if self._fisher_z_test(data, i, j, conditioning_set):
                        if self.graph.has_edge(i, j):
                            self.graph.remove_edge(i, j)
                            edges_removed = True
                        break
            if not edges_removed:
                break
            l += 1

    def _run_fci_orientation(self) -> None:
        pag = nx.DiGraph()
        pag.add_nodes_from(self.graph.nodes())

        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) < 2:
                continue
            for left, right in itertools.combinations(neighbors, 2):
                if self.graph.has_edge(left, right):
                    continue
                if node not in self.separating_sets.get((left, right), set()):
                    pag.add_edge(left, node)
                    pag.add_edge(right, node)

        self.pag = pag
        self.graph = self._orient_remaining_edges(pag)

    def _run_tetrad_fci(self, data: pd.DataFrame) -> nx.DiGraph:
        tetrad_jar_path = Path(self.tetrad_path)
        if not tetrad_jar_path.is_file():
            raise ExternalDependencyError(
                "Configured Tetrad JAR path does not exist.",
                context={"tetrad_path": str(tetrad_jar_path)},
            )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
            data.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name

        try:
            command = [
                "java",
                "-jar",
                str(tetrad_jar_path),
                "--algorithm",
                "fci",
                "--data",
                tmp_path,
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
            return self._parse_tetrad_output(result.stdout, data.columns.tolist())
        except subprocess.CalledProcessError as exc:
            raise ExternalDependencyError(
                "Tetrad FCI execution failed.",
                context={"stderr": exc.stderr, "stdout": exc.stdout},
                cause=exc,
            ) from exc
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _parse_tetrad_output(self, output: str, nodes: Sequence[str]) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line or "->" not in line:
                continue
            parts = [part.strip() for part in line.split("->")]
            if len(parts) != 2:
                continue
            source, target = parts
            if source in graph and target in graph:
                graph.add_edge(source, target)
        return graph

    def _detect_confounders(self, data: pd.DataFrame, sensitive_attrs: Sequence[str]) -> None:
        if self.pag is not None:
            self._analyze_pag(sensitive_attrs)

        ancestors_by_sensitive_attr: Dict[str, Set[str]] = {}
        for sensitive_attr in sensitive_attrs:
            if sensitive_attr in self.graph:
                ancestors_by_sensitive_attr[sensitive_attr] = set(nx.ancestors(self.graph, sensitive_attr))

        common_ancestors = set.intersection(*ancestors_by_sensitive_attr.values()) if ancestors_by_sensitive_attr else set()
        for candidate in common_ancestors:
            for sensitive_attr in sensitive_attrs:
                # Create a sorted tuple to avoid duplicate (a,b) vs (b,a)
                if candidate < sensitive_attr:
                    self.potential_latent_confounders.add((candidate, sensitive_attr))
                else:
                    self.potential_latent_confounders.add((sensitive_attr, candidate))

    def _analyze_pag(self, sensitive_attrs: Sequence[str]) -> None:
        if self.pag is None:
            return
        confounders: Set[Tuple[str, str]] = set()
        for u, v in self.pag.edges():
            if self.pag.has_edge(v, u):
                confounders.add(tuple(sorted((u, v))))
        self.potential_latent_confounders |= confounders
        for left, right in confounders:
            if left in sensitive_attrs or right in sensitive_attrs:
                logger.warning(
                    "Potential latent confounder detected involving a sensitive attribute: %s <-> %s",
                    left,
                    right,
                )

    # ------------------------------------------------------------------
    # Memory logging
    # ------------------------------------------------------------------
    def _log_memory_metric(
        self,
        metric: str,
        value: float,
        threshold: float,
        context: Mapping[str, Any],
        *,
        source: str,
        tags: Optional[Iterable[Any]] = None,
    ) -> None:
        if not self.enable_memory_logging:
            return
        try:
            self.alignment_memory.log_evaluation(
                metric=metric,
                value=float(value),
                threshold=float(threshold),
                context=dict(context),
                source=source,
                tags=tags,
                metadata={"module": "causal_model"},
            )
        except Exception as exc:
            logger.warning("Failed to log causal-model metric to AlignmentMemory: %s", exc)


class CausalModel:
    """
    Structural Causal Model implementing:
    - Potential outcome estimation
    - Backdoor adjustment
    - Counterfactual inference
    - Instrumental Variable estimation
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        data: pd.DataFrame,
        config_section_name: str = "causal_model",
        config_file_path: Optional[str] = None,
        alignment_memory: Optional[AlignmentMemory] = None,
    ):
        self.config = load_global_config()
        self.config_section_name = ensure_non_empty_string(
            config_section_name,
            "config_section_name",
            error_cls=ConfigurationError,
        )
        self.config_file_path = config_file_path
        self.causal_config = get_config_section(self.config_section_name)

        self._validate_model_config()

        if not isinstance(graph, nx.DiGraph):
            raise TypeMismatchError(
                "Input graph must be a networkx.DiGraph.",
                context={"actual_type": type(graph).__name__},
            )
        if data is None:
            data = pd.DataFrame()
        if not isinstance(data, pd.DataFrame):
            raise TypeMismatchError(
                "Input data for CausalModel must be a pandas DataFrame.",
                context={"actual_type": type(data).__name__},
            )

        self.graph = graph.copy()
        self.data = data.copy()
        self.alignment_memory = alignment_memory or AlignmentMemory()
        self.enable_memory_logging = coerce_bool(
            self.causal_config.get("enable_memory_logging", True),
            field_name="enable_memory_logging",
        )
        self.default_effect_method = ensure_non_empty_string(
            self.causal_config.get("default_effect_method", "backdoor.linear_regression"),
            "default_effect_method",
            error_cls=ConfigurationError,
        ).lower()
        self.propensity_clip = coerce_float(
            self.causal_config.get("propensity_clip", 1e-6),
            field_name="propensity_clip",
            minimum=1e-12,
            maximum=0.49,
        )
        self.iv_relevance_threshold = coerce_float(
            self.causal_config.get("iv_relevance_threshold", 5.0),
            field_name="iv_relevance_threshold",
            minimum=0.0,
        )
        self.counterfactual_predict_exogenous_strategy = ensure_non_empty_string(
            self.causal_config.get("counterfactual_predict_exogenous_strategy", "retain_observed"),
            "counterfactual_predict_exogenous_strategy",
            error_cls=ConfigurationError,
        ).lower()
        self.random_seed = coerce_int(
            self.causal_config.get("random_seed", 42),
            field_name="random_seed",
        )

        self.nodes = list(self.graph.nodes())
        self.graph_hash = stable_record_fingerprint(
            {"nodes": self.nodes, "edges": list(self.graph.edges())},
            namespace="causal_model_graph",
        )
        self.intervention_history: List[Dict[str, Any]] = []

        if self.nodes:
            missing_nodes = set(self.nodes) - set(self.data.columns)
            if missing_nodes:
                raise DataValidationError(
                    "Data is missing columns required by the causal graph nodes.",
                    context={"missing_nodes": sorted(missing_nodes)},
                )

        if self.graph.number_of_nodes() > 0 and not nx.is_directed_acyclic_graph(self.graph):
            raise CausalModelError("CausalModel requires a directed acyclic graph (DAG).")

        self.structural_equations: Optional[Dict[str, StructuralEquation]] = None

    # ------------------------------------------------------------------
    # Validation and serialization
    # ------------------------------------------------------------------
    def _validate_model_config(self) -> None:
        ensure_mapping(
            self.causal_config,
            self.config_section_name,
            allow_empty=False,
            error_cls=ConfigurationError,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.graph.nodes()),
            "edges": list(self.graph.edges()),
            "n_rows": int(len(self.data)),
            "graph_hash": self.graph_hash,
            "structural_equations_built": self.structural_equations is not None,
        }

    # ------------------------------------------------------------------
    # Graph logic and d-separation
    # ------------------------------------------------------------------
    def _find_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        if treatment not in self.graph or outcome not in self.graph:
            return []
        undirected_graph = self.graph.to_undirected()
        if not nx.has_path(undirected_graph, treatment, outcome):
            return []

        backdoor_paths: List[List[str]] = []
        for path in nx.all_simple_paths(undirected_graph, source=treatment, target=outcome):
            if len(path) < 2:
                continue
            second_node = path[1]
            if self.graph.has_edge(second_node, treatment):
                backdoor_paths.append(path)
        return backdoor_paths

    def _is_blocked_path(
        self,
        graph: nx.DiGraph,
        path: List[str],
        conditioning_set: Set[str],
    ) -> bool:
        if len(path) < 3:
            return False

        for idx in range(len(path) - 2):
            left, middle, right = path[idx], path[idx + 1], path[idx + 2]

            is_chain = graph.has_edge(left, middle) and graph.has_edge(middle, right)
            is_fork = graph.has_edge(middle, left) and graph.has_edge(middle, right)
            is_collider = graph.has_edge(left, middle) and graph.has_edge(right, middle)

            if (is_chain or is_fork) and middle in conditioning_set:
                return True

            if is_collider:
                descendants = nx.descendants(graph, middle) | {middle}
                if not descendants.intersection(conditioning_set):
                    return True
        return False

    def _is_blocked(self, path: List[str], conditioning_set: Set[str]) -> bool:
        return self._is_blocked_path(self.graph, path, conditioning_set)

    def _are_d_separated(
        self,
        graph: nx.DiGraph,
        node1: str,
        node2: str,
        conditioning_set: Set[str],
    ) -> bool:
        if node1 not in graph or node2 not in graph:
            return True
        undirected_graph = graph.to_undirected()
        if not nx.has_path(undirected_graph, node1, node2):
            return True
        for path in nx.all_simple_paths(undirected_graph, node1, node2):
            if not self._is_blocked_path(graph, path, conditioning_set):
                return False
        return True

    def _find_minimal_adjustment_set(self, treatment: str, outcome: str) -> Set[str]:
        backdoor_paths = self._find_backdoor_paths(treatment, outcome)
        if not backdoor_paths:
            return set()

        descendants_of_treatment = nx.descendants(self.graph, treatment)
        candidates: List[str] = []
        for path in backdoor_paths:
            for node in path[1:-1]:
                if node in descendants_of_treatment or node == treatment or node == outcome:
                    continue
                if node not in candidates:
                    candidates.append(node)

        greedy_set: Set[str] = set(parent for parent in self.graph.predecessors(treatment) if parent in candidates)
        remaining_paths = [path for path in backdoor_paths if not self._is_blocked(path, greedy_set)]

        while remaining_paths:
            best_candidate = None
            best_blocks = 0
            for candidate in candidates:
                if candidate in greedy_set:
                    continue
                blocked_count = sum(
                    1 for path in remaining_paths
                    if self._is_blocked(path, greedy_set | {candidate})
                )
                if blocked_count > best_blocks:
                    best_candidate = candidate
                    best_blocks = blocked_count
            if best_candidate is None:
                break
            greedy_set.add(best_candidate)
            remaining_paths = [path for path in backdoor_paths if not self._is_blocked(path, greedy_set)]

        return greedy_set

    def _find_frontdoor_mediators(self, treatment: str, outcome: str) -> List[str]:
        mediators: List[str] = []
        if treatment not in self.graph or outcome not in self.graph:
            return mediators
        for child in self.graph.successors(treatment):
            if child == outcome:
                continue
            if nx.has_path(self.graph, child, outcome):
                mediators.append(child)
        return mediators

    # ------------------------------------------------------------------
    # Structural equations
    # ------------------------------------------------------------------
    def _prepare_modeling_data(self, data: pd.DataFrame) -> pd.DataFrame:
        prepared = pd.DataFrame(index=data.index)
        for column in self.nodes:
            if column not in data.columns:
                continue
            series = data[column]
            if pd.api.types.is_bool_dtype(series):
                prepared[column] = series.astype(int)
            elif pd.api.types.is_numeric_dtype(series):
                prepared[column] = pd.to_numeric(series, errors="coerce")
            else:
                codes, _ = pd.factorize(series.astype(str), sort=True)
                prepared[column] = codes.astype(float)
        prepared = prepared.replace([np.inf, -np.inf], np.nan)
        return prepared

    def _get_structural_equations(self) -> Dict[str, StructuralEquation]:
        if self.structural_equations is not None:
            return self.structural_equations

        equations: Dict[str, StructuralEquation] = {}
        if self.graph.number_of_nodes() == 0 or self.data.empty:
            self.structural_equations = equations
            return equations

        modeling_data = self._prepare_modeling_data(self.data)
        for node in nx.topological_sort(self.graph):
            parents = list(self.graph.predecessors(node))
            node_series = modeling_data[node].dropna()

            if not parents:
                equations[node] = StructuralEquation(
                    node=node,
                    parents=[],
                    equation_type="exogenous",
                    mean=float(node_series.mean()) if not node_series.empty else 0.0,
                    variance=float(node_series.var(ddof=0)) if len(node_series) > 1 else 0.0,
                )
                continue

            X = modeling_data[parents]
            y = modeling_data[node]
            valid_idx = X.dropna().index.intersection(y.dropna().index)
            if len(valid_idx) < max(len(parents) + 2, 5):
                equations[node] = StructuralEquation(
                    node=node,
                    parents=parents,
                    equation_type="fallback_mean",
                    mean=float(node_series.mean()) if not node_series.empty else 0.0,
                    variance=float(node_series.var(ddof=0)) if len(node_series) > 1 else 0.0,
                )
                continue

            X_valid = X.loc[valid_idx]
            y_valid = y.loc[valid_idx]
            unique_values = sorted(set(y_valid.unique().tolist()))
            is_binary = len(unique_values) <= 2 and set(unique_values).issubset({0, 1})

            if is_binary:
                model = LogisticRegression(max_iter=2000, random_state=self.random_seed)
                model.fit(X_valid, y_valid)
                equations[node] = StructuralEquation(
                    node=node,
                    parents=parents,
                    equation_type="logistic",
                    model=model,
                    predictors=parents,
                )
            else:
                X_ols = add_constant(X_valid, has_constant="add")
                model = OLS(y_valid, X_ols).fit()
                # X_ols is a DataFrame at runtime; ignore type for .columns access
                predictors = list(X_ols.columns)  # type: ignore[attr-defined]
                equations[node] = StructuralEquation(
                    node=node,
                    parents=parents,
                    equation_type="ols",
                    model=model,
                    predictors=predictors,
                )

        self.structural_equations = equations
        return equations

    # ------------------------------------------------------------------
    # Effect estimation
    # ------------------------------------------------------------------
    def estimate_effect(
        self,
        treatment: str,
        outcome: str,
        method: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        instrument: Optional[str] = None,
        mediators: Optional[Sequence[str]] = None,
    ) -> pd.Series:
        """
        Estimate a causal effect using backdoor, frontdoor, or IV methods.
        """
        try:
            resolved_method = (method or self.default_effect_method).strip().lower()
            treatment_name = ensure_non_empty_string(treatment, "treatment", error_cls=DataValidationError)
            outcome_name = ensure_non_empty_string(outcome, "outcome", error_cls=DataValidationError)

            source_data = self.data if data is None else data.copy()
            if not isinstance(source_data, pd.DataFrame):
                raise TypeMismatchError(
                    "estimate_effect requires a pandas DataFrame as data input.",
                    context={"actual_type": type(source_data).__name__},
                )
            ensure_columns_present(
                source_data,
                [treatment_name, outcome_name],
                field_name="data",
                error_cls=DataValidationError,
            )

            if resolved_method.startswith("backdoor"):
                result = self._backdoor_adjustment(source_data, treatment_name, outcome_name, resolved_method)
            elif resolved_method.startswith("frontdoor"):
                result = self._frontdoor_adjustment(
                    source_data,
                    treatment_name,
                    outcome_name,
                    mediators=list(mediators or []),
                    method=resolved_method,
                )
            elif resolved_method.startswith("iv"):
                instrument_name = ensure_non_empty_string(
                    instrument,
                    "instrument",
                    error_cls=MissingFieldError,
                )
                result = self._instrumental_variables(
                    source_data,
                    treatment_name,
                    outcome_name,
                    instrument_name,
                    estimator=resolved_method.split(".")[-1] if "." in resolved_method else "2sls",
                )
            else:
                raise ConfigurationError(
                    "Unsupported causal effect estimation method.",
                    context={"method": resolved_method},
                )

            self._log_effect_estimate(result)
            return result.to_series() if isinstance(result, CausalEffectEstimate) else result
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=CausalModelError,
                message="Failed to estimate causal effect.",
                context={"treatment": treatment, "outcome": outcome, "method": method},
            ) from exc

    def _linear_adjustment(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: Set[str],
    ) -> CausalEffectEstimate:
        modeling_data = self._prepare_modeling_data(data)
        columns = [treatment, outcome, *sorted(adjustment_set)]
        ensure_columns_present(modeling_data, columns, field_name="modeling_data", error_cls=DataValidationError)

        valid_data = modeling_data[columns].dropna()
        if valid_data.empty:
            raise DataValidationError(
                "No valid observations remained after dropping NaNs for linear adjustment.",
                context={"columns": columns},
            )

        X = valid_data[[treatment, *sorted(adjustment_set)]]
        X = add_constant(X, has_constant="add")
        y = valid_data[outcome]
        model = OLS(y, X).fit()

        ci = model.conf_int().loc[treatment]
        # X is a DataFrame; ignore type for .columns access
        formula_terms = list(X.columns)  # type: ignore[attr-defined]
        return CausalEffectEstimate(
            treatment=treatment,
            outcome=outcome,
            method="backdoor.linear_regression",
            effect=float(model.params[treatment]),
            std_error=float(model.bse[treatment]),
            p_value=float(model.pvalues[treatment]),
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            n_obs=int(model.nobs),
            adjustment_set=sorted(adjustment_set),
            metadata={"formula_terms": formula_terms},
        )

    def _doubly_robust_estimation(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: Set[str],
    ) -> CausalEffectEstimate:
        modeling_data = self._prepare_modeling_data(data)
        columns = [treatment, outcome, *sorted(adjustment_set)]
        valid_data = modeling_data[columns].dropna()
        if valid_data.empty:
            raise DataValidationError(
                "No valid observations remained after dropping NaNs for doubly robust estimation.",
                context={"columns": columns},
            )

        T = valid_data[treatment]
        Y = valid_data[outcome]
        if not T.isin([0, 1]).all():
            raise DataValidationError(
                "Doubly robust estimation currently requires a binary treatment encoded as 0/1.",
                context={"treatment": treatment},
            )

        Z = valid_data[list(sorted(adjustment_set))] if adjustment_set else pd.DataFrame(index=valid_data.index)
        ps_model = LogisticRegression(solver="liblinear", C=1e6, random_state=self.random_seed)
        ps_model.fit(Z if not Z.empty else np.ones((len(valid_data), 1)), T)
        propensity_scores = ps_model.predict_proba(Z if not Z.empty else np.ones((len(valid_data), 1)))[:, 1]
        propensity_scores = np.clip(propensity_scores, self.propensity_clip, 1.0 - self.propensity_clip)

        XZ = Z.copy()
        XZ[treatment] = T
        XZ1 = Z.copy()
        XZ1[treatment] = 1
        XZ0 = Z.copy()
        XZ0[treatment] = 0

        outcome_model = GradientBoostingRegressor(random_state=self.random_seed)
        outcome_model.fit(XZ, Y)

        mu1_hat = outcome_model.predict(XZ1)
        mu0_hat = outcome_model.predict(XZ0)

        dr_y1 = (T * Y / propensity_scores) + (1.0 - T / propensity_scores) * mu1_hat
        dr_y0 = ((1.0 - T) * Y / (1.0 - propensity_scores)) + (1.0 - (1.0 - T) / (1.0 - propensity_scores)) * mu0_hat

        ate = float(np.mean(dr_y1 - dr_y0))
        influence = (dr_y1 - dr_y0) - ate
        variance = float(np.mean(influence ** 2) / max(len(valid_data), 1))
        std_error = math.sqrt(max(variance, 0.0))
        z_score = ate / std_error if std_error > 0 else np.inf
        p_value = 2 * (1 - norm.cdf(abs(z_score))) if np.isfinite(z_score) else 0.0
        ci_lower = ate - 1.96 * std_error
        ci_upper = ate + 1.96 * std_error

        return CausalEffectEstimate(
            treatment=treatment,
            outcome=outcome,
            method="backdoor.doubly_robust",
            effect=ate,
            std_error=std_error,
            p_value=float(p_value),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            n_obs=int(len(valid_data)),
            adjustment_set=sorted(adjustment_set),
            metadata={"propensity_clip": self.propensity_clip},
        )

    def _backdoor_adjustment(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        method: str,
    ) -> CausalEffectEstimate:
        adjustment_set = self._find_minimal_adjustment_set(treatment, outcome)
        if "linear_regression" in method:
            return self._linear_adjustment(data, treatment, outcome, adjustment_set)
        if "doubly_robust" in method:
            return self._doubly_robust_estimation(data, treatment, outcome, adjustment_set)
        raise ConfigurationError(
            "Unsupported backdoor adjustment estimator configured.",
            context={"method": method},
        )

    def _frontdoor_adjustment(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        mediators: List[str],
        method: str,
    ) -> CausalEffectEstimate:
        mediator_set = mediators or self._find_frontdoor_mediators(treatment, outcome)
        if not mediator_set:
            raise CausalModelError(
                "Frontdoor adjustment requires at least one mediator.",
                context={"treatment": treatment, "outcome": outcome},
            )

        modeling_data = self._prepare_modeling_data(data)
        ensure_columns_present(
            modeling_data,
            [treatment, outcome, *mediator_set],
            field_name="modeling_data",
            error_cls=DataValidationError,
        )
        valid_data = modeling_data[[treatment, outcome, *mediator_set]].dropna()
        if valid_data.empty:
            raise DataValidationError(
                "No valid observations remained after dropping NaNs for frontdoor adjustment.",
                context={"mediators": mediator_set},
            )

        mediator_effects: List[float] = []
        for mediator in mediator_set:
            mediator_model = OLS(valid_data[mediator], add_constant(valid_data[[treatment]], has_constant="add")).fit()
            outcome_model = OLS(
                valid_data[outcome],
                add_constant(valid_data[[mediator, treatment]], has_constant="add"),
            ).fit()
            mediator_effects.append(float(mediator_model.params[treatment] * outcome_model.params[mediator]))

        total_effect = float(np.sum(mediator_effects))
        return CausalEffectEstimate(
            treatment=treatment,
            outcome=outcome,
            method=method,
            effect=total_effect,
            std_error=None,
            p_value=None,
            ci_lower=None,
            ci_upper=None,
            n_obs=int(len(valid_data)),
            mediator_set=mediator_set,
            metadata={"frontdoor_mode": "product_of_coefficients"},
        )

    def _check_iv_conditions(self, instrument: str, treatment: str, outcome: str) -> Dict[str, bool]:
        conditions = {
            "relevance_path": False,
            "exclusion": False,
            "independence": False,
        }
        if instrument not in self.graph or treatment not in self.graph or outcome not in self.graph:
            return conditions

        if nx.has_path(self.graph.to_undirected(), instrument, treatment):
            conditions["relevance_path"] = True

        modified_graph = self.graph.copy()
        for child in list(modified_graph.successors(treatment)):
            modified_graph.remove_edge(treatment, child)

        conditions["exclusion"] = not nx.has_path(modified_graph, instrument, outcome)
        conditions["independence"] = self._are_d_separated(modified_graph, instrument, outcome, set())
        return conditions

    def _instrumental_variables(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        instrument: str,
        estimator: str = "2sls",
    ) -> CausalEffectEstimate:
        if estimator != "2sls":
            raise ConfigurationError(
                "Unsupported IV estimator configured for causal model.",
                context={"estimator": estimator},
            )

        modeling_data = self._prepare_modeling_data(data)
        ensure_columns_present(
            modeling_data,
            [treatment, outcome, instrument],
            field_name="modeling_data",
            error_cls=DataValidationError,
        )
        valid_data = modeling_data[[treatment, outcome, instrument]].dropna()
        if valid_data.empty:
            raise DataValidationError(
                "No valid observations remained after dropping NaNs for IV estimation.",
                context={"instrument": instrument},
            )

        iv_conditions = self._check_iv_conditions(instrument, treatment, outcome)
        weak_iv_warning = not all(iv_conditions.values())

        endog = valid_data[outcome]
        exog = add_constant(valid_data[[treatment]], has_constant="add")
        instruments = add_constant(valid_data[[instrument]], has_constant="add")
        iv_model = IV2SLS(endog=endog, exog=exog, instrument=instruments).fit()

        ci = iv_model.conf_int().loc[treatment]
        estimate = CausalEffectEstimate(
            treatment=treatment,
            outcome=outcome,
            method="iv.2sls",
            effect=float(iv_model.params[treatment]),
            std_error=float(iv_model.bse[treatment]),
            p_value=float(iv_model.pvalues[treatment]),
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            n_obs=int(iv_model.nobs),
            instrument=instrument,
            metadata={
                "iv_conditions": iv_conditions,
                "weak_iv_warning": weak_iv_warning,
            },
        )
        return estimate

    def _log_effect_estimate(self, estimate: Union[CausalEffectEstimate, pd.Series, None]) -> None:
        if not self.enable_memory_logging or estimate is None:
            return
        try:
            payload = estimate.to_dict() if isinstance(estimate, CausalEffectEstimate) else dict(estimate)
            effect_value = float(payload.get("effect", 0.0))
            self.alignment_memory.log_evaluation(
                metric="causal_effect_estimate",
                value=abs(effect_value),
                threshold=1.0,
                context={
                    "treatment": payload.get("treatment"),
                    "outcome": payload.get("outcome"),
                    "method": payload.get("method"),
                    "instrument": payload.get("instrument"),
                },
                source="causal_model",
                metadata={"estimate": payload},
            )
        except Exception as exc:
            logger.warning("Failed to log causal effect estimate to AlignmentMemory: %s", exc)

    # ------------------------------------------------------------------
    # Counterfactual inference
    # ------------------------------------------------------------------
    def compute_counterfactual(
        self,
        intervention: Dict[str, Union[float, int, str, bool]],
        observed_data_point: Optional[pd.Series] = None,
        method: str = "adjust",
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute counterfactual outcome(s) under a specific intervention using
        structural equation adjustment or, for specific future extension,
        Pearl's three-step method.
        """
        try:
            method_name = ensure_non_empty_string(method, "method", error_cls=DataValidationError).lower()
            intervention_mapping = ensure_mapping(
                intervention,
                "intervention",
                allow_empty=False,
                error_cls=DataValidationError,
            )
            ensure_columns_present(
                self.data if observed_data_point is None else pd.DataFrame([observed_data_point]),
                list(intervention_mapping.keys()),
                field_name="intervention_scope",
                error_cls=DataValidationError,
            )

            if method_name == "pearl3step":
                raise CounterfactualAuditError(
                    "Pearl's three-step counterfactual routine is not yet implemented in this module.",
                    context={"method": method_name},
                )
            if method_name != "adjust":
                raise ConfigurationError(
                    "Unsupported counterfactual method configured for causal model.",
                    context={"method": method_name},
                )

            equations = self._get_structural_equations()
            if observed_data_point is not None:
                base_frame = observed_data_point.to_frame().T.copy()
                base_frame.index = ["counterfactual"]
            else:
                base_frame = self.data.copy()

            numeric_frame = self._prepare_modeling_data(base_frame)
            if numeric_frame.empty and base_frame.empty:
                return base_frame.iloc[0] if observed_data_point is not None else base_frame

            for variable in nx.topological_sort(self.graph):
                if variable in intervention_mapping:
                    base_frame[variable] = intervention_mapping[variable]
                    if variable in numeric_frame.columns:
                        numeric_frame[variable] = pd.to_numeric(base_frame[variable], errors="coerce")
                    continue

                equation = equations.get(variable)
                if equation is None:
                    continue
                if equation.equation_type in {"exogenous", "fallback_mean"}:
                    if self.counterfactual_predict_exogenous_strategy == "mean_fill" and variable in numeric_frame.columns:
                        fill_value = equation.mean if equation.mean is not None else 0.0
                        numeric_frame[variable] = numeric_frame[variable].fillna(fill_value)
                        if pd.api.types.is_numeric_dtype(base_frame[variable]):
                            base_frame[variable] = numeric_frame[variable]
                    continue

                parents = equation.parents
                parent_frame = numeric_frame[parents].copy()
                if equation.equation_type == "ols" and isinstance(equation.model, RegressionResultsWrapper):
                    X_pred = add_constant(parent_frame, has_constant="add")
                    X_pred = X_pred.reindex(columns=equation.predictors, fill_value=1.0)
                    predicted = equation.model.predict(X_pred)
                    numeric_frame[variable] = predicted
                    if variable in base_frame.columns and pd.api.types.is_numeric_dtype(base_frame[variable]):
                        base_frame[variable] = predicted
                elif equation.equation_type == "logistic" and equation.model is not None:
                    probabilities = equation.model.predict_proba(parent_frame)[:, 1]
                    binary_prediction = (probabilities >= 0.5).astype(int)
                    numeric_frame[variable] = binary_prediction
                    if variable in base_frame.columns and pd.api.types.is_numeric_dtype(base_frame[variable]):
                        base_frame[variable] = binary_prediction

            cf_result = base_frame.iloc[0] if observed_data_point is not None else base_frame
            self.intervention_history.append(
                build_alignment_event(
                    "counterfactual_computed",
                    source="causal_model",
                    metadata={"method": method_name},
                    context={
                        "intervention": normalize_context(intervention_mapping),
                        "observed_data_point": observed_data_point.to_dict() if observed_data_point is not None else None,
                    },
                    payload={"shape": getattr(base_frame, "shape", None)},
                )
            )
            return cf_result
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=CounterfactualAuditError,
                message="Failed to compute counterfactual outcome.",
                context={"method": method, "intervention": sanitize_for_logging(intervention)},
            ) from exc

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------
    def get_graph_summary(self) -> Dict[str, Any]:
        # Retrieve latent confounders from graph metadata (or fallback to empty set)
        latent_confounders = self.graph.graph.get("potential_latent_confounders", set())
        return {
            "nodes": list(self.graph.nodes()),
            "edges": list(self.graph.edges()),
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(self.graph) if self.graph.number_of_nodes() > 0 else True,
            "potential_latent_confounders": list(latent_confounders),
            "graph_hash": self.graph_hash,
        }

    def get_structural_equation_summary(self) -> Dict[str, Any]:
        return {
            node: equation.to_dict()
            for node, equation in self._get_structural_equations().items()
        }


if __name__ == "__main__":
    print("\n=== Running Causal Model ===\n")
    printer.status("TEST", "Causal Model initialized", "info")

    np.random.seed(42)
    n_samples = 1200

    Z = np.random.normal(0, 1, n_samples)
    U = np.random.normal(0, 1, n_samples)
    X = 0.8 * Z + 0.4 * U + np.random.normal(0, 0.5, n_samples)
    M = 0.9 * X + np.random.normal(0, 0.5, n_samples)
    Y = 1.4 * X + 0.6 * M + 0.5 * U + np.random.normal(0, 0.8, n_samples)
    S = np.where(Z > 0, "group_a", "group_b")

    data = pd.DataFrame(
        {
            "Z": Z,
            "U_proxy": U,
            "X": X,
            "M": M,
            "Y": Y,
            "sensitive_group": S,
        }
    )

    builder = CausalGraphBuilder()
    builder.required_edges = [("Z", "X"), ("X", "M"), ("M", "Y")]
    builder.forbidden_edges = [("Y", "X"), ("Y", "M")]

    learned_model = builder.construct_graph(data, sensitive_attrs=["sensitive_group"])

    printer.pretty("graph_summary", learned_model.get_graph_summary(), "success")
    printer.pretty("structural_equations", learned_model.get_structural_equation_summary(), "success")

    print("\nEstimating effect X -> Y using backdoor linear regression:")
    backdoor_linear = learned_model.estimate_effect(
        treatment="X",
        outcome="Y",
        method="backdoor.linear_regression",
    )
    printer.pretty("backdoor_linear", backdoor_linear.to_dict(), "success")

    binary_treatment = (data["Z"] > 0).astype(int)
    binary_outcome = (data["Y"] > data["Y"].median()).astype(int)
    binary_data = data.copy()
    binary_data["T_bin"] = binary_treatment
    binary_data["Y_bin"] = binary_outcome

    print("\nEstimating effect T_bin -> Y_bin using doubly robust estimation:")
    try:
        dr_estimate = learned_model.estimate_effect(
            treatment="T_bin",
            outcome="Y_bin",
            method="backdoor.doubly_robust",
            data=binary_data,
        )
        printer.pretty("doubly_robust", dr_estimate.to_dict(), "success")
    except Exception as exc:
        printer.pretty("doubly_robust_error", str(exc), "warning")

    print("\nEstimating effect X -> Y using IV (Z as instrument):")
    iv_estimate = learned_model.estimate_effect(
        treatment="X",
        outcome="Y",
        method="iv.2sls",
        instrument="Z",
    )
    printer.pretty("iv_estimate", iv_estimate.to_dict(), "success")

    print("\nEstimating effect X -> Y using frontdoor mediation:")
    frontdoor_estimate = learned_model.estimate_effect(
        treatment="X",
        outcome="Y",
        method="frontdoor.linear_regression",
        mediators=["M"],
    )
    printer.pretty("frontdoor_estimate", frontdoor_estimate.to_dict(), "success")

    print("\nComputing counterfactual do(X = 2.0) over the population:")
    cf_population = learned_model.compute_counterfactual(intervention={"X": 2.0}, method="adjust")
    printer.pretty(
        "counterfactual_population_summary",
        {
            "mean_Y": float(pd.to_numeric(cf_population["Y"], errors="coerce").mean()),
            "mean_M": float(pd.to_numeric(cf_population["M"], errors="coerce").mean()),
            "rows": int(len(cf_population)),
        },
        "success",
    )

    print("\nComputing counterfactual do(X = 2.0) for one observed unit:")
    observed_point = data.iloc[0]
    cf_individual = learned_model.compute_counterfactual(
        intervention={"X": 2.0},
        observed_data_point=observed_point,
        method="adjust",
    )
    printer.pretty("counterfactual_individual", cf_individual.to_dict(), "success")

    graph_path = Path("/tmp/causal_graph_test.gml")
    saved_path = builder.save_graph(graph_path)
    printer.pretty("saved_graph_path", str(saved_path), "success")

    print("\n=== Test ran successfully ===\n")