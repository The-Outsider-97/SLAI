"""
Adaptive neural-probabilistic circuit for the reasoning subsystem.

The circuit bridges Bayesian network structure, symbolic knowledge-base facts,
and differentiable probability estimation.  It is intentionally compatible with
``ModelCompute`` and ``ProbabilisticModels``:

- exposes ``input_vars``, ``output_vars`` and ``var_index``;
- returns probability tensors in ``forward``;
- provides scope/activation/evidence introspection;
- keeps all configuration in ``reasoning_config.yaml`` via the existing loader.
"""
from __future__ import annotations

import math
import random
import time

from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.reasoning_errors import *
from ..utils.reasoning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Adaptive Circuit")
printer = PrettyPrinter()

Fact = Tuple[str, str, str]
TensorLike = Union[torch.Tensor, Sequence[float], Mapping[str, Any]]


class AdaptiveCircuit(nn.Module):
    """Hybrid neural circuit informed by Bayesian structure and KB evidence.

    Args:
        network_structure: Bayesian-network-like mapping with at least ``nodes``
            and optionally ``edges`` and ``cpt``.
        knowledge_base: Mapping of fact tuples/strings to confidence values.

    The class remains lightweight enough for fast probabilistic queries, but it
    now performs production-grade validation, deterministic indexing, structured
    diagnostics, KB-feature caching, and configurable Bayesian prior
    initialization.
    """

    def __init__(self, network_structure: Dict[str, Any],
                 knowledge_base: Optional[Mapping[Union[str, Sequence[Any]], Any]] = None) -> None:
        super().__init__()

        self.config: Dict[str, Any] = load_global_config()
        self.circuit_config: Dict[str, Any] = get_config_section("adaptive_circuit") or {}

        self.embedding_dim: int = self._cfg_int(
            "embedding_dim",
            self.config.get("embedding_dim", 64),
            minimum=1,
            maximum=4096,
        )
        self.hidden_dim: int = self._cfg_int("hidden_dim", 128, minimum=4, maximum=16384)
        self.num_kb_embeddings: int = self._cfg_int("num_kb_embeddings", 1000, minimum=0, maximum=250_000)
        self.dropout_rate: float = self._cfg_float("dropout", 0.1, minimum=0.0, maximum=0.95)
        self.min_prior_probability: float = self._cfg_float("min_prior_probability", 0.01, minimum=1e-8, maximum=0.49)
        self.max_prior_probability: float = self._cfg_float("max_prior_probability", 0.99, minimum=0.51, maximum=1.0 - 1e-8)
        self.default_prior: float = self._cfg_float("default_prior", 0.5, minimum=1e-8, maximum=1.0 - 1e-8)
        self.prior_confidence_default: float = self._cfg_float("prior_confidence_default", 0.7, minimum=0.0, maximum=1.0)
        self.enable_kb_embeddings: bool = bool(self.circuit_config.get("enable_kb_embeddings", True))
        self.kb_min_confidence: float = self._cfg_float("kb_min_confidence", 0.0, minimum=0.0, maximum=1.0)
        self.max_related_entities: int = self._cfg_int("max_related_entities", 64, minimum=1, maximum=4096)
        self.embedding_aggregation: str = str(self.circuit_config.get("embedding_aggregation", "weighted_mean")).lower()
        self.activation_name: str = str(self.circuit_config.get("activation", "relu")).lower()
        self.use_layer_norm: bool = bool(self.circuit_config.get("use_layer_norm", True))
        self.strict_structure: bool = bool(self.circuit_config.get("strict_structure", True))
        self.trace_detach: bool = bool(self.circuit_config.get("trace_detach", True))
        self.seed: Optional[int] = self._optional_int("seed", self.circuit_config.get("seed"))
        self.device_name: str = str(self.circuit_config.get("device", "auto")).lower()

        if self.min_prior_probability >= self.max_prior_probability:
            raise ReasoningConfigurationError(
                "adaptive_circuit min_prior_probability must be lower than max_prior_probability",
                context={
                    "min_prior_probability": self.min_prior_probability,
                    "max_prior_probability": self.max_prior_probability,
                },
            )
        if self.embedding_aggregation not in {"mean", "sum", "weighted_mean", "attention"}:
            raise ReasoningConfigurationError(
                "Unsupported adaptive_circuit embedding_aggregation",
                context={"embedding_aggregation": self.embedding_aggregation},
            )

        self._set_deterministic_seed()
        self._device = self._select_device(self.device_name)

        self.network_structure: Dict[str, Any] = self._validate_network_structure(network_structure)
        self.input_vars: List[str] = list(self.network_structure["nodes"])
        self.output_vars: List[str] = list(self.input_vars)
        self.var_index: Dict[str, int] = {node: idx for idx, node in enumerate(self.input_vars)}
        self.num_bn_nodes: int = len(self.input_vars)
        self.parent_map: Dict[str, List[str]] = self._build_parent_map()
        self.child_map: Dict[str, List[str]] = self._build_child_map()
        self.topological_order: List[str] = self._topological_sort()

        self.knowledge_base: Dict[Fact, float] = self._normalize_knowledge_base(knowledge_base or {})
        self._kb_signature: Tuple[Tuple[Fact, float], ...] = freeze_kb_signature(self.knowledge_base)
        self.kb_entity_to_idx: Dict[str, int] = {}
        self.kb_entity_confidence: Dict[str, float] = {}
        self.kb_node_entity_indices: Dict[str, List[int]] = {}
        self.kb_node_entity_weights: Dict[str, List[float]] = {}
        self.kb_embedding: Optional[nn.Embedding] = None
        self.kb_attention: Optional[nn.Linear] = None
        self._initialize_kb_embeddings()

        fc1_input_dim = self.num_bn_nodes + (self.embedding_dim if self.kb_embedding is not None else 0)
        self.fc1 = nn.Linear(fc1_input_dim, self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim) if self.use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, max(2, self.hidden_dim // 2))
        self.fc3 = nn.Linear(max(2, self.hidden_dim // 2), self.num_bn_nodes)

        self._initialize_network_parameters()
        self._initialize_with_priors()
        self.to(self._device)

        logger.info(
            "AdaptiveCircuit initialized | nodes=%s | edges=%s | kb_facts=%s | input_dim=%s | device=%s",
            self.num_bn_nodes,
            len(self.network_structure.get("edges", [])),
            len(self.knowledge_base),
            fc1_input_dim,
            self._device,
        )
        printer.status("INIT", f"Adaptive Circuit initialized with {self.num_bn_nodes} nodes", "success")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _cfg_int(self, key: str, default: Any, *, minimum: int, maximum: int) -> int:
        value = self.circuit_config.get(key, default)
        try:
            return bounded_iterations(value, minimum=minimum, maximum=maximum)
        except ReasoningError:
            raise
        except Exception as exc:
            raise ReasoningConfigurationError(
                f"adaptive_circuit.{key} must be an integer",
                cause=exc,
                context={"key": key, "value": value},
            ) from exc

    def _optional_int(self, key: str, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ReasoningConfigurationError(
                f"adaptive_circuit.{key} must be an integer when provided",
                cause=exc,
                context={"key": key, "value": value},
            ) from exc

    def _cfg_float(self, key: str, default: Any, *, minimum: float, maximum: float) -> float:
        value = self.circuit_config.get(key, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ReasoningConfigurationError(
                f"adaptive_circuit.{key} must be numeric",
                cause=exc,
                context={"key": key, "value": value},
            ) from exc
        if parsed < minimum or parsed > maximum:
            raise ReasoningConfigurationError(
                f"adaptive_circuit.{key} outside allowed range",
                context={"key": key, "value": parsed, "minimum": minimum, "maximum": maximum},
            )
        return parsed

    def _set_deterministic_seed(self) -> None:
        if self.seed is None:
            return
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _select_device(self, requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for AdaptiveCircuit but not available. Falling back to CPU.")
            return torch.device("cpu")
        try:
            return torch.device(requested)
        except Exception as exc:
            raise ReasoningConfigurationError(
                "Invalid adaptive_circuit device configuration",
                cause=exc,
                context={"device": requested},
            ) from exc

    # ------------------------------------------------------------------
    # Structure and knowledge preparation
    # ------------------------------------------------------------------
    def _validate_network_structure(self, network_structure: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(network_structure, dict):
            raise ModelInitializationError(
                "Network structure must be a dictionary",
                context={"type": type(network_structure).__name__},
            )

        raw_nodes = network_structure.get("nodes")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise ModelInitializationError("Network structure must contain a non-empty 'nodes' list")

        nodes = [str(node).strip() for node in raw_nodes if str(node).strip()]
        if len(nodes) != len(raw_nodes):
            raise ModelInitializationError("Network nodes cannot be empty", context={"nodes": raw_nodes})
        if len(set(nodes)) != len(nodes):
            duplicates = sorted({node for node in nodes if nodes.count(node) > 1})
            raise ModelInitializationError("Network nodes must be unique", context={"duplicates": duplicates})

        node_set = set(nodes)
        raw_edges = network_structure.get("edges", []) or []
        if not isinstance(raw_edges, list):
            raise ModelInitializationError("Network 'edges' must be a list", context={"type": type(raw_edges).__name__})

        edges: List[Tuple[str, str]] = []
        invalid_edges: List[Any] = []
        for edge in raw_edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                invalid_edges.append(edge)
                continue
            parent, child = str(edge[0]).strip(), str(edge[1]).strip()
            if parent not in node_set or child not in node_set or parent == child:
                invalid_edges.append(edge)
                continue
            edges.append((parent, child))

        if invalid_edges and self.strict_structure:
            raise CircuitConstraintError(
                "Network structure contains invalid edges",
                context={"invalid_edges": invalid_edges[:16], "count": len(invalid_edges)},
            )
        if invalid_edges:
            logger.warning("Ignoring invalid AdaptiveCircuit edges: %s", invalid_edges[:16])

        cpt = network_structure.get("cpt", {}) or {}
        if not isinstance(cpt, dict):
            raise ModelInitializationError("Network 'cpt' must be a dictionary when provided")

        unknown_cpt_nodes = sorted(str(node) for node in cpt.keys() if str(node) not in node_set)
        if unknown_cpt_nodes and self.strict_structure:
            raise CircuitConstraintError(
                "CPT contains nodes that are not present in network nodes",
                context={"unknown_cpt_nodes": unknown_cpt_nodes},
            )

        normalized = dict(network_structure)
        normalized["nodes"] = nodes
        normalized["edges"] = edges
        normalized["cpt"] = {str(k): v for k, v in cpt.items() if str(k) in node_set}
        return normalized

    def _normalize_knowledge_base(self, knowledge_base: Mapping[Union[str, Sequence[Any]], Any]) -> Dict[Fact, float]:
        if not knowledge_base:
            return {}
        normalized: Dict[Fact, float] = {}
        skipped = 0
        for raw_fact, raw_confidence in knowledge_base.items():
            try:
                fact = normalize_fact(raw_fact)
                confidence = clamp_confidence(raw_confidence)
                if confidence < self.kb_min_confidence:
                    continue
                normalized[fact] = merge_confidence(normalized.get(fact, 0.0), confidence)
            except ReasoningError:
                skipped += 1
            except Exception:
                skipped += 1
        if skipped:
            logger.warning("Skipped %s malformed AdaptiveCircuit KB facts", skipped)
        return normalized

    def _build_parent_map(self) -> Dict[str, List[str]]:
        parent_map: Dict[str, List[str]] = {node: [] for node in self.input_vars}
        for parent, child in self.network_structure.get("edges", []):
            parent_map[child].append(parent)
        return parent_map

    def _build_child_map(self) -> Dict[str, List[str]]:
        child_map: Dict[str, List[str]] = {node: [] for node in self.input_vars}
        for parent, child in self.network_structure.get("edges", []):
            child_map[parent].append(child)
        return child_map

    def _topological_sort(self) -> List[str]:
        """Return dependency-aware node order; preserve all nodes on cycles."""
        in_degree: Dict[str, int] = {node: len(self.parent_map.get(node, [])) for node in self.input_vars}
        queue = deque([node for node in self.input_vars if in_degree[node] == 0])
        sorted_nodes: List[str] = []

        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)
            for child in self.child_map.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(sorted_nodes) != len(self.input_vars):
            missing = [node for node in self.input_vars if node not in sorted_nodes]
            message = "Bayesian network contains a cycle; appending cyclic nodes after acyclic order"
            if self.strict_structure:
                raise CircuitConstraintError(message, context={"cyclic_nodes": missing})
            logger.warning("%s: %s", message, missing)
            sorted_nodes.extend(missing)

        return sorted_nodes

    # ------------------------------------------------------------------
    # Knowledge embedding layer
    # ------------------------------------------------------------------
    def _initialize_kb_embeddings(self) -> None:
        if not self.enable_kb_embeddings or not self.knowledge_base or self.num_kb_embeddings == 0:
            self.kb_embedding = None
            return

        entity_scores: Dict[str, float] = defaultdict(float)
        entity_nodes: Dict[str, Set[str]] = defaultdict(set)
        node_set = set(self.input_vars)

        for fact, confidence in self.knowledge_base.items():
            s, p, o = fact
            tokens = [str(s), str(p), str(o)]
            for token in tokens:
                entity_scores[token] += float(confidence)
            for node in node_set.intersection(tokens):
                for token in tokens:
                    if token != node:
                        entity_nodes[node].add(token)

        ranked_entities = sorted(entity_scores.items(), key=lambda item: (-item[1], item[0]))[: self.num_kb_embeddings]
        self.kb_entity_to_idx = {entity: idx for idx, (entity, _) in enumerate(ranked_entities)}
        self.kb_entity_confidence = {entity: clamp_confidence(score / max(1.0, len(self.knowledge_base))) for entity, score in ranked_entities}

        if not self.kb_entity_to_idx:
            self.kb_embedding = None
            return

        self.kb_embedding = nn.Embedding(len(self.kb_entity_to_idx), self.embedding_dim)
        if self.embedding_aggregation == "attention":
            self.kb_attention = nn.Linear(self.embedding_dim, 1)

        for node in self.input_vars:
            related = sorted(
                [entity for entity in entity_nodes.get(node, set()) if entity in self.kb_entity_to_idx],
                key=lambda entity: (-self.kb_entity_confidence.get(entity, 0.0), entity),
            )[: self.max_related_entities]
            self.kb_node_entity_indices[node] = [self.kb_entity_to_idx[entity] for entity in related]
            self.kb_node_entity_weights[node] = [self.kb_entity_confidence.get(entity, 0.0) for entity in related]

        logger.info(
            "Created AdaptiveCircuit KB embedding | entities=%s | node_links=%s",
            len(self.kb_entity_to_idx),
            sum(len(v) for v in self.kb_node_entity_indices.values()),
        )

    def _get_kb_features_for_input(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Generate deterministic KB-derived feature vector for each batch item."""
        if self.kb_embedding is None or not self.kb_entity_to_idx:
            return None
    
        # Local references for type narrowing
        kb_embed = self.kb_embedding
        kb_attn = self.kb_attention if self.embedding_aggregation == "attention" else None
    
        batch_size = x.size(0)
        device = x.device
        node_vectors: List[torch.Tensor] = []
    
        for node in self.input_vars:
            indices = self.kb_node_entity_indices.get(node, [])
            if not indices:
                continue
    
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            embeddings = kb_embed(idx_tensor)                     # safe now
    
            if self.embedding_aggregation == "sum":
                node_vector = embeddings.sum(dim=0)
            elif self.embedding_aggregation == "mean":
                node_vector = embeddings.mean(dim=0)
            elif self.embedding_aggregation == "attention" and kb_attn is not None:
                attention_logits = kb_attn(embeddings).squeeze(-1)   # safe
                attention = F.softmax(attention_logits, dim=0)
                node_vector = (embeddings * attention.unsqueeze(-1)).sum(dim=0)
            else:  # weighted_mean (default)
                weights = torch.tensor(
                    self.kb_node_entity_weights.get(node, [1.0] * len(indices)),
                    dtype=embeddings.dtype,
                    device=device,
                ).clamp_min(1e-8)
                weights = weights / weights.sum()
                node_vector = (embeddings * weights.unsqueeze(-1)).sum(dim=0)
    
            node_vectors.append(node_vector)
    
        if not node_vectors:
            return torch.zeros(batch_size, self.embedding_dim, dtype=x.dtype, device=device)
    
        aggregated = torch.stack(node_vectors, dim=0).mean(dim=0).to(dtype=x.dtype, device=device)
        return aggregated.unsqueeze(0).expand(batch_size, -1)

    # ------------------------------------------------------------------
    # Forward and tensor conversion
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run probability inference for a batch of evidence vectors."""
        x = self._validate_input_tensor(x)
        kb_features = self._get_kb_features_for_input(x)
        combined_input = torch.cat([x, kb_features], dim=1) if kb_features is not None else x

        h = self._activation(self.fc1(combined_input))
        h = self.layer_norm(h)
        h = self.dropout(h)
        h = self._activation(self.fc2(h))
        logits = self.fc3(h)
        return torch.sigmoid(logits).clamp(self.min_prior_probability, self.max_prior_probability)

    def _validate_input_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ModelInferenceError(
                "AdaptiveCircuit input must be a torch.Tensor",
                context={"input_type": type(x).__name__},
            )
        if x.dim() != 2 or x.size(1) != self.num_bn_nodes:
            raise ModelInferenceError(
                "AdaptiveCircuit input tensor has invalid shape",
                context={"expected": ("batch", self.num_bn_nodes), "actual": tuple(x.shape)},
            )
        if not torch.is_floating_point(x):
            x = x.float()
        if not torch.isfinite(x).all():
            raise ModelInferenceError("AdaptiveCircuit input contains NaN or infinite values")
        return x.to(device=self._device, dtype=next(self.parameters()).dtype)

    def evidence_to_tensor(self, evidence: Mapping[str, Any], *, batch_size: int = 1) -> torch.Tensor:
        """Convert evidence mapping into a circuit input tensor."""
        if not isinstance(evidence, Mapping):
            raise ReasoningValidationError(
                "Evidence must be a mapping from variable name to probability/value",
                context={"type": type(evidence).__name__},
            )
        if batch_size < 1:
            raise ReasoningValidationError("batch_size must be >= 1", context={"batch_size": batch_size})

        values: List[float] = []
        for var in self.input_vars:
            raw_value = evidence.get(var, self.default_prior)
            try:
                values.append(clamp_confidence(raw_value))
            except ReasoningError:
                raise
            except Exception as exc:
                raise ReasoningValidationError(
                    "Evidence value could not be converted to probability",
                    cause=exc,
                    context={"variable": var, "value": raw_value},
                ) from exc

        tensor = torch.tensor(values, dtype=torch.float32, device=self._device).unsqueeze(0)
        return tensor.expand(batch_size, -1).clone()

    def state_dict_from_output(self, output: torch.Tensor, *, batch_index: int = 0) -> Dict[str, float]:
        """Convert a probability output tensor to ``{node: probability}``."""
        if output.dim() != 2 or output.size(1) != self.num_bn_nodes:
            raise ModelInferenceError(
                "Output tensor has invalid shape for AdaptiveCircuit state conversion",
                context={"actual": tuple(output.shape), "expected_width": self.num_bn_nodes},
            )
        if batch_index < 0 or batch_index >= output.size(0):
            raise ReasoningValidationError("batch_index outside output batch", context={"batch_index": batch_index})
        row = output.detach().cpu()[batch_index]
        return {var: float(row[idx].item()) for idx, var in enumerate(self.output_vars)}

    def _activation(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "gelu":
            return F.gelu(value)
        if self.activation_name == "silu":
            return F.silu(value)
        if self.activation_name == "tanh":
            return torch.tanh(value)
        if self.activation_name == "leaky_relu":
            return F.leaky_relu(value, negative_slope=0.01)
        return F.relu(value)

    # ------------------------------------------------------------------
    # Initialization with Bayesian priors
    # ------------------------------------------------------------------
    def _initialize_network_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        if self.kb_embedding is not None:
            nn.init.normal_(self.kb_embedding.weight, mean=0.0, std=0.02)
        if self.kb_attention is not None:
            nn.init.xavier_uniform_(self.kb_attention.weight)
            nn.init.zeros_(self.kb_attention.bias)

    def _initialize_with_priors(self) -> None:
        cpt = self.network_structure.get("cpt", {}) or {}
        if not cpt:
            logger.warning("AdaptiveCircuit CPT not found. Using neutral prior initialization.")
            return

        with torch.no_grad():
            for node_name in self.topological_order:
                idx = self.var_index[node_name]
                node_cpt = cpt.get(node_name, {}) or {}
                prior = self._extract_node_prior(node_name, node_cpt)
                clamped_prior = self._clamp_probability(prior)
                self.fc3.bias.data[idx] = self._logit(clamped_prior)

                prior_confidence = self._extract_prior_confidence(node_cpt)
                scale = math.sqrt(2.0 / max(1, self.fc3.in_features + self.num_bn_nodes))
                self.fc3.weight.data[idx].normal_(mean=0.0, std=scale * max(prior_confidence, 1e-4))

                if idx < self.fc1.weight.data.size(1):
                    first_scale = math.sqrt(2.0 / max(1, self.fc1.in_features + self.hidden_dim))
                    self.fc1.weight.data[:, idx].normal_(mean=0.0, std=first_scale * max(prior_confidence, 1e-4))

        logger.info("AdaptiveCircuit weights initialized with Bayesian priors")

    def _extract_node_prior(self, node_name: str, node_cpt: Mapping[str, Any]) -> float:
        if "prior" in node_cpt:
            return self._coerce_probability_strict(node_cpt.get("prior"), self.default_prior)
        if "probability" in node_cpt:
            return self._coerce_probability_strict(node_cpt.get("probability"), self.default_prior)
        if "p_true" in node_cpt:
            return self._coerce_probability_strict(node_cpt.get("p_true"), self.default_prior)
        if "values" in node_cpt:
            value = self._extract_probability_from_values(node_cpt.get("values"))
            if value is not None:
                return value
        if "table" in node_cpt:
            value = self._extract_probability_from_values(node_cpt.get("table"))
            if value is not None:
                return value
    
        parent_priors = [
            self._clamp_probability(torch.sigmoid(self.fc3.bias.data[self.var_index[parent]]).item())
            for parent in self.parent_map.get(node_name, [])
            if parent in self.var_index
        ]
        if parent_priors:
            return sum(parent_priors) / len(parent_priors)
        return self.default_prior

    def _extract_probability_from_values(self, value: Any) -> Optional[float]:
        if isinstance(value, Mapping):
            for key in ("True", "true", True, "1", 1):
                if key in value:
                    return self._coerce_probability(value[key], fallback=self.default_prior)
            numeric_values = [self._coerce_probability(v, fallback=None) for v in value.values()]
            numeric_values = [v for v in numeric_values if v is not None]
            if numeric_values:
                return sum(numeric_values) / len(numeric_values)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            numeric_values = [self._coerce_probability(v, fallback=None) for v in value]
            numeric_values = [v for v in numeric_values if v is not None]
            if numeric_values:
                return sum(numeric_values) / len(numeric_values)
        return None

    def _extract_prior_confidence(self, node_cpt: Mapping[str, Any]) -> float:
        if "confidence" not in node_cpt:
            return self.prior_confidence_default
        try:
            return clamp_confidence(node_cpt.get("confidence"))
        except Exception:
            return self.prior_confidence_default

    def _coerce_probability(self, value: Any, *, fallback: Optional[float]) -> Optional[float]:
        try:
            return clamp_confidence(value)
        except Exception:
            return fallback

    def _clamp_probability(self, value: float) -> float:
        return max(self.min_prior_probability, min(self.max_prior_probability, float(value)))

    @staticmethod
    def _logit(probability: float) -> float:
        p = max(1e-8, min(1.0 - 1e-8, float(probability)))
        return math.log(p / (1.0 - p))

    # ------------------------------------------------------------------
    # Introspection and compatibility APIs
    # ------------------------------------------------------------------
    def to_evidence_dict(self, input_tensor: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Serialize circuit metadata and optional output probabilities."""
        evidence: Dict[str, Any] = {
            "metadata": {
                "model_type": "AdaptiveCircuit",
                "input_vars": list(self.input_vars),
                "output_vars": list(self.output_vars),
                "num_parameters": sum(p.numel() for p in self.parameters()),
                "num_trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "num_kb_facts": len(self.knowledge_base),
                "num_kb_entities": len(self.kb_entity_to_idx),
                "device": str(self._device),
                "topological_order": list(self.topological_order),
            },
            "timestamp": time.time(),
        }
        if input_tensor is not None:
            with torch.no_grad():
                output = self.forward(input_tensor)
            evidence["node_states"] = self.state_dict_from_output(output)
        return json_safe_reasoning_state(evidence)

    def compute_scopes(self) -> Dict[str, Set[str]]:
        """Compute approximate variable scopes for circuit components."""
        scopes: Dict[str, Set[str]] = {}
        for node_name in self.input_vars:
            scopes[f"input_{node_name}"] = {node_name}

        base_scope = set(self.input_vars)
        scopes["fc1"] = set(base_scope)
        scopes["layer_norm"] = set(base_scope)
        scopes["bn_layer_norm"] = scopes["layer_norm"]  # backward-compatible alias
        scopes["dropout"] = set(base_scope)
        scopes["fc2"] = set(base_scope)
        scopes["fc3"] = set(base_scope)
        scopes["output"] = set(base_scope)

        if self.kb_embedding is not None:
            kb_scope: Set[str] = set()
            for fact in self.knowledge_base:
                kb_scope.update(str(item) for item in fact)
            scopes["kb_embedding"] = kb_scope
            scopes["fc1"] = scopes["fc1"] | kb_scope

        return scopes

    def trace_activations(self, input_tensor: torch.Tensor, _depth: int = 0, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Trace activations through the circuit for diagnostics/explainability."""
        x = self._validate_input_tensor(input_tensor)
        previous_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                activations: Dict[str, torch.Tensor] = {"input": x.clone()}
                kb_features = self._get_kb_features_for_input(x)
                if kb_features is not None:
                    activations["kb_features"] = kb_features.clone()
                    combined_input = torch.cat([x, kb_features], dim=1)
                else:
                    combined_input = x

                fc1_pre = self.fc1(combined_input)
                relu1 = self._activation(fc1_pre)
                norm = self.layer_norm(relu1)
                dropped = self.dropout(norm)
                fc2_pre = self.fc2(dropped)
                relu2 = self._activation(fc2_pre)
                logits = self.fc3(relu2)
                probabilities = torch.sigmoid(logits).clamp(self.min_prior_probability, self.max_prior_probability)

                activations.update(
                    {
                        "fc1_pre_act": fc1_pre,
                        "activation1": relu1,
                        "layer_norm": norm,
                        "bn_layer_norm": norm,
                        "dropout": dropped,
                        "fc2_pre_act": fc2_pre,
                        "activation2": relu2,
                        "fc3_logits": logits,
                        "output_probabilities": probabilities,
                    }
                )
        finally:
            self.train(previous_training)

        if self.trace_detach:
            return {name: tensor.detach().cpu() for name, tensor in activations.items()}
        return activations

    def explain_node(self, node_name: str, evidence: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Return compact explanation metadata for a node under optional evidence."""
        if node_name not in self.var_index:
            raise ReasoningValidationError(
                "Unknown AdaptiveCircuit node requested",
                context={"node": node_name, "available": self.input_vars},
            )
        tensor = self.evidence_to_tensor(evidence or {})
        with torch.no_grad():
            output = self.forward(tensor)
        idx = self.var_index[node_name]
        related_entities = [
            entity for entity, entity_idx in self.kb_entity_to_idx.items()
            if entity_idx in set(self.kb_node_entity_indices.get(node_name, []))
        ]
        explanation = {
            "node": node_name,
            "probability": float(output[0, idx].detach().cpu().item()),
            "parents": list(self.parent_map.get(node_name, [])),
            "children": list(self.child_map.get(node_name, [])),
            "related_kb_entities": related_entities[: self.max_related_entities],
            "prior": self._extract_node_prior(node_name, self.network_structure.get("cpt", {}).get(node_name, {}) or {}),
        }
        return json_safe_reasoning_state(explanation)

    def describe_structure(self) -> Dict[str, Any]:
        """Return JSON-safe structural diagnostics for monitoring and tests."""
        roots = [node for node in self.input_vars if not self.parent_map.get(node)]
        leaves = [node for node in self.input_vars if not self.child_map.get(node)]
        payload = {
            "num_nodes": self.num_bn_nodes,
            "num_edges": len(self.network_structure.get("edges", [])),
            "roots": roots,
            "leaves": leaves,
            "has_kb_embedding": self.kb_embedding is not None,
            "num_kb_entities": len(self.kb_entity_to_idx),
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "activation": self.activation_name,
            "device": str(self._device),
            "topological_order": self.topological_order,
        }
        return json_safe_reasoning_state(payload)

    def parameter_summary(self) -> Dict[str, Any]:
        """Return parameter counts and simple norms for observability."""
        summary: Dict[str, Any] = {
            "total": 0,
            "trainable": 0,
            "layers": {},
        }
        for name, param in self.named_parameters():
            count = int(param.numel())
            summary["total"] += count
            if param.requires_grad:
                summary["trainable"] += count
            summary["layers"][name] = {
                "shape": list(param.shape),
                "trainable": bool(param.requires_grad),
                "norm": float(param.detach().norm().cpu().item()),
            }
        return json_safe_reasoning_state(summary)

    def update_knowledge_base(self, knowledge_base: Mapping[Union[str, Sequence[Any]], Any], *, rebuild_embeddings: bool = True) -> None:
        """Update KB state and optionally rebuild KB embeddings/indexes."""
        self.knowledge_base = self._normalize_knowledge_base(knowledge_base)
        self._kb_signature = freeze_kb_signature(self.knowledge_base)
        if rebuild_embeddings:
            self.kb_entity_to_idx = {}
            self.kb_entity_confidence = {}
            self.kb_node_entity_indices = {}
            self.kb_node_entity_weights = {}
            self.kb_embedding = None
            self.kb_attention = None
            self._initialize_kb_embeddings()
            # Register newly created submodules correctly after reassignment.
            self.to(self._device)

    def freeze_feature_extractor(self) -> None:
        """Freeze lower layers while leaving output calibration trainable."""
        for module in (self.fc1, self.fc2, self.layer_norm):
            for param in module.parameters():
                param.requires_grad = False
        if self.kb_embedding is not None:
            for param in self.kb_embedding.parameters():
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Make all circuit parameters trainable again."""
        for param in self.parameters():
            param.requires_grad = True

    def _coerce_probability_strict(self, value: Any, fallback: float) -> float:
        """Like _coerce_probability but guarantees float return."""
        result = self._coerce_probability(value, fallback=fallback)
        # result is float because fallback is float and _coerce_probability returns
        # the fallback on failure – but type checker needs help.
        assert result is not None
        return result

    @property
    def nodes(self) -> List[Tuple[str, nn.Module]]:
        """Return named modules used by structural tooling."""
        node_list: List[Tuple[str, nn.Module]] = [
            ("fc1", self.fc1),
            ("layer_norm", self.layer_norm),
            ("bn_layer_norm", self.layer_norm),
            ("dropout", self.dropout),
            ("fc2", self.fc2),
            ("fc3", self.fc3),
        ]
        if self.kb_embedding is not None:
            node_list.insert(0, ("kb_embedding", self.kb_embedding))
        if self.kb_attention is not None:
            node_list.insert(1, ("kb_attention", self.kb_attention))
        return node_list

    @property
    def root(self) -> nn.Module:
        """Return circuit root/output module for ``ModelCompute`` compatibility."""
        return self.fc3


if __name__ == "__main__":
    print("\n=== Running Adaptive Circuit ===\n")
    printer.status("TEST", "Adaptive Circuit initialized", "info")

    test_network = {
        "nodes": ["Rain", "Sprinkler", "WetGrass"],
        "edges": [["Rain", "WetGrass"], ["Sprinkler", "WetGrass"]],
        "cpt": {
            "Rain": {"prior": 0.35, "confidence": 0.9},
            "Sprinkler": {"prior": 0.2, "confidence": 0.8},
            "WetGrass": {"values": {"True": 0.78, "False": 0.22}, "confidence": 0.85},
        },
    }
    test_kb = {
        ("Rain", "causes", "WetGrass"): 0.92,
        ("Sprinkler", "causes", "WetGrass"): 0.88,
        ("Clouds", "increase", "Rain"): 0.73,
    }

    circuit = AdaptiveCircuit(network_structure=test_network, knowledge_base=test_kb) # type: ignore
    input_tensor = circuit.evidence_to_tensor({"Rain": 1.0, "Sprinkler": 0.0})
    output = circuit(input_tensor)

    assert output.shape == (1, len(circuit.output_vars))
    assert torch.isfinite(output).all()
    assert bool(((output >= 0.0) & (output <= 1.0)).all())

    evidence = circuit.to_evidence_dict(input_tensor)
    scopes = circuit.compute_scopes()
    activations = circuit.trace_activations(input_tensor)
    explanation = circuit.explain_node("WetGrass", {"Rain": 1.0, "Sprinkler": 0.0})
    structure = circuit.describe_structure()
    parameters = circuit.parameter_summary()

    assert "node_states" in evidence
    assert "output" in scopes
    assert "output_probabilities" in activations
    assert explanation["node"] == "WetGrass"
    assert structure["num_nodes"] == 3
    assert parameters["total"] > 0

    printer.status("TEST", f"Forward output shape: {tuple(output.shape)}", "success")
    printer.status("TEST", f"WetGrass probability: {explanation['probability']:.4f}", "success")
    printer.status("TEST", "Adaptive Circuit smoke test completed", "success")

    print("\n=== Test ran successfully ===\n")
