"""
Production-grade probabilistic circuit operations for the reasoning subsystem.

ModelCompute is the low-level execution and adaptation layer used by
ProbabilisticModels. It deliberately stays focused on tractable neural / SPN
circuit operations while delegating shared validation semantics, confidence
handling, timestamps, and JSON-safe state conversion to the reasoning helpers.
"""
from __future__ import annotations

import math
import time
import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore

from torch.utils.data import DataLoader, TensorDataset # type: ignore
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.reasoning_errors import *
from ..utils.reasoning_helpers import *
from .nodes import ProductNode, SumNode
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Model Compute")
printer = PrettyPrinter()

VariableSpec = Union[str, Sequence[str], Tuple[str, ...], List[str]]
Evidence = Mapping[str, Any]


class ModelCompute(nn.Module):
    """Execution, adaptation, diagnostics, and explainability for probabilistic circuits.

    Expected circuit interface:
    - ``input_vars``: ordered list of input variable names.
    - ``output_vars``: ordered list of output variable names. If absent, input vars
      are reused for backwards compatibility with AdaptiveCircuit.
    - ``var_index``: optional mapping used by legacy callers. If absent, it is
      derived from ``input_vars``.
    - ``forward(Tensor) -> Tensor``: returns probabilities or logits for output vars.

    The class is intentionally defensive: all public operations validate the circuit
    once, normalize evidence consistently, and raise subsystem-specific errors.
    """

    def __init__(self, circuit: Optional[nn.Module] = None):
        super().__init__()
        self.config: Dict[str, Any] = load_global_config()
        self.model_config: Dict[str, Any] = get_config_section("model_compute")

        self.schema_version: float = float(self.model_config.get("schema_version", 1.0))
        self.reduction: str = str(self.model_config.get("reduction", "batchmean"))
        self.learning_rate: float = float(
            self.model_config.get("learning_rate", self.config.get("learning_rate", 0.001))
        )
        self.weight_decay: float = float(self.model_config.get("weight_decay", 0.0))
        self.optimizer_name: str = str(self.model_config.get("optimizer", "adam")).strip().lower()
        self.loss_name: str = str(self.model_config.get("loss", "binary_cross_entropy")).strip().lower()

        self.default_probability: float = clamp_confidence(
            self.model_config.get("default_probability", 0.5)
        )
        self.epsilon: float = float(self.model_config.get("epsilon", 1e-7))
        if self.epsilon <= 0.0 or self.epsilon >= 0.5:
            raise ReasoningConfigurationError(
                "model_compute.epsilon must be within (0, 0.5)",
                context={"epsilon": self.epsilon},
            )

        self.batch_size: int = bounded_iterations(
            self.model_config.get("batch_size", 32), minimum=1, maximum=8192
        )
        self.revision_epochs: int = bounded_iterations(
            self.model_config.get("revision_epochs", 5), minimum=1, maximum=10_000
        )
        self.map_steps: int = bounded_iterations(
            self.model_config.get("map_steps", 100), minimum=1, maximum=10_000
        )
        self.marginal_map_steps: int = bounded_iterations(
            self.model_config.get("marginal_map_steps", 10), minimum=1, maximum=10_000
        )
        self.default_sample_count: int = bounded_iterations(
            self.model_config.get("sample_count", 1000), minimum=1, maximum=250_000
        )
        self.gradient_clip_norm: float = float(self.model_config.get("gradient_clip_norm", 5.0))
        self.schema_increment: float = float(self.model_config.get("schema_increment", 0.1))
        self.l1_regularization: float = float(self.model_config.get("l1_regularization", 0.01))
        self.determinism_threshold: float = clamp_confidence(
            self.model_config.get("determinism_threshold", 0.999)
        )
        self.strict_evidence_bounds: bool = bool(self.model_config.get("strict_evidence_bounds", False))
        self.auto_sigmoid_outputs: bool = bool(self.model_config.get("auto_sigmoid_outputs", True))
        self.project_sum_weights: bool = bool(self.model_config.get("project_sum_weights", False))
        self.enforce_non_negative_parameters: bool = bool(
            self.model_config.get("enforce_non_negative_parameters", False)
        )
        self.max_explanation_items: int = bounded_iterations(
            self.model_config.get("max_explanation_items", 32), minimum=1, maximum=10_000
        )
        self.seed: Optional[int] = self._optional_int(self.model_config.get("seed"))
        self.device: torch.device = self._resolve_device(self.model_config.get("device", "auto"))

        self.optimizer: Optional[optim.Optimizer] = None
        self.loss_fn = self._build_loss_fn()
        self.circuit: Optional[nn.Module] = None
        self.set_circuit(circuit)

        self.belief_history: Dict[Any, List[float]] = {}
        self.revision_history: Deque[Dict[str, Any]] = deque(
            maxlen=bounded_iterations(self.model_config.get("revision_history_size", 256), minimum=1, maximum=100_000)
        )

        printer.status(
            "INIT",
            f"Model Compute successfully initialized with: Schema V.{self.schema_version}",
            "success",
        )

    # ------------------------------------------------------------------
    # Circuit binding and configuration
    # ------------------------------------------------------------------
    def set_circuit(self, value: Optional[nn.Module]) -> None:
        """Attach or detach the underlying circuit and rebuild its optimizer."""
        if value is None:
            self.circuit = None
            self.optimizer = None
            logger.warning("Circuit set to None - optimizer disabled")
            return

        self._validate_circuit_interface(value)
        value.to(self.device)
        self.circuit = value
        self.optimizer = self._build_optimizer(value)
        logger.info(
            "Circuit attached to ModelCompute | inputs=%s | outputs=%s | optimizer=%s",
            len(getattr(value, "input_vars", [])),
            len(getattr(value, "output_vars", [])),
            self.optimizer_name if self.optimizer is not None else "disabled",
        )

    def _build_loss_fn(self) -> nn.Module:
        loss_name = self.loss_name.replace("-", "_")
        if loss_name in {"binary_cross_entropy", "bce"}:
            return nn.BCELoss(reduction=self._loss_reduction(default="mean"))
        if loss_name in {"mse", "mean_squared_error"}:
            return nn.MSELoss(reduction=self._loss_reduction(default="mean"))
        if loss_name in {"kl", "kl_divergence", "kullback_leibler"}:
            return nn.KLDivLoss(reduction=self.reduction)
        raise ReasoningConfigurationError(
            "Unsupported model_compute.loss",
            context={"loss": self.loss_name, "supported": ["binary_cross_entropy", "mse", "kl_divergence"]},
        )

    def _loss_reduction(self, *, default: str) -> str:
        if self.reduction in {"none", "mean", "sum"}:
            return self.reduction
        return default

    def _build_optimizer(self, circuit: nn.Module) -> Optional[optim.Optimizer]:
        params = [p for p in circuit.parameters() if p.requires_grad]
        if not params:
            logger.warning("Circuit has no trainable parameters - optimizer disabled")
            return None

        if self.learning_rate <= 0.0:
            raise ReasoningConfigurationError(
                "model_compute.learning_rate must be positive",
                context={"learning_rate": self.learning_rate},
            )

        if self.optimizer_name == "adamw":
            return optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_name == "sgd":
            momentum = float(self.model_config.get("momentum", 0.0))
            return optim.SGD(params, lr=self.learning_rate, momentum=momentum, weight_decay=self.weight_decay)
        if self.optimizer_name == "adam":
            return optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        raise ReasoningConfigurationError(
            "Unsupported model_compute.optimizer",
            context={"optimizer": self.optimizer_name, "supported": ["adam", "adamw", "sgd"]},
        )

    def _resolve_device(self, configured_device: Any) -> torch.device:
        device_name = str(configured_device or "auto").strip().lower()
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            return torch.device(device_name)
        except Exception as exc:
            raise ReasoningConfigurationError(
                "Invalid model_compute.device",
                cause=exc,
                context={"device": configured_device},
            ) from exc

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        return int(value)

    def _require_circuit(self) -> nn.Module:
        if self.circuit is None:
            raise ModelInitializationError("Circuit not initialized")
        self._validate_circuit_interface(self.circuit)
        return self.circuit

    def _validate_circuit_interface(self, circuit: nn.Module) -> None:
        input_vars = getattr(circuit, "input_vars", None)
        if not isinstance(input_vars, list) or not input_vars:
            raise ModelInitializationError(
                "Circuit input_vars must be a non-empty list",
                context={"actual_type": type(input_vars).__name__},
            )

        if not all(isinstance(v, str) and v.strip() for v in input_vars):
            raise ModelInitializationError(
                "Circuit input_vars must contain non-empty strings",
                context={"input_vars": input_vars},
            )

        if not hasattr(circuit, "output_vars") or not getattr(circuit, "output_vars"):
            setattr(circuit, "output_vars", list(input_vars))

        output_vars = getattr(circuit, "output_vars")
        if not isinstance(output_vars, list) or not output_vars:
            raise ModelInitializationError(
                "Circuit output_vars must be a non-empty list",
                context={"actual_type": type(output_vars).__name__},
            )

        if not hasattr(circuit, "var_index") or not isinstance(getattr(circuit, "var_index"), dict):
            setattr(circuit, "var_index", {var: idx for idx, var in enumerate(input_vars)})

        missing_input_indices = [v for v in input_vars if v not in getattr(circuit, "var_index")]
        if missing_input_indices:
            raise ModelInitializationError(
                "Circuit var_index is missing input variables",
                context={"missing": missing_input_indices},
            )

        if not callable(getattr(circuit, "forward", None)):
            raise ModelInitializationError("Circuit must implement forward(x)")

    # ------------------------------------------------------------------
    # Core probabilistic circuit operations
    # ------------------------------------------------------------------
    def compute_marginal_probability(self, variables: VariableSpec, evidence: Optional[Evidence] = None) -> float:
        """Compute an independent joint marginal estimate for selected variables.

        For a single variable this returns ``P(variable=True | evidence)``. For
        multiple variables, ModelCompute preserves the legacy circuit contract and
        returns the product of the selected marginal outputs.
        """
        circuit = self._require_circuit()
        query_vars = self._normalize_variables(variables, available=getattr(circuit, "output_vars"))
        evidence_map = self._normalize_evidence(evidence or {})
        input_tensor = self._evidence_to_tensor(evidence_map)

        try:
            with torch.no_grad():
                probs = self._forward_probabilities(input_tensor, training=False)
                output_indices = self._output_indices(query_vars)
                selected = probs[:, output_indices]
                result = float(torch.prod(selected, dim=1).mean().item())
            return clamp_confidence(result)
        except ReasoningError:
            raise
        except Exception as exc:
            raise ModelInferenceError(
                "Marginal probability computation failed",
                cause=exc,
                context={"variables": query_vars, "evidence_keys": sorted(evidence_map.keys())},
            ) from exc

    def compute_all_marginals(self, evidence: Optional[Evidence] = None) -> Dict[str, float]:
        """Return all output marginals for the current circuit and evidence."""
        circuit = self._require_circuit()
        input_tensor = self._evidence_to_tensor(self._normalize_evidence(evidence or {}))
        with torch.no_grad():
            probs = self._forward_probabilities(input_tensor, training=False)[0]
        return {
            var: clamp_confidence(float(probs[idx].item()))
            for idx, var in enumerate(getattr(circuit, "output_vars"))
        }

    def conditional_probability(self, variable: str, evidence: Optional[Evidence] = None) -> float:
        """Alias used by higher-level probabilistic components."""
        return self.compute_marginal_probability((variable,), evidence or {})

    def compute_map_estimate(
        self,
        evidence: Optional[Evidence] = None,
        variables: Optional[VariableSpec] = None,
    ) -> Tuple[Dict[str, float], float]:
        """Estimate a MAP-like assignment without mutating circuit parameters.

        The optimizer operates on a temporary differentiable input state, not on the
        circuit weights. Evidence variables are clamped after each step. The returned
        state maps requested output variables to probabilities.
        """
        circuit = self._require_circuit()
        evidence_map = self._normalize_evidence(evidence or {})
        query_vars = self._normalize_variables(variables or getattr(circuit, "output_vars"), available=getattr(circuit, "output_vars"))
        query_indices = self._output_indices(query_vars)

        try:
            was_training = circuit.training
            circuit.eval()
            input_state = self._evidence_to_tensor(evidence_map).clone().detach().requires_grad_(True)
            observed_indices = [
                getattr(circuit, "var_index")[var]
                for var in evidence_map
                if var in getattr(circuit, "var_index")
            ]
            optimizer = optim.Adam([input_state], lr=max(self.learning_rate, 1e-5))

            for _ in range(self.map_steps):
                optimizer.zero_grad()
                probs = self._forward_probabilities(input_state, training=False)
                selected = torch.clamp(probs[:, query_indices], self.epsilon, 1.0 - self.epsilon)
                log_probability = torch.log(selected).mean()
                regularizer = torch.mean((input_state - self.default_probability) ** 2)
                loss = -(log_probability - self.l1_regularization * regularizer)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    input_state.clamp_(self.epsilon, 1.0 - self.epsilon)
                    for var, idx in zip(evidence_map.keys(), observed_indices):
                        input_state[:, idx] = evidence_map[var]

            with torch.no_grad():
                probs = self._forward_probabilities(input_state, training=False)[0]
                state = {var: clamp_confidence(float(probs[self._output_index(var)].item())) for var in query_vars}
                confidence = weighted_confidence(list(state.values())) if state else self.default_probability

            if was_training:
                circuit.train()
            return state, clamp_confidence(confidence)
        except ReasoningError:
            raise
        except Exception as exc:
            raise ModelInferenceError(
                "MAP estimation failed",
                cause=exc,
                context={"variables": query_vars, "evidence_keys": sorted(evidence_map.keys())},
            ) from exc

    def compute_moment(self, variable: str, order: int = 1, n_samples: Optional[int] = None) -> float:
        """Compute the empirical k-th moment of a variable from circuit samples."""
        circuit = self._require_circuit()
        if variable not in getattr(circuit, "output_vars"):
            raise ModelInferenceError(
                "Variable not found in circuit outputs",
                context={"variable": variable, "available": getattr(circuit, "output_vars")},
            )
        if int(order) < 1:
            raise ReasoningValidationError("Moment order must be >= 1", context={"order": order})

        samples = self._sample_circuit(n_samples or self.default_sample_count)
        values = [float(sample.get(variable, self.default_probability)) for sample in samples]
        return float(np.mean(np.power(values, int(order)))) if values else 0.0

    def kullback_leibler_divergence(self, dist_p: Mapping[Any, Any], dist_q: Mapping[Any, Any]) -> float:
        """Compute KL(P || Q) over shared, sorted support with safe normalization."""
        if not dist_p or not dist_q:
            raise ModelInferenceError("KL divergence requires two non-empty distributions")

        support = sorted(set(dist_p.keys()) | set(dist_q.keys()), key=str)
        p_values = torch.tensor([float(dist_p.get(k, 0.0)) for k in support], dtype=torch.float32, device=self.device)
        q_values = torch.tensor([float(dist_q.get(k, 0.0)) for k in support], dtype=torch.float32, device=self.device)
        p = self._normalize_distribution_tensor(p_values)
        q = self._normalize_distribution_tensor(q_values)
        return float(torch.sum(p * (torch.log(p) - torch.log(q))).item())

    def marginal_map_query(
        self,
        marginal_vars: VariableSpec,
        map_vars: VariableSpec,
        evidence: Optional[Evidence] = None,
    ) -> Tuple[Dict[str, float], float]:
        """Hybrid marginal-MAP inference using bounded alternating refinement."""
        circuit = self._require_circuit()
        marginal = self._normalize_variables(marginal_vars, available=getattr(circuit, "output_vars"))
        map_query = self._normalize_variables(map_vars, available=getattr(circuit, "output_vars"))
        state: Dict[str, float] = dict(self._normalize_evidence(evidence or {}))

        for _ in range(self.marginal_map_steps):
            map_state, _ = self.compute_map_estimate(state, map_query)
            state.update(map_state)
            for var in marginal:
                state[var] = self.compute_marginal_probability((var,), state)

        confidence = self.compute_marginal_probability(map_query, state)
        return {var: clamp_confidence(state[var]) for var in map_query}, confidence

    # ------------------------------------------------------------------
    # Epistemological operations
    # ------------------------------------------------------------------
    def dynamic_model_revision(self, new_evidence: Union[Evidence, Sequence[Evidence]]) -> Dict[str, Any]:
        """Adapt circuit parameters from evidence batches with bounded training."""
        circuit = self._require_circuit()
        if self.optimizer is None:
            raise TrainingError("Cannot perform revision without an initialized optimizer")

        started = time.monotonic()
        try:
            inputs, targets = self._evidence_to_training_data(new_evidence)
            dataset = TensorDataset(inputs, targets)
            loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)
            losses: List[float] = []

            circuit.train()
            for _ in range(self.revision_epochs):
                for batch_inputs, batch_targets in loader:
                    self.optimizer.zero_grad()
                    outputs = self._forward_probabilities(batch_inputs, training=True)
                    loss = self._compute_training_loss(outputs, batch_targets)
                    loss.backward()
                    if self.gradient_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(circuit.parameters(), max_norm=self.gradient_clip_norm)
                    self.optimizer.step()
                    losses.append(float(loss.detach().cpu().item()))

                if self.project_sum_weights or self.enforce_non_negative_parameters:
                    self.enforce_circuit_constraints()

            self.schema_version = round(self.schema_version + self.schema_increment, 3)
            summary = json_safe_reasoning_state({
                "schema_version": self.schema_version,
                "cases": len(dataset),
                "epochs": self.revision_epochs,
                "loss_start": losses[0] if losses else None,
                "loss_end": losses[-1] if losses else None,
                "elapsed_seconds": elapsed_seconds(started),
                "timestamp_ms": monotonic_timestamp_ms(),
            })
            self.revision_history.append(summary)
            printer.status("Revision", f"Schema updated to V.{self.schema_version}", "success")
            return summary
        except ReasoningError:
            raise
        except Exception as exc:
            raise TrainingError("Dynamic model revision failed", cause=exc) from exc
        finally:
            circuit.eval()

    def schema_equilibration(self) -> Dict[str, float]:
        """Maintain cognitive consistency through finite-value checks and L1 pressure."""
        circuit = self._require_circuit()
        if self.optimizer is None:
            raise TrainingError("Cannot equilibrate schema without an initialized optimizer")

        self.optimizer.zero_grad()
        reg_terms = [param.abs().mean() for param in circuit.parameters() if param.requires_grad]
        if not reg_terms:
            return {"regularization_loss": 0.0}

        regularization_loss = torch.stack(reg_terms).mean() * max(self.l1_regularization, 0.0)
        regularization_loss.backward()
        self.optimizer.step()
        self.enforce_circuit_constraints()
        return {"regularization_loss": float(regularization_loss.detach().cpu().item())}

    # ------------------------------------------------------------------
    # Circuit structural operations
    # ------------------------------------------------------------------
    def check_decomposability(self) -> bool:
        """Verify decomposability where ProductNode metadata is available."""
        circuit = self._require_circuit()
        product_nodes = [module for module in circuit.modules() if isinstance(module, ProductNode)]
        for node in product_nodes:
            if hasattr(node, "is_decomposable") and not node.is_decomposable():
                return False
        return True

    def check_smoothness(self) -> bool:
        """Verify smoothness where SumNode child scope metadata is available."""
        circuit = self._require_circuit()
        for node in circuit.modules():
            if not isinstance(node, SumNode):
                continue
            scopes = [scope for scope in getattr(node, "_scopes", []) if scope]
            if scopes and any(scope != scopes[0] for scope in scopes[1:]):
                return False
        return True

    def check_determinism(self, n_samples: Optional[int] = None) -> bool:
        """Check whether sum nodes are effectively deterministic under current weights."""
        circuit = self._require_circuit()
        sum_nodes = [module for module in circuit.modules() if isinstance(module, SumNode)]
        if not sum_nodes:
            return True

        for node in sum_nodes:
            if hasattr(node, "normalized_weights"):
                weights = node.normalized_weights().detach()
            else:
                weights = torch.softmax(torch.as_tensor(getattr(node, "weights"), dtype=torch.float32), dim=0)
            if float(torch.max(weights).item()) < self.determinism_threshold:
                return False
        return True

    def enforce_circuit_constraints(self) -> Dict[str, int]:
        """Apply safe parameter projections without destroying neural expressivity."""
        circuit = self._require_circuit()
        corrected_parameters = 0
        projected_sum_nodes = 0

        with torch.no_grad():
            for node in circuit.modules():
                if isinstance(node, SumNode) and hasattr(node, "weights"):
                    weights = getattr(node, "weights")
                    if not torch.isfinite(weights).all():
                        weights.copy_(torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=0.0))
                        corrected_parameters += 1
                    if self.project_sum_weights:
                        projected = torch.clamp(weights, min=self.epsilon)
                        projected = projected / torch.clamp(projected.sum(), min=self.epsilon)
                        weights.copy_(projected)
                        projected_sum_nodes += 1

            for param in circuit.parameters():
                if not torch.isfinite(param).all():
                    param.copy_(torch.nan_to_num(param, nan=0.0, posinf=1.0, neginf=-1.0))
                    corrected_parameters += 1
                if self.enforce_non_negative_parameters:
                    param.clamp_(min=0.0)

        return {
            "corrected_parameters": corrected_parameters,
            "projected_sum_nodes": projected_sum_nodes,
        }

    # ------------------------------------------------------------------
    # Interpretability and explainability
    # ------------------------------------------------------------------
    def explain_inference_path(self, query: str, evidence: Optional[Evidence] = None) -> str:
        """Generate a compact human-readable explanation for a circuit query."""
        circuit = self._require_circuit()
        evidence_map = self._normalize_evidence(evidence or {})
        input_tensor = self._evidence_to_tensor(evidence_map)
        confidence = self.compute_marginal_probability((query,), evidence_map)

        lines = [
            f"Inference for '{query}' given evidence:",
            f"- evidence variables: {sorted(evidence_map.keys()) if evidence_map else 'none'}",
        ]

        if hasattr(circuit, "trace_activations"):
            try:
                trace = circuit.trace_activations(input_tensor)
                for item in self._flatten_trace(trace)[: self.max_explanation_items]:
                    lines.append(f"- {item}")
            except Exception as exc:
                logger.warning(f"Activation trace unavailable: {exc}")
                lines.append("- activation trace unavailable for this circuit")
        else:
            lines.append("- circuit does not expose trace_activations")

        lines.append(f"Conclusion: P({query}|evidence) = {confidence:.3f}")
        explanation = "\n".join(lines)
        printer.code_block(explanation, language="text")
        return explanation

    def introspect_model_biases(self) -> Dict[str, float]:
        """Quantify parameter bias and schema-rigidity diagnostics."""
        circuit = self._require_circuit()
        printer.section_header("Model Bias Introspection")
        diagnostics: Dict[str, float] = {}
        for name, param in circuit.named_parameters():
            if "bias" in name:
                diagnostics[f"{name}.mean"] = float(param.detach().mean().cpu().item())
                diagnostics[f"{name}.std"] = float(param.detach().std(unbiased=False).cpu().item())

        diagnostics["schema_rigidity"] = self._calculate_schema_rigidity()
        diagnostics["revision_count"] = float(len(self.revision_history))

        rows = [[name, f"{value:.6f}"] for name, value in sorted(diagnostics.items())]
        printer.table(["Metric", "Value"], rows, "Model Biases")
        return diagnostics

    def simulate_alternative_hypotheses(self, evidence: Evidence) -> Dict[str, float]:
        """Generate counterfactual deltas by perturbing each evidence variable."""
        evidence_map = self._normalize_evidence(evidence)
        if not evidence_map:
            return {}
        query_vars = tuple(var for var in evidence_map if var in getattr(self._require_circuit(), "output_vars"))
        if not query_vars:
            query_vars = tuple(getattr(self._require_circuit(), "output_vars"))

        base_probability = self.compute_marginal_probability(query_vars, evidence_map)
        hypotheses: Dict[str, float] = {}
        for var, value in evidence_map.items():
            counterfactual = dict(evidence_map)
            counterfactual[var] = 1.0 - float(value)
            counterfactual_probability = self.compute_marginal_probability(query_vars, counterfactual)
            hypotheses[var] = float(counterfactual_probability - base_probability)
        return hypotheses

    def belief_revision_trace(self, fact: Tuple[Any, ...], updates: Sequence[Evidence]) -> List[float]:
        """Track belief evolution through sequential revision updates."""
        if not fact:
            raise ReasoningValidationError("fact cannot be empty for belief tracing")
        query = str(fact[0])
        if query not in getattr(self._require_circuit(), "output_vars"):
            raise ModelInferenceError(
                "Belief trace query is not a circuit output",
                context={"query": query, "fact": fact},
            )

        printer.section_header(f"Belief Revision Trace:\n{fact}\n{updates}")
        trace: List[float] = []
        for update in updates:
            self.dynamic_model_revision(update)
            trace.append(self.compute_marginal_probability((query,), update))
        self.belief_history[fact] = trace
        return trace

    def to_state_dict_summary(self) -> Dict[str, Any]:
        """Serializable state summary for diagnostics and checkpoints."""
        circuit = self._require_circuit()
        return json_safe_reasoning_state({
            "component": "ModelCompute",
            "schema_version": self.schema_version,
            "device": str(self.device),
            "optimizer": self.optimizer_name if self.optimizer is not None else None,
            "loss": self.loss_name,
            "input_vars": list(getattr(circuit, "input_vars")),
            "output_vars": list(getattr(circuit, "output_vars")),
            "revision_count": len(self.revision_history),
            "decomposable": self.check_decomposability(),
            "smooth": self.check_smoothness(),
            "deterministic": self.check_determinism(),
            "timestamp_ms": monotonic_timestamp_ms(),
        })

    # ------------------------------------------------------------------
    # Internal utility methods
    # ------------------------------------------------------------------
    def _compute_training_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(self.loss_fn, nn.KLDivLoss):
            target_dist = self._normalize_distribution_tensor(targets)
            output_dist = self._normalize_distribution_tensor(outputs)
            return self.loss_fn(torch.log(output_dist), target_dist)
        return self.loss_fn(outputs, targets)

    def _normalize_variables(self, variables: VariableSpec, *, available: Sequence[str]) -> Tuple[str, ...]:
        if isinstance(variables, str):
            normalized = (variables,)
        else:
            normalized = tuple(str(v) for v in variables)
        if not normalized:
            raise ReasoningValidationError("At least one variable is required")
        invalid = [var for var in normalized if var not in available]
        if invalid:
            raise ModelInferenceError(
                "Variables not found in circuit outputs",
                context={"invalid": invalid, "available": list(available)},
            )
        return normalized

    def _normalize_evidence(self, evidence: Mapping[Any, Any]) -> Dict[str, float]:
        if evidence is None:
            return {}
        if not isinstance(evidence, Mapping):
            raise ReasoningValidationError(
                "Evidence must be a mapping",
                context={"actual_type": type(evidence).__name__},
            )

        normalized: Dict[str, float] = {}
        valid_inputs = set(getattr(self._require_circuit(), "input_vars", [])) if self.circuit is not None else set()
        for key, value in evidence.items():
            variable = self._evidence_key_to_variable(key)
            if valid_inputs and variable not in valid_inputs:
                logger.debug(f"Ignoring evidence variable not used by circuit: {variable}")
                continue
            normalized[variable] = self._coerce_probability_value(value, label=variable)
        return normalized

    def _evidence_key_to_variable(self, key: Any) -> str:
        if isinstance(key, tuple) and key:
            return str(key[0])
        return str(key)

    def _coerce_probability_value(self, value: Any, *, label: str) -> float:
        if isinstance(value, bool):
            numeric = 1.0 if value else 0.0
        elif isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ConfidenceBoundsError("Evidence tensor values must be scalar", context={"label": label})
            numeric = float(value.detach().cpu().item())
        elif isinstance(value, np.generic):
            numeric = float(value.item())
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            numeric = 1.0 if value.strip().lower() == "true" else 0.0
        else:
            numeric = float(value)

        if self.strict_evidence_bounds:
            return assert_confidence(numeric, label=f"evidence.{label}")
        return clamp_confidence(numeric)

    def _evidence_to_tensor(self, evidence: Mapping[str, Any]) -> torch.Tensor:
        circuit = self._require_circuit()
        evidence_map = evidence if all(isinstance(k, str) for k in evidence.keys()) else self._normalize_evidence(evidence)
        values = [
            self._coerce_probability_value(evidence_map.get(var, self.default_probability), label=var)
            for var in getattr(circuit, "input_vars")
        ]
        return torch.tensor([values], dtype=torch.float32, device=self.device)

    def _evidence_to_training_data(self, evidence: Union[Evidence, Sequence[Evidence]]) -> Tuple[torch.Tensor, torch.Tensor]:
        circuit = self._require_circuit()
        cases: Sequence[Evidence]
        if isinstance(evidence, Mapping):
            cases = [evidence]
        elif isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
            cases = list(evidence)
        else:
            raise ReasoningValidationError(
                "Training evidence must be a mapping or sequence of mappings",
                context={"actual_type": type(evidence).__name__},
            )

        if not cases:
            raise TrainingError("No evidence cases provided for revision")

        inputs: List[List[float]] = []
        targets: List[List[float]] = []
        for case in cases:
            normalized = self._normalize_evidence(case)
            inputs.append([
                normalized.get(var, self.default_probability)
                for var in getattr(circuit, "input_vars")
            ])
            targets.append([
                normalized.get(var, self.default_probability)
                for var in getattr(circuit, "output_vars")
            ])

        return (
            torch.tensor(inputs, dtype=torch.float32, device=self.device),
            torch.tensor(targets, dtype=torch.float32, device=self.device),
        )

    def _forward_probabilities(self, input_tensor: torch.Tensor, *, training: bool) -> torch.Tensor:
        circuit = self._require_circuit()
        if input_tensor.dim() != 2 or input_tensor.size(1) != len(getattr(circuit, "input_vars")):
            raise ModelInferenceError(
                "Input tensor shape does not match circuit input variables",
                context={"shape": tuple(input_tensor.shape), "expected_features": len(getattr(circuit, "input_vars"))},
            )

        output = circuit(input_tensor.to(self.device))
        if not isinstance(output, torch.Tensor):
            output = torch.as_tensor(output, dtype=torch.float32, device=self.device)
        output = output.to(self.device, dtype=torch.float32)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        if output.dim() != 2:
            raise ModelInferenceError(
                "Circuit output must be a 1D or 2D tensor",
                context={"shape": tuple(output.shape)},
            )
        if output.size(1) < len(getattr(circuit, "output_vars")):
            raise ModelInferenceError(
                "Circuit output width is smaller than output_vars",
                context={"output_width": output.size(1), "output_vars": len(getattr(circuit, "output_vars"))},
            )

        output = output[:, : len(getattr(circuit, "output_vars"))]
        if self.auto_sigmoid_outputs and (float(output.detach().min().item()) < 0.0 or float(output.detach().max().item()) > 1.0):
            output = torch.sigmoid(output)
        return torch.clamp(output, min=self.epsilon, max=1.0 - self.epsilon)

    def _tensor_to_state(self, probs: torch.Tensor) -> Dict[str, float]:
        circuit = self._require_circuit()
        if probs.dim() == 2:
            probs = probs[0]
        return {
            var: clamp_confidence(float(probs[i].detach().cpu().item()))
            for i, var in enumerate(getattr(circuit, "output_vars"))
        }

    def _output_index(self, variable: str) -> int:
        circuit = self._require_circuit()
        try:
            return list(getattr(circuit, "output_vars")).index(variable)
        except ValueError as exc:
            raise ModelInferenceError(
                "Variable not found in circuit outputs",
                cause=exc,
                context={"variable": variable, "available": getattr(circuit, "output_vars")},
            ) from exc

    def _output_indices(self, variables: Sequence[str]) -> List[int]:
        return [self._output_index(var) for var in variables]

    def _normalize_distribution_tensor(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.clamp(values.to(self.device, dtype=torch.float32), min=self.epsilon)
        denom = values.sum(dim=-1, keepdim=True) if values.dim() > 1 else values.sum()
        return values / torch.clamp(denom, min=self.epsilon)

    def _sample_circuit(self, n_samples: int) -> List[Dict[str, float]]:
        circuit = self._require_circuit()
        sample_count = bounded_iterations(n_samples, minimum=1, maximum=250_000)
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        if hasattr(circuit, "sample") and callable(getattr(circuit, "sample")):
            raw_samples = circuit.sample(sample_count)  # type: ignore[misc]
            return [self._normalize_sample(sample) for sample in raw_samples]

        if hasattr(circuit, "to_evidence_dict") and callable(getattr(circuit, "to_evidence_dict")):
            return [self._normalize_sample(circuit.to_evidence_dict()) for _ in range(sample_count)]  # type: ignore[misc]

        random_inputs = torch.rand(
            sample_count,
            len(getattr(circuit, "input_vars")),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            probs = self._forward_probabilities(random_inputs, training=False)
        return [self._tensor_to_state(probs[i]) for i in range(sample_count)]

    def _normalize_sample(self, sample: Mapping[Any, Any]) -> Dict[str, float]:
        if not isinstance(sample, Mapping):
            raise ModelInferenceError(
                "Circuit sample must be a mapping",
                context={"actual_type": type(sample).__name__},
            )
        circuit = self._require_circuit()
        return {
            var: self._coerce_probability_value(sample.get(var, self.default_probability), label=var)
            for var in getattr(circuit, "output_vars")
        }

    def _calculate_schema_rigidity(self) -> float:
        if not self.belief_history:
            return 0.0
        deviations = [float(np.std(trace)) for trace in self.belief_history.values() if trace]
        return float(np.mean(deviations)) if deviations else 0.0

    def _flatten_trace(self, trace: Any, *, prefix: str = "trace") -> List[str]:
        items: List[str] = []
        if isinstance(trace, Mapping):
            for key, value in trace.items():
                child_prefix = f"{prefix}.{key}"
                if isinstance(value, torch.Tensor):
                    items.append(f"{child_prefix}: tensor shape={tuple(value.shape)} mean={float(value.float().mean().item()):.4f}")
                elif isinstance(value, Mapping):
                    items.extend(self._flatten_trace(value, prefix=child_prefix))
                elif isinstance(value, list):
                    items.append(f"{child_prefix}: list[{len(value)}]")
                    for idx, child in enumerate(value[:3]):
                        items.extend(self._flatten_trace(child, prefix=f"{child_prefix}[{idx}]"))
                elif isinstance(value, (int, float, str, bool)):
                    items.append(f"{child_prefix}: {value}")
        else:
            items.append(f"{prefix}: {trace}")
        return items


if __name__ == "__main__":
    print("\n=== Running Model Compute ===\n")
    printer.status("TEST", "Model Compute initialized", "info")

    from .adaptive_circuit import AdaptiveCircuit

    test_network = {
        "nodes": ["A", "B", "C"],
        "edges": [["A", "B"], ["B", "C"]],
        "cpt": {
            "A": {"prior": 0.65, "confidence": 0.9},
            "B": {"true": 0.75, "false": 0.25, "confidence": 0.8},
            "C": {"true": 0.70, "false": 0.30, "confidence": 0.8},
        },
    }
    test_knowledge = {
        ("A", "supports", "B"): 0.85,
        ("B", "supports", "C"): 0.80,
        ("C", "explains", "Outcome"): 0.75,
    }

    circuit = AdaptiveCircuit(network_structure=test_network, knowledge_base=test_knowledge) # type: ignore
    compute = ModelCompute(circuit=circuit)
    compute.eval()
    # Keep the self-test bounded while preserving production defaults in config.
    compute.map_steps = 8
    compute.marginal_map_steps = 3
    compute.revision_epochs = 2

    evidence = {"A": 1.0, "B": 0.5}

    printer.status("TEST", "1/9 marginal probability", "info")
    marginal = compute.compute_marginal_probability(("C",), evidence)
    assert 0.0 <= marginal <= 1.0
    printer.status("PASS", f"P(C|evidence)={marginal:.4f}", "success")

    printer.status("TEST", "2/9 all marginals", "info")
    marginals = compute.compute_all_marginals(evidence)
    assert set(marginals) == {"A", "B", "C"}
    assert all(0.0 <= value <= 1.0 for value in marginals.values())
    printer.status("PASS", f"marginals={marginals}", "success")

    printer.status("TEST", "3/9 MAP estimate", "info")
    map_state, map_confidence = compute.compute_map_estimate(evidence, variables=("B", "C"))
    assert set(map_state) == {"B", "C"}
    assert 0.0 <= map_confidence <= 1.0
    printer.status("PASS", f"MAP={map_state}, confidence={map_confidence:.4f}", "success")

    printer.status("TEST", "4/9 moment + KL divergence", "info")
    moment = compute.compute_moment("C", order=2, n_samples=64)
    kl = compute.kullback_leibler_divergence({"true": 0.7, "false": 0.3}, {"true": 0.6, "false": 0.4})
    assert moment >= 0.0
    assert kl >= 0.0
    printer.status("PASS", f"moment={moment:.4f}, kl={kl:.4f}", "success")

    printer.status("TEST", "5/9 marginal-MAP query", "info")
    mm_state, mm_confidence = compute.marginal_map_query(("C",), ("B",), evidence)
    assert "B" in mm_state
    assert 0.0 <= mm_confidence <= 1.0
    printer.status("PASS", f"state={mm_state}, confidence={mm_confidence:.4f}", "success")

    printer.status("TEST", "6/9 dynamic revision", "info")
    revision = compute.dynamic_model_revision([
        {"A": 1.0, "B": 0.9, "C": 0.8},
        {"A": 0.0, "B": 0.2, "C": 0.1},
    ])
    assert revision["cases"] == 2
    printer.status("PASS", f"revision={revision}", "success")

    printer.status("TEST", "7/9 structural checks", "info")
    constraints = compute.enforce_circuit_constraints()
    assert compute.check_decomposability() is True
    assert compute.check_smoothness() is True
    assert compute.check_determinism() is True
    printer.status("PASS", f"constraints={constraints}", "success")

    printer.status("TEST", "8/9 explainability and bias diagnostics", "info")
    explanation = compute.explain_inference_path("C", evidence)
    biases = compute.introspect_model_biases()
    assert "Conclusion" in explanation
    assert "schema_rigidity" in biases
    printer.status("PASS", "explainability diagnostics completed", "success")

    printer.status("TEST", "9/9 counterfactuals, belief trace, summary", "info")
    counterfactuals = compute.simulate_alternative_hypotheses(evidence)
    trace = compute.belief_revision_trace(("C", "belief", "True"), [{"A": 1.0, "C": 0.9}])
    summary = compute.to_state_dict_summary()
    assert counterfactuals
    assert len(trace) == 1
    assert summary["component"] == "ModelCompute"
    printer.status("PASS", "counterfactuals, trace, and summary completed", "success")

    print("\n=== Test ran successfully ===\n")
