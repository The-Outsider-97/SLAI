import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

from typing import List, Dict

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.reasoning_errors import *
from ..utils.reasoning_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Nodes")
printer = PrettyPrinter()


_LOG_ZERO: float = -1e38          # log-domain floor to avoid -inf
_WEIGHT_FLOOR: float = 1e-8       # minimum soft weight before softmax
_EPS: float = 1e-9                # general floating-point epsilon

# ---------------------------------------------------------------------------
# Scope protocol
# ---------------------------------------------------------------------------
class ScopedModule(nn.Module):
    """Abstract base that every SPN node must satisfy.
 
    Concrete subclasses implement:
    - ``compute_scope() -> set`` — variable set this node is defined over.
    - ``forward(x) -> Tensor``   — log-space or linear-space density evaluation.
    - ``trace_activations(x)``   — structured activation map for debugging.
    """
 
    def compute_scope(self) -> set:  # noqa: D102
        raise NotImplementedError(f"{self.__class__.__name__} must implement compute_scope()")
 
    def trace_activations(self, x: torch.Tensor, _depth: int = 0, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement trace_activations()")
 
 
# ---------------------------------------------------------------------------
# SumNode
# ---------------------------------------------------------------------------
class SumNode(ScopedModule):
    """
    Weighted sum (mixture) node for tractable probabilistic circuits.
 
    Computes a convex combination of child distributions:
 
        p(x) = Σ_k  w_k · p_k(x),   Σ_k w_k = 1,  w_k > 0
 
    where weights are stored as unconstrained parameters and projected via
    softmax during the forward pass, ensuring the sum-to-one constraint
    holds throughout training without auxiliary projection steps.
 
    Smoothness is validated at construction: every child's scope must be
    identical (or fully determined), so the mixture is well-defined.
 
    Configuration keys (``sum_nodes`` section of ``reasoning_config.yaml``):
    - ``weight_init``       : ``"uniform"`` | ``"random"``  (default ``"uniform"``)
    - ``log_space``         : bool  – operate in log domain  (default ``False``)
    - ``temperature``       : float – softmax temperature     (default ``1.0``)
    - ``min_children``      : int   – minimum child count     (default ``2``)
    - ``max_children``      : int   – maximum child count     (default ``512``)
    - ``enforce_smoothness``: bool  – validate scope equality (default ``True``)
    - ``activation_depth``  : int   – max recursion depth for trace (default ``8``)
    """

    def __init__(self, children: List[nn.Module]) -> None:
        super().__init__()

        # ---- Configuration --------------------------------------------------
        self.config: Dict[str, Any] = load_global_config()
        self.sum_cfg: Dict[str, Any] = get_config_section("sum_nodes", self.config)

        weight_init: str       = str(self.sum_cfg.get("weight_init", "uniform")).lower()
        self._log_space: bool  = bool(self.sum_cfg.get("log_space", False))
        self._temperature: float = float(self.sum_cfg.get("temperature", 1.0))
        min_children: int      = bounded_iterations(self.sum_cfg.get("min_children", 2), minimum=1, maximum=4096)
        max_children: int      = bounded_iterations(self.sum_cfg.get("max_children", 512), minimum=1, maximum=4096)
        enforce_smooth: bool   = bool(self.sum_cfg.get("enforce_smoothness", True))
        self._max_trace_depth: int = bounded_iterations(self.sum_cfg.get("activation_depth", 8), minimum=1, maximum=32)

        # ---- Child validation -----------------------------------------------
        if not children:
            raise ModelInitializationError(
                "SumNode requires at least one child module",
                context={"min_children": min_children},
            )
        n = len(children)
        if n < min_children:
            raise ModelInitializationError(
                f"SumNode requires at least {min_children} children, got {n}",
                context={"min_children": min_children, "actual": n},
            )
        if n > max_children:
            raise ModelInitializationError(
                f"SumNode exceeds maximum child count of {max_children}, got {n}",
                context={"max_children": max_children, "actual": n},
            )

        self.children_modules: nn.ModuleList = nn.ModuleList(children)

        # ---- Weight initialisation ------------------------------------------
        if weight_init == "random":
            raw = torch.rand(n)
        else:  # uniform
            raw = torch.ones(n)
        self.weights: nn.Parameter = nn.Parameter(raw)

        # ---- Smoothness check -----------------------------------------------
        self._scopes: List[set] = [
            child.compute_scope() if isinstance(child, ScopedModule) else set()
            for child in children
        ]
        if enforce_smooth:
            self._validate_smoothness()

        logger.debug(
            f"SumNode initialised | n_children={n} | log_space={self._log_space} "
            f"| temperature={self._temperature}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
 
    def compute_scope(self) -> set:
        """Scope is the union of children scopes (identical when smooth)."""
        scope: set = set()
        for s in self._scopes:
            scope |= s
        return scope
 
    def normalized_weights(self) -> torch.Tensor:
        """Return softmax-normalized weights with temperature scaling."""
        if self._temperature <= 0:
            raise ReasoningConfigurationError(
                "SumNode temperature must be positive",
                context={"temperature": self._temperature},
            )
        return F.softmax(self.weights / self._temperature, dim=0)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum (or log-sum-exp) over children.
 
        Args:
            x: Input tensor of shape ``(..., feature_dim)``.
 
        Returns:
            Mixture output tensor matching the leading shape of ``x``.
        """
        w = self.normalized_weights()                          # (n,)
        outputs: List[torch.Tensor] = [
            child(x) for child in self.children_modules       # type: ignore[arg-type]
        ]
 
        if self._log_space:
            return self._log_weighted_sum(w, outputs)
        return self._linear_weighted_sum(w, outputs)
 
    def trace_activations(self, x: torch.Tensor, *, _depth: int = 0, **kwargs: Any) -> Dict[str, Any]: # type: ignore
        """Structured activation map for circuit debugging and visualization.
 
        Args:
            x:      Input tensor.
            _depth: Internal recursion depth guard (do not set externally).
 
        Returns:
            Nested dict with keys ``output``, ``weights``, ``children``.
        """
        if _depth > self._max_trace_depth:
            return {"truncated": True, "depth": _depth}
 
        w = self.normalized_weights().detach()
        child_traces: List[Dict[str, Any]] = []
        for i, child in enumerate(self.children_modules):     # type: ignore[arg-type]
            entry: Dict[str, Any] = {"child_index": i, "weight": float(w[i])}
            if isinstance(child, ScopedModule) and _depth < self._max_trace_depth:
                entry.update(child.trace_activations(x, _depth=_depth + 1))
            else:
                with torch.no_grad():
                    entry["output"] = child(x).detach()       # type: ignore[arg-type]
            child_traces.append(entry)
 
        with torch.no_grad():
            out = self.forward(x).detach()
 
        return {
            "node_type": "SumNode",
            "n_children": len(self.children_modules),
            "temperature": self._temperature,
            "log_space": self._log_space,
            "weights": w.tolist(),
            "output": out,
            "children": child_traces,
        }
 
    def weight_entropy(self) -> float:
        """Shannon entropy of the mixture weights — useful as a regularizer signal."""
        w = self.normalized_weights().detach()
        ent = float(-torch.sum(w * torch.clamp(torch.log(w), min=_LOG_ZERO)))
        return ent
 
    def effective_children(self, *, threshold: float = 0.01) -> int:
        """Number of children with weight above ``threshold`` after normalization."""
        assert_confidence(threshold, label="effective_children threshold")
        w = self.normalized_weights().detach()
        return int((w > threshold).sum().item())
 
    def to_state_dict_summary(self) -> Dict[str, Any]:
        """Serializable summary for logging/checkpointing without tensors."""
        return json_safe_reasoning_state({
            "node_type": "SumNode",
            "n_children": len(self.children_modules),
            "log_space": self._log_space,
            "temperature": self._temperature,
            "weight_entropy": self.weight_entropy(),
            "effective_children": self.effective_children(),
            "scope_size": len(self.compute_scope()),
            "timestamp_ms": monotonic_timestamp_ms(),
        })
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    def _linear_weighted_sum(
        self,
        w: torch.Tensor,
        outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Numerically stable linear weighted sum."""
        result = torch.zeros_like(outputs[0])
        for wi, out in zip(w, outputs):
            result = result + wi * out
        return result
 
    def _log_weighted_sum(
        self,
        w: torch.Tensor,
        outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Log-space weighted sum via log-sum-exp for numerical stability.
 
        Computes: log Σ_k exp(log w_k + log_child_k(x))
        """
        log_w = torch.log(torch.clamp(w, min=_WEIGHT_FLOOR))  # (n,)
        # stack: (n, *output_shape)
        stacked = torch.stack(outputs, dim=0)
        log_terms = log_w.view((-1,) + (1,) * (stacked.dim() - 1)) + stacked
        return torch.logsumexp(log_terms, dim=0)
 
    def _validate_smoothness(self) -> None:
        """Verify all children share the same scope (smoothness constraint)."""
        non_empty = [s for s in self._scopes if s]
        if not non_empty:
            return  # leaf children without scopes are exempt (e.g. nn.Linear)
        reference = non_empty[0]
        for i, scope in enumerate(non_empty[1:], start=1):
            if scope != reference:
                raise CircuitConstraintError(
                    "SumNode violates smoothness: children have differing scopes",
                    context={
                        "child_0_scope": sorted(reference),
                        f"child_{i}_scope": sorted(scope),
                        "symmetric_diff": sorted(reference.symmetric_difference(scope)),
                    },
                )
 
    def _register_constraints(self) -> None:
        """No-op registration hook; kept for API compatibility with external callers."""
        # Weight normalization is handled dynamically in normalized_weights().
        pass
 
 
# ---------------------------------------------------------------------------
# ProductNode
# ---------------------------------------------------------------------------
class ProductNode(ScopedModule):
    """Product node for factorized distributions over disjoint variable scopes.
 
    Computes the product of child densities:
 
        p(x) = Π_k  p_k(x_{S_k}),   S_i ∩ S_j = ∅  for i ≠ j
 
    Decomposability (pairwise disjoint scopes) is enforced at construction.
    In log-space mode the product becomes a sum of log-densities, which is
    both numerically stable and gradient-friendly for deep circuits.
 
    Configuration keys (``product_nodes`` section of ``reasoning_config.yaml``):
    - ``log_space``          : bool  – operate in log domain   (default ``False``)
    - ``min_children``       : int   – minimum child count      (default ``2``)
    - ``max_children``       : int   – maximum child count      (default ``512``)
    - ``strict_decomposable``: bool  – abort on scope overlap   (default ``True``)
    - ``activation_depth``   : int   – max trace recursion      (default ``8``)
    """
 
    def __init__(self, children: List[nn.Module]) -> None:
        super().__init__()
 
        # ---- Configuration --------------------------------------------------
        self.config: Dict[str, Any] = load_global_config()
        self.product_cfg: Dict[str, Any] = get_config_section("product_nodes", self.config)
 
        self._log_space: bool       = bool(self.product_cfg.get("log_space", False))
        min_children: int           = bounded_iterations(self.product_cfg.get("min_children", 2), minimum=1, maximum=4096)
        max_children: int           = bounded_iterations(self.product_cfg.get("max_children", 512), minimum=1, maximum=4096)
        self._strict: bool          = bool(self.product_cfg.get("strict_decomposable", True))
        self._max_trace_depth: int  = bounded_iterations(self.product_cfg.get("activation_depth", 8), minimum=1, maximum=32)
 
        # ---- Child validation -----------------------------------------------
        if not children:
            raise ModelInitializationError(
                "ProductNode requires at least one child module",
                context={"min_children": min_children},
            )
        n = len(children)
        if n < min_children:
            raise ModelInitializationError(
                f"ProductNode requires at least {min_children} children, got {n}",
                context={"min_children": min_children, "actual": n},
            )
        if n > max_children:
            raise ModelInitializationError(
                f"ProductNode exceeds maximum child count of {max_children}, got {n}",
                context={"max_children": max_children, "actual": n},
            )
 
        self.children_modules: nn.ModuleList = nn.ModuleList(children)
 
        # ---- Scope and decomposability --------------------------------------
        self._scopes: List[set] = [
            child.compute_scope() if isinstance(child, ScopedModule) else set()
            for child in children
        ]
        self._validate_decomposability()
 
        logger.debug(
            f"ProductNode initialised | n_children={n} | log_space={self._log_space}"
        )
 
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
 
    def compute_scope(self) -> set:
        """Scope is the union of all children scopes (disjoint by construction)."""
        scope: set = set()
        for s in self._scopes:
            scope |= s
        return scope
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute element-wise product (or sum in log domain) of children.
 
        Args:
            x: Input tensor of shape ``(..., feature_dim)``.
 
        Returns:
            Factorized output tensor matching the leading shape of ``x``.
        """
        outputs: List[torch.Tensor] = [
            child(x) for child in self.children_modules       # type: ignore[arg-type]
        ]
 
        if self._log_space:
            return self._log_product(outputs)
        return self._linear_product(outputs)
 
    def trace_activations(self, x: torch.Tensor, *, _depth: int = 0, **kwargs: Any) -> Dict[str, Any]: # type: ignore
        """Structured activation map for circuit debugging.
 
        Args:
            x:      Input tensor.
            _depth: Internal recursion depth guard (do not set externally).
 
        Returns:
            Nested dict with keys ``output``, ``scope``, ``children``.
        """
        if _depth > self._max_trace_depth:
            return {"truncated": True, "depth": _depth}
 
        child_traces: List[Dict[str, Any]] = []
        for i, (child, scope) in enumerate(
            zip(self.children_modules, self._scopes)           # type: ignore[arg-type]
        ):
            entry: Dict[str, Any] = {
                "child_index": i,
                "scope": sorted(scope),
            }
            if isinstance(child, ScopedModule) and _depth < self._max_trace_depth:
                entry.update(child.trace_activations(x, _depth=_depth + 1))
            else:
                with torch.no_grad():
                    entry["output"] = child(x).detach()        # type: ignore[arg-type]
            child_traces.append(entry)
 
        with torch.no_grad():
            out = self.forward(x).detach()
 
        return {
            "node_type": "ProductNode",
            "n_children": len(self.children_modules),
            "log_space": self._log_space,
            "scope": sorted(self.compute_scope()),
            "output": out,
            "children": child_traces,
        }
 
    def scope_sizes(self) -> List[int]:
        """Return the variable count per child — useful for capacity analysis."""
        return [len(s) for s in self._scopes]
 
    def is_decomposable(self) -> bool:
        """Runtime decomposability check (fast, non-raising)."""
        seen: set = set()
        for scope in self._scopes:
            if scope & seen:
                return False
            seen |= scope
        return True
 
    def to_state_dict_summary(self) -> Dict[str, Any]:
        """Serializable summary for logging/checkpointing without tensors."""
        return json_safe_reasoning_state({
            "node_type": "ProductNode",
            "n_children": len(self.children_modules),
            "log_space": self._log_space,
            "scope_sizes": self.scope_sizes(),
            "total_scope": len(self.compute_scope()),
            "decomposable": self.is_decomposable(),
            "timestamp_ms": monotonic_timestamp_ms(),
        })
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    def _linear_product(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Element-wise product of output tensors."""
        result = outputs[0]
        for out in outputs[1:]:
            result = result * out
        return result
 
    def _log_product(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Sum in log domain (equivalent to product of densities)."""
        result = outputs[0]
        for out in outputs[1:]:
            result = result + out
        return result
 
    def _validate_decomposability(self) -> None:
        """Verify pairwise disjoint children scopes (decomposability constraint).
 
        Non-empty scopes only: leaf nn.Module children without scopes are exempt.
        In non-strict mode overlapping scopes emit a warning rather than raising.
        """
        non_empty = [(i, s) for i, s in enumerate(self._scopes) if s]
        violations: List[str] = []
 
        for idx_i in range(len(non_empty)):
            i, scope_i = non_empty[idx_i]
            for idx_j in range(idx_i + 1, len(non_empty)):
                j, scope_j = non_empty[idx_j]
                overlap = scope_i & scope_j
                if overlap:
                    msg = (
                        f"Child {i} scope {sorted(scope_i)} overlaps with "
                        f"child {j} scope {sorted(scope_j)}: {sorted(overlap)}"
                    )
                    violations.append(msg)
 
        if violations:
            detail = "; ".join(violations)
            if self._strict:
                raise CircuitConstraintError(
                    "ProductNode violates decomposability: overlapping child scopes",
                    context={"violations": violations},
                )
            logger.warning(
                f"ProductNode decomposability warning (strict=False): {detail}"
            )
 
 
# ---------------------------------------------------------------------------
# Utility: build a balanced SPN circuit from leaf modules
# ---------------------------------------------------------------------------
def build_spn_circuit(
    leaves: List[nn.Module],
    *,
    sum_branching: int = 2,
    product_branching: int = 2,
    log_space: bool = False,
) -> nn.Module:
    """Recursively assemble a balanced SPN from a flat list of leaf nodes.
 
    The resulting circuit alternates Sum and Product layers bottom-up until
    a single root node is produced.  Useful for testing and prototyping.
 
    Args:
        leaves:           Leaf nn.Module list (ScopedModule encouraged).
        sum_branching:    How many children per SumNode.
        product_branching: How many children per ProductNode.
        log_space:        Propagate log-space mode to all nodes.
 
    Returns:
        Root ``SumNode`` or ``ProductNode`` (or a single leaf if ``len(leaves)==1``).
 
    Raises:
        ModelInitializationError: If leaves list is empty.
    """
    if not leaves:
        raise ModelInitializationError(
            "build_spn_circuit requires at least one leaf",
            context={"leaves": 0},
        )
    if len(leaves) == 1:
        return leaves[0]
 
    # Group leaves into ProductNodes, then wrap in SumNodes
    def chunk(lst: List[nn.Module], size: int) -> List[List[nn.Module]]:
        return [lst[i : i + size] for i in range(0, len(lst), size)]
 
    product_layer: List[nn.Module] = []
    for group in chunk(leaves, product_branching):
        if len(group) == 1:
            product_layer.append(group[0])
        else:
            product_layer.append(ProductNode(group))
 
    sum_layer: List[nn.Module] = []
    for group in chunk(product_layer, sum_branching):
        if len(group) == 1:
            sum_layer.append(group[0])
        else:
            sum_layer.append(SumNode(group))
 
    # Recurse until a single root remains
    return build_spn_circuit(
        sum_layer,
        sum_branching=sum_branching,
        product_branching=product_branching,
        log_space=log_space,
    )
 
 
# ---------------------------------------------------------------------------
# Utility: partition scopes for test leaf construction
# ---------------------------------------------------------------------------
def _make_leaf(scope_vars: Sequence[int], out_dim: int = 1) -> "LeafNode":
    """Create a minimal LeafNode with a fixed scope (for testing)."""
    return LeafNode(scope=set(scope_vars), out_dim=out_dim)
 
 
class LeafNode(ScopedModule):
    """Minimal univariate leaf distribution node (Gaussian density proxy).
 
    Used internally for tests and circuit scaffolding.  A production system
    would replace this with learned input distributions (e.g. isotonic,
    Bernoulli, Gaussian mixture) that wrap real feature columns.
 
    Configuration key ``leaf_nodes.hidden_dim`` (default 16) controls the
    internal linear projection size.
    """
 
    def __init__(self, scope: set, *, out_dim: int = 1) -> None:
        super().__init__()
 
        cfg = load_global_config()
        leaf_cfg = get_config_section("leaf_nodes", cfg)
        hidden_dim: int = bounded_iterations(leaf_cfg.get("hidden_dim", 16), minimum=1, maximum=4096)
 
        if not scope and not leaf_cfg.get("allow_empty_scope", False):
            raise ModelInitializationError(
                "LeafNode scope must be non-empty",
                context={"scope": scope},
            )
 
        self._scope = set(scope)
        in_dim = max(len(scope), 1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
 
    def compute_scope(self) -> set:  # noqa: D102
        return self._scope
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # Slice only the feature columns this leaf is responsible for
        indices = sorted(self._scope)
        if indices and x.shape[-1] >= max(indices) + 1:
            x_sub = x[..., indices]
        else:
            x_sub = x[..., : len(indices)] if indices else x
        return self.net(x_sub)

    def trace_activations(self, x: torch.Tensor, _depth: int = 0, **kwargs: Any) -> Dict[str, Any]:
        with torch.no_grad():
            out = self.forward(x).detach()
        return {
            "node_type": "LeafNode",
            "scope": sorted(self._scope),
            "output": out,
        }


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Nodes ===\n")
    printer.status("TEST", "Nodes initialized", "info")
 
    # ------------------------------------------------------------------
    # Helper — tiny mock config so tests run without the full project tree
    # ------------------------------------------------------------------
    import sys
    from unittest.mock import MagicMock, patch
 
    _mock_sum_cfg: Dict[str, Any] = {
        "weight_init": "uniform",
        "log_space": False,
        "temperature": 1.0,
        "min_children": 2,
        "max_children": 512,
        "enforce_smoothness": True,
        "activation_depth": 8,
    }
    _mock_product_cfg: Dict[str, Any] = {
        "log_space": False,
        "min_children": 2,
        "max_children": 512,
        "strict_decomposable": True,
        "activation_depth": 8,
    }
    _mock_leaf_cfg: Dict[str, Any] = {
        "hidden_dim": 16,
        "allow_empty_scope": False,
    }
    _mock_global: Dict[str, Any] = {
        "sum_nodes": _mock_sum_cfg,
        "product_nodes": _mock_product_cfg,
        "leaf_nodes": _mock_leaf_cfg,
    }
 
    def _mock_load_global_config(*_a: Any, **_kw: Any) -> Dict[str, Any]:
        return dict(_mock_global)
 
    def _mock_get_config_section(section: str, cfg: Optional[Dict] = None, **_kw: Any) -> Dict[str, Any]:
        return dict(_mock_global.get(section, {}))
 
    _patch_load   = patch("__main__.load_global_config",   side_effect=_mock_load_global_config)
    _patch_get    = patch("__main__.get_config_section",   side_effect=_mock_get_config_section)
    # Also patch inside the module namespace so LeafNode / SumNode / ProductNode pick them up
    _patch_load2  = patch("nodes.load_global_config",  side_effect=_mock_load_global_config)
    _patch_get2   = patch("nodes.get_config_section",  side_effect=_mock_get_config_section)
 
    # Run all tests inside patched context
    with _patch_load, _patch_get:
 
        BATCH = 4
        FEAT  = 6   # six input features, split across two leaves
 
        # ----------------------------------------------------------------
        # Test 1 — LeafNode construction and forward pass
        # ----------------------------------------------------------------
        printer.status("TEST", "1/8  LeafNode — construction and forward", "info")
        t0 = time.monotonic()
        leaf_a = LeafNode(scope={0, 1, 2}, out_dim=1)
        leaf_b = LeafNode(scope={3, 4, 5}, out_dim=1)
        x_in   = torch.randn(BATCH, FEAT)
        out_a  = leaf_a(x_in)
        out_b  = leaf_b(x_in)
        assert out_a.shape == (BATCH, 1), f"Expected ({BATCH},1), got {out_a.shape}"
        assert out_b.shape == (BATCH, 1), f"Expected ({BATCH},1), got {out_b.shape}"
        printer.status("PASS", f"LeafNode outputs: {out_a.shape}, {out_b.shape} "
                       f"[{elapsed_seconds(t0):.3f}s]", "success")
 
        # ----------------------------------------------------------------
        # Test 2 — ProductNode construction and decomposability
        # ----------------------------------------------------------------
        printer.status("TEST", "2/8  ProductNode — construction and decomposability", "info")
        t0 = time.monotonic()
        prod = ProductNode([leaf_a, leaf_b])
        out_prod = prod(x_in)
        assert out_prod.shape == (BATCH, 1), f"Unexpected shape {out_prod.shape}"
        assert prod.is_decomposable(), "ProductNode should be decomposable"
        assert prod.compute_scope() == {0, 1, 2, 3, 4, 5}
        printer.status("PASS", f"ProductNode output: {out_prod.shape} "
                       f"[{elapsed_seconds(t0):.3f}s]", "success")
 
        # ----------------------------------------------------------------
        # Test 3 — SumNode construction, weights, and forward pass
        # ----------------------------------------------------------------
        printer.status("TEST", "3/8  SumNode — construction and weighted forward", "info")
        t0 = time.monotonic()
        # Two product nodes with same scope pattern for smoothness
        leaf_c = LeafNode(scope={0, 1, 2}, out_dim=1)
        leaf_d = LeafNode(scope={3, 4, 5}, out_dim=1)
        prod2  = ProductNode([leaf_c, leaf_d])
        # Disable smoothness check for this test (leaves have identical scope sets)
        _mock_sum_cfg["enforce_smoothness"] = False
        sum_node = SumNode([prod, prod2])
        out_sum  = sum_node(x_in)
        assert out_sum.shape == (BATCH, 1), f"Unexpected shape {out_sum.shape}"
        w = sum_node.normalized_weights()
        assert abs(float(w.sum()) - 1.0) < _EPS * 10, "Weights must sum to 1"
        printer.status("PASS", f"SumNode output: {out_sum.shape}, weight_sum={float(w.sum()):.6f} "
                       f"[{elapsed_seconds(t0):.3f}s]", "success")
        _mock_sum_cfg["enforce_smoothness"] = True  # restore
 
        # ----------------------------------------------------------------
        # Test 4 — SumNode log-space forward
        # ----------------------------------------------------------------
        printer.status("TEST", "4/8  SumNode — log-space forward", "info")
        t0 = time.monotonic()
        _mock_sum_cfg["log_space"] = True
        leaf_e = LeafNode(scope={0, 1, 2}, out_dim=1)
        leaf_f = LeafNode(scope={3, 4, 5}, out_dim=1)
        prod3  = ProductNode([leaf_e, leaf_f])
        leaf_g = LeafNode(scope={0, 1, 2}, out_dim=1)
        leaf_h = LeafNode(scope={3, 4, 5}, out_dim=1)
        prod4  = ProductNode([leaf_g, leaf_h])
        _mock_sum_cfg["enforce_smoothness"] = False
        sum_log = SumNode([prod3, prod4])
        _mock_sum_cfg["enforce_smoothness"] = True
        out_log = sum_log(x_in)
        assert out_log.shape == (BATCH, 1), f"Unexpected log-space shape {out_log.shape}"
        assert torch.isfinite(out_log).all(), "Log-space output contains non-finite values"
        printer.status("PASS", f"Log-space SumNode output finite: {out_log.shape} "
                       f"[{elapsed_seconds(t0):.3f}s]", "success")
        _mock_sum_cfg["log_space"] = False
 
        # ----------------------------------------------------------------
        # Test 5 — trace_activations
        # ----------------------------------------------------------------
        printer.status("TEST", "5/8  trace_activations — SumNode and ProductNode", "info")
        t0 = time.monotonic()
        trace = sum_node.trace_activations(x_in)
        assert trace["node_type"] == "SumNode"
        assert "children" in trace
        assert len(trace["children"]) == 2
        p_trace = prod.trace_activations(x_in)
        assert p_trace["node_type"] == "ProductNode"
        printer.status("PASS", f"trace keys: {sorted(trace.keys())} "
                       f"[{elapsed_seconds(t0):.3f}s]", "success")
 
        # ----------------------------------------------------------------
        # Test 6 — weight_entropy and effective_children
        # ----------------------------------------------------------------
        printer.status("TEST", "6/8  SumNode — weight_entropy and effective_children", "info")
        t0 = time.monotonic()
        ent = sum_node.weight_entropy()
        eff = sum_node.effective_children()
        assert ent >= 0.0, "Entropy must be non-negative"
        assert 0 <= eff <= len(sum_node.children_modules)
        printer.status("PASS", f"entropy={ent:.4f}, effective_children={eff} "
                       f"[{elapsed_seconds(t0):.3f}s]", "success")
 
        # ----------------------------------------------------------------
        # Test 7 — to_state_dict_summary
        # ----------------------------------------------------------------
        printer.status("TEST", "7/8  to_state_dict_summary — both nodes", "info")
        t0 = time.monotonic()
        s_sum  = sum_node.to_state_dict_summary()
        s_prod = prod.to_state_dict_summary()
        assert s_sum["node_type"] == "SumNode"
        assert s_prod["node_type"] == "ProductNode"
        assert "timestamp_ms" in s_sum
        assert s_prod["decomposable"] is True
        printer.status("PASS", f"SumNode summary keys: {sorted(s_sum.keys())} "
                       f"[{elapsed_seconds(t0):.3f}s]", "success")
 
        # ----------------------------------------------------------------
        # Test 8 — build_spn_circuit and ProductNode overlap error
        # ----------------------------------------------------------------
        printer.status("TEST", "8/8  build_spn_circuit + constraint error paths", "info")
        t0 = time.monotonic()
 
        # 8a: Valid smooth circuit
        # Create two product nodes that each cover {0,1,2,3} but with different internal splits.
        leaf_0 = _make_leaf([0], out_dim=1)
        leaf_1 = _make_leaf([1], out_dim=1)
        leaf_2 = _make_leaf([2], out_dim=1)
        leaf_3 = _make_leaf([3], out_dim=1)
        
        # Product node A: scope {0,1,2,3} via (0,1) * (2,3)
        prod_a = ProductNode([ProductNode([leaf_0, leaf_1]), ProductNode([leaf_2, leaf_3])])
        # Product node B: scope {0,1,2,3} via (0,2) * (1,3)
        prod_b = ProductNode([ProductNode([leaf_0, leaf_2]), ProductNode([leaf_1, leaf_3])])
        
        # SumNode with two children that have identical scopes (both are {0,1,2,3})
        sum_node = SumNode([prod_a, prod_b])
        x8 = torch.randn(BATCH, 4)
        out8 = sum_node(x8)
        assert torch.isfinite(out8).all()
        
        # 8b: Overlapping scopes must raise CircuitConstraintError (unchanged)
        overlap_leaf1 = LeafNode(scope={0, 1}, out_dim=1)
        overlap_leaf2 = LeafNode(scope={1, 2}, out_dim=1)
        try:
            bad_prod = ProductNode([overlap_leaf1, overlap_leaf2])
            raise AssertionError("Expected CircuitConstraintError was not raised")
        except CircuitConstraintError as exc:
            printer.status("PASS", f"CircuitConstraintError raised correctly: {exc.code}", "success")
        
        # 8c: SumNode with too few children (unchanged)
        try:
            SumNode([leaf_0])   # min_children=2 in mock config
            raise AssertionError("Expected ModelInitializationError was not raised")
        except ModelInitializationError as exc:
            printer.status("PASS", f"ModelInitializationError raised correctly: {exc.code}", "success")

    print("\n=== Test ran successfully ===\n")