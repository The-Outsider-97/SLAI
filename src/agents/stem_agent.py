"""Production SLAI STEM Agent orchestration façade.

The STEM Agent is SLAI's deterministic quantitative scientific-computing
authority.  It validates requests, dispatches only explicitly approved STEM
operations, coordinates local deterministic memoization and SLAI SharedMemory,
exposes checkpoint-safe Agent-owned state, and preserves the scientific result
contracts implemented by :mod:`src.agents.stem`.

Architectural boundary::

    STEM computes.
    Reasoning infers.
    Simulation evolves.
    Optimization searches.

The Agent intentionally does not implement numerical algorithms itself and does
not resolve or load subsystem configuration files.  Agent-level behavior comes
exclusively from ``agents_config.yaml`` through the BaseAgent configuration
infrastructure; internal STEM components own their numerical/domain defaults.
"""
from __future__ import annotations

__version__ = "2.3.0"

import hashlib
import json
import random
import time
import uuid

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Dict, Optional, Tuple

from .base.utils.config_contract import ConfigContractError, assert_valid_config_contract
from .base.utils.main_config_loader import get_config_section, load_global_config
from .base_agent import BaseAgent
from .stem import (
    Algebra,
    Biology,
    Calculus,
    Computing,
    Dimensions,
    Engineering,
    NumericalMethods,
    NumericResult,
    Physics,
    SolverResult,
    Statistics,
    STEMMemory,
    UnitSystem,
    Uncertainty,
    combined_standard_uncertainty,
    covariance_propagation,
    coverage_interval,
    expanded_uncertainty,
    jacobian_propagation,
    monte_carlo_propagation,
    numerical_error_budget,
    sensitivity_coefficients,
    standard_uncertainty,
)
from .stem.utils.stem_errors import *
from .stem.utils.stem_helpers import json_safe
from .stem.utils.temp_loader import list_templates, load_template
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("STEM Agent")
printer = PrettyPrinter()


@dataclass(frozen=True)
class STEMResult:
    """Agent-level execution envelope preserving the original STEM result."""

    request_id: str
    domain: str
    operation: str
    result: Any
    cache_hit: bool
    cache_key: Optional[str]
    duration_ms: float
    published: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    template: Optional[Mapping[str, Any]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = "success"

    def to_dict(self, *, serialize_result: bool = False) -> Dict[str, Any]:
        return {
            "status": self.status,
            "request_id": self.request_id,
            "domain": self.domain,
            "operation": self.operation,
            "result": json_safe(self.result) if serialize_result else self.result,
            "cache_hit": self.cache_hit,
            "cache_key": self.cache_key,
            "duration_ms": self.duration_ms,
            "published": self.published,
            "diagnostics": json_safe(self.diagnostics),
            "template": json_safe(self.template) if self.template is not None else None,
            "metadata": json_safe(self.metadata),
        }


class STEMAgent(BaseAgent):
    """Validated orchestration boundary over the deterministic STEM subsystem."""

    AGENT_KEY = "stem_agent"
    CHECKPOINTING_SUPPORTED = True
    CHECKPOINT_SCHEMA = "slai.stem_agent.state.v1"

    _ALLOWED_CONFIG_KEYS = {
        "enabled",
        "supported_domains",
        "default_domain",
        "shared_memory_enabled",
        "publish_shared_results",
        "shared_result_key_prefix",
        "shared_result_topic",
        "local_memory_enabled",
        "cache_enabled",
        "template_loading_enabled",
        "diagnostics_enabled",
        "strict_validation",
        "checkpoint_local_memory",
        "restore_local_memory",
        "max_batch_size",
        "metrics_enabled",
    }

    _TASK_KEYS = {
        "domain",
        "operation",
        "args",
        "kwargs",
        "cache",
        "publish",
        "request_id",
        "seed",
        "template",
        "template_kwarg",
        "force_template_reload",
        "metadata",
        "tasks",
    }

    _DOMAIN_OPERATIONS: Mapping[str, Tuple[str, ...]] = {
        "algebra": (
            "rational", "normalize_polynomial", "polynomial_add", "polynomial_subtract",
            "polynomial_multiply", "polynomial_divmod", "polynomial_gcd",
            "polynomial_derivative", "polynomial_evaluate", "quadratic_roots",
            "solve_linear", "substitute_polynomial",
        ),
        "calculus": (
            "symbolic_derivative", "symbolic_integral", "gradient", "jacobian",
            "hessian", "directional_derivative", "limit", "taylor_series", "autodiff",
        ),
        "numerical_methods": (
            "lu_decomposition", "qr_decomposition", "cholesky", "solve_linear_system",
            "least_squares", "eigenvalues_symmetric", "singular_values", "matrix_inverse",
            "bisection", "newton_raphson", "secant", "fixed_point",
            "linear_interpolation", "polynomial_interpolation", "cubic_spline",
            "chebyshev_approximation", "forward_difference", "central_difference",
            "richardson_extrapolation", "trapezoidal", "simpson", "romberg",
            "gauss_legendre", "adaptive_quadrature", "euler", "rk4", "rk45_adaptive",
            "discretize_operator_1d", "apply_boundary_conditions",
            "solve_linearized_pde_system", "condition_numbers", "floating_point_effects",
            "algorithmic_stability", "catastrophic_cancellation", "residuals",
            "forward_error", "backward_error",
        ),
        "statistics": (
            "mean", "variance", "standard_deviation", "median", "quantile", "mode",
            "raw_moment", "central_moment", "standardized_moment", "skewness", "kurtosis",
            "covariance", "correlation", "spearman_correlation", "kendall_tau",
            "covariance_matrix", "correlation_matrix", "linear_regression",
            "polynomial_regression", "multiple_linear_regression", "weighted_least_squares",
            "normal_pdf", "normal_cdf", "normal_ppf", "t_pdf", "t_cdf",
            "chi_square_cdf", "f_cdf", "binomial_pmf", "log_likelihood_normal",
            "log_likelihood_bernoulli", "monte_carlo_integrate", "bootstrap_ci",
        ),
        "units": (
            "lookup", "known_units", "convert", "convert_quantity", "is_compatible",
            "require_compatible", "to_base", "from_base", "normalize", "compose",
            "is_affine", "dimensional_signature", "accepts",
        ),
        "dimensions": (
            "lookup", "base_dimensions", "derived_dimensions", "dimension",
            "is_compatible", "require_compatible", "is_dimensionless", "combine", "power",
            "product", "infer_from_unit", "dimension_from_expression", "check_consistency",
            "buckingham_pi", "to_dict",
        ),
        "uncertainty": (
            "standard_uncertainty", "combined_standard_uncertainty", "expanded_uncertainty",
            "covariance_propagation", "jacobian_propagation", "sensitivity_coefficients",
            "coverage_interval", "monte_carlo_propagation", "numerical_error_budget",
        ),
        "biology": (
            "logistic_growth_rate", "logistic_population", "michaelis_menten",
            "first_order_decay", "lotka_volterra_rates", "compartment_balance",
        ),
        "physics": (
            "kinetic_energy", "momentum", "gravitational_force", "ideal_gas_pressure",
            "relativistic_gamma",
        ),
        "engineering": (
            "axial_stress", "heat_conduction_1d", "electrical_power", "reynolds_number",
        ),
        "computing": (
            "gcd", "lcm", "factorial", "binomial", "recurrence", "breadth_first_search",
            "dijkstra", "topological_sort",
        ),
    }

    _CONFIG_ATTRS: Mapping[str, str] = {
        "algebra": "algebra_config",
        "calculus": "calculus_config",
        "numerical_methods": "num_config",
        "statistics": "statistics_config",
        "units": "unit_config",
        "dimensions": "dimension_config",
        "uncertainty": "uncertainty_config",
        "biology": "biology_config",
        "physics": "physics_config",
        "engineering": "engineering_config",
        "computing": "computing_config",
    }

    _STOCHASTIC_STATISTICS = {"monte_carlo_integrate", "bootstrap_ci"}

    def __init__(
        self,
        shared_memory: Any,
        agent_factory: Any,
        config: Optional[Mapping[str, Any]] = None,
        *,
        checkpoint_manager: Any = None,
    ) -> None:
        # Agent-level STEM options are intentionally not injected into
        # BaseAgent's ``base_agent`` config namespace.
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            checkpoint_manager=checkpoint_manager,
        )

        self._stem_lock = RLock()
        self._metrics_lock = RLock()
        self.config = load_global_config()
        self.global_config = self.config
        self.agent_config: Dict[str, Any] = dict(get_config_section(self.AGENT_KEY) or {})
        if config is not None:
            if not isinstance(config, Mapping):
                raise STEMConfigurationError(
                    "STEMAgent config override must be a mapping",
                    component=self.name,
                )
            self.agent_config.update(dict(config))

        try:
            assert_valid_config_contract(
                global_config=self.config,
                agent_key=self.AGENT_KEY,
                agent_config=self.agent_config,
                agent_allowed_keys=self._ALLOWED_CONFIG_KEYS,
                required_agent_keys={"enabled", "supported_domains", "default_domain"},
                require_global_keys=False,
                require_agent_section=True,
                warn_unknown_global_keys=False,
                logger=logger,
            )
        except ConfigContractError as exc:
            raise STEMConfigurationError(
                "STEMAgent configuration violates the SLAI agent contract",
                component=self.name,
                cause=exc,
            ) from exc
        self._load_agent_config()
        self._validate_agent_config()

        # Local memory is always materialized so ownership is explicit even if
        # cache use is disabled by policy.  It remains fully independent from
        # BaseAgent-owned SharedMemory.
        self.local_memory = STEMMemory()

        self.algebra = Algebra()
        self.calculus = Calculus()
        self.numerical_methods = NumericalMethods()
        self.statistics = Statistics()
        self.units = UnitSystem()
        self.dimensions = Dimensions()
        self.uncertainty = Uncertainty(memory=self.local_memory)
        self.biology = Biology(memory=self.local_memory)
        self.physics = Physics(memory=self.local_memory)
        self.engineering = Engineering(memory=self.local_memory)
        self.computing = Computing(memory=self.local_memory)

        self._dispatch = self._build_dispatch_registry()
        self._requests_total = 0
        self._requests_success = 0
        self._requests_failed = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._template_loads = 0
        self._non_convergence_count = 0
        self._solver_failures = 0
        self._runtime_total_ms = 0.0
        self._domain_counts: Counter[str] = Counter()

        logger.info(
            "STEMAgent initialized | domains=%s | local_cache=%s | shared_publish=%s | checkpointing=%s",
            ",".join(self.supported_domains),
            self.cache_enabled and self.local_memory_enabled,
            self.shared_memory_enabled and self.publish_shared_results,
            self.supports_checkpointing,
        )

    # ------------------------------------------------------------------
    # Agent-level configuration (agents_config.yaml only)
    # ------------------------------------------------------------------
    def _load_agent_config(self) -> None:
        cfg = self.agent_config
        self.enabled = bool(cfg.get("enabled", True))
        raw_domains = cfg.get("supported_domains", tuple(self._DOMAIN_OPERATIONS))
        if isinstance(raw_domains, str) or not isinstance(raw_domains, Sequence):
            raise STEMConfigurationError("supported_domains must be a sequence of domain names")
        self.supported_domains = tuple(str(item).strip().lower() for item in raw_domains if str(item).strip())
        self.default_domain = str(cfg.get("default_domain", "numerical_methods")).strip().lower()
        self.shared_memory_enabled = bool(cfg.get("shared_memory_enabled", True))
        self.publish_shared_results = bool(cfg.get("publish_shared_results", False))
        self.shared_result_key_prefix = str(cfg.get("shared_result_key_prefix", "stem:result")).strip()
        self.shared_result_topic = str(cfg.get("shared_result_topic", "stem.results")).strip()
        self.local_memory_enabled = bool(cfg.get("local_memory_enabled", True))
        self.cache_enabled = bool(cfg.get("cache_enabled", True))
        self.template_loading_enabled = bool(cfg.get("template_loading_enabled", True))
        self.diagnostics_enabled = bool(cfg.get("diagnostics_enabled", True))
        self.strict_validation = bool(cfg.get("strict_validation", True))
        self.checkpoint_local_memory = bool(cfg.get("checkpoint_local_memory", True))
        self.restore_local_memory = bool(cfg.get("restore_local_memory", True))
        self.metrics_enabled = bool(cfg.get("metrics_enabled", True))
        raw_batch = cfg.get("max_batch_size", 32)
        if isinstance(raw_batch, bool):
            raise STEMConfigurationError("max_batch_size must be a positive integer")
        self.max_batch_size = int(raw_batch)

    def _validate_agent_config(self) -> None:
        if not self.supported_domains:
            raise STEMConfigurationError("supported_domains must not be empty")
        unknown = sorted(set(self.supported_domains) - set(self._DOMAIN_OPERATIONS))
        if unknown:
            raise STEMConfigurationError("Unknown STEM domains in supported_domains", context={"unknown": unknown})
        if self.default_domain not in self.supported_domains:
            raise STEMConfigurationError(
                "default_domain must be enabled in supported_domains",
                context={"default_domain": self.default_domain},
            )
        if self.max_batch_size < 1:
            raise STEMConfigurationError("max_batch_size must be >= 1")
        if not self.shared_result_key_prefix:
            raise STEMConfigurationError("shared_result_key_prefix must be non-empty")
        if not self.shared_result_topic:
            raise STEMConfigurationError("shared_result_topic must be non-empty")

    # ------------------------------------------------------------------
    # Explicit dispatch registry
    # ------------------------------------------------------------------
    def _build_dispatch_registry(self) -> Mapping[str, Mapping[str, Callable[..., Any]]]:
        components: Mapping[str, Any] = {
            "algebra": self.algebra,
            "calculus": self.calculus,
            "numerical_methods": self.numerical_methods,
            "statistics": self.statistics,
            "units": self.units,
            "dimensions": self.dimensions,
            "biology": self.biology,
            "physics": self.physics,
            "engineering": self.engineering,
            "computing": self.computing,
        }
        uncertainty_ops: Mapping[str, Callable[..., Any]] = {
            "standard_uncertainty": standard_uncertainty,
            "combined_standard_uncertainty": combined_standard_uncertainty,
            "expanded_uncertainty": expanded_uncertainty,
            "covariance_propagation": covariance_propagation,
            "jacobian_propagation": jacobian_propagation,
            "sensitivity_coefficients": sensitivity_coefficients,
            "coverage_interval": coverage_interval,
            "monte_carlo_propagation": monte_carlo_propagation,
            "numerical_error_budget": numerical_error_budget,
        }
        registry: Dict[str, Dict[str, Callable[..., Any]]] = {"uncertainty": dict(uncertainty_ops)}
        for domain, component in components.items():
            operations: Dict[str, Callable[..., Any]] = {}
            for name in self._DOMAIN_OPERATIONS[domain]:
                candidate = getattr(component, name, None)
                if not callable(candidate):
                    raise STEMConfigurationError(
                        "Configured STEM operation is unavailable",
                        component=self.name,
                        context={"domain": domain, "operation": name, "component_type": type(component).__name__},
                    )
                operations[name] = candidate
            registry[domain] = operations
        return registry

    def available_operations(self, domain: Optional[str] = None) -> Mapping[str, Tuple[str, ...]] | Tuple[str, ...]:
        if domain is None:
            return {name: tuple(self._dispatch[name]) for name in self.supported_domains}
        normalized = self._normalize_domain(domain)
        return tuple(self._dispatch[normalized])

    def _normalize_domain(self, domain: Any) -> str:
        if not isinstance(domain, str) or not domain.strip():
            raise STEMValidationError("domain must be a non-empty string", component=self.name)
        normalized = domain.strip().lower()
        if normalized not in self.supported_domains:
            raise STEMValidationError(
                "Unsupported STEM domain",
                component=self.name,
                context={"domain": normalized, "supported": list(self.supported_domains)},
            )
        return normalized

    def _resolve_operation(self, domain: str, operation: Any) -> Tuple[str, Callable[..., Any]]:
        if not isinstance(operation, str) or not operation.strip():
            raise STEMValidationError("operation must be a non-empty string", component=self.name)
        name = operation.strip()
        if name.startswith("_") or "__" in name:
            raise STEMValidationError("Private or dunder STEM operations are not dispatchable", component=self.name)
        callback = self._dispatch[domain].get(name)
        if callback is None:
            raise STEMValidationError(
                "Unsupported STEM operation",
                component=self.name,
                context={"domain": domain, "operation": name, "supported": sorted(self._dispatch[domain])},
            )
        return name, callback

    # ------------------------------------------------------------------
    # Template access through the subsystem-owned API
    # ------------------------------------------------------------------
    def get_template(self, name: str, *, force_reload: bool = False) -> Any:
        if not self.template_loading_enabled:
            raise STEMConfigurationError("STEM template loading is disabled", component=self.name)
        value = load_template(name, force_reload=force_reload)
        if self.metrics_enabled:
            with self._metrics_lock:
                self._template_loads += 1
        return value

    def available_templates(self) -> Tuple[str, ...]:
        if not self.template_loading_enabled:
            return ()
        return list_templates()

    @staticmethod
    def _template_digest(value: Any) -> str:
        payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    # ------------------------------------------------------------------
    # Request validation / deterministic execution
    # ------------------------------------------------------------------
    def _validate_task_mapping(self, task_data: Any) -> Mapping[str, Any]:
        if not isinstance(task_data, Mapping):
            raise STEMValidationError(
                "STEMAgent task must be a mapping",
                component=self.name,
                context={"actual_type": type(task_data).__name__},
            )
        if self.strict_validation:
            unknown = sorted(set(task_data) - self._TASK_KEYS)
            if unknown:
                raise STEMValidationError("Unknown STEM task fields", component=self.name, context={"unknown": unknown})
        return task_data

    @staticmethod
    def _normalize_args(value: Any) -> Tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise STEMValidationError("args must be a sequence")
        return tuple(value)

    @staticmethod
    def _normalize_kwargs(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise STEMValidationError("kwargs must be a mapping")
        return dict(value)

    @staticmethod
    def _normalize_seed(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise STEMValidationError("seed must be an integer when provided")
        return value

    def _component_for_domain(self, domain: str) -> Any:
        return {
            "algebra": self.algebra,
            "calculus": self.calculus,
            "numerical_methods": self.numerical_methods,
            "statistics": self.statistics,
            "units": self.units,
            "dimensions": self.dimensions,
            "uncertainty": self.uncertainty,
            "biology": self.biology,
            "physics": self.physics,
            "engineering": self.engineering,
            "computing": self.computing,
        }[domain]

    def _component_config_digest(self, domain: str) -> Optional[str]:
        component = self._component_for_domain(domain)
        attribute = self._CONFIG_ATTRS.get(domain)
        if not attribute:
            return None
        value = getattr(component, attribute, None)
        return STEMMemory.stable_digest(value) if isinstance(value, Mapping) else None

    def _component_constants_digest(self, domain: str) -> Optional[str]:
        component = self._component_for_domain(domain)
        constants = getattr(component, "constants", None)
        return STEMMemory.stable_digest(constants) if isinstance(constants, Mapping) else None

    @staticmethod
    def _implementation_digest(callback: Callable[..., Any]) -> str:
        return STEMMemory.stable_digest(callback)

    @staticmethod
    def _cache_controls(kwargs: Mapping[str, Any]) -> Tuple[Optional[str], Mapping[str, Any], Mapping[str, Any], Any, Any]:
        method = kwargs.get("method")
        tolerance_keys = ("tol", "rtol", "atol", "h", "h0", "max_iter", "max_depth", "levels", "n")
        tolerances = {key: kwargs[key] for key in tolerance_keys if key in kwargs}
        precision_keys = ("precision", "significant_digits", "decimal_places", "rounding_mode")
        precision = {key: kwargs[key] for key in precision_keys if key in kwargs}
        initial = kwargs.get("initial_conditions", kwargs.get("y0"))
        boundary = kwargs.get("boundary_conditions")
        return str(method) if method is not None else None, precision, tolerances, initial, boundary

    def _prepare_execution_kwargs(
        self,
        domain: str,
        operation: str,
        kwargs: Mapping[str, Any],
        seed: Optional[int],
    ) -> Tuple[Dict[str, Any], bool]:
        execution_kwargs = dict(kwargs)
        deterministic_cache = True
        if domain == "statistics" and operation in self._STOCHASTIC_STATISTICS:
            if "rng" in execution_kwargs:
                if seed is not None:
                    raise STEMValidationError("Provide either rng or seed, not both, for stochastic statistics")
                deterministic_cache = False
            elif seed is not None:
                execution_kwargs["rng"] = random.Random(seed)
            else:
                deterministic_cache = False
        if domain == "uncertainty" and operation == "monte_carlo_propagation":
            kw_seed = execution_kwargs.get("seed")
            if seed is not None and kw_seed is not None and int(kw_seed) != seed:
                raise STEMValidationError("Conflicting Monte Carlo seeds")
            if kw_seed is None and seed is not None:
                execution_kwargs["seed"] = seed
            if execution_kwargs.get("seed") is None:
                deterministic_cache = False
        return execution_kwargs, deterministic_cache

    def _build_cache_key(
        self,
        *,
        domain: str,
        operation: str,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
        seed: Optional[int],
        template_digest: Optional[str],
        callback: Callable[..., Any],
    ) -> str:
        method, precision, tolerances, initial, boundary = self._cache_controls(kwargs)
        return self.local_memory.build_cache_key(
            domain=domain,
            operation=operation,
            args=args,
            kwargs=kwargs,
            method=method,
            precision=precision,
            tolerances=tolerances,
            algorithm_version=self.local_memory.algorithm_version,
            constants_version=self._component_constants_digest(domain),
            initial_conditions=initial,
            boundary_conditions=boundary,
            seed=seed,
            config_digest=self._component_config_digest(domain),
            template_digest=template_digest,
            extra={"implementation_digest": self._implementation_digest(callback)},
        )

    @staticmethod
    def _result_diagnostics(result: Any) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        if isinstance(result, SolverResult):
            diagnostics.update(
                {
                    "converged": result.converged,
                    "iterations": result.iterations,
                    "residual": result.residual,
                    "solver_status": result.status.value,
                    "solver_runtime": result.runtime,
                }
            )
            diagnostics.update(dict(result.diagnostics))
        elif isinstance(result, NumericResult):
            diagnostics.update(
                {
                    "absolute_error": result.absolute_error,
                    "relative_error": result.relative_error,
                    "residual": result.residual,
                    "condition_estimate": result.condition_estimate,
                    "warnings": list(result.warnings),
                    "method": result.method,
                }
            )
        elif isinstance(result, Mapping):
            for key in (
                "converged", "iterations", "residual", "condition_estimate", "condition_number",
                "absolute_error", "relative_error", "warnings", "method", "seed",
            ):
                if key in result:
                    diagnostics[key] = json_safe(result[key])
        return {key: value for key, value in diagnostics.items() if value is not None}

    def _record_completion(
        self,
        *,
        domain: str,
        success: bool,
        cache_hit: bool,
        cache_lookup_attempted: bool,
        duration_ms: float,
        result: Any = None,
    ) -> None:
        if not self.metrics_enabled:
            return
        with self._metrics_lock:
            self._requests_total += 1
            self._domain_counts[domain] += 1
            self._runtime_total_ms += duration_ms
            if success:
                self._requests_success += 1
            else:
                self._requests_failed += 1
            if cache_lookup_attempted:
                if cache_hit:
                    self._cache_hits += 1
                else:
                    self._cache_misses += 1
            if isinstance(result, SolverResult) and not result.converged:
                self._non_convergence_count += 1

    def _record_failure(self, domain: str, duration_ms: float, *, solver_failure: bool = False) -> None:
        if not self.metrics_enabled:
            return
        with self._metrics_lock:
            self._requests_total += 1
            self._requests_failed += 1
            self._domain_counts[domain] += 1
            self._runtime_total_ms += duration_ms
            if solver_failure:
                self._solver_failures += 1

    def _publish_result(self, envelope: STEMResult) -> bool:
        if not self.shared_memory_enabled:
            return False
        payload = envelope.to_dict(serialize_result=True)
        key = f"{self.shared_result_key_prefix}:{envelope.request_id}"
        setter = getattr(self.shared_memory, "set", None)
        putter = getattr(self.shared_memory, "put", None)
        if callable(setter):
            setter(key, payload)
        elif callable(putter):
            putter(key, payload)
        else:
            raise STEMConfigurationError("SharedMemory exposes neither set() nor put()", component=self.name)
        publisher = getattr(self.shared_memory, "publish", None)
        if callable(publisher):
            publisher(self.shared_result_topic, payload)
        return True

    def _execute_request(self, task_data: Mapping[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            raise STEMConfigurationError("STEMAgent is disabled", component=self.name)

        request_id = task_data.get("request_id")
        if request_id is None:
            request_id = f"stem-{uuid.uuid4().hex[:16]}"
        elif not isinstance(request_id, str) or not request_id.strip():
            raise STEMValidationError("request_id must be a non-empty string")
        else:
            request_id = request_id.strip()

        domain = self._normalize_domain(task_data.get("domain", self.default_domain))
        operation, callback = self._resolve_operation(domain, task_data.get("operation"))
        args = self._normalize_args(task_data.get("args", ()))
        kwargs = self._normalize_kwargs(task_data.get("kwargs", {}))
        seed = self._normalize_seed(task_data.get("seed", kwargs.get("seed")))
        publish = task_data.get("publish", self.publish_shared_results)
        if not isinstance(publish, bool):
            raise STEMValidationError("publish must be boolean")
        request_metadata = task_data.get("metadata", {})
        if request_metadata is None:
            request_metadata = {}
        if not isinstance(request_metadata, Mapping):
            raise STEMValidationError("metadata must be a mapping")
        request_metadata = dict(request_metadata)

        template_info: Optional[Dict[str, Any]] = None
        template_digest: Optional[str] = None
        template_name = task_data.get("template")
        template_kwarg = task_data.get("template_kwarg")
        if template_name is not None:
            if not isinstance(template_name, str) or not template_name.strip():
                raise STEMValidationError("template must be a non-empty template name")
            template_value = self.get_template(
                template_name.strip(),
                force_reload=bool(task_data.get("force_template_reload", False)),
            )
            digest = self._template_digest(template_value)
            template_info = {"name": template_name.strip(), "digest": digest}
            if template_kwarg is not None:
                if not isinstance(template_kwarg, str) or not template_kwarg.strip() or template_kwarg.startswith("_"):
                    raise STEMValidationError("template_kwarg must be a public non-empty keyword name")
                if template_kwarg in kwargs:
                    raise STEMValidationError("template_kwarg conflicts with an existing kwarg", context={"template_kwarg": template_kwarg})
                kwargs[template_kwarg.strip()] = template_value
                template_digest = digest
        elif template_kwarg is not None:
            raise STEMValidationError("template_kwarg requires template")

        execution_kwargs, stochastic_deterministic = self._prepare_execution_kwargs(domain, operation, kwargs, seed)
        requested_cache = task_data.get("cache", self.cache_enabled)
        if not isinstance(requested_cache, bool):
            raise STEMValidationError("cache must be boolean")
        cache_allowed = bool(self.cache_enabled and self.local_memory_enabled and requested_cache and stochastic_deterministic)
        reproducibility = {
            "cache_algorithm_version": self.local_memory.algorithm_version,
            "implementation_digest": self._implementation_digest(callback),
            "component_config_digest": self._component_config_digest(domain),
            "constants_digest": self._component_constants_digest(domain),
            "template_digest": template_digest,
            "seed": seed,
        }
        envelope_metadata = {
            "request": json_safe(request_metadata),
            "reproducibility": {key: value for key, value in reproducibility.items() if value is not None},
        }

        cache_key: Optional[str] = None
        started = time.perf_counter()
        if cache_allowed:
            cache_key = self._build_cache_key(
                domain=domain,
                operation=operation,
                args=args,
                kwargs=kwargs,
                seed=seed,
                template_digest=template_digest,
                callback=callback,
            )
            found, cached = self.local_memory.lookup(cache_key)
            if found:
                duration_ms = (time.perf_counter() - started) * 1000.0
                diagnostics = self._result_diagnostics(cached) if self.diagnostics_enabled else {}
                if seed is not None:
                    diagnostics.setdefault("seed", seed)
                envelope = STEMResult(
                    request_id=request_id,
                    domain=domain,
                    operation=operation,
                    result=cached,
                    cache_hit=True,
                    cache_key=cache_key,
                    duration_ms=duration_ms,
                    published=False,
                    diagnostics=diagnostics,
                    template=template_info,
                    metadata=envelope_metadata,
                    status="completed_with_warning" if diagnostics.get("converged") is False else "success",
                )
                published = self._publish_result(envelope) if publish else False
                if published:
                    envelope = STEMResult(**{**envelope.__dict__, "published": True})
                self._record_completion(
                    domain=domain, success=True, cache_hit=True, cache_lookup_attempted=True,
                    duration_ms=duration_ms, result=cached,
                )
                return envelope.to_dict()

        try:
            result = callback(*args, **execution_kwargs)
        except STEMError:
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._record_failure(domain, duration_ms, solver_failure=domain == "numerical_methods")
            raise
        except TypeError as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._record_failure(domain, duration_ms)
            raise STEMValidationError(
                "STEM operation arguments are invalid",
                component=self.name,
                operation=f"{domain}.{operation}",
                context={"domain": domain, "operation": operation},
                cause=exc,
            ) from exc
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._record_failure(domain, duration_ms, solver_failure=domain == "numerical_methods")
            raise STEMError(
                "Unexpected STEM subsystem failure",
                component=self.name,
                operation=f"{domain}.{operation}",
                context={"domain": domain, "operation": operation},
                cause=exc,
            ) from exc

        duration_ms = (time.perf_counter() - started) * 1000.0
        diagnostics = self._result_diagnostics(result) if self.diagnostics_enabled else {}
        if seed is not None:
            diagnostics.setdefault("seed", seed)
        if cache_allowed and cache_key is not None:
            self.local_memory.put(
                cache_key,
                result,
                metadata={
                    "domain": domain,
                    "operation": operation,
                    "algorithm_version": self.local_memory.algorithm_version,
                    **reproducibility,
                },
            )

        envelope = STEMResult(
            request_id=request_id,
            domain=domain,
            operation=operation,
            result=result,
            cache_hit=False,
            cache_key=cache_key,
            duration_ms=duration_ms,
            published=False,
            diagnostics=diagnostics,
            template=template_info,
            metadata=envelope_metadata,
            status="completed_with_warning" if diagnostics.get("converged") is False else "success",
        )
        published = self._publish_result(envelope) if publish else False
        if published:
            envelope = STEMResult(**{**envelope.__dict__, "published": True})
        self._record_completion(
            domain=domain, success=True, cache_hit=False, cache_lookup_attempted=cache_allowed,
            duration_ms=duration_ms, result=result,
        )
        return envelope.to_dict()

    def perform_task(self, task_data: Any) -> Dict[str, Any]:
        request = self._validate_task_mapping(task_data)
        batch = request.get("tasks")
        if batch is None:
            return self._execute_request(request)
        if isinstance(batch, (str, bytes, bytearray)) or not isinstance(batch, Sequence):
            raise STEMValidationError("tasks must be a sequence of request mappings")
        if len(batch) == 0 or len(batch) > self.max_batch_size:
            raise STEMValidationError(
                "STEM batch size is outside configured limits",
                context={"size": len(batch), "max_batch_size": self.max_batch_size},
            )
        results = []
        for item in batch:
            results.append(self._execute_request(self._validate_task_mapping(item)))
        return {"status": "success", "batch_size": len(results), "results": results}

    # ------------------------------------------------------------------
    # Shared/local memory lifecycle and checkpoint integration
    # ------------------------------------------------------------------
    def reset_local_state(self, *, clear_memory: bool = True, reset_metrics: bool = True) -> Mapping[str, Any]:
        """Reset only STEM-owned local state; SharedMemory is never cleared here."""
        removed = self.local_memory.clear(reset_stats=True) if clear_memory else 0
        if reset_metrics:
            with self._metrics_lock:
                self._requests_total = self._requests_success = self._requests_failed = 0
                self._cache_hits = self._cache_misses = self._template_loads = 0
                self._non_convergence_count = self._solver_failures = 0
                self._runtime_total_ms = 0.0
                self._domain_counts.clear()
        return {"local_memory_entries_removed": removed, "metrics_reset": reset_metrics}

    def checkpoint_metrics(self) -> Mapping[str, Any]:
        snapshot = self.metrics_snapshot()
        return {
            "requests_total": snapshot["requests_total"],
            "requests_success": snapshot["requests_success"],
            "cache_hits": snapshot["cache_hits"],
            "cache_misses": snapshot["cache_misses"],
            "local_memory_entries": snapshot["local_memory_entries"],
        }

    def _export_checkpoint_state(self) -> Mapping[str, Any]:
        state: Dict[str, Any] = {
            "schema": self.CHECKPOINT_SCHEMA,
            "version": __version__,
            "metrics": self.metrics_snapshot(),
            "supported_domains": list(self.supported_domains),
            "default_domain": self.default_domain,
            "local_memory": None,
        }
        if self.checkpoint_local_memory and self.local_memory_enabled:
            state["local_memory"] = self.local_memory.state_dict()
        return state

    def _import_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise STEMValidationError("STEMAgent checkpoint state must be a mapping")
        if state.get("schema") != self.CHECKPOINT_SCHEMA:
            raise STEMValidationError(
                "Incompatible STEMAgent checkpoint schema",
                context={"actual": state.get("schema"), "expected": self.CHECKPOINT_SCHEMA},
            )
        saved_version = str(state.get("version", ""))
        if saved_version != __version__:
            raise STEMValidationError(
                "STEMAgent checkpoint version is incompatible",
                context={"actual": saved_version, "expected": __version__},
            )
        saved_domains = tuple(str(item) for item in state.get("supported_domains", ()))
        if saved_domains and saved_domains != self.supported_domains:
            raise STEMValidationError(
                "STEMAgent checkpoint domain configuration is incompatible",
                context={"saved": list(saved_domains), "active": list(self.supported_domains)},
            )
        if state.get("default_domain", self.default_domain) != self.default_domain:
            raise STEMValidationError("STEMAgent checkpoint default_domain is incompatible")

        memory_state = state.get("local_memory")
        if self.restore_local_memory and self.local_memory_enabled and memory_state is not None:
            if not isinstance(memory_state, Mapping):
                raise STEMValidationError("STEMAgent local_memory checkpoint payload is invalid")
            self.local_memory.load_state_dict(memory_state, strict=True, merge=False)

        metrics = state.get("metrics", {})
        if isinstance(metrics, Mapping) and self.metrics_enabled:
            with self._metrics_lock:
                self._requests_total = max(0, int(metrics.get("requests_total", 0)))
                self._requests_success = max(0, int(metrics.get("requests_success", 0)))
                self._requests_failed = max(0, int(metrics.get("requests_failed", 0)))
                self._cache_hits = max(0, int(metrics.get("cache_hits", 0)))
                self._cache_misses = max(0, int(metrics.get("cache_misses", 0)))
                self._template_loads = max(0, int(metrics.get("template_loads", 0)))
                self._non_convergence_count = max(0, int(metrics.get("non_convergence_count", 0)))
                self._solver_failures = max(0, int(metrics.get("solver_failures", 0)))
                self._runtime_total_ms = max(0.0, float(metrics.get("runtime_total_ms", 0.0)))
                domain_counts = metrics.get("domain_dispatch_count", {})
                self._domain_counts = Counter(
                    {str(key): max(0, int(value)) for key, value in domain_counts.items()}
                    if isinstance(domain_counts, Mapping) else {}
                )

    # ------------------------------------------------------------------
    # Metrics / diagnostics
    # ------------------------------------------------------------------
    def metrics_snapshot(self) -> Dict[str, Any]:
        memory_stats = self.local_memory.stats()
        with self._metrics_lock:
            average_runtime = self._runtime_total_ms / self._requests_total if self._requests_total else 0.0
            return {
                "requests_total": self._requests_total,
                "requests_success": self._requests_success,
                "requests_failed": self._requests_failed,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "template_loads": self._template_loads,
                "non_convergence_count": self._non_convergence_count,
                "solver_failures": self._solver_failures,
                "runtime_total_ms": self._runtime_total_ms,
                "average_runtime_ms": average_runtime,
                "domain_dispatch_count": dict(self._domain_counts),
                "local_memory_entries": int(memory_stats["entries"]),
                "local_memory_hit_rate": float(memory_stats["hit_rate"]),
            }

    def extract_performance_metrics(self, result: Any) -> Dict[str, float]:
        metrics = super().extract_performance_metrics(result)
        if isinstance(result, Mapping):
            duration = result.get("duration_ms")
            if isinstance(duration, (int, float)):
                metrics["latency_ms"] = float(duration)
            cache_hit = result.get("cache_hit")
            if isinstance(cache_hit, bool):
                metrics["cache_hit"] = 1.0 if cache_hit else 0.0
            diagnostics = result.get("diagnostics")
            if isinstance(diagnostics, Mapping) and isinstance(diagnostics.get("converged"), bool):
                metrics["converged"] = 1.0 if diagnostics["converged"] else 0.0
        return metrics

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "agent": self.name,
            "enabled": self.enabled,
            "supported_domains": list(self.supported_domains),
            "default_domain": self.default_domain,
            "available_operations": {domain: sorted(self._dispatch[domain]) for domain in self.supported_domains},
            "shared_memory": {
                "enabled": self.shared_memory_enabled,
                "publish_shared_results": self.publish_shared_results,
            },
            "local_memory": dict(self.local_memory.stats()),
            "checkpointing": {
                "supported": self.supports_checkpointing,
                "enabled": self.checkpointing_enabled,
                "checkpoint_local_memory": self.checkpoint_local_memory,
                "restore_local_memory": self.restore_local_memory,
            },
            "templates": list(self.available_templates()) if self.template_loading_enabled else [],
            "metrics": self.metrics_snapshot(),
            "runtime": self.runtime_status(),
        }


__all__ = ["STEMAgent", "STEMResult"]


if __name__ == "__main__":
    import math

    configure_logging()
    printer.status("TEST", "STEMAgent production façade validation", "info")

    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    agent = STEMAgent(
        shared_memory=shared_memory,
        agent_factory=agent_factory,
        config={"publish_shared_results": False},
    )

    roots = agent.perform_task({"domain": "algebra", "operation": "quadratic_roots", "args": [1, 0, -4]})
    assert sorted(round(complex(root).real, 12) for root in roots["result"]) == [-2.0, 2.0]

    linear = agent.perform_task({
        "domain": "numerical_methods",
        "operation": "solve_linear_system",
        "args": [[[2.0, 1.0], [1.0, 3.0]], [5.0, 6.0]],
    })
    assert abs(linear["result"][0] - 1.8) < 1e-12 and abs(linear["result"][1] - 1.4) < 1e-12

    integral = agent.perform_task({
        "domain": "numerical_methods", "operation": "simpson",
        "args": [lambda x: x * x, 0.0, 1.0], "kwargs": {"n": 100},
    })
    assert abs(integral["result"] - 1.0 / 3.0) < 1e-12

    derivative = agent.perform_task({
        "domain": "numerical_methods", "operation": "central_difference",
        "args": [lambda x: x * x, 3.0],
    })
    assert abs(derivative["result"] - 6.0) < 1e-7

    ode = agent.perform_task({
        "domain": "numerical_methods", "operation": "rk4",
        "args": [lambda _t, y: [y[0]], [1.0], 0.0, 1.0], "kwargs": {"h": 0.01},
    })
    assert abs(ode["result"]["y"][-1][0] - math.e) < 1e-8

    unit_task = {"domain": "units", "operation": "convert", "args": [1.0, "m", "cm"]}
    first = agent.perform_task(unit_task)
    second = agent.perform_task(unit_task)
    assert abs(first["result"] - 100.0) < 1e-12 and second["cache_hit"] is True
    dimension_rejected = False
    try:
        agent.perform_task({"domain": "units", "operation": "convert", "args": [3.0, "m", "s"]})
    except STEMError:
        dimension_rejected = True
    assert dimension_rejected, "dimensionally invalid conversion must fail"

    propagated = agent.perform_task({
        "domain": "uncertainty", "operation": "jacobian_propagation",
        "args": [[2.0, 3.0], [0.1, 0.2]],
    })
    assert abs(propagated["result"] - math.sqrt(0.4)) < 1e-12

    assert agent.get_template("botany_plant_biology.json")["plant_profile"]["taxonomy"]["kingdom"] == "Plantae"

    checkpoint_state = agent._export_checkpoint_state()
    assert "shared_memory" not in checkpoint_state
    restored = STEMAgent(shared_memory=shared_memory, agent_factory=agent_factory, config={"publish_shared_results": False})
    restored._import_checkpoint_state(checkpoint_state)
    after_restore = restored.perform_task(unit_task)
    assert after_restore["cache_hit"] is True

    printer.status("SUCCESS", "STEMAgent numerical, cache, template, and local checkpoint checks passed", "success")
