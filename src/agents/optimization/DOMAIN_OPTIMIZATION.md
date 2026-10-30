# SLAI v2.4 — Reusable optimization domain problem builders

This is an incremental enhancement, not a replacement for the existing `OptimizationAgent` or its solvers. All four builders return **the existing canonical `OptimizationProblem`**. They do **not** solve, serialize or apply changes to external systems. Use the existing `OptimizationSolver` / `OptimizationAgent` to execute a returned problem and inspect feasibility, termination, and optimality separately.

## Implemented scope

- `geometry_parameter_problem`: accepts arbitrary typed variables, a trusted objective callback and canonical constraints; suitable for spatial and design parameter evaluation. Geometry kernels remain outside optimization.
- `trajectory_problem`: bounded or unbounded real-valued 2D/3D **interior waypoint** variables, fixed endpoints, a fixed duration, finite-difference acceleration/jerk objective, optional sampled velocity/acceleration/jerk upper bounds and externally evaluated minimum sampled clearance. `trajectory_samples` expands a result to its fixed-endpoint trajectory.
- `camera_pose_problem`: 3D camera-position optimization with a fixed look-at target, preferred-position and target-distance objectives, optional external visibility callback. `camera_look_at_basis` constructs a right-handed view basis and rejects coincident target positions and up-axis degeneracy.
- `animation_curve_problem`: scalar cubic Hermite tangent optimization with fixed knots, exact per-segment integral of squared jerk, optional fixed tangents, bounded tangents, exact C2 matching at internal knots and analytically checked segment monotonicity. `hermite_curve_value` evaluates a Hermite curve at a requested time.

## Example

```python
from src.agents.optimization.domain_problems import (
    geometry_parameter_problem, animation_curve_problem
)
from src.agents.optimization.optimization_types import Variable, VariableDomain
from src.agents.optimization.optimization_math import with_classification
from src.agents.optimization.optimization_solver import OptimizationSolver

p = geometry_parameter_problem(
    "fitting", variables=(Variable("width", VariableDomain.real(1, 5), initial=2),),
    objective=lambda x: (x["width"] - 3) ** 2,
)
result = OptimizationSolver().solve_single_detailed(with_classification(p))
assert result.solution is not None
# Before applying any proposal, check result.solution.feasibility, status,
# termination_reason, and independent domain-specific validation.

curve = animation_curve_problem(
    "curve", times=(0.0, 1.0, 2.0), values=(0.0, 1.0, 2.0),
    fixed_tangents={0: 1.0, 2: 1.0}, require_c2=True,
)
```

## Mathematical and architectural limitations

1. Trajectory velocity, acceleration, and jerk are finite-difference **sampled** quantities. No guarantee is given for a continuous-time curve or motion through unsampled obstacles. The task consumer must check continuous-time conditions independently if needed. The trajectory objective is a discrete approximation, **not** an exact continuous minimum-jerk solution.
2. Camera orientation is a derived look-at basis only. There is no camera-path timing, field-of-view model, renderer, quaternion manifold optimization, or intrinsically meaningful measure of occlusion. The visibility callback is the caller's responsibility.
3. Hermite curve optimization changes only tangents; knot times and values remain fixed. Continuity is C1 by construction and C2 optionally. Monotonicity is tested analytically per cubic segment, not through sampling. Intentional discontinuities and higher-order continuity are outside this model. The smoothness objective is evaluated exactly for the cubic representation.
4. Arbitrary external objective/constraint callbacks are **trusted code** and are not sandboxed. No security boundary is established by these builders. The caller owns evaluator isolation, units, caching/versioning, thread safety and effect prevention.
5. Each builder can expose an `OptimizationProblem` accepted by the registry, but a matching solver may be unavailable, or a solver may terminate without finding a feasible point. There is no guarantee of global optimality.
6. These do not introduce a new memory, config, factory or optimization algorithm; they rely on established SLAI v2.4 components and do not import SPADRA or other applications.

## Verification

`python -m pytest -q tests/agents/test_optimization_domain_problems.py`

Includes two integration smoke tests using the canonical nonlinear solver. A separate local formula/unit test harness may be used to verify mathematical behavior when the full SLAI runtime cannot be imported, but it **does not** establish real factory, memory, or solver integration readiness.

## Deferred work

Full source-wide numerical audit, robust/stochastic optimization, external solver integrations, multimodal/mixed-variable optimization, complete cancellation/checkpointing and uninterrupted full-repository regression tests remain pending. This extension must not be described as completion of all SLAI Optimization Agent production readiness gates.