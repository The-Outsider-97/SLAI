"""Deterministic interpolation primitives for animation data.

Scope
-----
* Scalars: linear, smoothstep/smootherstep easing, Hermite and uniform
  Catmull-Rom splines over ordered control points.
* Component sequences: component-wise linear interpolation of finite
  equal-length sequences.
* Angles: shortest-arc interpolation with wrap-around at ``+/-pi``.
* Quaternions ``(w, x, y, z)`` (scalar-first): normalised linear
  interpolation (nlerp), spherical linear interpolation (slerp), and a
  minimal algebra set (multiply, conjugate, inverse, from/to axis-angle,
  from Euler ZYX).

Out of scope
------------
* Timebase management (see :mod:`timebase`), playback position mapping
  (see :mod:`playback`), and keyframe track schemas (owned by SGEO; see
  :func:`validate_animation`).
* Non-uniform Catmull-Rom variants (centripetal/chordal), which require
  per-keyframe time spacing and belong in a future spline layer that
  consumes a full track.

Design invariants
-----------------
* All inputs are validated through :mod:`animation_helpers`; every
  failure surfaces as an :class:`AnimationValidationError`.
* Quaternion components are ordered scalar-first ``(w, x, y, z)`` to
  match the existing ``slai.animation.v1`` track schema and the shared
  helpers.
* Pure functions only: no I/O, no logging, no global state.
* Hemisphere correction (``dot < 0 ⇒ flip q1``) is applied by both
  ``slerp`` and ``nlerp`` so ``slerp(q, -q, a)`` returns ``q`` rather
  than rotating the long way round.
* The zero-quaternion guard is shared with :mod:`animation_helpers`
  (``_QUATERNION_EPSILON == 1e-15``) so validation, normalisation, and
  interpolation cannot disagree about what "zero magnitude" means.
* ``slerp`` falls back to an inlined nlerp branch when ``dot > 0.9995``
  (equivalently: separation below ≈3.6°). The nlerp error grows as
  :math:`O(\\theta^2)` in that regime and is negligible for real
  animation data; the fallback also avoids the ``sin(θ) → 0`` numerical
  cliff as ``θ → 0``.

References
----------
* Shoemake, K. (1985). "Animating rotation with quaternion curves."
  Proc. SIGGRAPH '85.
* Kochanek, D. & Bartels, R. (1984). "Interpolating splines with local
  tension, continuity, and bias control." Proc. SIGGRAPH '84.
* Goldberg, D. (1991). "What every computer scientist should know about
  floating-point arithmetic." ACM Computing Surveys.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import acos, atan2, cos, pi, sin, sqrt

from ..utils.animation_errors import AnimationValidationError
from ..utils.animation_helpers import (
    ensure_finite_scalar,
    ensure_quaternion,
    ensure_unit_interval,
    normalize_quaternion,
)

# Slerp / nlerp crossover.  ``dot > 0.9995`` corresponds to a rotational
# separation below ≈3.6°, where nlerp deviates from slerp by O(θ²).
_SLERP_NLERP_THRESHOLD = 0.9995

# Shared with :mod:`animation_helpers` so quaternion validity is judged
# identically across the animation stack.
_QUATERNION_EPSILON = 1e-15


# ---------------------------------------------------------------------------
# Scalar interpolation
# ---------------------------------------------------------------------------
def lerp(a: float, b: float, alpha: float) -> float:
    """Classic scalar linear interpolation.

    Computed as ``a + alpha * (b - a)`` rather than ``(1 - alpha) * a +
    alpha * b`` because the former is better conditioned when ``a`` and
    ``b`` differ by many orders of magnitude (only one multiply is
    performed; the subtraction of two near-equal large numbers that
    would otherwise occur in ``(1 - alpha) * a`` is avoided).

    Raises
    ------
    AnimationValidationError
        If any argument is a non-finite real or ``bool``, or if ``alpha``
        is outside ``[0, 1]``.
    """
    a_f = ensure_finite_scalar(a, "a")
    b_f = ensure_finite_scalar(b, "b")
    alpha_f = ensure_unit_interval(alpha, "alpha")
    return a_f + alpha_f * (b_f - a_f)


def smoothstep(alpha: float) -> float:
    """Classic Hermite ease-in/ease-out over ``[0, 1]``.

    ``3α² - 2α³``.  Zero first derivative at both endpoints, useful as a
    default easing for one-shot transitions (fade-ins, position tweens).

    Raises
    ------
    AnimationValidationError
        If ``alpha`` is not a finite real in ``[0, 1]``.
    """
    a = ensure_unit_interval(alpha, "alpha")
    return a * a * (3.0 - 2.0 * a)


def smootherstep(alpha: float) -> float:
    """Perlin's quintic variant: ``6α⁵ - 15α⁴ + 10α³``.

    Zero first *and second* derivative at both endpoints, which removes
    the slight acceleration discontinuity at the joins of chained
    ``smoothstep`` segments.  Preferred over :func:`smoothstep` when
    segments are concatenated (camera paths, sequential fades).

    Raises
    ------
    AnimationValidationError
        If ``alpha`` is not a finite real in ``[0, 1]``.
    """
    a = ensure_unit_interval(alpha, "alpha")
    return a * a * a * (a * (a * 6.0 - 15.0) + 10.0)


def lerp_angle(a: float, b: float, alpha: float) -> float:
    """Shortest-arc interpolation between two angles (radians).

    The result is not normalised into ``(-π, π]``; it tracks the shortest
    path *from* ``a`` *toward* ``b``, which is what animation code
    typically wants (avoiding 359° → 1° spinning through 180°).

    The "exactly half-turn" case (``b - a ≡ π (mod 2π)``) is resolved
    toward ``-π``, matching the Unity ``Mathf.LerpAngle`` / three.js
    convention.

    Raises
    ------
    AnimationValidationError
        If any argument is a non-finite real or ``bool``, or if ``alpha``
        is outside ``[0, 1]``.
    """
    a_f = ensure_finite_scalar(a, "a")
    b_f = ensure_finite_scalar(b, "b")
    alpha_f = ensure_unit_interval(alpha, "alpha")
    delta = ((b_f - a_f + pi) % (2.0 * pi)) - pi
    return a_f + delta * alpha_f


def lerp_components(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float,
) -> tuple[float, ...]:
    """Component-wise linear interpolation of two finite equal-length sequences.

    Accepts tuples, lists, and any other ``Sequence`` of reals.  Rejects
    strings/bytes (they are ``Sequence``s but never valid component
    containers), rejects mismatched lengths eagerly, and validates each
    component with :func:`ensure_finite_scalar` so the first bad element
    reports a precise index in its context.

    Raises
    ------
    AnimationValidationError
        For non-sequence inputs, length mismatch, non-finite components,
        or an out-of-range ``alpha``.
    """
    alpha_f = ensure_unit_interval(alpha, "alpha")
    for name, seq in (("a", a), ("b", b)):
        if isinstance(seq, (str, bytes, bytearray)) or not isinstance(seq, Sequence):
            raise AnimationValidationError(
                f"{name} must be a finite sequence, got {type(seq).__name__}",
                context={"field": name, "received_type": type(seq).__name__},
            )
    if len(a) != len(b):
        raise AnimationValidationError(
            f"lerp_components requires equal lengths, got {len(a)} and {len(b)}",
            context={"len_a": len(a), "len_b": len(b)},
        )
    result = []
    for i, (av, bv) in enumerate(zip(a, b)):
        a_f = ensure_finite_scalar(av, f"a[{i}]")
        b_f = ensure_finite_scalar(bv, f"b[{i}]")
        result.append(a_f + alpha_f * (b_f - a_f))
    return tuple(result)


# ---------------------------------------------------------------------------
# Spline interpolation
# ---------------------------------------------------------------------------
def cubic_hermite(
    p0: float,
    m0: float,
    p1: float,
    m1: float,
    alpha: float,
) -> float:
    """Cubic Hermite basis evaluated at ``alpha ∈ [0, 1]``.

    Evaluates ``h00·p0 + h10·m0 + h01·p1 + h11·m1`` using the standard
    Hermite polynomials:
    ``h00 = 2t³ - 3t² + 1``, ``h10 = t³ - 2t² + t``,
    ``h01 = -2t³ + 3t²``, ``h11 = t³ - t²``.

    ``m0``/``m1`` are the *time-derivatives* of the curve at ``p0``/``p1``
    expressed in the same units as the value; when ``alpha`` is the time
    fraction of the segment, scaling ``m0``/``m1`` by the segment
    duration yields unit-consistent tangents.

    Raises
    ------
    AnimationValidationError
        For non-finite inputs or an out-of-range ``alpha``.
    """
    p0_f = ensure_finite_scalar(p0, "p0")
    m0_f = ensure_finite_scalar(m0, "m0")
    p1_f = ensure_finite_scalar(p1, "p1")
    m1_f = ensure_finite_scalar(m1, "m1")
    t = ensure_unit_interval(alpha, "alpha")
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * p0_f + h10 * m0_f + h01 * p1_f + h11 * m1_f


def catmull_rom(
    p0: float,
    p1: float,
    p2: float,
    p3: float,
    alpha: float,
) -> float:
    """Uniform Catmull-Rom interpolation between ``p1`` and ``p2``.

    ``p0`` and ``p3`` are the neighbouring control points that determine
    the tangents at ``p1`` and ``p2`` respectively.  The tangents are the
    central differences ``m1 = (p2 - p0) / 2`` and
    ``m2 = (p3 - p1) / 2`` — the standard uniform (non-centripetal)
    variant.

    The uniform variant is only correct when ``p0``, ``p1``, ``p2``,
    ``p3`` are *equally spaced in time*.  For non-uniform keyframe
    spacing, a future chordal/centripetal Catmull-Rom variant will be
    needed; this function is intentionally documented as uniform-only.

    Raises
    ------
    AnimationValidationError
        For non-finite inputs or an out-of-range ``alpha``.
    """
    p0_f = ensure_finite_scalar(p0, "p0")
    p1_f = ensure_finite_scalar(p1, "p1")
    p2_f = ensure_finite_scalar(p2, "p2")
    p3_f = ensure_finite_scalar(p3, "p3")
    alpha_f = ensure_unit_interval(alpha, "alpha")
    m1 = (p2_f - p0_f) * 0.5
    m2 = (p3_f - p1_f) * 0.5
    return cubic_hermite(p1_f, m1, p2_f, m2, alpha_f)


# ---------------------------------------------------------------------------
# Quaternion interpolation
# ---------------------------------------------------------------------------
def nlerp(a, b, alpha: float) -> tuple[float, float, float, float]:
    """Normalised linear interpolation of two quaternions.

    Cheaper than :func:`slerp` and monotonic in angle, but with
    non-constant angular velocity.  Appropriate for short arcs (the same
    regime in which :func:`slerp` internally falls back to nlerp) and for
    cases where constant velocity is not required.

    Raises
    ------
    AnimationValidationError
        For malformed quaternions, non-unit ``alpha``, or a zero-magnitude
        blended result (impossible when the hemisphere correction is
        applied to valid unit quaternions).
    """
    alpha_f = ensure_unit_interval(alpha, "alpha")
    q0 = normalize_quaternion(a, "a")
    q1 = normalize_quaternion(b, "b")
    dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]
    if dot < 0.0:
        # q1 and -q1 encode the same rotation; pick the hemisphere that
        # minimises the arc so nlerp does not take the long way round.
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
    blended = (
        q0[0] + alpha_f * (q1[0] - q0[0]),
        q0[1] + alpha_f * (q1[1] - q0[1]),
        q0[2] + alpha_f * (q1[2] - q0[2]),
        q0[3] + alpha_f * (q1[3] - q0[3]),
    )
    return normalize_quaternion(blended)


def slerp(a, b, alpha: float) -> tuple[float, float, float, float]:
    """Spherical linear interpolation of two quaternions (Shoemake 1985).

    Short-arc, constant-angular-velocity interpolation between two unit
    quaternions.  For very close inputs (``dot > 0.9995``) the
    implementation switches to an inlined nlerp branch — both because the
    constant-velocity property is imperceptible at that scale and because
    the classical slerp formula divides by ``sin(θ)``, which underflows
    as ``θ → 0``.  The hemisphere correction (``dot < 0 ⇒ flip q1``)
    ensures ``slerp(q, -q, α)`` returns ``q`` for all ``α``.

    Raises
    ------
    AnimationValidationError
        For malformed quaternions, non-unit ``alpha``, or a zero-magnitude
        input.
    """
    alpha_f = ensure_unit_interval(alpha, "alpha")
    q0 = normalize_quaternion(a, "a")
    q1 = normalize_quaternion(b, "b")
    dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]
    if dot < 0.0:
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
        dot = -dot
    # Numerical clamp: acos is undefined outside [-1, 1], and float
    # accumulation can push dot a few ULPs past 1.0 even for unit inputs.
    dot = max(-1.0, min(1.0, dot))
    if dot > _SLERP_NLERP_THRESHOLD:
        blended = (
            q0[0] + alpha_f * (q1[0] - q0[0]),
            q0[1] + alpha_f * (q1[1] - q0[1]),
            q0[2] + alpha_f * (q1[2] - q0[2]),
            q0[3] + alpha_f * (q1[3] - q0[3]),
        )
        return normalize_quaternion(blended)
    theta = acos(dot)
    sin_theta = sin(theta)
    if abs(sin_theta) < _QUATERNION_EPSILON:
        # Unreachable by construction (the nlerp branch catches dot > 0.9995
        # and the hemisphere correction catches dot < 0), kept as a
        # defensive guard against future threshold changes.
        return normalize_quaternion((
            q0[0] + alpha_f * (q1[0] - q0[0]),
            q0[1] + alpha_f * (q1[1] - q0[1]),
            q0[2] + alpha_f * (q1[2] - q0[2]),
            q0[3] + alpha_f * (q1[3] - q0[3]),
        ))
    s0 = sin((1.0 - alpha_f) * theta) / sin_theta
    s1 = sin(alpha_f * theta) / sin_theta
    return normalize_quaternion((
        s0 * q0[0] + s1 * q1[0],
        s0 * q0[1] + s1 * q1[1],
        s0 * q0[2] + s1 * q1[2],
        s0 * q0[3] + s1 * q1[3],
    ))


# ---------------------------------------------------------------------------
# Quaternion algebra
# ---------------------------------------------------------------------------
def quat_multiply(p, q) -> tuple[float, float, float, float]:
    """Hamilton product ``p * q`` of two quaternions.

    Conventions: ``(w, x, y, z)`` with ``i² = j² = k² = ijk = -1``.
    Composition order follows ``(p * q)`` applying ``q`` first then
    ``p`` when used to rotate a vector as ``p * q * v * q* * p*``.

    Raises
    ------
    AnimationValidationError
        For malformed quaternion inputs.
    """
    pw, px, py, pz = ensure_quaternion(p, "p")
    qw, qx, qy, qz = ensure_quaternion(q, "q")
    return (
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    )


def quat_conjugate(q) -> tuple[float, float, float, float]:
    """Conjugate ``(w, -x, -y, -z)`` — the inverse of a *unit* quaternion."""
    w, x, y, z = ensure_quaternion(q, "q")
    return (w, -x, -y, -z)


def quat_inverse(q) -> tuple[float, float, float, float]:
    """Multiplicative inverse of a non-zero quaternion.

    For unit quaternions this equals :func:`quat_conjugate`; for general
    quaternions the conjugate must be divided by ``|q|²``.  Both branches
    are handled here so callers need not special-case.

    Raises
    ------
    AnimationValidationError
        For malformed or zero-magnitude quaternion inputs.
    """
    w, x, y, z = ensure_quaternion(q, "q")
    norm_sq = w * w + x * x + y * y + z * z
    if norm_sq < _QUATERNION_EPSILON:
        raise AnimationValidationError(
            "quat_inverse requires a non-zero quaternion",
            context={"quaternion": (w, x, y, z), "norm_squared": norm_sq},
        )
    return (w / norm_sq, -x / norm_sq, -y / norm_sq, -z / norm_sq)


def quat_from_axis_angle(
    axis: Sequence[float],
    angle: float,
) -> tuple[float, float, float, float]:
    """Build a quaternion from an axis-angle rotation.

    ``axis`` need not be unit length — it will be normalised.  A
    zero-magnitude axis is rejected; there is no meaningful rotation to
    construct in that case.

    Raises
    ------
    AnimationValidationError
        For non-finite ``angle``, non-sequence or wrong-length ``axis``,
        or a zero-magnitude axis.
    """
    if isinstance(axis, (str, bytes, bytearray)) or not isinstance(axis, Sequence):
        raise AnimationValidationError(
            f"axis must be a 3-element sequence, got {type(axis).__name__}",
            context={"field": "axis", "received_type": type(axis).__name__},
        )
    if len(axis) != 3:
        raise AnimationValidationError(
            f"axis must contain exactly 3 components, got {len(axis)}",
            context={"field": "axis", "length": len(axis)},
        )
    ax = (
        ensure_finite_scalar(axis[0], "axis[0]"),
        ensure_finite_scalar(axis[1], "axis[1]"),
        ensure_finite_scalar(axis[2], "axis[2]"),
    )
    magnitude = sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2])
    if magnitude < _QUATERNION_EPSILON:
        raise AnimationValidationError(
            "axis must have non-zero magnitude",
            context={"axis": ax, "magnitude": magnitude},
        )
    ux, uy, uz = ax[0] / magnitude, ax[1] / magnitude, ax[2] / magnitude
    half = ensure_finite_scalar(angle, "angle") * 0.5
    s = sin(half)
    return (cos(half), s * ux, s * uy, s * uz)


def quat_to_axis_angle(
    q,
) -> tuple[tuple[float, float, float], float]:
    """Decompose a quaternion into ``(unit_axis, angle)``.

    The ``w < 0`` hemisphere is first normalised to ``w >= 0`` (which
    represents the same rotation but with a smaller angle), producing the
    *minimal* rotation for a given orientation.  The identity quaternion
    returns ``((1, 0, 0), 0)`` because the axis is undefined in that
    case.

    Raises
    ------
    AnimationValidationError
        For malformed or zero-magnitude quaternion inputs.
    """
    w, x, y, z = normalize_quaternion(q, "q")
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    s = sqrt(x * x + y * y + z * z)
    if s < _QUATERNION_EPSILON:
        return ((1.0, 0.0, 0.0), 0.0)
    return ((x / s, y / s, z / s), 2.0 * atan2(s, w))


def quat_from_euler(
    roll: float,
    pitch: float,
    yaw: float,
) -> tuple[float, float, float, float]:
    """Build a quaternion from roll-pitch-yaw using the ZYX convention.

    The composition is ``Rz(yaw) · Ry(pitch) · Rx(roll)`` — standard
    aerospace ZYX extrinsic (equivalently XYZ intrinsic), matching most
    flight-dynamics and gimbal-oriented animation systems.  Other Euler
    orders are out of scope for this module; callers needing them should
    compose two :func:`quat_from_axis_angle` rotations and multiply with
    :func:`quat_multiply`.

    Raises
    ------
    AnimationValidationError
        If any of the angles is a non-finite real or ``bool``.
    """
    half_roll = ensure_finite_scalar(roll, "roll") * 0.5
    half_pitch = ensure_finite_scalar(pitch, "pitch") * 0.5
    half_yaw = ensure_finite_scalar(yaw, "yaw") * 0.5
    cr, sr = cos(half_roll), sin(half_roll)
    cp, sp = cos(half_pitch), sin(half_pitch)
    cy, sy = cos(half_yaw), sin(half_yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


__all__ = [
    # Scalar
    "lerp",
    "smoothstep",
    "smootherstep",
    "lerp_angle",
    "lerp_components",
    # Spline
    "cubic_hermite",
    "catmull_rom",
    # Quaternion interpolation
    "nlerp",
    "slerp",
    # Quaternion algebra
    "quat_multiply",
    "quat_conjugate",
    "quat_inverse",
    "quat_from_axis_angle",
    "quat_to_axis_angle",
    "quat_from_euler",
]