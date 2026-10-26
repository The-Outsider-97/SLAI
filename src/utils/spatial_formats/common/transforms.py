"""Row-major 4x4 affine matrices acting on column vectors; +Z up, right handed.

Convention
----------
A flat 16-element tuple stores::

    [ m00 m01 m02 m03 ]
    [ m10 m11 m12 m13 ]
    [ m20 m21 m22 m23 ]
    [  0   0   0   1  ]

Points are column vectors; ``p' = M · p``. Composition is ``A · B`` so
``compose(A, B) · p == A · (B · p)``. A 3x3 planar similarity matrix is
also accepted by the ``*2d`` helpers, flattened row-major as::

    [ m00 m01 m02 ]
    [ m10 m11 m12 ]
    [  0   0   1  ]

In both dimensions, the bottom row must be the homogeneous row
``(0 … 1)`` and the linear part must be invertible. ``require_transform``
additionally normalises the linear part before testing rank so that valid
uniform millimetre or micrometre scales are not mistaken for singular
matrices.
"""

import math

from .errors import ValidationError
from .helpers import vector, finite

IDENTITY = (1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.)
IDENTITY_2D = (1., 0., 0., 0., 1., 0., 0., 0., 1.)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def det3(m) -> float:
    """Determinant of the linear 3x3 block of a flat row-major 4x4 matrix.

    Accepts any sequence with at least 11 elements whose first eleven
    entries are row-major ``[m00 … m22]`` interleaved with the unused
    fourth column. :func:`require_transform` relies on this layout.
    """
    return (
        m[0] * (m[5] * m[10] - m[6] * m[9])
        - m[1] * (m[4] * m[10] - m[6] * m[8])
        + m[2] * (m[4] * m[9] - m[5] * m[8])
    )


def require_transform(values) -> tuple[float, ...]:
    out = vector(values, 16, "transform")
    for index, target in ((12, 0.0), (13, 0.0), (14, 0.0), (15, 1.0)):
        if abs(out[index] - target) > 1e-10:
            raise ValidationError(
                "only affine homogeneous transforms are supported",
                context={"index": index},
            )
    scale = max(abs(out[4 * r + c]) for r in range(3) for c in range(3))
    if scale == 0.0:
        raise ValidationError("singular spatial transform")
    normalized = tuple(
        out[i] / scale if (i // 4 < 3 and i % 4 < 3) else out[i]
        for i in range(16)
    )
    if not all(math.isfinite(x) for x in normalized):
        raise ValidationError("transform linear part overflows")
    if abs(det3(normalized)) < 1e-12:
        raise ValidationError("singular spatial transform")
    return out


def require_transform2d(values) -> tuple[float, ...]:
    """Validate a flat row-major 3x3 planar **similarity** matrix.

    The bottom row must be ``(0, 0, 1)`` and the top-left 2x2 must be an
    orthogonal 2x2 block with equal column norms (uniform scale,
    optionally reflected). Non-similarity affine 2x2 blocks are rejected
    because SDRW's batch ``transform`` operation uses the similarity
    convention.
    """
    out = vector(values, 9, "transform2d")
    if abs(out[6]) > 1e-10 or abs(out[7]) > 1e-10 or abs(out[8] - 1.0) > 1e-10:
        raise ValidationError(
            "planar transform must have homogeneous bottom row (0, 0, 1)",
        )
    a, b, c, d = out[0], out[1], out[3], out[4]
    scale = max(abs(a), abs(b), abs(c), abs(d))
    if scale == 0.0:
        raise ValidationError("singular planar transform")
    na, nb, nc, nd = a / scale, b / scale, c / scale, d / scale
    n0 = na * na + nc * nc
    n1 = nb * nb + nd * nd
    dot = na * nb + nc * nd
    if abs(n0 - n1) > 1e-10 or abs(dot) > 1e-10:
        raise ValidationError("planar transform is not a similarity")
    return out


# ---------------------------------------------------------------------------
# Composition and inversion
# ---------------------------------------------------------------------------

def compose(left, right) -> tuple[float, ...]:
    a, b = require_transform(left), require_transform(right)
    return require_transform(tuple(
        sum(a[4 * r + k] * b[4 * k + c] for k in range(4))
        for r in range(4) for c in range(4)
    ))


def compose_many(*transforms) -> tuple[float, ...]:
    """Left-to-right composition of two or more 4x4 transforms."""
    if not transforms:
        return IDENTITY
    result = require_transform(transforms[0])
    for nxt in transforms[1:]:
        result = compose(result, nxt)
    return result


def invert_transform(m) -> tuple[float, ...]:
    a = require_transform(m)
    rows = [[float(a[4 * r + c]) for c in range(4)] + [float(r == c) for c in range(4)]
            for r in range(4)]
    for j in range(4):
        pivot = max(range(j, 4), key=lambda i: abs(rows[i][j]))
        if rows[pivot][j] == 0:
            raise ValidationError("singular transform")
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scalar = rows[j][j]
        rows[j] = [x / scalar for x in rows[j]]
        for r in range(4):
            if r != j:
                factor = rows[r][j]
                rows[r] = [x - factor * y for x, y in zip(rows[r], rows[j])]
    return require_transform([rows[r][c + 4] for r in range(4) for c in range(4)])


def rigid_inverse(m) -> tuple[float, ...]:
    """Fast inverse for a rigid transform (rotation + translation, no scale).

    Falls back to :func:`invert_transform` (which raises) when the matrix
    is not rigid. Useful for adapter code that constructs many world-space
    queries from instance transforms.
    """
    a = require_transform(m)
    if not is_rigid(a):
        raise ValidationError("rigid_inverse requires a rigid transform")
    r00, r01, r02, tx = a[0], a[1], a[2], a[3]
    r10, r11, r12, ty = a[4], a[5], a[6], a[7]
    r20, r21, r22, tz = a[8], a[9], a[10], a[11]
    return (
        r00, r10, r20, -(r00 * tx + r10 * ty + r20 * tz),
        r01, r11, r21, -(r01 * tx + r11 * ty + r21 * tz),
        r02, r12, r22, -(r02 * tx + r12 * ty + r22 * tz),
        0.0, 0.0, 0.0, 1.0,
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def transform_point(m, xyz) -> tuple[float, float, float]:
    m = require_transform(m)
    p = vector(xyz, 3)
    return (
        finite(sum(m[k] * p[k] for k in range(3)) + m[3]),
        finite(sum(m[4 + k] * p[k] for k in range(3)) + m[7]),
        finite(sum(m[8 + k] * p[k] for k in range(3)) + m[11]),
    )


def transform_vector(m, xyz) -> tuple[float, float, float]:
    """Apply only the linear block; translation is ignored."""
    m = require_transform(m)
    p = vector(xyz, 3)
    return (
        finite(sum(m[k] * p[k] for k in range(3))),
        finite(sum(m[4 + k] * p[k] for k in range(3))),
        finite(sum(m[8 + k] * p[k] for k in range(3))),
    )


def transform_normal(m, xyz) -> tuple[float, float, float]:
    """Transform a normal by the inverse-transpose of the linear block."""
    a = require_transform(m)
    v = vector(xyz, 3)
    inv = invert_transform(a)
    n0 = inv[0] * v[0] + inv[4] * v[1] + inv[8] * v[2]
    n1 = inv[1] * v[0] + inv[5] * v[1] + inv[9] * v[2]
    n2 = inv[2] * v[0] + inv[6] * v[1] + inv[10] * v[2]
    norm = math.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
    if norm < 1e-300:
        raise ValidationError("normal collapses under transform")
    return (n0 / norm, n1 / norm, n2 / norm)


def transform_bounds(m, minimum, maximum):
    """Transform an AABB and return a conservative AABB of its image."""
    a = require_transform(m)
    lo = vector(minimum, 3, "minimum")
    hi = vector(maximum, 3, "maximum")
    if any(lo[i] > hi[i] for i in range(3)):
        raise ValidationError("minimum exceeds maximum in bounds")
    points = [
        transform_point(a, (x, y, z))
        for x in (lo[0], hi[0])
        for y in (lo[1], hi[1])
        for z in (lo[2], hi[2])
    ]
    return (
        (min(p[0] for p in points), min(p[1] for p in points), min(p[2] for p in points)),
        (max(p[0] for p in points), max(p[1] for p in points), max(p[2] for p in points)),
    )


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def translation_of(m) -> tuple[float, float, float]:
    a = require_transform(m)
    return (a[3], a[7], a[11])


def linear_part(m) -> tuple[float, ...]:
    a = require_transform(m)
    return (a[0], a[1], a[2], a[4], a[5], a[6], a[8], a[9], a[10])


def determinant(m) -> float:
    return det3(require_transform(m))


def scale_factors(m) -> tuple[float, float, float]:
    """Column norms of the linear block (positive by convention)."""
    a = require_transform(m)
    return (
        math.sqrt(a[0] ** 2 + a[4] ** 2 + a[8] ** 2),
        math.sqrt(a[1] ** 2 + a[5] ** 2 + a[9] ** 2),
        math.sqrt(a[2] ** 2 + a[6] ** 2 + a[10] ** 2),
    )


def is_identity(m, tolerance: float = 1e-12) -> bool:
    try:
        a = require_transform(m)
    except ValidationError:
        return False
    return all(abs(x - y) <= tolerance for x, y in zip(a, IDENTITY))


def is_rigid(m, tolerance: float = 1e-7) -> bool:
    """True for rotation + translation without scale or reflection."""
    a = require_transform(m)
    columns = [[a[4 * r + c] for r in range(3)] for c in range(3)]
    norms = [math.sqrt(sum(x * x for x in col)) for col in columns]
    if any(abs(n - 1.0) > tolerance for n in norms):
        return False
    return det3(a) > 0 and all(
        abs(sum(x * y for x, y in zip(columns[i], columns[j])) - float(i == j)) < tolerance
        for i in range(3) for j in range(3)
    )


def is_similarity(m, tolerance: float = 1e-7) -> bool:
    """True for uniform positive scale + rotation + translation."""
    a = require_transform(m)
    columns = [[a[4 * r + c] for r in range(3)] for c in range(3)]
    norms = [math.sqrt(sum(x * x for x in col)) for col in columns]
    mean = sum(norms) / 3.0
    if mean < 1e-300 or any(abs(n - mean) > tolerance * mean for n in norms):
        return False
    normalized = [[x / n for x in col] for col, n in zip(columns, norms)]
    return det3(a) > 0 and all(
        abs(sum(x * y for x, y in zip(normalized[i], normalized[j])) - float(i == j)) < tolerance
        for i in range(3) for j in range(3)
    )


def is_rotation(m, tolerance: float = 1e-7) -> bool:
    """True for a pure rotation (no translation component)."""
    a = require_transform(m)
    if any(abs(a[i]) > tolerance for i in (3, 7, 11)):
        return False
    return is_rigid(a, tolerance)


def almost_equal(left, right, tolerance: float = 1e-9) -> bool:
    try:
        a = require_transform(left)
        b = require_transform(right)
    except ValidationError:
        return False
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


def to_list(m) -> list[float]:
    return list(require_transform(m))


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

def from_translation(x: float, y: float, z: float) -> tuple[float, ...]:
    return (1.0, 0.0, 0.0, finite(x, "x"),
            0.0, 1.0, 0.0, finite(y, "y"),
            0.0, 0.0, 1.0, finite(z, "z"),
            0.0, 0.0, 0.0, 1.0)


def from_scale(sx: float, sy: float | None = None, sz: float | None = None) -> tuple[float, ...]:
    sx = finite(sx, "sx")
    sy = sx if sy is None else finite(sy, "sy")
    sz = sx if sz is None else finite(sz, "sz")
    return (sx, 0.0, 0.0, 0.0,
            0.0, sy, 0.0, 0.0,
            0.0, 0.0, sz, 0.0,
            0.0, 0.0, 0.0, 1.0)


def from_rotation_x(angle: float) -> tuple[float, ...]:
    c, s = math.cos(angle), math.sin(angle)
    return (1.0, 0.0, 0.0, 0.0,
            0.0, c, -s, 0.0,
            0.0, s, c, 0.0,
            0.0, 0.0, 0.0, 1.0)


def from_rotation_y(angle: float) -> tuple[float, ...]:
    c, s = math.cos(angle), math.sin(angle)
    return (c, 0.0, s, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -s, 0.0, c, 0.0,
            0.0, 0.0, 0.0, 1.0)


def from_rotation_z(angle: float) -> tuple[float, ...]:
    c, s = math.cos(angle), math.sin(angle)
    return (c, -s, 0.0, 0.0,
            s, c, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0)


def from_axis_angle(axis, angle: float) -> tuple[float, ...]:
    """Rodrigues rotation around an arbitrary axis; ``axis`` is normalised."""
    kx, ky, kz = vector(axis, 3, "axis")
    norm = math.sqrt(kx * kx + ky * ky + kz * kz)
    if norm < 1e-300:
        raise ValidationError("rotation axis must be nonzero")
    kx, ky, kz = kx / norm, ky / norm, kz / norm
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    r00 = c + kx * kx * one_minus_c
    r01 = kx * ky * one_minus_c - kz * s
    r02 = kx * kz * one_minus_c + ky * s
    r10 = ky * kx * one_minus_c + kz * s
    r11 = c + ky * ky * one_minus_c
    r12 = ky * kz * one_minus_c - kx * s
    r20 = kz * kx * one_minus_c - ky * s
    r21 = kz * ky * one_minus_c + kx * s
    r22 = c + kz * kz * one_minus_c
    return (r00, r01, r02, 0.0,
            r10, r11, r12, 0.0,
            r20, r21, r22, 0.0,
            0.0, 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Planar (SDRW) helpers
# ---------------------------------------------------------------------------

def compose2d(left, right) -> tuple[float, ...]:
    a, b = require_transform2d(left), require_transform2d(right)
    return require_transform2d(tuple(
        sum(a[3 * r + k] * b[3 * k + c] for k in range(3))
        for r in range(3) for c in range(3)
    ))


def invert_transform2d(m) -> tuple[float, ...]:
    a = require_transform2d(m)
    # Similarity inverse: R^T / s^2 for the linear 2x2 block, -inverse·t.
    a00, a01, a10, a11 = a[0], a[1], a[3], a[4]
    det = a00 * a11 - a01 * a10
    if abs(det) < 1e-30:
        raise ValidationError("singular planar transform")
    inv00, inv01 = a11 / det, -a01 / det
    inv10, inv11 = -a10 / det, a00 / det
    tx, ty = a[2], a[5]
    return require_transform2d((
        inv00, inv01, -(inv00 * tx + inv01 * ty),
        inv10, inv11, -(inv10 * tx + inv11 * ty),
        0.0, 0.0, 1.0,
    ))


def transform_point2d(m, xy) -> tuple[float, float]:
    a = require_transform2d(m)
    x, y = vector(xy, 2, "point2d")
    return (
        finite(a[0] * x + a[1] * y + a[2]),
        finite(a[3] * x + a[4] * y + a[5]),
    )


def is_rigid2d(m, tolerance: float = 1e-7) -> bool:
    a = require_transform2d(m)
    a00, a01, a10, a11 = a[0], a[1], a[3], a[4]
    if abs(a00 * a00 + a10 * a10 - 1.0) > tolerance:
        return False
    if abs(a01 * a01 + a11 * a11 - 1.0) > tolerance:
        return False
    if abs(a00 * a01 + a10 * a11) > tolerance:
        return False
    return a00 * a11 - a01 * a10 > 0


__all__ = [
    "IDENTITY",
    "IDENTITY_2D",
    "require_transform",
    "require_transform2d",
    "det3",
    "determinant",
    "compose",
    "compose_many",
    "invert_transform",
    "transform_point",
    "transform_vector",
    "transform_normal",
    "transform_bounds",
    "translation_of",
    "linear_part",
    "scale_factors",
    "is_identity",
    "is_rigid",
    "is_similarity",
    "is_rotation",
    "almost_equal",
    "to_list",
    "from_translation",
    "from_scale",
    "from_rotation_x",
    "from_rotation_y",
    "from_rotation_z",
    "from_axis_angle",
    "rigid_inverse",
    "compose2d",
    "invert_transform2d",
    "transform_point2d",
    "is_rigid2d",
]
