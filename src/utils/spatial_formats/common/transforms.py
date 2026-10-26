"""Row-major 4x4 affine matrices acting on column vectors; +Z up, right handed."""
from .errors import ValidationError
from .helpers import vector, finite

IDENTITY = (1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.)

def require_transform(values) -> tuple[float, ...]:
    out = vector(values, 16, "transform")
    if any(abs(out[i] - target) > 1e-10 for i, target in zip((12,13,14,15),(0,0,0,1))):
        raise ValidationError("only affine homogeneous transforms are supported")
    # Normalize the linear part before testing rank: valid uniform millimetre
    # or micrometre scales must not be mistaken for singular matrices.
    scale=max(abs(out[4*r+c]) for r in range(3) for c in range(3))
    normalized=tuple(v/scale if i//4<3 and i%4<3 else v for i,v in enumerate(out)) if scale else out
    if not scale or abs(det3(normalized)) < 1e-12:
        raise ValidationError("singular spatial transform")
    return out

def det3(m) -> float:
    return (m[0]*(m[5]*m[10]-m[6]*m[9])
            -m[1]*(m[4]*m[10]-m[6]*m[8])
            +m[2]*(m[4]*m[9]-m[5]*m[8]))

def compose(left, right) -> tuple[float,...]:
    a, b = require_transform(left), require_transform(right)
    return require_transform(tuple(sum(a[4*r+k] * b[4*k+c] for k in range(4)) for r in range(4) for c in range(4)))

def transform_point(m, xyz) -> tuple[float,float,float]:
    m = require_transform(m)
    p = vector(xyz,3)
    return tuple(finite(sum(m[4*r+k]*p[k] for k in range(3))+m[4*r+3]) for r in range(3))

def invert_transform(m) -> tuple[float,...]:
    a = require_transform(m)
    rows = [[float(a[4*r+c]) for c in range(4)]+[float(r==c) for c in range(4)] for r in range(4)]
    for j in range(4):
        pivot = max(range(j,4), key=lambda i: abs(rows[i][j]))
        if rows[pivot][j] == 0:
            raise ValidationError("singular transform")
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scalar = rows[j][j]
        rows[j] = [x/scalar for x in rows[j]]
        for r in range(4):
            if r != j:
                scalar = rows[r][j]
                rows[r] = [x-scalar*y for x,y in zip(rows[r],rows[j])]
    return require_transform([rows[r][c+4] for r in range(4) for c in range(4)])

def is_rigid(m, tolerance=1e-7) -> bool:
    a = require_transform(m)
    columns = [[a[4*r+c] for r in range(3)] for c in range(3)]
    return (det3(a) > 0 and all(abs(sum(x*y for x,y in zip(columns[i],columns[j])) - float(i==j)) < tolerance for i in range(3) for j in range(3)))
