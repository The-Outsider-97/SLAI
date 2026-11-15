"""Planar box-face lineage and conforming parametric face modifiers.

The source box and modifier stack remain authoritative. Derived face records
are scoped to evaluated SGEO objects and are never baked into source geometry.
Unsupported surfaces fail explicitly; triangulation density is not treated as
a substitute for selectable topological regions.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations, product
from math import sqrt
from uuid import UUID, uuid5

from ..common.errors import GeometryValidationError, ReferenceError, ResourceLimitError, UnsupportedFeatureError
from .geometry import validate_geometry

FACE_KEY = "slai.sgeo.face_lineage.v1"

# normal axis/sign, U axis/sign, V axis/sign; U cross V points outward.
_SIDES = ((0, 1, 1, 1, 2, 1), (0, -1, 2, 1, 1, 1),
          (1, 1, 2, 1, 0, 1), (1, -1, 0, 1, 2, 1),
          (2, 1, 0, 1, 1, 1), (2, -1, 1, 1, 0, 1))


def _face_id(item_id, axis, sign):
    return str(uuid5(UUID(item_id), f"sgeo.face.v1:{axis}:{sign}"))


def _box_faces(item_id):
    return {_face_id(item_id, axis, sign):
            {"axis": axis, "sign": sign, "u_axis": u, "u_sign": su,
             "v_axis": v, "v_sign": sv, "bounds": [0.0, 1.0, 0.0, 1.0],
             "parent": None, "offset": 0.0}
            for axis, sign, u, su, v, sv in _SIDES}


def _affine_box(geometry):
    """Recognize a full-volume parallelepiped; never guess arbitrary facets.

    Returns the origin and three independent positive-handed edge vectors of
    its unit-cube parameterization. This admits inclined-wall prisms that are
    affine images of a box, while rejecting genuinely non-box host topology.
    """
    if geometry is None or geometry.get("kind") != "mesh":
        return None
    vertices = [tuple(map(float, p)) for p in geometry["vertices"]]
    if len(vertices) != 8 or len(set(vertices)) != 8:
        return None
    from .analysis import analyze_mesh
    analysis = analyze_mesh(geometry)
    if not analysis.is_closed_oriented_surface or analysis.signed_volume is None or analysis.signed_volume <= 0:
        return None
    origin = min(vertices)
    others = [p for p in vertices if p != origin]
    magnitude = max(abs(c) for point in vertices for c in point) + 1
    eps = 1e-8 * magnitude

    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

    ranked = []
    for indices in combinations(range(7), 3):
        edges = [tuple(others[i][j] - origin[j] for j in range(3)) for i in indices]
        volume = dot(edges[0], cross(edges[1], edges[2]))
        if abs(volume) < eps**3:
            continue
        corners = [tuple(origin[j] + sum(bit * edges[k][j] for k, bit in enumerate(bits))
                         for j in range(3)) for bits in product((0, 1), repeat=3)]
        if not all(any(sum((a-b)**2 for a,b in zip(point, candidate)) < eps**2
                           for candidate in vertices) for point in corners):
            continue
        if abs(analysis.signed_volume - abs(volume)) > max(1e-8 * abs(volume), eps**3):
            continue
        if volume < 0:
            edges[1], edges[2] = edges[2], edges[1]
        ranked.append((sum(dot(e,e) for e in edges), indices, edges))
    if not ranked:
        return None
    return {"origin": list(origin), "edges": [list(edge) for edge in min(ranked)[2]]}


def _world_point(data, local):
    affine = data.get("affine")
    if affine is None:
        return list(local)
    return [affine["origin"][j] + sum(local[i] * affine["edges"][i][j] for i in range(3))
            for j in range(3)]


def _world_vector(data, local):
    affine = data.get("affine")
    if affine is None:
        return list(local)
    return [sum(local[i] * affine["edges"][i][j] for i in range(3)) for j in range(3)]


def _normal_and_thickness(data, face):
    from math import isfinite
    u = [0.0]*3
    v = [0.0]*3
    u[face["u_axis"]] = float(face["u_sign"])
    v[face["v_axis"]] = float(face["v_sign"])
    u, v = _world_vector(data, u), _world_vector(data, v)
    cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    length = sqrt(sum(c*c for c in cross))
    if not isfinite(length) or length < 1e-12:
        raise GeometryValidationError("face has a degenerate affine normal")
    normal = [c/length for c in cross]
    axis = [0.0]*3
    axis[face["axis"]] = 1.0
    thickness = abs(sum(a*b for a,b in zip(_world_vector(data, axis), normal))) * data["base_box"]["size"][face["axis"]]
    return normal, thickness


def _face_world_frame(data, face):
    base_box = data["base_box"]
    center, (width, height) = _face_region(base_box, face)
    normal, thickness = _normal_and_thickness(data, face)
    su = [0.0]*3
    sv = [0.0]*3
    su[face["u_axis"]] = face["u_sign"] * width
    sv[face["v_axis"]] = face["v_sign"] * height
    u = _world_vector(data, su)
    v = _world_vector(data, sv)
    # The offset in a derived face is a physical distance along its actual
    # normal, not along a sheared box coordinate axis.
    center[face["axis"]] -= face["sign"] * face.get("offset", 0)
    world = _world_point(data, center)
    return [p + face.get("offset", 0) * n for p, n in zip(world, normal)], u, v, normal, thickness


def face_catalog(record):
    """Stable face UUIDs mapped to local surface frames and parameter ranges.

    The returned map is detached and can be used by viewport face picking.
    Only analytic boxes and derived geometry carrying validated lineage are
    currently resolvable. Other meshes must not invent facet identity.
    """
    geometry = record.get("geometry")
    data = record.get("metadata", {}).get(FACE_KEY)
    if isinstance(data, dict) and isinstance(data.get("faces"), dict):
        return deepcopy(data["faces"])
    if geometry and geometry.get("kind") == "box":
        return _box_faces(record["id"])
    if _affine_box(geometry) is not None:
        return _box_faces(record["id"])
    return {}


def pick_face(record, position, normal, *, tolerance=1e-7):
    """Map a local raycast hit to the narrowest registered face or subface.

    The viewport supplies the actual intersection and normal from the current
    display mesh. This function only maps that validated surface hit to stable
    parametric face identity; it does not claim arbitrary mesh facet lineage.
    """
    from ..common.helpers import finite, vector
    point = vector(position, 3, "local face hit")
    direction = vector(normal, 3, "local face normal")
    limit = finite(tolerance, "face hit tolerance")
    if limit <= 0:
        raise GeometryValidationError("face hit tolerance must be positive")
    norm = sqrt(sum(value * value for value in direction))
    if norm <= 1e-12:
        raise GeometryValidationError("face hit normal must be nonzero")
    data = _lineage(record)
    matches = []
    for ident, face in data["faces"].items():
        center, u, v, face_normal, _ = _face_world_frame(data, face)
        if sum(a*b for a,b in zip(direction,face_normal)) / norm < 1 - 1e-5:
            continue
        delta = [p-c for p,c in zip(point,center)]
        if abs(sum(delta[j]*face_normal[j] for j in range(3))) > limit:
            continue
        uu, vv, uv = sum(x*x for x in u), sum(x*x for x in v), sum(a*b for a,b in zip(u,v))
        det = uu*vv-uv*uv
        if det <= 1e-24:
            continue
        du, dv = sum(a*b for a,b in zip(delta,u)), sum(a*b for a,b in zip(delta,v))
        s, t = (du*vv-dv*uv)/det, (dv*uu-du*uv)/det
        if abs(s) > .5 + limit or abs(t) > .5 + limit:
            continue
        matches.append((sqrt(det), ident))
    if not matches:
        return None
    # Stable tie-break for shared boundaries. The user can select parent faces
    # through the face browser when a subface covers the same position.
    return min(matches)[1]


def _lineage(record):
    data = record.get("metadata", {}).get(FACE_KEY)
    if isinstance(data, dict) and isinstance(data.get("base_box"), dict):
        return deepcopy(data)
    geometry = record.get("geometry")
    if geometry and geometry.get("kind") == "box":
        return {"base_box": deepcopy(geometry), "faces": _box_faces(record["id"])}
    affine = _affine_box(geometry)
    if affine is None:
        raise UnsupportedFeatureError("face modifier requires a box or a recognized affine box solid")
    return {"base_box": {"kind": "box", "center": [.5,.5,.5], "size": [1.,1.,1.]},
            "affine": affine, "faces": _box_faces(record["id"])}


def _face_region(base_box, face):
    center, size = base_box["center"], base_box["size"]
    axis = face["axis"]
    result = [float(c) for c in center]
    result[axis] += face["sign"] * (size[axis] / 2 + face.get("offset", 0.0))
    for a, sign, bounds in ((face["u_axis"], face["u_sign"], face["bounds"][:2]),
                            (face["v_axis"], face["v_sign"], face["bounds"][2:])):
        lo, hi = bounds
        result[a] += sign * (lo + hi - 1) * size[a] / 2
    widths = (size[face["u_axis"]] * (face["bounds"][1] - face["bounds"][0]),
              size[face["v_axis"]] * (face["bounds"][3] - face["bounds"][2]))
    return result, widths


def subdivide_face(source, modifier):
    """Create actual closed-shell partitions and persistent cell references."""
    target = modifier["target"]["subelement"]["id"]
    data = _lineage(source)
    if target not in data["faces"]:
        raise ReferenceError("subdivision target face is missing")
    if source["geometry"].get("kind") != "box" and _affine_box(source["geometry"]) is None:
        raise UnsupportedFeatureError("repeated face subdivision of derived meshes requires topology remapping")
    face = data["faces"][target]
    rows, columns = modifier["parameters"]["rows"], modifier["parameters"]["columns"]
    if rows * columns > 4096:
        raise ResourceLimitError("subdivision grid is too large")
    u0, u1, v0, v1 = face["bounds"]
    # Each cell's ID includes the layout: changing count invalidates stale
    # downstream references rather than attaching them to a different region.
    faces = data["faces"]
    for row in range(rows):
        for column in range(columns):
            ident = str(uuid5(UUID(modifier["id"]), f"cell:{rows}:{columns}:{row}:{column}"))
            bounds = [u0 + (u1 - u0) * column / columns,
                      u0 + (u1 - u0) * (column + 1) / columns,
                      v0 + (v1 - v0) * row / rows,
                      v0 + (v1 - v0) * (row + 1) / rows]
            faces[ident] = {**deepcopy(face), "bounds": bounds, "parent": target}
    # Split at every interior column/row station, not just the outer bounds.
    mesh = _conforming_grid_mesh(data["base_box"], face, rows, columns)
    if data.get("affine"):
        mesh["vertices"] = [_world_point(data, point) for point in mesh["vertices"]]
        validate_geometry(mesh)
    output = deepcopy(source)
    output["geometry"] = mesh
    output.setdefault("metadata", {})[FACE_KEY] = data
    return output, {"backend": "conforming-box-face-grid", "approximate": False,
                    "subface_ids": sorted(k for k, f in faces.items() if f.get("parent") == target),
                    "rows": rows, "columns": columns, "watertight": True}


def _conforming_grid_mesh(box, face, rows, columns):
    # Split all six sides at shared stations, retaining indexed edge vertices.
    size, center = box["size"], box["center"]
    stations = [[center[i] - size[i] / 2, center[i] + size[i] / 2] for i in range(3)]
    for axis, sign, low, high, count in ((face["u_axis"], face["u_sign"], face["bounds"][0], face["bounds"][1], columns),
                                         (face["v_axis"], face["v_sign"], face["bounds"][2], face["bounds"][3], rows)):
        for step in range(count + 1):
            t = low + (high - low) * step / count
            stations[axis].append(center[axis] + sign * (t - .5) * size[axis])
    return _box_mesh_at_stations(box, stations)


def _box_mesh_at_stations(box, stations):
    size, center = box["size"], box["center"]
    stations = [sorted(set(values)) for values in stations]
    vertices, triangles, indices = [], [], {}

    def vertex(values):
        key = tuple(values)
        if key not in indices:
            indices[key] = len(vertices)
            vertices.append(list(values))
        return indices[key]

    for axis, sign, u, su, v, sv in _SIDES:
        fixed = center[axis] + sign * size[axis] / 2
        for i in range(len(stations[u]) - 1):
            for j in range(len(stations[v]) - 1):
                a, b = stations[u][i:i+2]
                c, d = stations[v][j:j+2]
                corners = []
                for up, vp in ((a, c), (b, c), (b, d), (a, d)):
                    point = [0.0] * 3
                    point[axis], point[u], point[v] = fixed, up, vp
                    corners.append(vertex(point))
                parity = 1 if (u, v, axis) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1
                if parity == sign:
                    triangles.extend(([corners[0], corners[1], corners[2]],
                                      [corners[0], corners[2], corners[3]]))
                else:
                    triangles.extend(([corners[0], corners[2], corners[1]],
                                      [corners[0], corners[3], corners[2]]))
    return validate_geometry({"kind": "mesh", "vertices": vertices, "faces": triangles})


def _boolean_tool(source, tool, *, mode):
    from .modifier_ops import _from_trimesh, _manifold, _to_trimesh
    _, trimesh = _manifold()
    original, _ = _to_trimesh(source)
    if not original.is_volume:
        raise GeometryValidationError("face modifier requires a closed positive-volume host")
    try:
        if mode == "add":
            result = trimesh.boolean.union([original, tool], engine="manifold")
        else:
            result = trimesh.boolean.difference([original, tool], engine="manifold")
    except (ValueError, RuntimeError) as exc:
        raise GeometryValidationError("face modifier Boolean could not construct valid geometry") from exc
    return _from_trimesh(result)


def extrude_face(source, modifier):
    """Add or recess a planar face region along its true world-space normal."""
    data = _lineage(source)
    ident = modifier["target"]["subelement"]["id"]
    if ident not in data["faces"]:
        raise ReferenceError("extrusion face no longer exists")
    face = data["faces"][ident]
    params = modifier["parameters"]
    center, u, v, normal, thickness = _face_world_frame(data, face)
    depth = params["depth"]
    if depth > thickness + 1e-12:
        raise GeometryValidationError("extrusion exceeds actual local host thickness")
    # This simple host has a planar face; directional reverse would depart
    # from the requested material operation and is deliberately explicit.
    direction = 1 if params["direction"] == "normal" else -1
    material_direction = 1 if params["mode"] == "add" else -1
    if direction != material_direction:
        raise GeometryValidationError("chosen face direction is incompatible with extrusion material mode")
    from .modifier_ops import _dependencies
    _, trimesh = _dependencies()
    points = []
    for elevation in (0.0, material_direction * depth):
        for su, sv in ((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)):
            points.append([center[j] + su*u[j] + sv*v[j] + elevation*normal[j]
                           for j in range(3)])
    tool = trimesh.convex.convex_hull(points)
    geometry = _boolean_tool(source, tool, mode=params["mode"])
    if geometry is None:
        raise GeometryValidationError("extrusion removed the entire source solid")
    output = deepcopy(source)
    output["geometry"] = geometry
    face["offset"] = face.get("offset", 0.0) + (depth if params["mode"] == "add" else -depth)
    output.setdefault("metadata", {})[FACE_KEY] = data
    return output, {"backend": "manifold-face-extrusion", "approximate": True,
                    "face_id": ident, "depth": depth, "watertight": True}


def bevel_face(source, modifier):
    """Cut a straight, inline chamfer along one edge of a planar face region."""
    data = _lineage(source)
    ident = modifier["target"]["subelement"]["id"]
    if ident not in data["faces"]:
        raise ReferenceError("bevel face no longer exists")
    face = data["faces"][ident]
    params = modifier["parameters"]
    center, u, v, normal, thickness = _face_world_frame(data, face)
    width = sqrt(sum(c*c for c in u))
    height = sqrt(sum(c*c for c in v))
    depth, amount = params["depth"], params["width"]
    if depth > thickness or amount >= min(width, height):
        raise GeometryValidationError("bevel exceeds the selected region or host thickness")
    edge = params["edge"]
    lateral, along = (u,v) if edge.startswith("u") else (v,u)
    lateral_size = sqrt(sum(c*c for c in lateral))
    inward = [c/lateral_size for c in lateral]
    inner = 1 if edge.endswith("min") else -1
    boundary = [c-inner*lateral[j]/2 for j,c in enumerate(center)]
    # A triangular cutter profile reaches the top surface at 'width' and
    # removes a depth 'depth' at the selected boundary. Extend very slightly
    # beyond the face to make the Boolean unambiguous at coplanar surfaces.
    pad = max(width, height, thickness) * 1e-7
    poly = [( -pad, +pad), (amount + pad, +pad), (-pad, -(depth + pad))]
    # Build a watertight triangular prism along the selected face edge.
    points = []
    for along_fraction in (-.5, .5):
        for lat, axial in poly:
            points.append([boundary[j] + along_fraction*along[j] + inner*lat*inward[j]
                           + axial*normal[j] for j in range(3)])
    from .modifier_ops import _dependencies
    _, trimesh = _dependencies()
    cutter = trimesh.convex.convex_hull(points)
    geometry = _boolean_tool(source, cutter, mode="recess")
    if geometry is None:
        raise GeometryValidationError("bevel removed the entire source solid")
    output = deepcopy(source)
    output["geometry"] = geometry
    output.setdefault("metadata", {})[FACE_KEY] = data
    return output, {"backend": "manifold-inline-face-chamfer", "approximate": True,
                    "face_id": ident, "edge": edge, "watertight": True}
