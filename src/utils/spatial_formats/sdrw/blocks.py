"""SDRW blocks (reusable symbol definitions) and ``insert`` references to them.

A block is ``{name, base_point, entities[, description]}``. An ``insert`` entity
places one instance at ``position`` with a ``rotation`` and a uniform positive
``scale``; the block's ``base_point`` maps onto ``position``. Block content may
itself contain inserts, but the reference graph must be acyclic, at most
``MAX_BLOCK_DEPTH`` levels deep, and must expand to at most
``MAX_EXPANDED_ENTITIES`` primitives, so an instance can always be exploded.

Content on layer ``"0"`` inside a block floats to the layer of the insert when
exploded (the usual CAD convention); content on other layers keeps them.
Mirrored or non-uniformly scaled inserts are not representable (circles and
arcs would no longer be circles and arcs).
"""
from __future__ import annotations

from math import cos, sin

from ..common import ValidationError, finite, vector
from ..common.errors import GeometryValidationError, ReferenceError, ResourceLimitError

MAX_BLOCKS = 10_000
MAX_BLOCK_ENTITIES = 100_000
MAX_BLOCK_DEPTH = 8
MAX_EXPANDED_ENTITIES = 2_000_000
INSERT_KEYS = frozenset({"block", "position", "rotation", "scale"})


def validate_insert(g: dict) -> None:
    name = g.get("block")
    if not isinstance(name, str) or not 1 <= len(name) <= 255:
        raise GeometryValidationError("insert requires a block name")
    vector(g.get("position"), 2)
    if "rotation" in g:
        finite(g["rotation"], "insert rotation", limit=1e8)
    if "scale" in g:
        s = g["scale"]
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not finite(s, "insert scale") > 0:
            raise GeometryValidationError("insert scale must be a single positive number")


def validate_block_record(block) -> dict:
    """Structural checks of one block record (its entities are validated by the drawing)."""
    if not isinstance(block, dict) or set(block) - {"name", "base_point", "entities", "description"}:
        raise ValidationError("invalid block record")
    name = block.get("name")
    if not isinstance(name, str) or not 1 <= len(name) <= 255 or name.startswith("*"):
        raise ValidationError("invalid block name (1..255 characters, not starting with '*')")
    vector(block.get("base_point"), 2, "block base point")
    if "description" in block and (not isinstance(block["description"], str) or len(block["description"]) > 255):
        raise ValidationError("invalid block description")
    entities = block.get("entities")
    if not isinstance(entities, list) or not 1 <= len(entities) <= MAX_BLOCK_ENTITIES:
        raise ValidationError("a block requires 1..%d entities" % MAX_BLOCK_ENTITIES)
    return block


def insert_targets(block) -> set:
    return {e["geometry"]["block"] for e in block["entities"] if e.get("kind") == "insert"}


def check_block_graph(blocks) -> dict:
    """Reject cycles, excess depth and runaway expansion; return ``{name: expanded count}``."""
    by_name = {b["name"]: b for b in blocks}
    edges = {name: insert_targets(b) for name, b in by_name.items()}
    for name, targets in edges.items():
        missing = sorted(t for t in targets if t not in by_name)
        if missing:
            raise ReferenceError("block %r inserts unknown block %r" % (name, missing[0]), context={"block": missing[0]})
    state, depth, expanded = {}, {}, {}

    def visit(root):
        stack = [(root, iter(sorted(edges[root])))]
        state[root] = 1
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                state[node] = 2
                depth[node] = 1 + max((depth[t] for t in edges[node]), default=0)
                if depth[node] > MAX_BLOCK_DEPTH:
                    raise ResourceLimitError("blocks are nested deeper than %d levels" % MAX_BLOCK_DEPTH)
                total = 0
                for e in by_name[node]["entities"]:
                    total += expanded[e["geometry"]["block"]] if e.get("kind") == "insert" else 1
                    if total > MAX_EXPANDED_ENTITIES:
                        raise ResourceLimitError("block %r expands to more than %d entities" % (node, MAX_EXPANDED_ENTITIES))
                expanded[node] = total
                continue
            if state.get(nxt) == 1:
                raise ValidationError("block definitions form a cycle through %r" % nxt)
            if state.get(nxt) is None:
                state[nxt] = 1
                stack.append((nxt, iter(sorted(edges[nxt]))))

    for name in by_name:
        if state.get(name) is None:
            visit(name)
    return expanded


def insert_matrix(geometry: dict, base_point) -> list:
    """Row-major 2D matrix taking block coordinates to drawing coordinates."""
    s = float(geometry.get("scale", 1.0))
    th = float(geometry.get("rotation", 0.0))
    c, sn = s * cos(th), s * sin(th)
    bx, by = base_point
    px, py = geometry["position"]
    return [c, -sn, px - (c * bx - sn * by), sn, c, py - (sn * bx + c * by), 0.0, 0.0, 1.0]
