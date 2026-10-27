"""Atomic drawing batches with optimistic revisions and detached candidates."""
from copy import deepcopy
from ..common.errors import ConflictError, ValidationError, ResourceLimitError, ReferenceError
from ..common.ids import require_id
from .drawing import Drawing, MAX_ENTITIES, MAX_LAYERS, LAYER_KEYS
from .entities import make_entity, validate_entity
from .linetypes import validate_linetype, MAX_LINETYPES
from .blocks import validate_block_record, MAX_BLOCKS
from .operations import transform_entity

_ENTITY_ADD_KEYS = {"kind", "geometry", "layer", "name", "metadata", "style", "entity_id", "create_layer"}
_UPDATE_KEYS = {"geometry", "layer", "name", "metadata", "style"}
_LAYER_UPDATE_KEYS = {"visible", "stroke", "stroke_width", "linetype"}
_BLOCK_KEYS = {"name", "base_point", "entities", "description", "replace"}
_ENTITY_SPEC_KEYS = {"kind", "geometry", "layer", "name", "metadata", "style", "entity_id"}


def apply_drawing_operations(drawing, operations, *, expected_revision=None):
    """Apply entity, layer, linetype and block edits as one revision.

    Entity operations use ``id`` and ``value``; ``add`` uses make_entity keyword
    arguments (plus ``create_layer``). Layer edits use ``name``: ``add_layer``
    (``value`` = visibility, optional ``style``), ``rename_layer`` (``value`` = new
    name), ``layer_visibility`` and ``update_layer`` (``value`` = dict of
    ``visible``/``stroke``/``stroke_width``/``linetype``; ``None`` clears a style
    key). Registry edits: ``define_linetype`` (``name``, ``pattern``,
    ``description``, ``replace``), ``remove_linetype``, ``define_block`` (``name``,
    ``base_point``, ``entities`` as entity specs, ``description``, ``replace``),
    ``remove_block`` and ``rename_block`` (``value`` = new name). A referenced
    layer, linetype or block cannot be removed. Failed batches leave the drawing
    unchanged: every step is checked locally against a detached candidate and the
    candidate is audited once more in full before it replaces the live state.
    Direct mutation of public collections bypasses locking.
    """
    if not isinstance(drawing, Drawing) or not isinstance(operations, list) or not 1 <= len(operations) <= 1000:
        raise ValidationError("operations must be a bounded nonempty list for a Drawing")
    with drawing._mutation_lock:
        if expected_revision is not None and expected_revision != drawing.revision:
            raise ConflictError("drawing revision does not match expected revision")
        drawing.validate()
        candidate = drawing._detached_copy()
        # id -> entity and name -> record indexes keep every step O(1);
        # insertion-ordered dicts preserve document order for the final list.
        by_id = {e["id"]: e for e in candidate.entities}
        layer_by_name = {x["name"]: x for x in candidate.layers}
        linetype_by_name = {x["name"]: x for x in candidate.linetypes}
        block_by_name = {x["name"]: x for x in candidate.blocks}
        all_ids = set(by_id) | {e["id"] for b in candidate.blocks for e in b["entities"]}
        layer_use = {}
        for e in candidate.entities:
            layer_use[e.get("layer", "0")] = layer_use.get(e.get("layer", "0"), 0) + 1
        for b in candidate.blocks:
            for e in b["entities"]:
                layer_use[e.get("layer", "0")] = layer_use.get(e.get("layer", "0"), 0) + 1
        results = []

        def use(layer, delta):
            layer_use[layer] = layer_use.get(layer, 0) + delta

        def check(entity):
            validate_entity(entity, set(layer_by_name), linetypes=set(linetype_by_name), blocks=set(block_by_name))

        def check_light(entity):
            """Per-op reference checks that stay O(1) for plain adds."""
            layer = entity.get("layer", "0")
            if layer not in layer_by_name:
                raise ReferenceError("entity references nonexistent layer: " + layer, context={"layer": layer})
            style = entity.get("style")
            if style and "linetype" in style and style["linetype"].lower() not in ("continuous", "bylayer") \
                    and style["linetype"] not in linetype_by_name:
                raise ReferenceError("reference to unknown linetype: " + style["linetype"])
            if entity["kind"] == "insert" and entity["geometry"]["block"] not in block_by_name:
                raise ReferenceError("insert references unknown block: " + entity["geometry"]["block"])

        for operation in operations:
            if not isinstance(operation, dict) or not isinstance(operation.get("op"), str):
                raise ValidationError("invalid drawing operation")
            op = deepcopy(operation)
            kind = op.pop("op")
            if kind == "add":
                if set(op) - _ENTITY_ADD_KEYS or not {"kind", "geometry"} <= op.keys():
                    raise ValidationError("invalid entity add arguments")
                create_layer = op.pop("create_layer", False)
                if type(create_layer) is not bool:
                    raise ValidationError("create_layer must be Boolean")
                ent = make_entity(**op)
                if ent["id"] in all_ids:
                    raise ValidationError("duplicate identifier")
                if len(by_id) >= MAX_ENTITIES:
                    raise ResourceLimitError("too many drawing entities")
                layer = ent.get("layer", "0")
                if layer not in layer_by_name:
                    if not create_layer:
                        raise ReferenceError("entity references nonexistent layer: " + layer)
                    if len(layer_by_name) >= MAX_LAYERS:
                        raise ResourceLimitError("too many drawing layers")
                    record = {"name": layer, "visible": True}
                    candidate.layers.append(record)
                    layer_by_name[layer] = record
                check_light(ent)
                by_id[ent["id"]] = ent
                all_ids.add(ent["id"])
                use(layer, 1)
                results.append(ent["id"])
            elif kind in {"add_layer", "remove_layer", "rename_layer", "layer_visibility", "update_layer"}:
                if set(op) - {"name", "value", "style"} or not isinstance(op.get("name"), str):
                    raise ValidationError("invalid layer operation")
                name = op["name"]
                if kind == "add_layer":
                    style = op.get("style", {})
                    if not isinstance(style, dict):
                        raise ValidationError("layer style must be an object")
                    record = {"name": name, "visible": op.get("value", True), **style}
                    Drawing._check_layer_record(record, set(linetype_by_name))
                    if name in layer_by_name:
                        raise ValidationError("drawing layer already exists")
                    if len(layer_by_name) >= MAX_LAYERS:
                        raise ResourceLimitError("too many drawing layers")
                    candidate.layers.append(record)
                    layer_by_name[name] = record
                else:
                    layer = layer_by_name.get(name)
                    if layer is None:
                        raise ReferenceError("unknown drawing layer")
                    if kind == "remove_layer":
                        if layer_use.get(name, 0):
                            raise ValidationError("cannot remove a referenced layer")
                        if len(layer_by_name) == 1:
                            raise ValidationError("drawing requires at least one layer")
                        candidate.layers.remove(layer)
                        del layer_by_name[name]
                    elif kind == "rename_layer":
                        new = op.get("value")
                        Drawing._check_layer_record({**layer, "name": new}, set(linetype_by_name))
                        if new in layer_by_name and new != name:
                            raise ValidationError("drawing layer already exists")
                        layer["name"] = new
                        del layer_by_name[name]
                        layer_by_name[new] = layer
                        for entity in by_id.values():
                            if entity.get("layer", "0") == name:
                                entity["layer"] = new
                        for block in candidate.blocks:
                            for entity in block["entities"]:
                                if entity.get("layer", "0") == name:
                                    entity["layer"] = new
                        if new != name:
                            layer_use[new] = layer_use.get(new, 0) + layer_use.pop(name, 0)
                    elif kind == "layer_visibility":
                        if type(op.get("value")) is not bool:
                            raise ValidationError("layer visibility must be Boolean")
                        layer["visible"] = op["value"]
                    else:
                        value = op.get("value")
                        if not isinstance(value, dict) or set(value) - _LAYER_UPDATE_KEYS:
                            raise ValidationError("invalid layer update fields")
                        merged = dict(layer)
                        for key, item in value.items():
                            if item is None and key != "visible":
                                merged.pop(key, None)
                            else:
                                merged[key] = item
                        Drawing._check_layer_record(merged, set(linetype_by_name))
                        layer.clear()
                        layer.update(merged)
                results.append(name)
            elif kind in {"define_linetype", "remove_linetype"}:
                if kind == "remove_linetype":
                    if set(op) - {"name"} or op.get("name") not in linetype_by_name:
                        raise ReferenceError("unknown linetype")
                    candidate.linetypes.remove(linetype_by_name.pop(op["name"]))
                else:
                    replace = op.pop("replace", False)
                    if type(replace) is not bool or set(op) - {"name", "pattern", "description"}:
                        raise ValidationError("invalid linetype definition")
                    record = validate_linetype(op)
                    existing = linetype_by_name.get(record["name"])
                    if existing is not None and not replace:
                        raise ValidationError("linetype already exists")
                    if existing is None and len(linetype_by_name) >= MAX_LINETYPES:
                        raise ResourceLimitError("too many linetypes")
                    if existing is not None:
                        candidate.linetypes[candidate.linetypes.index(existing)] = record
                    else:
                        candidate.linetypes.append(record)
                    linetype_by_name[record["name"]] = record
                results.append(op.get("name"))
            elif kind in {"define_block", "remove_block", "rename_block"}:
                if kind == "remove_block":
                    if set(op) - {"name"} or op.get("name") not in block_by_name:
                        raise ReferenceError("unknown block")
                    old = block_by_name.pop(op["name"])
                    for e in old["entities"]:
                        use(e.get("layer", "0"), -1)
                        all_ids.discard(e["id"])
                    candidate.blocks.remove(old)
                elif kind == "rename_block":
                    new = op.get("value")
                    if set(op) - {"name", "value"} or op.get("name") not in block_by_name:
                        raise ReferenceError("unknown block")
                    block = block_by_name[op["name"]]
                    validate_block_record({**block, "name": new})
                    if new in block_by_name and new != op["name"]:
                        raise ValidationError("block already exists")
                    block_by_name[new] = block_by_name.pop(op["name"])
                    block["name"] = new
                    for entity in list(by_id.values()) + [e for b in candidate.blocks for e in b["entities"]]:
                        if entity["kind"] == "insert" and entity["geometry"]["block"] == op["name"]:
                            entity["geometry"]["block"] = new
                    op["name"] = new
                else:
                    replace = op.pop("replace", False)
                    if type(replace) is not bool or set(op) - _BLOCK_KEYS or not {"name", "entities"} <= op.keys():
                        raise ValidationError("invalid block definition")
                    specs = op.get("entities")
                    if not isinstance(specs, list) or any(not isinstance(s, dict) or set(s) - _ENTITY_SPEC_KEYS for s in specs):
                        raise ValidationError("block entities must be entity specs")
                    base = op.get("base_point", [0.0, 0.0])
                    built = [make_entity(**spec) for spec in specs]
                    record = {"name": op["name"], "base_point": [float(v) for v in base], "entities": built}
                    if "description" in op:
                        record["description"] = op["description"]
                    validate_block_record(record)
                    existing = block_by_name.get(record["name"])
                    if existing is not None and not replace:
                        raise ValidationError("block already exists")
                    if existing is None and len(block_by_name) >= MAX_BLOCKS:
                        raise ResourceLimitError("too many blocks")
                    if existing is not None:
                        for e in existing["entities"]:
                            use(e.get("layer", "0"), -1)
                            all_ids.discard(e["id"])
                    for e in built:
                        if e["id"] in all_ids:
                            raise ValidationError("duplicate identifier")
                        all_ids.add(e["id"])
                        use(e.get("layer", "0"), 1)
                        layer = e.get("layer", "0")
                        if layer not in layer_by_name:
                            raise ReferenceError("entity references nonexistent layer: " + layer)
                        style = e.get("style")
                        if style and "linetype" in style and style["linetype"].lower() not in ("continuous", "bylayer") \
                                and style["linetype"] not in linetype_by_name:
                            raise ReferenceError("reference to unknown linetype: " + style["linetype"])
                    if existing is not None:
                        candidate.blocks[candidate.blocks.index(existing)] = record
                    else:
                        candidate.blocks.append(record)
                    block_by_name[record["name"]] = record
                results.append(op.get("name"))
            elif kind in {"remove", "update", "transform"}:
                if set(op) - {"id", "value"}:
                    raise ValidationError("invalid entity operation")
                oid = require_id(op.get("id"))
                entity = by_id.get(oid)
                if entity is None:
                    raise ReferenceError("unknown drawing entity")
                if kind == "remove":
                    del by_id[oid]
                    all_ids.discard(oid)
                    use(entity.get("layer", "0"), -1)
                elif kind == "update":
                    value = op.get("value")
                    if not isinstance(value, dict) or set(value) - _UPDATE_KEYS:
                        raise ValidationError("invalid entity update fields")
                    updated = {**entity, **value}
                    check(updated)
                    use(entity.get("layer", "0"), -1)
                    use(updated.get("layer", "0"), 1)
                    by_id[oid] = updated
                else:
                    moved = transform_entity(entity, op.get("value"), preserve_id=True)
                    check(moved)
                    by_id[oid] = moved
                results.append(oid)
            else:
                raise ValidationError("unknown drawing operation")
        candidate.entities = list(by_id.values())  # dict order == document order
        candidate.revision = drawing.revision + 1
        candidate.validate()
        drawing.layers = candidate.layers
        drawing.entities = candidate.entities
        drawing.metadata = candidate.metadata
        drawing.linetypes = candidate.linetypes
        drawing.blocks = candidate.blocks
        drawing.revision = candidate.revision
        return results
