"""Atomic drawing batches with optimistic revisions and detached candidates."""
from copy import deepcopy
from ..common.errors import ConflictError, ValidationError
from ..common.ids import require_id
from .drawing import Drawing
from .operations import transform_entity


def apply_drawing_operations(drawing, operations, *, expected_revision=None):
    """Apply add/remove/update/transform and layer edits as one revision.

    Entity operations use ``id`` and ``value``; add uses make_entity keyword
    arguments. Layer edits use ``name``; rename_layer additionally takes
    ``value``. A referenced layer cannot be removed. Failed batches leave the
    drawing unchanged. Direct mutation of public collections bypasses locking.
    """
    if not isinstance(drawing, Drawing) or not isinstance(operations, list) or not 1 <= len(operations) <= 1000:
        raise ValidationError("operations must be a bounded nonempty list for a Drawing")
    with drawing._mutation_lock:
        if expected_revision is not None and expected_revision != drawing.revision:
            raise ConflictError("drawing revision does not match expected revision")
        candidate = Drawing(**drawing.as_dict(), unit=drawing.unit)
        results = []
        for operation in operations:
            if not isinstance(operation, dict) or not isinstance(operation.get("op"), str):
                raise ValidationError("invalid drawing operation")
            op = deepcopy(operation)
            kind = op.pop("op")
            if kind == "add":
                if set(op)-{"kind","geometry","layer","name","metadata","style","entity_id"} or not {"kind","geometry"} <= op.keys():
                    raise ValidationError("invalid entity add arguments")
                results.append(candidate.add(**op))
            elif kind in {"add_layer", "remove_layer", "rename_layer", "layer_visibility"}:
                if set(op)-{"name","value"} or not isinstance(op.get("name"), str):
                    raise ValidationError("invalid layer operation")
                name = op["name"]
                if kind == "add_layer":
                    candidate.add_layer(name, visible=op.get("value", True))
                else:
                    layer = next((x for x in candidate.layers if x["name"] == name), None)
                    if layer is None:
                        raise ValidationError("unknown drawing layer")
                    if kind == "remove_layer":
                        if any(e.get("layer","0") == name for e in candidate.entities):
                            raise ValidationError("cannot remove a referenced layer")
                        candidate.layers.remove(layer)
                    elif kind == "rename_layer":
                        layer["name"] = op.get("value")
                        for entity in candidate.entities:
                            if entity.get("layer","0") == name:
                                entity["layer"] = op.get("value")
                    else:
                        layer["visible"] = op.get("value")
                results.append(name)
            elif kind in {"remove", "update", "transform"}:
                if set(op)-{"id","value"}:
                    raise ValidationError("invalid entity operation")
                oid = require_id(op.get("id"))
                entity = next((e for e in candidate.entities if e["id"] == oid), None)
                if entity is None:
                    raise ValidationError("unknown drawing entity")
                if kind == "remove":
                    candidate.entities.remove(entity)
                elif kind == "update":
                    value = op.get("value")
                    if not isinstance(value, dict) or set(value)-{"geometry","layer","name","metadata","style"}:
                        raise ValidationError("invalid entity update fields")
                    entity.update(value)
                else:
                    candidate.entities[candidate.entities.index(entity)] = transform_entity(entity, op.get("value"), preserve_id=True)
                results.append(oid)
            else:
                raise ValidationError("unknown drawing operation")
            candidate.validate()
        drawing.layers = candidate.layers
        drawing.entities = candidate.entities
        drawing.metadata = candidate.metadata
        drawing.revision += 1
        return results
