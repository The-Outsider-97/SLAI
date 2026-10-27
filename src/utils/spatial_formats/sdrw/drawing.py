"""SDRW drawing document model with explicit layers, linetypes, blocks, stable IDs and revisions."""
from __future__ import annotations
from copy import deepcopy
from threading import RLock
from ..common.errors import ReferenceError
from ..common import ValidationError, ResourceLimitError, new_id, require_id, assert_unique, require_unit, require_metadata
from ..container.versions import CURRENT_VERSION, schema_id
from .entities import validate_entity, validate_style, make_entity, ANNOTATION_KINDS
from .linetypes import validate_linetype, check_linetype_ref, MAX_LINETYPES
from .blocks import validate_block_record, check_block_graph, MAX_BLOCKS, MAX_BLOCK_ENTITIES

MAX_LAYERS = 10_000
MAX_ENTITIES = 100_000
LAYER_KEYS = frozenset({"name", "visible", "stroke", "stroke_width", "linetype"})
_LAYER_STYLE_KEYS = ("stroke", "stroke_width", "linetype")


class Drawing:
    def __init__(self, name="Untitled", *, unit="mm", document_id=None, revision=1,
                 layers=None, entities=None, metadata=None, linetypes=None, blocks=None):
        self._mutation_lock = RLock()
        self.name = name
        self.unit = require_unit(unit)
        self.document_id = document_id or new_id()
        self.revision = revision
        self.layers = deepcopy(layers) if layers is not None else [{"name":"0", "visible":True}]
        self.entities = deepcopy(entities) if entities is not None else []
        self.metadata = deepcopy(metadata) if metadata is not None else {}
        self.linetypes = deepcopy(linetypes) if linetypes is not None else []
        self.blocks = deepcopy(blocks) if blocks is not None else []
        self.validate()

    def _detached_copy(self):
        """Deep copy for transactional candidates; the caller validates before commit."""
        clone=object.__new__(type(self))
        clone._mutation_lock=RLock()
        clone.name=self.name
        clone.unit=self.unit
        clone.document_id=self.document_id
        clone.revision=self.revision
        clone.layers=deepcopy(self.layers)
        clone.entities=deepcopy(self.entities)
        clone.metadata=deepcopy(self.metadata)
        clone.linetypes=deepcopy(self.linetypes)
        clone.blocks=deepcopy(self.blocks)
        return clone

    # -- reference sets -----------------------------------------------------------------
    def reference_sets(self):
        """``(layer names, linetype names, block names)`` currently defined."""
        return ({x["name"] for x in self.layers if isinstance(x, dict)},
                {x["name"] for x in self.linetypes if isinstance(x, dict)},
                {x["name"] for x in self.blocks if isinstance(x, dict)})

    def validate(self):
        require_id(self.document_id)
        if type(self.revision) is not int or self.revision < 1:
            raise ValidationError("revision must be a positive integer")
        if not isinstance(self.name, str) or len(self.name) > 255:
            raise ValidationError("invalid drawing name")
        require_metadata(self.metadata)
        if not isinstance(self.layers, list) or not 1 <= len(self.layers) <= MAX_LAYERS:
            raise ValidationError("drawing requires at least one layer")
        if not isinstance(self.linetypes, list) or len(self.linetypes) > MAX_LINETYPES:
            raise ValidationError("invalid linetype registry")
        if not isinstance(self.blocks, list) or len(self.blocks) > MAX_BLOCKS:
            raise ValidationError("invalid block registry")
        names = []
        for layer in self.layers:
            self._check_layer_record(layer)
            names.append(layer["name"])
        if len(names) != len(set(names)):
            raise ValidationError("duplicate drawing layer")
        for record in self.linetypes:
            validate_linetype(record)
        linetype_names = [r["name"] for r in self.linetypes]
        if len(linetype_names) != len(set(linetype_names)):
            raise ValidationError("duplicate linetype name")
        lt_known = set(linetype_names)
        for layer in self.layers:
            if "linetype" in layer:
                check_linetype_ref(layer["linetype"], lt_known)
        block_names = []
        block_entity_count = 0
        for block in self.blocks:
            validate_block_record(block)
            block_names.append(block["name"])
        if len(block_names) != len(set(block_names)):
            raise ValidationError("duplicate block name")
        layer_set, block_set = set(names), set(block_names)
        ids = []
        for block in self.blocks:
            block_entity_count += len(block["entities"])
            if block_entity_count > MAX_BLOCK_ENTITIES:
                raise ResourceLimitError("too many entities inside blocks")
            for ent in block["entities"]:
                validate_entity(ent, layer_set, linetypes=lt_known, blocks=block_set)
                ids.append(ent.get("id") if isinstance(ent, dict) else None)
        check_block_graph(self.blocks)
        if not isinstance(self.entities, list) or len(self.entities) > MAX_ENTITIES:
            raise ValidationError("too many drawing entities")
        for ent in self.entities:
            ids.append(ent.get("id") if isinstance(ent, dict) else None)
        assert_unique(ids)
        for ent in self.entities:
            validate_entity(ent, layer_set, linetypes=lt_known, blocks=block_set)
        return self

    # -- incremental mutation -------------------------------------------------
    # validate() is a full O(n) audit and is intentionally kept for loading,
    # serialising and for callers that edited the public collections directly.
    # Single-element mutations only need to check what they touch, which keeps
    # building a drawing entity by entity linear instead of quadratic.
    @staticmethod
    def _check_layer_record(layer, linetypes=None):
        if not isinstance(layer, dict) or set(layer) - LAYER_KEYS:
            raise ValidationError("invalid layer")
        name = layer.get("name")
        if not isinstance(name, str) or not name or len(name) > 255:
            raise ValidationError("invalid layer name")
        if type(layer.get("visible", True)) is not bool:
            raise ValidationError("layer visibility must be Boolean")
        style = {k: layer[k] for k in _LAYER_STYLE_KEYS if k in layer}
        if style:
            validate_style(style, linetypes)

    def add_layer(self, name, *, visible=True, **style):
        """Add a layer; ``stroke``, ``stroke_width`` and ``linetype`` may be given as keywords."""
        with self._mutation_lock:
            record = {"name": name, "visible": visible, **style}
            self._check_layer_record(record, {x["name"] for x in self.linetypes})
            if len(self.layers) >= MAX_LAYERS:
                raise ResourceLimitError("too many drawing layers")
            if any(x["name"] == name for x in self.layers):
                raise ValidationError("drawing layer already exists")
            self.layers.append(record)
            self.revision += 1
            return name

    def add_linetype(self, name, pattern, *, description=None):
        """Define a linetype (see :mod:`.linetypes` for the pattern convention)."""
        with self._mutation_lock:
            record = {"name": name, "pattern": list(pattern)}
            if description is not None:
                record["description"] = description
            validate_linetype(record)
            if len(self.linetypes) >= MAX_LINETYPES:
                raise ResourceLimitError("too many linetypes")
            if any(x["name"] == name for x in self.linetypes):
                raise ValidationError("linetype already exists")
            self.linetypes.append(record)
            self.revision += 1
            return name

    def add_block(self, name, entities, *, base_point=(0.0, 0.0), description=None):
        """Define a block from entity specs ``{"kind", "geometry"[, "layer", "name", "metadata", "style"]}``."""
        with self._mutation_lock:
            built = [make_entity(**spec) for spec in entities]
            record = {"name": name, "base_point": [float(base_point[0]), float(base_point[1])], "entities": built}
            if description is not None:
                record["description"] = description
            validate_block_record(record)
            if len(self.blocks) >= MAX_BLOCKS:
                raise ResourceLimitError("too many blocks")
            if any(b["name"] == name for b in self.blocks):
                raise ValidationError("block already exists")
            candidate = self._detached_copy()
            candidate.blocks.append(record)
            candidate.validate()
            self.blocks.append(record)
            self.revision += 1
            return name

    def _check_refs(self, ent):
        """Reference checks for one new entity (layer is handled by the caller)."""
        style = ent.get("style")
        if style and "linetype" in style:
            check_linetype_ref(style["linetype"], {x["name"] for x in self.linetypes})
        if ent["kind"] == "insert" and not any(b["name"] == ent["geometry"]["block"] for b in self.blocks):
            raise ReferenceError("insert references unknown block: " + ent["geometry"]["block"],
                                 context={"block": ent["geometry"]["block"]})

    def add(self, kind, geometry, *, create_layer=False, **options):
        """Add one entity and return its id.

        ``create_layer=True`` creates the entity's layer first when it does not
        exist (the whole call is atomic: nothing changes if the entity is invalid).
        """
        with self._mutation_lock:
            ent = make_entity(kind, geometry, **options)
            if len(self.entities) >= MAX_ENTITIES:
                raise ResourceLimitError("too many drawing entities")
            layer = ent.get("layer", "0")
            layer_known = any(x["name"] == layer for x in self.layers)
            if not layer_known and not create_layer:
                raise ReferenceError("entity references nonexistent layer: " + layer, context={"layer": layer})
            self._check_refs(ent)
            if (any(e.get("id") == ent["id"] for e in self.entities)
                    or any(e.get("id") == ent["id"] for b in self.blocks for e in b["entities"])):
                raise ValidationError("duplicate identifier")
            if not layer_known:
                if len(self.layers) >= MAX_LAYERS:
                    raise ResourceLimitError("too many drawing layers")
                self.layers.append({"name": layer, "visible": True})
            self.entities.append(ent)
            self.revision += 1
            return ent["id"]

    def remove(self, entity_id):
        with self._mutation_lock:
            require_id(entity_id)
            found = next((i for i,e in enumerate(self.entities) if e["id"] == entity_id), None)
            if found is None:
                raise ValidationError("drawing entity not found")
            self.entities.pop(found)
            self.revision += 1

    # -- serialisation ----------------------------------------------------------------
    def _requires(self):
        features = ["drawing-primitives"]
        entities = list(self.entities) + [e for b in self.blocks for e in b["entities"]]
        if any(e["kind"] in ANNOTATION_KINDS for e in entities):
            features.append("annotations")
        if self.blocks or any(e["kind"] == "insert" for e in entities):
            features.append("blocks")
        if self.linetypes:
            features.append("linetypes")
        return features

    def as_dict(self):
        with self._mutation_lock:
            self.validate()
            return deepcopy({"document_id":self.document_id,"revision":self.revision,
                             "name":self.name,"layers":self.layers,"entities":self.entities,
                             "metadata":self.metadata,"linetypes":self.linetypes,"blocks":self.blocks})

    def manifest(self):
        with self._mutation_lock:
            self.validate()
            return {"format_id":"slai.sdrw", "format_version":CURRENT_VERSION,
                    "schema":schema_id("slai.sdrw", CURRENT_VERSION),
                    "spatial_dimension":2,"document_kind":"drawing","document_id":self.document_id,
                    "revision":self.revision,"units":self.unit,"requires":self._requires()}

    @classmethod
    def from_package(cls, package):
        from ..verification import validate_document
        validate_document(package.manifest, package.document)
        if package.manifest["format_id"] != "slai.sdrw":
            raise ValidationError("package does not contain a drawing")
        return cls(**package.document, unit=package.manifest["units"])
