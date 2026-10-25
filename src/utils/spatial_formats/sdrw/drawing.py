"""SDRW drawing document model with explicit layers, stable IDs and revisions."""
from __future__ import annotations
from copy import deepcopy
from ..common import ValidationError, new_id, require_id, assert_unique, require_unit, require_metadata
from .entities import validate_entity, make_entity

class Drawing:
    def __init__(self, name="Untitled", *, unit="mm", document_id=None, revision=1,
                 layers=None, entities=None, metadata=None):
        self.name = name
        self.unit = require_unit(unit)
        self.document_id = document_id or new_id()
        self.revision = revision
        self.layers = deepcopy(layers) if layers is not None else [{"name":"0", "visible":True}]
        self.entities = deepcopy(entities) if entities is not None else []
        self.metadata = deepcopy(metadata) if metadata is not None else {}
        self.validate()

    def validate(self):
        require_id(self.document_id)
        if type(self.revision) is not int or self.revision < 1:
            raise ValidationError("revision must be a positive integer")
        if not isinstance(self.name, str) or len(self.name) > 255:
            raise ValidationError("invalid drawing name")
        require_metadata(self.metadata)
        if not isinstance(self.layers, list) or not 1 <= len(self.layers) <= 10_000:
            raise ValidationError("drawing requires at least one layer")
        names = []
        for layer in self.layers:
            if not isinstance(layer, dict) or set(layer) - {"name","visible"}:
                raise ValidationError("invalid layer")
            name = layer.get("name")
            if not isinstance(name, str) or not name or len(name) > 255:
                raise ValidationError("invalid layer name")
            if type(layer.get("visible", True)) is not bool:
                raise ValidationError("layer visibility must be Boolean")
            names.append(name)
        if len(names) != len(set(names)):
            raise ValidationError("duplicate drawing layer")
        if not isinstance(self.entities, list) or len(self.entities) > 100_000:
            raise ValidationError("too many drawing entities")
        assert_unique(ent.get("id") if isinstance(ent,dict) else None for ent in self.entities)
        for ent in self.entities:
            validate_entity(ent, set(names))
        return self

    def add_layer(self, name, *, visible=True):
        if any(x["name"] == name for x in self.layers):
            raise ValidationError("drawing layer already exists")
        self.layers.append({"name":name,"visible":visible})
        try:
            self.validate()
        except Exception:
            self.layers.pop()
            raise
        self.revision += 1
        return name

    def add(self, kind, geometry, **options):
        ent = make_entity(kind, geometry, **options)
        self.entities.append(ent)
        try:
            self.validate()
        except Exception:
            self.entities.pop()
            raise
        self.revision += 1
        return ent["id"]

    def remove(self, entity_id):
        require_id(entity_id)
        found = next((i for i,e in enumerate(self.entities) if e["id"] == entity_id), None)
        if found is None:
            raise ValidationError("drawing entity not found")
        self.entities.pop(found)
        self.revision += 1

    def as_dict(self):
        self.validate()
        return deepcopy({"document_id":self.document_id,"revision":self.revision,
                         "name":self.name,"layers":self.layers,"entities":self.entities,
                         "metadata":self.metadata})

    def manifest(self):
        self.validate()
        return {"format_id":"slai.sdrw", "format_version":"0.1.0", "schema":"slai.sdrw/0.1",
                "spatial_dimension":2,"document_kind":"drawing","document_id":self.document_id,
                "revision":self.revision,"units":self.unit,"requires":["drawing-primitives"]}

    @classmethod
    def from_package(cls, package):
        from ..verification import validate_document
        validate_document(package.manifest, package.document)
        if package.manifest["format_id"] != "slai.sdrw":
            raise ValidationError("package does not contain a drawing")
        return cls(**package.document, unit=package.manifest["units"])
