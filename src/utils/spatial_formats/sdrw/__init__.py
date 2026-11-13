"""Editable 2D drawing models; this is not the public SDRW facade."""
from .entities import KINDS, make_entity, validate_entity, entity_bounds
from .drawing import Drawing

from .operations import entity_length, enclosed_area, transform_entity
from .operations import entity_bounds as tight_entity_bounds
from .transactions import apply_drawing_operations
