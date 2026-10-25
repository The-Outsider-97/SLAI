"""Spatial serialization infrastructure independent of agent runtime."""
from .errors import (SpatialFormatError, ValidationError, IntegrityError, ResourceLimitError,
                     UnsupportedFeatureError, DependencyUnavailableError, ConflictError, MigrationError)
from .ids import new_id, require_id, assert_unique
from .helpers import canonical_json, parse_json, json_safe, finite, vector, digest
from .units import require_unit, convert_length, convert_area, convert_volume
from .transforms import IDENTITY, require_transform, transform_point, compose, invert_transform, is_rigid
from .metadata import require_metadata, require_claim
