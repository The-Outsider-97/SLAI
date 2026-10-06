from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Dict, Optional

from ...base.utils.base_errors import BaseError, BaseErrorType


class SpatialError(BaseError):
    """Root exception for the SLAI Spatial subsystem."""

    error_type = BaseErrorType.RUNTIME
    default_code = "SPA-1000"
    default_severity = "medium"
    default_retryable = False
    default_category = "spatial"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            code=code or self.default_code,
            context=context,
            **kwargs,
        )


class SpatialValidationError(SpatialError):
    error_type = BaseErrorType.VALIDATION
    default_code = "SPA-1100"
    default_category = "spatial.validation"


class SpatialFrameError(SpatialError):
    error_type = BaseErrorType.STATE
    default_code = "SPA-1200"
    default_category = "spatial.frames"


class SpatialGeometryError(SpatialError):
    error_type = BaseErrorType.VALIDATION
    default_code = "SPA-1300"
    default_category = "spatial.geometry"


class SpatialTopologyError(SpatialError):
    error_type = BaseErrorType.VALIDATION
    default_code = "SPA-1400"
    default_category = "spatial.topology"


class SpatialIndexError(SpatialError):
    error_type = BaseErrorType.STATE
    default_code = "SPA-1500"
    default_category = "spatial.index"


class SpatialMappingError(SpatialError):
    error_type = BaseErrorType.RUNTIME
    default_code = "SPA-1600"
    default_category = "spatial.mapping"


class SpatialOccupancyError(SpatialError):
    error_type = BaseErrorType.VALIDATION
    default_code = "SPA-1700"
    default_category = "spatial.occupancy"


class SpatialQueryError(SpatialError):
    error_type = BaseErrorType.RUNTIME
    default_code = "SPA-1800"
    default_category = "spatial.query"


__all__ = [
    "SpatialError",
    "SpatialValidationError",
    "SpatialFrameError",
    "SpatialGeometryError",
    "SpatialTopologyError",
    "SpatialIndexError",
    "SpatialMappingError",
    "SpatialOccupancyError",
    "SpatialQueryError",
]
