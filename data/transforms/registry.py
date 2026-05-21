from __future__ import annotations
 
import threading
 
from typing import Any, Dict, Optional, Type
 
from ..utils.data_error import *
from ..utils.data_helpers import *
from .base_transform import Transform
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]
 
logger = get_logger("Transform Registry")
printer = PrettyPrinter()
 
# ---------------------------------------------------------------------------
# Internal registry state
# ---------------------------------------------------------------------------
_REGISTRY_LOCK = threading.Lock()
_TRANSFORM_REGISTRY: Dict[str, Type[Transform]] = {}
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def register_transform(name: str, *, overwrite: bool = False):
    """Class decorator: register a ``Transform`` subclass under *name*.
 
    Parameters
    ----------
    name:
        The stable string key used in serialised pipeline configs.
    overwrite:
        When ``False`` (default), raise ``DataConfigError`` if *name* is
        already registered — prevents accidental silent replacements.
 
    Example
    -------
    ::
 
        @register_transform("resize_image")
        class ResizeImage(Transform):
            ...
    """
    def decorator(cls: Type[Transform]) -> Type[Transform]:
        with _REGISTRY_LOCK:
            if name in _TRANSFORM_REGISTRY:
                existing_cls = _TRANSFORM_REGISTRY[name]
                if existing_cls is cls:
                    # Same class re‑registered – this is safe (e.g., due to reload or __main__ import)
                    return cls
                if not overwrite:
                    raise DataConfigError(
                        f"Transform '{name}' is already registered. "
                        "Pass overwrite=True to replace it.",
                        context={"name": name, "existing": existing_cls.__name__},
                    )
            _TRANSFORM_REGISTRY[name] = cls
        logger.debug({"event": "transform_registered", "name": name, "class": cls.__name__})
        return cls
    return decorator
 
 
def get_transform(config: Dict[str, Any]) -> Transform:
    """Instantiate a ``Transform`` from a serialised config dict.
 
    The dict must contain a ``"type"`` key whose value matches a registered
    name.  An optional ``"params"`` sub-dict is unpacked as keyword arguments
    to the class constructor.
 
    .. important::
        This function does **not** mutate *config* — the original dict is left
        intact (the original code called ``dict.pop`` which silently destroyed
        the config for callers that reused it).
 
    Parameters
    ----------
    config:
        A dict of the form ``{"type": "resize_image", "params": {...}}``.
 
    Returns
    -------
    Transform
        A freshly constructed transform instance.
 
    Raises
    ------
    DataConfigError
        If ``"type"`` is missing, or the name is not in the registry, or
        constructor instantiation fails.
    """
    if not isinstance(config, dict):
        raise DataConfigError(
            "Transform config must be a dict",
            context={"got": type(config).__name__},
        )
 
    transform_type: Optional[str] = config.get("type")
    if not transform_type:
        raise DataConfigError(
            "Transform config missing required 'type' key",
            context={"config_keys": list(config.keys())},
        )
 
    with _REGISTRY_LOCK:
        cls = _TRANSFORM_REGISTRY.get(transform_type)
 
    if cls is None:
        raise DataConfigError(
            f"Unknown transform type: '{transform_type}'",
            context={
                "requested": transform_type,
                "available": sorted(_TRANSFORM_REGISTRY.keys()),
            },
        )
 
    params: Dict[str, Any] = config.get("params") or {}
    try:
        return cls(**params)
    except Exception as exc:
        raise DataConfigError(
            f"Failed to instantiate transform '{transform_type}'",
            context={"transform_type": transform_type, "params": params},
            cause=exc,
        ) from exc
 
 
def list_transforms() -> list[str]:
    """Return a sorted list of all registered transform names."""
    with _REGISTRY_LOCK:
        return sorted(_TRANSFORM_REGISTRY.keys())
 
 
def clear_registry() -> None:
    """Remove all registered transforms.
 
    Intended **only** for use in unit tests that need a clean slate between
    test cases.  Do not call in production code.
    """
    with _REGISTRY_LOCK:
        _TRANSFORM_REGISTRY.clear()
    logger.debug({"event": "registry_cleared"})


if __name__ == "__main__":
    print("\n=== Running registry ===\n")
    printer.status("TEST", "registry initialized", "info")
 
    clear_registry()
 
    # Register a transform
    @register_transform("_test_noop")
    class _NoOp(Transform):
        def __call__(self, record, modality):
            return record
 
    assert "_test_noop" in list_transforms()
    printer.status("PASS", "register_transform + list_transforms", "success")
 
    # get_transform round-trip
    t = get_transform({"type": "_test_noop", "params": {}})
    assert isinstance(t, _NoOp)
    printer.status("PASS", "get_transform instantiates correct class", "success")
 
    # Config dict is not mutated
    cfg = {"type": "_test_noop", "params": {}}
    get_transform(cfg)
    assert cfg == {"type": "_test_noop", "params": {}}, "config was mutated!"
    printer.status("PASS", "get_transform does not mutate config", "success")
 
    # Duplicate registration guarded
    try:
        @register_transform("_test_noop")
        class _NoOp2(Transform):
            def __call__(self, record, modality): return record
        assert False
    except DataConfigError:
        printer.status("PASS", "duplicate registration rejected without overwrite=True", "success")
 
    # overwrite=True replaces silently
    @register_transform("_test_noop", overwrite=True)
    class _NoOp3(Transform):
        def __call__(self, record, modality): return record
 
    printer.status("PASS", "overwrite=True accepted", "success")
 
    # Unknown type raises DataConfigError
    try:
        get_transform({"type": "_does_not_exist"})
        assert False
    except DataConfigError:
        printer.status("PASS", "unknown transform type raises DataConfigError", "success")
 
    # Missing 'type' key raises DataConfigError
    try:
        get_transform({"params": {}})
        assert False
    except DataConfigError:
        printer.status("PASS", "missing 'type' raises DataConfigError", "success")
 
    clear_registry()
    assert list_transforms() == []
    printer.status("PASS", "clear_registry empties registry", "success")
 
    print("\n=== Test ran successfully ===\n")