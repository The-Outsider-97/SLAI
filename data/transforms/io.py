"""
Pipeline I/O: serialise transform pipelines to/from JSON and YAML.
"""
from __future__ import annotations

import json
import yaml  # pyright: ignore[reportMissingModuleSource]

from pathlib import Path
from typing import Any, Dict, Union

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import *
from .base_transform import Transform
from .registry import get_transform
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Pipeline IO")
printer = PrettyPrinter()

_SUPPORTED_FORMATS = frozenset({"json", "yaml", "yml"})
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_JSON_SUFFIXES = frozenset({".json"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def save_pipeline(pipeline: Transform, path: Union[str, Path], format: str = "json") -> Path:
    """Serialise *pipeline* to a JSON or YAML file.

    Parameters
    ----------
    pipeline:
        Any ``Transform`` instance (including ``Sequential``, ``PerModality``,
        ``CachedTransform``, etc.).
    path:
        Destination file path.  Parent directory must already exist.
    format:
        ``"json"`` (default), ``"yaml"``, or ``"yml"``.

    Returns
    -------
    Path
        The resolved destination path.

    Raises
    ------
    DataConfigError
        If the format is unsupported or the parent directory does not exist.
    DataTransformError
        If the pipeline config is not JSON-serialisable (JSON path only).
    """
    if format not in _SUPPORTED_FORMATS:
        raise DataConfigError(
            f"Unsupported pipeline format: '{format}'",
            context={"format": format, "supported": sorted(_SUPPORTED_FORMATS)},
        )

    resolved = Path(path).expanduser().resolve()
    if not resolved.parent.exists():
        raise DataConfigError(
            "Destination directory does not exist",
            context={"directory": str(resolved.parent)},
        )

    config = pipeline.to_config()

    if format == "json":
        dest = atomic_write_json(config, resolved, indent=2)
    else:
        try:
            with open(resolved, "w", encoding="utf-8") as fh:
                yaml.safe_dump(config, fh, indent=2, sort_keys=False, allow_unicode=True)
        except (OSError, yaml.YAMLError) as exc:
            raise DataTransformError(
                "Failed to write pipeline YAML",
                context={"path": str(resolved)},
                cause=exc,
            ) from exc
        dest = resolved

    logger.info({"event": "pipeline_saved", "path": str(dest), "format": format})
    return dest


def load_pipeline(path: Union[str, Path]) -> Transform:
    """Load and reconstruct a pipeline from a JSON or YAML file.

    The file extension determines the parser:
    ``*.json`` → JSON; ``*.yaml`` / ``*.yml`` → YAML.

    Parameters
    ----------
    path:
        Path to a previously saved pipeline file.

    Returns
    -------
    Transform
        A fully reconstructed transform pipeline.

    Raises
    ------
    DataConfigError
        If the file does not exist, the extension is unsupported, or the
        config structure is invalid.
    DataSourceError
        If the file cannot be read.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise DataConfigError(
            "Pipeline file not found",
            context={"path": str(resolved)},
        )

    suffix = resolved.suffix.lower()
    try:
        with open(resolved, "r", encoding="utf-8") as fh:
            if suffix in _JSON_SUFFIXES:
                config: Dict[str, Any] = json.load(fh)
            elif suffix in _YAML_SUFFIXES:
                config = yaml.safe_load(fh) or {}
            else:
                raise DataConfigError(
                    f"Unsupported pipeline file extension: '{suffix}'",
                    context={"path": str(resolved), "supported": sorted(_JSON_SUFFIXES | _YAML_SUFFIXES)},
                )
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise DataSourceError(
            "Failed to read pipeline file",
            context={"path": str(resolved)},
            cause=exc,
        ) from exc

    logger.info({"event": "pipeline_loaded", "path": str(resolved)})
    return _build_from_config(config)


def load_pipeline_from_dict(config: Dict[str, Any]) -> Transform:
    """Reconstruct a pipeline from a configuration dictionary.

    Useful when the config has already been parsed from YAML/JSON elsewhere
    (e.g. embedded in a larger experiment config).

    Parameters
    ----------
    config:
        A dict of the form ``{"type": "<name>", "params": {...}}``.

    Returns
    -------
    Transform
        A reconstructed pipeline.
    """
    return _build_from_config(config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_from_config(config: Dict[str, Any]) -> Transform:
    """Recursively build a ``Transform`` from *config*.

    Delegates to ``registry.get_transform`` for leaf transforms.
    Composite transforms (``sequential``, ``per_modality``) recursively
    reconstruct their children so the full tree is restored.

    The ``composite`` module is imported here (not at module top-level) to
    avoid a circular import: ``composite`` → ``registry`` → ``io`` → ``composite``.
    """
    if not isinstance(config, dict):
        raise DataConfigError(
            "Pipeline config entry must be a dict",
            context={"got": type(config).__name__},
        )
    if "type" not in config:
        raise DataConfigError(
            "Pipeline config entry missing required 'type' key",
            context={"keys": list(config.keys())},
        )

    transform_type: str = config["type"]
    params: Dict[str, Any] = dict(config.get("params") or {})

    # Composite types require recursive child reconstruction.
    if transform_type == "sequential":
        from .composite import Sequential  # local import — avoids circular dep
        raw_transforms = params.pop("transforms", [])
        if not isinstance(raw_transforms, list):
            raise DataConfigError(
                "sequential.params.transforms must be a list",
                context={"got": type(raw_transforms).__name__},
            )
        children = [_build_from_config(t) for t in raw_transforms]
        return Sequential(transforms=children)

    if transform_type == "per_modality":
        from .composite import PerModality  # local import
        raw_mapping = params.pop("mapping", {})
        if not isinstance(raw_mapping, dict):
            raise DataConfigError(
                "per_modality.params.mapping must be a dict",
                context={"got": type(raw_mapping).__name__},
            )
        mapping = {k: _build_from_config(v) for k, v in raw_mapping.items()}
        return PerModality(mapping=mapping)

    if transform_type == "cached_transform":
        from .cache import CachedTransform  # local import
        inner_cfg = params.pop("inner", None)
        if inner_cfg is None:
            raise DataConfigError(
                "cached_transform.params.inner is required",
                context={},
            )
        inner = _build_from_config(inner_cfg)
        return CachedTransform(inner=inner, **params)

    # All other (leaf) transforms resolved via the registry.
    return get_transform({"type": transform_type, "params": params})


if __name__ == "__main__":
    import tempfile
    from .registry import register_transform, clear_registry

    print("\n=== Running io ===\n")
    printer.status("TEST", "io initialized", "info")

    clear_registry()

    @register_transform("_io_noop")
    class _NoOp(Transform):
        def __init__(self, tag: str = "default") -> None:
            super().__init__(name="_io_noop")
            self.tag = tag
        def __call__(self, record, modality): return record
        def _get_params(self): return {"tag": self.tag}

    from .composite import Sequential, PerModality

    # Build a nested pipeline
    pipeline = Sequential([
        PerModality({"text": _NoOp(tag="text_branch")}),
        _NoOp(tag="final"),
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        # save + load JSON
        json_path = Path(tmpdir) / "pipeline.json"
        saved = save_pipeline(pipeline, json_path, format="json")
        assert saved.exists()
        printer.status("PASS", "save_pipeline JSON", "success")

        restored = load_pipeline(json_path)
        from .composite import Sequential as Seq
        assert isinstance(restored, Seq)
        printer.status("PASS", "load_pipeline JSON round-trip", "success")

        # save + load YAML
        yaml_path = Path(tmpdir) / "pipeline.yaml"
        save_pipeline(pipeline, yaml_path, format="yaml")
        restored_yaml = load_pipeline(yaml_path)
        assert isinstance(restored_yaml, Seq)
        printer.status("PASS", "save/load YAML round-trip", "success")

        # load_pipeline_from_dict
        raw_cfg = pipeline.to_config()
        restored_dict = load_pipeline_from_dict(raw_cfg)
        assert isinstance(restored_dict, Seq)
        printer.status("PASS", "load_pipeline_from_dict", "success")

        # Unsupported format raises
        try:
            save_pipeline(pipeline, json_path, format="toml")
            assert False
        except DataConfigError:
            printer.status("PASS", "unsupported format raises DataConfigError", "success")

        # Missing file raises
        try:
            load_pipeline(Path(tmpdir) / "missing.json")
            assert False
        except DataConfigError:
            printer.status("PASS", "missing file raises DataConfigError", "success")

        # Bad extension raises
        bad = Path(tmpdir) / "pipeline.toml"
        bad.write_text("{}")
        try:
            load_pipeline(bad)
            assert False
        except DataConfigError:
            printer.status("PASS", "unsupported extension raises DataConfigError", "success")

    clear_registry()
    print("\n=== Test ran successfully ===\n")