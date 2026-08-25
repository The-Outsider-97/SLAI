from __future__ import annotations

from typing import Dict, Any, List, Optional

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import *
from .base_transform import Transform
from .registry import list_transforms
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Composite Transform")
printer = PrettyPrinter()


class Sequential(Transform):
    """Apply a list of transforms in order.

    Each transform's output becomes the next transform's input.  If any
    transform raises a ``DataError`` it propagates immediately — the
    pipeline does **not** catch or swallow per-step errors.

    Parameters
    ----------
    transforms:
        Ordered list of ``Transform`` instances.
    """

    def __init__(self, transforms: Optional[List[Transform]] = None) -> None:
        super().__init__(name="sequential")
        if transforms is not None and not isinstance(transforms, list):
            raise DataConfigError(
                "Sequential requires a list of Transform instances",
                context={"got": type(transforms).__name__},
            )
        self.transforms: List[Transform] = transforms or []

    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        for transform in self.transforms:
            record = transform(record, modality)
        return record

    def append(self, transform: Transform) -> "Sequential":
        """Add a transform to the end of the sequence and return self."""
        if not isinstance(transform, Transform):
            raise DataConfigError(
                "Only Transform instances can be appended to Sequential",
                context={"got": type(transform).__name__},
            )
        self.transforms.append(transform)
        return self

    def to_config(self) -> Dict[str, Any]:
        return {
            "type": "sequential",
            "params": {
                "transforms": [t.to_config() for t in self.transforms]
            },
        }

    def _get_params(self) -> Dict[str, Any]:
        return {"num_transforms": len(self.transforms)}

    def __len__(self) -> int:
        return len(self.transforms)


class PerModality(Transform):
    """Dispatch records to modality-specific sub-transforms.

    Only the sub-transform whose key matches *modality* is invoked.
    Records for unregistered modalities are returned unchanged —
    this is intentional: ``PerModality`` must be safe to use in a
    ``Sequential`` that receives heterogeneous records.

    Parameters
    ----------
    mapping:
        Dict mapping modality name → ``Transform`` instance.
    """

    def __init__(self, mapping: Optional[Dict[str, Transform]] = None) -> None:
        super().__init__(name="per_modality")
        if mapping is not None and not isinstance(mapping, dict):
            raise DataConfigError(
                "PerModality mapping must be a dict",
                context={"got": type(mapping).__name__},
            )
        self.mapping: Dict[str, Transform] = mapping or {}

    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        transform = self.mapping.get(modality)
        if transform is not None:
            record = transform(record, modality)
        else:
            logger.debug({
                "event": "per_modality_no_transform",
                "modality": modality,
                "registered": list(self.mapping.keys()),
            })
        return record

    def register(self, modality: str, transform: Transform, *, overwrite: bool = False) -> None:
        """Add or replace a modality-specific transform at runtime."""
        if modality in self.mapping and not overwrite:
            raise DataConfigError(
                f"PerModality already has a transform for '{modality}'. "
                "Pass overwrite=True to replace it.",
                context={"modality": modality},
            )
        self.mapping[modality] = transform

    def to_config(self) -> Dict[str, Any]:
        return {
            "type": "per_modality",
            "params": {
                "mapping": {k: t.to_config() for k, t in self.mapping.items()}
            },
        }

    def _get_params(self) -> Dict[str, Any]:
        return {"modalities": sorted(self.mapping.keys())}


# ---------------------------------------------------------------------------
# Manual registration (idempotent) – solves double‑execution issue
# ---------------------------------------------------------------------------
def _register_composite_transforms() -> None:
    """Register Sequential and PerModality safely, even if this module is
    executed twice (e.g., via `python -m`)."""
    registered = list_transforms()
    if "sequential" not in registered:
        # Manually register Sequential using the decorator's underlying mechanism.
        # We cannot call register_transform as a function because it's a decorator.
        # Instead, we directly add to the registry via the internal dict.
        # This avoids the decorator's identity check.
        from .registry import _TRANSFORM_REGISTRY, _REGISTRY_LOCK
        with _REGISTRY_LOCK:
            if "sequential" not in _TRANSFORM_REGISTRY:
                _TRANSFORM_REGISTRY["sequential"] = Sequential
                logger.debug({"event": "transform_registered", "name": "sequential", "class": "Sequential"})
            if "per_modality" not in _TRANSFORM_REGISTRY:
                _TRANSFORM_REGISTRY["per_modality"] = PerModality
                logger.debug({"event": "transform_registered", "name": "per_modality", "class": "PerModality"})

_register_composite_transforms()


if __name__ == "__main__":
    print("\n=== Running composite ===\n")
    printer.status("TEST", "composite initialized", "info")

    class _Append(Transform):
        def __init__(self, tag: str) -> None:
            super().__init__(name=f"append_{tag}")
            self.tag = tag
        def __call__(self, record, modality):
            record["log"] = record.get("log", "") + self.tag
            return record
        def _get_params(self): return {"tag": self.tag}

    # Sequential: transforms applied in order
    seq = Sequential([_Append("A"), _Append("B"), _Append("C")])
    out = seq({"log": ""}, "text")
    assert out["log"] == "ABC", out["log"]
    printer.status("PASS", "Sequential applies transforms in order", "success")

    # Sequential.append
    seq.append(_Append("D"))
    assert len(seq) == 4
    out2 = seq({"log": ""}, "text")
    assert out2["log"] == "ABCD"
    printer.status("PASS", "Sequential.append works", "success")

    # Sequential.to_config round-trip
    cfg = seq.to_config()
    assert cfg["type"] == "sequential"
    assert len(cfg["params"]["transforms"]) == 4
    printer.status("PASS", "Sequential.to_config correct", "success")

    # PerModality: only matching modality transform runs
    pm = PerModality({
        "text": _Append("T"),
        "audio": _Append("A"),
    })
    rec_text = pm({"log": ""}, "text")
    assert rec_text["log"] == "T"
    rec_vision = pm({"log": ""}, "vision")
    assert rec_vision["log"] == ""    # no transform for vision
    printer.status("PASS", "PerModality dispatches correctly", "success")

    # PerModality.register + overwrite guard
    pm.register("vision", _Append("V"))
    rec_v = pm({"log": ""}, "vision")
    assert rec_v["log"] == "V"
    try:
        pm.register("vision", _Append("X"), overwrite=False)
        assert False
    except DataConfigError:
        printer.status("PASS", "PerModality overwrite guard works", "success")

    # PerModality.to_config
    cfg_pm = pm.to_config()
    assert cfg_pm["type"] == "per_modality"
    assert set(cfg_pm["params"]["mapping"].keys()) == {"text", "audio", "vision"}
    printer.status("PASS", "PerModality.to_config correct", "success")

    # Nested Sequential → PerModality
    nested = Sequential([pm, _Append("Z")])
    rec_n = nested({"log": ""}, "text")
    assert rec_n["log"] == "TZ"
    printer.status("PASS", "Nested Sequential+PerModality works", "success")

    print("\n=== Test ran successfully ===\n")