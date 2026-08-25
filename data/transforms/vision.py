"""
Vision transforms: resize, normalize, and grayscale conversion.
"""
from __future__ import annotations

import numpy as np  # type: ignore

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import *
from .base_transform import Transform
from .registry import register_transform
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Vision Transform")
printer = PrettyPrinter()

try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

_INTERPOLATION_MAP = {
    "nearest":  "INTER_NEAREST",
    "linear":   "INTER_LINEAR",
    "bilinear": "INTER_LINEAR",
    "cubic":    "INTER_CUBIC",
    "area":     "INTER_AREA",
    "lanczos":  "INTER_LANCZOS4",
}


def _require_cv2(transform_name: str) -> None:
    if not CV2_AVAILABLE:
        raise DataConfigError(
            f"{transform_name} requires opencv-python: pip install opencv-python",
            context={"transform": transform_name},
        )


def _load_image(img: Any, transform_name: str) -> np.ndarray:
    """Coerce *img* to an ``np.ndarray``.

    Accepted input types:

    * ``np.ndarray`` — returned as-is.
    * ``str`` / ``Path`` — loaded from disk via ``cv2.imread``.
    * Any object with a ``__array__`` method (e.g. PIL Image).
    """
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, (str, Path)):
        _require_cv2(transform_name)
        loaded = cv2.imread(str(img))
        if loaded is None:
            raise DataSourceError(
                f"{transform_name}: cv2.imread returned None",
                context={"path": str(img), "transform": transform_name},
            )
        return loaded
    if hasattr(img, "__array__"):
        return np.asarray(img)
    raise DataTransformError(
        f"{transform_name}: unsupported image type",
        context={"got": type(img).__name__, "transform": transform_name},
    )


# @register_transform("resize_image")
class ResizeImage(Transform):
    """Resize an image to a fixed ``(width, height)`` using the specified
    interpolation method.

    Config keys (``transforms.vision``):

    * ``size`` (list[int, int], default ``[224, 224]``) — ``[width, height]``.
    * ``interpolation`` (str, default ``"bilinear"``).

    The ``size`` and ``interpolation`` constructor args override config values.
    """

    def __init__(self, size: Optional[Union[Tuple[int, int], List[int]]] = None,
                 interpolation: Optional[str] = None) -> None:
        super().__init__()
        self.vision_cfg: Dict[str, Any] = get_config_section("transforms").get("vision", {})

        _cfg_size = self.vision_cfg.get("size", [224, 224])
        raw_size = size if size is not None else _cfg_size
        if len(raw_size) != 2:
            raise DataConfigError(
                "ResizeImage: size must be [width, height]",
                context={"size": raw_size},
            )
        self.size: Tuple[int, int] = (int(raw_size[0]), int(raw_size[1]))

        _cfg_interp = self.vision_cfg.get("interpolation", "bilinear")
        interp_name = (interpolation or _cfg_interp).lower()
        cv2_attr = _INTERPOLATION_MAP.get(interp_name)
        if cv2_attr is None:
            raise DataConfigError(
                f"ResizeImage: unsupported interpolation '{interp_name}'",
                context={"supported": list(_INTERPOLATION_MAP.keys())},
            )
        self.interpolation_name: str = interp_name
        # Defer cv2 constant resolution to call time to avoid import at init.
        self._cv2_attr: str = cv2_attr

    def _interpolation_flag(self) -> int:
        _require_cv2("ResizeImage")
        return getattr(cv2, self._cv2_attr, cv2.INTER_LINEAR)

    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        if modality != "vision":
            return record

        raw = record.get("image")
        if raw is None:
            return record

        try:
            img = _load_image(raw, "ResizeImage")
            resized = cv2.resize(img, self.size, interpolation=self._interpolation_flag())
        except (DataConfigError, DataSourceError, DataTransformError):
            raise
        except Exception as exc:
            raise DataTransformError(
                "ResizeImage failed",
                context={"modality": modality, "size": self.size},
                cause=exc,
            ) from exc

        record["image"] = resized
        logger.debug({
            "event": "image_resized",
            "input_shape": list(img.shape),
            "output_shape": list(resized.shape),
            "size": self.size,
        })
        return record

    def _get_params(self) -> Dict[str, Any]:
        return {"size": list(self.size), "interpolation": self.interpolation_name}


# @register_transform("normalize_image")
class NormalizeImage(Transform):
    """Normalise an image array: ``(pixel / 255 - mean) / std``.

    Operates on ``float32`` arrays; uint8 arrays are converted automatically.
    Supports both grayscale (H, W) and colour (H, W, C) inputs.

    Config keys (``transforms.vision``):

    * ``mean`` (list[float]) — per-channel mean (default ImageNet: [0.485, 0.456, 0.406]).
    * ``std``  (list[float]) — per-channel std  (default ImageNet: [0.229, 0.224, 0.225]).
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, mean: Optional[List[float]] = None, std: Optional[List[float]] = None) -> None:
        super().__init__()
        self.vision_cfg: Dict[str, Any] = get_config_section("transforms").get("vision", {})

        self.mean: List[float] = mean or self.vision_cfg.get("mean", self._IMAGENET_MEAN)
        self.std:  List[float] = std  or self.vision_cfg.get("std",  self._IMAGENET_STD)

        if any(s == 0 for s in self.std):
            raise DataConfigError(
                "NormalizeImage: std contains zero — division by zero",
                context={"std": self.std},
            )

    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        if modality != "vision":
            return record

        img = record.get("image")
        if img is None:
            return record
        if not isinstance(img, np.ndarray):
            raise DataTransformError(
                "NormalizeImage expects record['image'] to be np.ndarray",
                context={"got": type(img).__name__},
            )

        try:
            arr = img.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0

            mean = np.array(self.mean, dtype=np.float32)
            std  = np.array(self.std,  dtype=np.float32)

            if arr.ndim == 2:
                # Grayscale: use first channel stats only
                arr = (arr - mean[0]) / std[0]
            else:
                arr = (arr - mean) / std
        except (DataConfigError, DataTransformError):
            raise
        except Exception as exc:
            raise DataTransformError(
                "NormalizeImage failed",
                context={"modality": modality, "image_shape": list(img.shape)},
                cause=exc,
            ) from exc

        record["image"] = arr
        return record

    def _get_params(self) -> Dict[str, Any]:
        return {"mean": self.mean, "std": self.std}


# @register_transform("to_grayscale")
class ToGrayscale(Transform):
    """Convert an RGB/BGR image to grayscale.

    Supports both ``(H, W, 3)`` and ``(H, W, 4)`` inputs; already-grayscale
    ``(H, W)`` arrays are returned unchanged.  Requires **opencv-python**.
    """

    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        if modality != "vision":
            return record

        img = record.get("image")
        if img is None:
            return record
        if not isinstance(img, np.ndarray):
            raise DataTransformError(
                "ToGrayscale expects record['image'] to be np.ndarray",
                context={"got": type(img).__name__},
            )
        if img.ndim == 2:
            return record  # already grayscale

        _require_cv2("ToGrayscale")
        try:
            code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            record["image"] = cv2.cvtColor(img, code)
        except Exception as exc:
            raise DataTransformError(
                "ToGrayscale failed",
                context={"modality": modality, "image_shape": list(img.shape)},
                cause=exc,
            ) from exc
        return record

    def _get_params(self) -> Dict[str, Any]:
        return {}


if __name__ == "__main__":
    print("\n=== Running vision ===\n")
    printer.status("TEST", "vision initialized", "info")

    # NormalizeImage — basic normalisation
    ni = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    img = np.full((4, 4, 3), 128, dtype=np.uint8)
    rec = {"image": img}
    out = ni(rec, "vision")
    arr = out["image"]
    assert arr.dtype == np.float32
    assert abs(arr.mean()) < 0.1, f"unexpected mean: {arr.mean()}"
    printer.status("PASS", "NormalizeImage uint8 → float32, correct normalisation", "success")

    # NormalizeImage — skips non-vision modality
    out_skip = ni({"image": img}, "audio")
    assert out_skip["image"] is img
    printer.status("PASS", "NormalizeImage skips non-vision modality", "success")

    # NormalizeImage — raises on zero std
    try:
        NormalizeImage(mean=[0.5], std=[0.0, 0.5, 0.5])
        assert False
    except DataConfigError:
        printer.status("PASS", "NormalizeImage rejects zero std", "success")

    # NormalizeImage — non-ndarray raises DataTransformError
    try:
        ni({"image": "path/to/img.jpg"}, "vision")
        assert False
    except DataTransformError:
        printer.status("PASS", "NormalizeImage raises on non-ndarray", "success")

    # ResizeImage — DataConfigError on bad interpolation
    try:
        ResizeImage(size=(64, 64), interpolation="unknown_interp")
        assert False
    except DataConfigError:
        printer.status("PASS", "ResizeImage rejects unknown interpolation", "success")

    # ResizeImage — skips record without 'image'
    ri = ResizeImage(size=(64, 64))
    out_no_img = ri({"text": "hi"}, "vision")
    assert "image" not in out_no_img
    printer.status("PASS", "ResizeImage skips record without image key", "success")

    if CV2_AVAILABLE:
        # ResizeImage — functional resize
        img_big = np.zeros((128, 128, 3), dtype=np.uint8)
        out_r = ri({"image": img_big}, "vision")
        assert out_r["image"].shape[:2] == (64, 64)
        printer.status("PASS", "ResizeImage resizes correctly", "success")

        # ToGrayscale — RGB → grayscale
        img_rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        out_g = ToGrayscale()({"image": img_rgb}, "vision")
        assert out_g["image"].ndim == 2
        printer.status("PASS", "ToGrayscale RGB→grayscale", "success")

        # ToGrayscale — already grayscale, unchanged
        img_gray = np.zeros((8, 8), dtype=np.uint8)
        out_g2 = ToGrayscale()({"image": img_gray}, "vision")
        assert out_g2["image"].ndim == 2 and out_g2["image"].shape == (8, 8)
        printer.status("PASS", "ToGrayscale no-op on already-grayscale image", "success")
    else:
        printer.status("SKIP", "ResizeImage / ToGrayscale cv2 tests skipped (cv2 not installed)", "warning")

    # _get_params coverage
    assert "size" in ResizeImage()._get_params()
    assert "mean" in NormalizeImage()._get_params()
    assert ToGrayscale()._get_params() == {}
    printer.status("PASS", "_get_params correct for all vision transforms", "success")

    print("\n=== Test ran successfully ===\n")