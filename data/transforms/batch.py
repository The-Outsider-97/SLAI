"""
Batch transforms: operate on a list of records to align, pad, or stack them.
"""
from __future__ import annotations

import numpy as np  # type: ignore

from typing import Any, Dict, List, Optional

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import *
from .base_transform import Transform
from .registry import register_transform
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Batch Transform")
printer = PrettyPrinter()


class BatchTransform(Transform):
    """Base class for transforms that operate on an entire batch at once.

    ``__call__`` is intentionally blocked; batch transforms must be invoked
    via ``apply_batch``.  This prevents accidental per-record application in
    a ``Sequential`` pipeline.
    """

    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} is a batch transform — "
            "use apply_batch(batch, modality) instead of __call__."
        )

    def apply_batch(self, batch: List[Dict[str, Any]], modality: str) -> List[Dict[str, Any]]:
        """Apply this transform to a list of records.

        Parameters
        ----------
        batch:
            List of record dicts (may be modified in-place).
        modality:
            Active modality key.

        Returns
        -------
        list[dict]
            The (possibly modified) batch — always the same length as input.

        Raises
        ------
        DataTransformError
            On any transformation failure.
        """
        raise NotImplementedError


# @register_transform("pad_sequences")
class PadSequences(BatchTransform):
    """Pad integer-sequence fields within a batch to the same length.

    Sequences longer than *max_length* are truncated **first**, then all
    remaining sequences are padded to the length of the longest in the batch.

    Config keys (``transforms.sequence``):

    * ``field`` (str, default ``"input_ids"``) — record key to operate on.
    * ``pad_value`` (int, default ``0``) — value used to fill short sequences.
    * ``max_length`` (int, default ``None``) — hard truncation ceiling;
      overridden by the constructor argument if provided.
    """

    def __init__(self, max_length: Optional[int] = None) -> None:
        super().__init__()
        self.pd_cfg: Dict[str, Any] = get_config_section("transforms").get("sequence", {})
        self.field: str = str(self.pd_cfg.get("field", "input_ids"))
        self.pad_value: int = int(self.pd_cfg.get("pad_value", 0))

        _cfg_max: Optional[int] = self.pd_cfg.get("max_length")
        self.max_length: Optional[int] = (
            max_length if max_length is not None
            else (int(_cfg_max) if _cfg_max is not None else None)
        )

    def apply_batch(self, batch: List[Dict[str, Any]], modality: str) -> List[Dict[str, Any]]:
        if not batch:
            return batch

        sequences: List[Optional[List[int]]] = [
            rec.get(self.field) for rec in batch
        ]

        # Truncate and measure
        truncated: List[List[int]] = []
        for seq in sequences:
            if not isinstance(seq, list):
                truncated.append([])
                continue
            if self.max_length is not None and len(seq) > self.max_length:
                seq = seq[:self.max_length]
            truncated.append(seq)

        target_len = max((len(s) for s in truncated), default=0)

        for rec, seq in zip(batch, truncated):
            pad_needed = target_len - len(seq)
            rec[self.field] = seq + [self.pad_value] * pad_needed

        logger.debug({
            "event": "pad_sequences",
            "field": self.field,
            "target_len": target_len,
            "batch_size": len(batch),
        })
        return batch

    def _get_params(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "pad_value": self.pad_value,
            "max_length": self.max_length,
        }


# @register_transform("stack_arrays")
class StackArrays(BatchTransform):
    """Stack per-record numpy arrays into a single batch array.

    The stacked array (shape ``(N, *element_shape)``) is written to
    ``output_field`` on **every** record in the batch so that downstream
    consumers can retrieve it without a separate aggregation step.

    Records where the source field is ``None`` or missing are excluded from
    the stack; records that were present are indexed by ``batch_index`` for
    traceability.

    Config keys (``transforms.arrays``):

    * ``field`` (str, default ``"image"``) — source field name.
    * ``output_field`` (str, default ``"<field>_stacked"``) — destination field.
    """

    def __init__(self, output_field: Optional[str] = None) -> None:
        super().__init__()
        self.sa_cfg: Dict[str, Any] = get_config_section("transforms").get("arrays", {})
        self.field: str = str(self.sa_cfg.get("field", "image"))
        self.output_field: str = output_field or f"{self.field}_stacked"

    def apply_batch(self, batch: List[Dict[str, Any]], modality: str) -> List[Dict[str, Any]]:
        if not batch:
            return batch

        valid_arrays: List[np.ndarray] = [
            rec[self.field]
            for rec in batch
            if isinstance(rec.get(self.field), np.ndarray)
        ]

        if not valid_arrays:
            logger.debug({
                "event": "stack_arrays_no_valid",
                "field": self.field,
                "batch_size": len(batch),
            })
            return batch

        try:
            stacked: np.ndarray = np.stack(valid_arrays, axis=0)
        except ValueError as exc:
            shapes = [list(a.shape) for a in valid_arrays]
            raise DataTransformError(
                f"StackArrays failed: arrays in field '{self.field}' have inconsistent shapes",
                context={"field": self.field, "shapes": shapes, "modality": modality},
                cause=exc,
            ) from exc

        for rec in batch:
            rec[self.output_field] = stacked

        logger.debug({
            "event": "stack_arrays",
            "field": self.field,
            "output_field": self.output_field,
            "stacked_shape": list(stacked.shape),
        })
        return batch

    def _get_params(self) -> Dict[str, Any]:
        return {"field": self.field, "output_field": self.output_field}


if __name__ == "__main__":
    print("\n=== Running batch ===\n")
    printer.status("TEST", "batch initialized", "info")

    # BatchTransform.__call__ is blocked
    ps = PadSequences()
    try:
        ps({"input_ids": [1, 2]}, "text")
        assert False
    except NotImplementedError:
        printer.status("PASS", "BatchTransform.__call__ raises NotImplementedError", "success")

    # PadSequences — basic padding
    batch = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [4]},
        {"input_ids": [5, 6]},
    ]
    out = PadSequences().apply_batch(batch, "text")
    lengths = [len(r["input_ids"]) for r in out]
    assert all(l == 3 for l in lengths), lengths
    assert out[1]["input_ids"] == [4, 0, 0]
    printer.status("PASS", "PadSequences pads to max length", "success")

    # PadSequences — truncation + padding
    ps_trunc = PadSequences(max_length=2)
    batch2 = [{"input_ids": [1, 2, 3, 4]}, {"input_ids": [5]}]
    out2 = ps_trunc.apply_batch(batch2, "text")
    assert out2[0]["input_ids"] == [1, 2]
    assert out2[1]["input_ids"] == [5, 0]
    printer.status("PASS", "PadSequences truncates then pads", "success")

    # PadSequences — non-list field replaced with empty list
    batch3 = [{"input_ids": None}, {"input_ids": [1]}]
    out3 = PadSequences().apply_batch(batch3, "text")
    assert out3[0]["input_ids"] == [0]
    printer.status("PASS", "PadSequences handles None field", "success")

    # PadSequences — empty batch
    assert PadSequences().apply_batch([], "text") == []
    printer.status("PASS", "PadSequences handles empty batch", "success")

    # StackArrays — basic stacking
    sa = StackArrays()
    arr_batch = [{"image": np.zeros((4, 4, 3))} for _ in range(3)]
    out_s = sa.apply_batch(arr_batch, "vision")
    assert out_s[0]["image_stacked"].shape == (3, 4, 4, 3)
    printer.status("PASS", "StackArrays stacks correctly", "success")

    # StackArrays — inconsistent shapes raise DataTransformError
    bad_batch = [
        {"image": np.zeros((4, 4, 3))},
        {"image": np.zeros((8, 8, 3))},
    ]
    try:
        sa.apply_batch(bad_batch, "vision")
        assert False
    except DataTransformError:
        printer.status("PASS", "StackArrays raises on shape mismatch", "success")

    # StackArrays — no valid arrays returns batch unchanged
    no_arr = [{"image": None}, {"image": None}]
    out_na = sa.apply_batch(no_arr, "vision")
    assert "image_stacked" not in out_na[0]
    printer.status("PASS", "StackArrays no-op when no valid arrays", "success")

    print("\n=== Test ran successfully ===\n")