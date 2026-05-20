from __future__ import annotations

import random

from collections.abc import Iterator
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Mapping, Sequence

from .utils.config_loader import get_config_section
from .utils.data_error import *
from .utils.data_helpers import *
from .utils.data_loader import *
from .governance import DatasetValidator
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Multimodal Dataset")
printer = PrettyPrinter()


# =============================================================================
# Data contracts
# =============================================================================

@dataclass(frozen=True)
class MultimodalBatch:
    """Serializable batch envelope for aligned multimodal records.

    `data` intentionally contains only modality keys so downstream training and
    evaluation code can keep consuming the same shape as the previous dataset
    iterator. `metadata` is kept beside the payload for observability and does
    not get passed into governance validation.
    """

    data: dict[str, tuple[Mapping[str, Any], ...]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_metadata: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {modality: list(rows) for modality, rows in self.data.items()}
        if include_metadata:
            payload["_metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class MultimodalDatasetSummary:
    """Compact operational summary for logging, debugging, and tests."""

    modalities: tuple[str, ...]
    record_count: int
    batch_size: int
    batch_count: int
    drop_remainder: bool
    shuffle: bool
    lengths: dict[str, int]
    quality: dict[str, Any] | None = None
    stats: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "modalities": list(self.modalities),
            "record_count": self.record_count,
            "batch_size": self.batch_size,
            "batch_count": self.batch_count,
            "drop_remainder": self.drop_remainder,
            "shuffle": self.shuffle,
            "lengths": dict(self.lengths),
            "quality": self.quality,
            "stats": self.stats,
        }


# =============================================================================
# Multimodal dataset
# =============================================================================

class MultimodalDataset:
    """Production-ready aligned dataset wrapper for vision/text/audio payloads.

    The class preserves the original call pattern while adding production
    controls expected by training, evaluation, and ingestion pipelines:

    * config-driven batch policy through `data_config.yaml`;
    * typed data-pipeline errors rather than bare built-in exceptions;
    * canonical helper usage for alignment, sanitisation, chunking, null audits,
      and modality statistics;
    * optional `DatasetValidator` integration for schema and quality gates;
    * deterministic shuffling, random-access samples, subset/split helpers, and
      operational metadata without changing the default batch payload shape.
    """

    DEFAULT_MODALITIES: tuple[str, str, str] = ("vision", "text", "audio")

    def __init__(
        self,
        vision_data: Sequence[Mapping[str, Any]],
        text_data: Sequence[Mapping[str, Any]],
        audio_data: Sequence[Mapping[str, Any]],
        batch_size: int | None = None,
        validator: DatasetValidator | None = None,
        *,
        extra_modalities: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        drop_remainder: bool | None = None,
        shuffle: bool | None = None,
        seed: int | None = None,
        validate_on_init: bool | None = None,
        validate_batches: bool | None = None,
        run_quality_gate_on_init: bool | None = None,
        sanitize_records: bool | None = None,
        copy_records: bool | None = None,
        include_batch_metadata: bool | None = None,
    ):
        self.dataset_cfg = get_config_section("dataset")
        self.ingestion_cfg = _mapping_section(self.dataset_cfg.get("ingestion", {}), "dataset.ingestion")
        self.runtime_cfg = _mapping_section(self.dataset_cfg.get("multimodal_dataset", {}), "dataset.multimodal_dataset")
        self.sanitization_cfg = _mapping_section(self.dataset_cfg.get("sanitization", {}), "dataset.sanitization")

        self.validator = validator
        self.batch_size = _config_int(
            batch_size if batch_size is not None else self.runtime_cfg.get("default_batch_size", 8),
            name="dataset.multimodal_dataset.default_batch_size",
            minimum=_config_int(self.ingestion_cfg.get("min_batch_size", 1), name="dataset.ingestion.min_batch_size", minimum=1),
        )
        self.max_batch_size = _optional_config_int(
            self.runtime_cfg.get("max_batch_size"),
            name="dataset.multimodal_dataset.max_batch_size",
            minimum=self.batch_size,
        )
        if self.max_batch_size is not None and self.batch_size > self.max_batch_size:
            raise DataIngestionContractError(
                "batch_size exceeds configured maximum",
                context={"batch_size": self.batch_size, "max_batch_size": self.max_batch_size},
            )

        self.drop_remainder = _config_bool(
            drop_remainder if drop_remainder is not None else self.runtime_cfg.get("drop_remainder", False),
            name="dataset.multimodal_dataset.drop_remainder",
        )
        self.shuffle = _config_bool(
            shuffle if shuffle is not None else self.runtime_cfg.get("shuffle", False),
            name="dataset.multimodal_dataset.shuffle",
        )
        self.reshuffle_each_epoch = _config_bool(
            self.runtime_cfg.get("reshuffle_each_epoch", True),
            name="dataset.multimodal_dataset.reshuffle_each_epoch",
        )
        self.validate_on_init = _config_bool(
            validate_on_init if validate_on_init is not None else self.runtime_cfg.get("validate_on_init", True),
            name="dataset.multimodal_dataset.validate_on_init",
        )
        self.validate_batches = _config_bool(
            validate_batches if validate_batches is not None else self.runtime_cfg.get("validate_batches", True),
            name="dataset.multimodal_dataset.validate_batches",
        )
        self.run_quality_gate_on_init = _config_bool(
            run_quality_gate_on_init
            if run_quality_gate_on_init is not None
            else self.runtime_cfg.get("run_quality_gate_on_init", False),
            name="dataset.multimodal_dataset.run_quality_gate_on_init",
        )
        self.sanitize_records = _config_bool(
            sanitize_records if sanitize_records is not None else self.runtime_cfg.get("sanitize_records", False),
            name="dataset.multimodal_dataset.sanitize_records",
        )
        self.copy_records = _config_bool(
            copy_records if copy_records is not None else self.runtime_cfg.get("copy_records", True),
            name="dataset.multimodal_dataset.copy_records",
        )
        self.include_batch_metadata = _config_bool(
            include_batch_metadata
            if include_batch_metadata is not None
            else self.runtime_cfg.get("include_batch_metadata", False),
            name="dataset.multimodal_dataset.include_batch_metadata",
        )
        self.allow_unaligned_batches = _config_bool(
            self.runtime_cfg.get("allow_unaligned_batches", False),
            name="dataset.multimodal_dataset.allow_unaligned_batches",
        )
        self.enforce_batch_alignment = _config_bool(
            self.ingestion_cfg.get("enforce_batch_alignment", True),
            name="dataset.ingestion.enforce_batch_alignment",
        )
        self.seed = _optional_config_int(
            seed if seed is not None else self.runtime_cfg.get("seed"),
            name="dataset.multimodal_dataset.seed",
            minimum=0,
        )

        raw_payload = self._build_raw_payload(vision_data, text_data, audio_data, extra_modalities)
        self.modalities = tuple(raw_payload.keys())
        self.expected_modalities = _expected_modalities(self.dataset_cfg, self.modalities)
        self._data = {
            modality: self._normalise_records(modality, records)
            for modality, records in raw_payload.items()
        }

        self.lengths = {modality: len(rows) for modality, rows in self._data.items()}
        self._assert_dataset_contract()
        self.total = min(self.lengths.values(), default=0)
        self.index = 0
        self._epoch = 0
        self._batch_iter: Iterator[Sequence[int]] | None = None
        self._last_quality_report: dict[str, Any] | None = None

        # Backward-compatible public modality attributes.
        self.vision = self._data["vision"]
        self.text = self._data["text"]
        self.audio = self._data["audio"]

        if self.validate_on_init:
            self.validate()
        if self.run_quality_gate_on_init:
            self._last_quality_report = self.quality_report()

        logger.info(
            {
                "event": "multimodal_dataset_initialized",
                "modalities": list(self.modalities),
                "lengths": self.lengths,
                "record_count": self.total,
                "batch_size": self.batch_size,
                "batch_count": len(self),
                "drop_remainder": self.drop_remainder,
                "shuffle": self.shuffle,
            }
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_raw_payload(
        self,
        vision_data: Sequence[Mapping[str, Any]],
        text_data: Sequence[Mapping[str, Any]],
        audio_data: Sequence[Mapping[str, Any]],
        extra_modalities: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    ) -> dict[str, Sequence[Mapping[str, Any]]]:
        payload: dict[str, Sequence[Mapping[str, Any]]] = {
            "vision": vision_data,
            "text": text_data,
            "audio": audio_data,
        }
        if extra_modalities:
            for modality, records in extra_modalities.items():
                key = str(modality).strip()
                if not key:
                    raise DataIngestionContractError("Modality name cannot be empty", context={"modality": modality})
                if key in payload:
                    raise DataIngestionContractError(
                        "extra_modalities cannot override built-in modalities",
                        context={"modality": key, "built_in_modalities": list(self.DEFAULT_MODALITIES)},
                    )
                payload[key] = records
        return payload

    def _normalise_records(
        self,
        modality: str,
        records: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], ...]:
        if records is None:  # type: ignore[comparison-overlap]
            raise DataIngestionContractError(f"{modality}: records cannot be None", context={"modality": modality})
        if isinstance(records, (str, bytes)) or not hasattr(records, "__len__"):
            raise DataIngestionContractError(
                f"{modality}: records must be a sized sequence of mappings",
                context={"modality": modality, "type": type(records).__name__},
            )

        escape_html = _config_bool(self.sanitization_cfg.get("escape_html", True), name="dataset.sanitization.escape_html")
        strip_control_chars = _config_bool(
            self.sanitization_cfg.get("strip_control_chars", True),
            name="dataset.sanitization.strip_control_chars",
        )
        max_string_length = _optional_config_int(
            self.sanitization_cfg.get("max_string_length"),
            name="dataset.sanitization.max_string_length",
            minimum=1,
        )
        max_depth = _config_int(self.sanitization_cfg.get("max_depth", 32), name="dataset.sanitization.max_depth", minimum=1)

        normalised: list[Mapping[str, Any]] = []
        for row_idx, row in enumerate(records):
            if not isinstance(row, Mapping):
                raise DataValidationError(
                    f"{modality}[{row_idx}] must be a mapping",
                    context={"modality": modality, "row_idx": row_idx, "type": type(row).__name__},
                )
            row_dict = dict(row) if self.copy_records or not isinstance(row, dict) else row
            if self.sanitize_records:
                row_dict = sanitize_dict(
                    row_dict,
                    escape_html=escape_html,
                    strip_control_chars=strip_control_chars,
                    max_string_length=max_string_length,
                    max_depth=max_depth,
                )
            normalised.append(row_dict)
        return tuple(normalised)

    def _assert_dataset_contract(self) -> None:
        if not self._data and not _config_bool(
            self.dataset_cfg.get("allow_empty_payload", False),
            name="dataset.allow_empty_payload",
        ):
            raise DataIngestionContractError("Dataset payload is empty", context={})

        if self.enforce_batch_alignment:
            assert_modalities_aligned(self._data, expected_modalities=self.expected_modalities)
            return

        expected_set = set(self.expected_modalities)
        actual_set = set(self._data.keys())
        missing = expected_set - actual_set
        if missing:
            raise DataIngestionContractError(
                "Dataset is missing required modalities",
                context={"missing": sorted(missing), "actual": sorted(actual_set)},
            )
        if len(set(self.lengths.values())) > 1 and not self.allow_unaligned_batches:
            raise DataIngestionContractError(
                "Dataset contains unaligned modalities and allow_unaligned_batches is disabled",
                context={"lengths": self.lengths},
            )

    # ------------------------------------------------------------------
    # Core Python protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        if self.total == 0:
            return 0
        return self.total // self.batch_size if self.drop_remainder else ceil(self.total / self.batch_size)

    def __iter__(self) -> "MultimodalDataset":
        self.index = 0
        indices = self._build_indices()
        self._batch_iter = chunk_sequence(indices, self.batch_size, drop_remainder=self.drop_remainder)
        self._epoch += 1
        return self

    def __next__(self) -> dict[str, tuple[Mapping[str, Any], ...]]:
        if self._batch_iter is None:
            self.__iter__()
        assert self._batch_iter is not None

        index_chunk = next(self._batch_iter)  # raises StopIteration naturally
        start = self.index
        self.index += len(index_chunk)
        batch = self._make_batch(index_chunk, batch_index=(start // self.batch_size))
        return batch.to_dict(include_metadata=self.include_batch_metadata)

    def __getitem__(self, item: int | slice | Sequence[int]) -> dict[str, Any]:
        if isinstance(item, int):
            idx = self._normalise_index(item)
            return {modality: rows[idx] for modality, rows in self._data.items()}
        if isinstance(item, slice):
            indices = range(self.total)[item]
            return self._make_batch(tuple(indices), batch_index=None).to_dict(include_metadata=self.include_batch_metadata)
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            indices = tuple(self._normalise_index(int(idx)) for idx in item)
            return self._make_batch(indices, batch_index=None).to_dict(include_metadata=self.include_batch_metadata)
        raise DataValidationError(
            "Dataset index must be int, slice, or sequence of ints",
            context={"type": type(item).__name__},
        )

    def _normalise_index(self, idx: int) -> int:
        if idx < 0:
            idx = self.total + idx
        if idx < 0 or idx >= self.total:
            raise DataValidationError(
                "Dataset index out of range",
                context={"index": idx, "record_count": self.total},
            )
        return idx

    # ------------------------------------------------------------------
    # Batch and indexing utilities
    # ------------------------------------------------------------------
    def _build_indices(self) -> Sequence[int]:
        indices: list[int] = list(range(self.total))
        if self.shuffle:
            seed = self.seed
            if seed is not None and self.reshuffle_each_epoch:
                seed += self._epoch
            rng = random.Random(seed)
            rng.shuffle(indices)
        return indices

    def _make_batch(self, indices: Sequence[int], *, batch_index: int | None) -> MultimodalBatch:
        data = {
            modality: tuple(rows[idx] for idx in indices)
            for modality, rows in self._data.items()
        }
        if self.validate_batches and self.validator is not None and data:
            self.validator.enforce_multimodal_alignment(data)

        metadata = {
            "batch_index": batch_index,
            "size": len(indices),
            "indices": list(indices),
            "epoch": self._epoch,
            "modalities": list(self.modalities),
            "drop_remainder": self.drop_remainder,
            "shuffle": self.shuffle,
        }
        return MultimodalBatch(data=data, metadata=metadata)

    def iter_batches(self, *, include_metadata: bool | None = None) -> Iterator[dict[str, Any]]:
        include_meta = self.include_batch_metadata if include_metadata is None else include_metadata
        for batch_index, index_chunk in enumerate(chunk_sequence(self._build_indices(), self.batch_size, drop_remainder=self.drop_remainder)):
            batch = self._make_batch(index_chunk, batch_index=batch_index)
            yield batch.to_dict(include_metadata=include_meta)

    # ------------------------------------------------------------------
    # Governance, quality, and inspection
    # ------------------------------------------------------------------
    @property
    def payload(self) -> dict[str, tuple[Mapping[str, Any], ...]]:
        return dict(self._data)

    def as_payload(self, *, copy: bool = False) -> dict[str, Sequence[Mapping[str, Any]]]:
        if not copy:
            return self.payload # type: ignore
        return {modality: tuple(dict(row) for row in rows) for modality, rows in self._data.items()}

    def validate(self) -> None:
        if self.validator is not None:
            self.validator.enforce_multimodal_alignment(self._data)
        elif self.enforce_batch_alignment:
            assert_modalities_aligned(self._data, expected_modalities=self.expected_modalities)

    def quality_report(self) -> dict[str, Any]:
        if self.validator is not None:
            self._last_quality_report = self.validator.quality_gate(self._data)
            return self._last_quality_report

        self.validate()
        if self.total == 0:
            self._last_quality_report = {
                "total_records": 0,
                "total_cells": 0,
                "total_nulls": 0,
                "null_ratio": 0.0,
                "modalities": list(self.modalities),
            }
        else:
            nulls = audit_nulls(self._data)
            self._last_quality_report = {
                **nulls,
                "modalities": sorted(self._data.keys()),
            }
        return self._last_quality_report

    def stats(self) -> dict[str, Any]:
        if not self._data or self.total == 0:
            return {}
        return compute_modality_stats(self._data)

    def summary(self, *, include_quality: bool = False, include_stats: bool = True) -> MultimodalDatasetSummary:
        quality = self.quality_report() if include_quality else self._last_quality_report
        stats = self.stats() if include_stats else None
        return MultimodalDatasetSummary(
            modalities=self.modalities,
            record_count=self.total,
            batch_size=self.batch_size,
            batch_count=len(self),
            drop_remainder=self.drop_remainder,
            shuffle=self.shuffle,
            lengths=dict(self.lengths),
            quality=quality,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Dataset manipulation
    # ------------------------------------------------------------------
    def subset(self, indices: Sequence[int], *, validate_on_init: bool | None = None) -> "MultimodalDataset":
        normalised = tuple(self._normalise_index(int(idx)) for idx in indices)
        payload = {
            modality: tuple(rows[idx] for idx in normalised)
            for modality, rows in self._data.items()
        }
        return self._from_payload(payload, validate_on_init=validate_on_init)

    def take(self, count: int) -> "MultimodalDataset":
        take_count = _config_int(count, name="count", minimum=0)
        return self.subset(tuple(range(min(take_count, self.total))), validate_on_init=False)

    def split(
        self,
        ratios: Sequence[float] = (0.8, 0.1, 0.1),
        *,
        names: Sequence[str] = ("train", "validation", "test"),
        shuffle: bool | None = None,
        seed: int | None = None,
    ) -> dict[str, "MultimodalDataset"]:
        if len(ratios) != len(names):
            raise DataValidationError(
                "split ratios and names must have the same length",
                context={"ratios": list(ratios), "names": list(names)},
            )
        if not ratios or any(r < 0 for r in ratios) or sum(ratios) <= 0:
            raise DataValidationError("split ratios must be non-negative and sum to > 0", context={"ratios": list(ratios)})

        indices = list(range(self.total))
        if self.shuffle if shuffle is None else shuffle:
            rng = random.Random(self.seed if seed is None else seed)
            rng.shuffle(indices)

        total_ratio = float(sum(ratios))
        sizes = [int(self.total * (ratio / total_ratio)) for ratio in ratios]
        sizes[-1] += self.total - sum(sizes)

        splits: dict[str, MultimodalDataset] = {}
        cursor = 0
        for name, size in zip(names, sizes):
            chunk = tuple(indices[cursor: cursor + size])
            splits[str(name)] = self.subset(chunk, validate_on_init=False)
            cursor += size
        return splits

    def _from_payload(
        self,
        payload: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        validate_on_init: bool | None,
    ) -> "MultimodalDataset":
        extras = {m: rows for m, rows in payload.items() if m not in self.DEFAULT_MODALITIES}
        return MultimodalDataset(
            payload.get("vision", ()),
            payload.get("text", ()),
            payload.get("audio", ()),
            batch_size=self.batch_size,
            validator=self.validator,
            extra_modalities=extras,
            drop_remainder=self.drop_remainder,
            shuffle=self.shuffle,
            seed=self.seed,
            validate_on_init=self.validate_on_init if validate_on_init is None else validate_on_init,
            validate_batches=self.validate_batches,
            run_quality_gate_on_init=False,
            sanitize_records=False,
            copy_records=True,
            include_batch_metadata=self.include_batch_metadata,
        )


# =============================================================================
# Config helpers
# =============================================================================

def _mapping_section(value: Any, section_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DataConfigError(
            "Config section must be a mapping",
            context={"section": section_name, "type": type(value).__name__},
        )
    return value


def _config_int(value: Any, *, name: str, minimum: int = 1) -> int:
    parsed = safe_int(value, field_name=name, modality="config", min_value=minimum)
    if parsed is None:
        raise DataConfigError(f"{name} cannot be null", context={"name": name})
    return parsed


def _optional_config_int(value: Any, *, name: str, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _config_int(value, name=name, minimum=minimum)


def _config_bool(value: Any, *, name: str) -> bool:
    parsed = safe_bool(value, field_name=name, modality="config")
    if parsed is None:
        raise DataConfigError(f"{name} cannot be null", context={"name": name})
    return parsed


def _expected_modalities(dataset_cfg: Mapping[str, Any], actual_modalities: Sequence[str]) -> tuple[str, ...]:
    configured = dataset_cfg.get("required_modalities")
    if configured is None:
        return tuple(actual_modalities)
    if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
        raise DataConfigError(
            "dataset.required_modalities must be a sequence of modality names",
            context={"type": type(configured).__name__},
        )
    return tuple(str(modality) for modality in configured)


if __name__ == "__main__":
    from .governance import DatasetField, DatasetSchema
    print("\n=== Running Multimodal Dataset ===\n")
    printer.status("TEST", "Multimodal Dataset initialized", "info")

    schemas = (
        DatasetSchema("vision_schema", "1.0", "vision", (DatasetField("id", str), DatasetField("image_tokens", list, max_items=8),)),
        DatasetSchema("text_schema", "1.0", "text", (DatasetField("id", str), DatasetField("text", str, max_items=128),)),
        DatasetSchema("audio_schema", "1.0", "audio", (DatasetField("id", str), DatasetField("audio_features", list, max_items=8),)),
    )
    validator = DatasetValidator(schemas)
    vision = [{"id": "r1", "image_tokens": [1, 2]}, {"id": "r2", "image_tokens": [3]}, {"id": "r3", "image_tokens": [4]}]
    text = [{"id": "r1", "text": "hello"}, {"id": "r2", "text": "world"}, {"id": "r3", "text": "test"}]
    audio = [{"id": "r1", "audio_features": [0.1]}, {"id": "r2", "audio_features": [0.2]}, {"id": "r3", "audio_features": [0.3]}]

    dataset = MultimodalDataset(vision, text, audio, batch_size=2, validator=validator, shuffle=False)
    assert dataset.total == 3
    assert len(dataset) == 2
    first = next(iter(dataset))
    assert len(first["vision"]) == 2
    assert dataset[0]["text"]["id"] == "r1"
    assert dataset.quality_report()["total_records"] == 9
    splits = dataset.split((0.67, 0.33), names=("train", "validation"), shuffle=False)
    assert set(splits) == {"train", "validation"}

    try:
        MultimodalDataset(vision, text[:-1], audio, batch_size=2, validator=validator)
    except DataIngestionContractError:
        printer.status("TEST", "Expected alignment failure captured", "warning")
    else:
        raise AssertionError("Expected DataIngestionContractError for unaligned modalities")

    print("\n=== Test ran successfully ===\n")
