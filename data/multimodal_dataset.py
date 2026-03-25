from __future__ import annotations

from typing import Any, Mapping, Sequence

from data.governance import DatasetValidator
from data.utils.config_loader import get_config_section
from data.utils.data_error import DataIngestionContractError


class MultimodalDataset:
    def __init__(
        self,
        vision_data: Sequence[Mapping[str, Any]],
        text_data: Sequence[Mapping[str, Any]],
        audio_data: Sequence[Mapping[str, Any]],
        batch_size: int = 8,
        validator: DatasetValidator | None = None,
    ):
        self.vision = vision_data
        self.text = text_data
        self.audio = audio_data
        self.batch_size = batch_size
        self.index = 0
        self.total = len(vision_data)
        self.validator = validator

        dataset_cfg = get_config_section("dataset")
        ingestion_cfg = dataset_cfg.get("ingestion", {})
        min_batch_size = int(ingestion_cfg.get("min_batch_size", 1))

        if self.batch_size < min_batch_size:
            raise DataIngestionContractError(
                f"batch_size must be >= {min_batch_size}",
                context={"batch_size": self.batch_size, "min_batch_size": min_batch_size},
            )

        if bool(ingestion_cfg.get("enforce_batch_alignment", True)) and not (
            len(self.vision) == len(self.text) == len(self.audio)
        ):
            raise DataIngestionContractError(
                "Ingestion contract failed: vision, text, and audio must have aligned lengths",
                context={
                    "vision_size": len(self.vision),
                    "text_size": len(self.text),
                    "audio_size": len(self.audio),
                },
            )

        if self.validator:
            self.validator.enforce_multimodal_alignment(
                {
                    "vision": self.vision,
                    "text": self.text,
                    "audio": self.audio,
                }
            )

    def __len__(self):
        return self.total // self.batch_size

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.total:
            raise StopIteration
        start, end = self.index, self.index + self.batch_size
        self.index += self.batch_size

        batch = {
            "vision": self.vision[start:end],
            "text": self.text[start:end],
            "audio": self.audio[start:end],
        }

        if self.validator:
            self.validator.enforce_multimodal_alignment(batch)

        return batch
