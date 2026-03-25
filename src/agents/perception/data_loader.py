import yaml
import torch
import librosa
import numpy as np

from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from src.agents.perception.modules.tokenizer import Tokenizer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("DataLoader")
printer = PrettyPrinter

class MultiModalDataset(Dataset):
    def __init__(self, config: Optional[Dict] = None, mode: str = "train"):
        self.config = load_global_config() or config
        self.loader_config = get_config_section('data_loader')
        self.mode = mode
        self.modalities = self._detect_modalities()

        self.tokenizer = Tokenizer() if "text" in self.modalities else None
        self.training_ext = TrainingExtensions(self.config, self.modalities)
        self.inference_opt = InferenceOptimizer(self.config)

        self.transforms = (
            self._create_inference_transforms()
            if self.mode == "inference"
            else self._create_transforms()
        )

    def _detect_modalities(self) -> List[str]:
        modalities = []
        if "vision_encoder" in self.config:
            modalities.append("vision")
        if "audio_encoder" in self.config:
            modalities.append("audio")
        if "text_encoder" in self.config:
            modalities.append("text")
        return modalities

    def _create_transforms(self) -> Dict[str, transforms.Compose]:
        transforms_dict: Dict[str, transforms.Compose] = {}

        if "vision" in self.modalities:
            vision_cfg = self.config["vision_encoder"]
            transforms_dict["vision"] = transforms.Compose(
                [
                    transforms.Resize(vision_cfg["img_size"]),
                    transforms.CenterCrop(vision_cfg["img_size"]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        if "audio" in self.modalities:
            audio_cfg = self.config["audio_encoder"]
            self.audio_length = audio_cfg["audio_length"]
            self.sample_rate = self.config["mfcc"]["sample_rate"]

        return transforms_dict

    def _create_inference_transforms(self) -> Dict[str, transforms.Compose]:
        transforms_dict: Dict[str, transforms.Compose] = {}
        if "vision" in self.modalities:
            vision_cfg = self.config["vision_encoder"]
            transforms_dict["vision"] = transforms.Compose(
                [
                    transforms.Resize(vision_cfg["img_size"]),
                    transforms.CenterCrop(vision_cfg["img_size"]),
                    transforms.ToTensor(),
                ]
            )

        if "audio" in self.modalities:
            audio_cfg = self.config["audio_encoder"]
            self.audio_length = audio_cfg["audio_length"]
            self.sample_rate = self.config["mfcc"]["sample_rate"]

        return transforms_dict

    def _process_vision(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = self.transforms["vision"](image)

        if (
            self.mode == "train"
            and "vision" in self.training_ext.augmentation
            and self.training_ext.augmentation["vision"] is not None
        ):
            image = self.training_ext.augmentation["vision"](image)
        return image

    def _process_audio(self, audio_path: str) -> torch.Tensor:
        waveform, _ = librosa.load(
            audio_path,
            sr=self.sample_rate,
            duration=self.audio_length / self.sample_rate,
        )

        if len(waveform) > self.audio_length:
            waveform = waveform[: self.audio_length]
        elif len(waveform) < self.audio_length:
            waveform = np.pad(waveform, (0, self.audio_length - len(waveform)))

        tensor = torch.from_numpy(waveform).float()

        if (
            self.mode == "train"
            and "audio" in self.training_ext.augmentation
            and self.training_ext.augmentation["audio"] is not None
        ):
            tensor = self.training_ext.augmentation["audio"](tensor)
        return tensor

    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Text tokenizer is not initialized for this dataset")
        return self.tokenizer(text)

    def __getitem__(self, idx: int) -> Dict:
        raise NotImplementedError("Subclasses must implement __getitem__")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses must implement __len__")


class MultiModalDataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True,
        )

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        collated: Dict = {"metadata": []}

        for modality in ["vision", "audio", "text"]:
            if modality in batch[0]:
                items = [item[modality] for item in batch]

                if modality == "vision":
                    collated["vision"] = torch.stack(items)
                elif modality == "audio":
                    collated["audio"] = torch.nn.utils.rnn.pad_sequence(
                        items, batch_first=True
                    )
                elif modality == "text":
                    collated["text"] = {
                        "input_ids": torch.stack([x["input_ids"] for x in items]),
                        "attention_mask": torch.stack(
                            [x["attention_mask"] for x in items]
                        ),
                    }

        if "metadata" in batch[0]:
            collated["metadata"] = [item.get("metadata", {}) for item in batch]

        return collated

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


class TrainingExtensions:
    def __init__(self, config: Dict, modalities: List[str]):
        self.config = config
        self.modalities = modalities
        self.augmentation = self._create_augmentations()

    def _create_augmentations(self) -> Dict:
        aug_config = self.config.get("training", {}).get("augmentations", {})
        aug_dict: Dict = {}

        if "vision" in self.modalities:
            vision_aug = []
            if aug_config.get("random_crop", False):
                vision_aug.append(
                    transforms.RandomResizedCrop(self.config["vision_encoder"]["img_size"])
                )
            if aug_config.get("color_jitter", False):
                vision_aug.append(
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                    )
                )
            if aug_config.get("horizontal_flip", False):
                vision_aug.append(transforms.RandomHorizontalFlip())

            aug_dict["vision"] = transforms.Compose(vision_aug) if vision_aug else None

        if "audio" in self.modalities and aug_config.get("audio_noise", False):
            aug_dict["audio"] = self._add_audio_noise

        return aug_dict

    @staticmethod
    def _add_audio_noise(waveform: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(waveform) * 0.005
        return waveform + noise


class InferenceOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.cache = {}
        inference_cfg = self.config.get("inference", {})
        self.batch_sizes = {
            "vision": inference_cfg.get("vision_batch", 8),
            "audio": inference_cfg.get("audio_batch", 16),
            "text": inference_cfg.get("text_batch", 32),
        }

    def dynamic_batching(self, samples: List[Dict]) -> Dict[str, List]:
        batched: Dict[str, List] = {"vision": [], "audio": [], "text": []}

        for sample in samples:
            for modality, value in sample.items():
                if modality in batched:
                    batched[modality].append(value)

        result: Dict[str, List] = {}
        for mod, values in batched.items():
            if not values:
                continue
            size = self.batch_sizes[mod]
            result[mod] = [values[i : i + size] for i in range(0, len(values), size)]

        return result


class TrainingManager:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or load_global_config()
        self._init_components()

    def _init_components(self):
        from src.agents.perception.encoders.vision_encoder import VisionEncoder
        from src.agents.perception.encoders.audio_encoder import AudioEncoder
        from src.agents.perception.encoders.text_encoder import TextEncoder

        self.models = {
            "vision": VisionEncoder(self.config),
            "audio": AudioEncoder(self.config),
            "text": TextEncoder(self.config),
        }

    def train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}

        if "vision" in batch:
            outputs["vision"] = self.models["vision"](batch["vision"])
        if "audio" in batch:
            outputs["audio"] = self.models["audio"](batch["audio"])
        if "text" in batch:
            outputs["text"] = self.models["text"](
                batch["text"]["input_ids"],
                attention_mask=batch["text"].get("attention_mask"),
            )

        fused_output = self.fuse_outputs(outputs)
        return {
            "fused_output": fused_output,
            "loss": self.calculate_loss(fused_output, batch.get("metadata", [])),
        }

    def fuse_outputs(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not outputs:
            raise ValueError("No modality outputs found for fusion")

        fusion_method = self.config.get("multimodal", {}).get("fusion_method", "concat")

        tensors = []
        for value in outputs.values():
            if value.dim() == 3:
                tensors.append(value.mean(dim=1))
            else:
                tensors.append(value)

        if fusion_method == "concat":
            return torch.cat(tensors, dim=-1)
        if fusion_method == "mean":
            return torch.mean(torch.stack(tensors), dim=0)
        raise ValueError(f"Unknown fusion method: {fusion_method}")

    @staticmethod
    def calculate_loss(fused_output: torch.Tensor, metadata: List[Dict]) -> torch.Tensor:
        if not metadata:
            return torch.tensor(0.0, device=fused_output.device)

        labels = [m.get("label") for m in metadata if isinstance(m, dict) and "label" in m]
        if not labels:
            return torch.tensor(0.0, device=fused_output.device)

        target = torch.tensor(labels, device=fused_output.device, dtype=fused_output.dtype)
        pred = fused_output.mean(dim=-1)
        return torch.nn.functional.mse_loss(pred, target)


class InferenceEngine:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or load_global_config()
        self.models = TrainingManager(self.config).models
        self.dataset = MultiModalDataset(self.config, mode="inference")

    def process_single(self, sample: Dict) -> Dict[str, torch.Tensor]:
        batched = self.dataset.inference_opt.dynamic_batching([sample])
        return self.process_batch(batched)

    def process_batch(self, batch: Dict[str, List]) -> Dict[str, torch.Tensor]:
        results: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for modality, chunks in batch.items():
                chunk_results = []
                for chunk in chunks:
                    if modality == "vision":
                        tensor = torch.stack(chunk)
                        chunk_results.append(self.models["vision"](tensor))
                    elif modality == "audio":
                        tensor = torch.stack(chunk)
                        chunk_results.append(self.models["audio"](tensor))
                    elif modality == "text":
                        input_ids = torch.stack([x["input_ids"] for x in chunk])
                        attention_mask = torch.stack([x["attention_mask"] for x in chunk])
                        chunk_results.append(
                            self.models["text"](input_ids, attention_mask=attention_mask)
                        )

                results[modality] = torch.cat(chunk_results, dim=0)

        return {"fused_output": self.fuse_results(results), "modal_outputs": results}

    def fuse_results(self, results: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not results:
            raise ValueError("No modality results were produced")

        pooled = []
        for value in results.values():
            if value.dim() == 3:
                pooled.append(value.mean(dim=1))
            else:
                pooled.append(value)

        fusion_method = self.config.get("multimodal", {}).get("fusion_method", "concat")
        if fusion_method == "mean":
            return torch.mean(torch.stack(pooled), dim=0)
        return torch.cat(pooled, dim=-1)


if __name__ == "__main__":
    class ExampleMultiModalDataset(MultiModalDataset):
        def __init__(self, data_root: str, config: Dict):
            super().__init__(config)
            self.data_root = Path(data_root)
            self.samples = self._load_samples()

        @staticmethod
        def _load_samples() -> List[Dict]:
            return [
                {
                    "vision": "path/to/image.jpg",
                    "audio": "path/to/audio.wav",
                    "text": "example text",
                    "metadata": {"label": 0},
                }
            ]

        def __getitem__(self, idx: int) -> Dict:
            sample = self.samples[idx]
            item = {}

            if "vision" in sample:
                item["vision"] = self._process_vision(sample["vision"])
            if "audio" in sample:
                item["audio"] = self._process_audio(sample["audio"])
            if "text" in sample:
                item["text"] = self._process_text(sample["text"])
            if "metadata" in sample:
                item["metadata"] = sample["metadata"]
            return item

        def __len__(self) -> int:
            return len(self.samples)

    print("\n=== Running DataLoader smoke init ===\n")
    cfg = load_global_config()
    dataset = ExampleMultiModalDataset(data_root="data", config=cfg)
    loader = MultiModalDataLoader(dataset, batch_size=32, shuffle=True)
    print(loader)
    print(dataset)
    print("\n=== Successfully initialized DataLoader components ===\n")
