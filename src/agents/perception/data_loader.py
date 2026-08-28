"""
Multi-modal data pipeline for the SLAI perception subsystem.
 
Key components:
1. MultiModalDataset    - modality detection, transform setup, per-modality
                           preprocessing (vision/audio/text) with structured
                           error handling for missing/corrupt files.
2. MultiModalDataLoader - config-driven torch DataLoader wrapper with a
                           modality-aware collate function.
3. TrainingExtensions   - per-modality data augmentation.
4. InferenceOptimizer   - dynamic per-modality batching plus optional
                           PerceptionMemory-backed result caching.
5. TrainingManager      - owns the per-modality encoders and runs a single
                           fused training step.
6. InferenceEngine      - runs the same encoders + fusion for inference.
 
``TrainingManager.fuse_outputs`` and ``InferenceEngine.fuse_results`` share
a single ``pool_and_fuse_modalities`` implementation so the train/inference
fusion paths cannot silently diverge.
"""

import hashlib
import torch
import librosa
import numpy as np
 
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

from .utils.config_loader import load_global_config, get_config_section
from .utils.perception_errors import *
from .utils.perception_helpers import *
from .modules.tokenizer import Tokenizer
from .perception_memory import PerceptionMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("DataLoader")
printer = PrettyPrinter()


_COMPONENT = "data_loader"
VALID_DATASET_MODES: Tuple[str, ...] = ("train", "val", "test", "inference")
 
 
# ---------------------------------------------------------------------------
# Shared fusion helper (single implementation for train + inference paths)
# ---------------------------------------------------------------------------
def pool_and_fuse_modalities(
    modal_outputs: Mapping[str, torch.Tensor],
    fusion_method: str,
    *,
    pooling_strategy: str = "mean",
) -> torch.Tensor:
    """Pool each modality's encoder output to ``(B, D)`` and fuse across modalities.
 
    Sequence outputs ``(B, L, D)`` are pooled via ``perception_helpers.pool_encoded``
    (default ``"mean"`` reproduces the original ``value.mean(dim=1)`` behavior);
    already-pooled ``(B, D)`` outputs pass through unchanged. Batch sizes are
    validated across modalities before fusing so a mismatch raises a clear
    ``PerceptionDimensionError`` instead of an opaque ``torch.cat``/``stack`` failure.
    """
    ensure_non_empty(dict(modal_outputs), "modal_outputs", component=_COMPONENT)
    ensure_one_of(fusion_method, ("concat", "mean"), "fusion_method", component=_COMPONENT)
 
    names = list(modal_outputs.keys())
    pooled = []
    for name in names:
        value = require_tensor(modal_outputs[name], f"modal_outputs[{name}]", component=_COMPONENT)
        pooled.append(pool_encoded(value, strategy=pooling_strategy) if value.dim() == 3 else value)
 
    ensure_same_batch(*pooled, names=names, component=_COMPONENT)
 
    if fusion_method == "concat":
        return torch.cat(pooled, dim=-1)
    return torch.mean(torch.stack(pooled), dim=0)
 
 
class MultiModalDataset(Dataset):
    """Base multi-modal dataset: modality detection, transform setup, and
    per-modality preprocessing. Subclasses supply the sample index and
    implement ``__getitem__``/``__len__``.
    """
 
    def __init__(self, config: Optional[Dict] = None, mode: str = "train"):
        # An explicit override is honored; otherwise fall back to the global config.
        self.config = config if config is not None else load_global_config()
        ensure_not_none(self.config, "config", component=_COMPONENT)
        ensure_one_of(mode, VALID_DATASET_MODES, "mode", component=_COMPONENT)
 
        self.loader_config = get_config_section('data_loader')
        self.mode = mode
        self.modalities = self._detect_modalities()
        ensure_non_empty(self.modalities, "modalities", component=_COMPONENT)
 
        self.tokenizer = Tokenizer() if "text" in self.modalities else None
        self.training_ext = TrainingExtensions(self.config, self.modalities)
        self.inference_opt = InferenceOptimizer(self.config)
 
        self.transforms = (
            self._create_inference_transforms()
            if self.mode == "inference"
            else self._create_transforms()
        )
 
    def _detect_modalities(self) -> List[str]:
        """Detect active modalities. An explicit ``data_loader.modalities`` list
        overrides auto-detection from the ``<modality>_encoder`` config sections."""
        override = self.loader_config.get('modalities')
        if override:
            return [normalize_modality_name(m) for m in override]
        return [modality for modality in VALID_MODALITIES if f"{modality}_encoder" in self.config]
 
    def _require_config_section(self, section_name: str, required_keys: Sequence[str]) -> Dict[str, Any]:
        """Validate that ``section_name`` exists in config and carries the
        given keys, raising a structured error instead of a bare KeyError."""
        ensure_keys(self.config, [section_name], name="config", component=_COMPONENT)
        return ensure_keys(self.config[section_name], required_keys, name=section_name, component=_COMPONENT)
 
    def _configure_audio(self) -> None:
        audio_cfg = self._require_config_section("audio_encoder", ["audio_length"])
        mfcc_cfg = ensure_keys(self.config.get("mfcc", {}), ["sample_rate"], name="mfcc", component=_COMPONENT)
        self.audio_length = audio_cfg["audio_length"]
        self.sample_rate = mfcc_cfg["sample_rate"]
 
    def _create_transforms(self) -> Dict[str, transforms.Compose]:
        transforms_dict: Dict[str, transforms.Compose] = {}
        if "vision" in self.modalities:
            vision_cfg = self._require_config_section("vision_encoder", ["img_size"])
            transforms_dict["vision"] = transforms.Compose([
                transforms.Resize(vision_cfg["img_size"]),
                transforms.CenterCrop(vision_cfg["img_size"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        if "audio" in self.modalities:
            self._configure_audio()
        return transforms_dict
 
    def _create_inference_transforms(self) -> Dict[str, transforms.Compose]:
        transforms_dict: Dict[str, transforms.Compose] = {}
        if "vision" in self.modalities:
            vision_cfg = self._require_config_section("vision_encoder", ["img_size"])
            transforms_dict["vision"] = transforms.Compose([
                transforms.Resize(vision_cfg["img_size"]),
                transforms.CenterCrop(vision_cfg["img_size"]),
                transforms.ToTensor(),
            ])
        if "audio" in self.modalities:
            self._configure_audio()
        return transforms_dict
 
    def _process_vision(self, image_path: Union[str, Path]) -> torch.Tensor:
        path = Path(image_path)
        ensure(
            path.exists(), f"Vision file not found: {path}",
            exc_type=ModalityInputError, component=_COMPONENT, details={"path": str(path)},
        )
        try:
            image = Image.open(path).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise wrap_exception(
                exc, ModalityDecodingError, f"Failed to decode image: {path}",
                component=_COMPONENT, details={"path": str(path)},
            )
 
        image = self.transforms["vision"](image)
 
        if (self.mode == "train" and "vision" in self.training_ext.augmentation
                and self.training_ext.augmentation["vision"] is not None):
            image = self.training_ext.augmentation["vision"](image)
        return image # type: ignore
 
    def _process_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        path = Path(audio_path)
        ensure(
            path.exists(), f"Audio file not found: {path}",
            exc_type=ModalityInputError, component=_COMPONENT, details={"path": str(path)},
        )
        try:
            waveform, _ = librosa.load(
                str(path), sr=self.sample_rate, duration=self.audio_length / self.sample_rate,
            )
        except Exception as exc:  # librosa surfaces varied backend-specific errors
            raise wrap_exception(
                exc, ModalityDecodingError, f"Failed to decode audio: {path}",
                component=_COMPONENT, details={"path": str(path)},
            )
 
        if len(waveform) > self.audio_length:
            waveform = waveform[: self.audio_length]
        elif len(waveform) < self.audio_length:
            waveform = np.pad(waveform, (0, self.audio_length - len(waveform)))
 
        tensor = torch.from_numpy(waveform).float()
 
        if (self.mode == "train" and "audio" in self.training_ext.augmentation
                and self.training_ext.augmentation["audio"] is not None):
            tensor = self.training_ext.augmentation["audio"](tensor)
        return tensor
 
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise PerceptionStateError(
                "Text tokenizer is not initialized for this dataset.",
                component=_COMPONENT,
                remediation="Configure 'text_encoder' or include 'text' in data_loader.modalities.",
            )
        return self.tokenizer(text)
 
    def __getitem__(self, idx: int) -> Dict:
        raise NotImplementedError("Subclasses must implement __getitem__")
 
    def __len__(self) -> int:
        raise NotImplementedError("Subclasses must implement __len__")
 
    def __repr__(self) -> str:
        return f"MultiModalDataset(mode={self.mode!r}, modalities={self.modalities})"
 
 
class MultiModalDataLoader:
    """Thin wrapper around ``torch.utils.data.DataLoader`` with a modality-aware
    collate function. Batch/shuffle/worker settings default to the
    ``data_loader`` config section but can always be overridden per call."""
 
    def __init__(self, dataset: Dataset, batch_size: Optional[int] = None, shuffle: Optional[bool] = None):
        loader_config = get_config_section('data_loader')
        resolved_batch_size = batch_size if batch_size is not None else loader_config.get('default_batch_size', 32)
        resolved_shuffle = shuffle if shuffle is not None else loader_config.get('default_shuffle', True)
        ensure_instance(resolved_batch_size, int, "batch_size", component=_COMPONENT)
        ensure_in_range(resolved_batch_size, "batch_size", minimum=1, component=_COMPONENT)
 
        num_workers = loader_config.get('num_workers', 4)
        self.dataloader = DataLoader(
            dataset,
            batch_size=resolved_batch_size,
            shuffle=resolved_shuffle,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=loader_config.get('pin_memory', True),
            drop_last=loader_config.get('drop_last', False),
            # persistent_workers/prefetch_factor are only valid when num_workers > 0.
            persistent_workers=loader_config.get('persistent_workers', False) and num_workers > 0,
            prefetch_factor=loader_config.get('prefetch_factor', 2) if num_workers > 0 else None,
        )
        self.batch_size = resolved_batch_size
 
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        ensure_non_empty(batch, "batch", component=_COMPONENT)
        collated: Dict = {"metadata": []}
 
        for modality in VALID_MODALITIES:
            if modality in batch[0]:
                items = [item[modality] for item in batch]
                try:
                    if modality == "vision":
                        collated["vision"] = torch.stack(items)
                    elif modality == "audio":
                        collated["audio"] = torch.nn.utils.rnn.pad_sequence(items, batch_first=True)
                    elif modality == "text":
                        collated["text"] = {
                            "input_ids": torch.stack([x["input_ids"] for x in items]),
                            "attention_mask": torch.stack([x["attention_mask"] for x in items]),
                        }
                except RuntimeError as exc:
                    raise wrap_exception(
                        exc, PerceptionShapeError,
                        f"Failed to collate modality '{modality}': shapes are inconsistent across the batch.",
                        component=_COMPONENT, details={"modality": modality},
                    )
 
        if "metadata" in batch[0]:
            collated["metadata"] = [item.get("metadata", {}) for item in batch]
 
        return collated
 
    def __iter__(self):
        return iter(self.dataloader)
 
    def __len__(self):
        return len(self.dataloader)
 
    def __repr__(self) -> str:
        return f"MultiModalDataLoader(batch_size={self.batch_size}, num_batches={len(self)})"
 
 
class TrainingExtensions:
    """Per-modality data augmentation, built once from ``training.augmentations``."""
 
    def __init__(self, config: Dict, modalities: List[str]):
        self.config = config
        self.modalities = modalities
        self.augmentation = self._create_augmentations()
 
    def _create_augmentations(self) -> Dict:
        training_cfg = self.config.get("training")
        aug_config = training_cfg.get("augmentations", {}) if isinstance(training_cfg, dict) else {}
        aug_dict: Dict = {}
    
        if "vision" in self.modalities:
            vision_aug = []
            if aug_config.get("random_crop", False):
                vision_aug.append(
                    transforms.RandomResizedCrop(self.config["vision_encoder"]["img_size"])
                )
            if aug_config.get("color_jitter", False):
                vision_aug.append(
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
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
    """Batches inference samples per modality and, if enabled, caches fused
    results in a :class:`PerceptionMemory` instance keyed by sample content
    (path-based samples only; raw-tensor samples are never fingerprinted)."""
 
    def __init__(self, config: Dict):
        self.config = config
        loader_config = get_config_section('data_loader')
        inference_cfg = self.config.get("inference", {})
        self.batch_sizes = {
            "vision": inference_cfg.get("vision_batch", 8),
            "audio": inference_cfg.get("audio_batch", 16),
            "text": inference_cfg.get("text_batch", 32),
        }
 
        self.enable_cache = loader_config.get('enable_inference_cache', False)
        self.memory: Optional[PerceptionMemory] = None
        if self.enable_cache:
            self.memory = PerceptionMemory(enable_checkpointing=False, enable_cache=True)
            self.memory.max_cache_size = loader_config.get('inference_cache_max_items', 256)
 
    def dynamic_batching(self, samples: List[Dict]) -> Dict[str, List]:
        ensure_non_empty(samples, "samples", component=_COMPONENT)
        batched: Dict[str, List] = {modality: [] for modality in VALID_MODALITIES}
 
        for sample in samples:
            for modality, value in sample.items():
                if modality in batched:
                    batched[modality].append(value)
 
        result: Dict[str, List] = {}
        for mod, values in batched.items():
            if not values:
                continue
            size = self.batch_sizes[mod]
            result[mod] = [values[i: i + size] for i in range(0, len(values), size)]
 
        return result
 
    def cache_key(self, sample: Dict) -> Optional[str]:
        """Best-effort stable key for path-based samples; ``None`` for samples
        carrying raw tensors (unsafe/expensive to fingerprint reliably)."""
        fingerprint_parts = []
        for modality in VALID_MODALITIES:
            value = sample.get(modality)
            if value is None:
                continue
            if isinstance(value, (str, Path)):
                fingerprint_parts.append(f"{modality}:{value}")
            else:
                return None
        if not fingerprint_parts:
            return None
        return hashlib.sha256("|".join(sorted(fingerprint_parts)).encode()).hexdigest()
 
 
class TrainingManager:
    """Owns per-modality encoders and executes a single fused training step."""
 
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or load_global_config()
        self.pooling_strategy = get_config_section('data_loader').get('pooling_strategy', 'mean')
        self.fusion_method = self.config.get("multimodal", {}).get("fusion_method", "concat")
        ensure_one_of(self.fusion_method, ("concat", "mean"), "fusion_method", component=_COMPONENT)
        self._init_components()
 
    def _init_components(self):
        from .encoders.vision_encoder import VisionEncoder
        from .encoders.audio_encoder import AudioEncoder
        from .encoders.text_encoder import TextEncoder
 
        self.models = {
            "vision": VisionEncoder(),
            "audio": AudioEncoder(),
            "text": TextEncoder(),
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
        return pool_and_fuse_modalities(outputs, self.fusion_method, pooling_strategy=self.pooling_strategy)
 
    @staticmethod
    def calculate_loss(fused_output: torch.Tensor, metadata: List[Dict]) -> torch.Tensor:
        if not metadata:
            return differentiable_zero(fused_output)
 
        labels = [m.get("label") for m in metadata if isinstance(m, dict) and "label" in m]
        if not labels:
            return differentiable_zero(fused_output)
 
        target = torch.tensor(labels, device=fused_output.device, dtype=fused_output.dtype)
        pred = fused_output.mean(dim=-1)
        loss = torch.nn.functional.mse_loss(pred, target)
        return ensure_finite_tensor(loss, "loss", component=_COMPONENT)
 
 
class InferenceEngine:
    """Runs the trained encoders + fusion for single-sample or batched inference,
    with optional result caching via ``InferenceOptimizer``."""
 
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or load_global_config()
        self.models = TrainingManager(self.config).models
        self.dataset = MultiModalDataset(self.config, mode="inference")
        self.pooling_strategy = get_config_section('data_loader').get('pooling_strategy', 'mean')
        self.fusion_method = self.config.get("multimodal", {}).get("fusion_method", "concat")
        ensure_one_of(self.fusion_method, ("concat", "mean"), "fusion_method", component=_COMPONENT)
 
    def process_single(self, sample: Dict) -> Dict[str, torch.Tensor]:
        cache_key = None
        optimizer = self.dataset.inference_opt
        if optimizer.enable_cache:
            cache_key = optimizer.cache_key(sample)
            assert optimizer.memory is not None
            if cache_key and optimizer.memory.contains(cache_key):
                assert optimizer.memory is not None
                return optimizer.memory.retrieve(key=cache_key)
 
        batched = optimizer.dynamic_batching([sample])
        result = self.process_batch(batched)
 
        if cache_key:
            assert optimizer.memory is not None
            optimizer.memory.cache_item(result, key=cache_key, tags=["inference_result"])
        return result
 
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
 
                if chunk_results:
                    results[modality] = torch.cat(chunk_results, dim=0)
 
        return {"fused_output": self.fuse_results(results), "modal_outputs": results}
 
    def fuse_results(self, results: Dict[str, torch.Tensor]) -> torch.Tensor:
        return pool_and_fuse_modalities(results, self.fusion_method, pooling_strategy=self.pooling_strategy)
 
 
__all__ = [
    "pool_and_fuse_modalities",
    "MultiModalDataset",
    "MultiModalDataLoader",
    "TrainingExtensions",
    "InferenceOptimizer",
    "TrainingManager",
    "InferenceEngine",
]
 
if __name__ == "__main__":
    print("\n=== Running Data Loader ===\n")
    printer.status("TEST", "Data Loader initialized", "info")
 
    # --- Shared fusion helper: concat vs mean, with automatic pooling ---
    seq_out = {"vision": torch.randn(4, 5, 8), "text": torch.randn(4, 8)}
    fused_concat = pool_and_fuse_modalities(seq_out, "concat")
    assert tuple(fused_concat.shape) == (4, 16)
    fused_mean = pool_and_fuse_modalities(seq_out, "mean")
    assert tuple(fused_mean.shape) == (4, 8)
    printer.status("TEST", "pool_and_fuse_modalities OK", "success")
 
    # --- Batch-mismatch and bad-fusion-method validation ---
    try:
        pool_and_fuse_modalities({"vision": torch.randn(4, 8), "audio": torch.randn(3, 8)}, "concat")
        raise AssertionError("expected PerceptionDimensionError")
    except PerceptionDimensionError:
        printer.status("TEST", "batch-mismatch validation OK", "success")
 
    try:
        pool_and_fuse_modalities(seq_out, "bogus")
        raise AssertionError("expected InvalidPerceptionValueError")
    except InvalidPerceptionValueError:
        printer.status("TEST", "fusion_method validation OK", "success")
 
    # --- Differentiable zero-loss fallback (no labels present) ---
    zero_loss = TrainingManager.calculate_loss(fused_mean, [])
    assert float(zero_loss) == 0.0
    printer.status("TEST", "calculate_loss fallback OK", "success")
 
    # --- collate_fn ---
    batch = [
        {"vision": torch.randn(3, 8, 8), "metadata": {"label": 1}},
        {"vision": torch.randn(3, 8, 8), "metadata": {"label": 0}},
    ]
    collated = MultiModalDataLoader.collate_fn(batch)
    assert collated["vision"].shape[0] == 2 and len(collated["metadata"]) == 2
    printer.status("TEST", "collate_fn OK", "success")
 
    # --- InferenceOptimizer: dynamic batching + cache-key rules ---
    opt = InferenceOptimizer(load_global_config())
    dyn = opt.dynamic_batching([{"vision": torch.randn(3, 8, 8)}] * 3)
    assert "vision" in dyn
    assert opt.cache_key({"vision": "img.png", "audio": "a.wav"}) is not None
    assert opt.cache_key({"vision": torch.randn(3, 8, 8)}) is None
    printer.status("TEST", "InferenceOptimizer OK", "success")
 
    # --- MultiModalDataset: missing-file error handling ---
    dataset = MultiModalDataset(mode="train")
    try:
        dataset._process_vision("does/not/exist.jpg")
        raise AssertionError("expected ModalityInputError")
    except ModalityInputError:
        printer.status("TEST", "missing-file validation OK", "success")
 
    print(dataset)
    print("\n=== Test ran successfully ===\n")