from __future__ import annotations

from torch import random

__version__ = "2.3.0"

"""SLAI Perception Agent.

The PerceptionAgent is the single externally routable orchestration boundary for
SLAI perception.  Modality-specific computation remains inside the perception
subsystem:

    PerceptionAgent
        -> TextPerception / VisionPerception / AudioPerception
        -> ModalityRepresentation
        -> PerceptionFusion
        -> FusedRepresentation
        -> PerceptionObjectives / downstream task heads
        -> PerceptionTrainer

Ownership boundaries
--------------------
- Agent lifecycle, routing policy, runtime configuration, SharedMemory access,
  and durable checkpoint orchestration are owned here.
- Encoder/decoder architecture, tokenization, modality preprocessing, masking,
  representation construction, objective mathematics, and optimizer-step
  mechanics are owned by ``src.agents.perception``.
- ``perception/configs/perception_config.yaml`` is not imported or read here.
  Lower-level subsystem classes may consume it through their own config loader.
- ``perception/perception_memory.py`` is not imported or used here.  Agent-level
  transient state is coordinated through SharedMemory; durable model recovery is
  delegated to the central CheckpointManager.

The module deliberately uses explicit imports rather than wildcard imports so
its dependency direction remains visible and circular-import risk stays low.
"""

from dataclasses import dataclass
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

from .base.utils.main_config_loader import get_config_section
from .base_agent import BaseAgent
from .perception.modalities import *
from .perception.perception_contracts import *
from .perception.perception_fusion import PerceptionFusion
from .perception.perception_objectives import PerceptionObjectives
from .perception.perception_trainer import PerceptionTrainer
from .perception.utils.perception_errors import *
from .perception.utils.perception_helpers import *
from .perception.utils.taskheads import *
from checkpointing.checkpoint_manager import CheckpointManager # pyright: ignore[reportMissingImports]
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Perception Agent")
printer = PrettyPrinter()


_SHARED_STATE_SCHEMA = "slai.perception-agent.state.v1"
_CHECKPOINT_STATE_SCHEMA = "slai.perception-agent.checkpoint.v1"

_SUPPORTED_TASK_TYPES = frozenset({"pretrain", "finetune", "inference"})
_SUPPORTED_PRETRAIN_OBJECTIVES = frozenset(
    {
        "mlm",
        "mpm",
        "mam",
        "contrastive_text_image",
        "contrastive_text_audio",
        "contrastive_vision_audio",
        "temporal_vision",
        "temporal_audio",
    }
)


class PerceptionAgent(BaseAgent, nn.Module):
    """Single SLAI orchestration boundary for text, vision, and audio perception."""

    def __init__(self, shared_memory: Any, agent_factory: Any, config: Optional[Mapping[str, Any]] = None) -> None:
        # PyTorch must be initialized before this object receives any nn.Module
        # attributes. BaseAgent itself does not inherit nn.Module.
        nn.Module.__init__(self)
        BaseAgent.__init__(self,
                           shared_memory=shared_memory,
                           agent_factory=agent_factory,
                           config=config,
                           )

        self._init_agent_config(config)
        self._init_components()
        self._init_shared_memory_keys()
        self._init_checkpoint_manager()
        self._validate_component_contracts()

        logger.info(
            "PerceptionAgent initialized: device=%s embed_dim=%s decoders=%s",
            self.device,
            self.embed_dim,
            self.decoder_policy,
        )

    # ------------------------------------------------------------------
    # Configuration ownership
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_known_config(
        base: Mapping[str, Any],
        override: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Merge only keys already owned by the agent configuration schema.

        Constructor overrides are intentionally bounded to fields that already
        exist in ``agents_config.yaml/perception_agent``.  This prevents an
        arbitrary constructor mapping from silently becoming a second config
        schema for perception.
        """

        result = deepcopy(dict(base))
        for key, value in override.items():
            if key not in result:
                continue
            current = result[key]
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                result[key] = PerceptionAgent._merge_known_config(current, value)
            else:
                result[key] = deepcopy(value)
        return result

    def _init_agent_config(self, constructor_config: Optional[Mapping[str, Any]] ) -> None:
        raw_config = get_config_section("perception_agent") or {}
        if not isinstance(raw_config, Mapping):
            raise InvalidPerceptionConfigurationError(
                "agents_config.yaml/perception_agent must be a mapping.",
                component="perception_agent",
                details={"actual_type": type(raw_config).__name__},
            )

        agent_config = dict(raw_config)
        if constructor_config:
            if not isinstance(constructor_config, Mapping):
                raise InvalidPerceptionTypeError(
                    "PerceptionAgent constructor config must be a mapping.",
                    component="perception_agent",
                    details={"actual_type": type(constructor_config).__name__},
                )
            nested = constructor_config.get("perception_agent")
            override = nested if isinstance(nested, Mapping) else constructor_config
            agent_config = self._merge_known_config(agent_config, override)

        self.agent_config = agent_config
        self._validate_agent_config()

        self.device = resolve_torch_device(self.agent_config.get("device", "cpu"))
        self.embed_dim = int(self.agent_config.get("embed_dim", 512))
        self.masking_ratio = float(self.agent_config.get("masking_ratio", 0.15))
        self.contrastive_temperature = float(self.agent_config.get("contrastive_temperature", 0.07))
        self.learning_rate = float(self.agent_config.get("learning_rate", 1e-4))
        self.weight_decay = float(self.agent_config.get("weight_decay", 1e-2))
        betas = self.agent_config.get("adam_betas", (0.9, 0.999))
        self.adam_betas = (float(betas[0]), float(betas[1]))
        self.adam_eps = float(self.agent_config.get("adam_eps", 1e-7))
        grad_clip = self.agent_config.get("grad_clip_norm")
        self.grad_clip_norm = None if grad_clip is None else float(grad_clip)
        self.decoder_policy = dict(self.agent_config.get("decoders", {}))
        self.fusion_config = dict(self.agent_config.get("fusion", {}))
        self.shared_memory_config = dict(self.agent_config.get("shared_memory", {}))
        self.checkpoint_config = dict(self.agent_config.get("checkpointing", {}))

    def _validate_agent_config(self) -> None:
        config = self.agent_config

        embed_dim = config.get("embed_dim", 512)
        if isinstance(embed_dim, bool) or not isinstance(embed_dim, int) or embed_dim <= 0:
            raise InvalidPerceptionConfigurationError(
                "perception_agent.embed_dim must be a positive integer.",
                component="perception_agent",
                details={"embed_dim": embed_dim},
            )

        masking_ratio = config.get("masking_ratio", 0.15)
        try:
            masking_ratio_value = float(masking_ratio)
        except (TypeError, ValueError) as exc:
            raise InvalidPerceptionConfigurationError(
                "perception_agent.masking_ratio must be numeric.",
                component="perception_agent",
                details={"masking_ratio": masking_ratio},
                cause=exc,
            ) from exc
        if not 0.0 <= masking_ratio_value <= 1.0:
            raise InvalidPerceptionConfigurationError(
                "perception_agent.masking_ratio must be in [0, 1].",
                component="perception_agent",
                details={"masking_ratio": masking_ratio_value},
            )

        betas = config.get("adam_betas", (0.9, 0.999))
        if (
            not isinstance(betas, Sequence)
            or isinstance(betas, (str, bytes))
            or len(betas) != 2
        ):
            raise InvalidPerceptionConfigurationError(
                "perception_agent.adam_betas must contain exactly two values.",
                component="perception_agent",
                details={"adam_betas": betas},
            )

        for section_name in ("fusion", "decoders", "shared_memory", "checkpointing"):
            section = config.get(section_name, {})
            if section is not None and not isinstance(section, Mapping):
                raise InvalidPerceptionConfigurationError(
                    f"perception_agent.{section_name} must be a mapping.",
                    component="perception_agent",
                    details={"section": section_name, "actual_type": type(section).__name__},
                )

        decoder_config = dict(config.get("decoders", {}))
        for modality in ("text", "vision", "audio"):
            enabled = decoder_config.get(modality, True)
            if not isinstance(enabled, bool):
                raise InvalidPerceptionConfigurationError(
                    "Decoder enable flags must be booleans.",
                    component="perception_agent",
                    details={"modality": modality, "value": enabled},
                )

        checkpoint_config = dict(config.get("checkpointing", {}))
        checkpoint_enabled = checkpoint_config.get("enabled", True)
        if not isinstance(checkpoint_enabled, bool):
            raise InvalidPerceptionConfigurationError(
                "perception_agent.checkpointing.enabled must be boolean.",
                component="perception_agent",
                details={"value": checkpoint_enabled},
            )
        retention = checkpoint_config.get("retention_limit")
        if retention is not None and (
            isinstance(retention, bool)
            or not isinstance(retention, int)
            or retention < 1
        ):
            raise InvalidPerceptionConfigurationError(
                "checkpointing.retention_limit must be null or a positive integer.",
                component="perception_agent",
                details={"retention_limit": retention},
            )

        for ttl_name in ("snapshot_ttl_seconds", "embedding_cache_ttl_seconds", "training_lock_ttl_seconds"):
            ttl = dict(config.get("shared_memory", {})).get(ttl_name)
            if ttl is None:
                continue
            if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or float(ttl) <= 0.0:
                raise InvalidPerceptionConfigurationError(
                    f"shared_memory.{ttl_name} must be null or > 0 seconds.",
                    component="perception_agent",
                    details={ttl_name: ttl},
                )

    # ------------------------------------------------------------------
    # Component graph
    # ------------------------------------------------------------------
    def _init_components(self) -> None:
        """Construct subsystem boundaries without constructing raw models here."""

        self.text_perception = TextPerception(enable_decoder=bool(self.decoder_policy.get("text", True)), device=self.device)
        self.vision_perception = VisionPerception(enable_decoder=bool(self.decoder_policy.get("vision", True)), device=self.device)
        self.audio_perception = AudioPerception(enable_decoder=bool(self.decoder_policy.get("audio", True)), device=self.device)

        input_dims = {
            Modality.TEXT: self.text_perception.embed_dim,
            Modality.VISION: self.vision_perception.embed_dim,
            Modality.AUDIO: self.audio_perception.embed_dim,
        }

        self.fusion = PerceptionFusion(
            input_dims=input_dims,
            output_dim=self.embed_dim,
            fusion_method=str(self.fusion_config.get("method", "concat")),
            use_attention=bool(self.fusion_config.get("use_attention", True)),
            num_heads=int(self.fusion_config.get("num_heads", 8)),
            dropout=float(self.fusion_config.get("dropout", 0.1)),
        )

        self.objectives = PerceptionObjectives(
            input_dims=input_dims,
            contrastive_projection_dim=int( self.agent_config.get("contrastive_projection_dim", 256)),
            contrastive_temperature=self.contrastive_temperature,
            symmetric_contrastive=bool(self.agent_config.get("symmetric_contrastive", False)),
            temporal_loss_type=str(self.agent_config.get("loss_type", "hybrid")),
            temporal_max_scale=int(self.agent_config.get("max_scale", 3)),
            temporal_temperature=float(self.agent_config.get("temperature", 0.1)),
            temporal_mse_weight=float(self.agent_config.get("mse_weight", 1.0)),
            temporal_contrastive_weight=float(self.agent_config.get("contrastive_weight", 1.0)),
        )

        self.task_heads = nn.ModuleDict()
        self._task_head_specs: list[dict[str, Any]] = []

        self.trainer = PerceptionTrainer(
            modalities=self._modality_pipelines(),
            fusion=self.fusion,
            objectives=self.objectives,
            task_heads=self.task_heads,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            adam_betas=self.adam_betas,
            adam_eps=self.adam_eps,
            grad_clip_norm=self.grad_clip_norm,
            device=self.device,
        )

        # PerceptionTrainer moves every registered trainable subsystem module to
        # the resolved device before it constructs the optimizer. Do not call
        # ``self.to(...)`` after optimizer construction, because a post-build
        # device conversion can make optimizer parameter identity assumptions
        # unnecessarily fragile.

    def _validate_component_contracts(self) -> None:
        dimensions = {
            "text": int(self.text_perception.embed_dim),
            "vision": int(self.vision_perception.embed_dim),
            "audio": int(self.audio_perception.embed_dim),
        }
        mismatched = {
            name: value
            for name, value in dimensions.items()
            if value != self.embed_dim
        }
        if mismatched:
            raise PerceptionDimensionError(
                "Agent embed_dim does not match one or more modality contracts.",
                component="perception_agent",
                details={
                    "agent_embed_dim": self.embed_dim,
                    "modality_embed_dims": dimensions,
                    "mismatched": mismatched,
                },
                remediation=(
                    "Align base/configs/agents_config.yaml/perception_agent.embed_dim "
                    "with the subsystem encoder dimensions. The agent does not read "
                    "perception_config.yaml to resolve this mismatch."
                ),
            )

        if self.fusion.output_dim != self.embed_dim:
            raise PerceptionDimensionError(
                "PerceptionFusion output dimension must match PerceptionAgent.embed_dim.",
                component="perception_agent",
                details={
                    "fusion_output_dim": self.fusion.output_dim,
                    "agent_embed_dim": self.embed_dim,
                },
            )

    def _modality_pipelines(self) -> dict[Modality, nn.Module]:
        return {
            Modality.TEXT: self.text_perception,
            Modality.VISION: self.vision_perception,
            Modality.AUDIO: self.audio_perception,
        }

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Backward-compatible view of the trainer-owned optimizer."""

        return self.trainer.optimizer

    @property
    def tokenizer(self) -> Any:
        return self.text_perception.tokenizer

    @property
    def text_encoder(self) -> nn.Module:
        return self.text_perception.encoder

    @property
    def vision_encoder(self) -> nn.Module:
        return self.vision_perception.encoder

    @property
    def audio_encoder(self) -> nn.Module:
        return self.audio_perception.encoder

    @property
    def text_generator(self) -> nn.Module:
        return self.text_perception.require_decoder()

    @property
    def vision_generator(self) -> nn.Module:
        return self.vision_perception.require_decoder()

    @property
    def audio_generator(self) -> nn.Module:
        return self.audio_perception.require_decoder()

    # ------------------------------------------------------------------
    # SharedMemory ownership
    # ------------------------------------------------------------------
    def _init_shared_memory_keys(self) -> None:
        prefix = str(self.shared_memory_config.get("key_prefix", "perception")).strip()
        if not prefix:
            raise InvalidPerceptionConfigurationError(
                "shared_memory.key_prefix must not be empty.",
                component="perception_agent",
            )

        self.sm_keys = {
            "model_snapshot": f"{prefix}:snapshot:{self.name}",
            "embeddings": f"{prefix}:embeddings:{self.name}",
            "training_state": f"{prefix}:training:{self.name}",
        }

    @staticmethod
    def _ttl_from_seconds(value: Any) -> Optional[timedelta]:
        if value is None:
            return None
        return timedelta(seconds=float(value))

    def _acquire_training_lock(self) -> bool:
        key = self.sm_keys["training_state"]
        acquired = bool(self.shared_memory.compare_and_swap(key, None, self.agent_id))
        if acquired:
            ttl_seconds = self.shared_memory_config.get("training_lock_ttl_seconds")
            if ttl_seconds is not None:
                try:
                    self.shared_memory.put(
                        key,
                        self.agent_id,
                        ttl=self._ttl_from_seconds(ttl_seconds),
                        notify=False,
                        metadata={"owner": self.agent_id, "purpose": "perception_training"},
                    )
                except Exception:
                    # Do not strand a lock if TTL/metadata publication fails
                    # after successful CAS acquisition.
                    self.shared_memory.compare_and_swap(key, self.agent_id, None)
                    raise
        return acquired

    def _release_training_lock(self) -> None:
        # CAS makes release ownership-aware. Storing None is intentional: a
        # subsequent owner can atomically acquire from the unlocked state.
        self.shared_memory.compare_and_swap(
            self.sm_keys["training_state"],
            self.agent_id,
            None,
        )

    # ------------------------------------------------------------------
    # Canonical representation routing
    # ------------------------------------------------------------------
    def _pipeline_for(self, modality: Modality | str) -> Any:
        active = Modality.parse(modality)
        pipeline = self._modality_pipelines().get(active)
        if pipeline is None:  # defensive; Modality.parse already bounds values
            raise ModalityInputError(
                f"No perception pipeline is registered for '{active.value}'.",
                component="perception_agent",
            )
        return pipeline

    def _encode_modality(self, modality: Modality | str, payload: Any, *, style_id: Any = None) -> ModalityRepresentation:
        active = Modality.parse(modality)
        pipeline = self._pipeline_for(active)
        return pipeline.encode(payload, style_id=style_id)

    def _encode_multimodal(self, payload: Mapping[str, Any]) -> dict[Modality, ModalityRepresentation]:
        if not isinstance(payload, Mapping):
            raise ModalityInputError(
                "Multimodal input_data must be a mapping keyed by text/vision/audio.",
                component="perception_agent",
                details={"actual_type": type(payload).__name__},
            )

        shared_style = payload.get("style_id")
        representations: dict[Modality, ModalityRepresentation] = {}
        for modality in (Modality.TEXT, Modality.VISION, Modality.AUDIO):
            if modality.value not in payload or payload[modality.value] is None:
                continue
            representations[modality] = self._encode_modality(
                modality,
                payload[modality.value],
                style_id=shared_style,
            )

        if not representations:
            raise ModalityInputError(
                "Multimodal input contains no text, vision, or audio payload.",
                component="perception_agent",
                details={"keys": sorted(str(key) for key in payload.keys())},
            )
        return representations

    def _representations_from_task(self, task_data: Mapping[str, Any]) -> dict[Modality, ModalityRepresentation]:
        """Normalize fine-tuning input without duplicating modality preprocessing."""

        if "input_data" in task_data:
            modality_name = task_data.get("modality")
            if modality_name is None:
                raise ModalityInputError(
                    "'modality' is required when fine-tuning uses 'input_data'.",
                    component="perception_agent",
                )
            if str(modality_name).strip().lower() == "multimodal":
                return self._encode_multimodal(task_data["input_data"])
            modality = Modality.parse(modality_name)
            return {
                modality: self._encode_modality(modality, task_data["input_data"])
            }

        legacy_keys = {
            Modality.TEXT: "text_data",
            Modality.VISION: "vision_data",
            Modality.AUDIO: "audio_data",
        }
        representations: dict[Modality, ModalityRepresentation] = {}
        for modality, key in legacy_keys.items():
            if key in task_data and task_data[key] is not None:
                representations[modality] = self._encode_modality(
                    modality,
                    task_data[key],
                )

        if not representations:
            raise ModalityInputError(
                "Fine-tuning requires input_data+modality or at least one modality payload.",
                component="perception_agent",
            )
        return representations

    def _fuse(self, representations: Mapping[Modality | str, ModalityRepresentation]) -> FusedRepresentation:
        return self.fusion(representations)

    # ------------------------------------------------------------------
    # Dynamic downstream heads
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_task_key(value: str) -> str:
        normalized = str(value).strip().lower()
        safe = "".join(
            char if char.isalnum() or char in {"_", "-"} else "_"
            for char in normalized
        )
        return safe.strip("_") or "task"

    @staticmethod
    def _task_head_type(downstream_task: str) -> str:
        normalized = str(downstream_task).strip().lower()
        if "classification" in normalized:
            return "classification"
        if "regression" in normalized:
            return "regression"
        raise UnsupportedPerceptionOptionError(
            "Perception fine-tuning currently supports classification and regression heads.",
            component="perception_agent",
            details={"downstream_task": downstream_task},
        )

    def _task_head_key(self, downstream_task: str, input_dim: int, num_classes: Optional[int]) -> str:
        head_type = self._task_head_type(downstream_task)
        task_key = self._safe_task_key(downstream_task)
        class_suffix = "na" if num_classes is None else str(int(num_classes))
        return f"{head_type}__{task_key}__d{int(input_dim)}__c{class_suffix}"

    def _get_task_head(
        self,
        downstream_task: str,
        input_dim: int,
        num_classes: Optional[int] = None,
    ) -> tuple[str, nn.Module]:
        head_type = self._task_head_type(downstream_task)
        input_dim = int(input_dim)
        if input_dim <= 0:
            raise InvalidPerceptionValueError(
                "Task-head input_dim must be positive.",
                component="perception_agent",
                details={"input_dim": input_dim},
            )

        if head_type == "classification":
            if (
                num_classes is None
                or isinstance(num_classes, bool)
                or not isinstance(num_classes, int)
                or num_classes < 2
            ):
                raise InvalidPerceptionValueError(
                    "Classification fine-tuning requires num_classes >= 2.",
                    component="perception_agent",
                    details={"num_classes": num_classes},
                )
        elif num_classes is not None:
            # num_classes has no semantic role for regression and must not alter
            # the identity of the regression head.
            num_classes = None

        key = self._task_head_key(downstream_task, input_dim, num_classes)
        if key in self.task_heads:
            return key, self.task_heads[key]

        if head_type == "classification":
            assert num_classes is not None
            head = ClassificationHead(hidden_dim=input_dim)
            classifier = getattr(head, "classifier", None)
            if (
                not isinstance(classifier, nn.Sequential)
                or len(classifier) == 0
                or not isinstance(classifier[-1], nn.Linear)
            ):
                raise PerceptionConfigurationError(
                    "ClassificationHead does not expose the expected final linear classifier.",
                    component="perception_agent",
                    details={"head_type": type(head).__name__},
                    remediation=(
                        "Update the subsystem ClassificationHead contract rather than "
                        "guessing how to mutate an unknown head structure."
                    ),
                )
            final = classifier[-1]
            if final.out_features != num_classes:
                classifier[-1] = nn.Linear(
                    final.in_features,
                    num_classes,
                    bias=final.bias is not None,
                )
            head.num_classes = int(num_classes)
            resolved_num_classes: Optional[int] = int(num_classes)
        else:
            head = RegressionHead(hidden_dim=input_dim)
            resolved_num_classes = None

        self.trainer.register_task_head(key, head)
        spec = {
            "key": key,
            "downstream_task": str(downstream_task),
            "type": head_type,
            "input_dim": input_dim,
            "num_classes": resolved_num_classes,
        }
        self._task_head_specs.append(spec)
        return key, self.task_heads[key]

    def _get_existing_task_head(
        self,
        downstream_task: str,
        input_dim: int,
        num_classes: Optional[int] = None,
    ) -> tuple[str, nn.Module]:
        """Resolve a trained/restored head without creating parameters at inference."""

        head_type = self._task_head_type(downstream_task)
        if head_type == "classification":
            if (
                num_classes is None
                or isinstance(num_classes, bool)
                or not isinstance(num_classes, int)
                or num_classes < 2
            ):
                raise InvalidPerceptionValueError(
                    "Classification inference requires num_classes >= 2 to identify the trained head.",
                    component="perception_agent",
                    details={"num_classes": num_classes},
                )
        else:
            num_classes = None

        key = self._task_head_key(downstream_task, int(input_dim), num_classes)
        if key not in self.task_heads:
            raise PerceptionStateError(
                "Requested downstream head is not registered in the active PerceptionAgent state.",
                component="perception_agent",
                details={
                    "requested_head": key,
                    "available_heads": list(self.task_heads.keys()),
                },
                remediation=(
                    "Fine-tune the downstream task first or restore a checkpoint/shared "
                    "snapshot containing the trained head. Inference does not create an "
                    "untrained head implicitly."
                ),
            )
        return key, self.task_heads[key]

    def _restore_task_head_structure(self, specs: Any) -> None:
        if specs is None:
            specs = []
        if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)):
            raise PerceptionStateError(
                "task_head_specs must be a sequence of mappings.",
                component="perception_agent",
                details={"actual_type": type(specs).__name__},
            )

        # Model topology must be reconstructed before optimizer moments are
        # loaded.  Rebuilding here intentionally resets only the in-memory
        # optimizer being replaced by checkpoint/shared-state restoration.
        self.task_heads = nn.ModuleDict()
        self.trainer.task_heads = self.task_heads
        self._task_head_specs = []
        self.trainer.rebuild_optimizer()

        for index, raw_spec in enumerate(specs):
            if not isinstance(raw_spec, Mapping):
                raise PerceptionStateError(
                    "Each task-head specification must be a mapping.",
                    component="perception_agent",
                    details={"index": index, "actual_type": type(raw_spec).__name__},
                )

            required = {"key", "downstream_task", "type", "input_dim", "num_classes"}
            missing = required - set(raw_spec)
            if missing:
                raise PerceptionStateError(
                    "Task-head specification is incomplete.",
                    component="perception_agent",
                    details={"index": index, "missing": sorted(missing)},
                )

            expected_type = self._task_head_type(str(raw_spec["downstream_task"]))
            if expected_type != str(raw_spec["type"]):
                raise PerceptionStateError(
                    "Saved task-head type conflicts with downstream_task.",
                    component="perception_agent",
                    details={
                        "index": index,
                        "saved_type": raw_spec["type"],
                        "derived_type": expected_type,
                    },
                )

            key, _ = self._get_task_head(
                downstream_task=str(raw_spec["downstream_task"]),
                input_dim=int(raw_spec["input_dim"]),
                num_classes=(
                    None
                    if raw_spec["num_classes"] is None
                    else int(raw_spec["num_classes"])
                ),
            )
            if key != str(raw_spec["key"]):
                raise PerceptionStateError(
                    "Saved task-head identity does not match reconstructed identity.",
                    component="perception_agent",
                    details={"saved_key": raw_spec["key"], "reconstructed_key": key},
                )

    # ------------------------------------------------------------------
    # Training orchestration
    # ------------------------------------------------------------------
    @staticmethod
    def _require_field(mapping: Mapping[str, Any], key: str, *, context: str) -> Any:
        if key not in mapping:
            raise ModalityInputError(
                f"Missing required field '{key}' for {context}.",
                component="perception_agent",
                details={"available": sorted(str(item) for item in mapping.keys())},
            )
        return mapping[key]

    def _pretraining_step(self, task_data: Mapping[str, Any]) -> dict[str, Any]:
        objective = str(self._require_field(task_data, "objective", context="pretraining")).strip().lower()
        if objective not in _SUPPORTED_PRETRAIN_OBJECTIVES:
            raise UnsupportedPerceptionOptionError(
                f"Unsupported perception pretraining objective: {objective!r}.",
                component="perception_agent",
                details={"supported": sorted(_SUPPORTED_PRETRAIN_OBJECTIVES)},
            )

        if not self._acquire_training_lock():
            return {
                "status": "paused",
                "reason": "perception_training_lock_held",
                "lock_owner": self.shared_memory.get(self.sm_keys["training_state"]),
            }

        try:
            self.train(True)

            if objective == "mlm":
                result = self.trainer.masked_step(
                    Modality.TEXT,
                    self._require_field(task_data, "text_data", context=objective),
                    mask_ratio=self.masking_ratio,
                )
            elif objective == "mpm":
                result = self.trainer.masked_step(
                    Modality.VISION,
                    self._require_field(task_data, "vision_data", context=objective),
                    mask_ratio=self.masking_ratio,
                )
            elif objective == "mam":
                result = self.trainer.masked_step(
                    Modality.AUDIO,
                    self._require_field(task_data, "audio_data", context=objective),
                    mask_ratio=self.masking_ratio,
                )
            elif objective == "contrastive_text_image":
                result = self.trainer.contrastive_step(
                    Modality.TEXT,
                    self._require_field(task_data, "text_data", context=objective),
                    Modality.VISION,
                    self._require_field(task_data, "vision_data", context=objective),
                )
            elif objective == "contrastive_text_audio":
                result = self.trainer.contrastive_step(
                    Modality.TEXT,
                    self._require_field(task_data, "text_data", context=objective),
                    Modality.AUDIO,
                    self._require_field(task_data, "audio_data", context=objective),
                )
            elif objective == "contrastive_vision_audio":
                result = self.trainer.contrastive_step(
                    Modality.VISION,
                    self._require_field(task_data, "vision_data", context=objective),
                    Modality.AUDIO,
                    self._require_field(task_data, "audio_data", context=objective),
                )
            elif objective == "temporal_vision":
                video_data = self._require_field(task_data, "video_data", context=objective)
                if not isinstance(video_data, Mapping):
                    raise ModalityInputError(
                        "video_data must be a mapping containing frame_sequence.",
                        component="perception_agent",
                    )
                result = self.trainer.temporal_step(
                    Modality.VISION,
                    self._require_field(video_data, "frame_sequence", context=objective),
                )
            else:  # temporal_audio
                sequence_data = self._require_field(
                    task_data,
                    "audio_sequence_data",
                    context=objective,
                )
                if not isinstance(sequence_data, Mapping):
                    raise ModalityInputError(
                        "audio_sequence_data must be a mapping containing segment_sequence.",
                        component="perception_agent",
                    )
                result = self.trainer.temporal_step(
                    Modality.AUDIO,
                    self._require_field(sequence_data, "segment_sequence", context=objective),
                )

            return {"status": "success", **result.to_dict()}
        finally:
            self._release_training_lock()

    @staticmethod
    def _classification_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if targets.dim() != 1:
            raise PerceptionDimensionError(
                "Classification labels must have shape (B,).",
                component="perception_agent",
                details={"target_shape": list(targets.shape)},
            )
        if predictions.size(0) != targets.size(0):
            raise PerceptionDimensionError(
                "Classification prediction and label batch sizes differ.",
                component="perception_agent",
                details={
                    "prediction_batch": predictions.size(0),
                    "target_batch": targets.size(0),
                },
            )
        return F.cross_entropy(predictions, targets.long())

    @staticmethod
    def _regression_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        targets = targets.to(dtype=predictions.dtype, device=predictions.device)
        if predictions.dim() == 2 and predictions.size(-1) == 1 and targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        if tuple(predictions.shape) != tuple(targets.shape):
            raise PerceptionDimensionError(
                "Regression labels must match the prediction shape.",
                component="perception_agent",
                details={
                    "prediction_shape": list(predictions.shape),
                    "target_shape": list(targets.shape),
                },
            )
        return F.mse_loss(predictions, targets)

    def _finetune_step(self, task_data: Mapping[str, Any]) -> dict[str, Any]:
        if not self._acquire_training_lock():
            return {
                "status": "paused",
                "reason": "perception_training_lock_held",
                "lock_owner": self.shared_memory.get(self.sm_keys["training_state"]),
            }

        try:
            self.train(True)
            downstream_task = str(
                self._require_field(task_data, "downstream_task", context="fine-tuning")
            )
            head_type = self._task_head_type(downstream_task)
            representations = self._representations_from_task(task_data)
            fused = self._fuse(representations)

            labels = self._require_field(task_data, "labels", context="fine-tuning")
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels)
            labels = labels.to(self.device)

            num_classes = task_data.get("num_classes")
            if num_classes is not None and not isinstance(num_classes, int):
                raise InvalidPerceptionValueError(
                    "num_classes must be an integer when provided.",
                    component="perception_agent",
                    details={"num_classes": num_classes},
                )

            head_name, _ = self._get_task_head(
                downstream_task=downstream_task,
                input_dim=fused.embedding_dim,
                num_classes=num_classes,
            )
            loss_fn = (
                self._classification_loss
                if head_type == "classification"
                else self._regression_loss
            )
            result = self.trainer.supervised_step(
                head_name,
                fused,
                labels,
                loss_fn,
            )
            return {
                "status": "success",
                "head": head_name,
                "modalities": [item.value for item in representations],
                **result.to_dict(),
            }
        finally:
            self._release_training_lock()

    # ------------------------------------------------------------------
    # Inference / generation
    # ------------------------------------------------------------------
    def _inference_step(self, task_data: Mapping[str, Any]) -> dict[str, Any]:
        self.eval()
        modality_name = str(
            self._require_field(task_data, "modality", context="inference")
        ).strip().lower()
        input_data = self._require_field(task_data, "input_data", context="inference")

        with torch.inference_mode():
            if modality_name == "multimodal":
                representations = self._encode_multimodal(input_data)
                fused = self._fuse(representations)
                canonical_output: torch.Tensor = fused.pooled
                representation_summary: dict[str, Any] = fused.summary()
                single_representation: Optional[ModalityRepresentation] = None
            else:
                modality = Modality.parse(modality_name)
                single_representation = self._encode_modality(modality, input_data)
                representations = {modality: single_representation}
                fused = None
                if task_data.get("return_sequence", False) and single_representation.sequence is not None:
                    canonical_output = single_representation.sequence
                else:
                    canonical_output = single_representation.pooled
                representation_summary = single_representation.summary()

            downstream_task = task_data.get("downstream_task")
            if downstream_task:
                # Downstream heads always consume the fixed-width fusion contract,
                # including single-modality tasks.  This keeps one stable head
                # interface and prevents modality-specific head dimensions.
                if fused is None:
                    fused = self._fuse(representations)
                head_name, _ = self._get_existing_task_head(
                    downstream_task=str(downstream_task),
                    input_dim=fused.embedding_dim,
                    num_classes=task_data.get("num_classes"),
                )
                output = self.trainer.forward_task(head_name, fused)
            elif bool(task_data.get("generate", False)):
                if modality_name == "multimodal":
                    raise UnsupportedPerceptionOptionError(
                        "Multimodal generation requires an explicit learned decoding contract and is not inferred by the agent.",
                        component="perception_agent",
                        remediation=(
                            "Decode a specific ModalityRepresentation with its subsystem "
                            "pipeline, or add an explicit multimodal decoder before routing "
                            "fused representations to generation."
                        ),
                    )
                assert single_representation is not None
                modality = single_representation.modality
                pipeline = self._pipeline_for(modality)
                if modality is Modality.TEXT:
                    output = pipeline.reconstruct(
                        single_representation,
                        strategy=str(task_data.get("generation_strategy", "greedy")),
                    )
                else:
                    output = pipeline.reconstruct(single_representation)
            else:
                output = canonical_output

            return {
                "status": "success",
                "output": detach_tree(output, cpu=True),
                "representation": representation_summary,
            }

    # ------------------------------------------------------------------
    # Public task boundary
    # ------------------------------------------------------------------
    def perform_task(self, task_data: Mapping[str, Any]) -> dict[str, Any]:
        """Dispatch an SLAI perception request to pretrain, fine-tune, or infer."""

        if not isinstance(task_data, Mapping):
            raise ModalityInputError(
                "PerceptionAgent task_data must be a mapping.",
                component="perception_agent",
                details={"actual_type": type(task_data).__name__},
            )

        if bool(task_data.get("use_cached_state", False)):
            restored = self.load_state_from_shared_memory()
            if restored:
                logger.info("Restored transient PerceptionAgent state from SharedMemory")

        task_type = str(task_data.get("task_type", "")).strip().lower()
        if task_type not in _SUPPORTED_TASK_TYPES:
            raise UnsupportedPerceptionOptionError(
                f"Unsupported perception task_type: {task_type!r}.",
                component="perception_agent",
                details={"supported": sorted(_SUPPORTED_TASK_TYPES)},
            )

        if task_type == "pretrain":
            result = self._pretraining_step(task_data)
        elif task_type == "finetune":
            result = self._finetune_step(task_data)
        else:
            result = self._inference_step(task_data)

        if bool(task_data.get("save_state_after", False)):
            self.save_state_to_shared_memory()
        return result

    def extract_performance_metrics(self, result: Any) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if not isinstance(result, Mapping):
            return metrics

        loss = result.get("loss")
        if isinstance(loss, (int, float)) and not isinstance(loss, bool):
            metrics["loss"] = float(loss)
        metrics["task_successful"] = 1.0 if result.get("status") == "success" else 0.0
        return metrics

    # ------------------------------------------------------------------
    # Transient SharedMemory state
    # ------------------------------------------------------------------
    def _agent_state_payload(self, *, schema: str) -> dict[str, Any]:
        return {
            "schema_version": schema,
            "agent_version": __version__,
            "global_step": int(self.trainer.global_step),
            "task_head_specs": deepcopy(self._task_head_specs),
            "agent_config": deepcopy(self.agent_config),
        }

    def _validate_restored_agent_state(
        self,
        state: Any,
        *,
        expected_schema: str,
    ) -> Mapping[str, Any]:
        if not isinstance(state, Mapping):
            raise PerceptionStateError(
                "Perception agent state must be a mapping.",
                component="perception_agent",
                details={"actual_type": type(state).__name__},
            )
        if state.get("schema_version") != expected_schema:
            raise PerceptionStateError(
                "Perception state schema is incompatible with this loader.",
                component="perception_agent",
                details={
                    "expected": expected_schema,
                    "actual": state.get("schema_version"),
                },
            )

        saved_config = state.get("agent_config")
        if isinstance(saved_config, Mapping) and "embed_dim" in saved_config:
            saved_dim = int(saved_config["embed_dim"])
            if saved_dim != self.embed_dim:
                raise PerceptionStateError(
                    "Saved PerceptionAgent embed_dim is incompatible with the active runtime.",
                    component="perception_agent",
                    details={"saved_embed_dim": saved_dim, "active_embed_dim": self.embed_dim},
                    remediation=(
                        "Restore with a compatible agent/subsystem architecture; do not "
                        "silently reshape learned perception state."
                    ),
                )
        return state

    def save_state_to_shared_memory(self) -> dict[str, Any]:
        """Publish a complete transient runtime snapshot to SharedMemory."""

        snapshot = {
            "model_state_dict": detach_tree(self.state_dict(), cpu=True),
            "optimizer_state_dict": detach_tree(self.trainer.optimizer.state_dict(), cpu=True),
            "agent_state": self._agent_state_payload(schema=_SHARED_STATE_SCHEMA),
        }
        ttl = self._ttl_from_seconds(
            self.shared_memory_config.get("snapshot_ttl_seconds", 86400)
        )
        self.shared_memory.put(
            self.sm_keys["model_snapshot"],
            snapshot,
            ttl=ttl,
            tags=("perception", "model_snapshot", self.name),
            metadata={"agent_id": self.agent_id, "schema": _SHARED_STATE_SCHEMA},
        )
        logger.info("Saved complete PerceptionAgent runtime snapshot to SharedMemory")
        return {
            "status": "success",
            "key": self.sm_keys["model_snapshot"],
            "global_step": int(self.trainer.global_step),
        }

    def load_state_from_shared_memory(self) -> bool:
        """Restore a transient snapshot after rebuilding dynamic head topology."""

        snapshot = self.shared_memory.get(self.sm_keys["model_snapshot"])
        if snapshot is None:
            return False
        if not isinstance(snapshot, Mapping):
            raise PerceptionStateError(
                "SharedMemory perception snapshot must be a mapping.",
                component="perception_agent",
                details={"actual_type": type(snapshot).__name__},
            )

        try:
            agent_state = self._validate_restored_agent_state(
                snapshot.get("agent_state"),
                expected_schema=_SHARED_STATE_SCHEMA,
            )
            self._restore_task_head_structure(agent_state.get("task_head_specs", []))

            model_state = snapshot.get("model_state_dict")
            optimizer_state = snapshot.get("optimizer_state_dict")
            if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping):
                raise PerceptionStateError(
                    "SharedMemory snapshot is missing model or optimizer state.",
                    component="perception_agent",
                )

            self.load_state_dict(model_state, strict=True)
            self.trainer.optimizer.load_state_dict(optimizer_state)
            self.trainer.global_step = int(agent_state.get("global_step", 0))
            logger.info("Loaded complete PerceptionAgent runtime snapshot from SharedMemory")
            return True
        except PerceptionError:
            raise
        except Exception as exc:
            raise PerceptionStateError.from_exception(
                exc,
                "Failed to restore PerceptionAgent SharedMemory state.",
                component="perception_agent",
            ) from exc

    @staticmethod
    def _tensor_digest(tensor: torch.Tensor) -> str:
        if not isinstance(tensor, torch.Tensor):
            raise InvalidPerceptionTypeError(
                "Embedding-cache inputs must be tensors.",
                component="perception_agent",
                details={"actual_type": type(tensor).__name__},
            )
        cpu = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tuple(cpu.shape)).encode("utf-8"))
        digest.update(str(cpu.dtype).encode("utf-8"))
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def cache_embeddings( self, modality: Modality | str, inputs: torch.Tensor, embeddings: torch.Tensor) -> str:
        active = Modality.parse(modality)
        if not isinstance(embeddings, torch.Tensor):
            raise InvalidPerceptionTypeError(
                "embeddings must be a tensor.",
                component="perception_agent",
            )
        key = f"{self.sm_keys['embeddings']}:{active.value}:{self._tensor_digest(inputs)}"
        ttl = self._ttl_from_seconds(
            self.shared_memory_config.get("embedding_cache_ttl_seconds")
        )
        self.shared_memory.put(
            key,
            embeddings.detach().cpu(),
            ttl=ttl,
            tags=("perception", "embedding", active.value),
            metadata={"agent_id": self.agent_id},
        )
        return key

    def get_cached_embeddings(self, modality: Modality | str, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        active = Modality.parse(modality)
        key = f"{self.sm_keys['embeddings']}:{active.value}:{self._tensor_digest(inputs)}"
        cached = self.shared_memory.get(key)
        if cached is None:
            return None
        if not isinstance(cached, torch.Tensor):
            raise PerceptionStateError(
                "Shared embedding cache contains a non-tensor value.",
                component="perception_agent",
                details={"key": key, "actual_type": type(cached).__name__},
            )
        return cached

    # ------------------------------------------------------------------
    # Durable recovery through central checkpointing
    # ------------------------------------------------------------------
    def _init_checkpoint_manager(self) -> None:
        enabled = bool(self.checkpoint_config.get("enabled", True))
        self.checkpoint_manager: Optional[CheckpointManager]
        if not enabled:
            self.checkpoint_manager = None
            return

        base_dir = Path(
            str(self.checkpoint_config.get("base_dir", "src/checkpoints/perception"))
        )
        retention_limit = self.checkpoint_config.get("retention_limit")
        self.checkpoint_manager = CheckpointManager(
            base_dir=base_dir,
            retention_limit=retention_limit,
        )

    def _require_checkpoint_manager(self) -> CheckpointManager:
        if self.checkpoint_manager is None:
            raise PerceptionStateError(
                "Perception durable checkpointing is disabled by agent configuration.",
                component="perception_agent",
                remediation=(
                    "Enable perception_agent.checkpointing.enabled in "
                    "base/configs/agents_config.yaml to use durable recovery."
                ),
            )
        return self.checkpoint_manager

    def save_checkpoint(self, version: Optional[str] = None, *, metadata: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Save complete durable model/optimizer/agent state transactionally."""

        if metadata is not None and not isinstance(metadata, Mapping):
            raise InvalidPerceptionTypeError(
                "checkpoint metadata must be a mapping.",
                component="perception_agent",
            )

        manager = self._require_checkpoint_manager()
        components: dict[str, Any] = {
            "model": self,
            "optimizer": self.trainer.optimizer,
            "agent_state": self._agent_state_payload(schema=_CHECKPOINT_STATE_SCHEMA),
        }
        codec_ids = {
            "model": "torch",
            "optimizer": "torch",
            "agent_state": "agent-state",
        }
        if bool(self.checkpoint_config.get("save_rng", True)):
            components["rng"] = None
            codec_ids["rng"] = "rng"

        try:
            result = manager.save_components(
                components,
                version=version,
                codec_ids=codec_ids,
                step=int(self.trainer.global_step),
                metadata={
                    "agent": self.name,
                    "agent_version": __version__,
                    **dict(metadata or {}),
                },
            )
        except Exception as exc:
            raise PerceptionStateError.from_exception(
                exc,
                "Durable PerceptionAgent checkpoint save failed.",
                component="perception_agent",
            ) from exc

        return {
            "status": "success",
            "version": result.record.version,
            "checkpoint_id": result.record.checkpoint_id,
            "committed": bool(result.committed),
            "global_step": int(self.trainer.global_step),
        }

    def restore_checkpoint(self, version: Optional[str] = None) -> dict[str, Any]:
        """Restore durable state using two-phase dynamic-head reconstruction."""

        manager = self._require_checkpoint_manager()
        try:
            # Phase 1: decode state metadata only. No model/optimizer target is
            # mutated until the dynamic task-head topology is reconstructed.
            state_result = manager.load_components(
                version,
                components=("agent_state",),
                strict=True,
                verify_integrity=True,
            )
            agent_state = self._validate_restored_agent_state(
                state_result.components.get("agent_state"),
                expected_schema=_CHECKPOINT_STATE_SCHEMA,
            )
            self._restore_task_head_structure(agent_state.get("task_head_specs", []))

            saved_components = set(state_result.record.manifest.saved_components)
            restore_components = ["model", "optimizer"]
            restore_rng = "rng" in saved_components
            if restore_rng:
                restore_components.append("rng")

            # Phase 2: CheckpointManager performs selection, integrity checks,
            # compatibility checks, and safe decoding for every requested
            # component before model state is mutated. The decoded optimizer is
            # then applied explicitly because the current generic Torch codec
            # target adapter passes ``strict=`` to load_state_dict(), while
            # torch.optim.Optimizer.load_state_dict() has no strict parameter.
            load_result = manager.load_components(
                state_result.record.version,
                components=tuple(restore_components),
                strict=True,
                restore_rng=restore_rng,
                verify_integrity=True,
            )

            model_state = load_result.components.get("model")
            optimizer_state = load_result.components.get("optimizer")
            if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping):
                raise PerceptionStateError(
                    "Checkpoint is missing decoded model or optimizer state.",
                    component="perception_agent",
                    details={
                        "decoded_components": sorted(load_result.components.keys()),
                    },
                )

            self.load_state_dict(model_state, strict=True)
            self.trainer.optimizer.load_state_dict(optimizer_state)
            self.trainer.global_step = int(agent_state.get("global_step", 0))
        except PerceptionError:
            raise
        except Exception as exc:
            raise PerceptionStateError.from_exception(
                exc,
                "Durable PerceptionAgent checkpoint restore failed.",
                component="perception_agent",
            ) from exc

        return {
            "status": "success",
            "version": load_result.record.version,
            "checkpoint_id": load_result.record.checkpoint_id,
            "loaded_components": list(load_result.loaded_components),
            "restored_rng": bool(load_result.restored_rng),
            "global_step": int(self.trainer.global_step),
        }

    # ------------------------------------------------------------------
    # Explicitly retired legacy mutation/import paths
    # ------------------------------------------------------------------
    def load_pretrained_weights(
        self,
        checkpoint_path: str | Path,
        source_format: str = "perception_agent_checkpoint",
    ) -> None:
        """Fail safely instead of performing heuristic cross-architecture mapping.

        The previous agent implementation guessed external parameter mappings,
        split ambiguous fusion tensors, and sometimes assigned unknown tensors to
        multiple modalities.  Such conversion belongs in an explicit,
        format-specific subsystem adapter with a validated schema—not in the
        agent orchestration boundary.
        """

        raise UnsupportedPerceptionOptionError(
            "Direct external-format pretrained-weight conversion is not an agent responsibility in v2.3.",
            component="perception_agent",
            details={
                "checkpoint_path": str(checkpoint_path),
                "source_format": source_format,
            },
            remediation=(
                "Use restore_checkpoint() for SLAI checkpoints. For third-party "
                "weights, implement or use an explicit subsystem-owned adapter that "
                "validates the source architecture and target parameter schema."
            ),
        )

    def update_projection(self, rewards: Any, lr: float) -> None:
        """Compatibility guard for the removed standalone projection parameter."""

        raise UnsupportedPerceptionOptionError(
            "PerceptionAgent no longer owns a standalone global_projection_param.",
            component="perception_agent",
            details={"learning_rate": lr, "reward_type": type(rewards).__name__},
            remediation=(
                "Route representation learning through PerceptionFusion, "
                "PerceptionObjectives, and PerceptionTrainer so learned parameters "
                "remain optimizer-registered and checkpoint-complete."
            ),
        )


__all__ = ["PerceptionAgent"]


if __name__ == "__main__":
    """
    End-to-end smoke test harness for PerceptionAgent.

    This block intentionally runs multiple critical paths and reports granular pass/fail
    outcomes so architecture regressions are visible immediately.
    """
    print("\n=== PerceptionAgent integration smoke test ===")
    printer.status("TEST", "Starting Task Coordinator tests", "info")
    from .collaborative.shared_memory import SharedMemory
    from .agent_factory import AgentFactory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    perception_config = get_config_section('perception_agent')
    agent_type="Perception"

    agent = PerceptionAgent(
        shared_memory=shared_memory,
        agent_factory=agent_factory,
        config=perception_config
    )
    print(agent)
    print("\n* * * * * Phase 2 * * * * *\n")
    def _run_test(name, fn):
        try:
            result = fn()
            print(f"[PASS] {name}")
            return True, result
        except Exception as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            return False, None

    torch.manual_seed(7)
    random.seed(7)

    # 1) Agent initialization
    init_ok, agent_obj = _run_test(
        "initialize agent",
        lambda: PerceptionAgent(shared_memory=shared_memory, agent_factory=agent_factory)
    )
    if init_ok:
        agent = agent_obj
        device = getattr(agent, "device", "cpu")
    else:
        print("Aborting remaining tests because initialization failed.")
        raise SystemExit(1)

    # 2) Build representative multimodal batch
    def _build_dummy_batch():
        batch_size = 2
        text_len = 32
        img_size = int(getattr(agent.vision_encoder, "img_size", 224))
        audio_len = int(getattr(agent.audio_encoder, "audio_length", 16000))
        vocab_size = int(getattr(agent, "tokenizer").get_vocab_size())

        return {
            "text": {
                "input_ids": torch.randint(0, vocab_size, (batch_size, text_len), dtype=torch.long, device=device),
                "attention_mask": torch.ones(batch_size, text_len, dtype=torch.long, device=device),
            },
            "vision": {
                "pixel_values": torch.randn(batch_size, 3, img_size, img_size, device=device),
            },
            "audio": {
                "waveform": torch.randn(batch_size, 1, audio_len, device=device),
            },
            "style_id": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    data_ok, batch = _run_test("build dummy multimodal batch", _build_dummy_batch)

    # 3) Encoder forward passes
    if data_ok:
        _run_test(
            "text encoder forward",
            lambda: agent.text_encoder(batch["text"]["input_ids"], style_id=batch["style_id"])
        )
        _run_test(
            "vision encoder forward",
            lambda: agent.vision_encoder(batch["vision"]["pixel_values"], style_id=batch["style_id"])
        )
        _run_test(
            "audio encoder forward",
            lambda: agent.audio_encoder(batch["audio"]["waveform"], style_id=batch["style_id"])
        )

    # 4) Inference step tests by modality
    if data_ok:
        _run_test(
            "inference:text",
            lambda: agent._inference_step({
                "modality": "text",
                "input_data": {
                    "input_ids": batch["text"]["input_ids"],
                    "style_id": batch["style_id"]
                }
            })
        )
        _run_test(
            "inference:vision",
            lambda: agent._inference_step({
                "modality": "vision",
                "input_data": {
                    "pixel_values": batch["vision"]["pixel_values"],
                    "style_id": batch["style_id"]
                }
            })
        )
        _run_test(
            "inference:audio",
            lambda: agent._inference_step({
                "modality": "audio",
                "input_data": {
                    "waveform": batch["audio"]["waveform"],
                    "style_id": batch["style_id"]
                }
            })
        )

    # 5) Shared-memory checkpoint roundtrip
    _run_test("save state to shared memory", lambda: agent.save_state_to_shared_memory())
    _run_test("load state from shared memory", lambda: agent.load_state_from_shared_memory())

    print("=== PerceptionAgent smoke test completed ===\n")
