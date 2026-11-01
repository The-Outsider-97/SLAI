"""Training orchestration for the SLAI perception subsystem.

``PerceptionTrainer`` owns optimizer lifecycle and training-step mechanics.  It
intentionally does not own SLAI agent lifecycle, shared memory, durable
checkpoint storage, or modality preprocessing.  Trainable parameters are
collected from the modality pipelines, fusion layer, objective projections, and
dynamically registered task heads exactly once.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .perception_contracts import *
from .perception_fusion import PerceptionFusion
from .perception_objectives import PerceptionObjectives
from .modalities.base import BasePerceptionModality
from .utils.config_loader import get_config_section, load_global_config
from .utils.perception_errors import *
from .utils.perception_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Perception Trainer")
printer = PrettyPrinter()


class PerceptionTrainer:
    """Optimizer and objective coordinator for internal perception components."""

    def __init__(
        self,
        *,
        modalities: Mapping[Union[Modality, str], BasePerceptionModality],
        fusion: PerceptionFusion,
        objectives: PerceptionObjectives,
        task_heads: Optional[nn.ModuleDict] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-7,
        grad_clip_norm: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if not modalities:
            raise PerceptionTrainingError("At least one modality pipeline is required.")

        normalized: Dict[Modality, BasePerceptionModality] = {}
        for key, pipeline in modalities.items():
            modality = Modality.parse(key)
            if not isinstance(pipeline, BasePerceptionModality):
                raise PerceptionTrainingError(
                    "Trainer modalities must implement BasePerceptionModality.",
                    details={"modality": modality.value, "actual_type": type(pipeline).__name__},
                )
            if pipeline.modality != modality:
                raise PerceptionTrainingError(
                    "Trainer modality key and pipeline modality disagree.",
                    details={"key": modality.value, "pipeline": pipeline.modality.value},
                )
            normalized[modality] = pipeline

        self.modalities = normalized
        self.fusion = fusion
        self.objectives = objectives
        self.task_heads = task_heads if task_heads is not None else nn.ModuleDict()

        first_pipeline = next(iter(self.modalities.values()))
        self.device = resolve_torch_device(device or first_pipeline.device)
        for pipeline in self.modalities.values():
            pipeline.to(self.device)
            pipeline.device = self.device
        self.fusion.to(self.device)
        self.objectives.to(self.device)
        self.task_heads.to(self.device)

        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.adam_betas = (float(adam_betas[0]), float(adam_betas[1]))
        self.adam_eps = float(adam_eps)
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self._validate_optimizer_config()

        self.global_step = 0
        self.optimizer = optimizer or self._build_optimizer()
        if optimizer is not None:
            add_missing_parameters_to_optimizer(
                self.optimizer,
                self._all_trainable_parameters(),
            )

    def _validate_optimizer_config(self) -> None:
        if self.learning_rate <= 0.0:
            raise OptimizerConfigurationError(
                "learning_rate must be > 0.",
                details={"learning_rate": self.learning_rate},
            )
        if self.weight_decay < 0.0:
            raise OptimizerConfigurationError(
                "weight_decay must be >= 0.",
                details={"weight_decay": self.weight_decay},
            )
        if len(self.adam_betas) != 2 or not all(0.0 <= value < 1.0 for value in self.adam_betas):
            raise OptimizerConfigurationError(
                "adam_betas must contain two values in [0,1).",
                details={"adam_betas": self.adam_betas},
            )
        if self.adam_eps <= 0.0:
            raise OptimizerConfigurationError(
                "adam_eps must be > 0.",
                details={"adam_eps": self.adam_eps},
            )
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise OptimizerConfigurationError(
                "grad_clip_norm must be > 0 when provided.",
                details={"grad_clip_norm": self.grad_clip_norm},
            )

    def _all_trainable_parameters(self) -> list[nn.Parameter]:
        return collect_unique_trainable_parameters(
            self.modalities,
            self.fusion,
            self.objectives,
            self.task_heads,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        parameters = self._all_trainable_parameters()
        if not parameters:
            raise OptimizerConfigurationError("No trainable perception parameters were found.")
        return torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            betas=self.adam_betas,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )

    def rebuild_optimizer(self) -> torch.optim.Optimizer:
        """Deliberately rebuild AdamW after structural parameter changes.

        Rebuilding resets optimizer moments.  Normal dynamic task-head addition
        should use ``register_task_head`` instead, which adds only the new
        parameters and preserves existing optimizer state.
        """

        self.optimizer = self._build_optimizer()
        return self.optimizer

    def register_task_head(self, name: str, head: nn.Module) -> int:
        """Register a new downstream task head and immediately optimize it."""

        normalized_name = str(name).strip()
        if not normalized_name:
            raise PerceptionTrainingError("Task-head name must not be empty.")
        if not isinstance(head, nn.Module):
            raise PerceptionTrainingError(
                "Task head must be an nn.Module.",
                details={"name": normalized_name, "actual_type": type(head).__name__},
            )
        if normalized_name in self.task_heads:
            if self.task_heads[normalized_name] is head:
                return 0
            raise PerceptionTrainingError(
                f"Task head '{normalized_name}' is already registered.",
                remediation="Use a new name or deliberately replace the head and rebuild the optimizer.",
            )

        self.task_heads[normalized_name] = head.to(self.device)
        added = add_missing_parameters_to_optimizer(
            self.optimizer,
            self.task_heads[normalized_name].parameters(),
        )
        return added

    def set_training_mode(self, mode: bool = True) -> None:
        for pipeline in self.modalities.values():
            pipeline.train(mode)
        self.fusion.train(mode)
        self.objectives.train(mode)
        self.task_heads.train(mode)

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def _optimize(self, objective: ObjectiveLoss) -> TrainingStepResult:
        loss = objective.value
        ensure_finite_tensor(loss, "loss", component="perception_trainer")
        if not loss.requires_grad:
            raise PerceptionTrainingError(
                "Training loss is detached from the autograd graph.",
                details={"objective": objective.name},
            )

        self.zero_grad(set_to_none=True)
        try:
            loss.backward()
        except Exception as exc:
            raise PerceptionTrainingError.from_exception(
                exc,
                "Backward pass failed.",
                details={"objective": objective.name},
            ) from exc

        grad_norm: Optional[float] = None
        parameters = self._all_trainable_parameters()
        if self.grad_clip_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip_norm)
            grad_norm = float(norm.detach().item() if isinstance(norm, torch.Tensor) else norm)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                self.zero_grad(set_to_none=True)
                raise NonFiniteLossError(
                    "Gradient norm became non-finite.",
                    details={"objective": objective.name, "grad_norm": grad_norm},
                )

        try:
            self.optimizer.step()
        except Exception as exc:
            raise PerceptionTrainingError.from_exception(
                exc,
                "Optimizer step failed.",
                details={"objective": objective.name},
            ) from exc

        self.global_step += 1
        metrics = objective.detached_metrics()
        return TrainingStepResult(
            objective=objective.name,
            loss=float(loss.detach().item()),
            global_step=self.global_step,
            grad_norm=grad_norm,
            metrics=metrics,
            skipped=False,
        )

    def masked_step(
        self,
        modality: Union[Modality, str],
        payload: Any,
        *,
        mask_ratio: float = 0.15,
        **kwargs: Any,
    ) -> TrainingStepResult:
        active = Modality.parse(modality)
        pipeline = self.modalities.get(active)
        if pipeline is None:
            raise PerceptionTrainingError(
                f"No trainer pipeline is registered for {active.value}.",
            )
        self.set_training_mode(True)
        prediction = pipeline.masked_prediction(
            payload,
            mask_ratio=mask_ratio,
            **kwargs,
        )
        objective = self.objectives.masked_reconstruction(prediction)
        return self._optimize(objective)

    def contrastive_step(
        self,
        first_modality: Union[Modality, str],
        first_payload: Any,
        second_modality: Union[Modality, str],
        second_payload: Any,
        *,
        symmetric: Optional[bool] = None,
        first_kwargs: Optional[Mapping[str, Any]] = None,
        second_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> TrainingStepResult:
        first_key = Modality.parse(first_modality)
        second_key = Modality.parse(second_modality)
        first_pipeline = self.modalities.get(first_key)
        second_pipeline = self.modalities.get(second_key)
        if first_pipeline is None or second_pipeline is None:
            raise PerceptionTrainingError(
                "Both contrastive modality pipelines must be registered.",
                details={"first": first_key.value, "second": second_key.value},
            )

        self.set_training_mode(True)
        first = first_pipeline.encode(first_payload, **dict(first_kwargs or {}))
        second = second_pipeline.encode(second_payload, **dict(second_kwargs or {}))
        objective = self.objectives.contrastive(first, second, symmetric=symmetric)
        return self._optimize(objective)

    def temporal_step(
        self,
        modality: Union[Modality, str],
        sequence_data: torch.Tensor,
        *,
        loss_type: Optional[str] = None,
        **kwargs: Any,
    ) -> TrainingStepResult:
        active = Modality.parse(modality)
        pipeline = self.modalities.get(active)
        if pipeline is None:
            raise PerceptionTrainingError(
                f"No trainer pipeline is registered for {active.value}.",
            )
        self.set_training_mode(True)
        embeddings = pipeline.encode_temporal(sequence_data, **kwargs)
        objective = self.objectives.temporal_coherence(embeddings, loss_type=loss_type)
        return self._optimize(objective)

    def fuse(
        self,
        representations: Mapping[Union[Modality, str], ModalityRepresentation],
    ) -> FusedRepresentation:
        return self.fusion(representations)

    def forward_task(
        self,
        head_name: str,
        representation: Union[
            FusedRepresentation,
            Mapping[Union[Modality, str], ModalityRepresentation],
        ],
    ) -> torch.Tensor:
        if head_name not in self.task_heads:
            raise PerceptionTrainingError(
                f"Unknown task head '{head_name}'.",
                details={"available": list(self.task_heads.keys())},
            )
        fused = representation if isinstance(representation, FusedRepresentation) else self.fusion(representation)
        return self.task_heads[head_name](fused.pooled)

    def supervised_step(
        self,
        head_name: str,
        representation: Union[
            FusedRepresentation,
            Mapping[Union[Modality, str], ModalityRepresentation],
        ],
        targets: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> TrainingStepResult:
        if not callable(loss_fn):
            raise PerceptionTrainingError("loss_fn must be callable.")
        self.set_training_mode(True)
        predictions = self.forward_task(head_name, representation)
        targets = targets.to(predictions.device)
        try:
            loss = loss_fn(predictions, targets)
        except Exception as exc:
            raise PerceptionTrainingError.from_exception(
                exc,
                "Downstream task loss computation failed.",
                details={"head": head_name},
            ) from exc
        objective = ObjectiveLoss(
            name=f"task_{head_name}",
            value=loss,
            components={"task_loss": loss},
        )
        return self._optimize(objective)

    def optimizer_parameter_count(self) -> int:
        return len(optimizer_parameter_ids(self.optimizer))

    def state_dict(self) -> Dict[str, Any]:
        """Return trainer-owned state only; model state remains with its modules."""

        return {
            "optimizer": self.optimizer.state_dict(),
            "global_step": int(self.global_step),
            "task_heads": self.task_heads.state_dict(),
            "optimizer_config": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "adam_betas": list(self.adam_betas),
                "adam_eps": self.adam_eps,
                "grad_clip_norm": self.grad_clip_norm,
            },
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        load_optimizer: bool = True,
        strict_task_heads: bool = True,
    ) -> None:
        if not isinstance(state, Mapping):
            raise PerceptionStateError(
                "Trainer state must be a mapping.",
                details={"actual_type": type(state).__name__},
            )

        task_state = state.get("task_heads")
        if task_state is not None:
            try:
                self.task_heads.load_state_dict(task_state, strict=strict_task_heads)
            except Exception as exc:
                raise PerceptionStateError.from_exception(
                    exc,
                    "Failed to restore perception task-head state.",
                    remediation="Register the same task-head structure before restoring trainer state.",
                ) from exc

        if load_optimizer and "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception as exc:
                raise PerceptionStateError.from_exception(
                    exc,
                    "Failed to restore perception optimizer state.",
                    remediation="Restore model/task-head structure before optimizer state.",
                ) from exc

        self.global_step = int(state.get("global_step", self.global_step))


__all__ = ["PerceptionTrainer"]
