"""Production-ready multi-task loss coordination utilities.

This module coordinates task-specific losses for multi-objective learning flows.
It keeps the original public API small and predictable while adding production
concerns expected by the SLAI learning subsystem:

- bounded task loss histories and per-task running statistics
- finite-value validation through shared learning errors/helpers
- tensor-safe weighted loss aggregation without breaking gradients
- configurable adaptive rebalancing strategies
- snapshot/restore and checkpoint persistence
- task lifecycle management for dynamic learning workloads
"""

from __future__ import annotations

import copy
import time
import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Mapping, MutableMapping, Optional, Sequence, Union, overload

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.learning_error import *
from ..utils.learning_calculations import *
from ..utils.learning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Multi Task Learner")
printer = PrettyPrinter()

LossLike = ScalarLike
HistoryMap = MutableMapping[str, Deque[float]]


class MultiTaskLearner(nn.Module):
    """Track task-specific losses and maintain adaptive task weights.

    The learner is intentionally independent from any concrete model so it can
    be shared by policy-gradient, value-based, supervised auxiliary, and
    meta-learning components. The public API remains compatible with the
    original implementation while adding stricter validation and richer runtime
    diagnostics.
    """

    SUPPORTED_REBALANCE_STRATEGIES = {
        "none",
        "uniform",
        "softmax",
        "inverse_softmax",
        "proportional",
        "inverse_proportional",
        "dwa",
        "uncertainty",
    }
    SUPPORTED_UNKNOWN_TASK_POLICIES = {"error", "warn_ignore", "auto_add"}
    SUPPORTED_LOSS_REDUCTIONS = {"mean", "sum", "none"}

    def __init__(
        self,
        task_ids: Sequence[str],
        initial_weights: Optional[Mapping[str, float]] = None,
        rebalance_strategy: Optional[str] = None,
        rebalance_temp: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.config = load_global_config()
        self.learner_config = get_config_section("multi_task_learner") or {}

        self.task_ids: List[str] = self._normalise_task_ids(task_ids)
        self.history_window = coerce_int(self.learner_config.get("history_window"), 5, minimum=1)
        self.max_history_per_task = coerce_int(self.learner_config.get("max_history_per_task"), 500, minimum=1)
        self.rebalance_strategy = str(
            rebalance_strategy or self.learner_config.get("rebalance_strategy", "softmax")
        ).lower()
        self.rebalance_temp = coerce_float(
            rebalance_temp if rebalance_temp is not None else self.learner_config.get("rebalance_temp"),
            1.0,
            minimum=1e-12,
        )
        self.normalize_weights = coerce_bool(self.learner_config.get("normalize_weights"), True)
        self.min_weight = coerce_float(self.learner_config.get("min_weight"), 0.0, minimum=0.0)
        self.max_weight = coerce_float(self.learner_config.get("max_weight"), 1.0, minimum=0.0)
        self.smoothing_alpha = coerce_float(self.learner_config.get("smoothing_alpha"), 0.0, minimum=0.0, maximum=1.0)
        self.max_weight_delta = self._optional_positive_float(self.learner_config.get("max_weight_delta"), "max_weight_delta")
        self.unknown_task_policy = str(self.learner_config.get("unknown_task_policy", "warn_ignore")).lower()
        self.loss_reduction = str(self.learner_config.get("loss_reduction", "mean")).lower()
        self.auto_rebalance_on_update = coerce_bool(self.learner_config.get("auto_rebalance_on_update"), False)
        self.rebalance_interval = coerce_int(self.learner_config.get("rebalance_interval"), 1, minimum=1)
        self.round_precision = coerce_int(self.learner_config.get("round_precision"), 8, minimum=0)
        self.checkpoint_dir = Path(str(self.learner_config.get("checkpoint_dir")))

        self._validate_runtime_config()

        self.loss_history: HistoryMap = defaultdict(self._new_history)
        self.loss_stats: Dict[str, RunningStats] = {task_id: RunningStats() for task_id in self.task_ids}
        self.rebalance_count = 0
        self.update_count = 0
        self.last_rebalance_time: Optional[float] = None

        self.task_weights: Dict[str, float] = {task_id: 1.0 for task_id in self.task_ids}
        configured_weights = initial_weights or self.learner_config.get("initial_weights") or {}
        if configured_weights:
            self.set_weights(configured_weights, strict=False)
        else:
            self._set_uniform_weights()

        logger.info("MultiTaskLearner initialized with %s tasks.", len(self.task_ids))

    # ------------------------------------------------------------------
    # Loss ingestion and aggregation
    # ------------------------------------------------------------------
    def update_loss(self, task_id: str, loss_value: LossLike) -> None:
        """Record a finite scalar loss for one task."""
        resolved_task = self._resolve_task_id(task_id)
        if resolved_task is None:
            return

        scalar = self._to_float(loss_value, name=f"loss[{resolved_task}]")
        self.loss_history[resolved_task].append(scalar)
        self.loss_stats.setdefault(resolved_task, RunningStats()).update(scalar)
        self.update_count += 1

        if self.auto_rebalance_on_update and self.update_count % self.rebalance_interval == 0:
            self.rebalance()

    def update_losses(self, losses: Mapping[str, LossLike]) -> None:
        """Record multiple task losses in one call."""
        validate_type(losses, "losses", Mapping)
        for task_id, loss_value in losses.items():
            self.update_loss(str(task_id), loss_value)

    def get_weighted_loss(
        self,
        losses: Optional[Mapping[str, LossLike]] = None,
        default_to_latest: bool = True,
    ) -> Union[float, torch.Tensor]:
        """Compute a weighted aggregate loss.

        Tensor losses keep their autograd graph. Python/numpy scalar losses are
        folded into the result as constants. Missing task losses may fall back to
        the latest recorded history value when ``default_to_latest`` is true.
        """
        loss_map = self._resolve_loss_map(losses, default_to_latest=default_to_latest)
        tensor_total: Optional[torch.Tensor] = None
        float_total = 0.0

        for task_id in self.task_ids:
            if task_id not in loss_map:
                continue
            weight = float(self.task_weights.get(task_id, 0.0))
            loss_value = loss_map[task_id]

            if isinstance(loss_value, torch.Tensor):
                reduced = self._reduce_tensor_loss(loss_value, task_id)
                term = reduced * weight
                tensor_total = term if tensor_total is None else tensor_total + term
            else:
                scalar = self._to_float(loss_value, name=f"loss[{task_id}]")
                float_total += weight * scalar

        if tensor_total is not None:
            if float_total:
                tensor_total = tensor_total + torch.as_tensor(float_total, dtype=tensor_total.dtype, device=tensor_total.device)
            return tensor_total
        return float(float_total)

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------
    def rebalance(self, strategy: Optional[str] = None) -> Dict[str, float]:
        """Adjust task weights based on recent task losses."""
        strategy_name = str(strategy or self.rebalance_strategy).lower()
        if strategy_name not in self.SUPPORTED_REBALANCE_STRATEGIES:
            raise UnknownStrategyError(strategy_name, sorted(self.SUPPORTED_REBALANCE_STRATEGIES))

        previous = self.get_weights()
        if strategy_name == "none":
            return previous
        if strategy_name == "uniform":
            self._set_uniform_weights()
            return self.get_weights()

        avg_losses = self._recent_average_losses(self.history_window, default=1.0)
        if strategy_name == "softmax":
            target = self._softmax_weights(avg_losses, inverse=False)
        elif strategy_name == "inverse_softmax":
            target = self._softmax_weights(avg_losses, inverse=True)
        elif strategy_name == "proportional":
            target = self._normalised_weight_map(avg_losses)
        elif strategy_name == "inverse_proportional":
            inv = {k: safe_divide(1.0, max(v, EPSILON), default=1.0) for k, v in avg_losses.items()}
            target = self._normalised_weight_map(inv)
        elif strategy_name == "dwa":
            target = self._dynamic_weight_average()
        elif strategy_name == "uncertainty":
            target = self._uncertainty_weights()
        else:  # protected by validation above
            raise UnknownStrategyError(strategy_name, sorted(self.SUPPORTED_REBALANCE_STRATEGIES))

        self.task_weights = self._blend_and_limit_weights(previous, target)
        self._postprocess_weights()
        self.rebalance_count += 1
        self.last_rebalance_time = time.time()
        logger.info("Rebalanced task weights using %s: %s", strategy_name, self.task_weights)
        return self.get_weights()

    def _softmax_weights(self, losses: Mapping[str, float], *, inverse: bool) -> Dict[str, float]:
        values = torch.as_tensor([float(losses[t]) for t in self.task_ids], dtype=torch.float32)
        logits = (-values if inverse else values) / max(self.rebalance_temp, EPSILON)
        weights = torch.softmax(logits, dim=0).detach().cpu().tolist()
        return {task_id: float(weight) for task_id, weight in zip(self.task_ids, weights)}

    def _dynamic_weight_average(self) -> Dict[str, float]:
        ratios: Dict[str, float] = {}
        for task_id in self.task_ids:
            recent, previous = self._two_window_means(task_id)
            ratios[task_id] = safe_divide(recent, previous, default=1.0)
        return self._softmax_weights(ratios, inverse=False)

    def _uncertainty_weights(self) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for task_id in self.task_ids:
            values = list(self.loss_history[task_id])[-self.history_window :]
            if len(values) < 2:
                scores[task_id] = 1.0
                continue
            mean = abs(float(np.mean(values)))
            std = float(np.std(values))
            scores[task_id] = std + safe_divide(std, mean + EPSILON, default=0.0)
        return self._normalised_weight_map(scores)

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------
    def set_weight(self, task_id: str, weight: float) -> None:
        resolved_task = self._resolve_task_id(task_id, strict=True)
        self.task_weights[resolved_task] = self._validated_weight(weight, resolved_task)
        self._postprocess_weights()

    def set_weights(self, weights: Mapping[str, float], *, strict: bool = True) -> None:
        validate_type(weights, "weights", Mapping)
        for task_id, weight in weights.items():
            resolved_task = self._resolve_task_id(str(task_id), strict=strict)
            if resolved_task is not None:
                self.task_weights[resolved_task] = self._validated_weight(weight, resolved_task)
        self._postprocess_weights()

    def get_weights(self) -> Dict[str, float]:
        return {task_id: round_float(self.task_weights[task_id], self.round_precision) for task_id in self.task_ids}

    def add_task(self, task_id: str, weight: Optional[float] = None) -> None:
        task = self._clean_task_id(task_id)
        if task in self.task_weights:
            return
        self.task_ids.append(task)
        self.loss_history[task] = self._new_history()
        self.loss_stats[task] = RunningStats()
        self.task_weights[task] = 1.0 / max(1, len(self.task_ids)) if weight is None else self._validated_weight(weight, task)
        self._postprocess_weights()
        logger.info("Added multi-task learner task %r.", task)

    def remove_task(self, task_id: str) -> None:
        task = self._resolve_task_id(task_id, strict=True)
        if len(self.task_ids) <= 1:
            raise InvalidConfigError("Cannot remove the final task from MultiTaskLearner.", context={"task_id": task})
        self.task_ids.remove(task)
        self.task_weights.pop(task, None)
        self.loss_history.pop(task, None)
        self.loss_stats.pop(task, None)
        self._postprocess_weights()
        logger.info("Removed multi-task learner task %r.", task)

    def _set_uniform_weights(self) -> None:
        uniform = 1.0 / len(self.task_ids)
        self.task_weights = {task_id: uniform for task_id in self.task_ids}
        self._postprocess_weights()

    def _postprocess_weights(self) -> None:
        if self.max_weight < self.min_weight:
            raise InvalidConfigError("max_weight must be >= min_weight.", context={"min_weight": self.min_weight, "max_weight": self.max_weight})

        processed: Dict[str, float] = {}
        for task_id in self.task_ids:
            weight = self._validated_weight(self.task_weights.get(task_id, 0.0), task_id)
            processed[task_id] = clamp(weight, self.min_weight, self.max_weight)

        if self.normalize_weights:
            values = normalize_probabilities([processed[task_id] for task_id in self.task_ids])
            processed = {task_id: values[idx] for idx, task_id in enumerate(self.task_ids)}

        self.task_weights = processed

    # ------------------------------------------------------------------
    # Inspection, snapshotting, and persistence
    # ------------------------------------------------------------------
    def reset(self, *, reset_weights: bool = False) -> None:
        """Clear loss history and optionally reset weights to uniform."""
        self.loss_history = defaultdict(self._new_history)
        self.loss_stats = {task_id: RunningStats() for task_id in self.task_ids}
        self.update_count = 0
        if reset_weights:
            self._set_uniform_weights()
        logger.info("Loss histories reset for all tasks.")

    def get_recent_losses(self, window: Optional[int] = None) -> Dict[str, float]:
        current_window = self.history_window if window is None else max(1, int(window))
        return self._recent_average_losses(current_window, default=0.0)

    def get_loss_trends(self, window: Optional[int] = None) -> Dict[str, float]:
        current_window = self.history_window if window is None else max(2, int(window))
        trends: Dict[str, float] = {}
        for task_id in self.task_ids:
            values = list(self.loss_history[task_id])[-current_window:]
            if len(values) < 2:
                trends[task_id] = 0.0
                continue
            x = np.arange(len(values), dtype=np.float64)
            y = np.asarray(values, dtype=np.float64)
            trends[task_id] = round_float(float(np.polyfit(x, y, 1)[0]), self.round_precision)
        return trends

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "task_count": len(self.task_ids),
            "task_ids": list(self.task_ids),
            "weights": self.get_weights(),
            "recent_losses": self.get_recent_losses(),
            "loss_trends": self.get_loss_trends(),
            "history_lengths": {task_id: len(self.loss_history[task_id]) for task_id in self.task_ids},
            "stats": {task_id: self.loss_stats[task_id].snapshot().__dict__ for task_id in self.task_ids},
            "rebalance_strategy": self.rebalance_strategy,
            "rebalance_count": self.rebalance_count,
            "update_count": self.update_count,
            "last_rebalance_time": self.last_rebalance_time,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "task_ids": list(self.task_ids),
            "task_weights": dict(self.task_weights),
            "loss_history": {task_id: list(self.loss_history[task_id]) for task_id in self.task_ids},
            "rebalance_strategy": self.rebalance_strategy,
            "rebalance_temp": self.rebalance_temp,
            "history_window": self.history_window,
            "max_history_per_task": self.max_history_per_task,
            "normalize_weights": self.normalize_weights,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "smoothing_alpha": self.smoothing_alpha,
            "max_weight_delta": self.max_weight_delta,
            "unknown_task_policy": self.unknown_task_policy,
            "loss_reduction": self.loss_reduction,
            "rebalance_count": self.rebalance_count,
            "update_count": self.update_count,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        validate_type(snapshot, "snapshot", Mapping)
        restored_tasks = self._normalise_task_ids(snapshot.get("task_ids", []))
        if restored_tasks != self.task_ids:
            raise InvalidConfigError(
                "Snapshot task_ids do not match this learner.",
                context={"expected": self.task_ids, "received": restored_tasks},
            )

        self.rebalance_strategy = str(snapshot.get("rebalance_strategy", self.rebalance_strategy)).lower()
        self.rebalance_temp = coerce_float(snapshot.get("rebalance_temp", self.rebalance_temp), self.rebalance_temp, minimum=1e-12)
        self.history_window = coerce_int(snapshot.get("history_window", self.history_window), self.history_window, minimum=1)
        self.max_history_per_task = coerce_int(snapshot.get("max_history_per_task", self.max_history_per_task), self.max_history_per_task, minimum=1)
        self.normalize_weights = coerce_bool(snapshot.get("normalize_weights", self.normalize_weights), self.normalize_weights)
        self.min_weight = coerce_float(snapshot.get("min_weight", self.min_weight), self.min_weight, minimum=0.0)
        self.max_weight = coerce_float(snapshot.get("max_weight", self.max_weight), self.max_weight, minimum=0.0)
        self.smoothing_alpha = coerce_float(snapshot.get("smoothing_alpha", self.smoothing_alpha), self.smoothing_alpha, minimum=0.0, maximum=1.0)
        self.max_weight_delta = self._optional_positive_float(snapshot.get("max_weight_delta", self.max_weight_delta), "max_weight_delta")
        self.unknown_task_policy = str(snapshot.get("unknown_task_policy", self.unknown_task_policy)).lower()
        self.loss_reduction = str(snapshot.get("loss_reduction", self.loss_reduction)).lower()
        self.rebalance_count = int(snapshot.get("rebalance_count", self.rebalance_count))
        self.update_count = int(snapshot.get("update_count", self.update_count))
        self._validate_runtime_config()

        history_payload = snapshot.get("loss_history", {})
        validate_type(history_payload, "snapshot.loss_history", Mapping)
        self.loss_history = defaultdict(self._new_history)
        self.loss_stats = {task_id: RunningStats() for task_id in self.task_ids}
        for task_id in self.task_ids:
            for value in history_payload.get(task_id, []):
                scalar = self._to_float(value, name=f"snapshot.loss_history[{task_id}]")
                self.loss_history[task_id].append(scalar)
                self.loss_stats[task_id].update(scalar)

        weights = snapshot.get("task_weights", {})
        validate_type(weights, "snapshot.task_weights", Mapping)
        self.task_weights = {task_id: self._validated_weight(weights.get(task_id, 0.0), task_id) for task_id in self.task_ids}
        self._postprocess_weights()

    def save_checkpoint(self, path: Optional[Union[str, Path]] = None) -> Path:
        checkpoint_path = Path(path) if path is not None else self.checkpoint_dir / "multi_task_learner.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(self.snapshot(), checkpoint_path)
        except Exception as exc:
            raise CheckpointError(
                str(checkpoint_path),                      # path (positional)
                message="Failed to save MultiTaskLearner checkpoint.",
                cause=exc
            ) from exc
        return checkpoint_path

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise CheckpointError(
                str(checkpoint_path),                      # path (positional)
                message="MultiTaskLearner checkpoint does not exist."
            )
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.restore(payload)
        except LearningError:
            raise
        except Exception as exc:
            raise CheckpointError(
                str(checkpoint_path),                      # path (positional)
                message="Failed to load MultiTaskLearner checkpoint.",
                cause=exc
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_runtime_config(self) -> None:
        if self.rebalance_strategy not in self.SUPPORTED_REBALANCE_STRATEGIES:
            raise UnknownStrategyError(self.rebalance_strategy, sorted(self.SUPPORTED_REBALANCE_STRATEGIES))
        if self.unknown_task_policy not in self.SUPPORTED_UNKNOWN_TASK_POLICIES:
            raise InvalidConfigError("Unsupported unknown_task_policy.", context={"received": self.unknown_task_policy, "allowed": sorted(self.SUPPORTED_UNKNOWN_TASK_POLICIES)})
        if self.loss_reduction not in self.SUPPORTED_LOSS_REDUCTIONS:
            raise InvalidConfigError("Unsupported loss_reduction.", context={"received": self.loss_reduction, "allowed": sorted(self.SUPPORTED_LOSS_REDUCTIONS)})
        validate_positive(self.history_window, "history_window")
        validate_positive(self.max_history_per_task, "max_history_per_task")
        validate_positive(self.rebalance_temp, "rebalance_temp")
        validate_in_range(self.smoothing_alpha, "smoothing_alpha", 0.0, 1.0)
        validate_non_negative(self.min_weight, "min_weight")
        validate_non_negative(self.max_weight, "max_weight")
        if self.max_weight < self.min_weight:
            raise InvalidConfigError("max_weight must be >= min_weight.", context={"min_weight": self.min_weight, "max_weight": self.max_weight})

    def _new_history(self) -> Deque[float]:
        return deque(maxlen=self.max_history_per_task)

    @classmethod
    def _normalise_task_ids(cls, task_ids: Sequence[str]) -> List[str]:
        validate_non_empty_sequence(task_ids, "task_ids")
        cleaned = [cls._clean_task_id(task_id) for task_id in task_ids]
        if len(set(cleaned)) != len(cleaned):
            raise InvalidConfigError("task_ids must be unique.", context={"task_ids": cleaned})
        return cleaned

    @staticmethod
    def _clean_task_id(task_id: Any) -> str:
        task = str(task_id).strip()
        if not task:
            raise InvalidConfigError("task_id must be a non-empty string.")
        return task

    @overload
    def _resolve_task_id(self, task_id: str, *, strict: Literal[True]) -> str: ...

    @overload
    def _resolve_task_id(self, task_id: str, *, strict: Literal[False] = False) -> Optional[str]: ...

    def _resolve_task_id(self, task_id: str, *, strict: bool = False) -> Optional[str]:
        task = self._clean_task_id(task_id)
        if task in self.task_weights:
            return task
        if strict or self.unknown_task_policy == "error":
            raise InvalidConfigError("Unknown task_id.", context={"task_id": task, "known_tasks": self.task_ids})
        if self.unknown_task_policy == "auto_add":
            self.add_task(task)
            return task
        logger.warning("Unknown task_id %r ignored.", task)
        return None

    def _resolve_loss_map(self, losses: Optional[Mapping[str, LossLike]], *, default_to_latest: bool) -> Dict[str, LossLike]:
        if losses is not None:
            validate_type(losses, "losses", Mapping)
            unknown = [str(task_id) for task_id in losses if str(task_id) not in self.task_weights]
            if unknown:
                for task_id in unknown:
                    self._resolve_task_id(task_id)
            loss_map = {str(task_id): value for task_id, value in losses.items() if str(task_id) in self.task_weights}
        else:
            loss_map = {}

        if default_to_latest:
            for task_id in self.task_ids:
                if task_id not in loss_map and self.loss_history[task_id]:
                    loss_map[task_id] = self.loss_history[task_id][-1]
        return loss_map

    def _reduce_tensor_loss(self, loss_value: torch.Tensor, task_id: str) -> torch.Tensor:
        if not torch.isfinite(loss_value.detach()).all().item():
            if torch.isnan(loss_value.detach()).any().item():
                raise NaNException(f"NaN detected in tensor loss for task '{task_id}'.", location=f"loss[{task_id}]")
            raise InfException(f"Inf detected in tensor loss for task '{task_id}'.", location=f"loss[{task_id}]")
        if self.loss_reduction == "mean":
            return loss_value.mean()
        if self.loss_reduction == "sum":
            return loss_value.sum()
        if loss_value.numel() != 1:
            raise InvalidConfigError("loss_reduction='none' requires scalar tensor losses.", context={"task_id": task_id, "shape": tuple(loss_value.shape)})
        return loss_value.reshape(())

    @staticmethod
    def _to_float(value: LossLike, *, name: str = "value") -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1: # type: ignore
                raise InvalidConfigError(
                    "Expected scalar tensor for loss history updates.",
                    context={"name": name, "shape": tuple(value.shape)} # type: ignore
                )
            scalar = float(value.detach().cpu().item()) # type: ignore
        else:
            try:
                scalar = float(value)
            except (TypeError, ValueError) as exc:
                raise InvalidConfigError(
                    f"{name} must be numeric.",
                    context={"received": repr(value)},
                    cause=exc
                ) from exc
        validate_finite(scalar, name)
        return scalar

    def _validated_weight(self, weight: Any, task_id: str) -> float:
        scalar = self._to_float(weight, name=f"task_weight[{task_id}]")
        validate_non_negative(scalar, f"task_weight[{task_id}]")
        return scalar

    @staticmethod
    def _optional_positive_float(value: Any, name: str) -> Optional[float]:
        if value is None:
            return None
        scalar = coerce_float(value, default=-1.0)
        validate_positive(scalar, name)
        return scalar

    def _recent_average_losses(self, window: int, *, default: float) -> Dict[str, float]:
        current_window = max(1, int(window))
        averages: Dict[str, float] = {}
        for task_id in self.task_ids:
            recent = list(self.loss_history[task_id])[-current_window:]
            averages[task_id] = float(np.mean(recent)) if recent else float(default)
        return averages

    def _two_window_means(self, task_id: str) -> tuple[float, float]:
        values = list(self.loss_history[task_id])
        window = max(1, self.history_window)
        if len(values) < window * 2:
            return (float(np.mean(values[-window:])) if values else 1.0, 1.0)
        return float(np.mean(values[-window:])), float(np.mean(values[-2 * window : -window]))

    def _normalised_weight_map(self, scores: Mapping[str, float]) -> Dict[str, float]:
        values = normalize_probabilities([max(0.0, float(scores.get(task_id, 0.0))) for task_id in self.task_ids])
        return {task_id: float(values[idx]) for idx, task_id in enumerate(self.task_ids)}

    def _blend_and_limit_weights(self, previous: Mapping[str, float], target: Mapping[str, float]) -> Dict[str, float]:
        alpha = self.smoothing_alpha
        blended: Dict[str, float] = {}
        for task_id in self.task_ids:
            old = float(previous.get(task_id, 0.0))
            new = float(target.get(task_id, old))
            value = (alpha * old) + ((1.0 - alpha) * new)
            if self.max_weight_delta is not None:
                value = clamp(value, old - self.max_weight_delta, old + self.max_weight_delta)
            blended[task_id] = value
        return blended


if __name__ == "__main__":
    print("\n=== Running Multi Task Learner ===\n")
    printer.status("TEST", "Multi Task Learner initialized", "info")

    tasks = ["policy", "value", "novelty"]
    learner = MultiTaskLearner(tasks, rebalance_strategy="softmax", rebalance_temp=0.75)
    for i in range(4):
        learner.update_losses({"policy": 1.0 + i, "value": 0.5 + i * 0.2, "novelty": 0.2 + i * 0.1})
    weights = learner.rebalance()
    assert set(weights) == set(tasks) and abs(sum(weights.values()) - 1.0) < 1e-6

    losses = {
        "policy": torch.tensor(2.0, requires_grad=True),
        "value": torch.tensor(1.0, requires_grad=True),
        "novelty": 0.5,
    }
    total = learner.get_weighted_loss(losses)
    assert isinstance(total, torch.Tensor) and torch.isfinite(total)
    total.backward() # type: ignore

    snap = learner.snapshot()
    clone = MultiTaskLearner(tasks)
    clone.restore(snap)
    assert clone.get_weights() == learner.get_weights()
    assert clone.get_recent_losses()["policy"] > 0

    path = learner.save_checkpoint("/tmp/multi_task_learner_test.pt")
    restored = MultiTaskLearner(tasks)
    restored.load_checkpoint(path)
    assert restored.get_weights() == learner.get_weights()
    printer.status("TEST", "Weighted loss, rebalance, snapshot, and checkpoint passed", "success")

    print("\n=== Test ran successfully ===\n")
