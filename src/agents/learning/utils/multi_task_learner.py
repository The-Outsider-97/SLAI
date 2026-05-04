"""Production-hardened multi-task loss coordination utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Union

from .config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Multi Task Learner")
printer = PrettyPrinter

class MultiTaskLearner(nn.Module):
    """
    Track task-specific losses and maintain adaptive task weights.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "initial_weights": None,
        "rebalance_strategy": "softmax",
        "rebalance_temp": 1.0,
        "history_window": 5,
        "normalize_weights": True,
        "min_weight": 0.0,
        "max_weight": 1.0,
    }

    def __init__(
        self,
        task_ids: Sequence[str],
        initial_weights: Optional[Mapping[str, float]] = None,
        rebalance_strategy: Optional[str] = None,
        rebalance_temp: Optional[float] = None,
    ) -> None:
        super().__init__()

        if not task_ids:
            raise ValueError("task_ids must be a non-empty sequence.")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task_ids must be unique.")

        self.config = load_global_config()
        self.learner_config = get_config_section('multi_task_learner')
        self.task_ids: List[str] = list(task_ids)
        self.loss_history: MutableMapping[str, List[float]] = defaultdict(list)
        self.history_window = int(self.learner_config.get("history_window", 5))
        self.rebalance_strategy = (rebalance_strategy or self.learner_config.get("rebalance_strategy", "softmax")).lower()
        self.rebalance_temp = float(rebalance_temp if rebalance_temp is not None else self.learner_config.get("rebalance_temp", 1.0))
        self.normalize_weights = bool(self.learner_config.get("normalize_weights", True))
        self.min_weight = float(self.learner_config.get("min_weight", 0.0))
        self.max_weight = float(self.learner_config.get("max_weight", 1.0))

        configured_weights = initial_weights or self.learner_config.get("initial_weights") or {}
        self.task_weights: Dict[str, float] = {task_id: 1.0 for task_id in self.task_ids}
        if configured_weights:
            self.set_weights(configured_weights)
        else:
            self._set_uniform_weights()

        logger.info("MultiTaskLearner initialized with %s tasks.", len(self.task_ids))

    def update_loss(self, task_id: str, loss_value: ScalarLike) -> None:
        """Record a new scalar loss value for a task."""
        if task_id not in self.task_weights:
            logger.warning("Unknown task_id %r ignored.", task_id)
            return

        scalar = self._to_float(loss_value)
        if not np.isfinite(scalar):
            raise ValueError(f"Loss for task '{task_id}' must be finite, got {scalar!r}.")
        self.loss_history[task_id].append(scalar)

    @staticmethod
    def _to_float(value: ScalarLike) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Expected scalar tensor for loss history updates.")
            return float(value.detach().cpu().item())
        return float(value)

    def get_weighted_loss(
        self,
        losses: Optional[Mapping[str, ScalarLike]] = None,
        default_to_latest: bool = True,
    ) -> Union[float, torch.Tensor]:
        """
        Compute a weighted aggregate loss.
        """
        if losses is None:
            losses = {
                task_id: self.loss_history[task_id][-1]
                for task_id in self.task_ids
                if self.loss_history[task_id]
            }

        tensor_terms = []
        float_total = 0.0

        for task_id in self.task_ids:
            if task_id in losses:
                loss_value = losses[task_id]
            elif default_to_latest and self.loss_history[task_id]:
                loss_value = self.loss_history[task_id][-1]
            else:
                continue

            weight = self.task_weights[task_id]
            if isinstance(loss_value, torch.Tensor):
                tensor_terms.append(loss_value * weight)
            else:
                float_total += weight * float(loss_value)

        if tensor_terms:
            total = tensor_terms[0]
            for term in tensor_terms[1:]:
                total = total + term
            if float_total:
                total = total + torch.as_tensor(float_total, dtype=total.dtype, device=total.device)
            return total

        return float_total

    def rebalance(self, strategy: Optional[str] = None) -> Dict[str, float]:
        """
        Adjust task weights based on recent performance.
        """
        strategy_name = (strategy or self.rebalance_strategy).lower()

        if strategy_name == "none":
            return self.get_weights()

        if strategy_name == "uniform":
            self._set_uniform_weights()
            return self.get_weights()

        avg_losses = []
        for task_id in self.task_ids:
            recent = self.loss_history[task_id][-self.history_window :]
            avg_losses.append(float(np.mean(recent)) if recent else 1.0)

        loss_tensor = torch.tensor(avg_losses, dtype=torch.float32)
        temperature = max(self.rebalance_temp, 1e-6)

        if strategy_name == "softmax":
            logits = loss_tensor / temperature
        elif strategy_name == "inverse_softmax":
            logits = -loss_tensor / temperature
        else:
            raise ValueError(f"Unknown rebalance strategy: {strategy_name}")

        weights = torch.softmax(logits, dim=0).tolist()
        self.task_weights = {task_id: float(weight) for task_id, weight in zip(self.task_ids, weights)}
        self._postprocess_weights()
        logger.info("Rebalanced task weights using %s: %s", strategy_name, self.task_weights)
        return self.get_weights()

    def set_weight(self, task_id: str, weight: float) -> None:
        if task_id not in self.task_weights:
            raise KeyError(f"Unknown task_id: {task_id}")
        self.task_weights[task_id] = float(weight)
        self._postprocess_weights()

    def set_weights(self, weights: Mapping[str, float]) -> None:
        for task_id, weight in weights.items():
            if task_id not in self.task_weights:
                raise KeyError(f"Unknown task_id: {task_id}")
            self.task_weights[task_id] = float(weight)
        self._postprocess_weights()

    def _set_uniform_weights(self) -> None:
        uniform = 1.0 / len(self.task_ids)
        self.task_weights = {task_id: uniform for task_id in self.task_ids}
        self._postprocess_weights()

    def _postprocess_weights(self) -> None:
        processed = {}
        for task_id, weight in self.task_weights.items():
            weight = float(weight)
            if not np.isfinite(weight):
                raise ValueError(f"Task weight for '{task_id}' must be finite.")
            processed[task_id] = min(self.max_weight, max(self.min_weight, weight))

        self.task_weights = processed

        if self.normalize_weights:
            total = sum(self.task_weights.values())
            if total <= 0.0:
                self._set_uniform_weights()
                return
            self.task_weights = {task_id: weight / total for task_id, weight in self.task_weights.items()}

    def reset(self) -> None:
        """Clear loss history while keeping task weights intact."""
        self.loss_history = defaultdict(list)
        logger.info("Loss histories reset for all tasks.")

    def get_weights(self) -> Dict[str, float]:
        return dict(self.task_weights)

    def get_recent_losses(self, window: Optional[int] = None) -> Dict[str, float]:
        current_window = self.history_window if window is None else max(1, int(window))
        return {
            task_id: (float(np.mean(self.loss_history[task_id][-current_window:])) if self.loss_history[task_id] else 0.0)
            for task_id in self.task_ids
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "task_ids": list(self.task_ids),
            "task_weights": dict(self.task_weights),
            "loss_history": {task_id: list(history) for task_id, history in self.loss_history.items()},
            "rebalance_strategy": self.rebalance_strategy,
            "rebalance_temp": self.rebalance_temp,
            "history_window": self.history_window,
            "normalize_weights": self.normalize_weights,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if list(snapshot.get("task_ids", [])) != self.task_ids:
            raise ValueError("Snapshot task_ids do not match this learner.")
        self.task_weights = {task_id: float(weight) for task_id, weight in snapshot["task_weights"].items()}
        self.loss_history = defaultdict(list, {task_id: list(history) for task_id, history in snapshot["loss_history"].items()})
        self.rebalance_strategy = str(snapshot.get("rebalance_strategy", self.rebalance_strategy))
        self.rebalance_temp = float(snapshot.get("rebalance_temp", self.rebalance_temp))
        self.history_window = int(snapshot.get("history_window", self.history_window))
        self.normalize_weights = bool(snapshot.get("normalize_weights", self.normalize_weights))
        self.min_weight = float(snapshot.get("min_weight", self.min_weight))
        self.max_weight = float(snapshot.get("max_weight", self.max_weight))
        self._postprocess_weights()


if __name__ == "__main__":
    print("\n=== Running Multi-Task Learner ===\n")
    printer.status("TEST", "Starting Multi-Task Learner tests", "info")
    task_ids = ["1354865"]
    learner = MultiTaskLearner(task_ids=task_ids)

    print("\n* * * * * Phase 1 Start * * * * *\n")
    print(learner)

    print("\n* * * * * Phase 2 * * * * *\n")
    values = np.array([1.2, 0.7, -0.1], dtype=np.float32)
    value = float(values[0])

    learner.update_loss(task_id=task_ids[0], loss_value=value)
    ok = len(learner.loss_history[task_ids[0]]) > 0
    printer.pretty("LOSS", "SUCCESS" if ok else "FAILURE", "success" if ok else "error")

    latest = True
    weighted_loss = learner.get_weighted_loss(default_to_latest=latest)
    re = learner.rebalance()
    printer.pretty("WEIGHTED", "SUCCESS" if weighted_loss else "FAILURE", "success" if weighted_loss else "error")
    printer.pretty("BALANCE", "SUCCESS" if re else "FAILURE", "success" if re else "error")

    print("\n=== All tests completed successfully! ===\n")