"""Production-ready strategy selector for SLAI learning agents.

This module coordinates a trainable meta-controller that maps task/state
embeddings to agent strategy choices. It intentionally keeps the original
surface area used by the learning stack:
- dependency injection for strategy maps, state embedders, policy networks,
  optimizers, and losses
- buffered supervised observations through ``observe``
- training through ``train_from_embeddings``
- inference through ``select_strategy``

The implementation hardens that surface with subsystem errors, shared helper
validation, deterministic tensor shaping, metric tracking, diagnostics, and
checkpoint support without introducing a separate config abstraction.
"""

from __future__ import annotations

import os
import random
import tempfile
import time
import torch  # type: ignore
import numpy as np  # type: ignore

from collections import deque
from pathlib import Path
from torch import nn  # type: ignore
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.learning_error import *
from .utils.learning_calculations import *
from .utils.learning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Strategy Selector")
printer = PrettyPrinter()

TensorLike = Union[torch.Tensor, np.ndarray, Sequence[float], Sequence[Sequence[float]]]
EmbeddingRecord = Tuple[torch.Tensor, torch.Tensor]


class StrategySelector:
    """Trainable meta-controller for choosing the best learning strategy.

    The selector is intentionally dependency-injected so it can integrate with
    the existing SLAI factory/agent stack without constructing concrete agents
    or policies itself.
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.strategy_config = get_config_section("strategy_selector") or {}

        self.task_embedding_dim = coerce_int(self.strategy_config.get("task_embedding_dim", 256), default=256, minimum=1)
        self.state_input_dim = coerce_int(self.strategy_config.get("state_input_dim", 256), default=256, minimum=1)
        self.min_batch = coerce_int(self.strategy_config.get("min_batch", 32), default=32, minimum=1)
        self.embedding_buffer_size = coerce_int(self.strategy_config.get("embedding_buffer_size", 2048), default=2048, minimum=1)
        self.gradient_clip_norm = coerce_float(self.strategy_config.get("gradient_clip_norm", 1.0), default=1.0, minimum=0.0)
        self.gradient_explosion_threshold = coerce_float(
            self.strategy_config.get("gradient_explosion_threshold", 1e4), default=1e4, minimum=0.0
        )
        self.resize_policy = str(self.strategy_config.get("resize_policy", "pad_truncate")).lower()
        self.unknown_strategy_policy = str(self.strategy_config.get("unknown_strategy_policy", "error")).lower()
        self.clear_buffer_after_train = coerce_bool(self.strategy_config.get("clear_buffer_after_train", True), default=True)
        self.detach_embeddings = coerce_bool(self.strategy_config.get("detach_embeddings", True), default=True)
        self.train_embedder = coerce_bool(self.strategy_config.get("train_embedder", False), default=False)
        self.confidence_threshold = coerce_float(self.strategy_config.get("confidence_threshold", 0.0), default=0.0, minimum=0.0, maximum=1.0)
        self.fallback_strategy = self.strategy_config.get("fallback_strategy")
        self.random_seed = self.strategy_config.get("seed")
        self.max_history = coerce_int(self.strategy_config.get("max_history", 200), default=200, minimum=1)

        if self.resize_policy not in {"pad_truncate", "error"}:
            raise InvalidConfigError("Unsupported resize policy", config_key="strategy_selector.resize_policy", received_value=self.resize_policy)
        if self.unknown_strategy_policy not in {"error", "warn_ignore"}:
            raise InvalidConfigError(
                "Unsupported unknown strategy policy",
                config_key="strategy_selector.unknown_strategy_policy",
                received_value=self.unknown_strategy_policy,
            )

        self.rng = random.Random(None if self.random_seed is None else int(self.random_seed))
        self.embedding_buffer: Deque[EmbeddingRecord] = deque(maxlen=self.embedding_buffer_size)
        self.training_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history)
        self.selection_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history)
        self.embedding_norm_stats = RunningStats()
        self.loss_stats = RunningStats()
        self.accuracy_stats = RunningStats()
        self.confidence_stats = RunningStats()
        self.calculations = LearningCalculations()

        self.agent_strategies_map: Optional[Dict[str, int]] = None
        self.index_to_strategy: Dict[int, str] = {}
        self.state_embedder: Optional[nn.Module] = None
        self.policy_net: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None

        logger.info("StrategySelector initialized | embedding_dim=%s min_batch=%s buffer=%s", self.task_embedding_dim, self.min_batch, self.embedding_buffer_size)

    # ------------------------------------------------------------------
    # Dependency injection
    # ------------------------------------------------------------------
    def set_agent_strategies_map(self, agent_strategies_map: Mapping[str, int]) -> None:
        """Set mapping from strategy name to policy-label index."""
        if not isinstance(agent_strategies_map, Mapping):
            raise InvalidConfigError("agent_strategies_map must be a mapping", received_value=type(agent_strategies_map).__name__)
        validate_non_empty_sequence(list(agent_strategies_map.keys()), "agent_strategies_map")

        cleaned: Dict[str, int] = {}
        labels: List[int] = []
        for name, label in agent_strategies_map.items():
            strategy_name = str(name).strip()
            if not strategy_name:
                raise InvalidConfigError("strategy names must be non-empty strings")
            label_int = coerce_int(label, default=-1, minimum=0)
            cleaned[strategy_name] = label_int
            labels.append(label_int)

        if len(set(labels)) != len(labels):
            raise InvalidConfigError("strategy labels must be unique", context={"labels": labels})

        self.agent_strategies_map = dict(cleaned)
        self.index_to_strategy = {label: name for name, label in cleaned.items()}
        logger.info("Registered %s strategies: %s", len(cleaned), sorted(cleaned))

    def set_state_embedder(self, state_embedder: nn.Module) -> None:
        """Set the module that maps raw state vectors to task embeddings."""
        if not isinstance(state_embedder, nn.Module):
            raise InvalidConfigError("state_embedder must be a torch.nn.Module", received_value=type(state_embedder).__name__)
        self.state_embedder = state_embedder

    def set_policy_network(
        self,
        policy_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: Union[str, torch.device],
    ) -> None:
        """Set the meta-controller policy network and training components."""
        if not isinstance(policy_net, nn.Module):
            raise InvalidConfigError("policy_net must be a torch.nn.Module", received_value=type(policy_net).__name__)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise InvalidConfigError("optimizer must be a torch.optim.Optimizer", received_value=type(optimizer).__name__)
        if not isinstance(loss_fn, nn.Module):
            raise InvalidConfigError("loss_fn must be a torch.nn.Module", received_value=type(loss_fn).__name__)
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.policy_net.to(self.device) # type: ignore

    # ------------------------------------------------------------------
    # Tensor preparation and guards
    # ------------------------------------------------------------------
    def _require_strategy_map(self) -> Dict[str, int]:
        if self.agent_strategies_map is None or not self.index_to_strategy:
            raise StrategySelectionError("agent strategies map is not set")
        return self.agent_strategies_map

    def _require_state_embedder(self) -> nn.Module:
        if self.state_embedder is None:
            raise StrategySelectionError("state embedder is not set")
        return self.state_embedder

    def _require_policy_components(self) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module, torch.device]:
        if self.policy_net is None or self.optimizer is None or self.loss_fn is None or self.device is None:
            raise StrategySelectionError("policy network, optimizer, loss_fn, and device must be set")
        return self.policy_net, self.optimizer, self.loss_fn, self.device

    @staticmethod
    def _first_module_device(module: nn.Module, fallback: Optional[torch.device] = None) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            return fallback or torch.device("cpu")

    def _to_2d_tensor(self, value: TensorLike, *, name: str, device: Optional[torch.device] = None) -> torch.Tensor:
        if value is None:
            raise ObservationShapeError(expected_shape="tensor-like", actual_shape="None")
        if torch.is_tensor(value):
            tensor = value.detach().clone() if self.detach_embeddings else value # type: ignore
            tensor = tensor.to(dtype=torch.float32) # type: ignore
        elif isinstance(value, np.ndarray):
            tensor = torch.as_tensor(value, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 0:
            raise ObservationShapeError(expected_shape=("features",), actual_shape=tuple(tensor.shape))
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.ndim != 2:
            raise ObservationShapeError(expected_shape=(None, "features"), actual_shape=tuple(tensor.shape))
        if torch.isnan(tensor).any():
            raise NaNException(f"NaN detected in {name}", location=name)
        if torch.isinf(tensor).any():
            raise InfException(f"Inf detected in {name}", location=name)
        return tensor.to(device=device) if device is not None else tensor

    def _fit_dim(self, tensor: torch.Tensor, target_dim: int, *, name: str) -> torch.Tensor:
        current_dim = int(tensor.shape[-1])
        if current_dim == int(target_dim):
            return tensor
        if self.resize_policy == "error":
            raise ObservationShapeError(expected_shape=(None, int(target_dim)), actual_shape=tuple(tensor.shape))
        if current_dim < int(target_dim):
            pad = torch.zeros((tensor.shape[0], int(target_dim) - current_dim), dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad], dim=-1)
        logger.debug("Truncating %s from %s to %s features", name, current_dim, target_dim)
        return tensor[:, : int(target_dim)]

    def _ensure_logits(self, output: torch.Tensor) -> torch.Tensor:
        if output.ndim == 1:
            output = output.unsqueeze(0)
        if output.ndim != 2:
            raise StrategySelectionError("policy output must be a 2D tensor", context={"shape": tuple(output.shape)})
        if torch.isnan(output).any():
            raise NaNException("NaN detected in strategy logits", location="strategy_logits")
        if torch.isinf(output).any():
            raise InfException("Inf detected in strategy logits", location="strategy_logits")
        strategy_count = len(self._require_strategy_map())
        if output.shape[-1] < strategy_count:
            raise StrategySelectionError(
                "policy output has fewer labels than registered strategies",
                context={"output_dim": int(output.shape[-1]), "strategies": strategy_count},
            )
        return output[:, :strategy_count]

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------
    def generate_task_embedding(self, state: TensorLike) -> torch.Tensor:
        """Generate a task embedding from a raw state vector."""
        embedder = self._require_state_embedder()
        device = self._first_module_device(embedder, self.device)
        state_tensor = self._to_2d_tensor(state, name="state", device=device)
        state_tensor = self._fit_dim(state_tensor, self.state_input_dim, name="state")

        previous_mode = embedder.training
        if not self.train_embedder:
            embedder.eval()
        with torch.set_grad_enabled(self.train_embedder):
            embedding = embedder(state_tensor)
        if previous_mode and not self.train_embedder:
            embedder.train()

        embedding = self._to_2d_tensor(embedding, name="task_embedding", device=device)
        embedding = self._fit_dim(embedding, self.task_embedding_dim, name="task_embedding")
        if embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        self.embedding_norm_stats.update(float(torch.linalg.vector_norm(embedding.detach()).cpu().item()))
        return embedding.detach() if self.detach_embeddings else embedding

    def observe(self, task_embedding: TensorLike, best_agent_strategy_name: str) -> bool:
        """Store a supervised embedding/strategy pair for meta-controller training."""
        strategy_map = self._require_strategy_map()
        if best_agent_strategy_name not in strategy_map:
            if self.unknown_strategy_policy == "warn_ignore":
                logger.warning("Ignoring unknown strategy %r. Valid: %s", best_agent_strategy_name, sorted(strategy_map))
                return False
            raise UnknownStrategyError(best_agent_strategy_name, strategy_map.keys())

        embedding = self._to_2d_tensor(task_embedding, name="task_embedding")
        embedding = self._fit_dim(embedding, self.task_embedding_dim, name="task_embedding")
        if embedding.shape[0] != 1:
            raise ObservationShapeError(expected_shape=(1, self.task_embedding_dim), actual_shape=tuple(embedding.shape))
        label = torch.tensor([int(strategy_map[best_agent_strategy_name])], dtype=torch.long)
        self.embedding_buffer.append((embedding.squeeze(0).detach().cpu(), label.detach().cpu()))
        return True

    def train_from_embeddings(self, batch_size: Optional[int] = None) -> Optional[Dict[str, float]]:
        """Train the meta-controller from buffered embeddings and return metrics."""
        policy_net, optimizer, loss_fn, device = self._require_policy_components()
        current_batch = coerce_int(batch_size if batch_size is not None else self.min_batch, default=self.min_batch, minimum=1)
        if len(self.embedding_buffer) < current_batch:
            logger.info("Deferring strategy-selector training: %s/%s samples", len(self.embedding_buffer), current_batch)
            return None

        started = time.perf_counter()
        records = self.rng.sample(list(self.embedding_buffer), current_batch)
        embeddings, labels = zip(*records)
        embedding_batch = torch.stack(list(embeddings)).to(device=device, dtype=torch.float32)
        label_batch = torch.cat(list(labels)).to(device=device, dtype=torch.long)

        policy_net.train()
        optimizer.zero_grad(set_to_none=True)
        logits = self._ensure_logits(policy_net(embedding_batch))
        loss = loss_fn(logits, label_batch)
        if torch.isnan(loss):
            raise NaNException("NaN detected in strategy training loss", location="strategy_training_loss")
        if torch.isinf(loss):
            raise InfException("Inf detected in strategy training loss", location="strategy_training_loss")
        loss.backward()

        gradients = [param.grad for param in policy_net.parameters() if param.grad is not None]
        grad_norm = self.calculations.gradient_global_norm(gradients)
        if self.gradient_explosion_threshold > 0.0 and grad_norm > self.gradient_explosion_threshold:
            raise GradientExplosionError(norm=grad_norm, threshold=self.gradient_explosion_threshold, layer_name="strategy_selector.policy_net")
        if self.gradient_clip_norm > 0.0:
            self.calculations.clip_gradients_by_global_norm(gradients, self.gradient_clip_norm)
        optimizer.step()

        accuracy = self.calculations.calculate_accuracy(logits.detach(), label_batch.detach())
        loss_value = float(loss.detach().cpu().item())
        self.loss_stats.update(loss_value)
        self.accuracy_stats.update(float(accuracy))
        self.calculations.update_performance(float(accuracy))
        metrics = {
            "loss": round_float(loss_value),
            "accuracy": round_float(float(accuracy)),
            "gradient_norm": round_float(grad_norm),
            "batch_size": float(current_batch),
            "buffer_size": float(len(self.embedding_buffer)),
            "elapsed_seconds": round_float(time.perf_counter() - started),
            "performance_trend": round_float(self.calculations.calculate_performance_trend()),
        }
        self.training_history.append(dict(metrics))
        if self.clear_buffer_after_train:
            self.embedding_buffer.clear()
        logger.info("Meta-controller training | loss=%.4f accuracy=%.2f%% grad_norm=%.4f", loss_value, accuracy * 100.0, grad_norm)
        return metrics

    def predict_distribution(self, state_embedding: TensorLike) -> Dict[str, float]:
        """Return a strategy probability distribution for a state/task embedding."""
        policy_net, _, _, device = self._require_policy_components()
        embedding = self._to_2d_tensor(state_embedding, name="state_embedding", device=device)
        embedding = self._fit_dim(embedding, self.task_embedding_dim, name="state_embedding")
        policy_net.eval()
        with torch.no_grad():
            logits = self._ensure_logits(policy_net(embedding))
            probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu()
        return {self.index_to_strategy[idx]: float(probabilities[idx].item()) for idx in sorted(self.index_to_strategy)}

    def select_strategy(self, state_embedding: TensorLike, return_details: bool = False) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Select the best strategy for a task/state embedding."""
        distribution = self.predict_distribution(state_embedding)
        if not distribution:
            raise StrategySelectionError("strategy distribution is empty")
        selected = max(distribution, key=distribution.get) # type: ignore
        confidence = float(distribution[selected])
        if self.fallback_strategy and confidence < self.confidence_threshold:
            strategy_map = self._require_strategy_map()
            if str(self.fallback_strategy) not in strategy_map:
                raise UnknownStrategyError(str(self.fallback_strategy), strategy_map.keys())
            selected = str(self.fallback_strategy)
        self.confidence_stats.update(confidence)
        details = {"selected_strategy": selected, "confidence": round_float(confidence), "distribution": distribution}
        self.selection_history.append(to_json_safe(details))
        return (selected, details) if return_details else selected

    def select_strategy_from_state(self, state: TensorLike, return_details: bool = False) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Generate an embedding from raw state and select a strategy."""
        embedding = self.generate_task_embedding(state)
        return self.select_strategy(embedding, return_details=return_details)

    # ------------------------------------------------------------------
    # State, diagnostics, and persistence
    # ------------------------------------------------------------------
    def snapshot(self, include_buffer: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "strategy_config": dict(self.strategy_config),
            "agent_strategies_map": dict(self.agent_strategies_map or {}),
            "training_history": list(self.training_history),
            "selection_history": list(self.selection_history),
            "embedding_norm_stats": to_json_safe(self.embedding_norm_stats.snapshot()),
            "loss_stats": to_json_safe(self.loss_stats.snapshot()),
            "accuracy_stats": to_json_safe(self.accuracy_stats.snapshot()),
            "confidence_stats": to_json_safe(self.confidence_stats.snapshot()),
        }
        if include_buffer:
            payload["embedding_buffer"] = [(emb.clone().cpu(), label.clone().cpu()) for emb, label in self.embedding_buffer]
        return payload

    def restore(self, snapshot: Mapping[str, Any], restore_buffer: bool = True) -> None:
        validate_required_keys(snapshot, ["agent_strategies_map"], name="strategy_selector_snapshot")
        self.set_agent_strategies_map(snapshot.get("agent_strategies_map", {}))
        self.training_history = deque(list(snapshot.get("training_history", [])), maxlen=self.max_history)
        self.selection_history = deque(list(snapshot.get("selection_history", [])), maxlen=self.max_history)
        if restore_buffer and "embedding_buffer" in snapshot:
            self.embedding_buffer.clear()
            for emb, label in snapshot.get("embedding_buffer", []):
                self.embedding_buffer.append((self._to_2d_tensor(emb, name="restored_embedding").squeeze(0).cpu(), torch.as_tensor(label, dtype=torch.long).view(1).cpu()))

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "version": 1,
                "selector": self.snapshot(include_buffer=True),
                "policy_state_dict": self.policy_net.state_dict() if self.policy_net is not None else None,
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
                "state_embedder_state_dict": self.state_embedder.state_dict() if self.state_embedder is not None else None,
            }
            fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
            with os.fdopen(fd, "wb") as fh:
                torch.save(state, fh)
            os.replace(tmp_path, target)
            return target
        except Exception as exc:
            raise CheckpointError(str(target), operation="save", cause=exc) from exc

    def load_checkpoint(self, path: Union[str, Path], map_location: Optional[Union[str, torch.device]] = None, strict: bool = True) -> Dict[str, Any]:
        source = Path(path)
        try:
            checkpoint = torch.load(source, map_location=map_location)
            if not isinstance(checkpoint, Mapping):
                raise CheckpointError(str(source), operation="load", message="Strategy selector checkpoint is not a mapping")
            self.restore(checkpoint.get("selector", checkpoint), restore_buffer=True)
            if self.policy_net is not None and checkpoint.get("policy_state_dict") is not None:
                self.policy_net.load_state_dict(checkpoint["policy_state_dict"], strict=strict)
            if self.optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.state_embedder is not None and checkpoint.get("state_embedder_state_dict") is not None:
                self.state_embedder.load_state_dict(checkpoint["state_embedder_state_dict"], strict=strict)
            return dict(checkpoint)
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(str(source), operation="load", cause=exc) from exc

    def clear_buffer(self) -> None:
        self.embedding_buffer.clear()

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "task_embedding_dim": self.task_embedding_dim,
            "state_input_dim": self.state_input_dim,
            "min_batch": self.min_batch,
            "buffer_size": len(self.embedding_buffer),
            "buffer_capacity": self.embedding_buffer.maxlen,
            "strategies": sorted((self.agent_strategies_map or {}).keys()),
            "has_state_embedder": self.state_embedder is not None,
            "has_policy_net": self.policy_net is not None,
            "device": str(self.device) if self.device is not None else None,
            "loss_stats": to_json_safe(self.loss_stats.snapshot()),
            "accuracy_stats": to_json_safe(self.accuracy_stats.snapshot()),
            "confidence_stats": to_json_safe(self.confidence_stats.snapshot()),
            "embedding_norm_stats": to_json_safe(self.embedding_norm_stats.snapshot()),
            "recent_training": list(self.training_history)[-5:],
            "recent_selections": list(self.selection_history)[-5:],
        }


if __name__ == "__main__":
    print("\n=== Running Strategy Selector ===\n")
    printer.status("TEST", "Strategy Selector initialized", "info")
    torch.manual_seed(7); np.random.seed(7); random.seed(7)
    selector = StrategySelector()
    strategies = {"dqn": 0, "maml": 1, "rsi": 2}
    selector.set_agent_strategies_map(strategies)
    dim = selector.task_embedding_dim
    inp = selector.state_input_dim
    embedder = nn.Sequential(nn.Linear(inp, 64), nn.ReLU(), nn.Linear(64, dim))
    policy = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, len(strategies)))
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    selector.set_state_embedder(embedder)
    selector.set_policy_network(policy, opt, nn.CrossEntropyLoss(), torch.device("cpu"))
    for i, name in enumerate(strategies):
        for _ in range(max(2, selector.min_batch // len(strategies))):
            emb = selector.generate_task_embedding(np.random.randn(inp).astype(np.float32))
            assert selector.observe(emb, name)
    metrics = selector.train_from_embeddings(batch_size=min(selector.min_batch, len(selector.embedding_buffer)))
    assert metrics and metrics["loss"] >= 0.0
    emb = selector.generate_task_embedding(np.random.randn(inp).astype(np.float32))
    chosen, details = selector.select_strategy(emb, return_details=True)
    assert chosen in strategies and details["confidence"] >= 0.0 # type: ignore
    path = Path("strategy_selector_test.pt")
    selector.save_checkpoint(path)
    restored = StrategySelector()
    restored.set_agent_strategies_map(strategies)
    restored.set_state_embedder(embedder)
    restored.set_policy_network(policy, opt, nn.CrossEntropyLoss(), torch.device("cpu"))
    restored.load_checkpoint(path)
    path.unlink(missing_ok=True)
    printer.status("TEST", "Strategy Selector verified", "success")
    print("\n=== Test ran successfully ===\n")
