"""Production-hardened PyTorch state processor for environment observations."""

from __future__ import annotations

import torch
import torch.nn as nn

from typing import Any, Optional, Sequence, Tuple, Union

from .config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("State Processor")
printer = PrettyPrinter

class StateProcessor(nn.Module):
    """
    Convert raw environment observations into consistent float tensors.

    This module remains intentionally lightweight, but now handles:
    - Gym/Gymnasium `(obs, info)` tuples safely
    - nested Python containers
    - tensors, numpy-like arrays, and scalars
    - optional normalization using explicit or inferred bounds
    - optional clipping of normalized outputs
    """

    def __init__(
        self,
        env: Optional[Any] = None,
        normalize: Optional[bool] = None,
        low: Optional[torch.Tensor] = None,
        high: Optional[torch.Tensor] = None,
        clip_normalized: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.env = env
        self.config = load_global_config()
        self.sp_config = get_config_section('state_processor_config') or {}

        self.normalize = bool(self.sp_config.get("normalize", False) if normalize is None else normalize)
        self.clip_normalized = bool(self.sp_config.get("clip_normalized", False) if clip_normalized is None else clip_normalized)

        inferred_low, inferred_high = self._infer_bounds(env)
        low_tensor = self._as_optional_tensor(low if low is not None else inferred_low)
        high_tensor = self._as_optional_tensor(high if high is not None else inferred_high)

        self.register_buffer("low", low_tensor if low_tensor is not None else torch.empty(0))
        self.register_buffer("high", high_tensor if high_tensor is not None else torch.empty(0))
        self.register_buffer("range_val", torch.empty(0))

        if self.normalize:
            if self.low.numel() == 0 or self.high.numel() == 0:
                raise ValueError("Normalization is enabled but no valid low/high bounds were provided or inferred.")
            self._refresh_range()

    @staticmethod
    def _as_optional_tensor(value: Optional[Any]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().to(dtype=torch.float32).flatten()
        return torch.as_tensor(value, dtype=torch.float32).flatten()

    @staticmethod
    def _infer_bounds(env: Optional[Any]) -> Tuple[Optional[Any], Optional[Any]]:
        if env is None:
            return None, None
        observation_space = getattr(env, "observation_space", None)
        if observation_space is None:
            return None, None
        low = getattr(observation_space, "low", None)
        high = getattr(observation_space, "high", None)
        if low is None or high is None:
            return None, None
        return low, high

    def _refresh_range(self) -> None:
        range_val = (self.high - self.low).to(dtype=torch.float32)
        range_val[range_val.abs() < 1e-6] = 1.0
        self.range_val = range_val

    def process(self, state: Any) -> torch.Tensor:
        """Compatibility alias for `forward`."""
        return self.forward(state)

    def forward(self, state: Any) -> torch.Tensor:
        """Convert a raw state into a flattened float tensor."""
        if state is None:
            logger.warning("Received None state; returning empty tensor.")
            return torch.empty(0, dtype=torch.float32)

        state = self._unwrap_env_state(state)
        processed = self._to_tensor(state).flatten().to(dtype=torch.float32)

        if self.normalize:
            processed = self._normalize(processed)

        return processed

    @staticmethod
    def _unwrap_env_state(state: Any) -> Any:
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            return state[0]
        return state

    def _to_tensor(self, state: Any) -> torch.Tensor:
        if torch.is_tensor(state):
            return state.detach()

        if hasattr(state, "__array__"):
            return torch.as_tensor(state.__array__(), dtype=torch.float32)

        if isinstance(state, (list, tuple)):
            return self._flatten_nested(state)

        if isinstance(state, dict):
            return self._flatten_nested(list(state.values()))

        return torch.as_tensor(state, dtype=torch.float32)

    def _flatten_nested(self, state: Union[Sequence[Any], Tuple[Any, ...]]) -> torch.Tensor:
        flattened = []
        stack = [state]
        while stack:
            current = stack.pop()
            if torch.is_tensor(current):
                flattened.extend(current.detach().flatten().to(dtype=torch.float32).tolist())
            elif hasattr(current, "__array__"):
                flattened.extend(torch.as_tensor(current.__array__(), dtype=torch.float32).flatten().tolist())
            elif isinstance(current, dict):
                stack.extend(reversed(list(current.values())))
            elif isinstance(current, (list, tuple)):
                stack.extend(reversed(current))
            else:
                try:
                    flattened.append(float(current))
                except (TypeError, ValueError):
                    logger.warning("Non-numeric element in state encountered: %r; replacing with 0.0", current)
                    flattened.append(0.0)

        return torch.tensor(flattened, dtype=torch.float32)

    def _normalize(self, state: torch.Tensor) -> torch.Tensor:
        if state.numel() != self.low.numel():
            raise ValueError(
                f"State dimension mismatch for normalization. Expected {self.low.numel()} values, got {state.numel()}."
            )

        low = self.low.to(device=state.device)
        range_val = self.range_val.to(device=state.device)
        normalized = (state - low) / range_val
        if self.clip_normalized:
            normalized = normalized.clamp(0.0, 1.0)
        return normalized

    def update_bounds(self, low: torch.Tensor, high: torch.Tensor) -> None:
        """Update normalization bounds dynamically."""
        if not self.normalize:
            logger.warning("update_bounds() called while normalization is disabled; enabling normalization.")
            self.normalize = True

        low_tensor = self._as_optional_tensor(low)
        high_tensor = self._as_optional_tensor(high)
        if low_tensor is None or high_tensor is None:
            raise ValueError("Both low and high must be provided.")

        if low_tensor.shape != high_tensor.shape:
            raise ValueError("low and high must have identical shapes.")

        self.low = low_tensor.to(device=self.low.device)
        self.high = high_tensor.to(device=self.high.device)
        self._refresh_range()
        logger.info("Normalization bounds updated.")

    @property
    def output_dim(self) -> Optional[int]:
        if self.normalize and self.low.numel() > 0:
            return int(self.low.numel())
        return None

if __name__ == "__main__":
    print("\n=== Running State Processor ===\n")
    printer.status("TEST", "Starting State Processor tests", "info")
    state = None
    processor = StateProcessor()
    print(processor)

    process = processor.process(state=state)
    ok = process is not None and torch.is_tensor(process)
    printer.pretty("PROCESS", "SUCCESS" if ok else "FAILURE", "success" if ok else "error")

    print("\n=== All tests completed successfully! ===\n")