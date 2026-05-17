"""Production-ready PyTorch state processor for SLAI environment observations.

This module converts raw environment observations into deterministic float
feature tensors while preserving the lightweight API used across the learning
stack. It intentionally stays focused on state processing and relies on the
shared learning error, calculation, and helper modules for validation,
telemetry, and safe serialization.
"""

from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.learning_error import *
from ..utils.learning_calculations import *
from ..utils.learning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("State Processor")
printer = PrettyPrinter()

TensorLike = Union[torch.Tensor, Sequence[float]]


class StateProcessor(nn.Module):
    """Convert raw observations into stable flattened float tensors.

    Supported inputs include Gym/Gymnasium ``(obs, info)`` tuples, tensors,
    numpy-like arrays, scalars, lists/tuples, and nested dictionaries. Optional
    normalization can use explicit bounds or bounds inferred from ``env``.
    """

    VALID_NORMALIZE_MODES = {"minmax", "centered"}
    VALID_NON_NUMERIC_POLICIES = {"zero", "error", "skip"}

    def __init__(
        self,
        env: Optional[Any] = None,
        normalize: Optional[bool] = None,
        low: Optional[TensorLike] = None,
        high: Optional[TensorLike] = None,
        clip_normalized: Optional[bool] = None,
        expected_dim: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.env = env
        self.config = load_global_config()
        self.sp_config = get_config_section("state_processor_config") or {}
        self.calculations = LearningCalculations()
        self.value_stats = RunningStats()
        self.norm_stats = RunningStats()
        self.length_stats = RunningStats()

        self.target_device = torch.device(device or self.sp_config.get("device", "cpu"))
        self.target_dtype = dtype
        self.normalize = coerce_bool(self.sp_config.get("normalize", False) if normalize is None else normalize, default=False)
        self.clip_normalized = coerce_bool(
            self.sp_config.get("clip_normalized", True) if clip_normalized is None else clip_normalized,
            default=True,
        )
        self.normalize_mode = str(self.sp_config.get("normalize_mode", "minmax")).lower()
        if self.normalize_mode not in self.VALID_NORMALIZE_MODES:
            raise InvalidConfigError(
                "Unsupported state normalization mode",
                config_key="state_processor_config.normalize_mode",
                received_value=self.normalize_mode,
                context={"supported": sorted(self.VALID_NORMALIZE_MODES)},
            )

        self.expected_dim = self._resolve_expected_dim(expected_dim, env)
        self.allow_none_state = coerce_bool(self.sp_config.get("allow_none_state", True), default=True)
        self.empty_state_dim = coerce_int(self.sp_config.get("empty_state_dim", 0), default=0, minimum=0)
        self.non_numeric_policy = str(self.sp_config.get("non_numeric_policy", "zero")).lower()
        if self.non_numeric_policy not in self.VALID_NON_NUMERIC_POLICIES:
            raise InvalidConfigError(
                "Unsupported non-numeric state policy",
                config_key="state_processor_config.non_numeric_policy",
                received_value=self.non_numeric_policy,
                context={"supported": sorted(self.VALID_NON_NUMERIC_POLICIES)},
            )
        self.sort_dict_keys = coerce_bool(self.sp_config.get("sort_dict_keys", True), default=True)
        self.detach_tensors = coerce_bool(self.sp_config.get("detach_tensors", True), default=True)
        self.replace_non_finite = coerce_bool(self.sp_config.get("replace_non_finite", False), default=False)
        self.non_finite_replacement = coerce_float(self.sp_config.get("non_finite_replacement", 0.0), default=0.0)
        self.clip_values = self.sp_config.get("clip_values")
        self.check_finite = coerce_bool(self.sp_config.get("check_finite", True), default=True)
        self.round_precision = coerce_int(self.sp_config.get("round_precision", 8), default=8, minimum=0)
        self.max_history = coerce_int(self.sp_config.get("max_history", 100), default=100, minimum=1)
        self.recent_norms: List[float] = []
        self.process_count = 0
        self.last_summary: Dict[str, float] = {}

        inferred_low, inferred_high = self._infer_bounds(env)
        low_tensor = self._as_optional_tensor(low if low is not None else inferred_low, name="low")
        high_tensor = self._as_optional_tensor(high if high is not None else inferred_high, name="high")
        self.register_buffer("low", low_tensor if low_tensor is not None else torch.empty(0, dtype=self.target_dtype, device=self.target_device))
        self.register_buffer("high", high_tensor if high_tensor is not None else torch.empty(0, dtype=self.target_dtype, device=self.target_device))
        self.register_buffer("range_val", torch.empty(0, dtype=self.target_dtype, device=self.target_device))

        if self.normalize:
            if self.low.numel() == 0 or self.high.numel() == 0:
                raise InvalidConfigError(
                    "Normalization is enabled but no valid low/high bounds were provided or inferred.",
                    config_key="state_processor_config.normalize",
                )
            self._validate_bounds(self.low, self.high)
            self._refresh_range()
        logger.info("StateProcessor initialized | normalize=%s expected_dim=%s", self.normalize, self.expected_dim)

    def _resolve_expected_dim(self, expected_dim: Optional[int], env: Optional[Any]) -> Optional[int]:
        value = expected_dim if expected_dim is not None else self.sp_config.get("expected_dim")
        if value is not None:
            validate_positive(value, "state_processor_config.expected_dim", strict=True)
            return int(value)
        space = getattr(env, "observation_space", None) if env is not None else None
        shape = getattr(space, "shape", None)
        if shape:
            total = 1
            for dim in shape:
                total *= int(dim)
            return int(total)
        return None

    def _as_optional_tensor(self, value: Optional[Any], *, name: str) -> Optional[torch.Tensor]:
        if value is None:
            return None
        tensor = self._to_tensor(value).flatten().to(device=self.target_device, dtype=self.target_dtype)
        self._ensure_finite_tensor(tensor, name)
        return tensor

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

    def _validate_bounds(self, low: torch.Tensor, high: torch.Tensor) -> None:
        if low.shape != high.shape:
            raise ObservationShapeError(expected_shape=tuple(low.shape), actual_shape=tuple(high.shape))
        if low.numel() == 0:
            raise InvalidConfigError("normalization bounds cannot be empty", config_key="state_processor_config.bounds")
        invalid = torch.logical_not(high > low)
        if bool(invalid.any().item()):
            raise InvalidConfigError(
                "Each high bound must be greater than its matching low bound.",
                config_key="state_processor_config.bounds",
                context={"invalid_count": int(invalid.sum().item())},
            )
        if self.expected_dim is not None and low.numel() != self.expected_dim:
            raise ObservationShapeError(expected_shape=(self.expected_dim,), actual_shape=(low.numel(),))

    def _refresh_range(self) -> None:
        self._validate_bounds(self.low, self.high)
        self.range_val = (self.high - self.low).to(device=self.target_device, dtype=self.target_dtype)
        self.range_val = torch.where(self.range_val.abs() < 1e-12, torch.ones_like(self.range_val), self.range_val)

    def process(self, state: Any) -> torch.Tensor:
        """Compatibility alias for ``forward``."""
        return self.forward(state)

    def forward(self, state: Any) -> torch.Tensor:
        """Convert a raw state into a flattened, validated float tensor."""
        if state is None:
            if not self.allow_none_state:
                raise ObservationShapeError(expected_shape="non-null state", actual_shape="None")
            logger.warning("Received None state; returning configured empty tensor.")
            processed = torch.zeros(self.empty_state_dim, dtype=self.target_dtype, device=self.target_device)
        else:
            unwrapped = self._unwrap_env_state(state)
            processed = self._to_tensor(unwrapped).flatten().to(device=self.target_device, dtype=self.target_dtype)

        if self.expected_dim is not None and processed.numel() != self.expected_dim:
            raise ObservationShapeError(expected_shape=(self.expected_dim,), actual_shape=(processed.numel(),))
        processed = self._sanitize_tensor(processed, "state")
        if self.clip_values is not None:
            limit = coerce_float(self.clip_values, default=0.0, minimum=0.0)
            if limit > 0.0:
                processed = processed.clamp(-limit, limit)
        if self.normalize:
            processed = self._normalize(processed)
        self._record_observation(processed)
        return processed

    @staticmethod
    def _unwrap_env_state(state: Any) -> Any:
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            return state[0]
        return state

    def _to_tensor(self, state: Any) -> torch.Tensor:
        if torch.is_tensor(state):
            return state.detach() if self.detach_tensors else state
        if hasattr(state, "__array__"):
            return torch.as_tensor(state.__array__(), dtype=self.target_dtype, device=self.target_device)
        if isinstance(state, Mapping):
            keys = sorted(state.keys()) if self.sort_dict_keys else list(state.keys())
            return self._flatten_nested([state[key] for key in keys])
        if isinstance(state, (list, tuple)):
            return self._flatten_nested(state)
        try:
            return torch.as_tensor(state, dtype=self.target_dtype, device=self.target_device)
        except Exception as exc:
            if self.non_numeric_policy == "error":
                raise ObservationShapeError(expected_shape="numeric state", actual_shape=type(state).__name__, cause=exc)
            if self.non_numeric_policy == "skip":
                return torch.empty(0, dtype=self.target_dtype, device=self.target_device)
            logger.warning("Non-numeric scalar state %r replaced with 0.0", state)
            return torch.zeros(1, dtype=self.target_dtype, device=self.target_device)

    def _flatten_nested(self, state: Union[Sequence[Any], Tuple[Any, ...]]) -> torch.Tensor:
        flattened: List[float] = []
        stack: List[Any] = [state]
        while stack:
            current = stack.pop()
            if torch.is_tensor(current):
                tensor = current.detach() if self.detach_tensors else current
                flattened.extend(tensor.flatten().to(dtype=self.target_dtype).cpu().tolist())
            elif hasattr(current, "__array__"):
                flattened.extend(torch.as_tensor(current.__array__(), dtype=self.target_dtype).flatten().cpu().tolist())
            elif isinstance(current, Mapping):
                keys = sorted(current.keys()) if self.sort_dict_keys else list(current.keys())
                stack.extend(reversed([current[key] for key in keys]))
            elif isinstance(current, (list, tuple)):
                stack.extend(reversed(current))
            else:
                try:
                    flattened.append(float(current))
                except (TypeError, ValueError) as exc:
                    if self.non_numeric_policy == "error":
                        raise ObservationShapeError(expected_shape="numeric nested state", actual_shape=type(current).__name__, cause=exc)
                    if self.non_numeric_policy == "zero":
                        logger.warning("Non-numeric state element %r replaced with 0.0", current)
                        flattened.append(0.0)
        return torch.tensor(flattened, dtype=self.target_dtype, device=self.target_device)

    def _ensure_finite_tensor(self, tensor: torch.Tensor, name: str) -> None:
        if not self.check_finite:
            return
        if bool(torch.isnan(tensor).any().item()):
            raise NaNException(f"NaN detected in {name}", location=name)
        if bool(torch.isinf(tensor).any().item()):
            raise InfException(f"Inf detected in {name}", location=name)

    def _sanitize_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if self.check_finite and bool(torch.isfinite(tensor).all().item()):
            return tensor
        if not self.replace_non_finite:
            self._ensure_finite_tensor(tensor, name)
            return tensor
        return torch.nan_to_num(
            tensor,
            nan=self.non_finite_replacement,
            posinf=self.non_finite_replacement,
            neginf=self.non_finite_replacement,
        )

    def _normalize(self, state: torch.Tensor) -> torch.Tensor:
        if state.numel() != self.low.numel():
            raise ObservationShapeError(expected_shape=(self.low.numel(),), actual_shape=(state.numel(),))
        low = self.low.to(device=state.device, dtype=state.dtype)
        high = self.high.to(device=state.device, dtype=state.dtype)
        range_val = self.range_val.to(device=state.device, dtype=state.dtype)
        if self.normalize_mode == "centered":
            normalized = 2.0 * ((state - low) / range_val) - 1.0
            if self.clip_normalized:
                normalized = normalized.clamp(-1.0, 1.0)
        else:
            normalized = (state - low) / range_val
            if self.clip_normalized:
                normalized = normalized.clamp(0.0, 1.0)
        self._ensure_finite_tensor(normalized, "normalized_state")
        return normalized

    def _record_observation(self, processed: torch.Tensor) -> None:
        values = processed.detach().float().cpu().tolist()
        self.process_count += 1
        self.length_stats.update(float(processed.numel()))
        norm = float(torch.linalg.vector_norm(processed.detach().float()).item()) if processed.numel() else 0.0
        self.norm_stats.update(norm)
        self.recent_norms.append(norm)
        if len(self.recent_norms) > self.max_history:
            self.recent_norms = self.recent_norms[-self.max_history:]
        if values:
            self.value_stats.extend(values)
            self.last_summary = self.calculations.summarize_rewards(values)
        else:
            self.last_summary = self.calculations.summarize_rewards([])

    def update_bounds(self, low: TensorLike, high: TensorLike) -> None:
        """Update normalization bounds dynamically and enable normalization."""
        low_tensor = self._as_optional_tensor(low, name="low")
        high_tensor = self._as_optional_tensor(high, name="high")
        if low_tensor is None or high_tensor is None:
            raise InvalidConfigError("Both low and high bounds must be provided.", config_key="state_processor_config.bounds")
        self._validate_bounds(low_tensor, high_tensor)
        self.low = low_tensor.to(device=self.target_device, dtype=self.target_dtype)
        self.high = high_tensor.to(device=self.target_device, dtype=self.target_dtype)
        self.normalize = True
        self._refresh_range()
        logger.info("Normalization bounds updated.")

    def to_device(self, device: Union[str, torch.device]) -> "StateProcessor":
        """Move processor buffers to a target device."""
        target = torch.device(device)
        self.target_device = target
        self.to(target)
        return self

    @property
    def output_dim(self) -> Optional[int]:
        if self.expected_dim is not None:
            return self.expected_dim
        if self.normalize and self.low.numel() > 0:
            return int(self.low.numel())
        return None

    def diagnostics(self) -> Dict[str, Any]:
        recent_ma = self.calculations.moving_average(self.recent_norms, window_size=min(5, max(1, len(self.recent_norms)))) if self.recent_norms else []
        return {
            "normalize": self.normalize,
            "normalize_mode": self.normalize_mode,
            "clip_normalized": self.clip_normalized,
            "expected_dim": self.expected_dim,
            "output_dim": self.output_dim,
            "device": str(self.target_device),
            "dtype": str(self.target_dtype),
            "process_count": self.process_count,
            "last_summary": {k: round_float(v, self.round_precision) for k, v in self.last_summary.items()},
            "value_stats": to_json_safe(self.value_stats.snapshot()),
            "norm_stats": to_json_safe(self.norm_stats.snapshot()),
            "length_stats": to_json_safe(self.length_stats.snapshot()),
            "recent_norm_moving_average": [round_float(v, self.round_precision) for v in recent_ma],
        }

    def get_checkpoint(self) -> Dict[str, Any]:
        return {
            "model_type": type(self).__name__,
            "state_dict": self.state_dict(),
            "config": {
                "normalize": self.normalize,
                "clip_normalized": self.clip_normalized,
                "normalize_mode": self.normalize_mode,
                "expected_dim": self.expected_dim,
                "empty_state_dim": self.empty_state_dim,
                "non_numeric_policy": self.non_numeric_policy,
                "replace_non_finite": self.replace_non_finite,
                "non_finite_replacement": self.non_finite_replacement,
                "clip_values": self.clip_values,
            },
            "diagnostics": self.diagnostics(),
        }

    def save(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.get_checkpoint(), target)
            return target
        except Exception as exc:
            raise CheckpointError(str(target), operation="save", cause=exc) from exc

    def load(self, path: Union[str, Path], map_location: Optional[Union[str, torch.device]] = None, strict: bool = True) -> Dict[str, Any]:
        source = Path(path)
        try:
            checkpoint = torch.load(source, map_location=map_location)
            if not isinstance(checkpoint, Mapping):
                raise CheckpointError(str(source), operation="load", message="StateProcessor checkpoint is not a mapping")
            validate_required_keys(checkpoint, ["state_dict"], name="state_processor_checkpoint")
            self.load_state_dict(checkpoint["state_dict"], strict=strict)
            self._refresh_range() if self.low.numel() and self.high.numel() else None
            return dict(checkpoint)
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(str(source), operation="load", cause=exc) from exc

    def extra_repr(self) -> str:
        return f"normalize={self.normalize}, mode='{self.normalize_mode}', expected_dim={self.expected_dim}"


if __name__ == "__main__":
    print("\n=== Running State Processor ===\n")
    printer.status("TEST", "State Processor initialized", "info")
    p = StateProcessor(normalize=True, low=torch.zeros(4), high=torch.ones(4) * 10, expected_dim=4)
    x = p.process(({"b": [4.0, 5.0], "a": torch.tensor([1.0, 2.0])}, {"source": "gym"}))
    assert x.shape == (4,) and torch.all((x >= 0.0) & (x <= 1.0))
    p.update_bounds(torch.zeros(4), torch.ones(4))
    y = p.process([0.1, 0.2, 0.3, 0.4])
    assert y.shape == (4,) and torch.isfinite(y).all()
    diag = p.diagnostics()
    assert diag["process_count"] == 2 and diag["output_dim"] == 4
    ckpt = Path("state_processor_test.pt")
    p.save(ckpt)
    restored = StateProcessor(normalize=True, low=torch.zeros(4), high=torch.ones(4), expected_dim=4)
    restored.load(ckpt)
    assert torch.allclose(p.low, restored.low) and torch.allclose(p.high, restored.high)
    ckpt.unlink(missing_ok=True)
    printer.status("TEST", "State conversion, bounds, diagnostics, checkpoint verified", "success")
    print("\n=== Test ran successfully ===\n")
