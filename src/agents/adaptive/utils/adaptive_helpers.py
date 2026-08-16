"""
Shared helper primitives for the adaptive agent subsystem.

This module centralizes cross-cutting, reusable mechanics used by multiple
adaptive components. It intentionally does not own configuration, validation
policy, model behavior, persistence orchestration, or agent lifecycle state.

The helpers here are dependency-direction leaves: they must not import
AdaptiveMemory, PolicyManager, SkillWorker, ImitationLearningWorker,
MetaLearningWorker, or LearningParameterTuner.
"""

from __future__ import annotations

import hashlib
import json

from collections import deque
from datetime import datetime, timedelta
from typing import Any, Mapping #, TYPE_CHECKING, 

import numpy as np # type: ignore
import pandas as pd # type: ignore

from .adaptive_errors import ensure_instance

# if TYPE_CHECKING:
import torch # type: ignore


def resolve_torch_device(
    device_setting: str = "auto",
    *,
    fallback_cuda_to_cpu: bool = False,
) -> torch.device:
    setting = str(device_setting).strip().lower()

    if setting == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if (
        setting.startswith("cuda")
        and not torch.cuda.is_available()
        and fallback_cuda_to_cpu
    ):
        return torch.device("cpu")

    return torch.device(setting)


def is_finite_number(value: Any) -> bool:
    """
    Return whether value satisfies the scalar numeric contract used by
    the Adaptive subsystem.

    This intentionally preserves SLAI v2.2 behavior by accepting the same
    Python and NumPy scalar types currently checked by Adaptive components.
    """
    return isinstance(value, (int, float, np.number)) and bool(np.isfinite(value))


def json_safe(value: Any) -> Any:
    """
    Convert Adaptive runtime values using the existing v2.2 serialization rules.

    This function intentionally preserves the current MultiModalMemory
    serialization behavior to avoid changing checkpoint or context-hash
    semantics during helper extraction.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()

    if isinstance(value, timedelta):
        return value.total_seconds()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if isinstance(value, deque):
        return [json_safe(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")

    return repr(value)


def stable_context_hash(
    context: Mapping[str, Any],
    *,
    component: str = "adaptive_helpers",
) -> str:
    """
    Create the deterministic SHA-256 context hash used by SLAI v2.2.

    The JSON representation intentionally matches MultiModalMemory's existing
    context-hash format. Changing the canonicalization here would invalidate
    persisted semantic context identities.
    """
    ensure_instance(
        context,
        Mapping,
        "context",
        component=component,
    )

    sanitized = json_safe(dict(context))

    context_blob = json.dumps(
        sanitized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(
        context_blob.encode("utf-8")
    ).hexdigest()


def stable_state_action_hash(state: Any, action: Any) -> str:
    """
    Create the fallback state/action hash used when explicit context is absent.

    This preserves the existing v2.2 JSON formatting separately from
    stable_context_hash().
    """
    payload = {
        "state": json_safe(state),
        "action": json_safe(action),
    }

    payload_blob = json.dumps(
        payload,
        sort_keys=True,
    )

    return hashlib.sha256(
        payload_blob.encode("utf-8")
    ).hexdigest()


def resolve_worker_device(device_preference: str) -> "torch.device":
    """
    Resolve the compute device for Adaptive learning workers.

    Current worker semantics are preserved:
    - "cpu" always resolves to CPU.
    - "cuda" uses CUDA when available and otherwise falls back to CPU.
    - every other value follows the workers' existing automatic-selection
      behavior.

    PolicyManager intentionally does not use this helper because its device
    contract differs.
    """
    import torch # type: ignore

    preference = str(device_preference).lower()

    if preference == "cuda":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if preference == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


__all__ = [
    "is_finite_number",
    "json_safe",
    "stable_context_hash",
    "stable_state_action_hash",
    "resolve_worker_device",
]