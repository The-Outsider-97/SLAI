"""Domain-independent contracts for learning from an external environment.

These types describe *what crosses the boundary* between any environment (a game, a simulator, a service, a bandit) and
SLAI's trainable policy.  They deliberately know nothing about a specific environment: an observation is a finite
float vector, an action is an index into a discrete action space, a reward is a finite float, and episodes have explicit
start/end.  An environment-specific *adapter* translates its own state, events and rewards into these types.

Everything here is validated at construction so that a malformed observation, action or reward is rejected visibly
(``ContractError`` / ``TransitionRejected`` with a stable ``code``) instead of silently becoming a zero vector or a default.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np  # type: ignore

SCHEMA_VERSION = "slai.external-learning.v1"

# ---------------------------------------------------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------------------------------------------------


class ContractError(ValueError):
    """A value does not satisfy a learning contract.  ``code`` is a stable, countable reason."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message


class TransitionRejected(ContractError):
    """The transition store refused an event (unknown/duplicate/closed episode, bad value, over capacity, ...)."""


class CheckpointIntegrityError(RuntimeError):
    """A restored checkpoint does not match what its metadata claims (hash mismatch, wrong shape, non-finite values)."""


# ---------------------------------------------------------------------------------------------------------------------
# value validation helpers
# ---------------------------------------------------------------------------------------------------------------------


def finite_vector(values: Any, *, name: str, size: Optional[int] = None) -> np.ndarray:
    """Return a private float64 copy of ``values`` as a finite 1-D vector, or raise ``ContractError``."""
    try:
        arr = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"invalid_{name}", f"{name} is not numeric: {exc}") from exc
    if arr.ndim != 1 or arr.size == 0:
        raise ContractError(f"invalid_{name}", f"{name} must be a non-empty 1-D vector, got shape {arr.shape}")
    if size is not None and arr.size != size:
        raise ContractError(f"invalid_{name}", f"{name} has {arr.size} values, expected {size}")
    if not np.all(np.isfinite(arr)):
        raise ContractError("non_finite", f"{name} contains NaN/inf")
    return arr


def finite_reward(value: Any, *, name: str = "reward") -> float:
    if isinstance(value, bool) or value is None:
        raise ContractError("invalid_reward", f"{name} must be a number, got {value!r}")
    try:
        r = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError("invalid_reward", f"{name} is not numeric: {exc}") from exc
    if not math.isfinite(r):
        raise ContractError("non_finite", f"{name} is NaN/inf")
    return r


def require_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"invalid_{name}", f"{name} must be a non-empty string, got {value!r}")
    return value


# ---------------------------------------------------------------------------------------------------------------------
# observation / action space / action
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionSpace:
    """A discrete action space of ``n`` actions (optionally named)."""

    n: int
    labels: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 2:
            raise ContractError("invalid_action_space", f"n must be an integer >= 2, got {self.n!r}")
        if self.labels and len(self.labels) != self.n:
            raise ContractError("invalid_action_space", "labels must have exactly n entries")


@dataclass(frozen=True)
class Observation:
    """What the policy sees for one decision.

    ``features``       finite vector, fixed length for a policy.
    ``action_mask``    optional boolean vector; False = the action is illegal now and can never be chosen.
    ``prior_logits``   optional finite vector added to the learned logits: an environment-supplied *prior* (for example a
                       hand-written or offline-trained policy).  The learned part is then a bounded residual on top of it.
    """

    features: np.ndarray
    action_mask: Optional[np.ndarray] = None
    prior_logits: Optional[np.ndarray] = None

    @staticmethod
    def make(features: Any, action_mask: Any = None, prior_logits: Any = None, *,
             feature_dim: Optional[int] = None, action_dim: Optional[int] = None) -> "Observation":
        feats = finite_vector(features, name="observation", size=feature_dim)
        mask = None
        if action_mask is not None:
            try:
                mask = np.array(action_mask, dtype=bool, copy=True)
            except (TypeError, ValueError) as exc:
                raise ContractError("invalid_action_mask", str(exc)) from exc
            if mask.ndim != 1 or (action_dim is not None and mask.size != action_dim):
                raise ContractError("invalid_action_mask", f"action_mask shape {mask.shape} does not match action space")
            if not mask.any():
                raise ContractError("no_legal_action", "action_mask leaves no legal action")
        prior = None
        if prior_logits is not None:
            prior = finite_vector(prior_logits, name="prior_logits", size=action_dim)
        return Observation(feats, mask, prior)


@dataclass(frozen=True)
class Action:
    """The decision returned to the environment."""

    index: int
    probability: float            # probability the behaviour policy assigned to ``index`` (1.0 for a greedy tie-free pick)
    probabilities: Tuple[float, ...]
    greedy: bool                  # True = argmax (evaluation), False = sampled (exploration during training)
    source: str = "learned"       # who produced it; the environment adapter must keep a fallback distinguishable


# ---------------------------------------------------------------------------------------------------------------------
# episodes and transitions
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeStart:
    episode_id: str
    timestamp: float = field(default_factory=time.time)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeEnd:
    """Terminal event.  ``terminal_reward`` is added to the last decision's reward (delayed credit)."""

    episode_id: str
    terminal_reward: float = 0.0
    terminated: bool = True       # reached a natural end (win/loss/draw)
    truncated: bool = False       # cut off (time limit); bootstrapping is the learner's concern, not the environment's
    timestamp: float = field(default_factory=time.time)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    """One decision and its consequences.  Immutable once the episode has ended."""

    episode_id: str
    decision_id: str
    sequence: int
    observation: Observation
    action: int
    reward: float
    next_observation: Optional[Observation]
    terminated: bool
    truncated: bool
    policy_version: str
    timestamp: float
    action_probability: float = 1.0
    executed_action: Optional[int] = None   # what the environment actually did if it overrode ``action`` (guardrail, fallback)

    @property
    def on_policy(self) -> bool:
        """True when the executed action is the one the policy chose, so its log-probability may be credited."""
        return self.executed_action is None or self.executed_action == self.action


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    transitions: Tuple[Transition, ...]
    terminated: bool
    truncated: bool
    total_reward: float
    started_at: float
    ended_at: float
    policy_version_start: str
    mode: str = "train"


# ---------------------------------------------------------------------------------------------------------------------
# identity / telemetry records
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyMetadata:
    """Who is answering.  ``policy_id`` is derived from the parameters, so equal ids imply equal parameters."""

    policy_id: str
    algorithm: str
    schema_version: str
    parameter_hash: str           # sha256 of the inference parameters
    state_hash: str               # sha256 of parameters + learner statistics (baseline); changes whenever anything learned changes
    hyperparameter_hash: str
    training_step: int            # number of parameter updates actually applied
    consumed_transitions: int
    consumed_episodes: int
    learning_enabled: bool
    mode: str                     # "train" | "evaluate"
    checkpoint_id: Optional[str]  # last checkpoint written/restored (None = never persisted)
    parent_checkpoint_id: Optional[str]
    restored_from: Optional[str]
    action_dim: int
    feature_dim: int
    name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def brief(self) -> Dict[str, Any]:
        """Small subset stamped on every decision / replay."""
        return {"policy_id": self.policy_id, "state_hash": self.state_hash[:16], "training_step": self.training_step,
                "consumed_transitions": self.consumed_transitions, "mode": self.mode, "learning_enabled": self.learning_enabled,
                "checkpoint_id": self.checkpoint_id}


@dataclass(frozen=True)
class LearningUpdateResult:
    """Telemetry for one attempt to update the policy (applied or skipped, always reported)."""

    updated: bool
    skip_reason: Optional[str]
    step_before: int
    step_after: int
    episodes: int = 0
    transitions: int = 0               # decisions in the batch
    credited_transitions: int = 0      # decisions that contributed a gradient (on-policy, finite)
    loss: Optional[float] = None       # surrogate -(adv * log pi) mean + entropy term; NOT a measure of improvement
    grad_norm: Optional[float] = None
    grad_norm_clipped: Optional[float] = None
    param_norm_before: Optional[float] = None
    param_norm_after: Optional[float] = None
    param_delta_norm: Optional[float] = None
    learning_rate: Optional[float] = None
    mean_return: Optional[float] = None
    mean_advantage: Optional[float] = None
    baseline: Optional[float] = None
    entropy: Optional[float] = None
    invalid_values: int = 0
    parameter_hash_before: Optional[str] = None
    parameter_hash_after: Optional[str] = None
    state_hash_before: Optional[str] = None
    state_hash_after: Optional[str] = None
    inference_uses_update: Optional[bool] = None   # True when the object that answers decisions now holds the new parameters
    elapsed_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class CheckpointMetadata:
    checkpoint_id: str
    version: str
    path: Optional[str]
    state_hash: str
    parameter_hash: str
    hyperparameter_hash: str
    parent_checkpoint_id: Optional[str]
    training_step: int
    consumed_transitions: int
    consumed_episodes: int
    schema_version: str
    algorithm: str
    created_at: str
    saved: bool = True            # False = nothing was written because the state is unchanged since the last checkpoint

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def describe_sequence(values: Sequence[float]) -> Dict[str, Any]:
    """Distribution summary used by reward reports (count, min, max, mean, std, zero share, positive/negative counts)."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None, "zero_fraction": None,
                "positive": 0, "negative": 0, "zero": 0}
    return {"count": int(arr.size), "min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean()),
            "std": float(arr.std()), "zero_fraction": float(np.mean(arr == 0.0)),
            "positive": int((arr > 0).sum()), "negative": int((arr < 0).sum()), "zero": int((arr == 0).sum())}
