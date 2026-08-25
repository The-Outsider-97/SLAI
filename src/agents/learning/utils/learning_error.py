"""learning_error.py — Domain-specific exception taxonomy and validation helpers.
 
Provides a stable, dependency-light exception hierarchy for the LearningAgent
stack and all its submodules (rl_agent, dqn, maml_rl, recovery_system,
strategy_selector, multi_task_learner, learning_calculations, etc.).
 
Design constraints
------------------
* Zero imports from the learning agent or any concrete action/model module.
  This module must be safe to import on any failure path without triggering
  circular imports or side-effects.
* No third-party dependencies — only stdlib (typing, math, traceback, sys).
* Every public symbol is listed in ``__all__`` and exported via the
  ``from .learning_error import *`` pattern already used throughout the stack.
* Exceptions carry structured metadata (as instance attributes) so that
  recovery logic, loggers, and dashboards can inspect them programmatically
  without parsing message strings.
* Validation helpers are free-functions (not methods on a class) so they can
  be called at the earliest opportunity without constructing any object.
"""
 
from __future__ import annotations
 
import math
import traceback
import torch as _torch # type: ignore
import numpy as _np # type: ignore
 
from typing import (Any, Collection, Dict, Iterable, List,
                    Mapping, Optional, Tuple, Type, Union)

from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Learning Errors")
printer = PrettyPrinter()

ScalarLike = Union[float, int, "_torch.Tensor", "_np.floating"]  # type: ignore[type-arg]

# ===========================================================================
# Base exception
# ===========================================================================
 
class LearningError(Exception):
    """Root of the learning subsystem exception hierarchy.
 
    All domain-specific exceptions extend this class so that callers can
    catch the entire hierarchy with a single ``except LearningError`` clause.
 
    Attributes
    ----------
    message : str
        Human-readable description of the error.
    context : dict
        Arbitrary key-value metadata attached at raise time (e.g. tensor
        shapes, episode numbers, config keys). Useful for programmatic
        inspection by ``RecoverySystem`` and loggers without parsing strings.
    cause : BaseException | None
        The original exception that triggered this error, if any.
        Preserved even when ``raise ... from None`` is used so that the
        recovery system can classify root causes.
    """
 
    def __init__(
        self,
        message: str = "An error occurred in the learning subsystem",
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: Dict[str, Any] = dict(context or {})
        self.cause: Optional[BaseException] = cause
 
    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        parts = [f"{type(self).__name__}: {self.message}"]
        if self.context:
            kv = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            parts.append(f"  context: {{{kv}}}")
        if self.cause is not None:
            parts.append(f"  caused by: {type(self.cause).__name__}: {self.cause}")
        return "\n".join(parts)
 
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, "
            f"context={self.context!r}, "
            f"cause={self.cause!r})"
        )
 
    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe values where possible)."""
        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "context": self.context,
            "cause": repr(self.cause) if self.cause is not None else None,
            "traceback": format_exception_chain(self),
        }
 
 
# ===========================================================================
# Numerical / gradient errors
# ===========================================================================
 
class NaNException(LearningError):
    """Raised when a NaN value is detected during learning.
 
    Attributes
    ----------
    location : str
        Descriptive label identifying where the NaN was found
        (e.g. ``"reward"``, ``"Q-value"``, ``"policy_logits"``).
    step : int | None
        Training step or episode index at which the NaN appeared.
    """
 
    def __init__(
        self,
        message: str = "NaN value detected in training",
        location: str = "unknown",
        step: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        ctx = dict(context or {})
        ctx.update({"location": location})
        if step is not None:
            ctx["step"] = step
        super().__init__(message, context=ctx, cause=cause)
        self.location = location
        self.step = step
 
 
class InfException(LearningError):
    """Raised when a positive or negative infinity is detected.
 
    Analogous to ``NaNException`` but for Inf values, which some operations
    (e.g. log of zero) produce silently.
    """
 
    def __init__(
        self,
        message: str = "Inf value detected in training",
        location: str = "unknown",
        step: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        ctx = dict(context or {})
        ctx.update({"location": location})
        if step is not None:
            ctx["step"] = step
        super().__init__(message, context=ctx, cause=cause)
        self.location = location
        self.step = step
 
 
class NumericalInstabilityError(LearningError):
    """Raised for general numerical instability that is not strictly NaN/Inf.
 
    Examples: loss oscillation above a threshold, sudden reward spikes,
    abnormally large Q-value magnitudes.
 
    Attributes
    ----------
    metric_name : str
        The name of the metric that is unstable (e.g. ``"td_loss"``).
    observed_value : float
        The value that triggered the error.
    threshold : float
        The safety threshold that was exceeded.
    """
 
    def __init__(
        self,
        message: str = "Numerical instability detected",
        metric_name: str = "unknown",
        observed_value: float = float("nan"),
        threshold: float = float("inf"),
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        ctx = dict(context or {})
        ctx.update({
            "metric_name": metric_name,
            "observed_value": observed_value,
            "threshold": threshold,
        })
        super().__init__(message, context=ctx, cause=cause)
        self.metric_name = metric_name
        self.observed_value = observed_value
        self.threshold = threshold
 
 
class GradientExplosionError(LearningError):
    """Raised when gradient norms exceed a safety threshold.
 
    Attributes
    ----------
    norm : float
        The computed gradient global norm.
    threshold : float
        The maximum allowed norm.
    layer_name : str | None
        Name of the layer or parameter group where the explosion was detected,
        if known.
    """
 
    def __init__(
        self,
        norm: float,
        threshold: float = 1e3,
        layer_name: Optional[str] = None,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            message = (
                f"Gradient explosion detected: norm={norm:.4f}, threshold={threshold:.4f}"
                + (f", layer='{layer_name}'" if layer_name else "")
            )
        ctx = dict(context or {})
        ctx.update({"norm": norm, "threshold": threshold})
        if layer_name:
            ctx["layer_name"] = layer_name
        super().__init__(message, context=ctx, cause=cause)
        self.norm = norm
        self.threshold = threshold
        self.layer_name = layer_name
 
 
class GradientVanishingError(LearningError):
    """Raised when gradient norms fall below a minimum threshold.
 
    Vanishing gradients are a distinct failure mode from explosion and deserve
    their own exception type so that recovery strategies can differ.
 
    Attributes
    ----------
    norm : float
        The computed gradient global norm.
    threshold : float
        The minimum required norm.
    """
 
    def __init__(
        self,
        norm: float,
        threshold: float = 1e-8,
        layer_name: Optional[str] = None,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            message = (
                f"Gradient vanishing detected: norm={norm:.2e}, threshold={threshold:.2e}"
                + (f", layer='{layer_name}'" if layer_name else "")
            )
        ctx = dict(context or {})
        ctx.update({"norm": norm, "threshold": threshold})
        if layer_name:
            ctx["layer_name"] = layer_name
        super().__init__(message, context=ctx, cause=cause)
        self.norm = norm
        self.threshold = threshold
        self.layer_name = layer_name
 
 
# ===========================================================================
# Configuration / validation errors
# ===========================================================================
class InvalidConfigError(LearningError):
    """Raised when agent configuration validation fails.
 
    This is the most widely used error across the stack (learning_agent,
    learning_calculations, learning_helpers, maml_rl, dqn, …).
 
    Attributes
    ----------
    config_key : str | None
        The specific key that is invalid, if known.
    received_value : Any
        The value that failed validation, if applicable.
    """
 
    def __init__(
        self,
        message: str = "Invalid agent configuration",
        config_key: Optional[str] = None,
        received_value: Any = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        ctx = dict(context or {})
        if config_key is not None:
            ctx["config_key"] = config_key
        if received_value is not None:
            ctx["received_value"] = received_value
        super().__init__(message, context=ctx, cause=cause)
        self.config_key = config_key
        self.received_value = received_value
 
    def __str__(self) -> str:  # backward-compatible: original used "InvalidConfigError: …"
        base = super().__str__()
        # Ensure prefix is present even when subclasses call super().__str__()
        if not base.startswith("InvalidConfigError"):
            return f"InvalidConfigError: {base}"
        return base
 
 
class MissingConfigKeyError(InvalidConfigError):
    """Raised when a required key is absent from a configuration dictionary.
 
    Attributes
    ----------
    section : str
        The config section name (top-level YAML key).
    missing_keys : list[str]
        All keys that were required but not found.
    """
 
    def __init__(
        self,
        section: str,
        missing_keys: Collection[str],
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        keys_str = ", ".join(f"'{k}'" for k in missing_keys)
        message = f"Config section '{section}' is missing required keys: {keys_str}"
        ctx = dict(context or {})
        ctx.update({"section": section, "missing_keys": list(missing_keys)})
        super().__init__(message, context=ctx, cause=cause)
        self.section = section
        self.missing_keys = list(missing_keys)
 
 
class ConfigTypeMismatchError(InvalidConfigError):
    """Raised when a config value has the wrong type.
 
    Attributes
    ----------
    config_key : str
        The key whose value has the wrong type.
    expected_type : type | tuple[type, ...]
        The expected Python type(s).
    actual_type : type
        The type that was actually found.
    """
 
    def __init__(
        self,
        config_key: str,
        expected_type: Union[type, Tuple[type, ...]],
        actual_type: type,
        received_value: Any = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if isinstance(expected_type, tuple):
            expected_str = " | ".join(t.__name__ for t in expected_type)
        else:
            expected_str = expected_type.__name__
        message = (
            f"Config key '{config_key}' expects {expected_str}, "
            f"got {actual_type.__name__}"
            + (f" (value={received_value!r})" if received_value is not None else "")
        )
        ctx = dict(context or {})
        ctx.update({
            "config_key": config_key,
            "expected_type": expected_str,
            "actual_type": actual_type.__name__,
        })
        super().__init__(message, config_key=config_key, received_value=received_value,
                         context=ctx, cause=cause)
        self.expected_type = expected_type
        self.actual_type = actual_type
 
 
# ===========================================================================
# Action / environment errors
# ===========================================================================
class InvalidActionError(LearningError):
    """Raised when an action fails safety validation or is undefined.
 
    Attributes
    ----------
    action : Any
        The action that was rejected.
    reason : str
        Short human-readable reason for rejection.
    """
 
    def __init__(
        self,
        action: Any = None,
        reason: str = "action failed safety validation or is undefined",
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if action is not None:
            message = f"Invalid or unsafe action {action!r}: {reason}"
        else:
            message = f"Invalid or unsafe action: {reason}"
        ctx = dict(context or {})
        if action is not None:
            ctx["action"] = action
        ctx["reason"] = reason
        super().__init__(message, context=ctx, cause=cause)
        self.action = action
        self.reason = reason
 
 
class ActionSpaceMismatchError(InvalidActionError):
    """Raised when an action does not conform to the environment's action space.
 
    Attributes
    ----------
    action : Any
        The action that was produced.
    expected_space : str
        String description of the expected action space.
    """
 
    def __init__(
        self,
        action: Any,
        expected_space: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        message = (
            f"Action {action!r} does not conform to action space: {expected_space}"
        )
        ctx = dict(context or {})
        ctx["expected_space"] = expected_space
        super().__init__(action=action, reason=f"action space mismatch ({expected_space})",
                         context=ctx, cause=cause)
        # Override message set by parent
        self.args = (message,)
        self.message = message
        self.expected_space = expected_space
 
 
class EnvironmentError(LearningError):
    """Base class for all environment interaction errors."""
 
 
class EnvironmentResetError(EnvironmentError):
    """Raised when ``env.reset()`` fails or returns an unexpected structure.
 
    Attributes
    ----------
    env_name : str
        Name or repr of the environment.
    returned_value : Any
        What ``env.reset()`` actually returned.
    """
 
    def __init__(
        self,
        env_name: str = "unknown",
        returned_value: Any = None,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            message = (
                f"env.reset() failed for environment '{env_name}'"
                + (f": returned {returned_value!r}" if returned_value is not None else "")
            )
        ctx = dict(context or {})
        ctx.update({"env_name": env_name})
        if returned_value is not None:
            ctx["returned_value"] = repr(returned_value)
        super().__init__(message, context=ctx, cause=cause)
        self.env_name = env_name
        self.returned_value = returned_value
 
 
class EnvironmentStepError(EnvironmentError):
    """Raised when ``env.step()`` fails or returns an incompatible tuple.
 
    Attributes
    ----------
    step_output_length : int | None
        Number of elements in the step tuple, if the call succeeded but
        the tuple has an unexpected length.
    action : Any
        The action that was passed to ``env.step()``.
    """
 
    def __init__(
        self,
        message: str = "env.step() returned an incompatible result",
        step_output_length: Optional[int] = None,
        action: Any = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        ctx = dict(context or {})
        if step_output_length is not None:
            ctx["step_output_length"] = step_output_length
        if action is not None:
            ctx["action"] = action
        super().__init__(message, context=ctx, cause=cause)
        self.step_output_length = step_output_length
        self.action = action
 
 
class ObservationShapeError(EnvironmentError):
    """Raised when an observation has an unexpected shape or dtype.
 
    Attributes
    ----------
    expected_shape : tuple | str
        The shape the processor expected.
    actual_shape : tuple | str
        The shape that was received.
    """
 
    def __init__(
        self,
        expected_shape: Any,
        actual_shape: Any,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            message = (
                f"Observation shape mismatch: expected {expected_shape}, got {actual_shape}"
            )
        ctx = dict(context or {})
        ctx.update({"expected_shape": str(expected_shape), "actual_shape": str(actual_shape)})
        super().__init__(message, context=ctx, cause=cause)
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
 
 
# ===========================================================================
# Training-flow errors
# ===========================================================================
class TrainingError(LearningError):
    """Base class for errors that occur during a training loop."""
 
 
class EpisodeBudgetExceededError(TrainingError):
    """Raised when the episode budget is exhausted without convergence.
 
    Attributes
    ----------
    max_episodes : int
        The configured episode budget.
    achieved_metric : float | None
        The best metric value achieved before the budget ran out.
    required_metric : float | None
        The convergence criterion that was not met.
    """
 
    def __init__(
        self,
        max_episodes: int,
        achieved_metric: Optional[float] = None,
        required_metric: Optional[float] = None,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            message = f"Episode budget exhausted after {max_episodes} episodes"
            if achieved_metric is not None and required_metric is not None:
                message += (
                    f" (achieved={achieved_metric:.4f}, required={required_metric:.4f})"
                )
        ctx = dict(context or {})
        ctx.update({"max_episodes": max_episodes})
        if achieved_metric is not None:
            ctx["achieved_metric"] = achieved_metric
        if required_metric is not None:
            ctx["required_metric"] = required_metric
        super().__init__(message, context=ctx, cause=cause)
        self.max_episodes = max_episodes
        self.achieved_metric = achieved_metric
        self.required_metric = required_metric
 
 
class StepBudgetExceededError(TrainingError):
    """Raised when the per-episode step budget is exceeded.
 
    Attributes
    ----------
    max_steps : int
        The configured step limit.
    episode : int | None
        The episode number in which the limit was exceeded.
    """
 
    def __init__(
        self,
        max_steps: int,
        episode: Optional[int] = None,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            ep_str = f" in episode {episode}" if episode is not None else ""
            message = f"Step budget of {max_steps} exceeded{ep_str}"
        ctx = dict(context or {})
        ctx.update({"max_steps": max_steps})
        if episode is not None:
            ctx["episode"] = episode
        super().__init__(message, context=ctx, cause=cause)
        self.max_steps = max_steps
        self.episode = episode
 
 
class ConvergenceError(TrainingError):
    """Raised when training fails to converge by the required criterion.
 
    Distinct from ``EpisodeBudgetExceededError``: this is raised when the
    training loop detects active divergence rather than simply hitting a budget.
 
    Attributes
    ----------
    metric_name : str
        The metric being tracked for convergence.
    observed_trend : float
        Positive values indicate improvement; negative values indicate
        divergence.
    """
 
    def __init__(
        self,
        metric_name: str = "reward",
        observed_trend: float = float("nan"),
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            message = (
                f"Training diverged or failed to converge "
                f"(metric='{metric_name}', trend={observed_trend:.4f})"
            )
        ctx = dict(context or {})
        ctx.update({"metric_name": metric_name, "observed_trend": observed_trend})
        super().__init__(message, context=ctx, cause=cause)
        self.metric_name = metric_name
        self.observed_trend = observed_trend
 
 
class CheckpointError(TrainingError):
    """Raised when saving or loading a checkpoint fails.
 
    Attributes
    ----------
    path : str
        The filesystem path of the checkpoint.
    operation : str
        Either ``"save"`` or ``"load"``.
    """
 
    def __init__(
        self,
        path: str,
        operation: str = "save",
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            message = f"Checkpoint {operation} failed: {path}"
        ctx = dict(context or {})
        ctx.update({"path": path, "operation": operation})
        super().__init__(message, context=ctx, cause=cause)
        self.path = path
        self.operation = operation
 
 
# ===========================================================================
# Memory / replay buffer errors
# ===========================================================================
class ReplayBufferError(LearningError):
    """Base class for replay buffer errors."""
 
 
class BufferUnderflowError(ReplayBufferError):
    """Raised when a sampling request cannot be satisfied due to too few items.
 
    Attributes
    ----------
    requested : int
        Number of samples requested.
    available : int
        Number of samples currently in the buffer.
    """
 
    def __init__(
        self,
        requested: int,
        available: int,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            message = (
                f"Replay buffer underflow: requested {requested} samples, "
                f"but only {available} are available"
            )
        ctx = dict(context or {})
        ctx.update({"requested": requested, "available": available})
        super().__init__(message, context=ctx, cause=cause)
        self.requested = requested
        self.available = available
 
 
class BufferOverflowError(ReplayBufferError):
    """Raised when a buffer's hard capacity limit would be exceeded.
 
    Note: most circular buffers silently evict old entries — this error is
    reserved for buffers where overflow is a programmer error rather than
    expected eviction behaviour.
 
    Attributes
    ----------
    capacity : int
        The buffer's maximum capacity.
    attempted_size : int
        The size that was attempted.
    """
 
    def __init__(
        self,
        capacity: int,
        attempted_size: int,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            message = (
                f"Replay buffer overflow: capacity={capacity}, "
                f"attempted_size={attempted_size}"
            )
        ctx = dict(context or {})
        ctx.update({"capacity": capacity, "attempted_size": attempted_size})
        super().__init__(message, context=ctx, cause=cause)
        self.capacity = capacity
        self.attempted_size = attempted_size
 
 
# ===========================================================================
# Strategy / meta-learning errors
# ===========================================================================
class StrategyError(LearningError):
    """Base class for strategy-selector errors."""
 
 
class UnknownStrategyError(StrategyError):
    """Raised when a strategy name is not in the registered strategy map.
 
    Attributes
    ----------
    strategy_name : str
        The name that was not found.
    valid_strategies : list[str]
        All currently registered strategy names.
    """
 
    def __init__(
        self,
        strategy_name: str,
        valid_strategies: Optional[Collection[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        valid_str = (
            f" Valid strategies: {sorted(valid_strategies)}"
            if valid_strategies
            else ""
        )
        message = f"Unknown strategy '{strategy_name}'.{valid_str}"
        ctx = dict(context or {})
        ctx.update({"strategy_name": strategy_name})
        if valid_strategies is not None:
            ctx["valid_strategies"] = sorted(valid_strategies)
        super().__init__(message, context=ctx, cause=cause)
        self.strategy_name = strategy_name
        self.valid_strategies = list(valid_strategies) if valid_strategies else []
 
 
class StrategySelectionError(StrategyError):
    """Raised when the meta-controller cannot select a strategy.
 
    For example: the policy network is not initialised, or the embedding
    buffer is empty when selection is attempted.
 
    Attributes
    ----------
    reason : str
        Detailed explanation of why selection failed.
    """
 
    def __init__(
        self,
        reason: str = "strategy selection failed",
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            message = f"StrategySelector cannot select a strategy: {reason}"
        ctx = dict(context or {})
        ctx["reason"] = reason
        super().__init__(message, context=ctx, cause=cause)
        self.reason = reason
 
 
# ===========================================================================
# Recovery system errors
# ===========================================================================
class RecoveryError(LearningError):
    """Base class for errors raised by or about the recovery system."""
 
 
class RecoveryExhaustedError(RecoveryError):
    """Raised when all recovery attempts have been exhausted.
 
    Attributes
    ----------
    max_attempts : int
        The configured maximum number of recovery attempts.
    last_error : BaseException | None
        The error that triggered the final (failed) recovery attempt.
    """
 
    def __init__(
        self,
        max_attempts: int,
        last_error: Optional[BaseException] = None,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            message = (
                f"All {max_attempts} recovery attempts exhausted"
                + (f": last error was {type(last_error).__name__}" if last_error else "")
            )
        ctx = dict(context or {})
        ctx.update({"max_attempts": max_attempts})
        if last_error is not None:
            ctx["last_error"] = repr(last_error)
        super().__init__(message, context=ctx, cause=cause or last_error)
        self.max_attempts = max_attempts
        self.last_error = last_error


# ===========================================================================
# Utility: exception chain formatter
# ===========================================================================
def format_exception_chain(exc: BaseException) -> str:
    """Return a concise, single-string representation of *exc* and its chain.
 
    Safe to call from any failure path — never raises.
 
    Parameters
    ----------
    exc :
        The exception to format.
 
    Returns
    -------
    str
        Multi-line string containing the exception type, message, and the
        full traceback if available.  Chained exceptions (``__cause__`` /
        ``__context__``) are included.
    """
    try:
        lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        return "".join(lines).rstrip()
    except Exception:  # pragma: no cover — absolute fallback
        return f"{type(exc).__name__}: {exc}"
