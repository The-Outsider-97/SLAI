"""Headless playback position mapping; rendering is delegated to applications.

``Playback`` is a pure state machine: given a ``duration`` (seconds), a
``position`` (seconds), a signed ``rate`` (dimensionless), a ``playing`` flag, and a ``loop`` flag, it advances position in response to elapsed
wall-clock time.

Scope
-----
* Position mapping for forward, reverse, paused, and looped playback.
* Convenience transports (``play``, ``pause``, ``toggle``, ``stop``, ``rewind``, ``set_rate``) plus ``seek``/``seek_relative``.
* Optional frame-based helpers (``advance_frames``, ``seek_frame``) that delegate conversion to a :class:`Timebase` so frame↔second arithmetic
  stays exact at the call site even though the stored position is float.

Out of scope
------------
* Actual rendering — this class never draws anything.
* Frame-exact scrubbing — that is :mod:`timebase`'s job; ``Playback``
  stores position as ``float`` because continuous playback is a real-time concern and sub-nanosecond precision is not observable.

Semantics
---------
* ``rate`` may be positive (forward), negative (reverse), or zero. A non-playing ``Playback`` ignores ``rate``; a playing one with
  ``rate == 0`` is a no-op step (useful for "held but not stopped").
* ``advance(elapsed)`` requires ``elapsed >= 0``.  Direction is entirely encoded in ``rate``; trying to go backwards by passing a negative
  ``elapsed`` is rejected rather than silently doubling the sign convention.
* In loop mode, position wraps in ``[0, duration)`` for both signs of
  ``rate``.  Python's modulo operator already gives the correct least-positive residue for negative operands (``-0.5 % 3.0 == 2.5``),
  so no branch on sign is needed inside the wrap.
* In non-loop mode, crossing either boundary clamps position to the boundary and clears ``playing``.  Crossing means strictly *reaching*
  the boundary (``proposed <= 0`` for reverse, ``proposed >= duration`` for forward), matching the historical behaviour of the module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..utils.animation_errors import AnimationValidationError
from ..utils.animation_helpers import *
from .timebase import Timebase


@dataclass
class Playback:
    """Mutable playback-position state machine.

    See module docstring for full semantics.  All fields are validated on construction (via ``__post_init__``) and again on every mutating
    method, so an object built by ``Playback(...)`` is guaranteed to hold a coherent state at all times.
    """

    duration: float
    position: float = 0.0
    rate: float = 1.0
    playing: bool = False
    loop: bool = False

    def __post_init__(self) -> None:
        self.duration = ensure_finite_positive(self.duration, "duration")
        self.rate = ensure_finite_scalar(self.rate, "rate")
        if not isinstance(self.playing, bool):
            raise AnimationValidationError(f"playing must be bool, got {type(self.playing).__name__}", context={"field": "playing"})
        if not isinstance(self.loop, bool):
            raise AnimationValidationError(f"loop must be bool, got {type(self.loop).__name__}", context={"field": "loop"})
        self.seek(self.position)

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------
    @property
    def progress(self) -> float:
        """Position as a fraction of duration, in ``[0, 1]``."""
        return self.position / self.duration

    @property
    def at_start(self) -> bool:
        return self.position == 0.0

    @property
    def at_end(self) -> bool:
        return self.position == self.duration

    # ------------------------------------------------------------------
    # Seek
    # ------------------------------------------------------------------
    def seek(self, position: float) -> float:
        """Set ``position`` exactly, rejecting out-of-range values.

        Seeking does *not* modify ``playing``: a paused playback stays paused and a playing playback continues from the new position on
        the next ``advance``.

        Raises
        ------
        AnimationValidationError
            If ``position`` is non-finite, negative, or exceeds ``duration``.
        """
        pos = ensure_finite_non_negative(position, "position")
        if pos > self.duration:
            raise AnimationValidationError(
                f"position {pos!r} exceeds duration {self.duration!r}",
                context={
                    "position": pos,
                    "duration": self.duration,
                },
            )
        self.position = pos
        return self.position

    def seek_relative(self, delta: float) -> float:
        """Seek by ``delta`` seconds, clamped to ``[0, duration]``.

        Unlike :meth:`seek`, this method *clamps* rather than raising on
        out-of-range results — useful when the caller does not want to
        pre-compute whether a relative jump will cross a boundary.
        ``delta`` may be negative.
        """
        delta_f = ensure_finite_scalar(delta, "delta")
        target = self.position + delta_f
        target = max(0.0, min(self.duration, target))
        self.position = target
        return self.position

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    def play(self) -> None:
        """Mark as playing without seeking.

        Idempotent.  If the playback is currently at the boundary opposite
        the direction of travel (e.g. at end with positive rate), the next
        ``advance`` will immediately clamp and re-pause; callers who need
        "always has room to move" behaviour should call :meth:`rewind` or
        :meth:`seek` first.
        """
        self.playing = True

    def pause(self) -> None:
        """Mark as not playing without moving.  Idempotent."""
        self.playing = False

    def toggle(self) -> bool:
        """Flip the playing state; return the new value."""
        self.playing = not self.playing
        return self.playing

    def stop(self) -> None:
        """Pause and seek to start.  Common "stop" semantics."""
        self.playing = False
        self.position = 0.0

    def rewind(self) -> None:
        """Seek to start, preserving the playing flag.

        Distinct from :meth:`stop` in that it can be used as a mid-playback
        rewind without interrupting play state.
        """
        self.position = 0.0

    def set_rate(self, rate: float) -> float:
        """Change playback rate without otherwise altering state.

        A zero rate is allowed: it holds position while ``playing``
        remains whatever it was, which is useful for "pause but keep the
        transport running" workflows.
        """
        self.rate = ensure_finite_scalar(rate, "rate")
        return self.rate

    # ------------------------------------------------------------------
    # Advance
    # ------------------------------------------------------------------
    def advance(self, elapsed: float) -> float:
        """Advance position by ``elapsed`` seconds at the current rate.

        Parameters
        ----------
        elapsed
            Non-negative real.  Direction is entirely encoded by ``rate``.

        Returns
        -------
        float
            The new position (same value stored on ``self.position``).

        Raises
        ------
        AnimationValidationError
            If ``elapsed`` is negative or non-finite.

        Notes
        -----
        * When not playing, or when ``rate == 0``, position is unchanged
          and no boundary check is performed — a paused transport must
          not silently clamp itself.
        * If ``elapsed * rate`` overflows to ``+/-inf``, position is
          clamped to the boundary in the direction of travel and
          ``playing`` is cleared.  This case is only reachable with
          pathological inputs; normal playback never reaches it.
        """
        elapsed_f = ensure_finite_non_negative(elapsed, "elapsed")
        if not self.playing or self.rate == 0.0:
            return self.position

        proposed = self.position + elapsed_f * self.rate
        if not isfinite(proposed):
            # Overflow of elapsed * rate: clamp to the boundary in the
            # direction of travel rather than propagating inf/nan.
            self.position = self.duration if self.rate > 0.0 else 0.0
            self.playing = False
            return self.position

        if self.loop:
            # Python's % gives the least-positive residue for negative
            # operands (e.g. -0.5 % 3.0 == 2.5), so this correctly
            # wraps reverse playback without a sign branch.
            self.position = proposed % self.duration
        else:
            if proposed <= 0.0:
                self.position = 0.0
                self.playing = False
            elif proposed >= self.duration:
                self.position = self.duration
                self.playing = False
            else:
                self.position = proposed
        return self.position

    # ------------------------------------------------------------------
    # Frame-aware helpers (delegate conversion to Timebase)
    # ------------------------------------------------------------------
    def advance_frames(self, frame_count: int, timebase: Timebase) -> float:
        """Advance by ``frame_count`` frames at the given ``timebase``.

        Direction is still encoded by ``self.rate``: passing a positive
        ``frame_count`` with a negative ``rate`` steps backward, matching
        the semantics of :meth:`advance`.

        Raises
        ------
        AnimationValidationError
            If ``frame_count`` is not a non-negative integer.
        """
        n = ensure_non_negative_integer(frame_count, "frame_count")
        seconds = float(timebase.frame_seconds(n))
        return self.advance(seconds)

    def seek_frame(self, frame: int, timebase: Timebase) -> float:
        """Seek to the exact time of ``frame`` under ``timebase``.

        The frame→seconds conversion is done in rational arithmetic
        (:class:`Fraction`) and only the final position is rounded to
        ``float`` — position itself is float-typed, but the mapping is
        not compounded across calls.

        Raises
        ------
        AnimationValidationError
            Propagated from :meth:`seek` if the target frame lies outside
            the playback range.
        """
        # ``Timebase.frame_seconds`` already validates ``frame`` as int.
        seconds = float(timebase.frame_seconds(frame))
        return self.seek(seconds)


__all__ = [
    "Playback",
]