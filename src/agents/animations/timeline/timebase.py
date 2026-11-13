"""Exact rational frame-rate arithmetic and SMPTE timecode conversion.

Scope
-----
* Frame ↔ second conversion via :class:`fractions.Fraction` so repeated
  conversion accumulates zero drift (unlike float-second arithmetic).
* Frame rounding (floor / ceil / round-half-up) for scrubbing,
  snapping, and "frame at or before time t" queries.
* SMPTE timecode formatting and parsing, including drop-frame for
  NTSC 30000/1001 and 60000/1001.
* Named NTSC / PAL / film presets.

Out of scope
------------
* Frame-accurate rendering scheduling — that belongs in the renderer.
* Non-SMPTE timecode conventions (e.g. MIDI clock, cue markers).

Naming note
-----------
``frame_seconds`` and ``seconds_frame`` are retained as the historical
names to preserve existing callers.  ``frame_seconds(frame)`` means
"the elapsed time, in seconds, of frame N" (i.e. *frame → seconds*);
``seconds_frame(seconds)`` means "the fractional frame count
corresponding to ``seconds``" (i.e. *seconds → frames*).  Read carefully
— the second name is not the inverse *spelling* of the first.

Rational arithmetic
-------------------
All primary conversions take and return :class:`Fraction`.  When an
``int`` or ``float`` is supplied where a ``Fraction`` is expected, it is
coerced via :class:`Fraction` directly, which for ``float`` gives the
*exact binary rational* of the input.  This is the correct semantic for
"give me the exact frame count of this float" — it makes float noise
visible rather than silently smoothing it.  For frame inputs, pass
``int`` (the natural type) and conversion is exact by construction.
"""

from __future__ import annotations

import re

from fractions import Fraction
from math import ceil, floor, isfinite

from ..utils.animation_errors import AnimationValidationError
from ..utils.animation_helpers import *


# SMPTE pattern: HH:MM:SS<sep>FF where separator is ':' (non-drop) or ';' (drop).
_TC_PATTERN = re.compile(r"^(\d{2}):(\d{2}):(\d{2})([:;])(\d{2})$")


class Timebase:
    """Immutable-by-convention rational frame-rate holder.

    The rate is stored as :class:`Fraction` so ``Timebase(30, 1)`` and
    ``Timebase(60, 2)`` compare equal and hash identically.  The
    original numerator/denominator are *not* preserved separately; the
    reduced fraction is authoritative.

    Attributes
    ----------
    fps : Fraction
        The frame rate as a reduced rational.  Read-only by convention;
        mutating it after construction breaks the invariants used by
        :meth:`timecode`, :meth:`parse_timecode`, and the frame
        converters.
    """

    def __init__(self, numerator: int = 30, denominator: int = 1) -> None:
        num = ensure_positive_integer(numerator, "numerator")
        den = ensure_positive_integer(denominator, "denominator")
        self.fps: Fraction = Fraction(num, den)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_fraction(cls, fps: Fraction) -> "Timebase":
        """Build from an existing :class:`Fraction`."""
        if not isinstance(fps, Fraction):
            raise AnimationValidationError(
                f"fps must be a Fraction, got {type(fps).__name__}",
                context={"received_type": type(fps).__name__},
            )
        return cls(fps.numerator, fps.denominator)

    @classmethod
    def from_string(cls, text: str) -> "Timebase":
        """Build from a rate string.

        Accepted forms:
        * ``"30000/1001"`` — exact rational.
        * ``"30"`` or ``"29.97"`` — :class:`Fraction`-parsed literal.

        Note that ``"29.97"`` is *not* the NTSC rate; it is
        ``Fraction(2997, 100)``, which is an approximation.  For NTSC,
        use the exact form ``"30000/1001"`` or the
        :meth:`ntsc_30` preset.
        """
        if not isinstance(text, str):
            raise AnimationValidationError(
                f"rate string must be str, got {type(text).__name__}",
                context={"received_type": type(text).__name__},
            )
        stripped = text.strip()
        if not stripped:
            raise AnimationValidationError("rate string must be non-empty")
        if "/" in stripped:
            num_str, _, den_str = stripped.partition("/")
            try:
                num = int(num_str.strip())
                den = int(den_str.strip())
            except ValueError as exc:
                raise AnimationValidationError(
                    f"invalid rational rate string {text!r}",
                    context={"input": text},
                    cause=exc,
                ) from exc
            return cls(num, den)
        try:
            rate = Fraction(stripped)
        except (ValueError, ZeroDivisionError) as exc:
            raise AnimationValidationError(
                f"invalid rate string {text!r}",
                context={"input": text},
                cause=exc,
            ) from exc
        return cls(rate.numerator, rate.denominator)

    @classmethod
    def ntsc_24(cls) -> "Timebase":
        """24000/1001 ≈ 23.976 fps (NTSC film transfer)."""
        return cls(24000, 1001)

    @classmethod
    def ntsc_30(cls) -> "Timebase":
        """30000/1001 ≈ 29.97 fps (NTSC video)."""
        return cls(30000, 1001)

    @classmethod
    def ntsc_60(cls) -> "Timebase":
        """60000/1001 ≈ 59.94 fps (NTSC high-frame-rate)."""
        return cls(60000, 1001)

    @classmethod
    def pal(cls) -> "Timebase":
        """25 fps (PAL / SECAM)."""
        return cls(25, 1)

    @classmethod
    def film(cls) -> "Timebase":
        """24 fps (film)."""
        return cls(24, 1)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def frame_duration(self) -> Fraction:
        """Exact duration of one frame, in seconds."""
        return Fraction(1, 1) / self.fps

    @property
    def is_integer_rate(self) -> bool:
        """True if ``fps`` is a whole number (PAL, film, 30, 60…)."""
        return self.fps.denominator == 1

    # ------------------------------------------------------------------
    # Frame ↔ second conversion (exact)
    # ------------------------------------------------------------------
    def frame_seconds(self, frame: int) -> Fraction:
        """Return the exact timestamp in seconds of integer frame ``frame``.

        ``frame_seconds(0) == 0``; ``frame_seconds(-n)`` is negative.
        Negative frames represent offsets before the origin and are
        allowed so callers can express "5 frames before t=0" without
        hand-arithmetic.

        Raises
        ------
        AnimationValidationError
            If ``frame`` is not a plain :class:`int` (``bool`` rejected).
        """
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise AnimationValidationError(
                f"frame must be an integer, got {type(frame).__name__}",
                context={"field": "frame", "received_type": type(frame).__name__},
            )
        return Fraction(frame, 1) / self.fps

    def seconds_frame(self, seconds) -> Fraction:
        """Return the exact (possibly fractional) frame count of ``seconds``.

        For ``int`` / ``Fraction`` inputs the result is exact.  For
        ``float`` inputs it is the exact rational of the float's binary
        value — e.g. ``Fraction(0.1)`` is not ``1/10``.  Callers who
        need "the intended real value" should pass a ``Fraction``
        (``Fraction("0.1")``, or ``Fraction(1, 10)``) instead.

        Raises
        ------
        AnimationValidationError
            If ``seconds`` is ``bool``, non-finite, or not a real number.
        """
        if isinstance(seconds, bool):
            raise AnimationValidationError(
                "seconds must be a real number, got bool",
                context={"field": "seconds", "received_type": "bool"},
            )
        if isinstance(seconds, Fraction):
            seconds_fraction = seconds
        elif isinstance(seconds, int):
            seconds_fraction = Fraction(seconds, 1)
        elif isinstance(seconds, float):
            if not isfinite(seconds):
                raise AnimationValidationError(
                    f"seconds must be finite, got {seconds!r}",
                    context={"field": "seconds", "received": seconds},
                )
            seconds_fraction = Fraction(seconds)
        else:
            raise AnimationValidationError(
                f"seconds must be int, float or Fraction, got {type(seconds).__name__}",
                context={"field": "seconds", "received_type": type(seconds).__name__},
            )
        return seconds_fraction * self.fps

    # ------------------------------------------------------------------
    # Frame rounding
    # ------------------------------------------------------------------
    def frame_at_or_before(self, seconds) -> int:
        """Largest integer frame ``n`` with ``frame_seconds(n) <= seconds``."""
        f = self.seconds_frame(seconds)
        return floor(f)

    def frame_at_or_after(self, seconds) -> int:
        """Smallest integer frame ``n`` with ``frame_seconds(n) >= seconds``."""
        f = self.seconds_frame(seconds)
        return ceil(f)

    def nearest_frame(self, seconds) -> int:
        """Integer frame closest to ``seconds``, ties rounded toward +∞.

        Half-values (``f + 0.5``) round *up* toward +∞, matching the
        common "round half up" convention.  This is distinct from Python's
        built-in ``round()``, which uses banker's (round-half-to-even)
        rounding and would map ``2.5 → 2`` where this method maps it to
        ``3``.
        """
        f = self.seconds_frame(seconds)
        return floor(f + Fraction(1, 2))

    # ------------------------------------------------------------------
    # SMPTE timecode
    # ------------------------------------------------------------------
    def timecode(self, frame: int, *, drop_frame: bool | None = None) -> str:
        """Format integer ``frame`` as SMPTE ``HH:MM:SS:FF`` timecode.

        Parameters
        ----------
        frame
            Non-negative integer frame number in *physical* frames (i.e.
            actual elapsed frames of playback, not the timecode value).
        drop_frame
            * ``False`` — non-drop-frame; separator is ``:``.
            * ``True`` — drop-frame; separator is ``;``.  Only valid for
              ``fps ∈ {30000/1001, 60000/1001}``.
            * ``None`` (default) — auto-detect: drop-frame for NTSC
              29.97 / 59.94, non-drop for all other rates.

        The nominal rate (``round(fps)``) determines the field layout
        (e.g. 30 for 29.97); drop-frame adjusts which frame *numbers* are
        skipped so the timecode label tracks wall-clock time.

        Raises
        ------
        AnimationValidationError
            For negative/non-integer ``frame``, or for ``drop_frame=True``
            on a rate that does not define drop-frame timecode.
        """
        ensure_non_negative_integer(frame, "frame")
        nominal = round(self.fps)
        is_ntsc = self.fps != Fraction(nominal, 1)

        if drop_frame is None:
            drop_frame = is_ntsc and nominal in (30, 60)
        elif not isinstance(drop_frame, bool):
            raise AnimationValidationError(
                "drop_frame must be bool or None",
                context={"received_type": type(drop_frame).__name__},
            )

        if not drop_frame:
            ff = frame % nominal
            ss = (frame // nominal) % 60
            mm = (frame // (nominal * 60)) % 60
            hh = frame // (nominal * 3600)
            return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

        if nominal not in (30, 60):
            raise AnimationValidationError(
                f"drop-frame timecode is only defined for NTSC 30000/1001 "
                f"and 60000/1001, got {self.fps}",
                context={"fps": str(self.fps)},
            )

        drop = nominal // 15  # 2 for 29.97, 4 for 59.94
        frames_per_minute = nominal * 60 - drop
        frames_per_10min = frames_per_minute * 10 + drop

        d = frame // frames_per_10min
        m = frame % frames_per_10min
        if m > drop:
            adjusted = frame + drop * 9 * d + drop * ((m - drop) // frames_per_minute)
        else:
            adjusted = frame + drop * 9 * d

        ff = adjusted % nominal
        ss = (adjusted // nominal) % 60
        mm = (adjusted // (nominal * 60)) % 60
        hh = adjusted // (nominal * 3600)
        return f"{hh:02d}:{mm:02d}:{ss:02d};{ff:02d}"

    def parse_timecode(self, timecode: str, *, drop_frame: bool | None = None) -> int:
        """Parse an SMPTE ``HH:MM:SS<sep>FF`` string into a frame number.

        ``drop_frame`` overrides the separator's implied convention when
        provided (useful when parsing externally-labelled strings); when
        ``None`` (default) the separator selects the convention:
        ``:`` → non-drop, ``;`` → drop.

        Raises
        ------
        AnimationValidationError
            For malformed strings, out-of-range fields (``FF >= nominal``,
            ``SS/MM >= 60``), or drop-frame timecodes whose ``FF`` value
            is one that drop-frame is defined to skip.
        """
        if not isinstance(timecode, str):
            raise AnimationValidationError(
                f"timecode must be str, got {type(timecode).__name__}",
                context={"received_type": type(timecode).__name__},
            )
        match = _TC_PATTERN.match(timecode)
        if match is None:
            raise AnimationValidationError(
                f"invalid SMPTE timecode: {timecode!r}",
                context={"timecode": timecode},
            )
        hh_s, mm_s, ss_s, sep, ff_s = match.groups()
        hh, mm, ss, ff = int(hh_s), int(mm_s), int(ss_s), int(ff_s)

        if drop_frame is None:
            drop_frame = (sep == ";")
        elif not isinstance(drop_frame, bool):
            raise AnimationValidationError(
                "drop_frame must be bool or None",
                context={"received_type": type(drop_frame).__name__},
            )

        nominal = round(self.fps)
        if ff >= nominal:
            raise AnimationValidationError(
                f"frame field {ff} exceeds nominal rate {nominal}",
                context={"timecode": timecode, "nominal": nominal},
            )
        if ss >= 60 or mm >= 60:
            raise AnimationValidationError(
                f"invalid minute/second field in timecode {timecode!r}",
                context={"timecode": timecode, "ss": ss, "mm": mm},
            )

        if not drop_frame:
            return (hh * 3600 + mm * 60 + ss) * nominal + ff

        if nominal not in (30, 60):
            raise AnimationValidationError(
                f"drop-frame timecode is only defined for NTSC 30000/1001 "
                f"and 60000/1001, got {self.fps}",
                context={"fps": str(self.fps)},
            )
        drop = nominal // 15
        # Frames 00..(drop-1) of second 00 are skipped at minute marks
        # that are not multiples of 10.
        if ss == 0 and ff < drop and mm % 10 != 0:
            raise AnimationValidationError(
                f"timecode {timecode!r} uses a drop-frame-skipped frame number",
                context={"timecode": timecode, "nominal": nominal, "drop": drop},
            )
        total_minutes = hh * 60 + mm
        dropped = drop * (total_minutes - total_minutes // 10)
        physical = (hh * 3600 + mm * 60 + ss) * nominal + ff
        return physical - dropped

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Timebase({self.fps.numerator}, {self.fps.denominator})"

    def __str__(self) -> str:
        if self.fps.denominator == 1:
            return f"{self.fps.numerator} fps"
        return (
            f"{self.fps.numerator}/{self.fps.denominator} fps "
            f"({float(self.fps):.6f})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Timebase):
            return NotImplemented
        return self.fps == other.fps

    def __hash__(self) -> int:
        return hash(self.fps)


__all__ = [
    "Timebase",
]