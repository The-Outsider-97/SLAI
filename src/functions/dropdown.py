"""Dropdown menu state helpers with animation presets and interpolation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from .utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Dropdown Menu")
printer = PrettyPrinter()


EASING_PRESETS: Dict[str, str] = {
    "smooth": "ease-in-out",
    "snappy": "cubic-bezier(0.2, 0.8, 0.2, 1)",
    "gentle": "cubic-bezier(0.25, 0.46, 0.45, 0.94)",
    "spring": "cubic-bezier(0.34, 1.56, 0.64, 1)",
}


def _linear(t: float) -> float:
    return t


def _ease_in(t: float) -> float:
    return t * t


def _ease_out(t: float) -> float:
    return 1 - (1 - t) * (1 - t)


def _ease_in_out(t: float) -> float:
    return 2 * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 2) / 2


INTERPOLATION_STRATEGIES: Dict[str, Callable[[float], float]] = {
    "linear": _linear,
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "ease_in_out": _ease_in_out,
}


@dataclass(frozen=True)
class DropdownOption:
    label: str
    value: str


@dataclass(frozen=True)
class AnimationConfig:
    duration_ms: int = 220
    preset: str = "smooth"

    @property
    def easing(self) -> str:
        return EASING_PRESETS.get(self.preset, EASING_PRESETS["smooth"])


class DropdownMenu:
    """Reusable dropdown model with smooth transition descriptors."""

    def __init__(
        self,
        options: Iterable[DropdownOption],
        default_value: Optional[str] = None,
        animation: Optional[AnimationConfig] = None,
    ) -> None:
        self.config = load_global_config()
        animation_config = get_config_section("dropdown_animation", self.config)

        self.options: List[DropdownOption] = list(options)
        if not self.options:
            raise ValueError("Dropdown must contain at least one option")

        self.animation = animation or AnimationConfig(
            duration_ms=int(animation_config.get("duration_ms", 220)),
            preset=str(animation_config.get("preset", "smooth")),
        )
        if self.animation.duration_ms < 0:
            raise ValueError("animation duration_ms must be >= 0")
        if self.animation.preset not in EASING_PRESETS:
            raise ValueError(f"Unknown animation preset: {self.animation.preset}")
        self.is_open = bool(animation_config.get("is_open", False))
        self.selected_value = self.options[0].value if default_value is None else default_value
        self._validate_selected_value(self.selected_value)

        logger.info(f"Dropdown Menu successfully initialized")

    def _validate_selected_value(self, value: str) -> None:
        if value not in {option.value for option in self.options}:
            raise ValueError(f"Invalid option value: {value}")

    def transition_style(self) -> str:
        """
        Return stylesheet fragments safe for Qt stylesheets (QSS).

        Qt does not support the web CSS ``transition`` property; animation is handled
        through QPropertyAnimation in the UI layer instead.
        """
        return ""

    def animation_frames(self, steps: int = 8, strategy: str = "ease_in_out") -> List[float]:
        if steps <= 0:
            raise ValueError("steps must be > 0")
        interpolator = INTERPOLATION_STRATEGIES.get(strategy)
        if interpolator is None:
            raise ValueError(f"Unknown interpolation strategy: {strategy}")
        return [round(interpolator(i / steps), 4) for i in range(steps + 1)]

    def toggle(self) -> bool:
        self.is_open = not self.is_open
        return self.is_open

    def close(self) -> None:
        """Close the dropdown menu (sets is_open to False)."""
        self.is_open = False

    def select(self, value: str) -> str:
        self._validate_selected_value(value)
        self.selected_value = value
        self.is_open = False
        return self.selected_value


__all__ = [
    # Dropdown
    "AnimationConfig",
    "DropdownMenu",
    "DropdownOption",
    "EASING_PRESETS",
    "INTERPOLATION_STRATEGIES",
]