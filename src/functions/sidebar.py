"""Sidebar helpers with animation presets and interpolation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .dropdown import EASING_PRESETS, INTERPOLATION_STRATEGIES


@dataclass
class SidebarSection:
    title: str
    expanded: bool = True


@dataclass(frozen=True)
class SidebarAnimation:
    duration_ms: int = 260
    preset: str = "snappy"

    @property
    def easing(self) -> str:
        return EASING_PRESETS.get(self.preset, EASING_PRESETS["smooth"])


class Sidebar:
    def __init__(self, sections: Iterable[str], animation: SidebarAnimation | None = None) -> None:
        sections = list(sections)
        if not sections:
            raise ValueError("Sidebar requires at least one section")

        self.animation = animation or SidebarAnimation()
        self.hidden = False
        self.sections: Dict[str, SidebarSection] = {
            name: SidebarSection(title=name, expanded=True) for name in sections
        }

    def transition_style(self) -> str:
        # Qt QSS does not support CSS transitions; animations should use Qt animation APIs.
        return ""

    def visibility_keyframes(self, steps: int = 10, strategy: str = "ease_out") -> List[float]:
        if steps <= 0:
            raise ValueError("steps must be > 0")
        interpolator = INTERPOLATION_STRATEGIES.get(strategy)
        if interpolator is None:
            raise ValueError(f"Unknown interpolation strategy: {strategy}")
        return [round(interpolator(i / steps), 4) for i in range(steps + 1)]

    def hide(self) -> None:
        self.hidden = True

    def show(self) -> None:
        self.hidden = False

    def toggle_visibility(self) -> bool:
        self.hidden = not self.hidden
        return self.hidden

    def toggle_section(self, section_name: str) -> bool:
        section = self.sections.get(section_name)
        if not section:
            raise ValueError(f"Unknown section: {section_name}")
        section.expanded = not section.expanded
        return section.expanded
