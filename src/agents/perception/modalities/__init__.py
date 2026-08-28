"""Internal modality pipelines for SLAI perception.

These classes are computational components owned by ``PerceptionAgent``; they
are intentionally not ``BaseAgent`` subclasses and are not independently
registered with the SLAI agent factory.
"""

from .base import BasePerceptionModality
from .audio import AudioPerception
from .text import TextPerception
from .vision import VisionPerception


__all__ = [
    "BasePerceptionModality",
    "AudioPerception",
    "TextPerception",
    "VisionPerception",
]
