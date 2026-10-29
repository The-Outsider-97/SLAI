"""Headless playback position mapping; rendering is delegated to applications."""
from dataclasses import dataclass
from math import isfinite

@dataclass
class Playback:
    duration: float
    position: float = 0.0
    rate: float = 1.0
    playing: bool = False
    loop: bool = False

    def __post_init__(self):
        if not isfinite(self.duration) or self.duration <= 0:
            raise ValueError("duration must be finite positive")
        self.seek(self.position)

    def seek(self, position: float):
        if not isfinite(position) or not 0 <= position <= self.duration:
            raise ValueError("seek outside timeline")
        self.position = position
        return self.position

    def advance(self, elapsed: float):
        if not isfinite(elapsed) or elapsed < 0 or not isfinite(self.rate):
            raise ValueError("invalid playback step")
        if self.playing:
            proposed = self.position + elapsed * self.rate
            self.position = proposed % self.duration if self.loop else max(0., min(self.duration, proposed))
            if not self.loop and (proposed <= 0 or proposed >= self.duration):
                self.playing = False
        return self.position
