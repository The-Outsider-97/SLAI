"""Rational frame-rate conversion without accumulated floating point drift."""
from fractions import Fraction

class Timebase:
    def __init__(self, numerator: int = 30, denominator: int = 1):
        if type(numerator) is not int or type(denominator) is not int or numerator <= 0 or denominator <= 0:
            raise ValueError("frame rate components must be positive integers")
        self.fps = Fraction(numerator, denominator)

    def frame_seconds(self, frame: int) -> Fraction:
        if type(frame) is not int:
            raise TypeError("frame must be integer")
        return Fraction(frame, 1) / self.fps

    def seconds_frame(self, seconds) -> Fraction:
        return Fraction(seconds) * self.fps
