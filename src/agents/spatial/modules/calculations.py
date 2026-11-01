"""Spatially specialised deterministic calculations.

Grounded primarily in de Berg et al. (computational geometry), Burago et al.
(metric spaces) and Ericson (robust real-time geometric primitives). General
scientific computing remains outside this module.
"""
from __future__ import annotations

__version__ = "2.3.0"

import numpy as np # type: ignore

from typing import Any

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.spatial_errors import SpatialGeometryError
from ..utils.spatial_helpers import DEFAULT_ABS_TOL, finite_vector, unit_vector
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Calculations")
printer = PrettyPrinter()


class Calculations:
    def __init__(self) -> None:
        self.config = load_global_config()
        self.calc_config = get_config_section("calculations", config=self.config, default={})
        self.tolerance = float(self.calc_config.get("tolerance", DEFAULT_ABS_TOL))
        if self.tolerance <= 0.0:
            self.tolerance = DEFAULT_ABS_TOL

    @staticmethod
    def squared_distance(a: Any, b: Any) -> float:
        av = finite_vector(a, name="a")
        bv = finite_vector(b, name="b", dimension=av.size)
        delta = av - bv
        return float(delta @ delta)

    @classmethod
    def euclidean_distance(cls, a: Any, b: Any) -> float:
        return float(np.sqrt(cls.squared_distance(a, b)))

    @staticmethod
    def manhattan_distance(a: Any, b: Any) -> float:
        av = finite_vector(a, name="a")
        bv = finite_vector(b, name="b", dimension=av.size)
        return float(np.abs(av - bv).sum())

    @staticmethod
    def dot(a: Any, b: Any) -> float:
        av = finite_vector(a, name="a")
        bv = finite_vector(b, name="b", dimension=av.size)
        return float(av @ bv)

    @staticmethod
    def cross(a: Any, b: Any) -> np.ndarray:
        av = finite_vector(a, name="a", dimension=3)
        bv = finite_vector(b, name="b", dimension=3)
        return np.cross(av, bv)

    @staticmethod
    def projection(vector: Any, onto: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> np.ndarray:
        v = finite_vector(vector, name="vector")
        u = finite_vector(onto, name="onto", dimension=v.size)
        denominator = float(u @ u)
        if denominator <= tolerance * tolerance:
            raise SpatialGeometryError("cannot project onto a zero-length vector")
        return (float(v @ u) / denominator) * u

    @staticmethod
    def closest_point_on_segment(point: Any, start: Any, end: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> np.ndarray:
        p = finite_vector(point, name="point")
        a = finite_vector(start, name="start", dimension=p.size)
        b = finite_vector(end, name="end", dimension=p.size)
        ab = b - a
        denominator = float(ab @ ab)
        if denominator <= tolerance * tolerance:
            return a.copy()
        t = float((p - a) @ ab) / denominator
        t = min(max(t, 0.0), 1.0)
        return a + t * ab

    @classmethod
    def point_segment_distance(cls, point: Any, start: Any, end: Any) -> float:
        closest = cls.closest_point_on_segment(point, start, end)
        return cls.euclidean_distance(point, closest)

    @staticmethod
    def signed_distance_to_plane(point: Any, plane_point: Any, plane_normal: Any) -> float:
        p = finite_vector(point, name="point", dimension=3)
        origin = finite_vector(plane_point, name="plane_point", dimension=3)
        normal = unit_vector(plane_normal, name="plane_normal")
        return float((p - origin) @ normal)

    @classmethod
    def point_plane_distance(cls, point: Any, plane_point: Any, plane_normal: Any) -> float:
        return abs(cls.signed_distance_to_plane(point, plane_point, plane_normal))

    @staticmethod
    def orientation_2d(a: Any, b: Any, c: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> int:
        av = finite_vector(a, name="a", dimension=2)
        bv = finite_vector(b, name="b", dimension=2)
        cv = finite_vector(c, name="c", dimension=2)
        determinant = float((bv[0] - av[0]) * (cv[1] - av[1]) - (bv[1] - av[1]) * (cv[0] - av[0]))
        if abs(determinant) <= tolerance:
            return 0
        return 1 if determinant > 0.0 else -1

    @staticmethod
    def barycentric_coordinates(point: Any, a: Any, b: Any, c: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> np.ndarray:
        p = finite_vector(point, name="point", dimension=3)
        av = finite_vector(a, name="a", dimension=3)
        bv = finite_vector(b, name="b", dimension=3)
        cv = finite_vector(c, name="c", dimension=3)
        v0, v1, v2 = bv - av, cv - av, p - av
        d00, d01, d11 = float(v0 @ v0), float(v0 @ v1), float(v1 @ v1)
        d20, d21 = float(v2 @ v0), float(v2 @ v1)
        denominator = d00 * d11 - d01 * d01
        if abs(denominator) <= tolerance:
            raise SpatialGeometryError("triangle is degenerate; barycentric coordinates are undefined")
        v = (d11 * d20 - d01 * d21) / denominator
        w = (d00 * d21 - d01 * d20) / denominator
        u = 1.0 - v - w
        return np.array([u, v, w], dtype=float)

    @staticmethod
    def aabb_distance(point: Any, minimum: Any, maximum: Any) -> float:
        p = finite_vector(point, name="point")
        lo = finite_vector(minimum, name="minimum", dimension=p.size)
        hi = finite_vector(maximum, name="maximum", dimension=p.size)
        if np.any(lo > hi):
            raise SpatialGeometryError("minimum must not exceed maximum")
        delta = np.maximum(np.maximum(lo - p, 0.0), p - hi)
        return float(np.linalg.norm(delta))


__all__ = ["Calculations"]


if __name__ == "__main__":
    configure_logging()
    calc = Calculations()
    printer.status("TEST", f"distance={calc.euclidean_distance((0, 0, 0), (1, 1, 1)):.6f}", "info")
