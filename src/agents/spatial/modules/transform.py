"""Rigid-transform mathematics for the Spatial subsystem.

Conventions
-----------
A :class:`RigidTransform` maps coordinates from ``source_frame`` to
``target_frame`` using ``p_target = R @ p_source + t``. Composition follows
that semantic order: ``a_to_b.then(b_to_c)`` returns ``a_to_c``.

The SO(3)/SE(3) operations follow standard robotics treatments in Lynch &
Park and Solà et al. Rotations are right-handed and represented by proper
orthonormal matrices with determinant +1.
"""
from __future__ import annotations

__version__ = "2.3.0"

import numpy as np # type: ignore

from dataclasses import dataclass
from math import acos, cos, sin
from typing import Any

from ..utils.spatial_errors import SpatialFrameError, SpatialValidationError
from ..utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Transform")
printer = PrettyPrinter()


def skew(vector: Any) -> np.ndarray:
    x, y, z = finite_vector(vector, name="vector", dimension=3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def is_rotation_matrix(matrix: Any, *, tolerance: float = DEFAULT_ANGLE_TOL) -> bool:
    try:
        rotation = finite_array(matrix, name="rotation", shape=(3, 3))
    except SpatialValidationError:
        return False
    identity = np.eye(3, dtype=float)
    return bool(
        np.allclose(rotation.T @ rotation, identity, atol=tolerance, rtol=0.0)
        and abs(float(np.linalg.det(rotation)) - 1.0) <= tolerance
    )


def validate_rotation_matrix(matrix: Any, *, tolerance: float = DEFAULT_ANGLE_TOL) -> np.ndarray:
    rotation = finite_array(matrix, name="rotation", shape=(3, 3))
    if not is_rotation_matrix(rotation, tolerance=tolerance):
        raise SpatialFrameError(
            "rotation matrix must be orthonormal with determinant +1",
            context={"determinant": float(np.linalg.det(rotation)), "tolerance": tolerance},
        )
    return rotation


def project_to_so3(matrix: Any) -> np.ndarray:
    """Project a near-rotation matrix to SO(3) using its polar/SVD factor."""
    raw = finite_array(matrix, name="matrix", shape=(3, 3))
    u, _, vt = np.linalg.svd(raw)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def rotation_from_axis_angle(axis: Any, angle: float) -> np.ndarray:
    unit = unit_vector(axis, name="axis")
    theta = float(angle)
    k = skew(unit)
    return np.eye(3) + sin(theta) * k + (1.0 - cos(theta)) * (k @ k)


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = cos(float(roll)), sin(float(roll))
    cp, sp = cos(float(pitch)), sin(float(pitch))
    cy, sy = cos(float(yaw)), sin(float(yaw))
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def quaternion_to_rotation(quaternion: Any) -> np.ndarray:
    """Convert quaternion ``(w, x, y, z)`` to a proper rotation matrix."""
    q = finite_vector(quaternion, name="quaternion", dimension=4)
    norm = float(np.linalg.norm(q))
    if norm <= DEFAULT_ABS_TOL:
        raise SpatialFrameError("quaternion norm must be non-zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotation_to_quaternion(matrix: Any) -> np.ndarray:
    """Convert a proper rotation matrix to ``(w, x, y, z)`` quaternion."""
    r = validate_rotation_matrix(matrix)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    else:
        index = int(np.argmax(np.diag(r)))
        if index == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
        elif index == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    if q[0] < 0.0:  # deterministic sign convention
        q = -q
    return q


def so3_exp(omega: Any) -> np.ndarray:
    vector = finite_vector(omega, name="omega", dimension=3)
    theta = float(np.linalg.norm(vector))
    k = skew(vector)
    if theta <= DEFAULT_ANGLE_TOL:
        return project_to_so3(np.eye(3) + k + 0.5 * (k @ k))
    return np.eye(3) + (sin(theta) / theta) * k + ((1.0 - cos(theta)) / (theta * theta)) * (k @ k)


def so3_log(rotation: Any) -> np.ndarray:
    r = validate_rotation_matrix(rotation)
    cos_theta = clamp((float(np.trace(r)) - 1.0) * 0.5, -1.0, 1.0)
    theta = acos(cos_theta)
    if theta <= DEFAULT_ANGLE_TOL:
        return 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if abs(np.pi - theta) <= 1.0e-6:
        eigenvalues, eigenvectors = np.linalg.eig(r)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
        axis = unit_vector(axis, name="rotation_axis")
        return axis * theta
    vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return (theta / (2.0 * sin(theta))) * vee


def se3_exp(twist: Any) -> np.ndarray:
    """Exponential map from a 6-vector ``[v, omega]`` to a 4x4 SE(3) matrix."""
    xi = finite_vector(twist, name="twist", dimension=6)
    v, omega = xi[:3], xi[3:]
    theta = float(np.linalg.norm(omega))
    omega_hat = skew(omega)
    r = so3_exp(omega)
    if theta <= DEFAULT_ANGLE_TOL:
        v_matrix = np.eye(3) + 0.5 * omega_hat + (1.0 / 6.0) * (omega_hat @ omega_hat)
    else:
        theta2 = theta * theta
        v_matrix = (
            np.eye(3)
            + ((1.0 - cos(theta)) / theta2) * omega_hat
            + ((theta - sin(theta)) / (theta2 * theta)) * (omega_hat @ omega_hat)
        )
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = r
    matrix[:3, 3] = v_matrix @ v
    return matrix


def se3_log(transform: Any) -> np.ndarray:
    """Logarithm map from a homogeneous SE(3) matrix to ``[v, omega]``."""
    matrix = finite_array(transform, name="transform", shape=(4, 4))
    if not all_close(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise SpatialFrameError("homogeneous transform bottom row must be [0, 0, 0, 1]")
    r = validate_rotation_matrix(matrix[:3, :3])
    t = matrix[:3, 3]
    omega = so3_log(r)
    theta = float(np.linalg.norm(omega))
    omega_hat = skew(omega)
    if theta <= DEFAULT_ANGLE_TOL:
        v_matrix = np.eye(3) + 0.5 * omega_hat + (1.0 / 6.0) * (omega_hat @ omega_hat)
    else:
        theta2 = theta * theta
        v_matrix = (
            np.eye(3)
            + ((1.0 - cos(theta)) / theta2) * omega_hat
            + ((theta - sin(theta)) / (theta2 * theta)) * (omega_hat @ omega_hat)
        )
    try:
        v = np.linalg.solve(v_matrix, t)
    except np.linalg.LinAlgError as exc:
        raise SpatialFrameError("SE(3) logarithm encountered a singular translation Jacobian", cause=exc) from exc
    return np.concatenate((v, omega))


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """A validated rigid transform mapping source-frame coordinates to target."""

    source_frame: str
    target_frame: str
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_frame", validate_identifier(self.source_frame, name="source_frame"))
        object.__setattr__(self, "target_frame", validate_identifier(self.target_frame, name="target_frame"))
        object.__setattr__(self, "rotation", validate_rotation_matrix(self.rotation).copy())
        object.__setattr__(self, "translation", finite_vector(self.translation, name="translation", dimension=3).copy())

    @classmethod
    def identity(cls, frame_id: str) -> "RigidTransform":
        frame = validate_identifier(frame_id, name="frame_id")
        return cls(frame, frame, np.eye(3), np.zeros(3))

    @classmethod
    def from_matrix(cls, source_frame: str, target_frame: str, matrix: Any) -> "RigidTransform":
        homogeneous = finite_array(matrix, name="matrix", shape=(4, 4))
        if not all_close(homogeneous[3], [0.0, 0.0, 0.0, 1.0]):
            raise SpatialFrameError("invalid homogeneous-transform bottom row")
        return cls(source_frame, target_frame, homogeneous[:3, :3], homogeneous[:3, 3])

    @property
    def matrix(self) -> np.ndarray:
        result = np.eye(4, dtype=float)
        result[:3, :3] = self.rotation
        result[:3, 3] = self.translation
        return result

    @property
    def quaternion(self) -> np.ndarray:
        return rotation_to_quaternion(self.rotation)

    def apply_point(self, point: Any) -> np.ndarray:
        vector = finite_vector(point, name="point", dimension=3)
        return self.rotation @ vector + self.translation

    def apply_vector(self, vector: Any) -> np.ndarray:
        return self.rotation @ finite_vector(vector, name="vector", dimension=3)

    def apply_points(self, points: Any) -> np.ndarray:
        array = finite_array(points, name="points", ndim=2)
        if array.shape[1] != 3:
            raise SpatialValidationError("points must have shape (N, 3)", context={"shape": array.shape})
        return (array @ self.rotation.T) + self.translation

    def inverse(self) -> "RigidTransform":
        rotation = self.rotation.T
        translation = -(rotation @ self.translation)
        return RigidTransform(self.target_frame, self.source_frame, rotation, translation)

    def then(self, next_transform: "RigidTransform") -> "RigidTransform":
        if self.target_frame != next_transform.source_frame:
            raise SpatialFrameError(
                "transform frames do not compose",
                context={
                    "first": f"{self.source_frame}->{self.target_frame}",
                    "second": f"{next_transform.source_frame}->{next_transform.target_frame}",
                },
            )
        rotation = next_transform.rotation @ self.rotation
        translation = next_transform.rotation @ self.translation + next_transform.translation
        return RigidTransform(self.source_frame, next_transform.target_frame, rotation, translation)

    def compose(self, next_transform: "RigidTransform") -> "RigidTransform":
        return self.then(next_transform)

    def almost_equal(self, other: "RigidTransform", *, tolerance: float = DEFAULT_ANGLE_TOL) -> bool:
        return (
            self.source_frame == other.source_frame
            and self.target_frame == other.target_frame
            and np.allclose(self.rotation, other.rotation, atol=tolerance, rtol=0.0)
            and np.allclose(self.translation, other.translation, atol=tolerance, rtol=0.0)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "quaternion_wxyz": self.quaternion.tolist(),
        }


__all__ = [
    "RigidTransform",
    "skew",
    "is_rotation_matrix",
    "validate_rotation_matrix",
    "project_to_so3",
    "rotation_from_axis_angle",
    "rotation_from_rpy",
    "quaternion_to_rotation",
    "rotation_to_quaternion",
    "so3_exp",
    "so3_log",
    "se3_exp",
    "se3_log",
]


if __name__ == "__main__":
    configure_logging()
    transform = RigidTransform("camera", "world", np.eye(3), np.array([1.0, 2.0, 3.0]))
    printer.status("TEST", f"Transform initialized: {transform.to_dict()}", "info")
