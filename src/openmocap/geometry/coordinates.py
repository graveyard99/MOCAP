"""Centralized right-handed coordinate conversions, all distances in metres."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from openmocap.types import FloatArray

# B maps canonical metric +Y-up world points into a named convention.
# Blender is +Z up and -Y forward; OpenCV axes below refer to a camera *local*
# orientation at the origin, not to an arbitrary surveyed camera transform.
BASES: dict[str, FloatArray] = {
    "world": np.eye(3),
    "smpl": np.eye(3),
    "maya": np.eye(3),
    "fbx": np.eye(3),
    "houdini": np.eye(3),
    "blender": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]),
    "opencv": np.diag([1.0, -1.0, -1.0]),
}


def conversion_matrix(source: str, target: str) -> FloatArray:
    try:
        return BASES[target] @ BASES[source].T
    except KeyError as error:
        raise ValueError(f"Unknown coordinate convention: {error.args[0]}") from error


def convert_points(
    points: ArrayLike,
    source: str,
    target: str,
    source_units: str = "metres",
    target_units: str = "metres",
) -> FloatArray:
    units = {"metres": 1.0, "centimetres": 0.01, "millimetres": 0.001}
    if source_units not in units or target_units not in units:
        raise ValueError("Units must be metres, centimetres, or millimetres")
    return (
        np.asarray(points, dtype=np.float64)
        @ conversion_matrix(source, target).T
        * (units[source_units] / units[target_units])
    )


def convert_rotation(rotation: ArrayLike, source: str, target: str) -> FloatArray:
    basis = conversion_matrix(source, target)
    return basis @ np.asarray(rotation, dtype=np.float64) @ basis.T


def align_similarity(
    source_points: ArrayLike, target_points: ArrayLike, *, estimate_scale: bool = True
) -> tuple[float, FloatArray, FloatArray]:
    """Umeyama alignment; surveyed target positions establish metric scale.

    Returns scale, rotation, translation such that target = scale*R*source+t.
    Reflection, collinear controls, and non-positive scale are rejected.
    """
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("Alignment requires three or more matching 3D control points")
    a, b = source - source.mean(0), target - target.mean(0)
    if np.linalg.matrix_rank(a) < 2:
        raise ValueError("Surveyed control points must not be collinear")
    u, singular_values, vt = np.linalg.svd(b.T @ a / len(source))
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1
    rotation = u @ np.diag(d) @ vt
    scale = (
        float((singular_values * d).sum() / np.mean(np.sum(a * a, axis=1)))
        if estimate_scale
        else 1.0
    )
    if scale <= 0:
        raise ValueError("Metric alignment recovered a non-positive scale")
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    return scale, rotation, translation


def scale_from_distance(point_a: ArrayLike, point_b: ArrayLike, known_metres: float) -> float:
    distance = float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))
    if distance <= 1e-12 or known_metres <= 0:
        raise ValueError("Metric reference must have positive known and reconstructed distance")
    return known_metres / distance
