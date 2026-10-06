"""Explicit controlled refinement of uncertain cameras against fixed evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from openmocap.types import Camera, CameraPose, ParameterState, SolveDiagnostics


@dataclass(frozen=True)
class CameraRefinementConfig:
    max_rotation_radians: float = 0.035
    max_translation_metres: float = 0.03
    rotation_prior_sigma: float = 0.01
    translation_prior_sigma: float = 0.01
    robust_scale_px: float = 2.0
    state: ParameterState = ParameterState.BOUNDED


def prefer_authoritative_camera(camera: Camera | None, learned_camera: Camera | None) -> Camera:
    """A learned estimate is considered only when calibrated geometry is absent."""
    if camera is not None:
        return camera
    if learned_camera is None:
        raise ValueError("No camera calibration exists; metric reconstruction cannot proceed")
    if learned_camera.quality.metric_scale_known is False:
        raise ValueError("Learned camera estimate lacks metric scale")
    return learned_camera


def refine_camera(
    camera: Camera,
    fixed_world_points: ArrayLike,
    pixels: ArrayLike,
    confidence: ArrayLike | None = None,
    config: CameraRefinementConfig | None = None,
) -> tuple[Camera, SolveDiagnostics]:
    """Return a new camera with bounded deltas. Never optimize camera and body together.

    Locked cameras return unchanged transforms and an explicit diagnostic. The
    input camera is never mutated. A BOUNDED control may further tighten limits.
    """
    config = config or CameraRefinementConfig()
    result = deepcopy(camera)
    control = camera.controls.get("extrinsics")
    state = control.state if control else config.state
    if camera.locked or state == ParameterState.LOCKED:
        return result, SolveDiagnostics(
            warnings=["Authoritative camera is locked; refinement skipped"],
            converged=True,
            parameter_deltas={"rotation_radians": 0.0, "translation_metres": 0.0},
        )
    if camera.trajectory is not None:
        raise ValueError("Static camera refinement cannot rewrite a moving-camera trajectory")
    world = np.asarray(fixed_world_points, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if len(world) != len(pixels) or len(world) < 6:
        raise ValueError("Camera refinement requires six or more fixed correspondences")
    weights = (
        np.ones(len(world)) if confidence is None else np.asarray(confidence, dtype=np.float64)
    )
    if weights.shape != (len(world),) or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("Correspondence weights must be finite and nonnegative")
    if min(config.rotation_prior_sigma, config.translation_prior_sigma) <= 0:
        raise ValueError("Camera refinement requires positive prior sigmas")

    def make_pose(delta: np.ndarray) -> CameraPose:
        rotation = cv2.Rodrigues(delta[:3])[0] @ camera.pose.rotation
        return CameraPose(rotation, camera.pose.translation + delta[3:])

    def residual(delta: np.ndarray) -> np.ndarray:
        result.pose = make_pose(delta)
        data = np.sqrt(weights[:, None]) * (result.project(world) - pixels)
        priors = np.concatenate(
            [delta[:3] / config.rotation_prior_sigma, delta[3:] / config.translation_prior_sigma]
        )
        return np.concatenate([np.nan_to_num(data.reshape(-1), nan=1e6), priors])

    limits = np.array([config.max_rotation_radians] * 3 + [config.max_translation_metres] * 3)
    lower, upper = -limits, limits
    if control and control.lower is not None:
        lower = np.maximum(lower, control.lower)
    if control and control.upper is not None:
        upper = np.minimum(upper, control.upper)
    if state == ParameterState.FREE:
        lower, upper = np.full(6, -np.inf), np.full(6, np.inf)
    before = float(np.mean(np.linalg.norm(camera.project(world) - pixels, axis=1)))
    fit = least_squares(
        residual, np.zeros(6), bounds=(lower, upper), loss="huber", f_scale=config.robust_scale_px
    )
    result.pose = make_pose(fit.x)
    result.source = "human_assisted"
    after = float(np.mean(np.linalg.norm(result.project(world) - pixels, axis=1)))
    diagnostics = SolveDiagnostics(
        objective_terms={
            "reprojection_before_px": before,
            "reprojection_after_px": after,
            "camera_prior": float(np.sum(residual(fit.x)[-6:] ** 2)),
        },
        converged=bool(fit.success),
        iterations=fit.nfev,
        parameter_deltas={
            "rotation_radians": float(np.linalg.norm(fit.x[:3])),
            "translation_metres": float(np.linalg.norm(fit.x[3:])),
            "delta_vector": fit.x.tolist(),
        },
    )
    return result, diagnostics
