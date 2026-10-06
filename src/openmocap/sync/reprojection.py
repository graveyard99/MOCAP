"""Controlled camera clock refinement against a frozen metric trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from openmocap.types import Camera, Pose2DObservation, TimeMapping

from .timing import refine_time_mapping


class EvaluatedTrajectory(Protocol):
    def evaluate(self, times: np.ndarray | float, derivative: int = 0) -> np.ndarray: ...


@dataclass
class TimingRefinementResult:
    mappings: dict[str, TimeMapping]
    diagnostics: dict[str, dict[str, Any]]


def refine_multiview_timing(
    cameras: list[Camera],
    observations: list[Pose2DObservation],
    trajectory: EvaluatedTrajectory,
    joint_names: list[str] | tuple[str, ...],
    *,
    max_offset_change: float = 0.02,
    max_drift_change: float = 0.0001,
    offset_prior_sigma: float = 0.01,
    drift_prior_sigma: float = 0.0001,
    minimum_confidence: float = 0.5,
    minimum_observations: int = 12,
) -> TimingRefinementResult:
    """Return candidate clock mappings without mutating cameras or raw data.

    The caller must fit/freeze motion from reliable camera evidence first.
    Camera geometry remains authoritative and fixed. Only clocks marked
    unlocked are eligible. Locked reference cameras fix the time gauge.
    Candidate residuals use original distorted plate pixels and actual camera
    timestamps, including time-varying camera poses/lenses when supplied.
    """
    if not 0 <= minimum_confidence <= 1 or minimum_observations < 1:
        raise ValueError(
            "Timing refinement needs confidence in [0,1] and a positive observation count"
        )
    if min(max_offset_change, max_drift_change, offset_prior_sigma, drift_prior_sigma) <= 0:
        raise ValueError("Timing refinement bounds and prior sigmas must be positive")
    if not any(camera.time_mapping.locked for camera in cameras if camera.enabled):
        raise ValueError(
            "Timing refinement requires a locked reference camera to fix the time gauge"
        )
    lookup = {name: index for index, name in enumerate(joint_names)}
    camera_lookup = {camera.id: camera for camera in cameras}
    if len(camera_lookup) != len(cameras):
        raise ValueError("Camera IDs must be unique")
    grouped: dict[str, list[Pose2DObservation]] = {camera.id: [] for camera in cameras}
    for observation in observations:
        if observation.camera_id not in camera_lookup:
            raise ValueError(f"Unknown observation camera {observation.camera_id}")
        if observation.joint_id not in lookup:
            raise ValueError(f"Unknown observation joint {observation.joint_id}")
        if (
            observation.enabled
            and observation.confidence > 0
            and observation.confidence >= minimum_confidence
            and np.isfinite(observation.xy).all()
        ):
            grouped[observation.camera_id].append(observation)
    mappings, diagnostics = {}, {}
    for camera in cameras:
        mapping = camera.time_mapping
        mappings[camera.id] = mapping
        if mapping.locked or not camera.enabled or camera.quality.confidence == 0:
            diagnostics[camera.id] = {
                "locked": mapping.locked,
                "enabled": camera.enabled,
                "offset_delta": 0.0,
                "scale_delta": 0.0,
                "geometry_modified": False,
            }
            continue
        # Keep a fixed residual dimension and avoid spline extrapolation for
        # all candidate clocks, not just the initial mapping.
        accepted = []
        for observation in grouped[camera.id]:
            corners = np.array(
                [
                    (mapping.scale + ds) * observation.camera_timestamp + mapping.offset + do
                    for ds in (-max_drift_change, max_drift_change)
                    for do in (-max_offset_change, max_offset_change)
                ]
            )
            prediction = trajectory.evaluate(corners)[:, lookup[observation.joint_id]]
            if not np.isfinite(prediction).all():
                continue
            if camera.trajectory is not None and (
                corners.min() < camera.trajectory.times[0]
                or corners.max() > camera.trajectory.times[-1]
            ):
                continue
            accepted.append(observation)
        if len(accepted) < minimum_observations:
            diagnostics[camera.id] = {
                "locked": False,
                "warning": "insufficient observations over bounded clock support",
                "observation_count": len(accepted),
                "offset_delta": 0.0,
                "scale_delta": 0.0,
                "geometry_modified": False,
            }
            continue
        timestamps = np.array([observation.camera_timestamp for observation in accepted])
        indices = np.array([lookup[observation.joint_id] for observation in accepted])
        measured = np.array([observation.xy for observation in accepted])
        weights = np.sqrt(
            np.array([observation.confidence for observation in accepted])
            * camera.quality.confidence
        )

        def residual(offset: float, scale: float) -> np.ndarray:
            world = scale * timestamps + offset
            points = trajectory.evaluate(world)[np.arange(len(world)), indices]
            predicted = np.array(
                [
                    camera.project(point, time=float(time))
                    for point, time in zip(points, world, strict=True)
                ]
            )
            return ((predicted - measured) * weights[:, None]).ravel()

        candidate, detail = refine_time_mapping(
            mapping,
            residual,
            max_offset_change=max_offset_change,
            max_drift_change=max_drift_change,
            offset_prior_sigma=offset_prior_sigma,
            drift_prior_sigma=drift_prior_sigma,
        )
        initial_pixels = np.linalg.norm(
            residual(mapping.offset, mapping.scale).reshape(-1, 2) / weights[:, None], axis=1
        )
        final_pixels = np.linalg.norm(
            residual(candidate.offset, candidate.scale).reshape(-1, 2) / weights[:, None], axis=1
        )
        detail.update(
            {
                "observation_count": len(accepted),
                "before_median_reprojection_px": float(np.median(initial_pixels)),
                "after_median_reprojection_px": float(np.median(final_pixels)),
                "geometry_modified": False,
                "applied": False,
            }
        )
        mappings[camera.id] = candidate
        diagnostics[camera.id] = detail
    return TimingRefinementResult(mappings, diagnostics)
