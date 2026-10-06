"""Confidence-weighted robust N-view geometric reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

from openmocap.cameras.models import project
from openmocap.types import (
    Camera,
    FloatArray,
    Joint3DObservation,
    Pose2DObservation,
    TriangulationDiagnostics,
)


@dataclass(frozen=True)
class TriangulationConfig:
    confidence_threshold: float = 0.1
    reprojection_threshold_px: float = 5.0
    robust_scale_px: float = 2.0
    minimum_ray_angle_degrees: float = 1.0
    max_hypotheses: int = 64
    minimum_views: int = 2
    max_iterations: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be in [0,1]")
        if min(self.reprojection_threshold_px, self.robust_scale_px) <= 0:
            raise ValueError("Reprojection thresholds must be positive pixels")
        if self.minimum_views < 2 or self.max_hypotheses < 1:
            raise ValueError("Triangulation needs two or more views and positive hypotheses")


def weighted_dlt(
    cameras: list[Camera],
    observations: list[Pose2DObservation],
    times: list[float],
    weights: FloatArray,
) -> FloatArray:
    rows = []
    for camera, observation, time, weight in zip(
        cameras, observations, times, weights, strict=True
    ):
        x, y = camera.undistort(observation.xy, time)
        p = camera.pose_at(time).projection
        rows.extend([np.sqrt(weight) * (x * p[2] - p[0]), np.sqrt(weight) * (y * p[2] - p[1])])
    return _dlt_from_rows(np.asarray(rows))


def _dlt_from_rows(rows: FloatArray) -> FloatArray:
    _, _, vt = np.linalg.svd(rows)
    if abs(vt[-1, 3]) <= 1e-12:
        return np.full(3, np.nan)
    return vt[-1, :3] / vt[-1, 3]


def triangulate(
    cameras: dict[str, Camera] | list[Camera],
    observations: list[Pose2DObservation],
    time: float | None = None,
    config: TriangulationConfig | None = None,
) -> Joint3DObservation:
    """Robustly reconstruct one actor joint from every available camera.

    Hypotheses initialize consensus only; nonlinear refinement always uses all
    surviving views. Invalid, disabled, low-confidence and outlier measurements
    are retained as rejected IDs in diagnostics. Camera objects are read-only.
    """
    config = config or TriangulationConfig()
    if isinstance(cameras, list):
        if len({c.id for c in cameras}) != len(cameras):
            raise ValueError("Camera IDs must be unique")
        cameras = {c.id: c for c in cameras}
    if len({o.person_id for o in observations}) > 1 or len({o.joint_id for o in observations}) > 1:
        raise ValueError("Cross-person or cross-joint triangulation is prohibited")
    unknown = {o.camera_id for o in observations} - cameras.keys()
    if unknown:
        raise ValueError(f"Observations reference unknown cameras: {sorted(unknown)}")

    # One view per camera: a repeated detection must not inflate support.
    chosen: dict[str, Pose2DObservation] = {}
    rejected = set()
    for observation in observations:
        camera = cameras[observation.camera_id]
        if (
            not observation.enabled
            or not camera.enabled
            or observation.confidence < config.confidence_threshold
            or camera.quality.confidence <= 0
            or (camera.trajectory is not None and camera.trajectory.confidence <= 0)
            or not np.isfinite(observation.xy).all()
        ):
            rejected.add(camera.id)
            continue
        if camera.id not in chosen or observation.confidence > chosen[camera.id].confidence:
            chosen[camera.id] = observation
    obs = list(chosen.values())
    cams = [cameras[o.camera_id] for o in obs]
    if any(not c.quality.metric_scale_known for c in cams):
        raise ValueError("Metric triangulation requires an established camera-world scale")
    times = [
        float(
            time
            if time is not None
            else (
                o.world_timestamp
                if o.world_timestamp is not None
                else c.time_mapping.to_world(o.camera_timestamp)
            )
        )
        for c, o in zip(cams, obs, strict=True)
    ]
    calibration_confidence = np.array(
        [c.quality.confidence * (c.trajectory.confidence if c.trajectory else 1.0) for c in cams]
    )
    weights = np.array(
        [
            o.confidence**2
            * quality**2
            / (
                (float(np.trace(o.covariance) / 2) if o.covariance is not None else 1.0)
                + (c.quality.reprojection_rmse or 0.0) ** 2
            )
            for c, o, quality in zip(cams, obs, calibration_confidence, strict=True)
        ]
    )
    first = observations[0] if observations else None
    timestamp = float(time if time is not None else np.median(times)) if times else float(time or 0)
    if len(obs) < config.minimum_views:
        diagnostics = TriangulationDiagnostics(
            rejected_camera_ids=sorted(rejected),
            degenerate=True,
            warnings=["Insufficient valid camera views"],
        )
        return Joint3DObservation(
            np.full(3, np.nan),
            0.0,
            diagnostics,
            first.joint_id if first else "",
            first.person_id if first else "",
            timestamp,
        )

    poses = [c.pose_at(t) for c, t in zip(cams, times, strict=True)]
    lenses = [c.intrinsics_at(t) for c, t in zip(cams, times, strict=True)]
    normalized = [c.undistort(o.xy, t) for c, o, t in zip(cams, obs, times, strict=True)]
    dlt_rows = np.array(
        [
            np.sqrt(weight)
            * np.stack(
                [
                    xy[0] * pose.projection[2] - pose.projection[0],
                    xy[1] * pose.projection[2] - pose.projection[1],
                ]
            )
            for pose, xy, weight in zip(poses, normalized, weights, strict=True)
        ]
    )

    def errors(point: FloatArray) -> FloatArray:
        return np.array(
            [
                np.linalg.norm(project(point, lens, pose) - o.xy)
                for lens, pose, o in zip(lenses, poses, obs, strict=True)
            ]
        )

    full = _dlt_from_rows(dlt_rows.reshape(-1, 4))
    hypotheses = [full]
    pairs = list(combinations(range(len(cams)), 2))
    if len(pairs) > config.max_hypotheses:
        indices = np.random.default_rng(config.seed).choice(
            len(pairs), config.max_hypotheses, replace=False
        )
        pairs = [pairs[i] for i in indices]
    for pair in pairs:
        indices = np.array(pair)
        hypotheses.append(_dlt_from_rows(dlt_rows[indices].reshape(-1, 4)))
    best_score = (-1.0, float("-inf"))
    point = full
    inliers = np.ones(len(obs), dtype=bool)
    finite_hypotheses = np.array([h for h in hypotheses if np.isfinite(h).all()]).reshape(-1, 3)
    # One vectorized projection per camera for all hypotheses: avoids thousands
    # of small OpenCV calls with 30+ views while preserving lens correctness.
    hypothesis_errors = (
        np.column_stack(
            [
                np.linalg.norm(project(finite_hypotheses, lens, pose) - o.xy, axis=1)
                for lens, pose, o in zip(lenses, poses, obs, strict=True)
            ]
        )
        if len(finite_hypotheses)
        else np.empty((0, len(cams)))
    )
    for hypothesis, e in zip(finite_hypotheses, hypothesis_errors, strict=True):
        mask = np.isfinite(e) & (e <= config.reprojection_threshold_px)
        score = (
            float(weights[mask].sum()),
            -float(
                np.sum(
                    weights[mask] * np.minimum(e[mask] ** 2, config.reprojection_threshold_px**2)
                )
            ),
        )
        if score > best_score:
            best_score, point, inliers = score, hypothesis, mask
    if inliers.sum() < config.minimum_views:
        diagnostics = TriangulationDiagnostics(
            rejected_camera_ids=sorted(set(cameras[c.id].id for c in cams) | rejected),
            degenerate=True,
            warnings=["No multiview consensus"],
        )
        return Joint3DObservation(
            np.full(3, np.nan),
            0.0,
            diagnostics,
            first.joint_id if first else "",
            first.person_id if first else "",
            timestamp,
        )

    def residual(point: FloatArray, mask: NDArrayBool) -> FloatArray:
        values = [
            np.sqrt(weights[i]) * (project(point, lenses[i], poses[i]) - obs[i].xy)
            for i in np.flatnonzero(mask)
        ]
        return np.nan_to_num(np.concatenate(values), nan=1e6, posinf=1e6, neginf=-1e6)

    fit = None
    for _ in range(3):
        fit = least_squares(
            residual,
            point,
            args=(inliers,),
            loss="cauchy",
            f_scale=config.robust_scale_px,
            max_nfev=config.max_iterations,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        point = fit.x
        e = errors(point)
        new_inliers = np.isfinite(e) & (e <= config.reprojection_threshold_px)
        if new_inliers.sum() < config.minimum_views or np.array_equal(inliers, new_inliers):
            break
        inliers = new_inliers
    e = errors(point)
    surviving = np.flatnonzero(inliers)
    ray_directions = np.array(
        [np.append(normalized[i], 1.0) @ poses[i].rotation for i in surviving]
    )
    ray_directions /= np.linalg.norm(ray_directions, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(ray_directions @ ray_directions.T, -1, 1)))
    # Anti-parallel rays have the same ill-conditioned geometry as parallel rays.
    angles = np.minimum(angles, 180 - angles)
    max_angle = float(angles.max())
    # Use physical reprojection Jacobian, not loss-modified solver Jacobian.
    h = 1e-5
    jacobian = np.column_stack(
        [
            (
                residual(point + np.eye(3)[axis] * h, inliers)
                - residual(point - np.eye(3)[axis] * h, inliers)
            )
            / (2 * h)
            for axis in range(3)
        ]
    )
    information = jacobian.T @ jacobian
    condition = float(np.linalg.cond(information))
    degenerate = max_angle < config.minimum_ray_angle_degrees or condition > 1e10
    variance = max(1.0, float(np.mean(e[inliers] ** 2)))
    covariance = np.linalg.pinv(information) * variance
    observation_confidence = float(
        np.average(
            [obs[i].confidence * calibration_confidence[i] for i in surviving],
            weights=weights[inliers],
        )
    )
    geometry_confidence = min(1.0, max_angle / 10.0)
    confidence = (
        observation_confidence * geometry_confidence * float(np.exp(-np.median(e[inliers]) / 5.0))
    )
    if degenerate:
        confidence = min(confidence, 0.05)
    rejected.update(cams[i].id for i in np.flatnonzero(~inliers))
    rejected.difference_update(cams[i].id for i in surviving)
    warnings = ["Poor triangulation angle/conditioning; depth is unreliable"] if degenerate else []
    if fit is not None and not fit.success:
        warnings.append("Nonlinear reprojection optimizer reached iteration limit")
    diagnostics = TriangulationDiagnostics(
        contributing_camera_ids=[cams[i].id for i in surviving],
        rejected_camera_ids=sorted(rejected),
        reprojection_errors={c.id: float(error) for c, error in zip(cams, e, strict=True)},
        mean_reprojection_error=float(np.mean(e[inliers])),
        median_reprojection_error=float(np.median(e[inliers])),
        max_reprojection_error=float(np.max(e[inliers])),
        max_ray_angle_degrees=max_angle,
        condition_number=condition,
        degenerate=degenerate,
        warnings=warnings,
    )
    return Joint3DObservation(
        point,
        float(np.clip(confidence, 0, 1)),
        diagnostics,
        first.joint_id if first else "",
        first.person_id if first else "",
        timestamp,
        covariance,
    )


NDArrayBool = np.ndarray
