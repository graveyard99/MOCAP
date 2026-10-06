"""Robust N-view initialization followed by native-time continuous ray fitting.

No frame-number synchronization is used. Initial plate interpolation is bounded
by a maximum temporal gap. The continuous fit consumes each original camera PTS
at its mapped world time; lens distortion is removed only for the ray equations.
QC projects back through the original lens model.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import lstsq

from openmocap.types import Camera, Pose2DObservation
from openmocap.triangulation import triangulate, TriangulationConfig
from openmocap.trajectories import ContinuousTrajectory, fit_trajectory


@dataclass
class Reconstruction:
    times: np.ndarray
    joints: np.ndarray
    confidence: np.ndarray
    trajectory: ContinuousTrajectory
    diagnostics: list[dict[str, Any]]


def reconstruct(
    cameras: list[Camera],
    records: list[dict[str, Any]],
    joint_names: list[str],
    *,
    fps: float = 30,
    actor_id: str = "actor01",
    max_gap: float = 0.12,
    callback: Callable | None = None,
    cancel_event: Any = None,
    continuous: bool = True,
    solve_range: list[float] | None = None,
    triangulation_options: dict | None = None,
    smoothing: float = 1e-5,
) -> Reconstruction:
    cameras_by_id = {c.id: c for c in cameras}
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for row in records:
        if row["camera_id"] not in cameras_by_id:
            raise ValueError(f"Observation references unknown camera {row['camera_id']}")
        if str(row.get("person_id", actor_id)) != actor_id or (
            row.get("disabled", False) or not row.get("enabled", True)
        ):
            continue
        camera = cameras_by_id[row["camera_id"]]
        time = float(camera.time_mapping.to_world(float(row["camera_timestamp"])))
        xy = row.get("xy", [row.get("x"), row.get("y")])
        confidence = float(row.get("confidence", 1))
        if (
            not np.isfinite([float(row["camera_timestamp"]), time, confidence]).all()
            or not 0 <= confidence <= 1
        ):
            raise ValueError(f"Malformed timestamp or confidence for {camera.id}/{row['joint_id']}")
        if np.asarray(xy).shape != (2,):
            raise ValueError(f"Expected two pixel coordinates for {camera.id}/{row['joint_id']}")
        if np.isfinite(xy).all() and confidence > 0.05:
            groups[(camera.id, str(row["joint_id"]))].append((time, xy, confidence))
    if not groups:
        raise ValueError("No valid timestamped 2D observations for selected actor")
    arrays = {}
    for group, data in groups.items():
        # Duplicate measurements are retained raw; effective view uses strongest.
        best = {}
        for t, xy, c in data:
            if t not in best or c > best[t][1]:
                best[t] = (xy, c)
        ts = np.array(sorted(best))
        arrays[group] = (ts, np.array([best[t][0] for t in ts]), np.array([best[t][1] for t in ts]))
    # Only two independent views are required; short cameras cannot trim all others.
    starts, ends = [], []
    for camera in cameras:
        available = [a[0] for (cid, _), a in arrays.items() if cid == camera.id]
        if available:
            starts.append(min(t[0] for t in available))
            ends.append(max(t[-1] for t in available))
    if len(starts) < 2:
        raise ValueError("Metric reconstruction requires observations from at least two cameras")
    events = sorted({float(t) for t in starts + ends})
    supported_intervals = [
        (a, b)
        for a, b in zip(events[:-1], events[1:], strict=True)
        if sum(s <= (a + b) / 2 <= e for s, e in zip(starts, ends, strict=True)) >= 2
    ]
    if not supported_intervals:
        raise ValueError("Camera observation ranges do not overlap in world time")
    start, end = supported_intervals[0][0], supported_intervals[-1][1]
    if solve_range:
        if len(solve_range) != 2 or solve_range[1] <= solve_range[0]:
            raise ValueError("Solve range requires start < end in world seconds")
        start, end = max(start, solve_range[0]), min(end, solve_range[1])
    triang_config = TriangulationConfig(**(triangulation_options or {}))
    if end <= start:
        raise ValueError("Camera observation ranges do not overlap in world time")
    times = start + np.arange(int(np.floor((end - start) * fps)) + 1) / fps
    if len(times) < 4:
        raise ValueError("At least four output samples are required for a continuous motion solve")
    positions = np.full((len(times), len(joint_names), 3), np.nan)
    confidence = np.zeros(positions.shape[:2])
    diagnostics = []
    for frame, time in enumerate(times):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Reconstruction cancelled; completed checkpoints retained")
        for joint, name in enumerate(joint_names):
            observations = []
            for camera in cameras:
                group = arrays.get((camera.id, name))
                if group is None:
                    continue
                ts, pixels, weights = group
                if time < ts[0] - 1e-9 or time > ts[-1] + 1e-9:
                    continue
                upper = int(np.clip(np.searchsorted(ts, time), 1, len(ts) - 1))
                if len(ts) < 2 or ts[upper] - ts[upper - 1] > max_gap:
                    continue
                xy = [np.interp(time, ts, pixels[:, k]) for k in range(2)]
                c = float(np.interp(time, ts, weights))
                observations.append(
                    Pose2DObservation(
                        camera.id,
                        float(camera.time_mapping.to_camera(time)),
                        actor_id,
                        name,
                        xy,
                        c,
                        float(time),
                    )
                )
            if len(observations) < 2:
                continue
            solved = triangulate(
                cameras_by_id, observations, time=float(time), config=triang_config
            )
            positions[frame, joint] = solved.position
            confidence[frame, joint] = solved.confidence
            detail = (
                solved.diagnostics.to_dict()
                if hasattr(solved.diagnostics, "to_dict")
                else vars(solved.diagnostics)
            )
            diagnostics.append(
                {"time": float(time), "joint_id": name, "confidence": solved.confidence, **detail}
            )
        if callback and frame % max(1, len(times) // 20) == 0:
            callback(
                {
                    "stage": "triangulate",
                    "progress": (frame + 1) / len(times),
                    "message": f"N-view reconstruction {frame + 1}/{len(times)}",
                }
            )
    trajectory = fit_trajectory(
        times, positions, confidence, joint_names=tuple(joint_names), smoothing=smoothing
    )
    if continuous:
        trajectory = _native_time_fit(cameras_by_id, arrays, times, trajectory, joint_names)
        positions = trajectory.evaluate(times)
    return Reconstruction(times, positions, confidence, trajectory, diagnostics)


def _native_time_fit(
    cameras: dict[str, Camera],
    groups: dict,
    times: np.ndarray,
    initial: ContinuousTrajectory,
    names: list[str],
) -> ContinuousTrajectory:
    """Confidence-weighted cubic B-spline ray bundle with robust pixel gating.

    Linear world-space ray equations keep measured camera geometry fixed.
    A weak second-difference term regularizes curvature. Iteratively reweighted
    original distorted reprojections reject gross detection/identity outliers.
    """
    interval_count = max(3, min(30, int(np.ceil((times[-1] - times[0]) / 0.08))))
    interior = np.linspace(times[0], times[-1], interval_count + 1)[1:-1]
    knots = np.r_[[times[0]] * 4, interior, [times[-1]] * 4]
    count = len(knots) - 4
    splines = list(initial.splines)
    accepted_counts, native_warnings, rejected_keys = [], [], []
    for j, name in enumerate(names):
        rows, rhs, sources, weights = [], [], [], []
        for (cid, jid), (ts, xy, conf) in groups.items():
            if jid != name:
                continue
            camera = cameras[cid]
            for t, pixel, c in zip(ts, xy, conf, strict=True):
                if not times[0] <= t <= times[-1]:
                    continue
                seed = initial.evaluate(float(t))[j]
                if not np.isfinite(seed).all():
                    continue
                error = np.linalg.norm(camera.project(seed[None], time=float(t))[0] - pixel)
                if not np.isfinite(error) or error > 12:
                    rejected_keys.append(f"{camera.id}|{name}|{float(t):.12f}")
                    continue
                ray = camera.undistort(np.asarray(pixel)[None], time=float(t))[0]
                pose = camera.pose_at(float(t))
                basis = BSpline.design_matrix([t], knots, 3).toarray()[0]
                plane = np.array(
                    [
                        pose.rotation[0] - ray[0] * pose.rotation[2],
                        pose.rotation[1] - ray[1] * pose.rotation[2],
                    ]
                )
                rows.append(np.einsum("ai,b->abi", plane, basis).reshape(2, count * 3))
                rhs.append(
                    [
                        ray[0] * pose.translation[2] - pose.translation[0],
                        ray[1] * pose.translation[2] - pose.translation[1],
                    ]
                )
                sources.append((camera, float(t), np.asarray(pixel), basis))
                weights.append(c * camera.quality.confidence)
        accepted_counts.append(len(rows))
        if len(rows) < count * 2 or len({source[0].id for source in sources}) < 2:
            native_warnings.append(
                {
                    "joint": name,
                    "reason": "insufficient independent native views; preserved robust seed",
                }
            )
            continue
        matrix, target = np.vstack(rows), np.concatenate(rhs)
        regularization = np.kron(np.diff(np.eye(count), n=2, axis=0), np.eye(3)) * 0.002
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular[-1] <= singular[0] * 1e-9:
            native_warnings.append(
                {
                    "joint": name,
                    "reason": "ill-conditioned native ray bundle; preserved robust seed",
                }
            )
            continue
        base = np.sqrt(np.repeat(weights, 2))
        robust = np.ones(len(rows))
        for _ in range(4):
            weight = base * np.repeat(np.sqrt(robust), 2)
            solution = lstsq(
                np.vstack([matrix * weight[:, None], regularization]),
                np.r_[target * weight, np.zeros(len(regularization))],
                lapack_driver="gelsy",
            )[0].reshape(count, 3)
            errors = np.array(
                [
                    np.linalg.norm(cam.project((basis @ solution)[None], time=t)[0] - pixel)
                    for cam, t, pixel, basis in sources
                ]
            )
            errors = np.nan_to_num(errors, nan=1e6, posinf=1e6, neginf=1e6)
            robust = np.minimum(1, 3 / np.maximum(errors, 1e-6))
        candidate = BSpline(knots, solution, 3, extrapolate=False)
        original = initial.evaluate(times)[:, j]
        strong = initial.confidence(times)[:, j] >= 0.85
        displacement = np.linalg.norm(candidate(times) - original, axis=1)
        if not np.isfinite(solution).all() or (
            strong.any() and np.max(displacement[strong]) > 0.003
        ):
            native_warnings.append(
                {
                    "joint": name,
                    "reason": "native fit exceeded 3mm strong-evidence guard; preserved robust seed",
                }
            )
            continue
        splines[j] = candidate
    return ContinuousTrajectory(
        splines,
        initial.measured_times,
        initial.measured_confidence,
        tuple(names),
        {
            **initial.diagnostics,
            "native_time_ray_fit": True,
            "native_observation_counts": accepted_counts,
            "native_rejected_measurements": rejected_keys,
            "native_fit_warnings": native_warnings,
            "camera_parameters_modified": False,
        },
    )


def sample_plate_evidence(
    cameras: list[Camera],
    records: list[dict],
    names: list[str],
    times: np.ndarray,
    actor_id: str,
    max_gap: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-destructive bounded interpolation for sampled body 2D residuals.

    The continuous geometry objective uses native times. The sampled kinematic
    body objective uses derived plates at output times and retains original
    plates unchanged for independent native-time QC.
    """
    pixels = np.full((len(times), len(cameras), len(names), 2), np.nan)
    confidence = np.zeros(pixels.shape[:3])
    grouped = defaultdict(list)
    for row in records:
        if str(row.get("person_id", actor_id)) == actor_id and not (
            row.get("disabled", False) or not row.get("enabled", True)
        ):
            grouped[(row["camera_id"], str(row["joint_id"]))].append(row)
    for ci, camera in enumerate(cameras):
        for ji, name in enumerate(names):
            rows = grouped.get((camera.id, name), [])
            if len(rows) < 2:
                continue
            unique = {}
            for row in rows:
                time = float(camera.time_mapping.to_world(row["camera_timestamp"]))
                if time not in unique or row.get("confidence", 1) > unique[time].get(
                    "confidence", 1
                ):
                    unique[time] = row
            ts = np.array(sorted(unique))
            if len(ts) < 2:
                continue
            xy = np.array(
                [unique[t].get("xy", [unique[t].get("x"), unique[t].get("y")]) for t in ts]
            )
            quality = np.array([unique[t].get("confidence", 1) for t in ts])
            upper = np.clip(np.searchsorted(ts, times), 1, len(ts) - 1)
            valid = (times >= ts[0]) & (times <= ts[-1]) & (ts[upper] - ts[upper - 1] <= max_gap)
            pixels[valid, ci, ji] = np.column_stack(
                [np.interp(times[valid], ts, xy[:, k]) for k in range(2)]
            )
            confidence[valid, ci, ji] = np.interp(times[valid], ts, quality)
    return pixels, confidence
