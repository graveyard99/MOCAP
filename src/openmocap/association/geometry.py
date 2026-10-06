"""Geometry-led cross-camera identity association with explicit ambiguity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from openmocap.types import Camera, Pose2DObservation


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def epipolar_error(
    camera_a: Camera,
    camera_b: Camera,
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    time_a: float = 0,
    time_b: float = 0,
) -> np.ndarray:
    """Symmetric normalized-ray epipolar distance, converted to approximate pixels.

    Original distorted plate points are undistorted first. Moving-camera poses
    are evaluated at each observation's own world time; clocks remain untouched.
    """
    a = camera_a.pose_at(time_a)
    b = camera_b.pose_at(time_b)
    relative = b.rotation @ a.rotation.T
    translation = b.translation - relative @ a.translation
    essential = _skew(translation) @ relative
    if np.linalg.norm(essential) < 1e-12:
        return np.full(len(np.atleast_2d(xy_a)), np.inf)
    rays_a = np.column_stack(
        [camera_a.undistort(xy_a, time_a).reshape(-1, 2), np.ones(len(np.atleast_2d(xy_a)))]
    )
    rays_b = np.column_stack(
        [camera_b.undistort(xy_b, time_b).reshape(-1, 2), np.ones(len(np.atleast_2d(xy_b)))]
    )
    lines_b, lines_a = rays_a @ essential.T, rays_b @ essential
    numerator = np.abs(np.sum(rays_b * lines_b, axis=1))
    denominator_a = np.linalg.norm(lines_a[:, :2], axis=1)
    denominator_b = np.linalg.norm(lines_b[:, :2], axis=1)
    intrinsics_a, intrinsics_b = camera_a.intrinsics_at(time_a), camera_b.intrinsics_at(time_b)
    focal_a, focal_b = (
        np.sqrt(intrinsics_a.fx * intrinsics_a.fy),
        np.sqrt(intrinsics_b.fx * intrinsics_b.fy),
    )
    return (
        0.5
        * numerator
        * (focal_a / np.maximum(denominator_a, 1e-12) + focal_b / np.maximum(denominator_b, 1e-12))
    )


@dataclass
class IdentityMatch:
    anchor_id: str
    target_id: str
    confidence: float
    median_epipolar_px: float
    ambiguous: bool = False
    source: str = "epipolar_geometry"


def associate_people(
    camera_a: Camera,
    camera_b: Camera,
    observations_a: list[Pose2DObservation],
    observations_b: list[Pose2DObservation],
    *,
    max_error_px: float = 8,
    min_joints: int = 3,
    max_time_delta: float = 0.05,
    appearance_a: dict[str, np.ndarray] | None = None,
    appearance_b: dict[str, np.ndarray] | None = None,
    ambiguity_margin_px: float = 1.0,
) -> list[IdentityMatch]:
    """Assign people between one reference camera and another.

    Pose observations must be from a common nearby world-time window. Geometry
    hard-gates candidates; appearance can rank geometrically valid candidates
    but cannot rescue an epipolar-inconsistent identity. Ambiguous assignments
    are returned as such and are not safe to triangulate without another cue.
    """
    groups_a: dict[str, dict[str, Pose2DObservation]] = {}
    groups_b: dict[str, dict[str, Pose2DObservation]] = {}
    for observations, groups in [(observations_a, groups_a), (observations_b, groups_b)]:
        for obs in observations:
            if obs.enabled and obs.confidence >= 0.2:
                groups.setdefault(obs.person_id, {})[obs.joint_id] = obs
    ids_a, ids_b = sorted(groups_a), sorted(groups_b)
    if not ids_a or not ids_b:
        return []
    cost = np.full((len(ids_a), len(ids_b)), 1e6)
    errors = np.full_like(cost, np.inf)
    for i, identity_a in enumerate(ids_a):
        for j, identity_b in enumerate(ids_b):
            pairs = [
                (groups_a[identity_a][name], groups_b[identity_b][name])
                for name in sorted(groups_a[identity_a].keys() & groups_b[identity_b].keys())
            ]
            residuals = []
            for a, b in pairs:
                time_a = float(camera_a.time_mapping.to_world(a.camera_timestamp))
                time_b = float(camera_b.time_mapping.to_world(b.camera_timestamp))
                if abs(time_a - time_b) <= max_time_delta:
                    residuals.append(
                        float(epipolar_error(camera_a, camera_b, a.xy, b.xy, time_a, time_b)[0])
                    )
            if len(residuals) < min_joints:
                continue
            median = float(np.median(residuals))
            errors[i, j] = median
            if not np.isfinite(median) or median > max_error_px:
                continue
            appearance_penalty = 0.0
            if (
                appearance_a
                and appearance_b
                and identity_a in appearance_a
                and identity_b in appearance_b
            ):
                descriptor_a, descriptor_b = appearance_a[identity_a], appearance_b[identity_b]
                similarity = float(
                    np.dot(descriptor_a, descriptor_b)
                    / max(np.linalg.norm(descriptor_a) * np.linalg.norm(descriptor_b), 1e-12)
                )
                appearance_penalty = max_error_px * 0.15 * (1 - np.clip(similarity, 0, 1))
            cost[i, j] = median + appearance_penalty
    rows, columns = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(rows, columns, strict=True):
        if cost[i, j] >= 1e6:
            continue
        alternatives = np.r_[np.delete(cost[i], j), np.delete(cost[:, j], i)]
        ambiguous = bool(
            len(alternatives) and alternatives.min() - cost[i, j] < ambiguity_margin_px
        )
        confidence = float(np.exp(-errors[i, j] / max_error_px)) * (0.25 if ambiguous else 1.0)
        matches.append(
            IdentityMatch(ids_a[i], ids_b[j], confidence, float(errors[i, j]), ambiguous)
        )
    return matches


def associate_records(
    cameras: list[Camera],
    records: list[dict[str, Any]],
    *,
    actor_id: str = "actor01",
    max_error_px: float = 8,
    max_time_delta: float = 0.05,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pool timestamp-local matches to persistent camera-track identities.

    Cost is linear in camera count using a deterministic reference camera.
    Pairwise all-camera matching is avoided; cameras lacking reference overlap
    remain explicitly unassociated instead of inventing identities.
    """
    if len(cameras) < 2:
        raise ValueError("Association requires two or more calibrated cameras")
    by_camera: dict[str, list[Pose2DObservation]] = {}
    for record in records:
        observation = Pose2DObservation.from_dict(record)
        by_camera.setdefault(observation.camera_id, []).append(observation)
    reference = max(cameras, key=lambda c: len(by_camera.get(c.id, [])))
    reference_rows = by_camera.get(reference.id, [])
    reference_ids = sorted({o.person_id for o in reference_rows})
    global_ids = {
        identity: actor_id if i == 0 else f"{actor_id}_{i + 1:02d}"
        for i, identity in enumerate(reference_ids)
    }
    mappings = {(reference.id, identity): global_ids[identity] for identity in reference_ids}
    evidence: list[dict[str, Any]] = []
    reference_times = np.unique(
        [float(reference.time_mapping.to_world(o.camera_timestamp)) for o in reference_rows]
    )
    if not len(reference_times):
        raise ValueError("No reference-camera pose observations to associate")
    # Uniform bounded evidence sampling prevents long takes from quadratic scans.
    sample_times = reference_times[
        np.linspace(0, len(reference_times) - 1, min(50, len(reference_times))).astype(int)
    ]
    for camera in cameras:
        if camera.id == reference.id:
            continue
        other = by_camera.get(camera.id, [])
        votes: dict[tuple[str, str], list[float]] = {}
        for time in sample_times:
            a = _nearest_frame(reference, reference_rows, float(time), max_time_delta)
            b = _nearest_frame(camera, other, float(time), max_time_delta)
            for match in associate_people(
                reference, camera, a, b, max_error_px=max_error_px, max_time_delta=max_time_delta
            ):
                evidence.append({"camera_id": camera.id, "world_time": float(time), **vars(match)})
                if not match.ambiguous:
                    votes.setdefault((match.anchor_id, match.target_id), []).append(
                        match.confidence
                    )
        accepted = sorted(votes, key=lambda pair: (-sum(votes[pair]), pair))
        used_anchor, used_target = set(), set()
        for anchor, target in accepted:
            if anchor not in used_anchor and target not in used_target:
                mappings[(camera.id, target)] = global_ids[anchor]
                used_anchor.add(anchor)
                used_target.add(target)
    effective = []
    missing = 0
    for record in records:
        row = dict(record)
        local = str(row["person_id"])
        identity = mappings.get((str(row["camera_id"]), local))
        row["local_person_id"] = local
        if identity is None:
            missing += 1
            row["association_ambiguous"] = True
            row["disabled"] = True
        else:
            row["person_id"] = identity
            row["association_source"] = "multiview_epipolar"
        effective.append(row)
    return effective, {
        "reference_camera": reference.id,
        "identity_mapping": [
            {"camera_id": cid, "local_person_id": lid, "person_id": gid}
            for (cid, lid), gid in mappings.items()
        ],
        "unassociated_observations": missing,
        "evidence": evidence,
        "limitations": "reference overlap required; crossings may remain ambiguous",
    }


def _nearest_frame(
    camera: Camera, rows: list[Pose2DObservation], time: float, tolerance: float
) -> list[Pose2DObservation]:
    if not rows:
        return []
    times = np.array([float(camera.time_mapping.to_world(o.camera_timestamp)) for o in rows])
    nearest = times[np.argmin(np.abs(times - time))]
    return (
        [row for row, t in zip(rows, times, strict=True) if t == nearest]
        if abs(nearest - time) <= tolerance
        else []
    )
