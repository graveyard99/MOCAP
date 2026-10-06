"""Distortion-aware camera projection. No learned camera estimation is used."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import ArrayLike

from openmocap.types import Camera, CameraIntrinsics, CameraPose, FloatArray


def project(points: ArrayLike, intrinsics: CameraIntrinsics, pose: CameraPose) -> FloatArray:
    """Project world points through the original lens model to plate pixels.

    Behind-camera points are returned as NaN. No clipping to the image boundary
    occurs, allowing offscreen QC predictions and geometry computations.
    """
    points = np.asarray(points, dtype=np.float64)
    scalar = points.ndim == 1
    points = points.reshape(-1, 3)
    camera_points = points @ pose.rotation.T + pose.translation
    rotation_vector = cv2.Rodrigues(pose.rotation)[0]
    if intrinsics.distortion_model == "fisheye":
        pixels = cv2.fisheye.projectPoints(
            points.reshape(-1, 1, 3),
            rotation_vector,
            pose.translation,
            intrinsics.matrix,
            intrinsics.distortion.reshape(4, 1),
        )[0]
    else:
        distortion = intrinsics.distortion if intrinsics.distortion_model == "opencv" else None
        pixels = cv2.projectPoints(
            points, rotation_vector, pose.translation, intrinsics.matrix, distortion
        )[0]
    result = pixels.reshape(-1, 2)
    result[camera_points[:, 2] <= 1e-9] = np.nan
    return result[0] if scalar else result


def undistort(pixels: ArrayLike, intrinsics: CameraIntrinsics) -> FloatArray:
    """Return normalized (+X right,+Y down,+Z forward) ray coordinates."""
    pixels = np.asarray(pixels, dtype=np.float64)
    scalar = pixels.ndim == 1
    shaped = pixels.reshape(-1, 1, 2)
    if intrinsics.distortion_model == "fisheye":
        result = cv2.fisheye.undistortPoints(
            shaped, intrinsics.matrix, intrinsics.distortion.reshape(4, 1)
        )
    else:
        distortion = intrinsics.distortion if intrinsics.distortion_model == "opencv" else None
        result = cv2.undistortPointsIter(
            shaped,
            intrinsics.matrix,
            distortion,
            None,
            None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12),
        )
    result = result.reshape(-1, 2)
    return result[0] if scalar else result


def distort(normalized: ArrayLike, intrinsics: CameraIntrinsics) -> FloatArray:
    """Map normalized coordinates through lens distortion into plate pixels."""
    xy = np.asarray(normalized, dtype=np.float64)
    scalar = xy.ndim == 1
    xy = xy.reshape(-1, 2)
    result = project(
        np.column_stack([xy, np.ones(len(xy))]), intrinsics, CameraPose(np.eye(3), np.zeros(3))
    )
    return result[0] if scalar else result


def look_at(
    camera_id: str,
    centre: ArrayLike,
    target: ArrayLike,
    intrinsics: CameraIntrinsics,
    world_up: ArrayLike = (0, 1, 0),
    **kwargs: object,
) -> Camera:
    """Construct an OpenCV world-to-camera transform in metric Y-up space."""
    centre = np.asarray(centre, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - centre
    if np.linalg.norm(forward) < 1e-10:
        raise ValueError("Camera centre and target cannot coincide")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(world_up, dtype=np.float64))
    if np.linalg.norm(right) < 1e-10:
        raise ValueError("Camera forward and world up cannot be parallel")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return Camera(camera_id, intrinsics, CameraPose(rotation, -rotation @ centre), **kwargs)


def ray(camera: Camera, pixels: ArrayLike, time: float = 0) -> tuple[FloatArray, FloatArray]:
    """Return camera centre and unit ray(s) in the metric world convention."""
    xy = camera.undistort(pixels, time)
    xy = np.asarray(xy).reshape(-1, 2)
    pose = camera.pose_at(time)
    directions = np.column_stack([xy, np.ones(len(xy))]) @ pose.rotation
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return pose.centre, directions
