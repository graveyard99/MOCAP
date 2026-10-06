"""Metric target calibration and surveyed camera localization."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike

from openmocap.types import Camera, CameraIntrinsics, CameraPose, CameraQuality, FloatArray


@dataclass
class CalibrationResult:
    intrinsics: CameraIntrinsics
    poses: list[CameraPose]
    rms_pixels: float
    per_image_rms_pixels: list[float]
    target_spacing_metres: float


def checkerboard_points(columns: int, rows: int, square_metres: float) -> FloatArray:
    """Inner-corner target coordinates in OpenCV target-local XY plane."""
    if columns < 2 or rows < 2 or square_metres <= 0:
        raise ValueError("Checkerboard dimensions and measured square spacing must be positive")
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_metres
    return points


def detect_checkerboard(image: ArrayLike, columns: int, rows: int) -> FloatArray | None:
    image = np.asarray(image, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found, corners = cv2.findChessboardCornersSB(gray, (columns, rows))
    return corners.reshape(-1, 2).astype(np.float64) if found else None


def detect_charuco(
    image: ArrayLike,
    columns: int,
    rows: int,
    square_metres: float,
    marker_metres: float,
    dictionary_id: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return detected metric target points and corresponding plate pixels."""
    if not 0 < marker_metres < square_metres:
        raise ValueError("ChArUco marker size must be smaller than square spacing")
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("ChArUco requires opencv-contrib-python-headless")
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50 if dictionary_id is None else dictionary_id
    )
    board = cv2.aruco.CharucoBoard((columns, rows), square_metres, marker_metres, dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(np.asarray(image, dtype=np.uint8))
    if corners is None or ids is None or len(corners) < 4:
        return np.empty((0, 3)), np.empty((0, 2))
    return board.getChessboardCorners()[ids.reshape(-1)].astype(np.float64), corners.reshape(-1, 2)


def detect_coded_controls(
    image: ArrayLike, surveyed_corners: dict[int, ArrayLike], *, dictionary_id: int | None = None
) -> tuple[FloatArray, FloatArray, list[int]]:
    """Detect AprilTag/ArUco corners with supplied surveyed metric coordinates.

    Each marker maps to four world corners in detector TL,TR,BR,BL order. Unknown
    marker IDs are ignored; known targets require fully surveyed XYZ corners.
    """
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("Coded controls require opencv-contrib-python-headless")
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11 if dictionary_id is None else dictionary_id
    )
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, _ = detector.detectMarkers(np.asarray(image, dtype=np.uint8))
    world, pixels, used = [], [], []
    if ids is not None:
        for marker, marker_id in zip(corners, ids.reshape(-1), strict=True):
            marker_id = int(marker_id)
            if marker_id in surveyed_corners:
                coordinates = np.asarray(surveyed_corners[marker_id], dtype=np.float64)
                if coordinates.shape != (4, 3):
                    raise ValueError("Surveyed marker corners require four metric XYZ points")
                world.extend(coordinates)
                pixels.extend(marker.reshape(4, 2))
                used.append(marker_id)
    return np.asarray(world).reshape(-1, 3), np.asarray(pixels).reshape(-1, 2), used


def calibrate_intrinsics(
    object_points: list[ArrayLike],
    image_points: list[ArrayLike],
    image_size: tuple[int, int],
    *,
    target_spacing_metres: float,
    fisheye: bool = False,
) -> CalibrationResult:
    if len(object_points) != len(image_points) or len(object_points) < 3:
        raise ValueError("Intrinsic calibration requires three or more matching target views")
    if target_spacing_metres <= 0:
        raise ValueError("Target spacing must be measured in metres")
    objects = [np.asarray(p, dtype=np.float64).reshape(-1, 1, 3) for p in object_points]
    images = [np.asarray(p, dtype=np.float64).reshape(-1, 1, 2) for p in image_points]
    if any(len(a) != len(b) or len(a) < 4 for a, b in zip(objects, images, strict=True)):
        raise ValueError("Each target view must contain four or more matched points")
    if fisheye:
        rms, matrix, distortion, rotations, translations = cv2.fisheye.calibrate(
            objects,
            images,
            image_size,
            np.eye(3),
            np.zeros((4, 1)),
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND,
        )
    else:
        rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
            [p.astype(np.float32) for p in objects],
            [p.astype(np.float32) for p in images],
            image_size,
            None,
            None,
        )
    intrinsics = CameraIntrinsics(
        matrix[0, 0],
        matrix[1, 1],
        matrix[0, 2],
        matrix[1, 2],
        *image_size,
        distortion.reshape(-1),
        "fisheye" if fisheye else "opencv",
        source="calibration_target",
    )
    poses = [
        CameraPose(cv2.Rodrigues(r)[0], t.reshape(3))
        for r, t in zip(rotations, translations, strict=True)
    ]
    errors = [
        float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (
                            Camera("target", intrinsics, pose).project(points.reshape(-1, 3))
                            - pixels.reshape(-1, 2)
                        )
                        ** 2,
                        axis=1,
                    )
                )
            )
        )
        for pose, points, pixels in zip(poses, objects, images, strict=True)
    ]
    return CalibrationResult(intrinsics, poses, float(rms), errors, target_spacing_metres)


def surveyed_pnp(
    camera_id: str,
    world_points_metres: ArrayLike,
    image_points: ArrayLike,
    intrinsics: CameraIntrinsics,
    *,
    threshold_px: float = 3.0,
) -> tuple[Camera, dict]:
    """Localize a camera against surveyed world controls, with robust PnP."""
    from openmocap.cameras import undistort

    world = np.asarray(world_points_metres, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(world) != len(pixels) or len(world) < 6:
        raise ValueError("Surveyed PnP requires six or more matched metric world controls")
    if np.linalg.matrix_rank(world - world.mean(0)) < 2:
        raise ValueError("Surveyed PnP controls must not be collinear")
    normalized = undistort(pixels, intrinsics)
    success, rotation, translation, inliers = cv2.solvePnPRansac(
        world,
        normalized,
        np.eye(3),
        None,
        iterationsCount=200,
        reprojectionError=threshold_px / max(intrinsics.fx, intrinsics.fy),
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or inliers is None or len(inliers) < 6:
        raise ValueError("Surveyed camera localization failed robust consensus")
    index = inliers.reshape(-1)
    rotation, translation = cv2.solvePnPRefineLM(
        world[index], normalized[index], np.eye(3), None, rotation, translation
    )
    camera = Camera(
        camera_id,
        intrinsics,
        CameraPose(cv2.Rodrigues(rotation)[0], translation.reshape(3)),
        locked=True,
        source="surveyed",
    )
    errors = np.linalg.norm(camera.project(world) - pixels, axis=1)
    camera.quality = CameraQuality(1.0, float(np.sqrt(np.mean(errors[index] ** 2))), "surveyed")
    return camera, {
        "inliers": index.tolist(),
        "rejected": sorted(set(range(len(world))) - set(index)),
        "rms_pixels": camera.quality.reprojection_rmse,
        "errors_pixels": errors.tolist(),
    }
