"""Explicit external calibration imports. Scale/convention never inferred silently."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from openmocap.types import Camera, CameraIntrinsics, CameraPose, CameraQuality


def load_cameras(path: str | Path) -> list[Camera]:
    """Read canonical calibrated camera JSON (world metres, +Y up)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        if data.get("units", "metres") != "metres" or data.get("world_up", "+Y") != "+Y":
            raise ValueError("Camera import requires explicit conversion to metres and +Y up")
        data = data.get("cameras", [])
    cameras = [Camera.from_dict(item) for item in data]
    if not cameras or len({c.id for c in cameras}) != len(cameras):
        raise ValueError("Camera file must contain uniquely named cameras")
    if any(not camera.quality.metric_scale_known for camera in cameras):
        raise ValueError("Metric reconstruction requires calibrated metric camera scale")
    return cameras


def save_cameras(cameras: list[Camera], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "units": "metres",
                "world_up": "+Y",
                "cameras": [camera.to_dict() for camera in cameras],
            },
            indent=2,
        )
    )


def import_opencv(
    path: str | Path, camera_id: str, *, world_convention: str, metric_scale: float = 1.0
) -> Camera:
    """Read OpenCV YAML/XML K,D,R,t. R,t MUST map world to OpenCV camera."""
    if world_convention != "Y_UP_METRES" or metric_scale <= 0:
        raise ValueError("Declare Y_UP_METRES and positive scale for OpenCV transform import")
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"Cannot open OpenCV calibration: {path}")

    def matrix(*names: str) -> np.ndarray:
        for name in names:
            node = storage.getNode(name)
            if not node.empty():
                result = node.mat()
                if result is not None:
                    return result
        raise ValueError(f"OpenCV calibration is missing {names}")

    try:
        k = matrix("K", "camera_matrix")
        distortion = matrix("D", "distortion_coefficients").reshape(-1)
        r = matrix("R", "rotation_matrix")
        t = matrix("t", "translation_vector").reshape(3) * metric_scale
        width = int(storage.getNode("image_width").real())
        height = int(storage.getNode("image_height").real())
        model = storage.getNode("distortion_model").string() or "opencv"
        intrinsics = CameraIntrinsics(
            k[0, 0],
            k[1, 1],
            k[0, 2],
            k[1, 2],
            width,
            height,
            distortion,
            model,
            source="calibration_target",
        )
        return Camera(
            camera_id, intrinsics, CameraPose(r, t), locked=True, source="calibration_target"
        )
    finally:
        storage.release()


def _colmap_intrinsics(
    model: str, width: int, height: int, params: list[float]
) -> CameraIntrinsics:
    distortion: list[float] = []
    distortion_model = "opencv"
    if model == "SIMPLE_PINHOLE":
        fx, cx, cy = params
        fy = fx
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
    elif model in ("SIMPLE_RADIAL", "RADIAL"):
        fx, cx, cy = params[:3]
        fy = fx
        distortion = [params[3], params[4] if model == "RADIAL" else 0.0, 0.0, 0.0, 0.0]
    elif model in ("OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
        fx, fy, cx, cy = params[:4]
        distortion = params[4:]
        if model == "OPENCV_FISHEYE":
            distortion_model = "fisheye"
    else:
        raise ValueError(
            f"Unsupported COLMAP camera model {model}; convert externally before import"
        )
    return CameraIntrinsics(
        fx, fy, cx, cy, width, height, distortion, distortion_model, source="colmap"
    )


def import_colmap_text(
    directory: str | Path,
    *,
    metric_scale: float,
    world_rotation: np.ndarray | None = None,
    world_translation: np.ndarray | None = None,
) -> list[Camera]:
    """Import COLMAP text reconstruction; each registered image is a pose.

    The scale must be externally established. By default COLMAP world axes are
    retained and MUST already be +Y up. For arbitrary SfM axes, explicitly supply
    world_rotation/world_translation to align SfM positions into surveyed world.
    Names use registered image names without suffix, suitable for shot plates.
    """
    directory = Path(directory)
    if metric_scale <= 0 or not np.isfinite(metric_scale):
        raise ValueError("COLMAP requires a positive surveyed/known-distance metric scale")
    alignment_r = (
        np.eye(3) if world_rotation is None else np.asarray(world_rotation, dtype=np.float64)
    )
    alignment_t = (
        np.zeros(3)
        if world_translation is None
        else np.asarray(world_translation, dtype=np.float64)
    )
    # Validates alignment rotation; catches axis reflections.
    CameraPose(alignment_r, alignment_t)
    intrinsics = {}
    for line in (directory / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        camera_id, model, width, height = parts[:4]
        intrinsics[camera_id] = _colmap_intrinsics(
            model, int(width), int(height), list(map(float, parts[4:]))
        )
    cameras = []
    expecting_pose = True
    for line in (directory / "images.txt").read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        if not expecting_pose:
            expecting_pose = True
            continue
        if not line.strip():
            continue
        parts = line.split(maxsplit=9)
        if len(parts) != 10:
            raise ValueError("Malformed COLMAP registered image pose")
        _, qw, qx, qy, qz, tx, ty, tz, camera_id, name = parts
        r = Rotation.from_quat(list(map(float, [qx, qy, qz, qw]))).as_matrix()
        t = np.array(list(map(float, [tx, ty, tz])))
        centre = alignment_r @ (-r.T @ t) * metric_scale + alignment_t
        rotation = r @ alignment_r.T
        cameras.append(
            Camera(
                str(Path(name).with_suffix("")),
                intrinsics[camera_id],
                CameraPose(rotation, -rotation @ centre),
                locked=True,
                source="colmap",
                source_path=name,
                quality=CameraQuality(1.0, source="colmap"),
            )
        )
        expecting_pose = False
    if not cameras:
        raise ValueError("COLMAP reconstruction contains no registered images")
    return cameras


def import_metashape_xml(
    path: str | Path,
    *,
    metric_scale: float,
    world_rotation: np.ndarray | None = None,
    world_translation: np.ndarray | None = None,
) -> list[Camera]:
    """Read frame-camera sensor calibrations and local camera-to-world transforms.

    Chunk geocentric transforms are deliberately rejected. Export/alignment must
    establish a local metric +Y-up world; no hidden GIS/axis assumptions are made.
    """
    if metric_scale <= 0:
        raise ValueError("Metashape import requires established positive metric scale")
    tree = ET.parse(path)
    chunk = tree.getroot().find(".//chunk")
    if chunk is None:
        raise ValueError("Metashape XML lacks a chunk")
    if chunk.find("transform") is not None:
        raise ValueError(
            "Metashape chunk transform requires explicit local-world alignment before import"
        )
    alignment_r = (
        np.eye(3) if world_rotation is None else np.asarray(world_rotation, dtype=np.float64)
    )
    alignment_t = (
        np.zeros(3)
        if world_translation is None
        else np.asarray(world_translation, dtype=np.float64)
    )
    CameraPose(alignment_r, alignment_t)
    sensors = {}
    for sensor in chunk.findall("./sensors/sensor"):
        calibration = sensor.find("calibration")
        resolution = sensor.find("resolution")
        if calibration is None or resolution is None or sensor.get("type", "frame") != "frame":
            continue
        width, height = int(resolution.get("width")), int(resolution.get("height"))

        def value(name: str, default: float = 0.0) -> float:
            return float(calibration.findtext(name, str(default)))

        if value("b2") != 0:
            raise ValueError("Metashape sensor skew b2 is unsupported; rectify before import")
        sensors[sensor.get("id")] = CameraIntrinsics(
            value("f") + value("b1"),
            value("f"),
            width / 2 + value("cx"),
            height / 2 + value("cy"),
            width,
            height,
            [value("k1"), value("k2"), value("p2"), value("p1"), value("k3")],
            source="metashape",
        )
    cameras = []
    for node in chunk.findall("./cameras/camera"):
        text = node.findtext("transform")
        if text is None:
            continue
        transform = np.fromstring(text, sep=" ").reshape(4, 4)
        c2w = alignment_r @ transform[:3, :3]
        centre = alignment_r @ transform[:3, 3] * metric_scale + alignment_t
        camera_id = node.get("label") or node.get("id")
        cameras.append(
            Camera(
                camera_id,
                sensors[node.get("sensor_id")],
                CameraPose(c2w.T, -c2w.T @ centre),
                source="metashape",
                locked=True,
            )
        )
    if not cameras:
        raise ValueError("Metashape export contains no supported calibrated cameras")
    return cameras
