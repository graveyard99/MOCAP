"""Deterministic virtual capture with independent camera clocks and known truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from openmocap.cameras import look_at
from openmocap.types import CameraIntrinsics, CameraTrajectory, TimeMapping


JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
REST_OFFSETS = np.array(
    [
        [0, 0, 0],
        [0.095, -0.09, 0],
        [-0.095, -0.09, 0],
        [0, 0.11, 0],
        [0, -0.43, 0],
        [0, -0.43, 0],
        [0, 0.12, 0],
        [0, -0.43, 0],
        [0, -0.43, 0],
        [0, 0.12, 0],
        [0, -0.035, 0.14],
        [0, -0.035, 0.14],
        [0, 0.13, 0],
        [0.08, 0.04, 0],
        [-0.08, 0.04, 0],
        [0, 0.16, 0],
        [0.105, 0, 0],
        [-0.105, 0, 0],
        [0.27, 0, 0],
        [-0.27, 0, 0],
        [0.25, 0, 0],
        [-0.25, 0, 0],
        [0.09, 0, 0],
        [-0.09, 0, 0],
    ],
    dtype=float,
)


def human_motion(
    times: np.ndarray, duration: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Known metric 24-joint FK motion with constant actor proportions.

    Root travel is exactly 2.43 metres over ``duration``. This is a kinematic
    geometry fixture, not a learned performer or a licensed SMPL asset.
    """
    times = np.asarray(times, dtype=float)
    phase = 2 * np.pi * times / max(duration, 1e-6)
    root = np.column_stack(
        [2.43 * times / max(duration, 1e-6), 1.0 + 0.02 * np.sin(phase), 0.10 * np.sin(phase / 2)]
    )
    pose = np.zeros((len(times), 24, 3))
    pose[:, 0, 1] = 0.15 * np.sin(phase / 2)
    pose[:, 1, 0] = 0.10 * np.sin(phase)
    pose[:, 2, 0] = -0.10 * np.sin(phase)
    pose[:, 4, 0] = 0.12 * (1 + np.sin(phase))
    pose[:, 5, 0] = 0.12 * (1 - np.sin(phase))
    pose[:, 16, 2] = -0.6 + 0.3 * np.sin(phase)
    pose[:, 17, 2] = 0.6 + 0.3 * np.sin(phase + 0.5)
    pose[:, 18, 1] = 0.4 * (1 + np.sin(phase * 0.7))
    pose[:, 19, 1] = -0.4 * (1 + np.sin(phase * 0.7))
    local = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(len(times), 24, 3, 3)
    world_rotations = np.empty_like(local)
    joints = np.empty((len(times), 24, 3))
    for joint, parent in enumerate(PARENTS):
        if parent < 0:
            world_rotations[:, joint] = local[:, joint]
            joints[:, joint] = root
        else:
            world_rotations[:, joint] = world_rotations[:, parent] @ local[:, joint]
            joints[:, joint] = joints[:, parent] + np.einsum(
                "nij,j->ni", world_rotations[:, parent], REST_OFFSETS[joint]
            )
    return joints, pose, root


def _render_frame(
    path: Path,
    xy: np.ndarray,
    camera_id: str,
    camera_time: float,
    world_time: float,
    width: int,
    height: int,
) -> None:
    import cv2

    image = np.full((height, width, 3), (30, 27, 23), dtype=np.uint8)
    for x in range(0, width, 40):
        cv2.line(image, (x, 0), (x, height - 1), (38, 35, 31), 1)
    for y in range(0, height, 40):
        cv2.line(image, (0, y), (width - 1, y), (38, 35, 31), 1)
    for joint, parent in enumerate(PARENTS):
        if parent >= 0 and np.isfinite(xy[[joint, parent]]).all():
            cv2.line(
                image,
                tuple(np.round(xy[parent]).astype(int)),
                tuple(np.round(xy[joint]).astype(int)),
                (155, 170, 190),
                4,
                cv2.LINE_AA,
            )
    for position in xy:
        if np.isfinite(position).all():
            cv2.circle(
                image, tuple(np.round(position).astype(int)), 4, (195, 208, 221), -1, cv2.LINE_AA
            )
    cv2.putText(
        image,
        f"SYNTHETIC {camera_id}  camera {camera_time:.4f}s  world {world_time:.4f}s",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (190, 196, 202),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write generated frame {path}")


def generate_capture(
    path: str | Path,
    camera_count: int = 8,
    frames: int = 60,
    *,
    fps: float = 30.0,
    noise_px: float = 0.5,
    missing_probability: float = 0.03,
    outlier_probability: float = 0.005,
    outlier_camera_ids: list[str] | None = None,
    asynchronous: bool = True,
    moving: bool = False,
    dropped_frame_probability: float = 0.0,
    duplicate_frame_probability: float = 0.0,
    render_images: bool = True,
    seed: int = 42,
) -> Path:
    """Write project.yaml, raw timestamped observations and ground-truth NPZ.

    Image fixtures are generated locally, carry no private footage or model
    assets, and can be redistributed with this project's license. Camera clocks
    include known sub-frame offsets and drift; camera observation times are
    genuinely asynchronous. Missing/outlier measurements remain explicit.
    """
    if camera_count < 2 or frames < 3 or fps <= 0 or noise_px < 0:
        raise ValueError(
            "Synthetic capture requires >=2 cameras, >=3 frames, positive FPS and nonnegative noise"
        )
    probabilities = (
        missing_probability,
        outlier_probability,
        dropped_frame_probability,
        duplicate_frame_probability,
    )
    if any(not 0 <= x <= 1 for x in probabilities):
        raise ValueError("Synthetic probabilities must lie in [0,1]")
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    times = np.arange(frames) / fps
    duration = float(times[-1])
    truth, pose, root = human_motion(times, duration)
    intrinsics = CameraIntrinsics(
        650, 650, 480, 270, 960, 540, [0.01, -0.001, 0.0001, -0.0001, 0], source="synthetic"
    )
    observations: list[dict[str, Any]] = []
    cameras: list[dict[str, Any]] = []
    frame_manifest: list[dict[str, Any]] = []
    excluded = 0
    outliers = 0
    for index in range(camera_count):
        camera_id = f"CAM_{index + 1:02d}"
        angle = index * 2 * np.pi / camera_count
        centre = [1.215 + 4.5 * np.cos(angle), 1.6 + 0.35 * np.sin(angle * 2), 4.5 * np.sin(angle)]
        camera = look_at(camera_id, centre, [1.215, 0.95, 0], intrinsics)
        camera.source = "synthetic"
        camera.quality.source = "synthetic"
        camera.time_mapping = TimeMapping(
            scale=1 + (index - camera_count / 2) * 25e-6 if asynchronous else 1,
            offset=0.013 * index if asynchronous else 0,
            locked=True,
            source="synthetic",
        )
        if moving:
            track_times = np.linspace(0, duration + 1 / fps, 7)
            tracked = [
                look_at(
                    camera_id,
                    np.asarray(centre)
                    + [0.20 * np.sin(t * 1.5), 0.05 * np.sin(t), 0.15 * np.cos(t)],
                    [1.215, 0.95, 0],
                    intrinsics,
                ).pose
                for t in track_times
            ]
            camera.trajectory = CameraTrajectory(track_times, tracked)
        media = output / "media" / camera_id
        if render_images:
            media.mkdir(parents=True, exist_ok=True)
            camera.source_path = str(media.resolve())
        camera_data = camera.to_dict()
        camera_data["fps"] = fps
        phase = index / camera_count / fps if asynchronous else 0
        world_times = times + phase
        world_times = world_times[world_times <= duration + 1e-10]
        projected_truth, _, _ = human_motion(world_times, duration)
        camera_frames = []
        for frame, world_time in enumerate(world_times):
            if rng.random() < dropped_frame_probability:
                continue
            camera_time = float(camera.time_mapping.to_camera(world_time))
            pixels = camera.project(projected_truth[frame], time=float(world_time))
            image_path = media / f"{frame:06d}.png"
            if render_images:
                _render_frame(
                    image_path,
                    pixels,
                    camera_id,
                    camera_time,
                    float(world_time),
                    intrinsics.width,
                    intrinsics.height,
                )
            source_frame = {
                "camera_id": camera_id,
                "camera_timestamp": camera_time,
                "world_timestamp": float(world_time),
                "frame_index": frame,
                "path": str(image_path.relative_to(output)) if render_images else None,
            }
            camera_frames.append(source_frame)
            frame_manifest.append(source_frame)
            repeats = 2 if rng.random() < duplicate_frame_probability else 1
            for _repeat in range(repeats):
                for joint, pixel in enumerate(pixels):
                    if not np.isfinite(pixel).all() or rng.random() < missing_probability:
                        excluded += 1
                        continue
                    measured = pixel + rng.normal(0, noise_px, 2)
                    bad = rng.random() < outlier_probability or camera_id in (
                        outlier_camera_ids or []
                    )
                    if bad:
                        measured += rng.normal(0, 50, 2)
                        outliers += 1
                    observations.append(
                        {
                            "camera_id": camera_id,
                            "camera_timestamp": camera_time,
                            "world_timestamp": float(world_time),
                            "person_id": "actor01",
                            "joint_id": JOINT_NAMES[joint],
                            "xy": measured.tolist(),
                            "confidence": 0.92,
                            "source": "synthetic",
                            "frame_index": frame,
                            "synthetic_outlier": bad,
                        }
                    )
        if render_images:
            # Native source clocks survive ingest; never derive these from FPS.
            (media / "timestamps.json").write_text(
                json.dumps(
                    {
                        Path(frame["path"]).name: frame["camera_timestamp"]
                        for frame in camera_frames
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        camera_data["frames"] = camera_frames
        cameras.append(camera_data)
    (output / "observations.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
    (output / "frames.json").write_text(json.dumps(frame_manifest, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "ground_truth.npz",
        times=times,
        joints=truth,
        pose=pose,
        root_translation=root,
        rest_offsets=REST_OFFSETS,
        joint_names=np.array(JOINT_NAMES),
        parents=np.array(PARENTS),
        actor_displacement_m=np.array(2.43),
    )
    project = {
        "schema_version": 1,
        "name": "Synthetic Multi-Camera Tutorial",
        "world": {
            "units": "metres",
            "up": "Y",
            "metric_scale": 1.0,
            "origin": [0, 0, 0],
            "floor": {"normal": [0, 1, 0], "offset": 0},
        },
        "cameras": cameras,
        "observations": "observations.json",
        "joint_names": list(JOINT_NAMES),
        "parents": list(PARENTS),
        "actor": {"id": "actor01", "name": "Synthetic actor", "body_model": "fixture"},
        "output_fps": fps,
        "output_dir": "outputs",
        "synthetic": {
            "seed": seed,
            "ground_truth": "ground_truth.npz",
            "frames": frames,
            "noise_px": noise_px,
            "missing_measurements": excluded,
            "outliers": outliers,
            "asynchronous": asynchronous,
            "moving_cameras": moving,
            "note": "Generated geometric fixture; procedural rig is not SMPL",
        },
    }
    config_path = output / "project.yaml"
    config_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    return config_path
