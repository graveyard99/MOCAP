"""Regression tests for production edits, stage invalidation and geometric trust."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openmocap.application import ProjectService
from openmocap.cameras import look_at
from openmocap.pipeline.reconstruction import _native_time_fit
from openmocap.trajectories import fit_trajectory
from openmocap.types import CameraIntrinsics


class ReconstructionBoundaryReached(RuntimeError):
    """Stop after inspecting effective input, before running a numerical job."""


def test_inference_change_marks_raw_stages_stale_without_changing_calibration(tmp_path):
    service = ProjectService()
    project = service.create_project(tmp_path / "inference-edit")
    for name in ["ingest", "calibrate", "sync", "detect", "pose2d", "segment", "triangulate"]:
        project["stages"][name]["status"] = "COMPLETE"
    original_cameras = json.dumps(project["cameras"], sort_keys=True)
    service.apply_override(project, "perception.pose_model", "new-pose.onnx")
    for name in ["detect", "pose2d", "segment", "triangulate"]:
        assert project["stages"][name]["status"] == "STALE"
    for name in ["ingest", "calibrate", "sync"]:
        assert project["stages"][name]["status"] == "COMPLETE"
    assert json.dumps(project["cameras"], sort_keys=True) == original_cameras


def _project_with_raw_identifiers(tmp_path):
    service = ProjectService()
    project = service.create_example(
        tmp_path / "capture",
        camera_count=3,
        frames=8,
        noise_px=0,
        missing_probability=0,
        outlier_probability=0,
        render_images=False,
    )
    source = Path(project["path"]) / project["observations"]
    rows = json.loads(source.read_text())
    for index, row in enumerate(rows):
        row["raw_identifier"] = index
    source.write_text(json.dumps(rows))
    return service, project, source, rows


def test_observation_edit_keeps_raw_identity_after_camera_is_disabled(tmp_path, monkeypatch):
    service, project, source, rows = _project_with_raw_identifiers(tmp_path)
    target = next(index for index, row in enumerate(rows) if row["camera_id"] == "CAM_02")
    service.apply_override(project, f"observation.{target}.disabled", True)
    service.apply_override(project, "camera.CAM_01.enabled", False)
    received = []

    def inspect(cameras, effective, *args, **kwargs):
        received.extend(effective)
        raise ReconstructionBoundaryReached()

    monkeypatch.setattr("openmocap.pipeline.reconstruction.reconstruct", inspect)
    with pytest.raises(ReconstructionBoundaryReached):
        service.run(project, stage="triangulate")
    disabled = [row["raw_identifier"] for row in received if row.get("disabled", False)]
    assert disabled == [target]
    assert not any(row.get("disabled", False) for row in json.loads(source.read_text()))


def test_unknown_camera_measurement_is_not_silently_discarded(tmp_path, monkeypatch):
    service, project, source, rows = _project_with_raw_identifiers(tmp_path)
    rows.append({**rows[0], "camera_id": "IMPOSSIBLE_CAMERA"})
    source.write_text(json.dumps(rows))

    def unexpected(*args, **kwargs):
        raise ReconstructionBoundaryReached("Unknown camera must fail before reconstruction")

    monkeypatch.setattr("openmocap.pipeline.reconstruction.reconstruct", unexpected)
    with pytest.raises(ValueError, match="[Uu]nknown.*camera|camera.*[Uu]nknown|IMPOSSIBLE_CAMERA"):
        service.run(project, stage="triangulate")


def test_export_can_be_regenerated_when_only_previous_export_is_stale(tmp_path):
    service = ProjectService()
    project = service.create_example(
        tmp_path / "capture",
        camera_count=2,
        frames=8,
        noise_px=0,
        missing_probability=0,
        outlier_probability=0,
        render_images=False,
    )
    service.run(project)
    service.export(project, "npz", tmp_path / "first.npz")
    project["stages"]["export"]["status"] = "STALE"
    service.save_project(project)
    result = service.export(project, "npz", tmp_path / "replacement.npz")
    assert result["validated"]
    assert project["stages"]["export"]["status"] == "COMPLETE"


def test_native_ray_refinement_cannot_replace_strong_3d_with_one_camera():
    intrinsics = CameraIntrinsics(800.0, 800.0, 640.0, 360.0, 1280, 720)
    camera = look_at("A", [3.0, 2.0, 4.0], [0.5, 1.0, 0.0], intrinsics)
    times = np.linspace(0, 1, 61)
    points = np.column_stack([times, np.ones(len(times)), np.zeros(len(times))])[:, None, :]
    initial = fit_trajectory(
        times, points, np.ones((len(times), 1)), joint_names=["wrist"], smoothing=0
    )
    pixels = np.array(
        [
            camera.project(pose, time=float(time))[0]
            for time, pose in zip(times, points, strict=True)
        ]
    )
    groups = {("A", "wrist"): (times, pixels, np.ones(len(times)))}
    refined = _native_time_fit({"A": camera}, groups, times, initial, ["wrist"])
    assert np.max(np.linalg.norm(refined.evaluate(times) - points, axis=-1)) < 0.002


def test_brief_extra_camera_does_not_truncate_other_multiview_support():
    from openmocap.pipeline.reconstruction import reconstruct

    intrinsics = CameraIntrinsics(800.0, 800.0, 640.0, 360.0, 1280, 720)
    cameras = [
        look_at("A", [3.0, 2.0, 4.0], [0.0, 1.0, 0.0], intrinsics),
        look_at("B", [-3.0, 2.0, 3.0], [0.0, 1.0, 0.0], intrinsics),
        look_at("C", [1.0, 2.0, -4.0], [0.0, 1.0, 0.0], intrinsics),
    ]
    times = np.arange(31) / 30
    rows = []
    for camera in cameras:
        for time in times:
            if camera.id == "C" and not 0.4 <= time <= 0.6:
                continue
            point = np.array([0.2 * time, 1.0, 0.0])
            rows.append(
                {
                    "camera_id": camera.id,
                    "camera_timestamp": float(time),
                    "person_id": "actor01",
                    "joint_id": "wrist",
                    "xy": camera.project(point, time=float(time)).tolist(),
                    "confidence": 1,
                }
            )
    result = reconstruct(cameras, rows, ["wrist"], fps=30)
    assert np.isclose(result.times[0], 0)
    assert np.isclose(result.times[-1], 1)
    assert len(result.times) == 31
    assert (
        np.max(
            np.linalg.norm(
                result.joints[:, 0] - np.column_stack([0.2 * times, np.ones(31), np.zeros(31)]),
                axis=1,
            )
        )
        < 0.001
    )
