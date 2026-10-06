"""Exercise actual shared GUI/CLI service, numerical solve and rig export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openmocap.application import ProjectService
from openmocap.pipeline.cache import CheckpointStore, fingerprint
from openmocap.trajectories import ContinuousTrajectory
from openmocap.synthetic.capture import human_motion


def test_complete_asynchronous_metric_vertical_slice(tmp_path):
    service = ProjectService()
    project = service.create_example(
        tmp_path / "capture",
        camera_count=3,
        frames=18,
        noise_px=0.1,
        missing_probability=0.01,
        outlier_probability=0.01,
        render_images=False,
    )
    locked = [json.dumps(c["pose"], sort_keys=True) for c in project["cameras"]]
    progress = []
    result = service.run(project, callback=progress.append)
    assert result["frames"] >= 14
    data = service.load_result(project)
    truth, _, _ = human_motion(data["times"], 17 / 30)
    assert np.nanmedian(np.linalg.norm(data["joints"] - truth, axis=2)) < 0.015
    assert data["animation"].shape.ndim == 1
    assert np.isfinite(data["animation"].rotations).all()
    assert data["animation"].rotations.shape[0] == len(data["times"])
    assert [json.dumps(c["pose"], sort_keys=True) for c in project["cameras"]] == locked
    assert data["report"]["camera_count"] == 3
    assert data["report"]["rejected_observations"] > 0
    assert data["report"]["used_observations"] > 700
    output = Path(result["output_dir"])
    trajectory = ContinuousTrajectory.load(output / "trajectory.npz")
    unseen = (data["times"][:-1] + data["times"][1:]) / 2
    unseen_truth, _, _ = human_motion(unseen, 17 / 30)
    assert np.nanmedian(np.linalg.norm(trajectory.evaluate(unseen) - unseen_truth, axis=2)) < 0.015
    for format in ["npz", "bvh"]:
        export = service.export(project, format, tmp_path / f"actor.{format}")
        assert export["validated"]
    modified_before = (output / "joints.npz").stat().st_mtime_ns
    service.run(project)
    assert (output / "joints.npz").stat().st_mtime_ns == modified_before
    service.apply_override(project, "camera.CAM_01.time_mapping.offset", 0.004)
    assert project["stages"]["fit-motion"]["status"] == "STALE"
    reopened = service.open_project(project["path"])
    assert reopened["overrides"]["camera.CAM_01.time_mapping.offset"] == 0.004
    with pytest.raises(ValueError, match="stale"):
        service.export(project, "npz", tmp_path / "stale.npz")


def test_metric_scale_and_calibration_fail_loudly(tmp_path):
    service = ProjectService()
    project = service.create_project(tmp_path / "project")
    with pytest.raises(ValueError, match="Metric reconstruction"):
        service.run(project)
    service.apply_override(project, "world.metric_scale", 1)
    with pytest.raises(ValueError, match="uncalibrated"):
        service.run(project)
    with pytest.raises(ValueError):
        service.apply_override(project, "camera.CAM_01.time_mapping.scale", -1)
    assert project["cameras"][0]["time_mapping"]["scale"] == 1


def test_checkpoint_rejects_changed_inputs_config_or_output(tmp_path):
    source, result = tmp_path / "input.json", tmp_path / "joints.npz"
    source.write_text("first")
    result.write_bytes(b"real-output")
    key = fingerprint({"smoothing": 1}, [source])
    store = CheckpointStore(tmp_path / "cache")
    store.complete("triangulate", key, [result])
    assert store.valid("triangulate", key, [result])
    assert not store.valid("triangulate", fingerprint({"smoothing": 2}, [source]), [result])
    source.write_text("changed")
    assert not store.valid("triangulate", fingerprint({"smoothing": 1}, [source]), [result])
    result.write_bytes(b"corrupt")
    assert not store.valid("triangulate", key, [result])


def test_manual_calibration_override_survives_trusted_import(tmp_path):
    import yaml

    service = ProjectService()
    project = service.create_example(
        tmp_path / "capture", camera_count=2, frames=8, render_images=False
    )
    baseline = tmp_path / "calibration.yaml"
    baseline.write_text(yaml.safe_dump({"cameras": project["cameras"], "world": project["world"]}))
    service.apply_override(project, "camera.CAM_01.time_mapping.offset", 0.125)
    service.apply_override(project, "camera.CAM_01.locked", True)
    service.import_calibration(project, baseline)
    assert project["cameras"][0]["time_mapping"]["offset"] == 0.125
    assert project["cameras"][0]["locked"]
    assert (Path(project["path"]) / "calibration/imported.json").exists()


def test_impossible_camera_ids_are_not_filtered_silently(tmp_path):
    service = ProjectService()
    project = service.create_example(
        tmp_path / "capture", camera_count=2, frames=8, render_images=False
    )
    source = Path(project["path"]) / project["observations"]
    rows = json.loads(source.read_text())
    rows[0]["camera_id"] = "UNKNOWN"
    source.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="unknown cameras"):
        service.run(project)


def test_stale_export_itself_does_not_block_fresh_dependencies(tmp_path):
    service = ProjectService()
    project = service.create_example(
        tmp_path / "capture", camera_count=3, frames=9, noise_px=0, render_images=False
    )
    service.run(project)
    service.export(project, "npz", tmp_path / "first.npz")
    service.apply_override(project, "camera.CAM_01.time_mapping.offset", 0.001)
    service.run(project)
    assert project["stages"]["export"]["status"] == "STALE"
    assert service.export(project, "npz", tmp_path / "fresh.npz")["validated"]


def test_short_extra_camera_does_not_trim_supported_take(tmp_path):
    from openmocap.pipeline.reconstruction import reconstruct
    from openmocap.types import Camera

    service = ProjectService()
    project = service.create_example(
        tmp_path / "capture",
        camera_count=3,
        frames=18,
        noise_px=0,
        missing_probability=0,
        outlier_probability=0,
        asynchronous=False,
        render_images=False,
    )
    source = Path(project["path"]) / project["observations"]
    rows = json.loads(source.read_text())
    rows = [r for r in rows if r["camera_id"] != "CAM_03" or 0.2 <= r["camera_timestamp"] <= 0.3]
    result = reconstruct(
        [Camera.from_dict(c) for c in project["cameras"]], rows, project["joint_names"], fps=30
    )
    assert result.times[0] == pytest.approx(0)
    assert result.times[-1] == pytest.approx(17 / 30)
    assert np.isfinite(result.joints).all()


def test_remove_manual_override_restores_imported_calibration(tmp_path):
    service = ProjectService()
    project = service.create_example(
        tmp_path / "capture", camera_count=2, frames=8, render_images=False
    )
    original = project["cameras"][0]["time_mapping"]["offset"]
    service.apply_override(project, "camera.CAM_01.time_mapping.offset", 0.25)
    service.apply_override(project, "camera.CAM_01.time_mapping.offset", None)
    assert project["cameras"][0]["time_mapping"]["offset"] == original
    assert "camera.CAM_01.time_mapping.offset" not in project["overrides"]


def test_observation_overrides_fail_if_raw_output_regenerated(tmp_path):
    service = ProjectService()
    project = service.create_example(
        tmp_path / "capture", camera_count=2, frames=8, render_images=False
    )
    source = Path(project["path"]) / project["observations"]
    service.apply_override(project, "observation.0.disabled", True)
    rows = json.loads(source.read_text())
    source.write_text(json.dumps(rows[::-1]))
    with pytest.raises(ValueError, match="Raw observations changed"):
        service.run(project)
