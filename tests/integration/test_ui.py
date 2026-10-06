"""Offscreen desktop workflow tests against the application/service boundary."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QColor

from openmocap.ui.app import MainWindow
from openmocap.ui.dialogs import CameraEditor, ExportDialog, ProjectWizard
from openmocap.ui.jobs import PipelineJob
from openmocap.ui.state import Preferences


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


class Service:
    """Instrumented service boundary, without alternate reconstruction logic."""

    def __init__(self):
        self.called = []

    def create_project(self, path, name, camera_count=2, **kwargs):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        project = {
            "path": str(path),
            "project_file": str(path / "project.json"),
            "name": name,
            "output_dir": str(path / "outputs"),
            "output_fps": 24,
            "overrides": {},
            "stages": {"triangulate": "COMPLETE", "qc": "COMPLETE"},
            "cameras": [
                {
                    "id": f"CAM_{i:02d}",
                    "intrinsics": {
                        "fx": 700,
                        "fy": 700,
                        "cx": 320,
                        "cy": 240,
                        "width": 640,
                        "height": 480,
                    },
                    "pose": {"rotation": np.eye(3).tolist(), "translation": [0, 0, 5]},
                    "time_mapping": {"offset": 0, "scale": 1, "locked": True},
                    "enabled": True,
                    "locked": True,
                    "source": "surveyed",
                }
                for i in range(camera_count)
            ],
        }
        project.update(kwargs)
        self.save_project(project)
        return project

    def save_project(self, project):
        Path(project["project_file"]).write_text(json.dumps(project))

    def open_project(self, path):
        path = Path(path)
        return json.loads((path / "project.json" if path.is_dir() else path).read_text())

    def import_camera(self, project, camera_id, source, **metadata):
        camera = next(c for c in project["cameras"] if c["id"] == camera_id)
        camera["source_path"] = str(source)
        camera.update(metadata)
        self.save_project(project)

    def apply_override(self, project, key, datum):
        project["overrides"][key] = datum
        parts = key.split(".")
        target = project
        if parts[0] == "camera":
            target = next(c for c in project["cameras"] if c["id"] == parts[1])
            parts = parts[2:]
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = datum
        for stage in ["triangulate", "qc"]:
            project["stages"][stage] = "STALE"
        self.save_project(project)

    def load_result(self, project):
        return {
            "times": np.array([0, 1 / 24, 2 / 24]),
            "joints": np.array([[[0, 0, 0], [0, 1, 0]]] * 3),
            "confidence": np.ones((3, 2)),
            "joint_names": ["pelvis", "head"],
            "edges": [(0, 1)],
            "report": {
                "per_camera": {"CAM_00": {"median_px": 1.3}},
                "samples": [
                    {
                        "camera_id": "CAM_00",
                        "world_time": 1 / 24,
                        "joint_id": "head",
                        "error_px": 2.0,
                    }
                ],
            },
        }

    def run(self, project, callback=None, cancel_event=None, stage=None):
        self.called.append(("run", stage))
        callback({"stage": "triangulate", "progress": 0.5, "message": "Reconstructing"})
        for _ in range(4):
            if cancel_event.is_set():
                return {"canceled": True}
            time.sleep(0.01)
        project["stages"]["triangulate"] = "COMPLETE"
        project["stages"]["qc"] = "COMPLETE"
        self.save_project(project)
        return {"output": project["output_dir"]}


@pytest.fixture
def window(app, tmp_path):
    service = Service()
    widget = MainWindow(service, Preferences(tmp_path / "preferences.json"))
    widget.error = lambda title, details: pytest.fail(f"{title}: {details}")
    yield widget
    if widget.job and widget.job.isRunning():
        widget.job.cancel()
        widget.job.wait(5000)
    widget.close()
    app.processEvents()


def wait_until(app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("Qt job did not reach expected state")


def test_application_launch_home_and_wizard(window, app, tmp_path):
    window.show()
    app.processEvents()
    assert window.stack.currentWidget() is window.home
    assert window.recent_list.count() == 0
    wizard = ProjectWizard(window)
    wizard.name.setText("Artist Capture")
    wizard.directory.setText(str(tmp_path / "capture"))
    wizard.count.setValue(30)
    settings = wizard.settings()
    assert settings["camera_count"] == 30
    assert settings["body_model"] == "smpl"
    assert settings["ground_y"] == 0
    wizard.close()


def test_project_create_reopen_media_and_lock(window, tmp_path):
    project = window.service.create_project(tmp_path / "project", "Take 01", camera_count=30)
    window.set_project(project)
    assert window.camera_table.rowCount() == 30
    assert window.sync_table.rowCount() == 30
    assert window.stack.currentWidget() is window.workspace
    window.service.import_camera(project, "CAM_00", tmp_path / "plate.mov")
    window.refresh_project()
    assert window.camera_table.item(0, 1).text().endswith("plate.mov")
    editor = CameraEditor(project["cameras"][0])
    assert editor.locked.isChecked()
    assert editor.overrides()["camera.CAM_00.time_mapping.locked"]
    window.open_project(tmp_path / "project")
    assert window.project["name"] == "Take 01"


def test_manual_override_persistence_stale_undo(window, tmp_path):
    project = window.service.create_project(tmp_path / "project", "Take")
    window.set_project(project)
    window.push_override("camera.CAM_00.enabled", False)
    assert not project["cameras"][0]["enabled"]
    assert window.stage_items["triangulate"].text(1) == "STALE"
    assert (
        window.service.open_project(tmp_path / "project")["overrides"]["camera.CAM_00.enabled"]
        is False
    )
    window.undo_stack.undo()
    assert project["cameras"][0]["enabled"]
    window.undo_stack.redo()
    assert not project["cameras"][0]["enabled"]
    assert (tmp_path / "preferences.json").exists()


def test_pipeline_trigger_job_status_and_qc(window, app, tmp_path):
    window.set_project(window.service.create_project(tmp_path / "project", "Take"))
    window.run_pipeline(stage="triangulate", confirm=False)
    assert window.cancel_button.isEnabled()
    wait_until(
        app,
        lambda: window.job is not None
        and not window.job.isRunning()
        and window.run_button.isEnabled(),
    )
    assert window.service.called == [("run", "triangulate")]
    assert window.progress.value() == 100
    assert window.stage_items["triangulate"].text(1) == "COMPLETE"
    assert window.qc_table.rowCount() == 2
    window.qc_filter.setText("CAM_00")
    assert not window.qc_table.isRowHidden(0)
    window.navigate_qc(window.qc_table.item(1, 0))
    assert window._time_index == 1


def test_tiled_view_world_timeline_and_scene_render(window, app, tmp_path):
    window.set_project(window.service.create_project(tmp_path / "project", "Take", camera_count=8))
    window.viewer_mode.setCurrentText("4 Cameras")
    window.set_time_index(2)
    window.show()
    app.processEvents()
    assert len(window.plate_views) == 4
    assert all(view.world_time == pytest.approx(2 / 24) for view in window.plate_views)
    assert not window.grab().isNull()
    window.viewer_mode.setCurrentText("3D Scene")
    window.scene_view.frame_actor()
    window.scene_view.orthographic = True
    app.processEvents()
    assert not window.scene_view.grab().isNull()


def test_export_dialog_valid_configuration(app, tmp_path):
    dialog = ExportDialog(tmp_path)
    dialog.format.setCurrentText("fbx")
    dialog.fps.setValue(23.976)
    config = dialog.configuration()
    assert config["format"] == "fbx"
    assert config["path"].endswith(".fbx")
    assert config["fps"] == pytest.approx(23.976)
    assert config["validate"]
    dialog.end.setValue(0.5)
    dialog.start.setValue(1)
    with pytest.raises(ValueError, match="end must follow"):
        dialog.configuration()
    dialog.close()


def test_plate_decode_uses_timestamp_manifest(window, app, tmp_path):
    project = window.service.create_project(tmp_path / "project", "VFR")
    source = tmp_path / "images"
    source.mkdir()
    frames = []
    for frame, timestamp in enumerate([0.0, 0.04, 0.08]):
        filename = source / f"frame{frame:03d}.png"
        image = QImage(640, 480, QImage.Format.Format_RGB32)
        image.fill(QColor("#273c50"))
        image.save(str(filename))
        frames.append({"camera_timestamp": timestamp, "path": str(filename)})
    project["cameras"][0].update(source_path=str(source), frames=frames)
    project["cameras"][0]["time_mapping"]["offset"] = 0.02
    window.set_project(project)
    window.viewer_mode.setCurrentText("1 Camera")
    window.set_time_index(2)
    view = window.plate_views[0]
    wait_until(app, lambda: not view.loading and view._background is not None)
    assert view.actual_camera_time == pytest.approx(0.08)
    assert view.world_time == pytest.approx(2 / 24)


def test_job_cooperative_cancel_and_failure(app):
    entered = threading.Event()

    def task(callback, cancel):
        entered.set()
        while not cancel.wait(0.01):
            callback({"progress": 0.1})
        return "cancelled"

    job = PipelineJob(task)
    outcomes = []
    job.succeeded.connect(outcomes.append)
    job.start()
    assert entered.wait(1)
    job.cancel()
    wait_until(app, lambda: not job.isRunning() and bool(outcomes))
    assert outcomes == ["cancelled"]
    failed = PipelineJob(lambda callback, cancel: 1 / 0)
    errors = []
    failed.failed.connect(errors.append)
    failed.start()
    wait_until(app, lambda: not failed.isRunning() and bool(errors))
    assert "ZeroDivisionError" in errors[0]


def test_camera_basic_editor_preserves_unedited_locks_and_updates_values(app):
    camera = {
        "id": "CAM_01",
        "locked": True,
        "intrinsics": {"fx": 700, "fy": 701, "cx": 320, "cy": 240, "width": 640, "height": 480},
        "pose": {"rotation": np.eye(3).tolist(), "translation": [0, 0, 5]},
    }
    editor = CameraEditor(camera)
    assert "camera.CAM_01.intrinsics" not in editor.overrides()
    editor.basic_intrinsics["fx"].setValue(900)
    editor.translation_controls[0].setValue(2.43)
    changes = editor.overrides()
    assert changes["camera.CAM_01.intrinsics"]["fx"] == 900
    assert changes["camera.CAM_01.intrinsics"]["fy"] == 701
    assert changes["camera.CAM_01.pose"]["translation"][0] == pytest.approx(2.43)
    assert changes["camera.CAM_01.locked"]
    editor.close()


def test_gui_inference_configuration_uses_service_overrides(window, tmp_path):
    window.set_project(window.service.create_project(tmp_path / "project", "Inference"))
    window.pose_checkpoint.setText(str(tmp_path / "pose.onnx"))
    window.detector_backend.setCurrentText("hog")
    window.keypoint_schema.setCurrentText("SMPL24")
    window.apply_inference()
    assert window.project["perception"]["pose_model"].endswith("pose.onnx")
    assert len(window.project["perception"]["joint_names"]) == 24
    assert window.project["perception"]["detector_backend"] == "hog"


def test_inference_preset_is_atomic_and_preserves_geometry(window, tmp_path, monkeypatch):
    import copy
    import yaml

    project = window.service.create_project(tmp_path / "project", "Take")
    project["body_model"] = "fixture"
    project["perception"] = {"frame_stride": 2, "pose": {"bbox_padding": 1.3}}
    window.set_project(project)
    camera_before = copy.deepcopy(project["cameras"])
    previous = copy.deepcopy(project["perception"])
    monkeypatch.setenv("OPENMOCAP_INSTALL_ROOT", str(tmp_path))
    model = tmp_path / "models" / "pose.onnx"
    model.parent.mkdir()
    model.write_bytes(b"test-only model-file fixture; not executable inference")
    preset = tmp_path / "installed.yaml"
    preset.write_text(
        yaml.safe_dump(
            {
                "name": "Installed test preset",
                "perception": {
                    "pose_model": "${INSTALL_ROOT}/models/pose.onnx",
                    "detector_backend": "hog",
                    "schema": "SMPL24",
                    "joint_names": [f"joint_{index}" for index in range(24)],
                    "pose": {"output_format": "heatmap"},
                    "providers": ["CPUExecutionProvider"],
                },
            }
        )
    )
    window.load_inference_preset(preset)
    assert project["perception"]["pose_model"] == str(model)
    assert project["perception"]["pose"]["bbox_padding"] == 1.3
    assert project["perception"]["pose"]["output_format"] == "heatmap"
    assert project["perception"]["frame_stride"] == 2
    assert project["cameras"] == camera_before
    assert project["body_model"] == "fixture"
    assert project["name"] == "Take"
    assert window.undo_stack.count() == 1
    window.undo_stack.undo()
    assert project["perception"] == previous


def test_inference_preset_rejects_missing_models_and_project_edits(window, tmp_path):
    import copy
    import yaml

    project = window.service.create_project(tmp_path / "project", "Take")
    window.set_project(project)
    before = copy.deepcopy(project)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump({"perception": {"detector_backend": "hog"}, "cameras": []}))
    with pytest.raises(ValueError, match="may not modify project"):
        window.load_inference_preset(invalid)
    assert project == before
    invalid.write_text(yaml.safe_dump({"perception": {"pose_model": "missing.onnx"}}))
    with pytest.raises(FileNotFoundError, match="missing pose_model"):
        window.load_inference_preset(invalid)
    assert project == before
    invalid.write_text(yaml.safe_dump({"perception": {"joint_names": ["wrist", "wrist"]}}))
    with pytest.raises(ValueError, match="distinct"):
        window.load_inference_preset(invalid)
    assert project == before


def test_inference_preset_dialog_reports_malformed_yaml(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    window.set_project(window.service.create_project(tmp_path / "project", "Take"))
    preset = tmp_path / "malformed.yaml"
    preset.write_text("perception: [this is malformed")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(preset), ""))
    errors = []
    window.error = lambda title, details: errors.append((title, details))
    window.import_inference_preset()
    assert len(errors) == 1
    assert "could not be loaded" in errors[0][0]
    assert "perception" not in window.project


def test_wholebody_preset_and_apply_keep_exact_133_landmarks(window, tmp_path):
    import copy
    import yaml

    from openmocap.pose2d.schemas import COCO_WHOLEBODY_133

    assert len(COCO_WHOLEBODY_133) == len(set(COCO_WHOLEBODY_133)) == 133
    assert COCO_WHOLEBODY_133[17:23] == (
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
    )
    assert COCO_WHOLEBODY_133[23] == "face-0"
    assert COCO_WHOLEBODY_133[90] == "face-67"
    assert COCO_WHOLEBODY_133[91] == "left_hand_root"
    assert COCO_WHOLEBODY_133[112] == "right_hand_root"
    project = window.service.create_project(tmp_path / "project", "Whole-body")
    window.set_project(project)
    checkpoint = tmp_path / "pose.onnx"
    checkpoint.write_bytes(b"test-only path fixture, not inference")
    preset = tmp_path / "wholebody.yaml"
    preset.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "perception": {
                    "pose_model": str(checkpoint),
                    "schema": "COCOWholeBody133",
                    "joint_names": list(COCO_WHOLEBODY_133),
                    "pose": {"output_format": "simcc", "split_ratio": 2},
                    "segmentation": {"iterations": 5},
                },
            }
        )
    )
    window.load_inference_preset(preset)
    assert window.keypoint_schema.currentText() == "COCOWholeBody133"
    codec = copy.deepcopy(project["perception"]["pose"])
    window.apply_inference()
    assert project["perception"]["joint_names"] == list(COCO_WHOLEBODY_133)
    assert project["perception"]["pose"] == codec
    reopened = window.service.open_project(project["path"])
    assert len(reopened["perception"]["joint_names"]) == 133


def test_shipped_wholebody_preset_imports_supported_configuration(window, tmp_path, monkeypatch):
    import copy
    import yaml

    from openmocap.pose2d.schemas import COCO_WHOLEBODY_133

    monkeypatch.setenv("OPENMOCAP_INSTALL_ROOT", str(tmp_path))
    for relative in (
        "models/detection/yolox_m_humanart.onnx",
        "models/pose/rtmw_wholebody133.onnx",
    ):
        model = tmp_path / relative
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"test-only path fixture, not neural inference")
    project = window.service.create_project(tmp_path / "project", "Whole-body preset")
    project["body_model"] = "fixture"
    before = copy.deepcopy(project["cameras"])
    window.set_project(project)
    preset = Path(__file__).resolve().parents[2] / "configs/pose/rtmpose-wholebody-yolox.yaml"
    expected = yaml.safe_load(preset.read_text())["perception"]
    window.load_inference_preset(preset)
    assert project["perception"]["joint_names"] == list(COCO_WHOLEBODY_133)
    assert project["perception"]["pose"] == expected["pose"]
    assert project["perception"]["detector"] == expected["detector"]
    assert project["perception"]["pose_model"] == str(
        tmp_path / "models/pose/rtmw_wholebody133.onnx"
    )
    assert window.keypoint_schema.currentText() == "COCOWholeBody133"
    assert project["cameras"] == before
    assert project["body_model"] == "fixture"


def test_real_service_gui_solve_qc_export_reopen(app, tmp_path):
    """A real geometric solve through the GUI's background job API."""
    from openmocap.application import ProjectService

    service = ProjectService()
    project = service.create_example(
        tmp_path / "synthetic",
        camera_count=3,
        frames=12,
        fps=24,
        noise_px=0.1,
        missing_probability=0,
        outlier_probability=0,
        asynchronous=True,
    )
    widget = MainWindow(service, Preferences(tmp_path / "real-ui-preferences.json"))
    try:
        errors = []
        widget.error = lambda title, details: errors.append((title, details))
        widget.set_project(project)
        widget.show()
        widget.run_pipeline(confirm=False)
        wait_until(
            app, lambda: not widget.job.isRunning() and widget.run_button.isEnabled(), timeout=120
        )
        assert not errors
        assert len(widget.result["times"]) >= 6
        assert widget.result["joints"].shape[1:] == (24, 3)
        assert widget.qc_table.rowCount() > 0
        assert widget.stage_items["fit-motion"].text(1) == "COMPLETE"
        widget.viewer_mode.setCurrentText("2 Cameras")
        widget.set_time_index(3)
        wait_until(app, lambda: all(not view.loading for view in widget.plate_views))
        assert all(view._background is not None for view in widget.plate_views)
        output = tmp_path / "delivered" / "actor.npz"
        widget.start_job(
            "Export NPZ",
            lambda callback, cancel: service.export(project, "npz", output, fps=24, validate=True),
            widget._export_complete,
        )
        wait_until(
            app, lambda: not widget.job.isRunning() and widget.run_button.isEnabled(), timeout=15
        )
        assert not errors
        assert output.exists()
        assert "EXPORT VALIDATED" in widget.export_summary.toPlainText()
        if shutil.which("blender"):
            character = output.with_suffix(".fbx")
            widget.start_job(
                "Export FBX",
                lambda callback, cancel: service.export(
                    project, "fbx", character, fps=24, end=0.16, validate=True
                ),
                widget._export_complete,
            )
            wait_until(
                app,
                lambda: not widget.job.isRunning() and widget.run_button.isEnabled(),
                timeout=45,
            )
            assert not errors
            validation = json.loads(character.with_suffix(".fbx.json").read_text())
            assert validation["validated"] and validation["bone_count"] == 24
            assert validation["weighted_vertices"] > 0
            assert "EXPORT VALIDATED" in widget.export_summary.toPlainText()
        raw_file = Path(project["path"]) / project["observations"]
        raw_before = raw_file.read_bytes()
        widget.inspect_raw_observation(0)
        widget.correct_observation(True)
        assert project["overrides"]["observation.0.disabled"]
        assert raw_file.read_bytes() == raw_before
        widget.push_override("camera.CAM_01.time_mapping.offset", 0.01)
        assert widget.stage_items["triangulate"].text(1) == "STALE"
        widget.open_project(project["path"])
        assert widget.project["overrides"]["camera.CAM_01.time_mapping.offset"] == 0.01
    finally:
        if widget.job is not None and widget.job.isRunning():
            widget.job.cancel()
            while widget.job.isRunning():
                widget.job.wait(50)
                app.processEvents()
        widget.close()
        app.processEvents()
