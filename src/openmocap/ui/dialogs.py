"""Guided project creation and explicit calibrated metadata editors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from .state import camera_id, install_root, value


def number(minimum: float, maximum: float, initial: float, decimals: int = 4) -> QDoubleSpinBox:
    control = QDoubleSpinBox()
    control.setRange(minimum, maximum)
    control.setDecimals(decimals)
    control.setValue(initial)
    return control


def browse_field(edit: QLineEdit, directory: bool = False) -> QWidget:
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit)
    button = QPushButton("Browse…")

    def browse() -> None:
        selected = (
            QFileDialog.getExistingDirectory(widget, "Choose directory", edit.text())
            if directory
            else QFileDialog.getOpenFileName(widget, "Choose file", edit.text())[0]
        )
        if selected:
            edit.setText(selected)

    button.clicked.connect(browse)
    layout.addWidget(button)
    return widget


class ProjectWizard(QWizard):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New capture project")
        self.setMinimumSize(720, 600)
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.name = QLineEdit("Untitled Capture")
        self.directory = QLineEdit(str(install_root() / "projects" / "untitled-capture"))
        self.footage = QLineEdit()
        self.count = QSpinBox()
        self.count.setRange(2, 256)
        self.count.setValue(4)
        self.prefix = QLineEdit("CAM_")
        self.input_fps = number(1, 240, 30, 3)
        self.calibration = QComboBox()
        self.calibration.addItems(
            ["Import calibrated cameras", "Surveyed / manual values", "Calibration target"]
        )
        self.calibration_file = QLineEdit()
        self.metric = QComboBox()
        self.metric.addItems(
            [
                "Calibrated metres",
                "Surveyed control / known distance",
                "Actor height (explicit scale assumption)",
            ]
        )
        self.floor = number(-10000, 10000, 0)
        self.sync = QComboBox()
        self.sync.addItems(
            ["Imported timestamp mapping", "Manual offsets", "Event / trajectory evidence"]
        )
        self.actor = QLineEdit("Actor 01")
        self.height = number(0, 3, 0)
        self.height.setSpecialValueText("Unknown")
        self.model = QComboBox()
        self.model.addItems(["smpl", "smplh", "smplx", "fixture (synthetic tutorial only)"])
        self.model_path = QLineEdit(str(install_root() / "models" / "body" / "SMPL_NEUTRAL.npz"))
        self.output_fps = QComboBox()
        self.output_fps.setEditable(True)
        self.output_fps.addItems(["24", "23.976", "25", "29.97", "30", "50", "59.94", "60"])
        self.output = QLineEdit()
        self._page(
            "Project and footage",
            "Create a self-contained capture workspace. Media stays at its original location.",
            [
                ("Project name", self.name),
                ("Project directory", browse_field(self.directory, True)),
                ("Footage folder (optional)", browse_field(self.footage, True)),
                ("Camera count", self.count),
                ("Camera name prefix", self.prefix),
                ("Expected source FPS", self.input_fps),
            ],
        )
        self._page(
            "Cameras, scale and timing",
            "Camera calibration is required before a metric solve. New cameras are uncalibrated until imported or measured.",
            [
                ("Calibration source", self.calibration),
                ("Calibration file", browse_field(self.calibration_file)),
                ("World axes", QLabel("Metres · +Y up · OpenCV camera coordinates")),
                ("Metric scale source", self.metric),
                ("Ground plane Y (m)", self.floor),
                ("Synchronization", self.sync),
            ],
        )
        self._page(
            "Actor and delivery",
            "Actor proportions remain constant across the take. SMPL-family assets must be obtained separately; the fixture is explicitly synthetic.",
            [
                ("Actor name", self.actor),
                ("Known height (m)", self.height),
                ("Body model", self.model),
                ("Licensed model NPZ file", browse_field(self.model_path)),
                ("Output FPS", self.output_fps),
                ("Output folder (blank = project outputs)", browse_field(self.output, True)),
            ],
        )
        self.name.setToolTip("Name shown in the project manager and solve reports.")
        self.floor.setToolTip(
            "Surveyed floor height in the configured metric world. This is not inferred from slow feet."
        )
        self.height.setToolTip(
            "Optional measured height. Zero means unknown, never an automatic scale estimate."
        )

    def _page(self, title: str, subtitle: str, fields: list[tuple[str, QWidget]]) -> None:
        page = QWizardPage()
        page.setTitle(title)
        page.setSubTitle(subtitle)
        layout = QFormLayout(page)
        for label, widget in fields:
            layout.addRow(label, widget)
        self.addPage(page)

    def settings(self) -> dict[str, Any]:
        fps = float(self.output_fps.currentText())
        if fps <= 0:
            raise ValueError("Output frame rate must be positive")
        if not self.name.text().strip() or not self.directory.text().strip():
            raise ValueError("Project name and directory are required")
        return {
            "name": self.name.text().strip(),
            "path": self.directory.text().strip(),
            "camera_count": self.count.value(),
            "camera_prefix": self.prefix.text(),
            "footage_dir": self.footage.text(),
            "source_fps": self.input_fps.value(),
            "calibration_source": self.calibration.currentText(),
            "calibration_file": self.calibration_file.text(),
            "metric_scale_source": self.metric.currentText(),
            "ground_y": self.floor.value(),
            "synchronization_method": self.sync.currentText(),
            "actor_name": self.actor.text(),
            "actor_height": self.height.value() or None,
            "body_model": self.model.currentText().split()[0],
            "model_path": self.model_path.text(),
            "output_fps": fps,
            "output_dir": self.output.text() or None,
        }


class CameraEditor(QDialog):
    """Editor produces audited overrides; raw imported calibration is retained."""

    def __init__(self, camera: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.identifier = camera_id(camera)
        self.setWindowTitle(f"Camera {self.identifier} · manual overrides")
        self.setMinimumWidth(640)
        layout = QVBoxLayout(self)
        notice = QLabel(
            "Authoritative calibration remains locked by default. Explicit manual edits are stored as reversible overrides."
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)
        container = QWidget()
        form = QFormLayout(container)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setMinimumHeight(500)
        scroll.setMaximumHeight(650)
        layout.addWidget(scroll)
        self.enabled = QCheckBox("Include this camera in reconstruction")
        self.enabled.setChecked(bool(value(camera, "enabled", True)))
        self.locked = QCheckBox("Lock transform and intrinsics")
        self.locked.setChecked(bool(value(camera, "locked", True)))
        mapping = value(camera, "time_mapping", {})
        self.offset = number(-86400, 86400, float(value(mapping, "offset", 0)), 6)
        self.scale = number(0.9, 1.1, float(value(mapping, "scale", 1)), 8)
        self.sync_lock = QCheckBox("Lock synchronization mapping")
        self.sync_lock.setChecked(bool(value(mapping, "locked", False)))
        quality = value(camera, "quality", 1)
        self.quality = number(
            0,
            1,
            float(
                quality if isinstance(quality, (int, float)) else value(quality, "confidence", 1)
            ),
        )
        self.intrinsics = QPlainTextEdit(
            json.dumps(value(camera, "intrinsics", {}), indent=2, default=str)
        )
        self.pose = QPlainTextEdit(json.dumps(value(camera, "pose", {}), indent=2, default=str))
        self.intrinsics.setMaximumHeight(170)
        self.pose.setMaximumHeight(170)
        self.original_intrinsics = self.intrinsics.toPlainText()
        self.original_pose = self.pose.toPlainText()
        self.basic_intrinsics: dict[str, QDoubleSpinBox] = {}
        self.original_basic: dict[str, float] = {}
        metadata = value(camera, "intrinsics", {})
        intrinsics_tabs = QTabWidget()
        basic = QWidget()
        basic_form = QFormLayout(basic)
        for key, label, maximum in [
            ("fx", "Horizontal focal length (px)", 100000),
            ("fy", "Vertical focal length (px)", 100000),
            ("cx", "Principal point X (px)", 100000),
            ("cy", "Principal point Y (px)", 100000),
            ("width", "Image width (px)", 100000),
            ("height", "Image height (px)", 100000),
            ("sensor_width_mm", "Sensor width (mm, optional)", 500),
            ("sensor_height_mm", "Sensor height (mm, optional)", 500),
            ("focal_length_mm", "Focal length (mm, optional)", 2000),
        ]:
            initial = float(value(metadata, key, 0) or 0)
            self.original_basic[key] = initial
            control = number(0, maximum, initial, 0 if key in {"width", "height"} else 4)
            control.setToolTip(
                "Manual measured metadata. Zero means unknown for optional sensor/focal metadata."
            )
            self.basic_intrinsics[key] = control
            basic_form.addRow(label, control)
        self.lens_model = QComboBox()
        self.lens_model.addItems(["opencv", "fisheye", "none"])
        self.lens_model.setCurrentText(str(value(metadata, "distortion_model", "opencv")))
        self.original_lens = self.lens_model.currentText()
        basic_form.addRow("Lens model", self.lens_model)
        self.distortion_coefficients = QLineEdit(
            ", ".join(str(v) for v in value(metadata, "distortion", []))
        )
        self.original_distortion = self.distortion_coefficients.text()
        self.distortion_coefficients.setToolTip(
            "Comma-separated measured coefficients. OpenCV k1,k2,p1,p2,k3...; fisheye exactly four."
        )
        basic_form.addRow("Distortion coefficients", self.distortion_coefficients)
        intrinsics_tabs.addTab(basic, "Camera / lens")
        intrinsics_tabs.addTab(self.intrinsics, "Expert values")
        pose_tabs = QTabWidget()
        pose_basic = QWidget()
        pose_form = QFormLayout(pose_basic)
        pose_data = value(camera, "pose", {})
        translation = value(pose_data, "translation", [0, 0, 0])
        self.translation_controls = [number(-1e6, 1e6, float(v), 6) for v in translation]
        self.original_translation = [c.value() for c in self.translation_controls]
        for axis, control in zip("XYZ", self.translation_controls, strict=True):
            pose_form.addRow(f"World→camera translation {axis} (m)", control)
        from scipy.spatial.transform import Rotation

        rotation = value(pose_data, "rotation", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        angles = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
        self.rotation_controls = [number(-360, 360, float(v), 5) for v in angles]
        self.original_angles = [c.value() for c in self.rotation_controls]
        for axis, control in zip("XYZ", self.rotation_controls, strict=True):
            pose_form.addRow(f"OpenCV world→camera Euler {axis} (deg)", control)
        pose_tabs.addTab(pose_basic, "Transform")
        pose_tabs.addTab(self.pose, "Expert values")
        for label, widget in [
            ("Enabled", self.enabled),
            ("Calibration lock", self.locked),
            ("World offset (s)", self.offset),
            ("Clock scale", self.scale),
            ("Timing lock", self.sync_lock),
            ("Quality weight", self.quality),
            ("Intrinsics", intrinsics_tabs),
            ("World→camera transform", pose_tabs),
        ]:
            form.addRow(label, widget)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def overrides(self) -> dict[str, Any]:
        prefix = f"camera.{self.identifier}."
        result = {
            prefix + "enabled": self.enabled.isChecked(),
            prefix + "locked": self.locked.isChecked(),
            prefix + "time_mapping.offset": self.offset.value(),
            prefix + "time_mapping.scale": self.scale.value(),
            prefix + "time_mapping.locked": self.sync_lock.isChecked(),
            prefix + "quality.confidence": self.quality.value(),
        }
        if self.intrinsics.toPlainText() != self.original_intrinsics:
            result[prefix + "intrinsics"] = json.loads(self.intrinsics.toPlainText())
        basic_changed = any(
            control.value() != self.original_basic[key]
            for key, control in self.basic_intrinsics.items()
        )
        if (
            basic_changed
            or self.lens_model.currentText() != self.original_lens
            or self.distortion_coefficients.text() != self.original_distortion
        ):
            intrinsic = result.get(prefix + "intrinsics", json.loads(self.original_intrinsics))
            for key, control in self.basic_intrinsics.items():
                if key in {"width", "height"}:
                    intrinsic[key] = int(control.value())
                elif control.value() or key in {"fx", "fy", "cx", "cy"}:
                    intrinsic[key] = control.value()
                else:
                    intrinsic.pop(key, None)
            intrinsic["distortion_model"] = self.lens_model.currentText()
            intrinsic["distortion"] = [
                float(v.strip())
                for v in self.distortion_coefficients.text().split(",")
                if v.strip()
            ]
            intrinsic["source"] = "manual"
            intrinsic["locked"] = self.locked.isChecked()
            result[prefix + "intrinsics"] = intrinsic
        if self.pose.toPlainText() != self.original_pose:
            result[prefix + "pose"] = json.loads(self.pose.toPlainText())
        translation = [c.value() for c in self.translation_controls]
        angles = [c.value() for c in self.rotation_controls]
        if translation != self.original_translation or angles != self.original_angles:
            from scipy.spatial.transform import Rotation

            pose = result.get(prefix + "pose", json.loads(self.original_pose))
            pose["translation"] = translation
            pose["rotation"] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix().tolist()
            result[prefix + "pose"] = pose
        return result


class ExportDialog(QDialog):
    def __init__(self, output: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export animation")
        self.setMinimumWidth(620)
        form = QFormLayout(self)
        self.format = QComboBox()
        self.format.addItems(["npz", "bvh", "fbx", "usd"])
        self.target = QComboBox()
        self.target.addItems(
            ["Metric +Y world", "Maya", "Houdini", "Blender", "Cinema 4D", "Unreal Engine"]
        )
        self.path = QLineEdit(str(output / "actor.npz"))
        self.fps = number(1, 240, 24, 3)
        self.start = number(0, 1e9, 0, 3)
        self.end = number(0, 1e9, 0, 3)
        self.end.setSpecialValueText("Full take")
        self.mesh = QCheckBox("Include mesh and skinning when format supports them")
        self.mesh.setChecked(True)
        self.cameras = QCheckBox("Include reconstructed cameras / metadata")
        self.cameras.setChecked(True)
        self.validate = QCheckBox(
            "Validate exported file (FBX: re-import into clean Blender scene)"
        )
        self.validate.setChecked(True)
        self.format.currentTextChanged.connect(self._format_changed)
        for label, widget in [
            ("Format", self.format),
            ("Output file", browse_field(self.path)),
            ("DCC target", self.target),
            ("Output FPS", self.fps),
            ("Start time (s)", self.start),
            ("End time (s)", self.end),
            ("Character", self.mesh),
            ("Scene", self.cameras),
            ("Verification", self.validate),
        ]:
            form.addRow(label, widget)
        notice = QLabel(
            "FBX requires the project-local Blender backend. SMPL-family export requires licensed assets. Format and backend failures are reported; only actual validation earns VALIDATED."
        )
        notice.setWordWrap(True)
        form.addRow(notice)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _format_changed(self, fmt: str) -> None:
        self.path.setText(str(Path(self.path.text()).with_suffix("." + fmt)))

    def configuration(self) -> dict[str, Any]:
        if self.end.value() and self.end.value() <= self.start.value():
            raise ValueError("Export end must follow start")
        return {
            "format": self.format.currentText(),
            "path": self.path.text(),
            "fps": self.fps.value(),
            "start": self.start.value(),
            "end": self.end.value() or None,
            "target": self.target.currentText(),
            "include_mesh": self.mesh.isChecked(),
            "include_cameras": self.cameras.isChecked(),
            "validate": self.validate.isChecked(),
        }
