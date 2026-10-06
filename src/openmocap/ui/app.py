"""Desktop workspace backed by the same application service as the CLI."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .dialogs import CameraEditor, ExportDialog, ProjectWizard, browse_field, number
from .jobs import PipelineJob
from .state import Preferences, camera_id, cameras, install_root, project_path, value
from .viewers import ContactTimeline, PlateView, SceneView

STAGES = [
    ("ingest", "01  Footage", "Cameras"),
    ("calibrate", "02  Calibration", "Calibration"),
    ("sync", "03  Synchronization", "Synchronization"),
    ("detect", "04  Detection", "Viewer"),
    ("pose2d", "05  2D Pose", "Viewer"),
    ("segment", "06  Segmentation", "Viewer"),
    ("associate", "07  Association", "Viewer"),
    ("triangulate", "08  3D Reconstruction", "Viewer"),
    ("fit-shape", "09  Actor Shape", "Actor"),
    ("fit-motion", "10  Body Solve", "Actor"),
    ("contacts", "11  Contacts", "Quality Control"),
    ("physics", "12  Physics", "Quality Control"),
    ("qc", "13  Quality Control", "Quality Control"),
    ("export", "14  Export", "Export"),
]
STYLE = """
QWidget { background:#1d2532; color:#dce5f0; font-family:'DejaVu Sans'; font-size:12px; }
QMainWindow,QDialog { background:#171f2b; }
QMenuBar,QToolBar { background:#202b3b; border-bottom:1px solid #344457; }
QMenuBar::item:selected,QMenu::item:selected { background:#344c66; }
QPushButton { background:#30445c; border:1px solid #425a73; border-radius:4px; padding:8px 13px; }
QPushButton:hover { background:#3c5875; } QPushButton:disabled { color:#6a7a8c; background:#243141; }
QPushButton[primary="true"] { background:#267b8e; border-color:#39a1aa; font-weight:600; }
QLineEdit,QPlainTextEdit,QTextBrowser,QComboBox,QSpinBox,QDoubleSpinBox { background:#141d2a; border:1px solid #36485e; border-radius:3px; padding:6px; selection-background-color:#337c96; }
QListWidget,QTreeWidget,QTableWidget { background:#182231; border:1px solid #344557; alternate-background-color:#202d3d; }
QHeaderView::section { background:#293a4f; padding:7px; border:0px; border-bottom:1px solid #40516a; }
QTabWidget::pane { border:1px solid #344557; } QTabBar::tab { background:#222f41; padding:10px 12px; }
QTabBar::tab:selected { background:#344a62; border-bottom:2px solid #57cbb7; }
QGroupBox { border:1px solid #34495e; border-radius:5px; margin-top:12px; padding:12px; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 5px; color:#88d2cf; }
QProgressBar { border:1px solid #3c536a; border-radius:4px; text-align:center; background:#111b28; }
QProgressBar::chunk { background:#2d9a9b; }
QSlider::groove:horizontal { height:5px; background:#405166; } QSlider::handle:horizontal { background:#6bc9c2; width:13px; margin:-5px 0; border-radius:6px; }
QScrollBar:vertical { background:#182231; width:12px; } QScrollBar::handle:vertical { background:#42566c; min-height:20px; }
QToolTip { color:#e8f1ff; background:#344861; border:1px solid #5a7a9c; padding:6px; }
"""


def button(label: str, callback: Any, primary: bool = False) -> QPushButton:
    control = QPushButton(label)
    control.setProperty("primary", primary)
    control.clicked.connect(callback)
    return control


def table(headers: list[str]) -> QTableWidget:
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.setAlternatingRowColors(True)
    widget.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    widget.horizontalHeader().setStretchLastSection(True)
    widget.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
    widget.verticalHeader().setVisible(False)
    return widget


def fill_table(widget: QTableWidget, rows: list[list[Any]]) -> None:
    widget.setSortingEnabled(False)
    widget.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for column, datum in enumerate(row):
            item = QTableWidgetItem(str(datum))
            item.setData(Qt.ItemDataRole.UserRole, datum)
            widget.setItem(row_index, column, item)
    widget.setSortingEnabled(True)


class OverrideCommand(QUndoCommand):
    def __init__(self, window: MainWindow, key: str, new: Any) -> None:
        super().__init__(f"Edit {key}")
        self.window, self.key, self.new = window, key, new
        self.old = window.override_value(key)

    def redo(self) -> None:
        self.window.service.apply_override(self.window.project, self.key, self.new)
        self.window.refresh_project()

    def undo(self) -> None:
        self.window.service.apply_override(self.window.project, self.key, self.old)
        self.window.refresh_project()


class MainWindow(QMainWindow):
    project_changed = Signal(object)

    def __init__(self, service: Any = None, preferences: Preferences | None = None) -> None:
        super().__init__()
        if service is None:
            from openmocap.application import ProjectService

            service = ProjectService()
        self.service = service
        self.preferences = preferences or Preferences()
        self.project: Any = None
        self.result: dict[str, Any] = {}
        self.job: PipelineJob | None = None
        self._last_job: Any = None
        self._time_index = 0
        self._selected_observation: int | None = None
        self.undo_stack = QUndoStack(self)
        self.setWindowTitle("OpenMocap VFX · Metric Motion Reconstruction")
        self.resize(1510, 960)
        self.setStyleSheet(STYLE)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self._build_home()
        self._build_workspace()
        self._build_menu()
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._advance)
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(1000)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
        self.statusBar().showMessage(
            "Measured geometry → persistent actor shape → animation. All project data remains local."
        )
        self._refresh_recent()
        sizes = self.preferences.data.get("workspace_sizes")
        if sizes:
            self.main_splitter.setSizes(sizes)
        for key, control in self.overlay_controls.items():
            control.setChecked(
                self.preferences.data.get("overlays", {}).get(key, control.isChecked())
            )
        self.viewer_mode.setCurrentText(self.preferences.data.get("viewer_mode", "3D Scene"))

    def _build_menu(self) -> None:
        groups = [
            (
                "Project",
                [
                    ("Home", self.show_home),
                    ("New Project…", self.new_project),
                    ("Open Project…", self.open_project_dialog),
                    ("Save Project", self.save_project),
                    ("Duplicate Project…", self.duplicate_project),
                    ("Project Settings…", self.project_settings),
                    ("Project Health / Doctor", self.show_doctor),
                ],
            ),
            (
                "Solve",
                [
                    ("Run Full Pipeline…", self.run_pipeline),
                    ("Run Selected Stage…", self.run_selected),
                    ("Cancel Job", self.cancel_job),
                ],
            ),
            (
                "Export",
                [
                    ("Export Animation…", self.export_dialog),
                    ("Open Output Folder", self.open_output),
                ],
            ),
            (
                "Help",
                [
                    ("User Manual", self.show_help),
                    ("Load Synthetic Tutorial…", self.create_example),
                    ("Environment Diagnostics", self.show_doctor),
                ],
            ),
        ]
        for title, entries in groups:
            menu = self.menuBar().addMenu(title)
            for label, callback in entries:
                action = QAction(label, self)
                action.triggered.connect(callback)
                menu.addAction(action)
        edit = self.menuBar().addMenu("Edit")
        undo = self.undo_stack.createUndoAction(self, "Undo")
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        redo = self.undo_stack.createRedoAction(self, "Redo")
        redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.undo_action, self.redo_action = undo, redo
        edit.addActions([undo, redo])

    def _build_home(self) -> None:
        self.home = QWidget()
        outer = QVBoxLayout(self.home)
        outer.setContentsMargins(65, 42, 65, 40)
        title = QLabel("OPENMOCAP  /  VFX")
        title.setStyleSheet("font-size:30px;font-weight:700;color:#78d8c9")
        outer.addWidget(title)
        subtitle = QLabel("Multi-camera capture. Measured geometry. Production visibility.")
        subtitle.setStyleSheet("font-size:17px;color:#9eafc6;margin-bottom:28px")
        outer.addWidget(subtitle)
        row = QHBoxLayout()
        outer.addLayout(row)
        actions = QGroupBox("Start a capture")
        action_layout = QVBoxLayout(actions)
        for label, callback in [
            ("＋  New Project", self.new_project),
            ("Open Project…", self.open_project_dialog),
            ("Synthetic Example / Tutorial", self.create_example),
            ("Project Health / Doctor", self.show_doctor),
            ("Documentation", self.show_help),
        ]:
            action_layout.addWidget(button(label, callback, label.startswith("＋")))
        action_layout.addStretch()
        row.addWidget(actions, 1)
        recent = QGroupBox("Recent projects")
        recent_layout = QVBoxLayout(recent)
        self.recent_list = QListWidget()
        self.recent_list.itemDoubleClicked.connect(
            lambda item: self.open_project(item.data(Qt.ItemDataRole.UserRole))
        )
        recent_layout.addWidget(self.recent_list)
        recent_layout.addWidget(
            QLabel("Double-click a project to reopen its configuration, QC and exports.")
        )
        row.addWidget(recent, 2)
        info = QLabel(
            "First run: use Project Health to verify runtime and optional backends. The synthetic tutorial works without footage or licensed SMPL assets. Real SMPL / SMPL-H / SMPL-X assets must be obtained separately."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#a2b5cd;padding:20px;background:#243246;border-radius:6px")
        outer.addWidget(info)
        outer.addStretch()
        self.stack.addWidget(self.home)

    def _build_workspace(self) -> None:
        self.workspace = QWidget()
        layout = QVBoxLayout(self.workspace)
        layout.setContentsMargins(10, 8, 10, 8)
        toolbar = QHBoxLayout()
        self.project_label = QLabel("No project")
        self.project_label.setStyleSheet("font-size:17px;font-weight:600;color:#83d7cc")
        toolbar.addWidget(self.project_label, 1)
        self.preset = QComboBox()
        self.preset.addItems(["Preview", "Standard", "High Quality", "Maximum / Final"])
        self.preset.setCurrentText("Standard")
        toolbar.addWidget(self.preset)
        self.run_button = button("Run Pipeline…", self.run_pipeline, True)
        toolbar.addWidget(self.run_button)
        toolbar.addWidget(button("Export…", self.export_dialog))
        layout.addLayout(toolbar)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        navigation = QWidget()
        nav_layout = QVBoxLayout(navigation)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.addWidget(QLabel("PIPELINE / EVIDENCE"))
        self.navigator = QTreeWidget()
        self.navigator.setHeaderLabels(["Stage", "State"])
        self.navigator.setColumnWidth(0, 155)
        self.navigator.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.navigator.itemClicked.connect(self._stage_clicked)
        self.stage_items: dict[str, QTreeWidgetItem] = {}
        for key, label, tab in STAGES:
            item = QTreeWidgetItem([label, "NOT RUN"])
            item.setData(0, Qt.ItemDataRole.UserRole, (key, tab))
            self.navigator.addTopLevelItem(item)
            self.stage_items[key] = item
        nav_layout.addWidget(self.navigator)
        nav_layout.addWidget(button("Run selected stage…", self.run_selected))
        nav_layout.addWidget(button("Home / projects", self.show_home))
        self.main_splitter.addWidget(navigation)
        self.tabs = QTabWidget()
        self.tab_index: dict[str, int] = {}
        self._camera_tab()
        self._calibration_tab()
        self._sync_tab()
        self._actor_tab()
        self._viewer_tab()
        self._qc_tab()
        self._export_tab()
        self._help_tab()
        self.main_splitter.addWidget(self.tabs)
        inspector = QWidget()
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.addWidget(QLabel("INSPECTOR / PROVENANCE"))
        self.inspector = QTextBrowser()
        self.inspector.setPlainText(
            "Select a camera, joint or QC measurement to inspect evidence, confidence and provenance."
        )
        inspector_layout.addWidget(self.inspector)
        self.override_key = QLineEdit()
        self.override_key.setPlaceholderText("Expert override key")
        self.override_value_edit = QLineEdit()
        self.override_value_edit.setPlaceholderText("Value as JSON, e.g. false")
        self.override_key.setToolTip(
            "Non-destructive service override: camera.CAM_01.enabled; actor_height; ground_y."
        )
        inspector_layout.addWidget(self.override_key)
        inspector_layout.addWidget(self.override_value_edit)
        inspector_layout.addWidget(button("Apply audited override", self.apply_expert_override))
        inspector_layout.addWidget(
            button("Disable measured observation", lambda: self.correct_observation(True))
        )
        inspector_layout.addWidget(
            button("Restore measured observation", lambda: self.correct_observation(False))
        )
        self.main_splitter.addWidget(inspector)
        self.main_splitter.setSizes([250, 1000, 250])
        layout.addWidget(self.main_splitter, 1)
        self._timeline(layout)
        self._jobs_panel(layout)
        self.stack.addWidget(self.workspace)

    def _tab(self, label: str) -> QVBoxLayout:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.tab_index[label] = self.tabs.addTab(widget, label)
        return layout

    def _camera_tab(self) -> None:
        layout = self._tab("Cameras")
        layout.addWidget(
            QLabel(
                "Manage arbitrary camera counts. Imported calibration and manual locks remain authoritative."
            )
        )
        row = QHBoxLayout()
        for label, callback in [
            ("Register media…", self.import_media),
            ("Import calibration…", self.import_calibration),
            ("Edit selected…", self.edit_camera),
            ("Enable / disable", self.toggle_camera),
        ]:
            row.addWidget(button(label, callback))
        row.addStretch()
        layout.addLayout(row)
        self.camera_table = table(
            [
                "Camera",
                "Source",
                "Resolution",
                "FPS",
                "Timestamps",
                "Calibration",
                "Offset s",
                "Drift ppm",
                "Median px",
                "Quality",
                "Enabled",
                "Locked",
            ]
        )
        self.camera_table.itemSelectionChanged.connect(self.inspect_camera)
        self.camera_table.itemDoubleClicked.connect(lambda _: self.edit_camera())
        layout.addWidget(self.camera_table)

    def _calibration_tab(self) -> None:
        layout = self._tab("Calibration")
        layout.addWidget(
            QLabel(
                "Import target / surveyed / SfM solutions. Inspect residuals, metric floor and camera locks."
            )
        )
        controls = QHBoxLayout()
        controls.addWidget(button("Import camera solution…", self.import_calibration))
        controls.addWidget(button("Edit / lock camera…", self.edit_camera))
        controls.addWidget(
            button("Run calibration stage…", lambda: self.run_pipeline(stage="calibrate"))
        )
        layout.addLayout(controls)
        self.calibration_scene = SceneView()
        layout.addWidget(self.calibration_scene, 1)
        floor_row = QHBoxLayout()
        floor_row.addWidget(QLabel("Surveyed ground Y (m)"))
        self.ground_control = number(-10000, 10000, 0)
        floor_row.addWidget(self.ground_control)
        floor_row.addWidget(
            button(
                "Apply floor", lambda: self.push_override("ground_y", self.ground_control.value())
            )
        )
        floor_row.addStretch()
        layout.addLayout(floor_row)
        self.calibration_summary = QLabel("No calibrated project loaded.")
        self.calibration_summary.setWordWrap(True)
        layout.addWidget(self.calibration_summary)

    def _sync_tab(self) -> None:
        layout = self._tab("Synchronization")
        notice = QLabel(
            "t_world = scale × t_camera + offset. Edit sub-frame offsets and drift. Locked timing is preserved."
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.sync_table = table(
            ["Camera", "Offset seconds", "Scale", "Drift ppm", "Locked", "Source"]
        )
        self.sync_table.itemDoubleClicked.connect(lambda _: self.edit_camera(sync=True))
        layout.addWidget(self.sync_table)
        layout.addWidget(button("Edit selected timing…", lambda: self.edit_camera(sync=True)))
        layout.addWidget(
            button("Run synchronization stage…", lambda: self.run_pipeline(stage="sync"))
        )
        help_text = QLabel(
            "Affine mappings and timestamp manifests support asynchronous media. Waveform / event display is not yet implemented; automatic estimation requires explicit synchronization evidence."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

    def _actor_tab(self) -> None:
        layout = self._tab("Actor")
        layout.addWidget(
            QLabel(
                "One persistent actor shape per take; pose and metric root motion vary over time."
            )
        )
        group = QGroupBox("Actor setup")
        form = QFormLayout(group)
        self.actor_name = QLineEdit("Actor 01")
        self.actor_model = QComboBox()
        self.actor_model.addItems(["smpl", "smplh", "smplx", "fixture"])
        self.actor_gender = QComboBox()
        self.actor_gender.addItems(["neutral", "male", "female"])
        self.actor_height = number(0, 3, 0)
        self.actor_height.setSpecialValueText("Unknown")
        self.actor_assets = QLineEdit(str(install_root() / "models" / "body" / "SMPL_NEUTRAL.npz"))
        for label, widget in [
            ("Name", self.actor_name),
            ("Body model", self.actor_model),
            ("Model option", self.actor_gender),
            ("Measured height (m)", self.actor_height),
            ("Licensed model NPZ file", browse_field(self.actor_assets)),
        ]:
            form.addRow(label, widget)
        form.addRow(button("Apply actor setup", self.apply_actor))
        self.actor_setup_tabs = QTabWidget()
        self.actor_setup_tabs.addTab(group, "Actor and model")
        layout.addWidget(self.actor_setup_tabs)
        inference = QGroupBox("Inference checkpoints · licensed user-supplied models")
        inference_form = QFormLayout(inference)
        self.pose_checkpoint = QLineEdit()
        self.pose_checkpoint.setPlaceholderText("Compatible pose ONNX checkpoint")
        self.detector_checkpoint = QLineEdit()
        self.detector_checkpoint.setPlaceholderText("Required only for ONNX detector")
        self.detector_backend = QComboBox()
        self.detector_backend.addItems(["hog", "onnx"])
        self.keypoint_schema = QComboBox()
        self.keypoint_schema.addItems(["SMPL24", "H36M17", "COCO17", "COCOWholeBody133"])
        self.keypoint_schema.setToolTip(
            "Select the checkpoint's actual joint order. Layout adaptation is explicit; missing joints are not invented as detections."
        )
        inference_form.addRow("Pose ONNX file", browse_field(self.pose_checkpoint))
        inference_form.addRow("Detector ONNX file", browse_field(self.detector_checkpoint))
        schema_row = QWidget()
        schema_layout = QHBoxLayout(schema_row)
        schema_layout.setContentsMargins(0, 0, 0, 0)
        schema_layout.addWidget(self.detector_backend)
        schema_layout.addWidget(self.keypoint_schema)
        schema_layout.addWidget(button("Apply inference setup", self.apply_inference))
        inference_form.addRow("Detector / joint order", schema_row)
        inference_form.addRow(
            button("Load installed whole-body preset…", self.import_inference_preset)
        )
        self.actor_setup_tabs.addTab(inference, "Inference checkpoints")
        self.actor_metrics = table(["Measurement", "Provided", "Fitted", "Confidence / residual"])
        layout.addWidget(self.actor_metrics)
        row = QHBoxLayout()
        row.addWidget(button("Fit persistent shape…", lambda: self.run_pipeline(stage="fit-shape")))
        row.addWidget(button("Solve body motion…", lambda: self.run_pipeline(stage="fit-motion")))
        layout.addLayout(row)

    def _viewer_tab(self) -> None:
        layout = self._tab("Viewer")
        controls = QHBoxLayout()
        self.viewer_mode = QComboBox()
        self.viewer_mode.addItems(
            ["3D Scene", "1 Camera", "2 Cameras", "4 Cameras", "9 Cameras", "16 Cameras"]
        )
        self.viewer_camera = QComboBox()
        controls.addWidget(self.viewer_mode)
        controls.addWidget(self.viewer_camera)
        controls.addWidget(button("Fit / frame", self.fit_view))
        controls.addWidget(button("Reset view", lambda: self.scene_view.reset_view()))
        self.ortho = QCheckBox("Ortho")
        self.ortho.toggled.connect(self._orthographic)
        controls.addWidget(self.ortho)
        controls.addStretch()
        layout.addLayout(controls)
        overlay_row = QHBoxLayout()
        self.overlay_controls: dict[str, QCheckBox] = {}
        for key, label, checked in [
            ("measured", "Measured 2D", True),
            ("reprojected", "Reprojected", True),
            ("vectors", "Error vectors", True),
            ("confidence", "Confidence", False),
            ("rejected", "Rejected", True),
        ]:
            control = QCheckBox(label)
            control.setChecked(checked)
            self.overlay_controls[key] = control
            overlay_row.addWidget(control)
        overlay_row.addStretch()
        layout.addLayout(overlay_row)
        legend = QLabel(
            '<span style="color:#52d9bc">● MEASURED</span> &nbsp; <span style="color:#f9b95f">● REPROJECTED</span> &nbsp; <span style="color:#ff667a">● REJECTED</span> &nbsp; <span style="color:#ad9bfa">● INFERRED</span>'
        )
        legend.setStyleSheet("color:#9eb4cc;font-size:10px")
        layout.addWidget(legend)
        self.viewer_stack = QStackedWidget()
        self.scene_view = SceneView()
        self.scene_view.selected.connect(self.inspect_joint)
        self.scene_view.camera_selected.connect(self.inspect_camera_id)
        self.viewer_stack.addWidget(self.scene_view)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_stack.addWidget(self.grid_container)
        self.plate_views: list[PlateView] = []
        layout.addWidget(self.viewer_stack, 1)
        self.pixel_label = QLabel(
            "World time drives each camera independently. Wheel: zoom. Drag: pan."
        )
        layout.addWidget(self.pixel_label)
        self.viewer_mode.currentTextChanged.connect(self.rebuild_view_grid)
        self.viewer_camera.currentIndexChanged.connect(self._refresh_viewers)
        for control in self.overlay_controls.values():
            control.toggled.connect(self._refresh_viewers)

    def _qc_tab(self) -> None:
        layout = self._tab("Quality Control")
        row = QHBoxLayout()
        self.qc_filter = QLineEdit()
        self.qc_filter.setPlaceholderText("Filter by camera, joint, source or warning…")
        self.qc_filter.textChanged.connect(self.filter_qc)
        row.addWidget(self.qc_filter, 1)
        row.addWidget(button("Refresh QC", self.load_result))
        row.addWidget(button("Open HTML Report", self.open_report))
        layout.addLayout(row)
        self.qc_table = table(
            ["Category", "Camera / joint", "Time", "Metric", "Value", "Evidence / action"]
        )
        self.qc_table.itemDoubleClicked.connect(self.navigate_qc)
        layout.addWidget(self.qc_table)
        self.qc_summary = QTextBrowser()
        self.qc_summary.setMaximumHeight(180)
        layout.addWidget(self.qc_summary)
        correction = QHBoxLayout()
        self.contact_joint = QComboBox()
        self.contact_joint.addItems(["left_heel", "left_toe", "right_heel", "right_toe"])
        correction.addWidget(self.contact_joint)
        correction.addWidget(button("Mark range as contact", self.mark_contact))
        correction.addWidget(
            button("Refine contacts…", lambda: self.run_pipeline(stage="contacts"))
        )
        correction.addWidget(
            button("Optional physics…", lambda: self.run_pipeline(stage="physics"))
        )
        layout.addLayout(correction)

    def _export_tab(self) -> None:
        layout = self._tab("Export")
        label = QLabel(
            "Deliver skeleton, root motion and animation. FBX adds a constant-topology weighted mesh through Blender. Validation appears only after real re-import checks succeed."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(button("Export Animation…", self.export_dialog, True))
        self.export_summary = QTextBrowser()
        self.export_summary.setPlainText("No export executed in this session.")
        layout.addWidget(self.export_summary)
        layout.addWidget(button("Open output folder", self.open_output))

    def _help_tab(self) -> None:
        layout = self._tab("Help")
        self.help_search = QLineEdit()
        self.help_search.setPlaceholderText("Search the installed user manual…")
        self.help_search.textChanged.connect(self.search_help)
        layout.addWidget(self.help_search)
        self.help_browser = QTextBrowser()
        self.help_browser.setOpenLinks(False)
        self.help_browser.anchorClicked.connect(self._help_link)
        layout.addWidget(self.help_browser)
        self.search_help("")

    def _timeline(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        self.play_button = button("▶", self.toggle_play)
        row.addWidget(button("|◀", lambda: self.set_time_index(0)))
        row.addWidget(button("◀", lambda: self.set_time_index(self._time_index - 1)))
        row.addWidget(self.play_button)
        row.addWidget(button("▶|", lambda: self.set_time_index(self._time_index + 1)))
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.valueChanged.connect(self.set_time_index)
        row.addWidget(self.timeline, 1)
        self.time_label = QLabel("world 0.000 s  |  frame 0")
        row.addWidget(self.time_label)
        self.range_start = number(0, 1e9, 0, 3)
        self.range_end = number(0, 1e9, 0, 3)
        self.range_end.setSpecialValueText("Full")
        row.addWidget(QLabel("Range"))
        row.addWidget(self.range_start)
        row.addWidget(self.range_end)
        layout.addLayout(row)
        self.contact_tracks = ContactTimeline()
        self.contact_tracks.time_selected.connect(self.seek_time)
        layout.addWidget(self.contact_tracks)

    def _jobs_panel(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        self.job_label = QLabel("Ready · CPU geometry / configured inference device")
        row.addWidget(self.job_label, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMaximumWidth(360)
        row.addWidget(self.progress)
        self.cancel_button = button("Cancel", self.cancel_job)
        self.cancel_button.setEnabled(False)
        row.addWidget(self.cancel_button)
        self.retry_button = button("Retry", self.retry_job)
        self.retry_button.setEnabled(False)
        row.addWidget(self.retry_button)
        layout.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(95)
        self.log.setMaximumBlockCount(1500)
        layout.addWidget(self.log)

    def _refresh_recent(self) -> None:
        self.recent_list.clear()
        for path in self.preferences.data.get("recent", []):
            item = QListWidgetItem(f"{Path(path).name}\n{path}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.recent_list.addItem(item)

    def _require_project(self) -> bool:
        if self.project is None:
            self.error(
                "Open a project first", "Use New Project, Open Project or Synthetic Example."
            )
            return False
        return True

    def error(self, title: str, details: str) -> None:
        self.log.appendPlainText(f"ERROR: {title}\n{details}")
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle(title)
        message.setText(title)
        message.setInformativeText(details.splitlines()[-1][:500] if details else title)
        message.setDetailedText(details)
        message.exec()

    def show_home(self) -> None:
        self.stack.setCurrentWidget(self.home)
        self._refresh_recent()

    def new_project(self) -> None:
        if not self._can_change_project():
            return
        if not self._can_edit():
            return
        wizard = ProjectWizard(self)
        if wizard.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            settings = wizard.settings()
            path, name, count = (
                settings.pop("path"),
                settings.pop("name"),
                settings.pop("camera_count"),
            )
            calibration_file = settings.pop("calibration_file")
            project = self.service.create_project(path, name, camera_count=count, **settings)
            if calibration_file:
                self.service.import_calibration(project, calibration_file)
            self.set_project(project)
        except Exception as exc:
            self.error("Project creation failed", str(exc))

    def open_project_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open capture project",
            str(install_root() / "projects"),
            "Project (*.yaml *.yml *.json);;All files (*)",
        )
        if filename:
            self.open_project(filename)

    def open_project(self, path: str | Path) -> None:
        if not self._can_change_project():
            return
        if not self._can_edit():
            return
        try:
            self.set_project(self.service.open_project(path))
        except Exception as exc:
            self.error("Cannot open project", str(exc))

    def set_project(self, project: Any) -> None:
        self.play_timer.stop()
        self.project = project
        self.undo_stack.clear()
        self.result = {}
        self._time_index = 0
        self._selected_observation = None
        self.preferences.remember(str(project_path(project)))
        self.stack.setCurrentWidget(self.workspace)
        self.refresh_project()
        self.load_result()
        self.project_changed.emit(project)

    def create_example(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select parent folder for tutorial", str(install_root() / "projects")
        )
        if path:
            self.create_example_at(Path(path) / "synthetic-tutorial")

    def create_example_at(self, path: Path) -> None:
        self.start_job(
            "Generate deterministic tutorial",
            lambda callback, cancel: self.service.create_example(path),
            self.set_project,
        )

    def save_project(self) -> None:
        if self._require_project():
            self.service.save_project(self.project)
            self.statusBar().showMessage("Project saved. Raw observations remain unchanged.")

    def duplicate_project(self) -> None:
        if not self._require_project() or not self._can_change_project():
            return
        target = QFileDialog.getExistingDirectory(self, "Choose destination parent")
        if not target:
            return
        source = project_path(self.project)
        destination = Path(target) / (source.name + "-copy")
        try:
            if destination.exists():
                raise FileExistsError(f"Destination already exists: {destination}")
            self.service.save_project(self.project)
            copied = self.service.duplicate_project(self.project, destination)
            self.set_project(copied)
        except Exception as exc:
            self.error("Duplicate project failed", str(exc))

    def project_settings(self) -> None:
        if not self._require_project():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Project settings")
        form = QFormLayout(dialog)
        name = QLineEdit(str(value(self.project, "name", "")))
        fps = number(1, 240, float(value(self.project, "output_fps", 24)), 3)
        form.addRow("Project", name)
        form.addRow("Output FPS", fps)
        form.addRow("World", QLabel("Metres · +Y up · explicit calibrated floor"))
        form.addRow("Path", QLabel(str(project_path(self.project))))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.push_override("name", name.text())
            self.push_override("output_fps", fps.value())

    def refresh_project(self) -> None:
        if self.project is None:
            return
        self.project_label.setText(str(value(self.project, "name", "Capture")))
        self.setWindowTitle(f"{value(self.project, 'name', 'Capture')} — OpenMocap VFX")
        camera_list = cameras(self.project)
        report = self.result.get("report", {})
        reprojection = report.get(
            "per_camera", report.get("reprojection", {}).get("per_camera", {})
        )
        rows, sync_rows = [], []
        selected = self.viewer_camera.currentText()
        self.viewer_camera.blockSignals(True)
        self.viewer_camera.clear()
        for camera in camera_list:
            identifier = camera_id(camera)
            self.viewer_camera.addItem(identifier)
            intrinsic = value(camera, "intrinsics", {})
            mapping = value(camera, "time_mapping", {})
            offset, scale = float(value(mapping, "offset", 0)), float(value(mapping, "scale", 1))
            quality = value(camera, "quality", {})
            weight = value(
                quality,
                "confidence",
                value(quality, "weight", quality if isinstance(quality, (int, float)) else 1),
            )
            camera_report = (
                reprojection.get(identifier, {}) if isinstance(reprojection, dict) else {}
            )
            rows.append(
                [
                    identifier,
                    value(camera, "source_path", value(camera, "media_source", "—")) or "—",
                    f"{value(intrinsic, 'width', '?')} × {value(intrinsic, 'height', '?')}",
                    value(camera, "fps", "—"),
                    value(camera, "timestamp_status", "Imported mapping"),
                    value(camera, "source", "manual"),
                    f"{offset:+.6f}",
                    f"{(scale - 1) * 1e6:+.2f}",
                    value(camera_report, "median_px", value(camera_report, "median", "—")),
                    weight,
                    "Yes" if value(camera, "enabled", True) else "No",
                    "LOCKED" if value(camera, "locked", True) else "Editable",
                ]
            )
            sync_rows.append(
                [
                    identifier,
                    f"{offset:+.6f}",
                    f"{scale:.8f}",
                    f"{(scale - 1) * 1e6:+.2f}",
                    "LOCKED" if value(mapping, "locked", False) else "Editable",
                    value(mapping, "source", "manual"),
                ]
            )
        if selected:
            self.viewer_camera.setCurrentText(selected)
        elif self.preferences.data.get("selected_camera"):
            self.viewer_camera.setCurrentText(self.preferences.data["selected_camera"])
        self.viewer_camera.blockSignals(False)
        fill_table(self.camera_table, rows)
        fill_table(self.sync_table, sync_rows)
        self.ground_control.setValue(float(value(self.project, "ground_y", 0)))
        self.actor_name.setText(str(value(self.project, "actor_name", "Actor 01")))
        self.actor_model.setCurrentText(str(value(self.project, "body_model", "smpl")))
        self.actor_height.setValue(float(value(self.project, "actor_height", 0) or 0))
        self.actor_assets.setText(str(value(self.project, "model_path", "") or ""))
        self.actor_assets.setPlaceholderText(
            "Not required for fixture; choose licensed NPZ for SMPL family"
        )
        perception = value(self.project, "perception", {})
        self.pose_checkpoint.setText(str(value(perception, "pose_model", "") or ""))
        self.detector_checkpoint.setText(str(value(perception, "detector_model", "") or ""))
        self.detector_backend.setCurrentText(str(value(perception, "detector_backend", "hog")))
        schema = str(value(perception, "schema", "SMPL24"))
        for index in range(self.keypoint_schema.count()):
            if self.keypoint_schema.itemText(index).casefold() == schema.casefold():
                self.keypoint_schema.setCurrentIndex(index)
                break
        stages = value(self.project, "stages", {})
        for key, item in self.stage_items.items():
            state = value(stages, key, "NOT RUN")
            state = str(value(state, "status", state)).upper().replace("_", " ")
            item.setText(1, state)
            color = {
                "COMPLETE": "#65d9bd",
                "RUNNING": "#ffd17f",
                "STALE": "#efa95e",
                "FAILED": "#ff6a7c",
                "WARNING": "#e9af65",
            }.get(state, "#8397af")
            item.setForeground(1, QColor(color))
        stale = any(
            str(value(state, "status", state)).upper() == "STALE" for state in stages.values()
        )
        self.scene_view.stale = stale
        self.calibration_scene.stale = stale
        self.scene_view.floor_y = self.ground_control.value()
        self.calibration_scene.floor_y = self.ground_control.value()
        for view in self.plate_views:
            view.stale = stale
        locked_count = sum(bool(value(camera, "locked", True)) for camera in camera_list)
        self.calibration_summary.setText(
            f"{len(camera_list)} cameras · {locked_count} authoritative locks · metres / +Y up. Camera transforms and provenance are shown in Cameras. Target correspondence editing is not yet available."
        )
        self._refresh_viewers()

    def _selected_camera(self, sync: bool = False) -> Any:
        widget = self.sync_table if sync else self.camera_table
        row = widget.currentRow()
        identifier = (
            widget.item(row, 0).text()
            if row >= 0 and widget.item(row, 0)
            else self.viewer_camera.currentText()
        )
        return next(
            (camera for camera in cameras(self.project) if camera_id(camera) == identifier), None
        )

    def inspect_camera(self) -> None:
        if self.project is not None:
            camera = self._selected_camera()
            if camera is not None:
                self.inspector.setPlainText(json.dumps(camera, indent=2, default=str))

    def inspect_camera_id(self, identifier: str) -> None:
        camera = next(
            (camera for camera in cameras(self.project) if camera_id(camera) == identifier), None
        )
        if camera is not None:
            self.inspector.setPlainText(json.dumps(camera, indent=2, default=str))
            self.viewer_camera.setCurrentText(identifier)

    def edit_camera(self, sync: bool = False) -> None:
        if not self._require_project():
            return
        camera = self._selected_camera(sync)
        if camera is None:
            return
        editor = CameraEditor(camera, self)
        if editor.exec() == QDialog.DialogCode.Accepted:
            try:
                changes = editor.overrides()
                self.undo_stack.beginMacro(f"Edit camera {camera_id(camera)}")
                for key, datum in changes.items():
                    self.push_override(key, datum)
                self.undo_stack.endMacro()
            except Exception as exc:
                self.error("Invalid camera override", str(exc))

    def toggle_camera(self) -> None:
        if self._require_project():
            camera = self._selected_camera()
            if camera:
                self.push_override(
                    f"camera.{camera_id(camera)}.enabled", not bool(value(camera, "enabled", True))
                )

    def import_media(self) -> None:
        if not self._require_project() or not self._can_edit():
            return
        camera = self._selected_camera()
        if camera is None:
            return
        choice = QMessageBox(self)
        choice.setWindowTitle(f"Register source for {camera_id(camera)}")
        choice.setText("Choose a video / single image or an image-sequence folder.")
        file_button = choice.addButton("Video / Image File", QMessageBox.ButtonRole.AcceptRole)
        sequence_button = choice.addButton(
            "Image Sequence Folder", QMessageBox.ButtonRole.AcceptRole
        )
        choice.addButton(QMessageBox.StandardButton.Cancel)
        choice.exec()
        if choice.clickedButton() is sequence_button:
            filename = QFileDialog.getExistingDirectory(self, "Choose image sequence")
        elif choice.clickedButton() is file_button:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                f"Register source for {camera_id(camera)}",
                "",
                "Capture (*.mp4 *.mov *.mkv *.avi *.png *.jpg);;All files (*)",
            )
        else:
            return
        if filename:
            try:
                self.service.import_camera(self.project, camera_id(camera), filename)
                self.refresh_project()
            except Exception as exc:
                self.error("Media registration failed", str(exc))

    def import_calibration(self) -> None:
        if not self._require_project() or not self._can_edit():
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import calibrated camera solution",
            "",
            "Calibration (*.yaml *.yml *.json);;All files (*)",
        )
        if filename:
            try:
                self.service.import_calibration(self.project, filename)
                self.refresh_project()
            except Exception as exc:
                self.error("Calibration import failed", str(exc))

    def override_value(self, key: str) -> Any:
        existing = value(self.project, "overrides", {})
        if key in existing:
            return copy.deepcopy(existing[key])
        parts = key.split(".")
        record = self.project
        if parts[0] == "camera" and len(parts) > 2:
            record = next((cam for cam in cameras(self.project) if camera_id(cam) == parts[1]), {})
            parts = parts[2:]
        for part in parts:
            record = value(record, part)
        return copy.deepcopy(record)

    def push_override(self, key: str, datum: Any) -> None:
        if self.job is not None and self.job.isRunning():
            self.error(
                "Processing is active",
                "Cancel or finish the active job before editing capture parameters.",
            )
            return
        if self._require_project() and self._can_edit():
            self.undo_stack.push(OverrideCommand(self, key, datum))
            self.log.appendPlainText(
                f"Manual override: {key} = {datum!r}; dependent results invalidated."
            )

    def _can_edit(self) -> bool:
        if self.job is not None and self.job.isRunning():
            self.error(
                "Processing is active",
                "Wait for the current job or cancel at a safe checkpoint before editing project inputs.",
            )
            return False
        return True

    def apply_expert_override(self) -> None:
        try:
            self.push_override(
                self.override_key.text().strip(), json.loads(self.override_value_edit.text())
            )
        except Exception as exc:
            self.error("Invalid override", str(exc))

    def apply_actor(self) -> None:
        self.undo_stack.beginMacro("Actor setup")
        changes = {
            "actor_name": self.actor_name.text(),
            "body_model": self.actor_model.currentText(),
            "actor_height": self.actor_height.value() or None,
            "model_path": self.actor_assets.text(),
            "gender": self.actor_gender.currentText(),
        }
        for key, datum in changes.items():
            self.push_override(key, datum)
        self.undo_stack.endMacro()

    def apply_inference(self) -> None:
        from openmocap.pose2d.schemas import COCO_BODY_17, COCO_WHOLEBODY_133
        from openmocap.synthetic import JOINT_NAMES

        schemas = {
            "SMPL24": list(JOINT_NAMES),
            "H36M17": [
                "pelvis",
                "right_hip",
                "right_knee",
                "right_ankle",
                "left_hip",
                "left_knee",
                "left_ankle",
                "spine1",
                "spine3",
                "neck",
                "head",
                "left_shoulder",
                "left_elbow",
                "left_wrist",
                "right_shoulder",
                "right_elbow",
                "right_wrist",
            ],
            "COCO17": list(COCO_BODY_17),
            "COCOWholeBody133": list(COCO_WHOLEBODY_133),
        }
        changes = {
            "perception.pose_model": self.pose_checkpoint.text() or None,
            "perception.detector_model": self.detector_checkpoint.text() or None,
            "perception.detector_backend": self.detector_backend.currentText(),
            "perception.schema": self.keypoint_schema.currentText(),
            "perception.joint_names": schemas[self.keypoint_schema.currentText()],
        }
        self.undo_stack.beginMacro("Inference checkpoint setup")
        for key, datum in changes.items():
            self.push_override(key, datum)
        self.undo_stack.endMacro()

    def _stage_clicked(self, item: QTreeWidgetItem) -> None:
        key, tab = item.data(0, Qt.ItemDataRole.UserRole)
        self.tabs.setCurrentIndex(self.tab_index[tab])
        self.inspector.setPlainText(
            f"Stage: {key}\nState: {item.text(1)}\n\nRaw observations and imported camera solutions are preserved. Changes create overrides and mark dependent results STALE. Unavailable neural stages or licensed assets report their precise requirement."
        )

    def import_inference_preset(self) -> None:
        import yaml

        if not self._require_project() or not self._can_edit():
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load installed whole-body inference preset",
            str(Path(__file__).resolve().parents[3] / "configs"),
            "Inference preset (*.yaml *.yml);;All files (*)",
        )
        if filename:
            try:
                self.load_inference_preset(Path(filename))
            except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
                self.error("Inference preset could not be loaded", str(error))

    def load_inference_preset(self, source: str | Path) -> None:
        """Import inference settings without touching geometry or body metadata."""
        import yaml

        if not self._require_project() or not self._can_edit():
            return
        source = Path(source).resolve()
        payload = yaml.safe_load(source.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Inference preset must contain a YAML mapping")
        if payload.get("schema_version", 1) != 1:
            raise ValueError("Unsupported inference preset schema version")
        if "perception" in payload:
            unexpected = set(payload) - {
                "perception",
                "schema_version",
                "name",
                "description",
                "license_notes",
                "preset",
            }
            if unexpected:
                raise ValueError(
                    "Inference preset may not modify project parameters: "
                    + ", ".join(sorted(unexpected))
                )
            config = copy.deepcopy(payload["perception"])
        else:
            config = copy.deepcopy(payload)
        if not isinstance(config, dict) or not config:
            raise ValueError("Preset perception settings must be a non-empty mapping")
        allowed = {
            "pose_model",
            "detector_model",
            "segmentation_model",
            "joint_names",
            "schema",
            "detector_backend",
            "pose",
            "detector",
            "segmentation",
            "providers",
            "frame_stride",
            "person_rois",
            "association",
            "derive_midpoints",
        }
        unexpected = set(config) - allowed
        if unexpected:
            raise ValueError("Unsupported inference settings: " + ", ".join(sorted(unexpected)))
        for key in ("pose", "detector", "segmentation", "person_rois", "association"):
            if key in config and not isinstance(config[key], dict):
                raise ValueError(f"Preset {key} must be a mapping")
        if "joint_names" in config:
            names = config["joint_names"]
            if (
                not isinstance(names, list)
                or not names
                or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError(
                    "Preset joint_names must contain distinct non-empty landmark names"
                )
        if "providers" in config and (
            not isinstance(config["providers"], list)
            or not config["providers"]
            or any(not isinstance(provider, str) for provider in config["providers"])
        ):
            raise ValueError(
                "Preset providers must be a non-empty list of execution-provider names"
            )
        if config.get("detector_backend", "onnx") not in {"hog", "onnx"}:
            raise ValueError("Preset detector_backend must be hog or onnx")
        if "frame_stride" in config and (
            not isinstance(config["frame_stride"], int)
            or isinstance(config["frame_stride"], bool)
            or config["frame_stride"] < 1
        ):
            raise ValueError("Preset frame_stride must be a positive integer")
        for key in ("pose_model", "detector_model", "segmentation_model"):
            if config.get(key) is None:
                continue
            if not isinstance(config[key], str):
                raise ValueError(f"Preset {key} must be a model file path")
            path_string = config[key]
            for placeholder in ("${INSTALL_ROOT}", "${OPENMOCAP_INSTALL_ROOT}", "<INSTALL_ROOT>"):
                path_string = path_string.replace(placeholder, str(install_root()))
            if "$" in path_string:
                raise ValueError(f"Unresolved model path in {key}: {path_string}")
            model = Path(path_string)
            if not model.is_absolute():
                model = source.parent / model
            if not model.is_file():
                raise FileNotFoundError(
                    f"Installed preset requires missing {key}: {model}. Obtain the checkpoint through its documented installer."
                )
            config[key] = str(model.resolve())
        effective = copy.deepcopy(value(self.project, "perception", {}))
        for key, datum in config.items():
            if isinstance(datum, dict) and isinstance(effective.get(key), dict):
                effective[key] = {**effective[key], **datum}
            else:
                effective[key] = datum
        self.push_override("perception", effective)
        self.inspector.setPlainText(
            f"Loaded inference preset: {source.name}\n"
            f"Landmarks: {len(effective.get('joint_names', []))}\n"
            f"Pose checkpoint: {effective.get('pose_model', 'not configured')}\n"
            f"Detector: {effective.get('detector_backend', 'checkpoint-defined')}\n\n"
            "Camera calibration, timing, world scale and body-model selection are preserved. "
            "Inference results retain model hashes and confidence. Edit → Undo restores previous inference settings."
        )

    def run_selected(self) -> None:
        item = self.navigator.currentItem()
        self.run_pipeline(stage=item.data(0, Qt.ItemDataRole.UserRole)[0] if item else None)

    def run_pipeline(
        self, checked: bool = False, stage: str | None = None, confirm: bool = True
    ) -> None:
        if not self._require_project() or not self._can_edit():
            return
        selected = [camera_id(cam) for cam in cameras(self.project) if value(cam, "enabled", True)]
        stages = value(self.project, "stages", {})
        cached = [
            key
            for key, state in stages.items()
            if str(value(state, "status", state)).upper() == "COMPLETE"
        ]
        if confirm:
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Review solve")
            dialog.setText(f"Run {stage or 'full pipeline'} · {self.preset.currentText()}")
            dialog.setInformativeText(
                f"Selected cameras: {len(selected)} ({', '.join(selected)})\nOutput: {value(self.project, 'output_fps', 24)} FPS\nWorld range: {self.range_start.value():.3f}s to {self.range_end.value() or 'full take'}\nCompute: CPU geometry / configured inference backend\nComplete stages: {', '.join(cached) or 'none'}\nContent-hash checkpoints determine reuse. Missing backends fail explicitly."
            )
            dialog.setStandardButtons(
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
            )
            if dialog.exec() != QMessageBox.StandardButton.Ok:
                return
        if value(self.project, "solve_preset") != self.preset.currentText():
            self.service.apply_override(self.project, "solve_preset", self.preset.currentText())
        if self.range_end.value():
            self.service.apply_override(
                self.project, "solve_range", [self.range_start.value(), self.range_end.value()]
            )
        self.start_job(
            stage or "Full pipeline",
            lambda callback, cancel: self.service.run(
                self.project, callback=callback, cancel_event=cancel, stage=stage
            ),
            self._solve_complete,
        )

    def start_job(self, name: str, task: Any, on_success: Any = None) -> None:
        if self.job is not None and self.job.isRunning():
            self.error("A job is already running", "Cancel the current job or wait for completion.")
            return
        self._last_job = (name, task, on_success)
        self.job = PipelineJob(task, self)
        self.job.progress.connect(self._job_progress)
        self.job.succeeded.connect(lambda result: self._job_succeeded(name, result, on_success))
        self.job.failed.connect(self._job_failed)
        self.job.finished.connect(self._job_finished)
        self.cancel_button.setEnabled(True)
        self.retry_button.setEnabled(False)
        self.run_button.setEnabled(False)
        for tab in ["Cameras", "Calibration", "Synchronization", "Actor"]:
            self.tabs.setTabEnabled(self.tab_index[tab], False)
        self.override_key.setEnabled(False)
        self.override_value_edit.setEnabled(False)
        self.undo_action.setEnabled(False)
        self.redo_action.setEnabled(False)
        self.job_label.setText(name)
        self.progress.setValue(0)
        self.log.appendPlainText(f"Started: {name}")
        self.elapsed_timer.start()
        self.job.start()

    def _job_progress(self, update: Any) -> None:
        if isinstance(update, dict):
            stage = update.get("stage", "Processing")
            progress = float(update.get("progress", update.get("fraction", 0)))
            self.progress.setValue(int(progress * 100 if progress <= 1 else progress))
            message = str(update.get("message", stage))
            self.job_label.setText(message)
            if stage in self.stage_items:
                self.stage_items[stage].setText(1, str(update.get("status", "RUNNING")).upper())
            self.log.appendPlainText(message)
        else:
            self.log.appendPlainText(str(update))

    def _job_succeeded(self, name: str, result: Any, callback: Any) -> None:
        canceled = bool(self.job and self.job.cancel_event.is_set())
        self.job_label.setText(f"{'Canceled safely' if canceled else 'Complete'} · {name}")
        if not canceled:
            self.progress.setValue(100)
        self.log.appendPlainText(f"{'Canceled' if canceled else 'Completed'}: {name}")
        if callback:
            callback(result)

    def _job_failed(self, details: str) -> None:
        if self.job is not None and self.job.cancel_event.is_set():
            self.job_label.setText("Canceled safely · completed checkpoints retained")
            self.log.appendPlainText("Cancellation completed at a safe checkpoint.")
            self.refresh_project()
            return
        self.job_label.setText("FAILED · inspect detailed log / troubleshooting")
        self.log.appendPlainText(details)
        self.retry_button.setEnabled(True)
        self.refresh_project()
        self.error("Processing failed", details)

    def _job_finished(self) -> None:
        self.cancel_button.setEnabled(False)
        self.run_button.setEnabled(True)
        self.elapsed_timer.stop()
        for tab in ["Cameras", "Calibration", "Synchronization", "Actor"]:
            self.tabs.setTabEnabled(self.tab_index[tab], True)
        self.override_key.setEnabled(True)
        self.override_value_edit.setEnabled(True)
        self.undo_action.setEnabled(self.undo_stack.canUndo())
        self.redo_action.setEnabled(self.undo_stack.canRedo())

    def _can_change_project(self) -> bool:
        if self.job is not None and self.job.isRunning():
            self.error(
                "Processing is active", "Cancel or finish the current job before changing projects."
            )
            return False
        return True

    def _update_elapsed(self) -> None:
        if self.job is not None and self.job.started_at:
            import time

            self.statusBar().showMessage(
                f"Job elapsed {time.monotonic() - self.job.started_at:.0f}s · cancellation is checked at safe stage boundaries"
            )

    def cancel_job(self) -> None:
        if self.job is not None and self.job.isRunning():
            self.job.cancel()
            self.job_label.setText("Cancellation requested · waiting for safe checkpoint")

    def retry_job(self) -> None:
        if self._last_job:
            self.start_job(*self._last_job)

    def _solve_complete(self, result: Any) -> None:
        self.load_result()
        self.tabs.setCurrentIndex(self.tab_index["Viewer"])

    def load_result(self) -> None:
        if self.project is None:
            return
        try:
            self.result = self.service.load_result(self.project) or {}
        except FileNotFoundError:
            self.result = {}
        except Exception as exc:
            self.log.appendPlainText(f"Result unavailable: {exc}")
            self.result = {}
        self._index_observations()
        times = np.asarray(self.result.get("times", []))
        self.timeline.setRange(0, max(0, len(times) - 1))
        self._populate_qc()
        self.refresh_project()
        self.set_time_index(min(self._time_index, max(0, len(times) - 1)))

    def _index_observations(self) -> None:
        """One camera/time index shared by all tiled viewers."""
        grouped: dict[str, list[Any]] = {}
        self.result["_ui_raw_lookup"] = {
            id(obs): index for index, obs in enumerate(self.result.get("observations", []))
        }
        for observation in self.result.get("observations", []):
            grouped.setdefault(str(value(observation, "camera_id", "")), []).append(observation)
        index = {}
        for camera, observations in grouped.items():
            timestamps = np.array(
                [
                    float(value(obs, "world_timestamp", value(obs, "world_time", 0)))
                    for obs in observations
                ]
            )
            order = np.argsort(timestamps, kind="stable")
            index[camera] = (timestamps[order], [observations[i] for i in order])
        self.result["_ui_observation_index"] = index

    def _populate_qc(self) -> None:
        report = self.result.get("report", {})
        rows: list[list[Any]] = []
        reprojection = report.get(
            "per_camera", report.get("reprojection", {}).get("per_camera", {})
        )
        if isinstance(reprojection, dict):
            for camera, metrics in reprojection.items():
                if isinstance(metrics, dict):
                    for key, datum in metrics.items():
                        if isinstance(datum, (int, float, str)):
                            rows.append(
                                [
                                    "Camera",
                                    camera,
                                    "—",
                                    key,
                                    f"{datum:.4f}" if isinstance(datum, float) else datum,
                                    "Inspect calibration / timing / detections",
                                ]
                            )
        per_joint = report.get("per_joint", report.get("reprojection", {}).get("per_joint", {}))
        if isinstance(per_joint, dict):
            for joint, metrics in per_joint.items():
                for key, datum in (
                    metrics.items() if isinstance(metrics, dict) else [("residual", metrics)]
                ):
                    rows.append(["Joint", joint, "—", key, datum, "Inspect contributing views"])
        for warning in report.get("warnings", report.get("solver_warnings", [])):
            rows.append(
                ["Warning", "—", "—", "quality", str(warning), "Inspect evidence before export"]
            )
        for measurement in report.get("measurements", report.get("samples", []))[:5000]:
            rows.append(
                [
                    "Observation",
                    value(measurement, "camera_id", "—"),
                    value(measurement, "time", value(measurement, "world_time", "—")),
                    value(measurement, "joint_id", "residual"),
                    value(measurement, "error_px", "—"),
                    f"{'REJECTED' if value(measurement, 'rejected', False) else 'MEASURED'} · double-click to inspect",
                ]
            )
        fill_table(self.qc_table, rows)
        statistics = report.get("reprojection", {})
        if report:
            self.qc_summary.setPlainText(
                f"{report.get('camera_count', len(cameras(self.project)))} cameras · "
                f"{report.get('used_observations', statistics.get('count', 0))} used observations · "
                f"{report.get('rejected_observations', 0)} rejected\n"
                f"Original-plate reprojection: mean {statistics.get('mean_px', '—')} px · "
                f"median {statistics.get('median_px', '—')} px · 95th percentile {statistics.get('p95_px', '—')} px\n"
                f"Missing joint samples: {report.get('missing_joint_sample_percentage', '—')}%\n"
                "Double-click an observation to inspect its camera, time and joint. Filter REJECTED to isolate geometric outliers. "
                "Open HTML Report for all numerical terms, camera locks and reproducibility metadata.\n"
                + "\n".join(str(warning) for warning in report.get("warnings", [])[:5])
            )
        else:
            self.qc_summary.setPlainText(
                "No solved QC available. Run calibrated reconstruction to calculate reprojection residuals, rejected views, confidence and contacts."
            )
        shape = self.result.get("shape", report.get("shape", {}))
        measurements = value(shape, "measurements", {})
        animation = self.result.get("animation")
        if not measurements and animation is not None:
            names = list(animation.names)
            pairs = {
                "Femur left (m)": ("left_hip", "left_knee"),
                "Tibia left (m)": ("left_knee", "left_ankle"),
                "Upper arm left (m)": ("left_shoulder", "left_elbow"),
                "Forearm left (m)": ("left_elbow", "left_wrist"),
                "Shoulder width (m)": ("left_shoulder", "right_shoulder"),
                "Hip width (m)": ("left_hip", "right_hip"),
                "Torso length (m)": ("pelvis", "neck"),
            }
            measurements = {
                label: float(
                    np.linalg.norm(
                        animation.rest_joints[names.index(a)]
                        - animation.rest_joints[names.index(b)]
                    )
                )
                for label, (a, b) in pairs.items()
                if a in names and b in names
            }
            if len(animation.vertices):
                measurements["Rest mesh height (m)"] = float(np.ptp(animation.vertices[:, 1]))
        if isinstance(measurements, dict):
            fill_table(
                self.actor_metrics,
                [
                    [
                        key,
                        value(self.project, "actor_height", "—")
                        if key == "Rest mesh height (m)"
                        else "—",
                        f"{datum:.4f}" if isinstance(datum, (float, int)) else datum,
                        f"Persistent · {value(shape, 'mean_joint_residual_m', value(shape, 'residual', '—'))} m residual",
                    ]
                    for key, datum in measurements.items()
                ],
            )
        self.contact_tracks.set_context(self.result, self.current_time())

    def filter_qc(self, text: str) -> None:
        text = text.casefold()
        for row in range(self.qc_table.rowCount()):
            match = any(
                text
                in (
                    self.qc_table.item(row, col).text().casefold()
                    if self.qc_table.item(row, col)
                    else ""
                )
                for col in range(self.qc_table.columnCount())
            )
            self.qc_table.setRowHidden(row, not match)

    def navigate_qc(self, item: QTableWidgetItem) -> None:
        row = item.row()
        self.viewer_camera.setCurrentText(self.qc_table.item(row, 1).text())
        try:
            times = np.asarray(self.result.get("times", []))
            if len(times):
                self.set_time_index(
                    int(np.argmin(abs(times - float(self.qc_table.item(row, 2).text()))))
                )
        except ValueError:
            pass
        self.inspector.setPlainText(
            "\n".join(
                self.qc_table.item(row, col).text() for col in range(self.qc_table.columnCount())
            )
        )
        self.viewer_mode.setCurrentText("1 Camera")
        self.tabs.setCurrentIndex(self.tab_index["Viewer"])
        joint = self.qc_table.item(row, 3).text()
        camera = self.qc_table.item(row, 1).text()
        candidates = [
            (i, obs)
            for i, obs in enumerate(self.result.get("observations", []))
            if value(obs, "camera_id") == camera and str(value(obs, "joint_id")) == joint
        ]
        if candidates:
            selected = min(
                candidates,
                key=lambda item_: abs(
                    float(value(item_[1], "world_timestamp", 0)) - self.current_time()
                ),
            )
            self.inspect_raw_observation(selected[0])

    def inspect_joint(self, joint: str) -> None:
        names = list(self.result.get("joint_names", []))
        try:
            identifier = names.index(joint) if joint in names else int(joint)
            points = np.asarray(self.result.get("joints", []))
            confidence = np.asarray(self.result.get("confidence", []))
            self.inspector.setPlainText(
                f"Joint: {joint}\nWorld time: {self.current_time():.4f}s\nXYZ metres: {points[self._time_index, identifier]}\nConfidence: {confidence[self._time_index, identifier] if confidence.size else 'unknown'}\nSource: multiview / continuous trajectory\n\nStrong geometry dominates learned priors and optional physics."
            )
        except (ValueError, IndexError):
            self.inspector.setPlainText(f"Joint {joint} · no solved diagnostics")

    def inspect_raw_observation(self, identifier: int) -> None:
        self._selected_observation = identifier
        observation = self.result["observations"][identifier]
        self.inspector.setPlainText(
            f"MEASURED · observation {identifier}\n"
            + json.dumps(observation, indent=2, default=str)
            + "\n\nDisable / Restore stores a reversible user override. Original detector data remains unchanged."
        )

    def correct_observation(self, disabled: bool) -> None:
        if self._selected_observation is None:
            self.error(
                "Select a measured observation",
                "Click a green keypoint in a plate viewer, or double-click an observation in QC.",
            )
            return
        self.push_override(f"observation.{self._selected_observation}.disabled", disabled)

    def mark_contact(self) -> None:
        start, end = self.range_start.value(), self.range_end.value()
        if end <= start:
            self.error(
                "Select a contact interval", "Set range start and end before marking contact."
            )
            return
        self.push_override(
            f"contacts.manual.{self.contact_joint.currentText()}",
            {"start": start, "end": end, "source": "manual", "locked": True},
        )

    def rebuild_view_grid(self) -> None:
        if self.viewer_mode.currentText() == "3D Scene":
            self.viewer_stack.setCurrentWidget(self.scene_view)
            self._refresh_viewers()
            return
        self.viewer_stack.setCurrentWidget(self.grid_container)
        while self.grid_layout.count():
            widget = self.grid_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        self.plate_views = []
        count = int(self.viewer_mode.currentText().split()[0])
        columns = int(np.ceil(np.sqrt(count)))
        for index in range(count):
            view = PlateView()
            view.pixel_inspected.connect(
                lambda x, y: self.pixel_label.setText(
                    f"Pixel x={x:.1f}, y={y:.1f} · measured green / reprojected amber / rejected red / inferred violet"
                )
            )
            view.observation_selected.connect(self.inspect_joint)
            view.raw_observation_selected.connect(self.inspect_raw_observation)
            self.grid_layout.addWidget(view, index // columns, index % columns)
            self.plate_views.append(view)
        self._refresh_viewers()
        QTimer.singleShot(0, self.fit_view)

    def _refresh_viewers(self, *args: Any) -> None:
        camera_list = cameras(self.project) if self.project is not None else []
        self.scene_view.set_context(self.result, camera_list, self._time_index)
        self.calibration_scene.set_context(self.result, camera_list, self._time_index)
        if not camera_list:
            return
        selected = max(0, self.viewer_camera.currentIndex())
        for offset, view in enumerate(self.plate_views):
            view.overlays = {
                key: control.isChecked() for key, control in self.overlay_controls.items()
            }
            view.set_context(
                camera_list[(selected + offset) % len(camera_list)],
                self.result,
                self._time_index,
                self.current_time(),
            )

    def _orthographic(self, checked: bool) -> None:
        self.scene_view.orthographic = checked
        self.scene_view.update()

    def fit_view(self) -> None:
        if self.viewer_mode.currentText() == "3D Scene":
            self.scene_view.frame_actor()
        else:
            for view in self.plate_views:
                view.fit_image()

    def current_time(self) -> float:
        times = np.asarray(self.result.get("times", []))
        return float(times[min(self._time_index, len(times) - 1)]) if len(times) else 0.0

    def set_time_index(self, index: int) -> None:
        self._time_index = min(max(0, int(index)), self.timeline.maximum())
        self.timeline.blockSignals(True)
        self.timeline.setValue(self._time_index)
        self.timeline.blockSignals(False)
        self.time_label.setText(f"world {self.current_time():.3f} s  |  frame {self._time_index}")
        self.contact_tracks.current_time = self.current_time()
        self.contact_tracks.update()
        self._refresh_viewers()

    def seek_time(self, timestamp: float) -> None:
        times = np.asarray(self.result.get("times", []))
        if len(times):
            self.set_time_index(int(np.argmin(abs(times - timestamp))))

    def toggle_play(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("▶")
        else:
            fps = float(value(self.project, "output_fps", 24)) if self.project else 24
            self.play_timer.start(max(1, round(1000 / fps)))
            self.play_button.setText("Ⅱ")

    def _advance(self) -> None:
        index = self._time_index + 1
        self.set_time_index(index if index <= self.timeline.maximum() else 0)

    def export_dialog(self) -> None:
        if not self._require_project():
            return
        dialog = ExportDialog(self.output_directory(), self)
        dialog.fps.setValue(float(value(self.project, "output_fps", 24)))
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                settings = dialog.configuration()
                fmt, path = settings.pop("format"), settings.pop("path")
                self.start_job(
                    f"Export {fmt.upper()}",
                    lambda callback, cancel: self.service.export(
                        self.project, fmt, path, **settings
                    ),
                    self._export_complete,
                )
            except Exception as exc:
                self.error("Invalid export settings", str(exc))

    def _export_complete(self, metadata: Any) -> None:
        validated = bool(value(metadata, "validated", False))
        self.export_summary.setPlainText(
            ("EXPORT VALIDATED\n" if validated else "EXPORTED · validation not confirmed\n")
            + json.dumps(metadata, indent=2, default=str)
        )
        self.tabs.setCurrentIndex(self.tab_index["Export"])
        self.refresh_project()

    def open_output(self) -> None:
        if self._require_project():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_directory())))

    def output_directory(self) -> Path:
        path = Path(value(self.project, "output_dir", "outputs"))
        return path if path.is_absolute() else project_path(self.project) / path

    def open_report(self) -> None:
        if self._require_project():
            output = self.output_directory()
            candidates = list(output.glob("**/*report*.html")) + list(output.glob("**/qc.html"))
            if candidates:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(candidates[0])))
            else:
                self.error(
                    "QC report unavailable", "Run reconstruction to generate the HTML QC report."
                )

    def show_doctor(self) -> None:
        self.start_job(
            "Environment diagnostics",
            lambda callback, cancel: self.service.doctor(),
            self._doctor_complete,
        )

    def _doctor_complete(self, diagnostics: Any) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Project Health / first-run setup")
        dialog.resize(830, 620)
        layout = QVBoxLayout(dialog)
        label = QLabel(
            "Verify runtime, model assets and export backend. Optional components are reported separately."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        browser = QTextBrowser()
        browser.setPlainText(json.dumps(diagnostics, indent=2, default=str))
        layout.addWidget(browser)
        row = QHBoxLayout()
        row.addWidget(
            button("Model setup manual", lambda: (dialog.accept(), self.show_help("model")))
        )
        row.addWidget(button("Locate licensed model assets…", self.locate_models))
        row.addWidget(button("Synthetic example", lambda: (dialog.accept(), self.create_example())))
        row.addWidget(button("Close", dialog.accept))
        layout.addLayout(row)
        self.preferences.data["setup_completed"] = True
        self.preferences.save()
        dialog.exec()

    def locate_models(self) -> None:
        selected = QFileDialog.getOpenFileName(
            self,
            "Locate licensed SMPL-family model file",
            str(install_root() / "models"),
            "Body model (*.npz);;All files (*)",
        )[0]
        if selected:
            self.preferences.data["model_path"] = selected
            self.preferences.save()
            self.actor_assets.setText(selected)
            if self.project is not None:
                self.push_override("model_path", selected)

    def show_help(self, search: str | bool = "") -> None:
        self.stack.setCurrentWidget(self.workspace)
        self.tabs.setCurrentIndex(self.tab_index["Help"])
        self.help_search.setText(search if isinstance(search, str) else "")

    def search_help(self, query: str) -> None:
        manual = Path(__file__).resolve().parents[3] / "docs" / "manual"
        matches = []
        for path in sorted(manual.glob("*.md")):
            contents = path.read_text()
            if not query or query.casefold() in (path.name + contents).casefold():
                title = next(
                    (line.lstrip("# ") for line in contents.splitlines() if line.startswith("#")),
                    path.stem,
                )
                matches.append(f'<p><a href="{path.as_uri()}">{title}</a></p>')
        self.help_browser.setHtml(
            "<h2>User manual</h2>" + "".join(matches)
            if matches
            else "No matching manual page. Try camera, timing, model, QC or export."
        )

    def _help_link(self, url: Any) -> None:
        path = Path(url.toLocalFile())
        if path.suffix == ".md" and path.exists():
            self.help_browser.setMarkdown(path.read_text())
        else:
            QDesktopServices.openUrl(url)

    def closeEvent(self, event: Any) -> None:
        if self.job is not None and self.job.isRunning():
            response = QMessageBox.question(
                self,
                "Processing is active",
                "Cancel at a safe checkpoint before closing? Completed artifacts are retained.",
            )
            if response != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.job.cancel()
            self.job.wait(1500)
            if self.job.isRunning():
                self.statusBar().showMessage(
                    "Cancellation pending. Close after the job reaches a safe checkpoint."
                )
                event.ignore()
                return
        self.preferences.data["viewer_mode"] = self.viewer_mode.currentText()
        self.preferences.data["workspace_sizes"] = self.main_splitter.sizes()
        self.preferences.data["selected_camera"] = self.viewer_camera.currentText()
        self.preferences.data["overlays"] = {
            key: control.isChecked() for key, control in self.overlay_controls.items()
        }
        self.preferences.save()
        if self.project is not None:
            self.service.save_project(self.project)
        super().closeEvent(event)


def launch(project: str | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("OpenMocap VFX")
    app.setOrganizationName("OpenMocap")
    window = MainWindow()
    if project:
        window.open_project(project)
    window.show()
    if not window.preferences.data.get("setup_completed"):
        QTimer.singleShot(250, window.show_doctor)
    return app.exec()
