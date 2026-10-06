"""Lazy plate viewer and interactive metric scene viewport.

The scene viewport uses Qt's raster backend so it also works on render-farm
machines without an OpenGL context. It displays actual solved geometry.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from .state import camera_id, value

MEASURED = QColor("#52d9bc")
REPROJECTED = QColor("#f9b95f")
REJECTED = QColor("#ff667a")
INFERRED = QColor("#ad9bfa")
EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
    (21, 23),
]


class ContactTimeline(QWidget):
    """World-time contact and missing-data tracks; click to seek."""

    time_selected = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(82)
        self.result: dict[str, Any] = {}
        self.current_time = 0.0
        self.setToolTip(
            "Green: planted foot. Amber: possible contact. Red: missing joint samples. Click a track to seek world time."
        )

    def set_context(self, result: dict[str, Any], timestamp: float) -> None:
        self.result, self.current_time = result, timestamp
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#172131"))
        names = ["Left heel", "Left forefoot", "Right heel", "Right forefoot", "Missing data"]
        left, usable = 110, max(1, self.width() - 122)
        contacts = self.result.get("contacts", {})
        times = np.asarray(
            contacts.get("times", self.result.get("times", []))
            if isinstance(contacts, dict)
            else self.result.get("times", [])
        )
        probabilities = np.asarray(
            contacts.get("probability", []) if isinstance(contacts, dict) else []
        )
        planted = np.asarray(contacts.get("planted", []) if isinstance(contacts, dict) else [])
        confidence = np.asarray(self.result.get("confidence", []))
        for row, name in enumerate(names):
            y = 3 + row * 15
            painter.setPen(QColor("#9fb3cc"))
            painter.drawText(6, y + 11, name)
            painter.fillRect(QRectF(left, y, usable, 12), QColor("#233448"))
            if len(times) > 1:
                span = max(1e-6, float(times[-1] - times[0]))
                for frame in range(len(times)):
                    start = float(times[frame] - times[0]) / span
                    end = float(times[min(frame + 1, len(times) - 1)] - times[0]) / span
                    width = max(1, (end - start) * usable)
                    color = None
                    if (
                        row < 4
                        and planted.ndim == 2
                        and frame < len(planted)
                        and row < planted.shape[1]
                        and planted[frame, row]
                    ):
                        color = MEASURED
                    elif (
                        row < 4
                        and probabilities.ndim == 2
                        and frame < len(probabilities)
                        and row < probabilities.shape[1]
                        and probabilities[frame, row] >= 0.35
                    ):
                        color = REPROJECTED
                    elif (
                        row == 4
                        and confidence.ndim == 2
                        and frame < len(confidence)
                        and np.any(confidence[frame] <= 0)
                    ):
                        color = REJECTED
                    if color:
                        painter.fillRect(QRectF(left + start * usable, y + 2, width, 8), color)
        if len(times) > 1:
            x = (
                left
                + np.clip((self.current_time - times[0]) / (times[-1] - times[0]), 0, 1) * usable
            )
            painter.setPen(QPen(QColor("#e5edf9"), 1))
            painter.drawLine(QPointF(x, 0), QPointF(x, self.height() - 2))
        else:
            painter.setPen(QColor("#7d93ae"))
            painter.drawText(left + 8, 15, "Solve contacts to display planted intervals")

    def mousePressEvent(self, event: Any) -> None:
        times = np.asarray(self.result.get("times", []))
        if len(times) > 1:
            fraction = np.clip((event.position().x() - 110) / max(1, self.width() - 122), 0, 1)
            self.time_selected.emit(float(times[0] + fraction * (times[-1] - times[0])))


class DecodeSignals(QObject):
    loaded = Signal(object, object, float)


class DecodePlate(QRunnable):
    """Decode one requested plate outside the UI thread."""

    def __init__(
        self, key: tuple[str, int], filename: str, timestamp: float, signals: DecodeSignals
    ) -> None:
        super().__init__()
        self.key, self.filename, self.timestamp, self.signals = key, filename, timestamp, signals

    def run(self) -> None:
        image = QImage()
        actual_time = self.timestamp
        path = Path(self.filename)
        try:
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".exr"}:
                image = QImage(self.filename)
            elif path.is_file():
                import cv2

                capture = cv2.VideoCapture(self.filename)
                try:
                    capture.set(cv2.CAP_PROP_POS_MSEC, max(0, self.timestamp * 1000))
                    success, frame = capture.read()
                    if success:
                        actual_time = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = QImage(
                            rgb.data,
                            rgb.shape[1],
                            rgb.shape[0],
                            rgb.strides[0],
                            QImage.Format.Format_RGB888,
                        ).copy()
                finally:
                    capture.release()
        except (OSError, ImportError, ValueError):
            pass
        try:
            self.signals.loaded.emit(self.key, image, actual_time)
        except RuntimeError:
            # The viewer may have been closed while a plate was decoding.
            return


class PlateView(QGraphicsView):
    pixel_inspected = Signal(float, float)
    observation_selected = Signal(str)
    raw_observation_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#151b24"))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.camera: Any = None
        self.result: dict[str, Any] = {}
        self.index = 0
        self.world_time = 0.0
        self.overlays = {
            "measured": True,
            "reprojected": True,
            "vectors": True,
            "confidence": False,
            "rejected": True,
        }
        self._capture: Any = None
        self._capture_source = ""
        self._last_frame_key: tuple[str, int] | None = None
        self._background: QPixmap | None = None
        self._image_item: QGraphicsPixmapItem | None = None
        self._sequence_cache: dict[str, list[Path]] = {}
        self._manifest_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.actual_camera_time = 0.0
        self.stale = False
        self._decode_signals = DecodeSignals(self)
        self._decode_signals.loaded.connect(self._decoded)
        self._decode_pool = QThreadPool.globalInstance()
        self._decode_pool.setMaxThreadCount(4)
        self._pending_decode = False
        self._requested_decode: tuple[tuple[str, int], str, float] | None = None
        self.loading = False

    def set_context(
        self, camera: Any, result: dict[str, Any], index: int, world_time: float
    ) -> None:
        self.camera, self.result, self.index, self.world_time = camera, result, index, world_time
        self.refresh()

    def _read_plate(self, source: str, camera_time: float) -> QPixmap | None:
        """Request lazy decode; retain only the latest scrub request."""
        if not source:
            return None
        path = Path(source)
        if path.is_dir():
            if source not in self._sequence_cache:
                self._sequence_cache[source] = sorted(
                    p
                    for p in path.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".exr", ".tif"}
                )
            files = self._sequence_cache[source]
            manifest = value(self.camera, "frames", [])
            if manifest:
                if source not in self._manifest_cache:
                    timestamps = np.array(
                        [value(frame, "camera_timestamp", 0) for frame in manifest]
                    )
                    order = np.argsort(timestamps, kind="stable")
                    self._manifest_cache[source] = timestamps[order], order
                timestamps, order = self._manifest_cache[source]
                position = int(np.searchsorted(timestamps, camera_time))
                candidates = np.clip([position - 1, position], 0, len(timestamps) - 1)
                position = int(candidates[np.argmin(abs(timestamps[candidates] - camera_time))])
                index = int(order[position])
                self.actual_camera_time = float(timestamps[position])
                filename = value(manifest[index], "path", "")
                candidate = Path(filename)
                if not candidate.is_absolute():
                    candidate = path / candidate.name
                if candidate.exists():
                    path = candidate
                    camera_time = self.actual_camera_time
                elif files:
                    path = files[min(index, len(files) - 1)]
            else:
                fps = float(value(self.camera, "fps", 30))
                index = max(0, round(camera_time * fps))
                self.actual_camera_time = index / fps
                if files:
                    path = files[min(index, len(files) - 1)]
            if path.is_dir():
                return None
        key = (source, round(camera_time * 1000))
        self._requested_decode = (self._last_frame_key or key, str(path), camera_time)
        self.loading = True
        self._schedule_decode()
        return self._background

    def _schedule_decode(self) -> None:
        if not self._pending_decode and self._requested_decode is not None:
            self._pending_decode = True
            self._decode_pool.start(DecodePlate(*self._requested_decode, self._decode_signals))

    def _decoded(self, key: tuple[str, int], image: QImage, timestamp: float) -> None:
        self._pending_decode = False
        if self._requested_decode is not None and key == self._requested_decode[0]:
            self._background = QPixmap.fromImage(image) if not image.isNull() else None
            self.actual_camera_time = timestamp
            self.loading = False
            self.refresh()
        else:
            self._schedule_decode()

    def _project(self, points: np.ndarray) -> np.ndarray | None:
        if self.camera is None:
            return None
        try:
            from openmocap.types import Camera

            core_camera = (
                Camera.from_dict(self.camera) if isinstance(self.camera, dict) else self.camera
            )
            return np.asarray(core_camera.project(points, self.world_time))
        except (AttributeError, TypeError, ValueError, KeyError):
            # Uncalibrated cameras cannot produce truthful reprojection overlays.
            return None

    def refresh(self) -> None:
        scene = self.scene()
        scene.clear()
        intrinsics = value(self.camera, "intrinsics", {})
        resolution = value(
            self.camera,
            "resolution",
            [value(intrinsics, "width", 1920), value(intrinsics, "height", 1080)],
        )
        width, height = int(resolution[0]), int(resolution[1])
        mapping = value(self.camera, "time_mapping", {})
        offset, scale = float(value(mapping, "offset", 0)), float(value(mapping, "scale", 1))
        camera_time = (self.world_time - offset) / scale
        source = str(
            value(self.camera, "media_source", value(self.camera, "source_path", "")) or ""
        )
        key = (source, round(camera_time * 1000))
        if self._last_frame_key != key:
            self._last_frame_key = key
            self._background = self._read_plate(source, camera_time)
        if self._background is not None and not self._background.isNull():
            scene.addPixmap(self._background)
            width, height = self._background.width(), self._background.height()
        else:
            scene.addRect(0, 0, width, height, QPen(QColor("#364151")), QColor("#182231"))
            for x in range(0, width, max(1, width // 12)):
                scene.addLine(x, 0, x, height, QPen(QColor("#213044")))
            for y in range(0, height, max(1, height // 8)):
                scene.addLine(0, y, width, y, QPen(QColor("#213044")))
            label = scene.addText("No plate available · calibrated projection")
            label.setDefaultTextColor(QColor("#aab9cd"))
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            label.setPos(width * 0.025, height * 0.06)
        joints = np.asarray(self.result.get("joints", []))
        if joints.ndim == 3 and len(joints):
            index = min(self.index, len(joints) - 1)
            projected = self._project(joints[index])
            if projected is not None and self.overlays["reprojected"]:
                self._draw_joints(projected, REPROJECTED, index, width)
        observations = self.result.get("observations", [])
        observation_index = self.result.get("_ui_observation_index", {}).get(camera_id(self.camera))
        if observation_index is not None:
            timestamps, camera_observations = observation_index
            left = int(np.searchsorted(timestamps, self.world_time - 0.025, side="left"))
            right = int(np.searchsorted(timestamps, self.world_time + 0.025, side="right"))
            observations = camera_observations[left:right]
        projected = (
            self._project(joints[min(self.index, len(joints) - 1)])
            if joints.ndim == 3 and len(joints)
            else None
        )
        if isinstance(observations, list) and self.overlays["measured"]:
            for observation in observations:
                if str(value(observation, "camera_id")) != camera_id(self.camera):
                    continue
                timestamp = value(
                    observation, "world_timestamp", value(observation, "world_time", 0)
                )
                if abs(float(timestamp) - self.world_time) > 0.025:
                    continue
                confidence = float(value(observation, "confidence", 1))
                rejected = bool(value(observation, "rejected", False))
                if rejected and not self.overlays["rejected"]:
                    continue
                color = REJECTED if rejected else MEASURED
                xy = value(
                    observation, "xy", [value(observation, "x", 0), value(observation, "y", 0)]
                )
                radius = max(3, width / 260)
                item = scene.addEllipse(
                    float(xy[0]) - radius,
                    float(xy[1]) - radius,
                    2 * radius,
                    2 * radius,
                    QPen(color, max(2, width / 650)),
                )
                raw_index = self.result.get("_ui_raw_lookup", {}).get(id(observation))
                if raw_index is not None:
                    item.setData(1, raw_index)
                    item.setToolTip(
                        f"Measured observation {raw_index} · {value(observation, 'joint_id', '')} · confidence {confidence:.2f}"
                    )
                names = list(self.result.get("joint_names", []))
                identifier = value(observation, "joint_id", -1)
                joint_index = names.index(identifier) if identifier in names else identifier
                if (
                    self.overlays["vectors"]
                    and projected is not None
                    and isinstance(joint_index, int)
                    and 0 <= joint_index < len(projected)
                    and np.all(np.isfinite(projected[joint_index]))
                ):
                    scene.addLine(
                        float(xy[0]),
                        float(xy[1]),
                        float(projected[joint_index, 0]),
                        float(projected[joint_index, 1]),
                        QPen(REJECTED if rejected else REPROJECTED, max(1, width / 950)),
                    )
                if self.overlays["confidence"]:
                    text = scene.addText(f"{confidence:.2f}")
                    text.setDefaultTextColor(color)
                    text.setPos(float(xy[0]) + radius, float(xy[1]))
        caption = scene.addText(
            f"{'LOADING · ' if self.loading else ''}{'STALE · ' if self.stale else ''}{camera_id(self.camera)}  plate {self.actual_camera_time:.3f}s  world {self.world_time:.3f}s  Δt {offset:+.4f}s"
        )
        caption.setDefaultTextColor(QColor("#e0e9f5"))
        caption.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        caption.setPos(width * 0.025, height * 0.015)
        scene.setSceneRect(QRectF(0, 0, width, height))

    def _draw_joints(self, projected: np.ndarray, color: QColor, index: int, width: int) -> None:
        edges = self.result.get("edges", EDGES)
        pen = QPen(color, max(2, width / 600))
        for left, right in edges:
            if max(left, right) < len(projected) and np.all(np.isfinite(projected[[left, right]])):
                self.scene().addLine(
                    *map(float, projected[left]), *map(float, projected[right]), pen
                )
        for joint, xy in enumerate(projected):
            if np.all(np.isfinite(xy)):
                radius = max(3, width / 350)
                item = self.scene().addEllipse(
                    float(xy[0]) - radius, float(xy[1]) - radius, radius * 2, radius * 2, pen, color
                )
                item.setData(0, joint)
                item.setToolTip(f"Reprojected joint {joint} · time {self.world_time:.4f}s")

    def fit_image(self) -> None:
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: Any) -> None:
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)
        event.accept()

    def mouseMoveEvent(self, event: Any) -> None:
        position = self.mapToScene(event.position().toPoint())
        self.pixel_inspected.emit(position.x(), position.y())
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: Any) -> None:
        item = self.itemAt(event.position().toPoint())
        if item is not None and item.data(1) is not None:
            self.raw_observation_selected.emit(int(item.data(1)))
        if item is not None and item.data(0) is not None:
            self.observation_selected.emit(str(item.data(0)))
        super().mousePressEvent(event)

    def closeEvent(self, event: Any) -> None:
        if self._capture is not None:
            self._capture.release()
        super().closeEvent(event)


class SceneView(QWidget):
    selected = Signal(str)
    camera_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.result: dict[str, Any] = {}
        self.cameras: list[Any] = []
        self.index = 0
        self.yaw, self.pitch, self.distance = 0.5, 0.22, 5.0
        self.target = np.array([0.0, 0.8, 0.0])
        self.orthographic = False
        self.show_mesh = True
        self.show_cameras = True
        self.show_uncertainty = True
        self.floor_y = 0.0
        self.stale = False
        self._mouse: Any = None
        self._screen_joints: list[tuple[float, float, int]] = []
        self._screen_cameras: list[tuple[float, float, str]] = []

    def set_context(self, result: dict[str, Any], camera_list: list[Any], index: int) -> None:
        self.result, self.cameras, self.index = result, camera_list, index
        self.update()

    def reset_view(self) -> None:
        self.yaw, self.pitch, self.distance = 0.5, 0.22, 5.0
        self.target = np.array([0.0, 0.8, 0.0])
        self.update()

    def frame_actor(self) -> None:
        joints = np.asarray(self.result.get("joints", []))
        if joints.ndim == 3 and len(joints):
            points = joints[min(self.index, len(joints) - 1)]
            self.target = np.nanmean(points, axis=0)
            self.distance = max(
                2, float(np.nanmax(np.linalg.norm(points - self.target, axis=1))) * 4
            )
        self.update()

    def _projection(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cy, sy, cp, sp = (
            math.cos(self.yaw),
            math.sin(self.yaw),
            math.cos(self.pitch),
            math.sin(self.pitch),
        )
        rotation = np.array([[cy, 0, -sy], [-sp * sy, cp, -sp * cy], [cp * sy, sp, cp * cy]])
        local = (np.asarray(points) - self.target) @ rotation.T
        depth = self.distance - local[:, 2]
        focal = min(self.width(), self.height()) * 1.1
        denominator = (
            np.full_like(depth, self.distance) if self.orthographic else np.maximum(depth, 0.05)
        )
        screen = np.column_stack(
            (
                self.width() / 2 + focal * local[:, 0] / denominator,
                self.height() / 2 - focal * local[:, 1] / denominator,
            )
        )
        return screen, depth

    def _line(self, painter: QPainter, points: np.ndarray, color: QColor, width: float = 1) -> None:
        screen, depth = self._projection(points)
        if np.all(depth > 0.05) and np.all(np.isfinite(screen)):
            painter.setPen(QPen(color, width))
            painter.drawLine(QPointF(*screen[0]), QPointF(*screen[1]))

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#121923"))
        for coordinate in np.arange(-5, 5.01, 0.5):
            self._line(
                painter,
                np.array([[coordinate, self.floor_y, -5], [coordinate, self.floor_y, 5]]),
                QColor("#263345"),
            )
            self._line(
                painter,
                np.array([[-5, self.floor_y, coordinate], [5, self.floor_y, coordinate]]),
                QColor("#263345"),
            )
        for axis, color in [(0, QColor("#ef777f")), (1, QColor("#76d9a2")), (2, QColor("#78aefa"))]:
            end = np.zeros(3)
            end[axis] = 0.6
            self._line(painter, np.stack((np.zeros(3), end)), color, 3)
        if self.show_cameras:
            self._screen_cameras = []
            for camera in self.cameras:
                try:
                    from openmocap.types import Camera

                    model = Camera.from_dict(camera) if isinstance(camera, dict) else camera
                    times = self.result.get("times", [])
                    timestamp = float(times[min(self.index, len(times) - 1)]) if len(times) else 0
                    pose = model.pose_at(timestamp)
                    intrinsic = model.intrinsics_at(timestamp)
                    normalized = model.undistort(
                        [
                            [0, 0],
                            [intrinsic.width, 0],
                            [intrinsic.width, intrinsic.height],
                            [0, intrinsic.height],
                        ],
                        timestamp,
                    )
                except (ValueError, KeyError, AttributeError, TypeError):
                    continue
                r, t = pose.rotation, pose.translation
                center = -r.T @ t
                corners = np.column_stack((normalized * 0.35, np.full(4, 0.35))) @ r + center
                if model.trajectory:
                    centers = np.stack(
                        [model.pose_at(time).centre for time in model.trajectory.times]
                    )
                    for start, end in zip(centers[:-1], centers[1:], strict=True):
                        self._line(painter, np.stack((start, end)), QColor("#867ac5"), 2)
                for corner in corners:
                    self._line(painter, np.stack((center, corner)), QColor("#608cb4"))
                for i in range(4):
                    self._line(
                        painter, np.stack((corners[i], corners[(i + 1) % 4])), QColor("#608cb4")
                    )
                screen, _ = self._projection(center[None])
                self._screen_cameras.append(
                    (float(screen[0, 0]), float(screen[0, 1]), camera_id(camera))
                )
                painter.setPen(QColor("#8eaccb"))
                painter.drawText(QPointF(*screen[0]), camera_id(camera))
        joints = np.asarray(self.result.get("joints", []))
        self._screen_joints = []
        if joints.ndim == 3 and len(joints):
            index = min(self.index, len(joints) - 1)
            points = joints[index]
            vertices = np.asarray(self.result.get("vertices", []))
            faces = np.asarray(self.result.get("faces", []))
            animation = self.result.get("animation")
            if animation is not None:
                vertices = np.asarray(animation.mesh_at(index))
                faces = np.asarray(animation.faces)
            if self.show_mesh and vertices.size and faces.size:
                verts = vertices[index] if vertices.ndim == 3 else vertices
                screen, depth = self._projection(verts)
                stride = max(1, len(faces) // 2500)
                painter.setPen(QPen(QColor("#405c75"), 0.4))
                painter.setBrush(QColor(71, 100, 129, 80))
                for face in faces[::stride]:
                    if np.all(depth[face] > 0.05):
                        painter.drawPolygon(QPolygonF([QPointF(*screen[v]) for v in face]))
            for left, right in self.result.get("edges", EDGES):
                if max(left, right) < len(points):
                    self._line(painter, points[[left, right]], MEASURED, 3)
            screen, depth = self._projection(points)
            confidence = np.asarray(self.result.get("confidence", np.ones(joints.shape[:2])))
            if confidence.ndim == 2:
                confidence = confidence[index]
            for joint, (xy, z) in enumerate(zip(screen, depth, strict=True)):
                if z <= 0.05 or not np.all(np.isfinite(xy)):
                    continue
                c = float(confidence[joint]) if len(confidence) > joint else 1
                color = MEASURED if c >= 0.65 else INFERRED
                painter.setPen(QPen(color, 1))
                painter.setBrush(color)
                painter.drawEllipse(QPointF(*xy), 4, 4)
                if self.show_uncertainty and c < 0.85:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    radius = 5 + (1 - c) * 22
                    painter.drawEllipse(QPointF(*xy), radius, radius)
                self._screen_joints.append((float(xy[0]), float(xy[1]), joint))
            contacts = self.result.get("contacts", [])
            if isinstance(contacts, dict):
                planted = np.asarray(contacts.get("planted", []))
                names = list(self.result.get("joint_names", []))
                if planted.ndim == 2 and index < len(planted):
                    contact_points = []
                    for foot_index, foot in enumerate(contacts.get("foot_names", [])):
                        aliases = {
                            "left_heel": "left_ankle",
                            "left_toe": "left_foot",
                            "right_heel": "right_ankle",
                            "right_toe": "right_foot",
                        }
                        joint_name = aliases.get(foot, foot)
                        if planted[index, foot_index] and joint_name in names:
                            contact_points.append(
                                {"start": index, "end": index, "joint_id": names.index(joint_name)}
                            )
                    contacts = contact_points
                else:
                    contacts = []
            if isinstance(contacts, list):
                for contact in contacts:
                    if value(contact, "start", -1) <= index <= value(contact, "end", -1):
                        joint = value(contact, "joint_id", -1)
                        if isinstance(joint, int) and 0 <= joint < len(screen):
                            painter.setPen(QPen(QColor("#f9b95f"), 2))
                            painter.setBrush(Qt.BrushStyle.NoBrush)
                            painter.drawEllipse(QPointF(*screen[joint]), 10, 10)
        painter.setPen(QColor("#91a3ba"))
        mode = "Orthographic" if self.orthographic else "Perspective"
        painter.drawText(
            16, 24, f"{'STALE SOLVE · ' if self.stale else ''}Metric world · +Y up · {mode}"
        )
        painter.drawText(
            16,
            self.height() - 15,
            "Drag: orbit    Shift-drag: pan    Wheel: zoom    Click: inspect joint",
        )

    def wheelEvent(self, event: Any) -> None:
        self.distance = np.clip(
            self.distance * (0.9 if event.angleDelta().y() > 0 else 1.1), 0.3, 100
        )
        self.update()

    def mousePressEvent(self, event: Any) -> None:
        self._mouse = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            for x, y, identifier in self._screen_cameras:
                if math.hypot(x - event.position().x(), y - event.position().y()) < 16:
                    self.camera_selected.emit(identifier)
                    return
            for x, y, joint in self._screen_joints:
                if math.hypot(x - event.position().x(), y - event.position().y()) < 10:
                    names = self.result.get("joint_names", [])
                    self.selected.emit(str(names[joint] if len(names) > joint else joint))
                    break

    def mouseMoveEvent(self, event: Any) -> None:
        if self._mouse is not None and event.buttons():
            delta = event.position() - self._mouse
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.target[0] -= delta.x() * self.distance / 700
                self.target[1] += delta.y() * self.distance / 700
            else:
                self.yaw += delta.x() / 180
                self.pitch = np.clip(self.pitch + delta.y() / 180, -1.5, 1.5)
            self._mouse = event.position()
            self.update()

    def mouseReleaseEvent(self, event: Any) -> None:
        self._mouse = None
