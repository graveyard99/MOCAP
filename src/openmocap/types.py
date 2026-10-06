"""Explicit metric data contracts shared by numerical, application, and UI layers.

World coordinates are metres, +Y up. Camera transforms map world coordinates
to OpenCV camera coordinates (+X right, +Y down, +Z forward).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class Source(str, Enum):
    MANUAL = "manual"
    CALIBRATION_TARGET = "calibration_target"
    COLMAP = "colmap"
    METASHAPE = "metashape"
    SURVEYED = "surveyed"
    BUNDLE_ADJUSTMENT = "bundle_adjustment"
    HUMAN_ASSISTED = "human_assisted"
    MULTIVIEW = "multiview"
    LEARNED = "learned"
    LEARNED_PRIOR = "learned_prior"
    DEFAULT = "default"
    SYNTHETIC = "synthetic"


class ParameterState(str, Enum):
    LOCKED = "LOCKED"
    BOUNDED = "BOUNDED"
    FREE = "FREE"


class DistortionModel(str, Enum):
    NONE = "none"
    OPENCV = "opencv"
    FISHEYE = "fisheye"


@dataclass
class ParameterControl:
    owner: str = "CALIBRATION"
    state: ParameterState = ParameterState.LOCKED
    source: str = Source.DEFAULT.value
    confidence: float = 1.0
    lower: list[float] | None = None
    upper: list[float] | None = None

    def __post_init__(self) -> None:
        self.state = ParameterState(self.state)
        if not 0 <= self.confidence <= 1:
            raise ValueError("Parameter confidence must be in [0, 1]")
        if self.lower is not None and self.upper is not None:
            if len(self.lower) != len(self.upper) or np.any(np.asarray(self.lower) >= self.upper):
                raise ValueError("Parameter bounds require matching lower < upper values")


def _array(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return result.copy()


@dataclass
class TimeMapping:
    """Affine camera clock mapping: world = scale * camera + offset (seconds)."""

    scale: float = 1.0
    offset: float = 0.0
    locked: bool = True
    source: str = Source.MANUAL.value
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite([self.scale, self.offset]).all() or self.scale <= 0:
            raise ValueError("Time mapping requires positive scale and finite offset")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Time confidence must be in [0, 1]")

    def to_world(self, timestamp: ArrayLike) -> Any:
        return np.asarray(timestamp) * self.scale + self.offset

    def to_camera(self, timestamp: ArrayLike) -> Any:
        return (np.asarray(timestamp) - self.offset) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: ArrayLike = field(default_factory=list)
    distortion_model: str = "opencv"
    sensor_width_mm: float | None = None
    sensor_height_mm: float | None = None
    focal_length_mm: float | None = None
    pixel_aspect: float = 1.0
    source: str = Source.MANUAL.value
    locked: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all():
            raise ValueError("Intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Focal lengths and image dimensions must be positive")
        if int(self.width) != self.width or int(self.height) != self.height:
            raise ValueError("Image dimensions must be integer pixels")
        self.width, self.height = int(self.width), int(self.height)
        if self.pixel_aspect <= 0 or not np.isfinite(self.pixel_aspect):
            raise ValueError("Pixel aspect must be positive and finite")
        self.distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        self.distortion_model = DistortionModel(self.distortion_model).value
        if not np.isfinite(self.distortion).all():
            raise ValueError("Distortion coefficients must be finite")
        if self.distortion_model == "fisheye" and len(self.distortion) != 4:
            raise ValueError("Fisheye requires four distortion coefficients")
        if self.distortion_model == "opencv" and len(self.distortion) not in (0, 4, 5, 8, 12, 14):
            raise ValueError("OpenCV distortion requires 0, 4, 5, 8, 12, or 14 coefficients")

    @property
    def matrix(self) -> FloatArray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def K(self) -> FloatArray:
        return self.matrix

    @classmethod
    def from_metadata(
        cls,
        width: int,
        height: int,
        focal_length_mm: float,
        sensor_width_mm: float,
        sensor_height_mm: float | None = None,
        pixel_aspect: float = 1.0,
        **kwargs: Any,
    ) -> CameraIntrinsics:
        if focal_length_mm <= 0 or sensor_width_mm <= 0 or pixel_aspect <= 0:
            raise ValueError("Sensor/focal metadata and pixel aspect must be positive")
        fx = focal_length_mm / sensor_width_mm * width
        fy = focal_length_mm / sensor_height_mm * height if sensor_height_mm else fx / pixel_aspect
        return cls(
            fx,
            fy,
            width / 2,
            height / 2,
            width,
            height,
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            focal_length_mm=focal_length_mm,
            pixel_aspect=pixel_aspect,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["distortion"] = self.distortion.tolist()
        return result


@dataclass
class CameraPose:
    rotation: ArrayLike
    translation: ArrayLike

    def __post_init__(self) -> None:
        self.rotation = _array(self.rotation, (3, 3), "Camera rotation")
        self.translation = _array(self.translation, (3,), "Camera translation")
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-5):
            raise ValueError("Camera rotation must be orthonormal")
        if not np.isclose(np.linalg.det(self.rotation), 1, atol=1e-5):
            raise ValueError("Camera rotation must be right handed (determinant +1)")

    @property
    def centre(self) -> FloatArray:
        return -self.rotation.T @ self.translation

    @property
    def projection(self) -> FloatArray:
        return np.column_stack([self.rotation, self.translation])

    def to_dict(self) -> dict[str, Any]:
        return {"rotation": self.rotation.tolist(), "translation": self.translation.tolist()}


@dataclass
class CameraTrajectory:
    """Timestamped metric camera poses with SLERP and camera-centre interpolation.

    Extrapolation is rejected: a moving camera outside its tracked range is not
    silently treated as static. Intrinsics can vary at the same timestamps.
    """

    times: ArrayLike
    poses: list[CameraPose]
    intrinsics: list[CameraIntrinsics] | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64)
        if self.times.ndim != 1 or len(self.times) != len(self.poses) or len(self.times) < 2:
            raise ValueError("Camera trajectory requires at least two matching times and poses")
        if not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError("Camera trajectory timestamps must be finite and strictly increasing")
        if self.intrinsics and len(self.intrinsics) != len(self.times):
            raise ValueError("Changing intrinsics must match trajectory timestamps")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Camera trajectory confidence must be in [0, 1]")

    def _interval(self, time: float) -> tuple[int, float]:
        if time < self.times[0] - 1e-9 or time > self.times[-1] + 1e-9:
            raise ValueError(f"Camera time {time} outside tracked range {self.times[[0, -1]]}")
        i = int(np.clip(np.searchsorted(self.times, time) - 1, 0, len(self.times) - 2))
        u = float(np.clip((time - self.times[i]) / (self.times[i + 1] - self.times[i]), 0, 1))
        return i, u

    def pose_at(self, time: float) -> CameraPose:
        from scipy.spatial.transform import Rotation, Slerp

        i, u = self._interval(time)
        rotations = Rotation.from_matrix([self.poses[i].rotation, self.poses[i + 1].rotation])
        rotation = Slerp([0, 1], rotations)([u]).as_matrix()[0]
        centre = (1 - u) * self.poses[i].centre + u * self.poses[i + 1].centre
        return CameraPose(rotation, -rotation @ centre)

    def intrinsics_at(self, time: float, fallback: CameraIntrinsics) -> CameraIntrinsics:
        if not self.intrinsics:
            return fallback
        i, u = self._interval(time)
        a, b = self.intrinsics[i], self.intrinsics[i + 1]
        if (a.width, a.height, a.distortion_model) != (b.width, b.height, b.distortion_model):
            raise ValueError(
                "Trajectory lens model and resolution cannot change during interpolation"
            )
        if len(a.distortion) != len(b.distortion):
            raise ValueError("Trajectory distortion dimensions cannot change")
        return CameraIntrinsics(
            *[(1 - u) * getattr(a, k) + u * getattr(b, k) for k in ("fx", "fy", "cx", "cy")],
            a.width,
            a.height,
            (1 - u) * a.distortion + u * b.distortion,
            a.distortion_model,
            source=a.source,
            locked=a.locked,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "times": self.times.tolist(),
            "poses": [p.to_dict() for p in self.poses],
            "intrinsics": [k.to_dict() for k in self.intrinsics] if self.intrinsics else None,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraTrajectory:
        return cls(
            data["times"],
            [CameraPose(**p) for p in data["poses"]],
            [CameraIntrinsics(**k) for k in data["intrinsics"]] if data.get("intrinsics") else None,
            data.get("confidence", 1.0),
        )


@dataclass
class CameraQuality:
    confidence: float = 1.0
    reprojection_rmse: float | None = None
    source: str = Source.MANUAL.value
    metric_scale_known: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("Camera confidence must be in [0, 1]")
        if self.reprojection_rmse is not None and (
            self.reprojection_rmse < 0 or not np.isfinite(self.reprojection_rmse)
        ):
            raise ValueError("Calibration reprojection RMSE must be finite and nonnegative")


@dataclass
class Camera:
    id: str
    intrinsics: CameraIntrinsics
    pose: CameraPose
    trajectory: CameraTrajectory | None = None
    time_mapping: TimeMapping = field(default_factory=TimeMapping)
    quality: CameraQuality | float = field(default_factory=CameraQuality)
    locked: bool = True
    source: str = Source.MANUAL.value
    source_path: str | None = None
    enabled: bool = True
    controls: dict[str, ParameterControl] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("A stable nonempty camera ID is required")
        if isinstance(self.quality, (float, int)):
            self.quality = CameraQuality(float(self.quality), source=self.source)
        if isinstance(self.time_mapping, dict):
            self.time_mapping = TimeMapping(**self.time_mapping)

    @property
    def camera_id(self) -> str:
        return self.id

    def pose_at(self, time: float = 0) -> CameraPose:
        return self.trajectory.pose_at(time) if self.trajectory else self.pose

    def intrinsics_at(self, time: float = 0) -> CameraIntrinsics:
        return (
            self.trajectory.intrinsics_at(time, self.intrinsics)
            if self.trajectory
            else self.intrinsics
        )

    def project(self, points: ArrayLike, time: float = 0) -> FloatArray:
        from openmocap.cameras.models import project

        return project(points, self.intrinsics_at(time), self.pose_at(time))

    def undistort(self, xy: ArrayLike, time: float = 0) -> FloatArray:
        from openmocap.cameras.models import undistort

        return undistort(xy, self.intrinsics_at(time))

    def projection_matrix(self, time: float = 0) -> FloatArray:
        """Ideal pinhole transform; distortion still requires project()."""
        return self.intrinsics_at(time).matrix @ self.pose_at(time).projection

    def set_pose(self, pose: CameraPose, *, source: str = Source.BUNDLE_ADJUSTMENT.value) -> None:
        control = self.controls.get("extrinsics")
        if self.locked or (control and control.state == ParameterState.LOCKED):
            raise PermissionError(f"Camera {self.id} extrinsics are locked and authoritative")
        self.pose = pose
        self.source = source

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "intrinsics": self.intrinsics.to_dict(),
            "pose": self.pose.to_dict(),
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
            "time_mapping": self.time_mapping.to_dict(),
            "quality": asdict(self.quality),
            "locked": self.locked,
            "source": self.source,
            "source_path": self.source_path,
            "enabled": self.enabled,
            "controls": {k: asdict(v) for k, v in self.controls.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Camera:
        quality = data.get("quality", {})
        return cls(
            data.get("id", data.get("camera_id")),
            CameraIntrinsics(**data["intrinsics"]),
            CameraPose(**data["pose"]),
            CameraTrajectory.from_dict(data["trajectory"]) if data.get("trajectory") else None,
            TimeMapping(**data.get("time_mapping", {})),
            CameraQuality(**quality) if isinstance(quality, dict) else quality,
            data.get("locked", True),
            data.get("source", "manual"),
            data.get("source_path"),
            data.get("enabled", True),
            {k: ParameterControl(**v) for k, v in data.get("controls", {}).items()},
        )


@dataclass
class FrameTimestamp:
    presentation_seconds: float
    frame_index: int
    source: str = "presentation_timestamp"
    duplicate: bool = False
    discontinuity: bool = False


@dataclass
class Frame:
    camera_id: str
    timestamp: FrameTimestamp
    image_path: str | None = None
    image: NDArray[Any] | None = None


@dataclass
class PersonDetection:
    camera_id: str
    camera_timestamp: float
    person_id: str
    bbox: ArrayLike
    confidence: float
    source: str = "detector"
    embedding: ArrayLike | None = None


@dataclass
class Track:
    id: str
    camera_id: str
    detections: list[PersonDetection] = field(default_factory=list)
    confidence: float = 1.0
    ambiguous: bool = False


@dataclass
class Pose2DObservation:
    camera_id: str
    camera_timestamp: float
    person_id: str
    joint_id: str
    xy: ArrayLike
    confidence: float = 1.0
    world_timestamp: float | None = None
    source: str = "pose2d"
    enabled: bool = True
    covariance: ArrayLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, dtype=np.float64).reshape(2)
        if not np.isfinite(self.camera_timestamp):
            raise ValueError("Camera timestamp must be finite")
        if self.world_timestamp is not None and not np.isfinite(self.world_timestamp):
            raise ValueError("World timestamp must be finite")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Observation confidence must be in [0, 1]")
        if self.covariance is not None:
            self.covariance = _array(self.covariance, (2, 2), "Plate covariance")
            if (
                not np.allclose(self.covariance, self.covariance.T)
                or np.linalg.eigvalsh(self.covariance).min() <= 0
            ):
                raise ValueError("Plate covariance must be symmetric positive definite")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["xy"] = self.xy.tolist()
        if self.covariance is not None:
            result["covariance"] = np.asarray(self.covariance).tolist()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pose2DObservation:
        values = dict(data)
        if "xy" not in values:
            values["xy"] = [values.pop("x"), values.pop("y")]
        known = {item.name for item in fields(cls)}
        metadata = dict(values.get("metadata", {}))
        for key in set(values) - known:
            metadata[key] = values.pop(key)
        values["metadata"] = metadata
        return cls(**values)


Joint2DObservation = Pose2DObservation


@dataclass
class SilhouetteObservation:
    camera_id: str
    camera_timestamp: float
    person_id: str
    mask_path: str
    confidence: float = 1.0
    source: str = "segmentation"
    world_timestamp: float | None = None


@dataclass
class TriangulationDiagnostics:
    contributing_camera_ids: list[str] = field(default_factory=list)
    rejected_camera_ids: list[str] = field(default_factory=list)
    reprojection_errors: dict[str, float] = field(default_factory=dict)
    mean_reprojection_error: float = float("inf")
    median_reprojection_error: float = float("inf")
    max_reprojection_error: float = float("inf")
    max_ray_angle_degrees: float = 0.0
    condition_number: float = float("inf")
    degenerate: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def num_views(self) -> int:
        return len(self.contributing_camera_ids)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["num_views"] = self.num_views
        for key in (
            "mean_reprojection_error",
            "median_reprojection_error",
            "max_reprojection_error",
            "max_ray_angle_degrees",
            "condition_number",
        ):
            if not np.isfinite(result[key]):
                result[key] = None
        result["reprojection_errors"] = {
            key: float(value) if np.isfinite(value) else None
            for key, value in self.reprojection_errors.items()
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TriangulationDiagnostics:
        values = {
            key: value for key, value in data.items() if key in {item.name for item in fields(cls)}
        }
        for key in (
            "mean_reprojection_error",
            "median_reprojection_error",
            "max_reprojection_error",
            "condition_number",
        ):
            if key in values and values[key] is None:
                values[key] = float("inf")
        if "reprojection_errors" in values:
            values["reprojection_errors"] = {
                key: float(value) if value is not None else float("inf")
                for key, value in values["reprojection_errors"].items()
            }
        return cls(**values)


@dataclass
class Joint3DObservation:
    position: ArrayLike
    confidence: float
    diagnostics: TriangulationDiagnostics = field(default_factory=TriangulationDiagnostics)
    joint_id: str = ""
    person_id: str = ""
    world_timestamp: float = 0.0
    covariance: ArrayLike | None = None
    source: str = Source.MULTIVIEW.value

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance, dtype=np.float64).reshape(3, 3)

    @property
    def xyz(self) -> FloatArray:
        return self.position

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": [float(value) if np.isfinite(value) else None for value in self.position],
            "confidence": self.confidence,
            "diagnostics": self.diagnostics.to_dict(),
            "joint_id": self.joint_id,
            "person_id": self.person_id,
            "world_timestamp": self.world_timestamp,
            "covariance": self.covariance.tolist() if self.covariance is not None else None,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Joint3DObservation:
        values = dict(data)
        values["diagnostics"] = TriangulationDiagnostics.from_dict(values.get("diagnostics", {}))
        return cls(**values)


@dataclass
class JointTrajectory:
    joint_id: str
    times: ArrayLike
    positions: ArrayLike
    confidence: ArrayLike
    source: str = Source.MULTIVIEW.value
    covariance: ArrayLike | None = None


@dataclass
class ActorShape:
    actor_id: str
    model_type: str
    betas: ArrayLike
    measurements: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    scope: str = "take"
    source: str = Source.MULTIVIEW.value


@dataclass
class BodyPose:
    rotations: ArrayLike
    root_orientation: ArrayLike
    translation: ArrayLike
    world_timestamp: float
    confidence: float = 1.0
    source: str = Source.MULTIVIEW.value


@dataclass
class BodyModelState:
    shape: ActorShape
    pose: BodyPose
    joints: ArrayLike
    vertices: ArrayLike | None = None
    faces: ArrayLike | None = None
    skin_weights: ArrayLike | None = None
    joint_names: list[str] = field(default_factory=list)


@dataclass
class ContactState:
    joint_id: str
    world_timestamp: float
    probability: float
    planted: bool
    interval_id: int | None = None
    source: str = "geometric_contact"


@dataclass
class SceneConstraint:
    id: str
    kind: str
    parameters: dict[str, Any]
    source: str = Source.SURVEYED.value
    confidence: float = 1.0
    locked: bool = True


@dataclass
class SolveDiagnostics:
    objective_terms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0
    parameter_deltas: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReprojectionDiagnostics:
    per_camera: dict[str, dict[str, float]] = field(default_factory=dict)
    per_joint: dict[str, dict[str, float]] = field(default_factory=dict)
    per_time: dict[str, dict[str, float]] = field(default_factory=dict)
    rejected_measurements: list[dict[str, Any]] = field(default_factory=list)
