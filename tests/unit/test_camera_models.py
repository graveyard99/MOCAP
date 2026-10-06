import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openmocap.cameras import distort, look_at
from openmocap.geometry import (
    align_similarity,
    convert_points,
    convert_rotation,
    scale_from_distance,
)
from openmocap.types import (
    Camera,
    CameraIntrinsics,
    CameraPose,
    CameraTrajectory,
    Pose2DObservation,
    TimeMapping,
)


@pytest.mark.parametrize(
    "model,coefficients",
    [
        ("none", []),
        ("opencv", [0.1, -0.04, 0.002, -0.001, 0.01]),
        ("opencv", [0.1, -0.04, 0.002, -0.001, 0.01, 0.001, -0.002, 0.001]),
        ("fisheye", [0.03, -0.02, 0.001, 0.001]),
    ],
)
def test_distortion_round_trip(model: str, coefficients: list[float]) -> None:
    intrinsics = CameraIntrinsics(900, 920, 640, 360, 1280, 720, coefficients, model)
    normalized = np.random.default_rng(1).uniform(-0.45, 0.45, (100, 2))
    pixels = distort(normalized, intrinsics)
    camera = Camera("camera", intrinsics, CameraPose(np.eye(3), np.zeros(3)))
    assert np.max(np.abs(camera.undistort(pixels) - normalized)) < 1e-9
    points = np.column_stack([normalized, np.ones(len(normalized))])
    assert np.allclose(camera.project(points), pixels)


def test_camera_world_projection_and_behind_camera() -> None:
    camera = look_at(
        "front", [0, 1, 5], [0, 1, 0], CameraIntrinsics(1000, 1000, 500, 300, 1000, 600)
    )
    assert np.allclose(camera.project([0, 1, 0]), [500, 300])
    assert camera.project([1, 1, 0])[0] > 500
    assert camera.project([0, 2, 0])[1] < 300
    assert np.isnan(camera.project([0, 1, 6])).all()


@pytest.mark.parametrize("convention", ["opencv", "smpl", "blender", "maya", "fbx", "houdini"])
def test_coordinate_roundtrip(convention: str) -> None:
    points = np.random.default_rng(9).normal(size=(20, 3))
    converted = convert_points(points, "world", convention, "metres", "centimetres")
    restored = convert_points(converted, convention, "world", "centimetres", "metres")
    assert np.allclose(restored, points)
    r = Rotation.from_rotvec([0.2, 0.1, -0.3]).as_matrix()
    assert np.allclose(
        convert_rotation(convert_rotation(r, "world", convention), convention, "world"), r
    )
    assert np.allclose(convert_points([[0, 1, 0]], "world", "blender"), [[0, 0, 1]])


def test_unknown_convention_fails() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        convert_points([1, 2, 3], "unspecified", "world")


def test_moving_camera_interpolation_and_dynamic_lens() -> None:
    a = CameraPose(np.eye(3), [0, 0, 0])
    r = Rotation.from_rotvec([0, 0.4, 0]).as_matrix()
    b = CameraPose(r, -r @ np.array([2.0, 0, 0]))
    lens_a = CameraIntrinsics(1000, 1000, 500, 300, 1000, 600)
    lens_b = CameraIntrinsics(1200, 1200, 500, 300, 1000, 600)
    trajectory = CameraTrajectory([0, 2], [a, b], [lens_a, lens_b])
    camera = Camera("moving", lens_a, a, trajectory)
    assert np.allclose(camera.pose_at(1).centre, [1, 0, 0])
    assert np.allclose(camera.pose_at(1).rotation, Rotation.from_rotvec([0, 0.2, 0]).as_matrix())
    assert camera.intrinsics_at(1).fx == 1100
    assert np.allclose(
        camera.project([1, 0, 5], 1),
        Camera("s", camera.intrinsics_at(1), camera.pose_at(1)).project([1, 0, 5]),
    )
    with pytest.raises(ValueError, match="outside tracked"):
        camera.pose_at(3)
    restored = Camera.from_dict(camera.to_dict())
    assert np.allclose(restored.project([1, 0, 5], 1), camera.project([1, 0, 5], 1))


def test_similarity_metric_alignment() -> None:
    source = np.random.default_rng(2).normal(size=(30, 3))
    rotation = Rotation.from_rotvec([0.1, 0.3, -0.5]).as_matrix()
    target = 2.4 * source @ rotation.T + [1, 2, 3]
    scale, r, t = align_similarity(source, target)
    assert scale == pytest.approx(2.4)
    assert np.allclose(r, rotation)
    assert np.allclose(t, [1, 2, 3])
    assert scale_from_distance([0, 0, 0], [2, 0, 0], 4.86) == pytest.approx(2.43)
    with pytest.raises(ValueError, match="collinear"):
        align_similarity([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 0, 0], [2, 0, 0]])


def test_pose_metadata_and_time_mapping_roundtrip() -> None:
    observation = Pose2DObservation.from_dict(
        {
            "camera_id": "A",
            "camera_timestamp": 1.0,
            "person_id": "P",
            "joint_id": "wrist",
            "x": 10.0,
            "y": 20.0,
            "confidence": 0.8,
            "frame_index": 24,
            "synthetic_outlier": False,
        }
    )
    assert observation.metadata["frame_index"] == 24
    restored = Pose2DObservation.from_dict(observation.to_dict())
    assert np.allclose(restored.xy, observation.xy)
    mapping = TimeMapping(1.00012, 0.035)
    times = np.array([0.0, 1.0, 100.0])
    assert np.allclose(mapping.to_camera(mapping.to_world(times)), times)


def test_intrinsics_metadata_and_validation() -> None:
    camera = CameraIntrinsics.from_metadata(1920, 1080, 50, 36, 20.25)
    assert camera.fx == pytest.approx(50 / 36 * 1920)
    assert camera.fy == pytest.approx(camera.fx)
    with pytest.raises(ValueError, match="four"):
        CameraIntrinsics(1, 1, 0, 0, 10, 10, [], "fisheye")
    with pytest.raises(ValueError, match="orthonormal"):
        CameraPose(np.ones((3, 3)), [0, 0, 0])
