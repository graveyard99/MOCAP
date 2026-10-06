"""Acceptance scenes isolate geometric accuracy from neural inference quality."""

from copy import deepcopy
import json

import numpy as np
import pytest

from openmocap.calibration import (
    CameraRefinementConfig,
    prefer_authoritative_camera,
    refine_camera,
    surveyed_pnp,
)
from openmocap.cameras import look_at
from openmocap.triangulation import TriangulationConfig, triangulate
from openmocap.types import (
    Camera,
    CameraIntrinsics,
    CameraPose,
    CameraTrajectory,
    Pose2DObservation,
    TimeMapping,
)


def cameras_on_arc(count: int) -> list[Camera]:
    intrinsics = CameraIntrinsics(
        1300, 1290, 960, 540, 1920, 1080, [0.03, -0.005, 0.0003, -0.0002, 0.001]
    )
    return [
        look_at(f"CAM_{i:02d}", [5 * np.sin(theta), 2.2, 5 * np.cos(theta)], [0, 1, 0], intrinsics)
        for i, theta in enumerate(np.linspace(-1.3, 1.3, count))
    ]


def observations(
    cameras: list[Camera], point: np.ndarray, noise: float = 0.0
) -> list[Pose2DObservation]:
    rng = np.random.default_rng(7)
    return [
        Pose2DObservation(
            c.id, 0.517, "actor", "wrist", c.project(point) + rng.normal(0, noise, 2), 0.95, 0.517
        )
        for c in cameras
    ]


@pytest.mark.parametrize("count", [2, 3, 8, 30])
def test_noiseless_n_view(count: int) -> None:
    cameras = cameras_on_arc(count)
    point = np.array([0.37, 1.43, -0.28])
    solved = triangulate(cameras, observations(cameras, point))
    assert np.linalg.norm(solved.position - point) < 1e-8
    assert solved.diagnostics.num_views == count
    assert solved.diagnostics.mean_reprojection_error < 1e-6
    assert not solved.diagnostics.degenerate
    assert solved.covariance.shape == (3, 3)
    assert np.linalg.eigvalsh(solved.covariance).min() > 0


@pytest.mark.parametrize("count", [3, 8, 30])
def test_pixel_noise_bounded(count: int) -> None:
    cameras = cameras_on_arc(count)
    point = np.array([0.37, 1.43, -0.28])
    solved = triangulate(cameras, observations(cameras, point, 1.0))
    assert np.linalg.norm(solved.position - point) < 0.015
    assert solved.diagnostics.mean_reprojection_error < 3


def test_multiple_bad_views_rejected() -> None:
    cameras = cameras_on_arc(30)
    point = np.array([0.37, 1.43, -0.28])
    obs = observations(cameras, point, 0.35)
    bad = [1, 6, 19, 23, 27]
    for index in bad:
        obs[index].xy += [180, -130]
    solved = triangulate(cameras, obs)
    assert np.linalg.norm(solved.position - point) < 0.005
    assert set(solved.diagnostics.rejected_camera_ids) == {cameras[i].id for i in bad}
    assert solved.diagnostics.num_views == 25


def test_missing_and_disabled_views() -> None:
    cameras = cameras_on_arc(30)
    point = np.array([0.1, 0.8, -0.1])
    obs = observations(cameras, point)
    obs = [o for i, o in enumerate(obs) if i % 3]
    obs[0].confidence = 0.01
    obs[1].enabled = False
    obs[2].xy[:] = np.nan
    cameras[4].enabled = False
    solved = triangulate(cameras, obs)
    assert np.linalg.norm(solved.position - point) < 1e-8
    assert 2 <= solved.diagnostics.num_views < len(obs)


def test_degenerate_baseline_is_reported() -> None:
    intrinsics = CameraIntrinsics(1000, 1000, 500, 500, 1000, 1000)
    cameras = [
        Camera(str(i), intrinsics, CameraPose(np.eye(3), [-i * 1e-5, 0, 0])) for i in range(3)
    ]
    point = np.array([0.2, 0.3, 5.0])
    solved = triangulate(cameras, observations(cameras, point))
    assert solved.diagnostics.degenerate
    assert solved.confidence <= 0.05
    assert solved.diagnostics.max_ray_angle_degrees < 0.01
    assert solved.diagnostics.warnings


def test_metric_displacement_is_preserved() -> None:
    cameras = cameras_on_arc(8)
    a, b = np.array([-0.5, 1.0, 0]), np.array([1.93, 1.0, 0])
    start = triangulate(cameras, observations(cameras, a)).position
    end = triangulate(cameras, observations(cameras, b)).position
    assert np.linalg.norm(end - start) == pytest.approx(2.43, abs=1e-9)


def test_cross_person_and_unknown_camera_fail_loudly() -> None:
    cameras = cameras_on_arc(3)
    obs = observations(cameras, np.array([0, 1, 0]))
    obs[1].person_id = "another_actor"
    with pytest.raises(ValueError, match="Cross-person"):
        triangulate(cameras, obs)
    obs[1].person_id = "actor"
    obs[1].camera_id = "missing"
    with pytest.raises(ValueError, match="unknown cameras"):
        triangulate(cameras, obs)


def test_insufficient_observations_preserve_uncertainty() -> None:
    cameras = cameras_on_arc(3)
    solved = triangulate(cameras, observations(cameras, np.array([0, 1, 0]))[:1])
    assert solved.confidence == 0
    assert np.isnan(solved.position).all()
    assert solved.diagnostics.degenerate
    # Missing geometry remains missing in strict JSON instead of invalid NaN/Inf.
    encoded = json.dumps(solved.to_dict(), allow_nan=False)
    assert json.loads(encoded)["position"] == [None, None, None]
    restored = type(solved).from_dict(json.loads(encoded))
    assert restored.confidence == 0
    assert np.isnan(restored.position).all()


def test_authoritative_camera_unchanged_by_refinement_and_learned_estimate() -> None:
    camera = cameras_on_arc(2)[0]
    before = deepcopy(camera.to_dict())
    rng = np.random.default_rng(17)
    points = rng.uniform([-0.8, 0.2, -0.7], [0.8, 1.8, 0.7], (20, 3))
    updated, diagnostics = refine_camera(camera, points, camera.project(points) + [50, 20])
    assert camera.to_dict() == before
    assert updated.to_dict() == before
    assert diagnostics.parameter_deltas["translation_metres"] == 0
    learned = cameras_on_arc(2)[1]
    learned.source = "learned"
    assert prefer_authoritative_camera(camera, learned) is camera
    with pytest.raises(PermissionError, match="locked"):
        camera.set_pose(learned.pose)


def test_bounded_refinement_reports_deltas_without_mutating_input() -> None:
    true = cameras_on_arc(2)[0]
    uncertain = deepcopy(true)
    uncertain.pose.translation += [0.015, -0.01, 0.012]
    uncertain.locked = False
    points = np.random.default_rng(4).uniform([-0.8, 0.2, -0.7], [0.8, 1.8, 0.7], (30, 3))
    before = deepcopy(uncertain.to_dict())
    config = CameraRefinementConfig(max_translation_metres=0.02)
    solved, diagnostics = refine_camera(uncertain, points, true.project(points), config=config)
    assert uncertain.to_dict() == before
    assert (
        diagnostics.objective_terms["reprojection_after_px"]
        < diagnostics.objective_terms["reprojection_before_px"] * 0.3
    )
    assert np.max(np.abs(solved.pose.translation - uncertain.pose.translation)) <= 0.020001


def test_surveyed_pnp_recovers_metric_camera() -> None:
    camera = cameras_on_arc(3)[0]
    points = np.random.default_rng(47).uniform([-0.8, 0.1, -0.7], [0.8, 1.9, 0.7], (30, 3))
    pixels = camera.project(points)
    pixels[4] += [130, -90]
    solved, diagnostics = surveyed_pnp(camera.id, points, pixels, camera.intrinsics)
    assert np.linalg.norm(solved.pose.centre - camera.pose.centre) < 1e-5
    assert diagnostics["rejected"] == [4]
    assert solved.locked


def test_zero_baseline_does_not_fake_success() -> None:
    camera = cameras_on_arc(3)[0]
    copied = deepcopy(camera)
    copied.id = "copy"
    solved = triangulate([camera, copied], observations([camera, copied], np.array([0, 1.0, 0])))
    assert solved.diagnostics.degenerate
    assert solved.confidence <= 0.05


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        TriangulationConfig(minimum_views=1)


def test_async_moving_camera_uses_each_world_timestamp() -> None:
    cameras = cameras_on_arc(3)
    point = np.array([0.2, 1.2, -0.3])
    obs = []
    for i, camera in enumerate(cameras):
        centre = camera.pose.centre
        end = look_at(camera.id, centre + [0.4, 0.1, 0.2], [0, 1, 0], camera.intrinsics).pose
        camera.trajectory = CameraTrajectory([0.0, 1.0], [camera.pose, end])
        camera.time_mapping = TimeMapping(1.0002, 0.037 * i)
        world_time = 0.45 + 0.015 * i
        camera_time = float(camera.time_mapping.to_camera(world_time))
        obs.append(
            Pose2DObservation(
                camera.id, camera_time, "actor", "wrist", camera.project(point, world_time)
            )
        )
    solved = triangulate(cameras, obs)
    assert np.linalg.norm(solved.position - point) < 1e-8
    assert solved.diagnostics.num_views == 3
