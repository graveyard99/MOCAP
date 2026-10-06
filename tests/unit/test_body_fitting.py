"""Licensed-asset-free tests of the actual numerical body and rig contract."""

from pathlib import Path

import numpy as np
import pytest

from openmocap.body_models import ModelAssetError, load_body_model
from openmocap.body_models.model import forward_kinematics
from openmocap.fitting import fit_actor, load_animation
from openmocap.retargeting import map_joints
from openmocap.synthetic.capture import human_motion
from openmocap.cameras import look_at
from openmocap.types import CameraIntrinsics


def numerical_asset(path: Path) -> Path:
    """Tiny invented coefficient asset; not SMPL topology or licensed data."""
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 1.0, 0.0], [0.0, 2.0, 0.0]])
    regressor = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])
    weights = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    shapedirs = np.zeros((4, 3, 1))
    shapedirs[:, 1, 0] = vertices[:, 1] * 0.1
    np.savez(
        path,
        v_template=vertices,
        f=np.array([[0, 1, 2], [1, 3, 2]]),
        shapedirs=shapedirs,
        posedirs=np.zeros((4, 3, 18)),
        J_regressor=regressor,
        weights=weights,
        parents=np.array([-1, 0, 1]),
        joint_names=np.array(["pelvis", "middle", "tip"]),
    )
    return path


def test_safe_asset_lbs_and_pose(tmp_path):
    body = load_body_model(numerical_asset(tmp_path / "coefficients.npz"))
    vertices, rest = body.rest(np.array([1.0]))
    assert np.isclose(rest[2, 1], 2.2)
    pose = np.zeros((3, 3))
    pose[0, 2] = np.pi / 2
    mesh, joints = body.evaluate(np.array([1.0]), pose, np.array([2.0, 0, 0]))
    assert np.allclose(joints[2], [-0.2, 0, 0], atol=1e-10)
    assert np.allclose(mesh[3], joints[2])
    assert len(mesh) == len(vertices)


def test_untrusted_pickle_and_missing_asset_fail_loudly(tmp_path):
    path = tmp_path / "body.pkl"
    path.write_bytes(b"not a real pickle")
    with pytest.raises(ModelAssetError, match="explicit trust"):
        load_body_model(path)
    with pytest.raises(ModelAssetError, match="Missing licensed"):
        load_body_model(tmp_path / "absent.npz")
    with pytest.raises(ValueError, match="licensed model_path"):
        fit_actor(np.array([0.0]), np.zeros((1, 24, 3)), model="smpl")


def test_persistent_shape_metric_motion_and_save(tmp_path):
    times = np.linspace(0, 2, 9)
    truth, _, root = human_motion(times)
    animation = fit_actor(times, truth)
    assert animation.model_name == "fixture"
    assert animation.shape.shape == (24,)
    assert animation.diagnostics["shape_constant"]
    assert np.max(np.linalg.norm(animation.joints - truth, axis=2)) < 1e-4
    assert np.allclose(animation.translations, root, atol=1e-4)
    assert np.isclose(animation.translations[-1, 0] - animation.translations[0, 0], 2.43, atol=1e-5)
    assert np.allclose(animation.weights.sum(axis=1), 1)
    for frame in [0, 4, 8]:
        solved, _ = forward_kinematics(
            animation.rest_joints,
            animation.parents,
            animation.rotations[frame],
            animation.translations[frame],
        )
        assert np.allclose(solved, animation.joints[frame])
        assert animation.mesh_at(frame).shape == animation.vertices.shape
    animation.save(tmp_path / "motion.npz")
    restored = load_animation(tmp_path / "motion.npz")
    assert np.allclose(restored.rotations, animation.rotations)
    assert restored.names == animation.names


def test_conflicting_weak_prior_cannot_move_strong_wrist():
    times = np.linspace(0, 0.4, 4)
    truth, _, _ = human_motion(times)
    conflict = truth.copy()
    conflict[:, 20] += np.array([0.8, -0.5, 0.5])
    animation = fit_actor(times, truth, prior_joints=conflict)
    assert animation.diagnostics["prior_used_observations"] == 0
    assert np.max(np.linalg.norm(animation.joints[:, 20] - truth[:, 20], axis=1)) < 1e-4


def test_explicit_mapping_keeps_unknown_joints_unknown():
    source = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    mapped, available = map_joints(
        source, ["hip", "hand"], ["pelvis", "hand", "nose"], {"pelvis": "hip"}
    )
    assert available.tolist() == [True, True, False]
    assert np.allclose(mapped[0, 0], source[0, 0])
    assert np.isnan(mapped[0, 2]).all()


def test_smpl_numerical_fixture_shape_fit(tmp_path):
    body = load_body_model(numerical_asset(tmp_path / "coefficients.npz"))
    times = np.array([0.0, 0.1, 0.2])
    beta = np.array([0.6])
    poses = np.zeros((3, 3, 3))
    poses[:, 1, 2] = [0.1, 0.3, 0.5]
    truth = np.array(
        [
            body.evaluate(beta, pose, np.array([time, 1.0, 0]))[1]
            for time, pose in zip(times, poses, strict=True)
        ]
    )
    result = fit_actor(times, truth, model="smpl", model_path=tmp_path / "coefficients.npz")
    assert np.allclose(result.shape, beta, atol=0.001)
    assert result.posedirs is not None
    assert np.max(np.linalg.norm(result.joints - truth, axis=2)) < 0.001


def test_direct_reprojection_term_preserves_locked_calibration():
    times = np.array([0.0, 0.1])
    truth, _, _ = human_motion(times)
    calibration = CameraIntrinsics(800.0, 800.0, 640.0, 360.0, 1280, 720)
    cameras = [
        look_at("A", [3.0, 2.0, 4.0], [0.0, 1.0, 0.0], calibration),
        look_at("B", [-3.0, 2.0, 3.0], [0.0, 1.0, 0.0], calibration),
    ]
    before = [(camera.pose.rotation.copy(), camera.pose.translation.copy()) for camera in cameras]
    pixels = np.array(
        [
            [camera.project(pose, time=float(time)) for camera in cameras]
            for time, pose in zip(times, truth, strict=True)
        ]
    )
    pixels += 0.1
    animation = fit_actor(times, truth, cameras=cameras, observations2d=pixels)
    assert animation.diagnostics["direct_2d_fit"] == "enabled"
    assert all(
        terms["multiview_2d_reprojection"] > 0 for terms in animation.diagnostics["objective_terms"]
    )
    assert np.max(np.linalg.norm(animation.joints - truth, axis=2)) < 0.001
    for camera, (rotation, translation) in zip(cameras, before, strict=True):
        assert camera.locked
        assert np.array_equal(camera.pose.rotation, rotation)
        assert np.array_equal(camera.pose.translation, translation)


def test_missing_model_slots_use_explicit_low_confidence_template(tmp_path):
    asset = numerical_asset(tmp_path / "body.npz")
    observations = np.array(
        [[[0.0, 1.0, 0.0], [0.0, 3.0, 0.0]], [[0.1, 1.0, 0.0], [0.1, 3.0, 0.0]]]
    )
    animation = fit_actor(
        np.array([0.0, 0.1]),
        observations,
        model="smpl",
        model_path=asset,
        joint_names=["pelvis", "tip"],
    )
    assert animation.names == ["pelvis", "middle", "tip"]
    assert animation.diagnostics["template_initialized_joints"] == ["middle"]
    assert np.allclose(animation.joints[:, [0, 2]], observations, atol=1e-4)


def test_family_names_include_hands_and_smplx_face_slots():
    from openmocap.body_models.model import family_joint_names

    smplh = family_joint_names("smplh", 52)
    smplx = family_joint_names("smplx", 55)
    assert smplh[20:24] == ["left_wrist", "right_wrist", "left_index1", "left_index2"]
    assert smplx[22:27] == ["jaw", "left_eye", "right_eye", "left_index1", "left_index2"]
    assert smplx[-1] == "right_thumb3"
    assert len(set(smplh)) == 52 and len(set(smplx)) == 55


def test_shape_selection_never_treats_expression_as_actor_shape(tmp_path):
    path = numerical_asset(tmp_path / "body.npz")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["shapedirs"] = np.repeat(arrays["shapedirs"], 400, axis=2)
    np.savez(path, **arrays)
    assert load_body_model(path, model_name="smplx").beta_count == 10
    with pytest.raises(ModelAssetError, match="expression"):
        load_body_model(path, model_name="smplx", num_betas=301)


def test_incompatible_proportions_fail_instead_of_moving_strong_geometry():
    # Three straight poses define a persistent 1m forearm; one deliberately
    # impossible 2m measurement cannot be satisfied by rigid body articulation.
    times = np.arange(4) / 30
    points = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]] * 4)
    points[-1, 2, 1] = 3.0
    with pytest.raises(ValueError, match="authoritative joint evidence"):
        fit_actor(times, points, parents=[-1, 0, 1], joint_names=["root", "elbow", "wrist"])
    result = fit_actor(times, points, parents=[-1, 0, 1], strong_evidence_max_displacement_m=None)
    assert result.diagnostics["strong_evidence_max_displacement_m"] > 0.3


def test_bad_2d_evidence_cannot_override_strong_3d():
    times = np.array([0.0, 0.1])
    truth, _, _ = human_motion(times)
    intrinsics = CameraIntrinsics(800.0, 800.0, 640.0, 360.0, 1280, 720)
    cameras = [
        look_at("A", [3.0, 2.0, 4.0], [0.0, 1.0, 0.0], intrinsics),
        look_at("B", [-3.0, 2.0, 3.0], [0.0, 1.0, 0.0], intrinsics),
    ]
    pixels = np.array(
        [
            [camera.project(pose, time=float(time)) for camera in cameras]
            for time, pose in zip(times, truth, strict=True)
        ]
    )
    pixels[:, :, 20] += 150
    result = fit_actor(
        times, truth, cameras=cameras, observations2d=pixels, reprojection_weight=0.1
    )
    assert result.diagnostics["reprojection_outliers_rejected"] == 4
    assert np.max(np.linalg.norm(result.joints[:, 20] - truth[:, 20], axis=1)) < 1e-4


def test_optional_refit_keeps_original_actor_shape_and_skin():
    times = np.arange(4) / 30
    truth, _, _ = human_motion(times)
    original = fit_actor(times, truth)
    adjusted = truth.copy()
    adjusted[:, 20:, 0] += 0.005
    refined = fit_actor(times, adjusted, reference_animation=original)
    assert np.array_equal(refined.shape, original.shape)
    assert np.array_equal(refined.rest_joints, original.rest_joints)
    assert np.array_equal(refined.vertices, original.vertices)
    assert np.array_equal(refined.weights, original.weights)
    assert refined.diagnostics["shape"]["locked_reference_shape"]


def test_reference_rig_refits_without_loading_licensed_asset_again(tmp_path):
    asset = numerical_asset(tmp_path / "body.npz")
    body = load_body_model(asset)
    times = np.array([0.0, 0.1])
    truth = np.array(
        [
            body.evaluate(np.zeros(1), np.zeros((3, 3)), np.array([time, 1.0, 0.0]))[1]
            for time in times
        ]
    )
    original = fit_actor(times, truth, model="smpl", model_path=asset)
    asset.unlink()
    result = fit_actor(times, truth, model="smpl", reference_animation=original)
    assert np.array_equal(result.shape, original.shape)
    assert np.allclose(result.joints, truth)


def test_named_fixture_never_guesses_an_incompatible_coco_hierarchy():
    times = np.array([0.0])
    with pytest.raises(ValueError, match="explicit parents"):
        fit_actor(times, np.ones((1, 17, 3)), joint_names=[f"coco_{i}" for i in range(17)])
