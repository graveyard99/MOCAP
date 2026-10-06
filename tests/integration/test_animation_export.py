import json
import shutil

import numpy as np
import pytest

from openmocap.export import export_animation
from openmocap.body_models import load_body_model
from openmocap.fitting import fit_actor, load_animation
from openmocap.synthetic.capture import human_motion


def test_npz_and_bvh_exports(tmp_path):
    times = np.arange(4) / 30
    joints, _, _ = human_motion(times)
    animation = fit_actor(times, joints)
    report = export_animation(animation, tmp_path / "actor.npz")
    assert report["validated"]
    assert np.allclose(load_animation(tmp_path / "actor.npz").joints, animation.joints)
    bvh = export_animation(animation, tmp_path / "actor.bvh")
    assert bvh["skeleton"] and not bvh["skinning"]
    assert "Frames: 4" in (tmp_path / "actor.bvh").read_text()
    assert "ROOT pelvis" in (tmp_path / "actor.bvh").read_text()


@pytest.mark.skipif(not shutil.which("blender"), reason="External Blender backend unavailable")
def test_fbx_contains_animated_skinned_metric_character(tmp_path):
    times = np.arange(4) / 30
    joints, _, _ = human_motion(times)
    animation = fit_actor(times, joints)
    report = export_animation(animation, tmp_path / "actor.fbx")
    assert report["validated"]
    assert report["bone_count"] == 24
    assert report["weighted_vertices"] == len(animation.vertices)
    assert report["frame_range"] == [1, 4]
    assert report["reimport_joint_max_error_m"] < 0.0002
    assert report["reimport_skinning_max_error_m"] < 0.0005
    assert np.linalg.norm(np.subtract(*report["root_positions"])) > 0.1
    stored = json.loads((tmp_path / "actor.fbx.json").read_text())
    assert stored["validation"] == "clean Blender FBX re-import"


@pytest.mark.skipif(not shutil.which("blender"), reason="External Blender backend unavailable")
def test_body_pose_correctives_survive_fbx_roundtrip(tmp_path):
    # Invented tiny numerical asset. No licensed body topology or coefficients.
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 1.0, 0.0], [0.0, 2.0, 0.0]])
    weights = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    asset = tmp_path / "numerical_fixture.npz"
    np.savez(
        asset,
        v_template=vertices,
        f=np.array([[0, 1, 2], [1, 3, 2]]),
        shapedirs=np.zeros((4, 3, 1)),
        posedirs=np.random.default_rng(0).normal(0, 0.001, (4, 3, 18)),
        J_regressor=np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]]),
        weights=weights,
        parents=np.array([-1, 0, 1]),
        joint_names=np.array(["pelvis", "middle", "tip"]),
    )
    body = load_body_model(asset)
    times = np.arange(4) / 30
    poses = np.zeros((4, 3, 3))
    poses[:, 1, 0] = [0.1, 0.2, 0.3, 0.4]
    truth = np.array(
        [
            body.evaluate(np.zeros(1), pose, np.array([time, 1.0, 0.0]))[1]
            for time, pose in zip(times, poses, strict=True)
        ]
    )
    animation = fit_actor(times, truth, model="smpl", model_path=asset)
    report = export_animation(animation, tmp_path / "corrective_character.fbx")
    assert report["validated"] and report["pose_correctives"]
    assert report["reimport_skinning_max_error_m"] < 0.0005


@pytest.mark.parametrize("fps", [24000 / 1001, 60000 / 1001])
def test_fractional_export_sampling_retains_rig_and_metric_root(tmp_path, fps):
    from openmocap.export.resample import resample_animation

    times = np.arange(10) / 30
    truth, _, _ = human_motion(times)
    animation = fit_actor(times, truth)
    sampled = resample_animation(animation, fps, start=0.05, end=0.25)
    assert np.allclose(np.diff(sampled.times), 1 / fps)
    assert np.array_equal(sampled.shape, animation.shape)
    assert np.array_equal(sampled.weights, animation.weights)
    assert sampled.times[0] >= 0.05 and sampled.times[-1] <= 0.25
    report = export_animation(sampled, tmp_path / "fractional.npz", fps=fps)
    assert report["validated"] and report["fps"] == fps
