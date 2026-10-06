import json

import numpy as np
import yaml
from scipy.interpolate import CubicSpline

from openmocap.synthetic import JOINT_NAMES, generate_capture, human_motion
from openmocap.trajectories import fit_trajectory
from openmocap.triangulation import triangulate
from openmocap.types import Camera, Pose2DObservation


def test_async_moving_camera_to_continuous_metric_motion(tmp_path):
    path = generate_capture(
        tmp_path,
        camera_count=3,
        frames=20,
        fps=30,
        moving=True,
        render_images=False,
        noise_px=0,
        missing_probability=0,
        outlier_probability=0,
    )
    project = yaml.safe_load(path.read_text())
    cameras = [Camera.from_dict(camera) for camera in project["cameras"]]
    observations = [
        Pose2DObservation.from_dict(row)
        for row in json.loads((tmp_path / "observations.json").read_text())
    ]
    selected = ["pelvis", "left_wrist"]
    times = np.linspace(0.08, 0.53, 16)
    positions = np.empty((len(times), 2, 3))
    confidence = np.empty((len(times), 2))
    for joint_index, joint_name in enumerate(selected):
        interpolators = {}
        for camera in cameras:
            measured = [
                o for o in observations if o.camera_id == camera.id and o.joint_id == joint_name
            ]
            world_times = camera.time_mapping.to_world([o.camera_timestamp for o in measured])
            interpolators[camera.id] = CubicSpline(world_times, [o.xy for o in measured])
        for index, time in enumerate(times):
            derived = [
                Pose2DObservation(
                    camera.id,
                    float(camera.time_mapping.to_camera(time)),
                    "actor01",
                    joint_name,
                    interpolators[camera.id](time),
                    0.92,
                    float(time),
                    "derived_temporal_interpolation",
                )
                for camera in cameras
            ]
            solved = triangulate(cameras, derived, time=float(time))
            positions[index, joint_index] = solved.position
            confidence[index, joint_index] = solved.confidence
            assert solved.diagnostics.num_views == 3
    spline = fit_trajectory(times, positions, confidence, smoothing=1e-8, joint_names=selected)
    query = np.linspace(0.08, 0.53, 60)
    truth, _, _ = human_motion(query, 19 / 30)
    expected = truth[:, [JOINT_NAMES.index(name) for name in selected]]
    assert np.max(np.linalg.norm(spline.evaluate(query) - expected, axis=2)) < 0.003
    assert (
        np.max(np.linalg.norm(positions[:, 0] - human_motion(times, 19 / 30)[0][:, 0], axis=1))
        < 0.001
    )
