import numpy as np
import pytest

from openmocap.cameras import look_at
from openmocap.sync import refine_multiview_timing
from openmocap.trajectories import fit_trajectory
from openmocap.types import CameraIntrinsics, Pose2DObservation, TimeMapping


def test_native_plate_clock_refinement_preserves_camera_geometry_and_locks():
    times = np.arange(0, 2, 0.02)
    positions = np.column_stack([0.4 * times, 0.9 + 0.2 * np.sin(times * 2), 0.1 * np.cos(times)])[
        :, None
    ]
    trajectory = fit_trajectory(times, positions, smoothing=0, joint_names=["wrist"])
    intrinsics = CameraIntrinsics(800, 800, 640, 360, 1280, 720)
    reference = look_at("reference", [0, 1, 4], [0, 1, 0], intrinsics)
    uncertain = look_at("uncertain", [3, 1, 2], [0, 1, 0], intrinsics)
    uncertain.time_mapping = TimeMapping(offset=0, locked=False, confidence=0.6)
    reference_before = reference.pose.to_dict()
    uncertain_before = uncertain.pose.to_dict()
    camera_times = np.arange(0.1, 1.8, 0.033)
    observations = [
        Pose2DObservation(
            "uncertain",
            t,
            "actor",
            "wrist",
            uncertain.project(trajectory.evaluate(t + 0.014)[0]),
            0.95,
        )
        for t in camera_times
    ]
    result = refine_multiview_timing(
        [reference, uncertain], observations, trajectory, ["wrist"], offset_prior_sigma=0.1
    )
    assert result.mappings["uncertain"].offset == pytest.approx(0.014, abs=0.001)
    assert result.diagnostics["uncertain"]["after_median_reprojection_px"] < 0.02
    assert result.diagnostics["uncertain"]["before_median_reprojection_px"] > 0.2
    assert result.mappings["reference"] is reference.time_mapping
    assert uncertain.time_mapping.offset == 0
    assert uncertain.pose.to_dict() == uncertain_before
    assert reference.pose.to_dict() == reference_before
    assert result.mappings["uncertain"].confidence == 0.6


def test_time_gauge_requires_authoritative_reference():
    times = np.arange(10) / 10
    trajectory = fit_trajectory(times, np.ones((10, 1, 3)))
    camera = look_at(
        "camera", [0, 1, 4], [0, 1, 0], CameraIntrinsics(800, 800, 640, 360, 1280, 720)
    )
    camera.time_mapping.locked = False
    with pytest.raises(ValueError, match="reference camera"):
        refine_multiview_timing([camera], [], trajectory, ["wrist"])
