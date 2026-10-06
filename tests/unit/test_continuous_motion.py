import numpy as np
import pytest

from openmocap.trajectories import ContinuousTrajectory, fit_trajectory


def test_continuous_spline_asynchronous_and_unseen_times():
    rng = np.random.default_rng(42)
    times = np.sort(rng.uniform(0, 3, 160))

    def truth(t):
        return np.column_stack([0.2 * t, np.sin(t), 0.3 * np.cos(t)])

    positions = truth(times)[:, None]
    trajectory = fit_trajectory(times, positions, np.ones((len(times), 1)), smoothing=1e-6)
    query = np.linspace(times[0], times[-1], 400)
    assert np.max(np.linalg.norm(trajectory.evaluate(query)[:, 0] - truth(query), axis=1)) < 0.002
    assert (
        np.max(np.linalg.norm(trajectory.evaluate(times)[:, 0] - positions[:, 0], axis=1)) < 0.002
    )
    assert trajectory.evaluate([times[0] - 0.1]).shape == (1, 1, 3)
    assert np.isnan(trajectory.evaluate(times[0] - 0.1)).all()


def test_strong_geometry_dominates_smoothing_and_raw_unchanged():
    times = np.arange(30) / 30
    values = np.zeros((30, 1, 3))
    values[15, 0, 0] = 1.0
    before = values.copy()
    trajectory = fit_trajectory(times, values, np.ones((30, 1)), smoothing=1.0)
    assert abs(trajectory.evaluate(times[15])[0, 0] - 1) <= 0.002
    assert np.array_equal(values, before)
    assert trajectory.diagnostics["joints"][0]["effective_smoothing"] < 1


def test_duplicate_missing_and_fractional_fps():
    times = np.array([0, 0.1, 0.1, 0.3, 0.5, 0.7, 1])
    values = np.broadcast_to(times[:, None, None], (len(times), 2, 3)).copy()
    values[:, 1] = np.nan
    trajectory = fit_trajectory(times, values)
    sampled_times, sampled = trajectory.sample(23.976)
    assert len(sampled_times) == 24
    assert np.max(np.abs(sampled[:, 0, 0] - sampled_times)) < 1e-8
    assert np.isnan(sampled[:, 1]).all()
    with pytest.raises(ValueError):
        fit_trajectory(times, values, confidence=np.full((7, 2), 2))


def test_constant_body_bone_lengths_are_input_not_frame_shapes():
    from openmocap.synthetic import PARENTS, human_motion

    joints, _, _ = human_motion(np.linspace(0, 2, 61), 2)
    for joint, parent in enumerate(PARENTS):
        if parent >= 0:
            assert np.ptp(np.linalg.norm(joints[:, joint] - joints[:, parent], axis=1)) < 1e-12
    assert joints[-1, 0, 0] - joints[0, 0, 0] == pytest.approx(2.43)


def test_exact_trajectory_checkpoint_without_pickle(tmp_path):
    times = np.linspace(0, 2, 30)
    positions = np.stack(
        [np.column_stack([times, np.sin(times), np.cos(times)]), np.full((len(times), 3), np.nan)],
        axis=1,
    )
    fitted = fit_trajectory(times, positions, smoothing=1e-5)
    path = tmp_path / "continuous.npz"
    fitted.save(path)
    loaded = ContinuousTrajectory.load(path)
    query = np.linspace(0.03, 1.96, 111)
    assert np.allclose(fitted.evaluate(query), loaded.evaluate(query), equal_nan=True)
    assert np.allclose(
        fitted.evaluate(query, derivative=1), loaded.evaluate(query, derivative=1), equal_nan=True
    )
    assert np.array_equal(fitted.confidence(query), loaded.confidence(query))
    assert loaded.diagnostics == fitted.diagnostics
