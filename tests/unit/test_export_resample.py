import numpy as np

from openmocap.export.resample import resample_animation
from openmocap.fitting import fit_actor
from openmocap.synthetic.capture import JOINT_NAMES, PARENTS, human_motion


def test_export_uniform_rate_range_shape_topology_and_fk():
    times = np.arange(12) / 30
    joints, _, _ = human_motion(times, 1)
    source = fit_actor(times, joints, joint_names=list(JOINT_NAMES), parents=PARENTS)
    result = resample_animation(source, 59.94, start=0.04, end=0.29)
    assert np.allclose(np.diff(result.times), 1 / 59.94)
    assert result.times[0] >= 0.04 and result.times[-1] <= 0.29
    assert np.array_equal(source.shape, result.shape)
    assert np.array_equal(source.faces, result.faces)
    assert np.array_equal(source.weights, result.weights)
    assert np.isfinite(result.joints).all()
    assert np.max(np.abs(result.joints[:, 0] - result.translations)) < 1e-8
