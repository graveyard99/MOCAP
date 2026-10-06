# Continuous-time motion and measurement precedence

`fit_trajectory(times, positions, confidence)` consumes world seconds and metric
`N × J × 3` joint positions. Missing points remain NaN. Raw triangulation arrays
are never modified. Per-joint duplicate timestamps are merged by weighted mean
only in the derived spline representation.

The implementation uses weighted cubic smoothing splines with configurable
regularization `smoothing`. Higher confidence increases observation weight.
For each joint, smoothing is reduced until samples above `strong_threshold`
move by no more than `max_strong_displacement_m` (default 2 mm). If no smoothing
strength satisfies the guard, interpolation preserves measured samples. This
is an explicit numerical enforcement of measured geometry outranking temporal
plausibility. It is not a statistical outlier filter: reject erroneous multiview
reconstructions before fitting a spline.

Evaluate positions or derivatives at arbitrary timestamps, including camera
observation times and custom output FPS:

```python
from openmocap.trajectories import fit_trajectory

trajectory = fit_trajectory(world_times, joints_m, joint_confidence,
                            smoothing=1e-5, joint_names=joint_names)
positions = trajectory.evaluate([0.025, 0.043])
velocities = trajectory.evaluate([0.025, 0.043], derivative=1)
times_23976, animation = trajectory.sample(24000 / 1001)
```

No extrapolation is allowed by default. Queries beyond measured support yield
NaN rather than unmarked invented positions. Confidence is interpolated from
observed evidence and decays in gaps beyond twice the median sample spacing.
This confidence is a support score, not an exact posterior covariance. The
curve can span an occlusion; its reduced confidence allows downstream priors
or optional kinematic constraints to act without redefining reliable samples.

Diagnostics record requested/effective smoothing, observation counts and the
largest displacement of strong evidence. Confidence-aware smoothing operates
on independent joints and does not guarantee exact bone lengths; persistent
body shape and conventional rig fitting enforce that subsequent constraint.

`trajectory.save("continuous.npz")` checkpoints the exact spline coefficients,
knot times, source confidence and diagnostic settings. `ContinuousTrajectory.load`
restores arbitrary-time evaluation and derivatives without refitting. Serialization
uses versioned NPZ arrays with `allow_pickle=False`; incompatible schemas fail.
The surrounding pipeline cache records input/config/code hashes separately.

Tests check noisy/irregular timestamp support, unseen interpolation times,
fractional output FPS, missing joints and evidence-displacement governance.
