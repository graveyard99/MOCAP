# Body assets, persistent shape, and pose fitting

`openmocap.body_models.load_body_model` loads user-supplied SMPL-family numerical
assets. This repository contains an independent NumPy implementation of the
linear blend skinning equations, not SMPL's distributed implementation or its
licensed topology, coefficients, weights, regressors, or textures. Obtain the
appropriate model and usage rights separately from the model owner. Selecting
`smpl`, `smplh`, or `smplx` without a model path fails; it never changes to the
fixture automatically.

The supported safe asset contract is a non-object NPZ containing:

| Array | Shape | Meaning |
| --- | --- | --- |
| `v_template` | V × 3 | Template mesh in metres, +Y up |
| `f` | F × 3 | Triangle indices |
| `shapedirs` | V × 3 × B | Shape blend coefficients |
| `posedirs` | V × 3 × 9(J−1), or 3V × 9(J−1) | Rotation-matrix pose correctives |
| `J_regressor` | J × V | Dense joint regressor |
| `weights` | V × J | Nonnegative skin weights summing to one |
| `parents` | J | Parent indices; exactly one root −1 |
| `joint_names` | J Unicode strings | Optional; standard SMPL names otherwise |

`kintree_table` (2 × J) can replace `parents`. A transposed flattened `posedirs`
array is accepted. Sparse official regressors must be converted to dense for
safe NPZ storage. The loader validates dimensions, finite coefficients, parent
graph, mesh indices, and skin weights. Source file paths are retained in fit
diagnostics together with the SHA-256 asset hash. The loader uses the first
10 shape directions by default (`num_betas` can select another positive count).
This avoids treating SMPL-X's appended expression directions as persistent
actor shape; SMPL-X permits at most 300 shape directions. Files containing fewer
directions retain their actual count. Full licensed assets remain unverified
until supplied by their owner.

Official `.pkl` files require explicit `trust_pickle=True`. Unpickling can
execute code, so use this option only for an asset whose origin you trust. Old
assets containing serialized Chumpy objects may additionally require Chumpy to
convert; this package does not silently install it or execute an untrusted file.
Users should convert these assets in a separately trusted environment. Do not
commit the resulting numerical asset. Full joint rotations are supported for
SMPL-H/X; hand-PCA convenience parameters, expression fitting, and face-specific
measurement correspondence are incomplete. Actual licensed assets were not
available for validation during this build.

## Fitting API

```python
from openmocap.fitting import fit_actor

animation = fit_actor(times, measured_joints, confidence,
                      model="smpl", model_path="/installation/models/body.npz")
```

Input joints use metres and world +Y up. Supply `joint_names` to map an explicit
observation skeleton into the asset's full joint slots. Missing temporal samples
are interpolated without modifying the original observations. Entirely unobserved
asset joints receive the model's neutral rest initialization at the measured
root, zero measurement confidence, and an explicit `template_initialized_joints`
diagnostic. A measured or explicitly initialized root is required; missing
metric trajectory is never invented. These inferred joints do not contribute
to shape evidence. `openmocap.retargeting.map_joints` separately provides explicit
name mapping and leaves unmapped joints as NaN, making missing evidence visible.

Shape is fitted once across the take from median observed bone lengths, with
bounded beta values and a small shape regularizer. The fixed fitted mesh and
fixed rest joints then participate in every pose solve. Shape is not fitted per
frame. The current fitting objective uses confidence-weighted 3D joint errors,
optional direct multiview 2D reprojection through the calibrated distortion model,
and a tiny rotational initialization term to resolve unobservable twist. Supply
`cameras`, `observations2d[T,N,J,2]`, and optional `confidence2d[T,N,J]` to enable
the 2D term. Its explicit `reprojection_weight` (default 0.001 metres per pixel)
converts image errors into a comparable residual scale, multiplied by detection
and calibration confidence. The per-joint 2D contribution is normalized across
participating cameras so adding views does not silently multiply its trust.
Image measurements exceeding `reprojection_outlier_px` (12 px by default) from
strong 3D evidence are rejected before optimization and counted in diagnostics.
Per-frame objective contributions are recorded.
Silhouettes, anatomical joint limits, hand/face-specific correspondence, and
learned temporal priors are not part of this fitter yet. Diagnostics explicitly
record these limitations.

For contact/physics refinement, pass `reference_animation=original_animation`.
The fitter then holds the original actor shape, rest joints, mesh, skin weights,
hierarchy, and pose-corrective coefficients unchanged and fits only the new
rotations/root motion. The reference contains enough rig data to re-fit without
opening the original licensed asset. This avoids changing actor proportions
when a later optional stage adjusts uncertain observations.

Pose fitting performs rigid inverse kinematics over root translation and local
axis-angle rotations. A learned initialization supplied through `prior_joints`
can fill measurements only where confidence is below 0.2. Strong observations
are never blended with that prior. Camera parameters are consumed only for
projection and are never optimized or written by the body fitter. Highly inconsistent
measured bone lengths can still create a fitting residual because a persistent
rig cannot stretch bones independently per frame. To prevent silently moving
authoritative measured joints, the fit fails with the joint, time, and distance
if confidence is at least 0.85 and displacement exceeds 0.03 metres. Configure
`strong_evidence_confidence` and `strong_evidence_max_displacement_m` explicitly
for a production capture's measurement uncertainty. Setting the bound to `None`
deliberately disables this fail-fast check; the maximum remains in diagnostics.
Inspect proportions, joint mapping and reconstruction before relaxing it.

`Animation` contains fixed rest mesh, faces, skin weights, rest joints,
hierarchy, names, shape, sampled local rotations, metric root translations,
actual FK joint positions, timestamps, pose-corrective coefficients, and
diagnostics. NPZ serialization is lossless numerical data with Unicode metadata
and does not enable pickle.

The saved translations are metric world pelvis positions. Official SMPL-family
`transl` instead offsets the shaped template: after aligning conventions, compute
`transl = world_pelvis_position - shaped_rest_pelvis`. Passing saved pelvis
positions directly as official `transl` would introduce a template-root offset.

## Explicit technical mannequin

The deterministic demo uses `model="fixture"`. Its visible mesh consists of
procedural cylinders along the fitted skeleton. Every vertex has persistent
normalized skin weights, every frame uses the same topology, and shape is a
single vector of metric bone lengths. It is a reproducible technical mannequin
for geometry/rig/export validation, **not SMPL, a fitted human surface, or a
licensed-asset substitute**. Generated geometry belongs to this project and can
be redistributed under its code license.
