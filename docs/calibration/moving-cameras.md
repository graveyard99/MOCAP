# Moving cameras

Moving camera tracks are supported in every projection/triangulation call.
Timestamps use **world seconds**, after the camera's affine clock mapping. Pose
samples use SLERP for world-to-camera rotations and linear camera-centre motion.
Lens samples may vary focal lengths, principal point and distortion coefficients
while retaining one image resolution and distortion model.

```python
from openmocap.types import CameraTrajectory
camera.trajectory = CameraTrajectory(times_world, surveyed_poses, intrinsics=lens_samples)
camera.project(joints_world, time=world_time)
```

Sparse tracking does not justify long-range extrapolation. Projection outside the
tracked range raises an error. Increase camera sample density for fast rotations,
handheld shots or sudden zooms. Actor and camera must use the same metric scene.

The optional `run_colmap` integration runs feature extraction, matching, mapping
and text conversion with argument arrays. It requires a separately installed
official COLMAP executable and calibrated scale alignment. Use
`static_scene_mask(performer_mask)` to exclude performer pixels plus a configurable
margin from static-environment feature extraction. Mask filenames follow COLMAP's
image-name + `.png` rule. The integration is not a tested autonomous camera solver
in this build when COLMAP is unavailable; imported timestamped tracks are fully
exercised by unit tests.

After registration, align scene coordinates with surveyed controls. Import each
registered image pose, map its presentation timestamp into world seconds, and
construct a `CameraTrajectory`. Do not assign image registration index as time.
