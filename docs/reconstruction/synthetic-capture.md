# Deterministic virtual capture

`openmocap.synthetic.generate_capture(directory)` creates a redistributable
generated example without footage, checkpoints, or licensed body files. It
writes a human-readable `project.yaml`, raw `observations.json`, per-camera
`frames.json`, `ground_truth.npz`, and generated PNG plates. The plates clearly
label themselves SYNTHETIC and render a moving skeleton on a neutral grid.

The actor uses the conventional 24-joint SMPL hierarchy, constant metric rest
offsets and known FK rotations. It is a procedural geometric fixture, not an
SMPL-family asset or an anatomical skin model. Ground truth includes all joint
positions, rotations, root translation and persistent proportions. Root travel
is exactly 2.43 m over the clip.

Camera count is arbitrary (minimum two), with an evenly distributed ring and
varying height. Intrinsics include OpenCV lens distortion. Clock mapping is
per-camera affine, with sub-frame acquisition phases, offsets and drift.
`moving=True` creates timestamped metric camera trajectories with SLERP-compatible
poses. Other controls include pixel noise, missing measurements, random or whole
camera outliers, dropped/duplicate frames, frame rate and random seed.

```python
from openmocap.synthetic import generate_capture

config = generate_capture("example", camera_count=30, frames=60,
                          moving=True, noise_px=.8,
                          missing_probability=.1,
                          outlier_probability=.01, seed=42)
```

Each observation preserves camera/world timestamps, actor and joint IDs,
confidence, source and fixture metadata. `world_timestamp` is included as known
truth for diagnostic comparison; production time conversion uses camera clocks
and recorded mappings. Root pipeline APIs ingest these observations through the
same typed contracts as imported neural results.
