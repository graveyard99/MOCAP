# Calibration workflows

Canonical JSON is the simplest interchange format. It records lens parameters,
world-to-camera transforms, provenance, confidence, time mappings and locks.
`save_cameras` writes schema version 1, units `metres`, world up `+Y`.
`load_cameras` rejects unknown units, duplicate IDs and unestablished metric scale.

```python
from openmocap.calibration import load_cameras, save_cameras
cameras = load_cameras("calibration/cameras.json")
```

An intrinsic record has `fx,fy,cx,cy,width,height,distortion,distortion_model`.
`opencv` supports 0/4/5/8/12/14 coefficients, including rational radial and thin
prism/tilt models. `fisheye` requires four coefficients. Predictions compare
directly to original distorted footage; undistortion returns normalized rays.
Anamorphic metadata can initialize independent horizontal/vertical focal lengths
and pixel aspect, but a specialized anamorphic lens distortion solve is incomplete.

## Calibration targets

`checkerboard_points(columns,rows,square_metres)` constructs inner corner positions.
`detect_checkerboard` finds precise SB corners. `detect_charuco` returns matched
metric board points and pixels. `detect_coded_controls` detects AprilTag 36h11 by
default and matches each surveyed marker's four world-space corners.

`calibrate_intrinsics(object_points,image_points,image_size,
target_spacing_metres=...,fisheye=False)` solves at least three target views and
reports RMS and per-image RMS. Use diverse board orientations covering the image,
then inspect residuals and rerun without contaminated or blurred views. Target
spacing must be physically measured in metres. Intrinsic board poses are target
local; they do not establish a common surveyed scene by themselves.

`surveyed_pnp(camera_id,world_points_metres,image_points,intrinsics)` robustly fits
extrinsics from at least six matching controls, returns inlier/rejected indices,
and marks the resulting camera surveyed/locked. Planar controls are allowed;
non-coplanar controls provide better depth reliability.

## External imports

* `import_opencv(path,id,world_convention="Y_UP_METRES")` reads OpenCV FileStorage
  `K/D/R/t` (or descriptive aliases), image width/height and optional lens model.
* `import_colmap_text(directory,metric_scale=...,world_rotation=...,
  world_translation=...)` reads `cameras.txt/images.txt`. Supported models include
  PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV and
  OPENCV_FISHEYE. Registered image stems become camera IDs. For a moving camera,
  construct a `CameraTrajectory` from timestamped registered image poses.
* `import_metashape_xml(path,metric_scale=...,world_rotation=...,
  world_translation=...)` reads local frame-camera transforms and sensor
  calibrations. Chunk GIS transforms and nonzero sensor skew are explicitly
  rejected; align/export a supported local frame first.

Photogrammetric reconstructions require surveyed controls/known distances for
scale and an explicit world-axis alignment. Passing a scale of 1 asserts that
the reconstruction is already metric; it does not estimate scale.

## Controlled refinement

Camera locks are authoritative. `refine_camera` cannot change locked cameras,
never mutates its input and cannot rewrite a moving track using a static solver.
An uncertain static camera may use `ParameterControl(state=BOUNDED,...)` with
six-component rotation/translation delta bounds or explicit FREE control. FREE
still has positive prior sigmas. Every result contains before/after reprojection
errors and camera deltas. Intrinsic/lens optimization and a joint camera/body
bundle-adjustment UI are not implemented; their locks are represented for future
controlled solvers.
