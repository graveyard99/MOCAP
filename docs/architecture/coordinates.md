# Coordinate systems and metric alignment

The canonical world uses metres, +Y up, a documented origin, and a surveyed floor.
`CameraPose.rotation` and `.translation` satisfy:

```text
point_camera = rotation @ point_world + translation
camera_centre_world = -rotation.T @ translation
```

The local OpenCV camera uses +X right, +Y down and +Z forward. Calibration target
coordinates may use an OpenCV target-local XY board plane; these are not world
axes until the target is surveyed. `surveyed_pnp` accepts metric world coordinates
and determines the corresponding world-to-camera transform.

`openmocap.geometry.coordinates` centralizes conventions. Canonical world, SMPL,
Maya, Houdini and canonical FBX use +Y up. Blender uses `(x,-z,y)` and +Z up. The
`opencv` basis conversion is a local orientation convention, not a replacement
for the surveyed transform of an arbitrary camera. Distances convert explicitly
between metres, centimetres and millimetres. Unknown conventions fail validation.

```python
from openmocap.geometry import convert_points, align_similarity
blender_vertices = convert_points(vertices_world, "world", "blender")
scale, rotation, translation = align_similarity(sfm_controls, surveyed_controls)
```

Similarity alignment needs at least three non-collinear matching controls and
rejects a non-positive scale. `scale_from_distance(a,b,known_metres)` establishes
scale from a measured distance. Camera importers require an explicit externally
established scale for COLMAP/Metashape. Actor-height scaling must be deliberately
chosen by the application; geometric camera APIs do not silently infer it.
