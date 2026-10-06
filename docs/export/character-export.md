# Animated character export

```python
from openmocap.export import export_animation

metadata = export_animation(animation, "exports/actor.fbx", fps=30)
```

The FBX backend uses a headless Blender process with factory settings and a
temporary directory adjacent to the output. It creates one persistent mesh,
one armature, conventional named bones, normalized vertex groups, an armature
modifier, quaternion rotation keys, and metric root motion. It exports skeletal
animation, rather than a mesh sequence. SMPL-family pose blend coefficients,
when present, are exported as animated relative shape keys while retaining the
same skeleton and skinning. This more expensive path should be profiled for
large SMPL-X assets before production use.

World coordinates are metres and +Y up. One documented proper rotation maps
world coordinates to Blender's +Z-up scene, preserving handedness and scale.
FBX carries Y-up/−Z-forward axis metadata and Blender's unit conversion. The
animation root translation is the world position of the skeleton root. This
differs from raw SMPL APIs that commonly add `transl` to an existing template
root offset; use the documented convention when consuming NPZ.

Export requires a uniform sampled time grid. Output FPS can be fractional (for
example 24000/1001). The exporter rejects a requested FPS inconsistent with the
provided times; resample the continuous trajectory and solve those times first.
The first exported frame is 1, with the original world start time retained in
NPZ. An adjacent `.json` records model, FPS, range, units and validation.

## Actual validation

The backend checks first/last native skeleton heads against the input FK joints,
then deletes the scene's objects and imports the exported FBX. It verifies mesh,
skeleton, armature modifier, weighted vertices, animation action, expected frame
range, root positions, first/last metric joint positions, and actual deformed
vertex positions against numerical linear blend skinning. For models with pose
correctives it additionally verifies imported shape keys and their animation.
It reports
`validated=true` only after this re-import succeeds. A Blender import does not
constitute independent verification in Maya, Houdini, Cinema 4D or Unreal; those
applications' import behavior remains unverified. Preserve importer units and
avoid applying an additional centimetre/metre scale conversion.

Blender is an external GPL-licensed executable, not linked into or redistributed
with the Python package. Pass `blender="/installation/tools/blender/blender"` to
use a project-local portable installation. An already installed Blender can be
used as a detected system prerequisite. No portable download is automatic.

## Other formats

| Format | Implemented contents | Validation |
| --- | --- | --- |
| NPZ | Rig, shape, mesh, skin, pose correctives, motion, timestamps, diagnostics | Numerical roundtrip |
| BVH | Named hierarchy and root/joint animation, metric +Y-up | Writer dimensions; no independent DCC roundtrip |
| USD | BLOCKED_EXTERNAL on this build: Blender lacks `bpy.ops.wm.usd_export` | A USD-enabled Blender build and skeletal re-import validation are still required |
| Alembic | Not implemented | Mesh caches are not a rigged-character deliverable |

BVH has no skinning or mesh by design. Its rotations use declared Z-X-Y channels.
`map_joints` is an explicit naming adapter; full production target-rig retargeting
still requires bind-pose orientations, rest proportions and target controls.
