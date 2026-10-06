# Retargeting

The solved character is a metric intermediate rig with persistent rest geometry, joint hierarchy, local rotations, skin weights and animated root. Names/mappings are explicit; no global joint-number guess is allowed. A target rig adapter must define source/target names, rest axes, scale, hierarchy and missing joints.

NPZ preserves numerical state; BVH is skeleton-only; FBX carries the weighted constant-topology mesh and animation. Transfer the animation in your DCC using your target's rest-pose and axis conventions, then verify root trajectory and feet. No validated turnkey MetaHuman/custom-rig retargeter is included. Generated marker names or missing COCO pelvis must not be silently invented as measured joints.

Blender re-import was verified; Maya, Houdini, Cinema 4D and Unreal target labels record the intended destination; all currently use the exporter’s documented convention and still need native application validation. A preset label is not evidence of tested downstream import behavior.
