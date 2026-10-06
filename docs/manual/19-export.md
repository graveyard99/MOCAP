# Export

Choose Export… in the toolbar or Export Animation… in the Export workspace. Select format, output path, output FPS, world-time range, mesh/camera options and validation.

FBX is the primary character deliverable. It contains one constant-topology weighted mesh, armature hierarchy, local animation, metric root motion and a documented frame range. The Blender backend exports and re-imports into a clean validation scene. EXPORT VALIDATED appears only when actual checks succeed.

NPZ retains numerical rig/body data, including shape, rest geometry, weights, rotations and translation. BVH carries skeleton animation without a skinned mesh. USD is BLOCKED_EXTERNAL on the installed Blender 4.3.2 build, which lacks the USD export operator. A compatible enabled backend plus skeletal validation is still required; do not assume a rigged USD deliverable. Alembic is not implemented.

Choose Metric +Y world for the validated path. All current DCC target choices use the same documented metre/+Y-up convention, with FBX forward −Z. DCC target labels are exposed in the dialog, but per-target presets are not a verified cross-application conversion contract. Read the export metadata; Maya, Houdini, Cinema 4D and Unreal import behaviour has not been tested in this build.

Output FPS resamples the motion. Range values are seconds in the solved world clock; zero End means full take. The first exported DCC frame corresponds to the selected take start. Cameras can be written as a JSON sidecar where supported.

Export refuses stale solve dependencies. Inspect QC, rerun invalidated stages, then export again. A mesh sequence alone does not satisfy the FBX character requirement.

Open output folder opens the solved project artifacts. Your exported character remains at the specific output file selected in the wizard.
