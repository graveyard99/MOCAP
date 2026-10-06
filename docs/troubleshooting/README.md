# Troubleshooting

- **Missing calibration or metric scale:** import measured camera parameters, verify units/world axes and establish scale from a survey/known target. Do not use a guessed camera to make the solve run.
- **High residual in one camera:** inspect raw/reprojected keypoints, image size, distortion, transform and timing. Disable it only as an auditable temporary correction, then rerun stale stages.
- **All cameras disagree around fast motion:** inspect sub-frame offsets, native PTS and drift. A nominal FPS match is not synchronization.
- **Unstable depth:** camera rays have inadequate baseline or the joint is seen by too few usable cameras. Add coverage; the solver intentionally reports uncertainty/degeneracy.
- **Good geometry, poor fitted body:** verify joint schema, named pelvis/root, body file and actor proportions. A strong-joint displacement beyond the allowed bound fails; weakening trusted measurements to hide a bad model is not a repair.
- **Missing licensed model:** acquire the correct family legally, put it under installation-root/models/body and select the actual numerical NPZ file. The fixture is for the tutorial only.
- **Inference checkpoint fails:** verify its actual input normalization, output contract, joint order and license. No production whole-body checkpoint is bundled.
- **Qt cannot launch:** interactive launch needs a display and compatible OS Qt libraries. `QT_QPA_PLATFORM=offscreen` is for tests, not normal artist viewing. Run doctor through the launcher environment.
- **FBX backend absent:** install an approved portable Blender under installation-root/tools or make an existing system Blender discoverable. No rig validation badge appears if re-import fails.
- **Stale stage:** save, review downstream invalidation and rerun the selected stage/full solve. Expensive raw inference artifacts remain reusable when compatible.

Keep doctor output, config, QC, input hashes and seed with a bug report. Do not send private footage, licensed models or secrets.
