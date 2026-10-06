# Troubleshooting

**Uncalibrated camera / missing metric scale.** Import measured canonical calibration. Check world units/axes and that external SfM scale was explicitly aligned. Choosing a wizard scale-source label does not establish a measured distance.

**Resolution mismatch.** Use original plate dimensions or explicitly recalibrate/rescale intrinsics. Do not stretch footage to conceal the mismatch.

**High reprojection error.** Filter QC by camera/joint. Check timestamp offsets, lens distortion, transforms, raw detections and actor identity. Disable a demonstrably bad observation with an audited correction; do not automatically loosen reliable camera locks.

**No usable detections / checkpoint failure.** Review registered media and local checkpoint compatibility. HOG is a limited baseline. Confirm joint schema, codec and normalization with the checkpoint documentation. No bundled production pretrained models are promised.

**ModelAssetError.** Choose the licensed numerical NPZ file matching smpl/smplh/smplx. A missing asset never invokes fixture automatically. Ask a technician to convert trusted official assets when necessary.

**FBX export fails.** Run Project Health; verify Blender and inspect the job's detailed log. Fix the stated validation failure rather than accepting a file merely because it exists. Stale dependencies require rerunning the solve.

**Application will not launch.** Use the isolated launcher on a graphical Linux session. CLI/offscreen testing is possible without a display, but the normal GUI still needs working Qt platform/display libraries.

Cancel cooperatively and retain completed checkpoints. Retry repeats the failed work through the service. Full tracebacks remain in logs for technical diagnosis; warnings and known limitations are recorded in BUILD_STATUS.
