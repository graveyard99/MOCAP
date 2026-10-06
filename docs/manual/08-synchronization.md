# Synchronization

Open Synchronization to inspect offsets, clock scales, drift in parts per million, locks and source. Double-click a camera or use Edit selected timing… to change its timing.

The mapping is t_world = scale × t_camera + offset. An offset of +0.010 means the camera's zero timestamp corresponds to world time +0.010 seconds. Drift is (scale−1) × 1,000,000 ppm. Enter sub-frame offsets directly; there is no integer-frame synchronization requirement.

Keep trustworthy timecode/manual mappings locked. Run synchronization stage… applies or refines the available evidence while respecting locks. Bounded reprojection refinement requires an existing reconstructed trajectory and uncertain camera clocks. If no suitable refinement evidence exists, the system preserves explicit mappings.

The GUI table is not a waveform editor. Audio, flash/event detection visualization and error-versus-offset plots remain incomplete, although signal/event and affine-estimation APIs exist for technical integration.

The multi-camera viewer follows world time and shows selected plate timestamps and offsets. Two cameras can display different source frame numbers at the same world time. The synthetic example's explicit frame manifest is tested. **Current limitation:** GUI sequence/video decoding does not consistently use the core's canonical native-PTS media index; nominal-FPS sequence fallback and video seeking can select an incorrect plate for VFR or sidecar-timestamped media. Stored observation world timestamps can also lag manual clock edits. Validate timing against native metadata; do not treat the viewer alone as proof of synchronization. See the handoff [known issues](../../PROJECT_STATE.md).

Reprojection error that increases during fast motion but decreases during still poses may indicate timing error. Check timestamps and offset sign before changing calibrated extrinsics.
