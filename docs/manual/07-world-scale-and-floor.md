# World scale and floor

The default world uses metres, +Y up and an explicit origin. OpenCV camera coordinates use +X right, +Y down and +Z forward. Camera translations must belong to the same metric world as actor motion.

Scale must come from surveyed points, known target spacing, a known distance or an explicitly aligned external camera solve. A focal length or a plausible-looking human mesh cannot establish metric scale by itself. The solver requires aligned camera data with metric_scale=1 and refuses ambiguous-scale output.

The wizard records the intended metric source. Calibrated metres explicitly confirms that imported camera coordinates are already metric. That selection alone does not apply a surveyed scale or convert an arbitrary SfM reconstruction. A technician must align imported camera data before solving. The known actor height field is retained metadata in the current release; it does not automatically rescale cameras or constrain the shape objective.

In Calibration, enter Surveyed ground Y (m) and click Apply floor. This creates an audited override for a horizontal calibrated floor. Zero means the Y=0 plane, not a guessed floor beneath the current pose.

General scene planes are supported in core data structures. The GUI currently provides horizontal floor editing; interactive stairs, wall and collision-mesh authoring are incomplete. Verify origin and floor in the 3D scene before evaluating foot contacts or physical refinement.
