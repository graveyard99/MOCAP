# Moving cameras

Moving cameras are represented by timestamped world-to-camera poses with interpolated rotation/translation. Intrinsics can also vary in a camera trajectory when explicitly supported by imported evidence.

A moving-camera solve should come from the static environment, coded targets or surveyed geometry. Mask the performer in the external camera tracker when actor movement would contaminate scene features. This application does not ship a complete visual-SLAM or performer-masked SfM desktop workflow.

Import a canonical calibrated camera solution containing the trajectory. Its timestamps must use the same declared camera/world timing convention as the body observations. Align external reconstruction scale, origin, floor and +Y world axes before import. Cameras and actor must share metres.

The plate reprojection code evaluates camera pose/intrinsics at world time. The 3D scene displays sampled frustums and trajectories. Reliable imported trajectories remain locked; learned camera estimates do not replace them.

Camera/lens metadata in the basic editor describes a static record. There is no full per-key camera-trajectory editing tool in the GUI; preserve trajectory samples through expert/import workflows.

The CLI synthetic demo supports --moving for deterministic camera-motion testing. Use it to verify timing/interpolation and reprojection independently of optical tracking quality. Changing distortion without measured lens evidence is not a justified camera-motion solve.
