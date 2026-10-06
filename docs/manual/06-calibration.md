# Calibration

A metric solve needs intrinsics, lens distortion and a shared world-to-camera transform for every enabled camera. At least two distinct calibrated cameras must remain enabled.

Use Cameras → Import calibration… or Calibration → Import camera solution…. The desktop importer accepts canonical camera YAML/JSON, such as the camera solution written by this pipeline. It does not directly parse every SfM application's native project. Technical import APIs support OpenCV files, COLMAP text and local Metashape XML; a technician must perform explicit scale/axis alignment before passing the canonical result to the GUI.

Calibration displays actual camera frustums and the metric scene. Select cameras from the Cameras table and inspect locks/provenance. Run calibration stage… validates available camera geometry; it is not an automatic checkerboard-detection wizard.

The core has checkerboard/ChArUco/PnP and controlled refinement APIs. Visual target correspondence selection, surveyed-point editing and before/after comparison are not yet implemented in this workspace.

Check original-plate reprojection in Quality Control. A consistently displaced image projection suggests a transform/convention problem; position-dependent errors near frame edges often suggest distortion. Do not unlock reliable cameras simply to reduce an incorrect body fit's residual.

The example cameras are generated from known metric geometry and are already locked. Running calibration on the example should validate them without moving their transforms.
