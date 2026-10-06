# OpenMocap VFX user manual

This manual describes the implemented desktop application. It is an engineering-preview manual, not a claim that every requested production feature is finished.

Start with [Your First Multi-Camera Solve](quick-start.md). Use the stage state and QC report to distinguish executed work, imported evidence, missing dependencies and stale results.

- [Introduction](00-introduction.md)
- [Installation](01-installation.md)
- [First launch](02-first-launch.md)
- [Projects](03-projects.md)
- [Importing footage](04-importing-footage.md)
- [Camera setup](05-camera-setup.md)
- [Calibration](06-calibration.md)
- [World scale and floor](07-world-scale-and-floor.md)
- [Synchronization](08-synchronization.md)
- [Detection and tracking](09-detection-and-tracking.md)
- [2D pose](10-2d-pose.md)
- [Segmentation](11-segmentation.md)
- [3D reconstruction](12-3d-reconstruction.md)
- [Actor shape](13-actor-shape.md)
- [Body solving](14-body-solving.md)
- [Contacts](15-contacts.md)
- [Physics and kinematic refinement](16-physics.md)
- [Quality control](17-quality-control.md)
- [Manual corrections and undo](18-manual-corrections.md)
- [Export](19-export.md)
- [Retargeting](20-retargeting.md)
- [Moving cameras](21-moving-cameras.md)
- [Advanced settings and solver governance](22-advanced-settings.md)
- [Troubleshooting](23-troubleshooting.md)
- [Command line](24-command-line.md)
- [Technical reference](25-technical-reference.md)

Screenshots were captured from the actual application against the generated demo. The desktop workflow test executed real asynchronous geometry, body fitting, QC and validated NPZ/Blender FBX export; raw observation corrections preserved their original file. See tests/integration/test_ui.py.
