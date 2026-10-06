# Introduction

OpenMocap VFX reconstructs metric human motion from calibrated, timestamped RGB camera observations. The current release is an engineering preview. Its tested path includes arbitrary-camera geometric reconstruction, continuous-time motion, persistent actor proportions, a weighted character rig, and validated animation export. The included character is an explicitly synthetic technical mannequin. Actual SMPL-family use requires separately licensed assets.

The desktop application and command line call the same project service. Work remains local. A GPU is useful for compatible inference checkpoints; the geometry, tutorial and numerical body fitter run on CPU.

Start with [Your First Multi-Camera Solve](quick-start.md). It uses generated plates and known 2D observations, so neither a model download nor private footage is required. The tutorial demonstrates the real geometry and character export without implying that a neural detector ran.

![Application home](assets/project-manager.png)

The evidence order is surveyed geometry, robust multiview measurements, temporal observations, body constraints, physical plausibility, and finally learned inference. Camera locks and source labels make that policy visible. A green COMPLETE stage describes executed processing; WARNING requires reading its explanation, especially when imported observations replaced inference. STALE means an upstream edit invalidated the solve.

Advanced calibration correspondence editing, waveform displays, silhouette shape objectives and comprehensive multi-actor correction remain incomplete. See [BUILD_STATUS](../BUILD_STATUS.md) before planning a production shoot.
