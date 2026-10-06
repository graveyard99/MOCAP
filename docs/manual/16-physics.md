# Physics and kinematic refinement

Open Quality Control and choose Optional physics… to run the available refinement stage after a geometric/body solve. It runs asynchronously and re-fits the same persistent rig.

The current implementation is confidence-bounded kinematic/contact refinement. It is not a MuJoCo, Isaac or OpenSim dynamics simulation. No validated momentum, inertial-balance or biomechanical-muscle solve is advertised.

The stage can reduce low-confidence jitter, planted-foot drift and floor penetration while limiting motion of reliable observations. Geometry remains authoritative. If a physical preference conflicts with surveyed cameras and many agreeing views, inspect the conflict rather than letting a hidden simulation relocate measured joints.

Before refinement, save the project and note QC residuals. Afterward, compare foot-sliding diagnostics, fitted-body reprojection and strongly measured joint displacement. Refined animation is checkpointed, and completed raw/geometry artifacts remain available.

General SceneConstraint records can represent environment features in the core. Interactive collision-mesh, stair, wall and seat configuration is not finished in this GUI.

For the tutorial, physics is optional. A NOT RUN physics stage is expected unless the user explicitly requests it; a geometric animation remains exportable. Use this stage to improve justified uncertainty, not to conceal bad timing, identity or camera calibration.
