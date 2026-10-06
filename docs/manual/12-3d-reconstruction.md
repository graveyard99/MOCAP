# 3D reconstruction

Import calibrated cameras and timestamped observations, or run compatible pose inference and association first. Select 3D Reconstruction → Run selected stage…, or use Run Pipeline… for the complete downstream solve.

The review dialog lists selected cameras, world-time range, output FPS, configured compute path and already completed stages. Content hashes determine cache reuse. The numerical core accepts arbitrary camera counts and whichever valid views observe each joint; every camera need not see every hand or foot.

The solver undistorts measured rays, tests geometric hypotheses, rejects reprojection outliers and refines using surviving N-view evidence. Tiny baselines and poor ray angles lower confidence or produce warnings. A plausible position from nearly parallel cameras is not a reliable metric measurement.

Reconstruction retains contributing/rejected cameras, pixel residuals, confidence, ray geometry and conditioning. The continuous trajectory uses native observation times and can sample an output rate independent of source FPS. Raw detector observations are retained.

Open Viewer → 3D Scene to inspect the metric skeleton, camera frustums, floor, confidence rings and fitted mesh when available. Drag to orbit; Shift-drag pans; wheel zooms; Fit / frame centers the actor. Ortho changes view projection only.

Then inspect Quality Control. Reliable geometry should explain plate observations before shape, contact or physics refinements are trusted.
