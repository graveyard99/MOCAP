# Quality control

Every full solve writes `outputs/qc/report.json` and `report.html`. Inspect camera/joint residual distributions, used/rejected observations, native timestamps, offset/drift, missing-data percentage, triangulation geometry and confidence, persistent shape residuals, floor/contact intervals and export validation. Geometric trajectory reprojection and fitted-body reprojection are separate so a good skeleton solve cannot hide a bad rig fit.

Large residuals can indicate calibration, timing, identity, detection or body-fitting failure. Inspect the affected camera/time/joint in Viewer before disabling evidence. A rejected observation remains visible and does not imply the original detector output was destroyed. Strong geometric support with poor fitted-body residuals points toward joint mapping/body proportions rather than the camera.

In the desktop Quality Control tab, filter diagnostics and double-click to navigate where camera/time identifiers exist. Contact tracks and selected ranges support review. Click measured observations and use inspector Disable / Restore to create reversible overrides; Edit Undo/Redo preserves an audit trail. Silhouette residuals are reported unavailable when no silhouette objective exists; absent metrics are not zero error.
