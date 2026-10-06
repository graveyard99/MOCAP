# Confidence, uncertainty and solver governance

Confidence values are structured quality signals, not calibrated probabilities of accuracy. Preserve original detector score, source, person/joint identity, camera/local/world timestamps and optional image covariance. Derived filters do not rewrite raw values. Camera quality/trajectory confidence and metric-scale state are separate from keypoint confidence.

## Triangulation

Each usable view receives weight `observation_confidence² × calibration_confidence² / (image_covariance_trace/2 + camera_reprojection_RMSE²)`; absent image covariance uses a one-pixel variance reference. Calibration confidence includes moving-camera trajectory confidence. Disabled cameras, invalid measurements and low-confidence views are rejected explicitly; duplicates preserve raw records while the strongest effective measurement for a view/time is used.

Robust consensus and nonlinear reprojection use all surviving views. Output diagnostics include contributing/rejected camera IDs, per-view pixel residuals, median/mean/max errors, maximum useful ray angle and information-matrix condition number. Parallel and antiparallel rays both provide poor depth conditioning. Confidence combines weighted observation/calibration quality, `min(1, angle_degrees/10)` and `exp(-median_error_px/5)`. Degenerate results are capped at 0.05. The scalar does not replace covariance.

Covariance is an approximate local uncertainty: pseudoinverse of the physical weighted reprojection-Jacobian information matrix scaled by residual variance, with a one-pixel floor. It does not model every lens/timing/identity systematic error. **Known defect:** for singular geometry the pseudoinverse can report falsely tiny variance along an unobservable direction. Do not trust covariance when degeneracy is reported; the scalar confidence cap and geometry flags remain the usable warning. Nullspace-aware covariance is an outstanding fix.

Covariance is not yet propagated through the complete pipeline: saved motion and downstream body fitting primarily use scalar confidence. Moving-camera trajectory confidence is included in triangulation but is currently omitted from native-ray and body-reprojection weights. These inconsistencies are recorded in [PROJECT_STATE.md](../PROJECT_STATE.md) and remain unfixed at handoff.

## Downstream control

The trust order is surveyed/calibrated geometry, robust multiview measurements, temporal evidence, body-model constraints, biomechanical plausibility, learned priors and monocular estimates. Parameters retain source/owner and LOCKED/BOUNDED/FREE state. The body fitter never edits camera transforms. Timing/camera refinement consumes uncertainty, explicit bounds and priors; it reports deltas and preserves locks.

Actor shape persists across a take. Learned initialization can fill joints only below confidence 0.2. Strong measured joints (default confidence at least 0.85) cannot move more than the explicit 0.03 metre bound without a diagnostic failure. Contact and optional kinematic corrections are confidence-bounded. Body-fit confidence decreases with fitting residual rather than copying triangulation score as proof that the rig agrees. Missing/template-initialized anatomy is labeled inferred and contributes no actor-shape measurement evidence.

Inspect original-plate geometry residuals and fitted-body residuals separately in QC. A smooth result or plausible pose is not evidence that calibration/timing is correct. Current uncertainty is conservative engineering quality, not certified metrological accuracy.
