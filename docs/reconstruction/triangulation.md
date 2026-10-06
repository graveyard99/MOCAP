# Robust N-view reconstruction and confidence

`triangulate(cameras, observations, time=None, config=None)` reconstructs a single
actor/joint from all valid available views. Cameras may be a dictionary keyed by
ID or a list of uniquely named cameras. Observation records retain raw plate
pixels, detector confidence, source and original camera timestamp.

If `time` is omitted, each camera is evaluated at the observation's explicit
world timestamp or its affine `TimeMapping`. An explicit `time` samples cameras
at that world time and assumes the caller has already aligned/interpolated
observations. Instantaneous triangulation does not turn asynchronous measurements
of a moving subject into simultaneous ones; use the motion subsystem to perform
continuous-time alignment. Never pair raw frame indices across cameras.

The solver removes disabled/nonfinite/low-confidence observations and prevents
mixed person or joint IDs. It uses confidence squared and camera-quality squared
as information weights. Normalized undistorted weighted DLT provides initial
geometry. Deterministic pair hypotheses find consensus, capped at 64 for large
camera counts. A Cauchy-loss nonlinear reprojection solve consumes every agreeing
view using the original lens model. It iteratively checks consistency and rejects
views outside the configured 5-pixel default threshold.

Diagnostics retain contributing/rejected IDs, per-camera errors, mean/median/max
inlier residuals, largest usable ray angle, information-matrix condition number,
warnings and a local 3x3 position covariance. Covariance is an approximate linear
propagation using a pixel-noise floor of 1 pixel; it excludes systematic calibration
error and time uncertainty and should not be interpreted as complete uncertainty.

Confidence combines observation/calibration quality, residual agreement and ray
geometry. Angles below 1 degree or condition number above 1e10 are degenerate and
confidence is capped at 0.05. Insufficient consensus returns NaN position and zero
confidence with diagnostics, never a fabricated 3D joint. Missing views do not
need to be present in the observation list.

The acceptance suite checks 2, 3, 8 and 30 cameras, known metre displacement,
pixel noise, multiple corrupted views, disabled/missing measurements, zero/tiny
baseline, distortion round trips, changing-camera interpolation and calibration
locks. These tests establish geometric correctness, not detector accuracy or
production readiness on arbitrary footage.
