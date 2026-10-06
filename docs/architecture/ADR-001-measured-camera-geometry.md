# ADR 001: Measured camera geometry owns world reconstruction

Status: accepted.

Camera transforms map metric +Y-up world coordinates into OpenCV camera axes.
Imported/surveyed/calibrated cameras default to `locked=True`. Geometry performs
CPU-only projection and triangulation with NumPy, SciPy and OpenCV. No network,
GPU, body asset or neural detector is required for the geometric acceptance suite.

The production reconstruction consumes `list[Camera]`, never a stereo-specific
object. Normalized undistorted rays initialize a weighted DLT estimate. A bounded
number of pair hypotheses identify consensus; the final Cauchy-loss nonlinear
solve consumes **all** agreeing cameras in original distorted plate pixels.
Pair selection is a deterministic initialization strategy, not the final solve.

`refine_camera` returns a separate object, bypasses locked parameters, and accepts
fixed world evidence. This avoids camera/body gauge freedom and stops body errors
from being hidden by uncontrolled camera motion. Default uncertain-camera limits
are 0.035 radians and 0.03 metres per component, with explicit priors and reported
deltas. `prefer_authoritative_camera` always returns an existing calibrated
camera even when a learned alternative is provided.

Moving cameras use camera-centre interpolation and rotational SLERP, with optional
linearly varying intrinsics/distortion samples. Out-of-range tracking timestamps
raise an error rather than silently extrapolating. Camera tracking itself should
use static-scene evidence; the optional COLMAP integration accepts performer
exclusion masks. SfM scale is never accepted as metric without external alignment.

Consequences: strong measured geometry remains inspectable and authoritative;
CPU geometry can be verified independently of inference quality. Sparse camera
tracking interpolation is piecewise rather than an optimized smooth continuous
camera trajectory. Sub-frame human trajectory reconstruction belongs to the
separate motion subsystem.
