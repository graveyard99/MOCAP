# ADR 003: Continuous observations, bounded clocks and kinematic contact targets

Status: implemented and tested for the numerical APIs.

## Context

RGB cameras have independent timestamps, uneven frame cadence and potential
clock drift. Rounding to shared frame indices destroys sub-frame information.
Temporal regularization and planted-foot corrections may help noisy/occluded
motion but must not silently override strong metric reconstruction.

## Decision

Preserve raw camera timestamps and use an explicit affine world-time mapping.
Estimate offset and drift from comparable measured signals with bounded robust
optimization, or refine uncertain clocks against a frozen metric trajectory.
Reprojection clock refinement requires a locked reference to fix the time gauge.
It keeps camera geometry fixed and returns candidate mappings with before/after
residuals; applying a mapping is an explicit service-layer action.

Use SciPy weighted cubic splines for CPU-testable continuous trajectories.
Adaptive regularization enforces a configurable strong-observation displacement
guard, default 2 mm. Queries outside support return missing data by default;
confidence decays across occlusions. The root reconstruction stage may jointly
optimize its spline basis on native-time multiview reprojection observations;
the standalone trajectory module accepts measured 3D targets and preserves the
same governance principle.

Infer contacts from calibrated-plane proximity, velocity, confidence and
hysteresis. Produce non-destructive bounded foot targets and before/after slide
metrics. Strong measured joints and the root trajectory remain protected. The
implemented optional physics stage is expressly kinematic; it reports the lack
of dynamics, balance, momentum and scene-mesh collision capabilities.

## Alternatives

Common-frame triangulation alone is useful for initialization but loses precise
timing. Neural temporal reconstruction is not needed to interpolate measured
metric motion. Unconstrained clock/body/camera joint optimization risks gauge
ambiguity and camera motion compensating for actor mistakes. Full dynamics may
be useful later, but requires validated model inertias, contacts and a compatible
simulation backend; an unexplained simulator wrapper is not a working dynamics
solution.

## Consequences and verification

There is no hidden camera rewrite. Signal alignment may be ambiguous for periodic
events and must be checked across a take. Per-joint splines do not independently
enforce anatomy; persistent shape/body fitting supplies that constraint. Contact
target correction requires downstream IK to preserve bone lengths. Tests verify
sub-frame offsets/drift, moving camera fixtures, irregular timestamp trajectories,
occlusion support, strong displacement limits, floor-conditioned contact and
locked calibrated camera invariance.
