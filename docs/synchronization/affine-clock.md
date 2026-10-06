# Asynchronous capture and clock alignment

Every measurement retains its camera timestamp in seconds. The mapping into the
metric solve's world timeline is `world = scale * camera + offset`. Scale is near
one; `(scale - 1) * 1e6` expresses clock drift in ppm. There is no association of
frame N across cameras.

`openmocap.sync.validate_timestamps` reports duplicate samples, gaps consistent
with dropped frames, and backwards discontinuities. Duplicates may be retained
in raw data; signal synchronization merges them by averaging. A backwards clock
jump raises an error unless the caller explicitly requests diagnostic-only
validation. Split or rebase such recordings with an auditable mapping before
reconstruction. VFR intervals are accepted: the dropped-frame diagnostic is
relative to median cadence, not a declaration that a VFR source is malformed.

`estimate_time_mapping` aligns comparable scalar or vector signals against a
reference using a coarse offset/drift search followed by robust nonlinear
optimization and cubic sub-frame interpolation. Appropriate signals include
audio envelopes, LED/flash brightness, or a shared motion descriptor. Raw X/Y
joint coordinates from different camera viewpoints are not interchangeable
motion signals. Signals must contain variation and enough overlap over the
entire configured bound. Optional normalization handles amplitude differences;
it should be disabled when values have a shared physical meaning.

The default offset bound is ±1 second and default drift bound ±1000 ppm. Narrow
these bounds using timecode or manual event evidence. Signal alignment returns
an unlocked estimated mapping with confidence based on normalized signal
residual; a small residual alone does not prove uniqueness for periodic signals.
Inspect multiple events over a long take before trusting a drift estimate.

For example:

```python
from openmocap.sync import estimate_time_mapping

mapping = estimate_time_mapping(camera_t, camera_led, reference_t, reference_led,
                                offset_bounds=(-0.1, 0.1),
                                scale_bounds=(0.9995, 1.0005))
world_t = mapping.to_world(camera_t)
```

`refine_time_mapping` accepts a residual callback into an existing continuous
motion solve. It optimizes only uncertain timing, within explicit offset/drift
bounds and priors. It returns before/after cost and deltas. A locked mapping is
returned unchanged without invoking optimization. This interface is fully
implemented; callers must provide residuals from trustworthy calibrated cameras.
Automatic event/audio extraction is not included in this module.

`refine_multiview_timing(cameras, observations, trajectory, joint_names)` supplies
the concrete residual integration: it projects the frozen trajectory through
each fixed calibrated camera at candidate world times and compares native
distorted plate pixels. A locked enabled reference camera is required to fix the
clock gauge. Results contain candidate mappings plus per-camera offset/drift
deltas, sample counts and before/after median pixel errors. The function never
mutates original observations, camera poses or camera clocks; the application
must explicitly apply accepted candidates and invalidate downstream results.

`estimate_from_events` consumes matching camera/world event times (claps,
flashes, timecode or surveyed timing cues). One event estimates offset while
retaining scale; multiple events allow bounded robust drift estimation. Event
IDs must be paired by the operator or an upstream detector. It reports event
count, residual in seconds, before/after deltas and whether drift was observable.

Synthetic tests recover a 17.3 ms offset and 370 ppm drift over a 24-second signal
to 0.15 ms and 10 ppm tolerances. Duplicate/dropped timestamps and manual locks
are also tested.
