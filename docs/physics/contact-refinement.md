# Foot contacts and optional kinematic refinement

Contact is supported separately for left heel, left forefoot, right heel and
right forefoot. Use real heel/toe landmarks where available. Standard SMPL
`left_ankle/right_ankle` and `left_foot/right_foot` landmarks are accepted as
proxies; the report explicitly records which landmarks were used. An ankle
marker above the floor is not a true heel-contact measurement.

The floor plane is `normal · X + offset = 0` in metric world coordinates. Normal
and offset are normalized together. Contact evidence combines distance to this
plane, horizontal speed, vertical speed and propagated confidence. Hysteresis
and a minimum planted duration prevent single-sample switching. A stationary
foot far above the floor is not contact. The result includes probabilities,
planted masks, start/end intervals, landmark sources and thresholds.

```python
from openmocap.contacts import estimate_contacts, refine_contacts

contacts = estimate_contacts(times, joints, names, confidence=confidence,
                             floor={"normal": [0, 1, 0], "offset": 0})
refined = refine_contacts(times, joints, confidence, contacts)
print(refined.diagnostics["before_slide_m_s"])
print(refined.diagnostics["after_slide_m_s"])
```

Optional correction proposes a confidence-weighted planted anchor and projects
it onto the calibrated plane. Weak evidence may move within an explicit bound
(default 5 cm); strong measurements may move by at most 2 mm. Root and non-foot
joints are preserved. Corrections are derived targets, not destructive edits to
raw data. A downstream IK/body solve should consume them to maintain anatomy;
independently adjusted joint targets must not be advertised as a skinned pose.

`apply_contact_overrides(raw, [{"foot": "left_heel", "start": 1.2, "end": 1.8,
"planted": true}])` creates a separate effective contact result with a manual
audit trail (use `True` in Python). The raw inferred probabilities are preserved.
Manual contact labels do not bypass the geometric displacement limits.

`openmocap.physics.refine_kinematics` adds bounded floor-penetration correction
and enforces the displacement bound across both passes combined. Its diagnostic
mode is `kinematic_only`. Full rigid-body dynamics, balance, torque/momentum,
collision meshes and inertial simulation are not implemented. No MuJoCo or
proprietary simulator is silently substituted. Floor/contact constraints can
help short occlusions; strong measured floor penetration remains visible if
correcting it would violate the evidence guard.

Acceptance tests show materially reduced planted drift with unchanged swing
phases/root motion, rejection of raised stationary feet, arbitrary plane normals,
and ≤2 mm total modification of strong data. Reports use mean tangential speed
in metres/second over adjacent planted samples.
