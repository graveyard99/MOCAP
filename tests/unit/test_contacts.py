import numpy as np

from openmocap.contacts import (
    ContactSettings,
    apply_contact_overrides,
    estimate_contacts,
    refine_contacts,
)
from openmocap.physics import refine_kinematics


def fixture():
    times = np.linspace(0, 3, 91)
    joints = np.zeros((91, 5, 3))
    # Root is well measured. Four planted markers contain small solver drift.
    joints[:, 0, 0] = times
    joints[:, 0, 1] = 1
    for foot in range(1, 5):
        joints[:, foot, 0] = 0.03 * np.sin(times * 1.3)
        joints[:, foot, 2] = 0.1 * foot
    # Swing is slow at its apex, but far above the floor.
    joints[60:, 1:, 1] = 0.3
    confidence = np.full((91, 5), 0.75)
    confidence[:, 0] = 1
    return times, joints, confidence


def test_contact_reduces_slide_without_pinning_swing_or_moving_root():
    times, joints, confidence = fixture()
    names = ["pelvis", "left_heel", "left_toe", "right_heel", "right_toe"]
    contacts = estimate_contacts(times, joints, names, confidence=confidence)
    assert contacts.planted[:50].mean() > 0.8
    assert not contacts.planted[65:].any()
    refined = refine_contacts(times, joints, confidence, contacts)
    assert np.array_equal(refined.positions[:, 0], joints[:, 0])
    assert np.array_equal(refined.positions[65:], joints[65:])
    for name, before in refined.diagnostics["before_slide_m_s"].items():
        assert refined.diagnostics["after_slide_m_s"][name] < before * 0.25


def test_strong_contacts_and_physics_preserve_measurements():
    times, joints, confidence = fixture()
    confidence[:] = 1
    joints[:60, 1:, 1] = -0.008
    contacts = estimate_contacts(
        times,
        joints,
        ["pelvis", "left_heel", "left_toe", "right_heel", "right_toe"],
        confidence=confidence,
    )
    refined = refine_kinematics(times, joints, confidence, contacts)
    assert np.max(np.linalg.norm(refined.positions - joints, axis=2)) <= 0.002 + 1e-12
    assert np.array_equal(refined.positions[:, 0], joints[:, 0])
    assert refined.diagnostics["mode"] == "kinematic_only"


def test_stationary_raised_feet_are_not_contact():
    times, joints, confidence = fixture()
    joints[:, 1:, 1] = 0.4
    contacts = estimate_contacts(
        times,
        joints,
        ["pelvis", "left_heel", "left_toe", "right_heel", "right_toe"],
        confidence=confidence,
    )
    assert not contacts.planted.any()
    assert not contacts.intervals


def test_calibrated_arbitrary_floor_and_explicit_smpl_proxies():
    times, joints, confidence = fixture()
    joints = joints[..., [0, 2, 1]]
    contacts = estimate_contacts(
        times,
        joints,
        ["pelvis", "left_ankle", "left_foot", "right_ankle", "right_foot"],
        floor={"normal": [0, 0, 1], "offset": 0},
        confidence=confidence,
        settings=ContactSettings(minimum_duration_s=0.05),
    )
    assert contacts.planted[:50].any()
    assert contacts.diagnostics["joint_sources"]["left_heel"] == "left_ankle"


def test_manual_contact_overrides_are_non_destructive():
    times, joints, confidence = fixture()
    raw = estimate_contacts(
        times,
        joints,
        ["pelvis", "left_heel", "left_toe", "right_heel", "right_toe"],
        confidence=confidence,
    )
    original = raw.planted.copy()
    effective = apply_contact_overrides(
        raw, [{"foot": "left_heel", "start": 0, "end": 1, "planted": False}]
    )
    assert not effective.planted[times <= 1, 0].any()
    assert np.array_equal(raw.planted, original)
    assert effective.diagnostics["manual_overrides"][0]["source"] == "manual"
