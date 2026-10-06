"""Conservative floor projection and missing-motion regularization.

This module does not simulate rigid-body dynamics. It deliberately identifies
its limited kinematic scope rather than pretending MuJoCo/SMPL dynamics exist.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openmocap.contacts import ContactResult, refine_contacts
from openmocap.contacts.feet import RefinementResult


def refine_kinematics(
    times: np.ndarray,
    joints: np.ndarray,
    confidence: np.ndarray,
    contacts: ContactResult,
    *,
    max_correction_m: float = 0.05,
    strong_max_correction_m: float = 0.002,
) -> RefinementResult:
    """Apply bounded contact targets and floor penetration correction.

    Scene meshes, balance, momentum, and torque feasibility require a dynamics
    backend. This stage reports them as unsupported; strong data retain their
    metric displacement guard and root trajectory.
    """
    refined = refine_contacts(
        times,
        joints,
        confidence,
        contacts,
        max_correction_m=max_correction_m,
        strong_max_correction_m=strong_max_correction_m,
    )
    result = refined.positions.copy()
    quality = np.asarray(confidence)
    penetration_before = []
    penetration_after = []
    for joint in {j for j in contacts.joint_indices if j is not None}:
        for frame, current in enumerate(result[:, joint]):
            height = float(current @ contacts.floor_normal + contacts.floor_offset)
            if not np.isfinite(height):
                continue
            original = np.asarray(joints)[frame, joint]
            original_height = float(original @ contacts.floor_normal + contacts.floor_offset)
            penetration_before.append(max(0.0, -original_height))
            if height < 0:
                limit = (
                    strong_max_correction_m if quality[frame, joint] >= 0.8 else max_correction_m
                )
                candidate = current + min(-height, limit) * contacts.floor_normal
                # Govern the combined correction, not each refinement pass.
                delta = candidate - original
                if np.linalg.norm(delta) > limit:
                    candidate = original + delta * limit / np.linalg.norm(delta)
                result[frame, joint] = candidate
            penetration_after.append(
                max(
                    0.0,
                    -float(result[frame, joint] @ contacts.floor_normal + contacts.floor_offset),
                )
            )
    diagnostics: dict[str, Any] = dict(refined.diagnostics)
    displacement = np.linalg.norm(result - joints, axis=2)
    maximum = (
        float(np.max(displacement[np.isfinite(displacement)]))
        if np.isfinite(displacement).any()
        else 0.0
    )
    diagnostics.update(
        {
            "mode": "kinematic_only",
            "mean_penetration_before_m": float(np.mean(penetration_before))
            if penetration_before
            else 0.0,
            "mean_penetration_after_m": float(np.mean(penetration_after))
            if penetration_after
            else 0.0,
            "unsupported": ["rigid_body_dynamics", "balance", "momentum", "collision_meshes"],
            "max_displacement_m": maximum,
        }
    )
    return RefinementResult(result, diagnostics)
