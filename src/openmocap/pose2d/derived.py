"""Explicit geometric landmark derivation, preserving all raw model outputs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from openmocap.types import Pose2DObservation


def derive_midpoint_landmarks(observations: list[Pose2DObservation]) -> list[Pose2DObservation]:
    """Append lower-confidence hip/shoulder midpoints when model omits roots.

    These are plate-space geometric midpoints, not measured anatomical centres.
    Perspective and landmark-definition biases are recorded explicitly. Never
    replace an existing measured pelvis or neck observation.
    """
    result = list(observations)
    groups: dict[tuple[str, float, str], dict[str, Pose2DObservation]] = {}
    for row in observations:
        groups.setdefault((row.camera_id, row.camera_timestamp, row.person_id), {})[
            row.joint_id
        ] = row
    for joints in groups.values():
        for target, left, right, weight in [
            ("pelvis", "left_hip", "right_hip", 0.65),
            ("neck", "left_shoulder", "right_shoulder", 0.5),
        ]:
            if target in joints or left not in joints or right not in joints:
                continue
            a, b = joints[left], joints[right]
            covariance = None
            if a.covariance is not None and b.covariance is not None:
                covariance = (np.asarray(a.covariance) + np.asarray(b.covariance)) / 4
            result.append(
                replace(
                    a,
                    joint_id=target,
                    xy=(a.xy + b.xy) / 2,
                    confidence=min(a.confidence, b.confidence) * weight,
                    covariance=covariance,
                    source="geometric_midpoint",
                    metadata={
                        **a.metadata,
                        "derived_from": [left, right],
                        "limitation": "plate midpoint; anatomical and perspective bias",
                    },
                )
            )
    return result
