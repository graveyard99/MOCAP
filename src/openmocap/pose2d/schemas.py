"""Explicit public landmark orders for supported pose-model checkpoints.

COCO WholeBody uses 17 body, six feet, 68 face and 21 landmarks per hand.
Names follow the MMPose COCO WholeBody metadata in this order; model output
indices remain untouched. See the preset/model-license documents for sources.
"""

from __future__ import annotations

COCO_BODY_17: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def _hand_names(side: str) -> tuple[str, ...]:
    return (f"{side}_hand_root",) + tuple(
        f"{side}_{finger}{joint}"
        for finger in ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger")
        for joint in range(1, 5)
    )


COCO_WHOLEBODY_133: tuple[str, ...] = (
    COCO_BODY_17
    + (
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
    )
    + tuple(f"face-{index}" for index in range(68))
    + _hand_names("left")
    + _hand_names("right")
)
