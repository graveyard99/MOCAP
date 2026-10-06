"""Uniform export sampling without changing persistent shape or rig topology."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from openmocap.body_models.model import forward_kinematics
from openmocap.fitting import Animation


def resample_animation(
    animation: Animation, fps: float, start: float | None = None, end: float | None = None
) -> Animation:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Export FPS must be finite and positive")
    first = float(animation.times[0] if start is None else max(animation.times[0], start))
    last = float(animation.times[-1] if end is None else min(animation.times[-1], end))
    if last < first:
        raise ValueError("Export range does not overlap solved motion")
    times = first + np.arange(int(np.floor((last - first) * fps + 1e-8)) + 1) / fps
    if len(animation.times) < 2:
        return animation
    rotations = np.empty((len(times), len(animation.parents), 3))
    for joint in range(len(animation.parents)):
        rotations[:, joint] = Slerp(
            animation.times, Rotation.from_rotvec(animation.rotations[:, joint])
        )(times).as_rotvec()
    translations = CubicSpline(animation.times, animation.translations)(times)
    joints = np.array(
        [
            forward_kinematics(animation.rest_joints, animation.parents, r, t)[0]
            for r, t in zip(rotations, translations, strict=True)
        ]
    )
    confidence = (
        None
        if animation.confidence is None
        else np.column_stack(
            [
                np.interp(times, animation.times, animation.confidence[:, joint])
                for joint in range(len(animation.parents))
            ]
        )
    )
    return Animation(
        times=times,
        joints=joints,
        rest_joints=animation.rest_joints.copy(),
        parents=animation.parents.copy(),
        names=animation.names.copy(),
        vertices=animation.vertices.copy(),
        faces=animation.faces.copy(),
        weights=animation.weights.copy(),
        rotations=rotations,
        translations=translations,
        shape=animation.shape.copy(),
        model_name=animation.model_name,
        diagnostics={**animation.diagnostics, "resampled_fps": fps},
        posedirs=animation.posedirs,
        confidence=confidence,
    )
