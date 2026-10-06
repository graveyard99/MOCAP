"""Explicit joint-name mapping without implicit anatomical substitutions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def map_joints(
    joints: NDArray[np.float64],
    source_names: list[str],
    target_names: list[str],
    mapping: dict[str, str] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return target slots and availability; missing joints remain NaN.

    mapping maps target names to source names. No pose, axes, or scale are
    inferred. Whole-character retargeting requires target rig bind transforms.
    """
    values = np.asarray(joints, float)
    if values.shape[-2:] != (len(source_names), 3) or len(set(source_names)) != len(source_names):
        raise ValueError("Distinct source names must match joints")
    lookup = {name: index for index, name in enumerate(source_names)}
    result = np.full(values.shape[:-2] + (len(target_names), 3), np.nan)
    available = np.zeros(len(target_names), dtype=bool)
    for index, name in enumerate(target_names):
        source = (mapping or {}).get(name, name)
        if source in lookup:
            result[..., index, :] = values[..., lookup[source], :]
            available[index] = True
    return result, available
