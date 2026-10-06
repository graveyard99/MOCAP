"""Separate observed contact evidence from optional corrections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.special import expit


FOOT_NAMES = ("left_heel", "left_toe", "right_heel", "right_toe")
ALIASES = {
    "left_heel": ("left_heel", "left_ankle"),
    "left_toe": ("left_toe", "left_foot", "left_big_toe"),
    "right_heel": ("right_heel", "right_ankle"),
    "right_toe": ("right_toe", "right_foot", "right_big_toe"),
}


@dataclass(frozen=True)
class ContactSettings:
    height_threshold_m: float = 0.06
    speed_threshold_m_s: float = 0.25
    height_softness_m: float = 0.015
    speed_softness_m_s: float = 0.08
    enter_probability: float = 0.60
    exit_probability: float = 0.35
    minimum_duration_s: float = 0.08
    minimum_confidence: float = 0.2


@dataclass
class ContactResult:
    times: np.ndarray
    probability: np.ndarray
    planted: np.ndarray
    joint_indices: tuple[int | None, ...]
    intervals: list[dict[str, Any]]
    floor_normal: np.ndarray
    floor_offset: float
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "times": self.times.tolist(),
            "foot_names": list(FOOT_NAMES),
            "probability": self.probability.tolist(),
            "planted": self.planted.tolist(),
            "joint_indices": list(self.joint_indices),
            "intervals": self.intervals,
            "floor_normal": self.floor_normal.tolist(),
            "floor_offset": self.floor_offset,
            "diagnostics": self.diagnostics,
        }


@dataclass
class RefinementResult:
    positions: np.ndarray
    diagnostics: dict[str, Any]


def _plane(floor: dict[str, Any] | None) -> tuple[np.ndarray, float]:
    floor = floor or {"normal": [0, 1, 0], "offset": 0}
    normal = np.asarray(floor.get("normal", [0, 1, 0]), dtype=float)
    norm = np.linalg.norm(normal)
    if normal.shape != (3,) or not np.isfinite(normal).all() or norm < 1e-10:
        raise ValueError("Floor requires a finite nonzero normal")
    offset = float(floor.get("offset", 0))
    if not np.isfinite(offset):
        raise ValueError("Floor offset must be finite")
    return normal / norm, offset / norm


def _validate(
    times: np.ndarray, joints: np.ndarray, confidence: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    joints = np.asarray(joints, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "Contact timestamps must be finite and strictly increasing, with at least two samples"
        )
    if joints.ndim != 3 or joints.shape[0] != len(times) or joints.shape[2] != 3:
        raise ValueError("Expected joints[N,J,3]")
    quality = (
        np.ones(joints.shape[:2]) if confidence is None else np.asarray(confidence, dtype=float)
    )
    if (
        quality.shape != joints.shape[:2]
        or not np.isfinite(quality).all()
        or np.any((quality < 0) | (quality > 1))
    ):
        raise ValueError("Confidence must be finite N x J values in [0,1]")
    return times, joints, quality


def _intervals(times: np.ndarray, planted: np.ndarray) -> list[dict[str, Any]]:
    intervals = []
    for foot in range(4):
        change = np.diff(np.r_[False, planted[:, foot], False].astype(int))
        for start, end in zip(
            np.flatnonzero(change == 1), np.flatnonzero(change == -1), strict=True
        ):
            intervals.append(
                {
                    "foot": FOOT_NAMES[foot],
                    "foot_index": foot,
                    "start_index": int(start),
                    "end_index": int(end - 1),
                    "start": float(times[start]),
                    "end": float(times[end - 1]),
                }
            )
    return intervals


def estimate_contacts(
    times: np.ndarray,
    joints: np.ndarray,
    joint_names: list[str] | tuple[str, ...],
    *,
    confidence: np.ndarray | None = None,
    floor: dict[str, Any] | None = None,
    settings: ContactSettings | None = None,
) -> ContactResult:
    """Infer heel/forefoot contact from floor proximity, speed, and confidence.

    Standard SMPL ankles/feet can be used as proxies, explicitly recorded in
    diagnostics. A true heel/toe detector is preferable for production contact.
    Slow motion away from the calibrated floor cannot become planted contact.
    """
    times, joints, quality = _validate(times, joints, confidence)
    if len(joint_names) != joints.shape[1]:
        raise ValueError("Joint-name count differs from motion")
    settings = settings or ContactSettings()
    if (
        min(
            settings.height_threshold_m,
            settings.speed_threshold_m_s,
            settings.height_softness_m,
            settings.speed_softness_m_s,
        )
        <= 0
    ):
        raise ValueError("Contact thresholds and softness must be positive")
    if not 0 <= settings.exit_probability < settings.enter_probability <= 1:
        raise ValueError("Contact hysteresis needs 0 <= exit < enter <= 1")
    normal, offset = _plane(floor)
    lookup = {name: index for index, name in enumerate(joint_names)}
    indices = tuple(
        next((lookup[name] for name in ALIASES[foot] if name in lookup), None)
        for foot in FOOT_NAMES
    )
    probability = np.zeros((len(times), 4))
    planted = np.zeros((len(times), 4), dtype=bool)
    proxies = {}
    for foot, joint in enumerate(indices):
        if joint is None:
            proxies[FOOT_NAMES[foot]] = "unobserved"
            continue
        proxies[FOOT_NAMES[foot]] = joint_names[joint]
        position = joints[:, joint]
        valid = np.isfinite(position).all(axis=1)
        if np.count_nonzero(valid) < 2:
            continue
        filled = np.column_stack(
            [np.interp(times, times[valid], position[valid, dimension]) for dimension in range(3)]
        )
        velocity = np.gradient(filled, times, axis=0)
        tangent = velocity - np.outer(velocity @ normal, normal)
        speed = np.linalg.norm(tangent, axis=1)
        vertical_speed = np.abs(velocity @ normal)
        height = np.abs(filled @ normal + offset)
        probability[:, foot] = (
            expit((settings.height_threshold_m - height) / settings.height_softness_m)
            * expit((settings.speed_threshold_m_s - speed) / settings.speed_softness_m_s)
            * expit((settings.speed_threshold_m_s - vertical_speed) / settings.speed_softness_m_s)
        )
        probability[:, foot] *= quality[:, joint]
        probability[~valid | (quality[:, joint] < settings.minimum_confidence), foot] = 0
        active = False
        for frame, likelihood in enumerate(probability[:, foot]):
            active = likelihood >= (
                settings.exit_probability if active else settings.enter_probability
            )
            planted[frame, foot] = active
    for interval in _intervals(times, planted):
        if interval["end"] - interval["start"] < settings.minimum_duration_s:
            planted[interval["start_index"] : interval["end_index"] + 1, interval["foot_index"]] = (
                False
            )
    intervals = _intervals(times, planted)
    return ContactResult(
        times,
        probability,
        planted,
        indices,
        intervals,
        normal,
        offset,
        {
            "settings": asdict(settings),
            "joint_sources": proxies,
            "floor_required": True,
            "source": "floor_velocity_hysteresis",
        },
    )


def _sliding(times: np.ndarray, positions: np.ndarray, contacts: ContactResult) -> dict[str, float]:
    metrics = {}
    for foot, joint in enumerate(contacts.joint_indices):
        if joint is None:
            continue
        differences = np.diff(positions[:, joint], axis=0)
        tangent = differences - np.outer(differences @ contacts.floor_normal, contacts.floor_normal)
        valid = (
            contacts.planted[:-1, foot]
            & contacts.planted[1:, foot]
            & np.isfinite(tangent).all(axis=1)
        )
        metrics[FOOT_NAMES[foot]] = (
            float(np.mean(np.linalg.norm(tangent[valid], axis=1) / np.diff(times)[valid]))
            if valid.any()
            else 0.0
        )
    return metrics


def apply_contact_overrides(
    contacts: ContactResult, overrides: list[dict[str, Any]]
) -> ContactResult:
    """Return effective contact intervals with auditable, reversible user edits.

    Each override supplies foot, start/end world seconds and a planted boolean.
    Raw detector/contact results remain untouched. Explicit manual states do
    not relax the geometric displacement guards in optional refinement.
    """
    probability = contacts.probability.copy()
    planted = contacts.planted.copy()
    validated = []
    for entry in overrides:
        foot = entry.get("foot")
        if foot not in FOOT_NAMES:
            raise ValueError(f"Unknown manual contact foot {foot}")
        start, end = float(entry["start"]), float(entry["end"])
        if not np.isfinite([start, end]).all() or start > end:
            raise ValueError("Contact override requires finite increasing world times")
        state = entry["planted"]
        if not isinstance(state, bool):
            raise ValueError("Manual planted state must be a boolean")
        selection = (contacts.times >= start) & (contacts.times <= end)
        index = FOOT_NAMES.index(foot)
        planted[selection, index] = state
        probability[selection, index] = float(state)
        validated.append(
            {"foot": foot, "start": start, "end": end, "planted": state, "source": "manual"}
        )
    diagnostics = dict(contacts.diagnostics)
    diagnostics["manual_overrides"] = validated
    return ContactResult(
        contacts.times.copy(),
        probability,
        planted,
        contacts.joint_indices,
        _intervals(contacts.times, planted),
        contacts.floor_normal.copy(),
        contacts.floor_offset,
        diagnostics,
    )


def refine_contacts(
    times: np.ndarray,
    joints: np.ndarray,
    confidence: np.ndarray,
    contacts: ContactResult,
    *,
    max_correction_m: float = 0.05,
    strong_threshold: float = 0.8,
    strong_max_correction_m: float = 0.002,
) -> RefinementResult:
    """Reduce observed planted-foot drift within confidence-governed bounds.

    Root and non-foot joints remain measured. This is a bounded kinematic
    correction, not a full-body dynamics solve; downstream IK/body fitting should
    consume the corrected targets to preserve anatomical consistency.
    """
    times, positions, quality = _validate(times, joints, confidence)
    if not np.array_equal(times, contacts.times):
        raise ValueError("Contact states and motion must share actual timestamps")
    if max_correction_m < 0 or strong_max_correction_m < 0:
        raise ValueError("Contact correction bounds must be nonnegative")
    result = positions.copy()
    corrected = 0
    for interval in contacts.intervals:
        foot = interval["foot_index"]
        joint = contacts.joint_indices[foot]
        if joint is None:
            continue
        sl = slice(interval["start_index"], interval["end_index"] + 1)
        sample = positions[sl, joint]
        valid = np.isfinite(sample).all(axis=1)
        if not valid.any():
            continue
        weights = np.maximum(quality[sl, joint][valid], 0.05)
        anchor = np.average(sample[valid], axis=0, weights=weights)
        anchor -= contacts.floor_normal * (anchor @ contacts.floor_normal + contacts.floor_offset)
        for index in range(interval["start_index"], interval["end_index"] + 1):
            current = positions[index, joint]
            if not np.isfinite(current).all():
                continue
            # Low confidence does not justify an unbounded physical invention.
            limit = (
                strong_max_correction_m
                if quality[index, joint] >= strong_threshold
                else max_correction_m * (1 - 0.5 * quality[index, joint])
            )
            delta = anchor - current
            magnitude = float(np.linalg.norm(delta))
            if magnitude > limit and magnitude > 0:
                delta *= limit / magnitude
            result[index, joint] += delta
            corrected += int(np.linalg.norm(delta) > 1e-12)
    displacement = np.linalg.norm(result - positions, axis=2)
    maximum = (
        float(np.max(displacement[np.isfinite(displacement)]))
        if np.isfinite(displacement).any()
        else 0.0
    )
    return RefinementResult(
        result,
        {
            "mode": "bounded_kinematic_contact_targets",
            "before_slide_m_s": _sliding(times, positions, contacts),
            "after_slide_m_s": _sliding(times, result, contacts),
            "corrected_samples": corrected,
            "max_displacement_m": maximum,
            "strong_measurement_max_correction_m": strong_max_correction_m,
            "root_preserved": True,
            "requires_downstream_ik": True,
        },
    )
