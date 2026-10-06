"""Sub-frame signal alignment; camera time never implies common frame numbers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from openmocap.types import TimeMapping


@dataclass(frozen=True)
class TimestampDiagnostics:
    count: int
    duplicate_indices: tuple[int, ...]
    discontinuity_indices: tuple[int, ...]
    dropped_after_indices: tuple[int, ...]
    nominal_interval: float | None

    @property
    def valid(self) -> bool:
        return not self.discontinuity_indices


def validate_timestamps(
    timestamps: np.ndarray,
    *,
    allow_duplicates: bool = True,
    allow_discontinuities: bool = False,
    expected_interval: float | None = None,
) -> TimestampDiagnostics:
    times = np.asarray(timestamps, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("Timestamps must be a one-dimensional finite array")
    deltas = np.diff(times)
    duplicate = tuple(int(i + 1) for i in np.flatnonzero(deltas == 0))
    discontinuity = tuple(int(i + 1) for i in np.flatnonzero(deltas < 0))
    positive = deltas[deltas > 0]
    nominal = (
        expected_interval
        if expected_interval is not None
        else (float(np.median(positive)) if len(positive) else None)
    )
    if nominal is not None and (not np.isfinite(nominal) or nominal <= 0):
        raise ValueError("Expected timestamp interval must be positive")
    dropped = tuple(int(i) for i in np.flatnonzero(deltas > 1.5 * nominal)) if nominal else ()
    if duplicate and not allow_duplicates:
        raise ValueError(f"Duplicate timestamps at indices {duplicate}")
    if discontinuity and not allow_discontinuities:
        raise ValueError(
            f"Timestamp discontinuity at indices {discontinuity}; split/rebase source explicitly"
        )
    return TimestampDiagnostics(len(times), duplicate, discontinuity, dropped, nominal)


def _unique_signal(times: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    validate_timestamps(times)
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.ndim != 2 or len(signal) != len(times) or not np.isfinite(signal).all():
        raise ValueError("Signal must contain finite values with one row per timestamp")
    unique, inverse = np.unique(times, return_inverse=True)
    counts = np.bincount(inverse)
    averaged = np.column_stack(
        [np.bincount(inverse, weights=signal[:, d]) / counts for d in range(signal.shape[1])]
    )
    if len(unique) < 6:
        raise ValueError("Synchronization needs at least six unique timestamps")
    return unique, averaged


def estimate_from_events(
    camera_events: np.ndarray,
    world_events: np.ndarray,
    *,
    initial: TimeMapping | None = None,
    estimate_drift: bool = True,
    scale_bounds: tuple[float, float] = (0.999, 1.001),
    event_sigma_s: float = 0.002,
) -> tuple[TimeMapping, dict[str, float | int | bool]]:
    """Align matched claps/flashes/timecode/control events, with robust priors.

    Event identities must already correspond; this function never silently
    matches unrelated events. One event solves offset only. Two or more events
    may estimate clock drift when explicitly enabled.
    """
    if initial is not None and initial.locked:
        return initial, {"locked": True, "offset_delta": 0.0, "scale_delta": 0.0}
    camera = np.asarray(camera_events, dtype=float)
    world = np.asarray(world_events, dtype=float)
    if (
        camera.ndim != 1
        or world.shape != camera.shape
        or not len(camera)
        or not np.isfinite(camera).all()
        or not np.isfinite(world).all()
    ):
        raise ValueError("Synchronization events need matching nonempty finite timestamp arrays")
    validate_timestamps(camera, allow_duplicates=False)
    validate_timestamps(world, allow_duplicates=False)
    if event_sigma_s <= 0 or scale_bounds[0] <= 0 or scale_bounds[0] >= scale_bounds[1]:
        raise ValueError("Event uncertainty and affine scale bounds must be positive")
    drift = estimate_drift and len(camera) > 1
    seed_scale = initial.scale if initial is not None else 1.0
    centre = float(camera.mean())
    offset_at_centre = float(np.median(world - camera * seed_scale)) + seed_scale * centre
    if drift:

        def residual(parameters: np.ndarray) -> np.ndarray:
            return ((camera - centre) * parameters[1] + parameters[0] - world) / event_sigma_s

        fit = least_squares(
            residual,
            [offset_at_centre, np.clip(seed_scale, *scale_bounds)],
            bounds=([-np.inf, scale_bounds[0]], [np.inf, scale_bounds[1]]),
            loss="soft_l1",
        )
        scale = float(fit.x[1])
        offset = float(fit.x[0] - scale * centre)
    else:
        scale = seed_scale
        offset = float(np.median(world - camera * scale))
    residual_seconds = camera * scale + offset - world
    rmse = float(np.sqrt(np.mean(residual_seconds**2)))
    confidence = float(np.clip(0.95 * np.exp(-rmse / event_sigma_s), 0, 0.95))
    result = TimeMapping(
        scale=scale,
        offset=offset,
        source="matched_sync_events",
        locked=False,
        confidence=confidence,
    )
    return result, {
        "locked": False,
        "event_count": len(camera),
        "drift_estimated": drift,
        "rmse_seconds": rmse,
        "offset_delta": offset - (initial.offset if initial is not None else 0),
        "scale_delta": scale - seed_scale,
    }


def estimate_time_mapping(
    camera_times: np.ndarray,
    camera_signal: np.ndarray,
    reference_times: np.ndarray,
    reference_signal: np.ndarray,
    *,
    offset_bounds: tuple[float, float] = (-1.0, 1.0),
    scale_bounds: tuple[float, float] = (0.999, 1.001),
    initial: TimeMapping | None = None,
    normalize: bool = False,
) -> TimeMapping:
    """Find t_world = scale * t_camera + offset from comparable signals.

    Suitable signals include LED intensity, audio envelopes, and shared motion
    descriptors. Signals must actually describe the same event; unrelated
    camera-space coordinates are not synchronization evidence. Bounds expose
    clock assumptions. A locked initial mapping is returned unchanged.
    """
    if initial is not None and initial.locked:
        return initial
    lower_offset, upper_offset = offset_bounds
    lower_scale, upper_scale = scale_bounds
    if lower_offset >= upper_offset or lower_scale <= 0 or lower_scale >= upper_scale:
        raise ValueError("Synchronization bounds must be increasing and scale positive")
    ct, cs = _unique_signal(np.asarray(camera_times, dtype=float), camera_signal)
    rt, rs = _unique_signal(np.asarray(reference_times, dtype=float), reference_signal)
    if cs.shape[1] != rs.shape[1]:
        raise ValueError("Camera and reference signals need matching dimensions")
    if normalize:
        cs = (cs - cs.mean(axis=0)) / np.maximum(cs.std(axis=0), 1e-9)
        rs = (rs - rs.mean(axis=0)) / np.maximum(rs.std(axis=0), 1e-9)
    if np.max(np.std(cs, axis=0)) < 1e-8 or np.max(np.std(rs, axis=0)) < 1e-8:
        raise ValueError("Constant signals provide no synchronization evidence")
    # Restrict to support valid for every bounded candidate. Optimizer cannot
    # improve its score by walking off the reference timeline.
    corners = np.stack([ct * s + o for s in scale_bounds for o in offset_bounds])
    safe = (corners.min(axis=0) >= rt[0]) & (corners.max(axis=0) <= rt[-1])
    if np.count_nonzero(safe) < 6:
        raise ValueError("Insufficient signal overlap for the requested synchronization bounds")
    ct, cs = ct[safe], cs[safe]
    reference = CubicSpline(rt, rs, axis=0, extrapolate=False)
    span = max(float(np.ptp(ct)), 1.0)

    def residual(parameters: np.ndarray) -> np.ndarray:
        offset, scaled_drift = parameters
        return (reference(ct * (1 + scaled_drift / span) + offset) - cs).ravel()

    scales = np.linspace(lower_scale, upper_scale, 5)
    offsets = np.linspace(lower_offset, upper_offset, 201)
    candidates = [
        (float(np.mean(residual(np.array([o, (s - 1) * span])) ** 2)), o, s)
        for s in scales
        for o in offsets
    ]
    candidates.sort(key=lambda x: x[0])
    if initial is not None:
        candidates.insert(
            0,
            (
                0.0,
                np.clip(initial.offset, lower_offset, upper_offset),
                np.clip(initial.scale, lower_scale, upper_scale),
            ),
        )
    bounds = ([lower_offset, (lower_scale - 1) * span], [upper_offset, (upper_scale - 1) * span])
    fits = [
        least_squares(
            residual,
            [o, (s - 1) * span],
            bounds=bounds,
            loss="soft_l1",
            f_scale=0.1,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        for _, o, s in candidates[:3]
    ]
    best = min(fits, key=lambda x: float(np.mean(x.fun**2)))
    normalized_rmse = float(np.sqrt(np.mean(best.fun**2)) / max(np.std(cs), 1e-9))
    quality = float(np.clip(0.95 * np.exp(-normalized_rmse), 0, 0.95))
    return TimeMapping(
        scale=float(1 + best.x[1] / span),
        offset=float(best.x[0]),
        locked=False,
        source="signal_alignment",
        confidence=quality,
    )


def refine_time_mapping(
    mapping: TimeMapping,
    reprojection_residual: Callable[[float, float], np.ndarray],
    *,
    max_offset_change: float = 0.02,
    max_drift_change: float = 0.0001,
    offset_prior_sigma: float = 0.01,
    drift_prior_sigma: float = 0.0001,
) -> tuple[TimeMapping, dict[str, float | bool]]:
    """Bounded joint-assisted timing refinement with priors and lock ownership."""
    if mapping.locked:
        return mapping, {"locked": True, "offset_delta": 0.0, "scale_delta": 0.0}
    if min(max_offset_change, max_drift_change, offset_prior_sigma, drift_prior_sigma) <= 0:
        raise ValueError("Timing bounds and prior sigmas must be positive")

    def residual(delta: np.ndarray) -> np.ndarray:
        observed = np.asarray(
            reprojection_residual(mapping.offset + delta[0], mapping.scale + delta[1]), dtype=float
        ).ravel()
        if not np.isfinite(observed).all():
            raise ValueError("Reprojection timing residual must be finite")
        return np.r_[observed, delta[0] / offset_prior_sigma, delta[1] / drift_prior_sigma]

    fit = least_squares(
        residual,
        [0.0, 0.0],
        bounds=([-max_offset_change, -max_drift_change], [max_offset_change, max_drift_change]),
        loss="huber",
    )
    result = TimeMapping(
        scale=mapping.scale + float(fit.x[1]),
        offset=mapping.offset + float(fit.x[0]),
        locked=False,
        source="bounded_reprojection_refinement",
        confidence=mapping.confidence,
    )
    return result, {
        "locked": False,
        "offset_delta": float(fit.x[0]),
        "scale_delta": float(fit.x[1]),
        "initial_cost": float(np.mean(residual(np.zeros(2)) ** 2)),
        "final_cost": float(np.mean(fit.fun**2)),
    }
