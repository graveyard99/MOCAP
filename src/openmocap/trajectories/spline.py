"""Weighted cubic smoothing splines with measured-evidence displacement guards."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline, make_smoothing_spline


@dataclass
class ContinuousTrajectory:
    """Continuous metre-valued trajectories, one independent spline per joint.

    The spline is supported only over measured timestamps. Missing joints and
    extrapolation are returned as NaN unless extrapolation is explicitly enabled.
    Confidence is interpolated from the original measurements, not invented by
    the smoothing stage.
    """

    splines: list[Any]
    measured_times: list[np.ndarray]
    measured_confidence: list[np.ndarray]
    joint_names: tuple[str, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    extrapolate: bool = False

    def save(self, path: str | Path) -> None:
        """Persist exact spline coefficients without pickle or executable data."""
        from scipy.interpolate import BSpline

        arrays: dict[str, Any] = {
            "schema_version": np.array(1),
            "joint_names": np.array(self.joint_names),
            "diagnostics_json": np.array(json.dumps(self.diagnostics)),
            "extrapolate": np.array(self.extrapolate),
        }
        for joint, spline in enumerate(self.splines):
            arrays[f"times_{joint}"] = self.measured_times[joint]
            arrays[f"confidence_{joint}"] = self.measured_confidence[joint]
            kind = (
                "missing"
                if spline is None
                else ("bspline" if isinstance(spline, BSpline) else "piecewise_cubic")
            )
            arrays[f"kind_{joint}"] = np.array(kind)
            if spline is not None:
                arrays[f"coefficients_{joint}"] = spline.c
                arrays[f"knots_{joint}"] = spline.t if kind == "bspline" else spline.x
                if kind == "bspline":
                    arrays[f"degree_{joint}"] = np.array(spline.k)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> ContinuousTrajectory:
        """Load a versioned exact trajectory; reject incompatible schema."""
        from scipy.interpolate import BSpline, PPoly

        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("Unsupported continuous trajectory checkpoint schema")
            names = tuple(str(value) for value in data["joint_names"])
            splines, times, confidence = [], [], []
            for joint in range(len(names)):
                kind = str(data[f"kind_{joint}"])
                times.append(data[f"times_{joint}"].copy())
                confidence.append(data[f"confidence_{joint}"].copy())
                if kind == "missing":
                    splines.append(None)
                elif kind == "bspline":
                    splines.append(
                        BSpline(
                            data[f"knots_{joint}"].copy(),
                            data[f"coefficients_{joint}"].copy(),
                            int(data[f"degree_{joint}"]),
                        )
                    )
                elif kind == "piecewise_cubic":
                    splines.append(
                        PPoly(data[f"coefficients_{joint}"].copy(), data[f"knots_{joint}"].copy())
                    )
                else:
                    raise ValueError(f"Unsupported trajectory spline type {kind}")
            diagnostics = json.loads(str(data["diagnostics_json"]))
            extrapolate = bool(data["extrapolate"])
        return cls(splines, times, confidence, names, diagnostics, extrapolate)

    def evaluate(self, times: np.ndarray | float, derivative: int = 0) -> np.ndarray:
        query = np.atleast_1d(np.asarray(times, dtype=float))
        result = np.full((len(query), len(self.splines), 3), np.nan)
        for index, spline in enumerate(self.splines):
            if spline is None:
                continue
            source = self.measured_times[index]
            valid = np.ones(len(query), dtype=bool)
            if not self.extrapolate:
                valid = (query >= source[0]) & (query <= source[-1])
            result[valid, index] = spline(query[valid], derivative)
        return result[0] if np.ndim(times) == 0 else result

    def confidence(self, times: np.ndarray | float) -> np.ndarray:
        query = np.atleast_1d(np.asarray(times, dtype=float))
        result = np.zeros((len(query), len(self.splines)))
        for index, measured in enumerate(self.measured_times):
            if len(measured):
                result[:, index] = np.interp(
                    query, measured, self.measured_confidence[index], left=0, right=0
                )
                if len(measured) > 1:
                    # A cubic curve across an occlusion is inferred motion;
                    # support confidence must fall with distance from evidence.
                    insertion = np.searchsorted(measured, query)
                    left = measured[np.clip(insertion - 1, 0, len(measured) - 1)]
                    right = measured[np.clip(insertion, 0, len(measured) - 1)]
                    distance = np.minimum(np.abs(query - left), np.abs(query - right))
                    support = max(float(np.median(np.diff(measured))) * 2, 1e-9)
                    result[:, index] *= np.exp(-np.maximum(distance - support, 0) / support)
        return result[0] if np.ndim(times) == 0 else result

    def sample(
        self, fps: float, start: float | None = None, end: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Output FPS must be positive and finite")
        observed = [x for x in self.measured_times if len(x)]
        if not observed:
            raise ValueError("Trajectory has no measured support")
        first = min(x[0] for x in observed) if start is None else start
        last = max(x[-1] for x in observed) if end is None else end
        if last < first:
            raise ValueError("Output end precedes start")
        times = first + np.arange(int(np.floor((last - first) * fps + 1e-8)) + 1) / fps
        return times, self.evaluate(times)


def fit_trajectory(
    times: np.ndarray,
    positions: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    smoothing: float = 1e-5,
    strong_threshold: float = 0.8,
    max_strong_displacement_m: float = 0.002,
    joint_names: list[str] | tuple[str, ...] | None = None,
    extrapolate: bool = False,
) -> ContinuousTrajectory:
    """Fit cubic trajectories using actual world timestamps and confidence.

    Positions are N x J x 3 (N x 3 is accepted for one joint). Duplicates are
    merged by confidence-weighted average without mutating caller data. A
    conservative displacement guard reduces smoothing until high-confidence
    samples move by at most the specified metric bound. Regularization cannot
    silently override strong observations.
    """
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 2:
        positions = positions[:, None, :]
    if (
        times.ndim != 1
        or positions.ndim != 3
        or positions.shape != (len(times), positions.shape[1], 3)
    ):
        raise ValueError("Expected times[N] and positions[N,J,3]")
    if not np.isfinite(times).all():
        raise ValueError("Trajectory timestamps must be finite")
    if smoothing < 0 or max_strong_displacement_m < 0:
        raise ValueError("Smoothing and evidence displacement bound must be nonnegative")
    quality = (
        np.ones(positions.shape[:2]) if confidence is None else np.asarray(confidence, dtype=float)
    )
    if quality.ndim == 1 and positions.shape[1] == 1:
        quality = quality[:, None]
    if (
        quality.shape != positions.shape[:2]
        or not np.isfinite(quality).all()
        or np.any((quality < 0) | (quality > 1))
    ):
        raise ValueError("Confidence must be finite N x J values in [0,1]")
    names = tuple(joint_names or [f"joint_{i}" for i in range(positions.shape[1])])
    if len(names) != positions.shape[1]:
        raise ValueError("Joint-name count differs from position array")
    splines, source_times, source_confidence, diagnostics = [], [], [], []
    for joint in range(positions.shape[1]):
        valid = np.isfinite(positions[:, joint]).all(axis=1) & (quality[:, joint] > 0)
        t, p, w = times[valid], positions[valid, joint], quality[valid, joint]
        order = np.argsort(t, kind="stable")
        t, p, w = t[order], p[order], w[order]
        unique, inverse = np.unique(t, return_inverse=True)
        if len(unique) != len(t):
            weight_sum = np.bincount(inverse, weights=w)
            p = np.column_stack(
                [np.bincount(inverse, weights=w * p[:, d]) / weight_sum for d in range(3)]
            )
            w = np.minimum(1, weight_sum)
            t = unique
        source_times.append(t)
        source_confidence.append(w)
        if len(t) < 2:
            splines.append(None)
            diagnostics.append(
                {"joint": names[joint], "samples": len(t), "warning": "insufficient observations"}
            )
            continue
        lam = smoothing
        max_shift = 0.0
        for _attempt in range(13):
            if len(t) < 5 or lam == 0:
                spline = CubicSpline(t, p, extrapolate=True)
            else:
                # SciPy's GCV implementation returns scalar splines; combine
                # their coefficient arrays into a vector-valued BSpline.
                from scipy.interpolate import BSpline

                coordinates = [
                    make_smoothing_spline(t, p[:, d], w=np.maximum(w, 1e-5), lam=lam)
                    for d in range(3)
                ]
                spline = BSpline(
                    coordinates[0].t, np.column_stack([s.c for s in coordinates]), coordinates[0].k
                )
            strong = w >= strong_threshold
            max_shift = (
                float(np.max(np.linalg.norm(spline(t[strong]) - p[strong], axis=1)))
                if strong.any()
                else 0.0
            )
            if max_shift <= max_strong_displacement_m + 1e-12:
                break
            lam /= 10
        else:
            spline = CubicSpline(t, p, extrapolate=True)
            lam, max_shift = 0.0, 0.0
        splines.append(spline)
        diagnostics.append(
            {
                "joint": names[joint],
                "samples": len(t),
                "requested_smoothing": smoothing,
                "effective_smoothing": lam,
                "max_strong_displacement_m": max_shift,
            }
        )
    return ContinuousTrajectory(
        splines,
        source_times,
        source_confidence,
        names,
        {"joints": diagnostics, "units": "metres"},
        extrapolate,
    )
