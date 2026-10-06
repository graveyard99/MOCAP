"""Machine-readable and reviewable reprojection QC from original observations."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from openmocap.pipeline.cache import atomic_json


def _statistics(errors: list[float]) -> dict[str, Any]:
    values = np.asarray(errors)
    if not len(values):
        return {"count": 0, "mean_px": None, "median_px": None, "p95_px": None}
    return {
        "count": len(values),
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(np.max(values)),
    }


def build_report(
    cameras: list,
    records: list[dict],
    trajectory: Any,
    names: list[str],
    diagnostics: list[dict],
    *,
    actor_id: str,
    shape: dict,
    contacts: dict,
    provenance: dict,
    overrides: dict,
) -> dict:
    lookup = {c.id: c for c in cameras}
    camera_errors, joint_errors, frame_errors = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    measurements, rejected = [], 0
    for row in records:
        if row["camera_id"] not in lookup or str(row.get("person_id", actor_id)) != actor_id:
            continue
        name = str(row["joint_id"])
        if name not in names:
            continue
        camera = lookup[row["camera_id"]]
        time = float(camera.time_mapping.to_world(row["camera_timestamp"]))
        xyz = trajectory.evaluate(time)[names.index(name)]
        if not np.isfinite(xyz).all():
            continue
        pixel = np.asarray(row.get("xy", [row.get("x"), row.get("y")]))
        predicted = camera.project(xyz[None], time=time)[0]
        error = float(np.linalg.norm(predicted - pixel))
        if not np.isfinite(error):
            continue
        key = f"{camera.id}|{name}|{float(time):.12f}"
        bad = (
            row.get("disabled", False)
            or not row.get("enabled", True)
            or row.get("confidence", 1) < 0.05
            or camera.quality.confidence <= 0
            or key in trajectory.diagnostics.get("native_rejected_measurements", [])
        )
        flagged = error > 8
        rejected += int(bad)
        camera_errors[camera.id].append(error)
        joint_errors[name].append(error)
        frame_errors[f"{time:.6f}"].append(error)
        measurements.append(
            {
                "camera_id": camera.id,
                "time": time,
                "joint_id": name,
                "error_px": error,
                "measured": pixel.tolist(),
                "reprojected": predicted.tolist(),
                "rejected": bad,
                "qc_flagged": flagged,
                "confidence": row.get("confidence", 1),
            }
        )
    per_camera = {c: _statistics(e) for c, e in camera_errors.items()}
    warnings = [
        f"{c}: median reprojection {s['median_px']:.2f} px; inspect calibration, synchronization and identity"
        for c, s in per_camera.items()
        if s["median_px"] is not None and s["median_px"] > 4
    ]
    if shape.get("mean_joint_residual_m", shape.get("joint_rmse_m", 0)) > 0.03:
        warnings.append(
            "Body fit residual exceeds 3 cm; inspect skeleton mapping and actor proportions"
        )
    all_errors = [e for values in camera_errors.values() for e in values]
    report = {
        "schema_version": 1,
        "camera_count": len(cameras),
        "used_observations": len(measurements) - rejected,
        "rejected_observations": rejected,
        "missing_joint_sample_percentage": 100
        * float(np.mean([d.get("confidence", 0) == 0 for d in diagnostics]))
        if diagnostics
        else 100,
        "qc_flagged_observations": sum(m["qc_flagged"] for m in measurements),
        "observation_classification": "native outlier gate, manual/disabled and input confidence; QC residual flags reported separately",
        "reprojection": _statistics(all_errors),
        "per_camera": per_camera,
        "per_joint": {j: _statistics(e) for j, e in joint_errors.items()},
        "per_time": {t: _statistics(e) for t, e in frame_errors.items()},
        "measurements": measurements,
        "triangulation": diagnostics,
        "calibration": {
            c.id: {"source": c.source, "locked": c.locked, "quality": c.quality.confidence}
            for c in cameras
        },
        "synchronization": {c.id: c.time_mapping.to_dict() for c in cameras},
        "shape": shape,
        "silhouette_residuals": None,
        "contacts": contacts,
        "warnings": warnings,
        "manual_overrides": overrides,
        "provenance": provenance,
        "exports": [],
        "evidence_policy": [
            "surveyed calibration",
            "robust multiview",
            "temporal observations",
            "body model",
            "physics",
            "learned priors",
            "monocular estimates",
        ],
    }
    return clean_json(report)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_report(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", clean_json(report))
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{metrics['count']}</td>"
        f"<td>{metrics['mean_px']:.3f}</td><td>{metrics['median_px']:.3f}</td><td>{metrics['p95_px']:.3f}</td></tr>"
        for name, metrics in report["per_camera"].items()
        if metrics["count"]
    )
    warning_text = "".join(f"<li>{html.escape(str(w))}</li>" for w in report["warnings"])
    payload = html.escape(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"measurements", "triangulation", "per_time"}
            },
            indent=2,
        )
    )
    page = f"""<!doctype html><html><meta charset="utf-8"><title>OpenMocap · solve QC</title>
<style>body{{background:#101923;color:#d9e4ec;font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}}h1{{color:#67dcc7}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #354355}}pre{{white-space:pre-wrap;background:#182735;padding:20px}}.cards{{display:flex;gap:30px}}</style>
<h1>OpenMocap / solve review</h1><p>Metric world · metres · +Y up · authoritative camera geometry</p>
<div class="cards"><p><b>{report["camera_count"]}</b> cameras</p><p><b>{report["used_observations"]}</b> observations used</p><p><b>{report["rejected_observations"]}</b> observations flagged</p></div>
<h2>Camera reprojection</h2><table><tr><th>Camera</th><th>Samples</th><th>Mean px</th><th>Median px</th><th>95% px</th></tr>{rows}</table>
<h2>Actionable warnings</h2><ul>{warning_text or "<li>No threshold warnings; inspect the residual distributions before delivery.</li>"}</ul>
<p>Original residuals include rejected observations. Full joint/time/measurement data: <a href="report.json">report.json</a>.</p>
<h2>Shape, contacts, locks and reproducibility</h2><pre>{payload}</pre></html>"""
    (output / "report.html").write_text(page)
