"""Versioned raw landmark interchange and non-destructive overrides."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from openmocap import SCHEMA_VERSION, __version__
from openmocap.pipeline.cache import atomic_json
from openmocap.types import Pose2DObservation


def load_observations(
    path: str | Path, *, camera_ids: set[str] | None = None
) -> list[Pose2DObservation]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError("Unsupported observation schema version")
        payload = payload.get("observations")
    if not isinstance(payload, list):
        raise ValueError("Observation JSON must be a list or contain an observations list")
    observations = [Pose2DObservation.from_dict(row) for row in payload]
    if camera_ids is not None:
        unknown = {o.camera_id for o in observations} - camera_ids
        if unknown:
            raise ValueError(f"Unknown camera IDs in observations: {sorted(unknown)}")
    return observations


def save_observations(
    path: str | Path, observations: list[Pose2DObservation], metadata: dict[str, Any] | None = None
) -> None:
    atomic_json(
        Path(path),
        {
            "schema_version": SCHEMA_VERSION,
            "code_version": __version__,
            "metadata": metadata or {},
            "observations": [o.to_dict() for o in observations],
        },
    )


def apply_observation_overrides(
    records: list[dict[str, Any]], overrides: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Explicit camera/time/person/joint selectors; retains original raw arrays.

    Edits accept xy, confidence or disabled fields only. Wildcard omission is
    deliberate for camera/time-range exclusions; every override has an audit ID.
    """
    result = copy.deepcopy(records)
    for override in overrides:
        if not override.get("id"):
            raise ValueError("Manual observation override requires an audit ID")
        changes = override.get("changes", {})
        if not set(changes) <= {"xy", "confidence", "disabled"}:
            raise ValueError("Observation overrides may edit xy, confidence, disabled only")
        selector = override.get("selector", {})
        for row in result:
            if any(
                row.get(key) != selector[key]
                for key in ("camera_id", "person_id", "joint_id")
                if key in selector
            ):
                continue
            time = float(row["camera_timestamp"])
            if "start" in selector and time < selector["start"]:
                continue
            if "end" in selector and time > selector["end"]:
                continue
            row.update(copy.deepcopy(changes))
            row.setdefault("applied_override_ids", []).append(override["id"])
            Pose2DObservation.from_dict(row)
    return result
