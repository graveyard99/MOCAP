"""Project-local desktop preferences and non-destructive edit history."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def install_root() -> Path:
    return Path(os.environ.get("OPENMOCAP_INSTALL_ROOT", Path(__file__).resolve().parents[4]))


class Preferences:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or install_root() / "cache" / "ui" / "preferences.json"
        try:
            self.data: dict[str, Any] = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2))
        temporary.replace(self.path)

    def remember(self, path: str) -> None:
        recent = [path] + [p for p in self.data.get("recent", []) if p != path]
        self.data["recent"] = recent[:12]
        self.save()


def value(record: Any, key: str, default: Any = None) -> Any:
    return record.get(key, default) if isinstance(record, dict) else getattr(record, key, default)


def project_path(project: Any) -> Path:
    candidate = value(project, "path") or value(project, "project_file")
    if candidate:
        path = Path(candidate)
        return path.parent if path.suffix in {".json", ".yaml", ".yml"} else path
    raise ValueError("Project service returned no project path")


def camera_id(camera: Any) -> str:
    return str(value(camera, "camera_id", value(camera, "id", "unknown")))


def cameras(project: Any) -> list[Any]:
    config = value(project, "config", project)
    return list(value(config, "cameras", value(project, "cameras", [])))
