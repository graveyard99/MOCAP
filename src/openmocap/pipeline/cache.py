"""Content-addressed stage checkpoints with atomic completion manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from openmocap import SCHEMA_VERSION, __version__


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(config: dict[str, Any], inputs: list[Path]) -> str:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode())
    digest.update(f"{SCHEMA_VERSION}:{__version__}".encode())
    for source in sorted(inputs):
        digest.update(str(source).encode())
        digest.update(file_hash(source).encode())
    for source in sorted(Path(__file__).parents[1].rglob("*.py")):
        digest.update(source.read_bytes())
    lock = Path(__file__).parents[3] / "uv.lock"
    if lock.exists():
        digest.update(lock.read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    os.replace(temporary, path)


class CheckpointStore:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def valid(self, stage: str, key: str, outputs: list[Path]) -> bool:
        path = self.directory / f"{stage}.json"
        if not path.exists() or not all(p.is_file() for p in outputs):
            return False
        try:
            manifest = json.loads(path.read_text())
            return (
                manifest["schema_version"] == SCHEMA_VERSION
                and manifest["key"] == key
                and all(manifest["hashes"].get(str(p)) == file_hash(p) for p in outputs)
            )
        except (KeyError, ValueError, OSError):
            return False

    def complete(self, stage: str, key: str, outputs: list[Path]) -> None:
        atomic_json(
            self.directory / f"{stage}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "code_version": __version__,
                "key": key,
                "outputs": [str(p) for p in outputs],
                "hashes": {str(p): file_hash(p) for p in outputs},
            },
        )
