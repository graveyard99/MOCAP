"""Inventory package metadata and wheel license texts without network access."""

from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/licenses/installed-review.json"))
    args = parser.parse_args()
    entries = []
    for dist in sorted(
        importlib.metadata.distributions(), key=lambda d: d.metadata.get("Name", "")
    ):
        licenses = []
        for file in dist.files or []:
            if any(
                part.lower().startswith(("license", "licence", "copying")) for part in file.parts
            ):
                path = Path(dist.locate_file(file))
                if path.is_file():
                    raw = path.read_bytes()
                    licenses.append(
                        {
                            "file": str(file),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "bytes": len(raw),
                        }
                    )
        entries.append(
            {
                "name": dist.metadata.get("Name"),
                "version": dist.version,
                "license_expression": dist.metadata.get("License-Expression"),
                "license": dist.metadata.get("License"),
                "classifiers": [
                    x for x in dist.metadata.get_all("Classifier", []) if x.startswith("License")
                ],
                "license_files_reviewed": licenses,
                "review_method": "installed authoritative wheel metadata and license file inventory",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": 1, "packages": entries}, indent=2) + "\n")
    print(f"Inventoried {len(entries)} installed distributions: {args.output}")


if __name__ == "__main__":
    main()
