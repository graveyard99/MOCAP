#!/usr/bin/env python3
"""Fetch public personal-experiment inference assets from publisher-hosted URLs.

No scripts from archives are executed. Only ONNX models and JSON deployment
metadata are extracted. No body models, gated accounts, or license agreements
are accepted by this command. Installed weights remain ignored by Git.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
ASSETS = {
    "pose": {
        "asset_id": "rtmw-dw-x-l-cocktail14-256x192-20231122",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
        "destination": "models/pose/rtmw_wholebody133.onnx",
        "terms_urls": [
            "https://github.com/open-mmlab/mmpose/blob/main/LICENSE",
            "https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose",
            "https://github.com/IDEA-Research/DWPose/blob/main/LICENSE",
        ],
        "terms_summary": "MMPose code Apache-2.0; public author-hosted pretrained asset. "
        "Training dataset/image rights and weight redistribution are distinct from code licensing. "
        "Installed for user-authorized personal noncommercial experimentation, not redistributed.",
    },
    "detection": {
        "asset_id": "yolox-m-humanart-c2c7a14a",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        "destination": "models/detection/yolox_m_humanart.onnx",
        "terms_urls": [
            "https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE",
            "https://github.com/open-mmlab/mmpose/blob/main/LICENSE",
            "https://github.com/IDEA-Research/HumanArt",
        ],
        "terms_summary": "YOLOX/MMPose code Apache-2.0; public author-hosted HumanArt-trained checkpoint. "
        "Dataset rights remain separate. Installed for personal noncommercial experimentation; "
        "not bundled into open-source distribution.",
    },
    "segmentation": {
        "asset_id": "opencv-zoo-pphumanseg-2023mar-float32",
        "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx",
        "destination": "models/segmentation/human_segmentation_pphumanseg_2023mar.onnx",
        "format": "onnx",
        "expected_sha256": "552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24",
        "expected_bytes": 6163938,
        "terms_urls": [
            "https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg",
            "https://github.com/opencv/opencv_zoo/blob/main/models/human_segmentation_pphumanseg/LICENSE",
        ],
        "terms_summary": "Official OpenCV Zoo PPHumanSeg directory Apache-2.0. Publisher Git LFS pointer supplies SHA256 and byte size. Installed asset remains outside Git.",
    },
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fetch_asset(name: str, root: Path = ROOT, timeout: int = 45) -> dict:
    """Idempotent file+observed-hash manifest, with bounded curl requests."""
    spec = ASSETS[name]
    archive = root / "downloads" / Path(spec["url"]).name
    destination = root / spec["destination"]
    manifest_path = destination.parent / "manifest.json"
    if destination.exists() and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (
            previous.get("sha256") == digest(destination)
            and previous.get("source_url") == spec["url"]
        ):
            return previous
        raise ValueError(
            f"Existing model/manifest mismatch: {destination}; inspect before replacing"
        )
    if not shutil.which("curl"):
        raise RuntimeError("curl is required for bounded HTTPS downloads")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix(archive.suffix + ".partial")
        try:
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--connect-timeout",
                    "10",
                    "--max-time",
                    str(timeout),
                    "--silent",
                    "--show-error",
                    "--output",
                    str(temporary),
                    spec["url"],
                ],
                check=True,
                timeout=timeout + 5,
            )
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if spec.get("format") == "onnx":
        if (
            digest(archive) != spec["expected_sha256"]
            or archive.stat().st_size != spec["expected_bytes"]
        ):
            raise ValueError("Downloaded ONNX does not match published Git LFS SHA256 and size")
        temporary_model = destination.with_suffix(".onnx.partial")
        try:
            shutil.copyfile(archive, temporary_model)
            temporary_model.replace(destination)
        finally:
            temporary_model.unlink(missing_ok=True)
    else:
        _extract_archive(archive, destination)
    record = {
        "asset_id": spec["asset_id"],
        "source_url": spec["url"],
        "file": str(destination),
        "archive_file": str(archive),
        "archive_sha256": digest(archive),
        "bytes": destination.stat().st_size,
        "sha256": digest(destination),
        "publisher_checksum": spec.get("expected_sha256"),
        "checksum_origin": "publisher Git LFS SHA256 verified"
        if spec.get("expected_sha256")
        else "locally observed SHA256, not publisher attestation",
        "terms_urls": spec["terms_urls"],
        "terms_summary": spec["terms_summary"],
        "review_date": datetime.now(timezone.utc).isoformat(),
        "inference_verification": {
            "executed": False,
            "reason": "Run installed model acceptance test after download",
        },
    }
    manifest_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def _extract_archive(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        model_members = [m for m in bundle.infolist() if m.filename.lower().endswith(".onnx")]
        chosen = [m for m in model_members if Path(m.filename).name == "end2end.onnx"]
        if len(chosen) != 1:
            chosen = model_members
        if len(chosen) != 1:
            raise ValueError(
                f"Archive must contain one identified ONNX model; found {[m.filename for m in model_members]}"
            )
        member = chosen[0]
        if member.file_size > 2 * 1024**3:
            raise ValueError("Unreasonably large model archive member")
        temporary_model = destination.with_suffix(".onnx.partial")
        try:
            with bundle.open(member) as incoming, temporary_model.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            temporary_model.replace(destination)
        finally:
            temporary_model.unlink(missing_ok=True)
        for metadata in bundle.infolist():
            if (
                Path(metadata.filename).name in {"pipeline.json", "deploy.json", "detail.json"}
                and metadata.file_size < 2 * 1024**2
            ):
                (destination.parent / Path(metadata.filename).name).write_bytes(
                    bundle.read(metadata)
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--asset", choices=["all", *ASSETS], default="all")
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()
    if args.timeout < 5 or args.timeout > 180:
        parser.error("Use bounded per-asset download timeout between 5 and 180 seconds")
    root = args.root.expanduser().resolve()
    try:
        for name in ASSETS if args.asset == "all" else [args.asset]:
            record = fetch_asset(name, root, args.timeout)
            print(
                json.dumps(
                    {
                        "asset": name,
                        "file": record["file"],
                        "sha256": record["sha256"],
                        "bytes": record["bytes"],
                    }
                )
            )
    except (OSError, ValueError, subprocess.SubprocessError, zipfile.BadZipFile) as error:
        print(f"Model acquisition failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
