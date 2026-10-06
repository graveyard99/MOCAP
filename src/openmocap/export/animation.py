from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from openmocap.fitting import Animation
from openmocap.geometry.coordinates import conversion_matrix


class ExportError(RuntimeError):
    """An exporter or a genuine post-export validation failed."""


def _uniform_animation(animation: Animation, fps: float | None) -> float:
    differences = np.diff(animation.times)
    inferred = 30.0 if not len(differences) else 1 / float(np.median(differences))
    selected = float(fps or inferred)
    if selected <= 0 or not np.isfinite(selected):
        raise ValueError("Export FPS must be finite and positive")
    if len(differences) and not np.allclose(differences, 1 / selected, rtol=1e-3, atol=1e-6):
        raise ValueError("Resample motion to the selected uniform FPS before exporting")
    return selected


def _bvh(animation: Animation, output: Path, fps: float) -> dict[str, Any]:
    root = int(np.flatnonzero(animation.parents == -1)[0])
    order: list[int] = []
    lines = ["HIERARCHY"]

    def joint_block(joint: int, depth: int) -> None:
        indent = "  " * depth
        parent = int(animation.parents[joint])
        name = animation.names[joint].replace(" ", "_")
        lines.extend([f"{indent}{'ROOT' if parent == -1 else 'JOINT'} {name}", f"{indent}{{"])
        offset = (
            np.zeros(3)
            if parent == -1
            else animation.rest_joints[joint] - animation.rest_joints[parent]
        )
        lines.append(f"{indent}  OFFSET " + " ".join(f"{value:.9f}" for value in offset))
        channels = (
            "6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
            if parent == -1
            else "3 Zrotation Xrotation Yrotation"
        )
        lines.append(f"{indent}  CHANNELS {channels}")
        order.append(joint)
        children = np.flatnonzero(animation.parents == joint)
        for child in children:
            joint_block(int(child), depth + 1)
        if not len(children):
            lines.extend(
                [
                    f"{indent}  End Site",
                    f"{indent}  {{",
                    f"{indent}    OFFSET 0 0.02 0",
                    f"{indent}  }}",
                ]
            )
        lines.append(f"{indent}}}")

    joint_block(root, 0)
    lines.extend(["MOTION", f"Frames: {len(animation.times)}", f"Frame Time: {1 / fps:.12f}"])
    for frame in range(len(animation.times)):
        values = animation.translations[frame].tolist()
        for joint in order:
            values.extend(
                Rotation.from_rotvec(animation.rotations[frame, joint])
                .as_euler("ZXY", degrees=True)
                .tolist()
            )
        lines.append(" ".join(f"{value:.9f}" for value in values))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # BVH is a hierarchy/animation-only interchange. It has no skinning mesh.
    return {
        "format": "bvh",
        "validated": True,
        "validation": "writer hierarchy and frame dimensions",
        "mesh": False,
        "skeleton": True,
        "skinning": False,
        "units": "metres",
        "world_up": "+Y",
    }


def export_animation(
    animation: Animation,
    path: str | Path,
    format: str | None = None,
    *,
    blender: str | Path | None = None,
    fps: float | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Write a persistent rig with animated local rotations and global motion.

    FBX and USD use Blender; FBX validates by importing into a clean scene. NPZ
    retains exact rig/shape/pose arrays and body pose-corrective coefficients.
    Alembic is deliberately not exposed as a rigged-character solution.
    """
    output = Path(path).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = (format or output.suffix.removeprefix(".")).lower()
    selected_fps = _uniform_animation(animation, fps)
    metadata: dict[str, Any] = {
        "path": str(output),
        "model": animation.model_name,
        "frames": len(animation.times),
        "fps": selected_fps,
        "frame_start": 1,
        "frame_end": len(animation.times),
        "units": "metres",
        "world_up": "+Y",
    }
    if selected == "npz":
        animation.save(output)
        with np.load(output, allow_pickle=False) as data:
            if data["rotations"].shape != animation.rotations.shape:
                raise ExportError("NPZ animation roundtrip failed")
        metadata.update(format="npz", validated=True, skeleton=True, mesh=True, skinning=True)
    elif selected == "bvh":
        metadata.update(_bvh(animation, output, selected_fps))
    elif selected in {"fbx", "usd", "usda", "usdc"}:
        executable = str(blender or shutil.which("blender") or "")
        if not executable:
            raise ExportError(
                "Blender not found; set blender= or install a portable Blender in INSTALL_ROOT/tools"
            )
        script = Path(__file__).with_name("blender_export.py")
        with tempfile.TemporaryDirectory(prefix="export-", dir=output.parent) as work:
            workdir = Path(work)
            arrays = workdir / "animation.npz"
            animation.save(arrays)
            report = workdir / "validation.json"
            command = [
                executable,
                "--background",
                "--factory-startup",
                "--python",
                str(script),
                "--",
                str(arrays),
                str(output),
                selected,
                str(selected_fps),
                str(report),
                str(int(validate)),
                json.dumps(conversion_matrix("world", "blender").tolist()),
            ]
            try:
                process = subprocess.run(
                    command, capture_output=True, text=True, timeout=600, check=False
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ExportError(f"Blender backend could not complete: {error}") from error
            if process.returncode or not output.is_file() or not report.is_file():
                raise ExportError(
                    f"Blender export failed ({process.returncode}):\n{process.stdout[-4000:]}\n{process.stderr[-4000:]}"
                )
            validation = json.loads(report.read_text())
            if validate and not validation.get("validated", False):
                raise ExportError(f"Export re-import validation failed: {validation}")
            metadata.update(validation)
            metadata.update(
                blender=executable,
                format=selected,
                pose_corrective_export="animated relative shape keys"
                if animation.posedirs is not None
                else "not required",
            )
    else:
        raise ExportError(f"Unsupported character format {selected!r}; use fbx, npz, bvh or usd")
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
