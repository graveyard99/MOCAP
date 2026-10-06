"""Shared project and processing API for desktop and command-line clients."""

from __future__ import annotations

import copy
import importlib
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Callable

import numpy as np
import yaml

from openmocap import SCHEMA_VERSION, __version__
from openmocap.pipeline.cache import CheckpointStore, atomic_json, file_hash, fingerprint

LOG = logging.getLogger(__name__)
PRESETS = {
    "Preview": {"max_hypotheses": 16, "max_iterations": 40},
    "Standard": {"max_hypotheses": 32, "max_iterations": 80},
    "High Quality": {"max_hypotheses": 64, "max_iterations": 100},
    "Maximum / Final": {"max_hypotheses": 128, "max_iterations": 200},
}
STAGES = [
    "ingest",
    "calibrate",
    "sync",
    "detect",
    "pose2d",
    "segment",
    "associate",
    "triangulate",
    "fit-shape",
    "fit-motion",
    "contacts",
    "physics",
    "qc",
    "export",
]


def installation_root() -> Path:
    return Path(os.environ.get("OPENMOCAP_INSTALL_ROOT", Path(__file__).resolve().parents[3]))


def configure_isolation() -> Path:
    root = installation_root()
    locations = {
        "PIP_CACHE_DIR": "cache/pip",
        "UV_CACHE_DIR": "cache/uv",
        "HF_HOME": "cache/huggingface",
        "HF_HUB_CACHE": "cache/huggingface/hub",
        "TRANSFORMERS_CACHE": "cache/huggingface/transformers",
        "TORCH_HOME": "models/torch",
        "XDG_CACHE_HOME": "cache",
        "XDG_CONFIG_HOME": "cache/config",
        "XDG_DATA_HOME": "cache/data",
        "MPLCONFIGDIR": "cache/matplotlib",
        "CUDA_CACHE_PATH": "cache/cuda",
        "TMPDIR": "temp",
    }
    for variable, relative in locations.items():
        destination = root / relative
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(destination)
    os.environ["OPENMOCAP_INSTALL_ROOT"] = str(root)
    return root


def _set_nested(record: dict, parts: list[str], value: Any) -> None:
    for part in parts[:-1]:
        record = record.setdefault(part, {})
    if value is None:
        record.pop(parts[-1], None)
    else:
        record[parts[-1]] = copy.deepcopy(value)


def _rig_joint_positions(animation: Any, names: list[str]) -> np.ndarray:
    points = np.full((len(animation.times), len(names), 3), np.nan)
    for index, name in enumerate(names):
        if name in animation.names:
            points[:, index] = animation.joints[:, animation.names.index(name)]
    return points


class ProjectService:
    def __init__(self) -> None:
        configure_isolation()

    def create_project(
        self, path: str | Path, name: str = "Capture", camera_count: int = 2, **settings: Any
    ) -> dict:
        directory = Path(path).expanduser().resolve()
        if (directory / "project.yaml").exists():
            raise FileExistsError(f"Project already exists: {directory}")
        if camera_count < 2 or not name.strip():
            raise ValueError("Provide a project name and at least two cameras")
        directory.mkdir(parents=True, exist_ok=True)
        for folder in ["media", "calibration", "raw", "cache", "overrides", "outputs", "exports"]:
            (directory / folder).mkdir(exist_ok=True)
        project = {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "path": str(directory),
            "project_file": str(directory / "project.yaml"),
            "world": {
                "units": "metres",
                "up": "Y",
                "metric_scale": None,
                "origin": [0, 0, 0],
                "floor": {"normal": [0, 1, 0], "offset": 0},
            },
            "cameras": [
                {
                    "id": f"CAM_{i + 1:02d}",
                    "source": "default",
                    "source_path": None,
                    "enabled": True,
                    "locked": True,
                    "fps": 30,
                    "time_mapping": {
                        "scale": 1,
                        "offset": 0,
                        "locked": True,
                        "source": "manual",
                        "confidence": 1,
                    },
                }
                for i in range(camera_count)
            ],
            "actor": {"id": "actor01", "name": "Actor", "body_model": "smpl"},
            "body_model": "smpl",
            "output_fps": 30,
            "output_dir": "outputs",
            "observations": "raw/observations.json",
            "overrides": {},
            "stages": {stage: {"status": "NOT RUN"} for stage in STAGES},
        }
        project.update({k: v for k, v in settings.items() if v is not None})
        prefix = settings.get("camera_prefix", "CAM_")
        for index, camera in enumerate(project["cameras"]):
            camera["id"] = f"{prefix}{index + 1:02d}"
            camera["fps"] = float(settings.get("source_fps", 30))
        if settings.get("ground_y") is not None:
            project["world"]["floor"]["offset"] = -float(settings["ground_y"])
        if settings.get("actor_name"):
            project["actor"]["name"] = settings["actor_name"]
        if settings.get("metric_scale_source") == "Calibrated metres":
            project["world"]["metric_scale"] = 1.0
            project["world"]["scale_source"] = "manual metres confirmation"
        self.save_project(project)
        return project

    def open_project(self, path: str | Path) -> dict:
        source = Path(path).expanduser().resolve()
        if source.is_dir():
            source = source / "project.yaml"
        project = yaml.safe_load(source.read_text())
        if not isinstance(project, dict) or project.get("schema_version", 1) != SCHEMA_VERSION:
            raise ValueError("Unsupported or malformed project configuration schema")
        project["path"], project["project_file"] = str(source.parent), str(source)
        project.setdefault("overrides", {})
        project.setdefault("override_source_hashes", {})
        project.setdefault("stages", {s: {"status": "NOT RUN"} for s in STAGES})
        project.setdefault("body_model", project.get("actor", {}).get("body_model", "smpl"))
        return project

    def save_project(self, project: dict) -> None:
        source = Path(project["project_file"])
        source.parent.mkdir(parents=True, exist_ok=True)
        temporary = source.with_suffix(".yaml.tmp")
        temporary.write_text(yaml.safe_dump(project, sort_keys=False))
        temporary.replace(source)

    def create_example(self, path: str | Path, **kwargs: Any) -> dict:
        from openmocap.synthetic import generate_capture

        source = generate_capture(path, **kwargs)
        project = self.open_project(source)
        self.save_project(project)
        return project

    def duplicate_project(self, project: dict, path: str | Path) -> dict:
        destination = Path(path).resolve()
        if destination.exists():
            raise FileExistsError("Duplicate destination must not exist")
        shutil.copytree(
            project["path"],
            destination,
            ignore=shutil.ignore_patterns("cache", "outputs", "exports"),
        )
        duplicate = self.open_project(destination)
        duplicate["name"] += " copy"
        self.invalidate(duplicate, "ingest")
        self.save_project(duplicate)
        return duplicate

    def invalidate(self, project: dict, stage: str = "triangulate") -> None:
        start = STAGES.index(stage) if stage in STAGES else 0
        for name in STAGES[start:]:
            if project["stages"].get(name, {}).get("status") in {"COMPLETE", "WARNING", "STALE"}:
                project["stages"][name]["status"] = "STALE"
        self.save_project(project)

    def apply_override(self, project: dict, key: str, value: Any) -> None:
        if not key or key.startswith(("path", "project_file", "schema_version")):
            raise ValueError("Overrides may edit capture parameters, not project identity")
        parts = key.split(".")
        target = project
        if parts[0] == "camera":
            if len(parts) < 3:
                raise ValueError("Camera override format is camera.CAM_01.parameter")
            target = next((c for c in project["cameras"] if c["id"] == parts[1]), None)
            if target is None:
                raise ValueError(f"Unknown camera ID {parts[1]}")
            parts = parts[2:]
        old = copy.deepcopy(project)
        baseline_path = Path(project["path"]) / "overrides" / "baseline.json"
        if not baseline_path.exists():
            atomic_json(baseline_path, old)
        effective_value = value
        if value is None:
            baseline = json.loads(baseline_path.read_text())
            baseline_target = baseline
            if key.startswith("camera."):
                imported_path = Path(project["path"]) / "calibration" / "imported.json"
                original_cameras = (
                    json.loads(imported_path.read_text())["cameras"]
                    if imported_path.exists()
                    else baseline.get("cameras", [])
                )
                baseline_target = next(
                    (c for c in original_cameras if c["id"] == key.split(".")[1]), {}
                )
            for part in parts:
                baseline_target = (
                    baseline_target.get(part) if isinstance(baseline_target, dict) else None
                )
            effective_value = baseline_target
        _set_nested(target, parts, effective_value)
        try:
            self._validate_edit(project)
        except Exception:
            project.clear()
            project.update(old)
            raise
        if key == "ground_y":
            project["world"]["floor"]["offset"] = -float(value or 0)
        if value is None:
            project["overrides"].pop(key, None)
        else:
            project["overrides"][key] = copy.deepcopy(value)
        if key.startswith("observation."):
            hashes = project.setdefault("override_source_hashes", {})
            source = self._resolve(project, project.get("observations", "raw/observations.json"))
            if value is None or value is False:
                hashes.pop(key, None)
            elif source.exists():
                hashes[key] = file_hash(source)
        atomic_json(Path(project["path"]) / "overrides" / "manual.json", project["overrides"])
        self.invalidate(
            project,
            "detect" if key == "perception" or key.startswith("perception.") else "triangulate",
        )
        self.save_project(project)

    def _validate_edit(self, project: dict) -> None:
        from openmocap.types import Camera, TimeMapping

        if (
            not np.isfinite(float(project.get("output_fps", 30)))
            or float(project.get("output_fps", 30)) <= 0
        ):
            raise ValueError("Output frame rate must be finite and positive")
        for camera in project["cameras"]:
            TimeMapping(**camera.get("time_mapping", {}))
            if camera.get("intrinsics") and camera.get("pose"):
                parsed = Camera.from_dict(camera)
                media = camera.get("media_index")
                if isinstance(media, dict) and (media["width"], media["height"]) != (
                    parsed.intrinsics.width,
                    parsed.intrinsics.height,
                ):
                    raise ValueError(
                        f"Camera {camera['id']} media resolution differs from calibrated intrinsics"
                    )

    def import_camera(
        self, project: dict, camera_id: str, source: str | Path, **metadata: Any
    ) -> None:
        from openmocap.io.media import ingest_media

        camera = next((c for c in project["cameras"] if c["id"] == camera_id), None)
        if camera is None:
            raise ValueError(f"Unknown camera {camera_id}")
        index = ingest_media(Path(source), nominal_fps=metadata.get("fps", camera.get("fps", 30)))
        camera["source_path"] = str(Path(source).resolve())
        camera["media_index"] = index.to_dict() if hasattr(index, "to_dict") else str(index)
        camera.update(metadata)
        self.invalidate(project, "ingest")
        self.save_project(project)

    def import_calibration(self, project: dict, path: str | Path) -> None:
        from openmocap.types import Camera

        imported = yaml.safe_load(Path(path).read_text())
        data = imported.get("cameras", []) if isinstance(imported, dict) else imported
        parsed = [Camera.from_dict(c) for c in data]
        existing = {c["id"]: c for c in project["cameras"]}
        for camera in parsed:
            if camera.id not in existing:
                raise ValueError(f"Calibration ID {camera.id} is not registered in project")
        raw_path = Path(project["path"]) / "calibration" / "imported.json"
        atomic_json(raw_path, {"source": str(Path(path).resolve()), "cameras": data})
        for camera in parsed:
            source_path = existing[camera.id].get("source_path")
            existing[camera.id].update(camera.to_dict())
            if source_path:
                existing[camera.id]["source_path"] = source_path
        if isinstance(imported, dict) and "world" in imported:
            project["world"] = imported["world"]
        # Manual overrides are the effective final layer above trusted imports.
        for key, datum in project.get("overrides", {}).items():
            parts = key.split(".")
            if parts[0] == "camera" and parts[1] in existing:
                _set_nested(existing[parts[1]], parts[2:], datum)
            elif parts[0] == "world":
                _set_nested(project, parts, datum)
        self._validate_edit(project)
        self.invalidate(project, "calibrate")
        self.save_project(project)

    def _resolve(self, project: dict, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else Path(project["path"]) / path

    def _status(
        self, project: dict, stage: str, status: str, message: str, callback: Callable | None
    ) -> None:
        project["stages"][stage] = {"status": status, "message": message}
        self.save_project(project)
        LOG.info(message, extra={"stage": stage, "status": status})
        if callback:
            callback(
                {
                    "stage": stage,
                    "status": status,
                    "message": message,
                    "progress": 1.0 if status == "COMPLETE" else 0.0,
                }
            )

    def validate(self, project: dict) -> list:
        from openmocap.types import Camera

        self._validate_edit(project)
        world = project.get("world", {})
        if world.get("units") != "metres" or world.get("up") not in ("Y", "+Y"):
            raise ValueError(
                "Import world conventions explicitly to metres, +Y before reconstruction"
            )
        if world.get("metric_scale") != 1.0:
            raise ValueError(
                "Metric reconstruction requires cameras aligned to metres (metric_scale=1). Apply surveyed scale before solving; no automatic height scale is invented"
            )
        for c in project["cameras"]:
            if c.get("enabled", True) and (not c.get("intrinsics") or not c.get("pose")):
                raise ValueError(
                    f"Camera {c['id']} is uncalibrated. Import or measure its intrinsics and metric world transform"
                )
        cameras = [Camera.from_dict(c) for c in project["cameras"] if c.get("enabled", True)]
        if len(cameras) < 2 or len({c.id for c in cameras}) != len(cameras):
            raise ValueError("At least two distinct enabled calibrated cameras are required")
        if any(not c.quality.metric_scale_known for c in cameras):
            raise ValueError("Camera calibration lacks authoritative metric scale")
        return cameras

    def run(
        self,
        project: dict,
        callback: Callable | None = None,
        cancel_event: Any = None,
        stage: str | None = None,
    ) -> dict:
        from openmocap.fitting import fit_actor, load_animation
        from openmocap.pipeline.reconstruction import reconstruct, sample_plate_evidence
        from openmocap.qc import build_report, write_report
        from openmocap.trajectories import ContinuousTrajectory
        from openmocap.contacts import estimate_contacts

        output = self._resolve(project, project.get("output_dir", "outputs"))
        output.mkdir(parents=True, exist_ok=True)
        store = CheckpointStore(Path(project["path"]) / "cache" / "checkpoints")
        current = stage or "triangulate"
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Job cancelled")
            if stage == "ingest":
                for camera in project["cameras"]:
                    if camera.get("source_path"):
                        self.import_camera(project, camera["id"], camera["source_path"])
                self._status(
                    project, stage, "COMPLETE", "Registered media and actual timestamps", callback
                )
                return {"output_dir": str(output)}
            if stage in {"detect", "pose2d", "segment", "associate"}:
                return self._run_perception(project, stage, callback, cancel_event)
            cameras = self.validate(project)
            if stage == "sync" and any(not c.time_mapping.locked for c in cameras):
                prior_trajectory = output / "trajectory.npz"
                if prior_trajectory.exists():
                    from openmocap.sync import refine_multiview_timing
                    from openmocap.types import Pose2DObservation

                    source = self._resolve(project, project["observations"])
                    raw = json.loads(source.read_text())
                    rows = raw.get("observations", []) if isinstance(raw, dict) else raw
                    observations = [
                        Pose2DObservation.from_dict(row)
                        for row in rows
                        if row["camera_id"] in {c.id for c in cameras}
                    ]
                    solution = refine_multiview_timing(
                        cameras,
                        observations,
                        ContinuousTrajectory.load(prior_trajectory),
                        project["joint_names"],
                    )
                    for c in project["cameras"]:
                        if c["id"] in solution.mappings and not c.get("time_mapping", {}).get(
                            "locked", True
                        ):
                            c["time_mapping"] = solution.mappings[c["id"]].to_dict()
                    atomic_json(
                        Path(project["path"]) / "calibration" / "sync_refinement.json",
                        solution.diagnostics,
                    )
                    self.invalidate(project, "triangulate")
                else:
                    self._status(
                        project,
                        "sync",
                        "WARNING",
                        "No frozen trajectory exists for automatic refinement. Set initial affine clocks then reconstruct",
                        callback,
                    )
                    return {"output_dir": str(output), "refined": False}
            if stage in {"calibrate", "sync"}:
                message = (
                    "Imported calibrated camera solution validated; no camera parameters modified"
                    if stage == "calibrate"
                    else "Explicit affine world time mappings validated; locked timing preserved"
                )
                self._status(project, stage, "COMPLETE", message, callback)
                return {"output_dir": str(output), "camera_count": len(cameras)}
            source = self._resolve(project, project["observations"])
            if project.get("perception", {}).get("pose_model") and (
                stage is None or not source.is_file()
            ):
                self._run_perception(project, "pose2d", callback, cancel_event)
                source = self._resolve(project, project["observations"])
            if not source.is_file():
                raise FileNotFoundError(
                    "2D observations missing. Import timestamped keypoints or configure detector and pose ONNX models and run pose2d"
                )
            source_digest = file_hash(source)
            for correction, expected_hash in project.get("override_source_hashes", {}).items():
                if source_digest != expected_hash:
                    raise ValueError(
                        f"Raw observations changed after {correction}. Restore the old observation override and reapply it to the new raw output; automatic index reuse is prohibited"
                    )
            raw = json.loads(source.read_text())
            records = raw["observations"] if isinstance(raw, dict) else raw
            names = list(project.get("joint_names", []))
            if not names:
                names = sorted({str(r["joint_id"]) for r in records})
            selected_ids = {c.id for c in cameras}
            known_ids = {c["id"] for c in project["cameras"]}
            unknown_ids = {r["camera_id"] for r in records} - known_ids
            if unknown_ids:
                raise ValueError(f"Observations reference unknown cameras: {sorted(unknown_ids)}")
            actor_id = str(project.get("actor", {}).get("id", "actor01"))
            for index, record in enumerate(records):
                record["raw_observation_index"] = index
                if project["overrides"].get(f"observation.{index}.disabled", False):
                    record["disabled"] = True
                correction = project["overrides"].get(f"observation.{index}.xy")
                if correction is not None:
                    pixel = np.asarray(correction, float)
                    if pixel.shape != (2,) or not np.isfinite(pixel).all():
                        raise ValueError(f"Invalid manual pixel correction for observation {index}")
                    record["xy"] = pixel.tolist()
                    record["effective_source"] = "manual_correction"
            records = [r for r in records if r["camera_id"] in selected_ids]
            config = {
                k: v for k, v in project.items() if k not in {"stages", "project_file", "path"}
            }
            if stage is None:
                self._status(
                    project,
                    "calibrate",
                    "COMPLETE",
                    "Authoritative calibrated camera parameters validated",
                    callback,
                )
                self._status(
                    project,
                    "sync",
                    "COMPLETE",
                    "Explicit affine camera clocks applied to native observations",
                    callback,
                )
                if not project.get("perception", {}).get("pose_model"):
                    self._status(
                        project,
                        "pose2d",
                        "WARNING",
                        "Timestamped input observations used; provenance retained",
                        callback,
                    )
                self._status(
                    project,
                    "associate",
                    "COMPLETE",
                    "Associated actor IDs selected; cross-person triangulation prevented",
                    callback,
                )
            hash_inputs = [source]
            asset = project.get("model_path") or project.get("actor", {}).get("model_path")
            if asset and self._resolve(project, asset).is_file():
                hash_inputs.append(self._resolve(project, asset))
            key = fingerprint(config, hash_inputs)
            joint_file, trajectory_file = output / "joints.npz", output / "trajectory.npz"
            reconstruction = None
            current = "triangulate"
            self._status(
                project, current, "RUNNING", "Reconstructing metric continuous motion", callback
            )
            if store.valid(current, key, [joint_file, trajectory_file]):
                with np.load(joint_file, allow_pickle=False) as data:
                    times, joints, confidence = data["times"], data["joints"], data["confidence"]
                    diagnostics = json.loads(str(data["diagnostics"]))
                trajectory = ContinuousTrajectory.load(trajectory_file)
            else:
                reconstruction = reconstruct(
                    cameras,
                    records,
                    names,
                    fps=float(project["output_fps"]),
                    actor_id=actor_id,
                    callback=callback,
                    cancel_event=cancel_event,
                    solve_range=project.get("solve_range"),
                    triangulation_options={
                        **PRESETS.get(
                            project.get("solve_preset", "High Quality"), PRESETS["High Quality"]
                        ),
                        **project.get("solver", {}).get("triangulation", {}),
                    },
                    smoothing=float(project.get("solver", {}).get("smoothing", 1e-5)),
                )
                times, joints, confidence = (
                    reconstruction.times,
                    reconstruction.joints,
                    reconstruction.confidence,
                )
                diagnostics, trajectory = reconstruction.diagnostics, reconstruction.trajectory
                from openmocap.qc.report import clean_json

                np.savez_compressed(
                    joint_file,
                    times=times,
                    joints=joints,
                    confidence=confidence,
                    names=np.array(names),
                    diagnostics=np.array(json.dumps(clean_json(diagnostics))),
                )
                trajectory.save(trajectory_file)
                store.complete(current, key, [joint_file, trajectory_file])
            self._status(
                project,
                current,
                "COMPLETE",
                "All surviving views reconstructed; original timestamps fitted",
                callback,
            )
            if stage == "triangulate":
                return {"output_dir": str(output), "joints": str(joint_file)}
            current = "fit-motion"
            self._status(
                project,
                current,
                "RUNNING",
                "Fitting one persistent actor shape and animated rig",
                callback,
            )
            animation_file = output / "animation.npz"
            raw_animation_file = output / "body_pose.npz"
            if store.valid(current, key, [raw_animation_file]):
                animation = load_animation(raw_animation_file)
            else:
                model = project.get(
                    "body_model", project.get("actor", {}).get("body_model", "smpl")
                )
                model_path = project.get("model_path") or project.get("actor", {}).get("model_path")
                if model_path:
                    model_path = self._resolve(project, model_path)
                pixels, pixel_confidence = sample_plate_evidence(
                    cameras, records, names, times, actor_id
                )
                animation = fit_actor(
                    times,
                    joints,
                    confidence,
                    model=model,
                    model_path=model_path,
                    joint_names=names,
                    parents=project.get("parents"),
                    cameras=cameras,
                    observations2d=pixels,
                    confidence2d=pixel_confidence,
                    reprojection_weight=float(
                        project.get("solver", {}).get("body_reprojection_m_per_px", 0.001)
                    ),
                    max_nfev=int(project.get("solver", {}).get("body_max_nfev", 35)),
                    num_betas=int(project.get("solver", {}).get("num_betas", 10)),
                    strong_evidence_max_displacement_m=project.get("solver", {}).get(
                        "strong_evidence_max_displacement_m", 0.03
                    ),
                )
                animation.save(raw_animation_file)
                store.complete(current, key, [raw_animation_file])
            animation.save(animation_file)
            self._status(
                project,
                "fit-shape",
                "COMPLETE",
                "Actor proportions held constant throughout take",
                callback,
            )
            self._status(
                project,
                current,
                "COMPLETE",
                "Local rotations and global root trajectory solved",
                callback,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Job cancelled after body solve checkpoint")
            current = "contacts"
            floor = project["world"].get("floor", {"normal": [0, 1, 0], "offset": 0})
            contacts = estimate_contacts(times, joints, names, confidence=confidence, floor=floor)
            from openmocap.contacts import apply_contact_overrides
            from openmocap.contacts.feet import _sliding

            marked = list(project.get("contact_overrides", []))
            for foot, interval in project.get("contacts", {}).get("manual", {}).items():
                marked.append(
                    {
                        "foot": foot,
                        "start": interval["start"],
                        "end": interval["end"],
                        "planted": True,
                    }
                )
            if marked:
                contacts = apply_contact_overrides(contacts, marked)
            contacts.diagnostics["geometric_slide_m_s"] = _sliding(times, joints, contacts)
            contacts.diagnostics["fitted_slide_before_refinement_m_s"] = _sliding(
                times, _rig_joint_positions(animation, names), contacts
            )
            if stage == "physics" or project.get("physics", {}).get("enabled", False):
                from openmocap.physics import refine_kinematics

                refined = refine_kinematics(times, joints, confidence, contacts)
                animation = fit_actor(
                    times,
                    refined.positions,
                    confidence,
                    reference_animation=animation,
                    model=animation.model_name,
                    model_path=project.get("model_path")
                    or project.get("actor", {}).get("model_path"),
                    joint_names=names,
                    parents=project.get("parents"),
                    max_nfev=int(project.get("solver", {}).get("body_max_nfev", 35)),
                    strong_evidence_max_displacement_m=project.get("solver", {}).get(
                        "strong_evidence_max_displacement_m", 0.03
                    ),
                )
                animation.save(output / "animation_refined.npz")
                animation.save(animation_file)
                self._status(
                    project,
                    "physics",
                    "COMPLETE",
                    "Confidence-bounded kinematic refinement; no dynamics simulation",
                    callback,
                )
            elif stage == "contacts" or project.get("contact_refinement", False):
                from openmocap.contacts import refine_contacts

                refined = refine_contacts(times, joints, confidence, contacts)
                animation = fit_actor(
                    times,
                    refined.positions,
                    confidence,
                    reference_animation=animation,
                    model=animation.model_name,
                    model_path=project.get("model_path")
                    or project.get("actor", {}).get("model_path"),
                    joint_names=names,
                    parents=project.get("parents"),
                    max_nfev=int(project.get("solver", {}).get("body_max_nfev", 35)),
                    strong_evidence_max_displacement_m=project.get("solver", {}).get(
                        "strong_evidence_max_displacement_m", 0.03
                    ),
                )
                animation.save(animation_file)
                contacts.diagnostics["refinement"] = refined.diagnostics
            contacts.diagnostics["fitted_slide_after_refinement_m_s"] = _sliding(
                times, _rig_joint_positions(animation, names), contacts
            )
            contact_data = contacts.to_dict() if hasattr(contacts, "to_dict") else vars(contacts)
            from openmocap.qc.report import clean_json

            contact_data = clean_json(contact_data)
            atomic_json(output / "contacts.json", contact_data)
            self._status(
                project,
                current,
                "COMPLETE",
                "Floor-aware heel / forefoot contact intervals estimated",
                callback,
            )
            current = "qc"
            self._status(
                project,
                current,
                "RUNNING",
                "Computing native geometric and body reprojection residuals",
                callback,
            )
            provenance = self.reproducibility(project, source)
            report = build_report(
                cameras,
                records,
                trajectory,
                names,
                diagnostics,
                actor_id=actor_id,
                shape=animation.diagnostics,
                contacts=contact_data,
                provenance=provenance,
                overrides=project["overrides"],
            )
            # Body fit has its own original-plate residuals; excellent triangulation
            # residuals must never conceal a poorly fitted/exported character.
            from openmocap.trajectories import fit_trajectory

            fitted_names = animation.names
            body_trajectory = fit_trajectory(
                times,
                animation.joints,
                animation.confidence if animation.confidence is not None else None,
                joint_names=fitted_names,
                smoothing=0,
            )
            body_report = build_report(
                cameras,
                records,
                body_trajectory,
                fitted_names,
                [],
                actor_id=actor_id,
                shape=animation.diagnostics,
                contacts=contact_data,
                provenance=provenance,
                overrides=project["overrides"],
            )
            report["body_fit_reprojection"] = body_report["reprojection"]
            report["body_fit_per_camera"] = body_report["per_camera"]
            report["body_fit_per_joint"] = body_report["per_joint"]
            report["warnings"].extend(["Body fit: " + w for w in body_report["warnings"]])
            report["missing_joint_sample_percentage"] = float(100 * np.mean(confidence <= 0))
            write_report(report, output / "qc")
            atomic_json(
                output / "cameras.json",
                {"world": project["world"], "cameras": [c.to_dict() for c in cameras]},
            )
            self._status(
                project, current, "COMPLETE", "Original-plate reprojection report written", callback
            )
            return {
                "output_dir": str(output),
                "animation": str(animation_file),
                "report": str(output / "qc" / "report.html"),
                "camera_count": len(cameras),
                "frames": len(times),
                "reprojection": report["reprojection"],
            }
        except Exception as error:
            self._status(project, current, "FAILED", str(error), callback)
            raise

    def _run_perception(
        self, project: dict, stage: str, callback: Callable | None, cancel_event: Any
    ) -> dict:
        from openmocap.perception import run_project_perception

        result = run_project_perception(
            project, stage, callback=callback, cancel_event=cancel_event
        )
        self._status(
            project,
            stage,
            result.get("status", "COMPLETE"),
            result.get("message", "Inference completed"),
            callback,
        )
        self.save_project(project)
        return result

    def export(self, project: dict, format: str, path: str | Path, **options: Any) -> dict:
        from openmocap.fitting import load_animation
        from openmocap.export import export_animation

        dependencies = {
            "calibrate",
            "sync",
            "pose2d",
            "associate",
            "triangulate",
            "fit-shape",
            "fit-motion",
            "contacts",
            "qc",
        }
        if project.get("physics", {}).get("enabled"):
            dependencies.add("physics")
        if any(project["stages"].get(stage, {}).get("status") == "STALE" for stage in dependencies):
            raise ValueError("Solve has stale dependencies. Rerun the pipeline before exporting")
        output = self._resolve(project, project.get("output_dir", "outputs"))
        animation = load_animation(output / "animation.npz")
        from openmocap.export.resample import resample_animation

        animation = resample_animation(
            animation,
            float(options.get("fps", project["output_fps"])),
            options.get("start"),
            options.get("end"),
        )
        if options.get("include_mesh") is False and format not in {"bvh"}:
            raise ValueError(
                "Mesh-free export is currently available as BVH; rigged FBX requires its fitted mesh"
            )
        result = export_animation(
            animation,
            path,
            format=format,
            fps=options.get("fps", project["output_fps"]),
            blender=options.get("blender"),
            validate=True,
        )
        if options.get("include_cameras", False):
            destination = Path(path).with_suffix(".cameras.json")
            source = output / "cameras.json"
            if source.exists():
                shutil.copyfile(source, destination)
                result["camera_sidecar"] = str(destination)
        self._status(project, "export", "COMPLETE", f"{format.upper()} export validated", None)
        report_path = output / "qc" / "report.json"
        if report_path.exists():
            from openmocap.qc import write_report

            report = json.loads(report_path.read_text())
            report.setdefault("exports", []).append(result)
            write_report(report, report_path.parent)
        return result

    def load_result(self, project: dict) -> dict:
        output = self._resolve(project, project.get("output_dir", "outputs"))
        result: dict[str, Any] = {"output_dir": str(output)}
        if (output / "joints.npz").exists():
            with np.load(output / "joints.npz", allow_pickle=False) as data:
                result.update(
                    {k: data[k].copy() for k in ["times", "joints", "confidence", "names"]}
                )
        if "names" in result:
            result["joint_names"] = result["names"].tolist()
        if (output / "animation.npz").exists():
            from openmocap.fitting import load_animation

            animation = load_animation(output / "animation.npz")
            result.update(
                animation=animation,
                vertices=animation.vertices,
                faces=animation.faces,
                mesh_vertices=animation.vertices,
                mesh_faces=animation.faces,
                parents=animation.parents,
                shape=animation.diagnostics,
            )
        if (output / "qc" / "report.json").exists():
            result["report"] = json.loads((output / "qc" / "report.json").read_text())
        if (output / "contacts.json").exists():
            result["contacts"] = json.loads((output / "contacts.json").read_text())
        source = self._resolve(project, project.get("observations", "raw/observations.json"))
        if source.exists():
            raw = json.loads(source.read_text())
            result["observations"] = raw.get("observations", []) if isinstance(raw, dict) else raw
            source_digest = file_hash(source)
            for index, record in enumerate(result["observations"]):
                record["raw_observation_index"] = index
                provenance = project.get("override_source_hashes", {})
                key = f"observation.{index}.disabled"
                compatible = key not in provenance or provenance[key] == source_digest
                if compatible and project["overrides"].get(key, False):
                    record["disabled"] = True
                if f"observation.{index}.xy" in project["overrides"] and (
                    f"observation.{index}.xy" not in provenance
                    or provenance[f"observation.{index}.xy"] == source_digest
                ):
                    record["xy"] = project["overrides"][f"observation.{index}.xy"]
        return result

    def reproducibility(self, project: dict, source: Path) -> dict:
        repo = Path(__file__).parents[2]
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
        )
        lock = repo / "uv.lock"
        return {
            "code_version": __version__,
            "git_commit": commit.stdout.strip() or "uncommitted",
            "python": sys.version,
            "platform": platform.platform(),
            "device": "cpu",
            "numpy": np.__version__,
            "input_sha256": file_hash(source),
            "dependency_lock_sha256": file_hash(lock) if lock.exists() else None,
            "seed": project.get("synthetic", {}).get("seed", 0),
            "configuration": project,
        }

    def doctor(self) -> dict:
        root = installation_root()
        imports = {}
        for name in ["numpy", "scipy", "cv2", "PySide6", "yaml", "PIL", "onnxruntime"]:
            try:
                module = importlib.import_module(name)
                imports[name] = {
                    "available": True,
                    "version": getattr(module, "__version__", "unknown"),
                }
            except ImportError as exc:
                imports[name] = {"available": False, "error": str(exc)}
        torch = {
            "installed": False,
            "reason": "CPU numerical core and ONNX inference do not require PyTorch",
        }
        if importlib.util.find_spec("torch"):
            import torch as torch_module

            torch = {
                "installed": True,
                "version": torch_module.__version__,
                "cuda_visible": torch_module.cuda.is_available(),
                "mps_visible": hasattr(torch_module.backends, "mps")
                and torch_module.backends.mps.is_available(),
            }
        gpu = {
            "nvidia_smi": shutil.which("nvidia-smi"),
            "rocm_smi": shutil.which("rocm-smi"),
            "onnx_providers": importlib.import_module("onnxruntime").get_available_providers()
            if imports["onnxruntime"]["available"]
            else [],
        }
        binaries = {name: shutil.which(name) for name in ["ffmpeg", "ffprobe", "blender"]}
        models = [
            str(p)
            for p in (root / "models").rglob("*")
            if p.is_file() and p.suffix in {".npz", ".pkl", ".onnx"}
        ]
        model_status = {}
        for category, directories, suffixes in [
            ("body", ["body"], {".npz", ".pkl"}),
            ("pose", ["pose"], {".onnx"}),
            ("detector", ["detection", "detectors"], {".onnx"}),
            ("segmentation", ["segmentation"], {".onnx"}),
        ]:
            files = sorted(
                p
                for p in models
                if Path(p).suffix in suffixes
                and any(Path(p).is_relative_to(root / "models" / d) for d in directories)
            )
            model_status[category] = {
                "files_present": bool(files),
                "files": files,
                "validation": "file_presence_only",
            }
        model_warnings = []
        if not model_status["body"]["files_present"]:
            model_warnings.append(
                "No SMPL-family asset in models/body; licensed body fitting remains "
                "unavailable. The explicit fixture demo is available."
            )
        if not model_status["pose"]["files_present"]:
            model_warnings.append("No whole-body pose ONNX checkpoint in models/pose")
        if not model_status["detector"]["files_present"]:
            model_warnings.append(
                "No person detector ONNX checkpoint in models/detection; HOG baseline available"
            )
        if not model_status["segmentation"]["files_present"]:
            model_warnings.append(
                "No segmentation ONNX checkpoint in models/segmentation; GrabCut baseline available"
            )
        repo = Path(__file__).parents[2]
        healthy = (repo / "pyproject.toml").is_file() and (repo / "uv.lock").is_file()
        isolated = Path(sys.prefix).resolve().is_relative_to(root.resolve())
        return {
            "healthy": all(v["available"] for v in imports.values()) and healthy and isolated,
            "python": sys.version.split()[0],
            "supported_python": (3, 11) <= sys.version_info[:2] < (3, 14),
            "environment": sys.prefix,
            "installation_root": str(root),
            "isolated_environment": isolated,
            "write_permission": os.access(root, os.W_OK),
            "imports": imports,
            "binaries": binaries,
            "gpu": gpu,
            "torch": torch,
            "models": models,
            "model_status": model_status,
            "repository_health": healthy,
            "caches": {
                k: v
                for k, v in os.environ.items()
                if k
                in {
                    "PIP_CACHE_DIR",
                    "UV_CACHE_DIR",
                    "HF_HOME",
                    "TORCH_HOME",
                    "XDG_CACHE_HOME",
                    "TMPDIR",
                }
            },
            "warnings": model_warnings,
        }
