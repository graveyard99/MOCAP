"""Shared CLI/UI perception orchestration, local models and resumable artifacts."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from openmocap import SCHEMA_VERSION, __version__
from openmocap.association import associate_records
from openmocap.detection import HOGDetector, ONNXDetector
from openmocap.io.media import FrameReader, ingest_media
from openmocap.io.observations import load_observations, save_observations
from openmocap.pipeline.cache import CheckpointStore, atomic_json, fingerprint
from openmocap.pose2d import ONNXPoseBackend
from openmocap.pose2d.derived import derive_midpoint_landmarks
from openmocap.segmentation import GrabCutSegmenter, ONNXSegmenter
from openmocap.tracking import IoUTracker, appearance_embedding
from openmocap.types import Camera, PersonDetection, SilhouetteObservation


def _resolve(project: dict, value: str | Path) -> Path:
    root = os.environ.get("OPENMOCAP_INSTALL_ROOT", str(Path(__file__).resolve().parents[3]))
    expanded = (
        str(value)
        .replace("${INSTALL_ROOT}", root)
        .replace("${OPENMOCAP_INSTALL_ROOT}", root)
        .replace("<INSTALL_ROOT>", root)
    )
    path = Path(expanded).expanduser()
    return path.resolve() if path.is_absolute() else (Path(project["path"]) / path).resolve()


def _check_cancel(cancel_event: Any) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Perception cancelled; complete stage checkpoints retained")


def _serial_detection(detection: PersonDetection) -> dict:
    row = asdict(detection)
    row["bbox"] = np.asarray(detection.bbox).tolist()
    row["embedding"] = (
        np.asarray(detection.embedding).tolist() if detection.embedding is not None else None
    )
    return row


def _associate(project: dict, records: list[dict], raw: Path) -> tuple[list[dict], dict]:
    cameras = [Camera.from_dict(c) for c in project["cameras"] if c.get("enabled", True)]
    association = project.get("perception", {}).get("association", {})
    effective, diagnostics = associate_records(
        cameras,
        records,
        actor_id=project.get("actor", {}).get("id", "actor01"),
        max_error_px=float(association.get("max_error_px", 8)),
        max_time_delta=float(association.get("max_time_delta", 0.05)),
    )
    atomic_json(
        raw / "associations.json",
        {"schema_version": SCHEMA_VERSION, "code_version": __version__, **diagnostics},
    )
    return effective, diagnostics


def run_project_perception(
    project: dict, stage: str, callback: Callable | None = None, cancel_event: Any = None
) -> dict[str, Any]:
    """Real RGB stages, sharing typed inference APIs between UI and CLI.

    Configuration keys: perception.detector_model, detector options,
    pose_model, joint_names, pose options, segmentation_model, segmentation
    options, providers, frame_stride; optional explicit person_rois per camera.
    Checkpoints hash raw media, model assets, settings, code and schema versions.
    Returned status is WARNING for imported data or low-quality CPU fallbacks.
    The caller persists stage status through its existing project service.
    """
    if stage not in {"detect", "pose2d", "segment", "associate"}:
        raise ValueError(f"Unsupported perception stage: {stage}")
    raw = Path(project["path"]) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    observations_path = _resolve(project, project.get("observations", "raw/observations.json"))
    config = dict(project.get("perception", {}))
    providers = config.get("providers", ["CPUExecutionProvider"])
    detector_backend = config.get(
        "detector_backend", "onnx" if config.get("detector_model") else "hog"
    )
    if detector_backend not in {"hog", "onnx"}:
        raise ValueError("perception.detector_backend must be hog or onnx")
    if detector_backend == "onnx" and not config.get("detector_model"):
        raise FileNotFoundError("ONNX detector selected; supply perception.detector_model")
    _check_cancel(cancel_event)
    if observations_path.exists() and not config.get("pose_model") and stage != "segment":
        imported = load_observations(
            observations_path, camera_ids={c["id"] for c in project["cameras"]}
        )
        return {
            "status": "WARNING",
            "message": f"{len(imported)} imported timestamped keypoints available; "
            f"{stage} neural inference was not executed",
            "observations": str(observations_path),
            "observation_count": len(imported),
            "source": "imported_observations",
        }
    detection_path = raw / "detections.json"
    pose_path = raw / "pose2d.json"
    if stage == "associate":
        source = pose_path if pose_path.exists() else observations_path
        observations = load_observations(source)
        records, diagnostics = _associate(project, [o.to_dict() for o in observations], raw)
        atomic_json(
            observations_path,
            {
                "schema_version": SCHEMA_VERSION,
                "observations": records,
                "metadata": {"source": "associated_pose2d"},
            },
        )
        ambiguous = diagnostics["unassociated_observations"]
        return {
            "status": "WARNING" if ambiguous else "COMPLETE",
            "message": f"Geometry identity association complete; {ambiguous} ambiguous measurements disabled",
            "observations": str(observations_path),
            "diagnostics": diagnostics,
        }
    cameras = [c for c in project["cameras"] if c.get("enabled", True)]
    if not cameras or any(not c.get("source_path") for c in cameras):
        raise ValueError("Register RGB media for every enabled camera before inference")
    indices = {
        c["id"]: ingest_media(_resolve(project, c["source_path"]), nominal_fps=c.get("fps", 30))
        for c in cameras
    }
    for camera in cameras:
        index = indices[camera["id"]]
        intrinsics = camera.get("intrinsics")
        if intrinsics and (index.width, index.height) != (
            intrinsics["width"],
            intrinsics["height"],
        ):
            raise ValueError(
                f"Camera {camera['id']} media resolution {index.width}x{index.height} "
                "does not match calibration; recalibrate or explicitly rescale intrinsics"
            )
    inputs = []
    for index in indices.values():
        source = Path(index.source)
        inputs.extend(sorted(source.iterdir()) if source.is_dir() else [source])
    inputs = [p for p in inputs if p.is_file()]
    for name in ("detector_model", "pose_model", "segmentation_model"):
        required = (
            (name == "detector_model" and detector_backend == "onnx")
            or (name == "pose_model" and stage == "pose2d")
            or (name == "segmentation_model" and stage == "segment")
        )
        if required and config.get(name):
            inputs.append(_resolve(project, config[name]))
    cache = CheckpointStore(Path(project["path"]) / "cache" / "perception")
    key = fingerprint({"stage": stage, "perception": config, "cameras": cameras}, inputs)
    final_path = (
        detection_path
        if stage == "detect"
        else raw / "silhouettes.json"
        if stage == "segment"
        else observations_path
    )
    if cache.valid(stage, key, [final_path]):
        payload = json.loads(final_path.read_text())
        status = payload.get("metadata", {}).get("status", "COMPLETE")
        return {
            "status": status,
            "message": f"{stage} reused validated content-hashed checkpoint",
            "cached": True,
            "artifact": str(final_path),
        }
    detector_options = dict(config.get("detector", {}))
    if detector_backend == "onnx":
        detector = ONNXDetector(
            _resolve(project, config["detector_model"]), providers=providers, **detector_options
        )
    else:
        detector = HOGDetector(**detector_options)
    detector_fallback = detector_backend == "hog" and not bool(config.get("person_rois"))
    pose = None
    if stage == "pose2d":
        if not config.get("pose_model"):
            raise FileNotFoundError(
                "Whole-body pose requires a local licensed ONNX checkpoint; "
                "configure perception.pose_model and exact joint_names"
            )
        names = config.get("joint_names", project.get("joint_names", []))
        pose = ONNXPoseBackend(
            _resolve(project, config["pose_model"]),
            names,
            providers=providers,
            **config.get("pose", {}),
        )
        project["joint_names"] = list(names)
    segmenter = None
    if stage == "segment":
        segmenter = (
            ONNXSegmenter(
                _resolve(project, config["segmentation_model"]),
                providers=providers,
                **config.get("segmentation", {}),
            )
            if config.get("segmentation_model")
            else GrabCutSegmenter(**config.get("segmentation", {}))
        )
    prior_detections = []
    # Reuse only matching detector hashes; changing pose weights does not repeat detection.
    detection_key = fingerprint(
        {
            "stage": "detect",
            "perception": {
                k: v
                for k, v in config.items()
                if k
                in {
                    "detector_backend",
                    "detector_model",
                    "detector",
                    "providers",
                    "frame_stride",
                    "person_rois",
                }
            },
            "cameras": cameras,
        },
        [
            p
            for p in inputs
            if p.suffix != ".onnx"
            or p == _resolve(project, config.get("detector_model", "missing.onnx"))
        ],
    )
    if stage != "detect" and cache.valid("detect-shared", detection_key, [detection_path]):
        prior_detections = json.loads(detection_path.read_text())["detections"]
    detection_lookup: dict[tuple[str, int], list[dict]] = {}
    for row in prior_detections:
        detection_lookup.setdefault((row["camera_id"], row["frame_index"]), []).append(row)
    all_detections: list[dict] = []
    all_observations = []
    all_silhouettes = []
    stride = int(config.get("frame_stride", 1))
    if stride < 1:
        raise ValueError("perception.frame_stride must be a positive integer")
    total_frames = sum(len(i.frames) for i in indices.values())
    processed = 0
    masks_dir = raw / "masks" / key[:12]
    for camera in cameras:
        cid = camera["id"]
        tracker = IoUTracker(cid)
        reader = FrameReader(
            indices[cid], cache_dir=Path(project["path"]) / "cache" / "frames" / cid
        )
        for timestamp, image in reader.iter_frames():
            _check_cancel(cancel_event)
            processed += 1
            if timestamp.frame_index % stride:
                continue
            time = timestamp.presentation_seconds
            if timestamp.discontinuity:
                tracker = IoUTracker(cid)
            stored = detection_lookup.get((cid, timestamp.frame_index))
            if prior_detections:
                detections = [
                    PersonDetection(**{k: v for k, v in row.items() if k != "frame_index"})
                    for row in stored or []
                ]
            elif cid in config.get("person_rois", {}):
                # Manual ROI is provided evidence, visibly distinguished from detections.
                rois = config["person_rois"][cid]
                if len(rois) == 4 and isinstance(rois[0], (int, float)):
                    rois = [rois]
                detections = [
                    PersonDetection(cid, time, "", np.asarray(box), 1.0, source="manual_roi")
                    for box in rois
                ]
                detections = tracker.update(detections, time)
            else:
                detections = tracker.update(detector.detect(image, cid, time), time)
            for detection in detections:
                if detection.embedding is None:
                    detection.embedding = appearance_embedding(image, np.asarray(detection.bbox))
                all_detections.append(
                    {**_serial_detection(detection), "frame_index": timestamp.frame_index}
                )
            if pose:
                rows = pose.infer(image, detections)
                if config.get("derive_midpoints", True):
                    rows = derive_midpoint_landmarks(rows)
                for row in rows:
                    row.world_timestamp = float(
                        camera.get("time_mapping", {}).get("scale", 1) * time
                        + camera.get("time_mapping", {}).get("offset", 0)
                    )
                    row.metadata["frame_index"] = timestamp.frame_index
                all_observations.extend(rows)
            if segmenter:
                for detection in detections:
                    mask, confidence = segmenter.infer(image, detection)
                    destination = (
                        masks_dir
                        / cid
                        / f"{timestamp.frame_index:08d}_{detection.person_id.split(':')[-1]}.png"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(destination), mask * 255):
                        raise OSError(f"Cannot save segmentation mask: {destination}")
                    all_silhouettes.append(
                        SilhouetteObservation(
                            cid,
                            time,
                            detection.person_id,
                            str(destination.relative_to(Path(project["path"]))),
                            confidence,
                            source=segmenter.metadata["backend"],
                        )
                    )
            if callback and processed % max(1, total_frames // 100) == 0:
                callback(
                    {
                        "stage": stage,
                        "progress": processed / total_frames,
                        "message": f"{cid} PTS {time:.4f}s; {len(detections)} tracked people",
                    }
                )
    status = (
        "WARNING"
        if detector_fallback or (stage == "segment" and not config.get("segmentation_model"))
        else "COMPLETE"
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "code_version": __version__,
        "detector": detector.metadata,
        "raw_input_hash": key,
        "status": status,
        "person_rois": config.get("person_rois", {}),
        "timestamp_source": "actual_media_index",
    }
    atomic_json(detection_path, {"metadata": metadata, "detections": all_detections})
    cache.complete("detect-shared", detection_key, [detection_path])
    message = f"{stage}: {len(all_detections)} detections from timestamped RGB media"
    if pose:
        if not all_observations:
            raise ValueError(
                "Detector found no usable people; inspect footage/ROI or configure a better detector"
            )
        save_observations(pose_path, all_observations, {**metadata, "pose": pose.metadata})
        records = [o.to_dict() for o in all_observations]
        project["joint_names"] = list(
            dict.fromkeys([*pose.joint_names, *[o.joint_id for o in all_observations]])
        )
        if all(c.get("intrinsics") and c.get("pose") for c in cameras):
            records, diagnostics = _associate(project, records, raw)
            if diagnostics["unassociated_observations"]:
                status = "WARNING"
            metadata["association"] = diagnostics
        else:
            status = "WARNING"
            metadata["warning"] = (
                "Uncalibrated: local track identities remain unassociated; import calibration and run associate"
            )
        metadata["status"] = status
        atomic_json(
            observations_path,
            {
                "schema_version": SCHEMA_VERSION,
                "metadata": {**metadata, "pose": pose.metadata},
                "observations": records,
            },
        )
        message += f"; {len(records)} raw landmarks preserved with geometry association"
    if segmenter:
        atomic_json(
            final_path,
            {
                "schema_version": SCHEMA_VERSION,
                "metadata": {**metadata, "segmentation": segmenter.metadata},
                "silhouettes": [asdict(s) for s in all_silhouettes],
            },
        )
        message += f"; {len(all_silhouettes)} masks preserved"
    if status == "WARNING":
        message += "; review baseline-backend quality or unresolved identity ambiguity"
    cache.complete(stage, key, [final_path])
    return {
        "status": status,
        "message": message,
        "artifact": str(final_path),
        "detections": len(all_detections),
        "observation_count": len(all_observations),
        "mask_count": len(all_silhouettes),
        "metadata": metadata,
    }
