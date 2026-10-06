#!/usr/bin/env python3
"""Verify installed public inference assets on a supplied real-person image.

This is an inference contract smoke test, not a detector/pose accuracy benchmark
or a multi-camera reconstruction acceptance test. No files are downloaded here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from openmocap.detection import ONNXDetector
from openmocap.perception import _resolve
from openmocap.pose2d import ONNXPoseBackend


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--preset",
        type=Path,
        default=Path(__file__).parents[1] / "configs/pose/rtmpose-wholebody-yolox.yaml",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "temp/model-verification")
    parser.add_argument(
        "--provider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
        help="Override preset execution provider; requested GPU initialization must succeed",
    )
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Cannot decode test image: {args.image}")
    config = yaml.safe_load(args.preset.read_text())["perception"]
    if args.provider:
        config["providers"] = [args.provider]
    project = {"path": str(args.preset.parent)}
    detector_model = _resolve(project, config["detector_model"])
    pose_model = _resolve(project, config["pose_model"])
    detector = ONNXDetector(detector_model, providers=config["providers"], **config["detector"])
    pose = ONNXPoseBackend(
        pose_model, config["joint_names"], providers=config["providers"], **config["pose"]
    )
    # Warm up both graphs separately before reporting execution timing.
    warm_detections = detector.detect(image, "VERIFY", 0)
    if not warm_detections:
        raise RuntimeError("No person detected during warmup; real-person smoke test failed")
    pose.infer(image, warm_detections)
    started = time.perf_counter()
    detections = detector.detect(image, "VERIFY", 0)
    elapsed_detector = time.perf_counter() - started
    if not detections:
        raise RuntimeError(
            "No person detected in the supplied image; inference executed, but real-person smoke test failed"
        )
    for i, detection in enumerate(detections):
        detection.person_id = f"verification-person-{i}"
    started = time.perf_counter()
    observations = pose.infer(image, detections)
    elapsed_pose = time.perf_counter() - started
    if len(observations) != len(detections) * len(config["joint_names"]):
        raise RuntimeError("Whole-body model output count does not match configured joint schema")
    if not np.isfinite([o.xy for o in observations]).all():
        raise RuntimeError("Nonfinite decoded keypoints")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    canvas = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = np.rint(detection.bbox).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 210, 80), 2)
    for observation in observations:
        if observation.confidence >= 0.3:
            cv2.circle(canvas, tuple(np.rint(observation.xy).astype(int)), 2, (255, 180, 20), -1)
    cv2.imwrite(str(output / "wholebody-overlay.png"), canvas)
    metrics = {
        "executed": True,
        "kind": "real-image inference contract smoke test",
        "limitations": "No ground truth accuracy, hand/face fitting, metric or multi-camera quality claim",
        "image": str(args.image.resolve()),
        "image_dimensions": list(image.shape[:2][::-1]),
        "detections": len(detections),
        "keypoints_per_person": len(config["joint_names"]),
        "observations": len(observations),
        "detector_seconds": elapsed_detector,
        "pose_seconds": elapsed_pose,
        "requested_providers": config["providers"],
        "active_detector_providers": detector.engine.session.get_providers(),
        "active_pose_providers": pose.engine.session.get_providers(),
        "warmup_completed": True,
        "detector_model": detector.metadata,
        "pose_model": pose.metadata,
        "high_confidence_landmarks": sum(o.confidence >= 0.3 for o in observations),
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "verification.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "observations.json").write_text(
        json.dumps([o.to_dict() for o in observations], indent=2) + "\n"
    )
    for model, stage in [(detector_model, "detector"), (pose_model, "pose")]:
        manifest_path = model.parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["inference_verification"] = {
                **metrics,
                "stage": stage,
                "verification_report": str(output / "verification.json"),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
