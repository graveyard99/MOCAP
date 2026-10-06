"""Perception math, immutable observations and actual local ONNX execution."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from openmocap.association import associate_people, epipolar_error
from openmocap.cameras.models import look_at
from openmocap.detection import HOGDetector, nms, projected_actor_roi
from openmocap.io.observations import (
    apply_observation_overrides,
    load_observations,
    save_observations,
)
from openmocap.perception import run_project_perception
from openmocap.pose2d import ONNXPoseBackend, decode_heatmaps, decode_simcc
from openmocap.pose2d.derived import derive_midpoint_landmarks
from openmocap.segmentation import GrabCutSegmenter, ONNXSegmenter
from openmocap.tracking import IoUTracker
from openmocap.types import CameraIntrinsics, PersonDetection, Pose2DObservation


def _varint(number: int) -> bytes:
    encoded = bytearray()
    while number > 127:
        encoded.append((number & 127) | 128)
        number >>= 7
    encoded.append(number)
    return bytes(encoded)


def _field(number: int, value: bytes | str | int) -> bytes:
    if isinstance(value, int):
        return _varint(number << 3) + _varint(value)
    data = value.encode() if isinstance(value, str) else value
    return _varint((number << 3) | 2) + _varint(len(data)) + data


def _value_info(name: str, shape: list[int]) -> bytes:
    dims = b"".join(_field(1, _field(1, d)) for d in shape)
    tensor_type = _field(1, 1) + _field(2, dims)
    return _field(1, name) + _field(2, _field(1, tensor_type))


def identity_onnx(path: Path) -> Path:
    """Generated ONNX Identity graph (IR8/opset13), no third-party weights.

    Protobuf's small wire format avoids adding a build-time onnx dependency.
    Real ONNX Runtime validates and executes this model in tests.
    """
    node = _field(1, "image") + _field(2, "result") + _field(4, "Identity")
    graph = _field(1, node) + _field(2, "generated_identity_fixture")
    graph += _field(11, _value_info("image", [1, 3, 8, 8]))
    graph += _field(12, _value_info("result", [1, 3, 8, 8]))
    model = _field(1, 8) + _field(2, "openmocap_test_fixture") + _field(7, graph)
    model += _field(8, _field(2, 13))
    path.write_bytes(model)
    return path


def detection(box=(2, 2, 10, 10), confidence=1.0, person=""):
    return PersonDetection("CAM_A", 0.0, person, np.array(box, float), confidence)


def cameras():
    intrinsics = CameraIntrinsics(800, 800, 320, 240, 640, 480)
    return [
        look_at("CAM_A", [3, 2, 3], [0, 1, 0], intrinsics),
        look_at("CAM_B", [-3, 2, 3], [0, 1, 0], intrinsics),
    ]


def test_simcc_and_subpixel_heatmaps():
    x, y = np.zeros((2, 3, 20)), np.zeros((2, 3, 30))
    x[..., 8], y[..., 12] = 0.9, 0.7
    positions, confidence = decode_simcc(x, y)
    np.testing.assert_allclose(positions, np.broadcast_to([4, 6], positions.shape))
    np.testing.assert_allclose(confidence, 0.7)
    heatmaps = np.zeros((1, 1, 8, 8))
    heatmaps[0, 0, 3, 4] = 1
    positions, confidence = decode_heatmaps(heatmaps, (80, 160))
    np.testing.assert_allclose(positions, [[[40, 60]]])


def test_actual_onnx_pose_and_segmentation(tmp_path):
    model = identity_onnx(tmp_path / "fixture.onnx")
    backend = ONNXPoseBackend(
        model,
        ["a", "b", "c"],
        output_format="heatmap",
        mean=(0, 0, 0),
        std=(255, 255, 255),
        bbox_padding=1.0,
        rgb=False,
    )
    image = np.zeros((8, 8, 3), np.uint8)
    image[2, 4, 0], image[3, 5, 1], image[6, 1, 2] = 255, 255, 255
    rows = backend.infer(image, [detection((0, 0, 8, 8), person="actor")])
    np.testing.assert_allclose([r.xy for r in rows], [[4, 2], [5, 3], [1, 6]])
    assert all(r.confidence == 1 and len(r.metadata["model_sha256"]) == 64 for r in rows)
    segmenter = ONNXSegmenter(
        model, foreground_class=0, logits=False, mean=(0, 0, 0), std=(1, 1, 1)
    )
    mask, confidence = segmenter.infer(image, detection((0, 0, 8, 8)))
    assert mask.shape == (8, 8) and mask[6, 1] == 1 and confidence == 1.0


def test_pose_joint_schema_cannot_be_guessed(tmp_path):
    backend = ONNXPoseBackend(
        identity_onnx(tmp_path / "fixture.onnx"), ["wrong"], output_format="heatmap"
    )
    with pytest.raises(ValueError, match="joint_names"):
        backend.infer(np.zeros((8, 8, 3), np.uint8), [detection((0, 0, 8, 8))])


def test_tracking_does_not_mutate_detections_and_expires():
    tracker = IoUTracker("CAM_A", max_age_seconds=0.5)
    raw = detection()
    first = tracker.update([raw], 0)[0]
    assert raw.person_id == ""
    second = tracker.update([detection((3, 2, 11, 10))], 0.1)[0]
    assert first.person_id == second.person_id
    third = tracker.update([detection()], 1)[0]
    assert third.person_id != first.person_id
    with pytest.raises(ValueError, match="monotonic"):
        tracker.update([], 0.2)


def test_nms_and_hog_blank():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 25, 25]])
    assert nms(boxes, np.array([0.9, 0.8, 0.7])) == [0, 2]
    assert HOGDetector().detect(np.zeros((200, 200, 3), np.uint8), "CAM_A", 0) == []


def test_geometry_association_prevents_cross_person():
    camera_a, camera_b = cameras()
    poses = [
        np.array([[0, 1, 0], [0.2, 1.4, 0], [0.1, 0.5, 0]]),
        np.array([[0.9, 1.2, 0.4], [1.1, 1.6, 0.4], [1.0, 0.7, 0.4]]),
    ]
    rows_a, rows_b = [], []
    for index, points in enumerate(poses):
        for joint, (a, b) in enumerate(
            zip(camera_a.project(points), camera_b.project(points), strict=True)
        ):
            rows_a.append(Pose2DObservation("CAM_A", 0, f"a{index}", str(joint), a))
            rows_b.append(Pose2DObservation("CAM_B", 0, f"b{1 - index}", str(joint), b))
    assert (
        np.max(
            epipolar_error(
                camera_a, camera_b, camera_a.project(poses[0]), camera_b.project(poses[0])
            )
        )
        < 1e-8
    )
    matches = associate_people(camera_a, camera_b, rows_a, rows_b)
    assert {(m.anchor_id, m.target_id) for m in matches} == {("a0", "b1"), ("a1", "b0")}
    assert not any(m.ambiguous for m in matches)
    assert projected_actor_roi(camera_a, poses[0], 0).shape == (4,)


def test_raw_serialization_overrides_and_derived_landmarks(tmp_path):
    rows = [
        Pose2DObservation("CAM_A", 0, "actor", "left_hip", [2, 3], 0.9),
        Pose2DObservation("CAM_A", 0, "actor", "right_hip", [4, 3], 0.8),
    ]
    derived = derive_midpoint_landmarks(rows)
    assert len(rows) == 2 and derived[-1].joint_id == "pelvis"
    np.testing.assert_allclose(derived[-1].xy, [3, 3])
    assert derived[-1].confidence < 0.8
    save_observations(tmp_path / "raw.json", rows)
    assert len(load_observations(tmp_path / "raw.json", camera_ids={"CAM_A"})) == 2
    records = [o.to_dict() for o in rows]
    effective = apply_observation_overrides(
        records,
        [
            {
                "id": "audit1",
                "selector": {"joint_id": "left_hip"},
                "changes": {"xy": [10, 20], "disabled": True},
            }
        ],
    )
    assert records[0]["xy"] == [2, 3] and effective[0]["xy"] == [10, 20]
    assert effective[0]["applied_override_ids"] == ["audit1"]


def test_grabcut_fallback_is_explicit_and_low_confidence():
    image = np.full((64, 64, 3), 255, np.uint8)
    image[20:45, 25:40] = 0
    backend = GrabCutSegmenter()
    mask, confidence = backend.infer(image, detection((15, 10, 50, 55)))
    assert mask.any() and 0 < confidence <= 0.35
    assert not backend.metadata["learned"]


def test_project_perception_real_rgb_and_checkpoint(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    image = np.zeros((8, 8, 3), np.uint8)
    image[2, 4, 0], image[3, 5, 1], image[6, 1, 2] = 255, 255, 255
    for frame in range(4):
        cv2.imwrite(str(media / f"{frame:04d}.png"), image)
    model = identity_onnx(tmp_path / "fixture.onnx")
    project = {
        "path": str(tmp_path),
        "cameras": [{"id": "CAM_A", "source_path": str(media), "fps": 30}],
        "perception": {
            "pose_model": str(model),
            "joint_names": ["a", "b", "c"],
            "person_rois": {"CAM_A": [0, 0, 8, 8]},
            "pose": {
                "output_format": "heatmap",
                "mean": [0, 0, 0],
                "std": [255, 255, 255],
                "rgb": False,
                "bbox_padding": 1,
            },
        },
    }
    progress = []
    result = run_project_perception(project, "pose2d", progress.append)
    assert result["status"] == "WARNING"  # uncalibrated identities remain local
    assert result["observation_count"] == 12 and len(progress) == 4
    payload = json.loads((tmp_path / "raw/observations.json").read_text())
    assert payload["observations"][0]["source"] == "onnx_pose"
    assert run_project_perception(project, "pose2d")["cached"]
