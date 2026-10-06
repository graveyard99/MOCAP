"""Official deployment preprocessing and real ORT execution of generated score fixtures.

No downloaded weights, real person footage or segmentation-quality claim is
needed to test the verified publisher preprocessing/output interpretation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from openmocap.segmentation import ONNXSegmenter
from openmocap.types import PersonDetection


def _varint(value: int) -> bytes:
    output = bytearray()
    while value > 127:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def _field(number: int, value: bytes | str | int) -> bytes:
    if isinstance(value, int):
        return _varint(number << 3) + _varint(value)
    raw = value.encode() if isinstance(value, str) else value
    return _varint((number << 3) | 2) + _varint(len(raw)) + raw


def _value_info(name: str, shape: list[int]) -> bytes:
    dimensions = b"".join(_field(1, _field(1, size)) for size in shape)
    tensor_type = _field(1, 1) + _field(2, dimensions)
    return _field(1, name) + _field(2, _field(1, tensor_type))


def score_fixture(path: Path, scores: np.ndarray) -> Path:
    """Original Constant graph fixture with actual ONNX float NCHW contracts."""
    scores = np.asarray(scores, np.float32)
    tensor = b"".join(_field(1, size) for size in scores.shape)
    tensor += _field(2, 1) + _field(8, "class_scores") + _field(9, scores.tobytes())
    attribute = _field(1, "value") + _field(5, tensor) + _field(20, 4)
    node = _field(2, "result") + _field(4, "Constant") + _field(5, attribute)
    graph = _field(1, node) + _field(2, "original_segmentation_score_fixture")
    graph += _field(11, _value_info("image", [1, 3, 192, 192]))
    graph += _field(12, _value_info("result", list(scores.shape)))
    model = _field(1, 8) + _field(2, "openmocap_original_fixture") + _field(7, graph)
    model += _field(8, _field(2, 13))
    path.write_bytes(model)
    return path


def segmenter(tmp_path: Path, scores: np.ndarray, **options) -> ONNXSegmenter:
    return ONNXSegmenter(
        score_fixture(tmp_path / "scores.onnx", scores),
        input_size=(192, 192),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        logits=None,
        output_format="pphumanseg",
        **options,
    )


def detection(bbox, confidence=1.0):
    return PersonDetection("CAM_01", 0.0, "actor", np.array(bbox, float), confidence)


def test_pphumanseg_official_color_order_normalization_and_float32(tmp_path):
    backend = segmenter(tmp_path, np.zeros((1, 2, 2, 2)))
    image = np.zeros((3, 4, 3), np.uint8)
    image[..., 2] = 255  # Red in a decoded OpenCV BGR image.
    tensor = backend.preprocess(image)
    assert tensor.dtype == np.float32 and tensor.shape == (1, 3, 192, 192)
    assert tensor.flags.c_contiguous
    np.testing.assert_array_equal(tensor[:, 0], 1)
    np.testing.assert_array_equal(tensor[:, 1:], -1)
    np.testing.assert_array_equal(image[..., 2], 255)  # Original plate is unchanged.
    np.testing.assert_array_equal(backend.preprocess(np.full_like(image, 255)), 1)
    np.testing.assert_array_equal(backend.preprocess(np.zeros_like(image)), -1)


def test_pphumanseg_resizes_scores_before_argmax_and_clips_to_instance_roi(tmp_path):
    scores = np.array([[[[1, 0], [1, 0]], [[0, 1], [0, 1]]]], np.float32)
    backend = segmenter(tmp_path, scores)
    image = np.zeros((6, 10, 3), np.uint8)
    mask, confidence = backend.infer(image, detection((0, 0, 8, 6), 0.9))
    assert mask.dtype == np.uint8 and mask.shape == image.shape[:2]
    assert not mask[:, :5].any() and mask[:, 5:8].all() and not mask[:, 8:].any()
    assert 0.5 < confidence <= 0.9
    assert backend.metadata["confidence_interpretation"] == "normalized_class_probabilities"
    assert backend.metadata["providers"] == ["CPUExecutionProvider"]
    assert len(backend.metadata["sha256"]) == 64


def test_pphumanseg_argmax_ties_are_background_and_logits_are_explicit(tmp_path):
    backend = segmenter(tmp_path, np.ones((1, 2, 2, 2)))
    image = np.zeros((4, 4, 3), np.uint8)
    mask, confidence = backend.infer(image, detection((0, 0, 4, 4)))
    assert not mask.any() and confidence == 0
    assert backend.metadata["confidence_interpretation"] == "softmax_normalized_class_scores"
    scores = np.full((1, 2, 2, 2), -2, np.float32)
    scores[:, 1] = 2
    backend = segmenter(tmp_path, scores)
    mask, confidence = backend.infer(image, detection((0, 0, 4, 4)))
    assert mask.all() and confidence == pytest.approx(1 / (1 + np.exp(-4)), abs=1e-6)


def test_pphumanseg_outside_detection_cannot_select_wrapped_negative_roi(tmp_path):
    scores = np.zeros((1, 2, 2, 2), np.float32)
    scores[:, 1] = 1
    backend = segmenter(tmp_path, scores)
    image = np.zeros((5, 5, 3), np.uint8)
    mask, confidence = backend.infer(image, detection((-8, -8, -1, -1)))
    assert not mask.any() and confidence == 0


def test_pphumanseg_rejects_malformed_scores_and_input(tmp_path):
    backend = segmenter(tmp_path, np.zeros((1, 3, 2, 2)))
    with pytest.raises(ValueError, match="class scores"):
        backend.infer(np.zeros((4, 4, 3), np.uint8), detection((0, 0, 4, 4)))
    with pytest.raises(ValueError, match="uint8"):
        backend.preprocess(np.zeros((4, 4, 3), np.float32))
    with pytest.raises(ValueError, match="three channels"):
        backend.preprocess(np.zeros((4, 4), np.uint8))
    model = score_fixture(tmp_path / "invalid_config.onnx", np.zeros((1, 2, 2, 2)))
    with pytest.raises(ValueError, match="mean/std"):
        ONNXSegmenter(model, output_format="pphumanseg")


def test_pphumanseg_config_selects_exact_verified_publisher_contract():
    config_path = Path(__file__).parents[2] / "configs/segmentation/pphumanseg.yaml"
    config = yaml.safe_load(config_path.read_text())["perception"]
    assert config["segmentation_model"].endswith("human_segmentation_pphumanseg_2023mar.onnx")
    assert config["segmentation"]["input_size"] == [192, 192]
    assert config["segmentation"]["mean"] == [0.5, 0.5, 0.5]
    assert config["segmentation"]["std"] == [0.5, 0.5, 0.5]
    assert config["segmentation"]["output_format"] == "pphumanseg"
    assert config["segmentation"]["logits"] is None
