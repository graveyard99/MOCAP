"""Publisher model contracts without network/model requirements in CI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import yaml

from openmocap.detection import ONNXDetector
from openmocap.detection.backends import decode_yolox_grid
from openmocap.perception import _resolve
from openmocap.pose2d.schemas import COCO_WHOLEBODY_133


def test_yolox_raw_grid_is_explicit_and_preserves_predictions():
    prediction = np.zeros((5, 6))
    prediction[:, 4:] = 0.8
    decoded = decode_yolox_grid(prediction, (16, 16))
    np.testing.assert_allclose(decoded[:, :2], [[0, 0], [8, 0], [0, 8], [8, 8], [0, 0]])
    np.testing.assert_allclose(decoded[:, 2:4], [[8, 8], [8, 8], [8, 8], [8, 8], [16, 16]])
    assert np.all(prediction[:, :4] == 0)
    with pytest.raises(ValueError, match="stride grid"):
        decode_yolox_grid(prediction[:-1], (16, 16))


def test_mmdeploy_person_labels_and_top_left_bgr_preprocessing(monkeypatch):
    from openmocap.detection import backends

    captured = []

    class FakeEngine:
        metadata = {"sha256": "fixture"}

        def __init__(self, *args, **kwargs):
            pass

        def image_size(self, fallback):
            return (16, 16)

        def run(self, tensor):
            captured.append(tensor)
            return [
                np.array([[[2.0, 4.0, 10.0, 12.0, 0.9], [0.0, 0.0, 2.0, 2.0, 0.8]]]),
                np.array([[0, 1]]),
            ]

    monkeypatch.setattr(backends, "ONNXEngine", FakeEngine)
    detector = ONNXDetector("fixture.onnx", output_format="mmdeploy")
    image = np.full((8, 16, 3), [10, 20, 30], dtype=np.uint8)
    detections = detector.detect(image, "CAM01", 2.13)
    assert len(detections) == 1 and detections[0].camera_timestamp == 2.13
    np.testing.assert_allclose(detections[0].bbox, [2, 4, 10, 8])
    np.testing.assert_allclose(captured[0][0, :, 0, 0], [10, 20, 30])
    np.testing.assert_allclose(captured[0][0, :, -1, 0], [114, 114, 114])


def test_preset_full133_and_installation_placeholders(tmp_path, monkeypatch):
    repo = Path(__file__).parents[2]
    preset = yaml.safe_load((repo / "configs/pose/rtmpose-wholebody-yolox.yaml").read_text())
    config = preset["perception"]
    assert config["joint_names"] == list(COCO_WHOLEBODY_133)
    assert len(config["joint_names"]) == 133 and config["pose"]["rgb"] is False
    assert config["pose"]["split_ratio"] == 2
    monkeypatch.setenv("OPENMOCAP_INSTALL_ROOT", str(tmp_path))
    assert _resolve({"path": "irrelevant"}, config["pose_model"]).is_relative_to(tmp_path)


def test_acquisition_extracts_only_weights_and_metadata_without_scripts(tmp_path):
    repo = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location("model_fetch", repo / "scripts/fetch_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / "downloads" / Path(module.ASSETS["pose"]["url"]).name
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("export/end2end.onnx", b"generated fixture data, not executed")
        bundle.writestr("export/deploy.json", json.dumps({"fixture": True}))
        bundle.writestr("../outside.sh", "malicious content must never be extracted")
        bundle.writestr("export/install.sh", "malicious content must never be executed")
    record = module.fetch_asset("pose", tmp_path)
    assert record["publisher_checksum"] is None
    assert len(record["sha256"]) == 64
    assert (
        not (tmp_path / "outside.sh").exists()
        and not (tmp_path / "models/pose/install.sh").exists()
    )
    assert module.fetch_asset("pose", tmp_path)["sha256"] == record["sha256"]
    Path(record["file"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="mismatch"):
        module.fetch_asset("pose", tmp_path)


def test_requested_gpu_cannot_silently_fall_back_to_cpu(tmp_path, monkeypatch):
    import onnxruntime as ort
    from types import SimpleNamespace
    from openmocap.detection import ONNXEngine

    model = tmp_path / "mock.onnx"
    model.write_bytes(b"mock runtime provider initialization test")
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: session)
    with pytest.raises(RuntimeError, match="failed to initialize.*CUDAExecutionProvider"):
        ONNXEngine(model, providers=["CUDAExecutionProvider"])


def test_direct_asset_published_hash_is_verified_before_install(tmp_path, monkeypatch):
    repo = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location(
        "model_fetch_direct", repo / "scripts/fetch_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    asset = module.ASSETS["segmentation"]
    incoming = tmp_path / "downloads" / Path(asset["url"]).name
    incoming.parent.mkdir()
    incoming.write_bytes(b"generated test fixture")
    with pytest.raises(ValueError, match="published Git LFS"):
        module.fetch_asset("segmentation", tmp_path)
    assert not (tmp_path / asset["destination"]).exists()
    monkeypatch.setitem(asset, "expected_sha256", module.digest(incoming))
    monkeypatch.setitem(asset, "expected_bytes", incoming.stat().st_size)
    record = module.fetch_asset("segmentation", tmp_path)
    assert record["publisher_checksum"] == record["sha256"]
    assert record["inference_verification"]["executed"] is False
