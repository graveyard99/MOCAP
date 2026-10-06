"""Model readiness must not confuse inference weights with licensed body assets."""

from openmocap.application import ProjectService


def test_inference_asset_does_not_clear_missing_body_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENMOCAP_INSTALL_ROOT", str(tmp_path))
    pose = tmp_path / "models/pose/wholebody.onnx"
    pose.parent.mkdir(parents=True)
    pose.write_bytes(b"presence is not inference validation")
    report = ProjectService().doctor()
    assert report["model_status"]["pose"]["files_present"]
    assert report["model_status"]["pose"]["validation"] == "file_presence_only"
    assert not report["model_status"]["body"]["files_present"]
    assert any("No SMPL-family asset" in warning for warning in report["warnings"])
    assert not any("No whole-body pose" in warning for warning in report["warnings"])


def test_body_file_presence_is_not_claimed_as_validated(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENMOCAP_INSTALL_ROOT", str(tmp_path))
    body = tmp_path / "models/body/SMPLX_NEUTRAL.npz"
    body.parent.mkdir(parents=True)
    body.write_bytes(b"invalid model must not be loaded by inventory")
    report = ProjectService().doctor()
    assert report["model_status"]["body"]["files"] == [str(body)]
    assert report["model_status"]["body"]["validation"] == "file_presence_only"
    assert not any("No SMPL-family asset" in warning for warning in report["warnings"])
    assert any("No whole-body pose" in warning for warning in report["warnings"])
