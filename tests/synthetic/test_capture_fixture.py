import json

import numpy as np
import yaml

from openmocap.synthetic import generate_capture
from openmocap.types import Camera


def test_capture_deterministic_async_metric(tmp_path):
    first = generate_capture(
        tmp_path / "first", camera_count=3, frames=9, render_images=False, seed=9
    )
    second = generate_capture(
        tmp_path / "second", camera_count=3, frames=9, render_images=False, seed=9
    )
    observations = json.loads((first.parent / "observations.json").read_text())
    assert observations == json.loads((second.parent / "observations.json").read_text())
    config = yaml.safe_load(first.read_text())
    assert len(config["cameras"]) == 3
    camera_times = {}
    for observation in observations:
        camera_times.setdefault(observation["camera_id"], observation["camera_timestamp"])
    assert len(set(camera_times.values())) == 3
    truth = np.load(first.parent / "ground_truth.npz")
    assert truth["joints"].shape == (9, 24, 3)
    assert np.isclose(truth["root_translation"][-1, 0] - truth["root_translation"][0, 0], 2.43)


def test_moving_capture_and_generated_images(tmp_path):
    project = generate_capture(
        tmp_path,
        camera_count=2,
        frames=4,
        moving=True,
        missing_probability=0,
        outlier_probability=0,
        noise_px=0,
    )
    config = yaml.safe_load(project.read_text())
    data = config["cameras"][0].copy()
    data.pop("fps")
    data.pop("frames")
    camera = Camera.from_dict(data)
    assert camera.trajectory is not None
    assert not np.array_equal(camera.pose_at(0).centre, camera.pose_at(0.1).centre)
    assert len(list((tmp_path / "media").rglob("*.png"))) == 7
    observations = json.loads((tmp_path / "observations.json").read_text())
    first = observations[0]
    truth = np.load(tmp_path / "ground_truth.npz")
    assert np.allclose(first["xy"], camera.project(truth["joints"][0, 0], time=0))


def test_rendered_sequence_ingest_preserves_native_camera_clock(tmp_path):
    import yaml
    from openmocap.io.media import ingest_media
    from openmocap.synthetic import generate_capture

    config_path = generate_capture(tmp_path / "capture", camera_count=3, frames=8)
    config = yaml.safe_load(config_path.read_text())
    for camera in config["cameras"]:
        index = ingest_media(camera["source_path"], nominal_fps=30)
        expected = [frame["camera_timestamp"] for frame in camera["frames"]]
        assert np.allclose(index.timestamps, expected)
        assert index.provenance["timestamp_source"] != "nominal_fps_assumption"
