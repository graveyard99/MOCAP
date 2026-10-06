"""Actual media decode/PTS tests, independent of neural-model weights."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from openmocap.io.media import FrameReader, MediaIndex, ingest_media


def _images(directory: Path, count: int = 6) -> list[np.ndarray]:
    directory.mkdir()
    frames = []
    for number in range(count):
        frame = np.full((48, 64, 3), (number * 20, 10, 200), dtype=np.uint8)
        assert cv2.imwrite(str(directory / f"frame_{number:03d}.png"), frame)
        frames.append(frame)
    return frames


def _ffmpeg() -> str:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe required for actual video tests")
    return str(shutil.which("ffmpeg"))


def test_sequence_natural_sort_explicit_timestamps_and_bgr(tmp_path: Path) -> None:
    directory = tmp_path / "images"
    directory.mkdir()
    for name, color in [("frame_10.png", 10), ("frame_2.png", 2), ("frame_1.png", 1)]:
        assert cv2.imwrite(str(directory / name), np.full((12, 16, 3), color, np.uint8))
    timestamps = {"frame_1.png": 2.5, "frame_2.png": 2.55, "frame_10.png": 2.67}
    (directory / "timestamps.json").write_text(json.dumps(timestamps))
    index = ingest_media(directory, nominal_fps=20)
    assert index.width == 16 and index.height == 12
    assert [Path(path).name for path in index.source_paths] == [
        "frame_1.png",
        "frame_2.png",
        "frame_10.png",
    ]
    np.testing.assert_array_equal(index.timestamps, [2.5, 2.55, 2.67])
    assert index.frames[0].source == "explicit_sidecar"
    assert index.diagnostics["dropped_after_indices"] == (1,)
    with FrameReader(index) as reader:
        frames = list(reader.iter_frames())
        assert [int(frame[0, 0, 0]) for _, frame in frames] == [1, 2, 10]
        assert [timestamp.presentation_seconds for timestamp, _ in frames] == list(
            timestamps.values()
        )


def test_sequence_requires_explicit_timing_and_records_assumption(tmp_path: Path) -> None:
    directory = tmp_path / "images"
    _images(directory)
    with pytest.raises(ValueError, match="explicit nominal_fps"):
        ingest_media(directory)
    index = ingest_media(directory, nominal_fps=24)
    assert index.provenance["timestamp_source"] == "nominal_fps_assumption"
    assert index.diagnostics["warnings"]
    np.testing.assert_allclose(index.timestamps, np.arange(6) / 24)


def test_sequence_duplicates_are_preserved_and_discontinuities_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "images"
    _images(directory, 4)
    (directory / "timestamps.json").write_text(json.dumps([0, 0.04, 0.04, 0.12]))
    index = ingest_media(directory, nominal_fps=25)
    assert index.diagnostics["duplicate_indices"] == (2,)
    assert index.frames[2].duplicate
    assert index.diagnostics["dropped_after_indices"] == (2,)
    (directory / "timestamps.json").write_text(json.dumps([0, 0.04, -0.02, 0.12]))
    with pytest.raises(ValueError, match="discontinuity"):
        ingest_media(directory, nominal_fps=25)


def test_sequence_csv_and_serialization(tmp_path: Path) -> None:
    directory = tmp_path / "images"
    _images(directory, 2)
    (directory / "timestamps.csv").write_text(
        "filename,timestamp\nframe_001.png,1.25\nframe_000.png,1.0\n"
    )
    index = ingest_media(directory)
    np.testing.assert_array_equal(index.timestamps, [1.0, 1.25])
    restored = MediaIndex.from_dict(json.loads(json.dumps(index.to_dict())))
    assert restored.to_dict() == json.loads(json.dumps(index.to_dict()))
    assert restored.frame_count == 2


def test_lazy_reader_has_bounded_non_destructive_memory_and_disk_cache(tmp_path: Path) -> None:
    directory = tmp_path / "images"
    frames = _images(directory)
    index = ingest_media(directory, nominal_fps=30)
    cache = tmp_path / "cache"
    with FrameReader(index, cache, max_memory_frames=2) as reader:
        for number, (_, decoded) in enumerate(reader.iter_frames()):
            np.testing.assert_array_equal(decoded, frames[number])
        assert len(reader._memory) == 2
        assert len(list(cache.rglob("*.npy"))) == 2
        edited = reader.get_frame(5)
        edited[:] = 0
        np.testing.assert_array_equal(reader.get_frame(5), frames[5])
        with pytest.raises(IndexError):
            reader.get_frame(6)
        with pytest.raises(IndexError):
            reader.get_frame(-1)
    with FrameReader(index, cache, max_memory_frames=2) as reader:
        np.testing.assert_array_equal(reader.get_frame(5), frames[5])


def test_actual_video_pts_and_lossless_decode(tmp_path: Path) -> None:
    directory = tmp_path / "images"
    expected = _images(directory, 8)
    video = tmp_path / "capture.mkv"
    subprocess.run(
        [
            _ffmpeg(),
            "-v",
            "error",
            "-framerate",
            "10",
            "-i",
            str(directory / "frame_%03d.png"),
            "-c:v",
            "ffv1",
            str(video),
        ],
        check=True,
    )
    index = ingest_media(video, nominal_fps=60)
    # Supplied nominal rate must not overwrite actual video timestamps or rate.
    assert index.kind == "video"
    assert index.nominal_fps == 10
    assert index.frame_count == 8
    np.testing.assert_allclose(index.timestamps, np.arange(8) / 10, atol=1e-6)
    assert index.provenance["timestamp_source"] == "ffprobe_presentation_timestamp"
    with FrameReader(index, max_memory_frames=2) as reader:
        for number in [7, 1, 4, 0]:
            np.testing.assert_array_equal(reader.get_frame(number), expected[number])
        decoded = list(reader.iter_frames())
        for number, (timestamp, frame) in enumerate(decoded):
            assert timestamp.frame_index == number
            np.testing.assert_array_equal(frame, expected[number])


def test_actual_variable_frame_rate_gap_uses_source_pts(tmp_path: Path) -> None:
    video = tmp_path / "vfr.mkv"
    subprocess.run(
        [
            _ffmpeg(),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=25",
            "-frames:v",
            "10",
            "-vf",
            "setpts=if(gte(N\\,5)\\,PTS+5\\,PTS)",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "ffv1",
            str(video),
        ],
        check=True,
    )
    index = ingest_media(video)
    assert index.frame_count == 10
    assert np.diff(index.timestamps)[4] > 0.2
    assert index.diagnostics["dropped_after_indices"] == (4,)
    with FrameReader(index) as reader:
        decoded = list(reader)
    np.testing.assert_array_equal(
        [timestamp.presentation_seconds for timestamp, _ in decoded], index.timestamps
    )


def test_image_dimension_mismatch_and_invalid_media_fail(tmp_path: Path) -> None:
    directory = tmp_path / "images"
    _images(directory, 2)
    cv2.imwrite(str(directory / "frame_001.png"), np.zeros((24, 32, 3), np.uint8))
    with pytest.raises(ValueError, match="dimensions change"):
        ingest_media(directory, nominal_fps=30)
    with pytest.raises(ValueError, match="positive"):
        ingest_media(directory, nominal_fps=-1)
    with pytest.raises(FileNotFoundError):
        ingest_media(tmp_path / "missing.mp4")


def test_stale_source_index_cannot_reuse_cached_images(tmp_path: Path) -> None:
    directory = tmp_path / "images"
    _images(directory, 2)
    index = ingest_media(directory, nominal_fps=30)
    with FrameReader(index, tmp_path / "cache") as reader:
        reader.get_frame(0)
    assert cv2.imwrite(str(directory / "frame_000.png"), np.zeros((48, 64, 3), np.uint8))
    with pytest.raises(ValueError, match="re-ingest"):
        FrameReader(index, tmp_path / "cache")
