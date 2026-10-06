"""Timestamp-aware media indexing and bounded, lazy BGR frame decoding.

Video timestamps are read from FFprobe; OpenCV frame numbers are used only for
decoding. Image sequences require a timestamp sidecar or an explicitly supplied
nominal rate. A nominal rate is never used to fabricate video timestamps.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterator

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from openmocap.sync.timing import validate_timestamps
from openmocap.types import FrameTimestamp

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class MediaIndex:
    """Small timestamp index, not decoded media. Times are seconds in source time."""

    source: str
    kind: str
    width: int
    height: int
    nominal_fps: float | None
    frames: list[FrameTimestamp]
    source_paths: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.kind not in {"video", "images"}:
            raise ValueError(f"Unsupported indexed media kind: {self.kind}")
        if self.width <= 0 or self.height <= 0 or not self.frames:
            raise ValueError("Media must contain frames with positive width and height")
        if self.nominal_fps is not None and (
            not np.isfinite(self.nominal_fps) or self.nominal_fps <= 0
        ):
            raise ValueError("Nominal FPS must be finite and positive")
        if self.kind == "images" and len(self.source_paths) != len(self.frames):
            raise ValueError("Every image frame must have a matching source path")
        if [frame.frame_index for frame in self.frames] != list(range(len(self.frames))):
            raise ValueError("Media frame indices must be consecutive in decode order")
        validate_timestamps(self.timestamps)

    @property
    def timestamps(self) -> NDArray[np.float64]:
        return np.asarray([frame.presentation_seconds for frame in self.frames], dtype=float)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MediaIndex:
        if data.get("schema_version", 1) != 1:
            raise ValueError("Incompatible media index schema; re-ingest source")
        values = dict(data)
        values["frames"] = [FrameTimestamp(**frame) for frame in values["frames"]]
        return cls(**values)


def _natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)
    ]


def _fps(value: str | float | None) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        result = float(numerator) / float(denominator) if float(denominator) else 0
    else:
        result = float(value)
    return result if np.isfinite(result) and result > 0 else None


def _timestamp_data(
    times: list[float], source: str, nominal_fps: float | None
) -> tuple[list[FrameTimestamp], dict[str, Any]]:
    quality = validate_timestamps(
        np.asarray(times), expected_interval=(1 / nominal_fps) if nominal_fps else None
    )
    duplicates = set(quality.duplicate_indices)
    frames = [
        FrameTimestamp(float(time), index, source, index in duplicates, False)
        for index, time in enumerate(times)
    ]
    diagnostics = asdict(quality)
    diagnostics["warnings"] = []
    if quality.duplicate_indices:
        diagnostics["warnings"].append("Duplicate presentation timestamps preserved")
    if quality.dropped_after_indices:
        diagnostics["warnings"].append(
            "Long timestamp gaps may indicate dropped frames or intentional VFR spacing"
        )
    return frames, diagnostics


def _fingerprint(paths: list[Path]) -> str:
    """Stat fingerprint for frame-cache invalidation, not an input content hash."""
    records = [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def _sequence_times(
    directory: Path, images: list[Path], nominal_fps: float | None
) -> tuple[list[float], str, str | None]:
    json_path, csv_path = directory / "timestamps.json", directory / "timestamps.csv"
    if json_path.is_file():
        data = json.loads(json_path.read_text())
        if isinstance(data, dict):
            data = data.get("timestamps", data.get("frames", data))
        if isinstance(data, dict):
            times = [float(data[image.name]) for image in images]
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            mapping = {
                str(item.get("filename", item.get("file", item.get("path")))): float(
                    item.get("timestamp", item.get("presentation_seconds"))
                )
                for item in data
            }
            times = [mapping[image.name] for image in images]
        elif isinstance(data, list):
            times = [float(value) for value in data]
        else:
            raise ValueError("timestamps.json must contain a timestamp list or filename map")
        return times, "explicit_sidecar", str(json_path)
    if csv_path.is_file():
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "timestamp" not in rows[0]:
            raise ValueError("timestamps.csv requires timestamp and optional filename columns")
        if "filename" in rows[0]:
            mapping = {row["filename"]: float(row["timestamp"]) for row in rows}
            times = [mapping[image.name] for image in images]
        else:
            times = [float(row["timestamp"]) for row in rows]
        return times, "explicit_sidecar", str(csv_path)
    if nominal_fps is None:
        raise ValueError("Image sequences require timestamps.json/csv or explicit nominal_fps")
    return [index / nominal_fps for index in range(len(images))], "nominal_fps_assumption", None


def ingest_media(source: Path, nominal_fps: float | None = None) -> MediaIndex:
    """Index video PTS or a naturally sorted image directory without decoding video.

    Sequence sidecars support JSON lists, filename-to-seconds maps, or a CSV
    with ``filename,timestamp`` columns. Source times are preserved, including a
    nonzero initial time. Backward discontinuities require explicit source splits.
    """
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Media source does not exist: {source}")
    if nominal_fps is not None and _fps(nominal_fps) is None:
        raise ValueError("Nominal FPS must be finite and positive")
    if source.is_dir() or source.suffix.lower() in IMAGE_SUFFIXES:
        images = (
            sorted(
                (
                    path
                    for path in source.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                ),
                key=_natural_key,
            )
            if source.is_dir()
            else [source]
        )
        if not images:
            raise ValueError(f"No supported image files in {source}")
        directory = source if source.is_dir() else source.parent
        try:
            times, timestamp_source, sidecar = _sequence_times(directory, images, nominal_fps)
        except (KeyError, TypeError) as error:
            raise ValueError(f"Malformed or incomplete timestamp sidecar in {directory}") from error
        if len(times) != len(images):
            raise ValueError("Sequence timestamp count does not match image count")
        width, height = 0, 0
        for image in images:
            with Image.open(image) as header:
                if width and header.size != (width, height):
                    raise ValueError(f"Sequence dimensions change at {image}")
                width, height = header.size
        frames, diagnostics = _timestamp_data(times, timestamp_source, nominal_fps)
        if timestamp_source == "nominal_fps_assumption":
            diagnostics["warnings"].append(
                "Sequence timing is assumed from explicitly supplied nominal FPS"
            )
        paths = images + ([Path(sidecar)] if sidecar else [])
        index = MediaIndex(
            str(source),
            "images",
            width,
            height,
            nominal_fps,
            frames,
            [str(image) for image in images],
            diagnostics,
            {
                "timestamp_source": timestamp_source,
                "sidecar": sidecar,
                "stat_fingerprint": _fingerprint(paths),
                "units": "seconds",
            },
        )
    else:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise RuntimeError("FFprobe is required for measured video PTS; run doctor")
        command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_streams",
            "-show_entries",
            "frame=best_effort_timestamp_time,pts_time,width,height:"
            "stream=width,height,avg_frame_rate,r_frame_rate,codec_name,time_base",
            "-of",
            "json",
            str(source),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise ValueError(f"FFprobe could not index {source}: {result.stderr.strip()}")
        try:
            data = json.loads(result.stdout)
            stream = data["streams"][0]
            video_frames = data["frames"]
        except (KeyError, IndexError, json.JSONDecodeError) as error:
            raise ValueError(f"No readable video stream in {source}") from error
        rate = _fps(stream.get("avg_frame_rate")) or _fps(stream.get("r_frame_rate"))
        times = []
        for number, frame in enumerate(video_frames):
            timestamp = frame.get("pts_time", frame.get("best_effort_timestamp_time"))
            if timestamp is None or timestamp == "N/A":
                raise ValueError(f"Frame {number} has no recoverable PTS; explicit timing required")
            if (frame.get("width", stream["width"]), frame.get("height", stream["height"])) != (
                stream["width"],
                stream["height"],
            ):
                raise ValueError(f"Video resolution changes at frame {number}")
            times.append(float(timestamp))
        frames, diagnostics = _timestamp_data(times, "ffprobe_presentation_timestamp", rate)
        index = MediaIndex(
            str(source),
            "video",
            int(stream["width"]),
            int(stream["height"]),
            rate,
            frames,
            diagnostics=diagnostics,
            provenance={
                "timestamp_source": "ffprobe_presentation_timestamp",
                "time_base": stream.get("time_base"),
                "codec": stream.get("codec_name"),
                "stat_fingerprint": _fingerprint([source]),
                "units": "seconds",
            },
        )
    LOGGER.info(
        "media.indexed",
        extra={
            "source": str(source),
            "frames": index.frame_count,
            "timestamp_source": index.provenance["timestamp_source"],
        },
    )
    return index


class FrameReader:
    """Lazy OpenCV BGR decoder with a bounded LRU; source PTS come from MediaIndex.

    The optional disk cache is explicitly located by the caller and bounded to
    ``max_memory_frames`` frames for this index. Decoding never substitutes
    OpenCV's synthetic ``CAP_PROP_POS_MSEC`` for measured presentation time.
    Use as a context manager or call close() to release video descriptors.
    """

    def __init__(
        self, index: MediaIndex, cache_dir: Path | None = None, max_memory_frames: int = 8
    ) -> None:
        if max_memory_frames < 1:
            raise ValueError("Frame cache capacity must be positive")
        fingerprint = index.provenance.get("stat_fingerprint")
        if fingerprint:
            paths = (
                [Path(index.source)]
                if index.kind == "video"
                else [Path(path) for path in index.source_paths]
            )
            if index.provenance.get("sidecar"):
                paths.append(Path(index.provenance["sidecar"]))
            if _fingerprint(paths) != fingerprint:
                raise ValueError(
                    "Media source or timestamp sidecar changed; re-ingest before decoding"
                )
        self.index = index
        self.max_memory_frames = max_memory_frames
        self._memory: OrderedDict[int, NDArray[np.uint8]] = OrderedDict()
        self._capture: cv2.VideoCapture | None = None
        self._next_index = 0
        self._disk: OrderedDict[int, Path] = OrderedDict()
        self.cache_dir: Path | None = None
        if cache_dir is not None:
            key = hashlib.sha256(json.dumps(index.to_dict(), sort_keys=True).encode()).hexdigest()
            self.cache_dir = Path(cache_dir).resolve() / key
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for path in sorted(
                self.cache_dir.glob("frame_*.npy"), key=lambda item: item.stat().st_mtime
            ):
                try:
                    self._disk[int(path.stem.removeprefix("frame_"))] = path
                except ValueError:
                    continue
            self._evict_disk()

    def _evict_disk(self) -> None:
        while len(self._disk) > self.max_memory_frames:
            _, path = self._disk.popitem(last=False)
            path.unlink(missing_ok=True)

    def _open(self) -> None:
        self.close()
        self._capture = cv2.VideoCapture(self.index.source)
        if not self._capture.isOpened():
            self.close()
            raise ValueError(f"OpenCV could not decode video: {self.index.source}")
        self._next_index = 0

    def _video_frame(self, number: int) -> NDArray[np.uint8]:
        if self._capture is None:
            self._open()
        assert self._capture is not None
        if number < self._next_index:
            self._open()
        assert self._capture is not None
        # Sequential grab is deliberate: backend-dependent seeking can land on
        # a different decode frame. Scrubbing is lazy and correct, though long
        # backward seeks may be expensive; proxies can improve that workflow.
        while self._next_index < number:
            if not self._capture.grab():
                raise ValueError(f"Video decoder ended before indexed frame {number}")
            self._next_index += 1
        success, frame = self._capture.read()
        self._next_index += 1
        if not success:
            raise ValueError(f"Video decoder failed at indexed frame {number}")
        return frame

    def get_frame(self, frame_index: int) -> NDArray[np.uint8]:
        """Return a copy of one BGR uint8 frame; raw cache cannot be edited by callers."""
        number = int(frame_index)
        if number != frame_index or not 0 <= number < self.index.frame_count:
            raise IndexError(f"Frame {frame_index} outside indexed range")
        if number in self._memory:
            self._memory.move_to_end(number)
            return self._memory[number].copy()
        disk_path = self._disk.get(number)
        if disk_path and disk_path.exists():
            try:
                frame = np.load(disk_path, allow_pickle=False)
                self._disk.move_to_end(number)
            except (OSError, ValueError):
                self._disk.pop(number, None)
                disk_path.unlink(missing_ok=True)
                frame = self._decode(number)
        else:
            frame = self._decode(number)
        if frame.shape != (self.index.height, self.index.width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"Decoded frame {number} differs from indexed dimensions/format")
        self._memory[number] = frame
        while len(self._memory) > self.max_memory_frames:
            self._memory.popitem(last=False)
        if self.cache_dir is not None and number not in self._disk:
            path = self.cache_dir / f"frame_{number:09d}.npy"
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, frame, allow_pickle=False)
            temporary.replace(path)
            self._disk[number] = path
            self._evict_disk()
        return frame.copy()

    def _decode(self, number: int) -> NDArray[np.uint8]:
        if self.index.kind == "video":
            return self._video_frame(number)
        frame = cv2.imread(self.index.source_paths[number], cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not decode image: {self.index.source_paths[number]}")
        return frame

    def iter_frames(self) -> Iterator[tuple[FrameTimestamp, NDArray[np.uint8]]]:
        for timestamp in self.index.frames:
            yield timestamp, self.get_frame(timestamp.frame_index)

    def __iter__(self) -> Iterator[tuple[FrameTimestamp, NDArray[np.uint8]]]:
        return self.iter_frames()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> FrameReader:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
