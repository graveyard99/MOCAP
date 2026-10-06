"""CPU-testable detection and local ONNX inference; no implicit downloads."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from openmocap.types import PersonDetection


class Detector(Protocol):
    def detect(
        self, image: np.ndarray, camera_id: str, timestamp: float
    ) -> list[PersonDetection]: ...


def model_digest(path: str | Path) -> str:
    """Stream checkpoint fingerprint; model assets are never redistributed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ONNXEngine:
    """Validated local model execution, bounded CPU threads and explicit providers."""

    def __init__(
        self, path: str | Path, *, providers: list[str] | None = None, threads: int = 2
    ) -> None:
        import onnxruntime as ort

        path = Path(path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".onnx":
            raise FileNotFoundError(f"Provide an existing local .onnx checkpoint: {path}")
        providers = providers or ["CPUExecutionProvider"]
        unavailable = set(providers) - set(ort.get_available_providers())
        if unavailable:
            raise RuntimeError(f"ONNX execution providers unavailable: {sorted(unavailable)}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        active = self.session.get_providers()
        inactive = set(providers) - set(active)
        if inactive:
            raise RuntimeError(
                f"Requested ONNX providers failed to initialize: {sorted(inactive)}; active providers: {active}. "
                "Check matching GPU runtime libraries; CPU fallback is not accepted for a requested GPU solve"
            )
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError(
                "Image backend requires one float32 NCHW input; export a compatible model"
            )
        self.input = inputs[0]
        if len(self.input.shape) != 4 or self.input.shape[1] not in (3, "channels", None):
            raise ValueError("Image ONNX input must have NCHW RGB/BGR shape with three channels")
        self.metadata = {
            "path": str(path),
            "sha256": model_digest(path),
            "providers": active,
            "input_shape": self.input.shape,
            "outputs": [
                {"name": item.name, "shape": item.shape, "type": item.type}
                for item in self.session.get_outputs()
            ],
        }

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input.name: np.asarray(tensor, np.float32)})

    def image_size(self, fallback: tuple[int, int]) -> tuple[int, int]:
        shape = self.input.shape
        return (
            int(shape[3]) if isinstance(shape[3], int) else fallback[0],
            int(shape[2]) if isinstance(shape[2], int) else fallback[1],
        )


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.45) -> list[int]:
    """Deterministic XYXY NMS, preserving input indices."""
    if not len(boxes):
        return []
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        selected = int(order[0])
        keep.append(selected)
        remaining = order[1:]
        a = np.maximum(boxes[selected, :2], boxes[remaining, :2])
        b = np.minimum(boxes[selected, 2:], boxes[remaining, 2:])
        intersection = np.prod(np.maximum(0, b - a), axis=1)
        areas = np.prod(np.maximum(0, boxes[remaining, 2:] - boxes[remaining, :2]), axis=1)
        selected_area = np.prod(np.maximum(0, boxes[selected, 2:] - boxes[selected, :2]))
        iou = intersection / np.maximum(areas + selected_area - intersection, 1e-12)
        order = remaining[iou <= threshold]
    return keep


class HOGDetector:
    """OpenCV's shipped pedestrian HOG/SVM, no network or checkpoint download.

    This baseline has limited recall and is unsuitable for whole-body landmark
    inference; use explicitly configured ONNX weights for production detections.
    """

    def __init__(self, confidence: float = 0.1, max_width: int = 960) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.confidence = confidence
        self.max_width = max_width
        self.metadata = {
            "backend": "opencv_hog",
            "opencv_version": cv2.__version__,
            "limitations": "upright pedestrian baseline; no neural whole-body inference",
        }

    def detect(self, image: np.ndarray, camera_id: str, timestamp: float) -> list[PersonDetection]:
        height, width = image.shape[:2]
        factor = min(1.0, self.max_width / width)
        resized = cv2.resize(image, (round(width * factor), round(height * factor)))
        if resized.shape[0] < 128 or resized.shape[1] < 64:
            return []
        boxes, weights = self.hog.detectMultiScale(
            resized, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        candidates, scores = [], []
        for (x, y, w, h), weight in zip(boxes, np.asarray(weights).reshape(-1), strict=True):
            score = float(1 / (1 + np.exp(-np.clip(weight, -30, 30))))
            if score >= self.confidence:
                candidates.append(np.array([x, y, x + w, y + h]) / factor)
                scores.append(score)
        boxes = np.asarray(candidates).reshape(-1, 4)
        scores = np.asarray(scores)
        return [
            PersonDetection(
                camera_id, timestamp, "", boxes[i], float(scores[i]), source="opencv_hog"
            )
            for i in nms(boxes, scores)
        ]


class ONNXDetector:
    """YOLOX raw/decoded boxes, YOLO centres and explicit deployed detections.

    ``format`` is mandatory evidence about output semantics, never guessed.
    YOLOX: [cx,cy,w,h,objectness,class probabilities...]; YOLO: centres and
    class probabilities (no objectness); XYXY: [x1,y1,x2,y2,score,class_id].
    ``yolox_raw`` decodes the specified stride grid before interpreting boxes.
    ``mmdeploy`` consumes [1,N,5] XYXY+score and [1,N] class-label outputs.
    Output semantics are configured explicitly, never guessed from tensor sizes.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        output_format: str = "yolox",
        input_size: tuple[int, int] = (640, 640),
        confidence: float = 0.3,
        person_class: int = 0,
        normalized_boxes: bool = False,
        rgb: bool = False,
        divide_by_255: bool = False,
        strides: tuple[int, ...] = (8, 16, 32),
        providers: list[str] | None = None,
    ) -> None:
        if output_format not in {"yolox", "yolox_raw", "yolo", "xyxy", "mmdeploy"}:
            raise ValueError(
                "Detector output format must be yolox, yolox_raw, yolo, xyxy, or mmdeploy"
            )
        if not strides or any(int(s) != s or s <= 0 for s in strides):
            raise ValueError("YOLOX strides must be positive integer pixel spacings")
        self.engine = ONNXEngine(model, providers=providers)
        self.input_size = self.engine.image_size(input_size)
        self.format, self.confidence, self.person_class = output_format, confidence, person_class
        self.normalized_boxes, self.rgb, self.divide_by_255 = normalized_boxes, rgb, divide_by_255
        self.strides = tuple(int(s) for s in strides)
        self.metadata = {
            **self.engine.metadata,
            "backend": "onnx_detector",
            "format": output_format,
            "strides": list(self.strides),
            "rgb": rgb,
            "divide_by_255": divide_by_255,
        }

    def detect(self, image: np.ndarray, camera_id: str, timestamp: float) -> list[PersonDetection]:
        width, height = self.input_size
        ratio = min(width / image.shape[1], height / image.shape[0])
        resized = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
        padded = np.full((height, width, 3), 114, np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        if self.rgb:
            padded = padded[..., ::-1]
        tensor = padded.transpose(2, 0, 1)[None].astype(np.float32)
        if self.divide_by_255:
            tensor /= 255
        outputs = self.engine.run(tensor)
        predictions = np.asarray(outputs[0])
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise ValueError(f"Detector output must be [N,C] or [1,N,C], got {predictions.shape}")
        if self.format == "yolo" and predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        if self.format == "mmdeploy":
            if predictions.shape[1] != 5 or len(outputs) < 2:
                raise ValueError("MMDeploy requires XYXY+score [1,N,5] and class labels [1,N]")
            labels = np.asarray(outputs[1]).reshape(-1)
            if len(labels) != len(predictions):
                raise ValueError("MMDeploy class labels must match the number of detections")
            boxes = predictions[:, :4].copy()
            scores = np.where(labels.astype(int) == self.person_class, predictions[:, 4], 0)
        elif self.format == "xyxy":
            if predictions.shape[1] != 6:
                raise ValueError("XYXY detections require exactly six fields")
            boxes = predictions[:, :4].copy()
            scores = predictions[:, 4]
            scores = np.where(predictions[:, 5].astype(int) == self.person_class, scores, 0)
        else:
            start = 5 if self.format in {"yolox", "yolox_raw"} else 4
            if predictions.shape[1] <= start + self.person_class:
                raise ValueError("Model has no configured person class probability")
            if self.format == "yolox_raw":
                predictions = decode_yolox_grid(predictions, self.input_size, self.strides)
            centres, sizes = predictions[:, :2], predictions[:, 2:4]
            boxes = np.column_stack([centres - sizes / 2, centres + sizes / 2])
            classes = predictions[:, start:]
            scores = classes[:, self.person_class]
            scores = np.where(classes.argmax(axis=1) == self.person_class, scores, 0)
            if self.format in {"yolox", "yolox_raw"}:
                scores = scores * predictions[:, 4]
        if self.normalized_boxes:
            boxes *= np.array([width, height, width, height])
        valid = (scores >= self.confidence) & np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
        boxes, scores = boxes[valid] / ratio, np.clip(scores[valid], 0, 1)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image.shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image.shape[0])
        keep = nms(boxes, scores)
        return [
            PersonDetection(
                camera_id, timestamp, "", boxes[i], float(scores[i]), source="onnx_detector"
            )
            for i in keep
            if boxes[i, 2] > boxes[i, 0] and boxes[i, 3] > boxes[i, 1]
        ]


def decode_yolox_grid(
    predictions: np.ndarray, input_size: tuple[int, int], strides: tuple[int, ...] = (8, 16, 32)
) -> np.ndarray:
    """Decode official YOLOX ONNX raw head, preserving the original output.

    Input is [N,5+C]. Grid flattening is row-major, x changes fastest. A tensor
    inconsistent with configured image size/strides fails loudly.
    """
    width, height = input_size
    grids, scales = [], []
    for stride in strides:
        x, y = np.meshgrid(np.arange(width // stride), np.arange(height // stride))
        grid = np.column_stack([x.reshape(-1), y.reshape(-1)])
        grids.append(grid)
        scales.append(np.full((len(grid), 1), stride))
    grid, scale = np.concatenate(grids), np.concatenate(scales)
    if predictions.ndim != 2 or predictions.shape[1] < 6 or len(predictions) != len(grid):
        raise ValueError(
            f"YOLOX raw output count {len(predictions)} does not match stride grid {len(grid)}"
        )
    decoded = np.asarray(predictions, dtype=float).copy()
    decoded[:, :2] = (decoded[:, :2] + grid) * scale
    decoded[:, 2:4] = np.exp(np.clip(decoded[:, 2:4], -30, 30)) * scale
    return decoded


def projected_actor_roi(
    camera: Any,
    joints: np.ndarray,
    timestamp: float,
    *,
    padding: float = 0.15,
    minimum_size: int = 64,
) -> np.ndarray | None:
    """Feedback from measured 3D tracking to an image-space search rectangle.

    This never changes camera geometry. It returns a predicted region tagged by
    the caller separately from detector measurements.
    """
    pixels = camera.project(joints, time=timestamp)
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    if not len(pixels):
        return None
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    extent = np.maximum(high - low, minimum_size)
    centre = (low + high) / 2
    low, high = centre - extent * (0.5 + padding), centre + extent * (0.5 + padding)
    intrinsics = camera.intrinsics_at(timestamp)
    return np.array(
        [
            max(0, low[0]),
            max(0, low[1]),
            min(intrinsics.width, high[0]),
            min(intrinsics.height, high[1]),
        ]
    )
