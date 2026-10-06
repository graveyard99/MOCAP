"""Non-destructive segmentation backends with explicit mask provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from openmocap.detection import ONNXEngine
from openmocap.types import PersonDetection


class Segmenter(Protocol):
    def infer(self, image: np.ndarray, detection: PersonDetection) -> tuple[np.ndarray, float]: ...


class GrabCutSegmenter:
    """Deterministic CPU rectangle-conditioned foreground fallback, not a neural model.

    Clothing/background colour ambiguity may produce poor masks. Low confidence
    is intentional; this fallback must not dominate measured skeleton geometry.
    """

    def __init__(self, iterations: int = 5) -> None:
        self.iterations = max(1, iterations)
        self.metadata = {
            "backend": "opencv_grabcut",
            "learned": False,
            "limitations": "colour foreground baseline, not human-semantic segmentation",
        }

    def infer(self, image: np.ndarray, detection: PersonDetection) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = np.rint(detection.bbox).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return np.zeros((height, width), np.uint8), 0.0
        # A background rim is needed by GrabCut even for a full-plate ROI.
        x1, y1 = max(1, x1), max(1, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        mask = np.zeros((height, width), np.uint8)
        cv2.setRNGSeed(0)
        try:
            cv2.grabCut(
                image,
                mask,
                (x1, y1, x2 - x1, y2 - y1),
                np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64),
                self.iterations,
                cv2.GC_INIT_WITH_RECT,
            )
        except cv2.error:
            return np.zeros((height, width), np.uint8), 0.0
        foreground = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
        return foreground, float(0.35 * detection.confidence if foreground.any() else 0.0)


class ONNXSegmenter:
    """Configured local human-semantic model [B,C,H,W]/[B,1,H,W]/[B,H,W].

    Entire plate is resized with explicit direct geometry; labels/probabilities
    are mapped back without modifying original output. Instance selection is
    clipped to the detection ROI, which is not a substitute for instance models.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        input_size: tuple[int, int] = (512, 512),
        foreground_class: int = 1,
        logits: bool | None = True,
        threshold: float = 0.5,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        output_format: str = "generic",
        providers: list[str] | None = None,
    ) -> None:
        if (
            len(mean) != 3
            or len(std) != 3
            or not np.isfinite([*mean, *std, threshold]).all()
            or any(s <= 0 for s in std)
            or not 0 <= threshold <= 1
            or foreground_class < 0
        ):
            raise ValueError("Segmentation normalization and probability threshold invalid")
        if output_format not in {"generic", "pphumanseg"}:
            raise ValueError("Segmentation output_format must be generic or pphumanseg")
        self.engine = ONNXEngine(model, providers=providers)
        self.input_size = self.engine.image_size(input_size)
        if output_format == "pphumanseg" and (
            self.input_size != (192, 192)
            or foreground_class != 1
            or not np.allclose(mean, (0.5, 0.5, 0.5))
            or not np.allclose(std, (0.5, 0.5, 0.5))
        ):
            raise ValueError("PPHumanSeg requires 192x192 RGB, mean/std 0.5 and foreground class 1")
        self.foreground_class, self.logits, self.threshold = foreground_class, logits, threshold
        self.output_format = output_format
        self.mean, self.std = np.array(mean, np.float32), np.array(std, np.float32)
        self.metadata = {
            **self.engine.metadata,
            "backend": "onnx_segmentation",
            "foreground_class": foreground_class,
            "logits": logits,
            "output_format": output_format,
            "input_size": list(self.input_size),
            "color_order": "RGB",
            "mean": list(mean),
            "std": list(std),
            "threshold": threshold,
        }

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """BGR uint8 plate to explicitly normalized float32 NCHW RGB input."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Segmentation requires a BGR uint8 image with three channels")
        rgb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), self.input_size)
        rgb = rgb.astype(np.float32) / 255.0
        return np.ascontiguousarray(((rgb - self.mean) / self.std).transpose(2, 0, 1)[None])

    def _pphumanseg_scores(
        self, result: np.ndarray, size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match publisher upsample-then-argmax masks; confidence is heuristic.

        The publisher selects the class from two score planes without exposing
        probability calibration. With logits=None, normalized probability planes
        are preserved; other scores receive softmax only to report confidence.
        Neither interpretation changes the score-argmax mask.
        """
        if result.ndim != 4 or result.shape[:2] != (1, 2) or not np.isfinite(result).all():
            raise ValueError("PPHumanSeg requires finite [1,2,H,W] class scores")
        scores = cv2.resize(result[0].transpose(1, 2, 0), size, interpolation=cv2.INTER_LINEAR)
        scores = scores.transpose(2, 0, 1)
        probabilities = bool(
            scores.min() >= 0
            and scores.max() <= 1
            and np.allclose(scores.sum(axis=0), 1, atol=1e-4)
        )
        if self.logits is False and not probabilities:
            raise ValueError(
                "PPHumanSeg probability output is not normalized; inspect logits setting"
            )
        if self.logits is True or (self.logits is None and not probabilities):
            normalized = np.exp(scores - scores.max(axis=0))
            normalized /= normalized.sum(axis=0)
            self.metadata["confidence_interpretation"] = "softmax_normalized_class_scores"
        else:
            normalized = scores
            self.metadata["confidence_interpretation"] = "normalized_class_probabilities"
        return normalized[self.foreground_class], np.argmax(scores, axis=0) == self.foreground_class

    def infer(self, image: np.ndarray, detection: PersonDetection) -> tuple[np.ndarray, float]:
        result = np.asarray(self.engine.run(self.preprocess(image))[0])
        foreground = None
        if self.output_format == "pphumanseg":
            probability, foreground = self._pphumanseg_scores(
                result, (image.shape[1], image.shape[0])
            )
        elif result.ndim == 4:
            result = result[0]
            if result.shape[0] == 1:
                probability = result[0]
                if self.logits:
                    probability = 1 / (1 + np.exp(-np.clip(probability, -30, 30)))
            else:
                if self.foreground_class >= result.shape[0]:
                    raise ValueError("Segmentation foreground_class exceeds model class count")
                if self.logits:
                    result = np.exp(result - result.max(axis=0))
                    result /= result.sum(axis=0)
                probability = result[self.foreground_class]
        elif result.ndim == 3 and result.shape[0] == 1:
            probability = result[0]
            if self.logits:
                probability = 1 / (1 + np.exp(-np.clip(probability, -30, 30)))
        else:
            raise ValueError("Segmentation output must be [1,C,H,W] or [1,H,W]")
        probability = cv2.resize(probability.astype(np.float32), (image.shape[1], image.shape[0]))
        x1, y1, x2, y2 = np.rint(detection.bbox).astype(int)
        x1, x2 = np.clip([x1, x2], 0, image.shape[1])
        y1, y2 = np.clip([y1, y2], 0, image.shape[0])
        roi = np.zeros(image.shape[:2], bool)
        roi[y1:y2, x1:x2] = True
        mask = ((probability >= self.threshold) & roi).astype(np.uint8)
        if foreground is not None:
            mask &= foreground.astype(np.uint8)
        confidence = (
            float(probability[mask.astype(bool)].mean() * detection.confidence)
            if mask.any()
            else 0.0
        )
        return mask, confidence
