"""Top-down local ONNX whole-body landmarks: SimCC, heatmaps, or XY scores."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from openmocap.detection import ONNXEngine
from openmocap.types import PersonDetection, Pose2DObservation


class PoseBackend(Protocol):
    def infer(
        self, image: np.ndarray, detections: list[PersonDetection]
    ) -> list[Pose2DObservation]: ...


def decode_simcc(
    x: np.ndarray, y: np.ndarray, split_ratio: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Decode RTMPose SimCC maxima in input-image coordinates.

    Scores are the minimum of the two measured network maxima, matching the
    official codec convention. Models exported with logits need explicit score
    calibration; confidence transform is configured on the backend.
    """
    if x.ndim != 3 or y.ndim != 3 or x.shape[:2] != y.shape[:2] or split_ratio <= 0:
        raise ValueError("SimCC outputs must be matching [B,J,Lx], [B,J,Ly] tensors")
    xy = np.stack([x.argmax(axis=-1), y.argmax(axis=-1)], axis=-1) / split_ratio
    confidence = np.minimum(x.max(axis=-1), y.max(axis=-1))
    return xy.astype(float), confidence


def decode_heatmaps(
    heatmaps: np.ndarray, input_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum + local parabolic subpixel heatmap refinement."""
    if heatmaps.ndim != 4:
        raise ValueError("Heatmap output must have [B,J,H,W] shape")
    batches, joints, height, width = heatmaps.shape
    flat = heatmaps.reshape(batches, joints, -1)
    locations = flat.argmax(axis=-1)
    xy = np.stack([locations % width, locations // width], axis=-1).astype(float)
    confidence = flat.max(axis=-1)
    for batch in range(batches):
        for joint in range(joints):
            x, y = xy[batch, joint].astype(int)
            for axis, position, maximum in [(0, x, width), (1, y, height)]:
                if 0 < position < maximum - 1:
                    left = (
                        heatmaps[batch, joint, y, x - 1]
                        if axis == 0
                        else heatmaps[batch, joint, y - 1, x]
                    )
                    right = (
                        heatmaps[batch, joint, y, x + 1]
                        if axis == 0
                        else heatmaps[batch, joint, y + 1, x]
                    )
                    middle = heatmaps[batch, joint, y, x]
                    denominator = left - 2 * middle + right
                    if abs(denominator) > 1e-10:
                        xy[batch, joint, axis] += np.clip(
                            0.5 * (left - right) / denominator, -0.5, 0.5
                        )
    xy *= np.array(input_size) / np.array([width, height])
    return xy, confidence


class ONNXPoseBackend:
    """Complete RTMPose/ViTPose-compatible top-down model execution.

    The joint_names mapping is required: no guessing of COCO versus whole-body
    topology. Input crop is affine, reversible, RGB by default. Original measured
    observations are returned; there is no destructive temporal smoothing.
    """

    def __init__(
        self,
        model: str | Path,
        joint_names: list[str],
        *,
        output_format: str = "simcc",
        input_size: tuple[int, int] = (192, 256),
        mean: tuple[float, float, float] = (123.675, 116.28, 103.53),
        std: tuple[float, float, float] = (58.395, 57.12, 57.375),
        rgb: bool = True,
        bbox_padding: float = 1.25,
        split_ratio: float = 2.0,
        confidence_transform: str = "clip",
        keypoints_normalized: bool = False,
        providers: list[str] | None = None,
        batch_size: int = 16,
    ) -> None:
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("Unique named joints are required for every model output landmark")
        if output_format not in {"simcc", "heatmap", "keypoints"}:
            raise ValueError("Pose format must be simcc, heatmap, or keypoints")
        if confidence_transform not in {"clip", "sigmoid"}:
            raise ValueError("Confidence transform must be clip or sigmoid")
        if any(s <= 0 for s in std) or bbox_padding <= 0:
            raise ValueError("Positive input standard deviations and crop padding required")
        self.engine = ONNXEngine(model, providers=providers)
        self.joint_names, self.format = list(joint_names), output_format
        self.input_size = self.engine.image_size(input_size)
        self.mean, self.std = np.array(mean), np.array(std)
        self.rgb, self.padding, self.split_ratio = rgb, bbox_padding, split_ratio
        self.confidence_transform, self.normalized = confidence_transform, keypoints_normalized
        batch = self.engine.input.shape[0]
        self.batch_size = int(batch) if isinstance(batch, int) else max(1, batch_size)
        self.fixed_batch = isinstance(batch, int)
        self.metadata = {
            **self.engine.metadata,
            "backend": "onnx_pose",
            "format": output_format,
            "joint_names": self.joint_names,
            "input_size": list(self.input_size),
            "mean": list(mean),
            "std": list(std),
            "rgb": rgb,
        }

    def _crop(self, image: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        box = np.asarray(box, dtype=float)
        centre = (box[:2] + box[2:]) / 2
        extent = (box[2:] - box[:2]) * self.padding
        if extent.min() <= 0:
            raise ValueError("Pose crop requires a positive XYXY person rectangle")
        width, height = self.input_size
        extent = np.array(
            [max(extent[0], extent[1] * width / height), max(extent[1], extent[0] * height / width)]
        )
        affine = np.array(
            [
                [width / extent[0], 0, width / 2 - centre[0] * width / extent[0]],
                [0, height / extent[1], height / 2 - centre[1] * height / extent[1]],
            ]
        )
        crop = cv2.warpAffine(
            image, affine, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        if self.rgb:
            crop = crop[..., ::-1]
        tensor = ((crop.astype(float) - self.mean) / self.std).transpose(2, 0, 1).astype(np.float32)
        return tensor, cv2.invertAffineTransform(affine)

    def infer(
        self, image: np.ndarray, detections: list[PersonDetection]
    ) -> list[Pose2DObservation]:
        result: list[Pose2DObservation] = []
        for start in range(0, len(detections), self.batch_size):
            selected = detections[start : start + self.batch_size]
            crops = [self._crop(image, np.asarray(d.bbox)) for d in selected]
            tensors = [c[0] for c in crops]
            if self.fixed_batch:
                tensors.extend([np.zeros_like(tensors[0])] * (self.batch_size - len(tensors)))
            output = self.engine.run(np.stack(tensors))
            if self.format == "simcc":
                if len(output) < 2:
                    raise ValueError("SimCC requires x and y output tensors")
                xy, confidence = decode_simcc(output[0], output[1], self.split_ratio)
            elif self.format == "heatmap":
                xy, confidence = decode_heatmaps(output[0], self.input_size)
            else:
                values = np.asarray(output[0])
                if values.ndim != 3 or values.shape[-1] != 3:
                    raise ValueError("Keypoint output requires [B,J,3] x,y,confidence")
                xy, confidence = values[..., :2].copy(), values[..., 2]
                if self.normalized:
                    xy *= np.array(self.input_size)
            if xy.shape[1] != len(self.joint_names):
                raise ValueError(
                    f"Model emits {xy.shape[1]} joints; joint_names has {len(self.joint_names)}"
                )
            if self.confidence_transform == "sigmoid":
                confidence = 1 / (1 + np.exp(-np.clip(confidence, -30, 30)))
            confidence = np.clip(confidence, 0, 1)
            for batch, (detection, (_, inverse)) in enumerate(zip(selected, crops, strict=True)):
                pixels = np.column_stack([xy[batch], np.ones(len(self.joint_names))]) @ inverse.T
                for joint, name in enumerate(self.joint_names):
                    if np.isfinite(pixels[joint]).all() and np.isfinite(confidence[batch, joint]):
                        result.append(
                            Pose2DObservation(
                                detection.camera_id,
                                detection.camera_timestamp,
                                detection.person_id,
                                name,
                                pixels[joint],
                                float(confidence[batch, joint] * detection.confidence),
                                source="onnx_pose",
                                metadata={
                                    "model_sha256": self.engine.metadata["sha256"],
                                    "raw_model_confidence": float(confidence[batch, joint]),
                                    "detector_confidence": detection.confidence,
                                },
                            )
                        )
        return result
