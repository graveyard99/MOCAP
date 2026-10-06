"""Camera-local deterministic assignment. Raw detections stay immutable."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from openmocap.types import PersonDetection


def appearance_embedding(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """HSV histogram descriptor; explicitly not a neural identity embedding."""
    x1, y1, x2, y2 = np.rint(bbox).astype(int)
    crop = image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    if not crop.size:
        return np.zeros(64)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    descriptor = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).reshape(-1)
    return descriptor / max(np.linalg.norm(descriptor), 1e-12)


@dataclass
class _State:
    id: str
    box: np.ndarray
    timestamp: float
    missed: int = 0


class IoUTracker:
    """Hungarian IoU assignment with bounded track age and explicit new IDs."""

    def __init__(self, camera_id: str, min_iou: float = 0.2, max_age_seconds: float = 1.0) -> None:
        self.camera_id, self.min_iou, self.max_age = camera_id, min_iou, max_age_seconds
        self.tracks: dict[str, _State] = {}
        self.counter = 0
        self.last_timestamp = -np.inf

    def update(self, detections: list[PersonDetection], timestamp: float) -> list[PersonDetection]:
        if timestamp < self.last_timestamp:
            raise ValueError(
                "Tracker timestamps must be monotonic within a continuous media segment"
            )
        self.last_timestamp = timestamp
        self.tracks = {
            k: v for k, v in self.tracks.items() if timestamp - v.timestamp <= self.max_age
        }
        states = list(self.tracks.values())
        identities = [""] * len(detections)
        if states and detections:
            old = np.array([s.box for s in states])
            new = np.array([d.bbox for d in detections])
            intersection = np.prod(
                np.maximum(
                    0,
                    np.minimum(old[:, None, 2:], new[None, :, 2:])
                    - np.maximum(old[:, None, :2], new[None, :, :2]),
                ),
                axis=2,
            )
            union = (
                np.prod(old[:, 2:] - old[:, :2], axis=1)[:, None]
                + np.prod(new[:, 2:] - new[:, :2], axis=1)[None, :]
                - intersection
            )
            iou = intersection / np.maximum(union, 1e-12)
            rows, columns = linear_sum_assignment(1 - iou)
            for row, column in zip(rows, columns, strict=True):
                if iou[row, column] >= self.min_iou:
                    identities[column] = states[row].id
        result = []
        for index, detection in enumerate(detections):
            identity = identities[index]
            if not identity:
                self.counter += 1
                identity = f"{self.camera_id}:track{self.counter:04d}"
            self.tracks[identity] = _State(identity, np.array(detection.bbox), timestamp)
            result.append(replace(detection, person_id=identity))
        return result
