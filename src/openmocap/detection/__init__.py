"""Interchangeable detection backends and measured-geometry ROI feedback."""

from .backends import Detector, HOGDetector, ONNXDetector, ONNXEngine, nms, projected_actor_roi

__all__ = ["Detector", "HOGDetector", "ONNXDetector", "ONNXEngine", "nms", "projected_actor_roi"]
