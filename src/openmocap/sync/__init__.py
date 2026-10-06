"""Affine camera clock synchronization and explicit timestamp diagnostics."""

from .reprojection import TimingRefinementResult, refine_multiview_timing
from .timing import (
    TimestampDiagnostics,
    estimate_from_events,
    estimate_time_mapping,
    refine_time_mapping,
    validate_timestamps,
)

__all__ = [
    "TimestampDiagnostics",
    "estimate_from_events",
    "estimate_time_mapping",
    "refine_time_mapping",
    "TimingRefinementResult",
    "refine_multiview_timing",
    "validate_timestamps",
]
