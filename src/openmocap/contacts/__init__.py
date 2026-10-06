"""Calibrated-plane contact inference and confidence-gated drift correction."""

from .feet import (
    ContactResult,
    ContactSettings,
    RefinementResult,
    apply_contact_overrides,
    estimate_contacts,
    refine_contacts,
)

__all__ = [
    "ContactResult",
    "ContactSettings",
    "RefinementResult",
    "apply_contact_overrides",
    "estimate_contacts",
    "refine_contacts",
]
