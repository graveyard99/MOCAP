"""Confidence-aware continuous-time motion; raw observations remain untouched."""

from .spline import ContinuousTrajectory, fit_trajectory

__all__ = ["ContinuousTrajectory", "fit_trajectory"]
