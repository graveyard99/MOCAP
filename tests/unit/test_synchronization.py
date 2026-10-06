import numpy as np
import pytest

from openmocap.sync import (
    estimate_from_events,
    estimate_time_mapping,
    refine_time_mapping,
    validate_timestamps,
)
from openmocap.types import TimeMapping


def signal(times):
    return np.column_stack(
        [
            np.sin(times * 2.13) + 0.3 * np.cos(times * 0.57),
            0.7 * np.sin(times * 0.83) + 0.1 * times,
        ]
    )


def test_subframe_offset_and_affine_drift():
    reference = np.arange(0, 25, 1 / 60)
    camera = np.arange(0, 24, 1 / 29.97)
    scale, offset = 1.00037, 0.0173
    mapping = estimate_time_mapping(
        camera,
        signal(camera * scale + offset),
        reference,
        signal(reference),
        offset_bounds=(-0.1, 0.1),
        scale_bounds=(0.999, 1.001),
    )
    assert mapping.offset == pytest.approx(offset, abs=0.00015)
    assert mapping.scale == pytest.approx(scale, abs=1e-5)


def test_timestamp_duplicate_drop_and_discontinuity():
    times = np.array([0, 0.1, 0.2, 0.2, 0.3, 0.5, 0.6])
    diagnostic = validate_timestamps(times)
    assert diagnostic.duplicate_indices == (3,)
    assert diagnostic.dropped_after_indices == (4,)
    with pytest.raises(ValueError, match="discontinuity"):
        validate_timestamps([0, 0.1, -0.1])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_timestamps(times, allow_duplicates=False)
    assert not validate_timestamps([0, 0.1, -0.1], allow_discontinuities=True).valid


def test_dropped_duplicate_samples_alignment_and_lock():
    reference = np.arange(0, 10, 0.02)
    camera = np.arange(0, 9, 0.033)[::2]
    camera = np.insert(camera, 5, camera[5])
    mapping = estimate_time_mapping(
        camera, signal(camera + 0.022), reference, signal(reference), offset_bounds=(-0.1, 0.1)
    )
    assert mapping.offset == pytest.approx(0.022, abs=0.001)
    locked = TimeMapping(scale=1.0001, offset=0.03, locked=True)
    assert (
        estimate_time_mapping(camera, signal(camera), reference, signal(reference), initial=locked)
        is locked
    )
    refined, diagnostics = refine_time_mapping(locked, lambda o, s: np.array([o - 0.5]))
    assert refined is locked
    assert diagnostics["offset_delta"] == 0


def test_bounded_refinement_records_deltas():
    mapping = TimeMapping(offset=0, locked=False)
    refined, diagnostic = refine_time_mapping(
        mapping,
        lambda offset, scale: np.array([1000 * (offset - 0.015)]),
        max_offset_change=0.02,
        offset_prior_sigma=0.1,
    )
    assert 0.014 < refined.offset < 0.02
    assert diagnostic["final_cost"] < diagnostic["initial_cost"]
    assert mapping.offset == 0


def test_matched_flash_events_offset_drift_and_single_event():
    camera = np.array([1.0, 20.0, 55.0])
    world = camera * 1.0002 + 0.0123
    mapping, diagnostics = estimate_from_events(camera, world)
    assert mapping.scale == pytest.approx(1.0002, abs=1e-8)
    assert mapping.offset == pytest.approx(0.0123, abs=1e-8)
    assert diagnostics["drift_estimated"]
    offset_only, diagnostics = estimate_from_events(np.array([1.0]), np.array([1.023]))
    assert offset_only.offset == pytest.approx(0.023)
    assert offset_only.scale == 1
    assert not diagnostics["drift_estimated"]
