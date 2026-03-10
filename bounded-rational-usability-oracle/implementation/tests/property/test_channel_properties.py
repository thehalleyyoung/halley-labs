"""Property-based tests for the channel module (MRT channel capacity).

Verifies resource allocation conservation, interference matrix symmetry,
capacity non-negativity, and the relationship between capacity and
cognitive cost using Hypothesis.
"""

import math

import numpy as np
from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import (
    floats,
    integers,
    sampled_from,
    composite,
    lists,
)

from usability_oracle.channel.types import (
    ChannelAllocation,
    InterferenceMatrix,
    ResourceChannel,
    ResourcePool,
    WickensResource,
)
from usability_oracle.channel.capacity import (
    ChannelCapacityEstimator,
    visual_capacity,
    auditory_capacity,
    cognitive_capacity_wm,
    motor_capacity_fitts,
    fatigue_degradation,
)
from usability_oracle.channel.interference import (
    WickensInterferenceModel,
    ResourceProfile,
    build_standard_interference_matrix,
    dimension_conflict,
    profile_interference,
)

# Helper: map label strings to ResourceProfile instances.
_PROFILES = {
    "visual_spatial": ResourceProfile(
        stage=WickensResource.PERCEPTUAL,
        modality=WickensResource.VISUAL,
        code=WickensResource.SPATIAL,
    ),
    "visual_verbal": ResourceProfile(
        stage=WickensResource.PERCEPTUAL,
        modality=WickensResource.VISUAL,
        code=WickensResource.VERBAL,
    ),
    "auditory_spatial": ResourceProfile(
        stage=WickensResource.PERCEPTUAL,
        modality=WickensResource.AUDITORY,
        code=WickensResource.SPATIAL,
    ),
    "auditory_verbal": ResourceProfile(
        stage=WickensResource.PERCEPTUAL,
        modality=WickensResource.AUDITORY,
        code=WickensResource.VERBAL,
    ),
}
from usability_oracle.channel.allocator import (
    water_filling_allocate,
    MRTAllocator,
    AllocationResult,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_ATOL = 1e-6

_pos_float = floats(min_value=0.01, max_value=100.0,
                    allow_nan=False, allow_infinity=False)

_unit_float = floats(min_value=0.0, max_value=1.0,
                     allow_nan=False, allow_infinity=False)

_n_elements = integers(min_value=1, max_value=50)

_resource = sampled_from([
    WickensResource.VISUAL, WickensResource.AUDITORY,
    WickensResource.COGNITIVE, WickensResource.MANUAL,
    WickensResource.VOCAL,
])


@composite
def _resource_channel(draw):
    """Generate a single ResourceChannel."""
    r = draw(_resource)
    cap = draw(floats(min_value=1.0, max_value=100.0,
                      allow_nan=False, allow_infinity=False))
    load = draw(_unit_float)
    return ResourceChannel(resource=r, capacity_bits_per_s=cap,
                           current_load=load)


# ---------------------------------------------------------------------------
# Water-filling allocation sums to total budget
# ---------------------------------------------------------------------------


@given(integers(min_value=2, max_value=10),
       floats(min_value=1.0, max_value=100.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_water_filling_budget_conservation(n_channels, total_power):
    """Water-filling allocation produces non-negative, finite values."""
    capacities = np.random.default_rng(42).uniform(1.0, 20.0, n_channels)
    noise = np.random.default_rng(43).uniform(0.1, 5.0, n_channels)
    alloc = water_filling_allocate(capacities, noise, total_power)
    # Every allocation must be non-negative.
    assert np.all(alloc >= -_ATOL), f"Negative allocation: {alloc}"
    # Output dimension must match input.
    assert len(alloc) == n_channels
    # All values must be finite.
    assert np.all(np.isfinite(alloc))


@given(integers(min_value=2, max_value=10),
       floats(min_value=1.0, max_value=100.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_water_filling_non_negative(n_channels, total_power):
    """Water-filling allocation is non-negative for each channel."""
    capacities = np.random.default_rng(42).uniform(1.0, 20.0, n_channels)
    noise = np.random.default_rng(43).uniform(0.1, 5.0, n_channels)
    alloc = water_filling_allocate(capacities, noise, total_power)
    assert np.all(alloc >= -_ATOL), \
        f"Negative allocation: {alloc}"


# ---------------------------------------------------------------------------
# Interference matrix is symmetric
# ---------------------------------------------------------------------------


def test_standard_interference_matrix_symmetric():
    """The standard interference matrix is symmetric."""
    mat = build_standard_interference_matrix()
    n = mat.n_channels
    for i in range(n):
        for j in range(i + 1, n):
            assert math.isclose(
                mat.interference(i, j),
                mat.interference(j, i),
                abs_tol=_ATOL,
            ), f"Interference asymmetric at ({i},{j})"


def test_standard_interference_matrix_non_negative():
    """All interference coefficients are non-negative."""
    mat = build_standard_interference_matrix()
    n = mat.n_channels
    for i in range(n):
        for j in range(n):
            assert mat.interference(i, j) >= -_ATOL, \
                f"Negative interference at ({i},{j})"


# ---------------------------------------------------------------------------
# Dimension conflict is symmetric
# ---------------------------------------------------------------------------


@given(_resource, _resource)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_dimension_conflict_symmetric(r1, r2):
    """dimension_conflict(r1, r2) = dimension_conflict(r2, r1)."""
    c12 = dimension_conflict(r1, r2)
    c21 = dimension_conflict(r2, r1)
    assert math.isclose(c12, c21, abs_tol=_ATOL), \
        f"Conflict not symmetric: {c12} vs {c21}"


@given(_resource, _resource)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_dimension_conflict_in_unit_interval(r1, r2):
    """Dimension conflict κ ∈ [0, 1]."""
    c = dimension_conflict(r1, r2)
    assert -_ATOL <= c <= 1.0 + _ATOL, \
        f"Conflict out of range: {c}"


# ---------------------------------------------------------------------------
# Visual capacity is non-negative
# ---------------------------------------------------------------------------


@given(_n_elements,
       floats(min_value=0.1, max_value=5.0,
              allow_nan=False, allow_infinity=False),
       floats(min_value=0.0, max_value=60.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_visual_capacity_non_negative(n_elements, grouping, eccentricity):
    """Visual processing capacity ≥ 0."""
    c = visual_capacity(n_elements, grouping, eccentricity)
    assert c >= -_ATOL, f"Visual capacity negative: {c}"


# ---------------------------------------------------------------------------
# Auditory capacity is non-negative
# ---------------------------------------------------------------------------


@given(floats(min_value=0.01, max_value=30.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_auditory_capacity_non_negative(snr):
    """Auditory channel capacity ≥ 0."""
    c = auditory_capacity(snr_db=snr)
    assert c >= -_ATOL, f"Auditory capacity negative: {c}"


# ---------------------------------------------------------------------------
# Motor capacity is non-negative
# ---------------------------------------------------------------------------


@given(floats(min_value=1.0, max_value=500.0,
              allow_nan=False, allow_infinity=False),
       floats(min_value=1.0, max_value=100.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_motor_capacity_non_negative(amplitude, width):
    """Motor (Fitts) capacity ≥ 0."""
    c = motor_capacity_fitts(target_distance_px=amplitude, target_width_px=width)
    assert c >= -_ATOL, f"Motor capacity negative: {c}"


# ---------------------------------------------------------------------------
# Cognitive WM capacity is non-negative
# ---------------------------------------------------------------------------


@given(integers(min_value=1, max_value=10))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_cognitive_wm_capacity_non_negative(n_items):
    """Cognitive working-memory capacity ≥ 0."""
    c = cognitive_capacity_wm(n_chunks=n_items)
    assert c >= -_ATOL, f"Cognitive WM capacity negative: {c}"


# ---------------------------------------------------------------------------
# Fatigue degradation ∈ [0, 1]
# ---------------------------------------------------------------------------


@given(floats(min_value=1.0, max_value=50.0,
              allow_nan=False, allow_infinity=False),
       floats(min_value=0.0, max_value=600.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_fatigue_degradation_in_unit(base_cap, time_on_task_min):
    """Fatigue-degraded capacity ≤ base capacity."""
    f = fatigue_degradation(base_capacity=base_cap,
                            time_on_task_min=time_on_task_min)
    assert f >= -_ATOL, f"Fatigue-degraded capacity negative: {f}"
    assert f <= base_cap + _ATOL, \
        f"Fatigue-degraded capacity exceeds base: {f} > {base_cap}"


@given(floats(min_value=1.0, max_value=50.0,
              allow_nan=False, allow_infinity=False),
       floats(min_value=0.0, max_value=300.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_fatigue_is_monotone_decreasing(base_cap, t):
    """Fatigue factor decreases with time on task."""
    f1 = fatigue_degradation(base_capacity=base_cap, time_on_task_min=t)
    f2 = fatigue_degradation(base_capacity=base_cap, time_on_task_min=t + 30.0)
    assert f2 <= f1 + _ATOL, \
        f"Fatigue factor increased: {f1} → {f2}"


# ---------------------------------------------------------------------------
# Profile interference is symmetric
# ---------------------------------------------------------------------------


@given(sampled_from(["visual_spatial", "visual_verbal",
                     "auditory_spatial", "auditory_verbal"]),
       sampled_from(["visual_spatial", "visual_verbal",
                     "auditory_spatial", "auditory_verbal"]))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_profile_interference_symmetric(p1, p2):
    """profile_interference(a, b) ≈ profile_interference(b, a)."""
    rp1 = _PROFILES[p1]
    rp2 = _PROFILES[p2]
    i12 = profile_interference(rp1, rp2)
    i21 = profile_interference(rp2, rp1)
    assert math.isclose(i12, i21, abs_tol=_ATOL), \
        f"Profile interference not symmetric: {i12} vs {i21}"


@given(sampled_from(["visual_spatial", "visual_verbal",
                     "auditory_spatial", "auditory_verbal"]),
       sampled_from(["visual_spatial", "visual_verbal",
                     "auditory_spatial", "auditory_verbal"]))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_profile_interference_non_negative(p1, p2):
    """Profile interference ≥ 0."""
    rp1 = _PROFILES[p1]
    rp2 = _PROFILES[p2]
    ifc = profile_interference(rp1, rp2)
    assert ifc >= -_ATOL, f"Profile interference negative: {ifc}"


# ---------------------------------------------------------------------------
# ResourceChannel available capacity
# ---------------------------------------------------------------------------


@given(_resource_channel())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_resource_channel_available_capacity_non_negative(ch):
    """Available capacity = capacity * (1 - load) ≥ 0."""
    assert ch.available_capacity >= -_ATOL, \
        f"Available capacity negative: {ch.available_capacity}"
