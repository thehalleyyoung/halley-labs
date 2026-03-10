"""Property-based tests for the resources module (Wickens' MRT).

Verifies properties of interference matrices, time-sharing efficiency,
and demand vectors using the Hypothesis library.
"""

from __future__ import annotations

import math

from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import (
    floats,
    integers,
    sampled_from,
    composite,
)

from usability_oracle.resources.types import (
    DemandVector,
    Resource,
    ResourceDemand,
    ProcessingStage,
    PerceptualModality,
    ProcessingCode,
    VisualChannel,
)
from usability_oracle.resources import WickensModel, ResourceConflictMatrix

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_demand_level = floats(min_value=0.01, max_value=1.0,
                       allow_nan=False, allow_infinity=False)

_stages = sampled_from(list(ProcessingStage))
_modalities = sampled_from(list(PerceptualModality))
_codes = sampled_from(list(ProcessingCode))
_channels = sampled_from(list(VisualChannel))


@composite
def _resource_demand(draw):
    """Generate a single ResourceDemand with random parameters."""
    stage = draw(_stages)
    modality = draw(_modalities)
    code = draw(_codes)
    channel = draw(_channels)
    level = draw(_demand_level)
    resource = Resource(
        stage=stage, modality=modality,
        code=code, visual_channel=channel,
    )
    return ResourceDemand(resource=resource, demand_level=level)


@composite
def _demand_vector(draw, min_demands=1, max_demands=3):
    """Generate a DemandVector with 1-3 resource demands."""
    n = draw(integers(min_value=min_demands, max_value=max_demands))
    demands = []
    for _ in range(n):
        demands.append(draw(_resource_demand()))
    return DemandVector(demands=tuple(demands))


_ATOL = 1e-6

# ---------------------------------------------------------------------------
# Interference is symmetric
# ---------------------------------------------------------------------------


@given(_resource_demand(), _resource_demand())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_interference_symmetric(rd_a, rd_b):
    """Interference(a, b) = Interference(b, a)."""
    model = WickensModel()
    i_ab = model.compute_interference(rd_a, rd_b)
    i_ba = model.compute_interference(rd_b, rd_a)
    assert math.isclose(i_ab, i_ba, abs_tol=_ATOL), \
        f"Interference not symmetric: {i_ab} vs {i_ba}"


# ---------------------------------------------------------------------------
# Self-interference is 0 (or at least non-negative)
# ---------------------------------------------------------------------------


@given(_resource_demand())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_self_interference_non_negative(rd):
    """Self-interference should be non-negative (and typically maximal)."""
    model = WickensModel()
    self_i = model.compute_interference(rd, rd)
    assert self_i >= -_ATOL, f"Self-interference is negative: {self_i}"


# ---------------------------------------------------------------------------
# Time-sharing efficiency in [0, 1]
# ---------------------------------------------------------------------------


@given(_demand_vector(), _demand_vector())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_time_sharing_efficiency_in_unit_interval(dv_a, dv_b):
    """Time-sharing efficiency η ∈ [0, 1]."""
    model = WickensModel()
    eff = model.compute_time_sharing_efficiency([dv_a, dv_b])
    assert 0.0 - _ATOL <= eff <= 1.0 + _ATOL, \
        f"Time-sharing efficiency out of range: {eff}"


# ---------------------------------------------------------------------------
# Single task efficiency = 1.0
# ---------------------------------------------------------------------------


@given(_demand_vector())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_single_task_efficiency_is_one(dv):
    """A single task should have time-sharing efficiency = 1.0."""
    model = WickensModel()
    eff = model.compute_time_sharing_efficiency([dv])
    assert math.isclose(eff, 1.0, abs_tol=_ATOL), \
        f"Single task efficiency should be 1.0, got {eff}"


# ---------------------------------------------------------------------------
# Adding demand never decreases total interference
# ---------------------------------------------------------------------------


@given(_demand_vector(), _demand_vector(), _demand_vector())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_adding_demand_increases_interference(dv_a, dv_b, dv_c):
    """Total interference with 3 tasks ≥ total interference with 2 tasks."""
    model = WickensModel()
    int_2 = model.total_interference([dv_a, dv_b])
    int_3 = model.total_interference([dv_a, dv_b, dv_c])
    assert int_3 >= int_2 - _ATOL, \
        f"Adding demand should not decrease interference: {int_2} vs {int_3}"


# ---------------------------------------------------------------------------
# Total interference is non-negative
# ---------------------------------------------------------------------------


@given(_demand_vector(), _demand_vector())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_total_interference_non_negative(dv_a, dv_b):
    """Total interference across tasks is non-negative."""
    model = WickensModel()
    total = model.total_interference([dv_a, dv_b])
    assert total >= -_ATOL, f"Total interference is negative: {total}"


# ---------------------------------------------------------------------------
# Interference matrix is consistent
# ---------------------------------------------------------------------------


@given(_demand_vector(), _demand_vector())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_interference_matrix_non_negative_entries(dv_a, dv_b):
    """All entries in the interference matrix are non-negative."""
    model = WickensModel()
    mat = model.compute_matrix([dv_a, dv_b])
    assert mat.max_interference >= -_ATOL, \
        f"Max interference is negative: {mat.max_interference}"


# ---------------------------------------------------------------------------
# ResourceConflictMatrix marginal conflict
# ---------------------------------------------------------------------------


@given(_demand_vector(), _demand_vector(), _demand_vector())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_marginal_conflict_non_negative(dv_a, dv_b, dv_new):
    """Marginal conflict of adding a new task is non-negative."""
    rcm = ResourceConflictMatrix()
    mc = rcm.compute_marginal_conflict(dv_new, [dv_a, dv_b])
    assert mc >= -_ATOL, f"Marginal conflict is negative: {mc}"


# ---------------------------------------------------------------------------
# Efficiency decreases with more tasks (more interference)
# ---------------------------------------------------------------------------


@given(_demand_vector(), _demand_vector(), _demand_vector())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_efficiency_decreases_with_more_tasks(dv_a, dv_b, dv_c):
    """Adding a third task should not increase time-sharing efficiency."""
    model = WickensModel()
    eff_2 = model.compute_time_sharing_efficiency([dv_a, dv_b])
    eff_3 = model.compute_time_sharing_efficiency([dv_a, dv_b, dv_c])
    assert eff_3 <= eff_2 + _ATOL, \
        f"More tasks should not improve efficiency: {eff_2} vs {eff_3}"
