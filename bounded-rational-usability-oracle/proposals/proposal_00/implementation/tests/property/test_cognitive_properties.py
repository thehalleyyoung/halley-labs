"""Property-based tests for cognitive law models.

This module verifies monotonicity, positivity, and range constraints of
FittsLaw, HickHymanLaw, VisualSearchModel, and WorkingMemoryModel using
Hypothesis. These tests ensure the cognitive models satisfy the fundamental
psychophysical properties that justify their use in usability cost prediction.
"""

import math

from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import (
    floats, integers, lists, tuples, sampled_from,
)

from usability_oracle.cognitive.fitts import FittsLaw
from usability_oracle.cognitive.hick import HickHymanLaw
from usability_oracle.cognitive.visual_search import VisualSearchModel
from usability_oracle.cognitive.working_memory import WorkingMemoryModel


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_pos_distance = floats(min_value=1.0, max_value=1000.0,
                       allow_nan=False, allow_infinity=False)

_pos_width = floats(min_value=0.5, max_value=500.0,
                    allow_nan=False, allow_infinity=False)

_pos_a = floats(min_value=0.01, max_value=1.0,
                allow_nan=False, allow_infinity=False)

_pos_b = floats(min_value=0.01, max_value=1.0,
                allow_nan=False, allow_infinity=False)

_n_items = integers(min_value=1, max_value=50)

_n_alternatives = integers(min_value=2, max_value=100)

_delay = floats(min_value=0.0, max_value=60.0,
                allow_nan=False, allow_infinity=False)

_capacity = integers(min_value=3, max_value=7)

_decay = floats(min_value=0.01, max_value=0.5,
                allow_nan=False, allow_infinity=False)


# =========================================================================
# Fitts' Law
# =========================================================================

# ---------------------------------------------------------------------------
# Monotonicity: distance ↑ → time ↑
# ---------------------------------------------------------------------------


@given(_pos_width, _pos_a, _pos_b)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_fitts_distance_monotonicity(width, a, b):
    """Increasing distance increases Fitts' law movement time."""
    t1 = FittsLaw.predict(10.0, width, a, b)
    t2 = FittsLaw.predict(100.0, width, a, b)
    assert t2 >= t1 - 1e-9


# ---------------------------------------------------------------------------
# Monotonicity: width ↑ → time ↓
# ---------------------------------------------------------------------------

@given(_pos_distance, _pos_a, _pos_b)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_fitts_width_monotonicity(distance, a, b):
    """Increasing target width decreases Fitts' law movement time."""
    t_narrow = FittsLaw.predict(distance, 5.0, a, b)
    t_wide = FittsLaw.predict(distance, 50.0, a, b)
    assert t_wide <= t_narrow + 1e-9


# ---------------------------------------------------------------------------
# Positivity
# ---------------------------------------------------------------------------

@given(_pos_distance, _pos_width, _pos_a, _pos_b)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_fitts_positivity(distance, width, a, b):
    """Fitts' law always predicts a positive movement time."""
    t = FittsLaw.predict(distance, width, a, b)
    assert t > 0.0


# ---------------------------------------------------------------------------
# Index of difficulty
# ---------------------------------------------------------------------------

@given(_pos_distance, _pos_width)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_fitts_id_positive(distance, width):
    """Index of difficulty is always positive for valid inputs."""
    id_val = FittsLaw.index_of_difficulty(distance, width)
    assert id_val > 0.0


@given(_pos_width)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_fitts_id_increases_with_distance(width):
    """Index of difficulty increases with distance."""
    id1 = FittsLaw.index_of_difficulty(10.0, width)
    id2 = FittsLaw.index_of_difficulty(100.0, width)
    assert id2 >= id1


# ---------------------------------------------------------------------------
# Throughput positivity
# ---------------------------------------------------------------------------

@given(_pos_distance, _pos_width,
       floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_fitts_throughput_positive(distance, width, mt):
    """Fitts' law throughput is positive: TP = ID / MT > 0."""
    tp = FittsLaw.throughput(distance, width, mt)
    assert tp > 0.0


# =========================================================================
# Hick-Hyman Law
# =========================================================================

# ---------------------------------------------------------------------------
# Monotonicity: n_alternatives ↑ → time ↑
# ---------------------------------------------------------------------------

@given(_pos_a, _pos_b)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_hick_monotonicity(a, b):
    """More alternatives increase Hick-Hyman choice reaction time."""
    t2 = HickHymanLaw.predict(2, a, b)
    t8 = HickHymanLaw.predict(8, a, b)
    assert t8 >= t2 - 1e-9


# ---------------------------------------------------------------------------
# Positivity
# ---------------------------------------------------------------------------

@given(_n_alternatives, _pos_a, _pos_b)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_hick_positivity(n, a, b):
    """Hick-Hyman law always predicts positive reaction time."""
    t = HickHymanLaw.predict(n, a, b)
    assert t > 0.0


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

@given(_n_alternatives)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_hick_entropy_positive(n):
    """Shannon entropy of equiprobable items is positive for n >= 2."""
    probs = [1.0 / n] * n
    h = HickHymanLaw.entropy(probs)
    assert h > 0.0


@given(_n_alternatives)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_hick_effective_alternatives_equals_n(n):
    """For equiprobable items, effective_alternatives == n."""
    probs = [1.0 / n] * n
    eff = HickHymanLaw.effective_alternatives(probs)
    assert math.isclose(eff, float(n), rel_tol=0.01)


# ---------------------------------------------------------------------------
# Unequal probabilities
# ---------------------------------------------------------------------------

@given(integers(min_value=1, max_value=1000),
       floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_hick_practice_factor_in_unit(trials, lr):
    """Practice factor is in (0, 1] — practice can only help."""
    f = HickHymanLaw.practice_factor(trials, lr)
    assert 0.0 < f <= 1.0 + 1e-9


# =========================================================================
# Visual Search Model
# =========================================================================

# ---------------------------------------------------------------------------
# Serial search: n_items ↑ → time ↑
# ---------------------------------------------------------------------------

@given(floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_visual_search_serial_monotonicity(slope):
    """More items increase serial visual search time."""
    t5 = VisualSearchModel.predict_serial(5, target_present=True, slope=slope)
    t20 = VisualSearchModel.predict_serial(20, target_present=True, slope=slope)
    assert t20 >= t5 - 1e-9


# ---------------------------------------------------------------------------
# Serial search: positivity
# ---------------------------------------------------------------------------

@given(_n_items)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_visual_search_serial_positive(n):
    """Serial visual search time is always positive."""
    t = VisualSearchModel.predict_serial(
        n, target_present=True,
        slope=VisualSearchModel.INEFFICIENT_SLOPE)
    assert t > 0.0


@given(_n_items)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_visual_search_parallel_positive(n):
    """Parallel (feature/pop-out) search time is positive."""
    t = VisualSearchModel.predict_parallel(n, base_rt=VisualSearchModel.BASE_RT)
    assert t > 0.0


# ---------------------------------------------------------------------------
# Guided search: monotonicity
# ---------------------------------------------------------------------------

@given(floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False),
       floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_visual_search_guided_monotonicity(slope, guidance):
    """Guided search time increases with set size."""
    t5 = VisualSearchModel.predict_guided(5, guidance, slope)
    t30 = VisualSearchModel.predict_guided(30, guidance, slope)
    assert t30 >= t5 - 1e-9


# ---------------------------------------------------------------------------
# Search time distribution
# ---------------------------------------------------------------------------

@given(_n_items,
       floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_search_time_distribution_positive(n, slope):
    """Mean and variance of search time are non-negative."""
    mean, var = VisualSearchModel.search_time_distribution(
        n, slope, target_present=True)
    assert mean >= 0.0
    assert var >= 0.0


@given(_n_items)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_fixations_positive(n):
    """Expected number of fixations is at least 1."""
    f = VisualSearchModel.number_of_fixations(n, target_present=True)
    assert f >= 1.0 - 1e-9


# =========================================================================
# Working Memory Model
# =========================================================================

# ---------------------------------------------------------------------------
# Recall probability in [0, 1]
# ---------------------------------------------------------------------------

@given(integers(min_value=1, max_value=15), _delay, _capacity, _decay)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_recall_probability_range(items, delay, cap, decay):
    """Recall probability is always in [0, 1]."""
    p = WorkingMemoryModel.predict_recall_probability(items, delay, cap, decay)
    assert -1e-9 <= p <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# More items → lower recall
# ---------------------------------------------------------------------------

@given(_delay, _capacity, _decay)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_recall_decreases_with_items(delay, cap, decay):
    """More items in working memory reduce recall probability."""
    p_few = WorkingMemoryModel.predict_recall_probability(2, delay, cap, decay)
    p_many = WorkingMemoryModel.predict_recall_probability(10, delay, cap, decay)
    assert p_many <= p_few + 1e-6


# ---------------------------------------------------------------------------
# More delay → lower recall
# ---------------------------------------------------------------------------

@given(integers(min_value=1, max_value=7), _capacity, _decay)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_recall_decreases_with_delay(items, cap, decay):
    """Longer delay reduces recall probability."""
    p_short = WorkingMemoryModel.predict_recall_probability(items, 0.0, cap, decay)
    p_long = WorkingMemoryModel.predict_recall_probability(items, 30.0, cap, decay)
    assert p_long <= p_short + 1e-6


# ---------------------------------------------------------------------------
# Load cost
# ---------------------------------------------------------------------------

@given(integers(min_value=1, max_value=10), _capacity)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_load_cost_non_negative(load, cap):
    """Working memory load cost is non-negative."""
    cost = WorkingMemoryModel.load_cost(load, cap)
    assert cost >= -1e-9


@given(_capacity)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_load_cost_increases_with_load(cap):
    """Load cost increases as current load approaches capacity."""
    c1 = WorkingMemoryModel.load_cost(1, cap)
    c2 = WorkingMemoryModel.load_cost(cap, cap)
    assert c2 >= c1 - 1e-9


# ---------------------------------------------------------------------------
# Interference factor
# ---------------------------------------------------------------------------

@given(integers(min_value=0, max_value=10),
       integers(min_value=1, max_value=15))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_interference_factor_geq_one(similar, total):
    """Interference factor is >= 1 (interference can only hurt)."""
    assume(similar <= total)
    f = WorkingMemoryModel.interference_factor(similar, total)
    assert f >= 1.0 - 1e-9


@given(integers(min_value=1, max_value=10),
       floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
       floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_rehearsal_cost_non_negative(items, time_avail, art_rate):
    """Rehearsal cost is non-negative."""
    cost = WorkingMemoryModel.rehearsal_cost(items, time_avail, art_rate)
    assert cost >= -1e-9


# ---------------------------------------------------------------------------
# All times are non-negative (comprehensive)
# ---------------------------------------------------------------------------

@given(_n_items)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_all_visual_search_times_positive(n):
    """Both serial and parallel search models yield positive times."""
    ts = VisualSearchModel.predict_serial(
        n, True, VisualSearchModel.INEFFICIENT_SLOPE)
    tp = VisualSearchModel.predict_parallel(
        n, VisualSearchModel.BASE_RT)
    assert ts > 0.0
    assert tp > 0.0
