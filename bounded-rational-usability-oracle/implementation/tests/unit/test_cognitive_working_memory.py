"""Unit tests for usability_oracle.cognitive.working_memory.WorkingMemoryModel.

Tests cover recall probability, interval-valued recall, chunk counting,
rehearsal cost, interference, load cost, proactive interference, composite
memory cost, monotonicity properties, capacity thresholds, and constants.

References
----------
Miller, G. A. (1956). The magical number seven, plus or minus two.
    *Psychological Review*, 63(2), 81-97.
Cowan, N. (2001). The magical number 4 in short-term memory.
    *Behav Brain Sci*, 24(1), 87-114.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.cognitive.working_memory import WorkingMemoryModel
from usability_oracle.interval.interval import Interval


# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #


class TestWorkingMemoryConstants:
    """Verify published capacity and decay constants."""

    def test_default_capacity(self) -> None:
        """DEFAULT_CAPACITY should be 4 (Cowan, 2001)."""
        assert WorkingMemoryModel.DEFAULT_CAPACITY == 4

    def test_min_capacity(self) -> None:
        """MIN_CAPACITY should be 3."""
        assert WorkingMemoryModel.MIN_CAPACITY == 3

    def test_max_capacity(self) -> None:
        """MAX_CAPACITY should be 7 (Miller, 1956)."""
        assert WorkingMemoryModel.MAX_CAPACITY == 7

    def test_default_decay_rate(self) -> None:
        """DEFAULT_DECAY_RATE ≈ ln(2)/9 ≈ 0.077 (Oberauer & Lewandowsky)."""
        assert WorkingMemoryModel.DEFAULT_DECAY_RATE == pytest.approx(0.077, abs=0.001)


# ------------------------------------------------------------------ #
# Recall probability
# ------------------------------------------------------------------ #


class TestRecallProbability:
    """Tests for WorkingMemoryModel.predict_recall_probability()."""

    def test_zero_delay_within_capacity(self) -> None:
        """With delay=0 and items <= capacity, recall should be ~1.0.

        exp(0) = 1.0 → probability = 1.0.
        """
        result = WorkingMemoryModel.predict_recall_probability(items=3, delay=0.0)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_decay_with_delay(self) -> None:
        """Non-zero delay should reduce recall probability.

        items=3 (within capacity), delay=9s →
        exp(-0.077 * 9) ≈ exp(-0.693) ≈ 0.5 (half-life ~9s).
        """
        result = WorkingMemoryModel.predict_recall_probability(items=3, delay=9.0)
        assert result == pytest.approx(0.5, abs=0.05)

    def test_over_capacity_reduces_recall(self) -> None:
        """Items > capacity introduces a capacity penalty.

        items=8, capacity=4, delay=0 → (4/8) * 1.0 = 0.5.
        """
        result = WorkingMemoryModel.predict_recall_probability(
            items=8, delay=0.0, capacity=4
        )
        assert result == pytest.approx(0.5, abs=1e-9)

    def test_recall_in_unit_interval(self) -> None:
        """Recall probability must lie in [0, 1] for all valid inputs."""
        for items in [1, 4, 10, 20]:
            for delay in [0, 5, 30, 120]:
                p = WorkingMemoryModel.predict_recall_probability(items, delay)
                assert 0.0 <= p <= 1.0, f"items={items}, delay={delay}, p={p}"

    def test_more_items_lower_recall(self) -> None:
        """More items → lower recall probability (monotonicity).

        With fixed delay, increasing items reduces the probability.
        """
        delay = 5.0
        probs = [
            WorkingMemoryModel.predict_recall_probability(n, delay)
            for n in [1, 2, 4, 8, 16]
        ]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], (
                f"Recall should decrease: items={2**i} p={probs[i]}, "
                f"items={2**(i+1)} p={probs[i+1]}"
            )

    def test_longer_delay_lower_recall(self) -> None:
        """Longer delay → lower recall probability (monotonicity).

        Memory traces decay exponentially with time.
        """
        items = 3
        probs = [
            WorkingMemoryModel.predict_recall_probability(items, d)
            for d in [0, 5, 10, 30, 60]
        ]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]

    def test_items_within_capacity_high_recall(self) -> None:
        """Items ≤ capacity at zero delay → perfect recall."""
        p = WorkingMemoryModel.predict_recall_probability(
            items=4, delay=0.0, capacity=4
        )
        assert p == pytest.approx(1.0, abs=1e-9)

    def test_items_over_capacity_low_recall(self) -> None:
        """Many items well over capacity → low recall."""
        p = WorkingMemoryModel.predict_recall_probability(
            items=20, delay=30.0, capacity=4
        )
        assert p < 0.1


# ------------------------------------------------------------------ #
# Interval-valued recall
# ------------------------------------------------------------------ #


class TestRecallInterval:
    """Tests for WorkingMemoryModel.predict_recall_interval()."""

    def test_point_interval_matches_scalar(self) -> None:
        """Degenerate delay interval should match scalar prediction."""
        p_scalar = WorkingMemoryModel.predict_recall_probability(3, 5.0)
        p_ivl = WorkingMemoryModel.predict_recall_interval(3, Interval(5.0, 5.0))
        assert p_ivl.low == pytest.approx(p_scalar, rel=1e-6)
        assert p_ivl.high == pytest.approx(p_scalar, rel=1e-6)

    def test_wider_delay_wider_recall(self) -> None:
        """Wider delay interval → wider recall probability interval."""
        narrow = WorkingMemoryModel.predict_recall_interval(
            3, Interval(4.5, 5.5)
        )
        wide = WorkingMemoryModel.predict_recall_interval(
            3, Interval(1.0, 10.0)
        )
        assert wide.width > narrow.width

    def test_monotone_inversion(self) -> None:
        """Lower delay bound → higher recall bound (monotone decreasing).

        Because recall decreases with delay, the interval bounds are swapped.
        """
        ivl = WorkingMemoryModel.predict_recall_interval(
            3, Interval(2.0, 10.0)
        )
        # high recall at low delay
        p_low_delay = WorkingMemoryModel.predict_recall_probability(3, 2.0)
        p_high_delay = WorkingMemoryModel.predict_recall_probability(3, 10.0)
        assert ivl.high == pytest.approx(p_low_delay, rel=1e-6)
        assert ivl.low == pytest.approx(p_high_delay, rel=1e-6)


# ------------------------------------------------------------------ #
# Chunk count
# ------------------------------------------------------------------ #


class TestChunkCount:
    """Tests for WorkingMemoryModel.chunk_count()."""

    def test_no_grouping(self) -> None:
        """With no groups, every element is its own chunk."""
        elements = list(range(5))
        result = WorkingMemoryModel.chunk_count(elements, [])
        assert result == 5

    def test_all_grouped(self) -> None:
        """Grouping all elements into one chunk → 1 chunk."""
        elements = list(range(4))
        result = WorkingMemoryModel.chunk_count(elements, [[0, 1, 2, 3]])
        assert result == 1

    def test_partial_grouping(self) -> None:
        """Some grouped, some ungrouped → mixed chunk count.

        5 elements, group [0,1] and [3,4] → 2 groups + 1 singleton = 3.
        """
        elements = list(range(5))
        result = WorkingMemoryModel.chunk_count(elements, [[0, 1], [3, 4]])
        assert result == 3


# ------------------------------------------------------------------ #
# Rehearsal cost
# ------------------------------------------------------------------ #


class TestRehearsalCost:
    """Tests for WorkingMemoryModel.rehearsal_cost()."""

    def test_sufficient_time_zero_cost(self) -> None:
        """If time_available >= items * articulation_rate → cost = 0."""
        result = WorkingMemoryModel.rehearsal_cost(items=3, time_available=2.0)
        assert result == pytest.approx(0.0)

    def test_insufficient_time_positive_cost(self) -> None:
        """Insufficient rehearsal time → positive effective delay."""
        result = WorkingMemoryModel.rehearsal_cost(items=10, time_available=1.0)
        assert result > 0.0

    def test_zero_time_maximum_cost(self) -> None:
        """Zero available time → maximum rehearsal cost."""
        result = WorkingMemoryModel.rehearsal_cost(items=5, time_available=0.0)
        assert result > 0.0


# ------------------------------------------------------------------ #
# Interference factor
# ------------------------------------------------------------------ #


class TestInterferenceFactor:
    """Tests for WorkingMemoryModel.interference_factor()."""

    def test_no_similar_items(self) -> None:
        """No similar items → interference factor = 1.0 (no penalty)."""
        result = WorkingMemoryModel.interference_factor(0, 5)
        assert result == pytest.approx(1.0)

    def test_all_similar(self) -> None:
        """All items similar → factor = 1.0 + 0.5 * 1.0 = 1.5."""
        result = WorkingMemoryModel.interference_factor(5, 5)
        assert result == pytest.approx(1.5)

    def test_partial_similarity(self) -> None:
        """Half similar → factor = 1.0 + 0.5 * 0.5 = 1.25."""
        result = WorkingMemoryModel.interference_factor(3, 6)
        assert result == pytest.approx(1.25)

    def test_factor_at_least_one(self) -> None:
        """Interference factor should always be >= 1.0."""
        for sim in range(10):
            f = WorkingMemoryModel.interference_factor(sim, 10)
            assert f >= 1.0


# ------------------------------------------------------------------ #
# Load cost
# ------------------------------------------------------------------ #


class TestLoadCost:
    """Tests for WorkingMemoryModel.load_cost()."""

    def test_zero_load(self) -> None:
        """Zero items in memory → cost = 1.0 (baseline)."""
        result = WorkingMemoryModel.load_cost(0, capacity=4)
        assert result == pytest.approx(1.0)

    def test_at_capacity_saturated(self) -> None:
        """Load == capacity → returns 10.0 (severe degradation)."""
        result = WorkingMemoryModel.load_cost(4, capacity=4)
        assert result == pytest.approx(10.0)

    def test_over_capacity_saturated(self) -> None:
        """Load > capacity → also returns 10.0."""
        result = WorkingMemoryModel.load_cost(8, capacity=4)
        assert result == pytest.approx(10.0)

    def test_increasing_load_increases_cost(self) -> None:
        """Cost should increase as load approaches capacity."""
        costs = [WorkingMemoryModel.load_cost(i, 4) for i in range(5)]
        for i in range(len(costs) - 1):
            assert costs[i] <= costs[i + 1]


# ------------------------------------------------------------------ #
# Proactive interference
# ------------------------------------------------------------------ #


class TestProactiveInterference:
    """Tests for WorkingMemoryModel.proactive_interference()."""

    def test_no_prior_items(self) -> None:
        """No prior items → zero proactive interference."""
        result = WorkingMemoryModel.proactive_interference(0, delay=5.0)
        assert result == pytest.approx(0.0)

    def test_decays_with_delay(self) -> None:
        """PI should decrease with longer delay (release from PI)."""
        pi_short = WorkingMemoryModel.proactive_interference(5, delay=1.0)
        pi_long = WorkingMemoryModel.proactive_interference(5, delay=30.0)
        assert pi_long < pi_short

    def test_pi_formula(self) -> None:
        """PI = prior_items * exp(-release_rate * delay).

        5 items, 10s delay, rate=0.1 → 5 * exp(-1.0) ≈ 1.839.
        """
        expected = 5 * math.exp(-0.1 * 10)
        result = WorkingMemoryModel.proactive_interference(5, 10.0)
        assert result == pytest.approx(expected, rel=1e-6)


# ------------------------------------------------------------------ #
# Total memory cost
# ------------------------------------------------------------------ #


class TestTotalMemoryCost:
    """Tests for WorkingMemoryModel.total_memory_cost()."""

    def test_minimal_load_low_cost(self) -> None:
        """1 item, no delay, no interference → low total cost."""
        cost = WorkingMemoryModel.total_memory_cost(
            items=1, delay=0.0, similar_items=0, prior_items=0
        )
        assert cost >= 0.0
        assert cost < 1.0

    def test_heavy_load_high_cost(self) -> None:
        """Many items, long delay, interference → high total cost."""
        cost = WorkingMemoryModel.total_memory_cost(
            items=10, delay=30.0, similar_items=8, prior_items=5
        )
        assert cost > 5.0

    def test_non_negative(self) -> None:
        """Total memory cost should always be non-negative."""
        for items in [1, 4, 8]:
            for delay in [0, 5, 30]:
                c = WorkingMemoryModel.total_memory_cost(items, delay)
                assert c >= 0.0

    def test_cost_increases_with_items(self) -> None:
        """More items → higher total memory cost."""
        costs = [
            WorkingMemoryModel.total_memory_cost(n, delay=5.0)
            for n in [1, 2, 4, 8]
        ]
        for i in range(len(costs) - 1):
            assert costs[i] <= costs[i + 1]
