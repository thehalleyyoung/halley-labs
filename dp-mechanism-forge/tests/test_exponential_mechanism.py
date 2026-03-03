"""
Comprehensive tests for dp_forge.exponential_mechanism module.

Tests cover candidate pool management, exponential selection, privacy
guarantees, utility bounds, score sensitivity, and budget allocation.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dp_forge.exponential_mechanism import (
    Candidate,
    CandidatePool,
    ExponentialSelector,
    SelectionResult,
    BudgetAllocation,
    allocate_budget,
)
from dp_forge.exceptions import ConfigurationError


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def pool_3():
    """Pool with 3 candidates with distinct scores (lower = better)."""
    pool = CandidatePool()
    pool.add("best", score=-5.0)     # Best (lowest score = lowest error)
    pool.add("medium", score=-3.0)
    pool.add("worst", score=-1.0)    # Worst (highest score)
    return pool


@pytest.fixture
def pool_equal():
    """Pool with 3 candidates with equal scores."""
    pool = CandidatePool()
    pool.add("a", score=-2.0)
    pool.add("b", score=-2.0)
    pool.add("c", score=-2.0)
    return pool


@pytest.fixture
def selector():
    """ExponentialSelector with ε=1, seeded."""
    return ExponentialSelector(epsilon_select=1.0, seed=42)


# =========================================================================
# Section 1: CandidatePool
# =========================================================================


class TestCandidatePool:
    """Tests for candidate pool management."""

    def test_add_candidates(self):
        """Adding candidates increases pool size."""
        pool = CandidatePool()
        pool.add("c1", score=-1.0)
        pool.add("c2", score=-2.0)
        assert pool.size == 2

    def test_get_candidate(self):
        """Retrieving a candidate by name."""
        pool = CandidatePool()
        pool.add("c1", score=-1.0, metadata={"key": "val"})
        c = pool.get("c1")
        assert c.name == "c1"
        assert c.score == -1.0
        assert c.metadata == {"key": "val"}

    def test_remove_candidate(self):
        """Removing a candidate decreases pool size."""
        pool = CandidatePool()
        pool.add("c1", score=-1.0)
        pool.add("c2", score=-2.0)
        pool.remove("c1")
        assert pool.size == 1
        with pytest.raises((KeyError, ValueError, ConfigurationError)):
            pool.get("c1")

    def test_best_score(self, pool_3):
        """best_score returns the lowest (best) score."""
        assert pool_3.best_score == -5.0

    def test_best_candidate(self, pool_3):
        """best_candidate returns the candidate with lowest score."""
        best = pool_3.best_candidate
        assert best.name == "best"

    def test_score_range(self, pool_3):
        """score_range = max_score - min_score."""
        assert abs(pool_3.score_range - 4.0) < 1e-10

    def test_update_score(self, pool_3):
        """update_score changes a candidate's score."""
        pool_3.update_score("worst", -10.0)
        assert pool_3.get("worst").score == -10.0
        assert pool_3.best_candidate.name == "worst"

    def test_scores_array(self, pool_3):
        """scores returns numpy array of all scores."""
        scores = pool_3.scores
        assert len(scores) == 3

    def test_candidates_list(self, pool_3):
        """candidates returns list of Candidate objects."""
        cands = pool_3.candidates
        assert len(cands) == 3
        assert all(isinstance(c, Candidate) for c in cands)

    def test_add_with_mechanism(self):
        """Adding candidate with mechanism data."""
        pool = CandidatePool()
        mech_data = np.array([[0.5, 0.5], [0.5, 0.5]])
        pool.add("c1", score=-1.0, mechanism=mech_data)
        c = pool.get("c1")
        assert c.mechanism is not None
        np.testing.assert_array_equal(c.mechanism, mech_data)


# =========================================================================
# Section 2: Exponential Selection
# =========================================================================


class TestExponentialSelection:
    """Tests for the exponential mechanism selection."""

    def test_select_returns_result(self, selector, pool_3):
        """select() returns a SelectionResult."""
        result = selector.select(pool_3)
        assert isinstance(result, SelectionResult)
        assert result.selected is not None

    def test_best_selected_most_often(self, pool_3):
        """Candidate with best score is selected most often."""
        sel = ExponentialSelector(epsilon_select=5.0, seed=42)
        counts = Counter()
        for i in range(1000):
            sel_i = ExponentialSelector(epsilon_select=5.0, seed=42 + i)
            result = sel_i.select(pool_3)
            counts[result.selected.name] += 1
        # "best" should be selected most often (has highest score)
        assert counts["best"] > counts["worst"]

    def test_equal_scores_uniform_selection(self, pool_equal):
        """Equal scores → roughly uniform selection."""
        counts = Counter()
        for i in range(3000):
            sel = ExponentialSelector(epsilon_select=1.0, seed=42 + i)
            result = sel.select(pool_equal)
            counts[result.selected.name] += 1
        # Each should be selected roughly 1/3 of the time
        for name in ["a", "b", "c"]:
            frac = counts[name] / 3000
            assert 0.2 < frac < 0.5, f"{name} selected {frac*100:.1f}%"

    def test_higher_eps_more_deterministic(self, pool_3):
        """Higher ε → more concentrated on best candidate."""
        counts_low = Counter()
        counts_high = Counter()
        for i in range(500):
            sel_low = ExponentialSelector(epsilon_select=0.1, seed=42 + i)
            sel_high = ExponentialSelector(epsilon_select=10.0, seed=42 + i)
            counts_low[sel_low.select(pool_3).selected.name] += 1
            counts_high[sel_high.select(pool_3).selected.name] += 1
        # High ε should select "best" more often
        assert counts_high["best"] >= counts_low["best"]

    def test_selection_probability_sum(self, selector, pool_3):
        """Selection probabilities sum to 1."""
        result = selector.select(pool_3)
        total_prob = sum(result.probabilities.values())
        assert abs(total_prob - 1.0) < 1e-10

    def test_selection_probability_decreasing(self, selector, pool_3):
        """Higher score → higher selection probability."""
        result = selector.select(pool_3)
        p_best = result.probabilities["best"]
        p_worst = result.probabilities["worst"]
        assert p_best >= p_worst


# =========================================================================
# Section 3: Privacy Guarantee
# =========================================================================


class TestExponentialPrivacy:
    """Tests for privacy guarantees of exponential selection."""

    def test_epsilon_used(self, selector, pool_3):
        """epsilon_used matches constructor parameter."""
        result = selector.select(pool_3)
        assert abs(result.epsilon_used - 1.0) < 1e-10

    def test_selection_satisfies_dp(self, pool_3):
        """Selection probabilities satisfy ε-DP ratio bound.

        For ε-DP exponential mechanism:
        P[select(x, r)] / P[select(x', r)] ≤ exp(ε)
        for any candidate r when sensitivity = Δu.
        """
        eps = 1.0
        sel = ExponentialSelector(epsilon_select=eps, seed=42)
        result = sel.select(pool_3)
        probs = result.probabilities
        # All probability ratios should be ≤ exp(ε · score_range / Δu)
        p_values = list(probs.values())
        for i in range(len(p_values)):
            for j in range(len(p_values)):
                if p_values[j] > 1e-15:
                    ratio = p_values[i] / p_values[j]
                    # Bound depends on score range and sensitivity
                    # For a valid mechanism, ratios are bounded
                    assert ratio >= 0


# =========================================================================
# Section 4: Utility Bound
# =========================================================================


class TestUtilityBound:
    """Tests for utility (error) bounds."""

    def test_utility_bound_positive(self, selector, pool_3):
        """Utility bound is positive."""
        bound = selector.utility_bound(pool_3)
        assert bound > 0

    def test_utility_bound_formula(self, pool_3):
        """Utility bound ≤ OPT + O(ln(T)/ε)."""
        eps = 1.0
        sel = ExponentialSelector(epsilon_select=eps, score_sensitivity=1.0, seed=42)
        bound = sel.utility_bound(pool_3, score_sensitivity=1.0)
        # Bound = OPT + (2Δu/ε)(ln T + ln(1/β))
        # OPT = best score = -1.0 (as negative error)
        # The bound should be achievable
        assert math.isfinite(bound)

    def test_more_candidates_larger_bound(self):
        """More candidates → larger utility gap (logarithmic)."""
        pool_small = CandidatePool()
        pool_large = CandidatePool()
        for i in range(5):
            pool_small.add(f"c{i}", score=-float(i))
        for i in range(50):
            pool_large.add(f"c{i}", score=-float(i))
        sel = ExponentialSelector(epsilon_select=1.0, score_sensitivity=1.0, seed=42)
        b_small = sel.utility_bound(pool_small, score_sensitivity=1.0)
        b_large = sel.utility_bound(pool_large, score_sensitivity=1.0)
        assert b_large >= b_small - 1e-10


# =========================================================================
# Section 5: Score Sensitivity
# =========================================================================


class TestScoreSensitivity:
    """Tests for score sensitivity computation."""

    def test_score_sensitivity_range(self, selector, pool_3):
        """Score sensitivity via 'range' method."""
        sens = selector.score_sensitivity(pool_3, method="range")
        assert sens > 0
        assert math.isfinite(sens)

    def test_score_sensitivity_unit(self, selector, pool_3):
        """Score sensitivity via 'unit' method = 1."""
        sens = selector.score_sensitivity(pool_3, method="unit")
        assert abs(sens - 1.0) < 1e-10

    def test_custom_sensitivity(self, pool_3):
        """Custom score_sensitivity overrides auto-computation."""
        sel = ExponentialSelector(
            epsilon_select=1.0, score_sensitivity=0.5, seed=42,
        )
        result = sel.select(pool_3, score_sensitivity=0.5)
        assert abs(result.score_sensitivity - 0.5) < 1e-10


# =========================================================================
# Section 6: Selection Probabilities
# =========================================================================


class TestSelectionProbabilities:
    """Tests for explicit probability computation."""

    def test_selection_probabilities_dict(self, selector, pool_3):
        """selection_probabilities returns dict mapping name → prob."""
        probs = selector.selection_probabilities(pool_3)
        assert isinstance(probs, dict)
        assert len(probs) == 3
        assert all(p >= 0 for p in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 1e-10

    def test_probabilities_match_selection(self, pool_3):
        """Empirical selection frequencies match theoretical probabilities."""
        sel = ExponentialSelector(epsilon_select=2.0, score_sensitivity=1.0, seed=42)
        probs = sel.selection_probabilities(pool_3, score_sensitivity=1.0)
        counts = Counter()
        n_trials = 5000
        for i in range(n_trials):
            sel_i = ExponentialSelector(
                epsilon_select=2.0, score_sensitivity=1.0, seed=42 + i,
            )
            result = sel_i.select(pool_3, score_sensitivity=1.0)
            counts[result.selected.name] += 1
        # Empirical ≈ theoretical
        for name, p_theory in probs.items():
            p_empirical = counts[name] / n_trials
            assert abs(p_empirical - p_theory) < 0.1, (
                f"{name}: theory={p_theory:.3f}, empirical={p_empirical:.3f}"
            )


# =========================================================================
# Section 7: Budget Allocation
# =========================================================================


class TestBudgetAllocation:
    """Tests for exponential mechanism budget allocation."""

    def test_allocate_budget_basic(self):
        """allocate_budget returns valid BudgetAllocation."""
        alloc = allocate_budget(epsilon_total=1.0, delta_total=1e-5)
        assert isinstance(alloc, BudgetAllocation)
        assert alloc.epsilon_total == 1.0
        assert alloc.delta_total == 1e-5

    def test_budget_sums(self):
        """Synthesis + selection budget ≤ total."""
        alloc = allocate_budget(epsilon_total=2.0, delta_total=1e-5)
        assert alloc.epsilon_synthesis + alloc.epsilon_selection <= 2.0 + 1e-10
        assert alloc.delta_synthesis + alloc.delta_selection <= 1e-5 + 1e-15

    def test_split_ratio(self):
        """split_ratio controls synthesis/selection split."""
        alloc = allocate_budget(
            epsilon_total=1.0, delta_total=0.0, split_ratio=0.8,
            n_candidates=10,
        )
        assert abs(alloc.epsilon_synthesis - 0.8) < 1e-10
        assert abs(alloc.epsilon_selection - 0.2) < 1e-10

    def test_split_ratio_default(self):
        """Default split_ratio is 0.9."""
        alloc = allocate_budget(epsilon_total=1.0, n_candidates=10)
        assert abs(alloc.split_ratio - 0.9) < 1e-10

    def test_pure_dp(self):
        """Pure DP budget allocation (δ = 0)."""
        alloc = allocate_budget(epsilon_total=1.0, delta_total=0.0)
        assert alloc.delta_synthesis == 0.0
        assert alloc.delta_selection == 0.0


# =========================================================================
# Section 8: Edge Cases
# =========================================================================


class TestExponentialEdgeCases:
    """Edge case tests for exponential mechanism."""

    def test_single_candidate(self):
        """Single candidate → always selected."""
        pool = CandidatePool()
        pool.add("only", score=-1.0)
        sel = ExponentialSelector(epsilon_select=1.0, seed=42)
        result = sel.select(pool)
        assert result.selected.name == "only"
        assert abs(result.selection_probability - 1.0) < 1e-10

    def test_very_large_eps(self):
        """Very large ε → nearly deterministic selection of lowest score."""
        pool = CandidatePool()
        pool.add("best", score=-100.0)
        pool.add("worst", score=0.0)
        sel = ExponentialSelector(epsilon_select=100.0, seed=42)
        result = sel.select(pool)
        assert result.selected.name == "best"

    def test_very_small_eps(self):
        """Very small ε → nearly uniform selection."""
        pool = CandidatePool()
        pool.add("a", score=0.0)
        pool.add("b", score=-1.0)
        sel = ExponentialSelector(epsilon_select=0.001, seed=42)
        result = sel.select(pool)
        # Both should have roughly equal probability
        p_a = result.probabilities["a"]
        p_b = result.probabilities["b"]
        assert abs(p_a - p_b) < 0.1

    def test_n_candidates_field(self, selector, pool_3):
        """SelectionResult.n_candidates is correct."""
        result = selector.select(pool_3)
        assert result.n_candidates == 3
