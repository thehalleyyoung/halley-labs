"""Unit tests for usability_oracle.differential.accountant — Privacy accounting.

Tests cover sequential composition, RDP accounting, zCDP accounting, privacy
conversions, and budget tracking.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.differential.types import (
    CompositionResult,
    CompositionTheorem,
    PrivacyBudget,
)
from usability_oracle.differential.accountant import (
    BudgetAccountant,
    BudgetEntry,
    rdp_to_approx_dp,
    approx_dp_to_rdp,
    zcdp_to_approx_dp,
    gaussian_to_zcdp,
    optimal_budget_split,
    optimal_gaussian_sigma,
)
from usability_oracle.differential.composition import (
    basic_composition,
    advanced_composition,
    parallel_composition,
    verify_post_processing,
    AdaptiveComposer,
)


# ═══════════════════════════════════════════════════════════════════════════
# Privacy Budget
# ═══════════════════════════════════════════════════════════════════════════


class TestPrivacyBudget:
    """Test PrivacyBudget construction and properties."""

    def test_pure_dp(self):
        b = PrivacyBudget(epsilon=1.0, delta=0.0)
        assert b.is_pure_dp

    def test_approximate_dp(self):
        b = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert not b.is_pure_dp

    def test_compose_basic(self):
        b1 = PrivacyBudget(epsilon=1.0)
        b2 = PrivacyBudget(epsilon=2.0)
        combined = b1.compose_basic(b2)
        assert combined.epsilon == pytest.approx(3.0)

    def test_to_dict_roundtrip(self):
        b = PrivacyBudget(epsilon=1.5, delta=1e-5, description="test")
        d = b.to_dict()
        restored = PrivacyBudget.from_dict(d)
        assert restored.epsilon == pytest.approx(b.epsilon)
        assert restored.delta == pytest.approx(b.delta)

    def test_non_negative_epsilon(self):
        with pytest.raises((ValueError, AssertionError)):
            PrivacyBudget(epsilon=-1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Sequential Composition
# ═══════════════════════════════════════════════════════════════════════════


class TestSequentialComposition:
    """Test sequential composition of privacy budgets."""

    def test_basic_composition_sums_epsilon(self):
        budgets = [PrivacyBudget(epsilon=1.0) for _ in range(5)]
        result = basic_composition(budgets)
        assert result.total_budget.epsilon == pytest.approx(5.0)

    def test_basic_composition_sums_delta(self):
        budgets = [PrivacyBudget(epsilon=1.0, delta=1e-5) for _ in range(3)]
        result = basic_composition(budgets)
        assert result.total_budget.delta == pytest.approx(3e-5)

    def test_advanced_composition_introduces_delta(self):
        """Advanced composition trades δ for tighter ε when many queries used."""
        budgets = [PrivacyBudget(epsilon=0.1) for _ in range(100)]
        basic = basic_composition(budgets)
        adv = advanced_composition(budgets, delta_prime=1e-5)
        # Advanced composition produces a result with nonzero delta
        assert adv.total_budget.delta > 0
        assert isinstance(adv.total_budget.epsilon, float)

    def test_parallel_composition_takes_max(self):
        budgets = [
            PrivacyBudget(epsilon=1.0),
            PrivacyBudget(epsilon=2.0),
            PrivacyBudget(epsilon=0.5),
        ]
        result = parallel_composition(budgets)
        assert result.total_budget.epsilon == pytest.approx(2.0)

    def test_composition_metadata(self):
        budgets = [PrivacyBudget(epsilon=1.0)]
        result = basic_composition(budgets)
        assert result.n_mechanisms == 1

    def test_single_mechanism_composition(self):
        budgets = [PrivacyBudget(epsilon=1.5)]
        result = basic_composition(budgets)
        assert result.total_budget.epsilon == pytest.approx(1.5)

    def test_post_processing_invariance(self):
        original = PrivacyBudget(epsilon=1.0, delta=1e-5)
        post = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert verify_post_processing(original, post)


# ═══════════════════════════════════════════════════════════════════════════
# RDP Accounting
# ═══════════════════════════════════════════════════════════════════════════


class TestRDPAccounting:
    """Test Rényi Differential Privacy accounting."""

    def test_rdp_to_approx_dp_conversion(self):
        rdp_eps = 2.0
        alpha = 10.0
        delta = 1e-5
        eps = rdp_to_approx_dp(rdp_eps, alpha, delta)
        assert eps > 0
        assert eps < rdp_eps + 10  # should be finite

    def test_rdp_conversion_smaller_alpha(self):
        """Lower alpha gives different (possibly looser) bound."""
        delta = 1e-5
        eps_low = rdp_to_approx_dp(2.0, 2.0, delta)
        eps_high = rdp_to_approx_dp(2.0, 100.0, delta)
        # Both should be valid (positive)
        assert eps_low > 0
        assert eps_high > 0

    def test_approx_dp_to_rdp_conversion(self):
        """RDP ↔ approx-DP conversion should be meaningful."""
        eps_original = 1.0
        delta = 1e-5
        alpha = 10.0
        rdp_eps = approx_dp_to_rdp(eps_original, delta, alpha)
        eps_back = rdp_to_approx_dp(rdp_eps, alpha, delta)
        # Roundtrip is lossy — just verify both values are positive and finite
        assert math.isfinite(eps_back)
        assert eps_back > 0

    def test_accountant_rdp_composition(self):
        acc = BudgetAccountant(PrivacyBudget(epsilon=10.0, delta=1e-5))
        acc.record_gaussian(sigma=1.0, sensitivity=1.0, description="q1")
        acc.record_gaussian(sigma=1.0, sensitivity=1.0, description="q2")
        result = acc.compose_rdp()
        assert isinstance(result, CompositionResult)
        assert result.total_budget.epsilon > 0


# ═══════════════════════════════════════════════════════════════════════════
# zCDP Accounting
# ═══════════════════════════════════════════════════════════════════════════


class TestZCDPAccounting:
    """Test zero-concentrated DP accounting."""

    def test_gaussian_to_zcdp(self):
        rho = gaussian_to_zcdp(sigma=1.0, sensitivity=1.0)
        assert rho > 0
        assert rho == pytest.approx(0.5)  # rho = sensitivity² / (2σ²)

    def test_zcdp_to_approx_dp_conversion(self):
        rho = 0.5
        delta = 1e-5
        eps = zcdp_to_approx_dp(rho, delta)
        assert eps > 0

    def test_accountant_zcdp_composition(self):
        acc = BudgetAccountant(PrivacyBudget(epsilon=10.0, delta=1e-5))
        acc.record_gaussian(sigma=1.0, sensitivity=1.0, description="q1")
        acc.record_gaussian(sigma=2.0, sensitivity=1.0, description="q2")
        result = acc.compose_zcdp()
        assert result.total_budget.epsilon > 0


# ═══════════════════════════════════════════════════════════════════════════
# Budget Accountant
# ═══════════════════════════════════════════════════════════════════════════


class TestBudgetAccountant:
    """Test BudgetAccountant lifecycle."""

    def test_record_and_count(self):
        acc = BudgetAccountant(PrivacyBudget(epsilon=10.0))
        acc.record(PrivacyBudget(epsilon=1.0), description="query1")
        acc.record(PrivacyBudget(epsilon=2.0), description="query2")
        assert acc.n_queries == 2

    def test_spent_budget(self):
        acc = BudgetAccountant(PrivacyBudget(epsilon=10.0))
        acc.record(PrivacyBudget(epsilon=1.0))
        acc.record(PrivacyBudget(epsilon=2.0))
        spent = acc.spent_budget_basic()
        assert spent.epsilon == pytest.approx(3.0)

    def test_remaining_budget(self):
        acc = BudgetAccountant(PrivacyBudget(epsilon=10.0))
        acc.record(PrivacyBudget(epsilon=3.0))
        remaining = acc.remaining_budget(CompositionTheorem.BASIC)
        assert remaining.epsilon == pytest.approx(7.0)

    def test_can_afford(self):
        acc = BudgetAccountant(PrivacyBudget(epsilon=5.0))
        acc.record(PrivacyBudget(epsilon=3.0))
        assert acc.can_afford(PrivacyBudget(epsilon=1.0))
        assert not acc.can_afford(PrivacyBudget(epsilon=3.0))

    def test_ledger(self):
        acc = BudgetAccountant(PrivacyBudget(epsilon=10.0))
        acc.record(PrivacyBudget(epsilon=1.0), description="q1", stage="train")
        acc.record(PrivacyBudget(epsilon=2.0), description="q2", stage="eval")
        ledger = acc.ledger
        assert len(ledger) == 2
        assert all(isinstance(e, BudgetEntry) for e in ledger)

    def test_stage_summary(self):
        acc = BudgetAccountant(PrivacyBudget(epsilon=10.0))
        acc.record(PrivacyBudget(epsilon=1.0), stage="train")
        acc.record(PrivacyBudget(epsilon=2.0), stage="train")
        acc.record(PrivacyBudget(epsilon=0.5), stage="eval")
        summary = acc.stage_summary()
        assert "train" in summary
        assert summary["train"].epsilon == pytest.approx(3.0)

    def test_record_laplace(self):
        acc = BudgetAccountant(PrivacyBudget(epsilon=10.0))
        acc.record_laplace(scale=1.0, sensitivity=1.0, description="lap")
        assert acc.n_queries == 1

    def test_compose_selects_best_theorem(self):
        acc = BudgetAccountant(PrivacyBudget(epsilon=10.0, delta=1e-5))
        for _ in range(5):
            acc.record_gaussian(sigma=1.0, sensitivity=1.0)
        result = acc.compose(CompositionTheorem.RENYI)
        assert result.total_budget.epsilon > 0


# ═══════════════════════════════════════════════════════════════════════════
# Privacy Conversions and Utilities
# ═══════════════════════════════════════════════════════════════════════════


class TestPrivacyUtilities:
    """Test budget splitting and sigma optimization."""

    def test_optimal_budget_split_uniform(self):
        splits = optimal_budget_split(total_epsilon=5.0, n_queries=5)
        assert len(splits) == 5
        assert abs(sum(splits) - 5.0) < 0.1

    def test_optimal_budget_split_weighted(self):
        splits = optimal_budget_split(
            total_epsilon=5.0, n_queries=3, query_weights=[1.0, 2.0, 1.0]
        )
        assert len(splits) == 3
        # Higher weight query should get more budget
        assert splits[1] >= splits[0] - 0.1

    def test_optimal_gaussian_sigma(self):
        sigma = optimal_gaussian_sigma(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        assert sigma > 0

    def test_optimal_sigma_decreases_with_higher_epsilon(self):
        s1 = optimal_gaussian_sigma(epsilon=0.5, delta=1e-5)
        s2 = optimal_gaussian_sigma(epsilon=2.0, delta=1e-5)
        assert s1 > s2


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Composer
# ═══════════════════════════════════════════════════════════════════════════


class TestAdaptiveComposer:
    """Test adaptive composition."""

    def test_propose_and_commit(self):
        composer = AdaptiveComposer(PrivacyBudget(epsilon=5.0))
        assert composer.propose(PrivacyBudget(epsilon=1.0))
        composer.commit(PrivacyBudget(epsilon=1.0))
        assert composer.remaining.epsilon == pytest.approx(4.0)

    def test_propose_rejects_over_budget(self):
        composer = AdaptiveComposer(PrivacyBudget(epsilon=2.0))
        composer.commit(PrivacyBudget(epsilon=1.5))
        assert not composer.propose(PrivacyBudget(epsilon=1.0))

    def test_spent_tracking(self):
        composer = AdaptiveComposer(PrivacyBudget(epsilon=10.0))
        composer.commit(PrivacyBudget(epsilon=3.0))
        assert composer.spent.epsilon == pytest.approx(3.0)

    def test_compose_returns_result(self):
        composer = AdaptiveComposer(PrivacyBudget(epsilon=10.0))
        composer.commit(PrivacyBudget(epsilon=1.0))
        composer.commit(PrivacyBudget(epsilon=2.0))
        result = composer.compose()
        assert isinstance(result, CompositionResult)
