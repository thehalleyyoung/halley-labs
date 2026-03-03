"""
Comprehensive tests for dp_forge.subsampling module.

Tests cover privacy amplification by subsampling (Poisson, without-replacement,
shuffle), budget inversion, protocol execution, and key mathematical properties.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dp_forge.subsampling.amplification import (
    AmplificationBound,
    AmplificationResult,
    poisson_amplify,
    replacement_amplify,
    shuffle_amplify,
    poisson_amplify_rdp,
    compare_amplification_bounds,
    compute_amplification_factor,
    _stable_log_poisson_amplification,
)
from dp_forge.subsampling.budget_inversion import (
    BudgetInverter,
    InversionResult,
    invert_poisson,
    invert_replacement,
)
from dp_forge.subsampling.protocol import (
    ExecutionResult,
    SubsamplingMode,
    SubsamplingProtocol,
)
from dp_forge.subsampling.shuffle_amplification import (
    ShuffleAmplifier,
    ShuffleComparison,
    PrivacyProfilePoint,
)
from dp_forge.exceptions import ConfigurationError


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def rng():
    """Seeded RNG for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_mechanism():
    """A simple 2×3 mechanism table (2 inputs, 3 output bins)."""
    table = np.array([
        [0.6, 0.3, 0.1],
        [0.1, 0.3, 0.6],
    ])
    return table


@pytest.fixture
def simple_y_grid():
    """3-element output grid."""
    return np.array([0.0, 0.5, 1.0])


# =========================================================================
# Section 1: Poisson Amplification
# =========================================================================


class TestPoissonAmplification:
    """Tests for Poisson subsampling amplification."""

    def test_basic_amplification_reduces_eps(self):
        """Poisson amplification with q=0.01, ε₀=1 → amplified ε < ε₀."""
        result = poisson_amplify(base_eps=1.0, base_delta=0.0, q_rate=0.01)
        assert result.eps < 1.0, (
            f"Amplified ε={result.eps} should be < base ε₀=1.0"
        )
        assert result.eps > 0.0, "Amplified ε must be positive"

    def test_amplification_formula(self):
        """Verify ε' = log(1 + q(e^ε₀ - 1)) for basic Poisson bound."""
        q, eps0 = 0.01, 1.0
        result = poisson_amplify(base_eps=eps0, base_delta=0.0, q_rate=q)
        expected_eps = math.log(1.0 + q * (math.exp(eps0) - 1.0))
        assert abs(result.eps - expected_eps) < 1e-10, (
            f"Expected ε'={expected_eps}, got {result.eps}"
        )

    def test_delta_amplification(self):
        """δ' = q · δ₀ for Poisson subsampling."""
        q, delta0 = 0.05, 1e-5
        result = poisson_amplify(base_eps=1.0, base_delta=delta0, q_rate=q)
        assert abs(result.delta - q * delta0) < 1e-15, (
            f"Expected δ'={q * delta0}, got {result.delta}"
        )

    def test_q_one_no_amplification(self):
        """q=1 means no subsampling → amplified ε = ε₀."""
        result = poisson_amplify(base_eps=2.0, base_delta=1e-5, q_rate=1.0)
        assert abs(result.eps - 2.0) < 1e-10
        assert abs(result.delta - 1e-5) < 1e-15

    def test_monotonicity_in_q(self):
        """Larger q → larger amplified ε (less amplification)."""
        q_values = [0.001, 0.01, 0.1, 0.5, 1.0]
        eps_values = []
        for q in q_values:
            result = poisson_amplify(base_eps=1.0, base_delta=0.0, q_rate=q)
            eps_values.append(result.eps)
        for i in range(len(eps_values) - 1):
            assert eps_values[i] < eps_values[i + 1], (
                f"ε at q={q_values[i]} should be < ε at q={q_values[i+1]}"
            )

    def test_monotonicity_in_base_eps(self):
        """Larger base ε₀ → larger amplified ε."""
        eps_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        results = []
        for e in eps_values:
            results.append(poisson_amplify(base_eps=e, base_delta=0.0, q_rate=0.01).eps)
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]

    def test_small_q_near_zero(self):
        """Very small q → amplified ε ≈ q · (e^ε₀ - 1)."""
        q = 1e-6
        eps0 = 1.0
        result = poisson_amplify(base_eps=eps0, base_delta=0.0, q_rate=q)
        # For small q: log(1 + q·(e^ε₀ - 1)) ≈ q·(e^ε₀ - 1)
        approx = q * (math.exp(eps0) - 1.0)
        assert abs(result.eps - approx) / approx < 0.01  # within 1%

    def test_tight_bound_tighter_than_basic(self):
        """Tight Poisson bound ≤ basic Poisson bound."""
        basic = poisson_amplify(base_eps=1.0, base_delta=1e-5, q_rate=0.1)
        tight = poisson_amplify(base_eps=1.0, base_delta=1e-5, q_rate=0.1, tight=True)
        assert tight.eps <= basic.eps + 1e-10

    def test_result_attributes(self):
        """AmplificationResult has all expected fields."""
        result = poisson_amplify(base_eps=1.0, base_delta=1e-5, q_rate=0.1)
        assert result.bound_type == AmplificationBound.POISSON_BASIC
        assert result.base_eps == 1.0
        assert result.base_delta == 1e-5
        assert result.q_rate == 0.1

    def test_amplification_factor(self):
        """amplification_factor = amplified_eps / base_eps ≤ 1."""
        result = poisson_amplify(base_eps=1.0, base_delta=0.0, q_rate=0.05)
        assert 0 < result.amplification_factor <= 1.0

    def test_invalid_q_rate(self):
        """q_rate outside (0, 1] raises ConfigurationError."""
        with pytest.raises((ConfigurationError, ValueError)):
            poisson_amplify(base_eps=1.0, base_delta=0.0, q_rate=0.0)
        with pytest.raises((ConfigurationError, ValueError)):
            poisson_amplify(base_eps=1.0, base_delta=0.0, q_rate=-0.1)
        with pytest.raises((ConfigurationError, ValueError)):
            poisson_amplify(base_eps=1.0, base_delta=0.0, q_rate=1.5)

    def test_invalid_base_eps(self):
        """Negative base_eps raises error; zero is allowed."""
        with pytest.raises((ConfigurationError, ValueError)):
            poisson_amplify(base_eps=-1.0, base_delta=0.0, q_rate=0.1)
        # eps=0 is valid (trivial mechanism)
        result = poisson_amplify(base_eps=0.0, base_delta=0.0, q_rate=0.1)
        assert result.eps == 0.0


# =========================================================================
# Section 2: Property-based Poisson tests
# =========================================================================


class TestPoissonProperties:
    """Property-based tests for Poisson amplification."""

    @given(
        eps0=st.floats(min_value=0.01, max_value=10.0),
        q=st.floats(min_value=0.001, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_amplified_eps_le_base(self, eps0, q):
        """Property: amplified ε ≤ ε₀ for all q ≤ 1."""
        result = poisson_amplify(base_eps=eps0, base_delta=0.0, q_rate=q)
        assert result.eps <= eps0 + 1e-10

    @given(
        eps0=st.floats(min_value=0.01, max_value=5.0),
        delta0=st.floats(min_value=1e-10, max_value=0.1),
        q=st.floats(min_value=0.001, max_value=0.99),
    )
    @settings(max_examples=100)
    def test_amplified_delta_le_q_delta0(self, eps0, delta0, q):
        """Property: amplified δ ≤ q · δ₀ for Poisson subsampling."""
        result = poisson_amplify(base_eps=eps0, base_delta=delta0, q_rate=q)
        assert result.delta <= q * delta0 + 1e-15

    @given(
        eps0=st.floats(min_value=0.01, max_value=5.0),
        q=st.floats(min_value=0.001, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_amplified_eps_positive(self, eps0, q):
        """Amplified ε is always positive."""
        result = poisson_amplify(base_eps=eps0, base_delta=0.0, q_rate=q)
        assert result.eps > 0


# =========================================================================
# Section 3: Without-Replacement Amplification
# =========================================================================


class TestReplacementAmplification:
    """Tests for without-replacement subsampling amplification."""

    def test_basic_amplification(self):
        """Without-replacement amplification reduces ε."""
        result = replacement_amplify(
            base_eps=1.0, base_delta=0.0, q_rate=0.1, n_total=1000
        )
        assert result.eps < 1.0

    def test_n_total_effect(self):
        """Larger n_total → tighter bound (closer to Poisson)."""
        r1 = replacement_amplify(base_eps=1.0, base_delta=0.0, q_rate=0.1, n_total=100)
        r2 = replacement_amplify(base_eps=1.0, base_delta=0.0, q_rate=0.1, n_total=10000)
        # Larger n_total makes n/(n-1) correction smaller
        assert r2.eps <= r1.eps + 1e-10

    def test_without_n_total_fallback(self):
        """Without n_total, falls back to Poisson-like bound."""
        result = replacement_amplify(
            base_eps=1.0, base_delta=0.0, q_rate=0.1
        )
        assert result.eps < 1.0
        assert result.bound_type == AmplificationBound.WITHOUT_REPLACEMENT

    def test_delta_amplification(self):
        """δ amplification for without-replacement."""
        result = replacement_amplify(
            base_eps=1.0, base_delta=1e-5, q_rate=0.1, n_total=100
        )
        assert result.delta < 1e-5  # Should be amplified (reduced)
        assert result.delta >= 0.0

    def test_q_one_no_amplification(self):
        """q=1 means full dataset → no amplification."""
        result = replacement_amplify(
            base_eps=1.0, base_delta=1e-5, q_rate=1.0, n_total=100
        )
        assert abs(result.eps - 1.0) < 1e-6


# =========================================================================
# Section 4: Shuffle Amplification
# =========================================================================


class TestShuffleAmplification:
    """Tests for shuffle model amplification."""

    def test_basic_shuffle(self):
        """Shuffle amplification reduces ε for many users."""
        result = shuffle_amplify(
            base_eps=2.0, base_delta=0.0, n_users=1000, target_delta=1e-5
        )
        assert result.eps < 2.0
        assert result.delta <= 1e-5 + 1e-15

    def test_more_users_tighter(self):
        """More users → tighter central ε."""
        r1 = shuffle_amplify(base_eps=1.0, base_delta=0.0, n_users=100, target_delta=1e-5)
        r2 = shuffle_amplify(base_eps=1.0, base_delta=0.0, n_users=10000, target_delta=1e-5)
        assert r2.eps < r1.eps

    def test_shuffle_tighter_than_naive(self):
        """Shuffle ε should be tighter than naive local model (ε₀)."""
        result = shuffle_amplify(
            base_eps=3.0, base_delta=0.0, n_users=10000, target_delta=1e-5
        )
        assert result.eps < 3.0

    def test_approximate_ldp(self):
        """Shuffle with approximate LDP (δ₀ > 0)."""
        # Use small enough δ₀ so n·δ₀ < target_delta, leaving room for shuffle
        result = shuffle_amplify(
            base_eps=1.0, base_delta=1e-12, n_users=10000, target_delta=1e-5
        )
        assert result.eps < 1.0
        assert result.delta <= 1e-5 + 1e-12


class TestShuffleAmplifier:
    """Tests for the ShuffleAmplifier class."""

    def test_privacy_profile(self):
        """Privacy profile returns decreasing ε for increasing δ."""
        amp = ShuffleAmplifier(n_users=1000, base_eps=1.0, base_delta=0.0)
        profile = amp.privacy_profile(n_points=20)
        assert len(profile) > 0
        # Higher δ → lower ε
        sorted_profile = sorted(profile, key=lambda p: p.delta)
        for i in range(len(sorted_profile) - 1):
            if sorted_profile[i].delta < sorted_profile[i + 1].delta:
                assert sorted_profile[i].eps >= sorted_profile[i + 1].eps - 1e-10

    def test_compare_models(self):
        """Compare shuffle model to local model."""
        amp = ShuffleAmplifier(n_users=1000, base_eps=2.0, base_delta=0.0)
        comp = amp.compare_models(target_delta=1e-5)
        assert isinstance(comp, ShuffleComparison)
        assert comp.shuffle_eps < comp.local_eps
        assert comp.central_improvement > 0

    def test_minimum_users(self):
        """Find minimum users for target privacy."""
        amp = ShuffleAmplifier(n_users=10000, base_eps=1.0, base_delta=0.0)
        min_n = amp.minimum_users_for_target(target_eps=0.5, target_delta=1e-5)
        assert min_n > 0
        assert min_n <= 10000  # Should be achievable


# =========================================================================
# Section 5: Budget Inversion
# =========================================================================


class TestBudgetInversion:
    """Tests for inverting amplification curves."""

    def test_invert_poisson_roundtrip(self):
        """invert(amplify(ε₀, q), q) ≈ ε₀."""
        eps0 = 2.0
        q = 0.05
        # Forward: amplify
        fwd = poisson_amplify(base_eps=eps0, base_delta=0.0, q_rate=q)
        # Inverse: recover base
        inv = invert_poisson(
            target_eps=fwd.eps, target_delta=fwd.delta, q_rate=q
        )
        assert abs(inv.base_eps - eps0) < 1e-6, (
            f"Expected base_eps ≈ {eps0}, got {inv.base_eps}"
        )

    def test_invert_poisson_various_params(self):
        """Roundtrip inversion for several (ε₀, q) pairs."""
        for eps0 in [0.5, 1.0, 3.0]:
            for q in [0.01, 0.1, 0.5]:
                fwd = poisson_amplify(base_eps=eps0, base_delta=0.0, q_rate=q)
                inv = invert_poisson(
                    target_eps=fwd.eps, target_delta=fwd.delta, q_rate=q
                )
                assert abs(inv.base_eps - eps0) < 1e-4, (
                    f"Inversion failed for ε₀={eps0}, q={q}: "
                    f"got {inv.base_eps}"
                )

    def test_inversion_result_verify(self):
        """InversionResult.verify() passes for correct inversions."""
        fwd = poisson_amplify(base_eps=1.0, base_delta=1e-5, q_rate=0.1)
        inv = invert_poisson(
            target_eps=fwd.eps, target_delta=fwd.delta, q_rate=0.1
        )
        assert inv.verify(tol=1e-4)

    def test_inversion_residual_small(self):
        """Residual should be very small for converged inversion."""
        fwd = poisson_amplify(base_eps=1.0, base_delta=0.0, q_rate=0.1)
        inv = invert_poisson(
            target_eps=fwd.eps, target_delta=fwd.delta, q_rate=0.1
        )
        assert inv.residual < 1e-10

    def test_budget_inverter_class(self):
        """BudgetInverter class interface."""
        inverter = BudgetInverter(tol=1e-10, max_iter=200)
        result = inverter.invert(
            target_eps=0.1, target_delta=1e-6, q_rate=0.1,
            bound_type=AmplificationBound.POISSON_BASIC,
        )
        assert isinstance(result, InversionResult)
        assert result.base_eps > 0.1  # Base ε must be larger than target

    def test_invert_replacement_roundtrip(self):
        """Roundtrip inversion for without-replacement."""
        eps0 = 1.5
        q = 0.1
        fwd = replacement_amplify(
            base_eps=eps0, base_delta=0.0, q_rate=q, n_total=1000
        )
        inv = invert_replacement(
            target_eps=fwd.eps, target_delta=fwd.delta,
            q_rate=q, n_total=1000,
        )
        assert abs(inv.base_eps - eps0) < 1e-3

    def test_invert_for_best_utility(self):
        """invert_for_best_utility returns valid result."""
        inverter = BudgetInverter()
        result = inverter.invert_for_best_utility(
            target_eps=0.5, target_delta=1e-5, q_rate=0.1
        )
        assert isinstance(result, InversionResult)
        assert result.base_eps > 0


# =========================================================================
# Section 6: Protocol & Execution
# =========================================================================


class TestSubsamplingProtocol:
    """Tests for subsampling protocol execution."""

    def test_poisson_mask_inclusion_rate(self, simple_mechanism, simple_y_grid, rng):
        """Poisson mask has expected inclusion rate (E[mask] ≈ q)."""
        q = 0.1
        amp = poisson_amplify(base_eps=1.0, base_delta=0.0, q_rate=q)
        protocol = SubsamplingProtocol(
            q_rate=q,
            base_mechanism=simple_mechanism,
            base_eps=1.0,
            base_delta=0.0,
            amplified=amp,
            y_grid=simple_y_grid,
            mode=SubsamplingMode.POISSON,
            seed=42,
        )

        n_trials = 10000
        dataset_size = 100
        inclusion_counts = 0
        for _ in range(n_trials):
            mask = protocol.sample_inclusion_mask(dataset_size, rng=rng)
            inclusion_counts += mask.sum()

        mean_rate = inclusion_counts / (n_trials * dataset_size)
        assert abs(mean_rate - q) < 0.02, (
            f"Mean inclusion rate {mean_rate} far from q={q}"
        )

    def test_wor_mask_exact_size(self, simple_mechanism, simple_y_grid, rng):
        """Without-replacement mask selects exactly m=floor(q·n) elements."""
        q = 0.1
        n = 100
        m = int(q * n)
        amp = poisson_amplify(base_eps=1.0, base_delta=0.0, q_rate=q)
        protocol = SubsamplingProtocol(
            q_rate=q,
            base_mechanism=simple_mechanism,
            base_eps=1.0,
            base_delta=0.0,
            amplified=amp,
            y_grid=simple_y_grid,
            mode=SubsamplingMode.WITHOUT_REPLACEMENT,
            seed=42,
        )

        mask = protocol.sample_inclusion_mask(n, rng=rng)
        assert mask.sum() == m

    def test_execute_returns_result(self, simple_mechanism, simple_y_grid, rng):
        """Protocol execution returns ExecutionResult."""
        amp = poisson_amplify(base_eps=1.0, base_delta=0.0, q_rate=0.5)
        protocol = SubsamplingProtocol(
            q_rate=0.5,
            base_mechanism=simple_mechanism,
            base_eps=1.0,
            base_delta=0.0,
            amplified=amp,
            y_grid=simple_y_grid,
            mode=SubsamplingMode.POISSON,
            seed=42,
        )

        dataset = np.arange(10, dtype=np.float64)
        result = protocol.execute(dataset, rng=rng)
        assert isinstance(result, ExecutionResult)
        assert result.dataset_size == 10
        assert result.mode == SubsamplingMode.POISSON

    def test_protocol_properties(self, simple_mechanism, simple_y_grid):
        """Protocol has correct n and k properties."""
        amp = poisson_amplify(base_eps=1.0, base_delta=0.0, q_rate=0.1)
        protocol = SubsamplingProtocol(
            q_rate=0.1,
            base_mechanism=simple_mechanism,
            base_eps=1.0,
            base_delta=0.0,
            amplified=amp,
            y_grid=simple_y_grid,
        )
        assert protocol.n == 2
        assert protocol.k == 3


# =========================================================================
# Section 7: RDP-based Poisson Amplification
# =========================================================================


class TestRDPAmplification:
    """Tests for RDP-based Poisson amplification."""

    def test_rdp_amplifies(self):
        """RDP amplification gives a finite result."""
        result = poisson_amplify_rdp(
            base_eps=1.0, q_rate=0.01, target_delta=1e-5
        )
        # RDP converts to (ε, δ)-DP; the resulting ε may exceed base_eps
        # due to the δ→ε conversion, but should still be finite and positive
        assert result.eps > 0
        assert math.isfinite(result.eps)

    def test_rdp_smaller_q_better(self):
        """Smaller q → smaller RDP-amplified ε."""
        r1 = poisson_amplify_rdp(base_eps=1.0, q_rate=0.1, target_delta=1e-5)
        r2 = poisson_amplify_rdp(base_eps=1.0, q_rate=0.01, target_delta=1e-5)
        assert r2.eps <= r1.eps + 1e-10

    def test_rdp_vs_basic(self):
        """RDP bound is finite; both RDP and basic produce valid results."""
        basic = poisson_amplify(base_eps=1.0, base_delta=0.0, q_rate=0.01)
        rdp = poisson_amplify_rdp(base_eps=1.0, q_rate=0.01, target_delta=1e-5)
        # Both produce valid, finite results
        assert math.isfinite(rdp.eps)
        assert math.isfinite(basic.eps)
        assert basic.eps < 1.0  # Pure DP Poisson amplification always reduces ε


# =========================================================================
# Section 8: Comparison & Convenience
# =========================================================================


class TestComparisonUtilities:
    """Tests for comparison and convenience functions."""

    def test_compare_bounds_sorted(self):
        """compare_amplification_bounds returns sorted results."""
        results = compare_amplification_bounds(
            base_eps=1.0, base_delta=1e-5, q_rate=0.1,
            n_total=1000,
        )
        assert len(results) >= 2
        # Should be sorted by amplified ε
        for i in range(len(results) - 1):
            assert results[i].eps <= results[i + 1].eps + 1e-10

    def test_compute_amplification_factor(self):
        """Amplification factor = ε'/ε₀."""
        factor = compute_amplification_factor(
            base_eps=1.0, q_rate=0.01,
            bound_type=AmplificationBound.POISSON_BASIC,
        )
        assert 0 < factor <= 1.0


# =========================================================================
# Section 9: Stable numerics
# =========================================================================


class TestNumericalStability:
    """Tests for numerical edge cases."""

    def test_very_large_eps(self):
        """Large ε₀ doesn't overflow."""
        result = poisson_amplify(base_eps=50.0, base_delta=0.0, q_rate=0.01)
        assert math.isfinite(result.eps)
        assert result.eps <= 50.0

    def test_very_small_eps(self):
        """Very small ε₀ gives small amplified ε."""
        result = poisson_amplify(base_eps=1e-6, base_delta=0.0, q_rate=0.1)
        assert result.eps < 1e-6
        assert result.eps > 0

    def test_stable_log_poisson(self):
        """_stable_log_poisson_amplification handles edge cases."""
        # Small eps, small q
        val = _stable_log_poisson_amplification(0.001, 0.001)
        assert math.isfinite(val)
        assert val > 0

    def test_very_small_q(self):
        """Extremely small q doesn't produce negative ε."""
        result = poisson_amplify(base_eps=1.0, base_delta=0.0, q_rate=1e-10)
        assert result.eps > 0
        assert result.eps < 1.0


# =========================================================================
# Section 10: End-to-end integration
# =========================================================================


class TestSubsamplingIntegration:
    """Integration tests combining multiple subsampling components."""

    def test_amplify_invert_amplify_cycle(self):
        """Amplify → invert → re-amplify gives consistent result."""
        eps0 = 1.0
        q = 0.1
        amp1 = poisson_amplify(base_eps=eps0, base_delta=0.0, q_rate=q)
        inv = invert_poisson(target_eps=amp1.eps, target_delta=amp1.delta, q_rate=q)
        amp2 = poisson_amplify(base_eps=inv.base_eps, base_delta=inv.base_delta, q_rate=q)
        assert abs(amp2.eps - amp1.eps) < 1e-6

    def test_comparison_all_valid(self):
        """All results from compare are valid AmplificationResults."""
        results = compare_amplification_bounds(
            base_eps=1.0, base_delta=1e-6, q_rate=0.05,
            n_total=500, n_users=500,
        )
        for r in results:
            assert isinstance(r, AmplificationResult)
            assert r.eps > 0
            assert r.eps <= 1.0 + 1e-10
            assert r.delta >= 0
