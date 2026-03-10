"""
Unit tests for usability_oracle.algebra.parallel — Parallel composition ⊗.

Tests the ParallelComposer class which implements concurrent task composition
grounded in Wickens' Multiple Resource Theory (MRT).

Mathematical model under test:
    μ_{a⊗b}  = max(μ_a, μ_b) + η·min(μ_a, μ_b)
    σ²_{a⊗b} = max(σ²_a, σ²_b) + η²·min(σ²_a, σ²_b)
    κ_{a⊗b}  = κ of the higher-variance channel
    λ_{a⊗b}  = λ_a + λ_b + η·λ_a·λ_b
"""

import math
import pytest
import numpy as np

from usability_oracle.algebra.models import CostElement
from usability_oracle.algebra.parallel import (
    ParallelComposer,
    CHANNELS,
    INTERFERENCE_MATRIX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def composer():
    """Return a fresh ParallelComposer instance."""
    return ParallelComposer()


@pytest.fixture
def elem_a():
    """A visual-focal task cost element."""
    return CostElement(mu=2.0, sigma_sq=0.5, kappa=0.2, lambda_=0.05)


@pytest.fixture
def elem_b():
    """An auditory-verbal task cost element."""
    return CostElement(mu=1.0, sigma_sq=0.2, kappa=0.1, lambda_=0.03)


@pytest.fixture
def elem_c():
    """A manual-response cost element."""
    return CostElement(mu=0.8, sigma_sq=0.15, kappa=0.0, lambda_=0.02)


@pytest.fixture
def zero():
    """The additive identity element."""
    return CostElement.zero()


# ---------------------------------------------------------------------------
# Basic parallel composition
# ---------------------------------------------------------------------------

class TestComposeBasic:
    """Tests for basic parallel composition with zero interference."""

    def test_compose_mu_max(self, composer, elem_a, elem_b):
        """With zero interference, composed μ = max(μ_a, μ_b)."""
        result = composer.compose(elem_a, elem_b, interference=0.0)
        assert math.isclose(result.mu, max(elem_a.mu, elem_b.mu), rel_tol=1e-10)

    def test_compose_sigma_sq_max(self, composer, elem_a, elem_b):
        """With zero interference, composed σ² = max(σ²_a, σ²_b)."""
        result = composer.compose(elem_a, elem_b, interference=0.0)
        assert math.isclose(result.sigma_sq, max(elem_a.sigma_sq, elem_b.sigma_sq), rel_tol=1e-10)

    def test_compose_kappa_from_higher_var(self, composer, elem_a, elem_b):
        """Kappa comes from the channel with higher variance."""
        result = composer.compose(elem_a, elem_b, interference=0.0)
        if elem_a.sigma_sq >= elem_b.sigma_sq:
            assert math.isclose(result.kappa, elem_a.kappa, abs_tol=1e-12)
        else:
            assert math.isclose(result.kappa, elem_b.kappa, abs_tol=1e-12)

    def test_compose_lambda_sum_no_interference(self, composer, elem_a, elem_b):
        """With η=0, λ = λ_a + λ_b + 0·λ_a·λ_b = λ_a + λ_b."""
        result = composer.compose(elem_a, elem_b, interference=0.0)
        expected = elem_a.lambda_ + elem_b.lambda_
        assert math.isclose(result.lambda_, expected, rel_tol=1e-10)

    def test_compose_returns_cost_element(self, composer, elem_a, elem_b):
        """compose() returns a CostElement instance."""
        result = composer.compose(elem_a, elem_b)
        assert isinstance(result, CostElement)

    def test_compose_result_is_valid(self, composer, elem_a, elem_b):
        """The composed result satisfies validity constraints."""
        result = composer.compose(elem_a, elem_b, interference=0.5)
        assert result.is_valid


# ---------------------------------------------------------------------------
# Interference tests
# ---------------------------------------------------------------------------

class TestComposeInterference:
    """Tests for parallel composition with non-zero interference."""

    def test_interference_increases_mu(self, composer, elem_a, elem_b):
        """Positive interference increases the composed mean cost."""
        clean = composer.compose(elem_a, elem_b, interference=0.0)
        dirty = composer.compose(elem_a, elem_b, interference=0.5)
        assert dirty.mu > clean.mu

    def test_interference_mu_formula(self, composer, elem_a, elem_b):
        """Verify exact formula: μ = max(μ_a, μ_b) + η·min(μ_a, μ_b)."""
        eta = 0.4
        result = composer.compose(elem_a, elem_b, interference=eta)
        expected = max(elem_a.mu, elem_b.mu) + eta * min(elem_a.mu, elem_b.mu)
        assert math.isclose(result.mu, expected, rel_tol=1e-10)

    def test_interference_sigma_sq_formula(self, composer, elem_a, elem_b):
        """Verify exact formula: σ² = max(σ²) + η²·min(σ²)."""
        eta = 0.6
        result = composer.compose(elem_a, elem_b, interference=eta)
        expected = max(elem_a.sigma_sq, elem_b.sigma_sq) + eta**2 * min(elem_a.sigma_sq, elem_b.sigma_sq)
        assert math.isclose(result.sigma_sq, expected, rel_tol=1e-10)

    def test_interference_lambda_formula(self, composer, elem_a, elem_b):
        """Verify exact formula: λ = λ_a + λ_b + η·λ_a·λ_b."""
        eta = 0.3
        result = composer.compose(elem_a, elem_b, interference=eta)
        expected = elem_a.lambda_ + elem_b.lambda_ + eta * elem_a.lambda_ * elem_b.lambda_
        assert math.isclose(result.lambda_, expected, rel_tol=1e-10)

    def test_full_interference_equals_sum(self, composer, elem_a, elem_b):
        """Full interference (η=1) gives μ = μ_a + μ_b (effectively serial)."""
        result = composer.compose(elem_a, elem_b, interference=1.0)
        assert math.isclose(result.mu, elem_a.mu + elem_b.mu, rel_tol=1e-10)

    def test_interference_monotonic_in_eta(self, composer, elem_a, elem_b):
        """Composed μ is monotonically non-decreasing in η."""
        mus = [composer.compose(elem_a, elem_b, interference=e).mu for e in np.linspace(0, 1, 20)]
        for i in range(1, len(mus)):
            assert mus[i] >= mus[i - 1] - 1e-12

    def test_lambda_clamped_to_one(self, composer):
        """Composed λ is clamped to [0, 1] even with high tail risk inputs."""
        a = CostElement(mu=1.0, sigma_sq=0.1, kappa=0.0, lambda_=0.7)
        b = CostElement(mu=1.0, sigma_sq=0.1, kappa=0.0, lambda_=0.8)
        result = composer.compose(a, b, interference=1.0)
        assert result.lambda_ <= 1.0


# ---------------------------------------------------------------------------
# Identity property
# ---------------------------------------------------------------------------

class TestComposeIdentity:
    """Tests for identity: a ⊗ 0 = a."""

    def test_compose_with_zero_right(self, composer, elem_a, zero):
        """Composing with zero on the right returns the original element's mu."""
        result = composer.compose(elem_a, zero, interference=0.0)
        assert math.isclose(result.mu, elem_a.mu, abs_tol=1e-12)
        assert math.isclose(result.sigma_sq, elem_a.sigma_sq, abs_tol=1e-12)

    def test_compose_with_zero_left(self, composer, elem_a, zero):
        """Composing with zero on the left returns the original element's mu."""
        result = composer.compose(zero, elem_a, interference=0.0)
        assert math.isclose(result.mu, elem_a.mu, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Commutativity
# ---------------------------------------------------------------------------

class TestCommutativity:
    """Tests for commutativity: a ⊗ b = b ⊗ a."""

    def test_compose_commutative_mu(self, composer, elem_a, elem_b):
        """Parallel composition is commutative in μ."""
        ab = composer.compose(elem_a, elem_b, interference=0.3)
        ba = composer.compose(elem_b, elem_a, interference=0.3)
        assert math.isclose(ab.mu, ba.mu, abs_tol=1e-12)

    def test_compose_commutative_sigma_sq(self, composer, elem_a, elem_b):
        """Parallel composition is commutative in σ²."""
        ab = composer.compose(elem_a, elem_b, interference=0.5)
        ba = composer.compose(elem_b, elem_a, interference=0.5)
        assert math.isclose(ab.sigma_sq, ba.sigma_sq, abs_tol=1e-12)

    def test_compose_commutative_lambda(self, composer, elem_a, elem_b):
        """Parallel composition is commutative in λ."""
        ab = composer.compose(elem_a, elem_b, interference=0.7)
        ba = composer.compose(elem_b, elem_a, interference=0.7)
        assert math.isclose(ab.lambda_, ba.lambda_, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Group composition
# ---------------------------------------------------------------------------

class TestComposeGroup:
    """Tests for compose_group — n-ary parallel composition."""

    def test_group_two_equals_compose(self, composer, elem_a, elem_b):
        """Grouping two elements equals a single compose call."""
        group = composer.compose_group([elem_a, elem_b], interference=0.3)
        direct = composer.compose(elem_a, elem_b, interference=0.3)
        assert math.isclose(group.mu, direct.mu, rel_tol=1e-10)

    def test_group_three_elements(self, composer, elem_a, elem_b, elem_c):
        """Grouping three elements produces a valid result."""
        result = composer.compose_group([elem_a, elem_b, elem_c], interference=0.2)
        assert result.is_valid
        assert result.mu >= max(elem_a.mu, elem_b.mu, elem_c.mu) - 1e-12

    def test_group_empty_returns_zero(self, composer):
        """Grouping an empty list returns the zero element."""
        result = composer.compose_group([])
        assert result == CostElement.zero()

    def test_group_single_returns_element(self, composer, elem_a):
        """Grouping a single element returns that element."""
        result = composer.compose_group([elem_a], interference=0.5)
        assert result.mu == elem_a.mu


# ---------------------------------------------------------------------------
# MRT channel-based composition
# ---------------------------------------------------------------------------

class TestComposeWithChannels:
    """Tests for compose_with_channels — MRT-based interference lookup."""

    def test_channels_same_modality_high_interference(self, composer):
        """Same-modality tasks (visual_focal + visual_ambient) have high interference."""
        a = CostElement(mu=1.5, sigma_sq=0.3, kappa=0.1, lambda_=0.03)
        b = CostElement(mu=1.0, sigma_sq=0.2, kappa=0.05, lambda_=0.02)
        result = composer.compose_with_channels([a, b], ["visual_focal", "visual_ambient"])
        no_interf = composer.compose(a, b, interference=0.0)
        assert result.mu > no_interf.mu

    def test_channels_cross_modal_low_interference(self, composer):
        """Cross-modal tasks (visual + auditory) have low interference."""
        a = CostElement(mu=1.5, sigma_sq=0.3, kappa=0.1, lambda_=0.03)
        b = CostElement(mu=1.0, sigma_sq=0.2, kappa=0.05, lambda_=0.02)
        cross = composer.compose_with_channels([a, b], ["visual_focal", "auditory_verbal"])
        same = composer.compose_with_channels([a, b], ["visual_focal", "visual_ambient"])
        assert cross.mu < same.mu

    def test_channels_mismatch_length_raises(self, composer, elem_a, elem_b):
        """Mismatched element and channel list lengths raise ValueError."""
        with pytest.raises(ValueError, match="match"):
            composer.compose_with_channels([elem_a, elem_b], ["visual_focal"])

    def test_channels_empty_returns_zero(self, composer):
        """Empty element/channel lists return zero."""
        result = composer.compose_with_channels([], [])
        assert result == CostElement.zero()


# ---------------------------------------------------------------------------
# Interference factor lookup
# ---------------------------------------------------------------------------

class TestInterferenceFactor:
    """Tests for the static interference_factor method."""

    def test_same_channel_full_interference(self):
        """Same channel → interference = 1.0."""
        for ch in CHANNELS:
            eta = ParallelComposer.interference_factor(ch, ch)
            assert math.isclose(eta, 1.0)

    def test_known_pair_value(self):
        """visual_focal + auditory_verbal → 0.2 (from Wickens MRT)."""
        eta = ParallelComposer.interference_factor("visual_focal", "auditory_verbal")
        assert math.isclose(eta, 0.2, abs_tol=1e-10)

    def test_symmetric(self):
        """interference_factor(a, b) == interference_factor(b, a)."""
        eta_ab = ParallelComposer.interference_factor("visual_focal", "response_manual")
        eta_ba = ParallelComposer.interference_factor("response_manual", "visual_focal")
        assert math.isclose(eta_ab, eta_ba)

    def test_unknown_channel_raises(self):
        """Unknown channel name raises KeyError."""
        with pytest.raises(KeyError):
            ParallelComposer.interference_factor("visual_focal", "nonexistent_channel")

    def test_all_channels_defined(self):
        """All 8 MRT channels are present in CHANNELS."""
        assert len(CHANNELS) == 8
        expected = {
            "visual_focal", "visual_ambient",
            "auditory_verbal", "auditory_spatial",
            "cognitive_spatial", "cognitive_verbal",
            "response_manual", "response_vocal",
        }
        assert set(CHANNELS) == expected


# ---------------------------------------------------------------------------
# INTERFERENCE_MATRIX
# ---------------------------------------------------------------------------

class TestInterferenceMatrix:
    """Tests for the global INTERFERENCE_MATRIX constant."""

    def test_matrix_populated(self):
        """INTERFERENCE_MATRIX has entries for all channel pairs."""
        for ch_a in CHANNELS:
            for ch_b in CHANNELS:
                assert (ch_a, ch_b) in INTERFERENCE_MATRIX

    def test_matrix_values_in_range(self):
        """All interference values are in [0, 1]."""
        for (_, _), val in INTERFERENCE_MATRIX.items():
            assert 0.0 <= val <= 1.0

    def test_matrix_symmetric(self):
        """Matrix is symmetric: M[a,b] == M[b,a]."""
        for ch_a in CHANNELS:
            for ch_b in CHANNELS:
                assert math.isclose(
                    INTERFERENCE_MATRIX[(ch_a, ch_b)],
                    INTERFERENCE_MATRIX[(ch_b, ch_a)],
                )

    def test_diagonal_is_one(self):
        """Diagonal entries (same channel) are 1.0."""
        for ch in CHANNELS:
            assert math.isclose(INTERFERENCE_MATRIX[(ch, ch)], 1.0)


# ---------------------------------------------------------------------------
# Capacity coefficient
# ---------------------------------------------------------------------------

class TestCapacityCoefficient:
    """Tests for capacity_coefficient — dual-task capacity analysis."""

    def test_unlimited_capacity(self):
        """When rt_ab = harmonic mean of rt_a, rt_b → C ≈ 1 (unlimited)."""
        rt_a, rt_b = 100.0, 200.0
        rt_ab = 1.0 / (1.0 / rt_a + 1.0 / rt_b)
        c = ParallelComposer.capacity_coefficient(rt_a, rt_b, rt_ab)
        assert math.isclose(c, 1.0, rel_tol=1e-8)

    def test_limited_capacity(self):
        """When rt_ab is large (slow dual-task), C < 1."""
        c = ParallelComposer.capacity_coefficient(100.0, 200.0, 500.0)
        assert c < 1.0

    def test_super_capacity(self):
        """When rt_ab is very short (parallel speedup), C > 1."""
        c = ParallelComposer.capacity_coefficient(100.0, 200.0, 50.0)
        assert c > 1.0

    def test_zero_rt_returns_one(self):
        """Edge case: zero reaction times return capacity 1.0."""
        c = ParallelComposer.capacity_coefficient(0.0, 100.0, 100.0)
        assert math.isclose(c, 1.0)


# ---------------------------------------------------------------------------
# Interference estimation
# ---------------------------------------------------------------------------

class TestEstimateInterference:
    """Tests for estimate_interference — reverse-engineering η from data."""

    def test_estimate_zero_interference(self, composer, elem_a, elem_b):
        """When dual-task cost equals max, estimated η ≈ 0."""
        combined = composer.compose(elem_a, elem_b, interference=0.0)
        eta = ParallelComposer.estimate_interference(elem_a, elem_b, combined)
        assert math.isclose(eta, 0.0, abs_tol=1e-6)

    def test_estimate_full_interference(self, composer, elem_a, elem_b):
        """When dual-task cost equals sum, estimated η ≈ 1."""
        combined = composer.compose(elem_a, elem_b, interference=1.0)
        eta = ParallelComposer.estimate_interference(elem_a, elem_b, combined)
        assert math.isclose(eta, 1.0, abs_tol=1e-6)

    def test_estimate_partial_interference(self, composer, elem_a, elem_b):
        """Round-tripping through compose/estimate recovers the η value."""
        for target_eta in [0.2, 0.5, 0.8]:
            combined = composer.compose(elem_a, elem_b, interference=target_eta)
            recovered = ParallelComposer.estimate_interference(elem_a, elem_b, combined)
            assert math.isclose(recovered, target_eta, abs_tol=1e-6)

    def test_estimate_clamped(self):
        """Estimated interference is clamped to [0, 1]."""
        a = CostElement(mu=1.0, sigma_sq=0.1, kappa=0.0, lambda_=0.01)
        b = CostElement(mu=1.0, sigma_sq=0.1, kappa=0.0, lambda_=0.01)
        huge = CostElement(mu=100.0, sigma_sq=1.0, kappa=0.0, lambda_=0.1)
        eta = ParallelComposer.estimate_interference(a, b, huge)
        assert 0.0 <= eta <= 1.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for interference validation."""

    def test_negative_interference_raises(self, composer, elem_a, elem_b):
        """Interference < 0 raises ValueError."""
        with pytest.raises(ValueError, match="[Ii]nterference"):
            composer.compose(elem_a, elem_b, interference=-0.1)

    def test_interference_above_one_raises(self, composer, elem_a, elem_b):
        """Interference > 1 raises ValueError."""
        with pytest.raises(ValueError, match="[Ii]nterference"):
            composer.compose(elem_a, elem_b, interference=1.5)

    def test_interference_boundary_zero_ok(self, composer, elem_a, elem_b):
        """Interference exactly 0.0 is valid."""
        result = composer.compose(elem_a, elem_b, interference=0.0)
        assert result.is_valid

    def test_interference_boundary_one_ok(self, composer, elem_a, elem_b):
        """Interference exactly 1.0 is valid."""
        result = composer.compose(elem_a, elem_b, interference=1.0)
        assert result.is_valid
