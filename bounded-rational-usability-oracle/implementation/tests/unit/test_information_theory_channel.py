"""Unit tests for usability_oracle.information_theory.channel_capacity.

Tests cover Blahut-Arimoto convergence, BSC capacity = 1 - H(p),
BEC capacity = 1 - ε, cost-constrained capacity, and cognitive
channel models (Fitts, Hick-Hyman, visual search).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.information_theory.channel_capacity import (
    bec_capacity,
    binary_erasure_channel,
    binary_symmetric_channel,
    blahut_arimoto,
    blahut_arimoto_cost_constrained,
    bsc_capacity,
    channel_mutual_information,
    fitts_channel_capacity,
    gaussian_channel_capacity,
    hick_hyman_rate,
    human_information_rate,
    visual_search_capacity,
    z_channel,
    z_channel_capacity,
)
from usability_oracle.information_theory.entropy import binary_entropy


# ------------------------------------------------------------------ #
# Blahut-Arimoto convergence
# ------------------------------------------------------------------ #


class TestBlahutArimoto:
    """Tests for the Blahut-Arimoto algorithm."""

    def test_noiseless_channel_capacity_1bit(self) -> None:
        """Identity channel [[1,0],[0,1]] has C = 1 bit."""
        W = np.eye(2)
        result = blahut_arimoto(W)
        assert result.capacity_bits == pytest.approx(1.0, rel=1e-6)
        assert result.converged

    def test_noiseless_channel_n(self) -> None:
        """Identity channel of size n has C = log2(n)."""
        for n in [2, 4, 8]:
            W = np.eye(n)
            result = blahut_arimoto(W)
            assert result.capacity_bits == pytest.approx(
                math.log2(n), rel=1e-5
            )

    def test_totally_noisy_channel_zero(self) -> None:
        """Useless channel (identical rows) has C = 0."""
        W = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = blahut_arimoto(W)
        assert result.capacity_bits == pytest.approx(0.0, abs=1e-6)

    def test_convergence_flag(self) -> None:
        W = np.eye(2)
        result = blahut_arimoto(W, tolerance=1e-10, max_iterations=1000)
        assert result.converged is True

    def test_bounds_gap(self) -> None:
        """Upper and lower bounds should be close at convergence."""
        W = binary_symmetric_channel(0.1)
        result = blahut_arimoto(W, tolerance=1e-10)
        assert result.upper_bound - result.lower_bound < 1e-8

    def test_optimal_input_dist_sums_to_one(self) -> None:
        W = np.eye(3)
        result = blahut_arimoto(W)
        assert sum(result.optimal_input_distribution) == pytest.approx(1.0, abs=1e-8)

    def test_custom_initial_distribution(self) -> None:
        W = np.eye(2)
        result = blahut_arimoto(W, initial_distribution=[0.9, 0.1])
        assert result.capacity_bits == pytest.approx(1.0, rel=1e-5)

    def test_asymmetric_channel(self) -> None:
        """Non-symmetric channel should still converge."""
        W = np.array([[0.9, 0.1], [0.3, 0.7]])
        result = blahut_arimoto(W)
        assert result.converged
        assert result.capacity_bits > 0
        assert result.capacity_bits < 1.0


# ------------------------------------------------------------------ #
# BSC capacity = 1 - H(p)
# ------------------------------------------------------------------ #


class TestBSCCapacity:
    """Tests for the Binary Symmetric Channel."""

    def test_bsc_noiseless(self) -> None:
        """BSC with p=0 has C=1."""
        assert bsc_capacity(0.0) == pytest.approx(1.0)

    def test_bsc_useless(self) -> None:
        """BSC with p=0.5 has C=0."""
        assert bsc_capacity(0.5) == pytest.approx(0.0, abs=1e-10)

    def test_bsc_formula(self) -> None:
        """C = 1 - H(p) for various p."""
        for p in [0.01, 0.1, 0.2, 0.3, 0.4]:
            expected = 1.0 - binary_entropy(p)
            assert bsc_capacity(p) == pytest.approx(expected, rel=1e-9)

    def test_bsc_via_blahut_arimoto(self) -> None:
        """BA on BSC transition matrix should match closed-form."""
        for p in [0.1, 0.25, 0.4]:
            W = binary_symmetric_channel(p)
            result = blahut_arimoto(W, tolerance=1e-12)
            assert result.capacity_bits == pytest.approx(bsc_capacity(p), rel=1e-4)

    def test_bsc_uniform_optimal_input(self) -> None:
        """BSC optimal input is uniform."""
        W = binary_symmetric_channel(0.2)
        result = blahut_arimoto(W, tolerance=1e-12)
        np.testing.assert_allclose(
            result.optimal_input_distribution, [0.5, 0.5], atol=1e-4,
        )

    def test_bsc_transition_matrix_shape(self) -> None:
        W = binary_symmetric_channel(0.3)
        assert W.shape == (2, 2)

    def test_bsc_rows_sum_to_one(self) -> None:
        W = binary_symmetric_channel(0.3)
        np.testing.assert_allclose(W.sum(axis=1), [1.0, 1.0])

    def test_bsc_invalid_p_raises(self) -> None:
        with pytest.raises(ValueError):
            binary_symmetric_channel(-0.1)
        with pytest.raises(ValueError):
            binary_symmetric_channel(1.1)

    def test_bsc_symmetric_around_half(self) -> None:
        """C(p) = C(1-p) by symmetry."""
        assert bsc_capacity(0.1) == pytest.approx(bsc_capacity(0.9), rel=1e-9)


# ------------------------------------------------------------------ #
# BEC capacity = 1 - ε
# ------------------------------------------------------------------ #


class TestBECCapacity:
    """Tests for the Binary Erasure Channel."""

    def test_bec_noiseless(self) -> None:
        assert bec_capacity(0.0) == pytest.approx(1.0)

    def test_bec_fully_erased(self) -> None:
        assert bec_capacity(1.0) == pytest.approx(0.0)

    def test_bec_formula(self) -> None:
        for eps in [0.1, 0.3, 0.5, 0.7]:
            assert bec_capacity(eps) == pytest.approx(1.0 - eps, rel=1e-12)

    def test_bec_transition_matrix_shape(self) -> None:
        W = binary_erasure_channel(0.3)
        assert W.shape == (2, 3)  # 2 inputs, 3 outputs

    def test_bec_rows_sum_to_one(self) -> None:
        W = binary_erasure_channel(0.4)
        np.testing.assert_allclose(W.sum(axis=1), [1.0, 1.0])

    def test_bec_invalid_eps_raises(self) -> None:
        with pytest.raises(ValueError):
            bec_capacity(-0.1)
        with pytest.raises(ValueError):
            bec_capacity(1.1)

    def test_bec_via_blahut_arimoto(self) -> None:
        """BA on BEC transition matrix should match closed-form."""
        eps = 0.3
        W = binary_erasure_channel(eps)
        result = blahut_arimoto(W, tolerance=1e-12)
        assert result.capacity_bits == pytest.approx(bec_capacity(eps), rel=1e-3)


# ------------------------------------------------------------------ #
# Z-channel and Gaussian channel
# ------------------------------------------------------------------ #


class TestOtherChannels:
    """Tests for Z-channel and Gaussian channel."""

    def test_z_channel_shape(self) -> None:
        W = z_channel(0.3)
        assert W.shape == (2, 2)

    def test_z_channel_rows_sum(self) -> None:
        W = z_channel(0.2)
        np.testing.assert_allclose(W.sum(axis=1), [1.0, 1.0])

    def test_z_channel_capacity_positive(self) -> None:
        cap = z_channel_capacity(0.2)
        assert cap > 0

    def test_z_channel_noiseless(self) -> None:
        """Z-channel with p=0 is noiseless → C=1."""
        cap = z_channel_capacity(0.0)
        assert cap == pytest.approx(1.0, rel=1e-4)

    def test_gaussian_capacity(self) -> None:
        """AWGN capacity = 0.5 log2(1 + P/N)."""
        P, N = 10.0, 1.0
        expected = 0.5 * math.log2(1 + P / N)
        assert gaussian_channel_capacity(P, N) == pytest.approx(expected, rel=1e-12)

    def test_gaussian_zero_power(self) -> None:
        assert gaussian_channel_capacity(0.0, 1.0) == pytest.approx(0.0)

    def test_gaussian_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            gaussian_channel_capacity(-1.0, 1.0)
        with pytest.raises(ValueError):
            gaussian_channel_capacity(1.0, 0.0)


# ------------------------------------------------------------------ #
# Cost-constrained capacity
# ------------------------------------------------------------------ #


class TestCostConstrainedCapacity:
    """Tests for Blahut-Arimoto with per-symbol cost constraints."""

    def test_unconstrained_matches_ba(self) -> None:
        """With high max_cost, result should match unconstrained BA."""
        W = binary_symmetric_channel(0.1)
        result_unconstrained = blahut_arimoto(W)
        result_constrained = blahut_arimoto_cost_constrained(
            W, cost_vector=[1.0, 1.0], max_cost=10.0,
        )
        assert result_constrained.capacity_bits == pytest.approx(
            result_unconstrained.capacity_bits, rel=1e-2,
        )

    def test_cost_constraint_reduces_capacity(self) -> None:
        """Tighter cost should reduce capacity."""
        W = np.eye(3)
        costs = [0.0, 1.0, 2.0]
        cap_loose = blahut_arimoto_cost_constrained(W, costs, max_cost=5.0)
        cap_tight = blahut_arimoto_cost_constrained(W, costs, max_cost=0.5)
        assert cap_tight.capacity_bits <= cap_loose.capacity_bits + 1e-6

    def test_cost_constrained_returns_capacity_result(self) -> None:
        W = binary_symmetric_channel(0.2)
        result = blahut_arimoto_cost_constrained(
            W, [1.0, 2.0], max_cost=1.5,
        )
        assert hasattr(result, "capacity_bits")
        assert hasattr(result, "optimal_input_distribution")


# ------------------------------------------------------------------ #
# Cognitive channel models
# ------------------------------------------------------------------ #


class TestCognitiveChannelModels:
    """Tests for Fitts, Hick-Hyman, and visual search models."""

    def test_fitts_throughput(self) -> None:
        """Fitts throughput = ID / MT."""
        w, a, mt = 10.0, 200.0, 0.5
        tp = fitts_channel_capacity(w, a, mt)
        id_bits = math.log2(2 * a / w)
        assert tp == pytest.approx(id_bits / mt, rel=1e-9)

    def test_fitts_positive(self) -> None:
        assert fitts_channel_capacity(10, 100, 0.3) > 0

    def test_fitts_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            fitts_channel_capacity(0, 100, 0.3)
        with pytest.raises(ValueError):
            fitts_channel_capacity(10, 0, 0.3)
        with pytest.raises(ValueError):
            fitts_channel_capacity(10, 100, 0)

    def test_hick_hyman_rate_positive(self) -> None:
        rate = hick_hyman_rate(4, 0.5)
        assert rate > 0

    def test_hick_hyman_rate_formula(self) -> None:
        """Rate = H / RT = log2(n) / RT for equal probs."""
        n, rt = 8, 1.0
        rate = hick_hyman_rate(n, rt)
        assert rate == pytest.approx(math.log2(n) / rt, rel=1e-9)

    def test_hick_hyman_custom_probs(self) -> None:
        rate = hick_hyman_rate(3, 0.5, equal_probability=False,
                               probabilities=[0.5, 0.3, 0.2])
        assert rate > 0

    def test_visual_search_positive(self) -> None:
        assert visual_search_capacity(10, 2.0) > 0

    def test_visual_search_invalid(self) -> None:
        with pytest.raises(ValueError):
            visual_search_capacity(0, 1.0)

    def test_human_information_rate_known_types(self) -> None:
        for task_type in ["choice_reaction", "reading", "typing", "pointing"]:
            rate = human_information_rate(task_type)
            assert rate > 0

    def test_human_information_rate_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            human_information_rate("juggling")

    def test_human_rate_error_reduces(self) -> None:
        """Error rate > 0 should reduce effective rate."""
        rate_perfect = human_information_rate("typing", error_rate=0.0)
        rate_error = human_information_rate("typing", error_rate=0.1)
        assert rate_error < rate_perfect

    def test_channel_mutual_information(self) -> None:
        """I(X;Y) for a known channel + input distribution."""
        W = np.eye(2)
        px = np.array([0.5, 0.5])
        mi = channel_mutual_information(W, px)
        assert mi == pytest.approx(1.0, rel=1e-6)
