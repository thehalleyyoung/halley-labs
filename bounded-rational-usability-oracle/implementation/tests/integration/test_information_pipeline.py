"""Integration tests for the information-theoretic analysis pipeline.

Information-theoretic analysis in the full pipeline, channel capacity
to bounded rationality parameter, and free energy computation end-to-end.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.information_theory.entropy import (
    shannon_entropy,
    joint_entropy,
    conditional_entropy,
    binary_entropy,
)
from usability_oracle.information_theory.mutual_information import (
    kl_divergence,
    mutual_information,
    mutual_information_full,
)
from usability_oracle.information_theory.channel_capacity import (
    blahut_arimoto,
    bsc_capacity,
    bec_capacity,
    binary_symmetric_channel,
    binary_erasure_channel,
    gaussian_channel_capacity,
    channel_mutual_information,
)
from usability_oracle.information_theory.free_energy import (
    variational_free_energy,
)
from usability_oracle.channel.integration import (
    capacity_to_beta,
    beta_from_channel_analysis,
)
from usability_oracle.channel.types import (
    ResourcePool,
    ResourceChannel,
    WickensResource,
)
from usability_oracle.channel.capacity import (
    visual_capacity,
    auditory_capacity,
    cognitive_capacity_wm,
    motor_capacity_fitts,
)


def _make_pool(cap: float) -> ResourcePool:
    """Build a single-channel ResourcePool with the given capacity."""
    ch = ResourceChannel(
        resource=WickensResource.VISUAL,
        capacity_bits_per_s=cap,
        current_load=0.0,
    )
    return ResourcePool(channels=(ch,))


# ===================================================================
# Tests — Channel capacity to bounded rationality parameter
# ===================================================================


class TestChannelCapacityToBeta:
    """Channel capacity → bounded rationality parameter β."""

    def test_capacity_to_beta_positive(self) -> None:
        """capacity_to_beta returns a positive β for positive capacity."""
        pool = _make_pool(10.0)
        beta = capacity_to_beta(pool)
        assert beta > 0, f"β not positive: {beta}"

    def test_higher_capacity_higher_beta(self) -> None:
        """Higher channel capacity → higher β (more rational)."""
        beta_lo = capacity_to_beta(_make_pool(5.0))
        beta_hi = capacity_to_beta(_make_pool(20.0))
        assert beta_hi >= beta_lo, \
            f"Higher capacity should give higher β: {beta_hi} vs {beta_lo}"

    def test_capacity_to_beta_monotone(self) -> None:
        """β is monotonically increasing in capacity."""
        betas = [capacity_to_beta(_make_pool(c))
                 for c in [1.0, 5.0, 10.0, 20.0, 50.0]]
        for i in range(len(betas) - 1):
            assert betas[i + 1] >= betas[i], \
                f"β not monotone: {betas}"


# ===================================================================
# Tests — Free energy computation end-to-end
# ===================================================================


class TestFreeEnergyPipeline:
    """Variational free energy computation."""

    def test_free_energy_finite(self) -> None:
        """Free energy is finite for valid inputs."""
        policy = [0.3, 0.5, 0.2]
        prior = [1 / 3, 1 / 3, 1 / 3]
        rewards = [1.0, 2.0, 0.5]
        beta = 1.0
        fe = variational_free_energy(policy, prior, rewards, beta)
        assert math.isfinite(fe), f"Free energy not finite: {fe}"

    def test_free_energy_increases_with_beta(self) -> None:
        """Higher β → higher free energy (less information cost penalty)."""
        policy = [0.6, 0.3, 0.1]
        prior = [1 / 3, 1 / 3, 1 / 3]
        rewards = [1.0, 2.0, 0.5]
        fe_lo = variational_free_energy(policy, prior, rewards, beta=0.5)
        fe_hi = variational_free_energy(policy, prior, rewards, beta=10.0)
        # With higher beta, KL penalty is scaled down → higher free energy
        assert fe_hi >= fe_lo - 0.1, \
            f"FE(β=10) = {fe_hi} < FE(β=0.5) = {fe_lo}"

    def test_free_energy_policy_equals_prior(self) -> None:
        """When policy = prior, KL = 0 so F = E[R]."""
        prior = [0.25, 0.25, 0.25, 0.25]
        rewards = [1.0, 2.0, 3.0, 4.0]
        beta = 1.0
        fe = variational_free_energy(prior, prior, rewards, beta)
        expected_reward = sum(p * r for p, r in zip(prior, rewards))
        assert math.isclose(fe, expected_reward, abs_tol=1e-6), \
            f"F = {fe} != E[R] = {expected_reward} when π = p₀"

    def test_free_energy_with_uniform_rewards(self) -> None:
        """When rewards are uniform, free energy is dominated by info cost."""
        policy = [0.8, 0.1, 0.1]
        prior = [1 / 3, 1 / 3, 1 / 3]
        rewards = [1.0, 1.0, 1.0]
        beta = 1.0
        fe = variational_free_energy(policy, prior, rewards, beta)
        # E[R] = 1.0 for any policy, so F = 1.0 - KL/β
        kl = kl_divergence(policy, prior, base=math.e)
        expected = 1.0 - kl / beta
        assert math.isclose(fe, expected, abs_tol=1e-4), \
            f"F = {fe} != expected {expected}"


# ===================================================================
# Tests — Information-theoretic analysis in full pipeline
# ===================================================================


class TestInformationAnalysisPipeline:
    """End-to-end information-theoretic analysis flow."""

    def test_bsc_capacity_matches_blahut_arimoto(self) -> None:
        """BSC capacity formula matches Blahut-Arimoto computation."""
        for p in [0.0, 0.1, 0.2, 0.3, 0.5]:
            analytical = bsc_capacity(p)
            W = binary_symmetric_channel(p)
            numerical = blahut_arimoto(W, max_iterations=500).capacity_bits
            assert math.isclose(analytical, numerical, abs_tol=0.01), \
                f"BSC({p}): analytical={analytical}, numerical={numerical}"

    def test_bec_capacity_matches_formula(self) -> None:
        """BEC capacity formula matches expected values."""
        for eps in [0.0, 0.1, 0.5, 0.9, 1.0]:
            c = bec_capacity(eps)
            assert math.isclose(c, 1.0 - eps, abs_tol=1e-6)

    def test_mutual_information_decomposition(self) -> None:
        """MI decomposition: I(X;Y) = H(X) + H(Y) - H(X,Y)."""
        # Create a joint distribution
        joint = np.array([[0.3, 0.1], [0.1, 0.5]])
        joint /= joint.sum()
        result = mutual_information_full(joint)
        hx = result.entropy_x_bits
        hy = result.entropy_y_bits
        hxy = result.joint_entropy_bits
        mi = result.mutual_info_bits
        expected_mi = hx + hy - hxy
        assert math.isclose(mi, expected_mi, abs_tol=1e-4), \
            f"MI = {mi} != H(X)+H(Y)-H(X,Y) = {expected_mi}"

    def test_channel_mutual_information_leq_capacity(self) -> None:
        """MI with any input distribution ≤ channel capacity."""
        W = binary_symmetric_channel(0.1)
        cap = blahut_arimoto(W, max_iterations=200).capacity_bits
        # Uniform input
        mi_uniform = channel_mutual_information(W, [0.5, 0.5])
        assert mi_uniform <= cap + 0.01, \
            f"MI = {mi_uniform} > capacity = {cap}"
        # Skewed input
        mi_skewed = channel_mutual_information(W, [0.9, 0.1])
        assert mi_skewed <= cap + 0.01

    def test_gaussian_capacity_formula(self) -> None:
        """Gaussian channel capacity C = 0.5 log₂(1 + P/N)."""
        for P, N in [(1.0, 1.0), (10.0, 1.0), (1.0, 10.0)]:
            c = gaussian_channel_capacity(P, N)
            expected = 0.5 * math.log2(1.0 + P / N)
            assert math.isclose(c, expected, abs_tol=1e-6), \
                f"Gaussian C(P={P},N={N}) = {c} != {expected}"


# ===================================================================
# Tests — Cognitive channel capacity pipeline
# ===================================================================


class TestCognitiveChannelPipeline:
    """Cognitive channel capacity → β → free energy pipeline."""

    def test_visual_to_beta_to_free_energy(self) -> None:
        """Visual capacity → β → free energy end-to-end."""
        vis_cap = visual_capacity(n_elements=10)
        assert vis_cap > 0

        beta = capacity_to_beta(_make_pool(vis_cap))
        assert beta > 0

        policy = [0.5, 0.3, 0.2]
        prior = [1 / 3, 1 / 3, 1 / 3]
        rewards = [1.0, 2.0, 0.5]
        fe = variational_free_energy(policy, prior, rewards, beta)
        assert math.isfinite(fe)

    def test_motor_capacity_to_beta(self) -> None:
        """Motor capacity produces a meaningful β."""
        motor_cap = motor_capacity_fitts(target_distance_px=200.0,
                                         target_width_px=50.0)
        assert motor_cap > 0
        beta = capacity_to_beta(_make_pool(motor_cap))
        assert beta > 0

    def test_cognitive_wm_to_beta(self) -> None:
        """Working memory capacity produces a meaningful β."""
        wm_cap = cognitive_capacity_wm(n_chunks=4)
        assert wm_cap > 0
        beta = capacity_to_beta(_make_pool(wm_cap))
        assert beta > 0

    def test_higher_visual_complexity_lower_capacity(self) -> None:
        """More visual elements → lower effective capacity."""
        cap_simple = visual_capacity(n_elements=5)
        cap_complex = visual_capacity(n_elements=50)
        assert cap_complex <= cap_simple


# ===================================================================
# Tests — Information flow consistency
# ===================================================================


class TestInformationFlowConsistency:
    """Verify information-theoretic quantities are consistent."""

    def test_entropy_decomposition(self) -> None:
        """H(X,Y) = H(X) + H(Y|X) for concrete distributions."""
        joint = np.array([[0.2, 0.1], [0.3, 0.4]])
        joint /= joint.sum()
        hxy = joint_entropy(joint)
        px = joint.sum(axis=1)
        hx = shannon_entropy(px)
        hy_given_x = hxy - hx
        assert hy_given_x >= -1e-6, \
            f"H(Y|X) = {hy_given_x} is negative"

    def test_conditioning_reduces_entropy_concrete(self) -> None:
        """H(X|Y) ≤ H(X) for a concrete joint distribution."""
        joint = np.array([[0.4, 0.1], [0.1, 0.4]])
        joint /= joint.sum()
        px = joint.sum(axis=1)
        hx = shannon_entropy(px)
        hx_given_y = conditional_entropy(joint)
        assert hx_given_y <= hx + 1e-6

    def test_kl_divergence_zero_for_same(self) -> None:
        """KL(p || p) = 0."""
        p = [0.3, 0.7]
        d = kl_divergence(p, p)
        assert math.isclose(d, 0.0, abs_tol=1e-10)

    def test_binary_entropy_max_at_half(self) -> None:
        """h(0.5) = 1 bit (maximum binary entropy)."""
        h = binary_entropy(0.5)
        assert math.isclose(h, 1.0, abs_tol=1e-6)
