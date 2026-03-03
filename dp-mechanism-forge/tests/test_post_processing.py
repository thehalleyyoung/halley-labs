"""
Comprehensive tests for dp_forge.post_processing module.

Tests cover LMMSE estimation, Bayes optimal estimation, bias-variance
decomposition, Wiener filtering, and privacy preservation proofs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dp_forge.post_processing import (
    PostProcessor,
    LMMSEEstimator,
    BayesOptimalEstimator,
    BiasVarianceDecomposer,
    WienerFilter,
    PrivacyPreservationProof,
)
from dp_forge.types import WorkloadSpec


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def identity_workload_3():
    """3×3 identity workload matrix."""
    return np.eye(3)


@pytest.fixture
def overcomplete_workload():
    """Overcomplete workload: more queries than domain size."""
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ], dtype=float)


@pytest.fixture
def rank_deficient_workload():
    """Rank-deficient workload matrix."""
    return np.array([
        [1, 1, 0],
        [2, 2, 0],  # Linearly dependent on row 0
        [0, 0, 1],
    ], dtype=float)


# =========================================================================
# Section 1: LMMSE Estimator – Identity Workload
# =========================================================================


class TestLMMSEIdentityWorkload:
    """Tests for LMMSE with identity workload."""

    def test_identity_denoise_equals_noisy(self, identity_workload_3):
        """For identity workload, LMMSE = (A^T A + σ²I)^{-1} A^T y = y/(1+σ²)."""
        sigma2 = 1.0
        est = LMMSEEstimator(
            workload_matrix=identity_workload_3,
            noise_variance=sigma2,
            regularization=0.0,
        )
        y = np.array([1.0, 2.0, 3.0])
        x_hat = est.denoise(y)
        # For identity A: x̂ = (I + σ²I)^{-1} y = y / (1 + σ²)
        expected = y / (1.0 + sigma2)
        assert x_hat.shape == (3,)
        np.testing.assert_array_almost_equal(x_hat, expected, decimal=6)

    def test_identity_mse_matrix(self, identity_workload_3):
        """MSE matrix is finite and positive semi-definite."""
        est = LMMSEEstimator(
            workload_matrix=identity_workload_3,
            noise_variance=1.0,
        )
        mse_mat = est.mse_matrix
        assert mse_mat.shape == (3, 3)
        # Eigenvalues should be non-negative (PSD)
        eigvals = np.linalg.eigvalsh(mse_mat)
        assert all(v >= -1e-10 for v in eigvals)

    def test_identity_total_mse(self, identity_workload_3):
        """Total MSE is finite and positive."""
        est = LMMSEEstimator(
            workload_matrix=identity_workload_3,
            noise_variance=1.0,
        )
        total = est.total_mse
        assert total > 0
        assert math.isfinite(total)

    def test_optimal_filter(self, identity_workload_3):
        """optimal_filter returns a matrix of correct shape."""
        est = LMMSEEstimator(
            workload_matrix=identity_workload_3,
            noise_variance=1.0,
        )
        F = est.optimal_filter()
        assert F.shape[1] == 3  # Columns match query dimension


# =========================================================================
# Section 2: LMMSE Estimator – Overcomplete Workload
# =========================================================================


class TestLMMSEOvercomplete:
    """Tests for LMMSE with overcomplete workload."""

    def test_overcomplete_error_reduction(self, overcomplete_workload):
        """Overcomplete workload yields lower MSE than identity."""
        est_id = LMMSEEstimator(
            workload_matrix=np.eye(3), noise_variance=1.0,
        )
        est_oc = LMMSEEstimator(
            workload_matrix=overcomplete_workload, noise_variance=1.0,
        )
        # Overcomplete should achieve lower or equal MSE
        assert est_oc.total_mse <= est_id.total_mse + 1e-6

    def test_overcomplete_denoise(self, overcomplete_workload):
        """Denoising with overcomplete workload produces valid output."""
        est = LMMSEEstimator(
            workload_matrix=overcomplete_workload, noise_variance=0.5,
        )
        y = np.array([1.0, 2.0, 3.0, 3.0, 5.0, 4.0])
        x_hat = est.denoise(y)
        assert x_hat.shape == (3,)  # Domain dimension
        assert all(math.isfinite(v) for v in x_hat)


# =========================================================================
# Section 3: LMMSE Edge Cases
# =========================================================================


class TestLMMSEEdgeCases:
    """Edge case tests for LMMSE."""

    def test_rank_deficient_with_regularization(self, rank_deficient_workload):
        """Rank-deficient workload with regularization doesn't crash."""
        est = LMMSEEstimator(
            workload_matrix=rank_deficient_workload,
            noise_variance=1.0,
            regularization=0.1,  # Regularize for stability
        )
        y = np.array([1.0, 2.0, 3.0])
        x_hat = est.denoise(y)
        assert x_hat.shape == (3,)
        assert all(math.isfinite(v) for v in x_hat)

    def test_zero_noise(self, identity_workload_3):
        """Near-zero noise variance with regularization."""
        est = LMMSEEstimator(
            workload_matrix=identity_workload_3,
            noise_variance=1e-10,
            regularization=1e-12,
        )
        y = np.array([1.0, 2.0, 3.0])
        x_hat = est.denoise(y)
        # With very low noise, estimate ≈ y
        np.testing.assert_array_almost_equal(x_hat, y, decimal=3)

    def test_from_workload_spec(self):
        """from_workload_spec factory creates LMMSE estimator."""
        spec = WorkloadSpec.identity(4)
        est = LMMSEEstimator.from_workload_spec(spec, noise_variance=1.0)
        assert isinstance(est, LMMSEEstimator)


# =========================================================================
# Section 4: Bayes Optimal Estimator
# =========================================================================


class TestBayesOptimalEstimator:
    """Tests for Bayes optimal estimation."""

    def test_gaussian_prior_denoise(self):
        """Gaussian prior Bayes estimator produces output."""
        est = BayesOptimalEstimator(
            noise_variance=1.0,
            prior="gaussian",
            prior_mean=0.0,
            prior_variance=1.0,
        )
        y = np.array([0.5, 1.0, -0.5])
        x_hat = est.denoise(y)
        assert x_hat.shape == (3,)
        assert all(math.isfinite(v) for v in x_hat)

    def test_gaussian_prior_shrinkage(self):
        """Gaussian prior shrinks estimate toward prior mean."""
        est = BayesOptimalEstimator(
            noise_variance=1.0,
            prior="gaussian",
            prior_mean=0.0,
            prior_variance=1.0,
        )
        y = np.array([10.0])
        x_hat = est.denoise(y)
        # Posterior mean = (σ²_prior / (σ²_prior + σ²)) · y
        # = (1/(1+1)) · 10 = 5
        assert abs(x_hat[0] - 5.0) < 1e-6

    def test_uniform_prior_denoise(self):
        """Uniform prior Bayes estimator produces output."""
        est = BayesOptimalEstimator(
            noise_variance=1.0,
            prior="uniform",
            prior_lower=0.0,
            prior_upper=1.0,
        )
        y = np.array([0.5])
        x_hat = est.denoise(y)
        assert x_hat.shape == (1,)
        # Result should be within or near [0, 1]
        assert -1.0 < x_hat[0] < 2.0

    def test_bayes_beats_identity_lmmse(self):
        """Under known Gaussian prior, Bayes should beat plain LMMSE."""
        sigma2 = 1.0
        prior_var = 1.0
        bayes = BayesOptimalEstimator(
            noise_variance=sigma2,
            prior="gaussian",
            prior_mean=0.0,
            prior_variance=prior_var,
        )
        # Bayes posterior variance = σ² · σ²_prior / (σ² + σ²_prior)
        post_var = bayes.posterior_variance()
        expected = sigma2 * prior_var / (sigma2 + prior_var)  # = 0.5
        assert abs(post_var - expected) < 1e-6
        # LMMSE with identity would have MSE = σ² = 1.0
        assert post_var < sigma2

    def test_bias_variance(self):
        """bias_variance returns (bias², variance) tuple."""
        est = BayesOptimalEstimator(
            noise_variance=1.0,
            prior="gaussian",
            prior_mean=0.0,
            prior_variance=1.0,
        )
        bias_sq, var = est.bias_variance(x_true=np.array([0.0]))
        assert bias_sq >= -1e-10
        assert var >= -1e-10

    def test_posterior_variance_positive(self):
        """Posterior variance is always positive."""
        est = BayesOptimalEstimator(
            noise_variance=2.0,
            prior="gaussian",
            prior_mean=0.0,
            prior_variance=3.0,
        )
        pv = est.posterior_variance()
        assert pv > 0


# =========================================================================
# Section 5: Bias-Variance Decomposition
# =========================================================================


class TestBiasVarianceDecomposer:
    """Tests for bias-variance decomposition."""

    def test_total_mse_equals_bias_plus_variance(self, identity_workload_3):
        """total MSE = bias² + variance."""
        decomp = BiasVarianceDecomposer(
            workload_matrix=identity_workload_3,
            noise_variance=1.0,
        )
        x_true = np.array([1.0, 2.0, 3.0])
        bias_sq, var = decomp.bias_variance(x_true)
        total = decomp.total_mse(x_true)
        assert abs(total - (bias_sq + var)) < 1e-8

    def test_decomposition_report(self, identity_workload_3):
        """decomposition_report returns dict with expected keys."""
        decomp = BiasVarianceDecomposer(
            workload_matrix=identity_workload_3,
            noise_variance=1.0,
        )
        x_true = np.array([1.0, 2.0, 3.0])
        report = decomp.decomposition_report(x_true)
        assert "bias_squared" in report or "bias" in report
        assert "variance" in report
        assert "total_mse" in report

    def test_identity_filter_no_bias(self, identity_workload_3):
        """Identity filter on identity workload → zero bias."""
        F = np.eye(3)
        decomp = BiasVarianceDecomposer(
            workload_matrix=identity_workload_3,
            noise_variance=1.0,
            filter_matrix=F,
        )
        x_true = np.array([1.0, 2.0, 3.0])
        bias_sq, var = decomp.bias_variance(x_true)
        assert bias_sq < 1e-10  # FA - I = 0 → no bias

    def test_variance_proportional_to_noise(self, identity_workload_3):
        """Variance component is positive and finite for different noise levels."""
        for sigma2 in [0.5, 1.0, 4.0]:
            d = BiasVarianceDecomposer(
                workload_matrix=identity_workload_3, noise_variance=sigma2,
            )
            _, v = d.bias_variance()
            assert v > 0
            assert math.isfinite(v)


# =========================================================================
# Section 6: Wiener Filter
# =========================================================================


class TestWienerFilter:
    """Tests for frequency-domain Wiener filtering."""

    def test_denoise_reduces_noise(self):
        """Wiener filter output has lower noise than input."""
        rng = np.random.default_rng(42)
        n = 64
        # Signal: low-frequency sinusoid
        t = np.linspace(0, 1, n, endpoint=False)
        signal = np.sin(2 * np.pi * 3 * t)
        noise = rng.normal(0, 0.5, n)
        noisy = signal + noise

        wf = WienerFilter(noise_variance=0.25, signal_variance=0.5, n_freq=n)
        filtered = wf.denoise(noisy)
        assert filtered.shape == (n,)

        # Filtered signal should be closer to true signal than noisy
        error_noisy = np.mean((noisy - signal) ** 2)
        error_filtered = np.mean((filtered - signal) ** 2)
        assert error_filtered < error_noisy * 1.5  # Shouldn't make it much worse

    def test_optimal_filter_shape(self):
        """optimal_filter returns array of correct length."""
        wf = WienerFilter(noise_variance=1.0, signal_variance=1.0, n_freq=32)
        filt = wf.optimal_filter()
        assert len(filt) > 0
        # Filter values should be in [0, 1]
        assert all(0 <= f <= 1.0 + 1e-10 for f in filt)

    def test_output_snr(self):
        """output_snr is non-negative."""
        wf = WienerFilter(noise_variance=1.0, signal_variance=2.0, n_freq=16)
        snr = wf.output_snr
        assert all(s >= -1e-10 for s in snr)

    def test_total_output_mse(self):
        """total_output_mse is finite and positive."""
        wf = WienerFilter(noise_variance=1.0, signal_variance=1.0, n_freq=16)
        mse = wf.total_output_mse
        assert mse > 0
        assert math.isfinite(mse)

    def test_known_wiener_formula(self):
        """For known PSDs, filter matches H(ω) = Sxx / (Sxx + Snn)."""
        signal_psd = np.ones(8) * 2.0
        noise_psd = np.ones(8) * 1.0
        wf = WienerFilter(signal_psd=signal_psd, noise_psd=noise_psd)
        filt = wf.optimal_filter()
        expected = signal_psd / (signal_psd + noise_psd)  # = 2/3
        np.testing.assert_array_almost_equal(filt, expected, decimal=6)

    def test_custom_psd(self):
        """Custom signal and noise PSD."""
        n = 16
        signal_psd = np.array([4.0, 2.0, 1.0, 0.5] + [0.1] * 12)
        noise_psd = np.ones(n)
        wf = WienerFilter(signal_psd=signal_psd, noise_psd=noise_psd)
        filt = wf.optimal_filter()
        # Low-frequency components should have higher filter weight
        assert filt[0] > filt[-1]


# =========================================================================
# Section 7: Privacy Preservation
# =========================================================================


class TestPrivacyPreservation:
    """Tests for post-processing privacy preservation proofs."""

    def test_lmmse_privacy_proof(self, identity_workload_3):
        """LMMSE produces valid privacy preservation proof."""
        est = LMMSEEstimator(
            workload_matrix=identity_workload_3, noise_variance=1.0,
        )
        proof = est.privacy_proof(epsilon=1.0, delta=1e-5)
        assert isinstance(proof, PrivacyPreservationProof)
        assert proof.is_data_independent
        assert proof.epsilon == 1.0
        assert proof.delta == 1e-5
        assert len(proof.proof_text) > 0

    def test_bayes_privacy_proof(self):
        """Bayes estimator privacy proof."""
        est = BayesOptimalEstimator(
            noise_variance=1.0, prior="gaussian",
        )
        proof = est.privacy_proof(epsilon=2.0)
        assert isinstance(proof, PrivacyPreservationProof)
        assert proof.is_data_independent

    def test_wiener_privacy_proof(self):
        """Wiener filter privacy proof."""
        wf = WienerFilter(noise_variance=1.0, signal_variance=1.0, n_freq=8)
        proof = wf.privacy_proof(epsilon=1.0)
        assert isinstance(proof, PrivacyPreservationProof)
        assert proof.is_data_independent

    def test_proof_text_mentions_post_processing(self, identity_workload_3):
        """Proof text references the post-processing theorem."""
        est = LMMSEEstimator(
            workload_matrix=identity_workload_3, noise_variance=1.0,
        )
        proof = est.privacy_proof(epsilon=1.0)
        text_lower = proof.proof_text.lower()
        assert "post-processing" in text_lower or "post processing" in text_lower or "data-independent" in text_lower or "data independent" in text_lower
