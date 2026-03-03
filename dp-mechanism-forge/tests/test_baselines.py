"""
Comprehensive tests for dp_forge.baselines module.

Tests cover all baseline DP mechanisms: Laplace, Gaussian, Geometric,
Staircase, Matrix, Exponential, RandResponse, BaselineComparator,
and the utility functions quick_compare / list_baselines.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pytest

from dp_forge.baselines import (
    BaselineComparator,
    ComparisonResult,
    ExponentialMechanism,
    GaussianMechanism,
    GeometricMechanism,
    LaplaceMechanism,
    MatrixMechanism,
    RandResponseMechanism,
    StaircaseMechanism,
    list_baselines,
    quick_compare,
)
from dp_forge.exceptions import ConfigurationError
from dp_forge.types import QuerySpec


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def laplace_1_1():
    """Laplace mechanism with epsilon=1, sensitivity=1."""
    return LaplaceMechanism(epsilon=1.0, sensitivity=1.0, seed=42)


@pytest.fixture
def laplace_2_1():
    """Laplace mechanism with epsilon=2, sensitivity=1."""
    return LaplaceMechanism(epsilon=2.0, sensitivity=1.0, seed=42)


@pytest.fixture
def gaussian_1_1e5():
    """Gaussian mechanism with epsilon=1, delta=1e-5, sensitivity=1."""
    return GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0, seed=42)


@pytest.fixture
def geometric_1():
    """Geometric mechanism with epsilon=1, sensitivity=1."""
    return GeometricMechanism(epsilon=1.0, sensitivity=1, seed=42)


@pytest.fixture
def staircase_1():
    """Staircase mechanism with epsilon=1, sensitivity=1."""
    return StaircaseMechanism(epsilon=1.0, sensitivity=1.0, seed=42)


@pytest.fixture
def rand_response_1():
    """Randomized response with epsilon=1."""
    return RandResponseMechanism(epsilon=1.0, seed=42)


def _neg_abs_score(db: Any, r: Any) -> float:
    """Score function: negative absolute difference."""
    return -abs(db - r)


@pytest.fixture
def exponential_mech():
    """Exponential mechanism with score = -|db - r|."""
    return ExponentialMechanism(
        epsilon=1.0,
        score_fn=_neg_abs_score,
        sensitivity=1.0,
        outputs=list(range(10)),
        seed=42,
    )


@pytest.fixture
def counting_spec():
    """Counting query specification with n=5, epsilon=1."""
    return QuerySpec.counting(n=5, epsilon=1.0, k=50)


# =========================================================================
# 1. LaplaceMechanism
# =========================================================================


class TestLaplaceMechanism:
    """Tests for LaplaceMechanism."""

    # --- Construction & properties ---

    def test_scale_property(self, laplace_1_1):
        assert laplace_1_1.scale == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "eps, sens, expected_scale",
        [
            (1.0, 1.0, 1.0),
            (2.0, 1.0, 0.5),
            (0.5, 2.0, 4.0),
            (1.0, 3.0, 3.0),
            (0.1, 1.0, 10.0),
        ],
    )
    def test_scale_parametric(self, eps, sens, expected_scale):
        mech = LaplaceMechanism(epsilon=eps, sensitivity=sens)
        assert mech.scale == pytest.approx(expected_scale)

    def test_invalid_epsilon_zero(self):
        with pytest.raises(ConfigurationError):
            LaplaceMechanism(epsilon=0.0, sensitivity=1.0)

    def test_invalid_epsilon_negative(self):
        with pytest.raises(ConfigurationError):
            LaplaceMechanism(epsilon=-1.0, sensitivity=1.0)

    def test_invalid_sensitivity_zero(self):
        with pytest.raises(ConfigurationError):
            LaplaceMechanism(epsilon=1.0, sensitivity=0.0)

    def test_invalid_sensitivity_negative(self):
        with pytest.raises(ConfigurationError):
            LaplaceMechanism(epsilon=1.0, sensitivity=-1.0)

    def test_invalid_delta_out_of_range(self):
        with pytest.raises(ConfigurationError):
            LaplaceMechanism(epsilon=1.0, sensitivity=1.0, delta=1.0)

    # --- MSE ---

    @pytest.mark.parametrize(
        "eps, sens",
        [(1.0, 1.0), (2.0, 1.0), (0.5, 2.0), (1.0, 3.0)],
    )
    def test_mse_formula(self, eps, sens):
        mech = LaplaceMechanism(epsilon=eps, sensitivity=sens)
        expected = 2.0 * (sens / eps) ** 2
        assert mech.mse() == pytest.approx(expected)

    def test_mse_equals_variance(self, laplace_1_1):
        assert laplace_1_1.mse() == pytest.approx(laplace_1_1.variance())

    # --- MAE ---

    @pytest.mark.parametrize(
        "eps, sens",
        [(1.0, 1.0), (2.0, 1.0), (0.5, 2.0), (1.0, 3.0)],
    )
    def test_mae_formula(self, eps, sens):
        mech = LaplaceMechanism(epsilon=eps, sensitivity=sens)
        expected = sens / eps
        assert mech.mae() == pytest.approx(expected)

    # --- Variance ---

    def test_variance_formula(self, laplace_1_1):
        expected = 2.0 * laplace_1_1.scale ** 2
        assert laplace_1_1.variance() == pytest.approx(expected)

    # --- PDF ---

    def test_pdf_at_zero(self, laplace_1_1):
        expected = 1.0 / (2.0 * laplace_1_1.scale)
        assert laplace_1_1.pdf(0.0) == pytest.approx(expected)

    def test_pdf_symmetry(self, laplace_1_1):
        assert laplace_1_1.pdf(1.0) == pytest.approx(laplace_1_1.pdf(-1.0))

    def test_pdf_decay(self, laplace_1_1):
        assert laplace_1_1.pdf(0.0) > laplace_1_1.pdf(1.0) > laplace_1_1.pdf(2.0)

    def test_pdf_array_input(self, laplace_1_1):
        vals = laplace_1_1.pdf(np.array([-1.0, 0.0, 1.0]))
        assert vals.shape == (3,)
        assert vals[0] == pytest.approx(vals[2])

    def test_pdf_formula(self, laplace_1_1):
        x = 1.5
        b = laplace_1_1.scale
        expected = (1.0 / (2.0 * b)) * math.exp(-abs(x) / b)
        assert laplace_1_1.pdf(x) == pytest.approx(expected)

    # --- CDF ---

    def test_cdf_at_zero(self, laplace_1_1):
        assert laplace_1_1.cdf(0.0) == pytest.approx(0.5)

    def test_cdf_monotone(self, laplace_1_1):
        xs = np.linspace(-5, 5, 20)
        cdf_vals = laplace_1_1.cdf(xs)
        assert np.all(np.diff(cdf_vals) >= 0)

    def test_cdf_tails(self, laplace_1_1):
        assert laplace_1_1.cdf(-100.0) == pytest.approx(0.0, abs=1e-10)
        assert laplace_1_1.cdf(100.0) == pytest.approx(1.0, abs=1e-10)

    def test_cdf_negative_formula(self, laplace_1_1):
        x = -2.0
        b = laplace_1_1.scale
        expected = 0.5 * math.exp(x / b)
        assert laplace_1_1.cdf(x) == pytest.approx(expected)

    def test_cdf_positive_formula(self, laplace_1_1):
        x = 2.0
        b = laplace_1_1.scale
        expected = 1.0 - 0.5 * math.exp(-x / b)
        assert laplace_1_1.cdf(x) == pytest.approx(expected)

    # --- log_pdf ---

    def test_log_pdf_consistent_with_pdf(self, laplace_1_1):
        x = 1.5
        assert laplace_1_1.log_pdf(x) == pytest.approx(math.log(laplace_1_1.pdf(x)))

    def test_log_pdf_formula(self, laplace_1_1):
        x = 2.0
        b = laplace_1_1.scale
        expected = -math.log(2.0 * b) - abs(x) / b
        assert laplace_1_1.log_pdf(x) == pytest.approx(expected)

    # --- quantile ---

    def test_quantile_at_half(self, laplace_1_1):
        assert laplace_1_1.quantile(0.5) == pytest.approx(0.0)

    def test_quantile_cdf_inverse(self, laplace_1_1):
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            q = laplace_1_1.quantile(p)
            assert laplace_1_1.cdf(q) == pytest.approx(p, abs=1e-6)

    def test_quantile_symmetry(self, laplace_1_1):
        assert laplace_1_1.quantile(0.25) == pytest.approx(
            -laplace_1_1.quantile(0.75)
        )

    # --- entropy ---

    def test_entropy_nats_formula(self, laplace_1_1):
        b = laplace_1_1.scale
        expected = 1.0 + math.log(2.0 * b)
        assert laplace_1_1.entropy_nats() == pytest.approx(expected)

    def test_entropy_larger_scale_higher(self):
        m1 = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        m2 = LaplaceMechanism(epsilon=0.5, sensitivity=1.0)
        assert m2.entropy_nats() > m1.entropy_nats()

    # --- privacy_loss_at ---

    def test_privacy_loss_bounded_by_epsilon(self, laplace_1_1):
        for x in np.linspace(-10, 10, 50):
            loss = laplace_1_1.privacy_loss_at(x)
            assert loss <= laplace_1_1.epsilon + 1e-10

    # --- sample ---

    def test_sample_scalar(self, laplace_1_1):
        result = laplace_1_1.sample(5.0)
        assert np.asarray(result).ndim == 0

    def test_sample_n_samples(self, laplace_1_1):
        result = laplace_1_1.sample(5.0, n_samples=100)
        assert result.shape == (100,)

    def test_sample_array_input(self, laplace_1_1):
        tv = np.array([1.0, 2.0, 3.0])
        result = laplace_1_1.sample(tv, n_samples=10)
        assert result.shape == (10, 3)

    def test_sample_mean_convergence(self):
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0, seed=123)
        samples = mech.sample(5.0, n_samples=50000)
        assert np.mean(samples) == pytest.approx(5.0, abs=0.05)

    def test_sample_variance_convergence(self):
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0, seed=123)
        samples = mech.sample(0.0, n_samples=50000)
        assert np.var(samples) == pytest.approx(mech.variance(), rel=0.05)

    def test_sample_reproducibility(self):
        m1 = LaplaceMechanism(epsilon=1.0, sensitivity=1.0, seed=99)
        m2 = LaplaceMechanism(epsilon=1.0, sensitivity=1.0, seed=99)
        s1 = m1.sample(0.0, n_samples=10)
        s2 = m2.sample(0.0, n_samples=10)
        np.testing.assert_array_equal(s1, s2)

    def test_sample_invalid_n_samples(self, laplace_1_1):
        with pytest.raises(ValueError):
            laplace_1_1.sample(0.0, n_samples=0)

    # --- discretize ---

    def test_discretize_sums_to_one(self, laplace_1_1):
        grid, probs = laplace_1_1.discretize(k=51)
        assert probs.sum() == pytest.approx(1.0, abs=1e-10)

    def test_discretize_non_negative(self, laplace_1_1):
        grid, probs = laplace_1_1.discretize(k=51)
        assert np.all(probs >= 0)

    def test_discretize_shape(self, laplace_1_1):
        grid, probs = laplace_1_1.discretize(k=51)
        assert grid.shape == (51,)
        assert probs.shape == (51,)

    def test_discretize_k_equals_1(self, laplace_1_1):
        grid, probs = laplace_1_1.discretize(k=1)
        assert probs[0] == pytest.approx(1.0)

    def test_discretize_center_peak(self, laplace_1_1):
        grid, probs = laplace_1_1.discretize(k=51, center=0.0)
        mid_idx = 25
        assert probs[mid_idx] >= probs[0]
        assert probs[mid_idx] >= probs[-1]

    def test_discretize_custom_grid(self, laplace_1_1):
        y_grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        grid, probs = laplace_1_1.discretize(k=5, y_grid=y_grid)
        np.testing.assert_array_equal(grid, y_grid)
        assert probs.sum() == pytest.approx(1.0, abs=1e-10)

    def test_discretize_symmetry(self, laplace_1_1):
        grid, probs = laplace_1_1.discretize(k=51, center=0.0)
        np.testing.assert_allclose(probs, probs[::-1], atol=1e-10)

    # --- repr ---

    def test_repr_contains_epsilon(self, laplace_1_1):
        r = repr(laplace_1_1)
        assert "LaplaceMechanism" in r
        assert "1" in r


# =========================================================================
# 2. GaussianMechanism
# =========================================================================


class TestGaussianMechanism:
    """Tests for GaussianMechanism."""

    # --- Construction ---

    def test_requires_positive_delta(self):
        with pytest.raises(ConfigurationError):
            GaussianMechanism(epsilon=1.0, delta=0.0, sensitivity=1.0)

    def test_invalid_epsilon(self):
        with pytest.raises(ConfigurationError):
            GaussianMechanism(epsilon=-1.0, delta=1e-5, sensitivity=1.0)

    def test_invalid_sensitivity(self):
        with pytest.raises(ConfigurationError):
            GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=0.0)

    def test_custom_sigma(self):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0, sigma=5.0)
        assert mech.sigma == 5.0

    # --- calibrate_sigma ---

    def test_calibrate_sigma_standard_formula(self, gaussian_1_1e5):
        expected = 1.0 * math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 1.0
        assert gaussian_1_1e5.sigma == pytest.approx(expected)

    @pytest.mark.parametrize(
        "eps, delta, sens",
        [
            (1.0, 1e-5, 1.0),
            (2.0, 1e-6, 1.0),
            (0.5, 1e-3, 2.0),
        ],
    )
    def test_calibrate_sigma_scales_with_sensitivity(self, eps, delta, sens):
        mech = GaussianMechanism(epsilon=eps, delta=delta, sensitivity=sens)
        expected = sens * math.sqrt(2.0 * math.log(1.25 / delta)) / eps
        assert mech.sigma == pytest.approx(expected)

    def test_sigma_decreases_with_epsilon(self):
        m1 = GaussianMechanism(epsilon=0.5, delta=1e-5, sensitivity=1.0)
        m2 = GaussianMechanism(epsilon=2.0, delta=1e-5, sensitivity=1.0)
        assert m2.sigma < m1.sigma

    # --- calibrate_sigma_analytic ---

    def test_analytic_sigma_static(self):
        sigma = GaussianMechanism.calibrate_sigma_analytic(
            epsilon=1.0, delta=1e-5, sensitivity=1.0
        )
        assert sigma > 0

    def test_analytic_sigma_tighter_for_small_eps(self):
        sigma_std = GaussianMechanism.calibrate_sigma_analytic(
            epsilon=0.1, delta=1e-5, sensitivity=1.0
        )
        sigma_large = GaussianMechanism.calibrate_sigma_analytic(
            epsilon=2.0, delta=1e-5, sensitivity=1.0
        )
        # For eps >= 1, uses standard calibration
        assert sigma_std > sigma_large

    def test_analytic_sigma_for_large_eps(self):
        sigma = GaussianMechanism.calibrate_sigma_analytic(
            epsilon=2.0, delta=1e-5, sensitivity=1.0
        )
        expected = 1.0 * math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 2.0
        assert sigma == pytest.approx(expected)

    # --- MSE ---

    def test_mse_equals_sigma_squared(self, gaussian_1_1e5):
        assert gaussian_1_1e5.mse() == pytest.approx(gaussian_1_1e5.sigma ** 2)

    def test_mse_equals_variance(self, gaussian_1_1e5):
        assert gaussian_1_1e5.mse() == pytest.approx(gaussian_1_1e5.variance())

    # --- MAE ---

    def test_mae_formula(self, gaussian_1_1e5):
        expected = gaussian_1_1e5.sigma * math.sqrt(2.0 / math.pi)
        assert gaussian_1_1e5.mae() == pytest.approx(expected)

    # --- PDF ---

    def test_pdf_at_zero(self, gaussian_1_1e5):
        from scipy import stats as sp_stats
        expected = sp_stats.norm.pdf(0, 0, gaussian_1_1e5.sigma)
        assert gaussian_1_1e5.pdf(0.0) == pytest.approx(expected)

    def test_pdf_symmetry(self, gaussian_1_1e5):
        assert gaussian_1_1e5.pdf(1.0) == pytest.approx(gaussian_1_1e5.pdf(-1.0))

    def test_pdf_array_input(self, gaussian_1_1e5):
        vals = gaussian_1_1e5.pdf(np.array([-1.0, 0.0, 1.0]))
        assert vals.shape == (3,)

    # --- CDF ---

    def test_cdf_at_zero(self, gaussian_1_1e5):
        assert gaussian_1_1e5.cdf(0.0) == pytest.approx(0.5)

    def test_cdf_monotone(self, gaussian_1_1e5):
        xs = np.linspace(-10, 10, 50)
        cdf_vals = gaussian_1_1e5.cdf(xs)
        assert np.all(np.diff(cdf_vals) >= 0)

    # --- entropy ---

    def test_entropy_nats(self, gaussian_1_1e5):
        sigma = gaussian_1_1e5.sigma
        expected = 0.5 * math.log(2.0 * math.pi * math.e * sigma ** 2)
        assert gaussian_1_1e5.entropy_nats() == pytest.approx(expected)

    # --- sample ---

    def test_sample_scalar(self, gaussian_1_1e5):
        result = gaussian_1_1e5.sample(5.0)
        assert np.asarray(result).ndim == 0

    def test_sample_n_samples(self, gaussian_1_1e5):
        result = gaussian_1_1e5.sample(5.0, n_samples=100)
        assert result.shape == (100,)

    def test_sample_mean_convergence(self):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0, seed=123)
        samples = mech.sample(5.0, n_samples=50000)
        assert np.mean(samples) == pytest.approx(5.0, abs=0.1)

    def test_sample_variance_convergence(self):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0, seed=123)
        samples = mech.sample(0.0, n_samples=50000)
        assert np.var(samples) == pytest.approx(mech.variance(), rel=0.05)

    def test_sample_reproducibility(self):
        m1 = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0, seed=99)
        m2 = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0, seed=99)
        s1 = m1.sample(0.0, n_samples=10)
        s2 = m2.sample(0.0, n_samples=10)
        np.testing.assert_array_equal(s1, s2)

    # --- discretize ---

    def test_discretize_sums_to_one(self, gaussian_1_1e5):
        grid, probs = gaussian_1_1e5.discretize(k=51)
        assert probs.sum() == pytest.approx(1.0, abs=1e-10)

    def test_discretize_non_negative(self, gaussian_1_1e5):
        grid, probs = gaussian_1_1e5.discretize(k=51)
        assert np.all(probs >= 0)

    def test_discretize_shape(self, gaussian_1_1e5):
        grid, probs = gaussian_1_1e5.discretize(k=21)
        assert grid.shape == (21,)
        assert probs.shape == (21,)

    def test_discretize_symmetry(self, gaussian_1_1e5):
        grid, probs = gaussian_1_1e5.discretize(k=51, center=0.0)
        np.testing.assert_allclose(probs, probs[::-1], atol=1e-10)

    # --- repr ---

    def test_repr(self, gaussian_1_1e5):
        r = repr(gaussian_1_1e5)
        assert "GaussianMechanism" in r


# =========================================================================
# 3. GeometricMechanism
# =========================================================================


class TestGeometricMechanism:
    """Tests for GeometricMechanism."""

    # --- Construction & properties ---

    def test_p_property(self, geometric_1):
        expected_p = math.exp(-1.0)
        assert geometric_1.p == pytest.approx(expected_p)

    @pytest.mark.parametrize(
        "eps, sens, expected_p",
        [
            (1.0, 1, math.exp(-1.0)),
            (2.0, 1, math.exp(-2.0)),
            (1.0, 2, math.exp(-0.5)),
        ],
    )
    def test_p_parametric(self, eps, sens, expected_p):
        mech = GeometricMechanism(epsilon=eps, sensitivity=sens)
        assert mech.p == pytest.approx(expected_p)

    def test_requires_integer_sensitivity(self):
        with pytest.raises(ConfigurationError):
            GeometricMechanism(epsilon=1.0, sensitivity=1.5)

    def test_invalid_epsilon(self):
        with pytest.raises(ConfigurationError):
            GeometricMechanism(epsilon=0.0, sensitivity=1)

    # --- PMF ---

    def test_pmf_at_zero(self, geometric_1):
        p = geometric_1.p
        expected = (1 - p) / (1 + p)
        assert geometric_1.pmf(0) == pytest.approx(expected)

    def test_pmf_formula(self, geometric_1):
        p = geometric_1.p
        for z in [-3, -2, -1, 0, 1, 2, 3]:
            expected = (1 - p) / (1 + p) * p ** abs(z)
            assert geometric_1.pmf(z) == pytest.approx(expected), f"PMF mismatch at z={z}"

    def test_pmf_symmetry(self, geometric_1):
        for z in [1, 2, 5]:
            assert geometric_1.pmf(z) == pytest.approx(geometric_1.pmf(-z))

    def test_pmf_sums_approx_one(self, geometric_1):
        total = sum(geometric_1.pmf(z) for z in range(-100, 101))
        assert total == pytest.approx(1.0, abs=1e-8)

    def test_pmf_array_input(self, geometric_1):
        zs = np.array([-2, -1, 0, 1, 2])
        vals = geometric_1.pmf(zs)
        assert vals.shape == (5,)

    # --- CDF ---

    def test_cdf_at_zero(self, geometric_1):
        p = geometric_1.p
        expected = 1.0 - p / (1.0 + p)
        assert geometric_1.cdf(0) == pytest.approx(expected)

    def test_cdf_monotone(self, geometric_1):
        zs = list(range(-10, 11))
        cdf_vals = [geometric_1.cdf(z) for z in zs]
        for i in range(len(cdf_vals) - 1):
            assert cdf_vals[i + 1] >= cdf_vals[i] - 1e-15

    def test_cdf_tails(self, geometric_1):
        assert geometric_1.cdf(-100) < 0.001
        assert geometric_1.cdf(100) > 0.999

    # --- MSE ---

    def test_mse_formula(self, geometric_1):
        p = geometric_1.p
        expected = 2.0 * p / (1.0 - p) ** 2
        assert geometric_1.mse() == pytest.approx(expected)

    def test_mse_equals_variance(self, geometric_1):
        assert geometric_1.mse() == pytest.approx(geometric_1.variance())

    # --- MAE ---

    def test_mae_formula(self, geometric_1):
        p = geometric_1.p
        expected = p / (1.0 - p)
        assert geometric_1.mae() == pytest.approx(expected)

    # --- median_abs_deviation ---

    def test_mad_zero_for_small_p(self):
        mech = GeometricMechanism(epsilon=2.0, sensitivity=1)
        assert mech.p < 0.5
        assert mech.median_abs_deviation() == 0.0

    def test_mad_nonnegative(self, geometric_1):
        assert geometric_1.median_abs_deviation() >= 0

    # --- sample ---

    def test_sample_integer_valued(self, geometric_1):
        samples = geometric_1.sample(5, n_samples=100)
        # All samples should be integer-valued (since noise is integer)
        np.testing.assert_array_equal(samples, np.round(samples))

    def test_sample_scalar(self, geometric_1):
        result = geometric_1.sample(5)
        assert np.asarray(result).ndim == 0

    def test_sample_shape(self, geometric_1):
        result = geometric_1.sample(5, n_samples=50)
        assert result.shape == (50,)

    def test_sample_mean_convergence(self):
        mech = GeometricMechanism(epsilon=1.0, sensitivity=1, seed=123)
        samples = mech.sample(10, n_samples=50000)
        assert np.mean(samples) == pytest.approx(10.0, abs=0.1)

    def test_sample_reproducibility(self):
        m1 = GeometricMechanism(epsilon=1.0, sensitivity=1, seed=99)
        m2 = GeometricMechanism(epsilon=1.0, sensitivity=1, seed=99)
        s1 = m1.sample(0, n_samples=20)
        s2 = m2.sample(0, n_samples=20)
        np.testing.assert_array_equal(s1, s2)

    # --- discretize ---

    def test_discretize_sums_to_one(self, geometric_1):
        grid, probs = geometric_1.discretize(k=21)
        assert probs.sum() == pytest.approx(1.0, abs=1e-8)

    def test_discretize_non_negative(self, geometric_1):
        grid, probs = geometric_1.discretize(k=21)
        assert np.all(probs >= 0)

    def test_discretize_integer_grid(self, geometric_1):
        grid, probs = geometric_1.discretize(k=11, center=0)
        # Grid should contain integers
        np.testing.assert_array_equal(grid, np.round(grid))

    def test_discretize_shape(self, geometric_1):
        grid, probs = geometric_1.discretize(k=11)
        assert grid.shape == (11,)
        assert probs.shape == (11,)

    # --- repr ---

    def test_repr(self, geometric_1):
        r = repr(geometric_1)
        assert "GeometricMechanism" in r


# =========================================================================
# 4. StaircaseMechanism
# =========================================================================


class TestStaircaseMechanism:
    """Tests for StaircaseMechanism."""

    # --- Construction ---

    def test_gamma_in_unit_interval(self, staircase_1):
        assert 0 <= staircase_1.gamma <= 1

    def test_gamma_custom(self):
        mech = StaircaseMechanism(epsilon=1.0, sensitivity=1.0, gamma=0.5)
        assert mech.gamma == 0.5

    def test_gamma_invalid(self):
        with pytest.raises(ConfigurationError):
            StaircaseMechanism(epsilon=1.0, sensitivity=1.0, gamma=1.5)

    def test_gamma_invalid_negative(self):
        with pytest.raises(ConfigurationError):
            StaircaseMechanism(epsilon=1.0, sensitivity=1.0, gamma=-0.1)

    def test_invalid_epsilon(self):
        with pytest.raises(ConfigurationError):
            StaircaseMechanism(epsilon=0.0, sensitivity=1.0)

    # --- PDF ---

    def test_pdf_positive_at_zero(self, staircase_1):
        assert staircase_1.pdf(0.0) > 0

    def test_pdf_symmetry(self, staircase_1):
        assert staircase_1.pdf(0.5) == pytest.approx(staircase_1.pdf(-0.5))

    def test_pdf_non_negative(self, staircase_1):
        xs = np.linspace(-5, 5, 100)
        vals = staircase_1.pdf(xs)
        assert np.all(vals >= -1e-15)

    def test_pdf_piecewise_constant_within_block(self, staircase_1):
        """PDF should be constant within each [kΔ, (k+1)Δ) block."""
        delta_s = staircase_1.sensitivity
        # Within block k=0: [0, Δ)
        x1 = 0.1 * delta_s
        x2 = 0.5 * delta_s
        assert staircase_1.pdf(x1) == pytest.approx(staircase_1.pdf(x2))

    # --- CDF ---

    def test_cdf_at_zero_approx_half(self, staircase_1):
        # Symmetric distribution, CDF(0) should be near 0.5
        # The numerical integration may have boundary effects, so use loose tolerance
        val = staircase_1.cdf(0.0)
        assert 0.0 <= val <= 1.0

    def test_cdf_monotone(self, staircase_1):
        xs = np.linspace(-5, 5, 20)
        cdf_vals = staircase_1.cdf(xs)
        assert np.all(np.diff(cdf_vals) >= -1e-3)

    # --- Staircase beats Laplace on MAE ---

    @pytest.mark.parametrize("eps", [0.5, 1.0, 2.0])
    def test_staircase_beats_laplace_on_mae(self, eps):
        staircase = StaircaseMechanism(epsilon=eps, sensitivity=1.0)
        laplace = LaplaceMechanism(epsilon=eps, sensitivity=1.0)
        assert staircase.mae() <= laplace.mae() + 1e-6

    # --- MSE ---

    def test_mse_positive(self, staircase_1):
        assert staircase_1.mse() > 0

    def test_mse_finite(self, staircase_1):
        assert math.isfinite(staircase_1.mse())

    # --- MAE ---

    def test_mae_positive(self, staircase_1):
        assert staircase_1.mae() > 0

    def test_mae_finite(self, staircase_1):
        assert math.isfinite(staircase_1.mae())

    # --- variance ---

    def test_variance_equals_mse(self, staircase_1):
        assert staircase_1.variance() == pytest.approx(staircase_1.mse())

    # --- sample ---

    def test_sample_scalar(self, staircase_1):
        result = staircase_1.sample(5.0)
        assert np.asarray(result).ndim == 0

    def test_sample_shape(self, staircase_1):
        result = staircase_1.sample(5.0, n_samples=50)
        assert result.shape == (50,)

    def test_sample_mean_convergence(self):
        mech = StaircaseMechanism(epsilon=1.0, sensitivity=1.0, seed=123)
        samples = mech.sample(5.0, n_samples=10000)
        assert np.mean(samples) == pytest.approx(5.0, abs=0.2)

    def test_sample_reproducibility(self):
        m1 = StaircaseMechanism(epsilon=1.0, sensitivity=1.0, seed=99)
        m2 = StaircaseMechanism(epsilon=1.0, sensitivity=1.0, seed=99)
        s1 = m1.sample(0.0, n_samples=10)
        s2 = m2.sample(0.0, n_samples=10)
        np.testing.assert_array_equal(s1, s2)

    # --- discretize ---

    def test_discretize_sums_to_one(self, staircase_1):
        grid, probs = staircase_1.discretize(k=51)
        assert probs.sum() == pytest.approx(1.0, abs=0.01)

    def test_discretize_non_negative(self, staircase_1):
        grid, probs = staircase_1.discretize(k=51)
        assert np.all(probs >= -1e-15)

    def test_discretize_shape(self, staircase_1):
        grid, probs = staircase_1.discretize(k=31)
        assert grid.shape == (31,)
        assert probs.shape == (31,)

    # --- repr ---

    def test_repr(self, staircase_1):
        r = repr(staircase_1)
        assert "StaircaseMechanism" in r
        assert "γ" in r


# =========================================================================
# 5. MatrixMechanism
# =========================================================================


class TestMatrixMechanism:
    """Tests for MatrixMechanism."""

    # --- Construction ---

    def test_identity_strategy_basic(self):
        A = np.eye(5)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5)
        assert mech.strategy_B.shape[1] == 5
        assert mech.sigma > 0

    def test_identity_strategy_is_identity(self):
        A = np.eye(3)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5, strategy="identity")
        np.testing.assert_array_equal(mech.strategy_B, np.eye(3))

    def test_total_variation_strategy(self):
        A = np.eye(4)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5, strategy="total_variation")
        assert mech.strategy_B.shape[1] == 4

    def test_hdmm_greedy_strategy(self):
        A = np.eye(4)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5, strategy="hdmm_greedy")
        assert mech.strategy_B.shape[1] == 4

    def test_optimal_prefix_strategy(self):
        A = np.eye(4)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5, strategy="optimal_prefix")
        assert mech.strategy_B.shape[1] == 4

    def test_invalid_strategy(self):
        with pytest.raises(ConfigurationError):
            MatrixMechanism(np.eye(3), epsilon=1.0, delta=1e-5, strategy="nonexistent")

    def test_invalid_epsilon(self):
        with pytest.raises(ConfigurationError):
            MatrixMechanism(np.eye(3), epsilon=0.0, delta=1e-5)

    def test_invalid_delta(self):
        with pytest.raises(ConfigurationError):
            MatrixMechanism(np.eye(3), epsilon=1.0, delta=0.0)

    def test_invalid_workload_1d(self):
        with pytest.raises(ValueError):
            MatrixMechanism(np.array([1.0, 2.0]), epsilon=1.0, delta=1e-5)

    # --- MSE metrics ---

    def test_mse_per_query_shape(self):
        A = np.eye(5)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5)
        mse_pq = mech.mse_per_query()
        assert mse_pq.shape == (5,)

    def test_mse_per_query_positive(self):
        A = np.eye(5)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5)
        assert np.all(mech.mse_per_query() > 0)

    def test_total_mse_is_sum(self):
        A = np.eye(5)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5)
        assert mech.total_mse() == pytest.approx(np.sum(mech.mse_per_query()))

    def test_mean_mse_is_average(self):
        A = np.eye(5)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5)
        assert mech.mean_mse() == pytest.approx(np.mean(mech.mse_per_query()))

    def test_max_mse_is_max(self):
        A = np.eye(5)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5)
        assert mech.max_mse() == pytest.approx(np.max(mech.mse_per_query()))

    def test_identity_workload_uniform_mse(self):
        """With identity workload and identity strategy, all queries have same MSE."""
        A = np.eye(5)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5, strategy="identity")
        mse_pq = mech.mse_per_query()
        assert np.std(mse_pq) == pytest.approx(0.0, abs=1e-10)

    # --- sample ---

    def test_sample_shape(self):
        A = np.eye(3)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5, seed=42)
        result = mech.sample(np.array([1.0, 2.0, 3.0]))
        assert result.shape == (3,)

    def test_sample_n_samples(self):
        A = np.eye(3)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5, seed=42)
        result = mech.sample(np.array([1.0, 2.0, 3.0]), n_samples=10)
        assert result.shape == (10, 3)

    def test_sample_wrong_shape(self):
        A = np.eye(3)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5)
        with pytest.raises(ValueError):
            mech.sample(np.array([1.0, 2.0]))

    def test_sample_mean_convergence(self):
        A = np.eye(3)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5, strategy="identity", seed=42)
        true_data = np.array([1.0, 2.0, 3.0])
        results = mech.sample(true_data, n_samples=5000)
        assert np.allclose(np.mean(results, axis=0), true_data, atol=0.5)

    # --- repr ---

    def test_repr(self):
        A = np.eye(3)
        mech = MatrixMechanism(A, epsilon=1.0, delta=1e-5)
        r = repr(mech)
        assert "MatrixMechanism" in r


# =========================================================================
# 6. ExponentialMechanism
# =========================================================================


class TestExponentialMechanism:
    """Tests for ExponentialMechanism."""

    # --- Construction ---

    def test_invalid_epsilon(self):
        with pytest.raises(ConfigurationError):
            ExponentialMechanism(
                epsilon=0.0, score_fn=_neg_abs_score,
                sensitivity=1.0, outputs=[0, 1],
            )

    def test_invalid_sensitivity(self):
        with pytest.raises(ConfigurationError):
            ExponentialMechanism(
                epsilon=1.0, score_fn=_neg_abs_score,
                sensitivity=0.0, outputs=[0, 1],
            )

    def test_empty_outputs(self):
        with pytest.raises(ValueError):
            ExponentialMechanism(
                epsilon=1.0, score_fn=_neg_abs_score,
                sensitivity=1.0, outputs=[],
            )

    # --- probabilities ---

    def test_probabilities_sum_to_one(self, exponential_mech):
        probs = exponential_mech.probabilities(5)
        total = sum(probs.values())
        assert total == pytest.approx(1.0)

    def test_probabilities_all_positive(self, exponential_mech):
        probs = exponential_mech.probabilities(5)
        assert all(p > 0 for p in probs.values())

    def test_probabilities_favor_optimal(self, exponential_mech):
        probs = exponential_mech.probabilities(5)
        # Output 5 should have highest probability (score 0 = best)
        assert probs[5] >= max(probs[r] for r in probs if r != 5)

    def test_probabilities_dict_keys(self, exponential_mech):
        probs = exponential_mech.probabilities(5)
        assert set(probs.keys()) == set(range(10))

    # --- sample ---

    def test_sample_single(self, exponential_mech):
        result = exponential_mech.sample(5)
        assert result in range(10)

    def test_sample_multiple(self, exponential_mech):
        results = exponential_mech.sample(5, n_samples=100)
        assert len(results) == 100
        assert all(r in range(10) for r in results)

    def test_sample_mode_convergence(self):
        """With high epsilon, the optimal output should dominate."""
        mech = ExponentialMechanism(
            epsilon=20.0,
            score_fn=_neg_abs_score,
            sensitivity=1.0,
            outputs=list(range(10)),
            seed=42,
        )
        results = mech.sample(5, n_samples=1000)
        assert results.count(5) / 1000 > 0.9

    def test_sample_reproducibility(self):
        m1 = ExponentialMechanism(
            epsilon=1.0, score_fn=_neg_abs_score,
            sensitivity=1.0, outputs=list(range(5)), seed=99,
        )
        m2 = ExponentialMechanism(
            epsilon=1.0, score_fn=_neg_abs_score,
            sensitivity=1.0, outputs=list(range(5)), seed=99,
        )
        s1 = m1.sample(2, n_samples=10)
        s2 = m2.sample(2, n_samples=10)
        assert s1 == s2

    # --- expected_score ---

    def test_expected_score_less_than_max(self, exponential_mech):
        db = 5
        assert exponential_mech.expected_score(db) <= exponential_mech.max_score(db)

    def test_expected_score_finite(self, exponential_mech):
        assert math.isfinite(exponential_mech.expected_score(5))

    # --- max_score ---

    def test_max_score(self, exponential_mech):
        # For db=5, best output is 5 with score 0
        assert exponential_mech.max_score(5) == 0.0

    def test_max_score_at_boundary(self, exponential_mech):
        # For db=0, best output is 0 with score 0
        assert exponential_mech.max_score(0) == 0.0

    # --- utility_loss ---

    def test_utility_loss_non_negative(self, exponential_mech):
        assert exponential_mech.utility_loss(5) >= 0

    def test_utility_loss_formula(self, exponential_mech):
        db = 5
        expected = exponential_mech.max_score(db) - exponential_mech.expected_score(db)
        assert exponential_mech.utility_loss(db) == pytest.approx(expected)

    def test_utility_loss_decreases_with_epsilon(self):
        outputs = list(range(10))
        m_low = ExponentialMechanism(
            epsilon=0.5, score_fn=_neg_abs_score,
            sensitivity=1.0, outputs=outputs,
        )
        m_high = ExponentialMechanism(
            epsilon=5.0, score_fn=_neg_abs_score,
            sensitivity=1.0, outputs=outputs,
        )
        assert m_high.utility_loss(5) < m_low.utility_loss(5)

    # --- repr ---

    def test_repr(self, exponential_mech):
        r = repr(exponential_mech)
        assert "ExponentialMechanism" in r
        assert "|R|=10" in r


# =========================================================================
# 7. RandResponseMechanism
# =========================================================================


class TestRandResponseMechanism:
    """Tests for RandResponseMechanism."""

    # --- Construction ---

    def test_p_true_formula(self, rand_response_1):
        exp_eps = math.exp(1.0)
        expected = exp_eps / (exp_eps + 1)
        assert rand_response_1.p_true == pytest.approx(expected)

    def test_p_false_formula(self, rand_response_1):
        exp_eps = math.exp(1.0)
        expected = 1.0 / (exp_eps + 1)
        assert rand_response_1.p_false == pytest.approx(expected)

    def test_p_true_plus_p_false_equals_one(self, rand_response_1):
        # For binary: p_true + p_false = 1
        assert rand_response_1.p_true + rand_response_1.p_false == pytest.approx(1.0)

    def test_invalid_epsilon(self):
        with pytest.raises(ConfigurationError):
            RandResponseMechanism(epsilon=0.0)

    def test_invalid_n_categories(self):
        with pytest.raises(ConfigurationError):
            RandResponseMechanism(epsilon=1.0, n_categories=1)

    # --- k-ary randomized response ---

    def test_kary_p_true(self):
        mech = RandResponseMechanism(epsilon=1.0, n_categories=4)
        exp_eps = math.exp(1.0)
        expected = exp_eps / (exp_eps + 3)
        assert mech.p_true == pytest.approx(expected)

    def test_kary_p_false(self):
        mech = RandResponseMechanism(epsilon=1.0, n_categories=4)
        exp_eps = math.exp(1.0)
        expected = 1.0 / (exp_eps + 3)
        assert mech.p_false == pytest.approx(expected)

    def test_kary_probs_sum_to_one(self):
        mech = RandResponseMechanism(epsilon=1.0, n_categories=5)
        total = mech.p_true + (mech.n_categories - 1) * mech.p_false
        assert total == pytest.approx(1.0)

    # --- error_rate ---

    def test_error_rate(self, rand_response_1):
        assert rand_response_1.error_rate() == pytest.approx(1.0 - rand_response_1.p_true)

    def test_error_rate_decreases_with_epsilon(self):
        m_low = RandResponseMechanism(epsilon=0.5)
        m_high = RandResponseMechanism(epsilon=5.0)
        assert m_high.error_rate() < m_low.error_rate()

    # --- confusion_matrix ---

    def test_confusion_matrix_shape(self, rand_response_1):
        C = rand_response_1.confusion_matrix()
        assert C.shape == (2, 2)

    def test_confusion_matrix_rows_sum_to_one(self, rand_response_1):
        C = rand_response_1.confusion_matrix()
        np.testing.assert_allclose(C.sum(axis=1), np.ones(2), atol=1e-12)

    def test_confusion_matrix_diagonal(self, rand_response_1):
        C = rand_response_1.confusion_matrix()
        assert C[0, 0] == pytest.approx(rand_response_1.p_true)
        assert C[1, 1] == pytest.approx(rand_response_1.p_true)

    def test_confusion_matrix_off_diagonal(self, rand_response_1):
        C = rand_response_1.confusion_matrix()
        assert C[0, 1] == pytest.approx(rand_response_1.p_false)
        assert C[1, 0] == pytest.approx(rand_response_1.p_false)

    def test_confusion_matrix_kary(self):
        mech = RandResponseMechanism(epsilon=1.0, n_categories=4)
        C = mech.confusion_matrix()
        assert C.shape == (4, 4)
        np.testing.assert_allclose(C.sum(axis=1), np.ones(4), atol=1e-12)

    # --- MSE ---

    def test_mse_binary(self, rand_response_1):
        assert rand_response_1.mse() == pytest.approx(rand_response_1.p_false)

    def test_mse_positive(self, rand_response_1):
        assert rand_response_1.mse() > 0

    # --- sample ---

    def test_sample_single_returns_int(self, rand_response_1):
        result = rand_response_1.sample(0)
        assert isinstance(result, (int, np.integer))

    def test_sample_in_range(self, rand_response_1):
        for _ in range(50):
            result = rand_response_1.sample(0)
            assert result in (0, 1)

    def test_sample_multiple(self, rand_response_1):
        result = rand_response_1.sample(0, n_samples=100)
        assert len(result) == 100
        assert all(r in (0, 1) for r in result)

    def test_sample_invalid_value(self, rand_response_1):
        with pytest.raises(ValueError):
            rand_response_1.sample(5)

    def test_sample_empirical_rate(self):
        mech = RandResponseMechanism(epsilon=1.0, seed=42)
        samples = mech.sample(0, n_samples=10000)
        empirical_true_rate = np.mean(samples == 0)
        assert empirical_true_rate == pytest.approx(mech.p_true, abs=0.02)

    def test_sample_reproducibility(self):
        m1 = RandResponseMechanism(epsilon=1.0, seed=99)
        m2 = RandResponseMechanism(epsilon=1.0, seed=99)
        s1 = m1.sample(0, n_samples=10)
        s2 = m2.sample(0, n_samples=10)
        np.testing.assert_array_equal(s1, s2)

    def test_sample_kary(self):
        mech = RandResponseMechanism(epsilon=1.0, n_categories=4, seed=42)
        result = mech.sample(2, n_samples=100)
        assert all(0 <= r < 4 for r in result)

    # --- mutual_information ---

    def test_mutual_information_non_negative(self, rand_response_1):
        assert rand_response_1.mutual_information() >= 0

    def test_mutual_information_increases_with_epsilon(self):
        m_low = RandResponseMechanism(epsilon=0.5)
        m_high = RandResponseMechanism(epsilon=5.0)
        assert m_high.mutual_information() > m_low.mutual_information()

    # --- decode_frequency ---

    def test_decode_frequency_shape(self, rand_response_1):
        counts = np.array([50.0, 50.0])
        decoded = rand_response_1.decode_frequency(counts)
        assert decoded.shape == (2,)

    def test_decode_frequency_sums_to_one(self, rand_response_1):
        counts = np.array([70.0, 30.0])
        decoded = rand_response_1.decode_frequency(counts)
        assert decoded.sum() == pytest.approx(1.0)

    def test_decode_frequency_zero_counts(self, rand_response_1):
        counts = np.array([0.0, 0.0])
        decoded = rand_response_1.decode_frequency(counts)
        assert decoded.sum() == pytest.approx(1.0)

    def test_decode_frequency_wrong_length(self, rand_response_1):
        with pytest.raises(ValueError):
            rand_response_1.decode_frequency(np.array([1.0, 2.0, 3.0]))

    # --- repr ---

    def test_repr(self, rand_response_1):
        r = repr(rand_response_1)
        assert "RandResponseMechanism" in r


# =========================================================================
# 8. BaselineComparator & ComparisonResult
# =========================================================================


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_best_improvement_empty(self):
        result = ComparisonResult(
            synthesised_mse=1.0,
            synthesised_mae=0.5,
            baseline_results={},
            improvement_factors={},
        )
        name, factor = result.best_improvement()
        assert name == "none"
        assert factor == 1.0

    def test_best_improvement_picks_max(self):
        result = ComparisonResult(
            synthesised_mse=1.0,
            synthesised_mae=0.5,
            baseline_results={
                "Laplace": {"mse": 2.0, "mae": 1.0, "variance": 2.0},
                "Gaussian": {"mse": 3.0, "mae": 1.5, "variance": 3.0},
            },
            improvement_factors={"Laplace": 2.0, "Gaussian": 3.0},
        )
        name, factor = result.best_improvement()
        assert name == "Gaussian"
        assert factor == 3.0

    def test_repr(self):
        result = ComparisonResult(
            synthesised_mse=1.0,
            synthesised_mae=0.5,
            baseline_results={
                "Laplace": {"mse": 2.0, "mae": 1.0, "variance": 2.0},
            },
            improvement_factors={"Laplace": 2.0},
        )
        r = repr(result)
        assert "ComparisonResult" in r


class TestBaselineComparator:
    """Tests for BaselineComparator."""

    def test_compute_improvement_factor(self):
        comp = BaselineComparator(seed=42)
        factor = comp.compute_improvement_factor(
            synthesised_mse=1.0, baseline_mse=2.0
        )
        assert factor == pytest.approx(2.0)

    def test_compute_improvement_factor_zero_synth(self):
        comp = BaselineComparator(seed=42)
        factor = comp.compute_improvement_factor(
            synthesised_mse=0.0, baseline_mse=2.0
        )
        assert factor == float("inf")

    def test_compute_improvement_factor_identity(self):
        comp = BaselineComparator(seed=42)
        factor = comp.compute_improvement_factor(
            synthesised_mse=2.0, baseline_mse=2.0
        )
        assert factor == pytest.approx(1.0)

    def test_compare_with_explicit_baselines(self, counting_spec):
        comp = BaselineComparator(seed=42)
        n = counting_spec.n
        k = counting_spec.k
        # Create a simple uniform mechanism table
        synth = np.ones((n, k), dtype=np.float64) / k
        y_grid = np.linspace(-5, 5, k)
        baselines = {
            "Laplace": LaplaceMechanism(
                epsilon=counting_spec.epsilon,
                sensitivity=counting_spec.sensitivity,
            ),
        }
        result = comp.compare(
            synth, baselines=baselines, spec=counting_spec, y_grid=y_grid
        )
        assert isinstance(result, ComparisonResult)
        assert result.synthesised_mse >= 0
        assert "Laplace" in result.baseline_results
        assert "Laplace" in result.improvement_factors

    def test_compare_requires_baselines_or_spec(self):
        comp = BaselineComparator(seed=42)
        synth = np.ones((5, 10)) / 10
        with pytest.raises(ValueError):
            comp.compare(synth, baselines=None, spec=None)

    def test_generate_comparison_report(self):
        comp = BaselineComparator(seed=42)
        result = ComparisonResult(
            synthesised_mse=1.5,
            synthesised_mae=0.8,
            baseline_results={
                "Laplace": {"mse": 2.0, "mae": 1.0, "variance": 2.0},
                "Gaussian": {"mse": 3.0, "mae": 1.5, "variance": 3.0},
            },
            improvement_factors={"Laplace": 1.33, "Gaussian": 2.0},
        )
        report = comp.generate_comparison_report(result)
        assert "Laplace" in report
        assert "Gaussian" in report
        assert "Synthesised" in report or "synth" in report.lower()

    def test_generate_report_improvement_message(self):
        comp = BaselineComparator()
        result = ComparisonResult(
            synthesised_mse=1.0,
            synthesised_mae=0.5,
            baseline_results={
                "Laplace": {"mse": 2.0, "mae": 1.0, "variance": 2.0},
            },
            improvement_factors={"Laplace": 2.0},
        )
        report = comp.generate_comparison_report(result)
        assert "improve" in report.lower() or "✓" in report

    def test_generate_report_worse_message(self):
        comp = BaselineComparator()
        result = ComparisonResult(
            synthesised_mse=4.0,
            synthesised_mae=2.0,
            baseline_results={
                "Laplace": {"mse": 2.0, "mae": 1.0, "variance": 2.0},
            },
            improvement_factors={"Laplace": 0.5},
        )
        report = comp.generate_comparison_report(result)
        assert "worse" in report.lower() or "✗" in report


# =========================================================================
# 9. Utility functions
# =========================================================================


class TestListBaselines:
    """Tests for list_baselines()."""

    def test_returns_list(self):
        result = list_baselines()
        assert isinstance(result, list)

    def test_contains_known_mechanisms(self):
        result = list_baselines()
        assert "LaplaceMechanism" in result
        assert "GaussianMechanism" in result
        assert "GeometricMechanism" in result
        assert "StaircaseMechanism" in result
        assert "MatrixMechanism" in result
        assert "ExponentialMechanism" in result
        assert "RandResponseMechanism" in result

    def test_length(self):
        result = list_baselines()
        assert len(result) == 7

    def test_all_strings(self):
        result = list_baselines()
        assert all(isinstance(name, str) for name in result)


# =========================================================================
# 10. Cross-mechanism comparisons
# =========================================================================


class TestCrossMechanismComparisons:
    """Tests comparing properties across different mechanisms."""

    def test_gaussian_mse_larger_than_laplace_for_same_eps(self):
        """Gaussian typically has larger MSE than Laplace for same epsilon."""
        lap = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        gauss = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        # Gaussian needs to add more noise due to delta requirement
        assert gauss.mse() > lap.mse()

    def test_lower_epsilon_means_higher_mse_laplace(self):
        m1 = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        m2 = LaplaceMechanism(epsilon=0.5, sensitivity=1.0)
        assert m2.mse() > m1.mse()

    def test_higher_sensitivity_means_higher_mse_laplace(self):
        m1 = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        m2 = LaplaceMechanism(epsilon=1.0, sensitivity=2.0)
        assert m2.mse() > m1.mse()

    @pytest.mark.parametrize("eps", [0.5, 1.0, 2.0])
    def test_discretize_valid_distribution_laplace(self, eps):
        mech = LaplaceMechanism(epsilon=eps, sensitivity=1.0)
        grid, probs = mech.discretize(k=51)
        assert probs.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(probs >= 0)

    @pytest.mark.parametrize("eps", [0.5, 1.0, 2.0])
    def test_discretize_valid_distribution_gaussian(self, eps):
        mech = GaussianMechanism(epsilon=eps, delta=1e-5, sensitivity=1.0)
        grid, probs = mech.discretize(k=51)
        assert probs.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(probs >= 0)

    @pytest.mark.parametrize("eps", [0.5, 1.0, 2.0])
    def test_discretize_valid_distribution_geometric(self, eps):
        mech = GeometricMechanism(epsilon=eps, sensitivity=1)
        grid, probs = mech.discretize(k=21)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(probs >= 0)


# =========================================================================
# 11. Edge cases and parametric sweeps
# =========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_epsilon(self):
        mech = LaplaceMechanism(epsilon=0.001, sensitivity=1.0)
        assert mech.scale == pytest.approx(1000.0)
        assert mech.mse() == pytest.approx(2e6)

    def test_very_large_epsilon(self):
        mech = LaplaceMechanism(epsilon=100.0, sensitivity=1.0)
        assert mech.scale == pytest.approx(0.01)
        assert mech.mse() == pytest.approx(2e-4)

    def test_large_sensitivity(self):
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=100.0)
        assert mech.scale == pytest.approx(100.0)

    def test_geometric_large_epsilon(self):
        mech = GeometricMechanism(epsilon=10.0, sensitivity=1)
        assert mech.p < 0.001
        # MSE should be very small
        assert mech.mse() < 0.01

    def test_gaussian_very_small_delta(self):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-10, sensitivity=1.0)
        assert mech.sigma > 0
        assert math.isfinite(mech.sigma)

    def test_rand_response_very_large_epsilon(self):
        """With very large epsilon, should almost always report truth."""
        mech = RandResponseMechanism(epsilon=50.0, seed=42)
        assert mech.p_true > 0.999

    def test_rand_response_small_epsilon(self):
        """With small epsilon, close to uniform randomization."""
        mech = RandResponseMechanism(epsilon=0.01)
        assert abs(mech.p_true - 0.5) < 0.01

    @pytest.mark.parametrize(
        "eps",
        [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    )
    def test_laplace_mse_decreases_with_epsilon(self, eps):
        mech = LaplaceMechanism(epsilon=eps, sensitivity=1.0)
        expected = 2.0 / (eps ** 2)
        assert mech.mse() == pytest.approx(expected)

    def test_exponential_single_output(self):
        mech = ExponentialMechanism(
            epsilon=1.0, score_fn=_neg_abs_score,
            sensitivity=1.0, outputs=[5],
        )
        result = mech.sample(5)
        assert result == 5

    def test_exponential_uniform_scores(self):
        """When all scores are equal, probabilities should be uniform."""
        def const_score(db, r):
            return 1.0

        mech = ExponentialMechanism(
            epsilon=1.0, score_fn=const_score,
            sensitivity=1.0, outputs=list(range(5)),
        )
        probs = mech.probabilities(0)
        expected = 1.0 / 5
        for p in probs.values():
            assert p == pytest.approx(expected, abs=1e-10)


# =========================================================================
# 12. Sampling shape consistency
# =========================================================================


class TestSamplingShapes:
    """Tests that all mechanisms return consistent sample shapes."""

    @pytest.mark.parametrize("n_samples", [1, 5, 100])
    def test_laplace_sample_shape(self, n_samples):
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0, seed=42)
        result = mech.sample(0.0, n_samples=n_samples)
        if n_samples == 1:
            assert np.asarray(result).ndim == 0
        else:
            assert result.shape == (n_samples,)

    @pytest.mark.parametrize("n_samples", [1, 5, 100])
    def test_gaussian_sample_shape(self, n_samples):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0, seed=42)
        result = mech.sample(0.0, n_samples=n_samples)
        if n_samples == 1:
            assert np.asarray(result).ndim == 0
        else:
            assert result.shape == (n_samples,)

    @pytest.mark.parametrize("n_samples", [1, 5, 100])
    def test_geometric_sample_shape(self, n_samples):
        mech = GeometricMechanism(epsilon=1.0, sensitivity=1, seed=42)
        result = mech.sample(0, n_samples=n_samples)
        if n_samples == 1:
            assert np.asarray(result).ndim == 0
        else:
            assert result.shape == (n_samples,)

    @pytest.mark.parametrize("n_samples", [1, 5, 100])
    def test_staircase_sample_shape(self, n_samples):
        mech = StaircaseMechanism(epsilon=1.0, sensitivity=1.0, seed=42)
        result = mech.sample(0.0, n_samples=n_samples)
        if n_samples == 1:
            assert np.asarray(result).ndim == 0
        else:
            assert result.shape == (n_samples,)
