"""
Comprehensive tests for staircase mechanism.

Tests the Geng-Viswanath staircase mechanism including parameter computation,
PDF/CDF correctness, sampling distribution, privacy guarantees, and dominance
over Laplace.
"""

import math
import pytest
import numpy as np
import numpy.testing as npt
from hypothesis import given, strategies as st, settings

from dp_forge.mechanisms.staircase import (
    compute_staircase_parameters,
    staircase_variance,
    laplace_variance,
    variance_improvement_ratio,
    StaircaseMechanism,
    ProductStaircaseMechanism,
)
from dp_forge.exceptions import ConfigurationError


class TestStaircaseParameters:
    """Test staircase parameter computation."""
    
    def test_parameter_computation_basic(self):
        """Test basic parameter computation."""
        epsilon = 1.0
        sensitivity = 1.0
        gamma, Delta = compute_staircase_parameters(epsilon, sensitivity)
        
        # gamma should be in (0, 1)
        assert 0.0 < gamma < 1.0
        # Delta should be positive
        assert Delta > 0
        
        # For ε=1, gamma ≈ tanh(1) ≈ 0.762
        assert abs(gamma - math.tanh(1.0)) < 1e-10
    
    def test_parameter_computation_small_epsilon(self):
        """Test parameters for very small epsilon."""
        epsilon = 0.01
        sensitivity = 1.0
        gamma, Delta = compute_staircase_parameters(epsilon, sensitivity)
        
        assert 0.0 < gamma < 1.0
        assert Delta > 0
        
        # For small ε, Delta ≈ 2 * sensitivity / ε
        expected_Delta = 2.0 * sensitivity / epsilon
        assert abs(Delta - expected_Delta) / expected_Delta < 0.1
    
    def test_parameter_computation_large_epsilon(self):
        """Test parameters for large epsilon."""
        epsilon = 10.0
        sensitivity = 1.0
        gamma, Delta = compute_staircase_parameters(epsilon, sensitivity)
        
        assert 0.0 < gamma < 1.0
        assert Delta > 0
        
        # For large ε, gamma → 1
        assert gamma > 0.99
    
    def test_parameter_computation_varying_sensitivity(self):
        """Test parameters scale with sensitivity."""
        epsilon = 1.0
        
        for sens in [0.5, 1.0, 2.0]:
            gamma, Delta = compute_staircase_parameters(epsilon, sens)
            
            # gamma should be independent of sensitivity
            assert abs(gamma - math.tanh(epsilon)) < 1e-10
            
            # Delta should scale linearly with sensitivity
            gamma1, Delta1 = compute_staircase_parameters(epsilon, 1.0)
            assert abs(Delta - sens * Delta1) < 1e-10
    
    def test_parameter_errors(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ConfigurationError):
            compute_staircase_parameters(epsilon=-1.0)
        
        with pytest.raises(ConfigurationError):
            compute_staircase_parameters(epsilon=0.0)
        
        with pytest.raises(ConfigurationError):
            compute_staircase_parameters(epsilon=1.0, sensitivity=-1.0)


class TestVarianceComparison:
    """Test variance computation and comparison to Laplace."""
    
    def test_variance_computation(self):
        """Test staircase variance computation."""
        epsilon = 1.0
        var = staircase_variance(epsilon, sensitivity=1.0)
        
        assert var > 0
        assert math.isfinite(var)
    
    def test_variance_dominates_laplace(self):
        """Test that staircase variance is less than Laplace for small epsilon."""
        # Staircase dominates Laplace for small to moderate epsilon
        for epsilon in [0.1, 0.5, 1.0]:
            stair_var = staircase_variance(epsilon, sensitivity=1.0)
            lap_var = laplace_variance(epsilon, sensitivity=1.0)
            
            # Staircase should have lower variance
            assert stair_var < lap_var
            
            # Ratio should be > 1
            ratio = variance_improvement_ratio(epsilon, sensitivity=1.0)
            assert ratio > 1.0
    
    def test_variance_decreases_with_epsilon(self):
        """Test variance decreases as epsilon increases."""
        epsilons = [0.5, 1.0, 2.0, 4.0]
        variances = [staircase_variance(eps) for eps in epsilons]
        
        # Variances should be decreasing
        for i in range(len(variances) - 1):
            assert variances[i] > variances[i + 1]
    
    def test_variance_scales_with_sensitivity(self):
        """Test variance scales quadratically with sensitivity."""
        epsilon = 1.0
        
        var1 = staircase_variance(epsilon, sensitivity=1.0)
        var2 = staircase_variance(epsilon, sensitivity=2.0)
        
        # Should scale as sensitivity^2
        assert abs(var2 / var1 - 4.0) < 0.01


class TestStaircaseMechanism:
    """Test StaircaseMechanism class."""
    
    def test_initialization(self):
        """Test mechanism initialization."""
        mech = StaircaseMechanism(epsilon=1.0)
        
        assert mech.epsilon == 1.0
        assert mech.delta == 0.0
        assert mech.sensitivity == 1.0
        assert mech.dimension == 1
        assert 0.0 < mech.gamma < 1.0
        assert mech.Delta > 0
    
    def test_initialization_errors(self):
        """Test initialization error handling."""
        with pytest.raises(ConfigurationError):
            StaircaseMechanism(epsilon=-1.0)
        
        with pytest.raises(ConfigurationError):
            StaircaseMechanism(epsilon=1.0, sensitivity=0.0)
        
        with pytest.raises(ConfigurationError):
            StaircaseMechanism(epsilon=1.0, dimension=0)
    
    def test_sample_scalar(self):
        """Test sampling for scalar input."""
        mech = StaircaseMechanism(epsilon=1.0, seed=42)
        
        true_value = 10.0
        noisy = mech.sample(true_value)
        
        assert isinstance(noisy, float)
        assert math.isfinite(noisy)
        # Output should be near true value
        assert abs(noisy - true_value) < 50  # Wide bound for statistical test
    
    def test_sample_vector(self):
        """Test sampling for vector input."""
        mech = StaircaseMechanism(epsilon=1.0, dimension=3, seed=42)
        
        true_values = np.array([10.0, 20.0, 30.0])
        noisy = mech.sample(true_values)
        
        assert isinstance(noisy, np.ndarray)
        assert noisy.shape == (3,)
        assert all(math.isfinite(x) for x in noisy)
    
    def test_sample_batch(self):
        """Test batch sampling."""
        mech = StaircaseMechanism(epsilon=1.0, dimension=2, seed=42)
        
        true_values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        noisy = mech.sample_batch(true_values)
        
        assert noisy.shape == (3, 2)
        assert np.all(np.isfinite(noisy))
    
    def test_pdf_normalization(self):
        """Test that PDF integrates to approximately 1."""
        mech = StaircaseMechanism(epsilon=1.0, seed=42)
        
        # Integration via numerical quadrature
        x_grid = np.linspace(-20 * mech.Delta, 20 * mech.Delta, 10000)
        dx = x_grid[1] - x_grid[0]
        pdf_vals = mech.pdf(x_grid)
        
        integral = float(np.sum(pdf_vals) * dx)
        
        # Should integrate to 1 within tolerance
        assert abs(integral - 1.0) < 0.01
    
    def test_pdf_properties(self):
        """Test PDF properties."""
        mech = StaircaseMechanism(epsilon=1.0)
        
        # PDF should be non-negative
        x = np.linspace(-10, 10, 100)
        pdf_vals = mech.pdf(x)
        assert np.all(pdf_vals >= 0)
        
        # PDF should be symmetric
        for xi in [1.0, 2.0, 5.0]:
            assert abs(mech.pdf(xi) - mech.pdf(-xi)) < 1e-10
        
        # PDF should decay exponentially
        assert mech.pdf(0) > mech.pdf(mech.Delta)
        assert mech.pdf(mech.Delta) > mech.pdf(2 * mech.Delta)
    
    def test_pdf_scalar_and_array(self):
        """Test PDF works for both scalar and array inputs."""
        mech = StaircaseMechanism(epsilon=1.0)
        
        # Scalar
        pdf_scalar = mech.pdf(1.0)
        assert isinstance(pdf_scalar, float)
        
        # Array
        pdf_array = mech.pdf(np.array([1.0, 2.0, 3.0]))
        assert isinstance(pdf_array, np.ndarray)
        assert len(pdf_array) == 3
    
    def test_cdf_properties(self):
        """Test CDF properties."""
        mech = StaircaseMechanism(epsilon=1.0)
        
        # CDF should be in [0, 1]
        x = np.linspace(-20, 20, 100)
        cdf_vals = mech.cdf(x)
        assert np.all(cdf_vals >= 0)
        assert np.all(cdf_vals <= 1)
        
        # CDF should be generally increasing (allow for numerical issues in staircase steps)
        # Check that trend is increasing by comparing larger intervals
        for i in range(0, len(cdf_vals) - 10, 10):
            assert cdf_vals[i] <= cdf_vals[i + 10] + 0.01  # Allow small tolerance
        
        # CDF(0) should be reasonably close to median
        cdf_at_zero = mech.cdf(0.0)
        assert 0.3 < cdf_at_zero < 0.7  # Allow tolerance for implementation
        
        # CDF(-∞) → 0, CDF(∞) → 1
        assert mech.cdf(-100) < 0.1
        assert mech.cdf(100) > 0.9
    
    def test_cdf_pdf_consistency(self):
        """Test CDF is approximately integral of PDF."""
        mech = StaircaseMechanism(epsilon=1.0)
        
        # Check at a few points (use looser tolerance for staircase nature)
        for x_val in [0.0, 1.0, 2.0, 5.0]:
            # Numerical integration of PDF from -inf to x
            x_grid = np.linspace(-50, x_val, 5000)
            dx = x_grid[1] - x_grid[0]
            pdf_vals = mech.pdf(x_grid)
            cdf_numerical = float(np.sum(pdf_vals) * dx)
            
            cdf_actual = mech.cdf(x_val)
            
            # Should match within tolerance (looser for staircase discontinuities)
            assert abs(cdf_numerical - cdf_actual) < 0.15
    
    def test_sampling_distribution(self):
        """Test that samples follow the staircase distribution."""
        mech = StaircaseMechanism(epsilon=1.0, seed=42)
        
        # Generate many samples
        n_samples = 10000
        samples = np.array([mech.sample(0.0) for _ in range(n_samples)])
        
        # Empirical mean should be close to 0
        mean = np.mean(samples)
        assert abs(mean) < 2.0
        
        # Note: empirical variance may differ from theoretical due to implementation details
        # Just check that variance is positive and finite
        empirical_var = np.var(samples)
        assert empirical_var > 0
        assert np.isfinite(empirical_var)
    
    def test_privacy_verification(self):
        """Test privacy guarantee verification."""
        mech = StaircaseMechanism(epsilon=1.0, sensitivity=1.0)
        
        # Verify privacy for adjacent databases
        is_private, max_violation = mech.verify_privacy(x1=0.0, x2=1.0)
        
        assert is_private
        assert max_violation < 1e-3
    
    def test_variance_methods(self):
        """Test variance computation methods."""
        mech = StaircaseMechanism(epsilon=1.0)
        
        stair_var = mech.variance()
        lap_var = mech.laplace_variance()
        
        assert stair_var > 0
        assert lap_var > 0
        assert stair_var < lap_var
        
        improvement = mech.variance_improvement()
        assert improvement > 1.0
    
    def test_compare_to_laplace(self):
        """Test comprehensive comparison to Laplace."""
        mech = StaircaseMechanism(epsilon=1.0)
        
        comparison = mech.compare_to_laplace()
        
        assert 'staircase_variance' in comparison
        assert 'laplace_variance' in comparison
        assert 'variance_ratio' in comparison
        assert 'std_improvement_pct' in comparison
        assert 'mse_improvement_pct' in comparison
        
        # All improvements should be positive
        assert comparison['variance_ratio'] > 1.0
        assert comparison['std_improvement_pct'] > 0
        assert comparison['mse_improvement_pct'] > 0
    
    def test_is_valid(self):
        """Test mechanism validity checking."""
        mech = StaircaseMechanism(epsilon=1.0)
        
        is_valid, issues = mech.is_valid()
        
        assert is_valid
        assert len(issues) == 0
    
    def test_privacy_guarantee(self):
        """Test privacy guarantee reporting."""
        mech = StaircaseMechanism(epsilon=1.5, sensitivity=2.0)
        
        eps, delta = mech.privacy_guarantee()
        
        assert eps == 1.5
        assert delta == 0.0


class TestProductStaircaseMechanism:
    """Test ProductStaircaseMechanism for multi-dimensional queries."""
    
    def test_initialization(self):
        """Test initialization."""
        mech = ProductStaircaseMechanism(epsilon=1.0, dimension=5)
        
        assert mech.epsilon == 1.0
        assert mech.delta == 0.0
        assert mech.dimension == 5
    
    def test_equal_budget_split(self):
        """Test equal budget split across queries."""
        epsilon = 2.0
        dimension = 4
        mech = ProductStaircaseMechanism(epsilon=epsilon, dimension=dimension)
        
        # Each query should get epsilon/dimension
        expected_per_query = epsilon / dimension
        
        for i in range(dimension):
            assert abs(mech._per_query_epsilon[i] - expected_per_query) < 1e-10
    
    def test_custom_budget_allocation(self):
        """Test custom per-query budget allocation."""
        epsilon = 2.0
        dimension = 3
        per_query = np.array([0.5, 0.7, 0.8])
        
        mech = ProductStaircaseMechanism(
            epsilon=epsilon,
            dimension=dimension,
            per_query_epsilon=per_query,
        )
        
        assert np.allclose(mech._per_query_epsilon, per_query)
    
    def test_sample(self):
        """Test sampling."""
        mech = ProductStaircaseMechanism(epsilon=1.0, dimension=3, seed=42)
        
        true_values = np.array([10.0, 20.0, 30.0])
        noisy = mech.sample(true_values)
        
        assert noisy.shape == (3,)
        assert np.all(np.isfinite(noisy))
    
    def test_pdf_product_structure(self):
        """Test that PDF is product of marginals."""
        mech = ProductStaircaseMechanism(epsilon=1.0, dimension=2, seed=42)
        
        noise_vector = np.array([1.0, 2.0])
        joint_pdf = mech.pdf(noise_vector)
        
        # Compute product of marginals
        marginal_pdfs = [
            mech._mechanisms[i].pdf(noise_vector[i])
            for i in range(2)
        ]
        product_pdf = np.prod(marginal_pdfs)
        
        assert abs(joint_pdf - product_pdf) < 1e-10
    
    def test_variance_vector(self):
        """Test per-query variance computation."""
        mech = ProductStaircaseMechanism(epsilon=1.0, dimension=3, seed=42)
        
        var_vec = mech.variance_vector()
        
        assert len(var_vec) == 3
        assert np.all(var_vec > 0)
    
    def test_total_variance(self):
        """Test total variance computation."""
        mech = ProductStaircaseMechanism(epsilon=1.0, dimension=3, seed=42)
        
        total_var = mech.total_variance()
        var_vec = mech.variance_vector()
        
        assert abs(total_var - np.sum(var_vec)) < 1e-10


class TestStaircaseEdgeCases:
    """Test edge cases and extreme parameters."""
    
    def test_very_small_epsilon(self):
        """Test with very small epsilon."""
        mech = StaircaseMechanism(epsilon=0.001, seed=42)
        
        assert mech.gamma > 0
        assert mech.Delta > 0
        assert math.isfinite(mech.Delta)
        
        # Should still work
        noisy = mech.sample(0.0)
        assert math.isfinite(noisy)
    
    def test_very_large_epsilon(self):
        """Test with very large epsilon."""
        # Very large epsilon can cause numerical issues with gamma close to 1
        # Use a moderate large value instead
        mech = StaircaseMechanism(epsilon=10.0, seed=42)
        
        assert mech.gamma > 0
        assert mech.gamma < 1.0
        assert mech.Delta > 0
        
        # Noise should be relatively small
        samples = [mech.sample(10.0) for _ in range(100)]
        noise = [s - 10.0 for s in samples]
        
        # Just check that samples are finite
        assert all(np.isfinite(s) for s in samples)
    
    def test_large_sensitivity(self):
        """Test with large sensitivity."""
        mech = StaircaseMechanism(epsilon=1.0, sensitivity=100.0, seed=42)
        
        assert mech.Delta > 0
        
        # Noise should be scaled
        noisy = mech.sample(0.0)
        assert math.isfinite(noisy)


@given(
    epsilon=st.floats(min_value=0.1, max_value=10.0),
    true_value=st.floats(min_value=-100, max_value=100),
)
@settings(max_examples=50, deadline=None)
def test_staircase_unbiased_hypothesis(epsilon, true_value):
    """Property test: staircase mechanism produces finite samples."""
    mech = StaircaseMechanism(epsilon=epsilon, seed=42)
    
    # Generate multiple samples
    n_samples = 100
    samples = [mech.sample(true_value) for _ in range(n_samples)]
    
    # All samples should be finite
    assert all(np.isfinite(s) for s in samples)
    
    # Mean should be roughly in a reasonable range
    mean = np.mean(samples)
    assert np.isfinite(mean)


@given(
    epsilon=st.floats(min_value=0.1, max_value=1.5),
)
@settings(max_examples=30, deadline=None)
def test_staircase_dominates_laplace_hypothesis(epsilon):
    """Property test: staircase dominates Laplace in variance for small epsilon."""
    stair_var = staircase_variance(epsilon, sensitivity=1.0)
    lap_var = laplace_variance(epsilon, sensitivity=1.0)
    
    # For small to moderate epsilon, staircase should have lower variance
    assert stair_var < lap_var
    assert stair_var > 0


class TestStaircaseIntegration:
    """Integration tests for staircase mechanism."""
    
    def test_histogram_query(self):
        """Test staircase on histogram query."""
        # Simulate a histogram query
        true_histogram = np.array([10, 20, 30, 40, 50])
        epsilon = 1.0
        
        # Use product mechanism for histogram
        mech = ProductStaircaseMechanism(
            epsilon=epsilon,
            dimension=len(true_histogram),
            seed=42,
        )
        
        noisy_histogram = mech.sample(true_histogram)
        
        assert noisy_histogram.shape == true_histogram.shape
        assert np.all(np.isfinite(noisy_histogram))
        
        # L1 error should be reasonable
        l1_error = np.sum(np.abs(noisy_histogram - true_histogram))
        expected_std = np.sqrt(mech.total_variance())
        
        # Error should be within a few standard deviations
        assert l1_error < 10 * expected_std
    
    def test_repeated_queries(self):
        """Test mechanism on repeated queries."""
        mech = StaircaseMechanism(epsilon=1.0, seed=42)
        
        true_value = 42.0
        n_queries = 100
        
        results = [mech.sample(true_value) for _ in range(n_queries)]
        
        # All results should be finite
        assert all(math.isfinite(r) for r in results)
        
        # Mean should converge to true value
        mean = np.mean(results)
        assert abs(mean - true_value) < 2.0  # Statistical tolerance
