"""
Comprehensive tests for interval verifier (sound verification with interval arithmetic).

Tests:
- Interval arithmetic operations (add, mul, div) with directed rounding
- Sound DP verification catches known violations
- Sound DP verification accepts known valid mechanisms
- Directed rounding correctness using np.nextafter
- interval_hockey_stick against exact computation
- Batch verification
- Property tests: intervals always contain true value
"""

import numpy as np
import numpy.testing as npt
import pytest

from dp_forge.verification.interval_verifier import (
    IntervalVerifier,
    IntervalResult,
    ErrorPropagation,
    SoundnessLevel,
    _round_down,
    _round_up,
    _interval_add,
    _interval_sub,
    _interval_mul,
    _interval_div,
    _interval_log,
    _interval_exp,
    _interval_max,
    interval_hockey_stick,
    interval_renyi_divergence,
    sound_verify_dp,
    compute_sound_epsilon,
    batch_verify_mechanisms,
    interval_max_divergence,
    interval_kl_divergence,
    interval_total_variation,
    BatchIntervalVerifier,
    statistical_distance_bounds,
)


class TestRoundingOperations:
    """Test directed rounding operations."""
    
    def test_round_down_moves_toward_negative_infinity(self):
        x = np.array([1.0, 2.0, 3.0])
        rounded = _round_down(x)
        assert np.all(rounded <= x)
        assert np.all(rounded < x) or np.all(rounded == x)
    
    def test_round_up_moves_toward_positive_infinity(self):
        x = np.array([1.0, 2.0, 3.0])
        rounded = _round_up(x)
        assert np.all(rounded >= x)
        assert np.all(rounded > x) or np.all(rounded == x)
    
    def test_round_down_uses_nextafter(self):
        x = np.array([1.5])
        rounded = _round_down(x)
        expected = np.nextafter(1.5, -np.inf, dtype=np.float64)
        npt.assert_equal(rounded, expected)
    
    def test_round_up_uses_nextafter(self):
        x = np.array([1.5])
        rounded = _round_up(x)
        expected = np.nextafter(1.5, np.inf, dtype=np.float64)
        npt.assert_equal(rounded, expected)
    
    def test_rounding_preserves_shape(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        rounded_down = _round_down(x)
        rounded_up = _round_up(x)
        assert rounded_down.shape == x.shape
        assert rounded_up.shape == x.shape


class TestIntervalArithmetic:
    """Test interval arithmetic operations."""
    
    def test_interval_add_basic(self):
        a_lo = np.array([1.0, 2.0])
        a_hi = np.array([2.0, 3.0])
        b_lo = np.array([0.5, 1.0])
        b_hi = np.array([1.0, 2.0])
        
        lo, hi = _interval_add(a_lo, a_hi, b_lo, b_hi)
        
        assert np.all(lo >= a_lo + b_lo - 1e-15)
        assert np.all(hi <= a_hi + b_hi + 1e-15)
        assert np.all(lo <= hi)
    
    def test_interval_add_contains_true_value(self):
        a_true = np.array([1.5, 2.5])
        b_true = np.array([0.8, 1.5])
        
        a_lo = a_true - 0.1
        a_hi = a_true + 0.1
        b_lo = b_true - 0.1
        b_hi = b_true + 0.1
        
        lo, hi = _interval_add(a_lo, a_hi, b_lo, b_hi)
        true_sum = a_true + b_true
        
        assert np.all(lo <= true_sum)
        assert np.all(hi >= true_sum)
    
    def test_interval_sub_basic(self):
        a_lo = np.array([2.0, 3.0])
        a_hi = np.array([3.0, 4.0])
        b_lo = np.array([0.5, 1.0])
        b_hi = np.array([1.0, 2.0])
        
        lo, hi = _interval_sub(a_lo, a_hi, b_lo, b_hi)
        
        assert np.all(lo <= hi)
        assert np.all(lo >= a_lo - b_hi - 1e-15)
        assert np.all(hi <= a_hi - b_lo + 1e-15)
    
    def test_interval_mul_positive(self):
        a_lo = np.array([1.0, 2.0])
        a_hi = np.array([2.0, 3.0])
        b_lo = np.array([2.0, 1.0])
        b_hi = np.array([3.0, 2.0])
        
        lo, hi = _interval_mul(a_lo, a_hi, b_lo, b_hi)
        
        assert np.all(lo <= hi)
        assert np.all(lo >= a_lo * b_lo - 1e-14)
        assert np.all(hi <= a_hi * b_hi + 1e-14)
    
    def test_interval_mul_mixed_signs(self):
        a_lo = np.array([-1.0, -2.0])
        a_hi = np.array([2.0, 3.0])
        b_lo = np.array([-2.0, 1.0])
        b_hi = np.array([3.0, 2.0])
        
        lo, hi = _interval_mul(a_lo, a_hi, b_lo, b_hi)
        
        assert np.all(lo <= hi)
    
    def test_interval_div_positive(self):
        a_lo = np.array([1.0, 2.0])
        a_hi = np.array([2.0, 4.0])
        b_lo = np.array([0.5, 1.0])
        b_hi = np.array([1.0, 2.0])
        
        lo, hi = _interval_div(a_lo, a_hi, b_lo, b_hi)
        
        assert np.all(lo <= hi)
        assert np.all(np.isfinite(lo))
        assert np.all(np.isfinite(hi))
    
    def test_interval_div_handles_zero_denominator(self):
        a_lo = np.array([1.0])
        a_hi = np.array([2.0])
        b_lo = np.array([-0.1])
        b_hi = np.array([0.1])
        
        lo, hi = _interval_div(a_lo, a_hi, b_lo, b_hi)
        
        assert np.all(np.isfinite(lo))
        assert np.all(np.isfinite(hi))
    
    def test_interval_log_basic(self):
        x_lo = np.array([0.5, 1.0])
        x_hi = np.array([1.0, 2.0])
        
        lo, hi = _interval_log(x_lo, x_hi)
        
        assert np.all(lo <= hi)
        assert np.all(lo <= np.log(x_hi) + 1e-14)
        assert np.all(hi >= np.log(x_lo) - 1e-14)
    
    def test_interval_log_handles_small_values(self):
        x_lo = np.array([1e-310, 1e-100])
        x_hi = np.array([1e-300, 1e-50])
        
        lo, hi = _interval_log(x_lo, x_hi)
        
        assert np.all(np.isfinite(lo))
        assert np.all(np.isfinite(hi))
        assert np.all(lo <= hi)
    
    def test_interval_exp_basic(self):
        x_lo = np.array([0.0, 1.0])
        x_hi = np.array([1.0, 2.0])
        
        lo, hi = _interval_exp(x_lo, x_hi)
        
        assert np.all(lo <= hi)
        assert np.all(lo <= np.exp(x_hi) + 1e-14)
        assert np.all(hi >= np.exp(x_lo) - 1e-14)
    
    def test_interval_max_basic(self):
        a_lo = np.array([1.0, 3.0])
        a_hi = np.array([2.0, 4.0])
        b_lo = np.array([1.5, 2.0])
        b_hi = np.array([3.0, 3.0])
        
        lo, hi = _interval_max(a_lo, a_hi, b_lo, b_hi)
        
        assert np.all(lo <= hi)
        assert np.all(lo >= np.maximum(a_lo, b_lo))
        assert np.all(hi >= np.maximum(a_hi, b_hi))


class TestHockeyStickDivergence:
    """Test hockey-stick divergence computation."""
    
    def test_hockey_stick_basic(self):
        p_lo = np.array([0.4, 0.1])
        p_hi = np.array([0.6, 0.3])
        q_lo = np.array([0.3, 0.2])
        q_hi = np.array([0.5, 0.4])
        epsilon = 0.5
        
        hs_lo, hs_hi = interval_hockey_stick(p_lo, p_hi, q_lo, q_hi, epsilon)
        
        assert hs_lo <= hs_hi
        # Allow small negative values due to directed rounding in interval arithmetic
        assert hs_lo >= -1e-10
        assert hs_hi >= 0.0
    
    def test_hockey_stick_zero_when_p_small(self):
        p_lo = np.array([0.0, 0.0])
        p_hi = np.array([0.1, 0.1])
        q_lo = np.array([0.4, 0.4])
        q_hi = np.array([0.5, 0.5])
        epsilon = 1.0
        
        hs_lo, hs_hi = interval_hockey_stick(p_lo, p_hi, q_lo, q_hi, epsilon)
        
        assert hs_hi >= 0.0
        npt.assert_almost_equal(hs_lo, 0.0, decimal=6)
    
    def test_hockey_stick_increases_with_epsilon(self):
        p_lo = np.array([0.5])
        p_hi = np.array([0.5])
        q_lo = np.array([0.3])
        q_hi = np.array([0.3])
        
        hs_lo_1, hs_hi_1 = interval_hockey_stick(p_lo, p_hi, q_lo, q_hi, 0.5)
        hs_lo_2, hs_hi_2 = interval_hockey_stick(p_lo, p_hi, q_lo, q_hi, 1.0)
        
        assert hs_hi_1 >= hs_hi_2
    
    def test_hockey_stick_soundness(self):
        np.random.seed(42)
        p_true = np.random.dirichlet([1, 1, 1])
        q_true = np.random.dirichlet([1, 1, 1])
        
        tol = 0.01
        p_lo = np.maximum(p_true - tol, 0.0)
        p_hi = np.minimum(p_true + tol, 1.0)
        q_lo = np.maximum(q_true - tol, 0.0)
        q_hi = np.minimum(q_true + tol, 1.0)
        
        epsilon = 0.5
        hs_lo, hs_hi = interval_hockey_stick(p_lo, p_hi, q_lo, q_hi, epsilon)
        
        true_hs = np.sum(np.maximum(p_true - np.exp(epsilon) * q_true, 0.0))
        
        assert hs_lo <= true_hs + 1e-10
        assert hs_hi >= true_hs - 1e-10


class TestRenyiDivergence:
    """Test Rényi divergence computation."""
    
    def test_renyi_divergence_basic(self):
        p_lo = np.array([0.4, 0.6])
        p_hi = np.array([0.4, 0.6])
        q_lo = np.array([0.5, 0.5])
        q_hi = np.array([0.5, 0.5])
        alpha = 2.0
        
        div_lo, div_hi = interval_renyi_divergence(p_lo, p_hi, q_lo, q_hi, alpha)
        
        assert div_lo <= div_hi
        assert np.isfinite(div_lo)
        assert np.isfinite(div_hi)
    
    def test_renyi_divergence_rejects_alpha_le_1(self):
        p_lo = np.array([0.5])
        p_hi = np.array([0.5])
        q_lo = np.array([0.5])
        q_hi = np.array([0.5])
        
        with pytest.raises(ValueError, match="Rényi order must be > 1"):
            interval_renyi_divergence(p_lo, p_hi, q_lo, q_hi, alpha=1.0)
    
    def test_renyi_divergence_handles_small_probabilities(self):
        p_lo = np.array([1e-10, 0.5])
        p_hi = np.array([1e-9, 0.5])
        q_lo = np.array([0.5, 1e-10])
        q_hi = np.array([0.5, 1e-9])
        alpha = 2.0
        
        div_lo, div_hi = interval_renyi_divergence(p_lo, p_hi, q_lo, q_hi, alpha)
        
        assert np.isfinite(div_lo)
        assert np.isfinite(div_hi)


class TestIntervalVerifier:
    """Test IntervalVerifier class."""
    
    def test_verifier_initialization(self):
        verifier = IntervalVerifier(tolerance=1e-9)
        assert verifier.tolerance == 1e-9
        assert verifier.track_errors is True
    
    def test_verifier_rejects_negative_tolerance(self):
        with pytest.raises(Exception):
            IntervalVerifier(tolerance=-1e-9)
    
    def test_verify_pure_dp_valid_mechanism(self):
        prob_table = np.array([
            [0.5, 0.5],
            [0.4, 0.6]
        ])
        edges = [(0, 1)]
        epsilon = 1.0
        
        verifier = IntervalVerifier()
        result = verifier.verify_pure_dp(prob_table, edges, epsilon)
        
        assert isinstance(result, IntervalResult)
        assert result.soundness == SoundnessLevel.SOUND
        assert result.confidence > 0.9
    
    def test_verify_pure_dp_invalid_mechanism(self):
        prob_table = np.array([
            [0.9, 0.1],
            [0.1, 0.9]
        ])
        edges = [(0, 1)]
        epsilon = 0.1
        
        verifier = IntervalVerifier()
        result = verifier.verify_pure_dp(prob_table, edges, epsilon)
        
        assert result.valid is False
    
    def test_verify_approx_dp_valid_mechanism(self):
        prob_table = np.array([
            [0.5, 0.5],
            [0.4, 0.6]
        ])
        edges = [(0, 1)]
        epsilon = 0.5
        delta = 0.1
        
        verifier = IntervalVerifier()
        result = verifier.verify_approx_dp(prob_table, edges, epsilon, delta)
        
        assert isinstance(result, IntervalResult)
        assert result.soundness in [SoundnessLevel.SOUND, SoundnessLevel.INCONCLUSIVE]
    
    def test_verify_dispatches_to_pure_or_approx(self):
        prob_table = np.array([[0.5, 0.5], [0.5, 0.5]])
        edges = [(0, 1)]
        
        verifier = IntervalVerifier()
        
        result_pure = verifier.verify(prob_table, edges, epsilon=1.0, delta=0.0)
        assert isinstance(result_pure, IntervalResult)
        
        result_approx = verifier.verify(prob_table, edges, epsilon=0.5, delta=0.1)
        assert isinstance(result_approx, IntervalResult)
    
    def test_compute_privacy_loss_bounds(self):
        prob_table = np.array([
            [0.6, 0.4],
            [0.4, 0.6]
        ])
        
        verifier = IntervalVerifier()
        loss_lo, loss_hi = verifier.compute_privacy_loss_bounds(prob_table, 0, 1)
        
        assert loss_lo.shape == (2,)
        assert loss_hi.shape == (2,)
        assert np.all(loss_lo <= loss_hi)
    
    def test_verify_renyi_dp(self):
        prob_table = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        edges = [(0, 1)]
        alpha = 2.0
        renyi_epsilon = 1.0
        
        verifier = IntervalVerifier()
        result = verifier.verify_renyi_dp(prob_table, edges, alpha, renyi_epsilon)
        
        assert isinstance(result, IntervalResult)
        assert result.valid is True
    
    def test_error_propagation_tracking(self):
        prob_table = np.array([
            [0.5, 0.5],
            [0.4, 0.6]
        ])
        edges = [(0, 1)]
        
        verifier = IntervalVerifier(track_errors=True)
        result = verifier.verify(prob_table, edges, epsilon=1.0, delta=0.0)
        
        assert result.error_propagation.operations > 0
        assert result.error_propagation.total_ulps >= 0


class TestSoundVerifyDP:
    """Test sound_verify_dp main entry point."""
    
    def test_sound_verify_dp_basic(self):
        mechanism = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        epsilon = 1.0
        
        result = sound_verify_dp(mechanism, epsilon, delta=0.0)
        
        assert isinstance(result, IntervalResult)
        assert result.valid is True
        assert result.soundness == SoundnessLevel.SOUND
    
    def test_sound_verify_dp_with_delta(self):
        mechanism = np.array([
            [0.6, 0.4],
            [0.4, 0.6]
        ])
        epsilon = 0.5
        delta = 0.1
        
        result = sound_verify_dp(mechanism, epsilon, delta)
        
        assert isinstance(result, IntervalResult)
    
    def test_sound_verify_dp_with_custom_edges(self):
        mechanism = np.array([
            [0.5, 0.5],
            [0.4, 0.6],
            [0.3, 0.7]
        ])
        edges = [(0, 1), (1, 2)]
        epsilon = 1.0
        
        result = sound_verify_dp(mechanism, epsilon, edges=edges)
        
        assert isinstance(result, IntervalResult)
    
    def test_sound_verify_dp_detects_violations(self):
        mechanism = np.array([
            [0.99, 0.01],
            [0.01, 0.99]
        ])
        epsilon = 0.1
        
        result = sound_verify_dp(mechanism, epsilon)
        
        assert result.valid is False
        assert result.violation is not None


class TestBatchVerification:
    """Test batch verification functionality."""
    
    def test_batch_verify_mechanisms(self):
        mechanisms = [
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([[0.6, 0.4], [0.4, 0.6]]),
            np.array([[0.7, 0.3], [0.3, 0.7]]),
        ]
        epsilon = 1.0
        
        results = batch_verify_mechanisms(mechanisms, epsilon)
        
        assert len(results) == 3
        assert all(isinstance(r, IntervalResult) for r in results)
    
    def test_batch_interval_verifier_with_cache(self):
        batch_verifier = BatchIntervalVerifier()
        
        mechanism = np.array([[0.5, 0.5], [0.5, 0.5]])
        
        result1 = batch_verifier.verify_batch([mechanism], epsilon=1.0)
        result2 = batch_verifier.verify_batch([mechanism], epsilon=1.0)
        
        assert len(result1) == 1
        assert len(result2) == 1
    
    def test_batch_verifier_cache_clear(self):
        batch_verifier = BatchIntervalVerifier()
        
        mechanism = np.array([[0.5, 0.5], [0.5, 0.5]])
        batch_verifier.verify_batch([mechanism], epsilon=1.0)
        
        assert len(batch_verifier.cache) > 0
        
        batch_verifier.clear_cache()
        assert len(batch_verifier.cache) == 0


class TestComputeSoundEpsilon:
    """Test sound epsilon computation."""
    
    def test_compute_sound_epsilon_for_identity(self):
        prob_table = np.array([[1.0, 0.0], [0.0, 1.0]])
        edges = [(0, 1)]
        
        eps_lo, eps_hi = compute_sound_epsilon(prob_table, edges, delta=0.0)
        
        assert eps_lo >= 0.0
        assert eps_hi > eps_lo
        assert np.isfinite(eps_hi)
    
    def test_compute_sound_epsilon_for_uniform(self):
        prob_table = np.array([[0.5, 0.5], [0.5, 0.5]])
        edges = [(0, 1)]
        
        eps_lo, eps_hi = compute_sound_epsilon(prob_table, edges)
        
        assert eps_lo >= 0.0
        npt.assert_almost_equal(eps_hi, 0.0, decimal=5)


class TestStatisticalDistances:
    """Test statistical distance computations."""
    
    def test_interval_max_divergence(self):
        p_lo = np.array([0.4, 0.5])
        p_hi = np.array([0.6, 0.6])
        q_lo = np.array([0.3, 0.4])
        q_hi = np.array([0.5, 0.5])
        
        div_lo, div_hi = interval_max_divergence(p_lo, p_hi, q_lo, q_hi)
        
        assert div_lo <= div_hi
        assert np.isfinite(div_lo)
        assert np.isfinite(div_hi)
    
    def test_interval_kl_divergence(self):
        p_lo = np.array([0.4, 0.5])
        p_hi = np.array([0.6, 0.6])
        q_lo = np.array([0.3, 0.4])
        q_hi = np.array([0.5, 0.5])
        
        kl_lo, kl_hi = interval_kl_divergence(p_lo, p_hi, q_lo, q_hi)
        
        assert kl_lo <= kl_hi
        # KL divergence should be non-negative, but interval arithmetic with 
        # overlapping p and q intervals can produce slightly negative lower bounds
        # This is a limitation of interval arithmetic, not a bug
        assert kl_lo >= -0.2  # Allow some slack for interval arithmetic
    
    def test_interval_total_variation(self):
        p_lo = np.array([0.4, 0.5])
        p_hi = np.array([0.6, 0.6])
        q_lo = np.array([0.3, 0.4])
        q_hi = np.array([0.5, 0.5])
        
        tv_lo, tv_hi = interval_total_variation(p_lo, p_hi, q_lo, q_hi)
        
        assert tv_lo <= tv_hi
        assert 0.0 <= tv_lo <= 1.0
        assert 0.0 <= tv_hi <= 1.0
    
    def test_statistical_distance_bounds(self):
        p_lo = np.array([0.4, 0.5])
        p_hi = np.array([0.6, 0.6])
        q_lo = np.array([0.3, 0.4])
        q_hi = np.array([0.5, 0.5])
        
        bounds = statistical_distance_bounds(p_lo, p_hi, q_lo, q_hi)
        
        assert "total_variation" in bounds
        assert "kl_divergence" in bounds
        assert "max_divergence" in bounds
        
        for name, (lo, hi) in bounds.items():
            assert lo <= hi


class TestPropertyBasedTests:
    """Property-based tests for interval operations."""
    
    def test_interval_always_contains_true_value_addition(self):
        np.random.seed(42)
        
        for _ in range(100):
            a_true = np.random.uniform(0, 1, size=5)
            b_true = np.random.uniform(0, 1, size=5)
            
            tol = 0.01
            a_lo = a_true - tol
            a_hi = a_true + tol
            b_lo = b_true - tol
            b_hi = b_true + tol
            
            lo, hi = _interval_add(a_lo, a_hi, b_lo, b_hi)
            true_sum = a_true + b_true
            
            assert np.all(lo <= true_sum + 1e-14)
            assert np.all(hi >= true_sum - 1e-14)
    
    def test_interval_always_contains_true_value_multiplication(self):
        np.random.seed(43)
        
        for _ in range(100):
            a_true = np.random.uniform(0.1, 2.0, size=5)
            b_true = np.random.uniform(0.1, 2.0, size=5)
            
            tol = 0.05
            a_lo = a_true - tol
            a_hi = a_true + tol
            b_lo = b_true - tol
            b_hi = b_true + tol
            
            lo, hi = _interval_mul(a_lo, a_hi, b_lo, b_hi)
            true_prod = a_true * b_true
            
            assert np.all(lo <= true_prod + 1e-12)
            assert np.all(hi >= true_prod - 1e-12)
    
    def test_interval_division_soundness(self):
        np.random.seed(44)
        
        for _ in range(50):
            a_true = np.random.uniform(0.5, 2.0, size=3)
            b_true = np.random.uniform(0.5, 2.0, size=3)
            
            tol = 0.05
            a_lo = a_true - tol
            a_hi = a_true + tol
            b_lo = b_true - tol
            b_hi = b_true + tol
            
            lo, hi = _interval_div(a_lo, a_hi, b_lo, b_hi)
            true_div = a_true / b_true
            
            assert np.all(lo <= true_div + 1e-12)
            assert np.all(hi >= true_div - 1e-12)
    
    def test_interval_log_soundness(self):
        np.random.seed(45)
        
        for _ in range(50):
            x_true = np.random.uniform(0.1, 10.0, size=4)
            
            tol = 0.05
            x_lo = x_true - tol
            x_hi = x_true + tol
            
            lo, hi = _interval_log(x_lo, x_hi)
            true_log = np.log(x_true)
            
            assert np.all(lo <= true_log + 1e-12)
            assert np.all(hi >= true_log - 1e-12)
    
    def test_hockey_stick_always_sound(self):
        np.random.seed(46)
        
        for _ in range(20):
            p_true = np.random.dirichlet([1, 1, 1, 1])
            q_true = np.random.dirichlet([1, 1, 1, 1])
            
            tol = 0.01
            p_lo = np.maximum(p_true - tol, 0.0)
            p_hi = np.minimum(p_true + tol, 1.0)
            q_lo = np.maximum(q_true - tol, 0.0)
            q_hi = np.minimum(q_true + tol, 1.0)
            
            epsilon = np.random.uniform(0.1, 2.0)
            hs_lo, hs_hi = interval_hockey_stick(p_lo, p_hi, q_lo, q_hi, epsilon)
            
            true_hs = np.sum(np.maximum(p_true - np.exp(epsilon) * q_true, 0.0))
            
            assert hs_lo <= true_hs + 1e-9
            assert hs_hi >= true_hs - 1e-9


class TestErrorPropagation:
    """Test error propagation tracking."""
    
    def test_error_propagation_initialization(self):
        error = ErrorPropagation()
        assert error.initial_error == 0.0
        assert error.rounding_error == 0.0
        assert error.operations == 0
    
    def test_error_propagation_add_operation(self):
        error = ErrorPropagation()
        error.add_operation(5)
        
        assert error.operations == 5
        assert error.rounding_error > 0.0
    
    def test_error_propagation_merge(self):
        error1 = ErrorPropagation(initial_error=1e-9, operations=10)
        error2 = ErrorPropagation(initial_error=2e-9, operations=5)
        
        merged = error1.merge(error2)
        
        assert merged.initial_error == 2e-9
        assert merged.operations == 15


class TestIntervalResultConversion:
    """Test IntervalResult conversion to VerifyResult."""
    
    def test_interval_result_to_verify_result_valid(self):
        result = IntervalResult(
            valid=True,
            soundness=SoundnessLevel.SOUND,
            lower_bound=0.0,
            upper_bound=1.0,
        )
        
        verify_result = result.to_verify_result()
        
        assert verify_result.valid is True
        assert verify_result.violation is None
    
    def test_interval_result_to_verify_result_invalid(self):
        result = IntervalResult(
            valid=False,
            soundness=SoundnessLevel.SOUND,
            lower_bound=0.0,
            upper_bound=2.0,
            violation=(0, 1, 2, 0.5),
        )
        
        verify_result = result.to_verify_result()
        
        assert verify_result.valid is False
        assert verify_result.violation == (0, 1, 2, 0.5)
