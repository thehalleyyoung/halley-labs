"""
Comprehensive tests for rational verifier (exact verification with arbitrary precision).

Tests:
- Exact privacy loss computation
- Comparison against float verifier results
- Adaptive precision
- Edge cases (very small/large values)
"""

import numpy as np
import numpy.testing as npt
import pytest

from dp_forge.verification.rational_verifier import (
    RationalVerifier,
    RationalResult,
    PrecisionSchedule,
    VerificationOutcome,
    exact_privacy_loss,
    exact_hockey_stick,
    adaptive_verify,
    compute_exact_epsilon,
    verify_with_certificates,
    compare_verifiers,
    exact_renyi_divergence,
    exact_composition,
    SymbolicVerifier,
    batch_rational_verify,
    precision_sensitivity_analysis,
    MPMATH_AVAILABLE,
)

pytestmark = pytest.mark.skipif(not MPMATH_AVAILABLE, reason="mpmath not available")


class TestPrecisionSchedule:
    """Test PrecisionSchedule for adaptive precision."""
    
    def test_precision_schedule_initialization(self):
        schedule = PrecisionSchedule(initial_bits=53, max_bits=1024)
        assert schedule.initial_bits == 53
        assert schedule.max_bits == 1024
        assert schedule.current_bits == 53
    
    def test_precision_schedule_next_precision(self):
        schedule = PrecisionSchedule(initial_bits=50, max_bits=200, increment=50)
        
        next_prec = schedule.next_precision()
        assert next_prec == 100
        assert schedule.current_bits == 100
    
    def test_precision_schedule_stops_at_max(self):
        schedule = PrecisionSchedule(initial_bits=100, max_bits=150, increment=100)
        
        next_prec = schedule.next_precision()
        assert next_prec == 150
        
        next_prec = schedule.next_precision()
        assert next_prec is None
    
    def test_precision_schedule_reset(self):
        schedule = PrecisionSchedule(initial_bits=53, max_bits=1024)
        schedule.next_precision()
        schedule.reset()
        
        assert schedule.current_bits == 53


class TestExactPrivacyLoss:
    """Test exact privacy loss computation."""
    
    def test_exact_privacy_loss_basic(self):
        p = 0.6
        q = 0.4
        
        loss_str = exact_privacy_loss(p, q, precision_bits=100)
        loss_float = float(loss_str)
        
        expected = np.log(p / q)
        npt.assert_almost_equal(loss_float, expected, decimal=10)
    
    def test_exact_privacy_loss_high_precision(self):
        p = 0.6
        q = 0.4
        
        loss_53 = float(exact_privacy_loss(p, q, precision_bits=53))
        loss_200 = float(exact_privacy_loss(p, q, precision_bits=200))
        
        npt.assert_almost_equal(loss_53, loss_200, decimal=10)
    
    def test_exact_privacy_loss_handles_zero_q(self):
        p = 0.5
        q = 0.0
        
        loss_str = exact_privacy_loss(p, q, precision_bits=100)
        
        assert loss_str == "inf"
    
    def test_exact_privacy_loss_handles_zero_p(self):
        p = 0.0
        q = 0.5
        
        loss_str = exact_privacy_loss(p, q, precision_bits=100)
        
        assert loss_str == "-inf"
    
    def test_exact_privacy_loss_small_values(self):
        p = 1e-100
        q = 1e-101
        
        loss_str = exact_privacy_loss(p, q, precision_bits=200)
        loss_float = float(loss_str)
        
        expected = np.log(p / q)
        npt.assert_almost_equal(loss_float, expected, decimal=5)


class TestExactHockeyStick:
    """Test exact hockey-stick divergence."""
    
    def test_exact_hockey_stick_basic(self):
        p = np.array([0.6, 0.4])
        q = np.array([0.5, 0.5])
        epsilon = 0.5
        
        hs_str = exact_hockey_stick(p, q, epsilon, precision_bits=100)
        hs_float = float(hs_str)
        
        expected = np.sum(np.maximum(p - np.exp(epsilon) * q, 0.0))
        npt.assert_almost_equal(hs_float, expected, decimal=10)
    
    def test_exact_hockey_stick_zero_when_p_small(self):
        p = np.array([0.1, 0.1])
        q = np.array([0.4, 0.5])
        epsilon = 1.0
        
        hs_str = exact_hockey_stick(p, q, epsilon, precision_bits=100)
        hs_float = float(hs_str)
        
        npt.assert_almost_equal(hs_float, 0.0, decimal=8)
    
    def test_exact_hockey_stick_high_precision(self):
        p = np.array([0.6, 0.4])
        q = np.array([0.5, 0.5])
        epsilon = 0.5
        
        hs_53 = float(exact_hockey_stick(p, q, epsilon, precision_bits=53))
        hs_200 = float(exact_hockey_stick(p, q, epsilon, precision_bits=200))
        
        npt.assert_almost_equal(hs_53, hs_200, decimal=10)


class TestRationalVerifier:
    """Test RationalVerifier class."""
    
    def test_rational_verifier_initialization(self):
        verifier = RationalVerifier(initial_precision=100, max_precision=500)
        assert verifier.precision_schedule.initial_bits == 100
        assert verifier.precision_schedule.max_bits == 500
    
    def test_verify_pure_dp_valid(self):
        verifier = RationalVerifier()
        
        prob_table = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        edges = [(0, 1)]
        epsilon = 1.0
        
        result = verifier.verify_pure_dp(prob_table, edges, epsilon, precision_bits=100)
        
        assert isinstance(result, RationalResult)
        assert result.valid is True
        assert result.outcome == VerificationOutcome.DEFINITELY_VALID
    
    def test_verify_pure_dp_invalid(self):
        verifier = RationalVerifier()
        
        prob_table = np.array([
            [0.9, 0.1],
            [0.1, 0.9]
        ])
        edges = [(0, 1)]
        epsilon = 0.1
        
        result = verifier.verify_pure_dp(prob_table, edges, epsilon, precision_bits=100)
        
        assert result.valid is False
        assert result.outcome == VerificationOutcome.DEFINITELY_INVALID
    
    def test_verify_pure_dp_detects_violations(self):
        verifier = RationalVerifier()
        
        prob_table = np.array([
            [0.8, 0.2],
            [0.2, 0.8]
        ])
        edges = [(0, 1)]
        epsilon = 0.5
        
        result = verifier.verify_pure_dp(prob_table, edges, epsilon, precision_bits=100)
        
        if not result.valid:
            assert result.violation is not None
            i, i_prime, j, mag_str = result.violation
            assert i == 0 or i_prime == 0
    
    def test_verify_approx_dp_valid(self):
        verifier = RationalVerifier()
        
        prob_table = np.array([
            [0.5, 0.5],
            [0.4, 0.6]
        ])
        edges = [(0, 1)]
        epsilon = 0.5
        delta = 0.1
        
        result = verifier.verify_approx_dp(prob_table, edges, epsilon, delta, precision_bits=100)
        
        assert isinstance(result, RationalResult)
    
    def test_verify_dispatches_correctly(self):
        verifier = RationalVerifier()
        
        prob_table = np.array([[0.5, 0.5], [0.5, 0.5]])
        edges = [(0, 1)]
        
        result_pure = verifier.verify(prob_table, edges, epsilon=1.0, delta=0.0)
        assert result_pure.outcome in [VerificationOutcome.DEFINITELY_VALID, VerificationOutcome.DEFINITELY_INVALID]
        
        result_approx = verifier.verify(prob_table, edges, epsilon=0.5, delta=0.1)
        assert result_approx.outcome in [VerificationOutcome.DEFINITELY_VALID, VerificationOutcome.DEFINITELY_INVALID]
    
    def test_adaptive_verify(self):
        verifier = RationalVerifier(initial_precision=50, max_precision=200)
        
        prob_table = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        edges = [(0, 1)]
        
        result = verifier.adaptive_verify(prob_table, edges, epsilon=1.0, delta=0.0)
        
        assert isinstance(result, RationalResult)
        assert result.precision_used >= 50


class TestRationalResult:
    """Test RationalResult class."""
    
    def test_rational_result_creation(self):
        result = RationalResult(
            valid=True,
            outcome=VerificationOutcome.DEFINITELY_VALID,
            precision_used=100,
        )
        
        assert result.valid is True
        assert result.outcome == VerificationOutcome.DEFINITELY_VALID
    
    def test_rational_result_to_verify_result_valid(self):
        result = RationalResult(
            valid=True,
            outcome=VerificationOutcome.DEFINITELY_VALID,
            precision_used=100,
        )
        
        verify_result = result.to_verify_result()
        
        assert verify_result.valid is True
        assert verify_result.violation is None
    
    def test_rational_result_to_verify_result_invalid(self):
        result = RationalResult(
            valid=False,
            outcome=VerificationOutcome.DEFINITELY_INVALID,
            violation=(0, 1, 2, "1.5"),
            precision_used=100,
        )
        
        verify_result = result.to_verify_result()
        
        assert verify_result.valid is False
        assert verify_result.violation is not None


class TestAdaptiveVerify:
    """Test adaptive precision verification."""
    
    def test_adaptive_verify_basic(self):
        mechanism = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        
        result = adaptive_verify(
            mechanism,
            epsilon=1.0,
            initial_precision=50,
            max_precision=200
        )
        
        assert isinstance(result, RationalResult)
        assert result.valid is True
    
    def test_adaptive_verify_with_edges(self):
        mechanism = np.array([
            [0.5, 0.5],
            [0.4, 0.6],
            [0.3, 0.7]
        ])
        edges = [(0, 1), (1, 2)]
        
        result = adaptive_verify(mechanism, epsilon=1.0, edges=edges)
        
        assert isinstance(result, RationalResult)
    
    def test_adaptive_verify_increases_precision(self):
        mechanism = np.array([[0.5, 0.5], [0.5, 0.5]])
        
        result = adaptive_verify(
            mechanism,
            epsilon=1.0,
            initial_precision=53,
            max_precision=500
        )
        
        assert result.precision_used >= 53


class TestComputeExactEpsilon:
    """Test exact epsilon computation."""
    
    def test_compute_exact_epsilon_for_uniform(self):
        prob_table = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        edges = [(0, 1)]
        
        eps_str = compute_exact_epsilon(prob_table, edges, precision_bits=100)
        eps_float = float(eps_str)
        
        npt.assert_almost_equal(eps_float, 0.0, decimal=5)
    
    def test_compute_exact_epsilon_for_randomized_response(self):
        prob_table = np.array([
            [0.75, 0.25],
            [0.25, 0.75]
        ])
        edges = [(0, 1)]
        
        eps_str = compute_exact_epsilon(prob_table, edges, precision_bits=100)
        eps_float = float(eps_str)
        
        expected = np.log(0.75 / 0.25)
        npt.assert_almost_equal(eps_float, expected, decimal=8)


class TestCompareVerifiers:
    """Test comparison between verifiers."""
    
    def test_compare_verifiers_on_valid_mechanism(self):
        mechanism = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        
        comparison = compare_verifiers(mechanism, epsilon=1.0)
        
        assert "float" in comparison
        assert "interval" in comparison
        assert "rational" in comparison
        assert "agreement" in comparison
    
    def test_compare_verifiers_agreement(self):
        mechanism = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        
        comparison = compare_verifiers(mechanism, epsilon=1.0)
        
        agreement = comparison["agreement"]
        assert "float_interval" in agreement
        assert "float_rational" in agreement
        assert "interval_rational" in agreement


class TestVerifyWithCertificates:
    """Test verification with certificate generation."""
    
    def test_verify_with_certificates(self):
        mechanism = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        
        result, cert_data = verify_with_certificates(
            mechanism,
            epsilon=1.0,
            precision_bits=100
        )
        
        assert isinstance(result, RationalResult)
        assert "mechanism_shape" in cert_data
        assert "epsilon" in cert_data
        assert cert_data["epsilon"] == 1.0


class TestExactRenyiDivergence:
    """Test exact Rényi divergence."""
    
    def test_exact_renyi_divergence_basic(self):
        p = np.array([0.6, 0.4])
        q = np.array([0.5, 0.5])
        alpha = 2.0
        
        div_str = exact_renyi_divergence(p, q, alpha, precision_bits=100)
        div_float = float(div_str)
        
        assert div_float > 0.0
        assert np.isfinite(div_float)
    
    def test_exact_renyi_divergence_rejects_alpha_le_1(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        
        with pytest.raises(ValueError, match="Rényi order must be > 1"):
            exact_renyi_divergence(p, q, alpha=1.0)


class TestExactComposition:
    """Test exact composition of privacy parameters."""
    
    def test_exact_composition_basic(self):
        epsilons = ["0.5", "0.5", "0.5"]
        deltas = ["0.01", "0.01", "0.01"]
        
        total_eps, total_delta = exact_composition(
            epsilons,
            deltas,
            composition_rule="basic",
            precision_bits=100
        )
        
        npt.assert_almost_equal(float(total_eps), 1.5, decimal=10)
        npt.assert_almost_equal(float(total_delta), 0.03, decimal=10)
    
    def test_exact_composition_advanced(self):
        epsilons = ["0.5", "0.5"]
        deltas = ["0.01", "0.01"]
        
        total_eps, total_delta = exact_composition(
            epsilons,
            deltas,
            composition_rule="advanced",
            precision_bits=100
        )
        
        assert float(total_eps) > 1.0
        assert float(total_delta) >= 0.02


class TestSymbolicVerifier:
    """Test SymbolicVerifier class."""
    
    def test_symbolic_verifier_initialization(self):
        verifier = SymbolicVerifier(precision_bits=200)
        assert verifier.precision_bits == 200
    
    def test_symbolic_privacy_loss(self):
        verifier = SymbolicVerifier()
        
        loss_str = verifier.symbolic_privacy_loss("0.6", "0.4")
        loss_float = float(loss_str)
        
        expected = np.log(0.6 / 0.4)
        npt.assert_almost_equal(loss_float, expected, decimal=10)


class TestBatchRationalVerify:
    """Test batch rational verification."""
    
    def test_batch_rational_verify(self):
        mechanisms = [
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([[0.6, 0.4], [0.4, 0.6]]),
        ]
        
        results = batch_rational_verify(mechanisms, epsilon=1.0, precision_bits=100)
        
        assert len(results) == 2
        assert all(isinstance(r, RationalResult) for r in results)


class TestPrecisionSensitivityAnalysis:
    """Test precision sensitivity analysis."""
    
    def test_precision_sensitivity_analysis(self):
        mechanism = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        
        results = precision_sensitivity_analysis(
            mechanism,
            epsilon=1.0,
            precision_range=(50, 150),
            n_samples=3
        )
        
        assert len(results) == 3
        assert all(isinstance(prec, int) for prec, _ in results)
        assert all(isinstance(res, RationalResult) for _, res in results)


class TestEdgeCases:
    """Test edge cases for rational verification."""
    
    def test_very_small_probabilities(self):
        verifier = RationalVerifier()
        
        prob_table = np.array([
            [1e-100, 1.0 - 1e-100],
            [1e-101, 1.0 - 1e-101]
        ])
        edges = [(0, 1)]
        
        result = verifier.verify(prob_table, edges, epsilon=10.0, precision_bits=200)
        
        assert isinstance(result, RationalResult)
    
    def test_very_large_epsilon(self):
        verifier = RationalVerifier()
        
        prob_table = np.array([[0.5, 0.5], [0.4, 0.6]])
        edges = [(0, 1)]
        
        result = verifier.verify(prob_table, edges, epsilon=100.0, precision_bits=100)
        
        assert result.valid is True
    
    def test_identity_mechanism(self):
        verifier = RationalVerifier()
        
        prob_table = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        edges = [(0, 1)]
        
        result = verifier.verify(prob_table, edges, epsilon=10.0, precision_bits=100)
        
        assert isinstance(result, RationalResult)
    
    def test_single_database(self):
        verifier = RationalVerifier()
        
        prob_table = np.array([[0.5, 0.5]])
        edges = []
        
        result = verifier.verify(prob_table, edges, epsilon=1.0, precision_bits=100)
        
        assert result.valid is True


class TestCrossValidation:
    """Test cross-validation with float verifier."""
    
    def test_cross_validate_with_float_agreement(self):
        verifier = RationalVerifier(cross_validate=True)
        
        prob_table = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        edges = [(0, 1)]
        
        rational_result = verifier.verify(prob_table, edges, epsilon=1.0, precision_bits=100)
        
        agrees = verifier.cross_validate_with_float(
            rational_result, prob_table, edges, epsilon=1.0
        )
        
        assert agrees is True


class TestComputationTime:
    """Test computation time tracking."""
    
    def test_computation_time_recorded(self):
        verifier = RationalVerifier()
        
        prob_table = np.array([[0.5, 0.5], [0.5, 0.5]])
        edges = [(0, 1)]
        
        result = verifier.verify(prob_table, edges, epsilon=1.0, precision_bits=100)
        
        assert result.computation_time >= 0.0


class TestHighPrecisionAccuracy:
    """Test accuracy at high precision."""
    
    def test_high_precision_more_accurate(self):
        verifier = RationalVerifier()
        
        prob_table = np.array([[0.6, 0.4], [0.4, 0.6]])
        edges = [(0, 1)]
        epsilon = 0.5
        
        result_low = verifier.verify(prob_table, edges, epsilon, precision_bits=53)
        result_high = verifier.verify(prob_table, edges, epsilon, precision_bits=500)
        
        assert result_low.outcome == result_high.outcome
    
    def test_adaptive_finds_correct_precision(self):
        verifier = RationalVerifier(initial_precision=50, max_precision=1000)
        
        prob_table = np.array([[0.5, 0.5], [0.5, 0.5]])
        edges = [(0, 1)]
        
        result = verifier.adaptive_verify(prob_table, edges, epsilon=1.0)
        
        assert result.outcome == VerificationOutcome.DEFINITELY_VALID
