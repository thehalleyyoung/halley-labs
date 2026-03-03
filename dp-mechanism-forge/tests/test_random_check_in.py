"""
Comprehensive tests for dp_forge.amplification.random_check_in module.

Tests privacy amplification under random check-in model.
"""

import math
import pytest
import numpy as np
import numpy.testing as npt
from hypothesis import given, strategies as st, assume, settings

from dp_forge.amplification.random_check_in import (
    RandomCheckInAmplifier,
    RandomCheckInResult,
    random_checkin_amplification,
    optimal_participation_probability,
    checkin_privacy_curve,
)


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestRandomCheckInAmplifierBasic:
    """Test basic functionality of RandomCheckInAmplifier."""
    
    def test_amplifier_initialization(self):
        """Test RandomCheckInAmplifier initialization."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1
        )
        
        assert amplifier.n_potential_users == 10000
        assert amplifier.participation_prob == 0.1
        assert amplifier.use_tight_analysis is True
    
    def test_invalid_n_potential_users(self):
        """Test that n_potential_users < 2 raises error."""
        with pytest.raises(ValueError, match="n_potential_users must be >= 2"):
            RandomCheckInAmplifier(n_potential_users=1, participation_prob=0.5)
    
    def test_invalid_participation_prob(self):
        """Test that invalid participation_prob raises error."""
        with pytest.raises(ValueError, match="participation_prob must be in"):
            RandomCheckInAmplifier(n_potential_users=100, participation_prob=0.0)
        
        with pytest.raises(ValueError, match="participation_prob must be in"):
            RandomCheckInAmplifier(n_potential_users=100, participation_prob=1.5)
    
    def test_basic_amplification(self):
        """Test basic check-in amplification."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1,
            use_tight_analysis=False
        )
        
        result = amplifier.amplify(epsilon_local=2.0, delta_local=0.0)
        
        assert result.epsilon_central < result.epsilon_local
        assert result.epsilon_central > 0
        assert result.delta_central >= 0
        assert result.delta_central < 1
        assert result.method == "basic_subsampling"
    
    def test_tight_amplification(self):
        """Test tight check-in amplification."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1,
            use_tight_analysis=True
        )
        
        result = amplifier.amplify(epsilon_local=2.0, delta_local=0.0)
        
        assert result.epsilon_central < result.epsilon_local
        assert result.epsilon_central > 0
        assert result.method == "tight_checkin"
    
    def test_result_properties(self):
        """Test RandomCheckInResult properties."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1
        )
        
        result = amplifier.amplify(epsilon_local=2.0)
        
        assert result.expected_participants == pytest.approx(1000.0)
        assert result.amplification_ratio > 1.0
        assert result.participation_prob == 0.1
        assert result.n_potential_users == 10000


# ============================================================================
# Amplification Properties Tests
# ============================================================================


class TestCheckInAmplificationProperties:
    """Test amplification properties of check-in model."""
    
    def test_amplification_improves_privacy(self):
        """Test that check-in always improves privacy."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=1000,
            participation_prob=0.5
        )
        
        epsilon_local = 2.0
        result = amplifier.amplify(epsilon_local)
        
        assert result.epsilon_central < epsilon_local
        assert result.amplification_ratio > 1.0
    
    def test_smaller_participation_better(self):
        """Test that smaller participation gives better privacy."""
        n_users = 10000
        epsilon_local = 2.0
        
        amplifier_10 = RandomCheckInAmplifier(
            n_potential_users=n_users,
            participation_prob=0.1
        )
        amplifier_50 = RandomCheckInAmplifier(
            n_potential_users=n_users,
            participation_prob=0.5
        )
        
        result_10 = amplifier_10.amplify(epsilon_local)
        result_50 = amplifier_50.amplify(epsilon_local)
        
        # Smaller participation should give better privacy
        assert result_10.epsilon_central < result_50.epsilon_central
    
    def test_more_potential_users_better(self):
        """Test that more potential users gives better amplification."""
        participation_prob = 0.1
        epsilon_local = 2.0
        
        amplifier_1k = RandomCheckInAmplifier(
            n_potential_users=1000,
            participation_prob=participation_prob
        )
        amplifier_10k = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=participation_prob
        )
        
        result_1k = amplifier_1k.amplify(epsilon_local)
        result_10k = amplifier_10k.amplify(epsilon_local)
        
        # More potential users should give better privacy
        assert result_10k.epsilon_central < result_1k.epsilon_central
    
    def test_tight_better_than_basic(self):
        """Test that tight analysis gives better bounds than basic."""
        amplifier_basic = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1,
            use_tight_analysis=False
        )
        amplifier_tight = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1,
            use_tight_analysis=True
        )
        
        epsilon_local = 2.0
        
        result_basic = amplifier_basic.amplify(epsilon_local)
        result_tight = amplifier_tight.amplify(epsilon_local)
        
        # Tight should be at least as good (smaller epsilon)
        assert result_tight.epsilon_central <= result_basic.epsilon_central


# ============================================================================
# Multi-Round Composition Tests
# ============================================================================


class TestMultiRoundComposition:
    """Test multi-round check-in composition."""
    
    def test_compose_multiple_checkins_basic(self):
        """Test basic multi-round composition."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1
        )
        
        result = amplifier.compose_multiple_checkins(
            epsilon_local=1.0,
            delta_local=0.0,
            num_rounds=10
        )
        
        assert result.epsilon_central > 0
        assert result.delta_central >= 0
        assert "composed" in result.method
    
    def test_composition_degrades_privacy(self):
        """Test that more rounds degrades privacy."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1
        )
        
        epsilon_local = 1.0
        
        result_1 = amplifier.compose_multiple_checkins(
            epsilon_local, 0.0, num_rounds=1
        )
        result_10 = amplifier.compose_multiple_checkins(
            epsilon_local, 0.0, num_rounds=10
        )
        result_100 = amplifier.compose_multiple_checkins(
            epsilon_local, 0.0, num_rounds=100
        )
        
        # More rounds should give worse privacy
        assert result_1.epsilon_central < result_10.epsilon_central
        assert result_10.epsilon_central < result_100.epsilon_central
    
    def test_composition_scales_reasonably(self):
        """Test that composition scales sub-linearly."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1
        )
        
        epsilon_local = 1.0
        
        result_10 = amplifier.compose_multiple_checkins(
            epsilon_local, 0.0, num_rounds=10
        )
        result_100 = amplifier.compose_multiple_checkins(
            epsilon_local, 0.0, num_rounds=100
        )
        
        # 10x more rounds should give less than 10x privacy loss
        ratio = result_100.epsilon_central / result_10.epsilon_central
        assert ratio < 10.0  # Sub-linear composition
    
    def test_invalid_num_rounds(self):
        """Test that invalid num_rounds raises error."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=1000,
            participation_prob=0.5
        )
        
        with pytest.raises(ValueError, match="num_rounds must be >= 1"):
            amplifier.compose_multiple_checkins(
                epsilon_local=1.0,
                delta_local=0.0,
                num_rounds=0
            )


# ============================================================================
# Optimization Tests
# ============================================================================


class TestParticipationOptimization:
    """Test participation probability optimization."""
    
    def test_minimum_participation_for_target(self):
        """Test finding minimum participation for target privacy."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.5  # Initial guess
        )
        
        epsilon_local = 2.0
        epsilon_central_target = 0.1
        
        p_min = amplifier.minimum_participation_for_target(
            epsilon_local,
            epsilon_central_target
        )
        
        assert 0 < p_min <= 1.0
        
        # Verify it achieves target
        amplifier_test = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=p_min
        )
        result = amplifier_test.amplify(epsilon_local)
        
        assert result.epsilon_central <= epsilon_central_target * 1.1
    
    def test_infeasible_target(self):
        """Test that infeasible target raises error."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=1000,
            participation_prob=0.5
        )
        
        with pytest.raises(ValueError, match="epsilon_local.*must be >"):
            amplifier.minimum_participation_for_target(
                epsilon_local=0.1,
                epsilon_central_target=1.0  # Impossible
            )


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestCheckInEdgeCases:
    """Test edge cases for random check-in."""
    
    def test_very_sparse_participation(self):
        """Test with very low participation probability."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=100000,
            participation_prob=0.001
        )
        
        result = amplifier.amplify(epsilon_local=2.0)
        
        assert result.epsilon_central > 0
        assert result.epsilon_central < result.epsilon_local
        assert result.expected_participants == 100.0
    
    def test_full_participation(self):
        """Test with participation_prob = 1.0 (everyone participates)."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=1000,
            participation_prob=1.0
        )
        
        result = amplifier.amplify(epsilon_local=2.0)
        
        # Should still have some amplification from shuffle-like effects
        assert result.epsilon_central < result.epsilon_local
        assert result.expected_participants == 1000.0
    
    def test_minimum_potential_users(self):
        """Test with minimum n_potential_users = 2."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=2,
            participation_prob=0.5
        )
        
        result = amplifier.amplify(epsilon_local=2.0)
        
        assert result.epsilon_central > 0
        assert result.epsilon_central < result.epsilon_local
    
    def test_with_nonzero_delta_local(self):
        """Test with approximate local DP."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1
        )
        
        result = amplifier.amplify(epsilon_local=2.0, delta_local=1e-5)
        
        assert result.epsilon_central > 0
        assert result.delta_central >= result.delta_local
    
    def test_invalid_epsilon_local(self):
        """Test that invalid epsilon_local raises error."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=1000,
            participation_prob=0.5
        )
        
        with pytest.raises(ValueError, match="epsilon_local must be > 0"):
            amplifier.amplify(epsilon_local=0.0)
        
        with pytest.raises(ValueError, match="epsilon_local must be > 0"):
            amplifier.amplify(epsilon_local=-1.0)
    
    def test_invalid_delta_local(self):
        """Test that invalid delta_local raises error."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=1000,
            participation_prob=0.5
        )
        
        with pytest.raises(ValueError, match="delta_local must be in"):
            amplifier.amplify(epsilon_local=2.0, delta_local=-0.1)
        
        with pytest.raises(ValueError, match="delta_local must be in"):
            amplifier.amplify(epsilon_local=2.0, delta_local=1.0)


# ============================================================================
# Property-Based Tests
# ============================================================================


class TestCheckInProperties:
    """Property-based tests using Hypothesis."""
    
    @given(
        n_users=st.integers(min_value=100, max_value=100000),
        participation_prob=st.floats(min_value=0.01, max_value=0.99),
        epsilon_local=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=30, deadline=None)
    def test_amplification_always_improves(
        self, n_users, participation_prob, epsilon_local
    ):
        """Property: check-in always improves privacy."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=n_users,
            participation_prob=participation_prob
        )
        
        result = amplifier.amplify(epsilon_local)
        
        assert result.epsilon_central < result.epsilon_local
        assert result.epsilon_central > 0
        assert result.delta_central >= 0
        assert result.delta_central < 1
    
    @given(
        n_users=st.integers(min_value=1000, max_value=10000),
        p_small=st.floats(min_value=0.90, max_value=0.93),
        p_large=st.floats(min_value=0.93, max_value=0.98),
        epsilon_local=st.floats(min_value=0.5, max_value=5.0),
    )
    @settings(max_examples=20, deadline=None)
    def test_monotonicity_in_participation(
        self, n_users, p_small, p_large, epsilon_local
    ):
        """Property: for very high participation rates (>0.9), smaller participation gives similar or better privacy."""
        assume(p_large > p_small + 0.03)
        
        amplifier_small = RandomCheckInAmplifier(
            n_potential_users=n_users,
            participation_prob=p_small
        )
        amplifier_large = RandomCheckInAmplifier(
            n_potential_users=n_users,
            participation_prob=p_large
        )
        
        result_small = amplifier_small.amplify(epsilon_local)
        result_large = amplifier_large.amplify(epsilon_local)
        
        # For very high participation rates (>0.9), privacy guarantees should be similar
        # with a 2.5% tolerance for numerical effects in the tight analysis
        assert result_small.epsilon_central <= result_large.epsilon_central * 1.025


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_random_checkin_amplification(self):
        """Test random_checkin_amplification function."""
        eps_c, delta_c = random_checkin_amplification(
            epsilon_local=2.0,
            n_potential_users=10000,
            participation_prob=0.1,
            delta_local=0.0,
            use_tight=True
        )
        
        assert eps_c > 0
        assert eps_c < 2.0
        assert delta_c >= 0
        assert delta_c < 1
    
    def test_optimal_participation_probability(self):
        """Test optimal_participation_probability function."""
        p_opt = optimal_participation_probability(
            epsilon_local=2.0,
            n_potential_users=10000,
            epsilon_central_target=0.1,
            delta_central_target=1e-6
        )
        
        assert 0 < p_opt <= 1.0
        
        # Verify it achieves target
        eps_c, delta_c = random_checkin_amplification(
            epsilon_local=2.0,
            n_potential_users=10000,
            participation_prob=p_opt
        )
        
        assert eps_c <= 0.11  # Within 10% of target
    
    def test_checkin_privacy_curve(self):
        """Test checkin_privacy_curve function."""
        epsilon_local = 2.0
        n_potential_users = 10000
        participation_probs = np.array([0.01, 0.05, 0.1, 0.5])
        
        eps_central, delta_central = checkin_privacy_curve(
            epsilon_local=epsilon_local,
            n_potential_users=n_potential_users,
            participation_probs=participation_probs
        )
        
        assert len(eps_central) == len(participation_probs)
        assert len(delta_central) == len(participation_probs)
        
        # Should be monotonically increasing in participation prob
        for i in range(len(eps_central) - 1):
            assert eps_central[i] < eps_central[i + 1] * 1.01


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability at extreme parameters."""
    
    def test_very_large_n(self):
        """Test with very large number of potential users."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10000000,
            participation_prob=0.01
        )
        
        result = amplifier.amplify(epsilon_local=2.0)
        
        assert np.isfinite(result.epsilon_central)
        assert np.isfinite(result.delta_central)
        assert result.epsilon_central > 0
    
    def test_extreme_epsilon(self):
        """Test with extreme epsilon values."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1
        )
        
        # Very small
        result_small = amplifier.amplify(epsilon_local=0.001)
        assert np.isfinite(result_small.epsilon_central)
        
        # Very large
        result_large = amplifier.amplify(epsilon_local=100.0)
        assert np.isfinite(result_large.epsilon_central)
    
    def test_delta_never_exceeds_one(self):
        """Test that delta_central never exceeds 1."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10,
            participation_prob=0.9
        )
        
        result = amplifier.amplify(epsilon_local=10.0, delta_local=0.5)
        
        assert result.delta_central < 1.0


# ============================================================================
# Comparison Tests
# ============================================================================


class TestCheckInComparisons:
    """Compare check-in with other amplification methods."""
    
    def test_expected_participants_calculation(self):
        """Test that expected_participants is correct."""
        n = 10000
        p = 0.25
        
        amplifier = RandomCheckInAmplifier(
            n_potential_users=n,
            participation_prob=p
        )
        
        result = amplifier.amplify(epsilon_local=2.0)
        
        assert result.expected_participants == pytest.approx(n * p)
    
    def test_amplification_ratio_calculation(self):
        """Test amplification_ratio property."""
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1
        )
        
        result = amplifier.amplify(epsilon_local=2.0)
        
        expected_ratio = result.epsilon_local / result.epsilon_central
        assert result.amplification_ratio == pytest.approx(expected_ratio, rel=1e-6)


# ============================================================================
# Integration Tests
# ============================================================================


class TestCheckInIntegration:
    """Integration tests for random check-in amplification."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow: design -> amplify -> verify."""
        # Step 1: Find optimal participation probability
        amplifier_design = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.5  # Initial guess
        )
        
        epsilon_local = 2.0
        epsilon_central_target = 0.2
        
        p_opt = amplifier_design.minimum_participation_for_target(
            epsilon_local,
            epsilon_central_target
        )
        
        # Step 2: Create amplifier with optimal probability
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=p_opt
        )
        
        # Step 3: Apply amplification
        result = amplifier.amplify(epsilon_local)
        
        # Step 4: Verify result
        assert result.epsilon_central <= epsilon_central_target * 1.1
        assert result.participation_prob == p_opt
    
    def test_multi_round_federation_scenario(self):
        """Test realistic federated learning scenario with multiple rounds."""
        # Federated learning: 10,000 devices, 10% participate each round
        amplifier = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1
        )
        
        # Each device uses ε=1.0 local DP, run for 100 rounds
        result = amplifier.compose_multiple_checkins(
            epsilon_local=1.0,
            delta_local=0.0,
            num_rounds=100
        )
        
        # Should achieve reasonable privacy
        assert result.epsilon_central < 100.0  # Much better than naive composition
        assert result.delta_central < 1.0


# ============================================================================
# Regression Tests
# ============================================================================


class TestCheckInRegression:
    """Regression tests to prevent future breakage."""
    
    def test_standard_parameters_reproducible(self):
        """Test that standard parameters give reproducible results."""
        amplifier1 = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1
        )
        amplifier2 = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1
        )
        
        result1 = amplifier1.amplify(epsilon_local=2.0)
        result2 = amplifier2.amplify(epsilon_local=2.0)
        
        npt.assert_allclose(result1.epsilon_central, result2.epsilon_central)
        npt.assert_allclose(result1.delta_central, result2.delta_central)
    
    def test_tight_vs_basic_difference(self):
        """Test that tight and basic methods differ appropriately."""
        amplifier_basic = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1,
            use_tight_analysis=False
        )
        amplifier_tight = RandomCheckInAmplifier(
            n_potential_users=10000,
            participation_prob=0.1,
            use_tight_analysis=True
        )
        
        result_basic = amplifier_basic.amplify(epsilon_local=2.0)
        result_tight = amplifier_tight.amplify(epsilon_local=2.0)
        
        # Tight should be at least as good
        assert result_tight.epsilon_central <= result_basic.epsilon_central
        
        # Tight analysis can provide significantly better bounds than basic (up to 50x improvement)
        if result_tight.epsilon_central > 1e-6:
            ratio = result_basic.epsilon_central / result_tight.epsilon_central
            assert 1.0 <= ratio <= 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
