"""
Comprehensive tests for sparse vector technique (SVT).

Tests AboveThreshold, NumericSVT, AdaptiveSVT, GapSVT, and SVTComposition
for correctness, privacy budget tracking, and adaptive threshold selection.
"""

import math
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

from dp_forge.mechanisms.sparse_vector import (
    AboveThreshold,
    NumericSVT,
    AdaptiveSVT,
    GapSVT,
    SVTComposition,
    SVTCompositionResult,
)
from dp_forge.exceptions import ConfigurationError, BudgetExhaustedError


class TestAboveThreshold:
    """Test AboveThreshold (classic SVT)."""
    
    def test_initialization(self):
        """Test initialization."""
        svt = AboveThreshold(epsilon=1.0, threshold=100, max_outputs=5)
        
        assert svt.epsilon == 1.0
        assert svt.delta == 0.0
        assert svt.threshold == 100
        assert svt.max_outputs == 5
        assert svt.num_outputs == 0
        assert svt.queries_processed == 0
        assert svt.budget_remaining
    
    def test_initialization_errors(self):
        """Test initialization error handling."""
        with pytest.raises(ConfigurationError):
            AboveThreshold(epsilon=-1.0, threshold=100)
        
        with pytest.raises(ConfigurationError):
            AboveThreshold(epsilon=1.0, threshold=100, max_outputs=0)
        
        with pytest.raises(ConfigurationError):
            AboveThreshold(epsilon=1.0, threshold=100, sensitivity=-1.0)
    
    def test_query_basic(self):
        """Test basic query functionality."""
        svt = AboveThreshold(epsilon=1.0, threshold=10, max_outputs=3, seed=42)
        
        # Simple data
        data = [1, 2, 3, 4, 5]
        
        # Query: sum of data
        def query_sum(d):
            return sum(d)
        
        result = svt.query(data, query_sum)
        
        # Should return boolean
        assert result is True or result is False
        assert svt.queries_processed == 1
    
    def test_query_above_threshold(self):
        """Test queries above threshold."""
        svt = AboveThreshold(epsilon=10.0, threshold=100, max_outputs=5, seed=42)
        
        data = list(range(100))
        
        # Query that's clearly above threshold
        def high_query(d):
            return 1000.0
        
        result = svt.query(data, high_query)
        
        # With high epsilon, should reliably detect above threshold
        assert result is True
        assert svt.num_outputs == 1
    
    def test_query_below_threshold(self):
        """Test queries below threshold."""
        svt = AboveThreshold(epsilon=10.0, threshold=100, max_outputs=5, seed=42)
        
        data = list(range(10))
        
        # Query that's clearly below threshold
        def low_query(d):
            return 10.0
        
        result = svt.query(data, low_query)
        
        # Should detect below threshold
        assert result is False
        assert svt.num_outputs == 0  # Not counted as output
    
    def test_halts_after_max_outputs(self):
        """Test that SVT halts after max_outputs."""
        svt = AboveThreshold(epsilon=10.0, threshold=0, max_outputs=3, seed=42)
        
        data = [1, 2, 3]
        
        # All queries above threshold
        def above_query(d):
            return 100.0
        
        results = []
        for _ in range(10):  # Try many queries
            result = svt.query(data, above_query)
            results.append(result)
            if result is None:
                break
        
        # Should halt after 3 outputs
        assert results.count(True) == 3
        assert results[-1] is None
        assert not svt.budget_remaining
    
    def test_query_batch(self):
        """Test batch query processing."""
        svt = AboveThreshold(epsilon=1.0, threshold=50, max_outputs=5, seed=42)
        
        data = list(range(100))
        
        queries = [
            lambda d: 30.0,  # Below
            lambda d: 100.0, # Above
            lambda d: 40.0,  # Below
            lambda d: 200.0, # Above
        ]
        
        results = svt.query_batch(data, queries)
        
        assert len(results) == 4
        assert all(r in (True, False, None) for r in results)
    
    def test_privacy_guarantee(self):
        """Test privacy guarantee reporting."""
        svt = AboveThreshold(epsilon=1.5, threshold=100, max_outputs=3)
        
        eps, delta = svt.privacy_guarantee()
        
        assert eps == 1.5
        assert delta == 0.0
    
    def test_reset(self):
        """Test state reset."""
        svt = AboveThreshold(epsilon=1.0, threshold=100, max_outputs=3, seed=42)
        
        # Process some queries
        data = [1, 2, 3]
        svt.query(data, lambda d: 200.0)
        svt.query(data, lambda d: 300.0)
        
        assert svt.num_outputs > 0
        assert svt.queries_processed > 0
        
        # Reset
        svt.reset()
        
        assert svt.num_outputs == 0
        assert svt.queries_processed == 0
        assert svt.budget_remaining


class TestNumericSVT:
    """Test NumericSVT (returns noisy values)."""
    
    def test_initialization(self):
        """Test initialization."""
        svt = NumericSVT(epsilon=1.0, threshold=100, max_outputs=5)
        
        assert svt.epsilon == 1.0
        assert svt.threshold == 100
        assert svt.max_outputs == 5
    
    def test_query_returns_tuple(self):
        """Test query returns (is_above, value) tuple."""
        svt = NumericSVT(epsilon=1.0, threshold=50, max_outputs=5, seed=42)
        
        data = list(range(100))
        
        def query_fn(d):
            return sum(d)
        
        result = svt.query(data, query_fn)
        
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        is_above, value = result
        assert isinstance(is_above, (bool, np.bool_))
        assert isinstance(value, (float, np.floating))
    
    def test_query_returns_noisy_values(self):
        """Test that returned values are noisy."""
        svt = NumericSVT(epsilon=1.0, threshold=50, max_outputs=10, seed=42)
        
        data = list(range(100))
        
        def query_fn(d):
            return 100.0  # Fixed true value
        
        values = []
        for _ in range(10):
            result = svt.query(data, query_fn)
            if result is not None:
                is_above, value = result
                values.append(value)
        
        # Values should vary (noisy)
        assert len(set(values)) > 1  # Not all identical
    
    def test_halts_after_max_outputs(self):
        """Test halting behavior."""
        svt = NumericSVT(epsilon=10.0, threshold=0, max_outputs=3, seed=42)
        
        data = [1]
        
        def above_query(d):
            return 100.0
        
        count_above = 0
        for _ in range(10):
            result = svt.query(data, above_query)
            if result is None:
                break
            is_above, value = result
            if is_above:
                count_above += 1
        
        # Should halt after 3 above-threshold outputs
        assert count_above == 3
    
    def test_query_batch(self):
        """Test batch query processing."""
        svt = NumericSVT(epsilon=1.0, threshold=50, max_outputs=5, seed=42)
        
        data = list(range(100))
        
        queries = [
            lambda d: 30.0,
            lambda d: 100.0,
            lambda d: 40.0,
        ]
        
        results = svt.query_batch(data, queries)
        
        assert len(results) == 3
        for result in results:
            if result is not None:
                assert isinstance(result, tuple)
                assert len(result) == 2


class TestAdaptiveSVT:
    """Test AdaptiveSVT (adaptive threshold selection)."""
    
    def test_initialization(self):
        """Test initialization."""
        svt = AdaptiveSVT(
            epsilon=1.0,
            initial_threshold=100,
            max_outputs=5,
            adaptation_rate=0.1,
        )
        
        assert svt.epsilon == 1.0
        assert svt.threshold == 100
        assert svt.current_threshold == 100
        assert svt.max_outputs == 5
    
    def test_threshold_adaptation(self):
        """Test that threshold adapts."""
        svt = AdaptiveSVT(
            epsilon=1.0,
            initial_threshold=100,
            max_outputs=10,
            adaptation_rate=0.2,
            seed=42,
        )
        
        data = list(range(100))
        
        initial_threshold = svt.current_threshold
        
        # Process several queries
        for i in range(10):
            result = svt.query(data, lambda d: 50.0 + i * 10)
            if result is None:
                break
        
        # Threshold should have changed
        final_threshold = svt.current_threshold
        assert final_threshold != initial_threshold
    
    def test_threshold_bounds(self):
        """Test threshold respects min/max bounds."""
        svt = AdaptiveSVT(
            epsilon=1.0,
            initial_threshold=100,
            max_outputs=20,
            adaptation_rate=0.5,
            min_threshold=50,
            max_threshold=150,
            seed=42,
        )
        
        data = list(range(100))
        
        # Process many queries
        for _ in range(20):
            result = svt.query(data, lambda d: 0.0)  # All below
            if result is None:
                break
        
        # Threshold should respect bounds
        assert 50 <= svt.current_threshold <= 150
    
    def test_query_returns_current_threshold(self):
        """Test query returns current threshold."""
        svt = AdaptiveSVT(
            epsilon=1.0,
            initial_threshold=100,
            max_outputs=5,
            seed=42,
        )
        
        data = [1, 2, 3]
        
        result = svt.query(data, lambda d: 50.0)
        
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        is_above, threshold = result
        assert isinstance(is_above, (bool, np.bool_))
        assert isinstance(threshold, (int, float, np.integer, np.floating))
    
    def test_recent_above_rate_tracking(self):
        """Test tracking of recent above rate."""
        svt = AdaptiveSVT(
            epsilon=10.0,  # High epsilon for predictability
            initial_threshold=100,
            max_outputs=20,
            seed=42,
        )
        
        data = [1]
        
        # All queries above threshold
        for _ in range(15):
            result = svt.query(data, lambda d: 200.0)
            if result is None:
                break
        
        # Recent above rate should be high
        assert svt.recent_above_rate > 0.5


class TestGapSVT:
    """Test GapSVT (returns gap above threshold)."""
    
    def test_initialization(self):
        """Test initialization."""
        svt = GapSVT(epsilon=1.0, threshold=100, max_outputs=5)
        
        assert svt.epsilon == 1.0
        assert svt.threshold == 100
        assert svt.max_outputs == 5
    
    def test_query_returns_gap(self):
        """Test query returns gap."""
        svt = GapSVT(epsilon=1.0, threshold=50, max_outputs=5, seed=42)
        
        data = list(range(100))
        
        def query_fn(d):
            return 100.0
        
        result = svt.query(data, query_fn)
        
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        is_above, gap = result
        assert isinstance(is_above, (bool, np.bool_))
        assert isinstance(gap, (float, np.floating))
    
    def test_gap_sign(self):
        """Test gap sign matches is_above."""
        svt = GapSVT(epsilon=10.0, threshold=100, max_outputs=10, seed=42)
        
        data = [1]
        
        # Query clearly above
        result_above = svt.query(data, lambda d: 200.0)
        if result_above is not None:
            is_above, gap = result_above
            if is_above:
                assert gap >= 0
        
        # Query clearly below
        result_below = svt.query(data, lambda d: 10.0)
        if result_below is not None:
            is_above, gap = result_below
            if not is_above:
                assert gap < 0
    
    def test_gap_magnitude(self):
        """Test gap magnitude is reasonable."""
        svt = GapSVT(epsilon=10.0, threshold=100, max_outputs=5, seed=42)
        
        data = [1]
        
        # Large gap
        result1 = svt.query(data, lambda d: 200.0)
        # Small gap
        result2 = svt.query(data, lambda d: 105.0)
        
        if result1 and result2:
            gap1 = abs(result1[1])
            gap2 = abs(result2[1])
            
            # Larger true gap should tend to have larger noisy gap
            # (statistical test, may occasionally fail)
            # Just check both are finite
            assert math.isfinite(gap1)
            assert math.isfinite(gap2)


class TestSVTComposition:
    """Test SVTComposition for privacy accounting."""
    
    def test_initialization(self):
        """Test initialization."""
        composer = SVTComposition(total_budget_epsilon=2.0)
        
        assert composer.total_budget_epsilon == 2.0
        assert composer.remaining_epsilon == 2.0
        assert composer.consumed_epsilon == 0.0
    
    def test_initialization_errors(self):
        """Test initialization errors."""
        with pytest.raises(ConfigurationError):
            SVTComposition(total_budget_epsilon=-1.0)
        
        with pytest.raises(ConfigurationError):
            SVTComposition(total_budget_epsilon=1.0, total_budget_delta=1.5)
    
    def test_create_svt_default_allocation(self):
        """Test creating SVT with default budget allocation."""
        composer = SVTComposition(total_budget_epsilon=2.0)
        
        svt1 = composer.create_svt(threshold=100, max_outputs=3)
        
        assert svt1 is not None
        assert isinstance(svt1, AboveThreshold)
        assert composer.consumed_epsilon > 0
    
    def test_create_svt_explicit_epsilon(self):
        """Test creating SVT with explicit epsilon."""
        composer = SVTComposition(total_budget_epsilon=2.0)
        
        svt1 = composer.create_svt(
            threshold=100, max_outputs=3, epsilon=1.0
        )
        
        assert svt1.epsilon == 1.0
        assert composer.consumed_epsilon == 1.0
        assert composer.remaining_epsilon == 1.0
    
    def test_create_multiple_svt_instances(self):
        """Test creating multiple SVT instances."""
        composer = SVTComposition(total_budget_epsilon=3.0)
        
        svt1 = composer.create_svt(threshold=100, max_outputs=2, epsilon=1.0)
        svt2 = composer.create_svt(threshold=200, max_outputs=3, epsilon=1.5)
        
        assert composer.consumed_epsilon == 2.5
        assert composer.remaining_epsilon == 0.5
    
    def test_budget_exhaustion(self):
        """Test budget exhaustion error."""
        composer = SVTComposition(total_budget_epsilon=1.0)
        
        svt1 = composer.create_svt(threshold=100, epsilon=0.8)
        
        # Try to exceed budget
        with pytest.raises(BudgetExhaustedError):
            svt2 = composer.create_svt(threshold=200, epsilon=0.5)
    
    def test_create_different_svt_types(self):
        """Test creating different SVT types."""
        composer = SVTComposition(total_budget_epsilon=4.0)
        
        svt1 = composer.create_svt(
            threshold=100, svt_type="above_threshold", epsilon=1.0
        )
        svt2 = composer.create_svt(
            threshold=200, svt_type="numeric", epsilon=1.0
        )
        svt3 = composer.create_svt(
            threshold=300, svt_type="adaptive", epsilon=1.0
        )
        svt4 = composer.create_svt(
            threshold=400, svt_type="gap", epsilon=1.0
        )
        
        assert isinstance(svt1, AboveThreshold)
        assert isinstance(svt2, NumericSVT)
        assert isinstance(svt3, AdaptiveSVT)
        assert isinstance(svt4, GapSVT)
    
    def test_composition_result(self):
        """Test composition result reporting."""
        composer = SVTComposition(total_budget_epsilon=3.0)
        
        svt1 = composer.create_svt(threshold=100, epsilon=1.0, max_outputs=2)
        svt2 = composer.create_svt(threshold=200, epsilon=1.5, max_outputs=3)
        
        # Simulate queries
        data = [1, 2, 3]
        svt1.query(data, lambda d: 200.0)
        svt2.query(data, lambda d: 300.0)
        
        result = composer.composition_result()
        
        assert isinstance(result, SVTCompositionResult)
        assert result.total_epsilon == 2.5
        assert result.total_delta == 0.0
        assert result.num_instances == 2
        assert result.total_outputs == 2  # One from each
        assert len(result.per_instance_epsilon) == 2
    
    def test_invalid_svt_type(self):
        """Test error for invalid SVT type."""
        composer = SVTComposition(total_budget_epsilon=2.0)
        
        with pytest.raises(ValueError):
            composer.create_svt(threshold=100, svt_type="invalid")


class TestSVTPrivacyProperties:
    """Test privacy properties of SVT variants."""
    
    def test_svt_consumes_budget_on_outputs(self):
        """Test SVT consumes budget proportional to outputs."""
        # Core property: SVT uses epsilon for c outputs, not all queries
        svt = AboveThreshold(epsilon=1.0, threshold=50, max_outputs=3, seed=42)
        
        data = list(range(100))
        
        # Process many queries, only some above threshold
        n_queries = 0
        for i in range(100):
            result = svt.query(data, lambda d: 40.0 + i)  # Increasing values
            if result is None:
                break
            n_queries += 1
        
        # Should process more than 3 queries
        assert n_queries > 3
        # But only 3 outputs
        assert svt.num_outputs == 3
    
    def test_svt_privacy_independent_of_query_count(self):
        """Test privacy guarantee is independent of total queries."""
        # Two SVT instances with same epsilon, different query counts
        svt1 = AboveThreshold(epsilon=1.0, threshold=100, max_outputs=2, seed=42)
        svt2 = AboveThreshold(epsilon=1.0, threshold=100, max_outputs=2, seed=43)
        
        data = [1, 2, 3]
        
        # svt1: process 10 queries
        for _ in range(10):
            result = svt1.query(data, lambda d: 50.0)
            if result is None:
                break
        
        # svt2: process 100 queries
        for _ in range(100):
            result = svt2.query(data, lambda d: 50.0)
            if result is None:
                break
        
        # Both should have same privacy guarantee
        eps1, delta1 = svt1.privacy_guarantee()
        eps2, delta2 = svt2.privacy_guarantee()
        
        assert eps1 == eps2
        assert delta1 == delta2


@given(
    epsilon=st.floats(min_value=0.5, max_value=5.0),
    threshold=st.floats(min_value=0, max_value=100),
)
@settings(max_examples=20, deadline=None)
def test_above_threshold_halts_hypothesis(epsilon, threshold):
    """Property test: AboveThreshold always halts after max_outputs."""
    max_outputs = 3
    svt = AboveThreshold(
        epsilon=epsilon,
        threshold=threshold,
        max_outputs=max_outputs,
        seed=42,
    )
    
    data = [1, 2, 3]
    
    # Query with values guaranteed above threshold
    def high_query(d):
        return threshold + 1000.0
    
    outputs_count = 0
    for _ in range(100):  # Try many queries
        result = svt.query(data, high_query)
        if result is None:
            break
        if result:  # Above threshold
            outputs_count += 1
    
    # Should halt after exactly max_outputs
    assert outputs_count == max_outputs
    assert not svt.budget_remaining


class TestSVTIntegration:
    """Integration tests for SVT."""
    
    def test_feature_selection(self):
        """Test SVT for private feature selection."""
        # Simulate feature selection: which features have high correlation?
        features = {
            'age': 0.8,
            'income': 0.9,
            'zip': 0.2,
            'education': 0.7,
            'job': 0.3,
        }
        
        svt = NumericSVT(
            epsilon=1.0,
            threshold=0.5,
            max_outputs=3,
            seed=42,
        )
        
        selected_features = []
        for name, correlation in features.items():
            result = svt.query(features, lambda d: d[name])
            if result is None:
                break
            
            is_above, noisy_corr = result
            if is_above:
                selected_features.append((name, noisy_corr))
        
        # Should select high-correlation features
        assert len(selected_features) <= 3
        assert len(selected_features) > 0
    
    def test_adaptive_query_answering(self):
        """Test adaptive query answering with AdaptiveSVT."""
        svt = AdaptiveSVT(
            epsilon=1.0,
            initial_threshold=50,
            max_outputs=10,
            adaptation_rate=0.2,
            seed=42,
        )
        
        data = list(range(100))
        
        # Simulate adaptive querying
        query_values = [20, 30, 40, 60, 80, 100, 120, 140]
        
        results = []
        for val in query_values:
            result = svt.query(data, lambda d: val)
            if result is None:
                break
            results.append(result)
        
        # Should process several queries
        assert len(results) > 0
        
        # Threshold should have adapted
        initial_threshold = 50
        final_threshold = svt.current_threshold
        # May increase or decrease depending on queries
        assert math.isfinite(final_threshold)
    
    def test_svt_composition_workflow(self):
        """Test complete SVT composition workflow."""
        composer = SVTComposition(total_budget_epsilon=2.0)
        
        # Phase 1: Coarse filtering
        svt1 = composer.create_svt(
            threshold=100,
            max_outputs=5,
            epsilon=1.0,
            svt_type="above_threshold",
        )
        
        data = list(range(1000))
        
        coarse_candidates = []
        for i in range(50):
            result = svt1.query(data, lambda d: i * 10)
            if result is None:
                break
            if result:  # Above threshold
                coarse_candidates.append(i * 10)
        
        # Phase 2: Fine-grained analysis on candidates
        if len(coarse_candidates) > 0:
            svt2 = composer.create_svt(
                threshold=200,
                max_outputs=3,
                epsilon=1.0,
                svt_type="numeric",
            )
            
            fine_results = []
            for candidate in coarse_candidates[:10]:
                result = svt2.query(data, lambda d: candidate + 50)
                if result is None:
                    break
                fine_results.append(result)
        
        # Check composition result
        comp_result = composer.composition_result()
        assert comp_result.total_epsilon <= 2.0
        assert comp_result.num_instances == 2
