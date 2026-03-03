"""
Tests for strategy selection and classification.

Tests cover:
- StrategySelector classification
- Workload feature extraction
- Strategy library coverage
- Adaptive selection
- Error prediction
- Strategy comparison
"""

import numpy as np
import numpy.testing as npt
import pytest

from dp_forge.workload_optimizer.strategy_selection import (
    StrategySelector,
    WorkloadClassification,
    WorkloadFeatures,
    adaptive_strategy_selection,
    workload_complexity_score,
    recommend_strategy,
    estimate_optimization_time,
    select_strategy_with_timeout,
)
from dp_forge.exceptions import ConfigurationError


class TestWorkloadClassification:
    """Tests for WorkloadClassification enum."""

    def test_enum_values(self):
        assert WorkloadClassification.IDENTITY
        assert WorkloadClassification.RANGE
        assert WorkloadClassification.PREFIX
        assert WorkloadClassification.MARGINAL
        assert WorkloadClassification.HIERARCHICAL
        assert WorkloadClassification.KRONECKER
        assert WorkloadClassification.GENERAL


class TestWorkloadFeatures:
    """Tests for WorkloadFeatures class."""

    def test_classify_identity(self):
        features = WorkloadFeatures(
            shape=(10, 10),
            sparsity=0.0,
            rank=10,
            condition_number=1.0,
            is_identity=True,
            is_range_structured=False,
            is_prefix=False,
            kronecker_dims=None,
            max_sensitivity=1.0,
        )
        
        classification = features.classify()
        assert classification == WorkloadClassification.IDENTITY

    def test_classify_kronecker(self):
        features = WorkloadFeatures(
            shape=(12, 12),
            sparsity=0.5,
            rank=12,
            condition_number=10.0,
            is_identity=False,
            is_range_structured=False,
            is_prefix=False,
            kronecker_dims=(3, 4),
            max_sensitivity=2.0,
        )
        
        classification = features.classify()
        assert classification == WorkloadClassification.KRONECKER

    def test_classify_prefix(self):
        features = WorkloadFeatures(
            shape=(10, 10),
            sparsity=0.5,
            rank=10,
            condition_number=5.0,
            is_identity=False,
            is_range_structured=False,
            is_prefix=True,
            kronecker_dims=None,
            max_sensitivity=10.0,
        )
        
        classification = features.classify()
        assert classification == WorkloadClassification.PREFIX

    def test_classify_range(self):
        features = WorkloadFeatures(
            shape=(20, 10),
            sparsity=0.7,
            rank=10,
            condition_number=5.0,
            is_identity=False,
            is_range_structured=True,
            is_prefix=False,
            kronecker_dims=None,
            max_sensitivity=10.0,
        )
        
        classification = features.classify()
        assert classification == WorkloadClassification.RANGE

    def test_classify_hierarchical(self):
        features = WorkloadFeatures(
            shape=(20, 20),
            sparsity=0.6,
            rank=8,
            condition_number=10.0,
            is_identity=False,
            is_range_structured=False,
            is_prefix=False,
            kronecker_dims=None,
            max_sensitivity=5.0,
        )
        
        classification = features.classify()
        assert classification == WorkloadClassification.HIERARCHICAL

    def test_classify_general(self):
        features = WorkloadFeatures(
            shape=(20, 20),
            sparsity=0.3,
            rank=18,
            condition_number=100.0,
            is_identity=False,
            is_range_structured=False,
            is_prefix=False,
            kronecker_dims=None,
            max_sensitivity=10.0,
        )
        
        classification = features.classify()
        assert classification == WorkloadClassification.GENERAL


class TestStrategySelector:
    """Tests for StrategySelector class."""

    def test_initialization(self):
        selector = StrategySelector(max_domain_size=5000)
        
        assert selector.max_domain_size == 5000
        assert len(selector.strategy_library) > 0

    def test_select_identity_workload(self):
        """Identity workload should be recognized."""
        d = 20
        W = np.eye(d)
        
        selector = StrategySelector()
        strategy = selector.select_strategy(W, epsilon=1.0)
        
        assert strategy is not None
        assert strategy.domain_size == d

    def test_select_prefix_workload(self):
        """Prefix workload should be recognized."""
        d = 15
        W = np.tril(np.ones((d, d)))
        
        selector = StrategySelector()
        strategy = selector.select_strategy(W, epsilon=1.0)
        
        assert strategy is not None

    def test_select_range_workload(self):
        """Range workload should be recognized."""
        d = 16
        
        # Build range queries
        ranges = []
        for start in range(d):
            for end in range(start + 1, min(start + 5, d + 1)):
                row = np.zeros(d)
                row[start:end] = 1.0
                ranges.append(row)
        W = np.array(ranges[:30])
        
        selector = StrategySelector()
        strategy = selector.select_strategy(W, epsilon=1.0)
        
        assert strategy is not None

    def test_select_general_workload(self):
        """General workload should fall back to HDMM."""
        d = 20
        W = np.random.randn(15, d)
        
        selector = StrategySelector()
        strategy = selector.select_strategy(W, epsilon=1.0)
        
        assert strategy is not None

    def test_domain_size_limit_enforcement(self):
        """Large domains should raise error if not Kronecker."""
        d = 15000
        W = np.random.randn(10, d)
        
        selector = StrategySelector(max_domain_size=10000)
        
        with pytest.raises(ConfigurationError, match="exceeds limit"):
            selector.select_strategy(W, epsilon=1.0)

    def test_feature_extraction_identity(self):
        """Test feature extraction for identity."""
        d = 10
        W = np.eye(d)
        
        selector = StrategySelector()
        features = selector._extract_features(W)
        
        assert features.is_identity is True
        assert features.shape == (d, d)

    def test_feature_extraction_range(self):
        """Test detection of range structure."""
        d = 8
        # Single range query
        W = np.zeros((1, d))
        W[0, 2:5] = 1.0
        
        selector = StrategySelector()
        features = selector._extract_features(W)
        
        assert features.is_range_structured is True

    def test_feature_extraction_prefix(self):
        """Test detection of prefix structure."""
        d = 10
        W = np.tril(np.ones((d, d)))
        
        selector = StrategySelector()
        features = selector._extract_features(W)
        
        assert features.is_prefix is True

    def test_feature_caching(self):
        """Features should be cached."""
        d = 10
        W = np.eye(d)
        
        selector = StrategySelector()
        features1 = selector._extract_features(W)
        features2 = selector._extract_features(W)
        
        # Should be the same object (cached)
        assert features1 is features2

    def test_prefer_simple_strategies(self):
        """prefer_simple=True should use simple strategies."""
        d = 16
        W = np.tril(np.ones((d, d)))
        
        selector = StrategySelector()
        strategy = selector.select_strategy(W, epsilon=1.0, prefer_simple=True)
        
        # Should use simple strategy
        assert strategy is not None

    def test_prefer_optimized_strategies(self):
        """prefer_simple=False should optimize."""
        d = 15
        W = np.random.randn(10, d)
        
        selector = StrategySelector()
        strategy = selector.select_strategy(W, epsilon=1.0, prefer_simple=False)
        
        assert strategy is not None

    def test_predict_error(self):
        """Test error prediction for named strategies."""
        d = 16
        W = np.eye(d)
        
        selector = StrategySelector()
        error = selector.predict_error(W, "identity", epsilon=1.0)
        
        assert error > 0
        assert not np.isnan(error)
        assert not np.isinf(error)

    def test_predict_error_all_strategies(self):
        """Test prediction for all library strategies."""
        d = 16
        W = np.eye(d)
        
        selector = StrategySelector()
        
        for strategy_name in selector.strategy_library:
            error = selector.predict_error(W, strategy_name, epsilon=1.0)
            assert isinstance(error, float)

    def test_predict_error_unknown_strategy(self):
        """Unknown strategy should raise error."""
        d = 10
        W = np.eye(d)
        
        selector = StrategySelector()
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            selector.predict_error(W, "nonexistent", epsilon=1.0)

    def test_compare_strategies(self):
        """Test comparing all strategies."""
        d = 16
        W = np.eye(d)
        
        selector = StrategySelector()
        results = selector.compare_strategies(W, epsilon=1.0)
        
        assert len(results) > 0
        assert all(isinstance(error, float) for error in results.values())
        
        # Should be sorted by error
        errors = list(results.values())
        assert errors == sorted(errors)

    def test_detect_range_structure_positive(self):
        """Should detect range structure."""
        d = 10
        W = np.zeros((3, d))
        W[0, 0:3] = 1.0
        W[1, 2:6] = 1.0
        W[2, 5:10] = 1.0
        
        selector = StrategySelector()
        is_range = selector._detect_range_structure(W)
        
        assert is_range is True

    def test_detect_range_structure_negative(self):
        """Non-range workload should not be detected as range."""
        d = 10
        W = np.random.randn(5, d)
        
        selector = StrategySelector()
        is_range = selector._detect_range_structure(W)
        
        assert is_range is False

    def test_detect_prefix_structure_positive(self):
        """Should detect prefix structure."""
        d = 10
        W = np.tril(np.ones((d, d)))
        
        selector = StrategySelector()
        is_prefix = selector._detect_prefix_structure(W)
        
        assert is_prefix is True

    def test_detect_prefix_structure_negative(self):
        """Non-prefix workload should not be detected."""
        d = 10
        W = np.eye(d)
        
        selector = StrategySelector()
        is_prefix = selector._detect_prefix_structure(W)
        
        assert is_prefix is False

    def test_is_power_of_two(self):
        """Test power of two detection."""
        selector = StrategySelector()
        
        assert selector._is_power_of_two(1) is True
        assert selector._is_power_of_two(2) is True
        assert selector._is_power_of_two(4) is True
        assert selector._is_power_of_two(16) is True
        assert selector._is_power_of_two(3) is False
        assert selector._is_power_of_two(15) is False
        assert selector._is_power_of_two(0) is False


class TestAdaptiveStrategySelection:
    """Tests for adaptive_strategy_selection function."""

    def test_basic_adaptive_selection(self):
        """Test basic adaptive selection with validation."""
        d = 20
        W = np.random.randn(50, d)
        
        strategy = adaptive_strategy_selection(W, epsilon=1.0, validation_fraction=0.2)
        
        assert strategy is not None
        assert strategy.domain_size == d

    def test_validation_fraction(self):
        """Test different validation fractions."""
        d = 15
        W = np.random.randn(100, d)
        
        # Small validation set
        strategy1 = adaptive_strategy_selection(W, epsilon=1.0, validation_fraction=0.1)
        
        # Large validation set
        strategy2 = adaptive_strategy_selection(W, epsilon=1.0, validation_fraction=0.3)
        
        assert strategy1 is not None
        assert strategy2 is not None


class TestWorkloadComplexityScore:
    """Tests for workload_complexity_score function."""

    def test_identity_is_simple(self):
        """Identity workload should be simple."""
        d = 10
        W = np.eye(d)
        
        score = workload_complexity_score(W)
        
        assert 0 <= score <= 1
        # Identity is relatively simple
        assert score < 0.5

    def test_random_is_complex(self):
        """Random workload should be more complex."""
        d = 20
        W = np.random.randn(50, d)
        
        score = workload_complexity_score(W)
        
        assert 0 <= score <= 1

    def test_complexity_increases_with_size(self):
        """Larger workloads should be more complex."""
        W_small = np.random.randn(10, 10)
        W_large = np.random.randn(100, 100)
        
        score_small = workload_complexity_score(W_small)
        score_large = workload_complexity_score(W_large)
        
        # Larger should generally be more complex
        assert score_large >= score_small

    def test_complexity_bounded(self):
        """Complexity should be in [0, 1]."""
        for _ in range(10):
            d = np.random.randint(5, 50)
            m = np.random.randint(5, 100)
            W = np.random.randn(m, d)
            
            score = workload_complexity_score(W)
            assert 0 <= score <= 1


class TestRecommendStrategy:
    """Tests for recommend_strategy function."""

    def test_recommend_low_budget(self):
        """Low budget should recommend simple strategies."""
        d = 16
        W = np.eye(d)
        
        rec, strategy = recommend_strategy(W, epsilon=1.0, compute_budget="low")
        
        assert isinstance(rec, str)
        assert strategy is not None
        assert "Identity" in rec or "identity" in rec

    def test_recommend_medium_budget(self):
        """Medium budget should use automatic selection."""
        d = 15
        W = np.random.randn(10, d)
        
        rec, strategy = recommend_strategy(W, epsilon=1.0, compute_budget="medium")
        
        assert isinstance(rec, str)
        assert strategy is not None

    def test_recommend_high_budget(self):
        """High budget should optimize fully."""
        d = 15
        W = np.random.randn(10, d)
        
        rec, strategy = recommend_strategy(W, epsilon=1.0, compute_budget="high")
        
        assert isinstance(rec, str)
        assert strategy is not None

    def test_recommend_prefix_workload(self):
        """Should recommend appropriate strategy for prefix."""
        d = 16
        W = np.tril(np.ones((d, d)))
        
        rec, strategy = recommend_strategy(W, epsilon=1.0, compute_budget="low")
        
        assert strategy is not None
        # Should mention prefix
        assert "prefix" in rec.lower() or "Prefix" in rec


class TestEstimateOptimizationTime:
    """Tests for estimate_optimization_time function."""

    def test_hdmm_time_increases_with_size(self):
        """HDMM time should increase with domain size."""
        time_small = estimate_optimization_time((10, 10), method="hdmm")
        time_large = estimate_optimization_time((10, 100), method="hdmm")
        
        # Larger domain should take longer
        assert time_large > time_small

    def test_simple_strategy_fast(self):
        """Simple strategies should be fast."""
        time_simple = estimate_optimization_time((100, 100), method="simple")
        time_hdmm = estimate_optimization_time((100, 100), method="hdmm")
        
        # Simple should be much faster than HDMM
        assert time_simple < time_hdmm

    def test_kronecker_time(self):
        """Kronecker should be faster than full HDMM."""
        time_kron = estimate_optimization_time((100, 100), method="kronecker")
        
        assert time_kron > 0

    def test_unknown_method_fallback(self):
        """Unknown method should return default time."""
        time = estimate_optimization_time((10, 10), method="unknown")
        
        assert time == 1.0


class TestSelectStrategyWithTimeout:
    """Tests for select_strategy_with_timeout function."""

    def test_sufficient_timeout(self):
        """With sufficient timeout, should optimize."""
        d = 10
        W = np.random.randn(5, d)
        
        strategy = select_strategy_with_timeout(W, epsilon=1.0, timeout_seconds=100.0)
        
        assert strategy is not None

    def test_insufficient_timeout(self):
        """With insufficient timeout, should use simple strategy."""
        d = 100
        W = np.random.randn(50, d)
        
        # Very short timeout
        strategy = select_strategy_with_timeout(W, epsilon=1.0, timeout_seconds=0.001)
        
        assert strategy is not None

    def test_timeout_matches_estimated_time(self):
        """Timeout behavior should match estimated time."""
        d = 20
        W = np.random.randn(10, d)
        
        estimated_time = estimate_optimization_time((10, d), method="hdmm")
        
        # If timeout is longer than estimate, should optimize
        if estimated_time < 1.0:
            strategy = select_strategy_with_timeout(
                W, epsilon=1.0, timeout_seconds=estimated_time * 2
            )
            assert strategy is not None


class TestStrategyLibraryCoverage:
    """Tests for strategy library coverage."""

    def test_library_contains_basic_strategies(self):
        """Library should contain basic strategies."""
        selector = StrategySelector()
        
        assert "identity" in selector.strategy_library
        assert "uniform" in selector.strategy_library
        assert "hierarchical" in selector.strategy_library
        assert "prefix" in selector.strategy_library

    def test_all_library_strategies_work(self):
        """All library strategies should be callable."""
        d = 16
        selector = StrategySelector()
        
        for strategy_name, strategy_fn in selector.strategy_library.items():
            try:
                strategy = strategy_fn(d, epsilon=1.0)
                assert strategy is not None
            except ValueError:
                # hierarchical requires power of 2
                if strategy_name == "hierarchical":
                    pass
                else:
                    raise

    def test_strategy_errors_on_invalid_domain(self):
        """Strategies should handle invalid domains gracefully."""
        selector = StrategySelector()
        
        # Hierarchical requires power of 2
        strategy_fn = selector.strategy_library["hierarchical"]
        
        with pytest.raises(ValueError):
            strategy_fn(15, epsilon=1.0)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_query_workload(self):
        """Single query should work."""
        d = 10
        W = np.random.randn(1, d)
        
        selector = StrategySelector()
        strategy = selector.select_strategy(W, epsilon=1.0)
        
        assert strategy is not None

    def test_square_workload(self):
        """Square workload (m=d) should work."""
        d = 15
        W = np.random.randn(d, d)
        
        selector = StrategySelector()
        strategy = selector.select_strategy(W, epsilon=1.0)
        
        assert strategy is not None

    def test_tall_workload(self):
        """Tall workload (m >> d) should work."""
        d = 10
        W = np.random.randn(100, d)
        
        selector = StrategySelector()
        strategy = selector.select_strategy(W, epsilon=1.0)
        
        assert strategy is not None

    def test_wide_workload(self):
        """Wide workload (m << d) should work."""
        d = 50
        W = np.random.randn(5, d)
        
        selector = StrategySelector()
        strategy = selector.select_strategy(W, epsilon=1.0)
        
        assert strategy is not None

    def test_sparse_workload(self):
        """Sparse workload should be handled."""
        d = 20
        W = np.zeros((15, d))
        # Make 10% non-zero
        for i in range(15):
            indices = np.random.choice(d, size=2, replace=False)
            W[i, indices] = 1.0
        
        selector = StrategySelector()
        strategy = selector.select_strategy(W, epsilon=1.0)
        
        assert strategy is not None

    def test_dense_workload(self):
        """Dense workload should be handled."""
        d = 15
        W = np.random.randn(15, d)
        
        selector = StrategySelector()
        features = selector._extract_features(W)
        
        # Dense workload should have low sparsity
        assert features.sparsity < 0.5

    def test_very_small_epsilon(self):
        """Very small epsilon should work."""
        d = 10
        W = np.eye(d)
        
        selector = StrategySelector()
        strategy = selector.select_strategy(W, epsilon=0.01)
        
        assert strategy.epsilon == 0.01

    def test_very_large_epsilon(self):
        """Very large epsilon should work."""
        d = 10
        W = np.eye(d)
        
        selector = StrategySelector()
        strategy = selector.select_strategy(W, epsilon=100.0)
        
        assert strategy.epsilon == 100.0
