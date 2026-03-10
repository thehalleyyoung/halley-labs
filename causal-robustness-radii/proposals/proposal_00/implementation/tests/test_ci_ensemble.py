"""Tests for causalcert.ci_testing.ensemble – Cauchy combination test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalcert.ci_testing.ensemble import (
    CauchyCombinationTest,
    cauchy_combine_pvalues,
    cauchy_combine_unweighted,
)
from causalcert.ci_testing.partial_corr import PartialCorrelationTest
from causalcert.ci_testing.kci import KernelCITest
from causalcert.types import CITestMethod


# ═══════════════════════════════════════════════════════════════════════════
# Cauchy combination validity
# ═══════════════════════════════════════════════════════════════════════════


class TestCauchyCombination:
    """Test the Cauchy combination p-value formula."""

    def test_single_pvalue(self) -> None:
        p = cauchy_combine_pvalues([0.5], [1.0])
        assert abs(p - 0.5) < 0.1

    def test_two_uniform_pvalues(self) -> None:
        p = cauchy_combine_pvalues([0.3, 0.7], [0.5, 0.5])
        assert 0.0 <= p <= 1.0

    def test_all_significant(self) -> None:
        p = cauchy_combine_pvalues([0.001, 0.002, 0.003], [1 / 3, 1 / 3, 1 / 3])
        assert p < 0.01

    def test_all_non_significant(self) -> None:
        p = cauchy_combine_pvalues([0.5, 0.6, 0.7], [1 / 3, 1 / 3, 1 / 3])
        assert p > 0.1

    def test_mixed_pvalues(self) -> None:
        p = cauchy_combine_pvalues([0.001, 0.9], [0.5, 0.5])
        assert 0.0 <= p <= 1.0

    def test_unweighted_equals_equal_weights(self) -> None:
        pvals = [0.1, 0.2, 0.3]
        p_uw = cauchy_combine_unweighted(pvals)
        p_w = cauchy_combine_pvalues(pvals, [1 / 3, 1 / 3, 1 / 3])
        assert abs(p_uw - p_w) < 1e-6

    def test_weight_normalization(self) -> None:
        pvals = [0.1, 0.2]
        p1 = cauchy_combine_pvalues(pvals, [1.0, 1.0])
        p2 = cauchy_combine_pvalues(pvals, [0.5, 0.5])
        assert abs(p1 - p2) < 1e-6

    def test_boundary_pvalues(self) -> None:
        # Very small p-value
        p = cauchy_combine_pvalues([1e-10, 0.5], [0.5, 0.5])
        assert 0.0 <= p <= 1.0

    def test_preserves_order(self) -> None:
        p_small = cauchy_combine_pvalues([0.01, 0.02], [0.5, 0.5])
        p_large = cauchy_combine_pvalues([0.5, 0.6], [0.5, 0.5])
        assert p_small < p_large


# ═══════════════════════════════════════════════════════════════════════════
# Ensemble on independent data
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def independent_gaussian() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 400
    return pd.DataFrame({
        0: rng.standard_normal(n),
        1: rng.standard_normal(n),
        2: rng.standard_normal(n),
    })


@pytest.fixture
def dependent_gaussian() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 400
    x = rng.standard_normal(n)
    y = 2.0 * x + 0.3 * rng.standard_normal(n)
    return pd.DataFrame({
        0: x,
        1: y,
        2: rng.standard_normal(n),
    })


@pytest.fixture
def ensemble() -> CauchyCombinationTest:
    base = [
        PartialCorrelationTest(alpha=0.05, seed=42),
        KernelCITest(alpha=0.05, seed=42, nystrom_rank=50),
    ]
    return CauchyCombinationTest(base_tests=base, alpha=0.05, seed=42)


class TestEnsembleIndependent:
    def test_not_reject_independent(
        self, ensemble: CauchyCombinationTest, independent_gaussian: pd.DataFrame
    ) -> None:
        result = ensemble.test(0, 1, frozenset(), independent_gaussian)
        assert not result.reject

    def test_pvalue_range(
        self, ensemble: CauchyCombinationTest, independent_gaussian: pd.DataFrame
    ) -> None:
        result = ensemble.test(0, 1, frozenset(), independent_gaussian)
        assert 0.0 <= result.p_value <= 1.0

    def test_method_is_ensemble(
        self, ensemble: CauchyCombinationTest, independent_gaussian: pd.DataFrame
    ) -> None:
        result = ensemble.test(0, 1, frozenset(), independent_gaussian)
        assert result.method == CITestMethod.ENSEMBLE


# ═══════════════════════════════════════════════════════════════════════════
# Ensemble on dependent data
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsembleDependent:
    def test_reject_dependent(
        self, ensemble: CauchyCombinationTest, dependent_gaussian: pd.DataFrame
    ) -> None:
        result = ensemble.test(0, 1, frozenset(), dependent_gaussian)
        assert result.reject
        assert result.p_value < 0.05


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive weights
# ═══════════════════════════════════════════════════════════════════════════


class TestAdaptiveWeights:
    def test_weights_property(self, ensemble: CauchyCombinationTest) -> None:
        w = ensemble.weights
        assert len(w) == 2
        assert all(ww > 0 for ww in w)

    def test_cache_size(self, ensemble: CauchyCombinationTest) -> None:
        assert ensemble.cache_size == 0

    def test_clear_cache(
        self, ensemble: CauchyCombinationTest, independent_gaussian: pd.DataFrame
    ) -> None:
        ensemble.test(0, 1, frozenset(), independent_gaussian)
        ensemble.clear_cache()
        assert ensemble.cache_size == 0


# ═══════════════════════════════════════════════════════════════════════════
# Batch testing
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsembleBatch:
    def test_batch(
        self, ensemble: CauchyCombinationTest, independent_gaussian: pd.DataFrame
    ) -> None:
        triples = [
            (0, 1, frozenset()),
            (0, 2, frozenset()),
            (1, 2, frozenset({0})),
        ]
        results = ensemble.test_batch(triples, independent_gaussian)
        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.p_value <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Cauchy combine static method
# ═══════════════════════════════════════════════════════════════════════════


class TestStaticCombine:
    def test_static_combine(self) -> None:
        p = CauchyCombinationTest.cauchy_combine([0.1, 0.2], [0.5, 0.5])
        assert 0.0 <= p <= 1.0

    def test_static_matches_module_fn(self) -> None:
        pvals = [0.05, 0.3]
        weights = [0.5, 0.5]
        p1 = CauchyCombinationTest.cauchy_combine(pvals, weights)
        p2 = cauchy_combine_pvalues(pvals, weights)
        assert abs(p1 - p2) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases and stress tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsembleEdgeCases:
    def test_single_base_test(self) -> None:
        base = [PartialCorrelationTest(alpha=0.05, seed=42)]
        ens = CauchyCombinationTest(base_tests=base, alpha=0.05, seed=42)
        data = _independent_data()
        result = ens.test(0, 1, frozenset(), data)
        assert 0.0 <= result.p_value <= 1.0

    def test_conditional_test(self) -> None:
        base = [
            PartialCorrelationTest(alpha=0.05, seed=42),
            KernelCITest(alpha=0.05, seed=42, nystrom_rank=50),
        ]
        ens = CauchyCombinationTest(base_tests=base, alpha=0.05, seed=42)
        data = _independent_data()
        result = ens.test(0, 1, frozenset({2}), data)
        assert 0.0 <= result.p_value <= 1.0

    @pytest.mark.parametrize("n", [100, 300, 500])
    def test_different_sample_sizes(self, n: int) -> None:
        rng = np.random.default_rng(42)
        data = pd.DataFrame({
            0: rng.standard_normal(n),
            1: rng.standard_normal(n),
            2: rng.standard_normal(n),
        })
        base = [PartialCorrelationTest(alpha=0.05, seed=42)]
        ens = CauchyCombinationTest(base_tests=base, alpha=0.05, seed=42)
        result = ens.test(0, 1, frozenset(), data)
        assert not result.reject

    def test_many_pvalues(self) -> None:
        pvals = [0.5] * 100
        weights = [1 / 100] * 100
        p = cauchy_combine_pvalues(pvals, weights)
        assert 0.3 <= p <= 0.7  # close to 0.5

    def test_extreme_pvalues(self) -> None:
        p = cauchy_combine_pvalues([1e-20, 1.0 - 1e-10], [0.5, 0.5])
        assert 0.0 <= p <= 1.0


class TestEnsembleWeightStrategies:
    """Test that different weight strategies produce valid p-values."""

    def test_uniform_weights(self) -> None:
        data = _independent_data()
        base_tests = [PartialCorrelationTest(alpha=0.05), KernelCITest(alpha=0.05)]
        ensemble = CauchyCombinationTest(base_tests=base_tests, adaptive=False)
        result = ensemble.test(0, 1, frozenset(), data)
        assert 0.0 <= result.p_value <= 1.0

    def test_adaptive_weights(self) -> None:
        data = _independent_data()
        base_tests = [PartialCorrelationTest(alpha=0.05), KernelCITest(alpha=0.05)]
        ensemble = CauchyCombinationTest(base_tests=base_tests, adaptive=True)
        result = ensemble.test(0, 1, frozenset(), data)
        assert 0.0 <= result.p_value <= 1.0

    @pytest.mark.parametrize("n_tests", [1, 2])
    def test_varying_ensemble_size(self, n_tests: int) -> None:
        all_tests = [PartialCorrelationTest(alpha=0.05), KernelCITest(alpha=0.05)]
        base_tests = all_tests[:n_tests]
        data = _independent_data()
        ensemble = CauchyCombinationTest(base_tests=base_tests)
        result = ensemble.test(0, 1, frozenset(), data)
        assert 0.0 <= result.p_value <= 1.0


def _independent_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 400
    return pd.DataFrame({
        0: rng.standard_normal(n),
        1: rng.standard_normal(n),
        2: rng.standard_normal(n),
    })
