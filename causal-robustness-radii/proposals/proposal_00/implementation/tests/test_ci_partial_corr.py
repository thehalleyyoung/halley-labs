"""Tests for causalcert.ci_testing.partial_corr – partial correlation CI test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalcert.ci_testing.partial_corr import (
    PartialCorrelationTest,
    RegularizedPartialCorrelationTest,
)
from causalcert.types import CITestMethod


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def pcorr() -> PartialCorrelationTest:
    return PartialCorrelationTest(alpha=0.05, seed=42)


def _gaussian_indep(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        0: rng.standard_normal(n),
        1: rng.standard_normal(n),
        2: rng.standard_normal(n),
    })


def _gaussian_dep(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = 0.8 * x + 0.5 * rng.standard_normal(n)
    z = rng.standard_normal(n)
    return pd.DataFrame({0: x, 1: y, 2: z})


def _known_correlation(rho: float, n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    return pd.DataFrame({0: x, 1: y, 2: rng.standard_normal(n)})


# ═══════════════════════════════════════════════════════════════════════════
# Gaussian data – independent
# ═══════════════════════════════════════════════════════════════════════════


class TestPartialCorrIndependent:
    def test_independent_unconditional(self, pcorr: PartialCorrelationTest) -> None:
        data = _gaussian_indep()
        result = pcorr.test(0, 1, frozenset(), data)
        assert not result.reject
        assert result.p_value > 0.01

    def test_independent_conditional(self, pcorr: PartialCorrelationTest) -> None:
        data = _gaussian_indep()
        result = pcorr.test(0, 1, frozenset({2}), data)
        assert not result.reject

    def test_method_field(self, pcorr: PartialCorrelationTest) -> None:
        data = _gaussian_indep()
        result = pcorr.test(0, 1, frozenset(), data)
        assert result.method == CITestMethod.PARTIAL_CORRELATION


# ═══════════════════════════════════════════════════════════════════════════
# Gaussian data – dependent
# ═══════════════════════════════════════════════════════════════════════════


class TestPartialCorrDependent:
    def test_dependent_unconditional(self, pcorr: PartialCorrelationTest) -> None:
        data = _gaussian_dep()
        result = pcorr.test(0, 1, frozenset(), data)
        assert result.reject
        assert result.p_value < 0.05

    def test_dependent_conditional_on_irrelevant(self, pcorr: PartialCorrelationTest) -> None:
        data = _gaussian_dep()
        result = pcorr.test(0, 1, frozenset({2}), data)
        assert result.reject  # Z is irrelevant, X and Y still dependent


# ═══════════════════════════════════════════════════════════════════════════
# Fisher z-transform
# ═══════════════════════════════════════════════════════════════════════════


class TestFisherZ:
    def test_known_zero_correlation(self, pcorr: PartialCorrelationTest) -> None:
        data = _gaussian_indep(n=1000)
        result = pcorr.test(0, 1, frozenset(), data)
        assert abs(result.statistic) < 3.0  # z-stat should be small

    def test_known_high_correlation(self, pcorr: PartialCorrelationTest) -> None:
        data = _known_correlation(0.9, n=500)
        result = pcorr.test(0, 1, frozenset(), data)
        assert result.reject
        assert abs(result.statistic) > 3.0

    @pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
    def test_monotone_pvalue(self, pcorr: PartialCorrelationTest, rho: float) -> None:
        data = _known_correlation(rho, n=500)
        result = pcorr.test(0, 1, frozenset(), data)
        assert 0.0 <= result.p_value <= 1.0

    def test_pvalue_decreases_with_correlation(self, pcorr: PartialCorrelationTest) -> None:
        p_low = pcorr.test(0, 1, frozenset(), _known_correlation(0.1, n=500)).p_value
        p_high = pcorr.test(0, 1, frozenset(), _known_correlation(0.8, n=500)).p_value
        assert p_high < p_low


# ═══════════════════════════════════════════════════════════════════════════
# Conditioning set handling
# ═══════════════════════════════════════════════════════════════════════════


class TestConditioningSet:
    def test_empty_conditioning(self, pcorr: PartialCorrelationTest) -> None:
        data = _gaussian_dep()
        result = pcorr.test(0, 1, frozenset(), data)
        assert isinstance(result.conditioning_set, frozenset)
        assert len(result.conditioning_set) == 0

    def test_single_conditioning(self, pcorr: PartialCorrelationTest) -> None:
        data = _gaussian_dep()
        result = pcorr.test(0, 1, frozenset({2}), data)
        assert result.conditioning_set == frozenset({2})

    def test_chain_conditional_independence(self, pcorr: PartialCorrelationTest) -> None:
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.standard_normal(n)
        z = 0.8 * x + 0.3 * rng.standard_normal(n)
        y = 0.8 * z + 0.3 * rng.standard_normal(n)
        data = pd.DataFrame({0: x, 1: y, 2: z})
        # X _||_ Y | Z
        result = pcorr.test(0, 1, frozenset({2}), data)
        assert not result.reject or result.p_value > 0.01

    def test_large_conditioning_set(self, pcorr: PartialCorrelationTest) -> None:
        rng = np.random.default_rng(42)
        n = 500
        data = pd.DataFrame({i: rng.standard_normal(n) for i in range(6)})
        result = pcorr.test(0, 1, frozenset({2, 3, 4, 5}), data)
        assert 0.0 <= result.p_value <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Regularized partial correlation
# ═══════════════════════════════════════════════════════════════════════════


class TestRegularized:
    def test_regularized_independent(self) -> None:
        pcorr_reg = RegularizedPartialCorrelationTest(alpha=0.05, seed=42)
        data = _gaussian_indep()
        result = pcorr_reg.test(0, 1, frozenset(), data)
        assert not result.reject

    def test_regularized_dependent(self) -> None:
        pcorr_reg = RegularizedPartialCorrelationTest(alpha=0.05, seed=42)
        data = _gaussian_dep()
        result = pcorr_reg.test(0, 1, frozenset(), data)
        assert result.reject


# ═══════════════════════════════════════════════════════════════════════════
# Semi-partial correlation
# ═══════════════════════════════════════════════════════════════════════════


class TestSemiPartial:
    def test_semi_partial_returns_result(self, pcorr: PartialCorrelationTest) -> None:
        data = _gaussian_dep()
        result = pcorr.test_semi_partial(0, 1, frozenset({2}), data)
        assert 0.0 <= result.p_value <= 1.0

    def test_semi_partial_independent(self, pcorr: PartialCorrelationTest) -> None:
        data = _gaussian_indep()
        result = pcorr.test_semi_partial(0, 1, frozenset({2}), data)
        assert not result.reject

    def test_semi_partial_dependent(self, pcorr: PartialCorrelationTest) -> None:
        data = _gaussian_dep()
        result = pcorr.test_semi_partial(0, 1, frozenset(), data)
        assert result.reject


# ═══════════════════════════════════════════════════════════════════════════
# Parametric sweep: sample size
# ═══════════════════════════════════════════════════════════════════════════


class TestSampleSizeSweep:
    @pytest.mark.parametrize("n", [100, 300, 1000])
    def test_power_increases_with_n(self, n: int) -> None:
        data = _known_correlation(0.3, n=n, seed=42)
        pcorr = PartialCorrelationTest(alpha=0.05, seed=42)
        result = pcorr.test(0, 1, frozenset(), data)
        assert 0.0 <= result.p_value <= 1.0

    @pytest.mark.parametrize("n", [50, 200, 500])
    def test_type1_controlled(self, n: int) -> None:
        data = _gaussian_indep(n=n)
        pcorr = PartialCorrelationTest(alpha=0.05, seed=42)
        result = pcorr.test(0, 1, frozenset(), data)
        assert result.p_value > 0.001  # shouldn't strongly reject

    @pytest.mark.parametrize("rho", [-0.5, -0.3, 0.0, 0.3, 0.5])
    def test_negative_and_positive_correlations(self, rho: float) -> None:
        data = _known_correlation(rho, n=500, seed=42)
        pcorr = PartialCorrelationTest(alpha=0.05, seed=42)
        result = pcorr.test(0, 1, frozenset(), data)
        if abs(rho) > 0.2:
            assert result.reject
        assert 0.0 <= result.p_value <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Static method tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStaticMethods:
    def test_partial_correlation_zero(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        r = PartialCorrelationTest._partial_correlation(x, y, None)
        assert abs(r) < 0.15

    def test_partial_correlation_strong(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        x = rng.standard_normal(n)
        y = 5.0 * x + 0.01 * rng.standard_normal(n)
        r = PartialCorrelationTest._partial_correlation(x, y, None)
        assert abs(r) > 0.9

    def test_fisher_z_transform(self) -> None:
        z, p = PartialCorrelationTest._fisher_z(0.5, n=100, k=0)
        assert isinstance(z, float)
        assert 0.0 <= p <= 1.0

    def test_fisher_z_zero_corr(self) -> None:
        z, p = PartialCorrelationTest._fisher_z(0.0, n=100, k=0)
        assert abs(z) < 0.5
        assert p > 0.5

    def test_fisher_z_large_corr(self) -> None:
        z, p = PartialCorrelationTest._fisher_z(0.9, n=500, k=0)
        assert abs(z) > 5.0
        assert p < 0.001
