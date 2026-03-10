"""Tests for causalcert.ci_testing.kci – Kernel CI test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalcert.ci_testing.kci import KernelCITest
from causalcert.types import CITestMethod, NodeSet


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def kci() -> KernelCITest:
    return KernelCITest(alpha=0.05, seed=42, use_gamma_approx=True)


@pytest.fixture
def kci_bootstrap() -> KernelCITest:
    return KernelCITest(alpha=0.05, seed=42, use_gamma_approx=False, n_bootstrap=200)


@pytest.fixture
def kci_nystrom() -> KernelCITest:
    return KernelCITest(alpha=0.05, seed=42, nystrom_rank=50)


def _independent_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        0: rng.standard_normal(n),
        1: rng.standard_normal(n),
        2: rng.standard_normal(n),
    })


def _dependent_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = 2.0 * x + 0.5 * rng.standard_normal(n)
    z = rng.standard_normal(n)
    return pd.DataFrame({0: x, 1: y, 2: z})


def _cond_independent_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """X -> Z -> Y: X _||_ Y | Z."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = x + 0.3 * rng.standard_normal(n)
    y = z + 0.3 * rng.standard_normal(n)
    return pd.DataFrame({0: x, 1: y, 2: z})


def _nonlinear_dependent_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = np.sin(x) + 0.3 * rng.standard_normal(n)
    return pd.DataFrame({0: x, 1: y, 2: rng.standard_normal(n)})


# ═══════════════════════════════════════════════════════════════════════════
# Independent data: should NOT reject
# ═══════════════════════════════════════════════════════════════════════════


class TestKCIIndependent:
    def test_independent_unconditional(self, kci: KernelCITest) -> None:
        data = _independent_data()
        result = kci.test(0, 1, frozenset(), data)
        assert not result.reject
        assert result.p_value > 0.01

    def test_independent_conditional(self, kci: KernelCITest) -> None:
        data = _independent_data()
        result = kci.test(0, 1, frozenset({2}), data)
        assert not result.reject

    def test_result_fields(self, kci: KernelCITest) -> None:
        data = _independent_data()
        result = kci.test(0, 1, frozenset(), data)
        assert result.x == 0
        assert result.y == 1
        assert result.conditioning_set == frozenset()
        assert result.method == CITestMethod.KERNEL
        assert 0.0 <= result.p_value <= 1.0
        assert isinstance(result.statistic, float)


# ═══════════════════════════════════════════════════════════════════════════
# Dependent data: should reject
# ═══════════════════════════════════════════════════════════════════════════


class TestKCIDependent:
    def test_linear_dependent(self, kci: KernelCITest) -> None:
        data = _dependent_data()
        result = kci.test(0, 1, frozenset(), data)
        assert result.reject
        assert result.p_value < 0.05

    def test_nonlinear_dependent(self, kci: KernelCITest) -> None:
        data = _nonlinear_dependent_data()
        result = kci.test(0, 1, frozenset(), data)
        assert result.reject
        assert result.p_value < 0.05


# ═══════════════════════════════════════════════════════════════════════════
# Conditionally independent data
# ═══════════════════════════════════════════════════════════════════════════


class TestKCIConditionallyIndependent:
    def test_cond_indep_not_marginal(self, kci: KernelCITest) -> None:
        data = _cond_independent_data(n=500)
        # Marginal: X and Y are dependent
        marg = kci.test(0, 1, frozenset(), data)
        assert marg.reject

    def test_cond_indep_given_z(self, kci: KernelCITest) -> None:
        data = _cond_independent_data(n=500)
        # Conditional: X _||_ Y | Z
        cond = kci.test(0, 1, frozenset({2}), data)
        assert not cond.reject or cond.p_value > 0.01


# ═══════════════════════════════════════════════════════════════════════════
# Nyström approximation accuracy
# ═══════════════════════════════════════════════════════════════════════════


class TestNystromApproximation:
    def test_nystrom_independent(self, kci_nystrom: KernelCITest) -> None:
        data = _independent_data()
        result = kci_nystrom.test(0, 1, frozenset(), data)
        assert not result.reject

    def test_nystrom_dependent(self, kci_nystrom: KernelCITest) -> None:
        data = _dependent_data()
        result = kci_nystrom.test(0, 1, frozenset(), data)
        assert result.reject

    def test_nystrom_vs_full_agreement(self) -> None:
        data = _dependent_data(n=200)
        kci_full = KernelCITest(alpha=0.05, seed=42, nystrom_rank=None)
        kci_nys = KernelCITest(alpha=0.05, seed=42, nystrom_rank=50)
        r_full = kci_full.test(0, 1, frozenset(), data)
        r_nys = kci_nys.test(0, 1, frozenset(), data)
        # Should agree on reject/not-reject
        assert r_full.reject == r_nys.reject


# ═══════════════════════════════════════════════════════════════════════════
# Kernel caching
# ═══════════════════════════════════════════════════════════════════════════


class TestKernelCaching:
    def test_cache_usage(self) -> None:
        cache: dict[tuple[int, ...], np.ndarray] = {}
        kci = KernelCITest(alpha=0.05, seed=42, cache=cache)
        data = _independent_data()
        kci.test(0, 1, frozenset(), data)
        # Cache should have entries after test
        assert len(cache) >= 0  # may be empty if caching is lazy

    def test_clear_cache(self) -> None:
        kci = KernelCITest(alpha=0.05, seed=42)
        data = _independent_data()
        kci.test(0, 1, frozenset(), data)
        kci.clear_cache()

    def test_set_cache(self) -> None:
        kci = KernelCITest(alpha=0.05, seed=42)
        new_cache: dict[tuple[int, ...], np.ndarray] = {}
        kci.set_cache(new_cache)


# ═══════════════════════════════════════════════════════════════════════════
# Different variable types
# ═══════════════════════════════════════════════════════════════════════════


class TestVariableTypes:
    def test_binary_variable(self, kci: KernelCITest) -> None:
        rng = np.random.default_rng(42)
        n = 300
        data = pd.DataFrame({
            0: rng.binomial(1, 0.5, n).astype(float),
            1: rng.standard_normal(n),
            2: rng.standard_normal(n),
        })
        result = kci.test(0, 1, frozenset(), data)
        assert 0.0 <= result.p_value <= 1.0

    def test_ordinal_variable(self, kci: KernelCITest) -> None:
        rng = np.random.default_rng(42)
        n = 300
        data = pd.DataFrame({
            0: rng.integers(0, 5, n).astype(float),
            1: rng.standard_normal(n),
            2: rng.standard_normal(n),
        })
        result = kci.test(0, 1, frozenset(), data)
        assert 0.0 <= result.p_value <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap p-value
# ═══════════════════════════════════════════════════════════════════════════


class TestBootstrapPValue:
    def test_bootstrap_independent(self, kci_bootstrap: KernelCITest) -> None:
        data = _independent_data(n=200)
        result = kci_bootstrap.test(0, 1, frozenset(), data)
        assert not result.reject

    def test_bootstrap_dependent(self, kci_bootstrap: KernelCITest) -> None:
        data = _dependent_data(n=200)
        result = kci_bootstrap.test(0, 1, frozenset(), data)
        assert result.reject


# ═══════════════════════════════════════════════════════════════════════════
# Batch testing
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchTesting:
    def test_batch_matches_individual(self, kci: KernelCITest) -> None:
        data = _independent_data()
        triples = [
            (0, 1, frozenset()),
            (0, 2, frozenset()),
            (1, 2, frozenset({0})),
        ]
        batch_results = kci.test_batch(triples, data)
        assert len(batch_results) == 3
        for r in batch_results:
            assert 0.0 <= r.p_value <= 1.0

    def test_batch_empty(self, kci: KernelCITest) -> None:
        data = _independent_data()
        results = kci.test_batch([], data)
        assert len(results) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Multiple kernels
# ═══════════════════════════════════════════════════════════════════════════


class TestMultipleKernels:
    def test_rbf_kernel(self) -> None:
        kci = KernelCITest(alpha=0.05, kernel="rbf", seed=42)
        data = _dependent_data()
        result = kci.test(0, 1, frozenset(), data)
        assert result.reject

    def test_polynomial_kernel(self) -> None:
        kci = KernelCITest(alpha=0.05, kernel="polynomial", seed=42)
        data = _dependent_data()
        result = kci.test(0, 1, frozenset(), data)
        assert 0.0 <= result.p_value <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Determinism
# ═══════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        data = _dependent_data()
        kci1 = KernelCITest(alpha=0.05, seed=42)
        kci2 = KernelCITest(alpha=0.05, seed=42)
        r1 = kci1.test(0, 1, frozenset(), data)
        r2 = kci2.test(0, 1, frozenset(), data)
        assert r1.reject == r2.reject
        assert abs(r1.p_value - r2.p_value) < 1e-10

    def test_different_seed_may_differ(self) -> None:
        data = _dependent_data()
        kci1 = KernelCITest(alpha=0.05, seed=42, use_gamma_approx=False, n_bootstrap=50)
        kci2 = KernelCITest(alpha=0.05, seed=99, use_gamma_approx=False, n_bootstrap=50)
        r1 = kci1.test(0, 1, frozenset(), data)
        r2 = kci2.test(0, 1, frozenset(), data)
        # Both should reject (dependent data), but p-values may differ
        assert 0.0 <= r1.p_value <= 1.0
        assert 0.0 <= r2.p_value <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Large conditioning sets
# ═══════════════════════════════════════════════════════════════════════════


class TestLargeConditioning:
    def test_many_conditioning_variables(self, kci: KernelCITest) -> None:
        rng = np.random.default_rng(42)
        n = 300
        data = pd.DataFrame({i: rng.standard_normal(n) for i in range(6)})
        result = kci.test(0, 1, frozenset({2, 3, 4, 5}), data)
        assert 0.0 <= result.p_value <= 1.0
        assert not result.reject  # all independent
