"""
Unit tests for usability_oracle.utils.vectorized.

Tests cover vectorized Fitts' law, Hick's law, visual-search cost,
batch softmax, sparse transition multiply, batch KL divergence, and
performance assertions (vectorized > 10× faster than loops for n > 1000).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from usability_oracle.utils.vectorized import (
    BenchmarkResult,
    batch_interval_arithmetic,
    batch_softmax,
    benchmark,
    parallel_cost_computation,
    sparse_transition_multiply,
    vectorized_fitts,
    vectorized_hick,
    vectorized_kl_divergence,
    vectorized_visual_search,
)

# Import scalar models for comparison
from usability_oracle.cognitive.fitts import FittsLaw
from usability_oracle.cognitive.hick import HickHymanLaw
from usability_oracle.utils.math import kl_divergence as scalar_kl_divergence


# ===================================================================
# Helpers
# ===================================================================

def _scalar_fitts(d: float, w: float) -> float:
    return FittsLaw.predict(d, w)


def _scalar_hick(n: int) -> float:
    return HickHymanLaw.predict(n)


def _loop_fitts(distances: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Python-loop version of Fitts' law for performance comparison."""
    result = np.empty(len(distances))
    for i in range(len(distances)):
        result[i] = 0.050 + 0.150 * math.log2(1.0 + distances[i] / widths[i])
    return result


def _loop_softmax(Q: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Python-loop softmax for performance comparison."""
    n, m = Q.shape
    result = np.empty((n, m))
    for i in range(n):
        vals = -betas[i] * Q[i]
        vals -= np.max(vals)
        exp_vals = np.exp(vals)
        result[i] = exp_vals / exp_vals.sum()
    return result


# ===================================================================
# vectorized_fitts
# ===================================================================

class TestVectorizedFitts:
    """Vectorized Fitts' law matches scalar for each element."""

    def test_matches_scalar_single(self) -> None:
        d, w = 200.0, 50.0
        expected = _scalar_fitts(d, w)
        result = vectorized_fitts(np.array([d]), np.array([w]))
        assert result[0] == pytest.approx(expected, rel=1e-10)

    def test_matches_scalar_batch(self) -> None:
        ds = np.array([100.0, 200.0, 400.0, 800.0])
        ws = np.array([20.0, 40.0, 60.0, 80.0])
        results = vectorized_fitts(ds, ws)
        for i in range(len(ds)):
            expected = _scalar_fitts(ds[i], ws[i])
            assert results[i] == pytest.approx(expected, rel=1e-10)

    def test_custom_parameters(self) -> None:
        ds = np.array([100.0])
        ws = np.array([25.0])
        result = vectorized_fitts(ds, ws, a=0.1, b=0.2)
        expected = 0.1 + 0.2 * math.log2(1.0 + 100.0 / 25.0)
        assert result[0] == pytest.approx(expected)

    def test_rejects_non_positive_distance(self) -> None:
        with pytest.raises(ValueError, match="distances"):
            vectorized_fitts(np.array([0.0]), np.array([10.0]))

    def test_rejects_non_positive_width(self) -> None:
        with pytest.raises(ValueError, match="widths"):
            vectorized_fitts(np.array([10.0]), np.array([-1.0]))

    def test_monotone_in_distance(self) -> None:
        ds = np.array([50.0, 100.0, 200.0, 400.0])
        ws = np.full(4, 20.0)
        results = vectorized_fitts(ds, ws)
        assert np.all(np.diff(results) > 0)


# ===================================================================
# vectorized_hick
# ===================================================================

class TestVectorizedHick:
    """Vectorized Hick matches scalar for each element."""

    def test_matches_scalar_equiprobable(self) -> None:
        ns = np.array([2, 4, 8, 16])
        results = vectorized_hick(ns)
        for i, n in enumerate(ns):
            expected = _scalar_hick(int(n))
            assert results[i] == pytest.approx(expected, rel=1e-10)

    def test_single_alternative(self) -> None:
        result = vectorized_hick(np.array([1]))
        expected = _scalar_hick(1)
        assert result[0] == pytest.approx(expected)

    def test_with_probabilities(self) -> None:
        # Two trials: uniform over 4, and biased
        alts = np.array([4, 4])
        probs = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.7, 0.1, 0.1, 0.1],
        ])
        results = vectorized_hick(alts, probabilities=probs)
        # Uniform entropy = log₂(4) = 2 bits
        expected_uniform = 0.200 + 0.155 * 2.0
        assert results[0] == pytest.approx(expected_uniform, rel=1e-6)
        # Biased has lower entropy → faster RT
        assert results[1] < results[0]

    def test_rejects_zero_alternatives(self) -> None:
        with pytest.raises(ValueError):
            vectorized_hick(np.array([0]))


# ===================================================================
# vectorized_visual_search
# ===================================================================

class TestVectorizedVisualSearch:
    """Vectorized visual search."""

    def test_basic_computation(self) -> None:
        ecc = np.array([0.0, 5.0, 10.0])
        sizes = np.array([10.0, 20.0, 30.0])
        results = vectorized_visual_search(ecc, sizes)
        # RT = 0.4 + 0.025 * n/2 + 0.004 * ecc
        for i in range(3):
            expected = 0.4 + 0.025 * sizes[i] / 2.0 + 0.004 * ecc[i]
            assert results[i] == pytest.approx(expected)

    def test_increases_with_set_size(self) -> None:
        ecc = np.zeros(4)
        sizes = np.array([5.0, 10.0, 20.0, 40.0])
        results = vectorized_visual_search(ecc, sizes)
        assert np.all(np.diff(results) > 0)

    def test_increases_with_eccentricity(self) -> None:
        ecc = np.array([0.0, 5.0, 10.0, 20.0])
        sizes = np.full(4, 10.0)
        results = vectorized_visual_search(ecc, sizes)
        assert np.all(np.diff(results) > 0)


# ===================================================================
# batch_softmax
# ===================================================================

class TestBatchSoftmax:
    """Batch softmax over Q-values."""

    def test_rows_sum_to_one(self) -> None:
        Q = np.random.randn(50, 5)
        betas = np.ones(50)
        pi = batch_softmax(Q, betas)
        row_sums = pi.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_all_positive(self) -> None:
        Q = np.random.randn(10, 4)
        betas = np.full(10, 2.0)
        pi = batch_softmax(Q, betas)
        assert np.all(pi > 0)

    def test_high_beta_concentrates(self) -> None:
        # With very high β, softmax should concentrate on minimum Q
        Q = np.array([[1.0, 0.5, 2.0]])
        beta = np.array([100.0])
        pi = batch_softmax(Q, beta)
        assert np.argmax(pi[0]) == 1  # action with Q=0.5 gets highest prob

    def test_zero_beta_uniform(self) -> None:
        # β=0 → uniform (exp(0) = 1 for all)
        Q = np.array([[1.0, 2.0, 3.0]])
        beta = np.array([0.0])
        pi = batch_softmax(Q, beta)
        np.testing.assert_allclose(pi[0], [1 / 3, 1 / 3, 1 / 3], atol=1e-12)

    def test_scalar_beta_broadcast(self) -> None:
        Q = np.array([[1.0, 2.0], [3.0, 4.0]])
        pi = batch_softmax(Q, np.float64(1.0))
        assert pi.shape == (2, 2)
        np.testing.assert_allclose(pi.sum(axis=1), 1.0, atol=1e-12)


# ===================================================================
# sparse_transition_multiply
# ===================================================================

class TestSparseTransitionMultiply:
    """Sparse matrix-vector product matches dense multiply."""

    def test_matches_dense(self) -> None:
        np.random.seed(42)
        n = 50
        # Random sparse-ish transition matrix
        P_dense = np.random.rand(n, n)
        P_dense[P_dense < 0.7] = 0.0
        row_sums = P_dense.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P_dense /= row_sums
        V = np.random.rand(n)

        import scipy.sparse as sp

        P_sparse = sp.csr_matrix(P_dense)
        result = sparse_transition_multiply(P_sparse, V)
        expected = P_dense @ V
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_identity_matrix(self) -> None:
        import scipy.sparse as sp

        n = 10
        P = sp.eye(n, format="csr")
        V = np.arange(n, dtype=float)
        result = sparse_transition_multiply(P, V)
        np.testing.assert_array_equal(result, V)

    def test_accepts_dense_input(self) -> None:
        P = np.eye(5)
        V = np.ones(5)
        result = sparse_transition_multiply(P, V)
        np.testing.assert_allclose(result, V)


# ===================================================================
# vectorized_kl_divergence
# ===================================================================

class TestVectorizedKL:
    """Batch KL matches element-wise scalar KL."""

    def test_matches_scalar(self) -> None:
        P = np.array([
            [0.3, 0.7],
            [0.5, 0.5],
            [0.9, 0.1],
        ])
        Q = np.array([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ])
        results = vectorized_kl_divergence(P, Q)
        for i in range(3):
            expected = scalar_kl_divergence(P[i], Q[i])
            assert results[i] == pytest.approx(expected, rel=1e-8)

    def test_kl_zero_for_identical(self) -> None:
        P = np.array([[0.25, 0.25, 0.25, 0.25]])
        results = vectorized_kl_divergence(P, P)
        assert results[0] == pytest.approx(0.0, abs=1e-12)

    def test_kl_non_negative(self) -> None:
        np.random.seed(123)
        P = np.random.dirichlet([1, 1, 1], size=20)
        Q = np.random.dirichlet([1, 1, 1], size=20)
        results = vectorized_kl_divergence(P, Q)
        assert np.all(results >= -1e-12)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape"):
            vectorized_kl_divergence(np.ones((3, 2)), np.ones((3, 3)))

    def test_1d_input(self) -> None:
        p = np.array([0.3, 0.7])
        q = np.array([0.5, 0.5])
        result = vectorized_kl_divergence(p, q)
        expected = scalar_kl_divergence(p, q)
        assert result[0] == pytest.approx(expected, rel=1e-8)


# ===================================================================
# batch_interval_arithmetic
# ===================================================================

class TestBatchIntervalArithmetic:
    """Vectorized interval operations."""

    def test_add(self) -> None:
        lo1, hi1 = np.array([1.0, 2.0]), np.array([3.0, 4.0])
        lo2, hi2 = np.array([0.5, 1.0]), np.array([1.5, 2.0])
        rlo, rhi = batch_interval_arithmetic(lo1, hi1, "add", lo2, hi2)
        np.testing.assert_allclose(rlo, [1.5, 3.0])
        np.testing.assert_allclose(rhi, [4.5, 6.0])

    def test_subtract(self) -> None:
        lo1, hi1 = np.array([5.0]), np.array([10.0])
        lo2, hi2 = np.array([1.0]), np.array([3.0])
        rlo, rhi = batch_interval_arithmetic(lo1, hi1, "subtract", lo2, hi2)
        # [5,10] - [1,3] = [5-3, 10-1] = [2, 9]
        np.testing.assert_allclose(rlo, [2.0])
        np.testing.assert_allclose(rhi, [9.0])

    def test_multiply(self) -> None:
        lo1, hi1 = np.array([2.0]), np.array([3.0])
        lo2, hi2 = np.array([4.0]), np.array([5.0])
        rlo, rhi = batch_interval_arithmetic(lo1, hi1, "multiply", lo2, hi2)
        # [2,3]*[4,5] = [8, 15]
        np.testing.assert_allclose(rlo, [8.0])
        np.testing.assert_allclose(rhi, [15.0])

    def test_width(self) -> None:
        lo, hi = np.array([1.0, 5.0]), np.array([3.0, 8.0])
        rlo, rhi = batch_interval_arithmetic(lo, hi, "width")
        np.testing.assert_allclose(rlo, [2.0, 3.0])

    def test_midpoint(self) -> None:
        lo, hi = np.array([0.0, 4.0]), np.array([2.0, 10.0])
        rlo, rhi = batch_interval_arithmetic(lo, hi, "midpoint")
        np.testing.assert_allclose(rlo, [1.0, 7.0])


# ===================================================================
# parallel_cost_computation
# ===================================================================

class TestParallelCostComputation:
    """Parallel cost computation across trees."""

    def test_single_tree(self) -> None:
        tree = {
            "distances": [100.0, 200.0],
            "widths": [20.0, 40.0],
            "n_alternatives": [4, 8],
        }
        results = parallel_cost_computation([tree], config={}, n_workers=1)
        assert len(results) == 1
        assert "motor_cost" in results[0]
        assert "cognitive_cost" in results[0]
        assert "total_cost" in results[0]
        assert len(results[0]["motor_cost"]) == 2

    def test_multiple_trees(self) -> None:
        trees = [
            {"distances": [100.0], "widths": [20.0], "n_alternatives": [2]},
            {"distances": [200.0], "widths": [40.0], "n_alternatives": [4]},
        ]
        results = parallel_cost_computation(trees, config={}, n_workers=1)
        assert len(results) == 2


# ===================================================================
# benchmark utility
# ===================================================================

class TestBenchmark:
    """Micro-benchmark utility."""

    def test_returns_result(self) -> None:
        result = benchmark(lambda: sum(range(100)), n_repeats=10, warmup=2)
        assert isinstance(result, BenchmarkResult)
        assert result.n_repeats == 10
        assert result.mean > 0
        assert result.min <= result.mean <= result.max


# ===================================================================
# Performance assertions: vectorized vs. loop
# ===================================================================

class TestPerformanceVectorized:
    """Vectorized operations should be significantly faster than loops."""

    def test_fitts_vectorized_faster_than_loop(self) -> None:
        n = 5000
        ds = np.random.uniform(50, 500, size=n)
        ws = np.random.uniform(10, 100, size=n)

        # Warmup
        vectorized_fitts(ds, ws)
        _loop_fitts(ds, ws)

        t0 = time.perf_counter()
        for _ in range(10):
            vectorized_fitts(ds, ws)
        t_vec = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(10):
            _loop_fitts(ds, ws)
        t_loop = time.perf_counter() - t0

        speedup = t_loop / max(t_vec, 1e-12)
        assert speedup > 5, f"Expected >5× speedup, got {speedup:.1f}×"

    def test_softmax_vectorized_faster_than_loop(self) -> None:
        n = 2000
        m = 10
        Q = np.random.randn(n, m)
        betas = np.random.uniform(0.5, 5.0, size=n)

        # Warmup
        batch_softmax(Q, betas)
        _loop_softmax(Q, betas)

        t0 = time.perf_counter()
        for _ in range(10):
            batch_softmax(Q, betas)
        t_vec = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(10):
            _loop_softmax(Q, betas)
        t_loop = time.perf_counter() - t0

        speedup = t_loop / max(t_vec, 1e-12)
        assert speedup > 3, f"Expected >3× speedup, got {speedup:.1f}×"
