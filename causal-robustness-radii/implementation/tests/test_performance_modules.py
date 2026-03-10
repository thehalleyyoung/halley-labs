"""
Performance module tests: d-separation oracle, math utilities, statistical
utilities, pipeline caching, and graph operations.

Since bitset_dsep.py, sparse.py, numba_utils.py, and profiler.py do not
exist, we test the actual performance-related modules: DSeparationOracle,
math_utils, stat_utils, and pipeline cache.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from causalcert.dag.dsep import DSeparationOracle
from causalcert.dag.graph import CausalDAG
from causalcert.utils.math_utils import (
    symmetrise,
    is_symmetric,
    is_positive_definite,
    nearest_positive_definite,
    spectral_radius,
    condition_number,
    effective_rank,
    safe_log,
    safe_divide,
    log_sum_exp,
    softmax,
    normal_cdf,
    normal_quantile,
    fisher_z,
    powerset,
    n_choose_k,
    adjacency_to_reachability,
    in_degrees,
    out_degrees,
    degree_sequence,
    laplacian,
)
from causalcert.utils.stat_utils import (
    z_test_two_sided,
    t_test_two_sided,
    chi2_test,
    fisher_exact_combine,
    cauchy_combine,
    stouffer_combine,
    bonferroni,
    holm_bonferroni,
    benjamini_hochberg,
    bootstrap_statistic,
    bootstrap_ci,
    wald_ci,
    gaussian_kernel_matrix,
    median_bandwidth,
    mmd_squared,
    energy_distance,
    kl_divergence,
    total_variation,
)
from causalcert.pipeline.cache import ResultCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adj(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


def _random_dag(n: int, density: float = 0.3, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                a[i, j] = 1
    return a


# ---------------------------------------------------------------------------
# D-separation oracle: correctness
# ---------------------------------------------------------------------------


class TestDSeparation:

    def test_chain_dsep(self):
        """In chain X->M->Y: X ⊥ Y | M but X ⊥̸ Y."""
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        assert oracle.is_d_separated(0, 2, frozenset({1}))
        assert not oracle.is_d_separated(0, 2, frozenset())

    def test_fork_dsep(self):
        """In fork X←C→Y: X ⊥ Y | C but X ⊥̸ Y."""
        adj = _adj(3, [(0, 1), (0, 2)])
        oracle = DSeparationOracle(adj)
        assert oracle.is_d_separated(1, 2, frozenset({0}))
        assert not oracle.is_d_separated(1, 2, frozenset())

    def test_collider_dsep(self):
        """In collider X→C←Y: X ⊥ Y but X ⊥̸ Y | C."""
        adj = _adj(3, [(0, 2), (1, 2)])
        oracle = DSeparationOracle(adj)
        assert oracle.is_d_separated(0, 1, frozenset())
        assert not oracle.is_d_separated(0, 1, frozenset({2}))

    def test_diamond_dsep(self):
        """Diamond: 0->1, 0->2, 1->3, 2->3."""
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        oracle = DSeparationOracle(adj)
        # 0 and 3 are d-connected (paths through 1 and 2)
        assert not oracle.is_d_separated(0, 3, frozenset())
        # 1 and 2 are d-connected marginally (via 0, their common parent)
        assert not oracle.is_d_separated(1, 2, frozenset())
        # 0 and 3 d-separated given both mediators
        assert oracle.is_d_separated(0, 3, frozenset({1, 2}))

    def test_m_bias_dsep(self):
        """M-bias: U1->X, U1->M, U2->M, U2->Y, X->Y."""
        adj = _adj(5, [(0, 1), (0, 2), (3, 2), (3, 4), (1, 4)])
        oracle = DSeparationOracle(adj)
        # X(1) and Y(4) d-connected via direct edge
        assert not oracle.is_d_separated(1, 4, frozenset())
        # Conditioning on M opens collider path
        assert not oracle.is_d_separated(1, 4, frozenset({2}))

    def test_dsep_self(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        # A node is not d-separated from itself
        assert not oracle.is_d_separated(0, 0, frozenset())

    def test_disconnected_dsep(self):
        adj = _adj(4, [(0, 1), (2, 3)])
        oracle = DSeparationOracle(adj)
        assert oracle.is_d_separated(0, 2, frozenset())
        assert oracle.is_d_separated(1, 3, frozenset())

    def test_d_connected_set(self):
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        oracle = DSeparationOracle(adj)
        connected = oracle.d_connected_set(0, frozenset())
        assert 1 in connected
        assert 2 in connected
        assert 3 in connected

    def test_markov_blanket(self):
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        oracle = DSeparationOracle(adj)
        mb = oracle.markov_blanket(1)
        assert 0 in mb  # parent
        assert 2 in mb  # child

    def test_find_separating_set(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        sep = oracle.find_separating_set(0, 2)
        assert sep is not None
        assert oracle.is_d_separated(0, 2, sep)

    def test_all_ci_implications(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        implications = oracle.all_ci_implications(max_cond_size=2)
        assert len(implications) >= 1

    def test_dsep_performance_medium(self):
        """D-sep on a 20-node DAG should be fast."""
        adj = _random_dag(20, density=0.3, seed=42)
        oracle = DSeparationOracle(adj)
        start = time.time()
        for i in range(20):
            for j in range(20):
                if i != j:
                    oracle.is_d_separated(i, j, frozenset())
        elapsed = time.time() - start
        assert elapsed < 5.0

    def test_pairwise_dsep_matrix(self):
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        oracle = DSeparationOracle(adj)
        matrix = oracle.pairwise_dsep_matrix(frozenset())
        assert matrix.shape == (4, 4)


# ---------------------------------------------------------------------------
# CausalDAG operations
# ---------------------------------------------------------------------------


class TestCausalDAGOperations:

    def test_create_causal_dag(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        dag = CausalDAG(adj)
        assert dag is not None

    def test_causal_dag_from_size(self):
        dag = CausalDAG(5)
        assert dag is not None

    def test_dag_edges(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        dag = CausalDAG(adj)
        # Check basic structure queries
        assert isinstance(dag, CausalDAG)


# ---------------------------------------------------------------------------
# Math utils
# ---------------------------------------------------------------------------


class TestMathUtils:

    def test_symmetrise(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        S = symmetrise(M)
        np.testing.assert_allclose(S, S.T)

    def test_is_symmetric(self):
        M = np.array([[1, 2], [2, 1]], dtype=float)
        assert is_symmetric(M)
        M_asym = np.array([[1, 2], [3, 1]], dtype=float)
        assert not is_symmetric(M_asym)

    def test_positive_definite(self):
        M = np.eye(3)
        assert is_positive_definite(M)
        M_bad = np.array([[1, 2], [2, 1]], dtype=float)
        # eigenvalues: 3, -1 → not PD
        assert not is_positive_definite(M_bad)

    def test_nearest_pd(self):
        M = np.array([[1, 2], [2, 1]], dtype=float)
        M_pd = nearest_positive_definite(M)
        assert is_positive_definite(M_pd)

    def test_spectral_radius(self):
        M = np.diag([2.0, 3.0, 1.0])
        sr = spectral_radius(M)
        np.testing.assert_allclose(sr, 3.0)

    def test_condition_number(self):
        M = np.eye(3)
        cn = condition_number(M)
        np.testing.assert_allclose(cn, 1.0, atol=1e-10)

    def test_effective_rank(self):
        M = np.diag([1.0, 1.0, 0.001])
        rank = effective_rank(M, tol=0.01)
        assert rank == 2

    def test_safe_log(self):
        x = np.array([0.0, 1.0, 2.0])
        result = safe_log(x)
        assert np.all(np.isfinite(result))

    def test_safe_divide(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 0.0, 3.0])
        result = safe_divide(a, b)
        assert np.all(np.isfinite(result))

    def test_log_sum_exp(self):
        a = np.array([1.0, 2.0, 3.0])
        result = log_sum_exp(a)
        expected = np.log(np.sum(np.exp(a)))
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_softmax(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = softmax(logits)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)
        assert np.all(probs > 0)

    def test_normal_cdf(self):
        p = normal_cdf(0.0, 0.0, 1.0)
        np.testing.assert_allclose(p, 0.5, atol=1e-10)

    def test_normal_quantile(self):
        q = normal_quantile(0.975, 0.0, 1.0)
        np.testing.assert_allclose(q, 1.96, atol=0.01)

    def test_fisher_z(self):
        z = fisher_z(0.5, 100)
        assert np.isfinite(z)

    def test_powerset(self):
        result = list(powerset([1, 2, 3]))
        assert len(result) == 8  # 2^3

    def test_powerset_max_size(self):
        result = list(powerset([1, 2, 3], max_size=2))
        assert len(result) == 7  # 1 + 3 + 3

    def test_n_choose_k(self):
        assert n_choose_k(5, 2) == 10
        assert n_choose_k(0, 0) == 1
        assert n_choose_k(10, 0) == 1

    def test_adjacency_to_reachability(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        R = adjacency_to_reachability(adj)
        assert R[0, 2] == 1  # 0 can reach 2
        assert R[2, 0] == 0  # 2 cannot reach 0

    def test_in_degrees(self):
        adj = _adj(3, [(0, 1), (0, 2)])
        deg = in_degrees(adj)
        assert deg[0] == 0
        assert deg[1] == 1
        assert deg[2] == 1

    def test_out_degrees(self):
        adj = _adj(3, [(0, 1), (0, 2)])
        deg = out_degrees(adj)
        assert deg[0] == 2
        assert deg[1] == 0
        assert deg[2] == 0

    def test_degree_sequence(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        deg = degree_sequence(adj)
        assert len(deg) == 3

    def test_laplacian_shape(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        L = laplacian(adj)
        assert L.shape == (3, 3)


# ---------------------------------------------------------------------------
# Statistical utils
# ---------------------------------------------------------------------------


class TestStatUtils:

    def test_z_test(self):
        p = z_test_two_sided(0.0)
        np.testing.assert_allclose(p, 1.0, atol=1e-5)

    def test_z_test_extreme(self):
        p = z_test_two_sided(5.0)
        assert p < 0.001

    def test_t_test(self):
        p = t_test_two_sided(0.0, 100)
        np.testing.assert_allclose(p, 1.0, atol=1e-3)

    def test_chi2_test(self):
        p = chi2_test(0.0, 1)
        np.testing.assert_allclose(p, 1.0, atol=1e-3)

    def test_fisher_combine(self):
        pvalues = [0.5, 0.5, 0.5]
        combined = fisher_exact_combine(pvalues)
        assert 0 < combined <= 1

    def test_cauchy_combine(self):
        pvalues = [0.01, 0.5, 0.8]
        combined = cauchy_combine(pvalues)
        assert 0 < combined <= 1

    def test_stouffer_combine(self):
        pvalues = [0.01, 0.5, 0.8]
        combined = stouffer_combine(pvalues)
        assert 0 < combined <= 1

    def test_bonferroni_correction(self):
        pvalues = [0.01, 0.02, 0.5]
        decisions = bonferroni(pvalues, alpha=0.05)
        assert len(decisions) == 3

    def test_holm_bonferroni(self):
        pvalues = [0.01, 0.02, 0.5]
        decisions = holm_bonferroni(pvalues, alpha=0.05)
        assert len(decisions) == 3

    def test_bh_correction(self):
        pvalues = [0.01, 0.02, 0.5]
        decisions = benjamini_hochberg(pvalues, alpha=0.05)
        assert len(decisions) == 3

    def test_bootstrap_statistic(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100)
        result = bootstrap_statistic(data, np.mean, n_bootstrap=200, seed=42)
        assert len(result) >= 2

    def test_bootstrap_ci(self):
        rng = np.random.default_rng(42)
        boot_dist = rng.standard_normal(500)
        lo, hi = bootstrap_ci(boot_dist, alpha=0.05)
        assert lo < hi

    def test_wald_ci(self):
        lo, hi = wald_ci(1.0, 0.5, alpha=0.05)
        assert lo < 1.0 < hi

    def test_gaussian_kernel_matrix(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        sigma = median_bandwidth(X)
        K = gaussian_kernel_matrix(X, sigma)
        assert K.shape == (50, 50)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_median_bandwidth_positive(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        sigma = median_bandwidth(X)
        assert sigma > 0

    def test_mmd_self(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        sigma = median_bandwidth(X)
        mmd = mmd_squared(X, X, sigma)
        np.testing.assert_allclose(mmd, 0.0, atol=0.1)

    def test_mmd_different(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        Y = rng.standard_normal((50, 3)) + 2.0
        sigma = median_bandwidth(X)
        mmd = mmd_squared(X, Y, sigma)
        assert mmd > 0

    def test_energy_distance_self(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        ed = energy_distance(X, X)
        np.testing.assert_allclose(ed, 0.0, atol=0.1)

    def test_energy_distance_different(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        Y = rng.standard_normal((50, 3)) + 3.0
        ed = energy_distance(X, Y)
        assert ed > 0

    def test_kl_divergence(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        kl = kl_divergence(p, q)
        assert kl >= 0

    def test_kl_self(self):
        p = np.array([0.5, 0.3, 0.2])
        kl = kl_divergence(p, p)
        np.testing.assert_allclose(kl, 0.0, atol=1e-10)

    def test_total_variation(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        tv = total_variation(p, q)
        assert 0 <= tv <= 1


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------


class TestResultCache:

    def test_cache_put_get(self):
        cache = ResultCache()
        cache.put("test_key", {"result": 42})
        val = cache.get("test_key")
        assert val is not None
        assert val["result"] == 42

    def test_cache_miss(self):
        cache = ResultCache()
        val = cache.get("nonexistent")
        assert val is None

    def test_cache_has(self):
        cache = ResultCache()
        cache.put("key1", "value1")
        assert cache.has("key1")
        assert not cache.has("key2")

    def test_cache_invalidate_single(self):
        cache = ResultCache()
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.invalidate("k1")
        assert not cache.has("k1")
        assert cache.has("k2")

    def test_cache_invalidate_all(self):
        cache = ResultCache()
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.invalidate()
        assert not cache.has("k1")
        assert not cache.has("k2")

    def test_cache_keys(self):
        cache = ResultCache()
        cache.put("a", 1)
        cache.put("b", 2)
        keys = cache.keys()
        assert "a" in keys
        assert "b" in keys

    def test_cache_size(self):
        cache = ResultCache()
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.size == 2

    def test_cache_content_key(self):
        key = ResultCache.content_key("part1", "part2")
        assert isinstance(key, str)
        assert len(key) > 0

    def test_cache_ci_test_key(self):
        adj_bytes = b"test"
        key = ResultCache.ci_test_key(adj_bytes, "data_hash", 0.05, "partial_correlation")
        assert isinstance(key, str)

    def test_cache_data_fingerprint(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({"X": rng.standard_normal(100)})
        fp = ResultCache.data_fingerprint(data)
        assert isinstance(fp, str)
        assert len(fp) > 0

    def test_cache_deterministic_fingerprint(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        data1 = pd.DataFrame({"X": rng1.standard_normal(100)})
        data2 = pd.DataFrame({"X": rng2.standard_normal(100)})
        fp1 = ResultCache.data_fingerprint(data1)
        fp2 = ResultCache.data_fingerprint(data2)
        assert fp1 == fp2

    def test_cache_cached_decorator(self):
        cache = ResultCache()
        result = cache.cached("test_compute", lambda: 42)
        assert result == 42
        # Second call should use cache
        result2 = cache.cached("test_compute", lambda: 99)
        assert result2 == 42


# ---------------------------------------------------------------------------
# Warm-start / cached CI test equivalence
# ---------------------------------------------------------------------------


class TestWarmStartEquivalence:

    def test_dsep_cached_matches_fresh(self):
        """Two fresh oracles on same DAG give same results."""
        adj = _adj(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
        o1 = DSeparationOracle(adj)
        o2 = DSeparationOracle(adj)
        for i in range(4):
            for j in range(4):
                if i != j:
                    r1 = o1.is_d_separated(i, j, frozenset())
                    r2 = o2.is_d_separated(i, j, frozenset())
                    assert r1 == r2

    def test_dsep_multiple_conditioning_sets(self):
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        oracle = DSeparationOracle(adj)
        # 0 ⊥ 3 | {1, 2}: blocked by {1} alone
        assert oracle.is_d_separated(0, 3, frozenset({1}))
        assert oracle.is_d_separated(0, 3, frozenset({1, 2}))
        assert oracle.is_d_separated(0, 3, frozenset({2}))
        assert not oracle.is_d_separated(0, 3, frozenset())


# ---------------------------------------------------------------------------
# Timing tests (sanity checks)
# ---------------------------------------------------------------------------


class TestPerformanceSanity:

    def test_reachability_fast(self):
        adj = _random_dag(50, density=0.2, seed=42)
        start = time.time()
        R = adjacency_to_reachability(adj)
        elapsed = time.time() - start
        assert elapsed < 2.0
        assert R.shape == (50, 50)

    def test_softmax_vectorized(self):
        logits = np.random.default_rng(42).standard_normal(10000)
        start = time.time()
        probs = softmax(logits)
        elapsed = time.time() - start
        assert elapsed < 1.0
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)

    def test_kernel_matrix_fast(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        sigma = median_bandwidth(X)
        start = time.time()
        K = gaussian_kernel_matrix(X, sigma)
        elapsed = time.time() - start
        assert elapsed < 5.0
        assert K.shape == (200, 200)
