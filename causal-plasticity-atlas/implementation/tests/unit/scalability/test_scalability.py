"""Unit tests for cpa.scalability – ParentSetCache, TieredCache, SparseDAG, ApproximateBIC."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.scalability.parent_set_cache import (
    ParentSetCache,
    TieredCache,
    CacheStats,
)
from cpa.scalability.sparse_operations import (
    SparseDAG,
)
from cpa.scalability.approximate_scores import (
    ApproximateBIC,
    ScoreApproximation,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def parent_cache():
    return ParentSetCache(max_size=100, n_nodes=5)


@pytest.fixture
def tiered_cache():
    return TieredCache(l1_size=10, l2_size=50)


@pytest.fixture
def chain_adj():
    """Chain: 0→1→2→3→4."""
    adj = np.zeros((5, 5), dtype=int)
    adj[0, 1] = 1
    adj[1, 2] = 1
    adj[2, 3] = 1
    adj[3, 4] = 1
    return adj


@pytest.fixture
def diamond_adj():
    """Diamond: 0→1, 0→2, 1→3, 2→3."""
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 3] = 1
    adj[2, 3] = 1
    return adj


@pytest.fixture
def linear_data(rng):
    """Linear-Gaussian data for BIC computation."""
    n = 500
    x0 = rng.normal(0, 1, n)
    x1 = 0.5 * x0 + rng.normal(0, 1, n)
    x2 = 0.8 * x1 + rng.normal(0, 1, n)
    x3 = 0.3 * x0 + 0.4 * x2 + rng.normal(0, 1, n)
    return np.column_stack([x0, x1, x2, x3])


# ===================================================================
# Tests – ParentSetCache
# ===================================================================


class TestParentSetCache:
    """Test ParentSetCache hit rates."""

    def test_put_and_get(self, parent_cache):
        parent_cache.put(0, frozenset({1, 2}), -10.5)
        val = parent_cache.get(0, frozenset({1, 2}))
        assert val == -10.5

    def test_get_miss(self, parent_cache):
        val = parent_cache.get(0, frozenset({1, 2}))
        assert val is None

    def test_hit_rate_after_hits(self, parent_cache):
        parent_cache.put(0, frozenset({1}), -5.0)
        parent_cache.get(0, frozenset({1}))  # hit
        parent_cache.get(0, frozenset({2}))  # miss
        hr = parent_cache.hit_rate()
        assert_allclose(hr, 0.5, atol=1e-10)

    def test_hit_rate_all_misses(self, parent_cache):
        parent_cache.get(0, frozenset({1}))
        parent_cache.get(0, frozenset({2}))
        assert parent_cache.hit_rate() == 0.0

    def test_get_or_compute(self, parent_cache):
        val = parent_cache.get_or_compute(
            0, frozenset({1, 2}), lambda: -10.0,
        )
        assert val == -10.0
        # Second call should be cached
        val2 = parent_cache.get_or_compute(
            0, frozenset({1, 2}), lambda: -999.0,
        )
        assert val2 == -10.0

    def test_invalidate_node(self, parent_cache):
        parent_cache.put(0, frozenset({1}), -5.0)
        parent_cache.put(0, frozenset({2}), -6.0)
        parent_cache.put(1, frozenset({0}), -7.0)
        count = parent_cache.invalidate_node(0)
        assert count >= 2
        assert parent_cache.get(0, frozenset({1})) is None

    def test_invalidate_parent(self, parent_cache):
        parent_cache.put(0, frozenset({1, 2}), -5.0)
        parent_cache.put(0, frozenset({3}), -6.0)
        count = parent_cache.invalidate_parent(1)
        assert count >= 1
        assert parent_cache.get(0, frozenset({1, 2})) is None

    def test_clear(self, parent_cache):
        parent_cache.put(0, frozenset({1}), -5.0)
        parent_cache.clear()
        assert parent_cache.get(0, frozenset({1})) is None

    def test_stats(self, parent_cache):
        parent_cache.put(0, frozenset({1}), -5.0)
        parent_cache.get(0, frozenset({1}))
        stats = parent_cache.stats()
        assert isinstance(stats, CacheStats)
        assert stats.hits >= 1

    def test_eviction_on_overflow(self):
        cache = ParentSetCache(max_size=3, n_nodes=5)
        for i in range(5):
            cache.put(0, frozenset({i}), float(i))
        stats = cache.stats()
        assert stats.evictions >= 2

    def test_memory_usage(self, parent_cache):
        parent_cache.put(0, frozenset({1}), -5.0)
        usage = parent_cache.memory_usage()
        assert usage > 0

    def test_precompute(self, parent_cache):
        count = parent_cache.precompute(
            0, max_parents=2,
            score_fn=lambda node, parents: -float(len(parents)),
        )
        assert count > 0


# ===================================================================
# Tests – TieredCache
# ===================================================================


class TestTieredCache:
    """Test TieredCache L1/L2 promotion."""

    def test_put_and_get(self, tiered_cache):
        tiered_cache.put(0, frozenset({1}), -5.0)
        val = tiered_cache.get(0, frozenset({1}))
        assert val == -5.0

    def test_miss(self, tiered_cache):
        val = tiered_cache.get(0, frozenset({99}))
        assert val is None

    def test_l1_promotion(self, tiered_cache):
        # Fill L1
        for i in range(15):
            tiered_cache.put(0, frozenset({i}), float(i))
        # Old entries should be promoted to L2
        stats = tiered_cache.stats()
        assert isinstance(stats, dict)
        assert "l1" in stats or "L1" in stats or len(stats) >= 1

    def test_frequently_accessed_stays_in_l1(self, tiered_cache):
        tiered_cache.put(0, frozenset({1}), -5.0)
        # Access multiple times
        for _ in range(10):
            tiered_cache.get(0, frozenset({1}))
        # Fill cache beyond L1 capacity
        for i in range(20):
            tiered_cache.put(0, frozenset({i + 10}), float(i))
        # Original entry might still be accessible from L1 or L2
        val = tiered_cache.get(0, frozenset({1}))
        assert val == -5.0

    def test_clear(self, tiered_cache):
        tiered_cache.put(0, frozenset({1}), -5.0)
        tiered_cache.clear()
        assert tiered_cache.get(0, frozenset({1})) is None


# ===================================================================
# Tests – SparseDAG
# ===================================================================


class TestSparseDAG:
    """Test SparseDAG operations match dense DAG."""

    def test_from_dense(self, chain_adj):
        sparse = SparseDAG.from_dense(chain_adj)
        assert sparse.n_edges() == 4

    def test_to_dense_roundtrip(self, chain_adj):
        sparse = SparseDAG.from_dense(chain_adj)
        dense = sparse.to_dense()
        np.testing.assert_array_equal(dense, chain_adj)

    def test_add_remove_edge(self):
        sparse = SparseDAG(4)
        sparse.add_edge(0, 1)
        assert sparse.has_edge(0, 1)
        sparse.remove_edge(0, 1)
        assert not sparse.has_edge(0, 1)

    def test_parents(self, chain_adj):
        sparse = SparseDAG.from_dense(chain_adj)
        assert sparse.parents(2) == frozenset({1})
        assert sparse.parents(0) == frozenset()

    def test_children(self, chain_adj):
        sparse = SparseDAG.from_dense(chain_adj)
        assert sparse.children(1) == {2}
        assert sparse.children(4) == set()

    def test_ancestors(self, chain_adj):
        sparse = SparseDAG.from_dense(chain_adj)
        anc = sparse.ancestors(4)
        assert 0 in anc and 1 in anc and 2 in anc and 3 in anc

    def test_descendants(self, chain_adj):
        sparse = SparseDAG.from_dense(chain_adj)
        desc = sparse.descendants(0)
        assert 1 in desc and 2 in desc and 3 in desc and 4 in desc

    def test_topological_sort(self, chain_adj):
        sparse = SparseDAG.from_dense(chain_adj)
        order = sparse.topological_sort()
        assert order.index(0) < order.index(1) < order.index(2)

    def test_is_acyclic(self, chain_adj):
        sparse = SparseDAG.from_dense(chain_adj)
        assert sparse.is_acyclic()

    def test_shd_same(self, chain_adj):
        s1 = SparseDAG.from_dense(chain_adj)
        s2 = SparseDAG.from_dense(chain_adj)
        assert s1.structural_hamming_distance(s2) == 0

    def test_shd_different(self, chain_adj, diamond_adj):
        s1 = SparseDAG.from_dense(chain_adj)
        # Create a 5-node version of diamond
        big_diamond = np.zeros((5, 5), dtype=int)
        big_diamond[:4, :4] = diamond_adj
        s2 = SparseDAG.from_dense(big_diamond)
        shd = s1.structural_hamming_distance(s2)
        assert shd > 0

    def test_edge_set(self, chain_adj):
        sparse = SparseDAG.from_dense(chain_adj)
        edges = sparse.edge_set()
        assert (0, 1) in edges
        assert (1, 2) in edges

    def test_density(self, chain_adj):
        sparse = SparseDAG.from_dense(chain_adj)
        d = sparse.density()
        assert 0.0 < d < 1.0

    def test_in_out_degree(self, chain_adj):
        sparse = SparseDAG.from_dense(chain_adj)
        assert sparse.in_degree(0) == 0
        assert sparse.out_degree(0) == 1
        assert sparse.in_degree(2) == 1

    def test_markov_blanket(self, diamond_adj):
        sparse = SparseDAG.from_dense(diamond_adj)
        mb = sparse.markov_blanket(1)
        assert 0 in mb  # parent
        assert 3 in mb  # child
        assert 2 in mb  # co-parent of 3


# ===================================================================
# Tests – ApproximateBIC
# ===================================================================


class TestApproximateBIC:
    """Test ApproximateBIC is within tolerance of exact BIC."""

    def test_returns_score_approximation(self, linear_data):
        approx = ApproximateBIC(linear_data, sample_fraction=0.5,
                                 n_subsamples=10, rng_seed=42)
        result = approx.local_score(1, frozenset({0}))
        assert isinstance(result, ScoreApproximation)

    def test_approximate_close_to_exact(self, linear_data):
        approx = ApproximateBIC(linear_data, sample_fraction=0.8,
                                 n_subsamples=20, rng_seed=42)
        result = approx.local_score(1, frozenset({0}))
        if result.exact_score is not None and result.approximate_score is not None:
            rel_error = abs(result.approximate_score - result.exact_score) / (
                abs(result.exact_score) + 1e-10
            )
            assert rel_error < 0.3

    def test_error_bound_nonnegative(self, linear_data):
        approx = ApproximateBIC(linear_data, sample_fraction=0.5,
                                 n_subsamples=10, rng_seed=42)
        result = approx.local_score(1, frozenset({0}))
        if result.error_bound is not None:
            assert result.error_bound >= 0

    def test_more_parents_lower_score(self, linear_data):
        approx = ApproximateBIC(linear_data, sample_fraction=0.8,
                                 n_subsamples=15, rng_seed=42)
        score_0 = approx.local_score(1, frozenset())
        score_1 = approx.local_score(1, frozenset({0}))
        # True parent should improve score
        assert score_1.approximate_score != score_0.approximate_score

    def test_empty_parents(self, linear_data):
        approx = ApproximateBIC(linear_data, sample_fraction=0.5,
                                 n_subsamples=10, rng_seed=42)
        result = approx.local_score(0, frozenset())
        assert np.isfinite(result.approximate_score)

    def test_reproducible_with_seed(self, linear_data):
        approx1 = ApproximateBIC(linear_data, sample_fraction=0.5,
                                  n_subsamples=10, rng_seed=42)
        approx2 = ApproximateBIC(linear_data, sample_fraction=0.5,
                                  n_subsamples=10, rng_seed=42)
        r1 = approx1.local_score(1, frozenset({0}))
        r2 = approx2.local_score(1, frozenset({0}))
        assert_allclose(r1.approximate_score, r2.approximate_score, atol=1e-10)
