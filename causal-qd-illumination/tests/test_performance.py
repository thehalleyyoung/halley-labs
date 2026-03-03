"""Tests for batch descriptor computation and score caching optimizations."""
from __future__ import annotations

import time

import numpy as np
import pytest

from causal_qd.descriptors.fast_descriptors import (
    FastInfoTheoreticDescriptor,
    FastStructuralDescriptor,
    batch_info_theoretic_descriptors,
    batch_structural_descriptors,
)
from causal_qd.scores.bic import BICScore
from causal_qd.scores.cached import (
    CachedScore,
    DecomposableCachedScore,
    ParentSetCache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_dag(n: int, edge_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a random DAG by sampling edges in upper-triangular form."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                adj[i, j] = 1
    return adj


def _make_dags(count: int, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [_random_dag(n, 0.3, rng) for _ in range(count)]


def _make_data(n_samples: int, n_vars: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((n_samples, n_vars))


# ---------------------------------------------------------------------------
# Batch structural descriptor tests
# ---------------------------------------------------------------------------


class TestBatchStructuralDescriptors:
    """Tests for batch_structural_descriptors."""

    def test_matches_individual_simple_features(self):
        """Batch results match individual computation for vectorized features."""
        rng = np.random.default_rng(42)
        dags = _make_dags(10, 6, rng)
        features = ["edge_density", "max_in_degree", "parent_set_entropy"]

        batch_result = batch_structural_descriptors(dags, features=features)
        desc = FastStructuralDescriptor(features=features)
        individual = np.array([desc.compute(d) for d in dags])

        np.testing.assert_allclose(batch_result, individual, atol=1e-10)

    def test_matches_individual_all_features(self):
        """Batch results match individual for all structural features."""
        rng = np.random.default_rng(123)
        dags = _make_dags(8, 5, rng)
        features = [
            "edge_density", "max_in_degree", "v_structure_count",
            "longest_path", "avg_markov_blanket", "dag_depth",
        ]

        batch_result = batch_structural_descriptors(dags, features=features)
        desc = FastStructuralDescriptor(features=features)
        individual = np.array([desc.compute(d) for d in dags])

        np.testing.assert_allclose(batch_result, individual, atol=1e-10)

    def test_empty_input(self):
        """Batch returns empty array for empty input."""
        result = batch_structural_descriptors([])
        assert result.shape[0] == 0

    def test_output_shape(self):
        """Output has correct shape (n_dags, descriptor_dim)."""
        rng = np.random.default_rng(7)
        dags = _make_dags(15, 5, rng)
        features = ["edge_density", "max_in_degree"]
        result = batch_structural_descriptors(dags, features=features)
        assert result.shape == (15, 2)


# ---------------------------------------------------------------------------
# Batch info-theoretic descriptor tests
# ---------------------------------------------------------------------------


class TestBatchInfoTheoreticDescriptors:
    """Tests for batch_info_theoretic_descriptors."""

    def test_matches_individual(self):
        """Batch info-theoretic results match individual computation."""
        rng = np.random.default_rng(99)
        n_vars = 5
        dags = _make_dags(8, n_vars, rng)
        data = _make_data(200, n_vars, rng)
        features = ["mean_mi", "std_mi", "min_mi", "max_mi"]

        batch_result = batch_info_theoretic_descriptors(
            dags, data, features=features
        )
        desc = FastInfoTheoreticDescriptor(features=features)
        individual = np.array([desc.compute(d, data) for d in dags])

        np.testing.assert_allclose(batch_result, individual, atol=1e-10)

    def test_matches_individual_with_conditional_entropy(self):
        """Batch matches individual for conditional entropy features."""
        rng = np.random.default_rng(77)
        n_vars = 4
        dags = _make_dags(5, n_vars, rng)
        data = _make_data(150, n_vars, rng)
        features = ["mean_mi", "mean_conditional_entropy"]

        batch_result = batch_info_theoretic_descriptors(
            dags, data, features=features
        )
        desc = FastInfoTheoreticDescriptor(features=features)
        individual = np.array([desc.compute(d, data) for d in dags])

        np.testing.assert_allclose(batch_result, individual, atol=1e-10)


# ---------------------------------------------------------------------------
# Batch timing tests
# ---------------------------------------------------------------------------


class TestBatchTiming:
    """Tests that batch computation is faster than looping."""

    def test_batch_structural_faster_than_loop(self):
        """Batch structural descriptors are no slower than loop (generous)."""
        rng = np.random.default_rng(55)
        dags = _make_dags(50, 8, rng)
        features = ["edge_density", "max_in_degree", "avg_markov_blanket"]

        # Warm up
        batch_structural_descriptors(dags[:2], features=features)
        desc = FastStructuralDescriptor(features=features)
        desc.compute(dags[0])

        t0 = time.perf_counter()
        batch_structural_descriptors(dags, features=features)
        t_batch = time.perf_counter() - t0

        t0 = time.perf_counter()
        for d in dags:
            desc.compute(d)
        t_loop = time.perf_counter() - t0

        # Batch should be no more than 5x slower (generous margin)
        assert t_batch < t_loop * 5, (
            f"Batch ({t_batch:.4f}s) too slow vs loop ({t_loop:.4f}s)"
        )

    def test_batch_info_theoretic_faster_than_loop(self):
        """Batch info-theoretic descriptors benefit from precomputed MI."""
        rng = np.random.default_rng(66)
        n_vars = 5
        dags = _make_dags(30, n_vars, rng)
        data = _make_data(200, n_vars, rng)
        features = ["mean_mi", "std_mi", "max_mi"]

        # Warm up
        batch_info_theoretic_descriptors(dags[:2], data, features=features)
        desc = FastInfoTheoreticDescriptor(features=features)
        desc.compute(dags[0], data)

        t0 = time.perf_counter()
        batch_info_theoretic_descriptors(dags, data, features=features)
        t_batch = time.perf_counter() - t0

        t0 = time.perf_counter()
        for d in dags:
            desc.compute(d, data)
        t_loop = time.perf_counter() - t0

        assert t_batch < t_loop * 5, (
            f"Batch ({t_batch:.4f}s) too slow vs loop ({t_loop:.4f}s)"
        )


# ---------------------------------------------------------------------------
# ParentSetCache tests
# ---------------------------------------------------------------------------


class TestParentSetCache:
    """Tests for ParentSetCache."""

    def test_hit_rate_improves(self):
        """Hit rate increases when the same queries are repeated."""
        scorer = BICScore()
        cache = ParentSetCache(scorer.local_score, max_size=1024)
        rng = np.random.default_rng(42)
        data = _make_data(200, 5, rng)

        # First round: all misses
        for node in range(5):
            cache.score_family(node, [], data)
            cache.score_family(node, [0] if node != 0 else [1], data)

        stats_after_first = cache.stats
        assert stats_after_first.misses == 10

        # Second round: all hits
        for node in range(5):
            cache.score_family(node, [], data)
            cache.score_family(node, [0] if node != 0 else [1], data)

        stats_after_second = cache.stats
        assert stats_after_second.hits == 10
        assert stats_after_second.hit_rate > stats_after_first.hit_rate

    def test_precompute_all(self):
        """precompute_all covers expected parent sets."""
        scorer = BICScore()
        cache = ParentSetCache(scorer.local_score, max_size=65536)
        rng = np.random.default_rng(11)
        data = _make_data(100, 4, rng)

        cache.precompute_all(data, max_parents=2)

        # For 4 vars, max_parents=2: per node 3 candidates
        # sizes 0,1,2: C(3,0)+C(3,1)+C(3,2) = 1+3+3 = 7 per node, 4 nodes = 28
        stats = cache.stats
        assert stats.current_size == 28
        assert stats.misses == 28

        # All lookups should now be hits
        cache.score_family(0, [1, 2], data)
        cache.score_family(1, [], data)
        stats2 = cache.stats
        assert stats2.hits == 2

    def test_cache_eviction(self):
        """Cache evicts entries when max_size is exceeded."""
        scorer = BICScore()
        cache = ParentSetCache(scorer.local_score, max_size=5)
        rng = np.random.default_rng(33)
        data = _make_data(100, 4, rng)

        # Insert more than max_size entries
        for node in range(4):
            for pa in [[], [0], [1], [2], [3]]:
                if node not in pa:
                    cache.score_family(node, pa, data)

        stats = cache.stats
        assert stats.current_size <= 5

    def test_cache_statistics_accurate(self):
        """Statistics correctly track hits, misses, and size."""
        scorer = BICScore()
        cache = ParentSetCache(scorer.local_score, max_size=100)
        rng = np.random.default_rng(44)
        data = _make_data(100, 3, rng)

        cache.score_family(0, [], data)  # miss
        cache.score_family(0, [], data)  # hit
        cache.score_family(1, [0], data)  # miss
        cache.score_family(1, [0], data)  # hit
        cache.score_family(1, [0], data)  # hit

        stats = cache.stats
        assert stats.misses == 2
        assert stats.hits == 3
        assert stats.current_size == 2
        assert stats.memory_estimate_bytes == 2 * 128
        assert abs(stats.hit_rate - 3 / 5) < 1e-10

    def test_clear_resets_everything(self):
        """Clearing cache resets stats and entries."""
        scorer = BICScore()
        cache = ParentSetCache(scorer.local_score, max_size=100)
        rng = np.random.default_rng(55)
        data = _make_data(100, 3, rng)

        cache.score_family(0, [], data)
        cache.clear()

        stats = cache.stats
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.current_size == 0


# ---------------------------------------------------------------------------
# DecomposableCachedScore tests
# ---------------------------------------------------------------------------


class TestDecomposableCachedScore:
    """Tests for DecomposableCachedScore."""

    def test_same_result_as_uncached(self):
        """Cached score produces identical results to uncached base score."""
        scorer = BICScore()
        cached = DecomposableCachedScore(scorer)
        rng = np.random.default_rng(42)
        data = _make_data(200, 5, rng)
        dag = _random_dag(5, 0.3, rng)

        uncached_score = scorer.score(dag, data)
        cached_score = cached.score(dag, data)

        assert abs(cached_score - uncached_score) < 1e-10

    def test_delta_score_matches_full_rescore(self):
        """delta_score matches difference of full rescoring."""
        scorer = BICScore()
        cached = DecomposableCachedScore(scorer)
        rng = np.random.default_rng(88)
        data = _make_data(200, 5, rng)

        dag = np.zeros((5, 5), dtype=np.int8)
        dag[0, 1] = 1
        dag[1, 2] = 1

        score_before = cached.score(dag, data)

        # Add edge 0->2
        delta = cached.delta_score(dag, (0, 2), data)
        dag_new = dag.copy()
        dag_new[0, 2] = 1
        score_after = cached.score(dag_new, data)

        assert abs(delta - (score_after - score_before)) < 1e-10

    def test_delta_score_remove_matches_full_rescore(self):
        """delta_score_remove matches difference of full rescoring."""
        scorer = BICScore()
        cached = DecomposableCachedScore(scorer)
        rng = np.random.default_rng(77)
        data = _make_data(200, 5, rng)

        dag = np.zeros((5, 5), dtype=np.int8)
        dag[0, 1] = 1
        dag[1, 2] = 1
        dag[0, 2] = 1

        score_before = cached.score(dag, data)

        # Remove edge 0->2
        delta = cached.delta_score_remove(dag, (0, 2), data)
        dag_new = dag.copy()
        dag_new[0, 2] = 0
        score_after = cached.score(dag_new, data)

        assert abs(delta - (score_after - score_before)) < 1e-10

    def test_cache_access(self):
        """Can access underlying cache and check stats."""
        scorer = BICScore()
        cached = DecomposableCachedScore(scorer)
        rng = np.random.default_rng(42)
        data = _make_data(100, 4, rng)
        dag = _random_dag(4, 0.3, rng)

        cached.score(dag, data)
        cached.score(dag, data)  # second call should hit cache

        stats = cached.cache.stats
        assert stats.hits > 0
        assert stats.current_size > 0
