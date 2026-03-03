"""Unit tests for cpa.operators.crossover."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from cpa.operators.crossover import (
    EdgePreservingCrossover,
    MarkovBlanketCrossover,
    SubgraphCrossover,
    UniformCrossover,
    _is_acyclic,
    _markov_blanket,
    _parents,
    _children,
    _repair_dag,
)


# ── helpers ─────────────────────────────────────────────────────────

def _chain_dag(n: int = 4):
    """0 -> 1 -> 2 -> 3."""
    adj = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
    return adj


def _fork_dag():
    """0 -> 1, 0 -> 2, 0 -> 3."""
    adj = np.zeros((4, 4), dtype=float)
    adj[0, 1] = 1.0
    adj[0, 2] = 1.0
    adj[0, 3] = 1.0
    return adj


def _diamond_dag():
    """0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3."""
    adj = np.zeros((4, 4), dtype=float)
    adj[0, 1] = 1.0
    adj[0, 2] = 1.0
    adj[1, 3] = 1.0
    adj[2, 3] = 1.0
    return adj


def _complex_dag():
    """0->1, 0->2, 1->3, 2->3, 3->4."""
    adj = np.zeros((5, 5), dtype=float)
    adj[0, 1] = 1.0
    adj[0, 2] = 1.0
    adj[1, 3] = 1.0
    adj[2, 3] = 1.0
    adj[3, 4] = 1.0
    return adj


def _empty_dag(n: int = 4):
    return np.zeros((n, n), dtype=float)


# ── Helper function tests ─────────────────────────────────────────

class TestHelpers:
    def test_is_acyclic_true(self):
        assert _is_acyclic(_chain_dag())

    def test_is_acyclic_false(self):
        adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        assert not _is_acyclic(adj)

    def test_is_acyclic_empty(self):
        assert _is_acyclic(_empty_dag())

    def test_parents(self):
        adj = _diamond_dag()
        assert sorted(_parents(adj, 3)) == [1, 2]
        assert _parents(adj, 0) == []

    def test_children(self):
        adj = _diamond_dag()
        assert sorted(_children(adj, 0)) == [1, 2]
        assert _children(adj, 3) == []

    def test_markov_blanket(self):
        adj = _diamond_dag()
        # MB of node 1: parents={0}, children={3}, co-parents of 3={2}
        mb = _markov_blanket(adj, 1)
        assert mb == {0, 2, 3}

    def test_markov_blanket_root(self):
        adj = _diamond_dag()
        mb = _markov_blanket(adj, 0)
        assert mb == {1, 2}

    def test_markov_blanket_leaf(self):
        adj = _diamond_dag()
        mb = _markov_blanket(adj, 3)
        assert mb == {1, 2}

    def test_repair_dag(self):
        adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        repaired = _repair_dag(adj)
        assert _is_acyclic(repaired)


# ── MarkovBlanketCrossover tests ──────────────────────────────────

class TestMarkovBlanketCrossover:
    def test_offspring_is_dag(self):
        p1 = _chain_dag()
        p2 = _fork_dag()
        xo = MarkovBlanketCrossover(crossover_rate=1.0, seed=42)
        offspring = xo.crossover(p1, p2)
        assert _is_acyclic(offspring)

    def test_crossover_rate_zero_returns_parent(self):
        p1 = _chain_dag()
        p2 = _fork_dag()
        xo = MarkovBlanketCrossover(crossover_rate=0.0, seed=42)
        offspring = xo.crossover(p1, p2)
        assert_array_almost_equal(offspring, p1)

    def test_empty_graph_handled(self):
        p1 = _empty_dag(0)
        p2 = _empty_dag(0)
        xo = MarkovBlanketCrossover(crossover_rate=1.0, seed=42)
        offspring = xo.crossover(p1, p2)
        assert offspring.shape == (0, 0)

    def test_select_subgraph_returns_mb(self):
        xo = MarkovBlanketCrossover(seed=42)
        adj = _diamond_dag()
        sg = xo.select_subgraph(adj, 1)
        assert sg == {0, 2, 3}

    def test_repeated_crossover_produces_dags(self):
        p1 = _complex_dag()
        p2 = _diamond_dag()
        # Pad p2 to 5 nodes
        p2_big = np.zeros((5, 5), dtype=float)
        p2_big[:4, :4] = p2
        xo = MarkovBlanketCrossover(crossover_rate=1.0, seed=42)
        for seed in range(20):
            rng = np.random.default_rng(seed)
            offspring = xo.crossover(p1, p2_big, rng=rng)
            assert _is_acyclic(offspring), f"Failed at seed={seed}"

    def test_crossover_rate_parameter(self):
        """Crossover rate=1.0 should usually modify the parent."""
        p1 = _chain_dag()
        p2 = _fork_dag()
        xo = MarkovBlanketCrossover(crossover_rate=1.0, seed=42)
        n_different = 0
        for s in range(20):
            rng = np.random.default_rng(s)
            offspring = xo.crossover(p1, p2, rng=rng)
            if not np.array_equal(offspring, p1):
                n_different += 1
        assert n_different > 5  # most should be different

    def test_diamond_crossover(self):
        p1 = _diamond_dag()
        p2 = _chain_dag()
        xo = MarkovBlanketCrossover(crossover_rate=1.0, seed=42)
        offspring = xo.crossover(p1, p2)
        assert _is_acyclic(offspring)
        assert offspring.shape == (4, 4)


# ── EdgePreservingCrossover tests ─────────────────────────────────

class TestEdgePreservingCrossover:
    def test_offspring_is_dag(self):
        p1 = _chain_dag()
        p2 = _fork_dag()
        xo = EdgePreservingCrossover(crossover_rate=1.0, seed=42)
        offspring = xo.crossover(p1, p2)
        assert _is_acyclic(offspring)

    def test_shared_edges_preserved(self):
        p1 = _diamond_dag()
        p2 = np.zeros((4, 4), dtype=float)
        p2[0, 1] = 1.0  # shared with diamond
        p2[0, 3] = 1.0
        xo = EdgePreservingCrossover(crossover_rate=1.0, seed=42)
        offspring = xo.crossover(p1, p2)
        # 0->1 is shared and should always be in offspring
        assert offspring[0, 1] != 0

    def test_identify_common_edges(self):
        p1 = _diamond_dag()
        p2 = _chain_dag()
        xo = EdgePreservingCrossover(seed=42)
        common = xo.identify_common_edges(p1, p2)
        assert (0, 1) in common  # both have 0->1

    def test_crossover_rate_zero(self):
        p1 = _chain_dag()
        p2 = _fork_dag()
        xo = EdgePreservingCrossover(crossover_rate=0.0, seed=42)
        offspring = xo.crossover(p1, p2)
        assert_array_almost_equal(offspring, p1)

    def test_preservation_rate(self):
        """Higher preservation_rate should keep more parent1 edges."""
        p1 = _complex_dag()
        p2 = _empty_dag(5)
        p2[4, 0] = 1.0
        p2[3, 1] = 1.0
        xo_high = EdgePreservingCrossover(
            crossover_rate=1.0, preservation_rate=0.9, seed=42
        )
        xo_low = EdgePreservingCrossover(
            crossover_rate=1.0, preservation_rate=0.1, seed=42
        )
        # Count unique p1 edges in offspring over multiple trials
        p1_edges_high, p1_edges_low = 0, 0
        for s in range(30):
            rng = np.random.default_rng(s)
            oh = xo_high.crossover(p1, p2, rng=rng)
            p1_edges_high += (oh * (p1 != 0)).sum()
            rng = np.random.default_rng(s)
            ol = xo_low.crossover(p1, p2, rng=rng)
            p1_edges_low += (ol * (p1 != 0)).sum()
        assert p1_edges_high >= p1_edges_low

    def test_repeated_produces_dags(self):
        p1 = _complex_dag()
        p2_big = np.zeros((5, 5), dtype=float)
        p2_big[:4, :4] = _diamond_dag()
        xo = EdgePreservingCrossover(crossover_rate=1.0, seed=42)
        for s in range(20):
            rng = np.random.default_rng(s)
            offspring = xo.crossover(p1, p2_big, rng=rng)
            assert _is_acyclic(offspring)


# ── SubgraphCrossover tests ──────────────────────────────────────

class TestSubgraphCrossover:
    def test_offspring_is_dag(self):
        p1 = _complex_dag()
        p2 = _complex_dag().copy()
        p2[3, 4] = 0.0
        p2[4, 3] = 1.0  # reverse one edge... wait, check acyclicity
        p2 = _repair_dag(p2)
        xo = SubgraphCrossover(crossover_rate=1.0, seed=42)
        offspring = xo.crossover(p1, p2)
        assert _is_acyclic(offspring)

    def test_crossover_rate_zero(self):
        p1 = _chain_dag()
        p2 = _fork_dag()
        xo = SubgraphCrossover(crossover_rate=0.0, seed=42)
        offspring = xo.crossover(p1, p2)
        assert_array_almost_equal(offspring, p1)

    def test_small_graph(self):
        p1 = np.array([[0, 1], [0, 0]], dtype=float)
        p2 = np.array([[0, 0], [1, 0]], dtype=float)
        xo = SubgraphCrossover(crossover_rate=1.0, seed=42)
        offspring = xo.crossover(p1, p2)
        assert _is_acyclic(offspring)

    def test_maintains_dag_property(self):
        p1 = _complex_dag()
        p2 = _complex_dag()
        xo = SubgraphCrossover(crossover_rate=1.0, seed=42)
        for s in range(20):
            rng = np.random.default_rng(s)
            offspring = xo.crossover(p1, p2, rng=rng)
            assert _is_acyclic(offspring)


# ── UniformCrossover tests ────────────────────────────────────────

class TestUniformCrossover:
    def test_offspring_is_dag(self):
        p1 = _chain_dag()
        p2 = _fork_dag()
        xo = UniformCrossover(crossover_rate=1.0, seed=42)
        offspring = xo.crossover(p1, p2)
        assert _is_acyclic(offspring)

    def test_crossover_rate_zero(self):
        p1 = _chain_dag()
        p2 = _fork_dag()
        xo = UniformCrossover(crossover_rate=0.0, seed=42)
        offspring = xo.crossover(p1, p2)
        assert_array_almost_equal(offspring, p1)

    def test_repeated_produces_dags(self):
        p1 = _complex_dag()
        p2_big = np.zeros((5, 5), dtype=float)
        p2_big[:4, :4] = _diamond_dag()
        xo = UniformCrossover(crossover_rate=1.0, seed=42)
        for s in range(30):
            rng = np.random.default_rng(s)
            offspring = xo.crossover(p1, p2_big, rng=rng)
            assert _is_acyclic(offspring)
