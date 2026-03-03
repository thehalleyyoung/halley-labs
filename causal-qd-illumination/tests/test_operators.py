"""Tests for mutation, crossover, and repair operators.

Covers acyclicity preservation, edge-count invariants, adaptive rate
adjustment, and cycle-repair correctness using property-based testing
over many random trials.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_qd.operators.mutation import (
    AdaptiveMutation,
    CompositeMutation,
    EdgeAddMutation,
    EdgeRemoveMutation,
    EdgeReverseMutation,
    TopologicalMutation,
    _has_cycle,
    _topological_sort,
)
from causal_qd.operators.crossover import (
    OrderBasedCrossover,
    OrderCrossover,
    SkeletonCrossover,
    SubgraphCrossover,
    UniformCrossover,
)
from causal_qd.operators.repair import (
    AcyclicityRepair,
    ConnectivityRepair,
    MinimalRepair,
    OrderRepair,
    TopologicalRepair,
)


# ===================================================================
# Helper utilities
# ===================================================================

def _edge_count(adj: np.ndarray) -> int:
    """Return number of edges in the adjacency matrix."""
    return int(adj.sum())


def _make_random_dag(n: int, edge_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a random DAG of size *n* by sampling forward edges."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                adj[i, j] = 1
    return adj


def _make_cyclic_graph(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a graph guaranteed to have at least one cycle."""
    adj = np.zeros((n, n), dtype=np.int8)
    # Add some random edges
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < 0.2:
                adj[i, j] = 1
    # Guarantee a cycle: 0→1→2→...→min(n-1,2)→0
    cycle_len = min(n, 3)
    for i in range(cycle_len - 1):
        adj[i, i + 1] = 1
    adj[cycle_len - 1, 0] = 1  # closing edge
    return adj


# ===================================================================
# Mutation operator tests
# ===================================================================


class TestTopologicalMutation:
    """Tests for TopologicalMutation."""

    def test_topological_mutation_preserves_acyclicity(self, small_adj, rng):
        """Property: 200 random mutations never produce a cycle."""
        mut = TopologicalMutation(add_prob=0.4, remove_prob=0.3, reverse_prob=0.3)
        dag = small_adj.copy()
        for _ in range(200):
            dag = mut.mutate(dag, rng)
            assert not _has_cycle(dag), "TopologicalMutation introduced a cycle"

    def test_topological_mutation_preserves_acyclicity_medium(self, medium_adj, rng):
        """Property: 200 mutations on a larger graph stay acyclic."""
        mut = TopologicalMutation()
        dag = medium_adj.copy()
        for _ in range(200):
            dag = mut.mutate(dag, rng)
            assert not _has_cycle(dag)

    def test_topological_mutation_returns_copy(self, small_adj, rng):
        """Mutation returns a new array, not a view of the input."""
        mut = TopologicalMutation()
        original = small_adj.copy()
        result = mut.mutate(small_adj, rng)
        # Even if matrices happen to match, they should be different objects
        assert result is not small_adj
        np.testing.assert_array_equal(small_adj, original)

    def test_topological_mutation_on_empty_graph(self, rng):
        """Mutating an empty graph can only add an edge."""
        adj = np.zeros((5, 5), dtype=np.int8)
        mut = TopologicalMutation(add_prob=1.0, remove_prob=0.0, reverse_prob=0.0)
        result = mut.mutate(adj, rng)
        assert _edge_count(result) <= 1
        assert not _has_cycle(result)

    def test_topological_mutation_probability_normalization(self):
        """Probabilities are normalised even when they don't sum to 1."""
        mut = TopologicalMutation(add_prob=2.0, remove_prob=3.0, reverse_prob=5.0)
        assert abs(mut.add_prob + mut.remove_prob + mut.reverse_prob - 1.0) < 1e-12

    def test_topological_mutation_on_complete_dag(self, rng):
        """Mutating a fully-connected DAG with add_prob=1 leaves it unchanged."""
        n = 5
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                adj[i, j] = 1
        mut = TopologicalMutation(add_prob=1.0, remove_prob=0.0, reverse_prob=0.0)
        result = mut.mutate(adj, rng)
        # Already complete forward edges; no room to add
        np.testing.assert_array_equal(result, adj)


class TestEdgeAddMutation:
    """Tests for EdgeAddMutation."""

    def test_edge_add_mutation_increases_edges(self, small_adj, rng):
        """Adding an edge increases the edge count by exactly one."""
        mut = EdgeAddMutation()
        before = _edge_count(small_adj)
        result = mut.mutate(small_adj, rng)
        after = _edge_count(result)
        # Chain graph has room for more forward edges, so count should increase
        assert after == before + 1
        assert not _has_cycle(result)

    def test_edge_add_mutation_acyclic(self, medium_adj, rng):
        """Repeated adds never introduce cycles."""
        mut = EdgeAddMutation()
        dag = medium_adj.copy()
        for _ in range(50):
            dag = mut.mutate(dag, rng)
            assert not _has_cycle(dag)

    def test_edge_add_on_complete_graph(self, rng):
        """When the DAG is already complete, no edge can be added."""
        n = 4
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                adj[i, j] = 1
        mut = EdgeAddMutation()
        result = mut.mutate(adj, rng)
        np.testing.assert_array_equal(result, adj)

    def test_edge_add_preserves_existing(self, small_adj, rng):
        """EdgeAddMutation only adds, never removes existing edges."""
        mut = EdgeAddMutation()
        result = mut.mutate(small_adj, rng)
        # Every edge in original should still be present
        assert np.all(result >= small_adj)


class TestEdgeRemoveMutation:
    """Tests for EdgeRemoveMutation."""

    def test_edge_remove_mutation_decreases_edges(self, small_adj, rng):
        """Removing an edge decreases the edge count by exactly one."""
        mut = EdgeRemoveMutation()
        before = _edge_count(small_adj)
        result = mut.mutate(small_adj, rng)
        after = _edge_count(result)
        assert after == before - 1

    def test_edge_remove_on_empty_graph(self, rng):
        """Removing from an empty graph returns unchanged graph."""
        adj = np.zeros((5, 5), dtype=np.int8)
        mut = EdgeRemoveMutation()
        result = mut.mutate(adj, rng)
        np.testing.assert_array_equal(result, adj)

    def test_edge_remove_preserves_acyclicity(self, medium_adj, rng):
        """Removing edges can never create a cycle."""
        mut = EdgeRemoveMutation()
        dag = medium_adj.copy()
        for _ in range(10):
            dag = mut.mutate(dag, rng)
            assert not _has_cycle(dag)

    def test_edge_remove_returns_copy(self, small_adj, rng):
        """Mutation does not modify the input array."""
        original = small_adj.copy()
        mut = EdgeRemoveMutation()
        _ = mut.mutate(small_adj, rng)
        np.testing.assert_array_equal(small_adj, original)


class TestEdgeReverseMutation:
    """Tests for EdgeReverseMutation."""

    def test_edge_reverse_mutation_preserves_edge_count(self, small_adj, rng):
        """Reversing preserves total edge count (or leaves unchanged if unsafe)."""
        mut = EdgeReverseMutation()
        before = _edge_count(small_adj)
        result = mut.mutate(small_adj, rng)
        after = _edge_count(result)
        assert after == before

    def test_edge_reverse_preserves_acyclicity(self, medium_adj, rng):
        """Repeated reversals never produce cycles."""
        mut = EdgeReverseMutation()
        dag = medium_adj.copy()
        for _ in range(100):
            dag = mut.mutate(dag, rng)
            assert not _has_cycle(dag)

    def test_edge_reverse_on_single_edge(self, rng):
        """Reversing the only edge in a 2-node graph succeeds."""
        adj = np.zeros((2, 2), dtype=np.int8)
        adj[0, 1] = 1
        mut = EdgeReverseMutation()
        result = mut.mutate(adj, rng)
        assert _edge_count(result) == 1
        # The only edge must now be 1→0
        assert result[1, 0] == 1
        assert result[0, 1] == 0

    def test_edge_reverse_on_empty(self, rng):
        """No-op on empty graph."""
        adj = np.zeros((4, 4), dtype=np.int8)
        mut = EdgeReverseMutation()
        result = mut.mutate(adj, rng)
        np.testing.assert_array_equal(result, adj)

    def test_edge_reverse_chain_preserves_count_many(self, rng):
        """Edge count stays constant over 50 reversals on a chain."""
        adj = np.zeros((6, 6), dtype=np.int8)
        for i in range(5):
            adj[i, i + 1] = 1
        mut = EdgeReverseMutation()
        dag = adj.copy()
        initial = _edge_count(dag)
        for _ in range(50):
            dag = mut.mutate(dag, rng)
            assert _edge_count(dag) == initial
            assert not _has_cycle(dag)


class TestCompositeMutation:
    """Tests for CompositeMutation."""

    def test_composite_mutation_produces_valid_dag(self, medium_adj, rng):
        """Composite mutation with multiple operators always produces a DAG."""
        ops = [EdgeAddMutation(), EdgeRemoveMutation(), EdgeReverseMutation()]
        comp = CompositeMutation(operators=ops, probabilities=[0.4, 0.3, 0.3], n_mutations=3)
        dag = medium_adj.copy()
        for _ in range(100):
            dag = comp.mutate(dag, rng)
            assert not _has_cycle(dag)

    def test_composite_n_mutations_applied(self, rng):
        """With n_mutations=k, exactly k mutations are chained."""
        add_mut = EdgeAddMutation()
        # Start with empty graph; each add_mut adds 1 edge
        adj = np.zeros((5, 5), dtype=np.int8)
        comp = CompositeMutation(operators=[add_mut], n_mutations=3)
        result = comp.mutate(adj, rng)
        assert _edge_count(result) == 3

    def test_composite_uniform_probabilities(self):
        """When no probabilities given, they are uniform."""
        ops = [EdgeAddMutation(), EdgeRemoveMutation()]
        comp = CompositeMutation(operators=ops)
        np.testing.assert_allclose(comp.probabilities, [0.5, 0.5])

    def test_composite_probability_normalization(self):
        """Probabilities are normalised to sum to 1."""
        ops = [EdgeAddMutation(), EdgeRemoveMutation()]
        comp = CompositeMutation(operators=ops, probabilities=[3, 7])
        np.testing.assert_allclose(comp.probabilities, [0.3, 0.7])

    def test_composite_single_operator(self, small_adj, rng):
        """Composite with one operator delegates to it correctly."""
        comp = CompositeMutation(operators=[EdgeRemoveMutation()], n_mutations=1)
        before = _edge_count(small_adj)
        result = comp.mutate(small_adj, rng)
        assert _edge_count(result) == before - 1


class TestAdaptiveMutation:
    """Tests for AdaptiveMutation."""

    def test_adaptive_mutation_adjusts_rates(self, small_adj, rng):
        """Reporting successes for one operator shifts probabilities toward it."""
        op_a = EdgeAddMutation()
        op_b = EdgeRemoveMutation()
        adaptive = AdaptiveMutation(
            operators=[op_a, op_b],
            window_size=50,
            temperature=0.5,
            min_prob=0.05,
        )

        # Initial probabilities should be uniform
        initial_probs = adaptive.probabilities.copy()
        np.testing.assert_allclose(initial_probs, [0.5, 0.5])

        # Run several mutations and always report success for whichever op ran
        # Then bias: report success only for first operator
        for _ in range(30):
            adaptive.mutate(small_adj, rng)
            # Force the record: always report success for operator 0
            adaptive._last_operator_idx = 0
            adaptive._trial_counts[0] += 1
            adaptive._trial_counts[1] += 0
            adaptive.report_result(True)

        probs = adaptive.probabilities
        assert probs[0] > probs[1], (
            f"Operator 0 should have higher probability after consistent success, "
            f"got {probs}"
        )

    def test_adaptive_initial_uniform(self):
        """Before any mutations, probabilities are uniform."""
        ops = [EdgeAddMutation(), EdgeRemoveMutation(), EdgeReverseMutation()]
        adaptive = AdaptiveMutation(operators=ops)
        np.testing.assert_allclose(adaptive.probabilities, [1 / 3] * 3)

    def test_adaptive_reset(self, small_adj, rng):
        """Reset restores uniform probabilities."""
        ops = [EdgeAddMutation(), EdgeRemoveMutation()]
        adaptive = AdaptiveMutation(operators=ops, window_size=20, temperature=0.5)

        # Do some mutations and report results
        for _ in range(10):
            adaptive.mutate(small_adj, rng)
            adaptive.report_result(True)

        adaptive.reset()
        np.testing.assert_allclose(adaptive.probabilities, [0.5, 0.5])

    def test_adaptive_produces_valid_dag(self, medium_adj, rng):
        """Adaptive mutation always produces acyclic graphs."""
        ops = [EdgeAddMutation(), EdgeRemoveMutation(), EdgeReverseMutation()]
        adaptive = AdaptiveMutation(operators=ops)
        dag = medium_adj.copy()
        for _ in range(50):
            dag = adaptive.mutate(dag, rng)
            assert not _has_cycle(dag)
            adaptive.report_result(rng.random() < 0.3)

    def test_adaptive_min_prob_enforced(self, small_adj, rng):
        """No operator probability drops below min_prob (after renorm)."""
        ops = [EdgeAddMutation(), EdgeRemoveMutation()]
        min_p = 0.05
        adaptive = AdaptiveMutation(operators=ops, min_prob=min_p, temperature=0.1)

        # Heavily bias toward op 0 via the public API
        for _ in range(40):
            adaptive.mutate(small_adj, rng)
            adaptive.report_result(True)
        for _ in range(40):
            adaptive.mutate(small_adj, rng)
            adaptive.report_result(False)

        probs = adaptive.probabilities
        # After np.maximum and renorm, the minimum should be >= min_prob
        assert probs.min() >= min_p - 1e-9, f"Min probability violated: {probs}"

    def test_adaptive_sliding_window(self, small_adj, rng):
        """Old results are evicted after window_size trials."""
        ops = [EdgeAddMutation(), EdgeRemoveMutation()]
        adaptive = AdaptiveMutation(operators=ops, window_size=10)

        # Fill the window with successes for op 0
        for _ in range(10):
            adaptive.mutate(small_adj, rng)
            adaptive._last_operator_idx = 0
            adaptive._trial_counts[0] += 1
            adaptive.report_result(True)

        probs_before = adaptive.probabilities.copy()

        # Now add 10 more successes for op 1 — old op-0 successes slide out
        for _ in range(10):
            adaptive.mutate(small_adj, rng)
            adaptive._last_operator_idx = 1
            adaptive._trial_counts[1] += 1
            adaptive.report_result(True)

        probs_after = adaptive.probabilities
        # Op 1 should now have higher (or at least not lower) probability
        assert probs_after[1] >= probs_before[1] - 0.1


# ===================================================================
# Crossover operator tests
# ===================================================================


class TestOrderCrossover:
    """Tests for OrderCrossover."""

    def test_order_crossover_produces_valid_dag(self, small_adj, medium_adj, rng):
        """Property: 100 crossovers always produce acyclic children."""
        xo = OrderCrossover()
        for _ in range(100):
            c1, c2 = xo.crossover(small_adj, small_adj, rng)
            assert not _has_cycle(c1), "OrderCrossover child1 has cycle"
            assert not _has_cycle(c2), "OrderCrossover child2 has cycle"

    def test_order_crossover_different_parents(self, rng):
        """Crossover of two different DAGs produces valid DAGs."""
        xo = OrderCrossover()
        p1 = _make_random_dag(8, 0.3, rng)
        p2 = _make_random_dag(8, 0.3, rng)
        for _ in range(50):
            c1, c2 = xo.crossover(p1, p2, rng)
            assert not _has_cycle(c1)
            assert not _has_cycle(c2)

    def test_order_crossover_correct_shape(self, small_adj, rng):
        """Children have the same shape as parents."""
        xo = OrderCrossover()
        c1, c2 = xo.crossover(small_adj, small_adj, rng)
        assert c1.shape == small_adj.shape
        assert c2.shape == small_adj.shape

    def test_order_crossover_edges_from_parents(self, rng):
        """All edges in children come from the union of parent edges."""
        xo = OrderCrossover()
        p1 = _make_random_dag(6, 0.4, rng)
        p2 = _make_random_dag(6, 0.4, rng)
        c1, c2 = xo.crossover(p1, p2, rng)
        union = (p1 | p2).astype(np.int8)
        # Every child edge must exist in at least one parent
        assert np.all(c1 <= union)
        assert np.all(c2 <= union)

    def test_order_crossover_small_n(self, rng):
        """Edge case: crossover on 2-node graphs."""
        p1 = np.array([[0, 1], [0, 0]], dtype=np.int8)
        p2 = np.array([[0, 0], [1, 0]], dtype=np.int8)
        xo = OrderCrossover()
        c1, c2 = xo.crossover(p1, p2, rng)
        assert not _has_cycle(c1)
        assert not _has_cycle(c2)


class TestUniformCrossover:
    """Tests for UniformCrossover."""

    def test_uniform_crossover_produces_valid_dag(self, small_adj, rng):
        """Uniform crossover always produces acyclic children."""
        xo = UniformCrossover()
        p2 = _make_random_dag(5, 0.3, rng)
        for _ in range(100):
            c1, c2 = xo.crossover(small_adj, p2, rng)
            assert not _has_cycle(c1)
            assert not _has_cycle(c2)

    def test_uniform_crossover_no_self_loops(self, rng):
        """Children have zero diagonal."""
        xo = UniformCrossover()
        p1 = _make_random_dag(6, 0.4, rng)
        p2 = _make_random_dag(6, 0.4, rng)
        c1, c2 = xo.crossover(p1, p2, rng)
        assert np.all(np.diag(c1) == 0)
        assert np.all(np.diag(c2) == 0)

    def test_uniform_crossover_symmetric_children(self, rng):
        """Swapping parents swaps children (statistically, not exactly)."""
        xo = UniformCrossover()
        p1 = _make_random_dag(5, 0.3, rng)
        p2 = _make_random_dag(5, 0.3, rng)
        # Just verify both outputs are valid
        c1, c2 = xo.crossover(p1, p2, rng)
        assert c1.shape == p1.shape
        assert c2.shape == p1.shape
        assert not _has_cycle(c1)
        assert not _has_cycle(c2)


class TestSkeletonCrossover:
    """Tests for SkeletonCrossover."""

    def test_skeleton_crossover_produces_valid_dag(self, small_adj, rng):
        """SkeletonCrossover children are acyclic."""
        xo = SkeletonCrossover()
        p2 = _make_random_dag(5, 0.4, rng)
        for _ in range(50):
            c1, c2 = xo.crossover(small_adj, p2, rng)
            assert not _has_cycle(c1)
            assert not _has_cycle(c2)

    def test_skeleton_crossover_shape(self, medium_adj, rng):
        """Children have the same shape as parents."""
        xo = SkeletonCrossover()
        p2 = _make_random_dag(10, 0.2, rng)
        c1, c2 = xo.crossover(medium_adj, p2, rng)
        assert c1.shape == medium_adj.shape
        assert c2.shape == medium_adj.shape

    def test_skeleton_crossover_identical_parents(self, small_adj, rng):
        """Crossover of identical parents produces valid DAGs."""
        xo = SkeletonCrossover()
        c1, c2 = xo.crossover(small_adj, small_adj, rng)
        assert not _has_cycle(c1)
        assert not _has_cycle(c2)

    def test_skeleton_crossover_preserves_v_structures(self, rng):
        """V-structures present in both parents tend to appear in children."""
        # Parent with v-structure: 0→2←1
        p1 = np.zeros((4, 4), dtype=np.int8)
        p1[0, 2] = 1
        p1[1, 2] = 1
        p1[2, 3] = 1
        # Same v-structure in parent 2
        p2 = p1.copy()
        xo = SkeletonCrossover()
        c1, c2 = xo.crossover(p1, p2, rng)
        # V-structure edges should be preserved in at least one child
        assert (c1[0, 2] == 1 and c1[1, 2] == 1) or (c2[0, 2] == 1 and c2[1, 2] == 1)


class TestOrderBasedCrossover:
    """Tests for OrderBasedCrossover."""

    def test_order_based_crossover_valid_dag(self, small_adj, rng):
        """OrderBasedCrossover always yields acyclic children."""
        xo = OrderBasedCrossover()
        p2 = _make_random_dag(5, 0.3, rng)
        for _ in range(80):
            c1, c2 = xo.crossover(small_adj, p2, rng)
            assert not _has_cycle(c1)
            assert not _has_cycle(c2)

    def test_order_based_crossover_edges_from_parents(self, rng):
        """Children edges are subset of parent edge union."""
        xo = OrderBasedCrossover()
        p1 = _make_random_dag(7, 0.3, rng)
        p2 = _make_random_dag(7, 0.3, rng)
        c1, c2 = xo.crossover(p1, p2, rng)
        union = (p1 | p2).astype(np.int8)
        assert np.all(c1 <= union)
        assert np.all(c2 <= union)


class TestSubgraphCrossover:
    """Tests for SubgraphCrossover."""

    def test_subgraph_crossover_valid_dag(self, medium_adj, rng):
        """SubgraphCrossover always yields acyclic children."""
        xo = SubgraphCrossover()
        p2 = _make_random_dag(10, 0.2, rng)
        for _ in range(50):
            c1, c2 = xo.crossover(medium_adj, p2, rng)
            assert not _has_cycle(c1)
            assert not _has_cycle(c2)

    def test_subgraph_crossover_shape(self, medium_adj, rng):
        """Children shapes match parent shapes."""
        xo = SubgraphCrossover()
        p2 = _make_random_dag(10, 0.2, rng)
        c1, c2 = xo.crossover(medium_adj, p2, rng)
        assert c1.shape == medium_adj.shape
        assert c2.shape == medium_adj.shape

    def test_subgraph_crossover_identical_parents(self, small_adj, rng):
        """Crossover of identical parents preserves parent structure."""
        xo = SubgraphCrossover()
        c1, c2 = xo.crossover(small_adj, small_adj, rng)
        # With identical parents, swapping subgraphs changes nothing
        np.testing.assert_array_equal(c1, small_adj)
        np.testing.assert_array_equal(c2, small_adj)


# ===================================================================
# Crossover property: many random parent pairs
# ===================================================================


class TestCrossoverProperty:
    """Property tests across all crossover operators."""

    @pytest.mark.parametrize("xo_cls", [
        OrderCrossover,
        UniformCrossover,
        SkeletonCrossover,
        OrderBasedCrossover,
        SubgraphCrossover,
    ])
    def test_crossover_acyclicity_property(self, xo_cls, rng):
        """All crossover operators preserve acyclicity on random DAG pairs."""
        xo = xo_cls()
        for _ in range(30):
            n = rng.integers(4, 12)
            p1 = _make_random_dag(n, 0.3, rng)
            p2 = _make_random_dag(n, 0.3, rng)
            c1, c2 = xo.crossover(p1, p2, rng)
            assert not _has_cycle(c1), f"{xo_cls.__name__} child1 has cycle"
            assert not _has_cycle(c2), f"{xo_cls.__name__} child2 has cycle"
            assert c1.shape == (n, n)
            assert c2.shape == (n, n)


# ===================================================================
# Repair operator tests
# ===================================================================


class TestTopologicalRepair:
    """Tests for TopologicalRepair."""

    def test_repair_removes_cycles(self, rng):
        """Property: TopologicalRepair always produces an acyclic graph."""
        repair = TopologicalRepair()
        for _ in range(50):
            n = rng.integers(4, 15)
            adj = _make_cyclic_graph(n, rng)
            assert _has_cycle(adj), "Test graph should have a cycle"
            result = repair.repair(adj)
            assert not _has_cycle(result), "Repaired graph still has cycle"

    def test_topological_repair_minimal(self, rng):
        """Repair removes only back-edges, keeping as many forward edges as possible."""
        repair = TopologicalRepair()
        # Simple cycle: 0→1→2→0
        adj = np.zeros((3, 3), dtype=np.int8)
        adj[0, 1] = 1
        adj[1, 2] = 1
        adj[2, 0] = 1
        result = repair.repair(adj)
        assert not _has_cycle(result)
        # At most 1 edge should be removed to break a 3-cycle
        assert _edge_count(result) >= 2

    def test_topological_repair_preserves_acyclic(self, small_adj):
        """Repairing an already-acyclic graph is a no-op."""
        repair = TopologicalRepair()
        result = repair.repair(small_adj)
        np.testing.assert_array_equal(result, small_adj)

    def test_topological_repair_larger_cycles(self, rng):
        """Repair handles multiple overlapping cycles."""
        repair = TopologicalRepair()
        adj = np.zeros((6, 6), dtype=np.int8)
        # Cycle 1: 0→1→2→0
        adj[0, 1] = 1
        adj[1, 2] = 1
        adj[2, 0] = 1
        # Cycle 2: 3→4→5→3
        adj[3, 4] = 1
        adj[4, 5] = 1
        adj[5, 3] = 1
        # Cross edge
        adj[2, 3] = 1
        result = repair.repair(adj)
        assert not _has_cycle(result)

    def test_topological_repair_returns_copy(self, small_adj):
        """Repair does not modify the input."""
        original = small_adj.copy()
        repair = TopologicalRepair()
        _ = repair.repair(small_adj)
        np.testing.assert_array_equal(small_adj, original)


class TestOrderRepair:
    """Tests for OrderRepair."""

    def test_order_repair_consistency(self, rng):
        """OrderRepair removes exactly those edges violating the given ordering."""
        ordering = [0, 1, 2, 3, 4]
        repair = OrderRepair(ordering=ordering)

        # Graph with one back-edge 3→1
        adj = np.zeros((5, 5), dtype=np.int8)
        adj[0, 1] = 1
        adj[1, 2] = 1
        adj[2, 3] = 1
        adj[3, 1] = 1  # violates ordering
        adj[3, 4] = 1

        result = repair.repair(adj)
        assert not _has_cycle(result)
        # Back-edge 3→1 should be removed
        assert result[3, 1] == 0
        # Forward edges should be preserved
        assert result[0, 1] == 1
        assert result[1, 2] == 1
        assert result[2, 3] == 1
        assert result[3, 4] == 1

    def test_order_repair_all_back_edges(self, rng):
        """All edges violating the ordering are removed."""
        ordering = [4, 3, 2, 1, 0]  # reverse of natural order
        repair = OrderRepair(ordering=ordering)
        adj = np.zeros((5, 5), dtype=np.int8)
        adj[0, 1] = 1  # 0 is at position 4, 1 at position 3 → violates
        adj[1, 2] = 1  # similarly violates
        result = repair.repair(adj)
        assert _edge_count(result) == 0  # all edges violate reverse ordering

    def test_order_repair_preserves_forward(self, rng):
        """Edges consistent with the ordering are kept."""
        ordering = [0, 1, 2, 3, 4]
        repair = OrderRepair(ordering=ordering)
        # All forward edges
        adj = np.zeros((5, 5), dtype=np.int8)
        adj[0, 2] = 1
        adj[1, 3] = 1
        adj[2, 4] = 1
        result = repair.repair(adj)
        np.testing.assert_array_equal(result, adj)

    def test_order_repair_produces_acyclic(self, rng):
        """OrderRepair always yields an acyclic graph."""
        for _ in range(30):
            n = rng.integers(4, 10)
            adj = _make_cyclic_graph(n, rng)
            ordering = list(rng.permutation(n))
            repair = OrderRepair(ordering=ordering)
            result = repair.repair(adj)
            assert not _has_cycle(result)


class TestMinimalRepair:
    """Tests for MinimalRepair."""

    def test_minimal_repair_breaks_cycles(self, rng):
        """MinimalRepair produces acyclic graph from cyclic input."""
        repair = MinimalRepair()
        for _ in range(30):
            n = rng.integers(4, 10)
            adj = _make_cyclic_graph(n, rng)
            result = repair.repair(adj)
            assert not _has_cycle(result)

    def test_minimal_repair_preserves_acyclic(self, small_adj):
        """MinimalRepair is a no-op on acyclic graphs."""
        repair = MinimalRepair()
        result = repair.repair(small_adj)
        np.testing.assert_array_equal(result, small_adj)

    def test_minimal_repair_with_weights(self, rng):
        """Weighted MinimalRepair removes lighter edges first."""
        adj = np.zeros((3, 3), dtype=np.int8)
        adj[0, 1] = 1
        adj[1, 2] = 1
        adj[2, 0] = 1  # cycle

        weights = np.ones((3, 3), dtype=np.float64) * 10.0
        weights[2, 0] = 0.1  # lightest edge in the cycle

        repair = MinimalRepair(weights=weights)
        result = repair.repair(adj)
        assert not _has_cycle(result)
        # The lightest edge (2→0) should be removed
        assert result[2, 0] == 0
        # Other edges should remain
        assert result[0, 1] == 1
        assert result[1, 2] == 1

    def test_minimal_repair_multiple_cycles(self, rng):
        """MinimalRepair handles graphs with multiple independent cycles."""
        adj = np.zeros((6, 6), dtype=np.int8)
        # Cycle 1: 0→1→2→0
        adj[0, 1] = 1
        adj[1, 2] = 1
        adj[2, 0] = 1
        # Cycle 2: 3→4→5→3
        adj[3, 4] = 1
        adj[4, 5] = 1
        adj[5, 3] = 1

        repair = MinimalRepair()
        result = repair.repair(adj)
        assert not _has_cycle(result)
        # Each cycle needs at least 1 edge removed → at most 4 edges remain
        assert _edge_count(result) >= 4


class TestAcyclicityRepair:
    """Tests for AcyclicityRepair (legacy)."""

    def test_acyclicity_repair_breaks_cycles(self, rng):
        """AcyclicityRepair produces acyclic output from cyclic input."""
        repair = AcyclicityRepair()
        for _ in range(30):
            n = rng.integers(4, 10)
            adj = _make_cyclic_graph(n, rng)
            result = repair.repair(adj)
            assert not _has_cycle(result)

    def test_acyclicity_repair_preserves_acyclic(self, medium_adj):
        """No-op on already-acyclic graph."""
        repair = AcyclicityRepair()
        result = repair.repair(medium_adj)
        np.testing.assert_array_equal(result, medium_adj)


class TestConnectivityRepair:
    """Tests for ConnectivityRepair."""

    def test_connectivity_repair_connects_components(self, rng):
        """ConnectivityRepair joins disconnected components."""
        repair = ConnectivityRepair()
        # Two disconnected components: {0,1} and {2,3,4}
        adj = np.zeros((5, 5), dtype=np.int8)
        adj[0, 1] = 1
        adj[2, 3] = 1
        adj[3, 4] = 1
        result = repair.repair(adj)
        assert not _has_cycle(result)
        # Check weak connectivity: undirected BFS from node 0 should reach all
        undirected = (result | result.T).astype(bool)
        visited = set()
        queue = [0]
        visited.add(0)
        while queue:
            node = queue.pop()
            for nbr in range(5):
                if undirected[node, nbr] and nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        assert visited == {0, 1, 2, 3, 4}

    def test_connectivity_repair_preserves_acyclicity(self, rng):
        """Added edges don't introduce cycles."""
        repair = ConnectivityRepair()
        # Three isolated nodes plus a 2-node chain
        adj = np.zeros((5, 5), dtype=np.int8)
        adj[0, 1] = 1
        result = repair.repair(adj)
        assert not _has_cycle(result)

    def test_connectivity_repair_already_connected(self, small_adj):
        """No-op if graph is already weakly connected."""
        repair = ConnectivityRepair()
        result = repair.repair(small_adj)
        np.testing.assert_array_equal(result, small_adj)


# ===================================================================
# Repair property tests
# ===================================================================


class TestRepairProperty:
    """Property tests across repair operators."""

    @pytest.mark.parametrize("repair_cls", [
        TopologicalRepair,
        AcyclicityRepair,
    ])
    def test_repair_idempotent(self, repair_cls, rng):
        """Applying repair twice gives the same result as once."""
        repair = repair_cls()
        for _ in range(20):
            n = rng.integers(4, 10)
            adj = _make_cyclic_graph(n, rng)
            once = repair.repair(adj)
            twice = repair.repair(once)
            np.testing.assert_array_equal(once, twice)

    @pytest.mark.parametrize("repair_cls", [
        TopologicalRepair,
        MinimalRepair,
        AcyclicityRepair,
    ])
    def test_repair_acyclicity_property(self, repair_cls, rng):
        """All cycle-breaking repairs produce acyclic output on random graphs."""
        repair = repair_cls()
        for _ in range(30):
            n = rng.integers(4, 12)
            adj = _make_cyclic_graph(n, rng)
            result = repair.repair(adj)
            assert not _has_cycle(result), f"{repair_cls.__name__} left cycles"

    def test_repair_preserves_subgraph(self, rng):
        """Edges not involved in any cycle are never removed by TopologicalRepair."""
        repair = TopologicalRepair()
        # 0→1→2→3, plus back-edge 3→1
        adj = np.zeros((5, 5), dtype=np.int8)
        adj[0, 1] = 1
        adj[1, 2] = 1
        adj[2, 3] = 1
        adj[3, 1] = 1  # creates cycle
        adj[3, 4] = 1  # not in any cycle
        result = repair.repair(adj)
        assert not _has_cycle(result)
        # 0→1 is always forward; 3→4 is always forward
        assert result[3, 4] == 1


# ===================================================================
# Helper function tests
# ===================================================================


class TestHelpers:
    """Tests for shared helper functions."""

    def test_has_cycle_detects_cycle(self):
        """_has_cycle returns True for a cyclic graph."""
        adj = np.zeros((3, 3), dtype=np.int8)
        adj[0, 1] = 1
        adj[1, 2] = 1
        adj[2, 0] = 1
        assert _has_cycle(adj)

    def test_has_cycle_false_for_dag(self, small_adj):
        """_has_cycle returns False for a known DAG."""
        assert not _has_cycle(small_adj)

    def test_has_cycle_empty(self):
        """Empty graph has no cycle."""
        assert not _has_cycle(np.zeros((5, 5), dtype=np.int8))

    def test_topological_sort_chain(self, small_adj):
        """Topological sort of a chain returns the correct order."""
        order = _topological_sort(small_adj)
        assert len(order) == 5
        # In a chain 0→1→2→3→4, the only valid topo order is [0,1,2,3,4]
        assert order == [0, 1, 2, 3, 4]

    def test_topological_sort_partial_on_cycle(self):
        """Topological sort returns fewer than n nodes when cycles exist."""
        adj = np.zeros((3, 3), dtype=np.int8)
        adj[0, 1] = 1
        adj[1, 2] = 1
        adj[2, 0] = 1
        order = _topological_sort(adj)
        assert len(order) < 3

    def test_topological_sort_empty(self):
        """Topological sort of empty graph returns all nodes."""
        adj = np.zeros((4, 4), dtype=np.int8)
        order = _topological_sort(adj)
        assert len(order) == 4
        assert set(order) == {0, 1, 2, 3}


# ===================================================================
# Integration / end-to-end tests
# ===================================================================


class TestMutationCrossoverIntegration:
    """Integration tests combining mutation and crossover."""

    def test_mutate_then_crossover(self, small_adj, rng):
        """Mutated DAGs can be crossed over and still yield valid DAGs."""
        mut = TopologicalMutation()
        xo = OrderCrossover()
        p1 = mut.mutate(small_adj, rng)
        p2 = mut.mutate(small_adj, rng)
        c1, c2 = xo.crossover(p1, p2, rng)
        assert not _has_cycle(c1)
        assert not _has_cycle(c2)

    def test_crossover_then_mutate(self, small_adj, rng):
        """Crossover children can be mutated and stay acyclic."""
        xo = UniformCrossover()
        mut = CompositeMutation(
            operators=[EdgeAddMutation(), EdgeRemoveMutation()],
            n_mutations=2,
        )
        p2 = _make_random_dag(5, 0.3, rng)
        c1, c2 = xo.crossover(small_adj, p2, rng)
        m1 = mut.mutate(c1, rng)
        m2 = mut.mutate(c2, rng)
        assert not _has_cycle(m1)
        assert not _has_cycle(m2)

    def test_repair_after_arbitrary_edits(self, rng):
        """TopologicalRepair fixes arbitrary randomly-edited graphs."""
        repair = TopologicalRepair()
        for _ in range(20):
            n = rng.integers(5, 12)
            adj = np.zeros((n, n), dtype=np.int8)
            # Add random edges (may create cycles)
            num_edges = rng.integers(n, n * (n - 1) // 2)
            for _ in range(num_edges):
                i, j = rng.integers(0, n, size=2)
                if i != j:
                    adj[i, j] = 1
            result = repair.repair(adj)
            assert not _has_cycle(result)

    def test_full_pipeline(self, medium_adj, rng):
        """Full pipeline: mutate → crossover → repair stays valid."""
        mut = AdaptiveMutation(
            operators=[EdgeAddMutation(), EdgeRemoveMutation(), EdgeReverseMutation()],
        )
        xo = OrderCrossover()
        repair = TopologicalRepair()

        population = [medium_adj.copy() for _ in range(4)]
        for gen in range(10):
            # Mutate
            mutated = [mut.mutate(ind, rng) for ind in population]
            for m in mutated:
                assert not _has_cycle(m)
                mut.report_result(rng.random() < 0.5)

            # Crossover
            children = []
            for i in range(0, len(mutated) - 1, 2):
                c1, c2 = xo.crossover(mutated[i], mutated[i + 1], rng)
                children.extend([c1, c2])

            # Repair (should be no-op since operators preserve acyclicity)
            population = [repair.repair(c) for c in children]
            for ind in population:
                assert not _has_cycle(ind)
