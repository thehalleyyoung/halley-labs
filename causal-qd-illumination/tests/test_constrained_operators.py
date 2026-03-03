"""Tests for constraint-aware mutation and crossover operators."""

from __future__ import annotations

import numpy as np
import pytest

from causal_qd.operators.constrained import (
    ConstrainedCrossover,
    ConstrainedMutation,
    EdgeConstraints,
    TierConstraints,
    _repair,
)
from causal_qd.operators.mutation import (
    EdgeAddMutation,
    EdgeRemoveMutation,
    TopologicalMutation,
    _has_cycle,
)
from causal_qd.operators.crossover import OrderCrossover, UniformCrossover
from causal_qd.types import AdjacencyMatrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dag(*edges: tuple[int, int], n: int = 4) -> AdjacencyMatrix:
    """Create a DAG adjacency matrix from edges."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i, j in edges:
        adj[i, j] = 1
    return adj


def _dag_edges(adj: AdjacencyMatrix) -> set[tuple[int, int]]:
    """Extract edges as a set of (i, j) tuples."""
    rows, cols = np.nonzero(adj)
    return set(zip(rows.tolist(), cols.tolist()))


# ---------------------------------------------------------------------------
# EdgeConstraints validation
# ---------------------------------------------------------------------------

class TestEdgeConstraintsValidation:

    def test_no_conflict(self):
        ec = EdgeConstraints(
            forbidden_edges=frozenset([(0, 1)]),
            required_edges=frozenset([(1, 2)]),
        )
        ec.validate()  # should not raise

    def test_conflict_raises(self):
        ec = EdgeConstraints(
            forbidden_edges=frozenset([(0, 1)]),
            required_edges=frozenset([(0, 1)]),
        )
        with pytest.raises(ValueError, match="both required and forbidden"):
            ec.validate()

    def test_tier_conflict_with_required(self):
        ec = EdgeConstraints(
            required_edges=frozenset([(2, 0)]),
            tier_ordering=[{0, 1}, {2, 3}],  # 0 in tier 0, 2 in tier 1
        )
        with pytest.raises(ValueError, match="tier ordering"):
            ec.validate()

    def test_empty_constraints_valid(self):
        ec = EdgeConstraints()
        ec.validate()


# ---------------------------------------------------------------------------
# is_valid_edge / is_valid_dag
# ---------------------------------------------------------------------------

class TestEdgeConstraintsChecks:

    def test_forbidden_edge_invalid(self):
        dag = _make_dag((0, 1))
        ec = EdgeConstraints(forbidden_edges=frozenset([(0, 2)]))
        assert not ec.is_valid_edge(0, 2, dag)
        assert ec.is_valid_edge(0, 1, dag)

    def test_max_parents_exceeded(self):
        dag = _make_dag((0, 2), (1, 2))
        ec = EdgeConstraints(max_parents=3)
        assert ec.is_valid_edge(3, 2, dag)  # current=2, limit=3 → OK
        ec2 = EdgeConstraints(max_parents=2)
        assert not ec2.is_valid_edge(3, 2, dag)  # current=2, limit=2 → rejected

    def test_tier_ordering_check(self):
        dag = _make_dag(n=4)
        ec = EdgeConstraints(tier_ordering=[{0, 1}, {2, 3}])
        assert ec.is_valid_edge(0, 2, dag)  # tier 0 -> tier 1
        assert not ec.is_valid_edge(2, 0, dag)  # tier 1 -> tier 0

    def test_is_valid_dag_all_pass(self):
        dag = _make_dag((0, 1), (1, 2))
        ec = EdgeConstraints(
            forbidden_edges=frozenset([(2, 0)]),
            required_edges=frozenset([(0, 1)]),
            max_parents=2,
        )
        assert ec.is_valid_dag(dag)

    def test_is_valid_dag_forbidden_fail(self):
        dag = _make_dag((0, 1), (1, 2))
        ec = EdgeConstraints(forbidden_edges=frozenset([(0, 1)]))
        assert not ec.is_valid_dag(dag)

    def test_is_valid_dag_required_missing(self):
        dag = _make_dag((0, 1))
        ec = EdgeConstraints(required_edges=frozenset([(1, 2)]))
        assert not ec.is_valid_dag(dag)

    def test_is_valid_dag_max_parents_fail(self):
        dag = _make_dag((0, 3), (1, 3), (2, 3))
        ec = EdgeConstraints(max_parents=2)
        assert not ec.is_valid_dag(dag)


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

class TestRepair:

    def test_removes_forbidden_edges(self):
        dag = _make_dag((0, 1), (1, 2), (0, 2))
        ec = EdgeConstraints(forbidden_edges=frozenset([(0, 2)]))
        repaired = _repair(dag, ec)
        assert repaired[0, 2] == 0
        assert repaired[0, 1] == 1
        assert repaired[1, 2] == 1

    def test_adds_required_edges(self):
        dag = _make_dag((0, 1))
        ec = EdgeConstraints(required_edges=frozenset([(1, 2)]))
        repaired = _repair(dag, ec)
        assert repaired[1, 2] == 1
        assert not _has_cycle(repaired)

    def test_enforces_max_parents(self):
        dag = _make_dag((0, 3), (1, 3), (2, 3))
        ec = EdgeConstraints(max_parents=2)
        repaired = _repair(dag, ec)
        assert int(repaired[:, 3].sum()) <= 2
        assert not _has_cycle(repaired)

    def test_removes_tier_violating_edges(self):
        # edge 2->0 goes from tier 1 back to tier 0
        dag = _make_dag((0, 2), (2, 0))
        ec = EdgeConstraints(tier_ordering=[{0, 1}, {2, 3}])
        repaired = _repair(dag, ec)
        assert repaired[2, 0] == 0
        assert repaired[0, 2] == 1

    def test_repair_preserves_acyclicity(self):
        dag = _make_dag((0, 1), (1, 2))
        ec = EdgeConstraints(
            forbidden_edges=frozenset([(0, 1)]),
            required_edges=frozenset([(0, 2)]),
        )
        repaired = _repair(dag, ec)
        assert not _has_cycle(repaired)

    def test_repair_noop_when_valid(self):
        dag = _make_dag((0, 1), (1, 2))
        ec = EdgeConstraints()
        repaired = _repair(dag, ec)
        np.testing.assert_array_equal(repaired, dag)


# ---------------------------------------------------------------------------
# ConstrainedMutation
# ---------------------------------------------------------------------------

class TestConstrainedMutation:

    def test_forbidden_edges_never_present(self):
        dag = _make_dag((0, 1), (1, 2), (2, 3))
        ec = EdgeConstraints(forbidden_edges=frozenset([(0, 3), (3, 0)]))
        cm = ConstrainedMutation(TopologicalMutation(), ec)
        rng = np.random.default_rng(42)
        for _ in range(50):
            result = cm.mutate(dag, rng)
            assert result[0, 3] == 0
            assert result[3, 0] == 0
            assert not _has_cycle(result)

    def test_required_edges_always_present(self):
        dag = _make_dag((0, 1), (1, 2))
        ec = EdgeConstraints(required_edges=frozenset([(0, 1)]))
        cm = ConstrainedMutation(EdgeRemoveMutation(), ec)
        rng = np.random.default_rng(123)
        for _ in range(50):
            result = cm.mutate(dag, rng)
            assert result[0, 1] == 1
            assert not _has_cycle(result)

    def test_max_parents_enforced(self):
        dag = _make_dag((0, 2), (1, 2), n=5)
        ec = EdgeConstraints(max_parents=2)
        cm = ConstrainedMutation(EdgeAddMutation(), ec)
        rng = np.random.default_rng(7)
        for _ in range(50):
            result = cm.mutate(dag, rng)
            for j in range(5):
                assert int(result[:, j].sum()) <= 2
            assert not _has_cycle(result)

    def test_no_constraints_behaves_like_unconstrained(self):
        dag = _make_dag((0, 1), (1, 2))
        ec = EdgeConstraints()
        inner = TopologicalMutation()
        cm = ConstrainedMutation(inner, ec)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        result_constrained = cm.mutate(dag, rng1)
        result_plain = inner.mutate(dag, rng2)
        np.testing.assert_array_equal(result_constrained, result_plain)

    def test_tier_ordering_enforced(self):
        dag = _make_dag((0, 2), n=4)
        tiers = [{0, 1}, {2, 3}]
        ec = EdgeConstraints(tier_ordering=tiers)
        cm = ConstrainedMutation(TopologicalMutation(), ec)
        rng = np.random.default_rng(42)
        tc = TierConstraints(tiers)
        for _ in range(50):
            result = cm.mutate(dag, rng)
            for i in range(4):
                for j in range(4):
                    if result[i, j]:
                        assert tc.is_valid_edge(i, j)


# ---------------------------------------------------------------------------
# ConstrainedCrossover
# ---------------------------------------------------------------------------

class TestConstrainedCrossover:

    def test_produces_valid_dags(self):
        p1 = _make_dag((0, 1), (1, 2), (2, 3))
        p2 = _make_dag((0, 2), (1, 3))
        ec = EdgeConstraints(
            forbidden_edges=frozenset([(3, 0)]),
            max_parents=2,
        )
        cc = ConstrainedCrossover(OrderCrossover(), ec)
        rng = np.random.default_rng(42)
        for _ in range(20):
            c1, c2 = cc.crossover(p1, p2, rng)
            assert not _has_cycle(c1)
            assert not _has_cycle(c2)
            assert ec.is_valid_dag(c1)
            assert ec.is_valid_dag(c2)

    def test_forbidden_edges_respected(self):
        p1 = _make_dag((0, 1), (1, 2))
        p2 = _make_dag((0, 2), (2, 3))
        ec = EdgeConstraints(forbidden_edges=frozenset([(0, 2), (1, 3)]))
        cc = ConstrainedCrossover(UniformCrossover(), ec)
        rng = np.random.default_rng(77)
        for _ in range(20):
            c1, c2 = cc.crossover(p1, p2, rng)
            assert c1[0, 2] == 0 and c1[1, 3] == 0
            assert c2[0, 2] == 0 and c2[1, 3] == 0

    def test_required_edges_present(self):
        p1 = _make_dag((0, 1), (1, 2))
        p2 = _make_dag((0, 2), (2, 3))
        ec = EdgeConstraints(required_edges=frozenset([(0, 1)]))
        cc = ConstrainedCrossover(UniformCrossover(), ec)
        rng = np.random.default_rng(55)
        for _ in range(20):
            c1, c2 = cc.crossover(p1, p2, rng)
            assert c1[0, 1] == 1
            assert c2[0, 1] == 1


# ---------------------------------------------------------------------------
# TierConstraints
# ---------------------------------------------------------------------------

class TestTierConstraints:

    def test_valid_forward_edge(self):
        tc = TierConstraints([{0, 1}, {2, 3}])
        assert tc.is_valid_edge(0, 2)
        assert tc.is_valid_edge(1, 3)

    def test_invalid_backward_edge(self):
        tc = TierConstraints([{0, 1}, {2, 3}])
        assert not tc.is_valid_edge(2, 0)
        assert not tc.is_valid_edge(3, 1)

    def test_same_tier_invalid(self):
        tc = TierConstraints([{0, 1}, {2, 3}])
        assert not tc.is_valid_edge(0, 1)
        assert not tc.is_valid_edge(2, 3)

    def test_to_edge_constraints(self):
        tc = TierConstraints([{0}, {1}])
        ec = tc.to_edge_constraints()
        assert (1, 0) in ec.forbidden_edges
        assert (0, 1) not in ec.forbidden_edges


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_node(self):
        dag = np.zeros((1, 1), dtype=np.int8)
        ec = EdgeConstraints()
        assert ec.is_valid_dag(dag)
        repaired = _repair(dag, ec)
        np.testing.assert_array_equal(repaired, dag)

    def test_empty_graph(self):
        dag = np.zeros((4, 4), dtype=np.int8)
        ec = EdgeConstraints(
            forbidden_edges=frozenset([(0, 1)]),
            max_parents=1,
        )
        assert ec.is_valid_dag(dag)

    def test_complete_dag_with_max_parents_1(self):
        # Chain 0->1->2->3 is a complete DAG path with max_parents=1
        dag = _make_dag((0, 1), (1, 2), (2, 3), (0, 2), (0, 3), (1, 3))
        ec = EdgeConstraints(max_parents=1)
        repaired = _repair(dag, ec)
        for j in range(4):
            assert int(repaired[:, j].sum()) <= 1
        assert not _has_cycle(repaired)

    def test_constrained_mutation_single_node(self):
        dag = np.zeros((1, 1), dtype=np.int8)
        ec = EdgeConstraints()
        cm = ConstrainedMutation(TopologicalMutation(), ec)
        rng = np.random.default_rng(0)
        result = cm.mutate(dag, rng)
        np.testing.assert_array_equal(result, dag)

    def test_all_edges_forbidden(self):
        n = 3
        forbidden = frozenset((i, j) for i in range(n) for j in range(n) if i != j)
        ec = EdgeConstraints(forbidden_edges=forbidden)
        dag = _make_dag((0, 1), (1, 2), n=3)
        repaired = _repair(dag, ec)
        assert repaired.sum() == 0
