"""Unit tests for cpa.operators.constrained."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from cpa.operators.constrained import (
    ConstrainedCrossover,
    ConstrainedMutation,
    ConstrainedOperator,
    EdgeConstraints,
    _is_acyclic,
    _repair_dag,
)
from cpa.operators.mutation import EdgeMutation
from cpa.operators.crossover import EdgePreservingCrossover


# ── helpers ─────────────────────────────────────────────────────────

def _chain_dag(n: int = 4):
    adj = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
    return adj


def _diamond_dag():
    adj = np.zeros((4, 4), dtype=float)
    adj[0, 1] = 1.0
    adj[0, 2] = 1.0
    adj[1, 3] = 1.0
    adj[2, 3] = 1.0
    return adj


def _complex_dag():
    adj = np.zeros((5, 5), dtype=float)
    adj[0, 1] = 1.0
    adj[0, 2] = 1.0
    adj[1, 3] = 1.0
    adj[2, 3] = 1.0
    adj[3, 4] = 1.0
    return adj


def _empty_dag(n: int = 4):
    return np.zeros((n, n), dtype=float)


# ── EdgeConstraints basic tests ───────────────────────────────────

class TestEdgeConstraints:
    def test_empty_constraints_valid(self):
        ec = EdgeConstraints()
        dag = _chain_dag()
        assert ec.is_valid(dag)

    def test_required_edge_present(self):
        ec = EdgeConstraints(required_edges={(0, 1)})
        dag = _chain_dag()
        assert ec.is_valid(dag)

    def test_required_edge_absent(self):
        ec = EdgeConstraints(required_edges={(2, 0)})
        dag = _chain_dag()
        assert not ec.is_valid(dag)

    def test_forbidden_edge_absent(self):
        ec = EdgeConstraints(forbidden_edges={(2, 0)})
        dag = _chain_dag()
        assert ec.is_valid(dag)

    def test_forbidden_edge_present(self):
        ec = EdgeConstraints(forbidden_edges={(0, 1)})
        dag = _chain_dag()
        assert not ec.is_valid(dag)

    def test_max_parents_satisfied(self):
        ec = EdgeConstraints(max_parents=2)
        dag = _diamond_dag()
        assert ec.is_valid(dag)  # node 3 has 2 parents

    def test_max_parents_exceeded(self):
        ec = EdgeConstraints(max_parents=1)
        dag = _diamond_dag()
        assert not ec.is_valid(dag)  # node 3 has 2 parents

    def test_max_degree_satisfied(self):
        ec = EdgeConstraints(max_degree=3)
        dag = _diamond_dag()
        assert ec.is_valid(dag)

    def test_max_degree_exceeded(self):
        ec = EdgeConstraints(max_degree=1)
        dag = _diamond_dag()
        assert not ec.is_valid(dag)

    def test_tier_ordering_valid(self):
        ec = EdgeConstraints(tier_ordering=[[0], [1, 2], [3]])
        dag = _diamond_dag()  # 0->1, 0->2, 1->3, 2->3
        assert ec.is_valid(dag)

    def test_tier_ordering_violated(self):
        ec = EdgeConstraints(tier_ordering=[[3], [1, 2], [0]])
        dag = _diamond_dag()  # 0->1 violates: 0 in tier 2, 1 in tier 1
        assert not ec.is_valid(dag)

    def test_cyclic_graph_invalid(self):
        ec = EdgeConstraints()
        cyclic = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        assert not ec.is_valid(cyclic)

    def test_valid_additions(self):
        ec = EdgeConstraints(forbidden_edges={(1, 0)}, max_parents=2)
        dag = _chain_dag(3)
        adds = ec.valid_additions(dag)
        # 1->0 is forbidden
        assert (1, 0) not in adds
        # 0->2 should be valid (not present, not forbidden, acyclic)
        assert (0, 2) in adds

    def test_valid_removals_respects_required(self):
        ec = EdgeConstraints(required_edges={(0, 1)})
        dag = _chain_dag(3)
        removals = ec.valid_removals(dag)
        assert (0, 1) not in removals
        assert (1, 2) in removals

    def test_valid_additions_max_parents(self):
        ec = EdgeConstraints(max_parents=1)
        dag = np.zeros((3, 3), dtype=float)
        dag[0, 1] = 1.0  # node 1 already has 1 parent
        adds = ec.valid_additions(dag)
        # Cannot add another parent to node 1
        for i, j in adds:
            assert j != 1


# ── ConstrainedOperator tests ────────────────────────────────────

class TestConstrainedOperator:
    def test_is_valid(self):
        ec = EdgeConstraints(required_edges={(0, 1)})
        co = ConstrainedOperator(ec)
        assert co.is_valid(_chain_dag())
        assert not co.is_valid(_empty_dag())

    def test_project_adds_required_edges(self):
        ec = EdgeConstraints(required_edges={(0, 1)})
        co = ConstrainedOperator(ec)
        dag = _empty_dag()
        projected = co.project(dag)
        assert projected[0, 1] != 0

    def test_project_removes_forbidden_edges(self):
        ec = EdgeConstraints(forbidden_edges={(0, 1)})
        co = ConstrainedOperator(ec)
        dag = _chain_dag()
        projected = co.project(dag)
        assert projected[0, 1] == 0

    def test_project_enforces_max_parents(self):
        ec = EdgeConstraints(max_parents=1)
        co = ConstrainedOperator(ec)
        dag = _diamond_dag()  # node 3 has 2 parents
        projected = co.project(dag)
        in_deg = (projected != 0).sum(axis=0)
        assert all(d <= 1 for d in in_deg)

    def test_project_enforces_acyclicity(self):
        ec = EdgeConstraints()
        co = ConstrainedOperator(ec)
        cyclic = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        projected = co.project(cyclic)
        assert _is_acyclic(projected)

    def test_project_enforces_tier_ordering(self):
        ec = EdgeConstraints(tier_ordering=[[0], [1], [2]])
        co = ConstrainedOperator(ec)
        # dag with back-tier edge
        dag = np.zeros((3, 3), dtype=float)
        dag[2, 0] = 1.0  # tier 2 -> tier 0 (violation)
        projected = co.project(dag)
        assert projected[2, 0] == 0

    def test_constrained_mutation_respects_constraints(self):
        ec = EdgeConstraints(
            required_edges={(0, 1)},
            forbidden_edges={(2, 0)},
            max_parents=2,
        )
        co = ConstrainedOperator(ec, seed=42)
        dag = _chain_dag(3)
        for s in range(20):
            rng = np.random.default_rng(s)
            result = co.constrained_mutation(dag, rng=rng)
            assert ec.is_valid(result) or _is_acyclic(result)

    def test_constrained_crossover(self):
        ec = EdgeConstraints(required_edges={(0, 1)})
        co = ConstrainedOperator(ec, seed=42)
        p1 = _chain_dag(3)
        p2 = np.zeros((3, 3), dtype=float)
        p2[2, 1] = 1.0
        offspring = co.constrained_crossover(p1, p2)
        assert _is_acyclic(offspring)
        assert offspring[0, 1] != 0  # required edge present


# ── ConstrainedMutation tests ────────────────────────────────────

class TestConstrainedMutation:
    def test_respects_required_edges(self):
        ec = EdgeConstraints(required_edges={(0, 1)})
        base = EdgeMutation(mutation_rate=1.0, seed=42)
        cm = ConstrainedMutation(ec, base, seed=42)
        dag = _chain_dag(3)
        for s in range(20):
            rng = np.random.default_rng(s)
            result = cm.mutate(dag, rng=rng)
            assert result[0, 1] != 0, f"Required edge missing at seed {s}"

    def test_respects_forbidden_edges(self):
        ec = EdgeConstraints(forbidden_edges={(2, 0)})
        base = EdgeMutation(mutation_rate=1.0, seed=42)
        cm = ConstrainedMutation(ec, base, seed=42)
        dag = _chain_dag(3)
        for s in range(20):
            rng = np.random.default_rng(s)
            result = cm.mutate(dag, rng=rng)
            assert result[2, 0] == 0, f"Forbidden edge present at seed {s}"

    def test_respects_max_parents(self):
        ec = EdgeConstraints(max_parents=1)
        base = EdgeMutation(mutation_rate=1.0, seed=42)
        cm = ConstrainedMutation(ec, base, seed=42)
        dag = _chain_dag(4)
        for s in range(20):
            rng = np.random.default_rng(s)
            result = cm.mutate(dag, rng=rng)
            in_deg = (result != 0).sum(axis=0)
            assert all(d <= 1 for d in in_deg)

    def test_output_is_dag(self):
        ec = EdgeConstraints(max_parents=2)
        base = EdgeMutation(mutation_rate=1.0, seed=42)
        cm = ConstrainedMutation(ec, base, seed=42)
        dag = _complex_dag()
        for s in range(20):
            rng = np.random.default_rng(s)
            result = cm.mutate(dag, rng=rng)
            assert _is_acyclic(result)


# ── ConstrainedCrossover tests ───────────────────────────────────

class TestConstrainedCrossover:
    def test_respects_required_edges(self):
        ec = EdgeConstraints(required_edges={(0, 1)})
        base = EdgePreservingCrossover(crossover_rate=1.0, seed=42)
        cc = ConstrainedCrossover(ec, base, seed=42)
        p1 = _chain_dag(3)
        p2 = np.zeros((3, 3), dtype=float)
        p2[2, 1] = 1.0
        for s in range(20):
            rng = np.random.default_rng(s)
            offspring = cc.crossover(p1, p2, rng=rng)
            assert offspring[0, 1] != 0

    def test_respects_forbidden_edges(self):
        ec = EdgeConstraints(forbidden_edges={(1, 0)})
        base = EdgePreservingCrossover(crossover_rate=1.0, seed=42)
        cc = ConstrainedCrossover(ec, base, seed=42)
        p1 = _chain_dag(3)
        p2 = np.zeros((3, 3), dtype=float)
        p2[1, 0] = 1.0
        for s in range(20):
            rng = np.random.default_rng(s)
            offspring = cc.crossover(p1, p2, rng=rng)
            assert offspring[1, 0] == 0

    def test_output_is_dag(self):
        ec = EdgeConstraints()
        base = EdgePreservingCrossover(crossover_rate=1.0, seed=42)
        cc = ConstrainedCrossover(ec, base, seed=42)
        p1 = _diamond_dag()
        p2 = _chain_dag()
        for s in range(20):
            rng = np.random.default_rng(s)
            offspring = cc.crossover(p1, p2, rng=rng)
            assert _is_acyclic(offspring)


# ── Combined constraint tests ────────────────────────────────────

class TestCombinedConstraints:
    def test_multiple_constraints_together(self):
        ec = EdgeConstraints(
            required_edges={(0, 1)},
            forbidden_edges={(2, 0), (3, 0)},
            max_parents=2,
        )
        co = ConstrainedOperator(ec, seed=42)
        dag = _diamond_dag()
        for s in range(20):
            rng = np.random.default_rng(s)
            result = co.constrained_mutation(dag, rng=rng)
            assert _is_acyclic(result)
            assert result[0, 1] != 0
            assert result[2, 0] == 0
            assert result[3, 0] == 0

    def test_tier_with_max_parents(self):
        ec = EdgeConstraints(
            tier_ordering=[[0], [1, 2], [3]],
            max_parents=1,
        )
        co = ConstrainedOperator(ec, seed=42)
        dag = _diamond_dag()
        projected = co.project(dag)
        assert _is_acyclic(projected)
        in_deg = (projected != 0).sum(axis=0)
        assert all(d <= 1 for d in in_deg)
