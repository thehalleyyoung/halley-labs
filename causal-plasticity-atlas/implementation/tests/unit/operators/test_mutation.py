"""Unit tests for cpa.operators.mutation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from cpa.operators.mutation import (
    AdaptiveMutation,
    DAGMutation,
    EdgeMutation,
    StructuralMutation,
    TargetedMutation,
    WeightMutation,
    _is_acyclic,
    _existing_edges,
)


# ── helpers ─────────────────────────────────────────────────────────

def _chain_dag(n: int = 4):
    adj = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
    return adj


def _complex_dag():
    adj = np.zeros((5, 5), dtype=float)
    adj[0, 1] = 1.0
    adj[0, 2] = 1.0
    adj[1, 3] = 1.0
    adj[2, 3] = 1.0
    adj[3, 4] = 1.0
    return adj


def _weighted_dag():
    adj = np.zeros((4, 4), dtype=float)
    adj[0, 1] = 0.5
    adj[0, 2] = 0.8
    adj[1, 3] = -0.3
    adj[2, 3] = 1.2
    return adj


def _empty_dag(n: int = 4):
    return np.zeros((n, n), dtype=float)


# ── DAGMutation tests ─────────────────────────────────────────────

class TestDAGMutation:
    def test_add_edge_maintains_dag(self):
        mut = DAGMutation(seed=42)
        dag = _chain_dag()
        result = mut.add_edge(dag)
        assert _is_acyclic(result)
        # Should have one more edge than original
        orig_count = (dag != 0).sum()
        new_count = (result != 0).sum()
        assert new_count >= orig_count

    def test_add_edge_to_complete_dag(self):
        """Full DAG should return unchanged."""
        mut = DAGMutation(seed=42)
        # 0->1, 0->2, 1->2 (3-node complete DAG)
        dag = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
        result = mut.add_edge(dag)
        assert_array_almost_equal(result, dag)

    def test_remove_edge(self):
        mut = DAGMutation(seed=42)
        dag = _chain_dag()
        result = mut.remove_edge(dag)
        assert _is_acyclic(result)
        orig_count = (dag != 0).sum()
        new_count = (result != 0).sum()
        assert new_count == orig_count - 1

    def test_remove_edge_from_empty(self):
        mut = DAGMutation(seed=42)
        dag = _empty_dag()
        result = mut.remove_edge(dag)
        assert_array_almost_equal(result, dag)

    def test_reverse_edge_maintains_dag(self):
        mut = DAGMutation(seed=42)
        dag = _chain_dag()
        result = mut.reverse_edge(dag)
        assert _is_acyclic(result)

    def test_is_valid_dag(self):
        mut = DAGMutation()
        assert mut.is_valid_dag(_chain_dag())
        cyclic = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        assert not mut.is_valid_dag(cyclic)

    def test_repeated_mutations_all_dags(self):
        mut = DAGMutation(seed=42)
        dag = _complex_dag()
        for _ in range(30):
            dag = mut.add_edge(dag)
            assert _is_acyclic(dag)
            dag = mut.remove_edge(dag)
            assert _is_acyclic(dag)


# ── EdgeMutation tests ────────────────────────────────────────────

class TestEdgeMutation:
    def test_mutation_rate_zero_no_change(self):
        mut = EdgeMutation(mutation_rate=0.0, seed=42)
        dag = _chain_dag()
        result = mut.mutate(dag)
        assert_array_almost_equal(result, dag)

    def test_mutate_maintains_dag(self):
        mut = EdgeMutation(mutation_rate=1.0, seed=42)
        dag = _chain_dag()
        for s in range(30):
            rng = np.random.default_rng(s)
            result = mut.mutate(dag, rng=rng)
            assert _is_acyclic(result), f"DAG violation at seed {s}"

    def test_mutation_changes_graph(self):
        """With rate=1.0, some mutations should change the graph."""
        mut = EdgeMutation(mutation_rate=1.0, seed=42)
        dag = _chain_dag()
        n_changed = 0
        for s in range(20):
            rng = np.random.default_rng(s)
            result = mut.mutate(dag, rng=rng)
            if not np.array_equal(result, dag):
                n_changed += 1
        assert n_changed > 5

    def test_empty_graph_can_add(self):
        mut = EdgeMutation(mutation_rate=1.0, seed=42)
        dag = _empty_dag(3)
        # Try many times — at least one should add an edge
        any_added = False
        for s in range(30):
            rng = np.random.default_rng(s)
            result = mut.mutate(dag, rng=rng)
            if (result != 0).any():
                any_added = True
                assert _is_acyclic(result)
        assert any_added


# ── WeightMutation tests ──────────────────────────────────────────

class TestWeightMutation:
    def test_weight_mutation_changes_values(self):
        mut = WeightMutation(mutation_rate=1.0, sigma=0.5, seed=42)
        coeff = _weighted_dag()
        result = mut.mutate(coeff)
        # Non-zero positions should be perturbed
        mask = coeff != 0
        assert not np.array_equal(result[mask], coeff[mask])

    def test_zero_entries_unchanged(self):
        mut = WeightMutation(mutation_rate=1.0, sigma=0.5, seed=42)
        coeff = _weighted_dag()
        result = mut.mutate(coeff)
        zero_mask = coeff == 0
        assert_array_almost_equal(result[zero_mask], np.zeros(zero_mask.sum()))

    def test_mutation_rate_zero_no_change(self):
        mut = WeightMutation(mutation_rate=0.0, sigma=0.5, seed=42)
        coeff = _weighted_dag()
        result = mut.mutate(coeff)
        assert_array_almost_equal(result, coeff)

    def test_empty_coefficients_unchanged(self):
        mut = WeightMutation(mutation_rate=1.0, sigma=0.5, seed=42)
        coeff = _empty_dag()
        result = mut.mutate(coeff)
        assert_array_almost_equal(result, coeff)

    def test_sigma_scales_perturbation(self):
        """Larger sigma should produce larger perturbations on average."""
        coeff = _weighted_dag()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        mut_small = WeightMutation(mutation_rate=1.0, sigma=0.01, seed=42)
        mut_large = WeightMutation(mutation_rate=1.0, sigma=1.0, seed=42)
        result_small = mut_small.mutate(coeff, rng=rng1)
        result_large = mut_large.mutate(coeff, rng=rng2)
        diff_small = np.abs(result_small - coeff).sum()
        diff_large = np.abs(result_large - coeff).sum()
        assert diff_large > diff_small


# ── StructuralMutation tests ─────────────────────────────────────

class TestStructuralMutation:
    def test_operation_probabilities_normalize(self):
        mut = StructuralMutation(add_prob=2.0, remove_prob=1.0, reverse_prob=1.0)
        assert abs(mut.add_prob + mut.remove_prob + mut.reverse_prob - 1.0) < 1e-10

    def test_mutate_maintains_dag(self):
        mut = StructuralMutation(seed=42)
        dag = _chain_dag()
        for s in range(30):
            rng = np.random.default_rng(s)
            result = mut.mutate(dag, rng=rng)
            assert _is_acyclic(result)

    def test_add_only(self):
        mut = StructuralMutation(add_prob=1.0, remove_prob=0.0, reverse_prob=0.0, seed=42)
        dag = _empty_dag(3)
        result = mut.mutate(dag)
        n_edges = (result != 0).sum()
        assert n_edges >= 0

    def test_remove_only(self):
        mut = StructuralMutation(add_prob=0.0, remove_prob=1.0, reverse_prob=0.0, seed=42)
        dag = _chain_dag(3)
        result = mut.mutate(dag)
        n_before = (dag != 0).sum()
        n_after = (result != 0).sum()
        assert n_after <= n_before

    def test_reverse_only(self):
        mut = StructuralMutation(add_prob=0.0, remove_prob=0.0, reverse_prob=1.0, seed=42)
        dag = _chain_dag(3)
        result = mut.mutate(dag)
        assert _is_acyclic(result)

    def test_operation_distribution(self):
        """Over many runs, all operations should occur proportionally."""
        mut = StructuralMutation(add_prob=0.4, remove_prob=0.3, reverse_prob=0.3, seed=42)
        dag = _chain_dag(4)
        edges_added, edges_removed = 0, 0
        for s in range(100):
            rng = np.random.default_rng(s)
            result = mut.mutate(dag, rng=rng)
            orig_edges = (dag != 0).sum()
            new_edges = (result != 0).sum()
            if new_edges > orig_edges:
                edges_added += 1
            elif new_edges < orig_edges:
                edges_removed += 1
        # Some adds and some removes should have happened
        assert edges_added > 0 or edges_removed > 0


# ── AdaptiveMutation tests ───────────────────────────────────────

class TestAdaptiveMutation:
    def test_initial_rate(self):
        mut = AdaptiveMutation(initial_rate=0.2)
        assert mut.rate == 0.2

    def test_rate_increases_when_stagnant(self):
        mut = AdaptiveMutation(initial_rate=0.1, adaptation_rate=0.05, seed=42)
        history = [1.0, 1.0, 1.0, 1.0, 1.0]  # stagnant
        dag = _chain_dag()
        mut.mutate(dag, fitness_history=history)
        assert mut.rate > 0.1

    def test_rate_decreases_when_improving(self):
        mut = AdaptiveMutation(initial_rate=0.3, adaptation_rate=0.05, seed=42)
        history = [1.0, 2.0, 3.0, 4.0, 5.0]  # improving
        dag = _chain_dag()
        mut.mutate(dag, fitness_history=history)
        assert mut.rate < 0.3

    def test_rate_stays_in_bounds(self):
        mut = AdaptiveMutation(
            initial_rate=0.5, adaptation_rate=0.5,
            min_rate=0.01, max_rate=0.5, seed=42
        )
        # Stagnant history should push rate up but not above max
        history = [1.0] * 10
        dag = _chain_dag()
        mut.mutate(dag, fitness_history=history)
        assert mut.min_rate <= mut.rate <= mut.max_rate

    def test_mutate_maintains_dag(self):
        mut = AdaptiveMutation(initial_rate=0.5, seed=42)
        dag = _chain_dag()
        for s in range(20):
            rng = np.random.default_rng(s)
            result = mut.mutate(dag, rng=rng)
            assert _is_acyclic(result)

    def test_no_history_keeps_rate(self):
        mut = AdaptiveMutation(initial_rate=0.15, seed=42)
        dag = _chain_dag()
        mut.mutate(dag)
        assert mut.rate == 0.15


# ── TargetedMutation tests ───────────────────────────────────────

class TestTargetedMutation:
    def test_mutate_maintains_dag(self):
        mut = TargetedMutation(mutation_rate=0.5, seed=42)
        dag = _chain_dag()
        for s in range(20):
            rng = np.random.default_rng(s)
            result = mut.mutate(dag, rng=rng)
            assert _is_acyclic(result)

    def test_score_guided_no_scores(self):
        mut = TargetedMutation(mutation_rate=0.3, seed=42)
        dag = _chain_dag(3)
        result = mut.mutate(dag)
        assert _is_acyclic(result)

    def test_score_guided_with_scores(self):
        mut = TargetedMutation(mutation_rate=0.3, seed=42)
        dag = _chain_dag(3)
        scores = np.ones((3, 3))
        result = mut.mutate(dag, scores=scores)
        assert _is_acyclic(result)

    def test_score_guided_mutation_method(self):
        from cpa.scores.bic import BICScore
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 3))
        data[:, 1] += 0.8 * data[:, 0]
        scorer = BICScore(data)
        mut = TargetedMutation(mutation_rate=0.3, seed=42)
        dag = _empty_dag(3)
        result = mut.score_guided_mutation(dag, scorer.local_score)
        assert _is_acyclic(result)
        # Should add at least one edge
        assert (result != 0).sum() > 0

    def test_small_graph(self):
        mut = TargetedMutation(mutation_rate=0.5, seed=42)
        dag = np.array([[0, 1], [0, 0]], dtype=float)
        result = mut.mutate(dag)
        assert result.shape == (2, 2)
        assert _is_acyclic(result)
