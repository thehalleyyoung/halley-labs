"""Unit tests for cpa.sampling.structure_mcmc."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from cpa.sampling.structure_mcmc import (
    EdgeProposal,
    StructureMCMC,
    StructureMCMCSample,
)
from cpa.scores.bic import BICScore


# ── helpers ─────────────────────────────────────────────────────────

def _chain_data(n: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = 0.8 * x0 + rng.standard_normal(n) * 0.3
    x2 = 0.7 * x1 + rng.standard_normal(n) * 0.3
    return np.column_stack([x0, x1, x2])


def _simple_score_fn(data):
    scorer = BICScore(data)
    return scorer.local_score


def _is_dag(adj):
    from collections import deque
    n = adj.shape[0]
    in_deg = (adj != 0).sum(axis=0).astype(int)
    queue = deque(int(i) for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        u = queue.popleft()
        count += 1
        for v in range(n):
            if adj[u, v] != 0:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)
    return count == n


# ── EdgeProposal tests ────────────────────────────────────────────

class TestEdgeProposal:
    def test_creation(self):
        ep = EdgeProposal("add", (0, 1), 1.5)
        assert ep.operation == "add"
        assert ep.edge == (0, 1)
        assert ep.score_diff == 1.5

    def test_default_score_diff(self):
        ep = EdgeProposal("remove", (1, 2))
        assert ep.score_diff == 0.0


# ── StructureMCMC proposal tests ──────────────────────────────────

class TestStructureMCMCProposals:
    def test_propose_from_empty(self):
        data = _chain_data(n=100)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        dag = np.zeros((3, 3), dtype=float)
        proposal = mcmc.propose_edge(dag)
        assert proposal is not None
        assert proposal.operation == "add"

    def test_enumerate_proposals(self):
        data = _chain_data(n=100)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        dag = np.zeros((3, 3), dtype=float)
        proposals = mcmc.enumerate_proposals(dag)
        # From empty graph, all valid additions
        assert len(proposals) > 0
        assert all(p.operation == "add" for p in proposals)

    def test_proposals_include_all_operations(self):
        data = _chain_data(n=100)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        proposals = mcmc.enumerate_proposals(dag)
        ops = {p.operation for p in proposals}
        assert "add" in ops
        assert "remove" in ops

    def test_proposals_maintain_dag(self):
        data = _chain_data(n=100)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        proposals = mcmc.enumerate_proposals(dag)
        for p in proposals:
            new_dag = mcmc._apply_proposal(dag, p)
            if new_dag is not None:
                assert _is_dag(new_dag), f"Proposal {p.operation} {p.edge} breaks DAG"


# ── StructureMCMC acyclicity tests ────────────────────────────────

class TestAcyclicity:
    def test_is_dag_true(self):
        data = _chain_data(n=100)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        assert mcmc.is_dag(dag)

    def test_is_dag_false_cycle(self):
        data = _chain_data(n=100)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        dag = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        assert not mcmc.is_dag(dag)

    def test_is_dag_empty(self):
        data = _chain_data(n=100)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        assert mcmc.is_dag(np.zeros((3, 3)))


# ── StructureMCMC run tests ──────────────────────────────────────

class TestStructureMCMCRun:
    def test_run_returns_dags(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        samples = mcmc.run(n_iterations=50, burnin=10)
        assert len(samples) > 0
        for dag in samples:
            assert _is_dag(dag)

    def test_all_samples_are_dags(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        samples = mcmc.run(n_iterations=100, burnin=0)
        for dag in samples:
            assert _is_dag(dag), "MCMC sample is not a DAG"

    def test_burnin_effect(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        s1 = mcmc.run(n_iterations=100, burnin=0)
        mcmc2 = StructureMCMC(score_fn, n_nodes=3, seed=42)
        s2 = mcmc2.run(n_iterations=100, burnin=50)
        assert len(s2) < len(s1)

    def test_acceptance_rate(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        mcmc.run(n_iterations=200, burnin=0)
        rate = mcmc.acceptance_rate
        assert 0.0 <= rate <= 1.0

    def test_restart_interval(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        samples = mcmc.run(n_iterations=100, burnin=0, restart_interval=30)
        assert len(samples) > 0
        for dag in samples:
            assert _is_dag(dag)

    @pytest.mark.slow
    def test_convergence_includes_true_dag(self):
        """With enough iterations, true DAG should appear in samples."""
        data = _chain_data(n=500, seed=42)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        samples = mcmc.run(n_iterations=500, burnin=100)
        # Check if any sample has edges 0->1 and 1->2
        found = any(
            dag[0, 1] != 0 and dag[1, 2] != 0
            for dag in samples
        )
        assert found, "True DAG never appeared in MCMC samples"


# ── Edge posterior tests ──────────────────────────────────────────

class TestEdgePosterior:
    def test_probabilities_shape(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        samples = mcmc.run(n_iterations=50, burnin=10)
        probs = StructureMCMC.edge_posterior_probabilities(samples)
        assert probs.shape == (3, 3)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_empty_samples(self):
        probs = StructureMCMC.edge_posterior_probabilities([])
        assert probs.shape == (0, 0)

    @pytest.mark.slow
    def test_true_edges_high_prob(self):
        data = _chain_data(n=500, seed=42)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        samples = mcmc.run(n_iterations=500, burnin=100)
        probs = StructureMCMC.edge_posterior_probabilities(samples)
        # True edges should have non-trivial probability
        assert probs[0, 1] > 0.1 or probs[1, 0] > 0.1


# ── MAP DAG tests ─────────────────────────────────────────────────

class TestMAPDag:
    def test_returns_best_dag(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, seed=42)
        samples = mcmc.run(n_iterations=50, burnin=10)
        dag, score = StructureMCMC.map_dag(samples, score_fn)
        assert _is_dag(dag)
        assert np.isfinite(score)

    def test_max_parents_respected(self):
        data = _chain_data(n=200)
        score_fn = _simple_score_fn(data)
        mcmc = StructureMCMC(score_fn, n_nodes=3, max_parents=1, seed=42)
        samples = mcmc.run(n_iterations=100, burnin=0)
        for dag in samples:
            max_in = (dag != 0).sum(axis=0).max()
            assert max_in <= 1
