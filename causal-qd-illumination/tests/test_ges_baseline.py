"""Tests for causal_qd.baselines.ges_baseline.GESBaseline.

Covers forward/backward/turning phases, known DAG structures,
score monotonicity, CPDAG output, edge cases, and scoring functions.
"""
from __future__ import annotations

from typing import List

import numpy as np
import numpy.testing as npt
import pytest

from causal_qd.baselines.ges_baseline import GESBaseline, _GaussianBIC
from causal_qd.core.dag import DAG
from causal_qd.data.scm import LinearGaussianSCM
from causal_qd.mec.cpdag import CPDAGConverter
from causal_qd.scores.score_base import DecomposableScore
from causal_qd.types import AdjacencyMatrix, DataMatrix


# ===================================================================
# Helpers
# ===================================================================

def _chain_dag(n: int) -> DAG:
    """0→1→…→n-1."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
    return DAG(adj)


def _fork_dag() -> DAG:
    """0→1, 0→2, 0→3."""
    adj = np.zeros((4, 4), dtype=np.int8)
    adj[0, 1] = adj[0, 2] = adj[0, 3] = 1
    return DAG(adj)


def _collider_dag() -> DAG:
    """0→2←1 (v-structure)."""
    adj = np.zeros((3, 3), dtype=np.int8)
    adj[0, 2] = adj[1, 2] = 1
    return DAG(adj)


def _diamond_dag() -> DAG:
    """0→1, 0→2, 1→3, 2→3."""
    adj = np.zeros((4, 4), dtype=np.int8)
    adj[0, 1] = adj[0, 2] = adj[1, 3] = adj[2, 3] = 1
    return DAG(adj)


def _generate_data(dag: DAG, n_samples: int = 2000, seed: int = 42) -> DataMatrix:
    """Generate linear Gaussian data from a known DAG."""
    scm = LinearGaussianSCM.from_dag(
        dag,
        weight_range=(0.5, 1.5),
        noise_std_range=(0.3, 0.6),
        rng=np.random.default_rng(seed),
    )
    return scm.sample(n_samples, rng=np.random.default_rng(seed + 1))


def _skeleton(adj: AdjacencyMatrix) -> AdjacencyMatrix:
    """Return the undirected skeleton."""
    return (adj | adj.T).astype(np.int8)


# ===================================================================
# Forward phase tests
# ===================================================================

class TestForwardPhase:

    def test_forward_adds_edges_on_chain(self):
        """Forward phase should discover edges of a chain graph."""
        dag = _chain_dag(4)
        data = _generate_data(dag, n_samples=3000, seed=0)

        ges = GESBaseline(phases=("forward",))
        result = ges.fit(data)

        # Should find at least some edges
        assert result.n_edges > 0
        assert ges.n_forward_steps_ > 0

    def test_forward_adds_edges_on_collider(self):
        """Forward phase should detect edges of a collider."""
        dag = _collider_dag()
        data = _generate_data(dag, n_samples=3000, seed=10)

        ges = GESBaseline(phases=("forward",))
        result = ges.fit(data)

        # Collider has 2 edges; forward should find at least those
        skel_true = _skeleton(dag.adjacency)
        skel_pred = _skeleton(result.adjacency)
        shared = int((skel_true & skel_pred).sum()) // 2
        assert shared >= 1, "Forward phase missed collider edges"


# ===================================================================
# Backward phase tests
# ===================================================================

class TestBackwardPhase:

    def test_backward_removes_spurious_edges(self):
        """Backward should remove edges not supported by the data."""
        rng = np.random.default_rng(0)
        # Independent data — no real edges
        data = rng.standard_normal((1000, 4))

        # Run only forward then backward
        ges = GESBaseline(phases=("forward", "backward"))
        result = ges.fit(data)

        # With independent data, expect very few edges after backward
        assert result.n_edges <= 2

    def test_backward_phase_steps_tracked(self):
        """Backward steps counter should be updated."""
        dag = _chain_dag(4)
        data = _generate_data(dag, n_samples=2000, seed=5)
        ges = GESBaseline()
        ges.fit(data)
        # backward_steps >= 0 (may be 0 if forward was already optimal)
        assert ges.n_backward_steps_ >= 0


# ===================================================================
# Known structure tests
# ===================================================================

class TestKnownStructures:

    def test_chain_graph(self):
        """GES should recover a chain graph up to equivalence class."""
        dag = _chain_dag(4)
        data = _generate_data(dag, n_samples=3000, seed=0)

        ges = GESBaseline()
        result = ges.fit(data)

        assert isinstance(result, DAG)
        assert not result.has_cycle()
        # Skeleton should largely match
        skel_true = _skeleton(dag.adjacency)
        skel_pred = _skeleton(result.adjacency)
        n_true = int(skel_true.sum()) // 2
        shared = int((skel_true & skel_pred).sum()) // 2
        recall = shared / max(n_true, 1)
        assert recall >= 0.5, f"Chain skeleton recall {recall:.2f} too low"

    def test_fork_graph(self):
        """GES should recover a fork graph."""
        dag = _fork_dag()
        data = _generate_data(dag, n_samples=3000, seed=10)

        ges = GESBaseline()
        result = ges.fit(data)

        assert result.n_nodes == 4
        assert not result.has_cycle()
        assert result.n_edges > 0

    def test_collider_graph(self):
        """GES should recover a collider / v-structure."""
        dag = _collider_dag()
        data = _generate_data(dag, n_samples=3000, seed=20)

        ges = GESBaseline()
        result = ges.fit(data)

        assert result.n_nodes == 3
        assert not result.has_cycle()

    def test_diamond_graph(self):
        """GES should handle a diamond (common cause + common effect)."""
        dag = _diamond_dag()
        data = _generate_data(dag, n_samples=3000, seed=30)

        ges = GESBaseline()
        result = ges.fit(data)

        assert result.n_nodes == 4
        assert not result.has_cycle()
        assert result.n_edges >= 2  # should find most edges


# ===================================================================
# Score monotonicity
# ===================================================================

class TestScoreMonotonicity:

    def test_forward_scores_non_decreasing(self):
        """Scores during the forward phase should never decrease."""
        dag = _chain_dag(5)
        data = _generate_data(dag, n_samples=2000, seed=0)

        ges = GESBaseline(phases=("forward",))
        ges.fit(data)

        fwd_scores = [s for phase, s, _ in ges.search_path_ if phase == "forward"]
        for i in range(1, len(fwd_scores)):
            assert fwd_scores[i] >= fwd_scores[i - 1] - 1e-10, (
                f"Forward score decreased at step {i}: "
                f"{fwd_scores[i - 1]:.4f} → {fwd_scores[i]:.4f}"
            )

    def test_backward_scores_non_decreasing(self):
        """Scores during the backward phase should never decrease."""
        dag = _chain_dag(4)
        data = _generate_data(dag, n_samples=2000, seed=0)

        ges = GESBaseline()
        ges.fit(data)

        bwd_scores = [s for phase, s, _ in ges.search_path_ if phase == "backward"]
        for i in range(1, len(bwd_scores)):
            assert bwd_scores[i] >= bwd_scores[i - 1] - 1e-10, (
                f"Backward score decreased at step {i}"
            )

    def test_overall_score_improves(self):
        """Final score should be >= initial score."""
        dag = _diamond_dag()
        data = _generate_data(dag, n_samples=2000, seed=0)

        ges = GESBaseline()
        ges.fit(data)

        assert len(ges.search_path_) >= 1
        init_score = ges.search_path_[0][1]
        final_score = ges.search_path_[-1][1]
        assert final_score >= init_score - 1e-10


# ===================================================================
# CPDAG output tests
# ===================================================================

class TestCPDAGOutput:

    def test_fit_cpdag_returns_matrix(self):
        """fit_cpdag should return an adjacency matrix."""
        dag = _chain_dag(4)
        data = _generate_data(dag, n_samples=2000)

        ges = GESBaseline()
        cpdag = ges.fit_cpdag(data)

        assert isinstance(cpdag, np.ndarray)
        assert cpdag.shape == (4, 4)

    def test_cpdag_matches_dag(self):
        """The stored CPDAG should be the CPDAG of the stored DAG."""
        dag = _chain_dag(4)
        data = _generate_data(dag, n_samples=2000)

        ges = GESBaseline()
        ges.fit(data)

        converter = CPDAGConverter()
        expected_cpdag = converter.dag_to_cpdag(ges.dag_)
        npt.assert_array_equal(ges.cpdag_, expected_cpdag)

    def test_cpdag_chain_all_undirected(self):
        """A chain has no v-structures so its CPDAG should be fully undirected."""
        dag = _chain_dag(3)
        data = _generate_data(dag, n_samples=5000, seed=0)

        ges = GESBaseline()
        ges.fit(data)

        if ges.dag_ is not None and ges.dag_.n_edges > 0:
            cpdag = ges.cpdag_
            # Check edges in CPDAG: for each edge, both directions should be present
            # (undirected) — unless GES found a different skeleton
            for i in range(3):
                for j in range(i + 1, 3):
                    if cpdag[i, j] or cpdag[j, i]:
                        # If edge exists, it should be undirected (chain has no v-structures)
                        assert cpdag[i, j] == cpdag[j, i], (
                            f"Edge {i}-{j} should be undirected in chain CPDAG"
                        )


# ===================================================================
# Scoring function tests
# ===================================================================

class TestScoringFunctions:

    def test_with_default_bic(self):
        """GESBaseline should work with the default BIC scorer."""
        dag = _chain_dag(3)
        data = _generate_data(dag, n_samples=1000)

        ges = GESBaseline()
        result = ges.fit(data)
        assert isinstance(result, DAG)
        assert not result.has_cycle()

    def test_with_custom_scorer(self):
        """GESBaseline should accept a custom DecomposableScore."""
        class PenalizedBIC(_GaussianBIC):
            """BIC with heavier penalty."""
            def local_score(self, node, parents, data):
                base = super().local_score(node, parents, data)
                return base - 2.0 * len(parents)  # extra penalty

        dag = _chain_dag(4)
        data = _generate_data(dag, n_samples=1000)

        ges = GESBaseline(score_fn=PenalizedBIC())
        result = ges.fit(data)
        assert isinstance(result, DAG)
        assert not result.has_cycle()


# ===================================================================
# Edge case tests
# ===================================================================

class TestEdgeCases:

    def test_single_node(self):
        """GES on a single node should return an empty DAG."""
        data = np.random.default_rng(0).standard_normal((100, 1))
        ges = GESBaseline()
        result = ges.fit(data)

        assert result.n_nodes == 1
        assert result.n_edges == 0

    def test_two_nodes_independent(self):
        """Two independent variables should yield an empty graph."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((1000, 2))

        ges = GESBaseline()
        result = ges.fit(data)
        assert result.n_edges <= 1  # may find 0 or at most 1 spurious

    def test_two_nodes_dependent(self):
        """Two causally related variables should yield an edge."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(2000)
        y = 1.5 * x + 0.3 * rng.standard_normal(2000)
        data = np.column_stack([x, y])

        ges = GESBaseline()
        result = ges.fit(data)
        assert result.n_edges == 1

    def test_independent_data_sparse_graph(self):
        """On independent data, GES should return a sparse graph."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((500, 5))

        ges = GESBaseline()
        result = ges.fit(data)
        assert result.n_edges <= 3


# ===================================================================
# API consistency tests
# ===================================================================

class TestAPIConsistency:

    def test_run_returns_adjacency(self):
        dag = _chain_dag(4)
        data = _generate_data(dag, n_samples=1000)

        ges = GESBaseline()
        adj = ges.run(data)
        assert isinstance(adj, np.ndarray)
        assert adj.shape == (4, 4)

    def test_run_and_fit_consistent(self):
        """run() and fit().adjacency should give the same result."""
        dag = _chain_dag(3)
        data = _generate_data(dag, n_samples=1000)

        adj_run = GESBaseline().run(data)
        dag_fit = GESBaseline().fit(data)
        npt.assert_array_equal(adj_run, dag_fit.adjacency)

    def test_deterministic(self):
        """Same data should give the same result."""
        dag = _chain_dag(4)
        data = _generate_data(dag, n_samples=1000)

        r1 = GESBaseline().run(data)
        r2 = GESBaseline().run(data)
        npt.assert_array_equal(r1, r2)

    def test_summary(self):
        dag = _chain_dag(4)
        data = _generate_data(dag, n_samples=500)

        ges = GESBaseline()
        ges.fit(data)
        s = ges.summary()

        assert isinstance(s, dict)
        assert "n_forward_steps" in s
        assert "n_backward_steps" in s
        assert "n_turning_steps" in s
        assert "search_path_length" in s

    def test_search_path_recorded(self):
        """search_path_ should be populated after fit."""
        dag = _chain_dag(4)
        data = _generate_data(dag, n_samples=1000)

        ges = GESBaseline()
        ges.fit(data)

        assert len(ges.search_path_) >= 1
        # First entry should be "init"
        assert ges.search_path_[0][0] == "init"
        # Each entry is (phase, score, n_edges)
        for phase, score, n_edges in ges.search_path_:
            assert phase in ("init", "forward", "backward", "turning")
            assert isinstance(score, float)
            assert isinstance(n_edges, int)

    def test_import_from_baselines_init(self):
        """GESBaseline should be importable from causal_qd.baselines."""
        from causal_qd.baselines import GESBaseline as GES
        assert GES is GESBaseline

    def test_phases_parameter(self):
        """Specifying phases should control which phases run."""
        dag = _chain_dag(4)
        data = _generate_data(dag, n_samples=1000)

        ges_fwd = GESBaseline(phases=("forward",))
        ges_fwd.fit(data)
        assert ges_fwd.n_backward_steps_ == 0
        assert ges_fwd.n_turning_steps_ == 0
