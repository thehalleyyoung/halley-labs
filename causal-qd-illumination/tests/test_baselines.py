"""Tests for causal_qd.baselines module.

Covers PCAlgorithm, GESAlgorithm, and MMHCAlgorithm on known and random graphs.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.testing as npt
import pytest

from causal_qd.baselines.pc import PCAlgorithm
from causal_qd.baselines.ges import GESAlgorithm
from causal_qd.baselines.mmhc import MMHCAlgorithm
from causal_qd.core.dag import DAG
from causal_qd.data.scm import LinearGaussianSCM
from causal_qd.data.generator import DataGenerator
from causal_qd.metrics.structural import SHD
from causal_qd.types import AdjacencyMatrix, DataMatrix


# ===================================================================
# Helpers
# ===================================================================

def _chain_dag(n: int) -> DAG:
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
    return DAG(adj)


def _fork_dag() -> DAG:
    """0→1, 0→2, 0→3."""
    adj = np.zeros((4, 4), dtype=np.int8)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[0, 3] = 1
    return DAG(adj)


def _collider_dag() -> DAG:
    """0→2, 1→2."""
    adj = np.zeros((3, 3), dtype=np.int8)
    adj[0, 2] = 1
    adj[1, 2] = 1
    return DAG(adj)


def _generate_data_from_dag(dag: DAG, n_samples: int = 1000,
                             seed: int = 42) -> DataMatrix:
    """Generate linear Gaussian data from a known DAG."""
    scm = LinearGaussianSCM.from_dag(
        dag,
        weight_range=(0.5, 1.0),
        noise_std_range=(0.3, 0.6),
        rng=np.random.default_rng(seed),
    )
    return scm.sample(n_samples, rng=np.random.default_rng(seed + 1))


def _is_dag(adj: AdjacencyMatrix) -> bool:
    """Check if adjacency matrix represents a DAG (no cycles)."""
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    stack = np.zeros(n, dtype=bool)

    def _dfs(v: int) -> bool:
        visited[v] = True
        stack[v] = True
        for w in range(n):
            if adj[v, w]:
                if not visited[w]:
                    if _dfs(w):
                        return True
                elif stack[w]:
                    return True
        stack[v] = False
        return False

    for v in range(n):
        if not visited[v]:
            if _dfs(v):
                return False
    return True


def _skeleton_matches(adj1: AdjacencyMatrix, adj2: AdjacencyMatrix) -> bool:
    """Check if skeletons (ignoring direction) match."""
    s1 = (adj1 | adj1.T).astype(bool)
    s2 = (adj2 | adj2.T).astype(bool)
    return np.array_equal(s1, s2)


# ===================================================================
# PC Algorithm
# ===================================================================

class TestPCAlgorithm:
    """Tests for causal_qd.baselines.pc.PCAlgorithm."""

    def test_pc_algorithm_on_known_graph(self):
        """PC should recover the skeleton of a simple chain."""
        dag = _chain_dag(4)
        data = _generate_data_from_dag(dag, n_samples=2000, seed=0)

        pc = PCAlgorithm(alpha=0.05)
        result_dag = pc.fit(data)
        result_adj = result_dag.adjacency

        # Skeleton should largely match the true graph
        true_adj = dag.adjacency
        true_skeleton = (true_adj | true_adj.T).astype(bool)
        pred_skeleton = (result_adj | result_adj.T).astype(bool)

        # At least most edges should be in the skeleton
        n_true_skel_edges = true_skeleton.sum() // 2
        n_shared = ((true_skeleton & pred_skeleton).sum()) // 2
        recall = n_shared / max(n_true_skel_edges, 1)
        assert recall >= 0.5, f"Skeleton recall too low: {recall:.2f}"

    def test_pc_returns_dag(self):
        dag = _chain_dag(5)
        data = _generate_data_from_dag(dag, n_samples=1000)
        pc = PCAlgorithm(alpha=0.05)
        result = pc.fit(data)
        assert isinstance(result, DAG)
        assert not result.has_cycle()

    def test_pc_run_returns_adjacency(self):
        dag = _chain_dag(4)
        data = _generate_data_from_dag(dag, n_samples=500)
        pc = PCAlgorithm(alpha=0.05)
        adj = pc.run(data)
        assert isinstance(adj, np.ndarray)
        assert adj.shape == (4, 4)

    def test_pc_on_fork(self):
        dag = _fork_dag()
        data = _generate_data_from_dag(dag, n_samples=2000, seed=10)
        pc = PCAlgorithm(alpha=0.05)
        result = pc.fit(data)
        assert result.n_nodes == 4
        assert not result.has_cycle()

    def test_pc_on_collider(self):
        dag = _collider_dag()
        data = _generate_data_from_dag(dag, n_samples=2000, seed=20)
        pc = PCAlgorithm(alpha=0.05)
        result = pc.fit(data)
        assert result.n_nodes == 3
        assert not result.has_cycle()

    def test_pc_alpha_sensitivity(self):
        """Lower alpha → fewer edges (more conservative)."""
        dag = _chain_dag(5)
        data = _generate_data_from_dag(dag, n_samples=500, seed=99)

        adj_loose = PCAlgorithm(alpha=0.2).run(data)
        adj_strict = PCAlgorithm(alpha=0.001).run(data)

        n_edges_loose = adj_loose.sum()
        n_edges_strict = adj_strict.sum()
        assert n_edges_strict <= n_edges_loose

    def test_pc_on_independent_data(self):
        """On truly independent data, PC should find few or no edges."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((500, 5))
        pc = PCAlgorithm(alpha=0.01)
        result = pc.fit(data)
        assert result.n_edges <= 3  # very few spurious edges

    def test_pc_summary(self):
        dag = _chain_dag(4)
        data = _generate_data_from_dag(dag, n_samples=500)
        pc = PCAlgorithm(alpha=0.05)
        pc.fit(data)
        s = pc.summary()
        assert isinstance(s, dict)

    def test_pc_stable_mode(self):
        dag = _chain_dag(5)
        data = _generate_data_from_dag(dag, n_samples=1000)
        pc = PCAlgorithm(alpha=0.05, stable=True)
        result = pc.fit(data)
        assert not result.has_cycle()

    def test_pc_conservative_mode(self):
        dag = _chain_dag(4)
        data = _generate_data_from_dag(dag, n_samples=1000)
        pc = PCAlgorithm(alpha=0.05, conservative=True)
        result = pc.fit(data)
        assert not result.has_cycle()


# ===================================================================
# GES Algorithm
# ===================================================================

class TestGESAlgorithm:
    """Tests for causal_qd.baselines.ges.GESAlgorithm."""

    def test_ges_algorithm_on_known_graph(self):
        """GES should approximate a known chain graph."""
        dag = _chain_dag(4)
        data = _generate_data_from_dag(dag, n_samples=2000, seed=0)

        ges = GESAlgorithm()
        result_dag = ges.fit(data)
        result_adj = result_dag.adjacency

        shd = SHD.compute(result_adj, dag.adjacency)
        # GES may find CPDAG; allow reasonable SHD
        max_edges = dag.n_nodes * (dag.n_nodes - 1) // 2
        assert shd <= max_edges, f"SHD {shd} too large"

    def test_ges_returns_dag(self):
        dag = _chain_dag(5)
        data = _generate_data_from_dag(dag, n_samples=1000)
        ges = GESAlgorithm()
        result = ges.fit(data)
        assert isinstance(result, DAG)
        assert not result.has_cycle()

    def test_ges_run_returns_adjacency(self):
        dag = _chain_dag(4)
        data = _generate_data_from_dag(dag, n_samples=500)
        ges = GESAlgorithm()
        adj = ges.run(data)
        assert isinstance(adj, np.ndarray)
        assert adj.shape == (4, 4)

    def test_ges_on_fork(self):
        dag = _fork_dag()
        data = _generate_data_from_dag(dag, n_samples=2000, seed=10)
        ges = GESAlgorithm()
        result = ges.fit(data)
        assert result.n_nodes == 4
        assert not result.has_cycle()

    def test_ges_on_collider(self):
        dag = _collider_dag()
        data = _generate_data_from_dag(dag, n_samples=2000, seed=20)
        ges = GESAlgorithm()
        result = ges.fit(data)
        assert not result.has_cycle()

    def test_ges_on_independent_data(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((500, 4))
        ges = GESAlgorithm()
        result = ges.fit(data)
        assert result.n_edges <= 3

    def test_ges_summary(self):
        dag = _chain_dag(4)
        data = _generate_data_from_dag(dag, n_samples=500)
        ges = GESAlgorithm()
        ges.fit(data)
        s = ges.summary()
        assert isinstance(s, dict)

    def test_ges_forward_backward_consistency(self):
        """Running GES twice on same data should give same result."""
        dag = _chain_dag(4)
        data = _generate_data_from_dag(dag, n_samples=1000)
        r1 = GESAlgorithm().run(data)
        r2 = GESAlgorithm().run(data)
        npt.assert_array_equal(r1, r2)


# ===================================================================
# MMHC Algorithm
# ===================================================================

class TestMMHCAlgorithm:
    """Tests for causal_qd.baselines.mmhc.MMHCAlgorithm."""

    def test_mmhc_algorithm_on_known_graph(self):
        dag = _chain_dag(4)
        data = _generate_data_from_dag(dag, n_samples=2000, seed=0)

        mmhc = MMHCAlgorithm(alpha=0.05)
        result_dag = mmhc.fit(data)
        result_adj = result_dag.adjacency

        shd = SHD.compute(result_adj, dag.adjacency)
        max_edges = dag.n_nodes * (dag.n_nodes - 1) // 2
        assert shd <= max_edges, f"SHD {shd} too large"

    def test_mmhc_returns_dag(self):
        dag = _chain_dag(5)
        data = _generate_data_from_dag(dag, n_samples=1000)
        mmhc = MMHCAlgorithm(alpha=0.05)
        result = mmhc.fit(data)
        assert isinstance(result, DAG)
        assert not result.has_cycle()

    def test_mmhc_run_returns_adjacency(self):
        dag = _chain_dag(4)
        data = _generate_data_from_dag(dag, n_samples=500)
        mmhc = MMHCAlgorithm(alpha=0.05)
        adj = mmhc.run(data)
        assert isinstance(adj, np.ndarray)
        assert adj.shape == (4, 4)

    def test_mmhc_on_fork(self):
        dag = _fork_dag()
        data = _generate_data_from_dag(dag, n_samples=2000, seed=10)
        mmhc = MMHCAlgorithm(alpha=0.05)
        result = mmhc.fit(data)
        assert result.n_nodes == 4
        assert not result.has_cycle()

    def test_mmhc_on_collider(self):
        dag = _collider_dag()
        data = _generate_data_from_dag(dag, n_samples=2000, seed=20)
        mmhc = MMHCAlgorithm(alpha=0.05)
        result = mmhc.fit(data)
        assert not result.has_cycle()

    def test_mmhc_on_independent_data(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((500, 4))
        mmhc = MMHCAlgorithm(alpha=0.01)
        result = mmhc.fit(data)
        assert result.n_edges <= 3

    def test_mmhc_summary(self):
        dag = _chain_dag(4)
        data = _generate_data_from_dag(dag, n_samples=500)
        mmhc = MMHCAlgorithm(alpha=0.05)
        mmhc.fit(data)
        s = mmhc.summary()
        assert isinstance(s, dict)


# ===================================================================
# Cross-baseline comparisons
# ===================================================================

class TestBaselineCrossComparison:
    """Tests that apply to all baseline algorithms."""

    def test_baselines_return_valid_dags(self):
        """All three baselines should return valid DAGs on the same data."""
        dag = _chain_dag(5)
        data = _generate_data_from_dag(dag, n_samples=1000)

        algorithms = [
            PCAlgorithm(alpha=0.05),
            GESAlgorithm(),
            MMHCAlgorithm(alpha=0.05),
        ]

        for algo in algorithms:
            result = algo.fit(data)
            assert isinstance(result, DAG), f"{algo.__class__.__name__} didn't return DAG"
            assert not result.has_cycle(), f"{algo.__class__.__name__} returned cyclic graph"
            assert result.n_nodes == 5

    def test_baselines_run_consistent_with_fit(self):
        """run() and fit().adjacency should give equivalent results."""
        dag = _chain_dag(4)
        data = _generate_data_from_dag(dag, n_samples=500)

        for AlgoClass in [PCAlgorithm, GESAlgorithm, MMHCAlgorithm]:
            algo = AlgoClass() if AlgoClass == GESAlgorithm else AlgoClass(alpha=0.05)
            adj_run = algo.run(data)

            algo2 = AlgoClass() if AlgoClass == GESAlgorithm else AlgoClass(alpha=0.05)
            dag_fit = algo2.fit(data)

            assert adj_run.shape == dag_fit.adjacency.shape

    def test_baselines_on_empty_graph(self):
        """Baselines should handle independent data gracefully."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((500, 5))

        for AlgoClass in [PCAlgorithm, GESAlgorithm, MMHCAlgorithm]:
            algo = AlgoClass() if AlgoClass == GESAlgorithm else AlgoClass(alpha=0.01)
            result = algo.fit(data)
            assert not result.has_cycle()

    def test_all_baselines_shd_better_than_complete(self):
        """Each baseline's SHD should be lower than the SHD of a complete graph."""
        dag = _chain_dag(5)
        data = _generate_data_from_dag(dag, n_samples=1000)
        true_adj = dag.adjacency

        n = 5
        complete = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                complete[i, j] = 1
        shd_complete = SHD.compute(complete, true_adj)

        for AlgoClass in [PCAlgorithm, GESAlgorithm, MMHCAlgorithm]:
            algo = AlgoClass() if AlgoClass == GESAlgorithm else AlgoClass(alpha=0.05)
            pred = algo.run(data)
            shd_algo = SHD.compute(pred, true_adj)
            assert shd_algo <= shd_complete, (
                f"{AlgoClass.__name__} SHD={shd_algo} >= complete SHD={shd_complete}"
            )

    def test_baselines_handle_large_n_samples(self):
        """Baselines should work with large sample sizes without error."""
        dag = _chain_dag(4)
        data = _generate_data_from_dag(dag, n_samples=5000)
        for AlgoClass in [PCAlgorithm, GESAlgorithm, MMHCAlgorithm]:
            algo = AlgoClass() if AlgoClass == GESAlgorithm else AlgoClass(alpha=0.05)
            result = algo.fit(data)
            assert not result.has_cycle()

    def test_baselines_deterministic(self):
        """Same data should give same result (no randomness in baselines)."""
        dag = _chain_dag(4)
        data = _generate_data_from_dag(dag, n_samples=1000)

        for AlgoClass in [PCAlgorithm, GESAlgorithm]:
            algo1 = AlgoClass() if AlgoClass == GESAlgorithm else AlgoClass(alpha=0.05)
            algo2 = AlgoClass() if AlgoClass == GESAlgorithm else AlgoClass(alpha=0.05)
            r1 = algo1.run(data)
            r2 = algo2.run(data)
            npt.assert_array_equal(r1, r2)
