"""Tests for structure learning.

Covers constraint-based, score-based, and hybrid learners
on known graph structures with different graph sizes.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.discovery.structure_learning import (
    ConstraintBasedLearner,
    ScoreBasedLearner,
    HybridLearner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _generate_linear_data(rng, adj, n=500, noise_scale=0.3):
    """Generate data from a linear SEM with given adjacency."""
    p = adj.shape[0]
    coefs = adj * rng.uniform(0.5, 1.5, size=(p, p)) * rng.choice([-1, 1], size=(p, p))
    topo = _topological_order(adj)
    if topo is None:
        topo = list(range(p))
    X = np.zeros((n, p))
    for node in topo:
        parents = np.where(adj[:, node] > 0)[0]
        if len(parents) > 0:
            X[:, node] = X[:, parents] @ coefs[parents, node] + rng.normal(0, noise_scale, size=n)
        else:
            X[:, node] = rng.normal(0, 1, size=n)
    return X


def _topological_order(adj):
    p = adj.shape[0]
    in_deg = np.sum(adj > 0, axis=0)
    order = []
    available = set(np.where(in_deg == 0)[0])
    remaining = set(range(p))
    while available:
        node = min(available)
        available.remove(node)
        remaining.remove(node)
        order.append(node)
        for child in np.where(adj[node] > 0)[0]:
            in_deg[child] -= 1
            if in_deg[child] == 0 and child in remaining:
                available.add(child)
    return order if len(order) == p else None


@pytest.fixture
def chain_3_adj():
    """X0 -> X1 -> X2."""
    return np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)


@pytest.fixture
def chain_3_data(rng, chain_3_adj):
    return _generate_linear_data(rng, chain_3_adj, n=500)


@pytest.fixture
def fork_adj():
    """X0 -> X1, X0 -> X2."""
    return np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=float)


@pytest.fixture
def fork_data(rng, fork_adj):
    return _generate_linear_data(rng, fork_adj, n=500)


@pytest.fixture
def collider_adj():
    """X0 -> X2 <- X1."""
    return np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=float)


@pytest.fixture
def collider_data(rng, collider_adj):
    return _generate_linear_data(rng, collider_adj, n=500)


@pytest.fixture
def diamond_adj():
    """X0 -> X1, X0 -> X2, X1 -> X3, X2 -> X3."""
    return np.array([
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ], dtype=float)


@pytest.fixture
def diamond_data(rng, diamond_adj):
    return _generate_linear_data(rng, diamond_adj, n=500)


@pytest.fixture
def five_node_adj():
    """5-node DAG."""
    adj = np.zeros((5, 5))
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 3] = 1
    adj[2, 3] = 1
    adj[3, 4] = 1
    return adj


@pytest.fixture
def five_node_data(rng, five_node_adj):
    return _generate_linear_data(rng, five_node_adj, n=800)


def _skeleton(adj):
    """Return undirected skeleton."""
    return ((adj + adj.T) > 0).astype(float)


def _shd(true_adj, est_adj):
    """Structural Hamming Distance."""
    return int(np.sum(np.abs(true_adj - est_adj) > 0.5))


# ---------------------------------------------------------------------------
# Test constraint-based learner
# ---------------------------------------------------------------------------

class TestConstraintBasedLearner:

    def test_fit_returns_adjacency(self, chain_3_data):
        learner = ConstraintBasedLearner(alpha=0.05)
        adj = learner.fit(chain_3_data)
        assert adj.shape == (3, 3)

    def test_chain_skeleton_recovery(self, chain_3_data, chain_3_adj):
        learner = ConstraintBasedLearner(alpha=0.05)
        est = learner.fit(chain_3_data)
        true_skel = _skeleton(chain_3_adj)
        est_skel = _skeleton(est)
        # Skeleton should match
        assert_allclose(est_skel, true_skel, atol=0.01)

    def test_fork_recovery(self, fork_data, fork_adj):
        learner = ConstraintBasedLearner(alpha=0.05)
        est = learner.fit(fork_data)
        true_skel = _skeleton(fork_adj)
        est_skel = _skeleton(est)
        assert_allclose(est_skel, true_skel, atol=0.01)

    def test_collider_v_structure(self, collider_data, collider_adj):
        learner = ConstraintBasedLearner(alpha=0.05)
        est = learner.fit(collider_data)
        # Should detect v-structure: 0 -> 2 <- 1
        assert est[0, 2] > 0 or est[2, 0] > 0  # edge exists

    def test_stable_option(self, chain_3_data):
        learner = ConstraintBasedLearner(alpha=0.05, stable=True)
        adj = learner.fit(chain_3_data)
        assert adj.shape == (3, 3)

    def test_max_cond_set(self, chain_3_data):
        learner = ConstraintBasedLearner(alpha=0.05, max_cond_set=1)
        adj = learner.fit(chain_3_data)
        assert adj.shape == (3, 3)

    def test_variable_names(self, chain_3_data):
        learner = ConstraintBasedLearner(alpha=0.05)
        adj = learner.fit(chain_3_data, variable_names=["A", "B", "C"])
        assert adj.shape == (3, 3)

    def test_larger_graph(self, five_node_data, five_node_adj):
        learner = ConstraintBasedLearner(alpha=0.01)
        est = learner.fit(five_node_data)
        assert est.shape == (5, 5)
        # SHD should be reasonable
        shd = _shd(_skeleton(five_node_adj), _skeleton(est))
        assert shd <= 5

    def test_alpha_affects_sparsity(self, chain_3_data):
        learner_strict = ConstraintBasedLearner(alpha=0.001)
        learner_lenient = ConstraintBasedLearner(alpha=0.1)
        adj_strict = learner_strict.fit(chain_3_data)
        adj_lenient = learner_lenient.fit(chain_3_data)
        # Stricter alpha -> sparser graph
        assert np.sum(adj_strict > 0) <= np.sum(adj_lenient > 0) + 2


# ---------------------------------------------------------------------------
# Test score-based learner
# ---------------------------------------------------------------------------

class TestScoreBasedLearner:

    def test_fit_returns_adjacency(self, chain_3_data):
        learner = ScoreBasedLearner(score_fn="bic")
        adj = learner.fit(chain_3_data)
        assert adj.shape == (3, 3)

    def test_chain_recovery(self, chain_3_data, chain_3_adj):
        learner = ScoreBasedLearner(score_fn="bic", max_parents=2)
        est = learner.fit(chain_3_data)
        true_skel = _skeleton(chain_3_adj)
        est_skel = _skeleton(est)
        assert_allclose(est_skel, true_skel, atol=0.01)

    def test_diamond_recovery(self, diamond_data, diamond_adj):
        learner = ScoreBasedLearner(score_fn="bic", max_parents=3)
        est = learner.fit(diamond_data)
        assert est.shape == (4, 4)

    def test_ges_method(self, chain_3_data):
        learner = ScoreBasedLearner(score_fn="bic")
        adj = learner.fit(chain_3_data, method="ges")
        assert adj.shape == (3, 3)

    def test_forward_backward_method(self, chain_3_data):
        learner = ScoreBasedLearner(score_fn="bic")
        adj = learner.fit(chain_3_data, method="forward_backward")
        assert adj.shape == (3, 3)

    def test_tabu_method(self, chain_3_data):
        learner = ScoreBasedLearner(score_fn="bic", tabu_length=5)
        adj = learner.fit(chain_3_data, method="tabu")
        assert adj.shape == (3, 3)

    def test_result_is_dag(self, chain_3_data):
        learner = ScoreBasedLearner(score_fn="bic")
        adj = learner.fit(chain_3_data)
        assert _topological_order(adj) is not None

    def test_max_parents_constraint(self, five_node_data):
        learner = ScoreBasedLearner(score_fn="bic", max_parents=1)
        adj = learner.fit(five_node_data)
        # No node should have more than 1 parent
        for j in range(adj.shape[1]):
            assert np.sum(adj[:, j] > 0) <= 1

    def test_five_node(self, five_node_data, five_node_adj):
        learner = ScoreBasedLearner(score_fn="bic", max_parents=3)
        est = learner.fit(five_node_data)
        assert est.shape == (5, 5)


# ---------------------------------------------------------------------------
# Test hybrid learner
# ---------------------------------------------------------------------------

class TestHybridLearner:

    def test_fit_returns_adjacency(self, chain_3_data):
        learner = HybridLearner(alpha=0.05, score_fn="bic")
        adj = learner.fit(chain_3_data)
        assert adj.shape == (3, 3)

    def test_chain_recovery(self, chain_3_data, chain_3_adj):
        learner = HybridLearner(alpha=0.05, score_fn="bic")
        est = learner.fit(chain_3_data)
        true_skel = _skeleton(chain_3_adj)
        est_skel = _skeleton(est)
        assert_allclose(est_skel, true_skel, atol=0.01)

    def test_result_is_dag(self, chain_3_data):
        learner = HybridLearner(alpha=0.05, score_fn="bic")
        adj = learner.fit(chain_3_data)
        assert _topological_order(adj) is not None

    def test_diamond_recovery(self, diamond_data, diamond_adj):
        learner = HybridLearner(alpha=0.05, score_fn="bic", max_parents=3)
        est = learner.fit(diamond_data)
        assert est.shape == (4, 4)

    def test_five_node(self, five_node_data, five_node_adj):
        learner = HybridLearner(alpha=0.05, score_fn="bic", max_parents=3)
        est = learner.fit(five_node_data)
        assert est.shape == (5, 5)

    def test_fit_restricted(self, chain_3_data):
        learner = HybridLearner(alpha=0.05)
        candidate_edges = {(0, 1), (1, 2)}
        adj = learner.fit_restricted(chain_3_data, candidate_edges)
        assert adj.shape == (3, 3)


# ---------------------------------------------------------------------------
# Test with different graph sizes
# ---------------------------------------------------------------------------

class TestDifferentSizes:

    @pytest.mark.parametrize("p", [3, 5, 8])
    def test_constraint_based_various_sizes(self, rng, p):
        adj = np.zeros((p, p))
        for i in range(p - 1):
            adj[i, i + 1] = 1
        data = _generate_linear_data(rng, adj, n=max(200, p * 50))
        learner = ConstraintBasedLearner(alpha=0.05)
        est = learner.fit(data)
        assert est.shape == (p, p)

    @pytest.mark.parametrize("p", [3, 5, 8])
    def test_score_based_various_sizes(self, rng, p):
        adj = np.zeros((p, p))
        for i in range(p - 1):
            adj[i, i + 1] = 1
        data = _generate_linear_data(rng, adj, n=max(200, p * 50))
        learner = ScoreBasedLearner(score_fn="bic", max_parents=3)
        est = learner.fit(data)
        assert est.shape == (p, p)


# ---------------------------------------------------------------------------
# Test with known-easy and known-hard graphs
# ---------------------------------------------------------------------------

class TestEasyHardGraphs:

    def test_independent_variables(self, rng):
        """All variables independent — empty graph."""
        data = rng.normal(0, 1, size=(300, 4))
        learner = ConstraintBasedLearner(alpha=0.01)
        adj = learner.fit(data)
        # Should find very few or no edges
        assert np.sum(adj > 0) <= 2

    def test_fully_connected_easy(self, rng):
        """Fully connected chain — strong signals."""
        p = 4
        adj = np.zeros((p, p))
        for i in range(p - 1):
            adj[i, i + 1] = 1
        data = _generate_linear_data(rng, adj, n=1000, noise_scale=0.1)
        learner = ConstraintBasedLearner(alpha=0.05)
        est = learner.fit(data)
        true_skel = _skeleton(adj)
        est_skel = _skeleton(est)
        assert_allclose(est_skel, true_skel, atol=0.01)

    def test_weak_signals(self, rng):
        """Weak signals — learner should still run without error."""
        p = 3
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        data = _generate_linear_data(rng, adj, n=100, noise_scale=5.0)
        learner = ConstraintBasedLearner(alpha=0.05)
        est = learner.fit(data)
        assert est.shape == (p, p)
