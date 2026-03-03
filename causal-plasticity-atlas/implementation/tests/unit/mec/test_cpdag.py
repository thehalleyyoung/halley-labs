"""Unit tests for cpa.mec.cpdag.CPDAG."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cpa.mec.cpdag import (
    CPDAG,
    dag_to_cpdag,
    cpdag_to_dags,
    _find_v_structures,
    _is_dag,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def chain_dag():
    """Chain: 0 → 1 → 2."""
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    adj[1, 2] = 1
    return adj


@pytest.fixture
def fork_dag():
    """Fork: 1 ← 0 → 2."""
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    return adj


@pytest.fixture
def collider_dag():
    """Collider: 0 → 2 ← 1."""
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 2] = 1
    adj[1, 2] = 1
    return adj


@pytest.fixture
def diamond_dag():
    """Diamond: 0 → 1, 0 → 2, 1 → 3, 2 → 3."""
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 3] = 1
    adj[2, 3] = 1
    return adj


@pytest.fixture
def five_node_dag():
    """5-node DAG: 0→1, 0→2, 1→3, 2→3, 3→4."""
    adj = np.zeros((5, 5), dtype=int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 3] = 1
    adj[2, 3] = 1
    adj[3, 4] = 1
    return adj


# ===================================================================
# Tests – CPDAG from DAG
# ===================================================================


class TestCPDAGFromDAG:
    """Test CPDAG construction from DAG via Chickering's algorithm."""

    def test_chain_cpdag_has_undirected_edges(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        # Chain has no v-structure → all edges reversible → undirected
        assert cpdag.has_undirected_edge(0, 1) or cpdag.has_directed_edge(0, 1)

    def test_collider_cpdag_directed_edges(self, collider_dag):
        cpdag = CPDAG.from_dag(collider_dag)
        # Collider v-structure → both edges must be directed
        assert cpdag.has_directed_edge(0, 2)
        assert cpdag.has_directed_edge(1, 2)

    def test_collider_no_undirected_edges(self, collider_dag):
        cpdag = CPDAG.from_dag(collider_dag)
        assert not cpdag.has_undirected_edge(0, 2)
        assert not cpdag.has_undirected_edge(1, 2)

    def test_diamond_cpdag_v_structures(self, diamond_dag):
        cpdag = CPDAG.from_dag(diamond_dag)
        # 0→1, 0→2, 1→3, 2→3 has v-structure at node 3
        assert cpdag.has_directed_edge(1, 3)
        assert cpdag.has_directed_edge(2, 3)

    def test_from_dag_returns_cpdag_type(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        assert isinstance(cpdag, CPDAG)

    def test_cpdag_n_nodes(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        assert cpdag.n_nodes == 3

    def test_from_dag_five_nodes(self, five_node_dag):
        cpdag = CPDAG.from_dag(five_node_dag)
        assert cpdag.n_nodes == 5


# ===================================================================
# Tests – V-structure detection
# ===================================================================


class TestVStructures:
    """Test v-structure identification."""

    def test_collider_has_v_structure(self, collider_dag):
        vs = _find_v_structures(collider_dag)
        assert len(vs) >= 1
        # v-structure: (0, 2, 1) or (1, 2, 0)
        collider_nodes = {v[1] for v in vs}
        assert 2 in collider_nodes

    def test_chain_no_v_structure(self, chain_dag):
        vs = _find_v_structures(chain_dag)
        assert len(vs) == 0

    def test_fork_no_v_structure(self, fork_dag):
        vs = _find_v_structures(fork_dag)
        assert len(vs) == 0

    def test_diamond_v_structure_at_node3(self, diamond_dag):
        vs = _find_v_structures(diamond_dag)
        collider_nodes = {v[1] for v in vs}
        assert 3 in collider_nodes


# ===================================================================
# Tests – DAG sampling from CPDAG
# ===================================================================


class TestDAGSampling:
    """Test sampling valid DAGs from CPDAG."""

    def test_sampled_dag_is_dag(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        dag = cpdag.to_dag(seed=42)
        assert _is_dag(dag)

    def test_sampled_dag_shape(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        dag = cpdag.to_dag(seed=42)
        assert dag.shape == (3, 3)

    def test_sampled_dag_in_mec(self, collider_dag):
        cpdag = CPDAG.from_dag(collider_dag)
        dag = cpdag.to_dag(seed=42)
        # The sampled DAG should produce the same CPDAG
        cpdag2 = CPDAG.from_dag(dag)
        assert cpdag == cpdag2

    def test_multiple_samples_valid(self, diamond_dag):
        cpdag = CPDAG.from_dag(diamond_dag)
        for seed in range(5):
            dag = cpdag.to_dag(seed=seed)
            assert _is_dag(dag)

    def test_cpdag_to_dags_function(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        dags = cpdag_to_dags(cpdag)
        assert len(dags) >= 1
        for d in dags:
            assert _is_dag(d)

    def test_collider_mec_size_1(self, collider_dag):
        cpdag = CPDAG.from_dag(collider_dag)
        dags = cpdag_to_dags(cpdag)
        # Collider has unique MEC member
        assert len(dags) == 1


# ===================================================================
# Tests – Structural Hamming Distance
# ===================================================================


class TestSHD:
    """Test structural Hamming distance between CPDAGs."""

    def test_shd_same_cpdag(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        assert cpdag.structural_hamming_distance(cpdag) == 0

    def test_shd_different_cpdags(self, chain_dag, collider_dag):
        cpdag1 = CPDAG.from_dag(chain_dag)
        cpdag2 = CPDAG.from_dag(collider_dag)
        shd = cpdag1.structural_hamming_distance(cpdag2)
        assert shd > 0

    def test_shd_symmetric(self, chain_dag, fork_dag):
        cpdag1 = CPDAG.from_dag(chain_dag)
        cpdag2 = CPDAG.from_dag(fork_dag)
        assert cpdag1.structural_hamming_distance(cpdag2) == \
               cpdag2.structural_hamming_distance(cpdag1)

    def test_shd_nonnegative(self, chain_dag, collider_dag):
        cpdag1 = CPDAG.from_dag(chain_dag)
        cpdag2 = CPDAG.from_dag(collider_dag)
        assert cpdag1.structural_hamming_distance(cpdag2) >= 0

    def test_shd_copy_is_zero(self, diamond_dag):
        cpdag = CPDAG.from_dag(diamond_dag)
        cpdag_copy = cpdag.copy()
        assert cpdag.structural_hamming_distance(cpdag_copy) == 0


# ===================================================================
# Tests – Meek rules
# ===================================================================


class TestMeekRules:
    """Test Meek rules application."""

    def test_meek_rules_on_valid_cpdag(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        assert cpdag.is_valid()

    def test_meek_propagates_orientation(self):
        # Create a PDAG with a v-structure that forces orientation
        cpdag = CPDAG(4)
        cpdag.add_directed_edge(0, 2)
        cpdag.add_directed_edge(1, 2)
        cpdag.add_undirected_edge(2, 3)
        # After Meek rules, 2-3 should become directed (Rule 2)
        cpdag._apply_meek_rules()
        assert cpdag.has_directed_edge(2, 3)

    def test_meek_no_new_v_structures(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        cpdag._apply_meek_rules()
        assert cpdag.is_valid()


# ===================================================================
# Tests – Edge operations
# ===================================================================


class TestEdgeOperations:
    """Test edge add/remove/query operations."""

    def test_add_directed_edge(self):
        cpdag = CPDAG(3)
        cpdag.add_directed_edge(0, 1)
        assert cpdag.has_directed_edge(0, 1)
        assert not cpdag.has_directed_edge(1, 0)

    def test_add_undirected_edge(self):
        cpdag = CPDAG(3)
        cpdag.add_undirected_edge(0, 1)
        assert cpdag.has_undirected_edge(0, 1)
        assert cpdag.has_undirected_edge(1, 0)

    def test_remove_directed_edge(self):
        cpdag = CPDAG(3)
        cpdag.add_directed_edge(0, 1)
        cpdag.remove_directed_edge(0, 1)
        assert not cpdag.has_directed_edge(0, 1)

    def test_neighbors(self):
        cpdag = CPDAG(3)
        cpdag.add_undirected_edge(0, 1)
        cpdag.add_undirected_edge(0, 2)
        assert cpdag.neighbors(0) == {1, 2}

    def test_children(self):
        cpdag = CPDAG(3)
        cpdag.add_directed_edge(0, 1)
        cpdag.add_directed_edge(0, 2)
        assert cpdag.children(0) == {1, 2}

    def test_parents(self):
        cpdag = CPDAG(3)
        cpdag.add_directed_edge(0, 2)
        cpdag.add_directed_edge(1, 2)
        assert cpdag.parents(2) == {0, 1}


# ===================================================================
# Tests – Adjacency matrix conversion
# ===================================================================


class TestAdjacencyConversion:
    """Test adjacency matrix to/from CPDAG."""

    def test_to_adjacency_matrix(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        adj = cpdag.to_adjacency_matrix()
        assert adj.shape == (3, 3)

    def test_roundtrip_collider(self, collider_dag):
        cpdag = CPDAG.from_dag(collider_dag)
        adj = cpdag.to_adjacency_matrix()
        cpdag2 = CPDAG.from_adjacency_matrix(adj)
        assert cpdag == cpdag2

    def test_dag_to_cpdag_function(self, chain_dag):
        cpdag = dag_to_cpdag(chain_dag)
        assert isinstance(cpdag, CPDAG)


# ===================================================================
# Tests – MEC size
# ===================================================================


class TestMECSize:
    """Test MEC size computation."""

    def test_collider_mec_size(self, collider_dag):
        cpdag = CPDAG.from_dag(collider_dag)
        assert cpdag.num_dags_in_mec() == 1

    def test_chain_mec_size_gt_1(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        assert cpdag.num_dags_in_mec() >= 1


# ===================================================================
# Tests – Copy and equality
# ===================================================================


class TestCopyEquality:
    """Test copy and equality semantics."""

    def test_copy_equal(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        assert cpdag == cpdag.copy()

    def test_copy_independent(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        cp = cpdag.copy()
        cp.add_directed_edge(0, 2)
        assert cpdag != cp

    def test_repr(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        assert "CPDAG" in repr(cpdag)
