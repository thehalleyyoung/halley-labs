"""Unit tests for cpa.core.scm.StructuralCausalModel."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cpa.core.scm import StructuralCausalModel


# ===================================================================
# Fixtures – common graph topologies
# ===================================================================


@pytest.fixture
def empty_3():
    """3-node graph with no edges."""
    return StructuralCausalModel(np.zeros((3, 3)))


@pytest.fixture
def single_node():
    """Single-node graph."""
    return StructuralCausalModel(np.zeros((1, 1)), variable_names=["Z"])


@pytest.fixture
def chain_3():
    """Chain: 0 → 1 → 2."""
    adj = np.zeros((3, 3))
    adj[0, 1] = 1.0
    adj[1, 2] = 1.0
    return StructuralCausalModel(adj, variable_names=["A", "B", "C"])


@pytest.fixture
def fork_3():
    """Fork: 1 ← 0 → 2."""
    adj = np.zeros((3, 3))
    adj[0, 1] = 1.0
    adj[0, 2] = 1.0
    return StructuralCausalModel(adj)


@pytest.fixture
def collider_3():
    """Collider: 0 → 2 ← 1."""
    adj = np.zeros((3, 3))
    adj[0, 2] = 1.0
    adj[1, 2] = 1.0
    return StructuralCausalModel(adj)


@pytest.fixture
def diamond_4():
    """Diamond: 0 → 1, 0 → 2, 1 → 3, 2 → 3."""
    adj = np.zeros((4, 4))
    adj[0, 1] = 1.0
    adj[0, 2] = 1.0
    adj[1, 3] = 1.0
    adj[2, 3] = 1.0
    return StructuralCausalModel(adj)


@pytest.fixture
def complete_dag_4():
    """Complete DAG on 4 nodes (upper-triangular)."""
    adj = np.zeros((4, 4))
    for i in range(4):
        for j in range(i + 1, 4):
            adj[i, j] = 1.0
    return StructuralCausalModel(adj)


# ===================================================================
# Construction & validation
# ===================================================================


class TestConstruction:
    def test_default_variable_names(self, empty_3):
        assert empty_3.variable_names == ["X0", "X1", "X2"]

    def test_custom_variable_names(self, chain_3):
        assert chain_3.variable_names == ["A", "B", "C"]

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            StructuralCausalModel(np.zeros((2, 3)))

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="square"):
            StructuralCausalModel(np.array([1, 2, 3]))

    def test_duplicate_names_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            StructuralCausalModel(np.zeros((2, 2)), variable_names=["A", "A"])

    def test_name_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            StructuralCausalModel(np.zeros((2, 2)), variable_names=["A"])

    def test_residual_variances_non_positive_raises(self):
        with pytest.raises(ValueError, match="must all be > 0"):
            StructuralCausalModel(
                np.zeros((2, 2)), residual_variances=np.array([1.0, -0.5])
            )

    def test_residual_variances_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            StructuralCausalModel(
                np.zeros((2, 2)), residual_variances=np.ones(3)
            )

    def test_regression_coefficients_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            StructuralCausalModel(
                np.zeros((2, 2)),
                regression_coefficients=np.zeros((3, 3)),
            )

    def test_default_residual_variances(self, empty_3):
        np.testing.assert_array_equal(empty_3.residual_variances, np.ones(3))

    def test_default_regression_coefficients_copy_adj(self):
        adj = np.array([[0, 0.5], [0, 0]])
        scm = StructuralCausalModel(adj)
        np.testing.assert_array_equal(scm.regression_coefficients, adj)

    def test_sample_size_default(self, empty_3):
        assert empty_3.sample_size == 0


# ===================================================================
# Properties
# ===================================================================


class TestProperties:
    def test_num_variables(self, chain_3, diamond_4, single_node):
        assert chain_3.num_variables == 3
        assert diamond_4.num_variables == 4
        assert single_node.num_variables == 1

    def test_num_edges(self, chain_3, fork_3, diamond_4, empty_3):
        assert chain_3.num_edges == 2
        assert fork_3.num_edges == 2
        assert diamond_4.num_edges == 4
        assert empty_3.num_edges == 0

    def test_adjacency_matrix_returns_copy(self, chain_3):
        a = chain_3.adjacency_matrix
        a[0, 1] = 999
        assert chain_3.adjacency_matrix[0, 1] == 1.0

    def test_regression_coefficients_returns_copy(self, chain_3):
        c = chain_3.regression_coefficients
        c[0, 1] = 999
        assert chain_3.regression_coefficients[0, 1] == 1.0

    def test_residual_variances_returns_copy(self, chain_3):
        r = chain_3.residual_variances
        r[0] = 999
        assert chain_3.residual_variances[0] == 1.0


# ===================================================================
# Edge operations
# ===================================================================


class TestEdgeOperations:
    def test_has_edge(self, chain_3):
        assert chain_3.has_edge(0, 1)
        assert chain_3.has_edge(1, 2)
        assert not chain_3.has_edge(0, 2)
        assert not chain_3.has_edge(1, 0)

    def test_add_edge(self, empty_3):
        empty_3.add_edge(0, 1, weight=2.0)
        assert empty_3.has_edge(0, 1)
        assert empty_3.adjacency_matrix[0, 1] == 2.0

    def test_add_edge_self_loop_raises(self, empty_3):
        with pytest.raises(ValueError, match="Self-loop"):
            empty_3.add_edge(0, 0)

    def test_add_edge_duplicate_raises(self, chain_3):
        with pytest.raises(ValueError, match="already exists"):
            chain_3.add_edge(0, 1)

    def test_add_edge_cycle_raises(self, chain_3):
        with pytest.raises(ValueError, match="cycle"):
            chain_3.add_edge(2, 0)

    def test_add_edge_no_dag_check(self, empty_3):
        empty_3.add_edge(0, 1, check_dag=False)
        empty_3.add_edge(1, 0, check_dag=False)
        assert empty_3.has_edge(0, 1) and empty_3.has_edge(1, 0)

    def test_remove_edge(self, chain_3):
        w = chain_3.remove_edge(0, 1)
        assert w == 1.0
        assert not chain_3.has_edge(0, 1)

    def test_remove_nonexistent_edge_raises(self, chain_3):
        with pytest.raises(ValueError, match="does not exist"):
            chain_3.remove_edge(0, 2)

    def test_reverse_edge(self, chain_3):
        chain_3.reverse_edge(0, 1)
        assert chain_3.has_edge(1, 0)
        assert not chain_3.has_edge(0, 1)

    def test_reverse_edge_cycle_raises(self):
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        scm = StructuralCausalModel(adj)
        # Reversing 0→1 to 1→0 would not create a cycle here,
        # but reversing 1→2 with 0→1 present: 0→1→… already fine.
        # Create a scenario where reversal creates a cycle: 0→1, 1→2, 0→2
        adj2 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
        scm2 = StructuralCausalModel(adj2)
        # Reversing 0→2 gives 2→0, creating 0→1→2→0 cycle
        with pytest.raises(ValueError):
            scm2.reverse_edge(0, 2)


# ===================================================================
# Graph queries
# ===================================================================


class TestGraphQueries:
    def test_parents(self, diamond_4):
        assert diamond_4.parents(0) == []
        assert sorted(diamond_4.parents(3)) == [1, 2]

    def test_children(self, fork_3):
        assert sorted(fork_3.children(0)) == [1, 2]
        assert fork_3.children(1) == []

    def test_ancestors(self, diamond_4):
        assert diamond_4.ancestors(3) == {0, 1, 2}
        assert diamond_4.ancestors(0) == set()

    def test_descendants(self, diamond_4):
        assert diamond_4.descendants(0) == {1, 2, 3}
        assert diamond_4.descendants(3) == set()

    def test_is_ancestor(self, diamond_4):
        assert diamond_4.is_ancestor(0, 3)
        assert not diamond_4.is_ancestor(3, 0)
        assert not diamond_4.is_ancestor(1, 2)

    def test_markov_blanket_chain(self, chain_3):
        # MB(1) = {0, 2} (parent + child)
        assert chain_3.markov_blanket(1) == {0, 2}

    def test_markov_blanket_collider(self, collider_3):
        # MB(2) = {0, 1} (parents)
        assert collider_3.markov_blanket(2) == {0, 1}
        # MB(0) = {1, 2} (child 2, co-parent 1)
        assert collider_3.markov_blanket(0) == {1, 2}

    def test_markov_blanket_root_no_children(self, empty_3):
        assert empty_3.markov_blanket(0) == set()

    def test_markov_blanket_diamond(self, diamond_4):
        # MB(1) = {0(parent), 3(child), 2(co-parent of 3)}
        assert diamond_4.markov_blanket(1) == {0, 2, 3}


# ===================================================================
# Topological sort & cycle detection
# ===================================================================


class TestTopologicalSort:
    def test_chain_order(self, chain_3):
        order = chain_3.topological_sort()
        assert order.index(0) < order.index(1) < order.index(2)

    def test_fork_order(self, fork_3):
        order = fork_3.topological_sort()
        assert order.index(0) < order.index(1)
        assert order.index(0) < order.index(2)

    def test_diamond_order(self, diamond_4):
        order = diamond_4.topological_sort()
        assert order.index(0) < order.index(1)
        assert order.index(0) < order.index(2)
        assert order.index(1) < order.index(3)
        assert order.index(2) < order.index(3)

    def test_cycle_raises(self):
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        scm = StructuralCausalModel(adj)
        with pytest.raises(ValueError, match="cycle"):
            scm.topological_sort()

    def test_is_dag(self, chain_3):
        assert chain_3.is_dag()

    def test_has_cycle_false(self, chain_3):
        assert not chain_3.has_cycle()

    def test_has_cycle_true(self):
        adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        scm = StructuralCausalModel(adj)
        assert scm.has_cycle()

    def test_single_node_dag(self, single_node):
        assert single_node.is_dag()
        assert single_node.topological_sort() == [0]

    def test_empty_graph_dag(self, empty_3):
        assert empty_3.is_dag()
        assert len(empty_3.topological_sort()) == 3

    def test_topo_sort_caching(self, chain_3):
        o1 = chain_3.topological_sort()
        o2 = chain_3.topological_sort()
        assert o1 == o2
        # Modifying returned list should not affect cache
        o1.reverse()
        assert chain_3.topological_sort() != o1


# ===================================================================
# d-separation
# ===================================================================


class TestDSeparation:
    def test_chain_unconditional(self, chain_3):
        # 0 — 1 — 2: 0 and 2 d-connected unconditionally
        assert not chain_3.d_separation({0}, {2}, set())

    def test_chain_conditioned_on_middle(self, chain_3):
        # 0 ⊥ 2 | 1
        assert chain_3.d_separation({0}, {2}, {1})

    def test_fork_unconditional(self, fork_3):
        # 1 ← 0 → 2: 1 and 2 d-connected unconditionally
        assert not fork_3.d_separation({1}, {2}, set())

    def test_fork_conditioned_on_common_cause(self, fork_3):
        # 1 ⊥ 2 | 0
        assert fork_3.d_separation({1}, {2}, {0})

    def test_collider_unconditional(self, collider_3):
        # 0 → 2 ← 1: 0 ⊥ 1 unconditionally
        assert collider_3.d_separation({0}, {1}, set())

    def test_collider_conditioned_on_collider(self, collider_3):
        # conditioning on collider 2 opens path
        assert not collider_3.d_separation({0}, {1}, {2})

    def test_overlapping_xy_returns_false(self, chain_3):
        assert not chain_3.d_separation({0}, {0}, set())

    def test_diamond_dsep(self, diamond_4):
        # 1 ⊥ 2 | 0
        assert diamond_4.d_separation({1}, {2}, {0})
        # 1 and 2 d-connected given 3 (collider opened)
        assert not diamond_4.d_separation({1}, {2}, {3})


# ===================================================================
# SHD
# ===================================================================


class TestSHD:
    def test_same_graph_zero(self, chain_3):
        assert chain_3.structural_hamming_distance(chain_3) == 0

    def test_one_edge_added(self, chain_3):
        adj2 = chain_3.adjacency_matrix.copy()
        adj2[0, 2] = 1
        other = StructuralCausalModel(adj2)
        assert chain_3.structural_hamming_distance(other) == 1

    def test_one_edge_reversed(self):
        adj1 = np.array([[0, 1], [0, 0]], dtype=float)
        adj2 = np.array([[0, 0], [1, 0]], dtype=float)
        m1 = StructuralCausalModel(adj1)
        m2 = StructuralCausalModel(adj2)
        assert m1.structural_hamming_distance(m2) == 1

    def test_static_shd(self):
        a1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        a2 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        assert StructuralCausalModel.shd(a1, a2) == 1

    def test_size_mismatch_raises(self):
        m1 = StructuralCausalModel(np.zeros((2, 2)))
        m2 = StructuralCausalModel(np.zeros((3, 3)))
        with pytest.raises(ValueError, match="different sizes"):
            m1.structural_hamming_distance(m2)


# ===================================================================
# CPDAG & v-structures
# ===================================================================


class TestCPDAG:
    def test_chain_cpdag_undirected(self, chain_3):
        # Chain 0→1→2 has no v-structure → all edges undirected in CPDAG
        cpdag = chain_3.to_cpdag()
        assert cpdag[0, 1] == 1 and cpdag[1, 0] == 1
        assert cpdag[1, 2] == 1 and cpdag[2, 1] == 1

    def test_collider_cpdag_directed(self, collider_3):
        # 0→2←1 is a v-structure → both edges compelled
        cpdag = collider_3.to_cpdag()
        assert cpdag[0, 2] == 1 and cpdag[2, 0] == 0
        assert cpdag[1, 2] == 1 and cpdag[2, 1] == 0

    def test_v_structures_collider(self, collider_3):
        vs = collider_3.v_structures()
        assert len(vs) == 1
        parent1, collider_node, parent2 = vs[0]
        assert collider_node == 2
        assert {parent1, parent2} == {0, 1}

    def test_v_structures_chain_empty(self, chain_3):
        assert chain_3.v_structures() == []

    def test_v_structures_diamond(self, diamond_4):
        # 1→3←2, and 1,2 not adjacent → v-structure at 3
        vs = diamond_4.v_structures()
        assert len(vs) == 1
        assert vs[0] == (1, 3, 2)

    def test_cpdag_cycle_raises(self):
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        scm = StructuralCausalModel(adj)
        with pytest.raises(ValueError, match="DAG"):
            scm.to_cpdag()


# ===================================================================
# Moral graph
# ===================================================================


class TestMoralGraph:
    def test_chain_moral(self, chain_3):
        moral = chain_3.moral_graph()
        assert moral[0, 1] == 1 and moral[1, 0] == 1
        assert moral[1, 2] == 1 and moral[2, 1] == 1
        assert moral[0, 2] == 0

    def test_collider_moral_marries_parents(self, collider_3):
        moral = collider_3.moral_graph()
        # Parents 0 and 1 of collider 2 should be married
        assert moral[0, 1] == 1 and moral[1, 0] == 1

    def test_moral_no_self_loops(self, diamond_4):
        moral = diamond_4.moral_graph()
        np.testing.assert_array_equal(np.diag(moral), 0)

    def test_moral_symmetric(self, diamond_4):
        moral = diamond_4.moral_graph()
        np.testing.assert_array_equal(moral, moral.T)


# ===================================================================
# Graph metrics
# ===================================================================


class TestGraphMetrics:
    def test_in_degree_single(self, chain_3):
        assert chain_3.in_degree(0) == 0
        assert chain_3.in_degree(1) == 1
        assert chain_3.in_degree(2) == 1

    def test_out_degree_single(self, fork_3):
        assert fork_3.out_degree(0) == 2
        assert fork_3.out_degree(1) == 0

    def test_in_degree_all(self, chain_3):
        deg = chain_3.in_degree()
        np.testing.assert_array_equal(deg, [0, 1, 1])

    def test_out_degree_all(self, chain_3):
        deg = chain_3.out_degree()
        np.testing.assert_array_equal(deg, [1, 1, 0])

    def test_density_empty(self, empty_3):
        assert empty_3.density() == 0.0

    def test_density_complete(self, complete_dag_4):
        expected = 6 / (4 * 3)  # 6 edges, 12 max
        assert complete_dag_4.density() == pytest.approx(expected)

    def test_density_single_node(self, single_node):
        assert single_node.density() == 0.0

    def test_diameter_chain(self, chain_3):
        assert chain_3.diameter() == 2

    def test_diameter_single_node(self, single_node):
        assert single_node.diameter() == 0

    def test_diameter_disconnected(self, empty_3):
        assert empty_3.diameter() == -1

    def test_roots(self, chain_3, diamond_4):
        assert chain_3.roots() == [0]
        assert diamond_4.roots() == [0]

    def test_leaves(self, chain_3, diamond_4):
        assert chain_3.leaves() == [2]
        assert diamond_4.leaves() == [3]

    def test_roots_empty(self, empty_3):
        assert sorted(empty_3.roots()) == [0, 1, 2]

    def test_leaves_empty(self, empty_3):
        assert sorted(empty_3.leaves()) == [0, 1, 2]


# ===================================================================
# All paths
# ===================================================================


class TestAllPaths:
    def test_chain_directed(self, chain_3):
        paths = chain_3.all_paths(0, 2)
        assert paths == [[0, 1, 2]]

    def test_diamond_directed(self, diamond_4):
        paths = diamond_4.all_paths(0, 3)
        assert len(paths) == 2
        path_sets = {tuple(p) for p in paths}
        assert (0, 1, 3) in path_sets
        assert (0, 2, 3) in path_sets

    def test_no_path(self, chain_3):
        assert chain_3.all_paths(2, 0) == []

    def test_undirected(self, chain_3):
        paths = chain_3.all_paths(2, 0, undirected=True)
        assert len(paths) == 1
        assert paths[0] == [2, 1, 0]

    def test_max_length(self, diamond_4):
        # max_length=2 should allow paths of up to 2 edges (3 nodes)
        paths = diamond_4.all_paths(0, 3, max_length=2)
        assert all(len(p) <= 3 for p in paths)

    def test_max_length_zero(self, chain_3):
        assert chain_3.all_paths(0, 2, max_length=0) == []


# ===================================================================
# Intervention (do-calculus)
# ===================================================================


class TestIntervention:
    def test_do_removes_incoming_edges(self, chain_3):
        mutilated = chain_3.do_intervention({1: 5.0})
        assert mutilated.parents(1) == []
        # Outgoing edge 1→2 remains
        assert mutilated.has_edge(1, 2)

    def test_do_preserves_other_edges(self, diamond_4):
        mutilated = diamond_4.do_intervention({1: 0.0})
        assert mutilated.has_edge(0, 2)
        assert mutilated.has_edge(2, 3)
        assert mutilated.has_edge(1, 3)
        assert not mutilated.has_edge(0, 1)

    def test_do_returns_new_scm(self, chain_3):
        mutilated = chain_3.do_intervention({1: 0.0})
        assert chain_3.has_edge(0, 1)  # original unmodified


# ===================================================================
# Sampling
# ===================================================================


class TestSampling:
    def test_sample_shape(self, chain_3):
        data = chain_3.sample(100, rng=np.random.default_rng(42))
        assert data.shape == (100, 3)

    def test_sample_reproducible(self, chain_3):
        d1 = chain_3.sample(50, rng=np.random.default_rng(0))
        d2 = chain_3.sample(50, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(d1, d2)

    def test_sample_negative_n_raises(self, chain_3):
        with pytest.raises(ValueError, match="n must be > 0"):
            chain_3.sample(0)

    def test_sample_with_intervention(self, chain_3):
        data = chain_3.sample(200, interventions={1: 5.0}, rng=np.random.default_rng(7))
        # Intervened variable should be constant
        np.testing.assert_array_almost_equal(data[:, 1], 5.0)

    def test_sample_correlation_structure(self, fork_3):
        rng = np.random.default_rng(123)
        data = fork_3.sample(5000, rng=rng)
        corr = np.corrcoef(data.T)
        # Children of common cause should be correlated
        assert abs(corr[1, 2]) > 0.1

    def test_sample_single_node(self, single_node):
        data = single_node.sample(10, rng=np.random.default_rng(0))
        assert data.shape == (10, 1)


# ===================================================================
# Fit from data
# ===================================================================


class TestFitFromData:
    def test_fit_recovers_coefficients(self):
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        coefs = np.array([[0, 0.8, 0], [0, 0, -0.6], [0, 0, 0]], dtype=float)
        true_scm = StructuralCausalModel(
            adj, regression_coefficients=coefs,
            residual_variances=np.array([1.0, 0.5, 0.5]),
        )
        data = true_scm.sample(10000, rng=np.random.default_rng(42))
        fitted = StructuralCausalModel.fit_from_data(data, adj)
        np.testing.assert_allclose(
            fitted.regression_coefficients[0, 1], 0.8, atol=0.05
        )
        np.testing.assert_allclose(
            fitted.regression_coefficients[1, 2], -0.6, atol=0.05
        )

    def test_fit_sample_size(self):
        adj = np.zeros((2, 2))
        data = np.random.default_rng(0).standard_normal((50, 2))
        fitted = StructuralCausalModel.fit_from_data(data, adj)
        assert fitted.sample_size == 50

    def test_fit_shape_mismatch_raises(self):
        adj = np.zeros((3, 3))
        data = np.random.default_rng(0).standard_normal((10, 2))
        with pytest.raises(ValueError):
            StructuralCausalModel.fit_from_data(data, adj)


# ===================================================================
# Subgraph & marginalize
# ===================================================================


class TestSubgraphMarginalize:
    def test_subgraph_preserves_edges(self, diamond_4):
        sub = diamond_4.subgraph([0, 1, 3])
        assert sub.num_variables == 3
        assert sub.has_edge(0, 1)
        assert sub.has_edge(1, 2)  # 3 → index 2 in subgraph
        assert not sub.has_edge(0, 2)  # no direct 0→3 in diamond

    def test_subgraph_names(self, chain_3):
        sub = chain_3.subgraph([0, 2])
        assert sub.variable_names == ["A", "C"]

    def test_subgraph_duplicate_raises(self, chain_3):
        with pytest.raises(ValueError, match="Duplicate"):
            chain_3.subgraph([0, 0])

    def test_marginalize_adds_transitive_edges(self, chain_3):
        # keep 0 and 2: 0 is ancestor of 2 → edge 0→2
        marg = chain_3.marginalize([0, 2])
        assert marg.num_variables == 2
        assert marg.has_edge(0, 1)  # 2 maps to index 1

    def test_marginalize_diamond(self, diamond_4):
        # keep 0 and 3: 0 ancestor of 3 → direct edge
        marg = diamond_4.marginalize([0, 3])
        assert marg.has_edge(0, 1)
        assert marg.num_edges == 1


# ===================================================================
# Serialization
# ===================================================================


class TestSerialization:
    def test_to_dict_roundtrip(self, chain_3):
        d = chain_3.to_dict()
        restored = StructuralCausalModel.from_dict(d)
        assert restored == chain_3

    def test_to_json_roundtrip(self, diamond_4):
        j = diamond_4.to_json()
        restored = StructuralCausalModel.from_json(j)
        assert restored == diamond_4

    def test_json_is_valid(self, chain_3):
        j = chain_3.to_json()
        parsed = json.loads(j)
        assert "adjacency_matrix" in parsed
        assert "variable_names" in parsed

    def test_from_adjacency_matrix(self):
        adj = np.array([[0, 1], [0, 0]], dtype=float)
        scm = StructuralCausalModel.from_adjacency_matrix(adj, ["U", "V"])
        assert scm.variable_names == ["U", "V"]
        assert scm.has_edge(0, 1)

    def test_to_dict_keys(self, chain_3):
        d = chain_3.to_dict()
        expected_keys = {
            "adjacency_matrix",
            "variable_names",
            "regression_coefficients",
            "residual_variances",
            "sample_size",
        }
        assert set(d.keys()) == expected_keys


# ===================================================================
# Edge set
# ===================================================================


class TestEdgeSet:
    def test_edge_set(self, chain_3):
        assert chain_3.edge_set() == {(0, 1), (1, 2)}

    def test_named_edge_set(self, chain_3):
        assert chain_3.named_edge_set() == {("A", "B"), ("B", "C")}

    def test_empty_edge_set(self, empty_3):
        assert empty_3.edge_set() == set()


# ===================================================================
# Parametrized tests
# ===================================================================


@pytest.mark.parametrize(
    "adj, expected_dag",
    [
        (np.zeros((3, 3)), True),
        (np.eye(3), False),  # self-loops
        (np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), True),
        (np.array([[0, 1], [1, 0]]), False),
    ],
    ids=["empty", "self-loops", "chain", "2-cycle"],
)
def test_is_dag_parametrized(adj, expected_dag):
    scm = StructuralCausalModel(adj.astype(float))
    assert scm.is_dag() == expected_dag


@pytest.mark.parametrize(
    "n_vars",
    [1, 2, 5, 10],
    ids=["1var", "2var", "5var", "10var"],
)
def test_empty_graph_properties(n_vars):
    scm = StructuralCausalModel(np.zeros((n_vars, n_vars)))
    assert scm.num_variables == n_vars
    assert scm.num_edges == 0
    assert scm.is_dag()
    assert len(scm.topological_sort()) == n_vars
    assert len(scm.roots()) == n_vars
    assert len(scm.leaves()) == n_vars


@pytest.mark.parametrize(
    "graph_name, adj, expected_vstruct_count",
    [
        ("chain", np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), 0),
        ("fork", np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]]), 0),
        ("collider", np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]), 1),
    ],
)
def test_v_structure_count_parametrized(graph_name, adj, expected_vstruct_count):
    scm = StructuralCausalModel(adj.astype(float))
    assert len(scm.v_structures()) == expected_vstruct_count


@pytest.mark.parametrize(
    "x, y, z, expected",
    [
        ({0}, {2}, set(), False),       # chain unconditional
        ({0}, {2}, {1}, True),          # chain conditioned on middle
    ],
)
def test_dsep_chain_parametrized(x, y, z, expected):
    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    scm = StructuralCausalModel(adj)
    assert scm.d_separation(x, y, z) == expected


# ===================================================================
# Complete DAG stress
# ===================================================================


class TestCompleteDag:
    def test_num_edges(self, complete_dag_4):
        assert complete_dag_4.num_edges == 6

    def test_topo_sort_respects_all(self, complete_dag_4):
        order = complete_dag_4.topological_sort()
        for i in range(4):
            for j in range(i + 1, 4):
                assert order.index(i) < order.index(j)

    def test_ancestors_of_last(self, complete_dag_4):
        assert complete_dag_4.ancestors(3) == {0, 1, 2}

    def test_descendants_of_first(self, complete_dag_4):
        assert complete_dag_4.descendants(0) == {1, 2, 3}


# ===================================================================
# Copy & equality
# ===================================================================


class TestCopyEquality:
    def test_copy_is_equal(self, diamond_4):
        c = diamond_4.copy()
        assert c == diamond_4

    def test_copy_is_independent(self, diamond_4):
        c = diamond_4.copy()
        c.add_edge(0, 3)
        assert c != diamond_4

    def test_repr(self, chain_3):
        r = repr(chain_3)
        assert "StructuralCausalModel" in r
        assert "p=3" in r
        assert "edges=2" in r
