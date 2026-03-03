"""Comprehensive tests for the DAG class in causal_qd.core.dag."""

from __future__ import annotations

import copy
from typing import Set

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from causal_qd.core.dag import DAG, DAGError


# ===================================================================
# Construction Tests
# ===================================================================


class TestConstruction:
    """Tests for DAG construction and factory methods."""

    def test_create_empty_dag(self, empty_dag: DAG) -> None:
        assert empty_dag.num_nodes == 5
        assert empty_dag.num_edges == 0
        assert_array_equal(empty_dag.adjacency, np.zeros((5, 5), dtype=np.int8))
        assert empty_dag.edges == []
        assert empty_dag.node_list == [0, 1, 2, 3, 4]

    def test_create_from_adjacency(self, small_dag: DAG) -> None:
        adj = small_dag.adjacency
        reconstructed = DAG.from_adjacency(adj)
        assert reconstructed == small_dag

    def test_create_from_adjacency_matrix(self, small_dag: DAG) -> None:
        adj = small_dag.adjacency_matrix
        reconstructed = DAG.from_adjacency_matrix(adj)
        assert reconstructed == small_dag

    def test_from_edge_list(self) -> None:
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        dag = DAG.from_edge_list(4, edges)
        assert dag.num_nodes == 4
        assert dag.num_edges == 4
        assert dag.has_edge(0, 1)
        assert dag.has_edge(1, 2)
        assert dag.has_edge(2, 3)
        assert dag.has_edge(0, 3)
        assert not dag.has_edge(3, 0)

    def test_from_edges_alias(self) -> None:
        edges = [(0, 2), (1, 2)]
        dag1 = DAG.from_edges(3, edges)
        dag2 = DAG.from_edge_list(3, edges)
        assert dag1 == dag2

    def test_empty_factory(self) -> None:
        dag = DAG.empty(7)
        assert dag.num_nodes == 7
        assert dag.num_edges == 0
        assert dag.validate()

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 50])
    def test_empty_various_sizes(self, n: int) -> None:
        dag = DAG.empty(n)
        assert dag.num_nodes == n
        assert dag.num_edges == 0

    def test_non_square_raises(self) -> None:
        with pytest.raises(DAGError, match="square"):
            DAG(np.zeros((3, 4), dtype=np.int8))

    def test_cyclic_adjacency_raises(self) -> None:
        adj = np.zeros((3, 3), dtype=np.int8)
        adj[0, 1] = 1
        adj[1, 2] = 1
        adj[2, 0] = 1
        with pytest.raises(DAGError, match="cycle"):
            DAG(adj)

    def test_self_loop_raises(self) -> None:
        adj = np.zeros((3, 3), dtype=np.int8)
        adj[0, 0] = 1
        with pytest.raises(DAGError, match="cycle"):
            DAG(adj)


# ===================================================================
# Edge Mutation Tests
# ===================================================================


class TestEdgeMutations:
    """Tests for add_edge, remove_edge, reverse_edge."""

    def test_add_edge_valid(self, empty_dag: DAG) -> None:
        empty_dag.add_edge(0, 1)
        assert empty_dag.has_edge(0, 1)
        assert empty_dag.num_edges == 1
        empty_dag.add_edge(1, 2)
        assert empty_dag.has_edge(1, 2)
        assert empty_dag.num_edges == 2

    def test_add_edge_idempotent(self, small_dag: DAG) -> None:
        n_edges_before = small_dag.num_edges
        small_dag.add_edge(0, 1)  # already exists
        assert small_dag.num_edges == n_edges_before

    def test_add_edge_creates_cycle(self, small_dag: DAG) -> None:
        # small_dag: 0→1→2→3→4, adding 4→0 creates cycle
        with pytest.raises(DAGError, match="cycle"):
            small_dag.add_edge(4, 0)

    def test_add_edge_creates_cycle_short(self) -> None:
        dag = DAG.from_edge_list(3, [(0, 1), (1, 2)])
        with pytest.raises(DAGError, match="cycle"):
            dag.add_edge(2, 0)

    def test_add_self_loop_raises(self, empty_dag: DAG) -> None:
        with pytest.raises(DAGError, match="Self-loop"):
            empty_dag.add_edge(0, 0)

    def test_remove_edge(self, small_dag: DAG) -> None:
        assert small_dag.has_edge(0, 1)
        small_dag.remove_edge(0, 1)
        assert not small_dag.has_edge(0, 1)
        assert small_dag.num_edges == 3

    def test_remove_edge_nonexistent(self, empty_dag: DAG) -> None:
        # Should be no-op, no error
        empty_dag.remove_edge(0, 1)
        assert empty_dag.num_edges == 0

    def test_reverse_edge_valid(self) -> None:
        # 0→1→2, reverse 0→1 to get 1→0, 1→2 — still acyclic
        dag = DAG.from_edge_list(3, [(0, 1), (1, 2)])
        dag.reverse_edge(0, 1)
        assert not dag.has_edge(0, 1)
        assert dag.has_edge(1, 0)
        assert dag.has_edge(1, 2)
        assert dag.validate()

    def test_reverse_edge_creates_cycle(self) -> None:
        # 0→1→2→3, reverse 0→1 would give 1→0 and path 1→2→3, no cycle
        # But 0→1, 0→2, 1→2: reverse 1→2 gives 2→1 plus 0→2→1→... 0→1 already
        # exists? No. Let's build: 0→1, 1→2, 0→2. Reverse 0→2 gives 2→0 plus
        # path 0→1→2→0 = cycle.
        dag = DAG.from_edge_list(3, [(0, 1), (1, 2), (0, 2)])
        with pytest.raises(DAGError, match="cycle"):
            dag.reverse_edge(0, 2)

    def test_reverse_nonexistent_edge_raises(self, empty_dag: DAG) -> None:
        with pytest.raises(DAGError, match="does not exist"):
            empty_dag.reverse_edge(0, 1)

    def test_add_remove_roundtrip(self, empty_dag: DAG) -> None:
        empty_dag.add_edge(2, 4)
        assert empty_dag.has_edge(2, 4)
        empty_dag.remove_edge(2, 4)
        assert not empty_dag.has_edge(2, 4)
        assert empty_dag.num_edges == 0


# ===================================================================
# Topological Sort Tests
# ===================================================================


class TestTopologicalSort:
    """Tests for topological ordering."""

    def test_topological_sort_correctness(self, small_dag: DAG) -> None:
        order = small_dag.topological_sort()
        pos = {node: i for i, node in enumerate(order)}
        for src, tgt in small_dag.edges:
            assert pos[src] < pos[tgt], (
                f"Edge {src}→{tgt} violates topological order"
            )

    def test_topological_sort_correctness_medium(self, medium_dag: DAG) -> None:
        order = medium_dag.topological_sort()
        pos = {node: i for i, node in enumerate(order)}
        for src, tgt in medium_dag.edges:
            assert pos[src] < pos[tgt]

    def test_topological_sort_correctness_complete(self, complete_dag: DAG) -> None:
        order = complete_dag.topological_sort()
        pos = {node: i for i, node in enumerate(order)}
        for src, tgt in complete_dag.edges:
            assert pos[src] < pos[tgt]

    def test_topological_sort_covers_all_nodes(self, medium_dag: DAG) -> None:
        order = medium_dag.topological_sort()
        assert set(order) == set(range(medium_dag.num_nodes))
        assert len(order) == medium_dag.num_nodes

    def test_topological_sort_empty_dag(self, empty_dag: DAG) -> None:
        order = empty_dag.topological_sort()
        assert set(order) == set(range(5))

    def test_topological_order_cached(self, small_dag: DAG) -> None:
        order1 = small_dag.topological_order
        order2 = small_dag.topological_order
        assert order1 == order2

    def test_topological_order_invalidated_on_mutation(self, empty_dag: DAG) -> None:
        _ = empty_dag.topological_order
        empty_dag.add_edge(0, 1)
        order = empty_dag.topological_order
        pos = {node: i for i, node in enumerate(order)}
        assert pos[0] < pos[1]


# ===================================================================
# Node Query Tests
# ===================================================================


class TestNodeQueries:
    """Tests for parents, children, ancestors, descendants, etc."""

    def test_parents_children_consistency(self, medium_dag: DAG) -> None:
        for node in range(medium_dag.num_nodes):
            for parent in medium_dag.parents(node):
                assert node in medium_dag.children(parent), (
                    f"Node {node} has parent {parent} but {parent} doesn't list {node} as child"
                )
            for child in medium_dag.children(node):
                assert node in medium_dag.parents(child), (
                    f"Node {node} has child {child} but {child} doesn't list {node} as parent"
                )

    def test_parents_children_chain(self, small_dag: DAG) -> None:
        # 0→1→2→3→4
        assert small_dag.parents(0) == frozenset()
        assert small_dag.parents(1) == frozenset({0})
        assert small_dag.parents(2) == frozenset({1})
        assert small_dag.children(0) == frozenset({1})
        assert small_dag.children(4) == frozenset()

    def test_parents_children_empty(self, empty_dag: DAG) -> None:
        for node in range(5):
            assert empty_dag.parents(node) == frozenset()
            assert empty_dag.children(node) == frozenset()

    def test_ancestors_descendants(self, small_dag: DAG) -> None:
        # Chain: 0→1→2→3→4
        assert small_dag.ancestors(0) == frozenset()
        assert small_dag.ancestors(4) == frozenset({0, 1, 2, 3})
        assert small_dag.ancestors(2) == frozenset({0, 1})
        assert small_dag.descendants(0) == frozenset({1, 2, 3, 4})
        assert small_dag.descendants(4) == frozenset()
        assert small_dag.descendants(2) == frozenset({3, 4})

    def test_ancestors_descendants_medium(self, medium_dag: DAG) -> None:
        # Node 9 is a leaf: ancestors should include all on path from root
        anc_9 = medium_dag.ancestors(9)
        assert 8 in anc_9
        assert 7 in anc_9
        assert 0 in anc_9
        # Node 0 is a root: descendants should include many
        desc_0 = medium_dag.descendants(0)
        assert 9 in desc_0
        assert 1 in desc_0
        assert 3 in desc_0

    def test_ancestors_excludes_self(self, small_dag: DAG) -> None:
        for node in range(5):
            assert node not in small_dag.ancestors(node)

    def test_descendants_excludes_self(self, small_dag: DAG) -> None:
        for node in range(5):
            assert node not in small_dag.descendants(node)

    def test_neighbors(self, small_dag: DAG) -> None:
        # Node 2: parent=1, child=3
        assert small_dag.neighbors(2) == {1, 3}
        # Node 0: no parent, child=1
        assert small_dag.neighbors(0) == {1}

    def test_in_out_degree(self, small_dag: DAG) -> None:
        # Chain: 0→1→2→3→4
        assert small_dag.in_degree(0) == 0
        assert small_dag.out_degree(0) == 1
        assert small_dag.in_degree(2) == 1
        assert small_dag.out_degree(2) == 1
        assert small_dag.in_degree(4) == 1
        assert small_dag.out_degree(4) == 0

    def test_degree(self, small_dag: DAG) -> None:
        assert small_dag.degree(0) == 1  # out=1
        assert small_dag.degree(2) == 2  # in=1, out=1
        assert small_dag.degree(4) == 1  # in=1

    @pytest.mark.parametrize("node", [0, 1, 2, 3, 4])
    def test_degree_equals_in_plus_out(self, small_dag: DAG, node: int) -> None:
        assert small_dag.degree(node) == small_dag.in_degree(node) + small_dag.out_degree(node)


# ===================================================================
# d-Separation Tests
# ===================================================================


class TestDSeparation:
    """Tests for d_separated using Bayes-Ball."""

    def test_d_separation_basic_cases(self) -> None:
        # Chain: 0→1→2
        chain = DAG.from_edge_list(3, [(0, 1), (1, 2)])
        # Without conditioning: 0 and 2 are dependent
        assert not chain.d_separated({0}, {2}, set())
        # Conditioning on middle blocks chain
        assert chain.d_separated({0}, {2}, {1})

    def test_d_separation_fork(self) -> None:
        # Fork: 1←0→2
        fork = DAG.from_edge_list(3, [(0, 1), (0, 2)])
        # Without conditioning: 1 and 2 are dependent (via common cause)
        assert not fork.d_separated({1}, {2}, set())
        # Conditioning on common cause blocks
        assert fork.d_separated({1}, {2}, {0})

    def test_d_separation_collider(self) -> None:
        # Collider: 0→2←1
        collider = DAG.from_edge_list(3, [(0, 2), (1, 2)])
        # Without conditioning: 0 and 1 are independent
        assert collider.d_separated({0}, {1}, set())
        # Conditioning on collider opens path
        assert not collider.d_separated({0}, {1}, {2})

    def test_d_separation_collider_descendant(self) -> None:
        # 0→2←1, 2→3: conditioning on descendant of collider also opens
        dag = DAG.from_edge_list(4, [(0, 2), (1, 2), (2, 3)])
        assert dag.d_separated({0}, {1}, set())
        assert not dag.d_separated({0}, {1}, {3})

    def test_d_separation_complex(self, medium_dag: DAG) -> None:
        # medium_dag: 0→1, 0→2, 1→3, 2→3, 3→4, 4→5, 4→6, 5→7, 6→7, 7→8, 8→9
        # Node 3 is collider of 1 and 2
        # Unconditionally, 1 and 2 are dependent (common parent 0, fork)
        assert not medium_dag.d_separated({1}, {2}, set())
        # Conditioning on 0 should block the fork path
        assert medium_dag.d_separated({1}, {2}, {0})
        # But conditioning on 0 and 3 should re-open via collider at 3
        assert not medium_dag.d_separated({1}, {2}, {0, 3})

    def test_d_separation_same_node(self, small_dag: DAG) -> None:
        # A node is never d-separated from itself
        assert not small_dag.d_separated({0}, {0}, set())

    def test_d_separation_disconnected_nodes(self) -> None:
        # Two isolated nodes
        dag = DAG.empty(3)
        dag.add_edge(0, 1)
        # Node 2 is disconnected from 0 and 1
        assert dag.d_separated({0}, {2}, set())
        assert dag.d_separated({1}, {2}, set())


# ===================================================================
# Structural Query Tests
# ===================================================================


class TestStructuralQueries:
    """Tests for v_structures, skeleton, moralize, subgraph, etc."""

    def test_v_structures_detection(self) -> None:
        # 0→2←1 is a v-structure (no edge between 0 and 1)
        dag = DAG.from_edge_list(3, [(0, 2), (1, 2)])
        vs = dag.v_structures()
        assert len(vs) == 1
        assert vs[0] == (0, 2, 1)

    def test_v_structures_non_collider(self) -> None:
        # 0→1→2: no v-structures
        dag = DAG.from_edge_list(3, [(0, 1), (1, 2)])
        assert dag.v_structures() == []

    def test_v_structures_with_adjacent_parents(self) -> None:
        # 0→1, 0→2, 1→2: both parents of 2 are adjacent, so no v-structure
        dag = DAG.from_edge_list(3, [(0, 1), (0, 2), (1, 2)])
        assert dag.v_structures() == []

    def test_v_structures_medium(self, medium_dag: DAG) -> None:
        # medium_dag has colliders at 3 (parents 1,2) and 7 (parents 5,6)
        vs = medium_dag.v_structures()
        collider_nodes = {v[1] for v in vs}
        assert 3 in collider_nodes
        assert 7 in collider_nodes

    def test_v_structures_canonical_ordering(self, medium_dag: DAG) -> None:
        vs = medium_dag.v_structures()
        for i, j, k in vs:
            assert i < k, f"v-structure ({i},{j},{k}) not canonically ordered"

    def test_skeleton(self, small_dag: DAG) -> None:
        skel = small_dag.skeleton()
        expected = np.zeros((5, 5), dtype=np.int8)
        # Chain 0-1-2-3-4 undirected
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            expected[a, b] = 1
            expected[b, a] = 1
        assert_array_equal(skel, expected)

    def test_skeleton_symmetric(self, medium_dag: DAG) -> None:
        skel = medium_dag.skeleton()
        assert_array_equal(skel, skel.T)

    def test_skeleton_empty(self, empty_dag: DAG) -> None:
        skel = empty_dag.skeleton()
        assert_array_equal(skel, np.zeros((5, 5), dtype=np.int8))

    def test_moralize_graph(self) -> None:
        # 0→2←1: moralize should marry parents 0 and 1
        dag = DAG.from_edge_list(3, [(0, 2), (1, 2)])
        moral = dag.moralize()
        # Should have edges: 0-2, 1-2, 0-1 (all undirected)
        assert moral[0, 2] == 1 and moral[2, 0] == 1
        assert moral[1, 2] == 1 and moral[2, 1] == 1
        assert moral[0, 1] == 1 and moral[1, 0] == 1  # married parents

    def test_moralize_chain_no_marriage(self) -> None:
        # Chain: 0→1→2 — no multiple parents, so moral = skeleton
        dag = DAG.from_edge_list(3, [(0, 1), (1, 2)])
        moral = dag.moralize()
        skel = dag.skeleton()
        assert_array_equal(moral, skel)

    def test_moralize_symmetric(self, medium_dag: DAG) -> None:
        moral = medium_dag.moral_graph()
        assert_array_equal(moral, moral.T)

    def test_moralize_alias(self, medium_dag: DAG) -> None:
        assert_array_equal(medium_dag.moralize(), medium_dag.moral_graph())

    def test_subgraph(self, medium_dag: DAG) -> None:
        sub = medium_dag.subgraph([0, 1, 2, 3])
        assert sub.num_nodes == 4
        # Edges within {0,1,2,3}: 0→1, 0→2, 1→3, 2→3
        assert sub.has_edge(0, 1)
        assert sub.has_edge(0, 2)
        assert sub.has_edge(1, 3)
        assert sub.has_edge(2, 3)
        assert sub.validate()

    def test_is_connected(self, small_dag: DAG) -> None:
        assert small_dag.is_connected()

    def test_is_connected_empty(self, empty_dag: DAG) -> None:
        # 5 isolated nodes are not connected
        assert not empty_dag.is_connected()

    def test_is_connected_single_node(self) -> None:
        dag = DAG.empty(1)
        assert dag.is_connected()

    def test_connected_components(self, empty_dag: DAG) -> None:
        comps = empty_dag.connected_components()
        assert len(comps) == 5
        all_nodes = set()
        for comp in comps:
            all_nodes.update(comp)
        assert all_nodes == set(range(5))

    def test_connected_components_connected(self, small_dag: DAG) -> None:
        comps = small_dag.connected_components()
        assert len(comps) == 1
        assert set(comps[0]) == set(range(5))

    def test_connected_components_partial(self) -> None:
        # 0→1, 2→3 — two components
        dag = DAG.from_edge_list(4, [(0, 1), (2, 3)])
        comps = dag.connected_components()
        assert len(comps) == 2
        comp_sets = [set(c) for c in comps]
        assert {0, 1} in comp_sets
        assert {2, 3} in comp_sets

    def test_longest_path(self, small_dag: DAG) -> None:
        # Chain 0→1→2→3→4 has longest path length 4
        assert small_dag.longest_path() == 4

    def test_longest_path_empty(self, empty_dag: DAG) -> None:
        assert empty_dag.longest_path() == 0

    def test_longest_path_complete(self, complete_dag: DAG) -> None:
        # Complete DAG with 5 nodes: longest path 0→1→2→3→4 = 4
        assert complete_dag.longest_path() == 4

    def test_longest_path_medium(self, medium_dag: DAG) -> None:
        # Path 0→1→3→4→5→7→8→9 = 7 edges or 0→2→3→4→5→7→8→9 = 7
        assert medium_dag.longest_path() == 7


# ===================================================================
# Property Tests
# ===================================================================


class TestProperties:
    """Tests for various DAG properties."""

    def test_adjacency_matrix_consistency(self, small_dag: DAG) -> None:
        adj = small_dag.adjacency
        adj2 = small_dag.adjacency_matrix
        assert_array_equal(adj, adj2)
        # adjacency returns a copy
        adj[0, 1] = 0
        assert_array_equal(small_dag.adjacency, small_dag.adjacency_matrix)
        assert small_dag.has_edge(0, 1)  # original unchanged

    def test_num_nodes_aliases(self, medium_dag: DAG) -> None:
        assert medium_dag.num_nodes == medium_dag.n_nodes == 10

    def test_num_edges_aliases(self, medium_dag: DAG) -> None:
        assert medium_dag.num_edges == medium_dag.n_edges == 11

    def test_edges_property(self, small_dag: DAG) -> None:
        edges = small_dag.edges
        assert set(edges) == {(0, 1), (1, 2), (2, 3), (3, 4)}

    def test_edge_list_alias(self, small_dag: DAG) -> None:
        assert small_dag.edges == small_dag.edge_list

    def test_node_list_property(self, medium_dag: DAG) -> None:
        assert medium_dag.node_list == list(range(10))

    @pytest.mark.parametrize(
        "n, n_edges, expected_density",
        [
            (2, 1, 0.5),
            (3, 3, 0.5),
            (5, 10, 0.5),
        ],
    )
    def test_density(self, n: int, n_edges: int, expected_density: float) -> None:
        adj = np.zeros((n, n), dtype=np.int8)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if count < n_edges:
                    adj[i, j] = 1
                    count += 1
        dag = DAG(adj)
        assert dag.density == pytest.approx(expected_density)

    def test_density_empty(self, empty_dag: DAG) -> None:
        assert empty_dag.density == pytest.approx(0.0)

    def test_density_complete(self, complete_dag: DAG) -> None:
        assert complete_dag.density == pytest.approx(0.5)


# ===================================================================
# Cycle Detection Tests
# ===================================================================


class TestCycleDetection:
    """Tests for has_cycle and is_acyclic."""

    def test_has_cycle_false(self, small_dag: DAG) -> None:
        assert not small_dag.has_cycle()

    def test_has_cycle_empty(self, empty_dag: DAG) -> None:
        assert not empty_dag.has_cycle()

    def test_is_acyclic_classmethod(self) -> None:
        adj = np.zeros((3, 3), dtype=np.int8)
        adj[0, 1] = 1
        adj[1, 2] = 1
        assert DAG.is_acyclic(adj)

    def test_is_acyclic_false(self) -> None:
        adj = np.zeros((3, 3), dtype=np.int8)
        adj[0, 1] = 1
        adj[1, 2] = 1
        adj[2, 0] = 1
        assert not DAG.is_acyclic(adj)

    @pytest.mark.parametrize(
        "edges",
        [
            [(0, 1)],
            [(0, 1), (1, 2)],
            [(0, 1), (0, 2), (1, 2)],
            [(0, 2), (1, 2)],
        ],
    )
    def test_is_acyclic_various(self, edges: list) -> None:
        adj = np.zeros((3, 3), dtype=np.int8)
        for i, j in edges:
            adj[i, j] = 1
        assert DAG.is_acyclic(adj)


# ===================================================================
# Conversion Tests
# ===================================================================


class TestConversions:
    """Tests for networkx roundtrip and CPDAG conversion."""

    def test_networkx_roundtrip(self, small_dag: DAG) -> None:
        nx_graph = small_dag.to_networkx()
        roundtrip = DAG.from_networkx(nx_graph)
        assert roundtrip == small_dag

    def test_networkx_roundtrip_medium(self, medium_dag: DAG) -> None:
        nx_graph = medium_dag.to_networkx()
        roundtrip = DAG.from_networkx(nx_graph)
        assert roundtrip == medium_dag

    def test_networkx_roundtrip_empty(self, empty_dag: DAG) -> None:
        nx_graph = empty_dag.to_networkx()
        roundtrip = DAG.from_networkx(nx_graph)
        assert roundtrip == empty_dag

    def test_to_networkx_nodes_and_edges(self, small_dag: DAG) -> None:
        G = small_dag.to_networkx()
        assert len(G.nodes()) == 5
        assert len(G.edges()) == 4
        assert (0, 1) in G.edges()

    def test_from_networkx_invalid_type(self) -> None:
        import networkx as nx

        with pytest.raises(DAGError, match="DiGraph"):
            DAG.from_networkx(nx.Graph())

    def test_to_cpdag_chain(self) -> None:
        # Chain 0→1→2: all edges reversible → CPDAG has both directions
        dag = DAG.from_edge_list(3, [(0, 1), (1, 2)])
        cpdag = dag.to_cpdag()
        # No v-structures ⇒ all edges are reversible ⇒ undirected in CPDAG
        assert cpdag[0, 1] == 1 and cpdag[1, 0] == 1
        assert cpdag[1, 2] == 1 and cpdag[2, 1] == 1

    def test_to_cpdag_collider(self) -> None:
        # 0→2←1: v-structure, edges are compelled
        dag = DAG.from_edge_list(3, [(0, 2), (1, 2)])
        cpdag = dag.to_cpdag()
        # Edges into collider are directed
        assert cpdag[0, 2] == 1 and cpdag[2, 0] == 0
        assert cpdag[1, 2] == 1 and cpdag[2, 1] == 0


# ===================================================================
# Copy / Equality / Hash Tests
# ===================================================================


class TestCopyEqualityHash:
    """Tests for copy, __eq__, __hash__, and related dunder methods."""

    def test_copy_independence(self, small_dag: DAG) -> None:
        dag_copy = small_dag.copy()
        assert dag_copy == small_dag
        # Mutating copy should not affect original
        dag_copy.add_edge(0, 2)
        assert dag_copy != small_dag
        assert small_dag.num_edges == 4
        assert dag_copy.num_edges == 5

    def test_deepcopy(self, small_dag: DAG) -> None:
        dag_copy = copy.deepcopy(small_dag)
        assert dag_copy == small_dag
        dag_copy.remove_edge(0, 1)
        assert dag_copy != small_dag

    def test_copy_module(self, medium_dag: DAG) -> None:
        dag_copy = copy.copy(medium_dag)
        assert dag_copy == medium_dag

    def test_equality_hash(self) -> None:
        dag1 = DAG.from_edge_list(3, [(0, 1), (1, 2)])
        dag2 = DAG.from_edge_list(3, [(0, 1), (1, 2)])
        dag3 = DAG.from_edge_list(3, [(0, 2), (1, 2)])
        assert dag1 == dag2
        assert dag1 != dag3
        assert hash(dag1) == hash(dag2)
        assert hash(dag1) != hash(dag3)

    def test_equality_different_type(self, small_dag: DAG) -> None:
        assert small_dag != "not a DAG"
        assert small_dag != 42

    def test_hash_usable_in_set(self) -> None:
        dag1 = DAG.from_edge_list(3, [(0, 1)])
        dag2 = DAG.from_edge_list(3, [(0, 1)])
        dag3 = DAG.from_edge_list(3, [(0, 2)])
        s = {dag1, dag2, dag3}
        assert len(s) == 2

    def test_repr(self, small_dag: DAG) -> None:
        r = repr(small_dag)
        assert "DAG" in r
        assert "5" in r
        assert "4" in r

    def test_len(self, small_dag: DAG) -> None:
        assert len(small_dag) == 5

    def test_contains(self, small_dag: DAG) -> None:
        assert (0, 1) in small_dag
        assert (1, 0) not in small_dag

    def test_iter(self, small_dag: DAG) -> None:
        nodes = list(small_dag)
        assert nodes == [0, 1, 2, 3, 4]


# ===================================================================
# Validation Tests
# ===================================================================


class TestValidation:
    """Tests for the validate method."""

    def test_validate_method(self, small_dag: DAG) -> None:
        assert small_dag.validate()

    def test_validate_empty(self, empty_dag: DAG) -> None:
        assert empty_dag.validate()

    def test_validate_complete(self, complete_dag: DAG) -> None:
        assert complete_dag.validate()

    def test_validate_medium(self, medium_dag: DAG) -> None:
        assert medium_dag.validate()

    def test_validate_after_mutations(self, empty_dag: DAG) -> None:
        empty_dag.add_edge(0, 1)
        empty_dag.add_edge(1, 2)
        assert empty_dag.validate()
        # Trigger topological order caching
        _ = empty_dag.topological_order
        assert empty_dag.validate()


# ===================================================================
# Random DAG Tests
# ===================================================================


class TestRandomDAG:
    """Property tests for randomly generated DAGs."""

    def test_random_dag_is_acyclic(self, rng: np.random.Generator) -> None:
        for i in range(100):
            seed_rng = np.random.default_rng(i)
            n = seed_rng.integers(3, 15)
            p = seed_rng.uniform(0.1, 0.6)
            dag = DAG.random_dag(int(n), float(p), rng=np.random.default_rng(i))
            assert not dag.has_cycle(), f"Random DAG {i} has a cycle"
            assert dag.validate(), f"Random DAG {i} fails validation"

    def test_random_dag_deterministic(self) -> None:
        dag1 = DAG.random_dag(10, 0.3, rng=np.random.default_rng(99))
        dag2 = DAG.random_dag(10, 0.3, rng=np.random.default_rng(99))
        assert dag1 == dag2

    def test_random_dag_respects_node_count(self) -> None:
        for n in [3, 5, 10, 20]:
            dag = DAG.random_dag(n, 0.3, rng=np.random.default_rng(0))
            assert dag.num_nodes == n

    def test_random_dag_edge_prob_zero(self) -> None:
        dag = DAG.random_dag(10, 0.0, rng=np.random.default_rng(0))
        assert dag.num_edges == 0

    @pytest.mark.parametrize("n_nodes", [2, 5, 8, 12])
    def test_random_dag_topological_order_valid(self, n_nodes: int) -> None:
        dag = DAG.random_dag(n_nodes, 0.4, rng=np.random.default_rng(42))
        order = dag.topological_sort()
        pos = {node: i for i, node in enumerate(order)}
        for src, tgt in dag.edges:
            assert pos[src] < pos[tgt]


# ===================================================================
# Edge Cases & Integration
# ===================================================================


class TestEdgeCases:
    """Edge cases and integration tests."""

    def test_single_node_dag(self) -> None:
        dag = DAG.empty(1)
        assert dag.num_nodes == 1
        assert dag.num_edges == 0
        assert dag.is_connected()
        assert dag.longest_path() == 0
        assert dag.topological_sort() == [0]
        assert dag.validate()

    def test_two_node_dag(self) -> None:
        dag = DAG.from_edge_list(2, [(0, 1)])
        assert dag.parents(1) == frozenset({0})
        assert dag.children(0) == frozenset({1})
        assert dag.ancestors(1) == frozenset({0})
        assert dag.descendants(0) == frozenset({1})

    def test_has_edge_boundary(self, complete_dag: DAG) -> None:
        for i in range(5):
            for j in range(5):
                if i < j:
                    assert complete_dag.has_edge(i, j)
                else:
                    assert not complete_dag.has_edge(i, j)

    def test_large_dag_validates(self, large_dag: DAG) -> None:
        assert large_dag.validate()
        assert not large_dag.has_cycle()
        order = large_dag.topological_sort()
        assert len(order) == 20

    @pytest.mark.parametrize(
        "n, edges",
        [
            (3, []),
            (3, [(0, 1)]),
            (4, [(0, 1), (2, 3)]),
            (5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
            (4, [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]),
        ],
    )
    def test_from_edge_list_parametrized(self, n: int, edges: list) -> None:
        dag = DAG.from_edge_list(n, edges)
        assert dag.num_nodes == n
        assert dag.num_edges == len(edges)
        for src, tgt in edges:
            assert dag.has_edge(src, tgt)
        assert dag.validate()

    def test_mutation_cache_invalidation(self) -> None:
        dag = DAG.from_edge_list(4, [(0, 1), (1, 2)])
        # Populate caches
        _ = dag.parents(2)
        _ = dag.children(1)
        _ = dag.topological_order
        # Mutate
        dag.add_edge(2, 3)
        # Caches should be fresh
        assert 2 in dag.parents(3)
        assert 3 in dag.children(2)
        order = dag.topological_order
        pos = {node: i for i, node in enumerate(order)}
        assert pos[2] < pos[3]

    def test_complete_dag_density(self, complete_dag: DAG) -> None:
        # 5 nodes, 10 forward edges, max = 20
        assert complete_dag.density == pytest.approx(0.5)

    def test_skeleton_of_complete(self, complete_dag: DAG) -> None:
        skel = complete_dag.skeleton()
        # Every pair connected
        for i in range(5):
            for j in range(5):
                if i != j:
                    assert skel[i, j] == 1

    def test_moralize_complete_unchanged(self, complete_dag: DAG) -> None:
        # All parents are already adjacent in the complete DAG
        moral = complete_dag.moralize()
        skel = complete_dag.skeleton()
        assert_array_equal(moral, skel)

    def test_subgraph_preserves_acyclicity(self, large_dag: DAG) -> None:
        nodes = [0, 3, 7, 12, 18]
        sub = large_dag.subgraph(nodes)
        assert sub.validate()

    def test_connected_components_cover_all_nodes(self, medium_dag: DAG) -> None:
        comps = medium_dag.connected_components()
        all_nodes = set()
        for comp in comps:
            all_nodes.update(comp)
        assert all_nodes == set(range(medium_dag.num_nodes))

    def test_ancestors_descendants_consistency(self, medium_dag: DAG) -> None:
        for node in range(medium_dag.num_nodes):
            for anc in medium_dag.ancestors(node):
                assert node in medium_dag.descendants(anc)
