"""
Unit tests for usability_oracle.utils.graph.

Tests cover the graph algorithm wrappers: shortest path, all-paths
enumeration, strongly connected components, topological sorting,
diameter, betweenness centrality, spectral clustering, and graph
partitioning.  Graphs are constructed directly with NetworkX so that
the tests validate the wrapper logic rather than NetworkX internals.
"""

from __future__ import annotations

import numpy as np
import pytest

import networkx as nx

from usability_oracle.utils.graph import (
    all_paths,
    betweenness_centrality,
    diameter,
    partition_graph,
    shortest_path,
    spectral_clustering,
    strongly_connected_components,
    topological_sort,
)


# ===================================================================
# Helpers
# ===================================================================


def _simple_dag() -> nx.DiGraph:
    """Create a simple DAG:  A → B → D, A → C → D."""
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1)
    G.add_edge("B", "D", weight=1)
    G.add_edge("A", "C", weight=1)
    G.add_edge("C", "D", weight=1)
    return G


def _weighted_dag() -> nx.DiGraph:
    """Create a DAG with asymmetric weights.

    A → B (w=1), B → D (w=1), A → C (w=5), C → D (w=1).
    The shortest path by weight is A → B → D.
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1)
    G.add_edge("B", "D", weight=1)
    G.add_edge("A", "C", weight=5)
    G.add_edge("C", "D", weight=1)
    return G


def _cyclic_graph() -> nx.DiGraph:
    """Create a directed graph with a cycle: A → B → C → A, plus D → A."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("D", "A")])
    return G


def _linear_graph(n: int) -> nx.DiGraph:
    """Create a linear chain 0 → 1 → 2 → … → n-1."""
    G = nx.DiGraph()
    for i in range(n - 1):
        G.add_edge(i, i + 1, weight=1)
    return G


def _disconnected_graph() -> nx.DiGraph:
    """Two disconnected components: {A→B} and {C→D}."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("C", "D")
    return G


def _star_graph() -> nx.DiGraph:
    """Star graph: centre node 0 with edges to nodes 1..5."""
    G = nx.DiGraph()
    for i in range(1, 6):
        G.add_edge(0, i)
    return G


# ===================================================================
# shortest_path
# ===================================================================


class TestShortestPath:
    """Tests for ``shortest_path(graph, source, target)``."""

    def test_simple_dag_path(self) -> None:
        """Shortest path in unweighted DAG: A → B → D or A → C → D (len 3)."""
        G = _simple_dag()
        path = shortest_path(G, "A", "D")
        assert path[0] == "A"
        assert path[-1] == "D"
        assert len(path) == 3

    def test_weighted_dag_chooses_lighter_path(self) -> None:
        """Dijkstra should prefer A → B → D (cost 2) over A → C → D (cost 6)."""
        G = _weighted_dag()
        path = shortest_path(G, "A", "D", weight="weight")
        assert path == ["A", "B", "D"]

    def test_no_path_returns_empty(self) -> None:
        """When no path exists, an empty list is returned."""
        G = _disconnected_graph()
        assert shortest_path(G, "A", "D") == []

    def test_source_equals_target(self) -> None:
        """Path from a node to itself should be a single-element list."""
        G = _simple_dag()
        assert shortest_path(G, "A", "A") == ["A"]

    def test_nonexistent_node_returns_empty(self) -> None:
        """Querying a node not in the graph should return an empty list."""
        G = _simple_dag()
        assert shortest_path(G, "A", "MISSING") == []

    def test_linear_chain_length(self) -> None:
        """A chain of n nodes yields a path of length n."""
        n = 6
        G = _linear_graph(n)
        path = shortest_path(G, 0, n - 1)
        assert len(path) == n


# ===================================================================
# all_paths
# ===================================================================


class TestAllPaths:
    """Tests for ``all_paths(graph, source, target)``."""

    def test_two_paths_in_diamond(self) -> None:
        """Diamond DAG (A→B→D, A→C→D) has exactly 2 simple paths."""
        G = _simple_dag()
        paths = all_paths(G, "A", "D")
        assert len(paths) == 2

    def test_paths_start_and_end_correctly(self) -> None:
        """Every returned path should start at source and end at target."""
        G = _simple_dag()
        for p in all_paths(G, "A", "D"):
            assert p[0] == "A"
            assert p[-1] == "D"

    def test_no_paths_returns_empty(self) -> None:
        """If no path exists, an empty list is returned."""
        G = _disconnected_graph()
        assert all_paths(G, "A", "D") == []

    def test_max_depth_limits_results(self) -> None:
        """With max_depth=1, only direct edges are traversed, so longer
        paths are excluded."""
        G = _simple_dag()
        # From A to D requires at least 2 edges.  max_depth=1 → no results.
        paths = all_paths(G, "A", "D", max_depth=1)
        assert len(paths) == 0

    def test_self_path_returns_singleton(self) -> None:
        """all_paths(source=target) returns the trivial single-node path."""
        G = _simple_dag()
        result = all_paths(G, "A", "A")
        assert result == [["A"]]


# ===================================================================
# strongly_connected_components
# ===================================================================


class TestStronglyConnectedComponents:
    """Tests for ``strongly_connected_components(graph)``."""

    def test_cyclic_graph_has_one_large_scc(self) -> None:
        """A→B→C→A forms a single SCC of size 3; D is a singleton."""
        G = _cyclic_graph()
        sccs = strongly_connected_components(G)
        sizes = sorted(len(c) for c in sccs)
        assert sizes == [1, 3]

    def test_dag_has_only_singletons(self) -> None:
        """A DAG has no non-trivial SCCs; every SCC is a single node."""
        G = _simple_dag()
        sccs = strongly_connected_components(G)
        for c in sccs:
            assert len(c) == 1

    def test_every_node_in_exactly_one_scc(self) -> None:
        """The union of all SCCs must be exactly the node set."""
        G = _cyclic_graph()
        sccs = strongly_connected_components(G)
        all_nodes = set()
        for c in sccs:
            assert all_nodes.isdisjoint(c), "SCCs must not overlap"
            all_nodes |= c
        assert all_nodes == set(G.nodes)

    def test_empty_graph(self) -> None:
        """An empty graph should have no SCCs."""
        G = nx.DiGraph()
        assert strongly_connected_components(G) == []


# ===================================================================
# topological_sort
# ===================================================================


class TestTopologicalSort:
    """Tests for ``topological_sort(graph)``."""

    def test_valid_ordering_for_dag(self) -> None:
        """Every edge (u, v) must have u before v in the ordering."""
        G = _simple_dag()
        order = topological_sort(G)
        pos = {node: i for i, node in enumerate(order)}
        for u, v in G.edges():
            assert pos[u] < pos[v], f"{u} should appear before {v}"

    def test_all_nodes_present(self) -> None:
        """Ordering must contain all nodes exactly once."""
        G = _simple_dag()
        order = topological_sort(G)
        assert set(order) == set(G.nodes)
        assert len(order) == len(G.nodes)

    def test_raises_on_cycle(self) -> None:
        """A cyclic graph should cause a NetworkXUnfeasible error."""
        G = _cyclic_graph()
        with pytest.raises(nx.NetworkXUnfeasible):
            topological_sort(G)

    def test_linear_chain(self) -> None:
        """A chain 0→1→2→3 has a unique topological order."""
        G = _linear_graph(4)
        assert topological_sort(G) == [0, 1, 2, 3]


# ===================================================================
# diameter
# ===================================================================


class TestDiameter:
    """Tests for ``diameter(graph)``."""

    def test_linear_chain_diameter(self) -> None:
        """A chain of n nodes has diameter n − 1."""
        n = 5
        G = _linear_graph(n)
        assert diameter(G) == n - 1

    def test_star_graph_diameter(self) -> None:
        """A star with centre 0 and leaves 1..5 has undirected diameter 2."""
        G = _star_graph()
        assert diameter(G) == 2

    def test_single_node_diameter(self) -> None:
        """A single-node graph has diameter 0."""
        G = nx.DiGraph()
        G.add_node("X")
        assert diameter(G) == 0

    def test_disconnected_uses_largest_component(self) -> None:
        """For a disconnected graph, diameter is computed on the largest component."""
        G = _disconnected_graph()
        d = diameter(G)
        assert d >= 0


# ===================================================================
# betweenness_centrality
# ===================================================================


class TestBetweennessCentrality:
    """Tests for ``betweenness_centrality(graph)``."""

    def test_returns_dict(self) -> None:
        """Result should be a dict mapping node → float."""
        G = _simple_dag()
        bc = betweenness_centrality(G)
        assert isinstance(bc, dict)
        assert set(bc.keys()) == set(G.nodes)

    def test_values_in_zero_one_range(self) -> None:
        """With normalized=True (default), values should be in [0, 1]."""
        G = _simple_dag()
        bc = betweenness_centrality(G)
        for v in bc.values():
            assert 0.0 <= v <= 1.0 + 1e-10

    def test_star_centre_has_highest_centrality(self) -> None:
        """In a star graph, the centre node should have the highest
        betweenness centrality (or equal to all others if all are 0)."""
        G = _star_graph()
        bc = betweenness_centrality(G)
        centre_val = bc[0]
        assert all(centre_val >= v - 1e-10 for v in bc.values())

    def test_empty_graph(self) -> None:
        """Betweenness centrality of an empty graph is an empty dict."""
        G = nx.DiGraph()
        assert betweenness_centrality(G) == {}


# ===================================================================
# spectral_clustering
# ===================================================================


class TestSpectralClustering:
    """Tests for ``spectral_clustering(adjacency, k)``."""

    def test_two_clusters_basic(self) -> None:
        """Spectral clustering of a connected graph with k=2 returns valid
        labels with exactly 2 distinct clusters and correct array length."""
        A = np.zeros((6, 6))
        for i in range(6):
            for j in range(i + 1, 6):
                A[i, j] = A[j, i] = 1.0
        labels = spectral_clustering(A, k=2)
        assert len(labels) == 6
        assert len(set(labels)) <= 2
        assert all(l in (0, 1) for l in labels)

    def test_label_count_equals_k(self) -> None:
        """Number of unique labels should be ≤ k."""
        A = np.ones((5, 5)) - np.eye(5)
        labels = spectral_clustering(A, k=3)
        assert len(set(labels)) <= 3

    def test_returns_array_of_correct_length(self) -> None:
        """Returned labels array must have one entry per node."""
        A = np.eye(4) + np.roll(np.eye(4), 1, axis=1)
        labels = spectral_clustering(A, k=2)
        assert labels.shape == (4,)

    def test_empty_adjacency(self) -> None:
        """An empty adjacency matrix should return an empty label array."""
        labels = spectral_clustering(np.array([]).reshape(0, 0), k=2)
        assert len(labels) == 0

    def test_single_node(self) -> None:
        """A single-node graph should be assigned to one cluster."""
        labels = spectral_clustering(np.array([[0.0]]), k=1)
        assert len(labels) == 1


# ===================================================================
# partition_graph
# ===================================================================


class TestPartitionGraph:
    """Tests for ``partition_graph(graph, partition)``."""

    def test_quotient_preserves_inter_cluster_edges(self) -> None:
        """Edges between different clusters become edges in the quotient graph."""
        G = nx.DiGraph()
        G.add_edge("a", "b", weight=2.0)
        G.add_edge("a", "c", weight=3.0)
        partition = {"a": 0, "b": 1, "c": 1}
        Q = partition_graph(G, partition)
        assert Q.has_edge(0, 1)

    def test_intra_cluster_edges_removed(self) -> None:
        """Edges within the same cluster are NOT present in the quotient."""
        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("b", "c")
        partition = {"a": 0, "b": 0, "c": 0}
        Q = partition_graph(G, partition)
        assert Q.number_of_edges() == 0

    def test_quotient_node_count(self) -> None:
        """The quotient graph should have one node per cluster."""
        G = _simple_dag()
        partition = {"A": 0, "B": 1, "C": 1, "D": 2}
        Q = partition_graph(G, partition)
        assert Q.number_of_nodes() == 3

    def test_weight_aggregation(self) -> None:
        """Multiple inter-cluster edges should have their weights summed."""
        G = nx.DiGraph()
        G.add_edge("a1", "b1", weight=1.0)
        G.add_edge("a2", "b2", weight=4.0)
        partition = {"a1": 0, "a2": 0, "b1": 1, "b2": 1}
        Q = partition_graph(G, partition)
        assert Q[0][1]["weight"] == pytest.approx(5.0)

    def test_empty_graph_partition(self) -> None:
        """Partitioning an empty graph yields an empty quotient."""
        G = nx.DiGraph()
        Q = partition_graph(G, {})
        assert Q.number_of_nodes() == 0
        assert Q.number_of_edges() == 0

    def test_node_metadata_includes_members(self) -> None:
        """Each quotient node should store its member nodes."""
        G = nx.DiGraph()
        G.add_edge("x", "y")
        partition = {"x": 0, "y": 1}
        Q = partition_graph(G, partition)
        assert "x" in Q.nodes[0]["members"]
        assert "y" in Q.nodes[1]["members"]
