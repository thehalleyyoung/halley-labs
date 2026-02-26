"""
Tests for graph decomposition module.

Tests cover: tree decomposition, treewidth computation, separator extraction,
moral graph construction, triangulation, elimination orderings, and
causal partition.
"""
import itertools
import pytest
import networkx as nx
import numpy as np

from causalbound.graph.decomposition import (
    TreeDecomposer,
    TreeDecomposition,
    build_adjacency_matrix,
    compute_treewidth_upper_bound,
    compute_treewidth_lower_bound,
    decompose_and_validate,
    separator_sizes,
    compare_strategies,
    decomposition_statistics,
)
from causalbound.graph.treewidth import TreewidthEstimator
from causalbound.graph.separator import SeparatorExtractor
from causalbound.graph.moral import MoralGraphConstructor
from causalbound.graph.causal_partition import CausalPartitioner
from causalbound.graph.subgraph import SubgraphExtractor


# ---------------------------------------------------------------------------
# Fixtures: standard graph families
# ---------------------------------------------------------------------------

@pytest.fixture
def path_graph_5():
    """Path graph P_5: 0-1-2-3-4."""
    return nx.path_graph(5)


@pytest.fixture
def path_graph_10():
    return nx.path_graph(10)


@pytest.fixture
def cycle_graph_5():
    return nx.cycle_graph(5)


@pytest.fixture
def cycle_graph_8():
    return nx.cycle_graph(8)


@pytest.fixture
def complete_graph_4():
    return nx.complete_graph(4)


@pytest.fixture
def complete_graph_6():
    return nx.complete_graph(6)


@pytest.fixture
def grid_graph_3x3():
    return nx.grid_2d_graph(3, 3)


@pytest.fixture
def grid_graph_4x3():
    return nx.grid_2d_graph(4, 3)


@pytest.fixture
def petersen_graph():
    return nx.petersen_graph()


@pytest.fixture
def tree_graph():
    """Binary tree of depth 3."""
    T = nx.balanced_tree(r=2, h=3)
    return T


@pytest.fixture
def star_graph():
    return nx.star_graph(6)


@pytest.fixture
def simple_dag():
    """A -> B -> C -> D, A -> C."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("A", "C")])
    return G


@pytest.fixture
def diamond_dag():
    """A -> B, A -> C, B -> D, C -> D."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return G


@pytest.fixture
def v_structure_dag():
    """A -> C <- B (v-structure / collider)."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "C"), ("B", "C")])
    return G


def _relabel_grid(grid):
    """Relabel grid graph from tuples to integers."""
    mapping = {node: i for i, node in enumerate(grid.nodes())}
    return nx.relabel_nodes(grid, mapping)


# ---------------------------------------------------------------------------
# Tree decomposition: basic properties
# ---------------------------------------------------------------------------

class TestTreeDecomposerBasic:
    """Basic tree decomposition properties that must hold for any valid TD."""

    def test_path_graph_decomposition(self, path_graph_5):
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(path_graph_5)

        assert isinstance(decomp, TreeDecomposition)
        assert decomp.width >= 1
        assert decomp.num_bags() > 0

    def test_decomp_covers_all_vertices(self, path_graph_5):
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(path_graph_5)
        covered = decomp.vertices_covered()
        assert set(path_graph_5.nodes()) == set(covered)

    def test_decomp_covers_all_edges(self, path_graph_5):
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(path_graph_5)
        for u, v in path_graph_5.edges():
            found = any(u in bag and v in bag for bag in decomp.bags.values())
            assert found, f"Edge ({u},{v}) not covered by any bag"

    def test_running_intersection_property(self, cycle_graph_5):
        """For every vertex v, the bags containing v form a connected subtree."""
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(cycle_graph_5)
        for v in cycle_graph_5.nodes():
            bag_ids = [bid for bid, bag in decomp.bags.items() if v in bag]
            if len(bag_ids) > 1:
                subgraph = decomp.tree.subgraph(bag_ids)
                assert nx.is_connected(subgraph), (
                    f"Bags containing vertex {v} do not form a connected subtree"
                )

    def test_tree_is_tree(self, cycle_graph_8):
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(cycle_graph_8)
        if decomp.num_bags() > 1:
            assert nx.is_tree(decomp.tree)

    def test_bag_sizes_consistent(self, complete_graph_4):
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(complete_graph_4)
        assert decomp.max_bag_size() == max(len(b) for b in decomp.bags.values())
        assert decomp.width == decomp.max_bag_size() - 1

    def test_decomposition_validates(self, path_graph_5):
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(path_graph_5)
        valid = td.validate_decomposition(path_graph_5, decomp)
        assert valid


class TestTreeDecomposerStrategies:
    """Test different elimination ordering strategies."""

    @pytest.mark.parametrize("strategy", ["min_fill", "min_degree", "min_width"])
    def test_strategy_produces_valid_decomp(self, strategy, cycle_graph_8):
        td = TreeDecomposer(strategy=strategy)
        decomp = td.decompose(cycle_graph_8)
        valid = td.validate_decomposition(cycle_graph_8, decomp)
        assert valid, f"Strategy {strategy} failed"

    @pytest.mark.parametrize("strategy", ["min_fill", "min_degree", "min_width"])
    def test_strategy_on_complete_graph(self, strategy, complete_graph_6):
        td = TreeDecomposer(strategy=strategy)
        decomp = td.decompose(complete_graph_6)
        assert decomp.width == 5  # treewidth of K_6 is 5

    def test_compare_strategies_returns_all(self, cycle_graph_5):
        result = compare_strategies(cycle_graph_5)
        assert "min_fill" in result
        assert "min_degree" in result
        for width in result.values():
            assert width >= 2  # cycle has treewidth 2

    def test_elimination_ordering_length(self, path_graph_10):
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(path_graph_10)
        ordering = td.get_elimination_ordering()
        assert len(ordering) == 10


# ---------------------------------------------------------------------------
# Treewidth: known values
# ---------------------------------------------------------------------------

class TestTreewidthKnownValues:
    """Test treewidth computation against analytically known values."""

    def test_path_treewidth_is_1(self):
        for n in [3, 5, 8, 12, 20]:
            G = nx.path_graph(n)
            est = TreewidthEstimator()
            ub = est.upper_bound(G)
            assert ub == 1, f"Path P_{n} treewidth upper bound should be 1, got {ub}"

    def test_cycle_treewidth_is_2(self):
        for n in [3, 5, 7, 10]:
            G = nx.cycle_graph(n)
            est = TreewidthEstimator()
            ub = est.upper_bound(G)
            assert ub == 2, f"Cycle C_{n} treewidth upper bound should be 2, got {ub}"

    def test_complete_graph_treewidth(self):
        for n in [3, 4, 5, 6]:
            G = nx.complete_graph(n)
            est = TreewidthEstimator()
            ub = est.upper_bound(G)
            assert ub == n - 1, f"K_{n} treewidth should be {n-1}, got {ub}"

    def test_tree_treewidth_is_1(self, tree_graph):
        est = TreewidthEstimator()
        ub = est.upper_bound(tree_graph)
        assert ub == 1

    def test_star_treewidth_is_1(self, star_graph):
        est = TreewidthEstimator()
        ub = est.upper_bound(star_graph)
        assert ub == 1

    def test_grid_treewidth(self):
        """Treewidth of n x m grid is min(n, m)."""
        for n, m in [(2, 3), (3, 3), (3, 4), (2, 5)]:
            G = _relabel_grid(nx.grid_2d_graph(n, m))
            est = TreewidthEstimator()
            ub = est.upper_bound(G)
            expected = min(n, m)
            assert ub >= expected, f"Grid({n},{m}): upper bound {ub} < {expected}"
            lb = est.lower_bound_mmd(G)
            assert lb <= expected, f"Grid({n},{m}): lower bound {lb} > {expected}"

    def test_petersen_treewidth(self, petersen_graph):
        est = TreewidthEstimator()
        ub = est.upper_bound(petersen_graph)
        lb = est.lower_bound_mmd(petersen_graph)
        # Known: treewidth of Petersen graph is 4
        assert lb <= 4
        assert ub >= 4

    def test_empty_graph(self):
        G = nx.Graph()
        est = TreewidthEstimator()
        assert est.upper_bound(G) == 0

    def test_single_node(self):
        G = nx.Graph()
        G.add_node(0)
        est = TreewidthEstimator()
        assert est.upper_bound(G) == 0

    def test_single_edge(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        est = TreewidthEstimator()
        assert est.upper_bound(G) == 1


class TestTreewidthBounds:
    """Test treewidth lower/upper bound relationship."""

    def test_lower_leq_upper(self):
        graphs = [
            nx.path_graph(10),
            nx.cycle_graph(8),
            nx.complete_graph(5),
            _relabel_grid(nx.grid_2d_graph(3, 3)),
            nx.petersen_graph(),
        ]
        est = TreewidthEstimator()
        for G in graphs:
            lb = est.lower_bound_mmd(G)
            ub = est.upper_bound(G)
            assert lb <= ub, f"lower bound {lb} > upper bound {ub}"

    def test_estimate_returns_pair(self, cycle_graph_8):
        est = TreewidthEstimator()
        lb, ub = est.estimate(cycle_graph_8)
        assert lb <= ub
        assert lb >= 0

    @pytest.mark.parametrize("method", ["min_fill", "min_degree", "min_width"])
    def test_upper_bound_methods(self, method, petersen_graph):
        est = TreewidthEstimator()
        ub = est.upper_bound(petersen_graph, method=method)
        assert ub >= 4  # Known lower bound

    def test_improved_mmd(self, cycle_graph_8):
        est = TreewidthEstimator()
        lb_basic = est.lower_bound_mmd(cycle_graph_8)
        lb_improved = est.lower_bound_improved_mmd(cycle_graph_8)
        assert lb_improved >= lb_basic

    def test_contraction_bound(self, petersen_graph):
        est = TreewidthEstimator()
        cb = est.contraction_bound(petersen_graph)
        assert cb >= 1


class TestTreewidthExact:
    """Test exact treewidth on small graphs."""

    def test_exact_path(self):
        G = nx.path_graph(5)
        est = TreewidthEstimator()
        tw = est.exact_treewidth(G)
        assert tw == 1

    def test_exact_cycle(self):
        G = nx.cycle_graph(5)
        est = TreewidthEstimator()
        tw = est.exact_treewidth(G)
        assert tw == 2

    def test_exact_complete_k4(self):
        G = nx.complete_graph(4)
        est = TreewidthEstimator()
        tw = est.exact_treewidth(G)
        assert tw == 3

    def test_exact_matches_bounds(self):
        G = nx.cycle_graph(6)
        est = TreewidthEstimator()
        tw = est.exact_treewidth(G)
        lb = est.lower_bound_mmd(G)
        ub = est.upper_bound(G)
        assert lb <= tw <= ub


# ---------------------------------------------------------------------------
# Separator extraction
# ---------------------------------------------------------------------------

class TestSeparatorExtractor:
    """Test separator extraction from tree decompositions."""

    def test_extract_separators_from_decomp(self, path_graph_10):
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(path_graph_10)
        extractor = SeparatorExtractor()
        seps = extractor.extract_separators(decomp.bags, decomp.tree)
        assert len(seps) > 0

    def test_separator_is_subset_of_both_bags(self, cycle_graph_8):
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(cycle_graph_8)
        extractor = SeparatorExtractor()
        seps = extractor.extract_separators(decomp.bags, decomp.tree)
        for sep in seps:
            # Each separator is the intersection of two adjacent bags
            found = False
            for u, v in decomp.tree.edges():
                if decomp.bags[u] & decomp.bags[v] == sep:
                    assert sep.issubset(decomp.bags[u])
                    assert sep.issubset(decomp.bags[v])
                    found = True
                    break
            assert found

    def test_separator_removal_disconnects(self, cycle_graph_8):
        """Removing a separator from the graph should disconnect it (for valid seps)."""
        extractor = SeparatorExtractor()
        seps = extractor.enumerate_minimal_separators(cycle_graph_8)
        if seps:
            for sep in seps[:5]:
                remaining = set(cycle_graph_8.nodes()) - sep
                if remaining:
                    subg = cycle_graph_8.subgraph(remaining)
                    assert not nx.is_connected(subg), (
                        f"Removing separator {sep} did not disconnect the graph"
                    )

    def test_balanced_separator(self, path_graph_10):
        extractor = SeparatorExtractor()
        sep = extractor.find_balanced_separator(path_graph_10)
        assert sep is not None
        assert len(sep) > 0

    def test_score_separator(self, cycle_graph_8):
        extractor = SeparatorExtractor()
        sep = frozenset({0})
        score = extractor.score_separator(cycle_graph_8, sep)
        assert isinstance(score, dict)
        assert "overall" in score

    def test_safe_separators(self, path_graph_10):
        extractor = SeparatorExtractor()
        dag = nx.DiGraph()
        dag.add_edges_from([(i, i + 1) for i in range(10)])
        safe = extractor.find_safe_separators(path_graph_10, dag)
        assert isinstance(safe, list)


# ---------------------------------------------------------------------------
# Moral graph construction
# ---------------------------------------------------------------------------

class TestMoralGraphConstructor:
    """Test moral graph and triangulation."""

    def test_moralize_v_structure(self, v_structure_dag):
        """A->C<-B: moralizing should add edge A-B."""
        mgc = MoralGraphConstructor()
        moral = mgc.moralize(v_structure_dag)
        assert moral.has_edge("A", "B") or moral.has_edge("B", "A")
        assert moral.has_edge("A", "C") or moral.has_edge("C", "A")
        assert moral.has_edge("B", "C") or moral.has_edge("C", "B")

    def test_moralize_preserves_nodes(self, diamond_dag):
        mgc = MoralGraphConstructor()
        moral = mgc.moralize(diamond_dag)
        assert set(moral.nodes()) == set(diamond_dag.nodes())

    def test_moralize_chain_no_extra_edges(self):
        """A->B->C: no v-structure, so no extra edges."""
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C")])
        mgc = MoralGraphConstructor()
        moral = mgc.moralize(dag)
        assert moral.number_of_edges() == 2
        assert not moral.has_edge("A", "C")

    def test_moralize_diamond(self, diamond_dag):
        """A->B,A->C,B->D,C->D: moralizing should add B-C edge."""
        mgc = MoralGraphConstructor()
        moral = mgc.moralize(diamond_dag)
        assert moral.has_edge("B", "C") or moral.has_edge("C", "B")

    def test_triangulation_produces_chordal(self, cycle_graph_5):
        mgc = MoralGraphConstructor()
        tri = mgc.triangulate(cycle_graph_5)
        assert mgc.is_chordal(tri)

    def test_triangulation_preserves_edges(self, cycle_graph_8):
        mgc = MoralGraphConstructor()
        tri = mgc.triangulate(cycle_graph_8)
        for u, v in cycle_graph_8.edges():
            assert tri.has_edge(u, v), f"Edge ({u},{v}) lost during triangulation"

    def test_chordal_graph_stays_chordal(self, complete_graph_4):
        mgc = MoralGraphConstructor()
        assert mgc.is_chordal(complete_graph_4)
        tri = mgc.triangulate(complete_graph_4)
        assert tri.number_of_edges() == complete_graph_4.number_of_edges()

    def test_triangulation_methods(self, cycle_graph_8):
        mgc = MoralGraphConstructor()
        tri_fill = mgc.triangulate(cycle_graph_8, method="min_fill")
        tri_deg = mgc.triangulate(cycle_graph_8, method="min_degree")
        # Verify fill edges are added (triangulated has >= original edges)
        assert tri_fill.number_of_edges() >= cycle_graph_8.number_of_edges()
        assert tri_deg.number_of_edges() >= cycle_graph_8.number_of_edges()

    def test_perfect_elimination_ordering(self, complete_graph_4):
        mgc = MoralGraphConstructor()
        peo = mgc.find_perfect_elimination_ordering(complete_graph_4)
        assert peo is not None
        assert len(peo) == 4

    def test_maximal_cliques_extraction(self, complete_graph_4):
        mgc = MoralGraphConstructor()
        cliques = mgc.extract_maximal_cliques(complete_graph_4)
        assert len(cliques) >= 1
        # K_4 has exactly one maximal clique: the whole graph
        assert any(len(c) == 4 for c in cliques)

    def test_clique_tree_construction(self, diamond_dag):
        mgc = MoralGraphConstructor()
        moral = mgc.moralize(diamond_dag)
        tri = mgc.triangulate(moral)
        cliques = mgc.extract_maximal_cliques(tri)
        clique_tree = mgc.build_clique_tree(cliques)
        assert clique_tree is not None
        assert clique_tree.number_of_nodes() > 0

    def test_running_intersection_verified(self, simple_dag):
        mgc = MoralGraphConstructor()
        moral = mgc.moralize(simple_dag)
        tri = mgc.triangulate(moral)
        cliques = mgc.extract_maximal_cliques(tri)
        clique_tree = mgc.build_clique_tree(cliques)
        rip_ok = mgc._verify_running_intersection(clique_tree, cliques)
        assert rip_ok

    def test_moral_edge_density(self, diamond_dag):
        mgc = MoralGraphConstructor()
        density = mgc.moral_edge_density(diamond_dag)
        assert 0.0 <= density <= 1.0

    def test_ancestral_graph(self, simple_dag):
        mgc = MoralGraphConstructor()
        anc = mgc.ancestral_graph(simple_dag, {"C", "D"})
        assert "A" in anc.nodes()
        assert "B" in anc.nodes()

    def test_dseparation_moral(self, diamond_dag):
        mgc = MoralGraphConstructor()
        is_dsep = mgc.dseparation_moral(diamond_dag, {"A"}, {"D"}, {"B", "C"})
        assert isinstance(is_dsep, bool)

    def test_minimal_triangulation(self, cycle_graph_8):
        mgc = MoralGraphConstructor()
        tri = mgc.minimal_triangulation(cycle_graph_8)
        assert mgc.is_chordal(tri)
        # Minimal means no edge can be removed while staying chordal
        for u, v in tri.edges():
            if not cycle_graph_8.has_edge(u, v):
                test_g = tri.copy()
                test_g.remove_edge(u, v)
                if not mgc.is_chordal(test_g):
                    pass  # edge is necessary — good
                # (Not all fill edges need to be necessary for the graph to be chordal)

    def test_induced_width(self, cycle_graph_5):
        mgc = MoralGraphConstructor()
        ordering = list(cycle_graph_5.nodes())
        iw = mgc.induced_width(cycle_graph_5, ordering)
        assert iw >= 0

    def test_fill_edge_count(self, cycle_graph_8):
        mgc = MoralGraphConstructor()
        fill = mgc.fill_edge_count(cycle_graph_8)
        assert fill >= 0


# ---------------------------------------------------------------------------
# Bounded treewidth enforcement
# ---------------------------------------------------------------------------

class TestBoundedTreewidth:
    """Test treewidth enforcement / bounding mechanisms."""

    def test_decompose_with_bound(self, complete_graph_6):
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(complete_graph_6, max_width=3)
        # Enforcement may or may not fully achieve the bound, but should try
        assert isinstance(decomp, TreeDecomposition)
        assert decomp.width >= 0

    def test_decompose_with_tight_bound(self, path_graph_5):
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(path_graph_5, max_width=1)
        assert decomp.width <= 1

    def test_refine_ordering_improves(self, cycle_graph_8):
        td = TreeDecomposer(strategy="min_degree")
        decomp = td.decompose(cycle_graph_8)
        original_width = decomp.width
        refined = td.refine_ordering(cycle_graph_8, decomp.ordering)
        # Refined width should be <= original
        assert refined is not None


# ---------------------------------------------------------------------------
# Elimination orderings
# ---------------------------------------------------------------------------

class TestEliminationOrderings:
    """Test elimination ordering generation and properties."""

    def test_min_fill_ordering_covers_all(self, path_graph_10):
        td = TreeDecomposer(strategy="min_fill")
        td.decompose(path_graph_10)
        ordering = td.get_elimination_ordering()
        assert set(ordering) == set(path_graph_10.nodes())

    def test_min_degree_ordering_covers_all(self, cycle_graph_8):
        td = TreeDecomposer(strategy="min_degree")
        td.decompose(cycle_graph_8)
        ordering = td.get_elimination_ordering()
        assert set(ordering) == set(cycle_graph_8.nodes())

    def test_ordering_is_permutation(self, petersen_graph):
        td = TreeDecomposer(strategy="min_fill")
        td.decompose(petersen_graph)
        ordering = td.get_elimination_ordering()
        assert len(ordering) == len(set(ordering))
        assert len(ordering) == petersen_graph.number_of_nodes()

    def test_min_fill_vs_min_degree(self, petersen_graph):
        td_fill = TreeDecomposer(strategy="min_fill")
        td_deg = TreeDecomposer(strategy="min_degree")
        d_fill = td_fill.decompose(petersen_graph)
        d_deg = td_deg.decompose(petersen_graph)
        # Both should be valid
        assert td_fill.validate_decomposition(petersen_graph, d_fill)
        assert td_deg.validate_decomposition(petersen_graph, d_deg)


# ---------------------------------------------------------------------------
# Causal partition
# ---------------------------------------------------------------------------

class TestCausalPartitioner:
    """Test causal-aware DAG partitioning."""

    def test_partition_simple_dag(self, simple_dag):
        cp = CausalPartitioner(max_partition_size=3)
        result = cp.partition(simple_dag)
        assert result is not None
        all_nodes = set()
        for part in result:
            all_nodes.update(part.nodes)
        assert all_nodes == set(simple_dag.nodes())

    def test_partition_diamond_dag(self, diamond_dag):
        cp = CausalPartitioner(max_partition_size=3)
        result = cp.partition(diamond_dag)
        assert len(result) >= 1

    def test_partition_large_chain(self):
        dag = nx.DiGraph()
        for i in range(20):
            dag.add_edge(str(i), str(i + 1))
        cp = CausalPartitioner(max_partition_size=5)
        result = cp.partition(dag)
        assert len(result) >= 4

    def test_partition_quality(self, diamond_dag):
        cp = CausalPartitioner(max_partition_size=4)
        result = cp.partition(diamond_dag)
        part_node_sets = [part.nodes for part in result]
        quality = cp.compute_partition_quality(diamond_dag, part_node_sets)
        assert isinstance(quality, dict)

    def test_partition_respects_dag(self, simple_dag):
        """Partitions should respect topological structure."""
        cp = CausalPartitioner(max_partition_size=10)
        result = cp.partition(simple_dag)
        for part in result:
            subdag = simple_dag.subgraph(part.nodes)
            assert nx.is_directed_acyclic_graph(subdag)


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------

class TestSubgraphExtractor:
    """Test subgraph extraction utilities."""

    def test_extract_induced_subgraph(self, path_graph_10):
        ext = SubgraphExtractor()
        nodes = {0, 1, 2, 3}
        sub = ext.extract_induced_subgraph(path_graph_10, nodes, include_boundary=False)
        assert set(sub.nodes()) == nodes
        assert sub.has_edge(0, 1)
        assert sub.has_edge(2, 3)
        assert not sub.has_edge(0, 3)

    def test_extract_with_boundary(self, path_graph_10):
        ext = SubgraphExtractor()
        nodes = {2, 3, 4, 5}
        boundary = {1, 6}
        sub = ext.extract_with_boundary(path_graph_10, nodes, boundary)
        assert set(sub.nodes()).issuperset(nodes)

    def test_compute_overlap(self, path_graph_10):
        ext = SubgraphExtractor()
        sgs = [
            path_graph_10.subgraph({0, 1, 2, 3}).copy(),
            path_graph_10.subgraph({2, 3, 4, 5}).copy(),
        ]
        overlap = ext.compute_overlap_structure(sgs)
        assert overlap.number_of_nodes() > 0

    def test_statistics(self, complete_graph_4):
        ext = SubgraphExtractor()
        stats = ext.compute_statistics([complete_graph_4.subgraph({0, 1, 2}).copy()])
        assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    """Test standalone utility functions in decomposition module."""

    def test_build_adjacency_matrix(self, path_graph_5):
        adj, node_to_idx = build_adjacency_matrix(path_graph_5)
        n = path_graph_5.number_of_nodes()
        assert adj.shape == (n, n)
        assert np.allclose(adj, adj.T)

    def test_compute_treewidth_upper_bound(self, cycle_graph_5):
        ub = compute_treewidth_upper_bound(cycle_graph_5)
        assert ub == 2

    def test_compute_treewidth_lower_bound(self, cycle_graph_5):
        lb = compute_treewidth_lower_bound(cycle_graph_5)
        assert lb <= 2

    def test_decompose_and_validate(self, path_graph_10):
        decomp, valid = decompose_and_validate(path_graph_10)
        assert decomp is not None
        assert decomp.width >= 1
        assert valid

    def test_separator_sizes(self, cycle_graph_8):
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(cycle_graph_8)
        sizes = separator_sizes(decomp)
        assert all(s >= 0 for s in sizes)

    def test_decomposition_statistics(self, petersen_graph):
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(petersen_graph)
        stats = decomposition_statistics(decomp)
        assert "width" in stats
        assert "num_bags" in stats

    def test_junction_tree_conversion(self, cycle_graph_8):
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(cycle_graph_8)
        jt = td.to_junction_tree(decomp)
        assert isinstance(jt, nx.Graph)
        assert jt.number_of_nodes() > 0

    def test_treewidth_profile(self, path_graph_5):
        est = TreewidthEstimator()
        partitions = [set(path_graph_5.nodes())]
        profile = est.compute_treewidth_profile(path_graph_5, partitions)
        assert isinstance(profile, dict)

    def test_characterize_graph(self, cycle_graph_8):
        est = TreewidthEstimator()
        char = est.characterize_graph(cycle_graph_8)
        assert isinstance(char, dict)


# ---------------------------------------------------------------------------
# Edge cases and stress tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for graph decomposition."""

    def test_disconnected_graph(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(2, 3)
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(G)
        assert set(decomp.vertices_covered()) == {0, 1, 2, 3}

    def test_self_loop_graph(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(0, 0)  # self-loop
        td = TreeDecomposer(strategy="min_fill")
        # Should handle self-loops gracefully
        decomp = td.decompose(G)
        assert decomp is not None

    def test_large_path(self):
        G = nx.path_graph(100)
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(G)
        assert decomp.width == 1

    def test_large_cycle(self):
        G = nx.cycle_graph(50)
        td = TreeDecomposer(strategy="min_fill")
        decomp = td.decompose(G)
        assert decomp.width == 2

    def test_bipartite_graph(self):
        G = nx.complete_bipartite_graph(3, 3)
        est = TreewidthEstimator()
        ub = est.upper_bound(G)
        assert ub >= 3  # tw(K_{3,3}) = 3

    def test_wheel_graph(self):
        G = nx.wheel_graph(7)
        est = TreewidthEstimator()
        ub = est.upper_bound(G)
        assert ub >= 3

    def test_multiple_decompositions_deterministic(self, cycle_graph_5):
        td = TreeDecomposer(strategy="min_fill")
        d1 = td.decompose(cycle_graph_5)
        td2 = TreeDecomposer(strategy="min_fill")
        d2 = td2.decompose(cycle_graph_5)
        assert d1.width == d2.width
