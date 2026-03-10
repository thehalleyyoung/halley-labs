"""
Elimination ordering algorithms for treewidth computation.

Provides multiple heuristic strategies for computing elimination orderings
of undirected graphs, plus exact checks for chordal graphs and treewidth
bounds.  These orderings are the basis for constructing tree decompositions
in :mod:`causalcert.treewidth.decomposition`.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from causalcert.types import AdjacencyMatrix, NodeId, NodeSet
from causalcert.treewidth.types import EliminationOrdering, TreeBag, TreeDecomposition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _graph_from_adj(adj: AdjacencyMatrix) -> nx.Graph:
    """Build an undirected networkx graph from a symmetric adjacency matrix."""
    adj = np.asarray(adj, dtype=np.int8)
    G = nx.Graph()
    n = adj.shape[0]
    G.add_nodes_from(range(n))
    rows, cols = np.nonzero(np.triu(adj, k=1))
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))
    return G


def _adj_from_graph(G: nx.Graph, n: int | None = None) -> NDArray[np.int8]:
    """Build a symmetric adjacency matrix from a networkx graph."""
    if n is None:
        n = G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in G.edges():
        adj[u, v] = 1
        adj[v, u] = 1
    return adj


# ---------------------------------------------------------------------------
# Min-degree heuristic
# ---------------------------------------------------------------------------


def min_degree_ordering(graph: nx.Graph) -> EliminationOrdering:
    """Compute a min-degree elimination ordering.

    At each step the vertex with the fewest neighbours in the remaining graph
    is eliminated and its neighbourhood is turned into a clique (fill edges
    are added).

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    EliminationOrdering
        The elimination ordering together with its induced width.
    """
    G = graph.copy()
    order: list[NodeId] = []
    induced_width = 0

    while G.number_of_nodes() > 0:
        # Pick vertex with minimum degree (break ties by smallest id)
        v = min(G.nodes(), key=lambda u: (G.degree(u), u))
        nbrs = list(G.neighbors(v))
        induced_width = max(induced_width, len(nbrs))

        # Make neighbours a clique
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if not G.has_edge(nbrs[i], nbrs[j]):
                    G.add_edge(nbrs[i], nbrs[j])

        order.append(v)
        G.remove_node(v)

    return EliminationOrdering(order=tuple(order), induced_width=induced_width)


# ---------------------------------------------------------------------------
# Min-fill heuristic
# ---------------------------------------------------------------------------


def _fill_count(G: nx.Graph, v: NodeId) -> int:
    """Count how many fill edges eliminating *v* would add."""
    nbrs = list(G.neighbors(v))
    count = 0
    for i in range(len(nbrs)):
        for j in range(i + 1, len(nbrs)):
            if not G.has_edge(nbrs[i], nbrs[j]):
                count += 1
    return count


def min_fill_ordering(graph: nx.Graph) -> EliminationOrdering:
    """Compute a min-fill elimination ordering.

    At each step the vertex whose elimination adds the fewest fill edges
    is selected.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    EliminationOrdering
        The elimination ordering together with its induced width.
    """
    G = graph.copy()
    order: list[NodeId] = []
    induced_width = 0

    while G.number_of_nodes() > 0:
        v = min(G.nodes(), key=lambda u: (_fill_count(G, u), G.degree(u), u))
        nbrs = list(G.neighbors(v))
        induced_width = max(induced_width, len(nbrs))

        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if not G.has_edge(nbrs[i], nbrs[j]):
                    G.add_edge(nbrs[i], nbrs[j])

        order.append(v)
        G.remove_node(v)

    return EliminationOrdering(order=tuple(order), induced_width=induced_width)


# ---------------------------------------------------------------------------
# Min-width heuristic
# ---------------------------------------------------------------------------


def min_width_ordering(graph: nx.Graph) -> EliminationOrdering:
    """Compute a min-width (greedy degree) elimination ordering.

    At each step the vertex with the smallest degree is eliminated
    *without* adding fill edges to track the induced width. Fill edges are
    still added internally for the actual triangulation, but the selection
    criterion is the degree before fill.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    EliminationOrdering
    """
    G = graph.copy()
    order: list[NodeId] = []
    induced_width = 0

    while G.number_of_nodes() > 0:
        v = min(G.nodes(), key=lambda u: (G.degree(u), u))
        nbrs = list(G.neighbors(v))
        induced_width = max(induced_width, len(nbrs))

        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if not G.has_edge(nbrs[i], nbrs[j]):
                    G.add_edge(nbrs[i], nbrs[j])

        order.append(v)
        G.remove_node(v)

    return EliminationOrdering(order=tuple(order), induced_width=induced_width)


# ---------------------------------------------------------------------------
# Maximum cardinality search (MCS)
# ---------------------------------------------------------------------------


def max_cardinality_search(graph: nx.Graph) -> EliminationOrdering:
    """Compute an elimination ordering via maximum cardinality search.

    MCS iteratively selects the un-numbered vertex with the most already-
    numbered neighbours.  On a chordal graph this produces a perfect
    elimination ordering (in reverse).

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    EliminationOrdering
        The MCS ordering (reversed to be an elimination ordering) with
        its induced width.
    """
    G = graph
    n = G.number_of_nodes()
    if n == 0:
        return EliminationOrdering(order=(), induced_width=0)

    nodes = sorted(G.nodes())
    weight: dict[NodeId, int] = {v: 0 for v in nodes}
    numbered: set[NodeId] = set()
    mcs_order: list[NodeId] = []

    for _ in range(n):
        v = max(
            (u for u in nodes if u not in numbered),
            key=lambda u: (weight[u], -u),
        )
        mcs_order.append(v)
        numbered.add(v)
        for nb in G.neighbors(v):
            if nb not in numbered:
                weight[nb] += 1

    # Reverse to get an elimination ordering
    elim_order = list(reversed(mcs_order))

    # Compute induced width by simulating the elimination
    induced_width = _compute_induced_width(graph, elim_order)

    return EliminationOrdering(order=tuple(elim_order), induced_width=induced_width)


def _compute_induced_width(graph: nx.Graph, order: list[NodeId]) -> int:
    """Simulate an elimination and return the induced width."""
    G = graph.copy()
    width = 0
    for v in order:
        nbrs = list(G.neighbors(v))
        width = max(width, len(nbrs))
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if not G.has_edge(nbrs[i], nbrs[j]):
                    G.add_edge(nbrs[i], nbrs[j])
        G.remove_node(v)
    return width


# ---------------------------------------------------------------------------
# Perfect elimination ordering detection
# ---------------------------------------------------------------------------


def is_perfect_elimination_ordering(
    graph: nx.Graph,
    order: Sequence[NodeId],
) -> bool:
    """Check whether *order* is a perfect elimination ordering of *graph*.

    A PEO exists if and only if the graph is chordal.  In a PEO, for each
    vertex *v* the neighbours of *v* that appear after *v* in the ordering
    form a clique.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.
    order : Sequence[NodeId]
        Candidate elimination ordering.

    Returns
    -------
    bool
    """
    pos = {v: i for i, v in enumerate(order)}
    for idx, v in enumerate(order):
        later_nbrs = [
            u for u in graph.neighbors(v) if pos.get(u, -1) > idx
        ]
        for i in range(len(later_nbrs)):
            for j in range(i + 1, len(later_nbrs)):
                if not graph.has_edge(later_nbrs[i], later_nbrs[j]):
                    return False
    return True


def detect_perfect_elimination_ordering(
    graph: nx.Graph,
) -> EliminationOrdering | None:
    """Return a perfect elimination ordering if the graph is chordal.

    Uses maximum cardinality search and then verifies the PEO property.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    EliminationOrdering | None
        A perfect elimination ordering, or ``None`` if *graph* is not chordal.
    """
    if graph.number_of_nodes() == 0:
        return EliminationOrdering(order=(), induced_width=0)

    eo = max_cardinality_search(graph)
    if is_perfect_elimination_ordering(graph, eo.order):
        return eo
    return None


# ---------------------------------------------------------------------------
# Greedy triangulation
# ---------------------------------------------------------------------------


def triangulate(graph: nx.Graph, order: Sequence[NodeId]) -> nx.Graph:
    """Triangulate *graph* according to the given elimination ordering.

    For each vertex in the ordering, its remaining neighbours are connected
    into a clique.  The result is a chordal supergraph of the input.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.
    order : Sequence[NodeId]
        Elimination ordering.

    Returns
    -------
    nx.Graph
        The triangulated (chordal) graph.
    """
    H = graph.copy()
    eliminated: set[NodeId] = set()

    for v in order:
        nbrs = [u for u in H.neighbors(v) if u not in eliminated]
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if not H.has_edge(nbrs[i], nbrs[j]):
                    H.add_edge(nbrs[i], nbrs[j])
        eliminated.add(v)

    return H


# ---------------------------------------------------------------------------
# Ordering → tree decomposition conversion
# ---------------------------------------------------------------------------


def ordering_to_decomposition(
    graph: nx.Graph,
    ordering: EliminationOrdering,
) -> TreeDecomposition:
    """Convert an elimination ordering into a tree decomposition.

    Each eliminated vertex *v* produces a bag ``{v} ∪ N(v)`` where ``N(v)``
    is the set of neighbours of *v* still alive at elimination time.  Bags
    are connected into a tree by linking each bag to the next bag that shares
    a vertex.

    Parameters
    ----------
    graph : nx.Graph
        Original undirected graph.
    ordering : EliminationOrdering
        The elimination ordering to convert.

    Returns
    -------
    TreeDecomposition
        A valid tree decomposition rooted at the last bag.
    """
    G = graph.copy()
    order = ordering.order
    n = len(order)
    if n == 0:
        return TreeDecomposition(bags=(), width=0, root_id=0, n_vertices=0,
                                 elimination_ordering=ordering)

    raw_bags: list[frozenset[NodeId]] = []
    vertex_to_bag: dict[NodeId, int] = {}

    for v in order:
        nbrs = frozenset(G.neighbors(v))
        bag = frozenset({v}) | nbrs
        bag_idx = len(raw_bags)
        raw_bags.append(bag)
        vertex_to_bag[v] = bag_idx

        # Add fill edges
        nbrs_list = list(nbrs)
        for i in range(len(nbrs_list)):
            for j in range(i + 1, len(nbrs_list)):
                if not G.has_edge(nbrs_list[i], nbrs_list[j]):
                    G.add_edge(nbrs_list[i], nbrs_list[j])
        G.remove_node(v)

    # Build tree: connect bag i to the earliest subsequent bag sharing a vertex
    children: dict[int, list[int]] = defaultdict(list)
    parent: dict[int, int | None] = {i: None for i in range(n)}

    for i in range(n):
        rest = raw_bags[i] - {order[i]}
        if not rest:
            continue
        for j in range(i + 1, n):
            if raw_bags[j] & rest:
                children[j].append(i)
                parent[i] = j
                break

    # Find root (the bag with no parent)
    root_id = n - 1
    for i in range(n):
        if parent[i] is None:
            root_id = i
            break
    # If multiple roots, connect them to a single root
    roots = [i for i in range(n) if parent[i] is None]
    if len(roots) > 1:
        root_id = roots[0]
        for r in roots[1:]:
            children[root_id].append(r)
            parent[r] = root_id

    # Build TreeBag objects
    bags: list[TreeBag] = []
    for i in range(n):
        bags.append(TreeBag(
            bag_id=i,
            vertices=raw_bags[i],
            parent_id=parent[i],
            children_ids=tuple(children.get(i, [])),
        ))

    width = max((len(b) - 1 for b in raw_bags), default=0)
    return TreeDecomposition(
        bags=tuple(bags),
        width=width,
        root_id=root_id,
        n_vertices=graph.number_of_nodes(),
        elimination_ordering=ordering,
    )


# ---------------------------------------------------------------------------
# Treewidth bounds
# ---------------------------------------------------------------------------


def compute_treewidth_upper_bound(graph: nx.Graph) -> int:
    """Compute an upper bound on treewidth using the best of several heuristics.

    Tries min-degree, min-fill, and min-width orderings and returns the
    minimum induced width across all of them.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    int
        Upper bound on the treewidth.
    """
    if graph.number_of_nodes() == 0:
        return 0

    orderings = [
        min_degree_ordering(graph),
        min_fill_ordering(graph),
        min_width_ordering(graph),
    ]
    return min(eo.induced_width for eo in orderings)


def compute_treewidth_lower_bound(graph: nx.Graph) -> int:
    """Compute a lower bound on treewidth via the minor-min-width heuristic.

    The MMW heuristic iteratively contracts the edge incident to the
    minimum-degree vertex, updating degrees.  The maximum minimum degree
    seen during the process is a lower bound on treewidth.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    int
        Lower bound on the treewidth.
    """
    if graph.number_of_nodes() <= 1:
        return 0

    G = graph.copy()
    lower_bound = 0

    while G.number_of_nodes() > 1:
        # Find minimum-degree vertex
        v = min(G.nodes(), key=lambda u: (G.degree(u), u))
        deg_v = G.degree(v)
        lower_bound = max(lower_bound, deg_v)

        if deg_v == 0:
            G.remove_node(v)
            continue

        # Contract edge to neighbour with smallest degree
        nbrs = list(G.neighbors(v))
        u = min(nbrs, key=lambda w: (G.degree(w), w))

        # Contract v into u: merge v's neighbours into u
        for w in nbrs:
            if w != u and not G.has_edge(u, w):
                G.add_edge(u, w)
        G.remove_node(v)

    return lower_bound


def degeneracy_lower_bound(graph: nx.Graph) -> int:
    """Compute the degeneracy of the graph as a treewidth lower bound.

    The degeneracy is the maximum over all subgraphs of the minimum degree.
    It is always at most the treewidth.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    int
        Degeneracy (a treewidth lower bound).
    """
    if graph.number_of_nodes() == 0:
        return 0

    G = graph.copy()
    degen = 0

    while G.number_of_nodes() > 0:
        v = min(G.nodes(), key=lambda u: G.degree(u))
        degen = max(degen, G.degree(v))
        G.remove_node(v)

    return degen


# ---------------------------------------------------------------------------
# Best-effort ordering selection
# ---------------------------------------------------------------------------


def best_heuristic_ordering(graph: nx.Graph) -> EliminationOrdering:
    """Return the best elimination ordering among several heuristics.

    Tries min-degree, min-fill, min-width, and MCS orderings and returns
    the one with the smallest induced width.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    EliminationOrdering
        The ordering with the smallest induced width.
    """
    if graph.number_of_nodes() == 0:
        return EliminationOrdering(order=(), induced_width=0)

    candidates = [
        min_degree_ordering(graph),
        min_fill_ordering(graph),
        min_width_ordering(graph),
        max_cardinality_search(graph),
    ]
    return min(candidates, key=lambda eo: eo.induced_width)


__all__ = [
    "min_degree_ordering",
    "min_fill_ordering",
    "min_width_ordering",
    "max_cardinality_search",
    "is_perfect_elimination_ordering",
    "detect_perfect_elimination_ordering",
    "triangulate",
    "ordering_to_decomposition",
    "compute_treewidth_upper_bound",
    "compute_treewidth_lower_bound",
    "degeneracy_lower_bound",
    "best_heuristic_ordering",
]
