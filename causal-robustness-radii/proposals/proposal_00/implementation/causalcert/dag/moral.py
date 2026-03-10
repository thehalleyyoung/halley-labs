"""
Moral graph construction, treewidth computation, and tree decomposition.

The moral graph is used by the FPT dynamic-programming solver (ALG 7).
Tree decompositions enable fixed-parameter tractable enumeration of edit
sets when the DAG's moral graph has bounded treewidth.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from causalcert.types import AdjacencyMatrix, NodeId, NodeSet


def moral_graph(adj: AdjacencyMatrix) -> np.ndarray:
    """Construct the moral graph of a DAG.

    The moral graph is the undirected graph obtained by:
    1. Connecting all pairs of nodes that share a common child ("marrying" parents).
    2. Dropping edge orientations.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.

    Returns
    -------
    np.ndarray
        Symmetric binary adjacency matrix of the moral graph.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    moral = np.zeros((n, n), dtype=np.int8)

    # Step 1: Add undirected versions of all existing edges
    moral = moral | adj | adj.T

    # Step 2: Marry parents — for each node, connect all pairs of parents
    for child in range(n):
        parents = list(np.nonzero(adj[:, child])[0])
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                moral[parents[i], parents[j]] = 1
                moral[parents[j], parents[i]] = 1

    # Remove self-loops
    np.fill_diagonal(moral, 0)
    return moral


def moralize_ancestral(
    adj: AdjacencyMatrix,
    targets: NodeSet,
) -> np.ndarray:
    """Moralize the ancestral subgraph of *targets*.

    First restricts the DAG to the ancestors of *targets* (including
    *targets*), then moralizes.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Full DAG adjacency matrix.
    targets : NodeSet
        Target nodes.

    Returns
    -------
    np.ndarray
        Moral graph of the ancestral subgraph.
    """
    from causalcert.dag.ancestors import ancestors, ancestral_subgraph

    sub_adj = ancestral_subgraph(adj, targets)
    return moral_graph(sub_adj)


@dataclass(frozen=True, slots=True)
class TreeDecomposition:
    """A tree decomposition of a graph.

    Attributes
    ----------
    bags : list[NodeSet]
        Each bag is a frozenset of node ids.
    tree_adj : list[list[int]]
        Adjacency list of the tree over bags.
    width : int
        Treewidth (max bag size − 1).
    """

    bags: list[NodeSet]
    tree_adj: list[list[int]]
    width: int


def _min_degree_ordering(adj: np.ndarray) -> list[int]:
    """Compute a min-degree elimination ordering.

    At each step, eliminate the vertex of minimum degree in the
    remaining graph, adding fill edges to make its neighbors a clique.

    Parameters
    ----------
    adj : np.ndarray
        Symmetric adjacency matrix.

    Returns
    -------
    list[int]
        Elimination ordering.
    """
    n = adj.shape[0]
    work = adj.astype(np.int8).copy()
    eliminated: set[int] = set()
    ordering: list[int] = []

    for _ in range(n):
        # Find un-eliminated node with minimum degree
        min_deg = n + 1
        min_node = -1
        for v in range(n):
            if v in eliminated:
                continue
            deg = 0
            for u in range(n):
                if u != v and u not in eliminated and work[v, u]:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                min_node = v

        ordering.append(min_node)
        eliminated.add(min_node)

        # Add fill edges between remaining neighbors
        neighbors = [
            u for u in range(n)
            if u != min_node and u not in eliminated and work[min_node, u]
        ]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                work[neighbors[i], neighbors[j]] = 1
                work[neighbors[j], neighbors[i]] = 1

    return ordering


def _min_fill_ordering(adj: np.ndarray) -> list[int]:
    """Compute a min-fill elimination ordering.

    At each step, eliminate the vertex whose elimination adds the fewest
    fill edges.

    Parameters
    ----------
    adj : np.ndarray
        Symmetric adjacency matrix.

    Returns
    -------
    list[int]
        Elimination ordering.
    """
    n = adj.shape[0]
    work = adj.astype(np.int8).copy()
    eliminated: set[int] = set()
    ordering: list[int] = []

    for _ in range(n):
        min_fill = n * n
        min_node = -1

        for v in range(n):
            if v in eliminated:
                continue
            neighbors = [
                u for u in range(n)
                if u != v and u not in eliminated and work[v, u]
            ]
            fill_count = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if not work[neighbors[i], neighbors[j]]:
                        fill_count += 1
            if fill_count < min_fill:
                min_fill = fill_count
                min_node = v

        ordering.append(min_node)
        eliminated.add(min_node)

        neighbors = [
            u for u in range(n)
            if u != min_node and u not in eliminated and work[min_node, u]
        ]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                work[neighbors[i], neighbors[j]] = 1
                work[neighbors[j], neighbors[i]] = 1

    return ordering


def _ordering_to_tree_decomposition(
    adj: np.ndarray,
    ordering: list[int],
) -> TreeDecomposition:
    """Convert an elimination ordering to a tree decomposition.

    For each vertex v eliminated, its bag consists of v together with
    its neighbors still alive at the time of elimination.

    Parameters
    ----------
    adj : np.ndarray
        Original symmetric adjacency matrix.
    ordering : list[int]
        Elimination ordering.

    Returns
    -------
    TreeDecomposition
    """
    n = adj.shape[0]
    work = adj.astype(np.int8).copy()
    eliminated: set[int] = set()
    bags: list[frozenset[int]] = []
    node_to_bag: dict[int, int] = {}  # Maps eliminated node -> bag index

    for v in ordering:
        neighbors = frozenset(
            u for u in range(n)
            if u != v and u not in eliminated and work[v, u]
        )
        bag = frozenset({v}) | neighbors
        bag_idx = len(bags)
        bags.append(bag)
        node_to_bag[v] = bag_idx
        eliminated.add(v)

        # Add fill edges
        nb_list = list(neighbors)
        for i in range(len(nb_list)):
            for j in range(i + 1, len(nb_list)):
                work[nb_list[i], nb_list[j]] = 1
                work[nb_list[j], nb_list[i]] = 1

    # Build tree adjacency: connect bag i to the earliest bag j > i that
    # shares a vertex with bag i (besides the eliminated vertex)
    n_bags = len(bags)
    tree_adj: list[list[int]] = [[] for _ in range(n_bags)]

    for i in range(n_bags):
        v = ordering[i]
        # Find the bag for the first neighbor of v in the ordering
        rest_of_bag = bags[i] - {v}
        if not rest_of_bag:
            continue
        # Find which bag was created for the first neighbor to be eliminated
        min_bag_idx = n_bags
        for nb in rest_of_bag:
            if nb in node_to_bag:
                bi = node_to_bag[nb]
                if bi < min_bag_idx and bi != i:
                    min_bag_idx = bi
        # Actually, connect to the next bag that contains a node from rest_of_bag
        for j in range(i + 1, n_bags):
            if bags[j] & rest_of_bag:
                tree_adj[i].append(j)
                tree_adj[j].append(i)
                break

    width = max((len(b) - 1 for b in bags), default=0)
    return TreeDecomposition(bags=bags, tree_adj=tree_adj, width=width)


def tree_decomposition(moral_adj: np.ndarray) -> TreeDecomposition:
    """Compute a tree decomposition of an undirected graph.

    Uses the min-fill heuristic for the elimination ordering.

    Parameters
    ----------
    moral_adj : np.ndarray
        Symmetric binary adjacency matrix (e.g. from :func:`moral_graph`).

    Returns
    -------
    TreeDecomposition
    """
    moral_adj = np.asarray(moral_adj, dtype=np.int8)
    n = moral_adj.shape[0]

    if n == 0:
        return TreeDecomposition(bags=[], tree_adj=[], width=0)

    if n == 1:
        return TreeDecomposition(
            bags=[frozenset({0})], tree_adj=[[]], width=0
        )

    # Try both heuristics and pick the one giving smaller width
    order_md = _min_degree_ordering(moral_adj)
    td_md = _ordering_to_tree_decomposition(moral_adj, order_md)

    order_mf = _min_fill_ordering(moral_adj)
    td_mf = _ordering_to_tree_decomposition(moral_adj, order_mf)

    return td_mf if td_mf.width <= td_md.width else td_md


def treewidth(moral_adj: np.ndarray) -> int:
    """Return the treewidth of an undirected graph.

    Parameters
    ----------
    moral_adj : np.ndarray
        Symmetric binary adjacency matrix.

    Returns
    -------
    int
        Treewidth (exact for small graphs, heuristic upper bound otherwise).
    """
    td = tree_decomposition(moral_adj)
    return td.width


def treewidth_of_dag(adj: AdjacencyMatrix) -> int:
    """Compute the treewidth of a DAG's moral graph.

    Convenience function that moralizes and computes treewidth in one step.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.

    Returns
    -------
    int
        Treewidth of the moral graph.
    """
    mg = moral_graph(adj)
    return treewidth(mg)


def is_chordal(adj: np.ndarray) -> bool:
    """Check whether an undirected graph is chordal.

    A graph is chordal if it admits a perfect elimination ordering.

    Parameters
    ----------
    adj : np.ndarray
        Symmetric binary adjacency matrix.

    Returns
    -------
    bool
    """
    peo = perfect_elimination_ordering(adj)
    return peo is not None


def perfect_elimination_ordering(adj: np.ndarray) -> list[int] | None:
    """Compute a perfect elimination ordering if the graph is chordal.

    Uses the maximum cardinality search (MCS) algorithm.

    Parameters
    ----------
    adj : np.ndarray
        Symmetric binary adjacency matrix.

    Returns
    -------
    list[int] | None
        A perfect elimination ordering, or None if the graph is not chordal.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return []

    # Maximum Cardinality Search (MCS)
    weight = np.zeros(n, dtype=int)
    ordering: list[int] = []
    chosen: set[int] = set()

    for _ in range(n):
        # Pick un-chosen vertex with max weight
        best = -1
        best_w = -1
        for v in range(n):
            if v not in chosen and weight[v] > best_w:
                best_w = weight[v]
                best = v
        ordering.append(best)
        chosen.add(best)
        for nb in np.nonzero(adj[best])[0]:
            nb = int(nb)
            if nb not in chosen:
                weight[nb] += 1

    ordering.reverse()  # PEO is reverse of MCS ordering

    # Verify: for each vertex v in the ordering, the neighbors of v that
    # come after v in the ordering must form a clique
    pos = {v: i for i, v in enumerate(ordering)}
    for idx, v in enumerate(ordering):
        later_neighbors = [
            u for u in np.nonzero(adj[v])[0]
            if pos[int(u)] > idx
        ]
        # Check these form a clique
        for i in range(len(later_neighbors)):
            for j in range(i + 1, len(later_neighbors)):
                if not adj[later_neighbors[i], later_neighbors[j]]:
                    return None  # Not chordal

    return ordering


def chordal_completion(adj: np.ndarray) -> np.ndarray:
    """Compute a minimal chordal completion (triangulation).

    Uses the min-fill heuristic to add edges that make the graph chordal.

    Parameters
    ----------
    adj : np.ndarray
        Symmetric adjacency matrix.

    Returns
    -------
    np.ndarray
        Symmetric adjacency matrix of the chordal completion.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    work = adj.copy()
    ordering = _min_fill_ordering(adj)
    eliminated: set[int] = set()

    for v in ordering:
        neighbors = [
            u for u in range(n)
            if u != v and u not in eliminated and work[v, u]
        ]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                work[neighbors[i], neighbors[j]] = 1
                work[neighbors[j], neighbors[i]] = 1
        eliminated.add(v)

    return work


def maximal_cliques(adj: np.ndarray) -> list[NodeSet]:
    """Find all maximal cliques in an undirected graph.

    Uses the Bron-Kerbosch algorithm with pivoting.

    Parameters
    ----------
    adj : np.ndarray
        Symmetric binary adjacency matrix.

    Returns
    -------
    list[NodeSet]
        List of maximal cliques.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    cliques: list[NodeSet] = []

    def _neighbors(v: int) -> set[int]:
        return set(int(u) for u in np.nonzero(adj[v])[0])

    def _bron_kerbosch(R: set[int], P: set[int], X: set[int]) -> None:
        if not P and not X:
            cliques.append(frozenset(R))
            return
        # Choose pivot: vertex in P ∪ X that maximizes |P ∩ N(u)|
        pivot = max(P | X, key=lambda u: len(P & _neighbors(u)))
        for v in list(P - _neighbors(pivot)):
            nv = _neighbors(v)
            _bron_kerbosch(R | {v}, P & nv, X & nv)
            P.remove(v)
            X.add(v)

    _bron_kerbosch(set(), set(range(n)), set())
    return cliques


@dataclass(slots=True)
class JunctionTree:
    """A junction tree (clique tree) derived from a chordal graph.

    Attributes
    ----------
    cliques : list[NodeSet]
        Maximal cliques of the chordal graph.
    tree_adj : list[list[int]]
        Adjacency list of the junction tree.
    separators : dict[tuple[int, int], NodeSet]
        Separator sets for each tree edge.
    """

    cliques: list[NodeSet]
    tree_adj: list[list[int]]
    separators: dict[tuple[int, int], NodeSet]


def junction_tree(adj: np.ndarray) -> JunctionTree:
    """Construct a junction tree from a chordal graph.

    If the graph is not chordal, it is first triangulated using
    the min-fill heuristic.

    Parameters
    ----------
    adj : np.ndarray
        Symmetric binary adjacency matrix (preferably chordal).

    Returns
    -------
    JunctionTree
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]

    if n == 0:
        return JunctionTree(cliques=[], tree_adj=[], separators={})

    # Triangulate if necessary
    if not is_chordal(adj):
        adj = chordal_completion(adj)

    # Find maximal cliques
    cliques = maximal_cliques(adj)
    if not cliques:
        # Each isolated node is a clique
        cliques = [frozenset({i}) for i in range(n)]

    nc = len(cliques)
    if nc <= 1:
        return JunctionTree(
            cliques=cliques,
            tree_adj=[[] for _ in range(nc)],
            separators={},
        )

    # Build maximum spanning tree over clique intersection weights
    # Weight of edge (i,j) = |C_i ∩ C_j|
    # Use Prim's algorithm
    in_tree: set[int] = {0}
    tree_adj_list: list[list[int]] = [[] for _ in range(nc)]
    separators: dict[tuple[int, int], NodeSet] = {}

    for _ in range(nc - 1):
        best_weight = -1
        best_i = -1
        best_j = -1
        for i in in_tree:
            for j in range(nc):
                if j in in_tree:
                    continue
                w = len(cliques[i] & cliques[j])
                if w > best_weight:
                    best_weight = w
                    best_i = i
                    best_j = j
        if best_j == -1:
            # Disconnected: pick any node not in tree
            for j in range(nc):
                if j not in in_tree:
                    best_i = min(in_tree)
                    best_j = j
                    break

        tree_adj_list[best_i].append(best_j)
        tree_adj_list[best_j].append(best_i)
        sep = cliques[best_i] & cliques[best_j]
        separators[(best_i, best_j)] = sep
        separators[(best_j, best_i)] = sep
        in_tree.add(best_j)

    return JunctionTree(
        cliques=cliques,
        tree_adj=tree_adj_list,
        separators=separators,
    )


def treewidth_feasibility(
    adj: AdjacencyMatrix,
    max_treewidth: int = 10,
) -> tuple[bool, int]:
    """Assess whether the DAG's moral graph treewidth is within budget.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    max_treewidth : int
        Maximum acceptable treewidth.

    Returns
    -------
    tuple[bool, int]
        ``(feasible, tw)`` where ``feasible`` is True if ``tw <= max_treewidth``.
    """
    tw = treewidth_of_dag(adj)
    return tw <= max_treewidth, tw
