"""
Separator-based algorithms for tree decompositions.

Implements minimal-separator enumeration, safe-separator detection,
clique-tree construction from chordal graphs, and atom decomposition.
These are used for divide-and-conquer strategies in treewidth computation
and as building blocks for improved tree decompositions.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from causalcert.types import NodeId, NodeSet


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Separator:
    """A vertex separator in an undirected graph.

    Attributes
    ----------
    vertices : NodeSet
        Vertices forming the separator.
    components : tuple[NodeSet, ...]
        Connected components of G − S.
    is_minimal : bool
        True if no proper subset of *vertices* separates the same pair.
    """

    vertices: NodeSet
    components: tuple[NodeSet, ...] = ()
    is_minimal: bool = False


@dataclass(frozen=True, slots=True)
class CliqueTree:
    """A clique tree (junction tree) of a chordal graph.

    Attributes
    ----------
    cliques : tuple[NodeSet, ...]
        Maximal cliques of the chordal graph.
    edges : tuple[tuple[int, int], ...]
        Edges of the clique tree (pairs of clique indices).
    separators : dict[tuple[int, int], NodeSet]
        Separator for each edge of the tree.
    """

    cliques: tuple[NodeSet, ...]
    edges: tuple[tuple[int, int], ...] = ()
    separators: dict[tuple[int, int], NodeSet] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Atom:
    """An atom in the atom decomposition of a graph.

    An atom is either a prime component with respect to minimal separators
    (i.e., it has no clique minimal separator), or a maximal clique.

    Attributes
    ----------
    vertices : NodeSet
        Vertices of the atom.
    is_clique : bool
        True if the atom is a clique.
    """

    vertices: NodeSet
    is_clique: bool = False


# ---------------------------------------------------------------------------
# Minimal separator enumeration
# ---------------------------------------------------------------------------


def _connected_components_after_removal(
    graph: nx.Graph,
    removed: NodeSet,
) -> list[NodeSet]:
    """Return connected components of G − removed as frozensets."""
    remaining = set(graph.nodes()) - set(removed)
    sub = graph.subgraph(remaining)
    return [frozenset(c) for c in nx.connected_components(sub)]


def _neighbourhood(graph: nx.Graph, component: NodeSet) -> NodeSet:
    """Return the open neighbourhood of a vertex set (vertices adjacent but outside)."""
    nbhd: set[NodeId] = set()
    for v in component:
        for u in graph.neighbors(v):
            if u not in component:
                nbhd.add(u)
    return frozenset(nbhd)


def is_minimal_separator(
    graph: nx.Graph,
    sep: NodeSet,
) -> bool:
    """Check whether *sep* is a minimal vertex separator.

    A set S is a minimal separator if there exist at least two connected
    components C₁, C₂ of G − S such that N(C₁) = S and N(C₂) = S.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.
    sep : NodeSet
        Candidate separator.

    Returns
    -------
    bool
    """
    components = _connected_components_after_removal(graph, sep)
    if len(components) < 2:
        return False

    full_count = 0
    for comp in components:
        nbhd = _neighbourhood(graph, comp)
        if nbhd == sep:
            full_count += 1
    return full_count >= 2


def enumerate_minimal_separators(graph: nx.Graph) -> list[Separator]:
    """Enumerate all minimal vertex separators of a graph.

    Uses the algorithm of Berry, Bordat, and Cogis (2000): for each vertex
    *v*, the neighbourhood of each connected component of ``G − N[v]`` that
    does not contain *v* is a candidate minimal separator.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    list[Separator]
        All minimal separators, without duplicates.

    Notes
    -----
    Complexity is O(n³) in the worst case.
    """
    n = graph.number_of_nodes()
    if n <= 2:
        return []

    found: set[NodeSet] = set()
    result: list[Separator] = []

    for v in graph.nodes():
        # Closed neighbourhood of v
        closed_nbhd = frozenset({v}) | frozenset(graph.neighbors(v))
        remaining = set(graph.nodes()) - set(closed_nbhd)
        if not remaining:
            continue

        sub = graph.subgraph(remaining)
        for comp in nx.connected_components(sub):
            comp_fs = frozenset(comp)
            sep = _neighbourhood(graph, comp_fs)
            if sep and sep not in found:
                # Verify minimality
                components = _connected_components_after_removal(graph, sep)
                if _is_min_sep_via_components(graph, sep, components):
                    found.add(sep)
                    result.append(Separator(
                        vertices=sep,
                        components=tuple(components),
                        is_minimal=True,
                    ))

    return result


def _is_min_sep_via_components(
    graph: nx.Graph,
    sep: NodeSet,
    components: list[NodeSet],
) -> bool:
    """Quick check: a separator is minimal iff at least two components are full."""
    full_count = 0
    for comp in components:
        nbhd = _neighbourhood(graph, comp)
        if nbhd == sep:
            full_count += 1
            if full_count >= 2:
                return True
    return False


def enumerate_minimal_separators_bounded(
    graph: nx.Graph,
    max_size: int,
) -> list[Separator]:
    """Enumerate minimal separators of size at most *max_size*.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.
    max_size : int
        Maximum separator size.

    Returns
    -------
    list[Separator]
        Minimal separators with at most *max_size* vertices.
    """
    all_seps = enumerate_minimal_separators(graph)
    return [s for s in all_seps if len(s.vertices) <= max_size]


# ---------------------------------------------------------------------------
# Safe separator detection
# ---------------------------------------------------------------------------


def is_clique(graph: nx.Graph, vertices: NodeSet) -> bool:
    """Check whether a set of vertices forms a clique.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.
    vertices : NodeSet
        Vertex set.

    Returns
    -------
    bool
    """
    vlist = list(vertices)
    for i in range(len(vlist)):
        for j in range(i + 1, len(vlist)):
            if not graph.has_edge(vlist[i], vlist[j]):
                return False
    return True


def is_safe_separator(
    graph: nx.Graph,
    sep: NodeSet,
) -> bool:
    """Check whether a separator is *safe* for divide-and-conquer.

    A minimal separator S is safe if it is a clique in the graph.  Safe
    separators can be used for divide-and-conquer treewidth computation
    because the optimal tree decomposition of each piece can be composed.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.
    sep : NodeSet
        Minimal separator.

    Returns
    -------
    bool
        True if *sep* is both minimal and a clique.
    """
    return is_clique(graph, sep) and is_minimal_separator(graph, sep)


def find_safe_separators(graph: nx.Graph) -> list[Separator]:
    """Find all safe (clique) minimal separators.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    list[Separator]
        All minimal separators that are cliques.
    """
    all_seps = enumerate_minimal_separators(graph)
    return [s for s in all_seps if is_clique(graph, s.vertices)]


# ---------------------------------------------------------------------------
# Clique tree construction
# ---------------------------------------------------------------------------


def build_clique_tree(graph: nx.Graph) -> CliqueTree:
    """Build a clique tree from a chordal graph.

    The clique tree is a tree whose nodes are the maximal cliques and whose
    edges satisfy the running intersection property.  It is built as the
    maximum-weight spanning tree of the clique intersection graph.

    Parameters
    ----------
    graph : nx.Graph
        A chordal graph.

    Returns
    -------
    CliqueTree

    Raises
    ------
    ValueError
        If the graph is not chordal.
    """
    if not nx.is_chordal(graph) and graph.number_of_edges() > 0:
        raise ValueError("Graph is not chordal.")

    cliques_raw = list(nx.find_cliques(graph))
    if not cliques_raw:
        nodes = list(graph.nodes())
        cliques_raw = [{v} for v in nodes] if nodes else []

    cliques = tuple(frozenset(c) for c in cliques_raw)
    nc = len(cliques)

    if nc <= 1:
        return CliqueTree(cliques=cliques, edges=(), separators={})

    # Maximum spanning tree via Prim's with intersection weights
    in_tree: set[int] = {0}
    edges: list[tuple[int, int]] = []
    separators: dict[tuple[int, int], NodeSet] = {}

    for _ in range(nc - 1):
        best_w = -1
        best_i = -1
        best_j = -1

        for i in in_tree:
            for j in range(nc):
                if j in in_tree:
                    continue
                w = len(cliques[i] & cliques[j])
                if w > best_w:
                    best_w = w
                    best_i = i
                    best_j = j

        if best_j == -1:
            for j in range(nc):
                if j not in in_tree:
                    best_i = min(in_tree)
                    best_j = j
                    break

        edges.append((best_i, best_j))
        sep = cliques[best_i] & cliques[best_j]
        separators[(best_i, best_j)] = sep
        separators[(best_j, best_i)] = sep
        in_tree.add(best_j)

    return CliqueTree(
        cliques=cliques,
        edges=tuple(edges),
        separators=separators,
    )


def clique_tree_from_elimination(
    graph: nx.Graph,
    ordering: Sequence[NodeId],
) -> CliqueTree:
    """Build a clique tree by first triangulating with the given ordering.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph (need not be chordal).
    ordering : Sequence[NodeId]
        Elimination ordering used for triangulation.

    Returns
    -------
    CliqueTree
        Clique tree of the triangulated graph.
    """
    from causalcert.treewidth.elimination import triangulate

    H = triangulate(graph, ordering)
    return build_clique_tree(H)


# ---------------------------------------------------------------------------
# Atom decomposition
# ---------------------------------------------------------------------------


def atom_decomposition(graph: nx.Graph) -> list[Atom]:
    """Compute the atom decomposition of a graph.

    The atom decomposition recursively splits the graph at clique minimal
    separators until no such separators remain.  Each resulting piece is
    an *atom*.  For a chordal graph, every atom is a maximal clique.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    list[Atom]
        Atoms of the decomposition, in no particular order.
    """
    if graph.number_of_nodes() == 0:
        return []

    atoms: list[Atom] = []
    _atom_decompose_recursive(graph, atoms)
    return atoms


def _atom_decompose_recursive(
    graph: nx.Graph,
    atoms: list[Atom],
) -> None:
    """Recursively split *graph* at clique minimal separators."""
    nodes = frozenset(graph.nodes())
    if graph.number_of_nodes() <= 1:
        atoms.append(Atom(vertices=nodes, is_clique=True))
        return

    # Find a clique minimal separator
    safe_seps = find_safe_separators(graph)
    if not safe_seps:
        # No clique minimal separator — this is an atom
        atoms.append(Atom(
            vertices=nodes,
            is_clique=is_clique(graph, nodes),
        ))
        return

    # Split at the first safe separator
    sep = safe_seps[0]
    components = _connected_components_after_removal(graph, sep.vertices)

    for comp in components:
        # Each piece is the component plus the separator
        piece_nodes = comp | sep.vertices
        piece_graph = graph.subgraph(piece_nodes).copy()
        _atom_decompose_recursive(piece_graph, atoms)


def atom_graph(atoms: list[Atom]) -> nx.Graph:
    """Build the atom graph where atoms sharing vertices are adjacent.

    Parameters
    ----------
    atoms : list[Atom]
        Atom decomposition.

    Returns
    -------
    nx.Graph
        Graph whose nodes are atom indices and edges connect atoms
        sharing at least one vertex.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if atoms[i].vertices & atoms[j].vertices:
                G.add_edge(i, j)
    return G


# ---------------------------------------------------------------------------
# Separator-based divide and conquer for treewidth
# ---------------------------------------------------------------------------


def decompose_via_safe_separators(
    graph: nx.Graph,
) -> list[nx.Graph]:
    """Split a graph at all safe (clique minimal) separators.

    Returns the resulting sub-problems, each of which is a graph that has
    no clique minimal separator.  Each can be decomposed independently and
    the results composed into a full tree decomposition.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    list[nx.Graph]
        Pieces after splitting.
    """
    pieces: list[nx.Graph] = []
    _split_recursive(graph, pieces)
    return pieces


def _split_recursive(graph: nx.Graph, pieces: list[nx.Graph]) -> None:
    """Recursively split at safe separators."""
    safe_seps = find_safe_separators(graph)
    if not safe_seps:
        pieces.append(graph.copy())
        return

    sep = safe_seps[0]
    components = _connected_components_after_removal(graph, sep.vertices)
    for comp in components:
        piece_nodes = comp | sep.vertices
        piece = graph.subgraph(piece_nodes).copy()
        _split_recursive(piece, pieces)


# ---------------------------------------------------------------------------
# Separator width bound
# ---------------------------------------------------------------------------


def separator_based_lower_bound(graph: nx.Graph) -> int:
    """Lower bound on treewidth from minimal separator sizes.

    The treewidth is at least as large as the minimum size of any minimal
    separator (minus one) that separates connected components.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    int
        Lower bound on treewidth.
    """
    seps = enumerate_minimal_separators(graph)
    if not seps:
        # For a complete graph the treewidth is n-1
        n = graph.number_of_nodes()
        if n <= 1:
            return 0
        if graph.number_of_edges() == n * (n - 1) // 2:
            return n - 1
        return 0
    return min(len(s.vertices) for s in seps)


__all__ = [
    "Separator",
    "CliqueTree",
    "Atom",
    "is_minimal_separator",
    "enumerate_minimal_separators",
    "enumerate_minimal_separators_bounded",
    "is_clique",
    "is_safe_separator",
    "find_safe_separators",
    "build_clique_tree",
    "clique_tree_from_elimination",
    "atom_decomposition",
    "atom_graph",
    "decompose_via_safe_separators",
    "separator_based_lower_bound",
]
