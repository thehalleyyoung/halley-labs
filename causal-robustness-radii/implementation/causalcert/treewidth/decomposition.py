"""
Tree decomposition algorithms.

Constructs tree decompositions of undirected graphs using heuristic
elimination orderings.  The main entry point is
:func:`compute_tree_decomposition`, which selects among min-degree,
min-fill, and other heuristics, returning a :class:`TreeDecomposition`
from :mod:`causalcert.treewidth.types`.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from causalcert.types import AdjacencyMatrix, NodeId, NodeSet, TreewidthBound
from causalcert.treewidth.types import EliminationOrdering, TreeBag, TreeDecomposition
from causalcert.treewidth.elimination import (
    best_heuristic_ordering,
    compute_treewidth_lower_bound,
    compute_treewidth_upper_bound,
    min_degree_ordering,
    min_fill_ordering,
    min_width_ordering,
    max_cardinality_search,
    ordering_to_decomposition,
    triangulate,
    detect_perfect_elimination_ordering,
)


# ---------------------------------------------------------------------------
# Graph ↔ adjacency-matrix helpers
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_tree_decomposition(
    graph: nx.Graph,
    method: str = "min_fill",
) -> TreeDecomposition:
    """Compute a tree decomposition of an undirected graph.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph (e.g. a moral graph).
    method : str
        Heuristic to use for the elimination ordering.  One of:

        * ``"min_fill"`` — minimise fill edges at each step (default).
        * ``"min_degree"`` — eliminate minimum-degree vertex.
        * ``"min_width"`` — greedy degree heuristic.
        * ``"mcs"`` — maximum cardinality search.
        * ``"best"`` — try all heuristics and pick the smallest width.

    Returns
    -------
    TreeDecomposition
        A valid tree decomposition.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return TreeDecomposition(bags=(), width=0, root_id=0, n_vertices=0)

    if n == 1:
        v = list(graph.nodes())[0]
        bag = TreeBag(bag_id=0, vertices=frozenset({v}))
        return TreeDecomposition(
            bags=(bag,), width=0, root_id=0, n_vertices=1,
        )

    _heuristics = {
        "min_fill": min_fill_ordering,
        "min_degree": min_degree_ordering,
        "min_width": min_width_ordering,
        "mcs": max_cardinality_search,
    }

    if method == "best":
        eo = best_heuristic_ordering(graph)
    elif method in _heuristics:
        eo = _heuristics[method](graph)
    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose from "
            f"{list(_heuristics) + ['best']}."
        )

    td = ordering_to_decomposition(graph, eo)
    return td


def compute_tree_decomposition_from_adj(
    adj: AdjacencyMatrix,
    method: str = "min_fill",
) -> TreeDecomposition:
    """Compute a tree decomposition from an adjacency matrix.

    Convenience wrapper around :func:`compute_tree_decomposition` that
    first converts the adjacency matrix to a networkx graph.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Symmetric binary adjacency matrix.
    method : str
        Heuristic method (see :func:`compute_tree_decomposition`).

    Returns
    -------
    TreeDecomposition
    """
    G = _graph_from_adj(adj)
    return compute_tree_decomposition(G, method=method)


# ---------------------------------------------------------------------------
# Width computation
# ---------------------------------------------------------------------------


def width_of_decomposition(td: TreeDecomposition) -> int:
    """Compute the width of a tree decomposition.

    The width is ``max_i |X_i| - 1`` where ``X_i`` are the bags.

    Parameters
    ----------
    td : TreeDecomposition
        A tree decomposition.

    Returns
    -------
    int
        Width of the decomposition.
    """
    if not td.bags:
        return 0
    return max(len(bag.vertices) - 1 for bag in td.bags)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_tree_decomposition(
    graph: nx.Graph,
    td: TreeDecomposition,
) -> tuple[bool, list[str]]:
    """Validate that a tree decomposition satisfies all three TD properties.

    1. **Vertex coverage**: every vertex appears in at least one bag.
    2. **Edge coverage**: for every edge ``(u, v)`` there is a bag containing
       both *u* and *v*.
    3. **Running intersection**: the bags containing any vertex *v* form a
       connected sub-tree.

    Parameters
    ----------
    graph : nx.Graph
        The original undirected graph.
    td : TreeDecomposition
        Tree decomposition to validate.

    Returns
    -------
    tuple[bool, list[str]]
        ``(valid, errors)`` where *valid* is ``True`` if all properties hold,
        and *errors* is a list of human-readable violation descriptions.
    """
    errors: list[str] = []

    # 1. Vertex coverage
    covered = set()
    for bag in td.bags:
        covered |= set(bag.vertices)
    for v in graph.nodes():
        if v not in covered:
            errors.append(f"Vertex {v} is not in any bag.")

    # 2. Edge coverage
    for u, v in graph.edges():
        found = False
        for bag in td.bags:
            if u in bag.vertices and v in bag.vertices:
                found = True
                break
        if not found:
            errors.append(f"Edge ({u}, {v}) is not covered by any bag.")

    # 3. Running intersection property
    # Build tree adjacency
    tree = nx.Graph()
    for bag in td.bags:
        tree.add_node(bag.bag_id)
    for bag in td.bags:
        for cid in bag.children_ids:
            tree.add_edge(bag.bag_id, cid)

    for v in graph.nodes():
        bags_with_v = [bag.bag_id for bag in td.bags if v in bag.vertices]
        if len(bags_with_v) <= 1:
            continue
        sub = tree.subgraph(bags_with_v)
        if not nx.is_connected(sub):
            errors.append(
                f"Bags containing vertex {v} do not form a connected sub-tree."
            )

    return (len(errors) == 0, errors)


# ---------------------------------------------------------------------------
# Triangulated-graph to tree decomposition
# ---------------------------------------------------------------------------


def decomposition_from_chordal_graph(graph: nx.Graph) -> TreeDecomposition:
    """Build a tree decomposition from a chordal graph using its clique tree.

    The bags of the decomposition are the maximal cliques of the chordal
    graph, connected via a maximum-weight spanning tree of the clique
    intersection graph.

    Parameters
    ----------
    graph : nx.Graph
        A chordal graph.

    Returns
    -------
    TreeDecomposition

    Raises
    ------
    ValueError
        If the graph is not chordal.
    """
    peo = detect_perfect_elimination_ordering(graph)
    if peo is None:
        raise ValueError("Graph is not chordal — cannot use clique-tree method.")

    # Find maximal cliques using networkx on the chordal graph
    cliques = list(nx.find_cliques(graph))
    if not cliques:
        n = graph.number_of_nodes()
        if n == 0:
            return TreeDecomposition(bags=(), width=0, root_id=0, n_vertices=0)
        cliques = [{v} for v in graph.nodes()]

    nc = len(cliques)
    clique_sets = [frozenset(c) for c in cliques]

    if nc == 1:
        bag = TreeBag(bag_id=0, vertices=clique_sets[0])
        return TreeDecomposition(
            bags=(bag,),
            width=len(clique_sets[0]) - 1,
            root_id=0,
            n_vertices=graph.number_of_nodes(),
            elimination_ordering=peo,
        )

    # Build maximum spanning tree via Prim's algorithm
    in_tree: set[int] = {0}
    parent: dict[int, int | None] = {0: None}
    children: dict[int, list[int]] = defaultdict(list)

    for _ in range(nc - 1):
        best_w = -1
        best_i = -1
        best_j = -1
        for i in in_tree:
            for j in range(nc):
                if j in in_tree:
                    continue
                w = len(clique_sets[i] & clique_sets[j])
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
        children[best_i].append(best_j)
        parent[best_j] = best_i
        in_tree.add(best_j)

    bags: list[TreeBag] = []
    for i in range(nc):
        bags.append(TreeBag(
            bag_id=i,
            vertices=clique_sets[i],
            parent_id=parent.get(i),
            children_ids=tuple(children.get(i, [])),
        ))

    width = max(len(c) - 1 for c in clique_sets)
    return TreeDecomposition(
        bags=tuple(bags),
        width=width,
        root_id=0,
        n_vertices=graph.number_of_nodes(),
        elimination_ordering=peo,
    )


# ---------------------------------------------------------------------------
# Re-rooting
# ---------------------------------------------------------------------------


def reroot_decomposition(
    td: TreeDecomposition,
    new_root: int,
) -> TreeDecomposition:
    """Re-root a tree decomposition at a different bag.

    Parameters
    ----------
    td : TreeDecomposition
        Original tree decomposition.
    new_root : int
        Bag id of the new root.

    Returns
    -------
    TreeDecomposition
        A new decomposition rooted at *new_root*.
    """
    # Build undirected adjacency from the existing parent/child structure
    adj_map: dict[int, set[int]] = defaultdict(set)
    bag_map: dict[int, TreeBag] = {}
    for bag in td.bags:
        bag_map[bag.bag_id] = bag
        for cid in bag.children_ids:
            adj_map[bag.bag_id].add(cid)
            adj_map[cid].add(bag.bag_id)

    # BFS from new_root to assign parents and children
    visited: set[int] = set()
    queue = deque([new_root])
    visited.add(new_root)
    new_parent: dict[int, int | None] = {new_root: None}
    new_children: dict[int, list[int]] = defaultdict(list)

    while queue:
        bid = queue.popleft()
        for nb in adj_map[bid]:
            if nb not in visited:
                visited.add(nb)
                new_parent[nb] = bid
                new_children[bid].append(nb)
                queue.append(nb)

    bags: list[TreeBag] = []
    for bag in td.bags:
        bid = bag.bag_id
        bags.append(TreeBag(
            bag_id=bid,
            vertices=bag.vertices,
            parent_id=new_parent.get(bid),
            children_ids=tuple(new_children.get(bid, [])),
        ))

    return TreeDecomposition(
        bags=tuple(bags),
        width=td.width,
        root_id=new_root,
        n_vertices=td.n_vertices,
        elimination_ordering=td.elimination_ordering,
    )


# ---------------------------------------------------------------------------
# Subsumption-based simplification
# ---------------------------------------------------------------------------


def simplify_decomposition(td: TreeDecomposition) -> TreeDecomposition:
    """Remove redundant bags that are subsets of their neighbours.

    If a bag ``X_i ⊆ X_j`` for some neighbour ``X_j``, then ``X_i`` can be
    removed and its other neighbours connected to ``X_j``.

    Parameters
    ----------
    td : TreeDecomposition
        Input tree decomposition.

    Returns
    -------
    TreeDecomposition
        Simplified decomposition with no subsumed bags.
    """
    # Build undirected adjacency
    adj_map: dict[int, set[int]] = defaultdict(set)
    bag_map: dict[int, TreeBag] = {}
    for bag in td.bags:
        bag_map[bag.bag_id] = bag
        for cid in bag.children_ids:
            adj_map[bag.bag_id].add(cid)
            adj_map[cid].add(bag.bag_id)

    removed: set[int] = set()
    changed = True

    while changed:
        changed = False
        for bid in list(bag_map.keys()):
            if bid in removed:
                continue
            bag = bag_map[bid]
            for nb_id in list(adj_map[bid]):
                if nb_id in removed:
                    continue
                nb_bag = bag_map[nb_id]
                if bag.vertices <= nb_bag.vertices and bid != nb_id:
                    # Subsume bid into nb_id
                    for other in list(adj_map[bid]):
                        if other != nb_id and other not in removed:
                            adj_map[nb_id].add(other)
                            adj_map[other].discard(bid)
                            adj_map[other].add(nb_id)
                    adj_map[nb_id].discard(bid)
                    del adj_map[bid]
                    removed.add(bid)
                    changed = True
                    break

    # Rebuild decomposition
    remaining = [bid for bid in bag_map if bid not in removed]
    if not remaining:
        return td

    # Re-index
    old_to_new = {old: i for i, old in enumerate(remaining)}
    new_root = old_to_new.get(td.root_id, 0)

    # Rebuild with BFS from root
    new_adj: dict[int, set[int]] = defaultdict(set)
    for bid in remaining:
        for nb in adj_map.get(bid, ()):
            if nb in old_to_new:
                new_adj[old_to_new[bid]].add(old_to_new[nb])

    visited_set: set[int] = set()
    bfs_queue = deque([new_root])
    visited_set.add(new_root)
    parent_map: dict[int, int | None] = {new_root: None}
    children_map: dict[int, list[int]] = defaultdict(list)

    while bfs_queue:
        cur = bfs_queue.popleft()
        for nb in new_adj[cur]:
            if nb not in visited_set:
                visited_set.add(nb)
                parent_map[nb] = cur
                children_map[cur].append(nb)
                bfs_queue.append(nb)

    bags: list[TreeBag] = []
    for new_id, old_id in enumerate(remaining):
        bags.append(TreeBag(
            bag_id=new_id,
            vertices=bag_map[old_id].vertices,
            parent_id=parent_map.get(new_id),
            children_ids=tuple(children_map.get(new_id, [])),
        ))

    width = max((len(b.vertices) - 1 for b in bags), default=0)
    return TreeDecomposition(
        bags=tuple(bags),
        width=width,
        root_id=new_root,
        n_vertices=td.n_vertices,
        elimination_ordering=td.elimination_ordering,
    )


# ---------------------------------------------------------------------------
# Treewidth bounds via DecompositionAlgorithm protocol
# ---------------------------------------------------------------------------


def compute_treewidth_bounds(graph: nx.Graph) -> TreewidthBound:
    """Compute lower and upper bounds on the treewidth.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    TreewidthBound
        Lower bound, upper bound, and whether the result is exact.
    """
    lb = compute_treewidth_lower_bound(graph)
    ub = compute_treewidth_upper_bound(graph)
    return TreewidthBound(lower=lb, upper=ub, exact=(lb == ub))


def compute_treewidth_bounds_from_adj(adj: AdjacencyMatrix) -> TreewidthBound:
    """Compute treewidth bounds from an adjacency matrix.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Symmetric binary adjacency matrix.

    Returns
    -------
    TreewidthBound
    """
    G = _graph_from_adj(adj)
    return compute_treewidth_bounds(G)


# ---------------------------------------------------------------------------
# Connected-component decomposition
# ---------------------------------------------------------------------------


def decompose_by_components(graph: nx.Graph) -> list[TreeDecomposition]:
    """Compute a tree decomposition for each connected component.

    This can be more efficient than decomposing the full graph, especially
    when the graph has many small components.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.

    Returns
    -------
    list[TreeDecomposition]
        One decomposition per connected component.
    """
    components = list(nx.connected_components(graph))
    results: list[TreeDecomposition] = []

    for comp in components:
        sub = graph.subgraph(comp).copy()
        td = compute_tree_decomposition(sub, method="best")
        results.append(td)

    return results


def merge_component_decompositions(
    decompositions: Sequence[TreeDecomposition],
) -> TreeDecomposition:
    """Merge decompositions from separate components into one.

    Creates a new root bag (empty) that connects to the roots of each
    component decomposition.

    Parameters
    ----------
    decompositions : Sequence[TreeDecomposition]
        Per-component decompositions.

    Returns
    -------
    TreeDecomposition
        Single merged decomposition.
    """
    if not decompositions:
        return TreeDecomposition(bags=(), width=0, root_id=0, n_vertices=0)

    if len(decompositions) == 1:
        return decompositions[0]

    all_bags: list[TreeBag] = []
    offset = 0
    component_roots: list[int] = []
    total_vertices = 0
    max_width = 0

    for td in decompositions:
        max_width = max(max_width, td.width)
        total_vertices += td.n_vertices
        component_roots.append(td.root_id + offset)

        for bag in td.bags:
            new_parent = None
            if bag.parent_id is not None:
                new_parent = bag.parent_id + offset
            new_children = tuple(c + offset for c in bag.children_ids)
            all_bags.append(TreeBag(
                bag_id=bag.bag_id + offset,
                vertices=bag.vertices,
                parent_id=new_parent,
                children_ids=new_children,
            ))
        offset += len(td.bags)

    # Add a synthetic root connecting all component roots
    root_id = offset
    root_bag = TreeBag(
        bag_id=root_id,
        vertices=frozenset(),
        parent_id=None,
        children_ids=tuple(component_roots),
    )

    # Update component roots to have the new root as parent
    updated_bags: list[TreeBag] = []
    for bag in all_bags:
        if bag.bag_id in component_roots:
            updated_bags.append(TreeBag(
                bag_id=bag.bag_id,
                vertices=bag.vertices,
                parent_id=root_id,
                children_ids=bag.children_ids,
            ))
        else:
            updated_bags.append(bag)

    updated_bags.append(root_bag)

    return TreeDecomposition(
        bags=tuple(updated_bags),
        width=max_width,
        root_id=root_id,
        n_vertices=total_vertices,
    )


__all__ = [
    "compute_tree_decomposition",
    "compute_tree_decomposition_from_adj",
    "width_of_decomposition",
    "validate_tree_decomposition",
    "decomposition_from_chordal_graph",
    "reroot_decomposition",
    "simplify_decomposition",
    "compute_treewidth_bounds",
    "compute_treewidth_bounds_from_adj",
    "decompose_by_components",
    "merge_component_decompositions",
]
