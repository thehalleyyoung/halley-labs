"""
Ancestral set extraction (ALG 1).

Given a DAG and a set of target nodes, computes the ancestral set — all nodes
from which there exists a directed path to at least one target — using a
reverse BFS.  Also provides topological ordering and descendant queries.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

from causalcert.types import AdjacencyMatrix, EdgeTuple, NodeId, NodeSet
from causalcert.exceptions import CyclicGraphError


def ancestors(adj: AdjacencyMatrix, targets: NodeSet) -> NodeSet:
    """Return the ancestral set of *targets* in the DAG.

    The ancestral set includes *targets* themselves together with every
    node from which a directed path to some member of *targets* exists.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    targets : NodeSet
        Nodes whose ancestors are sought.

    Returns
    -------
    NodeSet
        Ancestral set (includes *targets*).
    """
    adj = np.asarray(adj, dtype=np.int8)
    result: set[int] = set(targets)
    queue = deque(targets)
    while queue:
        v = queue.popleft()
        for p in np.nonzero(adj[:, v])[0]:
            p = int(p)
            if p not in result:
                result.add(p)
                queue.append(p)
    return frozenset(result)


def ancestors_of_node(adj: AdjacencyMatrix, v: NodeId) -> NodeSet:
    """Return the ancestors of a single node *v* (not including *v*)."""
    adj = np.asarray(adj, dtype=np.int8)
    result: set[int] = set()
    queue = deque[int]()
    for p in np.nonzero(adj[:, v])[0]:
        p = int(p)
        result.add(p)
        queue.append(p)
    while queue:
        node = queue.popleft()
        for p in np.nonzero(adj[:, node])[0]:
            p = int(p)
            if p not in result:
                result.add(p)
                queue.append(p)
    return frozenset(result)


def descendants(adj: AdjacencyMatrix, sources: NodeSet) -> NodeSet:
    """Return the descendant set of *sources* (including *sources*).

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    sources : NodeSet
        Nodes whose descendants are sought.

    Returns
    -------
    NodeSet
    """
    adj = np.asarray(adj, dtype=np.int8)
    result: set[int] = set(sources)
    queue = deque(sources)
    while queue:
        v = queue.popleft()
        for c in np.nonzero(adj[v, :])[0]:
            c = int(c)
            if c not in result:
                result.add(c)
                queue.append(c)
    return frozenset(result)


def descendants_of_node(adj: AdjacencyMatrix, v: NodeId) -> NodeSet:
    """Return the descendants of a single node *v* (not including *v*)."""
    adj = np.asarray(adj, dtype=np.int8)
    result: set[int] = set()
    queue = deque[int]()
    for c in np.nonzero(adj[v, :])[0]:
        c = int(c)
        result.add(c)
        queue.append(c)
    while queue:
        node = queue.popleft()
        for c in np.nonzero(adj[node, :])[0]:
            c = int(c)
            if c not in result:
                result.add(c)
                queue.append(c)
    return frozenset(result)


def topological_order(adj: AdjacencyMatrix) -> list[NodeId]:
    """Return a topological ordering of the DAG.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix (must be acyclic).

    Returns
    -------
    list[NodeId]
        Nodes in topological order (parents before children).

    Raises
    ------
    CyclicGraphError
        If the graph contains a cycle.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = deque(int(i) for i in range(n) if in_deg[i] == 0)
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in np.nonzero(adj[node])[0]:
            child = int(child)
            in_deg[child] -= 1
            if in_deg[child] == 0:
                queue.append(child)
    if len(order) != n:
        raise CyclicGraphError()
    return order


def reverse_topological_order(adj: AdjacencyMatrix) -> list[NodeId]:
    """Return a reverse topological ordering (children before parents)."""
    return list(reversed(topological_order(adj)))


def ancestral_subgraph(adj: AdjacencyMatrix, targets: NodeSet) -> AdjacencyMatrix:
    """Return the subgraph induced by the ancestral set of *targets*.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Full DAG adjacency matrix.
    targets : NodeSet
        Target nodes.

    Returns
    -------
    AdjacencyMatrix
        Adjacency matrix of the induced subgraph on ``ancestors(adj, targets)``.
        Rows/columns correspond to the sorted list of ancestral nodes.
    """
    adj = np.asarray(adj, dtype=np.int8)
    anc = ancestors(adj, targets)
    node_list = sorted(anc)
    idx = np.array(node_list, dtype=int)
    return adj[np.ix_(idx, idx)].copy()


def ancestral_subgraph_with_mapping(
    adj: AdjacencyMatrix, targets: NodeSet
) -> tuple[AdjacencyMatrix, dict[int, int], dict[int, int]]:
    """Return the ancestral subgraph along with index mappings.

    Returns
    -------
    tuple[AdjacencyMatrix, dict[int, int], dict[int, int]]
        ``(sub_adj, old_to_new, new_to_old)`` where old_to_new maps
        original node ids to subgraph ids and new_to_old is the inverse.
    """
    adj = np.asarray(adj, dtype=np.int8)
    anc = ancestors(adj, targets)
    node_list = sorted(anc)
    old_to_new = {old: new for new, old in enumerate(node_list)}
    new_to_old = {new: old for old, new in old_to_new.items()}
    idx = np.array(node_list, dtype=int)
    sub_adj = adj[np.ix_(idx, idx)].copy()
    return sub_adj, old_to_new, new_to_old


def candidate_edges(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> list[EdgeTuple]:
    """Identify candidate edges for fragility analysis.

    Only edges within the ancestral set of {treatment, outcome} can
    affect the causal conclusion between treatment and outcome.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment : NodeId
        Treatment node.
    outcome : NodeId
        Outcome node.

    Returns
    -------
    list[EdgeTuple]
        Edges within the ancestral subgraph of treatment and outcome.
    """
    adj = np.asarray(adj, dtype=np.int8)
    anc = ancestors(adj, frozenset({treatment, outcome}))
    edges: list[EdgeTuple] = []
    anc_list = sorted(anc)
    for u in anc_list:
        for v in anc_list:
            if adj[u, v]:
                edges.append((u, v))
    return edges


def relevant_edit_set(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> list[tuple[str, NodeId, NodeId]]:
    """Compute the relevant edit set for fragility analysis.

    Returns all possible single-edge edits (add, delete, reverse) within
    the ancestral set of {treatment, outcome} that could affect the
    causal conclusion.

    Returns
    -------
    list[tuple[str, NodeId, NodeId]]
        List of ``(edit_type, source, target)`` tuples.
    """
    adj = np.asarray(adj, dtype=np.int8)
    anc = ancestors(adj, frozenset({treatment, outcome}))
    anc_list = sorted(anc)
    edits: list[tuple[str, NodeId, NodeId]] = []

    for u in anc_list:
        for v in anc_list:
            if u == v:
                continue
            if adj[u, v]:
                # Can delete or reverse this edge
                edits.append(("delete", u, v))
                edits.append(("reverse", u, v))
            else:
                # Can potentially add this edge
                edits.append(("add", u, v))

    return edits


def districts(adj: AdjacencyMatrix) -> list[NodeSet]:
    """Identify districts (c-components) of the DAG.

    A district is a maximal set of nodes connected via bidirected edges
    in the corresponding ADMG. For a DAG (no hidden variables), districts
    are connected components in the graph where nodes sharing a common
    child are connected.

    In a pure DAG context, we define districts via the moral graph's
    connected components restricted to nodes that share children.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.

    Returns
    -------
    list[NodeSet]
        List of districts.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]

    # Build an undirected "spouse" graph: connect parents that share a child
    spouse = np.zeros((n, n), dtype=np.int8)
    for child in range(n):
        parents = list(np.nonzero(adj[:, child])[0])
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                spouse[parents[i], parents[j]] = 1
                spouse[parents[j], parents[i]] = 1

    # Find connected components in the spouse graph
    visited: set[int] = set()
    components: list[NodeSet] = []

    for start in range(n):
        if start in visited:
            continue
        # BFS in the spouse graph
        comp: set[int] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in comp:
                continue
            comp.add(node)
            visited.add(node)
            for nb in np.nonzero(spouse[node])[0]:
                nb = int(nb)
                if nb not in comp:
                    queue.append(nb)
        components.append(frozenset(comp))

    return components


def c_components(adj: AdjacencyMatrix) -> list[NodeSet]:
    """Alias for :func:`districts`."""
    return districts(adj)


def is_ancestral_set(adj: AdjacencyMatrix, node_set: NodeSet) -> bool:
    """Check whether *node_set* is an ancestral set.

    A set S is ancestral if for every node in S, all its ancestors
    are also in S.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    node_set : NodeSet
        Set of nodes to check.

    Returns
    -------
    bool
    """
    adj = np.asarray(adj, dtype=np.int8)
    s = set(node_set)
    for v in s:
        for p in np.nonzero(adj[:, v])[0]:
            if int(p) not in s:
                return False
    return True


def proper_ancestors(adj: AdjacencyMatrix, v: NodeId) -> NodeSet:
    """Return proper ancestors of *v* (ancestors not including *v*)."""
    return ancestors_of_node(adj, v)


def proper_descendants(adj: AdjacencyMatrix, v: NodeId) -> NodeSet:
    """Return proper descendants of *v* (descendants not including *v*)."""
    return descendants_of_node(adj, v)
