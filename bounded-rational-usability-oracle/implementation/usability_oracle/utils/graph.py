"""
usability_oracle.utils.graph — Graph algorithm utilities.

Thin wrappers around networkx that provide a consistent API for
shortest paths, connectivity, centrality, clustering, and graph
partitioning used throughout the usability oracle.
"""

from __future__ import annotations

from typing import Any, Hashable, Optional, Sequence

import numpy as np

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


# ---------------------------------------------------------------------------
# Path algorithms
# ---------------------------------------------------------------------------

def shortest_path(
    graph: "nx.DiGraph",
    source: Hashable,
    target: Hashable,
    weight: str | None = "weight",
) -> list[Hashable]:
    """Return the shortest path from *source* to *target*.

    Uses Dijkstra's algorithm when *weight* is specified, BFS otherwise.
    Returns an empty list if no path exists.
    """
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    try:
        return list(nx.shortest_path(graph, source, target, weight=weight))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def all_paths(
    graph: "nx.DiGraph",
    source: Hashable,
    target: Hashable,
    max_depth: int | None = None,
) -> list[list[Hashable]]:
    """Return all simple paths from *source* to *target* up to *max_depth*."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    try:
        cutoff = max_depth if max_depth is not None else None
        return [list(p) for p in nx.all_simple_paths(graph, source, target, cutoff=cutoff)]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

def strongly_connected_components(graph: "nx.DiGraph") -> list[set[Hashable]]:
    """Return the strongly connected components of a directed graph."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    return [set(c) for c in nx.strongly_connected_components(graph)]


def topological_sort(graph: "nx.DiGraph") -> list[Hashable]:
    """Return a topological ordering.  Raises if the graph has cycles."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    return list(nx.topological_sort(graph))


# ---------------------------------------------------------------------------
# Diameter
# ---------------------------------------------------------------------------

def diameter(graph: "nx.DiGraph") -> int:
    """Return the diameter of the (weakly connected) graph.

    Computes eccentricity on the undirected version.  Returns 0 for
    disconnected or trivial graphs.
    """
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    ug = graph.to_undirected()
    if not nx.is_connected(ug):
        # Compute for largest component
        largest_cc = max(nx.connected_components(ug), key=len)
        ug = ug.subgraph(largest_cc).copy()
    if ug.number_of_nodes() <= 1:
        return 0
    return nx.diameter(ug)


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------

def betweenness_centrality(
    graph: "nx.DiGraph",
    normalized: bool = True,
    weight: str | None = "weight",
) -> dict[Hashable, float]:
    """Return betweenness centrality for every node."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    return dict(nx.betweenness_centrality(graph, normalized=normalized, weight=weight))


# ---------------------------------------------------------------------------
# Spectral clustering
# ---------------------------------------------------------------------------

def spectral_clustering(
    adjacency: np.ndarray,
    k: int = 2,
) -> np.ndarray:
    """Partition nodes into *k* clusters using spectral clustering.

    Parameters:
        adjacency: Square adjacency/weight matrix.
        k: Number of clusters.

    Returns:
        Integer label array of length ``n``.
    """
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    if n == 0 or k <= 0:
        return np.array([], dtype=int)
    k = min(k, n)

    # Degree matrix and normalised Laplacian
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Compute D^{-1/2}
    d_inv_sqrt = np.zeros(n)
    diag = np.diag(D)
    nonzero = diag > 1e-12
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(diag[nonzero])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    # Take the k smallest eigenvectors
    idx = np.argsort(eigenvalues)[:k]
    V = eigenvectors[:, idx]

    # Row-normalise
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    V = V / norms

    # Simple k-means (manual implementation to avoid sklearn dependency)
    labels = _kmeans(V, k)
    return labels


def _kmeans(X: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    """Minimal k-means implementation for spectral clustering."""
    n = X.shape[0]
    rng = np.random.RandomState(42)
    # Initialise centroids randomly
    indices = rng.choice(n, size=min(k, n), replace=False)
    centroids = X[indices].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # Assign
        dists = np.zeros((n, k))
        for c in range(k):
            dists[:, c] = np.linalg.norm(X - centroids[c], axis=1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centroids
        for c in range(k):
            members = X[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)
    return labels


# ---------------------------------------------------------------------------
# Graph partitioning
# ---------------------------------------------------------------------------

def partition_graph(
    graph: "nx.DiGraph",
    partition: dict[Hashable, int],
) -> "nx.DiGraph":
    """Create a quotient (contracted) graph from a node partition.

    Parameters:
        graph: Original directed graph.
        partition: Mapping from node → cluster label.

    Returns:
        A new DiGraph where each node is a cluster label and edges
        represent inter-cluster connections with aggregated weights.
    """
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    Q = nx.DiGraph()
    cluster_ids = set(partition.values())
    for cid in cluster_ids:
        members = [n for n, c in partition.items() if c == cid]
        Q.add_node(cid, members=members, size=len(members))

    for u, v, data in graph.edges(data=True):
        cu = partition.get(u)
        cv = partition.get(v)
        if cu is None or cv is None:
            continue
        if cu == cv:
            continue
        w = data.get("weight", 1.0)
        if Q.has_edge(cu, cv):
            Q[cu][cv]["weight"] += w
        else:
            Q.add_edge(cu, cv, weight=w)
    return Q


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

def reachable_from(
    graph: "nx.DiGraph",
    source: Hashable,
) -> set[Hashable]:
    """Return all nodes reachable from *source* via directed edges."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    try:
        return set(nx.descendants(graph, source)) | {source}
    except nx.NodeNotFound:
        return set()


def reachability_matrix(graph: "nx.DiGraph") -> np.ndarray:
    """Compute the binary reachability matrix R[i,j] = 1 iff j is reachable from i."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    nodes = list(graph.nodes())
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}
    R = np.zeros((n, n), dtype=int)

    for src in nodes:
        descendants = nx.descendants(graph, src)
        i = node_idx[src]
        R[i, i] = 1
        for d in descendants:
            R[i, node_idx[d]] = 1
    return R


# ---------------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------------

def graph_density(graph: "nx.DiGraph") -> float:
    """Compute graph density: |E| / (|V| * (|V|-1))."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    return float(nx.density(graph))


def average_path_length(graph: "nx.DiGraph") -> float:
    """Average shortest path length across all reachable pairs."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    ug = graph.to_undirected()
    if not nx.is_connected(ug):
        largest_cc = max(nx.connected_components(ug), key=len)
        ug = ug.subgraph(largest_cc).copy()
    if ug.number_of_nodes() <= 1:
        return 0.0
    return float(nx.average_shortest_path_length(ug))


def clustering_coefficient(graph: "nx.DiGraph") -> float:
    """Average clustering coefficient of the graph."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    return float(nx.average_clustering(graph))


def degree_distribution(graph: "nx.DiGraph") -> dict[str, np.ndarray]:
    """Compute in-degree and out-degree distributions."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    in_degrees = np.array([d for _, d in graph.in_degree()])
    out_degrees = np.array([d for _, d in graph.out_degree()])
    return {
        "in_degree": in_degrees,
        "out_degree": out_degrees,
        "in_mean": float(np.mean(in_degrees)) if len(in_degrees) > 0 else 0.0,
        "out_mean": float(np.mean(out_degrees)) if len(out_degrees) > 0 else 0.0,
        "in_max": int(np.max(in_degrees)) if len(in_degrees) > 0 else 0,
        "out_max": int(np.max(out_degrees)) if len(out_degrees) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# PageRank
# ---------------------------------------------------------------------------

def pagerank(
    graph: "nx.DiGraph",
    alpha: float = 0.85,
) -> dict[Hashable, float]:
    """Compute PageRank centrality for each node."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    return dict(nx.pagerank(graph, alpha=alpha))


# ---------------------------------------------------------------------------
# Closeness centrality
# ---------------------------------------------------------------------------

def closeness_centrality(graph: "nx.DiGraph") -> dict[Hashable, float]:
    """Compute closeness centrality for each node."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    return dict(nx.closeness_centrality(graph))


# ---------------------------------------------------------------------------
# Graph condensation (DAG of SCCs)
# ---------------------------------------------------------------------------

def condensation(graph: "nx.DiGraph") -> "nx.DiGraph":
    """Compute the condensation (DAG of SCCs) of a directed graph."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    return nx.condensation(graph)


# ---------------------------------------------------------------------------
# Minimum spanning tree (undirected)
# ---------------------------------------------------------------------------

def minimum_spanning_tree_weight(graph: "nx.DiGraph") -> float:
    """Compute the total weight of the minimum spanning tree.

    Operates on the undirected version of the graph.
    """
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    ug = graph.to_undirected()
    if not nx.is_connected(ug):
        total = 0.0
        for cc in nx.connected_components(ug):
            sub = ug.subgraph(cc).copy()
            mst = nx.minimum_spanning_tree(sub)
            total += sum(d.get("weight", 1.0) for _, _, d in mst.edges(data=True))
        return total
    mst = nx.minimum_spanning_tree(ug)
    return sum(d.get("weight", 1.0) for _, _, d in mst.edges(data=True))


# ---------------------------------------------------------------------------
# Adjacency matrix conversion
# ---------------------------------------------------------------------------

def to_adjacency_matrix(
    graph: "nx.DiGraph",
    weight: str | None = "weight",
) -> tuple[np.ndarray, list[Hashable]]:
    """Convert graph to adjacency matrix.

    Returns:
        (matrix, node_list) tuple.
    """
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    nodes = list(graph.nodes())
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((n, n))
    for u, v, data in graph.edges(data=True):
        w = data.get(weight, 1.0) if weight else 1.0
        A[node_idx[u], node_idx[v]] = w
    return A, nodes


def from_adjacency_matrix(
    matrix: np.ndarray,
    node_names: list[Hashable] | None = None,
) -> "nx.DiGraph":
    """Create a directed graph from an adjacency matrix."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for graph utilities")
    n = matrix.shape[0]
    names = node_names or list(range(n))
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(names[i])
    for i in range(n):
        for j in range(n):
            if matrix[i, j] != 0:
                G.add_edge(names[i], names[j], weight=float(matrix[i, j]))
    return G
