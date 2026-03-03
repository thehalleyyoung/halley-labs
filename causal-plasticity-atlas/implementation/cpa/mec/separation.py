"""D-separation and m-separation algorithms.

Provides d-separation testing on DAGs via the Bayes-Ball algorithm,
m-separation testing on ancestral graphs / PAGs, d-separation set
search, Markov blanket computation, and enumeration utilities.
"""

from __future__ import annotations

import itertools
from collections import deque
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _validate_dag(adj: NDArray) -> NDArray[np.int_]:
    """Validate and return a binary adjacency matrix."""
    adj = np.asarray(adj)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {adj.shape}")
    return (adj != 0).astype(np.int_)


def _ancestors(adj: NDArray[np.int_], nodes: Set[int]) -> Set[int]:
    """Return ancestors of *nodes* (inclusive) using BFS on transpose."""
    n = adj.shape[0]
    visited: Set[int] = set()
    queue = deque(nodes)
    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        for p in range(n):
            if adj[p, v] and p not in visited:
                queue.append(p)
    return visited


def _parents(adj: NDArray[np.int_], node: int) -> Set[int]:
    return {int(i) for i in range(adj.shape[0]) if adj[i, node]}


def _children(adj: NDArray[np.int_], node: int) -> Set[int]:
    return {int(j) for j in range(adj.shape[0]) if adj[node, j]}


# -------------------------------------------------------------------
# Bayes-Ball Algorithm
# -------------------------------------------------------------------

class BayesBall:
    """Efficient d-separation via the Bayes-Ball algorithm.

    Reference: Shachter (1998) "Bayes-Ball: The Rational Pastime".

    The algorithm determines whether X ⊥ Y | Z in a DAG by simulating
    a ball bouncing through the graph.  A ball can pass through a node
    via "top" (from parent) or "bottom" (from child) visits, subject to
    rules depending on whether the node is in the conditioning set Z.
    """

    def __init__(self, adj: NDArray[np.int_]) -> None:
        self._adj = _validate_dag(adj)
        self._n = adj.shape[0]

    def _reachable(self, source: Set[int], conditioned: Set[int]) -> Set[int]:
        """Return all nodes reachable from *source* given *conditioned*.

        Uses Shachter's Bayes-Ball rules.  A ball visits a node in one
        of two modes:

        **from_parent** (ball traveling downward):
          - If node NOT in Z: pass DOWN to children (from_parent)
          - If node IN Z: pass UP to parents (from_child) — explaining away

        **from_child** (ball traveling upward):
          - If node NOT in Z: pass DOWN to children (from_parent)
            AND pass UP to parents (from_child) — fork propagation
          - If node IN Z: BLOCKED

        Returns
        -------
        set[int]
            Nodes reachable from *source* through active paths.
        """
        adj = self._adj
        z = set(conditioned)

        # (node, mode) where mode is "from_parent" or "from_child"
        queue: deque[Tuple[int, str]] = deque()
        for s in source:
            # Start by visiting s's neighbors in both directions
            for ch in _children(adj, s):
                queue.append((ch, "from_parent"))
            for pa in _parents(adj, s):
                queue.append((pa, "from_child"))

        visited_fp: Set[int] = set()   # visited via from_parent
        visited_fc: Set[int] = set()   # visited via from_child
        reachable: Set[int] = set(source)

        while queue:
            node, mode = queue.popleft()

            if mode == "from_parent" and node in visited_fp:
                continue
            if mode == "from_child" and node in visited_fc:
                continue

            if mode == "from_parent":
                visited_fp.add(node)
            else:
                visited_fc.add(node)

            reachable.add(node)

            if mode == "from_parent":
                if node not in z:
                    # Non-collider pass-through: continue down
                    for ch in _children(adj, node):
                        if ch not in visited_fp:
                            queue.append((ch, "from_parent"))
                else:
                    # Conditioned: bounce up (explaining away)
                    for pa in _parents(adj, node):
                        if pa not in visited_fc:
                            queue.append((pa, "from_child"))
            else:  # from_child
                if node not in z:
                    # Fork/chain pass-through: go both directions
                    for ch in _children(adj, node):
                        if ch not in visited_fp:
                            queue.append((ch, "from_parent"))
                    for pa in _parents(adj, node):
                        if pa not in visited_fc:
                            queue.append((pa, "from_child"))
                # If node IN Z: blocked (collider not conditioned = blocked)

        return reachable

    def is_d_separated(
        self, x: Set[int], y: Set[int], z: Set[int]
    ) -> bool:
        """Test d-separation X ⊥ Y | Z."""
        reachable = self._reachable(x, z)
        return len(reachable & y) == 0

    def _schedule_passes(
        self, source: Set[int], conditioned: Set[int]
    ) -> Tuple[Set[int], Set[int]]:
        """Return (visited_top, visited_bottom) sets after full passes."""
        self._reachable(source, conditioned)
        # Re-run to capture the sets
        adj = self._adj
        z = set(conditioned)
        queue: deque[Tuple[int, str]] = deque()
        for s in source:
            queue.append((s, "top"))
            queue.append((s, "bottom"))
        visited_top: Set[int] = set()
        visited_bottom: Set[int] = set()
        while queue:
            node, direction = queue.popleft()
            if direction == "top" and node in visited_top:
                continue
            if direction == "bottom" and node in visited_bottom:
                continue
            if direction == "top":
                visited_top.add(node)
            else:
                visited_bottom.add(node)
            if direction == "bottom" and node not in z:
                for ch in _children(adj, node):
                    if ch not in visited_bottom:
                        queue.append((ch, "bottom"))
                for pa in _parents(adj, node):
                    if pa not in visited_top:
                        queue.append((pa, "top"))
            elif direction == "top":
                if node not in z:
                    for ch in _children(adj, node):
                        if ch not in visited_bottom:
                            queue.append((ch, "bottom"))
                if node in z or self._has_conditioned_descendant(node, z):
                    for pa in _parents(adj, node):
                        if pa not in visited_top:
                            queue.append((pa, "top"))
        return visited_top, visited_bottom


# -------------------------------------------------------------------
# Public API: d_separation
# -------------------------------------------------------------------

def d_separation(
    adj_matrix: NDArray[np.int_],
    x: Set[int],
    y: Set[int],
    z: Set[int],
) -> bool:
    """Test whether *x* and *y* are d-separated by *z* in a DAG.

    Uses the Bayes-Ball algorithm for efficiency (linear in graph size).

    Parameters
    ----------
    adj_matrix : NDArray
        Binary adjacency matrix of the DAG.  ``adj[i, j] != 0`` means i → j.
    x, y : set[int]
        Disjoint sets of nodes whose independence is queried.
    z : set[int]
        Conditioning set.

    Returns
    -------
    bool
        ``True`` if X ⊥ Y | Z in the DAG.
    """
    adj = _validate_dag(adj_matrix)
    n = adj.shape[0]
    x_set = set(x)
    y_set = set(y)
    z_set = set(z)

    # Validate node indices
    all_nodes = set(range(n))
    for s, name in [(x_set, "x"), (y_set, "y"), (z_set, "z")]:
        if not s.issubset(all_nodes):
            raise ValueError(f"Node set {name} contains invalid indices")
    if x_set & y_set:
        raise ValueError("x and y must be disjoint")

    bb = BayesBall(adj)
    return bb.is_d_separated(x_set, y_set, z_set)


# -------------------------------------------------------------------
# Public API: m_separation (for MAGs and PAGs)
# -------------------------------------------------------------------

def m_separation(
    pag: Any,
    x: Set[int],
    y: Set[int],
    z: Set[int],
) -> bool:
    """Test m-separation in an ancestral graph or PAG.

    M-separation generalises d-separation for graphs with bidirected
    edges (representing latent common causes).  We use the augmented
    graph approach: construct the augmented DAG with explicit latent
    variables and run d-separation.

    Parameters
    ----------
    pag : object
        An object with ``n_nodes`` attribute and edge marks accessible
        via ``_marks`` or adjacency arrays.  Accepts PAG objects or
        (adj_directed, adj_bidirected) tuples.
    x, y : set[int]
        Disjoint node sets.
    z : set[int]
        Conditioning set.

    Returns
    -------
    bool
        ``True`` if *x* and *y* are m-separated given *z*.
    """
    x_set = set(x)
    y_set = set(y)
    z_set = set(z)

    if isinstance(pag, tuple) and len(pag) == 2:
        adj_dir, adj_bidir = pag
        adj_dir = np.asarray(adj_dir, dtype=np.int_)
        adj_bidir = np.asarray(adj_bidir, dtype=np.int_)
        n = adj_dir.shape[0]
    else:
        # PAG object — extract directed and bidirected edges
        n = pag.n_nodes
        adj_dir = np.zeros((n, n), dtype=np.int_)
        adj_bidir = np.zeros((n, n), dtype=np.int_)
        try:
            marks = pag._marks
            for i in range(n):
                for j in range(n):
                    mi = marks[i, j]
                    mj = marks[j, i]
                    mi_val = mi.value if hasattr(mi, 'value') else int(mi)
                    mj_val = mj.value if hasattr(mj, 'value') else int(mj)
                    # Directed: tail at i, arrowhead at j => i -> j
                    if mi_val == 1 and mj_val == 2:  # TAIL=1, ARROWHEAD=2
                        adj_dir[i, j] = 1
                    # Bidirected: arrowhead at both ends
                    elif mi_val == 2 and mj_val == 2 and i < j:
                        adj_bidir[i, j] = 1
                        adj_bidir[j, i] = 1
        except AttributeError:
            raise TypeError("pag must be a PAG object or (adj_dir, adj_bidir) tuple")

    # Build augmented DAG: add latent node for each bidirected edge
    latent_count = 0
    latent_pairs: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_bidir[i, j]:
                latent_pairs.append((i, j))
                latent_count += 1

    aug_n = n + latent_count
    aug_adj = np.zeros((aug_n, aug_n), dtype=np.int_)

    # Copy directed edges
    aug_adj[:n, :n] = (adj_dir != 0).astype(np.int_)

    # Add latent nodes
    for k, (i, j) in enumerate(latent_pairs):
        lat_idx = n + k
        aug_adj[lat_idx, i] = 1
        aug_adj[lat_idx, j] = 1

    return d_separation(aug_adj, x_set, y_set, z_set)


# -------------------------------------------------------------------
# Find d-separating sets
# -------------------------------------------------------------------

def find_dsep_set(
    adj_matrix: NDArray[np.int_],
    x: int,
    y: int,
    max_size: int | None = None,
) -> Optional[Set[int]]:
    """Find a minimal d-separating set between *x* and *y*.

    Strategy: start with the Markov boundary of x (excluding y), then
    try to reduce.  If that fails, search through subsets of increasing
    size.

    Returns ``None`` if no separating set of size ≤ *max_size* exists.
    """
    adj = _validate_dag(adj_matrix)
    n = adj.shape[0]
    if x < 0 or x >= n or y < 0 or y >= n:
        raise ValueError(f"Invalid node indices: x={x}, y={y}, n={n}")
    if x == y:
        raise ValueError("x and y must be different nodes")

    if max_size is None:
        max_size = n - 2  # exclude x, y

    candidates = set(range(n)) - {x, y}

    # First try: Markov boundary of x minus y
    mb = markov_boundary(adj, x) - {y}
    if len(mb) <= max_size:
        if d_separation(adj, {x}, {y}, mb):
            # Try to minimise
            return _minimise_dsep(adj, x, y, mb)

    # Second try: parents of x and y union (common approach)
    pa_x = _parents(adj, x)
    pa_y = _parents(adj, y)
    anc_xy = _ancestors(adj, {x, y}) - {x, y}
    if len(anc_xy) <= max_size and d_separation(adj, {x}, {y}, anc_xy):
        return _minimise_dsep(adj, x, y, anc_xy)

    # Enumerate subsets of increasing size
    for size in range(max_size + 1):
        for subset in itertools.combinations(sorted(candidates), size):
            z = set(subset)
            if d_separation(adj, {x}, {y}, z):
                return z

    return None


def _minimise_dsep(
    adj: NDArray[np.int_], x: int, y: int, z: Set[int]
) -> Set[int]:
    """Greedily remove nodes from *z* while maintaining d-separation."""
    z = set(z)
    for node in sorted(z):
        reduced = z - {node}
        if d_separation(adj, {x}, {y}, reduced):
            z = reduced
    return z


def find_all_d_separations(
    adj_matrix: NDArray[np.int_],
    x: int,
    y: int,
    max_size: int | None = None,
) -> List[FrozenSet[int]]:
    """Find all d-separating sets between *x* and *y* up to *max_size*.

    Parameters
    ----------
    adj_matrix : NDArray
        DAG adjacency matrix.
    x, y : int
        Node indices.
    max_size : int or None
        Maximum conditioning set size.

    Returns
    -------
    list[frozenset[int]]
        All d-separating sets found.
    """
    adj = _validate_dag(adj_matrix)
    n = adj.shape[0]
    if x < 0 or x >= n or y < 0 or y >= n:
        raise ValueError(f"Invalid node indices: x={x}, y={y}, n={n}")
    if x == y:
        raise ValueError("x and y must be different nodes")

    if max_size is None:
        max_size = n - 2

    candidates = sorted(set(range(n)) - {x, y})
    results: List[FrozenSet[int]] = []

    for size in range(min(max_size, len(candidates)) + 1):
        for subset in itertools.combinations(candidates, size):
            z = set(subset)
            if d_separation(adj, {x}, {y}, z):
                results.append(frozenset(z))

    return results


def minimal_d_sep_set(
    adj_matrix: NDArray[np.int_],
    x: int,
    y: int,
) -> Optional[Set[int]]:
    """Find a minimal-cardinality d-separating set between *x* and *y*.

    Returns ``None`` if *x* and *y* are adjacent (no separator exists)
    or if they are the same node.
    """
    adj = _validate_dag(adj_matrix)
    n = adj.shape[0]
    if x < 0 or x >= n or y < 0 or y >= n:
        raise ValueError(f"Invalid node indices: x={x}, y={y}, n={n}")
    if x == y:
        return None
    # If x and y are adjacent, no d-sep set exists in general
    # (though the empty set might work in some cases)
    candidates = sorted(set(range(n)) - {x, y})
    for size in range(len(candidates) + 1):
        for subset in itertools.combinations(candidates, size):
            z = set(subset)
            if d_separation(adj, {x}, {y}, z):
                return z
    return None


# -------------------------------------------------------------------
# Markov blanket / boundary
# -------------------------------------------------------------------

def markov_blanket(
    adj_matrix: NDArray[np.int_],
    node: int,
) -> Set[int]:
    """Return the Markov blanket of *node* in the DAG.

    The Markov blanket consists of parents, children, and parents of
    children (co-parents / spouses).

    Parameters
    ----------
    adj_matrix : NDArray
        DAG adjacency matrix.
    node : int
        Target node.

    Returns
    -------
    set[int]
        Markov blanket of *node*.
    """
    adj = _validate_dag(adj_matrix)
    n = adj.shape[0]
    if node < 0 or node >= n:
        raise ValueError(f"Invalid node index: {node}, n={n}")

    parents = _parents(adj, node)
    children = _children(adj, node)

    # Co-parents: parents of children
    co_parents: Set[int] = set()
    for ch in children:
        co_parents |= _parents(adj, ch)

    blanket = parents | children | co_parents
    blanket.discard(node)
    return blanket


def markov_boundary(
    adj_matrix: NDArray[np.int_],
    target: int,
) -> Set[int]:
    """Compute the Markov boundary of *target*.

    For DAGs the Markov boundary equals the Markov blanket.
    """
    return markov_blanket(adj_matrix, target)


# -------------------------------------------------------------------
# Enumerate d-sep sets (with size limit)
# -------------------------------------------------------------------

def list_dsep_sets(
    adj_matrix: NDArray[np.int_],
    x: int,
    y: int,
    max_size: int = 4,
) -> List[FrozenSet[int]]:
    """Enumerate d-separating sets between *x* and *y* up to *max_size*.

    This is a convenience wrapper around :func:`find_all_d_separations`.
    """
    return find_all_d_separations(adj_matrix, x, y, max_size=max_size)
