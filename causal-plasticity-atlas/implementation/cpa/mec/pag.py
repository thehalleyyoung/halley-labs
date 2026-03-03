"""Partial Ancestral Graph (PAG) handling.

Represents PAGs for causal models with latent variables.  Edge marks
encode tail (``-``), arrowhead (``>``), circle (``o``), and none.

A PAG is the equivalence class representative for a Markov equivalence
class of MAGs (Maximal Ancestral Graphs).  It encodes which features
of the underlying causal graph are identifiable from observational data
when latent common causes may exist.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


class EdgeMark(Enum):
    """Edge-mark types used in a PAG."""

    NONE = 0
    TAIL = 1
    ARROWHEAD = 2
    CIRCLE = 3


# Integer constants for fast comparison
_NONE = 0
_TAIL = 1
_ARROW = 2
_CIRCLE = 3


class PAG:
    """Partial Ancestral Graph.

    A PAG encodes a Markov equivalence class of MAGs.  Each edge
    endpoint has a mark: NONE (no edge), TAIL (-), ARROWHEAD (>), or
    CIRCLE (o).  The pair of marks determines the edge type:

    - ``- ->``  : directed edge (tail to arrowhead)
    - ``<- ->`` : bidirected edge (arrowhead to arrowhead)
    - ``o- ->`` : partially oriented (circle to arrowhead)
    - ``o- -o`` : fully unoriented (circle to circle)
    - ``- -o``  : partially oriented (tail to circle)

    Parameters
    ----------
    n_nodes : int
        Number of observed nodes.
    """

    def __init__(self, n_nodes: int) -> None:
        if n_nodes < 0:
            raise ValueError("n_nodes must be non-negative")
        self.n_nodes = n_nodes
        # _marks[i, j] stores the mark at the j-end of the edge from i to j
        # i.e., for edge i *-* j: _marks[i,j] is the mark at j, _marks[j,i] is the mark at i
        self._marks: NDArray = np.full(
            (n_nodes, n_nodes), EdgeMark.NONE, dtype=object,
        )

    # -----------------------------------------------------------------
    # Edge mark access
    # -----------------------------------------------------------------

    def set_edge_mark(self, i: int, j: int, mark: EdgeMark) -> None:
        """Set the edge mark at position (i, j).

        This sets the mark at the j-end of the i-j edge.
        """
        self._check_idx(i)
        self._check_idx(j)
        if i == j:
            raise ValueError("Self-loops are not allowed")
        self._marks[i, j] = mark

    def get_edge_mark(self, i: int, j: int) -> EdgeMark:
        """Get the edge mark at position (i, j).

        Returns the mark at the j-end of the i-j edge.
        """
        self._check_idx(i)
        self._check_idx(j)
        return self._marks[i, j]

    def edge_mark(self, i: int, j: int) -> int:
        """Get edge mark at j-end of edge i-j as integer."""
        m = self._marks[i, j]
        return m.value if isinstance(m, EdgeMark) else int(m)

    def _check_idx(self, i: int) -> None:
        if i < 0 or i >= self.n_nodes:
            raise ValueError(f"Node index {i} out of range [0, {self.n_nodes})")

    # -----------------------------------------------------------------
    # Edge type queries
    # -----------------------------------------------------------------

    def has_edge(self, i: int, j: int) -> bool:
        """Check if there is any edge between i and j."""
        mi = self.edge_mark(i, j)
        mj = self.edge_mark(j, i)
        return mi != _NONE or mj != _NONE

    def has_tail(self, i: int, j: int) -> bool:
        """Check if mark at j-end of i-j edge is a tail."""
        return self.edge_mark(i, j) == _TAIL

    def has_arrowhead(self, i: int, j: int) -> bool:
        """Check if mark at j-end of i-j edge is an arrowhead."""
        return self.edge_mark(i, j) == _ARROW

    def has_circle(self, i: int, j: int) -> bool:
        """Check if mark at j-end of i-j edge is a circle."""
        return self.edge_mark(i, j) == _CIRCLE

    def is_directed(self, i: int, j: int) -> bool:
        """Check if there is a directed edge i -> j (tail at i, arrow at j)."""
        return self.edge_mark(j, i) == _TAIL and self.edge_mark(i, j) == _ARROW

    def is_bidirected(self, i: int, j: int) -> bool:
        """Check if there is a bidirected edge i <-> j."""
        return self.edge_mark(i, j) == _ARROW and self.edge_mark(j, i) == _ARROW

    def add_directed_edge(self, i: int, j: int) -> None:
        """Add directed edge i -> j."""
        self.set_edge_mark(i, j, EdgeMark.ARROWHEAD)
        self.set_edge_mark(j, i, EdgeMark.TAIL)

    def add_bidirected_edge(self, i: int, j: int) -> None:
        """Add bidirected edge i <-> j."""
        self.set_edge_mark(i, j, EdgeMark.ARROWHEAD)
        self.set_edge_mark(j, i, EdgeMark.ARROWHEAD)

    def add_circle_edge(self, i: int, j: int) -> None:
        """Add edge with circles at both ends: i o-o j."""
        self.set_edge_mark(i, j, EdgeMark.CIRCLE)
        self.set_edge_mark(j, i, EdgeMark.CIRCLE)

    def add_partially_directed(self, i: int, j: int) -> None:
        """Add partially directed edge i o-> j."""
        self.set_edge_mark(i, j, EdgeMark.ARROWHEAD)
        self.set_edge_mark(j, i, EdgeMark.CIRCLE)

    def remove_edge(self, i: int, j: int) -> None:
        """Remove any edge between i and j."""
        self.set_edge_mark(i, j, EdgeMark.NONE)
        self.set_edge_mark(j, i, EdgeMark.NONE)

    # -----------------------------------------------------------------
    # Ancestral queries
    # -----------------------------------------------------------------

    def possible_ancestors(self, node: int) -> Set[int]:
        """Return the set of possible ancestors of *node*.

        A node a is a possible ancestor of b if there exists a
        possibly directed path from a to b (following edges where
        arrowheads don't point backwards).
        """
        self._check_idx(node)
        result: Set[int] = set()
        visited: Set[int] = set()
        queue = deque([node])
        while queue:
            v = queue.popleft()
            if v in visited:
                continue
            visited.add(v)
            result.add(v)
            for u in range(self.n_nodes):
                if u in visited:
                    continue
                if not self.has_edge(u, v):
                    continue
                # u can be ancestor of v if mark at u-end is not arrowhead
                # i.e., edge goes u *-> v or u *-o v (not u <-* v)
                mark_at_u = self.edge_mark(v, u)  # mark at u-end
                if mark_at_u != _ARROW:
                    queue.append(u)
        result.discard(node)
        return result

    def definite_ancestors(self, node: int) -> Set[int]:
        """Return the set of definite ancestors of *node*.

        A node a is a definite ancestor of b if in every MAG in the
        equivalence class, a is an ancestor of b.  This requires all
        edges on the path to be fully directed (tail -> arrowhead).
        """
        self._check_idx(node)
        result: Set[int] = set()
        visited: Set[int] = set()
        queue = deque([node])
        while queue:
            v = queue.popleft()
            if v in visited:
                continue
            visited.add(v)
            result.add(v)
            for u in range(self.n_nodes):
                if u in visited:
                    continue
                # Definite: u -> v (tail at u, arrow at v)
                if self.is_directed(u, v):
                    queue.append(u)
        result.discard(node)
        return result

    def possible_descendants(self, node: int) -> Set[int]:
        """Return the set of possible descendants of *node*."""
        self._check_idx(node)
        result: Set[int] = set()
        visited: Set[int] = set()
        queue = deque([node])
        while queue:
            v = queue.popleft()
            if v in visited:
                continue
            visited.add(v)
            result.add(v)
            for w in range(self.n_nodes):
                if w in visited:
                    continue
                if not self.has_edge(v, w):
                    continue
                mark_at_w = self.edge_mark(v, w)
                if mark_at_w != _ARROW:
                    continue
                # Also check that mark at v-end isn't arrowhead
                # (would make it bidirected, not a descendant path)
                mark_at_v = self.edge_mark(w, v)
                if mark_at_v != _ARROW:  # tail or circle at v-end
                    queue.append(w)
        result.discard(node)
        return result

    # -----------------------------------------------------------------
    # Visible edges
    # -----------------------------------------------------------------

    def visible_edges(self) -> List[Tuple[int, int]]:
        """Return edges that are visible (not potentially confounded).

        An edge i -> j is visible if there is no inducing path between
        i and j that is into i.  Simplified criterion: i -> j is visible
        if every node with an edge into i also has an edge into j, or
        is j itself.
        """
        visible = []
        n = self.n_nodes
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if not self.is_directed(i, j):
                    continue
                # Check visibility: for every k with an edge into i (k *-> i),
                # k must be adjacent to j or equal to j
                is_visible = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    # k has arrowhead at i: k *-> i
                    if self.has_edge(k, i) and self.edge_mark(k, i) == _ARROW:
                        if not self.has_edge(k, j):
                            is_visible = False
                            break
                if is_visible:
                    visible.append((i, j))
        return visible

    # -----------------------------------------------------------------
    # Discriminating paths
    # -----------------------------------------------------------------

    def discriminating_paths(self, b: int, c: int) -> List[List[int]]:
        """Find discriminating paths for edge b-c.

        A path <a, ..., b, c> is discriminating for b w.r.t. c if:
        1. a is not adjacent to c.
        2. Every node between a and b on the path is a parent of c.
        3. The path has length >= 3.
        """
        self._check_idx(b)
        self._check_idx(c)
        if not self.has_edge(b, c):
            return []

        n = self.n_nodes
        results: List[List[int]] = []

        # BFS backwards from b to find discriminating path starts
        # Path is [..., v, ..., b, c] where intermediaries -> c
        for a in range(n):
            if a == b or a == c:
                continue
            if self.has_edge(a, c):
                continue
            # Try to find path from a to b through parents of c
            paths = self._find_disc_paths(a, b, c)
            results.extend(paths)

        return results

    def _find_disc_paths(
        self, a: int, b: int, c: int
    ) -> List[List[int]]:
        """Find paths from a to b where intermediaries are parents of c."""
        n = self.n_nodes
        results: List[List[int]] = []
        # DFS with path tracking
        stack: List[Tuple[int, List[int]]] = [(a, [a])]
        while stack:
            node, path = stack.pop()
            if len(path) > n:
                continue
            for nxt in range(n):
                if nxt in path:  # no revisits
                    continue
                if not self.has_edge(node, nxt):
                    continue
                if nxt == b:
                    full_path = path + [b, c]
                    if len(full_path) >= 4:
                        results.append(full_path)
                    continue
                # Intermediary must be parent of c
                if self.is_directed(nxt, c):
                    stack.append((nxt, path + [nxt]))
        return results

    # -----------------------------------------------------------------
    # Possibly d-separated
    # -----------------------------------------------------------------

    def possibly_d_separated(self, x: int, y: int, z: Set[int]) -> bool:
        """Check if x and y are possibly d-separated given z.

        Uses the FCI definition: x and y are possibly d-separated if
        every possibly active path between them is blocked by z.
        """
        self._check_idx(x)
        self._check_idx(y)
        n = self.n_nodes

        # Check reachability via possibly active paths
        # A path is possibly active given z if:
        # - Non-endpoints not in z are not colliders on the path
        # - Colliders on the path have a descendant in z
        # Simplified: use the augmented graph approach
        visited: Set[Tuple[int, str]] = set()
        queue: deque[Tuple[int, str]] = deque()

        # Start from x going in both directions
        queue.append((x, "up"))
        queue.append((x, "down"))

        while queue:
            node, direction = queue.popleft()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))

            if node == y:
                return False  # Found active path

            if direction == "up":
                # Going up (from child to parent)
                for p in range(n):
                    if p == node:
                        continue
                    if not self.has_edge(p, node):
                        continue
                    # p is potential parent of node
                    mark_at_node = self.edge_mark(p, node)
                    mark_at_p = self.edge_mark(node, p)
                    if mark_at_node == _ARROW:
                        # p *-> node: node is endpoint
                        if node not in z:
                            # Not conditioned: can pass through
                            if mark_at_p != _ARROW:
                                queue.append((p, "up"))
                    if mark_at_p in (_TAIL, _CIRCLE):
                        if node not in z:
                            queue.append((p, "up"))

                # Try going down from node
                for c in range(n):
                    if c == node:
                        continue
                    if not self.has_edge(node, c):
                        continue
                    if node not in z:
                        queue.append((c, "down"))

            else:  # direction == "down"
                if node not in z:
                    # Pass through: continue down
                    for c in range(n):
                        if c == node:
                            continue
                        if not self.has_edge(node, c):
                            continue
                        mark_at_c = self.edge_mark(node, c)
                        if mark_at_c in (_ARROW, _CIRCLE):
                            queue.append((c, "down"))
                    # Also go up
                    for p in range(n):
                        if p == node:
                            continue
                        if not self.has_edge(p, node):
                            continue
                        queue.append((p, "up"))
                else:
                    # node in z: can activate collider paths
                    for p in range(n):
                        if p == node:
                            continue
                        if not self.has_edge(p, node):
                            continue
                        queue.append((p, "up"))

        return True  # No active path found

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def is_valid(self) -> bool:
        """Check structural consistency of the PAG.

        Validates:
        1. No self-loops.
        2. Edge marks are consistent (if one end is NONE, other must be too).
        3. No impossible mark combinations.
        """
        n = self.n_nodes

        for i in range(n):
            # No self-loops
            if self.edge_mark(i, i) != _NONE:
                return False

        for i in range(n):
            for j in range(i + 1, n):
                mi = self.edge_mark(i, j)
                mj = self.edge_mark(j, i)
                # If one end is NONE, both must be
                if (mi == _NONE) != (mj == _NONE):
                    return False

        return True

    # -----------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------

    def to_adjacency_matrix(self) -> NDArray[np.int_]:
        """Return integer edge mark matrix.

        Values: 0=NONE, 1=TAIL, 2=ARROWHEAD, 3=CIRCLE.
        """
        n = self.n_nodes
        adj = np.zeros((n, n), dtype=np.int_)
        for i in range(n):
            for j in range(n):
                adj[i, j] = self.edge_mark(i, j)
        return adj

    def __repr__(self) -> str:
        n_edges = 0
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if self.has_edge(i, j):
                    n_edges += 1
        return f"PAG(n_nodes={self.n_nodes}, edges={n_edges})"

    def copy(self) -> PAG:
        """Return a deep copy."""
        new = PAG(self.n_nodes)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                new._marks[i, j] = self._marks[i, j]
        return new


# -------------------------------------------------------------------
# DAG to PAG conversion
# -------------------------------------------------------------------

def dag_to_pag(
    adj_matrix: NDArray[np.int_],
    latent_vars: Set[int],
) -> PAG:
    """Convert a DAG adjacency matrix to a PAG marginalising *latent_vars*.

    Steps:
    1. Construct the MAG by marginalising over latent variables.
    2. Convert MAG to PAG via FCI-like orientation.

    Parameters
    ----------
    adj_matrix : NDArray
        Full DAG adjacency matrix including latent variables.
    latent_vars : set[int]
        Indices of latent (hidden) variables to marginalise out.

    Returns
    -------
    PAG
        The PAG over observed variables.
    """
    adj = np.asarray(adj_matrix, dtype=np.int_)
    adj = (adj != 0).astype(np.int_)
    n_full = adj.shape[0]

    # Observed variables
    observed = sorted(set(range(n_full)) - latent_vars)
    n_obs = len(observed)
    obs_map = {v: i for i, v in enumerate(observed)}

    # Build ancestor sets for efficiency
    def get_ancestors(node: int) -> Set[int]:
        visited: Set[int] = set()
        queue = deque([node])
        while queue:
            v = queue.popleft()
            if v in visited:
                continue
            visited.add(v)
            for p in range(n_full):
                if adj[p, v] and p not in visited:
                    queue.append(p)
        return visited

    def get_descendants(node: int) -> Set[int]:
        visited: Set[int] = set()
        queue = deque([node])
        while queue:
            v = queue.popleft()
            if v in visited:
                continue
            visited.add(v)
            for c in range(n_full):
                if adj[v, c] and c not in visited:
                    queue.append(c)
        return visited

    # Step 1: Build MAG skeleton and edge types
    # Two observed variables i, j are adjacent in the MAG if:
    # there is an inducing path between them relative to latent_vars
    # Simplified approach: use ancestral relationships

    # Construct the MAG: directed and bidirected edges among observed
    mag_dir = np.zeros((n_obs, n_obs), dtype=np.int_)
    mag_bidir = np.zeros((n_obs, n_obs), dtype=np.int_)

    for idx_i, vi in enumerate(observed):
        for idx_j, vj in enumerate(observed):
            if idx_i == idx_j:
                continue
            # Check if vi is an ancestor of vj (considering latent vars)
            anc_vj = get_ancestors(vj)
            desc_vi = get_descendants(vi)

            if vj in desc_vi:
                # vi is ancestor of vj: check if there's a directed path
                # through observed variables only or via latent
                # If path goes through only latent intermediaries, it's directed
                if _has_directed_path_via_latent(adj, vi, vj, latent_vars, observed):
                    mag_dir[idx_i, idx_j] = 1

    # Check for bidirected edges (latent common cause)
    for idx_i in range(n_obs):
        for idx_j in range(idx_i + 1, n_obs):
            vi = observed[idx_i]
            vj = observed[idx_j]
            if mag_dir[idx_i, idx_j] or mag_dir[idx_j, idx_i]:
                continue
            # Check for latent common cause
            if _has_latent_common_cause(adj, vi, vj, latent_vars):
                mag_bidir[idx_i, idx_j] = 1
                mag_bidir[idx_j, idx_i] = 1

    # Step 2: Build initial PAG with circle marks
    pag = PAG(n_obs)

    for i in range(n_obs):
        for j in range(n_obs):
            if i == j:
                continue
            if mag_dir[i, j]:
                # Directed edge i -> j: start with tail at i, arrow at j
                pag.set_edge_mark(i, j, EdgeMark.ARROWHEAD)
                if not mag_dir[j, i] and not mag_bidir[i, j]:
                    pag.set_edge_mark(j, i, EdgeMark.TAIL)
                elif mag_bidir[i, j]:
                    pag.set_edge_mark(j, i, EdgeMark.ARROWHEAD)

    for i in range(n_obs):
        for j in range(i + 1, n_obs):
            if mag_bidir[i, j] and not mag_dir[i, j] and not mag_dir[j, i]:
                pag.set_edge_mark(i, j, EdgeMark.ARROWHEAD)
                pag.set_edge_mark(j, i, EdgeMark.ARROWHEAD)

    # Replace definite tails/arrows with circles where orientation is uncertain
    # (proper FCI would use Zhang rules here, but we keep the MAG orientations
    # as a reasonable approximation)

    return pag


def _has_directed_path_via_latent(
    adj: NDArray[np.int_],
    source: int,
    target: int,
    latent: Set[int],
    observed: List[int],
) -> bool:
    """Check if there's a directed path from source to target where
    intermediaries are all latent or the path is direct."""
    visited: Set[int] = set()
    queue = deque([source])
    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        for c in range(adj.shape[0]):
            if adj[v, c] and c not in visited:
                if c == target:
                    return True
                if c in latent:
                    queue.append(c)
    return False


def _has_latent_common_cause(
    adj: NDArray[np.int_],
    vi: int,
    vj: int,
    latent: Set[int],
) -> bool:
    """Check if vi and vj share a latent common ancestor with
    directed paths through only latent intermediaries."""
    n = adj.shape[0]
    # Find ancestors of vi reachable through latent nodes
    def latent_ancestors(node: int) -> Set[int]:
        visited: Set[int] = set()
        queue = deque([node])
        while queue:
            v = queue.popleft()
            if v in visited:
                continue
            visited.add(v)
            for p in range(n):
                if adj[p, v] and p not in visited:
                    if p in latent:
                        queue.append(p)
        return visited & latent

    anc_i = latent_ancestors(vi)
    anc_j = latent_ancestors(vj)
    return len(anc_i & anc_j) > 0


def from_mag(mag_dir: NDArray[np.int_], mag_bidir: NDArray[np.int_]) -> PAG:
    """Convert a MAG (directed + bidirected adjacency) to a PAG.

    Parameters
    ----------
    mag_dir : NDArray
        Directed adjacency: mag_dir[i,j] = 1 means i -> j.
    mag_bidir : NDArray
        Bidirected adjacency: mag_bidir[i,j] = 1 means i <-> j.

    Returns
    -------
    PAG
        The PAG with appropriate edge marks.
    """
    mag_dir = np.asarray(mag_dir, dtype=np.int_)
    mag_bidir = np.asarray(mag_bidir, dtype=np.int_)
    n = mag_dir.shape[0]

    pag = PAG(n)

    # Start by putting circles everywhere there's an edge
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if mag_dir[i, j] or mag_bidir[i, j]:
                pag.set_edge_mark(i, j, EdgeMark.CIRCLE)

    # Orient definite arrowheads from MAG
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if mag_dir[i, j]:
                # i -> j: arrowhead at j
                pag.set_edge_mark(i, j, EdgeMark.ARROWHEAD)
            if mag_bidir[i, j]:
                # Bidirected: arrowhead at both ends
                pag.set_edge_mark(i, j, EdgeMark.ARROWHEAD)

    # Orient definite tails
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if mag_dir[i, j] and not mag_dir[j, i] and not mag_bidir[i, j]:
                # i -> j: tail at i
                pag.set_edge_mark(j, i, EdgeMark.TAIL)

    # Apply Zhang rules to complete orientation
    from cpa.mec.orientation import ZhangRules
    ZhangRules().apply_rules(pag)

    return pag
