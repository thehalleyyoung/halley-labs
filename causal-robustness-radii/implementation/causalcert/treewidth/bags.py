"""
Bag-level state operations for tree-decomposition dynamic programming.

Provides the :class:`BagState` dataclass for compact representation of
edge-assignment states within a tree bag, along with enumeration, merging,
and consistency-checking helpers used by :mod:`causalcert.treewidth.dp`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from causalcert.types import AdjacencyMatrix, NodeId, NodeSet

# Edge states for an unordered pair {u, v} with u < v
NO_EDGE: int = 0
FORWARD: int = 1    # u → v
BACKWARD: int = 2   # v → u

_EDGE_STATES = (NO_EDGE, FORWARD, BACKWARD)


# ---------------------------------------------------------------------------
# Compact bitmask representation
# ---------------------------------------------------------------------------


def _pair_index(u: NodeId, v: NodeId, n_vertices: int) -> int:
    """Map an ordered pair (u, v) with u < v to a linear index.

    For *k* vertices in a bag there are ``k*(k-1)/2`` unordered pairs.
    We enumerate them in lexicographic order of ``(min, max)``.

    Parameters
    ----------
    u, v : NodeId
        Vertex indices with ``u < v``.
    n_vertices : int
        Total number of vertices in the bag (used for indexing).

    Returns
    -------
    int
        Linear index of the pair.
    """
    # Position of pair (u, v) in the upper-triangular enumeration
    # Sum_{r=0}^{u-1} (n_vertices - 1 - r) + (v - u - 1)
    return u * n_vertices - u * (u + 1) // 2 + (v - u - 1)


def _n_pairs(k: int) -> int:
    """Number of unordered pairs for *k* vertices."""
    return k * (k - 1) // 2


# ---------------------------------------------------------------------------
# BagState
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BagState:
    """Compact representation of an edge-assignment state within a bag.

    Each unordered pair ``{u, v}`` (with ``u < v``) among the bag's vertices
    is assigned one of three states: ``NO_EDGE`` (0), ``FORWARD`` (1, meaning
    ``u → v``), or ``BACKWARD`` (2, meaning ``v → u``).  The full assignment
    is packed into a tuple of ternary digits.

    Attributes
    ----------
    vertices : tuple[NodeId, ...]
        Sorted vertices of the bag.
    assignment : tuple[int, ...]
        Ternary assignment for each unordered pair, in lexicographic pair
        order.  Length is ``|vertices| * (|vertices| - 1) / 2``.
    cost : int
        Accumulated edit cost for this partial state.
    """

    vertices: tuple[NodeId, ...]
    assignment: tuple[int, ...]
    cost: int = 0

    @property
    def n_pairs(self) -> int:
        """Number of vertex pairs in the bag."""
        return _n_pairs(len(self.vertices))

    def edge_state(self, u: NodeId, v: NodeId) -> int:
        """Return the edge state between *u* and *v*.

        Parameters
        ----------
        u, v : NodeId
            Two distinct vertices in the bag.

        Returns
        -------
        int
            ``NO_EDGE``, ``FORWARD``, or ``BACKWARD``.
        """
        a, b = (u, v) if u < v else (v, u)
        idx = _pair_index(
            self.vertices.index(a),
            self.vertices.index(b),
            len(self.vertices),
        )
        return self.assignment[idx]

    def to_bitmask(self) -> int:
        """Encode the assignment as a single integer (base-3 packing).

        Returns
        -------
        int
            Packed bitmask.
        """
        mask = 0
        for i, s in enumerate(self.assignment):
            mask += s * (3 ** i)
        return mask

    @classmethod
    def from_bitmask(
        cls,
        mask: int,
        vertices: tuple[NodeId, ...],
        cost: int = 0,
    ) -> BagState:
        """Reconstruct a :class:`BagState` from a packed bitmask.

        Parameters
        ----------
        mask : int
            Packed bitmask (base-3).
        vertices : tuple[NodeId, ...]
            Sorted bag vertices.
        cost : int
            Edit cost.

        Returns
        -------
        BagState
        """
        np_ = _n_pairs(len(vertices))
        assignment: list[int] = []
        rem = mask
        for _ in range(np_):
            assignment.append(rem % 3)
            rem //= 3
        return cls(vertices=vertices, assignment=tuple(assignment), cost=cost)


# ---------------------------------------------------------------------------
# State enumeration
# ---------------------------------------------------------------------------


def enumerate_bag_states(
    vertices: tuple[NodeId, ...],
    source_dag: AdjacencyMatrix | None = None,
    k_max: int | None = None,
) -> list[BagState]:
    """Enumerate all valid edge-assignment states for a bag.

    For *k* vertices there are ``3^{k(k-1)/2}`` possible states.
    If *source_dag* is given, the edit cost relative to the source is
    computed for each state.  States exceeding *k_max* edits are pruned.

    Parameters
    ----------
    vertices : tuple[NodeId, ...]
        Sorted vertices of the bag.
    source_dag : AdjacencyMatrix | None
        Original DAG adjacency matrix for cost computation.
    k_max : int | None
        If set, prune states with cost exceeding this threshold.

    Returns
    -------
    list[BagState]
        All feasible states, ordered by increasing cost.
    """
    k = len(vertices)
    np_ = _n_pairs(k)
    if np_ == 0:
        return [BagState(vertices=vertices, assignment=(), cost=0)]

    states: list[BagState] = []
    for combo in product(_EDGE_STATES, repeat=np_):
        cost = 0
        if source_dag is not None:
            cost = _compute_cost(vertices, combo, source_dag)
        if k_max is not None and cost > k_max:
            continue
        states.append(BagState(vertices=vertices, assignment=combo, cost=cost))

    states.sort(key=lambda s: s.cost)
    return states


def _compute_cost(
    vertices: tuple[NodeId, ...],
    assignment: tuple[int, ...],
    source_dag: AdjacencyMatrix,
) -> int:
    """Compute the edit cost of an assignment relative to the source DAG.

    Parameters
    ----------
    vertices : tuple[NodeId, ...]
        Sorted bag vertices.
    assignment : tuple[int, ...]
        Ternary assignment for each pair.
    source_dag : AdjacencyMatrix
        Original DAG adjacency matrix.

    Returns
    -------
    int
        Number of edge disagreements with the source DAG.
    """
    k = len(vertices)
    cost = 0
    idx = 0
    for i in range(k):
        for j in range(i + 1, k):
            u, v = vertices[i], vertices[j]
            state = assignment[idx]
            # Determine source state
            has_fwd = bool(source_dag[u, v])
            has_bwd = bool(source_dag[v, u])
            if has_fwd:
                src_state = FORWARD
            elif has_bwd:
                src_state = BACKWARD
            else:
                src_state = NO_EDGE
            if state != src_state:
                cost += 1
            idx += 1
    return cost


# ---------------------------------------------------------------------------
# State restriction / projection
# ---------------------------------------------------------------------------


def restrict_state(
    state: BagState,
    sub_vertices: tuple[NodeId, ...],
) -> BagState:
    """Restrict a bag state to a subset of its vertices.

    Projects out all pairs involving vertices not in *sub_vertices*.

    Parameters
    ----------
    state : BagState
        Full bag state.
    sub_vertices : tuple[NodeId, ...]
        Sorted subset of ``state.vertices``.

    Returns
    -------
    BagState
        State restricted to *sub_vertices*, cost unchanged.
    """
    k = len(sub_vertices)
    np_ = _n_pairs(k)
    if np_ == 0:
        return BagState(vertices=sub_vertices, assignment=(), cost=state.cost)

    new_assignment: list[int] = []
    for i in range(k):
        for j in range(i + 1, k):
            u, v = sub_vertices[i], sub_vertices[j]
            new_assignment.append(state.edge_state(u, v))

    return BagState(
        vertices=sub_vertices,
        assignment=tuple(new_assignment),
        cost=state.cost,
    )


def extend_state(
    state: BagState,
    new_vertex: NodeId,
    edge_choices: tuple[int, ...],
    extra_cost: int = 0,
) -> BagState:
    """Extend a bag state with a new vertex and its edge assignments.

    Parameters
    ----------
    state : BagState
        Existing state for the current vertices.
    new_vertex : NodeId
        Vertex being introduced.
    edge_choices : tuple[int, ...]
        Edge-state assignment for each pair ``(existing_vertex, new_vertex)``
        where existing_vertex < new_vertex, and ``(new_vertex, existing_vertex)``
        where new_vertex < existing_vertex.  Must be ordered to produce the
        correct lexicographic pair enumeration for the extended vertex set.
    extra_cost : int
        Additional cost incurred by the new edges.

    Returns
    -------
    BagState
        Extended state including the new vertex.
    """
    new_verts = tuple(sorted(set(state.vertices) | {new_vertex}))
    k = len(new_verts)
    np_ = _n_pairs(k)

    # Build full assignment for the new vertex set
    new_assignment: list[int] = [0] * np_
    for i in range(k):
        for j in range(i + 1, k):
            idx = _pair_index(i, j, k)
            u, v = new_verts[i], new_verts[j]
            if u in state.vertices and v in state.vertices:
                new_assignment[idx] = state.edge_state(u, v)
            else:
                # One of them is the new vertex — look up in edge_choices
                old_verts_sorted = state.vertices
                if new_vertex == u:
                    pos_other = old_verts_sorted.index(v) if v in old_verts_sorted else -1
                else:
                    pos_other = old_verts_sorted.index(u) if u in old_verts_sorted else -1
                if pos_other >= 0:
                    new_assignment[idx] = edge_choices[pos_other]
                # else: both new — should not happen in introduce

    return BagState(
        vertices=new_verts,
        assignment=tuple(new_assignment),
        cost=state.cost + extra_cost,
    )


# ---------------------------------------------------------------------------
# State merging for join nodes
# ---------------------------------------------------------------------------


def merge_states(
    left: BagState,
    right: BagState,
) -> BagState | None:
    """Merge two compatible states from the children of a join node.

    Both states must have the same vertices and identical edge assignments
    on the shared pairs.  The merged cost is ``left.cost + right.cost``
    minus the double-counted cost on the shared pairs.

    Parameters
    ----------
    left, right : BagState
        States from left and right child of a join node.

    Returns
    -------
    BagState | None
        Merged state, or ``None`` if the states are incompatible.
    """
    if left.vertices != right.vertices:
        return None
    if left.assignment != right.assignment:
        return None

    # The shared-pair cost is counted once (both have same assignment).
    # Each child's cost includes cost from its sub-tree below.
    return BagState(
        vertices=left.vertices,
        assignment=left.assignment,
        cost=left.cost + right.cost,
    )


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------


def is_acyclic_in_bag(state: BagState) -> bool:
    """Check whether the directed edges in a bag state form a DAG.

    Uses a simple topological sort on the bag vertices.

    Parameters
    ----------
    state : BagState
        Bag state to check.

    Returns
    -------
    bool
        ``True`` if the directed edges in the bag are acyclic.
    """
    verts = state.vertices
    k = len(verts)
    if k <= 1:
        return True

    # Build adjacency lists among bag vertices
    adj: dict[NodeId, set[NodeId]] = {v: set() for v in verts}
    in_deg: dict[NodeId, int] = {v: 0 for v in verts}

    idx = 0
    for i in range(k):
        for j in range(i + 1, k):
            u, v = verts[i], verts[j]
            s = state.assignment[idx]
            if s == FORWARD:
                adj[u].add(v)
                in_deg[v] += 1
            elif s == BACKWARD:
                adj[v].add(u)
                in_deg[u] += 1
            idx += 1

    # Kahn's algorithm
    queue = [v for v in verts if in_deg[v] == 0]
    count = 0
    while queue:
        node = queue.pop()
        count += 1
        for nb in adj[node]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)

    return count == k


def check_ci_constraint(
    state: BagState,
    x: NodeId,
    y: NodeId,
    conditioning: NodeSet,
    must_be_independent: bool,
) -> bool:
    """Check whether a CI constraint is satisfiable given the bag state.

    This performs a *local* check: if all of ``x``, ``y``, and the
    conditioning set are within the bag, we can verify whether the
    directed edge pattern is consistent with the required d-separation
    or d-connection.

    For a simple local check: if ``must_be_independent`` is True, we
    require that there is no directed path from ``x`` to ``y`` or
    ``y`` to ``x`` that avoids all conditioning vertices.  If False,
    we require such a path exists.

    Parameters
    ----------
    state : BagState
        Current bag state.
    x, y : NodeId
        Variables in the CI statement.
    conditioning : NodeSet
        Conditioning set.
    must_be_independent : bool
        True if X ⊥ Y | S is required.

    Returns
    -------
    bool
        ``True`` if the constraint is satisfiable given the bag state.
    """
    verts = set(state.vertices)
    if x not in verts or y not in verts:
        return True  # Cannot check — assume OK
    if not conditioning.issubset(verts):
        return True  # Cannot fully check

    # Build directed adjacency among bag vertices
    k = len(state.vertices)
    adj: dict[NodeId, set[NodeId]] = {v: set() for v in state.vertices}
    pair_idx = 0
    for i in range(k):
        for j in range(i + 1, k):
            u, v = state.vertices[i], state.vertices[j]
            s = state.assignment[pair_idx]
            if s == FORWARD:
                adj[u].add(v)
            elif s == BACKWARD:
                adj[v].add(u)
            pair_idx += 1

    # Check for directed path x → ... → y avoiding conditioning set
    has_path_xy = _has_directed_path(adj, x, y, conditioning)
    has_path_yx = _has_directed_path(adj, y, x, conditioning)

    if must_be_independent:
        # For independence: no active path.  A direct directed edge between
        # x and y that avoids conditioning constitutes a violation.
        return not has_path_xy and not has_path_yx
    else:
        # For dependence: need at least one active connection
        return has_path_xy or has_path_yx


def _has_directed_path(
    adj: dict[NodeId, set[NodeId]],
    src: NodeId,
    dst: NodeId,
    blocked: NodeSet,
) -> bool:
    """Check for a directed path from *src* to *dst* avoiding *blocked*."""
    if src == dst:
        return True
    visited: set[NodeId] = set()
    stack = [src]
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        if node in visited:
            continue
        visited.add(node)
        for nb in adj.get(node, ()):
            if nb not in blocked or nb == dst:
                stack.append(nb)
    return False


# ---------------------------------------------------------------------------
# State table operations
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StateTable:
    """DP table mapping assignments (as bitmasks) to minimum costs.

    Provides O(1) lookup and update operations for the DP on tree
    decompositions.

    Attributes
    ----------
    vertices : tuple[NodeId, ...]
        Sorted bag vertices.
    table : dict[int, int]
        Mapping from assignment bitmask to minimum cost.
    witnesses : dict[int, list[tuple[int, int]]]
        Optional mapping from bitmask to list of ``(pair_index, edge_state)``
        edits for reconstruction.
    """

    vertices: tuple[NodeId, ...]
    table: dict[int, int] = field(default_factory=dict)
    witnesses: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    def get_cost(self, mask: int) -> int | None:
        """Return the cost for a given assignment mask, or ``None``."""
        return self.table.get(mask)

    def update(self, mask: int, cost: int, witness: list[tuple[int, int]] | None = None) -> bool:
        """Update the table if *cost* improves the current best.

        Parameters
        ----------
        mask : int
            Assignment bitmask.
        cost : int
            Candidate cost.
        witness : list[tuple[int, int]] | None
            Optional edit witness for reconstruction.

        Returns
        -------
        bool
            ``True`` if the table was updated.
        """
        cur = self.table.get(mask)
        if cur is None or cost < cur:
            self.table[mask] = cost
            if witness is not None:
                self.witnesses[mask] = witness
            return True
        return False

    def min_cost(self) -> int:
        """Return the minimum cost across all states."""
        if not self.table:
            return 0
        return min(self.table.values())

    def best_state(self) -> BagState | None:
        """Return the :class:`BagState` achieving the minimum cost."""
        if not self.table:
            return None
        best_mask = min(self.table, key=self.table.__getitem__)
        return BagState.from_bitmask(
            best_mask,
            self.vertices,
            cost=self.table[best_mask],
        )

    def __len__(self) -> int:
        return len(self.table)


__all__ = [
    "NO_EDGE",
    "FORWARD",
    "BACKWARD",
    "BagState",
    "StateTable",
    "enumerate_bag_states",
    "restrict_state",
    "extend_state",
    "merge_states",
    "is_acyclic_in_bag",
    "check_ci_constraint",
]
