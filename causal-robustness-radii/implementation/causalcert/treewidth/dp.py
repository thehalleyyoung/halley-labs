"""
Dynamic programming on nice tree decompositions for DAG edit distance.

Implements the core DP routine (ALG 7) that computes the minimum number
of edge edits required to overturn a causal conclusion, parameterised
by the treewidth of the moral graph.

The DP processes the nice tree decomposition bottom-up.  At each bag,
the state tracks all feasible directed-edge configurations among the
bag's vertices.  Four handlers correspond to the four nice-node types:

* **Leaf**: initialise a state table for a single vertex.
* **Introduce**: extend each parent state by enumerating all possible
  edge states for pairs involving the introduced vertex.
* **Forget**: project out the forgotten vertex by minimising over its
  edge assignments.
* **Join**: combine states from two children that agree on the shared
  bag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from causalcert.types import AdjacencyMatrix, NodeId, NodeSet
from causalcert.treewidth.bags import (
    BACKWARD,
    FORWARD,
    NO_EDGE,
    BagState,
    StateTable,
    _compute_cost,
    _n_pairs,
    _pair_index,
    is_acyclic_in_bag,
)
from causalcert.treewidth.nice import (
    NiceNodeType,
    NiceTreeDecomposition,
    NiceTreeNode,
)

logger = logging.getLogger(__name__)

_EDGE_STATES = (NO_EDGE, FORWARD, BACKWARD)


# ---------------------------------------------------------------------------
# CI constraint representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CIConstraint:
    """A conditional-independence constraint.

    Attributes
    ----------
    x : NodeId
        First variable.
    y : NodeId
        Second variable.
    conditioning : NodeSet
        Conditioning set.
    must_hold : bool
        ``True`` if X ⊥ Y | S must hold; ``False`` if X ⊥̸ Y | S.
    """

    x: NodeId
    y: NodeId
    conditioning: NodeSet
    must_hold: bool


# ---------------------------------------------------------------------------
# Bitmask helpers
# ---------------------------------------------------------------------------


def _assignment_to_mask(assignment: tuple[int, ...]) -> int:
    """Pack a ternary assignment into a single integer."""
    mask = 0
    for i, s in enumerate(assignment):
        mask += s * (3 ** i)
    return mask


def _mask_to_assignment(mask: int, n_pairs: int) -> tuple[int, ...]:
    """Unpack a single integer into a ternary assignment."""
    result: list[int] = []
    rem = mask
    for _ in range(n_pairs):
        result.append(rem % 3)
        rem //= 3
    return tuple(result)


def _reindex_pair(
    i_old: int,
    j_old: int,
    k_old: int,
    old_to_new: dict[int, int],
    k_new: int,
) -> int:
    """Map a pair index from old vertex indexing to new vertex indexing."""
    ni, nj = old_to_new[i_old], old_to_new[j_old]
    a, b = (ni, nj) if ni < nj else (nj, ni)
    return _pair_index(a, b, k_new)


# ---------------------------------------------------------------------------
# State-table construction helpers
# ---------------------------------------------------------------------------


def _edge_cost(
    u: NodeId,
    v: NodeId,
    state: int,
    source_dag: AdjacencyMatrix,
) -> int:
    """Cost of assigning edge state *state* to pair (u, v) vs source DAG.

    Parameters
    ----------
    u, v : NodeId
        Vertex pair with u < v.
    state : int
        ``NO_EDGE``, ``FORWARD`` (u→v), or ``BACKWARD`` (v→u).
    source_dag : AdjacencyMatrix
        Original DAG adjacency matrix.

    Returns
    -------
    int
        0 if consistent with source, 1 otherwise.
    """
    has_fwd = bool(source_dag[u, v])
    has_bwd = bool(source_dag[v, u])
    if has_fwd:
        src = FORWARD
    elif has_bwd:
        src = BACKWARD
    else:
        src = NO_EDGE
    return 0 if state == src else 1


# ---------------------------------------------------------------------------
# TreewidthDP
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TreewidthDP:
    """Dynamic-programming engine on a nice tree decomposition.

    Computes the minimum-cost edge-edit set that transforms the source DAG
    into a DAG satisfying (or violating) given CI constraints.

    Attributes
    ----------
    source_dag : AdjacencyMatrix
        Adjacency matrix of the original DAG.
    n : int
        Number of vertices.
    k_max : int
        Maximum allowed edit distance (prune states exceeding this).
    ci_constraints : list[CIConstraint]
        CI constraints the target DAG must satisfy.
    tables : dict[int, StateTable]
        DP tables keyed by nice-node ID; populated during :meth:`run`.
    """

    source_dag: AdjacencyMatrix
    n: int = 0
    k_max: int = 10
    ci_constraints: list[CIConstraint] = field(default_factory=list)
    tables: dict[int, StateTable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_dag = np.asarray(self.source_dag, dtype=np.int8)
        self.n = self.source_dag.shape[0]

    # -----------------------------------------------------------------------
    # Leaf node
    # -----------------------------------------------------------------------

    def _process_leaf(self, node: NiceTreeNode) -> StateTable:
        """Initialise the DP table for a leaf node.

        A leaf node contains at most one vertex. There are no pairs, so the
        only state is the empty assignment with cost 0.

        Parameters
        ----------
        node : NiceTreeNode
            Leaf node.

        Returns
        -------
        StateTable
        """
        verts = tuple(sorted(node.bag))
        table = StateTable(vertices=verts)
        # Single state: empty assignment
        table.update(0, 0)
        return table

    # -----------------------------------------------------------------------
    # Introduce node
    # -----------------------------------------------------------------------

    def _process_introduce(
        self,
        node: NiceTreeNode,
        child_table: StateTable,
    ) -> StateTable:
        """Extend states with a newly introduced vertex.

        For each state in the child table, enumerate all possible edge
        assignments for pairs involving the introduced vertex and add valid
        extensions.

        Parameters
        ----------
        node : NiceTreeNode
            Introduce node.
        child_table : StateTable
            DP table from the (single) child.

        Returns
        -------
        StateTable
        """
        v_new = node.special_vertex
        assert v_new is not None

        new_verts = tuple(sorted(node.bag))
        old_verts = tuple(sorted(node.bag - frozenset({v_new})))
        k_new = len(new_verts)
        k_old = len(old_verts)
        np_new = _n_pairs(k_new)
        np_old = _n_pairs(k_old)

        # Map old vertex local indices to new vertex local indices
        old_to_new = {}
        for i, ov in enumerate(old_verts):
            old_to_new[i] = new_verts.index(ov)

        # Determine which new pairs involve v_new
        v_new_idx = new_verts.index(v_new)
        new_pair_positions: list[tuple[int, int, int]] = []
        # (new_pair_linear_idx, other_vertex_global_id, position_in_new_verts)
        for i in range(k_new):
            for j in range(i + 1, k_new):
                if new_verts[i] == v_new or new_verts[j] == v_new:
                    linear = _pair_index(i, j, k_new)
                    other_idx = j if new_verts[i] == v_new else i
                    new_pair_positions.append(
                        (linear, new_verts[other_idx], other_idx)
                    )

        n_new_pairs = len(new_pair_positions)

        table = StateTable(vertices=new_verts)

        for old_mask, old_cost in child_table.table.items():
            if old_cost > self.k_max:
                continue

            old_assignment = _mask_to_assignment(old_mask, np_old)

            # Enumerate all assignments for the new pairs
            for combo in product(_EDGE_STATES, repeat=n_new_pairs):
                # Build the full new assignment
                new_assignment = [0] * np_new

                # Copy old pairs
                for i_old in range(k_old):
                    for j_old in range(i_old + 1, k_old):
                        old_lin = _pair_index(i_old, j_old, k_old)
                        ni = old_to_new[i_old]
                        nj = old_to_new[j_old]
                        a, b = (ni, nj) if ni < nj else (nj, ni)
                        new_lin = _pair_index(a, b, k_new)
                        new_assignment[new_lin] = old_assignment[old_lin]

                # Set new pairs
                extra_cost = 0
                for idx, (lin, other_global, _) in enumerate(
                    new_pair_positions
                ):
                    new_assignment[lin] = combo[idx]
                    u, v = (min(v_new, other_global), max(v_new, other_global))
                    extra_cost += _edge_cost(u, v, combo[idx], self.source_dag)

                total_cost = old_cost + extra_cost
                if total_cost > self.k_max:
                    continue

                # Check local acyclicity
                state = BagState(
                    vertices=new_verts,
                    assignment=tuple(new_assignment),
                    cost=total_cost,
                )
                if not is_acyclic_in_bag(state):
                    continue

                new_mask = _assignment_to_mask(tuple(new_assignment))
                table.update(new_mask, total_cost)

        return table

    # -----------------------------------------------------------------------
    # Forget node
    # -----------------------------------------------------------------------

    def _process_forget(
        self,
        node: NiceTreeNode,
        child_table: StateTable,
    ) -> StateTable:
        """Project out the forgotten vertex by minimising over its assignments.

        Parameters
        ----------
        node : NiceTreeNode
            Forget node.
        child_table : StateTable
            DP table from the child.

        Returns
        -------
        StateTable
        """
        v_forget = node.special_vertex
        assert v_forget is not None

        new_verts = tuple(sorted(node.bag))
        child_verts = child_table.vertices
        k_child = len(child_verts)
        k_new = len(new_verts)
        np_child = _n_pairs(k_child)
        np_new = _n_pairs(k_new)

        # Map new vertex local indices to child vertex local indices
        new_to_child: dict[int, int] = {}
        for i, nv in enumerate(new_verts):
            new_to_child[i] = child_verts.index(nv)

        table = StateTable(vertices=new_verts)

        for child_mask, child_cost in child_table.table.items():
            if child_cost > self.k_max:
                continue

            child_assignment = _mask_to_assignment(child_mask, np_child)

            # Project: extract only the pairs among new_verts
            new_assignment = [0] * np_new
            for i in range(k_new):
                for j in range(i + 1, k_new):
                    ci = new_to_child[i]
                    cj = new_to_child[j]
                    a, b = (ci, cj) if ci < cj else (cj, ci)
                    child_lin = _pair_index(a, b, k_child)
                    new_lin = _pair_index(i, j, k_new)
                    new_assignment[new_lin] = child_assignment[child_lin]

            new_mask = _assignment_to_mask(tuple(new_assignment))
            table.update(new_mask, child_cost)

        return table

    # -----------------------------------------------------------------------
    # Join node
    # -----------------------------------------------------------------------

    def _process_join(
        self,
        node: NiceTreeNode,
        left_table: StateTable,
        right_table: StateTable,
    ) -> StateTable:
        """Combine states from two children at a join node.

        Both children have the same bag.  Two states are compatible iff
        they agree on all edge assignments within the bag.  The cost of
        the merged state is the sum of the child costs minus the cost
        attributed to edges within the bag (counted in both children).

        Parameters
        ----------
        node : NiceTreeNode
            Join node.
        left_table, right_table : StateTable
            DP tables from the two children.

        Returns
        -------
        StateTable
        """
        verts = tuple(sorted(node.bag))
        k = len(verts)
        np_ = _n_pairs(k)
        table = StateTable(vertices=verts)

        # Compute the "shared cost" for each mask (edges within bag)
        shared_cost_cache: dict[int, int] = {}

        for mask, left_cost in left_table.table.items():
            if mask not in right_table.table:
                continue
            right_cost = right_table.table[mask]

            if mask not in shared_cost_cache:
                assignment = _mask_to_assignment(mask, np_)
                shared = _compute_cost(verts, assignment, self.source_dag)
                shared_cost_cache[mask] = shared

            shared = shared_cost_cache[mask]
            merged_cost = left_cost + right_cost - shared

            if merged_cost > self.k_max:
                continue

            table.update(mask, merged_cost)

        return table

    # -----------------------------------------------------------------------
    # Main DP loop
    # -----------------------------------------------------------------------

    def run(self, ntd: NiceTreeDecomposition) -> StateTable:
        """Execute the DP bottom-up on the nice tree decomposition.

        Parameters
        ----------
        ntd : NiceTreeDecomposition
            Nice tree decomposition.

        Returns
        -------
        StateTable
            DP table at the root node.
        """
        self.tables.clear()
        node_map = {nd.node_id: nd for nd in ntd.nodes}
        order = ntd.postorder()

        for nd in order:
            if nd.node_type == NiceNodeType.LEAF:
                tbl = self._process_leaf(nd)

            elif nd.node_type == NiceNodeType.INTRODUCE:
                child_id = nd.children_ids[0]
                child_tbl = self.tables[child_id]
                tbl = self._process_introduce(nd, child_tbl)

            elif nd.node_type == NiceNodeType.FORGET:
                child_id = nd.children_ids[0]
                child_tbl = self.tables[child_id]
                tbl = self._process_forget(nd, child_tbl)

            elif nd.node_type == NiceNodeType.JOIN:
                left_id, right_id = nd.children_ids
                left_tbl = self.tables[left_id]
                right_tbl = self.tables[right_id]
                tbl = self._process_join(nd, left_tbl, right_tbl)

            else:
                raise ValueError(f"Unknown node type: {nd.node_type}")

            self.tables[nd.node_id] = tbl

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Node %d (%s): %d states, min cost %d",
                    nd.node_id,
                    nd.node_type.value,
                    len(tbl),
                    tbl.min_cost() if tbl.table else -1,
                )

        return self.tables[ntd.root_id]


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def compute_min_edit_distance(
    td: NiceTreeDecomposition,
    source_dag: AdjacencyMatrix,
    ci_constraints: list[CIConstraint] | None = None,
    k_max: int = 10,
) -> int:
    """Compute the minimum edit distance via DP on a nice tree decomposition.

    Parameters
    ----------
    td : NiceTreeDecomposition
        Nice tree decomposition of the moral graph.
    source_dag : AdjacencyMatrix
        Adjacency matrix of the original DAG.
    ci_constraints : list[CIConstraint] | None
        CI constraints the edited DAG must satisfy.  Currently used for
        post-filtering; full integration is a future extension.
    k_max : int
        Maximum edit distance to explore.

    Returns
    -------
    int
        Minimum number of edge edits, or ``k_max + 1`` if no solution
        exists within the budget.
    """
    dp = TreewidthDP(
        source_dag=source_dag,
        k_max=k_max,
        ci_constraints=ci_constraints or [],
    )

    root_table = dp.run(td)

    if not root_table.table:
        return k_max + 1

    return root_table.min_cost()


def compute_min_edit_distance_with_witness(
    td: NiceTreeDecomposition,
    source_dag: AdjacencyMatrix,
    ci_constraints: list[CIConstraint] | None = None,
    k_max: int = 10,
) -> tuple[int, AdjacencyMatrix | None]:
    """Compute minimum edit distance and reconstruct the witness DAG.

    Parameters
    ----------
    td : NiceTreeDecomposition
        Nice tree decomposition.
    source_dag : AdjacencyMatrix
        Original DAG.
    ci_constraints : list[CIConstraint] | None
        CI constraints.
    k_max : int
        Maximum edit distance.

    Returns
    -------
    tuple[int, AdjacencyMatrix | None]
        ``(distance, witness_dag)`` where *witness_dag* is the closest DAG
        achieving the minimum distance, or ``None`` if no solution found.
    """
    source_dag = np.asarray(source_dag, dtype=np.int8)
    dp = TreewidthDP(
        source_dag=source_dag,
        k_max=k_max,
        ci_constraints=ci_constraints or [],
    )

    root_table = dp.run(td)

    if not root_table.table:
        return k_max + 1, None

    min_dist = root_table.min_cost()

    # Reconstruct witness by tracing back through the DP tables
    witness = _reconstruct_witness(dp, td, source_dag)

    return min_dist, witness


def _reconstruct_witness(
    dp: TreewidthDP,
    ntd: NiceTreeDecomposition,
    source_dag: AdjacencyMatrix,
) -> AdjacencyMatrix:
    """Reconstruct the witness DAG from DP tables.

    Walks top-down through the nice tree decomposition, selecting the
    optimal assignment at each node and writing the corresponding edges
    into the result matrix.

    Parameters
    ----------
    dp : TreewidthDP
        Completed DP engine.
    ntd : NiceTreeDecomposition
        Nice tree decomposition.
    source_dag : AdjacencyMatrix
        Original DAG.

    Returns
    -------
    AdjacencyMatrix
        The witness DAG achieving minimum edit distance.
    """
    n = dp.n
    result = source_dag.copy()
    node_map = {nd.node_id: nd for nd in ntd.nodes}

    # Walk top-down, selecting best masks
    # Start at the root and propagate optimal choices
    chosen: dict[int, int] = {}  # node_id -> chosen mask

    root_table = dp.tables[ntd.root_id]
    if not root_table.table:
        return result

    root_mask = min(root_table.table, key=root_table.table.__getitem__)
    chosen[ntd.root_id] = root_mask

    # Top-down BFS
    queue = deque([ntd.root_id])
    visited: set[int] = {ntd.root_id}

    while queue:
        nid = queue.popleft()
        nd = node_map[nid]
        parent_mask = chosen[nid]

        for cid in nd.children_ids:
            if cid in visited:
                continue
            visited.add(cid)

            child_table = dp.tables[cid]
            if nd.node_type == NiceNodeType.JOIN:
                # Children must have same mask as parent
                if parent_mask in child_table.table:
                    chosen[cid] = parent_mask
                else:
                    # Fallback: best available
                    chosen[cid] = min(
                        child_table.table,
                        key=child_table.table.__getitem__,
                    )
            elif nd.node_type == NiceNodeType.FORGET:
                # Child has one extra vertex; find child mask consistent
                # with parent projection
                best_child_mask = _find_best_child_forget(
                    nd, parent_mask, child_table, dp,
                )
                chosen[cid] = best_child_mask
            elif nd.node_type == NiceNodeType.INTRODUCE:
                # Child has one fewer vertex; project parent to child
                best_child_mask = _find_best_child_introduce(
                    nd, parent_mask, child_table, dp,
                )
                chosen[cid] = best_child_mask
            else:
                if child_table.table:
                    chosen[cid] = min(
                        child_table.table,
                        key=child_table.table.__getitem__,
                    )

            queue.append(cid)

    # Now write edges from chosen assignments into result
    # For each node with a chosen mask, decode the assignment and write edges
    edge_written: set[tuple[int, int]] = set()
    for nid, mask in chosen.items():
        nd = node_map[nid]
        verts = tuple(sorted(nd.bag))
        k = len(verts)
        np_ = _n_pairs(k)
        if np_ == 0:
            continue

        assignment = _mask_to_assignment(mask, np_)
        pair_idx = 0
        for i in range(k):
            for j in range(i + 1, k):
                u, v = verts[i], verts[j]
                if (u, v) not in edge_written:
                    s = assignment[pair_idx]
                    # Clear both directions
                    result[u, v] = 0
                    result[v, u] = 0
                    if s == FORWARD:
                        result[u, v] = 1
                    elif s == BACKWARD:
                        result[v, u] = 1
                    edge_written.add((u, v))
                pair_idx += 1

    return result


def _find_best_child_forget(
    node: NiceTreeNode,
    parent_mask: int,
    child_table: StateTable,
    dp: TreewidthDP,
) -> int:
    """Find the best child mask when forgetting a vertex.

    The child has one extra vertex.  We look through all child masks
    and pick the one that (a) projects to parent_mask and (b) has
    minimum cost.
    """
    v_forget = node.special_vertex
    new_verts = tuple(sorted(node.bag))
    child_verts = child_table.vertices
    k_child = len(child_verts)
    k_new = len(new_verts)
    np_child = _n_pairs(k_child)
    np_new = _n_pairs(k_new)

    new_to_child: dict[int, int] = {}
    for i, nv in enumerate(new_verts):
        new_to_child[i] = child_verts.index(nv)

    parent_assignment = _mask_to_assignment(parent_mask, np_new)

    best_mask = -1
    best_cost = dp.k_max + 2

    for c_mask, c_cost in child_table.table.items():
        c_assign = _mask_to_assignment(c_mask, np_child)

        # Check projection matches parent
        match = True
        for i in range(k_new):
            for j in range(i + 1, k_new):
                ci = new_to_child[i]
                cj = new_to_child[j]
                a, b = (ci, cj) if ci < cj else (cj, ci)
                c_lin = _pair_index(a, b, k_child)
                p_lin = _pair_index(i, j, k_new)
                if c_assign[c_lin] != parent_assignment[p_lin]:
                    match = False
                    break
            if not match:
                break

        if match and c_cost < best_cost:
            best_cost = c_cost
            best_mask = c_mask

    if best_mask == -1 and child_table.table:
        best_mask = min(child_table.table, key=child_table.table.__getitem__)

    return best_mask


def _find_best_child_introduce(
    node: NiceTreeNode,
    parent_mask: int,
    child_table: StateTable,
    dp: TreewidthDP,
) -> int:
    """Find the best child mask when introducing a vertex.

    The child has one fewer vertex.  We project the parent's assignment
    onto the child's vertex set.
    """
    v_new = node.special_vertex
    parent_verts = tuple(sorted(node.bag))
    child_verts = child_table.vertices
    k_parent = len(parent_verts)
    k_child = len(child_verts)
    np_parent = _n_pairs(k_parent)
    np_child = _n_pairs(k_child)

    if np_child == 0:
        return 0

    parent_assignment = _mask_to_assignment(parent_mask, np_parent)

    # Map child vertex local indices to parent vertex local indices
    child_to_parent: dict[int, int] = {}
    for i, cv in enumerate(child_verts):
        child_to_parent[i] = parent_verts.index(cv)

    projected = [0] * np_child
    for i in range(k_child):
        for j in range(i + 1, k_child):
            pi = child_to_parent[i]
            pj = child_to_parent[j]
            a, b = (pi, pj) if pi < pj else (pj, pi)
            p_lin = _pair_index(a, b, k_parent)
            c_lin = _pair_index(i, j, k_child)
            projected[c_lin] = parent_assignment[p_lin]

    projected_mask = _assignment_to_mask(tuple(projected))
    if projected_mask in child_table.table:
        return projected_mask

    # Fallback
    if child_table.table:
        return min(child_table.table, key=child_table.table.__getitem__)
    return 0


__all__ = [
    "CIConstraint",
    "TreewidthDP",
    "compute_min_edit_distance",
    "compute_min_edit_distance_with_witness",
]
