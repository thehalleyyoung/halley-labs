"""
Edge edit operations and k-neighbourhood enumeration.

Provides functions for computing edit distance between DAGs, enumerating
all DAGs within a given edit distance (the *k-neighbourhood*), and applying
batches of edits.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Iterator, Sequence

import numpy as np

from causalcert.types import AdjacencyMatrix, EditType, NodeId, StructuralEdit
from causalcert.dag.validation import is_dag


def edit_distance(adj1: AdjacencyMatrix, adj2: AdjacencyMatrix) -> int:
    """Compute the structural Hamming distance (edit distance) between two DAGs.

    Each edge addition, deletion, or reversal counts as one edit.

    Parameters
    ----------
    adj1, adj2 : AdjacencyMatrix
        Adjacency matrices of the two DAGs (must be same dimension).

    Returns
    -------
    int
        Minimum number of single-edge edits to transform *adj1* into *adj2*.
    """
    adj1 = np.asarray(adj1, dtype=np.int8)
    adj2 = np.asarray(adj2, dtype=np.int8)
    if adj1.shape != adj2.shape:
        raise ValueError(
            f"Adjacency matrices must have same shape: {adj1.shape} vs {adj2.shape}"
        )
    n = adj1.shape[0]
    distance = 0

    # Track which edges have been accounted for
    handled = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(n):
            if handled[i, j]:
                continue
            if adj1[i, j] == adj2[i, j] and adj1[j, i] == adj2[j, i]:
                continue

            # Case: reversal — adj1 has i->j, adj2 has j->i (or vice versa)
            if (adj1[i, j] and not adj1[j, i] and
                    not adj2[i, j] and adj2[j, i]):
                distance += 1
                handled[i, j] = True
                handled[j, i] = True
            elif (adj1[j, i] and not adj1[i, j] and
                  not adj2[j, i] and adj2[i, j]):
                distance += 1
                handled[i, j] = True
                handled[j, i] = True
            else:
                # Addition or deletion
                if adj1[i, j] != adj2[i, j]:
                    distance += 1
                    handled[i, j] = True
                if adj1[j, i] != adj2[j, i] and not handled[j, i]:
                    distance += 1
                    handled[j, i] = True

    return distance


def diff_edits(adj1: AdjacencyMatrix, adj2: AdjacencyMatrix) -> list[StructuralEdit]:
    """Return the list of edits transforming *adj1* into *adj2*.

    Parameters
    ----------
    adj1, adj2 : AdjacencyMatrix
        Adjacency matrices.

    Returns
    -------
    list[StructuralEdit]
    """
    adj1 = np.asarray(adj1, dtype=np.int8)
    adj2 = np.asarray(adj2, dtype=np.int8)
    if adj1.shape != adj2.shape:
        raise ValueError("Adjacency matrices must have same shape")
    n = adj1.shape[0]
    edits: list[StructuralEdit] = []
    handled = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(n):
            if handled[i, j]:
                continue
            if adj1[i, j] == adj2[i, j] and adj1[j, i] == adj2[j, i]:
                continue

            # Reversal: i->j becomes j->i
            if (adj1[i, j] and not adj1[j, i] and
                    not adj2[i, j] and adj2[j, i]):
                edits.append(StructuralEdit(EditType.REVERSE, i, j))
                handled[i, j] = True
                handled[j, i] = True
            elif (adj1[j, i] and not adj1[i, j] and
                  not adj2[j, i] and adj2[i, j]):
                edits.append(StructuralEdit(EditType.REVERSE, j, i))
                handled[i, j] = True
                handled[j, i] = True
            else:
                if adj1[i, j] and not adj2[i, j]:
                    edits.append(StructuralEdit(EditType.DELETE, i, j))
                    handled[i, j] = True
                elif not adj1[i, j] and adj2[i, j]:
                    edits.append(StructuralEdit(EditType.ADD, i, j))
                    handled[i, j] = True
                if not handled[j, i]:
                    if adj1[j, i] and not adj2[j, i]:
                        edits.append(StructuralEdit(EditType.DELETE, j, i))
                        handled[j, i] = True
                    elif not adj1[j, i] and adj2[j, i]:
                        edits.append(StructuralEdit(EditType.ADD, j, i))
                        handled[j, i] = True

    return edits


def apply_edit(adj: AdjacencyMatrix, edit: StructuralEdit) -> AdjacencyMatrix:
    """Return a *new* adjacency matrix with *edit* applied.

    Does **not** validate acyclicity — use :class:`CausalDAG` for safe edits.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Original adjacency matrix.
    edit : StructuralEdit
        Edit to apply.

    Returns
    -------
    AdjacencyMatrix
        Modified copy of the adjacency matrix.
    """
    result = np.asarray(adj, dtype=np.int8).copy()
    if edit.edit_type == EditType.ADD:
        result[edit.source, edit.target] = 1
    elif edit.edit_type == EditType.DELETE:
        result[edit.source, edit.target] = 0
    elif edit.edit_type == EditType.REVERSE:
        result[edit.source, edit.target] = 0
        result[edit.target, edit.source] = 1
    return result


def apply_edits(
    adj: AdjacencyMatrix, edits: Sequence[StructuralEdit]
) -> AdjacencyMatrix:
    """Apply a sequence of edits, returning the final adjacency matrix.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Original adjacency matrix.
    edits : Sequence[StructuralEdit]
        Edits to apply in order.

    Returns
    -------
    AdjacencyMatrix
    """
    result = np.asarray(adj, dtype=np.int8).copy()
    for edit in edits:
        result = apply_edit(result, edit)
    return result


def all_single_edits(adj: AdjacencyMatrix) -> list[StructuralEdit]:
    """Enumerate all valid single-edge edits of a DAG.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Current DAG adjacency matrix.

    Returns
    -------
    list[StructuralEdit]
        All additions, deletions, and reversals that yield a valid DAG.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    edits: list[StructuralEdit] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            if adj[i, j]:
                # Can delete this edge
                edits.append(StructuralEdit(EditType.DELETE, i, j))

                # Can reverse this edge (if it doesn't create a cycle)
                trial = adj.copy()
                trial[i, j] = 0
                trial[j, i] = 1
                if is_dag(trial):
                    edits.append(StructuralEdit(EditType.REVERSE, i, j))
            else:
                # Can potentially add this edge
                trial = adj.copy()
                trial[i, j] = 1
                if is_dag(trial):
                    edits.append(StructuralEdit(EditType.ADD, i, j))

    return edits


def _all_possible_edits(adj: np.ndarray) -> list[StructuralEdit]:
    """Generate all structurally possible edits (without acyclicity check)."""
    n = adj.shape[0]
    edits: list[StructuralEdit] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if adj[i, j]:
                edits.append(StructuralEdit(EditType.DELETE, i, j))
                edits.append(StructuralEdit(EditType.REVERSE, i, j))
            else:
                edits.append(StructuralEdit(EditType.ADD, i, j))
    return edits


def k_neighbourhood(
    adj: AdjacencyMatrix,
    k: int,
    *,
    acyclic_only: bool = True,
) -> Iterator[tuple[AdjacencyMatrix, tuple[StructuralEdit, ...]]]:
    """Yield all DAGs within edit distance ≤ *k* from *adj*.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Centre DAG adjacency matrix.
    k : int
        Maximum edit distance.
    acyclic_only : bool
        If ``True`` (default), skip edits that create cycles.

    Yields
    ------
    tuple[AdjacencyMatrix, tuple[StructuralEdit, ...]]
        ``(perturbed_adj, edits)`` pairs.
    """
    adj = np.asarray(adj, dtype=np.int8)

    if k <= 0:
        yield adj.copy(), ()
        return

    # Yield the original graph
    yield adj.copy(), ()

    # BFS over the edit space
    # State: (adjacency_matrix_bytes, edit_sequence)
    seen: set[bytes] = {adj.tobytes()}

    # Use a queue of (current_adj, edit_sequence, depth)
    queue: deque[tuple[np.ndarray, tuple[StructuralEdit, ...], int]] = deque()
    queue.append((adj.copy(), (), 0))

    while queue:
        current, edit_seq, depth = queue.popleft()
        if depth >= k:
            continue

        # Generate all single edits from current
        for edit in _all_possible_edits(current):
            new_adj = apply_edit(current, edit)

            if acyclic_only and not is_dag(new_adj):
                continue

            key = new_adj.tobytes()
            if key in seen:
                continue
            seen.add(key)

            new_edits = edit_seq + (edit,)
            yield new_adj, new_edits

            if depth + 1 < k:
                queue.append((new_adj, new_edits, depth + 1))


def k_neighbourhood_pruned(
    adj: AdjacencyMatrix,
    k: int,
    relevant_nodes: set[int] | None = None,
) -> Iterator[tuple[AdjacencyMatrix, tuple[StructuralEdit, ...]]]:
    """Yield DAGs within edit distance ≤ *k*, restricted to relevant nodes.

    Like :func:`k_neighbourhood` but only considers edits involving nodes
    in *relevant_nodes*, which prunes the search space significantly.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Centre DAG adjacency matrix.
    k : int
        Maximum edit distance.
    relevant_nodes : set[int] | None
        If given, only edges between these nodes are considered for edits.

    Yields
    ------
    tuple[AdjacencyMatrix, tuple[StructuralEdit, ...]]
        ``(perturbed_adj, edits)`` pairs.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]

    if relevant_nodes is None:
        relevant_nodes = set(range(n))

    if k <= 0:
        yield adj.copy(), ()
        return

    yield adj.copy(), ()

    def _relevant_edits(a: np.ndarray) -> list[StructuralEdit]:
        edits: list[StructuralEdit] = []
        for i in relevant_nodes:
            for j in relevant_nodes:
                if i == j:
                    continue
                if a[i, j]:
                    edits.append(StructuralEdit(EditType.DELETE, i, j))
                    edits.append(StructuralEdit(EditType.REVERSE, i, j))
                else:
                    edits.append(StructuralEdit(EditType.ADD, i, j))
        return edits

    seen: set[bytes] = {adj.tobytes()}
    queue: deque[tuple[np.ndarray, tuple[StructuralEdit, ...], int]] = deque()
    queue.append((adj.copy(), (), 0))

    while queue:
        current, edit_seq, depth = queue.popleft()
        if depth >= k:
            continue

        for edit in _relevant_edits(current):
            new_adj = apply_edit(current, edit)
            if not is_dag(new_adj):
                continue

            key = new_adj.tobytes()
            if key in seen:
                continue
            seen.add(key)

            new_edits = edit_seq + (edit,)
            yield new_adj, new_edits

            if depth + 1 < k:
                queue.append((new_adj, new_edits, depth + 1))


def canonicalize_edit_sequence(
    edits: Sequence[StructuralEdit],
) -> tuple[StructuralEdit, ...]:
    """Canonicalize an edit sequence by sorting edits in a deterministic order.

    Edits are sorted by (edit_type, source, target) to provide a canonical
    representation for comparing edit sequences.

    Parameters
    ----------
    edits : Sequence[StructuralEdit]
        Edit sequence to canonicalize.

    Returns
    -------
    tuple[StructuralEdit, ...]
        Canonical ordering of the edits.
    """
    return tuple(sorted(
        edits,
        key=lambda e: (e.edit_type.value, e.source, e.target),
    ))


def edit_path(
    adj1: AdjacencyMatrix,
    adj2: AdjacencyMatrix,
) -> list[StructuralEdit]:
    """Find a sequence of edits transforming *adj1* into *adj2*.

    Each intermediate graph is guaranteed to be a DAG if both inputs are DAGs.
    Uses a greedy approach: apply deletions first, then reversals, then additions.

    Parameters
    ----------
    adj1, adj2 : AdjacencyMatrix
        Source and target DAG adjacency matrices.

    Returns
    -------
    list[StructuralEdit]
        Sequence of edits. Each intermediate result is a valid DAG.
    """
    adj1 = np.asarray(adj1, dtype=np.int8)
    adj2 = np.asarray(adj2, dtype=np.int8)
    edits = diff_edits(adj1, adj2)

    # Sort: deletions first, then reversals, then additions
    # This order tends to maintain acyclicity
    order = {EditType.DELETE: 0, EditType.REVERSE: 1, EditType.ADD: 2}
    edits.sort(key=lambda e: order[e.edit_type])
    return edits


def single_edit_perturbations(
    adj: AdjacencyMatrix,
) -> list[tuple[AdjacencyMatrix, StructuralEdit]]:
    """Generate all single-edit perturbations that yield valid DAGs.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.

    Returns
    -------
    list[tuple[AdjacencyMatrix, StructuralEdit]]
        List of ``(new_adj, edit)`` pairs.
    """
    adj = np.asarray(adj, dtype=np.int8)
    results: list[tuple[AdjacencyMatrix, StructuralEdit]] = []
    for edit in all_single_edits(adj):
        new_adj = apply_edit(adj, edit)
        results.append((new_adj, edit))
    return results
