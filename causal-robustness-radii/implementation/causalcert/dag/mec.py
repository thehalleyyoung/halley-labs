"""
Markov equivalence class (MEC) operations and CPDAG conversion.

Two DAGs are Markov-equivalent if they encode the same set of conditional
independencies.  The CPDAG (completed partially directed acyclic graph)
represents the equivalence class, with undirected edges for reversible
orientations and directed edges for compelled orientations.
"""

from __future__ import annotations

from collections import deque
from itertools import permutations
from typing import Iterator

import numpy as np

from causalcert.types import AdjacencyMatrix, NodeId
from causalcert.dag.validation import is_dag


def _skeleton(adj: np.ndarray) -> np.ndarray:
    """Return the skeleton (undirected adjacency matrix) of a DAG."""
    return np.maximum(adj, adj.T).astype(np.int8)


def _v_structures(adj: np.ndarray) -> set[tuple[int, int, int]]:
    """Return all v-structures (i -> j <- k where i and k are not adjacent).

    Returns a set of (i, j, k) with i < k for canonical ordering.
    """
    n = adj.shape[0]
    v_structs: set[tuple[int, int, int]] = set()
    for j in range(n):
        parents = list(np.nonzero(adj[:, j])[0])
        for pi in range(len(parents)):
            for pk in range(pi + 1, len(parents)):
                i, k = int(parents[pi]), int(parents[pk])
                # Check that i and k are NOT adjacent
                if not adj[i, k] and not adj[k, i]:
                    v_structs.add((min(i, k), j, max(i, k)))
    return v_structs


def to_cpdag(adj: AdjacencyMatrix) -> np.ndarray:
    """Convert a DAG adjacency matrix to its CPDAG representation.

    Uses the Chickering (2002) algorithm to identify compelled vs. reversible
    edges. An edge is compelled if reversing it would change the Markov
    equivalence class (i.e., change the skeleton or v-structures).

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.

    Returns
    -------
    np.ndarray
        Matrix where:
        - ``cpdag[i,j] == 1 and cpdag[j,i] == 0`` → compelled *i → j*
        - ``cpdag[i,j] == 1 and cpdag[j,i] == 1`` → reversible *i — j*
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]

    if n == 0:
        return np.zeros((0, 0), dtype=np.int8)

    # Start by labeling all edges as "unknown"
    # Then apply Meek rules to find compelled edges

    # Step 1: Find v-structures — these edges are always compelled
    v_structs = _v_structures(adj)
    compelled = np.zeros((n, n), dtype=bool)

    for i, j, k in v_structs:
        compelled[i, j] = True
        compelled[k, j] = True

    # Step 2: Iteratively apply Meek rules until convergence
    changed = True
    while changed:
        changed = False

        for i in range(n):
            for j in range(n):
                if not adj[i, j] or compelled[i, j]:
                    continue

                # Rule R1: If there exists k such that k -> i (compelled)
                # and k is not adjacent to j, then i -> j is compelled
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if adj[k, i] and compelled[k, i] and not adj[k, j] and not adj[j, k]:
                        compelled[i, j] = True
                        changed = True
                        break

                if compelled[i, j]:
                    continue

                # Rule R2: If there exists k such that i -> k (compelled)
                # and k -> j (compelled), then i -> j is compelled
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (adj[i, k] and compelled[i, k] and
                            adj[k, j] and compelled[k, j]):
                        compelled[i, j] = True
                        changed = True
                        break

                if compelled[i, j]:
                    continue

                # Rule R3: If there exist k, l such that k -> j and l -> j
                # (both compelled), k-i and l-i are undirected, and k,l not adj
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if not (adj[k, j] and compelled[k, j]):
                        continue
                    if not (adj[k, i] and adj[i, k] and not compelled[k, i] and not compelled[i, k]):
                        continue
                    for l in range(k + 1, n):
                        if l == i or l == j:
                            continue
                        if not (adj[l, j] and compelled[l, j]):
                            continue
                        if not (adj[l, i] and adj[i, l] and not compelled[l, i] and not compelled[i, l]):
                            continue
                        if not adj[k, l] and not adj[l, k]:
                            compelled[i, j] = True
                            changed = True
                            break
                    if compelled[i, j]:
                        break

                if compelled[i, j]:
                    continue

                # Rule R4: If there exists k such that k -> l (compelled),
                # l -> j (compelled), k-i undirected, l not adj to i
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if not (adj[k, i] and adj[i, k] and not compelled[k, i] and not compelled[i, k]):
                        continue
                    for l in range(n):
                        if l == i or l == j or l == k:
                            continue
                        if (adj[k, l] and compelled[k, l] and
                                adj[l, j] and compelled[l, j] and
                                not adj[l, i] and not adj[i, l]):
                            compelled[i, j] = True
                            changed = True
                            break
                    if compelled[i, j]:
                        break

    # Build CPDAG matrix
    cpdag = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                if compelled[i, j]:
                    cpdag[i, j] = 1
                else:
                    # Reversible: mark as undirected
                    cpdag[i, j] = 1
                    cpdag[j, i] = 1

    return cpdag


def mec_size_bound(adj: AdjacencyMatrix) -> int:
    """Upper bound on the number of DAGs in the Markov equivalence class.

    Uses the number of reversible edges to compute 2^(n_reversible) as
    a loose upper bound (the actual number is typically much smaller due
    to acyclicity constraints).

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.

    Returns
    -------
    int
        Upper bound on MEC size.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return 1

    cpdag = to_cpdag(adj)
    n_reversible = 0
    for i in range(n):
        for j in range(i + 1, n):
            if cpdag[i, j] and cpdag[j, i]:
                n_reversible += 1

    # The number of DAGs in MEC is at most 2^(n_reversible) but usually
    # much less due to acyclicity constraints
    return max(1, 2 ** n_reversible)


def is_mec_equivalent(adj1: AdjacencyMatrix, adj2: AdjacencyMatrix) -> bool:
    """Check whether two DAGs belong to the same Markov equivalence class.

    Two DAGs are Markov equivalent iff they have the same skeleton and
    same set of v-structures (Verma and Pearl, 1990).

    Parameters
    ----------
    adj1, adj2 : AdjacencyMatrix
        Two DAG adjacency matrices.

    Returns
    -------
    bool
        ``True`` if both DAGs have the same CPDAG.
    """
    adj1 = np.asarray(adj1, dtype=np.int8)
    adj2 = np.asarray(adj2, dtype=np.int8)

    if adj1.shape != adj2.shape:
        return False

    # Same skeleton?
    skel1 = _skeleton(adj1)
    skel2 = _skeleton(adj2)
    if not np.array_equal(skel1, skel2):
        return False

    # Same v-structures?
    return _v_structures(adj1) == _v_structures(adj2)


def compelled_edges(adj: AdjacencyMatrix) -> list[tuple[NodeId, NodeId]]:
    """Return edges that are compelled (present in every MEC member).

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.

    Returns
    -------
    list[tuple[NodeId, NodeId]]
        List of compelled directed edges ``(i, j)``.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    cpdag = to_cpdag(adj)
    result: list[tuple[NodeId, NodeId]] = []
    for i in range(n):
        for j in range(n):
            if cpdag[i, j] and not cpdag[j, i]:
                result.append((i, j))
    return result


def reversible_edges(adj: AdjacencyMatrix) -> list[tuple[NodeId, NodeId]]:
    """Return edges that are reversible within the MEC.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.

    Returns
    -------
    list[tuple[NodeId, NodeId]]
        List of reversible edges ``(i, j)`` (only one direction listed).
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    cpdag = to_cpdag(adj)
    result: list[tuple[NodeId, NodeId]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if cpdag[i, j] and cpdag[j, i]:
                # Report in the direction present in the original DAG
                if adj[i, j]:
                    result.append((i, j))
                else:
                    result.append((j, i))
    return result


def cpdag_to_dag(cpdag: np.ndarray) -> np.ndarray | None:
    """Extend a CPDAG to a consistent DAG (PDAG extension).

    Implements the algorithm by Dor and Tarsi (1992) for extending a
    partially directed graph to a fully directed DAG.

    Parameters
    ----------
    cpdag : np.ndarray
        CPDAG adjacency matrix.

    Returns
    -------
    np.ndarray | None
        A DAG consistent with the CPDAG, or None if no extension exists.
    """
    cpdag = np.asarray(cpdag, dtype=np.int8)
    n = cpdag.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.int8)

    dag = cpdag.copy()
    remaining = set(range(n))

    # Iteratively find a sink in the PDAG
    for _ in range(n):
        # A node v is a sink candidate if:
        # 1. Every undirected neighbor of v is also a neighbor of all other
        #    undirected neighbors of v
        # 2. v has no directed outgoing edges to remaining nodes
        sink = None
        for v in remaining:
            # Check no directed outgoing edges
            has_out = False
            for w in remaining:
                if w == v:
                    continue
                if dag[v, w] and not dag[w, v]:
                    has_out = True
                    break
            if has_out:
                continue

            # Find undirected neighbors
            undirected_nb = [
                w for w in remaining
                if w != v and dag[v, w] and dag[w, v]
            ]

            # Check that undirected neighbors form a clique together
            # with any directed parents
            valid = True
            for i in range(len(undirected_nb)):
                for j in range(i + 1, len(undirected_nb)):
                    if not dag[undirected_nb[i], undirected_nb[j]] and \
                       not dag[undirected_nb[j], undirected_nb[i]]:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                sink = v
                break

        if sink is None:
            return None  # No valid extension exists

        # Orient all undirected edges adjacent to sink as incoming
        for w in remaining:
            if w == sink:
                continue
            if dag[sink, w] and dag[w, sink]:
                # Undirected edge: orient as w -> sink
                dag[sink, w] = 0
                # dag[w, sink] remains 1

        remaining.remove(sink)

    return dag


def enumerate_mec(
    adj: AdjacencyMatrix,
    max_dags: int = 1000,
) -> list[np.ndarray]:
    """Enumerate DAGs in the Markov equivalence class.

    Parameters
    ----------
    adj : AdjacencyMatrix
        A DAG adjacency matrix.
    max_dags : int
        Maximum number of DAGs to enumerate.

    Returns
    -------
    list[np.ndarray]
        List of adjacency matrices in the MEC.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    cpdag = to_cpdag(adj)

    # Find reversible edges (undirected in CPDAG)
    rev_edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if cpdag[i, j] and cpdag[j, i]:
                rev_edges.append((i, j))

    if not rev_edges:
        return [adj.copy()]

    # Enumerate all orientations of reversible edges
    results: list[np.ndarray] = []

    def _enumerate(idx: int, current: np.ndarray) -> None:
        if len(results) >= max_dags:
            return
        if idx == len(rev_edges):
            if is_dag(current):
                # Verify same MEC
                if is_mec_equivalent(adj, current):
                    results.append(current.copy())
            return

        i, j = rev_edges[idx]
        # Try i -> j
        candidate = current.copy()
        candidate[i, j] = 1
        candidate[j, i] = 0
        _enumerate(idx + 1, candidate)

        if len(results) >= max_dags:
            return

        # Try j -> i
        candidate = current.copy()
        candidate[i, j] = 0
        candidate[j, i] = 1
        _enumerate(idx + 1, candidate)

    # Start with compelled edges
    base = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if cpdag[i, j] and not cpdag[j, i]:
                base[i, j] = 1

    _enumerate(0, base)
    return results


def apply_meek_rules(pdag: np.ndarray) -> np.ndarray:
    """Apply Meek rules R1-R4 to orient edges in a PDAG.

    Parameters
    ----------
    pdag : np.ndarray
        Partially directed graph adjacency matrix.

    Returns
    -------
    np.ndarray
        PDAG with additional edges oriented by Meek rules.
    """
    pdag = np.asarray(pdag, dtype=np.int8).copy()
    n = pdag.shape[0]
    changed = True

    while changed:
        changed = False

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Only consider undirected edges i -- j
                if not (pdag[i, j] and pdag[j, i]):
                    continue

                # R1: k -> i -- j, k not adj j  =>  i -> j
                oriented = False
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if pdag[k, i] and not pdag[i, k] and not pdag[k, j] and not pdag[j, k]:
                        pdag[j, i] = 0  # Orient as i -> j
                        changed = True
                        oriented = True
                        break
                if oriented:
                    continue

                # R2: i -> k -> j (with i -- j undirected)  =>  i -> j
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (pdag[i, k] and not pdag[k, i] and
                            pdag[k, j] and not pdag[j, k]):
                        pdag[j, i] = 0
                        changed = True
                        oriented = True
                        break
                if oriented:
                    continue

                # R3: i -- k -> j, i -- l -> j, k not adj l  =>  i -> j
                undirected_parents_of_j = []
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (pdag[k, i] and pdag[i, k] and  # k -- i
                            pdag[k, j] and not pdag[j, k]):  # k -> j
                        undirected_parents_of_j.append(k)

                for ki in range(len(undirected_parents_of_j)):
                    for li in range(ki + 1, len(undirected_parents_of_j)):
                        k = undirected_parents_of_j[ki]
                        l = undirected_parents_of_j[li]
                        if not pdag[k, l] and not pdag[l, k]:
                            pdag[j, i] = 0
                            changed = True
                            oriented = True
                            break
                    if oriented:
                        break

    return pdag


def skeleton(adj: AdjacencyMatrix) -> np.ndarray:
    """Return the skeleton (undirected version) of a DAG.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.

    Returns
    -------
    np.ndarray
        Symmetric binary adjacency matrix.
    """
    adj = np.asarray(adj, dtype=np.int8)
    return _skeleton(adj)


def v_structures(adj: AdjacencyMatrix) -> list[tuple[NodeId, NodeId, NodeId]]:
    """Return all v-structures as (parent1, child, parent2) triples.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.

    Returns
    -------
    list[tuple[NodeId, NodeId, NodeId]]
        Each triple ``(i, j, k)`` represents i -> j <- k with i, k non-adjacent.
    """
    adj = np.asarray(adj, dtype=np.int8)
    vs = _v_structures(adj)
    return [(i, j, k) for i, j, k in sorted(vs)]
