"""
Vectorised numerical utilities for CausalCert.

Fast matrix operations for small matrices (avoiding NumPy dispatch overhead),
optimised permutation testing, combinatorial enumeration, fast topological
sort for dense small DAGs, batch matrix multiply for kernel computations,
and vectorised d-separation checks.  All routines are implemented in pure
NumPy with minimal overhead.
"""

from __future__ import annotations

import math
from collections import deque
from itertools import combinations
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray


# ====================================================================
# 1. Fast small-matrix operations
# ====================================================================


def fast_det_2x2(M: NDArray[np.float64]) -> float:
    """Determinant of a 2×2 matrix without NumPy dispatch overhead.

    Parameters
    ----------
    M : NDArray[np.float64]
        A ``(2, 2)`` array.

    Returns
    -------
    float
    """
    return float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])


def fast_inv_2x2(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of a 2×2 matrix.

    Parameters
    ----------
    M : NDArray[np.float64]
        A ``(2, 2)`` array.

    Returns
    -------
    NDArray[np.float64]
    """
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) < 1e-15:
        return np.linalg.inv(M)
    inv_det = 1.0 / det
    out = np.empty((2, 2), dtype=np.float64)
    out[0, 0] = M[1, 1] * inv_det
    out[0, 1] = -M[0, 1] * inv_det
    out[1, 0] = -M[1, 0] * inv_det
    out[1, 1] = M[0, 0] * inv_det
    return out


def fast_det_3x3(M: NDArray[np.float64]) -> float:
    """Determinant of a 3×3 matrix via Sarrus' rule.

    Parameters
    ----------
    M : NDArray[np.float64]
        A ``(3, 3)`` array.

    Returns
    -------
    float
    """
    return float(
        M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
        - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
        + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
    )


def fast_inv_3x3(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of a 3×3 matrix via cofactors.

    Parameters
    ----------
    M : NDArray[np.float64]
        A ``(3, 3)`` array.

    Returns
    -------
    NDArray[np.float64]
    """
    det = fast_det_3x3(M)
    if abs(det) < 1e-15:
        return np.linalg.inv(M)
    inv_det = 1.0 / det
    out = np.empty((3, 3), dtype=np.float64)
    out[0, 0] = (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) * inv_det
    out[0, 1] = (M[0, 2] * M[2, 1] - M[0, 1] * M[2, 2]) * inv_det
    out[0, 2] = (M[0, 1] * M[1, 2] - M[0, 2] * M[1, 1]) * inv_det
    out[1, 0] = (M[1, 2] * M[2, 0] - M[1, 0] * M[2, 2]) * inv_det
    out[1, 1] = (M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]) * inv_det
    out[1, 2] = (M[0, 2] * M[1, 0] - M[0, 0] * M[1, 2]) * inv_det
    out[2, 0] = (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]) * inv_det
    out[2, 1] = (M[0, 1] * M[2, 0] - M[0, 0] * M[2, 1]) * inv_det
    out[2, 2] = (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) * inv_det
    return out


def fast_inv(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fast matrix inverse dispatching to specialised routines for small sizes.

    Parameters
    ----------
    M : NDArray[np.float64]
        Square matrix.

    Returns
    -------
    NDArray[np.float64]
    """
    n = M.shape[0]
    if n == 1:
        return np.array([[1.0 / M[0, 0]]], dtype=np.float64)
    if n == 2:
        return fast_inv_2x2(M)
    if n == 3:
        return fast_inv_3x3(M)
    return np.linalg.inv(M)


def fast_solve(A: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve ``Ax = b`` using fast inverse for small systems.

    Parameters
    ----------
    A : NDArray[np.float64]
        Coefficient matrix ``(n, n)``.
    b : NDArray[np.float64]
        Right-hand side ``(n,)`` or ``(n, m)``.

    Returns
    -------
    NDArray[np.float64]
    """
    n = A.shape[0]
    if n <= 3:
        return fast_inv(A) @ b
    return np.linalg.solve(A, b)


def fast_matmul_small(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Matrix multiply for small matrices, avoiding overhead of np.matmul dispatch.

    For matrices ≤ 4×4, uses explicit loops that are faster than NumPy's
    general-purpose dispatch.

    Parameters
    ----------
    A : NDArray[np.float64]
        ``(m, k)`` matrix.
    B : NDArray[np.float64]
        ``(k, n)`` matrix.

    Returns
    -------
    NDArray[np.float64]
        ``(m, n)`` result.
    """
    m, k = A.shape
    _, n = B.shape
    if m <= 4 and n <= 4 and k <= 4:
        C = np.zeros((m, n), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                s = 0.0
                for p in range(k):
                    s += A[i, p] * B[p, j]
                C[i, j] = s
        return C
    return A @ B


# ====================================================================
# 2. Vectorised permutation testing
# ====================================================================


def permutation_test_vectorised(
    stat_observed: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_perms: int = 1000,
    seed: int = 42,
) -> float:
    """Vectorised permutation test for independence.

    Computes a correlation-based test statistic under *n_perms* random
    permutations of *y* simultaneously.

    Parameters
    ----------
    stat_observed : float
        Observed test statistic.
    x : NDArray[np.float64]
        First variable ``(n,)``.
    y : NDArray[np.float64]
        Second variable ``(n,)``.
    n_perms : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    float
        p-value (fraction of permutation statistics ≥ observed).
    """
    rng = np.random.RandomState(seed)
    n = len(y)

    # Generate all permutation indices at once: (n_perms, n)
    perm_indices = np.array([rng.permutation(n) for _ in range(n_perms)])
    # Permuted y values: (n_perms, n)
    y_perms = y[perm_indices]

    # Compute correlations vectorised
    x_centered = x - np.mean(x)
    y_perms_centered = y_perms - np.mean(y_perms, axis=1, keepdims=True)

    # Numerator: (n_perms,)
    numerator = np.sum(x_centered * y_perms_centered, axis=1)
    # Denominator
    denom_x = np.sqrt(np.sum(x_centered ** 2))
    denom_y = np.sqrt(np.sum(y_perms_centered ** 2, axis=1))
    denom = denom_x * denom_y
    denom = np.maximum(denom, 1e-15)

    perm_stats = np.abs(numerator / denom)
    p_value = float(np.mean(perm_stats >= abs(stat_observed)))
    return p_value


def permutation_test_conditional(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    n_perms: int = 500,
    seed: int = 42,
) -> float:
    """Conditional permutation test: residualise then permute.

    Regresses *x* and *y* on *z* via OLS, then runs a permutation test on
    the residuals.

    Parameters
    ----------
    x : NDArray[np.float64]
        First variable ``(n,)``.
    y : NDArray[np.float64]
        Second variable ``(n,)``.
    z : NDArray[np.float64]
        Conditioning variables ``(n, d)``.
    n_perms : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    float
        Conditional p-value.
    """
    z = np.atleast_2d(z)
    if z.shape[0] == 1 and z.shape[1] > 1:
        z = z.T

    n = len(x)
    # OLS residuals
    Z = np.column_stack([np.ones(n), z])
    try:
        beta_x = np.linalg.lstsq(Z, x, rcond=None)[0]
        beta_y = np.linalg.lstsq(Z, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 1.0  # cannot residualise

    res_x = x - Z @ beta_x
    res_y = y - Z @ beta_y

    # Observed partial correlation
    denom = np.sqrt(np.sum(res_x ** 2) * np.sum(res_y ** 2))
    if denom < 1e-15:
        return 1.0
    stat_obs = abs(np.sum(res_x * res_y) / denom)

    return permutation_test_vectorised(stat_obs, res_x, res_y, n_perms, seed)


# ====================================================================
# 3. Combinatorial enumeration
# ====================================================================


def k_subsets(n: int, k: int) -> Iterator[tuple[int, ...]]:
    """Enumerate all k-element subsets of ``{0, 1, ..., n-1}``.

    Parameters
    ----------
    n : int
        Universe size.
    k : int
        Subset size.

    Yields
    ------
    tuple[int, ...]
        Each subset as a sorted tuple.
    """
    yield from combinations(range(n), k)


def k_subsets_array(n: int, k: int) -> NDArray[np.int32]:
    """Return all k-subsets as a 2-D array.

    Parameters
    ----------
    n : int
        Universe size.
    k : int
        Subset size.

    Returns
    -------
    NDArray[np.int32]
        Array of shape ``(C(n,k), k)``.
    """
    if k > n or k < 0:
        return np.empty((0, max(k, 0)), dtype=np.int32)
    return np.array(list(combinations(range(n), k)), dtype=np.int32)


def edge_edit_candidates(
    adj: NDArray[np.int8],
) -> list[tuple[int, int, str]]:
    """Enumerate all single-edge edit candidates for a DAG.

    Parameters
    ----------
    adj : NDArray[np.int8]
        Binary adjacency matrix ``(n, n)``.

    Returns
    -------
    list[tuple[int, int, str]]
        Each entry is ``(source, target, edit_type)`` where *edit_type*
        is ``"add"``, ``"delete"``, or ``"reverse"``.
    """
    n = adj.shape[0]
    edits: list[tuple[int, int, str]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if adj[i, j]:
                edits.append((i, j, "delete"))
                edits.append((i, j, "reverse"))
            else:
                edits.append((i, j, "add"))
    return edits


def n_edit_candidates(adj: NDArray[np.int8]) -> int:
    """Count the number of single-edge edit candidates.

    Parameters
    ----------
    adj : NDArray[np.int8]
        Adjacency matrix.

    Returns
    -------
    int
    """
    n = adj.shape[0]
    n_edges = int(np.sum(adj))
    n_non_edges = n * (n - 1) - n_edges
    return n_edges * 2 + n_non_edges  # delete + reverse for edges, add for non-edges


# ====================================================================
# 4. Fast topological sort for dense small DAGs
# ====================================================================


def fast_topological_sort(adj: NDArray[np.int8]) -> NDArray[np.int32]:
    """Topological sort optimised for small dense DAGs.

    Avoids Python-level BFS overhead by using numpy operations on the
    adjacency matrix directly.

    Parameters
    ----------
    adj : NDArray[np.int8]
        Binary adjacency matrix ``(n, n)``.

    Returns
    -------
    NDArray[np.int32]
        Topological order.  Length < n if the graph has a cycle.
    """
    n = adj.shape[0]
    in_deg = np.sum(adj, axis=0, dtype=np.int32)
    order = np.empty(n, dtype=np.int32)
    remaining = np.ones(n, dtype=np.bool_)
    idx = 0

    for _ in range(n):
        # Find nodes with in-degree 0 among remaining
        candidates = np.where((in_deg == 0) & remaining)[0]
        if len(candidates) == 0:
            break
        # Pick the first (deterministic)
        v = int(candidates[0])
        order[idx] = v
        idx += 1
        remaining[v] = False
        # Decrement in-degrees of successors
        in_deg -= adj[v].astype(np.int32)

    return order[:idx]


def fast_topological_sort_all(adj: NDArray[np.int8]) -> NDArray[np.int32]:
    """Topological sort using Kahn's algorithm, fully vectorised where possible.

    Parameters
    ----------
    adj : NDArray[np.int8]
        Binary adjacency matrix.

    Returns
    -------
    NDArray[np.int32]
        Topological order.
    """
    n = adj.shape[0]
    in_deg = np.sum(adj, axis=0, dtype=np.int32).copy()
    queue = list(np.where(in_deg == 0)[0])
    order = np.empty(n, dtype=np.int32)
    idx = 0

    while queue:
        v = queue.pop(0)
        order[idx] = v
        idx += 1
        children = np.nonzero(adj[v])[0]
        for c in children:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(int(c))

    return order[:idx]


def is_dag_fast(adj: NDArray[np.int8]) -> bool:
    """Check acyclicity via fast topological sort.

    Parameters
    ----------
    adj : NDArray[np.int8]
        Binary adjacency matrix.

    Returns
    -------
    bool
    """
    return len(fast_topological_sort(adj)) == adj.shape[0]


# ====================================================================
# 5. Batch matrix multiply for kernel computations
# ====================================================================


def batch_kernel_multiply(
    K_list: Sequence[NDArray[np.float64]],
    v: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Multiply a list of kernel matrices by a vector simultaneously.

    Parameters
    ----------
    K_list : Sequence[NDArray[np.float64]]
        List of kernel matrices, each ``(n, n)``.
    v : NDArray[np.float64]
        Vector ``(n,)``.

    Returns
    -------
    NDArray[np.float64]
        Stacked results ``(len(K_list), n)``.
    """
    if not K_list:
        return np.empty((0, len(v)), dtype=np.float64)
    # Stack into 3D array and use einsum
    K_stack = np.stack(K_list, axis=0)
    return np.einsum("bij,j->bi", K_stack, v)


def batch_trace(K_list: Sequence[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Compute traces of multiple matrices simultaneously.

    Parameters
    ----------
    K_list : Sequence[NDArray[np.float64]]
        List of square matrices.

    Returns
    -------
    NDArray[np.float64]
        Traces ``(len(K_list),)``.
    """
    if not K_list:
        return np.array([], dtype=np.float64)
    K_stack = np.stack(K_list, axis=0)
    return np.einsum("bii->b", K_stack)


def batch_hsic_statistic(
    K_x_list: Sequence[NDArray[np.float64]],
    K_y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute HSIC statistics for multiple kernel matrices against a fixed target.

    HSIC(X, Y) = (1/n²) tr(K_x H K_y H) where H = I - 11ᵀ/n.

    Parameters
    ----------
    K_x_list : Sequence[NDArray[np.float64]]
        List of kernel matrices for X under different conditioning.
    K_y : NDArray[np.float64]
        Kernel matrix for Y ``(n, n)``.

    Returns
    -------
    NDArray[np.float64]
        HSIC statistics ``(len(K_x_list),)``.
    """
    n = K_y.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    HKyH = H @ K_y @ H

    stats = np.empty(len(K_x_list), dtype=np.float64)
    for i, K_x in enumerate(K_x_list):
        stats[i] = np.sum(K_x * HKyH) / (n * n)

    return stats


def batch_centered_kernel_alignment(
    K_list: Sequence[NDArray[np.float64]],
    K_ref: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Centered kernel alignment (CKA) of each kernel against a reference.

    CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

    Parameters
    ----------
    K_list : Sequence[NDArray[np.float64]]
        List of kernel matrices.
    K_ref : NDArray[np.float64]
        Reference kernel matrix.

    Returns
    -------
    NDArray[np.float64]
        CKA values ``(len(K_list),)``.
    """
    n = K_ref.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    HKrH = H @ K_ref @ H
    hsic_rr = np.sum(K_ref * HKrH) / (n * n)

    cka = np.empty(len(K_list), dtype=np.float64)
    for i, K in enumerate(K_list):
        HKH = H @ K @ H
        hsic_kr = np.sum(K * HKrH) / (n * n)
        hsic_kk = np.sum(K * HKH) / (n * n)
        denom = np.sqrt(max(hsic_kk * hsic_rr, 1e-30))
        cka[i] = hsic_kr / denom

    return cka


# ====================================================================
# 6. Vectorised d-separation check
# ====================================================================


def vectorised_dsep_check(
    adj: NDArray[np.int8],
    pairs: NDArray[np.int32],
    conditioning: NDArray[np.int32],
) -> NDArray[np.bool_]:
    """Check d-separation for multiple (x, y) pairs under a shared conditioning set.

    Uses the Bayes-Ball algorithm per unique source, then batches target lookups.

    Parameters
    ----------
    adj : NDArray[np.int8]
        Binary adjacency matrix ``(n, n)``.
    pairs : NDArray[np.int32]
        Array of shape ``(m, 2)`` with columns ``[x, y]``.
    conditioning : NDArray[np.int32]
        Array of conditioning node indices ``(k,)``.

    Returns
    -------
    NDArray[np.bool_]
        Boolean array of length *m*; ``True`` where pair is d-separated.
    """
    n = adj.shape[0]
    cond_set = set(int(c) for c in conditioning)

    # Precompute parents/children
    children: list[list[int]] = [list(np.nonzero(adj[v])[0]) for v in range(n)]
    parents: list[list[int]] = [list(np.nonzero(adj[:, v])[0]) for v in range(n)]

    # Ancestors of conditioning
    anc_cond: set[int] = set(cond_set)
    q: deque[int] = deque(cond_set)
    while q:
        v = q.popleft()
        for p in parents[v]:
            if p not in anc_cond:
                anc_cond.add(p)
                q.append(p)

    # Cache reachable sets per source
    reachable_cache: dict[int, set[int]] = {}

    def bayes_ball(source: int) -> set[int]:
        if source in reachable_cache:
            return reachable_cache[source]

        visited_up: set[int] = set()
        visited_down: set[int] = set()
        reachable: set[int] = set()
        queue: deque[tuple[int, bool]] = deque()
        queue.append((source, True))
        queue.append((source, False))

        while queue:
            node, going_up = queue.popleft()
            if going_up:
                if node in visited_up:
                    continue
                visited_up.add(node)
                in_cond = node in cond_set
                if not in_cond:
                    reachable.add(node)
                    for p in parents[node]:
                        queue.append((p, True))
                    for c in children[node]:
                        queue.append((c, False))
                if in_cond or node in anc_cond:
                    for p in parents[node]:
                        queue.append((p, True))
            else:
                if node in visited_down:
                    continue
                visited_down.add(node)
                in_cond = node in cond_set
                if not in_cond:
                    reachable.add(node)
                    for c in children[node]:
                        queue.append((c, False))
                if not in_cond:
                    for p in parents[node]:
                        queue.append((p, True))

        reachable_cache[source] = reachable
        return reachable

    m = pairs.shape[0]
    results = np.empty(m, dtype=np.bool_)
    for i in range(m):
        x, y = int(pairs[i, 0]), int(pairs[i, 1])
        reach = bayes_ball(x)
        results[i] = y not in reach

    return results


def dsep_matrix(
    adj: NDArray[np.int8],
    conditioning: NDArray[np.int32],
) -> NDArray[np.bool_]:
    """Compute the full d-separation matrix for all node pairs.

    Parameters
    ----------
    adj : NDArray[np.int8]
        Adjacency matrix ``(n, n)``.
    conditioning : NDArray[np.int32]
        Conditioning set indices.

    Returns
    -------
    NDArray[np.bool_]
        ``(n, n)`` boolean matrix; ``True`` at ``(i, j)`` if *i* and *j*
        are d-separated given the conditioning set.
    """
    n = adj.shape[0]
    pairs = np.array(
        [(i, j) for i in range(n) for j in range(n)],
        dtype=np.int32,
    ).reshape(-1, 2)
    flat = vectorised_dsep_check(adj, pairs, conditioning)
    return flat.reshape(n, n)
