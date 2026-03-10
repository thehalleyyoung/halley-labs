"""
Single-edit BFS over perturbations.

Performs breadth-first search over the space of single-edge edits,
checking the conclusion predicate at each step.  Used as a baseline
and as a warm-start heuristic for the ILP solver.

The main entry point :func:`single_edit_bfs` searches edit distances
1, 2, …, max_k until a falsifying perturbation is found (or the
budget is exhausted).  Pruning and caching strategies keep the search
tractable for moderate DAG sizes.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from itertools import combinations
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from causalcert.types import (
    AdjacencyMatrix,
    ConclusionPredicate,
    EditType,
    NodeId,
    NodeSet,
    RobustnessRadius,
    SolverStrategy,
    StructuralEdit,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_edit(adj: np.ndarray, edit: StructuralEdit) -> np.ndarray:
    """Apply edit to adjacency matrix copy."""
    result = adj.copy()
    if edit.edit_type == EditType.ADD:
        result[edit.source, edit.target] = 1
    elif edit.edit_type == EditType.DELETE:
        result[edit.source, edit.target] = 0
    elif edit.edit_type == EditType.REVERSE:
        result[edit.source, edit.target] = 0
        result[edit.target, edit.source] = 1
    return result


def _is_dag(adj: np.ndarray) -> bool:
    """Quick Kahn-based acyclicity test."""
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = [i for i in range(n) if in_deg[i] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for c in np.nonzero(adj[node])[0]:
            in_deg[int(c)] -= 1
            if in_deg[int(c)] == 0:
                queue.append(int(c))
    return visited == n


def _adj_key(adj: np.ndarray) -> bytes:
    """Hash key for an adjacency matrix."""
    return adj.tobytes()


def _ancestors_inclusive(adj: np.ndarray, targets: set[int]) -> set[int]:
    """Ancestors of targets (inclusive)."""
    visited: set[int] = set()
    queue = deque(targets)
    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        for p in np.nonzero(adj[:, v])[0]:
            p = int(p)
            if p not in visited:
                queue.append(p)
    return visited


def _all_single_edits_in_set(
    adj: np.ndarray,
    relevant_nodes: set[int] | None = None,
) -> list[StructuralEdit]:
    """Generate all valid single-edge edits within relevant_nodes."""
    n = adj.shape[0]
    if relevant_nodes is None:
        relevant_nodes = set(range(n))

    edits: list[StructuralEdit] = []
    for i in sorted(relevant_nodes):
        for j in sorted(relevant_nodes):
            if i == j:
                continue
            if adj[i, j]:
                edits.append(StructuralEdit(EditType.DELETE, i, j))
                trial = adj.copy()
                trial[i, j] = 0
                trial[j, i] = 1
                if _is_dag(trial):
                    edits.append(StructuralEdit(EditType.REVERSE, i, j))
            else:
                trial = adj.copy()
                trial[i, j] = 1
                if _is_dag(trial):
                    edits.append(StructuralEdit(EditType.ADD, i, j))
    return edits


# ---------------------------------------------------------------------------
# BFS result cache
# ---------------------------------------------------------------------------


class _PredicateCache:
    """LRU cache for predicate evaluations keyed by adjacency matrix."""

    def __init__(self, max_size: int = 10000) -> None:
        self._cache: dict[bytes, bool] = {}
        self._max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, adj: np.ndarray) -> bool | None:
        key = _adj_key(adj)
        result = self._cache.get(key)
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
        return result

    def put(self, adj: np.ndarray, value: bool) -> None:
        if len(self._cache) >= self._max_size:
            # Evict oldest entries (approximate FIFO)
            keys = list(self._cache.keys())
            for k in keys[: self._max_size // 4]:
                del self._cache[k]
        self._cache[_adj_key(adj)] = value

    @property
    def size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Main BFS function
# ---------------------------------------------------------------------------


def single_edit_bfs(
    adj: AdjacencyMatrix,
    predicate: ConclusionPredicate,
    data: Any,
    treatment: NodeId,
    outcome: NodeId,
    max_k: int = 5,
    restrict_to_ancestral: bool = True,
    early_termination: bool = True,
    cache_size: int = 10000,
) -> RobustnessRadius:
    """BFS over k-neighbourhoods to find the first overturn.

    Enumerates all DAGs at edit distance 1, then 2, etc., evaluating the
    predicate on each.  Returns as soon as a falsifying perturbation is
    found.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Original DAG adjacency matrix.
    predicate : ConclusionPredicate
        Conclusion predicate.
    data : Any
        Observational data.
    treatment, outcome : NodeId
        Treatment and outcome nodes.
    max_k : int
        Maximum edit distance to search.
    restrict_to_ancestral : bool
        If True, only consider edits within ancestral set of {X, Y}.
    early_termination : bool
        If True, stop as soon as first violation is found.
    cache_size : int
        Maximum predicate cache entries.

    Returns
    -------
    RobustnessRadius
        Result with ``upper_bound`` equal to the first falsifying distance.
    """
    t_start = time.time()
    adj_arr = np.asarray(adj, dtype=np.int8)
    n = adj_arr.shape[0]

    # Determine relevant node set
    if restrict_to_ancestral:
        relevant_nodes = _ancestors_inclusive(adj_arr, {treatment, outcome})
    else:
        relevant_nodes = set(range(n))

    # Verify predicate holds on original DAG
    orig_holds = predicate(adj_arr, data, treatment=treatment, outcome=outcome)
    if not orig_holds:
        # Predicate already fails on original DAG — radius is 0
        return RobustnessRadius(
            lower_bound=0,
            upper_bound=0,
            witness_edits=(),
            solver_strategy=SolverStrategy.AUTO,
            solver_time_s=time.time() - t_start,
            gap=0.0,
            certified=True,
        )

    cache = _PredicateCache(max_size=cache_size)
    cache.put(adj_arr, True)

    # BFS state: (adjacency_matrix, edit_sequence, depth)
    seen: set[bytes] = {_adj_key(adj_arr)}
    current_level: list[tuple[np.ndarray, tuple[StructuralEdit, ...]]] = [
        (adj_arr.copy(), ())
    ]

    best_witness: tuple[StructuralEdit, ...] | None = None
    best_distance = max_k + 1
    n_evaluated = 0

    for k in range(1, max_k + 1):
        logger.info("BFS: exploring edit distance %d", k)
        next_level: list[tuple[np.ndarray, tuple[StructuralEdit, ...]]] = []

        for current_adj, edit_seq in current_level:
            # Generate all single-edge edits from current state
            single_edits = _all_single_edits_in_set(current_adj, relevant_nodes)

            for edit in single_edits:
                new_adj = _apply_edit(current_adj, edit)

                key = _adj_key(new_adj)
                if key in seen:
                    continue
                seen.add(key)

                if not _is_dag(new_adj):
                    continue

                new_edits = edit_seq + (edit,)

                # Check predicate (with cache)
                cached = cache.get(new_adj)
                if cached is not None:
                    holds = cached
                else:
                    holds = predicate(
                        new_adj, data, treatment=treatment, outcome=outcome
                    )
                    cache.put(new_adj, holds)
                    n_evaluated += 1

                if not holds:
                    # Found a violation!
                    if k < best_distance:
                        best_distance = k
                        best_witness = new_edits
                        logger.info(
                            "BFS: violation found at distance %d (%d edits)",
                            k, len(new_edits),
                        )
                    if early_termination:
                        elapsed = time.time() - t_start
                        return RobustnessRadius(
                            lower_bound=k,
                            upper_bound=k,
                            witness_edits=new_edits,
                            solver_strategy=SolverStrategy.AUTO,
                            solver_time_s=elapsed,
                            gap=0.0,
                            certified=True,
                        )

                if k < max_k:
                    next_level.append((new_adj, new_edits))

        current_level = next_level

        if not current_level and best_witness is None:
            logger.info("BFS: no more perturbations at distance %d", k)

    elapsed = time.time() - t_start

    if best_witness is not None:
        return RobustnessRadius(
            lower_bound=best_distance,
            upper_bound=best_distance,
            witness_edits=best_witness,
            solver_strategy=SolverStrategy.AUTO,
            solver_time_s=elapsed,
            gap=0.0,
            certified=True,
        )

    # No violation found within budget
    return RobustnessRadius(
        lower_bound=max_k,
        upper_bound=max_k + 1,
        witness_edits=(),
        solver_strategy=SolverStrategy.AUTO,
        solver_time_s=elapsed,
        gap=1.0 / (max_k + 1),
        certified=False,
    )


# ---------------------------------------------------------------------------
# Multi-edit extensions
# ---------------------------------------------------------------------------


def multi_edit_bfs(
    adj: AdjacencyMatrix,
    predicate: ConclusionPredicate,
    data: Any,
    treatment: NodeId,
    outcome: NodeId,
    max_k: int = 3,
    restrict_to_ancestral: bool = True,
    cache_size: int = 10000,
) -> RobustnessRadius:
    """BFS with combined multi-edit enumeration for k ≤ 3.

    For very small max_k, directly enumerates combinations of edits
    rather than level-by-level BFS, which can be more memory-efficient.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Original DAG adjacency matrix.
    predicate : ConclusionPredicate
        Conclusion predicate.
    data : Any
        Observational data.
    treatment, outcome : NodeId
        Treatment and outcome nodes.
    max_k : int
        Maximum number of simultaneous edits (≤ 3 recommended).
    restrict_to_ancestral : bool
        If True, only consider edits within ancestral set.
    cache_size : int
        Maximum predicate cache entries.

    Returns
    -------
    RobustnessRadius
    """
    t_start = time.time()
    adj_arr = np.asarray(adj, dtype=np.int8)

    if restrict_to_ancestral:
        relevant = _ancestors_inclusive(adj_arr, {treatment, outcome})
    else:
        relevant = set(range(adj_arr.shape[0]))

    orig_holds = predicate(adj_arr, data, treatment=treatment, outcome=outcome)
    if not orig_holds:
        return RobustnessRadius(
            lower_bound=0, upper_bound=0, witness_edits=(),
            solver_strategy=SolverStrategy.AUTO,
            solver_time_s=time.time() - t_start,
            gap=0.0, certified=True,
        )

    all_edits = _all_single_edits_in_set(adj_arr, relevant)
    cache = _PredicateCache(max_size=cache_size)
    n_evaluated = 0

    for k in range(1, max_k + 1):
        logger.info("Multi-edit BFS: checking %d-edit combinations", k)

        for combo in combinations(range(len(all_edits)), k):
            edits = tuple(all_edits[i] for i in combo)

            # Apply all edits
            new_adj = adj_arr.copy()
            for edit in edits:
                new_adj = _apply_edit(new_adj, edit)

            if not _is_dag(new_adj):
                continue

            cached = cache.get(new_adj)
            if cached is not None:
                holds = cached
            else:
                holds = predicate(
                    new_adj, data, treatment=treatment, outcome=outcome
                )
                cache.put(new_adj, holds)
                n_evaluated += 1

            if not holds:
                elapsed = time.time() - t_start
                return RobustnessRadius(
                    lower_bound=k, upper_bound=k,
                    witness_edits=edits,
                    solver_strategy=SolverStrategy.AUTO,
                    solver_time_s=elapsed,
                    gap=0.0, certified=True,
                )

    elapsed = time.time() - t_start
    return RobustnessRadius(
        lower_bound=max_k, upper_bound=max_k + 1,
        witness_edits=(),
        solver_strategy=SolverStrategy.AUTO,
        solver_time_s=elapsed,
        gap=1.0 / (max_k + 1), certified=False,
    )


# ---------------------------------------------------------------------------
# Pruned BFS with fragility-guided ordering
# ---------------------------------------------------------------------------


def fragility_guided_bfs(
    adj: AdjacencyMatrix,
    predicate: ConclusionPredicate,
    data: Any,
    treatment: NodeId,
    outcome: NodeId,
    fragility_scores: Sequence[Any] | None = None,
    max_k: int = 5,
    max_evaluations: int = 50000,
    restrict_to_ancestral: bool = True,
) -> RobustnessRadius:
    """BFS guided by fragility scores for better pruning.

    Explores higher-fragility edits first at each level, which tends to
    find violations faster.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Original DAG.
    predicate : ConclusionPredicate
        Conclusion predicate.
    data : Any
        Observational data.
    treatment, outcome : NodeId
        Treatment and outcome.
    fragility_scores : Sequence[FragilityScore] | None
        Pre-computed fragility scores for edit ordering.
    max_k : int
        Maximum edit distance.
    max_evaluations : int
        Maximum number of predicate evaluations.
    restrict_to_ancestral : bool
        Restrict to ancestral set.

    Returns
    -------
    RobustnessRadius
    """
    t_start = time.time()
    adj_arr = np.asarray(adj, dtype=np.int8)

    orig_holds = predicate(adj_arr, data, treatment=treatment, outcome=outcome)
    if not orig_holds:
        return RobustnessRadius(
            lower_bound=0, upper_bound=0, witness_edits=(),
            solver_strategy=SolverStrategy.AUTO,
            solver_time_s=time.time() - t_start,
            gap=0.0, certified=True,
        )

    if restrict_to_ancestral:
        relevant = _ancestors_inclusive(adj_arr, {treatment, outcome})
    else:
        relevant = set(range(adj_arr.shape[0]))

    # Build edit ordering from fragility scores
    edge_priority: dict[tuple[int, int], float] = {}
    if fragility_scores is not None:
        for fs in fragility_scores:
            edge_priority[fs.edge] = fs.total_score

    cache = _PredicateCache()
    cache.put(adj_arr, True)
    seen: set[bytes] = {_adj_key(adj_arr)}
    n_evaluated = 0

    current_level: list[tuple[np.ndarray, tuple[StructuralEdit, ...]]] = [
        (adj_arr.copy(), ())
    ]

    for k in range(1, max_k + 1):
        if n_evaluated >= max_evaluations:
            break

        next_level: list[tuple[np.ndarray, tuple[StructuralEdit, ...]]] = []

        for current_adj, edit_seq in current_level:
            if n_evaluated >= max_evaluations:
                break

            edits = _all_single_edits_in_set(current_adj, relevant)

            # Sort by fragility (highest first)
            edits.sort(
                key=lambda e: edge_priority.get((e.source, e.target), 0.0),
                reverse=True,
            )

            for edit in edits:
                if n_evaluated >= max_evaluations:
                    break

                new_adj = _apply_edit(current_adj, edit)
                key = _adj_key(new_adj)
                if key in seen:
                    continue
                seen.add(key)

                if not _is_dag(new_adj):
                    continue

                new_edits = edit_seq + (edit,)

                cached = cache.get(new_adj)
                if cached is not None:
                    holds = cached
                else:
                    holds = predicate(
                        new_adj, data, treatment=treatment, outcome=outcome
                    )
                    cache.put(new_adj, holds)
                    n_evaluated += 1

                if not holds:
                    elapsed = time.time() - t_start
                    return RobustnessRadius(
                        lower_bound=k, upper_bound=k,
                        witness_edits=new_edits,
                        solver_strategy=SolverStrategy.AUTO,
                        solver_time_s=elapsed,
                        gap=0.0, certified=True,
                    )

                if k < max_k:
                    next_level.append((new_adj, new_edits))

        current_level = next_level

    elapsed = time.time() - t_start
    certified = n_evaluated < max_evaluations
    return RobustnessRadius(
        lower_bound=max_k if certified else 0,
        upper_bound=max_k + 1,
        witness_edits=(),
        solver_strategy=SolverStrategy.AUTO,
        solver_time_s=elapsed,
        gap=1.0 / (max_k + 1) if not certified else 1.0 / (max_k + 1),
        certified=False,
    )


# ---------------------------------------------------------------------------
# BFS statistics
# ---------------------------------------------------------------------------


def bfs_statistics(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    max_k: int = 3,
    restrict_to_ancestral: bool = True,
) -> dict[str, Any]:
    """Compute statistics about the BFS search space.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment, outcome : NodeId
        Treatment and outcome.
    max_k : int
        Maximum edit distance.
    restrict_to_ancestral : bool
        Restrict to ancestral set.

    Returns
    -------
    dict
        'n_single_edits': number of valid single-edit perturbations
        'ancestral_set_size': size of ancestral set
        'estimated_k_neighbourhood': estimated sizes at each distance
    """
    adj_arr = np.asarray(adj, dtype=np.int8)
    n = adj_arr.shape[0]

    if restrict_to_ancestral:
        relevant = _ancestors_inclusive(adj_arr, {treatment, outcome})
    else:
        relevant = set(range(n))

    single_edits = _all_single_edits_in_set(adj_arr, relevant)
    n_single = len(single_edits)

    # Estimate neighbourhood sizes (branching factor heuristic)
    estimated_sizes = [1]
    for k in range(1, max_k + 1):
        est = min(estimated_sizes[-1] * n_single, 10 ** 7)
        estimated_sizes.append(int(est))

    return {
        "n_single_edits": n_single,
        "ancestral_set_size": len(relevant),
        "n_nodes": n,
        "estimated_k_neighbourhood": estimated_sizes,
    }
