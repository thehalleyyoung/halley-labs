"""Online DAG alignment with warm starts.

Provides incremental DAG alignment that reuses previous solutions
as warm starts, reducing computation when graphs change slightly.
Includes an alignment cache with LRU eviction for repeated queries.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class WarmStartState:
    """Persistent state for warm-starting alignment iterations."""

    previous_mapping: NDArray
    previous_cost: float
    iteration_count: int


# ---------------------------------------------------------------------------
# AlignmentCache
# ---------------------------------------------------------------------------

class AlignmentCache:
    """LRU cache for DAG alignment results.

    Parameters
    ----------
    max_size : int
        Maximum number of cached alignments.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached alignment for *key*, or ``None``."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, alignment: Any) -> None:
        """Store *alignment* under *key*, evicting LRU if needed."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = alignment
            return
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = alignment

    def invalidate(self, context_id: str) -> int:
        """Invalidate all entries whose key contains *context_id*.

        Returns the number of evicted entries.
        """
        to_remove = [k for k in self._cache if context_id in k]
        for k in to_remove:
            del self._cache[k]
        return len(to_remove)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"AlignmentCache(size={len(self._cache)}, "
            f"max={self._max_size}, hit_rate={self.hit_rate:.2%})"
        )


# ---------------------------------------------------------------------------
# OnlineAligner
# ---------------------------------------------------------------------------

class OnlineAligner:
    """Online DAG aligner with warm-start acceleration.

    Aligns DAGs by finding a node permutation that minimises the
    structural Hamming distance (or a weighted cost combining
    structural and parametric differences).  Previous solutions seed
    the optimisation (warm start) so that small perturbations converge
    quickly.

    Parameters
    ----------
    base_aligner : object or None
        Optional external aligner instance (unused if *None*).
    warm_start : bool
        Whether to reuse previous solutions as warm starts.
    cache_size : int
        Maximum entries in the alignment cache.
    """

    def __init__(
        self,
        base_aligner: object = None,
        warm_start: bool = True,
        cache_size: int = 1000,
    ) -> None:
        self._base_aligner = base_aligner
        self._warm_start = warm_start
        self._state: Optional[WarmStartState] = None
        self._cache = AlignmentCache(max_size=cache_size)
        self._total_alignments = 0
        self._total_iterations = 0

    # -- public API --------------------------------------------------------

    def align(
        self,
        dag1: NDArray,
        dag2: NDArray,
        warm_state: Optional[WarmStartState] = None,
    ) -> Tuple[NDArray, WarmStartState]:
        """Align *dag1* to *dag2*, optionally using *warm_state*.

        Parameters
        ----------
        dag1, dag2 : array of shape ``(n, n)``
            Binary adjacency matrices of the two DAGs.
        warm_state : WarmStartState or None
            Previous alignment state for warm starting.

        Returns
        -------
        mapping : array of shape ``(n,)``
            Permutation such that ``dag1[mapping][:, mapping] ≈ dag2``.
        state : WarmStartState
            Updated state for future warm starts.
        """
        dag1 = np.asarray(dag1, dtype=np.float64)
        dag2 = np.asarray(dag2, dtype=np.float64)
        n1, n2 = dag1.shape[0], dag2.shape[0]
        n = max(n1, n2)

        # Pad to equal size if necessary
        if n1 != n2:
            dag1 = self._pad(dag1, n)
            dag2 = self._pad(dag2, n)

        # Check cache
        cache_key = self._alignment_cache_key(dag1, dag2)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Build cost matrix
        cost = self._build_cost_matrix(dag1, dag2)

        # Warm start: perturb cost to favour previous assignment
        ws = warm_state if warm_state is not None else self._state
        if self._warm_start and ws is not None:
            cost = self._apply_warm_start(cost, ws)

        # Solve assignment via Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = np.full(n, -1, dtype=np.int64)
        mapping[row_ind] = col_ind

        total_cost = float(cost[row_ind, col_ind].sum())

        # Refine with local search
        mapping, total_cost, iters = self._refine_alignment(
            mapping, dag1, dag2, max_iter=50
        )

        state = WarmStartState(
            previous_mapping=mapping.copy(),
            previous_cost=total_cost,
            iteration_count=(
                (ws.iteration_count if ws else 0) + iters
            ),
        )
        self._state = state
        self._total_alignments += 1
        self._total_iterations += iters

        result = (mapping[:n1], state)
        self._cache.put(cache_key, result)
        return result

    def save_state(self) -> WarmStartState:
        """Snapshot the current warm-start state."""
        if self._state is None:
            return WarmStartState(
                previous_mapping=np.array([], dtype=np.int64),
                previous_cost=float("inf"),
                iteration_count=0,
            )
        return WarmStartState(
            previous_mapping=self._state.previous_mapping.copy(),
            previous_cost=self._state.previous_cost,
            iteration_count=self._state.iteration_count,
        )

    def load_state(self, state: WarmStartState) -> None:
        """Restore a previously saved warm-start state."""
        self._state = WarmStartState(
            previous_mapping=state.previous_mapping.copy(),
            previous_cost=state.previous_cost,
            iteration_count=state.iteration_count,
        )

    def incremental_update(
        self,
        old_dag: NDArray,
        new_dag: NDArray,
        changed_edges: Set[Tuple[int, int]],
    ) -> NDArray:
        """Re-align after only *changed_edges* have been modified.

        This is faster than a full ``align`` when few edges changed,
        because it restricts the search to nodes incident to the
        changed edges.
        """
        old_dag = np.asarray(old_dag, dtype=np.float64)
        new_dag = np.asarray(new_dag, dtype=np.float64)
        n = old_dag.shape[0]

        if self._state is None or len(self._state.previous_mapping) != n:
            mapping, _ = self.align(old_dag, new_dag)
            return mapping

        mapping = self._state.previous_mapping.copy()

        # Identify affected nodes
        affected: Set[int] = set()
        for i, j in changed_edges:
            if i < n:
                affected.add(i)
            if j < n:
                affected.add(j)

        if not affected:
            return mapping

        # Local re-optimisation for affected nodes only
        affected_list = sorted(affected)
        k = len(affected_list)
        if k <= 1:
            return mapping

        # Build local cost sub-matrix
        local_cost = np.zeros((k, k), dtype=np.float64)
        for li, ni in enumerate(affected_list):
            for lj, nj in enumerate(affected_list):
                row_i = new_dag[ni, :]
                col_j = old_dag[nj, :]
                row_j = new_dag[:, ni]
                col_i = old_dag[:, nj]
                local_cost[li, lj] = (
                    np.sum(np.abs(row_i - col_j))
                    + np.sum(np.abs(row_j - col_i))
                )

        lr, lc = linear_sum_assignment(local_cost)
        current_targets = mapping[affected_list]
        new_targets = current_targets.copy()
        for li in range(k):
            new_targets[lr[li]] = current_targets[lc[li]]
        for li, ni in enumerate(affected_list):
            mapping[ni] = new_targets[li]

        # Update state
        total_cost = self._alignment_cost(mapping, old_dag, new_dag)
        self._state = WarmStartState(
            previous_mapping=mapping.copy(),
            previous_cost=total_cost,
            iteration_count=self._state.iteration_count + 1,
        )
        return mapping

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _pad(adj: NDArray, n: int) -> NDArray:
        """Zero-pad adjacency matrix to ``(n, n)``."""
        m = adj.shape[0]
        if m >= n:
            return adj[:n, :n]
        padded = np.zeros((n, n), dtype=adj.dtype)
        padded[:m, :m] = adj
        return padded

    @staticmethod
    def _build_cost_matrix(dag1: NDArray, dag2: NDArray) -> NDArray:
        """Build the ``n × n`` assignment cost matrix.

        Cost(i, j) measures how dissimilar node *i* in dag1 is to
        node *j* in dag2 based on parent/child adjacency profiles.
        """
        n = dag1.shape[0]
        cost = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                # Row dissimilarity (outgoing edges)
                cost[i, j] += np.sum(np.abs(dag1[i, :] - dag2[j, :]))
                # Column dissimilarity (incoming edges)
                cost[i, j] += np.sum(np.abs(dag1[:, i] - dag2[:, j]))
        return cost

    @staticmethod
    def _apply_warm_start(
        cost: NDArray, ws: WarmStartState
    ) -> NDArray:
        """Perturb cost matrix to favour the previous assignment."""
        n = cost.shape[0]
        m = len(ws.previous_mapping)
        bonus = 0.1 * np.mean(cost)
        adjusted = cost.copy()
        for i in range(min(n, m)):
            j = ws.previous_mapping[i]
            if 0 <= j < n:
                adjusted[i, j] -= bonus
        return adjusted

    @staticmethod
    def _refine_alignment(
        mapping: NDArray,
        dag1: NDArray,
        dag2: NDArray,
        max_iter: int = 50,
    ) -> Tuple[NDArray, float, int]:
        """Local 2-opt refinement of *mapping*.

        Iteratively swaps pairs of assignments when doing so reduces
        the total alignment cost.
        """
        n = len(mapping)
        best_cost = OnlineAligner._alignment_cost(mapping, dag1, dag2)
        improved = True
        iters = 0

        while improved and iters < max_iter:
            improved = False
            iters += 1
            for i in range(n):
                for j in range(i + 1, n):
                    trial = mapping.copy()
                    trial[i], trial[j] = mapping[j], mapping[i]
                    trial_cost = OnlineAligner._alignment_cost(
                        trial, dag1, dag2
                    )
                    if trial_cost < best_cost - 1e-12:
                        mapping = trial
                        best_cost = trial_cost
                        improved = True
        return mapping, best_cost, iters

    @staticmethod
    def _alignment_cost(
        mapping: NDArray, dag1: NDArray, dag2: NDArray
    ) -> float:
        """Total alignment cost under the given *mapping*."""
        n = dag1.shape[0]
        perm = np.clip(mapping, 0, n - 1).astype(int)
        permuted = dag1[np.ix_(perm, perm)]
        return float(np.sum(np.abs(permuted - dag2)))

    @staticmethod
    def _alignment_cache_key(dag1: NDArray, dag2: NDArray) -> str:
        """Create a cache key from two adjacency matrices."""
        h = hashlib.md5()
        h.update(dag1.tobytes())
        h.update(dag2.tobytes())
        return h.hexdigest()

    # -- warm start matching -----------------------------------------------

    @staticmethod
    def _warm_start_matching(
        new_dag: NDArray,
        ref_dag: NDArray,
        prev_matching: NDArray,
    ) -> NDArray:
        """Initialise matching from a previous alignment.

        The previous matching is used as-is when node counts haven't
        changed; otherwise it is extended with an identity assignment
        for new nodes.
        """
        n_new = new_dag.shape[0]
        n_prev = len(prev_matching)
        matching = np.arange(n_new, dtype=np.int64)
        k = min(n_new, n_prev)
        matching[:k] = prev_matching[:k]
        # Clamp to valid range
        matching = np.clip(matching, 0, n_new - 1)
        return matching

    def __repr__(self) -> str:
        return (
            f"OnlineAligner(warm_start={self._warm_start}, "
            f"alignments={self._total_alignments}, "
            f"cache={self._cache})"
        )
