"""Skeleton-based restriction of the mutable edge set.

Provides:
  - SkeletonRestrictor: restrict CI testing / mutation to skeleton edges
  - Skeleton learning via PC with low alpha
  - Incremental skeleton updates after single-edge mutations
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from causal_qd.types import AdjacencyMatrix, DataMatrix, EdgeMask

if TYPE_CHECKING:
    from causal_qd.ci_tests.ci_base import CITest
    from causal_qd.core.dag import DAG

logger = logging.getLogger(__name__)


class SkeletonRestrictor:
    """Restrict the search space to edges supported by CI tests.

    The restrictor builds an undirected skeleton of plausible edges
    using pairwise and conditional independence tests, then provides
    a boolean mask that mutation/crossover operators can use to avoid
    wasting effort on clearly irrelevant edges.

    Supports:
    - Marginal (unconditional) filtering
    - PC-style skeleton with increasing conditioning set depth
    - Incremental updates when a single edge changes
    - Cached skeleton reuse across generations

    Parameters
    ----------
    alpha : float
        Significance level for CI tests (default 0.05).
    max_depth : int
        Maximum conditioning set depth (default 2).
    cache_generations : int
        Number of generations between full skeleton re-computations.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_depth: int = 2,
        cache_generations: int = 10,
    ) -> None:
        self._alpha = alpha
        self._max_depth = max_depth
        self._cache_generations = cache_generations
        # State
        self._cached_mask: Optional[EdgeMask] = None
        self._generation_counter: int = 0
        self._sep_sets: Dict[FrozenSet[int], List[int]] = {}
        self._n_ci_tests: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def restrict(
        self,
        dag: Any,
        data: DataMatrix,
        ci_test: CITest,
        alpha: Optional[float] = None,
    ) -> EdgeMask:
        """Compute an edge mask from CI-test filtering.

        Uses cached skeleton if available and not expired; otherwise
        recomputes from scratch.

        Parameters
        ----------
        dag :
            Current DAG (used to determine the number of nodes).
        data :
            ``N × p`` data matrix.
        ci_test :
            A CI test object.
        alpha :
            Override significance level.

        Returns
        -------
        EdgeMask
            ``n × n`` boolean array.  ``True`` entries mark edges that
            **may** be included.
        """
        adj = dag.adjacency if hasattr(dag, 'adjacency') else dag._adj.copy()
        n = adj.shape[0]
        use_alpha = alpha if alpha is not None else self._alpha

        self._generation_counter += 1

        if (self._cached_mask is not None
                and self._cached_mask.shape == (n, n)
                and self._generation_counter % self._cache_generations != 0):
            return self._cached_mask

        mask = self.learn_skeleton(data, n, ci_test, use_alpha)
        self._cached_mask = mask
        return mask

    def learn_skeleton(
        self,
        data: DataMatrix,
        n: int,
        ci_test: CITest,
        alpha: float,
    ) -> EdgeMask:
        """Learn an undirected skeleton using PC-style CI testing.

        Starts with a complete graph and removes edges where
        conditional independence is found, testing with increasing
        conditioning set sizes up to ``max_depth``.

        Parameters
        ----------
        data :
            Data matrix.
        n :
            Number of nodes.
        ci_test :
            CI test object.
        alpha :
            Significance level.

        Returns
        -------
        EdgeMask
            Symmetric boolean mask of plausible edges.
        """
        mask = np.ones((n, n), dtype=np.bool_)
        np.fill_diagonal(mask, False)
        self._sep_sets.clear()
        self._n_ci_tests = 0

        for depth in range(self._max_depth + 1):
            cont = False
            for i in range(n):
                adj_i = [j for j in range(n) if mask[i, j] and j != i]
                for j in list(adj_i):
                    if not mask[i, j]:
                        continue
                    possible_cond = [k for k in adj_i if k != j]
                    if len(possible_cond) < depth:
                        continue
                    cont = True
                    for subset in combinations(possible_cond, depth):
                        cond_set = frozenset(subset)
                        result = ci_test.test(i, j, cond_set, data, alpha)
                        self._n_ci_tests += 1
                        if result.is_independent:
                            mask[i, j] = False
                            mask[j, i] = False
                            self._sep_sets[frozenset({i, j})] = list(subset)
                            break
            if not cont:
                break

        logger.info("Skeleton learned: %d edges, %d CI tests",
                     int(mask.sum()) // 2, self._n_ci_tests)
        self._cached_mask = mask
        return mask

    def learn_skeleton_marginal(
        self,
        data: DataMatrix,
        n: int,
        ci_test: CITest,
        alpha: float,
    ) -> EdgeMask:
        """Learn skeleton using only marginal (unconditional) CI tests.

        Faster but less accurate than the full PC-style approach.

        Parameters
        ----------
        data : DataMatrix
        n : int
        ci_test : CITest
        alpha : float

        Returns
        -------
        EdgeMask
        """
        mask = np.zeros((n, n), dtype=np.bool_)
        self._n_ci_tests = 0

        for i in range(n):
            for j in range(i + 1, n):
                result = ci_test.test(i, j, frozenset(), data, alpha)
                self._n_ci_tests += 1
                if not result.is_independent:
                    mask[i, j] = True
                    mask[j, i] = True

        self._cached_mask = mask
        return mask

    # ------------------------------------------------------------------
    # Incremental updates
    # ------------------------------------------------------------------

    def update_after_edge_change(
        self,
        data: DataMatrix,
        ci_test: CITest,
        edge: Tuple[int, int],
        added: bool,
    ) -> EdgeMask:
        """Incrementally update the skeleton after a single edge change.

        When an edge is added, tests whether new conditional
        independencies are introduced in the neighborhood.
        When removed, tests whether previously removed edges should
        be reconsidered.

        Parameters
        ----------
        data :
            Data matrix.
        ci_test :
            CI test object.
        edge :
            ``(source, target)`` of the changed edge.
        added :
            *True* if edge was added, *False* if removed.

        Returns
        -------
        EdgeMask
            Updated edge mask.
        """
        if self._cached_mask is None:
            raise RuntimeError("No cached skeleton to update. Call restrict() first.")

        mask = self._cached_mask.copy()
        i, j = edge
        n = mask.shape[0]
        alpha = self._alpha

        if added:
            # Ensure the edge is in the mask
            mask[i, j] = True
            mask[j, i] = True

            # Re-test neighbors of i and j for new conditional independencies
            for node in (i, j):
                neighbors = [k for k in range(n) if mask[node, k] and k != node]
                for k in neighbors:
                    if k == i or k == j:
                        continue
                    result = ci_test.test(node, k, frozenset({i, j} - {node, k}),
                                          data, alpha)
                    self._n_ci_tests += 1
                    if result.is_independent:
                        mask[node, k] = False
                        mask[k, node] = False
        else:
            # Edge removed: re-test pairs near the removed edge
            mask[i, j] = False
            mask[j, i] = False

            # Check if the edge should still be in the skeleton
            result = ci_test.test(i, j, frozenset(), data, alpha)
            self._n_ci_tests += 1
            if not result.is_independent:
                mask[i, j] = True
                mask[j, i] = True

        self._cached_mask = mask
        return mask

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_allowed_edges(self) -> List[Tuple[int, int]]:
        """Return list of allowed edges from the current mask."""
        if self._cached_mask is None:
            return []
        rows, cols = np.nonzero(self._cached_mask)
        return list(zip(rows.tolist(), cols.tolist()))

    def invalidate_cache(self) -> None:
        """Force re-computation of skeleton on next call."""
        self._cached_mask = None

    @property
    def n_ci_tests(self) -> int:
        """Number of CI tests performed."""
        return self._n_ci_tests

    @property
    def separation_sets(self) -> Dict[FrozenSet[int], List[int]]:
        """Return learned separation sets."""
        return dict(self._sep_sets)
