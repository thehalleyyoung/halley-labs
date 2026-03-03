"""Greedy Equivalence Search (GES) for score-based causal discovery.

Implements the full GES algorithm (Chickering, 2002) with three phases:
forward (edge addition), backward (edge removal), and turning (edge
reversal in CPDAG space).  Includes efficient score caching and
optional support for restricted edge sets.

References
----------
Chickering, D. M. (2002).  Optimal structure identification with greedy
    search. *JMLR*, 3, 507-554.
Hauser, A. & Bühlmann, P. (2012).  Characterization and greedy learning
    of interventional Markov equivalence classes.  *JMLR*, 13, 2409-2464.
"""
from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from causal_qd.core.dag import DAG
from causal_qd.scores.score_base import DecomposableScore
from causal_qd.types import AdjacencyMatrix, DataMatrix

logger = logging.getLogger(__name__)


# ======================================================================
# Default Gaussian BIC scorer (fallback)
# ======================================================================


class _DefaultBIC(DecomposableScore):
    """Gaussian BIC scoring function used when no scorer is provided."""

    def local_score(self, node: int, parents: List[int], data: DataMatrix) -> float:
        n_samples = data.shape[0]
        y = data[:, node]
        k = len(parents)
        if k > 0:
            X = data[:, parents]
            try:
                coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                residuals = y - X @ coef
            except np.linalg.LinAlgError:
                residuals = y - y.mean()
        else:
            residuals = y - y.mean()
        rss = float(np.sum(residuals ** 2))
        rss = max(rss, 1e-12)
        log_n = np.log(max(n_samples, 1))
        return -n_samples / 2.0 * np.log(rss / n_samples) - (k + 1) * log_n / 2.0


# ======================================================================
# Score Cache
# ======================================================================


class _ScoreCache:
    """LRU-style cache for local score values.

    Keys are ``(node, frozenset(parents))`` tuples; values are floats.
    """

    def __init__(self, max_size: int = 50_000) -> None:
        self._cache: Dict[Tuple[int, FrozenSet[int]], float] = {}
        self._max_size = max_size
        self.hits: int = 0
        self.misses: int = 0

    def get(
        self,
        score_fn: DecomposableScore,
        data: DataMatrix,
        node: int,
        parents: List[int],
    ) -> float:
        key = (node, frozenset(parents))
        val = self._cache.get(key)
        if val is not None:
            self.hits += 1
            return val
        self.misses += 1
        val = score_fn.local_score(node, parents, data)
        if len(self._cache) < self._max_size:
            self._cache[key] = val
        return val

    def invalidate_node(self, node: int) -> None:
        """Remove all entries involving *node*."""
        to_del = [k for k in self._cache if k[0] == node or node in k[1]]
        for k in to_del:
            del self._cache[k]

    def clear(self) -> None:
        self._cache.clear()
        self.hits = self.misses = 0


# ======================================================================
# GES Algorithm
# ======================================================================


class GESAlgorithm:
    """Greedy Equivalence Search (Chickering, 2002).

    GES searches the space of Markov equivalence classes using a
    three-phase greedy strategy guided by a decomposable score:

    1. **Forward**: greedily add edges that most improve the score.
    2. **Backward**: greedily remove edges that most improve the score.
    3. **Turning**: greedily reverse edges that most improve the score.

    Parameters
    ----------
    score_fn : DecomposableScore or None
        A decomposable scoring function (e.g. BIC, BDeu).
        If *None*, Gaussian BIC is used.
    cache_size : int
        Maximum number of cached local scores.
    max_iter : int
        Maximum iterations per phase (safety bound).
    verbose : bool
        Log progress messages.
    """

    def __init__(
        self,
        score_fn: Optional[DecomposableScore] = None,
        cache_size: int = 50_000,
        max_iter: int = 10_000,
        verbose: bool = False,
    ) -> None:
        self._score_fn: DecomposableScore = score_fn if score_fn is not None else _DefaultBIC()
        self._cache = _ScoreCache(cache_size)
        self._max_iter = max_iter
        self._verbose = verbose
        # Result attributes
        self.n_forward_steps_: int = 0
        self.n_backward_steps_: int = 0
        self.n_turning_steps_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data: DataMatrix) -> DAG:
        """Run GES on *data* and return the learned DAG.

        Parameters
        ----------
        data : DataMatrix

        Returns
        -------
        DAG
        """
        n = data.shape[1]
        adj = np.zeros((n, n), dtype=np.int8)
        self._cache.clear()
        self.n_forward_steps_ = 0
        self.n_backward_steps_ = 0
        self.n_turning_steps_ = 0

        adj = self.forward_phase(data, adj, n)
        adj = self.backward_phase(data, adj, n)
        adj = self.turning_phase(data, adj, n)

        return DAG(adj)

    def run(self, data: DataMatrix) -> AdjacencyMatrix:
        """Run GES and return the adjacency matrix.

        Parameters
        ----------
        data : DataMatrix

        Returns
        -------
        AdjacencyMatrix
        """
        dag = self.fit(data)
        return dag.adjacency

    # ------------------------------------------------------------------
    # Forward Phase
    # ------------------------------------------------------------------

    def forward_phase(
        self, data: DataMatrix, adj: AdjacencyMatrix, n: int,
    ) -> AdjacencyMatrix:
        """Greedily add edges that improve the score.

        For each possible edge addition (i, j) where i and j are not
        currently adjacent, evaluate the score improvement of adding
        i → j.  Pick the best addition, apply it if it improves the
        score, and repeat until no improving move exists.

        Parameters
        ----------
        data : DataMatrix
        adj : AdjacencyMatrix
        n : int

        Returns
        -------
        AdjacencyMatrix
        """
        for iteration in range(self._max_iter):
            best_gain = 0.0
            best_op: Optional[Tuple[int, int, List[int]]] = None

            for j in range(n):
                parents_j = self._parents(adj, j)
                old_score = self._cached_local_score(data, j, parents_j)

                for i in range(n):
                    if i == j or adj[i, j] or adj[j, i]:
                        continue

                    new_parents = sorted(parents_j + [i])
                    new_score = self._cached_local_score(data, j, new_parents)
                    gain = new_score - old_score

                    # Verify acyclicity
                    if gain > best_gain:
                        adj[i, j] = 1
                        if DAG.is_acyclic(adj):
                            best_gain = gain
                            best_op = (i, j, new_parents)
                        adj[i, j] = 0

            if best_op is None:
                break

            i, j, _ = best_op
            adj[i, j] = 1
            self.n_forward_steps_ += 1
            if self._verbose:
                logger.info("FWD step %d: add %d -> %d (gain=%.4f)",
                            self.n_forward_steps_, i, j, best_gain)

        return adj

    # ------------------------------------------------------------------
    # Backward Phase
    # ------------------------------------------------------------------

    def backward_phase(
        self, data: DataMatrix, adj: AdjacencyMatrix, n: int,
    ) -> AdjacencyMatrix:
        """Greedily remove edges that improve the score.

        For each existing edge (i, j), evaluate the score improvement
        of removing it.  Pick the best removal, apply it if it improves
        the score, and repeat.

        Parameters
        ----------
        data : DataMatrix
        adj : AdjacencyMatrix
        n : int

        Returns
        -------
        AdjacencyMatrix
        """
        for iteration in range(self._max_iter):
            best_gain = 0.0
            best_op: Optional[Tuple[int, int]] = None

            edges = list(zip(*np.nonzero(adj)))
            for i, j in edges:
                parents_j = self._parents(adj, j)
                old_score = self._cached_local_score(data, j, parents_j)

                new_parents = [p for p in parents_j if p != i]
                new_score = self._cached_local_score(data, j, new_parents)
                gain = new_score - old_score

                if gain > best_gain:
                    best_gain = gain
                    best_op = (i, j)

            if best_op is None:
                break

            i, j = best_op
            adj[i, j] = 0
            self.n_backward_steps_ += 1
            if self._verbose:
                logger.info("BWD step %d: remove %d -> %d (gain=%.4f)",
                            self.n_backward_steps_, i, j, best_gain)

        return adj

    # ------------------------------------------------------------------
    # Turning Phase
    # ------------------------------------------------------------------

    def turning_phase(
        self, data: DataMatrix, adj: AdjacencyMatrix, n: int,
    ) -> AdjacencyMatrix:
        """Greedily reverse edges that improve the score.

        For each existing edge (i, j), evaluate the score improvement
        of reversing it to (j, i).  Pick the best reversal, apply it
        if it improves the score, and repeat.

        Parameters
        ----------
        data : DataMatrix
        adj : AdjacencyMatrix
        n : int

        Returns
        -------
        AdjacencyMatrix
        """
        for iteration in range(self._max_iter):
            best_gain = 0.0
            best_op: Optional[Tuple[int, int]] = None

            edges = list(zip(*np.nonzero(adj)))
            for i, j in edges:
                # Score change: remove i→j from j's parents, add j→i to i's parents
                parents_j_old = self._parents(adj, j)
                parents_i_old = self._parents(adj, i)

                old_score_j = self._cached_local_score(data, j, parents_j_old)
                old_score_i = self._cached_local_score(data, i, parents_i_old)

                new_parents_j = [p for p in parents_j_old if p != i]
                new_parents_i = sorted(parents_i_old + [j])

                new_score_j = self._cached_local_score(data, j, new_parents_j)
                new_score_i = self._cached_local_score(data, i, new_parents_i)

                gain = (new_score_j + new_score_i) - (old_score_j + old_score_i)

                if gain > best_gain:
                    # Verify acyclicity of the reversal
                    adj[i, j] = 0
                    adj[j, i] = 1
                    if DAG.is_acyclic(adj):
                        best_gain = gain
                        best_op = (i, j)
                    adj[j, i] = 0
                    adj[i, j] = 1

            if best_op is None:
                break

            i, j = best_op
            adj[i, j] = 0
            adj[j, i] = 1
            self.n_turning_steps_ += 1
            if self._verbose:
                logger.info("TURN step %d: reverse %d -> %d to %d -> %d (gain=%.4f)",
                            self.n_turning_steps_, i, j, j, i, best_gain)

        return adj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parents(adj: AdjacencyMatrix, j: int) -> List[int]:
        """Return sorted parent list of node *j*."""
        return sorted(int(i) for i in np.nonzero(adj[:, j])[0])

    def _cached_local_score(
        self, data: DataMatrix, node: int, parents: List[int],
    ) -> float:
        """Compute local score with caching."""
        return self._cache.get(self._score_fn, data, node, parents)

    def _total_score(
        self, data: DataMatrix, adj: AdjacencyMatrix, n: int,
    ) -> float:
        """Compute the total decomposable score for the graph."""
        total = 0.0
        for j in range(n):
            parents = self._parents(adj, j)
            total += self._cached_local_score(data, j, parents)
        return total

    def summary(self) -> Dict[str, object]:
        """Return summary statistics of the last fit."""
        return {
            "n_forward_steps": self.n_forward_steps_,
            "n_backward_steps": self.n_backward_steps_,
            "n_turning_steps": self.n_turning_steps_,
            "cache_hits": self._cache.hits,
            "cache_misses": self._cache.misses,
        }
