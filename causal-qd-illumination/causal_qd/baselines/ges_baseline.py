"""Proper Greedy Equivalence Search (GES) baseline (Chickering, 2002).

Implements GES with three phases operating in the space of Markov
equivalence classes (CPDAGs):

  1. **Forward**: greedily insert edges that most improve the score.
  2. **Backward**: greedily delete edges that most improve the score.
  3. **Turning**: greedily reverse undirected CPDAG edges.

At each step the current graph is converted to its CPDAG
representation so that the search traverses equivalence classes
rather than individual DAGs.

References
----------
Chickering, D. M. (2002).  Optimal structure identification with greedy
    search. *JMLR*, 3, 507-554.
"""
from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from causal_qd.core.dag import DAG
from causal_qd.mec.cpdag import CPDAGConverter
from causal_qd.scores.score_base import DecomposableScore
from causal_qd.types import AdjacencyMatrix, DataMatrix

logger = logging.getLogger(__name__)


# ======================================================================
# Default Gaussian BIC scorer
# ======================================================================


class _GaussianBIC(DecomposableScore):
    """Gaussian BIC scoring function used when no scorer is provided."""

    def local_score(self, node: int, parents: list[int], data: DataMatrix) -> float:
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
        rss = max(float(np.sum(residuals ** 2)), 1e-12)
        log_n = np.log(max(n_samples, 1))
        return -n_samples / 2.0 * np.log(rss / max(n_samples, 1)) - (k + 1) * log_n / 2.0


# ======================================================================
# Score cache
# ======================================================================


class _ScoreCache:
    """Cache for local score evaluations keyed by (node, frozenset(parents))."""

    def __init__(self, max_size: int = 50_000) -> None:
        self._cache: Dict[Tuple[int, FrozenSet[int]], float] = {}
        self._max_size = max_size

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
            return val
        val = score_fn.local_score(node, parents, data)
        if len(self._cache) < self._max_size:
            self._cache[key] = val
        return val

    def clear(self) -> None:
        self._cache.clear()


# ======================================================================
# GESBaseline
# ======================================================================


class GESBaseline:
    """Proper Greedy Equivalence Search operating in CPDAG space.

    Parameters
    ----------
    score_fn : DecomposableScore or None
        Decomposable scoring function (e.g. BIC, BDeu, BGe).
        If *None*, Gaussian BIC is used.
    max_iter : int
        Maximum iterations per phase (safety bound).
    phases : tuple of str
        Which phases to run.  Default ``("forward", "backward", "turning")``.
    verbose : bool
        Log progress messages.
    """

    def __init__(
        self,
        score_fn: Optional[DecomposableScore] = None,
        max_iter: int = 10_000,
        phases: Tuple[str, ...] = ("forward", "backward", "turning"),
        verbose: bool = False,
    ) -> None:
        self._score_fn: DecomposableScore = score_fn if score_fn is not None else _GaussianBIC()
        self._max_iter = max_iter
        self._phases = phases
        self._verbose = verbose
        self._converter = CPDAGConverter()
        self._cache = _ScoreCache()

        # Result attributes filled after :meth:`fit`
        self.dag_: Optional[DAG] = None
        self.cpdag_: Optional[AdjacencyMatrix] = None
        self.search_path_: List[Tuple[str, float, int]] = []
        self.n_forward_steps_: int = 0
        self.n_backward_steps_: int = 0
        self.n_turning_steps_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data: DataMatrix) -> DAG:
        """Run GES on *data* and return a representative DAG.

        Parameters
        ----------
        data : DataMatrix
            ``(n_samples, n_nodes)`` observation matrix.

        Returns
        -------
        DAG
            A DAG from the highest-scoring equivalence class found.
        """
        n = data.shape[1]
        adj: AdjacencyMatrix = np.zeros((n, n), dtype=np.int8)
        self._cache.clear()
        self.search_path_ = []
        self.n_forward_steps_ = 0
        self.n_backward_steps_ = 0
        self.n_turning_steps_ = 0

        # Record initial score
        init_score = self._total_score(data, adj, n)
        self.search_path_.append(("init", init_score, 0))

        if "forward" in self._phases:
            adj = self._forward_phase(data, adj, n)
        if "backward" in self._phases:
            adj = self._backward_phase(data, adj, n)
        if "turning" in self._phases:
            adj = self._turning_phase(data, adj, n)

        self.dag_ = DAG(adj)
        self.cpdag_ = self._dag_to_cpdag(adj, n)
        return self.dag_

    def fit_cpdag(self, data: DataMatrix) -> AdjacencyMatrix:
        """Run GES and return the learned CPDAG.

        Returns
        -------
        AdjacencyMatrix
            The CPDAG adjacency matrix.
        """
        self.fit(data)
        assert self.cpdag_ is not None
        return self.cpdag_

    def run(self, data: DataMatrix) -> AdjacencyMatrix:
        """Run GES and return the adjacency matrix of the learned DAG."""
        return self.fit(data).adjacency

    # ------------------------------------------------------------------
    # Forward Phase
    # ------------------------------------------------------------------

    def _forward_phase(
        self, data: DataMatrix, adj: AdjacencyMatrix, n: int,
    ) -> AdjacencyMatrix:
        """Greedily add edges that improve the score.

        At each step the CPDAG is computed to determine which node pairs
        are currently non-adjacent (candidates for insertion).
        """
        for _ in range(self._max_iter):
            cpdag = self._dag_to_cpdag(adj, n)

            best_gain = 0.0
            best_op: Optional[Tuple[int, int]] = None

            for j in range(n):
                parents_j = self._parents(adj, j)
                old_score = self._cached_local_score(data, j, parents_j)

                for i in range(n):
                    if i == j:
                        continue
                    # Only consider pairs non-adjacent in the CPDAG
                    if cpdag[i, j] or cpdag[j, i]:
                        continue

                    new_parents = sorted(parents_j + [i])
                    new_score = self._cached_local_score(data, j, new_parents)
                    gain = new_score - old_score

                    if gain > best_gain:
                        adj[i, j] = 1
                        if DAG.is_acyclic(adj):
                            best_gain = gain
                            best_op = (i, j)
                        adj[i, j] = 0

            if best_op is None:
                break

            i, j = best_op
            adj[i, j] = 1
            self.n_forward_steps_ += 1

            # Canonicalize through CPDAG
            adj = self._canonicalize(adj, n)

            score = self._total_score(data, adj, n)
            self.search_path_.append(("forward", score, int(adj.sum())))
            if self._verbose:
                logger.info(
                    "FWD %d: add %d→%d (gain=%.4f, score=%.4f)",
                    self.n_forward_steps_, i, j, best_gain, score,
                )

        return adj

    # ------------------------------------------------------------------
    # Backward Phase
    # ------------------------------------------------------------------

    def _backward_phase(
        self, data: DataMatrix, adj: AdjacencyMatrix, n: int,
    ) -> AdjacencyMatrix:
        """Greedily remove edges that improve the score."""
        for _ in range(self._max_iter):
            best_gain = 0.0
            best_op: Optional[Tuple[int, int]] = None

            edges = list(zip(*np.nonzero(adj)))
            for i_int, j_int in edges:
                i, j = int(i_int), int(j_int)
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

            # Canonicalize through CPDAG
            adj = self._canonicalize(adj, n)

            score = self._total_score(data, adj, n)
            self.search_path_.append(("backward", score, int(adj.sum())))
            if self._verbose:
                logger.info(
                    "BWD %d: del %d→%d (gain=%.4f, score=%.4f)",
                    self.n_backward_steps_, i, j, best_gain, score,
                )

        return adj

    # ------------------------------------------------------------------
    # Turning Phase
    # ------------------------------------------------------------------

    def _turning_phase(
        self, data: DataMatrix, adj: AdjacencyMatrix, n: int,
    ) -> AdjacencyMatrix:
        """Greedily reverse undirected CPDAG edges that improve the score."""
        for _ in range(self._max_iter):
            cpdag = self._dag_to_cpdag(adj, n)

            best_gain = 0.0
            best_op: Optional[Tuple[int, int]] = None

            # Iterate over undirected edges in the CPDAG
            for i in range(n):
                for j in range(i + 1, n):
                    if not (cpdag[i, j] and cpdag[j, i]):
                        continue

                    # Determine current DAG orientation and try reversal
                    if adj[i, j]:
                        src, tgt = i, j
                    elif adj[j, i]:
                        src, tgt = j, i
                    else:
                        continue

                    parents_tgt = self._parents(adj, tgt)
                    parents_src = self._parents(adj, src)

                    old_score_tgt = self._cached_local_score(data, tgt, parents_tgt)
                    old_score_src = self._cached_local_score(data, src, parents_src)

                    new_parents_tgt = [p for p in parents_tgt if p != src]
                    new_parents_src = sorted(parents_src + [tgt])

                    new_score_tgt = self._cached_local_score(data, tgt, new_parents_tgt)
                    new_score_src = self._cached_local_score(data, src, new_parents_src)

                    gain = (new_score_tgt + new_score_src) - (old_score_tgt + old_score_src)

                    if gain > best_gain:
                        test_adj = adj.copy()
                        test_adj[src, tgt] = 0
                        test_adj[tgt, src] = 1
                        if DAG.is_acyclic(test_adj):
                            best_gain = gain
                            best_op = (src, tgt)

            if best_op is None:
                break

            src, tgt = best_op
            adj[src, tgt] = 0
            adj[tgt, src] = 1
            self.n_turning_steps_ += 1

            adj = self._canonicalize(adj, n)

            score = self._total_score(data, adj, n)
            self.search_path_.append(("turning", score, int(adj.sum())))
            if self._verbose:
                logger.info(
                    "TURN %d: reverse %d→%d (gain=%.4f, score=%.4f)",
                    self.n_turning_steps_, src, tgt, best_gain, score,
                )

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
        return self._cache.get(self._score_fn, data, node, parents)

    def _total_score(
        self, data: DataMatrix, adj: AdjacencyMatrix, n: int,
    ) -> float:
        total = 0.0
        for j in range(n):
            parents = self._parents(adj, j)
            total += self._cached_local_score(data, j, parents)
        return total

    def _dag_to_cpdag(self, adj: AdjacencyMatrix, n: int) -> AdjacencyMatrix:
        """Convert a DAG adjacency matrix to its CPDAG."""
        if int(adj.sum()) == 0:
            return np.zeros((n, n), dtype=np.int8)
        dag = DAG(adj)
        return self._converter.dag_to_cpdag(dag)

    def _canonicalize(self, adj: AdjacencyMatrix, n: int) -> AdjacencyMatrix:
        """Round-trip DAG → CPDAG → canonical DAG.

        This ensures the search traverses equivalence classes rather than
        individual DAG orientations.
        """
        if int(adj.sum()) == 0:
            return adj

        cpdag = self._dag_to_cpdag(adj, n)
        new_adj = np.zeros((n, n), dtype=np.int8)

        # Copy compelled (directed) edges
        for i in range(n):
            for j in range(n):
                if cpdag[i, j] and not cpdag[j, i]:
                    new_adj[i, j] = 1

        # Orient undirected edges greedily to maintain acyclicity
        for i in range(n):
            for j in range(i + 1, n):
                if cpdag[i, j] and cpdag[j, i]:
                    new_adj[i, j] = 1
                    if not DAG.is_acyclic(new_adj):
                        new_adj[i, j] = 0
                        new_adj[j, i] = 1
                        if not DAG.is_acyclic(new_adj):
                            # Should not happen for a valid CPDAG
                            new_adj[j, i] = 0

        if DAG.is_acyclic(new_adj):
            return new_adj
        return adj  # fallback: keep original if canonicalization fails

    def summary(self) -> Dict[str, object]:
        """Return summary statistics of the last fit."""
        return {
            "n_forward_steps": self.n_forward_steps_,
            "n_backward_steps": self.n_backward_steps_,
            "n_turning_steps": self.n_turning_steps_,
            "search_path_length": len(self.search_path_),
        }
