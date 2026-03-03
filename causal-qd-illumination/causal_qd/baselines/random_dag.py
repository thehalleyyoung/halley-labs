"""Random DAG baseline for causal discovery.

Provides random DAG generation strategies for use as baselines:
  - Single-sample random DAG (Erdős–Rényi style on a random ordering)
  - Multi-restart random search (generate many, keep the best)
  - Random walk on DAG space (starting from empty, random single-edge mutations)

These serve as lower-bound sanity checks: any serious causal discovery
method should outperform these baselines.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from causal_qd.core.dag import DAG
from causal_qd.scores.score_base import ScoreFunction
from causal_qd.types import AdjacencyMatrix, DataMatrix

logger = logging.getLogger(__name__)


# ======================================================================
# Default scorer
# ======================================================================


class _GaussianBIC(ScoreFunction):
    """Simple Gaussian BIC score for baselines."""

    def local_score(self, data: DataMatrix, node: int, parents: List[int]) -> float:
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
        return -n_samples / 2.0 * np.log(rss / n_samples) - (k + 1) * log_n / 2.0


# ======================================================================
# RandomDAGBaseline
# ======================================================================


class RandomDAGBaseline:
    """Baseline that generates random DAGs and returns the best by score.

    Parameters
    ----------
    n_random : int
        Number of random DAGs to generate (default 100).
    edge_prob : float
        Probability of each potential edge being present (default 0.3).
    score_fn : ScoreFunction or None
        Scoring function.  If *None*, Gaussian BIC is used.
    max_parents : int or None
        Maximum number of parents per node.  *None* means no limit.
    strategy : str
        Generation strategy: ``"sample"`` (independent random DAGs),
        ``"walk"`` (random walk from empty graph).
    walk_steps : int
        Number of random-walk steps (only for ``strategy="walk"``).
    """

    def __init__(
        self,
        n_random: int = 100,
        edge_prob: float = 0.3,
        score_fn: Optional[ScoreFunction] = None,
        max_parents: Optional[int] = None,
        strategy: str = "sample",
        walk_steps: int = 200,
    ) -> None:
        self._n_random = n_random
        self._edge_prob = edge_prob
        self._score_fn: ScoreFunction = score_fn if score_fn is not None else _GaussianBIC()
        self._max_parents = max_parents
        self._strategy = strategy
        self._walk_steps = walk_steps
        # Diagnostics
        self.scores_: List[float] = []
        self.best_score_: float = -np.inf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data: DataMatrix,
        rng: Optional[np.random.Generator] = None,
    ) -> DAG:
        """Generate random DAGs and return the one with the best score.

        Parameters
        ----------
        data : DataMatrix
            ``(n_samples, n_nodes)`` observation matrix.
        rng : numpy.random.Generator or None
            Random number generator.

        Returns
        -------
        DAG
            The highest-scoring random DAG.
        """
        if rng is None:
            rng = np.random.default_rng()

        n = data.shape[1]
        self.scores_ = []

        if self._strategy == "walk":
            return self._random_walk_search(data, n, rng)
        return self._multi_restart_search(data, n, rng)

    def run(self, data: DataMatrix) -> AdjacencyMatrix:
        """Fit and return adjacency matrix."""
        return self.fit(data).adjacency

    # ------------------------------------------------------------------
    # Multi-restart random sampling
    # ------------------------------------------------------------------

    def _multi_restart_search(
        self, data: DataMatrix, n: int, rng: np.random.Generator,
    ) -> DAG:
        """Generate ``n_random`` random DAGs and return the best."""
        best_dag: Optional[DAG] = None
        best_score = -np.inf

        for trial in range(self._n_random):
            dag = self._random_dag(n, rng)
            score = self._score_dag(data, dag)
            self.scores_.append(score)
            if score > best_score:
                best_score = score
                best_dag = dag

        self.best_score_ = best_score
        assert best_dag is not None
        return best_dag

    # ------------------------------------------------------------------
    # Random walk on DAG space
    # ------------------------------------------------------------------

    def _random_walk_search(
        self, data: DataMatrix, n: int, rng: np.random.Generator,
    ) -> DAG:
        """Perform a random walk on DAG space, tracking the best DAG.

        Starting from an empty graph, at each step randomly add, remove,
        or reverse a single edge while maintaining acyclicity.
        """
        adj = np.zeros((n, n), dtype=np.int8)
        best_adj = adj.copy()
        current_score = self._score_adj(data, adj, n)
        best_score = current_score
        self.scores_.append(current_score)

        for step in range(self._walk_steps):
            op = rng.choice(["add", "remove", "reverse"])
            candidate = adj.copy()

            if op == "add":
                # Pick a random non-edge
                non_edges = [(i, j) for i in range(n) for j in range(n)
                             if i != j and not candidate[i, j]]
                if not non_edges:
                    continue
                i, j = non_edges[rng.integers(len(non_edges))]
                candidate[i, j] = 1
                if self._max_parents is not None:
                    if int(candidate[:, j].sum()) > self._max_parents:
                        continue
            elif op == "remove":
                edges = list(zip(*np.nonzero(candidate)))
                if not edges:
                    continue
                idx = rng.integers(len(edges))
                i, j = edges[idx]
                candidate[i, j] = 0
            else:  # reverse
                edges = list(zip(*np.nonzero(candidate)))
                if not edges:
                    continue
                idx = rng.integers(len(edges))
                i, j = edges[idx]
                candidate[i, j] = 0
                candidate[j, i] = 1

            if not DAG.is_acyclic(candidate):
                continue

            score = self._score_adj(data, candidate, n)
            self.scores_.append(score)

            # Accept with probability proportional to improvement
            # (always accept improvements, sometimes accept worse)
            delta = score - current_score
            if delta > 0 or rng.random() < np.exp(min(delta, 0)):
                adj = candidate
                current_score = score
                if score > best_score:
                    best_score = score
                    best_adj = adj.copy()

        self.best_score_ = best_score
        return DAG(best_adj)

    # ------------------------------------------------------------------
    # DAG generation
    # ------------------------------------------------------------------

    def _random_dag(self, n: int, rng: np.random.Generator) -> DAG:
        """Sample a random DAG with *n* nodes.

        Edges are placed only from lower-index to higher-index nodes in
        a random permutation, guaranteeing acyclicity.
        """
        perm = rng.permutation(n).tolist()
        adj = np.zeros((n, n), dtype=np.int8)
        for idx_i in range(n):
            for idx_j in range(idx_i + 1, n):
                if rng.random() < self._edge_prob:
                    if self._max_parents is not None:
                        target = perm[idx_j]
                        if int(adj[:, target].sum()) >= self._max_parents:
                            continue
                    adj[perm[idx_i], perm[idx_j]] = 1
        return DAG(adj)

    @staticmethod
    def random_erdos_renyi(n: int, p: float, rng: np.random.Generator) -> DAG:
        """Generate a random Erdős–Rényi DAG.

        Parameters
        ----------
        n : int
            Number of nodes.
        p : float
            Edge probability.
        rng : Generator
            Random number generator.

        Returns
        -------
        DAG
        """
        perm = rng.permutation(n)
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[perm[i], perm[j]] = 1
        return DAG(adj)

    @staticmethod
    def random_scale_free(n: int, m: int, rng: np.random.Generator) -> DAG:
        """Generate a random scale-free (Barabási–Albert) DAG.

        Parameters
        ----------
        n : int
            Number of nodes.
        m : int
            Number of edges each new node brings (≥ 1).
        rng : Generator

        Returns
        -------
        DAG
        """
        adj = np.zeros((n, n), dtype=np.int8)
        perm = rng.permutation(n).tolist()

        for idx in range(1, n):
            target_node = perm[idx]
            # Pick m parents from already-placed nodes with prob ∝ degree + 1
            placed = [perm[k] for k in range(idx)]
            degrees = np.array([int(adj[p, :].sum()) + int(adj[:, p].sum()) + 1
                                for p in placed], dtype=np.float64)
            probs = degrees / degrees.sum()
            n_parents = min(m, len(placed))
            chosen = rng.choice(placed, size=n_parents, replace=False, p=probs)
            for parent in chosen:
                adj[parent, target_node] = 1

        return DAG(adj)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_dag(self, data: DataMatrix, dag: DAG) -> float:
        """Score a DAG using the scoring function."""
        return self._score_fn.score_dag(data, dag.adjacency)

    def _score_adj(self, data: DataMatrix, adj: AdjacencyMatrix, n: int) -> float:
        """Score an adjacency matrix."""
        total = 0.0
        for j in range(n):
            parents = sorted(int(i) for i in np.nonzero(adj[:, j])[0])
            total += self._score_fn.local_score(data, j, parents)
        return total

    @staticmethod
    def _bic_score(data: DataMatrix, dag: DAG) -> float:
        """Legacy Gaussian BIC score for backward compatibility."""
        n_samples, n_nodes = data.shape
        log_n = np.log(max(n_samples, 1))
        total = 0.0
        for j in range(n_nodes):
            parents = sorted(dag.parents(j))
            y = data[:, j]
            if parents:
                X = data[:, parents]
                coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                residuals = y - X @ coef
            else:
                residuals = y - y.mean()
            rss = max(float(np.sum(residuals ** 2)), 1e-12)
            total += -n_samples / 2.0 * np.log(rss / n_samples) - len(parents) * log_n / 2.0
        return total

    def summary(self) -> Dict[str, object]:
        """Return diagnostics from the last fit."""
        return {
            "n_random": self._n_random,
            "strategy": self._strategy,
            "best_score": self.best_score_,
            "n_scores": len(self.scores_),
        }
