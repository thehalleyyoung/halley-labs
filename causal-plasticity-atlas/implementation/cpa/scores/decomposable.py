"""Decomposable score interface with parent-set caching.

Provides:
- ``DecomposableScore``: abstract base class for decomposable scores.
- ``ScoreCache``: LRU memoisation layer keyed by ``(node, frozenset(parents))``.
- ``CachedScore``: convenience wrapper that pairs a base score with a cache.
- ``ParentSetEnumerator``: combinatorial enumeration of candidate parent sets.
- ``ScoreComparator``: compare multiple scoring functions on the same DAG.

A *decomposable* score is one that can be written as a sum of local
terms, one per node:

    Score(G, D) = sum_i local_score(i, Pa_G(i))

Decomposability is the key property that makes greedy and exact
structure-learning algorithms tractable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from itertools import combinations
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# DecomposableScore  (abstract base)
# ---------------------------------------------------------------------------

class DecomposableScore(ABC):
    """Abstract base class for decomposable DAG scores.

    Sub-classes must implement :meth:`local_score`.  All other methods
    have working default implementations.
    """

    @abstractmethod
    def local_score(self, node: int, parents: Sequence[int]) -> float:
        """Return the local score for *node* given *parents*."""

    def is_decomposable(self) -> bool:
        """Return *True* — this is a decomposable score."""
        return True

    def score_dag(self, adj_matrix: NDArray) -> float:
        """Sum local scores over all nodes in the DAG.

        ``adj_matrix[i, j] != 0`` means i -> j.
        """
        adj = np.asarray(adj_matrix)
        n = adj.shape[0]
        total = 0.0
        for j in range(n):
            parents = list(np.nonzero(adj[:, j])[0])
            total += self.local_score(j, parents)
        return total

    def score_diff(
        self,
        adj_matrix: NDArray,
        i: int,
        j: int,
        operation: str,
    ) -> float:
        """Return the score difference for a single-edge *operation*.

        The difference is ``new_score - old_score`` so positive values
        indicate improvement.

        Parameters
        ----------
        adj_matrix : NDArray
            Current adjacency matrix.
        i, j : int
            Edge endpoints (edge direction is i -> j).
        operation : str
            ``"add"`` — add edge i -> j.
            ``"remove"`` — remove edge i -> j.
            ``"reverse"`` — reverse i -> j to j -> i.
        """
        adj = np.asarray(adj_matrix, dtype=float).copy()
        valid_ops = ("add", "remove", "reverse")
        if operation not in valid_ops:
            raise ValueError(f"operation must be one of {valid_ops}")

        old_parents_j = list(np.nonzero(adj[:, j])[0])
        old_score_j = self.local_score(j, old_parents_j)

        if operation == "add":
            new_parents_j = sorted(set(old_parents_j) | {i})
            new_score_j = self.local_score(j, new_parents_j)
            return new_score_j - old_score_j

        elif operation == "remove":
            new_parents_j = [p for p in old_parents_j if p != i]
            new_score_j = self.local_score(j, new_parents_j)
            return new_score_j - old_score_j

        else:  # reverse
            # Remove i -> j
            new_parents_j = [p for p in old_parents_j if p != i]
            new_score_j = self.local_score(j, new_parents_j)
            delta_j = new_score_j - old_score_j

            # Add j -> i
            old_parents_i = list(np.nonzero(adj[:, i])[0])
            old_score_i = self.local_score(i, old_parents_i)
            new_parents_i = sorted(set(old_parents_i) | {j})
            new_score_i = self.local_score(i, new_parents_i)
            delta_i = new_score_i - old_score_i

            return delta_j + delta_i

    def score_all_single_edge_changes(
        self, adj_matrix: NDArray
    ) -> Dict[Tuple[int, int, str], float]:
        """Evaluate all possible single-edge operations.

        Returns a dict mapping ``(i, j, op)`` to the score difference.
        Only operations that produce valid DAGs are included (no
        acyclicity check is performed here).
        """
        adj = np.asarray(adj_matrix)
        n = adj.shape[0]
        results: Dict[Tuple[int, int, str], float] = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if adj[i, j] == 0:
                    diff = self.score_diff(adj, i, j, "add")
                    results[(i, j, "add")] = diff
                else:
                    diff = self.score_diff(adj, i, j, "remove")
                    results[(i, j, "remove")] = diff
                    diff = self.score_diff(adj, i, j, "reverse")
                    results[(i, j, "reverse")] = diff
        return results


# ---------------------------------------------------------------------------
# ScoreCache  (LRU memoisation)
# ---------------------------------------------------------------------------

class ScoreCache:
    """LRU cache for ``local_score`` evaluations.

    Parameters
    ----------
    score_fn : Callable[[int, Sequence[int]], float]
        Any callable with signature ``(node, parents) -> float``.
    max_size : int
        Maximum number of cached entries.  When exceeded the
        least-recently-used entry is evicted.
    """

    def __init__(
        self,
        score_fn: Callable[[int, Sequence[int]], float],
        max_size: int = 10_000,
    ) -> None:
        self._score_fn = score_fn
        self._max_size = max_size
        self._cache: OrderedDict[Tuple[int, frozenset], float] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _cache_key(
        node: int, parents: Sequence[int]
    ) -> Tuple[int, frozenset]:
        """Create a hashable cache key."""
        return (node, frozenset(parents))

    def get(self, node: int, parents: Sequence[int]) -> float:
        """Return the cached score, computing and storing it if absent."""
        key = self._cache_key(node, parents)
        if key in self._cache:
            self._hits += 1
            # Move to end (most-recently-used)
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        value = self._score_fn(node, parents)
        self._cache[key] = value
        self._cache.move_to_end(key)
        # Evict LRU if over capacity
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return value

    def invalidate(self, node: int) -> None:
        """Remove all cached entries involving *node* (as child or parent)."""
        to_remove = [
            k for k in self._cache
            if k[0] == node or node in k[1]
        ]
        for k in to_remove:
            del self._cache[k]

    def clear(self) -> None:
        """Remove all cached entries and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def cache_info(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "max_size": self._max_size,
        }

    def precompute(
        self, node: int, candidates: Sequence[int], max_parents: int
    ) -> None:
        """Pre-populate the cache for all parent sets up to *max_parents*.

        Parameters
        ----------
        node : int
        candidates : Sequence[int]
            Candidate parent variables (excluding *node*).
        max_parents : int
            Maximum parent-set size to enumerate.
        """
        for size in range(0, max_parents + 1):
            for combo in combinations(candidates, size):
                self.get(node, list(combo))

    def __contains__(self, item: Tuple[int, Sequence[int]]) -> bool:
        node, parents = item
        return self._cache_key(node, parents) in self._cache

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# CachedScore  (wraps a base score with a cache)
# ---------------------------------------------------------------------------

class CachedScore:
    """Convenience wrapper: a base score + an LRU cache.

    Provides the same ``local_score`` / ``score_dag`` interface but
    transparently memoises ``local_score`` calls.

    Parameters
    ----------
    base_score : object
        Any object with a ``local_score(node, parents)`` method.
    cache_size : int
        Maximum cache entries.
    """

    def __init__(self, base_score: Any, cache_size: int = 100_000) -> None:
        self._base = base_score
        self._cache = ScoreCache(base_score.local_score, max_size=cache_size)

    def local_score(self, node: int, parents: Sequence[int]) -> float:
        """Cached local score."""
        return self._cache.get(node, parents)

    def score_dag(self, adj_matrix: NDArray) -> float:
        """Score a DAG using cached local scores."""
        adj = np.asarray(adj_matrix)
        n = adj.shape[0]
        total = 0.0
        for j in range(n):
            parents = list(np.nonzero(adj[:, j])[0])
            total += self.local_score(j, parents)
        return total

    def cache_stats(self) -> Dict[str, int]:
        """Return hit/miss/size statistics."""
        return self._cache.cache_info()

    def clear_cache(self) -> None:
        """Clear all cached scores."""
        self._cache.clear()

    def precompute(self, node: int, max_parents: int) -> None:
        """Pre-compute all parent sets up to *max_parents* for *node*."""
        n = self._base.n_variables if hasattr(self._base, "n_variables") else 0
        candidates = [v for v in range(n) if v != node]
        self._cache.precompute(node, candidates, max_parents)


# ---------------------------------------------------------------------------
# ParentSetEnumerator
# ---------------------------------------------------------------------------

class ParentSetEnumerator:
    """Enumerate candidate parent sets for structure learning.

    Parameters
    ----------
    num_variables : int
        Total number of variables in the model.
    max_parents : Optional[int]
        Global upper bound on parent-set size.  Defaults to
        ``num_variables - 1`` (no restriction beyond DAG constraint).
    """

    def __init__(
        self, num_variables: int, max_parents: Optional[int] = None
    ) -> None:
        self.num_variables = num_variables
        self.max_parents = max_parents if max_parents is not None else num_variables - 1

    def enumerate_parent_sets(
        self, node: int
    ) -> Generator[List[int], None, None]:
        """Generate all valid parent sets for *node* up to ``max_parents``."""
        yield from self.enumerate_parent_sets_up_to(node, self.max_parents)

    def enumerate_parent_sets_up_to(
        self, node: int, max_size: int
    ) -> Generator[List[int], None, None]:
        """Generate all parent sets for *node* up to *max_size*."""
        candidates = [v for v in range(self.num_variables) if v != node]
        yield from self._combinations_generator(candidates, max_size)

    @staticmethod
    def _combinations_generator(
        candidates: List[int], max_size: int
    ) -> Generator[List[int], None, None]:
        """Yield all subsets of *candidates* with size 0..max_size."""
        for size in range(0, min(max_size, len(candidates)) + 1):
            for combo in combinations(candidates, size):
                yield list(combo)

    def count_parent_sets(self, node: int) -> int:
        """Return the number of parent sets for *node*."""
        n = self.num_variables - 1
        total = 0
        for k in range(0, min(self.max_parents, n) + 1):
            total += _n_choose_k(n, k)
        return total


def _n_choose_k(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


# ---------------------------------------------------------------------------
# ScoreComparator
# ---------------------------------------------------------------------------

class ScoreComparator:
    """Compare multiple scoring functions on the same structures.

    Parameters
    ----------
    scores : dict
        Mapping from score name to a score object with
        ``local_score(node, parents)`` and ``score_dag(adj)`` methods.
    """

    def __init__(self, scores: Dict[str, Any]) -> None:
        self.scores = scores

    def compare_scores(
        self, dag: NDArray
    ) -> Dict[str, float]:
        """Return DAG score under each scoring function.

        Returns
        -------
        dict mapping score name -> total DAG score.
        """
        results = {}
        adj = np.asarray(dag)
        for name, scorer in self.scores.items():
            results[name] = scorer.score_dag(adj)
        return results

    def rank_parent_sets(
        self,
        node: int,
        max_parents: int,
        num_variables: int,
    ) -> Dict[str, List[Tuple[List[int], float]]]:
        """Rank parent sets by score under each scoring function.

        Returns
        -------
        dict mapping score name -> sorted list of (parents, score).
        """
        enum = ParentSetEnumerator(num_variables, max_parents)
        parent_sets = list(enum.enumerate_parent_sets(node))

        results: Dict[str, List[Tuple[List[int], float]]] = {}
        for name, scorer in self.scores.items():
            scored = [
                (ps, scorer.local_score(node, ps))
                for ps in parent_sets
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            results[name] = scored
        return results

    def agreement_matrix(
        self, dag: NDArray
    ) -> NDArray:
        """Return pairwise rank-correlation of node-level scores.

        For each pair of scoring functions, compute Spearman correlation
        of the per-node local scores.

        Returns
        -------
        NDArray of shape (n_scores, n_scores)
        """
        adj = np.asarray(dag)
        n = adj.shape[0]
        names = list(self.scores.keys())
        n_scores = len(names)

        # Collect per-node scores for each scoring function
        node_scores = np.zeros((n_scores, n))
        for si, name in enumerate(names):
            scorer = self.scores[name]
            for j in range(n):
                parents = list(np.nonzero(adj[:, j])[0])
                node_scores[si, j] = scorer.local_score(j, parents)

        # Spearman correlation
        from scipy.stats import spearmanr
        corr_matrix = np.eye(n_scores)
        for si in range(n_scores):
            for sj in range(si + 1, n_scores):
                if n < 3:
                    rho = 1.0
                else:
                    rho, _ = spearmanr(node_scores[si], node_scores[sj])
                    if np.isnan(rho):
                        rho = 0.0
                corr_matrix[si, sj] = rho
                corr_matrix[sj, si] = rho
        return corr_matrix
