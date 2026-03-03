"""Cached wrapper for scoring functions.

This module provides :class:`CachedScore`, a transparent caching layer that
wraps any :class:`DecomposableScore` and memoises local-score evaluations
using an LRU cache keyed by ``(node, frozenset(parents))``.  Repeated
queries for the same local structure—common during greedy hill-climbing and
MAP-Elites mutation loops—become essentially free after the first
evaluation.

The implementation is **thread-safe** with respect to the underlying
:func:`functools.lru_cache`, which uses an internal lock.  Manual hit/miss
counters are updated under their own :class:`threading.Lock` so that
:attr:`CachedScore.cache_info` is safe to read from any thread.

Typical usage::

    from causal_qd.scores.bic import BICScore
    from causal_qd.scores.cached import CachedScore

    cached = CachedScore(BICScore(), max_cache_size=16384)
    quality  = cached.score(dag, data)
    print(cached.cache_info)
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

from causal_qd.scores.score_base import DecomposableScore, ScoreFunction
from causal_qd.types import AdjacencyMatrix, DataMatrix, QualityScore


# ---------------------------------------------------------------------------
# Cache statistics dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheStats:
    """Snapshot of cache performance statistics.

    Attributes
    ----------
    hits : int
        Number of cache hits (queries answered from the cache).
    misses : int
        Number of cache misses (queries that required computation).
    hit_rate : float
        Fraction of total queries answered from cache (0.0–1.0).
    current_size : int
        Number of entries currently stored in the cache.
    max_size : int
        Maximum number of entries the cache can hold.
    memory_estimate_bytes : int
        Rough estimate of memory consumed by cached entries.  Each entry
        is assumed to occupy ~128 bytes (key tuple + float result +
        LRU bookkeeping overhead).
    """

    hits: int
    misses: int
    hit_rate: float
    current_size: int
    max_size: int
    memory_estimate_bytes: int

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.2%}, size={self.current_size}/"
            f"{self.max_size}, mem≈{self.memory_estimate_bytes} B)"
        )


# Rough per-entry memory overhead (key tuple + float + LRU pointers).
_BYTES_PER_ENTRY = 128


# ---------------------------------------------------------------------------
# CachedScore
# ---------------------------------------------------------------------------

class CachedScore(ScoreFunction):
    """Wrap a :class:`DecomposableScore` and cache local-score evaluations.

    Caching uses :func:`functools.lru_cache` keyed by
    ``(node: int, parents: frozenset[int])`` so that repeated queries for
    the same local structure are essentially free.

    The underlying ``lru_cache`` is created once per instance inside
    ``__init__`` as a closure over ``self``, which allows it to access
    ``self._data`` (set before each batch of lookups) without making the
    data part of the cache key.

    Parameters
    ----------
    base_score : DecomposableScore
        The underlying decomposable score to wrap.
    max_cache_size : int, optional
        Maximum number of cached entries (default ``8192``).

    Notes
    -----
    *   **Thread safety** – ``functools.lru_cache`` uses an internal
        reentrant lock, so concurrent calls to :meth:`local_score` or
        :meth:`score` are safe.  The manual hit/miss counters are guarded
        by a separate :class:`threading.Lock`.
    *   If the dataset changes between calls you should call
        :meth:`clear_cache` to invalidate stale entries, since the data
        reference is *not* part of the cache key.
    """

    def __init__(
        self,
        base_score: DecomposableScore,
        max_cache_size: int = 8192,
    ) -> None:
        self._base = base_score
        self._max_cache_size = max_cache_size
        self._data: Optional[DataMatrix] = None

        # Manual hit/miss tracking via a flag set inside the cached closure.
        # This avoids calling cache_info() (which acquires a lock) on every
        # lookup — the old code called it *twice* per lookup (before & after).
        self._hits: int = 0
        self._misses: int = 0
        self._stats_lock = threading.Lock()
        self._miss_flag: bool = False

        # Build the cached closure.  ``self`` is captured by reference so
        # that ``self._data`` can be swapped in before each scoring round.
        @lru_cache(maxsize=max_cache_size)
        def _cached_local(node: int, parents: frozenset[int]) -> float:
            assert self._data is not None, (
                "CachedScore._data must be set before calling _cached_local"
            )
            self._miss_flag = True
            return self._base.local_score(node, sorted(parents), self._data)

        self._cached_local = _cached_local

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def score(self, dag: AdjacencyMatrix, data: DataMatrix) -> QualityScore:
        """Compute the full DAG score as the sum of cached local scores.

        Parameters
        ----------
        dag : AdjacencyMatrix
            ``n × n`` binary adjacency matrix of the candidate DAG.
        data : DataMatrix
            ``N × p`` observed data matrix.

        Returns
        -------
        QualityScore
            Scalar score (higher is better by convention).
        """
        self._data = data
        n = dag.shape[0]
        total = 0.0
        for j in range(n):
            parents = frozenset(int(i) for i in np.where(dag[:, j])[0])
            total += self._lookup(j, parents)
        return total

    def local_score(
        self,
        node: int,
        parents: list[int],
        data: DataMatrix,
    ) -> float:
        """Return the local score for *node* given *parents*, using cache.

        The method first checks the LRU cache.  On a miss the underlying
        :class:`DecomposableScore` is queried and the result is stored.

        Parameters
        ----------
        node : int
            Index of the child node.
        parents : list[int]
            Sorted indices of the parent nodes.
        data : DataMatrix
            ``N × p`` data matrix.

        Returns
        -------
        float
            Local score contribution.
        """
        self._data = data
        return self._lookup(node, frozenset(parents))

    def score_diff(
        self,
        dag: AdjacencyMatrix,
        node: int,
        old_parents: list[int],
        new_parents: list[int],
        data: DataMatrix,
    ) -> float:
        """Return the score difference from changing *node*'s parent set.

        Computes ``local_score(node, new_parents) - local_score(node,
        old_parents)`` using cached lookups, avoiding a full DAG re-score.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Current adjacency matrix (unused directly, kept for API
            consistency with potential future extensions).
        node : int
            Index of the child node whose parents change.
        old_parents : list[int]
            Previous parent set indices.
        new_parents : list[int]
            Proposed parent set indices.
        data : DataMatrix
            ``N × p`` data matrix.

        Returns
        -------
        float
            ``score(new_parents) - score(old_parents)`` for *node*.
        """
        self._data = data
        new_val = self._lookup(node, frozenset(new_parents))
        old_val = self._lookup(node, frozenset(old_parents))
        return new_val - old_val

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def batch_score(
        self,
        nodes_and_parents: list[tuple[int, list[int]]],
        data: DataMatrix,
    ) -> list[float]:
        """Score multiple ``(node, parents)`` pairs in one call.

        This is a convenience wrapper that avoids repeated Python-level
        method-call overhead while still benefiting from caching.

        Parameters
        ----------
        nodes_and_parents : list[tuple[int, list[int]]]
            Each element is ``(node_index, parent_list)``.
        data : DataMatrix
            ``N × p`` data matrix.

        Returns
        -------
        list[float]
            Local scores in the same order as the input list.
        """
        self._data = data
        return [
            self._lookup(node, frozenset(parents))
            for node, parents in nodes_and_parents
        ]

    def warm_cache(self, dag: AdjacencyMatrix, data: DataMatrix) -> None:
        """Pre-populate the cache with every local score in *dag*.

        Useful before launching parallel mutation operators so that all
        baseline local scores are already cached.

        Parameters
        ----------
        dag : AdjacencyMatrix
            ``n × n`` adjacency matrix.
        data : DataMatrix
            ``N × p`` data matrix.
        """
        self._data = data
        n = dag.shape[0]
        for j in range(n):
            parents = frozenset(int(i) for i in np.where(dag[:, j])[0])
            self._lookup(j, parents)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the local-score cache and reset hit/miss counters."""
        self._cached_local.cache_clear()
        with self._stats_lock:
            self._hits = 0
            self._misses = 0

    @property
    def cache_info(self) -> CacheStats:
        """Return a :class:`CacheStats` snapshot of cache performance.

        Returns
        -------
        CacheStats
            Dataclass with *hits*, *misses*, *hit_rate*, *current_size*,
            *max_size*, and *memory_estimate_bytes*.
        """
        lru_info = self._cached_local.cache_info()
        with self._stats_lock:
            hits = self._hits
            misses = self._misses
        total = hits + misses
        return CacheStats(
            hits=hits,
            misses=misses,
            hit_rate=hits / total if total > 0 else 0.0,
            current_size=lru_info.currsize,
            max_size=self._max_cache_size,
            memory_estimate_bytes=lru_info.currsize * _BYTES_PER_ENTRY,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup(self, node: int, parents: frozenset[int]) -> float:
        """Query the LRU cache and update manual hit/miss counters."""
        self._miss_flag = False
        result = self._cached_local(node, parents)

        with self._stats_lock:
            if self._miss_flag:
                self._misses += 1
            else:
                self._hits += 1

        return result


# ---------------------------------------------------------------------------
# ParentSetCache
# ---------------------------------------------------------------------------


class ParentSetCache:
    """Cache for decomposed ``score(node, parent_set)`` evaluations.

    Uses an LRU eviction policy keyed by ``(node, frozenset(parents))``.

    Parameters
    ----------
    score_fn : callable
        Function ``(node: int, parents: list[int], data) -> float``.
    max_size : int
        Maximum number of cached entries.
    """

    def __init__(
        self,
        score_fn: callable,  # type: ignore[valid-type]
        max_size: int = 16384,
    ) -> None:
        self._score_fn = score_fn
        self._max_size = max_size
        self._hits: int = 0
        self._misses: int = 0
        self._lock = threading.Lock()

        @lru_cache(maxsize=max_size)
        def _cached(node: int, parents: frozenset[int]) -> float:
            self._miss_sentinel = True
            return self._score_fn(node, sorted(parents), self._data)

        self._cached = _cached
        self._data: Optional[DataMatrix] = None
        self._miss_sentinel: bool = False

    def score_family(
        self, node: int, parents: list[int], data: DataMatrix
    ) -> float:
        """Return cached local score for *node* given *parents*.

        Parameters
        ----------
        node : int
            Child node index.
        parents : list[int]
            Parent node indices.
        data : DataMatrix
            ``N × p`` data matrix.

        Returns
        -------
        float
            Local score.
        """
        self._data = data
        key = frozenset(parents)
        self._miss_sentinel = False
        result = self._cached(node, key)
        with self._lock:
            if self._miss_sentinel:
                self._misses += 1
            else:
                self._hits += 1
        return result

    def precompute_all(
        self, data: DataMatrix, max_parents: int = 3
    ) -> None:
        """Precompute scores for all parent sets up to *max_parents* size.

        Parameters
        ----------
        data : DataMatrix
            ``N × p`` data matrix.
        max_parents : int
            Maximum parent set size to enumerate.
        """
        from itertools import combinations

        self._data = data
        p = data.shape[1]
        for node in range(p):
            candidates = [i for i in range(p) if i != node]
            for size in range(max_parents + 1):
                for pa in combinations(candidates, size):
                    self.score_family(node, list(pa), data)

    @property
    def hit_rate(self) -> float:
        """Fraction of queries answered from cache."""
        with self._lock:
            total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def memory_usage(self) -> int:
        """Estimated memory in bytes."""
        info = self._cached.cache_info()
        return info.currsize * _BYTES_PER_ENTRY

    @property
    def stats(self) -> CacheStats:
        """Return cache performance statistics."""
        info = self._cached.cache_info()
        with self._lock:
            hits = self._hits
            misses = self._misses
        total = hits + misses
        return CacheStats(
            hits=hits,
            misses=misses,
            hit_rate=hits / total if total > 0 else 0.0,
            current_size=info.currsize,
            max_size=self._max_size,
            memory_estimate_bytes=info.currsize * _BYTES_PER_ENTRY,
        )

    def clear(self) -> None:
        """Clear cache and reset statistics."""
        self._cached.cache_clear()
        with self._lock:
            self._hits = 0
            self._misses = 0


# ---------------------------------------------------------------------------
# DecomposableCachedScore
# ---------------------------------------------------------------------------


class DecomposableCachedScore(ScoreFunction):
    """Wrap any decomposable score with parent-set caching and delta scoring.

    Parameters
    ----------
    base_score : DecomposableScore
        Underlying decomposable score.
    max_cache_size : int
        Maximum cache entries.
    """

    def __init__(
        self,
        base_score: DecomposableScore,
        max_cache_size: int = 16384,
    ) -> None:
        self._base = base_score
        self._cache = ParentSetCache(
            score_fn=base_score.local_score,
            max_size=max_cache_size,
        )

    def score(self, dag: AdjacencyMatrix, data: DataMatrix) -> QualityScore:
        """Compute full DAG score as sum of cached local scores."""
        n = dag.shape[0]
        total = 0.0
        for j in range(n):
            parents = [int(i) for i in np.where(dag[:, j])[0]]
            total += self._cache.score_family(j, parents, data)
        return total

    def delta_score(
        self,
        dag: AdjacencyMatrix,
        edge_add: tuple[int, int],
        data: DataMatrix,
    ) -> float:
        """Score change from adding an edge.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Current adjacency matrix (before adding the edge).
        edge_add : tuple[int, int]
            ``(source, target)`` edge to add.
        data : DataMatrix
            Data matrix.

        Returns
        -------
        float
            ``new_local_score(target) - old_local_score(target)``.
        """
        src, tgt = edge_add
        old_parents = [int(i) for i in np.where(dag[:, tgt])[0]]
        new_parents = sorted(set(old_parents) | {src})
        old_val = self._cache.score_family(tgt, old_parents, data)
        new_val = self._cache.score_family(tgt, new_parents, data)
        return new_val - old_val

    def delta_score_remove(
        self,
        dag: AdjacencyMatrix,
        edge_remove: tuple[int, int],
        data: DataMatrix,
    ) -> float:
        """Score change from removing an edge.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Current adjacency matrix (before removing the edge).
        edge_remove : tuple[int, int]
            ``(source, target)`` edge to remove.
        data : DataMatrix
            Data matrix.

        Returns
        -------
        float
            ``new_local_score(target) - old_local_score(target)``.
        """
        src, tgt = edge_remove
        old_parents = [int(i) for i in np.where(dag[:, tgt])[0]]
        new_parents = sorted(set(old_parents) - {src})
        old_val = self._cache.score_family(tgt, old_parents, data)
        new_val = self._cache.score_family(tgt, new_parents, data)
        return new_val - old_val

    @property
    def cache(self) -> ParentSetCache:
        """Access the underlying parent-set cache."""
        return self._cache
