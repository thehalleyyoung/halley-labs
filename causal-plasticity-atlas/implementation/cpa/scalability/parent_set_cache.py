"""Parent-set score caching with LRU eviction.

Caches local scores for (node, parent-set) pairs to avoid redundant
computation during structure search, with bounded memory via LRU.
Includes a two-tier (L1/L2) cache for hot/cold separation.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import combinations
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    Optional,
    Tuple,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CacheStats:
    """Aggregate statistics for a :class:`ParentSetCache`."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0


# ---------------------------------------------------------------------------
# ParentSetCache – LRU
# ---------------------------------------------------------------------------

class ParentSetCache:
    """LRU cache for parent-set local scores.

    Provides O(1) amortised ``get`` / ``put`` backed by a Python
    :class:`OrderedDict`.  When the cache exceeds *max_size*, the
    least-recently-used entry is evicted.

    Parameters
    ----------
    max_size : int
        Maximum number of entries before eviction.
    n_nodes : int or None
        Optional hint for the number of graph nodes (used by
        ``precompute``).
    """

    def __init__(
        self,
        max_size: int = 100_000,
        n_nodes: Optional[int] = None,
    ) -> None:
        self._max_size = max_size
        self._n_nodes = n_nodes
        self._cache: OrderedDict[Tuple[int, FrozenSet[int]], float] = (
            OrderedDict()
        )
        self._stats = CacheStats(max_size=max_size)

    # -- core operations ---------------------------------------------------

    @staticmethod
    def _cache_key(
        node: int, parents: FrozenSet[int]
    ) -> Tuple[int, FrozenSet[int]]:
        """Create a hashable key from *node* and *parents*."""
        return (node, parents)

    def get(
        self, node: int, parents: FrozenSet[int]
    ) -> Optional[float]:
        """Look up the cached score for (*node*, *parents*).

        Promotes the entry to most-recently-used on access.
        """
        key = self._cache_key(node, parents)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._stats.hits += 1
            return self._cache[key]
        self._stats.misses += 1
        return None

    def put(
        self, node: int, parents: FrozenSet[int], score: float
    ) -> None:
        """Insert or update the score for (*node*, *parents*)."""
        key = self._cache_key(node, parents)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = score
        else:
            self._cache[key] = score
            self._stats.size += 1
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._stats.evictions += 1
                self._stats.size -= 1

    def get_or_compute(
        self,
        node: int,
        parents: FrozenSet[int],
        compute_fn: Callable[[], float],
    ) -> float:
        """Return the cached score or compute, cache, and return it."""
        val = self.get(node, parents)
        if val is not None:
            return val
        score = compute_fn()
        self.put(node, parents, score)
        return score

    # -- invalidation ------------------------------------------------------

    def invalidate_node(self, node: int) -> int:
        """Evict all entries whose key mentions *node*.

        Returns the number of evicted entries.
        """
        to_remove = [
            k
            for k in self._cache
            if k[0] == node or node in k[1]
        ]
        for k in to_remove:
            del self._cache[k]
        evicted = len(to_remove)
        self._stats.size -= evicted
        self._stats.evictions += evicted
        return evicted

    def invalidate_parent(self, parent: int) -> int:
        """Evict all entries that include *parent* in the parent set."""
        to_remove = [k for k in self._cache if parent in k[1]]
        for k in to_remove:
            del self._cache[k]
        evicted = len(to_remove)
        self._stats.size -= evicted
        self._stats.evictions += evicted
        return evicted

    def clear(self) -> None:
        """Remove all cached entries."""
        self._cache.clear()
        self._stats = CacheStats(max_size=self._max_size)

    # -- statistics --------------------------------------------------------

    def stats(self) -> CacheStats:
        """Return current cache statistics."""
        self._stats.size = len(self._cache)
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=self._stats.evictions,
            size=len(self._cache),
            max_size=self._max_size,
        )

    def hit_rate(self) -> float:
        """Return fraction of accesses that were cache hits."""
        total = self._stats.hits + self._stats.misses
        return self._stats.hits / total if total > 0 else 0.0

    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        # Each entry ≈ key overhead + float value
        entry_size = sys.getsizeof(0.0) + 64  # rough per-entry overhead
        return len(self._cache) * entry_size + sys.getsizeof(self._cache)

    # -- precomputation ----------------------------------------------------

    def precompute(
        self,
        node: int,
        max_parents: int,
        score_fn: Callable[[int, FrozenSet[int]], float],
    ) -> int:
        """Eagerly compute and cache all parent sets up to size *max_parents*.

        Parameters
        ----------
        node : int
            Target node.
        max_parents : int
            Maximum parent-set cardinality.
        score_fn : callable
            ``score_fn(node, parents)`` → float.

        Returns
        -------
        int
            Number of parent sets evaluated.
        """
        if self._n_nodes is None:
            raise ValueError("n_nodes must be set to use precompute")

        candidates = [v for v in range(self._n_nodes) if v != node]
        count = 0
        for k in range(max_parents + 1):
            for parents_tuple in combinations(candidates, k):
                parents = frozenset(parents_tuple)
                if self.get(node, parents) is None:
                    score = score_fn(node, parents)
                    self.put(node, parents, score)
                count += 1
        return count

    def precompute_all(
        self,
        score_fn: Callable[[int, FrozenSet[int]], float],
        node: int,
        max_parents: int,
    ) -> int:
        """Alias for :meth:`precompute` with reordered arguments."""
        return self.precompute(node, max_parents, score_fn)

    # -- iteration ---------------------------------------------------------

    def __contains__(
        self, key: Tuple[int, FrozenSet[int]]
    ) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def __iter__(self) -> Iterator[Tuple[int, FrozenSet[int]]]:
        return iter(self._cache)

    def __repr__(self) -> str:
        return (
            f"ParentSetCache(size={len(self._cache)}, "
            f"max={self._max_size}, "
            f"hit_rate={self.hit_rate():.2%})"
        )


# ---------------------------------------------------------------------------
# TieredCache – L1 / L2
# ---------------------------------------------------------------------------

class TieredCache:
    """Two-tier cache with fast L1 and larger L2.

    Hot entries live in L1 (small, fast); cold entries demote to L2.
    On an L2 hit the entry is promoted back to L1.

    Parameters
    ----------
    l1_size : int
        Maximum entries in the hot tier.
    l2_size : int
        Maximum entries in the cold tier.
    """

    def __init__(
        self, l1_size: int = 10_000, l2_size: int = 100_000
    ) -> None:
        self._l1 = ParentSetCache(max_size=l1_size)
        self._l2 = ParentSetCache(max_size=l2_size)

    def get(
        self, node: int, parents: FrozenSet[int]
    ) -> Optional[float]:
        """Check L1 first, then L2.  Promotes on L2 hit."""
        val = self._l1.get(node, parents)
        if val is not None:
            return val
        val = self._l2.get(node, parents)
        if val is not None:
            self._promote(node, parents, val)
            return val
        return None

    def put(
        self, node: int, parents: FrozenSet[int], score: float
    ) -> None:
        """Add to L1 directly.  L1 evictions move to L2."""
        # If L1 is full, demote the LRU entry to L2
        if len(self._l1) >= self._l1._max_size:
            self._evict_l1()
        self._l1.put(node, parents, score)

    def _promote(
        self, node: int, parents: FrozenSet[int], value: float
    ) -> None:
        """Promote an entry from L2 to L1."""
        key = ParentSetCache._cache_key(node, parents)
        if key in self._l2._cache:
            del self._l2._cache[key]
            self._l2._stats.size -= 1
        if len(self._l1) >= self._l1._max_size:
            self._evict_l1()
        self._l1.put(node, parents, value)

    def _evict_l1(self) -> None:
        """Evict LRU entry from L1 into L2."""
        if not self._l1._cache:
            return
        key, value = self._l1._cache.popitem(last=False)
        self._l1._stats.size -= 1
        self._l1._stats.evictions += 1
        self._l2.put(key[0], key[1], value)

    def clear(self) -> None:
        """Clear both tiers."""
        self._l1.clear()
        self._l2.clear()

    def stats(self) -> Dict[str, CacheStats]:
        """Return statistics for both tiers."""
        return {"l1": self._l1.stats(), "l2": self._l2.stats()}

    def __len__(self) -> int:
        return len(self._l1) + len(self._l2)

    def __repr__(self) -> str:
        return (
            f"TieredCache(l1={len(self._l1)}, l2={len(self._l2)})"
        )
