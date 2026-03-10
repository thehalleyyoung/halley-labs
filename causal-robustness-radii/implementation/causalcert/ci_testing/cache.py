"""
CI test result caching with warm-start kernel matrices.

Caches CI test results keyed by ``(x, y, conditioning_set, method)`` and
optionally stores intermediate kernel matrices so that tests under
overlapping conditioning sets can share computation.

Features:
- LRU eviction for bounded memory usage
- Warm-start kernel matrix caching for KCI
- Cache invalidation strategies
- Serialization/deserialization
- Cache hit statistics
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from causalcert.types import CITestMethod, CITestResult, NodeId, NodeSet

_CacheKey = tuple[NodeId, NodeId, NodeSet, CITestMethod]
_KernelKey = tuple[NodeId, ...]


class CacheStats:
    """Tracks cache hit/miss statistics.

    Attributes
    ----------
    hits : int
        Number of cache hits.
    misses : int
        Number of cache misses.
    evictions : int
        Number of LRU evictions.
    """

    def __init__(self) -> None:
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    @property
    def total(self) -> int:
        """Total number of lookups."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that were cache hits."""
        if self.total == 0:
            return 0.0
        return self.hits / self.total

    def reset(self) -> None:
        """Reset all counters."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"evictions={self.evictions}, hit_rate={self.hit_rate:.2%})"
        )


class CITestCache:
    """Thread-safe cache for CI test results and kernel matrices.

    Supports warm-starting: when a new CI test is requested, the cache
    checks whether a previously computed kernel matrix can be reused.

    Uses LRU (Least Recently Used) eviction when the cache exceeds
    ``max_size``.

    Parameters
    ----------
    max_size : int
        Maximum number of cached results.
    max_kernel_size : int
        Maximum number of cached kernel matrices.
    store_kernels : bool
        Whether to cache intermediate kernel matrices.
    """

    def __init__(
        self,
        max_size: int = 10_000,
        max_kernel_size: int = 500,
        store_kernels: bool = True,
    ) -> None:
        self.max_size = max_size
        self.max_kernel_size = max_kernel_size
        self.store_kernels = store_kernels

        self._results: OrderedDict[_CacheKey, CITestResult] = OrderedDict()
        self._kernels: OrderedDict[_KernelKey, np.ndarray] = OrderedDict()

        self._lock = threading.Lock()
        self._stats = CacheStats()

    # ------------------------------------------------------------------
    # Result cache
    # ------------------------------------------------------------------

    def get(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        method: CITestMethod,
    ) -> CITestResult | None:
        """Look up a cached result.

        On a hit, moves the entry to the end (most recently used).

        Parameters
        ----------
        x, y : NodeId
            Test endpoints.
        conditioning_set : NodeSet
            Conditioning set.
        method : CITestMethod
            CI test method.

        Returns
        -------
        CITestResult | None
            Cached result or ``None`` on miss.
        """
        key: _CacheKey = (x, y, conditioning_set, method)
        with self._lock:
            result = self._results.get(key)
            if result is not None:
                self._stats.hits += 1
                # Move to end (most recently used)
                self._results.move_to_end(key)
                return result
            # Also check symmetric key (X ⊥ Y | Z == Y ⊥ X | Z)
            sym_key: _CacheKey = (y, x, conditioning_set, method)
            result = self._results.get(sym_key)
            if result is not None:
                self._stats.hits += 1
                self._results.move_to_end(sym_key)
                return result
            self._stats.misses += 1
            return None

    def put(self, result: CITestResult) -> None:
        """Store a CI test result in the cache.

        If the cache is at capacity, evicts the least recently used entry.

        Parameters
        ----------
        result : CITestResult
            Result to cache.
        """
        key: _CacheKey = (
            result.x, result.y, result.conditioning_set, result.method
        )
        with self._lock:
            if key in self._results:
                # Update and move to end
                self._results[key] = result
                self._results.move_to_end(key)
                return

            # Evict if at capacity
            while len(self._results) >= self.max_size:
                self._results.popitem(last=False)
                self._stats.evictions += 1

            self._results[key] = result

    def put_batch(self, results: list[CITestResult]) -> None:
        """Store multiple results at once.

        Parameters
        ----------
        results : list[CITestResult]
            Results to cache.
        """
        for r in results:
            self.put(r)

    # ------------------------------------------------------------------
    # Kernel cache
    # ------------------------------------------------------------------

    def get_kernel(self, node_ids: tuple[NodeId, ...]) -> np.ndarray | None:
        """Retrieve a cached kernel matrix for the given variable set.

        Parameters
        ----------
        node_ids : tuple[NodeId, ...]
            Sorted tuple of node ids used to construct the kernel.

        Returns
        -------
        np.ndarray | None
            Cached kernel matrix or ``None``.
        """
        if not self.store_kernels:
            return None
        with self._lock:
            K = self._kernels.get(node_ids)
            if K is not None:
                self._kernels.move_to_end(node_ids)
                return K
            return None

    def put_kernel(self, node_ids: tuple[NodeId, ...], kernel: np.ndarray) -> None:
        """Cache a kernel matrix.

        Parameters
        ----------
        node_ids : tuple[NodeId, ...]
            Sorted tuple of node ids.
        kernel : np.ndarray
            Kernel matrix.
        """
        if not self.store_kernels:
            return
        with self._lock:
            if node_ids in self._kernels:
                self._kernels[node_ids] = kernel
                self._kernels.move_to_end(node_ids)
                return

            while len(self._kernels) >= self.max_kernel_size:
                self._kernels.popitem(last=False)
                self._stats.evictions += 1

            self._kernels[node_ids] = kernel

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        """Clear all cached results and kernels."""
        with self._lock:
            self._results.clear()
            self._kernels.clear()

    def invalidate_results(self) -> None:
        """Clear only cached results (keep kernels)."""
        with self._lock:
            self._results.clear()

    def invalidate_kernels(self) -> None:
        """Clear only cached kernels (keep results)."""
        with self._lock:
            self._kernels.clear()

    def invalidate_node(self, node: NodeId) -> None:
        """Invalidate all cache entries involving a specific node.

        Parameters
        ----------
        node : NodeId
            The node whose entries should be removed.
        """
        with self._lock:
            keys_to_remove = [
                k for k in self._results
                if k[0] == node or k[1] == node or node in k[2]
            ]
            for k in keys_to_remove:
                del self._results[k]

            kernel_keys_to_remove = [
                k for k in self._kernels if node in k
            ]
            for k in kernel_keys_to_remove:
                del self._kernels[k]

    def invalidate_method(self, method: CITestMethod) -> None:
        """Invalidate all results from a specific method.

        Parameters
        ----------
        method : CITestMethod
            CI test method.
        """
        with self._lock:
            keys_to_remove = [k for k in self._results if k[3] == method]
            for k in keys_to_remove:
                del self._results[k]

    # ------------------------------------------------------------------
    # Statistics and inspection
    # ------------------------------------------------------------------

    @property
    def stats(self) -> CacheStats:
        """Cache hit/miss statistics."""
        return self._stats

    @property
    def size(self) -> int:
        """Number of cached results."""
        return len(self._results)

    @property
    def kernel_size(self) -> int:
        """Number of cached kernel matrices."""
        return len(self._kernels)

    @property
    def memory_estimate_bytes(self) -> int:
        """Rough estimate of cache memory usage in bytes."""
        result_bytes = len(self._results) * 200  # ~200 bytes per result
        kernel_bytes = sum(
            k.nbytes for k in self._kernels.values()
        )
        return result_bytes + kernel_bytes

    def keys(self) -> list[_CacheKey]:
        """Return all cached result keys."""
        with self._lock:
            return list(self._results.keys())

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the results cache to a dictionary.

        Kernel matrices are not serialized (too large).

        Returns
        -------
        dict
            Serializable dictionary.
        """
        with self._lock:
            entries = []
            for key, result in self._results.items():
                entries.append({
                    "x": result.x,
                    "y": result.y,
                    "conditioning_set": sorted(result.conditioning_set),
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "method": result.method.value,
                    "reject": result.reject,
                    "alpha": result.alpha,
                })
            return {
                "max_size": self.max_size,
                "entries": entries,
                "stats": {
                    "hits": self._stats.hits,
                    "misses": self._stats.misses,
                    "evictions": self._stats.evictions,
                },
            }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CITestCache:
        """Deserialize a cache from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary produced by :meth:`to_dict`.

        Returns
        -------
        CITestCache
            Restored cache (without kernels).
        """
        cache = cls(max_size=d.get("max_size", 10_000))
        for entry in d.get("entries", []):
            result = CITestResult(
                x=entry["x"],
                y=entry["y"],
                conditioning_set=frozenset(entry["conditioning_set"]),
                statistic=entry["statistic"],
                p_value=entry["p_value"],
                method=CITestMethod(entry["method"]),
                reject=entry["reject"],
                alpha=entry.get("alpha", 0.05),
            )
            cache.put(result)
        if "stats" in d:
            cache._stats.hits = d["stats"].get("hits", 0)
            cache._stats.misses = d["stats"].get("misses", 0)
            cache._stats.evictions = d["stats"].get("evictions", 0)
        return cache

    def __repr__(self) -> str:
        return (
            f"CITestCache(results={self.size}, kernels={self.kernel_size}, "
            f"max_size={self.max_size}, {self._stats})"
        )
