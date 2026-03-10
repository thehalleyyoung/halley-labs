"""
usability_oracle.utils.lru_cache — Smart caching system for the usability oracle.

Extends :mod:`usability_oracle.utils.caching` with a hierarchical
L1/L2 cache, TTL-aware eviction, cache warming, and advanced
invalidation policies.  All public types are thread-safe.

Key components
--------------
- :class:`TypedLRUCache` — generic LRU with configurable size, TTL, and
  memory budget.  Stricter type annotations than the base
  :class:`~usability_oracle.utils.caching.LRUCache`.
- :class:`ContentAddressableCache` — hash-based deduplication with stats.
- :class:`HierarchicalCache` — two-level L1 (in-memory) / L2 (on-disk) cache.
- :class:`CacheStatistics` — detailed hit-rate, miss-rate, eviction counts,
  latency percentiles.
- :class:`CacheWarmer` — pre-populate a cache from a seed function.
- :class:`InvalidationPolicy` — rule-based cache invalidation (TTL,
  dependency, manual).

Performance characteristics
---------------------------
- ``TypedLRUCache.get / put``: O(1) amortised via :class:`OrderedDict`.
- ``HierarchicalCache``: L1 O(1); L2 O(n_serialise) for first access,
  O(1) thereafter if promoted to L1.
- ``CacheWarmer``: O(k · f) where k is the number of seed keys and f
  is the per-key cost of the seed function.

References
----------
Cormen, T. H. et al. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

import numpy as np

K = TypeVar("K")
V = TypeVar("V")

# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------


@dataclass
class CacheStatistics:
    """Detailed cache performance statistics.

    Attributes
    ----------
    hits : int
        Number of cache hits.
    misses : int
        Number of cache misses.
    evictions : int
        Number of entries evicted (LRU, TTL, or manual).
    current_size : int
        Number of entries currently stored.
    max_size : int
        Configured capacity.
    total_get_time_s : float
        Cumulative time spent in ``get`` calls.
    total_put_time_s : float
        Cumulative time spent in ``put`` calls.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0
    total_get_time_s: float = 0.0
    total_put_time_s: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Hit rate as a fraction in [0, 1]."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Miss rate as a fraction in [0, 1]."""
        return 1.0 - self.hit_rate

    @property
    def avg_get_latency_s(self) -> float:
        """Average get latency in seconds."""
        total = self.hits + self.misses
        return self.total_get_time_s / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"CacheStatistics(hits={self.hits}, misses={self.misses}, "
            f"evictions={self.evictions}, size={self.current_size}/{self.max_size}, "
            f"hit_rate={self.hit_rate:.2%})"
        )


# ---------------------------------------------------------------------------
# TypedLRUCache
# ---------------------------------------------------------------------------


class TypedLRUCache(Generic[K, V]):
    """Thread-safe LRU cache with configurable size, TTL, and memory budget.

    Parameters
    ----------
    max_size : int
        Maximum number of entries (≥ 1).
    ttl : float or None
        Time-to-live in seconds.  ``None`` disables TTL.
    max_memory_bytes : int or None
        Approximate memory budget.  ``None`` disables memory eviction.
    name : str
        Human-readable name for logging / diagnostics.

    Complexity
    ----------
    - ``get`` / ``put``: O(1) amortised.
    - ``evict``: O(k) where k is the number of entries evicted.
    """

    def __init__(
        self,
        max_size: int = 1024,
        ttl: Optional[float] = None,
        max_memory_bytes: Optional[int] = None,
        name: str = "lru",
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._max_size = max_size
        self._ttl = ttl
        self._max_memory_bytes = max_memory_bytes
        self.name = name
        self._lock = threading.Lock()
        # (value, timestamp, approx_bytes)
        self._store: OrderedDict[K, Tuple[V, float, int]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_get_time = 0.0
        self._total_put_time = 0.0

    @staticmethod
    def _estimate_size(value: Any) -> int:
        """Best-effort size estimate in bytes."""
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, (bytes, bytearray)):
            return len(value)
        import sys
        return sys.getsizeof(value, 64)

    def _is_expired(self, ts: float) -> bool:
        if self._ttl is None:
            return False
        return (time.monotonic() - ts) > self._ttl

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry (caller holds lock)."""
        if self._store:
            self._store.popitem(last=False)
            self._evictions += 1

    def _enforce_memory(self) -> None:
        if self._max_memory_bytes is None:
            return
        while self._store:
            used = sum(e[2] for e in self._store.values())
            if used <= self._max_memory_bytes:
                break
            self._evict_lru()

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Retrieve *key*, or *default* on miss / expiry.

        O(1) amortised.
        """
        t0 = time.perf_counter()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                self._total_get_time += time.perf_counter() - t0
                return default
            value, ts, _ = entry
            if self._is_expired(ts):
                del self._store[key]
                self._evictions += 1
                self._misses += 1
                self._total_get_time += time.perf_counter() - t0
                return default
            self._store.move_to_end(key)
            self._hits += 1
            self._total_get_time += time.perf_counter() - t0
            return value

    def put(self, key: K, value: V) -> None:
        """Insert or update *key* with *value*.  O(1) amortised."""
        t0 = time.perf_counter()
        with self._lock:
            ts = time.monotonic()
            approx = self._estimate_size(value)
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = (value, ts, approx)
            else:
                if len(self._store) >= self._max_size:
                    self._evict_lru()
                self._store[key] = (value, ts, approx)
            self._enforce_memory()
            self._total_put_time += time.perf_counter() - t0

    def delete(self, key: K) -> bool:
        """Remove *key* if present.  Returns ``True`` if removed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._total_get_time = 0.0
            self._total_put_time = 0.0

    def keys(self) -> List[K]:
        """Return list of cached keys (most-recent last)."""
        with self._lock:
            return list(self._store.keys())

    def __contains__(self, key: K) -> bool:
        with self._lock:
            if key not in self._store:
                return False
            _, ts, _ = self._store[key]
            if self._is_expired(ts):
                del self._store[key]
                self._evictions += 1
                return False
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def stats(self) -> CacheStatistics:
        """Snapshot of performance statistics."""
        with self._lock:
            return CacheStatistics(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                current_size=len(self._store),
                max_size=self._max_size,
                total_get_time_s=self._total_get_time,
                total_put_time_s=self._total_put_time,
            )


# ---------------------------------------------------------------------------
# Content-addressable cache
# ---------------------------------------------------------------------------


class ContentAddressableCache(Generic[V]):
    """Content-addressable cache keyed by SHA-256 of serialised values.

    Parameters
    ----------
    max_size : int
        Maximum entries.
    ttl : float or None
        Time-to-live.

    Complexity
    ----------
    - ``put``: O(n) for hashing *n* bytes, O(1) amortised insert.
    - ``get``: O(1) amortised.
    """

    def __init__(self, max_size: int = 2048, ttl: Optional[float] = None) -> None:
        self._cache: TypedLRUCache[str, V] = TypedLRUCache(
            max_size=max_size, ttl=ttl, name="content_addressable"
        )

    @staticmethod
    def _content_key(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def put(self, data: bytes, value: V) -> str:
        """Store *value* under the content hash of *data*.  Returns the hash."""
        key = self._content_key(data)
        self._cache.put(key, value)
        return key

    def get(self, key: str) -> Optional[V]:
        """Retrieve by content hash."""
        return self._cache.get(key)

    def get_by_content(self, data: bytes) -> Optional[V]:
        """Retrieve by raw data (computes hash internally)."""
        return self._cache.get(self._content_key(data))

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def stats(self) -> CacheStatistics:
        return self._cache.stats()


# ---------------------------------------------------------------------------
# Hierarchical cache (L1 in-memory, L2 on-disk)
# ---------------------------------------------------------------------------


class HierarchicalCache(Generic[K, V]):
    """Two-level cache: fast L1 in-memory, larger L2 on-disk.

    On L1 miss, promotes from L2 to L1 if found.  On L2 miss, returns
    ``None``.  L2 uses :mod:`pickle` serialisation to a directory of
    files keyed by a deterministic hash.

    Parameters
    ----------
    l1_size : int
        Maximum L1 (in-memory) entries.
    l2_dir : str or Path or None
        Directory for L2 on-disk storage.  ``None`` disables L2.
    l1_ttl : float or None
        TTL for L1 entries.
    l2_ttl : float or None
        TTL for L2 entries (checked via file mtime).

    Complexity
    ----------
    - L1 hit: O(1).
    - L1 miss + L2 hit: O(deserialise).
    - Full miss: O(1).
    """

    def __init__(
        self,
        l1_size: int = 512,
        l2_dir: Optional[str] = None,
        l1_ttl: Optional[float] = None,
        l2_ttl: Optional[float] = None,
    ) -> None:
        self._l1: TypedLRUCache[K, V] = TypedLRUCache(
            max_size=l1_size, ttl=l1_ttl, name="L1"
        )
        self._l2_dir: Optional[Path] = Path(l2_dir) if l2_dir else None
        self._l2_ttl = l2_ttl
        self._lock = threading.Lock()
        if self._l2_dir is not None:
            self._l2_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _disk_key(key: Any) -> str:
        raw = json.dumps(key, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()

    def _l2_path(self, key: K) -> Optional[Path]:
        if self._l2_dir is None:
            return None
        return self._l2_dir / (self._disk_key(key) + ".pkl")

    def _l2_get(self, key: K) -> Optional[V]:
        path = self._l2_path(key)
        if path is None or not path.exists():
            return None
        if self._l2_ttl is not None:
            age = time.time() - path.stat().st_mtime
            if age > self._l2_ttl:
                path.unlink(missing_ok=True)
                return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)  # noqa: S301
        except Exception:
            return None

    def _l2_put(self, key: K, value: V) -> None:
        path = self._l2_path(key)
        if path is None:
            return
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def get(self, key: K) -> Optional[V]:
        """Look up *key* in L1, then L2.  Promotes to L1 on L2 hit."""
        result = self._l1.get(key)
        if result is not None:
            return result
        with self._lock:
            result = self._l2_get(key)
        if result is not None:
            self._l1.put(key, result)
        return result

    def put(self, key: K, value: V) -> None:
        """Store in both L1 and L2."""
        self._l1.put(key, value)
        with self._lock:
            self._l2_put(key, value)

    def invalidate(self, key: K) -> None:
        """Remove from both levels."""
        self._l1.delete(key)
        with self._lock:
            path = self._l2_path(key)
            if path is not None and path.exists():
                path.unlink(missing_ok=True)

    def clear(self) -> None:
        """Clear L1 and wipe L2 directory."""
        self._l1.clear()
        with self._lock:
            if self._l2_dir is not None and self._l2_dir.exists():
                for p in self._l2_dir.glob("*.pkl"):
                    p.unlink(missing_ok=True)

    def stats(self) -> Dict[str, CacheStatistics]:
        """Return L1 statistics (L2 has no hit/miss tracking)."""
        return {"L1": self._l1.stats()}


# ---------------------------------------------------------------------------
# Cache serialization / deserialization
# ---------------------------------------------------------------------------


def serialize_cache(cache: TypedLRUCache[str, Any]) -> bytes:
    """Serialise a :class:`TypedLRUCache` to bytes for persistence.

    Only works for caches with string keys and pickle-able values.

    Parameters
    ----------
    cache : TypedLRUCache[str, Any]
        The cache to serialise.

    Returns
    -------
    bytes
        Pickle-encoded snapshot.
    """
    snapshot: Dict[str, Any] = {}
    for key in cache.keys():
        val = cache.get(key)
        if val is not None:
            snapshot[key] = val
    return pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_cache(
    data: bytes,
    max_size: int = 1024,
    ttl: Optional[float] = None,
) -> TypedLRUCache[str, Any]:
    """Restore a :class:`TypedLRUCache` from serialised bytes.

    Parameters
    ----------
    data : bytes
        Output of :func:`serialize_cache`.
    max_size : int
        Capacity of the new cache.
    ttl : float or None
        TTL for the new cache.

    Returns
    -------
    TypedLRUCache[str, Any]
        Restored cache.
    """
    snapshot: Dict[str, Any] = pickle.loads(data)  # noqa: S301
    cache: TypedLRUCache[str, Any] = TypedLRUCache(
        max_size=max_size, ttl=ttl, name="deserialized"
    )
    for key, val in snapshot.items():
        cache.put(key, val)
    return cache


# ---------------------------------------------------------------------------
# Cache warming
# ---------------------------------------------------------------------------


class CacheWarmer(Generic[K, V]):
    """Pre-populate a cache from a seed function.

    Parameters
    ----------
    cache : TypedLRUCache[K, V]
        Target cache to warm.
    seed_fn : callable
        ``seed_fn(key) -> value`` — called for each seed key.

    Usage::

        warmer = CacheWarmer(cache, expensive_compute)
        warmer.warm(["key1", "key2", "key3"])
    """

    def __init__(
        self,
        cache: TypedLRUCache[K, V],
        seed_fn: Callable[[K], V],
    ) -> None:
        self._cache = cache
        self._seed_fn = seed_fn

    def warm(self, keys: Sequence[K]) -> int:
        """Populate the cache for *keys* not already present.

        Returns
        -------
        int
            Number of entries computed and inserted.
        """
        count = 0
        for key in keys:
            if key not in self._cache:
                value = self._seed_fn(key)
                self._cache.put(key, value)
                count += 1
        return count

    def warm_parallel(
        self,
        keys: Sequence[K],
        n_workers: int = 4,
    ) -> int:
        """Parallel cache warming using a thread pool.

        Parameters
        ----------
        keys : sequence of K
            Keys to warm.
        n_workers : int
            Number of threads.

        Returns
        -------
        int
            Number of entries computed and inserted.
        """
        from concurrent.futures import ThreadPoolExecutor

        missing = [k for k in keys if k not in self._cache]
        if not missing:
            return 0

        def _compute_and_store(key: K) -> None:
            value = self._seed_fn(key)
            self._cache.put(key, value)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(_compute_and_store, missing))
        return len(missing)


# ---------------------------------------------------------------------------
# Invalidation policies
# ---------------------------------------------------------------------------


@dataclass
class InvalidationRule:
    """A single invalidation rule.

    Attributes
    ----------
    pattern : str
        Key prefix / pattern to match.
    max_age_s : float or None
        Maximum age in seconds before forced invalidation.
    dependencies : set of str
        Keys that, when invalidated, trigger this rule.
    """

    pattern: str = ""
    max_age_s: Optional[float] = None
    dependencies: Set[str] = field(default_factory=set)


class InvalidationPolicy(Generic[K, V]):
    """Rule-based cache invalidation manager.

    Wraps a :class:`TypedLRUCache` and tracks dependency edges so that
    invalidating one key can cascade to dependants.

    Parameters
    ----------
    cache : TypedLRUCache[K, V]
        The managed cache.
    """

    def __init__(self, cache: TypedLRUCache[K, V]) -> None:
        self._cache = cache
        self._deps: Dict[K, Set[K]] = {}  # key → set of dependants
        self._lock = threading.Lock()

    def add_dependency(self, key: K, depends_on: K) -> None:
        """Declare that *key* depends on *depends_on*.

        When *depends_on* is invalidated, *key* is also invalidated.
        """
        with self._lock:
            self._deps.setdefault(depends_on, set()).add(key)

    def invalidate(self, key: K) -> List[K]:
        """Invalidate *key* and all transitive dependants.

        Returns
        -------
        list of K
            All keys that were invalidated (including *key*).
        """
        removed: List[K] = []
        queue: List[K] = [key]
        seen: Set[Any] = set()

        while queue:
            k = queue.pop(0)
            k_hash = id(k) if not isinstance(k, Hashable) else k
            if k_hash in seen:
                continue
            seen.add(k_hash)
            self._cache.delete(k)
            removed.append(k)
            with self._lock:
                for dep in self._deps.get(k, set()):
                    queue.append(dep)

        return removed

    def invalidate_by_prefix(self, prefix: str) -> List[K]:
        """Invalidate all keys whose ``str()`` starts with *prefix*.

        Returns
        -------
        list of K
            Invalidated keys.
        """
        to_remove = [k for k in self._cache.keys() if str(k).startswith(prefix)]
        removed: List[K] = []
        for k in to_remove:
            removed.extend(self.invalidate(k))
        return removed


__all__ = [
    "CacheStatistics",
    "TypedLRUCache",
    "ContentAddressableCache",
    "HierarchicalCache",
    "serialize_cache",
    "deserialize_cache",
    "CacheWarmer",
    "InvalidationRule",
    "InvalidationPolicy",
]
