"""
usability_oracle.utils.caching — Caching infrastructure for the usability oracle.

Provides configurable LRU caches with TTL support, content-addressable
storage, memoisation decorators, and domain-specific caches for parsed
accessibility trees, computed policies, and cognitive cost results.

All caches are thread-safe (guarded by :class:`threading.Lock`) and expose
hit/miss/eviction statistics via :func:`cache_stats`.

Performance characteristics
---------------------------
- **LRUCache**: O(1) amortised get/put via :class:`collections.OrderedDict`.
- **ContentAddressableStore**: O(n) hash on first store, O(1) lookup.
- **MemoizedFunction**: O(n_args) hash per call, O(1) lookup thereafter.
- **TreeCache / PolicyCache / CostCache**: thin wrappers with domain-typed keys.

References
----------
Cormen, T. H. et al. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
    — LRU replacement policy, hash-table amortised analysis.
"""

from __future__ import annotations

import functools
import hashlib
import json
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Hashable, Optional, Tuple, TypeVar

import numpy as np

K = TypeVar("K")
V = TypeVar("V")

# ---------------------------------------------------------------------------
# Global registry for cache-stats aggregation
# ---------------------------------------------------------------------------

_ALL_CACHES: list["LRUCache[Any, Any]"] = []
_REGISTRY_LOCK = threading.Lock()


def _register_cache(cache: "LRUCache[Any, Any]") -> None:
    """Register a cache instance for global statistics and clearing."""
    with _REGISTRY_LOCK:
        _ALL_CACHES.append(cache)


# ---------------------------------------------------------------------------
# CacheStats
# ---------------------------------------------------------------------------

@dataclass
class CacheStats:
    """Snapshot of cache hit / miss / eviction counters.

    Attributes
    ----------
    hits : int
        Number of cache hits.
    misses : int
        Number of cache misses.
    evictions : int
        Number of entries evicted (LRU or TTL).
    current_size : int
        Number of entries currently stored.
    max_size : int
        Configured maximum number of entries.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Hit rate as a fraction in [0, 1]."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"evictions={self.evictions}, size={self.current_size}/{self.max_size}, "
            f"hit_rate={self.hit_rate:.2%})"
        )


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------

class LRUCache(Generic[K, V]):
    """Thread-safe Least Recently Used cache with optional TTL.

    Parameters
    ----------
    max_size : int
        Maximum number of entries.  When exceeded the least-recently-used
        entry is evicted.  Must be ≥ 1.
    ttl : float or None
        Time-to-live in seconds.  Entries older than *ttl* are treated as
        expired on access and silently evicted.  ``None`` disables TTL.
    max_memory_bytes : int or None
        Approximate memory budget.  When the estimated memory of stored
        values exceeds this limit, the least-recently-used entries are
        evicted until usage drops below the limit.  ``None`` disables
        memory-based eviction.

    Complexity
    ----------
    - ``get`` / ``put``: O(1) amortised (OrderedDict move_to_end).
    - ``evict``: O(k) where k is the number of evicted entries.

    Thread safety
    -------------
    All public methods acquire an internal :class:`threading.Lock`.
    """

    def __init__(
        self,
        max_size: int = 1024,
        ttl: Optional[float] = None,
        max_memory_bytes: Optional[int] = None,
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._max_size = max_size
        self._ttl = ttl
        self._max_memory_bytes = max_memory_bytes
        self._lock = threading.Lock()
        # value  →  (value, timestamp, approx_bytes)
        self._store: OrderedDict[K, Tuple[V, float, int]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        _register_cache(self)

    # -- helpers --

    @staticmethod
    def _estimate_size(value: Any) -> int:
        """Best-effort memory estimate in bytes."""
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        return sys.getsizeof(value, 64)

    def _is_expired(self, ts: float) -> bool:
        if self._ttl is None:
            return False
        return (time.monotonic() - ts) > self._ttl

    def _current_memory(self) -> int:
        return sum(entry[2] for entry in self._store.values())

    def _evict_one(self) -> None:
        """Evict the least-recently-used entry (caller holds lock)."""
        if self._store:
            self._store.popitem(last=False)
            self._evictions += 1

    def _enforce_memory_limit(self) -> None:
        """Evict LRU entries until memory usage is within budget."""
        if self._max_memory_bytes is None:
            return
        while self._store and self._current_memory() > self._max_memory_bytes:
            self._evict_one()

    # -- public API --

    def get(self, key: K) -> Optional[V]:
        """Retrieve *key* from the cache, or ``None`` on miss.

        O(1) amortised.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, ts, size_bytes = entry
            if self._is_expired(ts):
                del self._store[key]
                self._evictions += 1
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: K, value: V) -> None:
        """Insert or update *key* with *value*.

        If the cache is full the least-recently-used entry is evicted.
        O(1) amortised.
        """
        with self._lock:
            ts = time.monotonic()
            approx = self._estimate_size(value)
            if key in self._store:
                # Update existing
                self._store.move_to_end(key)
                self._store[key] = (value, ts, approx)
            else:
                if len(self._store) >= self._max_size:
                    self._evict_one()
                self._store[key] = (value, ts, approx)
            self._enforce_memory_limit()

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

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> CacheStats:
        """Return a snapshot of cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                current_size=len(self._store),
                max_size=self._max_size,
            )


# ---------------------------------------------------------------------------
# ContentAddressableStore
# ---------------------------------------------------------------------------

class ContentAddressableStore:
    """Thread-safe content-addressable store keyed by SHA-256 hash.

    Duplicate values (by content hash) are stored only once, enabling
    automatic deduplication of repeated inputs.

    Complexity
    ----------
    - ``store``: O(n) for hashing *n* bytes, O(1) amortised dict insert.
    - ``fetch``: O(1) amortised dict lookup.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        self._cache: LRUCache[str, bytes] = LRUCache(max_size=max_entries)

    @staticmethod
    def _hash(data: bytes) -> str:
        """SHA-256 hex digest of *data*."""
        return hashlib.sha256(data).hexdigest()

    def store(self, data: bytes) -> str:
        """Store *data* and return its content hash.

        If *data* with the same hash is already stored the existing
        entry is returned without duplication.

        Parameters
        ----------
        data : bytes
            Arbitrary byte payload.

        Returns
        -------
        str
            SHA-256 hex digest identifying the stored content.
        """
        key = self._hash(data)
        self._cache.put(key, data)
        return key

    def fetch(self, key: str) -> Optional[bytes]:
        """Retrieve data by content hash, or ``None`` if absent."""
        return self._cache.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def stats(self) -> CacheStats:
        return self._cache.stats()


# ---------------------------------------------------------------------------
# MemoizedFunction decorator
# ---------------------------------------------------------------------------

def _make_key(args: tuple, kwargs: dict) -> str:
    """Produce a deterministic hash key from function arguments.

    Handles numpy arrays, dataclasses, and JSON-serialisable objects.
    Falls back to ``repr()`` for opaque types.
    """
    parts: list[str] = []
    for a in args:
        parts.append(_arg_digest(a))
    for k in sorted(kwargs):
        parts.append(f"{k}={_arg_digest(kwargs[k])}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


def _arg_digest(obj: Any) -> str:
    """Return a stable string representation of *obj*."""
    if isinstance(obj, np.ndarray):
        return hashlib.sha256(obj.tobytes()).hexdigest()
    try:
        return json.dumps(obj, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(obj)


class MemoizedFunction:
    """Decorator that caches function results by argument hash.

    Usage::

        @MemoizedFunction(max_size=512, ttl=60.0)
        def expensive(x, y):
            ...

    Complexity
    ----------
    - First call: O(f) where f is the cost of the wrapped function.
    - Subsequent calls with identical args: O(1) amortised.
    """

    def __init__(
        self,
        fn: Optional[Callable] = None,
        *,
        max_size: int = 1024,
        ttl: Optional[float] = None,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl
        if fn is not None:
            # Used as @MemoizedFunction (without arguments)
            self._fn = fn
            self._cache: LRUCache[str, Any] = LRUCache(
                max_size=self._max_size, ttl=self._ttl
            )
            functools.update_wrapper(self, fn)
        else:
            self._fn = None  # type: ignore[assignment]
            self._cache = None  # type: ignore[assignment]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._fn is None:
            # Called as @MemoizedFunction(max_size=...) — first arg is the fn
            fn = args[0]
            self._fn = fn
            self._cache = LRUCache(max_size=self._max_size, ttl=self._ttl)
            functools.update_wrapper(self, fn)
            return self

        key = _make_key(args, kwargs)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._fn(*args, **kwargs)
        self._cache.put(key, result)
        return result

    def cache_clear(self) -> None:
        """Clear the memoisation cache."""
        if self._cache is not None:
            self._cache.clear()

    def cache_stats(self) -> CacheStats:
        """Return cache statistics."""
        if self._cache is not None:
            return self._cache.stats()
        return CacheStats()


# ---------------------------------------------------------------------------
# Domain-specific caches
# ---------------------------------------------------------------------------

class TreeCache:
    """Cache parsed accessibility trees by content hash.

    Keys are SHA-256 digests of the serialised tree bytes; values are
    the parsed tree objects.

    Complexity
    ----------
    - ``get`` / ``put``: O(1) amortised (delegates to :class:`LRUCache`).
    - ``key_for``: O(n) for hashing *n* bytes of tree data.
    """

    def __init__(self, max_size: int = 256) -> None:
        self._cache: LRUCache[str, Any] = LRUCache(max_size=max_size)

    @staticmethod
    def key_for(tree_bytes: bytes) -> str:
        """Compute content-hash key for raw tree data."""
        return hashlib.sha256(tree_bytes).hexdigest()

    def get(self, tree_bytes: bytes) -> Optional[Any]:
        """Look up a parsed tree by raw-data hash."""
        key = self.key_for(tree_bytes)
        return self._cache.get(key)

    def put(self, tree_bytes: bytes, parsed_tree: Any) -> str:
        """Store a parsed tree keyed by its content hash.

        Returns the hash key.
        """
        key = self.key_for(tree_bytes)
        self._cache.put(key, parsed_tree)
        return key

    def stats(self) -> CacheStats:
        return self._cache.stats()

    def clear(self) -> None:
        self._cache.clear()


class PolicyCache:
    """Cache computed policies keyed by ``(mdp_hash, beta)``.

    Parameters
    ----------
    max_size : int
        Maximum number of cached policies.

    Complexity
    ----------
    - ``get`` / ``put``: O(1) amortised.
    """

    def __init__(self, max_size: int = 512) -> None:
        self._cache: LRUCache[Tuple[str, float], Any] = LRUCache(max_size=max_size)

    @staticmethod
    def _mdp_hash(mdp_data: bytes) -> str:
        return hashlib.sha256(mdp_data).hexdigest()

    def get(self, mdp_data: bytes, beta: float) -> Optional[Any]:
        """Look up a cached policy by MDP data and β value."""
        key = (self._mdp_hash(mdp_data), round(beta, 10))
        return self._cache.get(key)

    def put(self, mdp_data: bytes, beta: float, policy: Any) -> None:
        """Store a computed policy."""
        key = (self._mdp_hash(mdp_data), round(beta, 10))
        self._cache.put(key, policy)

    def stats(self) -> CacheStats:
        return self._cache.stats()

    def clear(self) -> None:
        self._cache.clear()


class CostCache:
    """Cache cognitive cost results keyed by ``(tree_hash, config_hash)``.

    Complexity
    ----------
    - ``get`` / ``put``: O(1) amortised.
    """

    def __init__(self, max_size: int = 1024) -> None:
        self._cache: LRUCache[Tuple[str, str], Any] = LRUCache(max_size=max_size)

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def get(self, tree_data: bytes, config_data: bytes) -> Optional[Any]:
        """Look up cached cost by tree and configuration hashes."""
        key = (self._hash_bytes(tree_data), self._hash_bytes(config_data))
        return self._cache.get(key)

    def put(self, tree_data: bytes, config_data: bytes, result: Any) -> None:
        """Store a cost computation result."""
        key = (self._hash_bytes(tree_data), self._hash_bytes(config_data))
        self._cache.put(key, result)

    def stats(self) -> CacheStats:
        return self._cache.stats()

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Global helpers
# ---------------------------------------------------------------------------

def cache_stats() -> Dict[str, CacheStats]:
    """Return hit/miss/eviction statistics for all registered caches.

    Returns
    -------
    dict[str, CacheStats]
        Mapping from cache identity string to its :class:`CacheStats`.
    """
    with _REGISTRY_LOCK:
        result: Dict[str, CacheStats] = {}
        for i, cache in enumerate(_ALL_CACHES):
            label = f"cache_{i}"
            result[label] = cache.stats()
        return result


def clear_all_caches() -> None:
    """Clear every registered cache and reset their statistics.

    This is useful in test teardown or when reconfiguring the pipeline
    to ensure no stale results leak across runs.
    """
    with _REGISTRY_LOCK:
        for cache in _ALL_CACHES:
            cache.clear()
