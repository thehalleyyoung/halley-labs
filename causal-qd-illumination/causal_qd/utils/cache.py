"""Caching utilities: LRU cache, memoization, and score caching.

Provides a generic ``LRUCache`` with hit/miss statistics, a ``cached_property``
descriptor, a ``memoize`` decorator for hashable arguments, and a specialized
``ScoreCache`` for BIC/BDeu score caching keyed by (node, parent_set).
"""

from __future__ import annotations

import functools
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Generic, Hashable, Optional, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


# ---------------------------------------------------------------------------
# LRU Cache with hit/miss stats
# ---------------------------------------------------------------------------

class LRUCache(Generic[K, V]):
    """Least-recently-used cache with a fixed maximum size and hit/miss stats.

    Parameters
    ----------
    max_size:
        Maximum number of entries.  When the cache is full the oldest
        (least-recently-used) entry is evicted.
    """

    def __init__(self, max_size: int = 1024) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        self._store: OrderedDict[K, V] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Return the cached value for *key*, or *default* if absent.

        Accessing a key moves it to the most-recently-used position.
        """
        if key not in self._store:
            self._misses += 1
            return default
        self._hits += 1
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: K, value: V) -> None:
        """Insert or update *key* with *value*.

        If the cache is full the least-recently-used entry is evicted.
        """
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        if len(self._store) >= self._max_size:
            self._store.popitem(last=False)
        self._store[key] = value

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()

    def reset_stats(self) -> None:
        """Reset hit/miss counters to zero."""
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def hits(self) -> int:
        """Number of cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of cache misses."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate in [0, 1].  Returns 0 if no lookups performed."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def max_size(self) -> int:
        """Maximum capacity of the cache."""
        return self._max_size

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return (
            f"LRUCache(max_size={self._max_size}, size={len(self._store)}, "
            f"hits={self._hits}, misses={self._misses})"
        )

    def __getitem__(self, key: K) -> V:
        val = self.get(key)
        if val is None and key not in self._store:
            raise KeyError(key)
        return val  # type: ignore[return-value]

    def __setitem__(self, key: K, value: V) -> None:
        self.put(key, value)

    def __delitem__(self, key: K) -> None:
        if key in self._store:
            del self._store[key]
        else:
            raise KeyError(key)


# ---------------------------------------------------------------------------
# cached_property descriptor
# ---------------------------------------------------------------------------

class cached_property(Generic[V]):
    """A descriptor that caches the result of a property method.

    Similar to ``functools.cached_property`` but works with slotted classes
    and provides a ``cache_clear`` method on the instance.

    Usage::

        class MyClass:
            @cached_property
            def expensive(self) -> int:
                return heavy_computation()

        obj = MyClass()
        obj.expensive       # Computes and caches
        obj.expensive       # Returns cached value
        del obj.expensive   # Clears the cache
    """

    def __init__(self, func: Callable[..., V]) -> None:
        self.func = func
        self.attrname: Optional[str] = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner: type, name: str) -> None:
        self.attrname = name

    def __get__(self, instance: Any, owner: type) -> V:
        if instance is None:
            return self  # type: ignore[return-value]
        if self.attrname is None:
            raise TypeError("Cannot use cached_property without calling __set_name__")
        cache = instance.__dict__
        val = cache.get(self.attrname)
        if val is None:
            val = self.func(instance)
            cache[self.attrname] = val
        return val

    def __delete__(self, instance: Any) -> None:
        if self.attrname and self.attrname in instance.__dict__:
            del instance.__dict__[self.attrname]


# ---------------------------------------------------------------------------
# memoize decorator
# ---------------------------------------------------------------------------

def memoize(maxsize: int = 256) -> Callable:
    """Decorator that memoizes function results with an LRU eviction policy.

    All arguments must be hashable.  The cache is per-function.

    Parameters
    ----------
    maxsize : int
        Maximum number of cached results.

    Usage::

        @memoize(maxsize=128)
        def expensive_func(x: int, y: int) -> int:
            return x ** y
    """

    def decorator(func: Callable) -> Callable:
        cache: LRUCache[Tuple, Any] = LRUCache(max_size=maxsize)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build hashable key
            key = args + tuple(sorted(kwargs.items()))
            result = cache.get(key)
            if result is not None or key in cache:
                return result
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result

        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        wrapper.cache_info = lambda: {  # type: ignore[attr-defined]
            "hits": cache.hits,
            "misses": cache.misses,
            "size": cache.size,
            "maxsize": cache.max_size,
        }
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# ScoreCache
# ---------------------------------------------------------------------------

class ScoreCache:
    """Specialized cache for BIC/BDeu scores keyed by (node, parent_set).

    Each entry maps a ``(node, frozenset_of_parents)`` key to a float score.
    Uses an underlying LRU cache for eviction.

    Parameters
    ----------
    max_size : int
        Maximum number of cached (node, parent_set) → score entries.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._cache: LRUCache[Tuple[int, FrozenSet[int]], float] = LRUCache(max_size=max_size)

    def get(self, node: int, parent_set: FrozenSet[int]) -> Optional[float]:
        """Retrieve the cached score for *(node, parent_set)*, or None."""
        key = (node, parent_set)
        return self._cache.get(key)

    def put(self, node: int, parent_set: FrozenSet[int], score: float) -> None:
        """Store a score for *(node, parent_set)*."""
        key = (node, parent_set)
        self._cache.put(key, score)

    def contains(self, node: int, parent_set: FrozenSet[int]) -> bool:
        """Return True if the cache contains an entry for *(node, parent_set)*."""
        return (node, parent_set) in self._cache

    def clear(self) -> None:
        """Clear all cached scores."""
        self._cache.clear()

    @property
    def hits(self) -> int:
        return self._cache.hits

    @property
    def misses(self) -> int:
        return self._cache.misses

    @property
    def hit_rate(self) -> float:
        return self._cache.hit_rate

    @property
    def size(self) -> int:
        return self._cache.size

    def __repr__(self) -> str:
        return (
            f"ScoreCache(size={self._cache.size}, "
            f"hits={self._cache.hits}, misses={self._cache.misses})"
        )
