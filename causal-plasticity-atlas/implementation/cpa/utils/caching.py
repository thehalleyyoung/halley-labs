"""Caching infrastructure for the CPA engine.

Provides in-memory LRU caches, disk-backed caching for expensive
computations, and a ``@memoize`` decorator for transparent caching of
function calls involving numpy arrays.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# ===================================================================
# Cache-key helpers
# ===================================================================


def array_cache_key(arr: np.ndarray) -> str:
    """Compute a stable hash key for a numpy array.

    Uses the raw byte buffer plus shape and dtype so that arrays with
    identical content always produce the same key.

    Parameters
    ----------
    arr : np.ndarray
        Array to hash.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    return h.hexdigest()


def _make_key(args: tuple, kwargs: dict) -> str:
    """Create a deterministic cache key from args and kwargs.

    Parameters
    ----------
    args : tuple
        Positional arguments.
    kwargs : dict
        Keyword arguments.

    Returns
    -------
    str
        SHA-256 hex digest.
    """
    h = hashlib.sha256()
    for a in args:
        h.update(_hash_value(a))
    for k in sorted(kwargs):
        h.update(k.encode())
        h.update(_hash_value(kwargs[k]))
    return h.hexdigest()


def _hash_value(v: Any) -> bytes:
    """Return bytes representation for hashing."""
    if isinstance(v, np.ndarray):
        return array_cache_key(v).encode()
    if isinstance(v, (list, tuple)):
        h = hashlib.sha256()
        for item in v:
            h.update(_hash_value(item))
        return h.hexdigest().encode()
    if isinstance(v, dict):
        h = hashlib.sha256()
        for k in sorted(v, key=str):
            h.update(str(k).encode())
            h.update(_hash_value(v[k]))
        return h.hexdigest().encode()
    if isinstance(v, set):
        h = hashlib.sha256()
        for item in sorted(v, key=str):
            h.update(_hash_value(item))
        return h.hexdigest().encode()
    try:
        return pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
    except (pickle.PicklingError, TypeError):
        return str(v).encode()


# ===================================================================
# In-memory LRU cache
# ===================================================================


class LRUCache(Generic[T]):
    """Thread-safe LRU cache with configurable maximum size.

    Parameters
    ----------
    maxsize : int
        Maximum number of entries.  When exceeded the least-recently-used
        entry is evicted.

    Attributes
    ----------
    hits : int
        Number of cache hits since creation.
    misses : int
        Number of cache misses since creation.

    Examples
    --------
    >>> cache: LRUCache[np.ndarray] = LRUCache(maxsize=128)
    >>> cache.put("k1", np.eye(3))
    >>> cache.get("k1")
    array([[1., 0., 0.], ...])
    """

    def __init__(self, maxsize: int = 256) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self._maxsize = maxsize
        self._store: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.Lock()
        self.hits: int = 0
        self.misses: int = 0

    @property
    def maxsize(self) -> int:
        """Maximum number of cached entries."""
        return self._maxsize

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._store

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Retrieve *key* from cache, updating recency.

        Parameters
        ----------
        key : str
            Cache key.
        default : optional
            Value to return on miss.

        Returns
        -------
        T or None
        """
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return default

    def put(self, key: str, value: T) -> None:
        """Insert or update *key* in cache.

        Parameters
        ----------
        key : str
            Cache key.
        value : T
            Value to store.
        """
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = value
            else:
                self._store[key] = value
                if len(self._store) > self._maxsize:
                    self._store.popitem(last=False)

    def invalidate(self, key: str) -> bool:
        """Remove *key* from cache.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            ``True`` if the key was present.
        """
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries and reset counters."""
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that were hits (0.0 if no lookups)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def keys(self) -> list[str]:
        """Return current cache keys in LRU order (oldest first).

        Returns
        -------
        list of str
        """
        with self._lock:
            return list(self._store.keys())

    def __repr__(self) -> str:
        return (
            f"LRUCache(maxsize={self._maxsize}, size={len(self)}, "
            f"hit_rate={self.hit_rate:.2%})"
        )


# ===================================================================
# Disk-backed cache
# ===================================================================


class DiskCache:
    """Persistent disk-backed cache using pickle serialisation.

    Parameters
    ----------
    directory : str or Path
        Directory for cached files.
    max_entries : int
        Maximum number of cached files.  Eviction removes the oldest
        (by modification time).
    ttl_seconds : float, optional
        Time-to-live in seconds.  Entries older than this are considered
        stale on read.

    Examples
    --------
    >>> dc = DiskCache("/tmp/cpa_cache", max_entries=500)
    >>> dc.put("my_key", large_array)
    >>> dc.get("my_key")  # returns large_array or None if evicted
    """

    def __init__(
        self,
        directory: Union[str, Path],
        max_entries: int = 1024,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def _key_path(self, key: str) -> Path:
        safe = hashlib.sha256(key.encode()).hexdigest()
        return self._dir / f"{safe}.pkl"

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a cached value.

        Parameters
        ----------
        key : str
            Cache key.
        default : optional
            Returned on miss or if the entry is expired.

        Returns
        -------
        Any
        """
        path = self._key_path(key)
        if not path.exists():
            return default
        if self._ttl is not None:
            age = time.time() - path.stat().st_mtime
            if age > self._ttl:
                path.unlink(missing_ok=True)
                return default
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return default

    def put(self, key: str, value: Any) -> None:
        """Store a value on disk.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Picklable value to store.
        """
        with self._lock:
            path = self._key_path(key)
            with open(path, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._evict_if_needed()

    def invalidate(self, key: str) -> bool:
        """Remove a cached entry.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            ``True`` if the entry was present.
        """
        path = self._key_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> None:
        """Remove all cached files."""
        with self._lock:
            for p in self._dir.glob("*.pkl"):
                p.unlink()

    def _evict_if_needed(self) -> None:
        files = sorted(self._dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
        while len(files) > self._max_entries:
            files[0].unlink(missing_ok=True)
            files.pop(0)

    @property
    def size(self) -> int:
        """Number of cached entries on disk."""
        return len(list(self._dir.glob("*.pkl")))

    @property
    def total_bytes(self) -> int:
        """Total size of cached files in bytes."""
        return sum(p.stat().st_size for p in self._dir.glob("*.pkl"))

    def __repr__(self) -> str:
        return (
            f"DiskCache(dir={str(self._dir)!r}, entries={self.size}, "
            f"max={self._max_entries})"
        )


# ===================================================================
# Memoisation decorator
# ===================================================================


_DEFAULT_CACHE: LRUCache[Any] = LRUCache(maxsize=1024)


def memoize(
    cache: Optional[LRUCache[Any]] = None,
    key_prefix: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator that memoises function results using an :class:`LRUCache`.

    Handles numpy arrays in arguments by hashing their content rather
    than relying on identity.

    Parameters
    ----------
    cache : LRUCache, optional
        Cache instance.  If ``None``, a module-level default is used.
    key_prefix : str, optional
        Prefix prepended to cache keys for namespacing.

    Returns
    -------
    Callable
        Decorated function.

    Examples
    --------
    >>> @memoize()
    ... def expensive(adj: np.ndarray, alpha: float) -> float:
    ...     return do_stuff(adj, alpha)
    """
    c = cache if cache is not None else _DEFAULT_CACHE

    def decorator(fn: F) -> F:
        prefix = key_prefix or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = prefix + ":" + _make_key(args, kwargs)
            result = c.get(key)
            if result is not None:
                return result
            result = fn(*args, **kwargs)
            c.put(key, result)
            return result

        wrapper.cache = c  # type: ignore[attr-defined]
        wrapper.cache_clear = c.clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
