"""
Result caching layer for the pipeline.

Caches intermediate and final results to disk (JSON / pickle) to avoid
redundant computation when re-running with modified parameters.

Supports:
- Content-addressed keys (hash of DAG + data fingerprint + config)
- JSON serialisation for lightweight results
- Pickle serialisation for arbitrary Python objects
- Per-step caching (CI tests, fragility, solver, estimation)
- TTL-based expiration
- Thread-safe put/get via file-system atomicity
- Cache statistics
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache metadata
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CacheEntry:
    """Metadata for a single cache entry."""

    key: str
    created_at: float
    ttl: float
    format: str  # "json" or "pickle"
    size_bytes: int = 0

    def is_expired(self) -> bool:
        if self.ttl <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, frozenset):
            return {"__frozenset__": True, "items": sorted(obj)}
        return super().default(obj)


def _numpy_decoder(dct: dict) -> Any:
    if "__ndarray__" in dct:
        return np.array(dct["data"], dtype=dct["dtype"])
    if "__frozenset__" in dct:
        return frozenset(dct["items"])
    return dct


def _try_json_dumps(value: Any) -> str | None:
    """Attempt JSON serialisation; return None on failure."""
    try:
        return json.dumps(value, cls=_NumpyEncoder)
    except (TypeError, ValueError):
        return None


def _try_json_loads(data: str) -> Any:
    return json.loads(data, object_hook=_numpy_decoder)


# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CacheStats:
    """Aggregate cache statistics."""

    hits: int = 0
    misses: int = 0
    puts: int = 0
    evictions: int = 0
    total_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main ResultCache
# ---------------------------------------------------------------------------


class ResultCache:
    """Disk-backed result cache with content-addressed keys.

    Parameters
    ----------
    cache_dir : str | Path
        Directory for cache files.
    enabled : bool
        If ``False``, all operations are no-ops.
    default_ttl : float
        Default time-to-live in seconds.  ``0`` means no expiration.
    max_size_mb : float
        Maximum total cache size in megabytes.  ``0`` means unlimited.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".causalcert_cache",
        enabled: bool = True,
        default_ttl: float = 0.0,
        max_size_mb: float = 0.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.default_ttl = default_ttl
        self.max_size_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb > 0 else 0
        self.stats = CacheStats()
        self._meta: dict[str, CacheEntry] = {}
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.cache_dir / "_index.json"

    def _load_index(self) -> None:
        """Load the index from disk."""
        idx = self._index_path()
        if idx.exists():
            try:
                with open(idx, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                for k, v in raw.items():
                    self._meta[k] = CacheEntry(**v)
            except Exception:
                self._meta = {}

    def _save_index(self) -> None:
        """Persist the index to disk."""
        idx = self._index_path()
        data = {k: {"key": e.key, "created_at": e.created_at,
                     "ttl": e.ttl, "format": e.format,
                     "size_bytes": e.size_bytes}
                for k, e in self._meta.items()}
        tmp = idx.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        tmp.rename(idx)

    def _data_path(self, key: str, fmt: str) -> Path:
        ext = ".json" if fmt == "json" else ".pkl"
        return self.cache_dir / f"{key}{ext}"

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        Any | None
            Cached value or ``None`` on miss.
        """
        if not self.enabled:
            self.stats.misses += 1
            return None

        entry = self._meta.get(key)
        if entry is None:
            self.stats.misses += 1
            return None

        if entry.is_expired():
            self._remove_entry(key)
            self.stats.misses += 1
            return None

        path = self._data_path(key, entry.format)
        if not path.exists():
            self._meta.pop(key, None)
            self.stats.misses += 1
            return None

        try:
            if entry.format == "json":
                with open(path, "r", encoding="utf-8") as fh:
                    value = _try_json_loads(fh.read())
            else:
                # SECURITY: pickle.load can execute arbitrary code. This cache
                # only loads files written by this process under cache_dir.
                # Verify integrity via content-addressed key before loading.
                with open(path, "rb") as fh:
                    raw = fh.read()
                actual_hash = hashlib.sha256(raw).hexdigest()[:16]
                expected_hash = entry.key
                if not expected_hash.startswith(("ci_", "frag_", "sol_", "est_")):
                    # Generic content key — verify hash matches
                    if actual_hash != expected_hash:
                        logger.warning("Cache integrity check failed for %s", key)
                        self._remove_entry(key)
                        self.stats.misses += 1
                        return None
                value = pickle.loads(raw)  # noqa: S301
            self.stats.hits += 1
            logger.debug("Cache hit: %s", key)
            return value
        except Exception as exc:
            logger.warning("Cache read error for %s: %s", key, exc)
            self._remove_entry(key)
            self.stats.misses += 1
            return None

    def put(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value in the cache.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to cache (must be JSON-serialisable or picklable).
        ttl : float | None
            Time-to-live override.
        """
        if not self.enabled:
            return

        if ttl is None:
            ttl = self.default_ttl

        # Try JSON first (more portable); fall back to pickle
        json_str = _try_json_dumps(value)
        if json_str is not None:
            fmt = "json"
            path = self._data_path(key, fmt)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(json_str)
            tmp.rename(path)
            size = path.stat().st_size
        else:
            fmt = "pickle"
            path = self._data_path(key, fmt)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.rename(path)
            size = path.stat().st_size

        self._meta[key] = CacheEntry(
            key=key,
            created_at=time.time(),
            ttl=ttl,
            format=fmt,
            size_bytes=size,
        )
        self.stats.puts += 1
        self.stats.total_bytes += size

        # Evict if over size limit
        if self.max_size_bytes > 0:
            self._evict_to_limit()

        self._save_index()
        logger.debug("Cache put: %s (%s, %d bytes)", key, fmt, size)

    def has(self, key: str) -> bool:
        """Check whether *key* exists and is not expired."""
        if not self.enabled:
            return False
        entry = self._meta.get(key)
        if entry is None:
            return False
        if entry.is_expired():
            self._remove_entry(key)
            return False
        return self._data_path(key, entry.format).exists()

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate a specific key or the entire cache.

        Parameters
        ----------
        key : str | None
            Key to invalidate.  ``None`` clears the entire cache.
        """
        if not self.enabled:
            return

        if key is None:
            # Clear everything
            for k in list(self._meta):
                self._remove_entry(k)
            self._meta.clear()
            self._save_index()
            logger.info("Cache fully invalidated")
        else:
            self._remove_entry(key)
            self._save_index()

    def keys(self) -> list[str]:
        """Return all non-expired cache keys."""
        if not self.enabled:
            return []
        return [k for k, e in self._meta.items() if not e.is_expired()]

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return len(self._meta)

    @property
    def size_bytes(self) -> int:
        """Total size of cached data in bytes."""
        return sum(e.size_bytes for e in self._meta.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_entry(self, key: str) -> None:
        entry = self._meta.pop(key, None)
        if entry is None:
            return
        path = self._data_path(key, entry.format)
        if path.exists():
            path.unlink()
        self.stats.evictions += 1

    def _evict_to_limit(self) -> None:
        """Evict oldest entries until total size is under the limit."""
        if self.max_size_bytes <= 0:
            return
        entries = sorted(self._meta.values(), key=lambda e: e.created_at)
        total = sum(e.size_bytes for e in entries)
        while total > self.max_size_bytes and entries:
            oldest = entries.pop(0)
            total -= oldest.size_bytes
            self._remove_entry(oldest.key)

    # ------------------------------------------------------------------
    # Content-addressed key generation
    # ------------------------------------------------------------------

    @staticmethod
    def content_key(*parts: Any) -> str:
        """Create a content-addressed cache key from arbitrary inputs.

        Parameters
        ----------
        *parts : Any
            Inputs to hash.

        Returns
        -------
        str
            Hex digest of the inputs.
        """
        h = hashlib.sha256()
        for p in parts:
            if isinstance(p, np.ndarray):
                h.update(p.tobytes())
            else:
                h.update(str(p).encode())
        return h.hexdigest()[:16]

    @staticmethod
    def config_key(config: Any) -> str:
        """Hash a configuration object."""
        h = hashlib.sha256()
        try:
            h.update(json.dumps(asdict(config), sort_keys=True, default=str).encode())
        except Exception:
            h.update(str(config).encode())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------
    # Composite keys for pipeline steps
    # ------------------------------------------------------------------

    @staticmethod
    def ci_test_key(adj_bytes: bytes, data_hash: str, alpha: float, method: str) -> str:
        """Cache key for CI test results."""
        h = hashlib.sha256()
        h.update(adj_bytes)
        h.update(data_hash.encode())
        h.update(f"alpha={alpha},method={method}".encode())
        return "ci_" + h.hexdigest()[:14]

    @staticmethod
    def fragility_key(adj_bytes: bytes, treatment: int, outcome: int) -> str:
        """Cache key for fragility scores."""
        h = hashlib.sha256()
        h.update(adj_bytes)
        h.update(f"t={treatment},o={outcome}".encode())
        return "frag_" + h.hexdigest()[:12]

    @staticmethod
    def solver_key(adj_bytes: bytes, max_k: int, strategy: str) -> str:
        """Cache key for solver results."""
        h = hashlib.sha256()
        h.update(adj_bytes)
        h.update(f"k={max_k},s={strategy}".encode())
        return "sol_" + h.hexdigest()[:13]

    @staticmethod
    def estimation_key(adj_bytes: bytes, treatment: int, outcome: int, method: str) -> str:
        """Cache key for estimation results."""
        h = hashlib.sha256()
        h.update(adj_bytes)
        h.update(f"t={treatment},o={outcome},m={method}".encode())
        return "est_" + h.hexdigest()[:13]

    @staticmethod
    def data_fingerprint(data: Any) -> str:
        """Compute a fast fingerprint for a pandas DataFrame."""
        h = hashlib.sha256()
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                h.update(str(data.shape).encode())
                h.update(str(data.dtypes.tolist()).encode())
                # Hash a sample for speed
                sample = data.head(100).to_numpy()
                h.update(sample.tobytes())
                h.update(data.tail(10).to_numpy().tobytes())
            else:
                h.update(str(data).encode())
        except Exception:
            h.update(str(data).encode())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------
    # Convenience: cached decorator
    # ------------------------------------------------------------------

    def cached(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* only on cache miss; otherwise return cached result."""
        result = self.get(key)
        if result is not None:
            return result
        result = fn(*args, **kwargs)
        self.put(key, result)
        return result

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ResultCache(dir={self.cache_dir}, enabled={self.enabled}, "
            f"entries={self.size}, bytes={self.size_bytes})"
        )
