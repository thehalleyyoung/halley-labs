"""
usability_oracle.pipeline.cache — File-based result caching with TTL.

Provides :class:`ResultCache` for caching stage outputs to avoid
redundant computation.  Supports both in-memory and file-based
caching with content-based keys and time-to-live eviction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    """Internal cache entry with value and metadata."""
    value: Any
    created_at: float
    ttl: Optional[float]  # None = never expires

    @property
    def expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl


# ---------------------------------------------------------------------------
# ResultCache
# ---------------------------------------------------------------------------

class ResultCache:
    """Pipeline result cache with optional file-system persistence.

    Parameters
    ----------
    cache_dir : str | Path | None
        Directory for file-based caching.  If None, only in-memory.
    default_ttl : float | None
        Default time-to-live in seconds.  None = no expiry.
    max_memory_entries : int
        Maximum number of in-memory cache entries.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        default_ttl: float | None = 3600.0,
        max_memory_entries: int = 1000,
    ) -> None:
        self._memory: dict[str, _CacheEntry] = {}
        self._default_ttl = default_ttl
        self._max_entries = max_memory_entries
        self._cache_dir: Optional[Path] = None

        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._hits = 0
        self._misses = 0

    # ── Public API --------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value.

        Checks in-memory first, then file cache.  Returns None on miss
        or if the entry has expired.
        """
        # In-memory lookup
        entry = self._memory.get(key)
        if entry is not None:
            if entry.expired:
                del self._memory[key]
            else:
                self._hits += 1
                return entry.value

        # File-based lookup
        if self._cache_dir is not None:
            file_path = self._key_to_path(key)
            if file_path.exists():
                try:
                    data = self._deserialize(file_path.read_bytes())
                    if data is not None:
                        meta = data.get("_meta", {})
                        created = meta.get("created_at", 0)
                        ttl = meta.get("ttl")
                        if ttl is not None and (time.time() - created) > ttl:
                            file_path.unlink(missing_ok=True)
                        else:
                            value = data.get("value")
                            # Promote to memory cache
                            self._memory[key] = _CacheEntry(
                                value=value,
                                created_at=created,
                                ttl=ttl,
                            )
                            self._hits += 1
                            return value
                except (json.JSONDecodeError, KeyError, OSError) as exc:
                    logger.debug("Cache file read error for %s: %s", key, exc)

        self._misses += 1
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Store a value in the cache.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to cache (must be JSON-serialisable for file cache).
        ttl : float | None
            Time-to-live in seconds.  Uses default if None.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl

        # Evict if at capacity
        if len(self._memory) >= self._max_entries:
            self._evict_oldest()

        self._memory[key] = _CacheEntry(
            value=value,
            created_at=time.time(),
            ttl=effective_ttl,
        )

        # File persistence
        if self._cache_dir is not None:
            try:
                file_path = self._key_to_path(key)
                data = {
                    "value": value,
                    "_meta": {
                        "created_at": time.time(),
                        "ttl": effective_ttl,
                        "key": key,
                    },
                }
                file_path.write_bytes(self._serialize(data))
            except (TypeError, OSError) as exc:
                logger.debug("Cache file write error for %s: %s", key, exc)

    def invalidate(self, key: str) -> None:
        """Remove a specific cache entry."""
        self._memory.pop(key, None)
        if self._cache_dir is not None:
            file_path = self._key_to_path(key)
            file_path.unlink(missing_ok=True)

    def clear(self) -> None:
        """Remove all cache entries."""
        self._memory.clear()
        if self._cache_dir is not None:
            for f in self._cache_dir.glob("*.json"):
                try:
                    f.unlink()
                except OSError:
                    pass
        self._hits = 0
        self._misses = 0

    # ── Key computation ---------------------------------------------------

    def compute_key(self, stage: str, inputs: dict[str, Any]) -> str:
        """Compute a content-based cache key.

        Hashes the stage name and a deterministic representation of
        the inputs to produce a stable cache key.
        """
        h = hashlib.sha256()
        h.update(stage.encode("utf-8"))

        for k in sorted(inputs.keys()):
            v = inputs[k]
            h.update(k.encode("utf-8"))
            h.update(self._hash_value(v).encode("utf-8"))

        return h.hexdigest()[:32]

    @staticmethod
    def _hash_value(value: Any) -> str:
        """Produce a stable hash string for a value."""
        if value is None:
            return "none"
        if isinstance(value, (int, float, bool, str)):
            return str(value)
        if isinstance(value, (list, tuple)):
            return f"[{','.join(ResultCache._hash_value(v) for v in value)}]"
        if isinstance(value, dict):
            parts = [
                f"{k}:{ResultCache._hash_value(v)}"
                for k, v in sorted(value.items(), key=lambda x: str(x[0]))
            ]
            return "{" + ",".join(parts) + "}"
        if hasattr(value, "to_dict"):
            return ResultCache._hash_value(value.to_dict())
        # Fallback: use id for non-serialisable objects
        return f"obj_{id(value)}"

    # ── Serialisation -----------------------------------------------------

    @staticmethod
    def _serialize(value: Any) -> bytes:
        """Serialise a value to bytes (JSON)."""
        return json.dumps(value, default=_json_default, indent=None).encode("utf-8")

    @staticmethod
    def _deserialize(data: bytes) -> Any:
        """Deserialise bytes back to a value."""
        return json.loads(data.decode("utf-8"))

    # ── File paths --------------------------------------------------------

    def _key_to_path(self, key: str) -> Path:
        """Map a cache key to a file path."""
        assert self._cache_dir is not None
        safe_key = hashlib.sha256(key.encode()).hexdigest()[:40]
        return self._cache_dir / f"{safe_key}.json"

    # ── Eviction ----------------------------------------------------------

    def _evict_oldest(self) -> None:
        """Remove the oldest entry from the memory cache."""
        if not self._memory:
            return
        # Remove expired first
        expired = [k for k, v in self._memory.items() if v.expired]
        for k in expired:
            del self._memory[k]
        if len(self._memory) < self._max_entries:
            return
        # Remove oldest by creation time
        oldest_key = min(self._memory, key=lambda k: self._memory[k].created_at)
        del self._memory[oldest_key]

    # ── Statistics --------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._memory)

    def stats(self) -> dict[str, Any]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "memory_entries": self.size,
            "file_backed": self._cache_dir is not None,
        }


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Default JSON serialiser for non-standard types."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)
