"""
Caching module for the CausalBound pipeline.

Provides content-addressed, LRU/LFU/TTL/size-aware caching with
in-memory and disk-backed storage for LP solutions and inference results.
"""

import enum
import gzip
import hashlib
import json
import logging
import os
import pickle
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CachePolicy(enum.Enum):
    """Eviction policy for the cache."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    SIZE = "size"


@dataclass
class CacheEntry:
    """A single entry stored in the cache."""
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl


@dataclass
class CacheStats:
    """Aggregate statistics for cache performance."""
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    disk_entry_count: int = 0
    total_access_time: float = 0.0
    access_operations: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return self.hit_count / total

    @property
    def average_access_time(self) -> float:
        if self.access_operations == 0:
            return 0.0
        return self.total_access_time / self.access_operations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_rate,
            "eviction_count": self.eviction_count,
            "total_size_bytes": self.total_size_bytes,
            "entry_count": self.entry_count,
            "disk_entry_count": self.disk_entry_count,
            "average_access_time": self.average_access_time,
        }

    def reset(self) -> None:
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.total_size_bytes = 0
        self.entry_count = 0
        self.disk_entry_count = 0
        self.total_access_time = 0.0
        self.access_operations = 0


class DiskCache:
    """Disk-backed cache using pickle + gzip serialization."""

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        logger.debug("DiskCache initialized at %s", self._cache_dir)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def _key_to_path(self, key: str) -> Path:
        safe_name = key[:64]
        subdir = key[:2]
        dir_path = self._cache_dir / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{safe_name}.cache.gz"

    def save_to_disk(self, key: str, value: Any) -> bool:
        path = self._key_to_path(key)
        with self._lock:
            try:
                serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                compressed = gzip.compress(serialized, compresslevel=6)
                tmp_path = path.with_suffix(".tmp")
                tmp_path.write_bytes(compressed)
                tmp_path.rename(path)
                logger.debug(
                    "Saved to disk: %s (%d bytes compressed)", key[:16], len(compressed)
                )
                return True
            except (pickle.PicklingError, OSError, TypeError) as exc:
                logger.warning("Failed to save key %s to disk: %s", key[:16], exc)
                tmp_path = path.with_suffix(".tmp")
                if tmp_path.exists():
                    tmp_path.unlink()
                return False

    def load_from_disk(self, key: str) -> Optional[Any]:
        path = self._key_to_path(key)
        with self._lock:
            if not path.exists():
                return None
            try:
                compressed = path.read_bytes()
                serialized = gzip.decompress(compressed)
                value = pickle.loads(serialized)  # noqa: S301
                logger.debug("Loaded from disk: %s", key[:16])
                return value
            except (
                pickle.UnpicklingError,
                gzip.BadGzipFile,
                OSError,
                EOFError,
            ) as exc:
                logger.warning("Failed to load key %s from disk: %s", key[:16], exc)
                return None

    def disk_contains(self, key: str) -> bool:
        path = self._key_to_path(key)
        return path.exists()

    def remove(self, key: str) -> bool:
        path = self._key_to_path(key)
        with self._lock:
            if path.exists():
                path.unlink()
                logger.debug("Removed from disk: %s", key[:16])
                return True
            return False

    def clear_disk(self) -> int:
        count = 0
        with self._lock:
            if self._cache_dir.exists():
                for item in self._cache_dir.rglob("*.cache.gz"):
                    item.unlink()
                    count += 1
                for subdir in self._cache_dir.iterdir():
                    if subdir.is_dir():
                        try:
                            subdir.rmdir()
                        except OSError:
                            pass
        logger.info("Cleared %d entries from disk cache", count)
        return count

    def disk_entry_count(self) -> int:
        if not self._cache_dir.exists():
            return 0
        return sum(1 for _ in self._cache_dir.rglob("*.cache.gz"))

    def disk_size_bytes(self) -> int:
        if not self._cache_dir.exists():
            return 0
        total = 0
        for item in self._cache_dir.rglob("*.cache.gz"):
            total += item.stat().st_size
        return total

    def list_keys(self) -> List[str]:
        keys = []
        if not self._cache_dir.exists():
            return keys
        for item in self._cache_dir.rglob("*.cache.gz"):
            key = item.stem.replace(".cache", "")
            keys.append(key)
        return keys


class CacheManager:
    """
    Content-addressed cache with in-memory LRU/LFU and disk-backed overflow.

    Keys are SHA256 hashes of cache key content, ensuring deterministic
    caching of LP solutions and inference results.
    """

    def __init__(
        self,
        max_entries: int = 1024,
        max_size_bytes: int = 256 * 1024 * 1024,
        policy: CachePolicy = CachePolicy.LRU,
        disk_cache_dir: Optional[str] = None,
        spill_to_disk: bool = True,
        default_ttl: Optional[float] = None,
    ) -> None:
        self._max_entries = max_entries
        self._max_size_bytes = max_size_bytes
        self._policy = policy
        self._spill_to_disk = spill_to_disk
        self._default_ttl = default_ttl
        self._lock = threading.Lock()

        self._store: Dict[str, CacheEntry] = {}
        self._stats = CacheStats()

        if disk_cache_dir is not None:
            self._disk = DiskCache(disk_cache_dir)
        elif spill_to_disk:
            default_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "causalbound"
            )
            self._disk = DiskCache(default_dir)
        else:
            self._disk = None

        logger.info(
            "CacheManager initialized: max_entries=%d, max_size=%d, policy=%s",
            max_entries,
            max_size_bytes,
            policy.value,
        )

    @property
    def policy(self) -> CachePolicy:
        return self._policy

    @policy.setter
    def policy(self, value: CachePolicy) -> None:
        with self._lock:
            self._policy = value
            logger.info("Cache policy changed to %s", value.value)

    def _compute_content_hash(self, key: Any) -> str:
        """Compute a deterministic SHA256 hash for an arbitrary cache key."""
        hasher = hashlib.sha256()
        self._hash_value(hasher, key)
        return hasher.hexdigest()

    def _hash_value(self, hasher: "hashlib._Hash", value: Any) -> None:
        """Recursively hash a value for content addressing."""
        if value is None:
            hasher.update(b"__none__")
        elif isinstance(value, bytes):
            hasher.update(b"bytes:")
            hasher.update(value)
        elif isinstance(value, str):
            hasher.update(b"str:")
            hasher.update(value.encode("utf-8"))
        elif isinstance(value, bool):
            hasher.update(b"bool:")
            hasher.update(b"1" if value else b"0")
        elif isinstance(value, int):
            hasher.update(b"int:")
            hasher.update(str(value).encode("utf-8"))
        elif isinstance(value, float):
            hasher.update(b"float:")
            hasher.update(repr(value).encode("utf-8"))
        elif isinstance(value, (list, tuple)):
            tag = b"list:" if isinstance(value, list) else b"tuple:"
            hasher.update(tag)
            hasher.update(str(len(value)).encode("utf-8"))
            for item in value:
                self._hash_value(hasher, item)
        elif isinstance(value, dict):
            hasher.update(b"dict:")
            hasher.update(str(len(value)).encode("utf-8"))
            for k in sorted(value.keys(), key=repr):
                self._hash_value(hasher, k)
                self._hash_value(hasher, value[k])
        elif isinstance(value, set):
            hasher.update(b"set:")
            for item in sorted(value, key=repr):
                self._hash_value(hasher, item)
        elif isinstance(value, frozenset):
            hasher.update(b"frozenset:")
            for item in sorted(value, key=repr):
                self._hash_value(hasher, item)
        else:
            # Handle numpy arrays and arbitrary objects
            type_name = type(value).__name__
            hasher.update(f"obj:{type_name}:".encode("utf-8"))
            if hasattr(value, "tobytes") and hasattr(value, "shape"):
                # numpy-like array
                hasher.update(str(value.shape).encode("utf-8"))
                hasher.update(str(value.dtype).encode("utf-8"))
                hasher.update(value.tobytes())
            elif hasattr(value, "__dict__"):
                self._hash_value(hasher, value.__dict__)
            else:
                hasher.update(repr(value).encode("utf-8"))

    def _estimate_size(self, value: Any, seen: Optional[set] = None) -> int:
        """Recursively estimate memory size of a value in bytes."""
        if seen is None:
            seen = set()

        obj_id = id(value)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        size = sys.getsizeof(value)

        if isinstance(value, str) or isinstance(value, bytes):
            return size
        elif isinstance(value, dict):
            for k, v in value.items():
                size += self._estimate_size(k, seen)
                size += self._estimate_size(v, seen)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                size += self._estimate_size(item, seen)
        elif hasattr(value, "nbytes"):
            # numpy array
            size += value.nbytes
        elif hasattr(value, "__dict__"):
            size += self._estimate_size(value.__dict__, seen)
        elif hasattr(value, "__slots__"):
            for slot in value.__slots__:
                if hasattr(value, slot):
                    size += self._estimate_size(getattr(value, slot), seen)

        return size

    def _current_total_size(self) -> int:
        return sum(entry.size_bytes for entry in self._store.values())

    def _purge_expired(self) -> int:
        """Remove all expired entries. Returns count of purged entries."""
        expired_keys = [k for k, v in self._store.items() if v.is_expired()]
        for key in expired_keys:
            del self._store[key]
            self._stats.eviction_count += 1
        if expired_keys:
            logger.debug("Purged %d expired entries", len(expired_keys))
        return len(expired_keys)

    def _select_victim(self) -> Optional[str]:
        """Select a cache entry to evict based on current policy."""
        if not self._store:
            return None

        if self._policy == CachePolicy.LRU:
            return min(self._store, key=lambda k: self._store[k].accessed_at)
        elif self._policy == CachePolicy.LFU:
            return min(self._store, key=lambda k: self._store[k].access_count)
        elif self._policy == CachePolicy.TTL:
            # Evict entries closest to expiration; non-TTL entries last
            def ttl_remaining(k: str) -> float:
                entry = self._store[k]
                if entry.ttl is None:
                    return float("inf")
                return entry.ttl - (time.time() - entry.created_at)
            return min(self._store, key=ttl_remaining)
        elif self._policy == CachePolicy.SIZE:
            return max(self._store, key=lambda k: self._store[k].size_bytes)
        else:
            return min(self._store, key=lambda k: self._store[k].accessed_at)

    def evict_lru(self) -> Optional[str]:
        """Evict the least recently used entry. Returns evicted key or None."""
        with self._lock:
            return self._evict_one()

    def _evict_one(self) -> Optional[str]:
        """Internal eviction (caller must hold lock)."""
        victim_key = self._select_victim()
        if victim_key is None:
            return None

        entry = self._store[victim_key]

        if self._spill_to_disk and self._disk is not None:
            success = self._disk.save_to_disk(victim_key, entry.value)
            if success:
                logger.debug("Spilled entry %s to disk", victim_key[:16])

        del self._store[victim_key]
        self._stats.eviction_count += 1
        logger.debug("Evicted entry: %s (policy=%s)", victim_key[:16], self._policy.value)
        return victim_key

    def _ensure_capacity(self, needed_bytes: int) -> None:
        """Evict entries until there is room for needed_bytes."""
        self._purge_expired()

        while len(self._store) >= self._max_entries:
            evicted = self._evict_one()
            if evicted is None:
                break

        while (
            self._current_total_size() + needed_bytes > self._max_size_bytes
            and self._store
        ):
            evicted = self._evict_one()
            if evicted is None:
                break

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value by key. Updates access metadata for LRU tracking.
        Returns None on cache miss.
        """
        t_start = time.monotonic()
        content_hash = self._compute_content_hash(key)

        with self._lock:
            entry = self._store.get(content_hash)

            if entry is not None:
                if entry.is_expired():
                    del self._store[content_hash]
                    self._stats.miss_count += 1
                    self._stats.eviction_count += 1
                    elapsed = time.monotonic() - t_start
                    self._stats.total_access_time += elapsed
                    self._stats.access_operations += 1
                    logger.debug("Cache expired: %s", content_hash[:16])
                    return None

                entry.accessed_at = time.time()
                entry.access_count += 1
                self._stats.hit_count += 1
                elapsed = time.monotonic() - t_start
                self._stats.total_access_time += elapsed
                self._stats.access_operations += 1
                logger.debug("Cache hit: %s", content_hash[:16])
                return entry.value

            # Check disk cache on memory miss
            if self._disk is not None and self._disk.disk_contains(content_hash):
                value = self._disk.load_from_disk(content_hash)
                if value is not None:
                    size = self._estimate_size(value)
                    now = time.time()
                    self._ensure_capacity(size)
                    self._store[content_hash] = CacheEntry(
                        value=value,
                        created_at=now,
                        accessed_at=now,
                        access_count=1,
                        size_bytes=size,
                        ttl=self._default_ttl,
                    )
                    self._stats.hit_count += 1
                    elapsed = time.monotonic() - t_start
                    self._stats.total_access_time += elapsed
                    self._stats.access_operations += 1
                    logger.debug("Cache hit (disk): %s", content_hash[:16])
                    return value

            self._stats.miss_count += 1
            elapsed = time.monotonic() - t_start
            self._stats.total_access_time += elapsed
            self._stats.access_operations += 1
            logger.debug("Cache miss: %s", content_hash[:16])
            return None

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> str:
        """
        Store a value in the cache. Returns the content hash used as key.
        Triggers eviction if capacity is exceeded.
        """
        content_hash = self._compute_content_hash(key)
        size = self._estimate_size(value)
        effective_ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()

        with self._lock:
            if content_hash in self._store:
                old_entry = self._store[content_hash]
                self._store[content_hash] = CacheEntry(
                    value=value,
                    created_at=now,
                    accessed_at=now,
                    access_count=old_entry.access_count + 1,
                    size_bytes=size,
                    ttl=effective_ttl,
                )
                logger.debug("Cache update: %s (%d bytes)", content_hash[:16], size)
                return content_hash

            self._ensure_capacity(size)

            self._store[content_hash] = CacheEntry(
                value=value,
                created_at=now,
                accessed_at=now,
                access_count=0,
                size_bytes=size,
                ttl=effective_ttl,
            )
            logger.debug("Cache put: %s (%d bytes)", content_hash[:16], size)
            return content_hash

    def invalidate(self, key: Any) -> bool:
        """Remove a specific key from both memory and disk. Returns True if found."""
        content_hash = self._compute_content_hash(key)
        removed = False
        with self._lock:
            if content_hash in self._store:
                del self._store[content_hash]
                removed = True

        if self._disk is not None:
            disk_removed = self._disk.remove(content_hash)
            removed = removed or disk_removed

        if removed:
            logger.debug("Invalidated: %s", content_hash[:16])
        return removed

    def clear(self) -> Tuple[int, int]:
        """
        Remove all entries from memory and disk.
        Returns (memory_cleared, disk_cleared).
        """
        with self._lock:
            mem_count = len(self._store)
            self._store.clear()

        disk_count = 0
        if self._disk is not None:
            disk_count = self._disk.clear_disk()

        self._stats.reset()
        logger.info("Cache cleared: %d memory, %d disk", mem_count, disk_count)
        return mem_count, disk_count

    def contains(self, key: Any) -> bool:
        """Check if key exists in memory or disk (without loading)."""
        content_hash = self._compute_content_hash(key)
        with self._lock:
            if content_hash in self._store:
                entry = self._store[content_hash]
                if not entry.is_expired():
                    return True
                del self._store[content_hash]
                self._stats.eviction_count += 1

        if self._disk is not None:
            return self._disk.disk_contains(content_hash)
        return False

    def get_stats(self) -> CacheStats:
        """Return a snapshot of cache statistics."""
        with self._lock:
            self._stats.total_size_bytes = self._current_total_size()
            self._stats.entry_count = len(self._store)

        if self._disk is not None:
            self._stats.disk_entry_count = self._disk.disk_entry_count()
        else:
            self._stats.disk_entry_count = 0

        return CacheStats(
            hit_count=self._stats.hit_count,
            miss_count=self._stats.miss_count,
            eviction_count=self._stats.eviction_count,
            total_size_bytes=self._stats.total_size_bytes,
            entry_count=self._stats.entry_count,
            disk_entry_count=self._stats.disk_entry_count,
            total_access_time=self._stats.total_access_time,
            access_operations=self._stats.access_operations,
        )

    def set_capacity(self, max_entries: Optional[int] = None, max_size_bytes: Optional[int] = None) -> None:
        """Resize the cache, evicting entries if the new capacity is smaller."""
        with self._lock:
            if max_entries is not None:
                self._max_entries = max_entries
            if max_size_bytes is not None:
                self._max_size_bytes = max_size_bytes

            while len(self._store) > self._max_entries:
                evicted = self._evict_one()
                if evicted is None:
                    break

            while self._current_total_size() > self._max_size_bytes and self._store:
                evicted = self._evict_one()
                if evicted is None:
                    break

        logger.info(
            "Cache resized: max_entries=%d, max_size_bytes=%d",
            self._max_entries,
            self._max_size_bytes,
        )

    def keys(self) -> List[str]:
        """Return content hashes of all in-memory entries."""
        with self._lock:
            return list(self._store.keys())

    def memory_entry_count(self) -> int:
        with self._lock:
            return len(self._store)

    def disk_entry_count(self) -> int:
        if self._disk is None:
            return 0
        return self._disk.disk_entry_count()

    def get_or_compute(self, key: Any, compute_fn: Any, ttl: Optional[float] = None) -> Any:
        """
        Return cached value for key, or call compute_fn() to produce it,
        cache the result, and return it.
        """
        result = self.get(key)
        if result is not None:
            return result
        value = compute_fn()
        self.put(key, value, ttl=ttl)
        return value

    def save_to_disk(self, key: Any, value: Any) -> bool:
        """Explicitly save a value to the disk cache."""
        if self._disk is None:
            logger.warning("No disk cache configured")
            return False
        content_hash = self._compute_content_hash(key)
        return self._disk.save_to_disk(content_hash, value)

    def load_from_disk(self, key: Any) -> Optional[Any]:
        """Explicitly load a value from the disk cache."""
        if self._disk is None:
            return None
        content_hash = self._compute_content_hash(key)
        return self._disk.load_from_disk(content_hash)

    def disk_contains(self, key: Any) -> bool:
        """Check if key exists on disk."""
        if self._disk is None:
            return False
        content_hash = self._compute_content_hash(key)
        return self._disk.disk_contains(content_hash)

    def clear_disk(self) -> int:
        """Remove all disk cache files. Returns number removed."""
        if self._disk is None:
            return 0
        return self._disk.clear_disk()

    def bulk_put(self, items: Dict[Any, Any], ttl: Optional[float] = None) -> List[str]:
        """Store multiple key-value pairs. Returns list of content hashes."""
        hashes = []
        for key, value in items.items():
            h = self.put(key, value, ttl=ttl)
            hashes.append(h)
        return hashes

    def bulk_get(self, keys: List[Any]) -> Dict[str, Any]:
        """Retrieve multiple values. Returns dict mapping content hash to value (misses omitted)."""
        results = {}
        for key in keys:
            content_hash = self._compute_content_hash(key)
            value = self.get(key)
            if value is not None:
                results[content_hash] = value
        return results

    def export_metadata(self) -> Dict[str, Any]:
        """Export cache metadata as a JSON-serializable dict."""
        with self._lock:
            entries_meta = {}
            for k, entry in self._store.items():
                entries_meta[k] = {
                    "created_at": entry.created_at,
                    "accessed_at": entry.accessed_at,
                    "access_count": entry.access_count,
                    "size_bytes": entry.size_bytes,
                    "ttl": entry.ttl,
                    "expired": entry.is_expired(),
                }

        stats = self.get_stats()
        return {
            "policy": self._policy.value,
            "max_entries": self._max_entries,
            "max_size_bytes": self._max_size_bytes,
            "stats": stats.to_dict(),
            "entries": entries_meta,
        }

    def warm_from_disk(self, max_entries: Optional[int] = None) -> int:
        """
        Pre-load entries from disk into memory cache.
        Returns the number of entries loaded.
        """
        if self._disk is None:
            return 0

        loaded = 0
        keys = self._disk.list_keys()
        limit = max_entries if max_entries is not None else len(keys)

        for disk_key in keys[:limit]:
            with self._lock:
                if disk_key in self._store:
                    continue
                if len(self._store) >= self._max_entries:
                    break

            value = self._disk.load_from_disk(disk_key)
            if value is None:
                continue

            size = self._estimate_size(value)
            now = time.time()

            with self._lock:
                if self._current_total_size() + size > self._max_size_bytes:
                    break
                self._store[disk_key] = CacheEntry(
                    value=value,
                    created_at=now,
                    accessed_at=now,
                    access_count=0,
                    size_bytes=size,
                    ttl=self._default_ttl,
                )
                loaded += 1

        logger.info("Warmed %d entries from disk", loaded)
        return loaded

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: Any) -> bool:
        return self.contains(key)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"CacheManager(entries={stats.entry_count}, "
            f"disk={stats.disk_entry_count}, "
            f"hit_rate={stats.hit_rate:.2%}, "
            f"policy={self._policy.value})"
        )
