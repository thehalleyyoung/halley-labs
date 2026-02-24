from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import struct
import tempfile
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)


# ---------------------------------------------------------------------------
# Local type definitions
# ---------------------------------------------------------------------------

class PredicateKind(Enum):
    TYPE_CHECK = auto()
    RANGE = auto()
    NULL_CHECK = auto()
    LENGTH = auto()
    MEMBER = auto()
    CUSTOM = auto()
    EQUALITY = auto()
    ISINSTANCE = auto()
    TRUTHY = auto()
    COMPARISON = auto()
    REGEX = auto()
    CALLABLE_CHECK = auto()
    ATTRIBUTE_CHECK = auto()
    CONTAINER_CHECK = auto()


@dataclass(frozen=True)
class Predicate:
    kind: PredicateKind
    name: str
    args: Tuple[str, ...] = ()

    def __hash__(self) -> int:
        return hash((self.kind, self.name, self.args))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Predicate):
            return NotImplemented
        return self.kind == other.kind and self.name == other.name and self.args == other.args

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind.name, "name": self.name, "args": list(self.args)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Predicate:
        return cls(kind=PredicateKind[data["kind"]], name=data["name"], args=tuple(data.get("args", [])))


class PredicateSet:
    """Lightweight predicate set used for cache key hashing."""

    def __init__(self, predicates: Optional[Set[Predicate]] = None) -> None:
        self._preds: Set[Predicate] = set(predicates) if predicates else set()

    def add(self, p: Predicate) -> None:
        self._preds.add(p)

    def remove(self, p: Predicate) -> None:
        self._preds.discard(p)

    def overlaps(self, other: PredicateSet) -> bool:
        return bool(self._preds & other._preds)

    def union(self, other: PredicateSet) -> PredicateSet:
        return PredicateSet(self._preds | other._preds)

    def intersection(self, other: PredicateSet) -> PredicateSet:
        return PredicateSet(self._preds & other._preds)

    def difference(self, other: PredicateSet) -> PredicateSet:
        return PredicateSet(self._preds - other._preds)

    def symmetric_difference(self, other: PredicateSet) -> PredicateSet:
        return PredicateSet(self._preds ^ other._preds)

    def is_empty(self) -> bool:
        return len(self._preds) == 0

    def size(self) -> int:
        return len(self._preds)

    def __hash__(self) -> int:
        return hash(frozenset(self._preds))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PredicateSet):
            return NotImplemented
        return self._preds == other._preds

    def __iter__(self) -> Iterator[Predicate]:
        return iter(self._preds)

    def __len__(self) -> int:
        return len(self._preds)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in sorted(self._preds, key=lambda p: (p.kind.name, p.name))]

    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]]) -> PredicateSet:
        return cls({Predicate.from_dict(d) for d in data})


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheKey:
    """Typed cache key that captures all inputs that affect an analysis result."""

    module_path: str
    function_name: str
    source_hash: str
    config_hash: str = ""
    dependency_hash: str = ""

    @staticmethod
    def _hash_str(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def compute(
        cls,
        module_path: str,
        function_name: str,
        source: str,
        config: Optional[Dict[str, Any]] = None,
        dependency_summaries: Optional[Dict[str, Any]] = None,
    ) -> CacheKey:
        source_hash = cls._hash_str(source)
        config_hash = ""
        if config:
            config_hash = cls._hash_str(json.dumps(config, sort_keys=True, default=str))
        dep_hash = ""
        if dependency_summaries:
            dep_hash = cls._hash_str(json.dumps(dependency_summaries, sort_keys=True, default=str))
        return cls(
            module_path=module_path,
            function_name=function_name,
            source_hash=source_hash,
            config_hash=config_hash,
            dependency_hash=dep_hash,
        )

    @property
    def composite_key(self) -> str:
        parts = [self.module_path, self.function_name, self.source_hash]
        if self.config_hash:
            parts.append(self.config_hash)
        if self.dependency_hash:
            parts.append(self.dependency_hash)
        return ":".join(parts)

    def __str__(self) -> str:
        return self.composite_key

    def to_dict(self) -> Dict[str, str]:
        return {
            "module_path": self.module_path,
            "function_name": self.function_name,
            "source_hash": self.source_hash,
            "config_hash": self.config_hash,
            "dependency_hash": self.dependency_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> CacheKey:
        return cls(
            module_path=data.get("module_path", ""),
            function_name=data.get("function_name", ""),
            source_hash=data.get("source_hash", ""),
            config_hash=data.get("config_hash", ""),
            dependency_hash=data.get("dependency_hash", ""),
        )


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cached analysis result with metadata."""

    key: CacheKey
    result: Dict[str, Any]
    timestamp: float = 0.0
    analysis_version: int = 1
    config_version: int = 1
    dependencies: List[str] = field(default_factory=list)
    predicates: PredicateSet = field(default_factory=PredicateSet)
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.size_bytes == 0:
            self.size_bytes = len(json.dumps(self.result, default=str).encode())

    def touch(self) -> None:
        self.access_count += 1
        self.last_access = time.time()

    def is_valid(self, current_dependencies: Optional[Dict[str, str]] = None) -> bool:
        if current_dependencies is None:
            return True
        for dep in self.dependencies:
            if dep not in current_dependencies:
                return False
            if current_dependencies[dep] != self.key.dependency_hash:
                return False
        return True

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "result": self.result,
            "timestamp": self.timestamp,
            "analysis_version": self.analysis_version,
            "config_version": self.config_version,
            "dependencies": self.dependencies,
            "predicates": self.predicates.to_dict(),
            "access_count": self.access_count,
            "last_access": self.last_access,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CacheEntry:
        return cls(
            key=CacheKey.from_dict(data.get("key", {})),
            result=data.get("result", {}),
            timestamp=data.get("timestamp", 0.0),
            analysis_version=data.get("analysis_version", 1),
            config_version=data.get("config_version", 1),
            dependencies=data.get("dependencies", []),
            predicates=PredicateSet.from_dict(data.get("predicates", [])),
            access_count=data.get("access_count", 0),
            last_access=data.get("last_access", 0.0),
            size_bytes=data.get("size_bytes", 0),
        )


# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------

@dataclass
class CacheStatistics:
    """Performance metrics for a cache layer."""

    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    invalidation_count: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    write_count: int = 0
    read_count: int = 0
    error_count: int = 0
    compression_ratio: float = 1.0
    _start_time: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    @property
    def average_entry_size(self) -> float:
        return self.total_size_bytes / self.entry_count if self.entry_count > 0 else 0.0

    @property
    def time_saved_estimate_ms(self) -> float:
        avg_analysis_ms = 50.0
        return self.hit_count * avg_analysis_ms

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def record_hit(self) -> None:
        self.hit_count += 1
        self.read_count += 1

    def record_miss(self) -> None:
        self.miss_count += 1
        self.read_count += 1

    def record_write(self, size_bytes: int) -> None:
        self.write_count += 1
        self.total_size_bytes += size_bytes
        self.entry_count += 1

    def record_eviction(self, size_bytes: int) -> None:
        self.eviction_count += 1
        self.total_size_bytes -= size_bytes
        self.entry_count -= 1

    def record_invalidation(self, count: int = 1) -> None:
        self.invalidation_count += count

    def record_error(self) -> None:
        self.error_count += 1

    def reset(self) -> None:
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.invalidation_count = 0
        self.write_count = 0
        self.read_count = 0
        self.error_count = 0
        self._start_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_rate,
            "eviction_count": self.eviction_count,
            "invalidation_count": self.invalidation_count,
            "total_size_bytes": self.total_size_bytes,
            "entry_count": self.entry_count,
            "average_entry_size": self.average_entry_size,
            "time_saved_estimate_ms": self.time_saved_estimate_ms,
            "write_count": self.write_count,
            "read_count": self.read_count,
            "error_count": self.error_count,
            "compression_ratio": self.compression_ratio,
            "uptime_seconds": self.uptime_seconds,
        }

    def summary(self) -> str:
        return (
            f"Cache: {self.hit_count} hits, {self.miss_count} misses "
            f"({self.hit_rate:.1%} rate), {self.entry_count} entries, "
            f"{self.total_size_bytes / 1024:.1f} KB"
        )

    def merge(self, other: CacheStatistics) -> CacheStatistics:
        return CacheStatistics(
            hit_count=self.hit_count + other.hit_count,
            miss_count=self.miss_count + other.miss_count,
            eviction_count=self.eviction_count + other.eviction_count,
            invalidation_count=self.invalidation_count + other.invalidation_count,
            total_size_bytes=self.total_size_bytes + other.total_size_bytes,
            entry_count=self.entry_count + other.entry_count,
            write_count=self.write_count + other.write_count,
            read_count=self.read_count + other.read_count,
            error_count=self.error_count + other.error_count,
        )


# ---------------------------------------------------------------------------
# Eviction policies
# ---------------------------------------------------------------------------

class EvictionPolicy(Enum):
    LRU = auto()
    LFU = auto()
    SIZE = auto()
    FIFO = auto()
    TTL = auto()


class EvictionStrategy:
    """Determines which entries to evict when the cache is full."""

    def __init__(self, policy: EvictionPolicy = EvictionPolicy.LRU) -> None:
        self._policy = policy

    def select_for_eviction(
        self,
        entries: Dict[str, CacheEntry],
        count: int = 1,
    ) -> List[str]:
        if not entries:
            return []
        if self._policy == EvictionPolicy.LRU:
            return self._evict_lru(entries, count)
        elif self._policy == EvictionPolicy.LFU:
            return self._evict_lfu(entries, count)
        elif self._policy == EvictionPolicy.SIZE:
            return self._evict_largest(entries, count)
        elif self._policy == EvictionPolicy.FIFO:
            return self._evict_fifo(entries, count)
        elif self._policy == EvictionPolicy.TTL:
            return self._evict_oldest(entries, count)
        return self._evict_lru(entries, count)

    @staticmethod
    def _evict_lru(entries: Dict[str, CacheEntry], count: int) -> List[str]:
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].last_access)
        return [k for k, _ in sorted_entries[:count]]

    @staticmethod
    def _evict_lfu(entries: Dict[str, CacheEntry], count: int) -> List[str]:
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].access_count)
        return [k for k, _ in sorted_entries[:count]]

    @staticmethod
    def _evict_largest(entries: Dict[str, CacheEntry], count: int) -> List[str]:
        sorted_entries = sorted(entries.items(), key=lambda x: -x[1].size_bytes)
        return [k for k, _ in sorted_entries[:count]]

    @staticmethod
    def _evict_fifo(entries: Dict[str, CacheEntry], count: int) -> List[str]:
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].timestamp)
        return [k for k, _ in sorted_entries[:count]]

    @staticmethod
    def _evict_oldest(entries: Dict[str, CacheEntry], count: int) -> List[str]:
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].timestamp)
        return [k for k, _ in sorted_entries[:count]]


# ---------------------------------------------------------------------------
# MemoryCache
# ---------------------------------------------------------------------------

class MemoryCache:
    """In-memory LRU cache for analysis results."""

    def __init__(
        self,
        max_entries: int = 10000,
        max_size_bytes: int = 100 * 1024 * 1024,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
    ) -> None:
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._max_size_bytes = max_size_bytes
        self._current_size = 0
        self._lock = threading.RLock()
        self._stats = CacheStatistics()
        self._eviction = EvictionStrategy(eviction_policy)

    def get(self, key: CacheKey) -> Optional[CacheEntry]:
        with self._lock:
            composite = key.composite_key
            entry = self._entries.get(composite)
            if entry is None:
                self._stats.record_miss()
                return None
            self._stats.record_hit()
            entry.touch()
            self._entries.move_to_end(composite)
            return entry

    def put(self, entry: CacheEntry) -> None:
        with self._lock:
            composite = entry.key.composite_key
            if composite in self._entries:
                old = self._entries[composite]
                self._current_size -= old.size_bytes
                del self._entries[composite]
            self._entries[composite] = entry
            self._current_size += entry.size_bytes
            self._stats.record_write(entry.size_bytes)
            self._evict_if_needed()

    def invalidate(self, key: CacheKey) -> bool:
        with self._lock:
            composite = key.composite_key
            if composite in self._entries:
                entry = self._entries.pop(composite)
                self._current_size -= entry.size_bytes
                self._stats.record_invalidation()
                return True
            return False

    def invalidate_by_prefix(self, prefix: str) -> int:
        with self._lock:
            to_remove = [k for k in self._entries if k.startswith(prefix)]
            for k in to_remove:
                entry = self._entries.pop(k)
                self._current_size -= entry.size_bytes
            self._stats.record_invalidation(len(to_remove))
            return len(to_remove)

    def invalidate_by_function(self, func_id: str) -> int:
        with self._lock:
            to_remove = [k for k in self._entries if func_id in k]
            for k in to_remove:
                entry = self._entries.pop(k)
                self._current_size -= entry.size_bytes
            self._stats.record_invalidation(len(to_remove))
            return len(to_remove)

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._current_size = 0
            self._stats.record_invalidation(count)
            return count

    def _evict_if_needed(self) -> None:
        while (
            len(self._entries) > self._max_entries
            or self._current_size > self._max_size_bytes
        ):
            if not self._entries:
                break
            victims = self._eviction.select_for_eviction(dict(self._entries), 1)
            for v in victims:
                if v in self._entries:
                    entry = self._entries.pop(v)
                    self._current_size -= entry.size_bytes
                    self._stats.record_eviction(entry.size_bytes)

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def size_bytes(self) -> int:
        return self._current_size

    @property
    def statistics(self) -> CacheStatistics:
        self._stats.entry_count = len(self._entries)
        self._stats.total_size_bytes = self._current_size
        return self._stats

    def keys(self) -> List[str]:
        return list(self._entries.keys())

    def entries(self) -> List[CacheEntry]:
        return list(self._entries.values())

    def contains(self, key: CacheKey) -> bool:
        return key.composite_key in self._entries


# ---------------------------------------------------------------------------
# DiskCache
# ---------------------------------------------------------------------------

class DiskCache:
    """
    Persistent disk cache for analysis results.

    Directory structure:
        .reftype-cache/
            module_path/
                function_name/
                    source_hash.json.gz
    """

    CACHE_FORMAT_VERSION = 2

    def __init__(
        self,
        cache_dir: str,
        max_size_bytes: int = 500 * 1024 * 1024,
        compression: bool = True,
        use_msgpack: bool = False,
    ) -> None:
        self._cache_dir = cache_dir
        self._max_size_bytes = max_size_bytes
        self._compression = compression
        self._use_msgpack = use_msgpack
        self._lock = threading.RLock()
        self._stats = CacheStatistics()
        os.makedirs(cache_dir, exist_ok=True)
        self._write_version_file()

    def _write_version_file(self) -> None:
        version_path = os.path.join(self._cache_dir, ".cache-version")
        try:
            with open(version_path, "w") as f:
                f.write(str(self.CACHE_FORMAT_VERSION))
        except OSError:
            pass

    def _read_version(self) -> int:
        version_path = os.path.join(self._cache_dir, ".cache-version")
        try:
            with open(version_path, "r") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return 0

    def _entry_path(self, key: CacheKey) -> str:
        safe_module = key.module_path.replace("/", "_").replace("\\", "_").replace(".", "_")
        safe_func = key.function_name.replace(".", "_")
        filename = key.source_hash
        if key.config_hash:
            filename += "_" + key.config_hash
        if key.dependency_hash:
            filename += "_" + key.dependency_hash
        ext = ".json.gz" if self._compression else ".json"
        return os.path.join(self._cache_dir, safe_module, safe_func, filename + ext)

    def get(self, key: CacheKey) -> Optional[CacheEntry]:
        with self._lock:
            path = self._entry_path(key)
            if not os.path.exists(path):
                self._stats.record_miss()
                return None
            try:
                data = self._read_file(path)
                entry = CacheEntry.from_dict(data)
                entry.touch()
                self._stats.record_hit()
                return entry
            except Exception:
                self._stats.record_error()
                return None

    def put(self, entry: CacheEntry) -> bool:
        with self._lock:
            path = self._entry_path(entry.key)
            directory = os.path.dirname(path)
            os.makedirs(directory, exist_ok=True)
            try:
                self._write_file_atomic(path, entry.to_dict())
                self._stats.record_write(entry.size_bytes)
                return True
            except Exception:
                self._stats.record_error()
                return False

    def invalidate(self, key: CacheKey) -> bool:
        with self._lock:
            path = self._entry_path(key)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    self._stats.record_invalidation()
                    return True
                except OSError:
                    self._stats.record_error()
            return False

    def invalidate_module(self, module_path: str) -> int:
        with self._lock:
            safe = module_path.replace("/", "_").replace("\\", "_").replace(".", "_")
            mod_dir = os.path.join(self._cache_dir, safe)
            count = 0
            if os.path.exists(mod_dir):
                for root, dirs, files in os.walk(mod_dir):
                    count += len(files)
                try:
                    shutil.rmtree(mod_dir)
                    self._stats.record_invalidation(count)
                except OSError:
                    self._stats.record_error()
            return count

    def invalidate_function(self, module_path: str, function_name: str) -> int:
        with self._lock:
            safe_mod = module_path.replace("/", "_").replace("\\", "_").replace(".", "_")
            safe_func = function_name.replace(".", "_")
            func_dir = os.path.join(self._cache_dir, safe_mod, safe_func)
            count = 0
            if os.path.exists(func_dir):
                for root, dirs, files in os.walk(func_dir):
                    count += len(files)
                try:
                    shutil.rmtree(func_dir)
                    self._stats.record_invalidation(count)
                except OSError:
                    self._stats.record_error()
            return count

    def clear(self) -> int:
        with self._lock:
            count = 0
            for item in os.listdir(self._cache_dir):
                if item.startswith("."):
                    continue
                item_path = os.path.join(self._cache_dir, item)
                if os.path.isdir(item_path):
                    for root, dirs, files in os.walk(item_path):
                        count += len(files)
                    try:
                        shutil.rmtree(item_path)
                    except OSError:
                        pass
            self._stats.record_invalidation(count)
            return count

    def _read_file(self, path: str) -> Dict[str, Any]:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(path, "r") as f:
                return json.load(f)

    def _write_file_atomic(self, path: str, data: Dict[str, Any]) -> None:
        directory = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            json_bytes = json.dumps(data, default=str).encode("utf-8")
            if path.endswith(".gz"):
                compressed = gzip.compress(json_bytes)
                os.write(fd, compressed)
                ratio = len(compressed) / max(len(json_bytes), 1)
                self._stats.compression_ratio = ratio
            else:
                os.write(fd, json_bytes)
            os.close(fd)
            os.replace(tmp_path, path)
        except Exception:
            os.close(fd)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def total_size(self) -> int:
        total = 0
        for root, dirs, files in os.walk(self._cache_dir):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    def entry_count(self) -> int:
        count = 0
        for root, dirs, files in os.walk(self._cache_dir):
            for f in files:
                if f.endswith((".json", ".json.gz")):
                    count += 1
        return count

    def list_entries(self) -> List[str]:
        entries: List[str] = []
        for root, dirs, files in os.walk(self._cache_dir):
            for f in files:
                if f.endswith((".json", ".json.gz")):
                    entries.append(os.path.relpath(os.path.join(root, f), self._cache_dir))
        return entries

    @property
    def statistics(self) -> CacheStatistics:
        self._stats.entry_count = self.entry_count()
        self._stats.total_size_bytes = self.total_size()
        return self._stats

    def cleanup_stale(self, max_age_seconds: float) -> int:
        cutoff = time.time() - max_age_seconds
        removed = 0
        for root, dirs, files in os.walk(self._cache_dir):
            for f in files:
                if not f.endswith((".json", ".json.gz")):
                    continue
                fpath = os.path.join(root, f)
                try:
                    if os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
                        removed += 1
                except OSError:
                    pass
        self._cleanup_empty_dirs()
        return removed

    def _cleanup_empty_dirs(self) -> None:
        for root, dirs, files in os.walk(self._cache_dir, topdown=False):
            if root == self._cache_dir:
                continue
            if not os.listdir(root):
                try:
                    os.rmdir(root)
                except OSError:
                    pass

    def enforce_size_limit(self) -> int:
        current = self.total_size()
        if current <= self._max_size_bytes:
            return 0

        entries_with_mtime: List[Tuple[str, float, int]] = []
        for root, dirs, files in os.walk(self._cache_dir):
            for f in files:
                if not f.endswith((".json", ".json.gz")):
                    continue
                fpath = os.path.join(root, f)
                try:
                    st = os.stat(fpath)
                    entries_with_mtime.append((fpath, st.st_mtime, st.st_size))
                except OSError:
                    pass

        entries_with_mtime.sort(key=lambda x: x[1])
        removed = 0
        for fpath, _, fsize in entries_with_mtime:
            if current <= self._max_size_bytes * 0.8:
                break
            try:
                os.remove(fpath)
                current -= fsize
                removed += 1
            except OSError:
                pass

        self._cleanup_empty_dirs()
        return removed


# ---------------------------------------------------------------------------
# ContentAddressableStore
# ---------------------------------------------------------------------------

class ContentAddressableStore:
    """Content-addressed storage using SHA-256 for deduplication."""

    def __init__(self, store_dir: str) -> None:
        self._store_dir = store_dir
        self._lock = threading.RLock()
        os.makedirs(store_dir, exist_ok=True)

    def _hash_content(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _blob_path(self, content_hash: str) -> str:
        prefix = content_hash[:2]
        return os.path.join(self._store_dir, prefix, content_hash)

    def store(self, content: Union[str, bytes, Dict[str, Any]]) -> str:
        with self._lock:
            if isinstance(content, dict):
                raw = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
            elif isinstance(content, str):
                raw = content.encode("utf-8")
            else:
                raw = content

            content_hash = self._hash_content(raw)
            blob_path = self._blob_path(content_hash)

            if os.path.exists(blob_path):
                return content_hash

            directory = os.path.dirname(blob_path)
            os.makedirs(directory, exist_ok=True)

            compressed = gzip.compress(raw)
            fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
            try:
                os.write(fd, compressed)
                os.close(fd)
                os.replace(tmp, blob_path)
            except Exception:
                os.close(fd)
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise

            return content_hash

    def retrieve(self, content_hash: str) -> Optional[bytes]:
        with self._lock:
            blob_path = self._blob_path(content_hash)
            if not os.path.exists(blob_path):
                return None
            try:
                with open(blob_path, "rb") as f:
                    compressed = f.read()
                return gzip.decompress(compressed)
            except Exception:
                return None

    def retrieve_json(self, content_hash: str) -> Optional[Dict[str, Any]]:
        raw = self.retrieve(content_hash)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def retrieve_str(self, content_hash: str) -> Optional[str]:
        raw = self.retrieve(content_hash)
        if raw is None:
            return None
        return raw.decode("utf-8")

    def contains(self, content_hash: str) -> bool:
        return os.path.exists(self._blob_path(content_hash))

    def delete(self, content_hash: str) -> bool:
        blob_path = self._blob_path(content_hash)
        if os.path.exists(blob_path):
            try:
                os.remove(blob_path)
                return True
            except OSError:
                return False
        return False

    def total_size(self) -> int:
        total = 0
        for root, dirs, files in os.walk(self._store_dir):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    def blob_count(self) -> int:
        count = 0
        for root, dirs, files in os.walk(self._store_dir):
            count += len(files)
        return count

    def gc(self, referenced_hashes: Set[str]) -> int:
        removed = 0
        for root, dirs, files in os.walk(self._store_dir):
            for f in files:
                if f.endswith(".tmp"):
                    continue
                if f not in referenced_hashes:
                    try:
                        os.remove(os.path.join(root, f))
                        removed += 1
                    except OSError:
                        pass
        return removed

    def verify_integrity(self) -> List[str]:
        corrupted: List[str] = []
        for root, dirs, files in os.walk(self._store_dir):
            for f in files:
                if f.endswith(".tmp"):
                    continue
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, "rb") as fh:
                        compressed = fh.read()
                    raw = gzip.decompress(compressed)
                    actual_hash = self._hash_content(raw)
                    if actual_hash != f:
                        corrupted.append(f)
                except Exception:
                    corrupted.append(f)
        return corrupted


# ---------------------------------------------------------------------------
# CacheInvalidator
# ---------------------------------------------------------------------------

class CacheInvalidator:
    """Smart cache invalidation with dependency tracking and predicate sensitivity."""

    def __init__(
        self,
        memory_cache: MemoryCache,
        disk_cache: Optional[DiskCache] = None,
    ) -> None:
        self._memory = memory_cache
        self._disk = disk_cache
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_deps: Dict[str, Set[str]] = defaultdict(set)
        self._predicate_map: Dict[str, PredicateSet] = {}

    def register_dependency(self, func_id: str, depends_on: str) -> None:
        self._dependency_graph[func_id].add(depends_on)
        self._reverse_deps[depends_on].add(func_id)

    def register_predicates(self, func_id: str, predicates: PredicateSet) -> None:
        self._predicate_map[func_id] = predicates

    def invalidate_function(self, func_id: str) -> Set[str]:
        invalidated: Set[str] = {func_id}
        self._do_invalidate(func_id)

        queue = list(self._reverse_deps.get(func_id, set()))
        visited: Set[str] = {func_id}
        while queue:
            dep = queue.pop()
            if dep in visited:
                continue
            visited.add(dep)
            invalidated.add(dep)
            self._do_invalidate(dep)
            queue.extend(self._reverse_deps.get(dep, set()))

        return invalidated

    def invalidate_function_predicate_sensitive(
        self,
        func_id: str,
        changed_predicates: PredicateSet,
    ) -> Set[str]:
        invalidated: Set[str] = {func_id}
        self._do_invalidate(func_id)

        if changed_predicates.is_empty():
            return invalidated

        queue = list(self._reverse_deps.get(func_id, set()))
        visited: Set[str] = {func_id}
        while queue:
            dep = queue.pop()
            if dep in visited:
                continue
            visited.add(dep)

            dep_preds = self._predicate_map.get(dep, PredicateSet())
            if dep_preds.overlaps(changed_predicates):
                invalidated.add(dep)
                self._do_invalidate(dep)
                queue.extend(self._reverse_deps.get(dep, set()))

        return invalidated

    def invalidate_module(self, module_path: str) -> int:
        count = self._memory.invalidate_by_prefix(module_path)
        if self._disk:
            count += self._disk.invalidate_module(module_path)
        return count

    def invalidate_all(self) -> int:
        count = self._memory.clear()
        if self._disk:
            count += self._disk.clear()
        return count

    def invalidate_by_dependency(self, changed_dep: str) -> Set[str]:
        return self.invalidate_function(changed_dep)

    def _do_invalidate(self, func_id: str) -> None:
        to_remove = [k for k in self._memory.keys() if func_id in k]
        for k in to_remove:
            parts = k.split(":")
            if len(parts) >= 3:
                key = CacheKey(module_path=parts[0], function_name=parts[1], source_hash=parts[2])
                self._memory.invalidate(key)

        if self._disk and ":" in func_id:
            parts = func_id.split(":", 1)
            if len(parts) == 2:
                self._disk.invalidate_function(parts[0], parts[1])

    def get_dependency_chain(self, func_id: str) -> List[str]:
        chain: List[str] = []
        visited: Set[str] = set()
        queue = [func_id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            chain.append(current)
            queue.extend(self._reverse_deps.get(current, set()))
        return chain


# ---------------------------------------------------------------------------
# AnalysisCache — multi-level cache
# ---------------------------------------------------------------------------

class AnalysisCache:
    """
    Main analysis cache with multi-level architecture:
    memory → disk → miss.

    Thread-safe with fine-grained locking.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        memory_max_entries: int = 5000,
        memory_max_bytes: int = 50 * 1024 * 1024,
        disk_max_bytes: int = 500 * 1024 * 1024,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        enable_disk: bool = True,
        enable_compression: bool = True,
    ) -> None:
        self._memory = MemoryCache(
            max_entries=memory_max_entries,
            max_size_bytes=memory_max_bytes,
            eviction_policy=eviction_policy,
        )
        self._disk: Optional[DiskCache] = None
        if enable_disk and cache_dir:
            self._disk = DiskCache(
                cache_dir=cache_dir,
                max_size_bytes=disk_max_bytes,
                compression=enable_compression,
            )
        self._cas: Optional[ContentAddressableStore] = None
        if cache_dir:
            self._cas = ContentAddressableStore(os.path.join(cache_dir, "cas"))
        self._invalidator = CacheInvalidator(self._memory, self._disk)
        self._lock = threading.RLock()
        self._analysis_version = 1
        self._config_version = 1

    def get(
        self,
        key: CacheKey,
        current_dependencies: Optional[Dict[str, str]] = None,
    ) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._memory.get(key)
            if entry is not None:
                if entry.is_valid(current_dependencies):
                    return entry
                else:
                    self._memory.invalidate(key)
                    return None

            if self._disk:
                entry = self._disk.get(key)
                if entry is not None:
                    if entry.is_valid(current_dependencies):
                        self._memory.put(entry)
                        return entry
                    else:
                        self._disk.invalidate(key)
                        return None

            return None

    def put(
        self,
        key: CacheKey,
        result: Dict[str, Any],
        dependencies: Optional[List[str]] = None,
        predicates: Optional[PredicateSet] = None,
    ) -> CacheEntry:
        with self._lock:
            entry = CacheEntry(
                key=key,
                result=result,
                analysis_version=self._analysis_version,
                config_version=self._config_version,
                dependencies=dependencies or [],
                predicates=predicates or PredicateSet(),
            )
            self._memory.put(entry)
            if self._disk:
                self._disk.put(entry)
            if self._cas:
                self._cas.store(result)
            if dependencies:
                func_id = f"{key.module_path}:{key.function_name}"
                for dep in dependencies:
                    self._invalidator.register_dependency(func_id, dep)
            if predicates:
                func_id = f"{key.module_path}:{key.function_name}"
                self._invalidator.register_predicates(func_id, predicates)
            return entry

    def invalidate_function(self, func_id: str) -> Set[str]:
        with self._lock:
            return self._invalidator.invalidate_function(func_id)

    def invalidate_function_predicate_sensitive(
        self, func_id: str, changed_predicates: PredicateSet
    ) -> Set[str]:
        with self._lock:
            return self._invalidator.invalidate_function_predicate_sensitive(
                func_id, changed_predicates
            )

    def invalidate_module(self, module_path: str) -> int:
        with self._lock:
            return self._invalidator.invalidate_module(module_path)

    def invalidate_all(self) -> int:
        with self._lock:
            return self._invalidator.invalidate_all()

    def get_statistics(self) -> Dict[str, Any]:
        mem_stats = self._memory.statistics
        result: Dict[str, Any] = {"memory": mem_stats.to_dict()}
        if self._disk:
            result["disk"] = self._disk.statistics.to_dict()
        if self._cas:
            result["cas"] = {
                "blob_count": self._cas.blob_count(),
                "total_size": self._cas.total_size(),
            }
        combined = CacheStatistics(
            hit_count=mem_stats.hit_count + (self._disk.statistics.hit_count if self._disk else 0),
            miss_count=mem_stats.miss_count,
            entry_count=mem_stats.entry_count,
        )
        result["combined"] = combined.to_dict()
        return result

    @property
    def memory_cache(self) -> MemoryCache:
        return self._memory

    @property
    def disk_cache(self) -> Optional[DiskCache]:
        return self._disk

    @property
    def content_store(self) -> Optional[ContentAddressableStore]:
        return self._cas

    @property
    def invalidator(self) -> CacheInvalidator:
        return self._invalidator

    def set_analysis_version(self, version: int) -> None:
        self._analysis_version = version

    def set_config_version(self, version: int) -> None:
        self._config_version = version


# ---------------------------------------------------------------------------
# CacheWarmer
# ---------------------------------------------------------------------------

class CacheWarmer:
    """Pre-warms the cache using various strategies."""

    def __init__(self, cache: AnalysisCache) -> None:
        self._cache = cache

    def warm_from_disk(self) -> int:
        disk = self._cache.disk_cache
        if not disk:
            return 0
        count = 0
        for entry_path in disk.list_entries():
            parts = entry_path.split(os.sep)
            if len(parts) >= 3:
                module = parts[0].replace("_", "/")
                func = parts[1].replace("_", ".")
                hash_part = parts[2].split(".")[0]
                key = CacheKey(module_path=module, function_name=func, source_hash=hash_part)
                entry = disk.get(key)
                if entry:
                    self._cache.memory_cache.put(entry)
                    count += 1
        return count

    def warm_commonly_used(
        self,
        sources: Dict[str, str],
        import_counts: Optional[Dict[str, int]] = None,
        analyze_func: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    ) -> int:
        if not analyze_func:
            return 0

        if import_counts:
            sorted_modules = sorted(import_counts.items(), key=lambda x: -x[1])
            modules_to_warm = [m for m, _ in sorted_modules[:20]]
        else:
            modules_to_warm = list(sources.keys())[:20]

        count = 0
        for mod in modules_to_warm:
            source = sources.get(mod, "")
            if not source:
                continue
            try:
                result = analyze_func(mod, source)
                key = CacheKey.compute(mod, "<module>", source)
                self._cache.put(key, result)
                count += 1
            except Exception:
                pass
        return count

    def warm_from_git_history(
        self,
        repo_dir: str,
        analyze_func: Optional[Callable[[str, str], Dict[str, Any]]] = None,
        max_commits: int = 10,
    ) -> int:
        if not analyze_func:
            return 0
        count = 0
        try:
            import subprocess
            result = subprocess.run(
                ["git", "log", f"--max-count={max_commits}", "--name-only", "--pretty=format:"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            files = set()
            for line in result.stdout.splitlines():
                line = line.strip()
                if line and line.endswith(".py"):
                    files.add(line)

            for fpath in files:
                full_path = os.path.join(repo_dir, fpath)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, "r") as f:
                            source = f.read()
                        analysis_result = analyze_func(fpath, source)
                        key = CacheKey.compute(fpath, "<module>", source)
                        self._cache.put(key, analysis_result)
                        count += 1
                    except Exception:
                        pass
        except Exception:
            pass
        return count

    def warm_from_ci_artifacts(
        self,
        artifact_dir: str,
    ) -> int:
        count = 0
        if not os.path.exists(artifact_dir):
            return 0
        for root, dirs, files in os.walk(artifact_dir):
            for f in files:
                if not f.endswith((".json", ".json.gz")):
                    continue
                fpath = os.path.join(root, f)
                try:
                    if f.endswith(".gz"):
                        with gzip.open(fpath, "rt") as fh:
                            data = json.load(fh)
                    else:
                        with open(fpath, "r") as fh:
                            data = json.load(fh)
                    if "key" in data and "result" in data:
                        entry = CacheEntry.from_dict(data)
                        self._cache.memory_cache.put(entry)
                        count += 1
                except Exception:
                    pass
        return count


# ---------------------------------------------------------------------------
# CacheMigration
# ---------------------------------------------------------------------------

class CacheMigration:
    """Handle cache format version changes."""

    CURRENT_VERSION = 2

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = cache_dir

    def get_version(self) -> int:
        version_path = os.path.join(self._cache_dir, ".cache-version")
        try:
            with open(version_path, "r") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return 0

    def set_version(self, version: int) -> None:
        version_path = os.path.join(self._cache_dir, ".cache-version")
        with open(version_path, "w") as f:
            f.write(str(version))

    def needs_migration(self) -> bool:
        return self.get_version() < self.CURRENT_VERSION

    def migrate(self) -> bool:
        current = self.get_version()
        if current >= self.CURRENT_VERSION:
            return True

        if current == 0:
            success = self._migrate_v0_to_v1()
            if not success:
                return False
            current = 1

        if current == 1:
            success = self._migrate_v1_to_v2()
            if not success:
                return False
            current = 2

        self.set_version(self.CURRENT_VERSION)
        return True

    def _migrate_v0_to_v1(self) -> bool:
        """v0 had flat JSON files; v1 uses directory structure."""
        try:
            for f in os.listdir(self._cache_dir):
                if not f.endswith(".json"):
                    continue
                fpath = os.path.join(self._cache_dir, f)
                try:
                    with open(fpath, "r") as fh:
                        data = json.load(fh)
                    if "module_path" in data and "function_name" in data:
                        mod = data["module_path"].replace("/", "_").replace(".", "_")
                        func = data["function_name"].replace(".", "_")
                        dest_dir = os.path.join(self._cache_dir, mod, func)
                        os.makedirs(dest_dir, exist_ok=True)
                        src_hash = data.get("source_hash", hashlib.sha256(f.encode()).hexdigest()[:16])
                        dest = os.path.join(dest_dir, src_hash + ".json")
                        shutil.move(fpath, dest)
                except (json.JSONDecodeError, KeyError, OSError):
                    pass
            return True
        except OSError:
            return False

    def _migrate_v1_to_v2(self) -> bool:
        """v2 adds compression (gzip) to stored entries."""
        try:
            for root, dirs, files in os.walk(self._cache_dir):
                for f in files:
                    if not f.endswith(".json"):
                        continue
                    fpath = os.path.join(root, f)
                    try:
                        with open(fpath, "r") as fh:
                            data = json.load(fh)

                        if "analysis_version" not in data:
                            data["analysis_version"] = 1

                        gz_path = fpath + ".gz"
                        json_bytes = json.dumps(data, default=str).encode("utf-8")
                        with gzip.open(gz_path, "wb") as gz:
                            gz.write(json_bytes)
                        os.remove(fpath)
                    except (json.JSONDecodeError, OSError):
                        pass
            return True
        except OSError:
            return False

    def cleanup_incompatible(self) -> int:
        current = self.get_version()
        if current >= self.CURRENT_VERSION:
            return 0
        count = 0
        for root, dirs, files in os.walk(self._cache_dir):
            for f in files:
                if f.startswith("."):
                    continue
                fpath = os.path.join(root, f)
                try:
                    os.remove(fpath)
                    count += 1
                except OSError:
                    pass
        self.set_version(self.CURRENT_VERSION)
        return count


# ---------------------------------------------------------------------------
# CacheGarbageCollector
# ---------------------------------------------------------------------------

class CacheGarbageCollector:
    """Cleans up stale or oversized cache entries."""

    def __init__(self, cache_dir: str, max_size_bytes: int = 500 * 1024 * 1024) -> None:
        self._cache_dir = cache_dir
        self._max_size = max_size_bytes

    def remove_older_than(self, max_age_seconds: float) -> int:
        cutoff = time.time() - max_age_seconds
        removed = 0
        for root, dirs, files in os.walk(self._cache_dir):
            for f in files:
                if f.startswith("."):
                    continue
                fpath = os.path.join(root, f)
                try:
                    if os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
                        removed += 1
                except OSError:
                    pass
        self._cleanup_empty_dirs()
        return removed

    def remove_for_deleted_files(self, existing_files: Set[str]) -> int:
        removed = 0
        for item in os.listdir(self._cache_dir):
            if item.startswith("."):
                continue
            item_path = os.path.join(self._cache_dir, item)
            if not os.path.isdir(item_path):
                continue
            mod_name = item.replace("_", "/")
            found = False
            for ef in existing_files:
                normalized = ef.replace("/", "_").replace("\\", "_").replace(".", "_")
                if normalized == item or mod_name in ef:
                    found = True
                    break
            if not found:
                try:
                    count = sum(len(fs) for _, _, fs in os.walk(item_path))
                    shutil.rmtree(item_path)
                    removed += count
                except OSError:
                    pass
        return removed

    def enforce_size_limit(self) -> int:
        total = 0
        entries: List[Tuple[str, float, int]] = []
        for root, dirs, files in os.walk(self._cache_dir):
            for f in files:
                if f.startswith("."):
                    continue
                fpath = os.path.join(root, f)
                try:
                    st = os.stat(fpath)
                    total += st.st_size
                    entries.append((fpath, st.st_mtime, st.st_size))
                except OSError:
                    pass

        if total <= self._max_size:
            return 0

        entries.sort(key=lambda x: x[1])
        removed = 0
        for fpath, _, fsize in entries:
            if total <= self._max_size * 0.8:
                break
            try:
                os.remove(fpath)
                total -= fsize
                removed += 1
            except OSError:
                pass

        self._cleanup_empty_dirs()
        return removed

    def compact(self) -> int:
        rewritten = 0
        for root, dirs, files in os.walk(self._cache_dir):
            for f in files:
                if not f.endswith(".json"):
                    continue
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, "r") as fh:
                        data = json.load(fh)
                    compact_json = json.dumps(data, separators=(",", ":"), default=str)
                    gz_path = fpath + ".gz"
                    with gzip.open(gz_path, "wt") as gz:
                        gz.write(compact_json)
                    os.remove(fpath)
                    rewritten += 1
                except (json.JSONDecodeError, OSError):
                    pass
        return rewritten

    def _cleanup_empty_dirs(self) -> None:
        for root, dirs, files in os.walk(self._cache_dir, topdown=False):
            if root == self._cache_dir:
                continue
            try:
                if not os.listdir(root):
                    os.rmdir(root)
            except OSError:
                pass

    def full_gc(self, existing_files: Optional[Set[str]] = None, max_age_seconds: float = 86400 * 30) -> Dict[str, int]:
        results: Dict[str, int] = {}
        results["stale"] = self.remove_older_than(max_age_seconds)
        if existing_files is not None:
            results["deleted_files"] = self.remove_for_deleted_files(existing_files)
        results["size_limit"] = self.enforce_size_limit()
        results["compacted"] = self.compact()
        return results


# ---------------------------------------------------------------------------
# DistributedCache
# ---------------------------------------------------------------------------

class StorageBackend(Enum):
    LOCAL = auto()
    S3 = auto()
    GCS = auto()
    AZURE = auto()
    HTTP = auto()


@dataclass
class RemoteCacheConfig:
    """Configuration for distributed cache."""
    backend: StorageBackend = StorageBackend.LOCAL
    endpoint: str = ""
    bucket: str = ""
    prefix: str = "reftype-cache"
    access_key: str = ""
    secret_key: str = ""
    region: str = ""
    timeout_seconds: int = 30
    max_retries: int = 3
    upload_concurrency: int = 4
    download_concurrency: int = 4


class DistributedCache:
    """
    Distributed caching layer for CI/CD environments.

    Supports uploading/downloading cache entries to remote storage.
    Currently implements a local filesystem backend; S3/GCS/Azure/HTTP
    backends provide the interface with placeholder implementations.
    """

    def __init__(self, config: RemoteCacheConfig, local_cache_dir: str) -> None:
        self._config = config
        self._local_dir = local_cache_dir
        self._lock = threading.Lock()
        self._stats = CacheStatistics()

        if config.backend == StorageBackend.LOCAL:
            self._remote_dir = os.path.join(local_cache_dir, ".remote")
            os.makedirs(self._remote_dir, exist_ok=True)

    def upload(self, entries: Dict[str, CacheEntry]) -> int:
        uploaded = 0
        for key_str, entry in entries.items():
            success = self._upload_one(key_str, entry)
            if success:
                uploaded += 1
        return uploaded

    def download(self, keys: List[str]) -> Dict[str, CacheEntry]:
        results: Dict[str, CacheEntry] = {}
        for key_str in keys:
            entry = self._download_one(key_str)
            if entry:
                results[key_str] = entry
        return results

    def _upload_one(self, key_str: str, entry: CacheEntry) -> bool:
        if self._config.backend == StorageBackend.LOCAL:
            return self._upload_local(key_str, entry)
        elif self._config.backend == StorageBackend.HTTP:
            return self._upload_http(key_str, entry)
        elif self._config.backend == StorageBackend.S3:
            return self._upload_s3(key_str, entry)
        elif self._config.backend == StorageBackend.GCS:
            return self._upload_gcs(key_str, entry)
        elif self._config.backend == StorageBackend.AZURE:
            return self._upload_azure(key_str, entry)
        return False

    def _download_one(self, key_str: str) -> Optional[CacheEntry]:
        if self._config.backend == StorageBackend.LOCAL:
            return self._download_local(key_str)
        elif self._config.backend == StorageBackend.HTTP:
            return self._download_http(key_str)
        elif self._config.backend == StorageBackend.S3:
            return self._download_s3(key_str)
        elif self._config.backend == StorageBackend.GCS:
            return self._download_gcs(key_str)
        elif self._config.backend == StorageBackend.AZURE:
            return self._download_azure(key_str)
        return None

    # -- local backend -------------------------------------------------------

    def _upload_local(self, key_str: str, entry: CacheEntry) -> bool:
        safe_key = key_str.replace("/", "_").replace(":", "_")
        path = os.path.join(self._remote_dir, safe_key + ".json.gz")
        try:
            data = json.dumps(entry.to_dict(), default=str).encode("utf-8")
            compressed = gzip.compress(data)
            fd, tmp = tempfile.mkstemp(dir=self._remote_dir, suffix=".tmp")
            os.write(fd, compressed)
            os.close(fd)
            os.replace(tmp, path)
            self._stats.record_write(len(compressed))
            return True
        except Exception:
            self._stats.record_error()
            return False

    def _download_local(self, key_str: str) -> Optional[CacheEntry]:
        safe_key = key_str.replace("/", "_").replace(":", "_")
        path = os.path.join(self._remote_dir, safe_key + ".json.gz")
        if not os.path.exists(path):
            self._stats.record_miss()
            return None
        try:
            with gzip.open(path, "rt") as f:
                data = json.load(f)
            self._stats.record_hit()
            return CacheEntry.from_dict(data)
        except Exception:
            self._stats.record_error()
            return None

    # -- HTTP backend --------------------------------------------------------

    def _upload_http(self, key_str: str, entry: CacheEntry) -> bool:
        try:
            import urllib.request
            url = f"{self._config.endpoint}/{self._config.prefix}/{key_str}"
            data = json.dumps(entry.to_dict(), default=str).encode("utf-8")
            compressed = gzip.compress(data)
            req = urllib.request.Request(
                url, data=compressed, method="PUT",
                headers={"Content-Type": "application/gzip"},
            )
            urllib.request.urlopen(req, timeout=self._config.timeout_seconds)
            self._stats.record_write(len(compressed))
            return True
        except Exception:
            self._stats.record_error()
            return False

    def _download_http(self, key_str: str) -> Optional[CacheEntry]:
        try:
            import urllib.request
            url = f"{self._config.endpoint}/{self._config.prefix}/{key_str}"
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=self._config.timeout_seconds)
            compressed = resp.read()
            data = json.loads(gzip.decompress(compressed).decode("utf-8"))
            self._stats.record_hit()
            return CacheEntry.from_dict(data)
        except Exception:
            self._stats.record_miss()
            return None

    # -- S3 backend (interface only) ----------------------------------------

    def _upload_s3(self, key_str: str, entry: CacheEntry) -> bool:
        try:
            import urllib.request
            data = json.dumps(entry.to_dict(), default=str).encode("utf-8")
            compressed = gzip.compress(data)
            s3_key = f"{self._config.prefix}/{key_str}.json.gz"
            host = f"{self._config.bucket}.s3.{self._config.region}.amazonaws.com"
            url = f"https://{host}/{s3_key}"
            req = urllib.request.Request(
                url, data=compressed, method="PUT",
                headers={
                    "Content-Type": "application/gzip",
                    "x-amz-content-sha256": hashlib.sha256(compressed).hexdigest(),
                },
            )
            urllib.request.urlopen(req, timeout=self._config.timeout_seconds)
            self._stats.record_write(len(compressed))
            return True
        except Exception:
            self._stats.record_error()
            return False

    def _download_s3(self, key_str: str) -> Optional[CacheEntry]:
        try:
            import urllib.request
            s3_key = f"{self._config.prefix}/{key_str}.json.gz"
            host = f"{self._config.bucket}.s3.{self._config.region}.amazonaws.com"
            url = f"https://{host}/{s3_key}"
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=self._config.timeout_seconds)
            compressed = resp.read()
            data = json.loads(gzip.decompress(compressed).decode("utf-8"))
            self._stats.record_hit()
            return CacheEntry.from_dict(data)
        except Exception:
            self._stats.record_miss()
            return None

    # -- GCS backend (interface only) ---------------------------------------

    def _upload_gcs(self, key_str: str, entry: CacheEntry) -> bool:
        try:
            import urllib.request
            data = json.dumps(entry.to_dict(), default=str).encode("utf-8")
            compressed = gzip.compress(data)
            gcs_key = f"{self._config.prefix}/{key_str}.json.gz"
            url = (
                f"https://storage.googleapis.com/upload/storage/v1/b/"
                f"{self._config.bucket}/o?uploadType=media&name={gcs_key}"
            )
            req = urllib.request.Request(
                url, data=compressed, method="POST",
                headers={"Content-Type": "application/gzip"},
            )
            urllib.request.urlopen(req, timeout=self._config.timeout_seconds)
            self._stats.record_write(len(compressed))
            return True
        except Exception:
            self._stats.record_error()
            return False

    def _download_gcs(self, key_str: str) -> Optional[CacheEntry]:
        try:
            import urllib.request
            gcs_key = f"{self._config.prefix}/{key_str}.json.gz"
            url = (
                f"https://storage.googleapis.com/storage/v1/b/"
                f"{self._config.bucket}/o/{gcs_key}?alt=media"
            )
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=self._config.timeout_seconds)
            compressed = resp.read()
            data = json.loads(gzip.decompress(compressed).decode("utf-8"))
            self._stats.record_hit()
            return CacheEntry.from_dict(data)
        except Exception:
            self._stats.record_miss()
            return None

    # -- Azure backend (interface only) -------------------------------------

    def _upload_azure(self, key_str: str, entry: CacheEntry) -> bool:
        try:
            import urllib.request
            data = json.dumps(entry.to_dict(), default=str).encode("utf-8")
            compressed = gzip.compress(data)
            blob_name = f"{self._config.prefix}/{key_str}.json.gz"
            url = f"{self._config.endpoint}/{self._config.bucket}/{blob_name}"
            req = urllib.request.Request(
                url, data=compressed, method="PUT",
                headers={
                    "Content-Type": "application/gzip",
                    "x-ms-blob-type": "BlockBlob",
                },
            )
            urllib.request.urlopen(req, timeout=self._config.timeout_seconds)
            self._stats.record_write(len(compressed))
            return True
        except Exception:
            self._stats.record_error()
            return False

    def _download_azure(self, key_str: str) -> Optional[CacheEntry]:
        try:
            import urllib.request
            blob_name = f"{self._config.prefix}/{key_str}.json.gz"
            url = f"{self._config.endpoint}/{self._config.bucket}/{blob_name}"
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=self._config.timeout_seconds)
            compressed = resp.read()
            data = json.loads(gzip.decompress(compressed).decode("utf-8"))
            self._stats.record_hit()
            return CacheEntry.from_dict(data)
        except Exception:
            self._stats.record_miss()
            return None

    def list_remote_keys(self) -> List[str]:
        if self._config.backend == StorageBackend.LOCAL:
            keys: List[str] = []
            for f in os.listdir(self._remote_dir):
                if f.endswith(".json.gz"):
                    key = f[:-8].replace("_", "/")
                    keys.append(key)
            return keys
        return []

    @property
    def statistics(self) -> CacheStatistics:
        return self._stats


# ---------------------------------------------------------------------------
# CacheCoordinator — high-level orchestrator
# ---------------------------------------------------------------------------

class CacheCoordinator:
    """
    Coordinates all cache layers and provides a unified interface
    for the incremental analysis engine.
    """

    def __init__(
        self,
        cache_dir: str,
        memory_max_entries: int = 5000,
        memory_max_bytes: int = 50 * 1024 * 1024,
        disk_max_bytes: int = 500 * 1024 * 1024,
        enable_distributed: bool = False,
        distributed_config: Optional[RemoteCacheConfig] = None,
    ) -> None:
        self._cache = AnalysisCache(
            cache_dir=cache_dir,
            memory_max_entries=memory_max_entries,
            memory_max_bytes=memory_max_bytes,
            disk_max_bytes=disk_max_bytes,
        )
        self._warmer = CacheWarmer(self._cache)
        self._gc = CacheGarbageCollector(cache_dir, disk_max_bytes)
        self._migration = CacheMigration(cache_dir)
        self._distributed: Optional[DistributedCache] = None
        if enable_distributed and distributed_config:
            self._distributed = DistributedCache(distributed_config, cache_dir)
        self._cache_dir = cache_dir

    def initialize(self) -> None:
        if self._migration.needs_migration():
            self._migration.migrate()

    def get(
        self,
        module_path: str,
        function_name: str,
        source: str,
        config: Optional[Dict[str, Any]] = None,
        dependency_summaries: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        key = CacheKey.compute(
            module_path, function_name, source, config, dependency_summaries
        )
        entry = self._cache.get(key)
        if entry:
            return entry.result

        if self._distributed:
            remote_entry = self._distributed.download([key.composite_key])
            if remote_entry:
                first = next(iter(remote_entry.values()))
                self._cache.memory_cache.put(first)
                return first.result

        return None

    def put(
        self,
        module_path: str,
        function_name: str,
        source: str,
        result: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        dependency_summaries: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        predicates: Optional[PredicateSet] = None,
    ) -> None:
        key = CacheKey.compute(
            module_path, function_name, source, config, dependency_summaries
        )
        self._cache.put(key, result, dependencies, predicates)

    def invalidate_function(self, func_id: str) -> Set[str]:
        return self._cache.invalidate_function(func_id)

    def invalidate_function_predicate_sensitive(
        self, func_id: str, changed_predicates: PredicateSet
    ) -> Set[str]:
        return self._cache.invalidate_function_predicate_sensitive(
            func_id, changed_predicates
        )

    def invalidate_module(self, module_path: str) -> int:
        return self._cache.invalidate_module(module_path)

    def invalidate_all(self) -> int:
        return self._cache.invalidate_all()

    def warm(
        self,
        sources: Optional[Dict[str, str]] = None,
        analyze_func: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        results: Dict[str, int] = {}
        results["from_disk"] = self._warmer.warm_from_disk()
        if sources and analyze_func:
            results["commonly_used"] = self._warmer.warm_commonly_used(
                sources, analyze_func=analyze_func
            )
        return results

    def gc(
        self,
        existing_files: Optional[Set[str]] = None,
        max_age_seconds: float = 86400 * 30,
    ) -> Dict[str, int]:
        return self._gc.full_gc(existing_files, max_age_seconds)

    def upload_to_remote(self) -> int:
        if not self._distributed:
            return 0
        entries: Dict[str, CacheEntry] = {}
        for entry in self._cache.memory_cache.entries():
            entries[entry.key.composite_key] = entry
        return self._distributed.upload(entries)

    def download_from_remote(self, keys: List[str]) -> int:
        if not self._distributed:
            return 0
        remote = self._distributed.download(keys)
        for entry in remote.values():
            self._cache.memory_cache.put(entry)
        return len(remote)

    def get_statistics(self) -> Dict[str, Any]:
        stats = self._cache.get_statistics()
        if self._distributed:
            stats["distributed"] = self._distributed.statistics.to_dict()
        return stats

    @property
    def analysis_cache(self) -> AnalysisCache:
        return self._cache


# ---------------------------------------------------------------------------
# CacheAwareAnalyzer — wraps an analysis function with caching
# ---------------------------------------------------------------------------

class CacheAwareAnalyzer:
    """
    Wraps a raw analysis function with caching logic.
    Transparently returns cached results when available.
    """

    def __init__(
        self,
        coordinator: CacheCoordinator,
        analyze_func: Callable[[str, str, Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        self._coordinator = coordinator
        self._analyze = analyze_func
        self._call_count = 0
        self._cache_hit_count = 0

    def __call__(
        self,
        func_id: str,
        source: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._call_count += 1
        parts = func_id.rsplit(":", 1)
        module_path = parts[0] if len(parts) > 1 else ""
        function_name = parts[1] if len(parts) > 1 else func_id

        dep_summaries = None
        if context and "callee_summaries" in context:
            dep_summaries = context["callee_summaries"]

        cached = self._coordinator.get(
            module_path, function_name, source,
            dependency_summaries=dep_summaries,
        )
        if cached is not None:
            self._cache_hit_count += 1
            return cached

        result = self._analyze(func_id, source, context or {})

        deps: List[str] = []
        if context and "callee_summaries" in context:
            deps = list(context["callee_summaries"].keys())

        self._coordinator.put(
            module_path, function_name, source, result,
            dependency_summaries=dep_summaries,
            dependencies=deps,
        )
        return result

    @property
    def hit_rate(self) -> float:
        return self._cache_hit_count / self._call_count if self._call_count > 0 else 0.0

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def cache_hit_count(self) -> int:
        return self._cache_hit_count


# ---------------------------------------------------------------------------
# Thread-safe cache wrapper for concurrent analysis
# ---------------------------------------------------------------------------

class ThreadSafeCacheWrapper:
    """Thread-safe wrapper around CacheCoordinator for parallel analysis."""

    def __init__(self, coordinator: CacheCoordinator) -> None:
        self._coordinator = coordinator
        self._lock = threading.RLock()
        self._per_key_locks: Dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()

    def _get_key_lock(self, key: str) -> threading.Lock:
        with self._meta_lock:
            if key not in self._per_key_locks:
                self._per_key_locks[key] = threading.Lock()
            return self._per_key_locks[key]

    def get(
        self,
        module_path: str,
        function_name: str,
        source: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        key_str = f"{module_path}:{function_name}"
        lock = self._get_key_lock(key_str)
        with lock:
            return self._coordinator.get(module_path, function_name, source, config)

    def put(
        self,
        module_path: str,
        function_name: str,
        source: str,
        result: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        key_str = f"{module_path}:{function_name}"
        lock = self._get_key_lock(key_str)
        with lock:
            self._coordinator.put(
                module_path, function_name, source, result,
                config=config, dependencies=dependencies,
            )

    def invalidate(self, func_id: str) -> Set[str]:
        with self._lock:
            return self._coordinator.invalidate_function(func_id)

    def cleanup_locks(self) -> None:
        with self._meta_lock:
            self._per_key_locks.clear()


# ---------------------------------------------------------------------------
# Snapshot / restore for cache state
# ---------------------------------------------------------------------------

class CacheSnapshot:
    """Save and restore entire cache state for debugging and testing."""

    def __init__(self, coordinator: CacheCoordinator) -> None:
        self._coordinator = coordinator

    def take_snapshot(self, output_path: str) -> int:
        entries = self._coordinator.analysis_cache.memory_cache.entries()
        data: List[Dict[str, Any]] = [e.to_dict() for e in entries]
        json_bytes = json.dumps(data, default=str).encode("utf-8")
        compressed = gzip.compress(json_bytes)
        with open(output_path, "wb") as f:
            f.write(compressed)
        return len(entries)

    def restore_snapshot(self, input_path: str) -> int:
        with open(input_path, "rb") as f:
            compressed = f.read()
        json_bytes = gzip.decompress(compressed)
        data = json.loads(json_bytes.decode("utf-8"))
        count = 0
        for entry_data in data:
            try:
                entry = CacheEntry.from_dict(entry_data)
                self._coordinator.analysis_cache.memory_cache.put(entry)
                count += 1
            except Exception:
                pass
        return count

    def diff_snapshots(self, path_a: str, path_b: str) -> Dict[str, Any]:
        with open(path_a, "rb") as f:
            data_a = json.loads(gzip.decompress(f.read()))
        with open(path_b, "rb") as f:
            data_b = json.loads(gzip.decompress(f.read()))

        keys_a = {d.get("key", {}).get("source_hash", str(i)) for i, d in enumerate(data_a)}
        keys_b = {d.get("key", {}).get("source_hash", str(i)) for i, d in enumerate(data_b)}

        return {
            "entries_a": len(data_a),
            "entries_b": len(data_b),
            "added": len(keys_b - keys_a),
            "removed": len(keys_a - keys_b),
            "common": len(keys_a & keys_b),
        }


# ---------------------------------------------------------------------------
# Cache health monitor
# ---------------------------------------------------------------------------

class CacheHealthMonitor:
    """Monitors cache health and emits warnings."""

    def __init__(self, coordinator: CacheCoordinator) -> None:
        self._coordinator = coordinator
        self._warnings: List[str] = []
        self._checks_run = 0

    def check_health(self) -> Dict[str, Any]:
        self._warnings.clear()
        self._checks_run += 1

        stats = self._coordinator.get_statistics()
        mem = stats.get("memory", {})
        disk = stats.get("disk", {})

        hit_rate = mem.get("hit_rate", 0)
        if hit_rate < 0.3 and mem.get("read_count", 0) > 100:
            self._warnings.append(
                f"Low cache hit rate: {hit_rate:.1%}. Consider warming the cache."
            )

        evictions = mem.get("eviction_count", 0)
        writes = mem.get("write_count", 0)
        if writes > 0 and evictions / writes > 0.5:
            self._warnings.append(
                f"High eviction rate: {evictions}/{writes}. Consider increasing cache size."
            )

        errors = mem.get("error_count", 0) + disk.get("error_count", 0)
        if errors > 10:
            self._warnings.append(f"Cache errors detected: {errors}")

        disk_size = disk.get("total_size_bytes", 0)
        if disk_size > 400 * 1024 * 1024:
            self._warnings.append(
                f"Disk cache approaching limit: {disk_size / 1024 / 1024:.0f} MB"
            )

        return {
            "healthy": len(self._warnings) == 0,
            "warnings": list(self._warnings),
            "stats": stats,
            "checks_run": self._checks_run,
        }

    @property
    def warnings(self) -> List[str]:
        return list(self._warnings)

    @property
    def is_healthy(self) -> bool:
        return len(self._warnings) == 0
