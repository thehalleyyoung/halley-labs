"""
Memoization cache for junction-tree messages.

Provides an LRU cache keyed by (clique_pair, evidence_signature,
intervention_signature) with support for:
  - configurable capacity and eviction
  - boundary-variable invalidation
  - subtree caching for MCTS rollouts
  - hit-rate / memory-usage statistics
  - serialization for checkpoint/restart
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# ------------------------------------------------------------------ #
#  Cache key
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class CacheKey:
    """Hashable key for a cached junction-tree message."""

    sender: FrozenSet[str]
    receiver: FrozenSet[str]
    evidence_sig: str
    intervention_sig: str

    @staticmethod
    def make(
        sender_vars: FrozenSet[str],
        receiver_vars: FrozenSet[str],
        evidence: Optional[Dict[str, int]] = None,
        intervention: Optional[Dict[str, float]] = None,
    ) -> "CacheKey":
        ev_sig = _dict_signature(evidence) if evidence else ""
        iv_sig = _dict_signature(intervention) if intervention else ""
        return CacheKey(
            sender=sender_vars,
            receiver=receiver_vars,
            evidence_sig=ev_sig,
            intervention_sig=iv_sig,
        )


def _dict_signature(d: Dict) -> str:
    """Deterministic hash of a dict for use as a cache-key component."""
    items = sorted(d.items(), key=lambda kv: str(kv[0]))
    raw = json.dumps(items, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ #
#  Cache entry
# ------------------------------------------------------------------ #

@dataclass
class CacheEntry:
    """Wraps a cached message (numpy array) with metadata."""

    key: CacheKey
    value: NDArray
    variables: List[str]
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0

    def __post_init__(self) -> None:
        self.size_bytes = self.value.nbytes

    def touch(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1


# ------------------------------------------------------------------ #
#  Statistics
# ------------------------------------------------------------------ #

@dataclass
class CacheStats:
    """Accumulated cache-performance statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    total_lookups: int = 0
    total_inserts: int = 0
    total_bytes_stored: int = 0
    peak_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.hits / self.total_lookups

    def summary(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "total_lookups": self.total_lookups,
            "total_inserts": self.total_inserts,
            "memory_bytes": self.total_bytes_stored,
            "peak_memory_bytes": self.peak_bytes,
        }


# ------------------------------------------------------------------ #
#  Main cache
# ------------------------------------------------------------------ #

class InferenceCache:
    """LRU memoization cache for junction-tree messages.

    Parameters
    ----------
    capacity : int
        Maximum number of entries before LRU eviction.
    max_memory_bytes : int
        Soft memory cap (bytes).  Entries are evicted when the total
        stored exceeds this limit.  0 means unlimited.
    """

    def __init__(
        self, capacity: int = 4096, max_memory_bytes: int = 0
    ) -> None:
        self.capacity = capacity
        self.max_memory_bytes = max_memory_bytes

        self._store: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._stats = CacheStats()

        # Mapping from variable name to the set of cache keys that
        # depend on it.  Used for targeted invalidation.
        self._var_dependencies: Dict[str, Set[CacheKey]] = {}

        # Subtree cache: maps a root-clique frozenset to the set of
        # cache keys that belong to its subtree.
        self._subtree_groups: Dict[FrozenSet[str], Set[CacheKey]] = {}

    # ------------------------------------------------------------------ #
    #  Lookup / insert
    # ------------------------------------------------------------------ #

    def get(self, key: CacheKey) -> Optional[CacheEntry]:
        """Look up a cached message.  Returns *None* on miss."""
        self._stats.total_lookups += 1
        entry = self._store.get(key)
        if entry is None:
            self._stats.misses += 1
            return None
        self._stats.hits += 1
        entry.touch()
        self._store.move_to_end(key)  # refresh LRU position
        return entry

    def put(
        self,
        key: CacheKey,
        value: NDArray,
        variables: List[str],
        boundary_vars: Optional[Set[str]] = None,
        subtree_root: Optional[FrozenSet[str]] = None,
    ) -> None:
        """Insert or overwrite a cache entry.

        Parameters
        ----------
        boundary_vars : set of variable names on which this entry depends.
            If any of these variables' evidence changes, the entry is
            invalidated.
        subtree_root : if this message belongs to a subtree rooted at a
            particular clique, record it for batch invalidation.
        """
        if key in self._store:
            old = self._store.pop(key)
            self._stats.total_bytes_stored -= old.size_bytes

        entry = CacheEntry(key=key, value=value.copy(), variables=list(variables))
        self._store[key] = entry
        self._stats.total_inserts += 1
        self._stats.total_bytes_stored += entry.size_bytes
        self._stats.peak_bytes = max(
            self._stats.peak_bytes, self._stats.total_bytes_stored
        )

        # Track variable dependencies
        dep_vars = set(key.sender | key.receiver)
        if boundary_vars:
            dep_vars |= boundary_vars
        for v in dep_vars:
            self._var_dependencies.setdefault(v, set()).add(key)

        if subtree_root is not None:
            self._subtree_groups.setdefault(subtree_root, set()).add(key)

        # Eviction
        self._enforce_capacity()
        self._enforce_memory()

    def contains(self, key: CacheKey) -> bool:
        return key in self._store

    # ------------------------------------------------------------------ #
    #  Invalidation
    # ------------------------------------------------------------------ #

    def invalidate_variable(self, variable: str) -> int:
        """Invalidate all entries that depend on *variable*.

        Returns the number of entries removed.
        """
        keys = self._var_dependencies.pop(variable, set())
        count = 0
        for k in keys:
            if k in self._store:
                self._remove(k)
                count += 1
        self._stats.invalidations += count
        return count

    def invalidate_variables(self, variables: Set[str]) -> int:
        """Invalidate entries depending on any variable in the set."""
        total = 0
        for v in variables:
            total += self.invalidate_variable(v)
        return total

    def invalidate_subtree(self, subtree_root: FrozenSet[str]) -> int:
        """Invalidate all entries belonging to a subtree."""
        keys = self._subtree_groups.pop(subtree_root, set())
        count = 0
        for k in keys:
            if k in self._store:
                self._remove(k)
                count += 1
        self._stats.invalidations += count
        return count

    def invalidate_intervention(self, intervention_sig: str) -> int:
        """Invalidate all entries matching a given intervention signature."""
        to_remove = [
            k for k in self._store if k.intervention_sig == intervention_sig
        ]
        for k in to_remove:
            self._remove(k)
        self._stats.invalidations += len(to_remove)
        return len(to_remove)

    def clear(self) -> None:
        """Remove all entries."""
        n = len(self._store)
        self._store.clear()
        self._var_dependencies.clear()
        self._subtree_groups.clear()
        self._stats.total_bytes_stored = 0
        self._stats.evictions += n

    # ------------------------------------------------------------------ #
    #  Eviction helpers
    # ------------------------------------------------------------------ #

    def _enforce_capacity(self) -> None:
        while len(self._store) > self.capacity:
            self._evict_lru()

    def _enforce_memory(self) -> None:
        if self.max_memory_bytes <= 0:
            return
        while (
            self._stats.total_bytes_stored > self.max_memory_bytes
            and self._store
        ):
            self._evict_lru()

    def _evict_lru(self) -> None:
        if not self._store:
            return
        key, _ = self._store.popitem(last=False)
        self._cleanup_key(key)
        self._stats.evictions += 1

    def _remove(self, key: CacheKey) -> None:
        entry = self._store.pop(key, None)
        if entry is not None:
            self._stats.total_bytes_stored -= entry.size_bytes
        self._cleanup_key(key)

    def _cleanup_key(self, key: CacheKey) -> None:
        """Remove *key* from dependency / subtree indices."""
        dep_vars = set(key.sender | key.receiver)
        for v in dep_vars:
            s = self._var_dependencies.get(v)
            if s:
                s.discard(key)
        for group in self._subtree_groups.values():
            group.discard(key)

    # ------------------------------------------------------------------ #
    #  Statistics & info
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def __len__(self) -> int:
        return len(self._store)

    @property
    def memory_usage(self) -> int:
        return self._stats.total_bytes_stored

    def reset_stats(self) -> None:
        self._stats = CacheStats()

    def entries_for_variable(self, variable: str) -> int:
        """Number of cache entries that depend on *variable*."""
        return len(self._var_dependencies.get(variable, set()))

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #

    def serialize(self) -> bytes:
        """Serialize the entire cache to bytes (pickle)."""
        payload = {
            "capacity": self.capacity,
            "max_memory_bytes": self.max_memory_bytes,
            "entries": [
                {
                    "key": (
                        list(e.key.sender),
                        list(e.key.receiver),
                        e.key.evidence_sig,
                        e.key.intervention_sig,
                    ),
                    "value": e.value,
                    "variables": e.variables,
                    "created_at": e.created_at,
                    "access_count": e.access_count,
                }
                for e in self._store.values()
            ],
            "stats": self._stats.summary(),
        }
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def deserialize(cls, data: bytes) -> "InferenceCache":
        """Reconstruct a cache from serialised bytes."""
        payload = pickle.loads(data)
        cache = cls(
            capacity=payload["capacity"],
            max_memory_bytes=payload["max_memory_bytes"],
        )
        for rec in payload["entries"]:
            sender, receiver, ev_sig, iv_sig = rec["key"]
            key = CacheKey(
                sender=frozenset(sender),
                receiver=frozenset(receiver),
                evidence_sig=ev_sig,
                intervention_sig=iv_sig,
            )
            entry = CacheEntry(
                key=key,
                value=rec["value"],
                variables=rec["variables"],
                created_at=rec["created_at"],
                access_count=rec["access_count"],
            )
            cache._store[key] = entry
            cache._stats.total_bytes_stored += entry.size_bytes
        cache._stats.peak_bytes = cache._stats.total_bytes_stored
        return cache

    def __repr__(self) -> str:
        return (
            f"InferenceCache(size={len(self)}, "
            f"capacity={self.capacity}, "
            f"hit_rate={self._stats.hit_rate:.2%})"
        )
