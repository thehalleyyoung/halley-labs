"""
High-performance state hash table for explicit-state model checking.

Implements Robin Hood hashing with Zobrist state hashing and
full-state comparison for collision resolution.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

# Default parameters
_INITIAL_CAPACITY = 1024
_LOAD_FACTOR_THRESHOLD = 0.75
_GROWTH_FACTOR = 2
_SHRINK_THRESHOLD = 0.25
_MIN_CAPACITY = 64


# ---------------------------------------------------------------------------
# Hash entry
# ---------------------------------------------------------------------------

@dataclass
class _HashEntry:
    """A slot in the Robin Hood hash table."""

    key_hash: int
    state: Dict[str, Any]
    fingerprint: str  # short fingerprint for fast reject
    probe_distance: int = 0


# ---------------------------------------------------------------------------
# Zobrist hasher
# ---------------------------------------------------------------------------

class ZobristHasher:
    """
    Zobrist hashing for TLA+ states.

    Each (variable, value) pair is assigned a random bit-string.
    The hash of a state is the XOR of the bit-strings for all its
    variable assignments.  This gives O(1) incremental update.
    """

    def __init__(self, bit_width: int = 64, seed: Optional[int] = None) -> None:
        self._bit_width = bit_width
        self._rng = random.Random(seed)
        self._table: Dict[Tuple[str, str], int] = {}
        self._mask = (1 << bit_width) - 1

    def _get_random_bits(self) -> int:
        return self._rng.getrandbits(self._bit_width)

    def _key_for_pair(self, variable: str, value: Any) -> Tuple[str, str]:
        return (variable, json.dumps(value, sort_keys=True, default=str))

    def _lookup_or_create(self, variable: str, value: Any) -> int:
        key = self._key_for_pair(variable, value)
        if key not in self._table:
            self._table[key] = self._get_random_bits()
        return self._table[key]

    def hash_state(self, state: Dict[str, Any]) -> int:
        """Compute the Zobrist hash of a full state."""
        h = 0
        for var in sorted(state.keys()):
            val = state[var]
            h ^= self._lookup_or_create(var, val)
        return h & self._mask

    def incremental_update(
        self,
        old_hash: int,
        variable: str,
        old_value: Any,
        new_value: Any,
    ) -> int:
        """Update hash when a single variable changes."""
        h = old_hash
        h ^= self._lookup_or_create(variable, old_value)
        h ^= self._lookup_or_create(variable, new_value)
        return h & self._mask

    def hash_partial(self, partial_state: Dict[str, Any], base_hash: int = 0) -> int:
        h = base_hash
        for var in sorted(partial_state.keys()):
            h ^= self._lookup_or_create(var, partial_state[var])
        return h & self._mask

    @property
    def table_size(self) -> int:
        return len(self._table)

    def memory_bytes(self) -> int:
        return sys.getsizeof(self._table) + sum(
            sys.getsizeof(k) + sys.getsizeof(v) for k, v in self._table.items()
        )


# ---------------------------------------------------------------------------
# State fingerprint
# ---------------------------------------------------------------------------

def compute_fingerprint(state: Dict[str, Any], length: int = 16) -> str:
    """Short fingerprint for fast negative comparison."""
    canonical = json.dumps(state, sort_keys=True, default=str)
    digest = hashlib.md5(canonical.encode()).hexdigest()
    return digest[:length]


def compute_full_hash(state: Dict[str, Any]) -> str:
    """Full SHA-256 hash for definitive comparison."""
    canonical = json.dumps(state, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Robin Hood hash table
# ---------------------------------------------------------------------------

class StateHashTable:
    """
    Robin Hood open-addressing hash table optimized for TLA+ states.

    Robin Hood hashing minimizes probe-distance variance, leading to
    more cache-friendly lookups and bounded worst-case search.

    Parameters
    ----------
    initial_capacity : int
        Starting number of slots.
    load_factor : float
        Maximum load factor before resizing.
    hasher : ZobristHasher, optional
        Custom hasher; a default one is created otherwise.
    """

    def __init__(
        self,
        initial_capacity: int = _INITIAL_CAPACITY,
        load_factor: float = _LOAD_FACTOR_THRESHOLD,
        hasher: Optional[ZobristHasher] = None,
    ) -> None:
        self._capacity = max(_MIN_CAPACITY, initial_capacity)
        self._load_factor = load_factor
        self._hasher = hasher or ZobristHasher()
        self._slots: List[Optional[_HashEntry]] = [None] * self._capacity
        self._size = 0
        self._num_probes = 0
        self._num_resizes = 0
        self._max_probe_distance = 0

    # -- Properties --------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def load(self) -> float:
        return self._size / self._capacity if self._capacity else 0.0

    @property
    def max_probe_distance(self) -> int:
        return self._max_probe_distance

    # -- Internal helpers --------------------------------------------------

    def _slot_index(self, key_hash: int) -> int:
        return key_hash % self._capacity

    def _states_equal(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """Full state comparison (handles collision resolution)."""
        return a == b

    # -- Core operations ---------------------------------------------------

    def insert(self, state: Dict[str, Any]) -> bool:
        """Insert a state.  Returns True if new, False if duplicate."""
        if self.load >= self._load_factor:
            self._resize(self._capacity * _GROWTH_FACTOR)

        key_hash = self._hasher.hash_state(state)
        fp = compute_fingerprint(state)
        entry = _HashEntry(key_hash=key_hash, state=state, fingerprint=fp)
        return self._robin_hood_insert(entry)

    def _robin_hood_insert(self, entry: _HashEntry) -> bool:
        """Insert with Robin Hood displacement."""
        idx = self._slot_index(entry.key_hash)
        entry.probe_distance = 0

        while True:
            slot = self._slots[idx]
            self._num_probes += 1

            if slot is None:
                self._slots[idx] = entry
                self._size += 1
                if entry.probe_distance > self._max_probe_distance:
                    self._max_probe_distance = entry.probe_distance
                return True

            if (
                slot.key_hash == entry.key_hash
                and slot.fingerprint == entry.fingerprint
                and self._states_equal(slot.state, entry.state)
            ):
                return False

            if entry.probe_distance > slot.probe_distance:
                self._slots[idx] = entry
                entry = slot

            entry.probe_distance += 1
            idx = (idx + 1) % self._capacity

    def contains(self, state: Dict[str, Any]) -> bool:
        """Check if a state is in the table."""
        key_hash = self._hasher.hash_state(state)
        fp = compute_fingerprint(state)
        idx = self._slot_index(key_hash)
        probe_dist = 0

        while True:
            slot = self._slots[idx]
            if slot is None:
                return False
            if probe_dist > slot.probe_distance:
                return False
            if (
                slot.key_hash == key_hash
                and slot.fingerprint == fp
                and self._states_equal(slot.state, state)
            ):
                return True
            probe_dist += 1
            idx = (idx + 1) % self._capacity

    def get(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve the stored copy of a state (useful after canonicalization)."""
        key_hash = self._hasher.hash_state(state)
        fp = compute_fingerprint(state)
        idx = self._slot_index(key_hash)
        probe_dist = 0

        while True:
            slot = self._slots[idx]
            if slot is None:
                return None
            if probe_dist > slot.probe_distance:
                return None
            if (
                slot.key_hash == key_hash
                and slot.fingerprint == fp
                and self._states_equal(slot.state, state)
            ):
                return slot.state
            probe_dist += 1
            idx = (idx + 1) % self._capacity

    def remove(self, state: Dict[str, Any]) -> bool:
        """Remove a state.  Returns True if found and removed."""
        key_hash = self._hasher.hash_state(state)
        fp = compute_fingerprint(state)
        idx = self._slot_index(key_hash)
        probe_dist = 0

        while True:
            slot = self._slots[idx]
            if slot is None:
                return False
            if probe_dist > slot.probe_distance:
                return False
            if (
                slot.key_hash == key_hash
                and slot.fingerprint == fp
                and self._states_equal(slot.state, state)
            ):
                self._slots[idx] = None
                self._size -= 1
                self._backward_shift(idx)
                return True
            probe_dist += 1
            idx = (idx + 1) % self._capacity

    def _backward_shift(self, empty_idx: int) -> None:
        """Shift entries backward to fill the gap after removal."""
        idx = (empty_idx + 1) % self._capacity
        while True:
            slot = self._slots[idx]
            if slot is None or slot.probe_distance == 0:
                break
            self._slots[(idx - 1) % self._capacity] = slot
            slot.probe_distance -= 1
            self._slots[idx] = None
            idx = (idx + 1) % self._capacity

    # -- Resizing ----------------------------------------------------------

    def _resize(self, new_capacity: int) -> None:
        new_capacity = max(_MIN_CAPACITY, new_capacity)
        old_slots = self._slots
        self._slots = [None] * new_capacity
        self._capacity = new_capacity
        self._size = 0
        self._max_probe_distance = 0
        self._num_resizes += 1

        for slot in old_slots:
            if slot is not None:
                slot.probe_distance = 0
                self._robin_hood_insert(slot)

    def shrink_if_needed(self) -> bool:
        if self.load < _SHRINK_THRESHOLD and self._capacity > _MIN_CAPACITY:
            self._resize(max(_MIN_CAPACITY, self._capacity // _GROWTH_FACTOR))
            return True
        return False

    # -- Bulk operations ---------------------------------------------------

    def insert_many(self, states: Iterable[Dict[str, Any]]) -> int:
        """Insert multiple states.  Return count of new insertions."""
        count = 0
        for state in states:
            if self.insert(state):
                count += 1
        return count

    def contains_any(self, states: Iterable[Dict[str, Any]]) -> bool:
        return any(self.contains(s) for s in states)

    def contains_all(self, states: Iterable[Dict[str, Any]]) -> bool:
        return all(self.contains(s) for s in states)

    # -- Iteration ---------------------------------------------------------

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        for slot in self._slots:
            if slot is not None:
                yield slot.state

    def __len__(self) -> int:
        return self._size

    def __contains__(self, state: Dict[str, Any]) -> bool:
        return self.contains(state)

    def all_states(self) -> List[Dict[str, Any]]:
        return [slot.state for slot in self._slots if slot is not None]

    def all_fingerprints(self) -> List[str]:
        return [slot.fingerprint for slot in self._slots if slot is not None]

    # -- Memory & statistics -----------------------------------------------

    def memory_usage_bytes(self) -> int:
        base = sys.getsizeof(self._slots)
        entry_size = sum(
            sys.getsizeof(s) + sys.getsizeof(s.state) if s else 0
            for s in self._slots
        )
        return base + entry_size + self._hasher.memory_bytes()

    def memory_usage_mb(self) -> float:
        return self.memory_usage_bytes() / (1024 * 1024)

    def statistics(self) -> Dict[str, Any]:
        probe_distances = [
            s.probe_distance for s in self._slots if s is not None
        ]
        return {
            "size": self._size,
            "capacity": self._capacity,
            "load_factor": round(self.load, 4),
            "num_resizes": self._num_resizes,
            "total_probes": self._num_probes,
            "max_probe_distance": self._max_probe_distance,
            "avg_probe_distance": (
                sum(probe_distances) / len(probe_distances)
                if probe_distances
                else 0.0
            ),
            "memory_mb": round(self.memory_usage_mb(), 3),
        }

    def clear(self) -> None:
        self._slots = [None] * self._capacity
        self._size = 0
        self._max_probe_distance = 0

    def __repr__(self) -> str:
        return (
            f"StateHashTable(size={self._size}, capacity={self._capacity}, "
            f"load={self.load:.2%})"
        )
