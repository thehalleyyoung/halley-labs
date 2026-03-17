"""
taintflow.utils.bitmap – Pure-Python roaring bitmap for row provenance tracking.

Implements a compressed bitmap data structure inspired by Roaring Bitmaps.
The bitmap is partitioned into 16-bit high chunks (containers), each holding
values in the range [0, 65535].  Three container types are supported:

* **ArrayContainer** – sorted list of uint16 values (sparse, ≤4096 elements).
* **BitmapContainer** – 65536-bit dense bitmap stored as a bytearray (dense).
* **RunContainer** – RLE-encoded list of (start, length) pairs.

``ProvenanceBitmap`` extends ``RoaringBitmap`` with helpers for ML pipeline
provenance tracking such as test/train fraction computation and propagation
through merges, filters, and group-by operations.
"""

from __future__ import annotations

import bisect
import math
import struct
from abc import ABC, abstractmethod
from typing import (
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

__all__ = [
    "ArrayContainer",
    "BitmapContainer",
    "RunContainer",
    "RoaringBitmap",
    "ProvenanceBitmap",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTAINER_MAX = 65536  # 2^16 values per container
ARRAY_MAX = 4096  # threshold to switch from Array → Bitmap
BITMAP_BYTES = 8192  # 65536 bits = 8192 bytes

# ---------------------------------------------------------------------------
# Abstract base class for containers
# ---------------------------------------------------------------------------


class _Container(ABC):
    """Abstract base for a 16-bit container."""

    __slots__ = ()

    @abstractmethod
    def add(self, val: int) -> "_Container":
        """Add *val* (uint16) and return self or an upgraded container."""

    @abstractmethod
    def remove(self, val: int) -> "_Container":
        """Remove *val*; may return a downgraded container."""

    @abstractmethod
    def __contains__(self, val: int) -> bool: ...

    @abstractmethod
    def __len__(self) -> int:
        """Return cardinality."""

    @abstractmethod
    def __iter__(self) -> Iterator[int]: ...

    @abstractmethod
    def union(self, other: "_Container") -> "_Container": ...

    @abstractmethod
    def intersection(self, other: "_Container") -> "_Container": ...

    @abstractmethod
    def difference(self, other: "_Container") -> "_Container": ...

    @abstractmethod
    def symmetric_difference(self, other: "_Container") -> "_Container": ...

    @abstractmethod
    def and_cardinality(self, other: "_Container") -> int: ...

    @abstractmethod
    def or_cardinality(self, other: "_Container") -> int: ...

    @abstractmethod
    def to_bytes(self) -> bytes: ...

    @abstractmethod
    def min_val(self) -> int: ...

    @abstractmethod
    def max_val(self) -> int: ...

    @abstractmethod
    def rank(self, val: int) -> int:
        """Number of elements ≤ *val*."""

    @abstractmethod
    def select(self, idx: int) -> int:
        """Return the *idx*-th smallest element (0-based)."""

    @abstractmethod
    def flip_range(self, start: int, end: int) -> "_Container":
        """Flip all bits in [start, end)."""

    @abstractmethod
    def clone(self) -> "_Container": ...

    def is_empty(self) -> bool:
        return len(self) == 0


# ---------------------------------------------------------------------------
# ArrayContainer
# ---------------------------------------------------------------------------


class ArrayContainer(_Container):
    """Sorted uint16 array container, efficient for cardinality ≤ 4096."""

    __slots__ = ("_data",)

    def __init__(self, data: Optional[List[int]] = None) -> None:
        self._data: List[int] = sorted(data) if data else []

    # -- mutation -----------------------------------------------------------

    def add(self, val: int) -> _Container:
        idx = bisect.bisect_left(self._data, val)
        if idx < len(self._data) and self._data[idx] == val:
            return self
        self._data.insert(idx, val)
        if len(self._data) > ARRAY_MAX:
            return self._to_bitmap()
        return self

    def remove(self, val: int) -> _Container:
        idx = bisect.bisect_left(self._data, val)
        if idx < len(self._data) and self._data[idx] == val:
            self._data.pop(idx)
        return self

    # -- query --------------------------------------------------------------

    def __contains__(self, val: int) -> bool:  # type: ignore[override]
        idx = bisect.bisect_left(self._data, val)
        return idx < len(self._data) and self._data[idx] == val

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[int]:
        return iter(self._data)

    # -- set operations -----------------------------------------------------

    def union(self, other: _Container) -> _Container:
        if isinstance(other, ArrayContainer):
            merged: List[int] = []
            i = j = 0
            a, b = self._data, other._data
            while i < len(a) and j < len(b):
                if a[i] < b[j]:
                    merged.append(a[i]); i += 1
                elif a[i] > b[j]:
                    merged.append(b[j]); j += 1
                else:
                    merged.append(a[i]); i += 1; j += 1
            merged.extend(a[i:])
            merged.extend(b[j:])
            if len(merged) > ARRAY_MAX:
                c = BitmapContainer()
                for v in merged:
                    c._set_bit(v)
                c._card = len(merged)
                return c
            return ArrayContainer(merged)
        if isinstance(other, BitmapContainer):
            return other.union(self)
        if isinstance(other, RunContainer):
            return self._to_bitmap().union(other._to_bitmap())
        raise TypeError(f"unsupported container type: {type(other)}")

    def intersection(self, other: _Container) -> _Container:
        if isinstance(other, ArrayContainer):
            result: List[int] = []
            i = j = 0
            a, b = self._data, other._data
            while i < len(a) and j < len(b):
                if a[i] < b[j]:
                    i += 1
                elif a[i] > b[j]:
                    j += 1
                else:
                    result.append(a[i]); i += 1; j += 1
            return ArrayContainer(result)
        if isinstance(other, BitmapContainer):
            return ArrayContainer([v for v in self._data if v in other])
        if isinstance(other, RunContainer):
            return ArrayContainer([v for v in self._data if v in other])
        raise TypeError

    def difference(self, other: _Container) -> _Container:
        if isinstance(other, ArrayContainer):
            oset = set(other._data)
            return ArrayContainer([v for v in self._data if v not in oset])
        if isinstance(other, (BitmapContainer, RunContainer)):
            return ArrayContainer([v for v in self._data if v not in other])
        raise TypeError

    def symmetric_difference(self, other: _Container) -> _Container:
        if isinstance(other, ArrayContainer):
            sa, sb = set(self._data), set(other._data)
            result = sorted(sa.symmetric_difference(sb))
            if len(result) > ARRAY_MAX:
                c = BitmapContainer()
                for v in result:
                    c._set_bit(v)
                c._card = len(result)
                return c
            return ArrayContainer(result)
        return self._to_bitmap().symmetric_difference(
            other if isinstance(other, BitmapContainer) else other._to_bitmap()
        )

    # -- cardinality without materializing ----------------------------------

    def and_cardinality(self, other: _Container) -> int:
        if isinstance(other, ArrayContainer):
            count = 0
            i = j = 0
            a, b = self._data, other._data
            while i < len(a) and j < len(b):
                if a[i] < b[j]:
                    i += 1
                elif a[i] > b[j]:
                    j += 1
                else:
                    count += 1; i += 1; j += 1
            return count
        if isinstance(other, BitmapContainer):
            return sum(1 for v in self._data if v in other)
        return sum(1 for v in self._data if v in other)

    def or_cardinality(self, other: _Container) -> int:
        return len(self) + len(other) - self.and_cardinality(other)

    # -- statistics ---------------------------------------------------------

    def min_val(self) -> int:
        if not self._data:
            raise ValueError("empty container")
        return self._data[0]

    def max_val(self) -> int:
        if not self._data:
            raise ValueError("empty container")
        return self._data[-1]

    def rank(self, val: int) -> int:
        return bisect.bisect_right(self._data, val)

    def select(self, idx: int) -> int:
        if idx < 0 or idx >= len(self._data):
            raise IndexError(f"select index {idx} out of range [0, {len(self._data)})")
        return self._data[idx]

    # -- mutation on ranges -------------------------------------------------

    def flip_range(self, start: int, end: int) -> _Container:
        present = set(self._data)
        for v in range(start, end):
            if v in present:
                present.discard(v)
            else:
                present.add(v)
        result = sorted(present)
        if len(result) > ARRAY_MAX:
            c = BitmapContainer()
            for v in result:
                c._set_bit(v)
            c._card = len(result)
            return c
        return ArrayContainer(result)

    # -- serialization ------------------------------------------------------

    def to_bytes(self) -> bytes:
        header = struct.pack("<BH", 0, len(self._data))
        body = struct.pack(f"<{len(self._data)}H", *self._data) if self._data else b""
        return header + body

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Tuple["ArrayContainer", int]:
        _, count = struct.unpack_from("<BH", data, offset)
        offset += 3
        values = list(struct.unpack_from(f"<{count}H", data, offset))
        offset += count * 2
        return cls(values), offset

    # -- conversion ---------------------------------------------------------

    def _to_bitmap(self) -> "BitmapContainer":
        c = BitmapContainer()
        for v in self._data:
            c._set_bit(v)
        c._card = len(self._data)
        return c

    def clone(self) -> "ArrayContainer":
        return ArrayContainer(list(self._data))

    def __repr__(self) -> str:
        return f"ArrayContainer(card={len(self._data)})"


# ---------------------------------------------------------------------------
# BitmapContainer
# ---------------------------------------------------------------------------


class BitmapContainer(_Container):
    """65536-bit dense bitmap container."""

    __slots__ = ("_bitmap", "_card")

    def __init__(self, bitmap: Optional[bytearray] = None, card: int = 0) -> None:
        self._bitmap: bytearray = bitmap if bitmap is not None else bytearray(BITMAP_BYTES)
        self._card: int = card

    # -- internal bit manipulation ------------------------------------------

    def _set_bit(self, val: int) -> None:
        byte_idx, bit_idx = divmod(val, 8)
        self._bitmap[byte_idx] |= 1 << bit_idx

    def _clear_bit(self, val: int) -> None:
        byte_idx, bit_idx = divmod(val, 8)
        self._bitmap[byte_idx] &= ~(1 << bit_idx)

    def _get_bit(self, val: int) -> bool:
        byte_idx, bit_idx = divmod(val, 8)
        return bool(self._bitmap[byte_idx] & (1 << bit_idx))

    # -- mutation -----------------------------------------------------------

    def add(self, val: int) -> _Container:
        if not self._get_bit(val):
            self._set_bit(val)
            self._card += 1
        return self

    def remove(self, val: int) -> _Container:
        if self._get_bit(val):
            self._clear_bit(val)
            self._card -= 1
        if self._card <= ARRAY_MAX:
            return self._to_array()
        return self

    # -- query --------------------------------------------------------------

    def __contains__(self, val: int) -> bool:  # type: ignore[override]
        return self._get_bit(val)

    def __len__(self) -> int:
        return self._card

    def __iter__(self) -> Iterator[int]:
        for byte_idx in range(BITMAP_BYTES):
            b = self._bitmap[byte_idx]
            if b == 0:
                continue
            base = byte_idx * 8
            for bit in range(8):
                if b & (1 << bit):
                    yield base + bit

    # -- set operations -----------------------------------------------------

    def union(self, other: _Container) -> _Container:
        if isinstance(other, ArrayContainer):
            result = BitmapContainer(bytearray(self._bitmap), self._card)
            for v in other._data:
                if not result._get_bit(v):
                    result._set_bit(v)
                    result._card += 1
            return result
        if isinstance(other, BitmapContainer):
            new_bm = bytearray(BITMAP_BYTES)
            card = 0
            for i in range(BITMAP_BYTES):
                new_bm[i] = self._bitmap[i] | other._bitmap[i]
                card += bin(new_bm[i]).count("1")
            return BitmapContainer(new_bm, card)
        if isinstance(other, RunContainer):
            return self.union(other._to_bitmap())
        raise TypeError

    def intersection(self, other: _Container) -> _Container:
        if isinstance(other, ArrayContainer):
            return ArrayContainer([v for v in other._data if self._get_bit(v)])
        if isinstance(other, BitmapContainer):
            new_bm = bytearray(BITMAP_BYTES)
            card = 0
            for i in range(BITMAP_BYTES):
                new_bm[i] = self._bitmap[i] & other._bitmap[i]
                card += bin(new_bm[i]).count("1")
            if card <= ARRAY_MAX:
                c = BitmapContainer(new_bm, card)
                return c._to_array()
            return BitmapContainer(new_bm, card)
        if isinstance(other, RunContainer):
            return self.intersection(other._to_bitmap())
        raise TypeError

    def difference(self, other: _Container) -> _Container:
        if isinstance(other, ArrayContainer):
            result = BitmapContainer(bytearray(self._bitmap), self._card)
            for v in other._data:
                if result._get_bit(v):
                    result._clear_bit(v)
                    result._card -= 1
            if result._card <= ARRAY_MAX:
                return result._to_array()
            return result
        if isinstance(other, BitmapContainer):
            new_bm = bytearray(BITMAP_BYTES)
            card = 0
            for i in range(BITMAP_BYTES):
                new_bm[i] = self._bitmap[i] & (~other._bitmap[i] & 0xFF)
                card += bin(new_bm[i]).count("1")
            if card <= ARRAY_MAX:
                c = BitmapContainer(new_bm, card)
                return c._to_array()
            return BitmapContainer(new_bm, card)
        if isinstance(other, RunContainer):
            return self.difference(other._to_bitmap())
        raise TypeError

    def symmetric_difference(self, other: _Container) -> _Container:
        if isinstance(other, ArrayContainer):
            result = BitmapContainer(bytearray(self._bitmap), self._card)
            for v in other._data:
                if result._get_bit(v):
                    result._clear_bit(v)
                    result._card -= 1
                else:
                    result._set_bit(v)
                    result._card += 1
            if result._card <= ARRAY_MAX:
                return result._to_array()
            return result
        if isinstance(other, BitmapContainer):
            new_bm = bytearray(BITMAP_BYTES)
            card = 0
            for i in range(BITMAP_BYTES):
                new_bm[i] = self._bitmap[i] ^ other._bitmap[i]
                card += bin(new_bm[i]).count("1")
            if card <= ARRAY_MAX:
                c = BitmapContainer(new_bm, card)
                return c._to_array()
            return BitmapContainer(new_bm, card)
        if isinstance(other, RunContainer):
            return self.symmetric_difference(other._to_bitmap())
        raise TypeError

    # -- cardinality without materializing ----------------------------------

    def and_cardinality(self, other: _Container) -> int:
        if isinstance(other, ArrayContainer):
            return sum(1 for v in other._data if self._get_bit(v))
        if isinstance(other, BitmapContainer):
            count = 0
            for i in range(BITMAP_BYTES):
                count += bin(self._bitmap[i] & other._bitmap[i]).count("1")
            return count
        if isinstance(other, RunContainer):
            return self.and_cardinality(other._to_bitmap())
        raise TypeError

    def or_cardinality(self, other: _Container) -> int:
        return len(self) + len(other) - self.and_cardinality(other)

    # -- statistics ---------------------------------------------------------

    def min_val(self) -> int:
        for byte_idx in range(BITMAP_BYTES):
            b = self._bitmap[byte_idx]
            if b:
                for bit in range(8):
                    if b & (1 << bit):
                        return byte_idx * 8 + bit
        raise ValueError("empty container")

    def max_val(self) -> int:
        for byte_idx in range(BITMAP_BYTES - 1, -1, -1):
            b = self._bitmap[byte_idx]
            if b:
                for bit in range(7, -1, -1):
                    if b & (1 << bit):
                        return byte_idx * 8 + bit
        raise ValueError("empty container")

    def rank(self, val: int) -> int:
        count = 0
        full_bytes = val // 8
        for i in range(full_bytes):
            count += bin(self._bitmap[i]).count("1")
        # partial byte
        byte_idx = full_bytes
        if byte_idx < BITMAP_BYTES:
            bit_pos = val % 8
            b = self._bitmap[byte_idx]
            for bit in range(bit_pos + 1):
                if b & (1 << bit):
                    count += 1
        return count

    def select(self, idx: int) -> int:
        count = 0
        for byte_idx in range(BITMAP_BYTES):
            b = self._bitmap[byte_idx]
            if b == 0:
                continue
            base = byte_idx * 8
            for bit in range(8):
                if b & (1 << bit):
                    if count == idx:
                        return base + bit
                    count += 1
        raise IndexError(f"select index {idx} out of range")

    # -- range operations ---------------------------------------------------

    def flip_range(self, start: int, end: int) -> _Container:
        result = BitmapContainer(bytearray(self._bitmap), self._card)
        for v in range(start, end):
            if result._get_bit(v):
                result._clear_bit(v)
                result._card -= 1
            else:
                result._set_bit(v)
                result._card += 1
        if result._card <= ARRAY_MAX:
            return result._to_array()
        return result

    # -- serialization ------------------------------------------------------

    def to_bytes(self) -> bytes:
        header = struct.pack("<BI", 1, self._card)
        return header + bytes(self._bitmap)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Tuple["BitmapContainer", int]:
        _, card = struct.unpack_from("<BI", data, offset)
        offset += 5
        bitmap = bytearray(data[offset : offset + BITMAP_BYTES])
        offset += BITMAP_BYTES
        return cls(bitmap, card), offset

    # -- conversion ---------------------------------------------------------

    def _to_array(self) -> ArrayContainer:
        return ArrayContainer(list(self))

    def clone(self) -> "BitmapContainer":
        return BitmapContainer(bytearray(self._bitmap), self._card)

    def __repr__(self) -> str:
        return f"BitmapContainer(card={self._card})"


# ---------------------------------------------------------------------------
# RunContainer
# ---------------------------------------------------------------------------


class RunContainer(_Container):
    """RLE-encoded container: list of (start, length) pairs.

    Each run ``(s, l)`` represents the integers ``s, s+1, …, s+l``.
    """

    __slots__ = ("_runs",)

    def __init__(self, runs: Optional[List[Tuple[int, int]]] = None) -> None:
        self._runs: List[Tuple[int, int]] = list(runs) if runs else []
        self._compact()

    # -- internal helpers ---------------------------------------------------

    def _compact(self) -> None:
        """Merge overlapping / adjacent runs."""
        if not self._runs:
            return
        self._runs.sort()
        merged: List[Tuple[int, int]] = [self._runs[0]]
        for start, length in self._runs[1:]:
            prev_start, prev_length = merged[-1]
            prev_end = prev_start + prev_length
            if start <= prev_end + 1:
                new_end = max(prev_end, start + length)
                merged[-1] = (prev_start, new_end - prev_start)
            else:
                merged.append((start, length))
        self._runs = merged

    def _find_run(self, val: int) -> int:
        """Return index of run containing *val* or -1."""
        lo, hi = 0, len(self._runs) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            s, l = self._runs[mid]
            if val < s:
                hi = mid - 1
            elif val > s + l:
                lo = mid + 1
            else:
                return mid
        return -1

    # -- mutation -----------------------------------------------------------

    def add(self, val: int) -> _Container:
        if val in self:
            return self
        self._runs.append((val, 0))
        self._compact()
        return self

    def remove(self, val: int) -> _Container:
        idx = self._find_run(val)
        if idx == -1:
            return self
        s, l = self._runs[idx]
        if s == val and l == 0:
            self._runs.pop(idx)
        elif s == val:
            self._runs[idx] = (s + 1, l - 1)
        elif s + l == val:
            self._runs[idx] = (s, l - 1)
        else:
            # split
            self._runs[idx] = (s, val - s - 1)
            self._runs.insert(idx + 1, (val + 1, s + l - val - 1))
        return self

    # -- query --------------------------------------------------------------

    def __contains__(self, val: int) -> bool:  # type: ignore[override]
        return self._find_run(val) != -1

    def __len__(self) -> int:
        return sum(l + 1 for _, l in self._runs)

    def __iter__(self) -> Iterator[int]:
        for s, l in self._runs:
            yield from range(s, s + l + 1)

    # -- set operations -----------------------------------------------------

    def union(self, other: _Container) -> _Container:
        return self._to_bitmap().union(
            other if isinstance(other, BitmapContainer) else other._to_bitmap()
            if isinstance(other, (ArrayContainer, RunContainer))
            else other
        )

    def intersection(self, other: _Container) -> _Container:
        return self._to_bitmap().intersection(
            other if isinstance(other, BitmapContainer) else other._to_bitmap()
            if isinstance(other, (ArrayContainer, RunContainer))
            else other
        )

    def difference(self, other: _Container) -> _Container:
        return self._to_bitmap().difference(
            other if isinstance(other, BitmapContainer) else other._to_bitmap()
            if isinstance(other, (ArrayContainer, RunContainer))
            else other
        )

    def symmetric_difference(self, other: _Container) -> _Container:
        return self._to_bitmap().symmetric_difference(
            other if isinstance(other, BitmapContainer) else other._to_bitmap()
            if isinstance(other, (ArrayContainer, RunContainer))
            else other
        )

    # -- cardinality --------------------------------------------------------

    def and_cardinality(self, other: _Container) -> int:
        return self._to_bitmap().and_cardinality(
            other if isinstance(other, BitmapContainer) else other._to_bitmap()
            if isinstance(other, (ArrayContainer, RunContainer))
            else other
        )

    def or_cardinality(self, other: _Container) -> int:
        return len(self) + len(other) - self.and_cardinality(other)

    # -- statistics ---------------------------------------------------------

    def min_val(self) -> int:
        if not self._runs:
            raise ValueError("empty container")
        return self._runs[0][0]

    def max_val(self) -> int:
        if not self._runs:
            raise ValueError("empty container")
        s, l = self._runs[-1]
        return s + l

    def rank(self, val: int) -> int:
        count = 0
        for s, l in self._runs:
            end = s + l
            if val < s:
                break
            if val >= end:
                count += l + 1
            else:
                count += val - s + 1
                break
        return count

    def select(self, idx: int) -> int:
        count = 0
        for s, l in self._runs:
            run_len = l + 1
            if count + run_len > idx:
                return s + (idx - count)
            count += run_len
        raise IndexError(f"select index {idx} out of range")

    # -- range operations ---------------------------------------------------

    def flip_range(self, start: int, end: int) -> _Container:
        return self._to_bitmap().flip_range(start, end)

    # -- serialization ------------------------------------------------------

    def to_bytes(self) -> bytes:
        header = struct.pack("<BH", 2, len(self._runs))
        body = b""
        for s, l in self._runs:
            body += struct.pack("<HH", s, l)
        return header + body

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Tuple["RunContainer", int]:
        _, n_runs = struct.unpack_from("<BH", data, offset)
        offset += 3
        runs: List[Tuple[int, int]] = []
        for _ in range(n_runs):
            s, l = struct.unpack_from("<HH", data, offset)
            runs.append((s, l))
            offset += 4
        return cls(runs), offset

    # -- conversion ---------------------------------------------------------

    def _to_bitmap(self) -> BitmapContainer:
        c = BitmapContainer()
        card = 0
        for s, l in self._runs:
            for v in range(s, s + l + 1):
                c._set_bit(v)
                card += 1
        c._card = card
        return c

    def _to_array(self) -> ArrayContainer:
        return ArrayContainer(list(self))

    def clone(self) -> "RunContainer":
        return RunContainer(list(self._runs))

    def __repr__(self) -> str:
        return f"RunContainer(runs={len(self._runs)}, card={len(self)})"


# ---------------------------------------------------------------------------
# Container conversion helper
# ---------------------------------------------------------------------------

def _optimal_container(container: _Container) -> _Container:
    """Return the most space-efficient representation of *container*."""
    card = len(container)
    if card == 0:
        return ArrayContainer()

    # Estimate run container size
    if isinstance(container, RunContainer):
        run_size = len(container._runs) * 4
    else:
        vals = sorted(container)
        runs: List[Tuple[int, int]] = []
        start = vals[0]
        length = 0
        for i in range(1, len(vals)):
            if vals[i] == vals[i - 1] + 1:
                length += 1
            else:
                runs.append((start, length))
                start = vals[i]
                length = 0
        runs.append((start, length))
        run_size = len(runs) * 4

    array_size = card * 2
    bitmap_size = BITMAP_BYTES

    best_size = min(array_size, bitmap_size, run_size)

    if best_size == run_size and (isinstance(container, RunContainer) or runs):
        if isinstance(container, RunContainer):
            return container
        return RunContainer(runs)
    if best_size == array_size and card <= ARRAY_MAX:
        if isinstance(container, ArrayContainer):
            return container
        return ArrayContainer(sorted(container))
    if isinstance(container, BitmapContainer):
        return container
    # convert to bitmap
    bm = BitmapContainer()
    for v in container:
        bm._set_bit(v)
    bm._card = card
    return bm


# ---------------------------------------------------------------------------
# RoaringBitmap
# ---------------------------------------------------------------------------


class RoaringBitmap:
    """Pure-Python roaring bitmap.

    Values are unsigned 32-bit integers.  Internally partitioned into
    containers keyed by the 16 high bits.
    """

    __slots__ = ("_containers",)

    def __init__(self) -> None:
        self._containers: dict[int, _Container] = {}

    @staticmethod
    def _split(val: int) -> Tuple[int, int]:
        """Split a 32-bit value into (high16, low16)."""
        return val >> 16, val & 0xFFFF

    @staticmethod
    def _combine(high: int, low: int) -> int:
        return (high << 16) | low

    # -- mutation -----------------------------------------------------------

    def add(self, val: int) -> None:
        """Add an integer *val* ∈ [0, 2³²)."""
        hi, lo = self._split(val)
        if hi not in self._containers:
            self._containers[hi] = ArrayContainer()
        self._containers[hi] = self._containers[hi].add(lo)

    def remove(self, val: int) -> None:
        """Remove *val*; no-op if absent."""
        hi, lo = self._split(val)
        if hi not in self._containers:
            return
        self._containers[hi] = self._containers[hi].remove(lo)
        if self._containers[hi].is_empty():
            del self._containers[hi]

    def discard(self, val: int) -> None:
        """Alias for :meth:`remove`."""
        self.remove(val)

    def add_range(self, start: int, end: int) -> None:
        """Add all integers in ``[start, end)``."""
        if start >= end:
            return
        start_hi, start_lo = self._split(start)
        end_hi, end_lo = self._split(end - 1)
        for hi in range(start_hi, end_hi + 1):
            lo_start = start_lo if hi == start_hi else 0
            lo_end = end_lo + 1 if hi == end_hi else CONTAINER_MAX
            if hi not in self._containers:
                self._containers[hi] = ArrayContainer()
            # batch add
            c = self._containers[hi]
            if lo_end - lo_start > ARRAY_MAX and not isinstance(c, BitmapContainer):
                c = c._to_bitmap() if isinstance(c, ArrayContainer) else c._to_bitmap()
            for v in range(lo_start, lo_end):
                c = c.add(v)
            self._containers[hi] = c

    def remove_range(self, start: int, end: int) -> None:
        """Remove all integers in ``[start, end)``."""
        if start >= end:
            return
        start_hi, start_lo = self._split(start)
        end_hi, end_lo = self._split(end - 1)
        for hi in range(start_hi, end_hi + 1):
            if hi not in self._containers:
                continue
            lo_start = start_lo if hi == start_hi else 0
            lo_end = end_lo + 1 if hi == end_hi else CONTAINER_MAX
            c = self._containers[hi]
            for v in range(lo_start, lo_end):
                c = c.remove(v)
            if c.is_empty():
                del self._containers[hi]
            else:
                self._containers[hi] = c

    def flip_range(self, start: int, end: int) -> "RoaringBitmap":
        """Return a new bitmap with bits in ``[start, end)`` flipped."""
        result = self.clone()
        if start >= end:
            return result
        start_hi, start_lo = self._split(start)
        end_hi, end_lo = self._split(end - 1)
        for hi in range(start_hi, end_hi + 1):
            lo_start = start_lo if hi == start_hi else 0
            lo_end = end_lo + 1 if hi == end_hi else CONTAINER_MAX
            if hi not in result._containers:
                result._containers[hi] = ArrayContainer()
            result._containers[hi] = result._containers[hi].flip_range(lo_start, lo_end)
            if result._containers[hi].is_empty():
                del result._containers[hi]
        return result

    # -- query --------------------------------------------------------------

    def __contains__(self, val: int) -> bool:
        hi, lo = self._split(val)
        if hi not in self._containers:
            return False
        return lo in self._containers[hi]

    def __len__(self) -> int:
        return sum(len(c) for c in self._containers.values())

    def __bool__(self) -> bool:
        return bool(self._containers)

    def is_empty(self) -> bool:
        return not self._containers

    def __iter__(self) -> Iterator[int]:
        for hi in sorted(self._containers):
            for lo in self._containers[hi]:
                yield self._combine(hi, lo)

    def cardinality(self) -> int:
        """Return the number of set bits."""
        return len(self)

    # -- set operations returning new bitmap --------------------------------

    def union(self, other: "RoaringBitmap") -> "RoaringBitmap":
        """Return ``self | other``."""
        result = RoaringBitmap()
        all_keys = set(self._containers) | set(other._containers)
        for hi in all_keys:
            a = self._containers.get(hi)
            b = other._containers.get(hi)
            if a is None:
                result._containers[hi] = b.clone()  # type: ignore[union-attr]
            elif b is None:
                result._containers[hi] = a.clone()
            else:
                result._containers[hi] = a.union(b)
        return result

    def __or__(self, other: "RoaringBitmap") -> "RoaringBitmap":
        return self.union(other)

    def intersection(self, other: "RoaringBitmap") -> "RoaringBitmap":
        """Return ``self & other``."""
        result = RoaringBitmap()
        common = set(self._containers) & set(other._containers)
        for hi in common:
            c = self._containers[hi].intersection(other._containers[hi])
            if not c.is_empty():
                result._containers[hi] = c
        return result

    def __and__(self, other: "RoaringBitmap") -> "RoaringBitmap":
        return self.intersection(other)

    def difference(self, other: "RoaringBitmap") -> "RoaringBitmap":
        """Return ``self - other``."""
        result = RoaringBitmap()
        for hi, c in self._containers.items():
            if hi in other._containers:
                diff = c.difference(other._containers[hi])
                if not diff.is_empty():
                    result._containers[hi] = diff
            else:
                result._containers[hi] = c.clone()
        return result

    def __sub__(self, other: "RoaringBitmap") -> "RoaringBitmap":
        return self.difference(other)

    def symmetric_difference(self, other: "RoaringBitmap") -> "RoaringBitmap":
        """Return ``self ^ other``."""
        result = RoaringBitmap()
        all_keys = set(self._containers) | set(other._containers)
        for hi in all_keys:
            a = self._containers.get(hi)
            b = other._containers.get(hi)
            if a is None:
                result._containers[hi] = b.clone()  # type: ignore[union-attr]
            elif b is None:
                result._containers[hi] = a.clone()
            else:
                c = a.symmetric_difference(b)
                if not c.is_empty():
                    result._containers[hi] = c
        return result

    def __xor__(self, other: "RoaringBitmap") -> "RoaringBitmap":
        return self.symmetric_difference(other)

    # -- cardinality without materializing ----------------------------------

    def and_cardinality(self, other: "RoaringBitmap") -> int:
        """Compute ``|self & other|`` without building the intersection."""
        count = 0
        common = set(self._containers) & set(other._containers)
        for hi in common:
            count += self._containers[hi].and_cardinality(other._containers[hi])
        return count

    def or_cardinality(self, other: "RoaringBitmap") -> int:
        """Compute ``|self | other|`` without building the union."""
        return len(self) + len(other) - self.and_cardinality(other)

    # -- statistics ---------------------------------------------------------

    def minimum(self) -> int:
        """Return the smallest element."""
        if not self._containers:
            raise ValueError("empty bitmap")
        hi = min(self._containers)
        lo = self._containers[hi].min_val()
        return self._combine(hi, lo)

    def maximum(self) -> int:
        """Return the largest element."""
        if not self._containers:
            raise ValueError("empty bitmap")
        hi = max(self._containers)
        lo = self._containers[hi].max_val()
        return self._combine(hi, lo)

    def rank(self, val: int) -> int:
        """Return number of elements ≤ *val*."""
        hi, lo = self._split(val)
        count = 0
        for k in sorted(self._containers):
            if k < hi:
                count += len(self._containers[k])
            elif k == hi:
                count += self._containers[k].rank(lo)
                break
            else:
                break
        return count

    def select(self, idx: int) -> int:
        """Return the *idx*-th element (0-based)."""
        count = 0
        for hi in sorted(self._containers):
            c = self._containers[hi]
            c_len = len(c)
            if count + c_len > idx:
                lo = c.select(idx - count)
                return self._combine(hi, lo)
            count += c_len
        raise IndexError(f"select index {idx} out of range [0, {len(self)})")

    # -- serialization ------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Serialize to a compact binary format."""
        keys = sorted(self._containers)
        header = struct.pack("<I", len(keys))
        parts = [header]
        for hi in keys:
            parts.append(struct.pack("<H", hi))
            parts.append(self._containers[hi].to_bytes())
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "RoaringBitmap":
        """Deserialize from bytes produced by :meth:`to_bytes`."""
        bm = cls()
        offset = 0
        (n_containers,) = struct.unpack_from("<I", data, offset)
        offset += 4
        for _ in range(n_containers):
            (hi,) = struct.unpack_from("<H", data, offset)
            offset += 2
            ctype = data[offset]
            if ctype == 0:
                container, offset = ArrayContainer.from_bytes(data, offset)
            elif ctype == 1:
                container, offset = BitmapContainer.from_bytes(data, offset)
            elif ctype == 2:
                container, offset = RunContainer.from_bytes(data, offset)
            else:
                raise ValueError(f"unknown container type byte: {ctype}")
            if not container.is_empty():
                bm._containers[hi] = container
        return bm

    # -- misc ---------------------------------------------------------------

    def clone(self) -> "RoaringBitmap":
        """Return a deep copy."""
        bm = RoaringBitmap()
        for hi, c in self._containers.items():
            bm._containers[hi] = c.clone()
        return bm

    @classmethod
    def from_iterable(cls, values: Iterable[int]) -> "RoaringBitmap":
        """Build a bitmap from an iterable of unsigned integers."""
        bm = cls()
        for v in values:
            bm.add(v)
        return bm

    @classmethod
    def from_range(cls, start: int, end: int) -> "RoaringBitmap":
        """Build a bitmap containing ``[start, end)``."""
        bm = cls()
        bm.add_range(start, end)
        return bm

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RoaringBitmap):
            return NotImplemented
        if set(self._containers) != set(other._containers):
            return False
        for hi in self._containers:
            if set(self._containers[hi]) != set(other._containers[hi]):
                return False
        return True

    def __repr__(self) -> str:
        return f"RoaringBitmap(card={len(self)}, containers={len(self._containers)})"


# ---------------------------------------------------------------------------
# ProvenanceBitmap – domain-specific extension for ML provenance
# ---------------------------------------------------------------------------


class ProvenanceBitmap(RoaringBitmap):
    """Roaring bitmap extended with ML pipeline provenance semantics.

    Tracks which row indices are test rows and provides fraction-based
    queries as well as propagation through common DataFrame operations.
    """

    __slots__ = ("_total_rows",)

    def __init__(self, total_rows: int = 0) -> None:
        super().__init__()
        self._total_rows: int = total_rows

    # -- properties ---------------------------------------------------------

    @property
    def test_fraction(self) -> float:
        """Fraction of rows in this bitmap that came from the test set.

        The bitmap stores the *test* row indices.  So ``test_fraction``
        is ``|bitmap| / total_rows``.
        """
        if self._total_rows == 0:
            return 0.0
        return len(self) / self._total_rows

    @property
    def train_fraction(self) -> float:
        """Fraction of rows *not* in the test set."""
        return 1.0 - self.test_fraction

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @total_rows.setter
    def total_rows(self, value: int) -> None:
        self._total_rows = value

    # -- factory methods ----------------------------------------------------

    @classmethod
    def from_split(
        cls,
        n_total: int,
        test_indices: Iterable[int],
    ) -> "ProvenanceBitmap":
        """Create a provenance bitmap from a train/test split.

        Parameters
        ----------
        n_total:
            Total number of rows in the dataset.
        test_indices:
            Row indices belonging to the test set.
        """
        bm = cls(total_rows=n_total)
        for idx in test_indices:
            if 0 <= idx < n_total:
                bm.add(idx)
        return bm

    @classmethod
    def from_bitmap(cls, base: RoaringBitmap, total_rows: int) -> "ProvenanceBitmap":
        """Wrap an existing :class:`RoaringBitmap` as a provenance bitmap."""
        pb = cls(total_rows=total_rows)
        pb._containers = {hi: c.clone() for hi, c in base._containers.items()}
        return pb

    # -- propagation through operations -------------------------------------

    def propagate_through_merge(
        self,
        other: "ProvenanceBitmap",
        how: str = "inner",
        left_on_indices: Optional[Sequence[int]] = None,
        right_on_indices: Optional[Sequence[int]] = None,
    ) -> "ProvenanceBitmap":
        """Propagate provenance through a merge / join operation.

        Parameters
        ----------
        other:
            Provenance bitmap of the right-hand dataframe.
        how:
            Join type: ``"inner"``, ``"left"``, ``"right"``, ``"outer"``.
        left_on_indices:
            Row indices from self that survive after the join.
        right_on_indices:
            Row indices from other that survive after the join.

        Returns
        -------
        A new :class:`ProvenanceBitmap` for the merged result.
        """
        if left_on_indices is None:
            left_bm = self.clone()
        else:
            left_bm = RoaringBitmap.from_iterable(
                i for i in left_on_indices if i in self
            )
        if right_on_indices is None:
            right_bm = other.clone()
        else:
            right_bm = RoaringBitmap.from_iterable(
                i for i in right_on_indices if i in other
            )

        if how == "inner":
            new_total = (
                (len(left_on_indices) if left_on_indices is not None else self._total_rows)
            )
            test_count = len(left_bm) + len(right_bm)
        elif how == "left":
            new_total = self._total_rows
            test_count = len(left_bm) + len(right_bm)
        elif how == "right":
            new_total = other._total_rows
            test_count = len(left_bm) + len(right_bm)
        else:  # outer
            new_total = self._total_rows + other._total_rows
            test_count = len(self) + len(other)

        # Build a combined provenance bitmap keyed by output row
        result = ProvenanceBitmap(total_rows=max(new_total, 1))
        # Mark test rows: the output row is tainted if *either* input was test
        combined = left_bm | right_bm
        result._containers = combined._containers
        result._total_rows = max(new_total, 1)
        return result

    def propagate_through_filter(
        self,
        kept_indices: Iterable[int],
    ) -> "ProvenanceBitmap":
        """Propagate provenance through a row filter / boolean indexing.

        Parameters
        ----------
        kept_indices:
            Row indices that survive the filter.

        Returns
        -------
        New :class:`ProvenanceBitmap` with reindexed rows.
        """
        kept = list(kept_indices)
        result = ProvenanceBitmap(total_rows=len(kept))
        for new_idx, old_idx in enumerate(kept):
            if old_idx in self:
                result.add(new_idx)
        return result

    def propagate_through_groupby(
        self,
        group_assignments: Sequence[Sequence[int]],
    ) -> "ProvenanceBitmap":
        """Propagate provenance through a groupby-aggregate.

        Each output row is the aggregate of a group of input rows.  If *any*
        input row in a group is a test row, the output row is marked test.

        Parameters
        ----------
        group_assignments:
            ``group_assignments[i]`` is a sequence of input row indices
            belonging to output group ``i``.

        Returns
        -------
        New :class:`ProvenanceBitmap` with one bit per output group.
        """
        result = ProvenanceBitmap(total_rows=len(group_assignments))
        for out_idx, group in enumerate(group_assignments):
            for in_idx in group:
                if in_idx in self:
                    result.add(out_idx)
                    break
        return result

    def propagate_through_sample(
        self,
        sampled_indices: Sequence[int],
    ) -> "ProvenanceBitmap":
        """Propagate provenance through row sampling (e.g. bootstrap)."""
        result = ProvenanceBitmap(total_rows=len(sampled_indices))
        for new_idx, old_idx in enumerate(sampled_indices):
            if old_idx in self:
                result.add(new_idx)
        return result

    def propagate_through_sort(
        self,
        sort_order: Sequence[int],
    ) -> "ProvenanceBitmap":
        """Propagate provenance through a row reordering."""
        result = ProvenanceBitmap(total_rows=len(sort_order))
        for new_idx, old_idx in enumerate(sort_order):
            if old_idx in self:
                result.add(new_idx)
        return result

    def propagate_through_concat(
        self,
        others: Sequence["ProvenanceBitmap"],
    ) -> "ProvenanceBitmap":
        """Propagate provenance through vertical concatenation."""
        total = self._total_rows + sum(o._total_rows for o in others)
        result = ProvenanceBitmap(total_rows=total)
        offset = 0
        for bm in [self, *others]:
            for val in bm:
                result.add(offset + val)
            offset += bm._total_rows
        return result

    # -- serialization override ---------------------------------------------

    def to_bytes(self) -> bytes:
        header = struct.pack("<I", self._total_rows)
        return header + super().to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ProvenanceBitmap":  # type: ignore[override]
        (total_rows,) = struct.unpack_from("<I", data, 0)
        base = RoaringBitmap.from_bytes(data[4:])
        return cls.from_bitmap(base, total_rows)

    # -- clone override -----------------------------------------------------

    def clone(self) -> "ProvenanceBitmap":  # type: ignore[override]
        return ProvenanceBitmap.from_bitmap(
            RoaringBitmap.clone(self), self._total_rows
        )

    def __repr__(self) -> str:
        return (
            f"ProvenanceBitmap(card={len(self)}, total={self._total_rows}, "
            f"test_frac={self.test_fraction:.4f})"
        )
