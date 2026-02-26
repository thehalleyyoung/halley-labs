"""
Memory management for BioProver.

Arena allocator for SMT clause databases, object pools for frequently
allocated/freed objects, reference counting for shared formula nodes,
and memory usage tracking with bulk deallocation for CEGAR iteration cleanup.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Memory tracker
# ---------------------------------------------------------------------------

@dataclass
class AllocationRecord:
    """Record of a single allocation event."""
    tag: str
    size_bytes: int
    timestamp: float = field(default_factory=time.time)
    freed: bool = False


class MemoryTracker:
    """Track memory allocations by tag and report usage.

    This is an application-level tracker (not a replacement for OS-level
    accounting).  Components register allocations/frees and the tracker
    maintains running totals per tag.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Dict[str, int] = defaultdict(int)
        self._peak: Dict[str, int] = defaultdict(int)
        self._total_alloc: Dict[str, int] = defaultdict(int)
        self._total_free: Dict[str, int] = defaultdict(int)
        self._records: List[AllocationRecord] = []
        self._record_history = True

    def alloc(self, tag: str, size_bytes: int) -> None:
        """Record an allocation of *size_bytes* under *tag*."""
        with self._lock:
            self._current[tag] += size_bytes
            self._total_alloc[tag] += size_bytes
            if self._current[tag] > self._peak[tag]:
                self._peak[tag] = self._current[tag]
            if self._record_history:
                self._records.append(AllocationRecord(tag=tag, size_bytes=size_bytes))

    def free(self, tag: str, size_bytes: int) -> None:
        """Record a deallocation."""
        with self._lock:
            self._current[tag] = max(0, self._current[tag] - size_bytes)
            self._total_free[tag] += size_bytes
            if self._record_history:
                self._records.append(
                    AllocationRecord(tag=tag, size_bytes=size_bytes, freed=True)
                )

    def current_bytes(self, tag: Optional[str] = None) -> int:
        """Current live bytes for *tag* or all tags."""
        with self._lock:
            if tag is not None:
                return self._current.get(tag, 0)
            return sum(self._current.values())

    def peak_bytes(self, tag: Optional[str] = None) -> int:
        with self._lock:
            if tag is not None:
                return self._peak.get(tag, 0)
            return sum(self._peak.values())

    def report(self) -> str:
        """Human-readable memory report."""
        with self._lock:
            lines = ["Memory Tracker Report", "=" * 50]
            for tag in sorted(set(self._current) | set(self._peak)):
                cur = self._current.get(tag, 0)
                pk = self._peak.get(tag, 0)
                alloc = self._total_alloc.get(tag, 0)
                freed = self._total_free.get(tag, 0)
                lines.append(
                    f"  {tag:30s}  current={_fmt(cur):>10s}  "
                    f"peak={_fmt(pk):>10s}  alloc={_fmt(alloc):>10s}  "
                    f"freed={_fmt(freed):>10s}"
                )
            lines.append(
                f"  {'TOTAL':30s}  current={_fmt(self.current_bytes()):>10s}  "
                f"peak={_fmt(self.peak_bytes()):>10s}"
            )
            return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._current.clear()
            self._peak.clear()
            self._total_alloc.clear()
            self._total_free.clear()
            self._records.clear()


def _fmt(n: int) -> str:
    """Format byte count as human-readable string."""
    if n < 1024:
        return f"{n}B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f}KB"
    elif n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f}MB"
    return f"{n / 1024 ** 3:.2f}GB"


# Global tracker instance
_global_tracker = MemoryTracker()


def get_global_tracker() -> MemoryTracker:
    return _global_tracker


# ---------------------------------------------------------------------------
# Arena allocator
# ---------------------------------------------------------------------------

class Arena:
    """Region-based allocator for SMT clause databases.

    Objects are appended to an internal list and only freed in bulk via
    :meth:`clear`.  This avoids per-object GC overhead for short-lived
    formula/clause objects that share a CEGAR iteration lifetime.
    """

    def __init__(
        self,
        name: str = "default",
        tracker: Optional[MemoryTracker] = None,
    ) -> None:
        self.name = name
        self._objects: List[Any] = []
        self._tracker = tracker or _global_tracker
        self._alive = True

    def alloc(self, obj: T) -> T:
        """Register *obj* in this arena and return it."""
        if not self._alive:
            raise RuntimeError(f"Arena '{self.name}' has been cleared")
        self._objects.append(obj)
        size = sys.getsizeof(obj)
        self._tracker.alloc(f"arena:{self.name}", size)
        return obj

    def alloc_many(self, objs: List[T]) -> List[T]:
        """Register a batch of objects."""
        for obj in objs:
            self.alloc(obj)
        return objs

    def clear(self) -> int:
        """Free all objects in the arena.  Returns the number freed."""
        count = len(self._objects)
        total_size = sum(sys.getsizeof(o) for o in self._objects)
        self._objects.clear()
        self._tracker.free(f"arena:{self.name}", total_size)
        logger.debug(
            "Arena '%s' cleared: %d objects, %s",
            self.name,
            count,
            _fmt(total_size),
        )
        return count

    def __len__(self) -> int:
        return len(self._objects)

    def __contains__(self, obj: Any) -> bool:
        return obj in self._objects

    @property
    def size_bytes(self) -> int:
        return sum(sys.getsizeof(o) for o in self._objects)

    def invalidate(self) -> None:
        """Clear and mark the arena as dead (no further allocations)."""
        self.clear()
        self._alive = False


# ---------------------------------------------------------------------------
# Object pool
# ---------------------------------------------------------------------------

class ObjectPool(Generic[T]):
    """Reusable object pool to avoid allocation churn.

    Callers :meth:`acquire` objects from the pool (creating new ones if the
    pool is empty) and :meth:`release` them back.  A *reset_fn* is called
    on each object before it re-enters the pool.
    """

    def __init__(
        self,
        factory: Callable[[], T],
        reset_fn: Optional[Callable[[T], None]] = None,
        max_size: int = 1024,
        name: str = "pool",
        tracker: Optional[MemoryTracker] = None,
    ) -> None:
        self._factory = factory
        self._reset_fn = reset_fn
        self._max_size = max_size
        self.name = name
        self._pool: List[T] = []
        self._lock = threading.Lock()
        self._tracker = tracker or _global_tracker
        self._created = 0
        self._reused = 0

    def acquire(self) -> T:
        """Get an object from the pool or create a new one."""
        with self._lock:
            if self._pool:
                obj = self._pool.pop()
                self._reused += 1
                return obj
        obj = self._factory()
        self._created += 1
        size = sys.getsizeof(obj)
        self._tracker.alloc(f"pool:{self.name}", size)
        return obj

    def release(self, obj: T) -> None:
        """Return *obj* to the pool (reset first)."""
        if self._reset_fn is not None:
            self._reset_fn(obj)
        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(obj)
            else:
                size = sys.getsizeof(obj)
                self._tracker.free(f"pool:{self.name}", size)

    def release_many(self, objs: List[T]) -> None:
        for obj in objs:
            self.release(obj)

    def clear(self) -> int:
        """Discard all pooled objects."""
        with self._lock:
            count = len(self._pool)
            total_size = sum(sys.getsizeof(o) for o in self._pool)
            self._pool.clear()
        self._tracker.free(f"pool:{self.name}", total_size)
        return count

    @property
    def available(self) -> int:
        with self._lock:
            return len(self._pool)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "created": self._created,
            "reused": self._reused,
            "available": self.available,
            "hit_rate_pct": (
                int(self._reused * 100 / max(self._created + self._reused, 1))
            ),
        }


# ---------------------------------------------------------------------------
# Reference-counted wrapper
# ---------------------------------------------------------------------------

class RefCounted(Generic[T]):
    """Reference-counted wrapper for shared formula / expression nodes.

    When the reference count drops to zero the wrapped value is released
    (via an optional *on_release* callback) and the value is set to ``None``.
    """

    def __init__(
        self,
        value: T,
        on_release: Optional[Callable[[T], None]] = None,
    ) -> None:
        self._value: Optional[T] = value
        self._count: int = 1
        self._on_release = on_release
        self._lock = threading.Lock()

    def inc_ref(self) -> "RefCounted[T]":
        """Increment reference count (returns self for chaining)."""
        with self._lock:
            if self._value is None:
                raise RuntimeError("Cannot inc_ref a released RefCounted")
            self._count += 1
        return self

    def dec_ref(self) -> int:
        """Decrement reference count; release when it hits zero."""
        with self._lock:
            if self._count <= 0:
                raise RuntimeError("Reference count already zero")
            self._count -= 1
            remaining = self._count
            if remaining == 0:
                val = self._value
                self._value = None
        if remaining == 0 and self._on_release is not None and val is not None:
            self._on_release(val)
        return remaining

    @property
    def ref_count(self) -> int:
        return self._count

    @property
    def value(self) -> T:
        if self._value is None:
            raise RuntimeError("Accessing released RefCounted value")
        return self._value

    @property
    def alive(self) -> bool:
        return self._value is not None

    def __repr__(self) -> str:
        status = "alive" if self.alive else "released"
        return f"RefCounted({status}, refs={self._count})"


# ---------------------------------------------------------------------------
# Bulk deallocator for CEGAR iterations
# ---------------------------------------------------------------------------

class IterationAllocator:
    """Manages per-iteration arenas with automatic bulk cleanup.

    Each CEGAR iteration gets its own :class:`Arena`.  When the iteration
    ends, all memory is freed in one shot.
    """

    def __init__(
        self,
        name_prefix: str = "cegar_iter",
        tracker: Optional[MemoryTracker] = None,
    ) -> None:
        self._prefix = name_prefix
        self._tracker = tracker or _global_tracker
        self._current: Optional[Arena] = None
        self._iteration: int = 0
        self._history: List[Dict[str, Any]] = []

    def begin_iteration(self) -> Arena:
        """Start a new iteration arena (clears the previous one)."""
        if self._current is not None:
            self.end_iteration()
        self._iteration += 1
        name = f"{self._prefix}_{self._iteration}"
        self._current = Arena(name=name, tracker=self._tracker)
        logger.debug("Started iteration arena '%s'", name)
        return self._current

    def end_iteration(self) -> None:
        """End the current iteration, freeing all its memory."""
        if self._current is None:
            return
        record = {
            "iteration": self._iteration,
            "objects": len(self._current),
            "size_bytes": self._current.size_bytes,
        }
        self._current.invalidate()
        self._history.append(record)
        self._current = None

    @property
    def current_arena(self) -> Optional[Arena]:
        return self._current

    @property
    def iteration_number(self) -> int:
        return self._iteration

    def report(self) -> str:
        lines = [f"IterationAllocator '{self._prefix}' — {self._iteration} iterations"]
        for rec in self._history[-10:]:  # last 10
            lines.append(
                f"  iter {rec['iteration']}: "
                f"{rec['objects']} objects, {_fmt(rec['size_bytes'])}"
            )
        return "\n".join(lines)
