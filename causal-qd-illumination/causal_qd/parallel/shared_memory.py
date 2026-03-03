"""Shared memory utilities for parallel computation.

This module provides abstractions for sharing numpy arrays and data
structures across multiple processes using ``multiprocessing.shared_memory``,
avoiding costly serialization and data duplication.

Key classes
-----------
* :class:`SharedArrayManager` – manage numpy arrays in shared memory
* :class:`SharedArchive` – archive accessible from multiple processes
* :class:`SharedScoreCache` – process-safe LRU score cache
* :class:`DataBroadcaster` – efficiently share data across processes
"""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import AdjacencyMatrix, DataMatrix

__all__ = [
    "SharedArrayManager",
    "SharedArchive",
    "SharedScoreCache",
    "DataBroadcaster",
]


# ---------------------------------------------------------------------------
# SharedArrayManager
# ---------------------------------------------------------------------------


class SharedArrayManager:
    """Manage numpy arrays in shared memory for inter-process communication.

    Creates and manages named shared memory blocks that back numpy arrays,
    allowing multiple processes to read/write the same data without
    serialization overhead.

    Examples
    --------
    >>> mgr = SharedArrayManager()
    >>> mgr.create("data", shape=(1000, 10), dtype=np.float64)
    >>> arr = mgr.get("data")
    >>> arr[:] = some_data
    >>> mgr.cleanup()
    """

    def __init__(self) -> None:
        self._blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._shapes: Dict[str, Tuple[int, ...]] = {}
        self._dtypes: Dict[str, np.dtype] = {}
        self._lock = threading.Lock()

    def create(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype | type = np.float64,
    ) -> np.ndarray:
        """Create a new shared memory array.

        Parameters
        ----------
        name : str
            Unique identifier for this array.
        shape : tuple of int
            Array shape.
        dtype : np.dtype or type
            Numpy data type.

        Returns
        -------
        np.ndarray
            View into the shared memory block.
        """
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize

        with self._lock:
            if name in self._blocks:
                raise ValueError(f"Shared array '{name}' already exists.")

            shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
            self._blocks[name] = shm
            self._shapes[name] = shape
            self._dtypes[name] = dtype

        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        arr[:] = 0
        return arr

    def get(self, name: str) -> np.ndarray:
        """Get a numpy view of an existing shared memory array.

        Parameters
        ----------
        name : str
            Identifier of the shared array.

        Returns
        -------
        np.ndarray
            View into the shared memory block.
        """
        with self._lock:
            if name not in self._blocks:
                raise KeyError(f"Shared array '{name}' not found.")
            shm = self._blocks[name]
            shape = self._shapes[name]
            dtype = self._dtypes[name]

        return np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    def get_info(self, name: str) -> Dict[str, Any]:
        """Get metadata for a shared array.

        Returns
        -------
        dict
            Keys: ``shm_name``, ``shape``, ``dtype``, ``nbytes``.
        """
        with self._lock:
            if name not in self._blocks:
                raise KeyError(f"Shared array '{name}' not found.")
            return {
                "shm_name": self._blocks[name].name,
                "shape": self._shapes[name],
                "dtype": str(self._dtypes[name]),
                "nbytes": self._blocks[name].size,
            }

    def attach(
        self,
        name: str,
        shm_name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype | type = np.float64,
    ) -> np.ndarray:
        """Attach to an existing shared memory block by OS name.

        Used in worker processes to connect to arrays created by the manager.

        Parameters
        ----------
        name : str
            Local name for the array.
        shm_name : str
            OS-level shared memory name.
        shape : tuple of int
            Array shape.
        dtype : np.dtype or type
            Data type.

        Returns
        -------
        np.ndarray
            View into the shared memory block.
        """
        dtype = np.dtype(dtype)
        shm = shared_memory.SharedMemory(name=shm_name, create=False)

        with self._lock:
            self._blocks[name] = shm
            self._shapes[name] = shape
            self._dtypes[name] = dtype

        return np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    def exists(self, name: str) -> bool:
        """Check if a shared array exists."""
        return name in self._blocks

    def delete(self, name: str) -> None:
        """Delete a shared memory array."""
        with self._lock:
            if name in self._blocks:
                shm = self._blocks.pop(name)
                self._shapes.pop(name, None)
                self._dtypes.pop(name, None)
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass

    def cleanup(self) -> None:
        """Clean up all shared memory blocks."""
        with self._lock:
            for name in list(self._blocks.keys()):
                shm = self._blocks.pop(name)
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
            self._shapes.clear()
            self._dtypes.clear()

    def __del__(self) -> None:
        self.cleanup()

    def __enter__(self) -> SharedArrayManager:
        return self

    def __exit__(self, *args: Any) -> None:
        self.cleanup()

    @property
    def names(self) -> List[str]:
        """List of managed array names."""
        return list(self._blocks.keys())

    def summary(self) -> Dict[str, Any]:
        """Summary of managed arrays."""
        total_bytes = 0
        info: Dict[str, Any] = {}
        for name in self._blocks:
            nbytes = self._blocks[name].size
            total_bytes += nbytes
            info[name] = {
                "shape": self._shapes[name],
                "dtype": str(self._dtypes[name]),
                "nbytes": nbytes,
            }
        return {"arrays": info, "total_bytes": total_bytes}


# ---------------------------------------------------------------------------
# SharedArchive
# ---------------------------------------------------------------------------


class SharedArchive:
    """Archive accessible from multiple processes via shared memory.

    The archive grid (quality scores, occupancy) is stored in shared
    memory, while solutions are stored locally per-process (since they
    are variable-sized adjacency matrices).

    Parameters
    ----------
    dims : tuple of int
        Grid dimensions.
    lower_bounds : np.ndarray
        Lower bounds for descriptor space.
    upper_bounds : np.ndarray
        Upper bounds for descriptor space.
    manager : SharedArrayManager, optional
        Existing shared memory manager.  If None, a new one is created.
    """

    def __init__(
        self,
        dims: Tuple[int, ...],
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        manager: Optional[SharedArrayManager] = None,
    ) -> None:
        self._dims = np.array(dims, dtype=np.int64)
        self._lower = np.asarray(lower_bounds, dtype=np.float64)
        self._upper = np.asarray(upper_bounds, dtype=np.float64)
        self._d = len(dims)
        self._total_cells = int(np.prod(self._dims))
        self._widths = (self._upper - self._lower) / self._dims.astype(np.float64)

        # Shared memory
        self._owns_manager = manager is None
        self._manager = manager or SharedArrayManager()

        self._qualities = self._manager.create(
            "archive_qualities",
            shape=(self._total_cells,),
            dtype=np.float64,
        )
        self._qualities[:] = float("-inf")

        self._occupancy = self._manager.create(
            "archive_occupancy",
            shape=(self._total_cells,),
            dtype=np.int8,
        )
        self._occupancy[:] = 0

        # Lock for thread/process safety
        self._lock = mp.Lock()

        # Local solution storage (not shared)
        self._solutions: Dict[int, np.ndarray] = {}
        self._descriptors_store: Dict[int, np.ndarray] = {}

    def _to_flat_index(self, descriptor: np.ndarray) -> int:
        clipped = np.clip(descriptor, self._lower, self._upper - 1e-12)
        coords = ((clipped - self._lower) / self._widths).astype(np.int64)
        coords = np.clip(coords, 0, self._dims - 1)
        multipliers = np.ones(self._d, dtype=np.int64)
        for i in range(self._d - 2, -1, -1):
            multipliers[i] = multipliers[i + 1] * self._dims[i + 1]
        return int(np.sum(coords * multipliers))

    def add(
        self,
        solution: np.ndarray,
        quality: float,
        descriptor: np.ndarray,
    ) -> bool:
        """Add a solution to the archive (thread-safe).

        Parameters
        ----------
        solution : np.ndarray
            Adjacency matrix.
        quality : float
            Quality score.
        descriptor : np.ndarray
            Descriptor vector.

        Returns
        -------
        bool
            True if the solution was added (improved the cell).
        """
        ci = self._to_flat_index(descriptor)

        with self._lock:
            if quality > self._qualities[ci]:
                self._qualities[ci] = quality
                self._occupancy[ci] = 1
                self._solutions[ci] = solution.copy()
                self._descriptors_store[ci] = descriptor.copy()
                return True
        return False

    def get(self, cell_index: int) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        """Get solution at a cell."""
        with self._lock:
            if cell_index not in self._solutions:
                return None
            return (
                self._solutions[cell_index].copy(),
                float(self._qualities[cell_index]),
                self._descriptors_store[cell_index].copy(),
            )

    def sample(
        self, n: int, rng: np.random.Generator
    ) -> List[Tuple[np.ndarray, float, np.ndarray]]:
        """Sample solutions uniformly from occupied cells."""
        with self._lock:
            occupied = list(self._solutions.keys())
            if not occupied:
                return []
            indices = rng.choice(
                occupied, size=min(n, len(occupied)), replace=True
            )
            return [
                (
                    self._solutions[i].copy(),
                    float(self._qualities[i]),
                    self._descriptors_store[i].copy(),
                )
                for i in indices
            ]

    @property
    def size(self) -> int:
        return int(np.sum(self._occupancy))

    @property
    def coverage(self) -> float:
        return int(np.sum(self._occupancy)) / self._total_cells

    @property
    def qd_score(self) -> float:
        mask = self._occupancy.astype(np.bool_)
        return float(np.sum(self._qualities[mask]))

    @property
    def best_quality(self) -> float:
        mask = self._occupancy.astype(np.bool_)
        if not np.any(mask):
            return float("-inf")
        return float(np.max(self._qualities[mask]))

    def clear(self) -> None:
        """Clear all solutions."""
        with self._lock:
            self._qualities[:] = float("-inf")
            self._occupancy[:] = 0
            self._solutions.clear()
            self._descriptors_store.clear()

    def cleanup(self) -> None:
        """Release shared memory resources."""
        if self._owns_manager:
            self._manager.cleanup()

    def __del__(self) -> None:
        pass  # Let manager handle cleanup

    def __repr__(self) -> str:
        return (
            f"SharedArchive(dims={tuple(self._dims)}, "
            f"size={self.size}, coverage={self.coverage:.3f})"
        )


# ---------------------------------------------------------------------------
# SharedScoreCache
# ---------------------------------------------------------------------------


class SharedScoreCache:
    """Process-safe score cache using shared memory.

    Stores local BIC scores in a fixed-size hash table backed by shared
    memory, allowing multiple worker processes to share cached scores.

    Parameters
    ----------
    max_size : int
        Maximum number of cache entries.
    n_nodes : int
        Number of nodes in the graph (for key computation).
    manager : SharedArrayManager, optional
        Existing shared memory manager.
    """

    def __init__(
        self,
        max_size: int = 100_000,
        n_nodes: int = 20,
        manager: Optional[SharedArrayManager] = None,
    ) -> None:
        self._max_size = max_size
        self._n_nodes = n_nodes
        self._owns_manager = manager is None
        self._manager = manager or SharedArrayManager()

        # Hash table: (node, parent_hash) -> score
        # Store as parallel arrays: keys and values
        self._keys = self._manager.create(
            "cache_keys",
            shape=(max_size, 2),  # [node, parent_hash]
            dtype=np.int64,
        )
        self._keys[:] = -1  # -1 = empty

        self._values = self._manager.create(
            "cache_values",
            shape=(max_size,),
            dtype=np.float64,
        )
        self._values[:] = float("nan")

        self._valid = self._manager.create(
            "cache_valid",
            shape=(max_size,),
            dtype=np.int8,
        )
        self._valid[:] = 0

        self._lock = mp.Lock()
        self._hits = mp.Value("i", 0)
        self._misses = mp.Value("i", 0)

    def _hash_parents(self, parents: Sequence[int]) -> int:
        """Hash a parent set to an integer."""
        h = 0
        for p in sorted(parents):
            h = h * 31 + p + 1
        return h

    def _slot(self, node: int, parent_hash: int) -> int:
        """Compute hash table slot."""
        combined = node * 2654435761 + parent_hash
        return int(abs(combined) % self._max_size)

    def get(
        self, node: int, parents: Sequence[int]
    ) -> Optional[float]:
        """Look up a cached score.

        Parameters
        ----------
        node : int
            Node index.
        parents : sequence of int
            Parent set.

        Returns
        -------
        float or None
            Cached score, or None if not found.
        """
        ph = self._hash_parents(parents)
        slot = self._slot(node, ph)

        # Linear probing
        for offset in range(min(8, self._max_size)):
            s = (slot + offset) % self._max_size
            if not self._valid[s]:
                self._misses.value += 1  # type: ignore[attr-defined]
                return None
            if self._keys[s, 0] == node and self._keys[s, 1] == ph:
                self._hits.value += 1  # type: ignore[attr-defined]
                return float(self._values[s])

        self._misses.value += 1  # type: ignore[attr-defined]
        return None

    def put(
        self, node: int, parents: Sequence[int], score: float
    ) -> None:
        """Store a score in the cache.

        Parameters
        ----------
        node : int
            Node index.
        parents : sequence of int
            Parent set.
        score : float
            Score to cache.
        """
        ph = self._hash_parents(parents)
        slot = self._slot(node, ph)

        with self._lock:
            for offset in range(min(8, self._max_size)):
                s = (slot + offset) % self._max_size
                if not self._valid[s] or (
                    self._keys[s, 0] == node and self._keys[s, 1] == ph
                ):
                    self._keys[s, 0] = node
                    self._keys[s, 1] = ph
                    self._values[s] = score
                    self._valid[s] = 1
                    return

            # Cache full at this region, overwrite first slot
            self._keys[slot, 0] = node
            self._keys[slot, 1] = ph
            self._values[slot] = score
            self._valid[slot] = 1

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._keys[:] = -1
            self._values[:] = float("nan")
            self._valid[:] = 0
            self._hits.value = 0  # type: ignore[attr-defined]
            self._misses.value = 0  # type: ignore[attr-defined]

    @property
    def hit_rate(self) -> float:
        h = self._hits.value  # type: ignore[attr-defined]
        m = self._misses.value  # type: ignore[attr-defined]
        total = h + m
        return h / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return int(np.sum(self._valid))

    def stats(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "max_size": self._max_size,
            "hits": self._hits.value,  # type: ignore[attr-defined]
            "misses": self._misses.value,  # type: ignore[attr-defined]
            "hit_rate": self.hit_rate,
        }

    def cleanup(self) -> None:
        """Release shared memory."""
        if self._owns_manager:
            self._manager.cleanup()


# ---------------------------------------------------------------------------
# DataBroadcaster
# ---------------------------------------------------------------------------


class DataBroadcaster:
    """Efficiently share data arrays across processes via shared memory.

    Provides a simple interface to broadcast a data matrix to worker
    processes without serialization.

    Parameters
    ----------
    manager : SharedArrayManager, optional
        Existing shared memory manager.

    Examples
    --------
    >>> bc = DataBroadcaster()
    >>> bc.broadcast("train_data", data)
    >>> # In worker process:
    >>> data = bc.receive("train_data")
    """

    def __init__(
        self, manager: Optional[SharedArrayManager] = None
    ) -> None:
        self._owns_manager = manager is None
        self._manager = manager or SharedArrayManager()
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def broadcast(self, name: str, data: np.ndarray) -> Dict[str, Any]:
        """Broadcast a numpy array to shared memory.

        Parameters
        ----------
        name : str
            Name for the shared array.
        data : np.ndarray
            Array to share.

        Returns
        -------
        dict
            Metadata needed to receive the data in another process.
        """
        arr = self._manager.create(name, data.shape, data.dtype)
        arr[:] = data
        info = self._manager.get_info(name)
        self._metadata[name] = info
        return info

    def receive(self, name: str) -> np.ndarray:
        """Receive a previously broadcast array.

        Parameters
        ----------
        name : str
            Name of the broadcast array.

        Returns
        -------
        np.ndarray
            View of the shared data.
        """
        return self._manager.get(name)

    def receive_by_info(
        self,
        name: str,
        shm_name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype | type,
    ) -> np.ndarray:
        """Receive a broadcast array using metadata (in worker process).

        Parameters
        ----------
        name : str
            Local name.
        shm_name : str
            OS-level shared memory name.
        shape : tuple of int
            Array shape.
        dtype : dtype
            Data type.

        Returns
        -------
        np.ndarray
            View of the shared data.
        """
        return self._manager.attach(name, shm_name, shape, dtype)

    @property
    def broadcast_names(self) -> List[str]:
        return list(self._metadata.keys())

    def cleanup(self) -> None:
        """Release shared memory."""
        if self._owns_manager:
            self._manager.cleanup()

    def __enter__(self) -> DataBroadcaster:
        return self

    def __exit__(self, *args: Any) -> None:
        self.cleanup()
