"""Efficient stream buffering with sliding windows.

Provides fixed-capacity sliding windows, circular buffers,
timestamped buffers, and multi-variable stream buffers with
on-the-fly sufficient statistics computation.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# SlidingWindow – fixed-capacity scalar stream
# ---------------------------------------------------------------------------

class SlidingWindow:
    """Fixed-capacity sliding window over a scalar stream.

    Internally uses a circular buffer so that ``append`` is O(1).

    Parameters
    ----------
    max_size : int
        Maximum number of elements the window can hold.
    dtype : type
        NumPy dtype for stored values.
    """

    def __init__(self, max_size: int, dtype: type = float) -> None:
        self._max_size = max_size
        self._dtype = dtype
        self._buffer = np.empty(max_size, dtype=dtype)
        self._count = 0
        self._head = 0  # next write position
        # Running statistics (Welford)
        self._running_mean = 0.0
        self._running_m2 = 0.0

    # -- core operations ---------------------------------------------------

    def append(self, value: float) -> None:
        """Append a single *value*, evicting the oldest if full."""
        if self._count < self._max_size:
            self._buffer[self._head] = value
            self._head = (self._head + 1) % self._max_size
            self._count += 1
            delta = value - self._running_mean
            self._running_mean += delta / self._count
            self._running_m2 += delta * (value - self._running_mean)
        else:
            old_val = self._buffer[self._head]
            self._buffer[self._head] = value
            self._head = (self._head + 1) % self._max_size
            # Update running stats by removing old and adding new
            self._running_mean += (value - old_val) / self._count
            # Recompute M2 is expensive in sliding mode; mark dirty
            self._running_m2 = -1.0  # sentinel for lazy recompute

    def extend(self, values: NDArray) -> None:
        """Append multiple *values* in bulk."""
        values = np.asarray(values, dtype=self._dtype).ravel()
        for v in values:
            self.append(v)

    def get_window(self) -> NDArray:
        """Return the current window contents as an array (oldest first)."""
        if self._count == 0:
            return np.empty(0, dtype=self._dtype)
        if self._count < self._max_size:
            return self._buffer[: self._count].copy()
        start = self._head  # oldest element
        return np.concatenate(
            [self._buffer[start:], self._buffer[:start]]
        )

    # -- statistics --------------------------------------------------------

    def mean(self) -> float:
        """Return the mean of the current window."""
        if self._count == 0:
            return 0.0
        return float(np.mean(self.get_window()))

    def std(self) -> float:
        """Return the standard deviation of the current window."""
        if self._count < 2:
            return 0.0
        return float(np.std(self.get_window(), ddof=1))

    def var(self) -> float:
        """Return the variance of the current window."""
        if self._count < 2:
            return 0.0
        return float(np.var(self.get_window(), ddof=1))

    # -- queries -----------------------------------------------------------

    def is_full(self) -> bool:
        """Return ``True`` if the window has reached *max_size*."""
        return self._count >= self._max_size

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return (
            f"SlidingWindow(max_size={self._max_size}, "
            f"count={self._count})"
        )


# ---------------------------------------------------------------------------
# CircularBuffer – fixed-capacity buffer for n-d items
# ---------------------------------------------------------------------------

class CircularBuffer:
    """Fixed-size circular buffer for array-valued items.

    When full, new pushes silently overwrite the oldest entry.

    Parameters
    ----------
    capacity : int
        Maximum number of items.
    shape : tuple of int
        Shape of each item (scalar items use ``()``).
    dtype
        NumPy dtype.
    """

    def __init__(
        self,
        capacity: int,
        shape: Tuple[int, ...] = (),
        dtype: Any = np.float64,
    ) -> None:
        self._capacity = capacity
        self._shape = shape
        self._dtype = dtype
        full_shape = (capacity,) + shape if shape else (capacity,)
        self._data = np.zeros(full_shape, dtype=dtype)
        self._head = 0
        self._count = 0

    def push(self, item: NDArray) -> None:
        """Push *item*, overwriting the oldest entry when full."""
        self._data[self._head] = item
        self._head = (self._head + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1

    def peek(self, n: int = 1) -> NDArray:
        """Return the last *n* items (most-recent last) without removal."""
        n = min(n, self._count)
        if n == 0:
            if self._shape:
                return np.empty((0,) + self._shape, dtype=self._dtype)
            return np.empty(0, dtype=self._dtype)
        indices = [
            (self._head - n + i) % self._capacity for i in range(n)
        ]
        return self._data[indices].copy()

    def get_all(self) -> NDArray:
        """Return all items in chronological order."""
        return self.peek(self._count)

    def is_full(self) -> bool:
        """Return ``True`` when the buffer is at capacity."""
        return self._count >= self._capacity

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return (
            f"CircularBuffer(capacity={self._capacity}, "
            f"count={self._count}, shape={self._shape})"
        )


# ---------------------------------------------------------------------------
# TimestampedBuffer – age-limited buffer
# ---------------------------------------------------------------------------

class TimestampedBuffer:
    """Buffer that automatically expires entries older than *max_age_seconds*.

    Parameters
    ----------
    max_age_seconds : float
        Maximum age of an entry before it is eligible for expiry.
    """

    def __init__(self, max_age_seconds: float = 3600.0) -> None:
        self._max_age = max_age_seconds
        self._entries: deque[Tuple[float, NDArray]] = deque()

    def add(self, timestamp: float, data: NDArray) -> None:
        """Insert *data* with an explicit *timestamp*."""
        self._entries.append((timestamp, np.asarray(data, dtype=np.float64)))

    def add_now(self, data: NDArray) -> None:
        """Insert *data* stamped with the current wall-clock time."""
        self.add(time.time(), data)

    def get_range(
        self, start_time: float, end_time: float
    ) -> List[Tuple[float, NDArray]]:
        """Return entries whose timestamp lies in [*start_time*, *end_time*]."""
        return [
            (ts, d)
            for ts, d in self._entries
            if start_time <= ts <= end_time
        ]

    def expire(self, current_time: Optional[float] = None) -> int:
        """Remove entries older than *max_age_seconds*.

        Returns the number of entries removed.
        """
        if current_time is None:
            current_time = time.time()
        cutoff = current_time - self._max_age
        removed = 0
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()
            removed += 1
        return removed

    def get_all(self) -> List[Tuple[float, NDArray]]:
        """Return all non-expired entries."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"TimestampedBuffer(max_age={self._max_age}s, "
            f"entries={len(self._entries)})"
        )


# ---------------------------------------------------------------------------
# StreamBuffer – multi-variable buffer with sufficient statistics
# ---------------------------------------------------------------------------

class StreamBuffer:
    """Multi-variable stream buffer backed by sliding windows.

    Maintains per-variable sliding windows and provides on-the-fly
    sufficient statistics (mean vector, covariance matrix, sample count).

    Parameters
    ----------
    n_variables : int
        Number of variables tracked simultaneously.
    window_size : int
        Sliding-window capacity for each variable.
    """

    def __init__(self, n_variables: int, window_size: int) -> None:
        self._n_variables = n_variables
        self._window_size = window_size
        self._windows = [
            SlidingWindow(window_size) for _ in range(n_variables)
        ]
        self._push_count = 0

    # -- ingestion ---------------------------------------------------------

    def push(self, observation: NDArray) -> None:
        """Push a single observation vector into all variable windows.

        Parameters
        ----------
        observation : array of shape ``(n_variables,)``
        """
        obs = np.asarray(observation, dtype=np.float64).ravel()
        if obs.shape[0] != self._n_variables:
            raise ValueError(
                f"Expected {self._n_variables} variables, "
                f"got {obs.shape[0]}"
            )
        for i, val in enumerate(obs):
            self._windows[i].append(val)
        self._push_count += 1

    def extend(self, data: NDArray) -> None:
        """Push multiple observation rows at once.

        Parameters
        ----------
        data : array of shape ``(n_obs, n_variables)``
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        for row in data:
            self.push(row)

    # -- retrieval ---------------------------------------------------------

    def get_data(self) -> NDArray:
        """Return buffered data as a 2-D array ``(time × variables)``."""
        columns = [w.get_window() for w in self._windows]
        if len(columns[0]) == 0:
            return np.empty((0, self._n_variables), dtype=np.float64)
        min_len = min(len(c) for c in columns)
        return np.column_stack([c[:min_len] for c in columns])

    def get_variable(self, idx: int) -> NDArray:
        """Return the sliding window for variable *idx*."""
        if idx < 0 or idx >= self._n_variables:
            raise IndexError(f"Variable index {idx} out of range")
        return self._windows[idx].get_window()

    def get_window(self, size: int) -> NDArray:
        """Return the last *size* observations as ``(size, n_variables)``."""
        data = self.get_data()
        if data.shape[0] == 0:
            return data
        return data[-size:]

    # -- statistics --------------------------------------------------------

    def sufficient_statistics(self) -> Dict:
        """Compute and return running sufficient statistics.

        Returns
        -------
        dict
            ``mean`` – ``(p,)`` mean vector,
            ``covariance`` – ``(p, p)`` sample covariance,
            ``n`` – number of observations in the window.
        """
        data = self.get_data()
        n = data.shape[0]
        if n == 0:
            p = self._n_variables
            return {
                "mean": np.zeros(p),
                "covariance": np.zeros((p, p)),
                "n": 0,
            }
        mean = np.mean(data, axis=0)
        if n < 2:
            cov = np.zeros((self._n_variables, self._n_variables))
        else:
            cov = np.cov(data, rowvar=False, ddof=1)
            if cov.ndim == 0:
                cov = cov.reshape(1, 1)
        return {"mean": mean, "covariance": cov, "n": n}

    def statistics(self) -> Dict:
        """Alias for :meth:`sufficient_statistics`."""
        return self.sufficient_statistics()

    # -- management --------------------------------------------------------

    def clear(self) -> None:
        """Reset all variable windows."""
        self._windows = [
            SlidingWindow(self._window_size) for _ in range(self._n_variables)
        ]
        self._push_count = 0

    def __len__(self) -> int:
        """Current number of observations stored (min across variables)."""
        if self._n_variables == 0:
            return 0
        return min(len(w) for w in self._windows)

    def __repr__(self) -> str:
        return (
            f"StreamBuffer(n_variables={self._n_variables}, "
            f"window_size={self._window_size}, "
            f"observations={len(self)})"
        )
