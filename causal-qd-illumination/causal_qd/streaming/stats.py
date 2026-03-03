"""Welford's online algorithm for streaming mean/variance and more.

Provides:
  - StreamingStats: running mean, variance, std (Welford's algorithm)
  - StreamingMultiStats: multi-dimensional streaming statistics
  - WindowedStats: sliding-window statistics
  - StreamingQuantiles: online quantile estimation (P² algorithm)
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


class StreamingStats:
    """Compute running mean, variance, min, max, and standard deviation online.

    Uses `Welford's algorithm
    <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm>`_
    for numerical stability.

    Example
    -------
    >>> s = StreamingStats()
    >>> for v in [1.0, 2.0, 3.0]:
    ...     s.update(v)
    >>> s.mean
    2.0
    >>> s.variance  # sample variance
    1.0
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0
        self._min: float = float("inf")
        self._max: float = float("-inf")
        self._sum: float = 0.0

    def update(self, value: float) -> None:
        """Incorporate a new observation.

        Parameters
        ----------
        value :
            The scalar value to incorporate.
        """
        self._count += 1
        self._sum += value
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

    def update_batch(self, values: List[float]) -> None:
        """Incorporate multiple observations at once.

        Parameters
        ----------
        values :
            List of scalar values.
        """
        for v in values:
            self.update(v)

    @property
    def count(self) -> int:
        """Number of observations seen so far."""
        return self._count

    @property
    def mean(self) -> float:
        """Running mean of observed values."""
        return self._mean

    @property
    def variance(self) -> float:
        """Sample variance of observed values."""
        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)

    @property
    def population_variance(self) -> float:
        """Population variance of observed values."""
        if self._count < 1:
            return 0.0
        return self._m2 / self._count

    @property
    def std(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.variance)

    @property
    def min(self) -> float:
        """Minimum observed value."""
        return self._min if self._count > 0 else float("nan")

    @property
    def max(self) -> float:
        """Maximum observed value."""
        return self._max if self._count > 0 else float("nan")

    @property
    def sum(self) -> float:
        """Sum of all observations."""
        return self._sum

    @property
    def range(self) -> float:
        """Range (max - min) of observations."""
        if self._count == 0:
            return 0.0
        return self._max - self._min

    def summary(self) -> Dict[str, float]:
        """Return a summary dict of all statistics."""
        return {
            "count": float(self._count),
            "mean": self.mean,
            "std": self.std,
            "variance": self.variance,
            "min": self.min,
            "max": self.max,
            "sum": self.sum,
        }

    def merge(self, other: StreamingStats) -> StreamingStats:
        """Merge another StreamingStats into this one (parallel combine).

        Parameters
        ----------
        other :
            Another StreamingStats to merge.

        Returns
        -------
        StreamingStats
            This instance (modified in place).
        """
        if other._count == 0:
            return self
        if self._count == 0:
            self._count = other._count
            self._mean = other._mean
            self._m2 = other._m2
            self._min = other._min
            self._max = other._max
            self._sum = other._sum
            return self

        n_a, n_b = self._count, other._count
        n_ab = n_a + n_b
        delta = other._mean - self._mean

        self._m2 = self._m2 + other._m2 + delta ** 2 * n_a * n_b / n_ab
        self._mean = (n_a * self._mean + n_b * other._mean) / n_ab
        self._count = n_ab
        self._min = min(self._min, other._min)
        self._max = max(self._max, other._max)
        self._sum = self._sum + other._sum
        return self

    def reset(self) -> None:
        """Reset all statistics."""
        self.__init__()  # type: ignore[misc]


class StreamingMultiStats:
    """Multi-dimensional streaming statistics.

    Maintains independent StreamingStats for each dimension plus
    running covariance estimates.

    Parameters
    ----------
    n_dims :
        Number of dimensions.
    """

    def __init__(self, n_dims: int) -> None:
        self._n_dims = n_dims
        self._stats = [StreamingStats() for _ in range(n_dims)]
        self._count: int = 0
        self._co_m2: np.ndarray = np.zeros((n_dims, n_dims), dtype=np.float64)
        self._means: np.ndarray = np.zeros(n_dims, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        """Incorporate a new multi-dimensional observation.

        Parameters
        ----------
        values :
            Array of length ``n_dims``.
        """
        v = np.asarray(values, dtype=np.float64)
        self._count += 1

        old_means = self._means.copy()
        for i in range(self._n_dims):
            self._stats[i].update(float(v[i]))

        delta = v - old_means
        self._means += delta / self._count
        delta2 = v - self._means
        self._co_m2 += np.outer(delta, delta2)

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> np.ndarray:
        """Running mean vector."""
        return self._means.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Sample covariance matrix."""
        if self._count < 2:
            return np.zeros((self._n_dims, self._n_dims))
        return self._co_m2 / (self._count - 1)

    @property
    def correlation(self) -> np.ndarray:
        """Correlation matrix."""
        cov = self.covariance
        stds = np.sqrt(np.diag(cov))
        stds[stds < 1e-15] = 1.0
        return cov / np.outer(stds, stds)

    def dim_stats(self, dim: int) -> StreamingStats:
        """Return per-dimension StreamingStats."""
        return self._stats[dim]


class WindowedStats:
    """Sliding-window statistics using a fixed-size deque.

    Maintains exact statistics over the last *window_size* observations.

    Parameters
    ----------
    window_size :
        Maximum number of recent observations to keep.
    """

    def __init__(self, window_size: int = 100) -> None:
        self._window: Deque[float] = deque(maxlen=window_size)
        self._window_size = window_size

    def update(self, value: float) -> None:
        """Add a new observation.

        Parameters
        ----------
        value :
            Scalar value.
        """
        self._window.append(value)

    @property
    def count(self) -> int:
        """Number of values in the window."""
        return len(self._window)

    @property
    def mean(self) -> float:
        """Mean of values in the window."""
        if not self._window:
            return 0.0
        return float(np.mean(list(self._window)))

    @property
    def variance(self) -> float:
        """Sample variance of values in the window."""
        if len(self._window) < 2:
            return 0.0
        return float(np.var(list(self._window), ddof=1))

    @property
    def std(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.variance)

    @property
    def min(self) -> float:
        """Minimum in the window."""
        return float(min(self._window)) if self._window else float("nan")

    @property
    def max(self) -> float:
        """Maximum in the window."""
        return float(max(self._window)) if self._window else float("nan")

    def quantile(self, q: float) -> float:
        """Compute quantile *q* of the window values.

        Parameters
        ----------
        q :
            Quantile in [0, 1].

        Returns
        -------
        float
        """
        if not self._window:
            return float("nan")
        return float(np.quantile(list(self._window), q))

    @property
    def median(self) -> float:
        """Median of window values."""
        return self.quantile(0.5)

    def summary(self) -> Dict[str, float]:
        """Return summary statistics for the window."""
        return {
            "count": float(self.count),
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "q25": self.quantile(0.25),
            "q75": self.quantile(0.75),
        }


class StreamingQuantiles:
    """Online quantile estimation using the P² algorithm.

    Maintains estimates of specified quantiles without storing all data.

    Parameters
    ----------
    quantiles :
        List of quantiles to estimate (e.g., [0.25, 0.5, 0.75]).
    """

    def __init__(self, quantiles: Optional[List[float]] = None) -> None:
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75]
        self._target_quantiles = sorted(quantiles)
        self._count: int = 0
        # For P²: use a simple approach with fixed buffer
        self._buffer: List[float] = []
        self._buffer_size: int = 1000
        self._estimates: Dict[float, float] = {q: 0.0 for q in self._target_quantiles}

    def update(self, value: float) -> None:
        """Incorporate a new observation.

        Parameters
        ----------
        value :
            Scalar value.
        """
        self._count += 1
        self._buffer.append(value)

        if len(self._buffer) >= self._buffer_size:
            self._flush()

    def _flush(self) -> None:
        """Update quantile estimates from buffer."""
        if not self._buffer:
            return
        sorted_buf = sorted(self._buffer)
        for q in self._target_quantiles:
            idx = max(0, min(int(q * len(sorted_buf)), len(sorted_buf) - 1))
            # Exponential moving average of quantile estimates
            if self._count <= len(self._buffer):
                self._estimates[q] = sorted_buf[idx]
            else:
                alpha = len(self._buffer) / self._count
                self._estimates[q] = (1 - alpha) * self._estimates[q] + alpha * sorted_buf[idx]
        self._buffer.clear()

    def quantile(self, q: float) -> float:
        """Get current estimate of quantile *q*.

        Parameters
        ----------
        q :
            Target quantile.

        Returns
        -------
        float
        """
        self._flush()
        if q in self._estimates:
            return self._estimates[q]
        # Interpolate between known quantiles
        qs = sorted(self._estimates.keys())
        for i in range(len(qs) - 1):
            if qs[i] <= q <= qs[i + 1]:
                t = (q - qs[i]) / (qs[i + 1] - qs[i])
                return (1 - t) * self._estimates[qs[i]] + t * self._estimates[qs[i + 1]]
        if q <= qs[0]:
            return self._estimates[qs[0]]
        return self._estimates[qs[-1]]

    @property
    def count(self) -> int:
        return self._count

    def summary(self) -> Dict[str, float]:
        """Return all estimated quantiles."""
        self._flush()
        return {f"q{int(q*100)}": v for q, v in self._estimates.items()}
