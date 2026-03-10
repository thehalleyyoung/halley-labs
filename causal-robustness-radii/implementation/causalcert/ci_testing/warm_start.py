"""
Warm-start manager for conditional independence testing.

Caches kernel matrices, Gram matrices, and Nyström landmarks across
related CI tests so that overlapping conditioning sets can share
expensive matrix computations.  Provides LRU eviction, memory-budget
tracking, and incremental kernel updates.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from causalcert.types import CITestMethod, CITestResult, NodeId, NodeSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

_KernelKey = tuple[int, ...]
"""Key for a kernel matrix: sorted tuple of column indices."""

_GramKey = tuple[int, int, tuple[int, ...]]
"""Key for a Gram matrix: (x, y, conditioning columns)."""

_LandmarkKey = tuple[int, ...]
"""Key for Nyström landmarks: sorted tuple of feature columns."""


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryBudget:
    """Tracks memory usage against a budget.

    Attributes
    ----------
    budget_bytes : int
        Maximum allowed memory in bytes.
    used_bytes : int
        Currently allocated memory in bytes.
    """

    budget_bytes: int = 512 * 1024 * 1024  # 512 MiB default
    used_bytes: int = 0

    @property
    def remaining_bytes(self) -> int:
        """Bytes remaining before budget is exceeded."""
        return max(0, self.budget_bytes - self.used_bytes)

    @property
    def utilisation(self) -> float:
        """Fraction of budget currently used."""
        if self.budget_bytes <= 0:
            return 1.0
        return self.used_bytes / self.budget_bytes

    def can_allocate(self, n_bytes: int) -> bool:
        """Return ``True`` if *n_bytes* can be allocated within budget."""
        return self.used_bytes + n_bytes <= self.budget_bytes

    def allocate(self, n_bytes: int) -> None:
        """Record an allocation of *n_bytes*."""
        self.used_bytes += n_bytes

    def free(self, n_bytes: int) -> None:
        """Record a deallocation of *n_bytes*."""
        self.used_bytes = max(0, self.used_bytes - n_bytes)


# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WarmStartStats:
    """Statistics for the warm-start cache.

    Attributes
    ----------
    kernel_hits : int
        Number of kernel matrix cache hits.
    kernel_misses : int
        Number of kernel matrix cache misses.
    gram_hits : int
        Gram matrix cache hits.
    gram_misses : int
        Gram matrix cache misses.
    landmark_hits : int
        Nyström landmark cache hits.
    landmark_misses : int
        Nyström landmark cache misses.
    evictions : int
        Total number of LRU evictions.
    incremental_updates : int
        Number of incremental kernel updates performed.
    """

    kernel_hits: int = 0
    kernel_misses: int = 0
    gram_hits: int = 0
    gram_misses: int = 0
    landmark_hits: int = 0
    landmark_misses: int = 0
    evictions: int = 0
    incremental_updates: int = 0

    @property
    def total_hits(self) -> int:
        return self.kernel_hits + self.gram_hits + self.landmark_hits

    @property
    def total_misses(self) -> int:
        return self.kernel_misses + self.gram_misses + self.landmark_misses

    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    def reset(self) -> None:
        """Reset all counters."""
        self.kernel_hits = 0
        self.kernel_misses = 0
        self.gram_hits = 0
        self.gram_misses = 0
        self.landmark_hits = 0
        self.landmark_misses = 0
        self.evictions = 0
        self.incremental_updates = 0


# ---------------------------------------------------------------------------
# Nyström landmark cache
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NystromLandmarks:
    """Cached Nyström landmarks and basis.

    Attributes
    ----------
    indices : NDArray[np.intp]
        Row indices of the selected landmark points.
    K_mm : NDArray[np.float64]
        Kernel sub-matrix among landmarks ``(m, m)``.
    K_mm_inv_sqrt : NDArray[np.float64]
        ``K_mm^{-1/2}`` for the low-rank approximation.
    feature_cols : tuple[int, ...]
        Feature columns used to compute the kernel.
    gamma : float
        Bandwidth parameter used.
    """

    indices: NDArray[np.intp]
    K_mm: NDArray[np.float64]
    K_mm_inv_sqrt: NDArray[np.float64]
    feature_cols: tuple[int, ...]
    gamma: float

    @property
    def n_landmarks(self) -> int:
        return len(self.indices)

    @property
    def memory_bytes(self) -> int:
        return (
            self.indices.nbytes
            + self.K_mm.nbytes
            + self.K_mm_inv_sqrt.nbytes
        )


# ---------------------------------------------------------------------------
# WarmStartManager
# ---------------------------------------------------------------------------


class WarmStartManager:
    """Cache manager for warm-starting kernel-based CI tests.

    Maintains LRU caches for kernel matrices, Gram matrices, and Nyström
    landmarks.  When a new CI test shares a conditioning set (or a superset /
    subset) with a previous test, the cached computation is reused or
    incrementally updated.

    Parameters
    ----------
    max_kernel_cache : int
        Maximum number of kernel matrices to cache.
    max_gram_cache : int
        Maximum number of Gram matrices to cache.
    max_landmark_cache : int
        Maximum number of Nyström landmark sets to cache.
    memory_budget_bytes : int
        Total memory budget in bytes for all caches.

    Examples
    --------
    >>> mgr = WarmStartManager(memory_budget_bytes=256 * 1024 * 1024)
    >>> K = mgr.get_or_compute_kernel(data, (0, 2, 3), gamma=0.5)
    >>> G = mgr.get_or_compute_gram(data, x=0, y=1, cond=(2, 3), gamma=0.5)
    """

    def __init__(
        self,
        max_kernel_cache: int = 128,
        max_gram_cache: int = 256,
        max_landmark_cache: int = 64,
        memory_budget_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self._max_kernel = max_kernel_cache
        self._max_gram = max_gram_cache
        self._max_landmark = max_landmark_cache

        self._kernel_cache: OrderedDict[_KernelKey, NDArray[np.float64]] = OrderedDict()
        self._gram_cache: OrderedDict[_GramKey, NDArray[np.float64]] = OrderedDict()
        self._landmark_cache: OrderedDict[_LandmarkKey, NystromLandmarks] = OrderedDict()

        self._budget = MemoryBudget(budget_bytes=memory_budget_bytes)
        self._stats = WarmStartStats()
        self._lock = threading.Lock()

    @property
    def stats(self) -> WarmStartStats:
        """Return cache statistics."""
        return self._stats

    @property
    def budget(self) -> MemoryBudget:
        """Return the memory budget tracker."""
        return self._budget

    # -- kernel cache ---

    def get_kernel(self, cols: tuple[int, ...]) -> NDArray[np.float64] | None:
        """Look up a cached kernel matrix by feature columns.

        Parameters
        ----------
        cols : tuple[int, ...]
            Sorted tuple of column indices.

        Returns
        -------
        NDArray[np.float64] | None
            Cached kernel matrix or ``None``.
        """
        key: _KernelKey = tuple(sorted(cols))
        with self._lock:
            if key in self._kernel_cache:
                self._kernel_cache.move_to_end(key)
                self._stats.kernel_hits += 1
                return self._kernel_cache[key]
            self._stats.kernel_misses += 1
            return None

    def put_kernel(
        self,
        cols: tuple[int, ...],
        K: NDArray[np.float64],
    ) -> None:
        """Store a kernel matrix in the cache.

        Parameters
        ----------
        cols : tuple[int, ...]
            Sorted tuple of column indices.
        K : NDArray[np.float64]
            Kernel matrix.
        """
        key: _KernelKey = tuple(sorted(cols))
        nbytes = K.nbytes
        with self._lock:
            self._evict_kernel_if_needed(nbytes)
            self._kernel_cache[key] = K.copy()
            self._kernel_cache.move_to_end(key)
            self._budget.allocate(nbytes)

    def _evict_kernel_if_needed(self, needed_bytes: int) -> None:
        """Evict LRU kernel entries until space is available."""
        while (
            len(self._kernel_cache) >= self._max_kernel
            or not self._budget.can_allocate(needed_bytes)
        ) and self._kernel_cache:
            _, evicted = self._kernel_cache.popitem(last=False)
            self._budget.free(evicted.nbytes)
            self._stats.evictions += 1

    def get_or_compute_kernel(
        self,
        data: NDArray[np.float64],
        cols: tuple[int, ...],
        gamma: float | None = None,
    ) -> NDArray[np.float64]:
        """Return a cached kernel matrix or compute and cache it.

        Uses the Gaussian RBF kernel.

        Parameters
        ----------
        data : NDArray[np.float64]
            Data matrix ``(n, p)``.
        cols : tuple[int, ...]
            Feature columns to use.
        gamma : float | None
            Bandwidth.  If ``None``, uses the median heuristic.

        Returns
        -------
        NDArray[np.float64]
            Kernel matrix ``(n, n)``.
        """
        cached = self.get_kernel(cols)
        if cached is not None:
            return cached

        X = np.asarray(data[:, list(cols)], dtype=np.float64)
        if gamma is None:
            gamma = self._median_heuristic(X)

        from scipy.spatial.distance import cdist
        sq_dists = cdist(X, X, metric="sqeuclidean")
        K = np.exp(-gamma * sq_dists)

        self.put_kernel(cols, K)
        return K

    # -- Gram matrix cache ---

    def get_gram(
        self,
        x: int,
        y: int,
        cond: tuple[int, ...],
    ) -> NDArray[np.float64] | None:
        """Look up a cached centered Gram matrix.

        Parameters
        ----------
        x, y : int
            Variable column indices.
        cond : tuple[int, ...]
            Conditioning column indices (sorted).

        Returns
        -------
        NDArray[np.float64] | None
        """
        key: _GramKey = (x, y, tuple(sorted(cond)))
        with self._lock:
            if key in self._gram_cache:
                self._gram_cache.move_to_end(key)
                self._stats.gram_hits += 1
                return self._gram_cache[key]
            self._stats.gram_misses += 1
            return None

    def put_gram(
        self,
        x: int,
        y: int,
        cond: tuple[int, ...],
        G: NDArray[np.float64],
    ) -> None:
        """Store a Gram matrix in the cache.

        Parameters
        ----------
        x, y : int
            Variable indices.
        cond : tuple[int, ...]
            Conditioning columns (sorted).
        G : NDArray[np.float64]
            Centered Gram matrix.
        """
        key: _GramKey = (x, y, tuple(sorted(cond)))
        nbytes = G.nbytes
        with self._lock:
            self._evict_gram_if_needed(nbytes)
            self._gram_cache[key] = G.copy()
            self._gram_cache.move_to_end(key)
            self._budget.allocate(nbytes)

    def _evict_gram_if_needed(self, needed_bytes: int) -> None:
        while (
            len(self._gram_cache) >= self._max_gram
            or not self._budget.can_allocate(needed_bytes)
        ) and self._gram_cache:
            _, evicted = self._gram_cache.popitem(last=False)
            self._budget.free(evicted.nbytes)
            self._stats.evictions += 1

    def get_or_compute_gram(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        cond: tuple[int, ...],
        gamma: float | None = None,
    ) -> NDArray[np.float64]:
        """Return a cached centered Gram matrix or compute and cache it.

        The Gram matrix is ``H K_x H`` where ``K_x`` is the kernel on column
        *x* residualised on the conditioning set via kernel regression.

        Parameters
        ----------
        data : NDArray[np.float64]
            Data matrix ``(n, p)``.
        x, y : int
            Variable column indices.
        cond : tuple[int, ...]
            Conditioning columns.
        gamma : float | None
            RBF bandwidth; ``None`` for median heuristic.

        Returns
        -------
        NDArray[np.float64]
            Centered Gram matrix ``(n, n)``.
        """
        cached = self.get_gram(x, y, cond)
        if cached is not None:
            return cached

        n = data.shape[0]

        # Kernel on x
        K_x = self.get_or_compute_kernel(data, (x,), gamma)

        if cond:
            K_z = self.get_or_compute_kernel(data, cond, gamma)
            # Residualise: K_x|z = K_x - K_z (K_z + λI)^{-1} K_x
            reg = 1e-5 * n
            K_z_reg = K_z + reg * np.eye(n, dtype=np.float64)
            try:
                L = np.linalg.cholesky(K_z_reg)
                alpha = np.linalg.solve(L, K_x)
                alpha = np.linalg.solve(L.T, alpha)
            except np.linalg.LinAlgError:
                alpha = np.linalg.solve(K_z_reg, K_x)
            K_residual = K_x - K_z @ alpha
        else:
            K_residual = K_x

        # Center: H K H where H = I - 11^T/n
        H = np.eye(n) - np.ones((n, n)) / n
        G = H @ K_residual @ H

        self.put_gram(x, y, cond, G)
        return G

    # -- incremental kernel update ---

    def incremental_kernel_update(
        self,
        data: NDArray[np.float64],
        old_cols: tuple[int, ...],
        new_cols: tuple[int, ...],
        gamma: float | None = None,
    ) -> NDArray[np.float64]:
        """Incrementally update a kernel matrix when the conditioning set changes.

        If the old kernel is cached and the new conditioning set differs by
        at most one column, computes a rank-1 update.  Otherwise falls back
        to full recomputation.

        Parameters
        ----------
        data : NDArray[np.float64]
            Data matrix.
        old_cols : tuple[int, ...]
            Previous conditioning columns.
        new_cols : tuple[int, ...]
            New conditioning columns.
        gamma : float | None
            RBF bandwidth.

        Returns
        -------
        NDArray[np.float64]
            Updated kernel matrix for *new_cols*.
        """
        old_set = set(old_cols)
        new_set = set(new_cols)
        added = new_set - old_set
        removed = old_set - new_set

        K_old = self.get_kernel(old_cols)

        if K_old is not None and len(added) <= 1 and len(removed) <= 1:
            self._stats.incremental_updates += 1

            n = data.shape[0]
            X_new = np.asarray(data[:, list(new_set)], dtype=np.float64)
            if gamma is None:
                gamma = self._median_heuristic(X_new)

            if added and not removed:
                # One column added: multiplicative kernel update
                col = list(added)[0]
                x_col = data[:, col : col + 1].astype(np.float64)
                from scipy.spatial.distance import cdist
                sq_d = cdist(x_col, x_col, metric="sqeuclidean")
                K_add = np.exp(-gamma * sq_d)
                # Product kernel: K_new = K_old * K_add (assuming product kernel)
                K_new = K_old * K_add
            elif removed and not added:
                # One column removed: divide out the removed kernel
                col = list(removed)[0]
                x_col = data[:, col : col + 1].astype(np.float64)
                from scipy.spatial.distance import cdist
                sq_d = cdist(x_col, x_col, metric="sqeuclidean")
                K_rem = np.exp(-gamma * sq_d)
                K_rem_safe = np.maximum(K_rem, 1e-12)
                K_new = K_old / K_rem_safe
            else:
                # One added, one removed: combine both
                K_new = self.get_or_compute_kernel(data, tuple(sorted(new_set)), gamma)
                return K_new

            self.put_kernel(tuple(sorted(new_set)), K_new)
            return K_new

        return self.get_or_compute_kernel(data, tuple(sorted(new_set)), gamma)

    # -- Nyström landmark cache ---

    def get_landmarks(
        self,
        cols: tuple[int, ...],
    ) -> NystromLandmarks | None:
        """Look up cached Nyström landmarks.

        Parameters
        ----------
        cols : tuple[int, ...]
            Feature columns.

        Returns
        -------
        NystromLandmarks | None
        """
        key: _LandmarkKey = tuple(sorted(cols))
        with self._lock:
            if key in self._landmark_cache:
                self._landmark_cache.move_to_end(key)
                self._stats.landmark_hits += 1
                return self._landmark_cache[key]
            self._stats.landmark_misses += 1
            return None

    def put_landmarks(
        self,
        cols: tuple[int, ...],
        landmarks: NystromLandmarks,
    ) -> None:
        """Store Nyström landmarks.

        Parameters
        ----------
        cols : tuple[int, ...]
            Feature columns.
        landmarks : NystromLandmarks
            Landmark data.
        """
        key: _LandmarkKey = tuple(sorted(cols))
        nbytes = landmarks.memory_bytes
        with self._lock:
            while (
                len(self._landmark_cache) >= self._max_landmark
                or not self._budget.can_allocate(nbytes)
            ) and self._landmark_cache:
                _, evicted = self._landmark_cache.popitem(last=False)
                self._budget.free(evicted.memory_bytes)
                self._stats.evictions += 1
            self._landmark_cache[key] = landmarks
            self._landmark_cache.move_to_end(key)
            self._budget.allocate(nbytes)

    def get_or_compute_landmarks(
        self,
        data: NDArray[np.float64],
        cols: tuple[int, ...],
        n_landmarks: int = 100,
        gamma: float | None = None,
        seed: int = 42,
    ) -> NystromLandmarks:
        """Return cached landmarks or compute and cache them.

        Parameters
        ----------
        data : NDArray[np.float64]
            Data matrix ``(n, p)``.
        cols : tuple[int, ...]
            Feature columns.
        n_landmarks : int
            Number of Nyström landmarks.
        gamma : float | None
            RBF bandwidth.
        seed : int
            Random seed for landmark selection.

        Returns
        -------
        NystromLandmarks
        """
        cached = self.get_landmarks(cols)
        if cached is not None:
            return cached

        X = np.asarray(data[:, list(cols)], dtype=np.float64)
        n = X.shape[0]
        m = min(n_landmarks, n)

        rng = np.random.RandomState(seed)
        indices = rng.choice(n, size=m, replace=False)
        indices.sort()

        if gamma is None:
            gamma = self._median_heuristic(X)

        X_m = X[indices]
        from scipy.spatial.distance import cdist
        sq_dists = cdist(X_m, X_m, metric="sqeuclidean")
        K_mm = np.exp(-gamma * sq_dists)

        # Compute K_mm^{-1/2} via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(K_mm)
        eigvals = np.maximum(eigvals, 1e-12)
        K_mm_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        lm = NystromLandmarks(
            indices=indices,
            K_mm=K_mm,
            K_mm_inv_sqrt=K_mm_inv_sqrt,
            feature_cols=tuple(sorted(cols)),
            gamma=gamma,
        )
        self.put_landmarks(cols, lm)
        return lm

    def nystrom_approximate_kernel(
        self,
        data: NDArray[np.float64],
        cols: tuple[int, ...],
        n_landmarks: int = 100,
        gamma: float | None = None,
        seed: int = 42,
    ) -> NDArray[np.float64]:
        """Compute a low-rank Nyström approximation of the kernel matrix.

        K ≈ K_nm K_mm^{-1} K_mn

        Parameters
        ----------
        data : NDArray[np.float64]
            Data matrix ``(n, p)``.
        cols : tuple[int, ...]
            Feature columns.
        n_landmarks : int
            Number of landmarks.
        gamma : float | None
            RBF bandwidth.
        seed : int
            Random seed.

        Returns
        -------
        NDArray[np.float64]
            Approximate kernel matrix ``(n, n)``.
        """
        lm = self.get_or_compute_landmarks(data, cols, n_landmarks, gamma, seed)

        X = np.asarray(data[:, list(cols)], dtype=np.float64)
        X_m = X[lm.indices]

        from scipy.spatial.distance import cdist
        sq_dists = cdist(X, X_m, metric="sqeuclidean")
        K_nm = np.exp(-lm.gamma * sq_dists)

        # Low-rank factor: L = K_nm @ K_mm^{-1/2}
        L = K_nm @ lm.K_mm_inv_sqrt
        return L @ L.T

    # -- cache management ---

    def clear(self) -> None:
        """Clear all caches and reset memory tracking."""
        with self._lock:
            self._kernel_cache.clear()
            self._gram_cache.clear()
            self._landmark_cache.clear()
            self._budget.used_bytes = 0

    def resize_budget(self, new_budget_bytes: int) -> None:
        """Resize the memory budget, evicting if necessary.

        Parameters
        ----------
        new_budget_bytes : int
            New budget in bytes.
        """
        with self._lock:
            self._budget.budget_bytes = new_budget_bytes
            while self._budget.used_bytes > new_budget_bytes:
                evicted_any = False
                if self._kernel_cache:
                    _, evicted = self._kernel_cache.popitem(last=False)
                    self._budget.free(evicted.nbytes)
                    self._stats.evictions += 1
                    evicted_any = True
                if self._gram_cache and self._budget.used_bytes > new_budget_bytes:
                    _, evicted = self._gram_cache.popitem(last=False)
                    self._budget.free(evicted.nbytes)
                    self._stats.evictions += 1
                    evicted_any = True
                if self._landmark_cache and self._budget.used_bytes > new_budget_bytes:
                    _, evicted_lm = self._landmark_cache.popitem(last=False)
                    self._budget.free(evicted_lm.memory_bytes)
                    self._stats.evictions += 1
                    evicted_any = True
                if not evicted_any:
                    break

    def summary(self) -> dict[str, Any]:
        """Return a summary of cache state.

        Returns
        -------
        dict[str, Any]
        """
        return {
            "kernel_cache_size": len(self._kernel_cache),
            "gram_cache_size": len(self._gram_cache),
            "landmark_cache_size": len(self._landmark_cache),
            "memory_used_bytes": self._budget.used_bytes,
            "memory_budget_bytes": self._budget.budget_bytes,
            "memory_utilisation": self._budget.utilisation,
            "stats": {
                "kernel_hits": self._stats.kernel_hits,
                "kernel_misses": self._stats.kernel_misses,
                "gram_hits": self._stats.gram_hits,
                "gram_misses": self._stats.gram_misses,
                "landmark_hits": self._stats.landmark_hits,
                "landmark_misses": self._stats.landmark_misses,
                "evictions": self._stats.evictions,
                "incremental_updates": self._stats.incremental_updates,
                "hit_rate": self._stats.hit_rate,
            },
        }

    # -- helpers ---

    @staticmethod
    def _median_heuristic(X: NDArray[np.float64]) -> float:
        """Compute the median-heuristic bandwidth for the RBF kernel."""
        from scipy.spatial.distance import pdist
        dists = pdist(X, metric="sqeuclidean")
        if len(dists) == 0:
            return 1.0
        med = float(np.median(dists))
        if med <= 0:
            return 1.0
        return 1.0 / med

    def __repr__(self) -> str:
        return (
            f"WarmStartManager("
            f"kernels={len(self._kernel_cache)}/{self._max_kernel}, "
            f"grams={len(self._gram_cache)}/{self._max_gram}, "
            f"landmarks={len(self._landmark_cache)}/{self._max_landmark}, "
            f"mem={self._budget.utilisation:.1%})"
        )
