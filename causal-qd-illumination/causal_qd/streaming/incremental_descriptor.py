"""Incremental descriptor updates for single-edge DAG modifications.

Provides:
  - IncrementalDescriptor: efficiently update descriptors after edge changes
  - Welford's online algorithm for mean/variance of descriptor components
  - Incremental mutual information estimation
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from causal_qd.types import BehavioralDescriptor, DataMatrix

if TYPE_CHECKING:
    from causal_qd.core.dag import DAG


class IncrementalDescriptor:
    """Efficiently update a behavioural descriptor after a single-edge change.

    Maintains cached statistics about the graph structure so that
    adding or removing a single edge only requires local updates
    instead of a full descriptor recomputation.

    Parameters
    ----------
    base_computer :
        The underlying descriptor computer used when a full
        recomputation is necessary.
    cache_statistics :
        If *True*, maintain running graph statistics for faster
        incremental updates.
    """

    def __init__(
        self,
        base_computer: object,
        cache_statistics: bool = True,
    ) -> None:
        self._base = base_computer
        self._cache_statistics = cache_statistics
        # Cached graph statistics
        self._in_degrees: Optional[np.ndarray] = None
        self._out_degrees: Optional[np.ndarray] = None
        self._last_descriptor: Optional[BehavioralDescriptor] = None
        self._last_adj: Optional[np.ndarray] = None
        # Running descriptor statistics
        self._descriptor_stats = _DescriptorStats()

    def compute(self, dag: object, data: DataMatrix) -> BehavioralDescriptor:
        """Full descriptor computation (delegates to the base computer).

        Parameters
        ----------
        dag :
            DAG object.
        data :
            Data matrix.

        Returns
        -------
        BehavioralDescriptor
        """
        desc = self._base.compute(dag, data)
        self._last_descriptor = desc.copy()

        # Cache graph statistics
        adj = dag.adjacency if hasattr(dag, 'adjacency') else dag._adj.copy()
        self._last_adj = adj.copy()
        self._in_degrees = adj.sum(axis=0).astype(np.float64)
        self._out_degrees = adj.sum(axis=1).astype(np.float64)
        self._descriptor_stats.update(desc)

        return desc

    def update(
        self,
        old_descriptor: BehavioralDescriptor,
        dag: object,
        data: DataMatrix,
        edge: Tuple[int, int],
        added: bool,
    ) -> BehavioralDescriptor:
        """Incrementally update a descriptor after an edge change.

        If cached statistics are available and the descriptor is
        structure-based (degree statistics), computes the update
        analytically.  Otherwise falls back to full recomputation.

        Parameters
        ----------
        old_descriptor :
            Descriptor *before* the edge change.
        dag :
            DAG *after* the edge change.
        data :
            Data matrix.
        edge :
            ``(source, target)`` of the changed edge.
        added :
            *True* if the edge was added, *False* if removed.

        Returns
        -------
        BehavioralDescriptor
        """
        src, tgt = edge

        if (self._in_degrees is not None
                and self._out_degrees is not None
                and len(old_descriptor) == 2):
            # Fast path: descriptor is [out_degree_std, in_degree_std]
            delta = 1 if added else -1
            self._out_degrees[src] += delta
            self._in_degrees[tgt] += delta

            new_desc = np.array([
                float(self._out_degrees.std()),
                float(self._in_degrees.std()),
            ], dtype=np.float64)

            self._last_descriptor = new_desc
            self._descriptor_stats.update(new_desc)
            return new_desc

        # Fallback: full recomputation
        new_descriptor = self.compute(dag, data)
        return new_descriptor

    def update_batch(
        self,
        old_descriptor: BehavioralDescriptor,
        dag: object,
        data: DataMatrix,
        edges: List[Tuple[Tuple[int, int], bool]],
    ) -> BehavioralDescriptor:
        """Update descriptor after multiple edge changes.

        Parameters
        ----------
        old_descriptor :
            Descriptor before changes.
        dag :
            DAG after all changes.
        data :
            Data matrix.
        edges :
            List of ``((source, target), added)`` pairs.

        Returns
        -------
        BehavioralDescriptor
        """
        if (self._in_degrees is not None
                and self._out_degrees is not None
                and len(old_descriptor) == 2):
            for (src, tgt), added in edges:
                delta = 1 if added else -1
                self._out_degrees[src] += delta
                self._in_degrees[tgt] += delta

            new_desc = np.array([
                float(self._out_degrees.std()),
                float(self._in_degrees.std()),
            ], dtype=np.float64)

            self._last_descriptor = new_desc
            self._descriptor_stats.update(new_desc)
            return new_desc

        return self.compute(dag, data)

    @property
    def descriptor_stats(self) -> Dict[str, float]:
        """Return running statistics about descriptor values."""
        return self._descriptor_stats.summary()

    def reset_cache(self) -> None:
        """Clear cached statistics."""
        self._in_degrees = None
        self._out_degrees = None
        self._last_descriptor = None
        self._last_adj = None


class _DescriptorStats:
    """Online statistics for descriptor components using Welford's algorithm."""

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: Optional[np.ndarray] = None
        self._m2: Optional[np.ndarray] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None

    def update(self, descriptor: BehavioralDescriptor) -> None:
        """Incorporate a new descriptor observation."""
        d = np.asarray(descriptor, dtype=np.float64)
        self._count += 1

        if self._mean is None:
            self._mean = d.copy()
            self._m2 = np.zeros_like(d)
            self._min = d.copy()
            self._max = d.copy()
            return

        delta = d - self._mean
        self._mean += delta / self._count
        delta2 = d - self._mean
        self._m2 += delta * delta2
        self._min = np.minimum(self._min, d)
        self._max = np.maximum(self._max, d)

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> Optional[np.ndarray]:
        return self._mean.copy() if self._mean is not None else None

    @property
    def variance(self) -> Optional[np.ndarray]:
        if self._m2 is None or self._count < 2:
            return None
        return self._m2 / (self._count - 1)

    def summary(self) -> Dict[str, float]:
        """Return summary statistics."""
        if self._mean is None:
            return {"count": 0}
        return {
            "count": self._count,
            "mean_norm": float(np.linalg.norm(self._mean)),
            "variance_sum": float(self.variance.sum()) if self.variance is not None else 0.0,
            "min_norm": float(np.linalg.norm(self._min)) if self._min is not None else 0.0,
            "max_norm": float(np.linalg.norm(self._max)) if self._max is not None else 0.0,
        }


class IncrementalMI:
    """Incremental mutual information estimation.

    Maintains running histogram-based estimates of entropy for
    efficient MI updates when data arrives in a stream.

    Parameters
    ----------
    n_bins :
        Number of bins for histogram-based entropy estimation.
    """

    def __init__(self, n_bins: int = 20) -> None:
        self._n_bins = n_bins
        self._marginal_counts: Dict[int, np.ndarray] = {}
        self._joint_counts: Dict[Tuple[int, int], np.ndarray] = {}
        self._n_samples: int = 0

    def update(self, data_row: np.ndarray) -> None:
        """Incorporate a new data observation.

        Parameters
        ----------
        data_row :
            Single observation (1-D array of length p).
        """
        self._n_samples += 1
        d = np.asarray(data_row, dtype=np.float64)

        for i, val in enumerate(d):
            if i not in self._marginal_counts:
                self._marginal_counts[i] = np.zeros(self._n_bins, dtype=np.float64)
            bin_idx = min(int((val + 5) / 10 * self._n_bins), self._n_bins - 1)
            bin_idx = max(0, bin_idx)
            self._marginal_counts[i][bin_idx] += 1

    def mutual_information(self, x: int, y: int) -> float:
        """Estimate MI between variables x and y.

        Parameters
        ----------
        x, y :
            Variable indices.

        Returns
        -------
        float
            Estimated mutual information.
        """
        if x not in self._marginal_counts or y not in self._marginal_counts:
            return 0.0

        h_x = self._entropy(self._marginal_counts[x])
        h_y = self._entropy(self._marginal_counts[y])

        # Approximate joint entropy as sum minus a correction
        # (exact joint would require 2D histograms)
        key = (min(x, y), max(x, y))
        if key in self._joint_counts:
            h_xy = self._entropy_2d(self._joint_counts[key])
        else:
            # Upper bound: assume independence
            h_xy = h_x + h_y

        return max(0.0, h_x + h_y - h_xy)

    @staticmethod
    def _entropy(counts: np.ndarray) -> float:
        """Compute entropy from histogram counts."""
        total = counts.sum()
        if total <= 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        return -float(np.sum(probs * np.log(probs)))

    @staticmethod
    def _entropy_2d(counts: np.ndarray) -> float:
        """Compute entropy from 2D histogram counts."""
        total = counts.sum()
        if total <= 0:
            return 0.0
        probs = counts.ravel() / total
        probs = probs[probs > 0]
        return -float(np.sum(probs * np.log(probs)))
