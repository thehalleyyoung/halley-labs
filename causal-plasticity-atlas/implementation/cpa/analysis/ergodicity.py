"""Ergodicity checking for atlas completeness.

Assesses whether the quality-diversity search has adequately explored
the behaviour-descriptor space by estimating coverage uniformity and
mixing time.

Relates to Theorem 6 (Atlas Completeness Under Ergodic Search):
if the exploration process is ergodic with respect to the descriptor
space, then the archive converges to full coverage as the number of
evaluations grows.

Provides:

* :class:`ErgodicityChecker` – incremental coverage tracker with
  Kolmogorov–Smirnov uniformity test and mixing-time estimation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


# ===================================================================
# Dataclass
# ===================================================================


@dataclass
class ErgodicityResult:
    """Result of an ergodicity assessment.

    Attributes
    ----------
    is_ergodic : bool
        Whether the exploration appears ergodic.
    mixing_time_estimate : float
        Estimated mixing time (iterations to near-uniform coverage).
    coverage_fraction : float
        Fraction of the descriptor space that has been visited.
    uncovered_regions : list
        Identifiers of regions not yet visited.
    uniformity_pvalue : float
        p-value from the KS uniformity test.
    total_visits : int
        Total number of descriptor observations processed.
    """

    is_ergodic: bool
    mixing_time_estimate: float
    coverage_fraction: float
    uncovered_regions: List[Any]
    uniformity_pvalue: float = 1.0
    total_visits: int = 0


# ===================================================================
# ErgodicityChecker
# ===================================================================


class ErgodicityChecker:
    """Check ergodicity of a QD exploration process.

    Maintains a multi-dimensional histogram of visits in the
    behaviour-descriptor space and uses statistical tests to assess
    whether the exploration is approaching uniform coverage.

    Parameters
    ----------
    descriptor_space_dim : int
        Dimensionality of the behaviour descriptor.
    n_bins_per_dim : int
        Number of bins per descriptor dimension.
    significance_level : float
        Significance level for uniformity tests.
    descriptor_bounds : list of (float, float) or None
        Bounds ``(lo, hi)`` for each descriptor dimension.
        If ``None``, bounds are estimated from incoming data.
    """

    def __init__(
        self,
        descriptor_space_dim: int = 4,
        n_bins_per_dim: int = 10,
        significance_level: float = 0.05,
        descriptor_bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self._dim = descriptor_space_dim
        self._n_bins = n_bins_per_dim
        self._alpha = significance_level

        self._n_cells: int = n_bins_per_dim ** descriptor_space_dim
        self._visit_counts: NDArray = np.zeros(self._n_cells, dtype=np.int64)
        self._total_visits: int = 0

        if descriptor_bounds is not None:
            self._bounds = np.asarray(descriptor_bounds, dtype=np.float64)
        else:
            self._bounds: Optional[NDArray] = None

        self._descriptor_log: List[NDArray] = []
        self._coverage_snapshots: List[float] = []

    # Also support the skeleton's n_cells-based constructor
    @classmethod
    def from_n_cells(
        cls,
        n_cells: int,
        significance_level: float = 0.05,
    ) -> "ErgodicityChecker":
        """Construct from a flat cell count (1-D descriptor space)."""
        obj = cls(
            descriptor_space_dim=1,
            n_bins_per_dim=n_cells,
            significance_level=significance_level,
        )
        obj._n_cells = n_cells
        obj._visit_counts = np.zeros(n_cells, dtype=np.int64)
        return obj

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def update(self, descriptor_or_cells: Any) -> None:
        """Record new descriptor observation(s).

        Accepts either:
        * A single descriptor vector (array-like of length ``dim``).
        * A 2-D array of descriptors, shape ``(n, dim)``.
        * A set of integer cell indices (for flat-cell mode).

        Parameters
        ----------
        descriptor_or_cells : array-like or set of int
        """
        if isinstance(descriptor_or_cells, set):
            for idx in descriptor_or_cells:
                if 0 <= idx < self._n_cells:
                    self._visit_counts[idx] += 1
                    self._total_visits += 1
            self._coverage_snapshots.append(self.coverage())
            return

        arr = np.asarray(descriptor_or_cells, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if self._bounds is None:
            self._bounds = np.column_stack(
                [arr.min(axis=0) - 1e-8, arr.max(axis=0) + 1e-8]
            )
        else:
            for row in arr:
                lo = self._bounds[:, 0]
                hi = self._bounds[:, 1]
                lo_new = np.minimum(lo, row - 1e-8)
                hi_new = np.maximum(hi, row + 1e-8)
                self._bounds = np.column_stack([lo_new, hi_new])

        for row in arr:
            bin_idx = self._bin_descriptor(row)
            if 0 <= bin_idx < self._n_cells:
                self._visit_counts[bin_idx] += 1
                self._total_visits += 1
            self._descriptor_log.append(row.copy())

        self._coverage_snapshots.append(self.coverage())

    def check(self) -> ErgodicityResult:
        """Evaluate the current ergodicity status.

        Returns
        -------
        ErgodicityResult
        """
        cov = self.coverage()
        mt = self.estimated_mixing_time()
        uncovered = self._uncovered_cells()
        pval = self._kolmogorov_smirnov_uniformity(self._visit_counts)

        is_erg = pval >= self._alpha and cov > 0.5
        if self._total_visits < self._n_cells:
            is_erg = False

        return ErgodicityResult(
            is_ergodic=is_erg,
            mixing_time_estimate=mt,
            coverage_fraction=cov,
            uncovered_regions=uncovered,
            uniformity_pvalue=pval,
            total_visits=self._total_visits,
        )

    def is_ergodic(self, tolerance: float = 0.01) -> bool:
        """Quick check if exploration is ergodic.

        Parameters
        ----------
        tolerance : float
            Minimum coverage fraction required.

        Returns
        -------
        bool
        """
        result = self.check()
        return result.is_ergodic and result.coverage_fraction >= (1.0 - tolerance)

    def coverage(self) -> float:
        """Fraction of the descriptor space that has been visited.

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        return self.coverage_fraction()

    def coverage_fraction(self) -> float:
        """Fraction of descriptor space explored."""
        if self._n_cells == 0:
            return 0.0
        filled = int(np.sum(self._visit_counts > 0))
        return float(filled) / float(self._n_cells)

    def expected_coverage(self, n_samples: int) -> float:
        """Expected coverage for *n_samples* under uniform sampling.

        Uses the coupon-collector approximation::

            E[coverage] = 1 - (1 - 1/M)^n

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        float
        """
        M = self._n_cells
        if M <= 0:
            return 0.0
        return 1.0 - (1.0 - 1.0 / M) ** n_samples

    def estimated_mixing_time(self) -> float:
        """Estimate the mixing time from the coverage trajectory.

        Uses the coupon collector bound: T_mix ~ M * ln(M) where
        M is the number of cells, and corrects using the observed
        coverage rate.

        Returns
        -------
        float
        """
        M = self._n_cells
        if M <= 1:
            return 1.0

        coupon_bound = float(M * np.log(M))

        if self._total_visits > 0 and self.coverage_fraction() > 0:
            empirical_rate = self.coverage_fraction() / self._total_visits
            if empirical_rate > 1e-12:
                empirical_estimate = 1.0 / empirical_rate
                return min(coupon_bound, empirical_estimate)
        return coupon_bound

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    def _bin_descriptor(self, descriptor: NDArray) -> int:
        """Map a descriptor vector to a flat bin index.

        Parameters
        ----------
        descriptor : NDArray, shape (dim,)

        Returns
        -------
        int
            Flat bin index in ``[0, n_cells)``.
        """
        if self._bounds is None:
            return 0

        lo = self._bounds[:, 0]
        hi = self._bounds[:, 1]
        ranges = hi - lo
        ranges[ranges < 1e-12] = 1.0

        normalised = (descriptor - lo) / ranges
        bin_per_dim = np.clip(
            np.floor(normalised * self._n_bins).astype(np.int64),
            0,
            self._n_bins - 1,
        )

        flat = 0
        for d in range(self._dim):
            if d < len(bin_per_dim):
                flat = flat * self._n_bins + int(bin_per_dim[d])
            else:
                flat = flat * self._n_bins
        return flat

    @staticmethod
    def _kolmogorov_smirnov_uniformity(bin_counts: NDArray) -> float:
        """Test uniformity of the bin counts using a chi-squared test.

        Parameters
        ----------
        bin_counts : NDArray
            Visit counts per bin.

        Returns
        -------
        float
            p-value (large = consistent with uniform).
        """
        total = np.sum(bin_counts)
        if total == 0:
            return 1.0

        n_bins = len(bin_counts)
        expected = float(total) / n_bins
        if expected < 1e-12:
            return 1.0

        chi2 = np.sum((bin_counts.astype(np.float64) - expected) ** 2 / expected)
        df = n_bins - 1
        if df <= 0:
            return 1.0
        pval = float(1.0 - sp_stats.chi2.cdf(chi2, df))
        return pval

    def _time_averaged_coverage(
        self,
        descriptors: Sequence[NDArray],
        window: int = 100,
    ) -> List[float]:
        """Compute time-averaged coverage over a sliding window.

        Parameters
        ----------
        descriptors : sequence of NDArray
            Descriptor vectors in arrival order.
        window : int
            Sliding window size.

        Returns
        -------
        list of float
        """
        if len(descriptors) == 0:
            return []

        coverages: List[float] = []
        local_counts = np.zeros(self._n_cells, dtype=np.int64)

        for i, desc in enumerate(descriptors):
            idx = self._bin_descriptor(desc)
            if 0 <= idx < self._n_cells:
                local_counts[idx] += 1

            if i >= window:
                old_desc = descriptors[i - window]
                old_idx = self._bin_descriptor(old_desc)
                if 0 <= old_idx < self._n_cells:
                    local_counts[old_idx] = max(0, local_counts[old_idx] - 1)

            filled = int(np.sum(local_counts > 0))
            coverages.append(float(filled) / max(self._n_cells, 1))

        return coverages

    def _uncovered_cells(self) -> List[int]:
        """Return the list of cell indices that have zero visits."""
        return [int(i) for i in range(self._n_cells) if self._visit_counts[i] == 0]
