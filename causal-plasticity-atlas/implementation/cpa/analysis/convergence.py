"""Convergence analysis for quality-diversity search.

Tracks archive coverage, mean quality, improvement rate, and
stagnation to decide when a QD search run has converged.

Provides two main classes:

* :class:`ConvergenceAnalyzer` – tracks per-iteration metrics and
  decides when to stop based on moving-average improvement rates.
* :class:`ArchiveDiversity` – computes diversity descriptors of the
  current archive (pairwise distances, coverage ratio, quality
  variance, descriptor entropy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


# ===================================================================
# Dataclasses
# ===================================================================


@dataclass
class ConvergenceMetrics:
    """Snapshot of convergence-related metrics.

    Attributes
    ----------
    archive_coverage : float
        Fraction of archive cells that are occupied.
    mean_quality : float
        Mean fitness / quality across occupied cells.
    quality_variance : float
        Variance of quality across occupied cells.
    improvement_rate : float
        Rate of quality improvement over the recent window.
    stagnation_count : int
        Number of consecutive updates with no improvement.
    """

    archive_coverage: float
    mean_quality: float
    quality_variance: float
    improvement_rate: float
    stagnation_count: int


# ===================================================================
# ConvergenceAnalyzer
# ===================================================================


class ConvergenceAnalyzer:
    """Track and assess QD search convergence.

    The analyser accumulates per-iteration snapshots of archive size,
    best quality and coverage.  It uses a moving-average window to
    estimate improvement rates and declares convergence when the
    improvement rate drops below a tolerance for ``patience``
    consecutive updates.

    Parameters
    ----------
    window_size : int
        Number of recent updates to consider for trend estimation.
    patience : int
        Number of consecutive stagnation steps before convergence is
        declared.
    """

    def __init__(self, window_size: int = 100, patience: int = 50) -> None:
        self._window_size = window_size
        self._patience = patience

        self._iterations: List[int] = []
        self._archive_sizes: List[int] = []
        self._best_qualities: List[float] = []
        self._coverage_history: List[float] = []
        self._quality_history: List[float] = []
        self._quality_var_history: List[float] = []
        self._improvement_history: List[float] = []
        self._stagnation_count: int = 0
        self._best_seen: float = -np.inf

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def update(
        self,
        archive_state: Dict[str, Any],
    ) -> ConvergenceMetrics:
        """Record a new archive snapshot and return current metrics.

        Parameters
        ----------
        archive_state : dict
            Must contain at least::

                {
                    "coverage": float,        # fraction in [0, 1]
                    "qualities": array-like,   # quality of each occupied cell
                }

            Optional keys: ``"iteration"`` (int), ``"archive_size"`` (int),
            ``"best_quality"`` (float).

        Returns
        -------
        ConvergenceMetrics
        """
        coverage = float(archive_state["coverage"])
        qualities = np.asarray(archive_state["qualities"], dtype=np.float64)
        iteration = int(archive_state.get("iteration", len(self._iterations)))
        archive_size = int(archive_state.get("archive_size", len(qualities)))
        best_quality = float(
            archive_state.get("best_quality", np.max(qualities) if len(qualities) > 0 else 0.0)
        )

        mean_q = float(np.mean(qualities)) if len(qualities) > 0 else 0.0
        var_q = float(np.var(qualities)) if len(qualities) > 0 else 0.0

        self._iterations.append(iteration)
        self._archive_sizes.append(archive_size)
        self._best_qualities.append(best_quality)
        self._coverage_history.append(coverage)
        self._quality_history.append(mean_q)
        self._quality_var_history.append(var_q)

        imp_rate = self._relative_improvement(self._quality_history, self._window_size)
        self._improvement_history.append(imp_rate)

        if best_quality > self._best_seen + 1e-12:
            self._best_seen = best_quality
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1

        return ConvergenceMetrics(
            archive_coverage=coverage,
            mean_quality=mean_q,
            quality_variance=var_q,
            improvement_rate=imp_rate,
            stagnation_count=self._stagnation_count,
        )

    def is_converged(self, tolerance: float = 1e-6) -> bool:
        """Check whether the search has converged.

        Convergence is declared when:

        1. The stagnation count exceeds ``patience``, **or**
        2. The moving-average improvement rate is below ``tolerance``
           for the last ``patience`` observations.

        Parameters
        ----------
        tolerance : float
            Minimum improvement rate to consider non-converged.

        Returns
        -------
        bool
        """
        if len(self._improvement_history) < self._patience:
            return False
        if self._stagnation_count >= self._patience:
            return True
        recent = self._improvement_history[-self._patience :]
        return all(abs(r) < tolerance for r in recent)

    def convergence_rate(self) -> float:
        """Estimate the current convergence rate.

        Uses an exponential-decay fit on the recent improvement history
        to estimate the rate parameter.

        Returns
        -------
        float
            Estimated rate (positive means converging).
        """
        if len(self._improvement_history) < 3:
            return 0.0
        window = self._improvement_history[-self._window_size :]
        y = np.array(window, dtype=np.float64)
        y_abs = np.abs(y) + 1e-15
        log_y = np.log(y_abs)
        x = np.arange(len(log_y), dtype=np.float64)
        if len(x) < 2:
            return 0.0
        slope, _, _, _, _ = sp_stats.linregress(x, log_y)
        return float(-slope)

    def coverage_curve(self) -> List[float]:
        """Return the full history of archive coverage values."""
        return list(self._coverage_history)

    def quality_curve(self) -> List[float]:
        """Return the full history of mean quality values."""
        return list(self._quality_history)

    def plot_convergence(self, ax: Any = None) -> Any:
        """Plot convergence curves on *ax*.

        If *ax* is ``None`` a new matplotlib figure is created.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Target axes.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        iters = np.arange(len(self._coverage_history))

        if hasattr(ax, "__len__") and len(ax) >= 2:
            ax0, ax1 = ax[0], ax[1]
        else:
            ax0 = ax1 = ax

        ax0.plot(iters, self._coverage_history, label="Coverage", color="steelblue")
        ax0.set_ylabel("Coverage")
        ax0.legend(loc="lower right")
        ax0.set_title("QD Search Convergence")

        ax1.plot(iters, self._quality_history, label="Mean Quality", color="coral")
        ma = self._moving_average(self._quality_history, self._window_size)
        if len(ma) > 0:
            ax1.plot(
                iters[self._window_size - 1 :],
                ma,
                label=f"MA({self._window_size})",
                color="darkred",
                linewidth=2,
            )
        ax1.set_ylabel("Quality")
        ax1.set_xlabel("Iteration")
        ax1.legend(loc="lower right")

        return ax

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _moving_average(values: Sequence[float], window: int) -> List[float]:
        """Compute a simple moving average.

        Parameters
        ----------
        values : sequence of float
            Input time series.
        window : int
            Window length.

        Returns
        -------
        List[float]
            Moving averages (length ``max(0, len(values) - window + 1)``).
        """
        if len(values) < window or window <= 0:
            return []
        arr = np.asarray(values, dtype=np.float64)
        cumsum = np.cumsum(arr)
        cumsum = np.insert(cumsum, 0, 0.0)
        return ((cumsum[window:] - cumsum[:-window]) / window).tolist()

    @staticmethod
    def _relative_improvement(values: Sequence[float], window: int) -> float:
        """Compute the relative improvement rate over a window.

        Uses least-squares slope normalised by the mean value in the
        window.

        Parameters
        ----------
        values : sequence of float
            Input time series.
        window : int
            Window length.

        Returns
        -------
        float
            Relative improvement rate (positive = improving).
        """
        if len(values) < 2:
            return 0.0
        w = min(window, len(values))
        segment = np.asarray(values[-w:], dtype=np.float64)
        x = np.arange(w, dtype=np.float64)
        mean_val = np.mean(segment)
        if abs(mean_val) < 1e-15:
            mean_val = 1.0
        slope, _, _, _, _ = sp_stats.linregress(x, segment)
        return float(slope / abs(mean_val))


# ===================================================================
# ArchiveDiversity
# ===================================================================


class ArchiveDiversity:
    """Compute diversity metrics of a QD archive.

    Measures how diverse the archive entries are in descriptor space,
    including pairwise distances, coverage ratio, quality variance and
    descriptor-distribution entropy.

    Parameters
    ----------
    descriptor_dim : int
        Dimensionality of the behaviour descriptor.
    n_bins_per_dim : int
        Number of bins per descriptor dimension for entropy estimation.
    """

    def __init__(self, descriptor_dim: int = 4, n_bins_per_dim: int = 10) -> None:
        self._descriptor_dim = descriptor_dim
        self._n_bins = n_bins_per_dim

    def compute(
        self,
        archive_entries: List[Dict[str, Any]],
        total_cells: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute diversity metrics for the archive.

        Parameters
        ----------
        archive_entries : list of dict
            Each entry must contain ``"descriptor"`` (array-like of
            length *descriptor_dim*) and ``"quality"`` (float).
        total_cells : int or None
            Total number of cells in the archive grid.  If ``None``,
            estimated from the number of bins.

        Returns
        -------
        dict
            Keys: ``"mean_pairwise_distance"``,
            ``"coverage_ratio"``, ``"quality_variance"``,
            ``"descriptor_entropy"``, ``"num_entries"``.
        """
        if len(archive_entries) == 0:
            return {
                "mean_pairwise_distance": 0.0,
                "coverage_ratio": 0.0,
                "quality_variance": 0.0,
                "descriptor_entropy": 0.0,
                "num_entries": 0,
            }

        descriptors = np.array(
            [e["descriptor"] for e in archive_entries], dtype=np.float64
        )
        qualities = np.array(
            [e["quality"] for e in archive_entries], dtype=np.float64
        )

        if total_cells is None:
            total_cells = self._n_bins ** self._descriptor_dim

        mean_pw = self._mean_pairwise_distance(descriptors)
        cov = self._coverage_ratio(len(archive_entries), total_cells)
        qvar = self._quality_variance(qualities)
        ent = self._entropy(descriptors)

        return {
            "mean_pairwise_distance": mean_pw,
            "coverage_ratio": cov,
            "quality_variance": qvar,
            "descriptor_entropy": ent,
            "num_entries": len(archive_entries),
        }

    # -----------------------------------------------------------------
    # Internal metrics
    # -----------------------------------------------------------------

    def _pairwise_distances(self, descriptors: NDArray) -> NDArray:
        """Compute full pairwise Euclidean distance matrix.

        Parameters
        ----------
        descriptors : NDArray, shape (n, d)

        Returns
        -------
        NDArray, shape (n, n)
        """
        diff = descriptors[:, np.newaxis, :] - descriptors[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def _mean_pairwise_distance(self, descriptors: NDArray) -> float:
        """Mean of the upper-triangle pairwise distances."""
        n = len(descriptors)
        if n < 2:
            return 0.0
        dists = self._pairwise_distances(descriptors)
        upper = dists[np.triu_indices(n, k=1)]
        return float(np.mean(upper))

    @staticmethod
    def _coverage_ratio(cells_filled: int, total_cells: int) -> float:
        """Fraction of cells that are occupied."""
        if total_cells <= 0:
            return 0.0
        return min(float(cells_filled) / float(total_cells), 1.0)

    @staticmethod
    def _quality_variance(qualities: NDArray) -> float:
        """Variance of quality scores."""
        if len(qualities) == 0:
            return 0.0
        return float(np.var(qualities))

    def _entropy(self, descriptors: NDArray) -> float:
        """Entropy of binned descriptor distribution.

        Discretises each dimension into ``n_bins`` equal-width bins
        and computes the Shannon entropy of the multinomial histogram.

        Parameters
        ----------
        descriptors : NDArray, shape (n, d)

        Returns
        -------
        float
            Shannon entropy in nats.
        """
        n, d = descriptors.shape
        if n == 0:
            return 0.0

        mins = descriptors.min(axis=0)
        maxs = descriptors.max(axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-12] = 1.0

        bin_indices = np.floor(
            (descriptors - mins) / ranges * (self._n_bins - 1)
        ).astype(np.int64)
        bin_indices = np.clip(bin_indices, 0, self._n_bins - 1)

        keys = [tuple(row) for row in bin_indices]
        from collections import Counter

        counts = Counter(keys)
        probs = np.array(list(counts.values()), dtype=np.float64) / n
        return float(-np.sum(probs * np.log(probs + 1e-30)))
