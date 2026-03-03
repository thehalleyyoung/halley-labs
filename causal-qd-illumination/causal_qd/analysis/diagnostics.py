"""Diagnostic tools for MAP-Elites archive health monitoring.

Provides diagnostics for archive coverage and quality, operator
effectiveness tracking, and score distribution analysis.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import AdjacencyMatrix, DataMatrix, QualityScore


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ArchiveHealthReport:
    """Health report for a MAP-Elites archive."""
    coverage: float
    qd_score: float
    mean_quality: float
    median_quality: float
    std_quality: float
    best_quality: float
    worst_quality: float
    quality_iqr: float
    n_elites: int
    stagnation_detected: bool
    stagnation_generations: int
    under_explored_fraction: float
    over_explored_fraction: float
    diversity_index: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorReport:
    """Report on operator effectiveness."""
    operator_name: str
    n_applications: int
    n_improvements: int
    success_rate: float
    mean_quality_gain: float
    total_quality_gain: float


@dataclass
class ScoreReport:
    """Report on score distribution."""
    mean: float
    std: float
    median: float
    skewness: float
    kurtosis: float
    n_plateaus: int
    n_local_optima: int
    entropy: float


# ---------------------------------------------------------------------------
# ArchiveDiagnostics
# ---------------------------------------------------------------------------


class ArchiveDiagnostics:
    """Check archive health and identify issues.

    Parameters
    ----------
    stagnation_window : int
        Number of recent generations to check for stagnation.
        Default ``50``.
    stagnation_threshold : float
        Minimum QD-score improvement ratio to avoid stagnation
        detection.  Default ``0.001``.
    """

    def __init__(
        self,
        stagnation_window: int = 50,
        stagnation_threshold: float = 0.001,
    ) -> None:
        self._window = stagnation_window
        self._threshold = stagnation_threshold
        self._qd_history: List[float] = []
        self._improvement_history: List[int] = []

    def record_iteration(
        self,
        qd_score: float,
        n_improvements: int,
    ) -> None:
        """Record metrics from one iteration.

        Parameters
        ----------
        qd_score : float
            Current QD-score.
        n_improvements : int
            Number of archive improvements in this iteration.
        """
        self._qd_history.append(qd_score)
        self._improvement_history.append(n_improvements)

    def health_check(
        self,
        qualities: List[QualityScore],
        descriptors: Optional[List[npt.NDArray[np.float64]]] = None,
        n_cells: int = 100,
    ) -> ArchiveHealthReport:
        """Run a comprehensive health check on the archive.

        Parameters
        ----------
        qualities : List[QualityScore]
            Quality scores of all elites.
        descriptors : List[ndarray] | None
            Descriptor vectors of all elites.
        n_cells : int
            Total number of archive cells.

        Returns
        -------
        ArchiveHealthReport
        """
        n_elites = len(qualities)
        if n_elites == 0:
            return ArchiveHealthReport(
                coverage=0.0, qd_score=0.0,
                mean_quality=0.0, median_quality=0.0,
                std_quality=0.0, best_quality=0.0,
                worst_quality=0.0, quality_iqr=0.0,
                n_elites=0, stagnation_detected=False,
                stagnation_generations=0,
                under_explored_fraction=1.0,
                over_explored_fraction=0.0,
                diversity_index=0.0,
            )

        q_arr = np.array(qualities, dtype=np.float64)
        coverage = n_elites / max(n_cells, 1)
        qd_score = float(q_arr.sum())

        q25, q50, q75 = np.percentile(q_arr, [25, 50, 75])

        # Stagnation detection
        stagnation = self._detect_stagnation()
        stag_gens = self._stagnation_duration()

        # Diversity index (Simpson's diversity using descriptor bins)
        diversity = 0.0
        under_explored = 0.0
        over_explored = 0.0

        if descriptors is not None and len(descriptors) > 1:
            diversity = self._simpson_diversity(descriptors)
            under_explored, over_explored = self._exploration_balance(
                descriptors, n_cells
            )

        return ArchiveHealthReport(
            coverage=coverage,
            qd_score=qd_score,
            mean_quality=float(q_arr.mean()),
            median_quality=float(q50),
            std_quality=float(q_arr.std()),
            best_quality=float(q_arr.max()),
            worst_quality=float(q_arr.min()),
            quality_iqr=float(q75 - q25),
            n_elites=n_elites,
            stagnation_detected=stagnation,
            stagnation_generations=stag_gens,
            under_explored_fraction=under_explored,
            over_explored_fraction=over_explored,
            diversity_index=diversity,
        )

    def identify_under_explored_regions(
        self,
        descriptors: List[npt.NDArray[np.float64]],
        bounds_low: npt.NDArray[np.float64],
        bounds_high: npt.NDArray[np.float64],
        grid_resolution: int = 10,
    ) -> List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Identify under-explored regions in descriptor space.

        Divides the descriptor space into a grid and returns cells
        with no elites.

        Parameters
        ----------
        descriptors : List[ndarray]
            Descriptor vectors of current elites.
        bounds_low, bounds_high : ndarray
            Descriptor space bounds.
        grid_resolution : int
            Grid resolution per dimension.

        Returns
        -------
        List[Tuple[ndarray, ndarray]]
            List of (cell_low, cell_high) bounds for empty cells.
        """
        d = len(bounds_low)
        if not descriptors:
            return []

        desc_arr = np.array(descriptors, dtype=np.float64)
        cell_sizes = (bounds_high - bounds_low) / grid_resolution

        # Count occupancy per cell
        occupied = set()
        for desc in desc_arr:
            cell = tuple(
                min(int((desc[k] - bounds_low[k]) / max(cell_sizes[k], 1e-10)), grid_resolution - 1)
                for k in range(d)
            )
            occupied.add(cell)

        # Find unoccupied cells
        empty_regions: List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
        total_cells = grid_resolution ** d
        if total_cells > 10000:
            return empty_regions  # Too many cells to enumerate

        def _enumerate_cells(dims: int, prefix: Tuple = ()) -> None:
            if dims == 0:
                if prefix not in occupied:
                    cell_low = bounds_low + np.array(prefix) * cell_sizes
                    cell_high = cell_low + cell_sizes
                    empty_regions.append((cell_low, cell_high))
                return
            for i in range(grid_resolution):
                _enumerate_cells(dims - 1, prefix + (i,))

        _enumerate_cells(d)
        return empty_regions

    def _detect_stagnation(self) -> bool:
        """Check if the archive has stagnated."""
        if len(self._qd_history) < self._window:
            return False

        recent = self._qd_history[-self._window:]
        if recent[0] == 0:
            return sum(self._improvement_history[-self._window:]) == 0

        improvement_ratio = (recent[-1] - recent[0]) / abs(recent[0])
        return improvement_ratio < self._threshold

    def _stagnation_duration(self) -> int:
        """Count consecutive generations with no improvement."""
        count = 0
        for imp in reversed(self._improvement_history):
            if imp == 0:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _simpson_diversity(
        descriptors: List[npt.NDArray[np.float64]],
    ) -> float:
        """Compute Simpson's diversity index from descriptors."""
        if len(descriptors) < 2:
            return 0.0

        desc_arr = np.array(descriptors)
        n = len(desc_arr)

        # Pairwise distance-based diversity
        total_dist = 0.0
        count = 0
        for i in range(min(n, 100)):
            for j in range(i + 1, min(n, 100)):
                total_dist += float(np.linalg.norm(desc_arr[i] - desc_arr[j]))
                count += 1

        if count == 0:
            return 0.0

        mean_dist = total_dist / count
        # Normalize by maximum possible distance
        max_dist = float(np.linalg.norm(desc_arr.max(axis=0) - desc_arr.min(axis=0)))
        return mean_dist / max(max_dist, 1e-10)

    @staticmethod
    def _exploration_balance(
        descriptors: List[npt.NDArray[np.float64]],
        n_cells: int,
    ) -> Tuple[float, float]:
        """Compute fraction of under/over-explored regions."""
        n = len(descriptors)
        if n == 0:
            return 1.0, 0.0

        # Simple heuristic: bin descriptors
        desc_arr = np.array(descriptors)
        d = desc_arr.shape[1]

        # Use 5 bins per dimension
        bins_per_dim = min(5, int(n_cells ** (1.0 / max(d, 1))))
        bins_per_dim = max(bins_per_dim, 2)

        total_bins = bins_per_dim ** d
        if total_bins > 10000:
            return 0.0, 0.0

        # Count per-bin occupancy
        mins = desc_arr.min(axis=0)
        ranges = desc_arr.max(axis=0) - mins
        ranges = np.maximum(ranges, 1e-10)

        bin_counts: Dict[Tuple, int] = defaultdict(int)
        for desc in desc_arr:
            cell = tuple(
                min(int((desc[k] - mins[k]) / ranges[k] * bins_per_dim), bins_per_dim - 1)
                for k in range(d)
            )
            bin_counts[cell] += 1

        n_occupied = len(bin_counts)
        under = 1.0 - n_occupied / total_bins

        # Over-explored: bins with > 2x average occupancy
        avg = n / total_bins
        over_count = sum(1 for c in bin_counts.values() if c > 2 * avg)
        over = over_count / total_bins

        return under, over


# ---------------------------------------------------------------------------
# OperatorDiagnostics
# ---------------------------------------------------------------------------


class OperatorDiagnostics:
    """Track and analyze operator effectiveness.

    Records each operator application and whether it resulted in
    an archive improvement, then provides summary statistics.
    """

    def __init__(self) -> None:
        self._records: Dict[str, List[Tuple[bool, float]]] = defaultdict(list)

    def record(
        self,
        operator_name: str,
        was_improvement: bool,
        quality_gain: float = 0.0,
    ) -> None:
        """Record one operator application.

        Parameters
        ----------
        operator_name : str
            Name of the operator.
        was_improvement : bool
            Whether it led to an archive improvement.
        quality_gain : float
            Quality gain (if improvement).
        """
        self._records[operator_name].append((was_improvement, quality_gain))

    def report(self) -> Dict[str, OperatorReport]:
        """Generate reports for all tracked operators.

        Returns
        -------
        Dict[str, OperatorReport]
        """
        reports: Dict[str, OperatorReport] = {}
        for name, records in self._records.items():
            n_apps = len(records)
            n_imp = sum(1 for imp, _ in records if imp)
            gains = [g for imp, g in records if imp and g > 0]
            reports[name] = OperatorReport(
                operator_name=name,
                n_applications=n_apps,
                n_improvements=n_imp,
                success_rate=n_imp / max(n_apps, 1),
                mean_quality_gain=float(np.mean(gains)) if gains else 0.0,
                total_quality_gain=float(sum(gains)),
            )
        return reports

    def best_operator(self) -> Optional[str]:
        """Return the operator with the highest success rate.

        Returns
        -------
        str | None
        """
        reports = self.report()
        if not reports:
            return None
        return max(reports.values(), key=lambda r: r.success_rate).operator_name

    def worst_operator(self) -> Optional[str]:
        """Return the operator with the lowest success rate.

        Returns
        -------
        str | None
        """
        reports = self.report()
        if not reports:
            return None
        return min(reports.values(), key=lambda r: r.success_rate).operator_name

    def reset(self) -> None:
        """Clear all records."""
        self._records.clear()


# ---------------------------------------------------------------------------
# ScoreDiagnostics
# ---------------------------------------------------------------------------


class ScoreDiagnostics:
    """Analyze score distributions to detect pathologies.

    Detects plateaus, identifies potential local optima, and
    provides distributional statistics.
    """

    def analyze_distribution(
        self, scores: List[QualityScore]
    ) -> ScoreReport:
        """Analyze the distribution of scores.

        Parameters
        ----------
        scores : List[QualityScore]
            Quality scores from the archive.

        Returns
        -------
        ScoreReport
        """
        if not scores:
            return ScoreReport(
                mean=0.0, std=0.0, median=0.0,
                skewness=0.0, kurtosis=0.0,
                n_plateaus=0, n_local_optima=0,
                entropy=0.0,
            )

        arr = np.array(scores, dtype=np.float64)
        n = len(arr)

        mean = float(arr.mean())
        std = float(arr.std())
        median = float(np.median(arr))

        # Skewness
        if std > 1e-15:
            skewness = float(np.mean(((arr - mean) / std) ** 3))
            kurtosis = float(np.mean(((arr - mean) / std) ** 4) - 3.0)
        else:
            skewness = 0.0
            kurtosis = 0.0

        # Detect plateaus (clusters of similar scores)
        n_plateaus = self._count_plateaus(arr)

        # Detect local optima (scores that are highest in their neighborhood)
        n_local_optima = self._count_local_optima(arr)

        # Entropy of score distribution
        entropy = self._score_entropy(arr)

        return ScoreReport(
            mean=mean, std=std, median=median,
            skewness=skewness, kurtosis=kurtosis,
            n_plateaus=n_plateaus,
            n_local_optima=n_local_optima,
            entropy=entropy,
        )

    def detect_score_plateaus(
        self,
        scores: List[QualityScore],
        tolerance: float = 0.01,
    ) -> List[Tuple[float, int]]:
        """Find score plateaus (many solutions with similar scores).

        Parameters
        ----------
        scores : List[QualityScore]
            Scores to analyze.
        tolerance : float
            Maximum relative difference for scores to be in same plateau.

        Returns
        -------
        List[Tuple[float, int]]
            List of (plateau_center, count) pairs.
        """
        if not scores:
            return []

        arr = np.sort(scores)
        plateaus: List[Tuple[float, int]] = []
        current_start = 0

        for i in range(1, len(arr)):
            ref = abs(arr[current_start])
            diff = abs(arr[i] - arr[current_start])
            if diff > tolerance * max(ref, 1.0):
                count = i - current_start
                if count >= 3:
                    center = float(np.mean(arr[current_start:i]))
                    plateaus.append((center, count))
                current_start = i

        # Check last group
        count = len(arr) - current_start
        if count >= 3:
            center = float(np.mean(arr[current_start:]))
            plateaus.append((center, count))

        return plateaus

    @staticmethod
    def _count_plateaus(
        scores: npt.NDArray[np.float64], n_bins: int = 20
    ) -> int:
        """Count number of score plateaus using histogram analysis."""
        if len(scores) < 3:
            return 0

        hist, _ = np.histogram(scores, bins=n_bins)
        threshold = len(scores) / n_bins * 2  # > 2x expected

        return int(np.sum(hist > threshold))

    @staticmethod
    def _count_local_optima(
        scores: npt.NDArray[np.float64],
    ) -> int:
        """Estimate number of local optima from sorted score gaps."""
        if len(scores) < 3:
            return 0

        sorted_scores = np.sort(scores)
        gaps = np.diff(sorted_scores)

        if len(gaps) == 0:
            return 0

        mean_gap = float(gaps.mean())
        # Large gaps suggest distinct local optima
        large_gaps = np.sum(gaps > 3 * mean_gap)
        return int(large_gaps) + 1

    @staticmethod
    def _score_entropy(
        scores: npt.NDArray[np.float64], n_bins: int = 20
    ) -> float:
        """Compute entropy of the score distribution."""
        if len(scores) < 2:
            return 0.0

        hist, _ = np.histogram(scores, bins=n_bins)
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        entropy = -float(np.sum(probs * np.log(probs)))
        max_entropy = math.log(n_bins)
        return entropy / max(max_entropy, 1e-10)
