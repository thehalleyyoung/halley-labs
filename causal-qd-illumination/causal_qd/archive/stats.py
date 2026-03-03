"""Archive statistics and convergence tracking over generations.

Provides :class:`ArchiveStats` (per-snapshot metrics) and
:class:`ArchiveStatsTracker` (longitudinal statistics across the
entire MAP-Elites run, including convergence detection).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from causal_qd.archive.archive_base import Archive


# ------------------------------------------------------------------
# Single-snapshot statistics
# ------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ArchiveStats:
    """Snapshot of archive performance metrics at a single point in time.

    Attributes
    ----------
    coverage : float
        Fraction of cells occupied.
    qd_score : float
        Sum of all elite quality scores.
    best_quality : float
        Quality of the single best elite.
    mean_quality : float
        Mean quality across all elites.
    num_elites : int
        Number of occupied cells.
    diversity : float
        Mean pairwise Euclidean distance between elite descriptors.
    """

    coverage: float
    qd_score: float
    best_quality: float
    mean_quality: float
    num_elites: int
    diversity: float

    @classmethod
    def from_archive(cls, archive: "Archive") -> "ArchiveStats":
        """Compute all statistics from an archive instance.

        Parameters
        ----------
        archive : Archive
            The archive to summarise.

        Returns
        -------
        ArchiveStats
            Frozen dataclass with computed metrics.
        """
        all_elites = archive.elites()
        num_elites = len(all_elites)

        if num_elites == 0:
            return cls(
                coverage=0.0,
                qd_score=0.0,
                best_quality=float("-inf"),
                mean_quality=0.0,
                num_elites=0,
                diversity=0.0,
            )

        qualities = [e.quality for e in all_elites]
        best_quality = max(qualities)
        mean_quality = float(np.mean(qualities))

        # Diversity: mean pairwise Euclidean distance of descriptors
        if num_elites < 2:
            diversity = 0.0
        else:
            descs = np.array([e.descriptor for e in all_elites])
            diff = descs[:, np.newaxis, :] - descs[np.newaxis, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=-1))
            n = len(descs)
            diversity = float(dists.sum() / (n * (n - 1)))

        return cls(
            coverage=archive.coverage(),
            qd_score=archive.qd_score(),
            best_quality=best_quality,
            mean_quality=mean_quality,
            num_elites=num_elites,
            diversity=diversity,
        )


# ------------------------------------------------------------------
# Longitudinal statistics tracker
# ------------------------------------------------------------------

@dataclasses.dataclass
class GenerationRecord:
    """Metrics recorded for a single generation.

    Attributes
    ----------
    generation : int
    coverage : float
    qd_score : float
    best_quality : float
    mean_quality : float
    num_elites : int
    diversity : float
    improvements : int
        Number of successful insertions in this generation.
    """

    generation: int
    coverage: float
    qd_score: float
    best_quality: float
    mean_quality: float
    num_elites: int
    diversity: float
    improvements: int = 0


class ArchiveStatsTracker:
    """Track archive evolution over multiple generations.

    Records per-generation :class:`ArchiveStats` snapshots and provides
    derived metrics such as convergence detection, improvement rates,
    and plateau detection.

    Parameters
    ----------
    window_size : int
        Rolling window size used for plateau / convergence detection.
    convergence_threshold : float
        Relative improvement threshold below which the archive is
        considered converged.
    """

    def __init__(
        self,
        window_size: int = 50,
        convergence_threshold: float = 1e-4,
    ) -> None:
        self._window_size = window_size
        self._convergence_threshold = convergence_threshold
        self._records: List[GenerationRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        generation: int,
        archive: "Archive",
        improvements: int = 0,
    ) -> GenerationRecord:
        """Record statistics for *generation*.

        Parameters
        ----------
        generation : int
            Current generation / iteration number.
        archive : Archive
            The archive to snapshot.
        improvements : int
            Number of successful insertions in this generation.

        Returns
        -------
        GenerationRecord
        """
        stats = ArchiveStats.from_archive(archive)
        record = GenerationRecord(
            generation=generation,
            coverage=stats.coverage,
            qd_score=stats.qd_score,
            best_quality=stats.best_quality,
            mean_quality=stats.mean_quality,
            num_elites=stats.num_elites,
            diversity=stats.diversity,
            improvements=improvements,
        )
        self._records.append(record)
        return record

    # ------------------------------------------------------------------
    # History accessors
    # ------------------------------------------------------------------

    @property
    def records(self) -> List[GenerationRecord]:
        """All recorded generation records."""
        return list(self._records)

    @property
    def coverage_history(self) -> List[float]:
        """Coverage at each recorded generation."""
        return [r.coverage for r in self._records]

    @property
    def qd_score_history(self) -> List[float]:
        """QD-score at each recorded generation."""
        return [r.qd_score for r in self._records]

    @property
    def best_quality_history(self) -> List[float]:
        """Best quality at each recorded generation."""
        return [r.best_quality for r in self._records]

    @property
    def mean_quality_history(self) -> List[float]:
        """Mean quality at each recorded generation."""
        return [r.mean_quality for r in self._records]

    @property
    def diversity_history(self) -> List[float]:
        """Diversity at each recorded generation."""
        return [r.diversity for r in self._records]

    @property
    def num_elites_history(self) -> List[int]:
        """Number of elites at each recorded generation."""
        return [r.num_elites for r in self._records]

    @property
    def improvement_history(self) -> List[int]:
        """Number of improvements at each recorded generation."""
        return [r.improvements for r in self._records]

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def improvement_rate(self, window: Optional[int] = None) -> float:
        """Compute the mean improvements per generation over a window.

        Parameters
        ----------
        window : int, optional
            Number of recent generations to average over.  Defaults to
            ``self._window_size``.

        Returns
        -------
        float
            Mean improvements per generation (0.0 if no records).
        """
        w = window or self._window_size
        recent = self._records[-w:]
        if not recent:
            return 0.0
        return float(np.mean([r.improvements for r in recent]))

    def qd_score_improvement_rate(self, window: Optional[int] = None) -> float:
        """Relative QD-score improvement over the last *window* generations.

        Returns
        -------
        float
            ``(qd_now - qd_then) / max(|qd_then|, 1e-12)``.
        """
        w = window or self._window_size
        if len(self._records) < 2:
            return float("inf")
        recent = self._records[-w:]
        qd_start = recent[0].qd_score
        qd_end = recent[-1].qd_score
        return (qd_end - qd_start) / max(abs(qd_start), 1e-12)

    def is_converged(self) -> bool:
        """Detect whether the archive has converged.

        Convergence is declared when the relative QD-score improvement
        over the last ``window_size`` generations falls below
        ``convergence_threshold``.

        Returns
        -------
        bool
        """
        if len(self._records) < self._window_size:
            return False
        rate = self.qd_score_improvement_rate(self._window_size)
        return abs(rate) < self._convergence_threshold

    def plateau_length(self) -> int:
        """Number of consecutive recent generations without improvement.

        Returns
        -------
        int
            0 if the most recent generation had improvements.
        """
        length = 0
        for r in reversed(self._records):
            if r.improvements == 0:
                length += 1
            else:
                break
        return length

    def best_generation(self) -> Optional[int]:
        """Generation at which the best quality was achieved.

        Returns
        -------
        int or None
            ``None`` if no records exist.
        """
        if not self._records:
            return None
        best_rec = max(self._records, key=lambda r: r.best_quality)
        return best_rec.generation

    def coverage_growth_rate(self, window: Optional[int] = None) -> float:
        """Coverage increase per generation over the last *window* generations.

        Returns
        -------
        float
        """
        w = window or self._window_size
        if len(self._records) < 2:
            return float("inf")
        recent = self._records[-w:]
        cov_start = recent[0].coverage
        cov_end = recent[-1].coverage
        n_gens = recent[-1].generation - recent[0].generation
        if n_gens == 0:
            return 0.0
        return (cov_end - cov_start) / n_gens

    def summary(self) -> Dict[str, float]:
        """Return a summary dictionary of current tracking state.

        Returns
        -------
        Dict[str, float]
        """
        if not self._records:
            return {"n_generations": 0}
        latest = self._records[-1]
        return {
            "n_generations": float(len(self._records)),
            "coverage": latest.coverage,
            "qd_score": latest.qd_score,
            "best_quality": latest.best_quality,
            "mean_quality": latest.mean_quality,
            "diversity": latest.diversity,
            "num_elites": float(latest.num_elites),
            "improvement_rate": self.improvement_rate(),
            "qd_improvement_rate": self.qd_score_improvement_rate(),
            "plateau_length": float(self.plateau_length()),
            "converged": float(self.is_converged()),
        }

    def clear(self) -> None:
        """Remove all recorded history."""
        self._records.clear()
