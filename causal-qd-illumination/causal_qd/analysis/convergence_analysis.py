"""Convergence analysis for MAP-Elites runs.

Tracks QD-score, coverage, fitness, and archive entropy over
generations and provides multiple convergence detection methods
(plateau, Mann-Kendall, relative improvement, Geweke diagnostic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from causal_qd.archive.archive_base import Archive
    from causal_qd.archive.stats import ArchiveStatsTracker
    from causal_qd.engine.map_elites import CausalMAPElites


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceSnapshot:
    """Metrics recorded at a single generation."""

    generation: int
    qd_score: float
    coverage: float
    max_fitness: float
    mean_fitness: float
    archive_entropy: float
    num_improvements: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_elites(archive: Any) -> list:
    """Extract elites from an archive (supports multiple conventions)."""
    if callable(getattr(archive, "elites", None)):
        return archive.elites()
    if hasattr(archive, "entries"):
        entries = archive.entries
        return entries if isinstance(entries, list) else list(entries)
    return []


def _get_total_cells(archive: Any) -> int:
    """Get total cell count from an archive."""
    for attr in ("total_cells", "_total_cells", "_n_cells"):
        val = getattr(archive, attr, None)
        if val is not None:
            return val() if callable(val) else int(val)
    return max(len(_get_elites(archive)), 1)


def _archive_entropy(archive: Any, n_bins: int = 30) -> float:
    """Shannon entropy of fitness distribution, normalised by log(n_filled)."""
    elites = _get_elites(archive)
    if len(elites) < 2:
        return 0.0

    qualities = np.array([e.quality for e in elites], dtype=np.float64)
    hist, _ = np.histogram(qualities, bins=n_bins)
    hist = hist[hist > 0]
    if len(hist) <= 1:
        return 0.0

    probs = hist / hist.sum()
    entropy = -float(np.sum(probs * np.log(probs)))
    max_entropy = math.log(len(elites))
    if max_entropy < 1e-12:
        return 0.0
    return entropy / max_entropy


def _mann_kendall_trend(values: List[float]) -> float:
    """Mann-Kendall S statistic normalised to [-1, 1].

    Positive values indicate an upward trend; values near zero mean
    no significant trend.
    """
    n = len(values)
    if n < 4:
        return 0.0
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = values[j] - values[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
    # Normalise by maximum possible S value: n*(n-1)/2
    max_s = n * (n - 1) / 2
    return s / max_s


# ---------------------------------------------------------------------------
# ConvergenceAnalyzer
# ---------------------------------------------------------------------------


class ConvergenceAnalyzer:
    """Track and analyse convergence of a MAP-Elites run.

    Parameters
    ----------
    window_size : int
        Number of recent generations to consider for convergence tests.
    significance_level : float
        Significance level for statistical tests (used as a threshold
        scaling factor for the Mann-Kendall and Geweke methods).
    """

    def __init__(
        self,
        window_size: int = 50,
        significance_level: float = 0.05,
    ) -> None:
        self._window = window_size
        self._alpha = significance_level
        self._snapshots: List[ConvergenceSnapshot] = []
        self._prev_elites: int = 0

    # -- recording ----------------------------------------------------------

    def record(self, archive: Any, generation: int) -> ConvergenceSnapshot:
        """Record a convergence snapshot from *archive*.

        Parameters
        ----------
        archive :
            Any archive object that exposes ``elites()`` / ``entries``,
            ``coverage()``, and ``qd_score()``.
        generation : int
            Current generation number.

        Returns
        -------
        ConvergenceSnapshot
        """
        elites = _get_elites(archive)
        n_elites = len(elites)

        if n_elites == 0:
            snap = ConvergenceSnapshot(
                generation=generation,
                qd_score=0.0,
                coverage=0.0,
                max_fitness=0.0,
                mean_fitness=0.0,
                archive_entropy=0.0,
                num_improvements=0,
            )
        else:
            qualities = [e.quality for e in elites]
            total_cells = _get_total_cells(archive)

            # num_improvements: difference in elite count since last snapshot
            num_improvements = max(n_elites - self._prev_elites, 0)

            snap = ConvergenceSnapshot(
                generation=generation,
                qd_score=sum(qualities),
                coverage=n_elites / max(total_cells, 1),
                max_fitness=max(qualities),
                mean_fitness=float(np.mean(qualities)),
                archive_entropy=_archive_entropy(archive),
                num_improvements=num_improvements,
            )

        self._prev_elites = n_elites
        self._snapshots.append(snap)
        return snap

    # -- history accessors --------------------------------------------------

    def qd_score_history(self) -> List[float]:
        """Return QD-score at each recorded generation."""
        return [s.qd_score for s in self._snapshots]

    def coverage_history(self) -> List[float]:
        """Return coverage at each recorded generation."""
        return [s.coverage for s in self._snapshots]

    # -- convergence detection ----------------------------------------------

    def has_converged(self, method: str = "plateau") -> bool:
        """Check whether the run has converged.

        Parameters
        ----------
        method : str
            ``'plateau'`` — QD-score unchanged (within threshold) over window.
            ``'mann_kendall'`` — Mann-Kendall trend test on recent QD-scores.
            ``'relative'`` — relative improvement below threshold.
            ``'geweke'`` — Geweke diagnostic comparing first/last portions.

        Returns
        -------
        bool
        """
        if len(self._snapshots) < self._window:
            return False

        if method == "plateau":
            return self._converged_plateau()
        elif method == "mann_kendall":
            return self._converged_mann_kendall()
        elif method == "relative":
            return self._converged_relative()
        elif method == "geweke":
            return self._converged_geweke()
        else:
            raise ValueError(f"Unknown convergence method: {method!r}")

    def _converged_plateau(self) -> bool:
        """QD-score hasn't improved by > threshold in window."""
        recent = [s.qd_score for s in self._snapshots[-self._window:]]
        if not recent:
            return False
        qd_range = max(recent) - min(recent)
        scale = max(abs(recent[0]), 1e-12)
        return (qd_range / scale) < self._alpha

    def _converged_mann_kendall(self) -> bool:
        """Mann-Kendall trend test: no significant upward trend."""
        recent = [s.qd_score for s in self._snapshots[-self._window:]]
        tau = _mann_kendall_trend(recent)
        # Converged if the trend strength is below significance level
        return abs(tau) < self._alpha

    def _converged_relative(self) -> bool:
        """Relative improvement below threshold."""
        recent = [s.qd_score for s in self._snapshots[-self._window:]]
        start = recent[0]
        end = recent[-1]
        rel = abs(end - start) / max(abs(start), 1e-12)
        return rel < self._alpha

    def _converged_geweke(self) -> bool:
        """Geweke diagnostic: compare mean of first 10% vs last 50%."""
        scores = [s.qd_score for s in self._snapshots[-self._window:]]
        n = len(scores)
        n_first = max(int(0.1 * n), 1)
        n_last = max(int(0.5 * n), 1)
        first = scores[:n_first]
        last = scores[-n_last:]

        mean_first = float(np.mean(first))
        mean_last = float(np.mean(last))
        var_first = float(np.var(first)) / max(len(first), 1)
        var_last = float(np.var(last)) / max(len(last), 1)

        denom = math.sqrt(var_first + var_last) if (var_first + var_last) > 0 else 1e-12
        z = abs(mean_last - mean_first) / denom
        # Use ~1.96 for 0.05 significance; scale with alpha
        critical = 1.96 * (0.05 / max(self._alpha, 1e-12))
        return z < critical

    # -- derived metrics ----------------------------------------------------

    def convergence_rate(self) -> float:
        """Estimated rate of QD-score improvement per generation.

        Returns
        -------
        float
            Slope of QD-score over recorded generations.
        """
        if len(self._snapshots) < 2:
            return 0.0
        gens = np.array([s.generation for s in self._snapshots], dtype=np.float64)
        scores = np.array([s.qd_score for s in self._snapshots], dtype=np.float64)
        # Simple linear regression slope
        n = len(gens)
        x_mean = gens.mean()
        y_mean = scores.mean()
        ss_xy = float(np.sum((gens - x_mean) * (scores - y_mean)))
        ss_xx = float(np.sum((gens - x_mean) ** 2))
        if ss_xx < 1e-15:
            return 0.0
        return ss_xy / ss_xx

    def expected_remaining_generations(self, target_coverage: float) -> int:
        """Extrapolate how many more generations to reach *target_coverage*.

        Uses linear extrapolation of the recent coverage trend.

        Parameters
        ----------
        target_coverage : float
            Desired coverage in ``[0, 1]``.

        Returns
        -------
        int
            Estimated remaining generations (0 if already reached,
            ``-1`` if coverage is not increasing).
        """
        if not self._snapshots:
            return -1
        current = self._snapshots[-1].coverage
        if current >= target_coverage:
            return 0

        if len(self._snapshots) < 2:
            return -1

        # Use last window_size snapshots for trend estimation
        recent = self._snapshots[-self._window:]
        cov_start = recent[0].coverage
        cov_end = recent[-1].coverage
        gen_span = recent[-1].generation - recent[0].generation
        if gen_span <= 0 or (cov_end - cov_start) <= 1e-12:
            return -1

        rate = (cov_end - cov_start) / gen_span
        remaining_cov = target_coverage - current
        return max(int(math.ceil(remaining_cov / rate)), 0)

    # -- callback -----------------------------------------------------------

    def as_callback(self) -> Callable:
        """Return a callback compatible with :class:`CausalMAPElites`.

        The callback signature is
        ``(engine, generation, archive, stats_tracker) -> None``.
        """

        def _callback(
            engine: Any,
            generation: int,
            archive: Any,
            stats_tracker: Any,
        ) -> None:
            self.record(archive, generation)

        return _callback

    # -- plotting -----------------------------------------------------------

    def plot_convergence(self) -> Any:
        """Plot QD-score and coverage convergence curves.

        Returns the matplotlib Figure, or ``None`` if matplotlib is
        unavailable.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            return None

        gens = [s.generation for s in self._snapshots]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot(gens, self.qd_score_history(), label="QD-score")
        ax1.set_ylabel("QD-score")
        ax1.legend()
        ax1.set_title("Convergence Analysis")

        ax2.plot(gens, self.coverage_history(), label="Coverage", color="tab:orange")
        ax2.set_ylabel("Coverage")
        ax2.set_xlabel("Generation")
        ax2.legend()

        fig.tight_layout()
        return fig

    # -- summary ------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary of the convergence state.

        Returns
        -------
        Dict[str, Any]
        """
        if not self._snapshots:
            return {"n_snapshots": 0}

        latest = self._snapshots[-1]
        result: Dict[str, Any] = {
            "n_snapshots": len(self._snapshots),
            "latest_generation": latest.generation,
            "qd_score": latest.qd_score,
            "coverage": latest.coverage,
            "max_fitness": latest.max_fitness,
            "mean_fitness": latest.mean_fitness,
            "archive_entropy": latest.archive_entropy,
            "convergence_rate": self.convergence_rate(),
        }

        # Add convergence flags if we have enough data
        if len(self._snapshots) >= self._window:
            result["converged_plateau"] = self.has_converged("plateau")
            result["converged_mann_kendall"] = self.has_converged("mann_kendall")
            result["converged_relative"] = self.has_converged("relative")
            result["converged_geweke"] = self.has_converged("geweke")

        return result
