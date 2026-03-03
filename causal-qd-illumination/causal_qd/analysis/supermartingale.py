"""Supermartingale convergence tracking for MAP-Elites archives.

Tracks per-cell supermartingale residuals M_t(c) = max(0, best(c) - q_t(c))
to provide empirical convergence diagnostics.  Since an elitist archive
only accepts quality improvements, the running best is monotone
non-decreasing and M_t is monotone non-increasing—a discrete
supermartingale that converges to zero.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


class SupermartingaleTracker:
    """Empirical supermartingale convergence observer.

    Parameters
    ----------
    epsilon : float
        Convergence threshold.  A cell is considered converged when
        its latest residual M_t falls below *epsilon*.
    """

    def __init__(self, epsilon: float = 1e-3) -> None:
        self._epsilon = epsilon
        self._running_best: Dict[Any, float] = {}
        self._residuals: Dict[Any, List[float]] = defaultdict(list)

    # -- recording ----------------------------------------------------------

    def record(
        self, iteration: int, cell_qualities: Dict[Any, float]
    ) -> None:
        """Record supermartingale residuals for each cell.

        Parameters
        ----------
        iteration : int
            Current iteration number (unused internally but kept for
            interface symmetry with other trackers).
        cell_qualities : Dict[Any, float]
            Mapping from cell identifier to the current quality stored
            in that cell.
        """
        for cell, quality in cell_qualities.items():
            best = self._running_best.get(cell)
            if best is None or quality > best:
                self._running_best[cell] = quality
                best = quality
            m_t = max(0.0, best - quality)
            self._residuals[cell].append(m_t)

    # -- diagnostics --------------------------------------------------------

    def convergence_diagnostic(self) -> Dict[str, Any]:
        """Return a summary of convergence across all tracked cells.

        Returns
        -------
        Dict[str, Any]
            ``converged_fraction`` — fraction of cells whose latest
            residual is below *epsilon*.
            ``mean_residual`` — mean of the latest residual across cells.
            ``per_cell`` — mapping from cell to the full list of M_t values.
        """
        if not self._residuals:
            return {
                "converged_fraction": 0.0,
                "mean_residual": 0.0,
                "per_cell": {},
            }

        latest = np.array(
            [vals[-1] for vals in self._residuals.values()], dtype=np.float64
        )
        converged = float(np.mean(latest < self._epsilon))
        mean_res = float(np.mean(latest))
        return {
            "converged_fraction": converged,
            "mean_residual": mean_res,
            "per_cell": dict(self._residuals),
        }

    def is_converged(self, cell: Any) -> bool:
        """Return ``True`` if *cell*'s latest residual is below epsilon."""
        vals = self._residuals.get(cell)
        if not vals:
            return False
        return vals[-1] < self._epsilon

    def all_converged(self) -> bool:
        """Return ``True`` if every tracked cell has converged."""
        if not self._residuals:
            return False
        return all(
            vals[-1] < self._epsilon for vals in self._residuals.values()
        )
