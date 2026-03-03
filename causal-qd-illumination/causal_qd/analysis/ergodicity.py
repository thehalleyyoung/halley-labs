"""Ergodicity verification for MAP-Elites archive exploration.

Tracks which cells are occupied over time and estimates whether the
search process is ergodic—i.e. whether it eventually explores a
sufficient fraction of the descriptor space.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Tuple

import numpy as np


class ErgodicityChecker:
    """Track archive coverage over time and estimate ergodicity.

    Parameters
    ----------
    total_cells : int
        Total number of cells in the archive grid.
    """

    def __init__(self, total_cells: int) -> None:
        self._total_cells = max(total_cells, 1)
        self._ever_occupied: Set[Any] = set()
        self._iterations: list[int] = []
        self._coverage_fractions: list[float] = []

    # -- recording ----------------------------------------------------------

    def record_occupied_cells(
        self, iteration: int, occupied_cells: Set[Any]
    ) -> None:
        """Record the set of occupied cells at *iteration*.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        occupied_cells : Set[Any]
            Set of cell identifiers that are currently occupied.
        """
        self._ever_occupied.update(occupied_cells)
        self._iterations.append(iteration)
        self._coverage_fractions.append(
            len(self._ever_occupied) / self._total_cells
        )

    # -- queries ------------------------------------------------------------

    def coverage_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the cumulative coverage curve.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(iterations, coverage_fraction)`` arrays.
        """
        return (
            np.array(self._iterations, dtype=np.float64),
            np.array(self._coverage_fractions, dtype=np.float64),
        )

    def estimate_mixing_time(self) -> Optional[float]:
        """Estimate the mixing time *tau* by fitting ``1 - exp(-t/tau)``.

        Uses a log-transform least-squares fit on the complement
        ``1 - coverage``.  Returns ``None`` if there are fewer than
        three data points or the fit is degenerate.

        Returns
        -------
        Optional[float]
            Estimated mixing time, or ``None``.
        """
        if len(self._iterations) < 3:
            return None

        t = np.array(self._iterations, dtype=np.float64)
        cov = np.array(self._coverage_fractions, dtype=np.float64)

        complement = 1.0 - cov
        # Only use points where complement > 0 (log is undefined at 0)
        mask = complement > 1e-12
        if mask.sum() < 2:
            return None

        t_fit = t[mask]
        log_comp = np.log(complement[mask])

        # Linear regression: log(1 - cov) ≈ -t / tau + intercept
        n = len(t_fit)
        t_mean = t_fit.mean()
        y_mean = log_comp.mean()
        ss_tt = float(np.sum((t_fit - t_mean) ** 2))
        if ss_tt < 1e-15:
            return None
        slope = float(np.sum((t_fit - t_mean) * (log_comp - y_mean))) / ss_tt

        if slope >= 0:
            # Coverage is not increasing — no valid mixing time
            return None

        tau = -1.0 / slope
        return tau

    def is_ergodic(self, coverage_threshold: float = 0.5) -> bool:
        """Return ``True`` if cumulative coverage exceeds *coverage_threshold*.

        Parameters
        ----------
        coverage_threshold : float
            Minimum fraction of cells that must have been visited at
            least once.

        Returns
        -------
        bool
        """
        if not self._coverage_fractions:
            return False
        return self._coverage_fractions[-1] >= coverage_threshold
