"""
Restart strategies for CDCL-based robustness radius search.

Implements Luby-sequence restarts, geometric restarts, Glucose-style
LBD-based restarts, and an adaptive strategy that switches between them
based on recent conflict-analysis quality.
"""

from __future__ import annotations

import enum
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------

class RestartPolicy(enum.Enum):
    """Available restart strategies."""

    LUBY = "luby"
    """Luby-sequence restarts (optimal for Las-Vegas algorithms)."""

    GEOMETRIC = "geometric"
    """Geometrically increasing intervals."""

    GLUCOSE = "glucose"
    """Glucose / LBD-based: restart when recent LBDs spike above average."""

    ADAPTIVE = "adaptive"
    """Dynamically switch between Luby and Glucose based on search quality."""

    NONE = "none"
    """No restarts."""


# ---------------------------------------------------------------------------
# Luby sequence
# ---------------------------------------------------------------------------

def luby_sequence(index: int) -> int:
    """Return the *index*-th term of the Luby sequence (0-based).

    The Luby sequence is: 1 1 2 1 1 2 4 1 1 2 1 1 2 4 8 ...

    Parameters
    ----------
    index : int
        0-based index into the sequence.

    Returns
    -------
    int
        Luby value at the given index.
    """
    # Standard iterative Luby computation.
    i = index + 1  # 1-based
    # Find k, seq such that i fits in the pattern
    k = 1
    p = 1
    while p < i:
        k += 1
        p = 2 * p + 1
    # Now p = 2^k - 1 >= i
    while p != i:
        p = (p - 1) >> 1
        if i > p:
            i -= p
    return (p + 1) >> 1


class LubyRestart:
    """Luby-sequence restart scheduler.

    Parameters
    ----------
    base_interval : int
        Base number of conflicts per unit of the Luby sequence.
    """

    def __init__(self, base_interval: int = 100) -> None:
        self._base = base_interval
        self._index: int = 0
        self._conflicts_until: int = self._base * luby_sequence(0)

    def on_conflict(self) -> bool:
        """Notify the scheduler of a new conflict.

        Returns
        -------
        bool
            ``True`` if a restart should be triggered.
        """
        self._conflicts_until -= 1
        if self._conflicts_until <= 0:
            self._index += 1
            self._conflicts_until = self._base * luby_sequence(self._index)
            return True
        return False

    def reset(self) -> None:
        """Reset the sequence (e.g. after a phase change)."""
        self._index = 0
        self._conflicts_until = self._base * luby_sequence(0)

    @property
    def next_restart_in(self) -> int:
        """Conflicts remaining until the next restart."""
        return self._conflicts_until


# ---------------------------------------------------------------------------
# Geometric restarts
# ---------------------------------------------------------------------------

class GeometricRestart:
    """Geometrically increasing restart intervals.

    Parameters
    ----------
    base_interval : int
        Initial conflict budget.
    multiplier : float
        Factor by which the interval grows after each restart.
    max_interval : int
        Upper cap on the interval.
    """

    def __init__(
        self,
        base_interval: int = 100,
        multiplier: float = 1.5,
        max_interval: int = 100_000,
    ) -> None:
        self._base = base_interval
        self._mult = multiplier
        self._max = max_interval
        self._current_interval: float = float(base_interval)
        self._remaining: int = base_interval

    def on_conflict(self) -> bool:
        """Notify of a conflict; return ``True`` to restart."""
        self._remaining -= 1
        if self._remaining <= 0:
            self._current_interval = min(
                self._current_interval * self._mult, self._max
            )
            self._remaining = int(self._current_interval)
            return True
        return False

    def reset(self) -> None:
        """Reset to base interval."""
        self._current_interval = float(self._base)
        self._remaining = self._base

    @property
    def next_restart_in(self) -> int:
        """Conflicts until next restart."""
        return self._remaining


# ---------------------------------------------------------------------------
# Glucose-style (LBD-based) restarts
# ---------------------------------------------------------------------------

class GlucoseRestart:
    """Glucose-style restart strategy based on Literal Block Distance (LBD).

    Restarts when the recent average LBD is significantly higher than the
    global running average, indicating the solver is exploring a low-quality
    region of the search space.

    Parameters
    ----------
    window_size : int
        Number of recent LBDs to average.
    margin : float
        Restart when ``recent_avg > margin * global_avg``.
    min_conflicts : int
        Minimum conflicts between restarts.
    """

    def __init__(
        self,
        window_size: int = 50,
        margin: float = 1.4,
        min_conflicts: int = 50,
    ) -> None:
        self._window_size = window_size
        self._margin = margin
        self._min_conflicts = min_conflicts

        self._recent_lbds: deque[int] = deque(maxlen=window_size)
        self._global_sum: float = 0.0
        self._global_count: int = 0
        self._conflicts_since_restart: int = 0

    def on_conflict(self, lbd: int) -> bool:
        """Register a conflict with its clause LBD.

        Parameters
        ----------
        lbd : int
            LBD of the learned clause.

        Returns
        -------
        bool
            ``True`` if a restart should be triggered.
        """
        self._recent_lbds.append(lbd)
        self._global_sum += lbd
        self._global_count += 1
        self._conflicts_since_restart += 1

        if self._conflicts_since_restart < self._min_conflicts:
            return False

        if len(self._recent_lbds) < self._window_size:
            return False

        recent_avg = sum(self._recent_lbds) / len(self._recent_lbds)
        global_avg = self._global_sum / self._global_count if self._global_count else 1.0

        if recent_avg > self._margin * global_avg:
            self._conflicts_since_restart = 0
            return True

        return False

    def reset(self) -> None:
        """Reset the recent window (keep global stats)."""
        self._recent_lbds.clear()
        self._conflicts_since_restart = 0

    @property
    def global_avg_lbd(self) -> float:
        """Running average LBD across all conflicts."""
        return self._global_sum / self._global_count if self._global_count else 0.0

    @property
    def recent_avg_lbd(self) -> float:
        """Average LBD in the recent window."""
        if not self._recent_lbds:
            return 0.0
        return sum(self._recent_lbds) / len(self._recent_lbds)


# ---------------------------------------------------------------------------
# Adaptive restart
# ---------------------------------------------------------------------------

class AdaptiveRestart:
    """Adaptive restart that switches between Luby and Glucose strategies.

    Uses Glucose-style restarts when learned clauses have high LBD (poor
    quality) and falls back to Luby restarts when clause quality is
    consistently good.

    Parameters
    ----------
    luby_base : int
        Base interval for the Luby component.
    glucose_window : int
        Window size for the Glucose component.
    glucose_margin : float
        Glucose restart margin.
    switch_threshold : int
        Number of consecutive Glucose-triggered restarts before switching
        to Luby.
    """

    def __init__(
        self,
        luby_base: int = 100,
        glucose_window: int = 50,
        glucose_margin: float = 1.4,
        switch_threshold: int = 5,
    ) -> None:
        self._luby = LubyRestart(luby_base)
        self._glucose = GlucoseRestart(glucose_window, glucose_margin)
        self._switch_threshold = switch_threshold

        self._consecutive_glucose: int = 0
        self._using_luby: bool = False
        self._total_restarts: int = 0

    def on_conflict(self, lbd: int) -> bool:
        """Register a conflict.

        Parameters
        ----------
        lbd : int
            LBD of the learned clause.

        Returns
        -------
        bool
            ``True`` if a restart should be triggered.
        """
        glucose_wants = self._glucose.on_conflict(lbd)
        luby_wants = self._luby.on_conflict()

        if self._using_luby:
            if luby_wants:
                self._total_restarts += 1
                # Check if we should switch back to Glucose
                if self._glucose.recent_avg_lbd < 0.9 * self._glucose.global_avg_lbd:
                    self._using_luby = False
                    self._consecutive_glucose = 0
                    logger.debug("Adaptive restart: switching to Glucose mode")
                return True
            return False

        # Glucose mode
        if glucose_wants:
            self._consecutive_glucose += 1
            self._total_restarts += 1
            if self._consecutive_glucose >= self._switch_threshold:
                self._using_luby = True
                logger.debug("Adaptive restart: switching to Luby mode")
            return True

        return False

    def reset(self) -> None:
        """Reset both inner strategies."""
        self._luby.reset()
        self._glucose.reset()
        self._consecutive_glucose = 0
        self._using_luby = False

    @property
    def current_strategy(self) -> str:
        """Name of the currently active inner strategy."""
        return "luby" if self._using_luby else "glucose"

    @property
    def total_restarts(self) -> int:
        """Total number of restarts triggered."""
        return self._total_restarts


# ---------------------------------------------------------------------------
# Unified scheduler
# ---------------------------------------------------------------------------

class RestartScheduler:
    """Unified restart scheduler with strategy selection.

    Parameters
    ----------
    policy : RestartPolicy
        Which restart strategy to use.
    luby_base : int
        Base interval for Luby / adaptive strategies.
    geo_base : int
        Base interval for geometric strategy.
    geo_mult : float
        Multiplier for geometric strategy.
    glucose_window : int
        LBD window size for Glucose / adaptive strategies.
    glucose_margin : float
        Restart margin for Glucose / adaptive strategies.
    """

    def __init__(
        self,
        policy: RestartPolicy = RestartPolicy.GLUCOSE,
        luby_base: int = 100,
        geo_base: int = 100,
        geo_mult: float = 1.5,
        glucose_window: int = 50,
        glucose_margin: float = 1.4,
    ) -> None:
        self._policy = policy
        self._n_restarts: int = 0

        if policy == RestartPolicy.LUBY:
            self._inner: LubyRestart | GeometricRestart | GlucoseRestart | AdaptiveRestart | None = (
                LubyRestart(luby_base)
            )
        elif policy == RestartPolicy.GEOMETRIC:
            self._inner = GeometricRestart(geo_base, geo_mult)
        elif policy == RestartPolicy.GLUCOSE:
            self._inner = GlucoseRestart(glucose_window, glucose_margin)
        elif policy == RestartPolicy.ADAPTIVE:
            self._inner = AdaptiveRestart(luby_base, glucose_window, glucose_margin)
        else:
            self._inner = None

    def on_conflict(self, lbd: int = 0) -> bool:
        """Notify the scheduler of a conflict.

        Parameters
        ----------
        lbd : int
            LBD of the learned clause (used by Glucose / adaptive).

        Returns
        -------
        bool
            ``True`` if a restart should be triggered.
        """
        if self._inner is None:
            return False

        if isinstance(self._inner, (GlucoseRestart, AdaptiveRestart)):
            should = self._inner.on_conflict(lbd)
        else:
            should = self._inner.on_conflict()

        if should:
            self._n_restarts += 1
        return should

    def reset(self) -> None:
        """Reset the inner strategy."""
        if self._inner is not None:
            self._inner.reset()

    @property
    def n_restarts(self) -> int:
        """Total number of restarts triggered so far."""
        return self._n_restarts

    @property
    def policy(self) -> RestartPolicy:
        """Currently active policy."""
        return self._policy

    def __repr__(self) -> str:  # pragma: no cover
        return f"RestartScheduler(policy={self._policy.value}, restarts={self._n_restarts})"
