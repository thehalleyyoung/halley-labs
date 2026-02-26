"""
Shield liveness theorem implementation.

Ensures that the synthesized shield permits a non-trivial set of actions,
preventing the shield from being overly conservative and effectively
shutting down trading. Computes permissivity ratios, lower bounds,
and generates liveness certificates.
"""

from __future__ import annotations

import logging
import time as time_module
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import binom

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PermissivityAlert:
    """
    Alert raised when permissivity drops below threshold.

    Attributes
    ----------
    level : AlertLevel
        Severity of the alert.
    state : int
        State where the alert was triggered.
    permissivity : float
        Permissivity ratio at the state.
    threshold : float
        Threshold that was violated.
    timestamp : float
        Time of the alert.
    message : str
        Human-readable alert message.
    spec_breakdown : dict
        Per-spec permissivity at the alerting state.
    """
    level: AlertLevel
    state: int
    permissivity: float
    threshold: float
    timestamp: float
    message: str
    spec_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class LivenessCertificate:
    """
    Certificate that the shield preserves liveness.

    Guarantees that the shield permits at least a fraction rho_min
    of actions in every reachable state.

    Attributes
    ----------
    min_permissivity : float
        Minimum permissivity ratio across all states.
    mean_permissivity : float
        Average permissivity ratio.
    threshold : float
        Liveness threshold.
    is_live : bool
        Whether the shield satisfies liveness.
    worst_state : int
        State with lowest permissivity.
    n_dead_states : int
        Number of states with permissivity below threshold.
    confidence : float
        Statistical confidence (if applicable).
    """
    min_permissivity: float
    mean_permissivity: float
    threshold: float
    is_live: bool
    worst_state: int
    n_dead_states: int
    confidence: float = 1.0

    def summary(self) -> str:
        """Human-readable summary."""
        status = "LIVE" if self.is_live else "DEAD"
        return (
            f"LivenessCertificate [{status}]:\n"
            f"  Min permissivity: {self.min_permissivity:.4f}\n"
            f"  Mean permissivity: {self.mean_permissivity:.4f}\n"
            f"  Threshold: {self.threshold:.4f}\n"
            f"  Worst state: {self.worst_state}\n"
            f"  Dead states: {self.n_dead_states}\n"
            f"  Confidence: {self.confidence:.4f}"
        )


class ShieldLiveness:
    """
    Shield liveness analysis and certification.

    Computes permissivity ratios for each state and verifies that
    the shield maintains liveness — i.e., permits enough actions
    to allow meaningful trading behavior.

    The permissivity ratio at state s is:
        rho(s) = |{a : shield permits a in s}| / |A|

    The shield is live at threshold tau iff:
        min_s rho(s) >= tau

    Parameters
    ----------
    n_states : int
        Number of states.
    n_actions : int
        Number of actions.
    threshold : float
        Liveness threshold. Default 0.1 (10% of actions must be permitted).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        threshold: float = 0.1,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.threshold = threshold

        # Permissivity storage
        self._permissivity: np.ndarray = np.ones(n_states)
        self._per_spec_permissivity: Dict[str, np.ndarray] = {}
        self._permitted_counts: np.ndarray = np.full(n_states, n_actions)

        # History
        self._permissivity_history: List[np.ndarray] = []
        self._timestamp_history: List[float] = []

        # Alerts
        self._alerts: List[PermissivityAlert] = []
        self._alert_callbacks: List[Callable[[PermissivityAlert], None]] = []

        # State visitation for weighted analysis
        self._state_visits: np.ndarray = np.zeros(n_states)

    def update_from_shield(
        self,
        permitted_table: np.ndarray,
        per_spec_tables: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Update permissivity from a shield's permitted action table.

        Parameters
        ----------
        permitted_table : np.ndarray
            Boolean array of shape (n_states, n_actions).
        per_spec_tables : dict, optional
            Per-specification safety tables for decomposition.
        """
        assert permitted_table.shape == (self.n_states, self.n_actions)

        counts = np.sum(permitted_table.astype(int), axis=1)
        self._permitted_counts = counts
        self._permissivity = counts / self.n_actions

        # Store per-spec data
        if per_spec_tables:
            for name, table in per_spec_tables.items():
                # Count actions that this spec alone would permit (at threshold 0.95)
                spec_permitted = np.sum(table >= 0.95, axis=1)
                self._per_spec_permissivity[name] = spec_permitted / self.n_actions

        # Record history
        self._permissivity_history.append(self._permissivity.copy())
        self._timestamp_history.append(time_module.time())

        # Check for alerts
        self._check_alerts()

    def update_state_visits(self, states: np.ndarray) -> None:
        """
        Update state visitation counts for weighted permissivity.

        Parameters
        ----------
        states : np.ndarray
            Array of visited state indices.
        """
        for s in states:
            if 0 <= s < self.n_states:
                self._state_visits[s] += 1

    def compute_permissivity(self, state: int) -> float:
        """
        Get permissivity ratio for a specific state.

        Parameters
        ----------
        state : int
            State index.

        Returns
        -------
        float
            Permissivity ratio rho(s) in [0, 1].
        """
        if state < 0 or state >= self.n_states:
            return 0.0
        return float(self._permissivity[state])

    def get_min_permissivity(self) -> Tuple[float, int]:
        """
        Get the minimum permissivity ratio and its state.

        Returns
        -------
        min_perm : float
            Minimum permissivity ratio.
        worst_state : int
            State index with minimum permissivity.
        """
        worst = int(np.argmin(self._permissivity))
        return float(self._permissivity[worst]), worst

    def get_mean_permissivity(self, weighted: bool = False) -> float:
        """
        Get mean permissivity, optionally weighted by state visitation.

        Parameters
        ----------
        weighted : bool
            If True, weight by state visitation frequency.

        Returns
        -------
        float
            Mean permissivity ratio.
        """
        if weighted and np.sum(self._state_visits) > 0:
            weights = self._state_visits / np.sum(self._state_visits)
            return float(np.dot(weights, self._permissivity))
        return float(np.mean(self._permissivity))

    def get_permissivity_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get histogram of permissivity ratios across states.

        Returns
        -------
        bin_edges : np.ndarray
            Histogram bin edges.
        counts : np.ndarray
            Number of states in each bin.
        """
        counts, bin_edges = np.histogram(
            self._permissivity, bins=20, range=(0, 1)
        )
        return bin_edges, counts

    def check_liveness(self, threshold: Optional[float] = None) -> LivenessCertificate:
        """
        Check if the shield satisfies liveness at the given threshold.

        Parameters
        ----------
        threshold : float, optional
            Liveness threshold. Defaults to self.threshold.

        Returns
        -------
        LivenessCertificate
            Certificate of liveness (or failure thereof).
        """
        if threshold is None:
            threshold = self.threshold

        min_perm, worst_state = self.get_min_permissivity()
        mean_perm = self.get_mean_permissivity()
        n_dead = int(np.sum(self._permissivity < threshold))
        is_live = min_perm >= threshold

        cert = LivenessCertificate(
            min_permissivity=min_perm,
            mean_permissivity=mean_perm,
            threshold=threshold,
            is_live=is_live,
            worst_state=worst_state,
            n_dead_states=n_dead,
        )

        if not is_live:
            logger.warning(
                "Liveness check FAILED: min_perm=%.4f < threshold=%.4f "
                "(worst_state=%d, dead_states=%d)",
                min_perm, threshold, worst_state, n_dead,
            )
        else:
            logger.info(
                "Liveness check PASSED: min_perm=%.4f >= threshold=%.4f",
                min_perm, threshold,
            )

        return cert

    def compute_expected_permissivity_bound(
        self,
        n_samples: int,
        delta: float = 0.05,
    ) -> float:
        """
        Compute a lower bound on the expected permissivity ratio.

        Uses Hoeffding's inequality:
            P(E[rho] >= hat{rho} - epsilon) >= 1 - delta

        where epsilon = sqrt(ln(1/delta) / (2n)).

        Parameters
        ----------
        n_samples : int
            Number of state samples used.
        delta : float
            Confidence parameter.

        Returns
        -------
        float
            Lower bound on expected permissivity.
        """
        mean_perm = self.get_mean_permissivity()
        epsilon = np.sqrt(np.log(1.0 / delta) / (2 * n_samples))
        lower_bound = max(0.0, mean_perm - epsilon)
        return float(lower_bound)

    def compute_permissivity_ci(
        self,
        state: int,
        n_episodes: int,
        delta: float = 0.05,
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for permissivity at a state.

        Uses the Clopper-Pearson (exact binomial) interval.

        Parameters
        ----------
        state : int
            State index.
        n_episodes : int
            Number of episodes observed.
        delta : float
            Confidence parameter for CI.

        Returns
        -------
        lower, upper : float
            Lower and upper CI bounds.
        """
        k = int(self._permitted_counts[state])
        n = self.n_actions

        # Clopper-Pearson interval
        alpha = delta
        if k == 0:
            lower = 0.0
        else:
            lower = float(beta_dist.ppf(alpha / 2, k, n - k + 1))

        if k == n:
            upper = 1.0
        else:
            upper = float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k))

        return lower, upper

    def _check_alerts(self) -> None:
        """Check for permissivity degradation and raise alerts."""
        for s in range(self.n_states):
            perm = self._permissivity[s]

            if perm < self.threshold:
                # Critical: below liveness threshold
                spec_breakdown = {}
                for name, spec_perm in self._per_spec_permissivity.items():
                    spec_breakdown[name] = float(spec_perm[s])

                alert = PermissivityAlert(
                    level=AlertLevel.CRITICAL,
                    state=s,
                    permissivity=float(perm),
                    threshold=self.threshold,
                    timestamp=time_module.time(),
                    message=(
                        f"State {s}: permissivity {perm:.4f} below "
                        f"threshold {self.threshold:.4f}"
                    ),
                    spec_breakdown=spec_breakdown,
                )
                self._alerts.append(alert)
                for callback in self._alert_callbacks:
                    callback(alert)

            elif perm < 2 * self.threshold:
                # Warning: approaching threshold
                alert = PermissivityAlert(
                    level=AlertLevel.WARNING,
                    state=s,
                    permissivity=float(perm),
                    threshold=self.threshold,
                    timestamp=time_module.time(),
                    message=(
                        f"State {s}: permissivity {perm:.4f} approaching "
                        f"threshold {self.threshold:.4f}"
                    ),
                )
                self._alerts.append(alert)

    def register_alert_callback(
        self, callback: Callable[[PermissivityAlert], None]
    ) -> None:
        """Register a callback for permissivity alerts."""
        self._alert_callbacks.append(callback)

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        since: Optional[float] = None,
    ) -> List[PermissivityAlert]:
        """
        Get alerts, optionally filtered by level and time.

        Parameters
        ----------
        level : AlertLevel, optional
            Filter to this severity level.
        since : float, optional
            Only return alerts after this timestamp.

        Returns
        -------
        list of PermissivityAlert
        """
        alerts = self._alerts
        if level is not None:
            alerts = [a for a in alerts if a.level == level]
        if since is not None:
            alerts = [a for a in alerts if a.timestamp >= since]
        return alerts

    def clear_alerts(self) -> None:
        """Clear all stored alerts."""
        self._alerts.clear()


class DegradationMonitor:
    """
    Monitor for shield permissivity degradation over time.

    Tracks trends in permissivity and raises alerts when sustained
    degradation is detected (e.g., regime changes making the shield
    overly conservative).

    Parameters
    ----------
    window_size : int
        Number of historical snapshots to consider.
    degradation_threshold : float
        Rate of permissivity decline (per snapshot) that triggers an alert.
    min_permissivity : float
        Absolute minimum permissivity before critical alert.
    """

    def __init__(
        self,
        window_size: int = 20,
        degradation_threshold: float = 0.01,
        min_permissivity: float = 0.1,
    ) -> None:
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.min_permissivity = min_permissivity

        self._mean_history: List[float] = []
        self._min_history: List[float] = []
        self._timestamps: List[float] = []
        self._alerts: List[PermissivityAlert] = []

    def record(self, mean_permissivity: float, min_permissivity: float) -> None:
        """
        Record a permissivity snapshot.

        Parameters
        ----------
        mean_permissivity : float
            Mean permissivity at this time.
        min_permissivity : float
            Min permissivity at this time.
        """
        self._mean_history.append(mean_permissivity)
        self._min_history.append(min_permissivity)
        self._timestamps.append(time_module.time())

        self._check_degradation()

    def _check_degradation(self) -> None:
        """Check for degradation trends."""
        if len(self._mean_history) < self.window_size:
            return

        recent = np.array(self._mean_history[-self.window_size :])

        # Linear trend
        x = np.arange(len(recent), dtype=float)
        coeffs = np.polyfit(x, recent, 1)
        slope = coeffs[0]

        if slope < -self.degradation_threshold:
            alert = PermissivityAlert(
                level=AlertLevel.WARNING,
                state=-1,
                permissivity=float(recent[-1]),
                threshold=self.min_permissivity,
                timestamp=time_module.time(),
                message=(
                    f"Permissivity degradation detected: slope={slope:.6f} "
                    f"over last {self.window_size} snapshots"
                ),
            )
            self._alerts.append(alert)
            logger.warning(alert.message)

        # Absolute check
        if self._min_history[-1] < self.min_permissivity:
            alert = PermissivityAlert(
                level=AlertLevel.CRITICAL,
                state=-1,
                permissivity=self._min_history[-1],
                threshold=self.min_permissivity,
                timestamp=time_module.time(),
                message=(
                    f"Min permissivity {self._min_history[-1]:.4f} below "
                    f"absolute threshold {self.min_permissivity:.4f}"
                ),
            )
            self._alerts.append(alert)
            logger.critical(alert.message)

    def get_trend(self) -> Tuple[float, float]:
        """
        Get the current permissivity trend.

        Returns
        -------
        slope : float
            Slope of mean permissivity trend (per snapshot).
        r_squared : float
            R^2 of linear fit.
        """
        if len(self._mean_history) < 3:
            return 0.0, 0.0

        recent = np.array(
            self._mean_history[-min(self.window_size, len(self._mean_history)):]
        )
        x = np.arange(len(recent), dtype=float)
        coeffs = np.polyfit(x, recent, 1)
        slope = coeffs[0]

        # R^2
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((recent - y_pred) ** 2)
        ss_tot = np.sum((recent - np.mean(recent)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-15)

        return float(slope), float(r_squared)

    def get_alerts(self) -> List[PermissivityAlert]:
        """Get all degradation alerts."""
        return list(self._alerts)

    def forecast_permissivity(self, steps_ahead: int = 10) -> np.ndarray:
        """
        Forecast future permissivity based on trend.

        Parameters
        ----------
        steps_ahead : int
            Number of future steps to forecast.

        Returns
        -------
        np.ndarray
            Forecasted mean permissivity values.
        """
        if len(self._mean_history) < 3:
            last = self._mean_history[-1] if self._mean_history else 1.0
            return np.full(steps_ahead, last)

        recent = np.array(
            self._mean_history[-min(self.window_size, len(self._mean_history)):]
        )
        x = np.arange(len(recent), dtype=float)
        coeffs = np.polyfit(x, recent, 1)

        future_x = np.arange(len(recent), len(recent) + steps_ahead, dtype=float)
        forecast = np.polyval(coeffs, future_x)

        # Clip to [0, 1]
        forecast = np.clip(forecast, 0.0, 1.0)
        return forecast

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        slope, r2 = self.get_trend()
        return {
            "n_snapshots": len(self._mean_history),
            "current_mean": self._mean_history[-1] if self._mean_history else None,
            "current_min": self._min_history[-1] if self._min_history else None,
            "trend_slope": slope,
            "trend_r_squared": r2,
            "n_alerts": len(self._alerts),
        }
