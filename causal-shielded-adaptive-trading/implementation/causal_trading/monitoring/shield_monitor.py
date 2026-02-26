"""
Shield permissivity tracking for Causal-Shielded Adaptive Trading.

Monitors the real-time permissivity ratio of the safety shield, detects
declining permissivity trends that signal overly conservative behaviour,
logs shield violations, and tracks action rejection rates per regime.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class ShieldAlertSeverity(Enum):
    """Severity levels for shield-related alerts."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class ShieldAlert:
    """Alert generated when shield permissivity degrades or violations occur."""
    timestamp: int
    severity: ShieldAlertSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionRecord:
    """Record of a single action decision by the shield."""
    timestamp: int
    state: np.ndarray
    action: Any
    permitted: bool
    regime_id: Optional[int] = None
    reason: Optional[str] = None


class PermissivityTracker:
    """
    Tracks permissivity ratio (fraction of actions permitted) over
    a rolling window and detects declining trends.

    Parameters
    ----------
    window_size : int
        Number of recent actions to use for ratio computation.
    warning_threshold : float
        Permissivity below this triggers a WARNING.
    critical_threshold : float
        Permissivity below this triggers a CRITICAL alert.
    trend_window : int
        Window size for computing the permissivity trend slope.
    """

    def __init__(
        self,
        window_size: int = 200,
        warning_threshold: float = 0.5,
        critical_threshold: float = 0.2,
        trend_window: int = 50,
    ) -> None:
        self.window_size = window_size
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.trend_window = trend_window

        self._decisions: List[bool] = []
        self._permissivity_history: List[float] = []

    def record(self, permitted: bool) -> None:
        """Record whether an action was permitted."""
        self._decisions.append(permitted)
        window = self._decisions[-self.window_size:]
        ratio = sum(window) / len(window)
        self._permissivity_history.append(ratio)

    @property
    def current_permissivity(self) -> float:
        """Current rolling permissivity ratio."""
        if not self._permissivity_history:
            return 1.0
        return self._permissivity_history[-1]

    @property
    def total_decisions(self) -> int:
        return len(self._decisions)

    @property
    def total_permitted(self) -> int:
        return sum(self._decisions)

    @property
    def total_rejected(self) -> int:
        return len(self._decisions) - sum(self._decisions)

    def get_trend(self) -> float:
        """
        Compute the linear trend slope of permissivity over the trend window.

        Returns
        -------
        slope : float
            Negative slope indicates declining permissivity.
        """
        vals = self._permissivity_history[-self.trend_window:]
        if len(vals) < 5:
            return 0.0
        x = np.arange(len(vals), dtype=np.float64)
        slope, _, _, _, _ = sp_stats.linregress(x, vals)
        return float(slope)

    def get_severity(self) -> Optional[ShieldAlertSeverity]:
        """Return alert severity based on current permissivity."""
        p = self.current_permissivity
        if p < self.critical_threshold:
            return ShieldAlertSeverity.CRITICAL
        if p < self.warning_threshold:
            return ShieldAlertSeverity.WARNING
        return None

    def get_history(self, last_n: Optional[int] = None) -> List[float]:
        """Return permissivity ratio history."""
        if last_n is not None:
            return self._permissivity_history[-last_n:]
        return list(self._permissivity_history)


class ViolationLogger:
    """
    Logs shield violations—cases where a downstream system bypasses
    the shield's recommendation and executes a blocked action.

    Each violation is recorded with full context for audit purposes.
    """

    def __init__(self, max_log_size: int = 10000) -> None:
        self.max_log_size = max_log_size
        self._violations: List[Dict[str, Any]] = []

    def log_violation(
        self,
        timestamp: int,
        state: np.ndarray,
        action: Any,
        reason: str = "",
    ) -> None:
        """Log a shield violation event."""
        entry = {
            "timestamp": timestamp,
            "state": state.tolist() if isinstance(state, np.ndarray) else state,
            "action": action,
            "reason": reason,
        }
        self._violations.append(entry)
        if len(self._violations) > self.max_log_size:
            self._violations = self._violations[-self.max_log_size:]
        logger.warning("Shield violation at t=%d: %s", timestamp, reason)

    @property
    def total_violations(self) -> int:
        return len(self._violations)

    def get_violations(
        self,
        last_n: Optional[int] = None,
        since: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return violation records.

        Parameters
        ----------
        last_n : int, optional
            Return only the last n violations.
        since : int, optional
            Return violations since the given timestamp.
        """
        result = self._violations
        if since is not None:
            result = [v for v in result if v["timestamp"] >= since]
        if last_n is not None:
            result = result[-last_n:]
        return result

    def get_violation_rate(self, window: int = 100) -> float:
        """
        Return the violation rate over the last *window* timestamps.

        This is an approximation based on the count of violations
        whose timestamps fall within the window.
        """
        if not self._violations:
            return 0.0
        latest = self._violations[-1]["timestamp"]
        start = latest - window
        recent = [v for v in self._violations if v["timestamp"] >= start]
        return len(recent) / max(window, 1)


class ShieldMonitor:
    """
    Real-time shield permissivity and violation monitoring.

    Integrates permissivity tracking, violation logging, and per-regime
    rejection rate analysis. Generates alerts when permissivity degrades
    or unusual violation patterns emerge.

    Parameters
    ----------
    window_size : int
        Rolling window for permissivity computation.
    warning_threshold : float
        Permissivity threshold for WARNING alerts.
    critical_threshold : float
        Permissivity threshold for CRITICAL alerts.
    trend_alert_slope : float
        If permissivity trend slope is more negative than this,
        generate a declining-permissivity alert.
    """

    def __init__(
        self,
        window_size: int = 200,
        warning_threshold: float = 0.5,
        critical_threshold: float = 0.2,
        trend_alert_slope: float = -0.005,
    ) -> None:
        self.trend_alert_slope = trend_alert_slope

        self._permissivity = PermissivityTracker(
            window_size=window_size,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
        )
        self._violations = ViolationLogger()

        self._t: int = 0
        self._alerts: List[ShieldAlert] = []
        self._action_log: List[ActionRecord] = []
        self._regime_rejection_counts: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "rejected": 0}
        )
        self._current_regime_id: int = 0

    def set_regime(self, regime_id: int) -> None:
        """Update the current regime identifier."""
        self._current_regime_id = regime_id

    def log_action(
        self,
        state: np.ndarray,
        action: Any,
        permitted: bool,
        reason: Optional[str] = None,
        bypass: bool = False,
    ) -> Optional[ShieldAlert]:
        """
        Log an action decision and check for alerts.

        Parameters
        ----------
        state : np.ndarray
            Current state observation.
        action : any
            The proposed action.
        permitted : bool
            Whether the shield permitted the action.
        reason : str, optional
            Reason for rejection (if not permitted).
        bypass : bool
            If True, the downstream system executed a blocked action
            (a shield violation).

        Returns
        -------
        alert : ShieldAlert or None
        """
        state = np.asarray(state, dtype=np.float64)

        record = ActionRecord(
            timestamp=self._t,
            state=state,
            action=action,
            permitted=permitted,
            regime_id=self._current_regime_id,
            reason=reason,
        )
        self._action_log.append(record)
        self._permissivity.record(permitted)

        # Per-regime stats
        regime_stats = self._regime_rejection_counts[self._current_regime_id]
        regime_stats["total"] += 1
        if not permitted:
            regime_stats["rejected"] += 1

        # Log violation if bypass
        if bypass and not permitted:
            self._violations.log_violation(
                timestamp=self._t,
                state=state,
                action=action,
                reason=reason or "Shield bypassed",
            )

        alert = self._check_for_alerts()
        self._t += 1
        return alert

    def get_permissivity_report(self) -> Dict[str, Any]:
        """
        Return a comprehensive permissivity report.

        Returns
        -------
        report : dict
            Current permissivity, trend, per-regime rejection rates,
            total decisions, and alert status.
        """
        per_regime = {}
        for rid, stats in self._regime_rejection_counts.items():
            total = stats["total"]
            rejected = stats["rejected"]
            per_regime[rid] = {
                "total": total,
                "rejected": rejected,
                "rejection_rate": rejected / max(total, 1),
                "permissivity": 1.0 - rejected / max(total, 1),
            }

        return {
            "current_permissivity": self._permissivity.current_permissivity,
            "trend_slope": self._permissivity.get_trend(),
            "total_decisions": self._permissivity.total_decisions,
            "total_permitted": self._permissivity.total_permitted,
            "total_rejected": self._permissivity.total_rejected,
            "per_regime": per_regime,
            "total_violations": self._violations.total_violations,
            "current_regime": self._current_regime_id,
        }

    def get_violations(
        self,
        last_n: Optional[int] = None,
        since: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return shield violation records."""
        return self._violations.get_violations(last_n=last_n, since=since)

    def get_alerts(self, last_n: Optional[int] = None) -> List[ShieldAlert]:
        """Return shield monitoring alerts."""
        if last_n is not None:
            return self._alerts[-last_n:]
        return list(self._alerts)

    def get_regime_rejection_rate(self, regime_id: int) -> float:
        """Return the rejection rate for a specific regime."""
        stats = self._regime_rejection_counts.get(regime_id)
        if stats is None or stats["total"] == 0:
            return 0.0
        return stats["rejected"] / stats["total"]

    def get_action_log(
        self,
        last_n: Optional[int] = None,
        regime_id: Optional[int] = None,
    ) -> List[ActionRecord]:
        """
        Return action records, optionally filtered.

        Parameters
        ----------
        last_n : int, optional
            Limit to last n records.
        regime_id : int, optional
            Filter by regime.
        """
        result = self._action_log
        if regime_id is not None:
            result = [r for r in result if r.regime_id == regime_id]
        if last_n is not None:
            result = result[-last_n:]
        return result

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _check_for_alerts(self) -> Optional[ShieldAlert]:
        """Check permissivity and trends, generate alert if warranted."""
        # Level-based alert
        severity = self._permissivity.get_severity()
        if severity is not None:
            alert = ShieldAlert(
                timestamp=self._t,
                severity=severity,
                message=(
                    f"Shield permissivity at "
                    f"{self._permissivity.current_permissivity:.3f}"
                ),
                details={
                    "permissivity": self._permissivity.current_permissivity,
                    "regime": self._current_regime_id,
                },
            )
            # Deduplicate: don't alert every single step
            if not self._alerts or self._t - self._alerts[-1].timestamp >= 10:
                self._alerts.append(alert)
                logger.info("ShieldMonitor alert: %s", alert.message)
                return alert

        # Trend-based alert
        trend = self._permissivity.get_trend()
        if trend < self.trend_alert_slope:
            alert = ShieldAlert(
                timestamp=self._t,
                severity=ShieldAlertSeverity.WARNING,
                message=f"Declining permissivity trend: slope={trend:.5f}",
                details={
                    "trend_slope": trend,
                    "permissivity": self._permissivity.current_permissivity,
                },
            )
            if not self._alerts or self._t - self._alerts[-1].timestamp >= 20:
                self._alerts.append(alert)
                logger.info("ShieldMonitor trend alert: slope=%.5f", trend)
                return alert

        return None
