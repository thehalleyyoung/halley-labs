"""
Permissivity ratio tracking and analysis.

Provides detailed tracking, decomposition, and forecasting of permissivity
ratios across states, regimes, and time. Identifies which safety specifications
are most restrictive and generates reports for monitoring.
"""

from __future__ import annotations

import logging
import time as time_module
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import linregress

logger = logging.getLogger(__name__)


@dataclass
class PermissivitySnapshot:
    """A snapshot of permissivity at a point in time."""
    timestamp: float
    per_state: np.ndarray
    mean: float
    minimum: float
    maximum: float
    std: float
    regime: Optional[str] = None
    n_dead_states: int = 0


@dataclass
class PermissivityReport:
    """
    Comprehensive permissivity report.

    Attributes
    ----------
    mean_permissivity : float
        Average permissivity across states.
    min_permissivity : float
        Minimum permissivity.
    max_permissivity : float
        Maximum permissivity.
    std_permissivity : float
        Standard deviation of permissivity.
    n_dead_states : int
        States with permissivity below threshold.
    most_restrictive_spec : str
        Specification causing most restriction.
    spec_restrictiveness : dict
        Restrictiveness score per spec.
    trend_slope : float
        Trend in mean permissivity over time.
    regime_breakdown : dict
        Permissivity statistics per regime.
    recommendations : list
        Suggested actions to improve permissivity.
    """
    mean_permissivity: float
    min_permissivity: float
    max_permissivity: float
    std_permissivity: float
    n_dead_states: int
    most_restrictive_spec: str
    spec_restrictiveness: Dict[str, float]
    trend_slope: float
    regime_breakdown: Dict[str, Dict[str, float]]
    recommendations: List[str]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=== Permissivity Report ===",
            f"Mean: {self.mean_permissivity:.4f}",
            f"Min:  {self.min_permissivity:.4f}",
            f"Max:  {self.max_permissivity:.4f}",
            f"Std:  {self.std_permissivity:.4f}",
            f"Dead states: {self.n_dead_states}",
            f"Most restrictive: {self.most_restrictive_spec}",
            "",
            "Spec restrictiveness:",
        ]
        for name, score in sorted(
            self.spec_restrictiveness.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {name}: {score:.4f}")

        lines.append(f"\nTrend slope: {self.trend_slope:.6f}")

        if self.regime_breakdown:
            lines.append("\nRegime breakdown:")
            for regime, stats in self.regime_breakdown.items():
                lines.append(
                    f"  {regime}: mean={stats['mean']:.4f}, "
                    f"min={stats['min']:.4f}"
                )

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class PermissivityDecomposition:
    """
    Decompose permissivity by safety specification.

    For each state, identifies which specifications are the binding
    constraints on permissivity, enabling targeted relaxation.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_actions : int
        Number of actions.
    """

    def __init__(self, n_states: int, n_actions: int) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self._spec_tables: Dict[str, np.ndarray] = {}
        self._overall_permissivity: Optional[np.ndarray] = None

    def update(
        self,
        overall_permitted: np.ndarray,
        per_spec_safety: Dict[str, np.ndarray],
        threshold: float = 0.95,
    ) -> None:
        """
        Update decomposition data.

        Parameters
        ----------
        overall_permitted : np.ndarray
            Boolean (n_states, n_actions) overall permitted actions.
        per_spec_safety : dict
            Per-spec safety probability tables.
        threshold : float
            Safety threshold for individual spec permissivity.
        """
        self._overall_permissivity = (
            np.sum(overall_permitted, axis=1) / self.n_actions
        )

        for name, safety_table in per_spec_safety.items():
            spec_permitted = (safety_table >= threshold).astype(float)
            self._spec_tables[name] = (
                np.sum(spec_permitted, axis=1) / self.n_actions
            )

    def get_binding_spec(self, state: int) -> Optional[str]:
        """
        Get the binding (most restrictive) spec at a state.

        Parameters
        ----------
        state : int
            State index.

        Returns
        -------
        str or None
            Name of the most restrictive spec.
        """
        if not self._spec_tables:
            return None

        min_perm = 1.0
        binding = None
        for name, perms in self._spec_tables.items():
            if perms[state] < min_perm:
                min_perm = perms[state]
                binding = name
        return binding

    def get_restrictiveness_scores(self) -> Dict[str, float]:
        """
        Compute a restrictiveness score for each spec.

        The score is 1 - mean(spec_permissivity), so higher means
        more restrictive. Scores are in [0, 1].

        Returns
        -------
        dict
            Mapping spec name -> restrictiveness score.
        """
        scores = {}
        for name, perms in self._spec_tables.items():
            scores[name] = float(1.0 - np.mean(perms))
        return scores

    def get_marginal_impact(self) -> Dict[str, float]:
        """
        Compute the marginal impact of each spec on overall permissivity.

        For each spec, estimates how much permissivity would increase
        if that spec were removed (leave-one-out analysis).

        Returns
        -------
        dict
            Mapping spec name -> marginal permissivity increase.
        """
        if self._overall_permissivity is None or not self._spec_tables:
            return {}

        overall_mean = float(np.mean(self._overall_permissivity))
        impacts = {}

        for name in self._spec_tables:
            # Permissivity without this spec = min of all other specs
            other_perms = [
                v for k, v in self._spec_tables.items() if k != name
            ]
            if other_perms:
                without_spec = other_perms[0].copy()
                for p in other_perms[1:]:
                    without_spec = np.minimum(without_spec, p)
                impacts[name] = float(np.mean(without_spec) - overall_mean)
            else:
                impacts[name] = float(1.0 - overall_mean)

        return impacts

    def get_state_decomposition(self, state: int) -> Dict[str, float]:
        """
        Get per-spec permissivity at a specific state.

        Parameters
        ----------
        state : int
            State index.

        Returns
        -------
        dict
            Mapping spec name -> permissivity at state.
        """
        return {
            name: float(perms[state])
            for name, perms in self._spec_tables.items()
        }

    def get_dead_state_causes(self, threshold: float = 0.1) -> Dict[str, int]:
        """
        Count how many dead states each spec is responsible for.

        A state is "dead" if permissivity < threshold. A spec is
        "responsible" if it's the binding constraint at that state.

        Parameters
        ----------
        threshold : float
            Dead state threshold.

        Returns
        -------
        dict
            Mapping spec name -> number of dead states caused.
        """
        causes: Dict[str, int] = defaultdict(int)
        if self._overall_permissivity is None:
            return dict(causes)

        for s in range(self.n_states):
            if self._overall_permissivity[s] < threshold:
                binding = self.get_binding_spec(s)
                if binding:
                    causes[binding] += 1

        return dict(causes)


class PermissivityForecaster:
    """
    Forecast future permissivity based on historical data.

    Uses linear regression, exponential smoothing, and change-point
    detection to predict permissivity evolution.

    Parameters
    ----------
    max_history : int
        Maximum number of historical snapshots to retain.
    smoothing_alpha : float
        Exponential smoothing parameter (0 = no smoothing, 1 = full).
    """

    def __init__(
        self,
        max_history: int = 1000,
        smoothing_alpha: float = 0.1,
    ) -> None:
        self.max_history = max_history
        self.smoothing_alpha = smoothing_alpha

        self._mean_history: List[float] = []
        self._min_history: List[float] = []
        self._timestamps: List[float] = []
        self._smoothed_mean: Optional[float] = None

    def record(self, mean_perm: float, min_perm: float) -> None:
        """
        Record a permissivity observation.

        Parameters
        ----------
        mean_perm : float
            Mean permissivity.
        min_perm : float
            Minimum permissivity.
        """
        self._mean_history.append(mean_perm)
        self._min_history.append(min_perm)
        self._timestamps.append(time_module.time())

        # Exponential smoothing
        if self._smoothed_mean is None:
            self._smoothed_mean = mean_perm
        else:
            self._smoothed_mean = (
                self.smoothing_alpha * mean_perm
                + (1 - self.smoothing_alpha) * self._smoothed_mean
            )

        # Trim history
        if len(self._mean_history) > self.max_history:
            excess = len(self._mean_history) - self.max_history
            self._mean_history = self._mean_history[excess:]
            self._min_history = self._min_history[excess:]
            self._timestamps = self._timestamps[excess:]

    def forecast_linear(self, steps_ahead: int = 10) -> np.ndarray:
        """
        Linear forecast of mean permissivity.

        Parameters
        ----------
        steps_ahead : int
            Number of future steps.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        if len(self._mean_history) < 3:
            last = self._mean_history[-1] if self._mean_history else 1.0
            return np.full(steps_ahead, last)

        y = np.array(self._mean_history)
        x = np.arange(len(y), dtype=float)

        result = linregress(x, y)
        future_x = np.arange(len(y), len(y) + steps_ahead, dtype=float)
        forecast = result.slope * future_x + result.intercept

        return np.clip(forecast, 0.0, 1.0)

    def forecast_exponential(self, steps_ahead: int = 10) -> np.ndarray:
        """
        Exponential smoothing forecast.

        Uses Holt's double exponential smoothing for trend.

        Parameters
        ----------
        steps_ahead : int
            Number of future steps.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        if len(self._mean_history) < 2:
            last = self._mean_history[-1] if self._mean_history else 1.0
            return np.full(steps_ahead, last)

        alpha = self.smoothing_alpha
        beta = 0.05  # trend smoothing

        y = np.array(self._mean_history)

        # Initialize
        level = y[0]
        trend = y[1] - y[0]

        for i in range(1, len(y)):
            new_level = alpha * y[i] + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            level = new_level
            trend = new_trend

        forecast = np.array([level + (i + 1) * trend for i in range(steps_ahead)])
        return np.clip(forecast, 0.0, 1.0)

    def detect_change_points(self, sensitivity: float = 2.0) -> List[int]:
        """
        Detect change points in permissivity using CUSUM.

        Parameters
        ----------
        sensitivity : float
            Detection threshold in standard deviations.

        Returns
        -------
        list of int
            Indices of detected change points.
        """
        if len(self._mean_history) < 10:
            return []

        y = np.array(self._mean_history)
        mean = np.mean(y)
        std = np.std(y)
        if std < 1e-10:
            return []

        # CUSUM
        cusum_pos = np.zeros(len(y))
        cusum_neg = np.zeros(len(y))
        threshold = sensitivity * std
        change_points = []

        for i in range(1, len(y)):
            diff = y[i] - mean
            cusum_pos[i] = max(0, cusum_pos[i - 1] + diff - std / 2)
            cusum_neg[i] = min(0, cusum_neg[i - 1] + diff + std / 2)

            if cusum_pos[i] > threshold or cusum_neg[i] < -threshold:
                change_points.append(i)
                cusum_pos[i] = 0
                cusum_neg[i] = 0

        return change_points

    def get_smoothed_series(self, window: int = 5) -> np.ndarray:
        """
        Get smoothed permissivity time series.

        Uses Savitzky-Golay filter for smooth derivative-preserving output.

        Parameters
        ----------
        window : int
            Smoothing window size (must be odd).

        Returns
        -------
        np.ndarray
            Smoothed mean permissivity series.
        """
        if len(self._mean_history) < window:
            return np.array(self._mean_history)

        if window % 2 == 0:
            window += 1

        y = np.array(self._mean_history)
        polyorder = min(3, window - 1)
        return savgol_filter(y, window, polyorder)

    def time_to_threshold(self, threshold: float = 0.1) -> Optional[int]:
        """
        Estimate how many steps until permissivity drops below threshold.

        Returns None if the trend is non-negative or current value is
        already below threshold.

        Parameters
        ----------
        threshold : float
            Permissivity threshold.

        Returns
        -------
        int or None
            Estimated steps to threshold crossing.
        """
        if not self._mean_history:
            return None

        current = self._mean_history[-1]
        if current <= threshold:
            return 0

        if len(self._mean_history) < 3:
            return None

        y = np.array(self._mean_history)
        x = np.arange(len(y), dtype=float)
        result = linregress(x, y)

        if result.slope >= 0:
            return None  # Not declining

        # Time = (threshold - current_level) / slope
        current_level = result.slope * x[-1] + result.intercept
        steps = (threshold - current_level) / result.slope

        return max(0, int(np.ceil(steps)))

    def summary(self) -> Dict[str, Any]:
        """Return forecaster summary."""
        if not self._mean_history:
            return {"n_observations": 0}

        return {
            "n_observations": len(self._mean_history),
            "current_mean": self._mean_history[-1],
            "current_min": self._min_history[-1],
            "smoothed_mean": self._smoothed_mean,
            "change_points": len(self.detect_change_points()),
        }


class PermissivityTracker:
    """
    Comprehensive permissivity tracking across states, regimes, and time.

    Integrates PermissivityDecomposition and PermissivityForecaster to
    provide a unified interface for monitoring shield permissivity.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_actions : int
        Number of actions.
    dead_threshold : float
        Permissivity below this is considered "dead" state.
    max_history : int
        Maximum snapshots to retain.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        dead_threshold: float = 0.1,
        max_history: int = 1000,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.dead_threshold = dead_threshold

        self._decomposition = PermissivityDecomposition(n_states, n_actions)
        self._forecaster = PermissivityForecaster(max_history=max_history)

        self._snapshots: List[PermissivitySnapshot] = []
        self._regime_snapshots: Dict[str, List[PermissivitySnapshot]] = defaultdict(list)
        self._current_regime: Optional[str] = None

        # Per-state running statistics
        self._state_sum: np.ndarray = np.zeros(n_states)
        self._state_sum_sq: np.ndarray = np.zeros(n_states)
        self._state_count: np.ndarray = np.zeros(n_states)
        self._state_min: np.ndarray = np.ones(n_states)
        self._state_max: np.ndarray = np.zeros(n_states)

    def set_regime(self, regime: str) -> None:
        """
        Set the current market regime for regime-conditioned tracking.

        Parameters
        ----------
        regime : str
            Regime label (e.g. 'bull', 'bear', 'high_vol').
        """
        self._current_regime = regime

    def record(
        self,
        permitted_table: np.ndarray,
        per_spec_safety: Optional[Dict[str, np.ndarray]] = None,
        threshold: float = 0.95,
    ) -> PermissivitySnapshot:
        """
        Record a permissivity snapshot.

        Parameters
        ----------
        permitted_table : np.ndarray
            Boolean (n_states, n_actions) table.
        per_spec_safety : dict, optional
            Per-spec safety tables for decomposition.
        threshold : float
            Safety threshold for spec decomposition.

        Returns
        -------
        PermissivitySnapshot
            The recorded snapshot.
        """
        per_state = np.sum(permitted_table.astype(float), axis=1) / self.n_actions
        mean_perm = float(np.mean(per_state))
        min_perm = float(np.min(per_state))
        max_perm = float(np.max(per_state))
        std_perm = float(np.std(per_state))
        n_dead = int(np.sum(per_state < self.dead_threshold))

        snapshot = PermissivitySnapshot(
            timestamp=time_module.time(),
            per_state=per_state.copy(),
            mean=mean_perm,
            minimum=min_perm,
            maximum=max_perm,
            std=std_perm,
            regime=self._current_regime,
            n_dead_states=n_dead,
        )

        self._snapshots.append(snapshot)
        if self._current_regime:
            self._regime_snapshots[self._current_regime].append(snapshot)

        # Update running statistics
        self._state_sum += per_state
        self._state_sum_sq += per_state ** 2
        self._state_count += 1
        self._state_min = np.minimum(self._state_min, per_state)
        self._state_max = np.maximum(self._state_max, per_state)

        # Update components
        if per_spec_safety:
            self._decomposition.update(permitted_table, per_spec_safety, threshold)
        self._forecaster.record(mean_perm, min_perm)

        # Trim history
        if len(self._snapshots) > self._forecaster.max_history:
            self._snapshots = self._snapshots[-self._forecaster.max_history:]

        return snapshot

    def get_current_permissivity(self) -> Optional[np.ndarray]:
        """Get the most recent per-state permissivity."""
        if not self._snapshots:
            return None
        return self._snapshots[-1].per_state.copy()

    def get_state_statistics(self, state: int) -> Dict[str, float]:
        """
        Get running statistics for a specific state.

        Parameters
        ----------
        state : int
            State index.

        Returns
        -------
        dict
            Statistics including mean, std, min, max, count.
        """
        count = self._state_count[state]
        if count == 0:
            return {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "count": 0}

        mean = self._state_sum[state] / count
        var = self._state_sum_sq[state] / count - mean ** 2
        std = float(np.sqrt(max(0, var)))

        return {
            "mean": float(mean),
            "std": std,
            "min": float(self._state_min[state]),
            "max": float(self._state_max[state]),
            "count": int(count),
        }

    def get_regime_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get permissivity statistics per regime.

        Returns
        -------
        dict
            Mapping regime -> statistics dict.
        """
        stats = {}
        for regime, snapshots in self._regime_snapshots.items():
            if not snapshots:
                continue
            means = [s.mean for s in snapshots]
            mins = [s.minimum for s in snapshots]
            stats[regime] = {
                "mean": float(np.mean(means)),
                "std": float(np.std(means)),
                "min": float(np.min(mins)),
                "n_snapshots": len(snapshots),
            }
        return stats

    def get_time_series(
        self, metric: str = "mean"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get time series of a permissivity metric.

        Parameters
        ----------
        metric : str
            'mean', 'minimum', 'maximum', or 'std'.

        Returns
        -------
        timestamps : np.ndarray
            Unix timestamps.
        values : np.ndarray
            Metric values.
        """
        if not self._snapshots:
            return np.array([]), np.array([])

        timestamps = np.array([s.timestamp for s in self._snapshots])
        if metric == "mean":
            values = np.array([s.mean for s in self._snapshots])
        elif metric == "minimum":
            values = np.array([s.minimum for s in self._snapshots])
        elif metric == "maximum":
            values = np.array([s.maximum for s in self._snapshots])
        elif metric == "std":
            values = np.array([s.std for s in self._snapshots])
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return timestamps, values

    def forecast(self, steps_ahead: int = 10, method: str = "linear") -> np.ndarray:
        """
        Forecast future permissivity.

        Parameters
        ----------
        steps_ahead : int
            Number of steps to forecast.
        method : str
            'linear' or 'exponential'.

        Returns
        -------
        np.ndarray
            Forecasted mean permissivity values.
        """
        if method == "linear":
            return self._forecaster.forecast_linear(steps_ahead)
        elif method == "exponential":
            return self._forecaster.forecast_exponential(steps_ahead)
        else:
            raise ValueError(f"Unknown forecast method: {method}")

    def generate_report(self) -> PermissivityReport:
        """
        Generate a comprehensive permissivity report.

        Returns
        -------
        PermissivityReport
            Full report with statistics, decomposition, and recommendations.
        """
        if not self._snapshots:
            return PermissivityReport(
                mean_permissivity=1.0,
                min_permissivity=1.0,
                max_permissivity=1.0,
                std_permissivity=0.0,
                n_dead_states=0,
                most_restrictive_spec="none",
                spec_restrictiveness={},
                trend_slope=0.0,
                regime_breakdown={},
                recommendations=["No data yet."],
            )

        latest = self._snapshots[-1]

        # Spec restrictiveness
        scores = self._decomposition.get_restrictiveness_scores()
        most_restrictive = max(scores, key=scores.get) if scores else "none"

        # Trend
        slope, _ = (0.0, 0.0)
        if len(self._snapshots) >= 3:
            y = np.array([s.mean for s in self._snapshots])
            x = np.arange(len(y), dtype=float)
            result = linregress(x, y)
            slope = result.slope

        # Regime breakdown
        regime_stats = self.get_regime_statistics()

        # Recommendations
        recs = self._generate_recommendations(latest, scores, slope)

        return PermissivityReport(
            mean_permissivity=latest.mean,
            min_permissivity=latest.minimum,
            max_permissivity=latest.maximum,
            std_permissivity=latest.std,
            n_dead_states=latest.n_dead_states,
            most_restrictive_spec=most_restrictive,
            spec_restrictiveness=scores,
            trend_slope=float(slope),
            regime_breakdown=regime_stats,
            recommendations=recs,
        )

    def _generate_recommendations(
        self,
        latest: PermissivitySnapshot,
        spec_scores: Dict[str, float],
        slope: float,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        if latest.minimum < self.dead_threshold:
            recs.append(
                f"CRITICAL: {latest.n_dead_states} states have permissivity "
                f"below {self.dead_threshold:.2f}. Consider relaxing the most "
                f"restrictive specification."
            )

        if slope < -0.005:
            ttl = self._forecaster.time_to_threshold(self.dead_threshold)
            msg = "Permissivity is declining."
            if ttl is not None:
                msg += f" Estimated {ttl} steps to threshold breach."
            recs.append(msg)

        if spec_scores:
            impacts = self._decomposition.get_marginal_impact()
            for name, impact in sorted(impacts.items(), key=lambda x: -x[1]):
                if impact > 0.05:
                    recs.append(
                        f"Relaxing '{name}' would increase mean permissivity "
                        f"by ~{impact:.3f}."
                    )

        if latest.std > 0.3:
            recs.append(
                "High permissivity variance across states. Consider "
                "state-dependent safety thresholds."
            )

        if not recs:
            recs.append("Permissivity is healthy. No action needed.")

        return recs

    @property
    def decomposition(self) -> PermissivityDecomposition:
        """Access the decomposition component."""
        return self._decomposition

    @property
    def forecaster(self) -> PermissivityForecaster:
        """Access the forecaster component."""
        return self._forecaster
