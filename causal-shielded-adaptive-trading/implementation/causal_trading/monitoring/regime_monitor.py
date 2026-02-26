"""
Regime change detection for Causal-Shielded Adaptive Trading.

Implements CUSUM-based and Bayesian online change point detection
(Adams & MacKay 2007) for identifying regime transitions in financial
time series. Tracks run length distributions and generates alerts
when statistically significant regime changes are detected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severity levels for monitoring alerts."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class RegimeAlert:
    """An alert generated when a regime change is detected."""
    timestamp: int
    severity: AlertSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"RegimeAlert(t={self.timestamp}, severity={self.severity.value}, "
            f"msg={self.message!r})"
        )


@dataclass
class RegimeStatistics:
    """Exponentially weighted statistics for a single regime."""
    mean: np.ndarray
    var: np.ndarray
    count: int
    start_time: int
    end_time: Optional[int] = None

    @property
    def duration(self) -> int:
        end = self.end_time if self.end_time is not None else self.start_time
        return end - self.start_time + 1


class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) change point detector.

    Monitors the cumulative sum of deviations from a target value.
    When the CUSUM statistic exceeds a threshold, a regime change is
    signalled. Supports both upward and downward shifts.

    Parameters
    ----------
    threshold : float
        Detection threshold for the CUSUM statistic.
    drift : float
        Allowable slack (drift) before accumulating evidence.
    dim : int
        Dimensionality of the observation space.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        drift: float = 0.5,
        dim: int = 1,
    ) -> None:
        self.threshold = threshold
        self.drift = drift
        self.dim = dim

        self.s_pos: np.ndarray = np.zeros(dim)
        self.s_neg: np.ndarray = np.zeros(dim)
        self._target: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._n_warmup: int = 0
        self._warmup_buffer: List[np.ndarray] = []
        self._warmup_size: int = 50

    def _ensure_warmup(self, obs: np.ndarray) -> bool:
        """Accumulate warmup data to estimate target mean and std."""
        if self._target is not None:
            return True
        self._warmup_buffer.append(obs.copy())
        self._n_warmup += 1
        if self._n_warmup >= self._warmup_size:
            buf = np.array(self._warmup_buffer)
            self._target = np.mean(buf, axis=0)
            self._std = np.std(buf, axis=0) + 1e-8
            self.s_pos = np.zeros(self.dim)
            self.s_neg = np.zeros(self.dim)
            return True
        return False

    def update(self, obs: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Update CUSUM statistics with a new observation.

        Returns
        -------
        detected : bool
            Whether a change was detected on any dimension.
        statistic : np.ndarray
            The maximum of upward and downward CUSUM statistics.
        """
        obs = np.asarray(obs, dtype=np.float64).ravel()
        if not self._ensure_warmup(obs):
            return False, np.zeros(self.dim)

        z = (obs - self._target) / self._std
        self.s_pos = np.maximum(0.0, self.s_pos + z - self.drift)
        self.s_neg = np.maximum(0.0, self.s_neg - z - self.drift)

        stat = np.maximum(self.s_pos, self.s_neg)
        detected = bool(np.any(stat > self.threshold))
        return detected, stat

    def reset(self, new_target: Optional[np.ndarray] = None) -> None:
        """Reset CUSUM statistics, optionally with a new target."""
        self.s_pos = np.zeros(self.dim)
        self.s_neg = np.zeros(self.dim)
        if new_target is not None:
            self._target = np.asarray(new_target, dtype=np.float64).ravel()


class BayesianChangepointDetector:
    """
    Bayesian Online Change Point Detection (Adams & MacKay, 2007).

    Maintains a distribution over the current run length (time since
    the last change point). Uses a Gaussian predictive model with
    conjugate Normal-Inverse-Gamma priors.

    Parameters
    ----------
    hazard_rate : float
        Prior probability of a change point at each time step (1/expected_run_length).
    dim : int
        Dimensionality of observations.
    mu0 : float
        Prior mean for Normal-Inverse-Gamma.
    kappa0 : float
        Prior precision scale.
    alpha0 : float
        Prior shape for inverse-gamma.
    beta0 : float
        Prior scale for inverse-gamma.
    """

    def __init__(
        self,
        hazard_rate: float = 1.0 / 200.0,
        dim: int = 1,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        self.hazard_rate = hazard_rate
        self.dim = dim
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

        self._max_run = 500
        self.log_run_length_dist: np.ndarray = np.full(self._max_run, -np.inf)
        self.log_run_length_dist[0] = 0.0  # log(1) = 0

        # Sufficient statistics arrays (one per possible run length)
        self._mu = np.full((self._max_run, dim), mu0, dtype=np.float64)
        self._kappa = np.full(self._max_run, kappa0, dtype=np.float64)
        self._alpha = np.full(self._max_run, alpha0, dtype=np.float64)
        self._beta = np.full((self._max_run, dim), beta0, dtype=np.float64)

        self.t: int = 0

    def _predictive_log_prob(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute log predictive probability under Student-t for each run length.

        The predictive distribution is a multivariate Student-t with
        2*alpha degrees of freedom.
        """
        obs = obs.ravel()
        df = 2.0 * self._alpha  # shape (max_run,)
        scale_sq = (
            self._beta * (self._kappa + 1.0)[:, None]
            / (self._kappa[:, None] * self._alpha[:, None])
        )  # shape (max_run, dim)

        # Per-dimension log prob under Student-t
        log_probs = np.zeros(self._max_run, dtype=np.float64)
        for d in range(self.dim):
            z = (obs[d] - self._mu[:, d]) ** 2 / (scale_sq[:, d] + 1e-30)
            lp = (
                sp_stats.t.logpdf(
                    obs[d],
                    df=df,
                    loc=self._mu[:, d],
                    scale=np.sqrt(scale_sq[:, d] + 1e-30),
                )
            )
            log_probs += lp

        return log_probs

    def update(self, obs: np.ndarray) -> np.ndarray:
        """
        Incorporate a new observation and return the run length distribution.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (dim,).

        Returns
        -------
        run_length_dist : np.ndarray
            Posterior distribution over run lengths (normalized).
        """
        obs = np.asarray(obs, dtype=np.float64).ravel()
        assert obs.shape[0] == self.dim

        log_pred = self._predictive_log_prob(obs)  # (max_run,)

        log_h = np.log(self.hazard_rate + 1e-30)
        log_1mh = np.log(1.0 - self.hazard_rate + 1e-30)

        # Growth probabilities: P(r_t = r_{t-1}+1, x_{1:t})
        log_growth = (
            self.log_run_length_dist + log_pred + log_1mh
        )

        # Change point probability: P(r_t = 0, x_{1:t})
        log_cp = logsumexp(
            self.log_run_length_dist + log_pred + log_h
        )

        # Shift distribution forward
        new_log_dist = np.full(self._max_run, -np.inf)
        new_log_dist[0] = log_cp
        new_log_dist[1:] = log_growth[:-1]

        # Normalize
        log_evidence = logsumexp(new_log_dist)
        new_log_dist -= log_evidence
        self.log_run_length_dist = new_log_dist

        # Update sufficient statistics
        new_mu = np.full((self._max_run, self.dim), self.mu0, dtype=np.float64)
        new_kappa = np.full(self._max_run, self.kappa0, dtype=np.float64)
        new_alpha = np.full(self._max_run, self.alpha0, dtype=np.float64)
        new_beta = np.full((self._max_run, self.dim), self.beta0, dtype=np.float64)

        for d in range(self.dim):
            old_mu_d = self._mu[:-1, d]
            old_kappa = self._kappa[:-1]
            old_alpha = self._alpha[:-1]
            old_beta_d = self._beta[:-1, d]

            new_kappa[1:] = old_kappa + 1.0
            new_mu[1:, d] = (old_kappa * old_mu_d + obs[d]) / new_kappa[1:]
            new_alpha[1:] = old_alpha + 0.5
            new_beta[1:, d] = (
                old_beta_d
                + 0.5
                * old_kappa
                / new_kappa[1:]
                * (obs[d] - old_mu_d) ** 2
            )

        self._mu = new_mu
        self._kappa = new_kappa
        self._alpha = new_alpha
        self._beta = new_beta

        self.t += 1
        return np.exp(new_log_dist)

    def get_map_run_length(self) -> int:
        """Return the MAP estimate of the current run length."""
        return int(np.argmax(self.log_run_length_dist))

    def get_changepoint_probability(self) -> float:
        """Return posterior probability that a change just occurred (r=0)."""
        return float(np.exp(self.log_run_length_dist[0]))

    def get_expected_run_length(self) -> float:
        """Return the expected run length under the posterior."""
        probs = np.exp(self.log_run_length_dist)
        lengths = np.arange(self._max_run, dtype=np.float64)
        return float(np.dot(probs, lengths))


class RegimeDurationTracker:
    """
    Tracks the duration and statistics of identified regimes over time.

    Each regime is characterized by a start time, running statistics,
    and an optional end time when the next regime begins.
    """

    def __init__(self, ewma_alpha: float = 0.05) -> None:
        self.ewma_alpha = ewma_alpha
        self.regimes: List[RegimeStatistics] = []
        self._current_mean: Optional[np.ndarray] = None
        self._current_var: Optional[np.ndarray] = None
        self._current_count: int = 0
        self._current_start: int = 0

    @property
    def current_regime_index(self) -> int:
        return len(self.regimes)

    @property
    def current_duration(self) -> int:
        return self._current_count

    def start_new_regime(self, t: int) -> None:
        """Finalize the current regime and start a new one."""
        if self._current_mean is not None and self._current_count > 0:
            regime = RegimeStatistics(
                mean=self._current_mean.copy(),
                var=self._current_var.copy(),
                count=self._current_count,
                start_time=self._current_start,
                end_time=t - 1,
            )
            self.regimes.append(regime)

        self._current_mean = None
        self._current_var = None
        self._current_count = 0
        self._current_start = t

    def update(self, obs: np.ndarray, t: int) -> None:
        """Update exponentially weighted statistics for the current regime."""
        obs = np.asarray(obs, dtype=np.float64).ravel()
        if self._current_mean is None:
            self._current_mean = obs.copy()
            self._current_var = np.zeros_like(obs)
            self._current_count = 1
            return

        self._current_count += 1
        alpha = self.ewma_alpha
        delta = obs - self._current_mean
        self._current_mean = (1.0 - alpha) * self._current_mean + alpha * obs
        delta2 = obs - self._current_mean
        self._current_var = (
            (1.0 - alpha) * self._current_var + alpha * delta * delta2
        )

    def get_regime_summary(self) -> List[Dict[str, Any]]:
        """Return summary statistics for all regimes including current."""
        summaries = []
        for i, r in enumerate(self.regimes):
            summaries.append({
                "regime_index": i,
                "start": r.start_time,
                "end": r.end_time,
                "duration": r.duration,
                "mean": r.mean.tolist(),
                "var": r.var.tolist(),
                "count": r.count,
            })
        # Current regime
        if self._current_mean is not None:
            summaries.append({
                "regime_index": self.current_regime_index,
                "start": self._current_start,
                "end": None,
                "duration": self._current_count,
                "mean": self._current_mean.tolist(),
                "var": self._current_var.tolist() if self._current_var is not None else None,
                "count": self._current_count,
            })
        return summaries


class RegimeMonitor:
    """
    Combined regime change detection monitor.

    Runs both CUSUM and Bayesian online change point detection in parallel
    and generates alerts when either (or both) detect a regime change.
    Tracks regime-specific statistics and durations.

    Parameters
    ----------
    dim : int
        Dimensionality of observations.
    cusum_threshold : float
        CUSUM detection threshold.
    cusum_drift : float
        CUSUM drift parameter.
    hazard_rate : float
        Prior change point probability for Bayesian detector.
    bayesian_cp_threshold : float
        Posterior probability threshold to declare a Bayesian change point.
    min_regime_length : int
        Minimum observations between successive change points.
    ewma_alpha : float
        Decay factor for exponentially weighted regime statistics.
    """

    def __init__(
        self,
        dim: int = 1,
        cusum_threshold: float = 5.0,
        cusum_drift: float = 0.5,
        hazard_rate: float = 1.0 / 200.0,
        bayesian_cp_threshold: float = 0.5,
        min_regime_length: int = 20,
        ewma_alpha: float = 0.05,
    ) -> None:
        self.dim = dim
        self.bayesian_cp_threshold = bayesian_cp_threshold
        self.min_regime_length = min_regime_length

        self.cusum = CUSUMDetector(
            threshold=cusum_threshold,
            drift=cusum_drift,
            dim=dim,
        )
        self.bayesian = BayesianChangepointDetector(
            hazard_rate=hazard_rate,
            dim=dim,
        )
        self.duration_tracker = RegimeDurationTracker(ewma_alpha=ewma_alpha)

        self._t: int = 0
        self._since_last_cp: int = 0
        self._alerts: List[RegimeAlert] = []
        self._change_points: List[int] = []
        self._stability_scores: List[float] = []

    def update(self, observation: np.ndarray) -> Optional[RegimeAlert]:
        """
        Process a new observation through all detection methods.

        Parameters
        ----------
        observation : np.ndarray
            Observation vector of shape (dim,).

        Returns
        -------
        alert : RegimeAlert or None
            An alert if a regime change was detected, else None.
        """
        observation = np.asarray(observation, dtype=np.float64).ravel()
        assert observation.shape[0] == self.dim, (
            f"Expected dim={self.dim}, got {observation.shape[0]}"
        )

        # Update detectors
        cusum_detected, cusum_stat = self.cusum.update(observation)
        run_dist = self.bayesian.update(observation)
        bayes_cp_prob = self.bayesian.get_changepoint_probability()

        # Update regime tracker
        self.duration_tracker.update(observation, self._t)

        # Check for regime change
        self._since_last_cp += 1
        alert = None

        if self._since_last_cp >= self.min_regime_length:
            bayes_detected = bayes_cp_prob > self.bayesian_cp_threshold

            if cusum_detected and bayes_detected:
                alert = self._generate_alert(
                    AlertSeverity.CRITICAL,
                    "Regime change detected by both CUSUM and Bayesian CPD",
                    cusum_stat=float(np.max(cusum_stat)),
                    bayes_cp_prob=bayes_cp_prob,
                )
            elif cusum_detected:
                alert = self._generate_alert(
                    AlertSeverity.WARNING,
                    "Regime change detected by CUSUM",
                    cusum_stat=float(np.max(cusum_stat)),
                    bayes_cp_prob=bayes_cp_prob,
                )
            elif bayes_detected:
                alert = self._generate_alert(
                    AlertSeverity.WARNING,
                    "Regime change detected by Bayesian CPD",
                    cusum_stat=float(np.max(cusum_stat)),
                    bayes_cp_prob=bayes_cp_prob,
                )

            if alert is not None:
                self._handle_change_point(observation)

        # Compute stability score
        stability = self._compute_stability(bayes_cp_prob, cusum_stat)
        self._stability_scores.append(stability)

        self._t += 1
        return alert

    def detect_change(self) -> bool:
        """Return whether the most recent observation triggered a change."""
        if not self._change_points:
            return False
        return self._change_points[-1] == self._t - 1

    def get_run_length(self) -> Dict[str, Any]:
        """
        Return current run length information from both detectors.

        Returns
        -------
        info : dict
            Contains MAP run length, expected run length, change point
            probability, and observations since last detected change.
        """
        return {
            "map_run_length": self.bayesian.get_map_run_length(),
            "expected_run_length": self.bayesian.get_expected_run_length(),
            "changepoint_probability": self.bayesian.get_changepoint_probability(),
            "since_last_cp": self._since_last_cp,
            "total_observations": self._t,
        }

    def get_alerts(self, last_n: Optional[int] = None) -> List[RegimeAlert]:
        """Return recent alerts, optionally limited to the last n."""
        if last_n is not None:
            return self._alerts[-last_n:]
        return list(self._alerts)

    def get_regime_summary(self) -> Dict[str, Any]:
        """Return a summary of all regimes and current regime statistics."""
        return {
            "num_regimes": self.duration_tracker.current_regime_index + 1,
            "current_regime_duration": self.duration_tracker.current_duration,
            "change_points": list(self._change_points),
            "regimes": self.duration_tracker.get_regime_summary(),
        }

    def get_stability_metrics(self) -> Dict[str, float]:
        """
        Return stability metrics for the current regime.

        Metrics include mean and recent stability score, coefficient
        of variation, and the regime fragility index.
        """
        if not self._stability_scores:
            return {"mean_stability": 1.0, "recent_stability": 1.0}

        recent = self._stability_scores[-50:]
        all_scores = np.array(self._stability_scores)
        recent_arr = np.array(recent)

        mean_stab = float(np.mean(all_scores))
        recent_stab = float(np.mean(recent_arr))
        cv = float(np.std(recent_arr) / (np.mean(recent_arr) + 1e-10))

        n_cp = len(self._change_points)
        fragility = n_cp / max(self._t, 1)

        return {
            "mean_stability": mean_stab,
            "recent_stability": recent_stab,
            "stability_cv": cv,
            "fragility_index": fragility,
            "total_change_points": n_cp,
        }

    def significance_test(
        self,
        window_before: int = 50,
        window_after: int = 50,
    ) -> Optional[Dict[str, Any]]:
        """
        Test significance of the most recent change point using a
        two-sample Kolmogorov–Smirnov test on stability scores before
        and after the change point.

        Returns None if insufficient data.
        """
        if not self._change_points:
            return None
        cp = self._change_points[-1]
        start_before = max(0, cp - window_before)
        end_after = min(len(self._stability_scores), cp + window_after)

        if cp - start_before < 5 or end_after - cp < 5:
            return None

        before = np.array(self._stability_scores[start_before:cp])
        after = np.array(self._stability_scores[cp:end_after])

        ks_stat, p_value = sp_stats.ks_2samp(before, after)
        return {
            "change_point": cp,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "significant_at_005": p_value < 0.05,
            "significant_at_001": p_value < 0.01,
            "n_before": len(before),
            "n_after": len(after),
        }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _generate_alert(
        self,
        severity: AlertSeverity,
        message: str,
        **details: Any,
    ) -> RegimeAlert:
        alert = RegimeAlert(
            timestamp=self._t,
            severity=severity,
            message=message,
            details=details,
        )
        self._alerts.append(alert)
        logger.info("RegimeMonitor alert: %s", alert)
        return alert

    def _handle_change_point(self, observation: np.ndarray) -> None:
        """Record a change point and reset detectors."""
        self._change_points.append(self._t)
        self._since_last_cp = 0
        self.cusum.reset(new_target=observation)
        self.duration_tracker.start_new_regime(self._t)

    def _compute_stability(
        self,
        cp_prob: float,
        cusum_stat: np.ndarray,
    ) -> float:
        """
        Compute a composite stability score in [0, 1].

        High values indicate stable regime; low values indicate
        impending or recent change.
        """
        cusum_component = 1.0 / (1.0 + float(np.max(cusum_stat)))
        bayes_component = 1.0 - cp_prob
        return 0.5 * cusum_component + 0.5 * bayes_component
