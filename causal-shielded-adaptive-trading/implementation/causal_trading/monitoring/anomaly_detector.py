"""
Anomaly detection for Causal-Shielded Adaptive Trading.

Provides statistical anomaly detection using Mahalanobis distance and
isolation-forest–style random partitioning, distribution shift detection
via Kolmogorov–Smirnov and Maximum Mean Discrepancy tests, and a
combined alert system with severity levels and deduplication.
"""

from __future__ import annotations

import hashlib
import logging
import time as _time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class AlertLevel(IntEnum):
    """Alert severity levels (ordered by severity)."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2


@dataclass
class AnomalyAlert:
    """An alert produced by the anomaly detection system."""
    timestamp: int
    level: AlertLevel
    source: str
    message: str
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            raw = f"{self.source}:{self.message}:{self.level}"
            self.fingerprint = hashlib.md5(raw.encode()).hexdigest()[:12]


class OnlineCovariance:
    """
    Welford-style online estimation of mean and covariance matrix.

    Used for Mahalanobis-distance–based anomaly detection.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.n: int = 0
        self.mean: np.ndarray = np.zeros(dim, dtype=np.float64)
        self._M2: np.ndarray = np.zeros((dim, dim), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._M2 += np.outer(delta, delta2)

    def covariance(self, regularize: float = 1e-6) -> np.ndarray:
        if self.n < 2:
            return np.eye(self.dim) * regularize
        cov = self._M2 / (self.n - 1)
        cov += np.eye(self.dim) * regularize
        return cov

    def precision(self, regularize: float = 1e-6) -> np.ndarray:
        return np.linalg.inv(self.covariance(regularize))


def mahalanobis_distance(
    x: np.ndarray,
    mean: np.ndarray,
    precision: np.ndarray,
) -> float:
    """Compute the Mahalanobis distance of x from the distribution."""
    diff = x - mean
    return float(np.sqrt(diff @ precision @ diff))


class IsolationScore:
    """
    Lightweight isolation-forest–style anomaly scoring.

    Builds a set of random axis-aligned splits and scores points
    by average path length. Anomalies have shorter average paths.

    Parameters
    ----------
    n_estimators : int
        Number of random trees (sets of splits).
    max_depth : int
        Maximum depth for each tree.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 10,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._rng = np.random.RandomState(seed)
        self._trees: List[List[Tuple[int, float]]] = []
        self._fitted = False

    def fit(self, X: np.ndarray) -> None:
        """Build isolation splits from reference data X of shape (n, d)."""
        n, d = X.shape
        self._trees = []
        for _ in range(self.n_estimators):
            splits: List[Tuple[int, float]] = []
            for depth in range(self.max_depth):
                feat = self._rng.randint(0, d)
                lo, hi = float(X[:, feat].min()), float(X[:, feat].max())
                if hi - lo < 1e-12:
                    threshold = lo
                else:
                    threshold = self._rng.uniform(lo, hi)
                splits.append((feat, threshold))
            self._trees.append(splits)
        self._fitted = True

    def score(self, x: np.ndarray) -> float:
        """
        Score an observation. Returns a value in (0, 1) where higher
        values indicate greater anomaly likelihood.
        """
        if not self._fitted:
            return 0.5
        x = np.asarray(x, dtype=np.float64).ravel()
        total_depth = 0.0
        for splits in self._trees:
            depth = 0
            for feat, threshold in splits:
                depth += 1
                # Simplified: just count how deep we go before isolating
                if x[feat] < threshold:
                    break
            total_depth += depth

        avg_depth = total_depth / self.n_estimators
        # Normalize: shorter path => more anomalous
        score = 2.0 ** (-avg_depth / self.max_depth)
        return float(score)


class DistributionShiftDetector:
    """
    Detects feature distribution shifts using Kolmogorov–Smirnov test
    and Maximum Mean Discrepancy (MMD).

    Parameters
    ----------
    reference_window : int
        Number of observations in the reference window.
    test_window : int
        Number of observations in the test (recent) window.
    ks_threshold : float
        p-value threshold below which KS test signals a shift.
    mmd_threshold : float
        MMD value above which a shift is declared.
    """

    def __init__(
        self,
        reference_window: int = 200,
        test_window: int = 50,
        ks_threshold: float = 0.01,
        mmd_threshold: float = 0.1,
    ) -> None:
        self.reference_window = reference_window
        self.test_window = test_window
        self.ks_threshold = ks_threshold
        self.mmd_threshold = mmd_threshold
        self._buffer: List[np.ndarray] = []

    def update(self, obs: np.ndarray) -> None:
        self._buffer.append(np.asarray(obs, dtype=np.float64).ravel())
        max_buf = self.reference_window + self.test_window + 100
        if len(self._buffer) > max_buf:
            self._buffer = self._buffer[-max_buf:]

    def check_shift(self) -> Dict[str, Any]:
        """
        Run KS and MMD tests comparing reference and test windows.

        Returns
        -------
        result : dict
            Contains per-feature KS statistics, aggregate MMD,
            and overall shift determination.
        """
        needed = self.reference_window + self.test_window
        if len(self._buffer) < needed:
            return {"shift_detected": False, "reason": "insufficient_data"}

        ref_data = np.array(
            self._buffer[-(self.reference_window + self.test_window):-self.test_window]
        )
        test_data = np.array(self._buffer[-self.test_window:])

        dim = ref_data.shape[1]

        # Per-feature KS test
        ks_results = []
        any_ks_shift = False
        for d in range(dim):
            stat, pval = sp_stats.ks_2samp(ref_data[:, d], test_data[:, d])
            shifted = pval < self.ks_threshold
            if shifted:
                any_ks_shift = True
            ks_results.append({
                "feature": d,
                "statistic": float(stat),
                "p_value": float(pval),
                "shifted": shifted,
            })

        # MMD (simplified with RBF kernel, median heuristic)
        mmd_val = self._compute_mmd(ref_data, test_data)
        mmd_shift = mmd_val > self.mmd_threshold

        return {
            "shift_detected": any_ks_shift or mmd_shift,
            "ks_shift": any_ks_shift,
            "mmd_shift": mmd_shift,
            "mmd_value": float(mmd_val),
            "ks_results": ks_results,
        }

    @staticmethod
    def _compute_mmd(X: np.ndarray, Y: np.ndarray) -> float:
        """Compute unbiased MMD^2 estimate with RBF kernel."""
        n, m = X.shape[0], Y.shape[0]
        if n < 2 or m < 2:
            return 0.0

        XY = np.vstack([X, Y])
        dists = np.sum(XY ** 2, axis=1, keepdims=True) - 2 * XY @ XY.T + np.sum(XY ** 2, axis=1)
        med = np.median(np.abs(dists[dists > 0])) + 1e-8
        sigma_sq = med

        def rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            sq = (
                np.sum(A ** 2, axis=1, keepdims=True)
                - 2 * A @ B.T
                + np.sum(B ** 2, axis=1)
            )
            return np.exp(-sq / (2 * sigma_sq))

        Kxx = rbf(X, X)
        Kyy = rbf(Y, Y)
        Kxy = rbf(X, Y)

        # Unbiased estimator
        np.fill_diagonal(Kxx, 0)
        np.fill_diagonal(Kyy, 0)

        mmd_sq = (
            Kxx.sum() / (n * (n - 1))
            + Kyy.sum() / (m * (m - 1))
            - 2 * Kxy.sum() / (n * m)
        )
        return max(0.0, float(mmd_sq))


class AnomalyDetector:
    """
    Combined anomaly detection system with alert management.

    Uses Mahalanobis distance, isolation scoring, and distribution
    shift detection to identify anomalous observations. Provides
    alert severity classification, history tracking, and deduplication.

    Parameters
    ----------
    dim : int
        Dimensionality of observations.
    mahal_warning : float
        Mahalanobis distance threshold for WARNING.
    mahal_critical : float
        Mahalanobis distance threshold for CRITICAL.
    isolation_threshold : float
        Isolation score above which an observation is anomalous.
    warmup : int
        Number of observations before anomaly detection activates.
    dedup_window : int
        Suppress duplicate alerts within this many time steps.
    """

    def __init__(
        self,
        dim: int = 1,
        mahal_warning: float = 3.0,
        mahal_critical: float = 5.0,
        isolation_threshold: float = 0.7,
        warmup: int = 100,
        dedup_window: int = 5,
    ) -> None:
        self.dim = dim
        self.mahal_warning = mahal_warning
        self.mahal_critical = mahal_critical
        self.isolation_threshold = isolation_threshold
        self.warmup = warmup
        self.dedup_window = dedup_window

        self._cov_estimator = OnlineCovariance(dim)
        self._isolation = IsolationScore()
        self._shift_detector = DistributionShiftDetector()

        self._t: int = 0
        self._warmup_buffer: List[np.ndarray] = []
        self._alerts: List[AnomalyAlert] = []
        self._recent_fingerprints: Dict[str, int] = {}
        self._is_warm: bool = False

    def check(self, observation: np.ndarray) -> List[AnomalyAlert]:
        """
        Check an observation for anomalies.

        Parameters
        ----------
        observation : np.ndarray
            Observation vector of shape (dim,).

        Returns
        -------
        alerts : list of AnomalyAlert
            Any new alerts (may be empty).
        """
        obs = np.asarray(observation, dtype=np.float64).ravel()
        assert obs.shape[0] == self.dim

        self._cov_estimator.update(obs)
        self._shift_detector.update(obs)

        new_alerts: List[AnomalyAlert] = []

        if not self._is_warm:
            self._warmup_buffer.append(obs)
            self._t += 1
            if len(self._warmup_buffer) >= self.warmup:
                buf = np.array(self._warmup_buffer)
                self._isolation.fit(buf)
                self._is_warm = True
            return new_alerts

        # Mahalanobis distance
        try:
            prec = self._cov_estimator.precision()
            md = mahalanobis_distance(obs, self._cov_estimator.mean, prec)
        except np.linalg.LinAlgError:
            md = 0.0

        if md > self.mahal_critical:
            alert = self._make_alert(
                AlertLevel.CRITICAL,
                "mahalanobis",
                f"Mahalanobis distance {md:.2f} exceeds critical threshold",
                score=md,
                details={"mahalanobis_distance": md},
            )
            if alert:
                new_alerts.append(alert)
        elif md > self.mahal_warning:
            alert = self._make_alert(
                AlertLevel.WARNING,
                "mahalanobis",
                f"Mahalanobis distance {md:.2f} exceeds warning threshold",
                score=md,
                details={"mahalanobis_distance": md},
            )
            if alert:
                new_alerts.append(alert)

        # Isolation score
        iso_score = self._isolation.score(obs)
        if iso_score > self.isolation_threshold:
            alert = self._make_alert(
                AlertLevel.WARNING,
                "isolation",
                f"Isolation score {iso_score:.3f} indicates anomaly",
                score=iso_score,
                details={"isolation_score": iso_score},
            )
            if alert:
                new_alerts.append(alert)

        # Distribution shift (check periodically)
        if self._t % 20 == 0:
            shift_result = self._shift_detector.check_shift()
            if shift_result.get("shift_detected", False):
                level = (
                    AlertLevel.CRITICAL
                    if shift_result.get("mmd_shift", False)
                    else AlertLevel.WARNING
                )
                alert = self._make_alert(
                    level,
                    "distribution_shift",
                    "Feature distribution shift detected",
                    score=shift_result.get("mmd_value", 0.0),
                    details=shift_result,
                )
                if alert:
                    new_alerts.append(alert)

        self._t += 1
        return new_alerts

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        last_n: Optional[int] = None,
    ) -> List[AnomalyAlert]:
        """
        Return alerts, optionally filtered by level.

        Parameters
        ----------
        level : AlertLevel, optional
            Minimum severity to include.
        last_n : int, optional
            Limit to last n alerts.
        """
        result = self._alerts
        if level is not None:
            result = [a for a in result if a.level >= level]
        if last_n is not None:
            result = result[-last_n:]
        return result

    def get_alert_history(self) -> Dict[str, Any]:
        """
        Return alert history summary.

        Returns
        -------
        summary : dict
            Counts by level and source, total alerts, and time span.
        """
        by_level = {lv.name: 0 for lv in AlertLevel}
        by_source: Dict[str, int] = {}
        for a in self._alerts:
            by_level[a.level.name] = by_level.get(a.level.name, 0) + 1
            by_source[a.source] = by_source.get(a.source, 0) + 1

        return {
            "total_alerts": len(self._alerts),
            "by_level": by_level,
            "by_source": by_source,
            "total_observations": self._t,
            "alert_rate": len(self._alerts) / max(self._t, 1),
        }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _make_alert(
        self,
        level: AlertLevel,
        source: str,
        message: str,
        score: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[AnomalyAlert]:
        """Create an alert with deduplication."""
        alert = AnomalyAlert(
            timestamp=self._t,
            level=level,
            source=source,
            message=message,
            score=score,
            details=details or {},
        )

        # Deduplication
        fp = alert.fingerprint
        last_seen = self._recent_fingerprints.get(fp)
        if last_seen is not None and (self._t - last_seen) < self.dedup_window:
            return None

        self._recent_fingerprints[fp] = self._t
        self._alerts.append(alert)
        logger.info("AnomalyDetector [%s] %s: %s", level.name, source, message)
        return alert
