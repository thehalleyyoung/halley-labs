"""Cross-firm correlation analysis for collusion detection.

Provides Pearson, Spearman, Kendall, partial, rolling and dynamic
correlations, Granger causality, mutual information, spectral coherence,
and change-point tests on correlation structure.
"""

from __future__ import annotations

import numpy as np
from scipy import stats, signal
from typing import Optional, Dict, Tuple, List, Any


class CorrelationAnalyzer:
    """Analyzes cross-firm price correlations."""

    def __init__(self, significance_level: float = 0.05) -> None:
        self.significance_level = significance_level

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def full_analysis(self, prices: np.ndarray) -> Dict[str, Any]:
        """Run all correlation analyses.

        Parameters
        ----------
        prices : ndarray of shape ``(num_rounds, num_players)``
        """
        prices = np.asarray(prices, dtype=float)
        if prices.ndim != 2 or prices.shape[1] < 2:
            raise ValueError("Need (T, N>=2) price matrix")

        pcorr, pcorr_p = self.pearson_correlation(prices)
        scorr, scorr_p = self.spearman_correlation(prices)
        kcorr, kcorr_p = self.kendall_correlation(prices)

        result: Dict[str, Any] = {
            "pearson": {"corr": pcorr.tolist(), "p_values": pcorr_p.tolist()},
            "spearman": {"corr": scorr.tolist(), "p_values": scorr_p.tolist()},
            "kendall": {"corr": kcorr.tolist(), "p_values": kcorr_p.tolist()},
            "partial_correlation": self.partial_correlation(prices).tolist(),
            "mutual_information": self.mutual_information(prices).tolist(),
            "granger_causality": self.granger_causality(prices),
            "cross_correlation": {
                k: v.tolist()
                for k, v in self.cross_correlation_function(prices).items()
                if isinstance(v, np.ndarray)
            },
        }
        if prices.shape[0] >= 64:
            result["coherence"] = self.coherence(prices)
        return result

    # ------------------------------------------------------------------
    # Basic correlations
    # ------------------------------------------------------------------

    def pearson_correlation(
        self, prices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pairwise Pearson correlations on first-differenced prices."""
        diffs = np.diff(prices, axis=0)
        n = diffs.shape[1]
        corr = np.eye(n)
        pval = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r, p = stats.pearsonr(diffs[:, i], diffs[:, j])
                corr[i, j] = corr[j, i] = r
                pval[i, j] = pval[j, i] = p
        return corr, pval

    def spearman_correlation(
        self, prices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        diffs = np.diff(prices, axis=0)
        n = diffs.shape[1]
        corr = np.eye(n)
        pval = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r, p = stats.spearmanr(diffs[:, i], diffs[:, j])
                corr[i, j] = corr[j, i] = r
                pval[i, j] = pval[j, i] = p
        return corr, pval

    def kendall_correlation(
        self, prices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        diffs = np.diff(prices, axis=0)
        n = diffs.shape[1]
        corr = np.eye(n)
        pval = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r, p = stats.kendalltau(diffs[:, i], diffs[:, j])
                corr[i, j] = corr[j, i] = r
                pval[i, j] = pval[j, i] = p
        return corr, pval

    # ------------------------------------------------------------------
    # Partial correlation
    # ------------------------------------------------------------------

    def partial_correlation(self, prices: np.ndarray) -> np.ndarray:
        """Partial correlation matrix controlling for other firms.

        Uses the inverse of the correlation matrix (precision matrix).
        """
        diffs = np.diff(prices, axis=0)
        corr = np.corrcoef(diffs.T)
        n = corr.shape[0]
        # Regularise to ensure invertibility
        corr += np.eye(n) * 1e-8
        try:
            precision = np.linalg.inv(corr)
        except np.linalg.LinAlgError:
            return np.full((n, n), float("nan"))
        d = np.sqrt(np.diag(precision))
        outer_d = np.outer(d, d)
        outer_d = np.where(outer_d < 1e-15, 1.0, outer_d)
        partial = -precision / outer_d
        np.fill_diagonal(partial, 1.0)
        return partial

    # ------------------------------------------------------------------
    # Rolling correlation
    # ------------------------------------------------------------------

    def rolling_correlation(
        self, prices: np.ndarray, window: int = 1000
    ) -> np.ndarray:
        """Rolling-window Pearson correlation.

        Returns array of shape ``(n_windows, n_pairs)`` where *n_pairs* is
        ``N*(N-1)/2`` for *N* players.
        """
        diffs = np.diff(prices, axis=0)
        T, N = diffs.shape
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        n_windows = max(T - window + 1, 0)
        result = np.empty((n_windows, len(pairs)))
        for w in range(n_windows):
            seg = diffs[w : w + window]
            for idx, (i, j) in enumerate(pairs):
                r, _ = stats.pearsonr(seg[:, i], seg[:, j])
                result[w, idx] = r
        return result

    # ------------------------------------------------------------------
    # Granger causality
    # ------------------------------------------------------------------

    def granger_causality(
        self, prices: np.ndarray, max_lag: int = 10
    ) -> Dict[str, Any]:
        """Pairwise Granger causality tests (OLS-based F-test)."""
        diffs = np.diff(prices, axis=0)
        N = diffs.shape[1]
        T = diffs.shape[0]
        results: Dict[str, Any] = {}

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                best_p = 1.0
                best_lag = 1
                best_f = 0.0
                for lag in range(1, min(max_lag, T // 3) + 1):
                    y = diffs[lag:, j]
                    # Restricted: only own lags
                    X_r = np.column_stack(
                        [diffs[lag - k - 1 : T - k - 1, j] for k in range(lag)]
                    )
                    # Unrestricted: own lags + i's lags
                    X_u = np.column_stack(
                        [X_r]
                        + [diffs[lag - k - 1 : T - k - 1, i] for k in range(lag)]
                    )
                    n_obs = len(y)
                    k_r = X_r.shape[1]
                    k_u = X_u.shape[1]
                    if n_obs <= k_u + 1:
                        continue
                    # OLS residuals
                    try:
                        beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
                        beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        continue
                    rss_r = float(np.sum((y - X_r @ beta_r) ** 2))
                    rss_u = float(np.sum((y - X_u @ beta_u) ** 2))
                    df1 = k_u - k_r
                    df2 = n_obs - k_u
                    if df1 <= 0 or df2 <= 0 or rss_u < 1e-15:
                        continue
                    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
                    p_val = float(1.0 - stats.f.cdf(f_stat, df1, df2))
                    if p_val < best_p:
                        best_p = p_val
                        best_lag = lag
                        best_f = float(f_stat)
                key = f"{i}->{j}"
                results[key] = {
                    "f_stat": best_f,
                    "p_value": best_p,
                    "best_lag": best_lag,
                    "reject": best_p < self.significance_level,
                }
        return results

    # ------------------------------------------------------------------
    # DCC-GARCH–style dynamic correlation (simplified EWMA version)
    # ------------------------------------------------------------------

    def dynamic_conditional_correlation(
        self, prices: np.ndarray, decay: float = 0.94
    ) -> Dict[str, Any]:
        """Exponentially weighted dynamic conditional correlation.

        Uses an EWMA covariance estimator (RiskMetrics-style) for speed.
        """
        diffs = np.diff(prices, axis=0)
        T, N = diffs.shape
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

        # Initialise with sample covariance
        cov = np.cov(diffs[:min(50, T)].T)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        dcc_series = {f"{i}-{j}": [] for i, j in pairs}
        for t in range(T):
            x = diffs[t].reshape(-1, 1)
            cov = decay * cov + (1 - decay) * (x @ x.T)
            d = np.sqrt(np.diag(cov))
            d = np.where(d < 1e-15, 1.0, d)
            corr_t = cov / np.outer(d, d)
            for i, j in pairs:
                dcc_series[f"{i}-{j}"].append(float(corr_t[i, j]))

        summary: Dict[str, Any] = {"series": {}}
        for key in dcc_series:
            arr = np.array(dcc_series[key])
            summary["series"][key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        summary["decay"] = decay
        return summary

    # ------------------------------------------------------------------
    # Mutual information
    # ------------------------------------------------------------------

    def mutual_information(
        self, prices: np.ndarray, n_bins: int = 50
    ) -> np.ndarray:
        """Mutual information matrix (using histogram estimator)."""
        diffs = np.diff(prices, axis=0)
        N = diffs.shape[1]
        mi = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                if i == j:
                    h_x = self._entropy_1d(diffs[:, i], n_bins)
                    mi[i, j] = h_x
                else:
                    mi_val = self._mi_2d(diffs[:, i], diffs[:, j], n_bins)
                    mi[i, j] = mi[j, i] = mi_val
        return mi

    @staticmethod
    def _entropy_1d(x: np.ndarray, n_bins: int) -> float:
        counts, _ = np.histogram(x, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    @staticmethod
    def _mi_2d(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
        hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
        pxy = hist_2d / hist_2d.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        # MI = sum p(x,y) log(p(x,y) / (p(x)*p(y)))
        mi = 0.0
        for ix in range(n_bins):
            for iy in range(n_bins):
                if pxy[ix, iy] > 0 and px[ix] > 0 and py[iy] > 0:
                    mi += pxy[ix, iy] * np.log(pxy[ix, iy] / (px[ix] * py[iy]))
        return float(mi)

    # ------------------------------------------------------------------
    # Cross-correlation function
    # ------------------------------------------------------------------

    def cross_correlation_function(
        self, prices: np.ndarray, max_lag: int = 20
    ) -> Dict[str, np.ndarray]:
        """Cross-correlation functions for all player pairs."""
        diffs = np.diff(prices, axis=0)
        N = diffs.shape[1]
        lags = np.arange(-max_lag, max_lag + 1)
        result: Dict[str, np.ndarray] = {"lags": lags}
        for i in range(N):
            for j in range(i + 1, N):
                ccf_vals = np.empty(len(lags))
                sx = diffs[:, i] - diffs[:, i].mean()
                sy = diffs[:, j] - diffs[:, j].mean()
                norm = np.sqrt(np.sum(sx ** 2) * np.sum(sy ** 2))
                if norm < 1e-15:
                    ccf_vals[:] = 0.0
                else:
                    for idx, lag in enumerate(lags):
                        if lag >= 0:
                            a = sx[: len(sx) - lag] if lag > 0 else sx
                            b = sy[lag:] if lag > 0 else sy
                        else:
                            a = sx[-lag:]
                            b = sy[: len(sy) + lag]
                        ccf_vals[idx] = float(np.sum(a * b) / norm)
                result[f"{i}-{j}"] = ccf_vals
        return result

    # ------------------------------------------------------------------
    # Spectral coherence
    # ------------------------------------------------------------------

    def coherence(self, prices: np.ndarray) -> Dict[str, Any]:
        """Spectral coherence between price series (Welch estimator)."""
        diffs = np.diff(prices, axis=0)
        N = diffs.shape[1]
        T = diffs.shape[0]
        nperseg = min(256, T // 2)
        if nperseg < 16:
            return {"error": "Series too short for coherence analysis"}

        result: Dict[str, Any] = {}
        for i in range(N):
            for j in range(i + 1, N):
                freqs, coh = signal.coherence(
                    diffs[:, i], diffs[:, j], nperseg=nperseg
                )
                result[f"{i}-{j}"] = {
                    "mean_coherence": float(coh.mean()),
                    "max_coherence": float(coh.max()),
                    "peak_frequency": float(freqs[np.argmax(coh)]),
                    "frequencies": freqs.tolist(),
                    "coherence": coh.tolist(),
                }
        return result

    # ------------------------------------------------------------------
    # Correlation change detection
    # ------------------------------------------------------------------

    def correlation_change_detection(
        self,
        prices: np.ndarray,
        window1: int = 10000,
        window2: int = 10000,
    ) -> Dict[str, Any]:
        """Test whether correlation structure changes between two halves.

        Splits the data at the midpoint (or uses *window1*/*window2*) and
        tests for equality of correlation matrices via a Box M-type
        statistic.
        """
        diffs = np.diff(prices, axis=0)
        T, N = diffs.shape
        split = min(window1, T // 2)
        d1 = diffs[:split]
        d2 = diffs[split : split + min(window2, T - split)]

        if len(d1) < N + 2 or len(d2) < N + 2:
            return {"error": "Insufficient data for change detection"}

        C1 = np.corrcoef(d1.T)
        C2 = np.corrcoef(d2.T)
        n1 = len(d1)
        n2 = len(d2)

        # Fisher z-transform each upper-triangle element and compare
        upper = np.triu_indices(N, k=1)
        z1 = np.arctanh(np.clip(C1[upper], -0.999, 0.999))
        z2 = np.arctanh(np.clip(C2[upper], -0.999, 0.999))
        se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
        z_diff = (z1 - z2) / se
        p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_diff)))

        # Combine with Bonferroni
        n_tests = len(p_values)
        reject_any = bool(np.any(p_values < self.significance_level / n_tests))

        # Frobenius distance
        frob = float(np.linalg.norm(C1 - C2, "fro"))

        return {
            "frobenius_distance": frob,
            "pairwise_z_stats": z_diff.tolist(),
            "pairwise_p_values": p_values.tolist(),
            "bonferroni_reject": reject_any,
            "bonferroni_alpha": self.significance_level / n_tests,
            "window1_corr": C1.tolist(),
            "window2_corr": C2.tolist(),
            "n_window1": n1,
            "n_window2": len(d2),
        }
