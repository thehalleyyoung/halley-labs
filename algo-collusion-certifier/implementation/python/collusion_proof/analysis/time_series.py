"""Time series analysis for collusion detection.

Provides autocorrelation, stationarity tests, trend detection,
and changepoint detection methods for price trajectories.
"""

import numpy as np
from scipy import stats, signal
from typing import Dict, List, Optional, Tuple, Any


class TimeSeriesAnalyzer:
    """Time series analysis tools for price data."""

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    # ------------------------------------------------------------------
    # Autocorrelation
    # ------------------------------------------------------------------

    def autocorrelation(self, data: np.ndarray,
                        max_lag: Optional[int] = None) -> np.ndarray:
        """Compute autocorrelation function (ACF).

        Returns an array of length *max_lag* + 1 where index *k* is the
        autocorrelation at lag *k*.  Lag 0 is always 1.0.
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        if max_lag is None:
            max_lag = min(n // 2, n - 1)
        max_lag = min(max_lag, n - 1)

        mean = data.mean()
        var = np.sum((data - mean) ** 2)
        if var == 0:
            return np.ones(max_lag + 1)

        acf = np.empty(max_lag + 1)
        for k in range(max_lag + 1):
            acf[k] = np.sum((data[:n - k] - mean) * (data[k:] - mean)) / var
        return acf

    def partial_autocorrelation(self, data: np.ndarray,
                                max_lag: Optional[int] = None) -> np.ndarray:
        """Compute partial ACF using Durbin-Levinson recursion.

        Returns an array of length *max_lag* + 1 (index 0 is 1.0).
        """
        acf = self.autocorrelation(data, max_lag)
        m = len(acf) - 1
        pacf = np.zeros(m + 1)
        pacf[0] = 1.0
        if m == 0:
            return pacf

        phi = np.zeros((m + 1, m + 1))
        phi[1, 1] = acf[1]
        pacf[1] = acf[1]

        for k in range(2, m + 1):
            num = acf[k] - sum(phi[k - 1, j] * acf[k - j] for j in range(1, k))
            den = 1.0 - sum(phi[k - 1, j] * acf[j] for j in range(1, k))
            if den == 0:
                break
            phi[k, k] = num / den
            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
            pacf[k] = phi[k, k]

        return pacf

    # ------------------------------------------------------------------
    # Stationarity tests
    # ------------------------------------------------------------------

    def adf_test(self, data: np.ndarray,
                 max_lag: Optional[int] = None) -> Dict[str, Any]:
        """Augmented Dickey-Fuller test for unit root (non-stationarity).

        H0: data has unit root (non-stationary)
        H1: data is stationary

        Implements the ADF regression:
            Δy_t = α + γ y_{t-1} + Σ β_i Δy_{t-i} + ε_t
        and tests γ = 0.
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        if max_lag is None:
            max_lag = int(np.floor((n - 1) ** (1.0 / 3.0)))
        max_lag = min(max_lag, n // 2 - 2)

        dy = np.diff(data)
        y_lag = data[max_lag:-1]
        dy_dep = dy[max_lag:]
        nobs = len(dy_dep)

        x_mat = np.column_stack([
            np.ones(nobs),
            y_lag,
        ] + [dy[max_lag - i - 1: -i - 1] for i in range(max_lag)])

        coeffs, residuals, _, _ = np.linalg.lstsq(x_mat, dy_dep, rcond=None)
        gamma = coeffs[1]

        fitted = x_mat @ coeffs
        resid = dy_dep - fitted
        sigma2 = np.sum(resid ** 2) / (nobs - x_mat.shape[1])
        cov_matrix = sigma2 * np.linalg.inv(x_mat.T @ x_mat)
        se_gamma = np.sqrt(cov_matrix[1, 1])
        adf_stat = gamma / se_gamma

        # Approximate p-value via MacKinnon critical values (n >= 25)
        crit_values = {1: -3.43, 5: -2.86, 10: -2.57}
        if adf_stat < crit_values[1]:
            p_value = 0.005
        elif adf_stat < crit_values[5]:
            p_value = 0.025
        elif adf_stat < crit_values[10]:
            p_value = 0.075
        else:
            p_value = min(1.0, stats.norm.sf(adf_stat))

        return {
            "test_statistic": float(adf_stat),
            "p_value": float(p_value),
            "lags_used": max_lag,
            "nobs": nobs,
            "critical_values": crit_values,
            "stationary": adf_stat < crit_values[5],
        }

    def kpss_test(self, data: np.ndarray,
                  regression: str = "c") -> Dict[str, Any]:
        """KPSS test for stationarity.

        H0: data is stationary
        H1: data has unit root

        Args:
            regression: "c" (level stationarity) or "ct" (trend stationarity).
        """
        data = np.asarray(data, dtype=float)
        n = len(data)

        if regression == "ct":
            t_idx = np.arange(1, n + 1, dtype=float)
            x = np.column_stack([np.ones(n), t_idx])
        else:
            x = np.ones((n, 1))

        coeffs = np.linalg.lstsq(x, data, rcond=None)[0]
        resid = data - x @ coeffs

        cumsum = np.cumsum(resid)
        s2 = np.sum(resid ** 2) / n

        # Newey-West bandwidth
        num_lags = int(np.ceil(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        for lag in range(1, num_lags + 1):
            weight = 1.0 - lag / (num_lags + 1.0)
            s2 += 2.0 * weight * np.sum(resid[lag:] * resid[:-lag]) / n

        if s2 <= 0:
            s2 = np.sum(resid ** 2) / n

        kpss_stat = np.sum(cumsum ** 2) / (n ** 2 * s2)

        if regression == "ct":
            crit = {10: 0.119, 5: 0.146, 2.5: 0.176, 1: 0.216}
        else:
            crit = {10: 0.347, 5: 0.463, 2.5: 0.574, 1: 0.739}

        if kpss_stat < crit[10]:
            p_value = 0.15
        elif kpss_stat < crit[5]:
            p_value = 0.075
        elif kpss_stat < crit[1]:
            p_value = 0.025
        else:
            p_value = 0.005

        return {
            "test_statistic": float(kpss_stat),
            "p_value": float(p_value),
            "regression": regression,
            "lags_used": num_lags,
            "critical_values": crit,
            "stationary": kpss_stat < crit[5],
        }

    def stationarity_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Combined stationarity analysis using ADF and KPSS.

        Interpretation matrix:
          ADF rejects & KPSS does not reject → stationary
          ADF does not reject & KPSS rejects → non-stationary
          Both reject → trend-stationary
          Neither rejects → inconclusive
        """
        adf = self.adf_test(data)
        kpss = self.kpss_test(data)

        adf_reject = adf["stationary"]
        kpss_reject = not kpss["stationary"]

        if adf_reject and not kpss_reject:
            conclusion = "stationary"
        elif not adf_reject and kpss_reject:
            conclusion = "non-stationary"
        elif adf_reject and kpss_reject:
            conclusion = "trend-stationary"
        else:
            conclusion = "inconclusive"

        return {
            "adf": adf,
            "kpss": kpss,
            "conclusion": conclusion,
        }

    # ------------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------------

    def trend_test(self, data: np.ndarray) -> Dict[str, Any]:
        """Mann-Kendall trend test.

        Returns test statistic S, normalised Z, p-value, and trend direction.
        """
        data = np.asarray(data, dtype=float)
        n = len(data)

        s = 0
        for i in range(n - 1):
            diff = data[i + 1:] - data[i]
            s += int(np.sum(np.sign(diff)))

        var_s = n * (n - 1) * (2 * n + 5) / 18.0

        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0.0

        p_value = 2.0 * stats.norm.sf(abs(z))

        if p_value < self.alpha:
            trend = "increasing" if z > 0 else "decreasing"
        else:
            trend = "no trend"

        return {
            "S": s,
            "Z": float(z),
            "p_value": float(p_value),
            "trend": trend,
            "significant": p_value < self.alpha,
        }

    def detrend(self, data: np.ndarray, method: str = "linear") -> np.ndarray:
        """Remove trend from data.

        Args:
            method: "linear" (OLS line), "mean" (subtract mean), or
                    "difference" (first difference).
        """
        data = np.asarray(data, dtype=float)
        if method == "linear":
            return signal.detrend(data, type="linear")
        elif method == "mean":
            return data - data.mean()
        elif method == "difference":
            return np.diff(data)
        else:
            raise ValueError(f"Unknown detrend method: {method}")

    # ------------------------------------------------------------------
    # Seasonal decomposition
    # ------------------------------------------------------------------

    def seasonal_decompose(self, data: np.ndarray,
                           period: int) -> Dict[str, np.ndarray]:
        """Simple additive seasonal decomposition.

        Returns dict with 'trend', 'seasonal', and 'residual' components.
        """
        data = np.asarray(data, dtype=float)
        n = len(data)

        # Moving-average trend
        kernel = np.ones(period) / period
        trend = np.convolve(data, kernel, mode="same")
        # Fix edges with partial windows
        half = period // 2
        for i in range(half):
            trend[i] = data[:i + half + 1].mean()
            trend[n - 1 - i] = data[n - 1 - i - half:].mean()

        detrended = data - trend

        # Average seasonal indices
        seasonal = np.zeros(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            seasonal_mean = detrended[indices].mean()
            seasonal[indices] = seasonal_mean

        residual = data - trend - seasonal

        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
        }

    # ------------------------------------------------------------------
    # Changepoint detection
    # ------------------------------------------------------------------

    def cusum_test(self, data: np.ndarray) -> Dict[str, Any]:
        """CUSUM test for structural change.

        Returns test statistic, critical value, estimated breakpoint, and
        an approximate p-value based on Brownian bridge asymptotics.
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        cumsum = np.cumsum(data - data.mean())
        scaled = cumsum / (np.std(data, ddof=1) * np.sqrt(n))
        test_stat = np.max(np.abs(scaled))
        breakpoint = int(np.argmax(np.abs(scaled)))

        # Approximate critical value from Brownian bridge supremum
        critical_value = 1.358  # 5 % level for sup|B(t)|
        # Approximate p-value using Kolmogorov-Smirnov-style formula
        p_value = 2.0 * np.exp(-2.0 * test_stat ** 2)
        p_value = float(np.clip(p_value, 0.0, 1.0))

        return {
            "test_statistic": float(test_stat),
            "critical_value": critical_value,
            "breakpoint": breakpoint,
            "p_value": p_value,
            "significant": test_stat > critical_value,
        }

    def cusum_changepoints(self, data: np.ndarray,
                           threshold: float = 1.0) -> List[int]:
        """Detect multiple changepoints using recursive CUSUM.

        Recursively splits segments where the CUSUM statistic exceeds
        *threshold*, returning a sorted list of breakpoint indices.
        """
        data = np.asarray(data, dtype=float)
        changepoints: List[int] = []
        self._recursive_cusum(data, 0, len(data), threshold, changepoints)
        changepoints.sort()
        return changepoints

    def _recursive_cusum(self, data: np.ndarray, start: int, end: int,
                         threshold: float,
                         changepoints: List[int]) -> None:
        segment = data[start:end]
        if len(segment) < 10:
            return
        cumsum = np.cumsum(segment - segment.mean())
        std = np.std(segment, ddof=1)
        if std == 0:
            return
        scaled = np.abs(cumsum) / (std * np.sqrt(len(segment)))
        max_val = scaled.max()
        if max_val > threshold:
            bp = int(np.argmax(scaled)) + start
            changepoints.append(bp)
            self._recursive_cusum(data, start, bp, threshold, changepoints)
            self._recursive_cusum(data, bp, end, threshold, changepoints)

    def pelt_changepoints(self, data: np.ndarray,
                          penalty: Optional[float] = None) -> List[int]:
        """PELT (Pruned Exact Linear Time) changepoint detection.

        Minimises  Σ C(segment_i) + penalty × num_changepoints
        where C is the Gaussian negative log-likelihood cost.
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        if penalty is None:
            penalty = 2.0 * np.log(n)

        cum_sum = np.zeros(n + 1)
        cum_sq = np.zeros(n + 1)
        for i in range(n):
            cum_sum[i + 1] = cum_sum[i] + data[i]
            cum_sq[i + 1] = cum_sq[i] + data[i] ** 2

        def segment_cost(s: int, t: int) -> float:
            length = t - s
            if length <= 0:
                return 0.0
            seg_sum = cum_sum[t] - cum_sum[s]
            seg_sq = cum_sq[t] - cum_sq[s]
            return seg_sq - seg_sum ** 2 / length

        f = np.full(n + 1, np.inf)
        f[0] = -penalty
        cp = [[] for _ in range(n + 1)]
        candidates = [0]

        for t_star in range(1, n + 1):
            best_cost = np.inf
            best_s = 0
            for s in candidates:
                cost = f[s] + segment_cost(s, t_star) + penalty
                if cost < best_cost:
                    best_cost = cost
                    best_s = s
            f[t_star] = best_cost
            cp[t_star] = cp[best_s] + [best_s]

            # Pruning step
            candidates = [
                s for s in candidates
                if f[s] + segment_cost(s, t_star) <= f[t_star]
            ]
            candidates.append(t_star)

        result = sorted(set(cp[n]) - {0})
        return result

    def binary_segmentation(self, data: np.ndarray,
                            max_changepoints: int = 5,
                            min_segment_length: int = 100) -> List[int]:
        """Binary segmentation changepoint detection.

        Greedily splits at the point that maximises the CUSUM statistic,
        up to *max_changepoints* times.
        """
        data = np.asarray(data, dtype=float)
        changepoints: List[int] = []
        segments: List[Tuple[int, int]] = [(0, len(data))]

        for _ in range(max_changepoints):
            best_stat = -np.inf
            best_bp = -1
            best_seg_idx = -1

            for seg_idx, (s, e) in enumerate(segments):
                seg = data[s:e]
                if len(seg) < 2 * min_segment_length:
                    continue
                cumsum = np.cumsum(seg - seg.mean())
                std = np.std(seg, ddof=1)
                if std == 0:
                    continue
                scaled = np.abs(cumsum) / (std * np.sqrt(len(seg)))
                stat_max = scaled[min_segment_length:-min_segment_length].max() \
                    if len(seg) > 2 * min_segment_length else scaled.max()
                local_bp = int(np.argmax(
                    scaled[min_segment_length:-min_segment_length]
                )) + min_segment_length if len(seg) > 2 * min_segment_length \
                    else int(np.argmax(scaled))
                if stat_max > best_stat:
                    best_stat = stat_max
                    best_bp = s + local_bp
                    best_seg_idx = seg_idx

            # Stop if statistic below Brownian bridge 5 % critical value
            if best_stat < 1.358 or best_seg_idx < 0:
                break

            changepoints.append(best_bp)
            s, e = segments.pop(best_seg_idx)
            segments.insert(best_seg_idx, (s, best_bp))
            segments.insert(best_seg_idx + 1, (best_bp, e))

        changepoints.sort()
        return changepoints

    def variance_changepoint(self, data: np.ndarray) -> Dict[str, Any]:
        """Test for change in variance (relevant for collusion transitions).

        Scans every candidate split point and picks the one that maximises
        the likelihood-ratio statistic comparing two-segment variances to
        the pooled variance.
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        min_seg = max(5, n // 20)

        total_var = np.var(data, ddof=1)
        if total_var == 0:
            return {
                "breakpoint": None, "test_statistic": 0.0,
                "p_value": 1.0, "significant": False,
                "var_before": 0.0, "var_after": 0.0,
            }

        log_total = n * np.log(total_var)

        best_stat = -np.inf
        best_bp = min_seg
        for bp in range(min_seg, n - min_seg):
            v1 = np.var(data[:bp], ddof=1)
            v2 = np.var(data[bp:], ddof=1)
            if v1 <= 0 or v2 <= 0:
                continue
            log_seg = bp * np.log(v1) + (n - bp) * np.log(v2)
            lr = log_total - log_seg
            if lr > best_stat:
                best_stat = lr
                best_bp = bp

        # Approximate p-value via chi-squared(1) on 2 × LR
        p_value = float(stats.chi2.sf(2 * best_stat, df=1)) if best_stat > 0 else 1.0

        return {
            "breakpoint": best_bp,
            "test_statistic": float(best_stat),
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "var_before": float(np.var(data[:best_bp], ddof=1)),
            "var_after": float(np.var(data[best_bp:], ddof=1)),
        }

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def spectral_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Power spectral density analysis using Welch's method.

        Returns frequencies, PSD values, dominant frequency and its power.
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        nperseg = min(256, n)
        freqs, psd = signal.welch(data, fs=1.0, nperseg=nperseg)

        # Exclude DC component
        if len(freqs) > 1:
            peak_idx = np.argmax(psd[1:]) + 1
        else:
            peak_idx = 0

        return {
            "frequencies": freqs,
            "psd": psd,
            "dominant_frequency": float(freqs[peak_idx]),
            "dominant_power": float(psd[peak_idx]),
            "total_power": float(np.sum(psd)),
        }

    def dominant_period(self, data: np.ndarray) -> Optional[int]:
        """Find dominant period in the data (if any).

        Returns the integer period corresponding to the highest spectral
        peak, or None if no clear periodicity is detected.
        """
        spec = self.spectral_analysis(data)
        freq = spec["dominant_frequency"]
        if freq <= 0:
            return None
        period = int(round(1.0 / freq))
        # Sanity check: period should be between 2 and half the data length
        if period < 2 or period > len(data) // 2:
            return None
        # Check peak is at least 3× the median power
        median_power = float(np.median(spec["psd"][1:])) if len(spec["psd"]) > 1 else 0.0
        if median_power > 0 and spec["dominant_power"] < 3.0 * median_power:
            return None
        return period

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def regime_detection(self, data: np.ndarray,
                         n_regimes: int = 2) -> Dict[str, Any]:
        """Detect regimes using a simple hidden Markov-like approach.

        Fits a Gaussian mixture via EM and assigns each observation to the
        component with highest posterior probability.
        """
        data = np.asarray(data, dtype=float).reshape(-1)
        n = len(data)

        # Initialise with quantile-based splits
        quantiles = np.linspace(0, 1, n_regimes + 1)
        boundaries = np.quantile(data, quantiles)
        means = np.array([
            data[(data >= boundaries[i]) & (data < boundaries[i + 1])].mean()
            if np.any((data >= boundaries[i]) & (data < boundaries[i + 1]))
            else boundaries[i]
            for i in range(n_regimes)
        ])
        stds = np.full(n_regimes, np.std(data, ddof=1))
        weights = np.ones(n_regimes) / n_regimes

        # EM iterations
        responsibilities = np.zeros((n, n_regimes))
        for _ in range(100):
            # E-step
            for k in range(n_regimes):
                responsibilities[:, k] = weights[k] * stats.norm.pdf(
                    data, means[k], max(stds[k], 1e-10)
                )
            row_sums = responsibilities.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-300)
            responsibilities /= row_sums

            # M-step
            nk = responsibilities.sum(axis=0)
            for k in range(n_regimes):
                if nk[k] < 1e-10:
                    continue
                means[k] = np.dot(responsibilities[:, k], data) / nk[k]
                diff = data - means[k]
                stds[k] = np.sqrt(
                    np.dot(responsibilities[:, k], diff ** 2) / nk[k]
                )
                stds[k] = max(stds[k], 1e-10)
                weights[k] = nk[k] / n

        labels = np.argmax(responsibilities, axis=1)

        # Sort regimes by mean so label 0 is the lowest-mean regime
        order = np.argsort(means)
        means = means[order]
        stds = stds[order]
        weights = weights[order]
        label_map = {int(order[i]): i for i in range(n_regimes)}
        labels = np.array([label_map[l] for l in labels])

        return {
            "labels": labels,
            "means": means,
            "stds": stds,
            "weights": weights,
            "n_regimes": n_regimes,
        }

    # ------------------------------------------------------------------
    # Diagnostic tests
    # ------------------------------------------------------------------

    def ljung_box_test(self, data: np.ndarray,
                       max_lag: int = 10) -> Dict[str, Any]:
        """Ljung-Box portmanteau test for autocorrelation.

        H0: no autocorrelation up to lag *max_lag*.
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        acf = self.autocorrelation(data, max_lag)

        q_stat = 0.0
        for k in range(1, max_lag + 1):
            q_stat += acf[k] ** 2 / (n - k)
        q_stat *= n * (n + 2)

        df = max_lag
        p_value = float(stats.chi2.sf(q_stat, df))

        return {
            "test_statistic": float(q_stat),
            "p_value": p_value,
            "df": df,
            "significant": p_value < self.alpha,
            "acf_values": acf,
        }

    def half_life(self, data: np.ndarray) -> Optional[float]:
        """Estimate half-life of mean reversion via AR(1) fit.

        Fits  y_t = φ y_{t-1} + ε  and returns -ln(2)/ln(φ).
        Returns None if the series is not mean-reverting (φ ≥ 1 or φ ≤ 0).
        """
        data = np.asarray(data, dtype=float)
        y = data[1:]
        x = data[:-1].reshape(-1, 1)
        x_with_const = np.column_stack([np.ones(len(x)), x])
        coeffs = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        phi = coeffs[1]
        if phi <= 0 or phi >= 1.0:
            return None
        return -np.log(2) / np.log(phi)

    def hurst_exponent(self, data: np.ndarray) -> float:
        """Estimate Hurst exponent using R/S analysis.

        H < 0.5: mean reverting
        H ≈ 0.5: random walk
        H > 0.5: trending
        """
        data = np.asarray(data, dtype=float)
        n = len(data)

        # Use a range of sub-series lengths
        min_window = 10
        max_window = n // 2
        if max_window < min_window:
            return 0.5

        sizes = []
        rs_values = []
        window = min_window
        while window <= max_window:
            sizes.append(window)
            num_blocks = n // window
            rs_block = []
            for b in range(num_blocks):
                segment = data[b * window:(b + 1) * window]
                mean_seg = segment.mean()
                deviations = segment - mean_seg
                cumdev = np.cumsum(deviations)
                r = cumdev.max() - cumdev.min()
                s = np.std(segment, ddof=1)
                if s > 0:
                    rs_block.append(r / s)
            if rs_block:
                rs_values.append(np.mean(rs_block))
            else:
                rs_values.append(np.nan)
            window = int(window * 1.5)
            if window == sizes[-1]:
                window += 1

        log_n = np.log(np.array(sizes, dtype=float))
        log_rs = np.log(np.array(rs_values, dtype=float))
        valid = np.isfinite(log_rs)
        if valid.sum() < 2:
            return 0.5

        slope, _, _, _, _ = stats.linregress(log_n[valid], log_rs[valid])
        return float(np.clip(slope, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Comprehensive analysis
    # ------------------------------------------------------------------

    def full_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Run comprehensive time series analysis.

        Combines stationarity tests, trend detection, autocorrelation
        diagnostics, changepoint detection, and regime analysis.
        """
        data = np.asarray(data, dtype=float)
        n = len(data)

        stationarity = self.stationarity_analysis(data)
        trend = self.trend_test(data)
        acf = self.autocorrelation(data, min(20, n - 1))
        pacf = self.partial_autocorrelation(data, min(20, n - 1))
        ljung_box = self.ljung_box_test(data, min(10, n // 5))
        h_exp = self.hurst_exponent(data)
        hl = self.half_life(data)
        cusum = self.cusum_test(data)

        changepoints = self.pelt_changepoints(data)
        var_cp = self.variance_changepoint(data)
        spectral = self.spectral_analysis(data)
        period = self.dominant_period(data)
        regimes = self.regime_detection(data)

        return {
            "n": n,
            "mean": float(data.mean()),
            "std": float(data.std(ddof=1)),
            "stationarity": stationarity,
            "trend": trend,
            "acf": acf,
            "pacf": pacf,
            "ljung_box": ljung_box,
            "hurst_exponent": h_exp,
            "half_life": hl,
            "cusum": cusum,
            "changepoints": changepoints,
            "variance_changepoint": var_cp,
            "spectral": spectral,
            "dominant_period": period,
            "regimes": regimes,
        }
