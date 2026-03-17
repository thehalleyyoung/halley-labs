"""Price trajectory analysis for collusion detection.

Provides tools for analysing individual and aggregate price trajectories,
including supra-competitive ratio, persistence, dispersion, convergence
speed, cycle detection (spectral), asymmetric adjustment (rockets &
feathers), leadership detection, stickiness and phase detection.
"""

from __future__ import annotations

import numpy as np
from scipy import stats, signal
from typing import Optional, Dict, List, Tuple, Any


class PriceAnalyzer:
    """Analyzes price trajectories for signs of collusion."""

    def __init__(
        self,
        nash_price: float,
        monopoly_price: float,
        marginal_cost: float = 0.0,
    ) -> None:
        self.nash_price = nash_price
        self.monopoly_price = monopoly_price
        self.marginal_cost = marginal_cost

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def analyze(self, prices: np.ndarray) -> Dict[str, Any]:
        """Full price analysis.

        Parameters
        ----------
        prices : ndarray of shape ``(num_rounds,)`` or ``(num_rounds, num_players)``
        """
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)

        result: Dict[str, Any] = {
            "supra_competitive_ratio": self.supra_competitive_ratio(prices),
            "relative_price_level": self.relative_price_level(prices),
            "price_stickiness": self.price_stickiness(prices),
            "price_dispersion": self.price_dispersion(prices),
            "convergence_speed": self.convergence_speed(prices),
            "price_cycles": self.price_cycles(prices),
            "asymmetric_adjustment": self.asymmetric_adjustment(prices),
            "variance_ratio": self.variance_ratio_test(prices),
        }
        if prices.shape[1] >= 2:
            result["price_leadership"] = self.price_leadership(prices)
        result["phase_detection"] = self.phase_detection(prices)
        result["persistence"] = {
            "mean_persistence": float(self.price_persistence(prices).mean())
        }
        return result

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def supra_competitive_ratio(self, prices: np.ndarray) -> float:
        """Fraction of prices above Nash equilibrium."""
        prices = np.asarray(prices, dtype=float)
        return float(np.mean(prices > self.nash_price))

    def price_persistence(self, prices: np.ndarray, window: int = 100) -> np.ndarray:
        """Measure how long prices stay at each level.

        Returns an array of run-lengths (number of consecutive periods
        where the rounded price does not change), computed on the
        mean-across-players series.
        """
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 2:
            series = prices.mean(axis=1)
        else:
            series = prices.ravel()

        # Discretise to *window*-th of the price range to measure persistence
        prange = series.max() - series.min()
        if prange < 1e-12:
            return np.array([len(series)])

        bins = np.round(series / (prange / window))
        run_lengths: List[int] = []
        current_len = 1
        for i in range(1, len(bins)):
            if bins[i] == bins[i - 1]:
                current_len += 1
            else:
                run_lengths.append(current_len)
                current_len = 1
        run_lengths.append(current_len)
        return np.array(run_lengths)

    def price_dispersion(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute price dispersion measures across players."""
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)
        # Per-round dispersion, then average
        if prices.shape[1] == 1:
            flat = prices.ravel()
            return {
                "std": float(np.std(flat)),
                "cv": float(np.std(flat) / np.mean(flat)) if np.mean(flat) != 0 else 0.0,
                "range": float(np.ptp(flat)),
                "iqr": float(np.subtract(*np.percentile(flat, [75, 25]))),
            }
        per_round_std = prices.std(axis=1)
        per_round_mean = prices.mean(axis=1)
        safe_mean = np.where(np.abs(per_round_mean) < 1e-12, 1.0, per_round_mean)
        cv = per_round_std / safe_mean
        per_round_range = prices.max(axis=1) - prices.min(axis=1)
        per_round_iqr = np.percentile(prices, 75, axis=1) - np.percentile(prices, 25, axis=1)
        return {
            "std": float(per_round_std.mean()),
            "cv": float(cv.mean()),
            "range": float(per_round_range.mean()),
            "iqr": float(per_round_iqr.mean()),
        }

    def convergence_speed(self, prices: np.ndarray, cv_threshold: float = 0.01, window: int = 200) -> Optional[int]:
        """Detect convergence and return number of rounds to converge.

        Returns ``None`` if the series never converges.
        """
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 2:
            series = prices.mean(axis=1)
        else:
            series = prices.ravel()
        n = len(series)
        if n < window:
            return None
        for start in range(0, n - window + 1):
            seg = series[start : start + window]
            mu = seg.mean()
            if abs(mu) < 1e-12:
                continue
            cv = seg.std() / abs(mu)
            if cv < cv_threshold:
                return start
        return None

    def price_cycles(self, prices: np.ndarray) -> Dict[str, Any]:
        """Detect price cycles using spectral analysis (Welch's method)."""
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 2:
            series = prices.mean(axis=1)
        else:
            series = prices.ravel()

        if len(series) < 64:
            return {"dominant_period": None, "spectral_power": 0.0, "has_cycle": False}

        # Detrend first
        series_detrended = signal.detrend(series, type="linear")
        nperseg = min(256, len(series_detrended) // 2)
        if nperseg < 16:
            return {"dominant_period": None, "spectral_power": 0.0, "has_cycle": False}
        freqs, psd = signal.welch(series_detrended, nperseg=nperseg)
        # Ignore DC component
        freqs = freqs[1:]
        psd = psd[1:]
        if len(psd) == 0:
            return {"dominant_period": None, "spectral_power": 0.0, "has_cycle": False}

        peak_idx = int(np.argmax(psd))
        peak_freq = freqs[peak_idx]
        peak_power = float(psd[peak_idx])
        total_power = float(psd.sum())
        dominant_period = float(1.0 / peak_freq) if peak_freq > 0 else None

        # Cycle significance: peak power > 3× mean power
        mean_power = total_power / len(psd)
        has_cycle = peak_power > 3.0 * mean_power

        return {
            "dominant_period": dominant_period,
            "peak_frequency": float(peak_freq) if peak_freq > 0 else None,
            "spectral_power": peak_power,
            "total_power": total_power,
            "concentration_ratio": peak_power / total_power if total_power > 0 else 0.0,
            "has_cycle": bool(has_cycle),
        }

    def asymmetric_adjustment(self, prices: np.ndarray) -> Dict[str, float]:
        """Test for asymmetric price adjustment (rockets & feathers).

        Returns average magnitude of positive vs negative changes, and the
        asymmetry ratio.
        """
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 2:
            series = prices.mean(axis=1)
        else:
            series = prices.ravel()

        changes = np.diff(series)
        positive = changes[changes > 0]
        negative = changes[changes < 0]

        avg_up = float(positive.mean()) if len(positive) > 0 else 0.0
        avg_down = float(np.abs(negative).mean()) if len(negative) > 0 else 0.0
        n_up = len(positive)
        n_down = len(negative)

        # Asymmetry ratio: >1 means prices rise faster than they fall
        if avg_down > 1e-12:
            asymmetry_ratio = avg_up / avg_down
        else:
            asymmetry_ratio = float("inf") if avg_up > 0 else 1.0

        # Wilcoxon rank-sum test on magnitudes
        if n_up >= 5 and n_down >= 5:
            stat, p_value = stats.mannwhitneyu(
                positive, np.abs(negative), alternative="two-sided"
            )
        else:
            stat, p_value = float("nan"), 1.0

        return {
            "avg_increase": avg_up,
            "avg_decrease": avg_down,
            "n_increases": n_up,
            "n_decreases": n_down,
            "asymmetry_ratio": asymmetry_ratio,
            "mannwhitney_stat": float(stat),
            "mannwhitney_p": float(p_value),
        }

    def price_leadership(self, prices: np.ndarray, max_lag: int = 5) -> Dict[str, Any]:
        """Detect price leadership patterns using cross-correlation on changes."""
        prices = np.asarray(prices, dtype=float)
        if prices.ndim != 2 or prices.shape[1] < 2:
            return {"leader": None, "details": "Need at least 2 players"}

        changes = np.diff(prices, axis=0)
        n_players = changes.shape[1]
        leadership: Dict[str, Any] = {"pairs": {}, "leader": None}

        best_lead = 0.0
        best_leader = -1
        for i in range(n_players):
            for j in range(n_players):
                if i == j:
                    continue
                max_corr = 0.0
                best_lag = 0
                for lag in range(1, max_lag + 1):
                    if lag >= len(changes):
                        break
                    x = changes[:-lag, i]
                    y = changes[lag:, j]
                    if len(x) < 10:
                        continue
                    r, _ = stats.pearsonr(x, y)
                    if abs(r) > abs(max_corr):
                        max_corr = float(r)
                        best_lag = lag
                pair_key = f"{i}->{j}"
                leadership["pairs"][pair_key] = {
                    "correlation": max_corr,
                    "lag": best_lag,
                }
                if max_corr > best_lead:
                    best_lead = max_corr
                    best_leader = i

        leadership["leader"] = best_leader
        leadership["max_lead_correlation"] = best_lead
        return leadership

    def price_stickiness(self, prices: np.ndarray) -> float:
        """Fraction of zero price changes (averaged across players)."""
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)
        changes = np.diff(prices, axis=0)
        return float(np.mean(np.abs(changes) < 1e-12))

    def relative_price_level(self, prices: np.ndarray) -> float:
        """Where prices lie between Nash and monopoly. 0 = Nash, 1 = monopoly."""
        prices = np.asarray(prices, dtype=float)
        avg = float(prices.mean())
        span = self.monopoly_price - self.nash_price
        if abs(span) < 1e-12:
            return 0.0
        return float(np.clip((avg - self.nash_price) / span, 0.0, 1.0))

    def phase_detection(
        self, prices: np.ndarray, n_phases: int = 3
    ) -> Dict[str, Any]:
        """Detect phases (exploration, learning, converged) via variance breakpoints.

        Splits the trajectory into *n_phases* segments by minimising total
        within-segment variance.
        """
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 2:
            series = prices.mean(axis=1)
        else:
            series = prices.ravel()

        n = len(series)
        if n < n_phases * 10:
            return {"phases": [{"start": 0, "end": n, "label": "unknown"}]}

        # Simple dynamic programming for optimal 1-D segmentation
        # Cost of segment [i, j) = variance(series[i:j]) * (j - i)
        prefix_sum = np.concatenate([[0.0], np.cumsum(series)])
        prefix_sq = np.concatenate([[0.0], np.cumsum(series ** 2)])

        def seg_cost(i: int, j: int) -> float:
            length = j - i
            if length <= 1:
                return 0.0
            s = prefix_sum[j] - prefix_sum[i]
            sq = prefix_sq[j] - prefix_sq[i]
            return float(sq - s * s / length)

        INF = float("inf")
        # dp[k][j] = min cost of partitioning series[0:j] into k segments
        dp = [[INF] * (n + 1) for _ in range(n_phases + 1)]
        split = [[0] * (n + 1) for _ in range(n_phases + 1)]
        dp[0][0] = 0.0
        for k in range(1, n_phases + 1):
            min_start = k - 1
            for j in range(k, n + 1):
                for i in range(k - 1, j):
                    cost = dp[k - 1][i] + seg_cost(i, j)
                    if cost < dp[k][j]:
                        dp[k][j] = cost
                        split[k][j] = i

        # Back-track
        boundaries = []
        j = n
        for k in range(n_phases, 0, -1):
            boundaries.append(split[k][j])
            j = split[k][j]
        boundaries.reverse()
        boundaries.append(n)

        phase_labels = ["exploration", "learning", "converged"]
        if n_phases > 3:
            phase_labels += [f"phase_{i}" for i in range(3, n_phases)]

        phases: List[Dict[str, Any]] = []
        for idx in range(n_phases):
            start = boundaries[idx]
            end = boundaries[idx + 1]
            seg = series[start:end]
            label = phase_labels[idx] if idx < len(phase_labels) else f"phase_{idx}"
            phases.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "label": label,
                    "mean_price": float(seg.mean()) if len(seg) > 0 else float("nan"),
                    "std_price": float(seg.std()) if len(seg) > 0 else float("nan"),
                    "length": int(end - start),
                }
            )

        return {"n_phases": n_phases, "phases": phases}

    def variance_ratio_test(
        self,
        prices: np.ndarray,
        periods: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Lo-MacKinlay variance ratio test for each holding period.

        Under a random walk H0, VR(q) = Var(q-period return) / (q * Var(1-period return)) ≈ 1.
        """
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 2:
            series = prices.mean(axis=1)
        else:
            series = prices.ravel()

        if periods is None:
            periods = [2, 5, 10, 20]

        log_prices = np.log(np.clip(series, 1e-12, None))
        returns_1 = np.diff(log_prices)
        n = len(returns_1)
        var_1 = float(np.var(returns_1, ddof=1))

        results: Dict[str, float] = {}
        for q in periods:
            if q >= n:
                continue
            returns_q = log_prices[q:] - log_prices[:-q]
            var_q = float(np.var(returns_q, ddof=1))
            vr = var_q / (q * var_1) if var_1 > 1e-15 else float("nan")
            # Asymptotic z-statistic under IID
            se_vr = np.sqrt(2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * n))
            z = (vr - 1.0) / se_vr if se_vr > 1e-15 else 0.0
            p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
            results[f"VR({q})"] = float(vr)
            results[f"z({q})"] = float(z)
            results[f"p({q})"] = float(p_value)
        return results

    # ------------------------------------------------------------------
    # Convenience: markup / margin metrics
    # ------------------------------------------------------------------

    def markup(self, prices: np.ndarray) -> float:
        """Average Lerner index: (P - MC) / P."""
        prices = np.asarray(prices, dtype=float)
        avg = float(prices.mean())
        if abs(avg) < 1e-12:
            return 0.0
        return (avg - self.marginal_cost) / avg

    def profit_ratio(self, prices: np.ndarray) -> float:
        """Ratio of observed profit to monopoly profit (single-product)."""
        avg = float(np.asarray(prices).mean())
        obs_profit = avg - self.marginal_cost
        mono_profit = self.monopoly_price - self.marginal_cost
        if mono_profit <= 0:
            return 0.0
        return float(np.clip(obs_profit / mono_profit, 0.0, 1.0))

    def price_volatility(self, prices: np.ndarray, window: int = 50) -> np.ndarray:
        """Rolling standard deviation of mean-across-players price."""
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 2:
            series = prices.mean(axis=1)
        else:
            series = prices.ravel()
        n = len(series)
        if n < window:
            return np.array([float(series.std())])
        out = np.empty(n - window + 1)
        for i in range(len(out)):
            out[i] = series[i : i + window].std()
        return out

    def mean_reversion_half_life(self, prices: np.ndarray) -> Optional[float]:
        """Estimate half-life of mean reversion using AR(1) on demeaned prices."""
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 2:
            series = prices.mean(axis=1)
        else:
            series = prices.ravel()
        demeaned = series - series.mean()
        y = demeaned[1:]
        x = demeaned[:-1]
        if len(x) < 10 or np.std(x) < 1e-12:
            return None
        slope, _, _, _, _ = stats.linregress(x, y)
        if slope >= 1.0 or slope <= 0.0:
            return None
        return float(-np.log(2) / np.log(slope))
