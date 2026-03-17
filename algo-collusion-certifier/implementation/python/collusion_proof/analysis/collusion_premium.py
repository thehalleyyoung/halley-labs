"""Collusion Premium computation for algorithmic collusion detection.

The Collusion Premium (CP) measures how far observed prices are above the
competitive (Nash) level, normalized by the range between Nash and monopoly:

    CP = (p_observed - p_nash) / (p_monopoly - p_nash)

CP = 0 means competitive pricing, CP = 1 means monopoly pricing.
Values > 0 indicate supra-competitive pricing.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any, Callable


class CollusionPremiumCalculator:
    """Compute collusion premium with statistical inference."""

    def __init__(
        self,
        nash_price: float,
        monopoly_price: float,
        marginal_cost: float = 0.0,
        bootstrap_samples: int = 10000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ):
        if monopoly_price <= nash_price:
            raise ValueError("monopoly_price must exceed nash_price")

        self.nash_price = nash_price
        self.monopoly_price = monopoly_price
        self.marginal_cost = marginal_cost
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.rng = np.random.RandomState(random_state)
        self._price_range = monopoly_price - nash_price

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def compute(self, prices: np.ndarray) -> Dict[str, Any]:
        """Compute collusion premium with bootstrap confidence interval.

        Args:
            prices: Price array, shape ``(num_rounds,)`` or
                    ``(num_rounds, num_players)``.

        Returns:
            Dict with keys: premium, ci_lower, ci_upper, collusion_index,
            absolute_margin, mean_price, std_price, n_obs.
        """
        flat = self._flatten_prices(prices)
        mean_price = float(np.mean(flat))
        premium = (mean_price - self.nash_price) / self._price_range

        ci_lower, ci_upper = self._bootstrap_ci(
            flat, statistic=self._premium_statistic, method="percentile"
        )

        ci_clipped = (max(0.0, min(1.0, ci_lower)), max(0.0, min(1.0, ci_upper)))

        return {
            "premium": premium,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "collusion_index": max(0.0, min(1.0, premium)),
            "collusion_index_ci": ci_clipped,
            "absolute_margin": mean_price - self.nash_price,
            "mean_price": mean_price,
            "std_price": float(np.std(flat, ddof=1)) if len(flat) > 1 else 0.0,
            "n_obs": len(flat),
        }

    def compute_per_player(self, prices: np.ndarray) -> List[Dict[str, Any]]:
        """Compute collusion premium separately for each player.

        Args:
            prices: Shape ``(num_rounds, num_players)``.

        Returns:
            List of result dicts, one per player column.
        """
        prices = np.asarray(prices)
        if prices.ndim == 1:
            return [self.compute(prices)]
        results: List[Dict[str, Any]] = []
        for j in range(prices.shape[1]):
            res = self.compute(prices[:, j])
            res["player"] = j
            results.append(res)
        return results

    def compute_windowed(
        self,
        prices: np.ndarray,
        window_size: int = 10000,
        step_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute collusion premium in rolling windows.

        Args:
            prices: 1-D or 2-D price array.
            window_size: Number of rounds per window.
            step_size: Stride between windows (default: ``window_size // 2``).

        Returns:
            Dict with arrays *window_starts*, *premiums*, *ci_lowers*,
            *ci_uppers*, and the overall *trend_slope* / *trend_pvalue*.
        """
        flat = self._flatten_prices(prices)
        n = len(flat)
        if step_size is None:
            step_size = max(1, window_size // 2)

        starts: List[int] = []
        premiums: List[float] = []
        ci_lowers: List[float] = []
        ci_uppers: List[float] = []

        pos = 0
        while pos + window_size <= n:
            window = flat[pos : pos + window_size]
            mean_w = float(np.mean(window))
            prem = (mean_w - self.nash_price) / self._price_range
            lo, hi = self._bootstrap_ci(
                window, statistic=self._premium_statistic, method="percentile"
            )
            starts.append(pos)
            premiums.append(prem)
            ci_lowers.append(lo)
            ci_uppers.append(hi)
            pos += step_size

        premiums_arr = np.array(premiums)
        starts_arr = np.array(starts, dtype=float)

        if len(premiums_arr) >= 3:
            slope, intercept, r, p_val, se = stats.linregress(starts_arr, premiums_arr)
        else:
            slope, intercept, r, p_val, se = 0.0, 0.0, 0.0, 1.0, 0.0

        return {
            "window_starts": starts,
            "premiums": premiums,
            "ci_lowers": ci_lowers,
            "ci_uppers": ci_uppers,
            "trend_slope": float(slope),
            "trend_intercept": float(intercept),
            "trend_r_squared": float(r ** 2),
            "trend_pvalue": float(p_val),
            "trend_slope_se": float(se),
        }

    def absolute_margin(self, prices: np.ndarray) -> Dict[str, Any]:
        """Compute absolute margin above Nash price.

        When the Nash price equals marginal cost the normalized premium is
        still well-defined, but the *absolute* margin can be more
        informative for practitioners.

        Returns:
            Dict with *margin*, *margin_ci*, *margin_pct* (percentage of
            Nash price), *mean_price*.
        """
        flat = self._flatten_prices(prices)
        mean_price = float(np.mean(flat))
        margin = mean_price - self.nash_price

        def margin_stat(d: np.ndarray) -> float:
            return float(np.mean(d)) - self.nash_price

        lo, hi = self._bootstrap_ci(flat, statistic=margin_stat, method="percentile")

        if self.nash_price != 0.0:
            margin_pct = margin / abs(self.nash_price) * 100.0
        else:
            margin_pct = float("inf") if margin > 0 else 0.0

        return {
            "margin": margin,
            "margin_ci": (lo, hi),
            "margin_pct": margin_pct,
            "mean_price": mean_price,
            "n_obs": len(flat),
        }

    def collusion_index(self, prices: np.ndarray) -> Dict[str, Any]:
        """Compute Collusion Index clipped to [0, 1].

        CI = max(0, min(1, CP)).

        Returns:
            Dict with *collusion_index*, *ci_lower*, *ci_upper*, *raw_premium*.
        """
        res = self.compute(prices)
        return {
            "collusion_index": res["collusion_index"],
            "ci_lower": res["collusion_index_ci"][0],
            "ci_upper": res["collusion_index_ci"][1],
            "raw_premium": res["premium"],
        }

    # ------------------------------------------------------------------
    # Robustness
    # ------------------------------------------------------------------

    def demand_robustness_check(
        self,
        prices: np.ndarray,
        demand_perturbations: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Check premium robustness to demand-specification uncertainty.

        Each perturbation dict may contain ``nash_delta`` and
        ``monopoly_delta`` that shift the benchmark prices.  The baseline
        premium is recomputed under every perturbation.

        Args:
            prices: Price data.
            demand_perturbations: List of dicts, each with optional keys
                ``nash_delta`` and ``monopoly_delta``.

        Returns:
            Dict with *baseline_premium*, *perturbed_premiums* list, and
            robustness statistics (*mean*, *std*, *min*, *max*,
            *coefficient_of_variation*).
        """
        flat = self._flatten_prices(prices)
        mean_price = float(np.mean(flat))
        baseline = (mean_price - self.nash_price) / self._price_range

        if demand_perturbations is None:
            demand_perturbations = [
                {"nash_delta": d_n, "monopoly_delta": d_m}
                for d_n in [-0.05, 0.0, 0.05]
                for d_m in [-0.05, 0.0, 0.05]
                if not (d_n == 0.0 and d_m == 0.0)
            ]

        perturbed: List[Dict[str, Any]] = []
        for pert in demand_perturbations:
            d_n = pert.get("nash_delta", 0.0)
            d_m = pert.get("monopoly_delta", 0.0)
            new_nash = self.nash_price + d_n
            new_mono = self.monopoly_price + d_m
            rng = new_mono - new_nash
            if rng <= 0:
                perturbed.append({"perturbation": pert, "premium": float("nan")})
                continue
            prem = (mean_price - new_nash) / rng
            perturbed.append({"perturbation": pert, "premium": prem})

        valid = [p["premium"] for p in perturbed if np.isfinite(p["premium"])]
        if valid:
            arr = np.array(valid)
            mean_p, std_p = float(np.mean(arr)), float(np.std(arr, ddof=1))
            cv = std_p / abs(mean_p) if mean_p != 0 else float("inf")
        else:
            mean_p, std_p, cv = float("nan"), float("nan"), float("nan")

        return {
            "baseline_premium": baseline,
            "perturbed_premiums": perturbed,
            "mean": mean_p,
            "std": std_p,
            "min": float(np.nanmin(valid)) if valid else float("nan"),
            "max": float(np.nanmax(valid)) if valid else float("nan"),
            "coefficient_of_variation": cv,
            "robust": cv < 0.25 if np.isfinite(cv) else False,
        }

    # ------------------------------------------------------------------
    # Bootstrap methods
    # ------------------------------------------------------------------

    def bootstrap_premium(
        self, prices: np.ndarray, method: str = "bca"
    ) -> Dict[str, Any]:
        """Detailed bootstrap inference for the collusion premium.

        Args:
            prices: Price data.
            method: One of ``"percentile"``, ``"basic"``, ``"bca"``,
                    ``"studentized"``.

        Returns:
            Dict with *premium*, *ci*, *se*, *bias*, *bootstrap_dist*.
        """
        flat = self._flatten_prices(prices)
        stat_fn = self._premium_statistic
        observed = stat_fn(flat)

        boot_dist = self._generate_bootstrap_distribution(flat, stat_fn)
        boot_mean = float(np.mean(boot_dist))
        boot_se = float(np.std(boot_dist, ddof=1))
        bias = boot_mean - observed

        if method == "percentile":
            ci = self._percentile_ci(boot_dist)
        elif method == "basic":
            ci = self._basic_ci(observed, boot_dist)
        elif method == "bca":
            ci = self._bca_ci(flat, stat_fn, boot_dist)
        elif method == "studentized":
            ci = self._studentized_ci(flat, stat_fn)
        else:
            raise ValueError(
                f"Unknown method {method!r}. "
                "Choose from 'percentile', 'basic', 'bca', 'studentized'."
            )

        return {
            "premium": observed,
            "ci": ci,
            "se": boot_se,
            "bias": bias,
            "bias_corrected_estimate": observed - bias,
            "method": method,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_dist": boot_dist,
        }

    def block_bootstrap_premium(
        self, prices: np.ndarray, block_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Block bootstrap for serially-correlated price data.

        Uses the circular block bootstrap of Politis & Romano (1992).

        Args:
            prices: 1-D or 2-D price array.
            block_size: Block length.  If *None* the optimal size is
                        estimated from the data.

        Returns:
            Dict with *premium*, *ci*, *se*, *block_size*, *bootstrap_dist*.
        """
        flat = self._flatten_prices(prices)
        n = len(flat)
        if block_size is None:
            block_size = self._optimal_block_size(flat)
        block_size = max(1, min(block_size, n))

        stat_fn = self._premium_statistic
        observed = stat_fn(flat)

        boot_dist = np.empty(self.bootstrap_samples)
        num_blocks = int(np.ceil(n / block_size))

        for b in range(self.bootstrap_samples):
            block_starts = self.rng.randint(0, n, size=num_blocks)
            indices: List[int] = []
            for s in block_starts:
                indices.extend([(s + k) % n for k in range(block_size)])
            sample = flat[np.array(indices[:n])]
            boot_dist[b] = stat_fn(sample)

        boot_se = float(np.std(boot_dist, ddof=1))
        ci = self._percentile_ci(boot_dist)

        return {
            "premium": observed,
            "ci": ci,
            "se": boot_se,
            "block_size": block_size,
            "n_obs": n,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_dist": boot_dist,
        }

    # ------------------------------------------------------------------
    # Hypothesis tests
    # ------------------------------------------------------------------

    def test_premium_positive(
        self, prices: np.ndarray, alternative: str = "greater"
    ) -> Dict[str, Any]:
        """Test whether the collusion premium is positive.

        H0: CP <= 0   vs   H1: CP > 0  (default).

        Uses a bootstrap-based test: the p-value is the proportion of
        bootstrap replicates that fall at or below zero (for
        ``alternative='greater'``).

        Args:
            prices: Price data.
            alternative: ``"greater"`` (default), ``"less"``, or
                         ``"two-sided"``.

        Returns:
            Dict with *premium*, *p_value*, *reject*, *alternative*,
            *bootstrap_se*.
        """
        flat = self._flatten_prices(prices)
        stat_fn = self._premium_statistic
        observed = stat_fn(flat)
        boot_dist = self._generate_bootstrap_distribution(flat, stat_fn)

        # Centre the distribution under H0 (shift so mean = 0)
        centred = boot_dist - np.mean(boot_dist)

        if alternative == "greater":
            p_value = float(np.mean(centred >= observed))
        elif alternative == "less":
            p_value = float(np.mean(centred <= observed))
        elif alternative == "two-sided":
            p_value = float(np.mean(np.abs(centred) >= abs(observed)))
        else:
            raise ValueError(
                f"alternative must be 'greater', 'less', or 'two-sided', "
                f"got {alternative!r}"
            )

        alpha = 1.0 - self.confidence_level
        return {
            "premium": observed,
            "p_value": p_value,
            "reject": p_value < alpha,
            "alternative": alternative,
            "alpha": alpha,
            "bootstrap_se": float(np.std(boot_dist, ddof=1)),
            "n_obs": len(flat),
        }

    def test_premium_threshold(
        self, prices: np.ndarray, threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Test whether the premium exceeds a practical threshold.

        H0: CP <= *threshold*  vs  H1: CP > *threshold*.

        Returns:
            Dict with *premium*, *threshold*, *excess*, *p_value*,
            *reject*, *bootstrap_se*.
        """
        flat = self._flatten_prices(prices)
        stat_fn = self._premium_statistic
        observed = stat_fn(flat)
        excess = observed - threshold

        boot_dist = self._generate_bootstrap_distribution(flat, stat_fn)
        # Shift distribution so that the null hypothesis mean equals threshold
        boot_shifted = boot_dist - np.mean(boot_dist) + threshold
        p_value = float(np.mean(boot_shifted >= observed))

        alpha = 1.0 - self.confidence_level
        return {
            "premium": observed,
            "threshold": threshold,
            "excess": excess,
            "p_value": p_value,
            "reject": p_value < alpha,
            "alpha": alpha,
            "bootstrap_se": float(np.std(boot_dist, ddof=1)),
            "n_obs": len(flat),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _premium_statistic(self, data: np.ndarray) -> float:
        """Point-estimate of the collusion premium."""
        return (float(np.mean(data)) - self.nash_price) / self._price_range

    def _flatten_prices(self, prices: np.ndarray) -> np.ndarray:
        """Return a 1-D array of per-round mean prices."""
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 2:
            prices = prices.mean(axis=1)
        return prices.ravel()

    def _generate_bootstrap_distribution(
        self, data: np.ndarray, statistic: Callable
    ) -> np.ndarray:
        """Draw *bootstrap_samples* resamples and evaluate *statistic*."""
        n = len(data)
        dist = np.empty(self.bootstrap_samples)
        for i in range(self.bootstrap_samples):
            idx = self.rng.randint(0, n, size=n)
            dist[i] = statistic(data[idx])
        return dist

    def _bootstrap_ci(
        self,
        data: np.ndarray,
        statistic: Callable,
        method: str = "percentile",
    ) -> Tuple[float, float]:
        """Internal bootstrap CI helper."""
        boot_dist = self._generate_bootstrap_distribution(data, statistic)
        if method == "percentile":
            return self._percentile_ci(boot_dist)
        elif method == "basic":
            return self._basic_ci(statistic(data), boot_dist)
        elif method == "bca":
            return self._bca_ci(data, statistic, boot_dist)
        elif method == "studentized":
            return self._studentized_ci(data, statistic)
        raise ValueError(f"Unknown CI method: {method!r}")

    def _percentile_ci(self, boot_dist: np.ndarray) -> Tuple[float, float]:
        """Percentile confidence interval."""
        alpha = 1.0 - self.confidence_level
        lo = float(np.percentile(boot_dist, 100 * alpha / 2))
        hi = float(np.percentile(boot_dist, 100 * (1 - alpha / 2)))
        return (lo, hi)

    def _basic_ci(
        self, observed: float, boot_dist: np.ndarray
    ) -> Tuple[float, float]:
        """Basic (reverse-percentile) confidence interval."""
        alpha = 1.0 - self.confidence_level
        q_lo = float(np.percentile(boot_dist, 100 * alpha / 2))
        q_hi = float(np.percentile(boot_dist, 100 * (1 - alpha / 2)))
        return (2 * observed - q_hi, 2 * observed - q_lo)

    def _bca_ci(
        self,
        data: np.ndarray,
        statistic: Callable,
        bootstrap_dist: np.ndarray,
    ) -> Tuple[float, float]:
        """Bias-corrected and accelerated (BCa) confidence interval.

        See Efron (1987) *Better Bootstrap Confidence Intervals*.
        """
        observed = statistic(data)
        n = len(data)
        B = len(bootstrap_dist)
        alpha = 1.0 - self.confidence_level

        # Bias correction constant z0
        prop_below = np.sum(bootstrap_dist < observed) / B
        prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
        z0 = stats.norm.ppf(prop_below)

        # Acceleration constant a via jackknife
        jack = np.empty(n)
        for i in range(n):
            jack_sample = np.concatenate([data[:i], data[i + 1 :]])
            jack[i] = statistic(jack_sample)
        jack_mean = np.mean(jack)
        diffs = jack_mean - jack
        sum_cubed = np.sum(diffs ** 3)
        sum_squared = np.sum(diffs ** 2)
        if sum_squared == 0:
            a = 0.0
        else:
            a = sum_cubed / (6.0 * (sum_squared ** 1.5))

        z_alpha_lo = stats.norm.ppf(alpha / 2)
        z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

        def _adjusted_quantile(z_alpha: float) -> float:
            numer = z0 + z_alpha
            denom = 1.0 - a * numer
            if abs(denom) < 1e-12:
                denom = 1e-12 * np.sign(denom) if denom != 0 else 1e-12
            adjusted_z = z0 + numer / denom
            prob = stats.norm.cdf(adjusted_z)
            prob = np.clip(prob, 1e-10, 1 - 1e-10)
            return float(np.percentile(bootstrap_dist, 100 * prob))

        return (_adjusted_quantile(z_alpha_lo), _adjusted_quantile(z_alpha_hi))

    def _studentized_ci(
        self, data: np.ndarray, statistic: Callable
    ) -> Tuple[float, float]:
        """Studentized (bootstrap-t) confidence interval."""
        n = len(data)
        observed = statistic(data)

        # Estimate SE of observed via jackknife
        jack = np.empty(n)
        for i in range(n):
            jack[i] = statistic(np.concatenate([data[:i], data[i + 1 :]]))
        se_obs = float(np.std(jack, ddof=1) * np.sqrt(n - 1))
        if se_obs < 1e-15:
            se_obs = 1e-15

        t_dist = np.empty(self.bootstrap_samples)
        for b in range(self.bootstrap_samples):
            idx = self.rng.randint(0, n, size=n)
            sample = data[idx]
            boot_stat = statistic(sample)
            # Inner jackknife SE for this resample
            jack_inner = np.empty(n)
            for i in range(n):
                jack_inner[i] = statistic(
                    np.concatenate([sample[:i], sample[i + 1 :]])
                )
            se_boot = float(np.std(jack_inner, ddof=1) * np.sqrt(n - 1))
            if se_boot < 1e-15:
                se_boot = 1e-15
            t_dist[b] = (boot_stat - observed) / se_boot

        alpha = 1.0 - self.confidence_level
        t_lo = float(np.percentile(t_dist, 100 * (1 - alpha / 2)))
        t_hi = float(np.percentile(t_dist, 100 * alpha / 2))
        return (observed - t_lo * se_obs, observed - t_hi * se_obs)

    def _optimal_block_size(self, data: np.ndarray) -> int:
        """Estimate optimal block size for the circular block bootstrap.

        Uses the rule-of-thumb  b* ~ n^{1/3} scaled by the first-order
        autocorrelation (Lahiri, 2003; §7.4).
        """
        n = len(data)
        if n < 4:
            return 1

        mean_d = np.mean(data)
        centred = data - mean_d
        var = np.dot(centred, centred) / n
        if var < 1e-15:
            return 1

        # First-order autocovariance
        gamma1 = np.dot(centred[:-1], centred[1:]) / n
        rho1 = gamma1 / var

        # b* ≈ (2 * rho1^2 / (1 - rho1^2)^2)^{1/3} * n^{1/3}
        rho1_sq = rho1 ** 2
        denom = max((1.0 - rho1_sq) ** 2, 1e-12)
        factor = (2.0 * rho1_sq / denom) ** (1.0 / 3.0)
        b_star = factor * (n ** (1.0 / 3.0))
        return max(1, int(round(b_star)))

    # ------------------------------------------------------------------
    # Display / factory
    # ------------------------------------------------------------------

    def summary(self, result: Dict[str, Any]) -> str:
        """Human-readable summary of a premium result dict."""
        lines: List[str] = ["=== Collusion Premium Summary ==="]

        if "premium" in result:
            lines.append(f"  Premium (CP):        {result['premium']:.4f}")
        if "ci" in result:
            lo, hi = result["ci"]
            lines.append(
                f"  {self.confidence_level*100:.0f}% CI:            "
                f"[{lo:.4f}, {hi:.4f}]"
            )
        elif "ci_lower" in result and "ci_upper" in result:
            lines.append(
                f"  {self.confidence_level*100:.0f}% CI:            "
                f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
            )
        if "collusion_index" in result:
            lines.append(
                f"  Collusion Index:     {result['collusion_index']:.4f}"
            )
        if "absolute_margin" in result:
            lines.append(
                f"  Absolute Margin:     {result['absolute_margin']:.4f}"
            )
        if "margin" in result:
            lines.append(f"  Margin:              {result['margin']:.4f}")
        if "margin_pct" in result:
            lines.append(f"  Margin (%):          {result['margin_pct']:.2f}%")
        if "se" in result:
            lines.append(f"  Bootstrap SE:        {result['se']:.4f}")
        if "bias" in result:
            lines.append(f"  Bootstrap Bias:      {result['bias']:.6f}")
        if "p_value" in result:
            lines.append(f"  p-value:             {result['p_value']:.4f}")
        if "reject" in result:
            tag = "REJECT H0" if result["reject"] else "fail to reject H0"
            lines.append(f"  Decision:            {tag}")
        if "threshold" in result:
            lines.append(f"  Threshold:           {result['threshold']:.4f}")
        if "mean_price" in result:
            lines.append(f"  Mean Price:          {result['mean_price']:.4f}")
        if "n_obs" in result:
            lines.append(f"  Observations:        {result['n_obs']}")
        if "method" in result:
            lines.append(f"  CI Method:           {result['method']}")
        if "block_size" in result:
            lines.append(f"  Block Size:          {result['block_size']}")
        if "robust" in result:
            tag = "YES" if result["robust"] else "NO"
            lines.append(f"  Robust to demand:    {tag}")

        lines.append("=" * 33)
        return "\n".join(lines)

    @staticmethod
    def from_game_config(
        game_config: Dict[str, Any], **kwargs: Any
    ) -> "CollusionPremiumCalculator":
        """Construct from a game configuration dict.

        Expected keys in *game_config*: ``nash_price``, ``monopoly_price``,
        and optionally ``marginal_cost``.

        Extra keyword arguments are forwarded to the constructor (e.g.
        ``bootstrap_samples``, ``confidence_level``, ``random_state``).
        """
        return CollusionPremiumCalculator(
            nash_price=game_config["nash_price"],
            monopoly_price=game_config["monopoly_price"],
            marginal_cost=game_config.get("marginal_cost", 0.0),
            **kwargs,
        )


# ======================================================================
# Multi-market analysis
# ======================================================================


class MultiMarketPremium:
    """Collusion premium analysis across multiple markets or time periods."""

    def __init__(self, calculators: List[CollusionPremiumCalculator]):
        if not calculators:
            raise ValueError("At least one calculator is required")
        self.calculators = calculators

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all(self, price_data: List[np.ndarray]) -> Dict[str, Any]:
        """Compute premiums for every market and aggregate.

        Args:
            price_data: One price array per calculator/market.

        Returns:
            Dict with *per_market* list, *mean_premium*, *median_premium*,
            *std_premium*, *min_premium*, *max_premium*, *weighted_premium*
            (weighted by number of observations).
        """
        self._check_lengths(price_data)
        per_market: List[Dict[str, Any]] = []
        for calc, prices in zip(self.calculators, price_data):
            res = calc.compute(prices)
            res["market_index"] = len(per_market)
            per_market.append(res)

        premiums = np.array([r["premium"] for r in per_market])
        n_obs = np.array([r["n_obs"] for r in per_market], dtype=float)
        weights = n_obs / n_obs.sum() if n_obs.sum() > 0 else np.ones_like(n_obs) / len(n_obs)

        return {
            "per_market": per_market,
            "mean_premium": float(np.mean(premiums)),
            "median_premium": float(np.median(premiums)),
            "std_premium": float(np.std(premiums, ddof=1)) if len(premiums) > 1 else 0.0,
            "min_premium": float(np.min(premiums)),
            "max_premium": float(np.max(premiums)),
            "weighted_premium": float(np.dot(weights, premiums)),
            "n_markets": len(per_market),
        }

    def test_joint_premium(self, price_data: List[np.ndarray]) -> Dict[str, Any]:
        """Joint test that the average premium across markets is positive.

        Uses a one-sample t-test on the vector of per-market premiums.

        H0: mean(CP_1, …, CP_K) <= 0   vs   H1: mean > 0.

        Returns:
            Dict with *premiums*, *mean_premium*, *t_stat*, *p_value*,
            *reject*, *df*.
        """
        self._check_lengths(price_data)
        premiums = np.array(
            [
                calc.compute(prices)["premium"]
                for calc, prices in zip(self.calculators, price_data)
            ]
        )

        k = len(premiums)
        mean_p = float(np.mean(premiums))

        if k == 1:
            return {
                "premiums": premiums.tolist(),
                "mean_premium": mean_p,
                "t_stat": float("nan"),
                "p_value": 0.0 if mean_p > 0 else 1.0,
                "reject": mean_p > 0,
                "df": 0,
            }

        se = float(np.std(premiums, ddof=1) / np.sqrt(k))
        if se < 1e-15:
            p_val = 0.0 if mean_p > 0 else 1.0
            t_stat = float("inf") if mean_p > 0 else float("-inf")
        else:
            t_stat = mean_p / se
            p_val = float(1.0 - stats.t.cdf(t_stat, df=k - 1))

        alpha = 1.0 - self.calculators[0].confidence_level
        return {
            "premiums": premiums.tolist(),
            "mean_premium": mean_p,
            "t_stat": t_stat,
            "p_value": p_val,
            "reject": p_val < alpha,
            "df": k - 1,
            "alpha": alpha,
        }

    def heterogeneity_test(self, price_data: List[np.ndarray]) -> Dict[str, Any]:
        """Test for heterogeneity in premiums across markets.

        Uses Cochran's Q statistic (inverse-variance weighting).  Under
        the null of homogeneity, Q ~ chi-squared with K-1 df.

        Returns:
            Dict with *premiums*, *Q*, *p_value*, *reject*, *I_squared*,
            *df*, *weighted_mean*.
        """
        self._check_lengths(price_data)

        premiums: List[float] = []
        variances: List[float] = []

        for calc, prices in zip(self.calculators, price_data):
            boot_res = calc.bootstrap_premium(prices, method="percentile")
            premiums.append(boot_res["premium"])
            se = boot_res["se"]
            variances.append(se ** 2 if se > 0 else 1e-30)

        k = len(premiums)
        p_arr = np.array(premiums)
        v_arr = np.array(variances)
        w_arr = 1.0 / v_arr  # inverse-variance weights

        weighted_mean = float(np.sum(w_arr * p_arr) / np.sum(w_arr))
        Q = float(np.sum(w_arr * (p_arr - weighted_mean) ** 2))
        df = k - 1

        if df > 0:
            p_value = float(1.0 - stats.chi2.cdf(Q, df))
            I_squared = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
        else:
            p_value = 1.0
            I_squared = 0.0

        alpha = 1.0 - self.calculators[0].confidence_level
        return {
            "premiums": [float(p) for p in premiums],
            "Q": Q,
            "p_value": p_value,
            "reject": p_value < alpha,
            "I_squared": I_squared,
            "df": df,
            "weighted_mean": weighted_mean,
            "alpha": alpha,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_lengths(self, price_data: List[np.ndarray]) -> None:
        if len(price_data) != len(self.calculators):
            raise ValueError(
                f"Expected {len(self.calculators)} price arrays, "
                f"got {len(price_data)}"
            )
