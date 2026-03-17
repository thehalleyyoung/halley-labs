"""M1 Composite Hypothesis Test for algorithmic collusion detection.

Implements a tiered testing framework:
- Tier 1: Supra-competitive price level tests
- Tier 2: Cross-firm correlation tests
- Tier 3: Punishment/retaliation tests
- Tier 4: Counterfactual tests (requires oracle access)

Uses alpha-spending to control family-wise error rate across tiers.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import List, Optional, Dict, Tuple, Any


# ---------------------------------------------------------------------------
# Alpha-spending across tiers
# ---------------------------------------------------------------------------

class AlphaSpender:
    """Manages alpha spending across test tiers.

    Implements an O'Brien-Fleming-like spending function to allocate
    significance level across tiers while controlling FWER.
    """

    def __init__(
        self,
        total_alpha: float = 0.05,
        allocation: Optional[Dict[str, float]] = None,
    ) -> None:
        self.total_alpha = total_alpha
        self.allocation = allocation or {
            "tier1": 0.020,
            "tier2": 0.015,
            "tier3": 0.010,
            "tier4": 0.005,
        }
        # Validate that the allocation sums to total_alpha
        alloc_sum = sum(self.allocation.values())
        if abs(alloc_sum - total_alpha) > 1e-9:
            raise ValueError(
                f"Tier allocations sum to {alloc_sum}, expected {total_alpha}"
            )
        self.spent: Dict[str, float] = {k: 0.0 for k in self.allocation}

    # -- public API ---------------------------------------------------------

    def get_alpha(self, tier: str) -> float:
        """Return the allocated alpha budget for *tier*."""
        if tier not in self.allocation:
            raise KeyError(f"Unknown tier '{tier}'")
        return self.allocation[tier]

    def spend(self, tier: str, p_value: float) -> bool:
        """Record a test at *tier* with the observed *p_value*.

        Returns ``True`` when the null can be rejected at the allocated
        level (i.e. *p_value* ≤ remaining alpha for the tier).
        """
        if tier not in self.allocation:
            raise KeyError(f"Unknown tier '{tier}'")
        remaining_for_tier = self.allocation[tier] - self.spent[tier]
        rejected = p_value <= remaining_for_tier
        if rejected:
            self.spent[tier] += p_value
        return rejected

    def remaining(self) -> float:
        """Total alpha remaining across all tiers."""
        return self.total_alpha - self.total_spent

    def can_reject(self, tier: str, p_value: float) -> bool:
        """Check whether *p_value* can be rejected without actually spending."""
        remaining_for_tier = self.allocation[tier] - self.spent[tier]
        return p_value <= remaining_for_tier

    @property
    def total_spent(self) -> float:
        return sum(self.spent.values())

    def summary(self) -> Dict[str, Any]:
        return {
            "total_alpha": self.total_alpha,
            "total_spent": self.total_spent,
            "remaining": self.remaining(),
            "per_tier": {
                t: {
                    "allocated": self.allocation[t],
                    "spent": self.spent[t],
                    "remaining": self.allocation[t] - self.spent[t],
                }
                for t in self.allocation
            },
        }


# ---------------------------------------------------------------------------
# Helper: Newey-West HAC standard errors
# ---------------------------------------------------------------------------

def _newey_west_se(x: np.ndarray, max_lag: Optional[int] = None) -> float:
    """Compute HAC (Newey-West) standard error for the sample mean of *x*."""
    n = len(x)
    if max_lag is None:
        max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    x_demeaned = x - np.mean(x)
    gamma_0 = np.dot(x_demeaned, x_demeaned) / n
    weighted_sum = 0.0
    for j in range(1, max_lag + 1):
        weight = 1.0 - j / (max_lag + 1.0)
        gamma_j = np.dot(x_demeaned[j:], x_demeaned[:-j]) / n
        weighted_sum += 2.0 * weight * gamma_j
    variance = (gamma_0 + weighted_sum) / n
    return float(np.sqrt(max(variance, 0.0)))


# ---------------------------------------------------------------------------
# Tier tests
# ---------------------------------------------------------------------------

class TierTest:
    """Base class for a single tier test."""

    def __init__(self, tier_name: str, null_hypothesis: str) -> None:
        self.tier_name = tier_name
        self.null_hypothesis = null_hypothesis

    def run(self, prices: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """Run the tier test.

        Returns a dict with keys ``test_statistic``, ``p_value``,
        ``reject``, ``details``.
        """
        raise NotImplementedError


class PriceLevelTest(TierTest):
    """Tier 1: Test whether prices are supra-competitive.

    H0: mean price ≤ Nash equilibrium price + margin
    H1: mean price > Nash equilibrium price + margin

    Uses a one-sided t-test with Newey-West HAC standard errors so
    inference is valid under serial correlation.
    """

    def __init__(self, nash_price: float, margin: float = 0.0) -> None:
        super().__init__("tier1_price_level", "no_supracompetitive")
        self.nash_price = nash_price
        self.margin = margin

    def run(self, prices: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        alpha = kwargs.get("alpha", 0.05)
        # Average across players for each round
        if prices.ndim == 2:
            avg_prices = prices.mean(axis=1)
        else:
            avg_prices = prices.ravel()

        threshold = self.nash_price + self.margin
        diffs = avg_prices - threshold
        n = len(diffs)
        mean_diff = float(np.mean(diffs))
        se = _newey_west_se(diffs)

        if se < 1e-15:
            t_stat = np.inf if mean_diff > 0 else -np.inf
        else:
            t_stat = mean_diff / se

        p_value = float(1.0 - stats.t.cdf(t_stat, df=n - 1))  # one-sided

        # Effect size (Cohen's d using HAC SE)
        pooled_std = float(np.std(diffs, ddof=1))
        cohens_d = mean_diff / pooled_std if pooled_std > 1e-15 else 0.0

        # Bootstrap CI for the premium
        rng = np.random.default_rng(kwargs.get("seed", 42))
        n_boot = kwargs.get("bootstrap_samples", 5000)
        boot_means = np.array(
            [np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(n_boot)]
        )
        ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(
            np.percentile(boot_means, 97.5)
        )

        reject = p_value <= alpha

        return {
            "tier": self.tier_name,
            "null_hypothesis": self.null_hypothesis,
            "test_statistic": float(t_stat),
            "p_value": p_value,
            "reject": reject,
            "details": {
                "mean_diff": mean_diff,
                "hac_se": se,
                "cohens_d": cohens_d,
                "ci_95": (ci_low, ci_high),
                "n_observations": n,
                "threshold": threshold,
            },
        }


class CorrelationTest(TierTest):
    """Tier 2: Test for supra-competitive cross-firm correlation.

    H0: Firms' price changes are independent
    H1: Firms' price changes show positive correlation beyond *min_correlation*

    Uses Fisher-transformed Pearson correlation with a permutation-based
    p-value to remain nonparametric.
    """

    def __init__(self, min_correlation: float = 0.0) -> None:
        super().__init__("tier2_correlation", "independent_play")
        self.min_correlation = min_correlation

    def run(self, prices: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        alpha = kwargs.get("alpha", 0.05)
        n_perm = kwargs.get("n_permutations", 5000)
        rng = np.random.default_rng(kwargs.get("seed", 42))

        if prices.ndim != 2 or prices.shape[1] < 2:
            raise ValueError("CorrelationTest requires (T, N>=2) price matrix")

        # First differences to avoid spurious correlation from trends
        diffs = np.diff(prices, axis=0)
        n_players = diffs.shape[1]
        T = diffs.shape[0]

        # Compute average pairwise Pearson correlation
        corr_matrix = np.corrcoef(diffs.T)
        upper_idx = np.triu_indices(n_players, k=1)
        observed_corrs = corr_matrix[upper_idx]
        observed_mean_corr = float(np.mean(observed_corrs))

        # Fisher z-transform of observed correlation
        z_obs = np.arctanh(np.clip(observed_mean_corr, -0.999, 0.999))
        z_null = np.arctanh(np.clip(self.min_correlation, -0.999, 0.999))
        z_stat = (z_obs - z_null) * np.sqrt(T - 3)

        # Permutation test
        perm_corrs = np.empty(n_perm)
        for i in range(n_perm):
            perm_diffs = diffs.copy()
            for j in range(1, n_players):
                rng.shuffle(perm_diffs[:, j])
            pc = np.corrcoef(perm_diffs.T)
            perm_corrs[i] = np.mean(pc[upper_idx])

        p_perm = float(np.mean(perm_corrs >= observed_mean_corr))
        # Also analytic p-value from Fisher z
        p_fisher = float(1.0 - stats.norm.cdf(z_stat))
        # Use the more conservative one
        p_value = max(p_perm, p_fisher)

        reject = p_value <= alpha

        return {
            "tier": self.tier_name,
            "null_hypothesis": self.null_hypothesis,
            "test_statistic": float(z_stat),
            "p_value": p_value,
            "reject": reject,
            "details": {
                "mean_pairwise_correlation": observed_mean_corr,
                "individual_correlations": observed_corrs.tolist(),
                "p_permutation": p_perm,
                "p_fisher": p_fisher,
                "n_permutations": n_perm,
                "n_timepoints": T,
            },
        }


class PunishmentTest(TierTest):
    """Tier 3: Test for punishment / retaliation behaviour.

    H0: No systematic price response to competitor deviations
    H1: Price decreases follow competitor deviations (punishment phase)

    A *deviation* is defined as a price drop > *deviation_threshold* × std
    below a rolling mean. The test measures the average rival-price change
    in the *response_window* periods after a deviation and checks whether
    it is significantly negative (punishment).
    """

    def __init__(
        self,
        deviation_threshold: float = 0.05,
        response_window: int = 10,
        lookback: int = 50,
    ) -> None:
        super().__init__("tier3_punishment", "no_punishment")
        self.deviation_threshold = deviation_threshold
        self.response_window = response_window
        self.lookback = lookback

    def run(self, prices: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        alpha = kwargs.get("alpha", 0.05)
        rng = np.random.default_rng(kwargs.get("seed", 42))

        if prices.ndim != 2 or prices.shape[1] < 2:
            raise ValueError("PunishmentTest requires (T, N>=2) price matrix")

        T, n_players = prices.shape
        all_responses: List[float] = []
        deviation_events: List[Dict[str, Any]] = []

        for player in range(n_players):
            player_prices = prices[:, player]
            rival_cols = [c for c in range(n_players) if c != player]
            rival_prices = prices[:, rival_cols].mean(axis=1)

            # Rolling mean and std for this player
            for t in range(self.lookback, T - self.response_window):
                window = player_prices[t - self.lookback : t]
                mu = window.mean()
                sigma = window.std()
                if sigma < 1e-12:
                    continue
                deviation = (player_prices[t] - mu) / sigma
                if deviation < -self.deviation_threshold:
                    # Player deviated downward — measure rival response
                    rival_before = rival_prices[t]
                    rival_after = rival_prices[
                        t + 1 : t + 1 + self.response_window
                    ].mean()
                    response = rival_after - rival_before
                    all_responses.append(float(response))
                    deviation_events.append(
                        {"player": player, "time": t, "deviation_z": float(deviation)}
                    )

        if len(all_responses) < 5:
            return {
                "tier": self.tier_name,
                "null_hypothesis": self.null_hypothesis,
                "test_statistic": 0.0,
                "p_value": 1.0,
                "reject": False,
                "details": {
                    "n_deviations": len(all_responses),
                    "message": "Too few deviations detected for reliable inference",
                },
            }

        responses = np.array(all_responses)
        mean_response = float(responses.mean())
        se = _newey_west_se(responses) if len(responses) > 10 else float(
            responses.std(ddof=1) / np.sqrt(len(responses))
        )

        if se < 1e-15:
            t_stat = -np.inf if mean_response < 0 else 0.0
        else:
            t_stat = mean_response / se

        # One-sided test: H1 is that response is negative (punishment)
        p_value = float(stats.t.cdf(t_stat, df=len(responses) - 1))

        # Permutation check: shuffle time labels of deviations
        n_perm = kwargs.get("n_permutations", 2000)
        perm_means = np.empty(n_perm)
        for i in range(n_perm):
            rng.shuffle(responses)
            perm_means[i] = responses.mean()
        p_perm = float(np.mean(perm_means <= mean_response))
        p_value = max(p_value, p_perm)

        reject = p_value <= alpha

        return {
            "tier": self.tier_name,
            "null_hypothesis": self.null_hypothesis,
            "test_statistic": float(t_stat),
            "p_value": p_value,
            "reject": reject,
            "details": {
                "n_deviations": len(all_responses),
                "mean_response": mean_response,
                "se": se,
                "p_parametric": float(stats.t.cdf(t_stat, df=len(responses) - 1)),
                "p_permutation": p_perm,
                "deviation_events": deviation_events[:20],
            },
        }


class CounterfactualTest(TierTest):
    """Tier 4: Counterfactual test using oracle access.

    H0: Algorithm would set same prices regardless of competitor behaviour
    H1: Algorithm's prices depend on competitor's past prices (coordination)

    Compares observed price trajectories against counterfactual trajectories
    where one firm's history has been replaced. Uses a paired difference test
    with block-bootstrap for the standard error.
    """

    def __init__(self, block_size: int = 50) -> None:
        super().__init__("tier4_counterfactual", "no_coordination")
        self.block_size = block_size

    def run(
        self,
        prices: np.ndarray,
        counterfactual_prices: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        alpha = kwargs.get("alpha", 0.05)
        rng = np.random.default_rng(kwargs.get("seed", 42))

        if counterfactual_prices is None:
            return {
                "tier": self.tier_name,
                "null_hypothesis": self.null_hypothesis,
                "test_statistic": float("nan"),
                "p_value": 1.0,
                "reject": False,
                "details": {"message": "No counterfactual prices provided; skipping Tier 4"},
            }

        # Ensure matching shapes
        min_len = min(prices.shape[0], counterfactual_prices.shape[0])
        actual = prices[:min_len].mean(axis=1) if prices.ndim == 2 else prices[:min_len]
        cf = (
            counterfactual_prices[:min_len].mean(axis=1)
            if counterfactual_prices.ndim == 2
            else counterfactual_prices[:min_len]
        )

        diffs = actual - cf
        n = len(diffs)
        mean_diff = float(np.mean(diffs))
        abs_mean_diff = float(np.mean(np.abs(diffs)))

        # Block bootstrap standard error
        n_boot = kwargs.get("bootstrap_samples", 5000)
        block = self.block_size
        n_blocks = max(n // block, 1)
        boot_means = np.empty(n_boot)
        for i in range(n_boot):
            starts = rng.integers(0, n - block + 1, size=n_blocks)
            indices = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            boot_means[i] = np.mean(np.abs(diffs[indices]))

        se = float(np.std(boot_means, ddof=1))
        if se < 1e-15:
            t_stat = abs_mean_diff / 1e-15
        else:
            t_stat = abs_mean_diff / se

        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))

        # Kolmogorov-Smirnov test on distributions
        ks_stat, ks_p = stats.ks_2samp(actual, cf)

        # Effect size: max divergence
        max_diff = float(np.max(np.abs(diffs)))
        rmse = float(np.sqrt(np.mean(diffs ** 2)))

        reject = p_value <= alpha

        return {
            "tier": self.tier_name,
            "null_hypothesis": self.null_hypothesis,
            "test_statistic": float(t_stat),
            "p_value": p_value,
            "reject": reject,
            "details": {
                "mean_abs_difference": abs_mean_diff,
                "mean_signed_difference": mean_diff,
                "rmse": rmse,
                "max_abs_difference": max_diff,
                "block_bootstrap_se": se,
                "ks_stat": float(ks_stat),
                "ks_p_value": float(ks_p),
                "n_observations": n,
            },
        }


# ---------------------------------------------------------------------------
# Convergence detection helper
# ---------------------------------------------------------------------------

def _detect_convergence(
    prices: np.ndarray, window: int = 200, threshold: float = 0.01
) -> int:
    """Return the index at which prices are judged to have converged.

    Uses a sliding coefficient-of-variation test: convergence is declared at
    the first window where ``CV < threshold`` for the average price across
    players.
    """
    if prices.ndim == 2:
        avg = prices.mean(axis=1)
    else:
        avg = prices.ravel()
    n = len(avg)
    if n < window:
        return 0
    for start in range(0, n - window + 1, window // 4):
        segment = avg[start : start + window]
        mu = segment.mean()
        if mu == 0:
            continue
        cv = segment.std() / abs(mu)
        if cv < threshold:
            return start
    return 0


# ---------------------------------------------------------------------------
# Composite test orchestrator
# ---------------------------------------------------------------------------

class CompositeTest:
    """M1 Composite Hypothesis Test.

    Runs tiered tests with alpha spending to detect algorithmic collusion.
    The test proceeds sequentially through tiers 1-4.  Each tier's p-value
    is checked against its allocated alpha budget; early stopping occurs if
    a tier fails to reject and the next tiers require its rejection.
    """

    def __init__(
        self,
        nash_price: float,
        monopoly_price: float,
        alpha: float = 0.05,
        bootstrap_samples: int = 10000,
        convergence_window: Optional[int] = None,
        price_margin: float = 0.0,
        deviation_threshold: float = 0.05,
    ) -> None:
        self.nash_price = nash_price
        self.monopoly_price = monopoly_price
        self.alpha = alpha
        self.bootstrap_samples = bootstrap_samples
        self.convergence_window = convergence_window or 200
        self.price_margin = price_margin
        self.deviation_threshold = deviation_threshold
        self.alpha_spender = AlphaSpender(alpha)
        self._results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        prices: np.ndarray,
        counterfactual_prices: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run the full composite test.

        Parameters
        ----------
        prices : ndarray of shape ``(num_rounds, num_players)``
        counterfactual_prices : optional ndarray, same shape as *prices*
        seed : random seed for reproducibility

        Returns
        -------
        dict with ``verdict``, ``confidence``, ``tier_results``,
        ``collusion_premium``, ``alpha_spending_summary``.
        """
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)

        # Detect convergence and use converged portion
        conv_idx = _detect_convergence(
            prices, window=self.convergence_window
        )
        converged_prices = prices[conv_idx:]
        if len(converged_prices) < 100:
            converged_prices = prices  # fall back to full trajectory

        tier_results: List[Dict[str, Any]] = []
        common_kwargs: Dict[str, Any] = {
            "alpha": self.alpha_spender.get_alpha("tier1"),
            "bootstrap_samples": self.bootstrap_samples,
            "seed": seed,
        }

        # --- Tier 1: Price Level ---
        t1 = PriceLevelTest(self.nash_price, self.price_margin)
        r1 = t1.run(converged_prices, **{**common_kwargs, "alpha": self.alpha_spender.get_alpha("tier1")})
        r1["tier_number"] = 1
        self.alpha_spender.spend("tier1", r1["p_value"])
        tier_results.append(r1)

        # --- Tier 2: Correlation ---
        if converged_prices.shape[1] >= 2:
            t2 = CorrelationTest()
            r2 = t2.run(
                converged_prices,
                alpha=self.alpha_spender.get_alpha("tier2"),
                seed=seed,
            )
        else:
            r2 = {
                "tier": "tier2_correlation",
                "null_hypothesis": "independent_play",
                "test_statistic": float("nan"),
                "p_value": 1.0,
                "reject": False,
                "details": {"message": "Single player — correlation test skipped"},
            }
        r2["tier_number"] = 2
        self.alpha_spender.spend("tier2", r2["p_value"])
        tier_results.append(r2)

        # --- Tier 3: Punishment ---
        if converged_prices.shape[1] >= 2:
            t3 = PunishmentTest(self.deviation_threshold)
            r3 = t3.run(
                converged_prices,
                alpha=self.alpha_spender.get_alpha("tier3"),
                seed=seed,
            )
        else:
            r3 = {
                "tier": "tier3_punishment",
                "null_hypothesis": "no_punishment",
                "test_statistic": float("nan"),
                "p_value": 1.0,
                "reject": False,
                "details": {"message": "Single player — punishment test skipped"},
            }
        r3["tier_number"] = 3
        self.alpha_spender.spend("tier3", r3["p_value"])
        tier_results.append(r3)

        # --- Tier 4: Counterfactual (optional) ---
        t4 = CounterfactualTest()
        r4 = t4.run(
            converged_prices,
            counterfactual_prices=counterfactual_prices,
            alpha=self.alpha_spender.get_alpha("tier4"),
            seed=seed,
            bootstrap_samples=self.bootstrap_samples,
        )
        r4["tier_number"] = 4
        self.alpha_spender.spend("tier4", r4["p_value"])
        tier_results.append(r4)

        # --- Aggregate ---
        collusion_premium = self._compute_collusion_premium(converged_prices)
        verdict, confidence = self._determine_verdict(tier_results, collusion_premium)

        result = {
            "verdict": verdict,
            "confidence": confidence,
            "tier_results": tier_results,
            "collusion_premium": collusion_premium,
            "convergence_index": conv_idx,
            "n_converged_rounds": len(converged_prices),
            "alpha_spending_summary": self.alpha_spender.summary(),
        }
        self._results.append(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_collusion_premium(self, prices: np.ndarray) -> float:
        """Return relative premium: how far prices are between Nash and monopoly."""
        avg_price = float(prices.mean())
        span = self.monopoly_price - self.nash_price
        if abs(span) < 1e-12:
            return 0.0
        premium = (avg_price - self.nash_price) / span
        return float(np.clip(premium, 0.0, 1.0))

    def _determine_verdict(
        self, tier_results: List[Dict[str, Any]], collusion_premium: float
    ) -> Tuple[str, float]:
        """Determine a verdict label and confidence from tier results.

        Rules:
        * COLLUSIVE: premium > 0.10 AND ≥ 2 tiers reject
        * SUSPICIOUS: premium > 0.05 OR exactly 1 tier rejects
        * COMPETITIVE: no rejections AND premium < 0.05
        * INCONCLUSIVE: otherwise
        """
        rejections = [r for r in tier_results if r.get("reject")]
        n_reject = len(rejections)
        confidence = self._compute_confidence(tier_results)

        if n_reject >= 2 and collusion_premium > 0.10:
            return "COLLUSIVE", confidence
        if n_reject == 1 or collusion_premium > 0.05:
            return "SUSPICIOUS", confidence
        if n_reject == 0 and collusion_premium < 0.05:
            return "COMPETITIVE", confidence
        return "INCONCLUSIVE", confidence

    def _compute_confidence(self, tier_results: List[Dict[str, Any]]) -> float:
        """Compute an overall confidence score in [0, 1] using Fisher's method.

        Combines p-values via Fisher's method, then maps the resulting
        chi-squared statistic onto a confidence scale.
        """
        p_values = [
            r["p_value"]
            for r in tier_results
            if np.isfinite(r["p_value"]) and r["p_value"] > 0
        ]
        if not p_values:
            return 0.0
        # Fisher's combined statistic: -2 * sum(log(p))
        fisher_stat = -2.0 * sum(np.log(p) for p in p_values)
        df = 2 * len(p_values)
        combined_p = float(1.0 - stats.chi2.cdf(fisher_stat, df))
        # Confidence = 1 - combined_p, clipped to [0,1]
        return float(np.clip(1.0 - combined_p, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the most recent test run."""
        if not self._results:
            return "No test has been run yet."
        r = self._results[-1]
        lines = [
            "=" * 60,
            "  M1 COMPOSITE HYPOTHESIS TEST — SUMMARY",
            "=" * 60,
            f"  Verdict     : {r['verdict']}",
            f"  Confidence  : {r['confidence']:.4f}",
            f"  Collusion premium: {r['collusion_premium']:.4f}",
            f"  Convergence at round: {r['convergence_index']}",
            f"  Converged rounds used: {r['n_converged_rounds']}",
            "-" * 60,
        ]
        for tr in r["tier_results"]:
            marker = "✗ REJECT" if tr.get("reject") else "✓ fail to reject"
            lines.append(
                f"  Tier {tr.get('tier_number', '?')}: "
                f"stat={tr['test_statistic']:.4f}  "
                f"p={tr['p_value']:.6f}  [{marker}]"
            )
        lines.append("-" * 60)
        sp = r["alpha_spending_summary"]
        lines.append(
            f"  Alpha budget: {sp['total_alpha']:.4f}  "
            f"spent: {sp['total_spent']:.6f}  "
            f"remaining: {sp['remaining']:.6f}"
        )
        lines.append("=" * 60)
        return "\n".join(lines)
