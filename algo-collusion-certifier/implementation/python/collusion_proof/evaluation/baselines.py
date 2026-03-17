"""Baseline detection methods for comparison.

Provides simple screening heuristics that serve as baselines against which
the full CollusionProof composite test can be compared.  Each baseline
returns a dict with at least ``verdict``, ``score``, and ``details``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Any, Dict, List, Optional, Tuple


class PriceCorrelationScreen:
    """Detect collusion via high pairwise price correlation."""

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    def detect(self, prices: np.ndarray) -> Dict[str, Any]:
        """Run correlation screen on a (num_rounds, num_players) price matrix."""
        prices = np.asarray(prices, dtype=float)
        n_players = prices.shape[1]

        if n_players < 2:
            return {
                "verdict": "competitive",
                "score": 0.0,
                "details": {"reason": "fewer than 2 players"},
            }

        corr_matrix = np.corrcoef(prices.T)
        upper_idx = np.triu_indices(n_players, k=1)
        pairwise = corr_matrix[upper_idx]
        mean_corr = float(np.mean(pairwise))
        max_corr = float(np.max(pairwise))
        min_corr = float(np.min(pairwise))

        verdict = "collusive" if mean_corr >= self.threshold else "competitive"

        return {
            "verdict": verdict,
            "score": mean_corr,
            "details": {
                "mean_correlation": mean_corr,
                "max_correlation": max_corr,
                "min_correlation": min_corr,
                "threshold": self.threshold,
                "n_pairs": len(pairwise),
            },
        }


class VarianceScreen:
    """Detect collusion via low price variance (price stabilisation)."""

    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    def detect(self, prices: np.ndarray) -> Dict[str, Any]:
        prices = np.asarray(prices, dtype=float)
        # Use last 40% of rounds for steady-state
        tail_start = int(prices.shape[0] * 0.6)
        tail = prices[tail_start:]

        per_player_var = np.var(tail, axis=0)
        mean_var = float(np.mean(per_player_var))
        max_var = float(np.max(per_player_var))

        verdict = "collusive" if mean_var <= self.threshold else "competitive"

        return {
            "verdict": verdict,
            "score": 1.0 - min(mean_var / max(self.threshold, 1e-12), 1.0),
            "details": {
                "mean_variance": mean_var,
                "max_variance": max_var,
                "per_player_variance": per_player_var.tolist(),
                "threshold": self.threshold,
                "tail_start_round": tail_start,
            },
        }


class GrangerCausalityScreen:
    """Detect collusion via Granger causality between player prices.

    If player i's lagged prices help predict player j's current price
    beyond player j's own lags, it suggests coordinated behaviour.
    """

    def __init__(self, max_lag: int = 5, alpha: float = 0.05) -> None:
        self.max_lag = max_lag
        self.alpha = alpha

    def _granger_f_test(
        self, x: np.ndarray, y: np.ndarray, lag: int,
    ) -> Tuple[float, float]:
        """Return (F-statistic, p-value) for Granger causality y → x."""
        n = len(x)
        if n <= 2 * lag + 1:
            return 0.0, 1.0

        # Build design matrices
        # Restricted model: x_t ~ x_{t-1}, ..., x_{t-lag}
        # Unrestricted: x_t ~ x_{t-1}, ..., x_{t-lag}, y_{t-1}, ..., y_{t-lag}
        dependent = x[lag:]
        n_obs = len(dependent)

        X_restricted = np.column_stack(
            [x[lag - k - 1 : n - k - 1] for k in range(lag)]
        )
        X_unrestricted = np.column_stack(
            [X_restricted]
            + [y[lag - k - 1 : n - k - 1] for k in range(lag)]
        )

        # Add constant
        X_r = np.column_stack([np.ones(n_obs), X_restricted])
        X_u = np.column_stack([np.ones(n_obs), X_unrestricted])

        # OLS residuals
        _, rss_r, _, _ = np.linalg.lstsq(X_r, dependent, rcond=None)
        _, rss_u, _, _ = np.linalg.lstsq(X_u, dependent, rcond=None)

        rss_r_val = float(rss_r[0]) if len(rss_r) > 0 else float(np.sum((dependent - X_r @ np.linalg.lstsq(X_r, dependent, rcond=None)[0]) ** 2))
        rss_u_val = float(rss_u[0]) if len(rss_u) > 0 else float(np.sum((dependent - X_u @ np.linalg.lstsq(X_u, dependent, rcond=None)[0]) ** 2))

        df_diff = lag
        df_resid = n_obs - X_u.shape[1]

        if df_resid <= 0 or rss_u_val <= 0:
            return 0.0, 1.0

        f_stat = ((rss_r_val - rss_u_val) / df_diff) / (rss_u_val / df_resid)
        p_value = float(stats.f.sf(max(f_stat, 0.0), df_diff, df_resid))
        return float(f_stat), p_value

    def detect(self, prices: np.ndarray) -> Dict[str, Any]:
        prices = np.asarray(prices, dtype=float)
        n_players = prices.shape[1]

        if n_players < 2:
            return {"verdict": "competitive", "score": 0.0,
                    "details": {"reason": "fewer than 2 players"}}

        # Use last 50% of rounds
        tail_start = int(prices.shape[0] * 0.5)
        tail = prices[tail_start:]

        significant_pairs = 0
        total_pairs = 0
        pair_details: List[Dict[str, Any]] = []

        for i in range(n_players):
            for j in range(n_players):
                if i == j:
                    continue
                total_pairs += 1
                best_f = 0.0
                best_p = 1.0
                for lag in range(1, self.max_lag + 1):
                    f_stat, p_val = self._granger_f_test(tail[:, i], tail[:, j], lag)
                    if p_val < best_p:
                        best_f, best_p = f_stat, p_val
                if best_p < self.alpha:
                    significant_pairs += 1
                pair_details.append({
                    "from_player": j, "to_player": i,
                    "f_statistic": best_f, "p_value": best_p,
                    "significant": best_p < self.alpha,
                })

        causality_rate = significant_pairs / total_pairs if total_pairs > 0 else 0.0
        verdict = "collusive" if causality_rate > 0.5 else "competitive"

        return {
            "verdict": verdict,
            "score": causality_rate,
            "details": {
                "significant_pairs": significant_pairs,
                "total_pairs": total_pairs,
                "causality_rate": causality_rate,
                "alpha": self.alpha,
                "max_lag": self.max_lag,
                "pair_results": pair_details,
            },
        }


class PriceLevelScreen:
    """Detect collusion if average price exceeds a threshold fraction
    of the Nash-to-monopoly gap."""

    def __init__(
        self, nash_price: float, threshold_fraction: float = 0.5,
    ) -> None:
        self.nash_price = nash_price
        self.threshold_fraction = threshold_fraction
        self._monopoly_price: Optional[float] = None

    def detect(self, prices: np.ndarray, monopoly_price: Optional[float] = None) -> Dict[str, Any]:
        prices = np.asarray(prices, dtype=float)
        mono = monopoly_price if monopoly_price is not None else self._monopoly_price
        if mono is None:
            # Estimate monopoly price as max observed
            mono = float(np.max(prices))

        # Steady-state analysis
        tail_start = int(prices.shape[0] * 0.6)
        tail = prices[tail_start:]
        mean_price = float(np.mean(tail))

        price_gap = mono - self.nash_price
        if price_gap <= 0:
            collusion_index = 0.0
        else:
            collusion_index = max(0.0, min((mean_price - self.nash_price) / price_gap, 1.0))

        verdict = "collusive" if collusion_index >= self.threshold_fraction else "competitive"

        return {
            "verdict": verdict,
            "score": collusion_index,
            "details": {
                "mean_price": mean_price,
                "nash_price": self.nash_price,
                "monopoly_price": mono,
                "collusion_index": collusion_index,
                "threshold_fraction": self.threshold_fraction,
            },
        }


class MarkovScreen:
    """Detect collusion via transition matrix analysis.

    Discretises prices into bins and analyses the Markov transition matrix.
    Collusive behaviour is characterised by high persistence (staying
    in the same state) at supra-competitive price levels.
    """

    def __init__(self, num_states: int = 10) -> None:
        self.num_states = num_states

    def _discretise(self, prices_1d: np.ndarray) -> np.ndarray:
        """Map continuous prices to discrete states."""
        lo, hi = float(np.min(prices_1d)), float(np.max(prices_1d))
        if hi <= lo:
            return np.zeros(len(prices_1d), dtype=int)
        bins = np.linspace(lo, hi + 1e-10, self.num_states + 1)
        return np.digitize(prices_1d, bins) - 1

    def _transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """Compute row-normalised transition matrix."""
        n_states = self.num_states
        T = np.zeros((n_states, n_states))
        for s_t, s_next in zip(states[:-1], states[1:]):
            s_t = min(max(int(s_t), 0), n_states - 1)
            s_next = min(max(int(s_next), 0), n_states - 1)
            T[s_t, s_next] += 1
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return T / row_sums

    def detect(self, prices: np.ndarray) -> Dict[str, Any]:
        prices = np.asarray(prices, dtype=float)
        n_players = prices.shape[1]

        # Analyse each player and average
        persistence_scores: List[float] = []
        high_state_fracs: List[float] = []
        transition_matrices: List[np.ndarray] = []

        for p in range(n_players):
            states = self._discretise(prices[:, p])
            T = self._transition_matrix(states)
            transition_matrices.append(T)

            # Persistence: average diagonal of T
            persistence = float(np.mean(np.diag(T)))
            persistence_scores.append(persistence)

            # Fraction of time in top 30% states
            high_threshold = int(self.num_states * 0.7)
            high_frac = float(np.mean(states >= high_threshold))
            high_state_fracs.append(high_frac)

        avg_persistence = float(np.mean(persistence_scores))
        avg_high_frac = float(np.mean(high_state_fracs))

        # Collusion: high persistence + high fraction in top states
        score = 0.5 * avg_persistence + 0.5 * avg_high_frac
        verdict = "collusive" if score > 0.6 else "competitive"

        return {
            "verdict": verdict,
            "score": score,
            "details": {
                "avg_persistence": avg_persistence,
                "avg_high_state_fraction": avg_high_frac,
                "per_player_persistence": persistence_scores,
                "per_player_high_frac": high_state_fracs,
                "num_states": self.num_states,
            },
        }


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def run_all_baselines(
    prices: np.ndarray,
    nash_price: float,
    monopoly_price: float,
) -> Dict[str, Dict[str, Any]]:
    """Run all baseline detection methods and return a dict of results.

    Keys: ``correlation``, ``variance``, ``granger``, ``price_level``, ``markov``.
    """
    results: Dict[str, Dict[str, Any]] = {}

    results["correlation"] = PriceCorrelationScreen(threshold=0.8).detect(prices)
    results["variance"] = VarianceScreen(threshold=0.01).detect(prices)
    results["granger"] = GrangerCausalityScreen(max_lag=5, alpha=0.05).detect(prices)
    results["price_level"] = PriceLevelScreen(
        nash_price=nash_price, threshold_fraction=0.5,
    ).detect(prices, monopoly_price=monopoly_price)
    results["markov"] = MarkovScreen(num_states=10).detect(prices)

    return results


def compare_baselines(
    prices: np.ndarray,
    nash_price: float,
    monopoly_price: float,
    composite_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare all baseline verdicts with the composite test result.

    Returns a comparison table with agreement rates and per-baseline verdicts.
    """
    baselines = run_all_baselines(prices, nash_price, monopoly_price)
    composite_verdict = composite_result.get("verdict", "competitive")

    comparison: Dict[str, Any] = {
        "composite_verdict": composite_verdict,
        "baselines": {},
        "agreement": {},
    }

    agree_count = 0
    total = 0
    for name, bl_result in baselines.items():
        bl_verdict = bl_result.get("verdict", "competitive")
        agrees = bl_verdict == composite_verdict
        comparison["baselines"][name] = {
            "verdict": bl_verdict,
            "score": bl_result.get("score", 0.0),
            "agrees_with_composite": agrees,
        }
        if agrees:
            agree_count += 1
        total += 1

    comparison["agreement"] = {
        "agree_count": agree_count,
        "total_baselines": total,
        "agreement_rate": agree_count / total if total > 0 else 0.0,
    }

    # Majority vote among baselines
    collusive_votes = sum(
        1 for bl in baselines.values() if bl.get("verdict") == "collusive"
    )
    comparison["baseline_majority_vote"] = (
        "collusive" if collusive_votes > total / 2 else "competitive"
    )

    return comparison
