"""
usability_oracle.analysis.convergence — Convergence analysis for Monte Carlo sampling.

Provides diagnostics to determine whether Monte Carlo estimates (e.g. expected
cost, policy value, sensitivity indices) have converged, including effective
sample size, Gelman-Rubin R-hat, batch means estimation, and sequential
stopping rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceResult:
    """Summary of convergence diagnostics.

    Attributes:
        converged: Whether the estimate is considered converged.
        n_samples: Number of samples analysed.
        effective_sample_size: ESS accounting for autocorrelation.
        r_hat: Gelman-Rubin R-hat statistic (should be ≤ 1.05).
        relative_error: Estimated relative MCSE.
        batch_means_se: Standard error from batch means.
        running_estimates: Cumulative estimates for convergence plotting.
    """
    converged: bool = False
    n_samples: int = 0
    effective_sample_size: float = 0.0
    r_hat: float = float("inf")
    relative_error: float = float("inf")
    batch_means_se: float = float("inf")
    running_estimates: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        return (
            f"Convergence: {status}\n"
            f"  N samples: {self.n_samples}\n"
            f"  ESS:       {self.effective_sample_size:.1f}\n"
            f"  R-hat:     {self.r_hat:.4f}\n"
            f"  Rel. err:  {self.relative_error:.6f}\n"
            f"  Batch SE:  {self.batch_means_se:.6f}"
        )


@dataclass
class SequentialStopResult:
    """Result from sequential stopping rule analysis."""
    should_stop: bool = False
    current_estimate: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    samples_used: int = 0
    relative_half_width: float = float("inf")


# ---------------------------------------------------------------------------
# Autocorrelation helpers
# ---------------------------------------------------------------------------

def _autocorrelation(x: np.ndarray, max_lag: int = -1) -> np.ndarray:
    """Compute normalised autocorrelation function up to max_lag."""
    n = len(x)
    if max_lag < 0:
        max_lag = min(n // 2, 500)
    x_centered = x - np.mean(x)
    var = np.var(x)
    if var < 1e-15:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = float(np.sum(x_centered[:n - lag] * x_centered[lag:])) / ((n - lag) * var)
    return acf


def _effective_sample_size(x: np.ndarray) -> float:
    """Compute effective sample size using initial monotone sequence estimator.

    ESS = N / (1 + 2 * sum of autocorrelations).
    Uses the initial positive sequence estimator (Geyer 1992).
    """
    n = len(x)
    if n < 4:
        return float(n)

    acf = _autocorrelation(x, max_lag=n // 2)
    # Sum pairs of consecutive autocorrelations, stop when sum goes negative
    tau = 1.0
    k = 1
    while k + 1 < len(acf):
        pair_sum = acf[k] + acf[k + 1]
        if pair_sum <= 0:
            break
        tau += 2.0 * pair_sum
        k += 2

    ess = n / max(tau, 1.0)
    return max(ess, 1.0)


# ---------------------------------------------------------------------------
# Gelman-Rubin R-hat
# ---------------------------------------------------------------------------

def _gelman_rubin(chains: list[np.ndarray]) -> float:
    """Compute the Gelman-Rubin R-hat statistic for multiple chains.

    R-hat = sqrt((n-1)/n + B/(n*W)) where B is between-chain variance
    and W is within-chain variance.  Values close to 1.0 indicate convergence.
    """
    m = len(chains)
    if m < 2:
        return 1.0

    n_vals = [len(c) for c in chains]
    n = min(n_vals)
    if n < 2:
        return float("inf")

    # Trim chains to same length
    trimmed = [c[:n] for c in chains]
    chain_means = np.array([np.mean(c) for c in trimmed])
    chain_vars = np.array([np.var(c, ddof=1) for c in trimmed])

    overall_mean = np.mean(chain_means)
    B = n * np.var(chain_means, ddof=1)  # between-chain variance
    W = np.mean(chain_vars)  # within-chain variance

    if W < 1e-15:
        return 1.0 if B < 1e-15 else float("inf")

    var_hat = ((n - 1) / n) * W + B / n
    r_hat = math.sqrt(var_hat / W)
    return r_hat


# ---------------------------------------------------------------------------
# Batch means
# ---------------------------------------------------------------------------

def _batch_means_se(x: np.ndarray, n_batches: int = 30) -> float:
    """Estimate standard error using the method of batch means.

    Splits the chain into n_batches contiguous blocks and computes
    the standard error of the batch means.
    """
    n = len(x)
    if n < n_batches * 2:
        n_batches = max(2, n // 2)

    batch_size = n // n_batches
    if batch_size < 1:
        return float(np.std(x)) / max(math.sqrt(n), 1.0)

    batch_means = np.array([
        np.mean(x[i * batch_size:(i + 1) * batch_size])
        for i in range(n_batches)
    ])

    se = float(np.std(batch_means, ddof=1)) / math.sqrt(n_batches)
    return se


# ---------------------------------------------------------------------------
# Running estimate
# ---------------------------------------------------------------------------

def _running_mean(x: np.ndarray, step: int = 1) -> np.ndarray:
    """Compute the running (cumulative) mean at specified step intervals."""
    n = len(x)
    cum = np.cumsum(x)
    indices = np.arange(step - 1, n, step)
    running = cum[indices] / (indices + 1)
    return running


# ---------------------------------------------------------------------------
# ConvergenceAnalyzer
# ---------------------------------------------------------------------------

class ConvergenceAnalyzer:
    """Diagnose convergence of Monte Carlo estimates.

    Parameters:
        r_hat_threshold: Maximum acceptable R-hat value.
        relative_error_threshold: Maximum acceptable relative MCSE.
        min_ess: Minimum effective sample size for convergence.
    """

    def __init__(
        self,
        r_hat_threshold: float = 1.05,
        relative_error_threshold: float = 0.05,
        min_ess: float = 400.0,
    ) -> None:
        self._r_hat_threshold = r_hat_threshold
        self._rel_err_threshold = relative_error_threshold
        self._min_ess = min_ess

    # ------------------------------------------------------------------
    # Full diagnostic
    # ------------------------------------------------------------------

    def diagnose(
        self,
        samples: np.ndarray | list[np.ndarray],
        compute_running: bool = True,
    ) -> ConvergenceResult:
        """Run full convergence diagnostics on one or more chains.

        Parameters:
            samples: Either a 1-D array (single chain) or a list of 1-D
                arrays (multiple chains).
            compute_running: If True, compute running estimates for plotting.
        """
        if isinstance(samples, np.ndarray) and samples.ndim == 1:
            chains = [samples]
        elif isinstance(samples, list):
            chains = [np.asarray(c, dtype=float) for c in samples]
        else:
            chains = [np.asarray(samples, dtype=float).ravel()]

        all_samples = np.concatenate(chains)
        n_total = len(all_samples)
        if n_total < 4:
            return ConvergenceResult(n_samples=n_total)

        # ESS from combined chain
        ess = _effective_sample_size(all_samples)

        # R-hat from multiple chains (split single chain in half if only one)
        if len(chains) == 1:
            mid = n_total // 2
            split_chains = [all_samples[:mid], all_samples[mid:]]
        else:
            split_chains = chains
        r_hat = _gelman_rubin(split_chains)

        # Batch means SE
        bm_se = _batch_means_se(all_samples)

        # Relative error
        mean_val = float(np.mean(all_samples))
        mc_se = float(np.std(all_samples)) / max(math.sqrt(ess), 1.0)
        rel_err = abs(mc_se / mean_val) if abs(mean_val) > 1e-12 else mc_se

        # Running estimates
        running = None
        if compute_running:
            step = max(1, n_total // 500)
            running = _running_mean(all_samples, step)

        converged = (
            r_hat <= self._r_hat_threshold
            and rel_err <= self._rel_err_threshold
            and ess >= self._min_ess
        )

        return ConvergenceResult(
            converged=converged,
            n_samples=n_total,
            effective_sample_size=ess,
            r_hat=r_hat,
            relative_error=rel_err,
            batch_means_se=bm_se,
            running_estimates=running,
            metadata={
                "n_chains": len(chains),
                "mean": mean_val,
                "mc_se": mc_se,
            },
        )

    # ------------------------------------------------------------------
    # Sequential stopping rule
    # ------------------------------------------------------------------

    def sequential_check(
        self,
        samples: np.ndarray,
        target_half_width: float = 0.05,
        confidence: float = 0.95,
    ) -> SequentialStopResult:
        """Apply a sequential stopping rule based on relative half-width.

        Stops when the confidence interval half-width relative to the
        estimate is below target_half_width.
        """
        x = np.asarray(samples, dtype=float)
        n = len(x)
        if n < 10:
            return SequentialStopResult(samples_used=n)

        mean = float(np.mean(x))
        ess = _effective_sample_size(x)
        se = float(np.std(x, ddof=1)) / max(math.sqrt(ess), 1.0)

        # Use normal approximation for CI
        z = _z_score(confidence)
        half_width = z * se
        rel_hw = abs(half_width / mean) if abs(mean) > 1e-12 else abs(half_width)

        return SequentialStopResult(
            should_stop=rel_hw <= target_half_width,
            current_estimate=mean,
            confidence_interval=(mean - half_width, mean + half_width),
            samples_used=n,
            relative_half_width=rel_hw,
        )

    # ------------------------------------------------------------------
    # Geweke diagnostic
    # ------------------------------------------------------------------

    @staticmethod
    def geweke_test(
        samples: np.ndarray,
        first_frac: float = 0.1,
        last_frac: float = 0.5,
    ) -> tuple[float, float]:
        """Geweke's convergence diagnostic comparing chain start and end.

        Returns (z_score, p_value).  A significant p-value (< 0.05)
        suggests the chain has not converged.
        """
        x = np.asarray(samples, dtype=float)
        n = len(x)
        n_first = max(2, int(n * first_frac))
        n_last = max(2, int(n * last_frac))

        first = x[:n_first]
        last = x[n - n_last:]

        se_first = _spectral_density_at_zero(first)
        se_last = _spectral_density_at_zero(last)

        denom = math.sqrt(se_first / n_first + se_last / n_last)
        if denom < 1e-15:
            return 0.0, 1.0

        z = (np.mean(first) - np.mean(last)) / denom
        p = 2.0 * (1.0 - _normal_cdf(abs(z)))
        return float(z), p

    # ------------------------------------------------------------------
    # Heidelberger-Welch stationarity test
    # ------------------------------------------------------------------

    @staticmethod
    def heidelberger_welch(
        samples: np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Simplified Heidelberger-Welch stationarity and half-width test.

        Returns dict with 'stationary' (bool), 'half_width_ok' (bool),
        'start_index' (int) and 'half_width' (float).
        """
        x = np.asarray(samples, dtype=float)
        n = len(x)
        if n < 20:
            return {"stationary": False, "half_width_ok": False, "start_index": 0, "half_width": float("inf")}

        # Stationarity: apply Cramer-von-Mises-like test on running means
        # Simplified: check if first 10% mean differs from last 50% mean
        start_idx = 0
        for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            idx = int(n * frac)
            sub = x[idx:]
            if len(sub) < 10:
                continue
            first_half = sub[:len(sub) // 2]
            second_half = sub[len(sub) // 2:]
            se = _spectral_density_at_zero(sub)
            if se < 1e-15:
                start_idx = idx
                break
            diff = abs(np.mean(first_half) - np.mean(second_half))
            threshold = _z_score(1 - alpha) * math.sqrt(2 * se / len(sub))
            if diff < threshold:
                start_idx = idx
                break

        # Half-width test on the stationary portion
        stationary_samples = x[start_idx:]
        mean = float(np.mean(stationary_samples))
        se = _spectral_density_at_zero(stationary_samples)
        half_width = _z_score(1 - alpha / 2) * math.sqrt(se / len(stationary_samples))
        hw_ok = abs(half_width / mean) < 0.1 if abs(mean) > 1e-12 else half_width < 0.1

        return {
            "stationary": start_idx < n // 2,
            "half_width_ok": hw_ok,
            "start_index": start_idx,
            "half_width": half_width,
        }

    # ------------------------------------------------------------------
    # Raftery-Lewis diagnostic
    # ------------------------------------------------------------------

    @staticmethod
    def raftery_lewis(
        samples: np.ndarray,
        quantile: float = 0.025,
        accuracy: float = 0.005,
        probability: float = 0.95,
    ) -> dict[str, Any]:
        """Simplified Raftery-Lewis diagnostic for required sample size.

        Estimates the total number of iterations needed to estimate
        a given quantile to a specified accuracy with desired probability.
        """
        x = np.asarray(samples, dtype=float)
        n = len(x)

        # Binary indicator for exceeding the quantile
        threshold = float(np.quantile(x, quantile))
        z = (x <= threshold).astype(float)

        # Estimate thinning interval from autocorrelation
        acf = _autocorrelation(z, max_lag=min(100, n // 2))
        thin = 1
        for lag in range(1, len(acf)):
            if abs(acf[lag]) < 0.05:
                thin = lag
                break

        # Estimate transition probabilities for thinned chain
        z_thin = z[::max(thin, 1)]
        n_thin = len(z_thin)
        if n_thin < 4:
            return {"thin": thin, "burn_in": 0, "total_needed": n, "dependence_factor": 1.0}

        # Transition counts
        n_00 = n_01 = n_10 = n_11 = 0
        for i in range(n_thin - 1):
            if z_thin[i] == 0 and z_thin[i + 1] == 0:
                n_00 += 1
            elif z_thin[i] == 0 and z_thin[i + 1] == 1:
                n_01 += 1
            elif z_thin[i] == 1 and z_thin[i + 1] == 0:
                n_10 += 1
            else:
                n_11 += 1

        alpha_hat = n_01 / max(n_00 + n_01, 1)
        beta_hat = n_10 / max(n_10 + n_11, 1)

        # Required sample size (Raftery and Lewis 1992)
        z_val = _z_score(probability)
        phi = _normal_cdf(z_val) * (1 - _normal_cdf(z_val))
        n_min = int(math.ceil(z_val ** 2 * phi / (accuracy ** 2)))

        # Dependence factor
        if alpha_hat + beta_hat > 0:
            dep_factor = (2 - alpha_hat - beta_hat) / (alpha_hat + beta_hat)
            dep_factor = max(dep_factor, 1.0)
        else:
            dep_factor = 1.0

        total_needed = int(math.ceil(n_min * dep_factor * thin))
        burn_in = int(math.ceil(
            math.log(0.001 * (alpha_hat + beta_hat) / max(alpha_hat, 1e-10))
            / math.log(abs(1 - alpha_hat - beta_hat) + 1e-10)
        )) * thin if (alpha_hat + beta_hat) < 1 else 0

        return {
            "thin": thin,
            "burn_in": max(burn_in, 0),
            "total_needed": total_needed,
            "dependence_factor": dep_factor,
            "n_min_iid": n_min,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _spectral_density_at_zero(x: np.ndarray) -> float:
    """Estimate the spectral density at frequency zero using a Bartlett window."""
    n = len(x)
    if n < 4:
        return float(np.var(x))
    max_lag = int(math.sqrt(n))
    acf = _autocorrelation(x, max_lag)
    # Bartlett windowed estimate
    s = float(np.var(x))
    for k in range(1, max_lag + 1):
        weight = 1.0 - k / (max_lag + 1)
        s += 2.0 * weight * acf[k] * float(np.var(x))
    return max(s, 0.0)


def _z_score(confidence: float) -> float:
    """Approximate z-score for a given confidence level using rational approximation."""
    p = (1.0 + confidence) / 2.0
    if p >= 1.0:
        return 5.0
    if p <= 0.0:
        return -5.0
    # Rational approximation (Abramowitz and Stegun 26.2.23)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t ** 3)
    return z


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
