"""MCMC convergence diagnostics for causal structure sampling.

Provides the Gelman-Rubin R-hat statistic for multi-chain convergence
assessment, effective sample size estimation, trace analysis for
stationarity detection, and autocorrelation diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Gelman-Rubin R-hat
# ---------------------------------------------------------------------------

@dataclass
class GelmanRubinResult:
    """Result of a Gelman-Rubin convergence diagnostic."""
    r_hat: float
    between_chain_var: float
    within_chain_var: float
    converged: bool
    threshold: float


class GelmanRubin:
    r"""Gelman-Rubin :math:`\hat R` convergence diagnostic.

    Compares between-chain and within-chain variance from multiple
    independent MCMC chains.  Values close to 1 indicate convergence;
    the conventional threshold is 1.1 (or 1.05 for stricter criteria).

    Parameters
    ----------
    threshold : float
        Convergence threshold for :math:`\hat R`.  Default 1.1.
    """

    def __init__(self, threshold: float = 1.1) -> None:
        if threshold <= 1.0:
            raise ValueError("threshold must be > 1.0")
        self.threshold = threshold

    def compute(self, chains: Sequence[Sequence[float]]) -> GelmanRubinResult:
        r"""Compute :math:`\hat R` from multiple chains.

        Parameters
        ----------
        chains : sequence of sequences of float
            Each element is a trace (sequence of scalar values) from one
            MCMC chain.  All chains must have the same length.

        Returns
        -------
        GelmanRubinResult
        """
        m = len(chains)
        if m < 2:
            raise ValueError("Need at least 2 chains for Gelman-Rubin.")
        n = len(chains[0])
        if n < 2:
            raise ValueError("Chains must have at least 2 samples.")
        for c in chains:
            if len(c) != n:
                raise ValueError("All chains must have the same length.")

        chain_means = np.array([np.mean(c) for c in chains])
        overall_mean = np.mean(chain_means)

        # Between-chain variance B
        B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)

        # Within-chain variance W
        chain_vars = np.array([np.var(c, ddof=1) for c in chains])
        W = np.mean(chain_vars)

        # Pooled variance estimate
        var_hat = (1.0 - 1.0 / n) * W + B / n

        r_hat = math.sqrt(var_hat / W) if W > 0 else float("inf")

        return GelmanRubinResult(
            r_hat=r_hat,
            between_chain_var=float(B),
            within_chain_var=float(W),
            converged=r_hat < self.threshold,
            threshold=self.threshold,
        )

    def compute_multivariate(
        self,
        chains: Sequence[Sequence[Sequence[float]]],
    ) -> List[GelmanRubinResult]:
        """Compute per-dimension R-hat for multivariate chains.

        Parameters
        ----------
        chains : list of list of list of float
            ``chains[c][t][d]`` = dimension *d* of sample *t* of chain *c*.

        Returns
        -------
        list of GelmanRubinResult
            One result per dimension.
        """
        m = len(chains)
        d = len(chains[0][0])
        results: List[GelmanRubinResult] = []
        for dim in range(d):
            scalar_chains = [[s[dim] for s in chain] for chain in chains]
            results.append(self.compute(scalar_chains))
        return results

    def check_all_converged(
        self,
        chains: Sequence[Sequence[float]],
        split: bool = True,
    ) -> Tuple[bool, float]:
        """Convenience method: check convergence, optionally with split chains.

        When *split* is True, each chain is split in half and the resulting
        doubled number of shorter chains are used — this additionally checks
        for within-chain non-stationarity.

        Returns
        -------
        (converged, r_hat) tuple
        """
        if split:
            split_chains: List[List[float]] = []
            for c in chains:
                half = len(c) // 2
                if half < 2:
                    split_chains.append(list(c))
                else:
                    split_chains.append(list(c[:half]))
                    split_chains.append(list(c[half:2 * half]))
            result = self.compute(split_chains)
        else:
            result = self.compute(chains)

        return result.converged, result.r_hat


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------

@dataclass
class ESSResult:
    """Result of an effective sample size computation."""
    ess: float
    n_samples: int
    ratio: float
    autocorrelation_time: float


class EffectiveSampleSize:
    """Estimate the effective sample size of an MCMC chain.

    Uses the initial positive sequence estimator (Geyer 1992) to compute
    the integrated autocorrelation time, from which ESS is derived.

    Parameters
    ----------
    max_lag : int or None
        Maximum lag for autocorrelation computation.  If ``None``, uses
        ``n // 2``.
    """

    def __init__(self, max_lag: Optional[int] = None) -> None:
        self.max_lag = max_lag

    @staticmethod
    def _autocorrelation(x: npt.NDArray[np.float64], max_lag: int) -> npt.NDArray[np.float64]:
        """Compute the normalised autocorrelation function up to *max_lag*."""
        n = len(x)
        x_centered = x - x.mean()
        var = np.var(x)
        if var < 1e-15:
            return np.zeros(max_lag + 1)
        acf = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            acf[lag] = np.mean(x_centered[:n - lag] * x_centered[lag:]) / var
        return acf

    def compute(self, chain: Sequence[float]) -> ESSResult:
        """Compute the effective sample size of a single chain.

        Parameters
        ----------
        chain : sequence of float
            Scalar MCMC trace.

        Returns
        -------
        ESSResult
        """
        x = np.array(chain, dtype=np.float64)
        n = len(x)
        if n < 4:
            return ESSResult(ess=float(n), n_samples=n, ratio=1.0, autocorrelation_time=1.0)

        max_lag = self.max_lag if self.max_lag is not None else n // 2
        max_lag = min(max_lag, n - 1)
        acf = self._autocorrelation(x, max_lag)

        # Initial positive sequence estimator (Geyer 1992)
        # Sum consecutive pairs of autocorrelations until first negative pair
        tau = 1.0
        for lag in range(1, max_lag, 2):
            pair_sum = acf[lag]
            if lag + 1 <= max_lag:
                pair_sum += acf[lag + 1]
            if pair_sum < 0:
                break
            tau += 2.0 * pair_sum

        tau = max(tau, 1.0)
        ess = n / tau
        return ESSResult(
            ess=ess,
            n_samples=n,
            ratio=ess / n,
            autocorrelation_time=tau,
        )

    def compute_multi_chain(self, chains: Sequence[Sequence[float]]) -> ESSResult:
        """Compute combined ESS across multiple chains.

        Concatenates chains and estimates ESS, then adjusts by comparing
        within-chain and between-chain variance.
        """
        all_values: List[float] = []
        for c in chains:
            all_values.extend(c)

        combined = self.compute(all_values)

        # Also compute per-chain ESS and sum
        per_chain_ess = sum(self.compute(c).ess for c in chains)

        # Take the minimum of combined and sum-of-per-chain estimates
        ess = min(combined.ess, per_chain_ess)
        n_total = len(all_values)
        return ESSResult(
            ess=ess,
            n_samples=n_total,
            ratio=ess / max(n_total, 1),
            autocorrelation_time=n_total / max(ess, 1e-12),
        )

    def minimum_ess_check(
        self,
        chains: Sequence[Sequence[float]],
        min_ess: float = 100.0,
    ) -> Tuple[bool, float]:
        """Check if the multi-chain ESS exceeds a minimum threshold.

        Returns
        -------
        (sufficient, ess) tuple
        """
        result = self.compute_multi_chain(chains)
        return result.ess >= min_ess, result.ess


# ---------------------------------------------------------------------------
# Trace analysis
# ---------------------------------------------------------------------------

@dataclass
class TraceAnalysisResult:
    """Result of trace stationarity analysis."""
    stationary: bool
    mean_first_half: float
    mean_second_half: float
    std_first_half: float
    std_second_half: float
    relative_change: float
    geweke_z: float
    geweke_converged: bool


class TraceAnalysis:
    """Stationarity analysis for MCMC traces.

    Provides the Geweke diagnostic (comparison of the first *a* fraction
    and last *b* fraction of the chain) and simple split-half tests.

    Parameters
    ----------
    first_fraction : float
        Fraction of the chain to use as the "early" segment.
    last_fraction : float
        Fraction of the chain to use as the "late" segment.
    z_threshold : float
        Critical z-value for the Geweke test (default 1.96 ≈ 5 % two-sided).
    """

    def __init__(
        self,
        first_fraction: float = 0.1,
        last_fraction: float = 0.5,
        z_threshold: float = 1.96,
    ) -> None:
        if first_fraction + last_fraction > 1.0:
            raise ValueError("first_fraction + last_fraction must be <= 1.0")
        self.first_fraction = first_fraction
        self.last_fraction = last_fraction
        self.z_threshold = z_threshold

    def analyse(self, trace: Sequence[float]) -> TraceAnalysisResult:
        """Run stationarity diagnostics on *trace*.

        Parameters
        ----------
        trace : sequence of float
            Scalar MCMC trace.

        Returns
        -------
        TraceAnalysisResult
        """
        x = np.array(trace, dtype=np.float64)
        n = len(x)
        if n < 4:
            return TraceAnalysisResult(
                stationary=True,
                mean_first_half=float(x.mean()),
                mean_second_half=float(x.mean()),
                std_first_half=0.0,
                std_second_half=0.0,
                relative_change=0.0,
                geweke_z=0.0,
                geweke_converged=True,
            )

        # Split-half comparison
        half = n // 2
        first_half = x[:half]
        second_half = x[half:]

        m1, m2 = float(first_half.mean()), float(second_half.mean())
        s1, s2 = float(first_half.std()), float(second_half.std())
        scale = max(abs(m1), abs(m2), 1e-12)
        rel_change = abs(m2 - m1) / scale

        # Geweke diagnostic
        na = max(1, int(n * self.first_fraction))
        nb = max(1, int(n * self.last_fraction))
        early = x[:na]
        late = x[n - nb:]

        se_early = np.std(early) / max(math.sqrt(na), 1.0)
        se_late = np.std(late) / max(math.sqrt(nb), 1.0)
        se_total = math.sqrt(se_early ** 2 + se_late ** 2)

        if se_total > 1e-15:
            z = (float(early.mean()) - float(late.mean())) / se_total
        else:
            z = 0.0

        geweke_ok = abs(z) < self.z_threshold
        stationary = geweke_ok and rel_change < 0.1

        return TraceAnalysisResult(
            stationary=stationary,
            mean_first_half=m1,
            mean_second_half=m2,
            std_first_half=s1,
            std_second_half=s2,
            relative_change=rel_change,
            geweke_z=z,
            geweke_converged=geweke_ok,
        )

    def find_burn_in(
        self,
        trace: Sequence[float],
        window_size: int = 50,
    ) -> int:
        """Heuristically determine burn-in length.

        Slides a window across the trace and detects when the running mean
        stabilises (relative change < 5 % over consecutive windows).

        Parameters
        ----------
        trace : sequence of float
            Scalar trace.
        window_size : int
            Size of the sliding window.

        Returns
        -------
        int
            Recommended burn-in length (number of samples to discard).
        """
        x = np.array(trace, dtype=np.float64)
        n = len(x)
        if n < 2 * window_size:
            return 0

        # Compute running means
        running = np.convolve(x, np.ones(window_size) / window_size, mode="valid")
        if len(running) < 2:
            return 0

        # Find first point where consecutive windows are within 5% of each other
        for i in range(1, len(running)):
            scale = max(abs(running[i]), abs(running[i - 1]), 1e-12)
            if abs(running[i] - running[i - 1]) / scale < 0.05:
                return max(0, i + window_size)

        return n // 2


# ---------------------------------------------------------------------------
# Autocorrelation analysis
# ---------------------------------------------------------------------------

@dataclass
class AutocorrelationResult:
    """Result of autocorrelation analysis."""
    lags: npt.NDArray[np.int64]
    autocorrelations: npt.NDArray[np.float64]
    first_negative_lag: int
    integrated_autocorrelation_time: float
    thinning_recommendation: int


class AutocorrelationAnalysis:
    """Autocorrelation function analysis for MCMC chains.

    Computes the autocorrelation function and derives thinning
    recommendations to produce approximately independent samples.

    Parameters
    ----------
    max_lag : int or None
        Maximum lag to compute.  ``None`` uses ``n // 4``.
    """

    def __init__(self, max_lag: Optional[int] = None) -> None:
        self.max_lag = max_lag

    def analyse(self, chain: Sequence[float]) -> AutocorrelationResult:
        """Compute autocorrelation diagnostics for *chain*.

        Parameters
        ----------
        chain : sequence of float
            Scalar MCMC trace.

        Returns
        -------
        AutocorrelationResult
        """
        x = np.array(chain, dtype=np.float64)
        n = len(x)
        max_lag = self.max_lag if self.max_lag is not None else n // 4
        max_lag = max(min(max_lag, n - 1), 1)

        acf = EffectiveSampleSize._autocorrelation(x, max_lag)
        lags = np.arange(max_lag + 1, dtype=np.int64)

        # First negative autocorrelation
        first_neg = max_lag + 1
        for k in range(1, max_lag + 1):
            if acf[k] < 0:
                first_neg = k
                break

        # Integrated autocorrelation time (sum up to first negative)
        tau = 1.0 + 2.0 * np.sum(acf[1:first_neg])
        tau = max(tau, 1.0)

        # Thinning recommendation: thin by ceil(tau) to get ~independent samples
        thinning = max(1, int(math.ceil(tau)))

        return AutocorrelationResult(
            lags=lags,
            autocorrelations=acf,
            first_negative_lag=first_neg,
            integrated_autocorrelation_time=tau,
            thinning_recommendation=thinning,
        )

    def cross_chain_correlation(
        self,
        chains: Sequence[Sequence[float]],
    ) -> npt.NDArray[np.float64]:
        """Compute pairwise correlation matrix between chain means.

        Low correlations indicate good mixing / independence of chains.

        Parameters
        ----------
        chains : sequence of sequences
            Multiple MCMC traces.

        Returns
        -------
        npt.NDArray[np.float64]
            m × m correlation matrix where m = number of chains.
        """
        m = len(chains)
        means = np.array([np.array(c) for c in chains])
        # Compute pairwise Pearson correlations
        corr = np.corrcoef(means)
        if corr.ndim == 0:
            corr = np.array([[1.0]])
        return corr

    def suggest_additional_samples(
        self,
        chain: Sequence[float],
        target_ess: float = 100.0,
    ) -> int:
        """Estimate how many additional samples are needed to achieve the
        target effective sample size.

        Parameters
        ----------
        chain : sequence of float
            Current MCMC trace.
        target_ess : float
            Desired effective sample size.

        Returns
        -------
        int
            Estimated number of additional iterations needed.  Returns 0 if
            the target ESS is already met.
        """
        ess_calc = EffectiveSampleSize(max_lag=self.max_lag)
        ess_result = ess_calc.compute(chain)
        current_ess = ess_result.ess
        n = len(chain)
        tau = ess_result.autocorrelation_time

        if current_ess >= target_ess:
            return 0

        # ESS ≈ n / tau → need n_needed = target_ess * tau
        n_needed = target_ess * tau
        additional = int(math.ceil(n_needed - n))
        return max(0, additional)
