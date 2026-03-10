"""
usability_oracle.statistics.bootstrap — Bootstrap resampling methods.

Provides non-parametric confidence intervals via bootstrap:

- BootstrapCI: main class with percentile, BCa, studentized,
  double-bootstrap, block-bootstrap, and stratified methods.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats as sp_stats

from usability_oracle.statistics.types import (
    BootstrapResult,
    ConfidenceInterval,
)


# Type alias for a statistic function
StatFn = Callable[[np.ndarray], float]


class BootstrapCI:
    """Bootstrap confidence interval estimator.

    Supports multiple CI methods:
    - percentile: basic percentile method
    - BCa: bias-corrected and accelerated
    - studentized: pivot-based with variance estimation
    - double bootstrap: calibrated coverage
    - block bootstrap: for dependent data
    - stratified bootstrap: for stratified samples
    """

    def __init__(
        self,
        n_bootstrap: int = 10_000,
        seed: Optional[int] = None,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------
    # Percentile CI
    # -------------------------------------------------------------------

    def percentile_ci(
        self,
        data: Sequence[float],
        statistic: StatFn,
        n_bootstrap: Optional[int] = None,
        alpha: float = 0.05,
    ) -> BootstrapResult:
        """Percentile bootstrap confidence interval.

        CI = [θ*_{α/2}, θ*_{1−α/2}]

        where θ*_{q} is the q-th quantile of the bootstrap distribution.

        Parameters:
            data: Observed data.
            statistic: Function computing the statistic from a 1-D array.
            n_bootstrap: Number of resamples (overrides instance default).
            alpha: Significance level (CI level = 1 − α).
        """
        arr = np.asarray(data, dtype=np.float64)
        n = len(arr)
        B = n_bootstrap or self.n_bootstrap
        observed = statistic(arr)

        boot_stats = np.empty(B)
        for i in range(B):
            idx = self._rng.integers(0, n, size=n)
            boot_stats[i] = statistic(arr[idx])

        boot_stats.sort()
        lo = float(np.percentile(boot_stats, 100 * alpha / 2))
        hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        bias = float(np.mean(boot_stats)) - observed
        se = float(np.std(boot_stats, ddof=1))

        ci = ConfidenceInterval(
            lower=lo, upper=hi, level=1.0 - alpha,
            point_estimate=observed, method="bootstrap percentile",
        )
        return BootstrapResult(
            statistic_name="custom",
            observed_statistic=observed,
            bootstrap_distribution=tuple(boot_stats.tolist()),
            ci=ci,
            bias=bias,
            standard_error=se,
            num_resamples=B,
            seed=self.seed,
        )

    # -------------------------------------------------------------------
    # BCa CI
    # -------------------------------------------------------------------

    def bca_ci(
        self,
        data: Sequence[float],
        statistic: StatFn,
        n_bootstrap: Optional[int] = None,
        alpha: float = 0.05,
    ) -> BootstrapResult:
        """Bias-corrected and accelerated (BCa) bootstrap CI.

        Corrects for both bias and skewness in the bootstrap distribution:

        z₀ = Φ⁻¹(#{θ* < θ̂} / B)        (bias correction)
        a  = Σ(θ̂₍₋ᵢ₎ − θ̂₍.₎)³ /        (acceleration)
             [6 · (Σ(θ̂₍₋ᵢ₎ − θ̂₍.₎)²)^{3/2}]

        Adjusted quantiles:
        α₁ = Φ(z₀ + (z₀ + z_{α/2}) / (1 − a(z₀ + z_{α/2})))
        α₂ = Φ(z₀ + (z₀ + z_{1−α/2}) / (1 − a(z₀ + z_{1−α/2})))
        """
        arr = np.asarray(data, dtype=np.float64)
        n = len(arr)
        B = n_bootstrap or self.n_bootstrap
        observed = statistic(arr)

        boot_stats = np.empty(B)
        for i in range(B):
            idx = self._rng.integers(0, n, size=n)
            boot_stats[i] = statistic(arr[idx])

        # Bias correction: z0
        prop_less = float(np.mean(boot_stats < observed))
        prop_less = max(1e-10, min(1 - 1e-10, prop_less))
        z0 = float(sp_stats.norm.ppf(prop_less))

        # Acceleration: jackknife
        jackknife_stats = np.empty(n)
        for i in range(n):
            jack_sample = np.delete(arr, i)
            jackknife_stats[i] = statistic(jack_sample)
        jack_mean = float(np.mean(jackknife_stats))
        diffs = jack_mean - jackknife_stats
        num = float(np.sum(diffs ** 3))
        den = 6.0 * float(np.sum(diffs ** 2)) ** 1.5
        a_hat = num / den if den != 0.0 else 0.0

        # Adjusted quantiles
        z_lo = float(sp_stats.norm.ppf(alpha / 2))
        z_hi = float(sp_stats.norm.ppf(1 - alpha / 2))

        def _adj_quantile(z_alpha: float) -> float:
            numer = z0 + z_alpha
            denom = 1.0 - a_hat * numer
            if denom == 0.0:
                denom = 1e-10
            return float(sp_stats.norm.cdf(z0 + numer / denom))

        alpha1 = _adj_quantile(z_lo)
        alpha2 = _adj_quantile(z_hi)

        boot_stats.sort()
        lo = float(np.percentile(boot_stats, 100 * alpha1))
        hi = float(np.percentile(boot_stats, 100 * alpha2))
        bias = float(np.mean(boot_stats)) - observed
        se = float(np.std(boot_stats, ddof=1))

        ci = ConfidenceInterval(
            lower=lo, upper=hi, level=1.0 - alpha,
            point_estimate=observed, method="BCa bootstrap",
        )
        return BootstrapResult(
            statistic_name="custom",
            observed_statistic=observed,
            bootstrap_distribution=tuple(np.sort(boot_stats).tolist()),
            ci=ci,
            bias=bias,
            standard_error=se,
            num_resamples=B,
            seed=self.seed,
        )

    # -------------------------------------------------------------------
    # Studentized bootstrap
    # -------------------------------------------------------------------

    def studentized_bootstrap(
        self,
        data: Sequence[float],
        statistic: StatFn,
        n_bootstrap: Optional[int] = None,
        n_inner: int = 200,
        alpha: float = 0.05,
    ) -> BootstrapResult:
        """Studentized (bootstrap-t) confidence interval.

        Uses a pivot:  t* = (θ* − θ̂) / se*
        CI = [θ̂ − t*_{1−α/2}·se, θ̂ − t*_{α/2}·se]

        The inner bootstrap estimates se* for each outer resample.
        """
        arr = np.asarray(data, dtype=np.float64)
        n = len(arr)
        B = n_bootstrap or self.n_bootstrap
        observed = statistic(arr)

        # Outer bootstrap: compute pivot statistics
        pivots = np.empty(B)
        boot_stats = np.empty(B)
        for i in range(B):
            idx = self._rng.integers(0, n, size=n)
            boot_sample = arr[idx]
            theta_star = statistic(boot_sample)
            boot_stats[i] = theta_star
            # Inner bootstrap for se*
            inner_stats = np.empty(n_inner)
            for j in range(n_inner):
                inner_idx = self._rng.integers(0, n, size=n)
                inner_stats[j] = statistic(boot_sample[inner_idx])
            se_star = float(np.std(inner_stats, ddof=1))
            if se_star < 1e-15:
                se_star = 1e-15
            pivots[i] = (theta_star - observed) / se_star

        # Overall SE from bootstrap
        se = float(np.std(boot_stats, ddof=1))
        pivots.sort()

        # Studentized CI
        q_lo = float(np.percentile(pivots, 100 * (1 - alpha / 2)))
        q_hi = float(np.percentile(pivots, 100 * alpha / 2))
        lo = observed - q_lo * se
        hi = observed - q_hi * se

        bias = float(np.mean(boot_stats)) - observed
        boot_stats.sort()
        ci = ConfidenceInterval(
            lower=lo, upper=hi, level=1.0 - alpha,
            point_estimate=observed, method="studentized bootstrap",
        )
        return BootstrapResult(
            statistic_name="custom",
            observed_statistic=observed,
            bootstrap_distribution=tuple(boot_stats.tolist()),
            ci=ci,
            bias=bias,
            standard_error=se,
            num_resamples=B,
            seed=self.seed,
        )

    # -------------------------------------------------------------------
    # Double bootstrap
    # -------------------------------------------------------------------

    def double_bootstrap(
        self,
        data: Sequence[float],
        statistic: StatFn,
        n_outer: Optional[int] = None,
        n_inner: int = 500,
        alpha: float = 0.05,
    ) -> BootstrapResult:
        """Double bootstrap for CI calibration.

        Uses a two-level bootstrap to calibrate the coverage of the
        percentile CI, producing adjusted quantiles α' such that the
        actual coverage is closer to the nominal level.
        """
        arr = np.asarray(data, dtype=np.float64)
        n = len(arr)
        B_outer = n_outer or self.n_bootstrap
        observed = statistic(arr)

        # Outer bootstrap
        outer_stats = np.empty(B_outer)
        for i in range(B_outer):
            idx = self._rng.integers(0, n, size=n)
            outer_stats[i] = statistic(arr[idx])

        # For each outer resample, compute inner bootstrap CI coverage
        coverage_lo = 0
        coverage_hi = 0
        n_calib = min(B_outer, 500)
        for i in range(n_calib):
            idx = self._rng.integers(0, n, size=n)
            boot_sample = arr[idx]
            theta_star = statistic(boot_sample)
            inner_stats = np.empty(n_inner)
            for j in range(n_inner):
                inner_idx = self._rng.integers(0, n, size=n)
                inner_stats[j] = statistic(boot_sample[inner_idx])
            lo_inner = float(np.percentile(inner_stats, 100 * alpha / 2))
            hi_inner = float(np.percentile(inner_stats, 100 * (1 - alpha / 2)))
            if observed < lo_inner:
                coverage_lo += 1
            if observed > hi_inner:
                coverage_hi += 1

        # Calibrated quantiles
        adj_lo = max(alpha / 2, coverage_lo / n_calib)
        adj_hi = min(1 - alpha / 2, 1 - coverage_hi / n_calib)

        outer_stats.sort()
        lo = float(np.percentile(outer_stats, 100 * adj_lo))
        hi = float(np.percentile(outer_stats, 100 * adj_hi))
        bias = float(np.mean(outer_stats)) - observed
        se = float(np.std(outer_stats, ddof=1))

        ci = ConfidenceInterval(
            lower=lo, upper=hi, level=1.0 - alpha,
            point_estimate=observed, method="double bootstrap",
        )
        return BootstrapResult(
            statistic_name="custom",
            observed_statistic=observed,
            bootstrap_distribution=tuple(outer_stats.tolist()),
            ci=ci,
            bias=bias,
            standard_error=se,
            num_resamples=B_outer,
            seed=self.seed,
        )

    # -------------------------------------------------------------------
    # Block bootstrap
    # -------------------------------------------------------------------

    def block_bootstrap(
        self,
        data: Sequence[float],
        statistic: StatFn,
        block_size: int = 5,
        n_bootstrap: Optional[int] = None,
        alpha: float = 0.05,
    ) -> BootstrapResult:
        """Block bootstrap for time-series or dependent data.

        Resamples non-overlapping blocks of consecutive observations
        to preserve local dependence structure.

        Parameters:
            data: Time-series observations.
            statistic: Statistic function.
            block_size: Size of each block.
            n_bootstrap: Number of resamples.
            alpha: Significance level.
        """
        arr = np.asarray(data, dtype=np.float64)
        n = len(arr)
        B = n_bootstrap or self.n_bootstrap
        observed = statistic(arr)

        if block_size < 1:
            raise ValueError("block_size must be >= 1.")
        if block_size > n:
            block_size = n

        n_blocks = math.ceil(n / block_size)
        max_start = n - block_size

        boot_stats = np.empty(B)
        for i in range(B):
            starts = self._rng.integers(0, max_start + 1, size=n_blocks)
            blocks = [arr[s: s + block_size] for s in starts]
            boot_sample = np.concatenate(blocks)[:n]
            boot_stats[i] = statistic(boot_sample)

        boot_stats.sort()
        lo = float(np.percentile(boot_stats, 100 * alpha / 2))
        hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        bias = float(np.mean(boot_stats)) - observed
        se = float(np.std(boot_stats, ddof=1))

        ci = ConfidenceInterval(
            lower=lo, upper=hi, level=1.0 - alpha,
            point_estimate=observed, method="block bootstrap",
        )
        return BootstrapResult(
            statistic_name="custom",
            observed_statistic=observed,
            bootstrap_distribution=tuple(boot_stats.tolist()),
            ci=ci,
            bias=bias,
            standard_error=se,
            num_resamples=B,
            seed=self.seed,
        )

    # -------------------------------------------------------------------
    # Stratified bootstrap
    # -------------------------------------------------------------------

    def stratified_bootstrap(
        self,
        data: Sequence[float],
        strata: Sequence[int],
        statistic: StatFn,
        n_bootstrap: Optional[int] = None,
        alpha: float = 0.05,
    ) -> BootstrapResult:
        """Stratified bootstrap — resample within each stratum.

        Ensures that each stratum maintains its proportion in each
        bootstrap sample, reducing variance when the strata are
        heterogeneous.

        Parameters:
            data: Observed data.
            strata: Integer stratum label for each observation.
            statistic: Statistic function.
            n_bootstrap: Number of resamples.
            alpha: Significance level.
        """
        arr = np.asarray(data, dtype=np.float64)
        strata_arr = np.asarray(strata, dtype=np.int64)
        if len(arr) != len(strata_arr):
            raise ValueError("data and strata must have the same length.")
        n = len(arr)
        B = n_bootstrap or self.n_bootstrap
        observed = statistic(arr)

        unique_strata = np.unique(strata_arr)
        stratum_indices = {
            s: np.where(strata_arr == s)[0] for s in unique_strata
        }

        boot_stats = np.empty(B)
        for i in range(B):
            parts = []
            for s in unique_strata:
                indices = stratum_indices[s]
                k = len(indices)
                resampled_idx = self._rng.choice(indices, size=k, replace=True)
                parts.append(arr[resampled_idx])
            boot_sample = np.concatenate(parts)
            boot_stats[i] = statistic(boot_sample)

        boot_stats.sort()
        lo = float(np.percentile(boot_stats, 100 * alpha / 2))
        hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        bias = float(np.mean(boot_stats)) - observed
        se = float(np.std(boot_stats, ddof=1))

        ci = ConfidenceInterval(
            lower=lo, upper=hi, level=1.0 - alpha,
            point_estimate=observed, method="stratified bootstrap",
        )
        return BootstrapResult(
            statistic_name="custom",
            observed_statistic=observed,
            bootstrap_distribution=tuple(boot_stats.tolist()),
            ci=ci,
            bias=bias,
            standard_error=se,
            num_resamples=B,
            seed=self.seed,
        )
