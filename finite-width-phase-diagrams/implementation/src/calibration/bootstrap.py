"""Bootstrap confidence intervals for finite-width phase diagram calibration.

Provides bootstrap-based uncertainty quantification for NTK regression
parameters and phase boundary locations.  Implements percentile, BCa
(bias-corrected and accelerated), block, and seed-resampling bootstrap
schemes together with implicit-function-theorem propagation of parameter
uncertainty to phase boundaries.

Mathematical background
-----------------------
Given i.i.d. samples X_1, …, X_n and a statistic θ̂ = T(X_1, …, X_n),
the bootstrap distribution {θ̂*_b} for b = 1, …, B approximates the
sampling distribution of θ̂.  The BCa interval corrects for bias z_0
and skewness (acceleration a) via

    α_1 = Φ(z_0 + (z_0 + z_α) / (1 - a (z_0 + z_α)))
    α_2 = Φ(z_0 + (z_0 + z_{1-α}) / (1 - a (z_0 + z_{1-α})))

where Φ is the standard-normal CDF and z_α = Φ^{-1}(α).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


# ======================================================================
#  Data containers
# ======================================================================


@dataclass
class BootstrapResult:
    """Container for bootstrap confidence interval results.

    Attributes
    ----------
    point_estimate : ndarray
        Original (non-bootstrapped) point estimate of the statistic.
    ci_lower : ndarray
        Lower confidence bound.
    ci_upper : ndarray
        Upper confidence bound.
    confidence_level : float
        Nominal confidence level, e.g. 0.95.
    method : str
        Bootstrap method used (``"percentile"``, ``"bca"``, ``"block"``,
        ``"seed"``).
    n_bootstrap : int
        Number of bootstrap replicates.
    bootstrap_distribution : ndarray
        Full array of bootstrap replicates, shape ``(n_bootstrap, ...)``.
    bias : float
        Estimated bootstrap bias, ``E[θ̂*] - θ̂``.
    acceleration : ndarray
        BCa acceleration constant *a*.  Zero for percentile method.
    se : ndarray
        Bootstrap standard error of the statistic.
    converged : bool
        Whether the bootstrap distribution has converged (assessed via
        :meth:`BootstrapCI.convergence_diagnostic`).
    """

    point_estimate: NDArray[np.floating]
    ci_lower: NDArray[np.floating]
    ci_upper: NDArray[np.floating]
    confidence_level: float
    method: str
    n_bootstrap: int
    bootstrap_distribution: NDArray[np.floating]
    bias: float
    acceleration: NDArray[np.floating]
    se: NDArray[np.floating]
    converged: bool


# ======================================================================
#  Core bootstrap engine
# ======================================================================


class BootstrapCI:
    """Bootstrap confidence interval calculator.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap resamples (default 1000).
    confidence_level : float
        Nominal coverage probability (default 0.95).
    random_state : int | np.random.Generator | None
        Seed or Generator for reproducibility.
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    #  Percentile CI
    # ------------------------------------------------------------------

    def percentile_ci(
        self,
        bootstrap_samples: NDArray[np.floating],
        confidence_level: Optional[float] = None,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Basic percentile confidence interval.

        Parameters
        ----------
        bootstrap_samples : ndarray, shape (B, ...)
            Bootstrap replicates along the first axis.
        confidence_level : float or None
            Override the instance-level confidence level.

        Returns
        -------
        ci_lower, ci_upper : ndarray
            Element-wise lower and upper percentile bounds.
        """
        alpha = 1.0 - (confidence_level or self.confidence_level)
        lower_q = 100.0 * alpha / 2.0
        upper_q = 100.0 * (1.0 - alpha / 2.0)
        ci_lower = np.percentile(bootstrap_samples, lower_q, axis=0)
        ci_upper = np.percentile(bootstrap_samples, upper_q, axis=0)
        return ci_lower, ci_upper

    # ------------------------------------------------------------------
    #  BCa CI
    # ------------------------------------------------------------------

    def bca_ci(
        self,
        bootstrap_samples: NDArray[np.floating],
        jackknife_samples: NDArray[np.floating],
        original_estimate: NDArray[np.floating],
        confidence_level: Optional[float] = None,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Bias-Corrected and Accelerated (BCa) confidence interval.

        Adjusts the percentile interval for bias (z_0) and skewness
        (acceleration *a*) following Efron (1987).

        Parameters
        ----------
        bootstrap_samples : ndarray, shape (B, ...)
            Bootstrap replicates.
        jackknife_samples : ndarray, shape (n, ...)
            Leave-one-out jackknife estimates used to compute the
            acceleration constant.
        original_estimate : ndarray
            Point estimate from the full sample.
        confidence_level : float or None
            Override confidence level.

        Returns
        -------
        ci_lower, ci_upper : ndarray
        """
        alpha = 1.0 - (confidence_level or self.confidence_level)

        z0 = self._compute_bias_correction(bootstrap_samples, original_estimate)
        acc = self._compute_acceleration(jackknife_samples)

        z_alpha_lo = stats.norm.ppf(alpha / 2.0)
        z_alpha_hi = stats.norm.ppf(1.0 - alpha / 2.0)

        # Adjusted quantiles ------------------------------------------
        def _adjusted_quantile(z_alpha: float) -> NDArray[np.floating]:
            numerator = z0 + z_alpha
            denominator = 1.0 - acc * numerator
            # Guard against division by zero / extreme acceleration
            with np.errstate(divide="ignore", invalid="ignore"):
                adjusted = z0 + numerator / denominator
            adjusted = np.where(np.isfinite(adjusted), adjusted, z_alpha)
            return stats.norm.cdf(adjusted)  # type: ignore[return-value]

        q_lo = _adjusted_quantile(z_alpha_lo)
        q_hi = _adjusted_quantile(z_alpha_hi)

        # Clip quantiles to [0, 1] for safety
        q_lo = np.clip(q_lo, 0.0, 1.0)
        q_hi = np.clip(q_hi, 0.0, 1.0)

        # Compute element-wise percentiles -----------------------------
        flat_boot = bootstrap_samples.reshape(bootstrap_samples.shape[0], -1)
        q_lo_flat = np.atleast_1d(q_lo).ravel()
        q_hi_flat = np.atleast_1d(q_hi).ravel()

        ci_lower_flat = np.empty(flat_boot.shape[1])
        ci_upper_flat = np.empty(flat_boot.shape[1])
        for j in range(flat_boot.shape[1]):
            ci_lower_flat[j] = np.percentile(
                flat_boot[:, j], 100.0 * q_lo_flat[min(j, len(q_lo_flat) - 1)]
            )
            ci_upper_flat[j] = np.percentile(
                flat_boot[:, j], 100.0 * q_hi_flat[min(j, len(q_hi_flat) - 1)]
            )

        out_shape = bootstrap_samples.shape[1:]
        ci_lower = ci_lower_flat.reshape(out_shape) if out_shape else ci_lower_flat[0]
        ci_upper = ci_upper_flat.reshape(out_shape) if out_shape else ci_upper_flat[0]
        return np.asarray(ci_lower), np.asarray(ci_upper)

    # ------------------------------------------------------------------
    #  Seed bootstrap
    # ------------------------------------------------------------------

    def bootstrap_over_seeds(
        self,
        ntk_measurements_by_seed: NDArray[np.floating],
        widths: NDArray[np.floating],
        regression_fn: Callable[
            [NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
        ],
    ) -> BootstrapResult:
        """Bootstrap by resampling initialisation seeds.

        Each bootstrap replicate draws *n_seeds* seeds with replacement
        from the original seed axis, averages the NTK measurements, and
        applies ``regression_fn`` to obtain a parameter estimate.

        Parameters
        ----------
        ntk_measurements_by_seed : ndarray, shape (n_seeds, n_widths, ...)
            NTK measurements indexed by (seed, width, ...).
        widths : ndarray, shape (n_widths,)
            Network widths corresponding to axis 1.
        regression_fn : callable
            ``regression_fn(widths, averaged_measurements) -> params``.

        Returns
        -------
        BootstrapResult
        """
        n_seeds = ntk_measurements_by_seed.shape[0]
        original_mean = ntk_measurements_by_seed.mean(axis=0)
        original_estimate = np.asarray(regression_fn(widths, original_mean))

        bootstrap_samples: list[NDArray[np.floating]] = []
        for _ in range(self.n_bootstrap):
            idx = self._rng.integers(0, n_seeds, size=n_seeds)
            resampled = ntk_measurements_by_seed[idx].mean(axis=0)
            bootstrap_samples.append(np.asarray(regression_fn(widths, resampled)))

        boot_arr = np.asarray(bootstrap_samples)

        # Jackknife for BCa
        jack = self._jackknife_samples(
            ntk_measurements_by_seed,
            lambda d: np.asarray(regression_fn(widths, d.mean(axis=0))),
            axis=0,
        )

        ci_lower, ci_upper = self.bca_ci(
            boot_arr, jack, original_estimate, self.confidence_level
        )

        bias = float(np.mean(boot_arr, axis=0).mean() - original_estimate.mean())
        acc = self._compute_acceleration(jack)
        se = np.std(boot_arr, axis=0, ddof=1)
        converged = self.convergence_diagnostic(boot_arr)

        return BootstrapResult(
            point_estimate=original_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            method="seed",
            n_bootstrap=self.n_bootstrap,
            bootstrap_distribution=boot_arr,
            bias=bias,
            acceleration=acc,
            se=se,
            converged=converged,
        )

    # ------------------------------------------------------------------
    #  Block bootstrap
    # ------------------------------------------------------------------

    def block_bootstrap(
        self,
        ntk_measurements: NDArray[np.floating],
        widths: NDArray[np.floating],
        block_size: int,
        regression_fn: Callable[
            [NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
        ],
    ) -> BootstrapResult:
        """Block bootstrap for samples correlated across widths.

        Resamples contiguous blocks of width indices to preserve the
        correlation structure.

        Parameters
        ----------
        ntk_measurements : ndarray, shape (n_samples, n_widths, ...)
            Measurement array; samples along axis 0, widths along axis 1.
        widths : ndarray, shape (n_widths,)
            Corresponding width values.
        block_size : int
            Number of consecutive width indices per block.
        regression_fn : callable
            ``regression_fn(widths_block, measurements_block) -> params``.

        Returns
        -------
        BootstrapResult
        """
        n_samples, n_widths = ntk_measurements.shape[:2]
        original_mean = ntk_measurements.mean(axis=0)
        original_estimate = np.asarray(regression_fn(widths, original_mean))

        n_blocks = max(1, n_widths // block_size)

        bootstrap_samples: list[NDArray[np.floating]] = []
        for _ in range(self.n_bootstrap):
            # Resample samples with replacement
            sample_idx = self._rng.integers(0, n_samples, size=n_samples)
            resampled_data = ntk_measurements[sample_idx]

            # Resample contiguous width blocks
            block_starts = self._rng.integers(
                0, max(1, n_widths - block_size + 1), size=n_blocks
            )
            width_idx = np.concatenate(
                [np.arange(s, min(s + block_size, n_widths)) for s in block_starts]
            )[:n_widths]

            resampled_widths = widths[width_idx]
            resampled_meas = resampled_data[:, width_idx].mean(axis=0)
            bootstrap_samples.append(
                np.asarray(regression_fn(resampled_widths, resampled_meas))
            )

        boot_arr = np.asarray(bootstrap_samples)
        ci_lower, ci_upper = self.percentile_ci(boot_arr, self.confidence_level)

        bias = float(np.mean(boot_arr, axis=0).mean() - original_estimate.mean())
        se = np.std(boot_arr, axis=0, ddof=1)
        converged = self.convergence_diagnostic(boot_arr)

        return BootstrapResult(
            point_estimate=original_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            method="block",
            n_bootstrap=self.n_bootstrap,
            bootstrap_distribution=boot_arr,
            bias=bias,
            acceleration=np.zeros_like(se),
            se=se,
            converged=converged,
        )

    # ------------------------------------------------------------------
    #  BCa helpers
    # ------------------------------------------------------------------

    def _compute_bias_correction(
        self,
        bootstrap_samples: NDArray[np.floating],
        original_estimate: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """BCa bias-correction factor z_0.

        z_0 = Φ^{-1}( #{θ̂*_b < θ̂} / B )

        Parameters
        ----------
        bootstrap_samples : ndarray, shape (B, ...)
        original_estimate : ndarray

        Returns
        -------
        z0 : ndarray, same shape as ``original_estimate``
        """
        prop_less = np.mean(bootstrap_samples < original_estimate, axis=0)
        # Clamp away from 0 and 1 to keep ppf finite
        eps = 1.0 / (2.0 * bootstrap_samples.shape[0])
        prop_less = np.clip(prop_less, eps, 1.0 - eps)
        z0: NDArray[np.floating] = stats.norm.ppf(prop_less)  # type: ignore[assignment]
        return np.asarray(z0)

    def _compute_acceleration(
        self,
        jackknife_samples: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """BCa acceleration factor *a*.

        a = (1/6) Σ (θ̂_{(·)} - θ̂_{(i)})^3
            / [ Σ (θ̂_{(·)} - θ̂_{(i)})^2 ]^{3/2}

        where θ̂_{(·)} is the mean of the jackknife estimates and θ̂_{(i)}
        is the *i*-th leave-one-out estimate.

        Parameters
        ----------
        jackknife_samples : ndarray, shape (n, ...)

        Returns
        -------
        acc : ndarray
        """
        jack_mean = np.mean(jackknife_samples, axis=0)
        diff = jack_mean - jackknife_samples  # (n, ...)

        num = np.sum(diff ** 3, axis=0)
        denom = np.sum(diff ** 2, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            acc = num / (6.0 * denom ** 1.5)
        # Replace non-finite values with 0 (no acceleration correction)
        acc = np.where(np.isfinite(acc), acc, 0.0)
        return np.asarray(acc)

    # ------------------------------------------------------------------
    #  Jackknife
    # ------------------------------------------------------------------

    def _jackknife_samples(
        self,
        data: NDArray[np.floating],
        statistic_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        axis: int = 0,
    ) -> NDArray[np.floating]:
        """Compute leave-one-out jackknife estimates.

        Parameters
        ----------
        data : ndarray
            Input data array.
        statistic_fn : callable
            Function that maps a reduced dataset to a scalar or array
            statistic.
        axis : int
            Axis along which observations are dropped one at a time.

        Returns
        -------
        jackknife : ndarray, shape (n, ...)
            Array of leave-one-out statistics.
        """
        n = data.shape[axis]
        results: list[NDArray[np.floating]] = []
        for i in range(n):
            reduced = np.delete(data, i, axis=axis)
            results.append(np.asarray(statistic_fn(reduced)))
        return np.asarray(results)

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------

    def convergence_diagnostic(
        self,
        bootstrap_samples: NDArray[np.floating],
        n_checkpoints: int = 10,
    ) -> bool:
        """Check bootstrap convergence via CI stability.

        Computes the 95 % percentile CI at ``n_checkpoints`` increasing
        subsample sizes.  Convergence is declared when the relative
        change in CI width between the last two checkpoints is < 5 %.

        Parameters
        ----------
        bootstrap_samples : ndarray, shape (B, ...)
        n_checkpoints : int
            Number of subsample sizes to evaluate.

        Returns
        -------
        converged : bool
        """
        B = bootstrap_samples.shape[0]
        if B < 2 * n_checkpoints:
            return False

        sizes = np.linspace(
            max(50, B // n_checkpoints), B, n_checkpoints, dtype=int
        )
        sizes = np.unique(sizes)
        if len(sizes) < 2:
            return True

        widths_at_checkpoints: list[float] = []
        for s in sizes:
            sub = bootstrap_samples[:s]
            lo, hi = self.percentile_ci(sub)
            ci_width = float(np.mean(hi - lo))
            widths_at_checkpoints.append(ci_width)

        prev, curr = widths_at_checkpoints[-2], widths_at_checkpoints[-1]
        if prev == 0.0:
            return curr == 0.0
        rel_change = abs(curr - prev) / abs(prev)
        return rel_change < 0.05

    def effective_sample_size(
        self,
        bootstrap_samples: NDArray[np.floating],
    ) -> float:
        """Estimate effective sample size from autocorrelation.

        Uses the initial-positive-sequence estimator: ESS = B / τ
        where τ = 1 + 2 Σ_{k=1}^{K} ρ(k) and the sum is truncated at
        the first negative autocorrelation.

        Parameters
        ----------
        bootstrap_samples : ndarray, shape (B, ...)
            If multidimensional, the first component (flattened) is used.

        Returns
        -------
        ess : float
            Estimated effective sample size.
        """
        x = bootstrap_samples.reshape(bootstrap_samples.shape[0], -1)[:, 0]
        B = len(x)
        if B < 4:
            return float(B)

        x_centred = x - np.mean(x)
        var = np.var(x_centred, ddof=0)
        if var == 0.0:
            return float(B)

        # Autocorrelation via FFT
        fft_x = np.fft.rfft(x_centred, n=2 * B)
        acf_full = np.fft.irfft(np.abs(fft_x) ** 2)[:B] / (var * B)

        # Initial positive sequence truncation
        tau = 1.0
        for k in range(1, B):
            rho_k = acf_full[k]
            if rho_k < 0.0:
                break
            tau += 2.0 * rho_k

        ess = B / tau
        return max(1.0, ess)


# ======================================================================
#  Boundary uncertainty propagation
# ======================================================================


class BoundaryUncertainty:
    """Propagate parameter uncertainty to phase boundary location.

    Given a phase boundary defined implicitly by F(γ, Θ^{(1)}) = 0,
    this class provides methods to map uncertainty in the regression
    parameter Θ^{(1)} to uncertainty in the critical width γ*.

    Parameters
    ----------
    confidence_level : float
        Desired coverage probability for boundary bands.
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        self.confidence_level = confidence_level

    # ------------------------------------------------------------------
    #  Direct propagation
    # ------------------------------------------------------------------

    def propagate_to_boundary(
        self,
        theta_1_samples: NDArray[np.floating],
        theta_0: NDArray[np.floating],
        boundary_fn: Callable[
            [NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
        ],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Propagate Θ^{(1)} uncertainty to phase-boundary location.

        Evaluates ``boundary_fn(theta_0, theta_1_sample)`` for each
        bootstrap sample of Θ^{(1)} and returns percentile CIs on γ*.

        Parameters
        ----------
        theta_1_samples : ndarray, shape (B, d)
            Bootstrap samples of the first-order correction parameter.
        theta_0 : ndarray, shape (d0,)
            Fixed infinite-width parameter Θ^{(0)}.
        boundary_fn : callable
            ``boundary_fn(theta_0, theta_1) -> gamma_star`` mapping
            parameters to the critical width (scalar or array).

        Returns
        -------
        gamma_mean : ndarray
            Mean boundary location across bootstrap replicates.
        gamma_lower : ndarray
            Lower confidence bound.
        gamma_upper : ndarray
            Upper confidence bound.
        """
        gamma_samples: list[NDArray[np.floating]] = []
        for sample in theta_1_samples:
            try:
                gamma = np.asarray(boundary_fn(theta_0, sample))
                if np.all(np.isfinite(gamma)):
                    gamma_samples.append(gamma)
            except (ValueError, np.linalg.LinAlgError):
                # Skip samples that lead to degenerate boundaries
                continue

        if len(gamma_samples) == 0:
            warnings.warn(
                "All boundary evaluations failed; returning NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            nan = np.full_like(theta_0, np.nan)
            return nan, nan, nan

        gamma_arr = np.asarray(gamma_samples)
        gamma_mean = np.mean(gamma_arr, axis=0)

        alpha = 1.0 - self.confidence_level
        gamma_lower = np.percentile(gamma_arr, 100.0 * alpha / 2.0, axis=0)
        gamma_upper = np.percentile(gamma_arr, 100.0 * (1.0 - alpha / 2.0), axis=0)
        return gamma_mean, gamma_lower, gamma_upper

    # ------------------------------------------------------------------
    #  Implicit function theorem
    # ------------------------------------------------------------------

    def implicit_function_propagation(
        self,
        theta_0: NDArray[np.floating],
        theta_1_mean: NDArray[np.floating],
        theta_1_cov: NDArray[np.floating],
        boundary_jacobian: Callable[
            [NDArray[np.floating], NDArray[np.floating]],
            Tuple[NDArray[np.floating], NDArray[np.floating]],
        ],
    ) -> Tuple[float, float]:
        """Propagate via implicit function theorem (delta method).

        For F(γ, Θ^{(1)}) = 0 the linearised sensitivity is

            δγ = -(∂F/∂γ)^{-1}  (∂F/∂Θ^{(1)})  δΘ^{(1)}

        so  Var(γ) = J  Cov(Θ^{(1)})  J^T  with
        J = -(∂F/∂γ)^{-1} (∂F/∂Θ^{(1)}).

        Parameters
        ----------
        theta_0 : ndarray
            Infinite-width parameter (context only, passed to Jacobian).
        theta_1_mean : ndarray, shape (d,)
            Point estimate of Θ^{(1)}.
        theta_1_cov : ndarray, shape (d, d)
            Covariance matrix of Θ^{(1)}.
        boundary_jacobian : callable
            ``boundary_jacobian(theta_0, theta_1_mean)`` returning
            ``(dF_dgamma, dF_dtheta1)`` where ``dF_dgamma`` is a scalar
            and ``dF_dtheta1`` has shape ``(d,)``.

        Returns
        -------
        gamma_var : float
            Variance of γ* induced by parameter uncertainty.
        gamma_se : float
            Standard error of γ*.
        """
        dF_dgamma, dF_dtheta1 = boundary_jacobian(theta_0, theta_1_mean)

        dF_dgamma = np.atleast_1d(np.asarray(dF_dgamma, dtype=float))
        dF_dtheta1 = np.atleast_1d(np.asarray(dF_dtheta1, dtype=float))

        if np.any(np.abs(dF_dgamma) < 1e-14):
            warnings.warn(
                "∂F/∂γ ≈ 0: boundary is insensitive to γ; "
                "returning infinite variance.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float("inf"), float("inf")

        # J = -(∂F/∂γ)^{-1} * (∂F/∂Θ^{(1)})  — row vector (1, d)
        J = -(dF_dtheta1 / dF_dgamma).reshape(1, -1)

        theta_1_cov = np.atleast_2d(theta_1_cov)
        gamma_var = float(J @ theta_1_cov @ J.T)
        gamma_se = float(np.sqrt(max(0.0, gamma_var)))
        return gamma_var, gamma_se

    # ------------------------------------------------------------------
    #  Confidence band
    # ------------------------------------------------------------------

    def boundary_confidence_band(
        self,
        gamma_values: NDArray[np.floating],
        boundary_values: NDArray[np.floating],
        uncertainties: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Compute point-wise confidence band for the boundary curve.

        Parameters
        ----------
        gamma_values : ndarray, shape (m,)
            Grid of γ values at which the boundary is evaluated.
        boundary_values : ndarray, shape (m,)
            Boundary function evaluated at each γ.
        uncertainties : ndarray, shape (m,)
            Standard error at each γ (e.g. from ``implicit_function_propagation``
            or bootstrap SE).

        Returns
        -------
        lower_band, upper_band : ndarray, shape (m,)
        """
        z = stats.norm.ppf(1.0 - (1.0 - self.confidence_level) / 2.0)
        lower_band = boundary_values - z * uncertainties
        upper_band = boundary_values + z * uncertainties
        return np.asarray(lower_band), np.asarray(upper_band)

    # ------------------------------------------------------------------
    #  Monte Carlo boundary sampling
    # ------------------------------------------------------------------

    def monte_carlo_boundary(
        self,
        theta_1_mean: NDArray[np.floating],
        theta_1_cov: NDArray[np.floating],
        n_samples: int,
        boundary_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Monte Carlo sampling of boundary uncertainty.

        Draws Θ^{(1)} ~ N(mean, cov) and evaluates ``boundary_fn`` for
        each draw to build an empirical distribution of γ*.

        Parameters
        ----------
        theta_1_mean : ndarray, shape (d,)
            Mean of Θ^{(1)}.
        theta_1_cov : ndarray, shape (d, d)
            Covariance of Θ^{(1)}.
        n_samples : int
            Number of Monte Carlo draws.
        boundary_fn : callable
            ``boundary_fn(theta_1) -> gamma_star``.
        random_state : int | Generator | None
            Seed or Generator.

        Returns
        -------
        gamma_mean : ndarray
            Mean boundary over MC samples.
        gamma_lower : ndarray
            Lower confidence bound.
        gamma_upper : ndarray
            Upper confidence bound.
        """
        rng = np.random.default_rng(random_state)

        theta_1_mean = np.atleast_1d(theta_1_mean).astype(float)
        theta_1_cov = np.atleast_2d(theta_1_cov).astype(float)

        # Stabilise covariance for Cholesky
        eigvals = np.linalg.eigvalsh(theta_1_cov)
        if np.any(eigvals < 0):
            min_eig = np.min(eigvals)
            theta_1_cov = theta_1_cov + (-min_eig + 1e-10) * np.eye(len(theta_1_mean))

        draws = rng.multivariate_normal(theta_1_mean, theta_1_cov, size=n_samples)

        gamma_samples: list[NDArray[np.floating]] = []
        for draw in draws:
            try:
                gamma = np.asarray(boundary_fn(draw))
                if np.all(np.isfinite(gamma)):
                    gamma_samples.append(gamma)
            except (ValueError, np.linalg.LinAlgError):
                continue

        if len(gamma_samples) == 0:
            warnings.warn(
                "All Monte Carlo boundary evaluations failed; returning NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            nan = np.array([np.nan])
            return nan, nan, nan

        gamma_arr = np.asarray(gamma_samples)
        gamma_mean = np.mean(gamma_arr, axis=0)

        alpha = 1.0 - self.confidence_level
        gamma_lower = np.percentile(gamma_arr, 100.0 * alpha / 2.0, axis=0)
        gamma_upper = np.percentile(gamma_arr, 100.0 * (1.0 - alpha / 2.0), axis=0)
        return gamma_mean, gamma_lower, gamma_upper
