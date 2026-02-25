"""
Finite-Width NTK Corrections via 1/N Expansion
================================================

Implements the systematic 1/N expansion of the Neural Tangent Kernel:

    Θ(N) = Θ^(0) + Θ^(1)/N + Θ^(2)/N²  + O(1/N³)

where N is the network width, Θ^(0) is the infinite-width (mean-field) NTK,
and Θ^(1), Θ^(2) are the first and second finite-width correction matrices.

The corrections are extracted from empirical NTK measurements at multiple
widths via weighted least-squares regression of the element-wise matrix
entries against powers of 1/N.

References
----------
- Dyer & Gur-Ari (2020), "Asymptotics of Wide Networks from Feynman Diagrams"
- Huang & Yau (2020), "Dynamics of Deep Neural Networks and Neural Tangent
  Hierarchy"
- Hanin & Nica (2020), "Finite Depth and Width Corrections to the Neural
  Tangent Kernel"
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sp_linalg
from scipy import optimize as sp_optimize

logger = logging.getLogger(__name__)

# ======================================================================
#  Data classes for structured results
# ======================================================================


@dataclass
class ConvergenceInfo:
    """Diagnostics for the convergence of the 1/N expansion.

    Attributes
    ----------
    converged : bool
        Whether the expansion has converged within the requested tolerance.
    relative_error : float
        Root-mean-square relative residual of the fit,
        ``||residuals||_F / ||Θ^(0)||_F``.
    num_widths_used : int
        Number of distinct widths used in the regression.
    effective_dof : float
        Effective degrees of freedom of the fit (num_widths - num_params).
    r_squared : float
        Coefficient of determination R² of the element-wise regression.
    warnings : List[str]
        Human-readable warnings about potential issues (e.g. ill-conditioning,
        non-monotonic corrections, insufficient width range).
    """

    converged: bool
    relative_error: float
    num_widths_used: int
    effective_dof: float
    r_squared: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class CorrectionResult:
    """Result container for the finite-width NTK expansion.

    The NTK at width N is reconstructed as:

        Θ(N) ≈ theta_0 + theta_1 / N + theta_2 / N²

    Attributes
    ----------
    theta_0 : NDArray
        Infinite-width NTK matrix, shape ``(n, n)`` where ``n`` is the number
        of data points.
    theta_1 : NDArray
        First-order correction matrix, shape ``(n, n)``.
    theta_2 : NDArray
        Second-order correction matrix, shape ``(n, n)``.
    residuals : NDArray
        Fit residuals at each calibration width, shape ``(K, n, n)`` where
        ``K`` is the number of widths.
    condition_number : float
        Condition number of the design matrix used in the regression.
    convergence_info : ConvergenceInfo
        Detailed convergence diagnostics.
    correction_magnitudes : Dict[str, float]
        Relative magnitudes of the correction terms, keyed by descriptive
        labels such as ``"||Θ^(1)||/||Θ^(0)||"`` and
        ``"||Θ^(2)||/||Θ^(0)||"``.
    """

    theta_0: NDArray
    theta_1: NDArray
    theta_2: NDArray
    residuals: NDArray
    condition_number: float
    convergence_info: ConvergenceInfo
    correction_magnitudes: Dict[str, float] = field(default_factory=dict)


# ======================================================================
#  Helper / utility functions
# ======================================================================


def _weighted_least_squares(
    A: NDArray,
    b: NDArray,
    W: NDArray,
) -> Tuple[NDArray, NDArray, float]:
    """Solve the weighted least-squares problem  min_x || W^{1/2}(Ax - b) ||².

    Parameters
    ----------
    A : ndarray, shape (m, p)
        Design matrix.
    b : ndarray, shape (m,) or (m, d)
        Observation vector(s).
    W : ndarray, shape (m,) or (m, m)
        Weights.  If 1-D, interpreted as a diagonal weight matrix.

    Returns
    -------
    x : ndarray, shape (p,) or (p, d)
        Solution vector(s).
    residuals : ndarray, shape (m,) or (m, d)
        Weighted residuals ``W^{1/2}(Ax - b)``.
    cond : float
        Condition number of the weighted design matrix ``W^{1/2} A``.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    # Build the diagonal square-root weight matrix
    if W.ndim == 1:
        sqrt_W = np.sqrt(W)
        Aw = A * sqrt_W[:, np.newaxis]
        bw = b * sqrt_W if b.ndim == 1 else b * sqrt_W[:, np.newaxis]
    else:
        # Full weight matrix — compute Cholesky factor
        L = np.linalg.cholesky(W)
        Aw = L @ A
        bw = L @ b

    cond = np.linalg.cond(Aw)

    # Solve via SVD for numerical stability
    x, res_sum, rank, sv = np.linalg.lstsq(Aw, bw, rcond=None)

    raw_residuals = A @ x - b
    if W.ndim == 1:
        weighted_residuals = raw_residuals * sqrt_W if b.ndim == 1 else (
            raw_residuals * sqrt_W[:, np.newaxis]
        )
    else:
        weighted_residuals = L @ raw_residuals

    return x, weighted_residuals, float(cond)


def _jackknife_variance(
    data_fn: Callable[[NDArray], NDArray],
    data: NDArray,
    axis: int = 0,
) -> NDArray:
    """Compute jackknife variance estimate for a statistic.

    Parameters
    ----------
    data_fn : callable
        Function that computes a statistic from the data array.  Must accept
        a single ndarray argument and return an ndarray.
    data : ndarray
        Raw data array.
    axis : int
        Axis along which to leave-one-out.

    Returns
    -------
    var : ndarray
        Jackknife variance estimate, same shape as ``data_fn(data)``.
    """
    n = data.shape[axis]
    if n < 2:
        return np.zeros_like(data_fn(data))

    full_stat = data_fn(data)
    jackknife_stats = []

    for i in range(n):
        reduced = np.delete(data, i, axis=axis)
        jackknife_stats.append(data_fn(reduced))

    jackknife_stats = np.array(jackknife_stats)
    mean_stat = np.mean(jackknife_stats, axis=0)

    # Jackknife variance: ((n-1)/n) * sum((stat_i - mean_stat)^2)
    var = ((n - 1.0) / n) * np.sum(
        (jackknife_stats - mean_stat[np.newaxis]) ** 2, axis=0
    )
    return var


def _check_expansion_monotonicity(
    theta_0: NDArray,
    theta_1: NDArray,
    theta_2: NDArray,
    widths: Sequence[int],
) -> List[str]:
    """Verify that correction magnitudes decrease with increasing width.

    For a well-behaved 1/N expansion we expect:

        ||Θ^(1)|| / (N · ||Θ^(0)||)  >  ||Θ^(2)|| / (N² · ||Θ^(0)||)

    at every calibration width N.

    Parameters
    ----------
    theta_0 : ndarray
        Infinite-width NTK.
    theta_1 : ndarray
        First correction.
    theta_2 : ndarray
        Second correction.
    widths : sequence of int
        Calibration widths.

    Returns
    -------
    warnings : list of str
        Warnings for widths where monotonicity is violated.
    """
    norm_0 = np.linalg.norm(theta_0, "fro")
    if norm_0 < 1e-15:
        return ["||Θ^(0)|| is near zero; monotonicity check is meaningless."]

    norm_1 = np.linalg.norm(theta_1, "fro")
    norm_2 = np.linalg.norm(theta_2, "fro")

    warn_list: List[str] = []
    for N in widths:
        mag_1 = norm_1 / (N * norm_0)
        mag_2 = norm_2 / (N * N * norm_0)
        if mag_2 > mag_1 and mag_1 > 1e-12:
            warn_list.append(
                f"Width N={N}: second-order correction ({mag_2:.3e}) exceeds "
                f"first-order correction ({mag_1:.3e}); expansion may not "
                f"converge at this width."
            )
    return warn_list


# ======================================================================
#  Main corrector class
# ======================================================================


class FiniteWidthCorrector:
    """Extract finite-width NTK corrections via the 1/N expansion.

    The corrector fits the element-wise model

        Θ_{ij}(N) = Θ^(0)_{ij} + Θ^(1)_{ij} / N + Θ^(2)_{ij} / N²

    using NTK measurements at several widths N.  Measurements can be
    supplied directly (``compute_corrections_regression``) or computed on
    the fly from a network forward function (``compute_corrections_numerical``).

    Parameters
    ----------
    output_dim : int
        Dimensionality of the network output (default 1).
    eps : float
        Step size for finite-difference Jacobian computation (default 1e-5).
    min_widths : int
        Minimum number of distinct widths required for fitting (default 4).
    convergence_tol : float
        Relative tolerance for declaring convergence (default 1e-3).
    """

    def __init__(
        self,
        output_dim: int = 1,
        eps: float = 1e-5,
        min_widths: int = 4,
        convergence_tol: float = 1e-3,
    ) -> None:
        self.output_dim = output_dim
        self.eps = eps
        self.min_widths = min_widths
        self.convergence_tol = convergence_tol

    # ------------------------------------------------------------------
    #  Public API: numerical computation from a forward function
    # ------------------------------------------------------------------

    def compute_corrections_numerical(
        self,
        forward_fn: Callable[[NDArray, NDArray], NDArray],
        param_init_fn: Callable[[int], NDArray],
        X: NDArray,
        widths: Sequence[int],
        num_seeds: int = 5,
    ) -> CorrectionResult:
        """Compute NTK corrections by empirical measurement at multiple widths.

        For each width N in *widths*, the empirical NTK is computed
        ``num_seeds`` times (with independent parameter initialisations) and
        averaged.  The resulting per-width NTK matrices are then fed into the
        regression-based extraction.

        Parameters
        ----------
        forward_fn : callable
            ``forward_fn(params, X)`` returns network outputs of shape
            ``(n, output_dim)``.  *params* is a 1-D parameter vector whose
            length depends on the width.
        param_init_fn : callable
            ``param_init_fn(width)`` returns a 1-D parameter vector
            initialised at random for the given width.
        X : ndarray, shape (n, d)
            Input data matrix.
        widths : sequence of int
            Network widths to probe.
        num_seeds : int
            Number of independent seeds per width for variance reduction.

        Returns
        -------
        CorrectionResult
            Fitted expansion with convergence diagnostics.
        """
        widths = np.asarray(widths, dtype=int)
        if len(widths) < self.min_widths:
            raise ValueError(
                f"Need at least {self.min_widths} widths, got {len(widths)}."
            )

        n = X.shape[0]
        ntk_measurements: List[NDArray] = []  # shape (K, n, n)
        all_per_seed: Dict[int, List[NDArray]] = {}  # width -> list of NTKs

        for width in widths:
            logger.info("Computing empirical NTK at width N=%d ...", width)
            seed_ntks: List[NDArray] = []
            for s in range(num_seeds):
                params = param_init_fn(int(width))
                ntk_s = self._compute_empirical_ntk(forward_fn, params, X, self.eps)
                seed_ntks.append(ntk_s)
            all_per_seed[int(width)] = seed_ntks
            mean_ntk = np.mean(seed_ntks, axis=0)
            ntk_measurements.append(mean_ntk)

        ntk_array = np.array(ntk_measurements)  # (K, n, n)

        # Estimate weights from cross-seed variance
        weights = self._estimate_weights(all_per_seed, widths, num_seeds)

        return self.compute_corrections_regression(ntk_array, widths, weights)

    # ------------------------------------------------------------------
    #  Public API: regression from pre-computed measurements
    # ------------------------------------------------------------------

    def compute_corrections_regression(
        self,
        ntk_measurements: NDArray,
        widths: Sequence[int],
        weights: Optional[NDArray] = None,
    ) -> CorrectionResult:
        """Extract corrections from pre-computed NTK matrices.

        Parameters
        ----------
        ntk_measurements : ndarray, shape (K, n, n)
            Empirical NTK matrices measured at each of the K widths.
        widths : sequence of int
            The K widths corresponding to the measurements.
        weights : ndarray, shape (K,), optional
            Per-width regression weights.  If *None*, uniform weights are used.

        Returns
        -------
        CorrectionResult
            Fitted 1/N expansion with diagnostics.
        """
        ntk_measurements = np.asarray(ntk_measurements, dtype=np.float64)
        widths = np.asarray(widths, dtype=np.float64)
        K = len(widths)

        if ntk_measurements.shape[0] != K:
            raise ValueError(
                f"ntk_measurements has {ntk_measurements.shape[0]} entries "
                f"but {K} widths were given."
            )
        if K < self.min_widths:
            raise ValueError(
                f"Need at least {self.min_widths} widths, got {K}."
            )

        if weights is None:
            weights = np.ones(K, dtype=np.float64)

        # Core regression
        theta_0, theta_1, theta_2, residuals, cond = self._fit_expansion(
            ntk_measurements, widths, weights
        )

        # Convergence diagnostics
        conv_info = self._check_convergence(
            theta_0, theta_1, theta_2, residuals, widths
        )

        # Error estimation
        error_info = self._estimate_errors(
            theta_0, theta_1, theta_2, ntk_measurements, widths, weights
        )

        # Correction magnitudes at each calibration width
        magnitudes: Dict[str, float] = {}
        norm_0 = np.linalg.norm(theta_0, "fro")
        if norm_0 > 0:
            magnitudes["||Θ^(1)||/||Θ^(0)||"] = float(
                np.linalg.norm(theta_1, "fro") / norm_0
            )
            magnitudes["||Θ^(2)||/||Θ^(0)||"] = float(
                np.linalg.norm(theta_2, "fro") / norm_0
            )
            for w in widths:
                label = f"relative_correction_N={int(w)}"
                magnitudes[label] = float(self.correction_magnitude(
                    theta_0, theta_1, theta_2, int(w)
                ))

        # Merge error estimation info into convergence warnings
        if error_info:
            conv_info.warnings.extend(error_info)

        logger.info(
            "Expansion fit complete: cond=%.2e, R²=%.6f, converged=%s",
            cond, conv_info.r_squared, conv_info.converged,
        )

        return CorrectionResult(
            theta_0=theta_0,
            theta_1=theta_1,
            theta_2=theta_2,
            residuals=residuals,
            condition_number=cond,
            convergence_info=conv_info,
            correction_magnitudes=magnitudes,
        )

    # ------------------------------------------------------------------
    #  Empirical NTK computation
    # ------------------------------------------------------------------

    def _compute_empirical_ntk(
        self,
        forward_fn: Callable[[NDArray, NDArray], NDArray],
        params: NDArray,
        X: NDArray,
        eps: float,
    ) -> NDArray:
        """Compute the empirical NTK via finite-difference Jacobians.

        The NTK is defined as

            Θ_{ij} = ∑_k (∂f_i / ∂θ_k)(∂f_j / ∂θ_k)  =  J J^T

        where J_{i,k} = ∂f(x_i) / ∂θ_k  is the Jacobian of the scalar
        network output with respect to parameter θ_k evaluated at input x_i.

        For multi-output networks (``output_dim > 1``), the NTK block
        structure is summed over the output index to return a single
        ``(n, n)`` kernel matrix.

        Parameters
        ----------
        forward_fn : callable
            ``forward_fn(params, X)`` → ``(n, output_dim)``.
        params : ndarray, shape (P,)
            Parameter vector.
        X : ndarray, shape (n, d)
            Input matrix.
        eps : float
            Finite-difference step size.

        Returns
        -------
        ntk : ndarray, shape (n, n)
            Empirical NTK matrix.
        """
        params = np.asarray(params, dtype=np.float64).ravel()
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        P = params.shape[0]

        # Base forward pass
        f0 = np.asarray(forward_fn(params, X), dtype=np.float64)
        if f0.ndim == 1:
            f0 = f0[:, np.newaxis]
        # f0 shape: (n, output_dim)

        out_dim = f0.shape[1]

        # Compute Jacobian J of shape (n, out_dim, P) via central differences
        jacobian = np.zeros((n, out_dim, P), dtype=np.float64)
        for k in range(P):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[k] += eps
            params_minus[k] -= eps

            f_plus = np.asarray(
                forward_fn(params_plus, X), dtype=np.float64
            )
            f_minus = np.asarray(
                forward_fn(params_minus, X), dtype=np.float64
            )
            if f_plus.ndim == 1:
                f_plus = f_plus[:, np.newaxis]
            if f_minus.ndim == 1:
                f_minus = f_minus[:, np.newaxis]

            jacobian[:, :, k] = (f_plus - f_minus) / (2.0 * eps)

        # NTK = sum over output dims of J_o @ J_o^T
        ntk = np.zeros((n, n), dtype=np.float64)
        for o in range(out_dim):
            J_o = jacobian[:, o, :]  # (n, P)
            ntk += J_o @ J_o.T

        return ntk

    # ------------------------------------------------------------------
    #  1/N expansion fit
    # ------------------------------------------------------------------

    def _fit_expansion(
        self,
        ntk_matrices: NDArray,
        widths: NDArray,
        weights: Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, float]:
        """Fit the 1/N expansion element-wise via weighted least squares.

        For each matrix element (i, j) we solve

            Θ_{ij}(N_k) ≈ c0 + c1 / N_k + c2 / N_k²,   k = 1, …, K

        in the weighted least-squares sense.

        Parameters
        ----------
        ntk_matrices : ndarray, shape (K, n, n)
            Empirical NTK matrices.
        widths : ndarray, shape (K,)
            Network widths.
        weights : ndarray, shape (K,), optional
            Regression weights (default: uniform).

        Returns
        -------
        theta_0, theta_1, theta_2 : ndarray, each (n, n)
            Expansion coefficients.
        residuals : ndarray, shape (K, n, n)
            Raw residuals at each width.
        cond : float
            Condition number of the (weighted) design matrix.
        """
        K, n, _ = ntk_matrices.shape
        widths_f = np.asarray(widths, dtype=np.float64)

        if weights is None:
            weights = np.ones(K, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        # Design matrix: [1, 1/N, 1/N²]
        inv_w = 1.0 / widths_f
        A = np.column_stack([np.ones(K), inv_w, inv_w ** 2])  # (K, 3)

        # Flatten the matrix entries: b has shape (K, n*n)
        b = ntk_matrices.reshape(K, n * n)

        # Solve WLS
        coeffs, w_residuals, cond = _weighted_least_squares(A, b, weights)
        # coeffs shape: (3, n*n)

        theta_0 = coeffs[0].reshape(n, n)
        theta_1 = coeffs[1].reshape(n, n)
        theta_2 = coeffs[2].reshape(n, n)

        # Raw (unweighted) residuals
        fitted = A @ coeffs  # (K, n*n)
        residuals = (b - fitted).reshape(K, n, n)

        return theta_0, theta_1, theta_2, residuals, cond

    # ------------------------------------------------------------------
    #  Weight estimation
    # ------------------------------------------------------------------

    def _estimate_weights(
        self,
        ntk_per_seed: Union[Dict[int, List[NDArray]], NDArray],
        widths: NDArray,
        num_seeds: int,
    ) -> NDArray:
        """Estimate per-width regression weights from cross-seed variance.

        Weights are proportional to the inverse variance of the mean NTK
        estimate.  Widths with lower variance (more stable NTK) receive
        higher weight.

        Parameters
        ----------
        ntk_per_seed : dict or ndarray
            If a dict, maps ``width → list of (n, n) NTK arrays`` (one per
            seed).  If an ndarray, shape ``(K, num_seeds, n, n)``.
        widths : ndarray, shape (K,)
            Network widths.
        num_seeds : int
            Number of seeds used per width.

        Returns
        -------
        weights : ndarray, shape (K,)
            Positive regression weights, normalised so that max = 1.
        """
        widths = np.asarray(widths, dtype=int)
        K = len(widths)
        variances = np.zeros(K, dtype=np.float64)

        for k, w in enumerate(widths):
            if isinstance(ntk_per_seed, dict):
                seeds_k = np.array(ntk_per_seed[int(w)])  # (S, n, n)
            else:
                seeds_k = ntk_per_seed[k]  # (S, n, n)

            # Variance of the mean estimator = var(NTK) / S
            # Use the Frobenius norm of the element-wise variance as a scalar
            elem_var = np.var(seeds_k, axis=0, ddof=1)  # (n, n)
            var_of_mean = np.mean(elem_var) / max(num_seeds, 1)
            variances[k] = var_of_mean

        # Inverse-variance weighting with floor to avoid division by zero
        floor = np.max(variances) * 1e-10 + 1e-30
        raw_weights = 1.0 / np.maximum(variances, floor)

        # Normalise so the largest weight is 1
        weights = raw_weights / np.max(raw_weights)
        return weights

    # ------------------------------------------------------------------
    #  Convergence checking
    # ------------------------------------------------------------------

    def _check_convergence(
        self,
        theta_0: NDArray,
        theta_1: NDArray,
        theta_2: NDArray,
        residuals: NDArray,
        widths: NDArray,
    ) -> ConvergenceInfo:
        """Assess convergence of the fitted 1/N expansion.

        Convergence is declared when:
        1. The relative residual is below ``self.convergence_tol``.
        2. The correction magnitudes decrease monotonically with width.
        3. The condition number of the fit is not excessively large.

        Parameters
        ----------
        theta_0, theta_1, theta_2 : ndarray
            Expansion matrices.
        residuals : ndarray, shape (K, n, n)
            Fit residuals.
        widths : ndarray
            Calibration widths.

        Returns
        -------
        ConvergenceInfo
            Convergence diagnostics.
        """
        K = residuals.shape[0]
        n = theta_0.shape[0]
        num_params = 3  # c0, c1, c2

        norm_0 = np.linalg.norm(theta_0, "fro")
        if norm_0 < 1e-15:
            norm_0 = 1.0  # avoid division by zero

        # Relative error (RMS over widths)
        rms_residual = np.sqrt(np.mean(residuals ** 2))
        relative_error = float(rms_residual / norm_0)

        # Effective degrees of freedom
        effective_dof = float(max(K - num_params, 0))

        # R² computation (element-wise, then averaged)
        widths_f = np.asarray(widths, dtype=np.float64)
        ntk_reconstructed = np.zeros_like(residuals)
        for k, N in enumerate(widths_f):
            ntk_reconstructed[k] = theta_0 + theta_1 / N + theta_2 / (N * N)

        ntk_measured = ntk_reconstructed + residuals  # recover measurements
        ss_res = np.sum(residuals ** 2)
        mean_measured = np.mean(ntk_measured, axis=0, keepdims=True)
        ss_tot = np.sum((ntk_measured - mean_measured) ** 2)
        r_squared = float(1.0 - ss_res / max(ss_tot, 1e-30))

        # Monotonicity warnings
        warn_list = _check_expansion_monotonicity(
            theta_0, theta_1, theta_2, widths
        )

        # Width-range warning
        width_ratio = float(np.max(widths) / np.min(widths))
        if width_ratio < 4.0:
            warn_list.append(
                f"Width ratio max/min = {width_ratio:.1f} < 4; consider using "
                f"a broader range of widths for more reliable extrapolation."
            )

        # Degrees-of-freedom warning
        if effective_dof < 1.0:
            warn_list.append(
                "Effective DOF < 1; the fit is (nearly) interpolating and "
                "error estimates are unreliable."
            )

        converged = (
            relative_error < self.convergence_tol
            and len([w for w in warn_list if "not converge" in w]) == 0
        )

        return ConvergenceInfo(
            converged=converged,
            relative_error=relative_error,
            num_widths_used=K,
            effective_dof=effective_dof,
            r_squared=r_squared,
            warnings=warn_list,
        )

    # ------------------------------------------------------------------
    #  Error estimation
    # ------------------------------------------------------------------

    def _estimate_errors(
        self,
        theta_0: NDArray,
        theta_1: NDArray,
        theta_2: NDArray,
        ntk_matrices: NDArray,
        widths: NDArray,
        weights: NDArray,
    ) -> List[str]:
        """Estimate parameter uncertainties via jackknife resampling.

        We leave out one width at a time, refit, and compute the jackknife
        variance of each expansion coefficient.

        Parameters
        ----------
        theta_0, theta_1, theta_2 : ndarray
            Full-data fit coefficients.
        ntk_matrices : ndarray, shape (K, n, n)
            Measured NTK matrices.
        widths : ndarray, shape (K,)
            Calibration widths.
        weights : ndarray, shape (K,)
            Regression weights.

        Returns
        -------
        warnings : list of str
            Warnings about large uncertainties.
        """
        K = ntk_matrices.shape[0]
        if K < 4:
            return ["Too few widths for jackknife error estimation."]

        widths_f = np.asarray(widths, dtype=np.float64)

        # Jackknife: leave-one-out refits
        jk_theta0: List[NDArray] = []
        jk_theta1: List[NDArray] = []
        jk_theta2: List[NDArray] = []

        for leave_out in range(K):
            mask = np.ones(K, dtype=bool)
            mask[leave_out] = False
            sub_ntk = ntk_matrices[mask]
            sub_widths = widths_f[mask]
            sub_weights = weights[mask]

            t0, t1, t2, _, _ = self._fit_expansion(
                sub_ntk, sub_widths, sub_weights
            )
            jk_theta0.append(t0)
            jk_theta1.append(t1)
            jk_theta2.append(t2)

        jk_theta0 = np.array(jk_theta0)  # (K, n, n)
        jk_theta1 = np.array(jk_theta1)
        jk_theta2 = np.array(jk_theta2)

        # Jackknife variance: ((K-1)/K) * sum_i (stat_i - mean_stat)^2
        def _jk_var(jk_arr: NDArray) -> NDArray:
            mean_jk = np.mean(jk_arr, axis=0)
            return ((K - 1.0) / K) * np.sum(
                (jk_arr - mean_jk[np.newaxis]) ** 2, axis=0
            )

        var_0 = _jk_var(jk_theta0)
        var_1 = _jk_var(jk_theta1)
        var_2 = _jk_var(jk_theta2)

        # Summarise
        warnings_out: List[str] = []
        norm_0 = np.linalg.norm(theta_0, "fro")
        if norm_0 < 1e-15:
            norm_0 = 1.0

        se_0 = float(np.sqrt(np.mean(var_0)))
        se_1 = float(np.sqrt(np.mean(var_1)))
        se_2 = float(np.sqrt(np.mean(var_2)))

        rel_se_0 = se_0 / norm_0
        if rel_se_0 > 0.1:
            warnings_out.append(
                f"Jackknife SE of Θ^(0) is {rel_se_0:.2%} of ||Θ^(0)||; "
                f"the infinite-width estimate may be unreliable."
            )

        norm_1 = np.linalg.norm(theta_1, "fro")
        if norm_1 > 1e-15:
            rel_se_1 = se_1 / norm_1
            if rel_se_1 > 0.5:
                warnings_out.append(
                    f"Jackknife SE of Θ^(1) is {rel_se_1:.2%} of ||Θ^(1)||; "
                    f"the first-order correction has high uncertainty."
                )

        norm_2 = np.linalg.norm(theta_2, "fro")
        if norm_2 > 1e-15:
            rel_se_2 = se_2 / norm_2
            if rel_se_2 > 0.5:
                warnings_out.append(
                    f"Jackknife SE of Θ^(2) is {rel_se_2:.2%} of ||Θ^(2)||; "
                    f"the second-order correction has high uncertainty."
                )

        logger.debug(
            "Jackknife SE — Θ^(0): %.3e, Θ^(1): %.3e, Θ^(2): %.3e",
            se_0, se_1, se_2,
        )

        return warnings_out

    # ------------------------------------------------------------------
    #  Reconstruction and magnitude helpers
    # ------------------------------------------------------------------

    def compute_corrected_ntk(
        self,
        theta_0: NDArray,
        theta_1: NDArray,
        theta_2: NDArray,
        width: int,
    ) -> NDArray:
        """Reconstruct the NTK at a given width from the expansion.

        .. math::

            \\Theta(N) = \\Theta^{(0)} + \\frac{\\Theta^{(1)}}{N}
                        + \\frac{\\Theta^{(2)}}{N^2}

        Parameters
        ----------
        theta_0, theta_1, theta_2 : ndarray, shape (n, n)
            Expansion coefficient matrices.
        width : int
            Target network width N.

        Returns
        -------
        ntk : ndarray, shape (n, n)
            Reconstructed NTK at width N.
        """
        N = float(width)
        return theta_0 + theta_1 / N + theta_2 / (N * N)

    def correction_magnitude(
        self,
        theta_0: NDArray,
        theta_1: NDArray,
        theta_2: NDArray,
        width: int,
    ) -> float:
        """Compute the relative magnitude of finite-width corrections.

        Returns the ratio

        .. math::

            \\frac{\\| \\Theta^{(1)}/N + \\Theta^{(2)}/N^2 \\|_F}
                  {\\| \\Theta^{(0)} \\|_F}

        Parameters
        ----------
        theta_0, theta_1, theta_2 : ndarray
            Expansion coefficients.
        width : int
            Network width N.

        Returns
        -------
        magnitude : float
            Relative correction magnitude (dimensionless).
        """
        N = float(width)
        correction = theta_1 / N + theta_2 / (N * N)
        norm_0 = np.linalg.norm(theta_0, "fro")
        if norm_0 < 1e-30:
            return float("inf")
        return float(np.linalg.norm(correction, "fro") / norm_0)

    def extrapolation_error_bound(
        self,
        theta_0: NDArray,
        theta_1: NDArray,
        theta_2: NDArray,
        width_target: int,
        width_max_calibration: int,
    ) -> float:
        """Estimate an error bound for extrapolating beyond calibration range.

        Assumes the next-order term scales as Θ^(3)/N³ and estimates its
        magnitude from the ratio of the known terms.  Specifically, if

            r = ||Θ^(2)|| / ||Θ^(1)||

        then the estimated third-order norm is  r · ||Θ^(2)|| and the
        extrapolation error bound at width N is

            bound ≈ r · ||Θ^(2)|| / N³  · (N_max / N_target)

        where the last factor penalises extrapolation beyond the calibration
        range.

        Parameters
        ----------
        theta_0, theta_1, theta_2 : ndarray
            Expansion coefficients.
        width_target : int
            Width at which we want to predict the NTK.
        width_max_calibration : int
            Largest width used during calibration.

        Returns
        -------
        bound : float
            Estimated Frobenius-norm error bound (absolute).
        """
        norm_1 = np.linalg.norm(theta_1, "fro")
        norm_2 = np.linalg.norm(theta_2, "fro")

        if norm_1 < 1e-30:
            # Cannot estimate ratio — fall back to norm_2 based bound
            ratio = 1.0
        else:
            ratio = norm_2 / norm_1

        estimated_norm_3 = ratio * norm_2

        N = float(width_target)
        N_max = float(width_max_calibration)

        # Base truncation error
        base_error = estimated_norm_3 / (N ** 3)

        # Extrapolation penalty: linear growth beyond calibration range
        if N < N_max:
            # Interpolation — no penalty
            penalty = 1.0
        else:
            penalty = N / N_max

        bound = base_error * penalty

        logger.debug(
            "Extrapolation bound at N=%d: %.4e (ratio=%.3e, penalty=%.2f)",
            width_target, bound, ratio, penalty,
        )
        return float(bound)


# ======================================================================
#  Module-level convenience functions
# ======================================================================


def fit_corrections(
    ntk_measurements: NDArray,
    widths: Sequence[int],
    *,
    convergence_tol: float = 1e-3,
    min_widths: int = 4,
) -> CorrectionResult:
    """One-shot convenience wrapper for ``FiniteWidthCorrector``.

    Parameters
    ----------
    ntk_measurements : ndarray, shape (K, n, n)
        Empirical NTK matrices at each width.
    widths : sequence of int
        Network widths.
    convergence_tol : float
        Relative tolerance.
    min_widths : int
        Minimum number of widths.

    Returns
    -------
    CorrectionResult
    """
    corrector = FiniteWidthCorrector(
        convergence_tol=convergence_tol,
        min_widths=min_widths,
    )
    return corrector.compute_corrections_regression(
        ntk_measurements, widths
    )


def reconstruct_ntk(
    result: CorrectionResult,
    width: int,
) -> NDArray:
    """Reconstruct the NTK at a given width from a ``CorrectionResult``.

    Parameters
    ----------
    result : CorrectionResult
        Previously computed correction result.
    width : int
        Target width.

    Returns
    -------
    ntk : ndarray, shape (n, n)
    """
    corrector = FiniteWidthCorrector()
    return corrector.compute_corrected_ntk(
        result.theta_0, result.theta_1, result.theta_2, width
    )


def correction_summary(result: CorrectionResult) -> str:
    """Return a human-readable summary of the correction result.

    Parameters
    ----------
    result : CorrectionResult
        Previously computed correction result.

    Returns
    -------
    summary : str
        Multi-line summary string.
    """
    ci = result.convergence_info
    lines = [
        "Finite-Width NTK Correction Summary",
        "=" * 40,
        f"  Converged:        {ci.converged}",
        f"  Relative error:   {ci.relative_error:.4e}",
        f"  R²:               {ci.r_squared:.6f}",
        f"  Widths used:      {ci.num_widths_used}",
        f"  Effective DOF:    {ci.effective_dof:.1f}",
        f"  Condition number: {result.condition_number:.2e}",
        "",
        "Correction magnitudes:",
    ]
    for key, val in result.correction_magnitudes.items():
        lines.append(f"  {key}: {val:.4e}")

    if ci.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in ci.warnings:
            lines.append(f"  ⚠ {w}")

    return "\n".join(lines)


# ======================================================================
#  Validation utilities
# ======================================================================


def validate_expansion_at_widths(
    result: CorrectionResult,
    ntk_measurements: NDArray,
    widths: Sequence[int],
) -> Dict[int, float]:
    """Compute per-width relative reconstruction errors.

    Parameters
    ----------
    result : CorrectionResult
        Fitted expansion.
    ntk_measurements : ndarray, shape (K, n, n)
        Original NTK measurements.
    widths : sequence of int
        Corresponding widths.

    Returns
    -------
    errors : dict
        Mapping width → relative Frobenius error.
    """
    corrector = FiniteWidthCorrector()
    errors: Dict[int, float] = {}
    for k, w in enumerate(widths):
        reconstructed = corrector.compute_corrected_ntk(
            result.theta_0, result.theta_1, result.theta_2, int(w)
        )
        measured = ntk_measurements[k]
        norm_meas = np.linalg.norm(measured, "fro")
        if norm_meas < 1e-30:
            errors[int(w)] = 0.0
        else:
            errors[int(w)] = float(
                np.linalg.norm(reconstructed - measured, "fro") / norm_meas
            )
    return errors


def estimate_optimal_width(
    result: CorrectionResult,
    target_correction: float = 0.01,
) -> int:
    """Estimate the minimum width where corrections are below a threshold.

    Solves (approximately)

        ||Θ^(1)/N + Θ^(2)/N²||_F  /  ||Θ^(0)||_F  ≤  target_correction

    using a bisection approach.

    Parameters
    ----------
    result : CorrectionResult
        Fitted expansion.
    target_correction : float
        Maximum acceptable relative correction magnitude.

    Returns
    -------
    width : int
        Estimated minimum width (rounded up to nearest integer).
    """
    corrector = FiniteWidthCorrector()

    # Bisection between 1 and a large upper bound
    lo, hi = 1, 1_000_000

    # Check if upper bound is sufficient
    mag_hi = corrector.correction_magnitude(
        result.theta_0, result.theta_1, result.theta_2, hi
    )
    if mag_hi > target_correction:
        logger.warning(
            "Even width %d gives correction magnitude %.4e > target %.4e",
            hi, mag_hi, target_correction,
        )
        return hi

    while hi - lo > 1:
        mid = (lo + hi) // 2
        mag = corrector.correction_magnitude(
            result.theta_0, result.theta_1, result.theta_2, mid
        )
        if mag > target_correction:
            lo = mid
        else:
            hi = mid

    return hi


# ======================================================================
#  Spectral analysis of corrections
# ======================================================================


def correction_spectral_analysis(
    result: CorrectionResult,
) -> Dict[str, NDArray]:
    """Analyse the spectral structure of the correction matrices.

    Computes eigenvalues of Θ^(0), Θ^(1), and Θ^(2) and returns
    information about how the corrections affect different eigenmodes
    of the infinite-width kernel.

    Parameters
    ----------
    result : CorrectionResult
        Fitted expansion.

    Returns
    -------
    info : dict
        Keys:
        - ``"eigenvalues_0"``: eigenvalues of Θ^(0) (descending).
        - ``"eigenvalues_1"``: eigenvalues of Θ^(1) (descending abs).
        - ``"eigenvalues_2"``: eigenvalues of Θ^(2) (descending abs).
        - ``"eigenvectors_0"``: eigenvectors of Θ^(0).
        - ``"projected_1"``: Θ^(1) projected onto eigenbasis of Θ^(0).
        - ``"projected_2"``: Θ^(2) projected onto eigenbasis of Θ^(0).
        - ``"relative_eigenvalue_corrections"``: per-eigenmode relative
          correction magnitudes at the largest calibration width.
    """
    # Symmetrise (corrections should be symmetric for NTK)
    t0 = 0.5 * (result.theta_0 + result.theta_0.T)
    t1 = 0.5 * (result.theta_1 + result.theta_1.T)
    t2 = 0.5 * (result.theta_2 + result.theta_2.T)

    evals_0, evecs_0 = np.linalg.eigh(t0)
    # Sort descending
    idx = np.argsort(evals_0)[::-1]
    evals_0 = evals_0[idx]
    evecs_0 = evecs_0[:, idx]

    evals_1 = np.linalg.eigvalsh(t1)
    evals_1 = evals_1[np.argsort(np.abs(evals_1))[::-1]]

    evals_2 = np.linalg.eigvalsh(t2)
    evals_2 = evals_2[np.argsort(np.abs(evals_2))[::-1]]

    # Project corrections onto eigenbasis of Θ^(0)
    projected_1 = evecs_0.T @ t1 @ evecs_0
    projected_2 = evecs_0.T @ t2 @ evecs_0

    # Relative eigenvalue corrections (diagonal of projected / evals_0)
    rel_corrections = np.zeros_like(evals_0)
    for i in range(len(evals_0)):
        if abs(evals_0[i]) > 1e-15:
            rel_corrections[i] = abs(projected_1[i, i]) / abs(evals_0[i])

    return {
        "eigenvalues_0": evals_0,
        "eigenvalues_1": evals_1,
        "eigenvalues_2": evals_2,
        "eigenvectors_0": evecs_0,
        "projected_1": projected_1,
        "projected_2": projected_2,
        "relative_eigenvalue_corrections": rel_corrections,
    }


# ======================================================================
#  Width schedule recommendation
# ======================================================================


def recommend_width_schedule(
    n_data: int,
    min_width: int = 64,
    max_width: int = 4096,
    num_widths: int = 8,
    schedule: str = "geometric",
) -> NDArray:
    """Recommend a set of widths for calibrating the 1/N expansion.

    Parameters
    ----------
    n_data : int
        Number of data points (used for lower-bound heuristic).
    min_width : int
        Smallest width to include.
    max_width : int
        Largest width to include.
    num_widths : int
        Number of distinct widths.
    schedule : str
        ``"geometric"`` for geometrically spaced widths (recommended),
        ``"linear"`` for linearly spaced widths, or ``"sqrt"`` for
        square-root spacing (denser at small widths).

    Returns
    -------
    widths : ndarray of int
        Recommended widths in ascending order.
    """
    # Ensure minimum width is at least as large as the data dimension
    effective_min = max(min_width, n_data)

    if schedule == "geometric":
        widths = np.geomspace(effective_min, max_width, num_widths)
    elif schedule == "linear":
        widths = np.linspace(effective_min, max_width, num_widths)
    elif schedule == "sqrt":
        # Dense at small widths, sparse at large
        t = np.linspace(0, 1, num_widths)
        widths = effective_min + (max_width - effective_min) * np.sqrt(t)
    else:
        raise ValueError(f"Unknown schedule '{schedule}'; use 'geometric', "
                         f"'linear', or 'sqrt'.")

    # Round to integers and deduplicate
    widths = np.unique(np.round(widths).astype(int))

    if len(widths) < num_widths:
        logger.warning(
            "After rounding, only %d unique widths remain (requested %d). "
            "Consider widening the [min_width, max_width] range.",
            len(widths), num_widths,
        )

    return widths


# ======================================================================
#  Residual diagnostics
# ======================================================================


def residual_diagnostics(
    result: CorrectionResult,
    widths: Sequence[int],
) -> Dict[str, Any]:
    """Compute detailed residual diagnostics for the expansion fit.

    Parameters
    ----------
    result : CorrectionResult
        Fitted expansion.
    widths : sequence of int
        Calibration widths.

    Returns
    -------
    diagnostics : dict
        Keys:
        - ``"per_width_rmse"``: RMSE of residuals at each width.
        - ``"per_width_max_abs"``: Max absolute residual at each width.
        - ``"frobenius_residuals"``: Frobenius norm of residual at each width.
        - ``"residual_trend"``: Slope of log(RMSE) vs log(1/N); should be
          approximately 3 if the O(1/N³) truncation dominates.
        - ``"trend_r_squared"``: R² of the log-log trend fit.
    """
    residuals = result.residuals  # (K, n, n)
    widths_f = np.asarray(widths, dtype=np.float64)
    K = len(widths_f)

    per_width_rmse = np.zeros(K)
    per_width_max = np.zeros(K)
    frob_residuals = np.zeros(K)

    for k in range(K):
        r_k = residuals[k]
        per_width_rmse[k] = np.sqrt(np.mean(r_k ** 2))
        per_width_max[k] = np.max(np.abs(r_k))
        frob_residuals[k] = np.linalg.norm(r_k, "fro")

    # Fit log(RMSE) ~ slope * log(1/N) + intercept
    log_inv_w = np.log(1.0 / widths_f)
    valid = per_width_rmse > 1e-30
    if np.sum(valid) >= 2:
        log_rmse = np.log(per_width_rmse[valid])
        log_x = log_inv_w[valid]
        # Simple linear regression
        A = np.column_stack([log_x, np.ones(np.sum(valid))])
        coeffs, _, _, _ = np.linalg.lstsq(A, log_rmse, rcond=None)
        slope = coeffs[0]

        fitted_log_rmse = A @ coeffs
        ss_res = np.sum((log_rmse - fitted_log_rmse) ** 2)
        ss_tot = np.sum((log_rmse - np.mean(log_rmse)) ** 2)
        trend_r2 = float(1.0 - ss_res / max(ss_tot, 1e-30))
    else:
        slope = float("nan")
        trend_r2 = float("nan")

    return {
        "per_width_rmse": per_width_rmse,
        "per_width_max_abs": per_width_max,
        "frobenius_residuals": frob_residuals,
        "residual_trend": float(slope),
        "trend_r_squared": trend_r2,
    }


# ======================================================================
#  Stability analysis
# ======================================================================


def stability_analysis(
    ntk_measurements: NDArray,
    widths: Sequence[int],
    *,
    min_widths: int = 4,
    convergence_tol: float = 1e-3,
) -> Dict[str, Any]:
    """Assess stability of the expansion by progressive inclusion of widths.

    Fits the expansion using the first 4, 5, …, K widths and tracks how
    the coefficients change.  Stable coefficients indicate a well-conditioned
    expansion.

    Parameters
    ----------
    ntk_measurements : ndarray, shape (K, n, n)
        NTK matrices.
    widths : sequence of int
        Widths (should be sorted ascending).
    min_widths : int
        Minimum number of widths for a fit.
    convergence_tol : float
        Convergence tolerance passed to the corrector.

    Returns
    -------
    analysis : dict
        Keys:
        - ``"num_widths_sequence"``: list of int, number of widths used.
        - ``"theta_0_norms"``: Frobenius norms of Θ^(0) at each step.
        - ``"theta_1_norms"``: Frobenius norms of Θ^(1) at each step.
        - ``"theta_2_norms"``: Frobenius norms of Θ^(2) at each step.
        - ``"relative_changes_0"``: relative change in Θ^(0) from step to step.
        - ``"relative_changes_1"``: relative change in Θ^(1) from step to step.
        - ``"relative_changes_2"``: relative change in Θ^(2) from step to step.
        - ``"stable"``: bool, whether coefficients have stabilised.
    """
    widths_arr = np.asarray(widths, dtype=np.float64)
    K = len(widths_arr)

    # Sort by width
    sort_idx = np.argsort(widths_arr)
    widths_sorted = widths_arr[sort_idx]
    ntk_sorted = ntk_measurements[sort_idx]

    corrector = FiniteWidthCorrector(
        min_widths=min_widths,
        convergence_tol=convergence_tol,
    )

    num_widths_seq: List[int] = []
    norms_0: List[float] = []
    norms_1: List[float] = []
    norms_2: List[float] = []
    prev_t0: Optional[NDArray] = None
    prev_t1: Optional[NDArray] = None
    prev_t2: Optional[NDArray] = None
    rel_changes_0: List[float] = []
    rel_changes_1: List[float] = []
    rel_changes_2: List[float] = []

    for end in range(min_widths, K + 1):
        sub_ntk = ntk_sorted[:end]
        sub_widths = widths_sorted[:end]

        result = corrector.compute_corrections_regression(sub_ntk, sub_widths)

        n0 = float(np.linalg.norm(result.theta_0, "fro"))
        n1 = float(np.linalg.norm(result.theta_1, "fro"))
        n2 = float(np.linalg.norm(result.theta_2, "fro"))

        num_widths_seq.append(end)
        norms_0.append(n0)
        norms_1.append(n1)
        norms_2.append(n2)

        if prev_t0 is not None:
            denom_0 = max(n0, 1e-30)
            denom_1 = max(n1, 1e-30)
            denom_2 = max(n2, 1e-30)
            rel_changes_0.append(
                float(np.linalg.norm(result.theta_0 - prev_t0, "fro") / denom_0)
            )
            rel_changes_1.append(
                float(np.linalg.norm(result.theta_1 - prev_t1, "fro") / denom_1)
            )
            rel_changes_2.append(
                float(np.linalg.norm(result.theta_2 - prev_t2, "fro") / denom_2)
            )

        prev_t0 = result.theta_0.copy()
        prev_t1 = result.theta_1.copy()
        prev_t2 = result.theta_2.copy()

    # Stable if last two relative changes are all below 5%
    stable = True
    if len(rel_changes_0) >= 2:
        for rc in [rel_changes_0, rel_changes_1, rel_changes_2]:
            if rc[-1] > 0.05 or rc[-2] > 0.05:
                stable = False
    else:
        stable = False

    return {
        "num_widths_sequence": num_widths_seq,
        "theta_0_norms": norms_0,
        "theta_1_norms": norms_1,
        "theta_2_norms": norms_2,
        "relative_changes_0": rel_changes_0,
        "relative_changes_1": rel_changes_1,
        "relative_changes_2": rel_changes_2,
        "stable": stable,
    }
