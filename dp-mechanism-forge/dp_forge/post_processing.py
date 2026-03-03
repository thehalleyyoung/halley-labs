"""
Post-processing optimality for DP-Forge.

This module provides optimal post-processing methods for differentially
private mechanism outputs.  By the post-processing theorem, any
data-independent transformation of a DP mechanism's output preserves the
same privacy guarantee.  The methods here exploit this to minimize error.

Key Components:
    - ``PostProcessor``: Base class with privacy-preservation proof.
    - ``LMMSEEstimator``: Linear minimum mean squared error post-processing
      for workload mechanisms.  Given noisy ``y = Ax + noise``, computes
      ``x̂ = (AᵀA + σ²I)⁻¹Aᵀy``.
    - ``BayesOptimalEstimator``: Prior-aware denoising under Gaussian or
      uniform prior distribution over the true data.
    - ``BiasVarianceDecomposer``: Decompose total MSE into bias² + variance.
    - ``WienerFilter``: Frequency-domain optimal filter for periodic queries.

Mathematical Background:
    Post-processing theorem: If M is (ε,δ)-DP and f is any
    (possibly randomized) mapping, then f ∘ M is also (ε,δ)-DP.

    LMMSE: For y = Ax + η where η ~ N(0, σ²I), the LMMSE estimate is
        x̂ = (AᵀA + σ²I)⁻¹ Aᵀy
    which minimizes E[||x - x̂||²] over all linear estimators.

    Wiener filter: In the frequency domain, the optimal filter for
    periodic queries is H(ω) = S_xx(ω) / (S_xx(ω) + S_nn(ω)).

Usage::

    from dp_forge.post_processing import LMMSEEstimator, BiasVarianceDecomposer

    est = LMMSEEstimator(workload_matrix=A, noise_variance=sigma2)
    x_hat = est.denoise(y_noisy)
    bv = BiasVarianceDecomposer(A, sigma2)
    bias_sq, var = bv.bias_variance(x_hat, x_true)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import linalg, sparse

from .exceptions import (
    ConfigurationError,
    DPForgeError,
    NumericalInstabilityError,
)
from .types import WorkloadSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Post-processing theorem proof
# ---------------------------------------------------------------------------


@dataclass
class PrivacyPreservationProof:
    """Proof that a post-processing step preserves DP.

    Attributes:
        processor_name: Name of the post-processing method.
        is_data_independent: Whether the processor is data-independent.
        epsilon: Privacy parameter of the input mechanism.
        delta: Privacy parameter of the input mechanism.
        proof_text: Human-readable proof text.
    """

    processor_name: str
    is_data_independent: bool
    epsilon: float
    delta: float
    proof_text: str

    def __repr__(self) -> str:
        status = "preserves" if self.is_data_independent else "may violate"
        return (
            f"PrivacyPreservationProof({self.processor_name} {status} "
            f"({self.epsilon},{self.delta})-DP)"
        )


# ---------------------------------------------------------------------------
# Base PostProcessor
# ---------------------------------------------------------------------------


class PostProcessor(ABC):
    """Abstract base class for post-processing DP mechanism outputs.

    Subclasses implement ``denoise()`` to transform noisy mechanism
    outputs into improved estimates.  The base class provides a
    ``privacy_proof()`` method that verifies the post-processing
    theorem applies (i.e., the transformation is data-independent).

    The post-processing theorem guarantees that any data-independent
    function applied to the output of an (ε,δ)-DP mechanism yields
    an (ε,δ)-DP result.
    """

    @abstractmethod
    def denoise(
        self, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Apply post-processing to noisy mechanism output.

        Args:
            y: Noisy output vector from the DP mechanism.

        Returns:
            Denoised estimate of the true query answer.
        """

    def privacy_proof(
        self, epsilon: float, delta: float = 0.0
    ) -> PrivacyPreservationProof:
        """Generate a proof that this post-processor preserves DP.

        The proof follows from the post-processing theorem: since
        the transformation depends only on the noisy output (not
        on the raw data), it cannot degrade the privacy guarantee.

        Args:
            epsilon: Privacy parameter of the input mechanism.
            delta: Privacy parameter of the input mechanism.

        Returns:
            A :class:`PrivacyPreservationProof` instance.
        """
        name = self.__class__.__name__
        proof_text = (
            f"Post-Processing Theorem Application:\n"
            f"  Processor: {name}\n"
            f"  Input mechanism: ({epsilon}, {delta})-DP\n"
            f"  The {name} transformation is a deterministic function of\n"
            f"  the mechanism output y only.  It does not access the raw\n"
            f"  database x.  By the post-processing theorem (Dwork & Roth,\n"
            f"  Theorem 2.1), any data-independent function applied to the\n"
            f"  output of an (ε,δ)-DP mechanism preserves (ε,δ)-DP.\n"
            f"  Therefore, the post-processed output remains ({epsilon}, {delta})-DP."
        )
        return PrivacyPreservationProof(
            processor_name=name,
            is_data_independent=True,
            epsilon=epsilon,
            delta=delta,
            proof_text=proof_text,
        )


# ---------------------------------------------------------------------------
# LMMSEEstimator
# ---------------------------------------------------------------------------


class LMMSEEstimator(PostProcessor):
    """Linear Minimum Mean Squared Error estimator.

    Given noisy observations ``y = Ax + η`` where ``η ~ N(0, σ²I)``,
    computes the LMMSE estimate:

        x̂ = (AᵀA + σ²I)⁻¹ Aᵀy

    This minimizes E[||x - x̂||²] among all linear estimators.
    Handles rank-deficient workload matrices via pseudoinverse or
    Tikhonov regularization.

    Args:
        workload_matrix: The m × d workload matrix A.
        noise_variance: Noise variance σ² > 0.
        regularization: Additional regularization parameter λ.
            The estimator uses (AᵀA + (σ² + λ)I)⁻¹.
        max_condition_number: Threshold for numerical stability check.

    Raises:
        ConfigurationError: If noise_variance <= 0.
    """

    def __init__(
        self,
        workload_matrix: npt.NDArray[np.float64],
        noise_variance: float,
        *,
        regularization: float = 0.0,
        max_condition_number: float = 1e12,
    ) -> None:
        if noise_variance <= 0:
            raise ConfigurationError(
                "noise_variance must be positive",
                parameter="noise_variance",
                value=noise_variance,
                constraint="noise_variance > 0",
            )
        if regularization < 0:
            raise ConfigurationError(
                "regularization must be non-negative",
                parameter="regularization",
                value=regularization,
                constraint="regularization >= 0",
            )

        self.A = np.asarray(workload_matrix, dtype=np.float64)
        if self.A.ndim != 2:
            raise ConfigurationError(
                "workload_matrix must be 2-D",
                parameter="workload_matrix",
                value=f"shape={self.A.shape}",
                constraint="ndim == 2",
            )

        self.noise_variance = noise_variance
        self.regularization = regularization
        self.max_condition_number = max_condition_number

        # Precompute the LMMSE matrix: (AᵀA + reg·I)⁻¹ Aᵀ
        self._m, self._d = self.A.shape
        self._lmmse_matrix = self._compute_lmmse_matrix()

    def _compute_lmmse_matrix(self) -> npt.NDArray[np.float64]:
        """Compute (AᵀA + reg·I)⁻¹ Aᵀ with numerical stability checks."""
        AtA = self.A.T @ self.A
        reg = self.noise_variance + self.regularization
        M = AtA + reg * np.eye(self._d)

        # Condition number check
        try:
            cond = np.linalg.cond(M)
        except np.linalg.LinAlgError:
            cond = float("inf")

        if cond > self.max_condition_number:
            logger.warning(
                "LMMSE matrix condition number %.2e exceeds threshold %.2e; "
                "using pseudoinverse",
                cond, self.max_condition_number,
            )
            return np.linalg.pinv(self.A) if self._m >= self._d else self.A.T @ np.linalg.pinv(self.A @ self.A.T + reg * np.eye(self._m))

        try:
            # Use Cholesky for efficiency since M is SPD
            L = linalg.cholesky(M, lower=True)
            # Solve L L^T X = A^T for X, giving (AᵀA + reg·I)⁻¹ Aᵀ
            Z = linalg.solve_triangular(L, self.A.T, lower=True)
            return linalg.solve_triangular(L.T, Z, lower=False)
        except linalg.LinAlgError:
            logger.warning("Cholesky failed; falling back to pseudoinverse")
            return np.linalg.lstsq(M, self.A.T, rcond=None)[0]

    def denoise(
        self, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Apply LMMSE denoising to noisy workload answers.

        Args:
            y: Noisy answer vector of shape (m,) or (m, n_samples).

        Returns:
            Denoised estimate x̂ of shape (d,) or (d, n_samples).

        Raises:
            DPForgeError: If y has incompatible dimensions.
        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            if len(y) != self._m:
                raise DPForgeError(
                    f"y length {len(y)} != workload rows {self._m}",
                    context={"expected": self._m, "got": len(y)},
                )
            return self._lmmse_matrix @ y
        elif y.ndim == 2:
            if y.shape[0] != self._m:
                raise DPForgeError(
                    f"y rows {y.shape[0]} != workload rows {self._m}",
                    context={"expected": self._m, "got": y.shape[0]},
                )
            return self._lmmse_matrix @ y
        else:
            raise DPForgeError(
                f"y must be 1-D or 2-D, got shape {y.shape}",
            )

    @property
    def mse_matrix(self) -> npt.NDArray[np.float64]:
        """MSE covariance matrix of the LMMSE estimator.

        Returns:
            The d × d matrix σ² (AᵀA + σ²I)⁻¹.
        """
        AtA = self.A.T @ self.A
        reg = self.noise_variance + self.regularization
        M = AtA + reg * np.eye(self._d)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)
        return self.noise_variance * M_inv

    @property
    def total_mse(self) -> float:
        """Total expected MSE: tr(σ² (AᵀA + σ²I)⁻¹).

        Returns:
            Scalar total MSE.
        """
        return float(np.trace(self.mse_matrix))

    def optimal_filter(self) -> npt.NDArray[np.float64]:
        """Return the LMMSE filter matrix.

        Returns:
            The d × m matrix (AᵀA + σ²I)⁻¹ Aᵀ.
        """
        return self._lmmse_matrix.copy()

    @classmethod
    def from_workload_spec(
        cls,
        spec: WorkloadSpec,
        noise_variance: float,
        **kwargs: Any,
    ) -> LMMSEEstimator:
        """Construct from a WorkloadSpec.

        Args:
            spec: Workload specification with matrix A.
            noise_variance: Noise variance σ².
            **kwargs: Additional arguments passed to constructor.

        Returns:
            An LMMSEEstimator instance.
        """
        return cls(
            workload_matrix=spec.matrix,
            noise_variance=noise_variance,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# BayesOptimalEstimator
# ---------------------------------------------------------------------------


class BayesOptimalEstimator(PostProcessor):
    """Bayes-optimal denoising with known prior distribution.

    Computes the posterior mean estimator E[x | y] under either a
    Gaussian or uniform prior distribution on the true data x.

    For Gaussian prior x ~ N(μ_prior, Σ_prior) and observation
    y = x + η with η ~ N(0, σ²):
        x̂ = Σ_post (Σ_prior⁻¹ μ_prior + y/σ²)
        where Σ_post = (Σ_prior⁻¹ + I/σ²)⁻¹

    For uniform prior x ∈ [a, b] and observation y = x + η:
        x̂ = E[x | y] = ∫ x · p(y|x) dx / ∫ p(y|x) dx
        computed numerically over the support.

    Args:
        noise_variance: Noise variance σ² > 0.
        prior: Prior type, either "gaussian" or "uniform".
        prior_mean: Mean of Gaussian prior (scalar or array).
        prior_variance: Variance of Gaussian prior (scalar or array).
        prior_lower: Lower bound for uniform prior.
        prior_upper: Upper bound for uniform prior.
        n_quadrature: Number of quadrature points for uniform prior.

    Raises:
        ConfigurationError: If noise_variance <= 0 or prior params invalid.
    """

    def __init__(
        self,
        noise_variance: float,
        prior: str = "gaussian",
        *,
        prior_mean: Union[float, npt.NDArray[np.float64]] = 0.0,
        prior_variance: Union[float, npt.NDArray[np.float64]] = 1.0,
        prior_lower: float = 0.0,
        prior_upper: float = 1.0,
        n_quadrature: int = 1000,
    ) -> None:
        if noise_variance <= 0:
            raise ConfigurationError(
                "noise_variance must be positive",
                parameter="noise_variance",
                value=noise_variance,
                constraint="noise_variance > 0",
            )
        if prior not in ("gaussian", "uniform"):
            raise ConfigurationError(
                f"prior must be 'gaussian' or 'uniform', got {prior!r}",
                parameter="prior",
                value=prior,
                constraint="prior in ('gaussian', 'uniform')",
            )

        self.noise_variance = noise_variance
        self.prior = prior
        self.prior_mean = np.atleast_1d(np.asarray(prior_mean, dtype=np.float64))
        self.prior_variance = np.atleast_1d(np.asarray(prior_variance, dtype=np.float64))
        self.prior_lower = prior_lower
        self.prior_upper = prior_upper
        self.n_quadrature = n_quadrature

        if prior == "gaussian":
            if np.any(self.prior_variance <= 0):
                raise ConfigurationError(
                    "prior_variance must be positive",
                    parameter="prior_variance",
                    value=prior_variance,
                    constraint="prior_variance > 0",
                )
        elif prior == "uniform":
            if prior_lower >= prior_upper:
                raise ConfigurationError(
                    "prior_lower must be < prior_upper",
                    parameter="prior_lower",
                    value=f"[{prior_lower}, {prior_upper}]",
                    constraint="prior_lower < prior_upper",
                )

    def denoise(
        self, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Apply Bayes-optimal denoising.

        Args:
            y: Noisy observation(s). Shape (d,) or (d, n_samples).

        Returns:
            Posterior mean estimate x̂.
        """
        y = np.asarray(y, dtype=np.float64)

        if self.prior == "gaussian":
            return self._denoise_gaussian(y)
        else:
            return self._denoise_uniform(y)

    def _denoise_gaussian(
        self, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Gaussian prior: closed-form posterior mean.

        x̂ = (σ²_prior / (σ²_prior + σ²)) · y + (σ² / (σ²_prior + σ²)) · μ
        """
        sigma2 = self.noise_variance
        # Broadcast prior_variance to match y dimensions
        if y.ndim == 1:
            pv = np.broadcast_to(self.prior_variance, y.shape)
            pm = np.broadcast_to(self.prior_mean, y.shape)
        else:
            pv = np.broadcast_to(self.prior_variance.reshape(-1, 1), y.shape)
            pm = np.broadcast_to(self.prior_mean.reshape(-1, 1), y.shape)

        weight_data = pv / (pv + sigma2)
        weight_prior = sigma2 / (pv + sigma2)
        return weight_data * y + weight_prior * pm

    def _denoise_uniform(
        self, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Uniform prior: numerical integration for posterior mean."""
        a, b = self.prior_lower, self.prior_upper
        # Quadrature grid
        x_grid = np.linspace(a, b, self.n_quadrature)
        dx = (b - a) / max(self.n_quadrature - 1, 1)
        sigma2 = self.noise_variance

        def _posterior_mean_scalar(y_val: float) -> float:
            # p(y|x) ∝ exp(-(y-x)²/(2σ²))
            log_lik = -0.5 * (y_val - x_grid) ** 2 / sigma2
            # Numerical stability: shift by max
            log_lik -= np.max(log_lik)
            weights = np.exp(log_lik)
            return float(np.sum(x_grid * weights) / np.sum(weights))

        y_flat = y.ravel()
        result = np.array([_posterior_mean_scalar(yi) for yi in y_flat])
        return result.reshape(y.shape)

    def posterior_variance(
        self, y: Optional[npt.NDArray[np.float64]] = None,
    ) -> Union[float, npt.NDArray[np.float64]]:
        """Compute posterior variance.

        For Gaussian prior, the posterior variance is independent of y:
            Var[x|y] = σ² · σ²_prior / (σ² + σ²_prior)

        For uniform prior, requires y for numerical computation.

        Args:
            y: Observation (required for uniform prior).

        Returns:
            Posterior variance (scalar or array).
        """
        sigma2 = self.noise_variance

        if self.prior == "gaussian":
            pv = self.prior_variance
            return float(sigma2 * pv / (sigma2 + pv)) if pv.size == 1 else sigma2 * pv / (sigma2 + pv)

        # Uniform prior: numerical computation
        if y is None:
            raise DPForgeError(
                "y is required for posterior variance under uniform prior"
            )
        a, b = self.prior_lower, self.prior_upper
        x_grid = np.linspace(a, b, self.n_quadrature)

        def _var_scalar(y_val: float) -> float:
            log_lik = -0.5 * (y_val - x_grid) ** 2 / sigma2
            log_lik -= np.max(log_lik)
            weights = np.exp(log_lik)
            weights /= np.sum(weights)
            mean = np.sum(x_grid * weights)
            return float(np.sum((x_grid - mean) ** 2 * weights))

        y_flat = np.asarray(y, dtype=np.float64).ravel()
        result = np.array([_var_scalar(yi) for yi in y_flat])
        if y_flat.size == 1:
            return float(result[0])
        return result.reshape(np.asarray(y).shape)

    def bias_variance(
        self,
        x_true: npt.NDArray[np.float64],
    ) -> Tuple[float, float]:
        """Compute expected bias² and variance for the estimator.

        For Gaussian prior:
            bias² = (σ² / (σ² + σ²_prior))² · ||x - μ||²  (conditional on x)
            variance = σ² · σ²_prior / (σ² + σ²_prior)  (per component)

        Args:
            x_true: True data vector.

        Returns:
            Tuple (expected_bias_squared, expected_variance).
        """
        x_true = np.asarray(x_true, dtype=np.float64)
        sigma2 = self.noise_variance

        if self.prior == "gaussian":
            pv = np.broadcast_to(self.prior_variance, x_true.shape)
            pm = np.broadcast_to(self.prior_mean, x_true.shape)
            shrinkage = sigma2 / (sigma2 + pv)
            bias_sq = float(np.sum(shrinkage ** 2 * (x_true - pm) ** 2))
            var = float(np.sum(pv * sigma2 / (pv + sigma2)))
            return bias_sq, var

        # Uniform: approximate via Monte Carlo
        n_mc = 10000
        rng = np.random.default_rng(42)
        noise = rng.normal(0, math.sqrt(sigma2), size=(n_mc, len(x_true)))
        y_samples = x_true + noise
        estimates = np.array([self.denoise(y_samples[i]) for i in range(n_mc)])
        mean_estimate = np.mean(estimates, axis=0)
        bias_sq = float(np.sum((mean_estimate - x_true) ** 2))
        var = float(np.sum(np.var(estimates, axis=0)))
        return bias_sq, var


# ---------------------------------------------------------------------------
# BiasVarianceDecomposer
# ---------------------------------------------------------------------------


class BiasVarianceDecomposer:
    """Decompose MSE into bias² + variance for linear post-processing.

    For a linear estimator x̂ = Fy applied to y = Ax + η, the MSE
    decomposes as:

        MSE = ||FA - I||² · ||x||²  (bias²)
            + σ² · tr(FFᵀ)          (variance)

    Args:
        workload_matrix: The m × d workload matrix A.
        noise_variance: Noise variance σ².
        filter_matrix: The d × m post-processing matrix F.
            If None, uses the LMMSE filter.
    """

    def __init__(
        self,
        workload_matrix: npt.NDArray[np.float64],
        noise_variance: float,
        filter_matrix: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        self.A = np.asarray(workload_matrix, dtype=np.float64)
        if self.A.ndim != 2:
            raise ConfigurationError(
                "workload_matrix must be 2-D",
                parameter="workload_matrix",
                value=f"shape={self.A.shape}",
            )

        self.noise_variance = noise_variance
        m, d = self.A.shape

        if filter_matrix is not None:
            self.F = np.asarray(filter_matrix, dtype=np.float64)
            if self.F.shape != (d, m):
                raise ConfigurationError(
                    f"filter_matrix shape {self.F.shape} != expected ({d}, {m})",
                    parameter="filter_matrix",
                )
        else:
            # Default to LMMSE
            AtA = self.A.T @ self.A
            reg = noise_variance * np.eye(d)
            try:
                self.F = np.linalg.solve(AtA + reg, self.A.T)
            except np.linalg.LinAlgError:
                self.F = np.linalg.lstsq(AtA + reg, self.A.T, rcond=None)[0]

    def bias_variance(
        self,
        x_true: Optional[npt.NDArray[np.float64]] = None,
    ) -> Tuple[float, float]:
        """Compute bias² and variance components of MSE.

        Args:
            x_true: True data vector. If None, returns the worst-case
                bias over unit-norm x.

        Returns:
            Tuple (bias_squared, variance).
        """
        d = self.A.shape[1]
        bias_matrix = self.F @ self.A - np.eye(d)

        if x_true is not None:
            x_true = np.asarray(x_true, dtype=np.float64)
            bias_vec = bias_matrix @ x_true
            bias_sq = float(np.sum(bias_vec ** 2))
        else:
            # Worst-case bias: largest singular value of bias_matrix squared
            s = np.linalg.svd(bias_matrix, compute_uv=False)
            bias_sq = float(s[0] ** 2) if len(s) > 0 else 0.0

        variance = self.noise_variance * float(np.trace(self.F @ self.F.T))

        return bias_sq, variance

    def total_mse(
        self,
        x_true: Optional[npt.NDArray[np.float64]] = None,
    ) -> float:
        """Total MSE = bias² + variance.

        Args:
            x_true: True data vector. If None, worst-case bias.

        Returns:
            Total MSE as a scalar.
        """
        bias_sq, var = self.bias_variance(x_true)
        return bias_sq + var

    def decomposition_report(
        self,
        x_true: Optional[npt.NDArray[np.float64]] = None,
    ) -> Dict[str, float]:
        """Generate a detailed decomposition report.

        Args:
            x_true: True data vector.

        Returns:
            Dict with bias², variance, total MSE, and ratios.
        """
        bias_sq, var = self.bias_variance(x_true)
        total = bias_sq + var
        return {
            "bias_squared": bias_sq,
            "variance": var,
            "total_mse": total,
            "bias_fraction": bias_sq / max(total, 1e-15),
            "variance_fraction": var / max(total, 1e-15),
            "noise_variance": self.noise_variance,
        }


# ---------------------------------------------------------------------------
# WienerFilter
# ---------------------------------------------------------------------------


class WienerFilter(PostProcessor):
    """Frequency-domain optimal filter for periodic queries.

    For periodic workload queries with known signal and noise power
    spectral densities, the Wiener filter computes the minimum MSE
    estimate in the frequency domain:

        H(ω) = S_xx(ω) / (S_xx(ω) + S_nn(ω))

    where S_xx is the signal PSD and S_nn is the noise PSD.

    Args:
        signal_psd: Signal power spectral density, shape (n_freq,).
            If None, assumed flat (white signal).
        noise_psd: Noise power spectral density, shape (n_freq,).
            If None, assumed flat with given noise_variance.
        noise_variance: Scalar noise variance (used if noise_psd is None).
        n_freq: Number of frequency bins (required if PSDs not given).

    Raises:
        ConfigurationError: If both PSDs are None and n_freq not given.
    """

    def __init__(
        self,
        signal_psd: Optional[npt.NDArray[np.float64]] = None,
        noise_psd: Optional[npt.NDArray[np.float64]] = None,
        *,
        noise_variance: float = 1.0,
        n_freq: Optional[int] = None,
        signal_variance: float = 1.0,
    ) -> None:
        if noise_variance <= 0:
            raise ConfigurationError(
                "noise_variance must be positive",
                parameter="noise_variance",
                value=noise_variance,
            )
        if signal_variance <= 0:
            raise ConfigurationError(
                "signal_variance must be positive",
                parameter="signal_variance",
                value=signal_variance,
            )

        self.noise_variance = noise_variance
        self.signal_variance = signal_variance

        # Determine frequency dimension
        if signal_psd is not None:
            self._n_freq = len(signal_psd)
            self.signal_psd = np.asarray(signal_psd, dtype=np.float64)
        elif noise_psd is not None:
            self._n_freq = len(noise_psd)
            self.signal_psd = np.full(self._n_freq, signal_variance, dtype=np.float64)
        elif n_freq is not None:
            self._n_freq = n_freq
            self.signal_psd = np.full(n_freq, signal_variance, dtype=np.float64)
        else:
            raise ConfigurationError(
                "At least one of signal_psd, noise_psd, or n_freq must be given",
                parameter="n_freq",
            )

        if noise_psd is not None:
            self.noise_psd = np.asarray(noise_psd, dtype=np.float64)
            if len(self.noise_psd) != self._n_freq:
                raise ConfigurationError(
                    f"noise_psd length {len(self.noise_psd)} != signal_psd "
                    f"length {self._n_freq}",
                    parameter="noise_psd",
                )
        else:
            self.noise_psd = np.full(self._n_freq, noise_variance, dtype=np.float64)

        # Compute Wiener filter in frequency domain
        self._filter = self._compute_filter()

    def _compute_filter(self) -> npt.NDArray[np.float64]:
        """Compute H(ω) = S_xx(ω) / (S_xx(ω) + S_nn(ω))."""
        denom = self.signal_psd + self.noise_psd
        # Avoid division by zero
        safe_denom = np.where(denom > 0, denom, 1.0)
        H = np.where(denom > 0, self.signal_psd / safe_denom, 0.0)
        return H

    def denoise(
        self, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Apply Wiener filter in the frequency domain.

        Takes the DFT of the input, multiplies by the filter, and
        takes the inverse DFT.

        Args:
            y: Noisy signal of length n_freq.

        Returns:
            Denoised signal of the same shape.
        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            if len(y) != self._n_freq:
                raise DPForgeError(
                    f"y length {len(y)} != n_freq {self._n_freq}",
                )
            Y = np.fft.fft(y)
            X_hat = Y * self._filter
            return np.real(np.fft.ifft(X_hat))
        elif y.ndim == 2:
            if y.shape[0] != self._n_freq:
                raise DPForgeError(
                    f"y rows {y.shape[0]} != n_freq {self._n_freq}",
                )
            Y = np.fft.fft(y, axis=0)
            X_hat = Y * self._filter[:, np.newaxis]
            return np.real(np.fft.ifft(X_hat, axis=0))
        else:
            raise DPForgeError(f"y must be 1-D or 2-D, got shape {y.shape}")

    def optimal_filter(self) -> npt.NDArray[np.float64]:
        """Return the Wiener filter coefficients H(ω).

        Returns:
            Array of shape (n_freq,) with filter values in [0, 1].
        """
        return self._filter.copy()

    @property
    def output_snr(self) -> npt.NDArray[np.float64]:
        """Output SNR after filtering, per frequency bin.

        Returns:
            Array of SNR values (S_xx · H²) / (S_nn · H²) = S_xx / S_nn
            where the filter is applied.
        """
        safe_noise = np.where(self.noise_psd > 0, self.noise_psd, 1.0)
        return np.where(self.noise_psd > 0, self.signal_psd / safe_noise, float("inf"))

    @property
    def total_output_mse(self) -> float:
        """Total MSE of the Wiener-filtered output.

        MSE = (1/N) Σ S_xx(ω) S_nn(ω) / (S_xx(ω) + S_nn(ω))

        Returns:
            Total MSE as a scalar.
        """
        denom = self.signal_psd + self.noise_psd
        safe_denom = np.where(denom > 0, denom, 1.0)
        per_freq_mse = np.where(
            denom > 0,
            self.signal_psd * self.noise_psd / safe_denom,
            0.0,
        )
        return float(np.mean(per_freq_mse))
