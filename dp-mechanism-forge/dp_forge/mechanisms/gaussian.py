"""
Gaussian workload mechanism implementation for DP-Forge.

Provides the :class:`GaussianWorkloadMechanism` class, which wraps the
noise covariance matrix produced by the SDP-based workload mechanism
synthesis path.

A Gaussian workload mechanism answers a linear workload A ∈ R^{m×d} by
adding zero-mean Gaussian noise: answer = A·x + z, where z ~ N(0, Σ).
The covariance Σ is optimised by the SDP to minimise total MSE subject
to (ε, δ)-DP constraints.

Features:
    - **Sampling**: Draw noisy answers for a true data vector.
    - **Batch sampling**: Vectorised sampling for multiple data vectors.
    - **Analytical MSE**: Per-query and total MSE from the covariance.
    - **Covariance access**: Full noise covariance and its decomposition.
    - **Validity checking**: PSD and privacy constraint verification.
    - **Decomposition**: Eigendecomposition of the noise structure.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
from scipy import linalg as sp_linalg

from dp_forge.exceptions import (
    ConfigurationError,
    InvalidMechanismError,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]


# =========================================================================
# GaussianWorkloadMechanism
# =========================================================================


class GaussianWorkloadMechanism:
    """Gaussian mechanism for linear workload queries.

    Given a workload matrix A ∈ R^{m×d} and a noise covariance Σ ∈ R^{d×d},
    this mechanism computes noisy answers as:

        answer = A · x + z,    z ~ N(0, A · Σ · A^T)

    The covariance Σ is typically the solution of an SDP that minimises
    tr(A Σ A^T) (total MSE) subject to the (ε, δ)-DP constraint:

        ||x − x'||_Σ ≥ √(2 ln(1.25/δ)) · Δ / ε

    for all adjacent pairs (x, x').

    Attributes:
        sigma: Noise covariance matrix Σ, shape (d, d).
        workload_A: Workload matrix A, shape (m, d).
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        metadata: Additional metadata.

    Usage::

        mech = GaussianWorkloadMechanism(sigma, A, epsilon=1.0, delta=1e-5)
        noisy_answer = mech.sample(true_data)
        per_query_mse = mech.mse_per_query()
    """

    def __init__(
        self,
        sigma: FloatArray,
        workload_A: FloatArray,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Gaussian workload mechanism.

        Args:
            sigma: Noise covariance Σ, shape (d, d). Must be PSD.
            workload_A: Workload matrix A, shape (m, d).
            epsilon: Privacy parameter ε > 0.
            delta: Privacy parameter δ ∈ (0, 1).
            metadata: Optional metadata dict.
            seed: Random seed for sampling.

        Raises:
            InvalidMechanismError: If sigma is not PSD or shapes mismatch.
            ConfigurationError: If privacy parameters are invalid.
        """
        self._sigma = np.asarray(sigma, dtype=np.float64)
        self._A = np.asarray(workload_A, dtype=np.float64)

        if self._sigma.ndim != 2 or self._sigma.shape[0] != self._sigma.shape[1]:
            raise InvalidMechanismError(
                f"sigma must be a square matrix, got shape {self._sigma.shape}",
                reason="non-square covariance",
            )

        d = self._sigma.shape[0]

        if self._A.ndim != 2:
            raise InvalidMechanismError(
                f"workload_A must be 2-D, got shape {self._A.shape}",
                reason="wrong dimensionality",
            )

        if self._A.shape[1] != d:
            raise InvalidMechanismError(
                f"workload_A columns ({self._A.shape[1]}) must match "
                f"sigma dimension ({d})",
                reason="dimension mismatch",
                expected_shape=(self._A.shape[0], d),
                actual_shape=self._A.shape,
            )

        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ConfigurationError(
                f"epsilon must be positive and finite, got {epsilon}",
                parameter="epsilon",
                value=epsilon,
            )
        if not (0.0 < delta < 1.0):
            raise ConfigurationError(
                f"delta must be in (0, 1), got {delta}",
                parameter="delta",
                value=delta,
            )

        self._epsilon = epsilon
        self._delta = delta
        self._metadata = metadata or {}
        self._rng = np.random.default_rng(seed)

        # Precompute answer covariance: A Σ A^T
        self._answer_cov = self._A @ self._sigma @ self._A.T

        # Precompute Cholesky of answer covariance for efficient sampling
        try:
            # Regularize slightly for numerical stability
            reg = np.eye(self.m) * 1e-12
            self._cholesky_L = np.linalg.cholesky(self._answer_cov + reg)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Answer covariance is not PSD; using eigendecomposition for sampling"
            )
            self._cholesky_L = None

    @property
    def sigma(self) -> FloatArray:
        """Noise covariance matrix Σ (read-only copy)."""
        return self._sigma.copy()

    @property
    def workload_A(self) -> FloatArray:
        """Workload matrix A (read-only copy)."""
        return self._A.copy()

    @property
    def epsilon(self) -> float:
        """Privacy parameter ε."""
        return self._epsilon

    @property
    def delta(self) -> float:
        """Privacy parameter δ."""
        return self._delta

    @property
    def m(self) -> int:
        """Number of queries in the workload."""
        return self._A.shape[0]

    @property
    def d(self) -> int:
        """Dimension of the data domain."""
        return self._A.shape[1]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Mechanism metadata."""
        return dict(self._metadata)

    # ----- Sampling -----

    def sample(
        self,
        true_data: FloatArray,
        rng: Optional[np.random.Generator] = None,
    ) -> FloatArray:
        """Sample noisy answers for a true data vector.

        Computes answer = A · x + z where z ~ N(0, A Σ A^T).

        Args:
            true_data: True data vector x of shape (d,).
            rng: Optional RNG override.

        Returns:
            Noisy answer vector of shape (m,).

        Raises:
            ConfigurationError: If true_data has wrong shape.
        """
        true_data = np.asarray(true_data, dtype=np.float64)
        if true_data.shape != (self.d,):
            raise ConfigurationError(
                f"true_data must have shape ({self.d},), got {true_data.shape}",
                parameter="true_data",
            )

        rng = rng or self._rng

        # True answers
        true_answers = self._A @ true_data

        # Add Gaussian noise
        noise = self._sample_noise(rng)
        return true_answers + noise

    def sample_batch(
        self,
        true_data_batch: FloatArray,
        rng: Optional[np.random.Generator] = None,
    ) -> FloatArray:
        """Sample noisy answers for a batch of data vectors.

        Args:
            true_data_batch: Batch of true data vectors, shape (batch, d).
            rng: Optional RNG override.

        Returns:
            Noisy answer batch of shape (batch, m).
        """
        true_data_batch = np.asarray(true_data_batch, dtype=np.float64)
        if true_data_batch.ndim == 1:
            true_data_batch = true_data_batch.reshape(1, -1)

        if true_data_batch.shape[1] != self.d:
            raise ConfigurationError(
                f"true_data_batch must have {self.d} columns, "
                f"got {true_data_batch.shape[1]}",
                parameter="true_data_batch",
            )

        rng = rng or self._rng
        batch_size = true_data_batch.shape[0]

        # True answers: (batch, m)
        true_answers = (self._A @ true_data_batch.T).T

        # Noise: (batch, m)
        noise = np.array([self._sample_noise(rng) for _ in range(batch_size)])

        return true_answers + noise

    def _sample_noise(self, rng: np.random.Generator) -> FloatArray:
        """Sample noise from N(0, A Σ A^T).

        Uses Cholesky decomposition for efficiency when available,
        falls back to eigendecomposition otherwise.

        Args:
            rng: Random number generator.

        Returns:
            Noise vector of shape (m,).
        """
        z = rng.standard_normal(self.m)

        if self._cholesky_L is not None:
            return self._cholesky_L @ z
        else:
            # Eigendecomposition fallback
            eigvals, eigvecs = np.linalg.eigh(self._answer_cov)
            eigvals = np.maximum(eigvals, 0.0)
            sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals))
            return sqrt_cov @ z

    # ----- Analytical MSE -----

    def mse_per_query(self) -> FloatArray:
        """Analytical MSE per query.

        For the i-th query, MSE_i = (A Σ A^T)_{ii} = diagonal entry of
        the answer covariance.

        Returns:
            Array of per-query MSE values, shape (m,).
        """
        return np.diag(self._answer_cov).copy()

    def total_mse(self) -> float:
        """Total MSE across all queries.

        Total MSE = tr(A Σ A^T) = Σ_i MSE_i.

        Returns:
            Total MSE.
        """
        return float(np.trace(self._answer_cov))

    def max_mse(self) -> float:
        """Maximum per-query MSE.

        Returns:
            The worst (largest) per-query MSE.
        """
        return float(np.max(self.mse_per_query()))

    # ----- Covariance access -----

    def covariance(self) -> FloatArray:
        """Return the noise covariance matrix in answer space.

        Returns A Σ A^T, which is the covariance of the noisy answers.

        Returns:
            Covariance matrix of shape (m, m).
        """
        return self._answer_cov.copy()

    def data_space_covariance(self) -> FloatArray:
        """Return the noise covariance in data space (Σ itself).

        Returns:
            Covariance matrix of shape (d, d).
        """
        return self._sigma.copy()

    # ----- Validity checking -----

    def is_valid(
        self,
        tol: float = 1e-6,
    ) -> Tuple[bool, List[str]]:
        """Check mechanism validity.

        Validates:
        1. Σ is positive semi-definite.
        2. Σ is symmetric.
        3. Privacy constraint: for Gaussian mechanism with sensitivity Δ,
           need σ ≥ Δ · √(2 ln(1.25/δ)) / ε.

        Args:
            tol: Numerical tolerance.

        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        issues: List[str] = []

        # Symmetry
        asym = float(np.max(np.abs(self._sigma - self._sigma.T)))
        if asym > tol:
            issues.append(f"Sigma is not symmetric (max asymmetry={asym:.2e})")

        # Positive semi-definiteness
        eigvals = np.linalg.eigvalsh(self._sigma)
        min_eigval = float(np.min(eigvals))
        if min_eigval < -tol:
            issues.append(f"Sigma is not PSD (min eigenvalue={min_eigval:.2e})")

        # Finiteness
        if not np.all(np.isfinite(self._sigma)):
            issues.append("Sigma contains non-finite values")

        # Privacy check (simplified: check minimum noise scale)
        if "sensitivity" in self._metadata:
            delta_f = self._metadata["sensitivity"]
            required_sigma = delta_f * math.sqrt(2.0 * math.log(1.25 / self._delta)) / self._epsilon
            min_noise_scale = math.sqrt(min_eigval) if min_eigval > 0 else 0.0
            if min_noise_scale < required_sigma - tol:
                issues.append(
                    f"Insufficient noise: min_scale={min_noise_scale:.4f} < "
                    f"required={required_sigma:.4f}"
                )

        return len(issues) == 0, issues

    # ----- Decomposition -----

    def decompose(self) -> Dict[str, Any]:
        """Eigendecomposition of the noise structure.

        Returns the eigenvalues and eigenvectors of both the data-space
        covariance Σ and the answer-space covariance A Σ A^T.

        Returns:
            Dict with keys:
                'sigma_eigenvalues': Eigenvalues of Σ.
                'sigma_eigenvectors': Eigenvectors of Σ.
                'answer_eigenvalues': Eigenvalues of A Σ A^T.
                'answer_eigenvectors': Eigenvectors of A Σ A^T.
                'effective_rank_sigma': Effective rank of Σ.
                'effective_rank_answer': Effective rank of A Σ A^T.
                'condition_number_sigma': Condition number of Σ.
        """
        sig_eigvals, sig_eigvecs = np.linalg.eigh(self._sigma)
        ans_eigvals, ans_eigvecs = np.linalg.eigh(self._answer_cov)

        # Effective rank: (Σ eigenvalues / max eigenvalue) where they
        # are above a threshold
        threshold = 1e-10
        sig_positive = sig_eigvals[sig_eigvals > threshold]
        ans_positive = ans_eigvals[ans_eigvals > threshold]

        eff_rank_sig = len(sig_positive)
        eff_rank_ans = len(ans_positive)

        max_sig = float(np.max(sig_eigvals)) if len(sig_eigvals) > 0 else 0.0
        min_sig_pos = float(np.min(sig_positive)) if len(sig_positive) > 0 else 1e-300
        cond = max_sig / min_sig_pos if min_sig_pos > 0 else float("inf")

        return {
            "sigma_eigenvalues": sig_eigvals,
            "sigma_eigenvectors": sig_eigvecs,
            "answer_eigenvalues": ans_eigvals,
            "answer_eigenvectors": ans_eigvecs,
            "effective_rank_sigma": eff_rank_sig,
            "effective_rank_answer": eff_rank_ans,
            "condition_number_sigma": cond,
        }

    # ----- Representation -----

    def __repr__(self) -> str:
        return (
            f"GaussianWorkloadMechanism(m={self.m}, d={self.d}, "
            f"ε={self._epsilon:.4f}, δ={self._delta:.2e}, "
            f"total_mse={self.total_mse():.4f})"
        )

    def __str__(self) -> str:
        valid, issues = self.is_valid(check_privacy=False) if hasattr(self, 'is_valid') else (True, [])
        status = "valid" if valid else f"{len(issues)} issues"
        return (
            f"GaussianWorkloadMechanism(m={self.m}, d={self.d}, "
            f"ε={self._epsilon}, δ={self._delta}, {status})"
        )
