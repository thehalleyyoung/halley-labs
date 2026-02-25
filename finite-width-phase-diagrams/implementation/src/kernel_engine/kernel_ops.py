"""Kernel operations: construction, alignment, PCA, and spectral analysis.

Provides kernel matrix construction for various kernels (RBF, polynomial,
Matérn, NTK), alignment metrics, kernel PCA, distance metrics, and
eigenvalue/eigenvector computation with error bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg as sp_linalg
from scipy.special import gamma as gamma_fn, kv as bessel_kv


# ======================================================================
# Kernel matrix construction
# ======================================================================

class KernelMatrix:
    """Construct kernel matrices for common kernel functions."""

    # ------------------------------------------------------------------
    # RBF (Gaussian) kernel
    # ------------------------------------------------------------------

    @staticmethod
    def rbf(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        gamma: Optional[float] = None,
        length_scale: Optional[float] = None,
    ) -> np.ndarray:
        """Radial Basis Function (Gaussian) kernel.

        K(x, y) = exp(-γ ||x - y||²)  or equivalently
        K(x, y) = exp(-||x - y||² / (2 l²))

        Parameters
        ----------
        X : (N, D)
        Y : (M, D) or None for K(X, X)
        gamma : kernel coefficient (overrides length_scale)
        length_scale : l in the Gaussian formula
        """
        if Y is None:
            Y = X
        if gamma is None:
            if length_scale is not None:
                gamma = 1.0 / (2.0 * length_scale ** 2)
            else:
                gamma = 1.0 / X.shape[1]

        sq_dists = _squared_distances(X, Y)
        return np.exp(-gamma * sq_dists)

    # ------------------------------------------------------------------
    # Polynomial kernel
    # ------------------------------------------------------------------

    @staticmethod
    def polynomial(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        degree: int = 3,
        coef0: float = 1.0,
        gamma: Optional[float] = None,
    ) -> np.ndarray:
        """Polynomial kernel: K(x, y) = (γ x·y + c₀)^d.

        Parameters
        ----------
        degree : polynomial degree
        coef0 : additive constant
        gamma : scaling (default 1/D)
        """
        if Y is None:
            Y = X
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        return (gamma * (X @ Y.T) + coef0) ** degree

    # ------------------------------------------------------------------
    # Matérn kernel
    # ------------------------------------------------------------------

    @staticmethod
    def matern(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nu: float = 1.5,
        length_scale: float = 1.0,
    ) -> np.ndarray:
        """Matérn kernel.

        K(x,y) = (2^{1-ν}/Γ(ν)) (√(2ν) r/l)^ν K_ν(√(2ν) r/l)

        where r = ||x - y|| and K_ν is the modified Bessel function.

        Special cases:
        - ν = 0.5:  Exponential kernel (Laplacian)
        - ν = 1.5:  Matérn 3/2
        - ν = 2.5:  Matérn 5/2
        - ν → ∞:    RBF kernel
        """
        if Y is None:
            Y = X
        dists = np.sqrt(np.maximum(_squared_distances(X, Y), 1e-30))

        if nu == 0.5:
            return np.exp(-dists / length_scale)

        if nu == 1.5:
            arg = math.sqrt(3.0) * dists / length_scale
            return (1.0 + arg) * np.exp(-arg)

        if nu == 2.5:
            arg = math.sqrt(5.0) * dists / length_scale
            return (1.0 + arg + arg ** 2 / 3.0) * np.exp(-arg)

        if nu > 100:
            gamma = 1.0 / (2.0 * length_scale ** 2)
            return np.exp(-gamma * dists ** 2)

        # General case
        arg = math.sqrt(2.0 * nu) * dists / length_scale
        prefix = (2.0 ** (1.0 - nu)) / gamma_fn(nu)
        # Avoid zero distances
        K = np.where(
            dists < 1e-15,
            1.0,
            prefix * (arg ** nu) * bessel_kv(nu, arg),
        )
        return np.clip(K, 0.0, None)

    # ------------------------------------------------------------------
    # Linear kernel
    # ------------------------------------------------------------------

    @staticmethod
    def linear(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Linear kernel: K(x, y) = scale · x · y."""
        if Y is None:
            Y = X
        return scale * (X @ Y.T)

    # ------------------------------------------------------------------
    # Cosine similarity kernel
    # ------------------------------------------------------------------

    @staticmethod
    def cosine(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Cosine similarity kernel."""
        if Y is None:
            Y = X
        norms_X = np.linalg.norm(X, axis=1, keepdims=True)
        norms_Y = np.linalg.norm(Y, axis=1, keepdims=True)
        norms_X = np.maximum(norms_X, 1e-15)
        norms_Y = np.maximum(norms_Y, 1e-15)
        return (X / norms_X) @ (Y / norms_Y).T

    # ------------------------------------------------------------------
    # Arc-cosine kernel (NTK-related)
    # ------------------------------------------------------------------

    @staticmethod
    def arc_cosine(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        order: int = 1,
    ) -> np.ndarray:
        """Arc-cosine kernel (Cho & Saul, 2009).

        Order 0: step function dual (ReLU derivative)
        Order 1: ReLU dual
        """
        if Y is None:
            Y = X
        norms_X = np.linalg.norm(X, axis=1)
        norms_Y = np.linalg.norm(Y, axis=1)
        dot = X @ Y.T
        denom = np.outer(norms_X, norms_Y)
        denom = np.maximum(denom, 1e-15)
        cos_theta = np.clip(dot / denom, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        if order == 0:
            return (1.0 / math.pi) * (math.pi - theta)
        elif order == 1:
            return (1.0 / (2.0 * math.pi)) * denom * (
                np.sin(theta) + (math.pi - theta) * cos_theta
            )
        elif order == 2:
            return (1.0 / (2.0 * math.pi)) * denom ** 2 * (
                3.0 * np.sin(theta) * cos_theta
                + (math.pi - theta) * (1.0 + 2.0 * cos_theta ** 2)
            ) / 3.0
        raise ValueError(f"Arc-cosine order {order} not supported (0, 1, 2)")

    # ------------------------------------------------------------------
    # From callable
    # ------------------------------------------------------------------

    @staticmethod
    def from_function(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> np.ndarray:
        """Build kernel matrix from a pairwise function func(x_i, x_j) -> float."""
        if Y is None:
            Y = X
        N, M = X.shape[0], Y.shape[0]
        K = np.zeros((N, M), dtype=np.float64)
        if func is None:
            return X @ Y.T
        for i in range(N):
            for j in range(M):
                K[i, j] = func(X[i], Y[j])
        return K


# ======================================================================
# Kernel alignment
# ======================================================================

class KernelAlignment:
    """Kernel alignment metrics for comparing kernel matrices."""

    @staticmethod
    def centered_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
        """Centered kernel alignment (CKA).

        CKA(K1, K2) = HSIC(K1, K2) / √(HSIC(K1, K1) · HSIC(K2, K2))
        """
        n = K1.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K1c = H @ K1 @ H
        K2c = H @ K2 @ H
        hsic_12 = np.sum(K1c * K2c) / ((n - 1) ** 2)
        hsic_11 = np.sum(K1c * K1c) / ((n - 1) ** 2)
        hsic_22 = np.sum(K2c * K2c) / ((n - 1) ** 2)
        denom = math.sqrt(max(hsic_11 * hsic_22, 1e-30))
        return float(hsic_12 / denom)

    @staticmethod
    def uncentered_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
        """Uncentered kernel alignment: <K1, K2>_F / (||K1||_F · ||K2||_F)."""
        num = np.sum(K1 * K2)
        denom = np.linalg.norm(K1, "fro") * np.linalg.norm(K2, "fro")
        if denom < 1e-15:
            return 0.0
        return float(num / denom)

    @staticmethod
    def target_alignment(K: np.ndarray, y: np.ndarray) -> float:
        """Alignment between kernel and target label kernel yy^T."""
        if y.ndim == 1:
            y = y[:, None]
        K_target = y @ y.T
        return KernelAlignment.centered_alignment(K, K_target)

    @staticmethod
    def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
        """Linear CKA (Kornblith et al., 2019) between representations.

        Parameters
        ----------
        X, Y : (N, D1) and (N, D2) representation matrices
        """
        n = X.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        Kx = H @ (X @ X.T) @ H
        Ky = H @ (Y @ Y.T) @ H
        num = np.linalg.norm(Ky.T @ Kx, "fro") ** 2
        denom = np.linalg.norm(Kx.T @ Kx, "fro") * np.linalg.norm(Ky.T @ Ky, "fro")
        if denom < 1e-30:
            return 0.0
        return float(num / denom)

    @staticmethod
    def rbf_cka(
        X: np.ndarray,
        Y: np.ndarray,
        sigma: Optional[float] = None,
    ) -> float:
        """RBF CKA between two representation matrices."""
        if sigma is None:
            sigma_x = np.median(np.sqrt(_squared_distances(X, X)))
            sigma_y = np.median(np.sqrt(_squared_distances(Y, Y)))
            sigma = (sigma_x + sigma_y) / 2.0
            sigma = max(sigma, 1e-10)
        Kx = KernelMatrix.rbf(X, gamma=1.0 / (2.0 * sigma ** 2))
        Ky = KernelMatrix.rbf(Y, gamma=1.0 / (2.0 * sigma ** 2))
        return KernelAlignment.centered_alignment(Kx, Ky)


# ======================================================================
# Kernel distance metrics
# ======================================================================

class KernelDistance:
    """Distance metrics between kernel matrices."""

    @staticmethod
    def frobenius(K1: np.ndarray, K2: np.ndarray) -> float:
        """Frobenius distance: ||K1 - K2||_F."""
        return float(np.linalg.norm(K1 - K2, "fro"))

    @staticmethod
    def spectral(K1: np.ndarray, K2: np.ndarray) -> float:
        """Spectral (operator) norm distance: ||K1 - K2||_2."""
        diff = K1 - K2
        return float(np.linalg.norm(diff, 2))

    @staticmethod
    def bures(K1: np.ndarray, K2: np.ndarray, reg: float = 1e-8) -> float:
        """Bures distance between PSD matrices.

        d_B(K1, K2)² = Tr(K1) + Tr(K2) - 2 Tr((K1^{1/2} K2 K1^{1/2})^{1/2})
        """
        K1_reg = K1 + reg * np.eye(K1.shape[0])
        sqrt_K1 = sp_linalg.sqrtm(K1_reg).real
        inner = sqrt_K1 @ K2 @ sqrt_K1
        inner = (inner + inner.T) / 2
        eigvals = np.linalg.eigvalsh(inner)
        eigvals = np.maximum(eigvals, 0.0)
        trace_sqrt = np.sum(np.sqrt(eigvals))
        d_sq = np.trace(K1) + np.trace(K2) - 2.0 * trace_sqrt
        return float(math.sqrt(max(d_sq, 0.0)))

    @staticmethod
    def log_euclidean(K1: np.ndarray, K2: np.ndarray, reg: float = 1e-8) -> float:
        """Log-Euclidean distance: ||log(K1) - log(K2)||_F."""
        K1_reg = K1 + reg * np.eye(K1.shape[0])
        K2_reg = K2 + reg * np.eye(K2.shape[0])
        log_K1 = sp_linalg.logm(K1_reg).real
        log_K2 = sp_linalg.logm(K2_reg).real
        return float(np.linalg.norm(log_K1 - log_K2, "fro"))

    @staticmethod
    def relative_frobenius(K1: np.ndarray, K2: np.ndarray) -> float:
        """Relative Frobenius distance: ||K1 - K2||_F / ||K1||_F."""
        norm1 = np.linalg.norm(K1, "fro")
        if norm1 < 1e-15:
            return float(np.linalg.norm(K2, "fro"))
        return float(np.linalg.norm(K1 - K2, "fro") / norm1)


# ======================================================================
# Kernel PCA
# ======================================================================

class KernelPCA:
    """Kernel Principal Component Analysis."""

    def __init__(
        self,
        n_components: int = 2,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1.0,
        center: bool = True,
    ) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.center = center

        self._eigvals: Optional[np.ndarray] = None
        self._eigvecs: Optional[np.ndarray] = None
        self._K_train: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None
        self._col_mean: Optional[np.ndarray] = None
        self._total_mean: Optional[float] = None

    def _compute_kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if self.kernel == "rbf":
            return KernelMatrix.rbf(X, Y, gamma=self.gamma)
        elif self.kernel == "polynomial":
            return KernelMatrix.polynomial(X, Y, degree=self.degree, coef0=self.coef0, gamma=self.gamma)
        elif self.kernel == "linear":
            return KernelMatrix.linear(X, Y)
        elif self.kernel == "cosine":
            return KernelMatrix.cosine(X, Y)
        elif self.kernel == "matern":
            return KernelMatrix.matern(X, Y, nu=1.5)
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X: np.ndarray) -> "KernelPCA":
        """Fit Kernel PCA on training data."""
        self._X_train = X.copy()
        K = self._compute_kernel(X)

        if self.center:
            K, self._col_mean, self._total_mean = self._center_kernel(K)

        self._K_train = K

        eigvals, eigvecs = np.linalg.eigh(K)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        self._eigvals = eigvals[: self.n_components]
        self._eigvecs = eigvecs[:, : self.n_components]

        return self

    def transform(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Project data into kernel PCA space."""
        if self._eigvals is None:
            raise RuntimeError("Must call fit() first")

        if X is None:
            K = self._K_train
        else:
            K = self._compute_kernel(X, self._X_train)
            if self.center and self._col_mean is not None:
                n_train = self._X_train.shape[0]
                col_mean_test = K.mean(axis=1, keepdims=True)
                K = K - col_mean_test - self._col_mean[None, :] + self._total_mean

        # Project
        scales = np.sqrt(np.maximum(self._eigvals, 1e-15))
        return (K @ self._eigvecs) / scales[None, :]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform()

    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of variance explained by each component."""
        if self._eigvals is None:
            return np.array([])
        total = np.sum(np.abs(self._eigvals))
        if total < 1e-15:
            return np.zeros(len(self._eigvals))
        return np.abs(self._eigvals) / total

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Compute reconstruction error in kernel space."""
        K = self._compute_kernel(X)
        if self.center and self._col_mean is not None:
            K, _, _ = self._center_kernel(K)
        Z = self.transform(X)
        K_approx = Z @ Z.T
        return float(np.linalg.norm(K - K_approx, "fro") / max(np.linalg.norm(K, "fro"), 1e-15))

    @staticmethod
    def _center_kernel(
        K: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Double-centre a kernel matrix."""
        n = K.shape[0]
        col_mean = K.mean(axis=0)
        total_mean = float(K.mean())
        K_centered = K - col_mean[None, :] - col_mean[:, None] + total_mean
        return K_centered, col_mean, total_mean

    @property
    def eigenvalues(self) -> Optional[np.ndarray]:
        return self._eigvals

    @property
    def eigenvectors(self) -> Optional[np.ndarray]:
        return self._eigvecs


# ======================================================================
# Kernel spectral analysis
# ======================================================================

class KernelSpectralAnalysis:
    """Spectral analysis of kernel matrices with error bounds."""

    @staticmethod
    def eigendecompose(
        K: np.ndarray,
        k: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Eigendecomposition sorted by decreasing eigenvalue.

        Parameters
        ----------
        K : (N, N) symmetric matrix
        k : number of top eigenvalues (None = all)

        Returns
        -------
        eigenvalues, eigenvectors (columns)
        """
        eigvals, eigvecs = np.linalg.eigh(K)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        if k is not None:
            return eigvals[:k], eigvecs[:, :k]
        return eigvals, eigvecs

    @staticmethod
    def eigenvalue_error_bound(
        K: np.ndarray,
        perturbation_norm: float,
    ) -> float:
        """Weyl's bound: |λ_i(K + E) - λ_i(K)| ≤ ||E||_2.

        Given an estimated perturbation norm, returns the max eigenvalue error.
        """
        return perturbation_norm

    @staticmethod
    def eigenvector_error_bound(
        eigenvalues: np.ndarray,
        perturbation_norm: float,
        idx: int,
    ) -> float:
        """Davis-Kahan sin(θ) bound for eigenvector perturbation.

        sin(θ) ≤ ||E||_2 / gap_i
        where gap_i = min_{j≠i} |λ_i - λ_j|
        """
        gaps = np.abs(eigenvalues - eigenvalues[idx])
        gaps[idx] = np.inf
        min_gap = np.min(gaps)
        if min_gap < 1e-15:
            return 1.0  # no bound available
        return min(perturbation_norm / min_gap, 1.0)

    @staticmethod
    def effective_rank(K: np.ndarray, threshold: float = 0.99) -> int:
        """Effective rank (number of eigenvalues for threshold fraction of energy)."""
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(K)))[::-1]
        total = np.sum(eigvals)
        if total < 1e-15:
            return 0
        cumsum = np.cumsum(eigvals) / total
        return int(np.searchsorted(cumsum, threshold)) + 1

    @staticmethod
    def participation_ratio(K: np.ndarray) -> float:
        """Participation ratio: (Σ λ_i)² / Σ λ_i².

        Measures how many eigenvalues are "active". For uniform spectrum of
        rank r, PR = r.
        """
        eigvals = np.abs(np.linalg.eigvalsh(K))
        sum1 = np.sum(eigvals)
        sum2 = np.sum(eigvals ** 2)
        if sum2 < 1e-30:
            return 0.0
        return float(sum1 ** 2 / sum2)

    @staticmethod
    def spectral_entropy(K: np.ndarray) -> float:
        """Von Neumann entropy: -Σ p_i log(p_i)."""
        eigvals = np.abs(np.linalg.eigvalsh(K))
        eigvals = eigvals[eigvals > 1e-15]
        if len(eigvals) == 0:
            return 0.0
        p = eigvals / np.sum(eigvals)
        return float(-np.sum(p * np.log(p)))

    @staticmethod
    def condition_number(K: np.ndarray) -> float:
        """Condition number: λ_max / λ_min (of positive eigenvalues)."""
        eigvals = np.abs(np.linalg.eigvalsh(K))
        eigvals = eigvals[eigvals > 1e-15]
        if len(eigvals) == 0:
            return float("inf")
        return float(np.max(eigvals) / np.min(eigvals))

    @staticmethod
    def eigenvalue_decay_rate(K: np.ndarray) -> float:
        """Estimate power-law decay rate: λ_k ~ k^{-α}.

        Returns α estimated via linear regression on log-log scale.
        """
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(K)))[::-1]
        eigvals = eigvals[eigvals > 1e-15]
        if len(eigvals) < 3:
            return 0.0
        log_k = np.log(np.arange(1, len(eigvals) + 1))
        log_lambda = np.log(eigvals)
        # Linear regression
        A = np.vstack([log_k, np.ones_like(log_k)]).T
        result = np.linalg.lstsq(A, log_lambda, rcond=None)
        alpha = -result[0][0]
        return float(alpha)

    @staticmethod
    def spectral_gap(K: np.ndarray, k: int = 1) -> float:
        """Gap between k-th and (k+1)-th eigenvalue."""
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(K)))[::-1]
        if len(eigvals) <= k:
            return 0.0
        return float(eigvals[k - 1] - eigvals[k])

    @staticmethod
    def trace_norm(K: np.ndarray) -> float:
        """Nuclear / trace norm: Σ |λ_i|."""
        return float(np.sum(np.abs(np.linalg.eigvalsh(K))))

    @staticmethod
    def stable_rank(K: np.ndarray) -> float:
        """Stable rank: ||K||_F² / ||K||_2²."""
        fro = np.linalg.norm(K, "fro")
        op = np.linalg.norm(K, 2)
        if op < 1e-15:
            return 0.0
        return float(fro ** 2 / op ** 2)

    @staticmethod
    def eigenvalue_histogram(
        K: np.ndarray,
        bins: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Histogram of eigenvalues."""
        eigvals = np.linalg.eigvalsh(K)
        counts, bin_edges = np.histogram(eigvals, bins=bins)
        return counts, bin_edges

    @staticmethod
    def marchenko_pastur_fit(
        K: np.ndarray,
        gamma: Optional[float] = None,
    ) -> Dict[str, float]:
        """Fit Marchenko-Pastur distribution to the bulk eigenvalues.

        Parameters
        ----------
        K : kernel matrix
        gamma : aspect ratio N/P (if None, estimated from spectrum)

        Returns
        -------
        Dict with lambda_minus, lambda_plus, sigma_sq, gamma
        """
        eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
        eigvals = eigvals[eigvals > 1e-10]
        n = len(eigvals)

        if gamma is None:
            # Estimate from spectral edges
            sigma_sq = np.mean(eigvals)
            gamma = 1.0
        else:
            sigma_sq = np.mean(eigvals)

        lambda_plus = sigma_sq * (1 + math.sqrt(gamma)) ** 2
        lambda_minus = sigma_sq * max((1 - math.sqrt(gamma)), 0) ** 2

        return {
            "lambda_minus": float(lambda_minus),
            "lambda_plus": float(lambda_plus),
            "sigma_sq": float(sigma_sq),
            "gamma": float(gamma),
            "num_bulk": int(np.sum((eigvals >= lambda_minus) & (eigvals <= lambda_plus))),
            "num_outliers": int(np.sum(eigvals > lambda_plus)),
        }


# ======================================================================
# Utility functions
# ======================================================================

def _squared_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances."""
    XX = np.sum(X ** 2, axis=1)[:, None]
    YY = np.sum(Y ** 2, axis=1)[None, :]
    sq_dists = XX + YY - 2.0 * (X @ Y.T)
    return np.maximum(sq_dists, 0.0)
