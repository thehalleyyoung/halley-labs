"""
Hilbert-Schmidt Independence Criterion (HSIC) for kernel-based
independence and conditional independence testing.

Implements the biased and unbiased HSIC estimators, multiple kernel
functions, permutation and gamma-approximation p-values, conditional
HSIC via residualisation, median-heuristic and cross-validated bandwidth
selection, block permutations for time-series data, and an efficient
incomplete-Cholesky approximation for large sample sizes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import cho_factor, cho_solve


# ====================================================================
# Kernel functions
# ====================================================================

class Kernel(ABC):
    """Abstract kernel function k(x, y) → ℝ."""

    @abstractmethod
    def compute(self, X: NDArray, Y: Optional[NDArray] = None) -> NDArray:
        """Return the kernel (Gram) matrix K[i,j] = k(X_i, Y_j)."""

    @abstractmethod
    def name(self) -> str:
        ...


class GaussianKernel(Kernel):
    """Gaussian / RBF kernel  k(x,y) = exp(-||x-y||² / (2σ²))."""

    def __init__(self, bandwidth: Optional[float] = None) -> None:
        self.bandwidth = bandwidth

    def compute(self, X: NDArray, Y: Optional[NDArray] = None) -> NDArray:
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)
        sq = cdist(X, Y, "sqeuclidean")
        bw = self.bandwidth if self.bandwidth is not None else median_bandwidth(X)
        return np.exp(-sq / (2.0 * bw ** 2))

    def name(self) -> str:
        return "gaussian"


class PolynomialKernel(Kernel):
    """Polynomial kernel  k(x,y) = (⟨x,y⟩ + c)^d."""

    def __init__(self, degree: int = 3, coef0: float = 1.0) -> None:
        self.degree = degree
        self.coef0 = coef0

    def compute(self, X: NDArray, Y: Optional[NDArray] = None) -> NDArray:
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)
        return (X @ Y.T + self.coef0) ** self.degree

    def name(self) -> str:
        return "polynomial"


class LinearKernel(Kernel):
    """Linear kernel  k(x,y) = ⟨x,y⟩."""

    def compute(self, X: NDArray, Y: Optional[NDArray] = None) -> NDArray:
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)
        return X @ Y.T

    def name(self) -> str:
        return "linear"


# ====================================================================
# Bandwidth selection
# ====================================================================

def median_bandwidth(X: NDArray) -> float:
    """Median-heuristic bandwidth for Gaussian kernel.

    σ = sqrt(median pairwise squared distance / 2).
    """
    X = np.atleast_2d(X)
    if X.shape[0] > 2000:
        idx = np.random.choice(X.shape[0], 2000, replace=False)
        X = X[idx]
    pw = pdist(X, "sqeuclidean")
    med = np.median(pw)
    return float(np.sqrt(med / 2.0)) if med > 0 else 1.0


def cross_validate_bandwidth(
    X: NDArray,
    Y: NDArray,
    bandwidths: Optional[Sequence[float]] = None,
    n_folds: int = 5,
    seed: Optional[int] = None,
) -> float:
    """Select Gaussian kernel bandwidth by leave-one-out cross-validated
    kernel-ridge regression of Y on X, minimising MSE.

    Parameters
    ----------
    X, Y : ndarray
        Data arrays.
    bandwidths : sequence of float, optional
        Candidate bandwidths; defaults to a log-spaced grid around the
        median heuristic.
    n_folds : int
        Number of CV folds.

    Returns
    -------
    float
        Best bandwidth.
    """
    X = np.atleast_2d(X)
    Y = np.asarray(Y).ravel()
    med = median_bandwidth(X)
    if bandwidths is None:
        bandwidths = med * np.logspace(-1, 1, 9)

    rng = np.random.default_rng(seed)
    n = len(Y)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    best_bw = med
    best_mse = np.inf

    reg = 1e-3
    for bw in bandwidths:
        mse_sum = 0.0
        for fold_idx in range(n_folds):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate(
                [folds[j] for j in range(n_folds) if j != fold_idx]
            )
            Ktr = np.exp(
                -cdist(X[train_idx], X[train_idx], "sqeuclidean")
                / (2 * bw ** 2)
            )
            alpha = np.linalg.solve(
                Ktr + reg * np.eye(len(train_idx)), Y[train_idx]
            )
            Kte = np.exp(
                -cdist(X[test_idx], X[train_idx], "sqeuclidean")
                / (2 * bw ** 2)
            )
            pred = Kte @ alpha
            mse_sum += np.mean((Y[test_idx] - pred) ** 2)
        avg_mse = mse_sum / n_folds
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_bw = bw

    return float(best_bw)


# ====================================================================
# Centering helper
# ====================================================================

def _center_kernel(K: NDArray) -> NDArray:
    """Double-centre a kernel matrix: H K H where H = I - 11ᵀ/n."""
    n = K.shape[0]
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    total_mean = K.mean()
    return K - row_mean - col_mean + total_mean


# ====================================================================
# Incomplete Cholesky approximation
# ====================================================================

def incomplete_cholesky(
    K: NDArray, max_rank: int = 50, tol: float = 1e-6
) -> NDArray:
    """Pivoted incomplete Cholesky decomposition of a kernel matrix.

    Returns an n × m lower-triangular factor L such that K ≈ L Lᵀ,
    where m ≤ max_rank.
    """
    n = K.shape[0]
    m = min(max_rank, n)
    diag = np.diag(K).copy()
    perm = np.arange(n)
    L = np.zeros((n, m))

    for j in range(m):
        # Select pivot: largest remaining diagonal
        idx = np.argmax(diag[j:]) + j
        if diag[idx] < tol:
            L = L[:, :j]
            break
        # Swap
        perm[[j, idx]] = perm[[idx, j]]
        diag[[j, idx]] = diag[[idx, j]]
        L[[j, idx], :] = L[[idx, j], :]

        L[j, j] = np.sqrt(diag[j])
        # Fill column j
        if j > 0:
            col = (K[perm[j + 1:], perm[j]] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
        else:
            col = K[perm[j + 1:], perm[j]] / L[j, j]
        L[j + 1:, j] = col
        diag[j + 1:] -= col ** 2
        diag[j + 1:] = np.maximum(diag[j + 1:], 0.0)

    # Undo permutation
    inv_perm = np.argsort(perm)
    return L[inv_perm]


# ====================================================================
# HSIC estimators
# ====================================================================

def _biased_hsic(Kx: NDArray, Ky: NDArray) -> float:
    """Biased HSIC estimator: (1/n²) trace(Kx H Ky H)."""
    n = Kx.shape[0]
    Hkx = _center_kernel(Kx)
    Hky = _center_kernel(Ky)
    return float(np.trace(Hkx @ Hky) / (n * n))


def _unbiased_hsic(Kx: NDArray, Ky: NDArray) -> float:
    """Unbiased HSIC estimator (Song et al. 2012).

    Computes the U-statistic:
        HSIC_u = [1/(n(n-3))] [tr(K̃x K̃y) + (1ᵀK̃x1)(1ᵀK̃y1)/((n-1)(n-2))
                                - 2/(n-2) 1ᵀ(K̃x K̃y)1]
    where K̃ has zeroed diagonal.
    """
    n = Kx.shape[0]
    if n < 4:
        return 0.0
    Kx_tilde = Kx.copy()
    Ky_tilde = Ky.copy()
    np.fill_diagonal(Kx_tilde, 0.0)
    np.fill_diagonal(Ky_tilde, 0.0)

    term1 = np.sum(Kx_tilde * Ky_tilde)  # trace of element-wise product
    term2 = np.sum(Kx_tilde) * np.sum(Ky_tilde) / ((n - 1) * (n - 2))
    term3 = 2.0 * np.sum(Kx_tilde @ Ky_tilde) / (n - 2)

    return float((term1 + term2 - term3) / (n * (n - 3)))


def _hsic_lowrank(Lx: NDArray, Ly: NDArray) -> float:
    """HSIC via low-rank (incomplete Cholesky) approximations.

    Lx, Ly are n × m factors such that K ≈ L Lᵀ.
    HSIC ≈ (1/n²) || Hx Lx ᵀ Ly Hy ||²_F  (biased).
    """
    n = Lx.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    HLx = H @ Lx
    HLy = H @ Ly
    M = HLx.T @ HLy
    return float(np.sum(M ** 2) / (n * n))


# ====================================================================
# Null distribution approximations
# ====================================================================

def _gamma_params_from_hsic(Kx: NDArray, Ky: NDArray) -> Tuple[float, float]:
    """Estimate Gamma distribution parameters for the null distribution
    of the biased HSIC test statistic (Gretton et al. 2005).

    Returns (shape, scale) of the fitted Gamma.
    """
    n = Kx.shape[0]
    Hkx = _center_kernel(Kx)
    Hky = _center_kernel(Ky)

    # Mean under H0
    mu_x = np.trace(Kx) / n
    mu_y = np.trace(Ky) / n
    mean_hsic = (1.0 + mu_x * mu_y - mu_x - mu_y) / n

    # Variance approximation (Gretton et al.)
    B = (Hkx * Hky)  # element-wise
    var_hsic = 2.0 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3))
    var_hsic *= np.sum(B @ B)

    if var_hsic <= 0 or mean_hsic <= 0:
        return 1.0, 1.0

    # Method of moments
    scale = var_hsic / mean_hsic
    shape = mean_hsic / scale
    return max(shape, 1e-8), max(scale, 1e-12)


# ====================================================================
# Block permutation for time series
# ====================================================================

def _block_permutation(
    n: int, block_size: int, rng: np.random.Generator
) -> NDArray:
    """Generate a block-wise permutation of indices 0..n-1."""
    n_blocks = int(np.ceil(n / block_size))
    blocks = [
        np.arange(i * block_size, min((i + 1) * block_size, n))
        for i in range(n_blocks)
    ]
    perm_blocks = rng.permutation(n_blocks)
    return np.concatenate([blocks[i] for i in perm_blocks])[:n]


# ====================================================================
# Main HSIC class
# ====================================================================

@dataclass
class HSICResult:
    """Container for HSIC test results."""
    statistic: float
    p_value: float
    threshold: float = 0.0
    method: str = ""


class HSIC:
    """Kernel independence test using the Hilbert-Schmidt Independence
    Criterion.

    Parameters
    ----------
    kernel_x : Kernel
        Kernel for X; defaults to GaussianKernel(median heuristic).
    kernel_y : Kernel
        Kernel for Y.
    unbiased : bool
        Use unbiased (True) or biased estimator.
    n_permutations : int
        Number of permutations for the permutation test; set to 0 to use
        the Gamma approximation instead.
    block_size : int or None
        If given, use block permutation (for serially dependent data).
    use_incomplete_cholesky : bool
        Approximate kernels with incomplete Cholesky for large n.
    max_rank : int
        Maximum rank for the incomplete Cholesky factor.
    """

    def __init__(
        self,
        kernel_x: Optional[Kernel] = None,
        kernel_y: Optional[Kernel] = None,
        unbiased: bool = True,
        n_permutations: int = 500,
        block_size: Optional[int] = None,
        use_incomplete_cholesky: bool = False,
        max_rank: int = 100,
        alpha: float = 0.05,
    ) -> None:
        self.kernel_x = kernel_x or GaussianKernel()
        self.kernel_y = kernel_y or GaussianKernel()
        self.unbiased = unbiased
        self.n_permutations = n_permutations
        self.block_size = block_size
        self.use_incomplete_cholesky = use_incomplete_cholesky
        self.max_rank = max_rank
        self.alpha = alpha

    def test(
        self,
        X: NDArray,
        Y: NDArray,
        seed: Optional[int] = None,
    ) -> HSICResult:
        """Run the HSIC independence test.

        Parameters
        ----------
        X : ndarray, shape (n,) or (n, d_x)
        Y : ndarray, shape (n,) or (n, d_y)

        Returns
        -------
        HSICResult
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        if X.shape[0] == 1:
            X = X.T
        if Y.shape[0] == 1:
            Y = Y.T
        n = X.shape[0]

        Kx = self.kernel_x.compute(X)
        Ky = self.kernel_y.compute(Y)

        if self.use_incomplete_cholesky and n > 500:
            return self._test_lowrank(Kx, Ky, X, Y, n, seed)

        hsic_fn = _unbiased_hsic if self.unbiased else _biased_hsic
        stat = hsic_fn(Kx, Ky)

        if self.n_permutations > 0:
            return self._permutation_test(Kx, Ky, stat, n, hsic_fn, seed)
        else:
            return self._gamma_test(Kx, Ky, stat, n)

    def _permutation_test(
        self,
        Kx: NDArray,
        Ky: NDArray,
        stat: float,
        n: int,
        hsic_fn,
        seed: Optional[int],
    ) -> HSICResult:
        rng = np.random.default_rng(seed)
        null_stats = np.empty(self.n_permutations)
        for i in range(self.n_permutations):
            if self.block_size is not None:
                perm = _block_permutation(n, self.block_size, rng)
            else:
                perm = rng.permutation(n)
            Ky_perm = Ky[np.ix_(perm, perm)]
            null_stats[i] = hsic_fn(Kx, Ky_perm)

        p_value = float(np.mean(null_stats >= stat))
        threshold = float(np.percentile(null_stats, 100 * (1 - self.alpha)))
        return HSICResult(
            statistic=stat,
            p_value=p_value,
            threshold=threshold,
            method="permutation",
        )

    def _gamma_test(
        self, Kx: NDArray, Ky: NDArray, stat: float, n: int
    ) -> HSICResult:
        shape, scale = _gamma_params_from_hsic(Kx, Ky)
        p_value = float(1.0 - stats.gamma.cdf(stat * n, a=shape, scale=scale / n))
        threshold = float(
            stats.gamma.ppf(1 - self.alpha, a=shape, scale=scale / n) / n
        )
        return HSICResult(
            statistic=stat,
            p_value=p_value,
            threshold=threshold,
            method="gamma",
        )

    def _test_lowrank(
        self,
        Kx: NDArray,
        Ky: NDArray,
        X: NDArray,
        Y: NDArray,
        n: int,
        seed: Optional[int],
    ) -> HSICResult:
        Lx = incomplete_cholesky(Kx, max_rank=self.max_rank)
        Ly = incomplete_cholesky(Ky, max_rank=self.max_rank)
        stat = _hsic_lowrank(Lx, Ly)

        rng = np.random.default_rng(seed)
        null_stats = np.empty(self.n_permutations)
        for i in range(self.n_permutations):
            perm = rng.permutation(n)
            null_stats[i] = _hsic_lowrank(Lx, Ly[perm])

        p_value = float(np.mean(null_stats >= stat))
        threshold = float(np.percentile(null_stats, 100 * (1 - self.alpha)))
        return HSICResult(
            statistic=stat,
            p_value=p_value,
            threshold=threshold,
            method="lowrank_permutation",
        )

    def statistic(self, X: NDArray, Y: NDArray) -> float:
        """Compute the HSIC statistic without a full test."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        if X.shape[0] == 1:
            X = X.T
        if Y.shape[0] == 1:
            Y = Y.T
        Kx = self.kernel_x.compute(X)
        Ky = self.kernel_y.compute(Y)
        fn = _unbiased_hsic if self.unbiased else _biased_hsic
        return fn(Kx, Ky)


# ====================================================================
# Conditional HSIC (via residualisation)
# ====================================================================

class ConditionalHSIC:
    """Test X ⊥ Y | Z via residualisation (kernel partial correlation).

    Strategy:
        1. Regress X on Z using kernel ridge regression → residuals Rx.
        2. Regress Y on Z → residuals Ry.
        3. Test Rx ⊥ Ry via HSIC.

    Parameters
    ----------
    kernel_x, kernel_y : Kernel
        Kernels for the marginal HSIC test on residuals.
    kernel_z : Kernel
        Kernel for the regression on Z.
    reg_lambda : float
        Regularisation strength for kernel ridge regression.
    n_permutations : int
        Permutations for the final HSIC test.
    block_size : int or None
        Block permutation size for dependent data.
    """

    def __init__(
        self,
        kernel_x: Optional[Kernel] = None,
        kernel_y: Optional[Kernel] = None,
        kernel_z: Optional[Kernel] = None,
        reg_lambda: float = 1e-3,
        n_permutations: int = 500,
        block_size: Optional[int] = None,
        alpha: float = 0.05,
    ) -> None:
        self.kernel_x = kernel_x or GaussianKernel()
        self.kernel_y = kernel_y or GaussianKernel()
        self.kernel_z = kernel_z or GaussianKernel()
        self.reg_lambda = reg_lambda
        self.n_permutations = n_permutations
        self.block_size = block_size
        self.alpha = alpha

    def _residualize(self, V: NDArray, Z: NDArray) -> NDArray:
        """Residualize V on Z using kernel ridge regression."""
        Kz = self.kernel_z.compute(Z)
        n = Kz.shape[0]
        alpha = np.linalg.solve(
            Kz + self.reg_lambda * np.eye(n),
            V,
        )
        pred = Kz @ alpha
        return V - pred

    def test(
        self,
        X: NDArray,
        Y: NDArray,
        Z: NDArray,
        seed: Optional[int] = None,
    ) -> HSICResult:
        """Test X ⊥ Y | Z.

        Parameters
        ----------
        X : ndarray, shape (n,) or (n, d_x)
        Y : ndarray, shape (n,) or (n, d_y)
        Z : ndarray, shape (n,) or (n, d_z)

        Returns
        -------
        HSICResult
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        if X.shape[0] == 1:
            X = X.T
        if Y.shape[0] == 1:
            Y = Y.T
        if Z.shape[0] == 1:
            Z = Z.T

        Rx = self._residualize(X, Z)
        Ry = self._residualize(Y, Z)

        hsic_test = HSIC(
            kernel_x=self.kernel_x,
            kernel_y=self.kernel_y,
            n_permutations=self.n_permutations,
            block_size=self.block_size,
            alpha=self.alpha,
        )
        result = hsic_test.test(Rx, Ry, seed=seed)
        result.method = f"conditional_{result.method}"
        return result


# ====================================================================
# Convenience wrappers
# ====================================================================

def hsic_independence_test(
    X: NDArray,
    Y: NDArray,
    kernel: str = "gaussian",
    n_permutations: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> HSICResult:
    """One-call wrapper for HSIC independence test."""
    kernel_map = {
        "gaussian": GaussianKernel,
        "polynomial": PolynomialKernel,
        "linear": LinearKernel,
    }
    k_cls = kernel_map.get(kernel, GaussianKernel)
    test = HSIC(
        kernel_x=k_cls(),
        kernel_y=k_cls(),
        n_permutations=n_permutations,
        alpha=alpha,
    )
    return test.test(X, Y, seed=seed)


def hsic_conditional_test(
    X: NDArray,
    Y: NDArray,
    Z: NDArray,
    n_permutations: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> HSICResult:
    """One-call wrapper for conditional HSIC test."""
    test = ConditionalHSIC(
        n_permutations=n_permutations,
        alpha=alpha,
    )
    return test.test(X, Y, Z, seed=seed)


# ====================================================================
# Normalised HSIC (NHSIC)
# ====================================================================

def normalised_hsic(
    X: NDArray,
    Y: NDArray,
    kernel_x: Optional[Kernel] = None,
    kernel_y: Optional[Kernel] = None,
) -> float:
    """Normalised HSIC: HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y)).

    Analogous to correlation; ranges in [0, 1] for positive-definite
    kernels.
    """
    kx = kernel_x or GaussianKernel()
    ky = kernel_y or GaussianKernel()

    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    if X.shape[0] == 1:
        X = X.T
    if Y.shape[0] == 1:
        Y = Y.T

    Kx = kx.compute(X)
    Ky = ky.compute(Y)

    hsic_xy = _unbiased_hsic(Kx, Ky)
    hsic_xx = _unbiased_hsic(Kx, Kx)
    hsic_yy = _unbiased_hsic(Ky, Ky)

    denom = np.sqrt(max(hsic_xx, 0) * max(hsic_yy, 0))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(hsic_xy / denom, 0.0, 1.0))
