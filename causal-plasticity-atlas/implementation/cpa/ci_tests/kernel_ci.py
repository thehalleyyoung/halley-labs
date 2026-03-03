"""Kernel-based conditional independence tests.

Implements the Kernel CI (KCI) test and the Hilbert-Schmidt
Independence Criterion (HSIC) test for detecting nonlinear
dependencies.

The KCI test is based on the framework of Zhang et al. (2011) and
tests X ⊥ Y | Z by regressing out the effect of Z from the kernel
matrices of X and Y, then measuring residual dependence via HSIC.

Both biased and unbiased HSIC estimators are provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class KCIResult:
    """Result of a kernel CI test."""

    statistic: float
    p_value: float
    independent: bool
    conditioning_set: Set[int] = field(default_factory=set)
    method: str = "kci"


# ---------------------------------------------------------------------------
# Kernel utilities
# ---------------------------------------------------------------------------

def _median_bandwidth(X: NDArray[np.float64]) -> float:
    """Median heuristic for the RBF kernel bandwidth.

    Sets bandwidth so that ``sigma = median(||x_i - x_j||)`` for all
    distinct pairs.

    Parameters
    ----------
    X : ndarray of shape (n, d)

    Returns
    -------
    float
        Bandwidth parameter (sigma).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    if n <= 1:
        return 1.0
    dists = pdist(X, metric="euclidean")
    med = float(np.median(dists))
    return med if med > 1e-10 else 1.0


def _compute_kernel_matrix(
    X: NDArray[np.float64],
    kernel: str = "rbf",
    bandwidth: Optional[float] = None,
    degree: int = 2,
    gamma: Optional[float] = None,
) -> NDArray[np.float64]:
    """Compute a kernel (Gram) matrix.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    kernel : str
        ``"rbf"`` (Gaussian), ``"polynomial"``, or ``"linear"``.
    bandwidth : float or None
        RBF bandwidth (sigma).  If ``None``, the median heuristic is used.
    degree : int
        Polynomial degree (only for ``"polynomial"`` kernel).
    gamma : float or None
        Scale for ``"polynomial"`` kernel.  Defaults to ``1/d``.

    Returns
    -------
    K : ndarray of shape (n, n)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape

    if kernel == "rbf":
        if bandwidth is None:
            bandwidth = _median_bandwidth(X)
        sq_dists = squareform(pdist(X, metric="sqeuclidean"))
        K = np.exp(-sq_dists / (2.0 * bandwidth ** 2))

    elif kernel == "polynomial":
        if gamma is None:
            gamma = 1.0 / max(d, 1)
        K = (gamma * (X @ X.T) + 1.0) ** degree

    elif kernel == "linear":
        K = X @ X.T

    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")

    return K


def _center_kernel_matrix(K: NDArray[np.float64]) -> NDArray[np.float64]:
    """Center a kernel matrix: K_c = H K H, where H = I - 11^T/n."""
    n = K.shape[0]
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    total_mean = K.mean()
    return K - row_mean - col_mean + total_mean


def _standardize(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Column-standardize to zero mean and unit variance."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-10] = 1.0
    return (X - mean) / std


# ---------------------------------------------------------------------------
# HSIC test (unconditional independence)
# ---------------------------------------------------------------------------

class HSICTest:
    """Hilbert-Schmidt Independence Criterion (unconditional) test.

    Tests X ⊥ Y by computing the empirical HSIC and comparing against a
    null distribution estimated via permutations or a Gamma approximation.

    Parameters
    ----------
    alpha : float
        Significance level.
    kernel : str
        Kernel function name (``"rbf"``, ``"polynomial"``, ``"linear"``).
    n_permutations : int
        Number of permutations for the null distribution.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        kernel: str = "rbf",
        n_permutations: int = 1000,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = alpha
        self.kernel = kernel
        self.n_permutations = n_permutations

    def test(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """Run the HSIC test.

        Parameters
        ----------
        x : ndarray of shape (n,) or (n, d1)
        y : ndarray of shape (n,) or (n, d2)

        Returns
        -------
        statistic : float
            The HSIC statistic (scaled by n).
        pvalue : float
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        n = x.shape[0]
        if n < 6:
            return (0.0, 1.0)

        x = _standardize(x)
        y = _standardize(y)

        Kx = _compute_kernel_matrix(x, self.kernel)
        Ky = _compute_kernel_matrix(y, self.kernel)

        stat = self.hsic_statistic(Kx, Ky)

        # Permutation test
        pvalue = self._permutation_p_value(stat, Kx, Ky, n)
        return (float(stat), float(pvalue))

    def hsic_statistic(
        self,
        kx: NDArray[np.float64],
        ky: NDArray[np.float64],
    ) -> float:
        """Compute the biased HSIC estimator from kernel matrices.

        HSIC_b = (1/n^2) Tr(Kx_c @ Ky_c)
        """
        n = kx.shape[0]
        Kxc = _center_kernel_matrix(kx)
        Kyc = _center_kernel_matrix(ky)
        return float(np.trace(Kxc @ Kyc) / (n * n))

    def unbiased_hsic(
        self,
        kx: NDArray[np.float64],
        ky: NDArray[np.float64],
    ) -> float:
        """Compute the unbiased HSIC estimator (Song et al., 2012).

        Uses the U-statistic form: HSIC_u = 1/(n(n-3)) * [...]
        """
        n = kx.shape[0]
        if n < 4:
            return 0.0
        return _unbiased_hsic(kx, ky, n)

    def _permutation_p_value(
        self,
        observed_stat: float,
        Kx: NDArray[np.float64],
        Ky: NDArray[np.float64],
        n: int,
    ) -> float:
        """Estimate p-value by permuting rows/columns of Ky."""
        rng = np.random.default_rng(seed=42)
        count = 0
        for _ in range(self.n_permutations):
            perm = rng.permutation(n)
            Ky_perm = Ky[np.ix_(perm, perm)]
            null_stat = self.hsic_statistic(Kx, Ky_perm)
            if null_stat >= observed_stat:
                count += 1
        return (count + 1) / (self.n_permutations + 1)

    def gamma_approximation(
        self,
        stat: float,
        kx: NDArray[np.float64],
        ky: NDArray[np.float64],
    ) -> float:
        """Approximate p-value via Gamma distribution fit.

        Matches the first two moments of the HSIC null distribution to
        a Gamma(shape, scale) distribution.
        """
        n = kx.shape[0]
        Kxc = _center_kernel_matrix(kx)
        Kyc = _center_kernel_matrix(ky)

        # Mean and variance of the null distribution
        mean_Kx = Kxc.mean()
        mean_Ky = Kyc.mean()
        # E[HSIC] under null
        mu = (1.0 + mean_Kx * mean_Ky - mean_Kx - mean_Ky) / n

        # Rough variance estimate
        Hx = Kxc - np.diag(np.diag(Kxc))
        Hy = Kyc - np.diag(np.diag(Kyc))
        var_approx = 2.0 * np.sum(Hx ** 2) * np.sum(Hy ** 2) / (n ** 4)

        if var_approx < 1e-15 or mu < 1e-15:
            return 1.0

        # Gamma parameters
        scale = var_approx / mu
        shape = mu / scale
        if shape <= 0 or scale <= 0:
            return 1.0

        pval = float(sp_stats.gamma.sf(stat, a=shape, scale=scale))
        return pval


def _unbiased_hsic(
    Kx: NDArray[np.float64],
    Ky: NDArray[np.float64],
    n: int,
) -> float:
    """Unbiased HSIC estimator (U-statistic form).

    HSIC_u = (1/(n(n-3))) * [tr(K̃x K̃y) + (1^T K̃x 1)(1^T K̃y 1)/((n-1)(n-2))
             - (2/(n-2)) 1^T K̃x K̃y 1]

    where K̃ has zeros on the diagonal.
    """
    if n < 4:
        return 0.0

    # Zero the diagonals
    Kx_tilde = Kx.copy()
    Ky_tilde = Ky.copy()
    np.fill_diagonal(Kx_tilde, 0.0)
    np.fill_diagonal(Ky_tilde, 0.0)

    term1 = np.trace(Kx_tilde @ Ky_tilde)
    ones = np.ones(n)
    sum_Kx = ones @ Kx_tilde @ ones  # 1^T K̃x 1
    sum_Ky = ones @ Ky_tilde @ ones  # 1^T K̃y 1
    term2 = (sum_Kx * sum_Ky) / ((n - 1) * (n - 2))
    term3 = (2.0 / (n - 2)) * (ones @ (Kx_tilde @ Ky_tilde) @ ones)

    hsic = (term1 + term2 - term3) / (n * (n - 3))
    return float(hsic)


# ---------------------------------------------------------------------------
# Kernel Conditional Independence test (KCI)
# ---------------------------------------------------------------------------

class KernelCITest:
    """Kernel conditional independence test (KCI).

    Implements the KCI test for X ⊥ Y | Z.  The conditioning is handled by
    regressing the effects of Z out of the kernel matrices for X and Y
    using kernel ridge regression, then testing residual dependence via HSIC.

    Parameters
    ----------
    alpha : float
        Significance level.
    kernel : str
        Kernel function name (``"rbf"``, ``"polynomial"``, ``"linear"``).
    n_bootstrap : int
        Number of bootstrap samples for the null distribution.
    regularization : float
        Ridge regularization parameter for kernel regression.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        kernel: str = "rbf",
        n_bootstrap: int = 1000,
        regularization: float = 1e-3,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = alpha
        self.kernel = kernel
        self.n_bootstrap = n_bootstrap
        self.regularization = regularization

    def test(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        conditioning_set: Set[int] | None = None,
    ) -> Tuple[float, float]:
        """Run the KCI test.

        Parameters
        ----------
        data : ndarray of shape (n, p)
        x, y : int
            Column indices to test.
        conditioning_set : set of int or None
            Columns to condition on.

        Returns
        -------
        statistic : float
        pvalue : float
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("data must be 2-D")
        n, p = data.shape
        z = conditioning_set if conditioning_set is not None else set()

        all_idx = {x, y} | z
        if any(i < 0 or i >= p for i in all_idx):
            raise IndexError(f"Variable index out of range [0, {p})")

        if n < 6:
            return (0.0, 1.0)

        X_col = _standardize(data[:, x])
        Y_col = _standardize(data[:, y])

        if len(z) == 0:
            # No conditioning — reduce to HSIC
            hsic = HSICTest(
                alpha=self.alpha,
                kernel=self.kernel,
                n_permutations=self.n_bootstrap,
            )
            return hsic.test(X_col, Y_col)

        Z_cols = _standardize(data[:, sorted(z)])

        Kx = _compute_kernel_matrix(X_col, self.kernel)
        Ky = _compute_kernel_matrix(Y_col, self.kernel)
        Kz = _compute_kernel_matrix(Z_cols, self.kernel)

        stat = self.compute_test_statistic(Kx, Ky, Kz)

        pvalue = self._bootstrap_p_value(stat, Kx, Ky, Kz, n)
        return (float(stat), float(pvalue))

    def test_full(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        conditioning_set: Set[int] | None = None,
    ) -> KCIResult:
        """Like :meth:`test` but returns a :class:`KCIResult`."""
        z = conditioning_set if conditioning_set is not None else set()
        stat, pval = self.test(data, x, y, z)
        return KCIResult(
            statistic=stat,
            p_value=pval,
            independent=(pval >= self.alpha),
            conditioning_set=set(z),
            method="kci",
        )

    def compute_test_statistic(
        self,
        kx: NDArray[np.float64],
        ky: NDArray[np.float64],
        kz: NDArray[np.float64],
    ) -> float:
        """Compute the KCI statistic from kernel matrices.

        Residualize Kx and Ky w.r.t. Kz using kernel ridge regression,
        then compute HSIC on the residual kernel matrices.

        Parameters
        ----------
        kx, ky, kz : ndarray of shape (n, n)
            Centered kernel matrices.

        Returns
        -------
        float
            KCI statistic.
        """
        n = kx.shape[0]
        I_n = np.eye(n)

        # Kernel ridge: R_z = (Kz + lambda*I)^{-1}
        Rz_inv = np.linalg.solve(
            kz + self.regularization * I_n, I_n
        )

        # Projection matrix: P = I - Kz (Kz + lambda I)^{-1}
        P = I_n - kz @ Rz_inv

        # Residualized kernel matrices
        Kx_res = P @ kx @ P.T
        Ky_res = P @ ky @ P.T

        # Center the residualized kernels
        Kx_res = _center_kernel_matrix(Kx_res)
        Ky_res = _center_kernel_matrix(Ky_res)

        # HSIC on residuals
        stat = float(np.trace(Kx_res @ Ky_res) / (n * n))
        return stat

    def _bootstrap_p_value(
        self,
        observed_stat: float,
        Kx: NDArray[np.float64],
        Ky: NDArray[np.float64],
        Kz: NDArray[np.float64],
        n: int,
    ) -> float:
        """Bootstrap p-value: permute Y-indices, recompute statistic."""
        rng = np.random.default_rng(seed=42)
        count = 0
        for _ in range(self.n_bootstrap):
            perm = rng.permutation(n)
            Ky_perm = Ky[np.ix_(perm, perm)]
            null_stat = self.compute_test_statistic(Kx, Ky_perm, Kz)
            if null_stat >= observed_stat:
                count += 1
        return (count + 1) / (self.n_bootstrap + 1)

    def _gamma_approximation(
        self,
        stat: float,
        Kx_res: NDArray[np.float64],
        Ky_res: NDArray[np.float64],
    ) -> float:
        """Gamma-distribution approximation for the null p-value.

        Matches the first two moments of the HSIC null under the Gamma
        family.
        """
        n = Kx_res.shape[0]
        Kxc = _center_kernel_matrix(Kx_res)
        Kyc = _center_kernel_matrix(Ky_res)

        # E[HSIC] under null ~ (1/n) trace(Kxc) * trace(Kyc) / n^2
        mu = float(np.trace(Kxc) * np.trace(Kyc)) / (n ** 3)

        # Var[HSIC] approximation
        Bx = Kxc * Kxc  # element-wise square
        By = Kyc * Kyc
        var = 2.0 * float(np.sum(Bx) * np.sum(By)) / (n ** 4)

        if var < 1e-15 or mu < 1e-15:
            return 1.0

        scale = var / mu
        shape = mu / scale
        if shape <= 0 or scale <= 0:
            return 1.0
        return float(sp_stats.gamma.sf(stat, a=shape, scale=scale))
