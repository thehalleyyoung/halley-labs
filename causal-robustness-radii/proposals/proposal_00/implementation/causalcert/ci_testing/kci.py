"""
Kernel conditional-independence test with Nyström approximation.

Implements the KCI test of Zhang et al. (2012) with an optional Nyström
low-rank approximation of the kernel matrices for scalability to large *n*.

Key features:
- Gaussian RBF kernel with median heuristic for bandwidth selection
- Nyström approximation with configurable number of inducing points
- Centralized kernel matrices
- Eigenvalue-based test statistic with gamma approximation for null distribution
- Bootstrap-based p-value as fallback
- Warm-start kernel caching for reuse across tests
- Handles continuous, discrete, and mixed variable types
- Proper regularisation for numerical stability
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from causalcert.ci_testing.base import (
    BaseCITest,
    _extract_columns,
    _insufficient_sample_result,
    _validate_inputs,
)
from causalcert.types import CITestMethod, CITestResult, NodeId, NodeSet

_EPS = 1e-10
_REGULARIZATION = 1e-5


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------


def _median_heuristic(X: np.ndarray) -> float:
    """Compute the RBF bandwidth via the median heuristic.

    The bandwidth is set to ``1 / (2 * median_dist^2)`` where
    ``median_dist`` is the median of all pairwise Euclidean distances.

    Parameters
    ----------
    X : np.ndarray
        Data matrix ``(n, d)``.

    Returns
    -------
    float
        Bandwidth parameter gamma for the RBF kernel.
    """
    if X.shape[0] < 2:
        return 1.0
    if X.ndim == 1:
        X = X[:, np.newaxis]

    # Subsample for efficiency if n is large
    n = X.shape[0]
    if n > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=2000, replace=False)
        X = X[idx]

    dists = pdist(X, metric="euclidean")
    if len(dists) == 0:
        return 1.0
    med = float(np.median(dists))
    if med < _EPS:
        return 1.0
    return 1.0 / (2.0 * med * med)


def _rbf_kernel(X: np.ndarray, gamma: float) -> np.ndarray:
    """Compute the Gaussian RBF kernel matrix.

    ``K[i,j] = exp(-gamma * ||x_i - x_j||^2)``

    Parameters
    ----------
    X : np.ndarray
        Data matrix ``(n, d)``.
    gamma : float
        Bandwidth parameter.

    Returns
    -------
    np.ndarray
        Kernel matrix ``(n, n)``.
    """
    sq_dists = squareform(pdist(X, metric="sqeuclidean"))
    return np.exp(-gamma * sq_dists)


def _rbf_kernel_cross(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """Compute the cross RBF kernel matrix between X and Y.

    ``K[i,j] = exp(-gamma * ||x_i - y_j||^2)``

    Parameters
    ----------
    X : np.ndarray
        First data matrix ``(n, d)``.
    Y : np.ndarray
        Second data matrix ``(m, d)``.
    gamma : float
        Bandwidth parameter.

    Returns
    -------
    np.ndarray
        Kernel matrix ``(n, m)``.
    """
    sq_dists = (
        np.sum(X ** 2, axis=1)[:, None]
        + np.sum(Y ** 2, axis=1)[None, :]
        - 2.0 * X @ Y.T
    )
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-gamma * sq_dists)


def _polynomial_kernel(
    X: np.ndarray, degree: int = 2, coef0: float = 1.0
) -> np.ndarray:
    """Compute the polynomial kernel matrix.

    ``K[i,j] = (coef0 + <x_i, x_j>)^degree``

    Parameters
    ----------
    X : np.ndarray
        Data matrix ``(n, d)``.
    degree : int
        Polynomial degree.
    coef0 : float
        Constant term.

    Returns
    -------
    np.ndarray
        Kernel matrix ``(n, n)``.
    """
    return (coef0 + X @ X.T) ** degree


def _delta_kernel(X: np.ndarray) -> np.ndarray:
    """Compute the delta (indicator) kernel for discrete data.

    ``K[i,j] = 1 if x_i == x_j else 0``

    For multi-dimensional X, requires equality in all dimensions.

    Parameters
    ----------
    X : np.ndarray
        Data matrix ``(n, d)``.

    Returns
    -------
    np.ndarray
        Kernel matrix ``(n, n)`` with binary entries.
    """
    if X.ndim == 1:
        X = X[:, np.newaxis]
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        K[i, :] = np.all(X[i] == X, axis=1).astype(np.float64)
    return K


# ---------------------------------------------------------------------------
# Centering and Nyström
# ---------------------------------------------------------------------------


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix in feature space.

    ``K_c = H @ K @ H`` where ``H = I - (1/n) * 1 * 1^T``.

    Parameters
    ----------
    K : np.ndarray
        Kernel matrix ``(n, n)``.

    Returns
    -------
    np.ndarray
        Centered kernel matrix ``(n, n)``.
    """
    n = K.shape[0]
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    total_mean = K.mean()
    return K - row_mean - col_mean + total_mean


def _nystrom_decomposition(
    K: np.ndarray,
    rank: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute a rank-*rank* Nyström approximation of kernel matrix *K*.

    Selects *rank* inducing points uniformly at random, computes the
    sub-matrices, and reconstructs the low-rank approximation.

    Parameters
    ----------
    K : np.ndarray
        Full kernel matrix ``(n, n)``.
    rank : int
        Target rank (number of inducing points).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Low-rank approximation ``(n, n)``.
    """
    n = K.shape[0]
    rank = min(rank, n)

    # Select inducing points
    indices = rng.choice(n, size=rank, replace=False)
    indices.sort()

    # Sub-matrices
    K_mm = K[np.ix_(indices, indices)]  # (m, m)
    K_nm = K[:, indices]  # (n, m)

    # Regularize K_mm
    K_mm += _REGULARIZATION * np.eye(rank)

    # Eigendecomposition of K_mm
    eigvals, eigvecs = np.linalg.eigh(K_mm)

    # Clip small eigenvalues for numerical stability
    eigvals = np.maximum(eigvals, _EPS)

    # Reconstruct: K_approx = K_nm @ K_mm^{-1} @ K_nm^T
    K_mm_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    L = K_nm @ K_mm_inv_sqrt  # (n, m)

    return L @ L.T


def _nystrom_from_data(
    X: np.ndarray,
    gamma: float,
    rank: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute a Nyström-approximated kernel matrix directly from data.

    More memory-efficient than computing the full kernel first.

    Parameters
    ----------
    X : np.ndarray
        Data matrix ``(n, d)``.
    gamma : float
        RBF bandwidth.
    rank : int
        Number of inducing points.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Low-rank kernel approximation ``(n, n)``.
    """
    n = X.shape[0]
    m = min(rank, n)

    indices = rng.choice(n, size=m, replace=False)
    indices.sort()

    X_m = X[indices]  # (m, d) — inducing points

    K_mm = _rbf_kernel(X_m, gamma)  # (m, m)
    K_mm += _REGULARIZATION * np.eye(m)

    K_nm = _rbf_kernel_cross(X, X_m, gamma)  # (n, m)

    eigvals, eigvecs = np.linalg.eigh(K_mm)
    eigvals = np.maximum(eigvals, _EPS)

    K_mm_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    L = K_nm @ K_mm_inv_sqrt  # (n, m)

    return L @ L.T


# ---------------------------------------------------------------------------
# Test statistic and null distribution
# ---------------------------------------------------------------------------


def _kci_test_statistic(
    Kx_c: np.ndarray,
    Ky_c: np.ndarray,
    Kz_c: np.ndarray | None,
    n: int,
    regularization: float = _REGULARIZATION,
) -> float:
    """Compute the KCI test statistic.

    When testing X ⊥ Y | Z:
    - If Z is empty: T = (1/n) * trace(Kx_c @ Ky_c)
    - If Z is non-empty: Project Kx and Ky onto the complement of the
      Z feature space, then compute the trace statistic.

    Parameters
    ----------
    Kx_c : np.ndarray
        Centered kernel matrix for X ``(n, n)``.
    Ky_c : np.ndarray
        Centered kernel matrix for Y ``(n, n)``.
    Kz_c : np.ndarray | None
        Centered kernel matrix for Z ``(n, n)`` or ``None``.
    n : int
        Sample size.
    regularization : float
        Ridge parameter for numerical stability.

    Returns
    -------
    float
        KCI test statistic.
    """
    if Kz_c is None:
        # Unconditional test: T = (1/n) * trace(Kx_c @ Ky_c)
        T = np.trace(Kx_c @ Ky_c) / n
        return float(T)

    # Conditional test: project onto orthogonal complement of Z feature space
    # R_z = I - Kz_c @ (Kz_c + lambda * I)^{-1}
    Rz = np.linalg.solve(
        Kz_c + regularization * n * np.eye(n),
        Kz_c,
    )
    Rz = np.eye(n) - Rz  # Residual projection

    # Projected kernels
    Kx_proj = Rz @ Kx_c @ Rz.T
    Ky_proj = Rz @ Ky_c @ Rz.T

    # Re-center projected kernels
    Kx_proj = _center_kernel(Kx_proj)
    Ky_proj = _center_kernel(Ky_proj)

    T = np.trace(Kx_proj @ Ky_proj) / n
    return float(T)


def _gamma_approximation(
    Kx_c: np.ndarray,
    Ky_c: np.ndarray,
    Kz_c: np.ndarray | None,
    n: int,
    regularization: float = _REGULARIZATION,
) -> tuple[float, float]:
    """Compute parameters for the gamma approximation to the null distribution.

    Under the null, the KCI statistic is approximately distributed as a
    weighted sum of chi-squared(1) variables.  We approximate this
    distribution by matching the first two moments to a Gamma distribution.

    Parameters
    ----------
    Kx_c, Ky_c : np.ndarray
        Centered kernel matrices for X and Y.
    Kz_c : np.ndarray | None
        Centered kernel matrix for Z.
    n : int
        Sample size.
    regularization : float
        Ridge parameter.

    Returns
    -------
    tuple[float, float]
        ``(shape, scale)`` parameters of the gamma approximation.
    """
    if Kz_c is None:
        # Eigenvalues of the product matrix determine the null distribution
        eigx = np.linalg.eigvalsh(Kx_c / n)
        eigy = np.linalg.eigvalsh(Ky_c / n)
    else:
        Rz = np.linalg.solve(
            Kz_c + regularization * n * np.eye(n),
            Kz_c,
        )
        Rz = np.eye(n) - Rz

        Kx_proj = _center_kernel(Rz @ Kx_c @ Rz.T)
        Ky_proj = _center_kernel(Rz @ Ky_c @ Rz.T)

        eigx = np.linalg.eigvalsh(Kx_proj / n)
        eigy = np.linalg.eigvalsh(Ky_proj / n)

    # Keep only positive eigenvalues
    eigx = eigx[eigx > _EPS]
    eigy = eigy[eigy > _EPS]

    if len(eigx) == 0 or len(eigy) == 0:
        return 1.0, 1.0

    # Mean and variance of the null distribution
    # E[T] = (1/n) * sum_i sum_j lambda_i * mu_j
    # Var[T] = (2/n^2) * sum_i sum_j (lambda_i * mu_j)^2
    mean_null = float(np.sum(eigx) * np.sum(eigy) / n)
    var_null = float(2.0 * np.sum(eigx ** 2) * np.sum(eigy ** 2) / (n * n))

    if mean_null < _EPS or var_null < _EPS:
        return 1.0, 1.0

    # Gamma parameters via moment matching
    # shape = mean^2 / var, scale = var / mean
    shape = mean_null ** 2 / var_null
    scale = var_null / mean_null

    return shape, scale


def _bootstrap_p_value(
    Kx_c: np.ndarray,
    Ky_c: np.ndarray,
    Kz_c: np.ndarray | None,
    n: int,
    observed_stat: float,
    n_bootstrap: int,
    rng: np.random.Generator,
    regularization: float = _REGULARIZATION,
) -> float:
    """Compute a bootstrap p-value for the KCI test.

    Permutes the rows/columns of Ky_c to simulate the null distribution.

    Parameters
    ----------
    Kx_c, Ky_c : np.ndarray
        Centered kernel matrices.
    Kz_c : np.ndarray | None
        Centered conditioning kernel matrix.
    n : int
        Sample size.
    observed_stat : float
        The observed test statistic.
    n_bootstrap : int
        Number of bootstrap iterations.
    rng : np.random.Generator
        Random number generator.
    regularization : float
        Ridge parameter.

    Returns
    -------
    float
        Bootstrap p-value.
    """
    null_stats = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        perm = rng.permutation(n)
        Ky_perm = Ky_c[np.ix_(perm, perm)]
        null_stats[b] = _kci_test_statistic(
            Kx_c, Ky_perm, Kz_c, n, regularization
        )

    p_value = float(np.mean(null_stats >= observed_stat))
    return max(p_value, 1.0 / (n_bootstrap + 1))


# ---------------------------------------------------------------------------
# Variable type detection
# ---------------------------------------------------------------------------


def _detect_variable_type(col: np.ndarray, max_discrete: int = 10) -> str:
    """Detect whether a variable is continuous or discrete.

    Parameters
    ----------
    col : np.ndarray
        1-D data array.
    max_discrete : int
        Maximum number of unique values to consider discrete.

    Returns
    -------
    str
        ``"continuous"`` or ``"discrete"``.
    """
    n_unique = len(np.unique(col))
    if n_unique <= max_discrete:
        return "discrete"
    return "continuous"


def _select_kernel_for_data(
    X: np.ndarray,
    gamma: float | None = None,
) -> tuple[np.ndarray, float | None]:
    """Select and compute kernel matrix based on data type.

    For continuous data, uses RBF with median heuristic.
    For discrete data, uses delta kernel.
    For mixed data, uses the product of individual kernels.

    Parameters
    ----------
    X : np.ndarray
        Data matrix ``(n, d)``.
    gamma : float | None
        Pre-specified bandwidth (only for continuous).

    Returns
    -------
    tuple[np.ndarray, float | None]
        ``(kernel_matrix, gamma_used)``.
    """
    if X.ndim == 1:
        X = X[:, np.newaxis]

    n, d = X.shape

    if d == 0:
        return np.eye(n), None

    # Check each column
    types = [_detect_variable_type(X[:, j]) for j in range(d)]

    cont_cols = [j for j, t in enumerate(types) if t == "continuous"]
    disc_cols = [j for j, t in enumerate(types) if t == "discrete"]

    K = np.ones((n, n))
    gamma_used = gamma

    if cont_cols:
        X_cont = X[:, cont_cols]
        if gamma is None:
            gamma_used = _median_heuristic(X_cont)
        K *= _rbf_kernel(X_cont, gamma_used)

    if disc_cols:
        X_disc = X[:, disc_cols]
        K *= _delta_kernel(X_disc)

    return K, gamma_used


# ---------------------------------------------------------------------------
# Main KCI test class
# ---------------------------------------------------------------------------


class KernelCITest(BaseCITest):
    """Kernel CI test (KCI) with optional Nyström approximation.

    Implements the test of Zhang et al. (2012) for testing X ⊥ Y | Z
    using kernel-based measures of conditional dependence.

    The test statistic is the trace of the product of centralized,
    residualized kernel matrices.  Under the null hypothesis of
    conditional independence, the statistic follows (approximately)
    a weighted sum of chi-squared variables, which is well-approximated
    by a Gamma distribution.

    Parameters
    ----------
    alpha : float
        Significance level.
    kernel : str
        Kernel type (``"rbf"`` or ``"polynomial"``).
    nystrom_rank : int | None
        Rank of the Nyström approximation.  ``None`` uses the full kernel.
    gamma : float | None
        RBF bandwidth parameter.  ``None`` uses the median heuristic.
    n_bootstrap : int
        Number of bootstrap samples for the fallback p-value.
    use_gamma_approx : bool
        If ``True`` (default), use the gamma approximation for the null
        distribution.  If ``False``, always use bootstrap.
    cache : dict | None
        Optional dictionary for kernel matrix caching (warm-start).
        Keys are ``tuple[NodeId, ...]``, values are kernel matrices.
    seed : int
        Random seed.
    """

    method = CITestMethod.KERNEL

    def __init__(
        self,
        alpha: float = 0.05,
        kernel: str = "rbf",
        nystrom_rank: int | None = 200,
        gamma: float | None = None,
        n_bootstrap: int = 500,
        use_gamma_approx: bool = True,
        cache: dict[tuple[int, ...], np.ndarray] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(alpha=alpha, seed=seed)
        self.kernel = kernel
        self.nystrom_rank = nystrom_rank
        self.gamma = gamma
        self.n_bootstrap = n_bootstrap
        self.use_gamma_approx = use_gamma_approx
        self._kernel_cache: dict[tuple[int, ...], np.ndarray] = cache if cache is not None else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def test(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: pd.DataFrame,
    ) -> CITestResult:
        """Run the kernel CI test.

        Parameters
        ----------
        x, y : NodeId
            Variables to test.
        conditioning_set : NodeSet
            Conditioning variables.
        data : pd.DataFrame
            Observational data.

        Returns
        -------
        CITestResult
        """
        _validate_inputs(data, x, y, conditioning_set)
        x_col, y_col, z_cols = _extract_columns(data, x, y, conditioning_set)
        n = len(x_col)

        if n < 10:
            return _insufficient_sample_result(
                x, y, conditioning_set, self.method, self.alpha
            )

        rng = np.random.default_rng(self.seed)

        # Build kernel matrices (with caching)
        Kx = self._get_or_compute_kernel(x_col, (x,), rng)
        Ky = self._get_or_compute_kernel(y_col, (y,), rng)
        Kz: np.ndarray | None = None
        if z_cols is not None and z_cols.shape[1] > 0:
            z_key = tuple(sorted(conditioning_set))
            Kz = self._get_or_compute_kernel(z_cols, z_key, rng)

        # Apply Nyström approximation if configured
        if self.nystrom_rank is not None and n > self.nystrom_rank:
            Kx = self._nystrom_approximation(Kx, self.nystrom_rank, rng)
            Ky = self._nystrom_approximation(Ky, self.nystrom_rank, rng)
            if Kz is not None:
                Kz = self._nystrom_approximation(Kz, self.nystrom_rank, rng)

        # Center kernel matrices
        Kx_c = _center_kernel(Kx)
        Ky_c = _center_kernel(Ky)
        Kz_c = _center_kernel(Kz) if Kz is not None else None

        # Compute test statistic
        T = _kci_test_statistic(Kx_c, Ky_c, Kz_c, n)

        # Compute p-value
        p_value = self._compute_p_value(Kx_c, Ky_c, Kz_c, n, T, rng)

        return self._make_result(x, y, conditioning_set, T, p_value)

    # ------------------------------------------------------------------
    # Kernel computation with caching
    # ------------------------------------------------------------------

    def _get_or_compute_kernel(
        self,
        data_cols: np.ndarray,
        cache_key: tuple[int, ...],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Retrieve a cached kernel or compute and cache a new one.

        Parameters
        ----------
        data_cols : np.ndarray
            Data for kernel computation ``(n, d)`` or ``(n,)``.
        cache_key : tuple[int, ...]
            Cache key (sorted node ids).
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Kernel matrix ``(n, n)``.
        """
        cached = self._kernel_cache.get(cache_key)
        if cached is not None and cached.shape[0] == data_cols.shape[0]:
            return cached.copy()

        K = self._compute_kernel_matrix(data_cols)
        self._kernel_cache[cache_key] = K.copy()
        return K

    def _compute_kernel_matrix(
        self,
        cols: np.ndarray,
    ) -> np.ndarray:
        """Compute the kernel matrix for the given columns.

        Automatically detects variable types and selects the appropriate
        kernel.  For continuous variables uses RBF with median heuristic
        (unless ``gamma`` was specified).  For discrete variables uses the
        delta kernel.

        Parameters
        ----------
        cols : np.ndarray
            Data matrix of shape ``(n, d)`` or ``(n,)``.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape ``(n, n)``.
        """
        if cols.ndim == 1:
            cols = cols[:, np.newaxis]

        if self.kernel == "polynomial":
            return _polynomial_kernel(cols)

        # RBF kernel (default) with automatic type detection
        K, _ = _select_kernel_for_data(cols, self.gamma)
        return K

    def _nystrom_approximation(
        self,
        K: np.ndarray,
        rank: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Compute rank-*rank* Nyström approximation of kernel matrix *K*.

        Parameters
        ----------
        K : np.ndarray
            Full kernel matrix ``(n, n)``.
        rank : int
            Target rank.
        rng : np.random.Generator | None
            Random number generator.

        Returns
        -------
        np.ndarray
            Low-rank approximation ``(n, n)``.
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)
        return _nystrom_decomposition(K, rank, rng)

    # ------------------------------------------------------------------
    # P-value computation
    # ------------------------------------------------------------------

    def _compute_p_value(
        self,
        Kx_c: np.ndarray,
        Ky_c: np.ndarray,
        Kz_c: np.ndarray | None,
        n: int,
        observed_stat: float,
        rng: np.random.Generator,
    ) -> float:
        """Compute the p-value using gamma approximation or bootstrap.

        First tries the gamma approximation.  If it produces degenerate
        results (e.g. shape/scale near zero), falls back to bootstrap.

        Parameters
        ----------
        Kx_c, Ky_c : np.ndarray
            Centered kernel matrices.
        Kz_c : np.ndarray | None
            Centered conditioning kernel.
        n : int
            Sample size.
        observed_stat : float
            Observed test statistic.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        float
            p-value in ``[0, 1]``.
        """
        if self.use_gamma_approx:
            p_value = self._gamma_p_value(Kx_c, Ky_c, Kz_c, n, observed_stat)
            if 0 < p_value < 1:
                return p_value

        # Fallback to bootstrap
        return _bootstrap_p_value(
            Kx_c, Ky_c, Kz_c, n, observed_stat,
            self.n_bootstrap, rng,
        )

    def _gamma_p_value(
        self,
        Kx_c: np.ndarray,
        Ky_c: np.ndarray,
        Kz_c: np.ndarray | None,
        n: int,
        observed_stat: float,
    ) -> float:
        """Compute the p-value using the gamma approximation.

        Parameters
        ----------
        Kx_c, Ky_c : np.ndarray
            Centered kernel matrices.
        Kz_c : np.ndarray | None
            Centered conditioning kernel.
        n : int
            Sample size.
        observed_stat : float
            Observed test statistic.

        Returns
        -------
        float
            p-value from the gamma approximation.
        """
        shape, scale = _gamma_approximation(Kx_c, Ky_c, Kz_c, n)

        if shape < _EPS or scale < _EPS:
            return -1.0  # Signal fallback

        try:
            p_value = 1.0 - stats.gamma.cdf(observed_stat, a=shape, scale=scale)
        except Exception:
            return -1.0

        return float(np.clip(p_value, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Batch test with kernel reuse
    # ------------------------------------------------------------------

    def test_batch(
        self,
        triples: list[tuple[NodeId, NodeId, NodeSet]],
        data: pd.DataFrame,
    ) -> list[CITestResult]:
        """Test a batch of CI queries with kernel caching.

        Pre-computes and caches kernel matrices for all unique variables
        appearing in the triples, then runs each test reusing cached kernels.

        Parameters
        ----------
        triples : list[tuple[NodeId, NodeId, NodeSet]]
            List of (x, y, conditioning_set) triples.
        data : pd.DataFrame
            Observational data.

        Returns
        -------
        list[CITestResult]
        """
        # Pre-warm the cache for all unique single-variable kernels
        all_nodes: set[NodeId] = set()
        for x, y, s in triples:
            all_nodes.add(x)
            all_nodes.add(y)
            all_nodes.update(s)

        rng = np.random.default_rng(self.seed)
        for node in sorted(all_nodes):
            if (node,) not in self._kernel_cache and node in data.columns:
                col = data[node].dropna().to_numpy(dtype=np.float64)
                if col.ndim == 1:
                    col = col[:, np.newaxis]
                K = self._compute_kernel_matrix(col)
                self._kernel_cache[(node,)] = K

        return [self.test(x, y, s, data) for x, y, s in triples]

    # ------------------------------------------------------------------
    # Kernel cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Remove all cached kernel matrices."""
        self._kernel_cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of cached kernel matrices."""
        return len(self._kernel_cache)

    def set_cache(self, cache: dict[tuple[int, ...], np.ndarray]) -> None:
        """Replace the kernel cache (for warm-starting from external cache).

        Parameters
        ----------
        cache : dict
            Pre-computed kernel matrices keyed by node-id tuples.
        """
        self._kernel_cache = cache

    def __repr__(self) -> str:
        return (
            f"KernelCITest(kernel={self.kernel!r}, "
            f"nystrom_rank={self.nystrom_rank}, "
            f"alpha={self.alpha}, gamma={self.gamma})"
        )
