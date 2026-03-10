"""
Hilbert-Schmidt Independence Criterion (HSIC) conditional-independence test.

Implements the HSIC test of Gretton et al. (2005, 2008) with:
- HSIC statistic via RBF and polynomial kernels
- Gamma approximation for the null distribution
- Permutation-based p-value as fallback
- Conditional HSIC via kernel-residualisation
- Bandwidth selection: median heuristic and cross-validation
- Block permutation for time-series data

References
----------
Gretton, A., Bousquet, O., Smola, A. & Schölkopf, B. (2005).
    Measuring statistical dependence with Hilbert-Schmidt norms.
    *ALT 2005*.

Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Schölkopf, B. &
    Smola, A. (2008). A kernel statistical test of independence.
    *NeurIPS 2008*.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist

from causalcert.ci_testing.base import (
    BaseCITest,
    CITestConfig,
    _extract_columns,
    _insufficient_sample_result,
    _validate_inputs,
)
from causalcert.ci_testing.kernel_ops import (
    KernelCache,
    block_diagonal_kernel,
    center_kernel,
    cross_validation_bandwidth,
    median_heuristic,
    nystrom_approximation,
    polynomial_kernel,
    rbf_kernel,
)
from causalcert.types import CITestMethod, CITestResult, NodeId, NodeSet

_EPS = 1e-10
_MIN_N = 10


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HSICConfig:
    """Configuration for the HSIC test.

    Attributes
    ----------
    kernel : str
        Kernel type for X and Y (``"rbf"`` or ``"polynomial"``).
    gamma : float | None
        RBF bandwidth.  ``None`` ⇒ automatic selection.
    bandwidth_method : str
        ``"median"`` or ``"cv"`` (cross-validation).
    degree : int
        Polynomial kernel degree (used when ``kernel="polynomial"``).
    coef0 : float
        Polynomial kernel bias.
    n_permutations : int
        Number of permutations for the fallback permutation test.
    use_gamma_approx : bool
        If ``True``, use gamma approximation for the null distribution
        before falling back to permutation.
    block_size : int | None
        Block size for block permutation (time-series data).
        ``None`` ⇒ standard i.i.d. permutation.
    nystrom_components : int | None
        If set, use Nyström approximation with this many components.
    """

    kernel: str = "rbf"
    gamma: float | None = None
    bandwidth_method: str = "median"
    degree: int = 3
    coef0: float = 1.0
    n_permutations: int = 500
    use_gamma_approx: bool = True
    block_size: int | None = None
    nystrom_components: int | None = None


# ---------------------------------------------------------------------------
# HSIC statistic computation
# ---------------------------------------------------------------------------


def _biased_hsic(
    Kx: np.ndarray,
    Ky: np.ndarray,
) -> float:
    """Compute the biased HSIC estimator.

    HSIC_b = (1/n^2) trace(Kx_c @ Ky_c)

    Parameters
    ----------
    Kx : np.ndarray
        Kernel matrix for X ``(n, n)`` (uncentered).
    Ky : np.ndarray
        Kernel matrix for Y ``(n, n)`` (uncentered).

    Returns
    -------
    float
        Biased HSIC value (non-negative).
    """
    n = Kx.shape[0]
    Kxc = center_kernel(Kx)
    Kyc = center_kernel(Ky)
    return max(float(np.trace(Kxc @ Kyc)) / (n * n), 0.0)


def _unbiased_hsic(
    Kx: np.ndarray,
    Ky: np.ndarray,
) -> float:
    """Compute the unbiased HSIC estimator (Song et al. 2012).

    Parameters
    ----------
    Kx : np.ndarray
        Kernel matrix for X ``(n, n)``.
    Ky : np.ndarray
        Kernel matrix for Y ``(n, n)``.

    Returns
    -------
    float
        Unbiased HSIC estimate.
    """
    n = Kx.shape[0]
    if n < 4:
        return 0.0

    # Zero the diagonal
    np.fill_diagonal(Kx, 0.0)
    np.fill_diagonal(Ky, 0.0)

    term1 = float(np.sum(Kx * Ky))
    term2 = float(np.sum(Kx) * np.sum(Ky)) / ((n - 1) * (n - 2))
    kx_row = Kx.sum(axis=1)
    ky_row = Ky.sum(axis=1)
    term3 = float(kx_row @ ky_row) * 2.0 / ((n - 2))

    hsic = (term1 + term2 - term3) / (n * (n - 3))
    return hsic


def _hsic_nystrom(
    X: np.ndarray,
    Y: np.ndarray,
    gamma_x: float,
    gamma_y: float,
    n_components: int,
    seed: int = 42,
) -> float:
    """HSIC via Nyström approximation for large samples.

    Uses low-rank kernel approximations to compute HSIC in
    O(n * m^2) instead of O(n^2).

    Parameters
    ----------
    X : np.ndarray
        First variable ``(n, d1)``.
    Y : np.ndarray
        Second variable ``(n, d2)``.
    gamma_x : float
        RBF bandwidth for X.
    gamma_y : float
        RBF bandwidth for Y.
    n_components : int
        Number of Nyström components.
    seed : int
        Random seed.

    Returns
    -------
    float
        Approximate HSIC value.
    """
    n = X.shape[0]
    Zx, _ = nystrom_approximation(
        X, n_components=n_components, gamma=gamma_x, seed=seed,
    )
    Zy, _ = nystrom_approximation(
        Y, n_components=n_components, gamma=gamma_y, seed=seed + 1,
    )

    # Center the Nyström features
    Zx = Zx - Zx.mean(axis=0, keepdims=True)
    Zy = Zy - Zy.mean(axis=0, keepdims=True)

    # HSIC ≈ ||Z_x^T Z_y||_F^2 / n^2
    cross = Zx.T @ Zy  # (m, m)
    return max(float(np.sum(cross ** 2)) / (n * n), 0.0)


# ---------------------------------------------------------------------------
# Gamma approximation for HSIC null distribution
# ---------------------------------------------------------------------------


def _gamma_approx_params(
    Kx: np.ndarray,
    Ky: np.ndarray,
) -> tuple[float, float]:
    """Estimate shape and scale of the gamma null for HSIC.

    Under H0 (independence), the distribution of ``n * HSIC_b`` is
    approximately a weighted chi-squared, which we approximate by a
    Gamma distribution via moment matching on eigenvalues.

    The null statistic is distributed as
    ``sum_{i,j} lx_i * ly_j * z_{ij}^2 / n`` where ``z_{ij} ~ N(0,1)``
    and ``lx_i``, ``ly_j`` are eigenvalues of the centered kernels
    divided by *n*.

    Parameters
    ----------
    Kx : np.ndarray
        Kernel matrix for X ``(n, n)`` (uncentered).
    Ky : np.ndarray
        Kernel matrix for Y ``(n, n)`` (uncentered).

    Returns
    -------
    tuple[float, float]
        ``(shape, scale)`` of the gamma distribution.
    """
    n = Kx.shape[0]
    Kxc = center_kernel(Kx)
    Kyc = center_kernel(Ky)

    # Eigenvalues of centred kernels / n
    eigx = np.linalg.eigvalsh(Kxc / n)
    eigy = np.linalg.eigvalsh(Kyc / n)

    # Keep only positive eigenvalues
    eigx = eigx[eigx > _EPS]
    eigy = eigy[eigy > _EPS]

    if len(eigx) == 0 or len(eigy) == 0:
        return 1.0, _EPS

    # Mean: E[n * HSIC_b] = sx * sy
    sx = float(np.sum(eigx))
    sy = float(np.sum(eigy))
    mu = sx * sy
    mu = max(mu, _EPS)

    # Variance: Var[n * HSIC_b] = 2 * sum(lx^2) * sum(ly^2)
    sx2 = float(np.sum(eigx ** 2))
    sy2 = float(np.sum(eigy ** 2))
    var = 2.0 * sx2 * sy2
    var = max(var, _EPS * _EPS)

    # Gamma: mean = shape * scale, var = shape * scale^2
    scale = var / mu if mu > _EPS else 1.0
    shape = mu / scale if scale > _EPS else 1.0

    return max(shape, _EPS), max(scale, _EPS)


def _gamma_pvalue(
    hsic_stat: float,
    Kx: np.ndarray,
    Ky: np.ndarray,
    n: int,
) -> float | None:
    """Compute p-value using the gamma approximation.

    Returns ``None`` if the approximation is unreliable (shape < 1).

    Parameters
    ----------
    hsic_stat : float
        Observed HSIC statistic.
    Kx : np.ndarray
        Kernel matrix for X.
    Ky : np.ndarray
        Kernel matrix for Y.
    n : int
        Sample size.

    Returns
    -------
    float | None
        p-value, or ``None`` if gamma approximation failed.
    """
    try:
        shape, scale = _gamma_approx_params(Kx, Ky)
        if shape < 0.1 or scale < _EPS:
            return None
        test_val = n * hsic_stat
        p = float(stats.gamma.sf(test_val, a=shape, scale=scale))
        return float(np.clip(p, 0.0, 1.0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Permutation p-value
# ---------------------------------------------------------------------------


def _permutation_pvalue(
    hsic_stat: float,
    Kx: np.ndarray,
    Ky: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    block_size: int | None = None,
) -> float:
    """Compute permutation-based p-value for HSIC.

    Parameters
    ----------
    hsic_stat : float
        Observed HSIC statistic.
    Kx : np.ndarray
        Kernel matrix for X ``(n, n)``.
    Ky : np.ndarray
        Kernel matrix for Y ``(n, n)``.
    n_permutations : int
        Number of permutations.
    rng : np.random.Generator
        Random number generator.
    block_size : int | None
        If set, uses block permutation for time-series data.

    Returns
    -------
    float
        Permutation p-value.
    """
    n = Kx.shape[0]
    Kxc = center_kernel(Kx)
    Kyc = center_kernel(Ky)
    count = 0

    for _ in range(n_permutations):
        perm = _block_permutation(n, block_size, rng)
        Ky_perm = Kyc[np.ix_(perm, perm)]
        null_stat = max(float(np.trace(Kxc @ Ky_perm)) / (n * n), 0.0)
        if null_stat >= hsic_stat:
            count += 1

    return (count + 1) / (n_permutations + 1)


def _block_permutation(
    n: int,
    block_size: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a permutation, optionally using blocks.

    For time-series data, blocks of consecutive indices are permuted
    as units to preserve within-block temporal structure.

    Parameters
    ----------
    n : int
        Number of observations.
    block_size : int | None
        Block size; ``None`` for standard permutation.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Permutation indices.
    """
    if block_size is None or block_size <= 1:
        return rng.permutation(n)

    n_blocks = int(np.ceil(n / block_size))
    block_order = rng.permutation(n_blocks)
    perm = np.concatenate(
        [np.arange(b * block_size, min((b + 1) * block_size, n))
         for b in block_order]
    )
    return perm[:n]


# ---------------------------------------------------------------------------
# Residualisation for conditional HSIC
# ---------------------------------------------------------------------------


def _residualise(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    regularization: float = 1e-5,
) -> np.ndarray:
    """Regress Z out of X using ridge regression.

    Parameters
    ----------
    X : np.ndarray
        Target variable ``(n,)`` or ``(n, d)``.
    Z : np.ndarray
        Conditioning variables ``(n, k)``.
    regularization : float
        Ridge penalty.

    Returns
    -------
    np.ndarray
        Residuals with the same shape as X.
    """
    Z = np.atleast_2d(Z)
    if Z.shape[0] != X.shape[0]:
        return X

    n, k = Z.shape
    Z_aug = np.column_stack([np.ones(n), Z])
    gram = Z_aug.T @ Z_aug + regularization * np.eye(k + 1)
    try:
        beta = np.linalg.solve(gram, Z_aug.T @ X)
        return X - Z_aug @ beta
    except np.linalg.LinAlgError:
        return X


def _kernel_residualise(
    Kx: np.ndarray,
    Kz: np.ndarray,
    *,
    regularization: float = 1e-5,
) -> np.ndarray:
    """Residualise a kernel matrix by projecting out the conditioning kernel.

    Computes ``Kx_res = (I - Kz (Kz + lambda I)^{-1}) Kx
                        (I - Kz (Kz + lambda I)^{-1})^T``

    Parameters
    ----------
    Kx : np.ndarray
        Kernel matrix for X ``(n, n)``.
    Kz : np.ndarray
        Kernel matrix for Z ``(n, n)``.
    regularization : float
        Ridge penalty.

    Returns
    -------
    np.ndarray
        Residualised kernel matrix.
    """
    n = Kx.shape[0]
    R = np.eye(n) - Kz @ np.linalg.solve(
        Kz + regularization * np.eye(n), np.eye(n),
    )
    return R @ Kx @ R.T


# ---------------------------------------------------------------------------
# HSICTest class
# ---------------------------------------------------------------------------


class HSICTest(BaseCITest):
    """HSIC conditional-independence test.

    Tests ``X ⊥ Y | Z`` using the Hilbert-Schmidt Independence Criterion.

    For unconditional testing (``Z = ∅``), computes HSIC(X, Y) directly.
    For conditional testing, residualises both X and Y with respect to Z
    (linear residualisation or kernel residualisation) before computing
    the HSIC.

    The p-value is obtained via:

    1. **Gamma approximation** (fast, O(n²) — default).
    2. **Permutation test** (exact, O(B · n²) — fallback).

    Block permutation is available for dependent (time-series) data.

    Parameters
    ----------
    alpha : float
        Significance level.
    seed : int
        Random seed.
    config : CITestConfig | None
        Base CI test configuration.
    hsic_config : HSICConfig | None
        HSIC-specific configuration.
    cache : KernelCache | None
        Optional kernel matrix cache.
    """

    method = CITestMethod.HSIC

    def __init__(
        self,
        alpha: float = 0.05,
        seed: int = 42,
        config: CITestConfig | None = None,
        hsic_config: HSICConfig | None = None,
        cache: KernelCache | None = None,
    ) -> None:
        super().__init__(alpha=alpha, seed=seed, config=config)
        self.hsic_config = hsic_config or HSICConfig()
        self._cache = cache

    # ------------------------------------------------------------------
    # Kernel matrix construction
    # ------------------------------------------------------------------

    def _build_kernel(
        self,
        X: np.ndarray,
        gamma: float | None = None,
    ) -> np.ndarray:
        """Build the kernel matrix for a single variable.

        Parameters
        ----------
        X : np.ndarray
            Data ``(n, d)``.
        gamma : float | None
            Bandwidth override.

        Returns
        -------
        np.ndarray
            Kernel matrix ``(n, n)``.
        """
        X = np.atleast_2d(X) if X.ndim == 1 else X
        if X.ndim == 1:
            X = X[:, np.newaxis]
        cfg = self.hsic_config

        g = gamma or cfg.gamma

        if cfg.kernel == "rbf":
            return rbf_kernel(X, gamma=g)
        elif cfg.kernel == "polynomial":
            g_eff = g if g is not None else 1.0 / max(X.shape[1], 1)
            return polynomial_kernel(
                X, degree=cfg.degree, gamma=g_eff, coef0=cfg.coef0,
            )
        else:
            return rbf_kernel(X, gamma=g)

    def _select_bandwidth(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> float:
        """Select bandwidth using the configured method.

        Parameters
        ----------
        X : np.ndarray
            First variable.
        Y : np.ndarray
            Second variable.

        Returns
        -------
        float
            Selected bandwidth.
        """
        cfg = self.hsic_config
        if cfg.gamma is not None:
            return cfg.gamma
        if cfg.bandwidth_method == "cv":
            return cross_validation_bandwidth(
                X, Y, seed=self.seed,
            )
        return median_heuristic(np.hstack([
            np.atleast_2d(X) if X.ndim > 1 else X[:, np.newaxis],
            np.atleast_2d(Y) if Y.ndim > 1 else Y[:, np.newaxis],
        ]))

    # ------------------------------------------------------------------
    # Core test logic
    # ------------------------------------------------------------------

    def _compute_hsic(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        gamma: float | None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute the HSIC statistic and return kernel matrices.

        Parameters
        ----------
        X : np.ndarray
            First variable ``(n,)`` or ``(n, d1)``.
        Y : np.ndarray
            Second variable ``(n,)`` or ``(n, d2)``.
        gamma : float | None
            Bandwidth.

        Returns
        -------
        tuple[float, np.ndarray, np.ndarray]
            ``(hsic_value, Kx, Ky)``.
        """
        cfg = self.hsic_config
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]

        n = X.shape[0]

        # Nyström path
        if cfg.nystrom_components is not None and n > cfg.nystrom_components * 2:
            gamma_x = gamma or median_heuristic(X)
            gamma_y = gamma or median_heuristic(Y)
            hsic_val = _hsic_nystrom(
                X, Y, gamma_x, gamma_y,
                cfg.nystrom_components, seed=self.seed,
            )
            Kx = rbf_kernel(X, gamma=gamma_x)
            Ky = rbf_kernel(Y, gamma=gamma_y)
            return hsic_val, Kx, Ky

        Kx = self._build_kernel(X, gamma)
        Ky = self._build_kernel(Y, gamma)
        hsic_val = _biased_hsic(Kx, Ky)
        return hsic_val, Kx, Ky

    def _compute_pvalue(
        self,
        hsic_stat: float,
        Kx: np.ndarray,
        Ky: np.ndarray,
        n: int,
    ) -> float:
        """Compute p-value using gamma approximation or permutation.

        Parameters
        ----------
        hsic_stat : float
            Observed HSIC.
        Kx : np.ndarray
            Kernel matrix for X.
        Ky : np.ndarray
            Kernel matrix for Y.
        n : int
            Sample size.

        Returns
        -------
        float
            p-value in ``[0, 1]``.
        """
        cfg = self.hsic_config

        # Try gamma approximation first
        if cfg.use_gamma_approx and n >= 20:
            p = _gamma_pvalue(hsic_stat, Kx, Ky, n)
            if p is not None:
                return p

        # Fall back to permutation
        rng = np.random.default_rng(self.seed)
        return _permutation_pvalue(
            hsic_stat, Kx, Ky,
            cfg.n_permutations, rng, cfg.block_size,
        )

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
        """Test X ⊥ Y | Z using the HSIC.

        Parameters
        ----------
        x : NodeId
            First variable.
        y : NodeId
            Second variable.
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
        if n < max(self.config.min_samples, _MIN_N):
            return _insufficient_sample_result(
                x, y, conditioning_set, self.method, self.alpha,
            )

        # Residualise if conditioning
        if z_cols is not None:
            x_res = _residualise(x_col, z_cols)
            y_res = _residualise(y_col, z_cols)
        else:
            x_res = x_col
            y_res = y_col

        gamma = self._select_bandwidth(x_res, y_res)
        hsic_stat, Kx, Ky = self._compute_hsic(x_res, y_res, gamma)
        p_value = self._compute_pvalue(hsic_stat, Kx, Ky, n)

        return self._make_result(x, y, conditioning_set, hsic_stat, p_value)

    def test_kernel_residualised(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: pd.DataFrame,
        *,
        regularization: float = 1e-5,
    ) -> CITestResult:
        """Conditional HSIC via kernel residualisation.

        Instead of linear residualisation, projects the kernel matrices
        of X and Y onto the complement of the conditioning kernel space.
        This preserves nonlinear dependencies with Z.

        Parameters
        ----------
        x : NodeId
            First variable.
        y : NodeId
            Second variable.
        conditioning_set : NodeSet
            Conditioning variables.
        data : pd.DataFrame
            Observational data.
        regularization : float
            Ridge penalty for kernel inversion.

        Returns
        -------
        CITestResult
        """
        _validate_inputs(data, x, y, conditioning_set)
        x_col, y_col, z_cols = _extract_columns(data, x, y, conditioning_set)

        n = len(x_col)
        if n < max(self.config.min_samples, _MIN_N):
            return _insufficient_sample_result(
                x, y, conditioning_set, self.method, self.alpha,
            )

        if z_cols is None:
            # No conditioning — standard HSIC
            return self.test(x, y, conditioning_set, data)

        gamma = self._select_bandwidth(x_col, y_col)
        Kx = self._build_kernel(
            x_col[:, np.newaxis] if x_col.ndim == 1 else x_col, gamma,
        )
        Ky = self._build_kernel(
            y_col[:, np.newaxis] if y_col.ndim == 1 else y_col, gamma,
        )
        Kz = block_diagonal_kernel(z_cols, gamma=gamma)

        Kx_res = _kernel_residualise(Kx, Kz, regularization=regularization)
        Ky_res = _kernel_residualise(Ky, Kz, regularization=regularization)

        hsic_stat = _biased_hsic(Kx_res, Ky_res)
        p_value = self._compute_pvalue(hsic_stat, Kx_res, Ky_res, n)

        return self._make_result(x, y, conditioning_set, hsic_stat, p_value)

    def __repr__(self) -> str:  # noqa: D105
        cfg = self.hsic_config
        return (
            f"HSICTest(kernel={cfg.kernel!r}, bw={cfg.bandwidth_method!r}, "
            f"gamma_approx={cfg.use_gamma_approx}, alpha={self.alpha})"
        )
