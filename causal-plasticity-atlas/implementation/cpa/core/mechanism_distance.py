"""
Mechanism Distance Computation (D4-D5 from Theory).

Implements Jensen-Shannon Distance-based mechanism comparison for the
Causal-Plasticity Atlas. The key insight from theory is that sqrt(JSD) is a
proper metric (satisfying triangle inequality), which enables pseudo-metric
properties on DAG alignment distances (Theorem T1).

Core classes:
    - MechanismDistanceComputer: sqrt(JSD) computations for various distribution families
    - DistanceMatrix: K x K pairwise distance storage with clustering and statistics
    - CIFingerprint: Conditional independence fingerprinting for variable comparison

Mathematical foundation:
    JSD(P || Q) = (1/2) KL(P || M) + (1/2) KL(Q || M),   M = (P + Q) / 2
    d_mu(P, Q) = sqrt(JSD(P || Q))  ∈ [0, 1]  (for base-2 log)

References:
    D4: Mechanism identity definition
    D5: Mechanism distance via Jensen-Shannon Divergence
    T1: DAG alignment distance is a pseudo-metric
"""

from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy import stats as sp_stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful imports from sibling modules (may not exist yet)
# ---------------------------------------------------------------------------
try:
    from cpa.core.types import MechanismParams, VariableID
except ImportError:

    @dataclass
    class MechanismParams:
        """Parameters of a linear-Gaussian conditional P(X_i | Pa_i).

        X_i = intercept + coeffs @ Pa_i + eps,   eps ~ N(0, noise_var)
        """

        coeffs: NDArray[np.floating]
        intercept: float = 0.0
        noise_var: float = 1.0

    VariableID = Union[int, str]

try:
    from cpa.core.scm import StructuralCausalModel as _RealSCM

    _HAS_REAL_SCM = True
except ImportError:
    _HAS_REAL_SCM = False

try:
    from cpa.core.mccm import MultiContextCausalModel as _RealMCCM

    _HAS_REAL_MCCM = True
except ImportError:
    _HAS_REAL_MCCM = False


# ---------------------------------------------------------------------------
# Adapter helpers — unified interface for real and stub SCM / MCCM classes
# ---------------------------------------------------------------------------
def _scm_adjacency(scm: Any) -> NDArray[np.floating]:
    """Get adjacency matrix from either real or stub SCM."""
    if hasattr(scm, "adjacency_matrix"):
        return np.asarray(scm.adjacency_matrix, dtype=np.float64)
    return np.asarray(scm.adjacency, dtype=np.float64)


def _scm_n_vars(scm: Any) -> int:
    """Get number of variables from either real or stub SCM."""
    if hasattr(scm, "num_variables"):
        return scm.num_variables
    return scm.n_vars


def _scm_residual_variances(scm: Any) -> NDArray[np.floating]:
    """Get residual/noise variances from either real or stub SCM."""
    if hasattr(scm, "residual_variances"):
        return np.asarray(scm.residual_variances, dtype=np.float64)
    return np.asarray(scm.noise_vars, dtype=np.float64)


def _scm_regression_coefficients(scm: Any) -> NDArray[np.floating]:
    """Get regression coefficient matrix from either real or stub SCM."""
    if hasattr(scm, "regression_coefficients"):
        return np.asarray(scm.regression_coefficients, dtype=np.float64)
    return np.asarray(scm.adjacency, dtype=np.float64)


def _scm_implied_covariance(scm: Any) -> NDArray[np.floating]:
    """Get implied covariance matrix from either real or stub SCM."""
    if hasattr(scm, "implied_covariance"):
        return np.asarray(scm.implied_covariance(), dtype=np.float64)
    if hasattr(scm, "covariance_matrix"):
        return np.asarray(scm.covariance_matrix(), dtype=np.float64)
    # Compute from adjacency and noise
    adj = _scm_adjacency(scm)
    n = adj.shape[0]
    I = np.eye(n)
    B = adj.T
    inv_I_B = la.inv(I - B)
    D = np.diag(_scm_residual_variances(scm))
    return inv_I_B @ D @ inv_I_B.T


def _scm_parents(scm: Any, var_idx: int) -> List[int]:
    """Get parents of a variable."""
    if hasattr(scm, "parents") and callable(scm.parents):
        return list(scm.parents(var_idx))
    # Fall back to adjacency matrix: parent j -> child var_idx iff adj[j, var_idx] != 0
    adj = _scm_adjacency(scm)
    return [int(j) for j in range(adj.shape[0]) if adj[j, var_idx] != 0]


def _scm_markov_blanket(scm: Any, var_idx: int) -> set:
    """Get Markov blanket of a variable."""
    if hasattr(scm, "markov_blanket") and callable(scm.markov_blanket):
        return set(scm.markov_blanket(var_idx))
    # Compute from adjacency: parents + children + co-parents of children
    adj = _scm_adjacency(scm)
    n = adj.shape[0]
    parents = {int(j) for j in range(n) if adj[j, var_idx] != 0}
    children = {int(j) for j in range(n) if adj[var_idx, j] != 0}
    co_parents: set = set()
    for ch in children:
        for j in range(n):
            if adj[j, ch] != 0 and j != var_idx:
                co_parents.add(int(j))
    return (parents | children | co_parents) - {var_idx}


def _scm_mechanism_params(scm: Any, var_idx: int) -> MechanismParams:
    """Extract mechanism parameters for a variable.

    Works with both StructuralCausalModel and stub LinearGaussianSCM.
    """
    if hasattr(scm, "mechanism_params"):
        return scm.mechanism_params(var_idx)
    pa = _scm_parents(scm, var_idx)
    adj = _scm_adjacency(scm)
    coeffs = adj[pa, var_idx] if pa else np.array([], dtype=np.float64)
    residuals = _scm_residual_variances(scm)
    return MechanismParams(
        coeffs=coeffs,
        intercept=0.0,
        noise_var=float(residuals[var_idx]),
    )


def _mccm_context_ids(mccm: Any) -> List[str]:
    """Get context IDs from MCCM."""
    return list(mccm.context_ids)


def _mccm_n_contexts(mccm: Any) -> int:
    """Get number of contexts."""
    if hasattr(mccm, "num_contexts"):
        return mccm.num_contexts
    return mccm.n_contexts


def _mccm_n_vars(mccm: Any) -> int:
    """Get number of variables from MCCM."""
    if hasattr(mccm, "n_vars"):
        return mccm.n_vars
    ctx_ids = _mccm_context_ids(mccm)
    if ctx_ids:
        return _scm_n_vars(mccm.get_scm(ctx_ids[0]))
    return 0


def _mccm_context_pairs(mccm: Any) -> List[Tuple[str, str]]:
    """Get all ordered pairs (c1, c2) with c1 < c2."""
    if hasattr(mccm, "context_pairs"):
        return mccm.context_pairs()
    ids = _mccm_context_ids(mccm)
    return [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOG2 = np.log(2.0)
_EPSILON = 1e-12  # guard against log(0)
_MIN_VARIANCE = 1e-14  # minimum variance for numerical stability
_REGULARIZATION = 1e-10  # covariance regularization


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _ensure_positive(x: float, name: str = "value", floor: float = _MIN_VARIANCE) -> float:
    """Clamp *x* to be at least *floor*, warning if adjustment was needed."""
    if x < floor:
        if x < 0:
            raise ValueError(f"{name} must be non-negative, got {x}")
        warnings.warn(
            f"{name}={x:.2e} is below minimum {floor:.2e}; clamping.",
            stacklevel=3,
        )
        return floor
    return float(x)


def _regularize_covariance(
    Sigma: NDArray[np.floating],
    reg: float = _REGULARIZATION,
) -> NDArray[np.floating]:
    """Add small ridge to ensure positive-definiteness."""
    Sigma = np.array(Sigma, dtype=np.float64)
    n = Sigma.shape[0]
    if n == 0:
        return Sigma
    eigvals = la.eigvalsh(Sigma)
    if eigvals[0] < reg:
        Sigma = Sigma + (reg - min(eigvals[0], 0.0) + reg) * np.eye(n)
    return Sigma


def _symmetrize(M: NDArray[np.floating]) -> NDArray[np.floating]:
    """Force a matrix to be symmetric."""
    return 0.5 * (M + M.T)


# ===================================================================
#  MechanismDistanceComputer
# ===================================================================
class MechanismDistanceComputer:
    """Compute mechanism distances based on sqrt(JSD) (Definition D5).

    The Jensen-Shannon Divergence is defined as:
        JSD(P || Q) = (1/2) KL(P || M) + (1/2) KL(Q || M)
    where M = (P + Q) / 2.

    We use sqrt(JSD) because it is a **proper metric** satisfying the
    triangle inequality (Endres & Schindelin, 2003), which is required
    for Theorem T1 (DAG alignment pseudo-metric).

    Parameters
    ----------
    n_mc_samples : int
        Number of Monte-Carlo samples for intractable JSD integrals.
    seed : int or None
        Random seed for reproducibility.
    regularization : float
        Ridge added to covariance matrices for numerical stability.
    cache_enabled : bool
        Whether to cache pairwise distance computations.
    """

    def __init__(
        self,
        n_mc_samples: int = 50_000,
        seed: Optional[int] = None,
        regularization: float = _REGULARIZATION,
        cache_enabled: bool = True,
    ) -> None:
        if n_mc_samples < 100:
            raise ValueError(f"n_mc_samples must be >= 100, got {n_mc_samples}")
        self.n_mc_samples = n_mc_samples
        self.rng = np.random.default_rng(seed)
        self.regularization = regularization
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _cache_key(self, *args: Any) -> str:
        """Build a hashable cache key from arguments."""
        parts: list[str] = []
        for a in args:
            if isinstance(a, np.ndarray):
                parts.append(a.tobytes().hex()[:32])
            else:
                parts.append(repr(a))
        return "|".join(parts)

    def _get_cached(self, key: str) -> Optional[float]:
        """Return cached value or None."""
        if self.cache_enabled:
            return self._cache.get(key)
        return None

    def _set_cached(self, key: str, value: float) -> None:
        """Store value in cache."""
        if self.cache_enabled:
            self._cache[key] = value

    def clear_cache(self) -> None:
        """Clear the distance cache."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Univariate Gaussian sqrt(JSD)
    # ------------------------------------------------------------------
    def kl_gaussian(
        self,
        mu1: float,
        sigma1_sq: float,
        mu2: float,
        sigma2_sq: float,
    ) -> float:
        """KL divergence KL(N(mu1,sigma1^2) || N(mu2,sigma2^2)).

        Closed-form:
            KL = log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2) / (2 sigma2^2) - 1/2

        Parameters
        ----------
        mu1, mu2 : float
            Means of the two Gaussians.
        sigma1_sq, sigma2_sq : float
            Variances (not standard deviations).

        Returns
        -------
        float
            KL divergence in nats.

        Raises
        ------
        ValueError
            If variances are negative.
        """
        sigma1_sq = _ensure_positive(sigma1_sq, "sigma1_sq")
        sigma2_sq = _ensure_positive(sigma2_sq, "sigma2_sq")

        return (
            0.5 * np.log(sigma2_sq / sigma1_sq)
            + (sigma1_sq + (mu1 - mu2) ** 2) / (2.0 * sigma2_sq)
            - 0.5
        )

    def sqrt_jsd_gaussian(
        self,
        mu1: float,
        sigma1: float,
        mu2: float,
        sigma2: float,
    ) -> float:
        """Closed-form sqrt(JSD) for univariate Gaussians.

        For P = N(mu1, sigma1^2) and Q = N(mu2, sigma2^2), we compute
        JSD(P || Q) via the mixture M = 0.5*(P + Q).

        The mixture of two Gaussians is not Gaussian, so we use the
        **variational upper bound** via moment-matched Gaussian:
            M_approx = N(mu_m, sigma_m^2)
        where mu_m = (mu1 + mu2)/2 and sigma_m^2 = (sigma1^2 + sigma2^2)/2 + (mu1-mu2)^2/4.

        This gives a tight approximation (exact when sigma1 == sigma2).

        Parameters
        ----------
        mu1, mu2 : float
            Means of the two Gaussians.
        sigma1, sigma2 : float
            Standard deviations (not variances).

        Returns
        -------
        float
            sqrt(JSD) in [0, sqrt(ln2)] for natural log, clipped to [0, 1].
        """
        cache_key = self._cache_key("gauss1d", mu1, sigma1, mu2, sigma2)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        s1_sq = _ensure_positive(sigma1 ** 2, "sigma1^2")
        s2_sq = _ensure_positive(sigma2 ** 2, "sigma2^2")

        # Moment-matched mixture parameters
        mu_m = 0.5 * (mu1 + mu2)
        sigma_m_sq = 0.5 * (s1_sq + s2_sq) + 0.25 * (mu1 - mu2) ** 2

        kl_p_m = self.kl_gaussian(mu1, s1_sq, mu_m, sigma_m_sq)
        kl_q_m = self.kl_gaussian(mu2, s2_sq, mu_m, sigma_m_sq)

        jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
        jsd = max(jsd, 0.0)  # numerical guard

        # Normalize to [0, 1] by dividing by ln(2)
        jsd_normalized = jsd / _LOG2
        result = float(np.sqrt(min(jsd_normalized, 1.0)))

        self._set_cached(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Multivariate Gaussian sqrt(JSD)
    # ------------------------------------------------------------------
    def kl_multivariate_gaussian(
        self,
        mu1: NDArray[np.floating],
        Sigma1: NDArray[np.floating],
        mu2: NDArray[np.floating],
        Sigma2: NDArray[np.floating],
    ) -> float:
        """KL divergence KL(N(mu1,Sigma1) || N(mu2,Sigma2)).

        Closed-form:
            KL = 0.5 * [tr(Sigma2^{-1} Sigma1) + (mu2-mu1)^T Sigma2^{-1} (mu2-mu1)
                         - k + ln(det(Sigma2)/det(Sigma1))]

        Parameters
        ----------
        mu1, mu2 : NDArray, shape (k,)
            Mean vectors.
        Sigma1, Sigma2 : NDArray, shape (k, k)
            Covariance matrices.

        Returns
        -------
        float
            KL divergence in nats.
        """
        mu1 = np.asarray(mu1, dtype=np.float64).ravel()
        mu2 = np.asarray(mu2, dtype=np.float64).ravel()
        Sigma1 = _regularize_covariance(np.asarray(Sigma1, dtype=np.float64), self.regularization)
        Sigma2 = _regularize_covariance(np.asarray(Sigma2, dtype=np.float64), self.regularization)

        k = len(mu1)
        if k == 0:
            return 0.0
        if mu1.shape != mu2.shape:
            raise ValueError(
                f"Mean vectors must have same shape: {mu1.shape} vs {mu2.shape}"
            )
        if Sigma1.shape != (k, k) or Sigma2.shape != (k, k):
            raise ValueError(
                f"Covariance matrices must be ({k},{k}), got {Sigma1.shape} and {Sigma2.shape}"
            )

        Sigma1 = _symmetrize(Sigma1)
        Sigma2 = _symmetrize(Sigma2)

        # Cholesky for numerical stability
        try:
            L2 = la.cholesky(Sigma2, lower=True)
        except la.LinAlgError:
            Sigma2 = _regularize_covariance(Sigma2, self.regularization * 100)
            L2 = la.cholesky(Sigma2, lower=True)

        # Sigma2^{-1} @ Sigma1 via solve
        Sigma2_inv_Sigma1 = la.cho_solve((L2, True), Sigma1)
        trace_term = np.trace(Sigma2_inv_Sigma1)

        diff = mu2 - mu1
        Sigma2_inv_diff = la.cho_solve((L2, True), diff)
        quad_term = diff @ Sigma2_inv_diff

        # log det ratio via Cholesky
        log_det_Sigma2 = 2.0 * np.sum(np.log(np.diag(L2)))
        try:
            L1 = la.cholesky(Sigma1, lower=True)
            log_det_Sigma1 = 2.0 * np.sum(np.log(np.diag(L1)))
        except la.LinAlgError:
            sign, log_det_Sigma1 = np.linalg.slogdet(Sigma1)
            if sign <= 0:
                log_det_Sigma1 = -300.0  # near-singular

        kl = 0.5 * (trace_term + quad_term - k + log_det_Sigma2 - log_det_Sigma1)
        return max(float(kl), 0.0)

    def sqrt_jsd_multivariate_gaussian(
        self,
        mu1: NDArray[np.floating],
        Sigma1: NDArray[np.floating],
        mu2: NDArray[np.floating],
        Sigma2: NDArray[np.floating],
    ) -> float:
        """Multivariate Gaussian sqrt(JSD) via moment-matched mixture approximation.

        For P = N(mu1, Sigma1) and Q = N(mu2, Sigma2), the mixture
        M = 0.5*(P+Q) is not Gaussian, so we moment-match:
            mu_m = (mu1 + mu2) / 2
            Sigma_m = (Sigma1 + Sigma2) / 2 + (mu1-mu2)(mu1-mu2)^T / 4

        This gives a tight upper bound on JSD (exact when Sigma1 == Sigma2).

        Parameters
        ----------
        mu1, mu2 : NDArray, shape (k,)
            Mean vectors.
        Sigma1, Sigma2 : NDArray, shape (k, k)
            Covariance matrices.

        Returns
        -------
        float
            sqrt(JSD) in [0, 1].
        """
        mu1 = np.asarray(mu1, dtype=np.float64).ravel()
        mu2 = np.asarray(mu2, dtype=np.float64).ravel()
        Sigma1 = np.asarray(Sigma1, dtype=np.float64)
        Sigma2 = np.asarray(Sigma2, dtype=np.float64)

        cache_key = self._cache_key("gaussMV", mu1, Sigma1, mu2, Sigma2)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        k = len(mu1)
        if k == 0:
            return 0.0

        # Moment-matched mixture
        diff = mu1 - mu2
        mu_m = 0.5 * (mu1 + mu2)
        Sigma_m = 0.5 * (Sigma1 + Sigma2) + 0.25 * np.outer(diff, diff)

        kl_p_m = self.kl_multivariate_gaussian(mu1, Sigma1, mu_m, Sigma_m)
        kl_q_m = self.kl_multivariate_gaussian(mu2, Sigma2, mu_m, Sigma_m)

        jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
        jsd = max(jsd, 0.0)

        jsd_normalized = jsd / _LOG2
        result = float(np.sqrt(min(jsd_normalized, 1.0)))

        self._set_cached(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Monte-Carlo JSD for arbitrary distributions
    # ------------------------------------------------------------------
    def sqrt_jsd_monte_carlo(
        self,
        log_pdf_p: Any,
        log_pdf_q: Any,
        samples_p: NDArray[np.floating],
        samples_q: NDArray[np.floating],
    ) -> float:
        """Monte-Carlo estimate of sqrt(JSD) for arbitrary distributions.

        Uses importance sampling with the mixture as proposal:
            JSD ≈ (1/2) * mean_P[log(p/m)] + (1/2) * mean_Q[log(q/m)]
        where m(x) = 0.5*p(x) + 0.5*q(x).

        Parameters
        ----------
        log_pdf_p : callable
            Log-density of distribution P.
        log_pdf_q : callable
            Log-density of distribution Q.
        samples_p : NDArray, shape (n, d)
            Samples drawn from P.
        samples_q : NDArray, shape (n, d)
            Samples drawn from Q.

        Returns
        -------
        float
            sqrt(JSD) estimate in [0, 1].
        """
        lp_p = log_pdf_p(samples_p)
        lq_p = log_pdf_q(samples_p)
        lm_p = np.logaddexp(lp_p, lq_p) - _LOG2  # log m(x) for x ~ P

        lp_q = log_pdf_p(samples_q)
        lq_q = log_pdf_q(samples_q)
        lm_q = np.logaddexp(lp_q, lq_q) - _LOG2

        kl_p_m = float(np.mean(lp_p - lm_p))
        kl_q_m = float(np.mean(lq_q - lm_q))

        jsd = max(0.5 * kl_p_m + 0.5 * kl_q_m, 0.0)
        jsd_normalized = jsd / _LOG2
        return float(np.sqrt(min(jsd_normalized, 1.0)))

    # ------------------------------------------------------------------
    # Conditional sqrt(JSD) for linear-Gaussian mechanisms
    # ------------------------------------------------------------------
    def sqrt_jsd_conditional(
        self,
        coeffs1: NDArray[np.floating],
        var1: float,
        coeffs2: NDArray[np.floating],
        var2: float,
        parent_cov: Optional[NDArray[np.floating]] = None,
        intercept1: float = 0.0,
        intercept2: float = 0.0,
    ) -> float:
        """Sqrt(JSD) between conditional distributions P(X | Pa).

        For linear-Gaussian conditionals:
            P1: X = intercept1 + coeffs1 @ Pa + eps1,  eps1 ~ N(0, var1)
            P2: X = intercept2 + coeffs2 @ Pa + eps2,  eps2 ~ N(0, var2)

        The marginal (unconditional) distributions are:
            P1(X) = N(intercept1 + coeffs1 @ mu_Pa, coeffs1^T Sigma_Pa coeffs1 + var1)
            P2(X) = N(intercept2 + coeffs2 @ mu_Pa, coeffs2^T Sigma_Pa coeffs2 + var2)

        When parent_cov is None, we compute the *average conditional JSD*
        as if conditioning on fixed parent values (which reduces to noise-only JSD
        when coefficients match, or uses Monte Carlo integration otherwise).

        Parameters
        ----------
        coeffs1, coeffs2 : NDArray, shape (d,)
            Regression coefficients for each mechanism.
        var1, var2 : float
            Noise variances.
        parent_cov : NDArray or None, shape (d, d)
            Covariance of parent variables. If None, uses identity.
        intercept1, intercept2 : float
            Intercepts.

        Returns
        -------
        float
            sqrt(JSD) in [0, 1].
        """
        coeffs1 = np.asarray(coeffs1, dtype=np.float64).ravel()
        coeffs2 = np.asarray(coeffs2, dtype=np.float64).ravel()

        cache_key = self._cache_key(
            "cond", coeffs1, var1, coeffs2, var2,
            parent_cov if parent_cov is not None else "none",
            intercept1, intercept2,
        )
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        var1 = _ensure_positive(var1, "var1")
        var2 = _ensure_positive(var2, "var2")

        d1 = len(coeffs1)
        d2 = len(coeffs2)

        # Different parent-set sizes: max-dimensionalize by zero-padding
        d = max(d1, d2)
        if d == 0:
            # No parents: just compare noise distributions
            result = self.sqrt_jsd_gaussian(intercept1, np.sqrt(var1), intercept2, np.sqrt(var2))
            self._set_cached(cache_key, result)
            return result

        c1 = np.zeros(d, dtype=np.float64)
        c2 = np.zeros(d, dtype=np.float64)
        c1[:d1] = coeffs1
        c2[:d2] = coeffs2

        if parent_cov is None:
            Sigma_pa = np.eye(d, dtype=np.float64)
            mu_pa = np.zeros(d, dtype=np.float64)
        else:
            Sigma_pa = np.asarray(parent_cov, dtype=np.float64)
            # Pad if necessary
            if Sigma_pa.shape[0] < d:
                new_cov = np.eye(d, dtype=np.float64)
                old_d = Sigma_pa.shape[0]
                new_cov[:old_d, :old_d] = Sigma_pa
                Sigma_pa = new_cov
            Sigma_pa = _regularize_covariance(Sigma_pa, self.regularization)
            mu_pa = np.zeros(d, dtype=np.float64)

        # Marginal distribution of X under each mechanism
        mu_x1 = intercept1 + c1 @ mu_pa
        mu_x2 = intercept2 + c2 @ mu_pa
        var_x1 = float(c1 @ Sigma_pa @ c1) + var1
        var_x2 = float(c2 @ Sigma_pa @ c2) + var2

        result = self.sqrt_jsd_gaussian(mu_x1, np.sqrt(var_x1), mu_x2, np.sqrt(var_x2))
        self._set_cached(cache_key, result)
        return result

    def sqrt_jsd_conditional_mc(
        self,
        coeffs1: NDArray[np.floating],
        var1: float,
        coeffs2: NDArray[np.floating],
        var2: float,
        parent_samples: NDArray[np.floating],
        intercept1: float = 0.0,
        intercept2: float = 0.0,
    ) -> float:
        """Monte-Carlo sqrt(JSD) for conditionals by averaging over parent values.

        For each parent configuration pa_j:
            JSD_j = JSD(N(c1@pa_j + i1, v1) || N(c2@pa_j + i2, v2))
        Then average:
            JSD_avg = mean_j(JSD_j)

        This is more accurate than the marginal approach when parent distributions
        are far from Gaussian.

        Parameters
        ----------
        coeffs1, coeffs2 : NDArray, shape (d,)
            Regression coefficients.
        var1, var2 : float
            Noise variances.
        parent_samples : NDArray, shape (n_samples, d)
            Observed parent configurations.
        intercept1, intercept2 : float
            Intercepts.

        Returns
        -------
        float
            sqrt(averaged JSD) in [0, 1].
        """
        coeffs1 = np.asarray(coeffs1, dtype=np.float64).ravel()
        coeffs2 = np.asarray(coeffs2, dtype=np.float64).ravel()
        parent_samples = np.asarray(parent_samples, dtype=np.float64)
        if parent_samples.ndim == 1:
            parent_samples = parent_samples.reshape(-1, 1)

        var1 = _ensure_positive(var1, "var1")
        var2 = _ensure_positive(var2, "var2")

        n_samples = parent_samples.shape[0]
        if n_samples == 0:
            return self.sqrt_jsd_gaussian(intercept1, np.sqrt(var1), intercept2, np.sqrt(var2))

        # Pad coefficients to match parent dimension
        d = parent_samples.shape[1]
        c1 = np.zeros(d, dtype=np.float64)
        c2 = np.zeros(d, dtype=np.float64)
        c1[: len(coeffs1)] = coeffs1
        c2[: len(coeffs2)] = coeffs2

        # Vectorized: compute conditional means for all parent samples
        mu1_all = parent_samples @ c1 + intercept1  # (n_samples,)
        mu2_all = parent_samples @ c2 + intercept2

        sd1 = np.sqrt(var1)
        sd2 = np.sqrt(var2)

        # Compute JSD for each parent configuration
        jsds = np.zeros(n_samples)
        for j in range(n_samples):
            d_j = self.sqrt_jsd_gaussian(mu1_all[j], sd1, mu2_all[j], sd2)
            jsds[j] = d_j ** 2  # square since we'll sqrt the average

        avg_jsd = float(np.mean(jsds))
        return float(np.sqrt(max(avg_jsd, 0.0)))

    # ------------------------------------------------------------------
    # Multi-distribution JSD (K-way)
    # ------------------------------------------------------------------
    def multi_distribution_jsd(
        self,
        distributions: List[Tuple[NDArray[np.floating], NDArray[np.floating]]],
        weights: Optional[NDArray[np.floating]] = None,
    ) -> float:
        """Generalized JSD over K distributions (not just pairwise).

        JSD_pi(P_1, ..., P_K) = H(sum_k pi_k P_k) - sum_k pi_k H(P_k)

        For Gaussians N(mu_k, Sigma_k):
            H(P_k) = 0.5 * ln((2*pi*e)^d * det(Sigma_k))

        The mixture entropy H(M) is computed via moment-matching.

        Parameters
        ----------
        distributions : list of (mu, Sigma) tuples
            Each entry is (mean_vector, covariance_matrix).
        weights : NDArray or None
            Mixing weights (must sum to 1). If None, uniform.

        Returns
        -------
        float
            sqrt(JSD) in [0, 1].
        """
        K = len(distributions)
        if K == 0:
            return 0.0
        if K == 1:
            return 0.0

        if weights is None:
            w = np.ones(K) / K
        else:
            w = np.asarray(weights, dtype=np.float64)
            if abs(w.sum() - 1.0) > 1e-6:
                raise ValueError(f"Weights must sum to 1, got {w.sum()}")

        mus = [np.asarray(d[0], dtype=np.float64).ravel() for d in distributions]
        Sigmas = [np.asarray(d[1], dtype=np.float64) for d in distributions]

        d = len(mus[0])
        for i, (mu, Sigma) in enumerate(zip(mus, Sigmas)):
            if len(mu) != d:
                raise ValueError(f"All means must have same dimension, dist {i} has {len(mu)}")
            if Sigma.ndim == 0:
                Sigmas[i] = np.array([[float(Sigma)]])
            Sigmas[i] = _regularize_covariance(Sigmas[i], self.regularization)

        if d == 0:
            return 0.0

        # Entropy of each component: H(P_k) = 0.5 * d * ln(2*pi*e) + 0.5 * ln(det(Sigma_k))
        component_entropies = np.zeros(K)
        for k in range(K):
            sign, logdet = np.linalg.slogdet(Sigmas[k])
            if sign <= 0:
                Sigmas[k] = _regularize_covariance(Sigmas[k], self.regularization * 100)
                sign, logdet = np.linalg.slogdet(Sigmas[k])
            component_entropies[k] = 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * logdet

        # Moment-matched mixture: mu_m = sum w_k mu_k
        mu_m = sum(w[k] * mus[k] for k in range(K))
        Sigma_m = np.zeros((d, d), dtype=np.float64)
        for k in range(K):
            diff = mus[k] - mu_m
            Sigma_m += w[k] * (Sigmas[k] + np.outer(diff, diff))
        Sigma_m = _regularize_covariance(Sigma_m, self.regularization)

        sign_m, logdet_m = np.linalg.slogdet(Sigma_m)
        if sign_m <= 0:
            Sigma_m = _regularize_covariance(Sigma_m, self.regularization * 100)
            sign_m, logdet_m = np.linalg.slogdet(Sigma_m)
        mixture_entropy = 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * logdet_m

        # JSD = H(mixture) - weighted sum of component entropies
        jsd = mixture_entropy - float(w @ component_entropies)
        jsd = max(jsd, 0.0)

        # Normalize by ln(K) to get [0,1] for K distributions
        # (maximum JSD for K distributions is ln(K))
        max_jsd = np.log(K)
        if max_jsd > 0:
            jsd_normalized = jsd / max_jsd
        else:
            jsd_normalized = 0.0

        return float(np.sqrt(min(jsd_normalized, 1.0)))

    # ------------------------------------------------------------------
    # Mechanism identity test
    # ------------------------------------------------------------------
    def mechanism_identity_test(
        self,
        scm1: Any,
        scm2: Any,
        var_idx: int,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]:
        """Statistical test for mechanism identity (D4).

        Tests H0: mechanism for var_idx is identical across scm1 and scm2.
        Uses the coefficient comparison approach:
          - Compare regression coefficients via Wald test
          - Compare noise variances via F-test
          - Combine with Bonferroni correction

        Parameters
        ----------
        scm1, scm2 : StructuralCausalModel (or compatible)
            The two SCMs to compare.
        var_idx : int
            Variable index to test.
        alpha : float
            Significance level.
        n_bootstrap : int
            Bootstrap iterations for confidence intervals.

        Returns
        -------
        dict with keys:
            - 'identical': bool -- True if we fail to reject H0
            - 'p_value': float -- combined p-value
            - 'distance': float -- sqrt(JSD) distance
            - 'coeff_p_value': float -- p-value for coefficient test
            - 'var_p_value': float -- p-value for variance test
            - 'details': dict -- additional test details
        """
        n1 = _scm_n_vars(scm1)
        n2 = _scm_n_vars(scm2)
        if var_idx < 0 or var_idx >= n1:
            raise ValueError(f"var_idx {var_idx} out of range [0, {n1})")
        if var_idx >= n2:
            raise ValueError(f"var_idx {var_idx} out of range for scm2 [0, {n2})")

        params1 = _scm_mechanism_params(scm1, var_idx)
        params2 = _scm_mechanism_params(scm2, var_idx)

        pa1 = _scm_parents(scm1, var_idx)
        pa2 = _scm_parents(scm2, var_idx)

        # Structural check: different parent sets => definitely different mechanisms
        if set(pa1) != set(pa2):
            distance = self.sqrt_jsd_conditional(
                params1.coeffs, params1.noise_var,
                params2.coeffs, params2.noise_var,
            )
            return {
                "identical": False,
                "p_value": 0.0,
                "distance": distance,
                "coeff_p_value": 0.0,
                "var_p_value": 0.0,
                "details": {
                    "reason": "different_parent_sets",
                    "parents1": pa1,
                    "parents2": pa2,
                },
            }

        # Coefficient comparison via Euclidean distance and chi-squared test
        c1 = params1.coeffs
        c2 = params2.coeffs
        d = len(c1)

        if d > 0:
            coeff_diff = c1 - c2
            coeff_dist_sq = float(coeff_diff @ coeff_diff)
            # Under H0, coeff_diff ~ N(0, Sigma_diff)
            # Approximate Sigma_diff using noise variances and assumed unit parent covariance
            sigma_diff_diag = (params1.noise_var + params2.noise_var) * np.ones(d)
            chi2_stat = coeff_dist_sq / float(np.mean(sigma_diff_diag))
            coeff_p_value = float(1.0 - sp_stats.chi2.cdf(chi2_stat, df=d))
        else:
            coeff_p_value = 1.0

        # Variance comparison via F-test
        v1 = params1.noise_var
        v2 = params2.noise_var
        f_stat = max(v1, v2) / max(min(v1, v2), _MIN_VARIANCE)
        # Approximate df; in practice, these would come from sample sizes
        df = 100  # placeholder
        var_p_value = float(2.0 * min(
            sp_stats.f.cdf(f_stat, df, df),
            1.0 - sp_stats.f.cdf(f_stat, df, df),
        ))

        # Bonferroni correction
        combined_p = min(1.0, 2.0 * min(coeff_p_value, var_p_value))

        # Compute sqrt(JSD) distance
        distance = self.sqrt_jsd_conditional(
            params1.coeffs, params1.noise_var,
            params2.coeffs, params2.noise_var,
            intercept1=params1.intercept,
            intercept2=params2.intercept,
        )

        # Bootstrap confidence interval for distance
        boot_distances = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            noise_c1 = c1 + self.rng.normal(0, np.sqrt(params1.noise_var / df), size=d) if d > 0 else c1
            noise_c2 = c2 + self.rng.normal(0, np.sqrt(params2.noise_var / df), size=d) if d > 0 else c2
            noise_v1 = v1 * self.rng.chisquare(df) / df
            noise_v2 = v2 * self.rng.chisquare(df) / df
            boot_distances[b] = self.sqrt_jsd_conditional(
                noise_c1, noise_v1, noise_c2, noise_v2,
            )

        ci_low = float(np.percentile(boot_distances, 2.5))
        ci_high = float(np.percentile(boot_distances, 97.5))

        return {
            "identical": combined_p > alpha,
            "p_value": combined_p,
            "distance": distance,
            "coeff_p_value": coeff_p_value,
            "var_p_value": var_p_value,
            "details": {
                "reason": "statistical_test",
                "f_stat": f_stat,
                "coeff_dist_sq": coeff_dist_sq if d > 0 else 0.0,
                "bootstrap_ci": (ci_low, ci_high),
                "n_bootstrap": n_bootstrap,
            },
        }

    # ------------------------------------------------------------------
    # Pairwise mechanism distances across contexts
    # ------------------------------------------------------------------
    def pairwise_mechanism_distances(
        self,
        mccm: Any,
        var_idx: int,
        alignment: Optional[Dict[str, Dict[int, Optional[int]]]] = None,
    ) -> DistanceMatrix:
        """Compute all K*(K-1)/2 pairwise sqrt(JSD) distances for *var_idx*.

        Parameters
        ----------
        mccm : MultiContextCausalModel (or compatible)
            Multi-context causal model.
        var_idx : int
            Variable index to compute distances for.
        alignment : dict or None
            Optional per-context variable alignment. Maps context_id -> {var_idx -> aligned_var_idx}.
            If None, identity alignment is assumed.

        Returns
        -------
        DistanceMatrix
            K x K symmetric distance matrix.
        """
        K = _mccm_n_contexts(mccm)
        ctx_ids = _mccm_context_ids(mccm)

        distances = np.zeros((K, K), dtype=np.float64)
        for i in range(K):
            for j in range(i + 1, K):
                scm_i = mccm.get_scm(ctx_ids[i])
                scm_j = mccm.get_scm(ctx_ids[j])

                # Resolve alignment
                vi = var_idx
                vj = var_idx
                if alignment is not None:
                    if ctx_ids[j] in alignment:
                        mapped = alignment[ctx_ids[j]].get(var_idx)
                        if mapped is None:
                            distances[i, j] = distances[j, i] = 1.0  # unaligned => max distance
                            continue
                        vj = mapped

                params_i = _scm_mechanism_params(scm_i, vi)
                params_j = _scm_mechanism_params(scm_j, vj)

                d = self.sqrt_jsd_conditional(
                    params_i.coeffs,
                    params_i.noise_var,
                    params_j.coeffs,
                    params_j.noise_var,
                    intercept1=params_i.intercept,
                    intercept2=params_j.intercept,
                )
                distances[i, j] = d
                distances[j, i] = d

        return DistanceMatrix(distances, labels=ctx_ids, variable_idx=var_idx)

    def batch_mechanism_distances(
        self,
        mccm: Any,
        var_indices: Optional[List[int]] = None,
        alignment: Optional[Dict[str, Dict[int, Optional[int]]]] = None,
    ) -> Dict[int, DistanceMatrix]:
        """Compute pairwise distances for all variables across all contexts.

        Parameters
        ----------
        mccm : MultiContextCausalModel (or compatible)
            Multi-context causal model.
        var_indices : list of int or None
            Variables to compute distances for. If None, all variables.
        alignment : dict or None
            Variable alignment mapping.

        Returns
        -------
        dict mapping var_idx -> DistanceMatrix
        """
        if var_indices is None:
            var_indices = list(range(_mccm_n_vars(mccm)))

        result: Dict[int, DistanceMatrix] = {}
        for vi in var_indices:
            logger.debug("Computing distances for variable %d", vi)
            result[vi] = self.pairwise_mechanism_distances(mccm, vi, alignment)

        return result

    # ------------------------------------------------------------------
    # Structural divergence
    # ------------------------------------------------------------------
    def structural_divergence(
        self,
        adj1: NDArray[np.floating],
        adj2: NDArray[np.floating],
        alignment: Optional[Dict[int, Optional[int]]] = None,
        w_addition: float = 1.0,
        w_deletion: float = 1.0,
        w_reversal: float = 0.5,
    ) -> float:
        """Weighted Structural Hamming Distance between two DAGs.

        Classifies edge differences as:
            - Addition: edge present in adj2 but not adj1 (weight w_addition)
            - Deletion: edge present in adj1 but not adj2 (weight w_deletion)
            - Reversal: edge i->j in one, j->i in other (weight w_reversal)

        Parameters
        ----------
        adj1, adj2 : NDArray, shape (n, n)
            Adjacency matrices (adj[i,j] != 0 means i->j).
        alignment : dict or None
            Maps variable indices in adj2 to indices in adj1.
            If None, identity alignment.
        w_addition : float
            Weight for edge additions.
        w_deletion : float
            Weight for edge deletions.
        w_reversal : float
            Weight for edge reversals.

        Returns
        -------
        float
            Weighted SHD, normalized by n*(n-1)/2.
        """
        adj1 = np.asarray(adj1, dtype=np.float64)
        adj2 = np.asarray(adj2, dtype=np.float64)

        n1 = adj1.shape[0]
        n2 = adj2.shape[0]
        n = max(n1, n2)

        if n == 0:
            return 0.0

        # Pad to same size
        if n1 < n:
            padded = np.zeros((n, n))
            padded[:n1, :n1] = adj1
            adj1 = padded
        if n2 < n:
            padded = np.zeros((n, n))
            padded[:n2, :n2] = adj2
            adj2 = padded

        # Apply alignment to adj2
        if alignment is not None:
            aligned_adj2 = np.zeros((n, n))
            for i2 in range(n):
                for j2 in range(n):
                    if adj2[i2, j2] != 0:
                        i1 = alignment.get(i2, i2)
                        j1 = alignment.get(j2, j2)
                        if i1 is not None and j1 is not None:
                            if 0 <= i1 < n and 0 <= j1 < n:
                                aligned_adj2[i1, j1] = adj2[i2, j2]
            adj2 = aligned_adj2

        # Compute binary edge presence
        e1 = (adj1 != 0).astype(int)
        e2 = (adj2 != 0).astype(int)

        weighted_shd = 0.0
        counted = set()

        for i in range(n):
            for j in range(n):
                if (i, j) in counted or (j, i) in counted:
                    continue
                if i == j:
                    continue

                e1_ij = e1[i, j]
                e1_ji = e1[j, i]
                e2_ij = e2[i, j]
                e2_ji = e2[j, i]

                # Check for reversal: i->j in one, j->i in other
                if e1_ij and not e1_ji and not e2_ij and e2_ji:
                    weighted_shd += w_reversal
                    counted.add((i, j))
                elif not e1_ij and e1_ji and e2_ij and not e2_ji:
                    weighted_shd += w_reversal
                    counted.add((i, j))
                else:
                    # Check additions and deletions for i->j
                    if e1_ij and not e2_ij:
                        weighted_shd += w_deletion
                    elif not e1_ij and e2_ij:
                        weighted_shd += w_addition

                    # Check for j->i
                    if e1_ji and not e2_ji:
                        weighted_shd += w_deletion
                    elif not e1_ji and e2_ji:
                        weighted_shd += w_addition

                    counted.add((i, j))

        # Normalize by n*(n-1)/2
        max_edges = n * (n - 1) / 2.0
        if max_edges > 0:
            return weighted_shd / max_edges
        return 0.0

    # ------------------------------------------------------------------
    # Combined mechanism divergence
    # ------------------------------------------------------------------
    def combined_mechanism_divergence(
        self,
        scm1: Any,
        scm2: Any,
        alignment: Optional[Dict[int, Optional[int]]] = None,
        structural_weight: float = 0.5,
        parametric_weight: float = 0.5,
        var_indices: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Combined structural + parametric mechanism divergence (D6).

        d_DAG(G1, G2) = lambda_s * structural_div + lambda_m * parametric_div

        Parameters
        ----------
        scm1, scm2 : StructuralCausalModel (or compatible)
            The two SCMs to compare.
        alignment : dict or None
            Variable alignment mapping i -> j (from scm2 to scm1).
        structural_weight : float
            Weight for structural divergence (lambda_s).
        parametric_weight : float
            Weight for parametric divergence (lambda_m).
        var_indices : list of int or None
            Which variables to compare. If None, all.

        Returns
        -------
        dict with keys:
            - 'combined': float -- overall divergence
            - 'structural': float -- structural divergence
            - 'parametric': float -- mean parametric divergence
            - 'per_variable': dict mapping var_idx -> parametric distance
        """
        if abs(structural_weight + parametric_weight - 1.0) > 1e-6:
            warnings.warn("structural_weight + parametric_weight != 1.0; they will not be re-normalized.")

        # Structural divergence
        adj1 = _scm_adjacency(scm1)
        adj2 = _scm_adjacency(scm2)
        struct_div = self.structural_divergence(adj1, adj2, alignment)

        # Parametric divergence per variable
        n1 = _scm_n_vars(scm1)
        n2 = _scm_n_vars(scm2)
        if var_indices is None:
            var_indices = list(range(min(n1, n2)))

        per_var: Dict[int, float] = {}
        for vi in var_indices:
            vj = vi
            if alignment is not None:
                mapped = alignment.get(vi)
                if mapped is None:
                    per_var[vi] = 1.0  # unaligned
                    continue
                vj = mapped

            if vj >= n2:
                per_var[vi] = 1.0
                continue

            params1 = _scm_mechanism_params(scm1, vi)
            params2 = _scm_mechanism_params(scm2, vj)
            per_var[vi] = self.sqrt_jsd_conditional(
                params1.coeffs, params1.noise_var,
                params2.coeffs, params2.noise_var,
                intercept1=params1.intercept,
                intercept2=params2.intercept,
            )

        param_div = float(np.mean(list(per_var.values()))) if per_var else 0.0

        combined = structural_weight * struct_div + parametric_weight * param_div

        return {
            "combined": combined,
            "structural": struct_div,
            "parametric": param_div,
            "per_variable": per_var,
        }


# ===================================================================
#  DistanceMatrix
# ===================================================================
class DistanceMatrix:
    """K x K pairwise distance matrix with statistics and clustering.

    Stores symmetric pairwise distances between K contexts for a single
    variable's mechanisms.

    Parameters
    ----------
    matrix : NDArray, shape (K, K)
        Symmetric distance matrix with zeros on diagonal.
    labels : list of str or None
        Context labels for rows/columns.
    variable_idx : int or None
        Index of the variable these distances are for.
    """

    def __init__(
        self,
        matrix: NDArray[np.floating],
        labels: Optional[List[str]] = None,
        variable_idx: Optional[int] = None,
    ) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)

        if self.matrix.ndim != 2:
            raise ValueError(f"Distance matrix must be 2D, got {self.matrix.ndim}D")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"Distance matrix must be square, got {self.matrix.shape}")

        self.K = self.matrix.shape[0]
        self.labels = labels if labels is not None else [str(i) for i in range(self.K)]
        self.variable_idx = variable_idx

        if len(self.labels) != self.K:
            raise ValueError(f"Expected {self.K} labels, got {len(self.labels)}")

    @property
    def upper_triangle(self) -> NDArray[np.floating]:
        """Return upper-triangular values (K*(K-1)/2 pairwise distances)."""
        return self.matrix[np.triu_indices(self.K, k=1)]

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    def mean(self) -> float:
        """Mean of all pairwise distances."""
        vals = self.upper_triangle
        return float(np.mean(vals)) if len(vals) > 0 else 0.0

    def max(self) -> float:
        """Maximum pairwise distance."""
        vals = self.upper_triangle
        return float(np.max(vals)) if len(vals) > 0 else 0.0

    def min(self) -> float:
        """Minimum pairwise distance."""
        vals = self.upper_triangle
        return float(np.min(vals)) if len(vals) > 0 else 0.0

    def median(self) -> float:
        """Median pairwise distance."""
        vals = self.upper_triangle
        return float(np.median(vals)) if len(vals) > 0 else 0.0

    def std(self) -> float:
        """Standard deviation of pairwise distances."""
        vals = self.upper_triangle
        return float(np.std(vals)) if len(vals) > 0 else 0.0

    def percentile(self, q: float) -> float:
        """Percentile of pairwise distances.

        Parameters
        ----------
        q : float
            Percentile in [0, 100].
        """
        vals = self.upper_triangle
        return float(np.percentile(vals, q)) if len(vals) > 0 else 0.0

    def threshold_count(self, threshold: float) -> int:
        """Count pairwise distances exceeding *threshold*.

        Parameters
        ----------
        threshold : float
            Distance threshold.

        Returns
        -------
        int
            Number of pairs with distance > threshold.
        """
        vals = self.upper_triangle
        return int(np.sum(vals > threshold))

    def summary(self) -> Dict[str, float]:
        """Return a summary dict of all statistics."""
        return {
            "mean": self.mean(),
            "median": self.median(),
            "std": self.std(),
            "min": self.min(),
            "max": self.max(),
            "p25": self.percentile(25),
            "p75": self.percentile(75),
            "p95": self.percentile(95),
            "n_pairs": len(self.upper_triangle),
        }

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    def hierarchical_clustering(
        self,
        n_clusters: Optional[int] = None,
        threshold: Optional[float] = None,
        method: str = "average",
    ) -> NDArray[np.int_]:
        """Hierarchical clustering of contexts.

        Parameters
        ----------
        n_clusters : int or None
            Number of clusters. If None, use threshold.
        threshold : float or None
            Distance threshold for flat clusters. Used if n_clusters is None.
        method : str
            Linkage method: 'single', 'complete', 'average', 'ward'.

        Returns
        -------
        NDArray of shape (K,)
            Cluster labels for each context.
        """
        if self.K <= 1:
            return np.zeros(self.K, dtype=int)

        condensed = squareform(self.matrix, checks=False)
        # Ensure non-negative
        condensed = np.maximum(condensed, 0.0)

        Z = linkage(condensed, method=method)

        if n_clusters is not None:
            labels = fcluster(Z, n_clusters, criterion="maxclust")
        elif threshold is not None:
            labels = fcluster(Z, threshold, criterion="distance")
        else:
            # Default: use elbow heuristic
            max_d = float(np.max(Z[:, 2])) if len(Z) > 0 else 1.0
            labels = fcluster(Z, 0.7 * max_d, criterion="distance")

        return labels.astype(int)

    def dbscan_clustering(
        self,
        eps: float = 0.3,
        min_samples: int = 2,
    ) -> NDArray[np.int_]:
        """DBSCAN clustering using precomputed distances.

        Parameters
        ----------
        eps : float
            Maximum distance for neighbourhood.
        min_samples : int
            Minimum cluster size.

        Returns
        -------
        NDArray of shape (K,)
            Cluster labels (-1 for noise).
        """
        from sklearn.cluster import DBSCAN

        db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        return db.fit_predict(self.matrix).astype(int)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def distance(self, label1: str, label2: str) -> float:
        """Get distance between two named contexts.

        Parameters
        ----------
        label1, label2 : str
            Context labels.

        Returns
        -------
        float
            Pairwise distance.
        """
        i = self.labels.index(label1)
        j = self.labels.index(label2)
        return float(self.matrix[i, j])

    def nearest_neighbor(self, label: str) -> Tuple[str, float]:
        """Find the nearest context to *label*.

        Returns
        -------
        (nearest_label, distance) tuple.
        """
        i = self.labels.index(label)
        row = self.matrix[i].copy()
        row[i] = np.inf
        j = int(np.argmin(row))
        return self.labels[j], float(row[j])

    def k_nearest_neighbors(self, label: str, k: int = 3) -> List[Tuple[str, float]]:
        """Find the k nearest contexts to *label*.

        Parameters
        ----------
        label : str
            Query context.
        k : int
            Number of neighbors.

        Returns
        -------
        list of (label, distance) tuples, sorted by distance.
        """
        i = self.labels.index(label)
        row = self.matrix[i].copy()
        row[i] = np.inf
        indices = np.argsort(row)[:k]
        return [(self.labels[int(j)], float(row[j])) for j in indices]

    # ------------------------------------------------------------------
    # Visualization support
    # ------------------------------------------------------------------
    def heatmap_data(self) -> Dict[str, Any]:
        """Return data for heatmap visualization.

        Returns
        -------
        dict with keys 'matrix', 'labels', 'variable_idx'.
        """
        return {
            "matrix": self.matrix.tolist(),
            "labels": self.labels,
            "variable_idx": self.variable_idx,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "matrix": self.matrix.tolist(),
            "labels": self.labels,
            "variable_idx": self.variable_idx,
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DistanceMatrix:
        """Deserialize from dict."""
        return cls(
            matrix=np.array(d["matrix"]),
            labels=d.get("labels"),
            variable_idx=d.get("variable_idx"),
        )

    def __repr__(self) -> str:
        return (
            f"DistanceMatrix(K={self.K}, var={self.variable_idx}, "
            f"mean={self.mean():.4f}, max={self.max():.4f})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DistanceMatrix):
            return NotImplemented
        return np.allclose(self.matrix, other.matrix) and self.labels == other.labels

    # ------------------------------------------------------------------
    # Submatrix and filtering
    # ------------------------------------------------------------------
    def submatrix(self, labels: List[str]) -> DistanceMatrix:
        """Extract a submatrix for a subset of contexts.

        Parameters
        ----------
        labels : list of str
            Context labels to keep.

        Returns
        -------
        DistanceMatrix
            Submatrix.
        """
        indices = [self.labels.index(l) for l in labels]
        sub = self.matrix[np.ix_(indices, indices)]
        return DistanceMatrix(sub, labels=labels, variable_idx=self.variable_idx)

    def filter_by_threshold(self, threshold: float) -> List[Tuple[str, str, float]]:
        """Return all pairs with distance > threshold.

        Returns
        -------
        list of (label1, label2, distance) tuples.
        """
        result = []
        for i in range(self.K):
            for j in range(i + 1, self.K):
                d = self.matrix[i, j]
                if d > threshold:
                    result.append((self.labels[i], self.labels[j], float(d)))
        return sorted(result, key=lambda x: -x[2])


# ===================================================================
#  CIFingerprint
# ===================================================================
class CIFingerprint:
    """Conditional Independence Fingerprint for a variable.

    A CI-fingerprint encodes the partial correlation structure of a variable
    with respect to all other variables. Two variables from different contexts
    that have similar CI-fingerprints are likely to represent the same
    underlying causal mechanism.

    The fingerprint is a vector of partial correlations:
        fp[j] = rho(X_i, X_j | X_{-i,-j})
    for all j != i.

    Parameters
    ----------
    variable_idx : int
        Index of the variable.
    context_id : str
        Context identifier.
    partial_correlations : NDArray
        Vector of partial correlations with all other variables.
    conditioning_sets : list of frozenset
        The conditioning set used for each partial correlation.
    variable_labels : list of str or None
        Labels for the variables.
    """

    def __init__(
        self,
        variable_idx: int,
        context_id: str,
        partial_correlations: NDArray[np.floating],
        conditioning_sets: Optional[List[frozenset]] = None,
        variable_labels: Optional[List[str]] = None,
    ) -> None:
        self.variable_idx = variable_idx
        self.context_id = context_id
        self.partial_correlations = np.asarray(partial_correlations, dtype=np.float64)
        self.conditioning_sets = conditioning_sets
        self.variable_labels = variable_labels
        self.n_vars = len(self.partial_correlations)

    @classmethod
    def from_covariance(
        cls,
        var_idx: int,
        context_id: str,
        cov_matrix: NDArray[np.floating],
        variable_labels: Optional[List[str]] = None,
    ) -> CIFingerprint:
        """Compute CI-fingerprint from a covariance matrix.

        Uses the precision matrix (inverse covariance) to get partial correlations:
            rho(X_i, X_j | X_{-i,-j}) = -Theta[i,j] / sqrt(Theta[i,i] * Theta[j,j])

        Parameters
        ----------
        var_idx : int
            Variable index.
        context_id : str
            Context identifier.
        cov_matrix : NDArray, shape (n, n)
            Covariance matrix.
        variable_labels : list of str or None
            Variable names.

        Returns
        -------
        CIFingerprint
        """
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
        n = cov_matrix.shape[0]

        if var_idx < 0 or var_idx >= n:
            raise ValueError(f"var_idx {var_idx} out of range [0, {n})")

        cov_matrix = _symmetrize(cov_matrix)
        cov_matrix = _regularize_covariance(cov_matrix)

        try:
            precision = la.inv(cov_matrix)
        except la.LinAlgError:
            cov_matrix = _regularize_covariance(cov_matrix, _REGULARIZATION * 100)
            precision = la.inv(cov_matrix)

        precision = _symmetrize(precision)

        # Partial correlations from precision matrix
        pcorrs = np.zeros(n, dtype=np.float64)
        for j in range(n):
            if j == var_idx:
                pcorrs[j] = 0.0
                continue
            denom = np.sqrt(abs(precision[var_idx, var_idx] * precision[j, j]))
            if denom < _EPSILON:
                pcorrs[j] = 0.0
            else:
                pcorrs[j] = -precision[var_idx, j] / denom

        # Clip to [-1, 1] for numerical stability
        pcorrs = np.clip(pcorrs, -1.0, 1.0)

        return cls(
            variable_idx=var_idx,
            context_id=context_id,
            partial_correlations=pcorrs,
            variable_labels=variable_labels,
        )

    @classmethod
    def from_scm(
        cls,
        var_idx: int,
        context_id: str,
        scm: Any,
        variable_labels: Optional[List[str]] = None,
    ) -> CIFingerprint:
        """Compute CI-fingerprint from an SCM.

        Parameters
        ----------
        var_idx : int
            Variable index.
        context_id : str
            Context identifier.
        scm : StructuralCausalModel (or compatible)
            The SCM to extract the fingerprint from.
        variable_labels : list of str or None
            Variable names.

        Returns
        -------
        CIFingerprint
        """
        cov = _scm_implied_covariance(scm)
        return cls.from_covariance(var_idx, context_id, cov, variable_labels)

    @classmethod
    def from_data(
        cls,
        var_idx: int,
        context_id: str,
        data: NDArray[np.floating],
        variable_labels: Optional[List[str]] = None,
    ) -> CIFingerprint:
        """Compute CI-fingerprint from observational data.

        Parameters
        ----------
        var_idx : int
            Variable index.
        context_id : str
            Context identifier.
        data : NDArray, shape (n_samples, n_vars)
            Observational data.
        variable_labels : list of str or None
            Variable names.

        Returns
        -------
        CIFingerprint
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got {data.ndim}D")

        # Robust covariance estimate with regularization
        n_samples, n_vars = data.shape
        if n_samples < n_vars:
            warnings.warn(
                f"Fewer samples ({n_samples}) than variables ({n_vars}); "
                "using shrinkage covariance estimator.",
                stacklevel=2,
            )
            from sklearn.covariance import LedoitWolf
            estimator = LedoitWolf().fit(data)
            cov = estimator.covariance_
        else:
            cov = np.cov(data, rowvar=False)

        return cls.from_covariance(var_idx, context_id, cov, variable_labels)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    def similarity(
        self,
        other: CIFingerprint,
        alignment: Optional[Dict[int, Optional[int]]] = None,
        weights: Optional[NDArray[np.floating]] = None,
    ) -> float:
        """Compute similarity between two CI-fingerprints.

        Uses weighted cosine similarity on aligned partial correlations.

        Parameters
        ----------
        other : CIFingerprint
            Other fingerprint to compare.
        alignment : dict or None
            Maps variable indices in self to indices in other.
            If None, identity alignment.
        weights : NDArray or None
            Per-variable weights. If None, uniform.

        Returns
        -------
        float
            Similarity in [0, 1].
        """
        if alignment is None:
            n = min(self.n_vars, other.n_vars)
            a_pcorrs = self.partial_correlations[:n]
            b_pcorrs = other.partial_correlations[:n]
        else:
            # Build aligned vectors
            aligned_a = []
            aligned_b = []
            for i in range(self.n_vars):
                j = alignment.get(i, i)
                if j is not None and j < other.n_vars:
                    aligned_a.append(self.partial_correlations[i])
                    aligned_b.append(other.partial_correlations[j])
            if not aligned_a:
                return 0.0
            a_pcorrs = np.array(aligned_a)
            b_pcorrs = np.array(aligned_b)
            n = len(a_pcorrs)

        if n == 0:
            return 0.0

        if weights is not None:
            w = np.asarray(weights[:n], dtype=np.float64)
        else:
            w = np.ones(n, dtype=np.float64)

        # Weighted cosine similarity
        wa = w * a_pcorrs
        wb = w * b_pcorrs
        dot = float(wa @ wb)
        norm_a = float(np.sqrt(wa @ wa))
        norm_b = float(np.sqrt(wb @ wb))

        if norm_a < _EPSILON or norm_b < _EPSILON:
            # Both nearly zero => identical (both independent of everything)
            if norm_a < _EPSILON and norm_b < _EPSILON:
                return 1.0
            return 0.0

        cosine = dot / (norm_a * norm_b)
        # Map from [-1, 1] to [0, 1]
        return float(np.clip(0.5 * (1.0 + cosine), 0.0, 1.0))

    def distance(
        self,
        other: CIFingerprint,
        alignment: Optional[Dict[int, Optional[int]]] = None,
    ) -> float:
        """Compute distance (1 - similarity) between fingerprints.

        Parameters
        ----------
        other : CIFingerprint
            Other fingerprint.
        alignment : dict or None
            Variable alignment.

        Returns
        -------
        float
            Distance in [0, 1].
        """
        return 1.0 - self.similarity(other, alignment)

    def nonzero_entries(self, threshold: float = 0.05) -> List[int]:
        """Return indices of variables with partial correlation above threshold.

        Parameters
        ----------
        threshold : float
            Minimum absolute partial correlation.

        Returns
        -------
        list of int
        """
        return [
            j for j in range(self.n_vars)
            if j != self.variable_idx and abs(self.partial_correlations[j]) > threshold
        ]

    def sparsity(self, threshold: float = 0.05) -> float:
        """Fraction of zero partial correlations (below threshold).

        Parameters
        ----------
        threshold : float
            Threshold for considering a partial correlation zero.

        Returns
        -------
        float
            Sparsity ratio in [0, 1].
        """
        n_other = self.n_vars - 1
        if n_other <= 0:
            return 1.0
        n_zero = sum(
            1 for j in range(self.n_vars)
            if j != self.variable_idx and abs(self.partial_correlations[j]) <= threshold
        )
        return n_zero / n_other

    def top_k_dependencies(self, k: int = 5) -> List[Tuple[int, float]]:
        """Return top-k strongest partial correlations.

        Parameters
        ----------
        k : int
            Number of top dependencies to return.

        Returns
        -------
        list of (var_idx, partial_corr) tuples, sorted by absolute value.
        """
        entries = [
            (j, self.partial_correlations[j])
            for j in range(self.n_vars)
            if j != self.variable_idx
        ]
        entries.sort(key=lambda x: abs(x[1]), reverse=True)
        return entries[:k]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "variable_idx": self.variable_idx,
            "context_id": self.context_id,
            "partial_correlations": self.partial_correlations.tolist(),
            "variable_labels": self.variable_labels,
            "n_vars": self.n_vars,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CIFingerprint:
        """Deserialize from dict."""
        return cls(
            variable_idx=d["variable_idx"],
            context_id=d["context_id"],
            partial_correlations=np.array(d["partial_correlations"]),
            variable_labels=d.get("variable_labels"),
        )

    def __repr__(self) -> str:
        nnz = len(self.nonzero_entries())
        return (
            f"CIFingerprint(var={self.variable_idx}, ctx={self.context_id!r}, "
            f"n_vars={self.n_vars}, nonzero={nnz})"
        )


# ===================================================================
#  Batch fingerprint computation
# ===================================================================
def compute_all_fingerprints(
    mccm: Any,
    var_indices: Optional[List[int]] = None,
) -> Dict[Tuple[str, int], CIFingerprint]:
    """Compute CI-fingerprints for all (context, variable) pairs.

    Parameters
    ----------
    mccm : MultiContextCausalModel (or compatible)
        Multi-context model.
    var_indices : list of int or None
        Variables to fingerprint. If None, all variables.

    Returns
    -------
    dict mapping (context_id, var_idx) -> CIFingerprint
    """
    if var_indices is None:
        var_indices = list(range(_mccm_n_vars(mccm)))

    result: Dict[Tuple[str, int], CIFingerprint] = {}
    for ctx_id in _mccm_context_ids(mccm):
        scm = mccm.get_scm(ctx_id)
        cov = _scm_implied_covariance(scm)
        for vi in var_indices:
            fp = CIFingerprint.from_covariance(vi, ctx_id, cov)
            result[(ctx_id, vi)] = fp

    return result


def fingerprint_distance_matrix(
    fingerprints: Dict[Tuple[str, int], CIFingerprint],
    var_idx: int,
    context_ids: List[str],
) -> DistanceMatrix:
    """Build a DistanceMatrix from CI-fingerprints for a given variable.

    Parameters
    ----------
    fingerprints : dict
        Mapping (context_id, var_idx) -> CIFingerprint.
    var_idx : int
        Variable index.
    context_ids : list of str
        Context identifiers.

    Returns
    -------
    DistanceMatrix
    """
    K = len(context_ids)
    mat = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            fp_i = fingerprints.get((context_ids[i], var_idx))
            fp_j = fingerprints.get((context_ids[j], var_idx))
            if fp_i is None or fp_j is None:
                mat[i, j] = mat[j, i] = 1.0
            else:
                mat[i, j] = mat[j, i] = fp_i.distance(fp_j)

    return DistanceMatrix(mat, labels=context_ids, variable_idx=var_idx)


# ===================================================================
#  Batch distance computation with progress tracking
# ===================================================================
class BatchDistanceComputer:
    """Orchestrate batch distance computation across variables and contexts.

    Combines MechanismDistanceComputer with CI-fingerprint distances
    and provides progress tracking, result aggregation, and export.

    Parameters
    ----------
    mdc : MechanismDistanceComputer or None
        Distance computer. If None, creates default.
    include_fingerprints : bool
        Whether to also compute fingerprint distances. Default True.
    """

    def __init__(
        self,
        mdc: Optional[MechanismDistanceComputer] = None,
        include_fingerprints: bool = True,
    ) -> None:
        self.mdc = mdc if mdc is not None else MechanismDistanceComputer()
        self.include_fingerprints = include_fingerprints

    def compute_all(
        self,
        mccm: Any,
        var_indices: Optional[List[int]] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compute all distance matrices and fingerprints.

        Parameters
        ----------
        mccm : MultiContextCausalModel (or compatible)
            Multi-context model.
        var_indices : list of int or None
            Variable indices. If None, all.
        progress_callback : callable or None
            Called as callback(step, total, description).

        Returns
        -------
        dict with keys:
            - 'mechanism_distances': dict mapping var_idx -> DistanceMatrix
            - 'fingerprints': dict mapping (ctx, var) -> CIFingerprint (if enabled)
            - 'fingerprint_distances': dict mapping var_idx -> DistanceMatrix (if enabled)
            - 'summary': dict with aggregate statistics
        """
        if var_indices is None:
            var_indices = list(range(_mccm_n_vars(mccm)))

        total_steps = len(var_indices) * (2 if self.include_fingerprints else 1)
        step = 0

        # Mechanism distances
        mech_dists = {}
        for vi in var_indices:
            mech_dists[vi] = self.mdc.pairwise_mechanism_distances(mccm, vi)
            step += 1
            if progress_callback:
                progress_callback(step, total_steps, f"Mechanism distance var {vi}")

        result: Dict[str, Any] = {"mechanism_distances": mech_dists}

        # Fingerprints
        if self.include_fingerprints:
            fingerprints = compute_all_fingerprints(mccm, var_indices)
            ctx_ids = _mccm_context_ids(mccm)

            fp_dists = {}
            for vi in var_indices:
                fp_dists[vi] = fingerprint_distance_matrix(fingerprints, vi, ctx_ids)
                step += 1
                if progress_callback:
                    progress_callback(step, total_steps, f"Fingerprint distance var {vi}")

            result["fingerprints"] = fingerprints
            result["fingerprint_distances"] = fp_dists

        # Summary
        result["summary"] = self._compute_summary(mech_dists, var_indices)
        return result

    def _compute_summary(
        self,
        mech_dists: Dict[int, DistanceMatrix],
        var_indices: List[int],
    ) -> Dict[str, Any]:
        """Compute aggregate summary statistics."""
        if not mech_dists:
            return {"n_variables": 0}

        means = [dm.mean() for dm in mech_dists.values()]
        maxes = [dm.max() for dm in mech_dists.values()]

        return {
            "n_variables": len(var_indices),
            "overall_mean_distance": float(np.mean(means)),
            "overall_max_distance": float(np.max(maxes)),
            "most_variable": int(var_indices[int(np.argmax(means))]),
            "most_stable": int(var_indices[int(np.argmin(means))]),
            "per_variable_means": {int(vi): float(m) for vi, m in zip(var_indices, means)},
            "per_variable_maxes": {int(vi): float(m) for vi, m in zip(var_indices, maxes)},
        }

    def variable_ranking(
        self,
        mech_dists: Dict[int, DistanceMatrix],
        metric: str = "mean",
    ) -> List[Tuple[int, float]]:
        """Rank variables by their mechanism distance.

        Parameters
        ----------
        mech_dists : dict
            Mechanism distance matrices per variable.
        metric : str
            'mean', 'max', or 'std'. Default 'mean'.

        Returns
        -------
        list of (var_idx, metric_value) tuples, sorted descending.
        """
        rankings = []
        for vi, dm in mech_dists.items():
            if metric == "mean":
                val = dm.mean()
            elif metric == "max":
                val = dm.max()
            elif metric == "std":
                val = dm.std()
            else:
                raise ValueError(f"Unknown metric {metric!r}")
            rankings.append((vi, val))

        rankings.sort(key=lambda x: -x[1])
        return rankings


# ===================================================================
#  Parametric distance helpers for non-Gaussian distributions
# ===================================================================
def empirical_jsd(
    samples_p: NDArray[np.floating],
    samples_q: NDArray[np.floating],
    n_bins: int = 50,
) -> float:
    """Estimate JSD from empirical samples using histogram binning.

    Parameters
    ----------
    samples_p, samples_q : NDArray, shape (n,)
        Univariate samples from P and Q.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    float
        sqrt(JSD) estimate in [0, 1].
    """
    samples_p = np.asarray(samples_p, dtype=np.float64).ravel()
    samples_q = np.asarray(samples_q, dtype=np.float64).ravel()

    if len(samples_p) == 0 or len(samples_q) == 0:
        return 1.0

    # Common bin edges
    all_samples = np.concatenate([samples_p, samples_q])
    lo, hi = float(np.min(all_samples)), float(np.max(all_samples))
    if hi - lo < _EPSILON:
        return 0.0  # identical point masses

    edges = np.linspace(lo - _EPSILON, hi + _EPSILON, n_bins + 1)

    # Histograms (normalized to probability)
    hist_p, _ = np.histogram(samples_p, bins=edges, density=False)
    hist_q, _ = np.histogram(samples_q, bins=edges, density=False)

    p = hist_p.astype(np.float64) / hist_p.sum()
    q = hist_q.astype(np.float64) / hist_q.sum()

    # Add small epsilon to avoid log(0)
    p = p + _EPSILON
    q = q + _EPSILON
    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)

    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    jsd = max(0.5 * kl_pm + 0.5 * kl_qm, 0.0)

    jsd_normalized = jsd / _LOG2
    return float(np.sqrt(min(jsd_normalized, 1.0)))


def kl_divergence_samples(
    samples_p: NDArray[np.floating],
    samples_q: NDArray[np.floating],
    k: int = 5,
) -> float:
    """Estimate KL(P || Q) from samples using k-nearest-neighbor estimator.

    Uses the Wang-Kulback-Jones k-NN estimator.

    Parameters
    ----------
    samples_p, samples_q : NDArray, shape (n, d) or (n,)
        Samples from P and Q.
    k : int
        Number of nearest neighbors. Default 5.

    Returns
    -------
    float
        KL divergence estimate (may be negative due to estimation error).
    """
    from scipy.spatial import KDTree

    samples_p = np.asarray(samples_p, dtype=np.float64)
    samples_q = np.asarray(samples_q, dtype=np.float64)

    if samples_p.ndim == 1:
        samples_p = samples_p.reshape(-1, 1)
    if samples_q.ndim == 1:
        samples_q = samples_q.reshape(-1, 1)

    n_p = samples_p.shape[0]
    n_q = samples_q.shape[0]
    d = samples_p.shape[1]

    if n_p < k + 1 or n_q < k + 1:
        return 0.0

    tree_p = KDTree(samples_p)
    tree_q = KDTree(samples_q)

    # For each point in P, find k-th nearest neighbor distance in P and Q
    dp, _ = tree_p.query(samples_p, k=k + 1)
    dq, _ = tree_q.query(samples_p, k=k)

    rho_k = dp[:, k]  # k-th NN in P (excluding self)
    nu_k = dq[:, k - 1]  # k-th NN in Q

    # Avoid log(0)
    rho_k = np.maximum(rho_k, _EPSILON)
    nu_k = np.maximum(nu_k, _EPSILON)

    kl = d * float(np.mean(np.log(nu_k / rho_k))) + np.log(n_q / (n_p - 1))
    return float(kl)


# ===================================================================
#  Triangle inequality checker for distance matrices
# ===================================================================
def check_triangle_inequality(
    dm: DistanceMatrix,
    tolerance: float = 1e-6,
) -> Tuple[bool, List[Tuple[str, str, str, float]]]:
    """Verify that a DistanceMatrix satisfies the triangle inequality.

    For a proper metric d, we require:
        d(x, z) <= d(x, y) + d(y, z)  for all x, y, z.

    This is guaranteed for sqrt(JSD) (Theorem T1) but might be violated
    by approximation errors.

    Parameters
    ----------
    dm : DistanceMatrix
        Distance matrix to check.
    tolerance : float
        Tolerance for violations.

    Returns
    -------
    (is_valid, violations) tuple:
        is_valid: bool
        violations: list of (label_x, label_y, label_z, violation_amount) tuples.
    """
    violations = []
    K = dm.K

    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            for k in range(K):
                if k == i or k == j:
                    continue
                d_ik = dm.matrix[i, k]
                d_ij = dm.matrix[i, j]
                d_jk = dm.matrix[j, k]
                violation = d_ik - (d_ij + d_jk)
                if violation > tolerance:
                    violations.append((
                        dm.labels[i],
                        dm.labels[j],
                        dm.labels[k],
                        float(violation),
                    ))

    return len(violations) == 0, violations


def distance_matrix_from_function(
    labels: List[str],
    dist_fn: Callable,
    variable_idx: Optional[int] = None,
) -> DistanceMatrix:
    """Build a DistanceMatrix from a pairwise distance function.

    Parameters
    ----------
    labels : list of str
        Labels for each element.
    dist_fn : callable
        Function(label_a, label_b) -> float.
    variable_idx : int or None
        Variable index for the distance matrix.

    Returns
    -------
    DistanceMatrix
    """
    K = len(labels)
    mat = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            d = dist_fn(labels[i], labels[j])
            mat[i, j] = d
            mat[j, i] = d
    return DistanceMatrix(mat, labels=labels, variable_idx=variable_idx)
