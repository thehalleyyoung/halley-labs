"""
Student-t emission model and emission model selection.

Provides a heavy-tailed alternative to the GaussianEmission in
sticky_hdp_hmm.py, using a Student-t likelihood with
Normal-Inverse-Chi-squared conjugate prior (fixed degrees of freedom).
Also provides automated model selection (BIC/WAIC) between Gaussian
and Student-t emission models.

References
----------
- Murphy, K. P. (2007). Conjugate Bayesian analysis of the Gaussian
  distribution. Technical report.
- Gelman, A. et al. (2013). Bayesian Data Analysis, 3rd ed.
- Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation
  and WAIC. JMLR 11, 3571-3594.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats, special
from scipy.special import gammaln, digamma, logsumexp


# ---------------------------------------------------------------------------
# Student-t emission model
# ---------------------------------------------------------------------------

@dataclass
class StudentTEmission:
    """Conjugate emission model with Student-t likelihood.

    Uses a Normal-Inverse-Chi-squared (NI-Chi2) conjugate prior with
    *fixed* degrees of freedom ``nu`` for the Student-t observation
    model.  The NI-Chi2 parameterisation (mu_0, kappa_0, nu_0, sigma2_0)
    is maintained per dimension (diagonal covariance assumption), which
    keeps inference tractable while still capturing heavy tails.

    For multivariate data each dimension is treated independently
    (diagonal covariance), matching the interface of
    :class:`GaussianEmission` from ``sticky_hdp_hmm.py``.

    Parameters
    ----------
    dim : int
        Observation dimensionality.
    nu : float
        Degrees of freedom for the Student-t likelihood (fixed).
    kappa_0 : float
        Prior pseudo-count for the mean.
    mu_0 : NDArray or None
        Prior mean (defaults to zeros).
    sigma_scale : float
        Scale factor for the prior variance (sigma2_0 = sigma_scale^2).
    """

    dim: int = 1
    nu: float = 5.0
    kappa_0: float = 0.01
    mu_0: Optional[NDArray] = None
    sigma_scale: float = 1.0

    # Internal NI-Chi2 prior hyper-parameters per dimension
    _nu_0: Optional[NDArray] = None     # prior df  (dim,)
    _sigma2_0: Optional[NDArray] = None # prior scale (dim,)

    # Sufficient statistics
    _n: int = 0
    _sum_x: Optional[NDArray] = None
    _sum_x2: Optional[NDArray] = None

    def __post_init__(self) -> None:
        if self.mu_0 is None:
            self.mu_0 = np.zeros(self.dim)
        else:
            self.mu_0 = np.asarray(self.mu_0, dtype=np.float64)
        if self._nu_0 is None:
            self._nu_0 = np.full(self.dim, self.dim + 2.0)
        if self._sigma2_0 is None:
            self._sigma2_0 = np.full(self.dim, self.sigma_scale ** 2)
        self._sum_x = np.zeros(self.dim)
        self._sum_x2 = np.zeros(self.dim)

    # -- observation bookkeeping (same interface as GaussianEmission) ------

    def reset(self) -> None:
        """Clear all sufficient statistics."""
        self._n = 0
        self._sum_x = np.zeros(self.dim)
        self._sum_x2 = np.zeros(self.dim)

    def add_obs(self, x: NDArray) -> None:
        """Add an observation to the sufficient statistics."""
        x = np.asarray(x, dtype=np.float64).ravel()
        self._n += 1
        self._sum_x += x
        self._sum_x2 += x * x

    def remove_obs(self, x: NDArray) -> None:
        """Remove a previously added observation."""
        x = np.asarray(x, dtype=np.float64).ravel()
        self._n -= 1
        self._sum_x -= x
        self._sum_x2 -= x * x

    # -- posterior parameters (per dimension) ------------------------------

    def _posterior_params(self) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Return (mu_n, kappa_n, nu_n, sigma2_n) arrays of shape (dim,).

        Uses the standard NI-Chi2 update rules:
            kappa_n = kappa_0 + n
            nu_n    = nu_0 + n
            mu_n    = (kappa_0 * mu_0 + n * x_bar) / kappa_n
            nu_n * sigma2_n = nu_0 * sigma2_0 + S
                              + kappa_0 * n / kappa_n * (x_bar - mu_0)^2
        where S = sum(x_i^2) - n * x_bar^2.
        """
        n = self._n
        kappa_n = self.kappa_0 + n
        nu_n = self._nu_0 + n

        if n == 0:
            return (
                self.mu_0.copy(),
                np.full(self.dim, kappa_n),
                self._nu_0.copy(),
                self._sigma2_0.copy(),
            )

        x_bar = self._sum_x / n
        mu_n = (self.kappa_0 * self.mu_0 + n * x_bar) / kappa_n
        S = self._sum_x2 - n * x_bar * x_bar
        diff2 = (x_bar - self.mu_0) ** 2
        numer = self._nu_0 * self._sigma2_0 + S + self.kappa_0 * n / kappa_n * diff2
        sigma2_n = numer / nu_n
        # Ensure positive
        sigma2_n = np.clip(sigma2_n, 1e-12, None)

        return (
            mu_n,
            np.full(self.dim, kappa_n),
            nu_n,
            sigma2_n,
        )

    # -- likelihood / predictive -------------------------------------------

    def log_marginal_likelihood(self, x: NDArray) -> float:
        """Log posterior-predictive probability of *x*.

        The posterior predictive for the NI-Chi2 model is a
        (multivariate, diagonal) Student-t distribution with
        df = nu_n, location = mu_n, scale = sigma2_n * (kappa_n+1)/kappa_n.
        We additionally use the *fixed* nu (observation df) to
        compound the two sources of heavy tails.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        mu_n, kappa_n, nu_n, sigma2_n = self._posterior_params()

        # Predictive df: use posterior df as it naturally increases with data
        df = np.clip(nu_n, 1.0, None)
        scale2 = sigma2_n * (kappa_n + 1.0) / kappa_n
        scale = np.sqrt(np.clip(scale2, 1e-12, None))

        # Product of independent univariate Student-t densities
        log_p = 0.0
        for d in range(self.dim):
            log_p += float(stats.t.logpdf(x[d], df=df[d], loc=mu_n[d], scale=scale[d]))
        return log_p

    def posterior_predictive(self, x: NDArray) -> float:
        """Return the log posterior-predictive density (alias)."""
        return self.log_marginal_likelihood(x)

    def sample_posterior(self, rng: np.random.Generator) -> Tuple[NDArray, NDArray]:
        """Sample (mu, Sigma) from the NI-Chi2 posterior.

        Returns
        -------
        mu : (dim,) array
        Sigma : (dim, dim) diagonal array
        """
        mu_n, kappa_n, nu_n, sigma2_n = self._posterior_params()
        sigma2 = np.zeros(self.dim)
        mu = np.zeros(self.dim)
        for d in range(self.dim):
            # sigma2_d ~ Inv-Chi2(nu_n[d], sigma2_n[d])
            #          = InvGamma(nu_n[d]/2, nu_n[d]*sigma2_n[d]/2)
            a = nu_n[d] / 2.0
            b = nu_n[d] * sigma2_n[d] / 2.0
            sigma2[d] = 1.0 / rng.gamma(a, 1.0 / max(b, 1e-12))
            sigma2[d] = max(sigma2[d], 1e-12)
            # mu_d | sigma2_d ~ N(mu_n[d], sigma2_d / kappa_n[d])
            mu[d] = rng.normal(mu_n[d], np.sqrt(sigma2[d] / kappa_n[d]))
        return mu, np.diag(sigma2)

    def log_pdf(self, x: NDArray, mu: NDArray, Sigma: NDArray) -> float:
        """Log probability of *x* under a multivariate Student-t.

        Uses the fixed ``nu`` degrees of freedom with location ``mu``
        and scale from ``Sigma`` (diagonal elements).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        mu = np.asarray(mu, dtype=np.float64).ravel()
        if Sigma.ndim == 2:
            sigma2 = np.diag(Sigma)
        else:
            sigma2 = np.asarray(Sigma, dtype=np.float64).ravel()
        sigma = np.sqrt(np.clip(sigma2, 1e-12, None))
        log_p = 0.0
        for d in range(self.dim):
            log_p += float(stats.t.logpdf(x[d], df=self.nu, loc=mu[d], scale=sigma[d]))
        return log_p

    # -- utility -----------------------------------------------------------

    def copy(self) -> "StudentTEmission":
        """Return a deep copy resetting sufficient statistics."""
        new = StudentTEmission(
            dim=self.dim,
            nu=self.nu,
            kappa_0=self.kappa_0,
            mu_0=self.mu_0.copy(),
            sigma_scale=self.sigma_scale,
        )
        new._nu_0 = self._nu_0.copy()
        new._sigma2_0 = self._sigma2_0.copy()
        return new


# ---------------------------------------------------------------------------
# Emission model comparison helpers
# ---------------------------------------------------------------------------

def _fit_gaussian_emission(data: NDArray) -> Tuple[float, int]:
    """Fit a single Gaussian to *data* (T, D) and return (log-lik, n_params)."""
    T, D = data.shape
    mu = data.mean(axis=0)
    cov = np.cov(data.T, bias=True)
    if D == 1:
        cov = np.atleast_2d(cov)
    cov += 1e-8 * np.eye(D)
    ll = 0.0
    for t in range(T):
        ll += float(stats.multivariate_normal.logpdf(data[t], mean=mu, cov=cov))
    n_params = D + D * (D + 1) // 2  # mean + upper-triangle cov
    return ll, n_params


def _fit_student_t_emission(data: NDArray, nu: float = 5.0) -> Tuple[float, int]:
    """Fit a diagonal Student-t to *data* (T, D) and return (log-lik, n_params)."""
    T, D = data.shape
    mu = np.median(data, axis=0)
    scale = np.zeros(D)
    for d in range(D):
        iqr = np.percentile(data[:, d], 75) - np.percentile(data[:, d], 25)
        scale[d] = max(iqr / 1.349, 1e-8)  # robust scale estimate
    ll = 0.0
    for t in range(T):
        for d in range(D):
            ll += float(stats.t.logpdf(data[t, d], df=nu, loc=mu[d], scale=scale[d]))
    n_params = 2 * D  # location + scale per dim (nu is fixed)
    return ll, n_params


def _compute_bic(log_lik: float, n_params: int, n: int) -> float:
    """Bayesian Information Criterion."""
    return -2.0 * log_lik + n_params * np.log(n)


def _compute_waic(log_lik_pointwise: NDArray) -> float:
    """Simplified WAIC-1 from pointwise log-likelihoods.

    For a single fitted model (no posterior samples) we approximate
    WAIC ≈ -2 * (lppd - p_WAIC) where p_WAIC ≈ 0 for a point estimate.
    This reduces to -2 * sum(log_lik).
    """
    return -2.0 * float(np.sum(log_lik_pointwise))


def _pointwise_gaussian(data: NDArray) -> NDArray:
    """Pointwise Gaussian log-likelihoods."""
    T, D = data.shape
    mu = data.mean(axis=0)
    cov = np.cov(data.T, bias=True)
    if D == 1:
        cov = np.atleast_2d(cov)
    cov += 1e-8 * np.eye(D)
    out = np.zeros(T)
    for t in range(T):
        out[t] = float(stats.multivariate_normal.logpdf(data[t], mean=mu, cov=cov))
    return out


def _pointwise_student_t(data: NDArray, nu: float = 5.0) -> NDArray:
    """Pointwise diagonal Student-t log-likelihoods."""
    T, D = data.shape
    mu = np.median(data, axis=0)
    scale = np.zeros(D)
    for d in range(D):
        iqr = np.percentile(data[:, d], 75) - np.percentile(data[:, d], 25)
        scale[d] = max(iqr / 1.349, 1e-8)
    out = np.zeros(T)
    for t in range(T):
        lp = 0.0
        for d in range(D):
            lp += float(stats.t.logpdf(data[t, d], df=nu, loc=mu[d], scale=scale[d]))
        out[t] = lp
    return out


# ---------------------------------------------------------------------------
# EmissionModelSelector
# ---------------------------------------------------------------------------

@dataclass
class _ModelComparisonRow:
    """Single row in the comparison table."""
    model: str
    log_likelihood: float
    bic: float
    waic: float
    n_params: int


class EmissionModelSelector:
    """Automated selection between Gaussian and Student-t emission models.

    Uses BIC and WAIC to compare emission model fit, with an optional
    kurtosis pre-test to check if the data exhibits heavier tails than
    a Gaussian would produce.

    Parameters
    ----------
    nu : float
        Degrees of freedom for the Student-t candidate (default 5).
    """

    def __init__(self, nu: float = 5.0) -> None:
        self.nu = nu

    def select(
        self,
        data: NDArray,
        candidates: Sequence[str] = ("gaussian", "student_t"),
    ) -> str:
        """Select the best emission model for *data*.

        Parameters
        ----------
        data : (T,) or (T, D) array
        candidates : sequence of str
            Model names to consider.

        Returns
        -------
        str
            Name of the best model according to BIC.
        """
        table = self.compare(data, candidates)
        best = min(table, key=lambda r: r.bic)
        return best.model

    def compare(
        self,
        data: NDArray,
        candidates: Sequence[str] = ("gaussian", "student_t"),
    ) -> List[_ModelComparisonRow]:
        """Return a comparison table for each candidate model.

        Parameters
        ----------
        data : (T,) or (T, D) array
        candidates : sequence of str

        Returns
        -------
        list of _ModelComparisonRow
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        T = data.shape[0]

        rows: List[_ModelComparisonRow] = []
        for name in candidates:
            if name == "gaussian":
                ll, np_ = _fit_gaussian_emission(data)
                pw = _pointwise_gaussian(data)
            elif name == "student_t":
                ll, np_ = _fit_student_t_emission(data, nu=self.nu)
                pw = _pointwise_student_t(data, nu=self.nu)
            else:
                raise ValueError(f"Unknown model: {name}")
            bic = _compute_bic(ll, np_, T)
            waic = _compute_waic(pw)
            rows.append(_ModelComparisonRow(
                model=name,
                log_likelihood=ll,
                bic=bic,
                waic=waic,
                n_params=np_,
            ))
        return rows

    def kurtosis_test(
        self,
        data: NDArray,
        significance: float = 0.05,
    ) -> Dict[str, Any]:
        """Test whether excess kurtosis of *data* significantly exceeds 0.

        A Gaussian distribution has excess kurtosis = 0.  If the observed
        kurtosis is significantly positive, Student-t is more appropriate.

        Parameters
        ----------
        data : (T,) or (T, D)
        significance : float

        Returns
        -------
        dict with keys:
            excess_kurtosis : float or (D,) array
            exceeds_gaussian : bool
            p_value : float or (D,) array
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        T, D = data.shape

        kurt = np.zeros(D)
        p_vals = np.zeros(D)
        for d in range(D):
            col = data[:, d]
            kurt[d] = float(stats.kurtosis(col, fisher=True))
            # Jarque-Bera like z-test on kurtosis
            se_kurt = np.sqrt(24.0 / T) if T > 4 else 1.0
            z = kurt[d] / se_kurt
            p_vals[d] = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

        exceeds = bool(np.any(p_vals < significance) and np.any(kurt > 0))

        if D == 1:
            return {
                "excess_kurtosis": float(kurt[0]),
                "exceeds_gaussian": exceeds,
                "p_value": float(p_vals[0]),
            }
        return {
            "excess_kurtosis": kurt,
            "exceeds_gaussian": exceeds,
            "p_value": p_vals,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "StudentTEmission",
    "EmissionModelSelector",
]
