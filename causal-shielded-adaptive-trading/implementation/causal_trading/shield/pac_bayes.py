"""
PAC-Bayes soundness guarantees for shield correctness.

Provides multiple PAC-Bayes bounds for certifying that a synthesized shield
correctly enforces safety specifications with high probability. Includes:

- McAllester bound (classical)
- Maurer bound (tighter empirical)
- Catoni bound (optimal for 0-1 loss)
- Sequential PAC-Bayes (Jun & Orabona 2019)
- Empirical Bernstein PAC-Bayes bound

These bounds produce certificates guaranteeing:
    P(shield violation) <= epsilon
with confidence at least 1 - delta, where epsilon depends on the
KL divergence between posterior and prior and the number of observations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln, logsumexp
from scipy.stats import beta as beta_dist
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class ShieldSoundnessCertificate:
    """
    Certificate of shield soundness.

    Guarantees that the shield's violation probability is bounded
    with specified confidence.

    Attributes
    ----------
    bound_type : str
        Type of PAC-Bayes bound used.
    violation_prob_bound : float
        Upper bound on P(shield violation).
    confidence : float
        1 - delta. Probability that the bound holds.
    kl_divergence : float
        KL(posterior || prior) used in the bound.
    n_samples : int
        Number of data points used.
    empirical_error : float
        Empirical violation rate on training data.
    is_valid : bool
        Whether the certificate is currently valid.
    metadata : dict
        Additional information about the bound computation.
    """
    bound_type: str
    violation_prob_bound: float
    confidence: float
    kl_divergence: float
    n_samples: int
    empirical_error: float
    is_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"ShieldSoundnessCertificate({self.bound_type}):\n"
            f"  P(violation) <= {self.violation_prob_bound:.6f}\n"
            f"  Confidence: {self.confidence:.4f}\n"
            f"  KL divergence: {self.kl_divergence:.4f}\n"
            f"  N samples: {self.n_samples}\n"
            f"  Empirical error: {self.empirical_error:.6f}\n"
            f"  Valid: {self.is_valid}"
        )


def kl_divergence_dirichlet(
    alpha_post: np.ndarray, alpha_prior: np.ndarray
) -> float:
    """
    Compute KL divergence between two Dirichlet distributions.

    KL(Dir(alpha_post) || Dir(alpha_prior)) using the closed-form
    expression involving the digamma function.

    Parameters
    ----------
    alpha_post : np.ndarray
        Posterior Dirichlet concentration parameters.
    alpha_prior : np.ndarray
        Prior Dirichlet concentration parameters.

    Returns
    -------
    float
        KL divergence (non-negative).
    """
    from scipy.special import digamma

    alpha_post = np.asarray(alpha_post, dtype=np.float64)
    alpha_prior = np.asarray(alpha_prior, dtype=np.float64)

    sum_post = np.sum(alpha_post)
    sum_prior = np.sum(alpha_prior)

    kl = float(
        gammaln(sum_post)
        - gammaln(sum_prior)
        - np.sum(gammaln(alpha_post))
        + np.sum(gammaln(alpha_prior))
        + np.sum((alpha_post - alpha_prior) * (digamma(alpha_post) - digamma(sum_post)))
    )

    return max(0.0, kl)


def kl_divergence_bernoulli(p: float, q: float) -> float:
    """
    KL divergence between Bernoulli(p) and Bernoulli(q).

    Parameters
    ----------
    p, q : float
        Bernoulli parameters in (0, 1).

    Returns
    -------
    float
        KL(Ber(p) || Ber(q)).
    """
    p = np.clip(p, 1e-15, 1 - 1e-15)
    q = np.clip(q, 1e-15, 1 - 1e-15)
    return float(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))


def kl_divergence_gaussian(
    mu1: float, sigma1: float, mu2: float, sigma2: float
) -> float:
    """
    KL divergence between two univariate Gaussians.

    Parameters
    ----------
    mu1, sigma1 : float
        Mean and std of first Gaussian.
    mu2, sigma2 : float
        Mean and std of second Gaussian.

    Returns
    -------
    float
        KL(N(mu1, sigma1^2) || N(mu2, sigma2^2)).
    """
    sigma1 = max(sigma1, 1e-15)
    sigma2 = max(sigma2, 1e-15)
    return float(
        np.log(sigma2 / sigma1)
        + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2)
        - 0.5
    )


def kl_divergence_multivariate_gaussian(
    mu1: np.ndarray,
    cov1: np.ndarray,
    mu2: np.ndarray,
    cov2: np.ndarray,
) -> float:
    """
    KL divergence between two multivariate Gaussians.

    Parameters
    ----------
    mu1, cov1 : np.ndarray
        Mean and covariance of first Gaussian.
    mu2, cov2 : np.ndarray
        Mean and covariance of second Gaussian.

    Returns
    -------
    float
        KL(N(mu1, cov1) || N(mu2, cov2)).
    """
    k = len(mu1)
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    cov1 = np.asarray(cov1, dtype=np.float64)
    cov2 = np.asarray(cov2, dtype=np.float64)

    cov2_inv = np.linalg.inv(cov2)
    diff = mu2 - mu1

    term1 = np.trace(cov2_inv @ cov1)
    term2 = diff @ cov2_inv @ diff
    term3 = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))

    return float(0.5 * (term1 + term2 - k + term3))


class PACBayesBound:
    """
    Base PAC-Bayes bound computation.

    Provides the classical PAC-Bayes-kl bound:
        kl(empirical_risk || true_risk) <= (KL(Q||P) + ln(2*sqrt(n)/delta)) / n

    Parameters
    ----------
    prior_type : str
        Type of prior distribution ('dirichlet', 'gaussian', 'uniform').
    """

    def __init__(self, prior_type: str = "dirichlet") -> None:
        self.prior_type = prior_type
        self._certificates: List[ShieldSoundnessCertificate] = []

    def compute_kl(
        self,
        posterior_params: np.ndarray,
        prior_params: np.ndarray,
    ) -> float:
        """
        Compute KL divergence between posterior and prior.

        Parameters
        ----------
        posterior_params : np.ndarray
            Posterior distribution parameters.
        prior_params : np.ndarray
            Prior distribution parameters.

        Returns
        -------
        float
            KL(posterior || prior).
        """
        if self.prior_type == "dirichlet":
            # Sum KL over all (state, action) Dirichlet distributions
            if posterior_params.ndim == 3:
                total_kl = 0.0
                n_s, n_a, _ = posterior_params.shape
                for s in range(n_s):
                    for a in range(n_a):
                        total_kl += kl_divergence_dirichlet(
                            posterior_params[s, a, :],
                            prior_params[s, a, :],
                        )
                return total_kl
            else:
                return kl_divergence_dirichlet(posterior_params, prior_params)

        elif self.prior_type == "gaussian":
            if posterior_params.ndim == 1:
                # Univariate: [mu, sigma]
                return kl_divergence_gaussian(
                    posterior_params[0], posterior_params[1],
                    prior_params[0], prior_params[1],
                )
            else:
                d = posterior_params.shape[0] // 2
                mu_post = posterior_params[:d]
                cov_post = np.diag(posterior_params[d:] ** 2)
                mu_prior = prior_params[:d]
                cov_prior = np.diag(prior_params[d:] ** 2)
                return kl_divergence_multivariate_gaussian(
                    mu_post, cov_post, mu_prior, cov_prior,
                )

        elif self.prior_type == "uniform":
            return 0.0

        raise ValueError(f"Unknown prior type: {self.prior_type}")

    def compute_bound(
        self,
        posterior_params: np.ndarray,
        prior_params: np.ndarray,
        n: int,
        delta: float = 0.05,
        empirical_error: float = 0.0,
    ) -> float:
        """
        Compute PAC-Bayes-kl bound on the true risk.

        Uses the classical result:
            kl(hat{R} || R) <= (KL(Q||P) + ln(2*sqrt(n)/delta)) / n

        Then inverts the kl inequality to get an upper bound on R.

        Parameters
        ----------
        posterior_params : np.ndarray
            Posterior parameters.
        prior_params : np.ndarray
            Prior parameters.
        n : int
            Number of samples.
        delta : float
            Confidence parameter.
        empirical_error : float
            Empirical error rate (0-1 loss).

        Returns
        -------
        float
            Upper bound on true risk.
        """
        kl = self.compute_kl(posterior_params, prior_params)
        complexity = (kl + np.log(2 * np.sqrt(n) / delta)) / n

        # Invert kl(hat{R} || R) <= complexity
        bound = self._invert_kl_bound(empirical_error, complexity)
        bound = min(bound, 1.0)

        cert = ShieldSoundnessCertificate(
            bound_type="PAC-Bayes-kl",
            violation_prob_bound=bound,
            confidence=1 - delta,
            kl_divergence=kl,
            n_samples=n,
            empirical_error=empirical_error,
            metadata={"complexity_term": complexity},
        )
        self._certificates.append(cert)

        logger.info(
            "PAC-Bayes-kl bound: P(violation) <= %.6f (conf=%.4f, KL=%.4f, n=%d)",
            bound, 1 - delta, kl, n,
        )

        return bound

    def _invert_kl_bound(self, p: float, c: float) -> float:
        """
        Find largest q such that kl(p || q) <= c.

        Uses Brent's method to solve kl(p, q) = c.

        Parameters
        ----------
        p : float
            Empirical error rate.
        c : float
            Complexity bound.

        Returns
        -------
        float
            Upper bound q on true risk.
        """
        if c <= 0:
            return p
        if p >= 1.0:
            return 1.0

        try:
            q = brentq(
                lambda q: kl_divergence_bernoulli(p, q) - c,
                p + 1e-10,
                1.0 - 1e-10,
            )
            return float(q)
        except ValueError:
            # If kl never reaches c in [p, 1], return 1
            return 1.0

    def get_certificate(self) -> Optional[ShieldSoundnessCertificate]:
        """Get the most recent certificate."""
        if not self._certificates:
            return None
        return self._certificates[-1]

    def get_all_certificates(self) -> List[ShieldSoundnessCertificate]:
        """Get all computed certificates."""
        return list(self._certificates)

    def bound_violation_prob(
        self,
        posterior_params: np.ndarray,
        prior_params: np.ndarray,
        n: int,
        delta: float = 0.05,
        empirical_violations: int = 0,
    ) -> float:
        """
        Compute bound on shield violation probability.

        Parameters
        ----------
        posterior_params, prior_params : np.ndarray
            Distribution parameters.
        n : int
            Number of test episodes.
        delta : float
            Confidence parameter.
        empirical_violations : int
            Number of observed violations.

        Returns
        -------
        float
            Upper bound on violation probability.
        """
        emp_error = empirical_violations / max(n, 1)
        return self.compute_bound(posterior_params, prior_params, n, delta, emp_error)


class McAllesterBound(PACBayesBound):
    """
    McAllester PAC-Bayes bound (2003).

    The simpler but looser bound:
        R(Q) <= hat{R}(Q) + sqrt((KL(Q||P) + ln(n/delta)) / (2n))

    Parameters
    ----------
    prior_type : str
        Type of prior distribution.
    """

    def __init__(self, prior_type: str = "dirichlet") -> None:
        super().__init__(prior_type)

    def compute_bound(
        self,
        posterior_params: np.ndarray,
        prior_params: np.ndarray,
        n: int,
        delta: float = 0.05,
        empirical_error: float = 0.0,
    ) -> float:
        """
        Compute McAllester bound.

        R(Q) <= hat{R}(Q) + sqrt((KL(Q||P) + ln(n/delta)) / (2n))
        """
        kl = self.compute_kl(posterior_params, prior_params)
        penalty = np.sqrt((kl + np.log(n / delta)) / (2 * n))
        bound = min(empirical_error + penalty, 1.0)

        cert = ShieldSoundnessCertificate(
            bound_type="McAllester",
            violation_prob_bound=bound,
            confidence=1 - delta,
            kl_divergence=kl,
            n_samples=n,
            empirical_error=empirical_error,
            metadata={"penalty": float(penalty)},
        )
        self._certificates.append(cert)

        logger.info(
            "McAllester bound: P(violation) <= %.6f (conf=%.4f, penalty=%.4f)",
            bound, 1 - delta, penalty,
        )
        return bound


class MaurerBound(PACBayesBound):
    """
    Maurer PAC-Bayes bound (2004).

    Tighter empirical bound using the variance proxy:
        R(Q) <= hat{R}(Q) + sqrt(hat{R}(Q) * 2C/n) + C/n

    where C = KL(Q||P) + ln(2*sqrt(n)/delta).

    Parameters
    ----------
    prior_type : str
        Type of prior distribution.
    """

    def __init__(self, prior_type: str = "dirichlet") -> None:
        super().__init__(prior_type)

    def compute_bound(
        self,
        posterior_params: np.ndarray,
        prior_params: np.ndarray,
        n: int,
        delta: float = 0.05,
        empirical_error: float = 0.0,
    ) -> float:
        """
        Compute Maurer bound.

        Uses the quadratic formula to get a tighter bound when
        empirical error is small.
        """
        kl = self.compute_kl(posterior_params, prior_params)
        C = (kl + np.log(2 * np.sqrt(n) / delta)) / n

        # Solve quadratic: R <= hat_R + sqrt(2 * hat_R * C) + C
        # This is tighter than McAllester when hat_R is small
        bound = empirical_error + np.sqrt(2 * empirical_error * C) + C
        bound = min(float(bound), 1.0)

        cert = ShieldSoundnessCertificate(
            bound_type="Maurer",
            violation_prob_bound=bound,
            confidence=1 - delta,
            kl_divergence=kl,
            n_samples=n,
            empirical_error=empirical_error,
            metadata={"complexity_per_sample": float(C)},
        )
        self._certificates.append(cert)

        logger.info(
            "Maurer bound: P(violation) <= %.6f (conf=%.4f)",
            bound, 1 - delta,
        )
        return bound


class CatoniBound(PACBayesBound):
    """
    Catoni PAC-Bayes bound (2007).

    Optimal bound for bounded losses using the Catoni function phi(x).
    Tighter than McAllester for all sample sizes.

    The bound inverts:
        E_Q[phi(lambda * (L - hat_R))] <= (KL(Q||P) + ln(1/delta)) / n

    where phi(x) = 1 - e^{-x} for x >= 0.

    Parameters
    ----------
    prior_type : str
        Type of prior distribution.
    lambda_grid_size : int
        Number of lambda values to optimize over.
    """

    def __init__(
        self,
        prior_type: str = "dirichlet",
        lambda_grid_size: int = 100,
    ) -> None:
        super().__init__(prior_type)
        self.lambda_grid_size = lambda_grid_size

    def _catoni_phi(self, x: float) -> float:
        """Catoni's function: phi(x) = log(1 + x + x^2/2) for stability."""
        if x > 20:
            return x - 0.5  # asymptotic
        return np.log(1 + x + x * x / 2)

    def _catoni_phi_inv(self, y: float) -> float:
        """Inverse of Catoni's function (numerical)."""
        try:
            return float(brentq(lambda x: self._catoni_phi(x) - y, -50, 50))
        except ValueError:
            return float('inf')

    def compute_bound(
        self,
        posterior_params: np.ndarray,
        prior_params: np.ndarray,
        n: int,
        delta: float = 0.05,
        empirical_error: float = 0.0,
    ) -> float:
        """
        Compute Catoni bound by optimizing over lambda.

        For each lambda > 0, the bound is:
            R <= hat_R + (KL(Q||P) + ln(1/delta) + psi(lambda)) / (lambda * n)

        where psi(lambda) = ln(E[e^{lambda(L-mu)}]).
        """
        kl = self.compute_kl(posterior_params, prior_params)

        best_bound = 1.0
        best_lambda = 1.0

        # Search over lambda
        lambdas = np.linspace(0.1, 10.0, self.lambda_grid_size)
        for lam in lambdas:
            # For 0-1 loss, the CGF bound gives:
            # R <= 1/(1 - e^{-lam}) * (1 - e^{-lam * hat_R}) + (KL + ln(1/delta))/(lam * n)
            psi = np.log(1 - empirical_error + empirical_error * np.exp(lam))
            numerator = kl + np.log(1.0 / delta) + n * psi
            bound_val = numerator / (lam * n)

            # Alternative formulation: direct inversion
            rhs = (kl + np.log(1.0 / delta)) / n
            bound_alt = empirical_error + rhs / lam + lam / (8 * n)

            bound_val = min(bound_val, bound_alt)

            if bound_val < best_bound:
                best_bound = bound_val
                best_lambda = lam

        best_bound = min(max(best_bound, empirical_error), 1.0)

        cert = ShieldSoundnessCertificate(
            bound_type="Catoni",
            violation_prob_bound=best_bound,
            confidence=1 - delta,
            kl_divergence=kl,
            n_samples=n,
            empirical_error=empirical_error,
            metadata={"optimal_lambda": float(best_lambda)},
        )
        self._certificates.append(cert)

        logger.info(
            "Catoni bound: P(violation) <= %.6f (lambda=%.3f, conf=%.4f)",
            best_bound, best_lambda, 1 - delta,
        )
        return best_bound


class SequentialPACBayes:
    """
    Sequential PAC-Bayes bound (Jun & Orabona 2019 style).

    Provides time-uniform PAC-Bayes bounds that hold simultaneously
    for all sample sizes n. Uses a mixture prior over time.

    The bound at time n is:
        R(Q_n) <= hat{R}_n + sqrt((KL(Q_n||P) + ln(ln(n)) + C) / (2n))

    where C accounts for the union bound over all times.

    Parameters
    ----------
    prior_type : str
        Type of prior distribution.
    initial_n : int
        Minimum sample size to start issuing bounds.
    """

    def __init__(
        self,
        prior_type: str = "dirichlet",
        initial_n: int = 10,
    ) -> None:
        self.prior_type = prior_type
        self.initial_n = initial_n
        self._base = PACBayesBound(prior_type)
        self._history: List[Tuple[int, float]] = []  # (n, bound)
        self._certificates: List[ShieldSoundnessCertificate] = []

    def compute_bound(
        self,
        posterior_params: np.ndarray,
        prior_params: np.ndarray,
        n: int,
        delta: float = 0.05,
        empirical_error: float = 0.0,
    ) -> float:
        """
        Compute sequential PAC-Bayes bound at time n.

        Uses the stitching technique to get a time-uniform bound.
        The effective delta at time n is:
            delta_n = delta * 6 / (pi^2 * ceil(log2(n))^2)

        Parameters
        ----------
        posterior_params, prior_params : np.ndarray
            Distribution parameters.
        n : int
            Current sample size.
        delta : float
            Overall confidence parameter.
        empirical_error : float
            Empirical error at time n.

        Returns
        -------
        float
            Time-uniform upper bound on true risk.
        """
        if n < self.initial_n:
            return 1.0

        kl = self._base.compute_kl(posterior_params, prior_params)

        # Time-uniform correction via peeling
        log2_n = max(1, int(np.ceil(np.log2(n))))
        delta_n = delta * 6.0 / (np.pi ** 2 * log2_n ** 2)

        # Sequential bound with log(log(n)) penalty
        log_term = np.log(np.log(max(n, np.e)) + 1)
        complexity = (kl + log_term + np.log(2.0 / delta_n)) / n

        # Invert kl inequality
        bound = self._base._invert_kl_bound(empirical_error, complexity)
        bound = min(bound, 1.0)

        self._history.append((n, bound))

        cert = ShieldSoundnessCertificate(
            bound_type="Sequential-PAC-Bayes",
            violation_prob_bound=bound,
            confidence=1 - delta,
            kl_divergence=kl,
            n_samples=n,
            empirical_error=empirical_error,
            metadata={
                "effective_delta": float(delta_n),
                "log_penalty": float(log_term),
                "time_index": len(self._history),
            },
        )
        self._certificates.append(cert)

        logger.info(
            "Sequential PAC-Bayes (n=%d): P(violation) <= %.6f",
            n, bound,
        )
        return bound

    def get_bound_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the sequence of bounds over time.

        Returns
        -------
        ns : np.ndarray
            Sample sizes.
        bounds : np.ndarray
            Corresponding bounds.
        """
        if not self._history:
            return np.array([]), np.array([])
        ns, bounds = zip(*self._history)
        return np.array(ns), np.array(bounds)

    def tighten_with_data(
        self,
        posterior_params: np.ndarray,
        prior_params: np.ndarray,
        new_n: int,
        new_empirical_error: float,
        delta: float = 0.05,
    ) -> float:
        """
        Tighten the bound with additional data.

        Parameters
        ----------
        posterior_params, prior_params : np.ndarray
            Updated distribution parameters.
        new_n : int
            New total sample size.
        new_empirical_error : float
            Updated empirical error.
        delta : float
            Confidence parameter.

        Returns
        -------
        float
            Tightened bound.
        """
        bound = self.compute_bound(
            posterior_params, prior_params, new_n, delta, new_empirical_error,
        )

        # Take minimum of all historical bounds (valid by union bound)
        if self._history:
            _, all_bounds = zip(*self._history)
            bound = min(bound, min(all_bounds))

        return bound

    def get_certificate(self) -> Optional[ShieldSoundnessCertificate]:
        """Get the tightest certificate."""
        if not self._certificates:
            return None
        return min(self._certificates, key=lambda c: c.violation_prob_bound)


class EmpiricalBernsteinPACBayes:
    """
    Empirical Bernstein PAC-Bayes bound.

    Uses the empirical variance to tighten the PAC-Bayes bound.
    Particularly effective when the loss variance is small (i.e.,
    when the shield is either almost always correct or almost always wrong).

    The bound is:
        R(Q) <= hat{R}(Q) + sqrt(2 * hat{V} * C / n) + 7 * C / (3 * (n-1))

    where hat{V} is the empirical variance and C = KL(Q||P) + ln(6/delta).

    Parameters
    ----------
    prior_type : str
        Type of prior distribution.
    """

    def __init__(self, prior_type: str = "dirichlet") -> None:
        self.prior_type = prior_type
        self._base = PACBayesBound(prior_type)
        self._certificates: List[ShieldSoundnessCertificate] = []

    def compute_bound(
        self,
        posterior_params: np.ndarray,
        prior_params: np.ndarray,
        n: int,
        delta: float = 0.05,
        empirical_error: float = 0.0,
        empirical_variance: Optional[float] = None,
        losses: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute empirical Bernstein PAC-Bayes bound.

        Parameters
        ----------
        posterior_params, prior_params : np.ndarray
            Distribution parameters.
        n : int
            Number of samples.
        delta : float
            Confidence parameter.
        empirical_error : float
            Mean loss.
        empirical_variance : float, optional
            Variance of losses. Computed from losses if not provided.
        losses : np.ndarray, optional
            Individual loss values for variance computation.

        Returns
        -------
        float
            Upper bound on true risk.
        """
        if n < 2:
            return 1.0

        kl = self._base.compute_kl(posterior_params, prior_params)

        # Compute empirical variance
        if empirical_variance is None:
            if losses is not None:
                empirical_variance = float(np.var(losses, ddof=1))
            else:
                # For Bernoulli loss, variance = p(1-p)
                empirical_variance = empirical_error * (1 - empirical_error)

        # Complexity term
        C = kl + np.log(6.0 / delta)

        # Empirical Bernstein bound
        bernstein_term = np.sqrt(2 * empirical_variance * C / n)
        remainder = 7 * C / (3 * (n - 1))
        bound = empirical_error + bernstein_term + remainder
        bound = min(float(bound), 1.0)

        # Compare with standard PAC-Bayes bound and take the tighter one
        standard_bound = self._base.compute_bound(
            posterior_params, prior_params, n, delta, empirical_error,
        )
        bound = min(bound, standard_bound)

        cert = ShieldSoundnessCertificate(
            bound_type="Empirical-Bernstein-PAC-Bayes",
            violation_prob_bound=bound,
            confidence=1 - delta,
            kl_divergence=kl,
            n_samples=n,
            empirical_error=empirical_error,
            metadata={
                "empirical_variance": float(empirical_variance),
                "bernstein_term": float(bernstein_term),
                "remainder": float(remainder),
                "standard_bound": float(standard_bound),
            },
        )
        self._certificates.append(cert)

        logger.info(
            "Empirical Bernstein PAC-Bayes: P(violation) <= %.6f "
            "(var=%.6f, conf=%.4f)",
            bound, empirical_variance, 1 - delta,
        )
        return bound

    def compute_bound_from_episodes(
        self,
        posterior_params: np.ndarray,
        prior_params: np.ndarray,
        episode_violations: np.ndarray,
        delta: float = 0.05,
    ) -> float:
        """
        Compute bound from per-episode violation indicators.

        Parameters
        ----------
        posterior_params, prior_params : np.ndarray
            Distribution parameters.
        episode_violations : np.ndarray
            Binary array: 1 if episode had a violation, 0 otherwise.
        delta : float
            Confidence parameter.

        Returns
        -------
        float
            Upper bound on violation probability.
        """
        n = len(episode_violations)
        emp_error = float(np.mean(episode_violations))
        emp_var = float(np.var(episode_violations, ddof=1)) if n > 1 else 0.25

        return self.compute_bound(
            posterior_params, prior_params, n, delta,
            empirical_error=emp_error,
            empirical_variance=emp_var,
            losses=episode_violations,
        )

    def get_certificate(self) -> Optional[ShieldSoundnessCertificate]:
        """Get the tightest certificate."""
        if not self._certificates:
            return None
        return min(self._certificates, key=lambda c: c.violation_prob_bound)

    def get_all_certificates(self) -> List[ShieldSoundnessCertificate]:
        """Get all certificates."""
        return list(self._certificates)


# ====================================================================
# PAC-Bayes Vacuity Analysis
# ====================================================================

@dataclass
class VacuityAnalysisResult:
    """Result of PAC-Bayes vacuity analysis.

    Attributes
    ----------
    k_values : list of int
        Number of regimes tested.
    n_values : np.ndarray
        Sample sizes tested.
    bounds : dict
        Mapping (K, bound_type) -> np.ndarray of bound values over n_values.
    min_n_nontrivial : dict
        Mapping (K, bound_type) -> minimum n where bound < threshold.
    threshold : float
        Vacuity threshold (bound is vacuous if >= threshold).
    kl_values : dict
        Mapping K -> estimated KL divergence for that K.
    """
    k_values: List[int]
    n_values: np.ndarray
    bounds: Dict[Tuple[int, str], np.ndarray]
    min_n_nontrivial: Dict[Tuple[int, str], Optional[int]]
    threshold: float
    kl_values: Dict[int, float]


class PACBayesVacuityAnalyzer:
    """Concrete numerical analysis of PAC-Bayes bound as function of n and K.

    Addresses the critique that PAC-Bayes bounds may be vacuous in early
    deployment because KL divergence dominates the O(1/sqrt(n)) rate.

    For a K-regime HMM with Dirichlet transition posteriors, the total KL
    divergence is:

        KL_total = sum_{s=1}^{|S|} sum_{a=1}^{|A|} KL(Dir(alpha_post_sa) || Dir(alpha_prior_sa))

    where |S| ~ O(K * |abstract_states|) and |A| is the action-space size.
    For K regimes with uniform Dirichlet prior alpha_0 = 1 and posterior
    concentrated after n_sa observations per (s,a) pair:

        KL per Dirichlet ~ K * (psi(n_sa + 1) - psi(K * n_sa + K)) * n_sa

    This grows as O(K^2 * |S| * |A| * log(n_sa)), making the bound vacuous
    unless n is large enough to compensate.

    Parameters
    ----------
    n_abstract_states_per_regime : int
        Number of abstract states per regime (default 140 for the
        5-price x 7-position x 4-drawdown discretization).
    n_actions : int
        Number of discrete actions (default 7 for position levels).
    delta : float
        Confidence parameter.
    vacuity_threshold : float
        Bound is considered vacuous if >= this value (default 0.5).
    """

    def __init__(
        self,
        n_abstract_states_per_regime: int = 140,
        n_actions: int = 7,
        delta: float = 0.05,
        vacuity_threshold: float = 0.5,
    ) -> None:
        self.n_abstract_states_per_regime = n_abstract_states_per_regime
        self.n_actions = n_actions
        self.delta = delta
        self.vacuity_threshold = vacuity_threshold

    def estimate_kl_for_k(
        self,
        K: int,
        n_observations: int,
        prior_concentration: float = 1.0,
        visit_fraction: float = 0.3,
    ) -> float:
        """Estimate total KL divergence for a K-regime HMM posterior.

        In practice, only a fraction of (state, action) pairs are visited.
        Visited pairs concentrate the posterior (high KL), while unvisited
        pairs retain the prior (zero KL). This models realistic deployment
        where trajectories cover a subset of the abstract state space.

        Parameters
        ----------
        K : int
            Number of regimes.
        n_observations : int
            Total number of observed transitions.
        prior_concentration : float
            Dirichlet prior concentration (symmetric).
        visit_fraction : float
            Fraction of (state, action) pairs actually visited (default 0.3).

        Returns
        -------
        float
            Estimated total KL(posterior || prior).
        """
        from scipy.special import digamma

        n_states = K * self.n_abstract_states_per_regime
        n_sa_pairs = n_states * self.n_actions
        n_visited = max(int(n_sa_pairs * visit_fraction), 1)

        # Observations concentrated on visited (s,a) pairs
        n_per_visited = max(n_observations / n_visited, 0.01)

        # Each visited (s,a) has a posterior Dirichlet over K_next successors
        # In practice, transitions go to a small number of next-states
        K_next = min(n_states, 20)  # effective successor set size
        alpha_prior = np.full(K_next, prior_concentration)

        # Posterior concentrates on observed successors (Zipf-like)
        # Top successor gets ~50% of counts, rest distributed
        counts = np.zeros(K_next)
        if K_next > 1:
            counts[0] = n_per_visited * 0.5
            counts[1] = n_per_visited * 0.25
            remaining = n_per_visited * 0.25
            counts[2:] = remaining / max(K_next - 2, 1)
        else:
            counts[0] = n_per_visited
        alpha_post = alpha_prior + counts

        # KL for one visited Dirichlet pair
        sum_post = np.sum(alpha_post)
        sum_prior = np.sum(alpha_prior)
        kl_one = float(
            gammaln(sum_post) - gammaln(sum_prior)
            - np.sum(gammaln(alpha_post)) + np.sum(gammaln(alpha_prior))
            + np.sum(
                (alpha_post - alpha_prior)
                * (digamma(alpha_post) - digamma(sum_post))
            )
        )
        kl_one = max(0.0, kl_one)

        # Total KL = sum over visited pairs only (unvisited have KL=0)
        total_kl = kl_one * n_visited
        return total_kl

    def compute_bound_curve(
        self,
        K: int,
        n_values: np.ndarray,
        bound_type: str = "pac-bayes-kl",
        empirical_error: float = 0.0,
        prior_concentration: float = 1.0,
    ) -> np.ndarray:
        """Compute PAC-Bayes bound for a range of sample sizes.

        Parameters
        ----------
        K : int
            Number of regimes.
        n_values : np.ndarray
            Array of sample sizes.
        bound_type : str
            One of 'pac-bayes-kl', 'mcallester', 'catoni'.
        empirical_error : float
            Assumed empirical violation rate.
        prior_concentration : float
            Dirichlet prior concentration.

        Returns
        -------
        np.ndarray
            Bound values, one per n.
        """
        bounds = np.ones(len(n_values))
        for i, n in enumerate(n_values):
            n = int(n)
            if n < 2:
                bounds[i] = 1.0
                continue

            kl = self.estimate_kl_for_k(K, n, prior_concentration)

            if bound_type == "pac-bayes-kl":
                complexity = (kl + np.log(2 * np.sqrt(n) / self.delta)) / n
                bounds[i] = self._invert_kl(empirical_error, complexity)
            elif bound_type == "mcallester":
                penalty = np.sqrt((kl + np.log(n / self.delta)) / (2 * n))
                bounds[i] = empirical_error + penalty
            elif bound_type == "catoni":
                rhs = (kl + np.log(1.0 / self.delta)) / n
                best = 1.0
                for lam in np.linspace(0.1, 10.0, 50):
                    val = empirical_error + rhs / lam + lam / (8 * n)
                    best = min(best, val)
                bounds[i] = best
            else:
                raise ValueError(f"Unknown bound type: {bound_type}")

            bounds[i] = min(max(bounds[i], empirical_error), 1.0)

        return bounds

    def _invert_kl(self, p: float, c: float) -> float:
        """Find q such that kl(p || q) = c."""
        if c <= 0:
            return p
        if p >= 1.0:
            return 1.0
        try:
            q = brentq(
                lambda q: kl_divergence_bernoulli(p, q) - c,
                p + 1e-10, 1.0 - 1e-10,
            )
            return float(q)
        except ValueError:
            return 1.0

    def full_analysis(
        self,
        k_values: Optional[List[int]] = None,
        n_range: Optional[Tuple[int, int]] = None,
        n_points: int = 200,
        bound_types: Optional[List[str]] = None,
        empirical_error: float = 0.0,
        prior_concentration: float = 1.0,
    ) -> VacuityAnalysisResult:
        """Run full vacuity analysis across K values and sample sizes.

        Parameters
        ----------
        k_values : list of int
            Regime counts to analyze (default [2, 3, 4, 5]).
        n_range : (int, int)
            Range of sample sizes (default (50, 100000)).
        n_points : int
            Number of sample size points.
        bound_types : list of str
            Bound types to compute.
        empirical_error : float
            Assumed empirical violation rate.
        prior_concentration : float
            Dirichlet prior concentration.

        Returns
        -------
        VacuityAnalysisResult
        """
        if k_values is None:
            k_values = [2, 3, 4, 5]
        if n_range is None:
            n_range = (50, 100_000)
        if bound_types is None:
            bound_types = ["pac-bayes-kl", "mcallester", "catoni"]

        n_values = np.logspace(
            np.log10(n_range[0]), np.log10(n_range[1]), n_points
        ).astype(int)
        n_values = np.unique(n_values)

        bounds: Dict[Tuple[int, str], np.ndarray] = {}
        min_n: Dict[Tuple[int, str], Optional[int]] = {}
        kl_vals: Dict[int, float] = {}

        for K in k_values:
            # Report KL at reference sample size
            kl_ref = self.estimate_kl_for_k(K, 10000, prior_concentration)
            kl_vals[K] = kl_ref
            logger.info("K=%d: KL at n=10000 is %.2f", K, kl_ref)

            for bt in bound_types:
                curve = self.compute_bound_curve(
                    K, n_values, bt, empirical_error, prior_concentration
                )
                bounds[(K, bt)] = curve

                # Find minimum n where bound < threshold
                nontrivial_idx = np.where(curve < self.vacuity_threshold)[0]
                if len(nontrivial_idx) > 0:
                    min_n[(K, bt)] = int(n_values[nontrivial_idx[0]])
                else:
                    min_n[(K, bt)] = None

                logger.info(
                    "K=%d, %s: min n for bound < %.2f is %s",
                    K, bt, self.vacuity_threshold,
                    min_n[(K, bt)] if min_n[(K, bt)] else "NEVER (in range)",
                )

        result = VacuityAnalysisResult(
            k_values=k_values,
            n_values=n_values,
            bounds=bounds,
            min_n_nontrivial=min_n,
            threshold=self.vacuity_threshold,
            kl_values=kl_vals,
        )

        return result

    def summary_table(self, result: VacuityAnalysisResult) -> str:
        """Format a human-readable summary table.

        Parameters
        ----------
        result : VacuityAnalysisResult
            Output of full_analysis().

        Returns
        -------
        str
            Formatted table.
        """
        lines = [
            f"PAC-Bayes Vacuity Analysis (delta={self.delta}, threshold={result.threshold})",
            f"States/regime={self.n_abstract_states_per_regime}, actions={self.n_actions}",
            "",
            f"{'K':>3} | {'Bound Type':>14} | {'KL(n=10K)':>10} | {'Min n (non-vacuous)':>20} | {'Bound@1K':>10} | {'Bound@10K':>10} | {'Bound@50K':>10}",
            "-" * 95,
        ]
        for K in result.k_values:
            for bt in set(bt for (k, bt) in result.bounds if k == K):
                key = (K, bt)
                curve = result.bounds[key]
                kl = result.kl_values[K]
                min_n = result.min_n_nontrivial.get(key)
                min_n_str = str(min_n) if min_n else ">100K"

                # Find bounds at specific n values
                def bound_at_n(target_n: int) -> str:
                    idx = np.argmin(np.abs(result.n_values - target_n))
                    val = curve[idx]
                    return f"{val:.4f}" if val < 1.0 else "1.0000"

                lines.append(
                    f"{K:>3} | {bt:>14} | {kl:>10.1f} | {min_n_str:>20} | "
                    f"{bound_at_n(1000):>10} | {bound_at_n(10000):>10} | {bound_at_n(50000):>10}"
                )

        return "\n".join(lines)
