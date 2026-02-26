"""
PAC-Bayes bound computation for Causal-Shielded Adaptive Trading.

Implements KL divergence computation between posterior and prior
distributions, McAllester and Catoni bounds, sequential (anytime-valid)
PAC-Bayes bounds, empirical risk computation on shield correctness,
and bound certificate generation.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import optimize as sp_optimize
from scipy import special as sp_special
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class BoundCertificate:
    """
    Certificate recording a PAC-Bayes bound computation.

    Contains all parameters, assumptions, and computed values needed
    to reproduce and verify the bound.
    """
    bound_type: str
    bound_value: float
    empirical_risk: float
    kl_divergence: float
    n_samples: int
    delta: float
    complexity_term: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bound_type": self.bound_type,
            "bound_value": self.bound_value,
            "empirical_risk": self.empirical_risk,
            "kl_divergence": self.kl_divergence,
            "n_samples": self.n_samples,
            "delta": self.delta,
            "complexity_term": self.complexity_term,
            "parameters": self.parameters,
            "assumptions": self.assumptions,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def kl_divergence_gaussians(
    mu_q: np.ndarray,
    sigma_q: np.ndarray,
    mu_p: np.ndarray,
    sigma_p: np.ndarray,
) -> float:
    """
    Compute KL(q || p) between two multivariate Gaussians.

    Parameters
    ----------
    mu_q, sigma_q : np.ndarray
        Mean and covariance of the posterior q.
    mu_p, sigma_p : np.ndarray
        Mean and covariance of the prior p.

    Returns
    -------
    kl : float
        KL divergence KL(q || p).
    """
    d = mu_q.shape[0]
    mu_q = mu_q.ravel()
    mu_p = mu_p.ravel()

    sigma_p_inv = np.linalg.inv(sigma_p)
    log_det_p = np.linalg.slogdet(sigma_p)[1]
    log_det_q = np.linalg.slogdet(sigma_q)[1]

    trace_term = float(np.trace(sigma_p_inv @ sigma_q))
    diff = mu_p - mu_q
    quad_term = float(diff @ sigma_p_inv @ diff)
    log_det_term = log_det_p - log_det_q

    kl = 0.5 * (trace_term + quad_term - d + log_det_term)
    return max(0.0, kl)


def kl_divergence_categorical(
    q: np.ndarray,
    p: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute KL(q || p) for categorical distributions.

    Parameters
    ----------
    q, p : np.ndarray
        Probability vectors (must sum to 1).

    Returns
    -------
    kl : float
    """
    q = np.asarray(q, dtype=np.float64) + eps
    p = np.asarray(p, dtype=np.float64) + eps
    q = q / q.sum()
    p = p / p.sum()
    return float(np.sum(q * np.log(q / p)))


def kl_divergence_bernoulli(q: float, p: float, eps: float = 1e-12) -> float:
    """KL divergence between two Bernoulli distributions."""
    q = np.clip(q, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)
    return float(
        q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))
    )


class PACBayesBoundComputer:
    """
    Compute PAC-Bayes bounds on the generalisation error of a
    stochastic predictor (shield policy).

    Supports McAllester, Catoni, and sequential (anytime-valid) bounds.

    Parameters
    ----------
    prior_type : str
        Type of prior: "gaussian" or "categorical".
    prior_params : dict
        Parameters of the prior distribution.
    """

    def __init__(
        self,
        prior_type: str = "gaussian",
        prior_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.prior_type = prior_type
        self.prior_params = prior_params or {}

        self._posterior_params: Optional[Dict[str, Any]] = None
        self._empirical_losses: List[float] = []
        self._certificates: List[BoundCertificate] = []

    def set_posterior(self, posterior_params: Dict[str, Any]) -> None:
        """Set the posterior distribution parameters."""
        self._posterior_params = posterior_params

    def add_loss(self, loss: float) -> None:
        """Record an empirical loss value (0-1 for classification)."""
        self._empirical_losses.append(loss)

    def add_losses(self, losses: np.ndarray) -> None:
        """Record multiple empirical loss values."""
        for l in np.asarray(losses).ravel():
            self._empirical_losses.append(float(l))

    def compute_kl(
        self,
        posterior: Optional[Dict[str, Any]] = None,
        prior: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Compute KL divergence between posterior and prior.

        Parameters
        ----------
        posterior : dict, optional
            Override posterior parameters.
        prior : dict, optional
            Override prior parameters.

        Returns
        -------
        kl : float
        """
        post = posterior or self._posterior_params
        pri = prior or self.prior_params

        if post is None:
            raise ValueError("Posterior not set")

        if self.prior_type == "gaussian":
            mu_q = np.asarray(post["mean"])
            sigma_q = np.asarray(post["cov"])
            mu_p = np.asarray(pri.get("mean", np.zeros_like(mu_q)))
            sigma_p = np.asarray(
                pri.get("cov", np.eye(mu_q.shape[0]))
            )
            return kl_divergence_gaussians(mu_q, sigma_q, mu_p, sigma_p)

        elif self.prior_type == "categorical":
            q = np.asarray(post["probs"])
            p = np.asarray(pri.get("probs", np.ones_like(q) / len(q)))
            return kl_divergence_categorical(q, p)

        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")

    def empirical_risk(self) -> float:
        """Compute the empirical risk (average loss)."""
        if not self._empirical_losses:
            return 0.0
        return float(np.mean(self._empirical_losses))

    def compute_bound(
        self,
        n: Optional[int] = None,
        delta: float = 0.05,
        bound_type: str = "mcallester",
    ) -> float:
        """
        Compute a PAC-Bayes bound on the true risk.

        Parameters
        ----------
        n : int, optional
            Number of samples. Defaults to number of recorded losses.
        delta : float
            Confidence parameter (bound holds with probability 1-delta).
        bound_type : str
            One of "mcallester", "catoni", "sequential".

        Returns
        -------
        bound : float
            Upper bound on the true risk.
        """
        if n is None:
            n = len(self._empirical_losses)
        if n == 0:
            raise ValueError("No samples available for bound computation")

        kl = self.compute_kl()
        emp_risk = self.empirical_risk()

        if bound_type == "mcallester":
            bound = self._mcallester_bound(emp_risk, kl, n, delta)
        elif bound_type == "catoni":
            bound = self._catoni_bound(emp_risk, kl, n, delta)
        elif bound_type == "sequential":
            bound = self._sequential_bound(emp_risk, kl, n, delta)
        else:
            raise ValueError(f"Unknown bound type: {bound_type}")

        return min(bound, 1.0)

    def generate_certificate(
        self,
        n: Optional[int] = None,
        delta: float = 0.05,
        bound_type: str = "mcallester",
    ) -> BoundCertificate:
        """
        Generate a formal certificate for the computed bound.

        Returns
        -------
        cert : BoundCertificate
            Contains all information needed to verify the bound.
        """
        if n is None:
            n = len(self._empirical_losses)

        kl = self.compute_kl()
        emp_risk = self.empirical_risk()
        bound = self.compute_bound(n=n, delta=delta, bound_type=bound_type)
        complexity = bound - emp_risk

        import datetime
        ts = datetime.datetime.utcnow().isoformat()

        assumptions = [
            "Samples are drawn i.i.d. from the data-generating distribution",
            "Prior is fixed before observing data",
            f"Loss function is bounded in [0, 1]",
            f"Confidence level: 1 - delta = {1 - delta}",
        ]
        if bound_type == "sequential":
            assumptions.append(
                "Sequential bound: valid at any stopping time (anytime guarantee)"
            )

        cert = BoundCertificate(
            bound_type=bound_type,
            bound_value=bound,
            empirical_risk=emp_risk,
            kl_divergence=kl,
            n_samples=n,
            delta=delta,
            complexity_term=complexity,
            parameters={
                "prior_type": self.prior_type,
                "prior_params": _safe_serialize(self.prior_params),
                "posterior_params": _safe_serialize(self._posterior_params),
            },
            assumptions=assumptions,
            timestamp=ts,
        )
        self._certificates.append(cert)
        return cert

    def get_certificates(self) -> List[BoundCertificate]:
        """Return all generated certificates."""
        return list(self._certificates)

    # ------------------------------------------------------------------ #
    # Bound implementations
    # ------------------------------------------------------------------ #

    def _mcallester_bound(
        self,
        emp_risk: float,
        kl: float,
        n: int,
        delta: float,
    ) -> float:
        """
        McAllester PAC-Bayes bound (McAllester 1999, 2003).

        R(rho) <= hat{R}(rho) + sqrt( (KL(rho||pi) + ln(2*sqrt(n)/delta)) / (2n) )
        """
        log_term = kl + math.log(2.0 * math.sqrt(n) / delta)
        complexity = math.sqrt(log_term / (2.0 * n))
        return emp_risk + complexity

    def _catoni_bound(
        self,
        emp_risk: float,
        kl: float,
        n: int,
        delta: float,
    ) -> float:
        """
        Catoni PAC-Bayes bound (Catoni 2007) for bounded losses in [0, 1].

        Solves for the tightest bound via the Catoni convex conjugate.
        Uses binary search over the bound value.

        The bound satisfies:
            n * kl_bernoulli(emp_risk, true_risk) <= KL(rho||pi) + ln(2*sqrt(n)/delta)
        """
        rhs = kl + math.log(2.0 * math.sqrt(n) / delta)

        def objective(r: float) -> float:
            if r <= emp_risk:
                return -rhs
            kl_bern = kl_divergence_bernoulli(emp_risk, r)
            return n * kl_bern - rhs

        # Binary search for the bound
        lo = emp_risk
        hi = 1.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if objective(mid) < 0:
                lo = mid
            else:
                hi = mid

        return hi

    def _sequential_bound(
        self,
        emp_risk: float,
        kl: float,
        n: int,
        delta: float,
    ) -> float:
        """
        Sequential (anytime-valid) PAC-Bayes bound.

        Uses the method of mixtures to produce a bound valid at any
        stopping time, not just a fixed sample size.

        R(rho) <= hat{R}(rho) + sqrt( (KL(rho||pi) + ln(ln(e*n)/delta) + ln(2)) / (2n) )

        Reference: Jun & Orabona (2019), Jang et al. (2023).
        """
        log_log_term = math.log(math.log(math.e * n) + 1e-10)
        log_term = kl + log_log_term + math.log(1.0 / delta) + math.log(2.0)
        complexity = math.sqrt(log_term / (2.0 * n))
        return emp_risk + complexity

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #

    def compute_bound_curve(
        self,
        n_range: np.ndarray,
        delta: float = 0.05,
        bound_type: str = "mcallester",
    ) -> np.ndarray:
        """
        Compute the bound as a function of sample size.

        Parameters
        ----------
        n_range : np.ndarray
            Array of sample sizes.
        delta : float
            Confidence parameter.
        bound_type : str
            Type of bound to compute.

        Returns
        -------
        bounds : np.ndarray
            Bound values for each n.
        """
        kl = self.compute_kl()
        emp_risk = self.empirical_risk()
        bounds = np.zeros(len(n_range))

        for i, n_val in enumerate(n_range):
            n_val = int(n_val)
            if n_val <= 0:
                bounds[i] = 1.0
                continue
            if bound_type == "mcallester":
                bounds[i] = self._mcallester_bound(emp_risk, kl, n_val, delta)
            elif bound_type == "catoni":
                bounds[i] = self._catoni_bound(emp_risk, kl, n_val, delta)
            elif bound_type == "sequential":
                bounds[i] = self._sequential_bound(emp_risk, kl, n_val, delta)

        return np.minimum(bounds, 1.0)

    def sensitivity_analysis(
        self,
        delta_range: Optional[np.ndarray] = None,
        bound_type: str = "mcallester",
    ) -> Dict[str, np.ndarray]:
        """
        Analyse sensitivity of the bound to the confidence parameter delta.

        Returns bound values for a range of delta values.
        """
        if delta_range is None:
            delta_range = np.logspace(-4, -1, 20)

        n = len(self._empirical_losses)
        if n == 0:
            return {"delta": delta_range, "bounds": np.ones_like(delta_range)}

        kl = self.compute_kl()
        emp_risk = self.empirical_risk()

        bounds = np.zeros(len(delta_range))
        for i, d in enumerate(delta_range):
            if bound_type == "mcallester":
                bounds[i] = self._mcallester_bound(emp_risk, kl, n, float(d))
            elif bound_type == "catoni":
                bounds[i] = self._catoni_bound(emp_risk, kl, n, float(d))
            elif bound_type == "sequential":
                bounds[i] = self._sequential_bound(emp_risk, kl, n, float(d))

        return {"delta": delta_range, "bounds": np.minimum(bounds, 1.0)}

    def compare_bounds(
        self,
        n: Optional[int] = None,
        delta: float = 0.05,
    ) -> Dict[str, float]:
        """Compare all three bound types for the same parameters."""
        if n is None:
            n = len(self._empirical_losses)
        return {
            "mcallester": self.compute_bound(n=n, delta=delta, bound_type="mcallester"),
            "catoni": self.compute_bound(n=n, delta=delta, bound_type="catoni"),
            "sequential": self.compute_bound(n=n, delta=delta, bound_type="sequential"),
            "empirical_risk": self.empirical_risk(),
            "kl_divergence": self.compute_kl(),
            "n": n,
            "delta": delta,
        }


def _safe_serialize(obj: Any) -> Any:
    """Convert numpy arrays to lists for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj
