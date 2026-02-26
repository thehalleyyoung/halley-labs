"""
Shield soundness verification for Causal-Shielded Adaptive Trading.

Verifies the PAC-Bayes bound on shield correctness:
    P(M |= phi | shielded policy) >= 1 - delta - complexity_term

where complexity_term = sqrt((KL(rho||pi) + ln(2*sqrt(n)/eps)) / (2n)).

Provides empirical verification via Monte Carlo simulation and
generates formal certificates of soundness.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class ShieldProtocol(Protocol):
    """Protocol defining the minimal interface for a safety shield."""

    def is_safe(self, state: np.ndarray, action: Any) -> bool:
        """Return whether the action is safe in the given state."""
        ...

    def get_safe_actions(self, state: np.ndarray) -> List[Any]:
        """Return the list of safe actions in the given state."""
        ...


@dataclass
class SoundnessCertificate:
    """
    Certificate attesting to the soundness of a safety shield.

    Records all parameters, assumptions, and verification results.
    """
    verified: bool
    bound_value: float
    safety_probability_lower_bound: float
    empirical_violation_rate: float
    kl_divergence: float
    complexity_term: float
    n_samples: int
    delta: float
    epsilon: float
    mc_samples: int
    mc_violation_rate: float
    confidence_interval: Tuple[float, float]
    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "bound_value": self.bound_value,
            "safety_probability_lower_bound": self.safety_probability_lower_bound,
            "empirical_violation_rate": self.empirical_violation_rate,
            "kl_divergence": self.kl_divergence,
            "complexity_term": self.complexity_term,
            "n_samples": self.n_samples,
            "delta": self.delta,
            "epsilon": self.epsilon,
            "mc_samples": self.mc_samples,
            "mc_violation_rate": self.mc_violation_rate,
            "confidence_interval": list(self.confidence_interval),
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ShieldSoundnessVerifier:
    """
    Verify the PAC-Bayes soundness guarantee for a safety shield.

    The core guarantee is:
        P(M |= phi | shielded policy) >= 1 - delta - C(rho, pi, n, eps)

    where C is the complexity term:
        C(rho, pi, n, eps) = sqrt( (KL(rho||pi) + ln(2*sqrt(n)/eps)) / (2n) )

    Parameters
    ----------
    n_mc_samples : int
        Number of Monte Carlo samples for empirical verification.
    confidence_level : float
        Confidence level for Monte Carlo confidence intervals.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_mc_samples: int = 10000,
        confidence_level: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.n_mc_samples = n_mc_samples
        self.confidence_level = confidence_level
        self._rng = np.random.RandomState(seed)

        self._kl_cache: Optional[float] = None
        self._bound_cache: Optional[float] = None
        self._certificates: List[SoundnessCertificate] = []

    def compute_complexity_term(
        self,
        kl: float,
        n: int,
        epsilon: float = 0.05,
    ) -> float:
        """
        Compute the PAC-Bayes complexity term.

        C = sqrt( (KL + ln(2*sqrt(n)/eps)) / (2n) )

        Parameters
        ----------
        kl : float
            KL divergence between posterior and prior.
        n : int
            Number of training samples.
        epsilon : float
            Slack parameter in the logarithmic term.

        Returns
        -------
        complexity : float
        """
        if n <= 0:
            return float('inf')
        log_term = math.log(2.0 * math.sqrt(n) / epsilon)
        numerator = kl + log_term
        return math.sqrt(numerator / (2.0 * n))

    def compute_bound(
        self,
        kl: float,
        n: int,
        delta: float = 0.05,
        epsilon: float = 0.05,
    ) -> float:
        """
        Compute the PAC-Bayes safety probability lower bound.

        Returns P_lower = 1 - delta - C(kl, n, eps).

        Parameters
        ----------
        kl : float
            KL divergence between posterior and prior.
        n : int
            Number of samples.
        delta : float
            Base failure probability.
        epsilon : float
            Slack parameter.

        Returns
        -------
        bound : float
            Lower bound on P(M |= phi | shielded policy).
        """
        complexity = self.compute_complexity_term(kl, n, epsilon)
        bound = 1.0 - delta - complexity
        self._kl_cache = kl
        self._bound_cache = bound
        return max(0.0, bound)

    def get_bound(self) -> Optional[float]:
        """Return the most recently computed bound, or None."""
        return self._bound_cache

    def verify(
        self,
        shield: ShieldProtocol,
        posterior_params: Dict[str, Any],
        prior_params: Dict[str, Any],
        data: np.ndarray,
        actions: Optional[np.ndarray] = None,
        delta: float = 0.05,
        epsilon: float = 0.05,
        property_checker: Optional[Callable[[np.ndarray, Any], bool]] = None,
    ) -> SoundnessCertificate:
        """
        Full verification pipeline for shield soundness.

        Steps:
        1. Compute KL divergence between posterior and prior.
        2. Compute the PAC-Bayes complexity term.
        3. Compute the theoretical bound.
        4. Empirically verify via Monte Carlo.
        5. Generate a soundness certificate.

        Parameters
        ----------
        shield : ShieldProtocol
            The safety shield to verify.
        posterior_params : dict
            Posterior distribution parameters (must contain 'mean' and 'cov'
            for Gaussian, or 'probs' for categorical).
        prior_params : dict
            Prior distribution parameters.
        data : np.ndarray
            Training/validation data, shape (n, dim).
        actions : np.ndarray, optional
            Actions corresponding to each data point. If None,
            uses shield.get_safe_actions.
        delta : float
            Confidence parameter.
        epsilon : float
            Slack parameter.
        property_checker : callable, optional
            Function (state, action) -> bool that checks whether the
            safety property phi holds. If None, uses shield.is_safe.

        Returns
        -------
        cert : SoundnessCertificate
        """
        import datetime
        n = data.shape[0]

        # Step 1: KL divergence
        kl = self._compute_kl(posterior_params, prior_params)

        # Step 2: Complexity term
        complexity = self.compute_complexity_term(kl, n, epsilon)

        # Step 3: Theoretical bound
        safety_lower = max(0.0, 1.0 - delta - complexity)

        # Step 4: Empirical verification
        emp_result = self._empirical_verification(
            shield=shield,
            data=data,
            actions=actions,
            property_checker=property_checker,
        )

        # Step 5: Monte Carlo verification
        mc_result = self._monte_carlo_verification(
            shield=shield,
            data=data,
            actions=actions,
            property_checker=property_checker,
        )

        # Determine verification status
        warnings: List[str] = []
        verified = True

        if mc_result["violation_rate"] > (delta + complexity):
            warnings.append(
                f"MC violation rate ({mc_result['violation_rate']:.4f}) exceeds "
                f"theoretical bound ({delta + complexity:.4f})"
            )
            verified = False

        if safety_lower < 0.0:
            warnings.append("Safety bound is vacuous (negative)")
            verified = False

        if safety_lower < 0.5:
            warnings.append(
                f"Safety bound is weak ({safety_lower:.4f} < 0.5)"
            )

        assumptions = [
            "Data samples are i.i.d. from the environment distribution",
            "Prior distribution is fixed before observing training data",
            "Shield policy is drawn from the posterior distribution",
            "Safety property phi is verifiable for each (state, action) pair",
            f"KL(posterior || prior) = {kl:.6f}",
            f"n = {n} samples, delta = {delta}, epsilon = {epsilon}",
        ]

        cert = SoundnessCertificate(
            verified=verified,
            bound_value=delta + complexity,
            safety_probability_lower_bound=safety_lower,
            empirical_violation_rate=emp_result["violation_rate"],
            kl_divergence=kl,
            complexity_term=complexity,
            n_samples=n,
            delta=delta,
            epsilon=epsilon,
            mc_samples=mc_result["n_samples"],
            mc_violation_rate=mc_result["violation_rate"],
            confidence_interval=mc_result["ci"],
            assumptions=assumptions,
            warnings=warnings,
            timestamp=datetime.datetime.utcnow().isoformat(),
        )
        self._certificates.append(cert)
        self._bound_cache = safety_lower
        self._kl_cache = kl

        logger.info(
            "Shield soundness verification: verified=%s, bound=%.4f, "
            "mc_violation=%.4f",
            verified, safety_lower, mc_result["violation_rate"],
        )
        return cert

    def get_certificate(self) -> Optional[SoundnessCertificate]:
        """Return the most recent soundness certificate."""
        if not self._certificates:
            return None
        return self._certificates[-1]

    def get_all_certificates(self) -> List[SoundnessCertificate]:
        """Return all generated certificates."""
        return list(self._certificates)

    def required_samples(
        self,
        kl: float,
        target_safety: float = 0.95,
        delta: float = 0.05,
        epsilon: float = 0.05,
    ) -> int:
        """
        Compute the minimum number of samples needed to achieve a
        target safety bound.

        Solves: 1 - delta - sqrt((KL + ln(2*sqrt(n)/eps))/(2n)) >= target

        Parameters
        ----------
        kl : float
            KL divergence.
        target_safety : float
            Desired lower bound on safety probability.
        delta : float
            Confidence parameter.
        epsilon : float
            Slack parameter.

        Returns
        -------
        n : int
            Minimum sample size.
        """
        max_complexity = 1.0 - delta - target_safety
        if max_complexity <= 0:
            return int(1e9)  # Essentially impossible

        # Binary search for n
        lo, hi = 1, int(1e8)
        while lo < hi:
            mid = (lo + hi) // 2
            c = self.compute_complexity_term(kl, mid, epsilon)
            if c <= max_complexity:
                hi = mid
            else:
                lo = mid + 1
        return lo

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _compute_kl(
        self,
        posterior: Dict[str, Any],
        prior: Dict[str, Any],
    ) -> float:
        """Compute KL divergence between posterior and prior."""
        if "mean" in posterior and "cov" in posterior:
            mu_q = np.asarray(posterior["mean"])
            sigma_q = np.asarray(posterior["cov"])
            mu_p = np.asarray(prior.get("mean", np.zeros_like(mu_q)))
            sigma_p = np.asarray(prior.get("cov", np.eye(mu_q.shape[0])))

            d = mu_q.shape[0]
            sigma_p_inv = np.linalg.inv(sigma_p)
            log_det_p = np.linalg.slogdet(sigma_p)[1]
            log_det_q = np.linalg.slogdet(sigma_q)[1]

            tr = float(np.trace(sigma_p_inv @ sigma_q))
            diff = mu_p - mu_q
            quad = float(diff @ sigma_p_inv @ diff)

            kl = 0.5 * (tr + quad - d + log_det_p - log_det_q)
            return max(0.0, kl)

        elif "probs" in posterior:
            q = np.asarray(posterior["probs"], dtype=np.float64) + 1e-12
            p = np.asarray(
                prior.get("probs", np.ones_like(q) / len(q)),
                dtype=np.float64,
            ) + 1e-12
            q /= q.sum()
            p /= p.sum()
            return float(np.sum(q * np.log(q / p)))

        else:
            raise ValueError("Unsupported distribution parameterization")

    def _empirical_verification(
        self,
        shield: ShieldProtocol,
        data: np.ndarray,
        actions: Optional[np.ndarray],
        property_checker: Optional[Callable],
    ) -> Dict[str, Any]:
        """Check shield correctness on the provided data."""
        n = data.shape[0]
        violations = 0
        checker = property_checker or shield.is_safe

        for i in range(n):
            state = data[i]
            if actions is not None:
                action = actions[i]
            else:
                safe_actions = shield.get_safe_actions(state)
                action = safe_actions[0] if safe_actions else None

            if action is not None and not checker(state, action):
                violations += 1

        rate = violations / max(n, 1)
        return {"violations": violations, "n": n, "violation_rate": rate}

    def _monte_carlo_verification(
        self,
        shield: ShieldProtocol,
        data: np.ndarray,
        actions: Optional[np.ndarray],
        property_checker: Optional[Callable],
    ) -> Dict[str, Any]:
        """
        Monte Carlo verification by randomly sampling states from data
        and checking the shield's behaviour.
        """
        n_data = data.shape[0]
        n_mc = min(self.n_mc_samples, n_data)
        checker = property_checker or shield.is_safe

        indices = self._rng.choice(n_data, size=n_mc, replace=True)
        violations = 0

        for idx in indices:
            state = data[idx]
            if actions is not None:
                action = actions[idx]
            else:
                safe_actions = shield.get_safe_actions(state)
                action = safe_actions[0] if safe_actions else None

            if action is not None and not checker(state, action):
                violations += 1

        rate = violations / n_mc

        # Wilson score confidence interval
        z = sp_stats.norm.ppf(0.5 + self.confidence_level / 2.0)
        denom = 1 + z ** 2 / n_mc
        centre = (rate + z ** 2 / (2 * n_mc)) / denom
        spread = z * math.sqrt(rate * (1 - rate) / n_mc + z ** 2 / (4 * n_mc ** 2)) / denom
        ci = (max(0.0, centre - spread), min(1.0, centre + spread))

        return {
            "violations": violations,
            "n_samples": n_mc,
            "violation_rate": rate,
            "ci": ci,
        }
