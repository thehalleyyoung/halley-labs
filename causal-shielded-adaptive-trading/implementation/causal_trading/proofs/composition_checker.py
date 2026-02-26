"""
Composition theorem checking for Causal-Shielded Adaptive Trading.

Verifies the composed guarantee from the causal identification module
and the shield soundness module:

    P(system correct) >= 1 - eps1 - eps2

where eps1 is the causal identification error and eps2 is the shield
failure probability. Checks interface conditions (feature set
factorization), non-vacuousness of the union bound, and produces
composed safety certificates.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CausalResult:
    """
    Result from the causal identification module.

    Parameters
    ----------
    invariant_features : set of str
        Features identified as causally invariant across environments.
    variant_features : set of str
        Features identified as causally variant (spurious).
    identification_confidence : float
        P(E_inv correct) >= 1 - eps1, i.e., eps1 = 1 - identification_confidence.
    n_environments : int
        Number of environments used in the identification.
    hsic_statistics : dict
        HSIC test statistics for each feature.
    """
    invariant_features: Set[str]
    variant_features: Set[str]
    identification_confidence: float
    n_environments: int = 2
    hsic_statistics: Dict[str, float] = field(default_factory=dict)

    @property
    def eps1(self) -> float:
        """Causal identification error bound."""
        return 1.0 - self.identification_confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invariant_features": sorted(self.invariant_features),
            "variant_features": sorted(self.variant_features),
            "identification_confidence": self.identification_confidence,
            "eps1": self.eps1,
            "n_environments": self.n_environments,
            "hsic_statistics": self.hsic_statistics,
        }


@dataclass
class ShieldResult:
    """
    Result from the shield soundness verification.

    Parameters
    ----------
    safety_bound : float
        P(phi holds | shielded policy) >= safety_bound.
    delta : float
        Confidence parameter from PAC-Bayes bound.
    complexity_term : float
        The PAC-Bayes complexity term.
    kl_divergence : float
        KL(posterior || prior).
    n_samples : int
        Number of samples used for the bound.
    feature_set : set of str
        Features used by the shield policy.
    """
    safety_bound: float
    delta: float
    complexity_term: float
    kl_divergence: float
    n_samples: int
    feature_set: Set[str]

    @property
    def eps2(self) -> float:
        """Shield failure bound: delta + complexity_term."""
        return self.delta + self.complexity_term

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safety_bound": self.safety_bound,
            "delta": self.delta,
            "complexity_term": self.complexity_term,
            "eps2": self.eps2,
            "kl_divergence": self.kl_divergence,
            "n_samples": self.n_samples,
            "feature_set": sorted(self.feature_set),
        }


@dataclass
class ComposedCertificate:
    """
    Certificate for the composed guarantee.

    The full system guarantee is:
        P(system correct) >= 1 - eps1 - eps2
    under the condition that the interface conditions hold.
    """
    verified: bool
    composed_bound: float
    eps1: float
    eps2: float
    interface_valid: bool
    non_vacuous: bool
    causal_result: Dict[str, Any] = field(default_factory=dict)
    shield_result: Dict[str, Any] = field(default_factory=dict)
    interface_details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "composed_bound": self.composed_bound,
            "eps1": self.eps1,
            "eps2": self.eps2,
            "interface_valid": self.interface_valid,
            "non_vacuous": self.non_vacuous,
            "causal_result": self.causal_result,
            "shield_result": self.shield_result,
            "interface_details": self.interface_details,
            "warnings": self.warnings,
            "assumptions": self.assumptions,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class CompositionChecker:
    """
    Check the composition theorem for the causal-shielded system.

    Verifies:
    1. Causal identification soundness: P(E_inv correct) >= 1 - eps1
    2. Shield soundness: P(phi holds) >= 1 - eps2
    3. Interface conditions: feature set factorization
    4. Composed bound: P(system correct) >= 1 - eps1 - eps2
    5. Non-vacuousness of the union bound

    Parameters
    ----------
    vacuousness_threshold : float
        The composed bound must exceed this to be considered non-vacuous.
    strict_interface : bool
        If True, require that the shield uses ONLY invariant features.
        If False, allow some variant features with a warning.
    """

    def __init__(
        self,
        vacuousness_threshold: float = 0.5,
        strict_interface: bool = True,
    ) -> None:
        self.vacuousness_threshold = vacuousness_threshold
        self.strict_interface = strict_interface
        self._certificates: List[ComposedCertificate] = []

    def check(
        self,
        causal_result: CausalResult,
        shield_result: ShieldResult,
    ) -> ComposedCertificate:
        """
        Run all composition checks and generate a certificate.

        Parameters
        ----------
        causal_result : CausalResult
            Output from the causal identification module.
        shield_result : ShieldResult
            Output from the shield soundness verifier.

        Returns
        -------
        cert : ComposedCertificate
        """
        import datetime

        warnings: List[str] = []

        # Check 1: Causal identification soundness
        eps1 = causal_result.eps1
        causal_ok = self._check_causal_soundness(causal_result, warnings)

        # Check 2: Shield soundness
        eps2 = shield_result.eps2
        shield_ok = self._check_shield_soundness(shield_result, warnings)

        # Check 3: Interface conditions
        interface_ok, interface_details = self._check_interface(
            causal_result, shield_result, warnings
        )

        # Check 4: Composed bound
        composed_bound = max(0.0, 1.0 - eps1 - eps2)

        # Check 5: Non-vacuousness
        non_vacuous = composed_bound > self.vacuousness_threshold
        if not non_vacuous:
            warnings.append(
                f"Composed bound {composed_bound:.4f} is below vacuousness "
                f"threshold {self.vacuousness_threshold:.4f}"
            )

        # Overall verification
        verified = causal_ok and shield_ok and interface_ok and non_vacuous

        assumptions = [
            "Causal identification and shield are applied to the same MDP",
            "Environments used for causal ID are representative of deployment",
            "Shield posterior is trained on data consistent with the prior",
            "Union bound (Boole's inequality) is used for composition",
            f"eps1 = {eps1:.6f} (causal identification error)",
            f"eps2 = {eps2:.6f} (shield failure probability)",
            f"Composed safety: P(correct) >= {composed_bound:.6f}",
        ]

        cert = ComposedCertificate(
            verified=verified,
            composed_bound=composed_bound,
            eps1=eps1,
            eps2=eps2,
            interface_valid=interface_ok,
            non_vacuous=non_vacuous,
            causal_result=causal_result.to_dict(),
            shield_result=shield_result.to_dict(),
            interface_details=interface_details,
            warnings=warnings,
            assumptions=assumptions,
            timestamp=datetime.datetime.utcnow().isoformat(),
        )
        self._certificates.append(cert)

        logger.info(
            "Composition check: verified=%s, bound=%.4f (eps1=%.4f, eps2=%.4f)",
            verified, composed_bound, eps1, eps2,
        )
        return cert

    def get_composed_bound(self) -> Optional[float]:
        """Return the most recent composed bound, or None."""
        if not self._certificates:
            return None
        return self._certificates[-1].composed_bound

    def get_certificate(self) -> Optional[ComposedCertificate]:
        """Return the most recent certificate."""
        if not self._certificates:
            return None
        return self._certificates[-1]

    def get_all_certificates(self) -> List[ComposedCertificate]:
        """Return all certificates."""
        return list(self._certificates)

    def required_eps_budget(
        self,
        target_safety: float = 0.95,
    ) -> Dict[str, float]:
        """
        Compute the maximum allowable eps1 and eps2 for a target safety.

        Assumes equal budget allocation: eps1 = eps2 = (1 - target) / 2.

        Returns
        -------
        budget : dict
            Contains max_eps1, max_eps2, and the target safety.
        """
        total_budget = 1.0 - target_safety
        return {
            "target_safety": target_safety,
            "total_eps_budget": total_budget,
            "max_eps1": total_budget / 2.0,
            "max_eps2": total_budget / 2.0,
        }

    def optimal_eps_split(
        self,
        target_safety: float = 0.95,
        causal_cost_per_eps: float = 1.0,
        shield_cost_per_eps: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute the optimal split of the error budget between causal
        identification (eps1) and shield (eps2) given relative costs.

        Uses Lagrange multiplier analysis: optimal split is proportional
        to sqrt(cost) for each component.

        Parameters
        ----------
        target_safety : float
            Desired safety probability.
        causal_cost_per_eps : float
            Relative cost of reducing eps1 by one unit.
        shield_cost_per_eps : float
            Relative cost of reducing eps2 by one unit.

        Returns
        -------
        split : dict
        """
        total = 1.0 - target_safety
        sqrt_c1 = math.sqrt(causal_cost_per_eps)
        sqrt_c2 = math.sqrt(shield_cost_per_eps)
        denom = sqrt_c1 + sqrt_c2

        eps1 = total * sqrt_c2 / denom  # inversely proportional to cost
        eps2 = total * sqrt_c1 / denom

        return {
            "target_safety": target_safety,
            "eps1": eps1,
            "eps2": eps2,
            "total": eps1 + eps2,
            "composed_bound": 1.0 - eps1 - eps2,
        }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _check_causal_soundness(
        self,
        causal: CausalResult,
        warnings: List[str],
    ) -> bool:
        """Check causal identification soundness."""
        ok = True

        if causal.eps1 < 0 or causal.eps1 > 1:
            warnings.append(
                f"Invalid eps1: {causal.eps1:.6f} (must be in [0, 1])"
            )
            ok = False

        if not causal.invariant_features:
            warnings.append("No invariant features identified")
            ok = False

        if causal.n_environments < 2:
            warnings.append(
                f"Only {causal.n_environments} environment(s) used; "
                "at least 2 are required for causal identification"
            )
            ok = False

        if causal.eps1 > 0.5:
            warnings.append(
                f"Causal identification error eps1={causal.eps1:.4f} is large"
            )

        return ok

    def _check_shield_soundness(
        self,
        shield: ShieldResult,
        warnings: List[str],
    ) -> bool:
        """Check shield soundness conditions."""
        ok = True

        if shield.eps2 < 0 or shield.eps2 > 1:
            warnings.append(
                f"Invalid eps2: {shield.eps2:.6f} (must be in [0, 1])"
            )
            ok = False

        if shield.kl_divergence < 0:
            warnings.append(
                f"Negative KL divergence: {shield.kl_divergence:.6f}"
            )
            ok = False

        if shield.n_samples < 10:
            warnings.append(
                f"Very few samples ({shield.n_samples}) for shield bound"
            )

        if shield.safety_bound < 0.5:
            warnings.append(
                f"Shield safety bound {shield.safety_bound:.4f} is weak"
            )

        return ok

    def _check_interface(
        self,
        causal: CausalResult,
        shield: ShieldResult,
        warnings: List[str],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check interface conditions: the shield should use only
        invariant features (feature set factorization).
        """
        invariant = causal.invariant_features
        variant = causal.variant_features
        shield_feats = shield.feature_set

        # Features used by shield that are variant (spurious)
        variant_in_shield = shield_feats & variant
        # Features used by shield that are invariant
        invariant_in_shield = shield_feats & invariant
        # Features used by shield that are unknown
        all_known = invariant | variant
        unknown_in_shield = shield_feats - all_known

        interface_ok = True
        details: Dict[str, Any] = {
            "shield_features": sorted(shield_feats),
            "invariant_features_used": sorted(invariant_in_shield),
            "variant_features_used": sorted(variant_in_shield),
            "unknown_features_used": sorted(unknown_in_shield),
            "invariant_coverage": (
                len(invariant_in_shield) / max(len(invariant), 1)
            ),
        }

        if variant_in_shield:
            msg = (
                f"Shield uses {len(variant_in_shield)} variant (spurious) "
                f"feature(s): {sorted(variant_in_shield)}"
            )
            if self.strict_interface:
                warnings.append(msg + " [STRICT: interface violation]")
                interface_ok = False
            else:
                warnings.append(msg + " [WARNING: may reduce robustness]")

        if unknown_in_shield:
            warnings.append(
                f"Shield uses {len(unknown_in_shield)} feature(s) with "
                f"unknown causal status: {sorted(unknown_in_shield)}"
            )

        if not invariant_in_shield:
            warnings.append(
                "Shield does not use any invariant features"
            )
            interface_ok = False

        details["interface_valid"] = interface_ok
        return interface_ok, details
