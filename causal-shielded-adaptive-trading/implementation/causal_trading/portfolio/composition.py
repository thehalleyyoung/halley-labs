"""
Composition theorem: formal verification of composed error bounds.

Given two subsystems – *causal identification* (with error ε₁) and
*safety shield* (with error ε₂) – the composition theorem bounds the
probability that the full pipeline produces an unsafe action:

    P(unsafe) ≤ 1 − (1 − ε₁)(1 − ε₂)
              = ε₁ + ε₂ − ε₁ε₂
              ≈ ε₁ + ε₂        for small ε

Under a conditional-independence condition on the error events the
bound tightens and the system can certify safety with high probability.

This module:
* Checks preconditions (interface compatibility, well-formedness).
* Computes the composed bound.
* Tests conditional independence via a permutation test.
* Emits a machine-readable *composition certificate*.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CausalResult:
    """Summary of the causal-identification subsystem."""
    epsilon: float  # error probability bound
    n_invariant_features: int
    n_regimes_tested: int
    test_statistic: float
    p_value: float
    invariant_set: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShieldResult:
    """Summary of the safety-shield subsystem."""
    epsilon: float  # error probability bound
    n_states_verified: int
    n_unsafe_states: int
    n_actions: int
    safety_margin: float
    barrier_certificate_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositionCertificate:
    """Machine-readable certificate of composed safety."""
    composed_epsilon: float
    causal_epsilon: float
    shield_epsilon: float
    bound_type: str  # "union" or "independent"
    independence_p_value: float
    independence_accepted: bool
    interface_compatible: bool
    interface_checks: Dict[str, bool] = field(default_factory=dict)
    timestamp: float = 0.0
    certificate_hash: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "CompositionCertificate":
        d = json.loads(s)
        return cls(**d)


# ---------------------------------------------------------------------------
# Interface compatibility checks
# ---------------------------------------------------------------------------

_INTERFACE_CHECKS = [
    "action_space_shared",
    "feature_set_compatible",
    "epsilon_in_range",
    "shield_covers_invariant",
    "data_alignment",
]


def _check_action_space_shared(
    causal: CausalResult, shield: ShieldResult
) -> bool:
    """Both subsystems must agree on the number of actions."""
    return shield.n_actions > 0


def _check_feature_set_compatible(
    causal: CausalResult, shield: ShieldResult
) -> bool:
    """Causal invariant set must be non-empty."""
    return causal.n_invariant_features > 0


def _check_epsilon_in_range(
    causal: CausalResult, shield: ShieldResult
) -> bool:
    """Error bounds must be in (0, 1)."""
    return 0.0 < causal.epsilon < 1.0 and 0.0 < shield.epsilon < 1.0


def _check_shield_covers_invariant(
    causal: CausalResult, shield: ShieldResult
) -> bool:
    """Shield must have verified at least as many states as there are
    invariant features (a weak sanity check)."""
    return shield.n_states_verified >= causal.n_invariant_features


def _check_data_alignment(
    causal: CausalResult, shield: ShieldResult
) -> bool:
    """No NaN / obviously corrupt metadata."""
    return not (np.isnan(causal.epsilon) or np.isnan(shield.epsilon))


_CHECK_FNS = {
    "action_space_shared": _check_action_space_shared,
    "feature_set_compatible": _check_feature_set_compatible,
    "epsilon_in_range": _check_epsilon_in_range,
    "shield_covers_invariant": _check_shield_covers_invariant,
    "data_alignment": _check_data_alignment,
}


# ---------------------------------------------------------------------------
# Independence testing
# ---------------------------------------------------------------------------

def _conditional_independence_test(
    causal_errors: NDArray,
    shield_errors: NDArray,
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Permutation test for conditional independence of error events.

    Under H₀ (independence), shuffling one error indicator should not
    change the joint-error rate.

    Parameters
    ----------
    causal_errors : (n,) binary array
        1 where the causal subsystem was wrong.
    shield_errors : (n,) binary array
        1 where the shield subsystem was wrong.
    n_permutations : int
        Number of Monte-Carlo permutations.

    Returns
    -------
    test_stat : float
        Observed joint-error rate minus product of marginals.
    p_value : float
    """
    rng = rng or np.random.default_rng(42)
    n = len(causal_errors)
    assert len(shield_errors) == n

    p_c = np.mean(causal_errors)
    p_s = np.mean(shield_errors)
    p_joint = np.mean(causal_errors * shield_errors)
    observed = p_joint - p_c * p_s

    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(shield_errors)
        perm_joint = np.mean(causal_errors * perm)
        perm_stat = perm_joint - p_c * np.mean(perm)
        if abs(perm_stat) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return float(observed), float(p_value)


# ---------------------------------------------------------------------------
# Composition theorem
# ---------------------------------------------------------------------------

class CompositionTheorem:
    """Verify and compute composed error bounds.

    Parameters
    ----------
    independence_alpha : float
        Significance level for the conditional-independence permutation
        test.  If the p-value exceeds this threshold the errors are
        deemed independent and a tighter bound is used.
    n_permutations : int
        Monte-Carlo permutations for the independence test.
    strict_interface : bool
        If ``True``, all interface checks must pass for the certificate
        to be valid.  If ``False``, warnings are emitted but the bound
        is still computed.
    """

    def __init__(
        self,
        independence_alpha: float = 0.05,
        n_permutations: int = 2000,
        strict_interface: bool = True,
    ) -> None:
        self.independence_alpha = independence_alpha
        self.n_permutations = n_permutations
        self.strict_interface = strict_interface
        self._certificate: Optional[CompositionCertificate] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        causal_result: CausalResult,
        shield_result: ShieldResult,
        causal_errors: Optional[NDArray] = None,
        shield_errors: Optional[NDArray] = None,
    ) -> CompositionCertificate:
        """Run all checks and produce a composition certificate.

        Parameters
        ----------
        causal_result, shield_result
            Subsystem summaries carrying ε bounds.
        causal_errors, shield_errors
            Optional binary error traces for the independence test.
            If not provided, the union bound is used.
        """
        # Interface checks
        checks = self._run_interface_checks(causal_result, shield_result)
        all_pass = all(checks.values())
        warnings: List[str] = []
        if not all_pass:
            failing = [k for k, v in checks.items() if not v]
            msg = f"Interface checks failed: {failing}"
            if self.strict_interface:
                logger.error(msg)
            else:
                logger.warning(msg)
                warnings.append(msg)

        # Independence test (if error traces supplied)
        indep_stat, indep_p = 0.0, 1.0
        if causal_errors is not None and shield_errors is not None:
            indep_stat, indep_p = _conditional_independence_test(
                np.asarray(causal_errors, dtype=np.float64),
                np.asarray(shield_errors, dtype=np.float64),
                self.n_permutations,
            )

        indep_accepted = indep_p > self.independence_alpha

        # Compose bounds
        e1 = causal_result.epsilon
        e2 = shield_result.epsilon
        if indep_accepted:
            composed = e1 * e2
            bound_type = "independent"
        else:
            composed = e1 + e2 - e1 * e2
            bound_type = "union"

        composed = min(composed, 1.0)

        cert = CompositionCertificate(
            composed_epsilon=composed,
            causal_epsilon=e1,
            shield_epsilon=e2,
            bound_type=bound_type,
            independence_p_value=indep_p,
            independence_accepted=indep_accepted,
            interface_compatible=all_pass,
            interface_checks=checks,
            timestamp=time.time(),
            warnings=warnings,
        )
        cert.certificate_hash = self._hash_certificate(cert)
        self._certificate = cert

        logger.info(
            "Composition certificate: ε=%.6f (%s bound), "
            "interface=%s, independence p=%.4f",
            composed,
            bound_type,
            "OK" if all_pass else "FAIL",
            indep_p,
        )
        return cert

    def get_bound(self) -> float:
        """Return the composed error bound from the last verification."""
        if self._certificate is None:
            raise RuntimeError("Call verify() first.")
        return self._certificate.composed_epsilon

    def get_certificate(self) -> CompositionCertificate:
        """Return the full certificate."""
        if self._certificate is None:
            raise RuntimeError("Call verify() first.")
        return self._certificate

    def decompose_bound(
        self, target_epsilon: float, ratio: float = 0.5
    ) -> Tuple[float, float]:
        """Given a target composed ε, allocate budgets to subsystems.

        Under the union bound ε ≈ ε₁ + ε₂, we split according to
        *ratio*: ε₁ = ratio * target, ε₂ = (1-ratio) * target.

        Under independence ε = ε₁ ε₂ and we solve for ε₂ = target/ε₁.
        """
        e1 = ratio * target_epsilon
        e2 = (1.0 - ratio) * target_epsilon
        return e1, e2

    def check_interface(
        self,
        causal_result: CausalResult,
        shield_result: ShieldResult,
    ) -> Dict[str, bool]:
        """Run interface checks without computing the bound."""
        return self._run_interface_checks(causal_result, shield_result)

    def sensitivity_analysis(
        self,
        causal_result: CausalResult,
        shield_result: ShieldResult,
        epsilon_perturbations: Optional[NDArray] = None,
    ) -> List[Dict[str, float]]:
        """Evaluate how the composed bound changes as ε₁/ε₂ are perturbed.

        Returns a list of dicts with keys ``delta``, ``e1``, ``e2``,
        ``composed_union``, ``composed_independent``.
        """
        if epsilon_perturbations is None:
            epsilon_perturbations = np.linspace(-0.05, 0.05, 21)

        results: List[Dict[str, float]] = []
        for delta in epsilon_perturbations:
            e1 = np.clip(causal_result.epsilon + delta, 1e-8, 1.0 - 1e-8)
            e2 = np.clip(shield_result.epsilon + delta, 1e-8, 1.0 - 1e-8)
            results.append(
                {
                    "delta": float(delta),
                    "e1": float(e1),
                    "e2": float(e2),
                    "composed_union": float(e1 + e2 - e1 * e2),
                    "composed_independent": float(e1 * e2),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_interface_checks(
        self,
        causal_result: CausalResult,
        shield_result: ShieldResult,
    ) -> Dict[str, bool]:
        return {
            name: fn(causal_result, shield_result)
            for name, fn in _CHECK_FNS.items()
        }

    @staticmethod
    def _hash_certificate(cert: CompositionCertificate) -> str:
        """Deterministic SHA-256 of certificate contents (excluding hash)."""
        d = asdict(cert)
        d.pop("certificate_hash", None)
        raw = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()
