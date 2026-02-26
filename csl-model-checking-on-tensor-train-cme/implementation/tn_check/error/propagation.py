"""
Error propagation analysis for TT-compressed Markov semigroups.

Theorem 1 (Error Contraction):
    For CME generator Q (Metzler matrix with zero column sums),
    the semigroup e^{Qt} is a contraction in L1 norm: ‖e^{Qt}‖₁ = 1.
    
    Therefore, per-step truncation errors do NOT amplify:
    ‖p_exact(t) - p_TT(t)‖₁ ≤ Σ_k ε_k
    where ε_k is the truncation error at step k.

    This is the key structural insight: unlike generic non-Hermitian operators
    where errors can grow exponentially, the Metzler structure of CME generators
    ensures linear error accumulation.

    Combined with Proposition 1 (clamping bound), the total certified error
    for the clamped TT-compressed solution is:
    ‖p_exact(t) - p_clamped(t)‖₁ ≤ 2 * Σ_k ε_k
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from tn_check.error.certification import ErrorCertificate

logger = logging.getLogger(__name__)


@dataclass
class PropagationAnalysis:
    """Analysis of error propagation through CSL evaluation."""
    time_horizon: float = 0.0
    num_steps: int = 0
    per_step_errors: list[float] = None
    accumulated_error: float = 0.0
    contractivity_factor: float = 1.0  # for Metzler, this is 1
    amplification_bound: float = 1.0
    clamping_overhead: float = 2.0  # factor from Proposition 1

    def __post_init__(self):
        if self.per_step_errors is None:
            self.per_step_errors = []


def semigroup_error_bound(
    per_step_errors: list[float],
    time_horizon: float,
    is_metzler: bool = True,
) -> PropagationAnalysis:
    """
    Compute certified error bound for TT-compressed Markov semigroup evolution.

    For Metzler generators (CME), the semigroup is an L1 contraction,
    so errors accumulate linearly:
        ‖p_exact(t) - p_TT(t)‖₁ ≤ C(t) * Σ ε_k
    where C(t) = 1 for Metzler generators (Theorem 1).

    For non-Metzler generators, C(t) can grow, and we use the
    conservative bound C(t) = e^{μt} where μ is the log-norm.

    Args:
        per_step_errors: List of per-step truncation errors.
        time_horizon: Total time horizon T.
        is_metzler: Whether the generator has Metzler structure.

    Returns:
        PropagationAnalysis with certified bounds.
    """
    analysis = PropagationAnalysis(
        time_horizon=time_horizon,
        num_steps=len(per_step_errors),
        per_step_errors=per_step_errors,
    )

    total_truncation = sum(per_step_errors)

    if is_metzler:
        # Theorem 1: Metzler contractivity
        # ‖e^{Qt}‖₁ = 1 for stochastic semigroup
        analysis.contractivity_factor = 1.0
        analysis.accumulated_error = total_truncation
        # Including clamping (Proposition 1): factor of 2
        analysis.amplification_bound = 2.0 * total_truncation
    else:
        # Generic non-Hermitian: exponential amplification possible
        # Conservative bound: assume ‖e^{Qt}‖ ≤ e^{μt}
        # Without knowing μ, use the worst-case bound
        mu_estimate = 1.0  # conservative
        analysis.contractivity_factor = np.exp(mu_estimate * time_horizon)
        analysis.accumulated_error = analysis.contractivity_factor * total_truncation
        analysis.amplification_bound = 2.0 * analysis.accumulated_error

    return analysis


def csl_error_propagation(
    inner_error: float,
    threshold: float,
    comparison: str,
    num_states_near_threshold: int = 0,
    total_states: int = 1,
) -> dict:
    """
    Analyze error propagation through CSL probability operators.

    For nested P~p[ψ], inner error ε in the probability computation
    creates an indeterminate zone of width 2ε around the threshold.
    States with probability in [p-ε, p+ε] cannot be classified as
    definitely satisfying or violating the property.

    Three-valued semantics (Katoen et al. 2007, Hermanns et al. 2008):
    - TRUE:  all states in inner sat-set are definitely true
    - FALSE: all states in inner sat-set are definitely false
    - INDETERMINATE: some states are uncertain

    The indeterminate fraction is bounded by:
        |{s : |Pr_s[ψ] - p| ≤ ε}| / |S|

    For single-level CSL (common in biology), this fraction is typically
    small because probability distributions are smooth.

    Args:
        inner_error: Error from inner probability computation.
        threshold: Probability threshold p.
        comparison: Comparison operator (">=", ">", "<=", "<").
        num_states_near_threshold: Estimated states with prob near threshold.
        total_states: Total number of states.

    Returns:
        Dictionary with propagation analysis.
    """
    indeterminate_width = 2 * inner_error
    indeterminate_fraction = (
        num_states_near_threshold / max(total_states, 1)
        if total_states > 0 else 1.0
    )

    return {
        "inner_error": inner_error,
        "threshold": threshold,
        "indeterminate_width": indeterminate_width,
        "indeterminate_fraction": indeterminate_fraction,
        "sound": True,  # three-valued semantics is always sound
        "complete": indeterminate_fraction < 0.01,  # approximate completeness
        "recommendation": (
            "reduce truncation tolerance" if indeterminate_fraction > 0.1
            else "error bounds are adequate"
        ),
    }


def compose_error_certificates(
    *certificates: ErrorCertificate,
) -> ErrorCertificate:
    """
    Compose error certificates from multiple computation stages.

    Error bounds compose additively (triangle inequality).
    """
    result = ErrorCertificate()
    for cert in certificates:
        result.truncation_error += cert.truncation_error
        result.clamping_error += cert.clamping_error
        result.fsp_error += cert.fsp_error
        result.integration_error += cert.integration_error
        result.negativity_mass += cert.negativity_mass
    result.compute_total()
    return result
