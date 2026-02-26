"""
Error certification and tracking for TT-compressed CSL model checking.

Implements:
- Non-negativity-preserving TT rounding (alternating projections)
- Clamping error bounds (Proposition 1)
- Error propagation through Markov semigroup (Theorem 1)
- Per-bond truncation error tracking
- Richardson extrapolation for convergence estimation
"""

from tn_check.error.certification import (
    ErrorCertificate,
    ErrorTracker,
    ClampingProof,
    ClampingProofIteration,
    nonneg_preserving_round,
    clamping_error_bound,
    tight_clamping_bound,
    verify_clamping_proposition,
)
from tn_check.error.propagation import (
    semigroup_error_bound,
    csl_error_propagation,
)

__all__ = [
    "ErrorCertificate",
    "ErrorTracker",
    "ClampingProof",
    "ClampingProofIteration",
    "nonneg_preserving_round",
    "clamping_error_bound",
    "tight_clamping_bound",
    "verify_clamping_proposition",
    "semigroup_error_bound",
    "csl_error_propagation",
]
