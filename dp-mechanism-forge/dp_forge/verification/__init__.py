"""
Formal verification layer for DP-Forge.

This package provides rigorous, sound verification of differential privacy
guarantees with multiple verification backends:

- **Interval Verifier**: Sound verification using vectorized interval arithmetic
- **Rational Verifier**: Exact verification using arbitrary-precision arithmetic
- **Proof Certificates**: Machine-checkable certificates of DP guarantees
- **Abstract Interpretation**: Abstract domain analysis for privacy properties
- **CEGAR**: Counterexample-guided abstraction refinement

All verifiers produce results with explicit soundness levels and confidence
bounds. The verification layer integrates with the main CEGIS loop to provide
rigorous privacy guarantees.
"""

from dp_forge.verification.interval_verifier import (
    IntervalVerifier,
    interval_hockey_stick,
    interval_renyi_divergence,
    sound_verify_dp,
)
from dp_forge.verification.proof_certificate import (
    CertificateBuilder,
    ProofCertificate,
    verify_certificate,
)
from dp_forge.verification.abstract_interpretation import (
    AbstractDomain,
    IntervalDomain,
    OctagonDomain,
    PrivacyAbstractTransformer,
    fixpoint_iteration,
)
from dp_forge.verification.cegar import (
    CEGARVerifier,
    AbstractionRefinement,
    InterpolantComputer,
    PredicateAbstraction,
)
from dp_forge.verification.rational_verifier import (
    RationalVerifier,
    exact_privacy_loss,
    exact_hockey_stick,
)

__all__ = [
    # Interval verification
    "IntervalVerifier",
    "interval_hockey_stick",
    "interval_renyi_divergence",
    "sound_verify_dp",
    # Proof certificates
    "ProofCertificate",
    "CertificateBuilder",
    "verify_certificate",
    # Abstract interpretation
    "AbstractDomain",
    "IntervalDomain",
    "OctagonDomain",
    "PrivacyAbstractTransformer",
    "fixpoint_iteration",
    # CEGAR
    "CEGARVerifier",
    "AbstractionRefinement",
    "InterpolantComputer",
    "PredicateAbstraction",
    # Rational verification
    "RationalVerifier",
    "exact_privacy_loss",
    "exact_hockey_stick",
]
