"""Standalone certificate verifier for BioProver.

This module provides an independent, minimal-TCB verifier that can replay
and validate proof certificates produced by BioProver without trusting
any of BioProver's solver infrastructure. The verifier only depends on:
  - mpmath for arbitrary-precision interval arithmetic
  - json for certificate parsing
  - hashlib for integrity checking

The verifier checks:
1. Flowpipe enclosures: that each segment is a valid over-approximation
   by re-integrating with validated arithmetic and checking containment.
2. Invariant certificates: that predicates hold over all flowpipe segments.
3. SMT certificates: that Z3 proof trees are structurally valid.
4. Compositional certificates: that assume-guarantee chains are well-formed.
5. Error budget certificates: that (ε,δ) error propagation is sound.
"""

from bioprover.certificate_verifier.verifier import (
    CertificateVerifier,
    VerificationReport,
    FlowpipeReplayVerifier,
    InvariantReplayVerifier,
    ErrorBudgetVerifier,
    CompositionalVerifier,
    verify_certificate_file,
    verify_certificate_dict,
)

__all__ = [
    "CertificateVerifier",
    "VerificationReport",
    "FlowpipeReplayVerifier",
    "InvariantReplayVerifier",
    "ErrorBudgetVerifier",
    "CompositionalVerifier",
    "verify_certificate_file",
    "verify_certificate_dict",
]
