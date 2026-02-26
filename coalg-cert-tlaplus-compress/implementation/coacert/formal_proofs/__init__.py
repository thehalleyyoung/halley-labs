"""
Formal proofs module for CoaCert-TLA.

Provides constructive proof witnesses, formal theorem statements,
and verification of categorical properties for the coalgebraic
compression framework.
"""

from .tfair_theorem import (
    TFairTheorem,
    TFairProofWitness,
    TFairProofObligation,
    PreservationTheorem,
    PreservationProofWitness,
    ProofCertificate,
    TFairCoherenceProver,
)
from .preservation_prover import (
    PreservationProver,
    PreservationCertificate,
)
from .coherence_certificate import (
    CoherenceCertificate,
    CoherenceCertificateBuilder,
    DistributiveLawWitness,
)
from .minimality_proof import (
    MinimalityProof,
    MinimalityWitness,
    MyHillNerodeWitness,
)
from .conformance_certificate import (
    ConformanceCertificate,
    ConformanceCertificateBuilder,
    DepthSufficiencyProof,
)
from .categorical_diagram import (
    CategoricalDiagramVerifier,
    DiagramVerificationResult,
    NaturalityWitness,
)
from .ctl_star_preservation import (
    CTLStarPreservationProof,
    FormulaInductionStep,
    FormulaNode,
    FormulaKind,
    StreettAcceptancePreservation,
)
from .proof_obligation_tracker import (
    ProofObligationTracker,
    ProofObligation,
    ObligationCategory,
    DischargeStatus,
)

__all__ = [
    "TFairTheorem",
    "TFairProofWitness",
    "TFairProofObligation",
    "PreservationTheorem",
    "PreservationProofWitness",
    "ProofCertificate",
    "TFairCoherenceProver",
    "PreservationProver",
    "PreservationCertificate",
    "CoherenceCertificate",
    "CoherenceCertificateBuilder",
    "DistributiveLawWitness",
    "MinimalityProof",
    "MinimalityWitness",
    "MyHillNerodeWitness",
    "ConformanceCertificate",
    "ConformanceCertificateBuilder",
    "DepthSufficiencyProof",
    "CategoricalDiagramVerifier",
    "DiagramVerificationResult",
    "NaturalityWitness",
    "CTLStarPreservationProof",
    "FormulaInductionStep",
    "FormulaNode",
    "FormulaKind",
    "StreettAcceptancePreservation",
    "ProofObligationTracker",
    "ProofObligation",
    "ObligationCategory",
    "DischargeStatus",
]
