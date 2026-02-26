"""
CoaCert-TLA Witness Verifier Module.

Standalone verification of Merkle-hashed bisimulation witnesses.
Validates hash chains, bisimulation closure, stuttering equivalence,
fairness preservation, and generates verification reports.
"""

from .deserializer import WitnessDeserializer, WitnessData, WitnessHeader
from .hash_verifier import HashChainVerifier, HashVerificationResult
from .closure_validator import ClosureValidator, ClosureResult
from .stuttering_verifier import StutteringVerifier, StutteringResult
from .fairness_verifier import FairnessVerifier, FairnessResult
from .verification_report import VerificationReport, Verdict
from .typed_verifier import (
    TypedWitnessVerifier,
    TypedVerificationReport,
    PhaseVerdict,
    PhaseStatus,
    TypeCheckError,
    checked,
)
from .independent_checker import (
    IndependentChecker,
    CrossValidationReport,
    IndependentPhaseResult,
    VerificationDiscrepancy,
)

__all__ = [
    "WitnessDeserializer",
    "WitnessData",
    "WitnessHeader",
    "HashChainVerifier",
    "HashVerificationResult",
    "ClosureValidator",
    "ClosureResult",
    "StutteringVerifier",
    "StutteringResult",
    "FairnessVerifier",
    "FairnessResult",
    "VerificationReport",
    "Verdict",
    "TypedWitnessVerifier",
    "TypedVerificationReport",
    "PhaseVerdict",
    "PhaseStatus",
    "TypeCheckError",
    "checked",
    "IndependentChecker",
    "CrossValidationReport",
    "IndependentPhaseResult",
    "VerificationDiscrepancy",
]


def verify_witness(witness_path: str, *, partial: bool = False,
                   sample_fraction: float = 0.1) -> "VerificationReport":
    """Top-level convenience: deserialize + verify all aspects + generate report."""
    deserializer = WitnessDeserializer()
    witness = deserializer.deserialize_file(witness_path)

    report = VerificationReport(witness)

    hash_result = HashChainVerifier(witness).verify_full()
    report.add_hash_result(hash_result)

    closure_result = ClosureValidator(witness).validate_full()
    report.add_closure_result(closure_result)

    stuttering_result = StutteringVerifier(witness).verify()
    report.add_stuttering_result(stuttering_result)

    fairness_result = FairnessVerifier(witness).verify()
    report.add_fairness_result(fairness_result)

    report.finalize()
    return report
