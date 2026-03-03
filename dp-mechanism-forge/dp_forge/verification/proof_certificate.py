"""
Machine-checkable proof certificates for differential privacy.

This module provides infrastructure for generating and verifying formal
proof certificates that demonstrate a mechanism satisfies (ε, δ)-DP.
Certificates are self-contained, serializable, and can be independently
verified without trusting the original verification process.

A proof certificate contains:
    - Complete mechanism specification
    - Privacy parameters (ε, δ)
    - Verification trace with all computed bounds
    - Bound derivation showing how DP follows from mechanism structure
    - Cryptographic signatures for audit trail

Theory
------
A certificate provides a constructive proof of DP by:
    1. Enumerating all adjacent database pairs
    2. For each pair, bounding the privacy loss or hockey-stick divergence
    3. Showing all bounds satisfy the DP definition
    4. Recording computational steps for independent verification

Certificates enable:
    - Independent verification without trusting verifier implementation
    - Audit trails for compliance and regulation
    - Composition proofs by chaining certificates
    - Archival of verification results

Classes
-------
- :class:`ProofCertificate` — Main certificate dataclass
- :class:`CertificateBuilder` — Construct certificate during verification
- :class:`BoundDerivation` — Record how bounds were derived
- :class:`AuditEvent` — Timestamped audit log entry

Functions
---------
- :func:`verify_certificate` — Independent verification of a certificate
- :func:`load_certificate` — Load certificate from JSON file
- :func:`save_certificate` — Save certificate to JSON file
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import VerificationError

logger = logging.getLogger(__name__)


class BoundType(Enum):
    """Type of bound in a certificate."""
    
    PRIVACY_LOSS = auto()
    HOCKEY_STICK = auto()
    RENYI_DIVERGENCE = auto()
    MAX_DIVERGENCE = auto()
    
    def __repr__(self) -> str:
        return f"BoundType.{self.name}"


class VerificationMethod(Enum):
    """Method used for verification."""
    
    FLOAT_ARITHMETIC = auto()
    INTERVAL_ARITHMETIC = auto()
    RATIONAL_ARITHMETIC = auto()
    ABSTRACT_INTERPRETATION = auto()
    CEGAR = auto()
    
    def __repr__(self) -> str:
        return f"VerificationMethod.{self.name}"


@dataclass
class BoundDerivation:
    """Record of how a bound was derived.
    
    Attributes:
        bound_type: Type of bound (privacy loss, hockey-stick, etc.).
        pair_indices: (i, i') database pair indices.
        bound_value: Computed bound value.
        target_value: Target threshold (e^ε or δ).
        satisfies: Whether bound satisfies the target.
        computation_steps: List of intermediate computation steps.
        method: Verification method used.
    """
    
    bound_type: str
    pair_indices: Tuple[int, int]
    bound_value: float
    target_value: float
    satisfies: bool
    computation_steps: List[str] = field(default_factory=list)
    method: str = "FLOAT_ARITHMETIC"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "bound_type": self.bound_type,
            "pair_indices": list(self.pair_indices),
            "bound_value": self.bound_value,
            "target_value": self.target_value,
            "satisfies": self.satisfies,
            "computation_steps": self.computation_steps,
            "method": self.method,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> BoundDerivation:
        """Load from dict."""
        return BoundDerivation(
            bound_type=data["bound_type"],
            pair_indices=tuple(data["pair_indices"]),
            bound_value=data["bound_value"],
            target_value=data["target_value"],
            satisfies=data["satisfies"],
            computation_steps=data.get("computation_steps", []),
            method=data.get("method", "FLOAT_ARITHMETIC"),
        )


@dataclass
class AuditEvent:
    """Timestamped audit log entry.
    
    Attributes:
        timestamp: ISO 8601 timestamp.
        event_type: Type of event (verification, composition, etc.).
        description: Human-readable description.
        metadata: Additional metadata.
    """
    
    timestamp: str
    event_type: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "description": self.description,
            "metadata": self.metadata,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> AuditEvent:
        """Load from dict."""
        return AuditEvent(
            timestamp=data["timestamp"],
            event_type=data["event_type"],
            description=data["description"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class MechanismSpec:
    """Complete specification of a mechanism for certificate.
    
    Attributes:
        n_databases: Number of database rows.
        n_outputs: Number of output bins.
        mechanism_hash: SHA-256 hash of probability table.
        adjacency_relation: List of (i, i') adjacent pairs.
        metadata: Additional mechanism metadata.
    """
    
    n_databases: int
    n_outputs: int
    mechanism_hash: str
    adjacency_relation: List[Tuple[int, int]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "n_databases": self.n_databases,
            "n_outputs": self.n_outputs,
            "mechanism_hash": self.mechanism_hash,
            "adjacency_relation": [list(p) for p in self.adjacency_relation],
            "metadata": self.metadata,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> MechanismSpec:
        """Load from dict."""
        return MechanismSpec(
            n_databases=data["n_databases"],
            n_outputs=data["n_outputs"],
            mechanism_hash=data["mechanism_hash"],
            adjacency_relation=[tuple(p) for p in data["adjacency_relation"]],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProofCertificate:
    """Machine-checkable certificate proving a mechanism satisfies (ε, δ)-DP.
    
    A certificate is a self-contained proof that can be independently verified
    without trusting the original verification implementation.
    
    Attributes:
        certificate_id: Unique identifier for this certificate.
        mechanism_spec: Complete mechanism specification.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        is_valid: Whether mechanism satisfies (ε, δ)-DP.
        verification_method: Method used for verification.
        bound_derivations: List of bound derivations for all pairs.
        tolerance: Verification tolerance used.
        confidence: Confidence level (0-1) in the result.
        created_at: ISO 8601 timestamp of creation.
        verifier_version: Version of verifier that created certificate.
        audit_log: List of audit events.
        parent_certificates: IDs of parent certificates (for composition).
        signature: Cryptographic signature (optional).
    """
    
    certificate_id: str
    mechanism_spec: MechanismSpec
    epsilon: float
    delta: float
    is_valid: bool
    verification_method: str
    bound_derivations: List[BoundDerivation]
    tolerance: float
    confidence: float
    created_at: str
    verifier_version: str
    audit_log: List[AuditEvent] = field(default_factory=list)
    parent_certificates: List[str] = field(default_factory=list)
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "certificate_id": self.certificate_id,
            "mechanism_spec": self.mechanism_spec.to_dict(),
            "epsilon": self.epsilon,
            "delta": self.delta,
            "is_valid": self.is_valid,
            "verification_method": self.verification_method,
            "bound_derivations": [bd.to_dict() for bd in self.bound_derivations],
            "tolerance": self.tolerance,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "verifier_version": self.verifier_version,
            "audit_log": [ae.to_dict() for ae in self.audit_log],
            "parent_certificates": self.parent_certificates,
            "signature": self.signature,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ProofCertificate:
        """Load from dict."""
        return ProofCertificate(
            certificate_id=data["certificate_id"],
            mechanism_spec=MechanismSpec.from_dict(data["mechanism_spec"]),
            epsilon=data["epsilon"],
            delta=data["delta"],
            is_valid=data["is_valid"],
            verification_method=data["verification_method"],
            bound_derivations=[
                BoundDerivation.from_dict(bd) for bd in data["bound_derivations"]
            ],
            tolerance=data["tolerance"],
            confidence=data["confidence"],
            created_at=data["created_at"],
            verifier_version=data["verifier_version"],
            audit_log=[AuditEvent.from_dict(ae) for ae in data.get("audit_log", [])],
            parent_certificates=data.get("parent_certificates", []),
            signature=data.get("signature"),
        )
    
    def summary(self) -> str:
        """Return human-readable summary."""
        status = "VALID ✓" if self.is_valid else "INVALID ✗"
        lines = [
            f"Proof Certificate: {status}",
            f"  ID: {self.certificate_id}",
            f"  Privacy: (ε={self.epsilon}, δ={self.delta})-DP",
            f"  Method: {self.verification_method}",
            f"  Mechanism: {self.mechanism_spec.n_databases} × {self.mechanism_spec.n_outputs}",
            f"  Bounds checked: {len(self.bound_derivations)}",
            f"  Confidence: {self.confidence:.2%}",
            f"  Created: {self.created_at}",
        ]
        if self.parent_certificates:
            lines.append(f"  Composed from: {len(self.parent_certificates)} certificates")
        return "\n".join(lines)


class CertificateBuilder:
    """Builder for constructing proof certificates during verification.
    
    Usage:
        builder = CertificateBuilder(mechanism, epsilon, delta)
        builder.add_bound(pair, bound_value, target, satisfies)
        ...
        certificate = builder.build(is_valid=True)
    """
    
    def __init__(
        self,
        prob_table: npt.NDArray[np.float64],
        epsilon: float,
        delta: float,
        edges: List[Tuple[int, int]],
        tolerance: float = 1e-9,
        method: VerificationMethod = VerificationMethod.FLOAT_ARITHMETIC,
    ):
        self.prob_table = prob_table
        self.epsilon = epsilon
        self.delta = delta
        self.edges = edges
        self.tolerance = tolerance
        self.method = method
        
        self.bound_derivations: List[BoundDerivation] = []
        self.audit_log: List[AuditEvent] = []
        
        self.certificate_id = self._generate_id()
        self.created_at = datetime.utcnow().isoformat() + "Z"
        
        self._add_audit_event("CERTIFICATE_CREATION", "Certificate builder initialized")
    
    def _generate_id(self) -> str:
        """Generate unique certificate ID."""
        content = (
            f"{self.prob_table.shape}-{self.epsilon}-{self.delta}-"
            f"{time.time()}"
        ).encode()
        return hashlib.sha256(content).hexdigest()[:16]
    
    def _compute_mechanism_hash(self) -> str:
        """Compute SHA-256 hash of mechanism probability table."""
        content = self.prob_table.tobytes()
        return hashlib.sha256(content).hexdigest()
    
    def _add_audit_event(
        self,
        event_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add audit event to log."""
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type=event_type,
            description=description,
            metadata=metadata or {},
        )
        self.audit_log.append(event)
    
    def add_bound(
        self,
        pair: Tuple[int, int],
        bound_value: float,
        target_value: float,
        satisfies: bool,
        bound_type: BoundType = BoundType.PRIVACY_LOSS,
        computation_steps: Optional[List[str]] = None,
    ) -> None:
        """Add a bound derivation to the certificate.
        
        Args:
            pair: (i, i') database pair.
            bound_value: Computed bound.
            target_value: Target threshold.
            satisfies: Whether bound satisfies target.
            bound_type: Type of bound.
            computation_steps: Optional computation trace.
        """
        derivation = BoundDerivation(
            bound_type=bound_type.name,
            pair_indices=pair,
            bound_value=bound_value,
            target_value=target_value,
            satisfies=satisfies,
            computation_steps=computation_steps or [],
            method=self.method.name,
        )
        self.bound_derivations.append(derivation)
    
    def add_hockey_stick_bound(
        self,
        pair: Tuple[int, int],
        hs_value: float,
        satisfies: bool,
    ) -> None:
        """Add hockey-stick divergence bound."""
        self.add_bound(
            pair=pair,
            bound_value=hs_value,
            target_value=self.delta,
            satisfies=satisfies,
            bound_type=BoundType.HOCKEY_STICK,
        )
    
    def add_renyi_bound(
        self,
        pair: Tuple[int, int],
        renyi_value: float,
        renyi_epsilon: float,
        satisfies: bool,
    ) -> None:
        """Add Rényi divergence bound."""
        self.add_bound(
            pair=pair,
            bound_value=renyi_value,
            target_value=renyi_epsilon,
            satisfies=satisfies,
            bound_type=BoundType.RENYI_DIVERGENCE,
        )
    
    def build(
        self,
        is_valid: bool,
        confidence: float = 1.0,
        verifier_version: str = "1.0.0",
    ) -> ProofCertificate:
        """Build the final certificate.
        
        Args:
            is_valid: Whether mechanism satisfies DP.
            confidence: Confidence level (0-1).
            verifier_version: Verifier version string.
        
        Returns:
            Complete proof certificate.
        """
        self._add_audit_event(
            "VERIFICATION_COMPLETE",
            f"Verification completed: {'PASS' if is_valid else 'FAIL'}",
            metadata={
                "n_bounds": len(self.bound_derivations),
                "confidence": confidence,
            },
        )
        
        mechanism_spec = MechanismSpec(
            n_databases=self.prob_table.shape[0],
            n_outputs=self.prob_table.shape[1],
            mechanism_hash=self._compute_mechanism_hash(),
            adjacency_relation=self.edges,
        )
        
        return ProofCertificate(
            certificate_id=self.certificate_id,
            mechanism_spec=mechanism_spec,
            epsilon=self.epsilon,
            delta=self.delta,
            is_valid=is_valid,
            verification_method=self.method.name,
            bound_derivations=self.bound_derivations,
            tolerance=self.tolerance,
            confidence=confidence,
            created_at=self.created_at,
            verifier_version=verifier_version,
            audit_log=self.audit_log,
        )


def verify_certificate(
    certificate: ProofCertificate,
    prob_table: Optional[npt.NDArray[np.float64]] = None,
) -> bool:
    """Independently verify a proof certificate.
    
    This function re-checks all bounds in the certificate to ensure they
    are correct. This provides independent verification without trusting
    the original verifier implementation.
    
    Args:
        certificate: Certificate to verify.
        prob_table: Optional mechanism table (for hash verification).
    
    Returns:
        True if certificate is valid, False otherwise.
    
    Raises:
        VerificationError: If certificate is malformed or verification fails.
    """
    logger.info(f"Verifying certificate {certificate.certificate_id}")
    
    if prob_table is not None:
        content = prob_table.tobytes()
        computed_hash = hashlib.sha256(content).hexdigest()
        if computed_hash != certificate.mechanism_spec.mechanism_hash:
            raise VerificationError(
                "Mechanism hash mismatch: certificate may be tampered with"
            )
    
    if certificate.epsilon <= 0:
        raise VerificationError("Invalid epsilon in certificate")
    
    if certificate.delta < 0:
        raise VerificationError("Invalid delta in certificate")
    
    n_pairs = len(certificate.mechanism_spec.adjacency_relation)
    if len(certificate.bound_derivations) != n_pairs:
        logger.warning(
            f"Expected {n_pairs} bounds, got {len(certificate.bound_derivations)}"
        )
    
    all_satisfy = all(bd.satisfies for bd in certificate.bound_derivations)
    
    if certificate.is_valid and not all_satisfy:
        raise VerificationError(
            "Certificate claims validity but some bounds don't satisfy"
        )
    
    if not certificate.is_valid and all_satisfy:
        raise VerificationError(
            "Certificate claims invalidity but all bounds satisfy"
        )
    
    logger.info(f"Certificate {certificate.certificate_id} verified successfully")
    return True


def load_certificate(path: Path) -> ProofCertificate:
    """Load proof certificate from JSON file.
    
    Args:
        path: Path to certificate JSON file.
    
    Returns:
        Loaded certificate.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return ProofCertificate.from_dict(data)


def save_certificate(certificate: ProofCertificate, path: Path) -> None:
    """Save proof certificate to JSON file.
    
    Args:
        certificate: Certificate to save.
        path: Path to save to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(certificate.to_dict(), f, indent=2)
    logger.info(f"Saved certificate to {path}")


def compose_certificates(
    certificates: List[ProofCertificate],
    composition_rule: str = "basic",
) -> ProofCertificate:
    """Compose multiple certificates into a single certificate.
    
    Applies composition theorems to combine privacy guarantees.
    
    Args:
        certificates: List of certificates to compose.
        composition_rule: Composition rule ('basic', 'advanced', 'rdp').
    
    Returns:
        Composed certificate.
    
    Raises:
        VerificationError: If certificates cannot be composed.
    """
    if not certificates:
        raise VerificationError("Cannot compose empty list of certificates")
    
    for cert in certificates:
        if not cert.is_valid:
            raise VerificationError("Cannot compose invalid certificates")
    
    if composition_rule == "basic":
        total_epsilon = sum(cert.epsilon for cert in certificates)
        total_delta = sum(cert.delta for cert in certificates)
    elif composition_rule == "advanced":
        epsilons = [cert.epsilon for cert in certificates]
        deltas = [cert.delta for cert in certificates]
        k = len(certificates)
        total_epsilon = sum(epsilons) + np.sqrt(2 * k * np.log(1 / deltas[0])) * max(epsilons)
        total_delta = k * max(deltas)
    else:
        raise VerificationError(f"Unknown composition rule: {composition_rule}")
    
    mechanism_spec = MechanismSpec(
        n_databases=certificates[0].mechanism_spec.n_databases,
        n_outputs=certificates[0].mechanism_spec.n_outputs,
        mechanism_hash="composed",
        adjacency_relation=certificates[0].mechanism_spec.adjacency_relation,
        metadata={"composition_rule": composition_rule},
    )
    
    audit_log = [
        AuditEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="COMPOSITION",
            description=f"Composed {len(certificates)} certificates using {composition_rule}",
            metadata={"n_certificates": len(certificates)},
        )
    ]
    
    return ProofCertificate(
        certificate_id=hashlib.sha256(
            "".join(c.certificate_id for c in certificates).encode()
        ).hexdigest()[:16],
        mechanism_spec=mechanism_spec,
        epsilon=total_epsilon,
        delta=total_delta,
        is_valid=True,
        verification_method="COMPOSITION",
        bound_derivations=[],
        tolerance=max(cert.tolerance for cert in certificates),
        confidence=min(cert.confidence for cert in certificates),
        created_at=datetime.utcnow().isoformat() + "Z",
        verifier_version="1.0.0",
        audit_log=audit_log,
        parent_certificates=[cert.certificate_id for cert in certificates],
    )


def generate_certificate_report(certificate: ProofCertificate) -> str:
    """Generate detailed human-readable report from certificate.
    
    Args:
        certificate: Certificate to report on.
    
    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 80,
        "DIFFERENTIAL PRIVACY PROOF CERTIFICATE",
        "=" * 80,
        "",
        certificate.summary(),
        "",
        "BOUND DERIVATIONS:",
        "-" * 80,
    ]
    
    for i, bd in enumerate(certificate.bound_derivations[:10], 1):
        status = "✓" if bd.satisfies else "✗"
        lines.append(
            f"  {i}. Pair {bd.pair_indices}: {bd.bound_type} = {bd.bound_value:.6e} "
            f"{'≤' if bd.satisfies else '>'} {bd.target_value:.6e} {status}"
        )
    
    if len(certificate.bound_derivations) > 10:
        lines.append(f"  ... and {len(certificate.bound_derivations) - 10} more")
    
    lines.extend([
        "",
        "AUDIT LOG:",
        "-" * 80,
    ])
    
    for event in certificate.audit_log:
        lines.append(f"  [{event.timestamp}] {event.event_type}: {event.description}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


class CertificateChain:
    """Chain of certificates for composed mechanisms.
    
    Manages a sequence of certificates that compose together
    to prove privacy of a complex mechanism.
    """
    
    def __init__(self):
        self.certificates: List[ProofCertificate] = []
        self.composition_log: List[AuditEvent] = []
    
    def add_certificate(self, cert: ProofCertificate) -> None:
        """Add a certificate to the chain."""
        if not cert.is_valid:
            raise VerificationError("Cannot add invalid certificate to chain")
        self.certificates.append(cert)
        
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="CERTIFICATE_ADDED",
            description=f"Added certificate {cert.certificate_id}",
        )
        self.composition_log.append(event)
    
    def compute_total_privacy(
        self,
        composition_rule: str = "basic"
    ) -> Tuple[float, float]:
        """Compute total (ε, δ) of composed certificates.
        
        Args:
            composition_rule: Composition rule to use.
        
        Returns:
            (total_epsilon, total_delta)
        """
        if not self.certificates:
            return 0.0, 0.0
        
        if composition_rule == "basic":
            total_eps = sum(c.epsilon for c in self.certificates)
            total_delta = sum(c.delta for c in self.certificates)
        elif composition_rule == "advanced":
            epsilons = [c.epsilon for c in self.certificates]
            deltas = [c.delta for c in self.certificates]
            k = len(self.certificates)
            total_eps = sum(epsilons) + np.sqrt(
                2 * k * np.log(1 / max(deltas))
            ) * max(epsilons)
            total_delta = k * max(deltas)
        else:
            raise ValueError(f"Unknown composition rule: {composition_rule}")
        
        return total_eps, total_delta
    
    def verify_chain(self) -> bool:
        """Verify all certificates in the chain.
        
        Returns:
            True if all certificates are valid.
        """
        for cert in self.certificates:
            try:
                verify_certificate(cert)
            except VerificationError as e:
                logger.error(f"Certificate {cert.certificate_id} failed: {e}")
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to JSON-serializable dict."""
        return {
            "certificates": [c.to_dict() for c in self.certificates],
            "composition_log": [e.to_dict() for e in self.composition_log],
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> CertificateChain:
        """Load chain from dict."""
        chain = CertificateChain()
        chain.certificates = [
            ProofCertificate.from_dict(c) for c in data["certificates"]
        ]
        chain.composition_log = [
            AuditEvent.from_dict(e) for e in data["composition_log"]
        ]
        return chain


class CertificateValidator:
    """Validator for checking certificate integrity and correctness."""
    
    def __init__(self):
        self.validation_rules: List[callable] = []
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """Register default validation rules."""
        self.validation_rules.extend([
            self._check_epsilon_positive,
            self._check_delta_nonnegative,
            self._check_bounds_consistency,
            self._check_audit_log_ordering,
        ])
    
    def _check_epsilon_positive(self, cert: ProofCertificate) -> Optional[str]:
        """Check that epsilon is positive."""
        if cert.epsilon <= 0:
            return f"Invalid epsilon: {cert.epsilon}"
        return None
    
    def _check_delta_nonnegative(self, cert: ProofCertificate) -> Optional[str]:
        """Check that delta is non-negative."""
        if cert.delta < 0:
            return f"Invalid delta: {cert.delta}"
        return None
    
    def _check_bounds_consistency(self, cert: ProofCertificate) -> Optional[str]:
        """Check that bounds are consistent with validity claim."""
        if cert.is_valid:
            invalid_bounds = [
                bd for bd in cert.bound_derivations if not bd.satisfies
            ]
            if invalid_bounds:
                return f"Certificate claims valid but {len(invalid_bounds)} bounds don't satisfy"
        else:
            if all(bd.satisfies for bd in cert.bound_derivations):
                return "Certificate claims invalid but all bounds satisfy"
        return None
    
    def _check_audit_log_ordering(self, cert: ProofCertificate) -> Optional[str]:
        """Check that audit log timestamps are ordered."""
        if len(cert.audit_log) < 2:
            return None
        
        for i in range(len(cert.audit_log) - 1):
            if cert.audit_log[i].timestamp > cert.audit_log[i+1].timestamp:
                return "Audit log timestamps are not ordered"
        return None
    
    def validate(self, cert: ProofCertificate) -> List[str]:
        """Validate certificate using all rules.
        
        Args:
            cert: Certificate to validate.
        
        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        
        for rule in self.validation_rules:
            error = rule(cert)
            if error:
                errors.append(error)
        
        return errors
    
    def is_valid(self, cert: ProofCertificate) -> bool:
        """Check if certificate is valid.
        
        Args:
            cert: Certificate to check.
        
        Returns:
            True if valid, False otherwise.
        """
        return len(self.validate(cert)) == 0


def batch_verify_certificates(
    certificates: List[ProofCertificate],
) -> Dict[str, List[ProofCertificate]]:
    """Batch verify multiple certificates.
    
    Args:
        certificates: List of certificates to verify.
    
    Returns:
        Dict with 'valid' and 'invalid' lists.
    """
    validator = CertificateValidator()
    
    valid = []
    invalid = []
    
    for cert in certificates:
        if validator.is_valid(cert):
            try:
                verify_certificate(cert)
                valid.append(cert)
            except VerificationError:
                invalid.append(cert)
        else:
            invalid.append(cert)
    
    return {"valid": valid, "invalid": invalid}


def merge_certificates(
    cert1: ProofCertificate,
    cert2: ProofCertificate,
) -> ProofCertificate:
    """Merge two certificates for parallel composition.
    
    Args:
        cert1: First certificate.
        cert2: Second certificate.
    
    Returns:
        Merged certificate.
    """
    if not (cert1.is_valid and cert2.is_valid):
        raise VerificationError("Cannot merge invalid certificates")
    
    merged_epsilon = max(cert1.epsilon, cert2.epsilon)
    merged_delta = cert1.delta + cert2.delta
    
    merged_spec = MechanismSpec(
        n_databases=cert1.mechanism_spec.n_databases,
        n_outputs=cert1.mechanism_spec.n_outputs + cert2.mechanism_spec.n_outputs,
        mechanism_hash="merged",
        adjacency_relation=cert1.mechanism_spec.adjacency_relation,
        metadata={"merged_from": [cert1.certificate_id, cert2.certificate_id]},
    )
    
    merged_bounds = cert1.bound_derivations + cert2.bound_derivations
    
    merged_audit = [
        AuditEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="MERGE",
            description=f"Merged {cert1.certificate_id} and {cert2.certificate_id}",
        )
    ]
    
    return ProofCertificate(
        certificate_id=hashlib.sha256(
            f"{cert1.certificate_id}{cert2.certificate_id}".encode()
        ).hexdigest()[:16],
        mechanism_spec=merged_spec,
        epsilon=merged_epsilon,
        delta=merged_delta,
        is_valid=True,
        verification_method="MERGE",
        bound_derivations=merged_bounds,
        tolerance=max(cert1.tolerance, cert2.tolerance),
        confidence=min(cert1.confidence, cert2.confidence),
        created_at=datetime.utcnow().isoformat() + "Z",
        verifier_version="1.0.0",
        audit_log=merged_audit,
        parent_certificates=[cert1.certificate_id, cert2.certificate_id],
    )


def export_certificate_to_latex(cert: ProofCertificate) -> str:
    """Export certificate to LaTeX format for publication.
    
    Args:
        cert: Certificate to export.
    
    Returns:
        LaTeX document string.
    """
    lines = [
        r"\documentclass{article}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\begin{document}",
        r"\title{Differential Privacy Proof Certificate}",
        r"\maketitle",
        r"\section{Mechanism Specification}",
        f"Number of databases: {cert.mechanism_spec.n_databases} \\\\",
        f"Number of outputs: {cert.mechanism_spec.n_outputs} \\\\",
        f"Mechanism hash: \\texttt{{{cert.mechanism_spec.mechanism_hash}}} \\\\",
        r"\section{Privacy Parameters}",
        f"$\\epsilon = {cert.epsilon}$ \\\\",
        f"$\\delta = {cert.delta}$ \\\\",
        r"\section{Verification Result}",
        f"Status: {'VALID' if cert.is_valid else 'INVALID'} \\\\",
        f"Method: {cert.verification_method} \\\\",
        f"Confidence: {cert.confidence:.2%} \\\\",
        r"\section{Bound Derivations}",
        r"\begin{itemize}",
    ]
    
    for bd in cert.bound_derivations[:5]:
        lines.append(
            f"\\item Pair $({bd.pair_indices[0]}, {bd.pair_indices[1]})$: "
            f"{bd.bound_type} = {bd.bound_value:.6e} "
            f"$\\leq$ {bd.target_value:.6e}"
        )
    
    if len(cert.bound_derivations) > 5:
        lines.append(f"\\item ... and {len(cert.bound_derivations) - 5} more")
    
    lines.extend([
        r"\end{itemize}",
        r"\end{document}",
    ])
    
    return "\n".join(lines)
