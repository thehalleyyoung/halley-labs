"""Proof certificates for MARACE race-condition verification.

This module provides generation, verification, and composition of formal
proof certificates that attest to the presence or absence of race
conditions in multi-agent reinforcement-learning environments.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, ClassVar


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CertificateFormat(Enum):
    """Serialisation formats supported for proof certificates."""

    JSON = "json"
    BINARY = "binary"
    HUMAN_READABLE = "human_readable"


# ---------------------------------------------------------------------------
# Core data-class
# ---------------------------------------------------------------------------

@dataclass
class ProofCertificate:
    """Formal certificate attesting to the race-safety analysis of an environment.

    A certificate records every detail needed to reproduce or audit the
    verification result: the environment under test, the policies, the
    abstract domain, fixpoint statistics, discovered races, and a
    coverage metric over the state–schedule space.
    """

    certificate_id: str
    timestamp: datetime
    environment_id: str
    policy_ids: list[str]
    specification: str
    verdict: str  # "SAFE", "UNSAFE", or "UNKNOWN"
    coverage_fraction: float
    abstract_domain_used: str
    fixpoint_iterations: int
    races_found: list[dict[str, Any]]
    regions_verified: list[dict[str, Any]]
    hash_digest: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Allowed verdict values
    VALID_VERDICTS: ClassVar[frozenset[str]] = frozenset({"SAFE", "UNSAFE", "UNKNOWN"})

    # Certificates older than this are considered temporally invalid.
    MAX_AGE: ClassVar[timedelta] = timedelta(days=365)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the certificate to a plain dictionary."""
        return {
            "certificate_id": self.certificate_id,
            "timestamp": self.timestamp.isoformat(),
            "environment_id": self.environment_id,
            "policy_ids": list(self.policy_ids),
            "specification": self.specification,
            "verdict": self.verdict,
            "coverage_fraction": self.coverage_fraction,
            "abstract_domain_used": self.abstract_domain_used,
            "fixpoint_iterations": self.fixpoint_iterations,
            "races_found": list(self.races_found),
            "regions_verified": list(self.regions_verified),
            "hash_digest": self.hash_digest,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProofCertificate:
        """Deserialise a certificate from a plain dictionary."""
        return cls(
            certificate_id=data["certificate_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            environment_id=data["environment_id"],
            policy_ids=data["policy_ids"],
            specification=data["specification"],
            verdict=data["verdict"],
            coverage_fraction=data["coverage_fraction"],
            abstract_domain_used=data["abstract_domain_used"],
            fixpoint_iterations=data["fixpoint_iterations"],
            races_found=data["races_found"],
            regions_verified=data["regions_verified"],
            hash_digest=data["hash_digest"],
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Certificate generation
# ---------------------------------------------------------------------------

class CertificateGenerator:
    """Generate a :class:`ProofCertificate` from a fixpoint analysis result."""

    def __init__(self) -> None:
        pass

    def generate(
        self,
        environment_id: str,
        policy_ids: list[str],
        specification: str,
        fixpoint_result: dict[str, Any],
        races: list[dict[str, Any]],
        coverage: float,
    ) -> ProofCertificate:
        """Create a new proof certificate.

        Args:
            environment_id: Unique identifier for the environment analysed.
            policy_ids: Identifiers of the agent policies included.
            specification: The safety specification checked.
            fixpoint_result: Dictionary produced by the abstract-interpretation
                engine.  Expected keys: ``abstract_domain``, ``iterations``,
                ``regions``.
            races: List of race descriptors found during analysis.
            coverage: Fraction of state–schedule space verified (0–1).

        Returns:
            A fully populated :class:`ProofCertificate`.
        """
        coverage = max(0.0, min(1.0, coverage))
        verdict = self._determine_verdict(races, coverage)

        cert = ProofCertificate(
            certificate_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            environment_id=environment_id,
            policy_ids=list(policy_ids),
            specification=specification,
            verdict=verdict,
            coverage_fraction=coverage,
            abstract_domain_used=fixpoint_result.get("abstract_domain", "unknown"),
            fixpoint_iterations=int(fixpoint_result.get("iterations", 0)),
            races_found=list(races),
            regions_verified=list(fixpoint_result.get("regions", [])),
            hash_digest="",
            metadata=fixpoint_result.get("metadata", {}),
        )
        cert.hash_digest = self._compute_hash(cert)
        return cert

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _compute_hash(cert: ProofCertificate) -> str:
        """Compute a SHA-256 digest over the certificate's key fields.

        The hash covers every semantically meaningful field so that any
        post-hoc tampering is detectable by :class:`CertificateVerifier`.
        """
        payload = json.dumps(
            {
                "certificate_id": cert.certificate_id,
                "timestamp": cert.timestamp.isoformat(),
                "environment_id": cert.environment_id,
                "policy_ids": cert.policy_ids,
                "specification": cert.specification,
                "verdict": cert.verdict,
                "coverage_fraction": cert.coverage_fraction,
                "abstract_domain_used": cert.abstract_domain_used,
                "fixpoint_iterations": cert.fixpoint_iterations,
                "races_found": cert.races_found,
                "regions_verified": cert.regions_verified,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _determine_verdict(
        races: list[dict[str, Any]], coverage: float
    ) -> str:
        """Derive the overall verdict from the analysis outputs.

        * **UNSAFE** – at least one race was found.
        * **SAFE** – no races found *and* full coverage (≥ 0.999).
        * **UNKNOWN** – no races found but coverage is incomplete.
        """
        if races:
            return "UNSAFE"
        if coverage >= 0.999:
            return "SAFE"
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Certificate verification
# ---------------------------------------------------------------------------

class CertificateVerifier:
    """Independently verify the integrity and consistency of a proof certificate."""

    def verify(
        self, certificate: ProofCertificate
    ) -> tuple[bool, list[str]]:
        """Run all consistency checks on *certificate*.

        Returns:
            A tuple ``(valid, issues)`` where *valid* is ``True`` only when
            every check passes and *issues* lists human-readable descriptions
            of any problems found.
        """
        issues: list[str] = []

        if not self._check_hash_integrity(certificate):
            issues.append("Hash digest does not match recomputed value.")

        if not self._check_coverage_consistency(certificate):
            issues.append("Coverage fraction is inconsistent with verdict.")

        if not self._check_temporal_validity(certificate):
            issues.append("Certificate has expired or has a future timestamp.")

        if certificate.verdict not in ProofCertificate.VALID_VERDICTS:
            issues.append(
                f"Unknown verdict '{certificate.verdict}'; expected one of "
                f"{sorted(ProofCertificate.VALID_VERDICTS)}."
            )

        return (len(issues) == 0, issues)

    # -- individual checks --------------------------------------------------

    @staticmethod
    def _check_hash_integrity(cert: ProofCertificate) -> bool:
        """Recompute the hash and compare to the stored digest."""
        expected = CertificateGenerator._compute_hash(cert)
        return cert.hash_digest == expected

    @staticmethod
    def _check_coverage_consistency(cert: ProofCertificate) -> bool:
        """Ensure the verdict is compatible with coverage and race data.

        * SAFE requires high coverage and no races.
        * UNSAFE requires at least one race.
        * UNKNOWN requires no races but incomplete coverage.
        """
        if cert.verdict == "SAFE":
            return cert.coverage_fraction >= 0.999 and len(cert.races_found) == 0
        if cert.verdict == "UNSAFE":
            return len(cert.races_found) > 0
        if cert.verdict == "UNKNOWN":
            return len(cert.races_found) == 0
        return False

    @staticmethod
    def _check_temporal_validity(cert: ProofCertificate) -> bool:
        """Check that the certificate is neither in the future nor expired."""
        now = datetime.now(timezone.utc)
        if cert.timestamp.tzinfo is None:
            ts = cert.timestamp.replace(tzinfo=timezone.utc)
        else:
            ts = cert.timestamp
        if ts > now + timedelta(minutes=5):
            return False
        if now - ts > ProofCertificate.MAX_AGE:
            return False
        return True


# ---------------------------------------------------------------------------
# Coverage map
# ---------------------------------------------------------------------------

@dataclass
class CoverageMap:
    """Spatial map tracking which state–schedule regions have been verified.

    Each region is a dictionary with at least a ``"bounds"`` key mapping
    dimension names to ``(lo, hi)`` pairs.
    """

    dimensions: list[str]
    verified_regions: list[dict[str, Any]] = field(default_factory=list)
    unverified_regions: list[dict[str, Any]] = field(default_factory=list)
    total_volume: float = 1.0
    verified_volume: float = 0.0

    def coverage_fraction(self) -> float:
        """Return the fraction of the total volume that has been verified."""
        if self.total_volume <= 0.0:
            return 0.0
        return min(self.verified_volume / self.total_volume, 1.0)

    def find_gaps(self) -> list[dict[str, Any]]:
        """Return a list of unverified region descriptors (coverage gaps)."""
        return list(self.unverified_regions)

    def merge(self, other: CoverageMap) -> CoverageMap:
        """Merge two coverage maps into a new combined map.

        The result contains the union of verified and unverified regions
        from both maps.  Volumes are summed; duplicate regions are *not*
        deduplicated (callers should ensure disjointness).
        """
        merged_dims = sorted(set(self.dimensions) | set(other.dimensions))
        return CoverageMap(
            dimensions=merged_dims,
            verified_regions=self.verified_regions + other.verified_regions,
            unverified_regions=self.unverified_regions + other.unverified_regions,
            total_volume=self.total_volume + other.total_volume,
            verified_volume=self.verified_volume + other.verified_volume,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the coverage map to a plain dictionary."""
        return {
            "dimensions": list(self.dimensions),
            "verified_regions": list(self.verified_regions),
            "unverified_regions": list(self.unverified_regions),
            "total_volume": self.total_volume,
            "verified_volume": self.verified_volume,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoverageMap:
        """Deserialise a coverage map from a plain dictionary."""
        return cls(
            dimensions=data["dimensions"],
            verified_regions=data.get("verified_regions", []),
            unverified_regions=data.get("unverified_regions", []),
            total_volume=data.get("total_volume", 1.0),
            verified_volume=data.get("verified_volume", 0.0),
        )


# ---------------------------------------------------------------------------
# Certificate chain (compositional verification)
# ---------------------------------------------------------------------------

@dataclass
class CertificateChain:
    """An ordered chain of certificates for compositional verification.

    Multiple sub-system certificates can be composed under a stated rule
    (e.g. ``"assume-guarantee"``) to derive an overall verdict.
    """

    certificates: list[ProofCertificate] = field(default_factory=list)
    composition_rule: str = "assume-guarantee"

    def add(self, cert: ProofCertificate) -> None:
        """Append a certificate to the chain."""
        self.certificates.append(cert)

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify every certificate in the chain individually.

        Returns:
            ``(all_valid, issues)`` where *all_valid* is ``True`` only when
            every certificate passes verification and *issues* aggregates
            problems across the whole chain.
        """
        verifier = CertificateVerifier()
        all_issues: list[str] = []
        all_valid = True

        for idx, cert in enumerate(self.certificates):
            valid, issues = verifier.verify(cert)
            if not valid:
                all_valid = False
                for issue in issues:
                    all_issues.append(f"Certificate {idx} ({cert.certificate_id}): {issue}")

        if not self.certificates:
            all_valid = False
            all_issues.append("Chain is empty; at least one certificate is required.")

        return (all_valid, all_issues)

    def overall_verdict(self) -> str:
        """Derive a combined verdict from all certificates in the chain.

        * If *any* certificate is UNSAFE the overall verdict is UNSAFE.
        * If *all* certificates are SAFE the overall verdict is SAFE.
        * Otherwise the verdict is UNKNOWN.
        """
        if not self.certificates:
            return "UNKNOWN"

        verdicts = [c.verdict for c in self.certificates]
        if any(v == "UNSAFE" for v in verdicts):
            return "UNSAFE"
        if all(v == "SAFE" for v in verdicts):
            return "SAFE"
        return "UNKNOWN"

    def overall_coverage(self) -> float:
        """Compute the average coverage fraction across all certificates."""
        if not self.certificates:
            return 0.0
        return sum(c.coverage_fraction for c in self.certificates) / len(
            self.certificates
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the chain to a plain dictionary."""
        return {
            "certificates": [c.to_dict() for c in self.certificates],
            "composition_rule": self.composition_rule,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CertificateChain:
        """Deserialise a certificate chain from a plain dictionary."""
        chain = cls(composition_rule=data.get("composition_rule", "assume-guarantee"))
        for cert_data in data.get("certificates", []):
            chain.add(ProofCertificate.from_dict(cert_data))
        return chain
