"""
CertificateEmitter: generate, validate, and serialise machine-checkable
proof certificates for verified causal-inference steps.

Each certificate is a sequence of (assertion, proof_tag, metadata) triples
produced during a verification session. Certificates can be chained
together for subgraph-to-global proofs and serialised to disk for
offline re-checking.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import z3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CertificateStep:
    """One verified assertion inside a certificate."""
    step_index: int
    assertion: str
    proof_tag: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.digest:
            self.digest = self._compute_digest()

    def _compute_digest(self) -> str:
        payload = f"{self.step_index}|{self.assertion}|{self.proof_tag}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "assertion": self.assertion,
            "proof_tag": self.proof_tag,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "digest": self.digest,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CertificateStep":
        return CertificateStep(
            step_index=d["step_index"],
            assertion=d["assertion"],
            proof_tag=d["proof_tag"],
            timestamp=d["timestamp"],
            metadata=d.get("metadata", {}),
            digest=d.get("digest", ""),
        )


@dataclass
class CertificateStats:
    """Summary statistics for a certificate."""
    total_steps: int = 0
    verified_steps: int = 0
    failed_steps: int = 0
    creation_time: float = 0.0
    finalization_time: float = 0.0
    chain_depth: int = 0
    total_digest_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Certificate:
    """
    Machine-checkable proof certificate.

    A certificate records a linear sequence of verification steps,
    each consisting of an assertion string, a proof tag (e.g. the
    SMT solver's ``sat``/``unsat`` result), and optional metadata.
    The certificate can be validated offline by replaying the
    assertions into a fresh solver.
    """
    certificate_id: str = ""
    session_id: str = ""
    steps: List[CertificateStep] = field(default_factory=list)
    finalized: bool = False
    root_digest: str = ""
    creation_time: float = 0.0
    finalization_time: float = 0.0
    parent_ids: List[str] = field(default_factory=list)
    stats: CertificateStats = field(default_factory=CertificateStats)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "session_id": self.session_id,
            "steps": [s.to_dict() for s in self.steps],
            "finalized": self.finalized,
            "root_digest": self.root_digest,
            "creation_time": self.creation_time,
            "finalization_time": self.finalization_time,
            "parent_ids": self.parent_ids,
            "stats": self.stats.to_dict(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Certificate":
        steps = [CertificateStep.from_dict(s) for s in d.get("steps", [])]
        stats = CertificateStats(**d.get("stats", {}))
        return Certificate(
            certificate_id=d.get("certificate_id", ""),
            session_id=d.get("session_id", ""),
            steps=steps,
            finalized=d.get("finalized", False),
            root_digest=d.get("root_digest", ""),
            creation_time=d.get("creation_time", 0.0),
            finalization_time=d.get("finalization_time", 0.0),
            parent_ids=d.get("parent_ids", []),
            stats=stats,
        )


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of re-checking a certificate offline."""
    valid: bool
    total_steps: int = 0
    checked_steps: int = 0
    failed_steps: int = 0
    digest_mismatches: int = 0
    messages: List[str] = field(default_factory=list)
    solver_results: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CertificateEmitter
# ---------------------------------------------------------------------------

class CertificateEmitter:
    """
    Emit machine-checkable proof certificates during verification.

    Usage
    -----
    >>> emitter = CertificateEmitter(session_id="s1")
    >>> emitter.create_certificate()
    >>> emitter.add_step("x >= 0", "verified:step_0")
    >>> emitter.finalize()
    >>> cert = emitter.get_certificate()
    >>> CertificateEmitter.validate(cert)
    """

    def __init__(self, session_id: str = "") -> None:
        self._session_id = session_id
        self._certificate: Optional[Certificate] = None
        self._step_counter = 0

    # ------------------------------------------------------------------
    # Certificate lifecycle
    # ------------------------------------------------------------------

    def create_certificate(
        self,
        certificate_id: Optional[str] = None,
        parent_ids: Optional[List[str]] = None,
    ) -> Certificate:
        """
        Create a new empty certificate and make it the active certificate.
        """
        cid = certificate_id or uuid.uuid4().hex[:16]
        now = time.time()
        self._certificate = Certificate(
            certificate_id=cid,
            session_id=self._session_id,
            creation_time=now,
            parent_ids=parent_ids or [],
        )
        self._certificate.stats.creation_time = now
        self._step_counter = 0
        return self._certificate

    def add_step(
        self,
        assertion: str,
        proof: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CertificateStep:
        """
        Append a verified step to the active certificate.

        Parameters
        ----------
        assertion : str
            Human-readable string representation of the Z3 assertion.
        proof : str
            Proof tag, typically ``"verified:<step_id>"``.
        metadata : dict, optional
            Arbitrary metadata (sender, receiver, bound values, etc.).
        """
        if self._certificate is None:
            raise RuntimeError("No active certificate – call create_certificate() first")
        if self._certificate.finalized:
            raise RuntimeError("Certificate already finalized")

        step = CertificateStep(
            step_index=self._step_counter,
            assertion=assertion,
            proof_tag=proof,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._certificate.steps.append(step)
        self._step_counter += 1
        return step

    def finalize(self) -> Certificate:
        """
        Finalize the active certificate: compute root digest, freeze steps.
        """
        if self._certificate is None:
            raise RuntimeError("No active certificate")
        if self._certificate.finalized:
            return self._certificate

        self._certificate.finalization_time = time.time()
        self._certificate.root_digest = self._compute_root_digest(
            self._certificate.steps
        )
        self._certificate.finalized = True

        # Fill stats
        st = self._certificate.stats
        st.total_steps = len(self._certificate.steps)
        st.verified_steps = sum(
            1 for s in self._certificate.steps if "verified" in s.proof_tag
        )
        st.failed_steps = st.total_steps - st.verified_steps
        st.finalization_time = self._certificate.finalization_time
        st.total_digest_bytes = sum(
            len(s.digest) for s in self._certificate.steps
        )
        st.chain_depth = len(self._certificate.parent_ids)

        return self._certificate

    def get_certificate(self) -> Optional[Certificate]:
        """Return the current (possibly non-finalized) certificate."""
        return self._certificate

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate(
        certificate: Certificate,
        timeout_ms: int = 10_000,
        re_check_smt: bool = False,
    ) -> ValidationResult:
        """
        Validate a certificate by re-checking digests and (optionally)
        replaying assertions into a fresh Z3 solver.

        Parameters
        ----------
        certificate : Certificate
            The certificate to validate.
        timeout_ms : int
            Z3 timeout per check (only used when *re_check_smt* is True).
        re_check_smt : bool
            If True, parse and re-check each assertion in Z3.

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult(
            valid=True,
            total_steps=len(certificate.steps),
        )

        if not certificate.finalized:
            result.valid = False
            result.messages.append("Certificate not finalized")
            return result

        # 1. Digest verification
        for step in certificate.steps:
            expected = step._compute_digest()
            if step.digest != expected:
                result.digest_mismatches += 1
                result.messages.append(
                    f"Step {step.step_index}: digest mismatch"
                )

        # Root digest
        root = CertificateEmitter._compute_root_digest(certificate.steps)
        if root != certificate.root_digest:
            result.valid = False
            result.messages.append("Root digest mismatch")

        # 2. Step-index monotonicity
        for i, step in enumerate(certificate.steps):
            if step.step_index != i:
                result.valid = False
                result.messages.append(
                    f"Non-monotonic step index at position {i}"
                )

        # 3. Timestamp monotonicity
        for i in range(1, len(certificate.steps)):
            if certificate.steps[i].timestamp < certificate.steps[i - 1].timestamp:
                result.messages.append(
                    f"Timestamp regression at step {i}"
                )

        # 4. Optional SMT replay
        if re_check_smt:
            solver = z3.Solver()
            solver.set("timeout", timeout_ms)
            for step in certificate.steps:
                result.checked_steps += 1
                try:
                    expr = z3.parse_smt2_string(
                        f"(assert {step.assertion})",
                    )
                    if expr:
                        solver.push()
                        for e in expr:
                            solver.add(e)
                        chk = solver.check()
                        result.solver_results.append(str(chk))
                        if chk == z3.unsat:
                            result.failed_steps += 1
                            result.messages.append(
                                f"Step {step.step_index} unsat on replay"
                            )
                        solver.pop()
                except z3.Z3Exception:
                    result.messages.append(
                        f"Step {step.step_index}: could not parse assertion"
                    )

        if result.digest_mismatches > 0:
            result.valid = False
        if result.failed_steps > 0:
            result.valid = False

        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def serialize(certificate: Certificate, path: str) -> None:
        """Serialize a certificate to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(certificate.to_dict(), f, indent=2, default=str)

    @staticmethod
    def deserialize(path: str) -> Certificate:
        """Deserialize a certificate from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return Certificate.from_dict(data)

    # ------------------------------------------------------------------
    # Certificate chaining
    # ------------------------------------------------------------------

    @staticmethod
    def chain_certificates(
        certificates: Sequence[Certificate],
        chain_id: Optional[str] = None,
        session_id: str = "",
    ) -> Certificate:
        """
        Merge multiple sub-certificates into a single global certificate.

        Steps are concatenated (re-indexed) and all parent certificates
        are recorded in ``parent_ids``.
        """
        cid = chain_id or uuid.uuid4().hex[:16]
        parent_ids = [c.certificate_id for c in certificates]

        merged_steps: List[CertificateStep] = []
        idx = 0
        for cert in certificates:
            for step in cert.steps:
                new_step = CertificateStep(
                    step_index=idx,
                    assertion=step.assertion,
                    proof_tag=step.proof_tag,
                    timestamp=step.timestamp,
                    metadata={
                        **step.metadata,
                        "source_certificate": cert.certificate_id,
                        "original_index": step.step_index,
                    },
                    digest="",
                )
                merged_steps.append(new_step)
                idx += 1

        now = time.time()
        chained = Certificate(
            certificate_id=cid,
            session_id=session_id,
            steps=merged_steps,
            creation_time=now,
            parent_ids=parent_ids,
        )
        # Finalise the chained certificate
        chained.finalization_time = now
        chained.root_digest = CertificateEmitter._compute_root_digest(
            merged_steps
        )
        chained.finalized = True

        st = chained.stats
        st.total_steps = len(merged_steps)
        st.verified_steps = sum(
            1 for s in merged_steps if "verified" in s.proof_tag
        )
        st.failed_steps = st.total_steps - st.verified_steps
        st.creation_time = now
        st.finalization_time = now
        st.chain_depth = 1 + max(
            (c.stats.chain_depth for c in certificates), default=0
        )
        st.total_digest_bytes = sum(len(s.digest) for s in merged_steps)

        return chained

    # ------------------------------------------------------------------
    # Digest helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_root_digest(steps: List[CertificateStep]) -> str:
        """
        Compute a Merkle-like root hash over step digests.

        Uses a sequential chaining approach: each step hash feeds into the
        next, giving O(n) computation.
        """
        h = hashlib.sha256()
        for step in steps:
            h.update(step.digest.encode())
        return h.hexdigest()[:32]

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    @staticmethod
    def pretty_print(certificate: Certificate) -> str:
        """Return a human-readable summary of a certificate."""
        lines: List[str] = []
        lines.append(f"Certificate: {certificate.certificate_id}")
        lines.append(f"  Session:   {certificate.session_id}")
        lines.append(f"  Finalized: {certificate.finalized}")
        lines.append(f"  Steps:     {len(certificate.steps)}")
        lines.append(f"  Root hash: {certificate.root_digest}")
        if certificate.parent_ids:
            lines.append(f"  Parents:   {certificate.parent_ids}")
        lines.append(f"  Stats:")
        lines.append(f"    Verified: {certificate.stats.verified_steps}")
        lines.append(f"    Failed:   {certificate.stats.failed_steps}")
        lines.append(f"    Chain depth: {certificate.stats.chain_depth}")
        lines.append("")
        for step in certificate.steps[:20]:
            tag = "✓" if "verified" in step.proof_tag else "✗"
            assertion_preview = step.assertion[:60]
            lines.append(
                f"  [{tag}] Step {step.step_index}: {assertion_preview}"
            )
        if len(certificate.steps) > 20:
            lines.append(f"  ... and {len(certificate.steps) - 20} more steps")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def filter_steps(
        certificate: Certificate,
        predicate: str = "",
        metadata_key: str = "",
        metadata_value: Any = None,
    ) -> List[CertificateStep]:
        """
        Return steps matching the given filters.

        Parameters
        ----------
        predicate : str
            Substring filter on the assertion string.
        metadata_key / metadata_value
            Filter by metadata key-value pair.
        """
        result: List[CertificateStep] = []
        for step in certificate.steps:
            if predicate and predicate not in step.assertion:
                continue
            if metadata_key:
                if metadata_key not in step.metadata:
                    continue
                if (
                    metadata_value is not None
                    and step.metadata[metadata_key] != metadata_value
                ):
                    continue
            result.append(step)
        return result
