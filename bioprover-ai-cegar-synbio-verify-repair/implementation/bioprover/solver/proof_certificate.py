"""
Proof certificate generation for validated ODE integration.

Certificates provide independently verifiable evidence that a computation
is correct. They can be serialized, stored, and checked by a separate
validator without re-running the integration.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bioprover.solver.interval import Interval, IntervalVector
from bioprover.soundness import ErrorBudget, SoundnessAnnotation, SoundnessLevel


# Current certificate format version
CERTIFICATE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Base certificate
# ---------------------------------------------------------------------------

@dataclass
class ProofCertificate:
    """
    Base class for proof certificates containing verification evidence.

    A certificate records the computation parameters, results, and
    sufficient information for independent validation.
    """

    certificate_type: str
    version: str = CERTIFICATE_VERSION
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    valid: Optional[bool] = None
    validation_message: str = ""

    def compute_hash(self) -> str:
        """Compute a SHA-256 hash of the certificate content."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "certificate_type": self.certificate_type,
            "version": self.version,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "valid": self.valid,
            "validation_message": self.validation_message,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProofCertificate":
        return cls(
            certificate_type=d["certificate_type"],
            version=d.get("version", CERTIFICATE_VERSION),
            timestamp=d.get("timestamp", 0.0),
            metadata=d.get("metadata", {}),
            valid=d.get("valid"),
            validation_message=d.get("validation_message", ""),
        )

    @classmethod
    def from_json(cls, s: str) -> "ProofCertificate":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Flowpipe certificate
# ---------------------------------------------------------------------------

@dataclass
class FlowpipeCertificate(ProofCertificate):
    """
    Certificate proving trajectory containment in a flowpipe.

    Records the ODE system, initial conditions, integration parameters,
    and the computed flowpipe segments. Validation checks that:
    1. Each segment's enclosure contains the predecessor's enclosure
       propagated forward by one step.
    2. The initial segment contains the initial condition.
    3. Consecutive segments overlap in time.
    """

    certificate_type: str = "flowpipe"
    system_description: str = ""
    dimension: int = 0
    t0: float = 0.0
    tf: float = 0.0
    initial_box_lo: List[float] = field(default_factory=list)
    initial_box_hi: List[float] = field(default_factory=list)
    segments: List[Dict[str, Any]] = field(default_factory=list)
    integration_method: str = ""
    taylor_order: int = 0
    total_steps: int = 0
    max_enclosure_width: float = 0.0

    @classmethod
    def from_integration(
        cls,
        result,
        system_description: str = "",
        initial_condition: Optional[IntervalVector] = None,
    ) -> "FlowpipeCertificate":
        """Build a certificate from an IntegrationResult."""
        cert = cls()
        cert.system_description = system_description
        cert.total_steps = len(result.steps)

        if result.steps:
            first = result.steps[0]
            last = result.steps[-1]
            cert.dimension = first.enclosure.dim
            cert.t0 = float(first.t_interval.lo)
            cert.tf = float(last.t_interval.hi)
            cert.integration_method = first.method_used
            cert.max_enclosure_width = max(s.enclosure_width for s in result.steps)

            if initial_condition is not None:
                cert.initial_box_lo = initial_condition.lo_array().tolist()
                cert.initial_box_hi = initial_condition.hi_array().tolist()

            cert.segments = []
            for step in result.steps:
                cert.segments.append({
                    "time_lo": float(step.t_interval.lo),
                    "time_hi": float(step.t_interval.hi),
                    "box_lo": step.enclosure.lo_array().tolist(),
                    "box_hi": step.enclosure.hi_array().tolist(),
                    "step_size": step.step_size,
                    "width": step.enclosure_width,
                    "method": step.method_used,
                })

        return cert

    def validate(self) -> bool:
        """
        Self-check the certificate for internal consistency.

        Checks:
        1. Segments are non-empty and ordered in time.
        2. Consecutive segments overlap or are contiguous in time.
        3. Initial condition is contained in first segment.
        4. Dimensions are consistent.
        """
        if not self.segments:
            self.valid = False
            self.validation_message = "No segments in certificate"
            return False

        # Check time ordering
        for i in range(len(self.segments) - 1):
            t_hi = self.segments[i]["time_hi"]
            t_lo_next = self.segments[i + 1]["time_lo"]
            if t_lo_next > t_hi + 1e-12:
                self.valid = False
                self.validation_message = (
                    f"Time gap between segments {i} and {i+1}: "
                    f"{t_hi} -> {t_lo_next}"
                )
                return False

        # Check dimension consistency
        for i, seg in enumerate(self.segments):
            if len(seg["box_lo"]) != self.dimension:
                self.valid = False
                self.validation_message = (
                    f"Segment {i} has wrong dimension: "
                    f"{len(seg['box_lo'])} != {self.dimension}"
                )
                return False

        # Check initial condition containment
        if self.initial_box_lo and self.initial_box_hi:
            first_seg = self.segments[0]
            for j in range(self.dimension):
                if self.initial_box_lo[j] < first_seg["box_lo"][j] - 1e-12:
                    self.valid = False
                    self.validation_message = (
                        f"Initial condition not contained in first segment "
                        f"(dim {j}): {self.initial_box_lo[j]} < {first_seg['box_lo'][j]}"
                    )
                    return False
                if self.initial_box_hi[j] > first_seg["box_hi"][j] + 1e-12:
                    self.valid = False
                    self.validation_message = (
                        f"Initial condition not contained in first segment "
                        f"(dim {j}): {self.initial_box_hi[j]} > {first_seg['box_hi'][j]}"
                    )
                    return False

        # Check box validity (lo <= hi)
        for i, seg in enumerate(self.segments):
            for j in range(self.dimension):
                if seg["box_lo"][j] > seg["box_hi"][j] + 1e-12:
                    self.valid = False
                    self.validation_message = (
                        f"Invalid box in segment {i}, dim {j}: "
                        f"{seg['box_lo'][j]} > {seg['box_hi'][j]}"
                    )
                    return False

        self.valid = True
        self.validation_message = "Certificate validated successfully"
        return True

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "system_description": self.system_description,
            "dimension": self.dimension,
            "t0": self.t0,
            "tf": self.tf,
            "initial_box_lo": self.initial_box_lo,
            "initial_box_hi": self.initial_box_hi,
            "segments": self.segments,
            "integration_method": self.integration_method,
            "taylor_order": self.taylor_order,
            "total_steps": self.total_steps,
            "max_enclosure_width": self.max_enclosure_width,
        })
        return base

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlowpipeCertificate":
        cert = cls()
        cert.certificate_type = d.get("certificate_type", "flowpipe")
        cert.version = d.get("version", CERTIFICATE_VERSION)
        cert.timestamp = d.get("timestamp", 0.0)
        cert.metadata = d.get("metadata", {})
        cert.valid = d.get("valid")
        cert.validation_message = d.get("validation_message", "")
        cert.system_description = d.get("system_description", "")
        cert.dimension = d.get("dimension", 0)
        cert.t0 = d.get("t0", 0.0)
        cert.tf = d.get("tf", 0.0)
        cert.initial_box_lo = d.get("initial_box_lo", [])
        cert.initial_box_hi = d.get("initial_box_hi", [])
        cert.segments = d.get("segments", [])
        cert.integration_method = d.get("integration_method", "")
        cert.taylor_order = d.get("taylor_order", 0)
        cert.total_steps = d.get("total_steps", 0)
        cert.max_enclosure_width = d.get("max_enclosure_width", 0.0)
        return cert


# ---------------------------------------------------------------------------
# Invariant certificate
# ---------------------------------------------------------------------------

@dataclass
class InvariantCertificate(ProofCertificate):
    """
    Certificate proving that an invariant holds throughout a trajectory.

    The invariant is a predicate on the state (e.g., x_i >= 0, or
    x_1 + x_2 <= K). The certificate records the flowpipe and shows
    that every segment satisfies the invariant.
    """

    certificate_type: str = "invariant"
    invariant_description: str = ""
    invariant_type: str = ""  # "lower_bound", "upper_bound", "linear", "custom"
    invariant_params: Dict[str, Any] = field(default_factory=dict)
    flowpipe_hash: str = ""
    segment_checks: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def check_lower_bound(
        cls,
        flowpipe_cert: FlowpipeCertificate,
        variable_index: int,
        bound: float,
        description: str = "",
    ) -> "InvariantCertificate":
        """
        Check that x[variable_index] >= bound for all flowpipe segments.
        """
        cert = cls()
        cert.invariant_description = description or f"x[{variable_index}] >= {bound}"
        cert.invariant_type = "lower_bound"
        cert.invariant_params = {"variable_index": variable_index, "bound": bound}
        cert.flowpipe_hash = flowpipe_cert.compute_hash()

        all_ok = True
        checks = []
        for i, seg in enumerate(flowpipe_cert.segments):
            lo = seg["box_lo"][variable_index]
            satisfied = lo >= bound - 1e-12
            checks.append({
                "segment": i,
                "value": lo,
                "satisfied": satisfied,
            })
            if not satisfied:
                all_ok = False

        cert.segment_checks = checks
        cert.valid = all_ok
        cert.validation_message = (
            "Invariant holds" if all_ok
            else f"Invariant violated in {sum(1 for c in checks if not c['satisfied'])} segments"
        )
        return cert

    @classmethod
    def check_upper_bound(
        cls,
        flowpipe_cert: FlowpipeCertificate,
        variable_index: int,
        bound: float,
        description: str = "",
    ) -> "InvariantCertificate":
        """Check that x[variable_index] <= bound for all flowpipe segments."""
        cert = cls()
        cert.invariant_description = description or f"x[{variable_index}] <= {bound}"
        cert.invariant_type = "upper_bound"
        cert.invariant_params = {"variable_index": variable_index, "bound": bound}
        cert.flowpipe_hash = flowpipe_cert.compute_hash()

        all_ok = True
        checks = []
        for i, seg in enumerate(flowpipe_cert.segments):
            hi = seg["box_hi"][variable_index]
            satisfied = hi <= bound + 1e-12
            checks.append({
                "segment": i,
                "value": hi,
                "satisfied": satisfied,
            })
            if not satisfied:
                all_ok = False

        cert.segment_checks = checks
        cert.valid = all_ok
        cert.validation_message = (
            "Invariant holds" if all_ok
            else f"Invariant violated in {sum(1 for c in checks if not c['satisfied'])} segments"
        )
        return cert

    @classmethod
    def check_linear_invariant(
        cls,
        flowpipe_cert: FlowpipeCertificate,
        coefficients: List[float],
        bound: float,
        direction: str = "leq",
        description: str = "",
    ) -> "InvariantCertificate":
        """
        Check linear invariant: coefficients . x <= bound (or >= bound).

        Uses interval arithmetic: evaluate c.x over each box and check bound.
        """
        cert = cls()
        cert.invariant_description = description or (
            f"{''.join(f'{c:+g}*x[{i}]' for i, c in enumerate(coefficients))} "
            f"{'<=' if direction == 'leq' else '>='} {bound}"
        )
        cert.invariant_type = "linear"
        cert.invariant_params = {
            "coefficients": coefficients,
            "bound": bound,
            "direction": direction,
        }
        cert.flowpipe_hash = flowpipe_cert.compute_hash()

        all_ok = True
        checks = []
        for i, seg in enumerate(flowpipe_cert.segments):
            # Compute c.x using interval arithmetic
            lo_arr = seg["box_lo"]
            hi_arr = seg["box_hi"]
            # Evaluate dot product at extreme values
            val_lo = 0.0
            val_hi = 0.0
            for j, c in enumerate(coefficients):
                if c >= 0:
                    val_lo += c * lo_arr[j]
                    val_hi += c * hi_arr[j]
                else:
                    val_lo += c * hi_arr[j]
                    val_hi += c * lo_arr[j]

            if direction == "leq":
                satisfied = val_hi <= bound + 1e-12
                check_val = val_hi
            else:
                satisfied = val_lo >= bound - 1e-12
                check_val = val_lo

            checks.append({
                "segment": i,
                "value": check_val,
                "satisfied": satisfied,
            })
            if not satisfied:
                all_ok = False

        cert.segment_checks = checks
        cert.valid = all_ok
        cert.validation_message = (
            "Invariant holds" if all_ok
            else f"Invariant violated in {sum(1 for c in checks if not c['satisfied'])} segments"
        )
        return cert

    def validate(self) -> bool:
        """Validate the certificate's internal consistency."""
        if not self.segment_checks:
            self.valid = False
            self.validation_message = "No segment checks recorded"
            return False

        all_satisfied = all(c["satisfied"] for c in self.segment_checks)
        self.valid = all_satisfied
        self.validation_message = (
            "Invariant holds" if all_satisfied
            else f"Invariant violated in {sum(1 for c in self.segment_checks if not c['satisfied'])} segments"
        )
        return all_satisfied

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "invariant_description": self.invariant_description,
            "invariant_type": self.invariant_type,
            "invariant_params": self.invariant_params,
            "flowpipe_hash": self.flowpipe_hash,
            "segment_checks": self.segment_checks,
        })
        return base

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InvariantCertificate":
        cert = cls()
        cert.certificate_type = d.get("certificate_type", "invariant")
        cert.version = d.get("version", CERTIFICATE_VERSION)
        cert.timestamp = d.get("timestamp", 0.0)
        cert.metadata = d.get("metadata", {})
        cert.valid = d.get("valid")
        cert.validation_message = d.get("validation_message", "")
        cert.invariant_description = d.get("invariant_description", "")
        cert.invariant_type = d.get("invariant_type", "")
        cert.invariant_params = d.get("invariant_params", {})
        cert.flowpipe_hash = d.get("flowpipe_hash", "")
        cert.segment_checks = d.get("segment_checks", [])
        return cert


# ---------------------------------------------------------------------------
# Certificate composition
# ---------------------------------------------------------------------------

def compose_flowpipe_certificates(
    certs: List[FlowpipeCertificate],
) -> FlowpipeCertificate:
    """
    Compose a sequence of flowpipe certificates into a single certificate
    covering the full time span.

    The certificates must be temporally contiguous (the end time of cert[i]
    must equal the start time of cert[i+1]).
    """
    if not certs:
        raise ValueError("No certificates to compose")

    # Validate temporal contiguity
    for i in range(len(certs) - 1):
        gap = certs[i + 1].t0 - certs[i].tf
        if abs(gap) > 1e-10:
            raise ValueError(
                f"Temporal gap between certificates {i} and {i+1}: "
                f"{certs[i].tf} -> {certs[i+1].t0}"
            )

    composed = FlowpipeCertificate()
    composed.system_description = certs[0].system_description
    composed.dimension = certs[0].dimension
    composed.t0 = certs[0].t0
    composed.tf = certs[-1].tf
    composed.initial_box_lo = certs[0].initial_box_lo
    composed.initial_box_hi = certs[0].initial_box_hi
    composed.integration_method = certs[0].integration_method
    composed.taylor_order = max(c.taylor_order for c in certs)

    all_segments: List[Dict[str, Any]] = []
    for cert in certs:
        all_segments.extend(cert.segments)

    composed.segments = all_segments
    composed.total_steps = len(all_segments)
    composed.max_enclosure_width = max(c.max_enclosure_width for c in certs)
    composed.metadata["composed_from"] = len(certs)
    composed.metadata["component_hashes"] = [c.compute_hash() for c in certs]

    return composed


# ---------------------------------------------------------------------------
# Soundness certificate (wraps proof certificates with soundness metadata)
# ---------------------------------------------------------------------------

@dataclass
class SoundnessCertificate:
    """Wraps a FlowpipeCertificate or InvariantCertificate with soundness metadata.

    Provides a standalone verifiable record of the soundness level,
    error budget, and assumptions under which a proof certificate is valid.
    """

    inner_certificate: ProofCertificate
    soundness_level: SoundnessLevel = SoundnessLevel.SOUND
    error_budget: Optional[ErrorBudget] = None
    assumptions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inner_certificate": self.inner_certificate.to_dict(),
            "soundness_level": self.soundness_level.name,
            "error_budget": self.error_budget.to_dict() if self.error_budget else None,
            "assumptions": self.assumptions,
            "timestamp": self.timestamp,
            "certificate_hash": self.inner_certificate.compute_hash(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SoundnessCertificate":
        cert_type = d["inner_certificate"].get("certificate_type", "")
        if cert_type == "flowpipe":
            inner = FlowpipeCertificate.from_dict(d["inner_certificate"])
        elif cert_type == "invariant":
            inner = InvariantCertificate.from_dict(d["inner_certificate"])
        else:
            inner = ProofCertificate.from_dict(d["inner_certificate"])

        budget = None
        if d.get("error_budget"):
            eb = d["error_budget"]
            budget = ErrorBudget(
                delta=eb.get("delta", 0.0),
                epsilon=eb.get("epsilon", 0.0),
                truncation=eb.get("truncation", 0.0),
                discretization=eb.get("discretization", 0.0),
            )

        return cls(
            inner_certificate=inner,
            soundness_level=SoundnessLevel[d.get("soundness_level", "SOUND")],
            error_budget=budget,
            assumptions=d.get("assumptions", []),
            timestamp=d.get("timestamp", 0.0),
        )

    def validate(self) -> bool:
        """Validate the inner certificate and soundness metadata consistency."""
        if hasattr(self.inner_certificate, "validate"):
            if not self.inner_certificate.validate():
                return False
        if self.error_budget and self.error_budget.combined > 1.0:
            return False
        return True


def validate_certificate(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Standalone certificate validator that can check a certificate independently.

    Accepts a dict (e.g. from JSON) and validates structural integrity and
    soundness metadata without requiring the original solver context.

    Returns (is_valid, message).
    """
    try:
        if "inner_certificate" in data:
            cert = SoundnessCertificate.from_dict(data)
            ok = cert.validate()
            if ok:
                return True, "Soundness certificate validated successfully"
            msg = getattr(cert.inner_certificate, "validation_message", "Validation failed")
            return False, msg

        cert_type = data.get("certificate_type", "")
        if cert_type == "flowpipe":
            cert = FlowpipeCertificate.from_dict(data)
        elif cert_type == "invariant":
            cert = InvariantCertificate.from_dict(data)
        else:
            return False, f"Unknown certificate type: {cert_type!r}"

        ok = cert.validate()
        return ok, cert.validation_message
    except Exception as exc:
        return False, f"Certificate validation error: {exc}"
