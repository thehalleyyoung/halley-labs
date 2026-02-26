"""
Verification report generation for CoaCert-TLA witnesses.

Collects results from hash verification, bisimulation closure validation,
stuttering equivalence, and fairness preservation into a unified report.
Supports human-readable text, machine-readable JSON, and timing breakdown.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence

from .closure_validator import ClosureResult, ClosureViolation, ViolationKind
from .deserializer import WitnessData, WitnessHeader
from .fairness_verifier import FairnessResult, FairnessViolation, FairnessViolationKind
from .hash_verifier import HashFailure, HashVerificationResult, FailureKind
from .stuttering_verifier import (
    StutteringResult,
    StutteringViolation,
    StutteringViolationKind,
)

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


class Verdict(Enum):
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"


class TrustLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


# ---------------------------------------------------------------------------
# Report phase summary
# ---------------------------------------------------------------------------


@dataclass
class PhaseSummary:
    """Summary of one verification phase."""
    name: str
    passed: bool
    checks_performed: int = 0
    failures_count: int = 0
    elapsed_seconds: float = 0.0
    details: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Verification report
# ---------------------------------------------------------------------------


class VerificationReport:
    """Collects and formats verification results from all phases."""

    def __init__(self, witness: WitnessData):
        self._witness = witness
        self._hash_result: Optional[HashVerificationResult] = None
        self._closure_result: Optional[ClosureResult] = None
        self._stuttering_result: Optional[StutteringResult] = None
        self._fairness_result: Optional[FairnessResult] = None
        self._phases: List[PhaseSummary] = []
        self._verdict: Verdict = Verdict.ERROR
        self._trust_level: TrustLevel = TrustLevel.NONE
        self._total_elapsed: float = 0.0
        self._finalized = False
        self._timestamp = datetime.now(timezone.utc)

    @property
    def verdict(self) -> Verdict:
        return self._verdict

    @property
    def trust_level(self) -> TrustLevel:
        return self._trust_level

    @property
    def phases(self) -> Sequence[PhaseSummary]:
        return list(self._phases)

    # -- adding results -----------------------------------------------------

    def add_hash_result(self, result: HashVerificationResult) -> None:
        self._hash_result = result
        phase = PhaseSummary(
            name="Hash Chain Verification",
            passed=result.passed,
            checks_performed=result.blocks_checked + result.merkle_proofs_checked,
            failures_count=len(result.failures),
            elapsed_seconds=result.elapsed_seconds,
        )
        phase.details.append(
            f"Blocks checked: {result.blocks_checked}/{result.blocks_total}"
        )
        phase.details.append(
            f"Merkle proofs: {result.merkle_proofs_passed}/"
            f"{result.merkle_proofs_checked} passed"
        )
        for f in result.failures[:5]:
            phase.details.append(f"  FAIL: {f.message}")
        if len(result.failures) > 5:
            phase.details.append(
                f"  ... and {len(result.failures) - 5} more failures"
            )
        self._phases.append(phase)

    def add_closure_result(self, result: ClosureResult) -> None:
        self._closure_result = result
        total_checks = (
            result.forward_checked + result.backward_checked +
            result.ap_checked + result.stutter_chains_checked
        )
        phase = PhaseSummary(
            name="Bisimulation Closure",
            passed=result.passed,
            checks_performed=total_checks,
            failures_count=len(result.violations),
            elapsed_seconds=result.elapsed_seconds,
        )
        phase.details.append(f"Mode: {result.mode}")
        phase.details.append(f"Forward checks: {result.forward_checked}")
        phase.details.append(f"Backward checks: {result.backward_checked}")
        phase.details.append(f"AP checks: {result.ap_checked}")
        if result.stutter_chains_checked > 0:
            phase.details.append(
                f"Stutter chains: {result.stutter_chains_checked}"
            )
        for v in result.violations[:5]:
            phase.details.append(f"  FAIL [{v.kind.name}]: {v.message}")
        if len(result.violations) > 5:
            phase.details.append(
                f"  ... and {len(result.violations) - 5} more violations"
            )
        self._phases.append(phase)

    def add_stuttering_result(self, result: StutteringResult) -> None:
        self._stuttering_result = result
        total_checks = (
            result.paths_checked + result.divergence_checks +
            result.trace_checks
        )
        phase = PhaseSummary(
            name="Stuttering Equivalence",
            passed=result.passed,
            checks_performed=total_checks,
            failures_count=len(result.violations),
            elapsed_seconds=result.elapsed_seconds,
        )
        phase.details.append(f"Paths checked: {result.paths_checked}")
        phase.details.append(f"Divergence checks: {result.divergence_checks}")
        phase.details.append(f"Trace checks: {result.trace_checks}")
        if result.hash_checks_total > 0:
            phase.details.append(
                f"Inline hash: {result.hash_checks_passed}/"
                f"{result.hash_checks_total} passed"
            )
        for v in result.violations[:5]:
            phase.details.append(f"  FAIL [{v.kind.name}]: {v.message}")
        if len(result.violations) > 5:
            phase.details.append(
                f"  ... and {len(result.violations) - 5} more violations"
            )
        self._phases.append(phase)

    def add_fairness_result(self, result: FairnessResult) -> None:
        self._fairness_result = result
        total_checks = (
            result.b_set_checks + result.g_set_checks +
            result.tfair_checks + result.cycle_checks
        )
        phase = PhaseSummary(
            name="Fairness Preservation",
            passed=result.passed,
            checks_performed=total_checks,
            failures_count=len(result.violations),
            elapsed_seconds=result.elapsed_seconds,
        )
        phase.details.append(f"Pairs checked: {result.pairs_checked}")
        phase.details.append(f"B-set checks: {result.b_set_checks}")
        phase.details.append(f"G-set checks: {result.g_set_checks}")
        phase.details.append(f"T-Fair checks: {result.tfair_checks}")
        phase.details.append(f"Cycle checks: {result.cycle_checks}")
        for v in result.violations[:5]:
            phase.details.append(f"  FAIL [{v.kind.name}]: {v.message}")
        if len(result.violations) > 5:
            phase.details.append(
                f"  ... and {len(result.violations) - 5} more violations"
            )
        self._phases.append(phase)

    # -- finalization -------------------------------------------------------

    def finalize(self) -> None:
        """Compute overall verdict and trust level."""
        self._total_elapsed = sum(p.elapsed_seconds for p in self._phases)

        all_passed = all(p.passed for p in self._phases)
        any_failed = any(not p.passed for p in self._phases)

        if not self._phases:
            self._verdict = Verdict.ERROR
            self._trust_level = TrustLevel.NONE
        elif all_passed:
            self._verdict = Verdict.VERIFIED
            self._trust_level = self._compute_trust_level()
        elif any_failed:
            self._verdict = Verdict.REJECTED
            self._trust_level = TrustLevel.NONE
        else:
            self._verdict = Verdict.PARTIAL
            self._trust_level = TrustLevel.LOW

        self._finalized = True

    def _compute_trust_level(self) -> TrustLevel:
        """Compute trust level based on verification thoroughness."""
        score = 0
        max_score = 0

        # Hash verification
        if self._hash_result is not None:
            max_score += 30
            if self._hash_result.passed:
                coverage = (
                    self._hash_result.blocks_checked /
                    max(1, self._hash_result.blocks_total)
                )
                score += int(30 * coverage)

        # Closure validation
        if self._closure_result is not None:
            max_score += 30
            if self._closure_result.passed:
                if self._closure_result.mode == "full":
                    score += 30
                else:
                    score += 15

        # Stuttering
        if self._stuttering_result is not None:
            max_score += 20
            if self._stuttering_result.passed:
                score += 20

        # Fairness
        if self._fairness_result is not None:
            max_score += 20
            if self._fairness_result.passed:
                score += 20

        if max_score == 0:
            return TrustLevel.NONE

        ratio = score / max_score
        if ratio >= 0.9:
            return TrustLevel.HIGH
        elif ratio >= 0.6:
            return TrustLevel.MEDIUM
        else:
            return TrustLevel.LOW

    # -- text report --------------------------------------------------------

    def to_text(self) -> str:
        """Generate a human-readable report."""
        lines: List[str] = []
        w = self._witness
        hdr = w.header

        lines.append("=" * 72)
        lines.append("  CoaCert-TLA Witness Verification Report")
        lines.append("=" * 72)
        lines.append("")

        # Header / metadata
        lines.append("WITNESS METADATA")
        lines.append(f"  Source:    {w.source_path or '(in-memory)'}")
        lines.append(f"  Version:  {hdr.version_major}.{hdr.version_minor}")
        lines.append(f"  Flags:    0x{hdr.flags:04X}")
        lines.append(f"  Sections: {hdr.num_sections}")
        lines.append(f"  Classes:  {len(w.equivalences)}")
        lines.append(f"  Trans:    {len(w.transitions)}")
        lines.append(f"  Fairness: {len(w.fairness)} pairs")
        lines.append(f"  Chain:    {len(w.hash_chain)} blocks")
        if w.metadata:
            spec = w.metadata.get("spec_name", "unknown")
            lines.append(f"  Spec:     {spec}")
        lines.append("")

        # Phase results
        lines.append("-" * 72)
        lines.append("VERIFICATION RESULTS")
        lines.append("-" * 72)
        lines.append("")

        for phase in self._phases:
            status = "PASS" if phase.passed else "FAIL"
            lines.append(f"[{status}] {phase.name}")
            lines.append(
                f"       Checks: {phase.checks_performed}  "
                f"Failures: {phase.failures_count}  "
                f"Time: {phase.elapsed_seconds:.4f}s"
            )
            for detail in phase.details:
                lines.append(f"       {detail}")
            lines.append("")

        # Overall verdict
        lines.append("-" * 72)
        lines.append(f"VERDICT:     {self._verdict.value}")
        lines.append(f"TRUST LEVEL: {self._trust_level.value}")
        lines.append(f"TOTAL TIME:  {self._total_elapsed:.4f}s")
        lines.append(f"TIMESTAMP:   {self._timestamp.isoformat()}")
        lines.append("")

        # Recommendation
        lines.append("RECOMMENDATION")
        lines.append(f"  {self._recommendation()}")
        lines.append("=" * 72)

        return "\n".join(lines)

    def _recommendation(self) -> str:
        if self._verdict == Verdict.VERIFIED:
            if self._trust_level == TrustLevel.HIGH:
                return (
                    "The witness has been fully verified.  The quotient "
                    "system is a sound abstraction of the original."
                )
            elif self._trust_level == TrustLevel.MEDIUM:
                return (
                    "The witness passed statistical checks.  Consider "
                    "running full verification for higher assurance."
                )
            else:
                return (
                    "The witness passed some checks but coverage is low.  "
                    "Re-run with --full for complete verification."
                )
        elif self._verdict == Verdict.REJECTED:
            first_failures: List[str] = []
            for p in self._phases:
                if not p.passed and p.details:
                    for d in p.details:
                        if d.strip().startswith("FAIL"):
                            first_failures.append(d.strip())
                            break
            if first_failures:
                return (
                    f"The witness is INVALID.  First failure: "
                    f"{first_failures[0]}"
                )
            return "The witness is INVALID.  See details above."
        elif self._verdict == Verdict.PARTIAL:
            return (
                "Verification is incomplete.  Some phases were skipped "
                "or encountered errors."
            )
        else:
            return "Verification could not be performed.  Check input."

    # -- JSON report --------------------------------------------------------

    def to_json(self, *, indent: int = 2) -> str:
        """Generate a machine-readable JSON report."""
        return json.dumps(self._to_dict(), indent=indent, default=str)

    def _to_dict(self) -> Dict[str, Any]:
        hdr = self._witness.header
        return {
            "verdict": self._verdict.value,
            "trust_level": self._trust_level.value,
            "timestamp": self._timestamp.isoformat(),
            "total_elapsed_seconds": self._total_elapsed,
            "witness": {
                "source_path": self._witness.source_path,
                "version": f"{hdr.version_major}.{hdr.version_minor}",
                "flags": hdr.flags,
                "num_equivalences": len(self._witness.equivalences),
                "num_transitions": len(self._witness.transitions),
                "num_fairness_pairs": len(self._witness.fairness),
                "num_hash_blocks": len(self._witness.hash_chain),
                "metadata": self._witness.metadata,
            },
            "phases": [self._phase_to_dict(p) for p in self._phases],
            "timing_breakdown": self._timing_breakdown(),
            "recommendation": self._recommendation(),
        }

    def _phase_to_dict(self, phase: PhaseSummary) -> Dict[str, Any]:
        return {
            "name": phase.name,
            "passed": phase.passed,
            "checks_performed": phase.checks_performed,
            "failures_count": phase.failures_count,
            "elapsed_seconds": phase.elapsed_seconds,
            "details": phase.details,
        }

    def _timing_breakdown(self) -> Dict[str, Any]:
        breakdown: Dict[str, Any] = {
            "total": self._total_elapsed,
        }
        if self._hash_result:
            breakdown["hash_chain"] = {
                "total": self._hash_result.elapsed_seconds,
                **self._hash_result.timing_breakdown,
            }
        if self._closure_result:
            breakdown["closure"] = {
                "total": self._closure_result.elapsed_seconds,
                **self._closure_result.timing_breakdown,
            }
        if self._stuttering_result:
            breakdown["stuttering"] = {
                "total": self._stuttering_result.elapsed_seconds,
                **self._stuttering_result.timing_breakdown,
            }
        if self._fairness_result:
            breakdown["fairness"] = {
                "total": self._fairness_result.elapsed_seconds,
                **self._fairness_result.timing_breakdown,
            }
        return breakdown

    # -- utilities ----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VerificationReport(verdict={self._verdict.value}, "
            f"trust={self._trust_level.value}, "
            f"phases={len(self._phases)})"
        )
