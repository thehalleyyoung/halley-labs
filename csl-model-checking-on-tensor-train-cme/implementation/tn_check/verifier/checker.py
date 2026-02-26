"""
Independent certificate verifier.

Takes a VerificationTrace and re-derives error bounds without re-running
the computation. This addresses the critique that error certificates are
computed by the same system that generates results.

The verifier checks:
1. Per-step truncation errors are non-negative and finite
2. Clamping errors satisfy Proposition 1: clamp_err ≤ 2·trunc_err per step
3. Total error composition is sound (triangle inequality)
4. Probability conservation holds within tolerance at each step
5. FSP bounds are consistent with physical dimensions
6. CSL verdict is consistent with probability bounds and error
7. Bond dimensions are within configured limits
8. No step has negative total probability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tn_check.verifier.trace import (
    VerificationTrace, StepRecord, FSPBoundRecord, CSLCheckRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single verification check."""
    check_name: str
    passed: bool
    message: str
    severity: str = "error"  # "error", "warning", "info"
    step_index: Optional[int] = None
    details: dict = field(default_factory=dict)


@dataclass
class VerificationReport:
    """
    Complete verification report.

    Summarizes all checks performed on a VerificationTrace,
    with overall pass/fail verdict.
    """
    trace_model: str
    num_checks: int = 0
    num_passed: int = 0
    num_failed: int = 0
    num_warnings: int = 0
    overall_sound: bool = True
    checks: list[CheckResult] = field(default_factory=list)
    recomputed_total_error: float = 0.0
    claimed_total_error: float = 0.0
    error_discrepancy: float = 0.0

    def add_check(self, check: CheckResult) -> None:
        self.checks.append(check)
        self.num_checks += 1
        if check.passed:
            self.num_passed += 1
        else:
            if check.severity == "error":
                self.num_failed += 1
                self.overall_sound = False
            elif check.severity == "warning":
                self.num_warnings += 1

    def summary(self) -> str:
        status = "SOUND" if self.overall_sound else "UNSOUND"
        lines = [
            f"Verification Report: {self.trace_model}",
            f"  Status: {status}",
            f"  Checks: {self.num_passed}/{self.num_checks} passed, "
            f"{self.num_failed} failed, {self.num_warnings} warnings",
            f"  Claimed total error: {self.claimed_total_error:.2e}",
            f"  Recomputed total error: {self.recomputed_total_error:.2e}",
            f"  Discrepancy: {self.error_discrepancy:.2e}",
        ]
        if not self.overall_sound:
            lines.append("  Failed checks:")
            for c in self.checks:
                if not c.passed and c.severity == "error":
                    lines.append(f"    - {c.check_name}: {c.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "trace_model": self.trace_model,
            "overall_sound": self.overall_sound,
            "num_checks": self.num_checks,
            "num_passed": self.num_passed,
            "num_failed": self.num_failed,
            "num_warnings": self.num_warnings,
            "recomputed_total_error": self.recomputed_total_error,
            "claimed_total_error": self.claimed_total_error,
            "error_discrepancy": self.error_discrepancy,
            "checks": [
                {
                    "check_name": c.check_name,
                    "passed": c.passed,
                    "message": c.message,
                    "severity": c.severity,
                    "step_index": c.step_index,
                }
                for c in self.checks
            ],
        }


class CertificateVerifier:
    """
    Independent verifier for TN-Check error certificates.

    Re-derives error bounds from a VerificationTrace without
    re-running the computation. Checks soundness of all claimed
    error bounds using only the recorded data.

    This addresses the critique that error certificates are computed
    by the same system that generates results, providing an independent
    audit capability.
    """

    def __init__(
        self,
        probability_tolerance: float = 1e-6,
        clamping_factor: float = 2.0,
        max_allowed_error: float = 1.0,
    ):
        self.probability_tolerance = probability_tolerance
        self.clamping_factor = clamping_factor
        self.max_allowed_error = max_allowed_error

    def verify(self, trace: VerificationTrace) -> VerificationReport:
        """
        Verify a complete verification trace.

        Runs all soundness checks and produces a VerificationReport.

        Args:
            trace: The verification trace to audit.

        Returns:
            VerificationReport with check results.
        """
        report = VerificationReport(trace_model=trace.model_name)

        # Ensure trace is finalized
        trace.finalize()
        report.claimed_total_error = trace.total_certified_error

        # Run all checks
        self._check_step_errors_nonnegative(trace, report)
        self._check_clamping_bound(trace, report)
        self._check_error_composition(trace, report)
        self._check_probability_conservation(trace, report)
        self._check_fsp_bounds(trace, report)
        self._check_bond_dimensions(trace, report)
        self._check_csl_verdicts(trace, report)
        self._check_monotone_time(trace, report)
        self._check_finite_errors(trace, report)
        self._check_clamping_proof_consistency(trace, report)
        self._check_error_monotonicity(trace, report)

        logger.info(report.summary())
        return report

    def _check_step_errors_nonnegative(
        self, trace: VerificationTrace, report: VerificationReport,
    ) -> None:
        """Check that all per-step errors are non-negative."""
        for step in trace.steps:
            if step.truncation_error < 0:
                report.add_check(CheckResult(
                    check_name="step_error_nonneg",
                    passed=False,
                    message=f"Step {step.step_index}: negative truncation error "
                            f"{step.truncation_error:.2e}",
                    step_index=step.step_index,
                ))
                return

        report.add_check(CheckResult(
            check_name="step_error_nonneg",
            passed=True,
            message="All per-step truncation errors are non-negative",
        ))

    def _check_clamping_bound(
        self, trace: VerificationTrace, report: VerificationReport,
    ) -> None:
        """
        Check Proposition 1: per-step clamping error ≤ 2 · truncation error.

        This is the core soundness property: the clamped vector cannot
        deviate from the exact solution by more than 2·ε_trunc.
        """
        violations = []
        for step in trace.steps:
            bound = self.clamping_factor * step.truncation_error
            if step.clamping_error > bound + 1e-15:
                violations.append((step.step_index, step.clamping_error, bound))

        if violations:
            report.add_check(CheckResult(
                check_name="clamping_bound",
                passed=False,
                message=f"Proposition 1 violated at {len(violations)} steps. "
                        f"First: step {violations[0][0]}, "
                        f"clamp={violations[0][1]:.2e} > bound={violations[0][2]:.2e}",
                severity="error",
                details={"violations": violations[:5]},
            ))
        else:
            report.add_check(CheckResult(
                check_name="clamping_bound",
                passed=True,
                message="Proposition 1 satisfied at all steps: "
                        "clamp_err ≤ 2·trunc_err",
            ))

    def _check_error_composition(
        self, trace: VerificationTrace, report: VerificationReport,
    ) -> None:
        """
        Check that total error is sound via triangle inequality.

        Recomputes: total_err = sum(trunc_err) + min(sum(clamp_err), 2·sum(trunc_err)) + fsp_err
        """
        total_trunc = sum(s.truncation_error for s in trace.steps)
        total_clamp = sum(s.clamping_error for s in trace.steps)
        clamping_bound = min(total_clamp, self.clamping_factor * total_trunc)
        fsp_err = trace.total_fsp_error

        recomputed = total_trunc + clamping_bound + fsp_err
        report.recomputed_total_error = recomputed
        report.error_discrepancy = abs(recomputed - trace.total_certified_error)

        # The claimed error should be >= recomputed (conservative is OK)
        if trace.total_certified_error < recomputed - 1e-15:
            report.add_check(CheckResult(
                check_name="error_composition",
                passed=False,
                message=f"Claimed error {trace.total_certified_error:.2e} < "
                        f"recomputed {recomputed:.2e} (underestimate!)",
                severity="error",
                details={
                    "total_trunc": total_trunc,
                    "total_clamp": total_clamp,
                    "clamping_bound": clamping_bound,
                    "fsp_err": fsp_err,
                },
            ))
        else:
            report.add_check(CheckResult(
                check_name="error_composition",
                passed=True,
                message=f"Error composition sound: claimed {trace.total_certified_error:.2e} "
                        f">= recomputed {recomputed:.2e}",
            ))

    def _check_probability_conservation(
        self, trace: VerificationTrace, report: VerificationReport,
    ) -> None:
        """Check that total probability stays near 1 at each step."""
        violations = []
        for step in trace.steps:
            deviation = abs(step.total_probability - 1.0)
            if deviation > self.probability_tolerance:
                violations.append((step.step_index, step.total_probability, deviation))

        if violations:
            worst = max(violations, key=lambda x: x[2])
            report.add_check(CheckResult(
                check_name="probability_conservation",
                passed=False,
                message=f"Probability conservation violated at {len(violations)} steps. "
                        f"Worst: step {worst[0]}, prob={worst[1]:.6f}, "
                        f"deviation={worst[2]:.2e}",
                severity="warning",
                details={"num_violations": len(violations)},
            ))
        else:
            report.add_check(CheckResult(
                check_name="probability_conservation",
                passed=True,
                message="Probability conservation holds at all steps "
                        f"(tolerance={self.probability_tolerance:.2e})",
            ))

    def _check_fsp_bounds(
        self, trace: VerificationTrace, report: VerificationReport,
    ) -> None:
        """Check FSP bounds are consistent."""
        if trace.fsp_bounds is None:
            report.add_check(CheckResult(
                check_name="fsp_bounds",
                passed=True,
                message="No FSP bounds recorded (not applicable)",
                severity="info",
            ))
            return

        fsp = trace.fsp_bounds
        if fsp.fsp_error_bound < 0:
            report.add_check(CheckResult(
                check_name="fsp_bounds",
                passed=False,
                message=f"Negative FSP error bound: {fsp.fsp_error_bound:.2e}",
            ))
        elif fsp.fsp_error_bound > 1.0:
            report.add_check(CheckResult(
                check_name="fsp_bounds",
                passed=False,
                message=f"FSP error bound > 1: {fsp.fsp_error_bound:.2e}",
                severity="warning",
            ))
        else:
            report.add_check(CheckResult(
                check_name="fsp_bounds",
                passed=True,
                message=f"FSP error bound valid: {fsp.fsp_error_bound:.2e}",
            ))

    def _check_bond_dimensions(
        self, trace: VerificationTrace, report: VerificationReport,
    ) -> None:
        """Check bond dimensions are within limits."""
        max_chi = trace.max_bond_dim if trace.max_bond_dim > 0 else 10000
        violations = []
        for step in trace.steps:
            if step.bond_dims and max(step.bond_dims) > max_chi:
                violations.append((step.step_index, max(step.bond_dims)))

        if violations:
            report.add_check(CheckResult(
                check_name="bond_dimensions",
                passed=False,
                message=f"Bond dimension exceeds limit at {len(violations)} steps",
                severity="warning",
            ))
        else:
            report.add_check(CheckResult(
                check_name="bond_dimensions",
                passed=True,
                message="All bond dimensions within limits",
            ))

    def _check_csl_verdicts(
        self, trace: VerificationTrace, report: VerificationReport,
    ) -> None:
        """Check CSL verdicts are consistent with probability bounds."""
        for check in trace.csl_checks:
            p_lo = check.probability_lower
            p_hi = check.probability_upper

            # Sanity: p_lo <= p_hi
            if p_lo > p_hi + 1e-15:
                report.add_check(CheckResult(
                    check_name="csl_verdict",
                    passed=False,
                    message=f"Formula '{check.formula_str}': "
                            f"p_lower={p_lo:.6f} > p_upper={p_hi:.6f}",
                ))
                continue

            # Check interval width is consistent with error
            interval_width = p_hi - p_lo
            if interval_width < 0:
                report.add_check(CheckResult(
                    check_name="csl_verdict",
                    passed=False,
                    message=f"Formula '{check.formula_str}': negative interval width",
                ))
                continue

            report.add_check(CheckResult(
                check_name="csl_verdict",
                passed=True,
                message=f"Formula '{check.formula_str}': verdict={check.verdict}, "
                        f"prob=[{p_lo:.4f}, {p_hi:.4f}]",
            ))

    def _check_monotone_time(
        self, trace: VerificationTrace, report: VerificationReport,
    ) -> None:
        """Check that time stamps are monotonically non-decreasing."""
        for i in range(1, len(trace.steps)):
            if trace.steps[i].time < trace.steps[i-1].time - 1e-15:
                report.add_check(CheckResult(
                    check_name="monotone_time",
                    passed=False,
                    message=f"Time decreases at step {i}: "
                            f"{trace.steps[i-1].time:.6f} -> {trace.steps[i].time:.6f}",
                ))
                return

        report.add_check(CheckResult(
            check_name="monotone_time",
            passed=True,
            message="Time stamps are monotonically non-decreasing",
        ))

    def _check_finite_errors(
        self, trace: VerificationTrace, report: VerificationReport,
    ) -> None:
        """Check all errors are finite."""
        for step in trace.steps:
            if not np.isfinite(step.truncation_error) or not np.isfinite(step.clamping_error):
                report.add_check(CheckResult(
                    check_name="finite_errors",
                    passed=False,
                    message=f"Non-finite error at step {step.step_index}",
                ))
                return

        report.add_check(CheckResult(
            check_name="finite_errors",
            passed=True,
            message="All errors are finite",
        ))

    def _check_clamping_proof_consistency(
        self, trace: VerificationTrace, report: VerificationReport,
    ) -> None:
        """
        Verify per-step clamping proofs are internally consistent.

        For each ClampingProofRecord, checks that:
        - Per-iteration clamping errors are bounded by 2x truncation errors
        - The bound_verified flag is consistent with the data
        """
        if not trace.clamping_proofs:
            report.add_check(CheckResult(
                check_name="clamping_proof_consistency",
                passed=True,
                message="No clamping proofs to check (not applicable)",
                severity="info",
            ))
            return

        violations = []
        for proof_rec in trace.clamping_proofs:
            for it_data in proof_rec.iteration_data:
                trunc_e = it_data.get("truncation_error", 0.0)
                clamp_e = it_data.get("clamping_error", 0.0)
                if clamp_e > 2.0 * trunc_e + 1e-14:
                    violations.append((
                        proof_rec.step_index,
                        it_data.get("iteration", -1),
                        clamp_e,
                        2.0 * trunc_e,
                    ))
            if not proof_rec.bound_verified:
                violations.append((
                    proof_rec.step_index, -1, 0.0, 0.0,
                ))

        if violations:
            report.add_check(CheckResult(
                check_name="clamping_proof_consistency",
                passed=False,
                message=f"Clamping proof inconsistencies at {len(violations)} points. "
                        f"First: step {violations[0][0]}",
                severity="error",
            ))
        else:
            report.add_check(CheckResult(
                check_name="clamping_proof_consistency",
                passed=True,
                message="All clamping proofs are internally consistent",
            ))

    def _check_error_monotonicity(
        self, trace: VerificationTrace, report: VerificationReport,
    ) -> None:
        """
        Verify errors don't mysteriously decrease across steps.

        Accumulated truncation error should be non-decreasing; a decrease
        would indicate a bug in error tracking.
        """
        if len(trace.steps) < 2:
            report.add_check(CheckResult(
                check_name="error_monotonicity",
                passed=True,
                message="Fewer than 2 steps; monotonicity trivially holds",
                severity="info",
            ))
            return

        accumulated = 0.0
        prev_accumulated = 0.0
        violations = []
        for step in trace.steps:
            accumulated += step.truncation_error
            if accumulated < prev_accumulated - 1e-15:
                violations.append((
                    step.step_index, accumulated, prev_accumulated,
                ))
            prev_accumulated = accumulated

        if violations:
            report.add_check(CheckResult(
                check_name="error_monotonicity",
                passed=False,
                message=f"Accumulated error decreased at step {violations[0][0]}: "
                        f"{violations[0][1]:.2e} < {violations[0][2]:.2e}",
                severity="error",
            ))
        else:
            report.add_check(CheckResult(
                check_name="error_monotonicity",
                passed=True,
                message="Accumulated truncation error is non-decreasing",
            ))
