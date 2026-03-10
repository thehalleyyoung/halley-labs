#!/usr/bin/env python3
"""
Example: CI/CD integration for usability regression testing.

Demonstrates how to integrate the usability oracle into a continuous
integration pipeline:

1. Define a regression test suite with pass/fail thresholds.
2. Run the oracle on before/after tree pairs.
3. Produce machine-readable (JSON) results for CI consumption.
4. Exit with appropriate status codes (0 = pass, 1 = regression found).

Usage in CI:
    python examples/ci_integration.py --threshold 0.3 --format json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.core.enums import (
    BottleneckType,
    RegressionVerdict,
)
from usability_oracle.benchmarks.generators import SyntheticUIGenerator
from usability_oracle.benchmarks.mutations import MutationOperator
from usability_oracle.evaluation.baselines import BaselineComparator
from usability_oracle.benchmarks.metrics import BenchmarkMetrics


# ---------------------------------------------------------------------------
# CI Test Case
# ---------------------------------------------------------------------------

@dataclass
class CITestCase:
    """A single CI regression test case."""

    name: str
    description: str
    before: AccessibilityTree
    after: AccessibilityTree
    severity_threshold: float = 0.3
    expected_verdict: RegressionVerdict | None = None


@dataclass
class CITestResult:
    """Result of a single CI test case."""

    name: str
    verdict: RegressionVerdict
    severity: float
    passed: bool
    duration_s: float
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CI Test Runner
# ---------------------------------------------------------------------------

class CITestRunner:
    """Run regression tests and produce CI-compatible output."""

    def __init__(
        self,
        severity_threshold: float = 0.3,
        seed: int = 42,
    ) -> None:
        self.severity_threshold = severity_threshold
        self.comparator = BaselineComparator(seed=seed)
        self.results: list[CITestResult] = []

    def add_case(self, case: CITestCase) -> None:
        """Register a test case."""
        self._cases = getattr(self, "_cases", [])
        self._cases.append(case)

    def run(self) -> list[CITestResult]:
        """Execute all registered test cases."""
        self.results = []
        cases = getattr(self, "_cases", [])

        for case in cases:
            t0 = time.time()
            verdict_map = self.comparator.run_all(case.before, case.after)
            elapsed = time.time() - t0

            # Compute aggregate severity
            n_regression = sum(
                1 for v in verdict_map.values()
                if v == RegressionVerdict.REGRESSION
            )
            n_total = len(verdict_map)
            severity = n_regression / n_total if n_total > 0 else 0.0

            # Determine overall verdict
            if severity >= case.severity_threshold:
                verdict = RegressionVerdict.REGRESSION
            elif n_regression == 0:
                verdict = RegressionVerdict.NEUTRAL
            else:
                verdict = RegressionVerdict.NEUTRAL  # below threshold

            # Check pass/fail
            passed = verdict != RegressionVerdict.REGRESSION
            if case.expected_verdict is not None:
                passed = verdict == case.expected_verdict

            result = CITestResult(
                name=case.name,
                verdict=verdict,
                severity=severity,
                passed=passed,
                duration_s=elapsed,
                details={
                    "baseline_verdicts": {k: v.value for k, v in verdict_map.items()},
                    "n_regression": n_regression,
                    "n_total": n_total,
                },
            )
            self.results.append(result)

        return self.results

    def summary(self) -> dict[str, Any]:
        """Produce a summary of all test results."""
        n_total = len(self.results)
        n_passed = sum(1 for r in self.results if r.passed)
        n_failed = n_total - n_passed
        total_time = sum(r.duration_s for r in self.results)

        return {
            "total": n_total,
            "passed": n_passed,
            "failed": n_failed,
            "pass_rate": n_passed / n_total if n_total > 0 else 0.0,
            "total_duration_s": total_time,
            "results": [
                {
                    "name": r.name,
                    "verdict": r.verdict.value,
                    "severity": r.severity,
                    "passed": r.passed,
                    "duration_s": r.duration_s,
                    "details": r.details,
                }
                for r in self.results
            ],
        }

    def exit_code(self) -> int:
        """Return 0 if all tests passed, 1 otherwise."""
        return 0 if all(r.passed for r in self.results) else 1


# ---------------------------------------------------------------------------
# Build test cases
# ---------------------------------------------------------------------------

def build_test_cases(threshold: float) -> list[CITestCase]:
    """Build a suite of regression test cases."""
    gen = SyntheticUIGenerator(seed=42)
    mutator = MutationOperator(seed=42)

    cases: list[CITestCase] = []

    # Case 1: No regression (identical trees)
    form = gen.generate_form(n_fields=5)
    cases.append(CITestCase(
        name="no_regression_identical",
        description="Same tree should produce no regression",
        before=form,
        after=form,
        severity_threshold=threshold,
        expected_verdict=RegressionVerdict.NEUTRAL,
    ))

    # Case 2: Mild mutation (should pass with default threshold)
    mild = mutator.apply_label_removal(form, fraction=0.1)
    cases.append(CITestCase(
        name="mild_label_removal",
        description="10% label removal should be below threshold",
        before=form,
        after=mild,
        severity_threshold=threshold,
    ))

    # Case 3: Severe mutation (should fail)
    severe = mutator.apply_perceptual_overload(form, severity=0.9)
    severe = mutator.apply_label_removal(severe, fraction=0.8)
    cases.append(CITestCase(
        name="severe_regression",
        description="90% overload + 80% label removal should trigger regression",
        before=form,
        after=severe,
        severity_threshold=threshold,
        expected_verdict=RegressionVerdict.REGRESSION,
    ))

    # Case 4: Navigation menu
    nav = gen.generate_navigation(n_items=8, depth=2)
    nav_mutated = mutator.apply_choice_paralysis(nav, severity=0.6)
    cases.append(CITestCase(
        name="navigation_choice_paralysis",
        description="Choice paralysis in navigation",
        before=nav,
        after=nav_mutated,
        severity_threshold=threshold,
    ))

    # Case 5: Motor difficulty on form
    motor = mutator.apply_motor_difficulty(form, severity=0.7)
    cases.append(CITestCase(
        name="form_motor_difficulty",
        description="Target size reduction on form",
        before=form,
        after=motor,
        severity_threshold=threshold,
    ))

    # Case 6: Tab order scrambling
    tab_scrambled = mutator.apply_tab_order_scramble(form, severity=0.8)
    cases.append(CITestCase(
        name="tab_order_scramble",
        description="Tab order disruption",
        before=form,
        after=tab_scrambled,
        severity_threshold=threshold,
    ))

    return cases


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def format_text(summary: dict[str, Any]) -> str:
    """Format results as human-readable text."""
    lines = [
        "=" * 60,
        "USABILITY REGRESSION TEST RESULTS",
        "=" * 60,
        "",
        f"Total:   {summary['total']}",
        f"Passed:  {summary['passed']}",
        f"Failed:  {summary['failed']}",
        f"Rate:    {summary['pass_rate']:.0%}",
        f"Time:    {summary['total_duration_s']:.3f}s",
        "",
        "-" * 60,
    ]

    for r in summary["results"]:
        icon = "✅" if r["passed"] else "❌"
        lines.append(
            f"  {icon} {r['name']:40s} {r['verdict']:12s} "
            f"severity={r['severity']:.3f}  ({r['duration_s']:.3f}s)"
        )

    lines.append("")
    lines.append("-" * 60)
    status = "PASS" if summary["failed"] == 0 else "FAIL"
    lines.append(f"  Overall: {status}")

    return "\n".join(lines)


def format_json(summary: dict[str, Any]) -> str:
    """Format results as machine-readable JSON."""
    return json.dumps(summary, indent=2)


def format_junit(summary: dict[str, Any]) -> str:
    """Format results as JUnit XML for CI systems."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<testsuite name="usability_regression" tests="{summary["total"]}" '
        f'failures="{summary["failed"]}" time="{summary["total_duration_s"]:.3f}">',
    ]
    for r in summary["results"]:
        lines.append(f'  <testcase name="{r["name"]}" time="{r["duration_s"]:.3f}">')
        if not r["passed"]:
            lines.append(
                f'    <failure message="Usability regression detected">'
                f'verdict={r["verdict"]}, severity={r["severity"]:.3f}'
                f'</failure>'
            )
        lines.append("  </testcase>")
    lines.append("</testsuite>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run usability regression tests (CI mode)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Severity threshold for regression detection (default: 0.3)",
    )
    parser.add_argument(
        "--format", choices=["text", "json", "junit"], default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: stdout)",
    )
    args = parser.parse_args()

    # Build and run
    cases = build_test_cases(args.threshold)
    runner = CITestRunner(severity_threshold=args.threshold)
    for case in cases:
        runner.add_case(case)

    runner.run()
    summary = runner.summary()

    # Format output
    formatters = {
        "text": format_text,
        "json": format_json,
        "junit": format_junit,
    }
    output = formatters[args.format](summary)

    # Write output
    if args.output:
        Path(args.output).write_text(output)
        print(f"Results written to {args.output}")
    else:
        print(output)

    # Exit with appropriate code
    sys.exit(runner.exit_code())


if __name__ == "__main__":
    main()
