"""Integration tests for the WCAG evaluation pipeline.

End-to-end: WCAG violation detection → cognitive cost mapping → reporting,
including SARIF output of WCAG results.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pytest

from usability_oracle.wcag.types import (
    ConformanceLevel,
    ImpactLevel,
    SuccessCriterion,
    WCAGGuideline,
    WCAGPrinciple,
    WCAGResult,
    WCAGViolation,
)
from usability_oracle.wcag.contrast import (
    Color,
    ContrastResult,
    check_contrast,
    contrast_ratio,
    relative_luminance,
)
from usability_oracle.wcag.mapping import (
    CognitiveCostDelta,
    compute_cost_delta,
    compute_violation_cost,
    to_cost_element,
    wcag_cost_summary,
)
from usability_oracle.wcag.reporter import (
    WCAGConformanceReporter,
    ReportSummary,
    compute_summary,
    rank_remediations,
)
from usability_oracle.sarif.converter import (
    wcag_violation_to_sarif,
    wcag_result_to_sarif,
    wcag_rule_id,
)
from usability_oracle.sarif.writer import SarifBuilder, sarif_to_string
from usability_oracle.sarif.schema import Level


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sc(sc_id: str, name: str, level: ConformanceLevel,
        principle: WCAGPrinciple, gl_id: str) -> SuccessCriterion:
    return SuccessCriterion(
        sc_id=sc_id, name=name, level=level, principle=principle,
        guideline_id=gl_id,
    )


SC_1_1_1 = _sc("1.1.1", "Non-text Content", ConformanceLevel.A,
                WCAGPrinciple.PERCEIVABLE, "1.1")
SC_1_4_3 = _sc("1.4.3", "Contrast (Minimum)", ConformanceLevel.AA,
                WCAGPrinciple.PERCEIVABLE, "1.4")
SC_2_1_1 = _sc("2.1.1", "Keyboard", ConformanceLevel.A,
                WCAGPrinciple.OPERABLE, "2.1")
SC_2_4_1 = _sc("2.4.1", "Bypass Blocks", ConformanceLevel.A,
                WCAGPrinciple.OPERABLE, "2.4")
SC_1_4_6 = _sc("1.4.6", "Contrast (Enhanced)", ConformanceLevel.AAA,
                WCAGPrinciple.PERCEIVABLE, "1.4")


def _make_violations() -> tuple[WCAGViolation, ...]:
    return (
        WCAGViolation(
            criterion=SC_1_1_1,
            node_id="img_1",
            impact=ImpactLevel.CRITICAL,
            message="Image missing alt text",
            evidence={"element": "img", "src": "photo.jpg"},
        ),
        WCAGViolation(
            criterion=SC_1_4_3,
            node_id="text_1",
            impact=ImpactLevel.SERIOUS,
            message="Contrast ratio 2.5:1 below 4.5:1 threshold",
            evidence={"contrast_ratio": 2.5, "fg": "#999999", "bg": "#ffffff"},
        ),
        WCAGViolation(
            criterion=SC_2_1_1,
            node_id="menu_1",
            impact=ImpactLevel.CRITICAL,
            message="Menu not keyboard accessible",
        ),
    )


def _make_wcag_result() -> WCAGResult:
    return WCAGResult(
        violations=_make_violations(),
        target_level=ConformanceLevel.AA,
        criteria_tested=20,
        criteria_passed=17,
        page_url="https://example.com/page",
    )


# ===================================================================
# Tests — WCAG evaluation to cognitive cost
# ===================================================================


class TestWCAGToCognitiveCost:
    """WCAG violations feed into cognitive cost adjustment."""

    def test_violation_cost_positive(self) -> None:
        """Each WCAG violation maps to a positive cognitive cost."""
        for v in _make_violations():
            cost = compute_violation_cost(v)
            assert cost > 0, f"Violation cost not positive for {v.sc_id}"

    def test_critical_costs_more_than_minor(self) -> None:
        """Critical impact violations have higher cost than minor ones."""
        critical_v = WCAGViolation(
            criterion=SC_1_1_1, node_id="n1",
            impact=ImpactLevel.CRITICAL, message="critical",
        )
        minor_v = WCAGViolation(
            criterion=SC_1_4_6, node_id="n2",
            impact=ImpactLevel.MINOR, message="minor",
        )
        c_critical = compute_violation_cost(critical_v)
        c_minor = compute_violation_cost(minor_v)
        assert c_critical > c_minor, \
            f"Critical cost {c_critical} ≤ minor cost {c_minor}"

    def test_cost_element_has_positive_mu(self) -> None:
        """to_cost_element returns a CostElement with positive mu."""
        result = _make_wcag_result()
        delta = compute_cost_delta(result)
        cost_elem = to_cost_element(delta)
        assert cost_elem.mu > 0, f"CostElement mu not positive: {cost_elem.mu}"

    def test_cost_element_has_non_negative_sigma(self) -> None:
        """to_cost_element returns a CostElement with non-negative sigma_sq."""
        result = _make_wcag_result()
        delta = compute_cost_delta(result)
        cost_elem = to_cost_element(delta)
        assert cost_elem.sigma_sq >= 0, \
            f"sigma_sq negative: {cost_elem.sigma_sq}"

    def test_cost_delta_from_result(self) -> None:
        """compute_cost_delta returns a positive delta for a result with violations."""
        result = _make_wcag_result()
        delta = compute_cost_delta(result)
        assert isinstance(delta, CognitiveCostDelta)
        assert delta.mu_delta > 0

    def test_no_violations_zero_cost(self) -> None:
        """A result with no violations has zero cognitive cost delta."""
        result = WCAGResult(
            violations=(),
            target_level=ConformanceLevel.AA,
            criteria_tested=10,
            criteria_passed=10,
        )
        delta = compute_cost_delta(result)
        assert math.isclose(delta.mu_delta, 0.0, abs_tol=1e-6)

    def test_cost_summary(self) -> None:
        """wcag_cost_summary returns a summary dict with expected keys."""
        result = _make_wcag_result()
        summary = wcag_cost_summary(result)
        assert isinstance(summary, dict)
        assert "total_cognitive_cost_bits" in summary


# ===================================================================
# Tests — WCAG report generation
# ===================================================================


class TestWCAGReporting:
    """WCAG conformance report generation."""

    def test_compute_summary(self) -> None:
        """compute_summary returns a ReportSummary with counts."""
        result = _make_wcag_result()
        summary = compute_summary(result)
        assert isinstance(summary, ReportSummary)
        assert summary.total_violations == 3
        assert summary.by_impact.get("critical", 0) >= 2

    def test_rank_remediations(self) -> None:
        """rank_remediations returns ordered list of fixes."""
        result = _make_wcag_result()
        remediations = rank_remediations(result)
        assert len(remediations) > 0
        # Critical violations should be prioritized
        if len(remediations) >= 2:
            assert remediations[0].priority_score >= remediations[-1].priority_score

    def test_reporter_produces_output(self) -> None:
        """WCAGConformanceReporter produces a non-empty report."""
        result = _make_wcag_result()
        reporter = WCAGConformanceReporter()
        report = reporter.format_result(result)
        assert report is not None
        assert len(report) > 0


# ===================================================================
# Tests — WCAG violations to SARIF
# ===================================================================


class TestWCAGToSARIF:
    """WCAG violations are correctly converted to SARIF format."""

    def test_violation_to_sarif_result(self) -> None:
        """A WCAG violation converts to a SARIF Result."""
        v = _make_violations()[0]
        sarif_result = wcag_violation_to_sarif(v)
        assert sarif_result is not None
        assert sarif_result.rule_id is not None
        assert sarif_result.message is not None

    def test_wcag_result_to_sarif_run(self) -> None:
        """A WCAGResult converts to a SarifLog with all violations."""
        result = _make_wcag_result()
        sarif_log = wcag_result_to_sarif(result)
        assert sarif_log is not None
        assert len(sarif_log.runs) > 0
        # Each violation should produce a result in the first run
        total_results = sum(len(r.results) for r in sarif_log.runs)
        assert total_results == len(result.violations)

    def test_wcag_rule_id_format(self) -> None:
        """WCAG rule IDs follow expected format."""
        v = _make_violations()[0]
        rule_id = wcag_rule_id(v)
        assert isinstance(rule_id, str)
        assert len(rule_id) > 0

    def test_sarif_report_valid_json(self) -> None:
        """SARIF output from WCAG results is valid JSON."""
        builder = SarifBuilder(tool_name="wcag-checker")
        for v in _make_violations():
            rule_id = wcag_rule_id(v)
            builder.add_rule(rule_id, short_description=v.message)
            builder.add_result(
                rule_id=rule_id,
                message=v.message,
                level=Level.ERROR if v.impact == ImpactLevel.CRITICAL else Level.WARNING,
            )
        log = builder.build()
        json_str = sarif_to_string(log)
        assert len(json_str) > 0
        import json
        parsed = json.loads(json_str)
        assert parsed["version"] == "2.1.0"
        assert len(parsed["runs"]) > 0

    def test_sarif_results_have_rule_references(self) -> None:
        """Each SARIF result references its rule correctly."""
        result = _make_wcag_result()
        sarif_log = wcag_result_to_sarif(result)
        for run in sarif_log.runs:
            for r in run.results:
                assert r.rule_id is not None or r.rule_index is not None


# ===================================================================
# Tests — Contrast analysis integration
# ===================================================================


class TestContrastIntegration:
    """End-to-end contrast check pipeline."""

    def test_check_contrast_returns_result(self) -> None:
        """check_contrast returns a ContrastResult for valid inputs."""
        fg = Color(0, 0, 0)
        bg = Color(255, 255, 255)
        result = check_contrast(fg, bg)
        assert isinstance(result, ContrastResult)
        assert result.ratio >= 1.0

    def test_low_contrast_fails_aa(self) -> None:
        """Low-contrast combination fails AA threshold."""
        fg = Color(150, 150, 150)
        bg = Color(200, 200, 200)
        result = check_contrast(fg, bg)
        assert result.ratio < 4.5  # AA threshold for normal text

    def test_high_contrast_passes_aaa(self) -> None:
        """High-contrast combination passes AAA threshold."""
        fg = Color(0, 0, 0)
        bg = Color(255, 255, 255)
        result = check_contrast(fg, bg)
        assert result.ratio >= 7.0  # AAA threshold for normal text
