"""Tests for causalcert.reporting – JSON, HTML, LaTeX, narrative."""

from __future__ import annotations

import json

import numpy as np
import pytest

from causalcert.reporting.json_report import to_json_report, to_json_dict, get_schema
from causalcert.reporting.html_report import to_html_report, fragility_heatmap_svg
from causalcert.reporting.latex_report import (
    to_latex_tables,
    fragility_table,
    radius_summary_table,
)
from causalcert.reporting.narrative import generate_narrative
from causalcert.types import (
    AuditReport,
    EditType,
    EstimationResult,
    FragilityChannel,
    FragilityScore,
    RobustnessRadius,
    SolverStrategy,
    StructuralEdit,
)

# ── shared fixtures ───────────────────────────────────────────────────────

from tests.conftest import make_audit_report


# ═══════════════════════════════════════════════════════════════════════════
# JSON report
# ═══════════════════════════════════════════════════════════════════════════


class TestJSONReport:
    def test_to_json_report_is_valid_json(self) -> None:
        report = make_audit_report()
        j = to_json_report(report)
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_to_json_dict(self) -> None:
        report = make_audit_report()
        d = to_json_dict(report)
        assert isinstance(d, dict)
        assert "radius" in d or "robustness_radius" in d or "query" in d

    def test_json_contains_treatment_outcome(self) -> None:
        report = make_audit_report(treatment=0, outcome=3)
        d = to_json_dict(report)
        j = json.dumps(d)
        assert "0" in j or "treatment" in j

    def test_json_node_names(self) -> None:
        report = make_audit_report()
        j = to_json_report(report, node_names=["A", "B", "C", "D"])
        assert "A" in j or "B" in j

    def test_json_fragility_scores(self) -> None:
        report = make_audit_report()
        d = to_json_dict(report)
        j = json.dumps(d)
        # Should contain fragility information
        assert "fragil" in j.lower() or "score" in j.lower()

    def test_json_schema(self) -> None:
        schema = get_schema()
        assert isinstance(schema, dict)

    def test_json_round_trip_parseable(self) -> None:
        report = make_audit_report()
        j = to_json_report(report)
        parsed = json.loads(j)
        # Should be able to re-serialize
        j2 = json.dumps(parsed, indent=2)
        assert len(j2) > 0


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════


class TestHTMLReport:
    def test_to_html_report(self) -> None:
        report = make_audit_report()
        html = to_html_report(report)
        assert isinstance(html, str)
        assert "<html" in html.lower() or "<div" in html.lower() or "<!doctype" in html.lower()

    def test_html_contains_radius(self) -> None:
        report = make_audit_report(radius_lb=3, radius_ub=3)
        html = to_html_report(report)
        assert "3" in html

    def test_html_with_node_names(self) -> None:
        report = make_audit_report()
        html = to_html_report(report, node_names=["A", "B", "C", "D"])
        assert isinstance(html, str)

    def test_html_output_to_file(self, tmp_dir) -> None:
        from pathlib import Path
        report = make_audit_report()
        out = tmp_dir / "report.html"
        to_html_report(report, output_path=str(out))
        assert out.exists()
        content = out.read_text()
        assert len(content) > 0

    def test_fragility_heatmap_svg(self) -> None:
        scores = [
            FragilityScore(
                edge=(0, 1), total_score=0.9,
                channel_scores={FragilityChannel.D_SEPARATION: 0.8},
            ),
            FragilityScore(
                edge=(1, 2), total_score=0.3,
                channel_scores={FragilityChannel.D_SEPARATION: 0.2},
            ),
        ]
        svg = fragility_heatmap_svg(scores)
        assert isinstance(svg, str)
        assert "svg" in svg.lower() or "rect" in svg.lower()


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX report
# ═══════════════════════════════════════════════════════════════════════════


class TestLatexReport:
    def test_to_latex_tables(self) -> None:
        report = make_audit_report()
        tables = to_latex_tables(report)
        assert isinstance(tables, dict)
        assert len(tables) >= 1

    def test_fragility_table(self) -> None:
        scores = [
            FragilityScore(
                edge=(0, 1), total_score=0.9,
                channel_scores={
                    FragilityChannel.D_SEPARATION: 0.8,
                    FragilityChannel.IDENTIFICATION: 0.7,
                },
            ),
        ]
        tex = fragility_table(scores, top_k=5)
        assert "tabular" in tex or "table" in tex.lower()
        assert "0.9" in tex or "0.8" in tex

    def test_radius_summary_table(self) -> None:
        radius = RobustnessRadius(
            lower_bound=2, upper_bound=2, certified=True,
            solver_strategy=SolverStrategy.ILP,
        )
        tex = radius_summary_table(radius)
        assert "tabular" in tex or "table" in tex.lower()
        assert "2" in tex

    def test_latex_node_names(self) -> None:
        report = make_audit_report()
        tables = to_latex_tables(report, node_names=["A", "B", "C", "D"])
        assert isinstance(tables, dict)

    def test_latex_escaping(self) -> None:
        # Names with special LaTeX characters
        report = make_audit_report()
        tables = to_latex_tables(report, node_names=["X_0", "X_1", "X_2", "X_3"])
        # Should produce valid LaTeX (not crash)
        assert isinstance(tables, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Narrative
# ═══════════════════════════════════════════════════════════════════════════


class TestNarrative:
    def test_generate_narrative(self) -> None:
        report = make_audit_report()
        text = generate_narrative(report)
        assert isinstance(text, str)
        assert len(text) > 50

    def test_narrative_mentions_radius(self) -> None:
        report = make_audit_report(radius_lb=3, radius_ub=3)
        text = generate_narrative(report)
        assert "3" in text

    def test_narrative_with_node_names(self) -> None:
        report = make_audit_report()
        text = generate_narrative(report, node_names=["Treatment", "Mediator", "Confounder", "Outcome"])
        assert isinstance(text, str)

    def test_narrative_with_recommendations(self) -> None:
        report = make_audit_report()
        text = generate_narrative(report, include_recommendations=True)
        assert isinstance(text, str)

    def test_narrative_without_recommendations(self) -> None:
        report = make_audit_report()
        text = generate_narrative(report, include_recommendations=False)
        assert isinstance(text, str)


# ═══════════════════════════════════════════════════════════════════════════
# Report completeness
# ═══════════════════════════════════════════════════════════════════════════


class TestReportCompleteness:
    def test_json_has_all_sections(self) -> None:
        report = make_audit_report()
        d = to_json_dict(report)
        j = json.dumps(d).lower()
        # Should contain key sections
        assert any(kw in j for kw in ["radius", "fragil", "estimat"])

    def test_minimal_report(self) -> None:
        # Minimal report with no fragility or estimates
        report = AuditReport(
            treatment=0,
            outcome=1,
            n_nodes=2,
            n_edges=1,
            radius=RobustnessRadius(lower_bound=1, upper_bound=1, certified=True),
        )
        j = to_json_report(report)
        assert len(j) > 0
        html = to_html_report(report)
        assert len(html) > 0
        text = generate_narrative(report)
        assert len(text) > 0

    def test_report_with_perturbed_estimates(self) -> None:
        report = make_audit_report()
        report.perturbed_estimates = [
            EstimationResult(
                ate=0.5, se=0.4, ci_lower=-0.3, ci_upper=1.3,
                adjustment_set=frozenset(), method="aipw", n_obs=100,
            ),
        ]
        j = to_json_report(report)
        assert len(j) > 0
        html = to_html_report(report)
        assert len(html) > 0

    def test_report_with_many_fragility_scores(self) -> None:
        report = make_audit_report()
        scores = [
            FragilityScore(
                edge=(i, i + 1), total_score=0.1 * i,
                channel_scores={FragilityChannel.D_SEPARATION: 0.1 * i},
            )
            for i in range(10)
        ]
        report.fragility_ranking = scores
        tables = to_latex_tables(report, top_k=5)
        assert isinstance(tables, dict)

    def test_report_different_radius_values(self) -> None:
        for lb, ub in [(0, 0), (1, 1), (1, 5), (3, 3), (0, 10)]:
            report = make_audit_report(radius_lb=lb, radius_ub=ub)
            j = to_json_report(report)
            assert len(j) > 0

    def test_all_formats_produce_output(self) -> None:
        report = make_audit_report()
        j = to_json_report(report)
        h = to_html_report(report)
        t = to_latex_tables(report)
        n = generate_narrative(report)
        assert all(len(x) > 0 for x in [j, h, n])
        assert len(t) >= 1

    def test_narrative_radius_interpretation(self) -> None:
        for radius in [1, 2, 5, 10]:
            report = make_audit_report(radius_lb=radius, radius_ub=radius)
            text = generate_narrative(report)
            assert str(radius) in text


# ═══════════════════════════════════════════════════════════════════════════
# Report structure validation
# ═══════════════════════════════════════════════════════════════════════════


class TestReportStructure:
    def test_json_is_valid_json(self) -> None:
        report = make_audit_report()
        j = to_json_report(report)
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_html_has_tags(self) -> None:
        report = make_audit_report()
        h = to_html_report(report)
        assert "<html" in h.lower() or "<div" in h.lower() or "<table" in h.lower()

    def test_latex_has_begin(self) -> None:
        report = make_audit_report()
        t = to_latex_tables(report)
        assert isinstance(t, dict)
        for key, table in t.items():
            assert "\\begin" in table or len(table) > 0

    def test_narrative_not_empty(self) -> None:
        report = make_audit_report()
        n = generate_narrative(report)
        assert len(n.strip()) > 10

    @pytest.mark.parametrize("n_scores", [0, 1, 5, 10])
    def test_varying_fragility_scores(self, n_scores: int) -> None:
        report = make_audit_report()
        report.fragility_ranking = [
            FragilityScore(edge=(i, i + 1), total_score=0.5)
            for i in range(n_scores)
        ]
        j = to_json_report(report)
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_report_with_estimation(self) -> None:
        report = make_audit_report()
        report.baseline_estimate = EstimationResult(
            ate=1.5, se=0.3, ci_lower=0.5, ci_upper=2.5,
            adjustment_set=frozenset({0}),
        )
        j = to_json_report(report)
        parsed = json.loads(j)
        assert isinstance(parsed, dict)
