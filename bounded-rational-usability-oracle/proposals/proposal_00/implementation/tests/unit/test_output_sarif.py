"""Unit tests for usability_oracle.output.sarif.SARIFFormatter.

Tests that the SARIF formatter produces valid SARIF 2.1.0 JSON,
includes proper tool metadata, rule definitions for every bottleneck type,
and correctly maps severity levels to SARIF levels.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from usability_oracle.core.enums import (
    BottleneckType,
    PipelineStage,
    RegressionVerdict,
    Severity,
)
from usability_oracle.core.types import CostTuple
from usability_oracle.output.models import (
    BottleneckDescription,
    CostComparison,
    PipelineResult,
    StageTimingInfo,
)
from usability_oracle.output.sarif import SARIFFormatter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bottleneck(
    bt: BottleneckType = BottleneckType.PERCEPTUAL_OVERLOAD,
    severity: Severity = Severity.HIGH,
    cost_impact: float = 2.0,
) -> BottleneckDescription:
    """Build a minimal BottleneckDescription."""
    return BottleneckDescription(
        bottleneck_type=bt,
        severity=severity,
        description=f"Test {bt.value}",
        affected_elements=[],
        cost_impact=cost_impact,
    )


def _make_timing(stage: PipelineStage = PipelineStage.PARSE, elapsed: float = 0.3) -> StageTimingInfo:
    now = time.time()
    return StageTimingInfo(
        stage=stage,
        elapsed_seconds=elapsed,
        start_time=now - elapsed,
        end_time=now,
    )


def _make_pipeline_result(**overrides) -> PipelineResult:
    """Build a PipelineResult with sensible defaults for SARIF tests."""
    defaults = dict(
        verdict=RegressionVerdict.REGRESSION,
        comparison=CostComparison(
            cost_a=CostTuple(mu=2.0),
            cost_b=CostTuple(mu=4.0),
            delta=CostTuple(mu=2.0),
            percentage_change=100.0,
            channel_deltas={},
        ),
        bottlenecks=[
            _make_bottleneck(BottleneckType.PERCEPTUAL_OVERLOAD, Severity.CRITICAL),
            _make_bottleneck(BottleneckType.MOTOR_DIFFICULTY, Severity.HIGH),
        ],
        annotated_elements=[],
        timing=[_make_timing()],
        sections=[],
        recommendations=["Fix perceptual overload"],
        metadata={},
        timestamp=time.time(),
    )
    defaults.update(overrides)
    return PipelineResult(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestSARIFFormatterConstruction:
    """Tests for SARIFFormatter instantiation."""

    def test_default_construction(self):
        """SARIFFormatter can be created with no arguments."""
        fmt = SARIFFormatter()
        assert fmt is not None

    def test_custom_indent(self):
        """indent parameter is stored."""
        fmt = SARIFFormatter(indent=4)
        assert fmt._indent == 4

    def test_custom_pretty(self):
        """pretty parameter is stored."""
        fmt = SARIFFormatter(pretty=False)
        assert fmt._pretty is False


# ---------------------------------------------------------------------------
# format() — valid SARIF JSON
# ---------------------------------------------------------------------------

class TestSARIFFormatterFormat:
    """Tests that format() produces valid SARIF 2.1.0 JSON."""

    def test_format_returns_string(self):
        """format() returns a string."""
        fmt = SARIFFormatter()
        result = fmt.format(_make_pipeline_result())
        assert isinstance(result, str)

    def test_format_valid_json(self):
        """format() output is parseable as JSON."""
        fmt = SARIFFormatter()
        output = fmt.format(_make_pipeline_result())
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_sarif_version(self):
        """SARIF version field is '2.1.0'."""
        fmt = SARIFFormatter()
        output = fmt.format(_make_pipeline_result())
        parsed = json.loads(output)
        assert parsed.get("version") == "2.1.0"

    def test_sarif_schema_present(self):
        """$schema field is present in the SARIF output."""
        fmt = SARIFFormatter()
        output = fmt.format(_make_pipeline_result())
        parsed = json.loads(output)
        assert "$schema" in parsed

    def test_sarif_has_runs(self):
        """SARIF output contains a 'runs' array."""
        fmt = SARIFFormatter()
        output = fmt.format(_make_pipeline_result())
        parsed = json.loads(output)
        assert "runs" in parsed
        assert isinstance(parsed["runs"], list)
        assert len(parsed["runs"]) >= 1


# ---------------------------------------------------------------------------
# Tool name and metadata
# ---------------------------------------------------------------------------

class TestSARIFToolMetadata:
    """Tests that the SARIF tool component has correct metadata."""

    def _get_tool(self) -> dict:
        """Parse SARIF and extract the tool component."""
        fmt = SARIFFormatter()
        output = fmt.format(_make_pipeline_result())
        parsed = json.loads(output)
        return parsed["runs"][0]["tool"]

    def test_tool_driver_present(self):
        """Tool section includes a 'driver' object."""
        tool = self._get_tool()
        assert "driver" in tool

    def test_tool_name(self):
        """Driver name identifies the usability oracle."""
        tool = self._get_tool()
        name = tool["driver"].get("name", "")
        assert "usability" in name.lower() or "oracle" in name.lower()

    def test_tool_version_present(self):
        """Driver includes a version string."""
        tool = self._get_tool()
        assert "version" in tool["driver"] or "semanticVersion" in tool["driver"]

    def test_tool_information_uri(self):
        """Driver may include an informationUri."""
        tool = self._get_tool()
        # May or may not be present; if present, should be a string
        uri = tool["driver"].get("informationUri")
        if uri is not None:
            assert isinstance(uri, str)


# ---------------------------------------------------------------------------
# Rules list
# ---------------------------------------------------------------------------

class TestSARIFRules:
    """Tests that SARIF rules cover all bottleneck types."""

    def _get_rules(self) -> list:
        """Parse SARIF and return the rules array."""
        fmt = SARIFFormatter()
        output = fmt.format(_make_pipeline_result())
        parsed = json.loads(output)
        return parsed["runs"][0]["tool"]["driver"].get("rules", [])

    def test_rules_present(self):
        """Rules array exists and is non-empty."""
        rules = self._get_rules()
        assert len(rules) >= 1

    def test_rules_have_ids(self):
        """Every rule has an 'id' field."""
        for rule in self._get_rules():
            assert "id" in rule, f"Rule missing id: {rule}"

    def test_rules_have_descriptions(self):
        """Every rule has a description or shortDescription."""
        for rule in self._get_rules():
            has_desc = (
                "shortDescription" in rule
                or "fullDescription" in rule
                or "help" in rule
            )
            assert has_desc, f"Rule {rule.get('id')} missing description"

    def test_rules_cover_bottleneck_types(self):
        """At least one rule per BottleneckType value."""
        rules = self._get_rules()
        rule_ids = {r["id"] for r in rules}
        # There should be at least as many rules as bottleneck types
        assert len(rule_ids) >= len(BottleneckType)


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

class TestSARIFSeverityMapping:
    """Tests that severity levels are correctly mapped to SARIF levels."""

    def _get_results(self) -> list:
        """Parse SARIF and return the results array."""
        fmt = SARIFFormatter()
        pr = _make_pipeline_result(bottlenecks=[
            _make_bottleneck(BottleneckType.PERCEPTUAL_OVERLOAD, Severity.CRITICAL),
            _make_bottleneck(BottleneckType.MOTOR_DIFFICULTY, Severity.HIGH),
            _make_bottleneck(BottleneckType.CHOICE_PARALYSIS, Severity.MEDIUM),
            _make_bottleneck(BottleneckType.MEMORY_DECAY, Severity.LOW),
        ])
        output = fmt.format(pr)
        parsed = json.loads(output)
        return parsed["runs"][0].get("results", [])

    def test_critical_maps_to_error(self):
        """Critical severity maps to SARIF level 'error'."""
        results = self._get_results()
        critical = [r for r in results if r.get("level") == "error"]
        assert len(critical) >= 1

    def test_high_maps_to_warning(self):
        """High severity maps to SARIF level 'warning'."""
        results = self._get_results()
        warnings = [r for r in results if r.get("level") == "warning"]
        assert len(warnings) >= 1

    def test_results_have_rule_id(self):
        """Each result references a ruleId."""
        results = self._get_results()
        for r in results:
            assert "ruleId" in r, f"Result missing ruleId: {r}"

    def test_results_have_message(self):
        """Each result has a message field."""
        results = self._get_results()
        for r in results:
            assert "message" in r, f"Result missing message: {r}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestSARIFEdgeCases:
    """Edge-case tests for the SARIF formatter."""

    def test_empty_bottlenecks(self):
        """format works with zero bottlenecks."""
        fmt = SARIFFormatter()
        pr = _make_pipeline_result(bottlenecks=[])
        output = fmt.format(pr)
        parsed = json.loads(output)
        assert parsed.get("version") == "2.1.0"

    def test_all_bottleneck_types(self):
        """format handles every BottleneckType."""
        fmt = SARIFFormatter()
        bns = [_make_bottleneck(bt) for bt in BottleneckType]
        pr = _make_pipeline_result(bottlenecks=bns)
        output = fmt.format(pr)
        parsed = json.loads(output)
        assert len(parsed["runs"][0].get("results", [])) >= len(BottleneckType)

    def test_improvement_verdict(self):
        """format works for IMPROVEMENT verdicts."""
        fmt = SARIFFormatter()
        pr = _make_pipeline_result(verdict=RegressionVerdict.IMPROVEMENT)
        output = fmt.format(pr)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_nan_cost_handled(self):
        """NaN in costs does not break SARIF generation."""
        fmt = SARIFFormatter()
        pr = _make_pipeline_result(
            comparison=CostComparison(
                cost_a=CostTuple(mu=float("nan")),
                cost_b=CostTuple(mu=1.0),
                delta=CostTuple(mu=float("nan")),
                percentage_change=float("nan"),
                channel_deltas={},
            )
        )
        output = fmt.format(pr)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_neutral_verdict_no_crash(self):
        """Neutral verdict formats without errors."""
        fmt = SARIFFormatter()
        pr = _make_pipeline_result(verdict=RegressionVerdict.NEUTRAL, bottlenecks=[])
        output = fmt.format(pr)
        parsed = json.loads(output)
        assert parsed["version"] == "2.1.0"
