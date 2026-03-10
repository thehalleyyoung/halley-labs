"""Unit tests for SARIF output formatting.

Tests SARIF JSON structure, version, $schema, rule generation, location
references, and JSON validity.
"""

from __future__ import annotations

import json
import time

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

def _bottleneck(bt=BottleneckType.PERCEPTUAL_OVERLOAD, severity=Severity.HIGH,
                cost_impact=2.0, affected=None, location=None) -> BottleneckDescription:
    return BottleneckDescription(
        bottleneck_type=bt, severity=severity,
        description=f"Test {bt.value}", affected_elements=affected or [],
        cost_impact=cost_impact, location=location,
    )


def _timing(stage=PipelineStage.PARSE, elapsed=0.3) -> StageTimingInfo:
    now = time.time()
    return StageTimingInfo(stage=stage, elapsed_seconds=elapsed,
                           start_time=now - elapsed, end_time=now)


def _pipeline_result(bottlenecks=None) -> PipelineResult:
    if bottlenecks is None:
        bottlenecks = [_bottleneck()]
    cost_a = CostTuple(mu=2.5, sigma_sq=0.01)
    cost_b = CostTuple(mu=3.7, sigma_sq=0.02)
    comparison = CostComparison(
        cost_a=cost_a, cost_b=cost_b, delta=cost_b.mu - cost_a.mu,
        percentage_change=48.0,
        channel_deltas={"perceptual": 0.5, "cognitive": 0.5, "motor": 0.2},
    )
    return PipelineResult(
        verdict=RegressionVerdict.REGRESSION,
        comparison=comparison,
        bottlenecks=bottlenecks,
        annotated_elements=[],
        timing=[_timing()],
        sections=[],
        recommendations=["Fix overload"],
        metadata={},
        timestamp=time.time(),
    )


# ===================================================================
# JSON validity
# ===================================================================


class TestSARIFJSON:

    def test_valid_json(self):
        fmt = SARIFFormatter()
        sarif_str = fmt.format(_pipeline_result())
        parsed = json.loads(sarif_str)
        assert isinstance(parsed, dict)

    def test_schema_field_present(self):
        fmt = SARIFFormatter()
        sarif_str = fmt.format(_pipeline_result())
        parsed = json.loads(sarif_str)
        assert "$schema" in parsed
        assert "sarif" in parsed["$schema"].lower()

    def test_version_is_2_1_0(self):
        fmt = SARIFFormatter()
        sarif_str = fmt.format(_pipeline_result())
        parsed = json.loads(sarif_str)
        assert parsed["version"] == "2.1.0"


# ===================================================================
# Runs structure
# ===================================================================


class TestSARIFRuns:

    def test_runs_array_present(self):
        fmt = SARIFFormatter()
        parsed = json.loads(fmt.format(_pipeline_result()))
        assert "runs" in parsed
        assert isinstance(parsed["runs"], list)
        assert len(parsed["runs"]) >= 1

    def test_tool_present(self):
        fmt = SARIFFormatter()
        parsed = json.loads(fmt.format(_pipeline_result()))
        run = parsed["runs"][0]
        assert "tool" in run
        assert "driver" in run["tool"]

    def test_tool_name(self):
        fmt = SARIFFormatter()
        parsed = json.loads(fmt.format(_pipeline_result()))
        driver = parsed["runs"][0]["tool"]["driver"]
        assert "name" in driver
        assert driver["name"] == "usability-oracle"


# ===================================================================
# Rules
# ===================================================================


class TestSARIFRules:

    def test_rules_generated(self):
        fmt = SARIFFormatter()
        parsed = json.loads(fmt.format(_pipeline_result()))
        rules = parsed["runs"][0]["tool"]["driver"]["rules"]
        assert isinstance(rules, list)
        assert len(rules) >= 1

    def test_rules_for_each_bottleneck_type(self):
        fmt = SARIFFormatter()
        bottlenecks = [
            _bottleneck(BottleneckType.PERCEPTUAL_OVERLOAD),
            _bottleneck(BottleneckType.CHOICE_PARALYSIS),
            _bottleneck(BottleneckType.MOTOR_DIFFICULTY),
        ]
        parsed = json.loads(fmt.format(_pipeline_result(bottlenecks)))
        rules = parsed["runs"][0]["tool"]["driver"]["rules"]
        rule_ids = {r["id"] for r in rules}
        # Should have rules for all registered bottleneck types
        assert len(rule_ids) >= 3

    def test_rule_has_description(self):
        fmt = SARIFFormatter()
        parsed = json.loads(fmt.format(_pipeline_result()))
        rules = parsed["runs"][0]["tool"]["driver"]["rules"]
        for rule in rules:
            assert "shortDescription" in rule or "fullDescription" in rule


# ===================================================================
# Results
# ===================================================================


class TestSARIFResults:

    def test_results_present(self):
        fmt = SARIFFormatter()
        parsed = json.loads(fmt.format(_pipeline_result()))
        results = parsed["runs"][0]["results"]
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_result_has_rule_id(self):
        fmt = SARIFFormatter()
        parsed = json.loads(fmt.format(_pipeline_result()))
        result = parsed["runs"][0]["results"][0]
        assert "ruleId" in result

    def test_result_has_level(self):
        fmt = SARIFFormatter()
        parsed = json.loads(fmt.format(_pipeline_result()))
        result = parsed["runs"][0]["results"][0]
        assert "level" in result
        assert result["level"] in ("error", "warning", "note", "none")


# ===================================================================
# Locations
# ===================================================================


class TestSARIFLocations:

    def test_locations_present(self):
        bottleneck = _bottleneck(affected=["elem1", "elem2"], location="ui-tree")
        fmt = SARIFFormatter()
        parsed = json.loads(fmt.format(_pipeline_result([bottleneck])))
        result = parsed["runs"][0]["results"][0]
        assert "locations" in result
        assert len(result["locations"]) >= 1

    def test_location_has_artifact(self):
        bottleneck = _bottleneck(affected=["elem1"], location="a11y-tree.json")
        fmt = SARIFFormatter()
        parsed = json.loads(fmt.format(_pipeline_result([bottleneck])))
        loc = parsed["runs"][0]["results"][0]["locations"][0]
        assert "physicalLocation" in loc
        assert "artifactLocation" in loc["physicalLocation"]

    def test_logical_locations_for_elements(self):
        bottleneck = _bottleneck(affected=["btn1", "input2"])
        fmt = SARIFFormatter()
        parsed = json.loads(fmt.format(_pipeline_result([bottleneck])))
        loc = parsed["runs"][0]["results"][0]["locations"][0]
        if "logicalLocations" in loc:
            names = [ll["name"] for ll in loc["logicalLocations"]]
            assert "btn1" in names


# ===================================================================
# No bottlenecks
# ===================================================================


class TestNoBottlenecks:

    def test_empty_bottlenecks_valid(self):
        fmt = SARIFFormatter()
        sarif_str = fmt.format(_pipeline_result([]))
        parsed = json.loads(sarif_str)
        assert parsed["version"] == "2.1.0"
        assert len(parsed["runs"][0]["results"]) == 0
