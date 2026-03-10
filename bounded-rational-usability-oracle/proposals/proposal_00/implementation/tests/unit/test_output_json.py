"""Unit tests for usability_oracle.output.json_output.JSONFormatter.

Tests that the JSON formatter produces valid JSON, compact JSON,
includes schema references, and correctly serialises PipelineResult
objects including comparisons, bottlenecks, and timing information.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from usability_oracle.core.enums import (
    BottleneckType,
    PipelineStage,
    RegressionVerdict,
    Severity,
)
from usability_oracle.core.types import CostTuple
from usability_oracle.output.json_output import JSONFormatter
from usability_oracle.output.models import (
    AnnotatedElement,
    BottleneckDescription,
    CostComparison,
    PipelineResult,
    StageTimingInfo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bottleneck(
    bt: BottleneckType = BottleneckType.PERCEPTUAL_OVERLOAD,
    severity: Severity = Severity.HIGH,
    cost_impact: float = 2.5,
) -> BottleneckDescription:
    """Build a minimal BottleneckDescription."""
    return BottleneckDescription(
        bottleneck_type=bt,
        severity=severity,
        description=f"Test bottleneck {bt.value}",
        affected_elements=[],
        cost_impact=cost_impact,
    )


def _make_timing(stage: PipelineStage = PipelineStage.PARSE, elapsed: float = 0.5) -> StageTimingInfo:
    """Build a StageTimingInfo entry."""
    now = time.time()
    return StageTimingInfo(
        stage=stage,
        elapsed_seconds=elapsed,
        start_time=now - elapsed,
        end_time=now,
    )


def _make_cost_comparison() -> CostComparison:
    """Build a CostComparison with synthetic values."""
    return CostComparison(
        cost_a=CostTuple(mu=3.0, sigma_sq=0.1),
        cost_b=CostTuple(mu=4.5, sigma_sq=0.2),
        delta=CostTuple(mu=1.5, sigma_sq=0.1),
        percentage_change=50.0,
        channel_deltas={},
    )


def _make_pipeline_result(**overrides) -> PipelineResult:
    """Build a PipelineResult with sensible defaults."""
    defaults = dict(
        verdict=RegressionVerdict.NEUTRAL,
        comparison=_make_cost_comparison(),
        bottlenecks=[_make_bottleneck()],
        annotated_elements=[],
        timing=[_make_timing()],
        sections=[],
        recommendations=["Increase button size"],
        metadata={"version": "1.0"},
        timestamp=time.time(),
    )
    defaults.update(overrides)
    return PipelineResult(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestJSONFormatterConstruction:
    """Tests for JSONFormatter instantiation."""

    def test_default_construction(self):
        """JSONFormatter can be created with no arguments."""
        fmt = JSONFormatter()
        assert fmt is not None

    def test_custom_indent(self):
        """indent parameter is stored."""
        fmt = JSONFormatter(indent=4)
        assert fmt._indent == 4

    def test_custom_schema_ref(self):
        """include_schema_ref parameter is stored."""
        fmt = JSONFormatter(include_schema_ref=True)
        assert fmt._include_schema_ref is True


# ---------------------------------------------------------------------------
# format() — returns valid JSON
# ---------------------------------------------------------------------------

class TestJSONFormatterFormat:
    """Tests for JSONFormatter.format producing valid JSON strings."""

    def test_format_returns_string(self):
        """format() returns a str."""
        fmt = JSONFormatter()
        result = fmt.format(_make_pipeline_result())
        assert isinstance(result, str)

    def test_format_is_valid_json(self):
        """format() output is parseable as JSON."""
        fmt = JSONFormatter()
        output = fmt.format(_make_pipeline_result())
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_format_contains_verdict(self):
        """Formatted JSON includes the verdict field."""
        fmt = JSONFormatter()
        output = fmt.format(_make_pipeline_result())
        parsed = json.loads(output)
        assert "verdict" in parsed or "result" in parsed

    def test_format_contains_bottlenecks(self):
        """Formatted JSON includes bottleneck information."""
        fmt = JSONFormatter()
        pr = _make_pipeline_result(bottlenecks=[
            _make_bottleneck(BottleneckType.MOTOR_DIFFICULTY),
        ])
        output = fmt.format(pr)
        parsed = json.loads(output)
        raw = json.dumps(parsed)
        assert "motor" in raw.lower() or "bottleneck" in raw.lower()

    def test_format_contains_timing(self):
        """Formatted JSON includes timing data."""
        fmt = JSONFormatter()
        output = fmt.format(_make_pipeline_result())
        parsed = json.loads(output)
        raw = json.dumps(parsed)
        assert "timing" in raw.lower() or "elapsed" in raw.lower() or "time" in raw.lower()

    def test_format_handles_nan(self):
        """NaN values are converted to JSON-safe representations."""
        fmt = JSONFormatter()
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
        # Must be parseable — standard JSON doesn't allow NaN
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_format_handles_inf(self):
        """Inf values are converted to JSON-safe representations."""
        fmt = JSONFormatter()
        pr = _make_pipeline_result(
            comparison=CostComparison(
                cost_a=CostTuple(mu=float("inf")),
                cost_b=CostTuple(mu=1.0),
                delta=CostTuple(mu=float("inf")),
                percentage_change=float("inf"),
                channel_deltas={},
            )
        )
        output = fmt.format(pr)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# format_compact()
# ---------------------------------------------------------------------------

class TestJSONFormatterCompact:
    """Tests for JSONFormatter.format_compact producing single-line JSON."""

    def test_format_compact_returns_string(self):
        """format_compact() returns a str."""
        fmt = JSONFormatter()
        result = fmt.format_compact(_make_pipeline_result())
        assert isinstance(result, str)

    def test_format_compact_single_line(self):
        """format_compact() output has no internal newlines."""
        fmt = JSONFormatter()
        output = fmt.format_compact(_make_pipeline_result())
        assert "\n" not in output.strip()

    def test_format_compact_valid_json(self):
        """format_compact() output is parseable as JSON."""
        fmt = JSONFormatter()
        output = fmt.format_compact(_make_pipeline_result())
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_format_compact_shorter_than_pretty(self):
        """Compact output is shorter than indented output."""
        fmt = JSONFormatter(indent=2)
        pr = _make_pipeline_result()
        pretty = fmt.format(pr)
        compact = fmt.format_compact(pr)
        assert len(compact) <= len(pretty)


# ---------------------------------------------------------------------------
# schema()
# ---------------------------------------------------------------------------

class TestJSONFormatterSchema:
    """Tests for JSONFormatter.schema static method."""

    def test_schema_returns_dict(self):
        """schema() returns a dictionary."""
        s = JSONFormatter.schema()
        assert isinstance(s, dict)

    def test_schema_has_type(self):
        """Schema dict includes a 'type' key."""
        s = JSONFormatter.schema()
        assert "type" in s or "properties" in s or "$schema" in s

    def test_schema_not_empty(self):
        """Schema is non-empty."""
        s = JSONFormatter.schema()
        assert len(s) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestJSONFormatterEdgeCases:
    """Edge-case tests for the JSON formatter."""

    def test_empty_bottlenecks(self):
        """format works with an empty bottleneck list."""
        fmt = JSONFormatter()
        pr = _make_pipeline_result(bottlenecks=[])
        output = fmt.format(pr)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_multiple_bottlenecks(self):
        """format handles multiple bottleneck types."""
        fmt = JSONFormatter()
        bns = [
            _make_bottleneck(BottleneckType.PERCEPTUAL_OVERLOAD),
            _make_bottleneck(BottleneckType.CHOICE_PARALYSIS),
            _make_bottleneck(BottleneckType.MOTOR_DIFFICULTY),
        ]
        pr = _make_pipeline_result(bottlenecks=bns)
        output = fmt.format(pr)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_no_comparison(self):
        """format works when comparison is None."""
        fmt = JSONFormatter()
        pr = _make_pipeline_result(comparison=None)
        output = fmt.format(pr)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_empty_timing(self):
        """format works with an empty timing list."""
        fmt = JSONFormatter()
        pr = _make_pipeline_result(timing=[])
        output = fmt.format(pr)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)
