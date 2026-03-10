"""
usability_oracle.output.json_output — JSON formatter for pipeline results.

Produces pretty-printed JSON with a well-defined schema.  Includes custom
serialisation logic for cost tuples, bottleneck descriptions, and timing
data, plus a JSON-Schema definition that consumers can use for validation.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Optional

from usability_oracle.core.enums import (
    BottleneckType,
    RegressionVerdict,
    Severity,
)
from usability_oracle.output.models import (
    AnnotatedElement,
    BottleneckDescription,
    CostComparison,
    OutputSection,
    PipelineResult,
    StageTimingInfo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: float) -> float:
    """Replace NaN / Inf with JSON-safe sentinels."""
    if math.isnan(value):
        return 0.0
    if math.isinf(value):
        return 1e308 if value > 0 else -1e308
    return round(value, 8)


class _ResultEncoder(json.JSONEncoder):
    """Encoder that handles enums, dataclasses, and numpy-like types."""

    def default(self, o: Any) -> Any:
        # Enum handling
        if hasattr(o, "value") and isinstance(o, type) is False:
            return o.value
        # dataclass handling
        if hasattr(o, "to_dict"):
            return o.to_dict()
        # numpy scalars
        type_name = type(o).__module__
        if type_name == "numpy":
            return o.item()
        return super().default(o)


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------

class JSONFormatter:
    """Formats a :class:`PipelineResult` as pretty-printed JSON.

    Usage::

        formatter = JSONFormatter(indent=2)
        json_str = formatter.format(result)
    """

    def __init__(self, indent: int = 2, include_schema_ref: bool = True) -> None:
        self._indent = indent
        self._include_schema_ref = include_schema_ref

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format(self, result: PipelineResult) -> str:
        """Serialise *result* to a pretty-printed JSON string."""
        payload = self._build_payload(result)
        return json.dumps(payload, indent=self._indent, cls=_ResultEncoder, ensure_ascii=False)

    def format_compact(self, result: PipelineResult) -> str:
        """Serialise *result* to compact (no whitespace) JSON."""
        payload = self._build_payload(result)
        return json.dumps(payload, cls=_ResultEncoder, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def schema() -> dict[str, Any]:
        """Return a JSON-Schema (draft-07) describing the output format."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "UsabilityOraclePipelineResult",
            "description": "Schema for the JSON output of the Bounded-Rational Usability Oracle pipeline.",
            "type": "object",
            "required": ["verdict", "timestamp", "version"],
            "properties": {
                "version": {"type": "string", "const": "1.0.0"},
                "verdict": {
                    "type": "string",
                    "enum": [v.value for v in RegressionVerdict],
                },
                "timestamp": {"type": "string", "format": "date-time"},
                "comparison": {
                    "type": ["object", "null"],
                    "properties": {
                        "cost_a": {"$ref": "#/definitions/CostTuple"},
                        "cost_b": {"$ref": "#/definitions/CostTuple"},
                        "delta": {"$ref": "#/definitions/CostTuple"},
                        "percentage_change": {"type": "number"},
                        "channel_deltas": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                        },
                    },
                },
                "bottlenecks": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/Bottleneck"},
                },
                "annotated_elements": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/AnnotatedElement"},
                },
                "timing": {
                    "type": "object",
                    "properties": {
                        "stages": {
                            "type": "array",
                            "items": {"$ref": "#/definitions/StageTimingInfo"},
                        },
                        "total_seconds": {"type": "number"},
                    },
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "metadata": {"type": "object"},
            },
            "definitions": {
                "CostTuple": {
                    "type": "object",
                    "required": ["mu", "sigma_sq", "kappa"],
                    "properties": {
                        "mu": {"type": "number"},
                        "sigma_sq": {"type": "number"},
                        "kappa": {"type": "number"},
                        "lambda": {"type": "number"},
                    },
                },
                "Bottleneck": {
                    "type": "object",
                    "required": ["bottleneck_type", "severity", "description"],
                    "properties": {
                        "bottleneck_type": {
                            "type": "string",
                            "enum": [b.value for b in BottleneckType],
                        },
                        "severity": {
                            "type": "string",
                            "enum": [s.value for s in Severity],
                        },
                        "description": {"type": "string"},
                        "affected_elements": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "cost_impact": {"type": "number"},
                        "location": {"type": ["string", "null"]},
                        "recommendation": {"type": ["string", "null"]},
                    },
                },
                "AnnotatedElement": {
                    "type": "object",
                    "required": ["element_id", "annotation"],
                    "properties": {
                        "element_id": {"type": "string"},
                        "annotation": {"type": "string"},
                        "severity": {"type": "string"},
                        "location": {"type": ["string", "null"]},
                        "cost_contribution": {"type": "number"},
                        "recommendation": {"type": ["string", "null"]},
                    },
                },
                "StageTimingInfo": {
                    "type": "object",
                    "required": ["stage", "elapsed_seconds"],
                    "properties": {
                        "stage": {"type": "string"},
                        "elapsed_seconds": {"type": "number"},
                    },
                },
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(self, result: PipelineResult) -> dict[str, Any]:
        ts = datetime.fromtimestamp(result.timestamp, tz=timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "version": "1.0.0",
            "verdict": result.verdict.value,
            "timestamp": ts,
        }
        if self._include_schema_ref:
            payload["$schema"] = "usability-oracle-result-v1.0.0.json"
        payload["comparison"] = (
            self._serialize_comparison(result.comparison)
            if result.comparison
            else None
        )
        payload["bottlenecks"] = self._serialize_bottlenecks(result.bottlenecks)
        payload["annotated_elements"] = [
            ae.to_dict() for ae in result.annotated_elements
        ]
        payload["timing"] = self._serialize_timing(result.timing)
        payload["recommendations"] = result.recommendations
        payload["metadata"] = result.metadata
        return payload

    # -- comparison -----------------------------------------------------

    def _serialize_comparison(self, comparison: CostComparison) -> dict[str, Any]:
        return {
            "cost_a": self._serialize_costs(comparison.cost_a),
            "cost_b": self._serialize_costs(comparison.cost_b),
            "delta": self._serialize_costs(comparison.delta),
            "percentage_change": _safe_float(comparison.percentage_change),
            "channel_deltas": {
                k: _safe_float(v) for k, v in comparison.channel_deltas.items()
            },
        }

    # -- costs ----------------------------------------------------------

    @staticmethod
    def _serialize_costs(costs: Any) -> dict[str, float]:
        d = costs.to_dict() if hasattr(costs, "to_dict") else {}
        return {k: _safe_float(float(v)) for k, v in d.items()}

    # -- bottlenecks ----------------------------------------------------

    @staticmethod
    def _serialize_bottlenecks(bottlenecks: list[BottleneckDescription]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for b in bottlenecks:
            out.append({
                "bottleneck_type": b.bottleneck_type.value,
                "severity": b.severity.value,
                "description": b.description,
                "affected_elements": b.affected_elements,
                "cost_impact": _safe_float(b.cost_impact),
                "location": b.location,
                "recommendation": b.recommendation,
                "metadata": b.metadata,
            })
        return out

    # -- timing ---------------------------------------------------------

    @staticmethod
    def _serialize_timing(timing: list[StageTimingInfo]) -> dict[str, Any]:
        stages = []
        for t in timing:
            stages.append({
                "stage": t.stage.value,
                "elapsed_seconds": _safe_float(t.elapsed_seconds),
                "start_time": _safe_float(t.start_time),
                "end_time": _safe_float(t.end_time),
            })
        total = sum(t.elapsed_seconds for t in timing)
        return {
            "stages": stages,
            "total_seconds": _safe_float(total),
        }
