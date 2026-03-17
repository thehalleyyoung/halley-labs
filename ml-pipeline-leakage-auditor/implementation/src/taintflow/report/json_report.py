"""
taintflow.report.json_report – Structured JSON report generation.

Produces machine-readable JSON output suitable for programmatic consumption,
CI pipelines, and long-term archival.  Supports compact and pretty-print
modes, streaming writes for large reports, and includes a JSON Schema
definition so consumers can validate the output.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    IO,
    Iterator,
    List,
    Optional,
    TextIO,
    Union,
)

from taintflow.core.types import (
    Severity,
    FeatureLeakage,
    StageLeakage,
    LeakageReport,
)
from taintflow.core.config import TaintFlowConfig, SeverityThresholds


# ===================================================================
#  Constants
# ===================================================================

_TOOL_VERSION = "0.1.0"
_SCHEMA_VERSION = "1.0.0"


# ===================================================================
#  JSON Schema for the report format
# ===================================================================

REPORT_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://taintflow.dev/schemas/report-v1.json",
    "title": "TaintFlow Leakage Report",
    "description": "Output of a TaintFlow ML pipeline leakage audit.",
    "type": "object",
    "required": ["metadata", "summary", "features", "stages"],
    "properties": {
        "metadata": {
            "type": "object",
            "required": ["tool", "version", "timestamp"],
            "properties": {
                "tool": {"type": "string", "const": "taintflow"},
                "version": {"type": "string"},
                "schema_version": {"type": "string"},
                "timestamp": {"type": "string", "format": "date-time"},
                "config": {"type": "object"},
            },
        },
        "summary": {
            "type": "object",
            "required": [
                "total_leakage_bits",
                "severity",
                "n_features",
                "n_stages",
            ],
            "properties": {
                "total_leakage_bits": {"type": "number", "minimum": 0},
                "severity": {
                    "type": "string",
                    "enum": ["negligible", "warning", "critical"],
                },
                "n_features": {"type": "integer", "minimum": 0},
                "n_leaking_features": {"type": "integer", "minimum": 0},
                "n_stages": {"type": "integer", "minimum": 0},
                "analysis_duration_ms": {"type": "number", "minimum": 0},
                "pipeline_name": {"type": "string"},
                "is_clean": {"type": "boolean"},
            },
        },
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["column", "bit_bound", "severity"],
                "properties": {
                    "column": {"type": "string"},
                    "bit_bound": {"type": "number", "minimum": 0},
                    "severity": {
                        "type": "string",
                        "enum": ["negligible", "warning", "critical"],
                    },
                    "origins": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "contributing_stages": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "remediation": {"type": "string"},
                    "explanation": {"type": "string"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
            },
        },
        "stages": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["stage_id", "stage_name", "severity"],
                "properties": {
                    "stage_id": {"type": "string"},
                    "stage_name": {"type": "string"},
                    "op_type": {"type": "string"},
                    "node_kind": {"type": "string"},
                    "max_bit_bound": {"type": "number", "minimum": 0},
                    "mean_bit_bound": {"type": "number", "minimum": 0},
                    "severity": {
                        "type": "string",
                        "enum": ["negligible", "warning", "critical"],
                    },
                    "n_leaking_features": {"type": "integer", "minimum": 0},
                    "total_bit_bound": {"type": "number", "minimum": 0},
                    "description": {"type": "string"},
                    "features": {"type": "array"},
                },
            },
        },
        "bottlenecks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["stage_id", "rank"],
                "properties": {
                    "rank": {"type": "integer", "minimum": 1},
                    "stage_id": {"type": "string"},
                    "stage_name": {"type": "string"},
                    "max_bit_bound": {"type": "number"},
                    "severity": {"type": "string"},
                },
            },
        },
        "remediations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "stage": {"type": "string"},
                    "severity": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
            },
        },
        "dag": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "kind": {"type": "string"},
                            "op_type": {"type": "string"},
                        },
                    },
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
    "additionalProperties": False,
}


# ===================================================================
#  Helper – collect all FeatureLeakage across stages
# ===================================================================

def _collect_features(report: LeakageReport) -> List[Dict[str, Any]]:
    """Flatten per-stage features into a sorted list of dicts."""
    seen: Dict[str, Dict[str, Any]] = {}
    for sl in report.stage_leakages:
        for fl in sl.feature_leakages:
            key = f"{fl.column_name}::{sl.stage_id}"
            entry: Dict[str, Any] = {
                "column": fl.column_name,
                "bit_bound": fl.bit_bound,
                "severity": fl.severity.value,
                "origins": sorted(o.value for o in fl.origins),
                "contributing_stages": fl.contributing_stages or [sl.stage_id],
                "remediation": fl.remediation,
                "explanation": fl.explanation,
                "confidence": fl.confidence,
                "stage_id": sl.stage_id,
            }
            seen[key] = entry

    return sorted(seen.values(), key=lambda d: (-d["bit_bound"], d["column"]))


def _collect_stages(report: LeakageReport) -> List[Dict[str, Any]]:
    """Build the per-stage section."""
    result: List[Dict[str, Any]] = []
    for sl in report.stage_leakages:
        result.append({
            "stage_id": sl.stage_id,
            "stage_name": sl.stage_name,
            "op_type": sl.op_type.value,
            "node_kind": sl.node_kind.value,
            "max_bit_bound": sl.max_bit_bound,
            "mean_bit_bound": sl.mean_bit_bound,
            "severity": sl.severity.value,
            "n_leaking_features": sl.n_leaking_features,
            "total_bit_bound": sl.total_bit_bound,
            "description": sl.description,
            "features": [fl.to_dict() for fl in sl.feature_leakages],
        })
    return sorted(result, key=lambda d: (-d["max_bit_bound"], d["stage_id"]))


def _collect_bottlenecks(report: LeakageReport, top_n: int = 10) -> List[Dict[str, Any]]:
    """Rank stages by max leakage as bottlenecks."""
    stages = report.stages_by_severity()
    result: List[Dict[str, Any]] = []
    for rank, sl in enumerate(stages[:top_n], start=1):
        result.append({
            "rank": rank,
            "stage_id": sl.stage_id,
            "stage_name": sl.stage_name,
            "max_bit_bound": sl.max_bit_bound,
            "mean_bit_bound": sl.mean_bit_bound,
            "severity": sl.severity.value,
            "n_leaking_features": sl.n_leaking_features,
        })
    return result


def _collect_remediations(report: LeakageReport) -> List[Dict[str, Any]]:
    """Collect unique remediation suggestions."""
    result: List[Dict[str, Any]] = []
    seen_keys: set = set()
    for sl in report.stages_by_severity():
        for fl in sl.feature_leakages:
            if not fl.remediation:
                continue
            key = (fl.column_name, sl.stage_id, fl.remediation)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            result.append({
                "column": fl.column_name,
                "stage_id": sl.stage_id,
                "stage_name": sl.stage_name,
                "severity": fl.severity.value,
                "bit_bound": fl.bit_bound,
                "suggestion": fl.remediation,
            })
    return result


def _build_dag_dict(report: LeakageReport) -> Dict[str, Any]:
    """Serialise the pipeline DAG from stage information."""
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    prev_id: Optional[str] = None

    for sl in report.stage_leakages:
        nodes.append({
            "id": sl.stage_id,
            "name": sl.stage_name,
            "kind": sl.node_kind.value,
            "op_type": sl.op_type.value,
            "severity": sl.severity.value,
            "max_bit_bound": sl.max_bit_bound,
        })
        if prev_id is not None:
            edges.append({"source": prev_id, "target": sl.stage_id})
        prev_id = sl.stage_id

    return {"nodes": nodes, "edges": edges}


# ===================================================================
#  StreamingJSONWriter – write large reports chunk-by-chunk
# ===================================================================


class StreamingJSONWriter:
    """Incrementally write a JSON report to a text stream.

    This avoids building the entire JSON string in memory for very large
    reports (thousands of features / stages).  It writes each top-level
    section as it is produced.

    Usage::

        with open("report.json", "w") as fh:
            writer = StreamingJSONWriter(fh, indent=2)
            writer.write_report(report)
    """

    def __init__(
        self,
        stream: IO[str],
        *,
        indent: Optional[int] = 2,
        sort_keys: bool = True,
    ) -> None:
        self._stream = stream
        self._indent = indent
        self._sort_keys = sort_keys
        self._first_key = True

    def write_report(
        self,
        report: LeakageReport,
        config: Optional[TaintFlowConfig] = None,
    ) -> None:
        """Write the full report to the stream."""
        self._stream.write("{\n" if self._indent else "{")
        self._first_key = True

        self._write_section("metadata", self._build_metadata(report, config))
        self._write_section("summary", self._build_summary(report))
        self._write_array_section("features", _collect_features(report))
        self._write_array_section("stages", _collect_stages(report))
        self._write_array_section("bottlenecks", _collect_bottlenecks(report))
        self._write_array_section("remediations", _collect_remediations(report))
        self._write_section("dag", _build_dag_dict(report))

        self._stream.write("\n}\n" if self._indent else "}")

    def _write_section(self, key: str, value: Any) -> None:
        """Write a single key-value pair."""
        prefix = "" if self._first_key else ","
        if self._indent:
            prefix += "\n" if not self._first_key else ""
        self._first_key = False
        encoded = json.dumps(
            value,
            indent=self._indent,
            sort_keys=self._sort_keys,
            default=str,
        )
        if self._indent:
            # Indent the entire value block by one level
            indented = _indent_json(encoded, self._indent)
            self._stream.write(f'{prefix}  "{key}": {indented}')
        else:
            self._stream.write(f'{prefix}"{key}":{encoded}')

    def _write_array_section(self, key: str, items: List[Any]) -> None:
        """Write an array section, item by item for memory efficiency."""
        prefix = "" if self._first_key else ","
        if self._indent:
            prefix += "\n" if not self._first_key else ""
        self._first_key = False

        if self._indent:
            self._stream.write(f'{prefix}  "{key}": [\n')
        else:
            self._stream.write(f'{prefix}"{key}":[')

        for i, item in enumerate(items):
            encoded = json.dumps(
                item,
                indent=self._indent,
                sort_keys=self._sort_keys,
                default=str,
            )
            comma = "," if i < len(items) - 1 else ""
            if self._indent:
                indented = _indent_json(encoded, self._indent, level=2)
                self._stream.write(f"    {indented}{comma}\n")
            else:
                self._stream.write(f"{encoded}{comma}")

        if self._indent:
            self._stream.write("  ]")
        else:
            self._stream.write("]")

    @staticmethod
    def _build_metadata(
        report: LeakageReport,
        config: Optional[TaintFlowConfig],
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "tool": "taintflow",
            "version": _TOOL_VERSION,
            "schema_version": _SCHEMA_VERSION,
            "timestamp": report.timestamp,
        }
        if report.metadata:
            meta["extra"] = report.metadata
        if config is not None:
            meta["config"] = config.to_dict() if hasattr(config, "to_dict") else {}
        elif report.config_snapshot:
            meta["config"] = report.config_snapshot
        return meta

    @staticmethod
    def _build_summary(report: LeakageReport) -> Dict[str, Any]:
        return {
            "pipeline_name": report.pipeline_name,
            "total_leakage_bits": report.total_bit_bound,
            "severity": report.overall_severity.value,
            "n_features": report.n_features,
            "n_leaking_features": report.n_leaking_features,
            "n_stages": report.n_stages,
            "analysis_duration_ms": report.analysis_duration_ms,
            "is_clean": report.is_clean,
        }


def _indent_json(text: str, indent: int, level: int = 1) -> str:
    """Re-indent a pre-formatted JSON string by *level* extra indent levels."""
    prefix = " " * indent * level
    lines = text.split("\n")
    if len(lines) <= 1:
        return text
    result = [lines[0]]
    for line in lines[1:]:
        result.append(prefix + line)
    return "\n".join(result)


# ===================================================================
#  JSONReportGenerator
# ===================================================================


@dataclass
class JSONReportGenerator:
    """Generate structured JSON audit reports.

    Parameters
    ----------
    config : TaintFlowConfig, optional
        Audit configuration (embedded in ``metadata.config``).
    pretty : bool
        Use indented ("pretty-print") output.  ``False`` produces compact
        single-line JSON.
    indent : int
        Indentation width when *pretty* is ``True``.
    sort_keys : bool
        Sort dictionary keys for diff-friendly, deterministic output.
    top_n_bottlenecks : int
        How many bottleneck stages to include.
    include_schema : bool
        If ``True``, include a ``$schema`` key pointing to the report
        JSON Schema URL.
    """

    config: Optional[TaintFlowConfig] = None
    pretty: bool = True
    indent: int = 2
    sort_keys: bool = True
    top_n_bottlenecks: int = 10
    include_schema: bool = False

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------

    def generate(self, report: LeakageReport) -> str:
        """Return the JSON report as a string."""
        doc = self._build_document(report)
        indent = self.indent if self.pretty else None
        return json.dumps(
            doc,
            indent=indent,
            sort_keys=self.sort_keys,
            default=str,
            ensure_ascii=False,
        ) + "\n"

    def generate_compact(self, report: LeakageReport) -> str:
        """Return compact (single-line) JSON regardless of *pretty* setting."""
        doc = self._build_document(report)
        return json.dumps(
            doc,
            indent=None,
            sort_keys=self.sort_keys,
            default=str,
            ensure_ascii=False,
        ) + "\n"

    def generate_to_file(self, report: LeakageReport, path: str) -> None:
        """Write the JSON report to *path*."""
        content = self.generate(report)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    def generate_to_stream(self, report: LeakageReport, stream: IO[str]) -> None:
        """Write the JSON report to an open text stream."""
        stream.write(self.generate(report))

    def generate_streaming(
        self,
        report: LeakageReport,
        stream: IO[str],
    ) -> None:
        """Write the report using the streaming writer for large reports.

        This writes each section incrementally instead of materialising the
        entire JSON string in memory first.
        """
        writer = StreamingJSONWriter(
            stream,
            indent=self.indent if self.pretty else None,
            sort_keys=self.sort_keys,
        )
        writer.write_report(report, config=self.config)

    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON Schema that describes the report format."""
        return dict(REPORT_JSON_SCHEMA)

    def validate_against_schema(self, json_text: str) -> List[str]:
        """Best-effort validation of *json_text* against the report schema.

        Returns a list of human-readable error strings (empty if valid).
        This does **not** require ``jsonschema``; it performs lightweight
        structural checks only.
        """
        errors: List[str] = []
        try:
            doc = json.loads(json_text)
        except json.JSONDecodeError as exc:
            return [f"Invalid JSON: {exc}"]

        if not isinstance(doc, dict):
            return ["Root must be a JSON object"]

        required_top = {"metadata", "summary", "features", "stages"}
        missing = required_top - set(doc.keys())
        if missing:
            errors.append(f"Missing required top-level keys: {sorted(missing)}")

        meta = doc.get("metadata", {})
        if isinstance(meta, dict):
            for req in ("tool", "version", "timestamp"):
                if req not in meta:
                    errors.append(f"metadata missing required key: {req}")

        summary = doc.get("summary", {})
        if isinstance(summary, dict):
            for req in ("total_leakage_bits", "severity", "n_features", "n_stages"):
                if req not in summary:
                    errors.append(f"summary missing required key: {req}")

        features = doc.get("features")
        if features is not None and not isinstance(features, list):
            errors.append("'features' must be an array")

        stages = doc.get("stages")
        if stages is not None and not isinstance(stages, list):
            errors.append("'stages' must be an array")

        return errors

    # -----------------------------------------------------------------
    #  Internal document builder
    # -----------------------------------------------------------------

    def _build_document(self, report: LeakageReport) -> Dict[str, Any]:
        """Assemble the complete JSON document dict."""
        doc: Dict[str, Any] = {}

        if self.include_schema:
            doc["$schema"] = REPORT_JSON_SCHEMA["$id"]

        doc["metadata"] = self._build_metadata(report)
        doc["summary"] = self._build_summary(report)
        doc["features"] = _collect_features(report)
        doc["stages"] = _collect_stages(report)
        doc["bottlenecks"] = _collect_bottlenecks(report, self.top_n_bottlenecks)
        doc["remediations"] = _collect_remediations(report)
        doc["dag"] = _build_dag_dict(report)
        return doc

    def _build_metadata(self, report: LeakageReport) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "tool": "taintflow",
            "version": _TOOL_VERSION,
            "schema_version": _SCHEMA_VERSION,
            "timestamp": report.timestamp,
        }
        if report.metadata:
            meta["extra"] = report.metadata
        if self.config is not None:
            meta["config"] = (
                self.config.to_dict() if hasattr(self.config, "to_dict") else {}
            )
        elif report.config_snapshot:
            meta["config"] = report.config_snapshot
        return meta

    @staticmethod
    def _build_summary(report: LeakageReport) -> Dict[str, Any]:
        return {
            "pipeline_name": report.pipeline_name,
            "total_leakage_bits": report.total_bit_bound,
            "severity": report.overall_severity.value,
            "n_features": report.n_features,
            "n_leaking_features": report.n_leaking_features,
            "n_stages": report.n_stages,
            "analysis_duration_ms": report.analysis_duration_ms,
            "is_clean": report.is_clean,
        }


# ===================================================================
#  Convenience function
# ===================================================================


def generate_json_report(
    report: LeakageReport,
    config: Optional[TaintFlowConfig] = None,
    path: Optional[str] = None,
    pretty: bool = True,
    **kwargs: Any,
) -> str:
    """One-shot helper: build a JSON report and optionally write to *path*.

    Parameters
    ----------
    report : LeakageReport
        The audit result to serialise.
    config : TaintFlowConfig, optional
        Audit configuration for metadata embedding.
    path : str, optional
        If given, write the report to this file path.
    pretty : bool
        Pretty-print (indented) output.
    **kwargs
        Forwarded to :class:`JSONReportGenerator`.

    Returns
    -------
    str
        The JSON document string.
    """
    gen = JSONReportGenerator(config=config, pretty=pretty, **kwargs)
    content = gen.generate(report)
    if path is not None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    return content
