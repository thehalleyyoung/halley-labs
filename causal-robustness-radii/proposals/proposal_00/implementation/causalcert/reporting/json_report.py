"""
JSON report format.

Serialises an :class:`AuditReport` to a structured JSON document with
full provenance metadata.  The schema is designed to be diff-friendly
(sorted keys, consistent indentation) and self-documenting.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

from causalcert.types import (
    AuditReport,
    EstimationResult,
    FragilityChannel,
    FragilityScore,
    RobustnessRadius,
    StructuralEdit,
)


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = "1.0.0"
_SCHEMA_URL = "https://causalcert.readthedocs.io/schema/audit-report-v1.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_json_report(
    report: AuditReport,
    node_names: list[str] | None = None,
    indent: int = 2,
) -> str:
    """Serialise an audit report to a JSON string.

    Parameters
    ----------
    report : AuditReport
        Audit report.
    node_names : list[str] | None
        Optional node names.
    indent : int
        JSON indentation level.

    Returns
    -------
    str
        Pretty-printed JSON string.
    """
    d = to_json_dict(report, node_names=node_names)
    return json.dumps(d, indent=indent, sort_keys=False, default=_json_default)


def to_json_dict(
    report: AuditReport,
    node_names: list[str] | None = None,
) -> dict[str, Any]:
    """Convert an audit report to a JSON-compatible dict.

    Parameters
    ----------
    report : AuditReport
    node_names : list[str] | None
        Optional node names.

    Returns
    -------
    dict[str, Any]
    """
    return {
        "$schema": _SCHEMA_URL,
        "schema_version": _SCHEMA_VERSION,
        "metadata": _build_metadata(report),
        "query": _build_query(report, node_names),
        "dag": _build_dag_info(report),
        "robustness_radius": _build_radius(report),
        "fragility": _build_fragility(report, node_names),
        "estimation": _build_estimation(report),
        "ci_tests": _build_ci_tests(report),
        "limitations": _build_limitations(report),
    }


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_metadata(report: AuditReport) -> dict[str, Any]:
    """Build the metadata section."""
    meta = dict(report.metadata)
    meta.setdefault("generated_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
    meta.setdefault("generator", "CausalCert")
    meta.setdefault("version", "0.1.0")
    meta.setdefault("schema_version", _SCHEMA_VERSION)
    return meta


def _build_query(
    report: AuditReport,
    node_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build the query section."""
    q: dict[str, Any] = {
        "treatment": report.treatment,
        "outcome": report.outcome,
    }
    if node_names:
        if report.treatment < len(node_names):
            q["treatment_name"] = node_names[report.treatment]
        if report.outcome < len(node_names):
            q["outcome_name"] = node_names[report.outcome]
    return q


def _build_dag_info(report: AuditReport) -> dict[str, Any]:
    """Build the DAG info section."""
    return {
        "n_nodes": report.n_nodes,
        "n_edges": report.n_edges,
    }


def _build_radius(report: AuditReport) -> dict[str, Any]:
    """Build the robustness radius section."""
    r = report.radius
    return {
        "lower_bound": r.lower_bound,
        "upper_bound": r.upper_bound,
        "certified": r.certified,
        "gap": r.gap,
        "solver_strategy": r.solver_strategy.value,
        "solver_time_s": r.solver_time_s,
        "witness_edits": [
            _serialize_edit(e) for e in r.witness_edits
        ],
    }


def _build_fragility(
    report: AuditReport,
    node_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build the fragility section."""
    ranking = []
    for i, fs in enumerate(report.fragility_ranking):
        entry = _serialize_fragility_score(fs, i + 1, node_names)
        ranking.append(entry)

    n_critical = sum(1 for s in report.fragility_ranking if s.total_score >= 0.7)
    n_important = sum(1 for s in report.fragility_ranking if 0.4 <= s.total_score < 0.7)
    n_moderate = sum(1 for s in report.fragility_ranking if 0.1 <= s.total_score < 0.4)
    n_cosmetic = sum(1 for s in report.fragility_ranking if s.total_score < 0.1)

    return {
        "n_scored": len(report.fragility_ranking),
        "severity_counts": {
            "critical": n_critical,
            "important": n_important,
            "moderate": n_moderate,
            "cosmetic": n_cosmetic,
        },
        "ranking": ranking,
    }


def _build_estimation(report: AuditReport) -> dict[str, Any]:
    """Build the estimation section."""
    result: dict[str, Any] = {
        "baseline": None,
        "perturbed": [],
    }

    if report.baseline_estimate is not None:
        result["baseline"] = _serialize_estimation(report.baseline_estimate)

    for pe in report.perturbed_estimates:
        result["perturbed"].append(_serialize_estimation(pe))

    return result


def _build_ci_tests(report: AuditReport) -> dict[str, Any]:
    """Build the CI tests section."""
    tests = []
    for ci in report.ci_results:
        tests.append({
            "x": ci.x,
            "y": ci.y,
            "conditioning_set": sorted(ci.conditioning_set),
            "statistic": ci.statistic,
            "p_value": ci.p_value,
            "method": ci.method.value,
            "reject": ci.reject,
            "alpha": ci.alpha,
        })

    return {
        "n_tests": len(tests),
        "tests": tests,
    }


def _build_limitations(report: AuditReport) -> dict[str, Any]:
    """Build the limitations section."""
    from causalcert.reporting.audit import build_limitations_section
    return build_limitations_section(report)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _serialize_edit(edit: StructuralEdit) -> dict[str, Any]:
    """Serialise a StructuralEdit to dict."""
    return {
        "edit_type": edit.edit_type.value,
        "source": edit.source,
        "target": edit.target,
    }


def _serialize_fragility_score(
    fs: FragilityScore,
    rank: int = 0,
    node_names: list[str] | None = None,
) -> dict[str, Any]:
    """Serialise a FragilityScore to dict."""
    src, tgt = fs.edge
    entry: dict[str, Any] = {
        "rank": rank,
        "edge": [src, tgt],
        "total_score": round(fs.total_score, 6),
        "channel_scores": {
            ch.value: round(v, 6) for ch, v in fs.channel_scores.items()
        },
    }

    if node_names:
        src_name = node_names[src] if src < len(node_names) else str(src)
        tgt_name = node_names[tgt] if tgt < len(node_names) else str(tgt)
        entry["edge_label"] = f"{src_name} → {tgt_name}"

    if fs.witness_ci is not None:
        entry["witness_ci"] = {
            "x": fs.witness_ci.x,
            "y": fs.witness_ci.y,
            "conditioning_set": sorted(fs.witness_ci.conditioning_set),
            "p_value": fs.witness_ci.p_value,
        }

    return entry


def _serialize_estimation(est: EstimationResult) -> dict[str, Any]:
    """Serialise an EstimationResult to dict."""
    return {
        "ate": est.ate,
        "se": est.se,
        "ci_lower": est.ci_lower,
        "ci_upper": est.ci_upper,
        "adjustment_set": sorted(est.adjustment_set),
        "method": est.method,
        "n_obs": est.n_obs,
    }


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for non-standard types."""
    if hasattr(obj, "__iter__"):
        return list(obj)
    if hasattr(obj, "value"):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------


def get_schema() -> dict[str, Any]:
    """Return the JSON schema for the audit report format.

    Returns
    -------
    dict
        JSON Schema object.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": _SCHEMA_URL,
        "title": "CausalCert Audit Report",
        "type": "object",
        "required": [
            "schema_version", "metadata", "query", "dag",
            "robustness_radius", "fragility",
        ],
        "properties": {
            "schema_version": {"type": "string"},
            "metadata": {
                "type": "object",
                "properties": {
                    "generated_at": {"type": "string", "format": "date-time"},
                    "generator": {"type": "string"},
                    "version": {"type": "string"},
                },
            },
            "query": {
                "type": "object",
                "required": ["treatment", "outcome"],
                "properties": {
                    "treatment": {"type": "integer"},
                    "outcome": {"type": "integer"},
                    "treatment_name": {"type": "string"},
                    "outcome_name": {"type": "string"},
                },
            },
            "dag": {
                "type": "object",
                "properties": {
                    "n_nodes": {"type": "integer"},
                    "n_edges": {"type": "integer"},
                },
            },
            "robustness_radius": {
                "type": "object",
                "required": ["lower_bound", "upper_bound", "certified"],
                "properties": {
                    "lower_bound": {"type": "integer"},
                    "upper_bound": {"type": "integer"},
                    "certified": {"type": "boolean"},
                    "gap": {"type": "number"},
                    "solver_strategy": {"type": "string"},
                    "solver_time_s": {"type": "number"},
                    "witness_edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "edit_type": {"type": "string"},
                                "source": {"type": "integer"},
                                "target": {"type": "integer"},
                            },
                        },
                    },
                },
            },
            "fragility": {
                "type": "object",
                "properties": {
                    "n_scored": {"type": "integer"},
                    "ranking": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "rank": {"type": "integer"},
                                "edge": {"type": "array", "items": {"type": "integer"}},
                                "total_score": {"type": "number"},
                                "channel_scores": {"type": "object"},
                            },
                        },
                    },
                },
            },
        },
    }
