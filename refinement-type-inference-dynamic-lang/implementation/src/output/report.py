"""
Report Generator for Refinement Type Analysis.

Generates structured reports with:
- Bug location with source context
- Refinement type at each program point
- Guard suggestions for unfixed bugs
- Confidence levels (high/medium/low)
- LSP-compatible diagnostics
- VS Code extension-friendly JSON
- SARIF output for CI integration
"""

from __future__ import annotations

import html
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 1. Data Models
# ---------------------------------------------------------------------------

class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def numeric(self) -> float:
        return {Confidence.HIGH: 0.95, Confidence.MEDIUM: 0.70, Confidence.LOW: 0.40}[self]


class BugCategory(Enum):
    NULL_DEREF = "null-dereference"
    TYPE_ERROR = "type-error"
    DIV_BY_ZERO = "division-by-zero"
    INDEX_OOB = "index-out-of-bounds"
    KEY_ERROR = "key-error"
    ATTR_ERROR = "attribute-error"
    RESOURCE_LEAK = "resource-leak"
    INJECTION = "injection"
    UNREACHABLE = "unreachable-code"
    REDUNDANT_GUARD = "redundant-guard"
    UNCHECKED_CAST = "unchecked-cast"

    @property
    def default_severity(self) -> Severity:
        _high = {self.NULL_DEREF, self.INJECTION, self.DIV_BY_ZERO, self.RESOURCE_LEAK}
        _warn = {self.TYPE_ERROR, self.INDEX_OOB, self.KEY_ERROR, self.ATTR_ERROR, self.UNCHECKED_CAST}
        if self in _high:
            return Severity.ERROR
        if self in _warn:
            return Severity.WARNING
        return Severity.INFO

    @property
    def rule_id(self) -> str:
        return f"refinement/{self.value}"


@dataclass
class SourceLocation:
    file: str
    line: int
    col: int
    end_line: int = -1
    end_col: int = -1

    def __post_init__(self) -> None:
        if self.end_line < 0:
            self.end_line = self.line
        if self.end_col < 0:
            self.end_col = self.col

    def as_range(self) -> Dict[str, Dict[str, int]]:
        return {
            "start": {"line": self.line, "character": self.col},
            "end": {"line": self.end_line, "character": self.end_col},
        }


@dataclass
class SourceContext:
    lines: List[str]
    start_line: int
    highlight_start: int
    highlight_end: int

    def render_text(self, gutter_width: int = 5) -> str:
        parts: List[str] = []
        for i, text in enumerate(self.lines):
            lineno = self.start_line + i
            prefix = ">>>" if self.highlight_start <= lineno <= self.highlight_end else "   "
            parts.append(f"{prefix} {lineno:>{gutter_width}} | {text}")
        return "\n".join(parts)


@dataclass
class RefinementAnnotation:
    location: SourceLocation
    variable: str
    refined_type: str
    base_type: str = ""
    constraint: str = ""

    def display(self) -> str:
        if self.constraint:
            return f"{self.variable}: {{{self.base_type} | {self.constraint}}}"
        return f"{self.variable}: {self.refined_type}"


@dataclass
class GuardSuggestion:
    guard_code: str
    insert_location: SourceLocation
    description: str = ""
    removes_category: Optional[BugCategory] = None

    def as_edit(self) -> Dict[str, Any]:
        return {
            "range": self.insert_location.as_range(),
            "newText": self.guard_code,
        }


@dataclass
class Finding:
    category: BugCategory
    severity: Severity
    confidence: Confidence
    location: SourceLocation
    message: str
    context: Optional[SourceContext] = None
    suggestions: List[GuardSuggestion] = field(default_factory=list)
    refinements: List[RefinementAnnotation] = field(default_factory=list)
    related: List[SourceLocation] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return (
            f"{self.category.value}:{self.location.file}:"
            f"{self.location.line}:{self.location.col}:{self.message}"
        )


@dataclass
class AnalysisReport:
    findings: List[Finding] = field(default_factory=list)
    annotations: List[RefinementAnnotation] = field(default_factory=list)
    files_analysed: int = 0
    analysis_duration_ms: float = 0.0
    tool_version: str = "0.1.0"
    config: Dict[str, Any] = field(default_factory=dict)

    # -- summary helpers --
    def by_severity(self) -> Dict[Severity, List[Finding]]:
        result: Dict[Severity, List[Finding]] = {s: [] for s in Severity}
        for f in self.findings:
            result[f.severity].append(f)
        return result

    def by_category(self) -> Dict[BugCategory, List[Finding]]:
        result: Dict[BugCategory, List[Finding]] = {}
        for f in self.findings:
            result.setdefault(f.category, []).append(f)
        return result

    def by_confidence(self) -> Dict[Confidence, int]:
        counts: Dict[Confidence, int] = {c: 0 for c in Confidence}
        for f in self.findings:
            counts[f.confidence] += 1
        return counts

    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)


# ---------------------------------------------------------------------------
# 2. SourceContextExtractor
# ---------------------------------------------------------------------------

class SourceContextExtractor:
    """Read source files and extract context around bug locations."""

    def __init__(self, context_lines: int = 3) -> None:
        self._context_lines = context_lines
        self._cache: Dict[str, List[str]] = {}

    def _read_file(self, path: str) -> List[str]:
        if path in self._cache:
            return self._cache[path]
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
        except (OSError, IOError):
            lines = []
        self._cache[path] = lines
        return lines

    def extract(self, loc: SourceLocation) -> Optional[SourceContext]:
        lines = self._read_file(loc.file)
        if not lines:
            return None
        total = len(lines)
        start = max(0, loc.line - 1 - self._context_lines)
        end = min(total, loc.end_line + self._context_lines)
        snippet = lines[start:end]
        return SourceContext(
            lines=snippet,
            start_line=start + 1,
            highlight_start=loc.line,
            highlight_end=loc.end_line,
        )

    def get_line_text(self, path: str, lineno: int) -> str:
        lines = self._read_file(path)
        if 0 < lineno <= len(lines):
            return lines[lineno - 1]
        return ""

    def annotate_finding(self, finding: Finding) -> Finding:
        if finding.context is None:
            finding.context = self.extract(finding.location)
        return finding

    def annotate_report(self, report: AnalysisReport) -> AnalysisReport:
        for finding in report.findings:
            self.annotate_finding(finding)
        return report

    def clear_cache(self) -> None:
        self._cache.clear()

    def preload(self, paths: Sequence[str]) -> None:
        for p in paths:
            self._read_file(p)

    def file_line_count(self, path: str) -> int:
        return len(self._read_file(path))


# ---------------------------------------------------------------------------
# 3. LSPDiagnosticGenerator
# ---------------------------------------------------------------------------

class DiagnosticSeverity(IntEnum):
    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


_SEVERITY_MAP = {
    Severity.ERROR: DiagnosticSeverity.ERROR,
    Severity.WARNING: DiagnosticSeverity.WARNING,
    Severity.INFO: DiagnosticSeverity.INFORMATION,
    Severity.HINT: DiagnosticSeverity.HINT,
}


class DiagnosticTag(IntEnum):
    UNNECESSARY = 1
    DEPRECATED = 2


@dataclass
class DiagnosticRelatedInfo:
    location: SourceLocation
    message: str

    def to_lsp(self) -> Dict[str, Any]:
        return {
            "location": {
                "uri": f"file://{self.location.file}",
                "range": self.location.as_range(),
            },
            "message": self.message,
        }


@dataclass
class Diagnostic:
    range: Dict[str, Dict[str, int]]
    message: str
    severity: DiagnosticSeverity
    source: str = "refinement-types"
    code: str = ""
    tags: List[int] = field(default_factory=list)
    related_information: List[DiagnosticRelatedInfo] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_lsp(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "range": self.range,
            "message": self.message,
            "severity": int(self.severity),
            "source": self.source,
        }
        if self.code:
            obj["code"] = self.code
        if self.tags:
            obj["tags"] = self.tags
        if self.related_information:
            obj["relatedInformation"] = [r.to_lsp() for r in self.related_information]
        if self.data:
            obj["data"] = self.data
        return obj


class LSPDiagnosticGenerator:
    """Generates LSP-compatible diagnostics from an AnalysisReport."""

    def __init__(self, source_name: str = "refinement-types") -> None:
        self._source = source_name

    def _tags_for(self, finding: Finding) -> List[int]:
        tags: List[int] = []
        if finding.category == BugCategory.UNREACHABLE:
            tags.append(int(DiagnosticTag.UNNECESSARY))
        if "deprecated" in finding.tags:
            tags.append(int(DiagnosticTag.DEPRECATED))
        return tags

    def _related(self, finding: Finding) -> List[DiagnosticRelatedInfo]:
        infos: List[DiagnosticRelatedInfo] = []
        for loc in finding.related:
            infos.append(DiagnosticRelatedInfo(location=loc, message="related location"))
        for ref in finding.refinements:
            infos.append(
                DiagnosticRelatedInfo(
                    location=ref.location,
                    message=f"Refined type: {ref.display()}",
                )
            )
        return infos

    def _code_actions(self, finding: Finding) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for sug in finding.suggestions:
            actions.append({
                "title": sug.description or f"Add guard: {sug.guard_code.strip()}",
                "kind": "quickfix",
                "edit": {
                    "changes": {
                        f"file://{sug.insert_location.file}": [sug.as_edit()],
                    }
                },
            })
        return actions

    def finding_to_diagnostic(self, finding: Finding) -> Diagnostic:
        lsp_sev = _SEVERITY_MAP.get(finding.severity, DiagnosticSeverity.WARNING)
        diag = Diagnostic(
            range=finding.location.as_range(),
            message=finding.message,
            severity=lsp_sev,
            source=self._source,
            code=finding.category.rule_id,
            tags=self._tags_for(finding),
            related_information=self._related(finding),
        )
        actions = self._code_actions(finding)
        if actions:
            diag.data["codeActions"] = actions
        return diag

    def generate_diagnostics(self, report: AnalysisReport) -> List[Diagnostic]:
        return [self.finding_to_diagnostic(f) for f in report.findings]

    def to_lsp_json(self, diagnostics: List[Diagnostic]) -> str:
        return json.dumps([d.to_lsp() for d in diagnostics], indent=2)

    def group_by_file(self, diagnostics: List[Diagnostic], report: AnalysisReport) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = {}
        for finding, diag in zip(report.findings, diagnostics):
            uri = f"file://{finding.location.file}"
            grouped.setdefault(uri, []).append(diag.to_lsp())
        return grouped


# ---------------------------------------------------------------------------
# 4. SARIFGenerator
# ---------------------------------------------------------------------------

class SARIFGenerator:
    """Generates SARIF 2.1.0 compliant output."""

    SARIF_VERSION = "2.1.0"
    SCHEMA_URI = "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json"

    _SARIF_LEVEL = {
        Severity.ERROR: "error",
        Severity.WARNING: "warning",
        Severity.INFO: "note",
        Severity.HINT: "note",
    }

    def __init__(self, tool_name: str = "refinement-type-analyser") -> None:
        self._tool_name = tool_name

    # -- rules --
    def _build_rules(self, report: AnalysisReport) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for f in report.findings:
            rid = f.category.rule_id
            if rid not in seen:
                seen[rid] = {
                    "id": rid,
                    "name": f.category.name,
                    "shortDescription": {"text": f.category.value.replace("-", " ").title()},
                    "defaultConfiguration": {
                        "level": self._SARIF_LEVEL[f.category.default_severity],
                    },
                    "helpUri": f"https://refinement-types.dev/rules/{f.category.value}",
                }
        return list(seen.values())

    # -- locations --
    @staticmethod
    def _physical_location(loc: SourceLocation) -> Dict[str, Any]:
        return {
            "physicalLocation": {
                "artifactLocation": {"uri": loc.file, "uriBaseId": "%SRCROOT%"},
                "region": {
                    "startLine": loc.line,
                    "startColumn": loc.col + 1,
                    "endLine": loc.end_line,
                    "endColumn": loc.end_col + 1,
                },
            }
        }

    # -- code flows --
    def _code_flows(self, finding: Finding) -> List[Dict[str, Any]]:
        if not finding.related:
            return []
        steps: List[Dict[str, Any]] = []
        for i, loc in enumerate(finding.related):
            steps.append({
                "location": self._physical_location(loc),
                "nestingLevel": 0,
                "importance": "essential" if i == 0 else "important",
            })
        steps.append({
            "location": self._physical_location(finding.location),
            "nestingLevel": 0,
            "importance": "essential",
        })
        return [{"threadFlows": [{"locations": steps}]}]

    # -- fixes --
    @staticmethod
    def _fixes(finding: Finding) -> List[Dict[str, Any]]:
        fixes: List[Dict[str, Any]] = []
        for sug in finding.suggestions:
            fixes.append({
                "description": {"text": sug.description or "Add guard"},
                "artifactChanges": [{
                    "artifactLocation": {
                        "uri": sug.insert_location.file,
                        "uriBaseId": "%SRCROOT%",
                    },
                    "replacements": [{
                        "deletedRegion": {
                            "startLine": sug.insert_location.line,
                            "startColumn": sug.insert_location.col + 1,
                            "endLine": sug.insert_location.end_line,
                            "endColumn": sug.insert_location.end_col + 1,
                        },
                        "insertedContent": {"text": sug.guard_code},
                    }],
                }],
            })
        return fixes

    def _result(self, finding: Finding, rule_index: Dict[str, int]) -> Dict[str, Any]:
        rid = finding.category.rule_id
        res: Dict[str, Any] = {
            "ruleId": rid,
            "ruleIndex": rule_index.get(rid, 0),
            "level": self._SARIF_LEVEL.get(finding.severity, "warning"),
            "message": {"text": finding.message},
            "locations": [self._physical_location(finding.location)],
        }
        res["partialFingerprints"] = {"primaryLocationLineHash": finding.fingerprint}
        flows = self._code_flows(finding)
        if flows:
            res["codeFlows"] = flows
        fixes = self._fixes(finding)
        if fixes:
            res["fixes"] = fixes
        if finding.refinements:
            res["properties"] = {
                "refinements": [r.display() for r in finding.refinements],
                "confidence": finding.confidence.value,
            }
        return res

    def generate_sarif(self, report: AnalysisReport) -> Dict[str, Any]:
        rules = self._build_rules(report)
        rule_index = {r["id"]: i for i, r in enumerate(rules)}
        results = [self._result(f, rule_index) for f in report.findings]
        return {
            "$schema": self.SCHEMA_URI,
            "version": self.SARIF_VERSION,
            "runs": [{
                "tool": {
                    "driver": {
                        "name": self._tool_name,
                        "version": report.tool_version,
                        "informationUri": "https://refinement-types.dev",
                        "rules": rules,
                    }
                },
                "results": results,
                "invocations": [{
                    "executionSuccessful": True,
                    "toolExecutionNotifications": [],
                }],
            }],
        }

    def to_sarif_json(self, report: AnalysisReport) -> str:
        return json.dumps(self.generate_sarif(report), indent=2)


# ---------------------------------------------------------------------------
# 5. VSCodeReportGenerator
# ---------------------------------------------------------------------------

class VSCodeReportGenerator:
    """Generate VS Code extension-friendly output."""

    def __init__(self) -> None:
        self._lsp_gen = LSPDiagnosticGenerator()

    # -- problem matcher format (line-based for terminal output) --
    def problem_matcher_lines(self, report: AnalysisReport) -> str:
        lines: List[str] = []
        for f in report.findings:
            sev = f.severity.value.upper()
            lines.append(
                f"{f.location.file}:{f.location.line}:{f.location.col + 1}: "
                f"{sev}: [{f.category.value}] {f.message}"
            )
        return "\n".join(lines)

    # -- inline decorations --
    def inline_decorations(self, report: AnalysisReport) -> List[Dict[str, Any]]:
        decorations: List[Dict[str, Any]] = []
        for ann in report.annotations:
            decorations.append({
                "range": ann.location.as_range(),
                "renderOptions": {
                    "after": {
                        "contentText": f"  // {ann.display()}",
                        "color": "#888888",
                        "fontStyle": "italic",
                    }
                },
            })
        return decorations

    # -- code lens data --
    def code_lens_items(self, report: AnalysisReport) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for ann in report.annotations:
            items.append({
                "range": ann.location.as_range(),
                "command": {
                    "title": ann.display(),
                    "command": "refinementTypes.showDetail",
                    "arguments": [ann.variable, ann.refined_type],
                },
            })
        return items

    # -- quick fix actions --
    def quick_fixes(self, report: AnalysisReport) -> List[Dict[str, Any]]:
        fixes: List[Dict[str, Any]] = []
        for finding in report.findings:
            for sug in finding.suggestions:
                fixes.append({
                    "diagnostic_range": finding.location.as_range(),
                    "title": sug.description or f"Add guard: {sug.guard_code.strip()}",
                    "kind": "quickfix",
                    "edit": sug.as_edit(),
                    "file": finding.location.file,
                })
        return fixes

    def to_json(self, report: AnalysisReport) -> str:
        diags = self._lsp_gen.generate_diagnostics(report)
        payload: Dict[str, Any] = {
            "diagnostics": self._lsp_gen.group_by_file(diags, report),
            "decorations": self.inline_decorations(report),
            "codeLens": self.code_lens_items(report),
            "quickFixes": self.quick_fixes(report),
            "summary": {
                "totalFindings": len(report.findings),
                "errors": report.error_count(),
                "filesAnalysed": report.files_analysed,
                "durationMs": report.analysis_duration_ms,
            },
        }
        return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# 6. HTMLReportRenderer
# ---------------------------------------------------------------------------

_CSS = """\
body{font-family:system-ui,sans-serif;margin:2rem;color:#222;background:#fafafa}
h1{border-bottom:2px solid #333}
.summary{display:flex;gap:1.5rem;margin:1rem 0}
.card{padding:1rem;border-radius:8px;min-width:120px;text-align:center;color:#fff}
.card .num{font-size:2rem;font-weight:700}
.card.error{background:#d32f2f}.card.warning{background:#f9a825;color:#222}
.card.info{background:#1976d2}.card.hint{background:#388e3c}
table{border-collapse:collapse;width:100%;margin-top:1rem}
th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:left}
th{background:#eee}
pre{background:#263238;color:#eee;padding:1rem;border-radius:4px;overflow-x:auto}
.hi{background:#ff980055}
.conf-high{color:#2e7d32}.conf-med{color:#f57f17}.conf-low{color:#c62828}
"""


class HTMLReportRenderer:
    """Render an AnalysisReport as a standalone HTML page."""

    def __init__(self, title: str = "Refinement Type Analysis Report") -> None:
        self._title = title

    def _severity_card(self, sev: Severity, count: int) -> str:
        return (
            f'<div class="card {sev.value}">'
            f'<div class="num">{count}</div><div>{sev.value.title()}</div></div>'
        )

    def _render_context(self, ctx: Optional[SourceContext]) -> str:
        if ctx is None:
            return ""
        escaped: List[str] = []
        for i, line in enumerate(ctx.lines):
            lineno = ctx.start_line + i
            esc = html.escape(line)
            if ctx.highlight_start <= lineno <= ctx.highlight_end:
                esc = f'<span class="hi">{esc}</span>'
            escaped.append(f"{lineno:>5} | {esc}")
        return "<pre>" + "\n".join(escaped) + "</pre>"

    def _conf_class(self, c: Confidence) -> str:
        return {Confidence.HIGH: "conf-high", Confidence.MEDIUM: "conf-med", Confidence.LOW: "conf-low"}[c]

    def _findings_table(self, findings: List[Finding]) -> str:
        rows: List[str] = ["<table><tr><th>#</th><th>File</th><th>Line</th>"
                           "<th>Category</th><th>Severity</th><th>Confidence</th><th>Message</th></tr>"]
        for i, f in enumerate(findings, 1):
            cc = self._conf_class(f.confidence)
            rows.append(
                f"<tr><td>{i}</td>"
                f"<td>{html.escape(os.path.basename(f.location.file))}</td>"
                f"<td>{f.location.line}</td>"
                f"<td>{html.escape(f.category.value)}</td>"
                f"<td>{f.severity.value}</td>"
                f'<td class="{cc}">{f.confidence.value}</td>'
                f"<td>{html.escape(f.message)}</td></tr>"
            )
        rows.append("</table>")
        return "\n".join(rows)

    def _detail_sections(self, findings: List[Finding]) -> str:
        parts: List[str] = []
        for i, f in enumerate(findings, 1):
            parts.append(f"<h3>#{i} — {html.escape(f.category.value)}</h3>")
            parts.append(f"<p><strong>{html.escape(f.message)}</strong></p>")
            parts.append(f"<p>File: {html.escape(f.location.file)} "
                         f"Line {f.location.line}</p>")
            parts.append(self._render_context(f.context))
            if f.refinements:
                parts.append("<p>Refinements:</p><ul>")
                for r in f.refinements:
                    parts.append(f"<li>{html.escape(r.display())}</li>")
                parts.append("</ul>")
            if f.suggestions:
                parts.append("<p>Suggested fixes:</p><ul>")
                for s in f.suggestions:
                    parts.append(f"<li><code>{html.escape(s.guard_code)}</code></li>")
                parts.append("</ul>")
        return "\n".join(parts)

    def _chart_data_json(self, report: AnalysisReport) -> str:
        sev_counts = {s.value: len(fs) for s, fs in report.by_severity().items()}
        cat_counts = {c.value: len(fs) for c, fs in report.by_category().items()}
        conf_counts = {c.value: n for c, n in report.by_confidence().items()}
        return json.dumps({"severity": sev_counts, "category": cat_counts, "confidence": conf_counts})

    def render(self, report: AnalysisReport) -> str:
        sev_map = report.by_severity()
        cards = "".join(self._severity_card(s, len(sev_map[s])) for s in Severity)
        body = (
            f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(self._title)}</title>"
            f"<style>{_CSS}</style></head><body>"
            f"<h1>{html.escape(self._title)}</h1>"
            f'<div class="summary">{cards}</div>'
            f"<p>Files analysed: {report.files_analysed} &mdash; "
            f"Duration: {report.analysis_duration_ms:.1f} ms</p>"
            f"<h2>Findings</h2>"
            f"{self._findings_table(report.findings)}"
            f"<h2>Details</h2>"
            f"{self._detail_sections(report.findings)}"
            f"<script>window.__chartData = {self._chart_data_json(report)};</script>"
            f"</body></html>"
        )
        return body


# ---------------------------------------------------------------------------
# 7. ReportAggregator
# ---------------------------------------------------------------------------

class ReportAggregator:
    """Aggregate multiple analysis reports and detect regressions."""

    def merge(self, reports: Sequence[AnalysisReport]) -> AnalysisReport:
        merged = AnalysisReport()
        seen_fingerprints: set[str] = set()
        for r in reports:
            merged.files_analysed += r.files_analysed
            merged.analysis_duration_ms += r.analysis_duration_ms
            for f in r.findings:
                fp = f.fingerprint
                if fp not in seen_fingerprints:
                    seen_fingerprints.add(fp)
                    merged.findings.append(f)
            merged.annotations.extend(r.annotations)
        if reports:
            merged.tool_version = reports[0].tool_version
        return merged

    def diff(
        self, current: AnalysisReport, baseline: AnalysisReport
    ) -> Tuple[List[Finding], List[Finding]]:
        """Return (new_findings, fixed_findings) compared to baseline."""
        cur_fps = {f.fingerprint for f in current.findings}
        base_fps = {f.fingerprint: f for f in baseline.findings}
        new = [f for f in current.findings if f.fingerprint not in base_fps]
        fixed = [f for fp, f in base_fps.items() if fp not in cur_fps]
        return new, fixed

    def regression_detected(
        self, current: AnalysisReport, baseline: AnalysisReport
    ) -> bool:
        new, _ = self.diff(current, baseline)
        return any(f.severity == Severity.ERROR for f in new)

    def trend(self, history: Sequence[AnalysisReport]) -> Dict[str, Any]:
        if not history:
            return {"direction": "stable", "counts": []}
        counts = [len(r.findings) for r in history]
        if len(counts) < 2:
            direction = "stable"
        elif counts[-1] < counts[-2]:
            direction = "improving"
        elif counts[-1] > counts[-2]:
            direction = "degrading"
        else:
            direction = "stable"
        error_counts = [r.error_count() for r in history]
        return {
            "direction": direction,
            "counts": counts,
            "error_counts": error_counts,
            "latest": counts[-1] if counts else 0,
            "delta": (counts[-1] - counts[-2]) if len(counts) >= 2 else 0,
        }

    def summary_text(self, report: AnalysisReport) -> str:
        sev = report.by_severity()
        parts = [f"{len(sev[s])} {s.value}(s)" for s in Severity if sev[s]]
        total = len(report.findings)
        return (
            f"Analysis complete: {total} finding(s) across "
            f"{report.files_analysed} file(s) in "
            f"{report.analysis_duration_ms:.0f} ms — "
            + ", ".join(parts)
        )
