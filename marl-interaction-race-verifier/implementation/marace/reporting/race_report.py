"""Reporting infrastructure for the MARACE race-condition verifier.

Provides dataclasses that capture every detected race together with
contextual metadata, formatters that serialise reports into plain-text,
JSON, HTML and LaTeX, and helpers that summarise, compare, and diff
race catalogs across configurations.
"""

from __future__ import annotations

import json
import html as _html
import re
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

if TYPE_CHECKING:
    import numpy as np  # noqa: F401


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RaceType(Enum):
    """Taxonomy of multi-agent race conditions."""

    COLLISION = auto()
    DEADLOCK = auto()
    LIVELOCK = auto()
    CORRELATED_FAILURE = auto()
    RESOURCE_CONFLICT = auto()


class Severity(Enum):
    """Impact severity of a detected race."""

    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    INFO = auto()


class SectionType(Enum):
    """Logical section of a generated report."""

    SUMMARY = auto()
    DETAILS = auto()
    RECOMMENDATIONS = auto()
    STATISTICS = auto()


# ---------------------------------------------------------------------------
# Placeholder type aliases for modules that are still stubs
# ---------------------------------------------------------------------------

# ``RaceCatalog`` will eventually come from the ``race`` sub-package.
# Until that module is fleshed out we reference it by name only.
RaceCatalog = Any
AbstractStateRegion = Any
Schedule = Any


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DetailedRaceReport:
    """Per-race detailed description with surrounding context.

    Each instance corresponds to exactly one race detected during
    verification and carries enough information for a developer to
    reproduce and reason about the issue.
    """

    race_id: str
    race_type: RaceType
    agents_involved: List[str]
    probability_bound: float
    triggering_schedule: Optional[Schedule] = None
    abstract_state_region: Optional[AbstractStateRegion] = None
    description: str = ""
    severity: Severity = Severity.MEDIUM
    recommendation: str = ""

    # Optional numpy-typed probability vector (lazy import)
    probability_distribution: Optional[Any] = field(default=None, repr=False)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "race_id": self.race_id,
            "race_type": self.race_type.name,
            "agents_involved": list(self.agents_involved),
            "probability_bound": self.probability_bound,
            "description": self.description,
            "severity": self.severity.name,
            "recommendation": self.recommendation,
        }


@dataclass
class ReportSection:
    """A single logical section of a :class:`RaceReport`.

    Sections may be nested via the *subsections* field to represent
    hierarchical report structures.
    """

    section_type: SectionType
    title: str
    content: str
    subsections: List["ReportSection"] = field(default_factory=list)

    def walk(self) -> List["ReportSection"]:
        """Depth-first traversal of this section and all descendants."""
        result: List[ReportSection] = [self]
        for child in self.subsections:
            result.extend(child.walk())
        return result


@dataclass
class RaceReport:
    """Comprehensive report aggregating all detected races.

    Serves as the top-level artefact produced by the MARACE
    verification pipeline.
    """

    races: List[DetailedRaceReport] = field(default_factory=list)
    summary: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    config_used: Dict[str, Any] = field(default_factory=dict)
    total_agents: int = 0
    total_states_explored: int = 0
    analysis_duration_s: float = 0.0
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def race_count(self) -> int:
        """Total number of races in this report."""
        return len(self.races)

    def races_by_severity(self, severity: Severity) -> List[DetailedRaceReport]:
        """Return the subset of races matching *severity*."""
        return [r for r in self.races if r.severity is severity]

    def races_by_type(self, race_type: RaceType) -> List[DetailedRaceReport]:
        """Return the subset of races matching *race_type*."""
        return [r for r in self.races if r.race_type is race_type]

    def as_dict(self) -> Dict[str, Any]:
        """Serialise the entire report to a plain dict."""
        return {
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
            "config_used": self.config_used,
            "total_agents": self.total_agents,
            "total_states_explored": self.total_states_explored,
            "analysis_duration_s": self.analysis_duration_s,
            "race_count": self.race_count,
            "races": [r.as_dict() for r in self.races],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


class SummaryGenerator:
    """Generates an executive summary from a :pytype:`RaceCatalog`.

    The catalog is expected to expose an iterable of race-like objects;
    until the ``race`` module is implemented we operate on
    :class:`DetailedRaceReport` instances directly.
    """

    def generate(self, race_catalog: Sequence[DetailedRaceReport]) -> str:
        """Produce a human-readable executive summary.

        Parameters
        ----------
        race_catalog:
            Iterable of :class:`DetailedRaceReport` instances (or any
            object exposing ``severity`` and ``race_type`` attributes).

        Returns
        -------
        str
            Multi-line summary suitable for the top of a report.
        """
        if not race_catalog:
            return "No races detected.  The system appears free of concurrency issues."

        lines: List[str] = [
            f"MARACE Executive Summary — {len(race_catalog)} race(s) detected",
            "=" * 60,
            "",
            self._format_severity_breakdown(race_catalog),
            "",
            self._format_agent_involvement(race_catalog),
            "",
            self._recommend_mitigations(race_catalog),
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_severity_breakdown(
        catalog: Sequence[DetailedRaceReport],
    ) -> str:
        """Tabulate the number of races at each severity level."""
        counts: Dict[Severity, int] = {}
        for race in catalog:
            counts[race.severity] = counts.get(race.severity, 0) + 1

        header = "Severity Breakdown:"
        rows = [
            f"  {sev.name:<12s} {counts.get(sev, 0)}"
            for sev in Severity
        ]
        return "\n".join([header, *rows])

    @staticmethod
    def _format_agent_involvement(
        catalog: Sequence[DetailedRaceReport],
    ) -> str:
        """List agents and how many races each participates in."""
        agent_counts: Dict[str, int] = {}
        for race in catalog:
            for agent in race.agents_involved:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1

        header = "Agent Involvement:"
        rows = [
            f"  {agent:<20s} {count} race(s)"
            for agent, count in sorted(
                agent_counts.items(), key=lambda kv: -kv[1]
            )
        ]
        return "\n".join([header, *rows])

    @staticmethod
    def _recommend_mitigations(
        catalog: Sequence[DetailedRaceReport],
    ) -> str:
        """Emit high-level mitigation advice based on observed race types."""
        types_seen = {r.race_type for r in catalog}
        lines: List[str] = ["Recommended Mitigations:"]

        _advice: Dict[RaceType, str] = {
            RaceType.COLLISION: "Introduce token-based mutual exclusion or spatial partitioning.",
            RaceType.DEADLOCK: "Apply a consistent resource-ordering protocol across agents.",
            RaceType.LIVELOCK: "Add randomised back-off or priority escalation.",
            RaceType.CORRELATED_FAILURE: "De-correlate failure modes with diverse redundancy.",
            RaceType.RESOURCE_CONFLICT: "Use a centralised resource broker or optimistic concurrency control.",
        }

        for rtype in RaceType:
            if rtype in types_seen:
                lines.append(f"  [{rtype.name}] {_advice[rtype]}")

        if len(lines) == 1:
            lines.append("  No specific mitigations identified.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Catalog comparison
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Outcome of comparing two or more race catalogs."""

    added: List[DetailedRaceReport] = field(default_factory=list)
    removed: List[DetailedRaceReport] = field(default_factory=list)
    unchanged: List[DetailedRaceReport] = field(default_factory=list)
    regressions: List[DetailedRaceReport] = field(default_factory=list)
    summary: str = ""


class ComparisonReport:
    """Compare race catalogs across different configurations.

    Useful for regression testing: run the verifier with two configs and
    check whether new races appeared or old ones disappeared.
    """

    def compare(
        self,
        catalogs: List[Sequence[DetailedRaceReport]],
    ) -> ComparisonResult:
        """Compare an ordered list of catalogs (earliest first).

        Parameters
        ----------
        catalogs:
            At least two catalogs.  The first is treated as the
            *baseline* and the last as the *current* run.

        Returns
        -------
        ComparisonResult
        """
        if len(catalogs) < 2:
            return ComparisonResult(summary="Need at least two catalogs to compare.")

        baseline = catalogs[0]
        current = catalogs[-1]

        added, removed, unchanged = self._diff_races(baseline, current)
        regressions = self._compute_regression(baseline, current)

        parts: List[str] = [
            f"Compared {len(catalogs)} catalog(s).",
            f"  Baseline races : {len(baseline)}",
            f"  Current races  : {len(current)}",
            f"  Added          : {len(added)}",
            f"  Removed        : {len(removed)}",
            f"  Unchanged      : {len(unchanged)}",
            f"  Regressions    : {len(regressions)}",
        ]

        return ComparisonResult(
            added=added,
            removed=removed,
            unchanged=unchanged,
            regressions=regressions,
            summary="\n".join(parts),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _diff_races(
        baseline: Sequence[DetailedRaceReport],
        current: Sequence[DetailedRaceReport],
    ) -> Tuple[
        List[DetailedRaceReport],
        List[DetailedRaceReport],
        List[DetailedRaceReport],
    ]:
        """Partition *current* into added / removed / unchanged vs *baseline*."""
        baseline_ids = {r.race_id for r in baseline}
        current_ids = {r.race_id for r in current}

        added = [r for r in current if r.race_id not in baseline_ids]
        removed = [r for r in baseline if r.race_id not in current_ids]
        unchanged = [r for r in current if r.race_id in baseline_ids]
        return added, removed, unchanged

    @staticmethod
    def _compute_regression(
        baseline: Sequence[DetailedRaceReport],
        current: Sequence[DetailedRaceReport],
    ) -> List[DetailedRaceReport]:
        """Identify races whose severity increased relative to baseline."""
        _severity_rank = {s: i for i, s in enumerate(Severity)}
        baseline_map = {r.race_id: r for r in baseline}

        regressions: List[DetailedRaceReport] = []
        for race in current:
            prev = baseline_map.get(race.race_id)
            if prev is None:
                continue
            if _severity_rank[race.severity] < _severity_rank[prev.severity]:
                regressions.append(race)
        return regressions


# ---------------------------------------------------------------------------
# Report formatters
# ---------------------------------------------------------------------------


class ReportFormatter(ABC):
    """Abstract base for report serialisation strategies."""

    @abstractmethod
    def format(self, report: RaceReport) -> str:
        """Render *report* as a string in the target format."""


class TextReportFormatter(ReportFormatter):
    """Plain-text output with ASCII box-drawing and indentation."""

    _WIDTH = 72
    _SEP = "=" * _WIDTH
    _THIN_SEP = "-" * _WIDTH

    def format(self, report: RaceReport) -> str:  # noqa: D401
        lines: List[str] = [
            self._SEP,
            "MARACE Race-Condition Report".center(self._WIDTH),
            self._SEP,
            f"Timestamp           : {report.timestamp.isoformat()}",
            f"Total agents        : {report.total_agents}",
            f"States explored     : {report.total_states_explored}",
            f"Analysis duration   : {report.analysis_duration_s:.2f} s",
            f"Races detected      : {report.race_count}",
            self._THIN_SEP,
        ]

        if report.summary:
            lines.append(report.summary)
            lines.append(self._THIN_SEP)

        for race in report.races:
            lines.extend(self._format_race(race))
            lines.append(self._THIN_SEP)

        for section in report.sections:
            lines.extend(self._format_section(section, depth=0))

        lines.append(self._SEP)
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------

    def _format_race(self, race: DetailedRaceReport) -> List[str]:
        return [
            f"  Race ID       : {race.race_id}",
            f"  Type          : {race.race_type.name}",
            f"  Severity      : {race.severity.name}",
            f"  Agents        : {', '.join(race.agents_involved)}",
            f"  P(trigger)    : {race.probability_bound:.4g}",
            f"  Description   : {race.description}",
            f"  Recommendation: {race.recommendation}",
        ]

    def _format_section(
        self, section: ReportSection, depth: int
    ) -> List[str]:
        indent = "  " * depth
        lines = [
            f"{indent}[{section.section_type.name}] {section.title}",
            textwrap.indent(section.content, indent + "  "),
        ]
        for child in section.subsections:
            lines.extend(self._format_section(child, depth + 1))
        return lines


class JSONReportFormatter(ReportFormatter):
    """Structured JSON output."""

    def __init__(self, *, indent: int = 2) -> None:
        self._indent = indent

    def format(self, report: RaceReport) -> str:
        payload = report.as_dict()
        # Sections are serialised recursively.
        payload["sections"] = [
            self._section_dict(s) for s in report.sections
        ]
        return json.dumps(payload, indent=self._indent, default=str)

    @staticmethod
    def _section_dict(section: ReportSection) -> Dict[str, Any]:
        return {
            "section_type": section.section_type.name,
            "title": section.title,
            "content": section.content,
            "subsections": [
                JSONReportFormatter._section_dict(c)
                for c in section.subsections
            ],
        }


class HTMLReportFormatter(ReportFormatter):
    """HTML report with inline CSS, tables, and severity colour-coding."""

    _SEVERITY_COLORS: Dict[Severity, str] = {
        Severity.CRITICAL: "#d32f2f",
        Severity.HIGH: "#f57c00",
        Severity.MEDIUM: "#fbc02d",
        Severity.LOW: "#388e3c",
        Severity.INFO: "#1976d2",
    }

    def format(self, report: RaceReport) -> str:
        parts: List[str] = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head><meta charset='utf-8'/>",
            "<title>MARACE Report</title>",
            "<style>",
            self._css(),
            "</style></head>",
            "<body>",
            "<h1>MARACE Race-Condition Report</h1>",
            self._meta_table(report),
        ]

        if report.summary:
            parts.append(f"<h2>Summary</h2><pre>{_html.escape(report.summary)}</pre>")

        if report.races:
            parts.append("<h2>Detected Races</h2>")
            parts.append(self._race_table(report.races))

        for section in report.sections:
            parts.append(self._render_section(section, level=2))

        parts.extend(["</body>", "</html>"])
        return "\n".join(parts)

    # ------------------------------------------------------------------

    @staticmethod
    def _css() -> str:
        return textwrap.dedent("""\
            body { font-family: sans-serif; margin: 2em; }
            table { border-collapse: collapse; width: 100%; margin: 1em 0; }
            th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
            th { background: #f5f5f5; }
            .sev-badge { padding: 2px 8px; border-radius: 4px; color: #fff;
                         font-weight: bold; font-size: 0.85em; }
            pre { background: #fafafa; padding: 1em; overflow-x: auto; }
        """)

    def _meta_table(self, report: RaceReport) -> str:
        rows = [
            ("Timestamp", report.timestamp.isoformat()),
            ("Total Agents", str(report.total_agents)),
            ("States Explored", str(report.total_states_explored)),
            ("Analysis Duration", f"{report.analysis_duration_s:.2f} s"),
            ("Races Detected", str(report.race_count)),
        ]
        body = "".join(
            f"<tr><th>{_html.escape(k)}</th><td>{_html.escape(v)}</td></tr>"
            for k, v in rows
        )
        return f"<table>{body}</table>"

    def _race_table(self, races: List[DetailedRaceReport]) -> str:
        header = (
            "<tr><th>ID</th><th>Type</th><th>Severity</th>"
            "<th>Agents</th><th>P(trigger)</th><th>Description</th></tr>"
        )
        rows: List[str] = []
        for r in races:
            color = self._SEVERITY_COLORS.get(r.severity, "#999")
            badge = (
                f"<span class='sev-badge' style='background:{color}'>"
                f"{_html.escape(r.severity.name)}</span>"
            )
            rows.append(
                f"<tr><td>{_html.escape(r.race_id)}</td>"
                f"<td>{_html.escape(r.race_type.name)}</td>"
                f"<td>{badge}</td>"
                f"<td>{_html.escape(', '.join(r.agents_involved))}</td>"
                f"<td>{r.probability_bound:.4g}</td>"
                f"<td>{_html.escape(r.description)}</td></tr>"
            )
        return f"<table>{header}{''.join(rows)}</table>"

    def _render_section(self, section: ReportSection, level: int) -> str:
        tag = f"h{min(level + 1, 6)}"
        parts = [
            f"<{tag}>{_html.escape(section.title)}</{tag}>",
            f"<p>{_html.escape(section.content)}</p>",
        ]
        for child in section.subsections:
            parts.append(self._render_section(child, level + 1))
        return "\n".join(parts)


class LaTeXReportFormatter(ReportFormatter):
    r"""LaTeX tables suitable for inclusion in academic papers.

    Escapes special characters (``%``, ``&``, ``_``, ``#``, ``$``,
    ``{``, ``}``, ``~``, ``^``) and wraps race data in a
    ``longtable`` environment.
    """

    _SPECIAL = re.compile(r"([%&_#\$\{\}~\^\\])")

    @classmethod
    def _escape(cls, text: str) -> str:
        """Escape LaTeX special characters in *text*."""
        def _repl(m: re.Match) -> str:  # type: ignore[type-arg]
            ch = m.group(1)
            if ch == "\\":
                return r"\textbackslash{}"
            if ch == "~":
                return r"\textasciitilde{}"
            if ch == "^":
                return r"\textasciicircum{}"
            return f"\\{ch}"

        return cls._SPECIAL.sub(_repl, text)

    def format(self, report: RaceReport) -> str:
        lines: List[str] = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{MARACE Race-Condition Report}",
            r"\label{tab:marace-report}",
            r"\begin{tabular}{l l l l r p{5cm}}",
            r"\toprule",
            r"ID & Type & Severity & Agents & $P$ & Description \\",
            r"\midrule",
        ]

        for race in report.races:
            agents = self._escape(", ".join(race.agents_involved))
            lines.append(
                f"{self._escape(race.race_id)} & "
                f"{self._escape(race.race_type.name)} & "
                f"{self._escape(race.severity.name)} & "
                f"{agents} & "
                f"{race.probability_bound:.4g} & "
                f"{self._escape(race.description)} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
        ])

        # Summary footnote
        meta = (
            f"Agents: {report.total_agents}, "
            f"States: {report.total_states_explored}, "
            f"Duration: {report.analysis_duration_s:.2f}\\,s"
        )
        lines.append(
            r"\par\smallskip\noindent\footnotesize " + self._escape(meta)
        )
        lines.append(r"\end{table}")
        return "\n".join(lines) + "\n"
