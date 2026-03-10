"""GitHub PR integration for usability-regression results.

Provides :class:`GitHubCommentFormatter` for producing GitHub-flavoured
Markdown comments, Check Run payloads, inline annotations, and status
badges suitable for posting to pull requests via the GitHub API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from usability_oracle.core.enums import (
    BottleneckType,
    RegressionVerdict,
    Severity,
)
from usability_oracle.output.models import (
    BottleneckDescription,
    CostComparison,
    PipelineResult,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERDICT_EMOJI: Dict[RegressionVerdict, str] = {
    RegressionVerdict.REGRESSION: "🔴",
    RegressionVerdict.IMPROVEMENT: "🟢",
    RegressionVerdict.NEUTRAL: "🟡",
    RegressionVerdict.INCONCLUSIVE: "⚪",
}

_VERDICT_LABEL: Dict[RegressionVerdict, str] = {
    RegressionVerdict.REGRESSION: "Regression Detected",
    RegressionVerdict.IMPROVEMENT: "Improvement Detected",
    RegressionVerdict.NEUTRAL: "No Significant Change",
    RegressionVerdict.INCONCLUSIVE: "Inconclusive",
}

_SEVERITY_EMOJI: Dict[Severity, str] = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🔵",
    Severity.INFO: "⚪",
}

_CHECK_CONCLUSION: Dict[RegressionVerdict, str] = {
    RegressionVerdict.REGRESSION: "failure",
    RegressionVerdict.IMPROVEMENT: "success",
    RegressionVerdict.NEUTRAL: "neutral",
    RegressionVerdict.INCONCLUSIVE: "neutral",
}

_ANNOTATION_LEVEL: Dict[Severity, str] = {
    Severity.CRITICAL: "failure",
    Severity.HIGH: "failure",
    Severity.MEDIUM: "warning",
    Severity.LOW: "notice",
    Severity.INFO: "notice",
}


def _safe_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return round(value, 4)


# ═══════════════════════════════════════════════════════════════════════════
# GitHubCommentFormatter
# ═══════════════════════════════════════════════════════════════════════════

class GitHubCommentFormatter:
    """Format usability-regression results for GitHub PR integration.

    Usage::

        fmt = GitHubCommentFormatter()
        comment = fmt.format_pr_comment(result)
        check_run = fmt.format_check_run(result)
    """

    def __init__(
        self,
        tool_name: str = "Usability Oracle",
        max_annotations: int = 50,
    ) -> None:
        self._tool_name = tool_name
        self._max_annotations = max_annotations

    # ------------------------------------------------------------------
    # PR comment (Markdown)
    # ------------------------------------------------------------------

    def format_pr_comment(self, result: PipelineResult) -> str:
        """Format a full PR comment in GitHub-flavoured Markdown.

        Parameters
        ----------
        result : PipelineResult

        Returns
        -------
        str
            Markdown-formatted comment body.
        """
        sections: list[str] = []

        # Header
        emoji = _VERDICT_EMOJI.get(result.verdict, "⚪")
        label = _VERDICT_LABEL.get(result.verdict, "Unknown")
        sections.append(f"## {emoji} {self._tool_name}: {label}\n")

        # Summary
        sections.append(self._summary_section(result))

        # Cost comparison
        if result.comparison:
            sections.append(self._cost_section(result.comparison))

        # Bottlenecks
        if result.bottlenecks:
            sections.append(self._bottleneck_section(result.bottlenecks))

        # Recommendations
        if result.recommendations:
            sections.append(self._recommendations_section(result.recommendations))

        # Footer
        sections.append(self._footer(result))

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Check Run payload
    # ------------------------------------------------------------------

    def format_check_run(self, result: PipelineResult) -> dict:
        """Format a GitHub Check Run API payload.

        Parameters
        ----------
        result : PipelineResult

        Returns
        -------
        dict
            Payload suitable for the GitHub Check Runs API.
        """
        conclusion = _CHECK_CONCLUSION.get(result.verdict, "neutral")
        label = _VERDICT_LABEL.get(result.verdict, "Unknown")

        annotations = self.format_annotations(result.bottlenecks)

        summary = self._check_summary(result)
        text = self.format_pr_comment(result)

        return {
            "name": self._tool_name,
            "status": "completed",
            "conclusion": conclusion,
            "output": {
                "title": f"{self._tool_name}: {label}",
                "summary": summary,
                "text": text,
                "annotations": annotations[:self._max_annotations],
            },
        }

    # ------------------------------------------------------------------
    # Status badge (SVG)
    # ------------------------------------------------------------------

    def format_status_badge(self, verdict: RegressionVerdict) -> str:
        """Generate an SVG badge for the regression status.

        Parameters
        ----------
        verdict : RegressionVerdict

        Returns
        -------
        str
            SVG markup.
        """
        colors = {
            RegressionVerdict.REGRESSION: "#e05d44",
            RegressionVerdict.IMPROVEMENT: "#44cc11",
            RegressionVerdict.NEUTRAL: "#dfb317",
            RegressionVerdict.INCONCLUSIVE: "#9f9f9f",
        }
        labels = {
            RegressionVerdict.REGRESSION: "regression",
            RegressionVerdict.IMPROVEMENT: "improvement",
            RegressionVerdict.NEUTRAL: "neutral",
            RegressionVerdict.INCONCLUSIVE: "inconclusive",
        }

        color = colors.get(verdict, "#9f9f9f")
        label = labels.get(verdict, "unknown")
        title_width = 80
        value_width = len(label) * 7 + 10
        total_width = title_width + value_width

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">'
            f'<rect width="{title_width}" height="20" fill="#555"/>'
            f'<rect x="{title_width}" width="{value_width}" height="20" fill="{color}"/>'
            f'<text x="{title_width // 2}" y="14" fill="#fff" '
            f'text-anchor="middle" font-size="11" font-family="sans-serif">'
            f"usability</text>"
            f'<text x="{title_width + value_width // 2}" y="14" fill="#fff" '
            f'text-anchor="middle" font-size="11" font-family="sans-serif">'
            f"{label}</text>"
            f"</svg>"
        )

    # ------------------------------------------------------------------
    # Inline annotations
    # ------------------------------------------------------------------

    def format_annotations(
        self, bottlenecks: Sequence[BottleneckDescription]
    ) -> list[dict]:
        """Convert bottlenecks to GitHub Check Run annotations.

        Parameters
        ----------
        bottlenecks : Sequence[BottleneckDescription]

        Returns
        -------
        list[dict]
            Annotation objects for the Check Runs API.
        """
        annotations: list[dict] = []
        for b in bottlenecks[:self._max_annotations]:
            level = _ANNOTATION_LEVEL.get(b.severity, "notice")
            path = b.location or "accessibility-tree"
            annotation: dict[str, Any] = {
                "path": path,
                "start_line": 1,
                "end_line": 1,
                "annotation_level": level,
                "title": f"{b.bottleneck_type.value} ({b.severity.value})",
                "message": b.description,
            }
            if b.recommendation:
                annotation["raw_details"] = b.recommendation
            annotations.append(annotation)
        return annotations

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def format_summary_table(
        self, results: Sequence[PipelineResult]
    ) -> str:
        """Format a markdown summary table for multiple comparison results.

        Parameters
        ----------
        results : Sequence[PipelineResult]

        Returns
        -------
        str
            Markdown table.
        """
        lines = [
            "| # | Verdict | Bottlenecks | Cost Δ | Time |",
            "|---|---------|-------------|--------|------|",
        ]
        for i, r in enumerate(results, 1):
            emoji = _VERDICT_EMOJI.get(r.verdict, "⚪")
            n_bn = len(r.bottlenecks)
            pct = ""
            if r.comparison:
                pct = f"{r.comparison.percentage_change:+.1f}%"
            t = f"{r.total_time:.2f}s"
            lines.append(f"| {i} | {emoji} {r.verdict.value} | {n_bn} | {pct} | {t} |")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Trend chart (ASCII)
    # ------------------------------------------------------------------

    def format_trend_chart(
        self,
        history: Sequence[float],
        width: int = 40,
        height: int = 10,
        label: str = "Cost",
    ) -> str:
        """Generate an ASCII trend chart for cost over time.

        Parameters
        ----------
        history : Sequence[float]
            Sequence of cost values over time.
        width : int
            Chart width in characters.
        height : int
            Chart height in lines.
        label : str
            Y-axis label.

        Returns
        -------
        str
            ASCII art chart.
        """
        if not history:
            return f"{label}: (no data)"

        values = list(history)
        n = len(values)
        lo = min(values)
        hi = max(values)
        span = hi - lo if hi != lo else 1.0

        # Resample to fit width
        if n > width:
            step = n / width
            resampled = [values[int(i * step)] for i in range(width)]
        else:
            resampled = values
            width = n

        lines: list[str] = []
        for row in range(height - 1, -1, -1):
            threshold = lo + (row / (height - 1)) * span if height > 1 else lo
            chars: list[str] = []
            for v in resampled:
                if v >= threshold:
                    chars.append("█")
                else:
                    chars.append(" ")
            y_label = f"{threshold:8.2f}" if row in (0, height - 1) else " " * 8
            lines.append(f"{y_label} │{''.join(chars)}")

        # X-axis
        lines.append(f"{'':8s} └{'─' * width}")
        lines.append(f"{'':8s}  {label} over {n} runs")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _summary_section(self, result: PipelineResult) -> str:
        lines = [
            f"**Verdict:** {result.verdict.value}",
            f"**Bottlenecks:** {len(result.bottlenecks)} "
            f"({len(result.critical_bottlenecks)} critical)",
            f"**Analysis time:** {result.total_time:.2f}s",
        ]
        if result.comparison:
            pct = _safe_float(result.comparison.percentage_change)
            lines.append(f"**Cost change:** {pct:+.1f}%")
        return "\n".join(lines) + "\n"

    def _cost_section(self, comparison: CostComparison) -> str:
        lines = [
            "<details>",
            "<summary>📊 Cost Comparison</summary>\n",
            "| Channel | Before | After | Δ |",
            "|---------|--------|-------|---|",
        ]
        for ch, delta in comparison.channel_deltas.items():
            lines.append(f"| {ch} | — | — | {delta:+.4f} |")
        lines.append(
            f"\n**Overall change:** {comparison.percentage_change:+.1f}%\n"
        )
        lines.append("</details>\n")
        return "\n".join(lines)

    def _bottleneck_section(
        self, bottlenecks: Sequence[BottleneckDescription]
    ) -> str:
        lines = [
            "<details>",
            "<summary>🔍 Bottlenecks</summary>\n",
        ]
        for b in bottlenecks:
            emoji = _SEVERITY_EMOJI.get(b.severity, "⚪")
            lines.append(
                f"- {emoji} **{b.bottleneck_type.value}** "
                f"({b.severity.value}): {b.description}"
            )
            if b.recommendation:
                lines.append(f"  - 💡 {b.recommendation}")
        lines.append("\n</details>\n")
        return "\n".join(lines)

    def _recommendations_section(self, recs: Sequence[str]) -> str:
        lines = [
            "<details>",
            "<summary>💡 Recommendations</summary>\n",
        ]
        for r in recs:
            lines.append(f"- {r}")
        lines.append("\n</details>\n")
        return "\n".join(lines)

    def _footer(self, result: PipelineResult) -> str:
        return (
            "---\n"
            f"*Generated by {self._tool_name} • "
            f"{len(result.bottlenecks)} issues found • "
            f"{result.total_time:.2f}s*"
        )

    def _check_summary(self, result: PipelineResult) -> str:
        emoji = _VERDICT_EMOJI.get(result.verdict, "⚪")
        label = _VERDICT_LABEL.get(result.verdict, "Unknown")
        return (
            f"{emoji} {label} — "
            f"{len(result.bottlenecks)} bottlenecks, "
            f"{len(result.critical_bottlenecks)} critical"
        )
