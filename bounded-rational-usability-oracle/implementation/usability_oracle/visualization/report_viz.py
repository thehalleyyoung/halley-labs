"""
usability_oracle.visualization.report_viz — Complete report visualization.

Assembles all visualization components into a structured analysis report
with executive summary, detailed findings, and recommendations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ReportSection:
    """A single section in the report."""
    title: str
    content: str
    severity: str = "info"
    subsections: list["ReportSection"] = field(default_factory=list)


class ReportVisualizer:
    """Generate complete text-based analysis reports."""

    def __init__(self, width: int = 80) -> None:
        self._width = width

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        analysis_result: dict[str, Any],
        title: str = "Usability Oracle Analysis Report",
    ) -> str:
        """Generate a complete analysis report from pipeline results."""
        sections: list[str] = []

        sections.append(self._header(title))
        sections.append(self._executive_summary(analysis_result))
        sections.append(self._verdict_section(analysis_result))
        sections.append(self._bottleneck_section(analysis_result))
        sections.append(self._cost_section(analysis_result))
        sections.append(self._recommendations_section(analysis_result))
        sections.append(self._footer(analysis_result))

        return "\n\n".join(s for s in sections if s)

    # ------------------------------------------------------------------
    # Report components
    # ------------------------------------------------------------------

    def _header(self, title: str) -> str:
        border = "═" * self._width
        lines = [
            border,
            f"  {title}",
            f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            border,
        ]
        return "\n".join(lines)

    def _executive_summary(self, result: dict[str, Any]) -> str:
        verdict = result.get("verdict", "unknown")
        confidence = result.get("confidence", 0.0)
        total_cost = result.get("total_cost", 0.0)
        n_bottlenecks = result.get("n_bottlenecks", 0)

        verdict_icon = {
            "regression": "🔴 REGRESSION",
            "improvement": "🟢 IMPROVEMENT",
            "neutral": "⚪ NEUTRAL",
        }.get(str(verdict).lower(), f"⚪ {verdict}")

        lines = [
            self._section_header("Executive Summary"),
            f"  Verdict:      {verdict_icon}",
            f"  Confidence:   {confidence:.1%}",
            f"  Total Cost:   {total_cost:.3f}",
            f"  Bottlenecks:  {n_bottlenecks}",
        ]

        description = result.get("description", "")
        if description:
            lines.append(f"\n  {description}")

        return "\n".join(lines)

    def _verdict_section(self, result: dict[str, Any]) -> str:
        comparison = result.get("comparison", {})
        if not comparison:
            return ""

        lines = [self._section_header("Verdict Details")]

        cost_a = comparison.get("cost_a", 0)
        cost_b = comparison.get("cost_b", 0)
        diff = comparison.get("cost_diff", cost_b - cost_a)
        pct = comparison.get("cost_pct_change", 0)

        lines.append(f"  Cost (before): {cost_a:.3f}")
        lines.append(f"  Cost (after):  {cost_b:.3f}")
        lines.append(f"  Change:        {'+' if diff > 0 else ''}{diff:.3f} ({'+' if pct > 0 else ''}{pct:.1f}%)")

        # Per-dimension breakdown
        dimensions = comparison.get("dimensions", {})
        if dimensions:
            lines.append(f"\n  Per-dimension breakdown:")
            for dim, vals in dimensions.items():
                before = vals.get("before", 0)
                after = vals.get("after", 0)
                change = after - before
                lines.append(f"    {dim:<20} {before:.3f} → {after:.3f} ({'+' if change > 0 else ''}{change:.3f})")

        return "\n".join(lines)

    def _bottleneck_section(self, result: dict[str, Any]) -> str:
        bottlenecks = result.get("bottlenecks", [])
        if not bottlenecks:
            return ""

        lines = [self._section_header(f"Bottlenecks ({len(bottlenecks)} found)")]

        for i, b in enumerate(sorted(bottlenecks, key=lambda x: -x.get("severity", 0)), 1):
            severity = b.get("severity", 0)
            btype = b.get("type", "unknown")
            location = b.get("location", "?")
            desc = b.get("description", "")

            icon = "🔴" if severity > 0.7 else ("🟡" if severity > 0.4 else "🟢")
            lines.append(f"\n  {icon} #{i}: {btype}")
            lines.append(f"     Location:  {location}")
            lines.append(f"     Severity:  {'█' * int(severity * 10)}{'░' * (10 - int(severity * 10))} {severity:.2f}")
            if desc:
                lines.append(f"     Details:   {desc}")

        return "\n".join(lines)

    def _cost_section(self, result: dict[str, Any]) -> str:
        costs = result.get("cost_breakdown", {})
        if not costs:
            return ""

        total = sum(costs.values())
        max_val = max(costs.values()) if costs else 1

        lines = [self._section_header("Cost Breakdown")]

        for name, val in sorted(costs.items(), key=lambda x: -x[1]):
            bar_len = int(val / max_val * 30) if max_val > 0 else 0
            pct = val / total * 100 if total > 0 else 0
            lines.append(f"  {name:<20} {'█' * bar_len}{'░' * (30 - bar_len)} {val:.3f} ({pct:.1f}%)")

        lines.append(f"  {'─' * 65}")
        lines.append(f"  {'TOTAL':<20} {' ' * 30} {total:.3f}")

        return "\n".join(lines)

    def _recommendations_section(self, result: dict[str, Any]) -> str:
        recommendations = result.get("recommendations", [])
        if not recommendations:
            return ""

        lines = [self._section_header("Recommendations")]

        for i, rec in enumerate(recommendations, 1):
            priority = rec.get("priority", "medium")
            title = rec.get("title", "")
            description = rec.get("description", "")
            impact = rec.get("impact", "")

            icon = {"high": "❗", "medium": "➤", "low": "ℹ"}.get(priority, "•")
            lines.append(f"\n  {icon} {i}. {title}")
            if description:
                lines.append(f"     {description}")
            if impact:
                lines.append(f"     Expected impact: {impact}")

        return "\n".join(lines)

    def _footer(self, result: dict[str, Any]) -> str:
        timing = result.get("timing", {})
        total = timing.get("total", 0)

        lines = [
            "─" * self._width,
            f"  Analysis completed in {total:.3f}s",
        ]

        if len(timing) > 1:
            stage_times = {k: v for k, v in timing.items() if k != "total"}
            stages = sorted(stage_times.items(), key=lambda x: -x[1])[:5]
            lines.append(f"  Top stages: " + ", ".join(f"{k}={v:.3f}s" for k, v in stages))

        lines.append("─" * self._width)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _section_header(self, title: str) -> str:
        return f"{'─' * self._width}\n  {title}\n{'─' * self._width}"

    # ------------------------------------------------------------------
    # Comparison report
    # ------------------------------------------------------------------

    def generate_comparison_report(
        self,
        result_a: dict[str, Any],
        result_b: dict[str, Any],
        title: str = "A/B Comparison Report",
    ) -> str:
        """Generate a comparison report between two analysis results."""
        lines = [self._header(title)]

        metrics_a = result_a.get("metrics", {})
        metrics_b = result_b.get("metrics", {})
        all_keys = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))

        lines.append(self._section_header("Metric Comparison"))
        lines.append(f"  {'Metric':<25}  {'Version A':>10}  {'Version B':>10}  {'Change':>10}")
        lines.append(f"  {'─' * 25}  {'─' * 10}  {'─' * 10}  {'─' * 10}")

        for key in all_keys:
            va = metrics_a.get(key, 0)
            vb = metrics_b.get(key, 0)
            diff = vb - va
            sign = "+" if diff > 0 else ""
            lines.append(f"  {key:<25}  {va:>10.3f}  {vb:>10.3f}  {sign}{diff:>9.3f}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Markdown export
    # ------------------------------------------------------------------

    def to_markdown(self, analysis_result: dict[str, Any]) -> str:
        """Export analysis results as Markdown."""
        verdict = analysis_result.get("verdict", "unknown")
        confidence = analysis_result.get("confidence", 0)

        lines = [
            "# Usability Oracle Report",
            "",
            f"**Verdict:** {verdict}  ",
            f"**Confidence:** {confidence:.1%}",
            "",
            "## Bottlenecks",
            "",
        ]

        for b in analysis_result.get("bottlenecks", []):
            severity = b.get("severity", 0)
            btype = b.get("type", "?")
            desc = b.get("description", "")
            lines.append(f"- **{btype}** (severity: {severity:.2f}): {desc}")

        costs = analysis_result.get("cost_breakdown", {})
        if costs:
            lines.extend(["", "## Cost Breakdown", "", "| Component | Cost | % |", "|---|---:|---:|"])
            total = sum(costs.values())
            for name, val in sorted(costs.items(), key=lambda x: -x[1]):
                pct = val / total * 100 if total > 0 else 0
                lines.append(f"| {name} | {val:.3f} | {pct:.1f}% |")

        return "\n".join(lines)
