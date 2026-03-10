"""
usability_oracle.comparison.reporter — Regression report generation.

Implements :class:`RegressionReporter`, which transforms raw
:class:`ComparisonResult` objects into structured reports in multiple
formats: human-readable text, JSON, SARIF (for GitHub integration),
and Markdown.

SARIF output follows the Static Analysis Results Interchange Format
v2.1.0, enabling integration with GitHub Code Scanning and other
SARIF-compatible tools.

References
----------
- OASIS (2020). SARIF v2.1.0 specification. https://docs.oasis-open.org
"""

from __future__ import annotations

import json
import logging
import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from usability_oracle.core.enums import RegressionVerdict, Severity
from usability_oracle.cognitive.models import CostElement
from usability_oracle.comparison.models import (
    BottleneckChange,
    ChangeDirection,
    ComparisonContext,
    ComparisonResult,
    RegressionReport,
)

logger = logging.getLogger(__name__)

# Verdict → severity mapping for SARIF
_VERDICT_SARIF_LEVEL = {
    RegressionVerdict.REGRESSION: "error",
    RegressionVerdict.IMPROVEMENT: "note",
    RegressionVerdict.NEUTRAL: "none",
    RegressionVerdict.INCONCLUSIVE: "warning",
}

_VERDICT_EMOJI = {
    RegressionVerdict.REGRESSION: "🔴",
    RegressionVerdict.IMPROVEMENT: "🟢",
    RegressionVerdict.NEUTRAL: "⚪",
    RegressionVerdict.INCONCLUSIVE: "🟡",
}


class RegressionReporter:
    """Generates structured regression reports from comparison results.

    Supports multiple output formats (JSON, SARIF, Markdown) and
    produces actionable recommendations based on detected bottleneck
    changes and cost deltas.

    Parameters
    ----------
    tool_name : str
        Name of the analysis tool (for SARIF metadata).
    tool_version : str
        Version string (for SARIF metadata).
    """

    def __init__(
        self,
        tool_name: str = "usability-oracle",
        tool_version: str = "0.1.0",
    ) -> None:
        self.tool_name = tool_name
        self.tool_version = tool_version

    def report(
        self,
        result: ComparisonResult,
        context: ComparisonContext,
    ) -> RegressionReport:
        """Generate a full regression report.

        Parameters
        ----------
        result : ComparisonResult
            The comparison result to report on.
        context : ComparisonContext
            Contextual information (MDPs, alignment, task, config).

        Returns
        -------
        RegressionReport
        """
        summary = self._generate_summary(result)
        recommendations = self._generate_recommendations(
            result, result.bottleneck_changes
        )

        return RegressionReport(
            comparison_result=result,
            task_results={context.task_spec.spec_id: result} if context.task_spec.spec_id else {},
            overall_verdict=result.verdict,
            recommendations=recommendations,
            metadata={
                "tool": self.tool_name,
                "version": self.tool_version,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "summary": summary,
                "mdp_before_states": context.mdp_before.n_states,
                "mdp_after_states": context.mdp_after.n_states,
                "alignment_mapped": context.alignment.n_mapped,
                "is_parameter_free": result.is_parameter_free,
            },
        )

    def _generate_summary(self, result: ComparisonResult) -> str:
        """Generate a human-readable summary.

        Parameters
        ----------
        result : ComparisonResult

        Returns
        -------
        str
            Summary paragraph.
        """
        emoji = _VERDICT_EMOJI.get(result.verdict, "❓")
        delta_ms = result.delta_cost.mean_time * 1000  # convert to ms

        parts: list[str] = [
            f"{emoji} Verdict: **{result.verdict.value.upper()}**",
            "",
            f"Expected task completion time changed by {delta_ms:+.1f} ms "
            f"(from {result.cost_before.mean_time * 1000:.1f} ms to "
            f"{result.cost_after.mean_time * 1000:.1f} ms).",
            "",
            f"Effect size: Cohen's d = {result.effect_size:.3f} "
            f"({result.effect_magnitude})",
            f"Statistical significance: p = {result.p_value:.4f} "
            f"(α = {1 - result.confidence:.2f})",
        ]

        if result.is_parameter_free:
            parts.append("✅ Verdict is parameter-free (holds for all β).")
        else:
            parts.append("⚠️  Verdict may depend on the rationality parameter β.")

        n_new = sum(
            1 for bc in result.bottleneck_changes
            if bc.direction == ChangeDirection.NEW
        )
        n_resolved = sum(
            1 for bc in result.bottleneck_changes
            if bc.direction == ChangeDirection.RESOLVED
        )
        if n_new or n_resolved:
            parts.append(
                f"Bottleneck changes: {n_new} new, {n_resolved} resolved."
            )

        return "\n".join(parts)

    def _generate_recommendations(
        self,
        result: ComparisonResult,
        bottleneck_changes: list[BottleneckChange],
    ) -> list[str]:
        """Generate actionable recommendations.

        Parameters
        ----------
        result : ComparisonResult
        bottleneck_changes : list[BottleneckChange]

        Returns
        -------
        list[str]
        """
        recs: list[str] = []

        if result.verdict == RegressionVerdict.REGRESSION:
            recs.append(
                "CRITICAL: This change introduces a usability regression. "
                "Review the delta cost breakdown and bottleneck changes before merging."
            )

            if result.effect_size > 0.8:
                recs.append(
                    f"Large effect size (d={result.effect_size:.2f}): "
                    "the regression is substantial and likely noticeable to users."
                )

        for bc in bottleneck_changes:
            if bc.direction == ChangeDirection.NEW:
                recs.append(
                    f"New {bc.bottleneck_type.value} bottleneck at state "
                    f"'{bc.state_id}' (severity {bc.after_severity:.2f}). "
                    f"{bc.description}"
                )
            elif bc.direction == ChangeDirection.WORSENED:
                recs.append(
                    f"Worsened {bc.bottleneck_type.value} at state "
                    f"'{bc.state_id}': severity {bc.before_severity:.2f} → "
                    f"{bc.after_severity:.2f}. Consider simplifying this "
                    "interaction point."
                )

        if not result.is_parameter_free:
            recs.append(
                "The verdict depends on the rationality parameter β. "
                "Run a parameter-free analysis to verify robustness."
            )

        if result.verdict == RegressionVerdict.INCONCLUSIVE:
            recs.append(
                "Inconclusive result: increase the number of Monte Carlo "
                "trajectories or reduce the significance threshold."
            )

        if result.verdict == RegressionVerdict.IMPROVEMENT:
            resolved = [
                bc for bc in bottleneck_changes
                if bc.direction == ChangeDirection.RESOLVED
            ]
            if resolved:
                recs.append(
                    f"Good: {len(resolved)} bottleneck(s) resolved by this change."
                )

        return recs

    def _format_cost_table(
        self,
        cost_before: CostElement,
        cost_after: CostElement,
        delta: CostElement,
    ) -> str:
        """Format a cost comparison table.

        Parameters
        ----------
        cost_before, cost_after, delta : CostElement

        Returns
        -------
        str
            Formatted table string.
        """
        lines = [
            "| Metric         | Before (ms) | After (ms)  | Delta (ms)  |",
            "|----------------|-------------|-------------|-------------|",
        ]
        b_ms = cost_before.mean_time * 1000
        a_ms = cost_after.mean_time * 1000
        d_ms = delta.mean_time * 1000
        b_sd = (cost_before.variance ** 0.5) * 1000 if cost_before.variance > 0 else 0
        a_sd = (cost_after.variance ** 0.5) * 1000 if cost_after.variance > 0 else 0

        lines.append(
            f"| Mean time      | {b_ms:>11.1f} | {a_ms:>11.1f} | {d_ms:>+11.1f} |"
        )
        lines.append(
            f"| Std deviation   | {b_sd:>11.1f} | {a_sd:>11.1f} | {'—':>11s} |"
        )

        return "\n".join(lines)

    def _format_bottleneck_changes(
        self, changes: list[BottleneckChange]
    ) -> str:
        """Format bottleneck changes as a readable list.

        Parameters
        ----------
        changes : list[BottleneckChange]

        Returns
        -------
        str
        """
        if not changes:
            return "No bottleneck changes detected."

        lines: list[str] = []
        for bc in changes:
            direction_emoji = {
                ChangeDirection.NEW: "🆕",
                ChangeDirection.RESOLVED: "✅",
                ChangeDirection.WORSENED: "⬆️",
                ChangeDirection.IMPROVED: "⬇️",
            }.get(bc.direction, "❓")

            lines.append(
                f"  {direction_emoji} [{bc.direction.value.upper()}] "
                f"{bc.bottleneck_type.value} at '{bc.state_id}': "
                f"{bc.before_severity:.2f} → {bc.after_severity:.2f}"
            )
            if bc.description:
                lines.append(f"     {bc.description}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Output formats
    # ------------------------------------------------------------------

    def to_json(self, report: RegressionReport) -> str:
        """Serialize the report to JSON.

        Parameters
        ----------
        report : RegressionReport

        Returns
        -------
        str
            JSON string.
        """
        def _serialize(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _serialize(v) for k, v in obj.__dict__.items()}
            if hasattr(obj, "value"):
                return obj.value
            if isinstance(obj, (set, frozenset)):
                return list(obj)
            if isinstance(obj, dict):
                return {str(k): _serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_serialize(v) for v in obj]
            return obj

        data = _serialize(report)
        return json.dumps(data, indent=2, default=str)

    def to_sarif(self, report: RegressionReport) -> str:
        """Serialize the report to SARIF v2.1.0 format.

        SARIF (Static Analysis Results Interchange Format) enables
        integration with GitHub Code Scanning, VS Code, and other tools.

        Parameters
        ----------
        report : RegressionReport

        Returns
        -------
        str
            SARIF JSON string.
        """
        results_list: list[dict[str, Any]] = []

        # Main result
        cr = report.comparison_result
        level = _VERDICT_SARIF_LEVEL.get(cr.verdict, "warning")

        main_result: dict[str, Any] = {
            "ruleId": "usability/regression",
            "level": level,
            "message": {
                "text": cr.description or f"Usability {cr.verdict.value}: "
                f"Δ={cr.delta_cost.mean_time * 1000:+.1f}ms, "
                f"d={cr.effect_size:.2f}, p={cr.p_value:.4f}",
            },
            "properties": {
                "verdict": cr.verdict.value,
                "effectSize": cr.effect_size,
                "pValue": cr.p_value,
                "isParameterFree": cr.is_parameter_free,
                "costBeforeMs": cr.cost_before.mean_time * 1000,
                "costAfterMs": cr.cost_after.mean_time * 1000,
            },
        }
        results_list.append(main_result)

        # Per-bottleneck results
        for bc in cr.bottleneck_changes:
            if bc.direction in (ChangeDirection.NEW, ChangeDirection.WORSENED):
                bc_result: dict[str, Any] = {
                    "ruleId": f"usability/bottleneck/{bc.bottleneck_type.value}",
                    "level": "warning" if bc.direction == ChangeDirection.WORSENED else "error",
                    "message": {
                        "text": bc.description or (
                            f"{bc.direction.value} {bc.bottleneck_type.value} "
                            f"at '{bc.state_id}'"
                        ),
                    },
                    "properties": {
                        "beforeSeverity": bc.before_severity,
                        "afterSeverity": bc.after_severity,
                        "direction": bc.direction.value,
                    },
                }
                results_list.append(bc_result)

        # Per-task results
        for tid, tr in report.task_results.items():
            task_level = _VERDICT_SARIF_LEVEL.get(tr.verdict, "warning")
            task_result: dict[str, Any] = {
                "ruleId": f"usability/task/{tid}",
                "level": task_level,
                "message": {
                    "text": f"Task '{tid}': {tr.verdict.value}, "
                    f"Δ={tr.delta_cost.mean_time * 1000:+.1f}ms",
                },
            }
            results_list.append(task_result)

        # SARIF envelope
        sarif: dict[str, Any] = {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": self.tool_name,
                            "version": self.tool_version,
                            "informationUri": "https://github.com/usability-oracle",
                            "rules": [
                                {
                                    "id": "usability/regression",
                                    "shortDescription": {
                                        "text": "Usability regression detection"
                                    },
                                    "fullDescription": {
                                        "text": (
                                            "Detects statistically significant "
                                            "regressions in task completion cost "
                                            "using bounded-rational policy analysis."
                                        ),
                                    },
                                    "helpUri": "https://github.com/usability-oracle/docs/regression",
                                },
                            ],
                        },
                    },
                    "results": results_list,
                },
            ],
        }

        return json.dumps(sarif, indent=2)

    def to_markdown(self, report: RegressionReport) -> str:
        """Serialize the report to Markdown.

        Parameters
        ----------
        report : RegressionReport

        Returns
        -------
        str
            Markdown string suitable for PR comments or documentation.
        """
        cr = report.comparison_result
        emoji = _VERDICT_EMOJI.get(cr.verdict, "❓")

        lines: list[str] = [
            f"# {emoji} Usability Regression Report",
            "",
            f"**Verdict:** {cr.verdict.value.upper()}",
            f"**Confidence:** {cr.confidence * 100:.0f}%",
            f"**Parameter-free:** {'Yes ✅' if cr.is_parameter_free else 'No ⚠️'}",
            "",
            "## Cost Summary",
            "",
            self._format_cost_table(cr.cost_before, cr.cost_after, cr.delta_cost),
            "",
            f"**Effect size:** Cohen's d = {cr.effect_size:.3f} ({cr.effect_magnitude})",
            f"**p-value:** {cr.p_value:.4f}",
            "",
        ]

        if cr.bottleneck_changes:
            lines.extend([
                "## Bottleneck Changes",
                "",
                self._format_bottleneck_changes(cr.bottleneck_changes),
                "",
            ])

        if report.task_results:
            lines.extend([
                "## Per-Task Results",
                "",
                "| Task | Verdict | Δ (ms) | Effect Size |",
                "|------|---------|--------|-------------|",
            ])
            for tid, tr in report.task_results.items():
                t_emoji = _VERDICT_EMOJI.get(tr.verdict, "❓")
                lines.append(
                    f"| {tid} | {t_emoji} {tr.verdict.value} | "
                    f"{tr.delta_cost.mean_time * 1000:+.1f} | "
                    f"{tr.effect_size:.2f} |"
                )
            lines.append("")

        if report.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        lines.extend([
            "---",
            f"*Generated by {self.tool_name} v{self.tool_version} "
            f"at {report.metadata.get('timestamp', 'unknown')}*",
        ])

        return "\n".join(lines)
