"""
usability_oracle.cli.formatters — Output formatting for CLI results.

Provides formatting functions that transform pipeline results into
human-readable console output, JSON, SARIF, or HTML using the ``rich``
library for terminal rendering.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def format_result(result: Any, fmt: str = "console") -> str:
    """Format a pipeline result for output.

    Parameters
    ----------
    result : PipelineResult-like
        Pipeline result object.
    fmt : str
        Output format: "json", "sarif", "html", "console".

    Returns
    -------
    str
        Formatted output string.
    """
    if fmt == "json":
        return _format_json(result)
    elif fmt == "sarif":
        return _format_sarif(result)
    elif fmt == "html":
        return _format_html(result)
    else:
        return _format_console(result)


# ---------------------------------------------------------------------------
# Console formatting (rich)
# ---------------------------------------------------------------------------

def _format_console(result: Any) -> str:
    """Format result for terminal display using rich."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from io import StringIO

        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=100)

        # Header
        console.print(
            Panel(
                "[bold]Usability Oracle Analysis Report[/bold]",
                style="blue",
            )
        )

        # Pipeline status
        if hasattr(result, "success"):
            status = "[green]✓ Success[/green]" if result.success else "[red]✗ Failed[/red]"
            console.print(f"Status: {status}")

        if hasattr(result, "timing"):
            total = result.timing.get("total", 0)
            console.print(f"Total time: {total:.3f}s")

        if hasattr(result, "cache_hits"):
            console.print(f"Cache hits: {result.cache_hits}")

        # Stage results table
        if hasattr(result, "stages") and result.stages:
            console.print()
            table = Table(title="Pipeline Stages")
            table.add_column("Stage", style="cyan")
            table.add_column("Status")
            table.add_column("Time", justify="right")
            table.add_column("Cached")

            for name, sr in result.stages.items():
                status_str = (
                    "[green]✓[/green]" if sr.success
                    else "[red]✗[/red]"
                )
                time_str = f"{sr.timing:.3f}s"
                cached_str = "✓" if sr.cached else ""
                table.add_row(name, status_str, time_str, cached_str)

            console.print(table)

        # Final result
        if hasattr(result, "final_result") and result.final_result is not None:
            console.print()
            final = result.final_result

            if isinstance(final, dict):
                verdict = final.get("verdict", "unknown")
                _print_verdict(console, verdict)

                if "details" in final:
                    console.print(format_comparison(final))
            elif isinstance(final, list):
                console.print(format_bottlenecks(final))
            elif hasattr(final, "candidates"):
                _print_repair_result(console, final)

        # Errors
        if hasattr(result, "errors") and result.errors:
            console.print()
            console.print("[red bold]Errors:[/red bold]")
            for err in result.errors:
                console.print(f"  [red]✗[/red] {err}")

        return buf.getvalue()

    except ImportError:
        return _format_plain(result)


def _print_verdict(console: Any, verdict: str) -> None:
    """Print a verdict with appropriate styling."""
    style_map = {
        "regression": "[red bold]⚠ REGRESSION DETECTED[/red bold]",
        "improvement": "[green bold]✓ IMPROVEMENT[/green bold]",
        "no_change": "[dim]— No significant change[/dim]",
        "neutral": "[dim]— No significant change[/dim]",
        "inconclusive": "[yellow]? Inconclusive[/yellow]",
    }
    console.print(style_map.get(verdict, f"Verdict: {verdict}"))


def _print_repair_result(console: Any, repair: Any) -> None:
    """Print repair synthesis results."""
    from rich.table import Table

    console.print("[bold]Repair Suggestions[/bold]")

    if hasattr(repair, "candidates"):
        candidates = repair.candidates
    elif isinstance(repair, dict):
        candidates = repair.get("candidates", [])
    else:
        candidates = []

    if not candidates:
        console.print("  No repairs found")
        return

    table = Table()
    table.add_column("#", justify="right")
    table.add_column("Mutations")
    table.add_column("Cost Reduction", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Status")

    for i, c in enumerate(candidates[:10], 1):
        if hasattr(c, "n_mutations"):
            n = c.n_mutations
            cr = f"{c.expected_cost_reduction:.3f}"
            conf = f"{c.confidence:.2f}"
            status = c.verification_status
        else:
            n = len(c.get("mutations", []))
            cr = f"{c.get('expected_cost_reduction', 0):.3f}"
            conf = f"{c.get('confidence', 0):.2f}"
            status = c.get("verification_status", "?")

        table.add_row(str(i), str(n), cr, conf, status)

    console.print(table)


# ---------------------------------------------------------------------------
# Comparison formatting
# ---------------------------------------------------------------------------

def format_comparison(result: Any) -> str:
    """Format a comparison result as a rich table string."""
    try:
        from rich.console import Console
        from rich.table import Table
        from io import StringIO

        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=100)

        if isinstance(result, dict):
            details = result.get("details", {})
        elif hasattr(result, "details"):
            details = result.details
        else:
            return str(result)

        table = Table(title="Comparison Details")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        for key, value in details.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))

        console.print(table)
        return buf.getvalue()

    except ImportError:
        return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Bottleneck formatting
# ---------------------------------------------------------------------------

def format_bottlenecks(bottlenecks: list[Any]) -> str:
    """Format bottleneck list with coloured severity indicators."""
    try:
        from rich.console import Console
        from rich.table import Table
        from io import StringIO

        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=100)

        if not bottlenecks:
            console.print("[green]No bottlenecks detected[/green]")
            return buf.getvalue()

        table = Table(title=f"Bottlenecks ({len(bottlenecks)})")
        table.add_column("#", justify="right")
        table.add_column("Type", style="cyan")
        table.add_column("Severity")
        table.add_column("State")
        table.add_column("Description")

        severity_colors = {
            "critical": "red bold",
            "high": "red",
            "medium": "yellow",
            "low": "dim",
            "info": "dim italic",
        }

        for i, bn in enumerate(bottlenecks, 1):
            if isinstance(bn, dict):
                bn_type = bn.get("bottleneck_type", "?")
                severity = bn.get("severity", "medium")
                state = bn.get("state_id", "?")
                desc = bn.get("description", "")
            else:
                bn_type = getattr(bn, "bottleneck_type", "?")
                severity = getattr(bn, "severity", "medium")
                state = getattr(bn, "state_id", "?")
                desc = getattr(bn, "description", "")

            style = severity_colors.get(severity, "")
            sev_str = f"[{style}]{severity}[/{style}]" if style else severity

            table.add_row(
                str(i), bn_type, sev_str, state, desc[:60],
            )

        console.print(table)
        return buf.getvalue()

    except ImportError:
        lines = ["Bottlenecks:"]
        for i, bn in enumerate(bottlenecks, 1):
            if isinstance(bn, dict):
                lines.append(f"  {i}. {bn.get('bottleneck_type', '?')} ({bn.get('severity', '?')})")
            else:
                lines.append(f"  {i}. {getattr(bn, 'bottleneck_type', '?')}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cost table formatting
# ---------------------------------------------------------------------------

def format_cost_table(costs: dict[str, Any]) -> str:
    """Format cognitive costs as a table."""
    try:
        from rich.console import Console
        from rich.table import Table
        from io import StringIO

        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=100)

        node_costs = costs.get("node_costs", {})
        if not node_costs:
            return "No cost data"

        table = Table(title="Cognitive Cost Breakdown")
        table.add_column("Node", style="cyan")
        table.add_column("Fitts (s)", justify="right")
        table.add_column("Hick (s)", justify="right")
        table.add_column("Total (s)", justify="right")

        for nid, nc in sorted(node_costs.items()):
            if isinstance(nc, dict):
                table.add_row(
                    nid,
                    f"{nc.get('fitts', 0):.3f}",
                    f"{nc.get('hick', 0):.3f}",
                    f"{nc.get('total', 0):.3f}",
                )

        console.print(table)
        return buf.getvalue()

    except ImportError:
        lines = ["Cost Breakdown:"]
        for nid, nc in sorted(costs.get("node_costs", {}).items()):
            if isinstance(nc, dict):
                lines.append(f"  {nid}: {nc.get('total', 0):.3f}s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON formatting
# ---------------------------------------------------------------------------

def _format_json(result: Any) -> str:
    """Serialise result as JSON."""
    if hasattr(result, "to_dict"):
        data = result.to_dict()
    elif isinstance(result, dict):
        data = result
    else:
        data = {"result": str(result)}

    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# SARIF formatting
# ---------------------------------------------------------------------------

def _format_sarif(result: Any) -> str:
    """Format result as SARIF (Static Analysis Results Interchange Format)."""
    sarif: dict[str, Any] = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "usability-oracle",
                        "version": "0.1.0",
                        "informationUri": "https://github.com/usability-oracle",
                        "rules": [],
                    }
                },
                "results": _sarif_results(result),
            }
        ],
    }
    return json.dumps(sarif, indent=2)


def _sarif_results(result: Any) -> list[dict[str, Any]]:
    """Extract SARIF result entries from the pipeline result."""
    results: list[dict[str, Any]] = []

    # Extract bottlenecks as results
    bottlenecks: list[Any] = []
    if hasattr(result, "stages"):
        bn_stage = result.stages.get("bottleneck")
        if bn_stage and hasattr(bn_stage, "output"):
            bottlenecks = bn_stage.output or []
    elif isinstance(result, list):
        bottlenecks = result

    severity_map = {
        "critical": "error",
        "high": "error",
        "medium": "warning",
        "low": "note",
        "info": "note",
    }

    for i, bn in enumerate(bottlenecks):
        if isinstance(bn, dict):
            bn_type = bn.get("bottleneck_type", "usability-issue")
            severity = bn.get("severity", "medium")
            desc = bn.get("description", "")
        else:
            bn_type = getattr(bn, "bottleneck_type", "usability-issue")
            severity = getattr(bn, "severity", "medium")
            desc = getattr(bn, "description", "")

        results.append({
            "ruleId": f"usability/{bn_type}",
            "level": severity_map.get(severity, "warning"),
            "message": {"text": desc or f"Usability issue: {bn_type}"},
        })

    return results


# ---------------------------------------------------------------------------
# HTML formatting
# ---------------------------------------------------------------------------

def _format_html(result: Any) -> str:
    """Format result as a standalone HTML report."""
    json_data = _format_json(result)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Usability Oracle Report</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; }}
h1 {{ color: #1a73e8; }}
pre {{ background: #f5f5f5; padding: 1em; border-radius: 4px; overflow-x: auto; }}
.pass {{ color: #0d8043; }}
.fail {{ color: #d93025; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f0f0f0; }}
</style>
</head>
<body>
<h1>Usability Oracle Report</h1>
<pre>{json_data}</pre>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Plain text fallback
# ---------------------------------------------------------------------------

def _format_plain(result: Any) -> str:
    """Plain text formatting when rich is unavailable."""
    lines: list[str] = ["=== Usability Oracle Report ==="]

    if hasattr(result, "success"):
        lines.append(f"Status: {'Success' if result.success else 'Failed'}")

    if hasattr(result, "timing"):
        lines.append(f"Total time: {result.timing.get('total', 0):.3f}s")

    if hasattr(result, "stages"):
        lines.append("\nStages:")
        for name, sr in result.stages.items():
            status = "OK" if sr.success else "FAIL"
            lines.append(f"  {name}: {status} ({sr.timing:.3f}s)")

    if hasattr(result, "errors") and result.errors:
        lines.append("\nErrors:")
        for err in result.errors:
            lines.append(f"  - {err}")

    return "\n".join(lines)
