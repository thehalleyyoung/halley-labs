"""
usability_oracle.output.console — Rich console formatter.

Pretty-prints the pipeline result to the terminal using the ``rich``
library.  Falls back to plain-text when rich is not available.
"""

from __future__ import annotations

import io
import sys
from typing import Any, Optional

from usability_oracle.core.enums import (
    BottleneckType,
    RegressionVerdict,
    Severity,
)
from usability_oracle.output.models import (
    BottleneckDescription,
    CostComparison,
    PipelineResult,
    StageTimingInfo,
)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.columns import Columns
    from rich.rule import Rule
    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

_SEVERITY_COLORS: dict[Severity, str] = {
    Severity.CRITICAL: "bold red",
    Severity.HIGH: "red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "dim",
}

_VERDICT_STYLES: dict[RegressionVerdict, tuple[str, str]] = {
    RegressionVerdict.REGRESSION: ("bold white on red", "⚠ REGRESSION"),
    RegressionVerdict.IMPROVEMENT: ("bold white on green", "✓ IMPROVEMENT"),
    RegressionVerdict.NEUTRAL: ("bold white on blue", "— NO CHANGE"),
    RegressionVerdict.INCONCLUSIVE: ("bold white on yellow", "? INCONCLUSIVE"),
}


def _severity_color(severity: Severity) -> str:
    return _SEVERITY_COLORS.get(severity, "dim")


def _fmt_secs(v: float) -> str:
    if v < 1.0:
        return f"{v * 1000:.1f} ms"
    return f"{v:.3f} s"


# ---------------------------------------------------------------------------
# ConsoleFormatter
# ---------------------------------------------------------------------------

class ConsoleFormatter:
    """Print a :class:`PipelineResult` to the console via *rich*.

    Falls back to a plain-text representation if *rich* is not installed.
    """

    def __init__(
        self,
        console: Optional[Any] = None,
        width: Optional[int] = None,
        force_terminal: bool = False,
    ) -> None:
        if _HAS_RICH:
            self._console = console or Console(
                width=width, force_terminal=force_terminal
            )
        else:
            self._console = None
        self._width = width

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format(self, result: PipelineResult) -> None:
        """Print *result* to the console."""
        if not _HAS_RICH or self._console is None:
            self._format_plain(result)
            return
        c = self._console
        c.print()
        c.print(Rule("[bold]Usability Oracle Results[/bold]"))
        c.print()
        c.print(self._verdict_panel(result.verdict))
        c.print()

        if result.comparison:
            c.print(self._cost_table(result.comparison))
            c.print()

        if result.bottlenecks:
            c.print(self._bottleneck_tree(result.bottlenecks))
            c.print()

        if result.recommendations:
            self._print_recommendations(result.recommendations)
            c.print()

        if result.timing:
            self._print_timing(result.timing)
            c.print()

        c.print(Rule())

    def format_to_string(self, result: PipelineResult) -> str:
        """Render the console output into a plain string (for testing)."""
        if not _HAS_RICH:
            buf = io.StringIO()
            self._format_plain(result, file=buf)
            return buf.getvalue()
        string_console = Console(file=io.StringIO(), width=self._width or 100)
        old = self._console
        self._console = string_console
        self.format(result)
        self._console = old
        assert isinstance(string_console.file, io.StringIO)
        return string_console.file.getvalue()

    # ------------------------------------------------------------------
    # Verdict panel
    # ------------------------------------------------------------------

    @staticmethod
    def _verdict_panel(verdict: RegressionVerdict) -> Panel:
        style, label = _VERDICT_STYLES.get(
            verdict, ("bold", verdict.value.upper())
        )
        text = Text(label, style=style, justify="center")
        return Panel(text, title="Verdict", border_style="bold", expand=False, padding=(1, 4))

    # ------------------------------------------------------------------
    # Cost table
    # ------------------------------------------------------------------

    @staticmethod
    def _cost_table(comparison: CostComparison) -> Table:
        table = Table(title="Cost Comparison", show_header=True, header_style="bold magenta")
        table.add_column("Channel", style="bold")
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")
        table.add_column("Delta", justify="right")

        cost_a_d = comparison.cost_a.to_dict()
        cost_b_d = comparison.cost_b.to_dict()
        delta_d = comparison.delta.to_dict()
        for channel in cost_a_d:
            va = cost_a_d[channel]
            vb = cost_b_d.get(channel, 0)
            vd = delta_d.get(channel, 0)
            delta_style = "red" if vd > 0 else "green" if vd < 0 else "dim"
            table.add_row(
                str(channel).title(),
                f"{va:.4f}",
                f"{vb:.4f}",
                Text(f"{vd:+.4f}", style=delta_style),
            )

        # Total row
        ta = comparison.cost_a.total_weighted_cost
        tb = comparison.cost_b.total_weighted_cost
        td = comparison.delta.total_weighted_cost
        pct = comparison.percentage_change
        delta_style = "bold red" if td > 0 else "bold green" if td < 0 else "bold"
        table.add_row(
            Text("Total (weighted)", style="bold"),
            Text(f"{ta:.4f}", style="bold"),
            Text(f"{tb:.4f}", style="bold"),
            Text(f"{td:+.4f} ({pct:+.1f}%)", style=delta_style),
            end_section=True,
        )
        return table

    # ------------------------------------------------------------------
    # Bottleneck tree
    # ------------------------------------------------------------------

    @staticmethod
    def _bottleneck_tree(bottlenecks: list[BottleneckDescription]) -> Tree:
        tree = Tree("[bold]Bottlenecks[/bold]")
        for b in bottlenecks:
            sev_style = _severity_color(b.severity)
            label = f"[{sev_style}]{b.bottleneck_type.value}[/{sev_style}] [{sev_style}]({b.severity.value})[/{sev_style}]"
            branch = tree.add(label)
            branch.add(f"[dim]Description:[/dim] {b.description}")
            branch.add(f"[dim]Cost impact:[/dim] {b.cost_impact:+.4f}")
            if b.affected_elements:
                elems = ", ".join(b.affected_elements[:8])
                branch.add(f"[dim]Affected:[/dim] {elems}")
            if b.recommendation:
                branch.add(f"[green]Recommendation:[/green] {b.recommendation}")
        return tree

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _print_recommendations(self, recommendations: list[str]) -> None:
        c = self._console
        assert c is not None
        c.print("[bold]Recommendations[/bold]")
        for i, rec in enumerate(recommendations, 1):
            c.print(f"  [cyan]{i}.[/cyan] {rec}")

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _print_timing(self, timing: list[StageTimingInfo]) -> None:
        c = self._console
        assert c is not None
        table = Table(title="Pipeline Timing", show_header=True, header_style="bold cyan")
        table.add_column("Stage", style="bold")
        table.add_column("Elapsed", justify="right")
        table.add_column("% of Total", justify="right")
        total = sum(t.elapsed_seconds for t in timing) or 1e-9
        for t in timing:
            pct = (t.elapsed_seconds / total) * 100
            bar_len = int(pct / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            table.add_row(
                t.stage.value,
                _fmt_secs(t.elapsed_seconds),
                f"{pct:.1f}% {bar}",
            )
        table.add_row(
            Text("Total", style="bold"),
            Text(_fmt_secs(total), style="bold"),
            Text("100.0%", style="bold"),
            end_section=True,
        )
        c.print(table)

    # ------------------------------------------------------------------
    # Progress bar (for live pipeline execution)
    # ------------------------------------------------------------------

    @staticmethod
    def _progress_bar(stages: list[str]) -> "Progress":
        """Create a rich Progress bar for pipeline stage execution."""
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        for stage in stages:
            progress.add_task(stage, total=1.0)
        return progress

    # ------------------------------------------------------------------
    # Plain text fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _format_plain(result: PipelineResult, file: Any = None) -> None:
        out = file or sys.stdout
        print("=" * 60, file=out)
        print("  Usability Oracle Results", file=out)
        print("=" * 60, file=out)
        print(f"\nVerdict: {result.verdict.value.upper()}", file=out)

        if result.comparison:
            comp = result.comparison
            print("\n--- Cost Comparison ---", file=out)
            cost_a_d = comp.cost_a.to_dict()
            cost_b_d = comp.cost_b.to_dict()
            delta_d = comp.delta.to_dict()
            for ch in cost_a_d:
                va = cost_a_d[ch]
                vb = cost_b_d.get(ch, 0)
                vd = delta_d.get(ch, 0)
                print(f"  {str(ch):12s}  {va:.4f} -> {vb:.4f}  (Δ {vd:+.4f})", file=out)
            print(
                f"  {'total(w)':12s}  {comp.cost_a.total_weighted_cost:.4f} -> {comp.cost_b.total_weighted_cost:.4f}"
                f"  (Δ {comp.delta.total_weighted_cost:+.4f}, {comp.percentage_change:+.1f}%)",
                file=out,
            )

        if result.bottlenecks:
            print("\n--- Bottlenecks ---", file=out)
            for b in result.bottlenecks:
                print(f"  [{b.severity.value}] {b.bottleneck_type.value}", file=out)
                print(f"    {b.description}", file=out)
                if b.recommendation:
                    print(f"    -> {b.recommendation}", file=out)

        if result.recommendations:
            print("\n--- Recommendations ---", file=out)
            for i, r in enumerate(result.recommendations, 1):
                print(f"  {i}. {r}", file=out)

        if result.timing:
            total = sum(t.elapsed_seconds for t in result.timing) or 1e-9
            print("\n--- Timing ---", file=out)
            for t in result.timing:
                pct = (t.elapsed_seconds / total) * 100
                print(f"  {t.stage.value:15s}  {_fmt_secs(t.elapsed_seconds):>10s}  ({pct:.1f}%)", file=out)
            print(f"  {'total':15s}  {_fmt_secs(total):>10s}", file=out)

        print("\n" + "=" * 60, file=out)
