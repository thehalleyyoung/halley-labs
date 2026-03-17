"""
taintflow.report.text_report – Terminal / plain-text report generation.

Produces human-readable output with ANSI colour codes for terminal display.
Supports verbose and compact modes, width-aware formatting, and automatic
ANSI-stripping when output is redirected to a file.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    IO,
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


# ===================================================================
#  ANSI escape codes
# ===================================================================

class _Ansi:
    """ANSI escape-code constants for terminal colouring."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Foreground colours
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Background colours
    BG_RED = "\033[101m"
    BG_GREEN = "\033[102m"
    BG_YELLOW = "\033[103m"


_SEVERITY_ANSI: Dict[str, str] = {
    "negligible": _Ansi.GREEN,
    "warning": _Ansi.YELLOW,
    "critical": _Ansi.RED,
}


def _severity_tag(severity: Severity, use_color: bool) -> str:
    """Format a severity value as a coloured tag string."""
    label = severity.value.upper()
    if not use_color:
        return f"[{label}]"
    color = _SEVERITY_ANSI.get(severity.value, "")
    return f"{color}{_Ansi.BOLD}[{label}]{_Ansi.RESET}"


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    import re
    return re.sub(r"\033\[[0-9;]*m", "", text)


def _visible_len(text: str) -> int:
    """Length of *text* ignoring ANSI escape codes."""
    return len(_strip_ansi(text))


# ===================================================================
#  Banner
# ===================================================================

_BANNER = r"""
  _____     _       _   _____ _
 |_   _|_ _(_)_ __ | |_|  ___| | _____      __
   | |/ _` | | '_ \| __| |_  | |/ _ \ \ /\ / /
   | | (_| | | | | | |_|  _| | | (_) \ V  V /
   |_|\__,_|_|_| |_|\__|_|   |_|\___/ \_/\_/
"""


def _render_banner(use_color: bool) -> str:
    """Return the TaintFlow ASCII banner."""
    if use_color:
        return f"{_Ansi.CYAN}{_Ansi.BOLD}{_BANNER}{_Ansi.RESET}"
    return _BANNER


# ===================================================================
#  Table formatting utilities
# ===================================================================

def _build_table(
    headers: List[str],
    rows: List[List[str]],
    *,
    alignments: Optional[List[str]] = None,
    max_width: int = 120,
    use_color: bool = True,
) -> str:
    """Build a formatted text table with aligned columns and borders.

    Parameters
    ----------
    headers : list of str
        Column header labels.
    rows : list of list of str
        Row data (may contain ANSI codes).
    alignments : list of str, optional
        Per-column alignment: ``'l'`` (left), ``'r'`` (right), ``'c'`` (center).
        Defaults to left-aligned.
    max_width : int
        Maximum table width.  Columns are truncated to fit.
    use_color : bool
        Whether ANSI codes are present (affects width calculations).

    Returns
    -------
    str
        The formatted table.
    """
    n_cols = len(headers)
    if alignments is None:
        alignments = ["l"] * n_cols

    # Compute column widths based on visible content length
    col_widths = [_visible_len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < n_cols:
                col_widths[i] = max(col_widths[i], _visible_len(cell))

    # Shrink columns to fit max_width (borders + padding)
    overhead = 1 + n_cols * 3 + 1  # "| " per col + trailing " |"
    available = max_width - overhead
    if available > 0 and sum(col_widths) > available:
        scale = available / max(sum(col_widths), 1)
        col_widths = [max(4, int(w * scale)) for w in col_widths]

    def _pad(text: str, width: int, align: str) -> str:
        vis = _visible_len(text)
        pad_needed = max(0, width - vis)
        if align == "r":
            return " " * pad_needed + text
        if align == "c":
            left = pad_needed // 2
            right = pad_needed - left
            return " " * left + text + " " * right
        return text + " " * pad_needed

    def _truncate(text: str, width: int) -> str:
        if _visible_len(text) <= width:
            return text
        # Truncate visible chars, preserve any trailing reset
        stripped = _strip_ansi(text)
        truncated = stripped[: max(0, width - 1)] + "…"
        return truncated

    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    lines: List[str] = [separator]

    # Header row
    header_cells = [
        " " + _pad(_truncate(h, col_widths[i]), col_widths[i], "l") + " "
        for i, h in enumerate(headers)
    ]
    if use_color:
        header_line = f"|{_Ansi.BOLD}" + "|".join(header_cells) + f"{_Ansi.RESET}|"
    else:
        header_line = "|" + "|".join(header_cells) + "|"
    lines.append(header_line)
    lines.append(separator.replace("-", "="))

    # Data rows
    for row in rows:
        cells: List[str] = []
        for i in range(n_cols):
            cell = row[i] if i < len(row) else ""
            cell = _truncate(cell, col_widths[i])
            padded = _pad(cell, col_widths[i], alignments[i] if i < len(alignments) else "l")
            cells.append(" " + padded + " ")
        lines.append("|" + "|".join(cells) + "|")

    lines.append(separator)
    return "\n".join(lines)


# ===================================================================
#  Progress bar
# ===================================================================


def render_progress_bar(
    current: int,
    total: int,
    *,
    width: int = 40,
    label: str = "",
    use_color: bool = True,
) -> str:
    """Render a text progress bar.

    Parameters
    ----------
    current : int
        Current step number.
    total : int
        Total number of steps.
    width : int
        Bar width in characters.
    label : str
        Optional label prefix.
    use_color : bool
        Use ANSI colour for the bar.

    Returns
    -------
    str
        A single-line progress bar string (no trailing newline).
    """
    if total <= 0:
        ratio = 1.0
    else:
        ratio = min(1.0, max(0.0, current / total))
    filled = int(width * ratio)
    bar = "█" * filled + "░" * (width - filled)
    pct = f"{ratio * 100:5.1f}%"

    if use_color:
        if ratio >= 1.0:
            color = _Ansi.GREEN
        elif ratio >= 0.5:
            color = _Ansi.YELLOW
        else:
            color = _Ansi.BLUE
        bar_str = f"{color}{bar}{_Ansi.RESET}"
    else:
        bar_str = bar

    prefix = f"{label} " if label else ""
    return f"\r{prefix}|{bar_str}| {pct} ({current}/{total})"


# ===================================================================
#  TextReportGenerator
# ===================================================================


@dataclass
class TextReportGenerator:
    """Generate terminal / plain-text audit reports.

    Parameters
    ----------
    config : TaintFlowConfig, optional
        Audit configuration for threshold display.
    verbose : bool
        Verbose mode: show all features and per-stage details.
    compact : bool
        Compact mode: summary line only.  Overrides *verbose*.
    use_color : bool, optional
        Force colour on/off.  ``None`` auto-detects based on terminal.
    max_width : int, optional
        Override terminal width detection.
    show_banner : bool
        Print the TaintFlow ASCII banner.
    top_bottlenecks : int
        How many bottleneck stages to show.
    """

    config: Optional[TaintFlowConfig] = None
    verbose: bool = False
    compact: bool = False
    use_color: Optional[bool] = None
    max_width: Optional[int] = None
    show_banner: bool = True
    top_bottlenecks: int = 5

    _color: bool = field(init=False, default=True)
    _width: int = field(init=False, default=120)

    def __post_init__(self) -> None:
        # Determine colour support
        if self.use_color is not None:
            self._color = self.use_color
        else:
            self._color = _is_tty()

        # Determine terminal width
        if self.max_width is not None:
            self._width = self.max_width
        else:
            self._width = _detect_terminal_width()

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------

    def generate(self, report: LeakageReport) -> str:
        """Return the full text report as a string."""
        if self.compact:
            return self._generate_compact(report)
        parts: List[str] = []
        if self.show_banner:
            parts.append(_render_banner(self._color))
        parts.append(self._render_summary_line(report))
        parts.append("")
        parts.append(self._render_feature_table(report))
        parts.append("")
        parts.append(self._render_bottlenecks(report))
        if self.verbose:
            parts.append("")
            parts.append(self._render_stage_details(report))
        parts.append("")
        parts.append(self._render_remediations(report))
        parts.append("")
        parts.append(self._render_footer())
        return "\n".join(parts) + "\n"

    def generate_to_file(self, report: LeakageReport, path: str) -> None:
        """Write the text report to *path* (ANSI codes stripped)."""
        content = self.generate(report)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_strip_ansi(content))

    def generate_to_stream(
        self,
        report: LeakageReport,
        stream: IO[str],
    ) -> None:
        """Write the text report to an open text stream.

        ANSI codes are stripped automatically if the stream is not a TTY.
        """
        content = self.generate(report)
        if not _stream_is_tty(stream):
            content = _strip_ansi(content)
        stream.write(content)

    # -----------------------------------------------------------------
    #  Compact mode
    # -----------------------------------------------------------------

    def _generate_compact(self, report: LeakageReport) -> str:
        tag = _severity_tag(report.overall_severity, self._color)
        return (
            f"{tag} Found {report.total_bit_bound:.2f} bits of leakage "
            f"across {report.n_features} features in {report.n_stages} stages\n"
        )

    # -----------------------------------------------------------------
    #  Section renderers
    # -----------------------------------------------------------------

    def _render_summary_line(self, report: LeakageReport) -> str:
        tag = _severity_tag(report.overall_severity, self._color)
        line = (
            f"{tag} Found {report.total_bit_bound:.2f} bits of leakage "
            f"across {report.n_features} features in {report.n_stages} stages"
        )
        details: List[str] = [
            f"Pipeline: {report.pipeline_name}",
            f"Leaking features: {report.n_leaking_features}/{report.n_features}",
        ]
        if report.analysis_duration_ms > 0:
            secs = report.analysis_duration_ms / 1000.0
            details.append(f"Analysis time: {secs:.1f}s")

        sep = "─" * min(self._width, 80)
        if self._color:
            sep = f"{_Ansi.DIM}{sep}{_Ansi.RESET}"

        detail_str = " · ".join(details)
        return f"{sep}\n{line}\n{detail_str}\n{sep}"

    def _render_feature_table(self, report: LeakageReport) -> str:
        header = self._section_header("Per-Feature Leakage")
        all_features: List[tuple] = []
        for sl in report.stage_leakages:
            for fl in sl.feature_leakages:
                all_features.append((sl.stage_name, fl))
        all_features.sort(key=lambda x: (-x[1].bit_bound, x[1].column_name))

        if not all_features:
            return f"{header}\n  No features analysed."

        headers = ["Feature", "Bits", "Severity", "Stage", "Confidence"]
        rows: List[List[str]] = []
        for stage_name, fl in all_features:
            sev_str = _severity_tag(fl.severity, self._color)
            rows.append([
                fl.column_name,
                f"{fl.bit_bound:.4f}",
                sev_str,
                stage_name,
                f"{fl.confidence:.0%}",
            ])

        table = _build_table(
            headers,
            rows,
            alignments=["l", "r", "l", "l", "r"],
            max_width=self._width,
            use_color=self._color,
        )
        return f"{header}\n{table}"

    def _render_bottlenecks(self, report: LeakageReport) -> str:
        header = self._section_header(f"Top-{self.top_bottlenecks} Bottleneck Stages")
        stages = report.stages_by_severity()[: self.top_bottlenecks]
        if not stages:
            return f"{header}\n  No stages to report."

        lines: List[str] = [header]
        for i, sl in enumerate(stages, start=1):
            tag = _severity_tag(sl.severity, self._color)
            lines.append(
                f"  {i}. {tag} {sl.stage_name} "
                f"({sl.max_bit_bound:.2f} bits, "
                f"{sl.n_leaking_features} leaking features)"
            )
        return "\n".join(lines)

    def _render_stage_details(self, report: LeakageReport) -> str:
        """Verbose per-stage breakdown."""
        header = self._section_header("Per-Stage Details")
        parts: List[str] = [header]

        for sl in report.stages_by_severity():
            tag = _severity_tag(sl.severity, self._color)
            parts.append(
                f"\n  {tag} {sl.stage_name} ({sl.stage_id})"
            )
            parts.append(f"    Op: {sl.op_type.value} | Kind: {sl.node_kind.value}")
            parts.append(
                f"    Max: {sl.max_bit_bound:.4f} bits | "
                f"Mean: {sl.mean_bit_bound:.4f} bits | "
                f"Leaking: {sl.n_leaking_features}"
            )
            if sl.description:
                parts.append(f"    Description: {sl.description}")

            if sl.feature_leakages:
                feat_headers = ["Column", "Bits", "Severity"]
                feat_rows: List[List[str]] = []
                for fl in sorted(sl.feature_leakages, key=lambda f: -f.bit_bound):
                    feat_rows.append([
                        fl.column_name,
                        f"{fl.bit_bound:.4f}",
                        _severity_tag(fl.severity, self._color),
                    ])
                table = _build_table(
                    feat_headers,
                    feat_rows,
                    alignments=["l", "r", "l"],
                    max_width=self._width - 4,
                    use_color=self._color,
                )
                for line in table.split("\n"):
                    parts.append("    " + line)

        return "\n".join(parts)

    def _render_remediations(self, report: LeakageReport) -> str:
        header = self._section_header("Remediation Suggestions")
        suggestions: List[str] = []
        seen: set = set()

        for sl in report.stages_by_severity():
            for fl in sl.feature_leakages:
                if fl.remediation and fl.remediation not in seen:
                    seen.add(fl.remediation)
                    tag = _severity_tag(fl.severity, self._color)
                    suggestions.append(
                        f"  {tag} {fl.column_name} in {sl.stage_name}:\n"
                        f"    → {fl.remediation}"
                    )

        if not suggestions:
            return (
                f"{header}\n"
                f"  No specific remediation suggestions."
            )

        return f"{header}\n" + "\n".join(suggestions)

    def _render_footer(self) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        line = f"TaintFlow v{_TOOL_VERSION} · {now}"
        if self._color:
            return f"{_Ansi.DIM}{line}{_Ansi.RESET}"
        return line

    # -----------------------------------------------------------------
    #  Helpers
    # -----------------------------------------------------------------

    def _section_header(self, title: str) -> str:
        if self._color:
            return f"{_Ansi.BOLD}{_Ansi.UNDERLINE}{title}{_Ansi.RESET}"
        return f"{title}\n{'=' * len(title)}"


# ===================================================================
#  Terminal detection helpers
# ===================================================================

def _is_tty() -> bool:
    """Return ``True`` if stdout is connected to a terminal."""
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _stream_is_tty(stream: IO[str]) -> bool:
    """Return ``True`` if *stream* looks like a TTY."""
    try:
        return hasattr(stream, "isatty") and stream.isatty()
    except Exception:
        return False


def _detect_terminal_width() -> int:
    """Detect the current terminal width, falling back to 120."""
    try:
        cols = shutil.get_terminal_size((120, 24)).columns
        return max(40, cols)
    except Exception:
        return 120


# ===================================================================
#  Convenience function
# ===================================================================


def generate_text_report(
    report: LeakageReport,
    config: Optional[TaintFlowConfig] = None,
    path: Optional[str] = None,
    verbose: bool = False,
    compact: bool = False,
    **kwargs: Any,
) -> str:
    """One-shot helper: build a text report and optionally write to *path*.

    Parameters
    ----------
    report : LeakageReport
        The audit result to render.
    config : TaintFlowConfig, optional
        Audit configuration for threshold display.
    path : str, optional
        If given, write the report to this file (ANSI codes stripped).
    verbose : bool
        Include per-stage verbose details.
    compact : bool
        Compact single-line summary only.
    **kwargs
        Forwarded to :class:`TextReportGenerator`.

    Returns
    -------
    str
        The text report string (with ANSI codes if colour enabled).
    """
    gen = TextReportGenerator(
        config=config,
        verbose=verbose,
        compact=compact,
        **kwargs,
    )
    content = gen.generate(report)
    if path is not None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_strip_ansi(content))
    return content
