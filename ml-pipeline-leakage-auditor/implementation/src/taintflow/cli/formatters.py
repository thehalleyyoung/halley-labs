"""TaintFlow CLI – output formatters.

Provides composable formatter classes for rendering analysis results in
terminals, files, and structured formats (JSON, SARIF).  All formatters
accept an output *stream* (defaulting to :data:`sys.stdout`) and expose a
simple ``write(...)`` method.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, TextIO

# ---------------------------------------------------------------------------
# ANSI colour support
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


class ColorFormatter:
    """ANSI colour support with automatic terminal detection.

    When the output *stream* is **not** a TTY (or ``NO_COLOR`` is set)
    all colour methods become identity functions.
    """

    RESET = "\033[0m"
    CODES: Dict[str, str] = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "underline": "\033[4m",
    }

    def __init__(self, *, stream: TextIO = sys.stdout) -> None:
        self.stream = stream
        self.enabled = self._detect(stream)

    @staticmethod
    def _detect(stream: TextIO) -> bool:
        """Return *True* if the stream supports ANSI colour codes."""
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("FORCE_COLOR"):
            return True
        return hasattr(stream, "isatty") and stream.isatty()

    def wrap(self, text: str, *codes: str) -> str:
        """Wrap *text* in the given ANSI *codes*.

        Parameters
        ----------
        text:
            The string to wrap.
        *codes:
            One or more code names recognised by :attr:`CODES`.

        Returns
        -------
        str
            *text* surrounded by ANSI escape sequences (or unchanged if
            colour is disabled).
        """
        if not self.enabled:
            return text
        prefix = "".join(self.CODES.get(c, "") for c in codes)
        return f"{prefix}{text}{self.RESET}" if prefix else text

    def red(self, text: str) -> str:
        """Wrap *text* in red."""
        return self.wrap(text, "red")

    def green(self, text: str) -> str:
        """Wrap *text* in green."""
        return self.wrap(text, "green")

    def yellow(self, text: str) -> str:
        """Wrap *text* in yellow."""
        return self.wrap(text, "yellow")

    def blue(self, text: str) -> str:
        """Wrap *text* in blue."""
        return self.wrap(text, "blue")

    def bold(self, text: str) -> str:
        """Wrap *text* in bold."""
        return self.wrap(text, "bold")

    def dim(self, text: str) -> str:
        """Wrap *text* in dim."""
        return self.wrap(text, "dim")


# ---------------------------------------------------------------------------
# ANSI stripper
# ---------------------------------------------------------------------------


class ANSIStripper:
    """Strip ANSI escape codes from text.

    Useful when redirecting coloured output to a file.
    """

    @staticmethod
    def strip(text: str) -> str:
        """Remove all ANSI escape sequences from *text*."""
        return _ANSI_RE.sub("", text)

    @classmethod
    def strip_lines(cls, lines: Sequence[str]) -> List[str]:
        """Strip ANSI codes from every line in *lines*."""
        return [cls.strip(line) for line in lines]


# ---------------------------------------------------------------------------
# Severity formatter
# ---------------------------------------------------------------------------


class SeverityFormatter:
    """Colour-code severity labels for terminal display."""

    def __init__(self, *, color: Optional[ColorFormatter] = None) -> None:
        self.color = color or ColorFormatter()

    def format(self, severity: str) -> str:
        """Return a colour-coded severity string."""
        lower = severity.lower()
        if lower == "critical":
            return self.color.red(severity)
        if lower == "warning":
            return self.color.yellow(severity)
        return self.color.green(severity)


# ---------------------------------------------------------------------------
# Terminal formatter
# ---------------------------------------------------------------------------


class TerminalFormatter:
    """Low-level helpers for writing styled text to a terminal stream."""

    def __init__(self, *, stream: TextIO = sys.stdout) -> None:
        self.stream = stream
        self.color = ColorFormatter(stream=stream)
        self.severity = SeverityFormatter(color=self.color)

    def header(self, text: str, *, char: str = "─") -> None:
        """Write a section header with an underline of *char*."""
        self.stream.write(f"\n{self.color.bold(text)}\n")
        visible_len = len(ANSIStripper.strip(text))
        self.stream.write(self.color.dim(char * visible_len) + "\n")
        self.stream.flush()

    def write_line(self, text: str) -> None:
        """Write a single line followed by a newline."""
        self.stream.write(text + "\n")
        self.stream.flush()

    def blank(self) -> None:
        """Write an empty line."""
        self.stream.write("\n")
        self.stream.flush()


# ---------------------------------------------------------------------------
# Table formatter
# ---------------------------------------------------------------------------


class TableFormatter:
    """Render a list of rows as an aligned ASCII table.

    Supports optional headers and automatic column-width calculation.
    """

    def __init__(
        self,
        *,
        stream: TextIO = sys.stdout,
        padding: int = 2,
        separator: str = " ",
    ) -> None:
        self.stream = stream
        self.padding = padding
        self.separator = separator
        self._color = ColorFormatter(stream=stream)

    def write(
        self,
        rows: List[List[str]],
        *,
        headers: Optional[List[str]] = None,
    ) -> None:
        """Write *rows* (and optional *headers*) as an aligned table.

        Parameters
        ----------
        rows:
            List of rows, each a list of cell strings.
        headers:
            Optional column header strings.
        """
        all_rows: List[List[str]] = []
        if headers:
            all_rows.append(headers)
        all_rows.extend(rows)

        if not all_rows:
            return

        n_cols = max(len(r) for r in all_rows)

        # Calculate column widths using stripped (visible) text
        widths = [0] * n_cols
        for row in all_rows:
            for i, cell in enumerate(row):
                visible = len(ANSIStripper.strip(cell))
                widths[i] = max(widths[i], visible)

        # Render rows
        if headers:
            self._write_row(headers, widths)
            rule = self.separator.join("─" * (w + self.padding) for w in widths)
            self.stream.write(self._color.dim(rule) + "\n")

        data_rows = all_rows[1:] if headers else all_rows
        for row in data_rows:
            self._write_row(row, widths)

        self.stream.flush()

    def _write_row(self, row: List[str], widths: List[int]) -> None:
        parts: List[str] = []
        for i, cell in enumerate(row):
            visible_len = len(ANSIStripper.strip(cell))
            pad = widths[i] - visible_len + self.padding if i < len(widths) else self.padding
            parts.append(cell + " " * max(pad, 1))
        self.stream.write(self.separator.join(parts).rstrip() + "\n")


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------


class ProgressBar:
    """Render a terminal progress bar.

    Updates in-place using carriage-return when the stream is a TTY,
    otherwise falls back to simple line-by-line output.
    """

    def __init__(
        self,
        total: int,
        *,
        width: int = 40,
        label: str = "",
        stream: TextIO = sys.stderr,
    ) -> None:
        self.total = max(total, 1)
        self.current = 0
        self.width = width
        self.label = label
        self.stream = stream
        self._color = ColorFormatter(stream=stream)

    def update(self, n: int = 1, *, message: str = "") -> None:
        """Advance the progress bar by *n* steps."""
        self.current = min(self.current + n, self.total)
        self._render(message)

    def finish(self, message: str = "done") -> None:
        """Mark the bar as complete."""
        self.current = self.total
        self._render(message)
        self.stream.write("\n")
        self.stream.flush()

    def _render(self, message: str) -> None:
        fraction = self.current / self.total
        filled = int(self.width * fraction)
        bar = "█" * filled + "░" * (self.width - filled)
        pct = int(fraction * 100)

        if self._color.enabled:
            line = f"\r{self.label} [{bar}] {pct:3d}% {message}"
            self.stream.write(line)
        else:
            self.stream.write(f"[{self.current}/{self.total}] {message}\n")
        self.stream.flush()


# ---------------------------------------------------------------------------
# Tree formatter
# ---------------------------------------------------------------------------


class TreeFormatter:
    """Render a tree-style view of pipeline stages and their features."""

    BRANCH = "├── "
    LAST = "└── "
    PIPE = "│   "
    SPACE = "    "

    def __init__(self, *, stream: TextIO = sys.stdout) -> None:
        self.stream = stream
        self._color = ColorFormatter(stream=stream)
        self._severity = SeverityFormatter(color=self._color)

    def write_stage_tree(self, stage: Any) -> None:
        """Render a single stage and its features as a tree.

        Parameters
        ----------
        stage:
            A ``StageLeakage`` instance (or any object with
            ``stage_name``, ``severity``, and ``feature_leakages``).
        """
        sev = self._severity.format(stage.severity.value)
        self.stream.write(
            f"{self._color.bold(stage.stage_name)} [{sev}]\n"
        )
        features = stage.feature_leakages
        for i, feat in enumerate(features):
            is_last = i == len(features) - 1
            prefix = self.LAST if is_last else self.BRANCH
            feat_sev = self._severity.format(feat.severity.value)
            self.stream.write(
                f"{prefix}{feat.column_name}: "
                f"{feat.bit_bound:.2f} bits [{feat_sev}]"
            )
            if feat.remediation:
                continuation = self.SPACE if is_last else self.PIPE
                self.stream.write(
                    f"\n{continuation}{self._color.dim(feat.remediation)}"
                )
            self.stream.write("\n")
        self.stream.write("\n")
        self.stream.flush()

    def write_dict_tree(self, data: Dict[str, Any], *, indent: int = 0) -> None:
        """Recursively render a dictionary as a tree."""
        items = list(data.items())
        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            prefix = self.LAST if is_last else self.BRANCH
            pad = self.SPACE * indent
            if isinstance(value, dict):
                self.stream.write(f"{pad}{prefix}{self._color.bold(str(key))}\n")
                child_indent = indent + 1
                self.write_dict_tree(value, indent=child_indent)
            else:
                self.stream.write(f"{pad}{prefix}{key}: {value}\n")
        if indent == 0:
            self.stream.flush()


# ---------------------------------------------------------------------------
# Summary formatter
# ---------------------------------------------------------------------------


class SummaryFormatter:
    """Render a one-line or short summary of a leakage report."""

    def __init__(self, *, stream: TextIO = sys.stdout) -> None:
        self.stream = stream
        self._color = ColorFormatter(stream=stream)
        self._severity = SeverityFormatter(color=self._color)

    def write_report_summary(self, report: Any) -> None:
        """Write a multi-line summary block for *report*.

        Parameters
        ----------
        report:
            A :class:`LeakageReport` instance.
        """
        sev = self._severity.format(report.overall_severity.value)
        self.stream.write(f"Pipeline:    {self._color.bold(report.pipeline_name)}\n")
        self.stream.write(f"Severity:    {sev}\n")
        self.stream.write(f"Total bits:  {report.total_bit_bound:.2f}\n")
        self.stream.write(f"Stages:      {report.n_stages}\n")
        self.stream.write(f"Features:    {report.n_leaking_features}/{report.n_features} leaking\n")
        if report.analysis_duration_ms:
            self.stream.write(f"Duration:    {report.analysis_duration_ms:.0f} ms\n")
        self.stream.flush()

    def one_line(self, report: Any) -> str:
        """Return a one-line summary string (no ANSI codes)."""
        return (
            f"{report.pipeline_name}: "
            f"{report.overall_severity.value} "
            f"({report.total_bit_bound:.2f} bits, "
            f"{report.n_leaking_features}/{report.n_features} features)"
        )


# ---------------------------------------------------------------------------
# Diff formatter
# ---------------------------------------------------------------------------


class DiffFormatter:
    """Format a comparison diff between two analysis results."""

    def __init__(self, *, stream: TextIO = sys.stdout) -> None:
        self.stream = stream
        self._color = ColorFormatter(stream=stream)
        self._table = TableFormatter(stream=stream)

    def write(self, diff: Dict[str, Any]) -> None:
        """Write a formatted comparison diff to the stream.

        Parameters
        ----------
        diff:
            A dictionary as produced by
            :meth:`CompareCommand._compute_diff`.
        """
        self.stream.write(self._color.bold("Comparison Report\n"))
        self.stream.write(f"Before: {diff.get('before_pipeline', '?')}"
                          f" ({diff.get('before_severity', '?')}, "
                          f"{diff.get('before_total_bits', 0):.2f} bits)\n")
        self.stream.write(f"After:  {diff.get('after_pipeline', '?')}"
                          f" ({diff.get('after_severity', '?')}, "
                          f"{diff.get('after_total_bits', 0):.2f} bits)\n\n")

        changes = diff.get("changes", [])
        if not changes:
            self.stream.write(self._color.green("No differences found.\n"))
            self.stream.flush()
            return

        rows: List[List[str]] = []
        for item in changes:
            change = item["change"]
            delta = item["delta"]
            if change == "regression":
                symbol = self._color.red("▲ regression")
            elif change == "improvement":
                symbol = self._color.green("▼ improvement")
            elif change == "new":
                symbol = self._color.yellow("+ new")
            elif change == "fixed":
                symbol = self._color.green("- fixed")
            else:
                symbol = self._color.dim("= unchanged")
            rows.append([
                item["feature"],
                f"{item['before_bits']:.2f}",
                f"{item['after_bits']:.2f}",
                f"{delta:+.2f}",
                symbol,
            ])

        self._table.write(
            headers=["Feature", "Before", "After", "Delta", "Status"],
            rows=rows,
        )

        summary = diff.get("summary", {})
        self.stream.write("\nSummary: ")
        parts: List[str] = []
        for label, key, color_fn in [
            ("regressions", "regressions", self._color.red),
            ("improvements", "improvements", self._color.green),
            ("new", "new", self._color.yellow),
            ("fixed", "fixed", self._color.green),
            ("unchanged", "unchanged", self._color.dim),
        ]:
            count = summary.get(key, 0)
            if count:
                parts.append(color_fn(f"{count} {label}"))
        self.stream.write(", ".join(parts) if parts else "no changes")
        self.stream.write("\n")
        self.stream.flush()


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class JSONFormatter:
    """Pretty-print JSON data to a stream."""

    def __init__(
        self,
        *,
        stream: TextIO = sys.stdout,
        indent: int = 2,
        sort_keys: bool = False,
    ) -> None:
        self.stream = stream
        self.indent = indent
        self.sort_keys = sort_keys

    def write(self, data: Any) -> None:
        """Serialise *data* as indented JSON and write to the stream.

        Non-serialisable objects are converted to their ``str()``
        representation.
        """
        json.dump(
            data,
            self.stream,
            indent=self.indent,
            sort_keys=self.sort_keys,
            default=str,
        )
        self.stream.write("\n")
        self.stream.flush()
