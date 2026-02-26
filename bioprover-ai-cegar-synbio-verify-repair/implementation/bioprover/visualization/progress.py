"""Progress reporting for BioProver verification workflows.

Provides ASCII progress bars, CEGAR iteration dashboards, coverage
meters, and logging integration for both interactive and batch modes.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, TextIO

from bioprover.cegar.cegar_engine import CEGARStatistics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ANSI_CODES: Dict[str, str] = {
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}

_BAR_FILL = "█"
_BAR_PARTIAL = "▓"
_BAR_EMPTY = "░"

TRACE = 5  # custom log level below DEBUG
logging.addLevelName(TRACE, "TRACE")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ProgressConfig:
    """Configuration for progress display behaviour."""

    color: bool = True
    """Enable ANSI colour output."""

    interactive: bool = True
    """Use interactive (overwriting) output vs simple line-by-line."""

    width: int = 50
    """Character width of progress bars."""

    refresh_interval: float = 0.5
    """Minimum seconds between interactive refreshes."""


# ---------------------------------------------------------------------------
# ProgressReporter
# ---------------------------------------------------------------------------


class ProgressReporter:
    """Renders progress indicators to a text stream.

    Supports ASCII progress bars, coverage meters, iteration status lines,
    and phase headers.  Colour output degrades gracefully when the output
    stream is not a terminal or when colour is explicitly disabled.
    """

    def __init__(
        self,
        config: Optional[ProgressConfig] = None,
        stream: Optional[TextIO] = None,
    ) -> None:
        self._config = config or ProgressConfig()
        self._stream: TextIO = stream or sys.stderr
        # Override colour setting when terminal does not support it.
        if self._config.color:
            self._config.color = self._detect_color_support()

    # -- public helpers ----------------------------------------------------

    def progress_bar(
        self,
        current: int,
        total: int,
        label: str = "",
        suffix: str = "",
    ) -> str:
        """Return an ASCII progress bar string.

        Example::

            [████████░░░░░░░░] 45% uploading  ETA 12s
        """
        fraction = current / total if total > 0 else 0.0
        fraction = max(0.0, min(1.0, fraction))
        filled = int(self._config.width * fraction)
        empty = self._config.width - filled
        bar = f"[{_BAR_FILL * filled}{_BAR_EMPTY * empty}]"
        pct = f"{fraction * 100:3.0f}%"
        parts = [bar, pct]
        if label:
            parts.append(label)
        if suffix:
            parts.append(suffix)
        line = " ".join(parts)
        return self._colorize(line, "green") if fraction >= 1.0 else line

    def coverage_meter(
        self,
        verified: float,
        falsified: float,
        unknown: float,
    ) -> str:
        """Return a three-region coverage bar.

        Example::

            [████▓▓▓░░░░░░░░░] V:30% F:15% U:55%
        """
        total = verified + falsified + unknown
        if total <= 0:
            total = 1.0
        v_frac = verified / total
        f_frac = falsified / total
        u_frac = 1.0 - v_frac - f_frac

        v_chars = int(self._config.width * v_frac)
        f_chars = int(self._config.width * f_frac)
        u_chars = self._config.width - v_chars - f_chars

        bar = (
            _BAR_FILL * v_chars
            + _BAR_PARTIAL * f_chars
            + _BAR_EMPTY * u_chars
        )
        legend = (
            f"V:{v_frac * 100:.0f}% "
            f"F:{f_frac * 100:.0f}% "
            f"U:{u_frac * 100:.0f}%"
        )
        return f"[{bar}] {legend}"

    def iteration_status(
        self,
        iteration: int,
        states: int,
        predicates: int,
        coverage: float,
        elapsed: float,
        strategy: str = "",
    ) -> str:
        """Return a single-line iteration status string.

        Example::

            Iter 5 | States: 128 | Preds: 12 | Cov: 45.2% | 3.2s | Strategy: monotonicity
        """
        parts = [
            f"Iter {iteration}",
            f"States: {states}",
            f"Preds: {predicates}",
            f"Cov: {coverage * 100:.1f}%",
            f"{elapsed:.1f}s",
        ]
        if strategy:
            parts.append(f"Strategy: {strategy}")
        line = " | ".join(parts)
        return self._maybe_color_coverage(line, coverage)

    def benchmark_progress(
        self,
        completed: int,
        total: int,
        current_name: str = "",
    ) -> str:
        """Return a benchmark-suite progress line.

        Example::

            Benchmarks: [████░░░░] 4/10  running: toggle_switch
        """
        bar = self.progress_bar(completed, total)
        line = f"Benchmarks: {bar} {completed}/{total}"
        if current_name:
            line += f"  running: {current_name}"
        return line

    def parameter_exploration_progress(
        self,
        explored: int,
        total: int,
        feasible: int,
    ) -> str:
        """Return a parameter-exploration progress line.

        Example::

            Params: [███░░░░░] 30/100  feasible: 12
        """
        bar = self.progress_bar(explored, total)
        return f"Params: {bar} {explored}/{total}  feasible: {feasible}"

    def synthesis_progress(
        self,
        iteration: int,
        best_robustness: float,
        elapsed: float,
    ) -> str:
        """Return a synthesis-loop progress line.

        Example::

            Synthesis iter 7 | best robustness: 0.342 | 12.5s
        """
        return (
            f"Synthesis iter {iteration} "
            f"| best robustness: {best_robustness:.3f} "
            f"| {elapsed:.1f}s"
        )

    def start_phase(self, name: str) -> None:
        """Print a phase-start header to the output stream."""
        header = f"── {name} "
        header += "─" * max(0, 60 - len(header))
        self._write(self._colorize(header, "bold"))

    def end_phase(self, name: str, elapsed: float) -> None:
        """Print a phase-end footer with elapsed time."""
        footer = f"── {name} done ({elapsed:.2f}s) "
        footer += "─" * max(0, 60 - len(footer))
        self._write(self._colorize(footer, "dim"))

    # -- internal ----------------------------------------------------------

    def _colorize(self, text: str, color: str) -> str:
        """Wrap *text* in ANSI escape codes if colour is enabled.

        Supported colour names: ``green``, ``red``, ``yellow``, ``blue``,
        ``bold``, ``dim``.
        """
        if not self._config.color:
            return text
        code = _ANSI_CODES.get(color, "")
        if not code:
            return text
        return f"{code}{text}{_ANSI_CODES['reset']}"

    def _detect_color_support(self) -> bool:
        """Return ``True`` when the output stream likely supports ANSI."""
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("FORCE_COLOR"):
            return True
        if not hasattr(self._stream, "isatty"):
            return False
        return self._stream.isatty()

    def _write(self, text: str) -> None:
        """Write a line to the output stream."""
        self._stream.write(text + "\n")
        self._stream.flush()

    def _maybe_color_coverage(self, text: str, coverage: float) -> str:
        """Colour the whole line based on coverage level."""
        if coverage >= 1.0:
            return self._colorize(text, "green")
        if coverage >= 0.5:
            return self._colorize(text, "yellow")
        return text


# ---------------------------------------------------------------------------
# CEGARDashboard
# ---------------------------------------------------------------------------


class CEGARDashboard:
    """Multi-line dashboard for CEGAR verification progress.

    Renders a bordered box showing iteration count, coverage bar,
    state / predicate counts, elapsed time, and current refinement
    strategy.
    """

    _BOX_WIDTH = 48

    def __init__(
        self,
        reporter: Optional[ProgressReporter] = None,
    ) -> None:
        self._reporter = reporter or ProgressReporter()
        self._coverage_history: list[tuple[float, float]] = []

    def update(
        self,
        stats: CEGARStatistics,
        current_strategy: str = "",
    ) -> str:
        """Return a multi-line dashboard string for the current state.

        Example::

            ╔══ BioProver CEGAR ══════════════════════════╗
            ║ Iteration:  5     Coverage: [████░░░] 45.2% ║
            ║ States:   128     Predicates: 12             ║
            ║ Elapsed:  3.2s    Strategy: monotonicity     ║
            ╚═════════════════════════════════════════════╝
        """
        self._coverage_history.append(
            (stats.total_time, stats.final_coverage)
        )

        inner = self._BOX_WIDTH
        title = " BioProver CEGAR "
        top_pad = inner - len(title) - 2  # account for ══ prefix
        top = f"╔══{title}{'═' * max(0, top_pad)}╗"

        cov_bar = self._mini_coverage_bar(stats.final_coverage, width=8)
        line1 = (
            f"Iteration: {stats.iterations:<5d} "
            f"Coverage: {cov_bar} {stats.final_coverage * 100:.1f}%"
        )
        line2 = (
            f"States: {stats.peak_states:>5d}     "
            f"Predicates: {stats.peak_predicates}"
        )
        strategy_label = current_strategy or "auto"
        line3 = (
            f"Elapsed: {stats.total_time:>5.1f}s    "
            f"Strategy: {strategy_label}"
        )

        rows = [
            self._box_row(line1, inner),
            self._box_row(line2, inner),
            self._box_row(line3, inner),
        ]
        bottom = f"╚{'═' * (inner + 2)}╝"
        return "\n".join([top, *rows, bottom])

    def estimated_remaining(
        self,
        stats: CEGARStatistics,
    ) -> Optional[float]:
        """Estimate remaining seconds based on the coverage rate.

        Returns ``None`` when insufficient data is available (fewer than
        two recorded updates) or when coverage is not advancing.
        """
        if len(self._coverage_history) < 2:
            return None
        t0, c0 = self._coverage_history[0]
        t1, c1 = self._coverage_history[-1]
        dt = t1 - t0
        dc = c1 - c0
        if dc <= 0 or dt <= 0:
            return None
        rate = dc / dt  # coverage per second
        remaining_coverage = 1.0 - c1
        if remaining_coverage <= 0:
            return 0.0
        return remaining_coverage / rate

    def summary(self, stats: CEGARStatistics) -> str:
        """Return a final summary after CEGAR completes.

        Includes total iterations, elapsed time, peak state-space size,
        coverage, and strategy breakdown.
        """
        lines = [
            "═══ CEGAR Summary ═══",
            f"  Iterations:   {stats.iterations}",
            f"  Total time:   {stats.total_time:.2f}s",
            f"  Peak states:  {stats.peak_states}",
            f"  Predicates:   {stats.peak_predicates}",
            f"  Coverage:     {stats.final_coverage * 100:.1f}%",
        ]
        if stats.strategies_used:
            lines.append("  Strategies:")
            for name, count in sorted(
                stats.strategies_used.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {name}: {count}")
        lines.append("═" * 22)
        return "\n".join(lines)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _mini_coverage_bar(coverage: float, width: int = 8) -> str:
        """Return a small coverage bar without brackets."""
        filled = int(width * max(0.0, min(1.0, coverage)))
        return _BAR_FILL * filled + _BAR_EMPTY * (width - filled)

    @staticmethod
    def _box_row(text: str, inner_width: int) -> str:
        """Pad *text* into a bordered row ``║ ... ║``."""
        padded = text.ljust(inner_width)
        return f"║ {padded} ║"


# ---------------------------------------------------------------------------
# Logging integration
# ---------------------------------------------------------------------------


class LoggingProgressHandler(logging.Handler):
    """A :class:`logging.Handler` that formats progress updates as records.

    Attach this handler to a logger to have progress-report strings
    emitted through the standard logging infrastructure, making it easy
    to interleave progress output with regular log messages in batch /
    CI environments.
    """

    def __init__(
        self,
        reporter: Optional[ProgressReporter] = None,
        level: int = logging.INFO,
    ) -> None:
        super().__init__(level=level)
        self.reporter = reporter or ProgressReporter(
            ProgressConfig(color=False, interactive=False)
        )

    def emit(self, record: logging.LogRecord) -> None:
        """Write the formatted record to the reporter's stream."""
        try:
            msg = self.format(record)
            self.reporter._write(msg)
        except Exception:  # pragma: no cover – best-effort
            self.handleError(record)


def setup_logging(verbosity: int = 0) -> None:
    """Configure the ``bioprover`` logger hierarchy.

    Parameters
    ----------
    verbosity:
        * 0 – WARNING (default)
        * 1 – INFO
        * 2 – DEBUG
        * 3 – TRACE (custom level 5)
    """
    level_map: Dict[int, int] = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: TRACE,
    }
    level = level_map.get(verbosity, TRACE)

    root = logging.getLogger("bioprover")
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setLevel(level)
