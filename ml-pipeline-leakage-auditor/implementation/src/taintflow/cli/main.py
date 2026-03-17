"""TaintFlow CLI – main entry point and argument parsing.

Provides the ``taintflow`` command with subcommands for auditing ML pipelines,
scanning for common leakage patterns, generating reports, comparing results,
managing configuration, and querying plugin information.

Exit codes
----------
0 – no leakage detected (or informational command succeeded)
1 – leakage detected above the configured severity threshold
2 – runtime / user error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, NoReturn, Optional, Sequence, TextIO

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_CLEAN = 0
EXIT_LEAKAGE = 1
EXIT_ERROR = 2

# ---------------------------------------------------------------------------
# ANSI colour helpers (auto-detected)
# ---------------------------------------------------------------------------

_COLOR_SUPPORTED: Optional[bool] = None


def _supports_color(stream: TextIO = sys.stderr) -> bool:
    """Return *True* if *stream* is connected to a colour-capable terminal."""
    global _COLOR_SUPPORTED
    if _COLOR_SUPPORTED is not None:
        return _COLOR_SUPPORTED

    if os.environ.get("NO_COLOR"):
        _COLOR_SUPPORTED = False
    elif os.environ.get("FORCE_COLOR"):
        _COLOR_SUPPORTED = True
    elif not hasattr(stream, "isatty"):
        _COLOR_SUPPORTED = False
    else:
        _COLOR_SUPPORTED = stream.isatty()
    return _COLOR_SUPPORTED


def _ansi(code: str, text: str, *, stream: TextIO = sys.stderr) -> str:
    """Wrap *text* in ANSI escape sequences when colour is supported."""
    if _supports_color(stream):
        return f"\033[{code}m{text}\033[0m"
    return text


def color_red(text: str) -> str:
    """Return *text* wrapped in red ANSI colour."""
    return _ansi("31", text)


def color_green(text: str) -> str:
    """Return *text* wrapped in green ANSI colour."""
    return _ansi("32", text)


def color_yellow(text: str) -> str:
    """Return *text* wrapped in yellow ANSI colour."""
    return _ansi("33", text)


def color_blue(text: str) -> str:
    """Return *text* wrapped in blue ANSI colour."""
    return _ansi("34", text)


def color_bold(text: str) -> str:
    """Return *text* wrapped in bold ANSI formatting."""
    return _ansi("1", text)


def color_dim(text: str) -> str:
    """Return *text* wrapped in dim ANSI formatting."""
    return _ansi("2", text)


def severity_color(severity: str) -> str:
    """Colour-code a severity string (negligible/warning/critical)."""
    severity_lower = severity.lower()
    if severity_lower == "critical":
        return color_red(severity)
    if severity_lower == "warning":
        return color_yellow(severity)
    return color_green(severity)


# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------


class ProgressReporter:
    """Simple progress reporter that writes to *stderr*.

    Renders a progress bar when the output stream supports colour, otherwise
    falls back to ``[step/total] message`` lines.
    """

    def __init__(self, total: int, *, label: str = "", stream: TextIO = sys.stderr) -> None:
        self.total = max(total, 1)
        self.current = 0
        self.label = label
        self.stream = stream
        self._start = time.monotonic()

    def update(self, message: str = "") -> None:
        """Advance the progress counter by one and display status."""
        self.current = min(self.current + 1, self.total)
        self._render(message)

    def finish(self, message: str = "done") -> None:
        """Mark progress as complete."""
        self.current = self.total
        self._render(message)
        self.stream.write("\n")
        self.stream.flush()

    def _render(self, message: str) -> None:
        elapsed = time.monotonic() - self._start
        if _supports_color(self.stream):
            width = 30
            filled = int(width * self.current / self.total)
            bar = "█" * filled + "░" * (width - filled)
            pct = self.current * 100 // self.total
            line = f"\r{self.label} [{bar}] {pct:3d}% {message} ({elapsed:.1f}s)"
            self.stream.write(line)
            self.stream.flush()
        else:
            self.stream.write(f"[{self.current}/{self.total}] {message} ({elapsed:.1f}s)\n")
            self.stream.flush()


# ---------------------------------------------------------------------------
# User-friendly error handling
# ---------------------------------------------------------------------------


def _error(message: str, *, hint: str = "") -> NoReturn:
    """Print an error message to *stderr* and exit with :data:`EXIT_ERROR`."""
    prefix = color_red("error:")
    sys.stderr.write(f"{prefix} {message}\n")
    if hint:
        sys.stderr.write(f"  {color_dim('hint:')} {hint}\n")
    sys.exit(EXIT_ERROR)


def _warn(message: str) -> None:
    """Print a warning message to *stderr*."""
    prefix = color_yellow("warning:")
    sys.stderr.write(f"{prefix} {message}\n")


def _info(message: str) -> None:
    """Print an informational message to *stderr*."""
    prefix = color_blue("info:")
    sys.stderr.write(f"{prefix} {message}\n")


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser and all subcommand parsers."""
    parser = argparse.ArgumentParser(
        prog="taintflow",
        description="TaintFlow – ML Pipeline Leakage Auditor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            exit codes:
              0  no leakage detected (or informational command succeeded)
              1  leakage detected above the configured severity threshold
              2  runtime or user error

            examples:
              taintflow audit --input pipeline.py
              taintflow audit --input pipeline.py --format json --output report.json
              taintflow scan --input pipeline.py --patterns all
              taintflow config init
        """),
    )
    parser.add_argument(
        "--version", action="version", version=f"taintflow {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- audit ---------------------------------------------------------------
    audit_parser = subparsers.add_parser(
        "audit",
        help="Run a full leakage audit on a pipeline script",
        description=(
            "Execute and instrument a Python ML pipeline script, build the "
            "dataflow DAG, estimate channel capacities, run taint propagation, "
            "and report any detected train/test leakage."
        ),
    )
    audit_parser.add_argument(
        "--input", "-i", required=True, metavar="FILE",
        help="Path to the Python pipeline script to audit",
    )
    audit_parser.add_argument(
        "--format", "-f", default="text",
        choices=["text", "json", "html", "sarif"],
        help="Output format (default: text)",
    )
    audit_parser.add_argument(
        "--output", "-o", metavar="FILE", default=None,
        help="Write report to FILE instead of stdout",
    )
    audit_parser.add_argument(
        "--severity", "-s", default="negligible",
        choices=["negligible", "warning", "critical"],
        help="Minimum severity to report (default: negligible)",
    )
    audit_parser.add_argument(
        "--config", "-c", metavar="FILE", default=None,
        help="Path to a TaintFlow TOML/JSON configuration file",
    )
    audit_parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase verbosity (may be repeated)",
    )
    audit_parser.add_argument(
        "--no-empirical", action="store_true", default=False,
        help="Skip empirical refinement of channel capacities",
    )
    audit_parser.add_argument(
        "--max-iterations", type=int, default=None, metavar="N",
        help="Maximum worklist / fixpoint iterations (overrides config)",
    )

    # -- scan ----------------------------------------------------------------
    scan_parser = subparsers.add_parser(
        "scan",
        help="Quick scan for common leakage patterns (no execution)",
        description=(
            "Statically scan a pipeline script for common leakage patterns "
            "without executing the script.  Faster but less precise than a "
            "full audit."
        ),
    )
    scan_parser.add_argument(
        "--input", "-i", required=True, metavar="FILE",
        help="Path to the Python pipeline script to scan",
    )
    scan_parser.add_argument(
        "--patterns", "-p", default="all",
        help=(
            "Comma-separated list of pattern names to check, or 'all' "
            "(default: all)"
        ),
    )

    # -- report --------------------------------------------------------------
    report_parser = subparsers.add_parser(
        "report",
        help="Generate a report from saved analysis results",
        description=(
            "Re-render a report from a previously saved JSON analysis "
            "results file.  Useful for converting between output formats "
            "without re-running the analysis."
        ),
    )
    report_parser.add_argument(
        "--input", "-i", required=True, metavar="FILE",
        help="Path to the analysis results JSON file",
    )
    report_parser.add_argument(
        "--format", "-f", default="text",
        choices=["text", "json", "html", "sarif"],
        help="Output format (default: text)",
    )
    report_parser.add_argument(
        "--output", "-o", metavar="FILE", default=None,
        help="Write report to FILE instead of stdout",
    )

    # -- compare -------------------------------------------------------------
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two analysis results",
        description=(
            "Produce a side-by-side diff of two analysis result files to "
            "highlight regressions and improvements."
        ),
    )
    compare_parser.add_argument(
        "--before", "-b", required=True, metavar="FILE",
        help="Path to the 'before' analysis results JSON",
    )
    compare_parser.add_argument(
        "--after", "-a", required=True, metavar="FILE",
        help="Path to the 'after' analysis results JSON",
    )

    # -- config --------------------------------------------------------------
    config_parser = subparsers.add_parser(
        "config",
        help="Manage TaintFlow configuration",
        description="Create, view, or validate TaintFlow configuration files.",
    )
    config_sub = config_parser.add_subparsers(dest="config_action", help="Config actions")

    config_sub.add_parser(
        "init", help="Create a default taintflow.toml in the current directory"
    )
    config_sub.add_parser(
        "show", help="Display the resolved configuration"
    )
    config_validate = config_sub.add_parser(
        "validate", help="Validate a configuration file"
    )
    config_validate.add_argument(
        "--file", "-f", metavar="FILE", default=None,
        help="Path to the config file to validate (default: auto-discover)",
    )

    # -- version -------------------------------------------------------------
    subparsers.add_parser(
        "version",
        help="Show detailed version information",
        description="Display TaintFlow version, Python version, and platform.",
    )

    # -- plugins -------------------------------------------------------------
    subparsers.add_parser(
        "plugins",
        help="List installed TaintFlow plugins",
        description="Discover and display all installed TaintFlow plugins.",
    )

    return parser


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------


def _dispatch(args: argparse.Namespace) -> int:
    """Route parsed arguments to the appropriate command handler.

    Returns an exit code: 0 (clean), 1 (leakage found), or 2 (error).
    """
    from taintflow.cli.commands import (
        AuditCommand,
        CompareCommand,
        ConfigCommand,
        PluginsCommand,
        ReportCommand,
        ScanCommand,
    )

    command_map: Dict[str, Any] = {
        "audit": AuditCommand,
        "scan": ScanCommand,
        "report": ReportCommand,
        "compare": CompareCommand,
        "config": ConfigCommand,
        "version": None,
        "plugins": PluginsCommand,
    }

    name = args.command
    if name is None:
        _build_parser().print_help(sys.stderr)
        return EXIT_ERROR

    if name == "version":
        return _print_version()

    cls = command_map.get(name)
    if cls is None:
        _error(f"unknown command: {name}")

    cmd = cls()
    errors = cmd.validate_args(args)
    if errors:
        for err in errors:
            _warn(err)
        _error(
            f"invalid arguments for '{name}'",
            hint=f"Run 'taintflow {name} --help' for usage information.",
        )

    return cmd.execute(args)


def _print_version() -> int:
    """Print detailed version information and return :data:`EXIT_CLEAN`."""
    import platform

    lines = [
        f"taintflow {__version__}",
        f"Python   {platform.python_version()}",
        f"Platform {platform.platform()}",
    ]

    try:
        from taintflow.core.config import TaintFlowConfig  # noqa: F811
        lines.append(f"Config   {TaintFlowConfig.__module__}")
    except ImportError:
        pass

    sys.stdout.write("\n".join(lines) + "\n")
    return EXIT_CLEAN


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the TaintFlow CLI.

    Parameters
    ----------
    argv:
        Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code suitable for passing to :func:`sys.exit`.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        sys.stderr.write("\n")
        _warn("interrupted by user")
        return EXIT_ERROR
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_ERROR
    except Exception as exc:  # noqa: BLE001
        _error(str(exc), hint="Re-run with --verbose for a full traceback.")


def cli() -> NoReturn:
    """Console-script entry point (calls :func:`main` and exits)."""
    sys.exit(main())


if __name__ == "__main__":
    cli()
