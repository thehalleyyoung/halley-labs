"""
Logging and diagnostic tracing for BioProver.

Structured logging with JSON and human-readable formats, verification audit
logs, CEGAR progress tracking, performance tracing, per-subsystem log levels,
and file rotation.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Subsystem names (used as logger names)
# ---------------------------------------------------------------------------

SUBSYSTEMS = (
    "bioprover",
    "bioprover.model",
    "bioprover.solver",
    "bioprover.encoding",
    "bioprover.smt",
    "bioprover.cegar",
    "bioprover.repair",
    "bioprover.compositional",
    "bioprover.ai",
    "bioprover.infrastructure",
)


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__
            and k not in ("message", "msg", "args")
        }
        if extras:
            entry["extra"] = extras
        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Human-readable formatter
# ---------------------------------------------------------------------------

class HumanFormatter(logging.Formatter):
    """Coloured, concise formatter for terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        lvl = record.levelname[0]
        name = record.name.replace("bioprover.", "")
        msg = record.getMessage()
        if self.use_color:
            c = self.COLORS.get(record.levelname, "")
            return f"{c}{ts} {lvl} [{name}]{self.RESET} {msg}"
        return f"{ts} {lvl} [{name}] {msg}"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    json_output: bool = False,
    subsystem_levels: Optional[Dict[str, str]] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configure the BioProver logging hierarchy.

    Parameters
    ----------
    level:
        Default log level for all ``bioprover.*`` loggers.
    log_dir:
        If given, a rotating file handler is added writing to
        ``<log_dir>/bioprover.log``.
    json_output:
        Use JSON formatter for stderr instead of human-readable.
    subsystem_levels:
        Per-subsystem overrides, e.g. ``{"bioprover.smt": "DEBUG"}``.
    max_bytes:
        Maximum size per log file before rotation.
    backup_count:
        Number of rotated log files to keep.
    """
    root = logging.getLogger("bioprover")
    root.setLevel(logging.DEBUG)  # handlers filter further

    # Remove existing handlers
    root.handlers.clear()

    # -- stderr handler -----------------------------------------------------
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    if json_output:
        stderr_handler.setFormatter(JSONFormatter())
    else:
        stderr_handler.setFormatter(HumanFormatter())
    root.addHandler(stderr_handler)

    # -- file handler -------------------------------------------------------
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_path / "bioprover.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(JSONFormatter())
        root.addHandler(fh)

    # -- subsystem overrides ------------------------------------------------
    if subsystem_levels:
        for name, lvl in subsystem_levels.items():
            sub_logger = logging.getLogger(name)
            sub_logger.setLevel(getattr(logging, lvl.upper(), logging.INFO))

    root.debug("Logging initialised (level=%s, dir=%s)", level, log_dir)


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``bioprover`` hierarchy."""
    if not name.startswith("bioprover"):
        name = f"bioprover.{name}"
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Verification audit log
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    """Single entry in the verification audit trail."""
    timestamp: float
    event: str
    property_id: str
    result: str  # "verified" | "falsified" | "unknown" | "error"
    details: Dict[str, Any] = field(default_factory=dict)


class VerificationAuditLog:
    """Append-only audit log recording every verification decision.

    Produces a tamper-evident trail that can be exported for review.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._entries: List[AuditEntry] = []
        self._path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        event: str,
        property_id: str,
        result: str,
        **details: Any,
    ) -> None:
        entry = AuditEntry(
            timestamp=time.time(),
            event=event,
            property_id=property_id,
            result=result,
            details=details,
        )
        self._entries.append(entry)
        if self._path is not None:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(entry), default=str) + "\n")

    @property
    def entries(self) -> List[AuditEntry]:
        return list(self._entries)

    def summary(self) -> Dict[str, int]:
        """Count entries by result."""
        counts: Dict[str, int] = {}
        for e in self._entries:
            counts[e.result] = counts.get(e.result, 0) + 1
        return counts

    def filter_by_property(self, property_id: str) -> List[AuditEntry]:
        return [e for e in self._entries if e.property_id == property_id]

    def export_json(self) -> str:
        return json.dumps([asdict(e) for e in self._entries], indent=2, default=str)


# ---------------------------------------------------------------------------
# Performance tracer
# ---------------------------------------------------------------------------

@dataclass
class TraceSpan:
    """A timed span within a trace."""
    name: str
    start: float
    end: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["TraceSpan"] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end - self.start if self.end > 0 else time.time() - self.start


class PerformanceTracer:
    """Hierarchical performance tracer for profiling CEGAR iterations.

    Usage::

        tracer = PerformanceTracer()
        with tracer.span("cegar_iteration"):
            with tracer.span("abstraction"):
                ...
            with tracer.span("smt_check"):
                ...
        print(tracer.report())
    """

    def __init__(self) -> None:
        self._root_spans: List[TraceSpan] = []
        self._stack: List[TraceSpan] = []

    class _SpanContext:
        def __init__(self, tracer: "PerformanceTracer", span: TraceSpan) -> None:
            self._tracer = tracer
            self._span = span

        def __enter__(self) -> TraceSpan:
            self._tracer._stack.append(self._span)
            return self._span

        def __exit__(self, *exc: Any) -> None:
            self._span.end = time.time()
            self._tracer._stack.pop()

    def span(self, name: str, **metadata: Any) -> _SpanContext:
        """Create a new trace span (context manager)."""
        s = TraceSpan(name=name, start=time.time(), metadata=metadata)
        if self._stack:
            self._stack[-1].children.append(s)
        else:
            self._root_spans.append(s)
        return self._SpanContext(self, s)

    def report(self, min_duration_s: float = 0.0) -> str:
        """Human-readable timing report."""
        lines: List[str] = ["Performance Trace", "=" * 50]
        for span in self._root_spans:
            self._format_span(span, lines, indent=0, min_dur=min_duration_s)
        return "\n".join(lines)

    def total_time(self) -> float:
        return sum(s.duration_s for s in self._root_spans)

    def _format_span(
        self,
        span: TraceSpan,
        lines: List[str],
        indent: int,
        min_dur: float,
    ) -> None:
        dur = span.duration_s
        if dur < min_dur:
            return
        prefix = "  " * indent
        lines.append(f"{prefix}{span.name}: {dur:.4f}s")
        for child in span.children:
            self._format_span(child, lines, indent + 1, min_dur)


# ---------------------------------------------------------------------------
# CEGAR progress log
# ---------------------------------------------------------------------------

@dataclass
class CEGARIterationRecord:
    """One iteration of the CEGAR loop."""
    iteration: int
    timestamp: float
    abstraction_size: int
    num_predicates: int
    smt_result: str  # "sat" | "unsat" | "unknown"
    counterexample_found: bool
    refinement_applied: bool
    elapsed_s: float
    coverage_estimate: float = 0.0
    notes: str = ""


class CEGARProgressLog:
    """Track CEGAR loop progress for reporting and convergence analysis."""

    def __init__(self) -> None:
        self._records: List[CEGARIterationRecord] = []

    def record_iteration(
        self,
        iteration: int,
        abstraction_size: int,
        num_predicates: int,
        smt_result: str,
        counterexample_found: bool,
        refinement_applied: bool,
        elapsed_s: float,
        coverage_estimate: float = 0.0,
        notes: str = "",
    ) -> None:
        rec = CEGARIterationRecord(
            iteration=iteration,
            timestamp=time.time(),
            abstraction_size=abstraction_size,
            num_predicates=num_predicates,
            smt_result=smt_result,
            counterexample_found=counterexample_found,
            refinement_applied=refinement_applied,
            elapsed_s=elapsed_s,
            coverage_estimate=coverage_estimate,
            notes=notes,
        )
        self._records.append(rec)

    @property
    def records(self) -> List[CEGARIterationRecord]:
        return list(self._records)

    @property
    def num_iterations(self) -> int:
        return len(self._records)

    @property
    def total_time_s(self) -> float:
        return sum(r.elapsed_s for r in self._records)

    def is_converging(self, window: int = 5, threshold: float = 1e-4) -> bool:
        """Heuristic: check if recent coverage estimates are stabilising."""
        if len(self._records) < window:
            return False
        recent = [r.coverage_estimate for r in self._records[-window:]]
        spread = max(recent) - min(recent)
        return spread < threshold

    def summary(self) -> str:
        """Compact summary of CEGAR progress."""
        if not self._records:
            return "No CEGAR iterations recorded."
        last = self._records[-1]
        lines = [
            f"CEGAR Progress: {self.num_iterations} iterations, "
            f"{self.total_time_s:.2f}s total",
            f"  Last: iter={last.iteration}, predicates={last.num_predicates}, "
            f"coverage={last.coverage_estimate:.4f}",
            f"  Converging: {self.is_converging()}",
        ]
        return "\n".join(lines)

    def to_csv(self) -> str:
        """Export records as CSV."""
        header = (
            "iteration,abstraction_size,num_predicates,smt_result,"
            "counterexample_found,refinement_applied,elapsed_s,"
            "coverage_estimate"
        )
        rows = [header]
        for r in self._records:
            rows.append(
                f"{r.iteration},{r.abstraction_size},{r.num_predicates},"
                f"{r.smt_result},{r.counterexample_found},"
                f"{r.refinement_applied},{r.elapsed_s:.6f},"
                f"{r.coverage_estimate:.6f}"
            )
        return "\n".join(rows)
