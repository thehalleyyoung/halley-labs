"""
taintflow.core.logging_utils – Structured logging, performance tracking,
and audit-trail infrastructure for TaintFlow.

This module provides:
* :class:`TaintFlowLogger`  – a thin wrapper around :mod:`logging` that
  emits structured :class:`LogEvent` records and integrates with
  :pypi:`rich` for pretty console output.
* :func:`log_performance` – a decorator that logs wall-clock time and
  optional memory deltas for any callable.
* :func:`track_memory` – a context manager that captures peak RSS delta.
* Console and file formatters suitable for CI and interactive use.
"""

from __future__ import annotations

import functools
import logging
import os
import resource
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Generator,
    Mapping,
    TypeVar,
)

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
#  Structured log event
# ---------------------------------------------------------------------------

@dataclass
class LogEvent:
    """A single structured log entry produced during an audit run."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    level: str = "INFO"
    logger_name: str = "taintflow"
    message: str = ""
    phase: str = ""
    node_id: str = ""
    operation: str = ""
    duration_ms: float | None = None
    memory_delta_kb: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ts": self.timestamp,
            "level": self.level,
            "logger": self.logger_name,
            "msg": self.message,
        }
        if self.phase:
            d["phase"] = self.phase
        if self.node_id:
            d["node"] = self.node_id
        if self.operation:
            d["op"] = self.operation
        if self.duration_ms is not None:
            d["dur_ms"] = round(self.duration_ms, 3)
        if self.memory_delta_kb is not None:
            d["mem_kb"] = round(self.memory_delta_kb, 1)
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LogEvent":
        return cls(
            timestamp=str(data.get("ts", "")),
            level=str(data.get("level", "INFO")),
            logger_name=str(data.get("logger", "taintflow")),
            message=str(data.get("msg", "")),
            phase=str(data.get("phase", "")),
            node_id=str(data.get("node", "")),
            operation=str(data.get("op", "")),
            duration_ms=data.get("dur_ms"),
            memory_delta_kb=data.get("mem_kb"),
            extra=dict(data.get("extra", {})),
        )

    def __str__(self) -> str:
        parts = [f"[{self.level}]", self.message]
        if self.phase:
            parts.append(f"(phase={self.phase})")
        if self.duration_ms is not None:
            parts.append(f"[{self.duration_ms:.1f}ms]")
        return " ".join(parts)


# ---------------------------------------------------------------------------
#  Formatters
# ---------------------------------------------------------------------------

class _ConsoleFormatter(logging.Formatter):
    """Human-friendly single-line formatter with optional colour."""

    _COLOURS: dict[int, str] = {
        logging.DEBUG: "\033[90m",      # grey
        logging.INFO: "\033[36m",       # cyan
        logging.WARNING: "\033[33m",    # yellow
        logging.ERROR: "\033[31m",      # red
        logging.CRITICAL: "\033[1;31m", # bold red
    }
    _RESET = "\033[0m"

    def __init__(self, *, use_colour: bool = True) -> None:
        super().__init__()
        self.use_colour = use_colour and hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S")
        lvl = record.levelname[:4]
        msg = record.getMessage()
        extra_parts: list[str] = []

        phase = getattr(record, "phase", "")
        if phase:
            extra_parts.append(f"phase={phase}")
        dur = getattr(record, "duration_ms", None)
        if dur is not None:
            extra_parts.append(f"{dur:.1f}ms")
        suffix = f" ({', '.join(extra_parts)})" if extra_parts else ""

        line = f"{ts} {lvl:>4s} | {msg}{suffix}"
        if self.use_colour:
            colour = self._COLOURS.get(record.levelno, "")
            return f"{colour}{line}{self._RESET}"
        return line


class _JsonFormatter(logging.Formatter):
    """Machine-readable JSON-lines formatter for file / CI output."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        import json

        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for attr in ("phase", "node_id", "operation", "duration_ms", "memory_delta_kb"):
            val = getattr(record, attr, None)
            if val is not None:
                entry[attr] = val
        extra = getattr(record, "event_extra", None)
        if extra:
            entry["extra"] = extra
        return json.dumps(entry, default=str)


class _AuditTrailFormatter(logging.Formatter):
    """Deterministic formatter for audit-trail logs (reproducibility)."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        phase = getattr(record, "phase", "-")
        node = getattr(record, "node_id", "-")
        op = getattr(record, "operation", "-")
        return f"{ts}\t{record.levelname}\t{phase}\t{node}\t{op}\t{record.getMessage()}"


# ---------------------------------------------------------------------------
#  Logger wrapper
# ---------------------------------------------------------------------------

class TaintFlowLogger:
    """Structured logger for TaintFlow audit runs.

    Wraps the standard :class:`logging.Logger` and adds:
    * per-event structured fields (phase, node_id, operation, …),
    * an in-memory audit trail (:attr:`audit_trail`),
    * integration hooks for progress bars (Rich).
    """

    _lock = threading.Lock()
    _instances: dict[str, "TaintFlowLogger"] = {}

    def __init__(
        self,
        name: str = "taintflow",
        *,
        level: int | str = logging.INFO,
        log_file: str | None = None,
        use_colour: bool = True,
        json_output: bool = False,
        audit_trail_enabled: bool = True,
        max_trail_size: int = 50_000,
    ) -> None:
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level if isinstance(level, int) else getattr(logging, level.upper(), logging.INFO))
        self._logger.propagate = False
        self.audit_trail: list[LogEvent] = []
        self._audit_enabled = audit_trail_enabled
        self._max_trail = max_trail_size
        self._current_phase: str = ""
        self._setup_handlers(log_file=log_file, use_colour=use_colour, json_output=json_output)

    # -- handler setup -------------------------------------------------------

    def _setup_handlers(
        self,
        *,
        log_file: str | None,
        use_colour: bool,
        json_output: bool,
    ) -> None:
        # clear existing handlers to avoid duplicate output on re-init
        self._logger.handlers.clear()

        # console handler
        console = logging.StreamHandler(sys.stderr)
        if json_output:
            console.setFormatter(_JsonFormatter())
        else:
            console.setFormatter(_ConsoleFormatter(use_colour=use_colour))
        self._logger.addHandler(console)

        # optional file handler
        if log_file:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(_AuditTrailFormatter())
            self._logger.addHandler(fh)

    # -- singleton / factory -------------------------------------------------

    @classmethod
    def get(cls, name: str = "taintflow", **kwargs: Any) -> "TaintFlowLogger":
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name, **kwargs)
            return cls._instances[name]

    @classmethod
    def reset_all(cls) -> None:
        with cls._lock:
            cls._instances.clear()

    # -- phase management ----------------------------------------------------

    def set_phase(self, phase: str) -> None:
        self._current_phase = phase
        self.info(f"Entering phase: {phase}", phase=phase)

    @property
    def current_phase(self) -> str:
        return self._current_phase

    # -- core logging methods ------------------------------------------------

    def _emit(
        self,
        level: int,
        msg: str,
        *,
        phase: str = "",
        node_id: str = "",
        operation: str = "",
        duration_ms: float | None = None,
        memory_delta_kb: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LogEvent:
        effective_phase = phase or self._current_phase
        event = LogEvent(
            level=logging.getLevelName(level),
            logger_name=self.name,
            message=msg,
            phase=effective_phase,
            node_id=node_id,
            operation=operation,
            duration_ms=duration_ms,
            memory_delta_kb=memory_delta_kb,
            extra=extra or {},
        )
        record = self._logger.makeRecord(
            self.name, level, "(taintflow)", 0, msg, (), None,
        )
        record.phase = effective_phase  # type: ignore[attr-defined]
        record.node_id = node_id  # type: ignore[attr-defined]
        record.operation = operation  # type: ignore[attr-defined]
        record.duration_ms = duration_ms  # type: ignore[attr-defined]
        record.memory_delta_kb = memory_delta_kb  # type: ignore[attr-defined]
        record.event_extra = extra  # type: ignore[attr-defined]
        self._logger.handle(record)

        if self._audit_enabled:
            with self._lock:
                self.audit_trail.append(event)
                if len(self.audit_trail) > self._max_trail:
                    self.audit_trail = self.audit_trail[-self._max_trail:]

        return event

    def debug(self, msg: str, **kw: Any) -> LogEvent:
        return self._emit(logging.DEBUG, msg, **kw)

    def info(self, msg: str, **kw: Any) -> LogEvent:
        return self._emit(logging.INFO, msg, **kw)

    def warning(self, msg: str, **kw: Any) -> LogEvent:
        return self._emit(logging.WARNING, msg, **kw)

    def error(self, msg: str, **kw: Any) -> LogEvent:
        return self._emit(logging.ERROR, msg, **kw)

    def critical(self, msg: str, **kw: Any) -> LogEvent:
        return self._emit(logging.CRITICAL, msg, **kw)

    # -- convenience: log an exception with context --------------------------

    def log_exception(
        self,
        exc: BaseException,
        *,
        phase: str = "",
        node_id: str = "",
    ) -> LogEvent:
        import traceback as _tb

        tb_str = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
        return self.error(
            f"{type(exc).__name__}: {exc}",
            phase=phase,
            node_id=node_id,
            extra={"traceback": tb_str},
        )

    # -- audit trail helpers -------------------------------------------------

    def get_trail(self, *, level: str | None = None, phase: str | None = None) -> list[LogEvent]:
        with self._lock:
            trail = list(self.audit_trail)
        if level:
            trail = [e for e in trail if e.level == level.upper()]
        if phase:
            trail = [e for e in trail if e.phase == phase]
        return trail

    def clear_trail(self) -> None:
        with self._lock:
            self.audit_trail.clear()

    def trail_summary(self) -> dict[str, Any]:
        with self._lock:
            trail = list(self.audit_trail)
        counts: dict[str, int] = {}
        total_ms = 0.0
        for ev in trail:
            counts[ev.level] = counts.get(ev.level, 0) + 1
            if ev.duration_ms is not None:
                total_ms += ev.duration_ms
        return {
            "total_events": len(trail),
            "by_level": counts,
            "total_duration_ms": round(total_ms, 3),
        }

    # -- progress reporting --------------------------------------------------

    @contextmanager
    def progress_context(self, description: str, total: int | None = None) -> Generator[Callable[[int], None], None, None]:
        self.info(f"Progress: {description} started", extra={"total": total})
        completed = [0]
        t0 = time.perf_counter()

        def advance(n: int = 1) -> None:
            completed[0] += n

        try:
            yield advance
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            self.info(
                f"Progress: {description} finished ({completed[0]}/{total or '?'})",
                duration_ms=elapsed,
            )


# ---------------------------------------------------------------------------
#  Performance decorator
# ---------------------------------------------------------------------------

def log_performance(
    logger: TaintFlowLogger | None = None,
    *,
    level: str = "DEBUG",
    include_memory: bool = False,
) -> Callable[[F], F]:
    """Decorator that logs wall-clock time (and optionally memory) of a function call."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logger or TaintFlowLogger.get()
            lvl = getattr(logging, level.upper(), logging.DEBUG)

            mem_before: int | None = None
            if include_memory:
                mem_before = _get_rss_kb()

            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except Exception:
                elapsed = (time.perf_counter() - t0) * 1000
                _logger._emit(
                    logging.ERROR,
                    f"{fn.__qualname__} raised after {elapsed:.1f}ms",
                    operation=fn.__qualname__,
                    duration_ms=elapsed,
                )
                raise
            elapsed = (time.perf_counter() - t0) * 1000

            mem_delta: float | None = None
            if include_memory and mem_before is not None:
                mem_delta = float(_get_rss_kb() - mem_before)

            _logger._emit(
                lvl,
                f"{fn.__qualname__} completed",
                operation=fn.__qualname__,
                duration_ms=elapsed,
                memory_delta_kb=mem_delta,
            )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
#  Memory tracking
# ---------------------------------------------------------------------------

def _get_rss_kb() -> int:
    """Return current RSS in kilobytes (platform-dependent)."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return int(usage.ru_maxrss / 1024)
        return int(usage.ru_maxrss)
    except Exception:
        return 0


@contextmanager
def track_memory(label: str = "", logger: TaintFlowLogger | None = None) -> Generator[dict[str, float], None, None]:
    """Context manager that captures RSS delta (in KB) within a block.

    Usage::

        with track_memory("fit step") as mem:
            model.fit(X, y)
        print(mem["delta_kb"])
    """
    _logger = logger or TaintFlowLogger.get()
    result: dict[str, float] = {"before_kb": 0.0, "after_kb": 0.0, "delta_kb": 0.0}
    result["before_kb"] = float(_get_rss_kb())
    try:
        yield result
    finally:
        result["after_kb"] = float(_get_rss_kb())
        result["delta_kb"] = result["after_kb"] - result["before_kb"]
        if label:
            _logger.debug(
                f"Memory [{label}]: delta={result['delta_kb']:.1f} KB",
                memory_delta_kb=result["delta_kb"],
                extra={"label": label},
            )


# ---------------------------------------------------------------------------
#  Module-level convenience
# ---------------------------------------------------------------------------

def get_logger(name: str = "taintflow", **kwargs: Any) -> TaintFlowLogger:
    """Convenience alias for :meth:`TaintFlowLogger.get`."""
    return TaintFlowLogger.get(name, **kwargs)


def configure_from_env() -> TaintFlowLogger:
    """Create / reconfigure the default logger from environment variables.

    Recognised variables:
    * ``TAINTFLOW_LOG_LEVEL``  – DEBUG, INFO, WARNING, ERROR, CRITICAL
    * ``TAINTFLOW_LOG_FILE``   – path to a log file
    * ``TAINTFLOW_LOG_JSON``   – ``1`` / ``true`` for JSON output
    * ``TAINTFLOW_LOG_COLOUR`` – ``0`` / ``false`` to disable colour
    """
    level = os.environ.get("TAINTFLOW_LOG_LEVEL", "INFO").upper()
    log_file = os.environ.get("TAINTFLOW_LOG_FILE")
    json_output = os.environ.get("TAINTFLOW_LOG_JSON", "").lower() in ("1", "true", "yes")
    use_colour = os.environ.get("TAINTFLOW_LOG_COLOUR", "1").lower() not in ("0", "false", "no")

    lgr = TaintFlowLogger(
        "taintflow",
        level=level,
        log_file=log_file,
        json_output=json_output,
        use_colour=use_colour,
    )
    TaintFlowLogger._instances["taintflow"] = lgr
    return lgr
