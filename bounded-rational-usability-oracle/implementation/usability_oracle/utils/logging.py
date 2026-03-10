"""
usability_oracle.utils.logging — Structured logging utilities.

Provides :func:`setup_logging` for initialising the logging system,
:func:`get_logger` for per-module loggers, and :class:`LogContext`
for injecting structured context into log records.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

_DEFAULT_FMT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DETAILED_FMT = "%(asctime)s [%(levelname)-8s] %(name)s (%(filename)s:%(lineno)d): %(message)s"


class _ContextFormatter(logging.Formatter):
    """Formatter that appends structured context from the record."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        ctx = getattr(record, "context", None)
        if ctx:
            ctx_str = " ".join(f"{k}={v}" for k, v in ctx.items())
            msg = f"{msg} [{ctx_str}]"
        return msg


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def setup_logging(
    level: str = "info",
    log_file: Optional[str] = None,
    detailed: bool = False,
) -> logging.Logger:
    """Initialise the ``usability_oracle`` root logger.

    Parameters:
        level: Log level name (debug/info/warning/error/critical).
        log_file: Optional file path; if given, logs are also written there.
        detailed: Use a more verbose format including file/line info.

    Returns:
        The root ``usability_oracle`` logger.
    """
    log_level = _LEVEL_MAP.get(level.lower(), logging.INFO)
    fmt_str = _DETAILED_FMT if detailed else _DEFAULT_FMT
    formatter = _ContextFormatter(fmt_str, datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger("usability_oracle")
    root.setLevel(log_level)

    # Avoid adding duplicate handlers
    if not root.handlers:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(log_level)
        console.setFormatter(formatter)
        root.addHandler(console)

    if log_file:
        already_has_file = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith(log_file)
            for h in root.handlers
        )
        if not already_has_file:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            root.addHandler(fh)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``usability_oracle`` namespace."""
    return logging.getLogger(f"usability_oracle.{name}")


# ---------------------------------------------------------------------------
# LogContext
# ---------------------------------------------------------------------------

class LogContext:
    """Context manager that attaches structured key-value pairs to log records.

    Usage::

        with LogContext(logger, stage="parse", file="test.html"):
            logger.info("Starting parse")
            # Record will include [stage=parse file=test.html]
    """

    def __init__(self, logger: logging.Logger, **context: Any) -> None:
        self._logger = logger
        self._context = context
        self._old_factory: Any = None

    def __enter__(self) -> "LogContext":
        ctx = self._context

        old_factory = logging.getLogRecordFactory()
        self._old_factory = old_factory

        def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = old_factory(*args, **kwargs)
            existing = getattr(record, "context", {})
            record.context = {**existing, **ctx}  # type: ignore[attr-defined]
            return record

        logging.setLogRecordFactory(factory)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._old_factory is not None:
            logging.setLogRecordFactory(self._old_factory)


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------

def format_stage_log(stage: str, message: str, timing: float | None = None) -> str:
    """Format a pipeline stage log message.

    Parameters:
        stage: Pipeline stage name.
        message: Human-readable message.
        timing: Optional elapsed seconds.

    Returns:
        Formatted string, e.g. ``"[PARSE] Completed in 0.123s"``
    """
    tag = f"[{stage.upper()}]"
    if timing is not None:
        if timing < 1.0:
            t_str = f"{timing * 1000:.1f}ms"
        else:
            t_str = f"{timing:.3f}s"
        return f"{tag} {message} ({t_str})"
    return f"{tag} {message}"
