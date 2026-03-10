"""
Structured logging configuration for the CausalCert pipeline.

Configures JSON-formatted structured logging with per-module log levels
and optional file output.  Includes performance-metrics helpers and an
audit-trail logger for full reproducibility.
"""

from __future__ import annotations

import logging
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data  # type: ignore[attr-defined]
        return json.dumps(log_entry)


# ---------------------------------------------------------------------------
# Performance-metrics formatter
# ---------------------------------------------------------------------------


class PerfFormatter(logging.Formatter):
    """Formatter that appends elapsed-time information when available."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        elapsed = getattr(record, "elapsed_s", None)
        if elapsed is not None:
            base += f"  [{elapsed:.3f}s]"
        return base


# ---------------------------------------------------------------------------
# Audit-trail handler
# ---------------------------------------------------------------------------


class AuditTrailHandler(logging.Handler):
    """Accumulates log records for reproducibility audit trails.

    Records are stored in memory and can be flushed to a JSON-lines file.
    """

    def __init__(self, max_records: int = 50_000) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[dict[str, Any]] = []
        self.max_records = max_records

    def emit(self, record: logging.LogRecord) -> None:
        entry: dict[str, Any] = {
            "ts": record.created,
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            entry["data"] = record.extra_data  # type: ignore[attr-defined]
        if len(self.records) < self.max_records:
            self.records.append(entry)

    def flush_to_file(self, path: str | Path) -> None:
        """Write all accumulated records to a JSON-lines file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps(rec) + "\n")

    def clear(self) -> None:
        self.records.clear()


# ---------------------------------------------------------------------------
# Module-level audit handler (singleton)
# ---------------------------------------------------------------------------

_audit_handler: AuditTrailHandler | None = None


def get_audit_handler() -> AuditTrailHandler:
    """Return the global audit-trail handler, creating one if needed."""
    global _audit_handler
    if _audit_handler is None:
        _audit_handler = AuditTrailHandler()
    return _audit_handler


# ---------------------------------------------------------------------------
# Performance-timer context manager
# ---------------------------------------------------------------------------


@contextmanager
def log_timing(
    logger: logging.Logger,
    description: str,
    level: int = logging.INFO,
) -> Iterator[dict[str, float]]:
    """Context manager that logs elapsed wall-clock time.

    Usage::

        with log_timing(logger, "CI testing"):
            run_ci_tests()

    Parameters
    ----------
    logger : logging.Logger
    description : str
    level : int
    """
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield timing
    finally:
        elapsed = time.perf_counter() - t0
        timing["elapsed_s"] = elapsed
        extra: dict[str, Any] = {"extra_data": {"elapsed_s": elapsed}}
        logger.log(
            level,
            "%s completed in %.3f s",
            description,
            elapsed,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Main configure function
# ---------------------------------------------------------------------------


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: str | None = None,
    enable_audit_trail: bool = False,
    module_levels: dict[str, str] | None = None,
) -> None:
    """Configure CausalCert logging.

    Parameters
    ----------
    level : str
        Root log level.
    json_format : bool
        If ``True``, use JSON formatting.
    log_file : str | None
        Optional file path for log output.
    enable_audit_trail : bool
        Attach a global :class:`AuditTrailHandler`.
    module_levels : dict[str, str] | None
        Per-module log levels, e.g. ``{"causalcert.solver": "DEBUG"}``.
    """
    root = logging.getLogger("causalcert")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # --- primary handler ---
    handler: logging.Handler
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stderr)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            PerfFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    root.addHandler(handler)

    # --- audit trail ---
    if enable_audit_trail:
        root.addHandler(get_audit_handler())

    # --- per-module overrides ---
    if module_levels:
        for mod, lvl in module_levels.items():
            logging.getLogger(mod).setLevel(
                getattr(logging, lvl.upper(), logging.INFO)
            )

    # Also honour the environment variable
    env_level = os.environ.get("CAUSALCERT_LOG_LEVEL")
    if env_level:
        root.setLevel(getattr(logging, env_level.upper(), logging.INFO))


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``causalcert`` namespace."""
    return logging.getLogger(f"causalcert.{name}")


def log_metric(
    logger: logging.Logger,
    metric_name: str,
    value: float,
    **extra: Any,
) -> None:
    """Emit a structured metric log entry."""
    data = {"metric": metric_name, "value": value, **extra}
    logger.info(
        "metric %s = %.6g",
        metric_name,
        value,
        extra={"extra_data": data},
    )
