"""
Structured logging configuration for DP-Forge.

Provides a :func:`setup_logging` function that configures Python logging with
Rich-based pretty output for interactive use and structured JSON for
production.  All DP-Forge modules use ``logging.getLogger(__name__)`` and
emit structured records through this configuration.

Usage::

    from dp_forge.logging_config import setup_logging

    setup_logging(verbosity=2)  # DEBUG level with rich output
    setup_logging(verbosity=0, json_output=True)  # WARNING level, JSON lines

Verbosity mapping:
    - 0 → WARNING
    - 1 → INFO
    - 2 → DEBUG
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

_CONFIGURED = False

# DP-Forge root logger name
_ROOT_LOGGER = "dp_forge"

# Verbosity → logging level
_VERBOSITY_MAP = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


class _JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Includes structured ``extra`` fields that DP-Forge modules attach to
    records (e.g., ``iteration``, ``solver``, ``epsilon``).
    """

    _RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach any extra fields the caller set
        for key, val in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                try:
                    json.dumps(val)
                    payload[key] = val
                except (TypeError, ValueError):
                    payload[key] = repr(val)
        if record.exc_info and record.exc_info[1] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _RichFormatter(logging.Formatter):
    """Concise coloured formatter using Rich markup when available.

    Falls back to plain formatting if Rich is not installed.
    """

    _LEVEL_STYLES = {
        "DEBUG": "\033[36m",      # cyan
        "INFO": "\033[32m",       # green
        "WARNING": "\033[33m",    # yellow
        "ERROR": "\033[31m",      # red
        "CRITICAL": "\033[1;31m", # bold red
    }
    _RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname
        name = record.name.removeprefix("dp_forge.")
        msg = record.getMessage()

        if self.use_color:
            color = self._LEVEL_STYLES.get(level, "")
            prefix = f"{color}{ts} {level:<8}{self._RESET} [{name}]"
        else:
            prefix = f"{ts} {level:<8} [{name}]"

        formatted = f"{prefix} {msg}"

        if record.exc_info and record.exc_info[1] is not None:
            formatted += "\n" + self.formatException(record.exc_info)
        return formatted


def setup_logging(
    verbosity: int = 1,
    json_output: bool = False,
    log_file: Optional[str] = None,
    use_color: Optional[bool] = None,
) -> logging.Logger:
    """Configure DP-Forge logging.

    Args:
        verbosity: 0 = WARNING, 1 = INFO, 2 = DEBUG.
        json_output: If True, emit JSON lines instead of human-readable output.
        log_file: Optional path to write logs to a file (always JSON format).
        use_color: Force colour on/off for console output. ``None`` auto-detects.

    Returns:
        The configured ``dp_forge`` root logger.
    """
    global _CONFIGURED

    level = _VERBOSITY_MAP.get(verbosity, logging.DEBUG)
    root = logging.getLogger(_ROOT_LOGGER)
    root.setLevel(level)

    # Clear previous handlers on reconfiguration
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)

    if json_output:
        console.setFormatter(_JSONFormatter())
    else:
        if use_color is None:
            use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        console.setFormatter(_RichFormatter(use_color=use_color))

    root.addHandler(console)

    # Optional file handler (always JSON)
    if log_file is not None:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_JSONFormatter())
        root.addHandler(fh)

    # Suppress noisy third-party loggers
    for noisy in ("cvxpy", "scs", "mosek", "scipy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    root.debug("DP-Forge logging configured: verbosity=%d, json=%s", verbosity, json_output)
    return root


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the dp_forge namespace.

    Args:
        name: Module name (typically ``__name__``).

    Returns:
        A :class:`logging.Logger` that inherits dp_forge configuration.
    """
    if not name.startswith(_ROOT_LOGGER):
        name = f"{_ROOT_LOGGER}.{name}"
    return logging.getLogger(name)
