"""
Rich error hierarchy for BioProver.

Provides structured, actionable error types for every subsystem: model parsing,
solver execution, SMT encoding, CEGAR verification, repair synthesis, and
compositional reasoning.  Each error carries diagnostic context, cause chains,
and human-readable suggestions so that callers can report *why* something failed
and *what to try next*.
"""

from __future__ import annotations

import traceback
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Generator, List, Optional, Sequence, Type


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

class ErrorSeverity(Enum):
    """How severe an error is for the overall verification pipeline."""
    WARNING = auto()
    ERROR = auto()
    FATAL = auto()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class BioProverError(Exception):
    """Base exception for every BioProver error.

    Attributes:
        message: Human-readable description.
        context: Arbitrary key/value diagnostic information.
        suggestions: Actionable hints for the user or calling code.
        severity: WARNING / ERROR / FATAL.
        cause: The underlying exception, if any.
    """

    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    severity: ErrorSeverity = ErrorSeverity.ERROR
    cause: Optional[BaseException] = None

    def __post_init__(self) -> None:
        super().__init__(self.message)
        if self.cause is not None:
            self.__cause__ = self.cause

    # -- helpers -------------------------------------------------------------

    def with_context(self, **kwargs: Any) -> "BioProverError":
        """Return *self* after merging extra context (fluent API)."""
        self.context.update(kwargs)
        return self

    def with_suggestion(self, suggestion: str) -> "BioProverError":
        """Append one actionable suggestion."""
        self.suggestions.append(suggestion)
        return self

    def chain(self, cause: BaseException) -> "BioProverError":
        """Set the underlying cause and Python ``__cause__``."""
        self.cause = cause
        self.__cause__ = cause
        return self

    @property
    def diagnostic_summary(self) -> str:
        """One-liner suitable for log messages."""
        tag = type(self).__name__
        return f"[{self.severity.name}] {tag}: {self.message}"

    def full_report(self) -> str:
        """Multi-line human-readable report with context and suggestions."""
        return ErrorFormatter.format(self)

    def __str__(self) -> str:
        return self.diagnostic_summary


# ---------------------------------------------------------------------------
# Model-layer errors
# ---------------------------------------------------------------------------

class ModelError(BioProverError):
    """Error related to biological model construction or manipulation."""
    pass


class ParseError(BioProverError):
    """Failure to parse a model file (SBML, BNGL, custom DSL)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if "format" not in self.context:
            self.context.setdefault("format", "unknown")
        if not self.suggestions:
            self.suggestions.append(
                "Check that the input file conforms to the expected format."
            )


class ValidationError(BioProverError):
    """Model passed parsing but failed semantic validation.

    Examples: negative initial concentrations, undefined species references,
    stoichiometry that violates conservation laws.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.suggestions:
            self.suggestions.append(
                "Run the model through BioProver's validator for details."
            )


# ---------------------------------------------------------------------------
# Solver-layer errors
# ---------------------------------------------------------------------------

class SolverError(BioProverError):
    """Generic solver failure (ODE, SSA, hybrid)."""
    pass


class TimeoutError(BioProverError):
    """A solver or verification procedure exceeded its time budget."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.severity = ErrorSeverity.ERROR
        if not self.suggestions:
            self.suggestions.append(
                "Increase the timeout or reduce model complexity."
            )


class NumericalError(BioProverError):
    """Numerical instability during integration or evaluation.

    Typical causes: stiff ODEs, near-singular Jacobians, overflow/underflow.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.suggestions:
            self.suggestions.extend([
                "Try a stiffer ODE solver (e.g., switch from RK45 to BDF).",
                "Reduce the simulation step size.",
                "Check for parameter values spanning many orders of magnitude.",
            ])


# ---------------------------------------------------------------------------
# SMT / encoding errors
# ---------------------------------------------------------------------------

class EncodingError(BioProverError):
    """Failure while encoding a biological property into SMT constraints."""
    pass


class SMTError(BioProverError):
    """Error returned by the SMT solver backend (dReal, Z3, etc.)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.suggestions:
            self.suggestions.append(
                "Check solver installation and version compatibility."
            )


# ---------------------------------------------------------------------------
# CEGAR errors
# ---------------------------------------------------------------------------

class VerificationError(BioProverError):
    """The verification procedure failed (not a *disproof* – an internal error)."""
    pass


class RefinementError(BioProverError):
    """Abstraction refinement could not make progress.

    This usually means the abstraction/concretisation gap cannot be closed
    with the current predicate library.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.suggestions:
            self.suggestions.extend([
                "Add more predicates to the refinement library.",
                "Increase the maximum number of CEGAR iterations.",
            ])


# ---------------------------------------------------------------------------
# Repair / synthesis errors
# ---------------------------------------------------------------------------

class RepairError(BioProverError):
    """Failure in the automated repair / parameter-tuning pipeline."""
    pass


class SynthesisError(BioProverError):
    """CEGIS-based synthesis loop failed to find a valid candidate."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.suggestions:
            self.suggestions.extend([
                "Widen the parameter search bounds.",
                "Increase the CEGIS iteration budget.",
                "Relax the specification if it is over-constrained.",
            ])


# ---------------------------------------------------------------------------
# Compositional errors
# ---------------------------------------------------------------------------

class CompositionError(BioProverError):
    """Failure during compositional (assume-guarantee) verification.

    Carries information about which module interface was violated.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.suggestions:
            self.suggestions.append(
                "Check interface contracts between composed modules."
            )


# ---------------------------------------------------------------------------
# Error formatter
# ---------------------------------------------------------------------------

class ErrorFormatter:
    """Renders :class:`BioProverError` instances into rich, human-readable text."""

    _SEPARATOR = "─" * 60

    @classmethod
    def format(cls, error: BioProverError) -> str:
        """Produce a full diagnostic report for *error*."""
        parts: List[str] = []
        parts.append(cls._SEPARATOR)
        parts.append(f"  {error.severity.name}  {type(error).__name__}")
        parts.append(cls._SEPARATOR)
        parts.append(f"Message : {error.message}")

        if error.context:
            parts.append("")
            parts.append("Context:")
            for key, value in error.context.items():
                parts.append(f"  {key}: {value}")

        if error.suggestions:
            parts.append("")
            parts.append("Suggestions:")
            for idx, s in enumerate(error.suggestions, 1):
                parts.append(f"  {idx}. {s}")

        if error.cause is not None:
            parts.append("")
            parts.append("Caused by:")
            cause_lines = traceback.format_exception(
                type(error.cause), error.cause, error.cause.__traceback__
            )
            parts.append(textwrap.indent("".join(cause_lines), "  "))

        parts.append(cls._SEPARATOR)
        return "\n".join(parts)

    @classmethod
    def format_chain(cls, errors: Sequence[BioProverError]) -> str:
        """Format a chain of errors (e.g., from nested CEGAR iterations)."""
        if not errors:
            return "(no errors)"
        sections = [cls.format(e) for e in errors]
        return "\n\n".join(sections)

    @classmethod
    def one_line(cls, error: BioProverError) -> str:
        """Single-line summary suitable for log output."""
        return error.diagnostic_summary


# ---------------------------------------------------------------------------
# Context manager for enriching errors
# ---------------------------------------------------------------------------

@contextmanager
def error_context(
    wrap_as: Type[BioProverError] = BioProverError,
    message: str = "Unexpected error",
    **ctx: Any,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager that catches exceptions and re-raises as *wrap_as*.

    Usage::

        with error_context(SolverError, message="ODE solve failed", model="toggle"):
            result = solver.solve(model)

    Any non-:class:`BioProverError` exception is caught, wrapped in *wrap_as*
    with the given *message* and *ctx*, and re-raised.  Existing
    :class:`BioProverError` instances are enriched with extra context but
    otherwise left alone.
    """
    extra: Dict[str, Any] = dict(ctx)
    try:
        yield extra
    except BioProverError as bp_err:
        bp_err.context.update(extra)
        raise
    except Exception as exc:
        raise wrap_as(
            message=message,
            context=extra,
            cause=exc,
        ) from exc


# ---------------------------------------------------------------------------
# Error collection
# ---------------------------------------------------------------------------

@dataclass
class ErrorCollector:
    """Accumulates multiple non-fatal errors for batch reporting.

    Useful during model validation or multi-property verification where we
    want to report *all* problems rather than stopping at the first one.
    """

    errors: List[BioProverError] = field(default_factory=list)

    def add(self, error: BioProverError) -> None:
        self.errors.append(error)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def fatal_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == ErrorSeverity.FATAL)

    def raise_if_fatal(self) -> None:
        """Raise the first FATAL error, if any."""
        for e in self.errors:
            if e.severity == ErrorSeverity.FATAL:
                raise e

    def summary(self) -> str:
        if not self.errors:
            return "No errors."
        lines = [f"Collected {len(self.errors)} error(s):"]
        for e in self.errors:
            lines.append(f"  - {e.diagnostic_summary}")
        return "\n".join(lines)

    def full_report(self) -> str:
        return ErrorFormatter.format_chain(self.errors)
