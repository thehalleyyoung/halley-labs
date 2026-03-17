"""
taintflow.core.errors – Exception hierarchy for TaintFlow.

Every public exception carries:
  * a human-readable message,
  * an ``error_code`` string  (e.g. ``"TF-CFG-001"``),
  * an optional :class:`ErrorContext` with structured diagnostic data,
  * an optional ``suggestion`` string with a recommended fix.

Error-code prefixes
-------------------
TF-CFG  configuration errors
TF-VAL  validation errors
TF-INS  instrumentation / monkey-patching errors
TF-DAG  DAG construction errors
TF-CAP  capacity / numerical errors
TF-ANA  abstract-interpretation / analysis errors
TF-ATT  attribution / flow-computation errors
TF-RPT  reporting / serialization errors
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
#  Error context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErrorContext:
    """Structured diagnostic data attached to a :class:`TaintFlowError`."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    phase: str = ""
    node_id: str = ""
    operation: str = ""
    column: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "timestamp": self.timestamp,
        }
        if self.phase:
            out["phase"] = self.phase
        if self.node_id:
            out["node_id"] = self.node_id
        if self.operation:
            out["operation"] = self.operation
        if self.column:
            out["column"] = self.column
        if self.extra:
            out["extra"] = dict(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ErrorContext":
        return cls(
            timestamp=str(data.get("timestamp", "")),
            phase=str(data.get("phase", "")),
            node_id=str(data.get("node_id", "")),
            operation=str(data.get("operation", "")),
            column=str(data.get("column", "")),
            extra=dict(data.get("extra", {})),
        )

    def __str__(self) -> str:
        parts: list[str] = []
        if self.phase:
            parts.append(f"phase={self.phase}")
        if self.node_id:
            parts.append(f"node={self.node_id}")
        if self.operation:
            parts.append(f"op={self.operation}")
        if self.column:
            parts.append(f"col={self.column}")
        return ", ".join(parts) if parts else "(no context)"


# ---------------------------------------------------------------------------
#  Base exception
# ---------------------------------------------------------------------------

class TaintFlowError(Exception):
    """Root of the TaintFlow exception hierarchy."""

    default_code: str = "TF-000"
    default_suggestion: str = ""

    def __init__(
        self,
        message: str = "",
        *,
        error_code: str | None = None,
        context: ErrorContext | None = None,
        suggestion: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.error_code: str = error_code or self.default_code
        self.context: ErrorContext = context or ErrorContext()
        self.suggestion: str = suggestion or self.default_suggestion
        self._cause = cause
        full = self._build_message(message)
        super().__init__(full)
        if cause is not None:
            self.__cause__ = cause

    # -- helpers -------------------------------------------------------------

    def _build_message(self, message: str) -> str:
        parts: list[str] = [f"[{self.error_code}]"]
        if message:
            parts.append(message)
        ctx_str = str(self.context)
        if ctx_str != "(no context)":
            parts.append(f"({ctx_str})")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": type(self).__name__,
            "error_code": self.error_code,
            "message": str(self),
            "context": self.context.to_dict(),
            "suggestion": self.suggestion,
        }

    @property
    def formatted_traceback(self) -> str:
        if self.__traceback__ is not None:
            return "".join(traceback.format_tb(self.__traceback__))
        return ""

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(code={self.error_code!r}, "
            f"msg={str(self)!r})"
        )


# ---------------------------------------------------------------------------
#  Configuration errors  (TF-CFG-*)
# ---------------------------------------------------------------------------

class ConfigError(TaintFlowError):
    """Raised when the configuration is invalid or cannot be loaded."""

    default_code = "TF-CFG-001"
    default_suggestion = "Check your taintflow.toml or environment variables."

    def __init__(
        self,
        message: str = "Invalid configuration",
        *,
        key: str = "",
        value: Any = None,
        **kwargs: Any,
    ) -> None:
        extra: dict[str, Any] = {}
        if key:
            extra["key"] = key
        if value is not None:
            extra["value"] = repr(value)
        ctx = kwargs.pop("context", None) or ErrorContext(extra=extra)
        super().__init__(message, context=ctx, **kwargs)
        self.key = key
        self.value = value


class ValidationError(TaintFlowError):
    """Raised when a value fails validation (schemas, ranges, types)."""

    default_code = "TF-VAL-001"
    default_suggestion = "Review the value and the expected constraints."

    def __init__(
        self,
        message: str = "Validation failed",
        *,
        field_name: str = "",
        expected: str = "",
        actual: Any = None,
        violations: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        extra: dict[str, Any] = {}
        if field_name:
            extra["field"] = field_name
        if expected:
            extra["expected"] = expected
        if actual is not None:
            extra["actual"] = repr(actual)
        if violations:
            extra["violations"] = list(violations)
        ctx = kwargs.pop("context", None) or ErrorContext(extra=extra)
        super().__init__(message, context=ctx, **kwargs)
        self.field_name = field_name
        self.expected = expected
        self.actual = actual
        self.violations: tuple[str, ...] = tuple(violations)


# ---------------------------------------------------------------------------
#  Instrumentation errors  (TF-INS-*)
# ---------------------------------------------------------------------------

class InstrumentationError(TaintFlowError):
    """Base for errors during monkey-patching or tracing setup."""

    default_code = "TF-INS-001"
    default_suggestion = "Ensure the target library version is supported."


class TraceError(InstrumentationError):
    """Raised when an execution trace cannot be recorded."""

    default_code = "TF-INS-002"
    default_suggestion = (
        "Verify the pipeline script runs without errors before auditing."
    )

    def __init__(
        self,
        message: str = "Trace recording failed",
        *,
        frame_info: str = "",
        **kwargs: Any,
    ) -> None:
        extra: dict[str, Any] = {}
        if frame_info:
            extra["frame"] = frame_info
        ctx = kwargs.pop("context", None) or ErrorContext(extra=extra)
        super().__init__(message, context=ctx, **kwargs)
        self.frame_info = frame_info


class MonkeyPatchError(InstrumentationError):
    """Raised when a monkey-patch cannot be applied or reverted."""

    default_code = "TF-INS-003"
    default_suggestion = (
        "Check for conflicting patches or unsupported library versions."
    )

    def __init__(
        self,
        message: str = "Monkey-patch failed",
        *,
        target_module: str = "",
        target_attr: str = "",
        **kwargs: Any,
    ) -> None:
        extra: dict[str, Any] = {}
        if target_module:
            extra["module"] = target_module
        if target_attr:
            extra["attr"] = target_attr
        ctx = kwargs.pop("context", None) or ErrorContext(extra=extra)
        super().__init__(message, context=ctx, **kwargs)
        self.target_module = target_module
        self.target_attr = target_attr


# ---------------------------------------------------------------------------
#  DAG construction errors  (TF-DAG-*)
# ---------------------------------------------------------------------------

class DAGConstructionError(TaintFlowError):
    """Raised when the pipeline DAG cannot be built."""

    default_code = "TF-DAG-001"
    default_suggestion = (
        "Ensure every data dependency is expressed through traced operations."
    )


class CycleDetectedError(DAGConstructionError):
    """Raised when the pipeline graph contains a cycle."""

    default_code = "TF-DAG-002"
    default_suggestion = (
        "Break the cycle by materialising intermediate results."
    )

    def __init__(
        self,
        message: str = "Cycle detected in pipeline graph",
        *,
        cycle_nodes: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        extra: dict[str, Any] = {}
        if cycle_nodes:
            extra["cycle"] = list(cycle_nodes)
        ctx = kwargs.pop("context", None) or ErrorContext(extra=extra)
        super().__init__(message, context=ctx, **kwargs)
        self.cycle_nodes: tuple[str, ...] = tuple(cycle_nodes)


class MissingNodeError(DAGConstructionError):
    """Raised when a referenced node does not exist in the DAG."""

    default_code = "TF-DAG-003"
    default_suggestion = (
        "Verify that all referenced pipeline stages exist."
    )

    def __init__(
        self,
        message: str = "Referenced node not found",
        *,
        node_id: str = "",
        available_nodes: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        extra: dict[str, Any] = {}
        if node_id:
            extra["missing"] = node_id
        if available_nodes:
            extra["available"] = list(available_nodes)
        ctx = kwargs.pop("context", None) or ErrorContext(node_id=node_id, extra=extra)
        super().__init__(message, context=ctx, **kwargs)
        self.missing_node = node_id
        self.available_nodes: tuple[str, ...] = tuple(available_nodes)


# ---------------------------------------------------------------------------
#  Capacity / numerical errors  (TF-CAP-*)
# ---------------------------------------------------------------------------

class CapacityComputationError(TaintFlowError):
    """Raised when a channel-capacity bound cannot be computed."""

    default_code = "TF-CAP-001"
    default_suggestion = (
        "Try increasing numerical precision or switching capacity tier."
    )

    def __init__(
        self,
        message: str = "Capacity computation failed",
        *,
        channel: str = "",
        tier: str = "",
        **kwargs: Any,
    ) -> None:
        extra: dict[str, Any] = {}
        if channel:
            extra["channel"] = channel
        if tier:
            extra["tier"] = tier
        ctx = kwargs.pop("context", None) or ErrorContext(extra=extra)
        super().__init__(message, context=ctx, **kwargs)
        self.channel = channel
        self.tier = tier


class NumericalInstabilityError(CapacityComputationError):
    """Raised when floating-point instability invalidates a result."""

    default_code = "TF-CAP-002"
    default_suggestion = (
        "Increase precision (use float128) or simplify the pipeline stage."
    )

    def __init__(
        self,
        message: str = "Numerical instability detected",
        *,
        value: float | None = None,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> None:
        extra: dict[str, Any] = {}
        if value is not None:
            extra["value"] = value
        if threshold is not None:
            extra["threshold"] = threshold
        ctx = kwargs.pop("context", None) or ErrorContext(extra=extra)
        super().__init__(message, context=ctx, **kwargs)
        self.unstable_value = value
        self.threshold = threshold


# ---------------------------------------------------------------------------
#  Analysis errors  (TF-ANA-*)
# ---------------------------------------------------------------------------

class AnalysisError(TaintFlowError):
    """Base for errors during abstract interpretation."""

    default_code = "TF-ANA-001"
    default_suggestion = (
        "Review the pipeline for unsupported patterns."
    )


class FixpointDivergenceError(AnalysisError):
    """Raised when the fixpoint iteration does not converge."""

    default_code = "TF-ANA-002"
    default_suggestion = (
        "Lower max_iterations or enable widening (config.use_widening=True)."
    )

    def __init__(
        self,
        message: str = "Fixpoint iteration did not converge",
        *,
        iterations: int = 0,
        max_iterations: int = 0,
        residual: float | None = None,
        **kwargs: Any,
    ) -> None:
        extra: dict[str, Any] = {}
        if iterations:
            extra["iterations"] = iterations
        if max_iterations:
            extra["max_iterations"] = max_iterations
        if residual is not None:
            extra["residual"] = residual
        ctx = kwargs.pop("context", None) or ErrorContext(extra=extra)
        super().__init__(message, context=ctx, **kwargs)
        self.iterations = iterations
        self.max_iterations = max_iterations
        self.residual = residual


class TransferFunctionError(AnalysisError):
    """Raised when a transfer function cannot be evaluated."""

    default_code = "TF-ANA-003"
    default_suggestion = (
        "Ensure the operation is among the supported OpType values."
    )

    def __init__(
        self,
        message: str = "Transfer function evaluation failed",
        *,
        op_type: str = "",
        **kwargs: Any,
    ) -> None:
        extra: dict[str, Any] = {}
        if op_type:
            extra["op_type"] = op_type
        ctx = kwargs.pop("context", None) or ErrorContext(operation=op_type, extra=extra)
        super().__init__(message, context=ctx, **kwargs)
        self.op_type = op_type


# ---------------------------------------------------------------------------
#  Attribution / flow-computation errors  (TF-ATT-*)
# ---------------------------------------------------------------------------

class AttributionError(TaintFlowError):
    """Raised when taint attribution cannot be computed."""

    default_code = "TF-ATT-001"
    default_suggestion = (
        "Check that the pipeline graph has a valid topological ordering."
    )


class FlowComputationError(AttributionError):
    """Raised when information-flow bounds fail to compute."""

    default_code = "TF-ATT-002"
    default_suggestion = (
        "Inspect the channel parameters for the failing edge."
    )

    def __init__(
        self,
        message: str = "Flow computation failed",
        *,
        source_node: str = "",
        target_node: str = "",
        **kwargs: Any,
    ) -> None:
        extra: dict[str, Any] = {}
        if source_node:
            extra["source"] = source_node
        if target_node:
            extra["target"] = target_node
        ctx = kwargs.pop("context", None) or ErrorContext(extra=extra)
        super().__init__(message, context=ctx, **kwargs)
        self.source_node = source_node
        self.target_node = target_node


# ---------------------------------------------------------------------------
#  Report / serialization errors  (TF-RPT-*)
# ---------------------------------------------------------------------------

class ReportError(TaintFlowError):
    """Raised when report generation fails."""

    default_code = "TF-RPT-001"
    default_suggestion = (
        "Check write permissions and available disk space."
    )

    def __init__(
        self,
        message: str = "Report generation failed",
        *,
        output_path: str = "",
        **kwargs: Any,
    ) -> None:
        extra: dict[str, Any] = {}
        if output_path:
            extra["output_path"] = output_path
        ctx = kwargs.pop("context", None) or ErrorContext(extra=extra)
        super().__init__(message, context=ctx, **kwargs)
        self.output_path = output_path


class SerializationError(ReportError):
    """Raised when an object cannot be serialised to the target format."""

    default_code = "TF-RPT-002"
    default_suggestion = (
        "Ensure all values are JSON/msgpack-serialisable."
    )

    def __init__(
        self,
        message: str = "Serialization failed",
        *,
        target_format: str = "",
        obj_type: str = "",
        **kwargs: Any,
    ) -> None:
        extra: dict[str, Any] = {}
        if target_format:
            extra["format"] = target_format
        if obj_type:
            extra["type"] = obj_type
        ctx = kwargs.pop("context", None) or ErrorContext(extra=extra)
        super().__init__(message, context=ctx, **kwargs)
        self.target_format = target_format
        self.obj_type = obj_type


# ---------------------------------------------------------------------------
#  Utility: collect all error classes
# ---------------------------------------------------------------------------

ALL_ERROR_CLASSES: tuple[type[TaintFlowError], ...] = (
    TaintFlowError,
    ConfigError,
    ValidationError,
    InstrumentationError,
    TraceError,
    MonkeyPatchError,
    DAGConstructionError,
    CycleDetectedError,
    MissingNodeError,
    CapacityComputationError,
    NumericalInstabilityError,
    AnalysisError,
    FixpointDivergenceError,
    TransferFunctionError,
    AttributionError,
    FlowComputationError,
    ReportError,
    SerializationError,
)

_CODE_TO_CLASS: dict[str, type[TaintFlowError]] = {
    cls.default_code: cls for cls in ALL_ERROR_CLASSES
}


def error_class_for_code(code: str) -> type[TaintFlowError] | None:
    """Look up the error class for a given error code string."""
    return _CODE_TO_CLASS.get(code)


def all_error_codes() -> list[str]:
    """Return every registered error code, sorted."""
    return sorted(_CODE_TO_CLASS.keys())
