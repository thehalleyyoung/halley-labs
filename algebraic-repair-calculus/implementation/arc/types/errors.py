"""
ARC custom exception hierarchy.

Provides structured, informative exceptions for every failure mode in the
Algebraic Repair Calculus pipeline—from schema validation through repair
execution.  Every exception carries machine-readable context (error code,
offending object) alongside a human-readable message so that both CLI
output and programmatic callers can react precisely.
"""

from __future__ import annotations

import enum
from typing import Any


# ── Error codes ──────────────────────────────────────────────────────────

class ErrorCode(enum.Enum):
    """Machine-readable error identifiers for every ARC failure mode."""

    # Schema errors (1xxx)
    SCHEMA_INVALID = 1000
    SCHEMA_COLUMN_NOT_FOUND = 1001
    SCHEMA_TYPE_MISMATCH = 1002
    SCHEMA_CONSTRAINT_VIOLATION = 1003
    SCHEMA_DUPLICATE_COLUMN = 1004
    SCHEMA_PRIMARY_KEY_VIOLATION = 1005
    SCHEMA_FOREIGN_KEY_VIOLATION = 1006
    SCHEMA_NULLABLE_VIOLATION = 1007
    SCHEMA_DEFAULT_TYPE_MISMATCH = 1008
    SCHEMA_EMPTY = 1009

    # Type errors (2xxx)
    TYPE_MISMATCH = 2000
    TYPE_INCOMPATIBLE_WIDENING = 2001
    TYPE_NARROWING_LOSS = 2002
    TYPE_UNSUPPORTED = 2003
    TYPE_PARAMETER_INVALID = 2004
    TYPE_CAST_FAILURE = 2005

    # Delta errors (3xxx)
    DELTA_COMPOSITION_FAILURE = 3000
    DELTA_PROPAGATION_FAILURE = 3001
    DELTA_INVERSION_FAILURE = 3002
    DELTA_IDENTITY_VIOLATION = 3003
    DELTA_SORT_MISMATCH = 3004
    DELTA_INTERACTION_FAILURE = 3005
    DELTA_ANNIHILATION_ERROR = 3006

    # Planner errors (4xxx)
    PLANNER_FAILURE = 4000
    PLANNER_INFEASIBLE = 4001
    PLANNER_TIMEOUT = 4002
    PLANNER_CYCLE_DETECTED = 4003
    PLANNER_COST_OVERFLOW = 4004
    PLANNER_LP_INFEASIBLE = 4005
    PLANNER_ILP_TIMEOUT = 4006

    # Execution errors (5xxx)
    EXECUTION_FAILURE = 5000
    EXECUTION_CHECKPOINT_FAILURE = 5001
    EXECUTION_ROLLBACK_FAILURE = 5002
    EXECUTION_SAGA_COMPENSATION_FAILURE = 5003
    EXECUTION_PARTIAL_FAILURE = 5004
    EXECUTION_BACKEND_UNREACHABLE = 5005
    EXECUTION_TIMEOUT = 5006

    # Validation errors (6xxx)
    VALIDATION_FAILURE = 6000
    VALIDATION_FRAGMENT_VIOLATION = 6001
    VALIDATION_QUALITY_VIOLATION = 6002
    VALIDATION_AVAILABILITY_VIOLATION = 6003
    VALIDATION_SPEC_INVALID = 6004
    VALIDATION_VERSION_MISMATCH = 6005

    # Serialization errors (7xxx)
    SERIALIZATION_FAILURE = 7000
    SERIALIZATION_PARSE_ERROR = 7001
    SERIALIZATION_ENCODE_ERROR = 7002
    SERIALIZATION_SCHEMA_VIOLATION = 7003
    SERIALIZATION_VERSION_UNSUPPORTED = 7004
    SERIALIZATION_FILE_NOT_FOUND = 7005
    SERIALIZATION_PERMISSION_ERROR = 7006

    # Graph errors (8xxx)
    GRAPH_CYCLE_DETECTED = 8000
    GRAPH_NODE_NOT_FOUND = 8001
    GRAPH_EDGE_NOT_FOUND = 8002
    GRAPH_DISCONNECTED = 8003
    GRAPH_SCHEMA_MISMATCH = 8004
    GRAPH_MERGE_CONFLICT = 8005
    GRAPH_INVALID_TOPOLOGY = 8006

    # Quality errors (9xxx)
    QUALITY_DRIFT_DETECTED = 9000
    QUALITY_CONSTRAINT_FAILED = 9001
    QUALITY_THRESHOLD_EXCEEDED = 9002
    QUALITY_STATISTICAL_ERROR = 9003
    QUALITY_MONITORING_FAILURE = 9004


# ── Base exception ───────────────────────────────────────────────────────

class ARCError(Exception):
    """Root of the ARC exception hierarchy.

    Parameters
    ----------
    message:
        Human-readable description of the error.
    code:
        Machine-readable :class:`ErrorCode`.
    context:
        Arbitrary key/value pairs that give extra detail (column name,
        expected vs. actual type, node id, …).
    cause:
        Optional wrapped exception that triggered this one.
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode | None = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context: dict[str, Any] = context or {}
        self.cause = cause

    # -- Representation ---------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"{self.__class__.__name__}({self.args[0]!r}"]
        if self.code is not None:
            parts.append(f", code={self.code.name}")
        if self.context:
            parts.append(f", context={self.context!r}")
        parts.append(")")
        return "".join(parts)

    def __str__(self) -> str:
        prefix = f"[{self.code.name}] " if self.code else ""
        suffix = ""
        if self.context:
            details = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            suffix = f" ({details})"
        return f"{prefix}{self.args[0]}{suffix}"

    # -- Serialization helpers --------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        d: dict[str, Any] = {
            "error_type": self.__class__.__name__,
            "message": str(self.args[0]),
        }
        if self.code is not None:
            d["code"] = self.code.value
            d["code_name"] = self.code.name
        if self.context:
            d["context"] = self.context
        if self.cause is not None:
            d["cause"] = str(self.cause)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ARCError":
        """Reconstruct from a dictionary (best-effort)."""
        code = None
        if "code" in d:
            try:
                code = ErrorCode(d["code"])
            except ValueError:
                pass
        return cls(
            message=d.get("message", "unknown error"),
            code=code,
            context=d.get("context"),
        )


# ── Schema errors ────────────────────────────────────────────────────────

class SchemaError(ARCError):
    """Raised when a schema is structurally invalid."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.SCHEMA_INVALID,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context, cause=cause)


class TypeMismatchError(SchemaError):
    """Expected one SQL type but found another."""

    def __init__(
        self,
        column: str,
        expected: str,
        actual: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = {"column": column, "expected": expected, "actual": actual}
        if context:
            ctx.update(context)
        super().__init__(
            f"Type mismatch on column '{column}': expected {expected}, got {actual}",
            code=ErrorCode.TYPE_MISMATCH,
            context=ctx,
        )
        self.column = column
        self.expected = expected
        self.actual = actual


class ColumnNotFoundError(SchemaError):
    """Referenced column does not exist in the schema."""

    def __init__(
        self,
        column: str,
        available: list[str] | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        avail_str = ""
        if available:
            avail_str = f"; available columns: {', '.join(available)}"
        ctx: dict[str, Any] = {"column": column}
        if available:
            ctx["available"] = available
        if context:
            ctx.update(context)
        super().__init__(
            f"Column '{column}' not found{avail_str}",
            code=ErrorCode.SCHEMA_COLUMN_NOT_FOUND,
            context=ctx,
        )
        self.column = column
        self.available = available or []


class DuplicateColumnError(SchemaError):
    """Duplicate column name in schema."""

    def __init__(self, column: str) -> None:
        super().__init__(
            f"Duplicate column name: '{column}'",
            code=ErrorCode.SCHEMA_DUPLICATE_COLUMN,
            context={"column": column},
        )
        self.column = column


class PrimaryKeyViolationError(SchemaError):
    """Primary key constraint violated."""

    def __init__(self, columns: list[str], detail: str = "") -> None:
        msg = f"Primary key violation on columns {columns}"
        if detail:
            msg += f": {detail}"
        super().__init__(
            msg,
            code=ErrorCode.SCHEMA_PRIMARY_KEY_VIOLATION,
            context={"columns": columns, "detail": detail},
        )
        self.columns = columns


class ForeignKeyViolationError(SchemaError):
    """Foreign key constraint violated."""

    def __init__(
        self,
        source_columns: list[str],
        target_table: str,
        target_columns: list[str],
        detail: str = "",
    ) -> None:
        msg = (
            f"Foreign key violation: {source_columns} -> "
            f"{target_table}({target_columns})"
        )
        if detail:
            msg += f": {detail}"
        super().__init__(
            msg,
            code=ErrorCode.SCHEMA_FOREIGN_KEY_VIOLATION,
            context={
                "source_columns": source_columns,
                "target_table": target_table,
                "target_columns": target_columns,
                "detail": detail,
            },
        )


# ── Type errors ──────────────────────────────────────────────────────────

class TypeCompatibilityError(ARCError):
    """Cannot widen/narrow between two types."""

    def __init__(
        self,
        source_type: str,
        target_type: str,
        direction: str = "widening",
    ) -> None:
        super().__init__(
            f"Incompatible {direction}: {source_type} -> {target_type}",
            code=ErrorCode.TYPE_INCOMPATIBLE_WIDENING,
            context={
                "source_type": source_type,
                "target_type": target_type,
                "direction": direction,
            },
        )
        self.source_type = source_type
        self.target_type = target_type


class TypeParameterError(ARCError):
    """Invalid type parameter (e.g., VARCHAR(-1))."""

    def __init__(self, type_name: str, parameter: str, detail: str) -> None:
        super().__init__(
            f"Invalid parameter for {type_name}({parameter}): {detail}",
            code=ErrorCode.TYPE_PARAMETER_INVALID,
            context={
                "type_name": type_name,
                "parameter": parameter,
                "detail": detail,
            },
        )


class TypeCastError(ARCError):
    """Runtime type-cast failure."""

    def __init__(self, value: Any, target_type: str) -> None:
        super().__init__(
            f"Cannot cast {value!r} to {target_type}",
            code=ErrorCode.TYPE_CAST_FAILURE,
            context={"value": repr(value), "target_type": target_type},
        )


# ── Delta algebra errors ────────────────────────────────────────────────

class DeltaError(ARCError):
    """Base for delta-algebra failures."""


class DeltaCompositionError(DeltaError):
    """Two deltas cannot be composed."""

    def __init__(
        self,
        left_type: str,
        right_type: str,
        reason: str = "",
    ) -> None:
        msg = f"Cannot compose {left_type} ∘ {right_type}"
        if reason:
            msg += f": {reason}"
        super().__init__(
            msg,
            code=ErrorCode.DELTA_COMPOSITION_FAILURE,
            context={
                "left_type": left_type,
                "right_type": right_type,
                "reason": reason,
            },
        )


class DeltaPropagationError(DeltaError):
    """Delta cannot propagate through a transformation."""

    def __init__(
        self,
        delta_type: str,
        node_id: str,
        reason: str = "",
    ) -> None:
        msg = f"Cannot propagate {delta_type} through node '{node_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(
            msg,
            code=ErrorCode.DELTA_PROPAGATION_FAILURE,
            context={
                "delta_type": delta_type,
                "node_id": node_id,
                "reason": reason,
            },
        )


class DeltaInversionError(DeltaError):
    """Delta is not invertible."""

    def __init__(self, delta_type: str, reason: str = "") -> None:
        msg = f"Cannot invert delta of type {delta_type}"
        if reason:
            msg += f": {reason}"
        super().__init__(
            msg,
            code=ErrorCode.DELTA_INVERSION_FAILURE,
            context={"delta_type": delta_type, "reason": reason},
        )


class DeltaSortMismatchError(DeltaError):
    """Attempted operation across incompatible delta sorts."""

    def __init__(self, expected_sort: str, actual_sort: str) -> None:
        super().__init__(
            f"Delta sort mismatch: expected {expected_sort}, got {actual_sort}",
            code=ErrorCode.DELTA_SORT_MISMATCH,
            context={"expected_sort": expected_sort, "actual_sort": actual_sort},
        )


class DeltaInteractionError(DeltaError):
    """Cross-sort interaction homomorphism failed."""

    def __init__(self, source_sort: str, target_sort: str, reason: str = "") -> None:
        msg = f"Interaction {source_sort} -> {target_sort} failed"
        if reason:
            msg += f": {reason}"
        super().__init__(
            msg,
            code=ErrorCode.DELTA_INTERACTION_FAILURE,
            context={
                "source_sort": source_sort,
                "target_sort": target_sort,
                "reason": reason,
            },
        )


# ── Planner errors ──────────────────────────────────────────────────────

class PlannerError(ARCError):
    """Repair planner failed."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.PLANNER_FAILURE,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context, cause=cause)


class InfeasibleRepairError(PlannerError):
    """No valid repair exists under the given constraints."""

    def __init__(
        self,
        reason: str,
        violated_constraints: list[str] | None = None,
    ) -> None:
        ctx: dict[str, Any] = {"reason": reason}
        if violated_constraints:
            ctx["violated_constraints"] = violated_constraints
        super().__init__(
            f"Infeasible repair: {reason}",
            code=ErrorCode.PLANNER_INFEASIBLE,
            context=ctx,
        )
        self.violated_constraints = violated_constraints or []


class PlannerTimeoutError(PlannerError):
    """Planner exceeded time budget."""

    def __init__(self, budget_seconds: float, elapsed_seconds: float) -> None:
        super().__init__(
            f"Planner timeout: budget={budget_seconds:.1f}s, elapsed={elapsed_seconds:.1f}s",
            code=ErrorCode.PLANNER_TIMEOUT,
            context={
                "budget_seconds": budget_seconds,
                "elapsed_seconds": elapsed_seconds,
            },
        )


class PlannerCycleError(PlannerError):
    """Pipeline graph contains a cycle that blocks planning."""

    def __init__(self, cycle: list[str]) -> None:
        super().__init__(
            f"Cycle detected during planning: {' -> '.join(cycle)}",
            code=ErrorCode.PLANNER_CYCLE_DETECTED,
            context={"cycle": cycle},
        )


# ── Execution errors ────────────────────────────────────────────────────

class ExecutionError(ARCError):
    """Repair execution failed."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.EXECUTION_FAILURE,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context, cause=cause)


class CheckpointError(ExecutionError):
    """Checkpoint creation or restoration failed."""

    def __init__(self, node_id: str, reason: str = "") -> None:
        msg = f"Checkpoint failed for node '{node_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(
            msg,
            code=ErrorCode.EXECUTION_CHECKPOINT_FAILURE,
            context={"node_id": node_id, "reason": reason},
        )


class RollbackError(ExecutionError):
    """Rollback / compensating action failed."""

    def __init__(self, node_id: str, reason: str = "") -> None:
        msg = f"Rollback failed for node '{node_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(
            msg,
            code=ErrorCode.EXECUTION_ROLLBACK_FAILURE,
            context={"node_id": node_id, "reason": reason},
        )


class SagaCompensationError(ExecutionError):
    """A saga compensation step itself failed."""

    def __init__(
        self,
        step_index: int,
        node_id: str,
        original_error: str,
        compensation_error: str,
    ) -> None:
        super().__init__(
            (
                f"Saga compensation failed at step {step_index} (node '{node_id}'): "
                f"original={original_error}, compensation={compensation_error}"
            ),
            code=ErrorCode.EXECUTION_SAGA_COMPENSATION_FAILURE,
            context={
                "step_index": step_index,
                "node_id": node_id,
                "original_error": original_error,
                "compensation_error": compensation_error,
            },
        )


class PartialExecutionError(ExecutionError):
    """Some steps succeeded, others failed."""

    def __init__(
        self,
        completed_nodes: list[str],
        failed_nodes: list[str],
        detail: str = "",
    ) -> None:
        msg = (
            f"Partial execution: {len(completed_nodes)} completed, "
            f"{len(failed_nodes)} failed"
        )
        if detail:
            msg += f" — {detail}"
        super().__init__(
            msg,
            code=ErrorCode.EXECUTION_PARTIAL_FAILURE,
            context={
                "completed_nodes": completed_nodes,
                "failed_nodes": failed_nodes,
                "detail": detail,
            },
        )
        self.completed_nodes = completed_nodes
        self.failed_nodes = failed_nodes


class BackendUnreachableError(ExecutionError):
    """Cannot connect to an execution backend."""

    def __init__(self, backend: str, reason: str = "") -> None:
        msg = f"Backend unreachable: {backend}"
        if reason:
            msg += f" ({reason})"
        super().__init__(
            msg,
            code=ErrorCode.EXECUTION_BACKEND_UNREACHABLE,
            context={"backend": backend, "reason": reason},
        )


# ── Validation errors ───────────────────────────────────────────────────

class ValidationError(ARCError):
    """Generic validation failure."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.VALIDATION_FAILURE,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context, cause=cause)


class FragmentViolationError(ValidationError):
    """Pipeline / node is outside Fragment F."""

    def __init__(self, node_id: str, reasons: list[str]) -> None:
        super().__init__(
            f"Node '{node_id}' violates Fragment F: {'; '.join(reasons)}",
            code=ErrorCode.VALIDATION_FRAGMENT_VIOLATION,
            context={"node_id": node_id, "reasons": reasons},
        )
        self.node_id = node_id
        self.reasons = reasons


class QualityViolationError(ValidationError):
    """A quality constraint is violated."""

    def __init__(
        self,
        constraint_id: str,
        metric_name: str,
        threshold: float,
        observed: float,
    ) -> None:
        super().__init__(
            (
                f"Quality violation [{constraint_id}]: "
                f"{metric_name}={observed:.4f} exceeds threshold {threshold:.4f}"
            ),
            code=ErrorCode.VALIDATION_QUALITY_VIOLATION,
            context={
                "constraint_id": constraint_id,
                "metric_name": metric_name,
                "threshold": threshold,
                "observed": observed,
            },
        )


class AvailabilityViolationError(ValidationError):
    """An availability contract is breached."""

    def __init__(
        self,
        node_id: str,
        sla_percentage: float,
        actual_percentage: float,
    ) -> None:
        super().__init__(
            (
                f"Availability violation for '{node_id}': "
                f"SLA={sla_percentage:.2f}%, actual={actual_percentage:.2f}%"
            ),
            code=ErrorCode.VALIDATION_AVAILABILITY_VIOLATION,
            context={
                "node_id": node_id,
                "sla_percentage": sla_percentage,
                "actual_percentage": actual_percentage,
            },
        )


class SpecValidationError(ValidationError):
    """Pipeline specification file is invalid."""

    def __init__(self, path: str, errors: list[str]) -> None:
        super().__init__(
            f"Invalid pipeline spec '{path}': {len(errors)} error(s)",
            code=ErrorCode.VALIDATION_SPEC_INVALID,
            context={"path": path, "errors": errors},
        )
        self.path = path
        self.errors = errors


# ── Serialization errors ────────────────────────────────────────────────

class SerializationError(ARCError):
    """Base for serialization/deserialization failures."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.SERIALIZATION_FAILURE,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context, cause=cause)


class ParseError(SerializationError):
    """Cannot parse input data."""

    def __init__(self, source: str, reason: str, line: int | None = None) -> None:
        loc = f" at line {line}" if line is not None else ""
        super().__init__(
            f"Parse error in '{source}'{loc}: {reason}",
            code=ErrorCode.SERIALIZATION_PARSE_ERROR,
            context={"source": source, "reason": reason, "line": line},
        )


class EncodeError(SerializationError):
    """Cannot encode object for serialization."""

    def __init__(self, obj_type: str, reason: str) -> None:
        super().__init__(
            f"Cannot encode {obj_type}: {reason}",
            code=ErrorCode.SERIALIZATION_ENCODE_ERROR,
            context={"obj_type": obj_type, "reason": reason},
        )


class SchemaViolationError(SerializationError):
    """Serialized data does not match expected schema."""

    def __init__(self, violations: list[str], schema_version: str = "") -> None:
        ver = f" (schema v{schema_version})" if schema_version else ""
        super().__init__(
            f"Schema violation{ver}: {'; '.join(violations[:5])}",
            code=ErrorCode.SERIALIZATION_SCHEMA_VIOLATION,
            context={
                "violations": violations,
                "schema_version": schema_version,
            },
        )


class VersionUnsupportedError(SerializationError):
    """Spec version is not supported."""

    def __init__(self, version: str, supported: list[str]) -> None:
        super().__init__(
            f"Unsupported spec version '{version}'; supported: {supported}",
            code=ErrorCode.SERIALIZATION_VERSION_UNSUPPORTED,
            context={"version": version, "supported": supported},
        )


# ── Graph errors ────────────────────────────────────────────────────────

class GraphError(ARCError):
    """Base for pipeline-graph structural errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.GRAPH_INVALID_TOPOLOGY,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context, cause=cause)


class CycleDetectedError(GraphError):
    """Pipeline DAG contains a cycle."""

    def __init__(self, cycle: list[str]) -> None:
        super().__init__(
            f"Cycle detected: {' -> '.join(cycle)}",
            code=ErrorCode.GRAPH_CYCLE_DETECTED,
            context={"cycle": cycle},
        )
        self.cycle = cycle


class NodeNotFoundError(GraphError):
    """Referenced node does not exist."""

    def __init__(self, node_id: str) -> None:
        super().__init__(
            f"Node '{node_id}' not found in pipeline graph",
            code=ErrorCode.GRAPH_NODE_NOT_FOUND,
            context={"node_id": node_id},
        )
        self.node_id = node_id


class EdgeNotFoundError(GraphError):
    """Referenced edge does not exist."""

    def __init__(self, source: str, target: str) -> None:
        super().__init__(
            f"Edge '{source}' -> '{target}' not found",
            code=ErrorCode.GRAPH_EDGE_NOT_FOUND,
            context={"source": source, "target": target},
        )
        self.source = source
        self.target = target


class GraphSchemaMismatchError(GraphError):
    """Output schema of source does not match input schema of target."""

    def __init__(
        self,
        source: str,
        target: str,
        mismatched_columns: list[str],
    ) -> None:
        super().__init__(
            (
                f"Schema mismatch on edge '{source}' -> '{target}': "
                f"columns {mismatched_columns}"
            ),
            code=ErrorCode.GRAPH_SCHEMA_MISMATCH,
            context={
                "source": source,
                "target": target,
                "mismatched_columns": mismatched_columns,
            },
        )


class GraphMergeConflictError(GraphError):
    """Conflict while merging two pipeline graphs."""

    def __init__(self, conflicting_nodes: list[str]) -> None:
        super().__init__(
            f"Merge conflict on nodes: {conflicting_nodes}",
            code=ErrorCode.GRAPH_MERGE_CONFLICT,
            context={"conflicting_nodes": conflicting_nodes},
        )


# ── Quality errors ──────────────────────────────────────────────────────

class QualityError(ARCError):
    """Base for data-quality failures."""


class QualityDriftError(QualityError):
    """Statistical drift detected."""

    def __init__(
        self,
        column: str,
        metric: str,
        p_value: float,
        threshold: float,
    ) -> None:
        super().__init__(
            (
                f"Quality drift on '{column}': {metric} p-value={p_value:.6f} "
                f"< threshold={threshold:.6f}"
            ),
            code=ErrorCode.QUALITY_DRIFT_DETECTED,
            context={
                "column": column,
                "metric": metric,
                "p_value": p_value,
                "threshold": threshold,
            },
        )


class QualityConstraintFailedError(QualityError):
    """A declared quality constraint failed."""

    def __init__(self, constraint_id: str, detail: str) -> None:
        super().__init__(
            f"Quality constraint '{constraint_id}' failed: {detail}",
            code=ErrorCode.QUALITY_CONSTRAINT_FAILED,
            context={"constraint_id": constraint_id, "detail": detail},
        )


class QualityThresholdExceededError(QualityError):
    """A quality threshold was exceeded."""

    def __init__(
        self,
        metric: str,
        value: float,
        threshold: float,
    ) -> None:
        super().__init__(
            f"Quality threshold exceeded: {metric}={value:.4f} > {threshold:.4f}",
            code=ErrorCode.QUALITY_THRESHOLD_EXCEEDED,
            context={
                "metric": metric,
                "value": value,
                "threshold": threshold,
            },
        )


# ── Convenience helpers ─────────────────────────────────────────────────

def chain_errors(*errors: ARCError) -> ARCError:
    """Combine multiple errors into a single composite error.

    The first error is returned with the rest attached as context.
    """
    if not errors:
        return ARCError("No errors provided")
    if len(errors) == 1:
        return errors[0]
    head = errors[0]
    head.context["chained_errors"] = [e.to_dict() for e in errors[1:]]
    return head


def wrap(exc: Exception, arc_cls: type[ARCError] = ARCError) -> ARCError:
    """Wrap a stdlib exception as an ARCError, preserving the traceback."""
    return arc_cls(str(exc), cause=exc)
