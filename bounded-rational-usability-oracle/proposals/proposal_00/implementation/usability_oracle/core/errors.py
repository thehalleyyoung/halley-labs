"""
usability_oracle.core.errors — Exception hierarchy for the oracle pipeline.

Every public exception carries contextual attributes (``stage``, ``details``,
``context``) so that callers can construct rich diagnostics without string
parsing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


# ═══════════════════════════════════════════════════════════════════════════
# Base exception
# ═══════════════════════════════════════════════════════════════════════════

class UsabilityOracleError(Exception):
    """Root of the usability-oracle exception hierarchy.

    Parameters
    ----------
    message : str
        Human-readable description.
    stage : str | None
        Pipeline stage that raised the error (e.g. "parse", "align").
    details : dict | None
        Machine-readable context (ids, counts, thresholds ...).
    """

    def __init__(
        self,
        message: str = "",
        *,
        stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.details: Dict[str, Any] = details or {}

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.stage:
            parts.append(f"[stage={self.stage}]")
        if self.details:
            detail_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            parts.append(f"({detail_str})")
        return " ".join(parts)

    def __repr__(self) -> str:
        cls = type(self).__name__
        return (
            f"{cls}({super().__str__()!r}, stage={self.stage!r}, "
            f"details={self.details!r})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the error for structured logging / JSON output."""
        return {
            "error_type": type(self).__name__,
            "message": str(self),
            "stage": self.stage,
            "details": self.details,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Parsing errors
# ═══════════════════════════════════════════════════════════════════════════

class ParseError(UsabilityOracleError):
    """Raised when the parser cannot process the UI source."""

    def __init__(
        self,
        message: str = "Failed to parse UI source",
        *,
        source_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, stage="parse", details=details)
        self.source_type = source_type


class InvalidAccessibilityTreeError(ParseError):
    """The parsed tree violates structural invariants.

    For example: cycles, missing root, duplicated node ids.
    """

    def __init__(
        self,
        message: str = "Invalid accessibility tree structure",
        *,
        node_id: Optional[str] = None,
        invariant: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)
        self.node_id = node_id
        self.invariant = invariant

    def __str__(self) -> str:
        base = super().__str__()
        extras = []
        if self.node_id:
            extras.append(f"node_id={self.node_id!r}")
        if self.invariant:
            extras.append(f"invariant={self.invariant!r}")
        if extras:
            return f"{base} [{', '.join(extras)}]"
        return base


class MalformedHTMLError(ParseError):
    """The HTML source is syntactically malformed beyond recovery."""

    def __init__(
        self,
        message: str = "Malformed HTML",
        *,
        line: Optional[int] = None,
        column: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)
        self.line = line
        self.column = column

    def __str__(self) -> str:
        base = super().__str__()
        if self.line is not None:
            loc = f"line {self.line}"
            if self.column is not None:
                loc += f", col {self.column}"
            return f"{base} at {loc}"
        return base


# ═══════════════════════════════════════════════════════════════════════════
# Alignment errors
# ═══════════════════════════════════════════════════════════════════════════

class AlignmentError(UsabilityOracleError):
    """Raised when tree alignment fails."""

    def __init__(
        self,
        message: str = "Alignment failed",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, stage="align", details=details)


class IncompatibleTreesError(AlignmentError):
    """The two trees are structurally incompatible for alignment.

    For instance, completely disjoint role sets or vastly different depths.
    """

    def __init__(
        self,
        message: str = "Trees are structurally incompatible",
        *,
        tree_a_size: Optional[int] = None,
        tree_b_size: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = dict(details or {})
        if tree_a_size is not None:
            d["tree_a_size"] = tree_a_size
        if tree_b_size is not None:
            d["tree_b_size"] = tree_b_size
        super().__init__(message, details=d)
        self.tree_a_size = tree_a_size
        self.tree_b_size = tree_b_size


# ═══════════════════════════════════════════════════════════════════════════
# Cost model errors
# ═══════════════════════════════════════════════════════════════════════════

class CostModelError(UsabilityOracleError):
    """Raised when the cognitive cost model cannot compute a cost."""

    def __init__(
        self,
        message: str = "Cost model error",
        *,
        law: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, stage="cost", details=details)
        self.law = law


class InvalidParameterError(CostModelError):
    """A cognitive-model parameter is outside its valid range."""

    def __init__(
        self,
        message: str = "Invalid cognitive parameter",
        *,
        parameter_name: Optional[str] = None,
        value: Optional[float] = None,
        valid_range: Optional[tuple[float, float]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)
        self.parameter_name = parameter_name
        self.value = value
        self.valid_range = valid_range

    def __str__(self) -> str:
        base = super().__str__()
        if self.parameter_name:
            extra = f"{self.parameter_name}={self.value}"
            if self.valid_range:
                extra += f" (valid: {self.valid_range})"
            return f"{base} [{extra}]"
        return base


class ConvergenceError(CostModelError):
    """An iterative computation did not converge within the allowed budget."""

    def __init__(
        self,
        message: str = "Iterative computation did not converge",
        *,
        iterations: Optional[int] = None,
        residual: Optional[float] = None,
        threshold: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = dict(details or {})
        if iterations is not None:
            d["iterations"] = iterations
        if residual is not None:
            d["residual"] = residual
        if threshold is not None:
            d["threshold"] = threshold
        super().__init__(message, details=d)
        self.iterations = iterations
        self.residual = residual
        self.threshold = threshold


# ═══════════════════════════════════════════════════════════════════════════
# MDP errors
# ═══════════════════════════════════════════════════════════════════════════

class MDPError(UsabilityOracleError):
    """Raised when MDP construction or analysis fails."""

    def __init__(
        self,
        message: str = "MDP error",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, stage="mdp_build", details=details)


class StateSpaceExplosionError(MDPError):
    """The state space exceeds the configured maximum."""

    def __init__(
        self,
        message: str = "State space explosion",
        *,
        num_states: Optional[int] = None,
        max_states: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = dict(details or {})
        if num_states is not None:
            d["num_states"] = num_states
        if max_states is not None:
            d["max_states"] = max_states
        super().__init__(message, details=d)
        self.num_states = num_states
        self.max_states = max_states


class UnreachableStateError(MDPError):
    """A state required for task completion is unreachable."""

    def __init__(
        self,
        message: str = "Unreachable state detected",
        *,
        state_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = dict(details or {})
        if state_id is not None:
            d["state_id"] = state_id
        super().__init__(message, details=d)
        self.state_id = state_id


# ═══════════════════════════════════════════════════════════════════════════
# Policy errors
# ═══════════════════════════════════════════════════════════════════════════

class PolicyError(UsabilityOracleError):
    """Raised during bounded-rational policy computation."""

    def __init__(
        self,
        message: str = "Policy computation error",
        *,
        beta: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, stage="policy", details=details)
        self.beta = beta


class NumericalInstabilityError(PolicyError):
    """Numerical instability during softmax or log-sum-exp computation."""

    def __init__(
        self,
        message: str = "Numerical instability in policy computation",
        *,
        beta: Optional[float] = None,
        max_value: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = dict(details or {})
        if max_value is not None:
            d["max_value"] = max_value
        super().__init__(message, beta=beta, details=d)
        self.max_value = max_value


# ═══════════════════════════════════════════════════════════════════════════
# Bisimulation errors
# ═══════════════════════════════════════════════════════════════════════════

class BisimulationError(UsabilityOracleError):
    """Raised during bisimulation-based state-space reduction."""

    def __init__(
        self,
        message: str = "Bisimulation error",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, stage="bisimulate", details=details)


class PartitionError(BisimulationError):
    """The partition refinement algorithm failed (e.g. non-determinism)."""

    def __init__(
        self,
        message: str = "Partition refinement failed",
        *,
        num_partitions: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = dict(details or {})
        if num_partitions is not None:
            d["num_partitions"] = num_partitions
        super().__init__(message, details=d)
        self.num_partitions = num_partitions


# ═══════════════════════════════════════════════════════════════════════════
# Bottleneck errors
# ═══════════════════════════════════════════════════════════════════════════

class BottleneckError(UsabilityOracleError):
    """Raised during bottleneck analysis."""

    def __init__(
        self,
        message: str = "Bottleneck analysis error",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, stage="bottleneck", details=details)


class ClassificationError(BottleneckError):
    """The classifier could not assign a bottleneck category."""

    def __init__(
        self,
        message: str = "Bottleneck classification failed",
        *,
        candidates: Optional[Sequence[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = dict(details or {})
        if candidates is not None:
            d["candidates"] = list(candidates)
        super().__init__(message, details=d)
        self.candidates = candidates


# ═══════════════════════════════════════════════════════════════════════════
# Comparison errors
# ═══════════════════════════════════════════════════════════════════════════

class ComparisonError(UsabilityOracleError):
    """Raised during statistical comparison of usability metrics."""

    def __init__(
        self,
        message: str = "Comparison error",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, stage="compare", details=details)


class InsufficientDataError(ComparisonError):
    """Not enough data points for a reliable statistical test."""

    def __init__(
        self,
        message: str = "Insufficient data for comparison",
        *,
        required: Optional[int] = None,
        actual: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = dict(details or {})
        if required is not None:
            d["required"] = required
        if actual is not None:
            d["actual"] = actual
        super().__init__(message, details=d)
        self.required = required
        self.actual = actual


# ═══════════════════════════════════════════════════════════════════════════
# Repair errors
# ═══════════════════════════════════════════════════════════════════════════

class RepairError(UsabilityOracleError):
    """Raised during automated repair synthesis."""

    def __init__(
        self,
        message: str = "Repair synthesis error",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, stage="repair", details=details)


class SynthesisTimeoutError(RepairError):
    """The repair synthesiser exceeded its time budget."""

    def __init__(
        self,
        message: str = "Repair synthesis timed out",
        *,
        timeout_seconds: Optional[float] = None,
        candidates_found: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = dict(details or {})
        if timeout_seconds is not None:
            d["timeout_seconds"] = timeout_seconds
        d["candidates_found"] = candidates_found
        super().__init__(message, details=d)
        self.timeout_seconds = timeout_seconds
        self.candidates_found = candidates_found


class InfeasibleRepairError(RepairError):
    """No repair exists that satisfies the given constraints."""

    def __init__(
        self,
        message: str = "No feasible repair found",
        *,
        constraint_summary: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = dict(details or {})
        if constraint_summary:
            d["constraint_summary"] = constraint_summary
        super().__init__(message, details=d)
        self.constraint_summary = constraint_summary


# ═══════════════════════════════════════════════════════════════════════════
# Configuration / validation errors
# ═══════════════════════════════════════════════════════════════════════════

class ConfigError(UsabilityOracleError):
    """Raised for invalid or inconsistent configuration."""

    def __init__(
        self,
        message: str = "Configuration error",
        *,
        key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, stage="config", details=details)
        self.key = key


class ValidationError(ConfigError):
    """Raised when configuration validation fails."""

    def __init__(
        self,
        message: str = "Validation error",
        *,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        constraint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = dict(details or {})
        if field:
            d["field"] = field
        if value is not None:
            d["value"] = value
        if constraint:
            d["constraint"] = constraint
        super().__init__(message, details=d)
        self.field = field
        self.value = value
        self.constraint = constraint


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline / stage / cache errors
# ═══════════════════════════════════════════════════════════════════════════

class PipelineError(UsabilityOracleError):
    """Raised when the pipeline orchestrator encounters a fatal problem."""

    def __init__(
        self,
        message: str = "Pipeline error",
        *,
        failed_stages: Optional[Sequence[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = dict(details or {})
        if failed_stages:
            d["failed_stages"] = list(failed_stages)
        super().__init__(message, stage="pipeline", details=d)
        self.failed_stages = failed_stages or []


class StageError(PipelineError):
    """Raised when a specific pipeline stage fails."""

    def __init__(
        self,
        message: str = "Stage execution error",
        *,
        stage_name: Optional[str] = None,
        cause: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = dict(details or {})
        if stage_name:
            d["stage_name"] = stage_name
        super().__init__(
            message,
            failed_stages=[stage_name] if stage_name else None,
            details=d,
        )
        self.stage_name = stage_name
        self.__cause__ = cause


class CacheError(UsabilityOracleError):
    """Raised on cache read/write failures."""

    def __init__(
        self,
        message: str = "Cache error",
        *,
        key: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = dict(details or {})
        if key:
            d["key"] = key
        if operation:
            d["operation"] = operation
        super().__init__(message, stage="cache", details=d)
        self.key = key
        self.operation = operation


__all__ = [
    "UsabilityOracleError",
    "ParseError",
    "InvalidAccessibilityTreeError",
    "MalformedHTMLError",
    "AlignmentError",
    "IncompatibleTreesError",
    "CostModelError",
    "InvalidParameterError",
    "ConvergenceError",
    "MDPError",
    "StateSpaceExplosionError",
    "UnreachableStateError",
    "PolicyError",
    "NumericalInstabilityError",
    "BisimulationError",
    "PartitionError",
    "BottleneckError",
    "ClassificationError",
    "ComparisonError",
    "InsufficientDataError",
    "RepairError",
    "SynthesisTimeoutError",
    "InfeasibleRepairError",
    "ConfigError",
    "ValidationError",
    "PipelineError",
    "StageError",
    "CacheError",
]
