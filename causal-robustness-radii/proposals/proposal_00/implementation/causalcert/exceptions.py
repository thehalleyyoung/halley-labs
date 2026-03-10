"""
Custom exception hierarchy for CausalCert.

All CausalCert exceptions inherit from :class:`CausalCertError` so that
callers can catch the full family with a single ``except`` clause.
"""

from __future__ import annotations

from causalcert.types import EdgeTuple, NodeId


class CausalCertError(Exception):
    """Base exception for all CausalCert errors."""


# ---------------------------------------------------------------------------
# DAG errors
# ---------------------------------------------------------------------------


class DAGError(CausalCertError):
    """Error related to DAG construction or validation."""


class CyclicGraphError(DAGError):
    """Raised when a purported DAG contains a cycle."""

    def __init__(self, cycle: list[NodeId] | None = None) -> None:
        self.cycle = cycle
        msg = "Graph contains a cycle"
        if cycle:
            msg += f": {' -> '.join(map(str, cycle))}"
        super().__init__(msg)


class InvalidEdgeError(DAGError):
    """Raised when an edge operation is invalid."""

    def __init__(self, edge: EdgeTuple, reason: str = "") -> None:
        self.edge = edge
        msg = f"Invalid edge {edge}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class NodeNotFoundError(DAGError):
    """Raised when a referenced node does not exist in the DAG."""

    def __init__(self, node: NodeId, n_nodes: int) -> None:
        self.node = node
        self.n_nodes = n_nodes
        super().__init__(f"Node {node} out of range [0, {n_nodes})")


# ---------------------------------------------------------------------------
# Solver errors
# ---------------------------------------------------------------------------


class SolverError(CausalCertError):
    """Error raised by the robustness-radius solver."""


class InfeasibleError(SolverError):
    """The ILP or LP relaxation is infeasible."""


class TimeLimitError(SolverError):
    """Solver hit the time limit before proving optimality."""


# ---------------------------------------------------------------------------
# Estimation errors
# ---------------------------------------------------------------------------


class EstimationError(CausalCertError):
    """Error during causal effect estimation."""


class NoValidAdjustmentSetError(EstimationError):
    """No valid adjustment set exists for the given treatment/outcome pair."""

    def __init__(self, treatment: NodeId, outcome: NodeId) -> None:
        self.treatment = treatment
        self.outcome = outcome
        super().__init__(
            f"No valid adjustment set for treatment={treatment}, outcome={outcome}"
        )


# ---------------------------------------------------------------------------
# Data errors
# ---------------------------------------------------------------------------


class DataError(CausalCertError):
    """Error related to data loading or validation."""


class SchemaError(DataError):
    """Column count or types do not match the DAG."""


class MissingValueError(DataError):
    """Dataset contains unexpected missing values."""


# ---------------------------------------------------------------------------
# CI testing errors
# ---------------------------------------------------------------------------


class CITestError(CausalCertError):
    """Error during conditional-independence testing."""


class NumericalInstabilityError(CITestError):
    """A CI test encountered a singular matrix or similar numerical issue."""
