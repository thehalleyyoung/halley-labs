"""
Custom exception hierarchy for DP-Forge.

All exceptions inherit from :class:`DPForgeError`, enabling callers to catch
the base class for broad error handling or specific subclasses for targeted
recovery.  Each exception carries structured context (e.g., which LP was
infeasible, which pair violated DP) so that callers can programmatically
inspect failures without parsing error messages.

Exception Hierarchy::

    DPForgeError
    ├── InfeasibleSpecError      — LP / SDP proved infeasible
    ├── VerificationError        — DP verification failed post-extraction
    ├── NumericalInstabilityError — condition number exceeded threshold
    ├── ConvergenceError         — CEGIS loop did not converge
    ├── SensitivityError         — sensitivity computation failed
    ├── CycleDetectedError       — CEGIS revisited a counterexample pair
    ├── SolverError              — underlying LP / SDP solver error
    ├── InvalidMechanismError    — mechanism validation failed
    ├── BudgetExhaustedError     — privacy budget exceeded
    └── ConfigurationError       — invalid configuration parameters
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple


class DPForgeError(Exception):
    """Base exception for all DP-Forge errors.

    Attributes:
        message: Human-readable error description.
        context: Optional dict of structured context for programmatic inspection.
    """

    def __init__(self, message: str, *, context: Optional[dict[str, Any]] = None) -> None:
        self.message = message
        self.context = context or {}
        super().__init__(message)

    def __repr__(self) -> str:
        ctx = f", context={self.context}" if self.context else ""
        return f"{self.__class__.__name__}({self.message!r}{ctx})"


class InfeasibleSpecError(DPForgeError):
    """Raised when the LP or SDP formulation is provably infeasible.

    This typically means the privacy parameters (ε, δ) are too tight for the
    given query sensitivity and discretization, or the adjacency structure
    admits no valid mechanism.

    Attributes:
        solver_status: Status string returned by the solver.
        epsilon: Privacy parameter ε used in the formulation.
        delta: Privacy parameter δ used in the formulation.
        n_vars: Number of decision variables in the program.
        n_constraints: Number of constraints in the program.
    """

    def __init__(
        self,
        message: str,
        *,
        solver_status: Optional[str] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        n_vars: Optional[int] = None,
        n_constraints: Optional[int] = None,
    ) -> None:
        context = {
            "solver_status": solver_status,
            "epsilon": epsilon,
            "delta": delta,
            "n_vars": n_vars,
            "n_constraints": n_constraints,
        }
        super().__init__(message, context={k: v for k, v in context.items() if v is not None})
        self.solver_status = solver_status
        self.epsilon = epsilon
        self.delta = delta
        self.n_vars = n_vars
        self.n_constraints = n_constraints


class VerificationError(DPForgeError):
    """Raised when a mechanism fails post-extraction DP verification.

    The verifier found a pair (i, i') and output bin j where the privacy
    constraint is violated beyond tolerance.

    Attributes:
        violation: Tuple (i, i_prime, j_worst, magnitude) of the worst violation.
        epsilon: Target privacy parameter ε.
        delta: Target privacy parameter δ.
        tolerance: Verification tolerance used.
    """

    def __init__(
        self,
        message: str,
        *,
        violation: Optional[Tuple[int, int, int, float]] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        tolerance: Optional[float] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if violation is not None:
            context["violation_i"] = violation[0]
            context["violation_i_prime"] = violation[1]
            context["violation_j_worst"] = violation[2]
            context["violation_magnitude"] = violation[3]
        if epsilon is not None:
            context["epsilon"] = epsilon
        if delta is not None:
            context["delta"] = delta
        if tolerance is not None:
            context["tolerance"] = tolerance
        super().__init__(message, context=context)
        self.violation = violation
        self.epsilon = epsilon
        self.delta = delta
        self.tolerance = tolerance


class NumericalInstabilityError(DPForgeError):
    """Raised when a matrix condition number exceeds the configured threshold.

    High condition numbers indicate that the LP/SDP solution may be
    numerically unreliable, and the extracted mechanism could violate DP
    despite the solver reporting feasibility.

    Attributes:
        condition_number: Observed condition number.
        max_condition_number: Configured maximum threshold.
        matrix_name: Identifier for the matrix that triggered the error.
    """

    def __init__(
        self,
        message: str,
        *,
        condition_number: Optional[float] = None,
        max_condition_number: Optional[float] = None,
        matrix_name: Optional[str] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if condition_number is not None:
            context["condition_number"] = condition_number
        if max_condition_number is not None:
            context["max_condition_number"] = max_condition_number
        if matrix_name is not None:
            context["matrix_name"] = matrix_name
        super().__init__(message, context=context)
        self.condition_number = condition_number
        self.max_condition_number = max_condition_number
        self.matrix_name = matrix_name


class ConvergenceError(DPForgeError):
    """Raised when the CEGIS loop fails to converge within max_iter iterations.

    Attributes:
        iterations: Number of iterations completed.
        max_iter: Maximum allowed iterations.
        final_obj: Objective value at the last iteration.
        convergence_history: List of objective values per iteration.
    """

    def __init__(
        self,
        message: str,
        *,
        iterations: Optional[int] = None,
        max_iter: Optional[int] = None,
        final_obj: Optional[float] = None,
        convergence_history: Optional[Sequence[float]] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if iterations is not None:
            context["iterations"] = iterations
        if max_iter is not None:
            context["max_iter"] = max_iter
        if final_obj is not None:
            context["final_obj"] = final_obj
        if convergence_history is not None:
            context["history_length"] = len(convergence_history)
        super().__init__(message, context=context)
        self.iterations = iterations
        self.max_iter = max_iter
        self.final_obj = final_obj
        self.convergence_history = list(convergence_history) if convergence_history else []


class SensitivityError(DPForgeError):
    """Raised when sensitivity computation fails.

    Common causes: unbounded query domain, non-finite query outputs, or
    adjacency graph yielding infinite sensitivity.

    Attributes:
        query_type: Type of query that failed.
        sensitivity_norm: Which norm (L1, L2, Linf) was being computed.
        domain_size: Size of the query domain, if known.
    """

    def __init__(
        self,
        message: str,
        *,
        query_type: Optional[str] = None,
        sensitivity_norm: Optional[str] = None,
        domain_size: Optional[int] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if query_type is not None:
            context["query_type"] = query_type
        if sensitivity_norm is not None:
            context["sensitivity_norm"] = sensitivity_norm
        if domain_size is not None:
            context["domain_size"] = domain_size
        super().__init__(message, context=context)
        self.query_type = query_type
        self.sensitivity_norm = sensitivity_norm
        self.domain_size = domain_size


class CycleDetectedError(DPForgeError):
    """Raised when CEGIS revisits a counterexample pair already in the constraint set.

    Per the approach spec, if a pair is revisited the correct action is to call
    ExtractMechanism with DP-preserving projection, NOT tighten constraints.

    Attributes:
        pair: The (i, i') pair that was revisited.
        iteration: CEGIS iteration where the cycle was detected.
        constraint_set_size: Number of pairs in the current constraint set.
    """

    def __init__(
        self,
        message: str,
        *,
        pair: Optional[Tuple[int, int]] = None,
        iteration: Optional[int] = None,
        constraint_set_size: Optional[int] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if pair is not None:
            context["pair"] = pair
        if iteration is not None:
            context["iteration"] = iteration
        if constraint_set_size is not None:
            context["constraint_set_size"] = constraint_set_size
        super().__init__(message, context=context)
        self.pair = pair
        self.iteration = iteration
        self.constraint_set_size = constraint_set_size


class SolverError(DPForgeError):
    """Raised when the underlying LP/SDP solver encounters an error.

    Wraps solver-specific exceptions with DP-Forge context.

    Attributes:
        solver_name: Name of the solver (HiGHS, GLPK, MOSEK, SCS).
        solver_status: Status string from the solver.
        original_error: The original exception from the solver, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        solver_name: Optional[str] = None,
        solver_status: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if solver_name is not None:
            context["solver_name"] = solver_name
        if solver_status is not None:
            context["solver_status"] = solver_status
        if original_error is not None:
            context["original_error_type"] = type(original_error).__name__
        super().__init__(message, context=context)
        self.solver_name = solver_name
        self.solver_status = solver_status
        self.original_error = original_error


class InvalidMechanismError(DPForgeError):
    """Raised when a mechanism fails structural validation.

    Examples: rows not summing to 1, negative probabilities, wrong shape.

    Attributes:
        reason: Specific validation failure reason.
        expected_shape: Expected mechanism table shape.
        actual_shape: Actual mechanism table shape.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: Optional[str] = None,
        expected_shape: Optional[Tuple[int, int]] = None,
        actual_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if reason is not None:
            context["reason"] = reason
        if expected_shape is not None:
            context["expected_shape"] = expected_shape
        if actual_shape is not None:
            context["actual_shape"] = actual_shape
        super().__init__(message, context=context)
        self.reason = reason
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape


class BudgetExhaustedError(DPForgeError):
    """Raised when the cumulative privacy budget has been exceeded.

    Relevant when composing multiple mechanism invocations under a shared
    privacy budget.

    Attributes:
        budget_epsilon: Total ε budget.
        budget_delta: Total δ budget.
        consumed_epsilon: ε consumed so far.
        consumed_delta: δ consumed so far.
    """

    def __init__(
        self,
        message: str,
        *,
        budget_epsilon: Optional[float] = None,
        budget_delta: Optional[float] = None,
        consumed_epsilon: Optional[float] = None,
        consumed_delta: Optional[float] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if budget_epsilon is not None:
            context["budget_epsilon"] = budget_epsilon
        if budget_delta is not None:
            context["budget_delta"] = budget_delta
        if consumed_epsilon is not None:
            context["consumed_epsilon"] = consumed_epsilon
        if consumed_delta is not None:
            context["consumed_delta"] = consumed_delta
        super().__init__(message, context=context)
        self.budget_epsilon = budget_epsilon
        self.budget_delta = budget_delta
        self.consumed_epsilon = consumed_epsilon
        self.consumed_delta = consumed_delta


class ConfigurationError(DPForgeError):
    """Raised when configuration parameters are invalid.

    Attributes:
        parameter: Name of the invalid parameter.
        value: The invalid value provided.
        constraint: Description of the constraint that was violated.
    """

    def __init__(
        self,
        message: str,
        *,
        parameter: Optional[str] = None,
        value: Any = None,
        constraint: Optional[str] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if parameter is not None:
            context["parameter"] = parameter
        if value is not None:
            context["value"] = repr(value)
        if constraint is not None:
            context["constraint"] = constraint
        super().__init__(message, context=context)
        self.parameter = parameter
        self.value = value
        self.constraint = constraint
