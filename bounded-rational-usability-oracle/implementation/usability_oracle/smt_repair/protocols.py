"""
usability_oracle.smt_repair.protocols — Structural interfaces for
SMT-backed repair synthesis.

Defines protocols for translating usability bottlenecks into SMT
constraints, solving the resulting optimisation problem, and validating
proposed mutations.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usability_oracle.smt_repair.types import (
        ConstraintSystem,
        MutationCandidate,
        RepairConstraint,
        RepairResult,
        UIVariable,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ConstraintGenerator
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ConstraintGenerator(Protocol):
    """Generate SMT repair constraints from usability analysis results.

    Translates high-level bottleneck information (e.g. "this button is
    too small for comfortable pointing") into formal SMT constraints
    over UI property variables.
    """

    def generate_variables(
        self,
        tree_dict: Dict[str, Any],
        mutable_properties: Sequence[str],
    ) -> Sequence[UIVariable]:
        """Extract mutable UI variables from an accessibility tree.

        Parameters:
            tree_dict: Serialised accessibility tree.
            mutable_properties: Names of properties that may be modified
                (e.g. ``["width", "height", "label", "role"]``).

        Returns:
            Sequence of :class:`UIVariable` instances.
        """
        ...

    def generate_constraints(
        self,
        variables: Sequence[UIVariable],
        bottleneck_report: Dict[str, Any],
    ) -> Sequence[RepairConstraint]:
        """Generate repair constraints from a bottleneck report.

        Parameters:
            variables: Available UI variables.
            bottleneck_report: Serialised bottleneck classification output.

        Returns:
            Hard and soft constraints addressing the identified bottlenecks.
        """
        ...

    def build_system(
        self,
        variables: Sequence[UIVariable],
        constraints: Sequence[RepairConstraint],
        objective_expression: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> ConstraintSystem:
        """Assemble a complete constraint system.

        Parameters:
            variables: All UI variables.
            constraints: All repair constraints.
            objective_expression: Optional minimisation objective
                in SMT-LIB syntax.
            timeout_seconds: Solver timeout.

        Returns:
            A ready-to-solve :class:`ConstraintSystem`.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# RepairSolver
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class RepairSolver(Protocol):
    """Solve an SMT constraint system to produce repair mutations.

    Wraps an underlying SMT solver (e.g. Z3, CVC5) and translates the
    satisfying model back into :class:`MutationCandidate` instances.
    """

    def solve(
        self,
        system: ConstraintSystem,
    ) -> RepairResult:
        """Solve the constraint system.

        Parameters:
            system: Complete constraint system to solve.

        Returns:
            A :class:`RepairResult` containing the solver status and,
            if SAT, the proposed mutations.

        Raises:
            SynthesisTimeoutError: If the solver exceeds the timeout.
        """
        ...

    def solve_incremental(
        self,
        base_system: ConstraintSystem,
        additional_constraints: Sequence[RepairConstraint],
    ) -> RepairResult:
        """Incrementally add constraints and re-solve.

        Useful for iterative refinement: first solve a relaxed problem,
        then tighten constraints based on validation feedback.

        Parameters:
            base_system: Previously solved constraint system.
            additional_constraints: New constraints to assert.

        Returns:
            Updated :class:`RepairResult`.
        """
        ...

    def extract_unsat_core(
        self,
        system: ConstraintSystem,
    ) -> Sequence[str]:
        """Extract the minimal unsatisfiable core.

        Parameters:
            system: An unsatisfiable constraint system.

        Returns:
            Sequence of constraint identifiers forming the UNSAT core.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# MutationValidator
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class MutationValidator(Protocol):
    """Validate that proposed mutations actually improve usability.

    After the SMT solver produces a candidate repair, the validator
    checks that applying the mutations to the real UI tree satisfies
    all constraints and does not introduce new regressions.
    """

    def validate(
        self,
        mutations: Sequence[MutationCandidate],
        original_tree: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate a set of mutations against the original UI tree.

        Parameters:
            mutations: Proposed mutations from the SMT solver.
            original_tree: Serialised accessibility tree before repair.

        Returns:
            Validation report with fields ``"valid"`` (bool),
            ``"violations"`` (list of constraint violations), and
            ``"estimated_improvement"`` (float, cost delta).
        """
        ...

    def apply_mutations(
        self,
        mutations: Sequence[MutationCandidate],
        tree_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply mutations to a serialised tree and return the result.

        Parameters:
            mutations: Ordered sequence of mutations to apply.
            tree_dict: Serialised accessibility tree.

        Returns:
            Updated serialised tree with mutations applied.
        """
        ...
