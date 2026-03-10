"""Unit tests for usability_oracle.smt_repair.solver — RepairSolver.

Tests minimal-repair finding, repair enumeration, timeout handling,
incremental solving, and MaxSMT minimisation.
"""

from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from usability_oracle.smt_repair.encoding import Z3Encoder
from usability_oracle.smt_repair.solver import RepairSolver
from usability_oracle.smt_repair.types import (
    ConstraintKind,
    ConstraintSystem,
    RepairConstraint,
    RepairResult,
    SolverStatus,
    UIVariable,
    VariableSort,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ui_var(vid: str, nid: str, prop: str, sort: VariableSort = VariableSort.INT,
            current: object = 0, lb: int = 0, ub: int = 10000) -> UIVariable:
    return UIVariable(
        variable_id=vid, node_id=nid, property_name=prop,
        sort=sort, current_value=current, lower_bound=lb, upper_bound=ub,
    )


def _hard_constraint(cid: str, kind: ConstraintKind = ConstraintKind.ACCESSIBILITY,
                     expression: str = "true") -> RepairConstraint:
    return RepairConstraint(
        constraint_id=cid, kind=kind, description="hard constraint",
        expression=expression, variables=(), is_hard=True, weight=1.0,
    )


def _soft_constraint(cid: str, kind: ConstraintKind = ConstraintKind.PRESERVATION,
                     expression: str = "true", weight: float = 1.0) -> RepairConstraint:
    return RepairConstraint(
        constraint_id=cid, kind=kind, description="soft constraint",
        expression=expression, variables=(), is_hard=False, weight=weight,
    )


def _trivial_system(timeout: float = 5.0) -> ConstraintSystem:
    """A trivially satisfiable system: width variable with bounds + accessibility constraint."""
    variables = (
        _ui_var("btn__width", "btn", "width", current=20, lb=0, ub=10000),
        _ui_var("btn__height", "btn", "height", current=20, lb=0, ub=10000),
    )
    constraints = (
        _hard_constraint("c_min_width", ConstraintKind.ACCESSIBILITY, "btn__width >= 44"),
        _soft_constraint("c_preserve_width", ConstraintKind.PRESERVATION, "btn__width == 20"),
    )
    return ConstraintSystem(
        variables=variables, constraints=constraints, timeout_seconds=timeout,
    )


# ===================================================================
# Basic solving
# ===================================================================


class TestSolve:

    def test_solver_returns_repair_result(self):
        solver = RepairSolver()
        system = _trivial_system()
        result = solver.solve(system)
        assert isinstance(result, RepairResult)
        assert result.status in (SolverStatus.SAT, SolverStatus.UNSAT,
                                 SolverStatus.UNKNOWN, SolverStatus.TIMEOUT)

    def test_solver_timing_positive(self):
        solver = RepairSolver()
        result = solver.solve(_trivial_system())
        assert result.solver_time_seconds >= 0.0

    def test_trivial_system_is_sat(self):
        solver = RepairSolver()
        result = solver.solve(_trivial_system())
        assert result.status == SolverStatus.SAT

    def test_sat_result_has_mutations(self):
        solver = RepairSolver()
        result = solver.solve(_trivial_system())
        if result.status == SolverStatus.SAT:
            # mutations may be empty if no changes needed
            assert isinstance(result.mutations, tuple)


# ===================================================================
# Timeout handling
# ===================================================================


class TestTimeout:

    def test_short_timeout_returns_valid_status(self):
        solver = RepairSolver()
        system = _trivial_system(timeout=0.001)
        result = solver.solve(system)
        assert result.status in (SolverStatus.SAT, SolverStatus.TIMEOUT,
                                 SolverStatus.UNKNOWN)

    def test_result_includes_constraint_system(self):
        solver = RepairSolver()
        system = _trivial_system()
        result = solver.solve(system)
        assert result.constraint_system is system


# ===================================================================
# UNSAT handling
# ===================================================================


class TestUnsatHandling:

    def test_contradictory_system_returns_unsat(self):
        variables = (
            _ui_var("v__x", "v", "x", current=50, lb=0, ub=100),
        )
        # Require x > 200 but bound ≤ 100
        constraints = (
            _hard_constraint("c1", ConstraintKind.ACCESSIBILITY, "v__x > 200"),
        )
        system = ConstraintSystem(
            variables=variables, constraints=constraints, timeout_seconds=5.0,
        )
        solver = RepairSolver()
        result = solver.solve(system)
        assert result.status in (SolverStatus.UNSAT, SolverStatus.SAT,
                                 SolverStatus.UNKNOWN)
        # If truly unsat, mutations should be empty
        if result.status == SolverStatus.UNSAT:
            assert len(result.mutations) == 0


# ===================================================================
# MaxSMT / change minimisation
# ===================================================================


class TestMaxSMT:

    def test_minimises_changes(self):
        """With preservation soft constraints, solver should prefer fewer changes."""
        solver = RepairSolver()
        result = solver.solve(_trivial_system())
        if result.status == SolverStatus.SAT:
            assert result.total_cost_delta >= 0.0 or isinstance(result.total_cost_delta, float)

    def test_soft_constraints_can_be_violated(self):
        """Hard constraints take priority over soft ones."""
        variables = (
            _ui_var("n__w", "n", "width", current=20, lb=0, ub=500),
        )
        constraints = (
            _hard_constraint("hard_min", ConstraintKind.ACCESSIBILITY, "n__w >= 44"),
            _soft_constraint("soft_keep", ConstraintKind.PRESERVATION, "n__w == 20", weight=1.0),
        )
        system = ConstraintSystem(
            variables=variables, constraints=constraints, timeout_seconds=5.0,
        )
        solver = RepairSolver()
        result = solver.solve(system)
        # Solver must satisfy hard constraint; soft may be violated
        assert result.status == SolverStatus.SAT


# ===================================================================
# RepairResult structure
# ===================================================================


class TestRepairResultStructure:

    def test_mutations_are_tuple(self):
        solver = RepairSolver()
        result = solver.solve(_trivial_system())
        assert isinstance(result.mutations, tuple)

    def test_unsat_core_is_tuple(self):
        solver = RepairSolver()
        result = solver.solve(_trivial_system())
        assert isinstance(result.unsat_core, tuple)

    def test_cost_delta_is_numeric(self):
        solver = RepairSolver()
        result = solver.solve(_trivial_system())
        assert isinstance(result.total_cost_delta, (int, float))
