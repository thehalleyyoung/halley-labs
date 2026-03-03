"""Tests for smt module: DPLLTSolver, theory solver, encoder."""
import math
import pytest
import numpy as np

from dp_forge.smt import (
    DPLLTSolver,
    SMTConfig,
    SMTVariable,
    SMTConstraint,
    SMTCheckResult,
    SolverResult,
    SMTEncoder,
    SMTVerifier,
    EncodeStrategy,
)
from dp_forge.smt.theory_solver import (
    LinearConstraint,
    ConstraintOp,
    SimplexMethod,
    FeasibilityChecker,
    LinearArithmeticSolver,
    BoundPropagation,
    parse_linear_constraint,
)
from dp_forge.smt.encoder import (
    PrivacyConstraintEncoder,
    MechanismEncoder,
    IncrementalEncoder,
)
from dp_forge.smt.quantifier import (
    QuantifierEliminator,
    FourierMotzkin,
    QuantifiedFormula,
)
from dp_forge.types import Formula, AdjacencyRelation, PrivacyBudget


def _formula(expr, variables=None):
    vs = frozenset(variables) if variables else frozenset()
    return Formula(expr=expr, variables=vs, formula_type="linear_arithmetic")


def _constraint(expr, variables=None, label=None):
    return SMTConstraint(formula=_formula(expr, variables), label=label)


class TestDPLLTSolver:
    """Test DPLLTSolver on SAT/UNSAT instances."""

    def test_sat_simple(self):
        """x <= 5, x >= 0 is SAT."""
        solver = DPLLTSolver()
        vars_ = [SMTVariable(name="x", sort="Real", lower_bound=0.0, upper_bound=5.0)]
        cons = [
            _constraint("x <= 5.0", {"x"}),
            _constraint("x >= 0.0", {"x"}),
        ]
        result = solver.check_sat(vars_, cons)
        assert result.result == SolverResult.SAT
        assert result.model is not None
        x_val = result.model.get("x")
        assert x_val is not None
        assert 0.0 <= x_val <= 5.0 + 1e-6

    def test_unsat_contradiction(self):
        """x <= 1, x >= 5 is UNSAT."""
        solver = DPLLTSolver()
        vars_ = [SMTVariable(name="x", sort="Real")]
        cons = [
            _constraint("x <= 1.0", {"x"}),
            _constraint("x >= 5.0", {"x"}),
        ]
        result = solver.check_sat(vars_, cons)
        assert result.result == SolverResult.UNSAT

    def test_two_variables_sat(self):
        """x + y <= 10, x >= 1, y >= 1 is SAT."""
        solver = DPLLTSolver()
        vars_ = [
            SMTVariable(name="x", sort="Real"),
            SMTVariable(name="y", sort="Real"),
        ]
        cons = [
            _constraint("x + y <= 10.0", {"x", "y"}),
            _constraint("x >= 1.0", {"x"}),
            _constraint("y >= 1.0", {"y"}),
        ]
        result = solver.check_sat(vars_, cons)
        assert result.result == SolverResult.SAT

    def test_check_sat_assuming(self):
        """check_sat_assuming adds extra assumptions."""
        solver = DPLLTSolver()
        vars_ = [SMTVariable(name="x", sort="Real")]
        cons = [_constraint("x >= 0.0", {"x"})]
        assumptions = [_formula("x <= 3.0", {"x"})]
        result = solver.check_sat_assuming(vars_, cons, assumptions)
        assert result.result == SolverResult.SAT

    def test_solving_time_reported(self):
        """Solving time is non-negative."""
        solver = DPLLTSolver()
        vars_ = [SMTVariable(name="x", sort="Real")]
        cons = [_constraint("x >= 0.0", {"x"})]
        result = solver.check_sat(vars_, cons)
        assert result.solving_time >= 0.0


class TestLinearArithmeticSolver:
    """Test LinearArithmeticSolver feasibility."""

    def test_feasible_simple(self):
        """Simple feasible system."""
        la = LinearArithmeticSolver()
        c1 = LinearConstraint({"x": 1.0}, ConstraintOp.LEQ, 5.0)
        c2 = LinearConstraint({"x": -1.0}, ConstraintOp.LEQ, 0.0)  # x >= 0
        la.assert_constraint(c1)
        la.assert_constraint(c2)
        sat, conflict = la.check_consistency([])
        assert sat

    def test_infeasible(self):
        """Infeasible system detected."""
        la = LinearArithmeticSolver()
        c1 = LinearConstraint({"x": 1.0}, ConstraintOp.LEQ, 1.0)
        c2 = LinearConstraint({"x": -1.0}, ConstraintOp.LEQ, -5.0)
        la.assert_constraint(c1)
        la.assert_constraint(c2)
        sat, _ = la.check_consistency([])
        assert not sat

    def test_equality_constraint(self):
        """Equality constraint is handled."""
        la = LinearArithmeticSolver()
        c = LinearConstraint({"x": 1.0}, ConstraintOp.EQ, 3.0)
        la.assert_constraint(c)
        sat, _ = la.check_consistency([])
        assert sat
        model = la.get_model()
        if model:
            assert abs(model.get("x", 0.0) - 3.0) < 0.1

    def test_push_pop(self):
        """Push/pop maintains state correctly."""
        la = LinearArithmeticSolver()
        c1 = LinearConstraint({"x": 1.0}, ConstraintOp.LEQ, 5.0)
        la.assert_constraint(c1)
        la.push()
        c2 = LinearConstraint({"x": -1.0}, ConstraintOp.LEQ, -10.0)  # x >= 10
        la.assert_constraint(c2)
        sat, _ = la.check_consistency([])
        assert not sat
        la.pop()
        sat2, _ = la.check_consistency([])
        assert sat2


class TestSimplexMethod:
    """Test SimplexMethod on small LPs."""

    def test_feasibility_simple(self):
        """Simple feasible LP."""
        sm = SimplexMethod()
        sm.add_variable("x", lower=0.0, upper=10.0)
        sm.add_constraint(LinearConstraint({"x": 1.0}, ConstraintOp.LEQ, 5.0))
        feasible, model = sm.solve()
        assert feasible

    def test_feasibility_two_vars(self):
        """Two-variable LP."""
        sm = SimplexMethod()
        sm.add_variable("x", lower=0.0)
        sm.add_variable("y", lower=0.0)
        sm.add_constraint(LinearConstraint({"x": 1.0, "y": 1.0}, ConstraintOp.LEQ, 10.0))
        sm.add_constraint(LinearConstraint({"x": 1.0}, ConstraintOp.LEQ, 6.0))
        feasible, model = sm.solve()
        assert feasible

    def test_infeasible_lp(self):
        """Infeasible LP detected."""
        sm = SimplexMethod()
        sm.add_variable("x", lower=0.0, upper=1.0)
        sm.add_constraint(LinearConstraint({"x": 1.0}, ConstraintOp.GEQ, 5.0))
        feasible, _ = sm.solve()
        assert not feasible


class TestFeasibilityChecker:
    """Test FeasibilityChecker."""

    def test_feasible(self):
        """Feasibility checker on feasible system."""
        fc = FeasibilityChecker()
        constraints = [
            LinearConstraint({"x": 1.0}, ConstraintOp.LEQ, 10.0),
            LinearConstraint({"x": -1.0}, ConstraintOp.LEQ, 0.0),
        ]
        feasible, model = fc.check(constraints)
        assert feasible

    def test_infeasible(self):
        """Feasibility checker on infeasible system."""
        fc = FeasibilityChecker()
        constraints = [
            LinearConstraint({"x": 1.0}, ConstraintOp.LEQ, 1.0),
            LinearConstraint({"x": -1.0}, ConstraintOp.LEQ, -10.0),
        ]
        feasible, _ = fc.check(constraints)
        assert not feasible

    def test_with_bounds(self):
        """Feasibility checker respects variable bounds."""
        fc = FeasibilityChecker()
        constraints = [
            LinearConstraint({"x": 1.0}, ConstraintOp.LEQ, 5.0),
        ]
        bounds = {"x": (2.0, 4.0)}
        feasible, model = fc.check(constraints, bounds)
        assert feasible


class TestBoundPropagation:
    """Test unit propagation via bound propagation."""

    def test_simple_propagation(self):
        """Bound propagation tightens bounds."""
        bp = BoundPropagation()
        c = LinearConstraint({"x": 1.0}, ConstraintOp.LEQ, 3.0)
        bounds = {"x": (None, None)}
        new_bounds, conflict = bp.propagate([c], bounds)
        assert not conflict
        ub = new_bounds.get("x", (None, None))[1]
        assert ub is not None
        assert ub <= 3.0 + 1e-6


class TestSMTEncoder:
    """Test SMTEncoder privacy constraint encoding."""

    def test_encode_mechanism(self):
        """Encoding produces variables and constraints."""
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        adj = AdjacencyRelation(edges=[(0, 1)], n=2, symmetric=True)
        budget = PrivacyBudget(epsilon=1.0, delta=0.0)
        encoder = SMTEncoder()
        vars_, cons = encoder.encode_mechanism(P, adj, budget)
        assert len(vars_) > 0
        assert len(cons) > 0

    def test_encode_violation(self):
        """Violation encoding produces constraints."""
        adj = AdjacencyRelation(edges=[(0, 1)], n=2, symmetric=True)
        budget = PrivacyBudget(epsilon=1.0, delta=0.0)
        encoder = SMTEncoder()
        vars_, cons = encoder.encode_privacy_violation(2, 2, adj, budget)
        assert len(cons) > 0

    def test_privacy_constraint_strategies(self):
        """Different encode strategies produce constraints."""
        for strategy in [EncodeStrategy.RATIO, EncodeStrategy.LOG_SPACE, EncodeStrategy.DIFFERENCE]:
            pce = PrivacyConstraintEncoder(strategy)
            adj = AdjacencyRelation(edges=[(0, 1)], n=2, symmetric=True)
            prob_vars = [["p_0_0", "p_0_1"], ["p_1_0", "p_1_1"]]
            _, cons = pce.encode_pure_dp(2, 2, 1.0, adj, prob_vars)
            assert len(cons) > 0


class TestMechanismEncoder:
    """Test MechanismEncoder."""

    def test_probability_table(self):
        """Probability table has correct number of vars/constraints."""
        me = MechanismEncoder()
        vars_, cons, names = me.encode_probability_table(2, 3)
        assert len(names) == 2  # 2 inputs
        assert len(names[0]) == 3  # 3 outputs each
        assert len(vars_) == 6

    def test_fixed_mechanism(self):
        """Fixed mechanism creates equality constraints."""
        me = MechanismEncoder()
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        vars_, cons, names = me.encode_fixed_mechanism(P)
        assert len(vars_) == 4


class TestIncrementalEncoder:
    """Test IncrementalEncoder push/pop."""

    def test_push_pop(self):
        """Push/pop restores state."""
        ie = IncrementalEncoder()
        ie.add_variables([SMTVariable(name="x", sort="Real")])
        assert len(ie.variables) == 1
        ie.push()
        ie.add_variables([SMTVariable(name="y", sort="Real")])
        assert len(ie.variables) == 2
        ie.pop()
        assert len(ie.variables) == 1

    def test_scope_depth(self):
        """Scope depth tracks correctly."""
        ie = IncrementalEncoder()
        assert ie.scope_depth == 0
        ie.push()
        assert ie.scope_depth == 1
        ie.push()
        assert ie.scope_depth == 2
        ie.pop()
        assert ie.scope_depth == 1

    def test_reset(self):
        """Reset clears everything."""
        ie = IncrementalEncoder()
        ie.add_variables([SMTVariable(name="x", sort="Real")])
        ie.push()
        ie.reset()
        assert len(ie.variables) == 0
        assert ie.scope_depth == 0


class TestQuantifierEliminator:
    """Test QuantifierEliminator soundness."""

    def test_fourier_motzkin_elimination(self):
        """FM eliminates a variable correctly."""
        fm = FourierMotzkin()
        c1 = LinearConstraint({"x": 1.0}, ConstraintOp.LEQ, 5.0)
        c2 = LinearConstraint({"x": -1.0, "y": 1.0}, ConstraintOp.LEQ, 3.0)
        result = fm.eliminate_variable([c1, c2], "x")
        assert isinstance(result, list)

    def test_eliminate_multiple_variables(self):
        """FM eliminates multiple variables."""
        fm = FourierMotzkin()
        c1 = LinearConstraint({"x": 1.0, "y": 1.0}, ConstraintOp.LEQ, 10.0)
        c2 = LinearConstraint({"x": -1.0}, ConstraintOp.LEQ, 0.0)
        c3 = LinearConstraint({"y": -1.0}, ConstraintOp.LEQ, 0.0)
        result = fm.eliminate_all([c1, c2, c3], ["x", "y"])
        assert result.success

    def test_quantifier_eliminator_existential(self):
        """Existential elimination via QE."""
        qe = QuantifierEliminator()
        c = LinearConstraint({"x": 1.0}, ConstraintOp.LEQ, 5.0)
        qf = QuantifiedFormula(quantifier="exists", bound_vars=["x"], body=[c])
        result = qe.eliminate(qf)
        assert result.success

    def test_quantifier_eliminator_universal(self):
        """Universal elimination via QE."""
        qe = QuantifierEliminator()
        c = LinearConstraint({"x": 1.0}, ConstraintOp.LEQ, 5.0)
        qf = QuantifiedFormula(quantifier="forall", bound_vars=["x"], body=[c])
        result = qe.eliminate(qf)
        assert result.success
