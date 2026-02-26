"""Unit tests for SMT solver interfaces — Z3, dReal, portfolio."""

import pytest

from bioprover.encoding.expression import (
    ZERO,
    ONE,
    Const,
    Var,
    Add,
    Mul,
    Ge,
    Le,
    Gt,
    Lt,
    And,
    Or,
    Not,
)
from bioprover.smt.solver_base import SMTResult, Model


# ===================================================================
# SMTResult and Model basics
# ===================================================================


class TestSolverBase:
    def test_smt_result_values(self):
        assert SMTResult.SAT is not None
        assert SMTResult.UNSAT is not None
        assert SMTResult.UNKNOWN is not None

    def test_model_creation(self):
        m = Model(values={"x": 1.0, "y": 2.0}, solver_name="test")
        assert m["x"] == 1.0
        assert "x" in m
        assert m.get("z", 99) == 99

    def test_model_variables(self):
        m = Model(values={"x": 1.0, "y": 2.0}, solver_name="test")
        assert set(m.variables()) == {"x", "y"}

    def test_model_get_float(self):
        m = Model(values={"x": 3.14}, solver_name="test")
        assert m.get_float("x") == pytest.approx(3.14)

    def test_model_to_dict(self):
        m = Model(values={"x": 1.0}, solver_name="test")
        d = m.to_dict()
        assert "x" in d

    def test_model_from_dict(self):
        m = Model.from_dict({"x": 1.0, "_solver": "test"})
        assert m is not None


# ===================================================================
# Z3 solver
# ===================================================================


class TestZ3Solver:
    def _make_solver(self):
        from bioprover.smt.z3_interface import Z3Solver
        return Z3Solver()

    def test_simple_sat(self):
        solver = self._make_solver()
        # x >= 0 AND x <= 10
        solver.assert_formula(Ge(Var("x"), ZERO))
        solver.assert_formula(Le(Var("x"), Const(10.0)))
        result = solver.check_sat()
        assert result == SMTResult.SAT

    def test_simple_unsat(self):
        solver = self._make_solver()
        # x > 5 AND x < 3
        solver.assert_formula(Gt(Var("x"), Const(5.0)))
        solver.assert_formula(Lt(Var("x"), Const(3.0)))
        result = solver.check_sat()
        assert result == SMTResult.UNSAT

    def test_get_model(self):
        solver = self._make_solver()
        solver.assert_formula(Ge(Var("x"), Const(1.0)))
        solver.assert_formula(Le(Var("x"), Const(2.0)))
        result = solver.check_sat()
        assert result == SMTResult.SAT
        model = solver.get_model()
        x_val = model.get_float("x")
        assert 1.0 <= x_val <= 2.0

    def test_push_pop(self):
        solver = self._make_solver()
        solver.assert_formula(Ge(Var("x"), ZERO))
        solver.push()
        solver.assert_formula(Lt(Var("x"), ZERO))  # contradicts
        result = solver.check_sat()
        assert result == SMTResult.UNSAT
        solver.pop()
        result = solver.check_sat()
        assert result == SMTResult.SAT

    def test_multiple_push_pop(self):
        solver = self._make_solver()
        solver.assert_formula(Ge(Var("x"), ZERO))
        solver.push()
        solver.assert_formula(Le(Var("x"), Const(5.0)))
        solver.push()
        solver.assert_formula(Ge(Var("x"), Const(10.0)))  # unsat
        assert solver.check_sat() == SMTResult.UNSAT
        solver.pop()
        assert solver.check_sat() == SMTResult.SAT
        solver.pop()
        assert solver.check_sat() == SMTResult.SAT

    def test_reset(self):
        solver = self._make_solver()
        solver.assert_formula(Gt(Var("x"), Const(5.0)))
        solver.assert_formula(Lt(Var("x"), Const(3.0)))
        assert solver.check_sat() == SMTResult.UNSAT
        solver.reset()
        solver.assert_formula(Ge(Var("x"), ZERO))
        assert solver.check_sat() == SMTResult.SAT

    def test_two_variables(self):
        solver = self._make_solver()
        # x + y >= 5 AND x >= 0 AND y >= 0 AND x <= 3 AND y <= 3
        solver.assert_formula(Ge(Add(Var("x"), Var("y")), Const(5.0)))
        solver.assert_formula(Ge(Var("x"), ZERO))
        solver.assert_formula(Ge(Var("y"), ZERO))
        solver.assert_formula(Le(Var("x"), Const(3.0)))
        solver.assert_formula(Le(Var("y"), Const(3.0)))
        result = solver.check_sat()
        assert result == SMTResult.SAT
        model = solver.get_model()
        x_val = model.get_float("x")
        y_val = model.get_float("y")
        assert x_val + y_val >= 5.0 - 1e-6

    def test_nonlinear(self):
        solver = self._make_solver()
        # x * x = 4 AND x > 0  →  x = 2
        from bioprover.encoding.expression import Eq
        solver.assert_formula(Eq(Mul(Var("x"), Var("x")), Const(4.0)))
        solver.assert_formula(Gt(Var("x"), ZERO))
        result = solver.check_sat()
        if result == SMTResult.SAT:
            model = solver.get_model()
            x_val = model.get_float("x")
            assert abs(x_val - 2.0) < 0.1


# ===================================================================
# Expression translator
# ===================================================================


class TestExprTranslator:
    def test_translate_const(self):
        from bioprover.smt.z3_interface import ExprTranslator
        tr = ExprTranslator()
        z3_expr = tr.translate(Const(3.14))
        assert z3_expr is not None

    def test_translate_var(self):
        from bioprover.smt.z3_interface import ExprTranslator
        tr = ExprTranslator()
        z3_expr = tr.translate(Var("x"))
        assert z3_expr is not None

    def test_translate_add(self):
        from bioprover.smt.z3_interface import ExprTranslator
        tr = ExprTranslator()
        z3_expr = tr.translate(Add(Var("x"), Const(1.0)))
        assert z3_expr is not None

    def test_translate_comparison(self):
        from bioprover.smt.z3_interface import ExprTranslator
        tr = ExprTranslator()
        z3_expr = tr.translate(Ge(Var("x"), ZERO))
        assert z3_expr is not None

    def test_real_var(self):
        from bioprover.smt.z3_interface import ExprTranslator
        tr = ExprTranslator()
        v = tr.real_var("x")
        assert v is not None
        assert tr.get_var("x") is v

    def test_clear_cache(self):
        from bioprover.smt.z3_interface import ExprTranslator
        tr = ExprTranslator()
        tr.real_var("x")
        tr.clear_cache()
        assert tr.get_var("x") is None


# ===================================================================
# Z3 optimizer
# ===================================================================


class TestZ3Optimizer:
    def test_minimize(self):
        from bioprover.smt.z3_interface import Z3Solver
        solver = Z3Solver()
        opt = solver.create_optimizer()
        opt.add(Ge(Var("x"), Const(1.0)))
        opt.add(Le(Var("x"), Const(10.0)))
        opt.minimize(Var("x"))
        result = opt.check()
        assert result == SMTResult.SAT
        model = opt.model()
        x_val = model.get_float("x")
        assert x_val == pytest.approx(1.0, abs=0.1)

    def test_maximize(self):
        from bioprover.smt.z3_interface import Z3Solver
        solver = Z3Solver()
        opt = solver.create_optimizer()
        opt.add(Ge(Var("x"), Const(1.0)))
        opt.add(Le(Var("x"), Const(10.0)))
        opt.maximize(Var("x"))
        result = opt.check()
        assert result == SMTResult.SAT
        model = opt.model()
        x_val = model.get_float("x")
        assert x_val == pytest.approx(10.0, abs=0.1)


# ===================================================================
# dReal ICP solver
# ===================================================================


class TestDRealICP:
    def test_icp_solver_sat(self):
        from bioprover.smt.dreal_interface import ICPSolver, ICPConstraint
        solver = ICPSolver(delta=0.01)
        solver.declare_variable("x", lo=0.0, hi=10.0)
        solver.add_constraint(ICPConstraint(kind="ge", lhs="x", rhs=5.0, original=None))
        result, model = solver.solve()
        assert result == SMTResult.SAT or result == SMTResult.UNKNOWN
        if model is not None:
            assert "x" in model
            assert model["x"].lo >= 5.0 - 0.1

    def test_icp_solver_unsat(self):
        from bioprover.smt.dreal_interface import ICPSolver, ICPConstraint
        solver = ICPSolver(delta=0.01)
        solver.declare_variable("x", lo=0.0, hi=3.0)
        solver.add_constraint(ICPConstraint(kind="ge", lhs="x", rhs=5.0, original=None))
        result, model = solver.solve()
        assert result == SMTResult.UNSAT

    def test_dreal_solver_declare_and_check(self):
        from bioprover.smt.dreal_interface import DRealSolver
        solver = DRealSolver(use_icp_fallback=True)
        solver.declare_variable("x", lo=0.0, hi=10.0)
        solver.assert_formula(Ge(Var("x"), Const(1.0)))
        solver.assert_formula(Le(Var("x"), Const(5.0)))
        result = solver.check_sat()
        assert result in (SMTResult.SAT, SMTResult.UNKNOWN)

    def test_dreal_push_pop(self):
        from bioprover.smt.dreal_interface import DRealSolver
        solver = DRealSolver(use_icp_fallback=True)
        solver.declare_variable("x", lo=0.0, hi=10.0)
        solver.assert_formula(Ge(Var("x"), Const(1.0)))
        solver.push()
        solver.assert_formula(Le(Var("x"), ZERO))  # contradicts
        result = solver.check_sat()
        assert result == SMTResult.UNSAT
        solver.pop()
        result = solver.check_sat()
        assert result in (SMTResult.SAT, SMTResult.UNKNOWN)

    def test_set_delta(self):
        from bioprover.smt.dreal_interface import DRealSolver
        solver = DRealSolver(use_icp_fallback=True)
        solver.set_delta(0.001)
        solver.declare_variable("x", lo=0.0, hi=10.0)
        solver.assert_formula(Ge(Var("x"), Const(1.0)))
        result = solver.check_sat()
        assert result in (SMTResult.SAT, SMTResult.UNKNOWN)


# ===================================================================
# dReal SMT-LIB generator
# ===================================================================


class TestSMTLIBGenerator:
    def test_generate(self):
        from bioprover.smt.dreal_interface import SMTLIBGenerator
        gen = SMTLIBGenerator(delta=0.01)
        gen.declare_variable("x", lo=0.0, hi=10.0)
        gen.assert_formula("(>= x 1.0)")
        output = gen.generate()
        assert "x" in output
        assert "check-sat" in output.lower()

    def test_reset(self):
        from bioprover.smt.dreal_interface import SMTLIBGenerator
        gen = SMTLIBGenerator(delta=0.01)
        gen.declare_variable("x", lo=0.0, hi=10.0)
        gen.reset()
        output = gen.generate()
        assert "x" not in output or "check-sat" in output.lower()


# ===================================================================
# dReal Interval
# ===================================================================


class TestDRealInterval:
    def test_point(self):
        from bioprover.smt.dreal_interface import Interval
        iv = Interval.point(3.0)
        assert iv.lo == 3.0 and iv.hi == 3.0

    def test_entire(self):
        from bioprover.smt.dreal_interface import Interval
        iv = Interval.entire()
        assert iv.lo < 0 and iv.hi > 0

    def test_contains(self):
        from bioprover.smt.dreal_interface import Interval
        iv = Interval(lo=1.0, hi=5.0)
        assert iv.contains(3.0)
        assert not iv.contains(6.0)

    def test_intersect(self):
        from bioprover.smt.dreal_interface import Interval
        a = Interval(lo=0.0, hi=5.0)
        b = Interval(lo=3.0, hi=8.0)
        c = a.intersect(b)
        assert c is not None
        assert c.lo == pytest.approx(3.0)
        assert c.hi == pytest.approx(5.0)


# ===================================================================
# Portfolio solver
# ===================================================================


class TestPortfolioSolver:
    def test_simple_sat(self):
        from bioprover.smt.portfolio import PortfolioSolver
        solver = PortfolioSolver(max_workers=1, total_timeout=30.0)
        solver.assert_formula(Ge(Var("x"), ZERO))
        solver.assert_formula(Le(Var("x"), Const(10.0)))
        result = solver.check_sat()
        assert result == SMTResult.SAT

    def test_simple_unsat(self):
        from bioprover.smt.portfolio import PortfolioSolver
        solver = PortfolioSolver(max_workers=1, total_timeout=30.0)
        solver.assert_formula(Gt(Var("x"), Const(5.0)))
        solver.assert_formula(Lt(Var("x"), Const(3.0)))
        result = solver.check_sat()
        assert result == SMTResult.UNSAT

    def test_get_model(self):
        from bioprover.smt.portfolio import PortfolioSolver
        solver = PortfolioSolver(max_workers=1, total_timeout=30.0)
        solver.assert_formula(Ge(Var("x"), Const(1.0)))
        solver.assert_formula(Le(Var("x"), Const(2.0)))
        result = solver.check_sat()
        assert result == SMTResult.SAT
        model = solver.get_model()
        assert model is not None

    def test_push_pop(self):
        from bioprover.smt.portfolio import PortfolioSolver
        solver = PortfolioSolver(max_workers=1, total_timeout=30.0)
        solver.assert_formula(Ge(Var("x"), ZERO))
        solver.push()
        solver.assert_formula(Lt(Var("x"), ZERO))
        assert solver.check_sat() == SMTResult.UNSAT
        solver.pop()
        assert solver.check_sat() == SMTResult.SAT

    def test_reset(self):
        from bioprover.smt.portfolio import PortfolioSolver
        solver = PortfolioSolver(max_workers=1, total_timeout=30.0)
        solver.assert_formula(Gt(Var("x"), Const(5.0)))
        solver.assert_formula(Lt(Var("x"), Const(3.0)))
        assert solver.check_sat() == SMTResult.UNSAT
        solver.reset()
        solver.assert_formula(Ge(Var("x"), ZERO))
        assert solver.check_sat() == SMTResult.SAT

    def test_strategy_statistics(self):
        from bioprover.smt.portfolio import PortfolioSolver
        solver = PortfolioSolver(max_workers=1, total_timeout=30.0)
        solver.assert_formula(Ge(Var("x"), ZERO))
        solver.check_sat()
        stats = solver.strategy_statistics()
        assert isinstance(stats, dict)
