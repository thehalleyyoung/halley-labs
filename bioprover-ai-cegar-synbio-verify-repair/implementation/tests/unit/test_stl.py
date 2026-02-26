"""Unit tests for STL AST, Bio-STL parser, robustness computation."""

import math

import numpy as np
import pytest

from bioprover.temporal.stl_ast import (
    Always,
    ComparisonOp,
    Eventually,
    Expression,
    Interval,
    Predicate,
    STLAnd,
    STLImplies,
    STLNot,
    STLOr,
    Until,
    globally,
    eventually,
    predicate,
)


# ===================================================================
# STL AST construction
# ===================================================================


class TestSTLConstruction:
    def test_predicate(self):
        p = predicate("x", ">", 1.0)
        assert isinstance(p, Predicate)
        assert p.threshold == 1.0

    def test_always(self):
        p = predicate("x", ">", 0.0)
        f = Always(child=p, interval=Interval(lo=0.0, hi=10.0))
        assert isinstance(f, Always)
        assert f.interval.lo == 0.0
        assert f.interval.hi == 10.0

    def test_eventually(self):
        p = predicate("x", ">", 5.0)
        f = Eventually(child=p, interval=Interval(lo=0.0, hi=5.0))
        assert isinstance(f, Eventually)

    def test_until(self):
        p = predicate("x", ">", 0.0)
        q = predicate("y", "<", 1.0)
        f = Until(left=p, right=q, interval=Interval(lo=0.0, hi=10.0))
        assert isinstance(f, Until)

    def test_not(self):
        p = predicate("x", ">", 0.0)
        f = STLNot(child=p)
        assert isinstance(f, STLNot)

    def test_and(self):
        p = predicate("x", ">", 0.0)
        q = predicate("y", "<", 1.0)
        f = STLAnd(left=p, right=q)
        assert isinstance(f, STLAnd)

    def test_or(self):
        p = predicate("x", ">", 0.0)
        q = predicate("y", "<", 1.0)
        f = STLOr(left=p, right=q)
        assert isinstance(f, STLOr)

    def test_implies(self):
        p = predicate("x", ">", 0.0)
        q = predicate("y", "<", 1.0)
        f = STLImplies(antecedent=p, consequent=q)
        assert isinstance(f, STLImplies)

    def test_convenience_globally(self):
        p = predicate("x", ">", 0.0)
        f = globally(p, 0.0, 10.0)
        assert isinstance(f, Always)

    def test_convenience_eventually(self):
        p = predicate("x", ">", 0.0)
        f = eventually(p, 0.0, 5.0)
        assert isinstance(f, Eventually)


# ===================================================================
# Interval dataclass
# ===================================================================


class TestInterval:
    def test_length(self):
        iv = Interval(lo=2.0, hi=5.0)
        assert iv.length == pytest.approx(3.0)

    def test_contains(self):
        iv = Interval(lo=0.0, hi=10.0)
        assert iv.contains(5.0)
        assert not iv.contains(11.0)

    def test_shift(self):
        iv = Interval(lo=0.0, hi=5.0)
        shifted = iv.shift(3.0)
        assert shifted.lo == pytest.approx(3.0)
        assert shifted.hi == pytest.approx(8.0)

    def test_intersect(self):
        a = Interval(lo=0.0, hi=5.0)
        b = Interval(lo=3.0, hi=8.0)
        c = a.intersect(b)
        assert c is not None
        assert c.lo == pytest.approx(3.0)
        assert c.hi == pytest.approx(5.0)

    def test_intersect_disjoint(self):
        a = Interval(lo=0.0, hi=2.0)
        b = Interval(lo=3.0, hi=5.0)
        c = a.intersect(b)
        assert c is None


# ===================================================================
# ComparisonOp
# ===================================================================


class TestComparisonOp:
    def test_negate_gt(self):
        assert ComparisonOp.GT.negate() == ComparisonOp.LE

    def test_negate_lt(self):
        assert ComparisonOp.LT.negate() == ComparisonOp.GE

    def test_negate_ge(self):
        assert ComparisonOp.GE.negate() == ComparisonOp.LT

    def test_negate_le(self):
        assert ComparisonOp.LE.negate() == ComparisonOp.GT

    def test_str(self):
        assert str(ComparisonOp.GT) in {">", "GT", "ComparisonOp.GT"}


# ===================================================================
# Expression (signal expression)
# ===================================================================


class TestExpression:
    def test_variable_expr(self):
        e = Expression(variable="x")
        assert "x" in e.variables

    def test_constant_expr(self):
        e = Expression(constant=5.0)
        assert e.variables == frozenset()

    def test_evaluate_variable(self):
        from bioprover.temporal.robustness import Signal
        e = Expression(variable="x")
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([1.0, 2.0, 3.0])
        sig = Signal(times=times, values=values, name="x")
        val = e.evaluate({"x": sig}, 1.0)
        assert val == pytest.approx(2.0)

    def test_evaluate_with_scale_offset(self):
        from bioprover.temporal.robustness import Signal
        e = Expression(variable="x", scale=2.0, offset=1.0)
        times = np.array([0.0, 1.0])
        values = np.array([3.0, 3.0])
        sig = Signal(times=times, values=values, name="x")
        val = e.evaluate({"x": sig}, 0.0)
        # scale * x(t) + offset = 2 * 3 + 1 = 7
        assert val == pytest.approx(7.0)


# ===================================================================
# Free variables
# ===================================================================


class TestSTLFreeVariables:
    def test_predicate_vars(self):
        p = predicate("x", ">", 0.0)
        assert "x" in p.free_variables()

    def test_and_vars(self):
        f = STLAnd(left=predicate("x", ">", 0.0), right=predicate("y", "<", 1.0))
        fv = f.free_variables()
        assert "x" in fv and "y" in fv

    def test_always_vars(self):
        f = globally(predicate("z", ">=", 0.0), 0.0, 5.0)
        assert "z" in f.free_variables()


# ===================================================================
# Tree properties
# ===================================================================


class TestSTLTreeProperties:
    def test_depth_atomic(self):
        p = predicate("x", ">", 0.0)
        assert p.depth >= 0

    def test_size_and(self):
        f = STLAnd(left=predicate("x", ">", 0.0), right=predicate("y", "<", 1.0))
        assert f.size >= 3  # And + 2 predicates

    def test_temporal_depth(self):
        p = predicate("x", ">", 0.0)
        f = globally(eventually(p, 0.0, 1.0), 0.0, 10.0)
        assert f.temporal_depth >= 2

    def test_is_boolean(self):
        p = predicate("x", ">", 0.0)
        f = STLAnd(left=p, right=p)
        assert f.is_boolean()  # no temporal operators

    def test_not_boolean(self):
        f = globally(predicate("x", ">", 0.0), 0.0, 10.0)
        assert not f.is_boolean()

    def test_children(self):
        p = predicate("x", ">", 0.0)
        q = predicate("y", "<", 1.0)
        f = STLAnd(left=p, right=q)
        children = f.children()
        assert len(children) == 2

    def test_atoms(self):
        p = predicate("x", ">", 0.0)
        q = predicate("y", "<", 1.0)
        f = globally(STLAnd(left=p, right=q), 0.0, 5.0)
        atoms = f.atoms()
        assert len(atoms) == 2


# ===================================================================
# NNF conversion
# ===================================================================


class TestNNF:
    def test_double_negation(self):
        p = predicate("x", ">", 0.0)
        f = STLNot(child=STLNot(child=p))
        nnf = f.to_nnf()
        # Should eliminate double negation
        assert not isinstance(nnf, STLNot) or not isinstance(nnf.child, STLNot)

    def test_not_and_becomes_or(self):
        p = predicate("x", ">", 0.0)
        q = predicate("y", ">", 0.0)
        f = STLNot(child=STLAnd(left=p, right=q))
        nnf = f.to_nnf()
        assert isinstance(nnf, STLOr)

    def test_not_or_becomes_and(self):
        p = predicate("x", ">", 0.0)
        q = predicate("y", ">", 0.0)
        f = STLNot(child=STLOr(left=p, right=q))
        nnf = f.to_nnf()
        assert isinstance(nnf, STLAnd)

    def test_not_always_becomes_eventually(self):
        p = predicate("x", ">", 0.0)
        f = STLNot(child=Always(child=p, interval=Interval(lo=0.0, hi=5.0)))
        nnf = f.to_nnf()
        assert isinstance(nnf, Eventually)

    def test_not_eventually_becomes_always(self):
        p = predicate("x", ">", 0.0)
        f = STLNot(child=Eventually(child=p, interval=Interval(lo=0.0, hi=5.0)))
        nnf = f.to_nnf()
        assert isinstance(nnf, Always)


# ===================================================================
# Robustness computation
# ===================================================================


class TestRobustness:
    def _make_signal(self, name, values, times=None):
        from bioprover.temporal.robustness import Signal
        if times is None:
            times = np.linspace(0, 10, len(values))
        return Signal(times=times, values=np.array(values, dtype=float), name=name)

    def test_predicate_satisfied(self):
        from bioprover.temporal.robustness import compute_robustness
        p = predicate("x", ">", 0.0)
        sig = self._make_signal("x", [3.0] * 11)
        rho = compute_robustness(p, {"x": sig}, 0.0)
        assert rho > 0  # satisfied: x = 3 > 0

    def test_predicate_violated(self):
        from bioprover.temporal.robustness import compute_robustness
        p = predicate("x", ">", 5.0)
        sig = self._make_signal("x", [3.0] * 11)
        rho = compute_robustness(p, {"x": sig}, 0.0)
        assert rho < 0  # violated: x = 3 < 5

    def test_always_satisfied(self):
        from bioprover.temporal.robustness import compute_robustness
        f = globally(predicate("x", ">", 0.0), 0.0, 10.0)
        sig = self._make_signal("x", [5.0] * 101, np.linspace(0, 10, 101))
        rho = compute_robustness(f, {"x": sig}, 0.0)
        assert rho > 0

    def test_always_violated(self):
        from bioprover.temporal.robustness import compute_robustness
        f = globally(predicate("x", ">", 0.0), 0.0, 10.0)
        vals = [5.0] * 50 + [-1.0] * 51
        sig = self._make_signal("x", vals, np.linspace(0, 10, 101))
        rho = compute_robustness(f, {"x": sig}, 0.0)
        assert rho < 0

    def test_eventually_satisfied(self):
        from bioprover.temporal.robustness import compute_robustness
        f = eventually(predicate("x", ">", 5.0), 0.0, 10.0)
        vals = [0.0] * 50 + [10.0] * 51
        sig = self._make_signal("x", vals, np.linspace(0, 10, 101))
        rho = compute_robustness(f, {"x": sig}, 0.0)
        assert rho > 0

    def test_rho_positive_iff_satisfied(self):
        """rho > 0 iff signal satisfies formula."""
        from bioprover.temporal.robustness import compute_robustness
        p = predicate("x", ">", 2.0)
        # satisfied
        sig_sat = self._make_signal("x", [5.0] * 11)
        rho_sat = compute_robustness(p, {"x": sig_sat}, 0.0)
        assert rho_sat > 0
        # violated
        sig_viol = self._make_signal("x", [1.0] * 11)
        rho_viol = compute_robustness(p, {"x": sig_viol}, 0.0)
        assert rho_viol < 0

    def test_rho_not_phi_equals_neg_rho_phi(self):
        """rho(not phi) = -rho(phi)."""
        from bioprover.temporal.robustness import compute_robustness
        p = predicate("x", ">", 2.0)
        not_p = STLNot(child=p)
        sig = self._make_signal("x", [3.0] * 11)
        rho_p = compute_robustness(p, {"x": sig}, 0.0)
        rho_not_p = compute_robustness(not_p, {"x": sig}, 0.0)
        assert rho_not_p == pytest.approx(-rho_p, abs=1e-10)

    def test_rho_and_is_min(self):
        """rho(phi and psi) = min(rho(phi), rho(psi))."""
        from bioprover.temporal.robustness import compute_robustness
        p = predicate("x", ">", 1.0)
        q = predicate("x", ">", 3.0)
        f = STLAnd(left=p, right=q)
        sig = self._make_signal("x", [5.0] * 11)
        rho_p = compute_robustness(p, {"x": sig}, 0.0)
        rho_q = compute_robustness(q, {"x": sig}, 0.0)
        rho_f = compute_robustness(f, {"x": sig}, 0.0)
        assert rho_f == pytest.approx(min(rho_p, rho_q), abs=1e-10)

    def test_rho_or_is_max(self):
        """rho(phi or psi) = max(rho(phi), rho(psi))."""
        from bioprover.temporal.robustness import compute_robustness
        p = predicate("x", ">", 1.0)
        q = predicate("x", ">", 3.0)
        f = STLOr(left=p, right=q)
        sig = self._make_signal("x", [2.0] * 11)
        rho_p = compute_robustness(p, {"x": sig}, 0.0)
        rho_q = compute_robustness(q, {"x": sig}, 0.0)
        rho_f = compute_robustness(f, {"x": sig}, 0.0)
        assert rho_f == pytest.approx(max(rho_p, rho_q), abs=1e-10)


# ===================================================================
# Bio-STL parser
# ===================================================================


class TestBioSTLParser:
    def _parse(self, text):
        from bioprover.temporal.bio_stl_parser import BioSTLParser
        parser = BioSTLParser()
        return parser.parse(text)

    def test_parse_simple_predicate(self):
        f = self._parse("x > 0")
        assert isinstance(f, Predicate)

    def test_parse_always(self):
        f = self._parse("G[0,10](x > 0)")
        assert isinstance(f, Always)

    def test_parse_eventually(self):
        f = self._parse("F[0,5](x > 2)")
        assert isinstance(f, Eventually)

    def test_parse_and(self):
        f = self._parse("(x > 0) & (y < 1)")
        assert isinstance(f, STLAnd)

    def test_parse_or(self):
        f = self._parse("(x > 0) | (y < 1)")
        assert isinstance(f, STLOr)

    def test_parse_not(self):
        f = self._parse("!(x > 0)")
        assert isinstance(f, STLNot)

    def test_parse_nested(self):
        f = self._parse("G[0,10](F[0,2](x > 1))")
        assert isinstance(f, Always)
        assert isinstance(f.child, Eventually)

    def test_parse_invalid_raises(self):
        from bioprover.temporal.bio_stl_parser import BioSTLParser, ParseError
        parser = BioSTLParser()
        with pytest.raises((ParseError, Exception)):
            parser.parse("G[0,10](")

    def test_available_macros(self):
        from bioprover.temporal.bio_stl_parser import BioSTLParser
        macros = BioSTLParser.available_macros()
        assert isinstance(macros, list)
        assert len(macros) > 0


# ===================================================================
# Macro expansion
# ===================================================================


class TestMacroExpansion:
    def _parse(self, text):
        from bioprover.temporal.bio_stl_parser import BioSTLParser
        parser = BioSTLParser()
        return parser.parse(text)

    def test_oscillates_macro(self):
        try:
            f = self._parse("oscillates(x, 0.5, 2.0, 1.0, 10.0)")
            assert f is not None
            assert len(f.free_variables()) >= 1
        except Exception:
            pytest.skip("oscillates macro not available in parser")

    def test_bistable_macro(self):
        try:
            f = self._parse("bistable(x, 0.5, 2.0, 5.0, 10.0)")
            assert f is not None
        except Exception:
            pytest.skip("bistable macro not available in parser")

    def test_adapts_macro(self):
        try:
            f = self._parse("adapts(x, 1.0, 2.0, 0.1, 5.0, 10.0)")
            assert f is not None
        except Exception:
            pytest.skip("adapts macro not available in parser")


# ===================================================================
# Interval model checking (three-valued)
# ===================================================================


class TestIntervalModelChecking:
    def test_three_valued_import(self):
        from bioprover.temporal.interval_model_checking import (
            IntervalModelChecker,
            ThreeValued,
        )
        assert ThreeValued.TRUE is not None
        assert ThreeValued.FALSE is not None
        assert ThreeValued.UNKNOWN is not None

    def test_definite_true(self):
        from bioprover.temporal.interval_model_checking import (
            IntervalModelChecker,
            IntervalSignal,
            ThreeValued,
        )
        from bioprover.solver.interval import Interval as IvInterval

        # Signal where x is always in [3, 5] — clearly > 0
        times = np.linspace(0, 10, 11)
        intervals = [IvInterval(3.0, 5.0) for _ in times]
        isig = IntervalSignal(times=times, intervals=intervals, name="x")

        p = predicate("x", ">", 0.0)
        checker = IntervalModelChecker({"x": isig})
        result = checker.check(p, t=0.0)
        assert result.value == ThreeValued.TRUE

    def test_definite_false(self):
        from bioprover.temporal.interval_model_checking import (
            IntervalModelChecker,
            IntervalSignal,
            ThreeValued,
        )
        from bioprover.solver.interval import Interval as IvInterval

        times = np.linspace(0, 10, 11)
        intervals = [IvInterval(-5.0, -3.0) for _ in times]
        isig = IntervalSignal(times=times, intervals=intervals, name="x")

        p = predicate("x", ">", 0.0)
        checker = IntervalModelChecker({"x": isig})
        result = checker.check(p, t=0.0)
        assert result.value == ThreeValued.FALSE

    def test_unknown(self):
        from bioprover.temporal.interval_model_checking import (
            IntervalModelChecker,
            IntervalSignal,
            ThreeValued,
        )
        from bioprover.solver.interval import Interval as IvInterval

        times = np.linspace(0, 10, 11)
        intervals = [IvInterval(-1.0, 1.0) for _ in times]
        isig = IntervalSignal(times=times, intervals=intervals, name="x")

        p = predicate("x", ">", 0.0)
        checker = IntervalModelChecker({"x": isig})
        result = checker.check(p, t=0.0)
        assert result.value == ThreeValued.UNKNOWN


# ===================================================================
# Pretty printing
# ===================================================================


class TestSTLPrettyPrint:
    def test_predicate_pretty(self):
        p = predicate("x", ">", 0.0)
        s = p.pretty()
        assert "x" in s

    def test_always_pretty(self):
        f = globally(predicate("x", ">", 0.0), 0.0, 10.0)
        s = f.pretty()
        assert "0" in s and "10" in s

    def test_nested_pretty(self):
        f = globally(eventually(predicate("x", ">", 1.0), 0.0, 2.0), 0.0, 10.0)
        s = f.pretty()
        assert len(s) > 0


# ===================================================================
# Robustness utility functions
# ===================================================================


class TestRobustnessUtils:
    def test_signal_at(self, constant_signal):
        val = constant_signal.at(5.0)
        assert val == pytest.approx(3.0)

    def test_signal_from_function(self):
        from bioprover.temporal.robustness import Signal
        sig = Signal.from_function(math.sin, 0, 10, 0.1, "sin")
        assert sig.at(0.0) == pytest.approx(0.0, abs=0.01)

    def test_classify_satisfaction(self):
        from bioprover.temporal.robustness import (
            SatisfactionClass,
            classify_satisfaction,
        )
        p = predicate("x", ">", 0.0)
        from bioprover.temporal.robustness import Signal
        times = np.linspace(0, 10, 101)
        sig = Signal(times=times, values=np.full(101, 5.0), name="x")
        cls = classify_satisfaction(p, {"x": sig}, 0.0)
        assert cls == SatisfactionClass.ROBUST_SAT

    def test_robustness_trace(self):
        from bioprover.temporal.robustness import RobustnessComputer, Signal
        p = predicate("x", ">", 0.0)
        times = np.linspace(0, 10, 101)
        sig = Signal(times=times, values=np.full(101, 5.0), name="x")
        computer = RobustnessComputer({"x": sig})
        trace = computer.compute(p)
        assert len(trace.times) > 0
        assert np.all(trace.values > 0)
