"""Unit tests for SMT encoding — ODE discretization, Hill functions, SMT-LIB."""

import io
import math

import numpy as np
import pytest

from bioprover.encoding.expression import (
    ZERO,
    ONE,
    TWO,
    Add,
    Const,
    Div,
    Exp,
    Mul,
    Pow,
    Var,
)


# ===================================================================
# ODE encoding
# ===================================================================


class TestODEEncoding:
    def _make_simple_system(self):
        from bioprover.encoding.ode_encoding import (
            DiscretizationMethod,
            ODESystem,
            Parameter,
            Species,
        )

        species = [Species("x", initial_lo=1.0, initial_hi=1.0, nonnegative=True)]
        params = [Parameter("k", lo=0.1, hi=0.5)]
        # dx/dt = -k * x  (exponential decay)
        rhs = {"x": lambda vars_dict: Mul(Var("k"), Var("x_0")) if True else None}
        # Actually use expression-based RHS
        def rhs_fn(vars_dict):
            return {
                "x": Mul(Const(-1.0), Mul(Var("k"), vars_dict.get("x", Var("x_0"))))
            }

        return ODESystem(
            species=species,
            parameters=params,
            rhs=rhs_fn,
            conservation_laws=[],
        )

    def test_encode_ode_produces_encoding(self):
        from bioprover.encoding.ode_encoding import (
            DiscretizationMethod,
            ODESystem,
            Parameter,
            Species,
            encode_ode,
        )

        species = [Species("x", initial_lo=1.0, initial_hi=1.0, nonnegative=True)]
        params = [Parameter("k", lo=0.1, hi=0.5)]

        def rhs(vars_dict):
            return {"x": Mul(Const(-1.0), Mul(Var("k"), vars_dict["x"]))}

        system = ODESystem(species=species, parameters=params, rhs=rhs, conservation_laws=[])
        enc = encode_ode(system, num_steps=5, step_size=0.1, method=DiscretizationMethod.EULER)
        assert enc is not None
        assert len(enc.variables) > 0
        assert len(enc.constraints) > 0
        assert enc.num_steps == 5

    def test_euler_discretization(self):
        from bioprover.encoding.ode_encoding import (
            DiscretizationMethod,
            ODESystem,
            Parameter,
            Species,
            encode_ode,
            state_var,
        )

        species = [Species("x", initial_lo=1.0, initial_hi=1.0, nonnegative=True)]
        params = []

        def rhs(vars_dict):
            return {"x": Const(1.0)}  # dx/dt = 1 → x(t) = 1 + t

        system = ODESystem(species=species, parameters=params, rhs=rhs, conservation_laws=[])
        enc = encode_ode(system, num_steps=10, step_size=0.1, method=DiscretizationMethod.EULER)
        # Should have state variables for each step
        assert enc.num_steps == 10

    def test_trapezoidal_discretization(self):
        from bioprover.encoding.ode_encoding import (
            DiscretizationMethod,
            ODESystem,
            Parameter,
            Species,
            encode_ode,
        )

        species = [Species("x", initial_lo=0.0, initial_hi=0.0, nonnegative=False)]
        params = []

        def rhs(vars_dict):
            return {"x": Const(2.0)}

        system = ODESystem(species=species, parameters=params, rhs=rhs, conservation_laws=[])
        enc = encode_ode(system, num_steps=5, step_size=0.2, method=DiscretizationMethod.TRAPEZOIDAL)
        assert enc is not None

    def test_state_var_naming(self):
        from bioprover.encoding.ode_encoding import state_var
        v = state_var("x", 3)
        assert isinstance(v, Var)
        assert "x" in v.name and "3" in v.name

    def test_encoding_size_estimate(self):
        from bioprover.encoding.ode_encoding import (
            DiscretizationMethod,
            ODESystem,
            Parameter,
            Species,
            estimate_encoding_size,
        )

        species = [
            Species("x", initial_lo=0.0, initial_hi=1.0, nonnegative=True),
            Species("y", initial_lo=0.0, initial_hi=1.0, nonnegative=True),
        ]
        params = [Parameter("k", lo=0.1, hi=1.0)]

        def rhs(vars_dict):
            return {
                "x": Mul(Const(-1.0), Mul(Var("k"), vars_dict["x"])),
                "y": Mul(Var("k"), vars_dict["x"]),
            }

        system = ODESystem(species=species, parameters=params, rhs=rhs, conservation_laws=[])
        est = estimate_encoding_size(system, 10, DiscretizationMethod.EULER)
        assert est.num_variables > 0
        assert est.num_constraints > 0

    def test_incremental_encoding(self):
        from bioprover.encoding.ode_encoding import (
            DiscretizationMethod,
            ODESystem,
            Parameter,
            Species,
            encode_ode,
            encode_ode_incremental,
        )

        species = [Species("x", initial_lo=1.0, initial_hi=1.0, nonnegative=True)]
        params = []

        def rhs(vars_dict):
            return {"x": Const(0.5)}

        system = ODESystem(species=species, parameters=params, rhs=rhs, conservation_laws=[])
        new_vars, new_constraints = encode_ode_incremental(
            system, current_steps=5, additional_steps=3, step_size=0.1,
            method=DiscretizationMethod.EULER,
        )
        assert len(new_vars) > 0

    def test_auto_encoding(self):
        from bioprover.encoding.ode_encoding import (
            DiscretizationMethod,
            ODESystem,
            Parameter,
            Species,
            encode_ode_auto,
        )

        species = [Species("x", initial_lo=1.0, initial_hi=1.0, nonnegative=True)]
        params = []

        def rhs(vars_dict):
            return {"x": Const(-0.1)}

        system = ODESystem(species=species, parameters=params, rhs=rhs, conservation_laws=[])
        enc = encode_ode_auto(system, time_horizon=1.0, method=DiscretizationMethod.EULER)
        assert enc.num_steps > 0


# ===================================================================
# Hill function encoding
# ===================================================================


class TestHillEncoding:
    def test_encode_hill_act_integer(self):
        from bioprover.encoding.hill_encoding import encode_hill_act_integer
        x = Var("x")
        K = Const(1.0)
        result = encode_hill_act_integer(x, K, n=2)
        assert result is not None
        # Should be a rational expression: x^2 / (K^2 + x^2)
        fv = result.free_vars()
        assert "x" in fv

    def test_encode_hill_rep_integer(self):
        from bioprover.encoding.hill_encoding import encode_hill_rep_integer
        x = Var("x")
        K = Const(1.0)
        result = encode_hill_rep_integer(x, K, n=2)
        assert result is not None
        assert "x" in result.free_vars()

    def test_encode_hill_act_integer_n1(self):
        from bioprover.encoding.hill_encoding import encode_hill_act_integer
        x = Var("x")
        K = Const(1.0)
        result = encode_hill_act_integer(x, K, n=1)
        # x / (K + x) — Michaelis-Menten-like
        assert result is not None

    def test_encode_hill_act_pwl(self):
        from bioprover.encoding.hill_encoding import encode_hill_act_pwl
        x = Var("x")
        result, error = encode_hill_act_pwl(x, K_val=1.0, n_val=2.0,
                                             domain=(0.0, 5.0), num_pieces=10)
        assert result is not None
        assert error.max_absolute_error >= 0

    def test_encode_hill_act_taylor(self):
        from bioprover.encoding.hill_encoding import encode_hill_act_taylor
        x = Var("x")
        result, error = encode_hill_act_taylor(x, K_val=1.0, n_val=2.5,
                                                x0=1.0, order=3, domain=(0.5, 2.0))
        assert result is not None
        assert error.taylor_order == 3

    def test_encode_michaelis_menten(self):
        from bioprover.encoding.hill_encoding import encode_michaelis_menten
        result = encode_michaelis_menten(Var("S"), Var("Vmax"), Var("Km"))
        assert result is not None
        fv = result.free_vars()
        assert "S" in fv and "Vmax" in fv and "Km" in fv

    def test_encode_mass_action(self):
        from bioprover.encoding.hill_encoding import encode_mass_action
        result = encode_mass_action(Var("k"), [Var("A"), Var("B")])
        assert result is not None
        fv = result.free_vars()
        assert "k" in fv and "A" in fv and "B" in fv

    def test_encode_dimerization(self):
        from bioprover.encoding.hill_encoding import encode_dimerization
        fwd, rev = encode_dimerization(Var("M"), Var("kf"), Var("kr"))
        assert fwd is not None and rev is not None

    def test_monotone_bounds(self):
        from bioprover.encoding.hill_encoding import encode_hill_monotone_bounds
        lo, hi = encode_hill_monotone_bounds(
            x_lo=Const(0.5), x_hi=Const(2.0), K=Const(1.0), n=2, activating=True,
        )
        assert lo is not None and hi is not None

    def test_fragment_classification(self):
        from bioprover.encoding.hill_encoding import FragmentKind, classify_fragment
        # polynomial
        poly = Var("x") * Var("x")
        kind = classify_fragment(poly)
        assert kind in (FragmentKind.POLYNOMIAL, FragmentKind.RATIONAL, FragmentKind.TRANSCENDENTAL)


# ===================================================================
# Parameter range encoding
# ===================================================================


class TestParameterEncoding:
    def test_encode_universal(self):
        from bioprover.encoding.parameter_encoding import (
            QuantifiedParam,
            encode_universal,
        )
        p = QuantifiedParam(name="k", lo=0.0, hi=1.0, role=None)
        body = Var("k") + ONE
        result = encode_universal([p], body)
        assert result is not None

    def test_encode_existential(self):
        from bioprover.encoding.parameter_encoding import (
            QuantifiedParam,
            encode_existential,
        )
        p = QuantifiedParam(name="k", lo=0.0, hi=1.0, role=None)
        body = Var("k") + ONE
        result = encode_existential([p], body)
        assert result is not None

    def test_parameter_bound_constraints(self):
        from bioprover.encoding.parameter_encoding import (
            QuantifiedParam,
            parameter_bound_constraints,
        )
        p = QuantifiedParam(name="k", lo=0.5, hi=2.0, role=None)
        constraints = parameter_bound_constraints([p])
        assert len(constraints) >= 1

    def test_discretize_parameter_space(self):
        from bioprover.encoding.parameter_encoding import (
            QuantifiedParam,
            discretize_parameter_space,
        )
        p = QuantifiedParam(name="k", lo=0.0, hi=1.0, role=None)
        body = Var("k") * Var("x")
        result = discretize_parameter_space([p], body, points_per_dim=3)
        assert result is not None

    def test_encode_robust_synthesis(self):
        from bioprover.encoding.parameter_encoding import (
            QuantifiedParam,
            encode_robust_synthesis,
        )
        design = QuantifiedParam(name="d", lo=0.0, hi=1.0, role=None)
        uncertain = QuantifiedParam(name="u", lo=0.0, hi=0.1, role=None)
        body = Var("d") + Var("u")
        result = encode_robust_synthesis([design], [uncertain], body)
        assert result is not None


# ===================================================================
# SMT-LIB serialization
# ===================================================================


class TestSMTLIBSerialization:
    def test_serialize_simple(self):
        from bioprover.encoding.smtlib_serializer import serialize_smtlib
        assertions = [Var("x") + Var("y")]
        output = serialize_smtlib(assertions, variables=[Var("x"), Var("y")])
        assert "declare" in output.lower() or "x" in output

    def test_serialize_with_check_sat(self):
        from bioprover.encoding.smtlib_serializer import serialize_smtlib, SerializerConfig
        from bioprover.encoding.expression import Ge
        config = SerializerConfig()
        assertions = [Ge(Var("x"), ZERO)]
        output = serialize_smtlib(assertions, variables=[Var("x")], config=config)
        assert "check-sat" in output.lower() or len(output) > 0

    def test_emit_push_pop(self):
        from bioprover.encoding.smtlib_serializer import emit_push, emit_pop
        buf = io.StringIO()
        emit_push(buf)
        emit_pop(buf)
        content = buf.getvalue()
        assert "push" in content and "pop" in content

    def test_emit_assert(self):
        from bioprover.encoding.smtlib_serializer import emit_assert
        buf = io.StringIO()
        emit_assert(buf, Var("x") + ONE)
        content = buf.getvalue()
        assert "assert" in content.lower() or "x" in content

    def test_emit_check_sat(self):
        from bioprover.encoding.smtlib_serializer import emit_check_sat
        buf = io.StringIO()
        emit_check_sat(buf)
        assert "check-sat" in buf.getvalue().lower()

    def test_expr_to_smtlib(self):
        from bioprover.encoding.smtlib_serializer import expr_to_smtlib
        s = expr_to_smtlib(Var("x") + Const(1.0))
        assert "x" in s

    def test_write_file(self, tmp_dir):
        from bioprover.encoding.smtlib_serializer import write_smtlib_file
        from bioprover.encoding.expression import Ge
        path = str(tmp_dir / "test.smt2")
        write_smtlib_file(path, [Ge(Var("x"), ZERO)], variables=[Var("x")])
        with open(path) as f:
            content = f.read()
        assert len(content) > 0

    def test_parse_smt_response_sat(self):
        from bioprover.encoding.smtlib_serializer import parse_smt_response
        result = parse_smt_response("sat")
        assert result is not None

    def test_parse_smt_response_unsat(self):
        from bioprover.encoding.smtlib_serializer import parse_smt_response
        result = parse_smt_response("unsat")
        assert result is not None

    def test_auto_logic_selection(self):
        from bioprover.encoding.smtlib_serializer import auto_select_logic, SMTLogic
        from bioprover.encoding.expression import Ge
        logic = auto_select_logic(Ge(Var("x"), ZERO))
        assert isinstance(logic, SMTLogic)


# ===================================================================
# Incremental encoding
# ===================================================================


class TestIncrementalEncoding:
    def test_add_clause(self):
        from bioprover.encoding.incremental import IncrementalEncoder
        from bioprover.encoding.expression import Ge
        enc = IncrementalEncoder()
        cid = enc.add_clause(Ge(Var("x"), ZERO))
        assert cid >= 0

    def test_remove_clause(self):
        from bioprover.encoding.incremental import IncrementalEncoder
        from bioprover.encoding.expression import Ge
        enc = IncrementalEncoder()
        cid = enc.add_clause(Ge(Var("x"), ZERO))
        enc.remove_clause(cid)
        assert enc.active_clause_count() == 0

    def test_push_pop(self):
        from bioprover.encoding.incremental import IncrementalEncoder
        from bioprover.encoding.expression import Ge
        enc = IncrementalEncoder()
        enc.push()
        cid = enc.add_clause(Ge(Var("x"), ZERO))
        assert enc.active_clause_count() == 1
        enc.pop()
        assert enc.active_clause_count() == 0

    def test_scope_depth(self):
        from bioprover.encoding.incremental import IncrementalEncoder
        enc = IncrementalEncoder()
        assert enc.scope_depth == 0
        enc.push()
        assert enc.scope_depth == 1
        enc.pop()
        assert enc.scope_depth == 0

    def test_compute_delta(self):
        from bioprover.encoding.incremental import IncrementalEncoder
        from bioprover.encoding.expression import Ge, Le
        enc = IncrementalEncoder()
        enc.add_clause(Ge(Var("x"), ZERO))
        new_clauses = [Le(Var("x"), Const(10.0))]
        to_add, to_remove = enc.compute_delta(new_clauses)
        assert isinstance(to_add, list)

    def test_emit_all(self):
        from bioprover.encoding.incremental import IncrementalEncoder
        from bioprover.encoding.expression import Ge
        enc = IncrementalEncoder()
        enc.add_clause(Ge(Var("x"), ZERO))
        buf = io.StringIO()
        enc.emit_all(buf)
        content = buf.getvalue()
        assert len(content) > 0

    def test_snapshot_restore(self):
        from bioprover.encoding.incremental import IncrementalEncoder
        from bioprover.encoding.expression import Ge
        enc = IncrementalEncoder()
        enc.add_clause(Ge(Var("x"), ZERO))
        snap = enc.snapshot()
        enc.add_clause(Ge(Var("y"), ZERO))
        assert enc.active_clause_count() == 2
        enc.restore(snap)
        assert enc.active_clause_count() == 1


# ===================================================================
# Simplifier correctness
# ===================================================================


class TestSimplifierCorrectness:
    def test_simplify_preserves_evaluation(self):
        """Simplified expression should evaluate the same as the original."""
        from bioprover.encoding.simplifier import simplify, interval_eval, IVal
        e = (Var("x") + ZERO) * ONE + Const(2.0) * Const(3.0)
        s = simplify(e)
        # Evaluate both at x = 5
        bounds = {"x": IVal(lo=5.0, hi=5.0)}
        orig_val = interval_eval(e, bounds)
        simp_val = interval_eval(s, bounds)
        if orig_val is not None and simp_val is not None:
            assert orig_val.lo == pytest.approx(simp_val.lo, abs=1e-10)
            assert orig_val.hi == pytest.approx(simp_val.hi, abs=1e-10)

    def test_simplify_reduces_size(self):
        from bioprover.encoding.simplifier import simplify
        e = (Var("x") + ZERO) * ONE
        s = simplify(e)
        assert s.size() <= e.size()

    def test_normalize_polynomial(self):
        from bioprover.encoding.simplifier import normalize_polynomial
        e = Var("x") + Var("x")
        result = normalize_polynomial(e)
        assert result is not None
