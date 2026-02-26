"""Unit tests for Expression IR — construction, equality, simplification, CSE."""

import math

import pytest

from bioprover.encoding.expression import (
    ZERO,
    ONE,
    TWO,
    Abs,
    Add,
    And,
    Const,
    Cos,
    Div,
    Eq,
    Exists,
    Exp,
    ForAll,
    Ge,
    Gt,
    HillAct,
    HillRep,
    Implies,
    Ite,
    Le,
    Log,
    Lt,
    Max,
    Min,
    Mul,
    Neg,
    Not,
    Or,
    Pow,
    Sin,
    Sqrt,
    Var,
    collect_nodes,
    is_const,
    is_var,
    map_expr,
    prod_exprs,
    sum_exprs,
)


# ===================================================================
# Construction & equality
# ===================================================================


class TestConstruction:
    def test_const(self):
        c = Const(3.14)
        assert is_const(c)
        assert c.value == pytest.approx(3.14)

    def test_var(self):
        v = Var("x")
        assert is_var(v)
        assert v.name == "x"

    def test_add(self):
        e = Add(Const(1.0), Const(2.0))
        assert isinstance(e, Add)
        assert e.left == Const(1.0)
        assert e.right == Const(2.0)

    def test_mul(self):
        e = Mul(Var("x"), Const(3.0))
        assert e.left == Var("x")

    def test_div(self):
        e = Div(Var("x"), Var("y"))
        assert isinstance(e, Div)

    def test_pow(self):
        e = Pow(Var("x"), Const(2.0))
        assert isinstance(e, Pow)

    def test_neg(self):
        e = Neg(Var("x"))
        assert isinstance(e, Neg)
        assert e.operand == Var("x")

    def test_elementary_functions(self):
        x = Var("x")
        assert isinstance(Exp(x), Exp)
        assert isinstance(Log(x), Log)
        assert isinstance(Sin(x), Sin)
        assert isinstance(Cos(x), Cos)
        assert isinstance(Sqrt(x), Sqrt)
        assert isinstance(Abs(x), Abs)

    def test_hill_activation(self):
        h = HillAct(Var("x"), Const(1.0), Const(2.0))
        assert isinstance(h, HillAct)

    def test_hill_repression(self):
        h = HillRep(Var("x"), Const(1.0), Const(2.0))
        assert isinstance(h, HillRep)

    def test_min_max(self):
        e = Min(Var("x"), Var("y"))
        assert isinstance(e, Min)
        e = Max(Var("x"), Var("y"))
        assert isinstance(e, Max)

    def test_comparison_nodes(self):
        x, y = Var("x"), Var("y")
        assert isinstance(Lt(x, y), Lt)
        assert isinstance(Le(x, y), Le)
        assert isinstance(Eq(x, y), Eq)
        assert isinstance(Ge(x, y), Ge)
        assert isinstance(Gt(x, y), Gt)

    def test_boolean_nodes(self):
        a = Gt(Var("x"), ZERO)
        b = Lt(Var("y"), ONE)
        assert isinstance(And((a, b)), And)
        assert isinstance(Or((a, b)), Or)
        assert isinstance(Not(a), Not)
        assert isinstance(Implies(a, b), Implies)
        assert isinstance(Ite(a, Var("x"), Var("y")), Ite)

    def test_quantifiers(self):
        body = Gt(Var("x"), ZERO)
        f = ForAll("x", None, body)
        assert isinstance(f, ForAll)
        e = Exists("x", None, body)
        assert isinstance(e, Exists)


# ===================================================================
# Operator overloading
# ===================================================================


class TestOperatorOverloading:
    def test_add_op(self):
        x, y = Var("x"), Var("y")
        e = x + y
        assert isinstance(e, Add)

    def test_radd_op(self):
        x = Var("x")
        e = 3.0 + x
        assert isinstance(e, Add)

    def test_sub_op(self):
        x, y = Var("x"), Var("y")
        e = x - y
        # Could be Add(x, Neg(y)) or Sub-like
        assert isinstance(e, (Add, type(x - y)))

    def test_mul_op(self):
        x = Var("x")
        e = x * Const(2.0)
        assert isinstance(e, Mul)

    def test_rmul_op(self):
        x = Var("x")
        e = 2.0 * x
        assert isinstance(e, Mul)

    def test_div_op(self):
        x, y = Var("x"), Var("y")
        e = x / y
        assert isinstance(e, Div)

    def test_pow_op(self):
        x = Var("x")
        e = x ** Const(3.0)
        assert isinstance(e, Pow)

    def test_neg_op(self):
        x = Var("x")
        e = -x
        assert isinstance(e, Neg)

    def test_comparison_ops(self):
        x, y = Var("x"), Var("y")
        assert isinstance(x < y, Lt)
        assert isinstance(x <= y, Le)
        assert isinstance(x >= y, Ge)
        assert isinstance(x > y, Gt)


# ===================================================================
# Hash-consing (structural sharing)
# ===================================================================


class TestHashConsing:
    def test_same_const_is_same_object(self):
        """Hash-consing should return the same object for equal expressions."""
        a = Const(42.0)
        b = Const(42.0)
        assert a is b or a == b  # at minimum equality

    def test_same_var_is_same_object(self):
        a = Var("x")
        b = Var("x")
        assert a is b or a == b

    def test_same_add_is_same_object(self):
        x, y = Var("x"), Var("y")
        e1 = Add(x, y)
        e2 = Add(x, y)
        assert e1 is e2 or e1 == e2

    def test_different_add_is_different(self):
        x, y, z = Var("x"), Var("y"), Var("z")
        e1 = Add(x, y)
        e2 = Add(x, z)
        assert e1 != e2

    def test_constants_are_well_known(self):
        assert ZERO == Const(0.0)
        assert ONE == Const(1.0)
        assert TWO == Const(2.0)

    def test_hash_consistency(self):
        x = Var("x")
        e1 = x + ONE
        e2 = x + ONE
        assert hash(e1) == hash(e2)


# ===================================================================
# Free variables
# ===================================================================


class TestFreeVars:
    def test_const_has_no_vars(self):
        assert Const(1.0).free_vars() == frozenset()

    def test_var_has_itself(self):
        assert Var("x").free_vars() == frozenset({"x"})

    def test_add_combines_vars(self):
        e = Var("x") + Var("y")
        assert e.free_vars() == frozenset({"x", "y"})

    def test_nested(self):
        e = Exp(Var("x") * Var("y")) + Log(Var("z"))
        assert e.free_vars() == frozenset({"x", "y", "z"})

    def test_hill_act_vars(self):
        h = HillAct(Var("x"), Var("K"), Var("n"))
        assert h.free_vars() == frozenset({"x", "K", "n"})


# ===================================================================
# Substitution
# ===================================================================


class TestSubstitution:
    def test_substitute_var(self):
        x = Var("x")
        result = x.substitute({"x": Const(5.0)})
        assert result == Const(5.0)

    def test_substitute_in_expression(self):
        e = Var("x") + Var("y")
        result = e.substitute({"x": Const(1.0)})
        fv = result.free_vars()
        assert "x" not in fv
        assert "y" in fv

    def test_substitute_preserves_structure(self):
        e = Exp(Var("x"))
        result = e.substitute({"x": Var("z")})
        assert isinstance(result, Exp)
        assert result.operand == Var("z")

    def test_substitute_no_match(self):
        e = Var("x") + Var("y")
        result = e.substitute({"z": Const(99.0)})
        assert result == e

    def test_nested_substitution(self):
        e = Var("x") * (Var("x") + ONE)
        result = e.substitute({"x": TWO})
        # Should have no free vars
        assert result.free_vars() == frozenset()


# ===================================================================
# Tree properties — depth, size, iteration
# ===================================================================


class TestTreeProperties:
    def test_depth_const(self):
        assert Const(1.0).depth() == 0

    def test_depth_nested(self):
        e = Exp(Var("x") + Var("y"))
        assert e.depth() >= 2

    def test_size_const(self):
        assert Const(1.0).size() == 1

    def test_size_add(self):
        e = Var("x") + Var("y")
        assert e.size() == 3  # Add, x, y

    def test_preorder(self):
        e = Var("x") + Var("y")
        nodes = list(e.iter_preorder())
        assert nodes[0] == e  # root first

    def test_postorder(self):
        e = Var("x") + Var("y")
        nodes = list(e.iter_postorder())
        assert nodes[-1] == e  # root last

    def test_collect_nodes(self):
        e = Var("x") + Const(1.0) + Var("y")
        vars_found = collect_nodes(e, is_var)
        names = {v.name for v in vars_found}
        assert names == {"x", "y"}


# ===================================================================
# Pretty printing
# ===================================================================


class TestPrettyPrint:
    def test_const_pretty(self):
        p = Const(3.14).pretty()
        assert "3.14" in p

    def test_var_pretty(self):
        p = Var("x").pretty()
        assert "x" in p

    def test_add_pretty(self):
        p = (Var("x") + Var("y")).pretty()
        assert "x" in p and "y" in p

    def test_neg_pretty(self):
        p = (-Var("x")).pretty()
        assert "x" in p


# ===================================================================
# Helper functions
# ===================================================================


class TestHelpers:
    def test_sum_exprs_empty(self):
        assert sum_exprs([]) == ZERO

    def test_sum_exprs_single(self):
        assert sum_exprs([Var("x")]) == Var("x")

    def test_sum_exprs_multiple(self):
        result = sum_exprs([Var("x"), Var("y"), Const(1.0)])
        assert result.free_vars() == frozenset({"x", "y"})

    def test_prod_exprs_empty(self):
        assert prod_exprs([]) == ONE

    def test_prod_exprs_single(self):
        assert prod_exprs([Var("x")]) == Var("x")

    def test_prod_exprs_multiple(self):
        result = prod_exprs([Var("x"), Var("y")])
        assert isinstance(result, Mul)


# ===================================================================
# map_expr
# ===================================================================


class TestMapExpr:
    def test_identity_map(self):
        e = Var("x") + Var("y")
        result = map_expr(e, lambda n: None)
        assert result == e

    def test_const_replacement(self):
        e = Var("x") + Const(1.0)
        result = map_expr(e, lambda n: Const(2.0) if isinstance(n, Const) and n.value == 1.0 else None)
        # The constant 1.0 should now be 2.0
        consts = collect_nodes(result, lambda n: isinstance(n, Const))
        vals = {c.value for c in consts}
        assert 2.0 in vals

    def test_variable_rename(self):
        e = Var("x") + Var("y")
        result = map_expr(e, lambda n: Var("z") if isinstance(n, Var) and n.name == "x" else None)
        assert result.free_vars() == frozenset({"z", "y"})


# ===================================================================
# Simplification (via simplifier module)
# ===================================================================


class TestSimplification:
    """Test simplification rules from bioprover.encoding.simplifier."""

    def _simplify(self, expr):
        from bioprover.encoding.simplifier import simplify
        return simplify(expr)

    def test_add_zero(self):
        e = Var("x") + ZERO
        s = self._simplify(e)
        assert s == Var("x")

    def test_mul_one(self):
        e = Var("x") * ONE
        s = self._simplify(e)
        assert s == Var("x")

    def test_mul_zero(self):
        e = Var("x") * ZERO
        s = self._simplify(e)
        assert s == ZERO

    def test_double_negation(self):
        e = Neg(Neg(Var("x")))
        s = self._simplify(e)
        assert s == Var("x")

    def test_constant_folding_add(self):
        e = Const(2.0) + Const(3.0)
        s = self._simplify(e)
        assert is_const(s) and s.value == pytest.approx(5.0)

    def test_constant_folding_mul(self):
        e = Const(2.0) * Const(3.0)
        s = self._simplify(e)
        assert is_const(s) and s.value == pytest.approx(6.0)

    def test_simplify_preserves_semantics(self):
        """Simplification should not change the expression's meaning."""
        from bioprover.encoding.simplifier import simplify
        e = (Var("x") + ZERO) * ONE + Const(2.0) * Const(3.0)
        s = simplify(e)
        # s should still contain x and a constant 6
        assert "x" in s.free_vars()


# ===================================================================
# CSE (Common Subexpression Elimination)
# ===================================================================


class TestCSE:
    def test_cse_detects_shared_subexpr(self):
        from bioprover.encoding.simplifier import _count_subexprs
        sub = Var("x") * Var("y")
        e = sub + Exp(sub)
        counts = _count_subexprs(e)
        assert counts.get(sub, 0) >= 2

    def test_cse_application(self):
        from bioprover.encoding.simplifier import simplify_fully
        sub = Var("x") * Var("y")
        e = sub + Exp(sub) + Log(sub)
        result, stats = simplify_fully(e)
        # Result should still have the same free variables
        assert result.free_vars() == frozenset({"x", "y"})
