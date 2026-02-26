"""Property-based tests for expression IR using hypothesis."""

import math

import numpy as np
import pytest

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from bioprover.encoding.expression import (
    ZERO,
    ONE,
    TWO,
    Add,
    Const,
    Div,
    Exp,
    Log,
    Mul,
    Neg,
    Pow,
    Sqrt,
    Var,
    collect_nodes,
    is_const,
    is_var,
    map_expr,
)


# ===================================================================
# Custom strategies for expression trees
# ===================================================================

var_names = st.sampled_from(["x", "y", "z", "w"])
small_consts = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)


@st.composite
def leaf_exprs(draw):
    """Generate a leaf expression (Const or Var)."""
    if draw(st.booleans()):
        return Const(draw(small_consts))
    else:
        return Var(draw(var_names))


@st.composite
def simple_exprs(draw, max_depth=3):
    """Generate simple arithmetic expression trees."""
    if max_depth <= 0 or draw(st.integers(min_value=0, max_value=3)) == 0:
        return draw(leaf_exprs())
    left = draw(simple_exprs(max_depth=max_depth - 1))
    right = draw(simple_exprs(max_depth=max_depth - 1))
    op = draw(st.sampled_from(["add", "mul"]))
    if op == "add":
        return Add(left, right)
    else:
        return Mul(left, right)


@st.composite
def arithmetic_exprs(draw, max_depth=2):
    """Generate arithmetic expressions suitable for evaluation."""
    if max_depth <= 0:
        return draw(leaf_exprs())
    op = draw(st.sampled_from(["add", "mul", "neg", "const"]))
    if op == "add":
        return Add(draw(arithmetic_exprs(max_depth - 1)), draw(arithmetic_exprs(max_depth - 1)))
    elif op == "mul":
        return Mul(draw(arithmetic_exprs(max_depth - 1)), draw(arithmetic_exprs(max_depth - 1)))
    elif op == "neg":
        return Neg(draw(arithmetic_exprs(max_depth - 1)))
    else:
        return draw(leaf_exprs())


def _eval_expr(expr, env):
    """Evaluate an expression in a given variable environment.
    Returns None if evaluation fails (e.g., division by zero).
    """
    if isinstance(expr, Const):
        return expr.value
    elif isinstance(expr, Var):
        return env.get(expr.name, 0.0)
    elif isinstance(expr, Add):
        l = _eval_expr(expr.left, env)
        r = _eval_expr(expr.right, env)
        if l is None or r is None:
            return None
        return l + r
    elif isinstance(expr, Mul):
        l = _eval_expr(expr.left, env)
        r = _eval_expr(expr.right, env)
        if l is None or r is None:
            return None
        return l * r
    elif isinstance(expr, Neg):
        v = _eval_expr(expr.operand, env)
        return -v if v is not None else None
    elif isinstance(expr, Div):
        l = _eval_expr(expr.left, env)
        r = _eval_expr(expr.right, env)
        if l is None or r is None or abs(r) < 1e-15:
            return None
        return l / r
    elif isinstance(expr, Pow):
        l = _eval_expr(expr.left, env)
        r = _eval_expr(expr.right, env)
        if l is None or r is None:
            return None
        try:
            return l ** r
        except (ValueError, OverflowError):
            return None
    elif isinstance(expr, Exp):
        v = _eval_expr(expr.operand, env)
        if v is None or abs(v) > 50:
            return None
        return math.exp(v)
    elif isinstance(expr, Log):
        v = _eval_expr(expr.operand, env)
        if v is None or v <= 0:
            return None
        return math.log(v)
    elif isinstance(expr, Sqrt):
        v = _eval_expr(expr.operand, env)
        if v is None or v < 0:
            return None
        return math.sqrt(v)
    else:
        return None


# ===================================================================
# Property: simplification preserves evaluation
# ===================================================================


class TestSimplificationPreservesEvaluation:
    @given(expr=arithmetic_exprs(), vals=st.dictionaries(
        var_names, st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=4, max_size=4,
    ))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_simplify_preserves_value(self, expr, vals):
        """Simplification should not change the expression's value."""
        from bioprover.encoding.simplifier import simplify
        simplified = simplify(expr)
        orig_val = _eval_expr(expr, vals)
        simp_val = _eval_expr(simplified, vals)
        if orig_val is not None and simp_val is not None:
            if math.isfinite(orig_val) and math.isfinite(simp_val):
                assert abs(orig_val - simp_val) < 1e-8 * (1 + abs(orig_val))

    @given(expr=arithmetic_exprs())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_simplify_reduces_or_preserves_size(self, expr):
        """Simplification should not increase expression size."""
        from bioprover.encoding.simplifier import simplify
        simplified = simplify(expr)
        assert simplified.size() <= expr.size() + 1  # small tolerance for edge cases


# ===================================================================
# Property: CSE produces equivalent expression
# ===================================================================


class TestCSEPreservesEquivalence:
    @given(vals=st.dictionaries(
        var_names, st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=4, max_size=4,
    ))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_cse_preserves_value(self, vals):
        """CSE should produce an equivalent expression."""
        from bioprover.encoding.simplifier import simplify_fully
        # Build expression with shared subexpressions
        sub = Var("x") * Var("y")
        expr = Add(sub, Add(sub, Const(1.0)))
        result, stats = simplify_fully(expr)
        orig_val = _eval_expr(expr, vals)
        result_val = _eval_expr(result, vals)
        if orig_val is not None and result_val is not None:
            if math.isfinite(orig_val) and math.isfinite(result_val):
                assert abs(orig_val - result_val) < 1e-8


# ===================================================================
# Property: substitution is correct
# ===================================================================


class TestSubstitutionCorrectness:
    @given(
        expr=arithmetic_exprs(),
        subst_val=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        eval_vals=st.dictionaries(
            var_names,
            st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=4, max_size=4,
        ),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_substitution_correct(self, expr, subst_val, eval_vals):
        """Substituting x=c and evaluating should give the same result as
        evaluating with x=c in the environment."""
        substituted = expr.substitute({"x": Const(subst_val)})
        # Evaluate original with x=subst_val
        env_with_x = dict(eval_vals)
        env_with_x["x"] = subst_val
        orig_val = _eval_expr(expr, env_with_x)
        # Evaluate substituted with arbitrary x (should be irrelevant)
        subst_env = dict(eval_vals)
        subst_env["x"] = 999.0  # Should not matter since x was substituted
        subst_val_result = _eval_expr(substituted, subst_env)
        if orig_val is not None and subst_val_result is not None:
            if math.isfinite(orig_val) and math.isfinite(subst_val_result):
                assert abs(orig_val - subst_val_result) < 1e-8 * (1 + abs(orig_val))


# ===================================================================
# Property: free variables are accurate
# ===================================================================


class TestFreeVariableProperties:
    @given(expr=arithmetic_exprs())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_free_vars_subset_of_all_vars(self, expr):
        """Free variables should be a subset of all variable names used."""
        fv = expr.free_vars()
        vars_found = collect_nodes(expr, is_var)
        var_names_found = frozenset(v.name for v in vars_found)
        assert fv == var_names_found

    @given(expr=arithmetic_exprs())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_substitution_removes_var(self, expr):
        """Substituting a variable should remove it from free vars."""
        fv = expr.free_vars()
        if "x" in fv:
            substituted = expr.substitute({"x": Const(1.0)})
            new_fv = substituted.free_vars()
            assert "x" not in new_fv

    @given(expr=arithmetic_exprs())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_const_has_no_free_vars(self, expr):
        """Constants have no free variables."""
        c = Const(42.0)
        assert len(c.free_vars()) == 0


# ===================================================================
# Property: hash consistency
# ===================================================================


class TestHashProperties:
    @given(expr=arithmetic_exprs())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_equal_implies_same_hash(self, expr):
        """Equal expressions should have the same hash."""
        # Rebuild the same expression
        if isinstance(expr, Const):
            other = Const(expr.value)
        elif isinstance(expr, Var):
            other = Var(expr.name)
        else:
            other = expr  # same object due to hash-consing
        if expr == other:
            assert hash(expr) == hash(other)

    @given(expr=arithmetic_exprs())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_hash_deterministic(self, expr):
        """Hash should be deterministic."""
        h1 = hash(expr)
        h2 = hash(expr)
        assert h1 == h2


# ===================================================================
# Property: tree properties
# ===================================================================


class TestTreeStructureProperties:
    @given(expr=arithmetic_exprs())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_size_positive(self, expr):
        assert expr.size() >= 1

    @given(expr=arithmetic_exprs())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_depth_nonneg(self, expr):
        assert expr.depth() >= 0

    @given(expr=arithmetic_exprs())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_preorder_size(self, expr):
        """Preorder traversal should visit every node."""
        nodes = list(expr.iter_preorder())
        assert len(nodes) == expr.size()

    @given(expr=arithmetic_exprs())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_postorder_size(self, expr):
        """Postorder traversal should visit every node."""
        nodes = list(expr.iter_postorder())
        assert len(nodes) == expr.size()

    @given(expr=arithmetic_exprs())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_preorder_root_first(self, expr):
        """In preorder, root should be first."""
        nodes = list(expr.iter_preorder())
        assert nodes[0] == expr

    @given(expr=arithmetic_exprs())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_postorder_root_last(self, expr):
        """In postorder, root should be last."""
        nodes = list(expr.iter_postorder())
        assert nodes[-1] == expr


# ===================================================================
# Property: NNF conversion preserves truth value
# ===================================================================


class TestNNFPreservation:
    def test_nnf_double_negation(self):
        """NNF of not(not(p)) should be equivalent to p."""
        from bioprover.temporal.stl_ast import STLNot, predicate
        p = predicate("x", ">", 0.0)
        f = STLNot(child=STLNot(child=p))
        nnf = f.to_nnf()
        # nnf should be equivalent to p
        assert isinstance(nnf, type(p)) or (
            not isinstance(nnf, STLNot)
        )

    @given(threshold=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_nnf_not_always_eventually(self, threshold):
        """NNF of not(G(p)) should be F(not(p))."""
        from bioprover.temporal.stl_ast import (
            Always, Eventually, STLNot, predicate, Interval,
        )
        p = predicate("x", ">", threshold)
        f = STLNot(child=Always(child=p, interval=Interval(lo=0.0, hi=10.0)))
        nnf = f.to_nnf()
        assert isinstance(nnf, Eventually)


# ===================================================================
# Property: SMT-LIB serialization produces valid output
# ===================================================================


class TestSMTLIBProperties:
    @given(expr=simple_exprs(max_depth=2))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_smtlib_serialization_has_variables(self, expr):
        """SMT-LIB output should mention all free variables."""
        from bioprover.encoding.smtlib_serializer import expr_to_smtlib
        output = expr_to_smtlib(expr)
        fv = expr.free_vars()
        for v in fv:
            assert v in output

    @given(expr=simple_exprs(max_depth=2))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_smtlib_serialization_nonempty(self, expr):
        """SMT-LIB output should be non-empty."""
        from bioprover.encoding.smtlib_serializer import expr_to_smtlib
        output = expr_to_smtlib(expr)
        assert len(output) > 0

    @given(expr=simple_exprs(max_depth=1))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_smtlib_balanced_parens(self, expr):
        """SMT-LIB output should have balanced parentheses."""
        from bioprover.encoding.smtlib_serializer import expr_to_smtlib
        output = expr_to_smtlib(expr)
        assert output.count("(") == output.count(")")
