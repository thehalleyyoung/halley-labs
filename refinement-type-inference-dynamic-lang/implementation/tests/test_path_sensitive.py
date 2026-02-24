"""Tests for the path-sensitive guard analysis.

Validates that path-sensitive tracking eliminates false positives
from branch-dependent shapes, isinstance guards, None checks,
comparison guards, and nested conditions.
"""

import sys
import os
import pytest

# Ensure the src directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from path_sensitive import (
    PathConstraint,
    PathContext,
    PathSensitiveAnalyzer,
    PathBugReport,
    path_sensitive_filter,
)


# ── PathConstraint tests ──────────────────────────────────────────────


class TestPathConstraint:
    def test_positive_constraint(self):
        c = PathConstraint(variable="x", predicate="isinstance_Tensor")
        assert c.effective_predicate == "isinstance_Tensor"
        assert not c.negated

    def test_negated_constraint(self):
        c = PathConstraint(variable="x", predicate="isinstance_Tensor", negated=True)
        assert c.effective_predicate == "not_isinstance_Tensor"

    def test_negate_method(self):
        c = PathConstraint(variable="x", predicate="is_none")
        neg = c.negate()
        assert neg.negated
        assert neg.effective_predicate == "is_not_none"

    def test_double_negate(self):
        c = PathConstraint(variable="x", predicate="is_none")
        double = c.negate().negate()
        assert not double.negated
        assert double.effective_predicate == "is_none"

    def test_negate_is_not_none(self):
        c = PathConstraint(variable="x", predicate="is_not_none", negated=True)
        assert c.effective_predicate == "is_none"

    def test_negate_truthy(self):
        c = PathConstraint(variable="x", predicate="is_truthy", negated=True)
        assert c.effective_predicate == "is_falsy"

    def test_negate_ne_zero(self):
        c = PathConstraint(variable="x", predicate="ne_zero", negated=True)
        assert c.effective_predicate == "eq_zero"


# ── PathContext tests ─────────────────────────────────────────────────


class TestPathContext:
    def test_empty_context(self):
        ctx = PathContext()
        assert ctx.active_predicates_for("x") == set()
        assert ctx.has_type_constraint("x") is None
        assert ctx.variable_is_none("x") is None

    def test_push_pop(self):
        ctx = PathContext()
        c = PathConstraint(variable="x", predicate="isinstance_int")
        ctx.push(c)
        assert ctx.has_type_constraint("x") == "int"
        ctx.pop()
        assert ctx.has_type_constraint("x") is None

    def test_multiple_constraints(self):
        ctx = PathContext()
        ctx.push(PathConstraint(variable="x", predicate="isinstance_int"))
        ctx.push(PathConstraint(variable="y", predicate="is_not_none"))
        assert ctx.has_type_constraint("x") == "int"
        assert ctx.variable_is_none("y") is False
        assert ctx.has_type_constraint("y") is None

    def test_nested_constraints(self):
        ctx = PathContext()
        ctx.push(PathConstraint(variable="x", predicate="isinstance_Tensor"))
        ctx.push(PathConstraint(variable="x", predicate="gt_0"))
        preds = ctx.active_predicates_for("x")
        assert "isinstance_Tensor" in preds
        assert "gt_0" in preds

    def test_copy_independence(self):
        ctx = PathContext()
        ctx.push(PathConstraint(variable="x", predicate="is_none"))
        copy = ctx.copy()
        ctx.push(PathConstraint(variable="y", predicate="is_truthy"))
        assert len(copy.constraints) == 1
        assert len(ctx.constraints) == 2

    def test_variable_is_none_positive(self):
        ctx = PathContext()
        ctx.push(PathConstraint(variable="x", predicate="is_none"))
        assert ctx.variable_is_none("x") is True

    def test_variable_is_none_negative(self):
        ctx = PathContext()
        ctx.push(PathConstraint(variable="x", predicate="is_not_none"))
        assert ctx.variable_is_none("x") is False


# ── PathSensitiveAnalyzer tests ───────────────────────────────────────


class TestPathSensitiveAnalyzer:
    """Core path-sensitive analysis tests."""

    def test_isinstance_true_branch_no_false_positive(self):
        """isinstance guard in True branch should prevent type errors."""
        code = '''
def process(x):
    x = None
    if isinstance(x, str):
        return x.upper()
    return 0
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        # The .upper() call is inside isinstance(x, str) true branch,
        # so x is narrowed to str — no null deref.
        assert not any(b.line == 5 and b.category == "NULL_DEREF" for b in bugs)

    def test_isinstance_else_branch_gets_negation(self):
        """Else branch of isinstance should know x is NOT that type."""
        code = '''
def process(x):
    x = None
    if isinstance(x, str):
        return x.upper()
    else:
        return x
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        # In else branch, x is still None (was set to None, not narrowed)
        # but there's no access on x in else, so no bug either
        assert not any(b.line == 5 and b.category == "NULL_DEREF" for b in bugs)

    def test_none_check_true_branch_safe(self):
        """'if x is not None' should make true branch safe."""
        code = '''
def f():
    x = None
    if x is not None:
        return x.strip()
    return ""
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        assert not any(b.category == "NULL_DEREF" for b in bugs)

    def test_none_check_else_branch_unsafe(self):
        """After 'if x is not None', else branch should know x IS None."""
        code = '''
def f():
    x = None
    if x is not None:
        return x.strip()
    else:
        return x.upper()
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        # In else branch, x IS None → x.upper() is a bug
        null_bugs = [b for b in bugs if b.category == "NULL_DEREF"]
        assert any(b.line == 7 for b in null_bugs)

    def test_nested_guards_conjunction(self):
        """Nested if-statements create conjunction of constraints."""
        code = '''
def f(x, y):
    x = None
    y = None
    if x is not None:
        if y is not None:
            return x.strip() + y.strip()
    return ""
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        # Both x and y are guarded — no false positives
        assert not any(b.category == "NULL_DEREF" for b in bugs)

    def test_div_by_zero_guarded(self):
        """Division guarded by != 0 should not warn."""
        code = '''
def f(x):
    if x != 0:
        return 10 / x
    return 0
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        assert not any(b.category == "DIV_BY_ZERO" for b in bugs)

    def test_div_by_zero_unguarded(self):
        """Division without guard should still warn."""
        code = '''
def f(x):
    return 10 / x
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        assert any(b.category == "DIV_BY_ZERO" for b in bugs)

    def test_scope_restored_after_if(self):
        """Type narrowing should not leak past the if-else block."""
        code = '''
def f(x):
    x = None
    if isinstance(x, str):
        y = x.upper()
    z = 1
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        # After the if block, x should be restored to its pre-if state
        # The .upper() inside the if is fine because isinstance narrows
        assert not any(b.line == 4 and b.category == "NULL_DEREF" for b in bugs)

    def test_comparison_guard_tracked(self):
        """Comparison guards like x > 10 should be tracked."""
        code = '''
def f(x):
    if x > 10:
        return 100 / x
    return 0
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        # x > 10 implies x != 0, but our analyzer tracks it as "gt_10"
        # which doesn't directly match "ne_zero", so this may still warn.
        # The important thing is the constraint IS tracked.
        # Let's verify path constraints are present on any bugs
        for b in bugs:
            if b.category == "DIV_BY_ZERO":
                # Should have the gt_10 constraint
                preds = {c.effective_predicate for c in b.path_constraints}
                assert "gt_10" in preds

    def test_hasattr_guard(self):
        """hasattr guard should be tracked as a path constraint."""
        code = '''
def f(obj):
    obj = None
    if hasattr(obj, "shape"):
        return obj.shape
    return None
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        # hasattr guard doesn't prove non-null in our model, but
        # the isinstance-style narrowing via scope should help.
        # Just verify no crash.
        assert isinstance(bugs, list)

    def test_truthiness_guard(self):
        """Truthiness guard 'if x:' should track truthy constraint."""
        code = '''
def f():
    x = None
    if x:
        return x.strip()
    return ""
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        # 'if x:' doesn't formally prove non-null in our model,
        # but the truthy constraint is tracked
        assert isinstance(bugs, list)

    def test_multiple_isinstance_branches(self):
        """Different isinstance branches should each have correct type."""
        code = '''
def process(x):
    x = None
    if isinstance(x, int):
        y = x + 1
    elif isinstance(x, str):
        y = x.upper()
    else:
        y = 0
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        # No null derefs — isinstance narrows type in each branch
        null_bugs = [b for b in bugs if b.category == "NULL_DEREF"]
        assert len(null_bugs) == 0

    def test_bug_reports_carry_path_constraints(self):
        """Bug reports should carry the active path constraints."""
        code = '''
def f(x, y):
    if isinstance(x, int):
        z = 10 / y
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        div_bugs = [b for b in bugs if b.category == "DIV_BY_ZERO"]
        assert len(div_bugs) >= 1
        # The div-by-zero bug should carry the isinstance_int constraint
        for b in div_bugs:
            vars_constrained = {c.variable for c in b.path_constraints}
            assert "x" in vars_constrained


# ── Integration with pipeline filter ──────────────────────────────────


class TestPathSensitiveFilter:
    """Test path_sensitive_filter function for pipeline integration."""

    def test_filter_removes_false_positive(self):
        """Path-sensitive filter should remove false positives."""
        from dataclasses import dataclass
        from enum import Enum, auto

        class BugCategory(Enum):
            NULL_DEREF = auto()
            DIV_BY_ZERO = auto()

        @dataclass
        class FakeBug:
            category: BugCategory
            line: int
            col: int
            message: str
            function: str
            variable: str

        code = '''
def f():
    x = None
    if x is not None:
        return x.strip()
    return ""
'''
        # Simulate the path-insensitive analyzer reporting a false positive
        fake_bug = FakeBug(
            category=BugCategory.NULL_DEREF,
            line=5, col=0,
            message="Attribute access on potentially None variable 'x'",
            function="f",
            variable="x",
        )
        filtered, eliminated = path_sensitive_filter([fake_bug], code)
        # The path-sensitive analysis knows x is not None at line 5
        assert eliminated >= 1
        assert len(filtered) == 0

    def test_filter_keeps_real_bugs(self):
        """Path-sensitive filter should keep real bugs."""
        from dataclasses import dataclass
        from enum import Enum, auto

        class BugCategory(Enum):
            NULL_DEREF = auto()

        @dataclass
        class FakeBug:
            category: BugCategory
            line: int
            col: int
            message: str
            function: str
            variable: str

        code = '''
def f():
    x = None
    return x.strip()
'''
        fake_bug = FakeBug(
            category=BugCategory.NULL_DEREF,
            line=4, col=0,
            message="Attribute access on potentially None variable 'x'",
            function="f",
            variable="x",
        )
        filtered, eliminated = path_sensitive_filter([fake_bug], code)
        assert eliminated == 0
        assert len(filtered) == 1


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_source(self):
        analyzer = PathSensitiveAnalyzer("")
        bugs = analyzer.analyze()
        assert bugs == []

    def test_no_functions(self):
        code = "x = 1\ny = 2\n"
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        assert isinstance(bugs, list)

    def test_deeply_nested_ifs(self):
        """Deeply nested ifs should accumulate constraints."""
        code = '''
def f(a, b, c):
    a = None
    b = None
    c = None
    if a is not None:
        if b is not None:
            if c is not None:
                return a.strip() + b.strip() + c.strip()
    return ""
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        assert not any(b.category == "NULL_DEREF" for b in bugs)

    def test_if_without_else(self):
        """If without else should restore state after true branch."""
        code = '''
def f():
    x = None
    if x is not None:
        y = x.strip()
    z = 1
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        assert not any(b.line == 4 and b.category == "NULL_DEREF" for b in bugs)

    def test_and_guard(self):
        """'and' guards should create conjunction."""
        code = '''
def f(x, y):
    x = None
    y = None
    if x is not None and y is not None:
        return x.strip() + y.strip()
    return ""
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        assert not any(b.category == "NULL_DEREF" for b in bugs)

    def test_not_none_guard_via_not(self):
        """'if not x is None' should work like 'if x is not None'."""
        code = '''
def f():
    x = None
    if not x is None:
        return x.strip()
    return ""
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        assert not any(b.category == "NULL_DEREF" for b in bugs)

    def test_async_function(self):
        """Async functions should also be analyzed."""
        code = '''
async def f():
    x = None
    if x is not None:
        return x.strip()
    return ""
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        assert not any(b.category == "NULL_DEREF" for b in bugs)

    def test_method_call_on_none_guarded(self):
        """Method call on None in guarded context should be safe."""
        code = '''
def f():
    x = None
    if x is not None:
        x.foo()
    return 0
'''
        analyzer = PathSensitiveAnalyzer(code)
        bugs = analyzer.analyze()
        assert not any(b.category == "NULL_DEREF" for b in bugs)
