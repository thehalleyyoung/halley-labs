"""
Comprehensive tests for the guard-harvesting analyzer.

Tests all guard patterns, edge cases, and the improvements made to address
reviewer critiques (truthiness semantics, OOB detection, type error detection).
"""

import pytest
from src.real_analyzer import (
    analyze_source, BugCategory, NullState, TypeTagSet, Interval,
    VarState, FlowSensitiveAnalyzer, AbstractEnv,
)


# ── Helpers ────────────────────────────────────────────────────────────

def bugs_in(source: str):
    """Return list of bugs found in source code."""
    result = analyze_source(source)
    all_bugs = []
    for fr in result.function_results:
        all_bugs.extend(fr.bugs)
    return all_bugs


def has_bug(source: str, category: BugCategory) -> bool:
    """Check if a specific bug category is found."""
    return any(b.category == category for b in bugs_in(source))


def no_bugs(source: str) -> bool:
    """Check that no bugs are found."""
    return len(bugs_in(source)) == 0


# ── Nullity guards ────────────────────────────────────────────────────

class TestNullityGuards:
    def test_is_none_guard(self):
        code = '''
def f(x):
    x = None
    if x is None:
        return 0
    return x.strip()
'''
        assert no_bugs(code)

    def test_is_not_none_guard(self):
        code = '''
def f():
    x = None
    if x is not None:
        return x.strip()
    return ""
'''
        assert no_bugs(code)

    def test_none_deref_detected(self):
        code = '''
def f():
    x = None
    return x.strip()
'''
        assert has_bug(code, BugCategory.NULL_DEREF)

    def test_dict_get_none_evidence(self):
        code = '''
def f(d):
    val = d.get("key")
    return val.upper()
'''
        assert has_bug(code, BugCategory.UNGUARDED_OPTIONAL)

    def test_dict_get_with_default(self):
        code = '''
def f(d):
    val = d.get("key", "default")
    return val.upper()
'''
        assert no_bugs(code)

    def test_regex_search_none(self):
        code = '''
import re
def f(text):
    m = re.search(r"pattern", text)
    return m.group(1)
'''
        assert has_bug(code, BugCategory.UNGUARDED_OPTIONAL)

    def test_regex_search_guarded(self):
        code = '''
import re
def f(text):
    m = re.search(r"pattern", text)
    if m:
        return m.group(1)
    return None
'''
        assert no_bugs(code)

    def test_early_return_narrowing(self):
        code = '''
def f():
    val = None
    if val is None:
        return ""
    return val.strip()
'''
        assert no_bugs(code)

    def test_or_default_pattern(self):
        code = '''
def f(d):
    val = d.get("key") or "fallback"
    return val.upper()
'''
        assert no_bugs(code)


# ── Division by zero ──────────────────────────────────────────────────

class TestDivByZero:
    def test_literal_zero(self):
        code = '''
def f():
    return 1 / 0
'''
        assert has_bug(code, BugCategory.DIV_BY_ZERO)

    def test_unguarded_divisor(self):
        code = '''
def f(values):
    return sum(values) / len(values)
'''
        assert has_bug(code, BugCategory.DIV_BY_ZERO)

    def test_guarded_by_truthiness(self):
        code = '''
def f(values):
    if not values:
        return 0.0
    return sum(values) / len(values)
'''
        assert no_bugs(code)

    def test_guarded_by_len_check(self):
        code = '''
def f(values):
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)
'''
        assert no_bugs(code)

    def test_ne_zero_guard(self):
        code = '''
def f(x, y):
    if y != 0:
        return x / y
    return 0
'''
        assert no_bugs(code)

    def test_gt_zero_guard(self):
        code = '''
def f(x, y):
    if y > 0:
        return x / y
    return 0
'''
        assert no_bugs(code)

    def test_abs_plus_constant(self):
        code = '''
def f(x):
    return x / (abs(x) + 1)
'''
        assert no_bugs(code)


# ── Index out of bounds ──────────────────────────────────────────────

class TestIndexOOB:
    def test_known_length_oob(self):
        code = '''
def f():
    items = [1, 2, 3]
    return items[5]
'''
        assert has_bug(code, BugCategory.INDEX_OUT_OF_BOUNDS)

    def test_known_length_safe(self):
        code = '''
def f():
    items = [1, 2, 3]
    return items[0]
'''
        assert no_bugs(code)

    def test_parameter_no_guard(self):
        code = '''
def f(items):
    return items[0]
'''
        assert has_bug(code, BugCategory.INDEX_OUT_OF_BOUNDS)

    def test_parameter_guarded_by_truthiness(self):
        code = '''
def f(items):
    if items:
        return items[0]
    return None
'''
        assert no_bugs(code)

    def test_parameter_guarded_by_len(self):
        code = '''
def f(items):
    if len(items) > 0:
        return items[0]
    return None
'''
        assert no_bugs(code)

    def test_parameter_guarded_by_early_return(self):
        code = '''
def f(items):
    if not items:
        return None
    return items[0]
'''
        assert no_bugs(code)

    def test_negative_index_known_length(self):
        code = '''
def f():
    items = [1, 2]
    return items[-3]
'''
        assert has_bug(code, BugCategory.INDEX_OUT_OF_BOUNDS)


# ── Type errors ──────────────────────────────────────────────────────

class TestTypeErrors:
    def test_none_plus_int(self):
        code = '''
def f():
    x = None
    y = 5
    return x + y
'''
        assert has_bug(code, BugCategory.TYPE_ERROR)

    def test_str_plus_int(self):
        code = '''
def f():
    x = "hello"
    y = 5
    return x + y
'''
        assert has_bug(code, BugCategory.TYPE_ERROR)

    def test_str_minus_str(self):
        code = '''
def f():
    a = "hello"
    b = "world"
    return a - b
'''
        assert has_bug(code, BugCategory.TYPE_ERROR)

    def test_int_not_subscriptable(self):
        code = '''
def f():
    x = 42
    return x[0]
'''
        assert has_bug(code, BugCategory.TYPE_ERROR)

    def test_list_plus_int(self):
        code = '''
def f():
    x = [1, 2, 3]
    y = 4
    return x + y
'''
        assert has_bug(code, BugCategory.TYPE_ERROR)

    def test_isinstance_guarded_addition(self):
        code = '''
def f(a, b):
    if isinstance(a, str) and isinstance(b, str):
        return a + b
    return None
'''
        assert no_bugs(code)


# ── Truthiness semantics ─────────────────────────────────────────────

class TestTruthiness:
    def test_truthy_narrows_null(self):
        """if x: should narrow null to DefNotNull."""
        code = '''
def f():
    x = None
    if x:
        return x.strip()
    return ""
'''
        assert no_bugs(code)

    def test_truthy_removes_nonetype_tag(self):
        """if x: should remove NoneType from type tags."""
        code = '''
def f():
    x = None
    if x:
        return len(x)  # x cannot be None here
    return 0
'''
        assert no_bugs(code)

    def test_falsy_does_not_narrow_to_none(self):
        """False branch of truthiness should be MayNull, not DefNull."""
        code = '''
def f(x):
    if x:
        pass
    else:
        # x could be 0, "", [], False, or None — not definitely None
        pass
'''
        assert no_bugs(code)


# ── De Morgan narrowing ──────────────────────────────────────────────

class TestDeMorgan:
    def test_or_early_return(self):
        code = '''
def f():
    x = None
    y = None
    if x is None or y is None:
        return 0
    return x.strip() + y.strip()
'''
        assert no_bugs(code)

    def test_and_guard(self):
        code = '''
def f():
    x = None
    y = None
    if x is not None and y is not None:
        return x.strip() + y.strip()
    return ""
'''
        assert no_bugs(code)


# ── Tuple return analysis ────────────────────────────────────────────

class TestTupleReturn:
    def test_tuple_elements_checked(self):
        code = '''
def f():
    x = None
    y = None
    return x.strip(), y.strip()
'''
        assert has_bug(code, BugCategory.NULL_DEREF)

    def test_guarded_tuple_return(self):
        code = '''
def f():
    x = "hello"
    y = "world"
    return x.strip(), y.strip()
'''
        assert no_bugs(code)


# ── Domain operations ────────────────────────────────────────────────

class TestDomainOps:
    def test_null_join(self):
        assert NullState.DEFINITELY_NULL.join(NullState.DEFINITELY_NOT_NULL) == NullState.MAYBE_NULL
        assert NullState.BOTTOM.join(NullState.DEFINITELY_NULL) == NullState.DEFINITELY_NULL
        assert NullState.DEFINITELY_NULL.join(NullState.DEFINITELY_NULL) == NullState.DEFINITELY_NULL

    def test_null_meet(self):
        assert NullState.DEFINITELY_NULL.meet(NullState.DEFINITELY_NOT_NULL) == NullState.BOTTOM
        assert NullState.MAYBE_NULL.meet(NullState.DEFINITELY_NULL) == NullState.DEFINITELY_NULL

    def test_interval_contains_zero(self):
        assert Interval(lo=-1, hi=1).contains_zero()
        assert not Interval(lo=1, hi=5).contains_zero()
        assert Interval(lo=None, hi=None).contains_zero()

    def test_interval_join(self):
        i1 = Interval(lo=1, hi=5)
        i2 = Interval(lo=3, hi=10)
        j = i1.join(i2)
        assert j.lo == 1 and j.hi == 10

    def test_interval_meet(self):
        i1 = Interval(lo=1, hi=5)
        i2 = Interval(lo=3, hi=10)
        m = i1.meet(i2)
        assert m.lo == 3 and m.hi == 5

    def test_typetag_join(self):
        t1 = TypeTagSet(frozenset({"int"}))
        t2 = TypeTagSet(frozenset({"str"}))
        j = t1.join(t2)
        assert j.tags == frozenset({"int", "str"})

    def test_typetag_meet(self):
        t1 = TypeTagSet(frozenset({"int", "str"}))
        t2 = TypeTagSet(frozenset({"str"}))
        m = t1.meet(t2)
        assert m.tags == frozenset({"str"})


# ── Interprocedural analysis tests ─────────────────────────────────────

class TestInterprocedural:
    """Test cross-function analysis via function summaries."""

    def test_none_returning_callee_detected(self):
        code = '''
def fetch(key):
    if key not in _cache:
        return None
    return _cache[key]

def process():
    val = fetch("x")
    val.strip()
'''
        from src.api import analyze
        result = analyze(code)
        null_bugs = result.by_category(result.bugs[0].category) if result.bugs else []
        assert result.bug_count >= 1, "Should detect null deref from callee returning None"

    def test_non_none_callee_no_fp(self):
        code = '''
def make_data(x):
    return {"value": x}

def process():
    d = make_data(42)
    print(d["value"])
'''
        from src.api import analyze
        result = analyze(code)
        assert result.bug_count == 0, "Should not FP when callee never returns None"

    def test_guarded_callee_return_no_fp(self):
        code = '''
def maybe_get(key):
    if not key:
        return None
    return {"key": key}

def use_it():
    result = maybe_get("hello")
    if result is not None:
        print(result["key"])
'''
        from src.api import analyze
        result = analyze(code)
        assert result.bug_count == 0, "Should not FP when caller guards against None"

    def test_summary_inference_bare_return(self):
        from src.real_analyzer import infer_function_summary, NullState
        import ast
        code = '''
def f(x):
    if x:
        return x
    return
'''
        tree = ast.parse(code)
        func = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][0]
        summary = infer_function_summary(func)
        assert summary.can_return_none
        assert summary.return_null_state == NullState.MAYBE_NULL

    def test_summary_inference_always_returns_value(self):
        from src.real_analyzer import infer_function_summary, NullState
        import ast
        code = '''
def f(x):
    if x > 0:
        return x + 1
    else:
        return 0
'''
        tree = ast.parse(code)
        func = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][0]
        summary = infer_function_summary(func)
        assert not summary.can_return_none
        assert summary.return_null_state == NullState.DEFINITELY_NOT_NULL
