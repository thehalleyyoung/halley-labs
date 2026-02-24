"""
Real tests for the refinement type inference pipeline.

Tests actual modules: guard extraction, SMT solving, CEGAR loop,
bug detection, and the integrated pipeline. No shadow implementations.
"""

import ast
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

z3 = pytest.importorskip("z3", reason="z3-solver not installed")

from src.pipeline import (
    analyze_python_source,
    analyze_python_file,
    run_cegar,
    PythonAnalyzer,
    BugCategory,
    BugReport,
)
from src.guard_extractor import (
    extract_guards,
    GuardPattern,
    PredicateKind,
)
from src.smt.solver import (
    Z3Solver,
    SatResult,
    Comparison,
    ComparisonOp,
    IsInstance,
    IsNone,
    IsTruthy,
    And,
    Not,
    BoolLit,
    Var,
    Const,
)


# ── Guard extraction tests ──────────────────────────────────────────────

class TestGuardExtraction:
    """Test guard extraction from real Python ASTs."""

    def test_isinstance_guard(self):
        code = "def f(x):\n    if isinstance(x, int): pass"
        guards = extract_guards(code)
        assert len(guards) >= 1
        type_guards = [g for g in guards if g.pattern == GuardPattern.TypeTag]
        assert len(type_guards) >= 1
        assert type_guards[0].predicate.target_variable == "x"
        assert "int" in type_guards[0].predicate.type_names

    def test_none_check_guard(self):
        code = "def f(x):\n    if x is None: pass"
        guards = extract_guards(code)
        null_guards = [g for g in guards if g.pattern == GuardPattern.Nullity]
        assert len(null_guards) >= 1

    def test_comparison_guard(self):
        code = "def f(x):\n    if x > 0: pass"
        guards = extract_guards(code)
        comp_guards = [g for g in guards if g.pattern == GuardPattern.Comparison]
        assert len(comp_guards) >= 1

    def test_combined_guards(self):
        code = """
def f(x, y):
    if isinstance(x, int) and x > 0:
        if y is not None:
            return x + y
"""
        guards = extract_guards(code)
        assert len(guards) >= 2  # isinstance + comparison + nullity

    def test_no_guards(self):
        code = "def f(x):\n    return x + 1"
        guards = extract_guards(code)
        # No if-guards in this code
        assert isinstance(guards, list)

    def test_nested_isinstance(self):
        code = """
def process(data):
    if isinstance(data, dict):
        if isinstance(data.get('key'), str):
            return data['key'].upper()
"""
        guards = extract_guards(code)
        type_guards = [g for g in guards if g.pattern == GuardPattern.TypeTag]
        assert len(type_guards) >= 1

    def test_multiple_functions(self):
        code = """
def f(x):
    if isinstance(x, int): pass

def g(y):
    if y is None: pass
"""
        guards = extract_guards(code)
        assert len(guards) >= 2


# ── SMT solver tests ────────────────────────────────────────────────────

class TestSMTSolver:
    """Test real Z3 solver integration."""

    def test_sat_simple(self):
        solver = Z3Solver(timeout_ms=5000)
        # x > 0 is satisfiable
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))
        assert solver.check_sat() == SatResult.SAT

    def test_unsat_contradiction(self):
        solver = Z3Solver(timeout_ms=5000)
        # x > 0 AND x < 0 is unsatisfiable
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))
        solver.assert_formula(Comparison(ComparisonOp.LT, Var("x"), Const(0)))
        assert solver.check_sat() == SatResult.UNSAT

    def test_isinstance_encoding(self):
        solver = Z3Solver(timeout_ms=5000)
        # isinstance(x, int) is satisfiable
        solver.assert_formula(IsInstance("x", "int"))
        assert solver.check_sat() == SatResult.SAT

    def test_isinstance_contradiction(self):
        solver = Z3Solver(timeout_ms=5000)
        # isinstance(x, int) AND isinstance(x, str) => unsat (different tags)
        solver.assert_formula(IsInstance("x", "int"))
        solver.assert_formula(IsInstance("x", "str"))
        result = solver.check_sat()
        assert result == SatResult.UNSAT

    def test_none_encoding(self):
        solver = Z3Solver(timeout_ms=5000)
        # is_none(x) AND NOT is_none(x) is unsat
        solver.assert_formula(IsNone("x"))
        solver.assert_formula(Not(IsNone("x")))
        assert solver.check_sat() == SatResult.UNSAT

    def test_model_extraction(self):
        solver = Z3Solver(timeout_ms=5000)
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(42)))
        result = solver.check_sat()
        assert result == SatResult.SAT
        model = solver.get_model()
        assert model is not None
        assert model.get_int("x") == 42

    def test_push_pop(self):
        solver = Z3Solver(timeout_ms=5000)
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))
        solver.push()
        solver.assert_formula(Comparison(ComparisonOp.LT, Var("x"), Const(0)))
        assert solver.check_sat() == SatResult.UNSAT
        solver.pop()
        assert solver.check_sat() == SatResult.SAT

    def test_and_predicate(self):
        solver = Z3Solver(timeout_ms=5000)
        pred = And((
            Comparison(ComparisonOp.GT, Var("x"), Const(0)),
            Comparison(ComparisonOp.LT, Var("x"), Const(10)),
        ))
        solver.assert_formula(pred)
        assert solver.check_sat() == SatResult.SAT
        model = solver.get_model()
        assert model is not None
        x_val = model.get_int("x")
        assert 0 < x_val < 10

    def test_interpolant(self):
        solver = Z3Solver(timeout_ms=5000)
        a = And((
            Comparison(ComparisonOp.GT, Var("x"), Const(5)),
            Comparison(ComparisonOp.LT, Var("y"), Const(3)),
        ))
        b = Comparison(ComparisonOp.LT, Var("x"), Const(0))
        interpolant = solver.compute_interpolant(a, b)
        # A ∧ B should be unsat (x > 5 and x < 0)
        assert interpolant is not None


# ── Bug detection tests ─────────────────────────────────────────────────

class TestBugDetection:
    """Test bug detection on real Python code."""

    def test_null_deref(self):
        code = """
def f():
    x = None
    x.method()
"""
        analyzer = PythonAnalyzer(code)
        bugs = analyzer.analyze()
        null_bugs = [b for b in bugs if b.category == BugCategory.NULL_DEREF]
        assert len(null_bugs) >= 1

    def test_div_by_zero_literal(self):
        code = """
def f():
    return 1 / 0
"""
        analyzer = PythonAnalyzer(code)
        bugs = analyzer.analyze()
        div_bugs = [b for b in bugs if b.category == BugCategory.DIV_BY_ZERO]
        assert len(div_bugs) >= 1

    def test_index_out_of_bounds(self):
        code = """
def f():
    a = [1, 2, 3]
    return a[10]
"""
        analyzer = PythonAnalyzer(code)
        bugs = analyzer.analyze()
        oob_bugs = [b for b in bugs if b.category == BugCategory.INDEX_OUT_OF_BOUNDS]
        assert len(oob_bugs) >= 1

    def test_guarded_none_no_bug(self):
        code = """
def f(x):
    if x is not None:
        x.method()
"""
        analyzer = PythonAnalyzer(code)
        bugs = analyzer.analyze()
        null_bugs = [b for b in bugs if b.category == BugCategory.NULL_DEREF]
        assert len(null_bugs) == 0

    def test_safe_code_no_bugs(self):
        code = """
def f(x, y):
    if isinstance(x, int) and isinstance(y, int):
        if y != 0:
            return x / y
    return 0
"""
        analyzer = PythonAnalyzer(code)
        bugs = analyzer.analyze()
        assert len(bugs) == 0

    def test_multiple_bugs(self):
        code = """
def f():
    x = None
    x.attr
    a = [1]
    a[5]
    result = 10 / 0
"""
        analyzer = PythonAnalyzer(code)
        bugs = analyzer.analyze()
        assert len(bugs) >= 3
        categories = {b.category for b in bugs}
        assert BugCategory.NULL_DEREF in categories
        assert BugCategory.INDEX_OUT_OF_BOUNDS in categories
        assert BugCategory.DIV_BY_ZERO in categories


# ── CEGAR loop tests ────────────────────────────────────────────────────

class TestCEGAR:
    """Test the real CEGAR loop with Z3."""

    def test_cegar_with_guards(self):
        code = "def f(x):\n    if isinstance(x, int): pass"
        guards = extract_guards(code)
        state = run_cegar(code, guards)
        assert state.converged
        assert state.iterations >= 1
        assert len(state.predicates) >= 1

    def test_cegar_without_guards(self):
        code = "def f(x):\n    return x + 1"
        guards = extract_guards(code)
        state = run_cegar(code, guards)
        assert state.converged

    def test_cegar_convergence(self):
        code = """
def f(x, y, z):
    if isinstance(x, int) and isinstance(y, str):
        if x > 0 and y is not None:
            return str(x) + y
"""
        guards = extract_guards(code)
        state = run_cegar(code, guards, max_iterations=30)
        assert state.converged
        assert state.iterations <= 30

    def test_cegar_predicates_grow(self):
        code = """
def f(x):
    if isinstance(x, int):
        if x > 0:
            if x < 100:
                return x
"""
        guards = extract_guards(code)
        state = run_cegar(code, guards)
        # Guards should produce seed predicates
        assert len(state.predicates) >= 2


# ── Integration tests ───────────────────────────────────────────────────

class TestIntegration:
    """End-to-end integration tests using real analysis."""

    def test_analyze_simple_code(self):
        code = """
def greet(name):
    if isinstance(name, str):
        return "Hello, " + name
    return "Hello!"
"""
        result = analyze_python_source(code)
        assert result.functions_analyzed == 1
        assert result.total_guards >= 1
        assert result.total_bugs == 0

    def test_analyze_buggy_code(self):
        code = """
def process(data):
    x = None
    return x.strip()
"""
        result = analyze_python_source(code)
        assert result.total_bugs >= 1

    def test_analyze_multiple_functions(self):
        code = """
def f(x):
    if isinstance(x, int): return x * 2

def g(y):
    if y is not None: return y.upper()

def h(z):
    return z + 1
"""
        result = analyze_python_source(code)
        assert result.functions_analyzed == 3
        assert result.total_guards >= 2

    def test_analysis_result_structure(self):
        code = "def f(x): return x"
        result = analyze_python_source(code)
        assert hasattr(result, 'file_path')
        assert hasattr(result, 'functions_analyzed')
        assert hasattr(result, 'total_guards')
        assert hasattr(result, 'total_bugs')
        assert hasattr(result, 'analysis_time_ms')
        assert result.analysis_time_ms >= 0

    def test_empty_source(self):
        result = analyze_python_source("")
        assert result.functions_analyzed == 0
        assert result.total_bugs == 0

    def test_syntax_error(self):
        result = analyze_python_source("def f(: broken")
        assert len(result.errors) >= 1

    def test_complex_real_world_pattern(self):
        code = """
def safe_get(mapping, key, default=None):
    if not isinstance(mapping, dict):
        return default
    if key not in mapping:
        return default
    value = mapping[key]
    if value is None:
        return default
    return value

def process_items(items):
    if items is None:
        return []
    if not isinstance(items, list):
        items = [items]
    results = []
    for item in items:
        if isinstance(item, str):
            results.append(item.upper())
        elif isinstance(item, int):
            results.append(item * 2)
    return results
"""
        result = analyze_python_source(code)
        assert result.functions_analyzed == 2
        assert result.total_guards >= 4
        assert result.total_bugs == 0  # all accesses are guarded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
