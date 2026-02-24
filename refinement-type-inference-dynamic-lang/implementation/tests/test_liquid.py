"""
Tests for the liquid type inference engine (src/liquid.py).

Covers:
  1. Guard harvesting
  2. Assert harvesting
  3. Default value harvesting
  4. Exception harvesting
  5. Null dereference detection
  6. Division by zero detection
  7. Interprocedural contract propagation
  8. CEGAR refinement
  9. Multi-function module analysis
"""

import ast
import pytest

from src.liquid import (
    PredicateHarvester,
    HarvestSource,
    HarvestedPred,
    LiquidTypeInferencer,
    LiquidAnalysisResult,
    LiquidBug,
    LiquidBugKind,
    FunctionContract,
    LiquidConstraint,
    ConstraintKind,
    ConstraintGenerator,
    CEGARRefinement,
    InterproceduralLiquidAnalyzer,
    PredicateTemplateLibrary,
    LiquidTypeReporter,
    analyze_liquid,
    harvest_predicates,
    infer_contract,
)
from src._experimental.refinement_lattice import (
    Pred,
    PredOp,
    RefType,
    BaseTypeR,
    BaseTypeKind,
    INT_TYPE,
    FLOAT_TYPE,
    STR_TYPE,
    BOOL_TYPE,
    NONE_TYPE,
    ANY_TYPE,
    RefinementLattice,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_func(source: str, name: str = None) -> ast.FunctionDef:
    """Parse source and return the first (or named) FunctionDef."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if name is None or node.name == name:
                return node
    raise ValueError(f"No function {name!r} found")


def _harvest(source: str) -> list:
    """Harvest predicates from all functions in source."""
    return harvest_predicates(source)


def _has_pred_op(preds, op: PredOp) -> bool:
    """Check if any harvested predicate has the given operator."""
    return any(hp.pred.op == op for hp in preds)


def _has_source(preds, source: HarvestSource) -> bool:
    """Check if any harvested predicate has the given source."""
    return any(hp.source == source for hp in preds)


def _bugs_of_kind(result: LiquidAnalysisResult, kind: LiquidBugKind) -> list:
    return [b for b in result.bugs if b.kind == kind]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Guard Harvesting
# ═══════════════════════════════════════════════════════════════════════════

class TestGuardHarvesting:
    def test_is_none_guard(self):
        code = '''
def f(x):
    if x is None:
        return 0
    return x.strip()
'''
        preds = _harvest(code)
        assert _has_pred_op(preds, PredOp.IS_NONE)

    def test_is_not_none_guard(self):
        code = '''
def f(x):
    if x is not None:
        return x.strip()
    return ""
'''
        preds = _harvest(code)
        assert _has_pred_op(preds, PredOp.IS_NOT_NONE)

    def test_isinstance_guard(self):
        code = '''
def f(x):
    if isinstance(x, int):
        return x + 1
    return 0
'''
        preds = _harvest(code)
        assert _has_pred_op(preds, PredOp.ISINSTANCE)

    def test_comparison_guard(self):
        code = '''
def f(x):
    if x > 0:
        return x
    return 0
'''
        preds = _harvest(code)
        assert _has_pred_op(preds, PredOp.GT)

    def test_truthiness_guard(self):
        code = '''
def f(x):
    if x:
        return x.strip()
    return ""
'''
        preds = _harvest(code)
        assert _has_pred_op(preds, PredOp.TRUTHY)

    def test_hasattr_guard(self):
        code = '''
def f(x):
    if hasattr(x, 'name'):
        return x.name
    return None
'''
        preds = _harvest(code)
        assert _has_pred_op(preds, PredOp.HASATTR)

    def test_numeric_comparison_le(self):
        code = '''
def f(x):
    if x <= 10:
        return x
    return 10
'''
        preds = _harvest(code)
        assert _has_pred_op(preds, PredOp.LE)

    def test_reversed_comparison(self):
        code = '''
def f(x):
    if 0 < x:
        return x
    return 0
'''
        preds = _harvest(code)
        assert _has_pred_op(preds, PredOp.GT)

    def test_guard_source_tag(self):
        code = '''
def f(x):
    if x is not None:
        pass
'''
        preds = _harvest(code)
        assert _has_source(preds, HarvestSource.GUARD)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Assert Harvesting
# ═══════════════════════════════════════════════════════════════════════════

class TestAssertHarvesting:
    def test_assert_gt(self):
        code = '''
def f(x):
    assert x > 0
    return 1 / x
'''
        preds = _harvest(code)
        assert_preds = [hp for hp in preds if hp.source == HarvestSource.ASSERT]
        assert len(assert_preds) > 0
        assert any(hp.pred.op == PredOp.GT for hp in assert_preds)

    def test_assert_is_not_none(self):
        code = '''
def f(x):
    assert x is not None
    return x.strip()
'''
        preds = _harvest(code)
        assert_preds = [hp for hp in preds if hp.source == HarvestSource.ASSERT]
        assert any(hp.pred.op == PredOp.IS_NOT_NONE for hp in assert_preds)

    def test_assert_isinstance(self):
        code = '''
def f(x):
    assert isinstance(x, str)
    return x.upper()
'''
        preds = _harvest(code)
        assert_preds = [hp for hp in preds if hp.source == HarvestSource.ASSERT]
        assert any(hp.pred.op == PredOp.ISINSTANCE for hp in assert_preds)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Default Value Harvesting
# ═══════════════════════════════════════════════════════════════════════════

class TestDefaultHarvesting:
    def test_dict_get_with_default(self):
        """dict.get(key, default) does NOT guarantee not-None because
        d[key] could be None. This was an unsoundness bug — now fixed."""
        code = '''
def f(d):
    x = d.get("key", 0)
    return x + 1
'''
        preds = _harvest(code)
        default_preds = [hp for hp in preds if hp.source == HarvestSource.DEFAULT]
        # Should NOT infer is_not_none — that would be unsound
        assert len(default_preds) == 0

    def test_or_default(self):
        code = '''
def f(x):
    y = x or "default"
    return y.upper()
'''
        preds = _harvest(code)
        default_preds = [hp for hp in preds if hp.source == HarvestSource.DEFAULT]
        assert len(default_preds) > 0

    def test_ternary_default(self):
        code = '''
def f(x):
    y = x if x is not None else "fallback"
    return y.upper()
'''
        preds = _harvest(code)
        default_preds = [hp for hp in preds if hp.source == HarvestSource.DEFAULT]
        assert len(default_preds) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Exception Harvesting
# ═══════════════════════════════════════════════════════════════════════════

class TestExceptionHarvesting:
    def test_attribute_error_refines(self):
        code = '''
def f(x):
    try:
        x.method()
    except AttributeError:
        return None
    return x
'''
        preds = _harvest(code)
        exc_preds = [hp for hp in preds if hp.source == HarvestSource.EXCEPTION]
        assert len(exc_preds) > 0
        assert any(hp.pred.op == PredOp.HASATTR for hp in exc_preds)

    def test_value_error_refines(self):
        code = '''
def f(x):
    try:
        n = int(x)
    except ValueError:
        return -1
    return n
'''
        preds = _harvest(code)
        exc_preds = [hp for hp in preds if hp.source == HarvestSource.EXCEPTION]
        assert len(exc_preds) > 0
        assert any(hp.pred.op == PredOp.ISINSTANCE for hp in exc_preds)

    def test_key_error_refines(self):
        code = '''
def f(d, k):
    try:
        v = d[k]
    except KeyError:
        return None
    return v
'''
        preds = _harvest(code)
        exc_preds = [hp for hp in preds if hp.source == HarvestSource.EXCEPTION]
        assert len(exc_preds) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 5. Null Dereference Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestNullDerefDetection:
    def test_unguarded_null_deref(self):
        code = '''
def f():
    x = None
    return x.strip()
'''
        result = analyze_liquid(code)
        null_bugs = _bugs_of_kind(result, LiquidBugKind.NULL_DEREF)
        assert len(null_bugs) > 0

    def test_guarded_no_bug(self):
        code = '''
def f(x):
    if x is not None:
        return x.strip()
    return ""
'''
        result = analyze_liquid(code)
        null_bugs = _bugs_of_kind(result, LiquidBugKind.NULL_DEREF)
        assert len(null_bugs) == 0

    def test_none_assignment_then_deref(self):
        code = '''
def f():
    x = None
    y = x.upper()
    return y
'''
        result = analyze_liquid(code)
        null_bugs = _bugs_of_kind(result, LiquidBugKind.NULL_DEREF)
        assert len(null_bugs) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 6. Division by Zero Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestDivByZeroDetection:
    def test_unguarded_div_by_zero(self):
        code = '''
def f(x, y):
    return x / y
'''
        result = analyze_liquid(code)
        div_bugs = _bugs_of_kind(result, LiquidBugKind.DIV_BY_ZERO)
        assert len(div_bugs) > 0

    def test_guarded_div_no_bug(self):
        code = '''
def f(x, y):
    if y != 0:
        return x / y
    return 0
'''
        result = analyze_liquid(code)
        div_bugs = _bugs_of_kind(result, LiquidBugKind.DIV_BY_ZERO)
        assert len(div_bugs) == 0

    def test_assert_guards_div(self):
        code = '''
def f(x, y):
    assert y != 0
    return x / y
'''
        result = analyze_liquid(code)
        div_bugs = _bugs_of_kind(result, LiquidBugKind.DIV_BY_ZERO)
        assert len(div_bugs) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. Interprocedural Contract Propagation
# ═══════════════════════════════════════════════════════════════════════════

class TestInterproceduralPropagation:
    def test_contract_inferred(self):
        code = '''
def div(x, y):
    if y != 0:
        return x / y
    return 0

def caller():
    return div(10, 2)
'''
        result = analyze_liquid(code)
        assert "div" in result.contracts
        contract = result.contracts["div"]
        assert contract.name == "div"

    def test_return_type_inferred(self):
        code = '''
def safe_get(d, k):
    return d.get(k, 0)
'''
        result = analyze_liquid(code)
        assert "safe_get" in result.contracts

    def test_interprocedural_analyzer(self):
        code = '''
def helper(x):
    if x is None:
        return ""
    return x.upper()

def main(val):
    result = helper(val)
    return result.lower()
'''
        analyzer = InterproceduralLiquidAnalyzer()
        result = analyzer.analyze(code)
        assert "helper" in result.contracts
        assert "main" in result.contracts


# ═══════════════════════════════════════════════════════════════════════════
# 8. CEGAR Refinement
# ═══════════════════════════════════════════════════════════════════════════

class TestCEGARRefinement:
    def test_cegar_refines_predicate(self):
        """A case where the assert provides the predicate, and CEGAR uses it."""
        code = '''
def f(x):
    assert x > 0
    return 10 / x
'''
        result = analyze_liquid(code)
        # The assert should have provided enough predicates
        div_bugs = _bugs_of_kind(result, LiquidBugKind.DIV_BY_ZERO)
        assert len(div_bugs) == 0

    def test_cegar_iterations_bounded(self):
        code = '''
def f(x):
    return x.strip()
'''
        result = analyze_liquid(code)
        assert result.cegar_iterations <= 10 * len(result.contracts)

    def test_cegar_with_none_check(self):
        code = '''
def f(x):
    if x is not None:
        return x.strip()
    return ""
'''
        result = analyze_liquid(code)
        null_bugs = _bugs_of_kind(result, LiquidBugKind.NULL_DEREF)
        assert len(null_bugs) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 9. Multi-Function Module Analysis
# ═══════════════════════════════════════════════════════════════════════════

class TestModuleAnalysis:
    def test_multiple_functions(self):
        code = '''
def add(a, b):
    return a + b

def div(a, b):
    return a / b

def safe_div(a, b):
    if b != 0:
        return a / b
    return 0
'''
        result = analyze_liquid(code)
        assert len(result.contracts) == 3
        assert "add" in result.contracts
        assert "div" in result.contracts
        assert "safe_div" in result.contracts

    def test_statistics(self):
        code = '''
def f(x):
    if x is not None:
        return x.strip()
    return ""

def g(y):
    assert y > 0
    return 10 / y
'''
        result = analyze_liquid(code)
        assert result.predicates_harvested > 0
        assert result.constraints_generated >= 0
        assert result.analysis_time_ms >= 0

    def test_empty_module(self):
        code = '''
x = 42
'''
        result = analyze_liquid(code)
        assert len(result.contracts) == 0
        assert len(result.bugs) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 10. Walrus & Comprehension Harvesting
# ═══════════════════════════════════════════════════════════════════════════

class TestWalrusAndComprehension:
    def test_walrus_guard(self):
        code = '''
import re
def f(s):
    if (m := re.match(r"\\d+", s)):
        return m.group()
    return ""
'''
        preds = _harvest(code)
        walrus_preds = [hp for hp in preds if hp.source == HarvestSource.WALRUS]
        assert len(walrus_preds) > 0

    def test_comprehension_filter(self):
        code = '''
def f(items):
    result = [x for x in items if x is not None]
    return result
'''
        preds = _harvest(code)
        comp_preds = [hp for hp in preds if hp.source == HarvestSource.COMPREHENSION]
        assert len(comp_preds) > 0

    def test_comprehension_isinstance_filter(self):
        code = '''
def f(items):
    ints = [x for x in items if isinstance(x, int)]
    return ints
'''
        preds = _harvest(code)
        comp_preds = [hp for hp in preds if hp.source == HarvestSource.COMPREHENSION]
        assert len(comp_preds) > 0
        assert any(hp.pred.op == PredOp.ISINSTANCE for hp in comp_preds)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Contract API
# ═══════════════════════════════════════════════════════════════════════════

class TestContractAPI:
    def test_contract_pretty(self):
        code = '''
def f(x):
    assert x > 0
    return x + 1
'''
        result = analyze_liquid(code)
        contract = result.contracts["f"]
        sig = contract.annotated_signature()
        assert "def f" in sig

    def test_infer_contract_api(self):
        code = '''
def add(a, b):
    return a + b
'''
        contract = infer_contract(code, "add")
        assert contract is not None
        assert contract.name == "add"

    def test_function_contract_dep_func(self):
        code = '''
def f(x):
    if x > 0:
        return x
    return 0
'''
        result = analyze_liquid(code)
        contract = result.contracts["f"]
        dft = contract.to_dep_func_type()
        assert dft is not None


# ═══════════════════════════════════════════════════════════════════════════
# 12. Reporter & Templates
# ═══════════════════════════════════════════════════════════════════════════

class TestReporterAndTemplates:
    def test_report_format(self):
        code = '''
def f(x):
    return x.strip()
'''
        result = analyze_liquid(code)
        report = LiquidTypeReporter.format_result(result)
        assert "Liquid Type Analysis Report" in report

    def test_stub_generation(self):
        code = '''
def f(x):
    return x + 1
'''
        result = analyze_liquid(code)
        stubs = LiquidTypeReporter.format_contracts_as_stubs(result)
        assert "def f" in stubs

    def test_template_library(self):
        tmpl = PredicateTemplateLibrary.all_templates("x")
        assert len(tmpl) > 0
        ops = {t.op for t in tmpl}
        assert PredOp.IS_NONE in ops
        assert PredOp.IS_NOT_NONE in ops
        assert PredOp.GT in ops

    def test_result_summary(self):
        code = '''
def f(x):
    return x
'''
        result = analyze_liquid(code)
        summary = result.summary()
        assert "Liquid Type Analysis" in summary


# ═══════════════════════════════════════════════════════════════════════════
# 13. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_nested_functions_skipped(self):
        code = '''
def outer():
    def inner():
        return None
    return inner()
'''
        result = analyze_liquid(code)
        assert "outer" in result.contracts
        assert "inner" in result.contracts

    def test_constant_return(self):
        code = '''
def f():
    return 42
'''
        result = analyze_liquid(code)
        contract = result.contracts["f"]
        assert contract.return_type.pred.op != PredOp.FALSE

    def test_while_loop(self):
        code = '''
def f(x):
    while x > 0:
        x = x - 1
    return x
'''
        result = analyze_liquid(code)
        assert "f" in result.contracts

    def test_for_loop(self):
        code = '''
def f(items):
    for x in items:
        print(x)
'''
        result = analyze_liquid(code)
        assert "f" in result.contracts

    def test_try_except(self):
        code = '''
def f(x):
    try:
        return int(x)
    except ValueError:
        return 0
'''
        result = analyze_liquid(code)
        assert "f" in result.contracts
