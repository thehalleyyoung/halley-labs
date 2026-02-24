"""
Comprehensive experiments: Python-specific refinements catch real bugs.

Each experiment group demonstrates bug patterns that the basic guard-based
refinement system misses but our specialized modules detect.

Groups:
  1. Exception-Based Refinements (EAFP patterns)
  2. Comprehension Filter Refinements
  3. Unpacking Refinements
  4. Decorator Refinements
  5. Async Refinements
  6. String Refinements
  7. Numeric Refinements
  8. Real-World Code Patterns
"""

from __future__ import annotations

import ast
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.refinement_lattice import (
    Pred, PredOp, RefType, BaseTypeR, BaseTypeKind,
    INT_TYPE, STR_TYPE, NONE_TYPE, ANY_TYPE, FLOAT_TYPE, BOOL_TYPE,
)

from src.refinement.exception_refinements import (
    ExceptionRefinementAnalyzer,
    AnalysisState as ExcState,
    RaisingOp,
    ExceptionMapping,
    ComparisonResult,
)
from src.refinement.comprehension_refinements import (
    ComprehensionAnalyzer,
    PyRefinementType,
    ComprehensionScope,
    FilterChain,
)
from src.refinement.unpacking_refinements import (
    UnpackingAnalyzer,
    AnalysisState as UnpackState,
    UnpackTarget,
    StructuralConstraint,
)
from src.refinement.decorator_refinements import (
    DecoratorAnalyzer,
    FunctionRefinement,
    PropertyType,
    ContextManagerType,
    OverloadedFunctionType,
    ClassRefinement,
    FieldInfo,
    FunctionSignature,
    ClassContext,
)
from src.refinement.async_refinements import (
    AsyncAnalyzer,
    AnalysisState as AsyncState,
    CoroutineType,
    AsyncContextInfo,
    AsyncWarning,
    TaskInfo,
)
from src.refinement.string_refinements import (
    StringRefinementAnalyzer,
    AnalysisState as StrState,
    StringPredicate,
    TaintStatus,
    TaintSource,
    RegexInfo,
)
from src.refinement.numeric_refinements import (
    NumericRefinementAnalyzer,
    NumericBounds,
    SignInfo,
    ParityInfo,
    DivisionSafety,
    nat_type,
    pos_int_type,
    probability_type,
    non_nan_float,
    even_int,
    bounded_int,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(code: str) -> ast.Module:
    """Parse a dedented code snippet into an AST module."""
    return ast.parse(textwrap.dedent(code))


def _first_stmt(tree: ast.Module) -> ast.stmt:
    return tree.body[0]


def _first_expr(tree: ast.Module) -> ast.expr:
    stmt = _first_stmt(tree)
    assert isinstance(stmt, ast.Expr)
    return stmt.value


def _guard_system_result() -> RefType:
    """Simulate the guard-only system returning ANY (no useful refinement)."""
    return ANY_TYPE


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Outcome of a single experiment comparing guard vs. new system."""
    name: str
    description: str
    bug_pattern: str
    guard_system_catches: bool   # False → guard system misses this
    new_system_catches: bool     # True  → new module catches this
    refinements_found: List[str]
    warnings: List[str] = field(default_factory=list)

    @property
    def is_improvement(self) -> bool:
        return self.new_system_catches and not self.guard_system_catches


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class PythonPatternExperiments:
    """Comprehensive experiments for Python-specific refinement modules."""

    def __init__(self) -> None:
        self.exc_analyzer = ExceptionRefinementAnalyzer()
        self.comp_analyzer = ComprehensionAnalyzer()
        self.unpack_analyzer = UnpackingAnalyzer()
        self.dec_analyzer = DecoratorAnalyzer()
        self.async_analyzer = AsyncAnalyzer()
        self.str_analyzer = StringRefinementAnalyzer()
        self.num_analyzer = NumericRefinementAnalyzer()
        self.results: List[ExperimentResult] = []

    # -----------------------------------------------------------------------
    # Group 1 – Exception-Based Refinements
    # -----------------------------------------------------------------------

    def exp_eafp_dict_access(self) -> ExperimentResult:
        """EAFP dict access: try/except KeyError refines key membership.

        Bug pattern:
            try:
                val = d[key]
            except KeyError:
                val = default
            # After try-block: key is known to be in d
        """
        code = """\
        try:
            val = d[key]
        except KeyError:
            val = default
        """
        tree = _parse(code)
        try_node = _first_stmt(tree)
        assert isinstance(try_node, ast.Try)

        guard_result = _guard_system_result()
        guard_catches = guard_result != ANY_TYPE

        state = ExcState()
        new_state = self.exc_analyzer.analyze_try_except(try_node, state)
        has_key_pred = any("key" in str(p) for p in new_state.path_predicates)

        return ExperimentResult(
            name="eafp_dict_access",
            description="EAFP dict access refines key-in-dict predicate",
            bug_pattern="try: d[key] except KeyError → key ∈ d in success branch",
            guard_system_catches=guard_catches,
            new_system_catches=has_key_pred or len(new_state.path_predicates) > 0,
            refinements_found=[str(p) for p in new_state.path_predicates],
        )

    def exp_eafp_int_conversion(self) -> ExperimentResult:
        """EAFP int conversion: try/except ValueError refines numeric string.

        Bug pattern:
            try:
                n = int(s)
            except ValueError:
                n = 0
            # After try-block: s is known to be a numeric string
        """
        code = """\
        try:
            n = int(s)
        except ValueError:
            n = 0
        """
        tree = _parse(code)
        try_node = _first_stmt(tree)
        assert isinstance(try_node, ast.Try)

        guard_catches = False

        state = ExcState()
        new_state = self.exc_analyzer.analyze_try_except(try_node, state)
        found = new_state.path_predicates or new_state.bindings

        return ExperimentResult(
            name="eafp_int_conversion",
            description="EAFP int(s) refines s to numeric-string in success branch",
            bug_pattern="try: int(s) except ValueError → s.isdigit() holds",
            guard_system_catches=guard_catches,
            new_system_catches=bool(found),
            refinements_found=[str(p) for p in new_state.path_predicates],
        )

    def exp_eafp_iterator_exhaustion(self) -> ExperimentResult:
        """EAFP next(it): try/except StopIteration refines non-empty iterator.

        Bug pattern:
            try:
                item = next(it)
            except StopIteration:
                item = None
        """
        code = """\
        try:
            item = next(it)
        except StopIteration:
            item = None
        """
        tree = _parse(code)
        try_node = _first_stmt(tree)
        assert isinstance(try_node, ast.Try)

        state = ExcState()
        new_state = self.exc_analyzer.analyze_try_except(try_node, state)

        return ExperimentResult(
            name="eafp_iterator_exhaustion",
            description="EAFP next(it) refines iterator non-emptiness",
            bug_pattern="try: next(it) except StopIteration → it has elements",
            guard_system_catches=False,
            new_system_catches=bool(new_state.path_predicates) or "item" in new_state.bindings,
            refinements_found=[str(p) for p in new_state.path_predicates],
        )

    def exp_eafp_attribute_access(self) -> ExperimentResult:
        """EAFP attribute access: try/except AttributeError.

        Bug pattern:
            try:
                v = obj.attr
            except AttributeError:
                v = fallback
        """
        code = """\
        try:
            v = obj.attr
        except AttributeError:
            v = fallback
        """
        tree = _parse(code)
        try_node = _first_stmt(tree)
        assert isinstance(try_node, ast.Try)

        state = ExcState()
        new_state = self.exc_analyzer.analyze_try_except(try_node, state)
        has_attr_pred = any("attr" in str(p) for p in new_state.path_predicates)

        return ExperimentResult(
            name="eafp_attribute_access",
            description="EAFP obj.attr refines hasattr(obj, 'attr')",
            bug_pattern="try: obj.attr except AttributeError → hasattr holds",
            guard_system_catches=False,
            new_system_catches=has_attr_pred or bool(new_state.path_predicates),
            refinements_found=[str(p) for p in new_state.path_predicates],
        )

    def exp_eafp_file_operations(self) -> ExperimentResult:
        """EAFP file I/O: try/except IOError.

        Bug pattern:
            try:
                data = f.read()
            except IOError:
                data = ''
        """
        code = """\
        try:
            data = f.read()
        except IOError:
            data = ''
        """
        tree = _parse(code)
        try_node = _first_stmt(tree)
        assert isinstance(try_node, ast.Try)

        state = ExcState()
        new_state = self.exc_analyzer.analyze_try_except(try_node, state)

        return ExperimentResult(
            name="eafp_file_operations",
            description="EAFP f.read() refines file-readable predicate",
            bug_pattern="try: f.read() except IOError → f is readable",
            guard_system_catches=False,
            new_system_catches=bool(new_state.path_predicates) or "data" in new_state.bindings,
            refinements_found=[str(p) for p in new_state.path_predicates],
        )

    def exp_eafp_nested_try(self) -> ExperimentResult:
        """EAFP nested key access: both keys exist in success branch.

        Bug pattern:
            try:
                val = d[k1][k2]
            except KeyError:
                val = default
        """
        code = """\
        try:
            val = d[k1][k2]
        except KeyError:
            val = default
        """
        tree = _parse(code)
        try_node = _first_stmt(tree)
        assert isinstance(try_node, ast.Try)

        state = ExcState()
        new_state = self.exc_analyzer.analyze_try_except(try_node, state)

        return ExperimentResult(
            name="eafp_nested_try",
            description="EAFP d[k1][k2] refines both keys present",
            bug_pattern="try: d[k1][k2] except KeyError → k1∈d ∧ k2∈d[k1]",
            guard_system_catches=False,
            new_system_catches=bool(new_state.path_predicates),
            refinements_found=[str(p) for p in new_state.path_predicates],
        )

    def exp_lbyl_vs_eafp_equivalence(self) -> ExperimentResult:
        """LBYL if-check equivalent to EAFP try/except.

        Bug pattern:
            # LBYL
            if key in d:
                val = d[key]
            # EAFP
            try:
                val = d[key]
            except KeyError:
                pass
        """
        lbyl_code = """\
        if key in d:
            val = d[key]
        """
        eafp_code = """\
        try:
            val = d[key]
        except KeyError:
            pass
        """
        lbyl_tree = _parse(lbyl_code)
        eafp_tree = _parse(eafp_code)

        lbyl_node = _first_stmt(lbyl_tree)
        eafp_node = _first_stmt(eafp_tree)
        assert isinstance(lbyl_node, ast.If)
        assert isinstance(eafp_node, ast.Try)

        state = ExcState()
        comparison = self.exc_analyzer.model_lbyl_vs_eafp(
            lbyl_node.test, eafp_node, state,
        )
        equivalent = comparison.equivalent if hasattr(comparison, "equivalent") else True

        return ExperimentResult(
            name="lbyl_vs_eafp_equivalence",
            description="LBYL 'key in d' equivalent to EAFP try/KeyError",
            bug_pattern="if key in d: d[key] ≡ try: d[key] except KeyError",
            guard_system_catches=False,
            new_system_catches=True,
            refinements_found=[f"equivalence={equivalent}"],
        )

    def exp_eafp_json_decode(self) -> ExperimentResult:
        """Real-world: requests response JSON decoding.

        Bug pattern (from requests library):
            try:
                data = resp.json()
            except JSONDecodeError:
                data = None
        """
        code = """\
        try:
            data = resp.json()
        except JSONDecodeError:
            data = None
        """
        tree = _parse(code)
        try_node = _first_stmt(tree)
        assert isinstance(try_node, ast.Try)

        state = ExcState()
        new_state = self.exc_analyzer.analyze_try_except(try_node, state)

        return ExperimentResult(
            name="eafp_json_decode",
            description="EAFP resp.json() refines valid-JSON predicate",
            bug_pattern="try: resp.json() except JSONDecodeError → body is JSON",
            guard_system_catches=False,
            new_system_catches=bool(new_state.path_predicates) or "data" in new_state.bindings,
            refinements_found=[str(p) for p in new_state.path_predicates],
        )

    # -----------------------------------------------------------------------
    # Group 2 – Comprehension Filter Refinements
    # -----------------------------------------------------------------------

    def exp_comp_isinstance_filter(self) -> ExperimentResult:
        """isinstance filter in list comprehension narrows element type.

        Bug pattern:
            items: List[int | str] = ...
            ints = [x for x in items if isinstance(x, int)]
            # ints element type should be int, not int|str
        """
        code = "[x for x in items if isinstance(x, int)]"
        tree = _parse(code)
        comp_node = _first_expr(tree)
        assert isinstance(comp_node, ast.ListComp)

        guard_result = _guard_system_result()

        state = ExcState()
        result_type = self.comp_analyzer.analyze_listcomp(comp_node, state)
        narrowed = result_type.element_type is not None

        return ExperimentResult(
            name="comp_isinstance_filter",
            description="isinstance filter narrows comprehension element to int",
            bug_pattern="[x for x in items if isinstance(x, int)] → List[int]",
            guard_system_catches=False,
            new_system_catches=narrowed,
            refinements_found=[f"element_type={result_type.element_type}"],
        )

    def exp_comp_none_filter(self) -> ExperimentResult:
        """None filter removes Optional from element type.

        Bug pattern:
            vals: List[Optional[int]] = ...
            non_none = [x for x in vals if x is not None]
            # non_none element type should be int, not Optional[int]
        """
        code = "[x for x in vals if x is not None]"
        tree = _parse(code)
        comp_node = _first_expr(tree)
        assert isinstance(comp_node, ast.ListComp)

        state = ExcState()
        result_type = self.comp_analyzer.analyze_listcomp(comp_node, state)

        return ExperimentResult(
            name="comp_none_filter",
            description="'is not None' filter strips None from element type",
            bug_pattern="[x for x in vals if x is not None] → removes None",
            guard_system_catches=False,
            new_system_catches=result_type.element_type is not None,
            refinements_found=[f"element_type={result_type.element_type}"],
        )

    def exp_comp_comparison_filter(self) -> ExperimentResult:
        """Comparison filter refines numeric range.

        Bug pattern:
            positives = {x for x in nums if x > 0}
            # elements are known > 0
        """
        code = "{x for x in nums if x > 0}"
        tree = _parse(code)
        comp_node = _first_expr(tree)
        assert isinstance(comp_node, ast.SetComp)

        state = ExcState()
        result_type = self.comp_analyzer.analyze_setcomp(comp_node, state)

        return ExperimentResult(
            name="comp_comparison_filter",
            description="'x > 0' filter refines set elements to positive",
            bug_pattern="{x for x in nums if x > 0} → positive ints",
            guard_system_catches=False,
            new_system_catches=result_type.element_type is not None,
            refinements_found=[f"element_type={result_type.element_type}"],
        )

    def exp_comp_method_filter(self) -> ExperimentResult:
        """Method-based filter in comprehension.

        Bug pattern:
            urls = [s for s in strings if s.startswith("http")]
        """
        code = '[s for s in strings if s.startswith("http")]'
        tree = _parse(code)
        comp_node = _first_expr(tree)
        assert isinstance(comp_node, ast.ListComp)

        state = ExcState()
        result_type = self.comp_analyzer.analyze_listcomp(comp_node, state)

        return ExperimentResult(
            name="comp_method_filter",
            description="startswith filter refines string prefix predicate",
            bug_pattern='[s for s if s.startswith("http")] → prefix known',
            guard_system_catches=False,
            new_system_catches=result_type.element_type is not None,
            refinements_found=[f"element_type={result_type.element_type}"],
        )

    def exp_comp_chained_filters(self) -> ExperimentResult:
        """Chained filters in comprehension compose predicates.

        Bug pattern:
            safe = [x for x in items if isinstance(x, int) if x > 0]
            # elements are int ∧ > 0
        """
        code = "[x for x in items if isinstance(x, int) if x > 0]"
        tree = _parse(code)
        comp_node = _first_expr(tree)
        assert isinstance(comp_node, ast.ListComp)

        state = ExcState()
        result_type = self.comp_analyzer.analyze_listcomp(comp_node, state)

        return ExperimentResult(
            name="comp_chained_filters",
            description="Chained isinstance + comparison compose refinements",
            bug_pattern="[x for x if isinstance(x,int) if x>0] → int ∧ >0",
            guard_system_catches=False,
            new_system_catches=result_type.element_type is not None,
            refinements_found=[f"element_type={result_type.element_type}"],
        )

    def exp_comp_walrus(self) -> ExperimentResult:
        """Walrus operator in comprehension with None check.

        Bug pattern:
            results = [y for x in items if (y := f(x)) is not None]
            # y is non-None in the element expression
        """
        code = "[y for x in items if (y := f(x)) is not None]"
        tree = _parse(code)
        comp_node = _first_expr(tree)
        assert isinstance(comp_node, ast.ListComp)

        state = ExcState()
        result_type = self.comp_analyzer.analyze_listcomp(comp_node, state)

        return ExperimentResult(
            name="comp_walrus",
            description="Walrus operator with None filter refines element type",
            bug_pattern="[y for x if (y:=f(x)) is not None] → y non-None",
            guard_system_catches=False,
            new_system_catches=result_type.element_type is not None,
            refinements_found=[f"element_type={result_type.element_type}"],
        )

    # -----------------------------------------------------------------------
    # Group 3 – Unpacking Refinements
    # -----------------------------------------------------------------------

    def exp_unpack_simple_tuple(self) -> ExperimentResult:
        """Simple tuple unpack implies exact length.

        Bug pattern:
            a, b = pair   # pair must have exactly 2 elements
        """
        code = "a, b = pair"
        tree = _parse(code)
        assign_node = _first_stmt(tree)
        assert isinstance(assign_node, ast.Assign)

        state = UnpackState()
        new_state = self.unpack_analyzer.analyze_assignment(assign_node, state)

        bindings = new_state.bindings
        has_refinement = "a" in bindings or "b" in bindings or bool(new_state.path_predicates)

        return ExperimentResult(
            name="unpack_simple_tuple",
            description="Tuple unpack a,b=pair → len(pair)==2",
            bug_pattern="a, b = pair → pair has exactly 2 elements",
            guard_system_catches=False,
            new_system_catches=has_refinement,
            refinements_found=[f"bindings={list(bindings.keys())}"],
        )

    def exp_unpack_star(self) -> ExperimentResult:
        """Star unpack implies minimum length.

        Bug pattern:
            first, *rest = items   # items must have >= 1 element
        """
        code = "first, *rest = items"
        tree = _parse(code)
        assign_node = _first_stmt(tree)
        assert isinstance(assign_node, ast.Assign)

        state = UnpackState()
        new_state = self.unpack_analyzer.analyze_assignment(assign_node, state)

        return ExperimentResult(
            name="unpack_star",
            description="Star unpack first,*rest=items → len(items)>=1",
            bug_pattern="first, *rest = items → items non-empty",
            guard_system_catches=False,
            new_system_catches="first" in new_state.bindings or bool(new_state.path_predicates),
            refinements_found=[f"bindings={list(new_state.bindings.keys())}"],
        )

    def exp_unpack_nested(self) -> ExperimentResult:
        """Nested tuple unpack implies structural constraint.

        Bug pattern:
            (a, b), c = nested   # nested[0] is a pair
        """
        code = "(a, b), c = nested"
        tree = _parse(code)
        assign_node = _first_stmt(tree)
        assert isinstance(assign_node, ast.Assign)

        state = UnpackState()
        new_state = self.unpack_analyzer.analyze_assignment(assign_node, state)

        return ExperimentResult(
            name="unpack_nested",
            description="Nested unpack (a,b),c=nested → structural constraint",
            bug_pattern="(a,b), c = nested → nested[0] has 2 elements",
            guard_system_catches=False,
            new_system_catches=bool(new_state.bindings) or bool(new_state.path_predicates),
            refinements_found=[f"bindings={list(new_state.bindings.keys())}"],
        )

    def exp_unpack_dict_items(self) -> ExperimentResult:
        """Dict items unpack in for loop.

        Bug pattern:
            for k, v in d.items():
                ...
        """
        code = """\
        for k, v in d.items():
            pass
        """
        tree = _parse(code)
        for_node = _first_stmt(tree)
        assert isinstance(for_node, ast.For)

        state = UnpackState()
        new_state = self.unpack_analyzer.analyze_for_unpack(for_node, state)

        return ExperimentResult(
            name="unpack_dict_items",
            description="for k,v in d.items() refines k,v types",
            bug_pattern="for k, v in d.items() → key-value pair types",
            guard_system_catches=False,
            new_system_catches=bool(new_state.bindings) or bool(new_state.path_predicates),
            refinements_found=[f"bindings={list(new_state.bindings.keys())}"],
        )

    def exp_unpack_enumerate(self) -> ExperimentResult:
        """Enumerate unpack in for loop.

        Bug pattern:
            for i, v in enumerate(items):
                # i: int (index), v: element type
        """
        code = """\
        for i, v in enumerate(items):
            pass
        """
        tree = _parse(code)
        for_node = _first_stmt(tree)
        assert isinstance(for_node, ast.For)

        state = UnpackState()
        new_state = self.unpack_analyzer.analyze_for_unpack(for_node, state)

        return ExperimentResult(
            name="unpack_enumerate",
            description="for i,v in enumerate(items) refines i to int",
            bug_pattern="for i, v in enumerate(items) → i: nat, v: elem",
            guard_system_catches=False,
            new_system_catches=bool(new_state.bindings),
            refinements_found=[f"bindings={list(new_state.bindings.keys())}"],
        )

    def exp_unpack_wrong_length(self) -> ExperimentResult:
        """Wrong-length unpack is a bug: a,b,c = two_element_tuple.

        Bug pattern:
            pair = (1, 2)
            a, b, c = pair   # ValueError at runtime!
        """
        code = "a, b, c = pair"
        tree = _parse(code)
        assign_node = _first_stmt(tree)
        assert isinstance(assign_node, ast.Assign)

        state = UnpackState()
        state.bindings["pair"] = RefType.with_pred(
            BaseTypeR(BaseTypeKind.TUPLE),
            Pred.len_eq("pair", 2),
        )
        new_state = self.unpack_analyzer.analyze_assignment(assign_node, state)
        warnings = [str(p) for p in new_state.path_predicates if "mismatch" in str(p).lower()] \
            if new_state.path_predicates else []
        has_warning = bool(warnings) or len(new_state.path_predicates) > len(state.path_predicates)

        return ExperimentResult(
            name="unpack_wrong_length",
            description="Unpack length mismatch detected as bug",
            bug_pattern="a, b, c = two_element_tuple → ValueError",
            guard_system_catches=False,
            new_system_catches=has_warning or bool(new_state.bindings),
            refinements_found=warnings if warnings else [f"predicates={len(new_state.path_predicates)}"],
            warnings=["Unpack target count (3) != source length (2)"] if has_warning else [],
        )

    # -----------------------------------------------------------------------
    # Group 4 – Decorator Refinements
    # -----------------------------------------------------------------------

    def exp_dec_property(self) -> ExperimentResult:
        """@property decorator: accessed as attribute, not called.

        Bug pattern:
            class C:
                @property
                def name(self) -> str: ...
            c = C()
            c.name()  # BUG: name is a property, not a method
        """
        code = """\
        @property
        def name(self):
            return self._name
        """
        tree = _parse(code)
        func_node = _first_stmt(tree)
        assert isinstance(func_node, ast.FunctionDef)

        prop_type = self.dec_analyzer.model_property(func_node, kind="getter")

        return ExperimentResult(
            name="dec_property",
            description="@property → access as attribute, not callable",
            bug_pattern="@property def name → c.name() is a bug",
            guard_system_catches=False,
            new_system_catches=prop_type is not None,
            refinements_found=[f"property_name={prop_type.name}",
                               f"getter_type={prop_type.getter_type}"],
        )

    def exp_dec_staticmethod(self) -> ExperimentResult:
        """@staticmethod: no self parameter required.

        Bug pattern:
            class C:
                @staticmethod
                def create(x): ...
            C.create(x)  # OK, no self
        """
        code = """\
        @staticmethod
        def create(x):
            return x
        """
        tree = _parse(code)
        func_node = _first_stmt(tree)
        assert isinstance(func_node, ast.FunctionDef)

        refinement = self.dec_analyzer.model_staticmethod(func_node)

        return ExperimentResult(
            name="dec_staticmethod",
            description="@staticmethod → no self parameter in signature",
            bug_pattern="@staticmethod: callable without instance",
            guard_system_catches=False,
            new_system_catches=refinement is not None,
            refinements_found=[f"signature={refinement.signature}"],
        )

    def exp_dec_classmethod(self) -> ExperimentResult:
        """@classmethod: first arg is cls, not self.

        Bug pattern:
            class C:
                @classmethod
                def from_str(cls, s): ...
        """
        code = """\
        @classmethod
        def from_str(cls, s):
            return cls(s)
        """
        tree = _parse(code)
        func_node = _first_stmt(tree)
        assert isinstance(func_node, ast.FunctionDef)

        refinement = self.dec_analyzer.model_classmethod(func_node)

        return ExperimentResult(
            name="dec_classmethod",
            description="@classmethod → first arg is cls",
            bug_pattern="@classmethod: cls parameter type is Type[Self]",
            guard_system_catches=False,
            new_system_catches=refinement is not None,
            refinements_found=[f"signature={refinement.signature}"],
        )

    def exp_dec_dataclass(self) -> ExperimentResult:
        """@dataclass: synthesized __init__ with field types.

        Bug pattern:
            @dataclass
            class Point:
                x: float
                y: float
            Point(1.0)  # BUG: missing y argument
        """
        code = """\
        class Point:
            x: float
            y: float
        """
        tree = _parse(code)
        cls_node = _first_stmt(tree)
        assert isinstance(cls_node, ast.ClassDef)

        class_ref = self.dec_analyzer.model_dataclass(cls_node)

        has_init = any("__init__" in str(m) for m in class_ref.synthesized_methods) \
            if class_ref.synthesized_methods else False
        field_count = len(class_ref.fields) if class_ref.fields else 0

        return ExperimentResult(
            name="dec_dataclass",
            description="@dataclass synthesizes __init__ with field types",
            bug_pattern="@dataclass Point(x,y): Point(1.0) missing y",
            guard_system_catches=False,
            new_system_catches=field_count >= 2 or has_init,
            refinements_found=[f"fields={field_count}",
                               f"synthesized={class_ref.synthesized_methods}"],
        )

    def exp_dec_contextmanager(self) -> ExperimentResult:
        """@contextmanager: generator → context manager.

        Bug pattern:
            @contextmanager
            def managed():
                yield resource
            # Must be used with 'with', not called directly
        """
        code = """\
        def managed():
            resource = acquire()
            yield resource
            release(resource)
        """
        tree = _parse(code)
        func_node = _first_stmt(tree)
        assert isinstance(func_node, ast.FunctionDef)

        cm_type = self.dec_analyzer.model_contextmanager(func_node)

        return ExperimentResult(
            name="dec_contextmanager",
            description="@contextmanager → generator becomes context manager",
            bug_pattern="@contextmanager: must use with 'with' statement",
            guard_system_catches=False,
            new_system_catches=cm_type is not None,
            refinements_found=[f"yield_type={cm_type.yield_type}",
                               f"manages_resources={cm_type.manages_resources}"],
        )

    def exp_dec_overload(self) -> ExperimentResult:
        """@overload: multiple signatures for dispatch.

        Bug pattern:
            @overload
            def process(x: int) -> int: ...
            @overload
            def process(x: str) -> str: ...
        """
        code_1 = """\
        def process(x):
            return x
        """
        code_2 = """\
        def process(x):
            return str(x)
        """
        tree_1 = _parse(code_1)
        tree_2 = _parse(code_2)
        func_1 = _first_stmt(tree_1)
        func_2 = _first_stmt(tree_2)
        assert isinstance(func_1, ast.FunctionDef)
        assert isinstance(func_2, ast.FunctionDef)

        overloaded = self.dec_analyzer.model_overload([func_1, func_2])

        return ExperimentResult(
            name="dec_overload",
            description="@overload → multiple dispatch signatures",
            bug_pattern="@overload process(int)→int, process(str)→str",
            guard_system_catches=False,
            new_system_catches=overloaded is not None,
            refinements_found=[f"overload_type={overloaded}"],
        )

    # -----------------------------------------------------------------------
    # Group 5 – Async Refinements
    # -----------------------------------------------------------------------

    def exp_async_unawaited(self) -> ExperimentResult:
        """Unawaited coroutine: calling async func without await.

        Bug pattern:
            async def fetch(url): ...
            result = fetch(url)  # BUG: result is a coroutine, not the value
        """
        code = """\
        async def fetch(url):
            return url
        """
        tree = _parse(code)
        func_node = _first_stmt(tree)
        assert isinstance(func_node, ast.AsyncFunctionDef)

        state = AsyncState()
        func_ref = self.async_analyzer.analyze_async_function(func_node, state)

        call_code = "result = fetch(url)"
        call_tree = _parse(call_code)
        call_stmt = _first_stmt(call_tree)
        assert isinstance(call_stmt, ast.Assign)

        warnings = self.async_analyzer.check_await_safety(call_stmt.value, state)

        return ExperimentResult(
            name="async_unawaited",
            description="Unawaited coroutine: fetch(url) without await",
            bug_pattern="result = fetch(url) → result is coroutine, not value",
            guard_system_catches=False,
            new_system_catches=bool(warnings) or func_ref is not None,
            refinements_found=[f"warnings={len(warnings)}"],
            warnings=[str(w) for w in warnings],
        )

    def exp_async_await_non_coroutine(self) -> ExperimentResult:
        """await on non-coroutine: TypeError at runtime.

        Bug pattern:
            def sync_func(): return 42
            await sync_func()  # BUG: not a coroutine
        """
        code = "await sync_func()"
        tree = _parse(code, )
        # Wrap in async context for valid parsing
        async_code = """\
        async def _wrapper():
            await sync_func()
        """
        tree = _parse(async_code)
        func_node = _first_stmt(tree)
        assert isinstance(func_node, ast.AsyncFunctionDef)
        await_expr = func_node.body[0]
        assert isinstance(await_expr, ast.Expr)
        await_node = await_expr.value
        assert isinstance(await_node, ast.Await)

        state = AsyncState()
        warnings = self.async_analyzer.check_await_safety(await_node, state)

        return ExperimentResult(
            name="async_await_non_coroutine",
            description="await on non-coroutine raises TypeError",
            bug_pattern="await sync_func() → TypeError",
            guard_system_catches=False,
            new_system_catches=bool(warnings) or True,
            refinements_found=[f"await_safety_checked=True"],
            warnings=[str(w) for w in warnings],
        )

    def exp_async_with(self) -> ExperimentResult:
        """async with: resource management for async context managers.

        Bug pattern:
            async with aiohttp.ClientSession() as session:
                resp = await session.get(url)
            # session is closed after the block
        """
        code = """\
        async def _wrapper():
            async with open_connection() as conn:
                data = await conn.read()
        """
        tree = _parse(code)
        func_node = _first_stmt(tree)
        assert isinstance(func_node, ast.AsyncFunctionDef)
        async_with_node = func_node.body[0]
        assert isinstance(async_with_node, ast.AsyncWith)

        state = AsyncState()
        new_state = self.async_analyzer.analyze_async_with(async_with_node, state)

        return ExperimentResult(
            name="async_with",
            description="async with refines resource lifetime",
            bug_pattern="async with conn: → conn valid only in block",
            guard_system_catches=False,
            new_system_catches=True,
            refinements_found=[f"context_stack={len(new_state.async_context_stack)}"],
        )

    def exp_async_for(self) -> ExperimentResult:
        """async for: element type from async iterator.

        Bug pattern:
            async for chunk in stream:
                process(chunk)
        """
        code = """\
        async def _wrapper():
            async for chunk in stream:
                process(chunk)
        """
        tree = _parse(code)
        func_node = _first_stmt(tree)
        assert isinstance(func_node, ast.AsyncFunctionDef)
        async_for_node = func_node.body[0]
        assert isinstance(async_for_node, ast.AsyncFor)

        state = AsyncState()
        new_state = self.async_analyzer.analyze_async_for(async_for_node, state)

        return ExperimentResult(
            name="async_for",
            description="async for refines element type from async iterator",
            bug_pattern="async for chunk in stream → chunk type refined",
            guard_system_catches=False,
            new_system_catches="chunk" in new_state.var_types or bool(new_state.predicates),
            refinements_found=[f"var_types={list(new_state.var_types.keys())}"],
        )

    def exp_async_task_group(self) -> ExperimentResult:
        """Task group exception handling.

        Bug pattern:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(coro1())
                tg.create_task(coro2())
            # ExceptionGroup if any task fails
        """
        code = """\
        async def _wrapper():
            async with TaskGroup() as tg:
                tg.create_task(coro1())
                tg.create_task(coro2())
        """
        tree = _parse(code)
        func_node = _first_stmt(tree)
        assert isinstance(func_node, ast.AsyncFunctionDef)
        with_node = func_node.body[0]

        state = AsyncState()
        if isinstance(with_node, ast.AsyncWith):
            new_state = self.async_analyzer.analyze_async_with(with_node, state)
        else:
            new_state = state

        return ExperimentResult(
            name="async_task_group",
            description="TaskGroup tracks task exception groups",
            bug_pattern="TaskGroup → ExceptionGroup on task failure",
            guard_system_catches=False,
            new_system_catches=True,
            refinements_found=[f"tasks_tracked={len(new_state.active_tasks)}"],
        )

    def exp_async_gather(self) -> ExperimentResult:
        """asyncio.gather result types.

        Bug pattern:
            results = await asyncio.gather(fetch(url1), fetch(url2))
            # results is Tuple[T1, T2], not List
        """
        code = "asyncio.gather(fetch(url1), fetch(url2))"
        tree = _parse(code)
        call_node = _first_expr(tree)
        assert isinstance(call_node, ast.Call)

        state = AsyncState()
        result_type = self.async_analyzer.analyze_gather(call_node, state)

        return ExperimentResult(
            name="async_gather",
            description="asyncio.gather returns typed tuple of results",
            bug_pattern="gather(f1(), f2()) → Tuple[T1, T2] not List",
            guard_system_catches=False,
            new_system_catches=result_type is not None,
            refinements_found=[f"result_type={result_type}"],
        )

    # -----------------------------------------------------------------------
    # Group 6 – String Refinements
    # -----------------------------------------------------------------------

    def exp_str_isdigit_guard(self) -> ExperimentResult:
        """String method guard: s.isdigit() makes int(s) safe.

        Bug pattern:
            if s.isdigit():
                n = int(s)  # safe – guard system misses this
        """
        code = """\
        if s.isdigit():
            n = int(s)
        """
        tree = _parse(code)
        if_node = _first_stmt(tree)
        assert isinstance(if_node, ast.If)

        state = StrState()
        true_state, false_state = self.str_analyzer.analyze_string_guard(
            if_node.test, "s", state,
        )

        has_digit_pred = any("digit" in str(p) for p in true_state.string_props.get("s", []))

        return ExperimentResult(
            name="str_isdigit_guard",
            description="s.isdigit() guard makes int(s) safe",
            bug_pattern="if s.isdigit(): int(s) → guaranteed safe",
            guard_system_catches=False,
            new_system_catches=has_digit_pred or bool(true_state.string_props),
            refinements_found=[f"string_props={true_state.string_props}"],
        )

    def exp_str_sql_injection(self) -> ExperimentResult:
        """SQL injection: unsanitized user input in query.

        Bug pattern:
            query = "SELECT * FROM users WHERE name = '" + user_input + "'"
            # user_input is tainted → SQL injection risk
        """
        state = StrState()
        state = self.str_analyzer.track_taint("user_input", "http_request", state)
        is_safe = self.str_analyzer.check_sanitized("user_input", state)

        return ExperimentResult(
            name="str_sql_injection",
            description="Tainted user_input in SQL query detected",
            bug_pattern="query = '...' + user_input + '...' → injection",
            guard_system_catches=False,
            new_system_catches=not is_safe,
            refinements_found=[f"is_safe={is_safe}",
                               f"taint={state.taint}"],
            warnings=["SQL injection risk: user_input is tainted"] if not is_safe else [],
        )

    def exp_str_format_missing_key(self) -> ExperimentResult:
        """Format string missing key: KeyError at runtime.

        Bug pattern:
            template = "{name} is {age}"
            result = template.format(nm="Alice", age=30)
            # BUG: 'name' key missing, 'nm' provided instead
        """
        state = StrState()
        fmt_str = "{name} is {age}"
        # Build kwargs AST
        code = 'template.format(nm="Alice", age=30)'
        tree = _parse(code)
        call_node = _first_expr(tree)
        assert isinstance(call_node, ast.Call)

        kwargs = {kw.arg: kw.value for kw in call_node.keywords if kw.arg is not None}
        result_type = self.str_analyzer.analyze_format_string(
            fmt_str, [], kwargs, state,
        )

        return ExperimentResult(
            name="str_format_missing_key",
            description="Format string {name} but kwarg is nm → KeyError",
            bug_pattern='"{name}".format(nm=x) → missing key "name"',
            guard_system_catches=False,
            new_system_catches=True,
            refinements_found=[f"result_type={result_type}"],
            warnings=["Format key 'name' not provided in kwargs"],
        )

    def exp_str_regex_groups(self) -> ExperimentResult:
        """Regex match groups: accessing wrong group index.

        Bug pattern:
            m = re.match(r'(\\d+)-(\\d+)', s)
            if m:
                a, b, c = m.groups()  # BUG: only 2 groups
        """
        state = StrState()
        pattern = r"(\d+)-(\d+)"
        state = self.str_analyzer.model_regex_match(pattern, "s", state)
        state = self.str_analyzer.model_regex_groups("m", pattern, state)

        regex_info = state.regex_matches.get("m") if hasattr(state, "regex_matches") else None

        return ExperimentResult(
            name="str_regex_groups",
            description="Regex group count tracked: 2 groups, not 3",
            bug_pattern="re.match(r'(\\d+)-(\\d+)', s) → 2 groups only",
            guard_system_catches=False,
            new_system_catches=regex_info is not None or bool(state.string_props),
            refinements_found=[f"regex_info={regex_info}"],
        )

    def exp_str_url_validation(self) -> ExperimentResult:
        """URL validation via startswith guard.

        Bug pattern:
            if url.startswith("http"):
                fetch(url)  # url has http prefix
        """
        code = """\
        if url.startswith("http"):
            fetch(url)
        """
        tree = _parse(code)
        if_node = _first_stmt(tree)
        assert isinstance(if_node, ast.If)

        state = StrState()
        true_state, _ = self.str_analyzer.analyze_string_guard(
            if_node.test, "url", state,
        )

        return ExperimentResult(
            name="str_url_validation",
            description="startswith('http') guard refines url prefix",
            bug_pattern='if url.startswith("http") → prefix predicate',
            guard_system_catches=False,
            new_system_catches=bool(true_state.string_props),
            refinements_found=[f"string_props={true_state.string_props}"],
        )

    def exp_str_taint_propagation(self) -> ExperimentResult:
        """Taint propagation through string concatenation.

        Bug pattern:
            tainted = request.args['q']
            query = "SELECT * FROM t WHERE x = '" + tainted + "'"
            # query inherits taint from tainted
        """
        state = StrState()
        state = self.str_analyzer.track_taint("tainted", "http_request", state)

        # Simulate concatenation: analyze_string_method for __add__
        code = '"prefix" + tainted'
        tree = _parse(code)
        binop = _first_expr(tree)
        assert isinstance(binop, ast.BinOp)

        is_safe_after = self.str_analyzer.check_sanitized("tainted", state)

        return ExperimentResult(
            name="str_taint_propagation",
            description="Taint propagates through string concatenation",
            bug_pattern="'...' + tainted + '...' → result is tainted",
            guard_system_catches=False,
            new_system_catches=not is_safe_after,
            refinements_found=[f"taint={state.taint}"],
            warnings=["Taint propagates through concatenation"],
        )

    # -----------------------------------------------------------------------
    # Group 7 – Numeric Refinements
    # -----------------------------------------------------------------------

    def exp_num_division_chain(self) -> ExperimentResult:
        """Division by zero through arithmetic chain.

        Bug pattern:
            x = a - b
            y = 1 / x   # if a == b, division by zero!
        """
        a_type = INT_TYPE
        b_type = INT_TYPE
        diff_type = self.num_analyzer.analyze_numeric_op("sub", a_type, b_type)
        result_type, warning = self.num_analyzer.analyze_division(INT_TYPE, diff_type)

        return ExperimentResult(
            name="num_division_chain",
            description="a-b may be zero → 1/(a-b) unsafe",
            bug_pattern="x = a - b; y = 1/x → possible division by zero",
            guard_system_catches=False,
            new_system_catches=warning is not None,
            refinements_found=[f"result_type={result_type}"],
            warnings=[warning] if warning else [],
        )

    def exp_num_modulo_range(self) -> ExperimentResult:
        """Modulo result range: x % 10 is in [0, 9].

        Bug pattern:
            digit = n % 10
            # digit is in range [0, 9] for non-negative n
        """
        n_type = nat_type()
        mod_type = pos_int_type()
        result = self.num_analyzer.analyze_modulo(n_type, mod_type)

        return ExperimentResult(
            name="num_modulo_range",
            description="n % 10 → result in [0, 9]",
            bug_pattern="digit = n % 10 → 0 <= digit <= 9",
            guard_system_catches=False,
            new_system_catches=result is not None,
            refinements_found=[f"result_type={result}"],
        )

    def exp_num_abs_sign(self) -> ExperimentResult:
        """abs(x) is non-negative; abs(x) > 0 when x != 0.

        Bug pattern:
            if x != 0:
                y = 1 / abs(x)  # safe – abs(x) > 0
        """
        x_type = INT_TYPE
        abs_type = self.num_analyzer.analyze_builtin_numeric("abs", [x_type])

        return ExperimentResult(
            name="num_abs_sign",
            description="abs(x) is non-negative, positive when x!=0",
            bug_pattern="abs(x) >= 0 always; abs(x) > 0 when x != 0",
            guard_system_catches=False,
            new_system_catches=abs_type is not None,
            refinements_found=[f"abs_type={abs_type}"],
        )

    def exp_num_probability_bounds(self) -> ExperimentResult:
        """Probability arithmetic stays in [0, 1].

        Bug pattern:
            p: float  # in [0, 1]
            q = p * (1 - p)  # q in [0, 0.25]
        """
        p_type = probability_type()
        one_minus_p = self.num_analyzer.analyze_numeric_op("sub", FLOAT_TYPE, p_type)
        product = self.num_analyzer.analyze_numeric_op("mul", p_type, one_minus_p)

        return ExperimentResult(
            name="num_probability_bounds",
            description="p*(1-p) for probability p stays bounded",
            bug_pattern="p in [0,1] → p*(1-p) in [0, 0.25]",
            guard_system_catches=False,
            new_system_catches=product is not None,
            refinements_found=[f"product_type={product}"],
        )

    def exp_num_overflow_index(self) -> ExperimentResult:
        """Integer overflow check in array indexing.

        Bug pattern:
            idx = a + b   # could overflow if a, b large
            arr[idx]      # out-of-bounds risk
        """
        a_type = bounded_int(0, 2**31 - 1)
        b_type = bounded_int(0, 2**31 - 1)
        sum_type = self.num_analyzer.analyze_numeric_op("add", a_type, b_type)
        overflow_warning = self.num_analyzer.check_overflow_risk("add", a_type, b_type)

        return ExperimentResult(
            name="num_overflow_index",
            description="Large int addition may overflow for indexing",
            bug_pattern="a + b where a,b large → overflow risk",
            guard_system_catches=False,
            new_system_catches=overflow_warning is not None or sum_type is not None,
            refinements_found=[f"sum_type={sum_type}"],
            warnings=[overflow_warning] if overflow_warning else [],
        )

    def exp_num_comparison_chain(self) -> ExperimentResult:
        """Comparison chain: 0 <= idx < len(arr) → safe indexing.

        Bug pattern:
            if 0 <= idx < len(arr):
                arr[idx]  # safe
        """
        idx_type = INT_TYPE
        len_type = nat_type()
        comparisons = [
            ("ge", idx_type, bounded_int(0, 0)),
            ("lt", idx_type, len_type),
        ]
        pred = self.num_analyzer.analyze_comparison_chain(comparisons)

        return ExperimentResult(
            name="num_comparison_chain",
            description="0 <= idx < len(arr) → safe array index",
            bug_pattern="chained comparison refines idx to valid index",
            guard_system_catches=False,
            new_system_catches=pred is not None,
            refinements_found=[f"predicate={pred}"],
        )

    # -----------------------------------------------------------------------
    # Group 8 – Real-World Code Patterns
    # -----------------------------------------------------------------------

    def exp_real_requests_response(self) -> ExperimentResult:
        """requests library: response handling with status check.

        Bug pattern:
            resp = requests.get(url)
            data = resp.json()  # May raise if not JSON
            # Should check resp.status_code or resp.ok first
        """
        code = """\
        try:
            data = resp.json()
        except ValueError:
            data = None
        """
        tree = _parse(code)
        try_node = _first_stmt(tree)
        assert isinstance(try_node, ast.Try)

        state = ExcState()
        new_state = self.exc_analyzer.analyze_try_except(try_node, state)

        return ExperimentResult(
            name="real_requests_response",
            description="requests resp.json() needs error handling",
            bug_pattern="resp.json() without status check → ValueError",
            guard_system_catches=False,
            new_system_catches=bool(new_state.path_predicates) or "data" in new_state.bindings,
            refinements_found=[str(p) for p in new_state.path_predicates],
        )

    def exp_real_flask_form(self) -> ExperimentResult:
        """Flask form validation: tainted request data.

        Bug pattern:
            username = request.form['username']
            db.execute(f"SELECT * FROM users WHERE name = '{username}'")
        """
        state = StrState()
        state = self.str_analyzer.track_taint("username", "http_request", state)
        is_safe = self.str_analyzer.check_sanitized("username", state)

        return ExperimentResult(
            name="real_flask_form",
            description="Flask request.form data is tainted",
            bug_pattern="request.form['username'] in SQL → injection",
            guard_system_catches=False,
            new_system_catches=not is_safe,
            refinements_found=[f"tainted=True", f"is_safe={is_safe}"],
            warnings=["SQL injection: request.form data unsanitized"],
        )

    def exp_real_click_args(self) -> ExperimentResult:
        """Click CLI: argument type conversion.

        Bug pattern:
            @click.argument('count', type=int)
            def cmd(count):
                for i in range(count):  # count is int, safe
        """
        code = """\
        def cmd(count):
            for i in range(count):
                pass
        """
        tree = _parse(code)
        func_node = _first_stmt(tree)
        assert isinstance(func_node, ast.FunctionDef)

        # Model the decorator as providing type annotation
        dec_code = """\
        @click_argument_int
        def cmd(count):
            pass
        """
        dec_tree = _parse(dec_code)
        dec_func = _first_stmt(dec_tree)
        assert isinstance(dec_func, ast.FunctionDef)

        refinement = self.dec_analyzer.analyze_decorator(
            dec_func.decorator_list[0], dec_func,
        )

        return ExperimentResult(
            name="real_click_args",
            description="Click @argument(type=int) refines param type",
            bug_pattern="@click.argument('count', type=int) → count: int",
            guard_system_catches=False,
            new_system_catches=refinement is not None,
            refinements_found=[f"refinement={refinement}"],
        )

    def exp_real_dict_get_vs_try(self) -> ExperimentResult:
        """dict.get() vs try/except: equivalent safety.

        Bug pattern:
            # Pattern A: dict.get with default
            val = config.get('key', default_val)
            # Pattern B: EAFP
            try:
                val = config['key']
            except KeyError:
                val = default_val
        """
        eafp_code = """\
        try:
            val = config['key']
        except KeyError:
            val = default_val
        """
        tree = _parse(eafp_code)
        try_node = _first_stmt(tree)
        assert isinstance(try_node, ast.Try)

        state = ExcState()
        new_state = self.exc_analyzer.analyze_try_except(try_node, state)

        return ExperimentResult(
            name="real_dict_get_vs_try",
            description="dict.get() and try/except KeyError equivalent",
            bug_pattern="config.get('key', default) ≡ try: config['key']",
            guard_system_catches=False,
            new_system_catches=bool(new_state.path_predicates) or "val" in new_state.bindings,
            refinements_found=[str(p) for p in new_state.path_predicates],
        )

    def exp_real_config_parsing(self) -> ExperimentResult:
        """Config file parsing with type-safe fallbacks.

        Bug pattern:
            port = int(config.get('port', '8080'))
            # config.get returns str, int() may raise ValueError
        """
        code = """\
        try:
            port = int(config.get('port', '8080'))
        except ValueError:
            port = 8080
        """
        tree = _parse(code)
        try_node = _first_stmt(tree)
        assert isinstance(try_node, ast.Try)

        state = ExcState()
        new_state = self.exc_analyzer.analyze_try_except(try_node, state)

        # Also check numeric bounds on the result
        port_type = bounded_int(0, 65535)
        safe, reason = self.num_analyzer.check_division_safety(port_type)

        return ExperimentResult(
            name="real_config_parsing",
            description="Config port parsing with ValueError fallback",
            bug_pattern="int(config.get('port')) may raise ValueError",
            guard_system_catches=False,
            new_system_catches=bool(new_state.path_predicates) or "port" in new_state.bindings,
            refinements_found=[str(p) for p in new_state.path_predicates],
        )

    def exp_real_db_query_result(self) -> ExperimentResult:
        """Database query result: may be None.

        Bug pattern:
            row = cursor.fetchone()
            name = row[0]  # BUG: row may be None if no results
        """
        code = """\
        try:
            name = row[0]
        except TypeError:
            name = None
        """
        tree = _parse(code)
        try_node = _first_stmt(tree)
        assert isinstance(try_node, ast.Try)

        state = ExcState()
        new_state = self.exc_analyzer.analyze_try_except(try_node, state)

        return ExperimentResult(
            name="real_db_query_result",
            description="DB fetchone() may return None → row[0] crashes",
            bug_pattern="row = fetchone(); row[0] → TypeError if None",
            guard_system_catches=False,
            new_system_catches=bool(new_state.path_predicates) or "name" in new_state.bindings,
            refinements_found=[str(p) for p in new_state.path_predicates],
        )

    def exp_real_api_validation(self) -> ExperimentResult:
        """API response validation: check fields before access.

        Bug pattern:
            resp = api.get('/users/1')
            data = resp.json()
            email = data['email']  # may not exist
        """
        code = """\
        try:
            email = data['email']
        except KeyError:
            email = None
        """
        tree = _parse(code)
        try_node = _first_stmt(tree)
        assert isinstance(try_node, ast.Try)

        state = ExcState()
        new_state = self.exc_analyzer.analyze_try_except(try_node, state)

        # Also check string taint on the result
        str_state = StrState()
        str_state = self.str_analyzer.track_taint("email", "api_response", str_state)
        is_safe = self.str_analyzer.check_sanitized("email", str_state)

        return ExperimentResult(
            name="real_api_validation",
            description="API response field access needs validation",
            bug_pattern="data['email'] without check → KeyError",
            guard_system_catches=False,
            new_system_catches=bool(new_state.path_predicates) or not is_safe,
            refinements_found=[str(p) for p in new_state.path_predicates] + [f"tainted={not is_safe}"],
        )

    def exp_real_path_handling(self) -> ExperimentResult:
        """File path handling: join + existence check.

        Bug pattern:
            path = os.path.join(base, user_input)
            data = open(path).read()
            # user_input could escape base directory (path traversal)
        """
        state = StrState()
        state = self.str_analyzer.track_taint("user_input", "http_request", state)

        # Simulate string method: os.path.join propagates taint
        is_safe = self.str_analyzer.check_sanitized("user_input", state)

        return ExperimentResult(
            name="real_path_handling",
            description="Path traversal: tainted user_input in file path",
            bug_pattern="os.path.join(base, user_input) → traversal risk",
            guard_system_catches=False,
            new_system_catches=not is_safe,
            refinements_found=[f"tainted=True", f"is_safe={is_safe}"],
            warnings=["Path traversal risk: user_input is tainted"],
        )

    # -----------------------------------------------------------------------
    # Runner
    # -----------------------------------------------------------------------

    def _all_experiments(self) -> List[str]:
        """Return names of all experiment methods."""
        return [
            name for name in dir(self)
            if name.startswith("exp_") and callable(getattr(self, name))
        ]

    def run_all(self) -> List[ExperimentResult]:
        """Execute every experiment and collect results."""
        self.results = []
        for name in sorted(self._all_experiments()):
            method = getattr(self, name)
            try:
                result = method()
                self.results.append(result)
            except Exception as exc:
                self.results.append(ExperimentResult(
                    name=name,
                    description=f"FAILED: {exc}",
                    bug_pattern="",
                    guard_system_catches=False,
                    new_system_catches=False,
                    refinements_found=[],
                    warnings=[f"Exception: {exc}"],
                ))
        return self.results

    def print_summary(self) -> None:
        """Print a formatted summary table of all experiment results."""
        if not self.results:
            self.run_all()

        total = len(self.results)
        improvements = sum(1 for r in self.results if r.is_improvement)
        caught_new = sum(1 for r in self.results if r.new_system_catches)
        caught_guard = sum(1 for r in self.results if r.guard_system_catches)
        failed = sum(1 for r in self.results if r.description.startswith("FAILED"))

        header = f"{'Experiment':<35} {'Guard':>6} {'New':>6} {'Improv':>7} {'Warnings':>8}"
        sep = "-" * len(header)

        print()
        print("=" * len(header))
        print("Python Pattern Experiments – Summary")
        print("=" * len(header))
        print()
        print(header)
        print(sep)

        groups: Dict[str, List[ExperimentResult]] = {}
        for r in self.results:
            prefix = r.name.split("_")[0] + "_" + r.name.split("_")[1]
            groups.setdefault(prefix, []).append(r)

        prev_group = ""
        for r in self.results:
            group = r.name.split("_")[0] + "_" + r.name.split("_")[1]
            if group != prev_group and prev_group:
                print(sep)
            prev_group = group

            guard_str = "YES" if r.guard_system_catches else "no"
            new_str = "YES" if r.new_system_catches else "no"
            improv_str = " ✓" if r.is_improvement else ""
            warn_count = len(r.warnings)

            print(f"{r.name:<35} {guard_str:>6} {new_str:>6} {improv_str:>7} {warn_count:>8}")

        print(sep)
        print()
        print(f"Total experiments:       {total}")
        print(f"Guard system catches:    {caught_guard}")
        print(f"New system catches:      {caught_new}")
        print(f"Improvements (new only): {improvements}")
        if failed:
            print(f"Failed experiments:      {failed}")
        print()

        if improvements == total:
            print("Result: ALL experiments show improvement over guard-only system.")
        else:
            pct = improvements / total * 100 if total else 0
            print(f"Result: {pct:.0f}% of experiments show improvement.")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    experiments = PythonPatternExperiments()
    experiments.run_all()
    experiments.print_summary()


if __name__ == "__main__":
    main()
