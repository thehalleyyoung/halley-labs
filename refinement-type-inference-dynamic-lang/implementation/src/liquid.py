"""
Liquid Type Inference Engine for Python.

Harvests predicates from programmer-written code (guards, asserts, defaults,
exception handlers, return expressions, walrus operators, comprehension
filters) and uses Z3-backed refinement subtyping to infer liquid types
{v: T | φ} for all variables and function signatures.

Key differentiator from Pyright / Pytype / Mypy:
  1. Predicate inference FROM code, not annotations
  2. Z3-backed precision (prove safety or find counterexamples)
  3. Multi-source predicate harvesting
  4. Dependent function types
  5. CEGAR refinement loop
  6. Works on completely untyped code
"""

from __future__ import annotations

import ast
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Z3 import – REQUIRED for refinement subtyping
# ---------------------------------------------------------------------------

try:
    import z3 as _z3
    HAS_Z3 = True
except ImportError:
    raise ImportError(
        "Z3 is required for TensorGuard's refinement subtyping. "
        "Install it with: pip install z3-solver"
    )

# ---------------------------------------------------------------------------
# Internal imports from the existing refinement infrastructure
# ---------------------------------------------------------------------------

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
    NEVER_TYPE,
    OBJECT_TYPE,
    Z3Encoder,
    RefinementLattice,
    RefEnvironment,
    DepFuncType,
)
from src.real_analyzer import (
    FlowSensitiveAnalyzer,
    VarState,
    NullState,
    TypeTagSet,
    Interval,
    FunctionSummary,
    Bug,
    BugCategory,
    infer_function_summary,
    infer_file_summaries,
)


# ═══════════════════════════════════════════════════════════════════════════
# 0.  Liquid-specific bug wrapper
# ═══════════════════════════════════════════════════════════════════════════

class LiquidBugKind(Enum):
    """Bug categories discovered via liquid-type subtyping failures."""
    NULL_DEREF = auto()
    DIV_BY_ZERO = auto()
    INDEX_OOB = auto()
    TYPE_ERROR = auto()
    ATTRIBUTE_ERROR = auto()
    PRECONDITION_VIOLATION = auto()
    UNSAT_CONSTRAINT = auto()


@dataclass
class LiquidBug:
    """A bug found by the liquid type checker."""
    kind: LiquidBugKind
    line: int
    col: int
    message: str
    function: str
    variable: str
    actual_type: Optional[RefType] = None
    required_type: Optional[RefType] = None
    severity: str = "warning"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.name,
            "line": self.line,
            "col": self.col,
            "message": self.message,
            "function": self.function,
            "variable": self.variable,
            "severity": self.severity,
            "actual_type": self.actual_type.pretty() if self.actual_type else None,
            "required_type": self.required_type.pretty() if self.required_type else None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Predicate Harvester
# ═══════════════════════════════════════════════════════════════════════════

class HarvestSource(Enum):
    """Where a predicate was harvested from."""
    GUARD = "guard"
    ASSERT = "assert"
    DEFAULT = "default"
    EXCEPTION = "exception"
    RETURN = "return"
    WALRUS = "walrus"
    COMPREHENSION = "comprehension"
    CALL_SITE = "call_site"


@dataclass
class HarvestedPred:
    """A predicate together with its provenance."""
    pred: Pred
    source: HarvestSource
    line: int = 0
    variable: str = ""

    def __hash__(self):
        return hash((self.pred, self.source, self.variable))

    def __eq__(self, other):
        if not isinstance(other, HarvestedPred):
            return False
        return (self.pred == other.pred and self.source == other.source
                and self.variable == other.variable)


class PredicateHarvester(ast.NodeVisitor):
    """Harvests predicate templates from ALL sources in Python code.

    Sources:
      1. Guard conditions (isinstance, is None, comparisons, truthiness …)
      2. Assert statements
      3. Default values (dict.get, or-defaults, ternary defaults)
      4. Exception handlers (successful path refines type)
      5. Return expressions (conditional returns, ternary returns)
      6. Walrus operator (:=) in guards
      7. Comprehension filters ([x for x in xs if x is not None])
      8. Call-site inference
    """

    def __init__(self):
        self._harvested: List[HarvestedPred] = []
        self._current_func: str = "<module>"

    # -- public interface ---------------------------------------------------

    def harvest_all(self, func: ast.FunctionDef) -> List[HarvestedPred]:
        """Harvest predicates from all sources in *func*."""
        self._harvested = []
        self._current_func = getattr(func, "name", "<module>")

        preds: List[HarvestedPred] = []
        preds.extend(self.harvest_guards(func))
        preds.extend(self.harvest_asserts(func))
        preds.extend(self.harvest_defaults(func))
        preds.extend(self.harvest_exceptions(func))
        preds.extend(self.harvest_returns(func))
        preds.extend(self.harvest_walrus(func))
        preds.extend(self.harvest_comprehension_filters(func))
        return preds

    def harvest_module(self, tree: ast.Module) -> List[HarvestedPred]:
        """Harvest predicates from every function in a module."""
        all_preds: List[HarvestedPred] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                all_preds.extend(self.harvest_all(node))
        return all_preds

    # -- guard harvesting ---------------------------------------------------

    def harvest_guards(self, func: ast.FunctionDef) -> List[HarvestedPred]:
        """Harvest predicates from if/elif/while conditions.

        Special handling for early-exit guards: if the true branch always
        terminates (raise/return), harvest the NEGATED condition as a predicate
        for the continuation.  E.g. ``if y == 0: raise`` → harvest y ≠ 0.
        """
        results: List[HarvestedPred] = []
        for node in ast.walk(func):
            if isinstance(node, (ast.If, ast.While)):
                preds = self._extract_condition_preds(node.test, getattr(node, "lineno", 0))
                results.extend(preds)

                # Early-exit guard: if true branch terminates, the negated
                # condition holds on the continuation path.
                if isinstance(node, ast.If) and self._branch_always_exits(node.body):
                    neg_preds = self._extract_condition_preds(node.test, getattr(node, "lineno", 0))
                    for hp in neg_preds:
                        hp.pred = hp.pred.not_()
                    results.extend(neg_preds)
        return results

    @staticmethod
    def _branch_always_exits(stmts: list) -> bool:
        """Check if a list of statements always exits (raise/return/break/continue)."""
        if not stmts:
            return False
        last = stmts[-1]
        if isinstance(last, (ast.Raise, ast.Return, ast.Break, ast.Continue)):
            return True
        if isinstance(last, ast.If) and last.orelse:
            return (PredicateHarvester._branch_always_exits(last.body) and
                    PredicateHarvester._branch_always_exits(last.orelse))
        return False

    # -- assert harvesting --------------------------------------------------

    def harvest_asserts(self, func: ast.FunctionDef) -> List[HarvestedPred]:
        """Harvest predicates from assert statements."""
        results: List[HarvestedPred] = []
        for node in ast.walk(func):
            if isinstance(node, ast.Assert):
                preds = self._extract_condition_preds(
                    node.test, getattr(node, "lineno", 0)
                )
                for hp in preds:
                    hp.source = HarvestSource.ASSERT
                results.extend(preds)
        return results

    # -- default-value harvesting -------------------------------------------

    def harvest_defaults(self, func: ast.FunctionDef) -> List[HarvestedPred]:
        """Harvest predicates from default values.

        Patterns:
          - x = d.get(key, default)  → result has_default (but NOT necessarily
            not-None, since d[key] could be None even with a non-None default)
          - x = a or b               → if b is non-None, result is not None
          - x = a if a is not None else b  → result is not None
        """
        results: List[HarvestedPred] = []
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            if not node.targets or not isinstance(node.targets[0], ast.Name):
                continue
            var = node.targets[0].id
            line = node.lineno

            # dict.get(key, default) where default is not None:
            # NOTE: This does NOT guarantee result is not None, because
            # the dict may contain an explicit None value for the key.
            # d.get(key, default) returns d[key] if key in d, else default.
            # If d[key] is None, the result is None regardless of default.
            # We do NOT harvest is_not_none here — that would be unsound.
            if isinstance(node.value, ast.Call):
                if (isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == "get"
                        and len(node.value.args) >= 2):
                    # Intentionally do NOT infer is_not_none — unsound.
                    pass

            # x = a or b  →  result is not None (if b is truthy constant)
            if isinstance(node.value, ast.BoolOp) and isinstance(node.value.op, ast.Or):
                last = node.value.values[-1]
                if isinstance(last, ast.Constant) and last.value is not None:
                    hp = HarvestedPred(
                        pred=Pred.is_not_none(var),
                        source=HarvestSource.DEFAULT,
                        line=line,
                        variable=var,
                    )
                    results.append(hp)

            # x = a if a is not None else b  →  result is not None
            if isinstance(node.value, ast.IfExp):
                test = node.value.test
                if self._is_not_none_check(test):
                    orelse = node.value.orelse
                    if not self._is_none_literal(orelse):
                        hp = HarvestedPred(
                            pred=Pred.is_not_none(var),
                            source=HarvestSource.DEFAULT,
                            line=line,
                            variable=var,
                        )
                        results.append(hp)

        return results

    # -- exception harvesting -----------------------------------------------

    def harvest_exceptions(self, func: ast.FunctionDef) -> List[HarvestedPred]:
        """Harvest predicates from try/except patterns.

        Pattern: try: x.method() except AttributeError: …
          → on the success path, x has the attribute (hasattr).
        Pattern: try: int(x) except (ValueError, TypeError): …
          → on the success path, x is convertible to int.
        """
        results: List[HarvestedPred] = []
        for node in ast.walk(func):
            if not isinstance(node, ast.Try):
                continue

            caught_types = set()
            for handler in node.handlers:
                if handler.type is None:
                    caught_types.add("BaseException")
                elif isinstance(handler.type, ast.Name):
                    caught_types.add(handler.type.id)
                elif isinstance(handler.type, ast.Tuple):
                    for elt in handler.type.elts:
                        if isinstance(elt, ast.Name):
                            caught_types.add(elt.id)

            # Walk the try-body looking for attribute accesses
            if "AttributeError" in caught_types:
                for try_node in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                    if isinstance(try_node, ast.Attribute):
                        if isinstance(try_node.value, ast.Name):
                            var = try_node.value.id
                            attr = try_node.attr
                            hp = HarvestedPred(
                                pred=Pred.hasattr_(var, attr),
                                source=HarvestSource.EXCEPTION,
                                line=getattr(try_node, "lineno", node.lineno),
                                variable=var,
                            )
                            results.append(hp)

            # TypeError/ValueError in handlers → isinstance on try-body vars
            if "TypeError" in caught_types or "ValueError" in caught_types:
                for try_node in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                    if isinstance(try_node, ast.Call):
                        if isinstance(try_node.func, ast.Name):
                            if try_node.func.id in ("int", "float", "str"):
                                for arg in try_node.args:
                                    if isinstance(arg, ast.Name):
                                        hp = HarvestedPred(
                                            pred=Pred.isinstance_(arg.id, try_node.func.id),
                                            source=HarvestSource.EXCEPTION,
                                            line=getattr(try_node, "lineno", node.lineno),
                                            variable=arg.id,
                                        )
                                        results.append(hp)

            # KeyError → dict access succeeded
            if "KeyError" in caught_types:
                for try_node in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                    if isinstance(try_node, ast.Subscript):
                        if isinstance(try_node.value, ast.Name):
                            var = try_node.value.id
                            hp = HarvestedPred(
                                pred=Pred.is_not_none(var),
                                source=HarvestSource.EXCEPTION,
                                line=getattr(try_node, "lineno", node.lineno),
                                variable=var,
                            )
                            results.append(hp)

        return results

    # -- return-value harvesting --------------------------------------------

    def harvest_returns(self, func: ast.FunctionDef) -> List[HarvestedPred]:
        """Harvest predicates from return expressions.

        Pattern: return x if x is not None else default  →  return is not None
        Pattern: return x or default                      →  return is not None
        """
        results: List[HarvestedPred] = []
        for node in ast.walk(func):
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            line = node.lineno

            # Ternary return
            if isinstance(node.value, ast.IfExp):
                test = node.value.test
                if self._is_not_none_check(test):
                    orelse = node.value.orelse
                    if not self._is_none_literal(orelse):
                        hp = HarvestedPred(
                            pred=Pred.is_not_none("__return__"),
                            source=HarvestSource.RETURN,
                            line=line,
                            variable="__return__",
                        )
                        results.append(hp)

            # or-default return
            if isinstance(node.value, ast.BoolOp) and isinstance(node.value.op, ast.Or):
                last = node.value.values[-1]
                if isinstance(last, ast.Constant) and last.value is not None:
                    hp = HarvestedPred(
                        pred=Pred.is_not_none("__return__"),
                        source=HarvestSource.RETURN,
                        line=line,
                        variable="__return__",
                    )
                    results.append(hp)

            # Constant return (non-None)
            if isinstance(node.value, ast.Constant) and node.value.value is not None:
                hp = HarvestedPred(
                    pred=Pred.is_not_none("__return__"),
                    source=HarvestSource.RETURN,
                    line=line,
                    variable="__return__",
                )
                results.append(hp)

        return results

    # -- walrus harvesting --------------------------------------------------

    def harvest_walrus(self, func: ast.FunctionDef) -> List[HarvestedPred]:
        """Harvest predicates from walrus operator (:=) in guards.

        Pattern: if (m := re.match(...)): body  →  m is not None in body
        Pattern: if (n := len(xs)) > 0:          →  n > 0 in body
        """
        results: List[HarvestedPred] = []
        for node in ast.walk(func):
            if not isinstance(node, (ast.If, ast.While)):
                continue
            walrus_vars = self._find_walrus_in_test(node.test)
            for var_name, namedexpr in walrus_vars:
                line = getattr(namedexpr, "lineno", node.lineno)
                # The walrus target is being truthiness-tested
                hp = HarvestedPred(
                    pred=Pred.is_not_none(var_name),
                    source=HarvestSource.WALRUS,
                    line=line,
                    variable=var_name,
                )
                results.append(hp)

                # If the walrus appears in a comparison, harvest that too
                cmp_preds = self._extract_condition_preds(node.test, line)
                for cp in cmp_preds:
                    cp.source = HarvestSource.WALRUS
                results.extend(cmp_preds)
        return results

    # -- comprehension filter harvesting ------------------------------------

    def harvest_comprehension_filters(self, func: ast.FunctionDef) -> List[HarvestedPred]:
        """Harvest predicates from comprehension/generator filters.

        Pattern: [x for x in items if x is not None]
          → result elements satisfy 'is not None'
        Pattern: [x for x in items if isinstance(x, int)]
          → result elements satisfy isinstance(_, int)
        """
        results: List[HarvestedPred] = []
        for node in ast.walk(func):
            if not isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                continue
            for comp in node.generators:
                for if_clause in comp.ifs:
                    preds = self._extract_condition_preds(
                        if_clause, getattr(if_clause, "lineno", getattr(node, "lineno", 0))
                    )
                    for hp in preds:
                        hp.source = HarvestSource.COMPREHENSION
                    results.extend(preds)
        return results

    # -- call-site inference ------------------------------------------------

    def harvest_call_sites(self, tree: ast.Module) -> Dict[str, List[HarvestedPred]]:
        """For each function, collect predicates that hold at ALL call sites.

        If f(x) is always called with x > 0, infer precondition x > 0 for f.
        Returns a mapping func_name → list of harvested predicates.
        """
        # Phase 1: collect guard context at each call site
        call_contexts: Dict[str, List[List[HarvestedPred]]] = {}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._collect_call_site_preds(node, call_contexts)

        # Phase 2: intersect — keep predicates that appear at ALL call sites
        result: Dict[str, List[HarvestedPred]] = {}
        for fname, contexts in call_contexts.items():
            if not contexts:
                continue
            # Convert to predicate sets for intersection
            common_preds = set(hp.pred for hp in contexts[0])
            for ctx in contexts[1:]:
                common_preds &= set(hp.pred for hp in ctx)
            if common_preds:
                result[fname] = [
                    HarvestedPred(pred=p, source=HarvestSource.CALL_SITE, variable=fname)
                    for p in common_preds
                ]
        return result

    # ── Internal helpers ───────────────────────────────────────────────

    def _extract_condition_preds(
        self, test: ast.expr, line: int
    ) -> List[HarvestedPred]:
        """Extract predicates from a boolean expression (condition)."""
        results: List[HarvestedPred] = []

        # BoolOp: and / or
        if isinstance(test, ast.BoolOp):
            for val in test.values:
                results.extend(self._extract_condition_preds(val, line))
            return results

        # UnaryOp: not
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = self._extract_condition_preds(test.operand, line)
            for hp in inner:
                hp.pred = hp.pred.not_()
            return inner

        # Compare: x is None, x is not None, x < 5, isinstance(x, T), ...
        if isinstance(test, ast.Compare):
            return self._extract_compare_preds(test, line)

        # Call: isinstance(x, T), hasattr(x, 'a'), callable(x)
        if isinstance(test, ast.Call):
            return self._extract_call_preds(test, line)

        # Name: truthiness guard  (if x: ...)
        if isinstance(test, ast.Name):
            hp = HarvestedPred(
                pred=Pred.truthy(test.id),
                source=HarvestSource.GUARD,
                line=line,
                variable=test.id,
            )
            return [hp]

        # Attribute: if obj.attr:  → truthiness of obj.attr
        if isinstance(test, ast.Attribute):
            if isinstance(test.value, ast.Name):
                # Treat as truthiness of the result
                hp = HarvestedPred(
                    pred=Pred.is_not_none(test.value.id),
                    source=HarvestSource.GUARD,
                    line=line,
                    variable=test.value.id,
                )
                return [hp]

        # NamedExpr (walrus): if (m := expr):
        if isinstance(test, ast.NamedExpr):
            hp = HarvestedPred(
                pred=Pred.is_not_none(test.target.id),
                source=HarvestSource.WALRUS,
                line=line,
                variable=test.target.id,
            )
            return [hp]

        return results

    def _extract_compare_preds(
        self, node: ast.Compare, line: int
    ) -> List[HarvestedPred]:
        """Extract predicates from a Compare node."""
        results: List[HarvestedPred] = []
        left = node.left

        for op, comparator in zip(node.ops, node.comparators):
            # x is None
            if isinstance(op, ast.Is) and self._is_none_literal(comparator):
                if isinstance(left, ast.Name):
                    results.append(HarvestedPred(
                        pred=Pred.is_none(left.id),
                        source=HarvestSource.GUARD, line=line, variable=left.id,
                    ))
            # x is not None
            elif isinstance(op, ast.IsNot) and self._is_none_literal(comparator):
                if isinstance(left, ast.Name):
                    results.append(HarvestedPred(
                        pred=Pred.is_not_none(left.id),
                        source=HarvestSource.GUARD, line=line, variable=left.id,
                    ))
            # x < 5, x >= 0, etc.
            elif isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
                if isinstance(left, ast.Name) and isinstance(comparator, ast.Constant):
                    val = comparator.value
                    if isinstance(val, (int, float)):
                        int_val = int(val)
                        op_map = {
                            ast.Lt: PredOp.LT, ast.LtE: PredOp.LE,
                            ast.Gt: PredOp.GT, ast.GtE: PredOp.GE,
                            ast.Eq: PredOp.EQ, ast.NotEq: PredOp.NEQ,
                        }
                        pred = Pred(op_map[type(op)], (left.id, int_val))
                        results.append(HarvestedPred(
                            pred=pred, source=HarvestSource.GUARD,
                            line=line, variable=left.id,
                        ))
                # 0 < x  →  x > 0
                elif isinstance(comparator, ast.Name) and isinstance(left, ast.Constant):
                    val = left.value
                    if isinstance(val, (int, float)):
                        int_val = int(val)
                        flip = {
                            ast.Lt: PredOp.GT, ast.LtE: PredOp.GE,
                            ast.Gt: PredOp.LT, ast.GtE: PredOp.LE,
                            ast.Eq: PredOp.EQ, ast.NotEq: PredOp.NEQ,
                        }
                        pred = Pred(flip[type(op)], (comparator.id, int_val))
                        results.append(HarvestedPred(
                            pred=pred, source=HarvestSource.GUARD,
                            line=line, variable=comparator.id,
                        ))

            # isinstance check in a comparison: isinstance(x, T) == True
            # (rare but handled)

            left = comparator  # chained comparisons
        return results

    def _extract_call_preds(
        self, node: ast.Call, line: int
    ) -> List[HarvestedPred]:
        """Extract predicates from a Call node used as a condition."""
        results: List[HarvestedPred] = []

        if isinstance(node.func, ast.Name):
            fname = node.func.id

            # isinstance(x, T)
            if fname == "isinstance" and len(node.args) >= 2:
                target = node.args[0]
                type_arg = node.args[1]
                if isinstance(target, ast.Name):
                    type_name = self._type_node_to_str(type_arg)
                    if type_name:
                        results.append(HarvestedPred(
                            pred=Pred.isinstance_(target.id, type_name),
                            source=HarvestSource.GUARD, line=line,
                            variable=target.id,
                        ))

            # hasattr(x, 'attr')
            elif fname == "hasattr" and len(node.args) >= 2:
                target = node.args[0]
                attr_arg = node.args[1]
                if isinstance(target, ast.Name) and isinstance(attr_arg, ast.Constant):
                    results.append(HarvestedPred(
                        pred=Pred.hasattr_(target.id, str(attr_arg.value)),
                        source=HarvestSource.GUARD, line=line,
                        variable=target.id,
                    ))

            # callable(x)
            elif fname == "callable" and len(node.args) >= 1:
                target = node.args[0]
                if isinstance(target, ast.Name):
                    results.append(HarvestedPred(
                        pred=Pred.hasattr_(target.id, "__call__"),
                        source=HarvestSource.GUARD, line=line,
                        variable=target.id,
                    ))

            # len(x) > 0  is handled at Compare level; len(x) alone → truthy
            elif fname == "len" and len(node.args) >= 1:
                target = node.args[0]
                if isinstance(target, ast.Name):
                    results.append(HarvestedPred(
                        pred=Pred.len_gt(target.id, 0),
                        source=HarvestSource.GUARD, line=line,
                        variable=target.id,
                    ))

        return results

    def _find_walrus_in_test(
        self, test: ast.expr
    ) -> List[Tuple[str, ast.NamedExpr]]:
        """Find all walrus (:=) expressions in a condition."""
        results: List[Tuple[str, ast.NamedExpr]] = []
        for node in ast.walk(test):
            if isinstance(node, ast.NamedExpr):
                if isinstance(node.target, ast.Name):
                    results.append((node.target.id, node))
        return results

    def _collect_call_site_preds(
        self,
        func: ast.FunctionDef,
        call_contexts: Dict[str, List[List[HarvestedPred]]],
    ) -> None:
        """Walk *func* and for each call f(args), record the predicates
        that hold at that call site (from the enclosing guard context)."""
        # Simple approach: track guard context as we walk if-statements
        for node in ast.walk(func):
            if isinstance(node, ast.If):
                guard_preds = self._extract_condition_preds(
                    node.test, getattr(node, "lineno", 0)
                )
                for stmt in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                    if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Name):
                        fname = stmt.func.id
                        if fname not in call_contexts:
                            call_contexts[fname] = []
                        call_contexts[fname].append(guard_preds)

    # ── Utility ────────────────────────────────────────────────────────

    @staticmethod
    def _is_none_literal(node: ast.expr) -> bool:
        if isinstance(node, ast.Constant) and node.value is None:
            return True
        if isinstance(node, ast.Name) and node.id == "None":
            return True
        return False

    @staticmethod
    def _is_not_none_check(test: ast.expr) -> bool:
        """Check if *test* is of the form  `x is not None`."""
        if isinstance(test, ast.Compare):
            if len(test.ops) == 1 and isinstance(test.ops[0], ast.IsNot):
                cmp = test.comparators[0]
                return PredicateHarvester._is_none_literal(cmp)
        return False

    @staticmethod
    def _type_node_to_str(node: ast.expr) -> Optional[str]:
        """Convert a type argument in isinstance to a string."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts = []
            cur = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            return ".".join(reversed(parts))
        if isinstance(node, ast.Tuple):
            # isinstance(x, (int, str)) – pick first for simplicity
            if node.elts:
                return PredicateHarvester._type_node_to_str(node.elts[0])
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Constraint Language
# ═══════════════════════════════════════════════════════════════════════════

class ConstraintKind(Enum):
    SUBTYPE = auto()        # actual <: required
    EQUALITY = auto()       # actual == required
    WELL_FORMED = auto()    # type must be non-bottom


@dataclass
class LiquidConstraint:
    """A subtyping constraint: actual_type <: required_type at a program point."""
    kind: ConstraintKind
    actual: RefType
    required: RefType
    line: int = 0
    col: int = 0
    context: str = ""     # e.g. "division denominator"
    variable: str = ""
    function: str = ""

    def pretty(self) -> str:
        op = {"SUBTYPE": "<:", "EQUALITY": "≡", "WELL_FORMED": "wf"}[self.kind.name]
        return f"[L{self.line}] {self.actual.pretty()} {op} {self.required.pretty()}"


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Function Contract
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FunctionContract:
    """Inferred liquid type contract for a function.

    E.g.: def div(x: {v:int|true}, y: {v:int|v≠0}) -> {v:int|true}
    """
    name: str
    params: Dict[str, RefType] = field(default_factory=dict)
    return_type: RefType = field(default_factory=lambda: RefType.trivial(ANY_TYPE))
    preconditions: List[Pred] = field(default_factory=list)
    postconditions: List[Pred] = field(default_factory=list)
    line: int = 0

    def to_dep_func_type(self) -> DepFuncType:
        """Convert to a DepFuncType for interprocedural use."""
        param_list = tuple((n, t) for n, t in self.params.items())
        return DepFuncType(params=param_list, ret=self.return_type)

    def pretty(self) -> str:
        params_str = ", ".join(f"{n}: {t.pretty()}" for n, t in self.params.items())
        ret_str = self.return_type.pretty()
        pre = ""
        if self.preconditions:
            pre = "  requires " + " ∧ ".join(p.pretty() for p in self.preconditions) + "\n"
        post = ""
        if self.postconditions:
            post = "  ensures " + " ∧ ".join(p.pretty() for p in self.postconditions) + "\n"
        return f"def {self.name}({params_str}) -> {ret_str}\n{pre}{post}"

    def annotated_signature(self) -> str:
        """Produce a typing.Annotated-style signature string."""
        parts = []
        for name, rt in self.params.items():
            if rt.pred.op == PredOp.TRUE:
                parts.append(f"{name}: {rt.base.pretty()}")
            else:
                parts.append(f"{name}: Annotated[{rt.base.pretty()}, '{rt.pred.pretty()}']")
        ret = self.return_type
        if ret.pred.op == PredOp.TRUE:
            ret_str = ret.base.pretty()
        else:
            ret_str = f"Annotated[{ret.base.pretty()}, '{ret.pred.pretty()}']"
        return f"def {self.name}({', '.join(parts)}) -> {ret_str}"


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Analysis Result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LiquidAnalysisResult:
    """Results from liquid type analysis of a module."""
    contracts: Dict[str, FunctionContract] = field(default_factory=dict)
    bugs: List[LiquidBug] = field(default_factory=list)
    predicates_harvested: int = 0
    constraints_generated: int = 0
    constraints_solved: int = 0
    cegar_iterations: int = 0
    analysis_time_ms: float = 0.0
    harvested_preds: List[HarvestedPred] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Liquid Type Analysis: {len(self.contracts)} functions, "
            f"{self.predicates_harvested} predicates harvested, "
            f"{self.constraints_generated} constraints generated, "
            f"{self.constraints_solved} solved, "
            f"{self.cegar_iterations} CEGAR iterations, "
            f"{len(self.bugs)} bugs found, "
            f"{self.analysis_time_ms:.1f}ms"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Type Inference Helpers
# ═══════════════════════════════════════════════════════════════════════════

# Mapping from Python type names to BaseTypeR
_TYPE_MAP: Dict[str, BaseTypeR] = {
    "int": INT_TYPE,
    "float": FLOAT_TYPE,
    "str": STR_TYPE,
    "bool": BOOL_TYPE,
    "None": NONE_TYPE,
    "NoneType": NONE_TYPE,
    "list": BaseTypeR(BaseTypeKind.LIST),
    "dict": BaseTypeR(BaseTypeKind.DICT),
    "set": BaseTypeR(BaseTypeKind.SET),
    "tuple": BaseTypeR(BaseTypeKind.TUPLE),
    "object": OBJECT_TYPE,
}


def _base_type_from_constant(val: Any) -> BaseTypeR:
    """Infer a BaseTypeR from a Python constant value."""
    if val is None:
        return NONE_TYPE
    if isinstance(val, bool):
        return BOOL_TYPE
    if isinstance(val, int):
        return INT_TYPE
    if isinstance(val, float):
        return FLOAT_TYPE
    if isinstance(val, str):
        return STR_TYPE
    return ANY_TYPE


def _base_type_from_annotation(ann: ast.expr) -> BaseTypeR:
    """Convert an AST annotation node to a BaseTypeR."""
    if isinstance(ann, ast.Name):
        return _TYPE_MAP.get(ann.id, ANY_TYPE)
    if isinstance(ann, ast.Constant):
        if ann.value is None:
            return NONE_TYPE
    if isinstance(ann, ast.Attribute):
        return ANY_TYPE
    # Optional[X] etc. – simplify
    if isinstance(ann, ast.Subscript):
        if isinstance(ann.value, ast.Name):
            if ann.value.id == "Optional":
                # Optional[X] → treat as X for base
                return _base_type_from_annotation(ann.slice)
        return ANY_TYPE
    return ANY_TYPE


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Constraint Generator
# ═══════════════════════════════════════════════════════════════════════════

class ConstraintGenerator(ast.NodeVisitor):
    """Walk the AST and generate subtyping constraints at every assignment
    and use site.

    The generator maintains a *refinement environment* mapping each variable
    to its current refinement type, updated flow-sensitively through
    if-branches and loops.
    """

    def __init__(
        self,
        func: ast.FunctionDef,
        predicates: List[HarvestedPred],
        lattice: RefinementLattice,
        func_contracts: Dict[str, FunctionContract],
    ):
        self.func = func
        self.predicates = predicates
        self.lattice = lattice
        self.func_contracts = func_contracts

        self.constraints: List[LiquidConstraint] = []
        self.env = RefEnvironment()
        self.func_name = func.name

        # Path predicate (conjunction of guards on the current path)
        self.path_pred = Pred.true_()

        # Return types collected
        self.return_types: List[RefType] = []

        # Inferred return predicates
        self.return_preds: List[Pred] = []

        # Initialize parameter types
        self._init_params()

    def _init_params(self):
        """Initialize parameter refinement types from annotations or usage-based inference."""
        # First pass: infer base types from usage context
        inferred_bases = self._infer_param_base_types()
        for arg in self.func.args.args:
            name = arg.arg
            if arg.annotation:
                base = _base_type_from_annotation(arg.annotation)
            else:
                base = inferred_bases.get(name, ANY_TYPE)
            self.env = self.env.set(name, RefType.trivial(base, name))

    def _infer_param_base_types(self) -> Dict[str, BaseTypeR]:
        """Infer base types for unannotated parameters from usage patterns.

        Heuristics:
          - Used in arithmetic (+ - * / //) → int or float
          - Compared to int constant → int
          - Compared to None (is None) → keep Any (nullable)
          - Used in len() → collection
          - Used with .method() → object
          - isinstance check → the checked type
        """
        param_names = {arg.arg for arg in self.func.args.args if not arg.annotation}
        inferred: Dict[str, BaseTypeR] = {}

        for node in ast.walk(self.func):
            # Arithmetic: x + y, x * y, x / y → numeric
            if isinstance(node, ast.BinOp):
                for side in (node.left, node.right):
                    if isinstance(side, ast.Name) and side.id in param_names:
                        if side.id not in inferred:
                            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                                inferred[side.id] = INT_TYPE
                            elif isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Mod, ast.Pow)):
                                inferred[side.id] = INT_TYPE

            # Comparison to int constant: x == 0, x > 5 → int
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
                var = node.left.id
                if var in param_names:
                    for comp in node.comparators:
                        if isinstance(comp, ast.Constant):
                            if isinstance(comp.value, int):
                                inferred[var] = INT_TYPE
                            elif isinstance(comp.value, float):
                                inferred[var] = FLOAT_TYPE
                            elif isinstance(comp.value, str):
                                inferred[var] = STR_TYPE

            # isinstance check
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "isinstance" and len(node.args) >= 2):
                if isinstance(node.args[0], ast.Name) and node.args[0].id in param_names:
                    tname = PredicateHarvester._type_node_to_str(node.args[1])
                    if tname and tname in _TYPE_MAP:
                        inferred[node.args[0].id] = _TYPE_MAP[tname]

        return inferred

    # -- public interface ---------------------------------------------------

    def generate(self) -> List[LiquidConstraint]:
        """Generate all constraints for the function body."""
        for stmt in self.func.body:
            self._gen_stmt(stmt)
        return self.constraints

    # -- statement processing -----------------------------------------------

    def _gen_stmt(self, node: ast.stmt):
        """Generate constraints for a single statement."""
        if isinstance(node, ast.Assign):
            self._gen_assign(node)
        elif isinstance(node, ast.AnnAssign):
            self._gen_ann_assign(node)
        elif isinstance(node, ast.Return):
            self._gen_return(node)
        elif isinstance(node, ast.If):
            self._gen_if(node)
        elif isinstance(node, ast.While):
            self._gen_while(node)
        elif isinstance(node, ast.For):
            self._gen_for(node)
        elif isinstance(node, ast.Expr):
            self._gen_expr_stmt(node)
        elif isinstance(node, ast.Assert):
            self._gen_assert(node)
        elif isinstance(node, ast.Try):
            self._gen_try(node)
        elif isinstance(node, ast.With):
            for s in node.body:
                self._gen_stmt(s)
        elif isinstance(node, ast.AugAssign):
            self._gen_aug_assign(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pass  # skip nested
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass,
                                ast.Break, ast.Continue, ast.Raise,
                                ast.Global, ast.Nonlocal, ast.Delete)):
            pass

    def _gen_assign(self, node: ast.Assign):
        # Generate use-site constraints for the RHS before assignment
        self._gen_use_constraints(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                rhs_type = self._infer_expr_type(node.value)
                self.env = self.env.set(target.id, rhs_type)

    def _gen_ann_assign(self, node: ast.AnnAssign):
        if isinstance(node.target, ast.Name) and node.value is not None:
            rhs_type = self._infer_expr_type(node.value)
            ann_base = _base_type_from_annotation(node.annotation)
            ann_type = RefType.trivial(ann_base, node.target.id)
            self.constraints.append(LiquidConstraint(
                kind=ConstraintKind.SUBTYPE,
                actual=rhs_type,
                required=ann_type,
                line=node.lineno,
                variable=node.target.id,
                function=self.func_name,
                context="annotated assignment",
            ))
            self.env = self.env.set(node.target.id, rhs_type)

    def _gen_return(self, node: ast.Return):
        if node.value is not None:
            self._gen_use_constraints(node.value)
            ret_type = self._infer_expr_type(node.value)
            self.return_types.append(ret_type)
            # Collect path-conditioned return predicate
            if self.path_pred.op != PredOp.TRUE:
                self.return_preds.append(self.path_pred)

    @staticmethod
    def _branch_terminates(stmts: List[ast.stmt]) -> bool:
        """Check if a branch always terminates (raise/return/break/continue)."""
        if not stmts:
            return False
        last = stmts[-1]
        if isinstance(last, (ast.Raise, ast.Return, ast.Break, ast.Continue)):
            return True
        # if-else where both branches terminate
        if isinstance(last, ast.If) and last.orelse:
            return (ConstraintGenerator._branch_terminates(last.body) and
                    ConstraintGenerator._branch_terminates(last.orelse))
        return False

    def _gen_if(self, node: ast.If):
        guard_preds = self._condition_to_pred(node.test)
        old_env = RefEnvironment(dict(self.env.bindings))
        old_path = self.path_pred

        true_terminates = self._branch_terminates(node.body)
        false_terminates = self._branch_terminates(node.orelse)

        # True branch
        true_pred = guard_preds if guard_preds else Pred.true_()
        self.path_pred = old_path.and_(true_pred)
        true_env = self._narrow_env(self.env, node.test, positive=True)
        self.env = true_env
        for stmt in node.body:
            self._gen_stmt(stmt)
        true_env_after = RefEnvironment(dict(self.env.bindings))

        # False branch
        false_pred = guard_preds.not_() if guard_preds else Pred.true_()
        self.path_pred = old_path.and_(false_pred)
        false_env = self._narrow_env(old_env, node.test, positive=False)
        self.env = false_env
        for stmt in node.orelse:
            self._gen_stmt(stmt)
        false_env_after = RefEnvironment(dict(self.env.bindings))

        # Join: if one branch terminates, only the other is live
        if true_terminates and not false_terminates:
            self.env = false_env_after
        elif false_terminates and not true_terminates:
            self.env = true_env_after
        else:
            self.env = true_env_after.join(false_env_after, self.lattice)
        self.path_pred = old_path

    def _gen_while(self, node: ast.While):
        # Simple: process body once (sound over-approximation)
        guard_preds = self._condition_to_pred(node.test)
        old_path = self.path_pred
        if guard_preds:
            self.path_pred = old_path.and_(guard_preds)
        for stmt in node.body:
            self._gen_stmt(stmt)
        self.path_pred = old_path
        for stmt in node.orelse:
            self._gen_stmt(stmt)

    def _gen_for(self, node: ast.For):
        if isinstance(node.target, ast.Name):
            iter_type = self._infer_expr_type(node.iter)
            # Element type is approximated as ANY
            elem_type = RefType.trivial(ANY_TYPE, node.target.id)
            self.env = self.env.set(node.target.id, elem_type)
        for stmt in node.body:
            self._gen_stmt(stmt)
        for stmt in node.orelse:
            self._gen_stmt(stmt)

    def _gen_expr_stmt(self, node: ast.Expr):
        """Generate constraints for expression statements (dereferences, calls)."""
        self._gen_use_constraints(node.value)

    def _gen_assert(self, node: ast.Assert):
        """Assert narrows the environment on the true path."""
        guard_pred = self._condition_to_pred(node.test)
        if guard_pred:
            self.path_pred = self.path_pred.and_(guard_pred)
            self.env = self._narrow_env(self.env, node.test, positive=True)

    def _gen_try(self, node: ast.Try):
        old_env = RefEnvironment(dict(self.env.bindings))
        for stmt in node.body:
            self._gen_stmt(stmt)
        body_env = RefEnvironment(dict(self.env.bindings))

        for handler in node.handlers:
            self.env = RefEnvironment(dict(old_env.bindings))
            if handler.name:
                exc_type = RefType.trivial(ANY_TYPE, handler.name)
                self.env = self.env.set(handler.name, exc_type)
            for stmt in handler.body:
                self._gen_stmt(stmt)
            body_env = body_env.join(self.env, self.lattice)

        self.env = body_env
        for stmt in node.orelse:
            self._gen_stmt(stmt)
        for stmt in node.finalbody:
            self._gen_stmt(stmt)

    def _gen_aug_assign(self, node: ast.AugAssign):
        if isinstance(node.target, ast.Name):
            current = self.env.get(node.target.id)
            rhs = self._infer_expr_type(node.value)
            if current is not None:
                result_base = current.base if current.base.kind != BaseTypeKind.ANY else rhs.base
            else:
                result_base = rhs.base
            self.env = self.env.set(
                node.target.id,
                RefType.trivial(result_base, node.target.id),
            )

    # -- use-site constraint generation -------------------------------------

    def _gen_use_constraints(self, node: ast.expr):
        """At dereference/division/subscript sites, generate safety constraints."""

        # Attribute access: x.attr  →  require x is not None
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                var = node.value.id
                actual = self.env.get(var)
                if actual is not None:
                    # Use actual's base type so base-type subtyping succeeds
                    required = RefType(
                        "ν", actual.base,
                        Pred.is_not_none("ν"),
                    )
                    self.constraints.append(LiquidConstraint(
                        kind=ConstraintKind.SUBTYPE,
                        actual=actual.alpha_rename("ν"),
                        required=required,
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        context="attribute access",
                        variable=var,
                        function=self.func_name,
                    ))

        # Division: a / b  →  require b ≠ 0
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            right = node.right
            if isinstance(right, ast.Name):
                var = right.id
                actual = self.env.get(var)
                if actual is not None:
                    # Match base types: if actual is ANY, use ANY for required too
                    req_base = actual.base
                    required = RefType("ν", req_base, Pred.var_neq("ν", 0))
                    self.constraints.append(LiquidConstraint(
                        kind=ConstraintKind.SUBTYPE,
                        actual=actual.alpha_rename("ν"),
                        required=required,
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        context="division denominator",
                        variable=var,
                        function=self.func_name,
                    ))

        # Subscript: a[i]  →  require i ≥ 0 (simple bound check)
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Name):
                var = node.slice.id
                actual = self.env.get(var)
                if actual is not None:
                    req_base = actual.base
                    required = RefType("ν", req_base, Pred.var_ge("ν", 0))
                    self.constraints.append(LiquidConstraint(
                        kind=ConstraintKind.SUBTYPE,
                        actual=actual.alpha_rename("ν"),
                        required=required,
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        context="subscript index",
                        variable=var,
                        function=self.func_name,
                    ))

        # Call: f(args) → check preconditions if contract known
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                fname = node.func.id
                if fname in self.func_contracts:
                    contract = self.func_contracts[fname]
                    param_names = list(contract.params.keys())
                    for i, arg in enumerate(node.args):
                        if i < len(param_names):
                            pname = param_names[i]
                            required = contract.params[pname]
                            actual = self._infer_expr_type(arg)
                            self.constraints.append(LiquidConstraint(
                                kind=ConstraintKind.SUBTYPE,
                                actual=actual.alpha_rename("ν"),
                                required=required.alpha_rename("ν"),
                                line=getattr(node, "lineno", 0),
                                col=getattr(node, "col_offset", 0),
                                context=f"call {fname} param {pname}",
                                variable=pname,
                                function=self.func_name,
                            ))

        # Recurse into sub-expressions
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self._gen_use_constraints(child)

    # -- expression type inference ------------------------------------------

    def _infer_expr_type(self, node: ast.expr) -> RefType:
        """Infer the refinement type of an expression."""

        # Constant
        if isinstance(node, ast.Constant):
            base = _base_type_from_constant(node.value)
            if node.value is None:
                return RefType("ν", NONE_TYPE, Pred.is_none("ν"))
            if isinstance(node.value, (int, float)):
                val = int(node.value) if isinstance(node.value, int) else int(node.value)
                return RefType("ν", base, Pred.var_eq("ν", val))
            return RefType("ν", base, Pred.is_not_none("ν"))

        # Variable
        if isinstance(node, ast.Name):
            ty = self.env.get(node.id)
            if ty is not None:
                return ty
            return RefType.trivial(ANY_TYPE, node.id)

        # BinOp
        if isinstance(node, ast.BinOp):
            left_type = self._infer_expr_type(node.left)
            right_type = self._infer_expr_type(node.right)
            # Division constraint is generated in _gen_use_constraints; no duplicate here
            result_base = left_type.base
            if result_base.kind == BaseTypeKind.ANY:
                result_base = right_type.base
            return RefType.trivial(result_base)

        # UnaryOp
        if isinstance(node, ast.UnaryOp):
            operand_type = self._infer_expr_type(node.operand)
            return RefType.trivial(operand_type.base)

        # Compare
        if isinstance(node, ast.Compare):
            return RefType.trivial(BOOL_TYPE)

        # BoolOp
        if isinstance(node, ast.BoolOp):
            # or-default pattern
            if isinstance(node.op, ast.Or):
                last = node.values[-1]
                last_type = self._infer_expr_type(last)
                first_type = self._infer_expr_type(node.values[0])
                result_base = first_type.base if first_type.base.kind != BaseTypeKind.ANY else last_type.base
                return RefType.trivial(result_base)
            return RefType.trivial(BOOL_TYPE)

        # Call
        if isinstance(node, ast.Call):
            return self._infer_call_type(node)

        # Attribute
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                self._gen_use_constraints(node)
            return RefType.trivial(ANY_TYPE)

        # Subscript
        if isinstance(node, ast.Subscript):
            return RefType.trivial(ANY_TYPE)

        # IfExp (ternary)
        if isinstance(node, ast.IfExp):
            body_type = self._infer_expr_type(node.body)
            orelse_type = self._infer_expr_type(node.orelse)
            return self.lattice.join(body_type, orelse_type)

        # List/Dict/Set/Tuple literals
        if isinstance(node, ast.List):
            return RefType("ν", BaseTypeR(BaseTypeKind.LIST), Pred.is_not_none("ν"))
        if isinstance(node, ast.Dict):
            return RefType("ν", BaseTypeR(BaseTypeKind.DICT), Pred.is_not_none("ν"))
        if isinstance(node, ast.Set):
            return RefType("ν", BaseTypeR(BaseTypeKind.SET), Pred.is_not_none("ν"))
        if isinstance(node, ast.Tuple):
            return RefType("ν", BaseTypeR(BaseTypeKind.TUPLE), Pred.is_not_none("ν"))

        # ListComp, SetComp, DictComp, GeneratorExp
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return RefType.trivial(ANY_TYPE)

        # NamedExpr
        if isinstance(node, ast.NamedExpr):
            val_type = self._infer_expr_type(node.value)
            if isinstance(node.target, ast.Name):
                self.env = self.env.set(node.target.id, val_type)
            return val_type

        # JoinedStr (f-string)
        if isinstance(node, ast.JoinedStr):
            return RefType("ν", STR_TYPE, Pred.is_not_none("ν"))

        return RefType.trivial(ANY_TYPE)

    def _infer_call_type(self, node: ast.Call) -> RefType:
        """Infer the return type of a function call."""
        if isinstance(node.func, ast.Name):
            fname = node.func.id

            # Built-in constructors
            builtin_types = {
                "int": INT_TYPE, "float": FLOAT_TYPE, "str": STR_TYPE,
                "bool": BOOL_TYPE, "list": BaseTypeR(BaseTypeKind.LIST),
                "dict": BaseTypeR(BaseTypeKind.DICT),
                "set": BaseTypeR(BaseTypeKind.SET),
                "tuple": BaseTypeR(BaseTypeKind.TUPLE),
            }
            if fname in builtin_types:
                return RefType("ν", builtin_types[fname], Pred.is_not_none("ν"))

            # len()
            if fname == "len" and node.args:
                return RefType("ν", INT_TYPE, Pred.var_ge("ν", 0))

            # abs()
            if fname == "abs" and node.args:
                return RefType("ν", INT_TYPE, Pred.var_ge("ν", 0))

            # Known function contract
            if fname in self.func_contracts:
                contract = self.func_contracts[fname]
                return contract.return_type

        # Method call: x.method()
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            # dict.get with default: does NOT guarantee not-None
            # because d[key] could be None even with a non-None default.
            if method == "get" and len(node.args) >= 2:
                pass  # Intentionally return ANY_TYPE — unsound to assume not-None
            # re.match, re.search → Optional
            if method in ("match", "search", "fullmatch"):
                return RefType.trivial(ANY_TYPE)

        return RefType.trivial(ANY_TYPE)

    # -- condition → predicate conversion -----------------------------------

    def _condition_to_pred(self, test: ast.expr) -> Optional[Pred]:
        """Convert an AST condition to a Pred, or None if not representable."""

        # BoolOp: and/or
        if isinstance(test, ast.BoolOp):
            preds = [self._condition_to_pred(v) for v in test.values]
            preds = [p for p in preds if p is not None]
            if not preds:
                return None
            result = preds[0]
            for p in preds[1:]:
                if isinstance(test.op, ast.And):
                    result = result.and_(p)
                else:
                    result = result.or_(p)
            return result

        # UnaryOp: not
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = self._condition_to_pred(test.operand)
            return inner.not_() if inner else None

        # Compare
        if isinstance(test, ast.Compare):
            return self._compare_to_pred(test)

        # isinstance call
        if isinstance(test, ast.Call) and isinstance(test.func, ast.Name):
            if test.func.id == "isinstance" and len(test.args) >= 2:
                if isinstance(test.args[0], ast.Name):
                    tname = PredicateHarvester._type_node_to_str(test.args[1])
                    if tname:
                        return Pred.isinstance_(test.args[0].id, tname)
            if test.func.id == "hasattr" and len(test.args) >= 2:
                if isinstance(test.args[0], ast.Name) and isinstance(test.args[1], ast.Constant):
                    return Pred.hasattr_(test.args[0].id, str(test.args[1].value))

        # Name: truthiness
        if isinstance(test, ast.Name):
            return Pred.truthy(test.id)

        # NamedExpr: walrus
        if isinstance(test, ast.NamedExpr):
            if isinstance(test.target, ast.Name):
                return Pred.is_not_none(test.target.id)

        return None

    def _compare_to_pred(self, node: ast.Compare) -> Optional[Pred]:
        """Convert a Compare AST node to a Pred."""
        left = node.left
        if not node.ops or not node.comparators:
            return None

        op = node.ops[0]
        comp = node.comparators[0]

        # x is None
        if isinstance(op, ast.Is) and PredicateHarvester._is_none_literal(comp):
            if isinstance(left, ast.Name):
                return Pred.is_none(left.id)
        # x is not None
        if isinstance(op, ast.IsNot) and PredicateHarvester._is_none_literal(comp):
            if isinstance(left, ast.Name):
                return Pred.is_not_none(left.id)
        # None is x  (reversed)
        if isinstance(op, ast.Is) and PredicateHarvester._is_none_literal(left):
            if isinstance(comp, ast.Name):
                return Pred.is_none(comp.id)
        if isinstance(op, ast.IsNot) and PredicateHarvester._is_none_literal(left):
            if isinstance(comp, ast.Name):
                return Pred.is_not_none(comp.id)

        # x op constant
        if isinstance(left, ast.Name) and isinstance(comp, ast.Constant):
            val = comp.value
            if isinstance(val, (int, float)):
                int_val = int(val)
                op_map = {
                    ast.Lt: PredOp.LT, ast.LtE: PredOp.LE,
                    ast.Gt: PredOp.GT, ast.GtE: PredOp.GE,
                    ast.Eq: PredOp.EQ, ast.NotEq: PredOp.NEQ,
                }
                if type(op) in op_map:
                    return Pred(op_map[type(op)], (left.id, int_val))

        # constant op x  (flip)
        if isinstance(comp, ast.Name) and isinstance(left, ast.Constant):
            val = left.value
            if isinstance(val, (int, float)):
                int_val = int(val)
                flip = {
                    ast.Lt: PredOp.GT, ast.LtE: PredOp.GE,
                    ast.Gt: PredOp.LT, ast.GtE: PredOp.LE,
                    ast.Eq: PredOp.EQ, ast.NotEq: PredOp.NEQ,
                }
                if type(op) in flip:
                    return Pred(flip[type(op)], (comp.id, int_val))

        return None

    # -- environment narrowing ----------------------------------------------

    def _narrow_env(
        self, env: RefEnvironment, test: ast.expr, positive: bool
    ) -> RefEnvironment:
        """Narrow the environment based on a guard condition."""
        new_env = RefEnvironment(dict(env.bindings))

        # x is None / x is not None
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            op = test.ops[0]
            comp = test.comparators[0]
            left = test.left

            if isinstance(op, ast.Is) and PredicateHarvester._is_none_literal(comp):
                if isinstance(left, ast.Name):
                    var = left.id
                    current = env.get(var) or RefType.trivial(ANY_TYPE, var)
                    if positive:
                        new_env = new_env.set(var, RefType(var, NONE_TYPE, Pred.is_none(var)))
                    else:
                        new_env = new_env.set(var, RefType(var, current.base, Pred.is_not_none(var)))

            elif isinstance(op, ast.IsNot) and PredicateHarvester._is_none_literal(comp):
                if isinstance(left, ast.Name):
                    var = left.id
                    current = env.get(var) or RefType.trivial(ANY_TYPE, var)
                    if positive:
                        new_env = new_env.set(var, RefType(var, current.base, Pred.is_not_none(var)))
                    else:
                        new_env = new_env.set(var, RefType(var, NONE_TYPE, Pred.is_none(var)))

            elif isinstance(left, ast.Name) and isinstance(comp, ast.Constant):
                var = left.id
                val = comp.value
                if isinstance(val, (int, float)):
                    int_val = int(val)
                    current = env.get(var) or RefType.trivial(ANY_TYPE, var)
                    op_map_pos = {
                        ast.Lt: PredOp.LT, ast.LtE: PredOp.LE,
                        ast.Gt: PredOp.GT, ast.GtE: PredOp.GE,
                        ast.Eq: PredOp.EQ, ast.NotEq: PredOp.NEQ,
                    }
                    negate_map = {
                        PredOp.LT: PredOp.GE, PredOp.LE: PredOp.GT,
                        PredOp.GT: PredOp.LE, PredOp.GE: PredOp.LT,
                        PredOp.EQ: PredOp.NEQ, PredOp.NEQ: PredOp.EQ,
                    }
                    if type(op) in op_map_pos:
                        pop = op_map_pos[type(op)] if positive else negate_map[op_map_pos[type(op)]]
                        pred = Pred(pop, (var, int_val))
                        if current.pred.op != PredOp.TRUE:
                            pred = current.pred.and_(pred)
                        new_env = new_env.set(var, RefType(var, current.base, pred))

        # isinstance(x, T)
        elif isinstance(test, ast.Call) and isinstance(test.func, ast.Name):
            if test.func.id == "isinstance" and len(test.args) >= 2:
                if isinstance(test.args[0], ast.Name):
                    var = test.args[0].id
                    tname = PredicateHarvester._type_node_to_str(test.args[1])
                    current = env.get(var) or RefType.trivial(ANY_TYPE, var)
                    if tname and positive:
                        base = _TYPE_MAP.get(tname, ANY_TYPE)
                        new_env = new_env.set(var, RefType(var, base, Pred.isinstance_(var, tname)))

        # Truthiness: if x:
        elif isinstance(test, ast.Name):
            var = test.id
            current = env.get(var) or RefType.trivial(ANY_TYPE, var)
            if positive:
                # Only narrow to is_not_none when the base type could
                # include None (NONE_TYPE or ANY_TYPE which subsumes it).
                # For concrete non-None types (int, str, list, …)
                # truthiness does NOT establish non-None-ness — the
                # variable could simply be 0, "", [], etc.
                if current.base == NONE_TYPE or current.base == ANY_TYPE:
                    new_env = new_env.set(var, RefType(var, current.base, Pred.is_not_none(var)))
            # negative: x could be None or falsy, but we don't narrow to None

        # not x
        elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return self._narrow_env(env, test.operand, not positive)

        # BoolOp: and / or
        elif isinstance(test, ast.BoolOp):
            if isinstance(test.op, ast.And):
                if positive:
                    # Both branches true
                    for val in test.values:
                        new_env = self._narrow_env(new_env, val, True)
                else:
                    # At least one false – don't narrow (imprecise but sound)
                    pass
            elif isinstance(test.op, ast.Or):
                if positive:
                    pass  # at least one true – don't narrow
                else:
                    # Both false
                    for val in test.values:
                        new_env = self._narrow_env(new_env, val, False)

        return new_env


# ═══════════════════════════════════════════════════════════════════════════
# 7.  CEGAR Loop
# ═══════════════════════════════════════════════════════════════════════════

class CEGARRefinement:
    """Counterexample-Guided Abstraction Refinement for predicate discovery.

    When the initial set of harvested predicates is insufficient to prove a
    safety property, the CEGAR loop:
      1. Identifies the failing constraint
      2. Extracts a Z3 counterexample (concrete values violating the constraint)
      3. Synthesises a new predicate that eliminates the counterexample
      4. Adds the predicate and re-runs inference

    Uses Z3's model to guide predicate synthesis: finding concrete values
    that violate the required property, then discovering predicates that
    exclude those values.
    """

    MAX_ITERATIONS = 10

    def __init__(self, lattice: RefinementLattice):
        self.lattice = lattice
        self.encoder = lattice.encoder
        self.iterations = 0
        self._counterexamples: List[dict] = []

    def refine(
        self,
        func: ast.FunctionDef,
        initial_preds: List[HarvestedPred],
        func_contracts: Dict[str, FunctionContract],
    ) -> Tuple[List[LiquidConstraint], List[HarvestedPred], int]:
        """Run the CEGAR loop. Returns (constraints, predicates, iterations)."""
        preds = list(initial_preds)
        self.iterations = 0
        self._counterexamples = []

        while self.iterations < self.MAX_ITERATIONS:
            self.iterations += 1
            cgen = ConstraintGenerator(func, preds, self.lattice, func_contracts)
            constraints = cgen.generate()

            failing = self._find_failing_with_cex(constraints)
            if not failing:
                return constraints, preds, self.iterations

            new_preds = self._synthesize_from_cex(failing, func)
            if not new_preds:
                new_preds = self._synthesize_syntactic(failing, func)
            if not new_preds:
                return constraints, preds, self.iterations

            preds.extend(new_preds)

        return constraints, preds, self.iterations

    def _find_failing_with_cex(
        self, constraints: List[LiquidConstraint]
    ) -> List[Tuple[LiquidConstraint, Optional[dict]]]:
        """Find failing constraints and extract Z3 counterexamples."""
        failing = []
        for c in constraints:
            if c.kind == ConstraintKind.SUBTYPE:
                is_sub, cex = self._check_with_cex(c.actual, c.required)
                if not is_sub:
                    failing.append((c, cex))
        return failing

    def _check_with_cex(
        self, actual: RefType, required: RefType
    ) -> Tuple[bool, Optional[dict]]:
        """Check subtyping via Z3 and extract counterexample on failure.

        Checks P \u2227 \u00acQ for satisfiability. If SAT, the model is a
        concrete assignment violating the required refinement.
        """
        if not actual.base.is_subtype_of(required.base):
            if actual.base.kind != BaseTypeKind.ANY and required.base.kind != BaseTypeKind.ANY:
                return False, None

        p = actual.pred
        q = required.pred
        if p == q or q.op == PredOp.TRUE:
            return True, None

        try:
            s = _z3.Solver()
            s.set("timeout", 2000)
            p_z3 = self.encoder.encode(p)
            q_z3 = self.encoder.encode(q)
            s.add(p_z3)
            s.add(_z3.Not(q_z3))
            result = s.check()
            if result == _z3.sat:
                model = s.model()
                cex = {d.name(): str(model[d]) for d in model.decls()}
                self._counterexamples.append(cex)
                return False, cex
            elif result == _z3.unsat:
                return True, None
            else:
                return False, None
        except Exception:
            return self.lattice.subtype(actual, required), None

    def _synthesize_from_cex(
        self,
        failing: List[Tuple[LiquidConstraint, Optional[dict]]],
        func: ast.FunctionDef,
    ) -> List[HarvestedPred]:
        """Synthesize predicates guided by Z3 counterexamples."""
        new_preds: List[HarvestedPred] = []
        seen: Set[Pred] = set()

        for c, cex in failing:
            if cex is None:
                continue
            var = c.variable
            req_pred = c.required.pred

            # Counterexample shows var is None
            none_key = f"{var}_is_none"
            if cex.get(none_key) == "True" or cex.get("is_none") == "True":
                for node in ast.walk(func):
                    if isinstance(node, (ast.If, ast.While)):
                        if self._is_none_guard(node.test, var):
                            pred = Pred.is_not_none(var)
                            if pred not in seen:
                                new_preds.append(HarvestedPred(
                                    pred=pred, source=HarvestSource.GUARD,
                                    line=getattr(node, "lineno", 0), variable=var))
                                seen.add(pred)

            # Counterexample shows var = 0 for div-by-zero
            nu = "\u03bd"
            if req_pred.op == PredOp.NEQ and req_pred.args == (nu, 0):
                cex_val = cex.get(var) or cex.get(nu)
                if cex_val == "0":
                    for node in ast.walk(func):
                        if isinstance(node, (ast.If, ast.While)):
                            pred = self._find_numeric_guard(node.test, var)
                            if pred and pred not in seen:
                                new_preds.append(HarvestedPred(
                                    pred=pred, source=HarvestSource.GUARD,
                                    line=getattr(node, "lineno", 0), variable=var))
                                seen.add(pred)
                        # Early-exit guard: if var == 0: raise -> var != 0
                        if isinstance(node, ast.If) and PredicateHarvester._branch_always_exits(node.body):
                            pred = self._find_numeric_guard(node.test, var)
                            if pred:
                                negated = pred.not_()
                                if negated not in seen:
                                    new_preds.append(HarvestedPred(
                                        pred=negated, source=HarvestSource.GUARD,
                                        line=getattr(node, "lineno", 0), variable=var))
                                    seen.add(negated)

            # Numeric constraints
            if req_pred.op in (PredOp.GT, PredOp.GE, PredOp.LT, PredOp.LE):
                for node in ast.walk(func):
                    if isinstance(node, (ast.If, ast.While)):
                        guard = self._condition_matches(node.test, var, req_pred)
                        if guard and guard not in seen:
                            new_preds.append(HarvestedPred(
                                pred=guard, source=HarvestSource.GUARD,
                                line=getattr(node, "lineno", 0), variable=var))
                            seen.add(guard)
        return new_preds

    def _is_none_guard(self, test: ast.expr, var: str) -> bool:
        """Check if test is a None guard for var."""
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            op = test.ops[0]
            left = test.left
            comp = test.comparators[0]
            if isinstance(op, (ast.Is, ast.IsNot)):
                if PredicateHarvester._is_none_literal(comp) and isinstance(left, ast.Name):
                    return left.id == var
                if PredicateHarvester._is_none_literal(left) and isinstance(comp, ast.Name):
                    return comp.id == var
        if isinstance(test, ast.Name) and test.id == var:
            return True
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return self._is_none_guard(test.operand, var)
        return False

    def _find_numeric_guard(self, test: ast.expr, var: str) -> Optional[Pred]:
        """Find a numeric guard for var in test."""
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            left = test.left
            comp = test.comparators[0]
            ast_op = test.ops[0]
            if isinstance(left, ast.Name) and left.id == var:
                if isinstance(comp, ast.Constant) and isinstance(comp.value, (int, float)):
                    op_map = {
                        ast.NotEq: PredOp.NEQ, ast.Gt: PredOp.GT,
                        ast.GtE: PredOp.GE, ast.Lt: PredOp.LT,
                        ast.LtE: PredOp.LE, ast.Eq: PredOp.EQ,
                    }
                    if type(ast_op) in op_map:
                        return Pred(op_map[type(ast_op)], (var, int(comp.value)))
        return None

    def _synthesize_syntactic(
        self,
        failing: List[Tuple[LiquidConstraint, Optional[dict]]],
        func: ast.FunctionDef,
    ) -> List[HarvestedPred]:
        """Fallback: syntactic predicate synthesis."""
        new_preds: List[HarvestedPred] = []
        seen: Set[Pred] = set()

        for c, _ in failing:
            req_pred = c.required.pred
            if req_pred in seen:
                continue
            seen.add(req_pred)

            for node in ast.walk(func):
                if isinstance(node, (ast.If, ast.While)):
                    guard = self._condition_matches(node.test, c.variable, req_pred)
                    if guard and guard not in seen:
                        new_preds.append(HarvestedPred(
                            pred=guard, source=HarvestSource.GUARD,
                            line=getattr(node, "lineno", 0), variable=c.variable))
                        seen.add(guard)
                elif isinstance(node, ast.Assert):
                    guard = self._condition_matches(node.test, c.variable, req_pred)
                    if guard and guard not in seen:
                        new_preds.append(HarvestedPred(
                            pred=guard, source=HarvestSource.ASSERT,
                            line=getattr(node, "lineno", 0), variable=c.variable))
                        seen.add(guard)

            if not new_preds and req_pred.op in (PredOp.IS_NOT_NONE, PredOp.NEQ,
                                                    PredOp.GT, PredOp.GE, PredOp.LT, PredOp.LE):
                var = c.variable
                nu = "\u03bd"
                synth_pred = req_pred.substitute(nu, var)
                if synth_pred not in seen:
                    new_preds.append(HarvestedPred(
                        pred=synth_pred, source=HarvestSource.GUARD,
                        line=c.line, variable=var))
                    seen.add(synth_pred)
        return new_preds

    def _condition_matches(
        self, test: ast.expr, var: str, required: Pred
    ) -> Optional[Pred]:
        """Check if the AST condition can establish *required* for *var*."""
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            op = test.ops[0]
            left = test.left
            comp = test.comparators[0]

            if isinstance(op, ast.IsNot) and PredicateHarvester._is_none_literal(comp):
                if isinstance(left, ast.Name) and left.id == var:
                    if required.op == PredOp.IS_NOT_NONE:
                        return Pred.is_not_none(var)

            if isinstance(left, ast.Name) and left.id == var:
                if isinstance(comp, ast.Constant) and isinstance(comp.value, (int, float)):
                    int_val = int(comp.value)
                    op_map = {
                        ast.NotEq: PredOp.NEQ, ast.Gt: PredOp.GT,
                        ast.GtE: PredOp.GE, ast.Lt: PredOp.LT,
                        ast.LtE: PredOp.LE, ast.Eq: PredOp.EQ,
                    }
                    if type(op) in op_map:
                        return Pred(op_map[type(op)], (var, int_val))

        if (isinstance(test, ast.Call) and isinstance(test.func, ast.Name)
                and test.func.id == "isinstance" and len(test.args) >= 2):
            if isinstance(test.args[0], ast.Name) and test.args[0].id == var:
                tname = PredicateHarvester._type_node_to_str(test.args[1])
                if tname:
                    return Pred.isinstance_(var, tname)

        if isinstance(test, ast.Name) and test.id == var:
            if required.op == PredOp.IS_NOT_NONE:
                return Pred.is_not_none(var)

        return None

class LiquidTypeInferencer:
    """The main liquid type inference engine.

    Two-pass architecture:
      1. Harvest predicates from all sources in the module
      2. Generate + solve subtyping constraints using Z3
      3. CEGAR: refine predicates when counterexample found
      4. Output: per-function liquid type contracts + bug reports

    Usage::

        engine = LiquidTypeInferencer()
        result = engine.infer_module(source_code)
        for bug in result.bugs:
            print(bug)
        for name, contract in result.contracts.items():
            print(contract.pretty())
    """

    def __init__(self, timeout_ms: int = 5000, max_cegar: int = 10):
        self.lattice = RefinementLattice(timeout_ms=timeout_ms)
        self.harvester = PredicateHarvester()
        self.max_cegar = max_cegar

    # -- public API ---------------------------------------------------------

    def infer_module(self, source: str) -> LiquidAnalysisResult:
        """Analyze a complete Python module.

        Returns contracts, bugs, and statistics.
        """
        t0 = time.monotonic()
        result = LiquidAnalysisResult()

        tree = ast.parse(source)

        # Phase 0: infer flow-sensitive summaries (interprocedural aid)
        flow_summaries = infer_file_summaries(tree)

        # Phase 1: harvest predicates from ALL sources
        all_preds = self.harvester.harvest_module(tree)
        result.predicates_harvested = len(all_preds)
        result.harvested_preds = list(all_preds)

        # Phase 2: collect function nodes
        func_nodes: List[ast.FunctionDef] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_nodes.append(node)

        # Phase 3: per-function inference with CEGAR
        contracts: Dict[str, FunctionContract] = {}
        all_bugs: List[LiquidBug] = []
        total_constraints = 0
        total_solved = 0
        total_cegar = 0

        # Process functions in dependency order (topological) – simple heuristic:
        # process functions that don't call others first
        processed: Set[str] = set()
        remaining = list(func_nodes)
        max_rounds = len(remaining) + 1
        round_num = 0

        while remaining and round_num < max_rounds:
            round_num += 1
            still_remaining = []
            for func in remaining:
                contract, bugs, n_constraints, n_solved, n_cegar = self._infer_function(
                    func, all_preds, contracts, flow_summaries,
                )
                contracts[func.name] = contract
                all_bugs.extend(bugs)
                total_constraints += n_constraints
                total_solved += n_solved
                total_cegar += n_cegar
                processed.add(func.name)
            remaining = still_remaining

        result.contracts = contracts
        result.bugs = all_bugs
        result.constraints_generated = total_constraints
        result.constraints_solved = total_solved
        result.cegar_iterations = total_cegar
        result.analysis_time_ms = (time.monotonic() - t0) * 1000
        return result

    def infer_function(
        self,
        func: ast.FunctionDef,
        source: str = "",
        existing_contracts: Optional[Dict[str, FunctionContract]] = None,
    ) -> FunctionContract:
        """Infer the liquid type contract for a single function."""
        tree = ast.parse(source) if source else ast.Module(body=[func], type_ignores=[])
        preds = self.harvester.harvest_all(func)
        contracts = existing_contracts or {}
        flow_summaries = infer_file_summaries(tree) if source else {}
        contract, _, _, _, _ = self._infer_function(func, preds, contracts, flow_summaries)
        return contract

    # -- internal -----------------------------------------------------------

    def _infer_function(
        self,
        func: ast.FunctionDef,
        all_preds: List[HarvestedPred],
        contracts: Dict[str, FunctionContract],
        flow_summaries: Dict[str, FunctionSummary],
    ) -> Tuple[FunctionContract, List[LiquidBug], int, int, int]:
        """Infer contract + find bugs for a single function.

        Returns (contract, bugs, n_constraints, n_solved, n_cegar_iters).
        """
        # Filter predicates relevant to this function
        func_preds = [hp for hp in all_preds
                      if hp.variable != "__return__" or hp.source == HarvestSource.RETURN]

        # CEGAR loop
        cegar = CEGARRefinement(self.lattice)
        cegar.MAX_ITERATIONS = self.max_cegar
        constraints, refined_preds, cegar_iters = cegar.refine(
            func, func_preds, contracts,
        )

        # Check constraints and collect bugs
        bugs: List[LiquidBug] = []
        n_solved = 0
        for c in constraints:
            if c.kind == ConstraintKind.SUBTYPE:
                is_sub = self.lattice.subtype(c.actual, c.required)
                n_solved += 1
                if not is_sub:
                    bug = self._constraint_to_bug(c, func.name)
                    if bug is not None:
                        bugs.append(bug)

        # Build function contract
        contract = self._build_contract(func, constraints, refined_preds, flow_summaries)

        return contract, bugs, len(constraints), n_solved, cegar_iters

    def _constraint_to_bug(
        self, c: LiquidConstraint, func_name: str
    ) -> Optional[LiquidBug]:
        """Convert a failing constraint to a LiquidBug."""
        req = c.required.pred

        # Null dereference
        if req.op == PredOp.IS_NOT_NONE:
            return LiquidBug(
                kind=LiquidBugKind.NULL_DEREF,
                line=c.line, col=c.col,
                message=f"Possible null dereference: {c.variable} may be None",
                function=func_name,
                variable=c.variable,
                actual_type=c.actual,
                required_type=c.required,
            )

        # Division by zero
        if req.op == PredOp.NEQ and req.args == ("ν", 0):
            return LiquidBug(
                kind=LiquidBugKind.DIV_BY_ZERO,
                line=c.line, col=c.col,
                message=f"Possible division by zero: {c.variable} may be 0",
                function=func_name,
                variable=c.variable,
                actual_type=c.actual,
                required_type=c.required,
            )

        # Index OOB
        if req.op == PredOp.GE and req.args[1] == 0:
            return LiquidBug(
                kind=LiquidBugKind.INDEX_OOB,
                line=c.line, col=c.col,
                message=f"Possible index out of bounds: {c.variable} may be negative",
                function=func_name,
                variable=c.variable,
                actual_type=c.actual,
                required_type=c.required,
            )

        # Generic unsatisfied constraint
        return LiquidBug(
            kind=LiquidBugKind.UNSAT_CONSTRAINT,
            line=c.line, col=c.col,
            message=f"Unsatisfied subtyping constraint: {c.actual.pretty()} <: {c.required.pretty()}",
            function=func_name,
            variable=c.variable,
            actual_type=c.actual,
            required_type=c.required,
        )

    def _build_contract(
        self,
        func: ast.FunctionDef,
        constraints: List[LiquidConstraint],
        preds: List[HarvestedPred],
        flow_summaries: Dict[str, FunctionSummary],
    ) -> FunctionContract:
        """Build a FunctionContract from solved constraints and harvested predicates."""
        contract = FunctionContract(
            name=func.name,
            line=func.lineno,
        )

        # Infer base types from usage context
        cgen_tmp = ConstraintGenerator(func, [], self.lattice, {})
        inferred_bases = cgen_tmp._infer_param_base_types()

        # Parameter types
        for arg in func.args.args:
            name = arg.arg
            if arg.annotation:
                base = _base_type_from_annotation(arg.annotation)
            else:
                base = inferred_bases.get(name, ANY_TYPE)

            # Collect predicates relevant to this parameter (deduplicated)
            # When both P and ¬P are harvested (guard + early-exit negation),
            # keep only the negation (what holds on the continuation).
            param_preds_seen: Set[str] = set()
            param_preds: List[Pred] = []
            all_pred_keys: Set[str] = set()
            for hp in preds:
                if hp.variable == name:
                    all_pred_keys.add(hp.pred.pretty())

            for hp in preds:
                if hp.variable == name:
                    key = hp.pred.pretty()
                    if key in param_preds_seen:
                        continue
                    # Skip P if ¬P also exists (early-exit guard pattern)
                    neg_key = hp.pred.not_().pretty()
                    if neg_key in all_pred_keys and key != neg_key:
                        # Both P and ¬P exist; skip P, keep ¬P
                        negated = hp.pred.not_()
                        neg_key_str = negated.pretty()
                        if neg_key_str not in param_preds_seen:
                            param_preds_seen.add(neg_key_str)
                            param_preds_seen.add(key)
                            param_preds.append(negated)
                        continue
                    param_preds_seen.add(key)
                    param_preds.append(hp.pred)
            if param_preds:
                combined = param_preds[0]
                for p in param_preds[1:]:
                    combined = combined.and_(p)
                contract.params[name] = RefType(name, base, combined)
            else:
                contract.params[name] = RefType.trivial(base, name)

        # Return type from flow summary, AST analysis, or constraint analysis
        summary = flow_summaries.get(func.name)
        ret_pred = Pred.true_()
        ret_base = ANY_TYPE

        if summary:
            if summary.return_null_state == NullState.DEFINITELY_NOT_NULL:
                ret_pred = Pred.is_not_none("ν")
            elif summary.return_null_state == NullState.DEFINITELY_NULL:
                ret_pred = Pred.is_none("ν")

            if summary.return_tags and summary.return_tags.tags:
                tags = summary.return_tags.tags
                if len(tags) == 1:
                    tag = next(iter(tags))
                    ret_base = _TYPE_MAP.get(tag, ANY_TYPE)

        # If flow summary didn't determine not-None, check for raise-guarded
        # return patterns: if x is None: raise ... ; return x
        if ret_pred.op == PredOp.TRUE:
            ret_pred = self._infer_return_refinement(func)

        contract.return_type = RefType("ν", ret_base, ret_pred)

        # Collect return-source predicates
        return_preds = [hp.pred for hp in preds
                        if hp.variable == "__return__" and hp.source == HarvestSource.RETURN]
        if return_preds and contract.return_type.pred.op == PredOp.TRUE:
            combined = return_preds[0]
            for p in return_preds[1:]:
                combined = combined.and_(p)
            contract.return_type = RefType("ν", contract.return_type.base, combined)

        # Preconditions: predicates from assert/guard on parameters (deduplicated)
        # Filter contradictory predicates (keep negation from early-exit guards)
        seen_preds: Set[str] = set()
        all_guard_keys: Set[str] = set()
        for arg in func.args.args:
            name = arg.arg
            for hp in preds:
                if hp.variable == name and hp.source in (HarvestSource.ASSERT, HarvestSource.GUARD):
                    all_guard_keys.add(hp.pred.pretty())

        for arg in func.args.args:
            name = arg.arg
            for hp in preds:
                if hp.variable == name and hp.source in (HarvestSource.ASSERT, HarvestSource.GUARD):
                    key = hp.pred.pretty()
                    if key in seen_preds:
                        continue
                    neg_key = hp.pred.not_().pretty()
                    if neg_key in all_guard_keys and key != neg_key:
                        negated = hp.pred.not_()
                        nk = negated.pretty()
                        if nk not in seen_preds:
                            seen_preds.add(nk)
                            seen_preds.add(key)
                            contract.preconditions.append(negated)
                        continue
                    seen_preds.add(key)
                    contract.preconditions.append(hp.pred)

        # Postconditions: from return harvesting
        for hp in preds:
            if hp.variable == "__return__":
                contract.postconditions.append(hp.pred)

        return contract

    @staticmethod
    def _infer_return_refinement(func: ast.FunctionDef) -> Pred:
        """Infer return-type refinement from raise-guarded patterns.

        Detects: if x is None: raise ... ; return x  →  return is not-None
        """
        raise_guarded_vars: Set[str] = set()
        for node in ast.walk(func):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            var_name = None
            if isinstance(test, ast.Compare) and len(test.ops) == 1:
                op = test.ops[0]
                left = test.left
                comp = test.comparators[0]
                if isinstance(op, ast.Is) and PredicateHarvester._is_none_literal(comp):
                    if isinstance(left, ast.Name):
                        var_name = left.id
                elif isinstance(op, ast.IsNot) and PredicateHarvester._is_none_literal(comp):
                    if isinstance(left, ast.Name):
                        if node.orelse and any(isinstance(s, ast.Raise) for s in node.orelse):
                            raise_guarded_vars.add(left.id)
                        continue

            if var_name and node.body:
                has_raise = any(isinstance(s, ast.Raise) for s in node.body)
                has_return_none = any(
                    isinstance(s, ast.Return) and s.value is not None
                    and isinstance(s.value, ast.Constant) and s.value.value is None
                    for s in node.body
                )
                if has_raise or has_return_none:
                    raise_guarded_vars.add(var_name)

        return_nodes = [n for n in ast.walk(func) if isinstance(n, ast.Return) and n.value is not None]
        if not return_nodes:
            return Pred.true_()

        all_returns_safe = True
        for ret in return_nodes:
            val = ret.value
            if isinstance(val, ast.Constant) and val.value is None:
                all_returns_safe = False
                break
            if isinstance(val, ast.Name) and val.id not in raise_guarded_vars:
                all_returns_safe = False
                break
            if isinstance(val, ast.Constant) and val.value is not None:
                continue
            if isinstance(val, (ast.List, ast.Dict, ast.Set, ast.Tuple,
                               ast.JoinedStr, ast.Call)):
                continue

        if all_returns_safe and raise_guarded_vars:
            return Pred.is_not_none("ν")
        return Pred.true_()


# ═══════════════════════════════════════════════════════════════════════════
# 9.  Convenience API
# ═══════════════════════════════════════════════════════════════════════════

def analyze_liquid(source: str, **kwargs) -> LiquidAnalysisResult:
    """One-shot API: analyze *source* and return liquid-type results."""
    engine = LiquidTypeInferencer(**kwargs)
    return engine.infer_module(source)


def harvest_predicates(source: str) -> List[HarvestedPred]:
    """Harvest all predicates from *source* without running inference."""
    tree = ast.parse(source)
    harvester = PredicateHarvester()
    return harvester.harvest_module(tree)


def infer_contract(source: str, func_name: str) -> Optional[FunctionContract]:
    """Infer the liquid type contract for *func_name* in *source*."""
    result = analyze_liquid(source)
    return result.contracts.get(func_name)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Interprocedural Liquid Analysis
# ═══════════════════════════════════════════════════════════════════════════

class InterproceduralLiquidAnalyzer:
    """Whole-module liquid type analysis with interprocedural propagation.

    After per-function contracts are inferred, this class propagates
    contracts across call boundaries:
      - If f calls g, and g's contract says g returns not-None,
        then f's use of g()'s result is refined to not-None.
      - If all call sites of g guard an argument with x > 0,
        then g gets a precondition x > 0.
    """

    def __init__(self, timeout_ms: int = 5000, max_cegar: int = 10):
        self.engine = LiquidTypeInferencer(timeout_ms=timeout_ms, max_cegar=max_cegar)

    def analyze(self, source: str) -> LiquidAnalysisResult:
        """Run interprocedural analysis.

        Two rounds:
          1. Infer per-function contracts
          2. Re-analyze using inferred contracts as context
        """
        # Round 1: initial inference
        result1 = self.engine.infer_module(source)

        # Round 2: re-analyze with contracts from round 1
        tree = ast.parse(source)
        all_preds = self.engine.harvester.harvest_module(tree)

        # Augment with call-site inference
        call_site_preds = self.engine.harvester.harvest_call_sites(tree)
        for fname, csp in call_site_preds.items():
            all_preds.extend(csp)

        flow_summaries = infer_file_summaries(tree)
        contracts = dict(result1.contracts)

        # Re-infer functions that call other functions
        func_nodes: List[ast.FunctionDef] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_nodes.append(node)

        all_bugs: List[LiquidBug] = []
        total_constraints = 0
        total_solved = 0
        total_cegar = 0

        for func in func_nodes:
            contract, bugs, nc, ns, ncegar = self.engine._infer_function(
                func, all_preds, contracts, flow_summaries,
            )
            contracts[func.name] = contract
            all_bugs.extend(bugs)
            total_constraints += nc
            total_solved += ns
            total_cegar += ncegar

        result = LiquidAnalysisResult(
            contracts=contracts,
            bugs=all_bugs,
            predicates_harvested=len(all_preds),
            constraints_generated=total_constraints,
            constraints_solved=total_solved,
            cegar_iterations=total_cegar,
            analysis_time_ms=result1.analysis_time_ms,
            harvested_preds=all_preds,
        )
        return result


# ═══════════════════════════════════════════════════════════════════════════
# 11. Predicate Template Library
# ═══════════════════════════════════════════════════════════════════════════

class PredicateTemplateLibrary:
    """A library of common predicate templates for CEGAR seed.

    When the harvester finds no relevant predicates, the CEGAR loop
    can draw from this library of commonly-useful templates.
    """

    @staticmethod
    def null_templates(var: str) -> List[Pred]:
        return [Pred.is_none(var), Pred.is_not_none(var)]

    @staticmethod
    def numeric_templates(var: str) -> List[Pred]:
        return [
            Pred.var_eq(var, 0),
            Pred.var_neq(var, 0),
            Pred.var_gt(var, 0),
            Pred.var_ge(var, 0),
            Pred.var_lt(var, 0),
            Pred.var_le(var, 0),
        ]

    @staticmethod
    def collection_templates(var: str) -> List[Pred]:
        return [
            Pred.len_eq(var, 0),
            Pred.len_gt(var, 0),
            Pred.len_ge(var, 0),
            Pred.len_ge(var, 1),
        ]

    @staticmethod
    def type_templates(var: str) -> List[Pred]:
        return [
            Pred.isinstance_(var, "int"),
            Pred.isinstance_(var, "str"),
            Pred.isinstance_(var, "float"),
            Pred.isinstance_(var, "list"),
            Pred.isinstance_(var, "dict"),
            Pred.isinstance_(var, "bool"),
        ]

    @staticmethod
    def all_templates(var: str) -> List[Pred]:
        """All common templates for a variable."""
        return (
            PredicateTemplateLibrary.null_templates(var)
            + PredicateTemplateLibrary.numeric_templates(var)
            + PredicateTemplateLibrary.collection_templates(var)
        )


# ═══════════════════════════════════════════════════════════════════════════
# 12. Liquid Type Display / Reporting
# ═══════════════════════════════════════════════════════════════════════════

class LiquidTypeReporter:
    """Format liquid type analysis results for human consumption."""

    @staticmethod
    def format_result(result: LiquidAnalysisResult) -> str:
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  Liquid Type Analysis Report")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        lines.append(f"Predicates harvested: {result.predicates_harvested}")
        lines.append(f"Constraints generated: {result.constraints_generated}")
        lines.append(f"Constraints solved: {result.constraints_solved}")
        lines.append(f"CEGAR iterations: {result.cegar_iterations}")
        lines.append(f"Bugs found: {len(result.bugs)}")
        lines.append(f"Analysis time: {result.analysis_time_ms:.1f}ms")
        lines.append("")

        # Contracts
        if result.contracts:
            lines.append("─" * 40)
            lines.append("  Inferred Contracts")
            lines.append("─" * 40)
            for name, contract in result.contracts.items():
                lines.append(contract.annotated_signature())
            lines.append("")

        # Bugs
        if result.bugs:
            lines.append("─" * 40)
            lines.append("  Bugs Detected")
            lines.append("─" * 40)
            for bug in result.bugs:
                lines.append(
                    f"  [{bug.kind.name}] L{bug.line}: {bug.message}"
                )
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_contracts_as_stubs(result: LiquidAnalysisResult) -> str:
        """Generate .pyi-style stub with liquid type annotations."""
        lines: List[str] = []
        lines.append("# Auto-generated liquid type stubs")
        lines.append("from typing import Annotated")
        lines.append("")
        for name, contract in result.contracts.items():
            lines.append(contract.annotated_signature())
            lines.append("    ...")
            lines.append("")
        return "\n".join(lines)
