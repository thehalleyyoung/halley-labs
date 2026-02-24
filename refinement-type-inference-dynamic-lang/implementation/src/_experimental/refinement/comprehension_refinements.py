"""
Refinement type inference for Python comprehensions and generator expressions.

Comprehension ``if``-filters map directly to refinement predicates on element
types, producing refined collection types such as ``list[{ν: int | ν > 0}]``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.refinement_lattice import (
    ANY_TYPE,
    BOOL_TYPE,
    FLOAT_TYPE,
    INT_TYPE,
    NEVER_TYPE,
    NONE_TYPE,
    OBJECT_TYPE,
    STR_TYPE,
    BaseTypeKind,
    BaseTypeR,
    Pred,
    PredOp,
    RefType,
)


# ---------------------------------------------------------------------------
# AST comparison-operator helpers
# ---------------------------------------------------------------------------

_AST_CMP_OP_MAP: Dict[type, str] = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<",
    ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
}

_TYPE_NAME_TO_BASE: Dict[str, BaseTypeR] = {
    "int": INT_TYPE, "float": FLOAT_TYPE, "str": STR_TYPE,
    "bool": BOOL_TYPE, "list": BaseTypeR(BaseTypeKind.LIST),
    "dict": BaseTypeR(BaseTypeKind.DICT), "set": BaseTypeR(BaseTypeKind.SET),
    "tuple": BaseTypeR(BaseTypeKind.TUPLE),
}


# ---------------------------------------------------------------------------
# Supporting data structures
# ---------------------------------------------------------------------------

@dataclass
class ComprehensionScope:
    """Tracks variables introduced by a comprehension ``for … in …`` clause."""

    target_vars: List[str] = field(default_factory=list)
    iter_type: RefType = field(default_factory=lambda: RefType.trivial(ANY_TYPE))
    bindings: Dict[str, RefType] = field(default_factory=dict)
    parent: Optional[ComprehensionScope] = None

    def lookup(self, name: str) -> Optional[RefType]:
        if name in self.bindings:
            return self.bindings[name]
        return self.parent.lookup(name) if self.parent is not None else None

    def bind(self, name: str, typ: RefType) -> None:
        self.bindings[name] = typ

    def child_scope(self) -> ComprehensionScope:
        return ComprehensionScope(parent=self)

    def all_bindings(self) -> Dict[str, RefType]:
        """Return flattened bindings from this scope and all parents."""
        result: Dict[str, RefType] = {}
        if self.parent is not None:
            result.update(self.parent.all_bindings())
        result.update(self.bindings)
        return result


@dataclass
class FilterChain:
    """Chain of ``if``-conditions merged into a conjunctive predicate."""

    conditions: List[ast.expr] = field(default_factory=list)
    predicates: List[Pred] = field(default_factory=list)

    def add(self, cond: ast.expr, pred: Pred) -> None:
        self.conditions.append(cond)
        self.predicates.append(pred)

    def merged_predicate(self) -> Pred:
        """Combine all collected predicates with logical AND."""
        result = Pred.true_()
        for p in self.predicates:
            result = result.and_(p)
        return result

    @property
    def is_empty(self) -> bool:
        return len(self.predicates) == 0


@dataclass
class AnalysisState:
    """Lightweight type environment snapshot for comprehension analysis."""

    bindings: Dict[str, RefType] = field(default_factory=dict)
    path_pred: Pred = field(default_factory=Pred.true_)
    scope: Optional[ComprehensionScope] = None

    def lookup(self, name: str) -> RefType:
        if self.scope is not None:
            found = self.scope.lookup(name)
            if found is not None:
                return found
        return self.bindings.get(name, RefType.trivial(ANY_TYPE))

    def bind(self, name: str, typ: RefType) -> AnalysisState:
        new_bindings = dict(self.bindings)
        new_bindings[name] = typ
        return AnalysisState(new_bindings, self.path_pred, self.scope)

    def with_scope(self, scope: ComprehensionScope) -> AnalysisState:
        return AnalysisState(dict(self.bindings), self.path_pred, scope)

    def add_path_pred(self, pred: Pred) -> AnalysisState:
        return AnalysisState(dict(self.bindings), self.path_pred.and_(pred), self.scope)

    def copy(self) -> AnalysisState:
        return AnalysisState(dict(self.bindings), self.path_pred, self.scope)


@dataclass
class PyRefinementType:
    """``RefType`` with collection-level metadata (container kind, key type)."""

    container_kind: str  # "list", "dict", "set", "generator"
    element_type: RefType
    key_type: Optional[RefType] = None
    length_pred: Optional[Pred] = None

    def pretty(self) -> str:
        elt = self.element_type.pretty()
        if self.container_kind == "dict" and self.key_type is not None:
            return f"dict[{self.key_type.pretty()}, {elt}]"
        return f"{self.container_kind}[{elt}]"

    def with_length(self, pred: Pred) -> PyRefinementType:
        return PyRefinementType(self.container_kind, self.element_type, self.key_type, pred)

    def to_ref_type(self) -> RefType:
        """Lower to a ``RefType`` over the container itself."""
        kind_map = {
            "list": BaseTypeKind.LIST, "dict": BaseTypeKind.DICT,
            "set": BaseTypeKind.SET, "generator": BaseTypeKind.LIST,
        }
        base_kind = kind_map.get(self.container_kind, BaseTypeKind.ANY)
        elt_base = self.element_type.base
        if self.key_type is not None:
            base = BaseTypeR(base_kind, (self.key_type.base, elt_base))
        else:
            base = BaseTypeR(base_kind, (elt_base,))
        pred = self.length_pred if self.length_pred is not None else Pred.true_()
        return RefType("ν", base, pred)


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class ComprehensionAnalyzer:
    """Infer refined collection types from comprehension expressions."""

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #

    def _analyze_collection_comp(
        self, generators: List[ast.comprehension], elt: ast.expr,
        state: AnalysisState, kind: str,
    ) -> PyRefinementType:
        """Shared logic for list/set/generator comprehensions."""
        comp_state = self.analyze_nested_comprehension(generators, state)
        elt_type = self._compute_element_type(elt, comp_state)
        filters = self._collect_all_filters(generators)
        target = self._primary_target(generators)
        if target is not None and filters:
            pred = self._merge_filter_predicates(filters, target)
            elt_type = RefType(elt_type.binder, elt_type.base, elt_type.pred.and_(pred))
        return PyRefinementType(container_kind=kind, element_type=elt_type)

    def analyze_listcomp(self, node: ast.ListComp, state: AnalysisState) -> PyRefinementType:
        return self._analyze_collection_comp(node.generators, node.elt, state, "list")

    def analyze_setcomp(self, node: ast.SetComp, state: AnalysisState) -> PyRefinementType:
        return self._analyze_collection_comp(node.generators, node.elt, state, "set")

    def analyze_genexpr(self, node: ast.GeneratorExp, state: AnalysisState) -> PyRefinementType:
        return self._analyze_collection_comp(node.generators, node.elt, state, "generator")

    def analyze_dictcomp(self, node: ast.DictComp, state: AnalysisState) -> PyRefinementType:
        comp_state = self.analyze_nested_comprehension(node.generators, state)
        key_type = self._compute_element_type(node.key, comp_state)
        val_type = self._compute_element_type(node.value, comp_state)
        filters = self._collect_all_filters(node.generators)
        target = self._primary_target(node.generators)
        if target is not None and filters:
            pred = self._merge_filter_predicates(filters, target)
            val_type = RefType(val_type.binder, val_type.base, val_type.pred.and_(pred))
        return PyRefinementType(container_kind="dict", element_type=val_type, key_type=key_type)

    # ------------------------------------------------------------------ #
    # Nested comprehension & walrus handling
    # ------------------------------------------------------------------ #

    def analyze_nested_comprehension(
        self, generators: List[ast.comprehension], state: AnalysisState
    ) -> AnalysisState:
        """Walk ``for … in … if …`` clauses, building scopes and refining."""
        current = state.copy()
        for gen in generators:
            scope = ComprehensionScope(parent=current.scope)
            iter_type = self._compute_element_type(gen.iter, current)
            scope.iter_type = iter_type
            target_names = self._extract_target_names(gen.target)
            scope.target_vars = target_names
            for name in target_names:
                scope.bind(name, self._element_type_of_iterable(iter_type))
            current = current.with_scope(scope)
            for if_clause in gen.ifs:
                for tgt in target_names:
                    extracted = self._extract_filter_refinement(if_clause, tgt)
                    if extracted is not None:
                        refined = scope.lookup(tgt)
                        if refined is not None:
                            scope.bind(tgt, RefType(refined.binder, refined.base,
                                                    refined.pred.and_(extracted)))
                        current = current.add_path_pred(extracted)

        return current

    def analyze_walrus_in_comp(
        self, node: ast.NamedExpr, comp_state: AnalysisState
    ) -> AnalysisState:
        """Handle ``:=`` inside a comprehension — leaks binding to enclosing scope."""
        val_type = self._compute_element_type(node.value, comp_state)
        target_name = node.target.id if isinstance(node.target, ast.Name) else "_"
        if comp_state.scope is not None:
            comp_state.scope.bind(target_name, val_type)
        return comp_state.bind(target_name, val_type)

    # ------------------------------------------------------------------ #
    # Filter → refinement extraction
    # ------------------------------------------------------------------ #

    def _extract_filter_refinement(
        self, condition: ast.expr, target_var: str
    ) -> Optional[Pred]:
        """Convert an ``if`` guard into a ``Pred``, or ``None`` if unmappable."""
        # isinstance(x, T)
        if isinstance(condition, ast.Call):
            pred = self._analyze_isinstance_filter(condition, target_var)
            if pred is not None:
                return pred
            pred = self._analyze_method_filter(condition, target_var)
            if pred is not None:
                return pred

        # x is not None / x is None / x > 0 / ...
        if isinstance(condition, ast.Compare):
            pred = self._analyze_none_filter(condition, target_var)
            if pred is not None:
                return pred
            pred = self._analyze_comparison_filter(condition, target_var)
            if pred is not None:
                return pred

        # ``not cond`` — negate the inner predicate
        if isinstance(condition, ast.UnaryOp) and isinstance(
            condition.op, ast.Not
        ):
            inner = self._extract_filter_refinement(
                condition.operand, target_var
            )
            if inner is not None:
                return inner.not_()

        # ``cond1 and cond2`` / ``cond1 or cond2``
        if isinstance(condition, ast.BoolOp):
            is_and = isinstance(condition.op, ast.And)
            parts: List[Pred] = []
            for val in condition.values:
                p = self._extract_filter_refinement(val, target_var)
                if p is not None:
                    parts.append(p)
            if parts:
                result = parts[0]
                for p in parts[1:]:
                    result = result.and_(p) if is_and else result.or_(p)
                return result

        # Bare name — ``if x`` ⟹ truthy(x)
        if isinstance(condition, ast.Name) and condition.id == target_var:
            return Pred.truthy(target_var)

        return None

    def _analyze_isinstance_filter(self, call: ast.Call, target_var: str) -> Optional[Pred]:
        """Handle ``isinstance(x, T)`` filters."""
        if not isinstance(call.func, ast.Name):
            return None
        if call.func.id != "isinstance":
            return None
        if len(call.args) != 2:
            return None

        # First arg must reference the target variable
        first = call.args[0]
        if not isinstance(first, ast.Name) or first.id != target_var:
            return None

        # Second arg: single type or tuple of types
        type_names = self._extract_type_names(call.args[1])
        if not type_names:
            return None

        if len(type_names) == 1:
            return Pred.isinstance_(target_var, type_names[0])
        # Union of isinstance checks
        preds = [Pred.isinstance_(target_var, t) for t in type_names]
        result = preds[0]
        for p in preds[1:]:
            result = result.or_(p)
        return result

    def _analyze_none_filter(self, compare: ast.Compare, target_var: str) -> Optional[Pred]:
        """Handle ``x is None`` / ``x is not None`` and equality variants."""
        if len(compare.ops) != 1 or len(compare.comparators) != 1:
            return None
        left, right, op = compare.left, compare.comparators[0], compare.ops[0]

        def _is_target(n: ast.expr) -> bool:
            return isinstance(n, ast.Name) and n.id == target_var

        def _is_none(n: ast.expr) -> bool:
            return isinstance(n, ast.Constant) and n.value is None

        if isinstance(op, ast.Is) and (
            (_is_target(left) and _is_none(right)) or (_is_target(right) and _is_none(left))
        ):
            return Pred.is_none(target_var)
        if isinstance(op, ast.IsNot) and (
            (_is_target(left) and _is_none(right)) or (_is_target(right) and _is_none(left))
        ):
            return Pred.is_not_none(target_var)
        if isinstance(op, ast.Eq) and _is_target(left) and _is_none(right):
            return Pred.is_none(target_var)
        if isinstance(op, ast.NotEq) and _is_target(left) and _is_none(right):
            return Pred.is_not_none(target_var)

        return None

    def _analyze_comparison_filter(self, compare: ast.Compare, target_var: str) -> Optional[Pred]:
        """Handle ``x > 0``, ``x <= 10``, ``x == 42``, chained comparisons, etc."""
        if len(compare.ops) != 1 or len(compare.comparators) != 1:
            # Chained comparisons: ``a < x < b`` → range
            return self._analyze_chained_comparison(compare, target_var)

        left = compare.left
        right = compare.comparators[0]
        op = compare.ops[0]

        op_str = _AST_CMP_OP_MAP.get(type(op))
        if op_str is None:
            return None

        # ``x <op> <const>``
        if isinstance(left, ast.Name) and left.id == target_var:
            val = self._extract_int_constant(right)
            if val is not None:
                return Pred.var_cmp(target_var, op_str, val)

        # ``<const> <op> x`` — flip the operator
        if isinstance(right, ast.Name) and right.id == target_var:
            val = self._extract_int_constant(left)
            if val is not None:
                flipped = self._flip_cmp(op_str)
                if flipped is not None:
                    return Pred.var_cmp(target_var, flipped, val)

        return None

    def _analyze_method_filter(self, call: ast.Call, target_var: str) -> Optional[Pred]:
        """Handle ``hasattr(x, 'a')``, ``callable(x)``, ``x.method()``."""
        # hasattr(x, "attr")
        if isinstance(call.func, ast.Name) and call.func.id == "hasattr":
            if len(call.args) == 2:
                obj = call.args[0]
                attr = call.args[1]
                if (isinstance(obj, ast.Name) and obj.id == target_var
                        and isinstance(attr, ast.Constant)
                        and isinstance(attr.value, str)):
                    return Pred.hasattr_(target_var, attr.value)

        # callable(x)
        if isinstance(call.func, ast.Name) and call.func.id == "callable":
            if (len(call.args) == 1 and isinstance(call.args[0], ast.Name)
                    and call.args[0].id == target_var):
                return Pred.isinstance_(target_var, "callable")

        # x.method() — record as hasattr evidence
        if isinstance(call.func, ast.Attribute):
            if isinstance(call.func.value, ast.Name):
                if call.func.value.id == target_var:
                    return Pred.hasattr_(target_var, call.func.attr)

        return None

    # ------------------------------------------------------------------ #
    # Element type computation
    # ------------------------------------------------------------------ #

    def _compute_element_type(self, elt_expr: ast.expr, state: AnalysisState) -> RefType:
        """Infer a ``RefType`` for the comprehension element expression."""
        if isinstance(elt_expr, ast.Constant):
            return self._reftype_of_constant(elt_expr.value)
        if isinstance(elt_expr, ast.Name):
            return state.lookup(elt_expr.id)
        if isinstance(elt_expr, ast.Tuple):
            elts = [self._compute_element_type(e, state) for e in elt_expr.elts]
            bases = tuple(e.base for e in elts) if elts else ()
            return RefType.trivial(BaseTypeR(BaseTypeKind.TUPLE, bases))
        if isinstance(elt_expr, ast.List):
            if elt_expr.elts:
                inner = self._compute_element_type(elt_expr.elts[0], state)
                return RefType.trivial(BaseTypeR(BaseTypeKind.LIST, (inner.base,)))
            return RefType.trivial(BaseTypeR(BaseTypeKind.LIST))
        if isinstance(elt_expr, ast.Subscript):
            return self._compute_subscript_type(elt_expr, state)
        if isinstance(elt_expr, ast.Attribute):
            return RefType.trivial(ANY_TYPE)
        if isinstance(elt_expr, ast.BinOp):
            return self._compute_binop_type(elt_expr, state)
        if isinstance(elt_expr, ast.UnaryOp):
            if isinstance(elt_expr.op, ast.Not):
                return RefType.trivial(BOOL_TYPE)
            return self._compute_element_type(elt_expr.operand, state)
        if isinstance(elt_expr, ast.Call):
            return self._compute_call_type(elt_expr, state)
        if isinstance(elt_expr, ast.IfExp):
            return self._join_ref_types(
                self._compute_element_type(elt_expr.body, state),
                self._compute_element_type(elt_expr.orelse, state),
            )
        if isinstance(elt_expr, ast.NamedExpr):
            val_type = self._compute_element_type(elt_expr.value, state)
            if isinstance(elt_expr.target, ast.Name) and state.scope is not None:
                state.scope.bind(elt_expr.target.id, val_type)
            return val_type
        if isinstance(elt_expr, ast.JoinedStr):
            return RefType.trivial(STR_TYPE)
        if isinstance(elt_expr, ast.Starred):
            return self._compute_element_type(elt_expr.value, state)

        return RefType.trivial(ANY_TYPE)

    # ------------------------------------------------------------------ #
    # Predicate merging & chaining
    # ------------------------------------------------------------------ #

    def _merge_filter_predicates(
        self, filters: List[ast.expr], target_var: str
    ) -> Pred:
        """Combine all ``if``-filters into a single conjunctive predicate."""
        chain = FilterChain()
        for f in filters:
            extracted = self._extract_filter_refinement(f, target_var)
            if extracted is not None:
                chain.add(f, extracted)
        return chain.merged_predicate()

    def _handle_chained_comprehension(
        self, outer_type: RefType, inner_comp: ast.ListComp,
    ) -> RefType:
        """Refine type for nested comprehensions (e.g. ``[y for x in xs for y in x]``)."""
        inner_state = AnalysisState()
        if inner_comp.generators:
            for name in self._extract_target_names(inner_comp.generators[0].target):
                inner_state = inner_state.bind(name, outer_type)
        return self.analyze_listcomp(inner_comp, inner_state).element_type

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_target_names(target: ast.expr) -> List[str]:
        """Collect all names bound by a comprehension target."""
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, ast.Tuple) or isinstance(target, ast.List):
            names: List[str] = []
            for elt in target.elts:
                names.extend(ComprehensionAnalyzer._extract_target_names(elt))
            return names
        if isinstance(target, ast.Starred):
            return ComprehensionAnalyzer._extract_target_names(target.value)
        return []

    @staticmethod
    def _extract_type_names(node: ast.expr) -> List[str]:
        """Extract type name(s) from the second argument of ``isinstance``."""
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, ast.Attribute):
            # e.g. ``collections.abc.Mapping`` — use final segment
            return [node.attr]
        if isinstance(node, ast.Tuple):
            result: List[str] = []
            for elt in node.elts:
                result.extend(
                    ComprehensionAnalyzer._extract_type_names(elt)
                )
            return result
        return []

    @staticmethod
    def _extract_int_constant(node: ast.expr) -> Optional[int]:
        """Return integer value if *node* is a constant int (including negated)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, int)):
            return -node.operand.value
        return None

    @staticmethod
    def _flip_cmp(op: str) -> Optional[str]:
        """Flip a comparison for ``const <op> x`` → ``x <flipped> const``."""
        return {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}.get(op)

    def _analyze_chained_comparison(
        self, compare: ast.Compare, target_var: str
    ) -> Optional[Pred]:
        """Handle chained comparisons like ``0 < x < 10``."""
        if len(compare.ops) != 2 or len(compare.comparators) != 2:
            return None

        left = compare.left
        mid = compare.comparators[0]
        right = compare.comparators[1]

        # Pattern: ``lo <op1> x <op2> hi``
        if not (isinstance(mid, ast.Name) and mid.id == target_var):
            return None

        lo = self._extract_int_constant(left)
        hi = self._extract_int_constant(right)
        if lo is None or hi is None:
            return None

        op1_str = _AST_CMP_OP_MAP.get(type(compare.ops[0]))
        op2_str = _AST_CMP_OP_MAP.get(type(compare.ops[1]))
        if op1_str is None or op2_str is None:
            return None

        # Convert ``lo <= x < hi`` to ``in_range`` when possible
        if op1_str in ("<=", "<") and op2_str in ("<=", "<"):
            effective_lo = lo if op1_str == "<=" else lo + 1
            effective_hi = hi if op2_str == "<=" else hi - 1
            return Pred.in_range(target_var, effective_lo, effective_hi)

        # Fall back to conjunction of two simple comparisons
        flipped1 = self._flip_cmp(op1_str)
        if flipped1 is not None:
            p1 = Pred.var_cmp(target_var, flipped1, lo)
            p2 = Pred.var_cmp(target_var, op2_str, hi)
            return p1.and_(p2)

        return None

    @staticmethod
    def _reftype_of_constant(value: object) -> RefType:
        """Map a Python constant to its ``RefType``."""
        if isinstance(value, bool):
            return RefType.trivial(BOOL_TYPE)
        if isinstance(value, int):
            return RefType("ν", INT_TYPE, Pred.var_eq("ν", value))
        if isinstance(value, float):
            return RefType.trivial(FLOAT_TYPE)
        if isinstance(value, str):
            return RefType.trivial(STR_TYPE)
        if value is None:
            return RefType.trivial(NONE_TYPE)
        return RefType.trivial(ANY_TYPE)

    @staticmethod
    def _element_type_of_iterable(iter_type: RefType) -> RefType:
        """Extract element type: ``list[T]`` → T, ``dict[K,V]`` → K, etc."""
        base = iter_type.base
        if base.kind in (BaseTypeKind.LIST, BaseTypeKind.SET, BaseTypeKind.TUPLE):
            return RefType.trivial(base.type_args[0]) if base.type_args else RefType.trivial(ANY_TYPE)
        if base.kind == BaseTypeKind.DICT:
            return RefType.trivial(base.type_args[0]) if base.type_args else RefType.trivial(ANY_TYPE)
        if base.kind == BaseTypeKind.STR:
            return RefType.trivial(STR_TYPE)
        return RefType.trivial(ANY_TYPE)

    def _collect_all_filters(
        self, generators: List[ast.comprehension]
    ) -> List[ast.expr]:
        """Gather every ``if``-clause across all generator clauses."""
        result: List[ast.expr] = []
        for gen in generators:
            result.extend(gen.ifs)
        return result

    @staticmethod
    def _primary_target(
        generators: List[ast.comprehension],
    ) -> Optional[str]:
        """Return the name of the first simple iteration target, if any."""
        if generators:
            names = ComprehensionAnalyzer._extract_target_names(
                generators[0].target
            )
            if names:
                return names[0]
        return None

    def _compute_binop_type(
        self, node: ast.BinOp, state: AnalysisState
    ) -> RefType:
        """Infer result type of a binary operation."""
        left_t = self._compute_element_type(node.left, state)
        right_t = self._compute_element_type(node.right, state)

        lb = left_t.base.kind
        rb = right_t.base.kind

        # str + str → str;  str * int → str
        if lb == BaseTypeKind.STR and isinstance(node.op, ast.Add):
            return RefType.trivial(STR_TYPE)
        if lb == BaseTypeKind.STR and isinstance(node.op, ast.Mult):
            return RefType.trivial(STR_TYPE)

        # list + list → list
        if lb == BaseTypeKind.LIST and isinstance(node.op, ast.Add):
            return RefType.trivial(left_t.base)

        # Numeric operations
        if lb == BaseTypeKind.FLOAT or rb == BaseTypeKind.FLOAT:
            return RefType.trivial(FLOAT_TYPE)
        if lb == BaseTypeKind.INT and rb == BaseTypeKind.INT:
            if isinstance(node.op, ast.Div):
                return RefType.trivial(FLOAT_TYPE)
            return RefType.trivial(INT_TYPE)

        return RefType.trivial(ANY_TYPE)

    def _compute_call_type(self, node: ast.Call, state: AnalysisState) -> RefType:
        """Infer return type for common built-in calls."""
        if isinstance(node.func, ast.Name):
            name = node.func.id
            _builtin: Dict[str, BaseTypeR] = {
                "int": INT_TYPE, "float": FLOAT_TYPE, "str": STR_TYPE,
                "bool": BOOL_TYPE, "len": INT_TYPE, "abs": INT_TYPE,
                "min": INT_TYPE, "max": INT_TYPE, "sum": INT_TYPE,
                "round": INT_TYPE, "ord": INT_TYPE, "chr": STR_TYPE,
                "repr": STR_TYPE, "hex": STR_TYPE, "oct": STR_TYPE,
                "bin": STR_TYPE, "type": OBJECT_TYPE,
            }
            if name in _builtin:
                return RefType.trivial(_builtin[name])
            _container: Dict[str, BaseTypeKind] = {
                "list": BaseTypeKind.LIST, "set": BaseTypeKind.SET,
                "dict": BaseTypeKind.DICT, "tuple": BaseTypeKind.TUPLE,
                "frozenset": BaseTypeKind.SET,
            }
            if name in _container:
                return RefType.trivial(BaseTypeR(_container[name]))
            func_type = state.lookup(name)
            if func_type.base.kind == BaseTypeKind.CALLABLE and func_type.base.return_type is not None:
                return RefType.trivial(func_type.base.return_type)
        return RefType.trivial(ANY_TYPE)

    def _compute_subscript_type(self, node: ast.Subscript, state: AnalysisState) -> RefType:
        """Infer result type for ``x[i]``."""
        base = self._compute_element_type(node.value, state).base
        if base.kind == BaseTypeKind.LIST and base.type_args:
            return RefType.trivial(base.type_args[0])
        if base.kind == BaseTypeKind.TUPLE and base.type_args:
            idx = self._extract_int_constant(node.slice) if not isinstance(node.slice, ast.Slice) else None
            if idx is not None and 0 <= idx < len(base.type_args):
                return RefType.trivial(base.type_args[idx])
            return RefType.trivial(base.type_args[0])
        if base.kind == BaseTypeKind.DICT and len(base.type_args) >= 2:
            return RefType.trivial(base.type_args[1])
        if base.kind == BaseTypeKind.STR:
            return RefType.trivial(STR_TYPE)
        return RefType.trivial(ANY_TYPE)

    @staticmethod
    def _join_ref_types(a: RefType, b: RefType) -> RefType:
        """Conservative join: same base → OR predicates, else widen to ANY."""
        if a.base == b.base:
            return RefType(a.binder, a.base, a.pred.or_(b.pred))
        if {a.base.kind, b.base.kind} == {BaseTypeKind.INT, BaseTypeKind.FLOAT}:
            return RefType.trivial(FLOAT_TYPE)
        return RefType.trivial(ANY_TYPE)
