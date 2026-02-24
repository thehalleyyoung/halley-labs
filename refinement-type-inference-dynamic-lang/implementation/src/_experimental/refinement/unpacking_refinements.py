"""
Structural refinements derived from Python unpacking patterns.

Analyzes tuple/star/nested unpacking in assignments, for-loops, and
with-statements to derive length predicates, element-type constraints,
and non-emptiness guarantees on container values.

Examples of patterns handled:

    a, b = pair          → pair has exactly 2 elements
    first, *rest = items → items has ≥ 1 element, rest is a list
    (x, y), z = nested  → nested[0] is a 2-tuple
    for k, v in d.items(): → d is non-empty, k/v typed from dict
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from src.refinement_lattice import (
    BaseTypeKind,
    BaseTypeR,
    Pred,
    PredOp,
    RefType,
    INT_TYPE,
    FLOAT_TYPE,
    STR_TYPE,
    BOOL_TYPE,
    NONE_TYPE,
    ANY_TYPE,
    NEVER_TYPE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LIST_TYPE = BaseTypeR(BaseTypeKind.LIST)
TUPLE_TYPE = BaseTypeR(BaseTypeKind.TUPLE)
DICT_TYPE = BaseTypeR(BaseTypeKind.DICT)
CALLABLE_TYPE = BaseTypeR(BaseTypeKind.CALLABLE)


def _name_of(node: ast.expr) -> Optional[str]:
    """Extract a simple variable name from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def _names_of(targets: List[ast.expr]) -> List[Optional[str]]:
    """Extract variable names from a list of target nodes."""
    return [_name_of(t) for t in targets]


def _is_starred(node: ast.expr) -> bool:
    """Return True if *node* is an ``ast.Starred`` node."""
    return isinstance(node, ast.Starred)


def _find_star_index(elts: List[ast.expr]) -> int:
    """Return the index of the starred element, or -1 if absent."""
    for i, elt in enumerate(elts):
        if _is_starred(elt):
            return i
    return -1


# ---------------------------------------------------------------------------
# Supporting data structures
# ---------------------------------------------------------------------------

class ConstraintKind(Enum):
    """Kind of structural constraint on the unpacked value."""
    LEN_EXACT = auto()   # exactly N elements
    LEN_GE = auto()      # at least N elements
    ELEMENT_TYPE = auto() # element at index has a given type
    NON_EMPTY = auto()    # container is non-empty
    NESTED = auto()       # nested tuple structure at an index


@dataclass(frozen=True)
class StructuralConstraint:
    """A constraint on the structure of the unpacked value.

    Attributes:
        kind: What structural property is being asserted.
        target_var: The source variable being constrained.
        value: Integer argument (length, index) depending on *kind*.
        element_type: Optional element refinement type.
        children: Nested constraints for ``NESTED`` kind.
    """

    kind: ConstraintKind
    target_var: str
    value: int = 0
    element_type: Optional[RefType] = None
    children: Tuple[StructuralConstraint, ...] = ()

    # -- convenience constructors -------------------------------------------

    @staticmethod
    def len_exact(var: str, n: int) -> StructuralConstraint:
        return StructuralConstraint(ConstraintKind.LEN_EXACT, var, value=n)

    @staticmethod
    def len_ge(var: str, n: int) -> StructuralConstraint:
        return StructuralConstraint(ConstraintKind.LEN_GE, var, value=n)

    @staticmethod
    def non_empty(var: str) -> StructuralConstraint:
        return StructuralConstraint(ConstraintKind.NON_EMPTY, var)

    @staticmethod
    def element_at(var: str, idx: int, rtype: RefType) -> StructuralConstraint:
        return StructuralConstraint(
            ConstraintKind.ELEMENT_TYPE, var, value=idx, element_type=rtype,
        )

    @staticmethod
    def nested(var: str, idx: int, children: Tuple[StructuralConstraint, ...]) -> StructuralConstraint:
        return StructuralConstraint(
            ConstraintKind.NESTED, var, value=idx, children=children,
        )


@dataclass(frozen=True)
class UnpackTarget:
    """Description of a single target inside an unpacking pattern.

    Attributes:
        name: Variable name bound by the target (``None`` for ``_``).
        is_star: Whether this is a starred target (``*rest``).
        index: Positional index inside the enclosing tuple.
        nested: If the target is itself a tuple, the recursive structure.
    """

    name: Optional[str]
    is_star: bool = False
    index: int = 0
    nested: Optional[Tuple[UnpackTarget, ...]] = None

    @property
    def is_nested(self) -> bool:
        return self.nested is not None

    @staticmethod
    def from_ast(node: ast.expr, index: int = 0) -> UnpackTarget:
        """Build an ``UnpackTarget`` from an AST target node."""
        if isinstance(node, ast.Starred):
            inner = _name_of(node.value)
            return UnpackTarget(name=inner, is_star=True, index=index)
        if isinstance(node, ast.Tuple):
            children = tuple(
                UnpackTarget.from_ast(elt, idx) for idx, elt in enumerate(node.elts)
            )
            return UnpackTarget(name=None, index=index, nested=children)
        return UnpackTarget(name=_name_of(node), index=index)


# ---------------------------------------------------------------------------
# AnalysisState
# ---------------------------------------------------------------------------

@dataclass
class AnalysisState:
    """Mapping from variable names to their current refinement types,
    together with an accumulated set of structural predicates.

    This is a lightweight state threaded through the unpacking analysis;
    the caller is responsible for merging it back into the wider abstract
    interpretation state.
    """

    bindings: Dict[str, RefType] = field(default_factory=dict)
    predicates: Dict[str, List[Pred]] = field(default_factory=dict)
    constraints: List[StructuralConstraint] = field(default_factory=list)

    # -- query --------------------------------------------------------------

    def get_type(self, var: str) -> Optional[RefType]:
        return self.bindings.get(var)

    def get_predicates(self, var: str) -> List[Pred]:
        return self.predicates.get(var, [])

    # -- mutation -----------------------------------------------------------

    def bind(self, var: str, rtype: RefType) -> AnalysisState:
        """Bind *var* to *rtype*, returning ``self`` for chaining."""
        self.bindings[var] = rtype
        return self

    def add_pred(self, var: str, pred: Pred) -> AnalysisState:
        self.predicates.setdefault(var, []).append(pred)
        return self

    def add_constraint(self, c: StructuralConstraint) -> AnalysisState:
        self.constraints.append(c)
        return self

    def copy(self) -> AnalysisState:
        return AnalysisState(
            bindings=dict(self.bindings),
            predicates={k: list(v) for k, v in self.predicates.items()},
            constraints=list(self.constraints),
        )

    def merge(self, other: AnalysisState) -> AnalysisState:
        """Merge *other* into ``self`` (union of bindings / predicates)."""
        for var, rtype in other.bindings.items():
            if var not in self.bindings:
                self.bindings[var] = rtype
        for var, preds in other.predicates.items():
            self.predicates.setdefault(var, []).extend(preds)
        self.constraints.extend(other.constraints)
        return self


# ---------------------------------------------------------------------------
# Value-source detection helpers
# ---------------------------------------------------------------------------

_DICT_ITER_METHODS = frozenset({"items", "values", "keys"})


def _is_method_call(node: ast.expr, method: str) -> bool:
    """Check if *node* is a call like ``x.<method>()``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == method:
        return True
    return False


def _is_builtin_call(node: ast.expr, name: str) -> bool:
    """Check if *node* is a call like ``<name>(...)``."""
    if not isinstance(node, ast.Call):
        return False
    return isinstance(node.func, ast.Name) and node.func.id == name


def _attribute_receiver(node: ast.Call) -> Optional[str]:
    """For ``x.method(...)``, return ``'x'``."""
    if isinstance(node.func, ast.Attribute):
        return _name_of(node.func.value)
    return None


# ---------------------------------------------------------------------------
# UnpackingAnalyzer
# ---------------------------------------------------------------------------

class UnpackingAnalyzer:
    """Derive refinement types from Python unpacking patterns.

    The analyzer is stateless: each ``analyze_*`` method takes a state and
    returns a new (or mutated) state with additional bindings / predicates.
    """

    # -----------------------------------------------------------------------
    # Top-level dispatch
    # -----------------------------------------------------------------------

    def analyze_assignment(
        self, node: ast.Assign, state: AnalysisState,
    ) -> AnalysisState:
        """Dispatch an ``ast.Assign`` to the appropriate handler.

        Handles plain, tuple, starred, and nested targets.
        """
        if len(node.targets) != 1:
            # Multiple assignment targets (a = b = expr) – refine each.
            for target in node.targets:
                state = self._dispatch_target(target, node.value, state)
            return state
        return self._dispatch_target(node.targets[0], node.value, state)

    def _dispatch_target(
        self, target: ast.expr, value: ast.expr, state: AnalysisState,
    ) -> AnalysisState:
        """Route a single assignment target to its handler."""
        if isinstance(target, ast.Name):
            return self._analyze_simple_assign(target, value, state)
        if isinstance(target, (ast.Tuple, ast.List)):
            star_idx = _find_star_index(target.elts)
            if star_idx >= 0:
                return self.analyze_star_unpack(
                    target.elts, star_idx, value, state,
                )
            has_nested = any(isinstance(e, ast.Tuple) for e in target.elts)
            if has_nested:
                return self.analyze_nested_unpack(target, value, state)
            return self.analyze_tuple_unpack(target.elts, value, state)
        # Attribute / subscript targets – no structural refinement.
        return state

    # -----------------------------------------------------------------------
    # Simple assignment  (x = expr)
    # -----------------------------------------------------------------------

    def _analyze_simple_assign(
        self, target: ast.Name, value: ast.expr, state: AnalysisState,
    ) -> AnalysisState:
        """Refine from a plain assignment ``x = expr``."""
        var = target.id
        inferred = self._infer_base_type(value, state)
        if inferred is not None:
            state.bind(var, inferred)
        return state

    # -----------------------------------------------------------------------
    # Tuple unpacking  (a, b, c = expr)
    # -----------------------------------------------------------------------

    def analyze_tuple_unpack(
        self,
        targets: List[ast.expr],
        value: ast.expr,
        state: AnalysisState,
    ) -> AnalysisState:
        """Handle ``a, b, c = expr`` → *expr* has exactly ``len(targets)``
        elements."""
        n = len(targets)
        val_var = _name_of(value)

        # Structural constraint on the RHS value.
        if val_var is not None:
            state.add_pred(val_var, Pred.len_eq(val_var, n))
            state.add_constraint(StructuralConstraint.len_exact(val_var, n))

        # Extract per-element types and bind targets.
        rhs_type = state.get_type(val_var) if val_var else None
        elem_types = self._extract_element_types(rhs_type, n) if rhs_type else None

        for i, tgt in enumerate(targets):
            name = _name_of(tgt)
            if name is None:
                # Nested target – recurse.
                if isinstance(tgt, ast.Tuple):
                    state = self.analyze_nested_unpack(tgt, value, state)
                continue
            if elem_types and i < len(elem_types):
                state.bind(name, elem_types[i])
            else:
                state.bind(name, RefType.trivial(ANY_TYPE))
            state.add_pred(name, Pred.is_not_none(name))

        return state

    # -----------------------------------------------------------------------
    # Star unpacking  (first, *middle, last = expr)
    # -----------------------------------------------------------------------

    def analyze_star_unpack(
        self,
        targets: List[ast.expr],
        star_idx: int,
        value: ast.expr,
        state: AnalysisState,
    ) -> AnalysisState:
        """Handle starred unpacking.

        ``first, *middle, last = expr``  →  *expr* has ≥ 2 elements,
        *middle* is a ``list``.
        """
        n_fixed = len(targets) - 1  # everything except the star
        min_len, has_star, _ = self._count_unpack_targets_from_list(targets)
        val_var = _name_of(value)

        if val_var is not None:
            state.add_pred(val_var, Pred.len_ge(val_var, n_fixed))
            state.add_constraint(StructuralConstraint.len_ge(val_var, n_fixed))

        rhs_type = state.get_type(val_var) if val_var else None
        elem_types = (
            self._extract_element_types(rhs_type, len(targets))
            if rhs_type
            else None
        )

        for i, tgt in enumerate(targets):
            if _is_starred(tgt):
                star_name = _name_of(tgt.value)
                if star_name is not None:
                    # The starred variable collects a list of the remaining
                    # elements.
                    star_ref = RefType("v", LIST_TYPE, Pred.true_())
                    state.bind(star_name, star_ref)
                continue

            name = _name_of(tgt)
            if name is None:
                continue

            if elem_types and i < len(elem_types):
                state.bind(name, elem_types[i])
            else:
                state.bind(name, RefType.trivial(ANY_TYPE))
            state.add_pred(name, Pred.is_not_none(name))

        return state

    # -----------------------------------------------------------------------
    # Nested unpacking  ((a, b), c = expr)
    # -----------------------------------------------------------------------

    def analyze_nested_unpack(
        self,
        target: ast.Tuple,
        value: ast.expr,
        state: AnalysisState,
    ) -> AnalysisState:
        """Handle nested tuple unpacking.

        ``(a, b), c = expr``  →  ``expr[0]`` is a 2-element tuple.
        """
        val_var = _name_of(value)

        for idx, elt in enumerate(target.elts):
            if isinstance(elt, (ast.Tuple, ast.List)):
                inner_len = len(elt.elts)
                # Constraint: value[idx] is a tuple of *inner_len* elements.
                if val_var is not None:
                    inner_constraints = tuple(
                        StructuralConstraint.len_exact(f"{val_var}[{idx}]", inner_len),
                    )
                    state.add_constraint(
                        StructuralConstraint.nested(val_var, idx, inner_constraints)
                    )

                # Bind inner names.
                for j, inner_elt in enumerate(elt.elts):
                    inner_name = _name_of(inner_elt)
                    if inner_name is not None:
                        state.bind(inner_name, RefType.trivial(ANY_TYPE))
                        state.add_pred(inner_name, Pred.is_not_none(inner_name))
            else:
                name = _name_of(elt)
                if name is not None:
                    state.bind(name, RefType.trivial(ANY_TYPE))
                    state.add_pred(name, Pred.is_not_none(name))

        # Top-level length constraint.
        if val_var is not None:
            n = len(target.elts)
            state.add_pred(val_var, Pred.len_eq(val_var, n))
            state.add_constraint(StructuralConstraint.len_exact(val_var, n))

        return state

    # -----------------------------------------------------------------------
    # For-loop unpacking  (for k, v in d.items(): ...)
    # -----------------------------------------------------------------------

    def analyze_for_unpack(
        self,
        target: ast.expr,
        iter_expr: ast.expr,
        state: AnalysisState,
    ) -> AnalysisState:
        """Handle ``for key, value in d.items(): ...`` and similar patterns.

        Recognized sources: ``dict.items()``, ``dict.values()``,
        ``enumerate()``, ``zip()``.
        """
        source = self._infer_value_source(iter_expr)
        iter_var = self._iter_source_var(iter_expr)

        # The iterator is non-empty inside the loop body.
        if iter_var is not None:
            state.add_pred(iter_var, Pred.is_not_none(iter_var))
            state.add_constraint(StructuralConstraint.non_empty(iter_var))

        if source == "items" and isinstance(target, (ast.Tuple, ast.List)):
            return self._handle_dict_items_unpack(target, iter_expr, state)
        if source == "enumerate" and isinstance(target, (ast.Tuple, ast.List)):
            return self._handle_enumerate_unpack(target, iter_expr, state)
        if source == "zip" and isinstance(target, (ast.Tuple, ast.List)):
            return self._handle_zip_unpack(target, iter_expr, state)
        if source == "values":
            return self._handle_dict_values_unpack(target, iter_expr, state)

        # Generic tuple unpack inside loop header.
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                name = _name_of(elt)
                if name is not None:
                    state.bind(name, RefType.trivial(ANY_TYPE))
                    state.add_pred(name, Pred.is_not_none(name))
        else:
            name = _name_of(target)
            if name is not None:
                state.bind(name, RefType.trivial(ANY_TYPE))

        return state

    # -----------------------------------------------------------------------
    # With-statement unpacking  (with open(f) as fh: ...)
    # -----------------------------------------------------------------------

    def analyze_with_unpack(
        self,
        target: ast.expr,
        context_expr: ast.expr,
        state: AnalysisState,
    ) -> AnalysisState:
        """Handle ``with open(f) as fh: ...``.

        The bound variable is guaranteed non-``None`` inside the body.
        """
        name = _name_of(target)
        if name is None:
            return state

        # The context manager result is non-None inside the with-body.
        state.add_pred(name, Pred.is_not_none(name))

        # Attempt to infer a base type from the context expression.
        inferred = self._infer_context_type(context_expr)
        if inferred is not None:
            state.bind(name, inferred)
        else:
            state.bind(name, RefType(name, ANY_TYPE, Pred.is_not_none(name)))

        return state

    # -----------------------------------------------------------------------
    # Target counting
    # -----------------------------------------------------------------------

    def _count_unpack_targets(
        self, target: ast.expr,
    ) -> Tuple[int, bool, int]:
        """Count the elements required by *target*.

        Returns ``(total_targets, has_star, min_elements)`` where
        *min_elements* is the minimum length the RHS must have.
        """
        if isinstance(target, (ast.Tuple, ast.List)):
            return self._count_unpack_targets_from_list(target.elts)
        return (1, False, 1)

    def _count_unpack_targets_from_list(
        self, elts: List[ast.expr],
    ) -> Tuple[int, bool, int]:
        total = len(elts)
        has_star = any(_is_starred(e) for e in elts)
        if has_star:
            min_elements = total - 1
        else:
            min_elements = total
        return (total, has_star, min_elements)

    # -----------------------------------------------------------------------
    # Element-type extraction
    # -----------------------------------------------------------------------

    def _extract_element_types(
        self, value_type: RefType, count: int,
    ) -> List[RefType]:
        """Extract *count* element types from a container refinement type.

        For a ``TUPLE`` base type with known parameters we can distribute
        the element types.  Otherwise we return the same (container-element)
        type for every position.
        """
        base = value_type.base
        if base.kind == BaseTypeKind.TUPLE and base.params:
            # Parametric tuple – return each parameter type.
            result: List[RefType] = []
            for i in range(count):
                if i < len(base.params):
                    result.append(base.params[i])
                else:
                    result.append(RefType.trivial(ANY_TYPE))
            return result

        if base.kind == BaseTypeKind.LIST and base.params:
            elem = base.params[0]
            return [elem] * count

        if base.kind == BaseTypeKind.DICT and base.params and len(base.params) >= 2:
            # dict unpacking yields keys.
            return [base.params[0]] * count

        # Fallback: each element is ``Any``.
        return [RefType.trivial(ANY_TYPE)] * count

    # -----------------------------------------------------------------------
    # Value-source inference
    # -----------------------------------------------------------------------

    def _infer_value_source(self, value: ast.expr) -> Optional[str]:
        """Detect special iteration sources.

        Returns one of ``'items'``, ``'values'``, ``'keys'``,
        ``'enumerate'``, ``'zip'``, or ``None``.
        """
        if isinstance(value, ast.Call):
            func = value.func
            # dict.items() / dict.values() / dict.keys()
            if isinstance(func, ast.Attribute) and func.attr in _DICT_ITER_METHODS:
                return func.attr
            # enumerate(...)
            if isinstance(func, ast.Name) and func.id == "enumerate":
                return "enumerate"
            # zip(...)
            if isinstance(func, ast.Name) and func.id == "zip":
                return "zip"
        return None

    def _iter_source_var(self, value: ast.expr) -> Optional[str]:
        """Return the primary variable being iterated over."""
        if isinstance(value, ast.Call):
            func = value.func
            if isinstance(func, ast.Attribute):
                return _name_of(func.value)
            if isinstance(func, ast.Name) and value.args:
                return _name_of(value.args[0])
        return _name_of(value)

    # -----------------------------------------------------------------------
    # Dict items / values helpers
    # -----------------------------------------------------------------------

    def _handle_dict_items_unpack(
        self,
        target: ast.Tuple | ast.List,
        iter_expr: ast.expr,
        state: AnalysisState,
    ) -> AnalysisState:
        """``for key, value in d.items()``."""
        elts = target.elts
        if len(elts) != 2:
            return state

        key_name = _name_of(elts[0])
        val_name = _name_of(elts[1])
        dict_var = _attribute_receiver(iter_expr)  # type: ignore[arg-type]

        if key_name is not None:
            state.bind(key_name, RefType.trivial(ANY_TYPE))
            state.add_pred(key_name, Pred.is_not_none(key_name))
        if val_name is not None:
            state.bind(val_name, RefType.trivial(ANY_TYPE))
            state.add_pred(val_name, Pred.is_not_none(val_name))

        if dict_var is not None and key_name and val_name:
            state = self._refine_dict_unpack(dict_var, key_name, val_name, state)

        return state

    def _handle_dict_values_unpack(
        self,
        target: ast.expr,
        iter_expr: ast.expr,
        state: AnalysisState,
    ) -> AnalysisState:
        """``for v in d.values()``."""
        name = _name_of(target)
        if name is not None:
            state.bind(name, RefType.trivial(ANY_TYPE))
            state.add_pred(name, Pred.is_not_none(name))
        return state

    def _handle_enumerate_unpack(
        self,
        target: ast.Tuple | ast.List,
        iter_expr: ast.expr,
        state: AnalysisState,
    ) -> AnalysisState:
        """``for idx, val in enumerate(xs)``."""
        elts = target.elts
        if len(elts) != 2:
            return state

        idx_name = _name_of(elts[0])
        val_name = _name_of(elts[1])
        iter_var = None
        if isinstance(iter_expr, ast.Call) and iter_expr.args:
            iter_var = _name_of(iter_expr.args[0])

        if idx_name is not None:
            state.bind(idx_name, RefType(idx_name, INT_TYPE, Pred.var_ge(idx_name, 0)))

        if val_name is not None:
            state.bind(val_name, RefType.trivial(ANY_TYPE))
            state.add_pred(val_name, Pred.is_not_none(val_name))

        if idx_name and val_name and iter_var:
            state = self._refine_enumerate_unpack(
                idx_name, val_name, iter_var, state,
            )

        return state

    def _handle_zip_unpack(
        self,
        target: ast.Tuple | ast.List,
        iter_expr: ast.expr,
        state: AnalysisState,
    ) -> AnalysisState:
        """``for a, b in zip(xs, ys)``."""
        names = _names_of(target.elts)
        for n in names:
            if n is not None:
                state.bind(n, RefType.trivial(ANY_TYPE))
                state.add_pred(n, Pred.is_not_none(n))

        if isinstance(iter_expr, ast.Call):
            valid_names = [n for n in names if n is not None]
            state = self._refine_zip_unpack(
                valid_names, iter_expr.args, state,
            )

        return state

    # -----------------------------------------------------------------------
    # Refinement helpers for specific patterns
    # -----------------------------------------------------------------------

    def _refine_dict_unpack(
        self,
        dict_var: str,
        key_target: str,
        val_target: str,
        state: AnalysisState,
    ) -> AnalysisState:
        """Propagate dict-level type information to key/value bindings.

        If the dict variable already has a ``DICT`` refinement type with
        known key/value parameter types, project them onto the loop targets.
        """
        dict_type = state.get_type(dict_var)
        if dict_type is None:
            return state

        base = dict_type.base
        if base.kind != BaseTypeKind.DICT:
            return state

        if base.params and len(base.params) >= 2:
            key_rt, val_rt = base.params[0], base.params[1]
            state.bind(key_target, key_rt)
            state.bind(val_target, val_rt)
        else:
            state.add_pred(dict_var, Pred.is_not_none(dict_var))

        state.add_constraint(StructuralConstraint.non_empty(dict_var))
        return state

    def _refine_enumerate_unpack(
        self,
        idx_target: str,
        val_target: str,
        iter_var: str,
        state: AnalysisState,
    ) -> AnalysisState:
        """Refine types from ``enumerate(iter_var)``.

        The index is a non-negative ``int``; the value type is projected
        from the iterable's element type when known.
        """
        state.add_pred(idx_target, Pred.var_ge(idx_target, 0))
        state.add_constraint(StructuralConstraint.non_empty(iter_var))

        iter_type = state.get_type(iter_var)
        if iter_type is not None and iter_type.base.params:
            elem_rt = iter_type.base.params[0]
            state.bind(val_target, elem_rt)

        return state

    def _refine_zip_unpack(
        self,
        targets: List[str],
        zip_args: List[ast.expr],
        state: AnalysisState,
    ) -> AnalysisState:
        """Refine types from ``zip(a, b, ...)``.

        Each target gets the element type of the corresponding zip argument
        when the argument's type is known.
        """
        for i, (tgt, arg) in enumerate(zip(targets, zip_args)):
            arg_var = _name_of(arg)
            if arg_var is None:
                continue
            state.add_constraint(StructuralConstraint.non_empty(arg_var))

            arg_type = state.get_type(arg_var)
            if arg_type is not None and arg_type.base.params:
                elem_rt = arg_type.base.params[0]
                state.bind(tgt, elem_rt)

        return state

    # -----------------------------------------------------------------------
    # Base-type inference for RHS expressions
    # -----------------------------------------------------------------------

    _BUILTIN_TYPE_MAP: Dict[str, BaseTypeR] = {
        "int": INT_TYPE,
        "float": FLOAT_TYPE,
        "str": STR_TYPE,
        "bool": BOOL_TYPE,
        "list": LIST_TYPE,
        "dict": DICT_TYPE,
        "tuple": TUPLE_TYPE,
    }

    def _infer_base_type(
        self, value: ast.expr, state: AnalysisState,
    ) -> Optional[RefType]:
        """Best-effort inference of a base refinement type for *value*."""
        # Constant literals.
        if isinstance(value, ast.Constant):
            return self._reftype_for_constant(value)

        # Variable reference – propagate existing type.
        if isinstance(value, ast.Name):
            return state.get_type(value.id)

        # Constructor call  (e.g. ``int(...)``, ``list(...)``).
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            base = self._BUILTIN_TYPE_MAP.get(value.func.id)
            if base is not None:
                return RefType.trivial(base)

        # List / tuple / set / dict displays.
        if isinstance(value, ast.List):
            return RefType("v", LIST_TYPE, Pred.len_eq("v", len(value.elts)))
        if isinstance(value, ast.Tuple):
            return RefType("v", TUPLE_TYPE, Pred.len_eq("v", len(value.elts)))
        if isinstance(value, ast.Dict):
            return RefType("v", DICT_TYPE, Pred.len_eq("v", len(value.keys)))

        return None

    @staticmethod
    def _reftype_for_constant(node: ast.Constant) -> RefType:
        """Infer a refinement type from a literal constant."""
        v = node.value
        if isinstance(v, bool):
            return RefType.trivial(BOOL_TYPE)
        if isinstance(v, int):
            return RefType("v", INT_TYPE, Pred.var_eq("v", v))
        if isinstance(v, float):
            return RefType.trivial(FLOAT_TYPE)
        if isinstance(v, str):
            return RefType("v", STR_TYPE, Pred.len_eq("v", len(v)))
        if v is None:
            return RefType.trivial(NONE_TYPE)
        return RefType.trivial(ANY_TYPE)

    # -----------------------------------------------------------------------
    # Context-manager type inference
    # -----------------------------------------------------------------------

    @staticmethod
    def _infer_context_type(ctx: ast.expr) -> Optional[RefType]:
        """Infer the type of a with-statement context expression.

        Recognizes ``open(...)`` as returning a file-like object.
        """
        if _is_builtin_call(ctx, "open"):
            return RefType("fh", ANY_TYPE, Pred.is_not_none("fh"))
        return None
