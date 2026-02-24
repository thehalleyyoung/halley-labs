"""
Translate Python guard patterns (isinstance, is None, comparisons, etc.)
to heap-aware refinement predicates.

The module bridges AST-level guards to the semantic refinement layer so that
the type-narrowing engine can track what is known on the true/false branches
of every conditional.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.heap.heap_model import HeapAddress, AbstractValue, AbstractHeap
from src.refinement.python_refinements import (
    HeapPredicate,
    HeapPredKind,
    PyRefinementType,
    PyType,
    IntPyType,
    StrPyType,
    BoolPyType,
    FloatPyType,
    NoneType as NoneRefType,
    ClassType,
    ProtocolType,
    PyUnionType,
    ListPyType,
    DictPyType,
    FunctionPyType,
    NarrowedType,
    OptionalType,
    AnyType,
    NeverType,
    TypeNarrower,
)


# ===================================================================
# Guard taxonomy
# ===================================================================

class GuardKind(Enum):
    """Classification of guard expressions recognised by the analyser."""

    ISINSTANCE = auto()
    HASATTR = auto()
    NONE_CHECK = auto()
    NOT_NONE_CHECK = auto()
    TRUTHINESS = auto()
    FALSINESS = auto()
    COMPARISON = auto()
    IN_CHECK = auto()
    NOT_IN_CHECK = auto()
    CALLABLE_CHECK = auto()
    TYPE_CHECK = auto()
    DICT_KEY_CHECK = auto()
    PATTERN_MATCH = auto()
    TRY_EXCEPT = auto()
    COMPOUND_AND = auto()
    COMPOUND_OR = auto()
    COMPOUND_NOT = auto()
    TYPE_GUARD = auto()
    TYPE_IS = auto()
    ASSERT = auto()


# ===================================================================
# Guard metadata
# ===================================================================

@dataclass
class GuardInfo:
    """Parsed information about a single guard expression."""

    kind: GuardKind
    variable: str
    path: Tuple[str, ...] = ()
    args: Dict[str, Any] = field(default_factory=dict)
    source_node: Optional[ast.AST] = None
    negated: bool = False

    # ----- helpers -----

    def full_path(self) -> str:
        """Return the dotted path ``variable.a.b`` as a single string."""
        if self.path:
            return self.variable + "." + ".".join(self.path)
        return self.variable

    def with_negation(self, neg: bool) -> "GuardInfo":
        """Return a copy with the negation flag set to *neg*."""
        return GuardInfo(
            kind=self.kind,
            variable=self.variable,
            path=self.path,
            args=dict(self.args),
            source_node=self.source_node,
            negated=neg,
        )


# ===================================================================
# Helpers – AST → name / type resolution
# ===================================================================

_BUILTIN_TYPE_MAP: Dict[str, PyType] = {
    "int": IntPyType(),
    "str": StrPyType(),
    "float": FloatPyType(),
    "bool": BoolPyType(),
    "list": ListPyType(AnyType()),
    "dict": DictPyType(AnyType(), AnyType()),
    "type": ClassType(HeapAddress("builtin:type")),
    "object": ClassType(HeapAddress("builtin:object")),
    "bytes": ClassType(HeapAddress("builtin:bytes")),
    "bytearray": ClassType(HeapAddress("builtin:bytearray")),
    "tuple": ClassType(HeapAddress("builtin:tuple")),
    "set": ClassType(HeapAddress("builtin:set")),
    "frozenset": ClassType(HeapAddress("builtin:frozenset")),
    "complex": ClassType(HeapAddress("builtin:complex")),
    "memoryview": ClassType(HeapAddress("builtin:memoryview")),
    "range": ClassType(HeapAddress("builtin:range")),
}


def _resolve_type_name(node: ast.expr) -> Optional[PyType]:
    """Best-effort type resolution from an AST node used in isinstance()."""
    if isinstance(node, ast.Name):
        return _BUILTIN_TYPE_MAP.get(node.id, ClassType(HeapAddress(f"class:{node.id}")))
    if isinstance(node, ast.Attribute):
        parts: List[str] = []
        cur: ast.expr = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        return ClassType(HeapAddress(f"class:{'.'.join(parts)}"))
    return None


def _resolve_type_tuple(node: ast.expr) -> List[PyType]:
    """Resolve a tuple of types from an isinstance second argument."""
    if isinstance(node, ast.Tuple):
        result: List[PyType] = []
        for elt in node.elts:
            resolved = _resolve_type_name(elt)
            if resolved is not None:
                result.append(resolved)
        return result
    single = _resolve_type_name(node)
    return [single] if single is not None else []


def _extract_name_and_path(node: ast.expr) -> Tuple[str, Tuple[str, ...]]:
    """
    Extract the root variable name and the attribute-access path from an
    expression such as ``x``, ``x.a``, or ``x.a.b``.
    """
    path_parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        path_parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        path_parts.reverse()
        return cur.id, tuple(path_parts)
    return "<unknown>", ()


def _make_pred(kind: HeapPredKind, variable: str, path: Tuple[str, ...],
               **kwargs: Any) -> HeapPredicate:
    """Convenience wrapper to build a :class:`HeapPredicate`."""
    return HeapPredicate(kind=kind, variable=variable, path=path, **kwargs)


def _negate_pred(pred: HeapPredicate) -> HeapPredicate:
    """Return the logical negation of *pred*."""
    return HeapPredicate(
        kind=pred.kind,
        variable=pred.variable,
        path=pred.path,
        refinement_type=pred.refinement_type,
        negated=not pred.negated,
    )


def _and_preds(left: HeapPredicate, right: HeapPredicate) -> HeapPredicate:
    """Logical conjunction of two predicates."""
    return HeapPredicate(
        kind=HeapPredKind.CONJUNCTION,
        variable=left.variable,
        path=left.path,
        children=[left, right],
    )


def _or_preds(left: HeapPredicate, right: HeapPredicate) -> HeapPredicate:
    """Logical disjunction of two predicates."""
    return HeapPredicate(
        kind=HeapPredKind.DISJUNCTION,
        variable=left.variable,
        path=left.path,
        children=[left, right],
    )


def _true_pred() -> HeapPredicate:
    """A trivially true predicate."""
    return HeapPredicate(kind=HeapPredKind.TRUE, variable="", path=())


def _false_pred() -> HeapPredicate:
    """A trivially false predicate."""
    return HeapPredicate(kind=HeapPredKind.FALSE, variable="", path=())


def _isinstance_pred(variable: str, path: Tuple[str, ...],
                     types: List[PyType]) -> HeapPredicate:
    """Predicate asserting *variable* is one of *types*."""
    if len(types) == 1:
        rtype = PyRefinementType(base=types[0])
    else:
        rtype = PyRefinementType(base=PyUnionType(types))
    return HeapPredicate(
        kind=HeapPredKind.TYPE_TEST,
        variable=variable,
        path=path,
        refinement_type=rtype,
    )


def _comparison_op_str(op: ast.cmpop) -> str:
    """Map an AST comparison operator to a human-readable string."""
    _MAP = {
        ast.Eq: "==", ast.NotEq: "!=",
        ast.Lt: "<", ast.LtE: "<=",
        ast.Gt: ">", ast.GtE: ">=",
        ast.Is: "is", ast.IsNot: "is not",
        ast.In: "in", ast.NotIn: "not in",
    }
    return _MAP.get(type(op), "??")


def _invert_cmp(op_str: str) -> str:
    """Return the inverse comparison operator."""
    _INV = {
        "==": "!=", "!=": "==",
        "<": ">=", ">=": "<",
        ">": "<=", "<=": ">",
        "is": "is not", "is not": "is",
        "in": "not in", "not in": "in",
    }
    return _INV.get(op_str, op_str)


# ===================================================================
# PythonGuardInterpreter
# ===================================================================

class PythonGuardInterpreter:
    """
    Translate an AST guard expression into a pair
    ``(true_predicate, false_predicate)`` describing what is known on
    each branch.
    """

    def __init__(
        self,
        type_env: Optional[Dict[str, PyRefinementType]] = None,
        narrower: Optional[TypeNarrower] = None,
    ) -> None:
        self.type_env: Dict[str, PyRefinementType] = type_env or {}
        self.narrower: TypeNarrower = narrower or TypeNarrower()

    # ----- dispatch entry point ------------------------------------

    def interpret(self, node: ast.expr) -> Tuple[HeapPredicate, HeapPredicate]:
        """Return ``(true_pred, false_pred)`` for the given guard *node*."""
        if isinstance(node, ast.Call):
            return self._interpret_call(node)
        if isinstance(node, ast.Compare):
            return self._interpret_compare(node)
        if isinstance(node, ast.BoolOp):
            return self.interpret_boolean_op(node)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self.interpret_unary_not(node)
        # Fall through to truthiness for bare names / expressions.
        return self.interpret_truthiness(node)

    # ----- call-based guards ---------------------------------------

    def _interpret_call(
        self, node: ast.Call
    ) -> Tuple[HeapPredicate, HeapPredicate]:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id == "isinstance" and len(node.args) >= 2:
                return self.interpret_isinstance(node)
            if func.id == "hasattr" and len(node.args) >= 2:
                return self.interpret_hasattr(node)
            if func.id == "callable" and len(node.args) >= 1:
                return self.interpret_callable_check(node)
            if func.id == "issubclass" and len(node.args) >= 2:
                return self._interpret_issubclass(node)
        # Possibly a TypeGuard / TypeIs function.
        return self.interpret_truthiness(node)

    # ----- isinstance ----------------------------------------------

    def interpret_isinstance(
        self, node: ast.Call
    ) -> Tuple[HeapPredicate, HeapPredicate]:
        """``isinstance(x, T)`` or ``isinstance(x, (T1, T2, ...))``."""
        subject = node.args[0]
        type_arg = node.args[1]
        var, path = _extract_name_and_path(subject)
        types = _resolve_type_tuple(type_arg)
        if not types:
            return _true_pred(), _false_pred()

        true_pred = _isinstance_pred(var, path, types)
        false_pred = _negate_pred(true_pred)
        return true_pred, false_pred

    # ----- issubclass (similar to isinstance) ----------------------

    def _interpret_issubclass(
        self, node: ast.Call
    ) -> Tuple[HeapPredicate, HeapPredicate]:
        """``issubclass(cls, T)``."""
        subject = node.args[0]
        type_arg = node.args[1]
        var, path = _extract_name_and_path(subject)
        types = _resolve_type_tuple(type_arg)
        if not types:
            return _true_pred(), _false_pred()
        true_pred = _isinstance_pred(var, path, types)
        false_pred = _negate_pred(true_pred)
        return true_pred, false_pred

    # ----- hasattr -------------------------------------------------

    def interpret_hasattr(
        self, node: ast.Call
    ) -> Tuple[HeapPredicate, HeapPredicate]:
        """``hasattr(x, 'foo')`` → x has attribute foo."""
        subject = node.args[0]
        attr_node = node.args[1]
        var, path = _extract_name_and_path(subject)

        attr_name: Optional[str] = None
        if isinstance(attr_node, ast.Constant) and isinstance(attr_node.value, str):
            attr_name = attr_node.value

        if attr_name is None:
            return _true_pred(), _false_pred()

        true_pred = HeapPredicate(
            kind=HeapPredKind.HAS_ATTR,
            variable=var,
            path=path,
            attr_name=attr_name,
        )
        false_pred = _negate_pred(true_pred)
        return true_pred, false_pred

    # ----- None checks ---------------------------------------------

    def interpret_none_check(
        self, node: ast.Compare
    ) -> Tuple[HeapPredicate, HeapPredicate]:
        """
        ``x is None`` / ``x is not None``.
        Returns (true_pred, false_pred) where true_pred describes the
        constraint when the comparison is True.
        """
        left = node.left
        var, path = _extract_name_and_path(left)
        op = node.ops[0]

        none_pred = HeapPredicate(
            kind=HeapPredKind.TYPE_TEST,
            variable=var,
            path=path,
            refinement_type=PyRefinementType(base=NoneRefType()),
        )
        not_none_pred = _negate_pred(none_pred)

        if isinstance(op, ast.Is):
            return none_pred, not_none_pred
        else:  # ast.IsNot
            return not_none_pred, none_pred

    # ----- truthiness / falsiness ----------------------------------

    def interpret_truthiness(
        self, node: ast.expr
    ) -> Tuple[HeapPredicate, HeapPredicate]:
        """
        ``if x:`` – the semantics depend on the inferred type of *x*.

        - str/bytes:  true when ``len(x) > 0``
        - int/float:  true when ``x != 0``
        - list/dict/set: true when ``len(x) > 0``
        - None: always false
        - bool: true when ``x is True``
        - object: depends on ``__bool__`` or ``__len__``
        """
        var, path = _extract_name_and_path(node)
        current_type = self.type_env.get(var)

        # Build a generic truthiness predicate.
        truthy_pred = HeapPredicate(
            kind=HeapPredKind.TRUTHY,
            variable=var,
            path=path,
        )
        falsy_pred = _negate_pred(truthy_pred)

        if current_type is None:
            return truthy_pred, falsy_pred

        base = current_type.base if current_type else None

        # Special-case Optional[T]: truthy branch narrows away None.
        if isinstance(base, OptionalType):
            not_none_pred = HeapPredicate(
                kind=HeapPredKind.TYPE_TEST,
                variable=var,
                path=path,
                refinement_type=PyRefinementType(base=base),
                negated=False,
            )
            none_pred = HeapPredicate(
                kind=HeapPredKind.TYPE_TEST,
                variable=var,
                path=path,
                refinement_type=PyRefinementType(base=NoneRefType()),
            )
            return not_none_pred, none_pred

        # None: always false.
        if isinstance(base, NoneRefType):
            return _false_pred(), _true_pred()

        # Bool: value-level.
        if isinstance(base, BoolPyType):
            return truthy_pred, falsy_pred

        # int/float: x != 0 / x == 0
        if isinstance(base, (IntPyType, FloatPyType)):
            nonzero = HeapPredicate(
                kind=HeapPredKind.VALUE_CONSTRAINT,
                variable=var,
                path=path,
                op="!=",
                value=0,
            )
            zero = HeapPredicate(
                kind=HeapPredKind.VALUE_CONSTRAINT,
                variable=var,
                path=path,
                op="==",
                value=0,
            )
            return nonzero, zero

        # str: len(x) > 0 / len(x) == 0
        if isinstance(base, StrPyType):
            nonempty = HeapPredicate(
                kind=HeapPredKind.LENGTH_CONSTRAINT,
                variable=var,
                path=path,
                op=">",
                value=0,
            )
            empty = HeapPredicate(
                kind=HeapPredKind.LENGTH_CONSTRAINT,
                variable=var,
                path=path,
                op="==",
                value=0,
            )
            return nonempty, empty

        # list / dict / set: length > 0
        if isinstance(base, (ListPyType, DictPyType)):
            nonempty = HeapPredicate(
                kind=HeapPredKind.LENGTH_CONSTRAINT,
                variable=var,
                path=path,
                op=">",
                value=0,
            )
            empty = HeapPredicate(
                kind=HeapPredKind.LENGTH_CONSTRAINT,
                variable=var,
                path=path,
                op="==",
                value=0,
            )
            return nonempty, empty

        return truthy_pred, falsy_pred

    # ----- comparisons ---------------------------------------------

    def _interpret_compare(
        self, node: ast.Compare
    ) -> Tuple[HeapPredicate, HeapPredicate]:
        """Dispatch Compare nodes to specific handlers."""
        if len(node.ops) == 1:
            op = node.ops[0]
            comparator = node.comparators[0]
            # ``x is None`` / ``x is not None``
            if isinstance(op, (ast.Is, ast.IsNot)):
                if isinstance(comparator, ast.Constant) and comparator.value is None:
                    return self.interpret_none_check(node)
                if isinstance(node.left, ast.Constant) and node.left.value is None:
                    # ``None is x`` – swap
                    swapped = ast.Compare(
                        left=comparator,
                        ops=[op],
                        comparators=[node.left],
                    )
                    return self.interpret_none_check(swapped)
            # ``'key' in d`` – dict key check
            if isinstance(op, (ast.In, ast.NotIn)):
                return self.interpret_dict_key_check(node)
            return self.interpret_comparison(node)
        # Chained comparisons: ``0 < x < 10``
        return self._interpret_chained_comparison(node)

    def interpret_comparison(
        self, node: ast.Compare
    ) -> Tuple[HeapPredicate, HeapPredicate]:
        """
        Handle single comparisons: ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``.
        """
        left = node.left
        op = node.ops[0]
        right = node.comparators[0]

        var, path = _extract_name_and_path(left)
        op_str = _comparison_op_str(op)
        inv_op_str = _invert_cmp(op_str)

        # Try to extract constant value.
        const_value: Any = None
        if isinstance(right, ast.Constant):
            const_value = right.value
        elif isinstance(right, ast.UnaryOp) and isinstance(right.op, ast.USub):
            if isinstance(right.operand, ast.Constant):
                const_value = -right.operand.value

        if const_value is not None:
            true_pred = HeapPredicate(
                kind=HeapPredKind.VALUE_CONSTRAINT,
                variable=var,
                path=path,
                op=op_str,
                value=const_value,
            )
            false_pred = HeapPredicate(
                kind=HeapPredKind.VALUE_CONSTRAINT,
                variable=var,
                path=path,
                op=inv_op_str,
                value=const_value,
            )
            return true_pred, false_pred

        # Variable-to-variable comparison: limited predicate.
        rvar, rpath = _extract_name_and_path(right)
        true_pred = HeapPredicate(
            kind=HeapPredKind.RELATIONAL,
            variable=var,
            path=path,
            op=op_str,
            other_variable=rvar,
            other_path=rpath,
        )
        false_pred = HeapPredicate(
            kind=HeapPredKind.RELATIONAL,
            variable=var,
            path=path,
            op=inv_op_str,
            other_variable=rvar,
            other_path=rpath,
        )
        return true_pred, false_pred

    def _interpret_chained_comparison(
        self, node: ast.Compare
    ) -> Tuple[HeapPredicate, HeapPredicate]:
        """
        Chained comparisons like ``0 < x < 10`` are conjunctions of
        pairwise comparisons.  The false branch is the disjunction of
        each pairwise negation.
        """
        true_parts: List[HeapPredicate] = []
        false_parts: List[HeapPredicate] = []

        left = node.left
        for op, comp in zip(node.ops, node.comparators):
            sub = ast.Compare(left=left, ops=[op], comparators=[comp])
            tp, fp = self.interpret_comparison(sub)
            true_parts.append(tp)
            false_parts.append(fp)
            left = comp

        combined_true = true_parts[0]
        for tp in true_parts[1:]:
            combined_true = _and_preds(combined_true, tp)

        combined_false = false_parts[0]
        for fp in false_parts[1:]:
            combined_false = _or_preds(combined_false, fp)

        return combined_true, combined_false

    # ----- in / not in ---------------------------------------------

    def interpret_dict_key_check(
        self, node: ast.Compare
    ) -> Tuple[HeapPredicate, HeapPredicate]:
        """``'key' in d`` / ``'key' not in d``."""
        op = node.ops[0]
        key_node = node.left
        container_node = node.comparators[0]

        cvar, cpath = _extract_name_and_path(container_node)

        key_value: Any = None
        if isinstance(key_node, ast.Constant):
            key_value = key_node.value

        # When the key is a variable (``x in collection``).
        key_var: Optional[str] = None
        if isinstance(key_node, ast.Name):
            key_var = key_node.id

        has_pred = HeapPredicate(
            kind=HeapPredKind.CONTAINS,
            variable=cvar,
            path=cpath,
            value=key_value,
            key_variable=key_var,
        )
        not_has_pred = _negate_pred(has_pred)

        if isinstance(op, ast.In):
            return has_pred, not_has_pred
        else:
            return not_has_pred, has_pred

    # ----- callable ------------------------------------------------

    def interpret_callable_check(
        self, node: ast.Call
    ) -> Tuple[HeapPredicate, HeapPredicate]:
        """``callable(x)`` → x has ``__call__``."""
        subject = node.args[0]
        var, path = _extract_name_and_path(subject)
        callable_pred = HeapPredicate(
            kind=HeapPredKind.HAS_ATTR,
            variable=var,
            path=path,
            attr_name="__call__",
        )
        not_callable = _negate_pred(callable_pred)
        return callable_pred, not_callable

    # ----- TypeGuard / TypeIs --------------------------------------

    def interpret_type_guard(
        self, node: ast.Call, annotation: Any
    ) -> HeapPredicate:
        """
        A function returning ``TypeGuard[T]`` or ``TypeIs[T]``.

        *annotation* is the resolved T from the return type.
        """
        if not node.args:
            return _true_pred()
        subject = node.args[0]
        var, path = _extract_name_and_path(subject)
        guarded_type = _resolve_type_name(annotation) if isinstance(
            annotation, ast.expr) else None
        if guarded_type is None and isinstance(annotation, str):
            guarded_type = _BUILTIN_TYPE_MAP.get(annotation, ClassType(HeapAddress(f"class:{annotation}")))
        if guarded_type is None:
            return _true_pred()
        return HeapPredicate(
            kind=HeapPredKind.TYPE_TEST,
            variable=var,
            path=path,
            refinement_type=PyRefinementType(base=guarded_type),
        )

    # ----- assert --------------------------------------------------

    def interpret_assert(self, node: ast.Assert) -> HeapPredicate:
        """
        ``assert isinstance(x, T)`` → treat the test as a one-sided guard
        (only the true branch survives).
        """
        true_pred, _ = self.interpret(node.test)
        return true_pred

    # ----- boolean combinators -------------------------------------

    def interpret_boolean_op(
        self, node: ast.BoolOp
    ) -> Tuple[HeapPredicate, HeapPredicate]:
        """
        ``x and y`` → true: both true, false: either false.
        ``x or y``  → true: either true, false: both false.
        """
        parts = [self.interpret(v) for v in node.values]

        if isinstance(node.op, ast.And):
            combined_true = parts[0][0]
            for tp, _ in parts[1:]:
                combined_true = _and_preds(combined_true, tp)
            combined_false = parts[0][1]
            for _, fp in parts[1:]:
                combined_false = _or_preds(combined_false, fp)
            return combined_true, combined_false
        else:  # ast.Or
            combined_true = parts[0][0]
            for tp, _ in parts[1:]:
                combined_true = _or_preds(combined_true, tp)
            combined_false = parts[0][1]
            for _, fp in parts[1:]:
                combined_false = _and_preds(combined_false, fp)
            return combined_true, combined_false

    def interpret_unary_not(
        self, node: ast.UnaryOp
    ) -> Tuple[HeapPredicate, HeapPredicate]:
        """``not x`` swaps the true/false predicates."""
        true_pred, false_pred = self.interpret(node.operand)
        return false_pred, true_pred

    # ----- pattern matching (Python 3.10+) -------------------------

    def interpret_pattern_match(
        self, node: ast.Match
    ) -> List[Tuple[ast.pattern, HeapPredicate]]:
        """
        Process a ``match`` statement. Returns a list of
        ``(pattern, predicate)`` pairs – one per case arm.
        """
        subject = node.subject
        var, path = _extract_name_and_path(subject)
        results: List[Tuple[ast.pattern, HeapPredicate]] = []

        for case in node.cases:
            pat = case.pattern
            pred = self._pattern_to_pred(pat, var, path)
            # If there is a guard on the case, conjoin it.
            if case.guard is not None:
                guard_true, _ = self.interpret(case.guard)
                pred = _and_preds(pred, guard_true)
            results.append((pat, pred))

        return results

    def _pattern_to_pred(
        self, pattern: ast.pattern, var: str, path: Tuple[str, ...]
    ) -> HeapPredicate:
        """Convert a single pattern node into a predicate."""
        if isinstance(pattern, ast.MatchValue):
            return self._match_value_pred(pattern, var, path)
        if isinstance(pattern, ast.MatchClass):
            return self._match_class_pred(pattern, var, path)
        if isinstance(pattern, ast.MatchMapping):
            return self._match_mapping_pred(pattern, var, path)
        if isinstance(pattern, ast.MatchSequence):
            return self._match_sequence_pred(pattern, var, path)
        if isinstance(pattern, ast.MatchStar):
            return _true_pred()  # captures rest – no refinement
        if isinstance(pattern, ast.MatchOr):
            return self._match_or_pred(pattern, var, path)
        if isinstance(pattern, ast.MatchAs):
            if pattern.pattern is not None:
                return self._pattern_to_pred(pattern.pattern, var, path)
            return _true_pred()  # wildcard ``_``
        if isinstance(pattern, ast.MatchSingleton):
            if pattern.value is None:
                return HeapPredicate(
                    kind=HeapPredKind.TYPE_TEST,
                    variable=var,
                    path=path,
                    refinement_type=PyRefinementType(base=NoneRefType()),
                )
            if pattern.value is True:
                return HeapPredicate(
                    kind=HeapPredKind.VALUE_CONSTRAINT,
                    variable=var, path=path, op="is", value=True,
                )
            if pattern.value is False:
                return HeapPredicate(
                    kind=HeapPredKind.VALUE_CONSTRAINT,
                    variable=var, path=path, op="is", value=False,
                )
        return _true_pred()

    def _match_value_pred(
        self, pattern: ast.MatchValue, var: str, path: Tuple[str, ...]
    ) -> HeapPredicate:
        const_val: Any = None
        if isinstance(pattern.value, ast.Constant):
            const_val = pattern.value.value
        return HeapPredicate(
            kind=HeapPredKind.VALUE_CONSTRAINT,
            variable=var, path=path, op="==", value=const_val,
        )

    def _match_class_pred(
        self, pattern: ast.MatchClass, var: str, path: Tuple[str, ...]
    ) -> HeapPredicate:
        cls_type = _resolve_type_name(pattern.cls)
        if cls_type is None:
            return _true_pred()
        type_pred = HeapPredicate(
            kind=HeapPredKind.TYPE_TEST,
            variable=var,
            path=path,
            refinement_type=PyRefinementType(base=cls_type),
        )
        # keyword sub-patterns refine attributes.
        attr_preds: List[HeapPredicate] = [type_pred]
        for kw_attr, kw_pat in zip(pattern.kwd_attrs, pattern.kwd_patterns):
            sub_path = path + (kw_attr,)
            attr_preds.append(self._pattern_to_pred(kw_pat, var, sub_path))
        # positional patterns refine by index.
        for idx, pos_pat in enumerate(pattern.patterns):
            idx_path = path + (f"__pos_{idx}",)
            attr_preds.append(self._pattern_to_pred(pos_pat, var, idx_path))

        result = attr_preds[0]
        for p in attr_preds[1:]:
            result = _and_preds(result, p)
        return result

    def _match_mapping_pred(
        self, pattern: ast.MatchMapping, var: str, path: Tuple[str, ...]
    ) -> HeapPredicate:
        preds: List[HeapPredicate] = []
        for key_node, val_pat in zip(pattern.keys, pattern.patterns):
            key_val: Any = None
            if isinstance(key_node, ast.Constant):
                key_val = key_node.value
            has_key = HeapPredicate(
                kind=HeapPredKind.CONTAINS,
                variable=var, path=path, value=key_val,
            )
            preds.append(has_key)
            if key_val is not None:
                sub_path = path + (f"[{key_val!r}]",)
                preds.append(self._pattern_to_pred(val_pat, var, sub_path))

        if not preds:
            return _true_pred()
        result = preds[0]
        for p in preds[1:]:
            result = _and_preds(result, p)
        return result

    def _match_sequence_pred(
        self, pattern: ast.MatchSequence, var: str, path: Tuple[str, ...]
    ) -> HeapPredicate:
        has_star = any(isinstance(p, ast.MatchStar) for p in pattern.patterns)
        fixed_count = sum(1 for p in pattern.patterns if not isinstance(p, ast.MatchStar))

        if has_star:
            len_pred = HeapPredicate(
                kind=HeapPredKind.LENGTH_CONSTRAINT,
                variable=var, path=path, op=">=", value=fixed_count,
            )
        else:
            len_pred = HeapPredicate(
                kind=HeapPredKind.LENGTH_CONSTRAINT,
                variable=var, path=path, op="==", value=fixed_count,
            )

        element_preds: List[HeapPredicate] = [len_pred]
        idx = 0
        for p in pattern.patterns:
            if isinstance(p, ast.MatchStar):
                continue
            idx_path = path + (f"[{idx}]",)
            element_preds.append(self._pattern_to_pred(p, var, idx_path))
            idx += 1

        result = element_preds[0]
        for ep in element_preds[1:]:
            result = _and_preds(result, ep)
        return result

    def _match_or_pred(
        self, pattern: ast.MatchOr, var: str, path: Tuple[str, ...]
    ) -> HeapPredicate:
        sub_preds = [self._pattern_to_pred(p, var, path)
                     for p in pattern.patterns]
        if not sub_preds:
            return _true_pred()
        result = sub_preds[0]
        for sp in sub_preds[1:]:
            result = _or_preds(result, sp)
        return result

    # ----- try/except ----------------------------------------------

    def interpret_try_except(
        self, handler: ast.ExceptHandler
    ) -> HeapPredicate:
        """
        In an ``except`` block the caught exception has the type named in
        the handler.  ``except ValueError as e:`` → ``e: ValueError``.
        """
        if handler.type is None:
            # Bare ``except:`` catches everything – BaseException.
            exc_type: PyType = ClassType(HeapAddress("builtin:BaseException"))
        else:
            resolved = _resolve_type_name(handler.type)
            exc_type = resolved if resolved is not None else ClassType(HeapAddress("builtin:BaseException"))

        var = handler.name if handler.name else "<exception>"
        return HeapPredicate(
            kind=HeapPredKind.TYPE_TEST,
            variable=var,
            path=(),
            refinement_type=PyRefinementType(base=exc_type),
        )


# ===================================================================
# GuardExtractor – pull guard info out of AST nodes
# ===================================================================

class GuardExtractor:
    """Extract :class:`GuardInfo` descriptors from control-flow AST nodes."""

    # ----- public API -----------------------------------------------

    def extract_from_if(self, node: ast.If) -> GuardInfo:
        """Extract the guard from an ``if`` statement."""
        return self._build_guard_info(node.test)

    def extract_from_while(self, node: ast.While) -> GuardInfo:
        """Extract the guard from a ``while`` loop."""
        return self._build_guard_info(node.test)

    def extract_from_assert(self, node: ast.Assert) -> GuardInfo:
        """Extract the guard from an ``assert`` statement."""
        info = self._build_guard_info(node.test)
        info.kind = GuardKind.ASSERT
        return info

    def extract_from_comprehension_if(
        self, node: ast.comprehension
    ) -> List[GuardInfo]:
        """Extract guards from comprehension ``if`` clauses."""
        return [self._build_guard_info(cond) for cond in node.ifs]

    def extract_from_ternary(self, node: ast.IfExp) -> GuardInfo:
        """Extract the guard from a ternary ``x if cond else y``."""
        return self._build_guard_info(node.test)

    # ----- classification -------------------------------------------

    def classify_guard(self, node: ast.expr) -> GuardKind:
        """Determine what kind of guard *node* represents."""
        if isinstance(node, ast.Call):
            return self._classify_call(node)
        if isinstance(node, ast.Compare):
            return self._classify_compare(node)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return GuardKind.COMPOUND_AND
            return GuardKind.COMPOUND_OR
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return GuardKind.COMPOUND_NOT
        return GuardKind.TRUTHINESS

    def extract_variable(
        self, node: ast.expr
    ) -> Tuple[str, Tuple[str, ...]]:
        """Extract the root variable name and attribute path."""
        return _extract_name_and_path(node)

    # ----- internals ------------------------------------------------

    def _classify_call(self, node: ast.Call) -> GuardKind:
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            if name == "isinstance":
                return GuardKind.ISINSTANCE
            if name == "hasattr":
                return GuardKind.HASATTR
            if name == "callable":
                return GuardKind.CALLABLE_CHECK
            if name == "issubclass":
                return GuardKind.TYPE_CHECK
        return GuardKind.TRUTHINESS

    def _classify_compare(self, node: ast.Compare) -> GuardKind:
        if len(node.ops) != 1:
            return GuardKind.COMPARISON

        op = node.ops[0]
        comparator = node.comparators[0]

        if isinstance(op, ast.Is):
            if isinstance(comparator, ast.Constant) and comparator.value is None:
                return GuardKind.NONE_CHECK
            return GuardKind.COMPARISON
        if isinstance(op, ast.IsNot):
            if isinstance(comparator, ast.Constant) and comparator.value is None:
                return GuardKind.NOT_NONE_CHECK
            return GuardKind.COMPARISON
        if isinstance(op, ast.In):
            return GuardKind.IN_CHECK
        if isinstance(op, ast.NotIn):
            return GuardKind.NOT_IN_CHECK
        return GuardKind.COMPARISON

    def _build_guard_info(self, node: ast.expr) -> GuardInfo:
        kind = self.classify_guard(node)
        var, path = self._extract_primary_variable(node, kind)
        args = self._extract_args(node, kind)
        return GuardInfo(
            kind=kind,
            variable=var,
            path=path,
            args=args,
            source_node=node,
            negated=False,
        )

    def _extract_primary_variable(
        self, node: ast.expr, kind: GuardKind
    ) -> Tuple[str, Tuple[str, ...]]:
        """Best-effort extraction of the variable being guarded."""
        if isinstance(node, ast.Call) and node.args:
            return _extract_name_and_path(node.args[0])
        if isinstance(node, ast.Compare):
            return _extract_name_and_path(node.left)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self._extract_primary_variable(node.operand, kind)
        if isinstance(node, ast.BoolOp):
            return self._extract_primary_variable(node.values[0], kind)
        return _extract_name_and_path(node)

    def _extract_args(
        self, node: ast.expr, kind: GuardKind
    ) -> Dict[str, Any]:
        """Extract kind-specific arguments from the guard node."""
        args: Dict[str, Any] = {}

        if kind == GuardKind.ISINSTANCE and isinstance(node, ast.Call):
            if len(node.args) >= 2:
                types = _resolve_type_tuple(node.args[1])
                args["types"] = types

        elif kind == GuardKind.HASATTR and isinstance(node, ast.Call):
            if len(node.args) >= 2:
                attr_node = node.args[1]
                if isinstance(attr_node, ast.Constant) and isinstance(attr_node.value, str):
                    args["attr"] = attr_node.value

        elif kind in (GuardKind.NONE_CHECK, GuardKind.NOT_NONE_CHECK):
            args["none"] = True

        elif kind == GuardKind.COMPARISON and isinstance(node, ast.Compare):
            if node.ops:
                args["op"] = _comparison_op_str(node.ops[0])
            if node.comparators:
                cmp = node.comparators[0]
                if isinstance(cmp, ast.Constant):
                    args["value"] = cmp.value

        elif kind in (GuardKind.IN_CHECK, GuardKind.NOT_IN_CHECK):
            if isinstance(node, ast.Compare) and node.comparators:
                cvar, cpath = _extract_name_and_path(node.comparators[0])
                args["container"] = cvar
                args["container_path"] = cpath

        elif kind == GuardKind.CALLABLE_CHECK:
            pass  # no extra args needed

        elif kind == GuardKind.COMPOUND_NOT and isinstance(node, ast.UnaryOp):
            args["inner_kind"] = self.classify_guard(node.operand)

        elif kind in (GuardKind.COMPOUND_AND, GuardKind.COMPOUND_OR):
            if isinstance(node, ast.BoolOp):
                args["operand_count"] = len(node.values)

        return args


# ===================================================================
# RefinementPropagator – apply predicates to type environments
# ===================================================================

class RefinementPropagator:
    """Propagate refinement predicates through control-flow branches."""

    def __init__(
        self,
        narrower: Optional[TypeNarrower] = None,
    ) -> None:
        self.narrower = narrower or TypeNarrower()

    # ----- public API -----------------------------------------------

    def propagate_true_branch(
        self,
        guard: HeapPredicate,
        env: Dict[str, PyRefinementType],
    ) -> Dict[str, PyRefinementType]:
        """Apply *guard* to *env* as if the guard evaluated to ``True``."""
        return self.narrow_env(guard, dict(env))

    def propagate_false_branch(
        self,
        guard: HeapPredicate,
        env: Dict[str, PyRefinementType],
    ) -> Dict[str, PyRefinementType]:
        """Apply the negation of *guard* to *env*."""
        negated = _negate_pred(guard)
        return self.narrow_env(negated, dict(env))

    def propagate_assert(
        self,
        guard: HeapPredicate,
        env: Dict[str, PyRefinementType],
    ) -> Dict[str, PyRefinementType]:
        """Assert is equivalent to entering the true branch permanently."""
        return self.propagate_true_branch(guard, env)

    def join_branches(
        self,
        true_env: Dict[str, PyRefinementType],
        false_env: Dict[str, PyRefinementType],
    ) -> Dict[str, PyRefinementType]:
        """
        After an ``if/else``, join the two environments. Variables present
        in both branches get a union type; variables only in one branch
        keep their type from that branch.
        """
        all_vars: Set[str] = set(true_env) | set(false_env)
        joined: Dict[str, PyRefinementType] = {}

        for var in all_vars:
            t_type = true_env.get(var)
            f_type = false_env.get(var)

            if t_type is not None and f_type is not None:
                joined[var] = self._join_types(t_type, f_type)
            elif t_type is not None:
                joined[var] = t_type
            else:
                assert f_type is not None
                joined[var] = f_type

        return joined

    # ----- narrowing logic ------------------------------------------

    def narrow_env(
        self,
        pred: HeapPredicate,
        env: Dict[str, PyRefinementType],
    ) -> Dict[str, PyRefinementType]:
        """
        Apply *pred* to narrow all affected variables in *env*.
        """
        kind = pred.kind

        if kind == HeapPredKind.TRUE:
            return env

        if kind == HeapPredKind.FALSE:
            # Unreachable branch – mark all types as Never.
            return {v: PyRefinementType(base=NeverType()) for v in env}

        if kind == HeapPredKind.CONJUNCTION:
            result = dict(env)
            for child in (pred.children or []):
                result = self.narrow_env(child, result)
            return result

        if kind == HeapPredKind.DISJUNCTION:
            if not pred.children:
                return env
            branches = [self.narrow_env(c, dict(env)) for c in pred.children]
            result = branches[0]
            for branch in branches[1:]:
                result = self.join_branches(result, branch)
            return result

        var = pred.variable
        if not var or var == "<unknown>":
            return env

        if var not in env:
            # Variable not in env – might be a new binding.
            if kind == HeapPredKind.TYPE_TEST and pred.refinement_type and not pred.negated:
                env[var] = pred.refinement_type
            return env

        current = env[var]

        if kind == HeapPredKind.TYPE_TEST:
            env[var] = self._narrow_type_test(current, pred)
        elif kind == HeapPredKind.HAS_ATTR:
            env[var] = self._narrow_has_attr(current, pred)
        elif kind == HeapPredKind.TRUTHY:
            env[var] = self._narrow_truthy(current, pred)
        elif kind == HeapPredKind.VALUE_CONSTRAINT:
            env[var] = self._narrow_value(current, pred)
        elif kind == HeapPredKind.LENGTH_CONSTRAINT:
            env[var] = self._narrow_length(current, pred)
        elif kind == HeapPredKind.CONTAINS:
            env[var] = self._narrow_contains(current, pred)
        elif kind == HeapPredKind.RELATIONAL:
            env[var] = self._narrow_relational(current, pred)

        return env

    # ----- individual narrowing strategies --------------------------

    def _narrow_type_test(
        self, current: PyRefinementType, pred: HeapPredicate
    ) -> PyRefinementType:
        """Narrow via ``isinstance``-style type tests."""
        target = pred.refinement_type
        if target is None:
            return current

        if pred.negated:
            # Remove the target type from the current type.
            return self._subtract_type(current, target)
        else:
            # Intersect with the target type.
            return self._intersect_type(current, target)

    def _narrow_has_attr(
        self, current: PyRefinementType, pred: HeapPredicate
    ) -> PyRefinementType:
        """Narrow via ``hasattr`` – result is a NarrowedType with the attr."""
        attr = getattr(pred, "attr_name", None)
        if attr is None:
            return current
        if pred.negated:
            return current  # Can't narrow away an attribute easily.
        return PyRefinementType(
            base=NarrowedType(current.base, has_attrs=frozenset([attr]))
        )

    def _narrow_truthy(
        self, current: PyRefinementType, pred: HeapPredicate
    ) -> PyRefinementType:
        """Narrow via truthiness check."""
        base = current.base
        if pred.negated:
            # falsy branch
            if isinstance(base, OptionalType):
                return PyRefinementType(base=NoneRefType())
            return current
        else:
            # truthy branch – remove None from Optional.
            if isinstance(base, OptionalType):
                inner = getattr(base, "inner", None)
                if inner is not None:
                    return PyRefinementType(base=inner)
            if isinstance(base, PyUnionType):
                members = [m for m in base.types if not isinstance(m, NoneRefType)]
                if len(members) == 1:
                    return PyRefinementType(base=members[0])
                if members:
                    return PyRefinementType(base=PyUnionType(members))
            return current

    def _narrow_value(
        self, current: PyRefinementType, pred: HeapPredicate
    ) -> PyRefinementType:
        """Narrow via value constraints (e.g. ``x > 0``)."""
        return PyRefinementType(
            base=NarrowedType(current.base, value_constraint=(
                getattr(pred, "op", "=="),
                getattr(pred, "value", None),
            )),
        )

    def _narrow_length(
        self, current: PyRefinementType, pred: HeapPredicate
    ) -> PyRefinementType:
        """Narrow via length constraints."""
        return PyRefinementType(
            base=NarrowedType(current.base, length_constraint=(
                getattr(pred, "op", ">="),
                getattr(pred, "value", 0),
            )),
        )

    def _narrow_contains(
        self, current: PyRefinementType, pred: HeapPredicate
    ) -> PyRefinementType:
        """Narrow via ``in`` / ``not in`` checks."""
        return current  # containment doesn't change the container's type

    def _narrow_relational(
        self, current: PyRefinementType, pred: HeapPredicate
    ) -> PyRefinementType:
        """Narrow via relational constraints between variables."""
        return current  # relational constraints don't change the type

    # ----- type algebra helpers -------------------------------------

    def _intersect_type(
        self, current: PyRefinementType, target: PyRefinementType
    ) -> PyRefinementType:
        """
        Compute the intersection: what remains when we know *current*
        must also be *target*.

        For a union ``A | B | C`` intersected with ``B``, the result is ``B``.
        """
        cur_base = current.base
        tgt_base = target.base

        if isinstance(cur_base, AnyType):
            return target

        if isinstance(cur_base, PyUnionType):
            kept: List[PyType] = []
            for member in cur_base.types:
                if self._is_subtype(member, tgt_base):
                    kept.append(member)
            if not kept:
                # No overlap found – trust the isinstance.
                return target
            if len(kept) == 1:
                return PyRefinementType(base=kept[0])
            return PyRefinementType(base=PyUnionType(kept))

        if isinstance(cur_base, OptionalType):
            inner = getattr(cur_base, "inner", None)
            if inner is not None and self._is_subtype(inner, tgt_base):
                return PyRefinementType(base=inner)
            if isinstance(tgt_base, NoneRefType):
                return PyRefinementType(base=NoneRefType())
            return target

        return target

    def _subtract_type(
        self, current: PyRefinementType, target: PyRefinementType
    ) -> PyRefinementType:
        """
        Remove *target* from *current*.

        ``Optional[int]`` minus ``None`` → ``int``.
        ``int | str | None`` minus ``int`` → ``str | None``.
        """
        cur_base = current.base
        tgt_base = target.base

        if isinstance(cur_base, PyUnionType):
            remaining = [
                m for m in cur_base.types
                if not self._is_subtype(m, tgt_base)
            ]
            if not remaining:
                return PyRefinementType(base=NeverType())
            if len(remaining) == 1:
                return PyRefinementType(base=remaining[0])
            return PyRefinementType(base=PyUnionType(remaining))

        if isinstance(cur_base, OptionalType):
            if isinstance(tgt_base, NoneRefType):
                inner = getattr(cur_base, "inner", None)
                if inner is not None:
                    return PyRefinementType(base=inner)
            return current

        if self._types_equal(cur_base, tgt_base):
            return PyRefinementType(base=NeverType())

        return current

    def _is_subtype(self, sub: PyType, sup: PyType) -> bool:
        """Conservative subtype check based on nominal equality."""
        if isinstance(sup, AnyType):
            return True
        if self._types_equal(sub, sup):
            return True
        if isinstance(sup, ClassType) and isinstance(sub, ClassType):
            return sub.name == sup.name
        if isinstance(sup, PyUnionType):
            return any(self._is_subtype(sub, m) for m in sup.types)
        return False

    def _types_equal(self, a: PyType, b: PyType) -> bool:
        """Structural equality check for two PyType instances."""
        if type(a) is type(b):
            if isinstance(a, ClassType) and isinstance(b, ClassType):
                return a.name == b.name
            return True  # same leaf type class
        return False

    def _join_types(
        self, a: PyRefinementType, b: PyRefinementType
    ) -> PyRefinementType:
        """Join (least upper bound) of two refinement types."""
        if self._types_equal(a.base, b.base):
            return a  # identical – keep one

        # Collect unique types.
        types_a = (
            list(a.base.types) if isinstance(a.base, PyUnionType) else [a.base]
        )
        types_b = (
            list(b.base.types) if isinstance(b.base, PyUnionType) else [b.base]
        )
        seen: Set[str] = set()
        merged: List[PyType] = []
        for t in types_a + types_b:
            key = self._type_key(t)
            if key not in seen:
                seen.add(key)
                merged.append(t)

        if len(merged) == 1:
            return PyRefinementType(base=merged[0])
        return PyRefinementType(base=PyUnionType(merged))

    @staticmethod
    def _type_key(t: PyType) -> str:
        """Produce a hashable key for deduplication."""
        if isinstance(t, ClassType):
            return f"class:{t.name}"
        return type(t).__name__


# ===================================================================
# PatternDesugarer – desugar match/case into flat guard lists
# ===================================================================

class PatternDesugarer:
    """
    Desugar ``match``/``case`` statements into lists of flat predicates
    so the narrowing engine can handle them uniformly.
    """

    def __init__(self) -> None:
        self._interpreter = PythonGuardInterpreter()

    # ----- public API -----------------------------------------------

    def desugar_match(
        self, node: ast.Match
    ) -> List[Tuple[List[HeapPredicate], ast.AST]]:
        """
        Convert each ``case`` into ``(predicates, body)`` where
        *predicates* is the conjunction of conditions that must hold.
        """
        subject = node.subject
        var, path = _extract_name_and_path(subject)
        results: List[Tuple[List[HeapPredicate], ast.AST]] = []
        negated_so_far: List[HeapPredicate] = []

        for case in node.cases:
            case_preds = self.desugar_pattern(case.pattern, var)

            # Add guard if present.
            if case.guard is not None:
                guard_true, _ = self._interpreter.interpret(case.guard)
                case_preds.append(guard_true)

            # Each subsequent case implicitly negates all prior cases.
            effective_preds = list(negated_so_far) + case_preds

            body: ast.AST = case  # the case node contains the body
            results.append((effective_preds, body))

            # Accumulate negation of this pattern for subsequent cases.
            if case_preds:
                combined = case_preds[0]
                for p in case_preds[1:]:
                    combined = _and_preds(combined, p)
                negated_so_far.append(_negate_pred(combined))

        return results

    def desugar_pattern(
        self, pattern: ast.pattern, var: str
    ) -> List[HeapPredicate]:
        """Convert a single pattern into a list of predicates."""
        if isinstance(pattern, ast.MatchValue):
            return [self.desugar_value_pattern(pattern, var)]
        if isinstance(pattern, ast.MatchSingleton):
            return [self._desugar_singleton(pattern, var)]
        if isinstance(pattern, ast.MatchClass):
            return self.desugar_class_pattern(pattern, var)
        if isinstance(pattern, ast.MatchMapping):
            return self.desugar_mapping_pattern(pattern, var)
        if isinstance(pattern, ast.MatchSequence):
            return self.desugar_sequence_pattern(pattern, var)
        if isinstance(pattern, ast.MatchOr):
            return [self.desugar_or_pattern(pattern, var)]
        if isinstance(pattern, ast.MatchAs):
            if pattern.pattern is not None:
                return self.desugar_pattern(pattern.pattern, var)
            return []  # wildcard
        if isinstance(pattern, ast.MatchStar):
            return []  # captures remaining – no predicate
        return []

    # ----- class pattern -------------------------------------------

    def desugar_class_pattern(
        self, pattern: ast.MatchClass, var: str
    ) -> List[HeapPredicate]:
        """
        ``case MyClass(a=p1, b=p2):`` →
        [isinstance(var, MyClass), desugar(p1, var.a), desugar(p2, var.b)]
        """
        preds: List[HeapPredicate] = []

        cls_type = _resolve_type_name(pattern.cls)
        if cls_type is not None:
            preds.append(HeapPredicate(
                kind=HeapPredKind.TYPE_TEST,
                variable=var,
                path=(),
                refinement_type=PyRefinementType(base=cls_type),
            ))

        for kw_attr, kw_pat in zip(pattern.kwd_attrs, pattern.kwd_patterns):
            sub_var = f"{var}.{kw_attr}"
            preds.extend(self.desugar_pattern(kw_pat, sub_var))

        for idx, pos_pat in enumerate(pattern.patterns):
            sub_var = f"{var}[{idx}]"
            preds.extend(self.desugar_pattern(pos_pat, sub_var))

        return preds

    # ----- mapping pattern -----------------------------------------

    def desugar_mapping_pattern(
        self, pattern: ast.MatchMapping, var: str
    ) -> List[HeapPredicate]:
        """
        ``case {'a': p1, 'b': p2}:`` →
        ['a' in var, desugar(p1, var['a']), 'b' in var, desugar(p2, var['b'])]
        """
        preds: List[HeapPredicate] = []
        for key_node, val_pat in zip(pattern.keys, pattern.patterns):
            key_val: Any = None
            if isinstance(key_node, ast.Constant):
                key_val = key_node.value

            preds.append(HeapPredicate(
                kind=HeapPredKind.CONTAINS,
                variable=var,
                path=(),
                value=key_val,
            ))

            if key_val is not None:
                sub_var = f"{var}[{key_val!r}]"
                preds.extend(self.desugar_pattern(val_pat, sub_var))

        return preds

    # ----- sequence pattern ----------------------------------------

    def desugar_sequence_pattern(
        self, pattern: ast.MatchSequence, var: str
    ) -> List[HeapPredicate]:
        """
        ``case [p1, p2, *rest]:`` →
        [len(var) >= 2, desugar(p1, var[0]), desugar(p2, var[1])]
        """
        preds: List[HeapPredicate] = []
        has_star = any(isinstance(p, ast.MatchStar) for p in pattern.patterns)
        fixed = [p for p in pattern.patterns if not isinstance(p, ast.MatchStar)]
        n_fixed = len(fixed)

        op = ">=" if has_star else "=="
        preds.append(HeapPredicate(
            kind=HeapPredKind.LENGTH_CONSTRAINT,
            variable=var,
            path=(),
            op=op,
            value=n_fixed,
        ))

        for idx, elem_pat in enumerate(fixed):
            sub_var = f"{var}[{idx}]"
            preds.extend(self.desugar_pattern(elem_pat, sub_var))

        return preds

    # ----- value pattern -------------------------------------------

    def desugar_value_pattern(
        self, pattern: ast.MatchValue, var: str
    ) -> HeapPredicate:
        """``case 42:`` → ``var == 42``."""
        const_val: Any = None
        if isinstance(pattern.value, ast.Constant):
            const_val = pattern.value.value
        return HeapPredicate(
            kind=HeapPredKind.VALUE_CONSTRAINT,
            variable=var,
            path=(),
            op="==",
            value=const_val,
        )

    # ----- or pattern ----------------------------------------------

    def desugar_or_pattern(
        self, pattern: ast.MatchOr, var: str
    ) -> HeapPredicate:
        """``case 1 | 2 | 3:`` → disjunction of value constraints."""
        sub_preds: List[HeapPredicate] = []
        for alt in pattern.patterns:
            alt_list = self.desugar_pattern(alt, var)
            if alt_list:
                combined = alt_list[0]
                for p in alt_list[1:]:
                    combined = _and_preds(combined, p)
                sub_preds.append(combined)

        if not sub_preds:
            return _true_pred()
        result = sub_preds[0]
        for sp in sub_preds[1:]:
            result = _or_preds(result, sp)
        return result

    # ----- singleton pattern ---------------------------------------

    def _desugar_singleton(
        self, pattern: ast.MatchSingleton, var: str
    ) -> HeapPredicate:
        """``case None:`` / ``case True:`` / ``case False:``."""
        if pattern.value is None:
            return HeapPredicate(
                kind=HeapPredKind.TYPE_TEST,
                variable=var,
                path=(),
                refinement_type=PyRefinementType(base=NoneRefType()),
            )
        return HeapPredicate(
            kind=HeapPredKind.VALUE_CONSTRAINT,
            variable=var,
            path=(),
            op="is",
            value=pattern.value,
        )
