"""Comprehensive modeling of Python dunder (magic) methods for refinement type inference.

This module provides precise type-level modeling of all Python special methods,
enabling the inference engine to determine result types, potential exceptions,
and refinement predicates for operations that desugar to dunder calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .class_hierarchy import ClassHierarchyAnalyzer, ClassInfo, MethodInfo


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class BinaryOpResult:
    """Result of modeling a binary operation (e.g. x + y)."""
    result_type: str
    may_raise: bool = False
    exception_type: Optional[str] = None
    commutative: bool = False
    reflected: bool = False


@dataclass
class ComparisonResult:
    """Result of modeling a comparison operation (e.g. x < y)."""
    result_type: str
    may_raise: bool = False
    always_bool: bool = True
    supports_chaining: bool = True


@dataclass
class ContainerAccessResult:
    """Result of modeling a container access (e.g. x[k], k in x)."""
    result_type: str
    may_raise: bool = False
    exception_type: Optional[str] = None
    modifies_container: bool = False


@dataclass
class ContextManagerType:
    """Result of modeling a context manager protocol."""
    enter_type: str
    exit_may_suppress: bool = False
    is_async: bool = False


@dataclass
class IterationType:
    """Result of modeling an iteration protocol."""
    element_type: str
    is_async: bool = False
    may_be_infinite: bool = False


@dataclass
class TruthinessInfo:
    """Result of modeling truthiness for ``if x:`` analysis."""
    always_true: bool = False
    always_false: bool = False
    depends_on: str = "bool"  # 'bool' | 'len' | 'custom' | 'identity'
    refinement_predicate: Optional[str] = None


# ---------------------------------------------------------------------------
# Dunder method tables
# ---------------------------------------------------------------------------

# Arithmetic operator -> (forward dunder, reflected dunder)
_ARITHMETIC_DUNDERS: Dict[str, Tuple[str, str]] = {
    "+":  ("__add__",      "__radd__"),
    "-":  ("__sub__",      "__rsub__"),
    "*":  ("__mul__",      "__rmul__"),
    "/":  ("__truediv__",  "__rtruediv__"),
    "//": ("__floordiv__", "__rfloordiv__"),
    "%":  ("__mod__",      "__rmod__"),
    "**": ("__pow__",      "__rpow__"),
    "@":  ("__matmul__",   "__rmatmul__"),
    "&":  ("__and__",      "__rand__"),
    "|":  ("__or__",       "__ror__"),
    "^":  ("__xor__",      "__rxor__"),
    "<<": ("__lshift__",   "__rlshift__"),
    ">>": ("__rshift__",   "__rrshift__"),
}

# Augmented-assignment operator -> in-place dunder
_AUGMENTED_DUNDERS: Dict[str, str] = {
    "+=":  "__iadd__",
    "-=":  "__isub__",
    "*=":  "__imul__",
    "/=":  "__itruediv__",
    "//=": "__ifloordiv__",
    "%=":  "__imod__",
    "**=": "__ipow__",
    "@=":  "__imatmul__",
    "&=":  "__iand__",
    "|=":  "__ior__",
    "^=":  "__ixor__",
    "<<=": "__ilshift__",
    ">>=": "__irshift__",
}

# Comparison operator -> (forward dunder, reflected dunder)
_COMPARISON_DUNDERS: Dict[str, Tuple[str, str]] = {
    "==": ("__eq__",  "__eq__"),
    "!=": ("__ne__",  "__ne__"),
    "<":  ("__lt__",  "__gt__"),
    ">":  ("__gt__",  "__lt__"),
    "<=": ("__le__",  "__ge__"),
    ">=": ("__ge__",  "__le__"),
}

# Unary operator -> dunder
_UNARY_DUNDERS: Dict[str, str] = {
    "-":   "__neg__",
    "+":   "__pos__",
    "~":   "__invert__",
    "abs": "__abs__",
}

# Container / subscript dunders
_CONTAINER_DUNDERS: Dict[str, str] = {
    "getitem":  "__getitem__",
    "setitem":  "__setitem__",
    "delitem":  "__delitem__",
    "contains": "__contains__",
    "len":      "__len__",
    "iter":     "__iter__",
    "next":     "__next__",
    "reversed": "__reversed__",
    "missing":  "__missing__",
}

# String-conversion dunders
_STRING_DUNDERS: Set[str] = {
    "__str__", "__repr__", "__format__", "__bytes__",
}

# Numeric-conversion dunders
_NUMERIC_DUNDERS: Set[str] = {
    "__int__", "__float__", "__complex__", "__bool__", "__index__",
    "__round__", "__trunc__", "__floor__", "__ceil__",
}

# All recognised dunders (superset used for look-ups)
_ALL_DUNDERS: Set[str] = (
    {fwd for fwd, _ in _ARITHMETIC_DUNDERS.values()}
    | {ref for _, ref in _ARITHMETIC_DUNDERS.values()}
    | set(_AUGMENTED_DUNDERS.values())
    | {fwd for fwd, _ in _COMPARISON_DUNDERS.values()}
    | {ref for _, ref in _COMPARISON_DUNDERS.values()}
    | set(_UNARY_DUNDERS.values())
    | set(_CONTAINER_DUNDERS.values())
    | _STRING_DUNDERS
    | _NUMERIC_DUNDERS
    | {
        "__hash__", "__call__", "__len__", "__length_hint__",
        "__enter__", "__exit__", "__aenter__", "__aexit__",
        "__await__", "__aiter__", "__anext__",
        "__init__", "__new__", "__del__",
        "__get__", "__set__", "__delete__", "__set_name__",
        "__init_subclass__", "__class_getitem__",
        "__instancecheck__", "__subclasscheck__",
        "__copy__", "__deepcopy__",
        "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__",
        "__sizeof__", "__dir__",
        "__getattr__", "__getattribute__", "__setattr__", "__delattr__",
    }
)


# ---------------------------------------------------------------------------
# Builtin type operation result tables
# ---------------------------------------------------------------------------

def _build_builtin_type_ops() -> Dict[str, Dict[str, str]]:
    """Return mapping: ``type_key -> result_type``.

    *type_key* is encoded as ``left_type:op:right_type`` for binary ops, or
    ``op:operand_type`` for unary ops.
    """
    table: Dict[str, Dict[str, str]] = {}

    # -- int ops ----------------------------------------------------------
    table["int"] = {
        "+:int":   "int",
        "+:float": "float",
        "+:complex": "complex",
        "+:bool":  "int",
        "-:int":   "int",
        "-:float": "float",
        "-:complex": "complex",
        "-:bool":  "int",
        "*:int":   "int",
        "*:float": "float",
        "*:complex": "complex",
        "*:bool":  "int",
        "*:str":   "str",
        "*:list":  "list",
        "*:tuple": "tuple",
        "*:bytes": "bytes",
        "/:int":   "float",
        "/:float": "float",
        "/:complex": "complex",
        "/:bool":  "float",
        "//:int":  "int",
        "//:float": "float",
        "//:bool": "int",
        "%:int":   "int",
        "%:float": "float",
        "%:bool":  "int",
        "**:int":  "int",
        "**:float": "float",
        "**:complex": "complex",
        "**:bool": "int",
        "&:int":   "int",
        "&:bool":  "int",
        "|:int":   "int",
        "|:bool":  "int",
        "^:int":   "int",
        "^:bool":  "int",
        "<<:int":  "int",
        "<<:bool": "int",
        ">>:int":  "int",
        ">>:bool": "int",
        "neg":     "int",
        "pos":     "int",
        "invert":  "int",
        "abs":     "int",
    }

    # -- bool ops (bool is subclass of int) --------------------------------
    table["bool"] = {
        "+:bool":  "int",
        "+:int":   "int",
        "+:float": "float",
        "+:complex": "complex",
        "-:bool":  "int",
        "-:int":   "int",
        "-:float": "float",
        "-:complex": "complex",
        "*:bool":  "int",
        "*:int":   "int",
        "*:float": "float",
        "*:complex": "complex",
        "/:bool":  "float",
        "/:int":   "float",
        "/:float": "float",
        "//:bool": "int",
        "//:int":  "int",
        "//:float": "float",
        "%:bool":  "int",
        "%:int":   "int",
        "%:float": "float",
        "**:bool": "int",
        "**:int":  "int",
        "**:float": "float",
        "&:bool":  "bool",
        "&:int":   "int",
        "|:bool":  "bool",
        "|:int":   "int",
        "^:bool":  "bool",
        "^:int":   "int",
        "<<:int":  "int",
        ">>:int":  "int",
        "neg":     "int",
        "pos":     "int",
        "invert":  "int",
        "abs":     "int",
    }

    # -- float ops ---------------------------------------------------------
    table["float"] = {
        "+:float":   "float",
        "+:int":     "float",
        "+:bool":    "float",
        "+:complex": "complex",
        "-:float":   "float",
        "-:int":     "float",
        "-:bool":    "float",
        "-:complex": "complex",
        "*:float":   "float",
        "*:int":     "float",
        "*:bool":    "float",
        "*:complex": "complex",
        "/:float":   "float",
        "/:int":     "float",
        "/:bool":    "float",
        "/:complex": "complex",
        "//:float":  "float",
        "//:int":    "float",
        "//:bool":   "float",
        "%:float":   "float",
        "%:int":     "float",
        "%:bool":    "float",
        "**:float":  "float",
        "**:int":    "float",
        "**:bool":   "float",
        "**:complex": "complex",
        "neg":       "float",
        "pos":       "float",
        "abs":       "float",
    }

    # -- complex ops -------------------------------------------------------
    table["complex"] = {
        "+:complex": "complex",
        "+:int":     "complex",
        "+:float":   "complex",
        "+:bool":    "complex",
        "-:complex": "complex",
        "-:int":     "complex",
        "-:float":   "complex",
        "-:bool":    "complex",
        "*:complex": "complex",
        "*:int":     "complex",
        "*:float":   "complex",
        "*:bool":    "complex",
        "/:complex": "complex",
        "/:int":     "complex",
        "/:float":   "complex",
        "/:bool":    "complex",
        "**:complex": "complex",
        "**:int":    "complex",
        "**:float":  "complex",
        "**:bool":   "complex",
        "neg":       "complex",
        "pos":       "complex",
        "abs":       "float",
    }

    # -- str ops -----------------------------------------------------------
    table["str"] = {
        "+:str":   "str",
        "*:int":   "str",
        "*:bool":  "str",
        "%:tuple": "str",
        "%:str":   "str",
        "%:int":   "str",
        "%:float": "str",
        "%:dict":  "str",
    }

    # -- bytes ops ---------------------------------------------------------
    table["bytes"] = {
        "+:bytes":     "bytes",
        "*:int":       "bytes",
        "*:bool":      "bytes",
        "%:tuple":     "bytes",
        "%:bytes":     "bytes",
        "%:int":       "bytes",
        "%:float":     "bytes",
        "%:dict":      "bytes",
    }

    # -- list ops ----------------------------------------------------------
    table["list"] = {
        "+:list":  "list",
        "*:int":   "list",
        "*:bool":  "list",
    }

    # -- tuple ops ---------------------------------------------------------
    table["tuple"] = {
        "+:tuple": "tuple",
        "*:int":   "tuple",
        "*:bool":  "tuple",
    }

    # -- set ops -----------------------------------------------------------
    table["set"] = {
        "&:set":         "set",
        "|:set":         "set",
        "^:set":         "set",
        "-:set":         "set",
        "&:frozenset":   "set",
        "|:frozenset":   "set",
        "^:frozenset":   "set",
        "-:frozenset":   "set",
    }

    # -- frozenset ops -----------------------------------------------------
    table["frozenset"] = {
        "&:frozenset":   "frozenset",
        "|:frozenset":   "frozenset",
        "^:frozenset":   "frozenset",
        "-:frozenset":   "frozenset",
        "&:set":         "frozenset",
        "|:set":         "frozenset",
        "^:set":         "frozenset",
        "-:set":         "frozenset",
    }

    # -- dict ops ----------------------------------------------------------
    table["dict"] = {
        "|:dict": "dict",
    }

    return table


# Commutative operators
_COMMUTATIVE_OPS: Set[str] = {"+", "*", "&", "|", "^"}

# Operators that can raise ZeroDivisionError with a zero right-hand side
_ZERO_DIV_OPS: Set[str] = {"/", "//", "%"}

# Builtin types that are always hashable
_HASHABLE_BUILTINS: Set[str] = {
    "int", "float", "complex", "bool", "str", "bytes",
    "tuple", "frozenset", "NoneType", "type", "range",
    "slice", "bytearray",
}

# Builtin types that are unhashable
_UNHASHABLE_BUILTINS: Set[str] = {"list", "dict", "set", "bytearray"}

# Builtin types and their dunders
_BUILTIN_DUNDERS: Dict[str, Set[str]] = {
    "int": {
        "__add__", "__radd__", "__sub__", "__rsub__",
        "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
        "__floordiv__", "__rfloordiv__", "__mod__", "__rmod__",
        "__pow__", "__rpow__", "__and__", "__rand__",
        "__or__", "__ror__", "__xor__", "__rxor__",
        "__lshift__", "__rlshift__", "__rshift__", "__rrshift__",
        "__neg__", "__pos__", "__abs__", "__invert__",
        "__int__", "__float__", "__complex__", "__bool__", "__index__",
        "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
        "__hash__", "__str__", "__repr__", "__format__",
        "__round__", "__trunc__", "__floor__", "__ceil__",
    },
    "float": {
        "__add__", "__radd__", "__sub__", "__rsub__",
        "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
        "__floordiv__", "__rfloordiv__", "__mod__", "__rmod__",
        "__pow__", "__rpow__",
        "__neg__", "__pos__", "__abs__",
        "__int__", "__float__", "__complex__", "__bool__",
        "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
        "__hash__", "__str__", "__repr__", "__format__",
        "__round__", "__trunc__", "__floor__", "__ceil__",
    },
    "complex": {
        "__add__", "__radd__", "__sub__", "__rsub__",
        "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
        "__pow__", "__rpow__",
        "__neg__", "__pos__", "__abs__",
        "__complex__", "__bool__",
        "__eq__", "__ne__",
        "__hash__", "__str__", "__repr__", "__format__",
    },
    "bool": {
        "__add__", "__radd__", "__sub__", "__rsub__",
        "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
        "__floordiv__", "__rfloordiv__", "__mod__", "__rmod__",
        "__pow__", "__rpow__", "__and__", "__rand__",
        "__or__", "__ror__", "__xor__", "__rxor__",
        "__neg__", "__pos__", "__abs__", "__invert__",
        "__int__", "__float__", "__complex__", "__bool__", "__index__",
        "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
        "__hash__", "__str__", "__repr__", "__format__",
    },
    "str": {
        "__add__", "__mul__", "__rmul__", "__mod__",
        "__contains__", "__getitem__", "__len__", "__iter__",
        "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
        "__hash__", "__str__", "__repr__", "__format__", "__bool__",
    },
    "bytes": {
        "__add__", "__mul__", "__rmul__", "__mod__",
        "__contains__", "__getitem__", "__len__", "__iter__",
        "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
        "__hash__", "__str__", "__repr__", "__format__", "__bool__",
    },
    "bytearray": {
        "__add__", "__iadd__", "__mul__", "__rmul__", "__imul__",
        "__contains__", "__getitem__", "__setitem__", "__delitem__",
        "__len__", "__iter__",
        "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
        "__str__", "__repr__", "__format__", "__bool__",
    },
    "list": {
        "__add__", "__iadd__", "__mul__", "__rmul__", "__imul__",
        "__contains__", "__getitem__", "__setitem__", "__delitem__",
        "__len__", "__iter__", "__reversed__",
        "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
        "__str__", "__repr__", "__format__", "__bool__",
    },
    "tuple": {
        "__add__", "__mul__", "__rmul__",
        "__contains__", "__getitem__", "__len__", "__iter__",
        "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
        "__hash__", "__str__", "__repr__", "__format__", "__bool__",
    },
    "dict": {
        "__contains__", "__getitem__", "__setitem__", "__delitem__",
        "__len__", "__iter__", "__reversed__",
        "__or__", "__ror__", "__ior__",
        "__eq__", "__ne__",
        "__str__", "__repr__", "__format__", "__bool__",
    },
    "set": {
        "__contains__", "__len__", "__iter__",
        "__and__", "__rand__", "__iand__",
        "__or__", "__ror__", "__ior__",
        "__xor__", "__rxor__", "__ixor__",
        "__sub__", "__rsub__", "__isub__",
        "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
        "__str__", "__repr__", "__format__", "__bool__",
    },
    "frozenset": {
        "__contains__", "__len__", "__iter__",
        "__and__", "__rand__", "__or__", "__ror__",
        "__xor__", "__rxor__", "__sub__", "__rsub__",
        "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
        "__hash__", "__str__", "__repr__", "__format__", "__bool__",
    },
    "range": {
        "__contains__", "__getitem__", "__len__", "__iter__", "__reversed__",
        "__eq__", "__ne__",
        "__hash__", "__str__", "__repr__", "__format__", "__bool__",
    },
    "NoneType": {
        "__eq__", "__ne__", "__hash__",
        "__bool__", "__str__", "__repr__",
    },
    "type": {
        "__call__", "__instancecheck__", "__subclasscheck__",
        "__eq__", "__ne__", "__hash__",
        "__str__", "__repr__", "__format__",
    },
    "slice": {
        "__eq__", "__ne__", "__hash__",
        "__str__", "__repr__",
    },
}

# Iteration element types for builtin containers
_ITER_ELEMENT_TYPES: Dict[str, str] = {
    "str":        "str",
    "bytes":      "int",
    "bytearray":  "int",
    "list":       "Any",   # parameterised: list[T] -> T
    "tuple":      "Any",
    "set":        "Any",
    "frozenset":  "Any",
    "dict":       "Any",   # dict[K,V] -> K
    "range":      "int",
    "enumerate":  "tuple",
    "zip":        "tuple",
    "map":        "Any",
    "filter":     "Any",
    "reversed":   "Any",
}

# Context-manager __enter__ return types for well-known stdlib types
_CONTEXT_MANAGER_ENTER: Dict[str, str] = {
    "open":                      "IO",
    "io.FileIO":                 "io.FileIO",
    "io.BufferedReader":         "io.BufferedReader",
    "io.BufferedWriter":         "io.BufferedWriter",
    "io.TextIOWrapper":          "io.TextIOWrapper",
    "contextlib.suppress":       "None",
    "contextlib.redirect_stdout": "None",
    "contextlib.redirect_stderr": "None",
    "contextlib.closing":        "T",
    "contextlib.ExitStack":      "contextlib.ExitStack",
    "contextlib.AsyncExitStack": "contextlib.AsyncExitStack",
    "threading.Lock":            "bool",
    "threading.RLock":           "bool",
    "threading.Condition":       "bool",
    "threading.Semaphore":       "bool",
    "decimal.localcontext":      "decimal.Context",
    "tempfile.TemporaryFile":    "IO",
    "tempfile.NamedTemporaryFile": "IO",
    "tempfile.TemporaryDirectory": "str",
    "unittest.mock.patch":       "unittest.mock.MagicMock",
    "socket.socket":             "socket.socket",
}

# Types whose __exit__ can suppress exceptions
_EXIT_MAY_SUPPRESS: Set[str] = {
    "contextlib.suppress",
}


# ---------------------------------------------------------------------------
# DunderModel
# ---------------------------------------------------------------------------

class DunderModel:
    """Models the semantics of Python dunder methods for type inference.

    Given an operator and operand types (as strings), this class determines
    the result type, whether the operation may raise, and what refinement
    predicates can be derived.
    """

    def __init__(self, hierarchy: ClassHierarchyAnalyzer) -> None:
        self._hierarchy = hierarchy
        self._arithmetic_dunders = self._init_arithmetic_dunders()
        self._comparison_dunders = self._init_comparison_dunders()
        self._unary_dunders = self._init_unary_dunders()
        self._builtin_type_ops = self._init_builtin_type_ops()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_arithmetic_dunders() -> Dict[str, Tuple[str, str]]:
        """Map operator symbols to (forward-dunder, reflected-dunder)."""
        return dict(_ARITHMETIC_DUNDERS)

    @staticmethod
    def _init_comparison_dunders() -> Dict[str, Tuple[str, str]]:
        """Map comparison operators to (forward-dunder, reflected-dunder)."""
        return dict(_COMPARISON_DUNDERS)

    @staticmethod
    def _init_unary_dunders() -> Dict[str, str]:
        """Map unary operator symbols to dunder names."""
        return dict(_UNARY_DUNDERS)

    @staticmethod
    def _init_builtin_type_ops() -> Dict[str, Dict[str, str]]:
        """Comprehensive mapping of builtin type -> op-key -> result type."""
        return _build_builtin_type_ops()

    # ------------------------------------------------------------------
    # Core modelling entry points
    # ------------------------------------------------------------------

    def model_binary_op(
        self,
        op: str,
        left_type: str,
        right_type: str,
    ) -> BinaryOpResult:
        """Determine the result type for ``left_type <op> right_type``.

        Resolution order mirrors CPython:
        1. If *right_type* is a proper subclass of *left_type*, try the
           reflected dunder on *right_type* first.
        2. Try the forward dunder on *left_type*.
        3. Fall back to the reflected dunder on *right_type*.
        4. If neither succeeds, the operation is unsupported (TypeError).
        """
        if op not in self._arithmetic_dunders:
            return BinaryOpResult(
                result_type="Never",
                may_raise=True,
                exception_type="TypeError",
            )

        fwd_dunder, ref_dunder = self._arithmetic_dunders[op]

        # Fast-path: builtin type table
        builtin_result = self._get_builtin_result(left_type, op, right_type)
        if builtin_result is not None:
            may_raise = op in _ZERO_DIV_OPS
            exc = "ZeroDivisionError" if may_raise else None
            return BinaryOpResult(
                result_type=builtin_result,
                may_raise=may_raise,
                exception_type=exc,
                commutative=op in _COMMUTATIVE_OPS,
                reflected=False,
            )

        # Check if right is a subclass of left – if so, try reflected first
        right_is_sub = self._is_subclass(right_type, left_type)

        if right_is_sub:
            ref_method = self._resolve_dunder(right_type, ref_dunder)
            if ref_method is not None:
                return BinaryOpResult(
                    result_type=self._return_type_str(ref_method),
                    may_raise=True,
                    commutative=op in _COMMUTATIVE_OPS,
                    reflected=True,
                )

        # Try forward dunder on left
        fwd_method = self._resolve_dunder(left_type, fwd_dunder)
        if fwd_method is not None:
            return BinaryOpResult(
                result_type=self._return_type_str(fwd_method),
                may_raise=True,
                commutative=op in _COMMUTATIVE_OPS,
                reflected=False,
            )

        # Try reflected dunder on right (if not already tried)
        if not right_is_sub:
            ref_method = self._resolve_dunder(right_type, ref_dunder)
            if ref_method is not None:
                return BinaryOpResult(
                    result_type=self._return_type_str(ref_method),
                    may_raise=True,
                    commutative=op in _COMMUTATIVE_OPS,
                    reflected=True,
                )

        # Unsupported
        return BinaryOpResult(
            result_type="Never",
            may_raise=True,
            exception_type="TypeError",
        )

    def model_unary_op(self, op: str, operand_type: str) -> BinaryOpResult:
        """Determine the result type for a unary operation."""
        if op not in self._unary_dunders:
            return BinaryOpResult(
                result_type="Never",
                may_raise=True,
                exception_type="TypeError",
            )

        dunder = self._unary_dunders[op]

        # Fast-path for builtins
        op_key_map = {"-": "neg", "+": "pos", "~": "invert", "abs": "abs"}
        op_key = op_key_map.get(op, op)
        type_table = self._builtin_type_ops.get(operand_type)
        if type_table is not None and op_key in type_table:
            return BinaryOpResult(
                result_type=type_table[op_key],
                may_raise=False,
            )

        method = self._resolve_dunder(operand_type, dunder)
        if method is not None:
            return BinaryOpResult(
                result_type=self._return_type_str(method),
                may_raise=True,
            )

        return BinaryOpResult(
            result_type="Never",
            may_raise=True,
            exception_type="TypeError",
        )

    def model_comparison(
        self,
        op: str,
        left_type: str,
        right_type: str,
    ) -> ComparisonResult:
        """Model a comparison operation.

        Notes:
        - ``__eq__`` may return non-bool (e.g. NumPy arrays).
        - All comparisons support chaining in Python.
        - ``!=`` falls back to negating ``__eq__`` when ``__ne__`` is absent.
        """
        if op not in self._comparison_dunders:
            return ComparisonResult(
                result_type="Never",
                may_raise=True,
                always_bool=True,
                supports_chaining=False,
            )

        fwd_dunder, ref_dunder = self._comparison_dunders[op]

        # For builtin types, comparisons always return bool
        if left_type in _BUILTIN_DUNDERS or right_type in _BUILTIN_DUNDERS:
            # Special case: complex doesn't support ordering
            if op in ("<", ">", "<=", ">="):
                if left_type == "complex" or right_type == "complex":
                    return ComparisonResult(
                        result_type="Never",
                        may_raise=True,
                        always_bool=True,
                        supports_chaining=True,
                    )
            return ComparisonResult(
                result_type="bool",
                may_raise=False,
                always_bool=True,
                supports_chaining=True,
            )

        # Check user-defined types via hierarchy
        fwd_method = self._resolve_dunder(left_type, fwd_dunder)
        if fwd_method is not None:
            ret = self._return_type_str(fwd_method)
            return ComparisonResult(
                result_type=ret,
                may_raise=True,
                always_bool=(ret == "bool"),
                supports_chaining=True,
            )

        ref_method = self._resolve_dunder(right_type, ref_dunder)
        if ref_method is not None:
            ret = self._return_type_str(ref_method)
            return ComparisonResult(
                result_type=ret,
                may_raise=True,
                always_bool=(ret == "bool"),
                supports_chaining=True,
            )

        # Default: identity comparison for == / !=
        if op in ("==", "!="):
            return ComparisonResult(
                result_type="bool",
                may_raise=False,
                always_bool=True,
                supports_chaining=True,
            )

        return ComparisonResult(
            result_type="Never",
            may_raise=True,
            always_bool=True,
            supports_chaining=True,
        )

    def model_augmented_assign(
        self,
        op: str,
        target_type: str,
        value_type: str,
    ) -> BinaryOpResult:
        """Model ``target <op>= value`` (e.g. ``x += 1``).

        Resolution: try ``__iadd__`` first; fall back to ``__add__``.
        Mutable builtins (list, set, dict) use in-place variants that
        modify the container and return the same object.
        """
        # Normalise op: "+=" -> "+", etc.
        base_op = op.rstrip("=")

        # Try in-place dunder
        iadd_dunder = _AUGMENTED_DUNDERS.get(op)
        if iadd_dunder is not None:
            method = self._resolve_dunder(target_type, iadd_dunder)
            if method is not None:
                return BinaryOpResult(
                    result_type=self._return_type_str(method),
                    may_raise=True,
                    reflected=False,
                )

        # For mutable builtins, augmented assignment returns same type
        mutable_aug: Dict[str, Dict[str, str]] = {
            "list":  {"+=": "list", "*=": "list"},
            "set":   {"|=": "set", "&=": "set", "^=": "set", "-=": "set"},
            "dict":  {"|=": "dict"},
            "bytearray": {"+=": "bytearray", "*=": "bytearray"},
        }
        if target_type in mutable_aug and op in mutable_aug[target_type]:
            return BinaryOpResult(
                result_type=mutable_aug[target_type][op],
                may_raise=False,
                reflected=False,
            )

        # Fall back to regular binary op
        return self.model_binary_op(base_op, target_type, value_type)

    # ------------------------------------------------------------------
    # Container access
    # ------------------------------------------------------------------

    def model_container_access(
        self,
        container_type: str,
        access: str,
        key_type: Optional[str] = None,
    ) -> ContainerAccessResult:
        """Model subscript / container protocol operations.

        *access* is one of: ``getitem``, ``setitem``, ``delitem``,
        ``contains``, ``len``, ``iter``.
        """
        handlers: Dict[str, Dict[str, ContainerAccessResult]] = {
            "list": {
                "getitem": ContainerAccessResult(
                    result_type="list" if key_type == "slice" else "Any",
                    may_raise=key_type != "slice",
                    exception_type="IndexError" if key_type != "slice" else None,
                ),
                "setitem": ContainerAccessResult(
                    result_type="None",
                    may_raise=True,
                    exception_type="IndexError",
                    modifies_container=True,
                ),
                "delitem": ContainerAccessResult(
                    result_type="None",
                    may_raise=True,
                    exception_type="IndexError",
                    modifies_container=True,
                ),
                "contains": ContainerAccessResult(result_type="bool"),
                "len": ContainerAccessResult(result_type="int"),
                "iter": ContainerAccessResult(result_type="list_iterator"),
            },
            "tuple": {
                "getitem": ContainerAccessResult(
                    result_type="tuple" if key_type == "slice" else "Any",
                    may_raise=key_type != "slice",
                    exception_type="IndexError" if key_type != "slice" else None,
                ),
                "contains": ContainerAccessResult(result_type="bool"),
                "len": ContainerAccessResult(result_type="int"),
                "iter": ContainerAccessResult(result_type="tuple_iterator"),
            },
            "dict": {
                "getitem": ContainerAccessResult(
                    result_type="Any",
                    may_raise=True,
                    exception_type="KeyError",
                ),
                "setitem": ContainerAccessResult(
                    result_type="None",
                    modifies_container=True,
                ),
                "delitem": ContainerAccessResult(
                    result_type="None",
                    may_raise=True,
                    exception_type="KeyError",
                    modifies_container=True,
                ),
                "contains": ContainerAccessResult(result_type="bool"),
                "len": ContainerAccessResult(result_type="int"),
                "iter": ContainerAccessResult(result_type="dict_keyiterator"),
            },
            "str": {
                "getitem": ContainerAccessResult(
                    result_type="str",
                    may_raise=key_type != "slice",
                    exception_type="IndexError" if key_type != "slice" else None,
                ),
                "contains": ContainerAccessResult(result_type="bool"),
                "len": ContainerAccessResult(result_type="int"),
                "iter": ContainerAccessResult(result_type="str_ascii_iterator"),
            },
            "bytes": {
                "getitem": ContainerAccessResult(
                    result_type="bytes" if key_type == "slice" else "int",
                    may_raise=key_type != "slice",
                    exception_type="IndexError" if key_type != "slice" else None,
                ),
                "contains": ContainerAccessResult(result_type="bool"),
                "len": ContainerAccessResult(result_type="int"),
                "iter": ContainerAccessResult(result_type="bytes_iterator"),
            },
            "bytearray": {
                "getitem": ContainerAccessResult(
                    result_type="bytearray" if key_type == "slice" else "int",
                    may_raise=key_type != "slice",
                    exception_type="IndexError" if key_type != "slice" else None,
                ),
                "setitem": ContainerAccessResult(
                    result_type="None",
                    may_raise=True,
                    exception_type="IndexError",
                    modifies_container=True,
                ),
                "delitem": ContainerAccessResult(
                    result_type="None",
                    may_raise=True,
                    exception_type="IndexError",
                    modifies_container=True,
                ),
                "contains": ContainerAccessResult(result_type="bool"),
                "len": ContainerAccessResult(result_type="int"),
                "iter": ContainerAccessResult(result_type="bytearray_iterator"),
            },
            "set": {
                "contains": ContainerAccessResult(result_type="bool"),
                "len": ContainerAccessResult(result_type="int"),
                "iter": ContainerAccessResult(result_type="set_iterator"),
            },
            "frozenset": {
                "contains": ContainerAccessResult(result_type="bool"),
                "len": ContainerAccessResult(result_type="int"),
                "iter": ContainerAccessResult(result_type="frozenset_iterator"),
            },
            "range": {
                "getitem": ContainerAccessResult(
                    result_type="range" if key_type == "slice" else "int",
                    may_raise=key_type != "slice",
                    exception_type="IndexError" if key_type != "slice" else None,
                ),
                "contains": ContainerAccessResult(result_type="bool"),
                "len": ContainerAccessResult(result_type="int"),
                "iter": ContainerAccessResult(result_type="range_iterator"),
            },
        }

        # Check builtin table first
        if container_type in handlers:
            type_table = handlers[container_type]
            if access in type_table:
                return type_table[access]

        # Fall through to hierarchy-based resolution
        dunder = _CONTAINER_DUNDERS.get(access)
        if dunder is None:
            return ContainerAccessResult(
                result_type="Never",
                may_raise=True,
                exception_type="TypeError",
            )

        method = self._resolve_dunder(container_type, dunder)
        if method is not None:
            modifies = access in ("setitem", "delitem")
            return ContainerAccessResult(
                result_type=self._return_type_str(method),
                may_raise=True,
                modifies_container=modifies,
            )

        return ContainerAccessResult(
            result_type="Never",
            may_raise=True,
            exception_type="TypeError",
        )

    # ------------------------------------------------------------------
    # Protocols
    # ------------------------------------------------------------------

    def model_context_manager(self, class_name: str) -> ContextManagerType:
        """Determine ``__enter__`` return type and ``__exit__`` behaviour."""
        # Well-known stdlib types
        if class_name in _CONTEXT_MANAGER_ENTER:
            return ContextManagerType(
                enter_type=_CONTEXT_MANAGER_ENTER[class_name],
                exit_may_suppress=class_name in _EXIT_MAY_SUPPRESS,
                is_async=class_name.startswith("contextlib.Async"),
            )

        # Hierarchy-based lookup
        enter = self._resolve_dunder(class_name, "__enter__")
        aexit = self._resolve_dunder(class_name, "__aexit__")
        is_async = aexit is not None and enter is None

        if is_async:
            aenter = self._resolve_dunder(class_name, "__aenter__")
            enter_type = self._return_type_str(aenter) if aenter else class_name
        else:
            enter_type = self._return_type_str(enter) if enter else class_name

        exit_method = self._resolve_dunder(
            class_name, "__aexit__" if is_async else "__exit__",
        )
        exit_may_suppress = False
        if exit_method is not None:
            ret = self._return_type_str(exit_method)
            if ret not in ("None", "NoneType", "bool"):
                exit_may_suppress = True
            elif ret == "bool":
                exit_may_suppress = True

        return ContextManagerType(
            enter_type=enter_type,
            exit_may_suppress=exit_may_suppress,
            is_async=is_async,
        )

    def model_callable(self, class_name: str) -> Optional[str]:
        """If *class_name* defines ``__call__``, return a signature description."""
        if class_name == "type":
            return "(cls, *args, **kwargs) -> instance"

        # Builtin callable types
        builtin_callables: Dict[str, str] = {
            "function":     "(*args, **kwargs) -> Any",
            "builtin_function_or_method": "(*args, **kwargs) -> Any",
            "method":       "(self, *args, **kwargs) -> Any",
            "staticmethod": "(*args, **kwargs) -> Any",
            "classmethod":  "(cls, *args, **kwargs) -> Any",
        }
        if class_name in builtin_callables:
            return builtin_callables[class_name]

        method = self._resolve_dunder(class_name, "__call__")
        if method is not None:
            params = ", ".join(
                f"{name}: {self._type_to_str(ty)}"
                for name, ty in method.params
                if name != "self"
            )
            ret = self._return_type_str(method)
            return f"({params}) -> {ret}"

        return None

    def model_hash(self, class_name: str) -> Optional[str]:
        """Check ``__hash__`` for *class_name*.

        Returns a description string or ``None`` if unhashable.
        """
        if class_name in _UNHASHABLE_BUILTINS:
            return None
        if class_name in _HASHABLE_BUILTINS:
            return "int"

        method = self._resolve_dunder(class_name, "__hash__")
        if method is not None:
            return self._return_type_str(method)

        # If __eq__ is defined without __hash__, the type is unhashable
        eq_method = self._resolve_dunder(class_name, "__eq__")
        if eq_method is not None:
            hash_method = self._resolve_dunder(class_name, "__hash__")
            if hash_method is None:
                return None

        # Default: inherits object.__hash__
        return "int"

    def model_string_conversion(
        self,
        class_name: str,
        method: str = "__str__",
    ) -> str:
        """Determine result of ``str(x)``, ``repr(x)``, or ``format(x)``."""
        if method not in ("__str__", "__repr__", "__format__", "__bytes__"):
            return "str"

        if method == "__bytes__":
            return "bytes"

        # All builtins have __str__ and __repr__
        if class_name in _BUILTIN_DUNDERS:
            return "str"

        resolved = self._resolve_dunder(class_name, method)
        if resolved is not None:
            return self._return_type_str(resolved)

        # Falls back to object.__str__ / object.__repr__
        return "str"

    def model_numeric_conversion(self, class_name: str) -> Dict[str, str]:
        """Determine which numeric conversions are supported and their types."""
        results: Dict[str, str] = {}

        conversion_map: Dict[str, str] = {
            "__int__":     "int",
            "__float__":   "float",
            "__complex__": "complex",
            "__bool__":    "bool",
            "__index__":   "int",
        }

        for dunder, ret_type in conversion_map.items():
            # Check builtins
            builtin_dunders = _BUILTIN_DUNDERS.get(class_name, set())
            if dunder in builtin_dunders:
                results[dunder] = ret_type
                continue

            # Check hierarchy
            method = self._resolve_dunder(class_name, dunder)
            if method is not None:
                results[dunder] = ret_type

        return results

    # ------------------------------------------------------------------
    # Refinement predicates
    # ------------------------------------------------------------------

    def get_truthiness_refinement(self, type_name: str) -> TruthinessInfo:
        """Model what ``if x:`` means for *type_name*."""
        static_table: Dict[str, TruthinessInfo] = {
            "NoneType": TruthinessInfo(
                always_true=False,
                always_false=True,
                depends_on="identity",
                refinement_predicate="x is None",
            ),
            "bool": TruthinessInfo(
                depends_on="identity",
                refinement_predicate="x == True",
            ),
            "int": TruthinessInfo(
                depends_on="bool",
                refinement_predicate="x != 0",
            ),
            "float": TruthinessInfo(
                depends_on="bool",
                refinement_predicate="x != 0.0",
            ),
            "complex": TruthinessInfo(
                depends_on="bool",
                refinement_predicate="x != 0j",
            ),
            "str": TruthinessInfo(
                depends_on="len",
                refinement_predicate="len(x) > 0",
            ),
            "bytes": TruthinessInfo(
                depends_on="len",
                refinement_predicate="len(x) > 0",
            ),
            "bytearray": TruthinessInfo(
                depends_on="len",
                refinement_predicate="len(x) > 0",
            ),
            "list": TruthinessInfo(
                depends_on="len",
                refinement_predicate="len(x) > 0",
            ),
            "tuple": TruthinessInfo(
                depends_on="len",
                refinement_predicate="len(x) > 0",
            ),
            "dict": TruthinessInfo(
                depends_on="len",
                refinement_predicate="len(x) > 0",
            ),
            "set": TruthinessInfo(
                depends_on="len",
                refinement_predicate="len(x) > 0",
            ),
            "frozenset": TruthinessInfo(
                depends_on="len",
                refinement_predicate="len(x) > 0",
            ),
            "range": TruthinessInfo(
                depends_on="len",
                refinement_predicate="len(x) > 0",
            ),
            "type": TruthinessInfo(
                always_true=True,
                depends_on="identity",
                refinement_predicate=None,
            ),
            "function": TruthinessInfo(
                always_true=True,
                depends_on="identity",
                refinement_predicate=None,
            ),
            "module": TruthinessInfo(
                always_true=True,
                depends_on="identity",
                refinement_predicate=None,
            ),
        }

        if type_name in static_table:
            return static_table[type_name]

        # User-defined types: check for __bool__ or __len__
        bool_method = self._resolve_dunder(type_name, "__bool__")
        if bool_method is not None:
            return TruthinessInfo(
                depends_on="custom",
                refinement_predicate=f"{type_name}.__bool__(x)",
            )

        len_method = self._resolve_dunder(type_name, "__len__")
        if len_method is not None:
            return TruthinessInfo(
                depends_on="len",
                refinement_predicate="len(x) > 0",
            )

        # Default: objects without __bool__/__len__ are always truthy
        return TruthinessInfo(
            always_true=True,
            depends_on="identity",
            refinement_predicate=None,
        )

    def get_len_refinement(self, type_name: str) -> Optional[str]:
        """Return the constraint on ``len(x)`` if supported, else ``None``.

        For all types that support ``__len__``, the result is always ``>= 0``.
        """
        types_with_len: Set[str] = {
            "str", "bytes", "bytearray", "list", "tuple",
            "dict", "set", "frozenset", "range",
            "memoryview", "array.array", "collections.deque",
        }
        if type_name in types_with_len:
            return "int >= 0"

        method = self._resolve_dunder(type_name, "__len__")
        if method is not None:
            return "int >= 0"

        return None

    def get_iter_refinement(self, type_name: str) -> Optional[IterationType]:
        """Determine the element type produced by iterating over *type_name*."""
        if type_name in _ITER_ELEMENT_TYPES:
            return IterationType(
                element_type=_ITER_ELEMENT_TYPES[type_name],
                is_async=False,
                may_be_infinite=type_name in ("map", "filter", "itertools.count",
                                               "itertools.cycle", "itertools.repeat"),
            )

        # Check for async iteration
        aiter_method = self._resolve_dunder(type_name, "__aiter__")
        if aiter_method is not None:
            anext_method = self._resolve_dunder(type_name, "__anext__")
            elem = self._return_type_str(anext_method) if anext_method else "Any"
            return IterationType(
                element_type=elem,
                is_async=True,
                may_be_infinite=False,
            )

        # Synchronous iteration
        iter_method = self._resolve_dunder(type_name, "__iter__")
        if iter_method is not None:
            next_method = self._resolve_dunder(type_name, "__next__")
            elem = self._return_type_str(next_method) if next_method else "Any"
            return IterationType(
                element_type=elem,
                is_async=False,
                may_be_infinite=False,
            )

        # Fallback: __getitem__ with integer keys (old-style iteration)
        getitem_method = self._resolve_dunder(type_name, "__getitem__")
        if getitem_method is not None:
            return IterationType(
                element_type=self._return_type_str(getitem_method),
                is_async=False,
                may_be_infinite=False,
            )

        return None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_supported_dunders(self, type_name: str) -> Set[str]:
        """Return the full set of dunders supported by *type_name*."""
        result: Set[str] = set()

        # Builtin dunders
        if type_name in _BUILTIN_DUNDERS:
            result.update(_BUILTIN_DUNDERS[type_name])

        # Hierarchy-based lookup
        class_info = self._hierarchy.get_class(type_name)
        if class_info is not None:
            for method_name in class_info.methods:
                if method_name.startswith("__") and method_name.endswith("__"):
                    result.add(method_name)

            # Walk MRO for inherited dunders
            for base_name in class_info.mro:
                if base_name == type_name:
                    continue
                base_info = self._hierarchy.get_class(base_name)
                if base_info is not None:
                    for method_name in base_info.methods:
                        if method_name.startswith("__") and method_name.endswith("__"):
                            result.add(method_name)

        return result

    def check_dunder_consistency(self, class_name: str) -> List[str]:
        """Check for common dunder-method inconsistencies.

        Returns a list of warning messages (empty if consistent).
        """
        warnings: List[str] = []

        supported = self.get_supported_dunders(class_name)

        # __eq__ without __hash__ makes the class unhashable (by design in
        # Python 3), but users may not realise this.
        if "__eq__" in supported and "__hash__" not in supported:
            if class_name not in _UNHASHABLE_BUILTINS:
                warnings.append(
                    f"{class_name} defines __eq__ but not __hash__; "
                    f"instances will be unhashable"
                )

        # Ordering operators without @total_ordering may be incomplete
        ordering_ops = {"__lt__", "__le__", "__gt__", "__ge__"}
        present_ordering = ordering_ops & supported
        if present_ordering and present_ordering != ordering_ops:
            missing = ordering_ops - present_ordering
            warnings.append(
                f"{class_name} defines {sorted(present_ordering)} but not "
                f"{sorted(missing)}; consider using @functools.total_ordering"
            )

        # __enter__ without __exit__ (or vice versa)
        has_enter = "__enter__" in supported
        has_exit = "__exit__" in supported
        if has_enter != has_exit:
            missing_cm = "__exit__" if has_enter else "__enter__"
            warnings.append(
                f"{class_name} defines one of __enter__/__exit__ but not "
                f"{missing_cm}; context manager protocol is incomplete"
            )

        # Async context manager consistency
        has_aenter = "__aenter__" in supported
        has_aexit = "__aexit__" in supported
        if has_aenter != has_aexit:
            missing_acm = "__aexit__" if has_aenter else "__aenter__"
            warnings.append(
                f"{class_name} defines one of __aenter__/__aexit__ but not "
                f"{missing_acm}; async context manager protocol is incomplete"
            )

        # __iter__ without __next__ (iterable vs iterator confusion)
        if "__next__" in supported and "__iter__" not in supported:
            warnings.append(
                f"{class_name} defines __next__ but not __iter__; "
                f"iterators should return self from __iter__"
            )

        # __aiter__ without __anext__
        if "__anext__" in supported and "__aiter__" not in supported:
            warnings.append(
                f"{class_name} defines __anext__ but not __aiter__; "
                f"async iterators should return self from __aiter__"
            )

        # __del__ usage warning
        if "__del__" in supported:
            warnings.append(
                f"{class_name} defines __del__; note that destructor "
                f"invocation timing is not guaranteed by CPython"
            )

        # __contains__ without __iter__ (unusual but not wrong)
        if "__contains__" in supported and "__iter__" not in supported:
            if "__getitem__" not in supported:
                warnings.append(
                    f"{class_name} defines __contains__ without __iter__ "
                    f"or __getitem__; the 'in' operator works but iteration "
                    f"does not"
                )

        # Reflected without forward (unusual)
        for op, (fwd, ref) in _ARITHMETIC_DUNDERS.items():
            if ref in supported and fwd not in supported:
                warnings.append(
                    f"{class_name} defines {ref} without {fwd}; "
                    f"the reflected operator may never be called for "
                    f"same-type operations"
                )

        # __setitem__ without __getitem__
        if "__setitem__" in supported and "__getitem__" not in supported:
            warnings.append(
                f"{class_name} defines __setitem__ without __getitem__"
            )

        # __delitem__ without __getitem__
        if "__delitem__" in supported and "__getitem__" not in supported:
            warnings.append(
                f"{class_name} defines __delitem__ without __getitem__"
            )

        return warnings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_dunder(
        self,
        type_name: str,
        dunder_name: str,
    ) -> Optional[MethodInfo]:
        """Find *dunder_name* in the MRO of *type_name*.

        Returns the ``MethodInfo`` for the first match, or ``None``.
        """
        class_info = self._hierarchy.get_class(type_name)
        if class_info is None:
            return None

        # Check the class itself
        if dunder_name in class_info.methods:
            return class_info.methods[dunder_name]

        # Walk the MRO
        for base_name in class_info.mro:
            if base_name == type_name:
                continue
            base_info = self._hierarchy.get_class(base_name)
            if base_info is not None and dunder_name in base_info.methods:
                return base_info.methods[dunder_name]

        return None

    def _get_builtin_result(
        self,
        left_type: str,
        op: str,
        right_type: str,
    ) -> Optional[str]:
        """Look up the result type in the builtin type table.

        Returns ``None`` if the combination is not in the table.
        """
        type_table = self._builtin_type_ops.get(left_type)
        if type_table is not None:
            key = f"{op}:{right_type}"
            result = type_table.get(key)
            if result is not None:
                return result

        # Try reverse lookup for commutative ops
        if op in _COMMUTATIVE_OPS:
            rev_table = self._builtin_type_ops.get(right_type)
            if rev_table is not None:
                key = f"{op}:{left_type}"
                result = rev_table.get(key)
                if result is not None:
                    return result

        return None

    def _is_subclass(self, child: str, parent: str) -> bool:
        """Check whether *child* is a subclass of *parent* using the hierarchy."""
        if child == parent:
            return False

        # Builtin subclass relationships
        builtin_subclass: Dict[str, str] = {
            "bool": "int",
        }
        if builtin_subclass.get(child) == parent:
            return True

        class_info = self._hierarchy.get_class(child)
        if class_info is not None:
            return parent in class_info.mro

        return False

    @staticmethod
    def _return_type_str(method: MethodInfo) -> str:
        """Extract a string representation of the return type from *method*."""
        ret = method.return_type
        if ret is None:
            return "Any"
        if isinstance(ret, str):
            return ret
        # If it's a RefinementType or similar object, use str()
        return str(ret)

    @staticmethod
    def _type_to_str(ty: object) -> str:
        """Convert a type annotation to a string."""
        if ty is None:
            return "Any"
        if isinstance(ty, str):
            return ty
        return str(ty)
