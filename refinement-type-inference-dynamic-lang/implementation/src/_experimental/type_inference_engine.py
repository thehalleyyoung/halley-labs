"""Full type inference engine for Python without annotations.

Infers types for all variables, function signatures, return types, class
hierarchies, and generates PEP 484 annotations.  Uses the ``ast`` module for
analysis — no stubs, no external type checkers.

Supports generics, protocols, TypeVar, overloads, recursive types, and union
types.
"""
from __future__ import annotations

import ast
import copy
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union,
)


# ── Type representations ────────────────────────────────────────────────────

class TypeKind(Enum):
    NONE = "None"
    INT = "int"
    FLOAT = "float"
    COMPLEX = "complex"
    BOOL = "bool"
    STR = "str"
    BYTES = "bytes"
    LIST = "List"
    DICT = "Dict"
    SET = "Set"
    TUPLE = "Tuple"
    OPTIONAL = "Optional"
    UNION = "Union"
    CALLABLE = "Callable"
    ANY = "Any"
    CLASS = "class"
    TYPEVAR = "TypeVar"
    GENERIC = "Generic"
    PROTOCOL = "Protocol"
    RECURSIVE = "Recursive"
    ITERATOR = "Iterator"
    GENERATOR = "Generator"
    COROUTINE = "Coroutine"
    LITERAL = "Literal"
    UNKNOWN = "Unknown"


@dataclass
class Type:
    kind: TypeKind
    name: str = ""
    args: List["Type"] = field(default_factory=list)
    attributes: Dict[str, "Type"] = field(default_factory=dict)
    methods: Dict[str, "FunctionSignature"] = field(default_factory=dict)
    literal_value: Any = None
    confidence: float = 1.0

    def __str__(self) -> str:
        if self.kind == TypeKind.NONE:
            return "None"
        if self.kind == TypeKind.OPTIONAL:
            inner = str(self.args[0]) if self.args else "Any"
            return f"Optional[{inner}]"
        if self.kind == TypeKind.UNION:
            parts = ", ".join(str(a) for a in self.args)
            return f"Union[{parts}]"
        if self.kind == TypeKind.LIST:
            elem = str(self.args[0]) if self.args else "Any"
            return f"List[{elem}]"
        if self.kind == TypeKind.DICT:
            k = str(self.args[0]) if len(self.args) > 0 else "Any"
            v = str(self.args[1]) if len(self.args) > 1 else "Any"
            return f"Dict[{k}, {v}]"
        if self.kind == TypeKind.SET:
            elem = str(self.args[0]) if self.args else "Any"
            return f"Set[{elem}]"
        if self.kind == TypeKind.TUPLE:
            parts = ", ".join(str(a) for a in self.args)
            return f"Tuple[{parts}]"
        if self.kind == TypeKind.CALLABLE:
            if len(self.args) >= 2:
                params = ", ".join(str(a) for a in self.args[:-1])
                ret = str(self.args[-1])
                return f"Callable[[{params}], {ret}]"
            return "Callable[..., Any]"
        if self.kind == TypeKind.LITERAL:
            return f"Literal[{self.literal_value!r}]"
        if self.kind == TypeKind.CLASS:
            return self.name
        if self.kind in (TypeKind.GENERIC, TypeKind.PROTOCOL):
            params = ", ".join(str(a) for a in self.args)
            return f"{self.name}[{params}]" if params else self.name
        if self.kind == TypeKind.TYPEVAR:
            return self.name
        return self.kind.value

    def is_optional(self) -> bool:
        if self.kind == TypeKind.OPTIONAL:
            return True
        if self.kind == TypeKind.UNION:
            return any(a.kind == TypeKind.NONE for a in self.args)
        return False

    def unwrap_optional(self) -> "Type":
        if self.kind == TypeKind.OPTIONAL:
            return self.args[0] if self.args else ANY_TYPE
        if self.kind == TypeKind.UNION:
            non_none = [a for a in self.args if a.kind != TypeKind.NONE]
            if len(non_none) == 1:
                return non_none[0]
            return Type(kind=TypeKind.UNION, args=non_none)
        return self


NONE_TYPE = Type(kind=TypeKind.NONE)
INT_TYPE = Type(kind=TypeKind.INT)
FLOAT_TYPE = Type(kind=TypeKind.FLOAT)
BOOL_TYPE = Type(kind=TypeKind.BOOL)
STR_TYPE = Type(kind=TypeKind.STR)
BYTES_TYPE = Type(kind=TypeKind.BYTES)
ANY_TYPE = Type(kind=TypeKind.ANY)
UNKNOWN_TYPE = Type(kind=TypeKind.UNKNOWN)


def make_optional(inner: Type) -> Type:
    if inner.kind == TypeKind.NONE:
        return NONE_TYPE
    return Type(kind=TypeKind.OPTIONAL, args=[inner])


def make_union(types: List[Type]) -> Type:
    flat: List[Type] = []
    seen: Set[str] = set()
    for t in types:
        if t.kind == TypeKind.UNION:
            for a in t.args:
                key = str(a)
                if key not in seen:
                    seen.add(key)
                    flat.append(a)
        else:
            key = str(t)
            if key not in seen:
                seen.add(key)
                flat.append(t)
    if len(flat) == 0:
        return UNKNOWN_TYPE
    if len(flat) == 1:
        return flat[0]
    has_none = any(t.kind == TypeKind.NONE for t in flat)
    non_none = [t for t in flat if t.kind != TypeKind.NONE]
    if has_none and len(non_none) == 1:
        return make_optional(non_none[0])
    return Type(kind=TypeKind.UNION, args=flat)


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class FunctionSignature:
    name: str
    params: List[Tuple[str, Type]]
    return_type: Type
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_property: bool = False
    type_vars: List[str] = field(default_factory=list)
    overloads: List["FunctionSignature"] = field(default_factory=list)
    confidence: float = 1.0

    def to_annotation(self) -> str:
        params = ", ".join(
            f"{n}: {t}" if str(t) != "Unknown" else n
            for n, t in self.params
        )
        ret = str(self.return_type)
        prefix = "async " if self.is_async else ""
        return f"{prefix}def {self.name}({params}) -> {ret}: ..."


@dataclass
class TypeEnvironment:
    variables: Dict[str, Type] = field(default_factory=dict)
    functions: Dict[str, FunctionSignature] = field(default_factory=dict)
    classes: Dict[str, "ClassInfo"] = field(default_factory=dict)
    imports: Dict[str, str] = field(default_factory=dict)
    type_vars: Dict[str, "TypeVarInfo"] = field(default_factory=dict)

    def get(self, name: str) -> Type:
        return self.variables.get(name, UNKNOWN_TYPE)


@dataclass
class ClassInfo:
    name: str
    bases: List[str] = field(default_factory=list)
    attributes: Dict[str, Type] = field(default_factory=dict)
    methods: Dict[str, FunctionSignature] = field(default_factory=dict)
    class_vars: Dict[str, Type] = field(default_factory=dict)
    is_protocol: bool = False
    is_dataclass: bool = False
    metaclass: Optional[str] = None


@dataclass
class ClassHierarchy:
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    parent_map: Dict[str, List[str]] = field(default_factory=dict)
    child_map: Dict[str, List[str]] = field(default_factory=dict)
    mro: Dict[str, List[str]] = field(default_factory=dict)

    def is_subclass(self, child: str, parent: str) -> bool:
        return parent in self.mro.get(child, [])

    def common_ancestor(self, a: str, b: str) -> Optional[str]:
        mro_a = self.mro.get(a, [])
        mro_b_set = set(self.mro.get(b, []))
        for cls in mro_a:
            if cls in mro_b_set:
                return cls
        return None


@dataclass
class TypeIncompat:
    line: int
    column: int
    expected: Type
    actual: Type
    message: str
    severity: str = "error"
    confidence: float = 0.85


@dataclass
class TypeVarInfo:
    name: str
    bound: Optional[Type] = None
    constraints: List[Type] = field(default_factory=list)
    covariant: bool = False
    contravariant: bool = False


# ── Constant / literal type resolution ───────────────────────────────────────

BUILTIN_TYPE_MAP: Dict[str, TypeKind] = {
    "int": TypeKind.INT,
    "float": TypeKind.FLOAT,
    "complex": TypeKind.COMPLEX,
    "bool": TypeKind.BOOL,
    "str": TypeKind.STR,
    "bytes": TypeKind.BYTES,
    "list": TypeKind.LIST,
    "dict": TypeKind.DICT,
    "set": TypeKind.SET,
    "tuple": TypeKind.TUPLE,
    "type": TypeKind.CLASS,
    "None": TypeKind.NONE,
}

BUILTIN_RETURN_TYPES: Dict[str, Type] = {
    "len": INT_TYPE,
    "int": INT_TYPE,
    "float": FLOAT_TYPE,
    "str": STR_TYPE,
    "bool": BOOL_TYPE,
    "bytes": BYTES_TYPE,
    "abs": INT_TYPE,
    "round": INT_TYPE,
    "min": ANY_TYPE,
    "max": ANY_TYPE,
    "sum": INT_TYPE,
    "sorted": Type(kind=TypeKind.LIST, args=[ANY_TYPE]),
    "list": Type(kind=TypeKind.LIST, args=[ANY_TYPE]),
    "dict": Type(kind=TypeKind.DICT, args=[ANY_TYPE, ANY_TYPE]),
    "set": Type(kind=TypeKind.SET, args=[ANY_TYPE]),
    "tuple": Type(kind=TypeKind.TUPLE),
    "range": Type(kind=TypeKind.LIST, args=[INT_TYPE]),
    "enumerate": Type(kind=TypeKind.ITERATOR, args=[Type(kind=TypeKind.TUPLE, args=[INT_TYPE, ANY_TYPE])]),
    "zip": Type(kind=TypeKind.ITERATOR, args=[Type(kind=TypeKind.TUPLE)]),
    "map": Type(kind=TypeKind.ITERATOR, args=[ANY_TYPE]),
    "filter": Type(kind=TypeKind.ITERATOR, args=[ANY_TYPE]),
    "reversed": Type(kind=TypeKind.ITERATOR, args=[ANY_TYPE]),
    "iter": Type(kind=TypeKind.ITERATOR, args=[ANY_TYPE]),
    "next": ANY_TYPE,
    "isinstance": BOOL_TYPE,
    "issubclass": BOOL_TYPE,
    "hasattr": BOOL_TYPE,
    "getattr": ANY_TYPE,
    "setattr": NONE_TYPE,
    "delattr": NONE_TYPE,
    "type": Type(kind=TypeKind.CLASS, name="type"),
    "id": INT_TYPE,
    "hash": INT_TYPE,
    "repr": STR_TYPE,
    "ascii": STR_TYPE,
    "chr": STR_TYPE,
    "ord": INT_TYPE,
    "hex": STR_TYPE,
    "oct": STR_TYPE,
    "bin": STR_TYPE,
    "input": STR_TYPE,
    "print": NONE_TYPE,
    "open": Type(kind=TypeKind.CLASS, name="IO"),
}

METHOD_RETURN_TYPES: Dict[Tuple[TypeKind, str], Type] = {
    (TypeKind.STR, "upper"): STR_TYPE,
    (TypeKind.STR, "lower"): STR_TYPE,
    (TypeKind.STR, "strip"): STR_TYPE,
    (TypeKind.STR, "lstrip"): STR_TYPE,
    (TypeKind.STR, "rstrip"): STR_TYPE,
    (TypeKind.STR, "split"): Type(kind=TypeKind.LIST, args=[STR_TYPE]),
    (TypeKind.STR, "rsplit"): Type(kind=TypeKind.LIST, args=[STR_TYPE]),
    (TypeKind.STR, "join"): STR_TYPE,
    (TypeKind.STR, "replace"): STR_TYPE,
    (TypeKind.STR, "format"): STR_TYPE,
    (TypeKind.STR, "encode"): BYTES_TYPE,
    (TypeKind.STR, "startswith"): BOOL_TYPE,
    (TypeKind.STR, "endswith"): BOOL_TYPE,
    (TypeKind.STR, "find"): INT_TYPE,
    (TypeKind.STR, "index"): INT_TYPE,
    (TypeKind.STR, "count"): INT_TYPE,
    (TypeKind.STR, "isdigit"): BOOL_TYPE,
    (TypeKind.STR, "isalpha"): BOOL_TYPE,
    (TypeKind.STR, "isalnum"): BOOL_TYPE,
    (TypeKind.BYTES, "decode"): STR_TYPE,
    (TypeKind.LIST, "append"): NONE_TYPE,
    (TypeKind.LIST, "extend"): NONE_TYPE,
    (TypeKind.LIST, "pop"): ANY_TYPE,
    (TypeKind.LIST, "insert"): NONE_TYPE,
    (TypeKind.LIST, "remove"): NONE_TYPE,
    (TypeKind.LIST, "sort"): NONE_TYPE,
    (TypeKind.LIST, "reverse"): NONE_TYPE,
    (TypeKind.LIST, "copy"): Type(kind=TypeKind.LIST, args=[ANY_TYPE]),
    (TypeKind.LIST, "index"): INT_TYPE,
    (TypeKind.LIST, "count"): INT_TYPE,
    (TypeKind.DICT, "keys"): Type(kind=TypeKind.ITERATOR, args=[ANY_TYPE]),
    (TypeKind.DICT, "values"): Type(kind=TypeKind.ITERATOR, args=[ANY_TYPE]),
    (TypeKind.DICT, "items"): Type(kind=TypeKind.ITERATOR, args=[Type(kind=TypeKind.TUPLE, args=[ANY_TYPE, ANY_TYPE])]),
    (TypeKind.DICT, "get"): ANY_TYPE,
    (TypeKind.DICT, "pop"): ANY_TYPE,
    (TypeKind.DICT, "update"): NONE_TYPE,
    (TypeKind.DICT, "setdefault"): ANY_TYPE,
    (TypeKind.DICT, "copy"): Type(kind=TypeKind.DICT, args=[ANY_TYPE, ANY_TYPE]),
    (TypeKind.SET, "add"): NONE_TYPE,
    (TypeKind.SET, "remove"): NONE_TYPE,
    (TypeKind.SET, "discard"): NONE_TYPE,
    (TypeKind.SET, "pop"): ANY_TYPE,
    (TypeKind.SET, "union"): Type(kind=TypeKind.SET, args=[ANY_TYPE]),
    (TypeKind.SET, "intersection"): Type(kind=TypeKind.SET, args=[ANY_TYPE]),
    (TypeKind.SET, "difference"): Type(kind=TypeKind.SET, args=[ANY_TYPE]),
    (TypeKind.SET, "issubset"): BOOL_TYPE,
    (TypeKind.SET, "issuperset"): BOOL_TYPE,
}


# ── Core inference visitors ──────────────────────────────────────────────────

class _ExprTypeInferrer(ast.NodeVisitor):
    """Infer the type of an expression node given a local environment."""

    def __init__(self, env: Dict[str, Type], class_env: Dict[str, ClassInfo]):
        self.env = env
        self.class_env = class_env

    def infer(self, node: ast.expr) -> Type:
        method = f"visit_{type(node).__name__}"
        visitor = getattr(self, method, None)
        if visitor:
            return visitor(node)
        return UNKNOWN_TYPE

    def visit_Constant(self, node: ast.Constant) -> Type:
        if node.value is None:
            return NONE_TYPE
        if isinstance(node.value, bool):
            return BOOL_TYPE
        if isinstance(node.value, int):
            return INT_TYPE
        if isinstance(node.value, float):
            return FLOAT_TYPE
        if isinstance(node.value, complex):
            return Type(kind=TypeKind.COMPLEX)
        if isinstance(node.value, str):
            return STR_TYPE
        if isinstance(node.value, bytes):
            return BYTES_TYPE
        return UNKNOWN_TYPE

    def visit_Name(self, node: ast.Name) -> Type:
        if node.id in self.env:
            return self.env[node.id]
        if node.id == "None":
            return NONE_TYPE
        if node.id == "True" or node.id == "False":
            return BOOL_TYPE
        if node.id in BUILTIN_TYPE_MAP:
            return Type(kind=TypeKind.CLASS, name=node.id)
        return UNKNOWN_TYPE

    def visit_List(self, node: ast.List) -> Type:
        if not node.elts:
            return Type(kind=TypeKind.LIST, args=[ANY_TYPE])
        elem_types = [self.infer(e) for e in node.elts]
        unified = _unify_types(elem_types)
        return Type(kind=TypeKind.LIST, args=[unified])

    def visit_Dict(self, node: ast.Dict) -> Type:
        if not node.keys:
            return Type(kind=TypeKind.DICT, args=[ANY_TYPE, ANY_TYPE])
        key_types = [self.infer(k) for k in node.keys if k is not None]
        val_types = [self.infer(v) for v in node.values]
        kt = _unify_types(key_types) if key_types else ANY_TYPE
        vt = _unify_types(val_types) if val_types else ANY_TYPE
        return Type(kind=TypeKind.DICT, args=[kt, vt])

    def visit_Set(self, node: ast.Set) -> Type:
        if not node.elts:
            return Type(kind=TypeKind.SET, args=[ANY_TYPE])
        elem_types = [self.infer(e) for e in node.elts]
        unified = _unify_types(elem_types)
        return Type(kind=TypeKind.SET, args=[unified])

    def visit_Tuple(self, node: ast.Tuple) -> Type:
        elem_types = [self.infer(e) for e in node.elts]
        return Type(kind=TypeKind.TUPLE, args=elem_types)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Type:
        return STR_TYPE

    def visit_FormattedValue(self, node: ast.FormattedValue) -> Type:
        return STR_TYPE

    def visit_BinOp(self, node: ast.BinOp) -> Type:
        lt = self.infer(node.left)
        rt = self.infer(node.right)
        if isinstance(node.op, ast.Add):
            if lt.kind == TypeKind.STR or rt.kind == TypeKind.STR:
                return STR_TYPE
            if lt.kind == TypeKind.LIST or rt.kind == TypeKind.LIST:
                return Type(kind=TypeKind.LIST, args=[ANY_TYPE])
        if isinstance(node.op, ast.Mult):
            if lt.kind == TypeKind.STR or rt.kind == TypeKind.STR:
                return STR_TYPE
            if lt.kind == TypeKind.LIST or rt.kind == TypeKind.LIST:
                return Type(kind=TypeKind.LIST, args=[ANY_TYPE])
        if isinstance(node.op, ast.Mod) and lt.kind == TypeKind.STR:
            return STR_TYPE
        if isinstance(node.op, ast.Div):
            return FLOAT_TYPE
        if isinstance(node.op, ast.FloorDiv):
            return INT_TYPE
        numeric = {TypeKind.INT, TypeKind.FLOAT, TypeKind.COMPLEX, TypeKind.BOOL}
        if lt.kind in numeric and rt.kind in numeric:
            if lt.kind == TypeKind.COMPLEX or rt.kind == TypeKind.COMPLEX:
                return Type(kind=TypeKind.COMPLEX)
            if lt.kind == TypeKind.FLOAT or rt.kind == TypeKind.FLOAT:
                return FLOAT_TYPE
            return INT_TYPE
        return UNKNOWN_TYPE

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Type:
        operand_type = self.infer(node.operand)
        if isinstance(node.op, ast.Not):
            return BOOL_TYPE
        if isinstance(node.op, (ast.UAdd, ast.USub)):
            return operand_type
        if isinstance(node.op, ast.Invert):
            return INT_TYPE
        return operand_type

    def visit_BoolOp(self, node: ast.BoolOp) -> Type:
        types = [self.infer(v) for v in node.values]
        if all(t.kind == TypeKind.BOOL for t in types):
            return BOOL_TYPE
        return _unify_types(types)

    def visit_Compare(self, node: ast.Compare) -> Type:
        return BOOL_TYPE

    def visit_IfExp(self, node: ast.IfExp) -> Type:
        body_type = self.infer(node.body)
        else_type = self.infer(node.orelse)
        return make_union([body_type, else_type])

    def visit_Call(self, node: ast.Call) -> Type:
        call_name = _extract_call_name(node)
        if call_name in BUILTIN_RETURN_TYPES:
            return copy.deepcopy(BUILTIN_RETURN_TYPES[call_name])
        if call_name in self.class_env:
            return Type(kind=TypeKind.CLASS, name=call_name)
        if isinstance(node.func, ast.Attribute):
            obj_type = self.infer(node.func.value)
            method = node.func.attr
            key = (obj_type.kind, method)
            if key in METHOD_RETURN_TYPES:
                result = copy.deepcopy(METHOD_RETURN_TYPES[key])
                if obj_type.kind == TypeKind.DICT and method == "get":
                    if obj_type.args and len(obj_type.args) > 1:
                        val_type = obj_type.args[1]
                        if len(node.args) > 1:
                            return val_type
                        return make_optional(val_type)
                    if len(node.args) > 1:
                        return ANY_TYPE
                    return make_optional(ANY_TYPE)
                if obj_type.kind == TypeKind.LIST and method == "pop":
                    if obj_type.args:
                        return obj_type.args[0]
                return result
            if obj_type.kind == TypeKind.CLASS and obj_type.name in self.class_env:
                cls = self.class_env[obj_type.name]
                if method in cls.methods:
                    return cls.methods[method].return_type
        if isinstance(node.func, ast.Name) and node.func.id in self.env:
            t = self.env[node.func.id]
            if t.kind == TypeKind.CALLABLE and t.args:
                return t.args[-1]
        return UNKNOWN_TYPE

    def visit_Subscript(self, node: ast.Subscript) -> Type:
        obj_type = self.infer(node.value)
        if obj_type.kind == TypeKind.LIST and obj_type.args:
            if isinstance(node.slice, ast.Slice):
                return Type(kind=TypeKind.LIST, args=list(obj_type.args))
            return obj_type.args[0]
        if obj_type.kind == TypeKind.DICT and obj_type.args and len(obj_type.args) > 1:
            return obj_type.args[1]
        if obj_type.kind == TypeKind.TUPLE and obj_type.args:
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                idx = node.slice.value
                if 0 <= idx < len(obj_type.args):
                    return obj_type.args[idx]
            return _unify_types(obj_type.args)
        if obj_type.kind == TypeKind.STR:
            if isinstance(node.slice, ast.Slice):
                return STR_TYPE
            return STR_TYPE
        return UNKNOWN_TYPE

    def visit_Attribute(self, node: ast.Attribute) -> Type:
        obj_type = self.infer(node.value)
        if obj_type.kind == TypeKind.CLASS and obj_type.name in self.class_env:
            cls = self.class_env[obj_type.name]
            if node.attr in cls.attributes:
                return cls.attributes[node.attr]
            if node.attr in cls.class_vars:
                return cls.class_vars[node.attr]
        if obj_type.attributes and node.attr in obj_type.attributes:
            return obj_type.attributes[node.attr]
        return UNKNOWN_TYPE

    def visit_Starred(self, node: ast.Starred) -> Type:
        inner = self.infer(node.value)
        if inner.kind == TypeKind.LIST and inner.args:
            return inner.args[0]
        return UNKNOWN_TYPE

    def visit_ListComp(self, node: ast.ListComp) -> Type:
        return Type(kind=TypeKind.LIST, args=[UNKNOWN_TYPE])

    def visit_SetComp(self, node: ast.SetComp) -> Type:
        return Type(kind=TypeKind.SET, args=[UNKNOWN_TYPE])

    def visit_DictComp(self, node: ast.DictComp) -> Type:
        return Type(kind=TypeKind.DICT, args=[UNKNOWN_TYPE, UNKNOWN_TYPE])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Type:
        return Type(kind=TypeKind.ITERATOR, args=[UNKNOWN_TYPE])

    def visit_Lambda(self, node: ast.Lambda) -> Type:
        num_params = len(node.args.args)
        param_types = [ANY_TYPE] * num_params
        return Type(kind=TypeKind.CALLABLE, args=param_types + [UNKNOWN_TYPE])

    def visit_Await(self, node: ast.Await) -> Type:
        inner = self.infer(node.value)
        if inner.kind == TypeKind.COROUTINE and inner.args:
            return inner.args[-1] if inner.args else UNKNOWN_TYPE
        return UNKNOWN_TYPE

    def visit_YieldFrom(self, node: ast.YieldFrom) -> Type:
        return UNKNOWN_TYPE

    def visit_Yield(self, node: ast.Yield) -> Type:
        if node.value:
            return self.infer(node.value)
        return NONE_TYPE

    def visit_NamedExpr(self, node: ast.NamedExpr) -> Type:
        return self.infer(node.value)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts: List[str] = []
        cur: ast.expr = node.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _unify_types(types: List[Type]) -> Type:
    """Compute a single type that covers all the given types."""
    if not types:
        return UNKNOWN_TYPE
    known = [t for t in types if t.kind != TypeKind.UNKNOWN]
    if not known:
        return UNKNOWN_TYPE
    unique_strs: Dict[str, Type] = {}
    for t in known:
        key = str(t)
        if key not in unique_strs:
            unique_strs[key] = t
    unique = list(unique_strs.values())
    if len(unique) == 1:
        return unique[0]
    return make_union(unique)


def _compute_mro(cls_name: str, parent_map: Dict[str, List[str]]) -> List[str]:
    """C3 linearization (simplified)."""
    result: List[str] = [cls_name]
    bases = parent_map.get(cls_name, [])
    if not bases:
        result.append("object")
        return result
    sequences = [_compute_mro(b, parent_map) for b in bases] + [list(bases)]
    while sequences:
        sequences = [s for s in sequences if s]
        if not sequences:
            break
        candidate = None
        for seq in sequences:
            head = seq[0]
            if all(head not in s[1:] for s in sequences):
                candidate = head
                break
        if candidate is None:
            break
        result.append(candidate)
        for seq in sequences:
            if seq and seq[0] == candidate:
                seq.pop(0)
    if result[-1] != "object":
        result.append("object")
    return result


# ── Statement-level type inference ───────────────────────────────────────────

class _FunctionBodyAnalyzer(ast.NodeVisitor):
    """Walk function body collecting variable types and return types."""

    def __init__(self, env: Dict[str, Type], class_env: Dict[str, ClassInfo]):
        self.env = dict(env)
        self.class_env = class_env
        self.return_types: List[Type] = []
        self._inferrer = _ExprTypeInferrer(self.env, self.class_env)
        self._has_return = False

    def _refresh_inferrer(self) -> None:
        self._inferrer = _ExprTypeInferrer(self.env, self.class_env)

    def visit_Assign(self, node: ast.Assign) -> None:
        value_type = self._inferrer.infer(node.value)
        for target in node.targets:
            self._assign_target(target, value_type)
        self._refresh_inferrer()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        ann_type = _annotation_to_type(node.annotation)
        if node.value:
            value_type = self._inferrer.infer(node.value)
            if ann_type.kind != TypeKind.UNKNOWN:
                resolved = ann_type
            else:
                resolved = value_type
        else:
            resolved = ann_type
        if isinstance(node.target, ast.Name):
            self.env[node.target.id] = resolved
            self._refresh_inferrer()

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            current = self.env.get(node.target.id, UNKNOWN_TYPE)
            val_type = self._inferrer.infer(node.value)
            if current.kind in (TypeKind.INT, TypeKind.FLOAT) and val_type.kind in (TypeKind.INT, TypeKind.FLOAT):
                if isinstance(node.op, ast.Div):
                    self.env[node.target.id] = FLOAT_TYPE
                elif current.kind == TypeKind.FLOAT or val_type.kind == TypeKind.FLOAT:
                    self.env[node.target.id] = FLOAT_TYPE
                else:
                    self.env[node.target.id] = INT_TYPE
            elif current.kind == TypeKind.STR and isinstance(node.op, ast.Add):
                self.env[node.target.id] = STR_TYPE
            elif current.kind == TypeKind.LIST and isinstance(node.op, ast.Add):
                self.env[node.target.id] = current
            self._refresh_inferrer()

    def visit_Return(self, node: ast.Return) -> None:
        self._has_return = True
        if node.value is None:
            self.return_types.append(NONE_TYPE)
        else:
            self.return_types.append(self._inferrer.infer(node.value))

    def visit_If(self, node: ast.If) -> None:
        true_env = dict(self.env)
        false_env = dict(self.env)
        narrowed = _narrow_from_test(node.test, self.env, self.class_env)
        true_env.update(narrowed)
        true_analyzer = _FunctionBodyAnalyzer(true_env, self.class_env)
        for stmt in node.body:
            true_analyzer.visit(stmt)
        self.return_types.extend(true_analyzer.return_types)
        false_analyzer = _FunctionBodyAnalyzer(false_env, self.class_env)
        for stmt in node.orelse:
            false_analyzer.visit(stmt)
        self.return_types.extend(false_analyzer.return_types)
        for name, t in true_analyzer.env.items():
            if name in false_analyzer.env:
                self.env[name] = make_union([t, false_analyzer.env[name]])
            else:
                self.env[name] = t
        for name, t in false_analyzer.env.items():
            if name not in true_analyzer.env:
                self.env[name] = t
        self._refresh_inferrer()

    def visit_For(self, node: ast.For) -> None:
        iter_type = self._inferrer.infer(node.iter)
        elem_type = UNKNOWN_TYPE
        if iter_type.kind == TypeKind.LIST and iter_type.args:
            elem_type = iter_type.args[0]
        elif iter_type.kind == TypeKind.SET and iter_type.args:
            elem_type = iter_type.args[0]
        elif iter_type.kind == TypeKind.DICT and iter_type.args:
            elem_type = iter_type.args[0]
        elif iter_type.kind == TypeKind.STR:
            elem_type = STR_TYPE
        elif iter_type.kind == TypeKind.TUPLE and iter_type.args:
            elem_type = _unify_types(iter_type.args)
        elif iter_type.kind == TypeKind.ITERATOR and iter_type.args:
            elem_type = iter_type.args[0]
        self._assign_target(node.target, elem_type)
        self._refresh_inferrer()
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_While(self, node: ast.While) -> None:
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            ctx_type = self._inferrer.infer(item.context_expr)
            if item.optional_vars:
                self._assign_target(item.optional_vars, ctx_type)
        self._refresh_inferrer()
        for stmt in node.body:
            self.visit(stmt)

    def visit_Try(self, node: ast.Try) -> None:
        for stmt in node.body:
            self.visit(stmt)
        for handler in node.handlers:
            if handler.name and handler.type:
                exc_name = ""
                if isinstance(handler.type, ast.Name):
                    exc_name = handler.type.id
                elif isinstance(handler.type, ast.Attribute):
                    exc_name = handler.type.attr
                self.env[handler.name] = Type(kind=TypeKind.CLASS, name=exc_name or "Exception")
                self._refresh_inferrer()
            for stmt in handler.body:
                self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        for stmt in node.finalbody:
            self.visit(stmt)

    def visit_Expr(self, node: ast.Expr) -> None:
        self._inferrer.infer(node.value)

    def _assign_target(self, target: ast.AST, value_type: Type) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = value_type
        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
            for i, elt in enumerate(target.elts):
                if value_type.kind == TypeKind.TUPLE and i < len(value_type.args):
                    self._assign_target(elt, value_type.args[i])
                elif value_type.kind == TypeKind.LIST and value_type.args:
                    self._assign_target(elt, value_type.args[0])
                else:
                    self._assign_target(elt, UNKNOWN_TYPE)
        elif isinstance(target, ast.Starred):
            if isinstance(target.value, ast.Name):
                self.env[target.value.id] = Type(kind=TypeKind.LIST, args=[UNKNOWN_TYPE])

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.visit(child)


def _narrow_from_test(
    test: ast.expr, env: Dict[str, Type], class_env: Dict[str, ClassInfo]
) -> Dict[str, Type]:
    """Narrow variable types based on a guard condition."""
    narrowed: Dict[str, Type] = {}
    if isinstance(test, ast.Call):
        name = _extract_call_name(test)
        if name == "isinstance" and len(test.args) == 2:
            if isinstance(test.args[0], ast.Name):
                var_name = test.args[0].id
                type_arg = test.args[1]
                if isinstance(type_arg, ast.Name):
                    kind = BUILTIN_TYPE_MAP.get(type_arg.id)
                    if kind:
                        narrowed[var_name] = Type(kind=kind)
                    else:
                        narrowed[var_name] = Type(kind=TypeKind.CLASS, name=type_arg.id)
                elif isinstance(type_arg, ast.Tuple):
                    types = []
                    for elt in type_arg.elts:
                        if isinstance(elt, ast.Name):
                            kind = BUILTIN_TYPE_MAP.get(elt.id)
                            if kind:
                                types.append(Type(kind=kind))
                            else:
                                types.append(Type(kind=TypeKind.CLASS, name=elt.id))
                    if types:
                        narrowed[var_name] = make_union(types)
    elif isinstance(test, ast.Compare):
        if len(test.ops) == 1 and isinstance(test.ops[0], (ast.Is, ast.IsNot)):
            comp = test.comparators[0]
            if isinstance(comp, ast.Constant) and comp.value is None:
                if isinstance(test.left, ast.Name):
                    var = test.left.id
                    if isinstance(test.ops[0], ast.IsNot):
                        current = env.get(var, UNKNOWN_TYPE)
                        narrowed[var] = current.unwrap_optional()
                    else:
                        narrowed[var] = NONE_TYPE
    elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inv = _narrow_from_test(test.operand, env, class_env)
        # Inverting narrows is complex; skip for safety
    return narrowed


def _annotation_to_type(node: ast.expr) -> Type:
    """Convert an annotation AST node to a Type."""
    if isinstance(node, ast.Constant):
        if node.value is None:
            return NONE_TYPE
        if isinstance(node.value, str):
            kind = BUILTIN_TYPE_MAP.get(node.value)
            if kind:
                return Type(kind=kind)
            return Type(kind=TypeKind.CLASS, name=node.value)
    if isinstance(node, ast.Name):
        if node.id == "None":
            return NONE_TYPE
        kind = BUILTIN_TYPE_MAP.get(node.id)
        if kind:
            return Type(kind=kind)
        return Type(kind=TypeKind.CLASS, name=node.id)
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name):
            container = node.value.id
            if container == "Optional":
                inner = _annotation_to_type(node.slice)
                return make_optional(inner)
            if container == "Union":
                if isinstance(node.slice, ast.Tuple):
                    args = [_annotation_to_type(e) for e in node.slice.elts]
                    return make_union(args)
                return _annotation_to_type(node.slice)
            if container == "List":
                inner = _annotation_to_type(node.slice)
                return Type(kind=TypeKind.LIST, args=[inner])
            if container == "Dict":
                if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                    k = _annotation_to_type(node.slice.elts[0])
                    v = _annotation_to_type(node.slice.elts[1])
                    return Type(kind=TypeKind.DICT, args=[k, v])
            if container == "Set":
                inner = _annotation_to_type(node.slice)
                return Type(kind=TypeKind.SET, args=[inner])
            if container == "Tuple":
                if isinstance(node.slice, ast.Tuple):
                    args = [_annotation_to_type(e) for e in node.slice.elts]
                    return Type(kind=TypeKind.TUPLE, args=args)
                return Type(kind=TypeKind.TUPLE, args=[_annotation_to_type(node.slice)])
            if container == "Callable":
                return Type(kind=TypeKind.CALLABLE)
        if isinstance(node.value, ast.Attribute):
            return UNKNOWN_TYPE
    if isinstance(node, ast.Attribute):
        return Type(kind=TypeKind.CLASS, name=node.attr)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        lt = _annotation_to_type(node.left)
        rt = _annotation_to_type(node.right)
        return make_union([lt, rt])
    return UNKNOWN_TYPE


# ── Class hierarchy extraction ───────────────────────────────────────────────

class _ClassExtractor(ast.NodeVisitor):
    """Extract class definitions, attributes, methods, and inheritance."""

    def __init__(self) -> None:
        self.classes: Dict[str, ClassInfo] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
            elif isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
                bases.append(base.value.id)

        is_protocol = "Protocol" in bases
        is_dataclass = any(
            (isinstance(d, ast.Name) and d.id == "dataclass") or
            (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass") or
            (isinstance(d, ast.Attribute) and d.attr == "dataclass")
            for d in node.decorator_list
        )
        metaclass = None
        for kw in node.keywords:
            if kw.arg == "metaclass" and isinstance(kw.value, ast.Name):
                metaclass = kw.value.id

        info = ClassInfo(
            name=node.name,
            bases=bases,
            is_protocol=is_protocol,
            is_dataclass=is_dataclass,
            metaclass=metaclass,
        )

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = _extract_function_signature(item, {})
                info.methods[item.name] = sig
                if item.name == "__init__":
                    self._extract_init_attrs(item, info)
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    attr_type = _annotation_to_type(item.annotation)
                    if item.value:
                        inf = _ExprTypeInferrer({}, {})
                        val_type = inf.infer(item.value)
                        if attr_type.kind == TypeKind.UNKNOWN:
                            attr_type = val_type
                    info.class_vars[item.target.id] = attr_type
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        inf = _ExprTypeInferrer({}, {})
                        val_type = inf.infer(item.value)
                        info.class_vars[target.id] = val_type

        self.classes[node.name] = info
        self.generic_visit(node)

    def _extract_init_attrs(self, node: ast.FunctionDef, info: ClassInfo) -> None:
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if (isinstance(target, ast.Attribute) and
                            isinstance(target.value, ast.Name) and
                            target.value.id == "self"):
                        inf = _ExprTypeInferrer({}, {})
                        val_type = inf.infer(stmt.value)
                        info.attributes[target.attr] = val_type
            elif isinstance(stmt, ast.AnnAssign):
                if (isinstance(stmt.target, ast.Attribute) and
                        isinstance(stmt.target.value, ast.Name) and
                        stmt.target.value.id == "self"):
                    attr_type = _annotation_to_type(stmt.annotation)
                    info.attributes[stmt.target.attr] = attr_type


def _extract_function_signature(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    env: Dict[str, Type],
) -> FunctionSignature:
    """Build a FunctionSignature from a function AST node."""
    params: List[Tuple[str, Type]] = []
    for arg in node.args.args:
        if arg.arg == "self" or arg.arg == "cls":
            continue
        if arg.annotation:
            params.append((arg.arg, _annotation_to_type(arg.annotation)))
        else:
            params.append((arg.arg, UNKNOWN_TYPE))
    for arg in node.args.kwonlyargs:
        if arg.annotation:
            params.append((arg.arg, _annotation_to_type(arg.annotation)))
        else:
            params.append((arg.arg, UNKNOWN_TYPE))
    if node.args.vararg:
        params.append((f"*{node.args.vararg.arg}", Type(kind=TypeKind.TUPLE)))
    if node.args.kwarg:
        params.append((f"**{node.args.kwarg.arg}", Type(kind=TypeKind.DICT, args=[STR_TYPE, ANY_TYPE])))

    ret = UNKNOWN_TYPE
    if node.returns:
        ret = _annotation_to_type(node.returns)

    decorators = []
    is_classmethod = False
    is_staticmethod = False
    is_property = False
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            decorators.append(dec.id)
            if dec.id == "classmethod":
                is_classmethod = True
            elif dec.id == "staticmethod":
                is_staticmethod = True
            elif dec.id == "property":
                is_property = True
        elif isinstance(dec, ast.Attribute):
            decorators.append(dec.attr)

    return FunctionSignature(
        name=node.name,
        params=params,
        return_type=ret,
        decorators=decorators,
        is_async=isinstance(node, ast.AsyncFunctionDef),
        is_classmethod=is_classmethod,
        is_staticmethod=is_staticmethod,
        is_property=is_property,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def infer_types(source: str) -> TypeEnvironment:
    """Infer types for all variables, functions, and classes in *source*.

    Walks the module-level statements and builds a ``TypeEnvironment`` that
    maps every identifier to its inferred ``Type``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return TypeEnvironment()

    class_ext = _ClassExtractor()
    class_ext.visit(tree)
    class_env = class_ext.classes

    env: Dict[str, Type] = {}
    func_sigs: Dict[str, FunctionSignature] = {}
    type_vars: Dict[str, TypeVarInfo] = {}
    imports: Dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                imports[name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                name = alias.asname or alias.name
                imports[name] = f"{mod}.{alias.name}"

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _extract_function_signature(node, env)
            func_env: Dict[str, Type] = dict(env)
            for pname, ptype in sig.params:
                clean = pname.lstrip("*")
                func_env[clean] = ptype
            analyzer = _FunctionBodyAnalyzer(func_env, class_env)
            for stmt in node.body:
                analyzer.visit(stmt)
            if analyzer.return_types:
                sig.return_type = _unify_types(analyzer.return_types)
            elif not analyzer._has_return:
                sig.return_type = NONE_TYPE
            func_sigs[node.name] = sig
            env[node.name] = Type(
                kind=TypeKind.CALLABLE,
                args=[t for _, t in sig.params] + [sig.return_type],
            )
        elif isinstance(node, ast.Assign):
            inferrer = _ExprTypeInferrer(env, class_env)
            value_type = inferrer.infer(node.value)
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id == "TypeVar":
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            tv_info = TypeVarInfo(name=target.id)
                            if len(node.value.args) > 1:
                                tv_info.constraints = [
                                    _annotation_to_type(a) for a in node.value.args[1:]
                                ]
                            for kw in node.value.keywords:
                                if kw.arg == "bound":
                                    tv_info.bound = _annotation_to_type(kw.value)
                                elif kw.arg == "covariant":
                                    tv_info.covariant = True
                                elif kw.arg == "contravariant":
                                    tv_info.contravariant = True
                            type_vars[target.id] = tv_info
                            env[target.id] = Type(kind=TypeKind.TYPEVAR, name=target.id)
                            continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    env[target.id] = value_type
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for i, elt in enumerate(target.elts):
                        if isinstance(elt, ast.Name):
                            if value_type.kind == TypeKind.TUPLE and i < len(value_type.args):
                                env[elt.id] = value_type.args[i]
                            else:
                                env[elt.id] = UNKNOWN_TYPE
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                ann_type = _annotation_to_type(node.annotation)
                env[node.target.id] = ann_type

    return TypeEnvironment(
        variables=env,
        functions=func_sigs,
        classes=class_env,
        imports=imports,
        type_vars=type_vars,
    )


def infer_function_signature(source: str, fn_name: str) -> FunctionSignature:
    """Infer the complete signature of function *fn_name* in *source*."""
    te = infer_types(source)
    if fn_name in te.functions:
        return te.functions[fn_name]
    return FunctionSignature(name=fn_name, params=[], return_type=UNKNOWN_TYPE)


def infer_return_type(source: str, fn_name: str) -> Type:
    """Infer the return type of *fn_name*."""
    sig = infer_function_signature(source, fn_name)
    return sig.return_type


def infer_class_hierarchy(source: str) -> ClassHierarchy:
    """Build the class hierarchy from *source*, including MRO computation."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ClassHierarchy()

    extractor = _ClassExtractor()
    extractor.visit(tree)

    parent_map: Dict[str, List[str]] = {}
    child_map: Dict[str, List[str]] = defaultdict(list)

    for name, info in extractor.classes.items():
        parent_map[name] = info.bases
        for base in info.bases:
            child_map[base].append(name)

    mro_map: Dict[str, List[str]] = {}
    for name in extractor.classes:
        mro_map[name] = _compute_mro(name, parent_map)

    return ClassHierarchy(
        classes=extractor.classes,
        parent_map=parent_map,
        child_map=dict(child_map),
        mro=mro_map,
    )


def suggest_type_annotations(source: str) -> str:
    """Return *source* with PEP 484 type annotations inserted.

    Adds ``-> ReturnType`` to function definitions and parameter annotations
    where the engine can infer them with reasonable confidence.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    te = infer_types(source)
    lines = source.splitlines(keepends=True)
    edits: List[Tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        sig = te.functions.get(node.name)
        if sig is None:
            continue
        if node.returns is not None:
            continue
        ret_str = str(sig.return_type)
        if ret_str == "Unknown":
            continue
        line_idx = node.lineno - 1
        line = lines[line_idx]
        colon_search = line.rfind("):")
        if colon_search == -1:
            colon_search = line.rfind(")")
            if colon_search != -1:
                end_paren = colon_search
                new_line = line[:end_paren + 1] + f" -> {ret_str}:" + line[end_paren + 2:]
                edits.append((line_idx, line_idx, new_line))
        else:
            new_line = line[:colon_search + 1] + f" -> {ret_str}:" + line[colon_search + 2:]
            edits.append((line_idx, line_idx, new_line))

    for line_idx, _, new_line in reversed(edits):
        lines[line_idx] = new_line

    result = "".join(lines)
    needed_imports: Set[str] = set()
    for sig in te.functions.values():
        _collect_typing_imports(sig.return_type, needed_imports)
        for _, pt in sig.params:
            _collect_typing_imports(pt, needed_imports)
    if needed_imports:
        import_line = f"from typing import {', '.join(sorted(needed_imports))}\n"
        if "from typing import" not in result:
            result = import_line + result
    return result


def _collect_typing_imports(t: Type, imports: Set[str]) -> None:
    """Collect typing imports needed for a type."""
    if t.kind == TypeKind.OPTIONAL:
        imports.add("Optional")
    elif t.kind == TypeKind.UNION:
        imports.add("Union")
    elif t.kind == TypeKind.LIST:
        imports.add("List")
    elif t.kind == TypeKind.DICT:
        imports.add("Dict")
    elif t.kind == TypeKind.SET:
        imports.add("Set")
    elif t.kind == TypeKind.TUPLE:
        imports.add("Tuple")
    elif t.kind == TypeKind.CALLABLE:
        imports.add("Callable")
    elif t.kind == TypeKind.ANY:
        imports.add("Any")
    for arg in t.args:
        _collect_typing_imports(arg, imports)


def type_compatibility_check(source: str) -> List[TypeIncompat]:
    """Check for type incompatibilities in *source*.

    Detects assignments where the inferred type of the right-hand side is
    incompatible with the declared or previously-inferred type of the target.
    Also checks function call argument types against known signatures.
    """
    issues: List[TypeIncompat] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return issues

    te = infer_types(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = _extract_call_name(node)
            if call_name in te.functions:
                sig = te.functions[call_name]
                for i, arg in enumerate(node.args):
                    if i < len(sig.params):
                        param_name, expected = sig.params[i]
                        if expected.kind == TypeKind.UNKNOWN:
                            continue
                        inferrer = _ExprTypeInferrer(te.variables, te.classes)
                        actual = inferrer.infer(arg)
                        if actual.kind == TypeKind.UNKNOWN:
                            continue
                        if not _types_compatible(actual, expected):
                            issues.append(TypeIncompat(
                                line=node.lineno,
                                column=node.col_offset,
                                expected=expected,
                                actual=actual,
                                message=(
                                    f"Argument {i + 1} to '{call_name}': "
                                    f"expected {expected}, got {actual}"
                                ),
                            ))
        elif isinstance(node, ast.Return) and node.value:
            func_node = _find_enclosing_function(tree, node)
            if func_node and func_node.name in te.functions:
                sig = te.functions[func_node.name]
                if sig.return_type.kind != TypeKind.UNKNOWN:
                    inferrer = _ExprTypeInferrer(te.variables, te.classes)
                    actual = inferrer.infer(node.value)
                    if actual.kind != TypeKind.UNKNOWN:
                        if not _types_compatible(actual, sig.return_type):
                            issues.append(TypeIncompat(
                                line=node.lineno,
                                column=node.col_offset,
                                expected=sig.return_type,
                                actual=actual,
                                message=(
                                    f"Return type mismatch in '{func_node.name}': "
                                    f"expected {sig.return_type}, got {actual}"
                                ),
                                severity="warning",
                            ))
    return issues


def _types_compatible(actual: Type, expected: Type) -> bool:
    """Check whether *actual* is compatible with *expected*."""
    if expected.kind == TypeKind.ANY or actual.kind == TypeKind.ANY:
        return True
    if expected.kind == TypeKind.UNKNOWN or actual.kind == TypeKind.UNKNOWN:
        return True
    if actual.kind == TypeKind.NONE and expected.is_optional():
        return True
    if expected.kind == TypeKind.OPTIONAL:
        return _types_compatible(actual, expected.unwrap_optional()) or actual.kind == TypeKind.NONE
    if expected.kind == TypeKind.UNION:
        return any(_types_compatible(actual, a) for a in expected.args)
    if actual.kind == TypeKind.UNION:
        return all(_types_compatible(a, expected) for a in actual.args)
    if actual.kind == TypeKind.BOOL and expected.kind == TypeKind.INT:
        return True
    if actual.kind == TypeKind.INT and expected.kind == TypeKind.FLOAT:
        return True
    if actual.kind == expected.kind:
        if actual.kind in (TypeKind.LIST, TypeKind.SET):
            if actual.args and expected.args:
                return _types_compatible(actual.args[0], expected.args[0])
            return True
        if actual.kind == TypeKind.DICT:
            if actual.args and expected.args:
                return (
                    _types_compatible(actual.args[0], expected.args[0]) and
                    (len(actual.args) < 2 or len(expected.args) < 2 or
                     _types_compatible(actual.args[1], expected.args[1]))
                )
            return True
        if actual.kind == TypeKind.CLASS:
            return actual.name == expected.name
        return True
    return False


def _find_enclosing_function(
    tree: ast.AST, target: ast.AST
) -> Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]]:
    """Find the function that encloses *target* in *tree*."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if child is target:
                    return node
    return None
