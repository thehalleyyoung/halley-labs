"""
Class hierarchy analyzer for a refinement type inference system.

Analyses Python ASTs to extract class definitions, compute MRO via C3
linearisation, resolve methods through the inheritance chain, check
protocol compliance (structural subtyping), and handle dataclasses,
enums, and named tuples.
"""

from __future__ import annotations

import ast
import copy
import itertools
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# ===================================================================
# Data classes
# ===================================================================

@dataclass
class ParamInfo:
    """Information about a single function/method parameter."""

    name: str
    annotation: Optional[str] = None
    default: Optional[str] = None
    kind: str = "positional"  # positional | keyword | var_positional | var_keyword

    def __repr__(self) -> str:
        parts = [self.name]
        if self.annotation:
            parts.append(f": {self.annotation}")
        if self.default:
            parts.append(f" = {self.default}")
        return "".join(parts)


@dataclass
class MethodInfo:
    """Information about a single method inside a class."""

    name: str
    params: List[ParamInfo] = field(default_factory=list)
    return_annotation: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    is_abstract: bool = False
    is_override: bool = False
    body_complexity: int = 1

    @property
    def is_classmethod(self) -> bool:
        return "classmethod" in self.decorators

    @property
    def is_staticmethod(self) -> bool:
        return "staticmethod" in self.decorators

    @property
    def is_property(self) -> bool:
        return "property" in self.decorators

    def __repr__(self) -> str:
        params_str = ", ".join(repr(p) for p in self.params)
        ret = f" -> {self.return_annotation}" if self.return_annotation else ""
        return f"MethodInfo({self.name}({params_str}){ret})"


@dataclass
class PropertyInfo:
    """Information about a @property definition."""

    name: str
    getter_return: Optional[str] = None
    setter_param: Optional[str] = None
    deleter: bool = False

    def __repr__(self) -> str:
        parts = [f"PropertyInfo({self.name}"]
        if self.getter_return:
            parts.append(f", getter -> {self.getter_return}")
        if self.setter_param:
            parts.append(f", setter({self.setter_param})")
        if self.deleter:
            parts.append(", deleter")
        parts.append(")")
        return "".join(parts)


@dataclass
class ClassInfo:
    """Complete information about a single class definition."""

    name: str
    bases: List[str] = field(default_factory=list)
    mro: List[str] = field(default_factory=list)
    methods: Dict[str, MethodInfo] = field(default_factory=dict)
    class_attrs: Dict[str, Optional[ast.expr]] = field(default_factory=dict)
    instance_attrs: Dict[str, Optional[ast.expr]] = field(default_factory=dict)
    properties: Dict[str, PropertyInfo] = field(default_factory=dict)
    class_methods: Set[str] = field(default_factory=set)
    static_methods: Set[str] = field(default_factory=set)
    slots: Optional[List[str]] = None
    is_abstract: bool = False
    is_dataclass: bool = False
    is_namedtuple: bool = False
    is_frozen: bool = False
    decorators: List[str] = field(default_factory=list)
    metaclass: Optional[str] = None
    type_params: List[str] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        bases_str = ", ".join(self.bases) if self.bases else ""
        return f"ClassInfo({self.name}({bases_str}))"


# ===================================================================
# Helper utilities
# ===================================================================

def _annotation_to_str(node: Optional[ast.expr]) -> Optional[str]:
    """Convert an AST annotation node to its string representation."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _decorator_name(node: ast.expr) -> str:
    """Extract the decorator name from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _decorator_name(node.value)
        return f"{value}.{node.attr}"
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    try:
        return ast.unparse(node)
    except Exception:
        return "<unknown>"


def _base_name(node: ast.expr) -> str:
    """Extract base class name from an AST expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _base_name(node.value)
        return f"{value}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    try:
        return ast.unparse(node)
    except Exception:
        return "<unknown>"


def _const_to_str(node: ast.expr) -> Optional[str]:
    """Convert an AST constant node to a string representation."""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    try:
        return ast.unparse(node)
    except Exception:
        return None


# ===================================================================
# Builtin method/param helpers
# ===================================================================

def _make_param(name: str, annotation: Optional[str] = None,
                default: Optional[str] = None,
                kind: str = "positional") -> ParamInfo:
    return ParamInfo(name=name, annotation=annotation, default=default, kind=kind)


def _make_method(name: str,
                 params: Optional[List[ParamInfo]] = None,
                 return_annotation: Optional[str] = None,
                 decorators: Optional[List[str]] = None,
                 is_abstract: bool = False) -> MethodInfo:
    return MethodInfo(
        name=name,
        params=params or [_make_param("self")],
        return_annotation=return_annotation,
        decorators=decorators or [],
        is_abstract=is_abstract,
        is_override=False,
        body_complexity=1,
    )


def _object_methods() -> Dict[str, MethodInfo]:
    """Methods defined on ``object``."""
    return {
        "__init__": _make_method("__init__", return_annotation="None"),
        "__new__": _make_method("__new__", [_make_param("cls")], return_annotation="object"),
        "__repr__": _make_method("__repr__", return_annotation="str"),
        "__str__": _make_method("__str__", return_annotation="str"),
        "__hash__": _make_method("__hash__", return_annotation="int"),
        "__eq__": _make_method("__eq__", [_make_param("self"), _make_param("other")], return_annotation="bool"),
        "__ne__": _make_method("__ne__", [_make_param("self"), _make_param("other")], return_annotation="bool"),
        "__bool__": _make_method("__bool__", return_annotation="bool"),
        "__getattribute__": _make_method("__getattribute__", [_make_param("self"), _make_param("name", "str")]),
        "__setattr__": _make_method("__setattr__", [_make_param("self"), _make_param("name", "str"), _make_param("value")], return_annotation="None"),
        "__delattr__": _make_method("__delattr__", [_make_param("self"), _make_param("name", "str")], return_annotation="None"),
        "__dir__": _make_method("__dir__", return_annotation="List[str]"),
        "__class__": _make_method("__class__", return_annotation="type"),
        "__sizeof__": _make_method("__sizeof__", return_annotation="int"),
        "__reduce__": _make_method("__reduce__"),
        "__reduce_ex__": _make_method("__reduce_ex__", [_make_param("self"), _make_param("protocol", "int")]),
        "__format__": _make_method("__format__", [_make_param("self"), _make_param("format_spec", "str")], return_annotation="str"),
        "__init_subclass__": _make_method("__init_subclass__", [_make_param("cls")], return_annotation="None"),
        "__subclasshook__": _make_method("__subclasshook__", [_make_param("cls"), _make_param("subclass")], return_annotation="bool"),
    }


def _numeric_methods(type_name: str) -> Dict[str, MethodInfo]:
    """Common numeric dunder methods for int / float / complex."""
    meths: Dict[str, MethodInfo] = {}
    binary_ops = [
        "__add__", "__radd__", "__sub__", "__rsub__",
        "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
        "__floordiv__", "__rfloordiv__", "__mod__", "__rmod__",
        "__pow__", "__rpow__",
    ]
    for op in binary_ops:
        meths[op] = _make_method(op, [_make_param("self"), _make_param("other")], return_annotation=type_name)

    unary_ops = ["__neg__", "__pos__", "__abs__"]
    for op in unary_ops:
        meths[op] = _make_method(op, return_annotation=type_name)

    comparison_ops = ["__lt__", "__le__", "__gt__", "__ge__"]
    for op in comparison_ops:
        meths[op] = _make_method(op, [_make_param("self"), _make_param("other")], return_annotation="bool")

    meths["__int__"] = _make_method("__int__", return_annotation="int")
    meths["__float__"] = _make_method("__float__", return_annotation="float")
    meths["__complex__"] = _make_method("__complex__", return_annotation="complex")
    meths["__round__"] = _make_method("__round__", [_make_param("self"), _make_param("ndigits", "int", "None")], return_annotation=type_name)
    return meths


def _int_extra_methods() -> Dict[str, MethodInfo]:
    """Methods specific to int (bitwise, etc.)."""
    meths: Dict[str, MethodInfo] = {}
    bitwise_ops = [
        "__and__", "__rand__", "__or__", "__ror__",
        "__xor__", "__rxor__", "__lshift__", "__rlshift__",
        "__rshift__", "__rrshift__",
    ]
    for op in bitwise_ops:
        meths[op] = _make_method(op, [_make_param("self"), _make_param("other")], return_annotation="int")
    meths["__invert__"] = _make_method("__invert__", return_annotation="int")
    meths["bit_length"] = _make_method("bit_length", return_annotation="int")
    meths["bit_count"] = _make_method("bit_count", return_annotation="int")
    meths["to_bytes"] = _make_method(
        "to_bytes",
        [_make_param("self"), _make_param("length", "int"), _make_param("byteorder", "str"),
         _make_param("signed", "bool", "False", kind="keyword")],
        return_annotation="bytes",
    )
    meths["from_bytes"] = _make_method(
        "from_bytes",
        [_make_param("cls"), _make_param("bytes_val", "bytes"), _make_param("byteorder", "str"),
         _make_param("signed", "bool", "False", kind="keyword")],
        return_annotation="int",
        decorators=["classmethod"],
    )
    meths["__index__"] = _make_method("__index__", return_annotation="int")
    meths["__trunc__"] = _make_method("__trunc__", return_annotation="int")
    meths["__floor__"] = _make_method("__floor__", return_annotation="int")
    meths["__ceil__"] = _make_method("__ceil__", return_annotation="int")
    meths["conjugate"] = _make_method("conjugate", return_annotation="int")
    meths["real"] = _make_method("real", return_annotation="int")
    meths["imag"] = _make_method("imag", return_annotation="int")
    meths["numerator"] = _make_method("numerator", return_annotation="int")
    meths["denominator"] = _make_method("denominator", return_annotation="int")
    return meths


def _str_methods() -> Dict[str, MethodInfo]:
    """Methods on str."""
    meths: Dict[str, MethodInfo] = {}
    simple_str_methods = [
        "capitalize", "casefold", "lower", "upper", "title", "swapcase",
        "strip", "lstrip", "rstrip", "expandtabs",
    ]
    for m in simple_str_methods:
        meths[m] = _make_method(m, return_annotation="str")

    meths["center"] = _make_method("center", [_make_param("self"), _make_param("width", "int"), _make_param("fillchar", "str", "' '")], return_annotation="str")
    meths["ljust"] = _make_method("ljust", [_make_param("self"), _make_param("width", "int"), _make_param("fillchar", "str", "' '")], return_annotation="str")
    meths["rjust"] = _make_method("rjust", [_make_param("self"), _make_param("width", "int"), _make_param("fillchar", "str", "' '")], return_annotation="str")
    meths["zfill"] = _make_method("zfill", [_make_param("self"), _make_param("width", "int")], return_annotation="str")

    meths["count"] = _make_method("count", [_make_param("self"), _make_param("sub", "str"), _make_param("start", "int", "0"), _make_param("end", "int", "None")], return_annotation="int")
    meths["find"] = _make_method("find", [_make_param("self"), _make_param("sub", "str"), _make_param("start", "int", "0"), _make_param("end", "int", "None")], return_annotation="int")
    meths["rfind"] = _make_method("rfind", [_make_param("self"), _make_param("sub", "str"), _make_param("start", "int", "0"), _make_param("end", "int", "None")], return_annotation="int")
    meths["index"] = _make_method("index", [_make_param("self"), _make_param("sub", "str"), _make_param("start", "int", "0"), _make_param("end", "int", "None")], return_annotation="int")
    meths["rindex"] = _make_method("rindex", [_make_param("self"), _make_param("sub", "str"), _make_param("start", "int", "0"), _make_param("end", "int", "None")], return_annotation="int")

    meths["replace"] = _make_method("replace", [_make_param("self"), _make_param("old", "str"), _make_param("new", "str"), _make_param("count", "int", "-1")], return_annotation="str")
    meths["split"] = _make_method("split", [_make_param("self"), _make_param("sep", "Optional[str]", "None"), _make_param("maxsplit", "int", "-1")], return_annotation="List[str]")
    meths["rsplit"] = _make_method("rsplit", [_make_param("self"), _make_param("sep", "Optional[str]", "None"), _make_param("maxsplit", "int", "-1")], return_annotation="List[str]")
    meths["splitlines"] = _make_method("splitlines", [_make_param("self"), _make_param("keepends", "bool", "False")], return_annotation="List[str]")
    meths["join"] = _make_method("join", [_make_param("self"), _make_param("iterable")], return_annotation="str")
    meths["partition"] = _make_method("partition", [_make_param("self"), _make_param("sep", "str")], return_annotation="Tuple[str, str, str]")
    meths["rpartition"] = _make_method("rpartition", [_make_param("self"), _make_param("sep", "str")], return_annotation="Tuple[str, str, str]")

    bool_methods = [
        "startswith", "endswith", "isalpha", "isalnum", "isdigit",
        "isdecimal", "isnumeric", "isspace", "istitle", "isupper",
        "islower", "isidentifier", "isprintable", "isascii",
    ]
    for m in bool_methods:
        if m in ("startswith", "endswith"):
            meths[m] = _make_method(m, [_make_param("self"), _make_param("prefix", "str"), _make_param("start", "int", "0"), _make_param("end", "int", "None")], return_annotation="bool")
        else:
            meths[m] = _make_method(m, return_annotation="bool")

    meths["encode"] = _make_method("encode", [_make_param("self"), _make_param("encoding", "str", "'utf-8'"), _make_param("errors", "str", "'strict'")], return_annotation="bytes")
    meths["format"] = _make_method("format", [_make_param("self"), _make_param("args", kind="var_positional"), _make_param("kwargs", kind="var_keyword")])
    meths["format_map"] = _make_method("format_map", [_make_param("self"), _make_param("mapping")], return_annotation="str")
    meths["maketrans"] = _make_method("maketrans", [_make_param("self"), _make_param("x"), _make_param("y", default="None"), _make_param("z", default="None")], return_annotation="dict", decorators=["staticmethod"])
    meths["translate"] = _make_method("translate", [_make_param("self"), _make_param("table")], return_annotation="str")
    meths["removeprefix"] = _make_method("removeprefix", [_make_param("self"), _make_param("prefix", "str")], return_annotation="str")
    meths["removesuffix"] = _make_method("removesuffix", [_make_param("self"), _make_param("suffix", "str")], return_annotation="str")

    # dunders
    meths["__len__"] = _make_method("__len__", return_annotation="int")
    meths["__contains__"] = _make_method("__contains__", [_make_param("self"), _make_param("item", "str")], return_annotation="bool")
    meths["__getitem__"] = _make_method("__getitem__", [_make_param("self"), _make_param("key")], return_annotation="str")
    meths["__iter__"] = _make_method("__iter__")
    meths["__add__"] = _make_method("__add__", [_make_param("self"), _make_param("other", "str")], return_annotation="str")
    meths["__mul__"] = _make_method("__mul__", [_make_param("self"), _make_param("n", "int")], return_annotation="str")
    meths["__rmul__"] = _make_method("__rmul__", [_make_param("self"), _make_param("n", "int")], return_annotation="str")
    meths["__mod__"] = _make_method("__mod__", [_make_param("self"), _make_param("args")], return_annotation="str")
    meths["__lt__"] = _make_method("__lt__", [_make_param("self"), _make_param("other", "str")], return_annotation="bool")
    meths["__le__"] = _make_method("__le__", [_make_param("self"), _make_param("other", "str")], return_annotation="bool")
    meths["__gt__"] = _make_method("__gt__", [_make_param("self"), _make_param("other", "str")], return_annotation="bool")
    meths["__ge__"] = _make_method("__ge__", [_make_param("self"), _make_param("other", "str")], return_annotation="bool")
    return meths


def _sequence_methods(elem_type: str = "Any") -> Dict[str, MethodInfo]:
    """Common sequence dunder methods (list / tuple)."""
    return {
        "__len__": _make_method("__len__", return_annotation="int"),
        "__contains__": _make_method("__contains__", [_make_param("self"), _make_param("item")], return_annotation="bool"),
        "__getitem__": _make_method("__getitem__", [_make_param("self"), _make_param("key")], return_annotation=elem_type),
        "__iter__": _make_method("__iter__"),
        "__reversed__": _make_method("__reversed__"),
        "__add__": _make_method("__add__", [_make_param("self"), _make_param("other")]),
        "__mul__": _make_method("__mul__", [_make_param("self"), _make_param("n", "int")]),
        "__rmul__": _make_method("__rmul__", [_make_param("self"), _make_param("n", "int")]),
        "index": _make_method("index", [_make_param("self"), _make_param("value"), _make_param("start", "int", "0"), _make_param("end", "int", "None")], return_annotation="int"),
        "count": _make_method("count", [_make_param("self"), _make_param("value")], return_annotation="int"),
    }


def _mapping_methods() -> Dict[str, MethodInfo]:
    """Methods on dict."""
    return {
        "__len__": _make_method("__len__", return_annotation="int"),
        "__contains__": _make_method("__contains__", [_make_param("self"), _make_param("key")], return_annotation="bool"),
        "__getitem__": _make_method("__getitem__", [_make_param("self"), _make_param("key")]),
        "__setitem__": _make_method("__setitem__", [_make_param("self"), _make_param("key"), _make_param("value")], return_annotation="None"),
        "__delitem__": _make_method("__delitem__", [_make_param("self"), _make_param("key")], return_annotation="None"),
        "__iter__": _make_method("__iter__"),
        "keys": _make_method("keys"),
        "values": _make_method("values"),
        "items": _make_method("items"),
        "get": _make_method("get", [_make_param("self"), _make_param("key"), _make_param("default", default="None")]),
        "pop": _make_method("pop", [_make_param("self"), _make_param("key"), _make_param("default", kind="positional")]),
        "setdefault": _make_method("setdefault", [_make_param("self"), _make_param("key"), _make_param("default", default="None")]),
        "update": _make_method("update", [_make_param("self"), _make_param("other", default="None"), _make_param("kwargs", kind="var_keyword")], return_annotation="None"),
        "clear": _make_method("clear", return_annotation="None"),
        "copy": _make_method("copy", return_annotation="dict"),
        "fromkeys": _make_method("fromkeys", [_make_param("cls"), _make_param("iterable"), _make_param("value", default="None")], decorators=["classmethod"]),
        "popitem": _make_method("popitem"),
    }


def _set_methods() -> Dict[str, MethodInfo]:
    """Methods on set."""
    return {
        "__len__": _make_method("__len__", return_annotation="int"),
        "__contains__": _make_method("__contains__", [_make_param("self"), _make_param("item")], return_annotation="bool"),
        "__iter__": _make_method("__iter__"),
        "add": _make_method("add", [_make_param("self"), _make_param("elem")], return_annotation="None"),
        "remove": _make_method("remove", [_make_param("self"), _make_param("elem")], return_annotation="None"),
        "discard": _make_method("discard", [_make_param("self"), _make_param("elem")], return_annotation="None"),
        "pop": _make_method("pop"),
        "clear": _make_method("clear", return_annotation="None"),
        "copy": _make_method("copy", return_annotation="set"),
        "union": _make_method("union", [_make_param("self"), _make_param("others", kind="var_positional")], return_annotation="set"),
        "intersection": _make_method("intersection", [_make_param("self"), _make_param("others", kind="var_positional")], return_annotation="set"),
        "difference": _make_method("difference", [_make_param("self"), _make_param("others", kind="var_positional")], return_annotation="set"),
        "symmetric_difference": _make_method("symmetric_difference", [_make_param("self"), _make_param("other")], return_annotation="set"),
        "issubset": _make_method("issubset", [_make_param("self"), _make_param("other")], return_annotation="bool"),
        "issuperset": _make_method("issuperset", [_make_param("self"), _make_param("other")], return_annotation="bool"),
        "isdisjoint": _make_method("isdisjoint", [_make_param("self"), _make_param("other")], return_annotation="bool"),
        "update": _make_method("update", [_make_param("self"), _make_param("others", kind="var_positional")], return_annotation="None"),
        "__and__": _make_method("__and__", [_make_param("self"), _make_param("other")], return_annotation="set"),
        "__or__": _make_method("__or__", [_make_param("self"), _make_param("other")], return_annotation="set"),
        "__sub__": _make_method("__sub__", [_make_param("self"), _make_param("other")], return_annotation="set"),
        "__xor__": _make_method("__xor__", [_make_param("self"), _make_param("other")], return_annotation="set"),
    }


def _exception_methods() -> Dict[str, MethodInfo]:
    """Methods common to Exception classes."""
    return {
        "__init__": _make_method("__init__", [_make_param("self"), _make_param("args", kind="var_positional")], return_annotation="None"),
        "__str__": _make_method("__str__", return_annotation="str"),
        "__repr__": _make_method("__repr__", return_annotation="str"),
        "with_traceback": _make_method("with_traceback", [_make_param("self"), _make_param("tb")]),
    }


# ===================================================================
# ClassHierarchyAnalyzer
# ===================================================================

class ClassHierarchyAnalyzer:
    """Analyses Python source ASTs to build a class hierarchy.

    Provides:
    - Class extraction from AST (methods, attributes, properties, …)
    - C3 linearisation for MRO computation
    - Method resolution across the inheritance chain
    - Abstract-method detection
    - Protocol-compliance checking (structural subtyping)
    - Dataclass / NamedTuple / Enum awareness
    """

    def __init__(self) -> None:
        self.classes: Dict[str, ClassInfo] = {}
        self.builtin_classes: Dict[str, ClassInfo] = self._init_builtins()

    # ---------------------------------------------------------------
    # Builtin class definitions
    # ---------------------------------------------------------------

    def _init_builtins(self) -> Dict[str, ClassInfo]:
        """Build a dictionary of ClassInfo for common Python builtins."""
        builtins: Dict[str, ClassInfo] = {}

        # -- object --
        builtins["object"] = ClassInfo(
            name="object",
            bases=[],
            mro=["object"],
            methods=_object_methods(),
        )

        # -- int --
        int_methods = {**_numeric_methods("int"), **_int_extra_methods()}
        builtins["int"] = ClassInfo(
            name="int", bases=["object"], mro=["int", "object"],
            methods=int_methods,
        )

        # -- float --
        float_methods = _numeric_methods("float")
        float_methods["is_integer"] = _make_method("is_integer", return_annotation="bool")
        float_methods["as_integer_ratio"] = _make_method("as_integer_ratio", return_annotation="Tuple[int, int]")
        float_methods["hex"] = _make_method("hex", return_annotation="str")
        float_methods["fromhex"] = _make_method("fromhex", [_make_param("cls"), _make_param("s", "str")], return_annotation="float", decorators=["classmethod"])
        float_methods["conjugate"] = _make_method("conjugate", return_annotation="float")
        builtins["float"] = ClassInfo(
            name="float", bases=["object"], mro=["float", "object"],
            methods=float_methods,
        )

        # -- complex --
        complex_methods = _numeric_methods("complex")
        complex_methods["conjugate"] = _make_method("conjugate", return_annotation="complex")
        complex_methods["real"] = _make_method("real", return_annotation="float")
        complex_methods["imag"] = _make_method("imag", return_annotation="float")
        builtins["complex"] = ClassInfo(
            name="complex", bases=["object"], mro=["complex", "object"],
            methods=complex_methods,
        )

        # -- bool (inherits from int) --
        builtins["bool"] = ClassInfo(
            name="bool", bases=["int"], mro=["bool", "int", "object"],
            methods={
                "__and__": _make_method("__and__", [_make_param("self"), _make_param("other")], return_annotation="bool"),
                "__or__": _make_method("__or__", [_make_param("self"), _make_param("other")], return_annotation="bool"),
                "__xor__": _make_method("__xor__", [_make_param("self"), _make_param("other")], return_annotation="bool"),
                "__invert__": _make_method("__invert__", return_annotation="int"),
            },
        )

        # -- str --
        builtins["str"] = ClassInfo(
            name="str", bases=["object"], mro=["str", "object"],
            methods=_str_methods(),
        )

        # -- bytes --
        bytes_methods: Dict[str, MethodInfo] = {
            "__len__": _make_method("__len__", return_annotation="int"),
            "__contains__": _make_method("__contains__", [_make_param("self"), _make_param("item")], return_annotation="bool"),
            "__getitem__": _make_method("__getitem__", [_make_param("self"), _make_param("key")]),
            "__iter__": _make_method("__iter__"),
            "__add__": _make_method("__add__", [_make_param("self"), _make_param("other")], return_annotation="bytes"),
            "decode": _make_method("decode", [_make_param("self"), _make_param("encoding", "str", "'utf-8'"), _make_param("errors", "str", "'strict'")], return_annotation="str"),
            "hex": _make_method("hex", return_annotation="str"),
            "count": _make_method("count", [_make_param("self"), _make_param("sub")], return_annotation="int"),
            "find": _make_method("find", [_make_param("self"), _make_param("sub")], return_annotation="int"),
            "replace": _make_method("replace", [_make_param("self"), _make_param("old"), _make_param("new")], return_annotation="bytes"),
            "split": _make_method("split", [_make_param("self"), _make_param("sep", default="None")], return_annotation="List[bytes]"),
            "join": _make_method("join", [_make_param("self"), _make_param("iterable")], return_annotation="bytes"),
            "strip": _make_method("strip", [_make_param("self"), _make_param("chars", default="None")], return_annotation="bytes"),
            "startswith": _make_method("startswith", [_make_param("self"), _make_param("prefix")], return_annotation="bool"),
            "endswith": _make_method("endswith", [_make_param("self"), _make_param("suffix")], return_annotation="bool"),
            "upper": _make_method("upper", return_annotation="bytes"),
            "lower": _make_method("lower", return_annotation="bytes"),
        }
        builtins["bytes"] = ClassInfo(
            name="bytes", bases=["object"], mro=["bytes", "object"],
            methods=bytes_methods,
        )

        # -- bytearray --
        bytearray_methods = dict(bytes_methods)
        bytearray_methods["append"] = _make_method("append", [_make_param("self"), _make_param("item", "int")], return_annotation="None")
        bytearray_methods["extend"] = _make_method("extend", [_make_param("self"), _make_param("iterable")], return_annotation="None")
        bytearray_methods["insert"] = _make_method("insert", [_make_param("self"), _make_param("index", "int"), _make_param("item", "int")], return_annotation="None")
        bytearray_methods["pop"] = _make_method("pop", [_make_param("self"), _make_param("index", "int", "-1")], return_annotation="int")
        bytearray_methods["remove"] = _make_method("remove", [_make_param("self"), _make_param("value", "int")], return_annotation="None")
        bytearray_methods["reverse"] = _make_method("reverse", return_annotation="None")
        bytearray_methods["clear"] = _make_method("clear", return_annotation="None")
        bytearray_methods["copy"] = _make_method("copy", return_annotation="bytearray")
        builtins["bytearray"] = ClassInfo(
            name="bytearray", bases=["object"], mro=["bytearray", "object"],
            methods=bytearray_methods,
        )

        # -- list --
        list_methods = dict(_sequence_methods())
        list_methods["append"] = _make_method("append", [_make_param("self"), _make_param("item")], return_annotation="None")
        list_methods["extend"] = _make_method("extend", [_make_param("self"), _make_param("iterable")], return_annotation="None")
        list_methods["insert"] = _make_method("insert", [_make_param("self"), _make_param("index", "int"), _make_param("item")], return_annotation="None")
        list_methods["remove"] = _make_method("remove", [_make_param("self"), _make_param("value")], return_annotation="None")
        list_methods["pop"] = _make_method("pop", [_make_param("self"), _make_param("index", "int", "-1")])
        list_methods["clear"] = _make_method("clear", return_annotation="None")
        list_methods["sort"] = _make_method("sort", [_make_param("self"), _make_param("key", default="None", kind="keyword"), _make_param("reverse", "bool", "False", kind="keyword")], return_annotation="None")
        list_methods["reverse"] = _make_method("reverse", return_annotation="None")
        list_methods["copy"] = _make_method("copy", return_annotation="list")
        list_methods["__setitem__"] = _make_method("__setitem__", [_make_param("self"), _make_param("key"), _make_param("value")], return_annotation="None")
        list_methods["__delitem__"] = _make_method("__delitem__", [_make_param("self"), _make_param("key")], return_annotation="None")
        list_methods["__iadd__"] = _make_method("__iadd__", [_make_param("self"), _make_param("other")], return_annotation="list")
        list_methods["__imul__"] = _make_method("__imul__", [_make_param("self"), _make_param("n", "int")], return_annotation="list")
        builtins["list"] = ClassInfo(
            name="list", bases=["object"], mro=["list", "object"],
            methods=list_methods,
        )

        # -- tuple --
        tuple_methods = _sequence_methods()
        tuple_methods["__hash__"] = _make_method("__hash__", return_annotation="int")
        builtins["tuple"] = ClassInfo(
            name="tuple", bases=["object"], mro=["tuple", "object"],
            methods=tuple_methods,
        )

        # -- dict --
        builtins["dict"] = ClassInfo(
            name="dict", bases=["object"], mro=["dict", "object"],
            methods=_mapping_methods(),
        )

        # -- set --
        builtins["set"] = ClassInfo(
            name="set", bases=["object"], mro=["set", "object"],
            methods=_set_methods(),
        )

        # -- frozenset --
        fs_methods = dict(_set_methods())
        # frozenset is immutable – remove mutating methods
        for m in ("add", "remove", "discard", "pop", "clear", "update"):
            fs_methods.pop(m, None)
        fs_methods["__hash__"] = _make_method("__hash__", return_annotation="int")
        builtins["frozenset"] = ClassInfo(
            name="frozenset", bases=["object"], mro=["frozenset", "object"],
            methods=fs_methods,
        )

        # -- range --
        builtins["range"] = ClassInfo(
            name="range", bases=["object"], mro=["range", "object"],
            methods={
                "__len__": _make_method("__len__", return_annotation="int"),
                "__contains__": _make_method("__contains__", [_make_param("self"), _make_param("item")], return_annotation="bool"),
                "__getitem__": _make_method("__getitem__", [_make_param("self"), _make_param("key")], return_annotation="int"),
                "__iter__": _make_method("__iter__"),
                "__reversed__": _make_method("__reversed__"),
                "start": _make_method("start", return_annotation="int"),
                "stop": _make_method("stop", return_annotation="int"),
                "step": _make_method("step", return_annotation="int"),
                "index": _make_method("index", [_make_param("self"), _make_param("value")], return_annotation="int"),
                "count": _make_method("count", [_make_param("self"), _make_param("value")], return_annotation="int"),
            },
        )

        # -- type --
        builtins["type"] = ClassInfo(
            name="type", bases=["object"], mro=["type", "object"],
            methods={
                "__call__": _make_method("__call__", [_make_param("self"), _make_param("args", kind="var_positional"), _make_param("kwargs", kind="var_keyword")]),
                "__instancecheck__": _make_method("__instancecheck__", [_make_param("self"), _make_param("instance")], return_annotation="bool"),
                "__subclasscheck__": _make_method("__subclasscheck__", [_make_param("self"), _make_param("subclass")], return_annotation="bool"),
                "mro": _make_method("mro", return_annotation="List[type]"),
                "__name__": _make_method("__name__", return_annotation="str"),
                "__bases__": _make_method("__bases__"),
                "__mro__": _make_method("__mro__"),
                "__subclasses__": _make_method("__subclasses__", return_annotation="list"),
            },
        )

        # -- NoneType --
        builtins["NoneType"] = ClassInfo(
            name="NoneType", bases=["object"], mro=["NoneType", "object"],
            methods={
                "__bool__": _make_method("__bool__", return_annotation="bool"),
                "__repr__": _make_method("__repr__", return_annotation="str"),
            },
        )

        # -- slice --
        builtins["slice"] = ClassInfo(
            name="slice", bases=["object"], mro=["slice", "object"],
            methods={
                "indices": _make_method("indices", [_make_param("self"), _make_param("length", "int")], return_annotation="Tuple[int, int, int]"),
                "start": _make_method("start"),
                "stop": _make_method("stop"),
                "step": _make_method("step"),
            },
        )

        # -- memoryview --
        builtins["memoryview"] = ClassInfo(
            name="memoryview", bases=["object"], mro=["memoryview", "object"],
            methods={
                "__len__": _make_method("__len__", return_annotation="int"),
                "__getitem__": _make_method("__getitem__", [_make_param("self"), _make_param("key")]),
                "tobytes": _make_method("tobytes", return_annotation="bytes"),
                "tolist": _make_method("tolist", return_annotation="list"),
                "release": _make_method("release", return_annotation="None"),
                "cast": _make_method("cast", [_make_param("self"), _make_param("format", "str")], return_annotation="memoryview"),
                "hex": _make_method("hex", return_annotation="str"),
            },
        )

        # -- property, staticmethod, classmethod, super --
        builtins["property"] = ClassInfo(
            name="property", bases=["object"], mro=["property", "object"],
            methods={
                "__init__": _make_method("__init__", [
                    _make_param("self"), _make_param("fget", default="None"),
                    _make_param("fset", default="None"), _make_param("fdel", default="None"),
                    _make_param("doc", default="None"),
                ], return_annotation="None"),
                "getter": _make_method("getter", [_make_param("self"), _make_param("fget")]),
                "setter": _make_method("setter", [_make_param("self"), _make_param("fset")]),
                "deleter": _make_method("deleter", [_make_param("self"), _make_param("fdel")]),
                "__get__": _make_method("__get__", [_make_param("self"), _make_param("obj"), _make_param("objtype", default="None")]),
                "__set__": _make_method("__set__", [_make_param("self"), _make_param("obj"), _make_param("value")], return_annotation="None"),
                "__delete__": _make_method("__delete__", [_make_param("self"), _make_param("obj")], return_annotation="None"),
            },
        )

        builtins["staticmethod"] = ClassInfo(
            name="staticmethod", bases=["object"], mro=["staticmethod", "object"],
            methods={
                "__init__": _make_method("__init__", [_make_param("self"), _make_param("function")], return_annotation="None"),
                "__get__": _make_method("__get__", [_make_param("self"), _make_param("obj"), _make_param("objtype", default="None")]),
            },
        )

        builtins["classmethod"] = ClassInfo(
            name="classmethod", bases=["object"], mro=["classmethod", "object"],
            methods={
                "__init__": _make_method("__init__", [_make_param("self"), _make_param("function")], return_annotation="None"),
                "__get__": _make_method("__get__", [_make_param("self"), _make_param("obj"), _make_param("objtype", default="None")]),
            },
        )

        builtins["super"] = ClassInfo(
            name="super", bases=["object"], mro=["super", "object"],
            methods={
                "__init__": _make_method("__init__", [_make_param("self"), _make_param("type_", default="None"), _make_param("obj", default="None")], return_annotation="None"),
                "__getattr__": _make_method("__getattr__", [_make_param("self"), _make_param("name", "str")]),
            },
        )

        # -- enumerate, zip, map, filter, reversed --
        for iter_cls in ("enumerate", "zip", "map", "filter", "reversed"):
            builtins[iter_cls] = ClassInfo(
                name=iter_cls, bases=["object"], mro=[iter_cls, "object"],
                methods={
                    "__init__": _make_method("__init__", [_make_param("self"), _make_param("args", kind="var_positional")], return_annotation="None"),
                    "__iter__": _make_method("__iter__"),
                    "__next__": _make_method("__next__"),
                },
            )

        # -- Exception hierarchy --
        builtins["BaseException"] = ClassInfo(
            name="BaseException", bases=["object"], mro=["BaseException", "object"],
            methods=_exception_methods(),
        )
        builtins["Exception"] = ClassInfo(
            name="Exception", bases=["BaseException"], mro=["Exception", "BaseException", "object"],
            methods=_exception_methods(),
        )

        exception_subclasses = [
            ("TypeError", ["Exception"]),
            ("ValueError", ["Exception"]),
            ("KeyError", ["LookupError"]),
            ("IndexError", ["LookupError"]),
            ("AttributeError", ["Exception"]),
            ("StopIteration", ["Exception"]),
            ("RuntimeError", ["Exception"]),
            ("NotImplementedError", ["RuntimeError"]),
            ("LookupError", ["Exception"]),
            ("ArithmeticError", ["Exception"]),
            ("ZeroDivisionError", ["ArithmeticError"]),
            ("OverflowError", ["ArithmeticError"]),
            ("OSError", ["Exception"]),
            ("FileNotFoundError", ["OSError"]),
            ("PermissionError", ["OSError"]),
            ("FileExistsError", ["OSError"]),
            ("IsADirectoryError", ["OSError"]),
            ("NotADirectoryError", ["OSError"]),
            ("TimeoutError", ["OSError"]),
            ("ConnectionError", ["OSError"]),
            ("ImportError", ["Exception"]),
            ("ModuleNotFoundError", ["ImportError"]),
            ("NameError", ["Exception"]),
            ("UnboundLocalError", ["NameError"]),
            ("SyntaxError", ["Exception"]),
            ("IndentationError", ["SyntaxError"]),
            ("TabError", ["IndentationError"]),
            ("SystemError", ["Exception"]),
            ("UnicodeError", ["ValueError"]),
            ("UnicodeDecodeError", ["UnicodeError"]),
            ("UnicodeEncodeError", ["UnicodeError"]),
            ("RecursionError", ["RuntimeError"]),
            ("StopAsyncIteration", ["Exception"]),
            ("GeneratorExit", ["BaseException"]),
            ("KeyboardInterrupt", ["BaseException"]),
            ("SystemExit", ["BaseException"]),
            ("AssertionError", ["Exception"]),
            ("BufferError", ["Exception"]),
            ("EOFError", ["Exception"]),
            ("MemoryError", ["Exception"]),
            ("Warning", ["Exception"]),
            ("DeprecationWarning", ["Warning"]),
            ("UserWarning", ["Warning"]),
            ("FutureWarning", ["Warning"]),
            ("RuntimeWarning", ["Warning"]),
        ]

        for exc_name, exc_bases in exception_subclasses:
            mro = [exc_name]
            # Walk the chain to build an approximate MRO for single inheritance
            current = exc_bases[0]
            while current in builtins:
                mro.append(current)
                parent_info = builtins[current]
                if parent_info.bases:
                    current = parent_info.bases[0]
                else:
                    break
            if "object" not in mro:
                mro.append("object")

            builtins[exc_name] = ClassInfo(
                name=exc_name,
                bases=exc_bases,
                mro=mro,
                methods=_exception_methods(),
            )

        return builtins

    # ---------------------------------------------------------------
    # Module-level analysis
    # ---------------------------------------------------------------

    def analyze_module(self, tree: ast.Module) -> None:
        """Walk an AST module and extract all class definitions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                info = self._analyze_class(node)
                self.classes[info.name] = info

        # Compute MROs after all classes have been collected
        for name in self.classes:
            try:
                self.classes[name].mro = self.get_mro(name)
            except ValueError:
                # If MRO computation fails (cycles, etc.), fall back to
                # simple linearisation: [class] + bases + [object]
                self.classes[name].mro = [name] + self.classes[name].bases
                if "object" not in self.classes[name].mro:
                    self.classes[name].mro.append("object")

    # ---------------------------------------------------------------
    # Class-level analysis
    # ---------------------------------------------------------------

    def _analyze_class(self, node: ast.ClassDef) -> ClassInfo:
        """Extract a complete ClassInfo from a class AST node."""
        name = node.name
        bases = self._extract_bases(node)
        methods = self._extract_methods(node)
        class_attrs = self._extract_class_attrs(node)
        instance_attrs = self._extract_instance_attrs(node)
        properties = self._extract_properties(node)
        slots = self._extract_slots(node)
        metaclass = self._detect_metaclass(node)
        type_params = self._detect_type_params(node)
        protocols = self._detect_protocols(node)
        decorators = [_decorator_name(d) for d in node.decorator_list]

        class_methods_set: Set[str] = set()
        static_methods_set: Set[str] = set()
        for mname, minfo in methods.items():
            if "classmethod" in minfo.decorators:
                class_methods_set.add(mname)
            if "staticmethod" in minfo.decorators:
                static_methods_set.add(mname)

        is_abstract = any(
            _decorator_name(d) in ("ABCMeta", "ABC", "abstractmethod")
            for d in node.decorator_list
        ) or metaclass in ("ABCMeta", "abc.ABCMeta")
        # Also mark abstract if any method is abstract
        if not is_abstract:
            is_abstract = any(m.is_abstract for m in methods.values())

        is_dc = self._is_dataclass(node)
        is_nt = self._is_namedtuple(node)
        is_frozen = self._is_frozen_dataclass(node)

        info = ClassInfo(
            name=name,
            bases=bases,
            mro=[],  # computed after all classes collected
            methods=methods,
            class_attrs=class_attrs,
            instance_attrs=instance_attrs,
            properties=properties,
            class_methods=class_methods_set,
            static_methods=static_methods_set,
            slots=slots,
            is_abstract=is_abstract,
            is_dataclass=is_dc,
            is_namedtuple=is_nt,
            is_frozen=is_frozen,
            decorators=decorators,
            metaclass=metaclass,
            type_params=type_params,
            protocols=protocols,
        )

        # Generate synthetic methods for dataclasses
        if is_dc:
            dc_fields = self._analyze_dataclass_fields(node)
            info.instance_attrs.update(dc_fields)
            generated = self._generate_dataclass_methods(info)
            for gname, gmethod in generated.items():
                if gname not in info.methods:
                    info.methods[gname] = gmethod

        # Detect enum members
        if any(b in ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag",
                      "enum.Enum", "enum.IntEnum", "enum.StrEnum")
               for b in bases):
            enum_members = self._analyze_enum_members(node)
            info.class_attrs.update(enum_members)

        return info

    # ---------------------------------------------------------------
    # Base class extraction
    # ---------------------------------------------------------------

    def _extract_bases(self, node: ast.ClassDef) -> List[str]:
        """Get the list of base class names from a class definition."""
        bases: List[str] = []
        for base in node.bases:
            name = _base_name(base)
            # Filter out Generic[...] and Protocol – they go elsewhere
            if name not in ("Generic", "Protocol", "typing.Generic",
                            "typing.Protocol", "typing_extensions.Protocol"):
                bases.append(name)
        # Every class implicitly inherits from object if no bases given
        if not bases:
            bases = ["object"]
        return bases

    # ---------------------------------------------------------------
    # Method extraction
    # ---------------------------------------------------------------

    def _extract_methods(self, node: ast.ClassDef) -> Dict[str, MethodInfo]:
        """Extract all method definitions from a class body."""
        methods: Dict[str, MethodInfo] = {}
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                minfo = self._analyze_method(item)
                methods[minfo.name] = minfo
        return methods

    def _analyze_method(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> MethodInfo:
        """Analyse a single method definition."""
        decorators = [_decorator_name(d) for d in node.decorator_list]
        params = self._extract_params(node)
        return_ann = _annotation_to_str(node.returns)

        is_abstract = any(
            d in ("abstractmethod", "abc.abstractmethod",
                  "abstractclassmethod", "abstractstaticmethod",
                  "abstractproperty")
            for d in decorators
        )

        # Check for @override decorator
        is_override = any(
            d in ("override", "typing.override", "typing_extensions.override")
            for d in decorators
        )

        complexity = self._compute_body_complexity(node.body)

        return MethodInfo(
            name=node.name,
            params=params,
            return_annotation=return_ann,
            decorators=decorators,
            is_abstract=is_abstract,
            is_override=is_override,
            body_complexity=complexity,
        )

    def _extract_params(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[ParamInfo]:
        """Extract parameter information from a function definition."""
        params: List[ParamInfo] = []
        args = node.args

        # Positional-only, regular positional, keyword-only
        all_positional = args.posonlyargs + args.args
        # Defaults align to the end of the positional args
        num_defaults = len(args.defaults)
        num_pos = len(all_positional)

        for i, arg in enumerate(all_positional):
            ann = _annotation_to_str(arg.annotation)
            default: Optional[str] = None
            default_idx = i - (num_pos - num_defaults)
            if default_idx >= 0 and default_idx < len(args.defaults):
                default = _const_to_str(args.defaults[default_idx])
            params.append(ParamInfo(
                name=arg.arg, annotation=ann, default=default,
                kind="positional",
            ))

        if args.vararg:
            ann = _annotation_to_str(args.vararg.annotation)
            params.append(ParamInfo(
                name=args.vararg.arg, annotation=ann, kind="var_positional",
            ))

        for i, arg in enumerate(args.kwonlyargs):
            ann = _annotation_to_str(arg.annotation)
            default = None
            if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
                default = _const_to_str(args.kw_defaults[i])
            params.append(ParamInfo(
                name=arg.arg, annotation=ann, default=default,
                kind="keyword",
            ))

        if args.kwarg:
            ann = _annotation_to_str(args.kwarg.annotation)
            params.append(ParamInfo(
                name=args.kwarg.arg, annotation=ann, kind="var_keyword",
            ))

        return params

    # ---------------------------------------------------------------
    # Attribute extraction
    # ---------------------------------------------------------------

    def _extract_instance_attrs(self, node: ast.ClassDef) -> Dict[str, Optional[ast.expr]]:
        """Find ``self.x = ...`` assignments in ``__init__`` and other methods."""
        attrs: Dict[str, Optional[ast.expr]] = {}
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Look in __init__ first, but also scan other methods
            self_name = "self"
            if item.args.args:
                self_name = item.args.args[0].arg

            for stmt in ast.walk(item):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if (isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == self_name):
                            attrs[target.attr] = stmt.value
                elif isinstance(stmt, ast.AnnAssign):
                    target = stmt.target
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == self_name):
                        attrs[target.attr] = stmt.value

        return attrs

    def _extract_class_attrs(self, node: ast.ClassDef) -> Dict[str, Optional[ast.expr]]:
        """Find class-level attribute assignments (not inside methods)."""
        attrs: Dict[str, Optional[ast.expr]] = {}
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attrs[target.id] = item.value
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                attrs[elt.id] = item.value
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    attrs[item.target.id] = item.value
        return attrs

    # ---------------------------------------------------------------
    # Property extraction
    # ---------------------------------------------------------------

    def _extract_properties(self, node: ast.ClassDef) -> Dict[str, PropertyInfo]:
        """Find @property, @x.setter, and @x.deleter methods."""
        props: Dict[str, PropertyInfo] = {}

        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = [_decorator_name(d) for d in item.decorator_list]

            if "property" in decorators:
                ret = _annotation_to_str(item.returns)
                props[item.name] = PropertyInfo(
                    name=item.name,
                    getter_return=ret,
                )
            else:
                # Check for @name.setter or @name.deleter
                for dec in decorators:
                    if dec.endswith(".setter"):
                        prop_name = dec.rsplit(".setter", 1)[0]
                        if prop_name not in props:
                            props[prop_name] = PropertyInfo(name=prop_name)
                        # Extract setter parameter type
                        if len(item.args.args) >= 2:
                            setter_ann = _annotation_to_str(item.args.args[1].annotation)
                            props[prop_name].setter_param = setter_ann
                    elif dec.endswith(".deleter"):
                        prop_name = dec.rsplit(".deleter", 1)[0]
                        if prop_name not in props:
                            props[prop_name] = PropertyInfo(name=prop_name)
                        props[prop_name].deleter = True

        return props

    # ---------------------------------------------------------------
    # Slots extraction
    # ---------------------------------------------------------------

    def _extract_slots(self, node: ast.ClassDef) -> Optional[List[str]]:
        """Find ``__slots__`` in the class body."""
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == "__slots__":
                        return self._parse_slots_value(item.value)
            elif isinstance(item, ast.AnnAssign):
                if (isinstance(item.target, ast.Name)
                        and item.target.id == "__slots__"
                        and item.value is not None):
                    return self._parse_slots_value(item.value)
        return None

    @staticmethod
    def _parse_slots_value(node: ast.expr) -> List[str]:
        """Parse the value of ``__slots__ = ...``."""
        slots: List[str] = []
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    slots.append(elt.value)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    slots.append(key.value)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            slots.append(node.value)
        return slots

    # ---------------------------------------------------------------
    # Metaclass & type parameter detection
    # ---------------------------------------------------------------

    def _detect_metaclass(self, node: ast.ClassDef) -> Optional[str]:
        """Find ``metaclass=X`` in the class bases keywords."""
        for keyword in node.keywords:
            if keyword.arg == "metaclass":
                return _base_name(keyword.value)
        return None

    def _detect_type_params(self, node: ast.ClassDef) -> List[str]:
        """Find Generic[T, U] in base classes and extract type parameters."""
        type_params: List[str] = []
        for base in node.bases:
            if isinstance(base, ast.Subscript):
                base_name = _base_name(base.value) if isinstance(base.value, (ast.Name, ast.Attribute)) else ""
                if base_name in ("Generic", "typing.Generic"):
                    if isinstance(base.slice, ast.Tuple):
                        for elt in base.slice.elts:
                            if isinstance(elt, ast.Name):
                                type_params.append(elt.id)
                    elif isinstance(base.slice, ast.Name):
                        type_params.append(base.slice.id)
        # Also check for PEP 695 type_params attribute (Python 3.12+)
        if hasattr(node, "type_params"):
            for tp in getattr(node, "type_params", []):
                if hasattr(tp, "name"):
                    name = tp.name
                    if name not in type_params:
                        type_params.append(name)
        return type_params

    def _detect_protocols(self, node: ast.ClassDef) -> List[str]:
        """Find Protocol in base classes."""
        protocols: List[str] = []
        for base in node.bases:
            name = _base_name(base)
            if name in ("Protocol", "typing.Protocol", "typing_extensions.Protocol"):
                protocols.append(name)
        return protocols

    # ---------------------------------------------------------------
    # Special class patterns
    # ---------------------------------------------------------------

    def _is_dataclass(self, node: ast.ClassDef) -> bool:
        """Check whether the class has a @dataclass decorator."""
        for dec in node.decorator_list:
            name = _decorator_name(dec)
            if name in ("dataclass", "dataclasses.dataclass"):
                return True
        return False

    def _is_namedtuple(self, node: ast.ClassDef) -> bool:
        """Check whether the class inherits from NamedTuple."""
        for base in node.bases:
            name = _base_name(base)
            if name in ("NamedTuple", "typing.NamedTuple"):
                return True
        # Also check for assignment-style:  X = namedtuple("X", ...)
        return False

    def _is_frozen_dataclass(self, node: ast.ClassDef) -> bool:
        """Check for ``@dataclass(frozen=True)``."""
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call):
                name = _decorator_name(dec.func)
                if name in ("dataclass", "dataclasses.dataclass"):
                    for kw in dec.keywords:
                        if (kw.arg == "frozen"
                                and isinstance(kw.value, ast.Constant)
                                and kw.value.value is True):
                            return True
        return False

    # ---------------------------------------------------------------
    # Complexity
    # ---------------------------------------------------------------

    def _compute_body_complexity(self, body: List[ast.stmt]) -> int:
        """Compute a simple cyclomatic-complexity-like metric for a body.

        Each branching construct (if/elif/for/while/try/except/with/
        assert/raise/boolean-op) adds 1 to the base complexity of 1.
        """
        complexity = 1
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, (ast.If, ast.IfExp)):
                complexity += 1
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, (ast.While,)):
                complexity += 1
            elif isinstance(node, (ast.ExceptHandler,)):
                complexity += 1
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # Each additional boolean operand adds a path
                complexity += len(node.values) - 1
            elif isinstance(node, ast.Assert):
                complexity += 1
            elif isinstance(node, (ast.Try,)):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
            elif isinstance(node, ast.Lambda):
                complexity += 1
        return complexity

    # ---------------------------------------------------------------
    # MRO – C3 linearisation
    # ---------------------------------------------------------------

    def get_mro(self, class_name: str) -> List[str]:
        """Compute the MRO for *class_name* using C3 linearisation.

        Raises ``ValueError`` on unresolvable hierarchies (cycles,
        inconsistent orderings, etc.).
        """
        return self._c3_linearize(class_name, set())

    def _c3_linearize(self, class_name: str, visited: Set[str]) -> List[str]:
        """Recursively compute C3 linearisation with cycle detection."""
        if class_name in visited:
            raise ValueError(
                f"Cycle detected in class hierarchy while computing MRO "
                f"for '{class_name}'. Visited so far: {visited}"
            )

        info = self.get_class_info(class_name)
        if info is None:
            # Unknown class – treat as having no bases (like object)
            return [class_name]

        bases = info.bases
        if not bases:
            return [class_name]

        visited_copy = visited | {class_name}

        # Linearisations of each base
        linearisations: List[List[str]] = []
        for base in bases:
            lin = self._c3_linearize(base, visited_copy)
            linearisations.append(lin)

        # Plus the list of direct bases themselves (in order)
        linearisations.append(list(bases))

        try:
            merged = self._c3_merge(linearisations)
        except ValueError as exc:
            raise ValueError(
                f"Cannot compute a consistent MRO for class '{class_name}': "
                f"{exc}"
            ) from exc

        return [class_name] + merged

    @staticmethod
    def _c3_merge(sequences: List[List[str]]) -> List[str]:
        """Perform the merge step of C3 linearisation.

        Algorithm:
        1. Take the head of the first non-empty sequence.
        2. If that head does not appear in the tail of *any* other
           sequence, output it and remove it from every sequence.
        3. Otherwise, try the head of the next sequence.
        4. If no head is valid, the hierarchy is inconsistent.
        """
        result: List[str] = []
        # Work on copies so we don't mutate the originals
        seqs = [list(s) for s in sequences]

        while True:
            # Remove empty sequences
            seqs = [s for s in seqs if s]
            if not seqs:
                break

            # Find a good head
            candidate: Optional[str] = None
            for seq in seqs:
                head = seq[0]
                # Check whether head appears in the tail of any sequence
                in_tail = any(head in s[1:] for s in seqs)
                if not in_tail:
                    candidate = head
                    break

            if candidate is None:
                remaining = [s[0] for s in seqs]
                raise ValueError(
                    f"Inconsistent method resolution order. "
                    f"Remaining heads: {remaining}. "
                    f"No valid candidate found – the class hierarchy "
                    f"contains a conflict that prevents C3 linearisation."
                )

            result.append(candidate)
            # Remove candidate from all sequences
            for seq in seqs:
                if seq and seq[0] == candidate:
                    seq.pop(0)

        return result

    # ---------------------------------------------------------------
    # Subclass / method resolution
    # ---------------------------------------------------------------

    def is_subclass(self, child: str, parent: str) -> bool:
        """Check whether *child* is a subclass of *parent* via MRO."""
        if child == parent:
            return True
        try:
            mro = self.get_mro(child)
        except ValueError:
            return False
        return parent in mro

    def resolve_method(self, class_name: str, method_name: str) -> Optional[MethodInfo]:
        """Resolve a method by walking the MRO."""
        try:
            mro = self.get_mro(class_name)
        except ValueError:
            mro = [class_name]

        for cls_name in mro:
            info = self.get_class_info(cls_name)
            if info is not None and method_name in info.methods:
                return info.methods[method_name]
        return None

    def get_abstract_methods(self, class_name: str) -> Set[str]:
        """Return the set of abstract methods that are *not* concretely
        implemented in *class_name* or any of its bases.
        """
        try:
            mro = self.get_mro(class_name)
        except ValueError:
            return set()

        abstract: Set[str] = set()
        concrete: Set[str] = set()

        # Walk MRO from most-base to most-derived
        for cls_name in reversed(mro):
            info = self.get_class_info(cls_name)
            if info is None:
                continue
            for mname, minfo in info.methods.items():
                if minfo.is_abstract:
                    abstract.add(mname)
                else:
                    concrete.add(mname)

        return abstract - concrete

    def get_all_methods(self, class_name: str) -> Dict[str, MethodInfo]:
        """Return all methods available on *class_name* through its MRO.

        If the same method name appears in multiple classes, the version
        from the earliest class in the MRO (most derived) wins.
        """
        try:
            mro = self.get_mro(class_name)
        except ValueError:
            mro = [class_name]

        all_methods: Dict[str, MethodInfo] = {}
        for cls_name in reversed(mro):
            info = self.get_class_info(cls_name)
            if info is not None:
                all_methods.update(info.methods)
        return all_methods

    def get_all_attributes(self, class_name: str) -> Dict[str, str]:
        """Return all attributes available on *class_name*, mapping each
        attribute name to the class that defines it.

        Instance attributes shadow class attributes; more-derived classes
        shadow base classes.
        """
        try:
            mro = self.get_mro(class_name)
        except ValueError:
            mro = [class_name]

        attrs: Dict[str, str] = {}
        for cls_name in reversed(mro):
            info = self.get_class_info(cls_name)
            if info is None:
                continue
            for attr_name in info.class_attrs:
                attrs[attr_name] = cls_name
            for attr_name in info.instance_attrs:
                attrs[attr_name] = cls_name
            # Properties also count as attributes
            for prop_name in info.properties:
                attrs[prop_name] = cls_name
        return attrs

    # ---------------------------------------------------------------
    # Protocol compliance (structural subtyping)
    # ---------------------------------------------------------------

    def check_protocol_compliance(self, class_name: str,
                                  protocol_name: str) -> List[str]:
        """Check whether *class_name* structurally conforms to
        *protocol_name*.

        Returns a list of human-readable violation descriptions.
        An empty list means the class is compliant.
        """
        violations: List[str] = []

        proto_info = self.get_class_info(protocol_name)
        if proto_info is None:
            violations.append(
                f"Protocol '{protocol_name}' is not defined or not found."
            )
            return violations

        class_info = self.get_class_info(class_name)
        if class_info is None:
            violations.append(
                f"Class '{class_name}' is not defined or not found."
            )
            return violations

        all_class_methods = self.get_all_methods(class_name)
        all_class_attrs = self.get_all_attributes(class_name)

        # Check methods
        for mname, minfo in proto_info.methods.items():
            # Skip dunder methods inherited from object unless protocol
            # explicitly defines them
            if mname.startswith("__") and mname.endswith("__"):
                if mname in ("__init__", "__new__", "__repr__", "__str__",
                             "__hash__", "__eq__", "__ne__", "__bool__",
                             "__getattribute__", "__setattr__", "__delattr__",
                             "__dir__", "__sizeof__", "__format__",
                             "__init_subclass__", "__subclasshook__",
                             "__class__", "__reduce__", "__reduce_ex__"):
                    continue

            if mname not in all_class_methods:
                violations.append(
                    f"Missing method '{mname}' required by protocol "
                    f"'{protocol_name}'."
                )
                continue

            # Check parameter count (rough check – ignoring *args/**kwargs)
            proto_params = [p for p in minfo.params
                           if p.kind not in ("var_positional", "var_keyword")]
            class_params = [p for p in all_class_methods[mname].params
                           if p.kind not in ("var_positional", "var_keyword")]

            # Allow the class to have more parameters (with defaults)
            required_proto = [p for p in proto_params if p.default is None]
            required_class = [p for p in class_params if p.default is None]

            if len(required_class) > len(required_proto):
                violations.append(
                    f"Method '{mname}' in '{class_name}' requires more "
                    f"arguments ({len(required_class)}) than protocol "
                    f"'{protocol_name}' specifies ({len(required_proto)})."
                )

        # Check attributes (instance + class)
        for attr_name in proto_info.instance_attrs:
            if attr_name not in all_class_attrs and attr_name not in all_class_methods:
                violations.append(
                    f"Missing attribute '{attr_name}' required by protocol "
                    f"'{protocol_name}'."
                )

        for attr_name in proto_info.class_attrs:
            if attr_name not in all_class_attrs and attr_name not in all_class_methods:
                violations.append(
                    f"Missing class attribute '{attr_name}' required by "
                    f"protocol '{protocol_name}'."
                )

        # Check properties
        for prop_name, prop_info in proto_info.properties.items():
            if prop_name not in all_class_attrs and prop_name not in all_class_methods:
                violations.append(
                    f"Missing property '{prop_name}' required by protocol "
                    f"'{protocol_name}'."
                )

        return violations

    # ---------------------------------------------------------------
    # Dataclass helpers
    # ---------------------------------------------------------------

    def _analyze_dataclass_fields(self, node: ast.ClassDef) -> Dict[str, Optional[ast.expr]]:
        """Extract fields from a dataclass (annotated assignments in body)."""
        fields: Dict[str, Optional[ast.expr]] = {}
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                # Skip ClassVar annotations
                ann_str = _annotation_to_str(item.annotation) or ""
                if ann_str.startswith("ClassVar") or ann_str.startswith("typing.ClassVar"):
                    continue
                fields[item.target.id] = item.value
        return fields

    def _generate_dataclass_methods(self, info: ClassInfo) -> Dict[str, MethodInfo]:
        """Generate synthetic __init__, __eq__, __repr__, and optionally
        __hash__ for a dataclass.
        """
        generated: Dict[str, MethodInfo] = {}

        # __init__
        init_params = [_make_param("self")]
        for attr_name, default_val in info.instance_attrs.items():
            default_str = _const_to_str(default_val) if default_val is not None else None
            init_params.append(ParamInfo(
                name=attr_name,
                annotation=None,
                default=default_str,
                kind="positional",
            ))
        generated["__init__"] = MethodInfo(
            name="__init__",
            params=init_params,
            return_annotation="None",
            decorators=[],
            is_abstract=False,
            is_override=False,
            body_complexity=1,
        )

        # __repr__
        generated["__repr__"] = _make_method("__repr__", return_annotation="str")

        # __eq__
        generated["__eq__"] = _make_method(
            "__eq__",
            [_make_param("self"), _make_param("other")],
            return_annotation="bool",
        )

        # __hash__ – generated only if frozen or eq=False
        if info.is_frozen:
            generated["__hash__"] = _make_method("__hash__", return_annotation="int")
        else:
            # By default dataclass with eq=True sets __hash__ = None
            generated["__hash__"] = _make_method("__hash__", return_annotation="None")

        return generated

    # ---------------------------------------------------------------
    # Enum helpers
    # ---------------------------------------------------------------

    def _analyze_enum_members(self, node: ast.ClassDef) -> Dict[str, Optional[ast.expr]]:
        """Extract Enum member definitions (class-level assignments)."""
        members: Dict[str, Optional[ast.expr]] = {}
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        # Skip dunders and methods
                        if not target.id.startswith("_"):
                            members[target.id] = item.value
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if not item.target.id.startswith("_"):
                    members[item.target.id] = item.value
        return members

    # ---------------------------------------------------------------
    # Lookup helpers
    # ---------------------------------------------------------------

    def get_class_info(self, name: str) -> Optional[ClassInfo]:
        """Look up a class by name in user-defined and builtin classes."""
        if name in self.classes:
            return self.classes[name]
        if name in self.builtin_classes:
            return self.builtin_classes[name]
        return None

    def get_all_classes(self) -> Dict[str, ClassInfo]:
        """Return a merged dictionary of all known classes."""
        merged: Dict[str, ClassInfo] = {}
        merged.update(self.builtin_classes)
        merged.update(self.classes)  # user classes shadow builtins
        return merged
