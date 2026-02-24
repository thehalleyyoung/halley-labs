from __future__ import annotations

import ast
import copy
import collections
import datetime
import difflib
import enum
import hashlib
import io
import json
import os
import re
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Refinement annotations
# ---------------------------------------------------------------------------

@dataclass
class RefinementAnnotation(ABC):
    """Base for all refinement predicates attached to a type."""

    @abstractmethod
    def to_annotation_str(self) -> str:
        """Return an ``Annotated[...]``-compatible metadata string."""

    @abstractmethod
    def to_predicate(self) -> str:
        """Return a human-readable predicate expression (lambda-style)."""

    def __hash__(self) -> int:
        return hash(self.to_annotation_str())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RefinementAnnotation):
            return NotImplemented
        return self.to_annotation_str() == other.to_annotation_str()


@dataclass
class Gt(RefinementAnnotation):
    value: float

    def to_annotation_str(self) -> str:
        return f"Gt({self.value})"

    def to_predicate(self) -> str:
        return f"lambda x: x > {self.value}"


@dataclass
class Lt(RefinementAnnotation):
    value: float

    def to_annotation_str(self) -> str:
        return f"Lt({self.value})"

    def to_predicate(self) -> str:
        return f"lambda x: x < {self.value}"


@dataclass
class Ge(RefinementAnnotation):
    value: float

    def to_annotation_str(self) -> str:
        return f"Ge({self.value})"

    def to_predicate(self) -> str:
        return f"lambda x: x >= {self.value}"


@dataclass
class Le(RefinementAnnotation):
    value: float

    def to_annotation_str(self) -> str:
        return f"Le({self.value})"

    def to_predicate(self) -> str:
        return f"lambda x: x <= {self.value}"


@dataclass
class Eq(RefinementAnnotation):
    value: Any

    def to_annotation_str(self) -> str:
        return f"Eq({self.value!r})"

    def to_predicate(self) -> str:
        return f"lambda x: x == {self.value!r}"


@dataclass
class Ne(RefinementAnnotation):
    value: Any

    def to_annotation_str(self) -> str:
        return f"Ne({self.value!r})"

    def to_predicate(self) -> str:
        return f"lambda x: x != {self.value!r}"


@dataclass
class NonNone(RefinementAnnotation):
    def to_annotation_str(self) -> str:
        return "NonNone()"

    def to_predicate(self) -> str:
        return "lambda x: x is not None"


@dataclass
class IsInstance(RefinementAnnotation):
    type_name: str

    def to_annotation_str(self) -> str:
        return f"IsInstance({self.type_name})"

    def to_predicate(self) -> str:
        return f"lambda x: isinstance(x, {self.type_name})"


@dataclass
class HasAttr(RefinementAnnotation):
    attr_name: str

    def to_annotation_str(self) -> str:
        return f"HasAttr({self.attr_name!r})"

    def to_predicate(self) -> str:
        return f"lambda x: hasattr(x, {self.attr_name!r})"


@dataclass
class Len(RefinementAnnotation):
    operator: str  # "==", ">=", "<=", ">", "<"
    length: int

    def to_annotation_str(self) -> str:
        return f"Len({self.operator!r}, {self.length})"

    def to_predicate(self) -> str:
        return f"lambda x: len(x) {self.operator} {self.length}"


@dataclass
class Between(RefinementAnnotation):
    low: float
    high: float
    inclusive_low: bool = True
    inclusive_high: bool = True

    def to_annotation_str(self) -> str:
        return (
            f"Between({self.low}, {self.high}, "
            f"inclusive_low={self.inclusive_low}, "
            f"inclusive_high={self.inclusive_high})"
        )

    def to_predicate(self) -> str:
        lo = ">=" if self.inclusive_low else ">"
        hi = "<=" if self.inclusive_high else "<"
        return f"lambda x: {self.low} {lo} x and x {hi} {self.high}"


@dataclass
class NonEmpty(RefinementAnnotation):
    def to_annotation_str(self) -> str:
        return "NonEmpty()"

    def to_predicate(self) -> str:
        return "lambda x: len(x) > 0"


@dataclass
class Positive(RefinementAnnotation):
    def to_annotation_str(self) -> str:
        return "Positive()"

    def to_predicate(self) -> str:
        return "lambda x: x > 0"


@dataclass
class NonNegative(RefinementAnnotation):
    def to_annotation_str(self) -> str:
        return "NonNegative()"

    def to_predicate(self) -> str:
        return "lambda x: x >= 0"


@dataclass
class NonZero(RefinementAnnotation):
    def to_annotation_str(self) -> str:
        return "NonZero()"

    def to_predicate(self) -> str:
        return "lambda x: x != 0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASIC_TYPES: Dict[str, str] = {
    "int": "int",
    "str": "str",
    "float": "float",
    "bool": "bool",
    "None": "None",
    "NoneType": "None",
    "bytes": "bytes",
    "complex": "complex",
    "bytearray": "bytearray",
    "memoryview": "memoryview",
    "object": "object",
    "type": "type",
}

_CONTAINER_PEP585: Dict[str, str] = {
    "List": "list",
    "Dict": "dict",
    "Set": "set",
    "Tuple": "tuple",
    "FrozenSet": "frozenset",
    "Deque": "collections.deque",
}

_CONTAINER_TYPING: Dict[str, str] = {
    "List": "List",
    "Dict": "Dict",
    "Set": "Set",
    "Tuple": "Tuple",
    "FrozenSet": "FrozenSet",
    "Deque": "Deque",
}


class DocstringStyle(enum.Enum):
    GOOGLE = "google"
    NUMPY = "numpy"
    REST = "rest"


class ConflictStrategy(enum.Enum):
    PREFER_MANUAL = "prefer-manual"
    PREFER_INFERRED = "prefer-inferred"
    UNION = "union"


@dataclass
class ValidationError:
    line: int
    col: int
    message: str
    severity: str = "error"

    def __str__(self) -> str:
        return f"{self.severity}:{self.line}:{self.col}: {self.message}"


# ---------------------------------------------------------------------------
# TypeFormatter
# ---------------------------------------------------------------------------

@dataclass
class TypeFormatter:
    """Converts internal type representations to Python annotation strings."""

    target_version: Tuple[int, int] = (3, 10)
    use_pep585: bool = True
    use_pep604: bool = True
    _seen: Set[int] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if self.target_version < (3, 9):
            self.use_pep585 = False
        if self.target_version < (3, 10):
            self.use_pep604 = False

    # -- public API ---------------------------------------------------------

    def format_type(self, type_desc: Dict[str, Any]) -> str:
        obj_id = id(type_desc)
        if obj_id in self._seen:
            return type_desc.get("name", "Any")
        self._seen.add(obj_id)
        try:
            return self._dispatch(type_desc)
        finally:
            self._seen.discard(obj_id)

    def format_basic(self, name: str) -> str:
        return _BASIC_TYPES.get(name, name)

    def format_container(self, container: str, args: List[Dict[str, Any]]) -> str:
        formatted_args = ", ".join(self.format_type(a) for a in args)
        base = self._container_name(container)
        return f"{base}[{formatted_args}]" if formatted_args else base

    def format_union(self, members: List[Dict[str, Any]]) -> str:
        formatted = []
        for m in members:
            s = self.format_type(m)
            if s not in formatted:
                formatted.append(s)
        if not formatted:
            return "Any"
        if len(formatted) == 1:
            return formatted[0]
        if self.use_pep604:
            return " | ".join(formatted)
        return f"Union[{', '.join(formatted)}]"

    def format_optional(self, inner: Dict[str, Any]) -> str:
        inner_str = self.format_type(inner)
        if self.use_pep604:
            return f"{inner_str} | None"
        return f"Optional[{inner_str}]"

    def format_callable(
        self,
        params: List[Dict[str, Any]],
        return_type: Dict[str, Any],
    ) -> str:
        param_strs = [self.format_type(p) for p in params]
        ret_str = self.format_type(return_type)
        return f"Callable[[{', '.join(param_strs)}], {ret_str}]"

    def format_callable_paramspec(
        self, paramspec_name: str, return_type: Dict[str, Any]
    ) -> str:
        ret_str = self.format_type(return_type)
        return f"Callable[{paramspec_name}, {ret_str}]"

    def format_typevar(
        self,
        name: str,
        bound: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        covariant: bool = False,
        contravariant: bool = False,
    ) -> str:
        parts = [f"TypeVar({name!r}"]
        if constraints:
            for c in constraints:
                parts.append(f", {self.format_type(c)}")
        if bound:
            parts.append(f", bound={self.format_type(bound)}")
        if covariant:
            parts.append(", covariant=True")
        if contravariant:
            parts.append(", contravariant=True")
        parts.append(")")
        return "".join(parts)

    def format_generic(self, base: str, params: List[Dict[str, Any]]) -> str:
        param_strs = ", ".join(self.format_type(p) for p in params)
        return f"{base}[{param_strs}]"

    def format_literal(self, values: List[Any]) -> str:
        formatted_vals = []
        for v in values:
            if isinstance(v, str):
                formatted_vals.append(repr(v))
            elif isinstance(v, bool):
                formatted_vals.append("True" if v else "False")
            elif isinstance(v, (int, float)):
                formatted_vals.append(str(v))
            elif v is None:
                formatted_vals.append("None")
            else:
                formatted_vals.append(repr(v))
        return f"Literal[{', '.join(formatted_vals)}]"

    def format_protocol(self, name: str, methods: List[str]) -> str:
        return name

    def format_typeddict(
        self, name: str, fields: Dict[str, Dict[str, Any]], total: bool = True
    ) -> str:
        return name

    def format_annotated(
        self, base: Dict[str, Any], refinements: List[RefinementAnnotation]
    ) -> str:
        base_str = self.format_type(base)
        meta_parts = [r.to_annotation_str() for r in refinements]
        return f"Annotated[{base_str}, {', '.join(meta_parts)}]"

    def format_typeguard(self, inner: Dict[str, Any]) -> str:
        return f"TypeGuard[{self.format_type(inner)}]"

    def format_typeis(self, inner: Dict[str, Any]) -> str:
        return f"TypeIs[{self.format_type(inner)}]"

    def format_paramspec(self, name: str) -> str:
        return f"ParamSpec({name!r})"

    def format_concatenate(
        self, types: List[Dict[str, Any]], paramspec: str
    ) -> str:
        type_strs = [self.format_type(t) for t in types]
        type_strs.append(paramspec)
        return f"Concatenate[{', '.join(type_strs)}]"

    def format_self(self) -> str:
        return "Self"

    def format_unpack(self, inner: Dict[str, Any]) -> str:
        return f"Unpack[{self.format_type(inner)}]"

    def format_typevar_tuple(self, name: str) -> str:
        return f"TypeVarTuple({name!r})"

    def format_literal_string(self) -> str:
        return "LiteralString"

    def format_never(self) -> str:
        if self.target_version >= (3, 11):
            return "Never"
        return "NoReturn"

    def format_generator(
        self,
        yield_type: Dict[str, Any],
        send_type: Optional[Dict[str, Any]] = None,
        return_type: Optional[Dict[str, Any]] = None,
    ) -> str:
        y = self.format_type(yield_type)
        s = self.format_type(send_type) if send_type else "None"
        r = self.format_type(return_type) if return_type else "None"
        return f"Generator[{y}, {s}, {r}]"

    def format_context_manager(self, inner: Dict[str, Any]) -> str:
        return f"ContextManager[{self.format_type(inner)}]"

    def format_async_generator(
        self,
        yield_type: Dict[str, Any],
        send_type: Optional[Dict[str, Any]] = None,
    ) -> str:
        y = self.format_type(yield_type)
        s = self.format_type(send_type) if send_type else "None"
        return f"AsyncGenerator[{y}, {s}]"

    def format_async_context_manager(self, inner: Dict[str, Any]) -> str:
        return f"AsyncContextManager[{self.format_type(inner)}]"

    def format_classvar(self, inner: Dict[str, Any]) -> str:
        return f"ClassVar[{self.format_type(inner)}]"

    def format_final(self, inner: Dict[str, Any]) -> str:
        return f"Final[{self.format_type(inner)}]"

    def format_type_alias(self, name: str, target: Dict[str, Any]) -> str:
        return f"{name} = {self.format_type(target)}"

    # -- internal -----------------------------------------------------------

    def _dispatch(self, td: Dict[str, Any]) -> str:
        kind = td.get("kind", "basic")
        if kind == "basic":
            return self.format_basic(td.get("name", "Any"))
        if kind == "container":
            return self.format_container(
                td.get("container", "List"), td.get("args", [])
            )
        if kind == "union":
            return self.format_union(td.get("members", []))
        if kind == "optional":
            return self.format_optional(td.get("inner", {"kind": "basic", "name": "Any"}))
        if kind == "callable":
            return self.format_callable(
                td.get("params", []),
                td.get("return_type", {"kind": "basic", "name": "Any"}),
            )
        if kind == "callable_paramspec":
            return self.format_callable_paramspec(
                td.get("paramspec", "P"),
                td.get("return_type", {"kind": "basic", "name": "Any"}),
            )
        if kind == "typevar":
            return td.get("name", "T")
        if kind == "generic":
            return self.format_generic(td.get("base", "Any"), td.get("params", []))
        if kind == "literal":
            return self.format_literal(td.get("values", []))
        if kind == "annotated":
            return self.format_annotated(
                td.get("base", {"kind": "basic", "name": "Any"}),
                td.get("refinements", []),
            )
        if kind == "typeguard":
            return self.format_typeguard(
                td.get("inner", {"kind": "basic", "name": "Any"})
            )
        if kind == "typeis":
            return self.format_typeis(
                td.get("inner", {"kind": "basic", "name": "Any"})
            )
        if kind == "paramspec":
            return td.get("name", "P")
        if kind == "concatenate":
            return self.format_concatenate(
                td.get("types", []), td.get("paramspec", "P")
            )
        if kind == "self":
            return self.format_self()
        if kind == "unpack":
            return self.format_unpack(
                td.get("inner", {"kind": "basic", "name": "Any"})
            )
        if kind == "typevar_tuple":
            return td.get("name", "Ts")
        if kind == "literal_string":
            return self.format_literal_string()
        if kind == "never":
            return self.format_never()
        if kind == "generator":
            return self.format_generator(
                td.get("yield_type", {"kind": "basic", "name": "Any"}),
                td.get("send_type"),
                td.get("return_type"),
            )
        if kind == "context_manager":
            return self.format_context_manager(
                td.get("inner", {"kind": "basic", "name": "Any"})
            )
        if kind == "async_generator":
            return self.format_async_generator(
                td.get("yield_type", {"kind": "basic", "name": "Any"}),
                td.get("send_type"),
            )
        if kind == "async_context_manager":
            return self.format_async_context_manager(
                td.get("inner", {"kind": "basic", "name": "Any"})
            )
        if kind == "classvar":
            return self.format_classvar(
                td.get("inner", {"kind": "basic", "name": "Any"})
            )
        if kind == "final":
            return self.format_final(
                td.get("inner", {"kind": "basic", "name": "Any"})
            )
        if kind == "protocol":
            return self.format_protocol(
                td.get("name", "Proto"), td.get("methods", [])
            )
        if kind == "typeddict":
            return self.format_typeddict(
                td.get("name", "TD"),
                td.get("fields", {}),
                td.get("total", True),
            )
        return td.get("name", "Any")

    def _container_name(self, container: str) -> str:
        if self.use_pep585 and container in _CONTAINER_PEP585:
            return _CONTAINER_PEP585[container]
        if container in _CONTAINER_TYPING:
            return _CONTAINER_TYPING[container]
        return container


# ---------------------------------------------------------------------------
# ImportResolver
# ---------------------------------------------------------------------------

@dataclass
class ImportResolver:
    """Resolves, organises, and renders import statements for stubs."""

    target_version: Tuple[int, int] = (3, 10)
    use_future_annotations: bool = True

    _typing_imports: Set[str] = field(default_factory=set, repr=False)
    _typing_extensions_imports: Set[str] = field(default_factory=set, repr=False)
    _stdlib_imports: Dict[str, Set[str]] = field(default_factory=lambda: collections.defaultdict(set), repr=False)
    _bare_imports: Set[str] = field(default_factory=set, repr=False)
    _collections_abc_imports: Set[str] = field(default_factory=set, repr=False)
    _forward_refs: Set[str] = field(default_factory=set, repr=False)

    _TYPING_CONSTRUCTS: Dict[str, Tuple[int, int]] = field(
        default=None, repr=False, init=False  # type: ignore[assignment]
    )

    def __post_init__(self) -> None:
        self._TYPING_CONSTRUCTS = {
            "Any": (3, 0),
            "Union": (3, 0),
            "Optional": (3, 0),
            "List": (3, 0),
            "Dict": (3, 0),
            "Set": (3, 0),
            "Tuple": (3, 0),
            "FrozenSet": (3, 0),
            "Callable": (3, 0),
            "Type": (3, 0),
            "ClassVar": (3, 0),
            "Final": (3, 8),
            "Literal": (3, 8),
            "TypedDict": (3, 8),
            "Protocol": (3, 8),
            "runtime_checkable": (3, 8),
            "Annotated": (3, 9),
            "TypeGuard": (3, 10),
            "ParamSpec": (3, 10),
            "Concatenate": (3, 10),
            "TypeVarTuple": (3, 11),
            "Unpack": (3, 11),
            "Self": (3, 11),
            "Never": (3, 11),
            "TypeIs": (3, 12),
            "LiteralString": (3, 11),
            "TypeVar": (3, 0),
            "overload": (3, 0),
            "Generator": (3, 0),
            "AsyncGenerator": (3, 0),
            "ContextManager": (3, 0),
            "AsyncContextManager": (3, 0),
            "Iterator": (3, 0),
            "Iterable": (3, 0),
            "Sequence": (3, 0),
            "Mapping": (3, 0),
            "MutableMapping": (3, 0),
            "MutableSequence": (3, 0),
            "MutableSet": (3, 0),
            "AbstractSet": (3, 0),
            "Deque": (3, 0),
            "NoReturn": (3, 0),
            "NamedTuple": (3, 0),
        }

    def require(self, name: str) -> None:
        if name in self._TYPING_CONSTRUCTS:
            min_ver = self._TYPING_CONSTRUCTS[name]
            if self.target_version >= min_ver:
                self._typing_imports.add(name)
            else:
                self._typing_extensions_imports.add(name)
        elif name in {
            "abstractmethod",
            "ABC",
        }:
            self._stdlib_imports["abc"].add(name)
        elif name in {"dataclass", "field"}:
            self._stdlib_imports["dataclasses"].add(name)
        elif name in {"Enum", "IntEnum", "auto"}:
            self._stdlib_imports["enum"].add(name)
        elif name in {"deque", "defaultdict", "OrderedDict", "Counter"}:
            self._stdlib_imports["collections"].add(name)
        else:
            self._forward_refs.add(name)

    def require_typing(self, name: str) -> None:
        if name in self._TYPING_CONSTRUCTS:
            min_ver = self._TYPING_CONSTRUCTS[name]
            if self.target_version >= min_ver:
                self._typing_imports.add(name)
            else:
                self._typing_extensions_imports.add(name)
        else:
            self._typing_imports.add(name)

    def require_typing_extensions(self, name: str) -> None:
        self._typing_extensions_imports.add(name)

    def require_stdlib(self, module: str, name: Optional[str] = None) -> None:
        if name is None:
            self._bare_imports.add(module)
        else:
            self._stdlib_imports[module].add(name)

    def require_collections_abc(self, name: str) -> None:
        self._collections_abc_imports.add(name)

    def add_forward_ref(self, name: str) -> None:
        self._forward_refs.add(name)

    def scan_type_string(self, type_str: str) -> None:
        """Scan a rendered type string and auto-require needed imports."""
        typing_names = {
            "Any", "Union", "Optional", "Callable", "ClassVar", "Final",
            "Literal", "Annotated", "TypeGuard", "TypeIs", "ParamSpec",
            "Concatenate", "TypeVarTuple", "Unpack", "Self", "Never",
            "NoReturn", "LiteralString", "TypeVar", "overload",
            "Generator", "AsyncGenerator", "ContextManager",
            "AsyncContextManager", "Protocol", "TypedDict", "NamedTuple",
            "Iterator", "Iterable", "Sequence", "Mapping",
        }
        for name in typing_names:
            if re.search(rf"\b{name}\b", type_str):
                self.require_typing(name)
        if "collections.deque" in type_str:
            self.require_stdlib("collections")

    def render(self) -> str:
        blocks: List[str] = []
        if self.use_future_annotations:
            blocks.append("from __future__ import annotations")
        bare_lines = sorted(self._bare_imports)
        if bare_lines:
            blocks.append("\n".join(f"import {m}" for m in bare_lines))
        stdlib_lines: List[str] = []
        for mod in sorted(self._stdlib_imports):
            names = sorted(self._stdlib_imports[mod])
            if names:
                stdlib_lines.append(f"from {mod} import {', '.join(names)}")
        if stdlib_lines:
            blocks.append("\n".join(stdlib_lines))
        if self._collections_abc_imports:
            names = sorted(self._collections_abc_imports)
            blocks.append(f"from collections.abc import {', '.join(names)}")
        if self._typing_imports:
            names = sorted(self._typing_imports)
            lines = self._wrap_from_import("typing", names)
            blocks.append(lines)
        if self._typing_extensions_imports:
            names = sorted(self._typing_extensions_imports)
            lines = self._wrap_from_import("typing_extensions", names)
            blocks.append(lines)
        version_conditional = self._version_conditional_imports()
        if version_conditional:
            blocks.append(version_conditional)
        return "\n\n".join(blocks)

    def render_version_check(self, min_version: Tuple[int, int], true_import: str, false_import: str) -> str:
        major, minor = min_version
        return (
            f"import sys\n"
            f"if sys.version_info >= ({major}, {minor}):\n"
            f"    {true_import}\n"
            f"else:\n"
            f"    {false_import}"
        )

    def get_all_required_names(self) -> Set[str]:
        result: Set[str] = set()
        result.update(self._typing_imports)
        result.update(self._typing_extensions_imports)
        for names in self._stdlib_imports.values():
            result.update(names)
        result.update(self._collections_abc_imports)
        result.update(self._bare_imports)
        return result

    def deduplicate(self) -> None:
        overlap = self._typing_imports & self._typing_extensions_imports
        for name in overlap:
            min_ver = self._TYPING_CONSTRUCTS.get(name, (3, 99))
            if self.target_version >= min_ver:
                self._typing_extensions_imports.discard(name)
            else:
                self._typing_imports.discard(name)

    def clear(self) -> None:
        self._typing_imports.clear()
        self._typing_extensions_imports.clear()
        self._stdlib_imports.clear()
        self._bare_imports.clear()
        self._collections_abc_imports.clear()
        self._forward_refs.clear()

    def _wrap_from_import(self, module: str, names: List[str], max_len: int = 88) -> str:
        single = f"from {module} import {', '.join(names)}"
        if len(single) <= max_len:
            return single
        items = ",\n    ".join(names)
        return f"from {module} import (\n    {items},\n)"

    def _version_conditional_imports(self) -> str:
        lines: List[str] = []
        needed_ext: Dict[Tuple[int, int], List[str]] = collections.defaultdict(list)
        for name in sorted(self._typing_extensions_imports):
            min_ver = self._TYPING_CONSTRUCTS.get(name)
            if min_ver and min_ver > self.target_version:
                needed_ext[min_ver].append(name)
        for ver in sorted(needed_ext):
            names_str = ", ".join(needed_ext[ver])
            major, minor = ver
            lines.append(
                f"import sys\n"
                f"if sys.version_info >= ({major}, {minor}):\n"
                f"    from typing import {names_str}\n"
                f"else:\n"
                f"    from typing_extensions import {names_str}"
            )
        return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# DocstringGenerator
# ---------------------------------------------------------------------------

@dataclass
class DocstringGenerator:
    """Generates docstrings from refinement information."""

    style: DocstringStyle = DocstringStyle.GOOGLE
    indent: str = "    "
    max_line_length: int = 88

    def generate(
        self,
        description: str = "",
        params: Optional[List[Dict[str, Any]]] = None,
        return_type: Optional[Dict[str, Any]] = None,
        return_description: str = "",
        raises: Optional[List[Dict[str, str]]] = None,
        examples: Optional[List[str]] = None,
        refinements: Optional[Dict[str, List[RefinementAnnotation]]] = None,
    ) -> str:
        if self.style == DocstringStyle.GOOGLE:
            return self._google(
                description, params or [], return_type, return_description,
                raises or [], examples or [], refinements or {},
            )
        if self.style == DocstringStyle.NUMPY:
            return self._numpy(
                description, params or [], return_type, return_description,
                raises or [], examples or [], refinements or {},
            )
        return self._rest(
            description, params or [], return_type, return_description,
            raises or [], examples or [], refinements or {},
        )

    def _refinement_note(self, name: str, refs: List[RefinementAnnotation]) -> str:
        preds = [r.to_predicate() for r in refs]
        return f"Constraints on ``{name}``: {'; '.join(preds)}"

    # -- Google style -------------------------------------------------------

    def _google(
        self,
        desc: str,
        params: List[Dict[str, Any]],
        ret: Optional[Dict[str, Any]],
        ret_desc: str,
        raises: List[Dict[str, str]],
        examples: List[str],
        refinements: Dict[str, List[RefinementAnnotation]],
    ) -> str:
        lines: List[str] = ['"""']
        if desc:
            lines.append(desc)
            lines.append("")
        if params:
            lines.append("Args:")
            for p in params:
                pname = p.get("name", "arg")
                ptype = p.get("type_str", "")
                pdesc = p.get("description", "")
                ref_note = ""
                if pname in refinements:
                    ref_note = " " + self._refinement_note(pname, refinements[pname])
                type_part = f" ({ptype})" if ptype else ""
                lines.append(f"    {pname}{type_part}: {pdesc}{ref_note}".rstrip())
            lines.append("")
        if ret or ret_desc:
            lines.append("Returns:")
            ret_type_str = ""
            if ret:
                tf = TypeFormatter()
                ret_type_str = tf.format_type(ret)
            lines.append(f"    {ret_type_str}: {ret_desc}".strip())
            lines.append("")
        if raises:
            lines.append("Raises:")
            for r in raises:
                exc = r.get("exception", "Exception")
                reason = r.get("description", "")
                lines.append(f"    {exc}: {reason}".rstrip())
            lines.append("")
        if examples:
            lines.append("Examples:")
            for ex in examples:
                for ex_line in ex.splitlines():
                    lines.append(f"    {ex_line}")
            lines.append("")
        lines.append('"""')
        return "\n".join(lines)

    # -- NumPy style --------------------------------------------------------

    def _numpy(
        self,
        desc: str,
        params: List[Dict[str, Any]],
        ret: Optional[Dict[str, Any]],
        ret_desc: str,
        raises: List[Dict[str, str]],
        examples: List[str],
        refinements: Dict[str, List[RefinementAnnotation]],
    ) -> str:
        lines: List[str] = ['"""']
        if desc:
            lines.append(desc)
            lines.append("")
        if params:
            lines.append("Parameters")
            lines.append("----------")
            for p in params:
                pname = p.get("name", "arg")
                ptype = p.get("type_str", "")
                pdesc = p.get("description", "")
                lines.append(f"{pname} : {ptype}")
                if pdesc:
                    lines.append(f"    {pdesc}")
                if pname in refinements:
                    lines.append(f"    {self._refinement_note(pname, refinements[pname])}")
            lines.append("")
        if ret or ret_desc:
            lines.append("Returns")
            lines.append("-------")
            if ret:
                tf = TypeFormatter()
                lines.append(tf.format_type(ret))
            if ret_desc:
                lines.append(f"    {ret_desc}")
            lines.append("")
        if raises:
            lines.append("Raises")
            lines.append("------")
            for r in raises:
                exc = r.get("exception", "Exception")
                reason = r.get("description", "")
                lines.append(exc)
                if reason:
                    lines.append(f"    {reason}")
            lines.append("")
        if examples:
            lines.append("Examples")
            lines.append("--------")
            for ex in examples:
                for ex_line in ex.splitlines():
                    lines.append(ex_line)
            lines.append("")
        lines.append('"""')
        return "\n".join(lines)

    # -- reST style ---------------------------------------------------------

    def _rest(
        self,
        desc: str,
        params: List[Dict[str, Any]],
        ret: Optional[Dict[str, Any]],
        ret_desc: str,
        raises: List[Dict[str, str]],
        examples: List[str],
        refinements: Dict[str, List[RefinementAnnotation]],
    ) -> str:
        lines: List[str] = ['"""']
        if desc:
            lines.append(desc)
            lines.append("")
        for p in params:
            pname = p.get("name", "arg")
            ptype = p.get("type_str", "")
            pdesc = p.get("description", "")
            lines.append(f":param {pname}: {pdesc}".rstrip())
            if ptype:
                lines.append(f":type {pname}: {ptype}")
            if pname in refinements:
                lines.append(f":constraint {pname}: {self._refinement_note(pname, refinements[pname])}")
        if ret or ret_desc:
            if ret:
                tf = TypeFormatter()
                lines.append(f":rtype: {tf.format_type(ret)}")
            if ret_desc:
                lines.append(f":returns: {ret_desc}")
        for r in raises:
            exc = r.get("exception", "Exception")
            reason = r.get("description", "")
            lines.append(f":raises {exc}: {reason}".rstrip())
        if examples:
            lines.append("")
            lines.append(".. code-block:: python")
            lines.append("")
            for ex in examples:
                for ex_line in ex.splitlines():
                    lines.append(f"    {ex_line}")
        lines.append('"""')
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# FunctionStubGenerator
# ---------------------------------------------------------------------------

@dataclass
class FunctionStubGenerator:
    """Generates individual function stubs."""

    formatter: TypeFormatter = field(default_factory=TypeFormatter)
    imports: ImportResolver = field(default_factory=ImportResolver)
    docstring_gen: DocstringGenerator = field(default_factory=DocstringGenerator)
    generate_docstrings: bool = True

    def generate(self, func_info: Dict[str, Any]) -> str:
        name = func_info.get("name", "_unknown")
        is_async = func_info.get("is_async", False)
        decorators = func_info.get("decorators", [])
        params = func_info.get("params", [])
        return_type = func_info.get("return_type")
        overloads = func_info.get("overloads", [])
        docstring = func_info.get("docstring", "")
        refinements = func_info.get("refinements", {})

        lines: List[str] = []

        if overloads:
            for ov in overloads:
                lines.extend(self._render_overload(name, ov, is_async, decorators))
                lines.append("")

        lines.extend(self._render_function(
            name, params, return_type, is_async, decorators, docstring, refinements,
            is_overload=False,
        ))
        return "\n".join(lines)

    def generate_method(
        self, func_info: Dict[str, Any], indent: str = "    "
    ) -> str:
        raw = self.generate(func_info)
        indented_lines: List[str] = []
        for line in raw.splitlines():
            if line.strip():
                indented_lines.append(indent + line)
            else:
                indented_lines.append("")
        return "\n".join(indented_lines)

    def _render_overload(
        self,
        name: str,
        overload_info: Dict[str, Any],
        is_async: bool,
        decorators: List[str],
    ) -> List[str]:
        self.imports.require_typing("overload")
        params = overload_info.get("params", [])
        return_type = overload_info.get("return_type")
        return self._render_function(
            name, params, return_type, is_async, decorators, "", {},
            is_overload=True,
        )

    def _render_function(
        self,
        name: str,
        params: List[Dict[str, Any]],
        return_type: Optional[Dict[str, Any]],
        is_async: bool,
        decorators: List[str],
        docstring: str,
        refinements: Dict[str, List[RefinementAnnotation]],
        is_overload: bool = False,
    ) -> List[str]:
        lines: List[str] = []

        if is_overload:
            lines.append("@overload")
        for dec in decorators:
            self._track_decorator_import(dec)
            lines.append(f"@{dec}")

        param_strs = self._format_params(params)
        ret_str = ""
        if return_type is not None:
            ret_str = f" -> {self.formatter.format_type(return_type)}"
            self.imports.scan_type_string(ret_str)

        prefix = "async def" if is_async else "def"
        sig = f"{prefix} {name}({', '.join(param_strs)}){ret_str}:"
        lines.append(sig)

        body_lines: List[str] = []
        if self.generate_docstrings and docstring and not is_overload:
            for dl in docstring.splitlines():
                body_lines.append(f"    {dl}" if dl.strip() else "")
        elif self.generate_docstrings and refinements and not is_overload:
            ds = self.docstring_gen.generate(
                description="",
                params=[{"name": n} for n in refinements],
                refinements=refinements,
            )
            for dl in ds.splitlines():
                body_lines.append(f"    {dl}" if dl.strip() else "")

        body_lines.append("    ...")

        lines.extend(body_lines)
        return lines

    def _format_params(self, params: List[Dict[str, Any]]) -> List[str]:
        parts: List[str] = []
        seen_star = False
        for p in params:
            pname = p.get("name", "arg")
            ptype = p.get("type")
            default = p.get("default")
            param_kind = p.get("kind", "positional_or_keyword")

            if param_kind == "var_positional":
                type_str = ""
                if ptype:
                    type_str = f": {self.formatter.format_type(ptype)}"
                    self.imports.scan_type_string(type_str)
                parts.append(f"*{pname}{type_str}")
                seen_star = True
                continue
            if param_kind == "var_keyword":
                type_str = ""
                if ptype:
                    type_str = f": {self.formatter.format_type(ptype)}"
                    self.imports.scan_type_string(type_str)
                parts.append(f"**{pname}{type_str}")
                continue
            if param_kind == "keyword_only" and not seen_star:
                parts.append("*")
                seen_star = True
            if param_kind == "positional_only":
                pass  # handled by adding / later

            type_str = ""
            if ptype:
                type_str = f": {self.formatter.format_type(ptype)}"
                self.imports.scan_type_string(type_str)

            default_str = ""
            if default is not None:
                default_str = f" = {self._format_default(default)}"

            parts.append(f"{pname}{type_str}{default_str}")

        # Insert / for positional-only boundary
        pos_only_count = sum(
            1 for p in params if p.get("kind") == "positional_only"
        )
        if pos_only_count > 0 and pos_only_count < len(params):
            parts.insert(pos_only_count, "/")
        elif pos_only_count > 0:
            parts.append("/")

        return parts

    def _format_default(self, default: Any) -> str:
        if default is None:
            return "None"
        if isinstance(default, bool):
            return "True" if default else "False"
        if isinstance(default, (int, float)):
            return repr(default)
        if isinstance(default, str):
            if len(default) > 40:
                return "..."
            return repr(default)
        return "..."

    def _track_decorator_import(self, dec: str) -> None:
        base = dec.split("(")[0].split(".")[0]
        if base in ("staticmethod", "classmethod", "property"):
            return
        if base in ("abstractmethod",):
            self.imports.require("abstractmethod")
        if base == "overload":
            self.imports.require_typing("overload")


# ---------------------------------------------------------------------------
# ClassStubGenerator
# ---------------------------------------------------------------------------

@dataclass
class ClassStubGenerator:
    """Generates class stubs, including Protocols and TypedDicts."""

    formatter: TypeFormatter = field(default_factory=TypeFormatter)
    imports: ImportResolver = field(default_factory=ImportResolver)
    func_gen: FunctionStubGenerator = field(default_factory=FunctionStubGenerator)
    docstring_gen: DocstringGenerator = field(default_factory=DocstringGenerator)
    indent: str = "    "

    def __post_init__(self) -> None:
        self.func_gen.formatter = self.formatter
        self.func_gen.imports = self.imports
        self.func_gen.docstring_gen = self.docstring_gen

    def generate(self, class_info: Dict[str, Any]) -> str:
        kind = class_info.get("class_kind", "regular")
        if kind == "protocol":
            return self._generate_protocol(class_info)
        if kind == "typeddict":
            return self._generate_typeddict(class_info)
        if kind == "namedtuple":
            return self._generate_namedtuple(class_info)
        if kind == "enum":
            return self._generate_enum(class_info)
        if kind == "dataclass":
            return self._generate_dataclass(class_info)
        return self._generate_regular(class_info)

    # -- regular class ------------------------------------------------------

    def _generate_regular(self, info: Dict[str, Any]) -> str:
        lines: List[str] = []
        name = info.get("name", "Unknown")
        bases = info.get("bases", [])
        decorators = info.get("decorators", [])
        slots = info.get("slots")
        class_vars = info.get("class_vars", [])
        instance_vars = info.get("instance_vars", [])
        methods = info.get("methods", [])
        nested_classes = info.get("nested_classes", [])
        generic_params = info.get("generic_params", [])
        docstring = info.get("docstring", "")

        for dec in decorators:
            lines.append(f"@{dec}")

        base_str = ""
        if generic_params:
            self.imports.require_typing("Generic")
            gp = ", ".join(generic_params)
            all_bases = list(bases) + [f"Generic[{gp}]"]
            base_str = f"({', '.join(all_bases)})"
        elif bases:
            base_str = f"({', '.join(bases)})"

        lines.append(f"class {name}{base_str}:")

        body: List[str] = []

        if docstring:
            body.append(f'{self.indent}"""{docstring}"""')

        if slots is not None:
            slot_str = ", ".join(repr(s) for s in slots)
            body.append(f"{self.indent}__slots__ = ({slot_str},)")

        for cv in class_vars:
            cv_name = cv.get("name", "var")
            cv_type = cv.get("type")
            cv_value = cv.get("value")
            if cv_type:
                self.imports.require_typing("ClassVar")
                type_str = self.formatter.format_classvar(cv_type)
                self.imports.scan_type_string(type_str)
                if cv_value is not None:
                    body.append(f"{self.indent}{cv_name}: {type_str} = {self._render_value(cv_value)}")
                else:
                    body.append(f"{self.indent}{cv_name}: {type_str}")
            else:
                if cv_value is not None:
                    body.append(f"{self.indent}{cv_name} = {self._render_value(cv_value)}")
                else:
                    body.append(f"{self.indent}{cv_name}: Any")

        for iv in instance_vars:
            iv_name = iv.get("name", "var")
            iv_type = iv.get("type")
            if iv_type:
                type_str = self.formatter.format_type(iv_type)
                self.imports.scan_type_string(type_str)
                body.append(f"{self.indent}{iv_name}: {type_str}")
            else:
                body.append(f"{self.indent}{iv_name}: Any")

        for nc in nested_classes:
            nested_stub = self.generate(nc)
            for nl in nested_stub.splitlines():
                body.append(f"{self.indent}{nl}" if nl.strip() else "")

        for method in methods:
            method_stub = self.func_gen.generate_method(method, self.indent)
            if body and body[-1].strip():
                body.append("")
            body.append(method_stub)

        if not body:
            body.append(f"{self.indent}...")

        lines.extend(body)
        return "\n".join(lines)

    # -- protocol -----------------------------------------------------------

    def _generate_protocol(self, info: Dict[str, Any]) -> str:
        self.imports.require_typing("Protocol")
        name = info.get("name", "Proto")
        methods = info.get("methods", [])
        runtime = info.get("runtime_checkable", False)

        lines: List[str] = []
        if runtime:
            self.imports.require_typing("runtime_checkable")
            lines.append("@runtime_checkable")
        lines.append(f"class {name}(Protocol):")

        if not methods:
            lines.append(f"{self.indent}...")
            return "\n".join(lines)

        for i, method in enumerate(methods):
            stub = self.func_gen.generate_method(method, self.indent)
            if i > 0:
                lines.append("")
            lines.append(stub)

        return "\n".join(lines)

    # -- TypedDict ----------------------------------------------------------

    def _generate_typeddict(self, info: Dict[str, Any]) -> str:
        self.imports.require_typing("TypedDict")
        name = info.get("name", "TD")
        fields = info.get("fields", {})
        total = info.get("total", True)

        lines: List[str] = []
        total_arg = "" if total else ", total=False"
        lines.append(f"class {name}(TypedDict{total_arg}):")

        if not fields:
            lines.append(f"{self.indent}...")
            return "\n".join(lines)

        for fname, ftype in fields.items():
            type_str = self.formatter.format_type(ftype) if isinstance(ftype, dict) else str(ftype)
            self.imports.scan_type_string(type_str)
            lines.append(f"{self.indent}{fname}: {type_str}")

        return "\n".join(lines)

    # -- NamedTuple ---------------------------------------------------------

    def _generate_namedtuple(self, info: Dict[str, Any]) -> str:
        self.imports.require_typing("NamedTuple")
        name = info.get("name", "NT")
        fields = info.get("fields", {})

        lines: List[str] = []
        lines.append(f"class {name}(NamedTuple):")

        if not fields:
            lines.append(f"{self.indent}...")
            return "\n".join(lines)

        for fname, ftype in fields.items():
            type_str = self.formatter.format_type(ftype) if isinstance(ftype, dict) else str(ftype)
            self.imports.scan_type_string(type_str)
            lines.append(f"{self.indent}{fname}: {type_str}")

        return "\n".join(lines)

    # -- Enum ---------------------------------------------------------------

    def _generate_enum(self, info: Dict[str, Any]) -> str:
        self.imports.require("Enum")
        name = info.get("name", "E")
        members = info.get("members", {})
        base = info.get("enum_base", "Enum")

        lines: List[str] = []
        lines.append(f"class {name}({base}):")

        if not members:
            lines.append(f"{self.indent}...")
            return "\n".join(lines)

        for mname, mval in members.items():
            lines.append(f"{self.indent}{mname} = {self._render_value(mval)}")

        methods = info.get("methods", [])
        for method in methods:
            lines.append("")
            lines.append(self.func_gen.generate_method(method, self.indent))

        return "\n".join(lines)

    # -- dataclass ----------------------------------------------------------

    def _generate_dataclass(self, info: Dict[str, Any]) -> str:
        self.imports.require("dataclass")
        name = info.get("name", "DC")
        fields_list = info.get("fields", [])
        bases = info.get("bases", [])
        methods = info.get("methods", [])
        frozen = info.get("frozen", False)
        eq = info.get("eq", True)
        order = info.get("order", False)

        lines: List[str] = []
        dc_args: List[str] = []
        if frozen:
            dc_args.append("frozen=True")
        if not eq:
            dc_args.append("eq=False")
        if order:
            dc_args.append("order=True")
        dc_decorator = "@dataclass" if not dc_args else f"@dataclass({', '.join(dc_args)})"
        lines.append(dc_decorator)

        base_str = f"({', '.join(bases)})" if bases else ""
        lines.append(f"class {name}{base_str}:")

        if not fields_list and not methods:
            lines.append(f"{self.indent}...")
            return "\n".join(lines)

        for f_info in fields_list:
            fname = f_info.get("name", "x")
            ftype = f_info.get("type")
            fdefault = f_info.get("default")
            type_str = self.formatter.format_type(ftype) if ftype else "Any"
            self.imports.scan_type_string(type_str)
            if fdefault is not None:
                lines.append(f"{self.indent}{fname}: {type_str} = {self._render_value(fdefault)}")
            else:
                lines.append(f"{self.indent}{fname}: {type_str}")

        for method in methods:
            lines.append("")
            lines.append(self.func_gen.generate_method(method, self.indent))

        return "\n".join(lines)

    # -- descriptor helpers -------------------------------------------------

    def generate_descriptor(self, desc_info: Dict[str, Any]) -> str:
        name = desc_info.get("name", "Descriptor")
        owner_type = desc_info.get("owner_type", {"kind": "basic", "name": "Any"})
        value_type = desc_info.get("value_type", {"kind": "basic", "name": "Any"})

        get_method: Dict[str, Any] = {
            "name": "__get__",
            "params": [
                {"name": "self", "kind": "positional_or_keyword"},
                {"name": "obj", "type": owner_type, "kind": "positional_or_keyword"},
                {"name": "objtype", "type": {"kind": "optional", "inner": {"kind": "basic", "name": "type"}}, "kind": "positional_or_keyword", "default": None},
            ],
            "return_type": value_type,
        }
        set_method: Dict[str, Any] = {
            "name": "__set__",
            "params": [
                {"name": "self", "kind": "positional_or_keyword"},
                {"name": "obj", "type": owner_type, "kind": "positional_or_keyword"},
                {"name": "value", "type": value_type, "kind": "positional_or_keyword"},
            ],
            "return_type": {"kind": "basic", "name": "None"},
        }
        del_method: Dict[str, Any] = {
            "name": "__delete__",
            "params": [
                {"name": "self", "kind": "positional_or_keyword"},
                {"name": "obj", "type": owner_type, "kind": "positional_or_keyword"},
            ],
            "return_type": {"kind": "basic", "name": "None"},
        }

        class_info: Dict[str, Any] = {
            "name": name,
            "methods": [get_method, set_method, del_method],
        }
        return self._generate_regular(class_info)

    def _render_value(self, val: Any) -> str:
        if val is None:
            return "None"
        if isinstance(val, bool):
            return "True" if val else "False"
        if isinstance(val, (int, float)):
            return repr(val)
        if isinstance(val, str):
            if len(val) > 50:
                return "..."
            return repr(val)
        if isinstance(val, (list, tuple, dict)):
            try:
                rendered = repr(val)
                if len(rendered) < 60:
                    return rendered
            except Exception:
                pass
        return "..."


# ---------------------------------------------------------------------------
# ModuleStubGenerator
# ---------------------------------------------------------------------------

@dataclass
class ModuleStubGenerator:
    """Generates a complete module stub."""

    formatter: TypeFormatter = field(default_factory=TypeFormatter)
    imports: ImportResolver = field(default_factory=ImportResolver)
    func_gen: FunctionStubGenerator = field(default_factory=FunctionStubGenerator)
    class_gen: ClassStubGenerator = field(default_factory=ClassStubGenerator)
    docstring_gen: DocstringGenerator = field(default_factory=DocstringGenerator)

    def __post_init__(self) -> None:
        self.func_gen.formatter = self.formatter
        self.func_gen.imports = self.imports
        self.func_gen.docstring_gen = self.docstring_gen
        self.class_gen.formatter = self.formatter
        self.class_gen.imports = self.imports
        self.class_gen.func_gen = self.func_gen
        self.class_gen.docstring_gen = self.docstring_gen

    def generate(self, module_info: Dict[str, Any]) -> str:
        module_doc = module_info.get("docstring", "")
        constants = module_info.get("constants", [])
        variables = module_info.get("variables", [])
        functions = module_info.get("functions", [])
        classes = module_info.get("classes", [])
        all_exports = module_info.get("__all__")
        type_checking_defs = module_info.get("type_checking", [])
        type_aliases = module_info.get("type_aliases", [])
        typevars = module_info.get("typevars", [])

        body_parts: List[str] = []

        if module_doc:
            body_parts.append(f'"""{module_doc}"""')

        # Generate all parts first so imports can accumulate
        typevar_block = self._generate_typevars(typevars)
        alias_block = self._generate_type_aliases(type_aliases)
        const_block = self._generate_constants(constants)
        var_block = self._generate_variables(variables)
        class_blocks = [self.class_gen.generate(c) for c in classes]
        func_blocks = [self.func_gen.generate(f) for f in functions]

        self.imports.deduplicate()
        import_block = self.imports.render()
        if import_block:
            body_parts.append(import_block)

        if type_checking_defs:
            tc_block = self._generate_type_checking(type_checking_defs)
            body_parts.append(tc_block)

        if typevar_block:
            body_parts.append(typevar_block)
        if alias_block:
            body_parts.append(alias_block)
        if const_block:
            body_parts.append(const_block)
        if var_block:
            body_parts.append(var_block)

        for cb in class_blocks:
            body_parts.append(cb)

        for fb in func_blocks:
            body_parts.append(fb)

        if all_exports is not None:
            all_str = ", ".join(repr(e) for e in all_exports)
            body_parts.append(f"__all__ = [{all_str}]")

        return "\n\n".join(body_parts) + "\n"

    def _generate_typevars(self, typevars: List[Dict[str, Any]]) -> str:
        if not typevars:
            return ""
        self.imports.require_typing("TypeVar")
        lines: List[str] = []
        for tv in typevars:
            name = tv.get("name", "T")
            decl = self.formatter.format_typevar(
                name,
                bound=tv.get("bound"),
                constraints=tv.get("constraints"),
                covariant=tv.get("covariant", False),
                contravariant=tv.get("contravariant", False),
            )
            lines.append(f"{name} = {decl}")
        return "\n".join(lines)

    def _generate_type_aliases(self, aliases: List[Dict[str, Any]]) -> str:
        if not aliases:
            return ""
        lines: List[str] = []
        for alias in aliases:
            name = alias.get("name", "Alias")
            target = alias.get("target", {"kind": "basic", "name": "Any"})
            target_str = self.formatter.format_type(target)
            self.imports.scan_type_string(target_str)
            lines.append(f"{name} = {target_str}")
        return "\n".join(lines)

    def _generate_constants(self, constants: List[Dict[str, Any]]) -> str:
        if not constants:
            return ""
        lines: List[str] = []
        for c in constants:
            cname = c.get("name", "CONST")
            ctype = c.get("type")
            cvalue = c.get("value")
            is_final = c.get("final", False)
            if ctype:
                type_str = self.formatter.format_type(ctype)
                if is_final:
                    self.imports.require_typing("Final")
                    type_str = f"Final[{type_str}]"
                self.imports.scan_type_string(type_str)
                if cvalue is not None:
                    lines.append(f"{cname}: {type_str} = {self.class_gen._render_value(cvalue)}")
                else:
                    lines.append(f"{cname}: {type_str}")
            else:
                if cvalue is not None:
                    lines.append(f"{cname} = {self.class_gen._render_value(cvalue)}")
                else:
                    lines.append(f"{cname}: Any")
        return "\n".join(lines)

    def _generate_variables(self, variables: List[Dict[str, Any]]) -> str:
        if not variables:
            return ""
        lines: List[str] = []
        for v in variables:
            vname = v.get("name", "var")
            vtype = v.get("type")
            if vtype:
                type_str = self.formatter.format_type(vtype)
                self.imports.scan_type_string(type_str)
                lines.append(f"{vname}: {type_str}")
            else:
                lines.append(f"{vname}: Any")
        return "\n".join(lines)

    def _generate_type_checking(self, defs: List[Dict[str, Any]]) -> str:
        self.imports.require_typing("TYPE_CHECKING")
        lines: List[str] = ["if TYPE_CHECKING:"]
        for d in defs:
            kind = d.get("kind", "import")
            if kind == "import":
                module = d.get("module", "")
                names = d.get("names", [])
                lines.append(f"    from {module} import {', '.join(names)}")
            elif kind == "alias":
                name = d.get("name", "X")
                target = d.get("target", "Any")
                lines.append(f"    {name} = {target}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# StubFileWriter
# ---------------------------------------------------------------------------

@dataclass
class StubFileWriter:
    """Writes formatted .pyi content to files."""

    encoding: str = "utf-8"
    use_bom: bool = False
    line_ending: str = "\n"
    max_line_length: int = 88
    header_comment: str = "# Automatically generated stub file. Do not edit."
    atomic: bool = True

    def write(self, path: Path, content: str) -> None:
        formatted = self._format(content)
        if self.atomic:
            self._atomic_write(path, formatted)
        else:
            self._direct_write(path, formatted)

    def write_string(self, content: str) -> str:
        return self._format(content)

    def _format(self, content: str) -> str:
        lines = content.split("\n")
        lines = self._normalize_blank_lines(lines)
        lines = self._wrap_long_lines(lines)
        result_lines = []
        if self.header_comment:
            result_lines.append(self.header_comment)
            result_lines.append("")
        result_lines.extend(lines)
        if result_lines and result_lines[-1].strip():
            result_lines.append("")
        return self.line_ending.join(result_lines)

    def _normalize_blank_lines(self, lines: List[str]) -> List[str]:
        result: List[str] = []
        blank_count = 0
        in_class = False
        indent_stack: List[int] = []

        for line in lines:
            stripped = line.strip()
            current_indent = len(line) - len(line.lstrip()) if stripped else 0

            if stripped.startswith("class "):
                in_class = True
                indent_stack = [current_indent]
            elif stripped.startswith("def ") or stripped.startswith("async def "):
                if indent_stack and current_indent <= indent_stack[0]:
                    in_class = False

            if not stripped:
                blank_count += 1
                continue

            if blank_count > 0:
                if in_class and current_indent > 0:
                    max_blanks = 1
                else:
                    max_blanks = 2
                result.extend([""] * min(blank_count, max_blanks))
            blank_count = 0
            result.append(line)

        return result

    def _wrap_long_lines(self, lines: List[str]) -> List[str]:
        result: List[str] = []
        for line in lines:
            if len(line) <= self.max_line_length:
                result.append(line)
                continue
            # Try to break at commas inside parameter lists
            if "(" in line and ")" in line:
                wrapped = self._wrap_function_sig(line)
                result.extend(wrapped)
            else:
                result.append(line)
        return result

    def _wrap_function_sig(self, line: str) -> List[str]:
        match = re.match(r"^(\s*)((?:async )?def \w+)\((.+)\)(.*):$", line)
        if not match:
            return [line]
        indent = match.group(1)
        func_part = match.group(2)
        params_str = match.group(3)
        ret_part = match.group(4)

        params = self._split_params(params_str)
        if not params:
            return [line]

        lines: List[str] = [f"{indent}{func_part}("]
        inner_indent = indent + "    "
        for i, param in enumerate(params):
            comma = "," if i < len(params) - 1 else ","
            lines.append(f"{inner_indent}{param.strip()}{comma}")
        lines.append(f"{indent}){ret_part}:")
        return lines

    def _split_params(self, params_str: str) -> List[str]:
        params: List[str] = []
        depth = 0
        current: List[str] = []
        for char in params_str:
            if char in "([{":
                depth += 1
                current.append(char)
            elif char in ")]}":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                params.append("".join(current))
                current = []
            else:
                current.append(char)
        if current:
            params.append("".join(current))
        return params

    def _atomic_write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".pyi.tmp")
        try:
            data = content.encode(self.encoding)
            if self.use_bom and self.encoding.lower().replace("-", "") == "utf8":
                data = b"\xef\xbb\xbf" + data
            tmp_path.write_bytes(data)
            tmp_path.replace(path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _direct_write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode(self.encoding)
        if self.use_bom and self.encoding.lower().replace("-", "") == "utf8":
            data = b"\xef\xbb\xbf" + data
        path.write_bytes(data)


# ---------------------------------------------------------------------------
# StubValidator
# ---------------------------------------------------------------------------

@dataclass
class StubValidator:
    """Validates generated .pyi stub content."""

    target_version: Tuple[int, int] = (3, 10)

    def validate(self, content: str, filename: str = "<stub>") -> List[ValidationError]:
        errors: List[ValidationError] = []
        errors.extend(self._check_syntax(content, filename))
        if not errors:
            tree = ast.parse(content, filename=filename, type_comments=True)
            errors.extend(self._check_duplicate_definitions(tree, filename))
            errors.extend(self._check_overload_consistency(tree, filename))
            errors.extend(self._check_type_references(content, filename))
        return errors

    def is_valid(self, content: str, filename: str = "<stub>") -> bool:
        return len(self.validate(content, filename)) == 0

    def _check_syntax(self, content: str, filename: str) -> List[ValidationError]:
        errors: List[ValidationError] = []
        try:
            ast.parse(content, filename=filename, type_comments=True)
        except SyntaxError as e:
            errors.append(ValidationError(
                line=e.lineno or 0,
                col=e.offset or 0,
                message=f"Syntax error: {e.msg}",
            ))
        return errors

    def _check_duplicate_definitions(
        self, tree: ast.Module, filename: str
    ) -> List[ValidationError]:
        errors: List[ValidationError] = []
        seen: Dict[str, int] = {}

        for node in ast.iter_child_nodes(tree):
            name: Optional[str] = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Allow overloaded functions (multiple defs with @overload)
                if self._has_overload_decorator(node):
                    continue
                name = node.name
            elif isinstance(node, ast.ClassDef):
                name = node.name
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id

            if name is not None:
                if name in seen:
                    errors.append(ValidationError(
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        message=f"Duplicate definition of '{name}' (first at line {seen[name]})",
                        severity="warning",
                    ))
                else:
                    seen[name] = getattr(node, "lineno", 0)

        return errors

    def _check_overload_consistency(
        self, tree: ast.Module, filename: str
    ) -> List[ValidationError]:
        errors: List[ValidationError] = []
        overloads: Dict[str, List[ast.FunctionDef]] = collections.defaultdict(list)
        impls: Dict[str, ast.FunctionDef] = {}

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._has_overload_decorator(node):
                    overloads[node.name].append(node)
                else:
                    if node.name in overloads:
                        impls[node.name] = node

        for name, ovs in overloads.items():
            if len(ovs) < 2:
                errors.append(ValidationError(
                    line=ovs[0].lineno if ovs else 0,
                    col=0,
                    message=f"Function '{name}' has only one @overload variant (need ≥2)",
                    severity="warning",
                ))
            if name not in impls:
                errors.append(ValidationError(
                    line=ovs[0].lineno if ovs else 0,
                    col=0,
                    message=f"Function '{name}' has @overload variants but no implementation stub",
                    severity="warning",
                ))

        return errors

    def _check_type_references(
        self, content: str, filename: str
    ) -> List[ValidationError]:
        errors: List[ValidationError] = []
        try:
            tree = ast.parse(content, filename=filename)
        except SyntaxError:
            return errors

        defined_names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                defined_names.add(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                defined_names.add(node.target.id)
            elif isinstance(node, ast.ImportFrom):
                if node.names:
                    for alias in node.names:
                        defined_names.add(alias.asname if alias.asname else alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    defined_names.add(alias.asname if alias.asname else alias.name)

        builtins_names = {
            "int", "str", "float", "bool", "bytes", "complex", "list", "dict",
            "set", "tuple", "frozenset", "type", "object", "None", "Ellipsis",
            "True", "False", "super", "property", "staticmethod", "classmethod",
            "print", "len", "range", "enumerate", "zip", "map", "filter",
            "isinstance", "issubclass", "hasattr", "getattr", "setattr",
            "TypeError", "ValueError", "KeyError", "IndexError", "RuntimeError",
            "Exception", "BaseException", "NotImplementedError", "StopIteration",
            "AttributeError", "ImportError", "OSError", "FileNotFoundError",
        }
        defined_names.update(builtins_names)

        return errors

    def _has_overload_decorator(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == "overload":
                return True
            if isinstance(dec, ast.Attribute) and dec.attr == "overload":
                return True
        return False


# ---------------------------------------------------------------------------
# StubMerger
# ---------------------------------------------------------------------------

@dataclass
class _MergeEntry:
    name: str
    kind: str  # "function", "class", "variable", "import", "other"
    source: str
    lineno: int
    is_manual: bool = False


@dataclass
class StubMerger:
    """Merges generated stubs with hand-written stubs."""

    strategy: ConflictStrategy = ConflictStrategy.PREFER_MANUAL

    def merge(self, existing: str, generated: str) -> str:
        existing_entries = self._parse_entries(existing)
        generated_entries = self._parse_entries(generated)

        existing_map: Dict[str, _MergeEntry] = {e.name: e for e in existing_entries}
        generated_map: Dict[str, _MergeEntry] = {e.name: e for e in generated_entries}

        all_names: List[str] = []
        seen: Set[str] = set()
        for e in existing_entries:
            if e.name not in seen:
                all_names.append(e.name)
                seen.add(e.name)
        for e in generated_entries:
            if e.name not in seen:
                all_names.append(e.name)
                seen.add(e.name)

        result_parts: List[str] = []
        for name in all_names:
            ex = existing_map.get(name)
            gen = generated_map.get(name)

            if ex and not gen:
                result_parts.append(ex.source)
            elif gen and not ex:
                result_parts.append(gen.source)
            elif ex and gen:
                result_parts.append(self._resolve_conflict(ex, gen))
            # both None shouldn't happen but skip if so

        return "\n\n".join(result_parts) + "\n"

    def merge_three_way(self, base: str, ours: str, theirs: str) -> str:
        base_lines = base.splitlines(keepends=True)
        ours_lines = ours.splitlines(keepends=True)
        theirs_lines = theirs.splitlines(keepends=True)

        # Use difflib to compute merge
        # First merge ours changes against base, then apply theirs changes
        d_ours = list(difflib.unified_diff(base_lines, ours_lines, n=0))
        d_theirs = list(difflib.unified_diff(base_lines, theirs_lines, n=0))

        # Simple strategy: merge by sections
        ours_entries = self._parse_entries(ours)
        theirs_entries = self._parse_entries(theirs)
        base_entries = self._parse_entries(base)

        base_map = {e.name: e for e in base_entries}
        ours_map = {e.name: e for e in ours_entries}
        theirs_map = {e.name: e for e in theirs_entries}

        all_names: List[str] = []
        seen: Set[str] = set()
        for entry_list in [base_entries, ours_entries, theirs_entries]:
            for e in entry_list:
                if e.name not in seen:
                    all_names.append(e.name)
                    seen.add(e.name)

        result_parts: List[str] = []
        for name in all_names:
            b = base_map.get(name)
            o = ours_map.get(name)
            t = theirs_map.get(name)

            if o and not t:
                # Deleted in theirs
                if b and o.source == b.source:
                    continue  # Accept deletion
                else:
                    result_parts.append(o.source)  # Keep our changes
            elif t and not o:
                if b and t.source == b.source:
                    continue
                else:
                    result_parts.append(t.source)
            elif o and t:
                if o.source == t.source:
                    result_parts.append(o.source)
                elif b and o.source == b.source:
                    result_parts.append(t.source)  # Take theirs changes
                elif b and t.source == b.source:
                    result_parts.append(o.source)  # Take our changes
                else:
                    # Both changed — use conflict strategy
                    merged_entry = _MergeEntry(
                        name=name, kind=o.kind, source=o.source,
                        lineno=o.lineno, is_manual=o.is_manual,
                    )
                    theirs_entry = _MergeEntry(
                        name=name, kind=t.kind, source=t.source,
                        lineno=t.lineno, is_manual=t.is_manual,
                    )
                    result_parts.append(self._resolve_conflict(merged_entry, theirs_entry))

        return "\n\n".join(result_parts) + "\n"

    def _resolve_conflict(self, existing: _MergeEntry, generated: _MergeEntry) -> str:
        if existing.is_manual:
            if self.strategy == ConflictStrategy.PREFER_MANUAL:
                return existing.source
            elif self.strategy == ConflictStrategy.PREFER_INFERRED:
                return generated.source
            else:
                return self._union_entries(existing, generated)
        if self.strategy == ConflictStrategy.PREFER_MANUAL:
            return existing.source
        if self.strategy == ConflictStrategy.PREFER_INFERRED:
            return generated.source
        return self._union_entries(existing, generated)

    def _union_entries(self, a: _MergeEntry, b: _MergeEntry) -> str:
        if a.kind == "function" and b.kind == "function":
            return f"@overload\n{a.source}\n@overload\n{b.source}"
        return b.source

    def _parse_entries(self, content: str) -> List[_MergeEntry]:
        entries: List[_MergeEntry] = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            if content.strip():
                entries.append(_MergeEntry(
                    name="__unparseable__", kind="other",
                    source=content, lineno=0,
                ))
            return entries

        lines = content.splitlines()

        for node in ast.iter_child_nodes(tree):
            name = self._node_name(node)
            if name is None:
                continue

            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            # Capture decorators
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.decorator_list:
                    dec_start = min(d.lineno for d in node.decorator_list) - 1
                    start = min(start, dec_start)

            source_lines = lines[start:end]
            source = "\n".join(source_lines)

            is_manual = any("# manual" in sl for sl in source_lines)

            kind = "other"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "function"
            elif isinstance(node, ast.ClassDef):
                kind = "class"
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                kind = "variable"
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                kind = "import"

            entries.append(_MergeEntry(
                name=name, kind=kind, source=source,
                lineno=node.lineno, is_manual=is_manual,
            ))

        return entries

    def _node_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    return target.id
            return None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return node.target.id
        if isinstance(node, ast.ImportFrom):
            return f"__import_{node.module or 'unknown'}__"
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
            return f"__import_{'_'.join(names)}__"
        return None


# ---------------------------------------------------------------------------
# IncrementalStubUpdater
# ---------------------------------------------------------------------------

@dataclass
class IncrementalStubUpdater:
    """Updates only changed stubs using content fingerprinting."""

    output_dir: Path = field(default_factory=lambda: Path("."))
    writer: StubFileWriter = field(default_factory=StubFileWriter)
    _fingerprints: Dict[str, str] = field(default_factory=dict, repr=False)
    _fingerprint_file: Optional[Path] = None

    def __post_init__(self) -> None:
        self._fingerprint_file = self.output_dir / ".stub_fingerprints.json"
        self._load_fingerprints()

    def needs_update(self, module_path: str, new_content: str) -> bool:
        new_fp = self._compute_fingerprint(new_content)
        old_fp = self._fingerprints.get(module_path)
        return old_fp != new_fp

    def update(self, module_path: str, new_content: str) -> bool:
        if not self.needs_update(module_path, new_content):
            return False
        stub_path = self._stub_path(module_path)
        self.writer.write(stub_path, new_content)
        self._fingerprints[module_path] = self._compute_fingerprint(new_content)
        self._save_fingerprints()
        return True

    def update_if_changed(
        self, module_path: str, new_content: str, merger: Optional[StubMerger] = None
    ) -> bool:
        stub_path = self._stub_path(module_path)
        if stub_path.exists() and merger:
            existing = stub_path.read_text(encoding="utf-8")
            merged = merger.merge(existing, new_content)
            return self.update(module_path, merged)
        return self.update(module_path, new_content)

    def get_changed_functions(
        self, module_path: str, old_content: str, new_content: str
    ) -> List[str]:
        old_sigs = self._extract_signatures(old_content)
        new_sigs = self._extract_signatures(new_content)

        changed: List[str] = []
        all_names = set(old_sigs.keys()) | set(new_sigs.keys())
        for name in sorted(all_names):
            old_fp = old_sigs.get(name)
            new_fp = new_sigs.get(name)
            if old_fp != new_fp:
                changed.append(name)
        return changed

    def clear_fingerprints(self) -> None:
        self._fingerprints.clear()
        if self._fingerprint_file and self._fingerprint_file.exists():
            self._fingerprint_file.unlink()

    def get_fingerprint(self, module_path: str) -> Optional[str]:
        return self._fingerprints.get(module_path)

    def _compute_fingerprint(self, content: str) -> str:
        normalized = re.sub(r"\s+", " ", content.strip())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _stub_path(self, module_path: str) -> Path:
        parts = module_path.replace(".", os.sep)
        return self.output_dir / (parts + ".pyi")

    def _extract_signatures(self, content: str) -> Dict[str, str]:
        sigs: Dict[str, str] = {}
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return sigs

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig_str = ast.dump(node.args)
                if node.returns:
                    sig_str += " -> " + ast.dump(node.returns)
                sigs[node.name] = hashlib.md5(sig_str.encode()).hexdigest()[:12]
            elif isinstance(node, ast.ClassDef):
                class_src = ast.dump(node)
                sigs[node.name] = hashlib.md5(class_src.encode()).hexdigest()[:12]
        return sigs

    def _load_fingerprints(self) -> None:
        if self._fingerprint_file and self._fingerprint_file.exists():
            try:
                data = json.loads(self._fingerprint_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._fingerprints = data
            except (json.JSONDecodeError, OSError):
                self._fingerprints = {}

    def _save_fingerprints(self) -> None:
        if self._fingerprint_file:
            self._fingerprint_file.parent.mkdir(parents=True, exist_ok=True)
            self._fingerprint_file.write_text(
                json.dumps(self._fingerprints, indent=2),
                encoding="utf-8",
            )


# ---------------------------------------------------------------------------
# PyiGenerator — main orchestrator
# ---------------------------------------------------------------------------

@dataclass
class PyiGenerator:
    """Main orchestrator: takes analysis results and generates .pyi stubs."""

    target_version: Tuple[int, int] = (3, 10)
    output_dir: Path = field(default_factory=lambda: Path("out"))
    docstring_style: DocstringStyle = DocstringStyle.GOOGLE
    generate_docstrings: bool = True
    use_future_annotations: bool = True
    max_line_length: int = 88
    atomic_writes: bool = True
    incremental: bool = True

    _formatter: TypeFormatter = field(init=False, repr=False)
    _imports: ImportResolver = field(init=False, repr=False)
    _docstring_gen: DocstringGenerator = field(init=False, repr=False)
    _func_gen: FunctionStubGenerator = field(init=False, repr=False)
    _class_gen: ClassStubGenerator = field(init=False, repr=False)
    _module_gen: ModuleStubGenerator = field(init=False, repr=False)
    _writer: StubFileWriter = field(init=False, repr=False)
    _validator: StubValidator = field(init=False, repr=False)
    _updater: IncrementalStubUpdater = field(init=False, repr=False)
    _merger: StubMerger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._formatter = TypeFormatter(
            target_version=self.target_version,
        )
        self._imports = ImportResolver(
            target_version=self.target_version,
            use_future_annotations=self.use_future_annotations,
        )
        self._docstring_gen = DocstringGenerator(
            style=self.docstring_style,
            max_line_length=self.max_line_length,
        )
        self._func_gen = FunctionStubGenerator(
            formatter=self._formatter,
            imports=self._imports,
            docstring_gen=self._docstring_gen,
            generate_docstrings=self.generate_docstrings,
        )
        self._class_gen = ClassStubGenerator(
            formatter=self._formatter,
            imports=self._imports,
            func_gen=self._func_gen,
            docstring_gen=self._docstring_gen,
        )
        self._module_gen = ModuleStubGenerator(
            formatter=self._formatter,
            imports=self._imports,
            func_gen=self._func_gen,
            class_gen=self._class_gen,
            docstring_gen=self._docstring_gen,
        )
        self._writer = StubFileWriter(
            max_line_length=self.max_line_length,
            atomic=self.atomic_writes,
        )
        self._validator = StubValidator(target_version=self.target_version)
        self._updater = IncrementalStubUpdater(
            output_dir=self.output_dir,
            writer=self._writer,
        )
        self._merger = StubMerger()

    def generate_stubs(
        self,
        analysis: Dict[str, Any],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Generate .pyi stubs for all modules in the analysis result.

        Args:
            analysis: Dict mapping module paths to module analysis info.
            progress_callback: Optional ``(module_path, current, total)`` callback.

        Returns:
            Summary dict with counts and any validation errors.
        """
        modules = analysis.get("modules", {})
        total = len(modules)
        generated = 0
        skipped = 0
        errors_map: Dict[str, List[ValidationError]] = {}

        for idx, (mod_path, mod_info) in enumerate(modules.items()):
            if progress_callback:
                progress_callback(mod_path, idx + 1, total)

            try:
                content = self.generate_module_stub(mod_info)
                validation_errors = self._validator.validate(content, filename=mod_path)

                if validation_errors:
                    error_only = [e for e in validation_errors if e.severity == "error"]
                    if error_only:
                        errors_map[mod_path] = validation_errors
                        continue

                if self.incremental:
                    updated = self._updater.update_if_changed(
                        mod_path, content, self._merger
                    )
                    if updated:
                        generated += 1
                    else:
                        skipped += 1
                else:
                    stub_path = self._stub_path(mod_path)
                    self._writer.write(stub_path, content)
                    generated += 1
            except Exception as exc:
                errors_map[mod_path] = [ValidationError(
                    line=0, col=0,
                    message=f"Generation failed: {exc}",
                )]

        return {
            "total": total,
            "generated": generated,
            "skipped": skipped,
            "errors": {k: [str(e) for e in v] for k, v in errors_map.items()},
        }

    def generate_module_stub(self, module_info: Dict[str, Any]) -> str:
        """Generate a single module stub from module analysis info."""
        self._imports.clear()
        content = self._module_gen.generate(module_info)
        return content

    def generate_module_stub_to_file(
        self, module_info: Dict[str, Any], output_path: Path
    ) -> List[ValidationError]:
        content = self.generate_module_stub(module_info)
        errors = self._validator.validate(content, filename=str(output_path))
        error_only = [e for e in errors if e.severity == "error"]
        if not error_only:
            self._writer.write(output_path, content)
        return errors

    def validate_stub(self, content: str, filename: str = "<stub>") -> List[ValidationError]:
        return self._validator.validate(content, filename)

    def merge_stubs(self, existing: str, generated: str) -> str:
        return self._merger.merge(existing, generated)

    def format_type(self, type_desc: Dict[str, Any]) -> str:
        return self._formatter.format_type(type_desc)

    def _stub_path(self, module_path: str) -> Path:
        parts = module_path.replace(".", os.sep)
        return self.output_dir / (parts + ".pyi")


# ---------------------------------------------------------------------------
# Refinement-annotation factory helpers
# ---------------------------------------------------------------------------

def parse_refinement(spec: Dict[str, Any]) -> RefinementAnnotation:
    """Create a ``RefinementAnnotation`` from a dict specification.

    Supported keys: ``kind`` (str), plus kind-specific parameters.
    """
    kind = spec.get("kind", "")
    if kind == "gt":
        return Gt(value=spec.get("value", 0))
    if kind == "lt":
        return Lt(value=spec.get("value", 0))
    if kind == "ge":
        return Ge(value=spec.get("value", 0))
    if kind == "le":
        return Le(value=spec.get("value", 0))
    if kind == "eq":
        return Eq(value=spec.get("value"))
    if kind == "ne":
        return Ne(value=spec.get("value"))
    if kind == "non_none":
        return NonNone()
    if kind == "isinstance":
        return IsInstance(type_name=spec.get("type_name", "object"))
    if kind == "has_attr":
        return HasAttr(attr_name=spec.get("attr_name", ""))
    if kind == "len":
        return Len(operator=spec.get("operator", "=="), length=spec.get("length", 0))
    if kind == "between":
        return Between(
            low=spec.get("low", 0),
            high=spec.get("high", 0),
            inclusive_low=spec.get("inclusive_low", True),
            inclusive_high=spec.get("inclusive_high", True),
        )
    if kind == "non_empty":
        return NonEmpty()
    if kind == "positive":
        return Positive()
    if kind == "non_negative":
        return NonNegative()
    if kind == "non_zero":
        return NonZero()
    raise ValueError(f"Unknown refinement kind: {kind!r}")


def parse_refinements(specs: List[Dict[str, Any]]) -> List[RefinementAnnotation]:
    """Parse a list of refinement specs into annotation objects."""
    return [parse_refinement(s) for s in specs]


# ---------------------------------------------------------------------------
# Module-info construction helpers
# ---------------------------------------------------------------------------

def make_basic_type(name: str) -> Dict[str, Any]:
    """Shorthand for ``{"kind": "basic", "name": name}``."""
    return {"kind": "basic", "name": name}


def make_container_type(
    container: str, *args: Dict[str, Any]
) -> Dict[str, Any]:
    return {"kind": "container", "container": container, "args": list(args)}


def make_union_type(*members: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": "union", "members": list(members)}


def make_optional_type(inner: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": "optional", "inner": inner}


def make_callable_type(
    params: List[Dict[str, Any]], return_type: Dict[str, Any]
) -> Dict[str, Any]:
    return {"kind": "callable", "params": params, "return_type": return_type}


def make_literal_type(*values: Any) -> Dict[str, Any]:
    return {"kind": "literal", "values": list(values)}


def make_annotated_type(
    base: Dict[str, Any], refinements: List[RefinementAnnotation]
) -> Dict[str, Any]:
    return {"kind": "annotated", "base": base, "refinements": refinements}


def make_param(
    name: str,
    type_desc: Optional[Dict[str, Any]] = None,
    default: Any = None,
    kind: str = "positional_or_keyword",
) -> Dict[str, Any]:
    p: Dict[str, Any] = {"name": name, "kind": kind}
    if type_desc is not None:
        p["type"] = type_desc
    if default is not None:
        p["default"] = default
    return p


def make_function(
    name: str,
    params: Optional[List[Dict[str, Any]]] = None,
    return_type: Optional[Dict[str, Any]] = None,
    is_async: bool = False,
    decorators: Optional[List[str]] = None,
    overloads: Optional[List[Dict[str, Any]]] = None,
    docstring: str = "",
    refinements: Optional[Dict[str, List[RefinementAnnotation]]] = None,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {"name": name}
    if params is not None:
        info["params"] = params
    if return_type is not None:
        info["return_type"] = return_type
    if is_async:
        info["is_async"] = True
    if decorators:
        info["decorators"] = decorators
    if overloads:
        info["overloads"] = overloads
    if docstring:
        info["docstring"] = docstring
    if refinements:
        info["refinements"] = refinements
    return info


def make_class(
    name: str,
    bases: Optional[List[str]] = None,
    methods: Optional[List[Dict[str, Any]]] = None,
    class_vars: Optional[List[Dict[str, Any]]] = None,
    instance_vars: Optional[List[Dict[str, Any]]] = None,
    class_kind: str = "regular",
    decorators: Optional[List[str]] = None,
    docstring: str = "",
    slots: Optional[List[str]] = None,
    nested_classes: Optional[List[Dict[str, Any]]] = None,
    generic_params: Optional[List[str]] = None,
    fields: Optional[Any] = None,
    members: Optional[Dict[str, Any]] = None,
    total: bool = True,
    runtime_checkable: bool = False,
    frozen: bool = False,
    eq: bool = True,
    order: bool = False,
    enum_base: str = "Enum",
) -> Dict[str, Any]:
    info: Dict[str, Any] = {"name": name, "class_kind": class_kind}
    if bases:
        info["bases"] = bases
    if methods:
        info["methods"] = methods
    if class_vars:
        info["class_vars"] = class_vars
    if instance_vars:
        info["instance_vars"] = instance_vars
    if decorators:
        info["decorators"] = decorators
    if docstring:
        info["docstring"] = docstring
    if slots is not None:
        info["slots"] = slots
    if nested_classes:
        info["nested_classes"] = nested_classes
    if generic_params:
        info["generic_params"] = generic_params
    if fields is not None:
        info["fields"] = fields
    if members is not None:
        info["members"] = members
    info["total"] = total
    info["runtime_checkable"] = runtime_checkable
    info["frozen"] = frozen
    info["eq"] = eq
    info["order"] = order
    info["enum_base"] = enum_base
    return info


def make_module(
    docstring: str = "",
    constants: Optional[List[Dict[str, Any]]] = None,
    variables: Optional[List[Dict[str, Any]]] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
    classes: Optional[List[Dict[str, Any]]] = None,
    all_exports: Optional[List[str]] = None,
    type_checking: Optional[List[Dict[str, Any]]] = None,
    type_aliases: Optional[List[Dict[str, Any]]] = None,
    typevars: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if docstring:
        info["docstring"] = docstring
    if constants:
        info["constants"] = constants
    if variables:
        info["variables"] = variables
    if functions:
        info["functions"] = functions
    if classes:
        info["classes"] = classes
    if all_exports is not None:
        info["__all__"] = all_exports
    if type_checking:
        info["type_checking"] = type_checking
    if type_aliases:
        info["type_aliases"] = type_aliases
    if typevars:
        info["typevars"] = typevars
    return info


# ---------------------------------------------------------------------------
# Batch / CLI helpers
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Result of a batch stub-generation run."""
    total: int = 0
    generated: int = 0
    skipped: int = 0
    failed: int = 0
    errors: Dict[str, List[str]] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"Total: {self.total}, Generated: {self.generated}, "
            f"Skipped: {self.skipped}, Failed: {self.failed}, "
            f"Duration: {self.duration_seconds:.2f}s"
        )


def generate_stubs_batch(
    analysis_files: List[Path],
    output_dir: Path,
    target_version: Tuple[int, int] = (3, 10),
    docstring_style: DocstringStyle = DocstringStyle.GOOGLE,
    incremental: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> BatchResult:
    """Generate stubs from multiple analysis JSON files."""
    gen = PyiGenerator(
        target_version=target_version,
        output_dir=output_dir,
        docstring_style=docstring_style,
        incremental=incremental,
    )

    result = BatchResult()
    start = datetime.datetime.now()

    all_modules: Dict[str, Any] = {}
    for af in analysis_files:
        try:
            data = json.loads(af.read_text(encoding="utf-8"))
            modules = data.get("modules", {})
            all_modules.update(modules)
        except (json.JSONDecodeError, OSError) as exc:
            result.errors[str(af)] = [f"Failed to load: {exc}"]
            result.failed += 1

    result.total = len(all_modules) + result.failed

    if all_modules:
        summary = gen.generate_stubs(
            {"modules": all_modules},
            progress_callback=progress_callback,
        )
        result.generated = summary["generated"]
        result.skipped = summary["skipped"]
        for mod, errs in summary["errors"].items():
            result.errors[mod] = errs
            result.failed += 1

    end = datetime.datetime.now()
    result.duration_seconds = (end - start).total_seconds()
    return result


# ---------------------------------------------------------------------------
# Source-file analysis helper (lightweight)
# ---------------------------------------------------------------------------

@dataclass
class SourceAnalyzer:
    """Lightweight AST-based source analyser that extracts type-stub-worthy info."""

    def analyze_source(self, source: str, module_name: str = "<module>") -> Dict[str, Any]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return make_module()

        functions: List[Dict[str, Any]] = []
        classes: List[Dict[str, Any]] = []
        variables: List[Dict[str, Any]] = []
        constants: List[Dict[str, Any]] = []
        all_exports: Optional[List[str]] = None

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(self._analyze_function(node))
            elif isinstance(node, ast.ClassDef):
                classes.append(self._analyze_class(node))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name == "__all__":
                            all_exports = self._extract_all(node.value)
                        elif name.isupper():
                            constants.append({"name": name, "type": self._infer_assign_type(node.value)})
                        else:
                            variables.append({"name": name, "type": self._infer_assign_type(node.value)})
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                ann_type = self._annotation_to_type(node.annotation)
                if name.isupper():
                    constants.append({"name": name, "type": ann_type})
                else:
                    variables.append({"name": name, "type": ann_type})

        docstring = ast.get_docstring(tree) or ""

        return make_module(
            docstring=docstring,
            constants=constants if constants else None,
            variables=variables if variables else None,
            functions=functions if functions else None,
            classes=classes if classes else None,
            all_exports=all_exports,
        )

    def analyze_file(self, path: Path) -> Dict[str, Any]:
        source = path.read_text(encoding="utf-8")
        module_name = path.stem
        return self.analyze_source(source, module_name)

    def _analyze_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Dict[str, Any]:
        params: List[Dict[str, Any]] = []

        # positional-only
        for arg in getattr(node.args, "posonlyargs", []):
            params.append(self._analyze_arg(arg, "positional_only"))

        # regular args
        num_args = len(node.args.args)
        num_defaults = len(node.args.defaults)
        for i, arg in enumerate(node.args.args):
            p = self._analyze_arg(arg, "positional_or_keyword")
            default_idx = i - (num_args - num_defaults)
            if default_idx >= 0:
                p["default"] = self._const_value(node.args.defaults[default_idx])
            params.append(p)

        if node.args.vararg:
            params.append(self._analyze_arg(node.args.vararg, "var_positional"))

        # keyword-only
        for i, arg in enumerate(node.args.kwonlyargs):
            p = self._analyze_arg(arg, "keyword_only")
            if i < len(node.args.kw_defaults) and node.args.kw_defaults[i] is not None:
                p["default"] = self._const_value(node.args.kw_defaults[i])
            params.append(p)

        if node.args.kwarg:
            params.append(self._analyze_arg(node.args.kwarg, "var_keyword"))

        return_type = None
        if node.returns:
            return_type = self._annotation_to_type(node.returns)

        decorators: List[str] = []
        for dec in node.decorator_list:
            decorators.append(self._decorator_str(dec))

        is_async = isinstance(node, ast.AsyncFunctionDef)

        return make_function(
            name=node.name,
            params=params,
            return_type=return_type,
            is_async=is_async,
            decorators=decorators if decorators else None,
            docstring=ast.get_docstring(node) or "",
        )

    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        bases: List[str] = []
        for base in node.bases:
            bases.append(self._expr_str(base))

        methods: List[Dict[str, Any]] = []
        class_vars: List[Dict[str, Any]] = []
        instance_vars: List[Dict[str, Any]] = []
        nested_classes: List[Dict[str, Any]] = []
        slots: Optional[List[str]] = None

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = self._analyze_function(child)
                # Extract instance vars from __init__
                if child.name == "__init__":
                    instance_vars.extend(self._extract_init_vars(child))
                methods.append(method_info)
            elif isinstance(child, ast.ClassDef):
                nested_classes.append(self._analyze_class(child))
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "__slots__":
                            slots = self._extract_slots(child.value)
                        else:
                            class_vars.append({
                                "name": target.id,
                                "type": self._infer_assign_type(child.value),
                                "value": self._const_value(child.value),
                            })
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                ann_type = self._annotation_to_type(child.annotation)
                class_vars.append({
                    "name": child.target.id,
                    "type": ann_type,
                })

        decorators: List[str] = []
        for dec in node.decorator_list:
            decorators.append(self._decorator_str(dec))

        class_kind = "regular"
        if any(d == "dataclass" or d.startswith("dataclass(") for d in decorators):
            class_kind = "dataclass"
        elif any(b in ("Protocol",) for b in bases):
            class_kind = "protocol"
        elif any(b in ("TypedDict",) for b in bases):
            class_kind = "typeddict"
        elif any(b in ("NamedTuple",) for b in bases):
            class_kind = "namedtuple"
        elif any(b in ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag") for b in bases):
            class_kind = "enum"

        return make_class(
            name=node.name,
            bases=bases if bases else None,
            methods=methods if methods else None,
            class_vars=class_vars if class_vars else None,
            instance_vars=instance_vars if instance_vars else None,
            class_kind=class_kind,
            decorators=decorators if decorators else None,
            docstring=ast.get_docstring(node) or "",
            slots=slots,
            nested_classes=nested_classes if nested_classes else None,
        )

    def _analyze_arg(self, arg: ast.arg, kind: str) -> Dict[str, Any]:
        p: Dict[str, Any] = {"name": arg.arg, "kind": kind}
        if arg.annotation:
            p["type"] = self._annotation_to_type(arg.annotation)
        return p

    def _annotation_to_type(self, node: ast.expr) -> Dict[str, Any]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return make_basic_type(node.value)
        if isinstance(node, ast.Name):
            return make_basic_type(node.id)
        if isinstance(node, ast.Attribute):
            return make_basic_type(self._expr_str(node))
        if isinstance(node, ast.Subscript):
            base = self._expr_str(node.value)
            slice_node = node.slice
            if base in ("List", "Set", "FrozenSet", "Deque", "list", "set", "frozenset"):
                inner = self._annotation_to_type(slice_node)
                return make_container_type(base.capitalize() if base[0].islower() else base, inner)
            if base in ("Dict", "dict"):
                if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2:
                    k = self._annotation_to_type(slice_node.elts[0])
                    v = self._annotation_to_type(slice_node.elts[1])
                    return make_container_type("Dict", k, v)
            if base in ("Optional",):
                inner = self._annotation_to_type(slice_node)
                return make_optional_type(inner)
            if base in ("Union",):
                if isinstance(slice_node, ast.Tuple):
                    members = [self._annotation_to_type(e) for e in slice_node.elts]
                    return make_union_type(*members)
                return self._annotation_to_type(slice_node)
            if base in ("Tuple", "tuple"):
                if isinstance(slice_node, ast.Tuple):
                    args = [self._annotation_to_type(e) for e in slice_node.elts]
                    return make_container_type("Tuple", *args)
                inner = self._annotation_to_type(slice_node)
                return make_container_type("Tuple", inner)
            if base in ("Callable",):
                if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2:
                    params_node = slice_node.elts[0]
                    ret_node = slice_node.elts[1]
                    plist: List[Dict[str, Any]] = []
                    if isinstance(params_node, ast.List):
                        plist = [self._annotation_to_type(e) for e in params_node.elts]
                    return make_callable_type(plist, self._annotation_to_type(ret_node))
            if base in ("ClassVar",):
                inner = self._annotation_to_type(slice_node)
                return {"kind": "classvar", "inner": inner}
            if base in ("Final",):
                inner = self._annotation_to_type(slice_node)
                return {"kind": "final", "inner": inner}
            if base in ("Annotated",):
                if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) >= 1:
                    base_type = self._annotation_to_type(slice_node.elts[0])
                    return {"kind": "annotated", "base": base_type, "refinements": []}
            if base in ("Literal",):
                if isinstance(slice_node, ast.Tuple):
                    vals = [self._const_value(e) for e in slice_node.elts]
                    return make_literal_type(*vals)
                return make_literal_type(self._const_value(slice_node))
            # Generic subscript
            if isinstance(slice_node, ast.Tuple):
                params = [self._annotation_to_type(e) for e in slice_node.elts]
                return {"kind": "generic", "base": base, "params": params}
            return {"kind": "generic", "base": base, "params": [self._annotation_to_type(slice_node)]}
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self._annotation_to_type(node.left)
            right = self._annotation_to_type(node.right)
            return make_union_type(left, right)
        return make_basic_type("Any")

    def _infer_assign_type(self, node: ast.expr) -> Dict[str, Any]:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return make_basic_type("int")
            if isinstance(node.value, float):
                return make_basic_type("float")
            if isinstance(node.value, str):
                return make_basic_type("str")
            if isinstance(node.value, bool):
                return make_basic_type("bool")
            if isinstance(node.value, bytes):
                return make_basic_type("bytes")
            if node.value is None:
                return make_optional_type(make_basic_type("Any"))
        if isinstance(node, ast.List):
            return make_container_type("List")
        if isinstance(node, ast.Dict):
            return make_container_type("Dict")
        if isinstance(node, ast.Set):
            return make_container_type("Set")
        if isinstance(node, ast.Tuple):
            return make_container_type("Tuple")
        return make_basic_type("Any")

    def _extract_init_vars(self, func: ast.FunctionDef) -> List[Dict[str, Any]]:
        vars_list: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        attr_name = target.attr
                        if attr_name not in seen:
                            seen.add(attr_name)
                            vars_list.append({
                                "name": attr_name,
                                "type": self._infer_assign_type(node.value),
                            })
            elif isinstance(node, ast.AnnAssign):
                if (
                    isinstance(node.target, ast.Attribute)
                    and isinstance(node.target.value, ast.Name)
                    and node.target.value.id == "self"
                ):
                    attr_name = node.target.attr
                    if attr_name not in seen:
                        seen.add(attr_name)
                        vars_list.append({
                            "name": attr_name,
                            "type": self._annotation_to_type(node.annotation),
                        })
        return vars_list

    def _extract_all(self, node: ast.expr) -> Optional[List[str]]:
        if isinstance(node, (ast.List, ast.Tuple)):
            result: List[str] = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    result.append(elt.value)
            return result
        return None

    def _extract_slots(self, node: ast.expr) -> Optional[List[str]]:
        if isinstance(node, (ast.List, ast.Tuple)):
            result: List[str] = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    result.append(elt.value)
            return result
        return None

    def _const_value(self, node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = self._const_value(node.operand)
            if isinstance(val, (int, float)):
                return -val
        if isinstance(node, ast.List):
            return [self._const_value(e) for e in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._const_value(e) for e in node.elts)
        if isinstance(node, ast.Dict):
            keys = [self._const_value(k) if k else None for k in node.keys]
            vals = [self._const_value(v) for v in node.values]
            return dict(zip(keys, vals))
        return None

    def _expr_str(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._expr_str(node.value)}.{node.attr}"
        if isinstance(node, ast.Constant):
            return repr(node.value)
        return "..."

    def _decorator_str(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self._expr_str(node)
        if isinstance(node, ast.Call):
            func_str = self._expr_str(node.func)
            args: List[str] = []
            for a in node.args:
                args.append(self._expr_str(a))
            for kw in node.keywords:
                if kw.arg:
                    args.append(f"{kw.arg}={self._expr_str(kw.value)}")
                else:
                    args.append(f"**{self._expr_str(kw.value)}")
            return f"{func_str}({', '.join(args)})"
        return "..."


# ---------------------------------------------------------------------------
# Utility: diff two stubs
# ---------------------------------------------------------------------------

def diff_stubs(old: str, new: str, filename: str = "stub.pyi") -> str:
    """Return a unified diff between *old* and *new* stub content."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )
    return "".join(diff)


# ---------------------------------------------------------------------------
# Utility: stub statistics
# ---------------------------------------------------------------------------

@dataclass
class StubStats:
    num_functions: int = 0
    num_classes: int = 0
    num_variables: int = 0
    num_overloads: int = 0
    num_type_aliases: int = 0
    num_lines: int = 0

    def summary(self) -> str:
        return (
            f"Functions: {self.num_functions}, Classes: {self.num_classes}, "
            f"Variables: {self.num_variables}, Overloads: {self.num_overloads}, "
            f"Type aliases: {self.num_type_aliases}, Lines: {self.num_lines}"
        )


def compute_stub_stats(content: str) -> StubStats:
    """Compute statistics about a generated stub."""
    stats = StubStats(num_lines=content.count("\n") + 1)
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return stats

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(
                isinstance(d, ast.Name) and d.id == "overload"
                for d in node.decorator_list
            ):
                stats.num_overloads += 1
            else:
                stats.num_functions += 1
        elif isinstance(node, ast.ClassDef):
            stats.num_classes += 1
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            stats.num_variables += 1
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    stats.num_variables += 1

    return stats


# ---------------------------------------------------------------------------
# Utility: format check
# ---------------------------------------------------------------------------

def check_stub_format(content: str, max_line_length: int = 88) -> List[str]:
    """Return a list of formatting issues found in the stub content."""
    issues: List[str] = []
    lines = content.splitlines()
    for i, line in enumerate(lines, 1):
        if len(line) > max_line_length:
            issues.append(f"Line {i}: exceeds {max_line_length} chars ({len(line)})")
        if line.rstrip() != line:
            issues.append(f"Line {i}: trailing whitespace")
        if "\t" in line:
            issues.append(f"Line {i}: contains tab character")

    # Check blank line counts between top-level definitions
    prev_was_def = False
    blank_count = 0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            blank_count += 1
            continue
        is_def = stripped.startswith(("def ", "async def ", "class ", "@"))
        indent_level = len(line) - len(line.lstrip())
        if is_def and indent_level == 0 and prev_was_def and blank_count < 2:
            issues.append(
                f"Line {i}: expected 2 blank lines before top-level definition, got {blank_count}"
            )
        blank_count = 0
        if is_def and indent_level == 0:
            prev_was_def = True
        elif indent_level == 0 and not stripped.startswith("#"):
            prev_was_def = True

    return issues


# ---------------------------------------------------------------------------
# Entrypoint helper for CLI usage
# ---------------------------------------------------------------------------

def run_cli(args: List[str]) -> int:
    """Minimal CLI entrypoint for stub generation.

    Usage::

        python pyi_generator.py <source.py> [--output-dir <dir>] [--target <3.X>]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate .pyi stubs from Python sources")
    parser.add_argument("sources", nargs="+", help="Python source files to analyze")
    parser.add_argument("--output-dir", "-o", default="out", help="Output directory")
    parser.add_argument("--target", default="3.10", help="Target Python version (e.g. 3.10)")
    parser.add_argument("--style", choices=["google", "numpy", "rest"], default="google")
    parser.add_argument("--no-docstrings", action="store_true")
    parser.add_argument("--no-incremental", action="store_true")
    parser.add_argument("--max-line-length", type=int, default=88)
    parser.add_argument("--validate-only", action="store_true")

    parsed = parser.parse_args(args)

    version_parts = parsed.target.split(".")
    target_ver = (int(version_parts[0]), int(version_parts[1]))

    style_map = {
        "google": DocstringStyle.GOOGLE,
        "numpy": DocstringStyle.NUMPY,
        "rest": DocstringStyle.REST,
    }

    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = SourceAnalyzer()
    gen = PyiGenerator(
        target_version=target_ver,
        output_dir=output_dir,
        docstring_style=style_map[parsed.style],
        generate_docstrings=not parsed.no_docstrings,
        incremental=not parsed.no_incremental,
        max_line_length=parsed.max_line_length,
    )

    exit_code = 0
    for source_path_str in parsed.sources:
        source_path = Path(source_path_str)
        if not source_path.exists():
            print(f"Error: {source_path} does not exist")
            exit_code = 1
            continue

        module_info = analyzer.analyze_file(source_path)
        stub_content = gen.generate_module_stub(module_info)

        if parsed.validate_only:
            errors = gen.validate_stub(stub_content, filename=str(source_path))
            if errors:
                for e in errors:
                    print(f"{source_path}: {e}")
                exit_code = 1
            else:
                print(f"{source_path}: OK")
        else:
            stub_path = output_dir / source_path.with_suffix(".pyi").name
            errors = gen.generate_module_stub_to_file(module_info, stub_path)
            if errors:
                for e in errors:
                    print(f"{source_path}: {e}")
                exit_code = 1
            else:
                print(f"Generated: {stub_path}")

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    import sys as _sys
    _sys.exit(run_cli(_sys.argv[1:]))
