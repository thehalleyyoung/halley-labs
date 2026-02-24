from __future__ import annotations

"""
Type annotation parsing and interpretation for refinement type inference.

Handles PEP 484-681 type annotation features, converting Python type
annotations into an internal representation suitable for refinement type
inference in dynamically-typed languages using counterexample-guided contract discovery.
"""

import ast
import copy
import enum
import hashlib
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# 1. InternalType – core representation
# ---------------------------------------------------------------------------

@dataclass
class InternalType:
    """Core internal representation used throughout the pipeline.

    ``kind`` acts as a discriminated-union tag.  Valid values:
        basic, union, callable, generic, literal, tuple, protocol,
        typeguard, never, self_type, annotated, typevar, typevar_tuple,
        paramspec, namedtuple, typeddict, dataclass_type, enum_type,
        overloaded, intersection, unpack, concatenate, readonly,
        type_alias, any, none
    """

    kind: str = "basic"
    name: str = ""
    args: List[InternalType] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    nullable: bool = False
    literal_values: Optional[List[Any]] = None
    is_final: bool = False
    is_classvar: bool = False
    required: Optional[bool] = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def is_none_type(self) -> bool:
        return self.kind == "none" or (self.kind == "basic" and self.name in ("None", "NoneType"))

    def is_any(self) -> bool:
        return self.kind == "any" or (self.kind == "basic" and self.name == "Any")

    def is_never(self) -> bool:
        return self.kind == "never"

    def flatten_union(self) -> List[InternalType]:
        """Recursively flatten nested unions into a single list."""
        if self.kind != "union":
            return [self]
        result: List[InternalType] = []
        for arg in self.args:
            result.extend(arg.flatten_union())
        return result

    def substitute(self, mapping: Dict[str, InternalType]) -> InternalType:
        """Return a copy with type-variables replaced according to *mapping*."""
        if self.kind == "typevar" and self.name in mapping:
            replacement = copy.deepcopy(mapping[self.name])
            if self.nullable and not replacement.nullable:
                replacement.nullable = True
            return replacement
        new_args = [a.substitute(mapping) for a in self.args]
        result = copy.deepcopy(self)
        result.args = new_args
        return result

    def contains_typevar(self, name: Optional[str] = None) -> bool:
        if self.kind == "typevar":
            return name is None or self.name == name
        return any(a.contains_typevar(name) for a in self.args)

    def walk(self) -> Iterator[InternalType]:
        """Depth-first traversal of the type tree."""
        yield self
        for arg in self.args:
            yield from arg.walk()

    def depth(self) -> int:
        if not self.args:
            return 1
        return 1 + max(a.depth() for a in self.args)

    def structural_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.kind.encode())
        h.update(self.name.encode())
        for a in self.args:
            h.update(a.structural_hash().encode())
        if self.literal_values:
            for v in self.literal_values:
                h.update(repr(v).encode())
        if self.nullable:
            h.update(b"nullable")
        return h.hexdigest()[:16]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InternalType):
            return NotImplemented
        return (
            self.kind == other.kind
            and self.name == other.name
            and self.args == other.args
            and self.nullable == other.nullable
            and self.literal_values == other.literal_values
            and self.is_final == other.is_final
            and self.is_classvar == other.is_classvar
            and self.required == other.required
        )

    def __hash__(self) -> int:
        return hash(self.structural_hash())

    def pretty(self, indent: int = 0) -> str:
        pad = "  " * indent
        parts = [f"{pad}{self.kind}"]
        if self.name:
            parts[0] += f"({self.name})"
        flags: List[str] = []
        if self.nullable:
            flags.append("nullable")
        if self.is_final:
            flags.append("final")
        if self.is_classvar:
            flags.append("classvar")
        if self.required is not None:
            flags.append("required" if self.required else "not_required")
        if flags:
            parts[0] += " [" + ", ".join(flags) + "]"
        if self.literal_values is not None:
            parts[0] += f" values={self.literal_values!r}"
        if self.metadata:
            parts[0] += f" meta={self.metadata!r}"
        for a in self.args:
            parts.append(a.pretty(indent + 1))
        return "\n".join(parts)

    def to_annotation_string(self) -> str:
        """Reconstruct a human-readable annotation string."""
        if self.kind == "none":
            return "None"
        if self.kind == "any":
            return "Any"
        if self.kind == "never":
            return "Never"
        if self.kind == "self_type":
            return "Self"
        if self.kind == "literal":
            vals = ", ".join(repr(v) for v in (self.literal_values or []))
            return f"Literal[{vals}]"
        if self.kind == "union":
            parts = [a.to_annotation_string() for a in self.args]
            return " | ".join(parts)
        if self.kind == "callable":
            if len(self.args) >= 2:
                param_types = [a.to_annotation_string() for a in self.args[:-1]]
                ret = self.args[-1].to_annotation_string()
                return f"Callable[[{', '.join(param_types)}], {ret}]"
            return "Callable[..., Any]"
        if self.kind == "tuple":
            if self.metadata.get("variable_length"):
                if self.args:
                    return f"Tuple[{self.args[0].to_annotation_string()}, ...]"
                return "Tuple[()]"
            inner = ", ".join(a.to_annotation_string() for a in self.args)
            return f"Tuple[{inner}]"
        if self.kind == "generic" and self.args:
            inner = ", ".join(a.to_annotation_string() for a in self.args)
            return f"{self.name}[{inner}]"
        if self.kind == "typevar":
            return self.name
        if self.kind == "annotated" and self.args:
            base = self.args[0].to_annotation_string()
            meta_strs = [repr(m) for m in self.metadata.get("annotations", [])]
            return f"Annotated[{base}, {', '.join(meta_strs)}]"
        if self.kind == "basic":
            base = self.name or "Unknown"
            if self.nullable:
                return f"Optional[{base}]"
            return base
        if self.name:
            if self.args:
                inner = ", ".join(a.to_annotation_string() for a in self.args)
                return f"{self.name}[{inner}]"
            return self.name
        return repr(self)


def make_basic(name: str, *, nullable: bool = False) -> InternalType:
    return InternalType(kind="basic", name=name, nullable=nullable)


def make_union(types: List[InternalType]) -> InternalType:
    flat: List[InternalType] = []
    nullable = False
    for t in types:
        if t.is_none_type():
            nullable = True
            continue
        if t.kind == "union":
            for sub in t.flatten_union():
                if sub.is_none_type():
                    nullable = True
                else:
                    flat.append(sub)
        else:
            flat.append(t)
    # de-duplicate
    seen_hashes: Set[str] = set()
    deduped: List[InternalType] = []
    for t in flat:
        h = t.structural_hash()
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(t)
    if not deduped:
        result = InternalType(kind="none")
        return result
    if len(deduped) == 1:
        result = copy.deepcopy(deduped[0])
        result.nullable = result.nullable or nullable
        return result
    return InternalType(kind="union", args=deduped, nullable=nullable)


def make_callable(param_types: List[InternalType], return_type: InternalType) -> InternalType:
    return InternalType(kind="callable", args=param_types + [return_type])


def make_literal(values: List[Any]) -> InternalType:
    return InternalType(kind="literal", literal_values=values)


def make_tuple(element_types: List[InternalType], *, variable_length: bool = False) -> InternalType:
    return InternalType(
        kind="tuple",
        args=element_types,
        metadata={"variable_length": variable_length},
    )


def make_typevar(
    name: str,
    bound: Optional[InternalType] = None,
    constraints: Optional[List[InternalType]] = None,
    covariant: bool = False,
    contravariant: bool = False,
) -> InternalType:
    meta: Dict[str, Any] = {
        "covariant": covariant,
        "contravariant": contravariant,
    }
    if bound is not None:
        meta["bound"] = bound
    if constraints:
        meta["constraints"] = constraints
    return InternalType(kind="typevar", name=name, metadata=meta)


# ---------------------------------------------------------------------------
# 2. AnnotationParser – parse AST nodes into InternalType
# ---------------------------------------------------------------------------

class AnnotationParser:
    """Parse Python type-annotation AST nodes into *InternalType*."""

    def __init__(self) -> None:
        self._typing_handler = TypingModuleHandler()
        self._forward_ref_cache: Dict[str, InternalType] = {}
        self._recursion_guard: Set[str] = set()
        self._string_annotation_mode = True  # PEP 563

    def parse_annotation(self, node: ast.AST, *, context: Optional[Dict[str, Any]] = None) -> InternalType:
        """Main entry point – dispatch to the correct sub-parser."""
        if node is None:
            return InternalType(kind="any", name="Any")

        ctx = context or {}

        if isinstance(node, ast.Constant):
            return self._parse_constant(node, ctx)
        if isinstance(node, ast.Name):
            return self._parse_name(node, ctx)
        if isinstance(node, ast.Attribute):
            return self._parse_attribute(node, ctx)
        if isinstance(node, ast.Subscript):
            return self._parse_subscript(node, ctx)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self._parse_binop(node, ctx)
        if isinstance(node, ast.Tuple):
            elements = [self.parse_annotation(e, context=ctx) for e in node.elts]
            return make_tuple(elements)
        if isinstance(node, ast.List):
            elements = [self.parse_annotation(e, context=ctx) for e in node.elts]
            return InternalType(kind="generic", name="list", args=elements)
        if isinstance(node, ast.Starred):
            inner = self.parse_annotation(node.value, context=ctx)
            return InternalType(kind="unpack", name="Unpack", args=[inner])
        if isinstance(node, ast.Index):  # Python 3.8 compat
            return self.parse_annotation(node.value, context=ctx)  # type: ignore[attr-defined]

        return InternalType(kind="any", name="Unknown", metadata={"unparsed_node": ast.dump(node)})

    # ------------------------------------------------------------------

    def _parse_constant(self, node: ast.Constant, ctx: Dict[str, Any]) -> InternalType:
        if node.value is None:
            return InternalType(kind="none", name="None")
        if isinstance(node.value, str):
            return self._parse_string_annotation(node.value, ctx)
        if isinstance(node.value, bool):
            return make_literal([node.value])
        if isinstance(node.value, int):
            return make_literal([node.value])
        if isinstance(node.value, float):
            return make_literal([node.value])
        if isinstance(node.value, bytes):
            return make_literal([node.value])
        if node.value is ...:
            return InternalType(kind="basic", name="Ellipsis")
        return InternalType(kind="basic", name=type(node.value).__name__)

    def _parse_string_annotation(self, s: str, ctx: Dict[str, Any]) -> InternalType:
        """Handle PEP 563 string annotations and explicit forward references."""
        s = s.strip()
        cache_key = s
        if cache_key in self._forward_ref_cache:
            return copy.deepcopy(self._forward_ref_cache[cache_key])
        if cache_key in self._recursion_guard:
            return InternalType(kind="basic", name=s, metadata={"forward_ref": True, "recursive": True})

        self._recursion_guard.add(cache_key)
        try:
            tree = ast.parse(s, mode="eval")
            result = self.parse_annotation(tree.body, context=ctx)
            self._forward_ref_cache[cache_key] = result
            return copy.deepcopy(result)
        except SyntaxError:
            return InternalType(kind="basic", name=s, metadata={"forward_ref": True, "invalid_syntax": True})
        finally:
            self._recursion_guard.discard(cache_key)

    def _parse_name(self, node: ast.Name, ctx: Dict[str, Any]) -> InternalType:
        name = node.id
        builtins_map: Dict[str, str] = {
            "int": "int",
            "float": "float",
            "str": "str",
            "bytes": "bytes",
            "bool": "bool",
            "complex": "complex",
            "bytearray": "bytearray",
            "memoryview": "memoryview",
            "object": "object",
            "type": "type",
            "list": "list",
            "dict": "dict",
            "set": "set",
            "frozenset": "frozenset",
            "tuple": "tuple",
            "range": "range",
        }
        if name == "None":
            return InternalType(kind="none", name="None")
        if name == "Any":
            return InternalType(kind="any", name="Any")
        if name == "NoReturn" or name == "Never":
            return InternalType(kind="never", name=name)
        if name == "Self":
            return InternalType(kind="self_type", name="Self")
        if name in builtins_map:
            return make_basic(builtins_map[name])
        # typing aliases without subscription
        simple_typing = self._typing_handler.simple_name_map()
        if name in simple_typing:
            return simple_typing[name]
        # treat as user-defined type name
        return make_basic(name)

    def _parse_attribute(self, node: ast.Attribute, ctx: Dict[str, Any]) -> InternalType:
        parts = self._collect_attribute_parts(node)
        qualified = ".".join(parts)
        # Handle typing.X / typing_extensions.X
        if parts[0] in ("typing", "typing_extensions") and len(parts) == 2:
            return self._parse_name(ast.Name(id=parts[1]), ctx)
        if parts[0] == "collections" and len(parts) >= 2:
            if parts[-1] in ("OrderedDict", "defaultdict", "deque", "Counter", "ChainMap"):
                return make_basic(parts[-1])
        return make_basic(qualified)

    @staticmethod
    def _collect_attribute_parts(node: ast.AST) -> List[str]:
        parts: List[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        return parts

    def _parse_subscript(self, node: ast.Subscript, ctx: Dict[str, Any]) -> InternalType:
        base = node.value
        base_type = self.parse_annotation(base, context=ctx)
        slice_node = node.slice

        # Extract args from slice
        args = self._extract_subscript_args(slice_node, ctx)

        # Delegate to typing handler for known forms
        type_name = base_type.name
        result = self._typing_handler.handle_subscript(type_name, args, ctx, parser=self)
        if result is not None:
            return result

        # Generic instantiation: SomeClass[int, str]
        return InternalType(kind="generic", name=type_name, args=args)

    def _extract_subscript_args(self, slice_node: ast.AST, ctx: Dict[str, Any]) -> List[InternalType]:
        if isinstance(slice_node, ast.Tuple):
            return [self.parse_annotation(e, context=ctx) for e in slice_node.elts]
        if isinstance(slice_node, ast.Index):  # Python 3.8
            return self._extract_subscript_args(slice_node.value, ctx)  # type: ignore[attr-defined]
        return [self.parse_annotation(slice_node, context=ctx)]

    def _parse_binop(self, node: ast.BinOp, ctx: Dict[str, Any]) -> InternalType:
        """Handle PEP 604 ``X | Y`` union syntax."""
        left = self.parse_annotation(node.left, context=ctx)
        right = self.parse_annotation(node.right, context=ctx)
        return make_union([left, right])

    # ------------------------------------------------------------------
    # convenience helpers
    # ------------------------------------------------------------------

    def parse_annotation_string(self, source: str) -> InternalType:
        tree = ast.parse(source, mode="eval")
        return self.parse_annotation(tree.body)

    def parse_function_annotations(self, func_node: ast.FunctionDef) -> Dict[str, InternalType]:
        """Return mapping of argument-name → type for a function definition."""
        result: Dict[str, InternalType] = {}
        for arg in func_node.args.args + func_node.args.posonlyargs + func_node.args.kwonlyargs:
            if arg.annotation:
                result[arg.arg] = self.parse_annotation(arg.annotation)
            else:
                result[arg.arg] = InternalType(kind="any", name="Any")
        if func_node.args.vararg and func_node.args.vararg.annotation:
            result["*" + func_node.args.vararg.arg] = self.parse_annotation(func_node.args.vararg.annotation)
        if func_node.args.kwarg and func_node.args.kwarg.annotation:
            result["**" + func_node.args.kwarg.arg] = self.parse_annotation(func_node.args.kwarg.annotation)
        if func_node.returns:
            result["return"] = self.parse_annotation(func_node.returns)
        else:
            result["return"] = InternalType(kind="any", name="Any")
        return result

    def parse_variable_annotation(self, ann_node: ast.AnnAssign) -> Tuple[str, InternalType]:
        target_name = ""
        if isinstance(ann_node.target, ast.Name):
            target_name = ann_node.target.id
        elif isinstance(ann_node.target, ast.Attribute):
            target_name = ".".join(self._collect_attribute_parts(ann_node.target))
        ann_type = self.parse_annotation(ann_node.annotation)
        return target_name, ann_type


# ---------------------------------------------------------------------------
# 3. TypingModuleHandler – handle typing module constructs
# ---------------------------------------------------------------------------

class TypingModuleHandler:
    """Map ``typing`` constructs to *InternalType*."""

    _CONTAINER_MAP: Dict[str, str] = {
        "List": "list",
        "Dict": "dict",
        "Set": "set",
        "FrozenSet": "frozenset",
        "Deque": "collections.deque",
        "DefaultDict": "collections.defaultdict",
        "OrderedDict": "collections.OrderedDict",
        "Counter": "collections.Counter",
        "ChainMap": "collections.ChainMap",
        "Sequence": "Sequence",
        "MutableSequence": "MutableSequence",
        "AbstractSet": "AbstractSet",
        "MutableSet": "MutableSet",
        "Mapping": "Mapping",
        "MutableMapping": "MutableMapping",
        "Iterable": "Iterable",
        "Iterator": "Iterator",
        "Generator": "Generator",
        "AsyncGenerator": "AsyncGenerator",
        "AsyncIterator": "AsyncIterator",
        "AsyncIterable": "AsyncIterable",
        "Awaitable": "Awaitable",
        "Coroutine": "Coroutine",
        "ContextManager": "ContextManager",
        "AsyncContextManager": "AsyncContextManager",
        "Pattern": "re.Pattern",
        "Match": "re.Match",
        "IO": "IO",
        "TextIO": "TextIO",
        "BinaryIO": "BinaryIO",
        "SupportsInt": "SupportsInt",
        "SupportsFloat": "SupportsFloat",
        "SupportsComplex": "SupportsComplex",
        "SupportsBytes": "SupportsBytes",
        "SupportsAbs": "SupportsAbs",
        "SupportsRound": "SupportsRound",
        "Reversible": "Reversible",
        "Hashable": "Hashable",
        "Sized": "Sized",
        "Collection": "Collection",
        "ByteString": "ByteString",
    }

    def simple_name_map(self) -> Dict[str, InternalType]:
        """Return a map for un-subscripted typing names."""
        m: Dict[str, InternalType] = {}
        for alias, canon in self._CONTAINER_MAP.items():
            m[alias] = make_basic(canon)
        m["Any"] = InternalType(kind="any", name="Any")
        m["NoReturn"] = InternalType(kind="never", name="NoReturn")
        m["Never"] = InternalType(kind="never", name="Never")
        m["Self"] = InternalType(kind="self_type", name="Self")
        m["LiteralString"] = make_basic("str", nullable=False)
        m["TypeAlias"] = InternalType(kind="type_alias", name="TypeAlias")
        return m

    def handle_subscript(
        self,
        name: str,
        args: List[InternalType],
        ctx: Dict[str, Any],
        *,
        parser: Optional[AnnotationParser] = None,
    ) -> Optional[InternalType]:
        handler_name = f"_handle_{name}"
        handler = getattr(self, handler_name, None)
        if handler is not None:
            return handler(args, ctx)

        # containers
        if name in self._CONTAINER_MAP:
            return InternalType(kind="generic", name=self._CONTAINER_MAP[name], args=args)
        return None

    # -- specific handlers ------------------------------------------------

    def _handle_Optional(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        if not args:
            return InternalType(kind="any", name="Any", nullable=True)
        inner = args[0]
        return make_union([inner, InternalType(kind="none", name="None")])

    def _handle_Union(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        return make_union(args)

    def _handle_Callable(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        if not args:
            return InternalType(kind="callable", name="Callable")
        if len(args) == 2 and args[0].kind == "basic" and args[0].name == "Ellipsis":
            return InternalType(
                kind="callable",
                name="Callable",
                args=[args[1]],
                metadata={"param_spec": "..."},
            )
        if len(args) == 2 and args[0].kind in ("generic", "tuple"):
            param_types = args[0].args if args[0].args else []
            return make_callable(param_types, args[1])
        if len(args) >= 2:
            return make_callable(args[:-1], args[-1])
        return InternalType(kind="callable", name="Callable", args=args)

    def _handle_Tuple(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        if not args:
            return make_tuple([])
        if len(args) == 2 and args[1].kind == "basic" and args[1].name == "Ellipsis":
            return make_tuple([args[0]], variable_length=True)
        if len(args) == 1 and args[0].kind == "tuple" and not args[0].args:
            return make_tuple([])  # Tuple[()]
        return make_tuple(args)

    def _handle_tuple(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        return self._handle_Tuple(args, ctx)

    def _handle_list(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        return InternalType(kind="generic", name="list", args=args)

    def _handle_dict(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        return InternalType(kind="generic", name="dict", args=args)

    def _handle_set(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        return InternalType(kind="generic", name="set", args=args)

    def _handle_frozenset(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        return InternalType(kind="generic", name="frozenset", args=args)

    def _handle_Type(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        return InternalType(kind="generic", name="Type", args=args, metadata={"metatype": True})

    def _handle_type(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        return self._handle_Type(args, ctx)

    def _handle_ClassVar(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        inner = args[0] if args else InternalType(kind="any", name="Any")
        inner = copy.deepcopy(inner)
        inner.is_classvar = True
        return inner

    def _handle_Final(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        inner = args[0] if args else InternalType(kind="any", name="Any")
        inner = copy.deepcopy(inner)
        inner.is_final = True
        return inner

    def _handle_Literal(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        values: List[Any] = []
        for a in args:
            if a.literal_values is not None:
                values.extend(a.literal_values)
            elif a.kind == "none":
                values.append(None)
            elif a.kind == "basic" and a.name in ("True",):
                values.append(True)
            elif a.kind == "basic" and a.name in ("False",):
                values.append(False)
            else:
                values.append(a.name)
        return make_literal(values)

    def _handle_Annotated(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        if not args:
            return InternalType(kind="any", name="Any")
        base = args[0]
        metadata_items: List[Any] = []
        for a in args[1:]:
            if a.literal_values:
                metadata_items.extend(a.literal_values)
            elif a.metadata:
                metadata_items.append(a.metadata)
            else:
                metadata_items.append(a.name or a.kind)
        return InternalType(
            kind="annotated",
            name="Annotated",
            args=[base],
            metadata={"annotations": metadata_items},
        )

    def _handle_TypeGuard(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        inner = args[0] if args else InternalType(kind="any", name="Any")
        return InternalType(kind="typeguard", name="TypeGuard", args=[inner], metadata={"narrowing": "typeguard"})

    def _handle_TypeIs(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        inner = args[0] if args else InternalType(kind="any", name="Any")
        return InternalType(kind="typeguard", name="TypeIs", args=[inner], metadata={"narrowing": "typeis"})

    def _handle_Never(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        return InternalType(kind="never", name="Never")

    def _handle_NoReturn(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        return InternalType(kind="never", name="NoReturn")

    def _handle_Self(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        return InternalType(kind="self_type", name="Self")

    def _handle_Unpack(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        inner = args[0] if args else InternalType(kind="any", name="Any")
        return InternalType(kind="unpack", name="Unpack", args=[inner])

    def _handle_Required(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        inner = args[0] if args else InternalType(kind="any", name="Any")
        inner = copy.deepcopy(inner)
        inner.required = True
        return inner

    def _handle_NotRequired(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        inner = args[0] if args else InternalType(kind="any", name="Any")
        inner = copy.deepcopy(inner)
        inner.required = False
        return inner

    def _handle_ReadOnly(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        inner = args[0] if args else InternalType(kind="any", name="Any")
        return InternalType(kind="readonly", name="ReadOnly", args=[inner])

    def _handle_ParamSpec(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        name = args[0].name if args else "P"
        return InternalType(kind="paramspec", name=name)

    def _handle_Concatenate(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        return InternalType(kind="concatenate", name="Concatenate", args=args)

    def _handle_TypeVarTuple(self, args: List[InternalType], ctx: Dict[str, Any]) -> InternalType:
        name = args[0].name if args else "Ts"
        return InternalType(kind="typevar_tuple", name=name)


# ---------------------------------------------------------------------------
# 4. GenericHandler – handle generic types
# ---------------------------------------------------------------------------

@dataclass
class TypeVarInfo:
    """Extracted information about a TypeVar."""
    name: str
    bound: Optional[InternalType] = None
    constraints: List[InternalType] = field(default_factory=list)
    covariant: bool = False
    contravariant: bool = False
    default: Optional[InternalType] = None

    def variance_label(self) -> str:
        if self.covariant:
            return "covariant"
        if self.contravariant:
            return "contravariant"
        return "invariant"


class GenericHandler:
    """Handle generic type definitions and instantiations."""

    def __init__(self) -> None:
        self._typevars: Dict[str, TypeVarInfo] = {}
        self._generic_classes: Dict[str, List[str]] = {}  # class_name → [typevar_names]
        self._instantiations: Dict[str, Dict[str, InternalType]] = {}

    # ------------------------------------------------------------------
    # TypeVar registration
    # ------------------------------------------------------------------

    def register_typevar(self, info: TypeVarInfo) -> None:
        self._typevars[info.name] = info

    def parse_typevar_definition(self, node: ast.Assign) -> Optional[TypeVarInfo]:
        """Parse ``T = TypeVar('T', bound=int)`` style definitions."""
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            return None
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return None
        if not isinstance(node.value, ast.Call):
            return None
        func = node.value
        func_name = ""
        if isinstance(func.func, ast.Name):
            func_name = func.func.id
        elif isinstance(func.func, ast.Attribute):
            func_name = func.func.attr
        if func_name != "TypeVar":
            return None

        tv_name = target.id
        bound: Optional[InternalType] = None
        constraints: List[InternalType] = []
        covariant = False
        contravariant = False
        default: Optional[InternalType] = None
        parser = AnnotationParser()

        # positional args after name → constraints
        for i, arg in enumerate(func.args):
            if i == 0:
                # first arg is the name string, skip
                continue
            constraints.append(parser.parse_annotation(arg))

        for kw in func.keywords:
            if kw.arg == "bound":
                bound = parser.parse_annotation(kw.value)
            elif kw.arg == "covariant":
                if isinstance(kw.value, ast.Constant):
                    covariant = bool(kw.value.value)
            elif kw.arg == "contravariant":
                if isinstance(kw.value, ast.Constant):
                    contravariant = bool(kw.value.value)
            elif kw.arg == "default":
                default = parser.parse_annotation(kw.value)

        info = TypeVarInfo(
            name=tv_name,
            bound=bound,
            constraints=constraints,
            covariant=covariant,
            contravariant=contravariant,
            default=default,
        )
        self.register_typevar(info)
        return info

    # ------------------------------------------------------------------
    # Generic class handling
    # ------------------------------------------------------------------

    def register_generic_class(self, class_name: str, typevar_names: List[str]) -> None:
        self._generic_classes[class_name] = typevar_names

    def detect_generic_class(self, node: ast.ClassDef) -> Optional[Tuple[str, List[str]]]:
        """Detect ``class Foo(Generic[T, U]):`` pattern."""
        for base in node.bases:
            if isinstance(base, ast.Subscript):
                base_name = ""
                if isinstance(base.value, ast.Name):
                    base_name = base.value.id
                elif isinstance(base.value, ast.Attribute):
                    base_name = base.value.attr
                if base_name == "Generic":
                    tvars = self._extract_typevar_names(base.slice)
                    self.register_generic_class(node.name, tvars)
                    return node.name, tvars
        return None

    def _extract_typevar_names(self, node: ast.AST) -> List[str]:
        names: List[str] = []
        if isinstance(node, ast.Tuple):
            for elt in node.elts:
                names.extend(self._extract_typevar_names(elt))
        elif isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Starred):
            if isinstance(node.value, ast.Name):
                names.append("*" + node.value.id)
        elif isinstance(node, ast.Index):
            names.extend(self._extract_typevar_names(node.value))  # type: ignore[attr-defined]
        return names

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def instantiate(self, class_name: str, type_args: List[InternalType]) -> InternalType:
        if class_name not in self._generic_classes:
            return InternalType(kind="generic", name=class_name, args=type_args)
        tvars = self._generic_classes[class_name]
        mapping: Dict[str, InternalType] = {}
        for i, tv_name in enumerate(tvars):
            if i < len(type_args):
                mapping[tv_name] = type_args[i]
        key = f"{class_name}[{','.join(t.to_annotation_string() for t in type_args)}]"
        self._instantiations[key] = mapping
        return InternalType(
            kind="generic",
            name=class_name,
            args=type_args,
            metadata={"type_param_mapping": {k: v.to_annotation_string() for k, v in mapping.items()}},
        )

    def resolve_type_in_context(self, ty: InternalType, class_name: str, type_args: List[InternalType]) -> InternalType:
        if class_name not in self._generic_classes:
            return ty
        tvars = self._generic_classes[class_name]
        mapping: Dict[str, InternalType] = {}
        for i, tv_name in enumerate(tvars):
            if i < len(type_args):
                mapping[tv_name] = type_args[i]
        return ty.substitute(mapping)

    def infer_variance(self, typevar_name: str, class_body: List[ast.stmt]) -> str:
        """Heuristic variance inference from method signatures.

        Scans methods in the class body:
        - If the typevar only appears in return types → covariant
        - If it only appears in parameter types → contravariant
        - Otherwise → invariant
        """
        parser = AnnotationParser()
        appears_in_params = False
        appears_in_return = False

        for stmt in class_body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for arg in stmt.args.args + stmt.args.posonlyargs + stmt.args.kwonlyargs:
                if arg.annotation:
                    ann = parser.parse_annotation(arg.annotation)
                    if ann.contains_typevar(typevar_name):
                        appears_in_params = True
            if stmt.returns:
                ret = parser.parse_annotation(stmt.returns)
                if ret.contains_typevar(typevar_name):
                    appears_in_return = True

        if appears_in_return and not appears_in_params:
            return "covariant"
        if appears_in_params and not appears_in_return:
            return "contravariant"
        return "invariant"

    def get_typevar_info(self, name: str) -> Optional[TypeVarInfo]:
        return self._typevars.get(name)

    def get_generic_params(self, class_name: str) -> List[str]:
        return self._generic_classes.get(class_name, [])


# ---------------------------------------------------------------------------
# 5. ProtocolHandler – handle Protocol types
# ---------------------------------------------------------------------------

@dataclass
class ProtocolMember:
    name: str
    kind: str  # "method", "property", "attribute"
    type: InternalType
    is_abstract: bool = True
    has_default: bool = False


@dataclass
class ProtocolInfo:
    name: str
    members: List[ProtocolMember] = field(default_factory=list)
    runtime_checkable: bool = False
    bases: List[str] = field(default_factory=list)
    type_params: List[str] = field(default_factory=list)


class ProtocolHandler:
    """Handle Protocol classes and structural subtyping."""

    def __init__(self) -> None:
        self._protocols: Dict[str, ProtocolInfo] = {}

    def detect_protocol(self, node: ast.ClassDef) -> Optional[ProtocolInfo]:
        is_protocol = False
        bases: List[str] = []
        type_params: List[str] = []
        for base in node.bases:
            base_name = self._base_name(base)
            if base_name == "Protocol":
                is_protocol = True
            elif base_name and base_name.endswith("Protocol"):
                is_protocol = True
            if base_name and base_name != "Protocol":
                bases.append(base_name)
            # check for Generic[T] in subscript form
            if isinstance(base, ast.Subscript):
                bname = self._base_name(base.value)
                if bname == "Protocol":
                    is_protocol = True
                    type_params = self._extract_type_param_names(base.slice)
        if not is_protocol:
            return None

        runtime_checkable = any(
            self._is_runtime_checkable_decorator(d) for d in node.decorator_list
        )

        members = self._extract_members(node)

        info = ProtocolInfo(
            name=node.name,
            members=members,
            runtime_checkable=runtime_checkable,
            bases=bases,
            type_params=type_params,
        )
        self._protocols[node.name] = info
        return info

    def _base_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            return self._base_name(node.value)
        return None

    def _extract_type_param_names(self, node: ast.AST) -> List[str]:
        names: List[str] = []
        if isinstance(node, ast.Tuple):
            for e in node.elts:
                if isinstance(e, ast.Name):
                    names.append(e.id)
        elif isinstance(node, ast.Name):
            names.append(node.id)
        return names

    def _is_runtime_checkable_decorator(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "runtime_checkable"
        if isinstance(node, ast.Attribute):
            return node.attr == "runtime_checkable"
        return False

    def _extract_members(self, node: ast.ClassDef) -> List[ProtocolMember]:
        members: List[ProtocolMember] = []
        parser = AnnotationParser()
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) or isinstance(stmt, ast.AsyncFunctionDef):
                is_property = any(self._is_property_decorator(d) for d in stmt.decorator_list)
                if is_property:
                    ret_type = parser.parse_annotation(stmt.returns) if stmt.returns else InternalType(kind="any", name="Any")
                    members.append(ProtocolMember(
                        name=stmt.name,
                        kind="property",
                        type=ret_type,
                        is_abstract=self._is_abstract(stmt),
                        has_default=self._has_concrete_body(stmt),
                    ))
                else:
                    anns = parser.parse_function_annotations(stmt)
                    param_types = [v for k, v in anns.items() if k not in ("self", "cls", "return")]
                    ret_type = anns.get("return", InternalType(kind="any", name="Any"))
                    func_type = make_callable(param_types, ret_type)
                    members.append(ProtocolMember(
                        name=stmt.name,
                        kind="method",
                        type=func_type,
                        is_abstract=self._is_abstract(stmt),
                        has_default=self._has_concrete_body(stmt),
                    ))
            elif isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name):
                    ann_type = parser.parse_annotation(stmt.annotation)
                    members.append(ProtocolMember(
                        name=stmt.target.id,
                        kind="attribute",
                        type=ann_type,
                        has_default=stmt.value is not None,
                    ))
        return members

    @staticmethod
    def _is_property_decorator(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "property"
        if isinstance(node, ast.Attribute):
            return node.attr == "property"
        return False

    @staticmethod
    def _is_abstract(node: ast.AST) -> bool:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        for d in node.decorator_list:
            if isinstance(d, ast.Name) and d.id in ("abstractmethod", "abstractproperty"):
                return True
            if isinstance(d, ast.Attribute) and d.attr in ("abstractmethod", "abstractproperty"):
                return True
        return False

    @staticmethod
    def _has_concrete_body(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Return True if the body is more than just ``...`` or ``pass``."""
        if len(node.body) == 1:
            stmt = node.body[0]
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...:
                return False
            if isinstance(stmt, ast.Pass):
                return False
        return True

    # ------------------------------------------------------------------
    # Structural subtyping
    # ------------------------------------------------------------------

    def structural_subtype_check(
        self,
        protocol_name: str,
        class_members: Dict[str, InternalType],
    ) -> Tuple[bool, List[str]]:
        """Check whether *class_members* satisfy *protocol_name*.

        Returns ``(is_subtype, list_of_missing_member_names)``.
        """
        proto = self._protocols.get(protocol_name)
        if proto is None:
            return False, [f"unknown protocol {protocol_name}"]
        missing: List[str] = []
        for member in proto.members:
            if member.name.startswith("_") and member.name != "__init__":
                continue
            if member.name not in class_members:
                missing.append(member.name)
        return len(missing) == 0, missing

    def get_protocol(self, name: str) -> Optional[ProtocolInfo]:
        return self._protocols.get(name)

    def all_protocols(self) -> Dict[str, ProtocolInfo]:
        return dict(self._protocols)

    def protocol_to_internal_type(self, name: str) -> InternalType:
        proto = self._protocols.get(name)
        if proto is None:
            return make_basic(name)
        member_data = {
            m.name: {"kind": m.kind, "type_str": m.type.to_annotation_string()}
            for m in proto.members
        }
        return InternalType(
            kind="protocol",
            name=name,
            metadata={
                "members": member_data,
                "runtime_checkable": proto.runtime_checkable,
                "type_params": proto.type_params,
            },
        )


# ---------------------------------------------------------------------------
# 6. TypedDictHandler – handle TypedDict
# ---------------------------------------------------------------------------

@dataclass
class TypedDictFieldInfo:
    name: str
    type: InternalType
    required: bool = True
    readonly: bool = False


@dataclass
class TypedDictInfo:
    name: str
    fields: List[TypedDictFieldInfo] = field(default_factory=list)
    total: bool = True
    bases: List[str] = field(default_factory=list)
    closed: bool = False


class TypedDictHandler:
    """Parse and represent TypedDict definitions."""

    def __init__(self) -> None:
        self._typeddicts: Dict[str, TypedDictInfo] = {}

    def detect_typeddict(self, node: ast.ClassDef) -> Optional[TypedDictInfo]:
        is_td = False
        bases: List[str] = []
        total = True
        for base in node.bases:
            bname = self._base_name(base)
            if bname == "TypedDict":
                is_td = True
            elif bname:
                bases.append(bname)
        for kw in node.keywords:
            if kw.arg == "total":
                if isinstance(kw.value, ast.Constant):
                    total = bool(kw.value.value)
            if kw.arg == "closed":
                if isinstance(kw.value, ast.Constant):
                    pass  # PEP 728
        if not is_td:
            return None

        fields = self._extract_fields(node, total)

        # inherit fields from bases
        inherited: List[TypedDictFieldInfo] = []
        for b in bases:
            parent = self._typeddicts.get(b)
            if parent:
                inherited.extend(parent.fields)

        info = TypedDictInfo(name=node.name, fields=inherited + fields, total=total, bases=bases)
        self._typeddicts[node.name] = info
        return info

    def parse_functional_typeddict(self, node: ast.Assign) -> Optional[TypedDictInfo]:
        """Parse ``Movie = TypedDict('Movie', {'name': str, 'year': int})``."""
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            return None
        if not isinstance(node.targets[0], ast.Name):
            return None
        if not isinstance(node.value, ast.Call):
            return None
        func = node.value.func
        func_name = ""
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        if func_name != "TypedDict":
            return None

        name = node.targets[0].id
        total = True
        fields: List[TypedDictFieldInfo] = []
        parser = AnnotationParser()

        for kw in node.value.keywords:
            if kw.arg == "total" and isinstance(kw.value, ast.Constant):
                total = bool(kw.value.value)

        # second positional arg is a dict
        if len(node.value.args) >= 2:
            dict_node = node.value.args[1]
            if isinstance(dict_node, ast.Dict):
                for key_node, val_node in zip(dict_node.keys, dict_node.values):
                    if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                        ftype = parser.parse_annotation(val_node)
                        required = total
                        if ftype.required is not None:
                            required = ftype.required
                        fields.append(TypedDictFieldInfo(
                            name=key_node.value,
                            type=ftype,
                            required=required,
                        ))

        info = TypedDictInfo(name=name, fields=fields, total=total)
        self._typeddicts[name] = info
        return info

    def _extract_fields(self, node: ast.ClassDef, total: bool) -> List[TypedDictFieldInfo]:
        fields: List[TypedDictFieldInfo] = []
        parser = AnnotationParser()
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                ftype = parser.parse_annotation(stmt.annotation)
                required = total
                readonly = False
                if ftype.required is not None:
                    required = ftype.required
                if ftype.kind == "readonly":
                    readonly = True
                    ftype = ftype.args[0] if ftype.args else InternalType(kind="any", name="Any")
                fields.append(TypedDictFieldInfo(
                    name=stmt.target.id,
                    type=ftype,
                    required=required,
                    readonly=readonly,
                ))
        return fields

    @staticmethod
    def _base_name(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                return node.value.id
        return None

    def get_typeddict(self, name: str) -> Optional[TypedDictInfo]:
        return self._typeddicts.get(name)

    def typeddict_to_internal_type(self, name: str) -> InternalType:
        info = self._typeddicts.get(name)
        if info is None:
            return make_basic(name)
        field_data = {
            f.name: {
                "type_str": f.type.to_annotation_string(),
                "required": f.required,
                "readonly": f.readonly,
            }
            for f in info.fields
        }
        return InternalType(
            kind="typeddict",
            name=name,
            metadata={"fields": field_data, "total": info.total, "bases": info.bases},
        )


# ---------------------------------------------------------------------------
# 7. NamedTupleHandler – handle NamedTuple
# ---------------------------------------------------------------------------

@dataclass
class NamedTupleFieldInfo:
    name: str
    type: InternalType
    default: Optional[Any] = None
    has_default: bool = False


@dataclass
class NamedTupleInfo:
    name: str
    fields: List[NamedTupleFieldInfo] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


class NamedTupleHandler:
    """Parse NamedTuple definitions."""

    def __init__(self) -> None:
        self._namedtuples: Dict[str, NamedTupleInfo] = {}

    def detect_namedtuple_class(self, node: ast.ClassDef) -> Optional[NamedTupleInfo]:
        is_nt = False
        for base in node.bases:
            bname = self._base_name(base)
            if bname in ("NamedTuple", "typing.NamedTuple"):
                is_nt = True
        if not is_nt:
            return None

        fields = self._extract_fields(node)
        methods = self._extract_methods(node)
        docstring = ast.get_docstring(node)

        info = NamedTupleInfo(name=node.name, fields=fields, methods=methods, docstring=docstring)
        self._namedtuples[node.name] = info
        return info

    def detect_functional_namedtuple(self, node: ast.Assign) -> Optional[NamedTupleInfo]:
        """Parse ``Point = NamedTuple('Point', [('x', int), ('y', int)])``
        and ``Point = namedtuple('Point', ['x', 'y'])`` forms.
        """
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            return None
        if not isinstance(node.targets[0], ast.Name):
            return None
        if not isinstance(node.value, ast.Call):
            return None
        func_name = ""
        if isinstance(node.value.func, ast.Name):
            func_name = node.value.func.id
        elif isinstance(node.value.func, ast.Attribute):
            func_name = node.value.func.attr
        if func_name not in ("NamedTuple", "namedtuple"):
            return None

        name = node.targets[0].id
        fields: List[NamedTupleFieldInfo] = []
        parser = AnnotationParser()

        args = node.value.args
        if len(args) >= 2:
            fields_arg = args[1]
            if isinstance(fields_arg, ast.List):
                for elt in fields_arg.elts:
                    if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2:
                        if isinstance(elt.elts[0], ast.Constant) and isinstance(elt.elts[0].value, str):
                            ftype = parser.parse_annotation(elt.elts[1])
                            fields.append(NamedTupleFieldInfo(name=elt.elts[0].value, type=ftype))
                    elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        # namedtuple without types
                        fields.append(NamedTupleFieldInfo(name=elt.value, type=InternalType(kind="any", name="Any")))
            elif isinstance(fields_arg, ast.Constant) and isinstance(fields_arg.value, str):
                # namedtuple('Point', 'x y')
                for fname in fields_arg.value.replace(",", " ").split():
                    fname = fname.strip()
                    if fname:
                        fields.append(NamedTupleFieldInfo(name=fname, type=InternalType(kind="any", name="Any")))

        # keyword defaults
        for kw in node.value.keywords:
            if kw.arg == "defaults" and isinstance(kw.value, (ast.List, ast.Tuple)):
                defaults = kw.value.elts
                offset = len(fields) - len(defaults)
                for i, d in enumerate(defaults):
                    idx = offset + i
                    if 0 <= idx < len(fields):
                        fields[idx].has_default = True
                        if isinstance(d, ast.Constant):
                            fields[idx].default = d.value

        info = NamedTupleInfo(name=name, fields=fields)
        self._namedtuples[name] = info
        return info

    def _extract_fields(self, node: ast.ClassDef) -> List[NamedTupleFieldInfo]:
        fields: List[NamedTupleFieldInfo] = []
        parser = AnnotationParser()
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                ftype = parser.parse_annotation(stmt.annotation)
                has_default = stmt.value is not None
                default_val = None
                if has_default and isinstance(stmt.value, ast.Constant):
                    default_val = stmt.value.value
                fields.append(NamedTupleFieldInfo(
                    name=stmt.target.id,
                    type=ftype,
                    default=default_val,
                    has_default=has_default,
                ))
        return fields

    def _extract_methods(self, node: ast.ClassDef) -> List[str]:
        methods: List[str] = []
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(stmt.name)
        return methods

    @staticmethod
    def _base_name(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: List[str] = []
            cur = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            parts.reverse()
            return ".".join(parts)
        return None

    def get_namedtuple(self, name: str) -> Optional[NamedTupleInfo]:
        return self._namedtuples.get(name)

    def namedtuple_to_internal_type(self, name: str) -> InternalType:
        info = self._namedtuples.get(name)
        if info is None:
            return make_basic(name)
        field_data = {
            f.name: {
                "type_str": f.type.to_annotation_string(),
                "has_default": f.has_default,
                "default": repr(f.default) if f.has_default else None,
            }
            for f in info.fields
        }
        return InternalType(
            kind="namedtuple",
            name=name,
            args=[f.type for f in info.fields],
            metadata={"fields": field_data, "methods": info.methods},
        )


# ---------------------------------------------------------------------------
# 8. DataclassHandler – handle @dataclass
# ---------------------------------------------------------------------------

@dataclass
class DataclassFieldInfo:
    name: str
    type: InternalType
    has_default: bool = False
    has_default_factory: bool = False
    init: bool = True
    repr_: bool = True
    compare: bool = True
    hash_: Optional[bool] = None
    kw_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    alias: Optional[str] = None


@dataclass
class DataclassInfo:
    name: str
    fields: List[DataclassFieldInfo] = field(default_factory=list)
    frozen: bool = False
    order: bool = False
    eq: bool = True
    unsafe_hash: bool = False
    match_args: bool = True
    slots: bool = False
    kw_only: bool = False
    has_post_init: bool = False
    bases: List[str] = field(default_factory=list)


class DataclassHandler:
    """Parse ``@dataclass`` decorated classes."""

    def __init__(self) -> None:
        self._dataclasses: Dict[str, DataclassInfo] = {}

    def detect_dataclass(self, node: ast.ClassDef) -> Optional[DataclassInfo]:
        dc_kwargs = self._find_dataclass_decorator(node)
        if dc_kwargs is None:
            return None

        frozen = dc_kwargs.get("frozen", False)
        order = dc_kwargs.get("order", False)
        eq = dc_kwargs.get("eq", True)
        unsafe_hash = dc_kwargs.get("unsafe_hash", False)
        match_args = dc_kwargs.get("match_args", True)
        slots = dc_kwargs.get("slots", False)
        kw_only = dc_kwargs.get("kw_only", False)

        has_post_init = any(
            isinstance(s, ast.FunctionDef) and s.name == "__post_init__"
            for s in node.body
        )

        bases = [self._base_name(b) for b in node.bases]
        bases = [b for b in bases if b is not None]

        fields = self._extract_fields(node, kw_only)

        # inherit fields from known dataclass bases
        inherited: List[DataclassFieldInfo] = []
        for b in bases:
            parent = self._dataclasses.get(b)
            if parent:
                inherited.extend(parent.fields)

        info = DataclassInfo(
            name=node.name,
            fields=inherited + fields,
            frozen=frozen,
            order=order,
            eq=eq,
            unsafe_hash=unsafe_hash,
            match_args=match_args,
            slots=slots,
            kw_only=kw_only,
            has_post_init=has_post_init,
            bases=bases,
        )
        self._dataclasses[node.name] = info
        return info

    def _find_dataclass_decorator(self, node: ast.ClassDef) -> Optional[Dict[str, Any]]:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == "dataclass":
                return {}
            if isinstance(dec, ast.Attribute) and dec.attr == "dataclass":
                return {}
            if isinstance(dec, ast.Call):
                func = dec.func
                fname = ""
                if isinstance(func, ast.Name):
                    fname = func.id
                elif isinstance(func, ast.Attribute):
                    fname = func.attr
                if fname == "dataclass":
                    kwargs: Dict[str, Any] = {}
                    for kw in dec.keywords:
                        if kw.arg and isinstance(kw.value, ast.Constant):
                            kwargs[kw.arg] = kw.value.value
                    return kwargs
        return None

    def _extract_fields(self, node: ast.ClassDef, class_kw_only: bool) -> List[DataclassFieldInfo]:
        fields: List[DataclassFieldInfo] = []
        parser = AnnotationParser()
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue
            if not isinstance(stmt.target, ast.Name):
                continue
            ftype = parser.parse_annotation(stmt.annotation)
            if ftype.is_classvar:
                continue  # ClassVar fields are not dataclass fields

            fi = DataclassFieldInfo(name=stmt.target.id, type=ftype, kw_only=class_kw_only)

            if stmt.value is not None:
                if isinstance(stmt.value, ast.Call):
                    self._parse_field_call(stmt.value, fi)
                else:
                    fi.has_default = True
            fields.append(fi)
        return fields

    def _parse_field_call(self, call: ast.Call, fi: DataclassFieldInfo) -> None:
        func_name = ""
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        if func_name != "field":
            fi.has_default = True
            return
        for kw in call.keywords:
            if not kw.arg:
                continue
            if kw.arg == "default":
                fi.has_default = True
            elif kw.arg == "default_factory":
                fi.has_default_factory = True
            elif kw.arg == "init" and isinstance(kw.value, ast.Constant):
                fi.init = bool(kw.value.value)
            elif kw.arg == "repr" and isinstance(kw.value, ast.Constant):
                fi.repr_ = bool(kw.value.value)
            elif kw.arg == "compare" and isinstance(kw.value, ast.Constant):
                fi.compare = bool(kw.value.value)
            elif kw.arg == "hash" and isinstance(kw.value, ast.Constant):
                fi.hash_ = bool(kw.value.value) if kw.value.value is not None else None
            elif kw.arg == "kw_only" and isinstance(kw.value, ast.Constant):
                fi.kw_only = bool(kw.value.value)
            elif kw.arg == "alias" and isinstance(kw.value, ast.Constant):
                fi.alias = str(kw.value.value)
            elif kw.arg == "metadata":
                fi.metadata["raw_metadata"] = ast.dump(kw.value)

    @staticmethod
    def _base_name(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                return node.value.id
        return None

    def get_dataclass(self, name: str) -> Optional[DataclassInfo]:
        return self._dataclasses.get(name)

    def init_signature(self, name: str) -> Optional[InternalType]:
        """Build the effective ``__init__`` signature type."""
        info = self._dataclasses.get(name)
        if info is None:
            return None
        param_types: List[InternalType] = []
        for f in info.fields:
            if not f.init:
                continue
            param_types.append(f.type)
        ret = make_basic(name)
        return make_callable(param_types, ret)

    def post_init_type(self, name: str) -> Optional[InternalType]:
        info = self._dataclasses.get(name)
        if info is None or not info.has_post_init:
            return None
        init_var_params: List[InternalType] = []
        for f in info.fields:
            if f.type.kind == "generic" and f.type.name == "InitVar":
                init_var_params.append(f.type.args[0] if f.type.args else InternalType(kind="any", name="Any"))
        return make_callable(init_var_params, InternalType(kind="none", name="None"))

    def dataclass_to_internal_type(self, name: str) -> InternalType:
        info = self._dataclasses.get(name)
        if info is None:
            return make_basic(name)
        field_data = {
            f.name: {
                "type_str": f.type.to_annotation_string(),
                "has_default": f.has_default or f.has_default_factory,
                "init": f.init,
                "kw_only": f.kw_only,
            }
            for f in info.fields
        }
        return InternalType(
            kind="dataclass_type",
            name=name,
            metadata={
                "fields": field_data,
                "frozen": info.frozen,
                "order": info.order,
                "eq": info.eq,
                "slots": info.slots,
                "has_post_init": info.has_post_init,
            },
        )


# ---------------------------------------------------------------------------
# 9. EnumHandler – handle Enum types
# ---------------------------------------------------------------------------

@dataclass
class EnumMemberInfo:
    name: str
    value: Any
    value_type: InternalType
    is_auto: bool = False


@dataclass
class EnumInfo:
    name: str
    members: List[EnumMemberInfo] = field(default_factory=list)
    base_type: Optional[str] = None  # "int" for IntEnum, "str" for StrEnum
    is_flag: bool = False
    bases: List[str] = field(default_factory=list)


class EnumHandler:
    """Parse ``Enum`` subclasses."""

    _ENUM_BASES: FrozenSet[str] = frozenset({
        "Enum", "IntEnum", "StrEnum", "Flag", "IntFlag",
        "enum.Enum", "enum.IntEnum", "enum.StrEnum", "enum.Flag", "enum.IntFlag",
    })

    def __init__(self) -> None:
        self._enums: Dict[str, EnumInfo] = {}

    def detect_enum(self, node: ast.ClassDef) -> Optional[EnumInfo]:
        enum_base: Optional[str] = None
        is_flag = False
        bases: List[str] = []
        for base in node.bases:
            bname = self._base_name(base)
            if bname and bname.split(".")[-1] in ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"):
                enum_base = bname.split(".")[-1]
                if "Flag" in enum_base:
                    is_flag = True
            elif bname:
                bases.append(bname)
        if enum_base is None:
            return None

        base_type: Optional[str] = None
        if enum_base == "IntEnum" or enum_base == "IntFlag":
            base_type = "int"
        elif enum_base == "StrEnum":
            base_type = "str"
        else:
            # check for class Color(str, Enum): pattern
            for bname in bases:
                if bname in ("int", "str", "float", "bytes"):
                    base_type = bname
                    break

        members = self._extract_members(node, base_type)

        info = EnumInfo(
            name=node.name,
            members=members,
            base_type=base_type,
            is_flag=is_flag,
            bases=bases,
        )
        self._enums[node.name] = info
        return info

    def _extract_members(self, node: ast.ClassDef, base_type: Optional[str]) -> List[EnumMemberInfo]:
        members: List[EnumMemberInfo] = []
        auto_counter = 1
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    name = target.id
                    if name.startswith("_"):
                        continue
                    is_auto = self._is_auto_call(stmt.value)
                    if is_auto:
                        value = auto_counter
                        auto_counter += 1
                        val_type = make_basic("int")
                    elif isinstance(stmt.value, ast.Constant):
                        value = stmt.value.value
                        val_type = make_basic(type(value).__name__)
                    elif isinstance(stmt.value, ast.Tuple):
                        value = tuple(
                            e.value if isinstance(e, ast.Constant) else None
                            for e in stmt.value.elts
                        )
                        val_type = make_tuple([make_basic(type(v).__name__) for v in value])
                    else:
                        value = None
                        val_type = make_basic(base_type or "Any")
                    members.append(EnumMemberInfo(name=name, value=value, value_type=val_type, is_auto=is_auto))
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                if name.startswith("_"):
                    continue
                parser = AnnotationParser()
                val_type = parser.parse_annotation(stmt.annotation)
                value = None
                is_auto = False
                if stmt.value is not None:
                    is_auto = self._is_auto_call(stmt.value)
                    if is_auto:
                        value = auto_counter
                        auto_counter += 1
                    elif isinstance(stmt.value, ast.Constant):
                        value = stmt.value.value
                members.append(EnumMemberInfo(name=name, value=value, value_type=val_type, is_auto=is_auto))
        return members

    @staticmethod
    def _is_auto_call(node: ast.AST) -> bool:
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "auto":
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "auto":
                return True
        return False

    @staticmethod
    def _base_name(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: List[str] = []
            cur = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            parts.reverse()
            return ".".join(parts)
        return None

    def get_enum(self, name: str) -> Optional[EnumInfo]:
        return self._enums.get(name)

    def enum_to_internal_type(self, name: str) -> InternalType:
        info = self._enums.get(name)
        if info is None:
            return make_basic(name)
        member_data = {
            m.name: {
                "value": repr(m.value),
                "value_type": m.value_type.to_annotation_string(),
                "is_auto": m.is_auto,
            }
            for m in info.members
        }
        # A Literal of all member values can represent the type in refinement context
        literal_vals = [m.name for m in info.members]
        return InternalType(
            kind="enum_type",
            name=name,
            literal_values=literal_vals,
            metadata={
                "members": member_data,
                "base_type": info.base_type,
                "is_flag": info.is_flag,
            },
        )

    def value_type_of_member(self, enum_name: str, member_name: str) -> InternalType:
        info = self._enums.get(enum_name)
        if info is None:
            return InternalType(kind="any", name="Any")
        for m in info.members:
            if m.name == member_name:
                return m.value_type
        return InternalType(kind="any", name="Any")


# ---------------------------------------------------------------------------
# 10. OverloadHandler – handle @overload
# ---------------------------------------------------------------------------

@dataclass
class OverloadSignature:
    """One @overload signature."""
    param_names: List[str]
    param_types: List[InternalType]
    return_type: InternalType
    has_self: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class OverloadHandler:
    """Collect and resolve ``@overload`` decorated functions."""

    def __init__(self) -> None:
        self._overloads: Dict[str, List[OverloadSignature]] = {}
        self._implementation_present: Dict[str, bool] = {}

    def is_overload_decorator(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "overload"
        if isinstance(node, ast.Attribute):
            return node.attr == "overload"
        return False

    def collect_overloads(self, stmts: List[ast.stmt], scope: str = "") -> None:
        """Scan a list of statements for overloaded functions.

        Groups overload signatures by qualified name.
        """
        parser = AnnotationParser()
        for stmt in stmts:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            is_ol = any(self.is_overload_decorator(d) for d in stmt.decorator_list)
            qualified = f"{scope}.{stmt.name}" if scope else stmt.name
            if is_ol:
                sig = self._build_signature(stmt, parser)
                self._overloads.setdefault(qualified, []).append(sig)
            else:
                if qualified in self._overloads:
                    self._implementation_present[qualified] = True

    def _build_signature(
        self, func: Union[ast.FunctionDef, ast.AsyncFunctionDef], parser: AnnotationParser
    ) -> OverloadSignature:
        param_names: List[str] = []
        param_types: List[InternalType] = []
        has_self = False
        all_args = func.args.posonlyargs + func.args.args + func.args.kwonlyargs
        for i, arg in enumerate(all_args):
            if i == 0 and arg.arg in ("self", "cls"):
                has_self = True
                continue
            param_names.append(arg.arg)
            if arg.annotation:
                param_types.append(parser.parse_annotation(arg.annotation))
            else:
                param_types.append(InternalType(kind="any", name="Any"))
        if func.args.vararg:
            param_names.append("*" + func.args.vararg.arg)
            ann = func.args.vararg.annotation
            param_types.append(parser.parse_annotation(ann) if ann else InternalType(kind="any", name="Any"))
        if func.args.kwarg:
            param_names.append("**" + func.args.kwarg.arg)
            ann = func.args.kwarg.annotation
            param_types.append(parser.parse_annotation(ann) if ann else InternalType(kind="any", name="Any"))
        ret = parser.parse_annotation(func.returns) if func.returns else InternalType(kind="any", name="Any")
        return OverloadSignature(
            param_names=param_names,
            param_types=param_types,
            return_type=ret,
            has_self=has_self,
        )

    def match_overload(
        self,
        func_name: str,
        arg_types: List[InternalType],
    ) -> Optional[OverloadSignature]:
        """Find the best matching overload for the given argument types.

        Uses a simple scoring heuristic: count how many parameter types match exactly.
        """
        sigs = self._overloads.get(func_name, [])
        if not sigs:
            return None
        best: Optional[OverloadSignature] = None
        best_score = -1
        for sig in sigs:
            score = self._score_match(sig, arg_types)
            if score > best_score:
                best_score = score
                best = sig
        return best

    def _score_match(self, sig: OverloadSignature, arg_types: List[InternalType]) -> int:
        score = 0
        for i, pt in enumerate(sig.param_types):
            if i >= len(arg_types):
                break
            at = arg_types[i]
            if pt.kind == "any" or at.kind == "any":
                score += 1
            elif pt == at:
                score += 10
            elif pt.kind == "union" and at in pt.flatten_union():
                score += 5
            elif pt.kind == at.kind and pt.name == at.name:
                score += 7
            elif at.kind == "literal" and pt.kind == "basic":
                # Literal[1] matches int
                if at.literal_values:
                    val = at.literal_values[0]
                    if isinstance(val, int) and pt.name == "int":
                        score += 6
                    elif isinstance(val, str) and pt.name == "str":
                        score += 6
        # penalise arity mismatch
        diff = abs(len(sig.param_types) - len(arg_types))
        score -= diff * 2
        return score

    def return_type_for(self, func_name: str, arg_types: List[InternalType]) -> InternalType:
        sig = self.match_overload(func_name, arg_types)
        if sig is None:
            return InternalType(kind="any", name="Any")
        return sig.return_type

    def get_overloads(self, func_name: str) -> List[OverloadSignature]:
        return self._overloads.get(func_name, [])

    def overloaded_to_internal_type(self, func_name: str) -> InternalType:
        sigs = self._overloads.get(func_name, [])
        if not sigs:
            return InternalType(kind="any", name="Any")
        sig_types: List[InternalType] = []
        for sig in sigs:
            sig_types.append(make_callable(sig.param_types, sig.return_type))
        return InternalType(kind="overloaded", name=func_name, args=sig_types)

    def validate_overloads(self, func_name: str) -> List[str]:
        """Return a list of warnings/errors for the overloads of *func_name*."""
        sigs = self._overloads.get(func_name, [])
        errors: List[str] = []
        if len(sigs) < 2:
            errors.append(f"{func_name}: need at least 2 overloads, found {len(sigs)}")
        if not self._implementation_present.get(func_name, False):
            errors.append(f"{func_name}: missing non-@overload implementation")
        # check return types are compatible
        ret_kinds = {s.return_type.kind for s in sigs}
        if "never" in ret_kinds and len(ret_kinds) > 1:
            errors.append(f"{func_name}: mixing Never with other return types")
        return errors


# ---------------------------------------------------------------------------
# 11. TypeCommentParser – parse # type: comments
# ---------------------------------------------------------------------------

class TypeCommentParser:
    """Parse legacy ``# type:`` comments (PEP 484)."""

    _TYPE_COMMENT_RE = re.compile(r"#\s*type\s*:\s*(.+)")
    _TYPE_IGNORE_RE = re.compile(r"#\s*type\s*:\s*ignore\s*(\[.+?\])?")

    def __init__(self) -> None:
        self._parser = AnnotationParser()

    def parse_type_comment(self, comment: str) -> Optional[InternalType]:
        """Parse a ``# type: <type>`` comment and return the type.

        Returns ``None`` if the comment is a ``# type: ignore``.
        """
        ignore_m = self._TYPE_IGNORE_RE.match(comment)
        if ignore_m:
            return None
        m = self._TYPE_COMMENT_RE.match(comment)
        if not m:
            return None
        type_str = m.group(1).strip()
        if type_str.startswith("ignore"):
            return None
        return self._parser.parse_annotation_string(type_str)

    def parse_func_type_comment(self, comment: str) -> Optional[Tuple[List[InternalType], InternalType]]:
        """Parse function-level ``# type: (int, str) -> bool`` comments."""
        m = self._TYPE_COMMENT_RE.match(comment)
        if not m:
            return None
        body = m.group(1).strip()
        arrow_idx = body.find("->")
        if arrow_idx == -1:
            return None
        params_str = body[:arrow_idx].strip()
        ret_str = body[arrow_idx + 2:].strip()

        # parse params
        if params_str.startswith("(") and params_str.endswith(")"):
            params_str = params_str[1:-1].strip()
        param_types: List[InternalType] = []
        if params_str and params_str != "...":
            for part in self._split_type_args(params_str):
                part = part.strip()
                if part:
                    param_types.append(self._parser.parse_annotation_string(part))

        ret_type = self._parser.parse_annotation_string(ret_str)
        return param_types, ret_type

    @staticmethod
    def _split_type_args(s: str) -> List[str]:
        """Split comma-separated type args, respecting bracket nesting."""
        parts: List[str] = []
        depth = 0
        current: List[str] = []
        for ch in s:
            if ch in ("(", "[", "{"):
                depth += 1
                current.append(ch)
            elif ch in (")", "]", "}"):
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts

    def extract_type_comments(self, source: str) -> Dict[int, InternalType]:
        """Extract all ``# type:`` comments from *source* keyed by line number."""
        result: Dict[int, InternalType] = {}
        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            idx = stripped.find("# type:")
            if idx == -1:
                idx = stripped.find("#type:")
            if idx == -1:
                continue
            comment = stripped[idx:]
            parsed = self.parse_type_comment(comment)
            if parsed is not None:
                result[lineno] = parsed
        return result

    def extract_type_ignore_codes(self, source: str) -> Dict[int, List[str]]:
        """Return ``{lineno: [error_codes]}`` for ``# type: ignore[code,...]``."""
        result: Dict[int, List[str]] = {}
        for lineno, line in enumerate(source.splitlines(), start=1):
            m = self._TYPE_IGNORE_RE.search(line)
            if m:
                codes_str = m.group(1)
                if codes_str:
                    codes = [c.strip() for c in codes_str.strip("[]").split(",")]
                else:
                    codes = []
                result[lineno] = codes
        return result

    def merge_type_comments(
        self,
        func_node: ast.FunctionDef,
        source_lines: List[str],
    ) -> Dict[str, InternalType]:
        """Merge ``# type:`` comments into function annotations.

        If the function has a ``# type: (...) -> ...`` comment on its ``def`` line,
        apply those types to the arguments.  Individual variable ``# type:`` comments
        on assignment lines are also resolved.
        """
        result: Dict[str, InternalType] = {}
        # function-level type comment
        def_lineno = func_node.lineno  # 1-indexed
        if 0 < def_lineno <= len(source_lines):
            line = source_lines[def_lineno - 1]
            idx = line.find("# type:")
            if idx != -1:
                comment = line[idx:]
                ftc = self.parse_func_type_comment(comment)
                if ftc is not None:
                    param_types, ret_type = ftc
                    args_no_self = [
                        a for a in func_node.args.args
                        if a.arg not in ("self", "cls")
                    ]
                    for i, arg in enumerate(args_no_self):
                        if i < len(param_types):
                            result[arg.arg] = param_types[i]
                    result["return"] = ret_type
        # inline type comments on assignments
        for stmt in ast.walk(func_node):
            if isinstance(stmt, ast.Assign) and hasattr(stmt, "lineno"):
                ln = stmt.lineno
                if 0 < ln <= len(source_lines):
                    line = source_lines[ln - 1]
                    cidx = line.find("# type:")
                    if cidx != -1:
                        comment = line[cidx:]
                        parsed = self.parse_type_comment(comment)
                        if parsed is not None and stmt.targets:
                            t = stmt.targets[0]
                            if isinstance(t, ast.Name):
                                result[t.id] = parsed
        return result


# ---------------------------------------------------------------------------
# 12. StubFileParser – parse .pyi stub files
# ---------------------------------------------------------------------------

@dataclass
class StubFunction:
    name: str
    params: Dict[str, InternalType]
    return_type: InternalType
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    overloads: List[OverloadSignature] = field(default_factory=list)


@dataclass
class StubClass:
    name: str
    bases: List[str] = field(default_factory=list)
    methods: List[StubFunction] = field(default_factory=list)
    attributes: Dict[str, InternalType] = field(default_factory=dict)
    type_params: List[str] = field(default_factory=list)


@dataclass
class StubModule:
    path: str
    functions: List[StubFunction] = field(default_factory=list)
    classes: List[StubClass] = field(default_factory=list)
    variables: Dict[str, InternalType] = field(default_factory=dict)
    type_aliases: Dict[str, InternalType] = field(default_factory=dict)
    imports: Dict[str, str] = field(default_factory=dict)


class StubFileParser:
    """Parse ``.pyi`` stub files and extract type information."""

    def __init__(self) -> None:
        self._parser = AnnotationParser()
        self._overload_handler = OverloadHandler()
        self._generic_handler = GenericHandler()
        self._stubs: Dict[str, StubModule] = {}

    def parse_stub_file(self, path: str) -> StubModule:
        source = Path(path).read_text(encoding="utf-8")
        return self.parse_stub_source(source, path)

    def parse_stub_source(self, source: str, path: str = "<stub>") -> StubModule:
        tree = ast.parse(source, filename=path, type_comments=True)
        module = StubModule(path=path)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                cls = self._parse_class(node)
                module.classes.append(cls)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = self._parse_function(node)
                module.functions.append(func)
            elif isinstance(node, ast.AnnAssign):
                name, ty = self._parse_variable(node)
                if name:
                    module.variables[name] = ty
            elif isinstance(node, ast.Assign):
                self._parse_assign(node, module)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._parse_import(node, module)

        self._overload_handler.collect_overloads(list(ast.iter_child_nodes(tree)))
        # attach overloads to functions
        for func in module.functions:
            overloads = self._overload_handler.get_overloads(func.name)
            if overloads:
                func.overloads = overloads

        self._stubs[path] = module
        return module

    def _parse_class(self, node: ast.ClassDef) -> StubClass:
        bases = [self._base_name_str(b) for b in node.bases]
        bases = [b for b in bases if b]
        type_params: List[str] = []
        result = self._generic_handler.detect_generic_class(node)
        if result:
            _, type_params = result

        methods: List[StubFunction] = []
        attributes: Dict[str, InternalType] = {}
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._parse_function(stmt))
            elif isinstance(stmt, ast.AnnAssign):
                name, ty = self._parse_variable(stmt)
                if name:
                    attributes[name] = ty

        return StubClass(
            name=node.name,
            bases=bases,
            methods=methods,
            attributes=attributes,
            type_params=type_params,
        )

    def _parse_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> StubFunction:
        anns = self._parser.parse_function_annotations(node)
        decorators = [self._decorator_name(d) for d in node.decorator_list]
        decorators = [d for d in decorators if d]
        params = {k: v for k, v in anns.items() if k != "return"}
        return StubFunction(
            name=node.name,
            params=params,
            return_type=anns.get("return", InternalType(kind="any", name="Any")),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=decorators,
        )

    def _parse_variable(self, node: ast.AnnAssign) -> Tuple[str, InternalType]:
        name = ""
        if isinstance(node.target, ast.Name):
            name = node.target.id
        ty = self._parser.parse_annotation(node.annotation)
        return name, ty

    def _parse_assign(self, node: ast.Assign, module: StubModule) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return
        name = node.targets[0].id
        # type alias: X = Union[int, str]
        if isinstance(node.value, (ast.Subscript, ast.Name, ast.Attribute, ast.BinOp)):
            ty = self._parser.parse_annotation(node.value)
            module.type_aliases[name] = ty
        # TypeVar
        tv = self._generic_handler.parse_typevar_definition(node)
        if tv:
            module.variables[name] = make_typevar(tv.name, tv.bound, tv.constraints, tv.covariant, tv.contravariant)

    def _parse_import(self, node: Union[ast.Import, ast.ImportFrom], module: StubModule) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module.imports[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                full = f"{node.module}.{alias.name}"
                module.imports[alias.asname or alias.name] = full

    @staticmethod
    def _base_name_str(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                return node.value.id
        return ""

    @staticmethod
    def _decorator_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
            if isinstance(node.func, ast.Attribute):
                return node.func.attr
        return ""

    def merge_with_source(
        self,
        stub_module: StubModule,
        source_tree: ast.Module,
    ) -> Dict[str, InternalType]:
        """Merge stub type information with a source module AST.

        Returns a mapping of dotted names to types.  Stub types override any
        annotations in the source.
        """
        merged: Dict[str, InternalType] = {}
        parser = AnnotationParser()

        # collect source annotations
        for node in ast.walk(source_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                anns = parser.parse_function_annotations(node)
                for k, v in anns.items():
                    merged[f"{node.name}.{k}"] = v
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                merged[node.target.id] = parser.parse_annotation(node.annotation)

        # override with stub
        for func in stub_module.functions:
            for k, v in func.params.items():
                merged[f"{func.name}.{k}"] = v
            merged[f"{func.name}.return"] = func.return_type
        for cls in stub_module.classes:
            for attr_name, attr_ty in cls.attributes.items():
                merged[f"{cls.name}.{attr_name}"] = attr_ty
            for method in cls.methods:
                for k, v in method.params.items():
                    merged[f"{cls.name}.{method.name}.{k}"] = v
                merged[f"{cls.name}.{method.name}.return"] = method.return_type
        for vname, vtype in stub_module.variables.items():
            merged[vname] = vtype
        for aname, atype in stub_module.type_aliases.items():
            merged[aname] = atype
        return merged

    def get_stub(self, path: str) -> Optional[StubModule]:
        return self._stubs.get(path)

    def find_stub_for_module(self, module_name: str, search_paths: Optional[List[str]] = None) -> Optional[StubModule]:
        """Search for a .pyi stub matching *module_name*."""
        parts = module_name.split(".")
        candidates: List[Path] = []
        paths = [Path(p) for p in (search_paths or ["."])]
        for base in paths:
            candidates.append(base / "/".join(parts[:-1]) / (parts[-1] + ".pyi"))
            candidates.append(base / "/".join(parts) / "__init__.pyi")
            candidates.append(base / (module_name.replace(".", "/") + ".pyi"))
        for candidate in candidates:
            if candidate.is_file():
                return self.parse_stub_file(str(candidate))
        return None


# ---------------------------------------------------------------------------
# 13. TypeAnnotationResolver – resolve forward references
# ---------------------------------------------------------------------------

class TypeAnnotationResolver:
    """Resolve forward references, TYPE_CHECKING-guarded imports, and
    deferred annotation evaluation (PEP 563).
    """

    def __init__(self) -> None:
        self._known_types: Dict[str, InternalType] = {}
        self._type_checking_imports: Dict[str, str] = {}  # local_name → qualified
        self._resolved_cache: Dict[str, InternalType] = {}
        self._resolving: Set[str] = set()  # guard cycles

    def register_type(self, name: str, ty: InternalType) -> None:
        self._known_types[name] = ty

    def register_type_checking_import(self, local_name: str, qualified_name: str) -> None:
        self._type_checking_imports[local_name] = qualified_name

    def collect_type_checking_imports(self, tree: ast.Module) -> Dict[str, str]:
        """Find imports inside ``if TYPE_CHECKING:`` blocks."""
        result: Dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if not self._is_type_checking_guard(node.test):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.Import):
                    for alias in stmt.names:
                        local = alias.asname or alias.name
                        result[local] = alias.name
                elif isinstance(stmt, ast.ImportFrom) and stmt.module:
                    for alias in stmt.names:
                        local = alias.asname or alias.name
                        result[local] = f"{stmt.module}.{alias.name}"
        for k, v in result.items():
            self.register_type_checking_import(k, v)
        return result

    @staticmethod
    def _is_type_checking_guard(node: ast.AST) -> bool:
        if isinstance(node, ast.Name) and node.id == "TYPE_CHECKING":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING":
            return True
        return False

    def resolve(self, ty: InternalType) -> InternalType:
        """Resolve forward references in *ty*.

        Forward references are types with ``metadata["forward_ref"] == True``.
        """
        if ty.kind == "basic" and ty.metadata.get("forward_ref"):
            return self._resolve_name(ty.name)
        resolved_args = [self.resolve(a) for a in ty.args]
        if resolved_args != ty.args:
            result = copy.deepcopy(ty)
            result.args = resolved_args
            return result
        return ty

    def _resolve_name(self, name: str) -> InternalType:
        if name in self._resolved_cache:
            return copy.deepcopy(self._resolved_cache[name])
        if name in self._resolving:
            return InternalType(kind="basic", name=name, metadata={"forward_ref": True, "circular": True})
        self._resolving.add(name)
        try:
            if name in self._known_types:
                resolved = self._known_types[name]
                self._resolved_cache[name] = resolved
                return copy.deepcopy(resolved)
            if name in self._type_checking_imports:
                qualified = self._type_checking_imports[name]
                resolved = make_basic(qualified)
                resolved.metadata["resolved_from_type_checking"] = True
                self._resolved_cache[name] = resolved
                return copy.deepcopy(resolved)
            return make_basic(name)
        finally:
            self._resolving.discard(name)

    def resolve_all_in_module(self, tree: ast.Module) -> Dict[str, InternalType]:
        """Collect all annotations in *tree* and resolve forward references."""
        self.collect_type_checking_imports(tree)
        parser = AnnotationParser()
        annotations: Dict[str, InternalType] = {}

        # first pass: register class names
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self.register_type(node.name, make_basic(node.name))

        # second pass: collect and resolve annotations
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                anns = parser.parse_function_annotations(node)
                for k, v in anns.items():
                    resolved = self.resolve(v)
                    annotations[f"{node.name}.{k}"] = resolved
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                ty = parser.parse_annotation(node.annotation)
                resolved = self.resolve(ty)
                annotations[node.target.id] = resolved
        return annotations

    def resolve_deferred_annotations(self, module_source: str) -> Dict[str, InternalType]:
        """Evaluate PEP 563 deferred annotations by parsing the entire module."""
        tree = ast.parse(module_source)
        return self.resolve_all_in_module(tree)


# ---------------------------------------------------------------------------
# 14. AnnotationValidator – validate type annotations
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    severity: str  # "error", "warning", "info"
    message: str
    lineno: Optional[int] = None
    col_offset: Optional[int] = None
    node_dump: Optional[str] = None


class AnnotationValidator:
    """Validate type annotations for correctness and consistency."""

    def __init__(self) -> None:
        self._known_types: Set[str] = {
            "int", "float", "str", "bytes", "bool", "complex", "object",
            "list", "dict", "set", "frozenset", "tuple", "type", "range",
            "bytearray", "memoryview", "None", "NoneType",
            "Any", "NoReturn", "Never", "Self",
        }
        self._generic_handler = GenericHandler()
        self._issues: List[ValidationIssue] = []

    def add_known_type(self, name: str) -> None:
        self._known_types.add(name)

    def validate_module(self, tree: ast.Module) -> List[ValidationIssue]:
        self._issues = []
        # register all class names
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._known_types.add(node.name)
        # validate
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._validate_function(node)
            elif isinstance(node, ast.AnnAssign):
                self._validate_annotation_node(node.annotation, node)
        return list(self._issues)

    def _validate_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            if arg.annotation:
                self._validate_annotation_node(arg.annotation, arg)
        if node.returns:
            self._validate_annotation_node(node.returns, node)

    def _validate_annotation_node(self, ann: ast.AST, parent: ast.AST) -> None:
        parser = AnnotationParser()
        try:
            ty = parser.parse_annotation(ann)
        except Exception as exc:
            self._issues.append(ValidationIssue(
                severity="error",
                message=f"Failed to parse annotation: {exc}",
                lineno=getattr(ann, "lineno", None),
                col_offset=getattr(ann, "col_offset", None),
            ))
            return
        self._validate_type(ty, ann)

    def _validate_type(self, ty: InternalType, node: ast.AST) -> None:
        lineno = getattr(node, "lineno", None)
        col = getattr(node, "col_offset", None)

        if ty.kind == "basic" and ty.name and ty.name not in self._known_types:
            if not ty.metadata.get("forward_ref"):
                self._issues.append(ValidationIssue(
                    severity="warning",
                    message=f"Unknown type '{ty.name}'",
                    lineno=lineno,
                    col_offset=col,
                ))

        if ty.kind == "generic":
            self._validate_generic_params(ty, node)

        if ty.kind == "union" and not ty.args:
            self._issues.append(ValidationIssue(
                severity="error",
                message="Empty Union type",
                lineno=lineno,
                col_offset=col,
            ))

        if ty.kind == "callable" and not ty.args:
            self._issues.append(ValidationIssue(
                severity="warning",
                message="Callable without parameter/return type specification",
                lineno=lineno,
                col_offset=col,
            ))

        if ty.kind == "literal":
            self._validate_literal(ty, node)

        for arg in ty.args:
            self._validate_type(arg, node)

    def _validate_generic_params(self, ty: InternalType, node: ast.AST) -> None:
        lineno = getattr(node, "lineno", None)
        expected_params: Dict[str, int] = {
            "list": 1, "set": 1, "frozenset": 1,
            "dict": 2, "Mapping": 2, "MutableMapping": 2,
            "Type": 1, "type": 1,
            "Sequence": 1, "MutableSequence": 1,
            "Iterable": 1, "Iterator": 1,
            "collections.deque": 1, "collections.Counter": 1,
            "collections.defaultdict": 2, "collections.OrderedDict": 2,
            "collections.ChainMap": 2,
        }
        if ty.name in expected_params:
            expected = expected_params[ty.name]
            actual = len(ty.args)
            if actual != 0 and actual != expected:
                self._issues.append(ValidationIssue(
                    severity="error",
                    message=f"'{ty.name}' expects {expected} type parameter(s), got {actual}",
                    lineno=lineno,
                ))

    def _validate_literal(self, ty: InternalType, node: ast.AST) -> None:
        lineno = getattr(node, "lineno", None)
        if ty.literal_values is None or len(ty.literal_values) == 0:
            self._issues.append(ValidationIssue(
                severity="error",
                message="Literal type must have at least one value",
                lineno=lineno,
            ))
            return
        allowed_types = (int, str, bytes, bool, type(None), enum.Enum)
        for val in ty.literal_values:
            if val is not None and not isinstance(val, allowed_types):
                self._issues.append(ValidationIssue(
                    severity="warning",
                    message=f"Literal value {val!r} has unsupported type {type(val).__name__}",
                    lineno=lineno,
                ))

    def validate_overload_consistency(self, handler: OverloadHandler, func_name: str) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        sigs = handler.get_overloads(func_name)
        if len(sigs) < 2:
            issues.append(ValidationIssue(
                severity="error",
                message=f"@overload for '{func_name}': need >= 2 signatures",
            ))
        # check parameter count consistency
        param_counts = {len(s.param_types) for s in sigs}
        if len(param_counts) > 1:
            # not necessarily wrong, but worth flagging
            issues.append(ValidationIssue(
                severity="info",
                message=f"@overload for '{func_name}': varying parameter counts {param_counts}",
            ))
        return issues

    def get_issues(self) -> List[ValidationIssue]:
        return list(self._issues)


# ---------------------------------------------------------------------------
# 15. RefinementExtractor – extract refinement-relevant info
# ---------------------------------------------------------------------------

@dataclass
class RefinementConstraint:
    """A single refinement constraint extracted from an annotation."""
    kind: str  # "gt", "lt", "ge", "le", "eq", "ne", "multiple_of", "regex", "length", "predicate", "custom"
    value: Any = None
    description: str = ""
    source: str = ""  # "annotated", "beartype", "icontract"


@dataclass
class RefinementInfo:
    """All refinement information for a single type."""
    base_type: InternalType
    constraints: List[RefinementConstraint] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)


class RefinementExtractor:
    """Extract refinement-relevant information from annotations.

    Supports:
    - ``Annotated[int, Gt(0)]`` style (pydantic / beartype validators)
    - ``icontract`` ``@require`` / ``@ensure`` decorators
    - Custom refinement metadata markers
    """

    # known validator class names mapped to constraint kinds
    _VALIDATOR_MAP: Dict[str, str] = {
        "Gt": "gt",
        "Lt": "lt",
        "Ge": "ge",
        "Le": "le",
        "Eq": "eq",
        "Ne": "ne",
        "MultipleOf": "multiple_of",
        "Regex": "regex",
        "MinLen": "ge",
        "MaxLen": "le",
        "MinLength": "ge",
        "MaxLength": "le",
        "Positive": "gt",
        "Negative": "lt",
        "NonNegative": "ge",
        "NonPositive": "le",
        "IsFinite": "custom",
        "IsNotNan": "custom",
    }

    _COMPARISON_PATTERN = re.compile(
        r"(?:lambda\s+\w+\s*:\s*)?\w+\s*(>|<|>=|<=|==|!=)\s*(\S+)"
    )

    def __init__(self) -> None:
        self._parser = AnnotationParser()

    def extract_from_type(self, ty: InternalType) -> RefinementInfo:
        """Extract refinement constraints from an *InternalType*."""
        if ty.kind == "annotated" and ty.args:
            return self._extract_from_annotated(ty)
        if ty.kind == "literal":
            return self._extract_from_literal(ty)
        if ty.kind == "union":
            return self._extract_from_union(ty)
        if ty.kind == "typeguard":
            return self._extract_from_typeguard(ty)
        return RefinementInfo(base_type=ty)

    def _extract_from_annotated(self, ty: InternalType) -> RefinementInfo:
        base = ty.args[0]
        constraints: List[RefinementConstraint] = []
        raw_annotations = ty.metadata.get("annotations", [])
        for ann in raw_annotations:
            constraints.extend(self._parse_annotation_metadata(ann))
        return RefinementInfo(base_type=base, constraints=constraints)

    def _parse_annotation_metadata(self, ann: Any) -> List[RefinementConstraint]:
        constraints: List[RefinementConstraint] = []
        if isinstance(ann, str):
            # try to interpret as a class-name-based constraint
            for validator_name, kind in self._VALIDATOR_MAP.items():
                if ann == validator_name:
                    constraints.append(RefinementConstraint(
                        kind=kind,
                        description=f"{validator_name} constraint",
                        source="annotated",
                    ))
                    return constraints
            # try comparison pattern
            m = self._COMPARISON_PATTERN.match(ann)
            if m:
                op_map = {">": "gt", "<": "lt", ">=": "ge", "<=": "le", "==": "eq", "!=": "ne"}
                op = m.group(1)
                val_str = m.group(2)
                try:
                    val: Any = int(val_str)
                except ValueError:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str
                constraints.append(RefinementConstraint(
                    kind=op_map.get(op, "custom"),
                    value=val,
                    description=f"comparison {op} {val}",
                    source="annotated",
                ))
                return constraints
            # generic predicate
            constraints.append(RefinementConstraint(
                kind="predicate",
                value=ann,
                description=f"predicate: {ann}",
                source="annotated",
            ))
        elif isinstance(ann, dict):
            # structured metadata
            if "gt" in ann:
                constraints.append(RefinementConstraint(kind="gt", value=ann["gt"], source="annotated"))
            if "lt" in ann:
                constraints.append(RefinementConstraint(kind="lt", value=ann["lt"], source="annotated"))
            if "ge" in ann:
                constraints.append(RefinementConstraint(kind="ge", value=ann["ge"], source="annotated"))
            if "le" in ann:
                constraints.append(RefinementConstraint(kind="le", value=ann["le"], source="annotated"))
            if "multiple_of" in ann:
                constraints.append(RefinementConstraint(kind="multiple_of", value=ann["multiple_of"], source="annotated"))
            if "regex" in ann:
                constraints.append(RefinementConstraint(kind="regex", value=ann["regex"], source="annotated"))
            if "min_length" in ann:
                constraints.append(RefinementConstraint(kind="ge", value=ann["min_length"], description="min_length", source="annotated"))
            if "max_length" in ann:
                constraints.append(RefinementConstraint(kind="le", value=ann["max_length"], description="max_length", source="annotated"))
            if "predicate" in ann:
                constraints.append(RefinementConstraint(kind="predicate", value=ann["predicate"], source="annotated"))
        return constraints

    def _extract_from_literal(self, ty: InternalType) -> RefinementInfo:
        values = ty.literal_values or []
        constraints: List[RefinementConstraint] = []
        if values:
            constraints.append(RefinementConstraint(
                kind="eq",
                value=values if len(values) > 1 else values[0],
                description=f"must be one of {values!r}",
                source="literal",
            ))
        return RefinementInfo(base_type=ty, constraints=constraints)

    def _extract_from_union(self, ty: InternalType) -> RefinementInfo:
        # union branches may individually carry refinements
        all_constraints: List[RefinementConstraint] = []
        for branch in ty.args:
            branch_info = self.extract_from_type(branch)
            all_constraints.extend(branch_info.constraints)
        return RefinementInfo(base_type=ty, constraints=all_constraints)

    def _extract_from_typeguard(self, ty: InternalType) -> RefinementInfo:
        inner = ty.args[0] if ty.args else InternalType(kind="any", name="Any")
        constraints = [RefinementConstraint(
            kind="predicate",
            value=f"isinstance check for {inner.to_annotation_string()}",
            description=f"TypeGuard narrowing to {inner.to_annotation_string()}",
            source="typeguard",
        )]
        return RefinementInfo(base_type=make_basic("bool"), constraints=constraints)

    # ------------------------------------------------------------------
    # icontract support
    # ------------------------------------------------------------------

    def extract_icontract_conditions(
        self, func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> RefinementInfo:
        """Extract ``@icontract.require`` and ``@icontract.ensure`` conditions."""
        preconditions: List[str] = []
        postconditions: List[str] = []
        invariants: List[str] = []
        constraints: List[RefinementConstraint] = []

        for dec in func_node.decorator_list:
            dec_name, dec_args = self._parse_decorator(dec)
            if dec_name in ("require", "icontract.require"):
                cond_str = self._extract_lambda_or_string(dec_args)
                if cond_str:
                    preconditions.append(cond_str)
                    constraints.append(RefinementConstraint(
                        kind="predicate",
                        value=cond_str,
                        description=f"precondition: {cond_str}",
                        source="icontract",
                    ))
            elif dec_name in ("ensure", "icontract.ensure"):
                cond_str = self._extract_lambda_or_string(dec_args)
                if cond_str:
                    postconditions.append(cond_str)
                    constraints.append(RefinementConstraint(
                        kind="predicate",
                        value=cond_str,
                        description=f"postcondition: {cond_str}",
                        source="icontract",
                    ))
            elif dec_name in ("invariant", "icontract.invariant"):
                cond_str = self._extract_lambda_or_string(dec_args)
                if cond_str:
                    invariants.append(cond_str)

        ret_ty = self._parser.parse_annotation(func_node.returns) if func_node.returns else InternalType(kind="any", name="Any")
        return RefinementInfo(
            base_type=ret_ty,
            constraints=constraints,
            preconditions=preconditions,
            postconditions=postconditions,
            invariants=invariants,
        )

    def _parse_decorator(self, node: ast.AST) -> Tuple[str, List[ast.AST]]:
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                parts: List[str] = []
                cur: ast.AST = node.func
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                parts.reverse()
                name = ".".join(parts)
            return name, list(node.args)
        if isinstance(node, ast.Name):
            return node.id, []
        if isinstance(node, ast.Attribute):
            return node.attr, []
        return "", []

    @staticmethod
    def _extract_lambda_or_string(args: List[ast.AST]) -> Optional[str]:
        if not args:
            return None
        first = args[0]
        if isinstance(first, ast.Lambda):
            try:
                return ast.unparse(first)
            except Exception:
                return "<lambda>"
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
        try:
            return ast.unparse(first)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # beartype support
    # ------------------------------------------------------------------

    def extract_beartype_validators(self, ty: InternalType) -> List[RefinementConstraint]:
        """Extract beartype ``Is[lambda x: ...]`` validators from Annotated metadata."""
        constraints: List[RefinementConstraint] = []
        if ty.kind != "annotated":
            return constraints
        for ann in ty.metadata.get("annotations", []):
            if isinstance(ann, str) and ann.startswith("Is["):
                inner = ann[3:-1].strip() if ann.endswith("]") else ann[3:]
                constraints.append(RefinementConstraint(
                    kind="predicate",
                    value=inner,
                    description=f"beartype validator: {inner}",
                    source="beartype",
                ))
            elif isinstance(ann, str) and "Is" in ann:
                constraints.append(RefinementConstraint(
                    kind="predicate",
                    value=ann,
                    description=f"beartype validator: {ann}",
                    source="beartype",
                ))
        return constraints

    def extract_all(
        self,
        func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> Dict[str, RefinementInfo]:
        """Extract refinement info for every annotated parameter and return."""
        result: Dict[str, RefinementInfo] = {}
        parser = AnnotationParser()
        anns = parser.parse_function_annotations(func_node)
        for name, ty in anns.items():
            info = self.extract_from_type(ty)
            bt = self.extract_beartype_validators(ty)
            if bt:
                info.constraints.extend(bt)
            result[name] = info
        # icontract
        ic = self.extract_icontract_conditions(func_node)
        if ic.preconditions or ic.postconditions:
            if "return" in result:
                result["return"].preconditions.extend(ic.preconditions)
                result["return"].postconditions.extend(ic.postconditions)
                result["return"].constraints.extend(ic.constraints)
            else:
                result["return"] = ic
        return result


# ---------------------------------------------------------------------------
# 16. AnnotationStatistics – statistics about annotations
# ---------------------------------------------------------------------------

@dataclass
class AnnotationStats:
    total_functions: int = 0
    annotated_functions: int = 0
    partially_annotated_functions: int = 0
    total_parameters: int = 0
    annotated_parameters: int = 0
    total_returns: int = 0
    annotated_returns: int = 0
    total_variables: int = 0
    annotated_variables: int = 0
    type_distribution: Dict[str, int] = field(default_factory=dict)
    max_depth: int = 0
    avg_depth: float = 0.0
    complexity_histogram: Dict[int, int] = field(default_factory=dict)


class AnnotationStatistics:
    """Compute statistics about type annotations in a module or project."""

    def __init__(self) -> None:
        self._parser = AnnotationParser()

    def analyse_module(self, tree: ast.Module) -> AnnotationStats:
        stats = AnnotationStats()
        depths: List[int] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._analyse_function(node, stats, depths)
            elif isinstance(node, ast.AnnAssign):
                stats.total_variables += 1
                stats.annotated_variables += 1
                ty = self._parser.parse_annotation(node.annotation)
                self._record_type(ty, stats, depths)

        if depths:
            stats.max_depth = max(depths)
            stats.avg_depth = sum(depths) / len(depths)
        return stats

    def _analyse_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        stats: AnnotationStats,
        depths: List[int],
    ) -> None:
        stats.total_functions += 1
        all_args = node.args.args + node.args.posonlyargs + node.args.kwonlyargs
        params_annotated = 0
        params_total = 0
        for arg in all_args:
            if arg.arg in ("self", "cls"):
                continue
            params_total += 1
            stats.total_parameters += 1
            if arg.annotation:
                params_annotated += 1
                stats.annotated_parameters += 1
                ty = self._parser.parse_annotation(arg.annotation)
                self._record_type(ty, stats, depths)

        stats.total_returns += 1
        if node.returns:
            stats.annotated_returns += 1
            ret_ty = self._parser.parse_annotation(node.returns)
            self._record_type(ret_ty, stats, depths)

        if params_total == 0:
            has_params = False
        else:
            has_params = True

        if node.returns and (not has_params or params_annotated == params_total):
            stats.annotated_functions += 1
        elif node.returns or params_annotated > 0:
            stats.partially_annotated_functions += 1

    def _record_type(self, ty: InternalType, stats: AnnotationStats, depths: List[int]) -> None:
        d = ty.depth()
        depths.append(d)
        stats.complexity_histogram[d] = stats.complexity_histogram.get(d, 0) + 1
        for sub in ty.walk():
            kind_key = sub.kind
            if sub.kind == "basic":
                kind_key = sub.name or "basic"
            elif sub.kind == "generic":
                kind_key = f"generic:{sub.name}"
            stats.type_distribution[kind_key] = stats.type_distribution.get(kind_key, 0) + 1

    def analyse_source(self, source: str) -> AnnotationStats:
        tree = ast.parse(source)
        return self.analyse_module(tree)

    def analyse_files(self, paths: List[str]) -> AnnotationStats:
        combined = AnnotationStats()
        for path in paths:
            try:
                source = Path(path).read_text(encoding="utf-8")
                stats = self.analyse_source(source)
                self._merge_stats(combined, stats)
            except Exception:
                continue
        return combined

    @staticmethod
    def _merge_stats(target: AnnotationStats, source: AnnotationStats) -> None:
        target.total_functions += source.total_functions
        target.annotated_functions += source.annotated_functions
        target.partially_annotated_functions += source.partially_annotated_functions
        target.total_parameters += source.total_parameters
        target.annotated_parameters += source.annotated_parameters
        target.total_returns += source.total_returns
        target.annotated_returns += source.annotated_returns
        target.total_variables += source.total_variables
        target.annotated_variables += source.annotated_variables
        for k, v in source.type_distribution.items():
            target.type_distribution[k] = target.type_distribution.get(k, 0) + v
        if source.max_depth > target.max_depth:
            target.max_depth = source.max_depth
        for k, v in source.complexity_histogram.items():
            target.complexity_histogram[k] = target.complexity_histogram.get(k, 0) + v

    def coverage_report(self, stats: AnnotationStats) -> Dict[str, Any]:
        def pct(num: int, den: int) -> float:
            return round(num / den * 100, 1) if den else 0.0

        return {
            "function_coverage": pct(stats.annotated_functions, stats.total_functions),
            "parameter_coverage": pct(stats.annotated_parameters, stats.total_parameters),
            "return_coverage": pct(stats.annotated_returns, stats.total_returns),
            "variable_coverage": pct(stats.annotated_variables, stats.total_variables),
            "partially_annotated_functions": stats.partially_annotated_functions,
            "total_functions": stats.total_functions,
            "max_annotation_depth": stats.max_depth,
            "avg_annotation_depth": round(stats.avg_depth, 2),
            "top_types": sorted(
                stats.type_distribution.items(), key=lambda x: x[1], reverse=True
            )[:20],
        }

    def summary_text(self, stats: AnnotationStats) -> str:
        report = self.coverage_report(stats)
        lines = [
            "=== Annotation Statistics ===",
            f"Functions: {stats.annotated_functions}/{stats.total_functions} fully annotated "
            f"({report['function_coverage']}%)",
            f"Parameters: {stats.annotated_parameters}/{stats.total_parameters} annotated "
            f"({report['parameter_coverage']}%)",
            f"Returns: {stats.annotated_returns}/{stats.total_returns} annotated "
            f"({report['return_coverage']}%)",
            f"Variables: {stats.annotated_variables}/{stats.total_variables} annotated "
            f"({report['variable_coverage']}%)",
            f"Partially annotated functions: {stats.partially_annotated_functions}",
            f"Max annotation depth: {stats.max_depth}",
            f"Avg annotation depth: {report['avg_annotation_depth']}",
            "",
            "Top types:",
        ]
        for name, count in report["top_types"]:
            lines.append(f"  {name}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: module-level parse/resolve/validate pipeline
# ---------------------------------------------------------------------------

class TypeAnnotationPipeline:
    """Convenience wrapper combining all handlers for full module processing."""

    def __init__(self) -> None:
        self.parser = AnnotationParser()
        self.typing_handler = TypingModuleHandler()
        self.generic_handler = GenericHandler()
        self.protocol_handler = ProtocolHandler()
        self.typeddict_handler = TypedDictHandler()
        self.namedtuple_handler = NamedTupleHandler()
        self.dataclass_handler = DataclassHandler()
        self.enum_handler = EnumHandler()
        self.overload_handler = OverloadHandler()
        self.type_comment_parser = TypeCommentParser()
        self.stub_parser = StubFileParser()
        self.resolver = TypeAnnotationResolver()
        self.validator = AnnotationValidator()
        self.refinement_extractor = RefinementExtractor()
        self.statistics = AnnotationStatistics()

    def process_module(self, source: str, *, stub_source: Optional[str] = None) -> Dict[str, Any]:
        """Full pipeline: parse → detect special forms → resolve → validate → extract refinements."""
        tree = ast.parse(source)
        source_lines = source.splitlines()

        # 1. detect special class forms
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self.generic_handler.detect_generic_class(node)
                self.protocol_handler.detect_protocol(node)
                self.typeddict_handler.detect_typeddict(node)
                self.namedtuple_handler.detect_namedtuple_class(node)
                self.dataclass_handler.detect_dataclass(node)
                self.enum_handler.detect_enum(node)
                self.validator.add_known_type(node.name)
            elif isinstance(node, ast.Assign):
                self.generic_handler.parse_typevar_definition(node)
                self.typeddict_handler.parse_functional_typeddict(node)
                self.namedtuple_handler.detect_functional_namedtuple(node)

        # 2. collect overloads
        top_stmts = list(ast.iter_child_nodes(tree))
        self.overload_handler.collect_overloads(top_stmts)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self.overload_handler.collect_overloads(node.body, scope=node.name)

        # 3. resolve forward references
        annotations = self.resolver.resolve_all_in_module(tree)

        # 4. merge type comments
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                tc = self.type_comment_parser.merge_type_comments(node, source_lines)
                for k, v in tc.items():
                    key = f"{node.name}.{k}"
                    if key not in annotations or annotations[key].is_any():
                        annotations[key] = v

        # 5. merge stubs
        if stub_source:
            stub_mod = self.stub_parser.parse_stub_source(stub_source)
            merged = self.stub_parser.merge_with_source(stub_mod, tree)
            for k, v in merged.items():
                annotations[k] = v

        # 6. validate
        issues = self.validator.validate_module(tree)

        # 7. extract refinements
        refinements: Dict[str, RefinementInfo] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_refs = self.refinement_extractor.extract_all(node)
                for k, v in func_refs.items():
                    refinements[f"{node.name}.{k}"] = v

        # 8. statistics
        stats = self.statistics.analyse_module(tree)

        return {
            "annotations": annotations,
            "issues": issues,
            "refinements": refinements,
            "statistics": stats,
            "protocols": self.protocol_handler.all_protocols(),
            "typeddicts": {n: self.typeddict_handler.get_typeddict(n) for n in self.typeddict_handler._typeddicts},
            "dataclasses": {n: self.dataclass_handler.get_dataclass(n) for n in self.dataclass_handler._dataclasses},
            "enums": {n: self.enum_handler.get_enum(n) for n in self.enum_handler._enums},
            "overloads": {n: self.overload_handler.get_overloads(n) for n in self.overload_handler._overloads},
        }

    def process_file(self, path: str, *, stub_path: Optional[str] = None) -> Dict[str, Any]:
        source = Path(path).read_text(encoding="utf-8")
        stub_source: Optional[str] = None
        if stub_path:
            stub_source = Path(stub_path).read_text(encoding="utf-8")
        return self.process_module(source, stub_source=stub_source)
