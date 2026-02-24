from __future__ import annotations

import base64
import copy
import collections
import datetime
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
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INDENT = "    "


def _indent(text: str, level: int) -> str:
    prefix = _INDENT * level
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def _wrap_line(text: str, max_len: int = 100) -> List[str]:
    if len(text) <= max_len:
        return [text]
    parts: List[str] = []
    current = ""
    for token in text.split(", "):
        candidate = f"{current}, {token}" if current else token
        if len(candidate) > max_len and current:
            parts.append(current + ",")
            current = token
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


# ---------------------------------------------------------------------------
# 1. TypeScriptType hierarchy
# ---------------------------------------------------------------------------


class TypeScriptType(ABC):
    """Base for all TypeScript type representations."""

    @abstractmethod
    def to_ts_string(self, indent_level: int = 0) -> str:
        ...

    def __str__(self) -> str:  # pragma: no cover
        return self.to_ts_string()


@dataclass
class TSPrimitive(TypeScriptType):
    name: str  # string | number | boolean | null | undefined | void | never | unknown | any | bigint | symbol

    _VALID = frozenset(
        {"string", "number", "boolean", "null", "undefined", "void", "never", "unknown", "any", "bigint", "symbol"}
    )

    def __post_init__(self) -> None:
        if self.name not in self._VALID:
            raise ValueError(f"Invalid TS primitive: {self.name}")

    def to_ts_string(self, indent_level: int = 0) -> str:
        return self.name


@dataclass
class TSLiteral(TypeScriptType):
    value: Union[str, int, float, bool]

    def to_ts_string(self, indent_level: int = 0) -> str:
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        if isinstance(self.value, str):
            escaped = self.value.replace("\\", "\\\\").replace("'", "\\'")
            return f"'{escaped}'"
        return str(self.value)


@dataclass
class TSArray(TypeScriptType):
    element: TypeScriptType

    def to_ts_string(self, indent_level: int = 0) -> str:
        inner = self.element.to_ts_string(indent_level)
        if isinstance(self.element, (TSUnion, TSIntersection, TSFunction)):
            return f"({inner})[]"
        return f"{inner}[]"


@dataclass
class TSTuple(TypeScriptType):
    elements: List[TypeScriptType]
    rest_element: Optional[TypeScriptType] = None

    def to_ts_string(self, indent_level: int = 0) -> str:
        parts = [e.to_ts_string(indent_level) for e in self.elements]
        if self.rest_element:
            parts.append(f"...{self.rest_element.to_ts_string(indent_level)}[]")
        return f"[{', '.join(parts)}]"


@dataclass
class TSUnion(TypeScriptType):
    members: List[TypeScriptType]

    def to_ts_string(self, indent_level: int = 0) -> str:
        if not self.members:
            return "never"
        parts = []
        for m in self.members:
            s = m.to_ts_string(indent_level)
            parts.append(s)
        return " | ".join(parts)


@dataclass
class TSIntersection(TypeScriptType):
    members: List[TypeScriptType]

    def to_ts_string(self, indent_level: int = 0) -> str:
        if not self.members:
            return "unknown"
        parts = []
        for m in self.members:
            s = m.to_ts_string(indent_level)
            if isinstance(m, TSUnion):
                s = f"({s})"
            parts.append(s)
        return " & ".join(parts)


@dataclass
class TSObjectProperty:
    name: str
    type: TypeScriptType
    optional: bool = False
    readonly: bool = False
    jsdoc: Optional[str] = None

    def to_ts_string(self, indent_level: int = 0) -> str:
        prefix = "readonly " if self.readonly else ""
        opt = "?" if self.optional else ""
        type_str = self.type.to_ts_string(indent_level)
        return f"{prefix}{self.name}{opt}: {type_str}"


@dataclass
class TSIndexSignature:
    key_name: str
    key_type: TypeScriptType
    value_type: TypeScriptType
    readonly: bool = False

    def to_ts_string(self, indent_level: int = 0) -> str:
        prefix = "readonly " if self.readonly else ""
        return f"{prefix}[{self.key_name}: {self.key_type.to_ts_string(indent_level)}]: {self.value_type.to_ts_string(indent_level)}"


@dataclass
class TSCallSignature:
    params: List[Tuple[str, TypeScriptType]]
    return_type: TypeScriptType
    type_params: List[str] = field(default_factory=list)

    def to_ts_string(self, indent_level: int = 0) -> str:
        tp = f"<{', '.join(self.type_params)}>" if self.type_params else ""
        ps = ", ".join(f"{n}: {t.to_ts_string(indent_level)}" for n, t in self.params)
        return f"{tp}({ps}): {self.return_type.to_ts_string(indent_level)}"


@dataclass
class TSConstructSignature:
    params: List[Tuple[str, TypeScriptType]]
    return_type: TypeScriptType
    type_params: List[str] = field(default_factory=list)

    def to_ts_string(self, indent_level: int = 0) -> str:
        tp = f"<{', '.join(self.type_params)}>" if self.type_params else ""
        ps = ", ".join(f"{n}: {t.to_ts_string(indent_level)}" for n, t in self.params)
        return f"new {tp}({ps}): {self.return_type.to_ts_string(indent_level)}"


@dataclass
class TSObject(TypeScriptType):
    properties: List[TSObjectProperty] = field(default_factory=list)
    index_signatures: List[TSIndexSignature] = field(default_factory=list)
    call_signatures: List[TSCallSignature] = field(default_factory=list)
    construct_signatures: List[TSConstructSignature] = field(default_factory=list)

    def to_ts_string(self, indent_level: int = 0) -> str:
        if not self.properties and not self.index_signatures and not self.call_signatures and not self.construct_signatures:
            return "{}"
        lines: List[str] = []
        lines.append("{")
        inner = indent_level + 1
        for cs in self.construct_signatures:
            lines.append(f"{_INDENT * inner}{cs.to_ts_string(inner)};")
        for cs in self.call_signatures:
            lines.append(f"{_INDENT * inner}{cs.to_ts_string(inner)};")
        for idx in self.index_signatures:
            lines.append(f"{_INDENT * inner}{idx.to_ts_string(inner)};")
        for prop in self.properties:
            lines.append(f"{_INDENT * inner}{prop.to_ts_string(inner)};")
        lines.append(f"{_INDENT * indent_level}}}")
        return "\n".join(lines)


@dataclass
class TSFunctionParam:
    name: str
    type: TypeScriptType
    optional: bool = False
    rest: bool = False
    default_value: Optional[str] = None

    def to_ts_string(self, indent_level: int = 0) -> str:
        prefix = "..." if self.rest else ""
        opt = "?" if self.optional and not self.rest else ""
        return f"{prefix}{self.name}{opt}: {self.type.to_ts_string(indent_level)}"


@dataclass
class TSTypeParam:
    name: str
    constraint: Optional[TypeScriptType] = None
    default: Optional[TypeScriptType] = None
    variance: Optional[str] = None  # "in" | "out" | "in out"

    def to_ts_string(self, indent_level: int = 0) -> str:
        parts: List[str] = []
        if self.variance:
            parts.append(self.variance)
        parts.append(self.name)
        if self.constraint:
            parts.append(f"extends {self.constraint.to_ts_string(indent_level)}")
        if self.default:
            parts.append(f"= {self.default.to_ts_string(indent_level)}")
        return " ".join(parts)


@dataclass
class TSFunction(TypeScriptType):
    params: List[TSFunctionParam] = field(default_factory=list)
    return_type: TypeScriptType = field(default_factory=lambda: TSPrimitive("void"))
    type_params: List[TSTypeParam] = field(default_factory=list)
    is_async: bool = False
    is_generator: bool = False

    def to_ts_string(self, indent_level: int = 0) -> str:
        tp = ""
        if self.type_params:
            tp = f"<{', '.join(p.to_ts_string(indent_level) for p in self.type_params)}>"
        ps = ", ".join(p.to_ts_string(indent_level) for p in self.params)
        ret = self.return_type.to_ts_string(indent_level)
        return f"{tp}({ps}) => {ret}"


@dataclass
class TSConditional(TypeScriptType):
    check_type: TypeScriptType
    extends_type: TypeScriptType
    true_type: TypeScriptType
    false_type: TypeScriptType

    def to_ts_string(self, indent_level: int = 0) -> str:
        check = self.check_type.to_ts_string(indent_level)
        ext = self.extends_type.to_ts_string(indent_level)
        t = self.true_type.to_ts_string(indent_level)
        f = self.false_type.to_ts_string(indent_level)
        return f"{check} extends {ext} ? {t} : {f}"


@dataclass
class TSMapped(TypeScriptType):
    key_name: str
    key_source: TypeScriptType
    value_type: TypeScriptType
    readonly_mod: Optional[str] = None   # "+", "-", or None
    optional_mod: Optional[str] = None   # "+", "-", or None
    as_clause: Optional[TypeScriptType] = None

    def to_ts_string(self, indent_level: int = 0) -> str:
        ro = ""
        if self.readonly_mod == "+":
            ro = "+readonly "
        elif self.readonly_mod == "-":
            ro = "-readonly "
        elif self.readonly_mod is not None:
            ro = "readonly "

        opt = ""
        if self.optional_mod == "+":
            opt = "+?"
        elif self.optional_mod == "-":
            opt = "-?"
        elif self.optional_mod is not None:
            opt = "?"

        as_part = ""
        if self.as_clause:
            as_part = f" as {self.as_clause.to_ts_string(indent_level)}"

        key_src = self.key_source.to_ts_string(indent_level)
        val = self.value_type.to_ts_string(indent_level)
        return f"{{ {ro}[{self.key_name} in {key_src}{as_part}]{opt}: {val} }}"


@dataclass
class TSTemplateLiteral(TypeScriptType):
    head: str
    spans: List[Tuple[TypeScriptType, str]]  # (type, tail_text)

    def to_ts_string(self, indent_level: int = 0) -> str:
        result = f"`{self.head}"
        for tp, tail in self.spans:
            result += f"${{{tp.to_ts_string(indent_level)}}}{tail}"
        result += "`"
        return result


@dataclass
class TSKeyof(TypeScriptType):
    operand: TypeScriptType

    def to_ts_string(self, indent_level: int = 0) -> str:
        return f"keyof {self.operand.to_ts_string(indent_level)}"


@dataclass
class TSTypeof(TypeScriptType):
    operand: str  # variable name / qualified name

    def to_ts_string(self, indent_level: int = 0) -> str:
        return f"typeof {self.operand}"


@dataclass
class TSIndexedAccess(TypeScriptType):
    object_type: TypeScriptType
    index_type: TypeScriptType

    def to_ts_string(self, indent_level: int = 0) -> str:
        return f"{self.object_type.to_ts_string(indent_level)}[{self.index_type.to_ts_string(indent_level)}]"


@dataclass
class TSInfer(TypeScriptType):
    name: str

    def to_ts_string(self, indent_level: int = 0) -> str:
        return f"infer {self.name}"


@dataclass
class TSTypeReference(TypeScriptType):
    name: str
    type_args: List[TypeScriptType] = field(default_factory=list)

    def to_ts_string(self, indent_level: int = 0) -> str:
        if not self.type_args:
            return self.name
        args = ", ".join(a.to_ts_string(indent_level) for a in self.type_args)
        return f"{self.name}<{args}>"


# ---------------------------------------------------------------------------
# Helpers: convert dict → TypeScriptType
# ---------------------------------------------------------------------------


def _dict_to_ts_type(d: Any) -> TypeScriptType:
    """Convert a plain dict / primitive description to a TypeScriptType tree."""
    if isinstance(d, TypeScriptType):
        return d
    if isinstance(d, str):
        prim_names = TSPrimitive._VALID
        if d in prim_names:
            return TSPrimitive(d)
        return TSTypeReference(d)
    if isinstance(d, bool):
        return TSLiteral(d)
    if isinstance(d, (int, float)):
        return TSLiteral(d)
    if not isinstance(d, dict):
        return TSPrimitive("any")

    kind = d.get("kind", "reference")
    if kind == "primitive":
        return TSPrimitive(d.get("name", "any"))
    if kind == "literal":
        return TSLiteral(d.get("value", ""))
    if kind == "array":
        return TSArray(_dict_to_ts_type(d.get("element", "any")))
    if kind == "tuple":
        elems = [_dict_to_ts_type(e) for e in d.get("elements", [])]
        rest = _dict_to_ts_type(d["rest"]) if "rest" in d else None
        return TSTuple(elems, rest)
    if kind == "union":
        return TSUnion([_dict_to_ts_type(m) for m in d.get("members", [])])
    if kind == "intersection":
        return TSIntersection([_dict_to_ts_type(m) for m in d.get("members", [])])
    if kind == "object":
        props = []
        for p in d.get("properties", []):
            props.append(TSObjectProperty(
                name=p.get("name", ""),
                type=_dict_to_ts_type(p.get("type", "any")),
                optional=p.get("optional", False),
                readonly=p.get("readonly", False),
            ))
        idxs = []
        for ix in d.get("indexSignatures", []):
            idxs.append(TSIndexSignature(
                key_name=ix.get("keyName", "key"),
                key_type=_dict_to_ts_type(ix.get("keyType", "string")),
                value_type=_dict_to_ts_type(ix.get("valueType", "any")),
            ))
        return TSObject(properties=props, index_signatures=idxs)
    if kind == "function":
        params = []
        for p in d.get("params", []):
            params.append(TSFunctionParam(
                name=p.get("name", "arg"),
                type=_dict_to_ts_type(p.get("type", "any")),
                optional=p.get("optional", False),
                rest=p.get("rest", False),
            ))
        ret = _dict_to_ts_type(d.get("returnType", "void"))
        tps = [_dict_to_type_param(tp) for tp in d.get("typeParams", [])]
        return TSFunction(params=params, return_type=ret, type_params=tps)
    if kind == "conditional":
        return TSConditional(
            check_type=_dict_to_ts_type(d.get("check", "any")),
            extends_type=_dict_to_ts_type(d.get("extends", "any")),
            true_type=_dict_to_ts_type(d.get("true", "any")),
            false_type=_dict_to_ts_type(d.get("false", "never")),
        )
    if kind == "mapped":
        return TSMapped(
            key_name=d.get("keyName", "K"),
            key_source=_dict_to_ts_type(d.get("keySource", "string")),
            value_type=_dict_to_ts_type(d.get("valueType", "any")),
            readonly_mod=d.get("readonlyMod"),
            optional_mod=d.get("optionalMod"),
            as_clause=_dict_to_ts_type(d["asClause"]) if "asClause" in d else None,
        )
    if kind == "template_literal":
        spans = []
        for s in d.get("spans", []):
            spans.append((_dict_to_ts_type(s.get("type", "string")), s.get("tail", "")))
        return TSTemplateLiteral(head=d.get("head", ""), spans=spans)
    if kind == "keyof":
        return TSKeyof(_dict_to_ts_type(d.get("operand", "any")))
    if kind == "typeof":
        return TSTypeof(d.get("operand", ""))
    if kind == "indexed_access":
        return TSIndexedAccess(
            object_type=_dict_to_ts_type(d.get("object", "any")),
            index_type=_dict_to_ts_type(d.get("index", "string")),
        )
    if kind == "infer":
        return TSInfer(d.get("name", "U"))
    if kind == "reference":
        args = [_dict_to_ts_type(a) for a in d.get("typeArgs", [])]
        return TSTypeReference(d.get("name", "any"), args)
    return TSPrimitive("any")


def _dict_to_type_param(d: Any) -> TSTypeParam:
    if isinstance(d, str):
        return TSTypeParam(d)
    if isinstance(d, TSTypeParam):
        return d
    return TSTypeParam(
        name=d.get("name", "T"),
        constraint=_dict_to_ts_type(d["constraint"]) if "constraint" in d else None,
        default=_dict_to_ts_type(d["default"]) if "default" in d else None,
        variance=d.get("variance"),
    )


def _type_params_str(params: List[TSTypeParam], indent_level: int = 0) -> str:
    if not params:
        return ""
    return "<" + ", ".join(p.to_ts_string(indent_level) for p in params) + ">"


def _safe_identifier(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_$]", "_", name)
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned or "_"


# ---------------------------------------------------------------------------
# 2. InterfaceGenerator
# ---------------------------------------------------------------------------


@dataclass
class InterfaceDeclaration:
    name: str
    properties: List[TSObjectProperty] = field(default_factory=list)
    methods: List[Dict[str, Any]] = field(default_factory=list)
    index_signatures: List[TSIndexSignature] = field(default_factory=list)
    extends: List[str] = field(default_factory=list)
    type_params: List[TSTypeParam] = field(default_factory=list)
    call_signatures: List[TSCallSignature] = field(default_factory=list)
    construct_signatures: List[TSConstructSignature] = field(default_factory=list)
    jsdoc: Optional[str] = None
    exported: bool = True

    def render(self, indent_level: int = 0) -> str:
        lines: List[str] = []
        pad = _INDENT * indent_level
        inner = _INDENT * (indent_level + 1)

        if self.jsdoc:
            for jl in self.jsdoc.splitlines():
                lines.append(f"{pad}{jl}")

        exp = "export " if self.exported else ""
        tp = _type_params_str(self.type_params, indent_level)
        ext = ""
        if self.extends:
            ext = " extends " + ", ".join(self.extends)
        lines.append(f"{pad}{exp}interface {self.name}{tp}{ext} {{")

        for cs in self.construct_signatures:
            lines.append(f"{inner}{cs.to_ts_string(indent_level + 1)};")

        for cs in self.call_signatures:
            lines.append(f"{inner}{cs.to_ts_string(indent_level + 1)};")

        for idx in self.index_signatures:
            lines.append(f"{inner}{idx.to_ts_string(indent_level + 1)};")

        for prop in self.properties:
            lines.append(f"{inner}{prop.to_ts_string(indent_level + 1)};")

        for meth in self.methods:
            m_name = meth.get("name", "method")
            overloads = meth.get("overloads", [meth])
            for ov in overloads:
                ov_tp = _type_params_str(
                    [_dict_to_type_param(t) for t in ov.get("typeParams", [])],
                    indent_level + 1,
                )
                params_parts: List[str] = []
                for p in ov.get("params", []):
                    pt = _dict_to_ts_type(p.get("type", "any"))
                    opt = "?" if p.get("optional", False) else ""
                    rest = "..." if p.get("rest", False) else ""
                    params_parts.append(f"{rest}{p.get('name', 'arg')}{opt}: {pt.to_ts_string(indent_level + 1)}")
                ret = _dict_to_ts_type(ov.get("returnType", "void")).to_ts_string(indent_level + 1)
                lines.append(f"{inner}{m_name}{ov_tp}({', '.join(params_parts)}): {ret};")

        lines.append(f"{pad}}}")
        return "\n".join(lines)


@dataclass
class InterfaceGenerator:
    declarations: List[InterfaceDeclaration] = field(default_factory=list)

    def add_interface(self, spec: Dict[str, Any]) -> InterfaceDeclaration:
        props = []
        for p in spec.get("properties", []):
            props.append(TSObjectProperty(
                name=p.get("name", ""),
                type=_dict_to_ts_type(p.get("type", "any")),
                optional=p.get("optional", False),
                readonly=p.get("readonly", False),
                jsdoc=p.get("jsdoc"),
            ))

        idxs = []
        for ix in spec.get("indexSignatures", []):
            idxs.append(TSIndexSignature(
                key_name=ix.get("keyName", "key"),
                key_type=_dict_to_ts_type(ix.get("keyType", "string")),
                value_type=_dict_to_ts_type(ix.get("valueType", "any")),
                readonly=ix.get("readonly", False),
            ))

        call_sigs = []
        for cs in spec.get("callSignatures", []):
            call_sigs.append(TSCallSignature(
                params=[(p["name"], _dict_to_ts_type(p["type"])) for p in cs.get("params", [])],
                return_type=_dict_to_ts_type(cs.get("returnType", "void")),
                type_params=cs.get("typeParams", []),
            ))

        ctor_sigs = []
        for cs in spec.get("constructSignatures", []):
            ctor_sigs.append(TSConstructSignature(
                params=[(p["name"], _dict_to_ts_type(p["type"])) for p in cs.get("params", [])],
                return_type=_dict_to_ts_type(cs.get("returnType", "void")),
                type_params=cs.get("typeParams", []),
            ))

        tps = [_dict_to_type_param(tp) for tp in spec.get("typeParams", [])]

        decl = InterfaceDeclaration(
            name=spec.get("name", "UnnamedInterface"),
            properties=props,
            methods=spec.get("methods", []),
            index_signatures=idxs,
            extends=spec.get("extends", []),
            type_params=tps,
            call_signatures=call_sigs,
            construct_signatures=ctor_sigs,
            jsdoc=spec.get("jsdoc"),
            exported=spec.get("exported", True),
        )
        self.declarations.append(decl)
        return decl

    def add_from_refinement(self, name: str, refined_props: Dict[str, Any], *, extends: Optional[List[str]] = None) -> InterfaceDeclaration:
        props: List[TSObjectProperty] = []
        for pname, pinfo in refined_props.items():
            if isinstance(pinfo, dict):
                ptype = _dict_to_ts_type(pinfo.get("type", "any"))
                opt = pinfo.get("optional", False)
                ro = pinfo.get("readonly", False)
            else:
                ptype = _dict_to_ts_type(pinfo)
                opt = False
                ro = False
            props.append(TSObjectProperty(name=pname, type=ptype, optional=opt, readonly=ro))

        decl = InterfaceDeclaration(
            name=name,
            properties=props,
            extends=extends or [],
            exported=True,
        )
        self.declarations.append(decl)
        return decl

    def render_all(self, indent_level: int = 0) -> str:
        return "\n\n".join(d.render(indent_level) for d in self.declarations)


# ---------------------------------------------------------------------------
# 3. TypeAliasGenerator
# ---------------------------------------------------------------------------


@dataclass
class TypeAliasDeclaration:
    name: str
    type: TypeScriptType
    type_params: List[TSTypeParam] = field(default_factory=list)
    jsdoc: Optional[str] = None
    exported: bool = True

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        lines: List[str] = []
        if self.jsdoc:
            for jl in self.jsdoc.splitlines():
                lines.append(f"{pad}{jl}")
        exp = "export " if self.exported else ""
        tp = _type_params_str(self.type_params, indent_level)
        ts = self.type.to_ts_string(indent_level)
        lines.append(f"{pad}{exp}type {self.name}{tp} = {ts};")
        return "\n".join(lines)


@dataclass
class TypeAliasGenerator:
    declarations: List[TypeAliasDeclaration] = field(default_factory=list)

    _UTILITY_TYPES: Dict[str, int] = field(default_factory=lambda: {
        "Partial": 1, "Required": 1, "Readonly": 1, "Pick": 2, "Omit": 2,
        "Record": 2, "Exclude": 2, "Extract": 2, "NonNullable": 1,
        "ReturnType": 1, "Parameters": 1, "ConstructorParameters": 1,
        "InstanceType": 1,
    })

    def add_alias(self, spec: Dict[str, Any]) -> TypeAliasDeclaration:
        tps = [_dict_to_type_param(tp) for tp in spec.get("typeParams", [])]
        decl = TypeAliasDeclaration(
            name=spec.get("name", "UnnamedAlias"),
            type=_dict_to_ts_type(spec.get("type", "any")),
            type_params=tps,
            jsdoc=spec.get("jsdoc"),
            exported=spec.get("exported", True),
        )
        self.declarations.append(decl)
        return decl

    def add_simple_alias(self, name: str, target: Any, *, exported: bool = True) -> TypeAliasDeclaration:
        decl = TypeAliasDeclaration(name=name, type=_dict_to_ts_type(target), exported=exported)
        self.declarations.append(decl)
        return decl

    def add_utility_type(self, name: str, utility: str, args: List[Any], *, exported: bool = True) -> TypeAliasDeclaration:
        expected = self._UTILITY_TYPES.get(utility)
        if expected is not None and len(args) != expected:
            raise ValueError(f"{utility} expects {expected} type arg(s), got {len(args)}")
        ref = TSTypeReference(utility, [_dict_to_ts_type(a) for a in args])
        decl = TypeAliasDeclaration(name=name, type=ref, exported=exported)
        self.declarations.append(decl)
        return decl

    def add_discriminated_union(self, name: str, discriminant: str, variants: List[Dict[str, Any]], *, exported: bool = True) -> TypeAliasDeclaration:
        members: List[TypeScriptType] = []
        for v in variants:
            props: List[TSObjectProperty] = [
                TSObjectProperty(name=discriminant, type=TSLiteral(v.get("tag", "")))
            ]
            for pname, pinfo in v.get("properties", {}).items():
                props.append(TSObjectProperty(name=pname, type=_dict_to_ts_type(pinfo)))
            members.append(TSObject(properties=props))
        decl = TypeAliasDeclaration(name=name, type=TSUnion(members), exported=exported)
        self.declarations.append(decl)
        return decl

    def add_conditional_alias(self, name: str, check: Any, extends: Any, true_branch: Any, false_branch: Any, type_params: Optional[List[Any]] = None, *, exported: bool = True) -> TypeAliasDeclaration:
        cond = TSConditional(
            check_type=_dict_to_ts_type(check),
            extends_type=_dict_to_ts_type(extends),
            true_type=_dict_to_ts_type(true_branch),
            false_type=_dict_to_ts_type(false_branch),
        )
        tps = [_dict_to_type_param(tp) for tp in (type_params or [])]
        decl = TypeAliasDeclaration(name=name, type=cond, type_params=tps, exported=exported)
        self.declarations.append(decl)
        return decl

    def add_template_literal_alias(self, name: str, head: str, spans: List[Tuple[Any, str]], *, exported: bool = True) -> TypeAliasDeclaration:
        tl = TSTemplateLiteral(head=head, spans=[(_dict_to_ts_type(t), tail) for t, tail in spans])
        decl = TypeAliasDeclaration(name=name, type=tl, exported=exported)
        self.declarations.append(decl)
        return decl

    def render_all(self, indent_level: int = 0) -> str:
        return "\n\n".join(d.render(indent_level) for d in self.declarations)


# ---------------------------------------------------------------------------
# 4. FunctionDeclarationGenerator
# ---------------------------------------------------------------------------


@dataclass
class FunctionDeclaration:
    name: str
    params: List[TSFunctionParam] = field(default_factory=list)
    return_type: TypeScriptType = field(default_factory=lambda: TSPrimitive("void"))
    type_params: List[TSTypeParam] = field(default_factory=list)
    overloads: List[Dict[str, Any]] = field(default_factory=list)
    is_async: bool = False
    is_generator: bool = False
    this_type: Optional[TypeScriptType] = None
    jsdoc: Optional[str] = None
    exported: bool = True

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        lines: List[str] = []
        if self.jsdoc:
            for jl in self.jsdoc.splitlines():
                lines.append(f"{pad}{jl}")

        exp = "export " if self.exported else ""
        func_keyword = "function"
        if self.is_generator:
            func_keyword = "function*"

        # Overload signatures
        for ov in self.overloads:
            ov_tps = [_dict_to_type_param(t) for t in ov.get("typeParams", [])]
            ov_params = _build_params(ov.get("params", []), indent_level)
            ov_ret = _dict_to_ts_type(ov.get("returnType", "void")).to_ts_string(indent_level)
            ov_tp_str = _type_params_str(ov_tps, indent_level)
            async_kw = "async " if ov.get("isAsync", self.is_async) else ""
            lines.append(f"{pad}{exp}declare {async_kw}{func_keyword} {self.name}{ov_tp_str}({ov_params}): {ov_ret};")

        tp_str = _type_params_str(self.type_params, indent_level)
        params_parts: List[str] = []
        if self.this_type:
            params_parts.append(f"this: {self.this_type.to_ts_string(indent_level)}")
        for p in self.params:
            params_parts.append(p.to_ts_string(indent_level))
        ps = ", ".join(params_parts)
        ret = self.return_type.to_ts_string(indent_level)
        async_kw = "async " if self.is_async else ""
        lines.append(f"{pad}{exp}declare {async_kw}{func_keyword} {self.name}{tp_str}({ps}): {ret};")
        return "\n".join(lines)


def _build_params(params_spec: List[Dict[str, Any]], indent_level: int) -> str:
    parts: List[str] = []
    for p in params_spec:
        pt = _dict_to_ts_type(p.get("type", "any"))
        opt = "?" if p.get("optional", False) else ""
        rest = "..." if p.get("rest", False) else ""
        parts.append(f"{rest}{p.get('name', 'arg')}{opt}: {pt.to_ts_string(indent_level)}")
    return ", ".join(parts)


@dataclass
class FunctionDeclarationGenerator:
    declarations: List[FunctionDeclaration] = field(default_factory=list)

    def add_function(self, spec: Dict[str, Any]) -> FunctionDeclaration:
        params = []
        for p in spec.get("params", []):
            params.append(TSFunctionParam(
                name=p.get("name", "arg"),
                type=_dict_to_ts_type(p.get("type", "any")),
                optional=p.get("optional", False),
                rest=p.get("rest", False),
                default_value=p.get("default"),
            ))
        tps = [_dict_to_type_param(tp) for tp in spec.get("typeParams", [])]
        this_type = _dict_to_ts_type(spec["thisType"]) if "thisType" in spec else None
        decl = FunctionDeclaration(
            name=spec.get("name", "unnamedFunction"),
            params=params,
            return_type=_dict_to_ts_type(spec.get("returnType", "void")),
            type_params=tps,
            overloads=spec.get("overloads", []),
            is_async=spec.get("isAsync", False),
            is_generator=spec.get("isGenerator", False),
            this_type=this_type,
            jsdoc=spec.get("jsdoc"),
            exported=spec.get("exported", True),
        )
        self.declarations.append(decl)
        return decl

    def add_callback_type(self, name: str, params: List[Dict[str, Any]], return_type: Any, *, exported: bool = True) -> TypeAliasDeclaration:
        fp = []
        for p in params:
            fp.append(TSFunctionParam(
                name=p.get("name", "arg"),
                type=_dict_to_ts_type(p.get("type", "any")),
                optional=p.get("optional", False),
            ))
        fn_type = TSFunction(params=fp, return_type=_dict_to_ts_type(return_type))
        return TypeAliasDeclaration(name=name, type=fn_type, exported=exported)

    def render_all(self, indent_level: int = 0) -> str:
        return "\n\n".join(d.render(indent_level) for d in self.declarations)


# ---------------------------------------------------------------------------
# 5. ClassDeclarationGenerator
# ---------------------------------------------------------------------------


class AccessModifier(enum.Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"


@dataclass
class ClassMember:
    name: str
    type: TypeScriptType
    access: AccessModifier = AccessModifier.PUBLIC
    readonly: bool = False
    static: bool = False
    abstract: bool = False
    optional: bool = False
    is_private_field: bool = False  # #field syntax
    jsdoc: Optional[str] = None

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        parts: List[str] = []
        if self.jsdoc:
            for jl in self.jsdoc.splitlines():
                parts.append(f"{pad}{jl}")
        mods: List[str] = []
        if self.abstract:
            mods.append("abstract")
        if self.static:
            mods.append("static")
        if self.access != AccessModifier.PUBLIC or not self.is_private_field:
            mods.append(self.access.value)
        if self.readonly:
            mods.append("readonly")
        mods_str = " ".join(mods)
        if mods_str:
            mods_str += " "
        field_name = f"#{self.name}" if self.is_private_field else self.name
        opt = "?" if self.optional else ""
        parts.append(f"{pad}{mods_str}{field_name}{opt}: {self.type.to_ts_string(indent_level)};")
        return "\n".join(parts)


@dataclass
class ClassMethodDecl:
    name: str
    params: List[TSFunctionParam] = field(default_factory=list)
    return_type: TypeScriptType = field(default_factory=lambda: TSPrimitive("void"))
    type_params: List[TSTypeParam] = field(default_factory=list)
    access: AccessModifier = AccessModifier.PUBLIC
    static: bool = False
    abstract: bool = False
    is_async: bool = False
    is_generator: bool = False
    overloads: List[Dict[str, Any]] = field(default_factory=list)
    jsdoc: Optional[str] = None

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        lines: List[str] = []
        if self.jsdoc:
            for jl in self.jsdoc.splitlines():
                lines.append(f"{pad}{jl}")

        mods: List[str] = []
        if self.abstract:
            mods.append("abstract")
        if self.static:
            mods.append("static")
        mods.append(self.access.value)
        mods_str = " ".join(mods) + " "

        async_kw = "async " if self.is_async else ""
        gen = "*" if self.is_generator else ""

        for ov in self.overloads:
            ov_tps = [_dict_to_type_param(t) for t in ov.get("typeParams", [])]
            ov_params = _build_params(ov.get("params", []), indent_level)
            ov_ret = _dict_to_ts_type(ov.get("returnType", "void")).to_ts_string(indent_level)
            ov_tp_str = _type_params_str(ov_tps, indent_level)
            lines.append(f"{pad}{mods_str}{async_kw}{self.name}{gen}{ov_tp_str}({ov_params}): {ov_ret};")

        tp_str = _type_params_str(self.type_params, indent_level)
        ps = ", ".join(p.to_ts_string(indent_level) for p in self.params)
        ret = self.return_type.to_ts_string(indent_level)
        lines.append(f"{pad}{mods_str}{async_kw}{self.name}{gen}{tp_str}({ps}): {ret};")
        return "\n".join(lines)


@dataclass
class ClassAccessor:
    name: str
    kind: str  # "get" | "set"
    type: TypeScriptType
    access: AccessModifier = AccessModifier.PUBLIC
    static: bool = False
    abstract: bool = False

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        mods: List[str] = []
        if self.abstract:
            mods.append("abstract")
        if self.static:
            mods.append("static")
        mods.append(self.access.value)
        mods_str = " ".join(mods) + " "
        if self.kind == "get":
            return f"{pad}{mods_str}get {self.name}(): {self.type.to_ts_string(indent_level)};"
        else:
            return f"{pad}{mods_str}set {self.name}(value: {self.type.to_ts_string(indent_level)});"


@dataclass
class ClassConstructorDecl:
    params: List[TSFunctionParam] = field(default_factory=list)
    access: AccessModifier = AccessModifier.PUBLIC
    overloads: List[Dict[str, Any]] = field(default_factory=list)

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        lines: List[str] = []
        for ov in self.overloads:
            ov_params = _build_params(ov.get("params", []), indent_level)
            lines.append(f"{pad}{self.access.value} constructor({ov_params});")
        ps = ", ".join(p.to_ts_string(indent_level) for p in self.params)
        lines.append(f"{pad}{self.access.value} constructor({ps});")
        return "\n".join(lines)


@dataclass
class ClassDeclaration:
    name: str
    members: List[ClassMember] = field(default_factory=list)
    methods: List[ClassMethodDecl] = field(default_factory=list)
    accessors: List[ClassAccessor] = field(default_factory=list)
    constructor: Optional[ClassConstructorDecl] = None
    extends: Optional[str] = None
    implements: List[str] = field(default_factory=list)
    type_params: List[TSTypeParam] = field(default_factory=list)
    abstract: bool = False
    index_signatures: List[TSIndexSignature] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    jsdoc: Optional[str] = None
    exported: bool = True

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        inner_level = indent_level + 1
        inner_pad = _INDENT * inner_level
        lines: List[str] = []

        if self.jsdoc:
            for jl in self.jsdoc.splitlines():
                lines.append(f"{pad}{jl}")

        for dec in self.decorators:
            lines.append(f"{pad}@{dec}")

        exp = "export " if self.exported else ""
        abs_kw = "abstract " if self.abstract else ""
        tp_str = _type_params_str(self.type_params, indent_level)
        ext = f" extends {self.extends}" if self.extends else ""
        impl = ""
        if self.implements:
            impl = " implements " + ", ".join(self.implements)

        lines.append(f"{pad}{exp}declare {abs_kw}class {self.name}{tp_str}{ext}{impl} {{")

        for idx in self.index_signatures:
            lines.append(f"{inner_pad}{idx.to_ts_string(inner_level)};")

        if self.constructor:
            lines.append(self.constructor.render(inner_level))

        for member in self.members:
            lines.append(member.render(inner_level))

        for accessor in self.accessors:
            lines.append(accessor.render(inner_level))

        for method in self.methods:
            lines.append(method.render(inner_level))

        lines.append(f"{pad}}}")
        return "\n".join(lines)


@dataclass
class ClassDeclarationGenerator:
    declarations: List[ClassDeclaration] = field(default_factory=list)

    def add_class(self, spec: Dict[str, Any]) -> ClassDeclaration:
        members: List[ClassMember] = []
        for m in spec.get("members", []):
            members.append(ClassMember(
                name=m.get("name", ""),
                type=_dict_to_ts_type(m.get("type", "any")),
                access=AccessModifier(m.get("access", "public")),
                readonly=m.get("readonly", False),
                static=m.get("static", False),
                abstract=m.get("abstract", False),
                optional=m.get("optional", False),
                is_private_field=m.get("privateField", False),
                jsdoc=m.get("jsdoc"),
            ))

        methods: List[ClassMethodDecl] = []
        for m in spec.get("methods", []):
            params = []
            for p in m.get("params", []):
                params.append(TSFunctionParam(
                    name=p.get("name", "arg"),
                    type=_dict_to_ts_type(p.get("type", "any")),
                    optional=p.get("optional", False),
                    rest=p.get("rest", False),
                ))
            methods.append(ClassMethodDecl(
                name=m.get("name", "method"),
                params=params,
                return_type=_dict_to_ts_type(m.get("returnType", "void")),
                type_params=[_dict_to_type_param(tp) for tp in m.get("typeParams", [])],
                access=AccessModifier(m.get("access", "public")),
                static=m.get("static", False),
                abstract=m.get("abstract", False),
                is_async=m.get("isAsync", False),
                is_generator=m.get("isGenerator", False),
                overloads=m.get("overloads", []),
                jsdoc=m.get("jsdoc"),
            ))

        accessors: List[ClassAccessor] = []
        for a in spec.get("accessors", []):
            accessors.append(ClassAccessor(
                name=a.get("name", ""),
                kind=a.get("kind", "get"),
                type=_dict_to_ts_type(a.get("type", "any")),
                access=AccessModifier(a.get("access", "public")),
                static=a.get("static", False),
                abstract=a.get("abstract", False),
            ))

        ctor = None
        if "constructor" in spec:
            cs = spec["constructor"]
            c_params = []
            for p in cs.get("params", []):
                c_params.append(TSFunctionParam(
                    name=p.get("name", "arg"),
                    type=_dict_to_ts_type(p.get("type", "any")),
                    optional=p.get("optional", False),
                    rest=p.get("rest", False),
                ))
            ctor = ClassConstructorDecl(
                params=c_params,
                access=AccessModifier(cs.get("access", "public")),
                overloads=cs.get("overloads", []),
            )

        idx_sigs = []
        for ix in spec.get("indexSignatures", []):
            idx_sigs.append(TSIndexSignature(
                key_name=ix.get("keyName", "key"),
                key_type=_dict_to_ts_type(ix.get("keyType", "string")),
                value_type=_dict_to_ts_type(ix.get("valueType", "any")),
                readonly=ix.get("readonly", False),
            ))

        tps = [_dict_to_type_param(tp) for tp in spec.get("typeParams", [])]

        decl = ClassDeclaration(
            name=spec.get("name", "UnnamedClass"),
            members=members,
            methods=methods,
            accessors=accessors,
            constructor=ctor,
            extends=spec.get("extends"),
            implements=spec.get("implements", []),
            type_params=tps,
            abstract=spec.get("abstract", False),
            index_signatures=idx_sigs,
            decorators=spec.get("decorators", []),
            jsdoc=spec.get("jsdoc"),
            exported=spec.get("exported", True),
        )
        self.declarations.append(decl)
        return decl

    def render_all(self, indent_level: int = 0) -> str:
        return "\n\n".join(d.render(indent_level) for d in self.declarations)


# ---------------------------------------------------------------------------
# 6. ModuleDeclarationGenerator
# ---------------------------------------------------------------------------


@dataclass
class ExportDeclaration:
    names: List[str] = field(default_factory=list)
    from_module: Optional[str] = None
    is_default: bool = False
    is_type_only: bool = False

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        type_kw = "type " if self.is_type_only else ""
        if self.is_default:
            if self.names:
                return f"{pad}export default {self.names[0]};"
            return f"{pad}export default {{}};"
        if not self.names:
            if self.from_module:
                return f"{pad}export * from '{self.from_module}';"
            return ""
        names_str = ", ".join(self.names)
        if self.from_module:
            return f"{pad}export {type_kw}{{ {names_str} }} from '{self.from_module}';"
        return f"{pad}export {type_kw}{{ {names_str} }};"


@dataclass
class ModuleDeclaration:
    name: str
    body_lines: List[str] = field(default_factory=list)
    is_ambient: bool = False
    is_namespace: bool = False
    is_global: bool = False
    exports: List[ExportDeclaration] = field(default_factory=list)
    jsdoc: Optional[str] = None
    exported: bool = True

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        inner_pad = _INDENT * (indent_level + 1)
        lines: List[str] = []
        if self.jsdoc:
            for jl in self.jsdoc.splitlines():
                lines.append(f"{pad}{jl}")

        if self.is_global:
            lines.append(f"{pad}declare global {{")
        elif self.is_namespace:
            exp = "export " if self.exported else ""
            lines.append(f"{pad}{exp}namespace {self.name} {{")
        elif self.is_ambient:
            lines.append(f"{pad}declare module '{self.name}' {{")
        else:
            exp = "export " if self.exported else ""
            lines.append(f"{pad}{exp}module {self.name} {{")

        for bl in self.body_lines:
            lines.append(f"{inner_pad}{bl}")

        for ex in self.exports:
            lines.append(ex.render(indent_level + 1))

        lines.append(f"{pad}}}")
        return "\n".join(lines)


@dataclass
class ModuleDeclarationGenerator:
    declarations: List[ModuleDeclaration] = field(default_factory=list)

    def add_module(self, spec: Dict[str, Any]) -> ModuleDeclaration:
        exports = []
        for e in spec.get("exports", []):
            exports.append(ExportDeclaration(
                names=e.get("names", []),
                from_module=e.get("from"),
                is_default=e.get("isDefault", False),
                is_type_only=e.get("isTypeOnly", False),
            ))

        decl = ModuleDeclaration(
            name=spec.get("name", "unnamed"),
            body_lines=spec.get("bodyLines", []),
            is_ambient=spec.get("isAmbient", False),
            is_namespace=spec.get("isNamespace", False),
            is_global=spec.get("isGlobal", False),
            exports=exports,
            jsdoc=spec.get("jsdoc"),
            exported=spec.get("exported", True),
        )
        self.declarations.append(decl)
        return decl

    def add_augmentation(self, module_name: str, body_lines: List[str]) -> ModuleDeclaration:
        decl = ModuleDeclaration(
            name=module_name,
            body_lines=body_lines,
            is_ambient=True,
            exported=False,
        )
        self.declarations.append(decl)
        return decl

    def add_global_augmentation(self, body_lines: List[str]) -> ModuleDeclaration:
        decl = ModuleDeclaration(
            name="__global__",
            body_lines=body_lines,
            is_global=True,
            exported=False,
        )
        self.declarations.append(decl)
        return decl

    def add_namespace(self, name: str, body_lines: List[str], *, exported: bool = True) -> ModuleDeclaration:
        decl = ModuleDeclaration(
            name=name,
            body_lines=body_lines,
            is_namespace=True,
            exported=exported,
        )
        self.declarations.append(decl)
        return decl

    def render_all(self, indent_level: int = 0) -> str:
        return "\n\n".join(d.render(indent_level) for d in self.declarations)


# ---------------------------------------------------------------------------
# 7. GenericTypeGenerator
# ---------------------------------------------------------------------------


@dataclass
class GenericTypeGenerator:
    """Handles creation of generic type constructs."""

    def create_type_param(self, name: str, constraint: Optional[Any] = None, default: Optional[Any] = None, variance: Optional[str] = None) -> TSTypeParam:
        c = _dict_to_ts_type(constraint) if constraint is not None else None
        d = _dict_to_ts_type(default) if default is not None else None
        return TSTypeParam(name=name, constraint=c, default=d, variance=variance)

    def create_constrained_identity(self, param_name: str, constraint: Any) -> TypeAliasDeclaration:
        tp = self.create_type_param(param_name, constraint=constraint)
        return TypeAliasDeclaration(
            name=f"Constrained{param_name}",
            type=TSTypeReference(param_name),
            type_params=[tp],
            exported=True,
        )

    def create_conditional_type(self, name: str, check_param: str, extends_type: Any, true_type: Any, false_type: Any, extra_params: Optional[List[TSTypeParam]] = None) -> TypeAliasDeclaration:
        tp = TSTypeParam(name=check_param)
        all_params = [tp] + (extra_params or [])
        cond = TSConditional(
            check_type=TSTypeReference(check_param),
            extends_type=_dict_to_ts_type(extends_type),
            true_type=_dict_to_ts_type(true_type),
            false_type=_dict_to_ts_type(false_type),
        )
        return TypeAliasDeclaration(name=name, type=cond, type_params=all_params, exported=True)

    def create_recursive_type(self, name: str, param_name: str, build_body: Any) -> TypeAliasDeclaration:
        tp = TSTypeParam(name=param_name)
        body = _dict_to_ts_type(build_body)
        return TypeAliasDeclaration(name=name, type=body, type_params=[tp], exported=True)

    def create_higher_kinded_emulation(self, name: str, container_name: str, inner_name: str) -> TypeAliasDeclaration:
        inner_tp = TSTypeParam(name=inner_name)
        container_tp = TSTypeParam(
            name=container_name,
            constraint=TSTypeReference("Record", [TSPrimitive("string"), TSPrimitive("any")]),
        )
        mapped = TSMapped(
            key_name="K",
            key_source=TSKeyof(TSTypeReference(container_name)),
            value_type=TSTypeReference(inner_name),
        )
        return TypeAliasDeclaration(name=name, type=mapped, type_params=[container_tp, inner_tp], exported=True)

    def create_infer_type(self, name: str, param_name: str, wrapper_type: str, infer_name: str, fallback: Any = "never") -> TypeAliasDeclaration:
        tp = TSTypeParam(name=param_name)
        cond = TSConditional(
            check_type=TSTypeReference(param_name),
            extends_type=TSTypeReference(wrapper_type, [TSInfer(infer_name)]),
            true_type=TSTypeReference(infer_name),
            false_type=_dict_to_ts_type(fallback),
        )
        return TypeAliasDeclaration(name=name, type=cond, type_params=[tp], exported=True)

    def render_type_params(self, params: List[TSTypeParam], indent_level: int = 0) -> str:
        return _type_params_str(params, indent_level)


# ---------------------------------------------------------------------------
# 8. EnumGenerator
# ---------------------------------------------------------------------------


@dataclass
class EnumMember:
    name: str
    value: Optional[Union[str, int, float]] = None
    computed_expr: Optional[str] = None
    jsdoc: Optional[str] = None


@dataclass
class EnumDeclaration:
    name: str
    members: List[EnumMember] = field(default_factory=list)
    is_const: bool = False
    jsdoc: Optional[str] = None
    exported: bool = True

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        inner_pad = _INDENT * (indent_level + 1)
        lines: List[str] = []
        if self.jsdoc:
            for jl in self.jsdoc.splitlines():
                lines.append(f"{pad}{jl}")

        exp = "export " if self.exported else ""
        const_kw = "const " if self.is_const else ""
        lines.append(f"{pad}{exp}declare {const_kw}enum {self.name} {{")

        auto_val = 0
        for i, member in enumerate(self.members):
            if member.jsdoc:
                for jl in member.jsdoc.splitlines():
                    lines.append(f"{inner_pad}{jl}")
            sep = "," if i < len(self.members) - 1 else ","
            if member.computed_expr is not None:
                lines.append(f"{inner_pad}{member.name} = {member.computed_expr}{sep}")
                auto_val = 0  # reset after computed
            elif member.value is not None:
                if isinstance(member.value, str):
                    escaped = member.value.replace("'", "\\'")
                    lines.append(f"{inner_pad}{member.name} = '{escaped}'{sep}")
                else:
                    lines.append(f"{inner_pad}{member.name} = {member.value}{sep}")
                    if isinstance(member.value, int):
                        auto_val = member.value + 1
            else:
                lines.append(f"{inner_pad}{member.name} = {auto_val}{sep}")
                auto_val += 1

        lines.append(f"{pad}}}")
        return "\n".join(lines)

    def reverse_mapping(self) -> Dict[Union[str, int], str]:
        mapping: Dict[Union[str, int], str] = {}
        auto_val = 0
        for member in self.members:
            if member.value is not None:
                mapping[member.value] = member.name
                if isinstance(member.value, int):
                    auto_val = member.value + 1
            elif member.computed_expr is None:
                mapping[auto_val] = member.name
                auto_val += 1
        return mapping


@dataclass
class EnumGenerator:
    declarations: List[EnumDeclaration] = field(default_factory=list)

    def add_enum(self, spec: Dict[str, Any]) -> EnumDeclaration:
        members = []
        for m in spec.get("members", []):
            members.append(EnumMember(
                name=m.get("name", ""),
                value=m.get("value"),
                computed_expr=m.get("computedExpr"),
                jsdoc=m.get("jsdoc"),
            ))
        decl = EnumDeclaration(
            name=spec.get("name", "UnnamedEnum"),
            members=members,
            is_const=spec.get("isConst", False),
            jsdoc=spec.get("jsdoc"),
            exported=spec.get("exported", True),
        )
        self.declarations.append(decl)
        return decl

    def add_string_enum(self, name: str, values: Dict[str, str], *, exported: bool = True, is_const: bool = False) -> EnumDeclaration:
        members = [EnumMember(name=k, value=v) for k, v in values.items()]
        decl = EnumDeclaration(name=name, members=members, is_const=is_const, exported=exported)
        self.declarations.append(decl)
        return decl

    def add_numeric_enum(self, name: str, values: List[str], start: int = 0, *, exported: bool = True, is_const: bool = False) -> EnumDeclaration:
        members = [EnumMember(name=v, value=start + i) for i, v in enumerate(values)]
        decl = EnumDeclaration(name=name, members=members, is_const=is_const, exported=exported)
        self.declarations.append(decl)
        return decl

    def merge_enums(self, a: EnumDeclaration, b: EnumDeclaration) -> EnumDeclaration:
        if a.name != b.name:
            raise ValueError(f"Cannot merge enums with different names: {a.name} vs {b.name}")
        existing_names = {m.name for m in a.members}
        merged_members = list(a.members)
        for m in b.members:
            if m.name not in existing_names:
                merged_members.append(m)
                existing_names.add(m.name)
        return EnumDeclaration(
            name=a.name,
            members=merged_members,
            is_const=a.is_const and b.is_const,
            exported=a.exported or b.exported,
        )

    def render_all(self, indent_level: int = 0) -> str:
        return "\n\n".join(d.render(indent_level) for d in self.declarations)


# ---------------------------------------------------------------------------
# 9. ConditionalTypeGenerator
# ---------------------------------------------------------------------------


@dataclass
class ConditionalTypeGenerator:
    """Generates conditional type expressions and aliases."""

    def basic(self, name: str, check: Any, extends: Any, true_branch: Any, false_branch: Any, type_params: Optional[List[Any]] = None) -> TypeAliasDeclaration:
        cond = TSConditional(
            check_type=_dict_to_ts_type(check),
            extends_type=_dict_to_ts_type(extends),
            true_type=_dict_to_ts_type(true_branch),
            false_type=_dict_to_ts_type(false_branch),
        )
        tps = [_dict_to_type_param(tp) for tp in (type_params or [])]
        return TypeAliasDeclaration(name=name, type=cond, type_params=tps, exported=True)

    def nested(self, name: str, conditions: List[Dict[str, Any]], fallback: Any = "never", type_params: Optional[List[Any]] = None) -> TypeAliasDeclaration:
        result = _dict_to_ts_type(fallback)
        for cond in reversed(conditions):
            result = TSConditional(
                check_type=_dict_to_ts_type(cond["check"]),
                extends_type=_dict_to_ts_type(cond["extends"]),
                true_type=_dict_to_ts_type(cond["true"]),
                false_type=result,
            )
        tps = [_dict_to_type_param(tp) for tp in (type_params or [])]
        return TypeAliasDeclaration(name=name, type=result, type_params=tps, exported=True)

    def distributive(self, name: str, param_name: str, extends_type: Any, true_type: Any, false_type: Any) -> TypeAliasDeclaration:
        tp = TSTypeParam(name=param_name)
        cond = TSConditional(
            check_type=TSTypeReference(param_name),
            extends_type=_dict_to_ts_type(extends_type),
            true_type=_dict_to_ts_type(true_type),
            false_type=_dict_to_ts_type(false_type),
        )
        return TypeAliasDeclaration(name=name, type=cond, type_params=[tp], exported=True)

    def with_infer(self, name: str, param_name: str, pattern_type: str, pattern_args: List[Any], infer_positions: List[int], true_type: Any, false_type: Any = "never") -> TypeAliasDeclaration:
        tp = TSTypeParam(name=param_name)
        pat_args: List[TypeScriptType] = []
        infer_counter = 0
        for i, a in enumerate(pattern_args):
            if i in infer_positions:
                pat_args.append(TSInfer(f"_Infer{infer_counter}"))
                infer_counter += 1
            else:
                pat_args.append(_dict_to_ts_type(a))
        ext = TSTypeReference(pattern_type, pat_args)
        cond = TSConditional(
            check_type=TSTypeReference(param_name),
            extends_type=ext,
            true_type=_dict_to_ts_type(true_type),
            false_type=_dict_to_ts_type(false_type),
        )
        return TypeAliasDeclaration(name=name, type=cond, type_params=[tp], exported=True)

    def narrowing(self, name: str, param_name: str, narrow_checks: List[Tuple[Any, Any]]) -> TypeAliasDeclaration:
        tp = TSTypeParam(name=param_name)
        result: TypeScriptType = TSPrimitive("never")
        for ext, narrow_result in reversed(narrow_checks):
            result = TSConditional(
                check_type=TSTypeReference(param_name),
                extends_type=_dict_to_ts_type(ext),
                true_type=_dict_to_ts_type(narrow_result),
                false_type=result,
            )
        return TypeAliasDeclaration(name=name, type=result, type_params=[tp], exported=True)


# ---------------------------------------------------------------------------
# 10. MappedTypeGenerator
# ---------------------------------------------------------------------------


@dataclass
class MappedTypeGenerator:
    """Generates mapped type expressions and aliases."""

    def basic(self, name: str, source_type: Any, value_transform: Any, key_name: str = "K", type_params: Optional[List[Any]] = None) -> TypeAliasDeclaration:
        mapped = TSMapped(
            key_name=key_name,
            key_source=_dict_to_ts_type(source_type),
            value_type=_dict_to_ts_type(value_transform),
        )
        tps = [_dict_to_type_param(tp) for tp in (type_params or [])]
        return TypeAliasDeclaration(name=name, type=mapped, type_params=tps, exported=True)

    def with_modifiers(self, name: str, source_type: Any, value_type: Any, key_name: str = "K", readonly_mod: Optional[str] = None, optional_mod: Optional[str] = None, type_params: Optional[List[Any]] = None) -> TypeAliasDeclaration:
        mapped = TSMapped(
            key_name=key_name,
            key_source=_dict_to_ts_type(source_type),
            value_type=_dict_to_ts_type(value_type),
            readonly_mod=readonly_mod,
            optional_mod=optional_mod,
        )
        tps = [_dict_to_type_param(tp) for tp in (type_params or [])]
        return TypeAliasDeclaration(name=name, type=mapped, type_params=tps, exported=True)

    def with_key_remapping(self, name: str, source_type: Any, value_type: Any, as_clause: Any, key_name: str = "K", type_params: Optional[List[Any]] = None) -> TypeAliasDeclaration:
        mapped = TSMapped(
            key_name=key_name,
            key_source=_dict_to_ts_type(source_type),
            value_type=_dict_to_ts_type(value_type),
            as_clause=_dict_to_ts_type(as_clause),
        )
        tps = [_dict_to_type_param(tp) for tp in (type_params or [])]
        return TypeAliasDeclaration(name=name, type=mapped, type_params=tps, exported=True)

    def partial_type(self, param_name: str = "T") -> TypeAliasDeclaration:
        tp = TSTypeParam(name=param_name)
        mapped = TSMapped(
            key_name="K",
            key_source=TSKeyof(TSTypeReference(param_name)),
            value_type=TSIndexedAccess(TSTypeReference(param_name), TSTypeReference("K")),
            optional_mod="+",
        )
        return TypeAliasDeclaration(name="MyPartial", type=mapped, type_params=[tp], exported=True)

    def required_type(self, param_name: str = "T") -> TypeAliasDeclaration:
        tp = TSTypeParam(name=param_name)
        mapped = TSMapped(
            key_name="K",
            key_source=TSKeyof(TSTypeReference(param_name)),
            value_type=TSIndexedAccess(TSTypeReference(param_name), TSTypeReference("K")),
            optional_mod="-",
        )
        return TypeAliasDeclaration(name="MyRequired", type=mapped, type_params=[tp], exported=True)

    def readonly_type(self, param_name: str = "T") -> TypeAliasDeclaration:
        tp = TSTypeParam(name=param_name)
        mapped = TSMapped(
            key_name="K",
            key_source=TSKeyof(TSTypeReference(param_name)),
            value_type=TSIndexedAccess(TSTypeReference(param_name), TSTypeReference("K")),
            readonly_mod="+",
        )
        return TypeAliasDeclaration(name="MyReadonly", type=mapped, type_params=[tp], exported=True)

    def template_literal_keys(self, name: str, prefix: str, source_type: Any, value_type: Any, type_params: Optional[List[Any]] = None) -> TypeAliasDeclaration:
        tl = TSTemplateLiteral(head=prefix, spans=[(_dict_to_ts_type(source_type), "")])
        mapped = TSMapped(
            key_name="K",
            key_source=_dict_to_ts_type(source_type),
            value_type=_dict_to_ts_type(value_type),
            as_clause=tl,
        )
        tps = [_dict_to_type_param(tp) for tp in (type_params or [])]
        return TypeAliasDeclaration(name=name, type=mapped, type_params=tps, exported=True)


# ---------------------------------------------------------------------------
# 11. TypeGuardGenerator
# ---------------------------------------------------------------------------


@dataclass
class TypeGuardDeclaration:
    name: str
    param_name: str
    param_type: TypeScriptType
    guarded_type: TypeScriptType
    is_assertion: bool = False
    jsdoc: Optional[str] = None
    exported: bool = True

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        lines: List[str] = []
        if self.jsdoc:
            for jl in self.jsdoc.splitlines():
                lines.append(f"{pad}{jl}")
        exp = "export " if self.exported else ""
        guarded = self.guarded_type.to_ts_string(indent_level)
        if self.is_assertion:
            ret = f"asserts {self.param_name} is {guarded}"
        else:
            ret = f"{self.param_name} is {guarded}"
        pt = self.param_type.to_ts_string(indent_level)
        lines.append(f"{pad}{exp}declare function {self.name}({self.param_name}: {pt}): {ret};")
        return "\n".join(lines)


@dataclass
class TypeGuardGenerator:
    declarations: List[TypeGuardDeclaration] = field(default_factory=list)

    def add_type_guard(self, spec: Dict[str, Any]) -> TypeGuardDeclaration:
        decl = TypeGuardDeclaration(
            name=spec.get("name", "isType"),
            param_name=spec.get("paramName", "value"),
            param_type=_dict_to_ts_type(spec.get("paramType", "unknown")),
            guarded_type=_dict_to_ts_type(spec.get("guardedType", "any")),
            is_assertion=spec.get("isAssertion", False),
            jsdoc=spec.get("jsdoc"),
            exported=spec.get("exported", True),
        )
        self.declarations.append(decl)
        return decl

    def add_predicate(self, name: str, param_name: str, param_type: Any, guarded_type: Any, *, exported: bool = True) -> TypeGuardDeclaration:
        decl = TypeGuardDeclaration(
            name=name,
            param_name=param_name,
            param_type=_dict_to_ts_type(param_type),
            guarded_type=_dict_to_ts_type(guarded_type),
            is_assertion=False,
            exported=exported,
        )
        self.declarations.append(decl)
        return decl

    def add_assertion(self, name: str, param_name: str, param_type: Any, guarded_type: Any, *, exported: bool = True) -> TypeGuardDeclaration:
        decl = TypeGuardDeclaration(
            name=name,
            param_name=param_name,
            param_type=_dict_to_ts_type(param_type),
            guarded_type=_dict_to_ts_type(guarded_type),
            is_assertion=True,
            exported=exported,
        )
        self.declarations.append(decl)
        return decl

    def from_refinement(self, name: str, param_name: str, checks: List[Dict[str, Any]]) -> TypeGuardDeclaration:
        """Create a type guard from a list of refinement property checks."""
        props: List[TSObjectProperty] = []
        for chk in checks:
            props.append(TSObjectProperty(
                name=chk.get("property", ""),
                type=_dict_to_ts_type(chk.get("type", "any")),
                optional=chk.get("optional", False),
            ))
        guarded = TSObject(properties=props)
        decl = TypeGuardDeclaration(
            name=name,
            param_name=param_name,
            param_type=TSPrimitive("unknown"),
            guarded_type=guarded,
            is_assertion=False,
            exported=True,
        )
        self.declarations.append(decl)
        return decl

    def narrowing_helpers(self, base_type: str, variants: Dict[str, Any]) -> List[TypeGuardDeclaration]:
        decls: List[TypeGuardDeclaration] = []
        for vname, vtype in variants.items():
            fn_name = f"is{vname[0].upper()}{vname[1:]}"
            d = TypeGuardDeclaration(
                name=fn_name,
                param_name="value",
                param_type=_dict_to_ts_type(base_type),
                guarded_type=_dict_to_ts_type(vtype),
                exported=True,
            )
            self.declarations.append(d)
            decls.append(d)
        return decls

    def render_all(self, indent_level: int = 0) -> str:
        return "\n\n".join(d.render(indent_level) for d in self.declarations)


# ---------------------------------------------------------------------------
# 12. AssertionFunctionGenerator (note: name per spec)
# ---------------------------------------------------------------------------


@dataclass
class AssertionDeclaration:
    name: str
    param_name: str
    param_type: TypeScriptType
    asserted_type: Optional[TypeScriptType] = None
    condition_only: bool = False
    overloads: List[Dict[str, Any]] = field(default_factory=list)
    jsdoc: Optional[str] = None
    exported: bool = True

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        lines: List[str] = []
        if self.jsdoc:
            for jl in self.jsdoc.splitlines():
                lines.append(f"{pad}{jl}")
        exp = "export " if self.exported else ""

        for ov in self.overloads:
            ov_pname = ov.get("paramName", self.param_name)
            ov_ptype = _dict_to_ts_type(ov.get("paramType", "unknown")).to_ts_string(indent_level)
            if ov.get("assertedType"):
                at = _dict_to_ts_type(ov["assertedType"]).to_ts_string(indent_level)
                ret = f"asserts {ov_pname} is {at}"
            else:
                ret = f"asserts {ov_pname}"
            lines.append(f"{pad}{exp}declare function {self.name}({ov_pname}: {ov_ptype}): {ret};")

        pt = self.param_type.to_ts_string(indent_level)
        if self.condition_only:
            ret_str = f"asserts {self.param_name}"
        elif self.asserted_type:
            at_str = self.asserted_type.to_ts_string(indent_level)
            ret_str = f"asserts {self.param_name} is {at_str}"
        else:
            ret_str = f"asserts {self.param_name}"
        lines.append(f"{pad}{exp}declare function {self.name}({self.param_name}: {pt}): {ret_str};")
        return "\n".join(lines)


@dataclass
class AssertionFunctionGenerator:
    declarations: List[AssertionDeclaration] = field(default_factory=list)

    def add_assertion(self, spec: Dict[str, Any]) -> AssertionDeclaration:
        asserted = _dict_to_ts_type(spec["assertedType"]) if "assertedType" in spec else None
        decl = AssertionDeclaration(
            name=spec.get("name", "assertType"),
            param_name=spec.get("paramName", "value"),
            param_type=_dict_to_ts_type(spec.get("paramType", "unknown")),
            asserted_type=asserted,
            condition_only=spec.get("conditionOnly", False),
            overloads=spec.get("overloads", []),
            jsdoc=spec.get("jsdoc"),
            exported=spec.get("exported", True),
        )
        self.declarations.append(decl)
        return decl

    def add_condition_assert(self, name: str, param_name: str, param_type: Any = "unknown", *, exported: bool = True) -> AssertionDeclaration:
        decl = AssertionDeclaration(
            name=name,
            param_name=param_name,
            param_type=_dict_to_ts_type(param_type),
            condition_only=True,
            exported=exported,
        )
        self.declarations.append(decl)
        return decl

    def add_type_assert(self, name: str, param_name: str, param_type: Any, asserted_type: Any, *, exported: bool = True) -> AssertionDeclaration:
        decl = AssertionDeclaration(
            name=name,
            param_name=param_name,
            param_type=_dict_to_ts_type(param_type),
            asserted_type=_dict_to_ts_type(asserted_type),
            exported=exported,
        )
        self.declarations.append(decl)
        return decl

    def render_all(self, indent_level: int = 0) -> str:
        return "\n\n".join(d.render(indent_level) for d in self.declarations)


# ---------------------------------------------------------------------------
# 13. DeclarationFileWriter
# ---------------------------------------------------------------------------


@dataclass
class FileSection:
    kind: str  # "header" | "imports" | "types" | "interfaces" | "classes" | "functions" | "enums" | "modules" | "guards" | "exports"
    content: str
    order: int = 0


@dataclass
class DeclarationFileWriter:
    output_dir: Path = field(default_factory=lambda: Path("."))
    line_width: int = 100
    indent_size: int = 4
    newline: str = "\n"
    encoding: str = "utf-8"
    header_comment: Optional[str] = None

    def _format_content(self, sections: List[FileSection]) -> str:
        sorted_sections = sorted(sections, key=lambda s: s.order)
        parts: List[str] = []
        if self.header_comment:
            parts.append(self.header_comment)
            parts.append("")

        for section in sorted_sections:
            content = section.content.strip()
            if content:
                parts.append(content)
                parts.append("")

        result = self.newline.join(parts)
        if not result.endswith(self.newline):
            result += self.newline
        return result

    def _enforce_line_width(self, content: str) -> str:
        out_lines: List[str] = []
        for line in content.splitlines():
            if len(line) <= self.line_width:
                out_lines.append(line)
            else:
                wrapped = _wrap_line(line, self.line_width)
                out_lines.extend(wrapped)
        return self.newline.join(out_lines)

    def write_file(self, filename: str, sections: List[FileSection], *, atomic: bool = True) -> Path:
        content = self._format_content(sections)
        content = self._enforce_line_width(content)
        target = self.output_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)

        if atomic:
            tmp_path = target.with_suffix(target.suffix + ".tmp")
            try:
                tmp_path.write_text(content, encoding=self.encoding)
                tmp_path.replace(target)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise
        else:
            target.write_text(content, encoding=self.encoding)
        return target

    def write_string(self, sections: List[FileSection]) -> str:
        content = self._format_content(sections)
        return self._enforce_line_width(content)

    def build_header_section(self, module_name: str, version: Optional[str] = None) -> FileSection:
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines = [
            "/**",
            f" * Declaration file for module '{module_name}'",
        ]
        if version:
            lines.append(f" * TypeScript target: {version}")
        lines.append(f" * Auto-generated on {now}")
        lines.append(" * DO NOT EDIT MANUALLY")
        lines.append(" */")
        return FileSection(kind="header", content="\n".join(lines), order=0)

    def build_section(self, kind: str, content: str, order: int) -> FileSection:
        return FileSection(kind=kind, content=content, order=order)


# ---------------------------------------------------------------------------
# 14. DeclarationMerger
# ---------------------------------------------------------------------------


@dataclass
class MergeConflict:
    kind: str  # "property_type_mismatch" | "method_signature_mismatch" | "duplicate_member"
    entity_name: str
    member_name: str
    details: str


@dataclass
class DeclarationMerger:
    conflicts: List[MergeConflict] = field(default_factory=list)
    _manual_declarations: Dict[str, str] = field(default_factory=dict)

    def preserve_manual(self, name: str, content: str) -> None:
        self._manual_declarations[name] = content

    def merge_interfaces(self, interfaces: List[InterfaceDeclaration]) -> List[InterfaceDeclaration]:
        by_name: Dict[str, List[InterfaceDeclaration]] = collections.OrderedDict()
        for iface in interfaces:
            by_name.setdefault(iface.name, []).append(iface)

        merged: List[InterfaceDeclaration] = []
        for name, group in by_name.items():
            if len(group) == 1:
                merged.append(group[0])
                continue
            base = copy.deepcopy(group[0])
            seen_props: Dict[str, TSObjectProperty] = {p.name: p for p in base.properties}
            seen_methods: Set[str] = set()
            for meth in base.methods:
                seen_methods.add(meth.get("name", ""))

            for other in group[1:]:
                for prop in other.properties:
                    if prop.name in seen_props:
                        existing = seen_props[prop.name]
                        if existing.type.to_ts_string() != prop.type.to_ts_string():
                            self.conflicts.append(MergeConflict(
                                kind="property_type_mismatch",
                                entity_name=name,
                                member_name=prop.name,
                                details=f"{existing.type.to_ts_string()} vs {prop.type.to_ts_string()}",
                            ))
                    else:
                        base.properties.append(prop)
                        seen_props[prop.name] = prop

                for meth in other.methods:
                    mname = meth.get("name", "")
                    if mname not in seen_methods:
                        base.methods.append(meth)
                        seen_methods.add(mname)

                for idx in other.index_signatures:
                    base.index_signatures.append(idx)

                for ext in other.extends:
                    if ext not in base.extends:
                        base.extends.append(ext)

                for cs in other.call_signatures:
                    base.call_signatures.append(cs)
                for cs in other.construct_signatures:
                    base.construct_signatures.append(cs)

            merged.append(base)
        return merged

    def merge_namespaces(self, namespaces: List[ModuleDeclaration]) -> List[ModuleDeclaration]:
        by_name: Dict[str, List[ModuleDeclaration]] = collections.OrderedDict()
        for ns in namespaces:
            by_name.setdefault(ns.name, []).append(ns)
        merged: List[ModuleDeclaration] = []
        for name, group in by_name.items():
            if len(group) == 1:
                merged.append(group[0])
                continue
            base = copy.deepcopy(group[0])
            seen_lines: Set[str] = set(base.body_lines)
            for other in group[1:]:
                for line in other.body_lines:
                    if line not in seen_lines:
                        base.body_lines.append(line)
                        seen_lines.add(line)
                for ex in other.exports:
                    base.exports.append(ex)
            merged.append(base)
        return merged

    def merge_module_augmentations(self, modules: List[ModuleDeclaration]) -> List[ModuleDeclaration]:
        ambient: Dict[str, List[ModuleDeclaration]] = collections.OrderedDict()
        non_ambient: List[ModuleDeclaration] = []
        for m in modules:
            if m.is_ambient:
                ambient.setdefault(m.name, []).append(m)
            else:
                non_ambient.append(m)
        merged_ambient: List[ModuleDeclaration] = []
        for name, group in ambient.items():
            if len(group) == 1:
                merged_ambient.append(group[0])
            else:
                base = copy.deepcopy(group[0])
                for other in group[1:]:
                    base.body_lines.extend(other.body_lines)
                    base.exports.extend(other.exports)
                merged_ambient.append(base)
        return non_ambient + merged_ambient

    def detect_conflicts(self) -> List[MergeConflict]:
        return list(self.conflicts)

    def get_preserved(self) -> Dict[str, str]:
        return dict(self._manual_declarations)


# ---------------------------------------------------------------------------
# 15. JSDocCommentGenerator
# ---------------------------------------------------------------------------


@dataclass
class JSDocTag:
    tag: str
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None


@dataclass
class JSDocComment:
    description: Optional[str] = None
    tags: List[JSDocTag] = field(default_factory=list)

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        lines = [f"{pad}/**"]
        if self.description:
            for dl in self.description.splitlines():
                lines.append(f"{pad} * {dl}")
        if self.description and self.tags:
            lines.append(f"{pad} *")
        for tag in self.tags:
            parts = [f"@{tag.tag}"]
            if tag.type:
                parts.append(f"{{{tag.type}}}")
            if tag.name:
                parts.append(tag.name)
            if tag.description:
                parts.append(f"- {tag.description}")
            tag_line = " ".join(parts)
            lines.append(f"{pad} * {tag_line}")
        lines.append(f"{pad} */")
        return "\n".join(lines)


@dataclass
class JSDocCommentGenerator:

    def create_comment(self, spec: Dict[str, Any]) -> JSDocComment:
        tags: List[JSDocTag] = []
        for t in spec.get("tags", []):
            tags.append(JSDocTag(
                tag=t.get("tag", ""),
                name=t.get("name"),
                type=t.get("type"),
                description=t.get("description"),
            ))
        return JSDocComment(description=spec.get("description"), tags=tags)

    def for_function(self, description: Optional[str], params: List[Dict[str, Any]], return_type: Optional[str] = None, return_desc: Optional[str] = None, type_params: Optional[List[str]] = None, throws: Optional[List[str]] = None, examples: Optional[List[str]] = None, deprecated: Optional[str] = None, since: Optional[str] = None, see: Optional[List[str]] = None) -> JSDocComment:
        tags: List[JSDocTag] = []
        if deprecated is not None:
            tags.append(JSDocTag(tag="deprecated", description=deprecated or None))
        if since:
            tags.append(JSDocTag(tag="since", description=since))
        for tp in (type_params or []):
            tags.append(JSDocTag(tag="template", name=tp))
        for p in params:
            tags.append(JSDocTag(
                tag="param",
                name=p.get("name"),
                type=p.get("type"),
                description=p.get("description"),
            ))
        if return_type or return_desc:
            tags.append(JSDocTag(tag="returns", type=return_type, description=return_desc))
        for th in (throws or []):
            tags.append(JSDocTag(tag="throws", description=th))
        for ex in (examples or []):
            tags.append(JSDocTag(tag="example", description=ex))
        for s in (see or []):
            tags.append(JSDocTag(tag="see", description=s))
        return JSDocComment(description=description, tags=tags)

    def for_interface(self, description: Optional[str], type_params: Optional[List[str]] = None, see: Optional[List[str]] = None) -> JSDocComment:
        tags: List[JSDocTag] = []
        for tp in (type_params or []):
            tags.append(JSDocTag(tag="template", name=tp))
        for s in (see or []):
            tags.append(JSDocTag(tag="see", description=s))
        return JSDocComment(description=description, tags=tags)

    def for_property(self, description: Optional[str], type_str: Optional[str] = None, deprecated: Optional[str] = None) -> JSDocComment:
        tags: List[JSDocTag] = []
        if type_str:
            tags.append(JSDocTag(tag="type", type=type_str))
        if deprecated is not None:
            tags.append(JSDocTag(tag="deprecated", description=deprecated or None))
        return JSDocComment(description=description, tags=tags)

    def for_refinement(self, description: str, predicates: List[str]) -> JSDocComment:
        tags: List[JSDocTag] = []
        for pred in predicates:
            tags.append(JSDocTag(tag="refinement", description=pred))
        return JSDocComment(description=description, tags=tags)

    def render(self, comment: JSDocComment, indent_level: int = 0) -> str:
        return comment.render(indent_level)


# ---------------------------------------------------------------------------
# 16. ImportExportResolver
# ---------------------------------------------------------------------------


@dataclass
class ImportDeclaration:
    module_path: str
    named: List[str] = field(default_factory=list)
    default_name: Optional[str] = None
    namespace_name: Optional[str] = None
    is_type_only: bool = False

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        type_kw = "type " if self.is_type_only else ""
        parts: List[str] = []
        if self.default_name and self.named:
            names = ", ".join(sorted(self.named))
            parts.append(f"{pad}import {type_kw}{self.default_name}, {{ {names} }} from '{self.module_path}';")
        elif self.default_name:
            parts.append(f"{pad}import {type_kw}{self.default_name} from '{self.module_path}';")
        elif self.namespace_name:
            parts.append(f"{pad}import {type_kw}* as {self.namespace_name} from '{self.module_path}';")
        elif self.named:
            names = ", ".join(sorted(self.named))
            parts.append(f"{pad}import {type_kw}{{ {names} }} from '{self.module_path}';")
        else:
            parts.append(f"{pad}import '{self.module_path}';")
        return "\n".join(parts)


@dataclass
class ReExportDeclaration:
    module_path: str
    names: List[str] = field(default_factory=list)
    is_type_only: bool = False
    export_all: bool = False

    def render(self, indent_level: int = 0) -> str:
        pad = _INDENT * indent_level
        type_kw = "type " if self.is_type_only else ""
        if self.export_all:
            return f"{pad}export * from '{self.module_path}';"
        if self.names:
            names = ", ".join(sorted(self.names))
            return f"{pad}export {type_kw}{{ {names} }} from '{self.module_path}';"
        return f"{pad}export * from '{self.module_path}';"


@dataclass
class DynamicImportType:
    module_path: str
    accessed_name: Optional[str] = None

    def to_ts_type(self) -> TypeScriptType:
        base = TSTypeReference(f"import('{self.module_path}')")
        if self.accessed_name:
            return TSIndexedAccess(base, TSLiteral(self.accessed_name))
        return base


@dataclass
class ImportExportResolver:
    imports: List[ImportDeclaration] = field(default_factory=list)
    re_exports: List[ReExportDeclaration] = field(default_factory=list)
    path_aliases: Dict[str, str] = field(default_factory=dict)
    _type_origins: Dict[str, str] = field(default_factory=dict)

    def register_type_origin(self, type_name: str, module_path: str) -> None:
        self._type_origins[type_name] = module_path

    def resolve_path(self, raw_path: str) -> str:
        for alias, replacement in self.path_aliases.items():
            if raw_path.startswith(alias):
                return raw_path.replace(alias, replacement, 1)
        return raw_path

    def add_import(self, module_path: str, named: Optional[List[str]] = None, default_name: Optional[str] = None, namespace_name: Optional[str] = None, *, type_only: bool = False) -> ImportDeclaration:
        resolved = self.resolve_path(module_path)
        existing = self._find_import(resolved)
        if existing is not None:
            if named:
                existing_set = set(existing.named)
                for n in named:
                    if n not in existing_set:
                        existing.named.append(n)
            if default_name and not existing.default_name:
                existing.default_name = default_name
            if namespace_name and not existing.namespace_name:
                existing.namespace_name = namespace_name
            return existing

        decl = ImportDeclaration(
            module_path=resolved,
            named=named or [],
            default_name=default_name,
            namespace_name=namespace_name,
            is_type_only=type_only,
        )
        self.imports.append(decl)
        return decl

    def add_re_export(self, module_path: str, names: Optional[List[str]] = None, *, type_only: bool = False, export_all: bool = False) -> ReExportDeclaration:
        resolved = self.resolve_path(module_path)
        decl = ReExportDeclaration(
            module_path=resolved,
            names=names or [],
            is_type_only=type_only,
            export_all=export_all,
        )
        self.re_exports.append(decl)
        return decl

    def add_dynamic_import(self, module_path: str, accessed_name: Optional[str] = None) -> DynamicImportType:
        return DynamicImportType(module_path=self.resolve_path(module_path), accessed_name=accessed_name)

    def auto_resolve_imports(self, used_types: Set[str]) -> None:
        for type_name in used_types:
            if type_name in self._type_origins:
                mod = self._type_origins[type_name]
                self.add_import(mod, named=[type_name], type_only=True)

    def _find_import(self, module_path: str) -> Optional[ImportDeclaration]:
        for imp in self.imports:
            if imp.module_path == module_path:
                return imp
        return None

    def group_imports(self) -> List[List[ImportDeclaration]]:
        node_modules: List[ImportDeclaration] = []
        relative: List[ImportDeclaration] = []
        aliased: List[ImportDeclaration] = []
        for imp in self.imports:
            if imp.module_path.startswith("."):
                relative.append(imp)
            elif imp.module_path.startswith("@") or "/" not in imp.module_path:
                node_modules.append(imp)
            else:
                aliased.append(imp)
        groups: List[List[ImportDeclaration]] = []
        if node_modules:
            groups.append(sorted(node_modules, key=lambda i: i.module_path))
        if aliased:
            groups.append(sorted(aliased, key=lambda i: i.module_path))
        if relative:
            groups.append(sorted(relative, key=lambda i: i.module_path))
        return groups

    def render_imports(self, indent_level: int = 0) -> str:
        groups = self.group_imports()
        blocks: List[str] = []
        for group in groups:
            lines = [imp.render(indent_level) for imp in group]
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def render_re_exports(self, indent_level: int = 0) -> str:
        return "\n".join(re.render(indent_level) for re in self.re_exports)


# ---------------------------------------------------------------------------
# 17. SourceMapGenerator
# ---------------------------------------------------------------------------


def _vlq_encode(value: int) -> str:
    """Encode an integer using VLQ (Variable-Length Quantity) for source maps."""
    VLQ_BASE_SHIFT = 5
    VLQ_BASE = 1 << VLQ_BASE_SHIFT  # 32
    VLQ_BASE_MASK = VLQ_BASE - 1  # 31
    VLQ_CONTINUATION_BIT = VLQ_BASE  # 32

    B64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

    if value < 0:
        vlq = ((-value) << 1) + 1
    else:
        vlq = value << 1

    encoded = ""
    while True:
        digit = vlq & VLQ_BASE_MASK
        vlq >>= VLQ_BASE_SHIFT
        if vlq > 0:
            digit |= VLQ_CONTINUATION_BIT
        encoded += B64_CHARS[digit]
        if vlq == 0:
            break
    return encoded


@dataclass
class SourceMapping:
    generated_line: int
    generated_column: int
    source_index: int
    original_line: int
    original_column: int
    name_index: Optional[int] = None


@dataclass
class SourceMapGenerator:
    source_root: str = ""
    sources: List[str] = field(default_factory=list)
    sources_content: List[Optional[str]] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    mappings: List[SourceMapping] = field(default_factory=list)
    file: str = ""

    _source_index_map: Dict[str, int] = field(default_factory=dict)
    _name_index_map: Dict[str, int] = field(default_factory=dict)

    def add_source(self, path: str, content: Optional[str] = None) -> int:
        if path in self._source_index_map:
            return self._source_index_map[path]
        idx = len(self.sources)
        self.sources.append(path)
        self.sources_content.append(content)
        self._source_index_map[path] = idx
        return idx

    def add_name(self, name: str) -> int:
        if name in self._name_index_map:
            return self._name_index_map[name]
        idx = len(self.names)
        self.names.append(name)
        self._name_index_map[name] = idx
        return idx

    def add_mapping(self, gen_line: int, gen_col: int, source: str, orig_line: int, orig_col: int, name: Optional[str] = None) -> SourceMapping:
        src_idx = self.add_source(source)
        name_idx = self.add_name(name) if name else None
        m = SourceMapping(
            generated_line=gen_line,
            generated_column=gen_col,
            source_index=src_idx,
            original_line=orig_line,
            original_column=orig_col,
            name_index=name_idx,
        )
        self.mappings.append(m)
        return m

    def _encode_mappings(self) -> str:
        sorted_mappings = sorted(self.mappings, key=lambda m: (m.generated_line, m.generated_column))

        lines_map: Dict[int, List[SourceMapping]] = collections.defaultdict(list)
        for m in sorted_mappings:
            lines_map[m.generated_line].append(m)

        if not sorted_mappings:
            return ""

        max_line = max(m.generated_line for m in sorted_mappings)
        result_parts: List[str] = []

        prev_gen_col = 0
        prev_src_idx = 0
        prev_orig_line = 0
        prev_orig_col = 0
        prev_name_idx = 0

        for line_num in range(1, max_line + 1):
            line_mappings = lines_map.get(line_num, [])
            if not line_mappings:
                result_parts.append("")
                continue

            segments: List[str] = []
            prev_gen_col = 0
            for m in line_mappings:
                seg = _vlq_encode(m.generated_column - prev_gen_col)
                seg += _vlq_encode(m.source_index - prev_src_idx)
                seg += _vlq_encode(m.original_line - prev_orig_line)
                seg += _vlq_encode(m.original_column - prev_orig_col)
                if m.name_index is not None:
                    seg += _vlq_encode(m.name_index - prev_name_idx)
                    prev_name_idx = m.name_index
                prev_gen_col = m.generated_column
                prev_src_idx = m.source_index
                prev_orig_line = m.original_line
                prev_orig_col = m.original_column
                segments.append(seg)
            result_parts.append(",".join(segments))

        return ";".join(result_parts)

    def generate(self) -> Dict[str, Any]:
        return {
            "version": 3,
            "file": self.file,
            "sourceRoot": self.source_root,
            "sources": self.sources,
            "sourcesContent": self.sources_content,
            "names": self.names,
            "mappings": self._encode_mappings(),
        }

    def generate_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.generate(), indent=indent)

    def generate_inline(self) -> str:
        raw = self.generate_json()
        encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
        return f"//# sourceMappingURL=data:application/json;base64,{encoded}"

    def write_file(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.generate_json(indent=2), encoding="utf-8")
        return output_path

    def generate_multi_file(self, file_mapping: Dict[str, List[SourceMapping]]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for fname, maps in file_mapping.items():
            gen = SourceMapGenerator(
                source_root=self.source_root,
                file=fname,
            )
            for m in maps:
                src = self.sources[m.source_index] if m.source_index < len(self.sources) else ""
                name = self.names[m.name_index] if m.name_index is not None and m.name_index < len(self.names) else None
                gen.add_mapping(m.generated_line, m.generated_column, src, m.original_line, m.original_column, name)
            result[fname] = gen.generate()
        return result


# ---------------------------------------------------------------------------
# 18. TripleSlashReferenceGenerator
# ---------------------------------------------------------------------------


@dataclass
class TripleSlashReference:
    kind: str  # "path" | "types" | "lib" | "no-default-lib"
    value: str = ""

    def render(self) -> str:
        if self.kind == "path":
            return f'/// <reference path="{self.value}" />'
        if self.kind == "types":
            return f'/// <reference types="{self.value}" />'
        if self.kind == "lib":
            return f'/// <reference lib="{self.value}" />'
        if self.kind == "no-default-lib":
            return '/// <reference no-default-lib="true" />'
        return f'/// <reference {self.kind}="{self.value}" />'


@dataclass
class TripleSlashReferenceGenerator:
    references: List[TripleSlashReference] = field(default_factory=list)

    def add_path(self, path: str) -> TripleSlashReference:
        ref = TripleSlashReference(kind="path", value=path)
        if ref not in self.references:
            self.references.append(ref)
        return ref

    def add_types(self, package: str) -> TripleSlashReference:
        ref = TripleSlashReference(kind="types", value=package)
        if ref not in self.references:
            self.references.append(ref)
        return ref

    def add_lib(self, lib: str) -> TripleSlashReference:
        ref = TripleSlashReference(kind="lib", value=lib)
        if ref not in self.references:
            self.references.append(ref)
        return ref

    def add_no_default_lib(self) -> TripleSlashReference:
        ref = TripleSlashReference(kind="no-default-lib")
        for r in self.references:
            if r.kind == "no-default-lib":
                return r
        self.references.append(ref)
        return ref

    def render(self) -> str:
        return "\n".join(r.render() for r in self.references)

    def render_for_file(self, file_path: str) -> str:
        relevant: List[TripleSlashReference] = []
        for ref in self.references:
            if ref.kind == "no-default-lib":
                relevant.append(ref)
            elif ref.kind == "path":
                relevant.append(ref)
            elif ref.kind == "types":
                relevant.append(ref)
            elif ref.kind == "lib":
                relevant.append(ref)
        return "\n".join(r.render() for r in relevant)


# ---------------------------------------------------------------------------
# 19. DtsGenerator - main orchestrator
# ---------------------------------------------------------------------------


@dataclass
class DtsGeneratorConfig:
    ts_version: str = "5.0"
    output_dir: Path = field(default_factory=lambda: Path("dist"))
    generate_source_maps: bool = False
    inline_source_maps: bool = False
    line_width: int = 100
    header_comment: Optional[str] = None
    include_jsdoc: bool = True
    module_name: Optional[str] = None


@dataclass
class DtsGenerator:
    config: DtsGeneratorConfig = field(default_factory=DtsGeneratorConfig)
    interface_gen: InterfaceGenerator = field(default_factory=InterfaceGenerator)
    type_alias_gen: TypeAliasGenerator = field(default_factory=TypeAliasGenerator)
    function_gen: FunctionDeclarationGenerator = field(default_factory=FunctionDeclarationGenerator)
    class_gen: ClassDeclarationGenerator = field(default_factory=ClassDeclarationGenerator)
    module_gen: ModuleDeclarationGenerator = field(default_factory=ModuleDeclarationGenerator)
    generic_gen: GenericTypeGenerator = field(default_factory=GenericTypeGenerator)
    enum_gen: EnumGenerator = field(default_factory=EnumGenerator)
    conditional_gen: ConditionalTypeGenerator = field(default_factory=ConditionalTypeGenerator)
    mapped_gen: MappedTypeGenerator = field(default_factory=MappedTypeGenerator)
    guard_gen: TypeGuardGenerator = field(default_factory=TypeGuardGenerator)
    assertion_gen: AssertionFunctionGenerator = field(default_factory=AssertionFunctionGenerator)
    file_writer: DeclarationFileWriter = field(default_factory=DeclarationFileWriter)
    merger: DeclarationMerger = field(default_factory=DeclarationMerger)
    jsdoc_gen: JSDocCommentGenerator = field(default_factory=JSDocCommentGenerator)
    import_resolver: ImportExportResolver = field(default_factory=ImportExportResolver)
    source_map_gen: SourceMapGenerator = field(default_factory=SourceMapGenerator)
    triple_slash_gen: TripleSlashReferenceGenerator = field(default_factory=TripleSlashReferenceGenerator)

    def __post_init__(self) -> None:
        self.file_writer.output_dir = self.config.output_dir
        self.file_writer.line_width = self.config.line_width
        if self.config.header_comment:
            self.file_writer.header_comment = self.config.header_comment

    def load_analysis(self, analysis: Dict[str, Any]) -> None:
        """Load analysis results from a dict and populate sub-generators."""
        for iface in analysis.get("interfaces", []):
            self.interface_gen.add_interface(iface)

        for alias in analysis.get("typeAliases", []):
            self.type_alias_gen.add_alias(alias)

        for func in analysis.get("functions", []):
            self.function_gen.add_function(func)

        for cls in analysis.get("classes", []):
            self.class_gen.add_class(cls)

        for mod in analysis.get("modules", []):
            self.module_gen.add_module(mod)

        for en in analysis.get("enums", []):
            self.enum_gen.add_enum(en)

        for guard in analysis.get("typeGuards", []):
            self.guard_gen.add_type_guard(guard)

        for assertion in analysis.get("assertions", []):
            self.assertion_gen.add_assertion(assertion)

        for imp in analysis.get("imports", []):
            self.import_resolver.add_import(
                imp.get("module", ""),
                named=imp.get("named"),
                default_name=imp.get("default"),
                namespace_name=imp.get("namespace"),
                type_only=imp.get("typeOnly", False),
            )

        for ref in analysis.get("references", []):
            kind = ref.get("kind", "types")
            if kind == "path":
                self.triple_slash_gen.add_path(ref.get("value", ""))
            elif kind == "types":
                self.triple_slash_gen.add_types(ref.get("value", ""))
            elif kind == "lib":
                self.triple_slash_gen.add_lib(ref.get("value", ""))
            elif kind == "no-default-lib":
                self.triple_slash_gen.add_no_default_lib()

    def generate_declarations(self) -> str:
        """Generate full .d.ts file content as a string."""
        # Merge interfaces
        merged_interfaces = self.merger.merge_interfaces(self.interface_gen.declarations)
        self.interface_gen.declarations = merged_interfaces

        # Merge namespaces
        ns_decls = [m for m in self.module_gen.declarations if m.is_namespace]
        non_ns = [m for m in self.module_gen.declarations if not m.is_namespace]
        merged_ns = self.merger.merge_namespaces(ns_decls)
        merged_modules = self.merger.merge_module_augmentations(non_ns)
        self.module_gen.declarations = merged_ns + merged_modules

        sections: List[FileSection] = []

        # Triple-slash references
        refs_str = self.triple_slash_gen.render()
        if refs_str:
            sections.append(FileSection(kind="references", content=refs_str, order=1))

        # Header
        mod_name = self.config.module_name or "generated"
        sections.append(self.file_writer.build_header_section(mod_name, self.config.ts_version))

        # Imports
        imports_str = self.import_resolver.render_imports()
        if imports_str:
            sections.append(FileSection(kind="imports", content=imports_str, order=10))

        # Enums
        enums_str = self.enum_gen.render_all()
        if enums_str:
            sections.append(FileSection(kind="enums", content=enums_str, order=20))

        # Type aliases
        aliases_str = self.type_alias_gen.render_all()
        if aliases_str:
            sections.append(FileSection(kind="types", content=aliases_str, order=30))

        # Interfaces
        ifaces_str = self.interface_gen.render_all()
        if ifaces_str:
            sections.append(FileSection(kind="interfaces", content=ifaces_str, order=40))

        # Classes
        classes_str = self.class_gen.render_all()
        if classes_str:
            sections.append(FileSection(kind="classes", content=classes_str, order=50))

        # Functions
        funcs_str = self.function_gen.render_all()
        if funcs_str:
            sections.append(FileSection(kind="functions", content=funcs_str, order=60))

        # Type guards
        guards_str = self.guard_gen.render_all()
        if guards_str:
            sections.append(FileSection(kind="guards", content=guards_str, order=70))

        # Assertion functions
        asserts_str = self.assertion_gen.render_all()
        if asserts_str:
            sections.append(FileSection(kind="assertions", content=asserts_str, order=75))

        # Modules / namespaces
        modules_str = self.module_gen.render_all()
        if modules_str:
            sections.append(FileSection(kind="modules", content=modules_str, order=80))

        # Re-exports
        reexports_str = self.import_resolver.render_re_exports()
        if reexports_str:
            sections.append(FileSection(kind="exports", content=reexports_str, order=90))

        # Preserved manual declarations
        for name, content in self.merger.get_preserved().items():
            sections.append(FileSection(kind="manual", content=f"// Preserved: {name}\n{content}", order=95))

        return self.file_writer.write_string(sections)

    def generate_file(self, filename: str = "index.d.ts") -> Path:
        """Generate and write a .d.ts file."""
        content = self.generate_declarations()
        sections = [FileSection(kind="full", content=content, order=0)]
        path = self.file_writer.write_file(filename, sections)

        if self.config.generate_source_maps and not self.config.inline_source_maps:
            sm_path = path.with_suffix(path.suffix + ".map")
            self.source_map_gen.file = filename
            self.source_map_gen.write_file(sm_path)

        return path

    def generate_batch(self, file_specs: Dict[str, Dict[str, Any]]) -> Dict[str, Path]:
        """Generate multiple .d.ts files from a mapping of filename → analysis."""
        results: Dict[str, Path] = {}
        for fname, analysis in file_specs.items():
            gen = DtsGenerator(config=copy.deepcopy(self.config))
            gen.load_analysis(analysis)
            results[fname] = gen.generate_file(fname)
        return results

    def generate_index(self, module_files: List[str]) -> str:
        """Generate an index.d.ts that re-exports from all module files."""
        lines: List[str] = []
        for mf in sorted(module_files):
            mod_path = "./" + mf.replace(".d.ts", "").replace("\\", "/")
            lines.append(f"export * from '{mod_path}';")
        return "\n".join(lines) + "\n"

    def get_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def diff_declarations(self, old_content: str, new_content: str) -> str:
        """Produce a unified diff between old and new declaration content."""
        import difflib as _difflib
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = _difflib.unified_diff(old_lines, new_lines, fromfile="old.d.ts", tofile="new.d.ts")
        return "".join(diff)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_dts_generator(
    output_dir: str = "dist",
    ts_version: str = "5.0",
    generate_source_maps: bool = False,
    module_name: Optional[str] = None,
) -> DtsGenerator:
    cfg = DtsGeneratorConfig(
        ts_version=ts_version,
        output_dir=Path(output_dir),
        generate_source_maps=generate_source_maps,
        module_name=module_name,
    )
    return DtsGenerator(config=cfg)
