"""
TypeScript SSA Compiler — TS AST → unified SSA IR translation.

Consumes a JSON-based AST representation (from the TypeScript Compiler API
via a Node.js bridge) and produces the unified SSA-based IR used by the
refinement type inference engine.

Key components
--------------
* ``TypeScriptSSACompiler`` – top-level driver (parse → desugar → lower → SSA)
* ``TSASTNode`` / ``TSNodeKind`` – lightweight Python mirror of TS AST nodes
* ``TSSSABuilder`` – statement/expression → IR basic-block translation
* ``TSScopeResolver`` – block (let/const) vs function (var) scoping
* ``TSDesugarer`` – optional chaining, nullish coalescing, enum, destructuring
* ``TSTypeExtractor`` – declared TS types → ``IRType``
* ``DiscriminatedUnionHandler`` – discriminated-union narrowing
* ``TSBuiltinSignatures`` – known signatures for JS/TS builtins
* ``NodeJSBridge`` – subprocess communication with the TS compiler API
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from src.ir.unified import (
    ANY_TYPE,
    BOOL_TYPE,
    BOTTOM_TYPE,
    FLOAT_TYPE,
    INT_TYPE,
    NONE_TYPE,
    STR_TYPE,
    TOP_TYPE,
    UNKNOWN_TYPE,
    AssignNode,
    BinOp,
    BinOpNode,
    BranchNode,
    CFG,
    CFGBuilder,
    CFGEdge,
    CallNode,
    CompareNode,
    CompareOp,
    ConditionalBranchNode,
    ConstantNode,
    DominatorTree,
    EdgeKind,
    GatedPhiNode,
    IRBasicBlock,
    IRClass,
    IRFunction,
    IRModule,
    IRNode,
    IRType,
    ImportNode,
    LanguageTag,
    LiteralNode,
    LoadAttrNode,
    LoadSubscriptNode,
    PhiNode,
    RaiseNode,
    RefinementType,
    ReturnNode,
    SSABuilder,
    SSAValue,
    SSAVariable,
    SourceLocation,
    StoreAttrNode,
    StoreSubscriptNode,
    TruthinessCoercionNode,
    TypeNarrowNode,
    UnaryOp,
    UnaryOpNode,
    YieldNode,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TS AST Node Kind Enum
# ---------------------------------------------------------------------------

class TSNodeKind(Enum):
    """Mirrors the ``SyntaxKind`` enum from the TypeScript compiler API."""
    SourceFile = auto()
    # Declarations
    FunctionDeclaration = auto()
    VariableDeclaration = auto()
    VariableDeclarationList = auto()
    VariableStatement = auto()
    ClassDeclaration = auto()
    InterfaceDeclaration = auto()
    TypeAliasDeclaration = auto()
    EnumDeclaration = auto()
    EnumMember = auto()
    NamespaceDeclaration = auto()
    ModuleDeclaration = auto()
    ImportDeclaration = auto()
    ExportDeclaration = auto()
    ExportAssignment = auto()
    MethodDeclaration = auto()
    PropertyDeclaration = auto()
    Constructor = auto()
    GetAccessor = auto()
    SetAccessor = auto()
    Parameter = auto()
    Decorator = auto()
    # Statements
    Block = auto()
    IfStatement = auto()
    ForStatement = auto()
    ForOfStatement = auto()
    ForInStatement = auto()
    WhileStatement = auto()
    DoStatement = auto()
    SwitchStatement = auto()
    CaseClause = auto()
    DefaultClause = auto()
    ReturnStatement = auto()
    ThrowStatement = auto()
    TryStatement = auto()
    CatchClause = auto()
    BreakStatement = auto()
    ContinueStatement = auto()
    LabeledStatement = auto()
    ExpressionStatement = auto()
    EmptyStatement = auto()
    WithStatement = auto()
    DebuggerStatement = auto()
    # Expressions
    Identifier = auto()
    NumericLiteral = auto()
    StringLiteral = auto()
    NoSubstitutionTemplateLiteral = auto()
    TemplateLiteral = auto()
    TemplateExpression = auto()
    TemplateHead = auto()
    TemplateMiddle = auto()
    TemplateTail = auto()
    TemplateSpan = auto()
    RegularExpressionLiteral = auto()
    TrueKeyword = auto()
    FalseKeyword = auto()
    NullKeyword = auto()
    UndefinedKeyword = auto()
    ThisKeyword = auto()
    SuperKeyword = auto()
    VoidExpression = auto()
    DeleteExpression = auto()
    TypeOfExpression = auto()
    InstanceOfExpression = auto()
    InExpression = auto()
    BinaryExpression = auto()
    PrefixUnaryExpression = auto()
    PostfixUnaryExpression = auto()
    ConditionalExpression = auto()
    CallExpression = auto()
    NewExpression = auto()
    PropertyAccessExpression = auto()
    ElementAccessExpression = auto()
    TaggedTemplateExpression = auto()
    AsExpression = auto()
    NonNullExpression = auto()
    TypeAssertionExpression = auto()
    ParenthesizedExpression = auto()
    ArrowFunction = auto()
    FunctionExpression = auto()
    ObjectLiteralExpression = auto()
    ObjectLiteral = auto()
    ArrayLiteralExpression = auto()
    ArrayLiteral = auto()
    SpreadElement = auto()
    SpreadAssignment = auto()
    PropertyAssignment = auto()
    ShorthandPropertyAssignment = auto()
    ComputedPropertyName = auto()
    AwaitExpression = auto()
    YieldExpression = auto()
    ClassExpression = auto()
    OmittedExpression = auto()
    CommaExpression = auto()
    SatisfiesExpression = auto()
    # Type guard
    TypeGuardExpression = auto()
    TypePredicateNode = auto()
    # Type nodes
    TypeReference = auto()
    ArrayType = auto()
    TupleType = auto()
    UnionType = auto()
    IntersectionType = auto()
    LiteralType = auto()
    FunctionType = auto()
    TypeQuery = auto()
    IndexedAccessType = auto()
    MappedType = auto()
    ConditionalType = auto()
    InferType = auto()
    TemplateLiteralType = auto()
    TypeParameter = auto()
    AnyKeyword = auto()
    NumberKeyword = auto()
    StringKeyword_ = auto()
    BooleanKeyword = auto()
    NeverKeyword = auto()
    UnknownKeyword = auto()
    ObjectKeyword = auto()
    VoidKeyword = auto()
    SymbolKeyword = auto()
    BigIntKeyword = auto()
    # Binding patterns
    ObjectBindingPattern = auto()
    ArrayBindingPattern = auto()
    BindingElement = auto()
    # Misc
    EndOfFileToken = auto()
    Unknown = auto()

    @classmethod
    def from_ts_kind(cls, kind_str: str) -> TSNodeKind:
        """Map a TypeScript ``SyntaxKind`` string to our enum."""
        try:
            return cls[kind_str]
        except KeyError:
            return cls.Unknown


# ---------------------------------------------------------------------------
# TS AST Node dataclass
# ---------------------------------------------------------------------------

@dataclass
class TSASTNode:
    """Lightweight mirror of a TypeScript AST node (from JSON)."""
    kind: TSNodeKind
    text: str = ""
    children: List[TSASTNode] = field(default_factory=list)
    pos: int = 0
    end: int = 0
    line: int = 0
    column: int = 0
    type_info: Optional[str] = None
    flags: int = 0
    modifiers: List[str] = field(default_factory=list)
    name: Optional[str] = None
    operator_token: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> TSASTNode:
        """Recursively build from JSON produced by the Node.js bridge."""
        kind = TSNodeKind.from_ts_kind(data.get("kind", "Unknown"))
        children = [cls.from_json(c) for c in data.get("children", [])]
        return cls(
            kind=kind,
            text=data.get("text", ""),
            children=children,
            pos=data.get("pos", 0),
            end=data.get("end", 0),
            line=data.get("line", 0),
            column=data.get("column", 0),
            type_info=data.get("typeInfo"),
            flags=data.get("flags", 0),
            modifiers=data.get("modifiers", []),
            name=data.get("name"),
            operator_token=data.get("operatorToken"),
            properties=data.get("properties", {}),
        )

    def children_of_kind(self, kind: TSNodeKind) -> List[TSASTNode]:
        return [c for c in self.children if c.kind == kind]

    def first_child_of_kind(self, kind: TSNodeKind) -> Optional[TSASTNode]:
        for c in self.children:
            if c.kind == kind:
                return c
        return None

    def child_by_field(self, field_name: str) -> Optional[TSASTNode]:
        """Retrieve a named child from ``properties``."""
        data = self.properties.get(field_name)
        if isinstance(data, dict):
            return TSASTNode.from_json(data)
        return None

    @property
    def source_location(self) -> SourceLocation:
        return SourceLocation(
            file="<ts>",
            line=self.line,
            column=self.column,
        )


# ---------------------------------------------------------------------------
# Scope Resolution
# ---------------------------------------------------------------------------

class _ScopeKind(Enum):
    Module = auto()
    Function = auto()
    Block = auto()
    Class = auto()


@dataclass
class _Scope:
    kind: _ScopeKind
    variables: Dict[str, SSAValue] = field(default_factory=dict)
    parent: Optional[_Scope] = None
    label: Optional[str] = None

    def lookup(self, name: str) -> Optional[SSAValue]:
        if name in self.variables:
            return self.variables[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None


class TSScopeResolver:
    """
    Manages TypeScript scoping semantics:
    - ``let`` / ``const``: block-scoped
    - ``var``: function-scoped (hoisted)
    - Top-level declarations: module-scoped
    """

    def __init__(self) -> None:
        self._root = _Scope(kind=_ScopeKind.Module)
        self._current = self._root
        self._function_scope: _Scope = self._root

    @property
    def current(self) -> _Scope:
        return self._current

    def push_scope(self, kind: _ScopeKind, label: Optional[str] = None) -> _Scope:
        scope = _Scope(kind=kind, parent=self._current, label=label)
        self._current = scope
        if kind == _ScopeKind.Function:
            self._function_scope = scope
        return scope

    def pop_scope(self) -> _Scope:
        old = self._current
        if self._current.parent is not None:
            self._current = self._current.parent
        if old.kind == _ScopeKind.Function:
            p = self._current
            while p.kind != _ScopeKind.Function and p.parent is not None:
                p = p.parent
            self._function_scope = p
        return old

    def declare(self, name: str, val: SSAValue, decl_kind: str = "let") -> None:
        """
        Declare *name* in the appropriate scope.

        ``let`` / ``const`` go into the current (block) scope.
        ``var`` is hoisted to the enclosing function scope.
        """
        if decl_kind == "var":
            self._function_scope.variables[name] = val
        else:
            self._current.variables[name] = val

    def resolve(self, name: str) -> Optional[SSAValue]:
        return self._current.lookup(name)

    def update(self, name: str, val: SSAValue) -> None:
        scope: Optional[_Scope] = self._current
        while scope is not None:
            if name in scope.variables:
                scope.variables[name] = val
                return
            scope = scope.parent
        self._current.variables[name] = val


# ---------------------------------------------------------------------------
# TypeScript Type Extractor
# ---------------------------------------------------------------------------

_TS_TYPE_MAP: Dict[str, IRType] = {
    "number": FLOAT_TYPE,
    "string": STR_TYPE,
    "boolean": BOOL_TYPE,
    "void": NONE_TYPE,
    "null": NONE_TYPE,
    "undefined": NONE_TYPE,
    "never": BOTTOM_TYPE,
    "any": ANY_TYPE,
    "unknown": UNKNOWN_TYPE,
    "object": IRType("object"),
    "symbol": IRType("symbol"),
    "bigint": IRType("bigint"),
}


class TSTypeExtractor:
    """Extract declared TypeScript types and map them to IR types."""

    def extract(self, node: TSASTNode) -> Optional[IRType]:
        """Convert a TS type annotation AST node to an ``IRType``."""
        if node.kind == TSNodeKind.TypeReference:
            return self._extract_type_reference(node)
        if node.kind == TSNodeKind.ArrayType:
            elem = node.children[0] if node.children else None
            elem_type = self.extract(elem) if elem else UNKNOWN_TYPE
            return IRType("Array", args=[elem_type] if elem_type else [])
        if node.kind == TSNodeKind.UnionType:
            return self._extract_union(node)
        if node.kind == TSNodeKind.IntersectionType:
            return self._extract_intersection(node)
        if node.kind == TSNodeKind.TupleType:
            elems = [self.extract(c) or UNKNOWN_TYPE for c in node.children]
            return IRType("Tuple", args=elems)
        if node.kind == TSNodeKind.FunctionType:
            return IRType("Function")
        if node.kind == TSNodeKind.LiteralType:
            return self._extract_literal_type(node)
        if node.kind in (
            TSNodeKind.AnyKeyword,
            TSNodeKind.NumberKeyword,
            TSNodeKind.StringKeyword_,
            TSNodeKind.BooleanKeyword,
            TSNodeKind.NeverKeyword,
            TSNodeKind.UnknownKeyword,
            TSNodeKind.ObjectKeyword,
            TSNodeKind.VoidKeyword,
            TSNodeKind.SymbolKeyword,
            TSNodeKind.BigIntKeyword,
        ):
            return self._keyword_to_type(node.kind)
        return UNKNOWN_TYPE

    def extract_from_string(self, type_str: str) -> IRType:
        """Map a raw type-info string to an ``IRType``."""
        normalized = type_str.strip()
        if normalized in _TS_TYPE_MAP:
            return _TS_TYPE_MAP[normalized]
        if normalized.endswith("[]"):
            inner = self.extract_from_string(normalized[:-2])
            return IRType("Array", args=[inner])
        if "|" in normalized:
            parts = [self.extract_from_string(p.strip()) for p in normalized.split("|")]
            nullable = any(
                p.name in ("None", "undefined", "null") for p in parts
            )
            non_null = [p for p in parts if p.name not in ("None", "undefined", "null")]
            if len(non_null) == 1:
                t = IRType(non_null[0].name, args=list(non_null[0].args), nullable=nullable)
                return t
            return IRType("Union", args=non_null, nullable=nullable)
        if normalized.startswith("Array<") and normalized.endswith(">"):
            inner = self.extract_from_string(normalized[6:-1])
            return IRType("Array", args=[inner])
        if normalized.startswith("Promise<") and normalized.endswith(">"):
            inner = self.extract_from_string(normalized[8:-1])
            return IRType("Promise", args=[inner])
        if normalized.startswith("Map<") and normalized.endswith(">"):
            return IRType("Map")
        if normalized.startswith("Set<") and normalized.endswith(">"):
            return IRType("Set")
        return IRType(normalized)

    # ----- helpers ----------------------------------------------------------

    def _extract_type_reference(self, node: TSASTNode) -> IRType:
        ref_name = node.name or node.text or "unknown"
        type_args: List[IRType] = []
        for child in node.children:
            extracted = self.extract(child)
            if extracted:
                type_args.append(extracted)
        return IRType(ref_name, args=type_args)

    def _extract_union(self, node: TSASTNode) -> IRType:
        members = [self.extract(c) or UNKNOWN_TYPE for c in node.children]
        nullable = any(m.name in ("None", "undefined", "null") for m in members)
        non_null = [m for m in members if m.name not in ("None", "undefined", "null")]
        if len(non_null) == 1:
            return IRType(non_null[0].name, args=list(non_null[0].args), nullable=nullable)
        return IRType("Union", args=non_null, nullable=nullable)

    def _extract_intersection(self, node: TSASTNode) -> IRType:
        members = [self.extract(c) or UNKNOWN_TYPE for c in node.children]
        return IRType("Intersection", args=members)

    def _extract_literal_type(self, node: TSASTNode) -> IRType:
        if node.text in ("true", "false"):
            return BOOL_TYPE
        if node.text == "null":
            return NONE_TYPE
        try:
            float(node.text)
            return FLOAT_TYPE
        except (ValueError, TypeError):
            pass
        return STR_TYPE

    def _keyword_to_type(self, kind: TSNodeKind) -> IRType:
        mapping: Dict[TSNodeKind, IRType] = {
            TSNodeKind.AnyKeyword: ANY_TYPE,
            TSNodeKind.NumberKeyword: FLOAT_TYPE,
            TSNodeKind.StringKeyword_: STR_TYPE,
            TSNodeKind.BooleanKeyword: BOOL_TYPE,
            TSNodeKind.NeverKeyword: BOTTOM_TYPE,
            TSNodeKind.UnknownKeyword: UNKNOWN_TYPE,
            TSNodeKind.ObjectKeyword: IRType("object"),
            TSNodeKind.VoidKeyword: NONE_TYPE,
            TSNodeKind.SymbolKeyword: IRType("symbol"),
            TSNodeKind.BigIntKeyword: IRType("bigint"),
        }
        return mapping.get(kind, UNKNOWN_TYPE)


# ---------------------------------------------------------------------------
# Discriminated Union Handler
# ---------------------------------------------------------------------------

@dataclass
class _DiscriminantInfo:
    """Tracks a discriminant field for a union type."""
    field_name: str
    tag_values: Dict[str, IRType]  # tag_value → narrowed type


class DiscriminatedUnionHandler:
    """
    Models discriminated-union narrowing.

    Given::

        type Shape =
            | { kind: "circle"; radius: number }
            | { kind: "square"; side: number }

    After checking ``shape.kind === "circle"`` the handler narrows to the
    ``circle`` variant.
    """

    def __init__(self) -> None:
        self._unions: Dict[str, _DiscriminantInfo] = {}

    def register_union(
        self,
        type_name: str,
        discriminant_field: str,
        variants: Dict[str, IRType],
    ) -> None:
        self._unions[type_name] = _DiscriminantInfo(
            field_name=discriminant_field,
            tag_values=variants,
        )

    def try_narrow(
        self,
        type_name: str,
        field_name: str,
        value: str,
    ) -> Optional[IRType]:
        """
        Attempt to narrow *type_name* when ``field_name === value``.

        Returns the narrowed ``IRType`` or ``None`` if not a known
        discriminated union.
        """
        info = self._unions.get(type_name)
        if info is None or info.field_name != field_name:
            return None
        return info.tag_values.get(value)

    def is_discriminant_access(self, type_name: str, field_name: str) -> bool:
        info = self._unions.get(type_name)
        return info is not None and info.field_name == field_name


# ---------------------------------------------------------------------------
# TS Builtin Signatures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _BuiltinSig:
    """Simplified builtin signature: param types → return type."""
    name: str
    params: Tuple[IRType, ...]
    return_type: IRType
    is_type_guard: bool = False


class TSBuiltinSignatures:
    """Known signatures for commonly-used TypeScript / JavaScript builtins."""

    _BUILTINS: Dict[str, _BuiltinSig] = {}

    @classmethod
    def _init_builtins(cls) -> None:
        if cls._BUILTINS:
            return
        array_type = IRType("Array")
        object_type = IRType("object")
        string_type = STR_TYPE
        number_type = FLOAT_TYPE
        bool_type = BOOL_TYPE
        any_type = ANY_TYPE

        sigs: List[_BuiltinSig] = [
            _BuiltinSig("Array.isArray", (any_type,), bool_type, is_type_guard=True),
            _BuiltinSig("Array.from", (any_type,), array_type),
            _BuiltinSig("Array.of", (), array_type),
            _BuiltinSig("Object.keys", (object_type,), IRType("Array", args=[string_type])),
            _BuiltinSig("Object.values", (object_type,), IRType("Array", args=[any_type])),
            _BuiltinSig("Object.entries", (object_type,), IRType("Array", args=[IRType("Tuple", args=[string_type, any_type])])),
            _BuiltinSig("Object.assign", (object_type,), object_type),
            _BuiltinSig("Object.freeze", (any_type,), any_type),
            _BuiltinSig("Object.create", (any_type,), object_type),
            _BuiltinSig("Object.hasOwnProperty", (string_type,), bool_type),
            _BuiltinSig("JSON.parse", (string_type,), any_type),
            _BuiltinSig("JSON.stringify", (any_type,), string_type),
            _BuiltinSig("Number.isNaN", (any_type,), bool_type),
            _BuiltinSig("Number.isFinite", (any_type,), bool_type),
            _BuiltinSig("Number.parseInt", (string_type,), number_type),
            _BuiltinSig("Number.parseFloat", (string_type,), number_type),
            _BuiltinSig("parseInt", (string_type,), number_type),
            _BuiltinSig("parseFloat", (string_type,), number_type),
            _BuiltinSig("isNaN", (any_type,), bool_type),
            _BuiltinSig("isFinite", (any_type,), bool_type),
            _BuiltinSig("String", (any_type,), string_type),
            _BuiltinSig("Number", (any_type,), number_type),
            _BuiltinSig("Boolean", (any_type,), bool_type),
            _BuiltinSig("typeof", (any_type,), string_type),
            _BuiltinSig("Math.abs", (number_type,), number_type),
            _BuiltinSig("Math.floor", (number_type,), number_type),
            _BuiltinSig("Math.ceil", (number_type,), number_type),
            _BuiltinSig("Math.round", (number_type,), number_type),
            _BuiltinSig("Math.max", (), number_type),
            _BuiltinSig("Math.min", (), number_type),
            _BuiltinSig("Math.random", (), number_type),
            _BuiltinSig("Math.sqrt", (number_type,), number_type),
            _BuiltinSig("Math.pow", (number_type, number_type), number_type),
            _BuiltinSig("Math.log", (number_type,), number_type),
            _BuiltinSig("console.log", (), NONE_TYPE),
            _BuiltinSig("console.error", (), NONE_TYPE),
            _BuiltinSig("console.warn", (), NONE_TYPE),
            _BuiltinSig("Promise.resolve", (any_type,), IRType("Promise", args=[any_type])),
            _BuiltinSig("Promise.reject", (any_type,), IRType("Promise", args=[BOTTOM_TYPE])),
            _BuiltinSig("Promise.all", (any_type,), IRType("Promise", args=[array_type])),
        ]
        for sig in sigs:
            cls._BUILTINS[sig.name] = sig

    @classmethod
    def lookup(cls, name: str) -> Optional[_BuiltinSig]:
        cls._init_builtins()
        return cls._BUILTINS.get(name)

    @classmethod
    def return_type(cls, name: str) -> Optional[IRType]:
        sig = cls.lookup(name)
        return sig.return_type if sig else None

    @classmethod
    def is_type_guard(cls, name: str) -> bool:
        sig = cls.lookup(name)
        return sig.is_type_guard if sig else False


# ---------------------------------------------------------------------------
# TypeScript Desugarer
# ---------------------------------------------------------------------------

class TSDesugarer:
    """
    Desugars TypeScript-specific syntax into simpler IR-level constructs.

    * Optional chaining (``a?.b``) → null-check + property access
    * Nullish coalescing (``a ?? b``) → null-check + ternary
    * Enum → object literal with numeric/string values
    * Destructuring → individual assignments
    * Spread → explicit iteration / ``Object.assign``
    """

    def __init__(self, builder: TSSSABuilder) -> None:
        self._builder = builder

    # -- Optional chaining ---------------------------------------------------

    def desugar_optional_chain(
        self,
        obj_val: SSAValue,
        prop_name: str,
        loc: Optional[SourceLocation] = None,
    ) -> SSAValue:
        """
        Lower ``obj?.prop`` to::

            _nc = obj === null || obj === undefined
            if _nc goto null_branch else access_branch
            null_branch: result = undefined
            access_branch: result = obj.prop
            merge: result_phi = φ(null_branch: undefined, access_branch: obj.prop)
        """
        b = self._builder
        null_val = b._emit_constant(None, NONE_TYPE, loc)
        nc_cond = b._new_ssa("_oc_null_check")
        b._emit(CompareNode(
            target=nc_cond, left=obj_val, right=null_val,
            op=CompareOp.Eq, location=loc,
            language_tag=LanguageTag.TypeScript,
        ))

        null_lbl = b._fresh_label("oc_null")
        access_lbl = b._fresh_label("oc_access")
        merge_lbl = b._fresh_label("oc_merge")

        b._terminate(ConditionalBranchNode(
            condition=nc_cond, true_label=null_lbl,
            false_label=access_lbl, location=loc,
        ))

        # null branch
        b._start_block(null_lbl)
        undef_val = b._emit_constant(None, NONE_TYPE, loc)
        b._terminate(BranchNode(target_label=merge_lbl, location=loc))

        # access branch
        b._start_block(access_lbl)
        access_result = b._new_ssa("_oc_result")
        b._emit(LoadAttrNode(
            target=access_result, obj=obj_val, attr=prop_name,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        b._terminate(BranchNode(target_label=merge_lbl, location=loc))

        # merge
        b._start_block(merge_lbl)
        merged = b._new_ssa("_oc_merged")
        b._emit(PhiNode(
            target=merged,
            incoming={null_lbl: undef_val, access_lbl: access_result},
            location=loc,
        ))

        b._cfg_builder.connect(b._prev_block_label(2), null_lbl, EdgeKind.TrueGuard)
        b._cfg_builder.connect(b._prev_block_label(2), access_lbl, EdgeKind.FalseGuard)
        b._cfg_builder.connect(null_lbl, merge_lbl)
        b._cfg_builder.connect(access_lbl, merge_lbl)

        return merged

    # -- Nullish coalescing --------------------------------------------------

    def desugar_nullish_coalescing(
        self,
        left_val: SSAValue,
        right_node: TSASTNode,
        loc: Optional[SourceLocation] = None,
    ) -> SSAValue:
        """
        Lower ``a ?? b`` to::

            _nc = a === null || a === undefined
            if _nc goto rhs_branch else lhs_branch
            lhs_branch: result = a
            rhs_branch: result = <compile b>
            merge: result_phi = φ(...)
        """
        b = self._builder
        null_val = b._emit_constant(None, NONE_TYPE, loc)
        nc_cond = b._new_ssa("_nc_check")
        b._emit(CompareNode(
            target=nc_cond, left=left_val, right=null_val,
            op=CompareOp.Eq, location=loc,
            language_tag=LanguageTag.TypeScript,
        ))

        lhs_lbl = b._fresh_label("nc_lhs")
        rhs_lbl = b._fresh_label("nc_rhs")
        merge_lbl = b._fresh_label("nc_merge")
        origin = b._current_label()

        b._terminate(ConditionalBranchNode(
            condition=nc_cond, true_label=rhs_lbl,
            false_label=lhs_lbl, location=loc,
        ))

        # lhs (non-null) branch
        b._start_block(lhs_lbl)
        lhs_result = b._new_ssa("_nc_lhs")
        b._emit(AssignNode(target=lhs_result, value=left_val, location=loc))
        b._terminate(BranchNode(target_label=merge_lbl, location=loc))

        # rhs (null) branch
        b._start_block(rhs_lbl)
        rhs_result = b.compile_expression(right_node)
        b._terminate(BranchNode(target_label=merge_lbl, location=loc))

        # merge
        b._start_block(merge_lbl)
        merged = b._new_ssa("_nc_merged")
        b._emit(PhiNode(
            target=merged,
            incoming={lhs_lbl: lhs_result, rhs_lbl: rhs_result},
            location=loc,
        ))

        b._cfg_builder.connect(origin, rhs_lbl, EdgeKind.TrueGuard)
        b._cfg_builder.connect(origin, lhs_lbl, EdgeKind.FalseGuard)
        b._cfg_builder.connect(lhs_lbl, merge_lbl)
        b._cfg_builder.connect(rhs_lbl, merge_lbl)

        return merged

    # -- Enum desugaring -----------------------------------------------------

    def desugar_enum(self, node: TSASTNode) -> SSAValue:
        """
        Lower a TS ``enum`` declaration to an object literal with numbered
        or string-valued members.
        """
        b = self._builder
        loc = node.source_location
        enum_name = node.name or "_enum"
        members = node.children_of_kind(TSNodeKind.EnumMember)

        obj = b._new_ssa(enum_name)
        b._emit(ConstantNode(
            target=obj, value={}, type_annotation=IRType("object"),
            location=loc, language_tag=LanguageTag.TypeScript,
        ))

        auto_value = 0
        for member in members:
            member_name = member.name or member.text
            initializer = member.child_by_field("initializer")
            if initializer is not None:
                val = b.compile_expression(initializer)
            else:
                val = b._emit_constant(auto_value, FLOAT_TYPE, loc)
                auto_value += 1
            key = b._emit_constant(member_name, STR_TYPE, loc)
            b._emit(StoreSubscriptNode(
                obj=obj, index=key, value=val,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))

        b._scope.declare(enum_name, obj, "const")
        return obj

    # -- Destructuring -------------------------------------------------------

    def desugar_object_destructuring(
        self,
        source_val: SSAValue,
        pattern: TSASTNode,
        decl_kind: str = "let",
        loc: Optional[SourceLocation] = None,
    ) -> None:
        """Lower ``const { a, b: c, ...rest } = source`` to individual loads."""
        b = self._builder
        for child in pattern.children:
            if child.kind == TSNodeKind.BindingElement:
                prop_name = child.properties.get("propertyName") or child.name or child.text
                local_name = child.name or child.text
                loaded = b._new_ssa(local_name)
                b._emit(LoadAttrNode(
                    target=loaded, obj=source_val, attr=prop_name,
                    location=loc, language_tag=LanguageTag.TypeScript,
                ))
                b._scope.declare(local_name, loaded, decl_kind)
            elif child.kind == TSNodeKind.SpreadElement:
                rest_name = child.name or child.text or "_rest"
                rest_val = b._new_ssa(rest_name)
                spread_fn = b._resolve_or_create("Object.assign")
                empty_obj = b._emit_constant({}, IRType("object"), loc)
                b._emit(CallNode(
                    target=rest_val, callee=spread_fn,
                    args=[empty_obj, source_val],
                    location=loc, language_tag=LanguageTag.TypeScript,
                ))
                b._scope.declare(rest_name, rest_val, decl_kind)

    def desugar_array_destructuring(
        self,
        source_val: SSAValue,
        pattern: TSASTNode,
        decl_kind: str = "let",
        loc: Optional[SourceLocation] = None,
    ) -> None:
        """Lower ``const [a, b, ...rest] = source`` to indexed loads."""
        b = self._builder
        idx = 0
        for child in pattern.children:
            if child.kind == TSNodeKind.OmittedExpression:
                idx += 1
                continue
            if child.kind == TSNodeKind.SpreadElement:
                rest_name = child.name or child.text or "_rest"
                rest_val = b._new_ssa(rest_name)
                slice_fn = b._resolve_or_create("Array.prototype.slice")
                idx_val = b._emit_constant(idx, FLOAT_TYPE, loc)
                b._emit(CallNode(
                    target=rest_val, callee=slice_fn,
                    args=[source_val, idx_val],
                    location=loc, language_tag=LanguageTag.TypeScript,
                ))
                b._scope.declare(rest_name, rest_val, decl_kind)
                break

            local_name = child.name or child.text or f"_destr_{idx}"
            loaded = b._new_ssa(local_name)
            idx_val = b._emit_constant(idx, FLOAT_TYPE, loc)
            b._emit(LoadSubscriptNode(
                target=loaded, obj=source_val, index=idx_val,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            b._scope.declare(local_name, loaded, decl_kind)
            idx += 1

    # -- Spread --------------------------------------------------------------

    def desugar_spread_in_call(
        self,
        args: List[SSAValue],
        spread_indices: Set[int],
        loc: Optional[SourceLocation] = None,
    ) -> SSAValue:
        """Concatenate spread arguments into a single array."""
        b = self._builder
        result = b._new_ssa("_spread_args")
        concat_fn = b._resolve_or_create("Array.prototype.concat")
        empty = b._emit_constant([], IRType("Array"), loc)
        b._emit(CallNode(
            target=result, callee=concat_fn,
            args=[empty] + args,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        return result


# ---------------------------------------------------------------------------
# TS SSA Builder — core AST → IR translation
# ---------------------------------------------------------------------------

_TS_BINOP_MAP: Dict[str, BinOp] = {
    "+": BinOp.Add,
    "-": BinOp.Sub,
    "*": BinOp.Mul,
    "/": BinOp.Div,
    "%": BinOp.Mod,
    "**": BinOp.Pow,
    "&": BinOp.BitAnd,
    "|": BinOp.BitOr,
    "^": BinOp.BitXor,
    "<<": BinOp.LShift,
    ">>": BinOp.RShift,
}

_TS_COMPARE_MAP: Dict[str, CompareOp] = {
    "===": CompareOp.Eq,
    "!==": CompareOp.NotEq,
    "==": CompareOp.Eq,
    "!=": CompareOp.NotEq,
    "<": CompareOp.Lt,
    "<=": CompareOp.LtE,
    ">": CompareOp.Gt,
    ">=": CompareOp.GtE,
    "instanceof": CompareOp.Instanceof,
    "in": CompareOp.In,
}


class TSSSABuilder:
    """
    Translates a TypeScript AST (``TSASTNode`` tree) into the unified SSA IR.

    Handles all statement and expression forms, emitting IR nodes into an
    incrementally-constructed CFG via ``CFGBuilder``.
    """

    def __init__(self, source_file: str = "<ts>") -> None:
        self._source_file = source_file
        self._cfg_builder = CFGBuilder()
        self._scope = TSScopeResolver()
        self._type_extractor = TSTypeExtractor()
        self._discriminated_unions = DiscriminatedUnionHandler()
        self._desugarer = TSDesugarer(self)

        self._ssa_counter: int = 0
        self._label_counter: int = 0
        self._block_history: List[str] = []

        self._module = IRModule(
            name=source_file,
            source_language=LanguageTag.TypeScript,
            source_file=source_file,
        )

        # Loop control flow stacks for break/continue
        self._break_targets: List[str] = []
        self._continue_targets: List[str] = []
        # Switch fallthrough support
        self._switch_break_targets: List[str] = []
        # Try/catch/finally
        self._exception_handlers: List[str] = []

        # Start the entry block
        self._start_block("entry")

    # -- SSA helpers ---------------------------------------------------------

    def _new_ssa(self, hint: str = "_tmp") -> SSAValue:
        self._ssa_counter += 1
        return SSAValue(name=f"{hint}", version=0)

    def _fresh_label(self, hint: str = "bb") -> str:
        self._label_counter += 1
        return f"{hint}_{self._label_counter}"

    def _current_label(self) -> str:
        blk = self._cfg_builder.current_block
        return blk.label if blk else "entry"

    def _prev_block_label(self, n: int = 1) -> str:
        if len(self._block_history) >= n:
            return self._block_history[-n]
        return "entry"

    def _emit(self, node: IRNode) -> None:
        self._cfg_builder.emit(node)

    def _terminate(self, node: IRNode) -> None:
        self._cfg_builder.set_terminator(node)

    def _start_block(self, label: str) -> None:
        self._block_history.append(label)
        blk = self._cfg_builder.new_block(label)
        self._cfg_builder.set_current_block(label)

    def _emit_constant(
        self, value: Any, ty: IRType, loc: Optional[SourceLocation] = None,
    ) -> SSAValue:
        target = self._new_ssa("_const")
        self._emit(ConstantNode(
            target=target, value=value, type_annotation=ty,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        return target

    def _resolve_or_create(self, name: str) -> SSAValue:
        existing = self._scope.resolve(name)
        if existing is not None:
            return existing
        val = self._new_ssa(name)
        self._scope.declare(name, val, "const")
        return val

    # -- Module-level compilation -------------------------------------------

    def build_module(self, root: TSASTNode) -> IRModule:
        """Compile a full ``SourceFile`` AST into an ``IRModule``."""
        for child in root.children:
            self._compile_top_level(child)

        # Close the entry block
        if self._cfg_builder.current_block and self._cfg_builder.current_block.terminator is None:
            self._terminate(ReturnNode(location=SourceLocation(file=self._source_file)))

        cfg = self._cfg_builder.finalize()

        # Create a top-level function wrapping module statements
        top_func = IRFunction(
            name="<module>",
            body=cfg,
            language_tag=LanguageTag.TypeScript,
            location=SourceLocation(file=self._source_file),
        )
        self._module.add_function(top_func)

        return self._module

    def _compile_top_level(self, node: TSASTNode) -> None:
        if node.kind == TSNodeKind.FunctionDeclaration:
            self._compile_function_declaration(node, top_level=True)
        elif node.kind == TSNodeKind.ClassDeclaration:
            self._compile_class_declaration(node)
        elif node.kind == TSNodeKind.InterfaceDeclaration:
            self._compile_interface_declaration(node)
        elif node.kind == TSNodeKind.TypeAliasDeclaration:
            pass  # Type aliases have no runtime effect
        elif node.kind == TSNodeKind.EnumDeclaration:
            self._desugarer.desugar_enum(node)
        elif node.kind == TSNodeKind.ImportDeclaration:
            self._compile_import(node)
        elif node.kind in (TSNodeKind.ExportDeclaration, TSNodeKind.ExportAssignment):
            self._compile_export(node)
        elif node.kind == TSNodeKind.NamespaceDeclaration:
            self._compile_namespace(node)
        else:
            self._compile_statement(node)

    # -- Statement compilation -----------------------------------------------

    def _compile_statement(self, node: TSASTNode) -> None:
        kind = node.kind
        if kind == TSNodeKind.VariableStatement:
            self._compile_variable_statement(node)
        elif kind in (TSNodeKind.VariableDeclaration, TSNodeKind.VariableDeclarationList):
            self._compile_variable_declaration(node)
        elif kind == TSNodeKind.ExpressionStatement:
            if node.children:
                self.compile_expression(node.children[0])
        elif kind == TSNodeKind.IfStatement:
            self._compile_if(node)
        elif kind == TSNodeKind.ForStatement:
            self._compile_for(node)
        elif kind == TSNodeKind.ForOfStatement:
            self._compile_for_of(node)
        elif kind == TSNodeKind.ForInStatement:
            self._compile_for_in(node)
        elif kind == TSNodeKind.WhileStatement:
            self._compile_while(node)
        elif kind == TSNodeKind.DoStatement:
            self._compile_do_while(node)
        elif kind == TSNodeKind.SwitchStatement:
            self._compile_switch(node)
        elif kind == TSNodeKind.ReturnStatement:
            self._compile_return(node)
        elif kind == TSNodeKind.ThrowStatement:
            self._compile_throw(node)
        elif kind == TSNodeKind.TryStatement:
            self._compile_try(node)
        elif kind == TSNodeKind.Block:
            self._compile_block(node)
        elif kind == TSNodeKind.BreakStatement:
            self._compile_break(node)
        elif kind == TSNodeKind.ContinueStatement:
            self._compile_continue(node)
        elif kind == TSNodeKind.LabeledStatement:
            self._compile_labeled(node)
        elif kind == TSNodeKind.FunctionDeclaration:
            self._compile_function_declaration(node, top_level=False)
        elif kind == TSNodeKind.ClassDeclaration:
            self._compile_class_declaration(node)
        elif kind == TSNodeKind.EnumDeclaration:
            self._desugarer.desugar_enum(node)
        elif kind in (TSNodeKind.EmptyStatement, TSNodeKind.DebuggerStatement):
            pass
        elif kind == TSNodeKind.WithStatement:
            # 'with' is deprecated; compile the body only
            body = node.children[-1] if node.children else None
            if body:
                self._compile_statement(body)

    def _compile_variable_statement(self, node: TSASTNode) -> None:
        decl_list = node.first_child_of_kind(TSNodeKind.VariableDeclarationList)
        if decl_list is not None:
            self._compile_variable_declaration(decl_list)
        else:
            for child in node.children:
                if child.kind == TSNodeKind.VariableDeclaration:
                    self._compile_single_var_decl(child, "let")

    def _compile_variable_declaration(self, node: TSASTNode) -> None:
        decl_kind = "let"
        flags_text = node.properties.get("declarationKind", "")
        if "const" in str(flags_text):
            decl_kind = "const"
        elif "var" in str(flags_text):
            decl_kind = "var"
        # Also check modifiers
        if "const" in node.modifiers:
            decl_kind = "const"
        elif "var" in node.modifiers:
            decl_kind = "var"

        decls = node.children_of_kind(TSNodeKind.VariableDeclaration)
        if not decls:
            decls = [node] if node.kind == TSNodeKind.VariableDeclaration else []
        for decl in decls:
            self._compile_single_var_decl(decl, decl_kind)

    def _compile_single_var_decl(self, decl: TSASTNode, decl_kind: str) -> None:
        loc = decl.source_location
        var_name = decl.name or decl.text

        # Check for binding patterns (destructuring)
        binding = decl.first_child_of_kind(TSNodeKind.ObjectBindingPattern)
        if binding is not None:
            initializer = decl.child_by_field("initializer")
            init_val: SSAValue
            if initializer is not None:
                init_val = self.compile_expression(initializer)
            else:
                init_val = self._emit_constant(None, NONE_TYPE, loc)
            self._desugarer.desugar_object_destructuring(init_val, binding, decl_kind, loc)
            return

        array_binding = decl.first_child_of_kind(TSNodeKind.ArrayBindingPattern)
        if array_binding is not None:
            initializer = decl.child_by_field("initializer")
            if initializer is not None:
                init_val = self.compile_expression(initializer)
            else:
                init_val = self._emit_constant(None, NONE_TYPE, loc)
            self._desugarer.desugar_array_destructuring(init_val, array_binding, decl_kind, loc)
            return

        if not var_name:
            return

        # Type annotation
        type_ann: Optional[IRType] = None
        type_node = decl.child_by_field("type")
        if type_node is not None:
            type_ann = self._type_extractor.extract(type_node)
        elif decl.type_info:
            type_ann = self._type_extractor.extract_from_string(decl.type_info)

        target = self._new_ssa(var_name)
        if type_ann:
            target.type_annotation = type_ann

        initializer = decl.child_by_field("initializer")
        if initializer is None and decl.children:
            # The last child is often the initializer
            for c in decl.children:
                if c.kind not in (
                    TSNodeKind.Identifier,
                    TSNodeKind.TypeReference,
                    TSNodeKind.ArrayType,
                    TSNodeKind.UnionType,
                ):
                    initializer = c
                    break

        if initializer is not None:
            init_val = self.compile_expression(initializer)
            self._emit(AssignNode(
                target=target, value=init_val,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
        else:
            self._emit(ConstantNode(
                target=target, value=None,
                type_annotation=type_ann or UNKNOWN_TYPE,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))

        self._scope.declare(var_name, target, decl_kind)

    # -- If/else -------------------------------------------------------------

    def _compile_if(self, node: TSASTNode) -> None:
        loc = node.source_location
        cond_node = node.child_by_field("condition")
        then_node = node.child_by_field("thenStatement")
        else_node = node.child_by_field("elseStatement")

        # Fallback: parse children positionally
        children = node.children
        if cond_node is None and len(children) >= 2:
            cond_node = children[0]
            then_node = children[1]
            if len(children) >= 3:
                else_node = children[2]

        if cond_node is None:
            return

        cond_val = self.compile_expression(cond_node)
        truth_val = self._new_ssa("_if_cond")
        self._emit(TruthinessCoercionNode(
            target=truth_val, operand=cond_val,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))

        then_lbl = self._fresh_label("if_then")
        merge_lbl = self._fresh_label("if_merge")
        else_lbl = self._fresh_label("if_else") if else_node else merge_lbl
        origin = self._current_label()

        self._terminate(ConditionalBranchNode(
            condition=truth_val, true_label=then_lbl,
            false_label=else_lbl, location=loc,
        ))

        self._cfg_builder.connect(origin, then_lbl, EdgeKind.TrueGuard)
        self._cfg_builder.connect(origin, else_lbl, EdgeKind.FalseGuard)

        # then branch
        self._scope.push_scope(_ScopeKind.Block)
        self._start_block(then_lbl)
        if then_node:
            self._compile_statement(then_node)
        if self._cfg_builder.current_block and self._cfg_builder.current_block.terminator is None:
            self._terminate(BranchNode(target_label=merge_lbl, location=loc))
            self._cfg_builder.connect(then_lbl, merge_lbl)
        self._scope.pop_scope()

        # else branch
        if else_node:
            self._scope.push_scope(_ScopeKind.Block)
            self._start_block(else_lbl)
            self._compile_statement(else_node)
            if self._cfg_builder.current_block and self._cfg_builder.current_block.terminator is None:
                self._terminate(BranchNode(target_label=merge_lbl, location=loc))
                self._cfg_builder.connect(else_lbl, merge_lbl)
            self._scope.pop_scope()

        self._start_block(merge_lbl)

    # -- For loop ------------------------------------------------------------

    def _compile_for(self, node: TSASTNode) -> None:
        loc = node.source_location
        init_node = node.child_by_field("initializer")
        cond_node = node.child_by_field("condition")
        update_node = node.child_by_field("incrementor")
        body_node = node.child_by_field("statement")

        # Positional fallback
        parts = node.children
        if init_node is None and len(parts) >= 1:
            init_node = parts[0] if parts[0].kind != TSNodeKind.EmptyStatement else None
        if cond_node is None and len(parts) >= 2:
            cond_node = parts[1] if parts[1].kind != TSNodeKind.EmptyStatement else None
        if update_node is None and len(parts) >= 3:
            update_node = parts[2] if parts[2].kind != TSNodeKind.EmptyStatement else None
        if body_node is None and len(parts) >= 4:
            body_node = parts[3]

        self._scope.push_scope(_ScopeKind.Block)

        if init_node:
            if init_node.kind in (TSNodeKind.VariableDeclarationList, TSNodeKind.VariableDeclaration):
                self._compile_variable_declaration(init_node)
            else:
                self.compile_expression(init_node)

        header_lbl = self._fresh_label("for_header")
        body_lbl = self._fresh_label("for_body")
        update_lbl = self._fresh_label("for_update")
        exit_lbl = self._fresh_label("for_exit")
        origin = self._current_label()

        self._terminate(BranchNode(target_label=header_lbl, location=loc))
        self._cfg_builder.connect(origin, header_lbl)

        # header (condition check)
        self._start_block(header_lbl)
        if cond_node:
            cond_val = self.compile_expression(cond_node)
            truth = self._new_ssa("_for_cond")
            self._emit(TruthinessCoercionNode(
                target=truth, operand=cond_val,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            self._terminate(ConditionalBranchNode(
                condition=truth, true_label=body_lbl,
                false_label=exit_lbl, location=loc,
            ))
        else:
            self._terminate(BranchNode(target_label=body_lbl, location=loc))
        self._cfg_builder.connect(header_lbl, body_lbl, EdgeKind.TrueGuard)
        self._cfg_builder.connect(header_lbl, exit_lbl, EdgeKind.FalseGuard)

        # body
        self._break_targets.append(exit_lbl)
        self._continue_targets.append(update_lbl)
        self._start_block(body_lbl)
        if body_node:
            self._compile_statement(body_node)
        if self._cfg_builder.current_block and self._cfg_builder.current_block.terminator is None:
            self._terminate(BranchNode(target_label=update_lbl, location=loc))
            self._cfg_builder.connect(body_lbl, update_lbl)
        self._continue_targets.pop()
        self._break_targets.pop()

        # update
        self._start_block(update_lbl)
        if update_node:
            self.compile_expression(update_node)
        self._terminate(BranchNode(target_label=header_lbl, location=loc))
        self._cfg_builder.connect(update_lbl, header_lbl)

        self._start_block(exit_lbl)
        self._scope.pop_scope()

    # -- For-of / For-in -----------------------------------------------------

    def _compile_for_of(self, node: TSASTNode) -> None:
        self._compile_for_iter(node, "for_of", is_for_in=False)

    def _compile_for_in(self, node: TSASTNode) -> None:
        self._compile_for_iter(node, "for_in", is_for_in=True)

    def _compile_for_iter(self, node: TSASTNode, prefix: str, is_for_in: bool) -> None:
        loc = node.source_location
        children = node.children
        # Expect: [initializer, expression, body]
        init_node = children[0] if len(children) > 0 else None
        iter_expr = children[1] if len(children) > 1 else None
        body_node = children[2] if len(children) > 2 else None

        self._scope.push_scope(_ScopeKind.Block)

        iterable = self.compile_expression(iter_expr) if iter_expr else self._emit_constant([], IRType("Array"), loc)

        # Create iterator
        iter_val = self._new_ssa(f"_{prefix}_iter")
        iter_fn = self._resolve_or_create("Symbol.iterator" if not is_for_in else "Object.keys")
        self._emit(CallNode(
            target=iter_val, callee=iter_fn, args=[iterable],
            location=loc, language_tag=LanguageTag.TypeScript,
        ))

        header_lbl = self._fresh_label(f"{prefix}_header")
        body_lbl = self._fresh_label(f"{prefix}_body")
        exit_lbl = self._fresh_label(f"{prefix}_exit")
        origin = self._current_label()

        self._terminate(BranchNode(target_label=header_lbl, location=loc))
        self._cfg_builder.connect(origin, header_lbl)

        # Header: check if iterator is exhausted
        self._start_block(header_lbl)
        has_next = self._new_ssa(f"_{prefix}_has_next")
        next_fn = self._resolve_or_create("iterator.next")
        self._emit(CallNode(
            target=has_next, callee=next_fn, args=[iter_val],
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        done_val = self._new_ssa(f"_{prefix}_done")
        self._emit(LoadAttrNode(
            target=done_val, obj=has_next, attr="done",
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        not_done = self._new_ssa(f"_{prefix}_not_done")
        self._emit(UnaryOpNode(
            target=not_done, operand=done_val, op=UnaryOp.Not,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        self._terminate(ConditionalBranchNode(
            condition=not_done, true_label=body_lbl,
            false_label=exit_lbl, location=loc,
        ))
        self._cfg_builder.connect(header_lbl, body_lbl, EdgeKind.TrueGuard)
        self._cfg_builder.connect(header_lbl, exit_lbl, EdgeKind.FalseGuard)

        # Body
        self._break_targets.append(exit_lbl)
        self._continue_targets.append(header_lbl)
        self._start_block(body_lbl)

        # Bind the loop variable
        elem_val = self._new_ssa(f"_{prefix}_value")
        self._emit(LoadAttrNode(
            target=elem_val, obj=has_next, attr="value",
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        if init_node:
            var_name = init_node.name or init_node.text or "_iter_var"
            decl_kind = "const" if "const" in init_node.modifiers else "let"
            loop_var = self._new_ssa(var_name)
            self._emit(AssignNode(
                target=loop_var, value=elem_val,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            self._scope.declare(var_name, loop_var, decl_kind)

        if body_node:
            self._compile_statement(body_node)
        if self._cfg_builder.current_block and self._cfg_builder.current_block.terminator is None:
            self._terminate(BranchNode(target_label=header_lbl, location=loc))
            self._cfg_builder.connect(body_lbl, header_lbl)

        self._continue_targets.pop()
        self._break_targets.pop()
        self._start_block(exit_lbl)
        self._scope.pop_scope()

    # -- While / Do-while ----------------------------------------------------

    def _compile_while(self, node: TSASTNode) -> None:
        loc = node.source_location
        children = node.children
        cond_node = children[0] if children else None
        body_node = children[1] if len(children) > 1 else None

        header_lbl = self._fresh_label("while_header")
        body_lbl = self._fresh_label("while_body")
        exit_lbl = self._fresh_label("while_exit")
        origin = self._current_label()

        self._terminate(BranchNode(target_label=header_lbl, location=loc))
        self._cfg_builder.connect(origin, header_lbl)

        self._start_block(header_lbl)
        if cond_node:
            cond_val = self.compile_expression(cond_node)
            truth = self._new_ssa("_while_cond")
            self._emit(TruthinessCoercionNode(
                target=truth, operand=cond_val,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            self._terminate(ConditionalBranchNode(
                condition=truth, true_label=body_lbl,
                false_label=exit_lbl, location=loc,
            ))
        else:
            self._terminate(BranchNode(target_label=body_lbl, location=loc))
        self._cfg_builder.connect(header_lbl, body_lbl, EdgeKind.TrueGuard)
        self._cfg_builder.connect(header_lbl, exit_lbl, EdgeKind.FalseGuard)

        self._break_targets.append(exit_lbl)
        self._continue_targets.append(header_lbl)
        self._scope.push_scope(_ScopeKind.Block)
        self._start_block(body_lbl)
        if body_node:
            self._compile_statement(body_node)
        if self._cfg_builder.current_block and self._cfg_builder.current_block.terminator is None:
            self._terminate(BranchNode(target_label=header_lbl, location=loc))
            self._cfg_builder.connect(body_lbl, header_lbl)
        self._scope.pop_scope()
        self._continue_targets.pop()
        self._break_targets.pop()

        self._start_block(exit_lbl)

    def _compile_do_while(self, node: TSASTNode) -> None:
        loc = node.source_location
        children = node.children
        body_node = children[0] if children else None
        cond_node = children[1] if len(children) > 1 else None

        body_lbl = self._fresh_label("do_body")
        cond_lbl = self._fresh_label("do_cond")
        exit_lbl = self._fresh_label("do_exit")
        origin = self._current_label()

        self._terminate(BranchNode(target_label=body_lbl, location=loc))
        self._cfg_builder.connect(origin, body_lbl)

        self._break_targets.append(exit_lbl)
        self._continue_targets.append(cond_lbl)
        self._scope.push_scope(_ScopeKind.Block)
        self._start_block(body_lbl)
        if body_node:
            self._compile_statement(body_node)
        if self._cfg_builder.current_block and self._cfg_builder.current_block.terminator is None:
            self._terminate(BranchNode(target_label=cond_lbl, location=loc))
            self._cfg_builder.connect(body_lbl, cond_lbl)
        self._scope.pop_scope()
        self._continue_targets.pop()
        self._break_targets.pop()

        self._start_block(cond_lbl)
        if cond_node:
            cond_val = self.compile_expression(cond_node)
            truth = self._new_ssa("_do_cond")
            self._emit(TruthinessCoercionNode(
                target=truth, operand=cond_val,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            self._terminate(ConditionalBranchNode(
                condition=truth, true_label=body_lbl,
                false_label=exit_lbl, location=loc,
            ))
        else:
            self._terminate(BranchNode(target_label=body_lbl, location=loc))
        self._cfg_builder.connect(cond_lbl, body_lbl, EdgeKind.TrueGuard)
        self._cfg_builder.connect(cond_lbl, exit_lbl, EdgeKind.FalseGuard)

        self._start_block(exit_lbl)

    # -- Switch --------------------------------------------------------------

    def _compile_switch(self, node: TSASTNode) -> None:
        loc = node.source_location
        children = node.children
        discriminant = children[0] if children else None
        cases = [c for c in children if c.kind in (TSNodeKind.CaseClause, TSNodeKind.DefaultClause)]

        disc_val = self.compile_expression(discriminant) if discriminant else self._emit_constant(None, NONE_TYPE, loc)

        exit_lbl = self._fresh_label("switch_exit")
        self._switch_break_targets.append(exit_lbl)
        self._break_targets.append(exit_lbl)

        case_labels: List[str] = []
        default_label: Optional[str] = None

        for i, case in enumerate(cases):
            lbl = self._fresh_label(f"case_{i}")
            case_labels.append(lbl)
            if case.kind == TSNodeKind.DefaultClause:
                default_label = lbl

        if default_label is None:
            default_label = exit_lbl

        # Emit cascading condition checks
        for i, case in enumerate(cases):
            if case.kind == TSNodeKind.DefaultClause:
                continue
            test_expr = case.children[0] if case.children else None
            if test_expr is None:
                continue
            test_val = self.compile_expression(test_expr)
            cmp_result = self._new_ssa(f"_case_cmp_{i}")
            self._emit(CompareNode(
                target=cmp_result, left=disc_val, right=test_val,
                op=CompareOp.Eq, location=loc,
                language_tag=LanguageTag.TypeScript,
            ))
            next_check = self._fresh_label(f"case_check_{i}")
            self._terminate(ConditionalBranchNode(
                condition=cmp_result, true_label=case_labels[i],
                false_label=next_check, location=loc,
            ))
            origin = self._current_label()
            self._cfg_builder.connect(origin, case_labels[i], EdgeKind.TrueGuard)
            self._cfg_builder.connect(origin, next_check, EdgeKind.FalseGuard)
            self._start_block(next_check)

        # Fall through to default
        self._terminate(BranchNode(target_label=default_label, location=loc))
        self._cfg_builder.connect(self._current_label(), default_label)

        # Compile case bodies
        for i, case in enumerate(cases):
            self._start_block(case_labels[i])
            body_nodes = case.children[1:] if case.kind == TSNodeKind.CaseClause else case.children
            for stmt in body_nodes:
                self._compile_statement(stmt)
            # Fallthrough to next case if no break
            if self._cfg_builder.current_block and self._cfg_builder.current_block.terminator is None:
                next_lbl = case_labels[i + 1] if i + 1 < len(case_labels) else exit_lbl
                self._terminate(BranchNode(target_label=next_lbl, location=loc))
                self._cfg_builder.connect(case_labels[i], next_lbl)

        self._break_targets.pop()
        self._switch_break_targets.pop()
        self._start_block(exit_lbl)

    # -- Return / Throw / Break / Continue -----------------------------------

    def _compile_return(self, node: TSASTNode) -> None:
        loc = node.source_location
        if node.children:
            val = self.compile_expression(node.children[0])
            self._terminate(ReturnNode(value=val, location=loc, language_tag=LanguageTag.TypeScript))
        else:
            self._terminate(ReturnNode(location=loc, language_tag=LanguageTag.TypeScript))

    def _compile_throw(self, node: TSASTNode) -> None:
        loc = node.source_location
        if node.children:
            val = self.compile_expression(node.children[0])
            self._terminate(RaiseNode(exception=val, location=loc, language_tag=LanguageTag.TypeScript))
        else:
            self._terminate(RaiseNode(location=loc, language_tag=LanguageTag.TypeScript))

    def _compile_break(self, _node: TSASTNode) -> None:
        if self._break_targets:
            target = self._break_targets[-1]
            origin = self._current_label()
            self._terminate(BranchNode(target_label=target, location=_node.source_location))
            self._cfg_builder.connect(origin, target)

    def _compile_continue(self, _node: TSASTNode) -> None:
        if self._continue_targets:
            target = self._continue_targets[-1]
            origin = self._current_label()
            self._terminate(BranchNode(target_label=target, location=_node.source_location))
            self._cfg_builder.connect(origin, target)

    def _compile_labeled(self, node: TSASTNode) -> None:
        if node.children:
            self._compile_statement(node.children[-1])

    # -- Try / Catch / Finally -----------------------------------------------

    def _compile_try(self, node: TSASTNode) -> None:
        loc = node.source_location
        try_body = node.child_by_field("tryBlock")
        catch_clause = node.first_child_of_kind(TSNodeKind.CatchClause)
        finally_block = node.child_by_field("finallyBlock")

        # Positional fallback
        if try_body is None and node.children:
            for c in node.children:
                if c.kind == TSNodeKind.Block and catch_clause is None:
                    try_body = c
                    break
                elif c.kind == TSNodeKind.CatchClause:
                    catch_clause = c
                elif c.kind == TSNodeKind.Block and try_body is not None:
                    finally_block = c

        try_lbl = self._fresh_label("try")
        catch_lbl = self._fresh_label("catch")
        finally_lbl = self._fresh_label("finally") if finally_block else None
        exit_lbl = self._fresh_label("try_exit")
        origin = self._current_label()

        self._terminate(BranchNode(target_label=try_lbl, location=loc))
        self._cfg_builder.connect(origin, try_lbl)

        # Try body
        self._exception_handlers.append(catch_lbl)
        self._start_block(try_lbl)
        if try_body:
            self._compile_statement(try_body)
        after_try_target = finally_lbl or exit_lbl
        if self._cfg_builder.current_block and self._cfg_builder.current_block.terminator is None:
            self._terminate(BranchNode(target_label=after_try_target, location=loc))
            self._cfg_builder.connect(try_lbl, after_try_target)
        self._cfg_builder.connect(try_lbl, catch_lbl, EdgeKind.Exception)
        self._exception_handlers.pop()

        # Catch
        self._start_block(catch_lbl)
        if catch_clause:
            self._scope.push_scope(_ScopeKind.Block)
            # Bind catch parameter
            catch_param = catch_clause.child_by_field("variableDeclaration")
            if catch_param is None and catch_clause.children:
                for c in catch_clause.children:
                    if c.kind == TSNodeKind.VariableDeclaration:
                        catch_param = c
                        break
            if catch_param:
                err_name = catch_param.name or catch_param.text or "error"
                err_val = self._new_ssa(err_name)
                err_val.type_annotation = ANY_TYPE
                self._emit(ConstantNode(
                    target=err_val, value="<caught_exception>",
                    type_annotation=ANY_TYPE,
                    location=loc, language_tag=LanguageTag.TypeScript,
                ))
                self._scope.declare(err_name, err_val, "let")

            catch_body = catch_clause.first_child_of_kind(TSNodeKind.Block)
            if catch_body is None and catch_clause.children:
                catch_body = catch_clause.children[-1]
            if catch_body:
                self._compile_statement(catch_body)
            self._scope.pop_scope()

        after_catch_target = finally_lbl or exit_lbl
        if self._cfg_builder.current_block and self._cfg_builder.current_block.terminator is None:
            self._terminate(BranchNode(target_label=after_catch_target, location=loc))
            self._cfg_builder.connect(catch_lbl, after_catch_target)

        # Finally
        if finally_lbl and finally_block:
            self._start_block(finally_lbl)
            self._compile_statement(finally_block)
            if self._cfg_builder.current_block and self._cfg_builder.current_block.terminator is None:
                self._terminate(BranchNode(target_label=exit_lbl, location=loc))
                self._cfg_builder.connect(finally_lbl, exit_lbl)

        self._start_block(exit_lbl)

    # -- Block ---------------------------------------------------------------

    def _compile_block(self, node: TSASTNode) -> None:
        self._scope.push_scope(_ScopeKind.Block)
        for child in node.children:
            self._compile_statement(child)
        self._scope.pop_scope()

    # -- Function declaration ------------------------------------------------

    def _compile_function_declaration(self, node: TSASTNode, top_level: bool = False) -> None:
        loc = node.source_location
        func_name = node.name or node.text or "<anonymous>"

        params: List[SSAVariable] = []
        param_nodes = node.children_of_kind(TSNodeKind.Parameter)
        for p in param_nodes:
            p_name = p.name or p.text or "_param"
            p_type: Optional[IRType] = None
            type_child = p.child_by_field("type")
            if type_child:
                p_type = self._type_extractor.extract(type_child)
            elif p.type_info:
                p_type = self._type_extractor.extract_from_string(p.type_info)
            params.append(SSAVariable(
                ssa_value=SSAValue(p_name, version=0, type_annotation=p_type),
                is_parameter=True,
            ))

        ret_type: Optional[IRType] = None
        ret_node = node.child_by_field("returnType") or node.child_by_field("type")
        if ret_node:
            ret_type = self._type_extractor.extract(ret_node)

        is_async = "async" in node.modifiers
        is_generator = "*" in (node.properties.get("asteriskToken", "") or "")

        type_params: List[str] = []
        for tp in node.children_of_kind(TSNodeKind.TypeParameter):
            tp_name = tp.name or tp.text
            if tp_name:
                type_params.append(tp_name)

        # Compile body in a new builder
        body_builder = TSSSABuilder(source_file=self._source_file)
        body_builder._scope.push_scope(_ScopeKind.Function)
        for param in params:
            body_builder._scope.declare(param.name, param.ssa_value, "const")

        body_ast = node.first_child_of_kind(TSNodeKind.Block)
        if body_ast:
            for stmt in body_ast.children:
                body_builder._compile_statement(stmt)

        if body_builder._cfg_builder.current_block and body_builder._cfg_builder.current_block.terminator is None:
            undef = body_builder._emit_constant(None, NONE_TYPE, loc)
            body_builder._terminate(ReturnNode(value=undef, location=loc, language_tag=LanguageTag.TypeScript))

        body_builder._scope.pop_scope()
        cfg = body_builder._cfg_builder.finalize()

        ir_func = IRFunction(
            name=func_name,
            params=params,
            return_type=ret_type,
            body=cfg,
            is_async=is_async,
            is_generator=is_generator,
            type_params=type_params,
            location=loc,
            language_tag=LanguageTag.TypeScript,
        )
        self._module.add_function(ir_func)

        # Create a binding in the current scope
        func_val = self._new_ssa(func_name)
        self._emit(ConstantNode(
            target=func_val, value=func_name,
            type_annotation=IRType("Function"),
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        self._scope.declare(func_name, func_val, "const")

    # -- Class declaration ---------------------------------------------------

    def _compile_class_declaration(self, node: TSASTNode) -> None:
        loc = node.source_location
        class_name = node.name or node.text or "<anonymous_class>"

        bases: List[str] = []
        heritage = node.properties.get("heritageClauses", [])
        if isinstance(heritage, list):
            for h in heritage:
                if isinstance(h, dict):
                    for t in h.get("types", []):
                        if isinstance(t, dict):
                            bases.append(t.get("expression", {}).get("text", ""))
                elif isinstance(h, str):
                    bases.append(h)

        ir_class = IRClass(
            name=class_name,
            bases=bases,
            location=loc,
            language_tag=LanguageTag.TypeScript,
        )

        type_params: List[str] = []
        for tp in node.children_of_kind(TSNodeKind.TypeParameter):
            tp_name = tp.name or tp.text
            if tp_name:
                type_params.append(tp_name)
        ir_class.type_params = type_params

        # Process members
        for member in node.children:
            if member.kind == TSNodeKind.MethodDeclaration:
                method_func = self._compile_method(member, class_name)
                if method_func:
                    if "static" in member.modifiers:
                        ir_class.static_methods[method_func.name] = method_func
                    else:
                        ir_class.methods[method_func.name] = method_func
            elif member.kind == TSNodeKind.Constructor:
                ctor_func = self._compile_method(member, class_name)
                if ctor_func:
                    ctor_func.name = "constructor"
                    ir_class.methods["constructor"] = ctor_func
            elif member.kind == TSNodeKind.PropertyDeclaration:
                prop_name = member.name or member.text or "_prop"
                prop_type = UNKNOWN_TYPE
                type_child = member.child_by_field("type")
                if type_child:
                    prop_type = self._type_extractor.extract(type_child) or UNKNOWN_TYPE
                elif member.type_info:
                    prop_type = self._type_extractor.extract_from_string(member.type_info)
                ir_class.fields[prop_name] = prop_type
            elif member.kind in (TSNodeKind.GetAccessor, TSNodeKind.SetAccessor):
                accessor_func = self._compile_method(member, class_name)
                if accessor_func:
                    ir_class.properties[accessor_func.name] = accessor_func

        self._module.add_class(ir_class)

        class_val = self._new_ssa(class_name)
        self._emit(ConstantNode(
            target=class_val, value=class_name,
            type_annotation=IRType(class_name),
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        self._scope.declare(class_name, class_val, "const")

    def _compile_method(self, node: TSASTNode, class_name: str) -> Optional[IRFunction]:
        method_name = node.name or node.text or "<method>"
        loc = node.source_location

        params: List[SSAVariable] = []
        # Implicit 'this' parameter
        params.append(SSAVariable(
            ssa_value=SSAValue("this", version=0, type_annotation=IRType(class_name)),
            is_parameter=True,
        ))
        for p in node.children_of_kind(TSNodeKind.Parameter):
            p_name = p.name or p.text or "_param"
            p_type: Optional[IRType] = None
            type_child = p.child_by_field("type")
            if type_child:
                p_type = self._type_extractor.extract(type_child)
            elif p.type_info:
                p_type = self._type_extractor.extract_from_string(p.type_info)
            params.append(SSAVariable(
                ssa_value=SSAValue(p_name, version=0, type_annotation=p_type),
                is_parameter=True,
            ))

        ret_type: Optional[IRType] = None
        ret_node = node.child_by_field("returnType")
        if ret_node:
            ret_type = self._type_extractor.extract(ret_node)

        body_builder = TSSSABuilder(source_file=self._source_file)
        body_builder._scope.push_scope(_ScopeKind.Function)
        for param in params:
            body_builder._scope.declare(param.name, param.ssa_value, "const")

        body_ast = node.first_child_of_kind(TSNodeKind.Block)
        if body_ast:
            for stmt in body_ast.children:
                body_builder._compile_statement(stmt)

        if body_builder._cfg_builder.current_block and body_builder._cfg_builder.current_block.terminator is None:
            undef = body_builder._emit_constant(None, NONE_TYPE, loc)
            body_builder._terminate(ReturnNode(value=undef, location=loc, language_tag=LanguageTag.TypeScript))

        body_builder._scope.pop_scope()
        cfg = body_builder._cfg_builder.finalize()

        return IRFunction(
            name=method_name,
            params=params,
            return_type=ret_type,
            body=cfg,
            is_async="async" in node.modifiers,
            location=loc,
            language_tag=LanguageTag.TypeScript,
        )

    # -- Interface / Namespace / Import / Export ------------------------------

    def _compile_interface_declaration(self, node: TSASTNode) -> None:
        """Interfaces have no runtime representation but we record them as empty classes."""
        iface_name = node.name or node.text or "<interface>"
        ir_class = IRClass(
            name=iface_name,
            is_abstract=True,
            location=node.source_location,
            language_tag=LanguageTag.TypeScript,
        )
        for member in node.children:
            if member.kind == TSNodeKind.PropertyDeclaration:
                prop_name = member.name or member.text or "_prop"
                prop_type = UNKNOWN_TYPE
                type_child = member.child_by_field("type")
                if type_child:
                    prop_type = self._type_extractor.extract(type_child) or UNKNOWN_TYPE
                ir_class.fields[prop_name] = prop_type
        self._module.add_class(ir_class)

    def _compile_namespace(self, node: TSASTNode) -> None:
        body = node.first_child_of_kind(TSNodeKind.Block)
        if body:
            self._scope.push_scope(_ScopeKind.Block, label=node.name)
            for child in body.children:
                self._compile_top_level(child)
            self._scope.pop_scope()

    def _compile_import(self, node: TSASTNode) -> None:
        loc = node.source_location
        module_name = ""
        for c in node.children:
            if c.kind == TSNodeKind.StringLiteral:
                module_name = c.text.strip("'\"")
                break

        import_target = self._new_ssa(f"_import_{module_name}")
        self._emit(ImportNode(
            target=import_target, module_name=module_name,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        self._module.add_import(ImportNode(
            target=import_target, module_name=module_name,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))

        # Bind named imports
        for c in node.children:
            if c.kind == TSNodeKind.Identifier:
                self._scope.declare(c.text, import_target, "const")

    def _compile_export(self, node: TSASTNode) -> None:
        for child in node.children:
            self._compile_top_level(child)

    # -- Expression compilation -----------------------------------------------

    def compile_expression(self, node: TSASTNode) -> SSAValue:
        """Compile an expression AST node, returning the SSA value it produces."""
        kind = node.kind
        loc = node.source_location

        if kind == TSNodeKind.Identifier:
            return self._compile_identifier(node)
        if kind == TSNodeKind.NumericLiteral:
            return self._compile_numeric_literal(node)
        if kind in (TSNodeKind.StringLiteral, TSNodeKind.NoSubstitutionTemplateLiteral):
            return self._compile_string_literal(node)
        if kind == TSNodeKind.TrueKeyword:
            return self._emit_constant(True, BOOL_TYPE, loc)
        if kind == TSNodeKind.FalseKeyword:
            return self._emit_constant(False, BOOL_TYPE, loc)
        if kind in (TSNodeKind.NullKeyword, TSNodeKind.UndefinedKeyword):
            return self._emit_constant(None, NONE_TYPE, loc)
        if kind == TSNodeKind.ThisKeyword:
            return self._compile_identifier(TSASTNode(kind=TSNodeKind.Identifier, text="this"))
        if kind == TSNodeKind.SuperKeyword:
            return self._compile_identifier(TSASTNode(kind=TSNodeKind.Identifier, text="super"))
        if kind == TSNodeKind.BinaryExpression:
            return self._compile_binary_expression(node)
        if kind == TSNodeKind.PrefixUnaryExpression:
            return self._compile_prefix_unary(node)
        if kind == TSNodeKind.PostfixUnaryExpression:
            return self._compile_postfix_unary(node)
        if kind == TSNodeKind.CallExpression:
            return self._compile_call(node)
        if kind == TSNodeKind.NewExpression:
            return self._compile_new(node)
        if kind == TSNodeKind.PropertyAccessExpression:
            return self._compile_property_access(node)
        if kind == TSNodeKind.ElementAccessExpression:
            return self._compile_element_access(node)
        if kind == TSNodeKind.ConditionalExpression:
            return self._compile_conditional(node)
        if kind in (TSNodeKind.ArrowFunction, TSNodeKind.FunctionExpression):
            return self._compile_arrow_or_func_expr(node)
        if kind in (TSNodeKind.TemplateExpression, TSNodeKind.TemplateLiteral):
            return self._compile_template_literal(node)
        if kind == TSNodeKind.TypeOfExpression:
            return self._compile_typeof(node)
        if kind == TSNodeKind.VoidExpression:
            if node.children:
                self.compile_expression(node.children[0])
            return self._emit_constant(None, NONE_TYPE, loc)
        if kind == TSNodeKind.DeleteExpression:
            return self._compile_delete(node)
        if kind in (TSNodeKind.AsExpression, TSNodeKind.TypeAssertionExpression, TSNodeKind.SatisfiesExpression):
            return self._compile_type_assertion(node)
        if kind == TSNodeKind.NonNullExpression:
            return self._compile_non_null_assertion(node)
        if kind in (TSNodeKind.ObjectLiteralExpression, TSNodeKind.ObjectLiteral):
            return self._compile_object_literal(node)
        if kind in (TSNodeKind.ArrayLiteralExpression, TSNodeKind.ArrayLiteral):
            return self._compile_array_literal(node)
        if kind == TSNodeKind.SpreadElement:
            if node.children:
                return self.compile_expression(node.children[0])
            return self._emit_constant(None, NONE_TYPE, loc)
        if kind == TSNodeKind.ParenthesizedExpression:
            if node.children:
                return self.compile_expression(node.children[0])
            return self._emit_constant(None, NONE_TYPE, loc)
        if kind == TSNodeKind.AwaitExpression:
            return self._compile_await(node)
        if kind == TSNodeKind.YieldExpression:
            return self._compile_yield(node)
        if kind == TSNodeKind.ClassExpression:
            self._compile_class_declaration(node)
            return self._emit_constant(None, NONE_TYPE, loc)
        if kind == TSNodeKind.RegularExpressionLiteral:
            return self._emit_constant(node.text, IRType("RegExp"), loc)
        if kind == TSNodeKind.CommaExpression:
            result = self._emit_constant(None, NONE_TYPE, loc)
            for child in node.children:
                result = self.compile_expression(child)
            return result

        # Fallback: try to compile children and return last result
        if node.children:
            result = self._emit_constant(None, UNKNOWN_TYPE, loc)
            for child in node.children:
                result = self.compile_expression(child)
            return result

        return self._emit_constant(None, UNKNOWN_TYPE, loc)

    # -- Identifier ----------------------------------------------------------

    def _compile_identifier(self, node: TSASTNode) -> SSAValue:
        name = node.name or node.text
        if not name:
            return self._emit_constant(None, UNKNOWN_TYPE, node.source_location)
        existing = self._scope.resolve(name)
        if existing is not None:
            return existing
        # Create an unresolved reference
        val = self._new_ssa(name)
        self._emit(ConstantNode(
            target=val, value=name,
            type_annotation=UNKNOWN_TYPE,
            location=node.source_location,
            language_tag=LanguageTag.TypeScript,
        ))
        self._scope.declare(name, val, "let")
        return val

    # -- Literals ------------------------------------------------------------

    def _compile_numeric_literal(self, node: TSASTNode) -> SSAValue:
        text = node.text
        try:
            if "." in text or "e" in text.lower():
                value = float(text)
            else:
                value = int(text, 0)
        except (ValueError, TypeError):
            value = 0
        return self._emit_constant(value, FLOAT_TYPE, node.source_location)

    def _compile_string_literal(self, node: TSASTNode) -> SSAValue:
        text = node.text
        if text and text[0] in ("'", '"', "`"):
            text = text[1:]
        if text and text[-1] in ("'", '"', "`"):
            text = text[:-1]
        return self._emit_constant(text, STR_TYPE, node.source_location)

    # -- Binary expression ---------------------------------------------------

    def _compile_binary_expression(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        children = node.children
        op_str = node.operator_token or ""

        # Get operands from children or properties
        left_node = node.child_by_field("left")
        right_node = node.child_by_field("right")
        if left_node is None and len(children) >= 2:
            left_node = children[0]
            if len(children) >= 3:
                op_str = op_str or children[1].text
                right_node = children[2]
            elif len(children) == 2:
                right_node = children[1]

        if not op_str:
            op_str = node.properties.get("operatorToken", "")

        if left_node is None or right_node is None:
            return self._emit_constant(None, UNKNOWN_TYPE, loc)

        # Short-circuit: &&, ||
        if op_str == "&&":
            return self._compile_logical_and(left_node, right_node, loc)
        if op_str == "||":
            return self._compile_logical_or(left_node, right_node, loc)
        # Nullish coalescing
        if op_str == "??":
            left_val = self.compile_expression(left_node)
            return self._desugarer.desugar_nullish_coalescing(left_val, right_node, loc)

        # Assignment operators
        if op_str == "=":
            return self._compile_assignment(left_node, right_node, loc)
        if op_str in ("+=", "-=", "*=", "/=", "%=", "**=", "&=", "|=", "^=", "<<=", ">>="):
            return self._compile_compound_assignment(left_node, right_node, op_str, loc)
        if op_str == "??=":
            left_val = self.compile_expression(left_node)
            result = self._desugarer.desugar_nullish_coalescing(left_val, right_node, loc)
            return self._compile_assignment(left_node, TSASTNode(kind=TSNodeKind.Unknown), loc, override_val=result)
        if op_str in ("&&=", "||="):
            left_val = self.compile_expression(left_node)
            if op_str == "&&=":
                result = self._compile_logical_and(left_node, right_node, loc)
            else:
                result = self._compile_logical_or(left_node, right_node, loc)
            return self._compile_assignment(left_node, TSASTNode(kind=TSNodeKind.Unknown), loc, override_val=result)

        left_val = self.compile_expression(left_node)
        right_val = self.compile_expression(right_node)

        # Comparison operators
        if op_str in _TS_COMPARE_MAP:
            result = self._new_ssa("_cmp")
            cmp_op = _TS_COMPARE_MAP[op_str]
            self._emit(CompareNode(
                target=result, left=left_val, right=right_val,
                op=cmp_op, location=loc,
                language_tag=LanguageTag.TypeScript,
                metadata={"strict": op_str in ("===", "!==")},
            ))
            return result

        # Arithmetic / bitwise operators
        if op_str in _TS_BINOP_MAP:
            result = self._new_ssa("_binop")
            self._emit(BinOpNode(
                target=result, left=left_val, right=right_val,
                op=_TS_BINOP_MAP[op_str], location=loc,
                language_tag=LanguageTag.TypeScript,
            ))
            return result

        # instanceof
        if op_str == "instanceof":
            result = self._new_ssa("_instanceof")
            self._emit(CompareNode(
                target=result, left=left_val, right=right_val,
                op=CompareOp.Instanceof, location=loc,
                language_tag=LanguageTag.TypeScript,
            ))
            return result

        # in
        if op_str == "in":
            result = self._new_ssa("_in")
            self._emit(CompareNode(
                target=result, left=left_val, right=right_val,
                op=CompareOp.In, location=loc,
                language_tag=LanguageTag.TypeScript,
            ))
            return result

        # Fallback: treat as generic binary op
        result = self._new_ssa("_binop")
        self._emit(BinOpNode(
            target=result, left=left_val, right=right_val,
            op=BinOp.Add, location=loc,
            language_tag=LanguageTag.TypeScript,
        ))
        return result

    def _compile_assignment(
        self,
        left: TSASTNode,
        right: TSASTNode,
        loc: Optional[SourceLocation],
        override_val: Optional[SSAValue] = None,
    ) -> SSAValue:
        rhs = override_val if override_val is not None else self.compile_expression(right)

        if left.kind == TSNodeKind.Identifier:
            name = left.name or left.text or "_tmp"
            new_val = self._new_ssa(name)
            self._emit(AssignNode(
                target=new_val, value=rhs,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            self._scope.update(name, new_val)
            return new_val
        if left.kind == TSNodeKind.PropertyAccessExpression:
            children = left.children
            if len(children) >= 2:
                obj_val = self.compile_expression(children[0])
                attr = children[-1].text or children[-1].name or ""
                self._emit(StoreAttrNode(
                    obj=obj_val, attr=attr, value=rhs,
                    location=loc, language_tag=LanguageTag.TypeScript,
                ))
                return rhs
        if left.kind == TSNodeKind.ElementAccessExpression:
            children = left.children
            if len(children) >= 2:
                obj_val = self.compile_expression(children[0])
                idx_val = self.compile_expression(children[1])
                self._emit(StoreSubscriptNode(
                    obj=obj_val, index=idx_val, value=rhs,
                    location=loc, language_tag=LanguageTag.TypeScript,
                ))
                return rhs

        return rhs

    def _compile_compound_assignment(
        self,
        left: TSASTNode,
        right: TSASTNode,
        op_str: str,
        loc: Optional[SourceLocation],
    ) -> SSAValue:
        base_op = op_str[:-1]  # e.g., "+=" → "+"
        left_val = self.compile_expression(left)
        right_val = self.compile_expression(right)
        result = self._new_ssa("_compound")
        ir_op = _TS_BINOP_MAP.get(base_op, BinOp.Add)
        self._emit(BinOpNode(
            target=result, left=left_val, right=right_val,
            op=ir_op, location=loc,
            language_tag=LanguageTag.TypeScript,
        ))
        return self._compile_assignment(left, TSASTNode(kind=TSNodeKind.Unknown), loc, override_val=result)

    # -- Short-circuit logic -------------------------------------------------

    def _compile_logical_and(
        self, left: TSASTNode, right: TSASTNode, loc: Optional[SourceLocation],
    ) -> SSAValue:
        left_val = self.compile_expression(left)
        truth = self._new_ssa("_and_truth")
        self._emit(TruthinessCoercionNode(
            target=truth, operand=left_val,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))

        rhs_lbl = self._fresh_label("and_rhs")
        merge_lbl = self._fresh_label("and_merge")
        origin = self._current_label()

        self._terminate(ConditionalBranchNode(
            condition=truth, true_label=rhs_lbl,
            false_label=merge_lbl, location=loc,
        ))
        self._cfg_builder.connect(origin, rhs_lbl, EdgeKind.TrueGuard)
        self._cfg_builder.connect(origin, merge_lbl, EdgeKind.FalseGuard)

        self._start_block(rhs_lbl)
        right_val = self.compile_expression(right)
        self._terminate(BranchNode(target_label=merge_lbl, location=loc))
        self._cfg_builder.connect(rhs_lbl, merge_lbl)

        self._start_block(merge_lbl)
        merged = self._new_ssa("_and_result")
        self._emit(PhiNode(
            target=merged,
            incoming={origin: left_val, rhs_lbl: right_val},
            location=loc,
        ))
        return merged

    def _compile_logical_or(
        self, left: TSASTNode, right: TSASTNode, loc: Optional[SourceLocation],
    ) -> SSAValue:
        left_val = self.compile_expression(left)
        truth = self._new_ssa("_or_truth")
        self._emit(TruthinessCoercionNode(
            target=truth, operand=left_val,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))

        rhs_lbl = self._fresh_label("or_rhs")
        merge_lbl = self._fresh_label("or_merge")
        origin = self._current_label()

        self._terminate(ConditionalBranchNode(
            condition=truth, true_label=merge_lbl,
            false_label=rhs_lbl, location=loc,
        ))
        self._cfg_builder.connect(origin, merge_lbl, EdgeKind.TrueGuard)
        self._cfg_builder.connect(origin, rhs_lbl, EdgeKind.FalseGuard)

        self._start_block(rhs_lbl)
        right_val = self.compile_expression(right)
        self._terminate(BranchNode(target_label=merge_lbl, location=loc))
        self._cfg_builder.connect(rhs_lbl, merge_lbl)

        self._start_block(merge_lbl)
        merged = self._new_ssa("_or_result")
        self._emit(PhiNode(
            target=merged,
            incoming={origin: left_val, rhs_lbl: right_val},
            location=loc,
        ))
        return merged

    # -- Prefix / postfix unary ----------------------------------------------

    def _compile_prefix_unary(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        op_str = node.operator_token or node.properties.get("operator", "")
        operand = node.children[0] if node.children else None

        if operand is None:
            return self._emit_constant(None, UNKNOWN_TYPE, loc)

        if op_str == "!":
            val = self.compile_expression(operand)
            result = self._new_ssa("_not")
            self._emit(UnaryOpNode(
                target=result, operand=val, op=UnaryOp.Not,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            return result
        if op_str == "-":
            val = self.compile_expression(operand)
            result = self._new_ssa("_neg")
            self._emit(UnaryOpNode(
                target=result, operand=val, op=UnaryOp.Neg,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            return result
        if op_str == "+":
            val = self.compile_expression(operand)
            result = self._new_ssa("_pos")
            self._emit(UnaryOpNode(
                target=result, operand=val, op=UnaryOp.Pos,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            return result
        if op_str == "~":
            val = self.compile_expression(operand)
            result = self._new_ssa("_inv")
            self._emit(UnaryOpNode(
                target=result, operand=val, op=UnaryOp.Invert,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            return result
        if op_str == "++":
            val = self.compile_expression(operand)
            one = self._emit_constant(1, FLOAT_TYPE, loc)
            result = self._new_ssa("_preinc")
            self._emit(BinOpNode(
                target=result, left=val, right=one, op=BinOp.Add,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            self._compile_assignment(operand, TSASTNode(kind=TSNodeKind.Unknown), loc, override_val=result)
            return result
        if op_str == "--":
            val = self.compile_expression(operand)
            one = self._emit_constant(1, FLOAT_TYPE, loc)
            result = self._new_ssa("_predec")
            self._emit(BinOpNode(
                target=result, left=val, right=one, op=BinOp.Sub,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            self._compile_assignment(operand, TSASTNode(kind=TSNodeKind.Unknown), loc, override_val=result)
            return result

        val = self.compile_expression(operand)
        return val

    def _compile_postfix_unary(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        op_str = node.operator_token or node.properties.get("operator", "")
        operand = node.children[0] if node.children else None
        if operand is None:
            return self._emit_constant(None, UNKNOWN_TYPE, loc)

        val = self.compile_expression(operand)
        old = self._new_ssa("_postfix_old")
        self._emit(AssignNode(
            target=old, value=val,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        one = self._emit_constant(1, FLOAT_TYPE, loc)
        new_val = self._new_ssa("_postfix_new")
        binop = BinOp.Add if op_str == "++" else BinOp.Sub
        self._emit(BinOpNode(
            target=new_val, left=val, right=one, op=binop,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        self._compile_assignment(operand, TSASTNode(kind=TSNodeKind.Unknown), loc, override_val=new_val)
        return old  # postfix returns old value

    # -- Call / New ----------------------------------------------------------

    def _compile_call(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        children = node.children
        if not children:
            return self._emit_constant(None, UNKNOWN_TYPE, loc)

        callee_node = children[0]
        arg_nodes = children[1:]

        callee_val = self.compile_expression(callee_node)
        args: List[SSAValue] = []
        spread_indices: Set[int] = set()
        for i, arg in enumerate(arg_nodes):
            if arg.kind == TSNodeKind.SpreadElement:
                spread_indices.add(i)
                if arg.children:
                    args.append(self.compile_expression(arg.children[0]))
                else:
                    args.append(self._emit_constant([], IRType("Array"), loc))
            else:
                args.append(self.compile_expression(arg))

        if spread_indices:
            args_val = self._desugarer.desugar_spread_in_call(args, spread_indices, loc)
            result = self._new_ssa("_call")
            # Use apply-style call
            apply_fn = self._resolve_or_create("Function.prototype.apply")
            self._emit(CallNode(
                target=result, callee=callee_val, args=[args_val],
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            return result

        result = self._new_ssa("_call")

        # Check for builtin return type
        callee_name = self._callee_name(callee_node)
        ret_type = TSBuiltinSignatures.return_type(callee_name) if callee_name else None
        if ret_type:
            result.type_annotation = ret_type

        self._emit(CallNode(
            target=result, callee=callee_val, args=args,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        return result

    def _compile_new(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        children = node.children
        if not children:
            return self._emit_constant(None, UNKNOWN_TYPE, loc)

        callee_node = children[0]
        arg_nodes = children[1:]

        callee_val = self.compile_expression(callee_node)
        args = [self.compile_expression(a) for a in arg_nodes]

        result = self._new_ssa("_new")
        self._emit(CallNode(
            target=result, callee=callee_val, args=args,
            is_new=True,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        return result

    def _callee_name(self, node: TSASTNode) -> Optional[str]:
        if node.kind == TSNodeKind.Identifier:
            return node.name or node.text
        if node.kind == TSNodeKind.PropertyAccessExpression and node.children:
            obj_name = self._callee_name(node.children[0])
            prop_name = node.children[-1].text if len(node.children) > 1 else ""
            if obj_name and prop_name:
                return f"{obj_name}.{prop_name}"
        return None

    # -- Property / Element Access -------------------------------------------

    def _compile_property_access(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        children = node.children
        if len(children) < 2:
            return self._emit_constant(None, UNKNOWN_TYPE, loc)

        # Check for optional chaining (obj?.prop)
        is_optional = node.properties.get("questionDotToken") is not None

        obj_val = self.compile_expression(children[0])
        prop_name = children[-1].name or children[-1].text or ""

        if is_optional:
            return self._desugarer.desugar_optional_chain(obj_val, prop_name, loc)

        result = self._new_ssa("_prop")
        self._emit(LoadAttrNode(
            target=result, obj=obj_val, attr=prop_name,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        return result

    def _compile_element_access(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        children = node.children
        if len(children) < 2:
            return self._emit_constant(None, UNKNOWN_TYPE, loc)

        obj_val = self.compile_expression(children[0])
        idx_val = self.compile_expression(children[1])

        is_optional = node.properties.get("questionDotToken") is not None
        if is_optional:
            # Desugar to null-check + subscript
            null_val = self._emit_constant(None, NONE_TYPE, loc)
            nc_cond = self._new_ssa("_ea_nc")
            self._emit(CompareNode(
                target=nc_cond, left=obj_val, right=null_val,
                op=CompareOp.Eq, location=loc,
                language_tag=LanguageTag.TypeScript,
            ))
            null_lbl = self._fresh_label("ea_null")
            access_lbl = self._fresh_label("ea_access")
            merge_lbl = self._fresh_label("ea_merge")
            origin = self._current_label()
            self._terminate(ConditionalBranchNode(
                condition=nc_cond, true_label=null_lbl,
                false_label=access_lbl, location=loc,
            ))
            self._cfg_builder.connect(origin, null_lbl, EdgeKind.TrueGuard)
            self._cfg_builder.connect(origin, access_lbl, EdgeKind.FalseGuard)

            self._start_block(null_lbl)
            undef_val = self._emit_constant(None, NONE_TYPE, loc)
            self._terminate(BranchNode(target_label=merge_lbl, location=loc))

            self._start_block(access_lbl)
            access_result = self._new_ssa("_ea_result")
            self._emit(LoadSubscriptNode(
                target=access_result, obj=obj_val, index=idx_val,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            self._terminate(BranchNode(target_label=merge_lbl, location=loc))

            self._start_block(merge_lbl)
            merged = self._new_ssa("_ea_merged")
            self._emit(PhiNode(
                target=merged,
                incoming={null_lbl: undef_val, access_lbl: access_result},
                location=loc,
            ))
            self._cfg_builder.connect(null_lbl, merge_lbl)
            self._cfg_builder.connect(access_lbl, merge_lbl)
            return merged

        result = self._new_ssa("_elem")
        self._emit(LoadSubscriptNode(
            target=result, obj=obj_val, index=idx_val,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        return result

    # -- Conditional (ternary) -----------------------------------------------

    def _compile_conditional(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        children = node.children
        if len(children) < 3:
            return self._emit_constant(None, UNKNOWN_TYPE, loc)

        cond_val = self.compile_expression(children[0])
        truth = self._new_ssa("_tern_cond")
        self._emit(TruthinessCoercionNode(
            target=truth, operand=cond_val,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))

        then_lbl = self._fresh_label("tern_then")
        else_lbl = self._fresh_label("tern_else")
        merge_lbl = self._fresh_label("tern_merge")
        origin = self._current_label()

        self._terminate(ConditionalBranchNode(
            condition=truth, true_label=then_lbl,
            false_label=else_lbl, location=loc,
        ))
        self._cfg_builder.connect(origin, then_lbl, EdgeKind.TrueGuard)
        self._cfg_builder.connect(origin, else_lbl, EdgeKind.FalseGuard)

        self._start_block(then_lbl)
        then_val = self.compile_expression(children[1])
        self._terminate(BranchNode(target_label=merge_lbl, location=loc))
        self._cfg_builder.connect(then_lbl, merge_lbl)

        self._start_block(else_lbl)
        else_val = self.compile_expression(children[2])
        self._terminate(BranchNode(target_label=merge_lbl, location=loc))
        self._cfg_builder.connect(else_lbl, merge_lbl)

        self._start_block(merge_lbl)
        merged = self._new_ssa("_tern_result")
        self._emit(PhiNode(
            target=merged,
            incoming={then_lbl: then_val, else_lbl: else_val},
            location=loc,
        ))
        return merged

    # -- Arrow function / function expression --------------------------------

    def _compile_arrow_or_func_expr(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        func_name = node.name or node.text or f"_lambda_{self._ssa_counter}"

        params: List[SSAVariable] = []
        for p in node.children_of_kind(TSNodeKind.Parameter):
            p_name = p.name or p.text or "_param"
            p_type: Optional[IRType] = None
            if p.type_info:
                p_type = self._type_extractor.extract_from_string(p.type_info)
            params.append(SSAVariable(
                ssa_value=SSAValue(p_name, version=0, type_annotation=p_type),
                is_parameter=True,
            ))

        body_builder = TSSSABuilder(source_file=self._source_file)
        body_builder._scope.push_scope(_ScopeKind.Function)
        for param in params:
            body_builder._scope.declare(param.name, param.ssa_value, "const")

        body_ast = node.first_child_of_kind(TSNodeKind.Block)
        if body_ast:
            for stmt in body_ast.children:
                body_builder._compile_statement(stmt)
        else:
            # Concise arrow body (expression)
            expr_children = [c for c in node.children if c.kind != TSNodeKind.Parameter]
            if expr_children:
                expr_val = body_builder.compile_expression(expr_children[-1])
                body_builder._terminate(ReturnNode(
                    value=expr_val, location=loc,
                    language_tag=LanguageTag.TypeScript,
                ))

        if body_builder._cfg_builder.current_block and body_builder._cfg_builder.current_block.terminator is None:
            undef = body_builder._emit_constant(None, NONE_TYPE, loc)
            body_builder._terminate(ReturnNode(value=undef, location=loc, language_tag=LanguageTag.TypeScript))

        body_builder._scope.pop_scope()
        cfg = body_builder._cfg_builder.finalize()

        ir_func = IRFunction(
            name=func_name,
            params=params,
            body=cfg,
            is_async="async" in node.modifiers,
            location=loc,
            language_tag=LanguageTag.TypeScript,
        )
        self._module.add_function(ir_func)

        result = self._new_ssa(func_name)
        self._emit(ConstantNode(
            target=result, value=func_name,
            type_annotation=IRType("Function"),
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        return result

    # -- Template literal ----------------------------------------------------

    def _compile_template_literal(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        parts: List[SSAValue] = []
        for child in node.children:
            if child.kind in (TSNodeKind.TemplateHead, TSNodeKind.TemplateMiddle,
                              TSNodeKind.TemplateTail, TSNodeKind.NoSubstitutionTemplateLiteral):
                text = child.text.strip("`").rstrip("${").lstrip("}")
                if text:
                    parts.append(self._emit_constant(text, STR_TYPE, loc))
            elif child.kind == TSNodeKind.TemplateSpan:
                for span_child in child.children:
                    if span_child.kind in (TSNodeKind.TemplateMiddle, TSNodeKind.TemplateTail):
                        text = span_child.text.strip("`").rstrip("${").lstrip("}")
                        if text:
                            parts.append(self._emit_constant(text, STR_TYPE, loc))
                    else:
                        parts.append(self.compile_expression(span_child))
            else:
                parts.append(self.compile_expression(child))

        if not parts:
            return self._emit_constant("", STR_TYPE, loc)

        # Concatenate all parts
        result = parts[0]
        for part in parts[1:]:
            new_result = self._new_ssa("_tmpl")
            self._emit(BinOpNode(
                target=new_result, left=result, right=part,
                op=BinOp.Add, location=loc,
                language_tag=LanguageTag.TypeScript,
            ))
            result = new_result
        return result

    # -- typeof --------------------------------------------------------------

    def _compile_typeof(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        if node.children:
            val = self.compile_expression(node.children[0])
            result = self._new_ssa("_typeof")
            self._emit(UnaryOpNode(
                target=result, operand=val, op=UnaryOp.Not,
                location=loc, language_tag=LanguageTag.TypeScript,
                metadata={"original_op": "typeof"},
            ))
            # Typeof returns a string; override semantics
            result.type_annotation = STR_TYPE
            return result
        return self._emit_constant("undefined", STR_TYPE, loc)

    # -- delete --------------------------------------------------------------

    def _compile_delete(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        if node.children:
            self.compile_expression(node.children[0])
        return self._emit_constant(True, BOOL_TYPE, loc)

    # -- Type assertion (as / angle-bracket) ---------------------------------

    def _compile_type_assertion(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        if not node.children:
            return self._emit_constant(None, UNKNOWN_TYPE, loc)

        expr_val = self.compile_expression(node.children[0])
        # Extract the target type
        asserted_type: Optional[IRType] = None
        if len(node.children) >= 2:
            asserted_type = self._type_extractor.extract(node.children[-1])
        elif node.type_info:
            asserted_type = self._type_extractor.extract_from_string(node.type_info)

        if asserted_type:
            result = self._new_ssa("_as")
            self._emit(TypeNarrowNode(
                target=result, source=expr_val,
                narrowed_type=asserted_type,
                guard_condition="type_assertion",
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            return result
        return expr_val

    # -- Non-null assertion (!) ----------------------------------------------

    def _compile_non_null_assertion(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        if not node.children:
            return self._emit_constant(None, UNKNOWN_TYPE, loc)

        expr_val = self.compile_expression(node.children[0])
        result = self._new_ssa("_nonnull")
        base_type = expr_val.type_annotation or UNKNOWN_TYPE
        non_null_type = IRType(base_type.name, args=list(base_type.args), nullable=False)
        self._emit(TypeNarrowNode(
            target=result, source=expr_val,
            narrowed_type=non_null_type,
            guard_condition="non_null_assertion",
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        return result

    # -- Object / Array literals ---------------------------------------------

    def _compile_object_literal(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        obj = self._new_ssa("_obj")
        self._emit(ConstantNode(
            target=obj, value={},
            type_annotation=IRType("object"),
            location=loc, language_tag=LanguageTag.TypeScript,
        ))

        for child in node.children:
            if child.kind == TSNodeKind.PropertyAssignment:
                key_name = child.name or (child.children[0].text if child.children else "")
                value_node = child.children[-1] if child.children else None
                if key_name and value_node:
                    val = self.compile_expression(value_node)
                    key = self._emit_constant(key_name, STR_TYPE, loc)
                    self._emit(StoreSubscriptNode(
                        obj=obj, index=key, value=val,
                        location=loc, language_tag=LanguageTag.TypeScript,
                    ))
            elif child.kind == TSNodeKind.ShorthandPropertyAssignment:
                prop_name = child.name or child.text or ""
                if prop_name:
                    val = self._compile_identifier(child)
                    key = self._emit_constant(prop_name, STR_TYPE, loc)
                    self._emit(StoreSubscriptNode(
                        obj=obj, index=key, value=val,
                        location=loc, language_tag=LanguageTag.TypeScript,
                    ))
            elif child.kind in (TSNodeKind.SpreadAssignment, TSNodeKind.SpreadElement):
                if child.children:
                    spread_val = self.compile_expression(child.children[0])
                    assign_fn = self._resolve_or_create("Object.assign")
                    merged = self._new_ssa("_spread_obj")
                    self._emit(CallNode(
                        target=merged, callee=assign_fn,
                        args=[obj, spread_val],
                        location=loc, language_tag=LanguageTag.TypeScript,
                    ))
                    obj = merged
            elif child.kind == TSNodeKind.MethodDeclaration:
                method_name = child.name or child.text or "_method"
                method_val = self._compile_arrow_or_func_expr(child)
                key = self._emit_constant(method_name, STR_TYPE, loc)
                self._emit(StoreSubscriptNode(
                    obj=obj, index=key, value=method_val,
                    location=loc, language_tag=LanguageTag.TypeScript,
                ))
            elif child.kind == TSNodeKind.ComputedPropertyName:
                if len(child.children) >= 2:
                    key_val = self.compile_expression(child.children[0])
                    val = self.compile_expression(child.children[1])
                    self._emit(StoreSubscriptNode(
                        obj=obj, index=key_val, value=val,
                        location=loc, language_tag=LanguageTag.TypeScript,
                    ))

        return obj

    def _compile_array_literal(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        elements: List[SSAValue] = []
        has_spread = False
        for child in node.children:
            if child.kind == TSNodeKind.SpreadElement:
                has_spread = True
                if child.children:
                    elements.append(self.compile_expression(child.children[0]))
            elif child.kind == TSNodeKind.OmittedExpression:
                elements.append(self._emit_constant(None, NONE_TYPE, loc))
            else:
                elements.append(self.compile_expression(child))

        if has_spread:
            concat_fn = self._resolve_or_create("Array.prototype.concat")
            result = self._new_ssa("_arr")
            empty = self._emit_constant([], IRType("Array"), loc)
            self._emit(CallNode(
                target=result, callee=concat_fn,
                args=[empty] + elements,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            return result

        result = self._new_ssa("_arr")
        self._emit(ConstantNode(
            target=result, value=elements,
            type_annotation=IRType("Array"),
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        return result

    # -- Await / Yield -------------------------------------------------------

    def _compile_await(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        if node.children:
            val = self.compile_expression(node.children[0])
            result = self._new_ssa("_await")
            self._emit(YieldNode(
                target=result, value=val, is_await=True,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            return result
        return self._emit_constant(None, NONE_TYPE, loc)

    def _compile_yield(self, node: TSASTNode) -> SSAValue:
        loc = node.source_location
        is_delegate = "yield*" in (node.text or "")
        if node.children:
            val = self.compile_expression(node.children[0])
            result = self._new_ssa("_yield")
            self._emit(YieldNode(
                target=result, value=val,
                is_yield_from=is_delegate,
                location=loc, language_tag=LanguageTag.TypeScript,
            ))
            return result
        result = self._new_ssa("_yield")
        undef = self._emit_constant(None, NONE_TYPE, loc)
        self._emit(YieldNode(
            target=result, value=undef,
            location=loc, language_tag=LanguageTag.TypeScript,
        ))
        return result


# ---------------------------------------------------------------------------
# Node.js Bridge
# ---------------------------------------------------------------------------

class NodeJSBridge:
    """
    Manages a Node.js subprocess that runs the TypeScript Compiler API to
    parse TS source into a JSON AST.

    The bridge script is expected to:
    1. Read a JSON message ``{"source": "...", "filename": "..."}`` on stdin.
    2. Write a JSON AST on stdout.
    3. Stay alive for multiple requests (REPL mode).
    """

    _BRIDGE_SCRIPT = r"""
const ts = require('typescript');
const readline = require('readline');

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
    try {
        const req = JSON.parse(line);
        const sf = ts.createSourceFile(
            req.filename || 'input.ts',
            req.source,
            ts.ScriptTarget.Latest,
            true,
            ts.ScriptKind.TS,
        );
        function visit(node) {
            const result = {
                kind: ts.SyntaxKind[node.kind],
                text: node.getText ? node.getText(sf) : '',
                pos: node.pos,
                end: node.end,
                children: [],
            };
            if (node.name && node.name.getText)
                result.name = node.name.getText(sf);
            if (node.operatorToken)
                result.operatorToken = ts.SyntaxKind[node.operatorToken.kind]
                    ? node.operatorToken.getText(sf) : '';
            const modifiers = ts.canHaveModifiers(node) ? ts.getModifiers(node) : undefined;
            if (modifiers)
                result.modifiers = modifiers.map(m => m.getText(sf));
            ts.forEachChild(node, child => {
                result.children.push(visit(child));
            });
            return result;
        }
        const ast = visit(sf);
        process.stdout.write(JSON.stringify(ast) + '\n');
    } catch (e) {
        process.stdout.write(JSON.stringify({error: e.message}) + '\n');
    }
});
"""

    def __init__(
        self,
        node_executable: str = "node",
        timeout: float = 30.0,
    ) -> None:
        self._node_executable = node_executable
        self._timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._script_path: Optional[str] = None

    def start(self) -> None:
        """Start the Node.js bridge subprocess."""
        if self._process is not None and self._process.poll() is None:
            return  # already running

        fd, self._script_path = tempfile.mkstemp(suffix=".js", prefix="ts_bridge_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(self._BRIDGE_SCRIPT)
        except Exception:
            os.close(fd)
            raise

        self._process = subprocess.Popen(
            [self._node_executable, self._script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def stop(self) -> None:
        """Stop the bridge subprocess."""
        if self._process is not None:
            try:
                self._process.stdin.close()  # type: ignore[union-attr]
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                if self._process.poll() is None:
                    self._process.kill()
            finally:
                self._process = None

        if self._script_path and os.path.exists(self._script_path):
            try:
                os.unlink(self._script_path)
            except OSError:
                pass
            self._script_path = None

    def parse(self, source: str, filename: str = "input.ts") -> Dict[str, Any]:
        """
        Send *source* to the TS compiler API and return the JSON AST.

        Raises ``RuntimeError`` on timeout or bridge failure.
        """
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self.start()

            request = json.dumps({"source": source, "filename": filename}) + "\n"
            try:
                assert self._process is not None
                assert self._process.stdin is not None
                assert self._process.stdout is not None
                self._process.stdin.write(request)
                self._process.stdin.flush()

                # Read response with timeout
                result: Optional[str] = None
                reader_exc: List[Exception] = []

                def _read() -> None:
                    nonlocal result
                    try:
                        result = self._process.stdout.readline()  # type: ignore[union-attr]
                    except Exception as e:
                        reader_exc.append(e)

                t = threading.Thread(target=_read, daemon=True)
                t.start()
                t.join(timeout=self._timeout)

                if t.is_alive():
                    raise RuntimeError(
                        f"Node.js bridge timed out after {self._timeout}s"
                    )
                if reader_exc:
                    raise RuntimeError(
                        f"Node.js bridge read error: {reader_exc[0]}"
                    )
                if not result:
                    raise RuntimeError("Node.js bridge returned empty response")

                data = json.loads(result)
                if "error" in data:
                    raise RuntimeError(f"Node.js bridge error: {data['error']}")
                return data

            except (BrokenPipeError, OSError) as e:
                self.stop()
                raise RuntimeError(f"Node.js bridge pipe error: {e}") from e

    def __enter__(self) -> NodeJSBridge:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def __del__(self) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Top-level Compiler
# ---------------------------------------------------------------------------

class TypeScriptSSACompiler:
    """
    Top-level driver that orchestrates TS source → SSA IR compilation.

    Supports three entry points:

    * ``compile_source(source)`` — parse from raw TS text
    * ``compile_ast(ast_json)`` — from pre-parsed JSON AST
    * ``compile_file(path)`` — read a ``.ts`` file and compile it

    The compiler will attempt to use the Node.js bridge for parsing
    (if Node.js and ``typescript`` are available), falling back to a
    simple JSON-based AST if provided directly.
    """

    def __init__(
        self,
        use_bridge: bool = True,
        node_executable: str = "node",
        bridge_timeout: float = 30.0,
    ) -> None:
        self._use_bridge = use_bridge
        self._bridge: Optional[NodeJSBridge] = None
        self._node_executable = node_executable
        self._bridge_timeout = bridge_timeout

    def compile_source(self, source: str, filename: str = "input.ts") -> IRModule:
        """
        Compile TypeScript source code to an ``IRModule``.

        Parses *source* using the Node.js bridge (if available) or expects
        *source* to be a JSON AST string.
        """
        ast_json = self._parse_source(source, filename)
        return self.compile_ast(ast_json, source_file=filename)

    def compile_ast(self, ast_json: Dict[str, Any], source_file: str = "<ts>") -> IRModule:
        """
        Compile a pre-parsed JSON AST (from the TS Compiler API) into an
        ``IRModule``.
        """
        root = TSASTNode.from_json(ast_json)
        builder = TSSSABuilder(source_file=source_file)
        module = builder.build_module(root)

        # Run SSA construction on each function body
        self._run_ssa_construction(module)

        return module

    def compile_file(self, path: Path) -> IRModule:
        """Read a ``.ts`` file and compile it to ``IRModule``."""
        source = path.read_text(encoding="utf-8")
        return self.compile_source(source, filename=str(path))

    def _parse_source(self, source: str, filename: str) -> Dict[str, Any]:
        """Parse source via Node.js bridge or as JSON."""
        if self._use_bridge:
            try:
                return self._parse_via_bridge(source, filename)
            except RuntimeError:
                logger.warning(
                    "Node.js bridge unavailable; attempting JSON parse fallback"
                )

        # Fallback: assume source is already JSON AST
        try:
            return json.loads(source)
        except json.JSONDecodeError:
            # Return a minimal AST wrapping the source as a single statement
            logger.warning("Could not parse source; creating minimal AST stub")
            return {
                "kind": "SourceFile",
                "text": source,
                "children": [],
                "pos": 0,
                "end": len(source),
            }

    def _parse_via_bridge(self, source: str, filename: str) -> Dict[str, Any]:
        """Parse using the Node.js bridge."""
        if self._bridge is None:
            self._bridge = NodeJSBridge(
                node_executable=self._node_executable,
                timeout=self._bridge_timeout,
            )
            self._bridge.start()
        return self._bridge.parse(source, filename)

    def _run_ssa_construction(self, module: IRModule) -> None:
        """Run SSA construction (phi-insertion + renaming) on all function CFGs."""
        for func in module.functions.values():
            if func.body is not None:
                initial_defs: Dict[str, int] = {}
                for param in func.params:
                    initial_defs[param.name] = 1
                try:
                    ssa = SSABuilder(func.body)
                    ssa.build(initial_defs=initial_defs)
                except Exception as e:
                    logger.warning("SSA construction failed for %s: %s", func.name, e)

        for cls in module.classes.values():
            for method in cls.all_methods().values():
                if method.body is not None:
                    initial_defs = {}
                    for param in method.params:
                        initial_defs[param.name] = 1
                    try:
                        ssa = SSABuilder(method.body)
                        ssa.build(initial_defs=initial_defs)
                    except Exception as e:
                        logger.warning(
                            "SSA construction failed for %s.%s: %s",
                            cls.name, method.name, e,
                        )

    def close(self) -> None:
        """Shut down the Node.js bridge if running."""
        if self._bridge is not None:
            self._bridge.stop()
            self._bridge = None

    def __enter__(self) -> TypeScriptSSACompiler:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Smoke test: compile a simple TS AST and verify IR output."""
    ast_json: Dict[str, Any] = {
        "kind": "SourceFile",
        "text": "",
        "pos": 0,
        "end": 0,
        "children": [
            {
                "kind": "FunctionDeclaration",
                "text": "function greet(name: string): string { return name; }",
                "name": "greet",
                "pos": 0,
                "end": 52,
                "modifiers": [],
                "children": [
                    {
                        "kind": "Parameter",
                        "text": "name: string",
                        "name": "name",
                        "typeInfo": "string",
                        "pos": 15,
                        "end": 27,
                        "children": [],
                    },
                    {
                        "kind": "Block",
                        "text": "{ return name; }",
                        "pos": 38,
                        "end": 52,
                        "children": [
                            {
                                "kind": "ReturnStatement",
                                "text": "return name;",
                                "pos": 40,
                                "end": 51,
                                "children": [
                                    {
                                        "kind": "Identifier",
                                        "text": "name",
                                        "name": "name",
                                        "pos": 47,
                                        "end": 51,
                                        "children": [],
                                    },
                                ],
                            },
                        ],
                    },
                ],
                "properties": {},
            },
            {
                "kind": "VariableStatement",
                "text": "const x = 42;",
                "pos": 53,
                "end": 66,
                "children": [
                    {
                        "kind": "VariableDeclarationList",
                        "text": "const x = 42",
                        "pos": 53,
                        "end": 65,
                        "modifiers": ["const"],
                        "children": [
                            {
                                "kind": "VariableDeclaration",
                                "text": "x = 42",
                                "name": "x",
                                "pos": 59,
                                "end": 65,
                                "children": [
                                    {
                                        "kind": "NumericLiteral",
                                        "text": "42",
                                        "pos": 63,
                                        "end": 65,
                                        "children": [],
                                    },
                                ],
                                "properties": {},
                            },
                        ],
                        "properties": {"declarationKind": "const"},
                    },
                ],
                "properties": {},
            },
        ],
        "properties": {},
    }

    compiler = TypeScriptSSACompiler(use_bridge=False)
    module = compiler.compile_ast(ast_json, source_file="test.ts")

    assert module.source_language == LanguageTag.TypeScript
    assert "greet" in module.functions
    assert "<module>" in module.functions
    greet = module.functions["greet"]
    assert len(greet.params) == 1
    assert greet.params[0].name == "name"
    assert greet.language_tag == LanguageTag.TypeScript

    print("compiler.py self-test passed.")


if __name__ == "__main__":
    _self_test()
