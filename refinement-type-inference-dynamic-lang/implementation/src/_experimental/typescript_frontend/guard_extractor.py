"""
TypeScript guard extraction module.

Extracts TypeScript-specific runtime guards from a simplified AST
representation and converts them to predicate templates suitable for
refinement-type inference and contract discovery seeding.

Since we cannot use the TypeScript compiler API directly from Python,
this module operates on a JSON-based AST representation (as produced by
a TS-to-JSON exporter) or on a lightweight internal AST built from
source analysis.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TSGuardPattern(Enum):
    """Classification of a TypeScript runtime guard pattern."""
    TypeOf = auto()
    InstanceOf = auto()
    Nullity = auto()
    StrictEquality = auto()
    LooseEquality = auto()
    PropertyIn = auto()
    ArrayIsArray = auto()
    Truthiness = auto()
    DiscriminatedUnion = auto()
    TypePredicate = auto()
    OptionalChaining = auto()
    NullishCoalescing = auto()
    Comparison = auto()
    Negation = auto()
    Conjunction = auto()
    Disjunction = auto()


class Polarity(Enum):
    """Which branch the guard applies to."""
    TRUE_BRANCH = auto()
    FALSE_BRANCH = auto()


class TSPredicateKind(Enum):
    """Kind of predicate produced by TS guard extraction."""
    TypeTag = auto()
    Nullity = auto()
    Truthiness = auto()
    HasAttr = auto()
    Comparison = auto()
    Membership = auto()
    Callable = auto()
    Conjunction = auto()
    Disjunction = auto()
    Negation = auto()
    Identity = auto()
    DiscriminatedUnion = auto()
    TypePredicate = auto()
    OptionalChain = auto()
    NullishCoalesce = auto()
    ArrayCheck = auto()


class ComparisonOp(Enum):
    """Comparison operators."""
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    STRICT_EQ = "==="
    STRICT_NE = "!=="
    LOOSE_EQ = "=="
    LOOSE_NE = "!="


class TypeOfResult(Enum):
    """Possible results of the typeof operator."""
    STRING = "string"
    NUMBER = "number"
    BIGINT = "bigint"
    BOOLEAN = "boolean"
    SYMBOL = "symbol"
    UNDEFINED = "undefined"
    OBJECT = "object"
    FUNCTION = "function"


# ---------------------------------------------------------------------------
# Source location
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceLocation:
    """Location in source code."""
    filename: str = ""
    line: int = 0
    col: int = 0
    end_line: int = 0
    end_col: int = 0

    @staticmethod
    def from_dict(data: Dict[str, Any], filename: str = "") -> SourceLocation:
        return SourceLocation(
            filename=filename or data.get("filename", ""),
            line=data.get("line", 0),
            col=data.get("col", data.get("column", 0)),
            end_line=data.get("endLine", data.get("end_line", 0)),
            end_col=data.get("endCol", data.get("end_col", 0)),
        )


# ---------------------------------------------------------------------------
# Simplified TS AST nodes
# ---------------------------------------------------------------------------

@dataclass
class TSNode:
    """Base class for simplified TypeScript AST nodes."""
    kind: str = ""
    loc: Optional[SourceLocation] = None
    raw: str = ""

    @staticmethod
    def from_dict(data: Dict[str, Any], filename: str = "") -> TSNode:
        kind = data.get("kind", data.get("type", ""))
        loc = SourceLocation.from_dict(data.get("loc", {}), filename)
        raw = data.get("raw", data.get("text", ""))
        return TSNode(kind=kind, loc=loc, raw=raw)


@dataclass
class TSIdentifier(TSNode):
    name: str = ""


@dataclass
class TSLiteral(TSNode):
    value: Any = None
    literal_type: str = ""


@dataclass
class TSBinaryExpr(TSNode):
    left: Optional[TSNode] = None
    operator: str = ""
    right: Optional[TSNode] = None


@dataclass
class TSUnaryExpr(TSNode):
    operator: str = ""
    operand: Optional[TSNode] = None
    prefix: bool = True


@dataclass
class TSCallExpr(TSNode):
    callee: Optional[TSNode] = None
    arguments: List[TSNode] = field(default_factory=list)


@dataclass
class TSMemberExpr(TSNode):
    object_node: Optional[TSNode] = None
    property_name: str = ""
    computed: bool = False
    optional: bool = False


@dataclass
class TSTypeOfExpr(TSNode):
    operand: Optional[TSNode] = None


@dataclass
class TSConditionalExpr(TSNode):
    test: Optional[TSNode] = None
    consequent: Optional[TSNode] = None
    alternate: Optional[TSNode] = None


@dataclass
class TSOptionalChainExpr(TSNode):
    """Represents x?.prop or x?.[index] or x?.()."""
    base: Optional[TSNode] = None
    chain_parts: List[str] = field(default_factory=list)


@dataclass
class TSNullishCoalesceExpr(TSNode):
    """Represents x ?? default."""
    left: Optional[TSNode] = None
    right: Optional[TSNode] = None


# ---------------------------------------------------------------------------
# Predicate templates
# ---------------------------------------------------------------------------

@dataclass
class TSPredicateTemplate:
    """Base predicate template."""
    kind: TSPredicateKind
    variables: List[str] = field(default_factory=list)

    def negate(self) -> TSPredicateTemplate:
        return TSNegationPredicate(child=self, variables=list(self.variables))

    def conjoin(self, other: TSPredicateTemplate) -> TSConjunctionPredicate:
        combined = sorted(set(self.variables + other.variables))
        return TSConjunctionPredicate(children=[self, other], variables=combined)

    def disjoin(self, other: TSPredicateTemplate) -> TSDisjunctionPredicate:
        combined = sorted(set(self.variables + other.variables))
        return TSDisjunctionPredicate(children=[self, other], variables=combined)


@dataclass
class TSTypeTagPredicate(TSPredicateTemplate):
    """Guard that narrows a type tag (typeof, instanceof, Array.isArray)."""
    kind: TSPredicateKind = field(default=TSPredicateKind.TypeTag, init=False)
    target_variable: str = ""
    type_names: Tuple[str, ...] = ()
    tag_source: str = ""  # "typeof", "instanceof", "Array.isArray"

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class TSNullityPredicate(TSPredicateTemplate):
    """Guard that checks for null / undefined."""
    kind: TSPredicateKind = field(default=TSPredicateKind.Nullity, init=False)
    target_variable: str = ""
    is_null: bool = False
    is_undefined: bool = False
    covers_both: bool = False  # True for == null (covers null + undefined)

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class TSTruthinessPredicate(TSPredicateTemplate):
    """Guard that checks truthiness."""
    kind: TSPredicateKind = field(default=TSPredicateKind.Truthiness, init=False)
    target_variable: str = ""
    narrows_out: List[str] = field(default_factory=list)  # types removed by truthiness

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class TSHasAttrPredicate(TSPredicateTemplate):
    """Guard from 'prop' in obj."""
    kind: TSPredicateKind = field(default=TSPredicateKind.HasAttr, init=False)
    target_variable: str = ""
    property_name: str = ""

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class TSComparisonPredicate(TSPredicateTemplate):
    """General comparison predicate."""
    kind: TSPredicateKind = field(default=TSPredicateKind.Comparison, init=False)
    left_expr: str = ""
    op: ComparisonOp = ComparisonOp.STRICT_EQ
    right_expr: str = ""
    is_strict: bool = True

    def __post_init__(self) -> None:
        if not self.variables:
            self.variables = []


@dataclass
class TSMembershipPredicate(TSPredicateTemplate):
    """Membership check."""
    kind: TSPredicateKind = field(default=TSPredicateKind.Membership, init=False)
    element_expr: str = ""
    collection_expr: str = ""

    def __post_init__(self) -> None:
        if not self.variables:
            self.variables = []


@dataclass
class TSCallablePredicate(TSPredicateTemplate):
    """Callable check (typeof x === 'function')."""
    kind: TSPredicateKind = field(default=TSPredicateKind.Callable, init=False)
    target_variable: str = ""

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class TSDiscriminatedUnionPredicate(TSPredicateTemplate):
    """Guard for discriminated union narrowing: x.kind === 'circle'."""
    kind: TSPredicateKind = field(default=TSPredicateKind.DiscriminatedUnion, init=False)
    target_variable: str = ""
    discriminant_property: str = ""
    discriminant_value: str = ""

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class TSTypePredicatePredicate(TSPredicateTemplate):
    """Guard from user-defined type predicates: function isFoo(x): x is Foo."""
    kind: TSPredicateKind = field(default=TSPredicateKind.TypePredicate, init=False)
    target_variable: str = ""
    predicate_type: str = ""
    function_name: str = ""
    asserts: bool = False

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class TSOptionalChainPredicate(TSPredicateTemplate):
    """Implicit null guard from optional chaining (?.)."""
    kind: TSPredicateKind = field(default=TSPredicateKind.OptionalChain, init=False)
    target_variable: str = ""
    chain_path: str = ""
    guarded_segments: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class TSNullishCoalescePredicate(TSPredicateTemplate):
    """Implicit null guard from nullish coalescing (??)."""
    kind: TSPredicateKind = field(default=TSPredicateKind.NullishCoalesce, init=False)
    target_variable: str = ""
    default_expr: str = ""

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class TSArrayCheckPredicate(TSPredicateTemplate):
    """Guard from Array.isArray(x)."""
    kind: TSPredicateKind = field(default=TSPredicateKind.ArrayCheck, init=False)
    target_variable: str = ""

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class TSIdentityPredicate(TSPredicateTemplate):
    """Identity comparison."""
    kind: TSPredicateKind = field(default=TSPredicateKind.Identity, init=False)
    left_variable: str = ""
    right_expr: str = ""
    is_strict: bool = True
    is_positive: bool = True

    def __post_init__(self) -> None:
        if self.left_variable and self.left_variable not in self.variables:
            self.variables = [self.left_variable] + self.variables


@dataclass
class TSConjunctionPredicate(TSPredicateTemplate):
    """Conjunction of predicates."""
    kind: TSPredicateKind = field(default=TSPredicateKind.Conjunction, init=False)
    children: List[TSPredicateTemplate] = field(default_factory=list)


@dataclass
class TSDisjunctionPredicate(TSPredicateTemplate):
    """Disjunction of predicates."""
    kind: TSPredicateKind = field(default=TSPredicateKind.Disjunction, init=False)
    children: List[TSPredicateTemplate] = field(default_factory=list)


@dataclass
class TSNegationPredicate(TSPredicateTemplate):
    """Negation of a predicate."""
    kind: TSPredicateKind = field(default=TSPredicateKind.Negation, init=False)
    child: Optional[TSPredicateTemplate] = None


# ---------------------------------------------------------------------------
# ExtractedTSGuard
# ---------------------------------------------------------------------------

@dataclass
class ExtractedTSGuard:
    """A guard extracted from TypeScript code."""
    pattern: TSGuardPattern
    variables: List[str]
    predicate: TSPredicateTemplate
    source_location: SourceLocation
    polarity: Polarity = Polarity.TRUE_BRANCH
    raw_source: str = ""
    confidence: float = 1.0
    is_strict: bool = True  # whether strict equality was used

    def negated(self) -> ExtractedTSGuard:
        new_polarity = (
            Polarity.FALSE_BRANCH
            if self.polarity == Polarity.TRUE_BRANCH
            else Polarity.TRUE_BRANCH
        )
        return ExtractedTSGuard(
            pattern=self.pattern,
            variables=list(self.variables),
            predicate=self.predicate.negate(),
            source_location=self.source_location,
            polarity=new_polarity,
            raw_source=self.raw_source,
            confidence=self.confidence,
            is_strict=self.is_strict,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_to_str(node: TSNode) -> str:
    """Best-effort stringification of a TS AST node."""
    if node.raw:
        return node.raw
    if isinstance(node, TSIdentifier):
        return node.name
    if isinstance(node, TSLiteral):
        if isinstance(node.value, str):
            return repr(node.value)
        return str(node.value)
    if isinstance(node, TSMemberExpr):
        obj = _node_to_str(node.object_node) if node.object_node else "?"
        sep = "?." if node.optional else "."
        return f"{obj}{sep}{node.property_name}"
    if isinstance(node, TSBinaryExpr):
        l = _node_to_str(node.left) if node.left else "?"
        r = _node_to_str(node.right) if node.right else "?"
        return f"{l} {node.operator} {r}"
    if isinstance(node, TSUnaryExpr):
        operand = _node_to_str(node.operand) if node.operand else "?"
        if node.prefix:
            return f"{node.operator}{operand}"
        return f"{operand}{node.operator}"
    if isinstance(node, TSCallExpr):
        callee = _node_to_str(node.callee) if node.callee else "?"
        args = ", ".join(_node_to_str(a) for a in node.arguments)
        return f"{callee}({args})"
    if isinstance(node, TSTypeOfExpr):
        operand = _node_to_str(node.operand) if node.operand else "?"
        return f"typeof {operand}"
    return node.raw or "<expr>"


def _collect_identifiers(node: TSNode) -> List[str]:
    """Collect all identifier names from a TS node tree."""
    names: List[str] = []
    if isinstance(node, TSIdentifier):
        names.append(node.name)
    if isinstance(node, TSMemberExpr) and node.object_node:
        names.extend(_collect_identifiers(node.object_node))
    if isinstance(node, TSBinaryExpr):
        if node.left:
            names.extend(_collect_identifiers(node.left))
        if node.right:
            names.extend(_collect_identifiers(node.right))
    if isinstance(node, TSUnaryExpr) and node.operand:
        names.extend(_collect_identifiers(node.operand))
    if isinstance(node, TSCallExpr):
        if node.callee:
            names.extend(_collect_identifiers(node.callee))
        for arg in node.arguments:
            names.extend(_collect_identifiers(arg))
    if isinstance(node, TSTypeOfExpr) and node.operand:
        names.extend(_collect_identifiers(node.operand))
    if isinstance(node, TSOptionalChainExpr) and node.base:
        names.extend(_collect_identifiers(node.base))
    if isinstance(node, TSNullishCoalesceExpr):
        if node.left:
            names.extend(_collect_identifiers(node.left))
        if node.right:
            names.extend(_collect_identifiers(node.right))
    return sorted(set(names))


def _is_null_literal(node: TSNode) -> bool:
    if isinstance(node, TSLiteral) and node.value is None:
        return True
    if isinstance(node, TSIdentifier) and node.name == "null":
        return True
    return False


def _is_undefined(node: TSNode) -> bool:
    if isinstance(node, TSIdentifier) and node.name == "undefined":
        return True
    if isinstance(node, TSUnaryExpr) and node.operator == "void":
        return True
    return False


def _is_null_or_undefined(node: TSNode) -> Tuple[bool, bool, bool]:
    """Return (is_nullish, is_null, is_undefined)."""
    is_null = _is_null_literal(node)
    is_undef = _is_undefined(node)
    return (is_null or is_undef, is_null, is_undef)


def _is_typeof_expr(node: TSNode) -> Optional[TSNode]:
    """If node is a typeof expression, return the operand."""
    if isinstance(node, TSTypeOfExpr):
        return node.operand
    if isinstance(node, TSUnaryExpr) and node.operator == "typeof":
        return node.operand
    return None


def _is_string_literal(node: TSNode) -> Optional[str]:
    if isinstance(node, TSLiteral) and isinstance(node.value, str):
        return node.value
    return None


def _flip_op(op: ComparisonOp) -> ComparisonOp:
    flip_map = {
        ComparisonOp.LT: ComparisonOp.GT,
        ComparisonOp.LE: ComparisonOp.GE,
        ComparisonOp.GT: ComparisonOp.LT,
        ComparisonOp.GE: ComparisonOp.LE,
        ComparisonOp.STRICT_EQ: ComparisonOp.STRICT_EQ,
        ComparisonOp.STRICT_NE: ComparisonOp.STRICT_NE,
        ComparisonOp.LOOSE_EQ: ComparisonOp.LOOSE_EQ,
        ComparisonOp.LOOSE_NE: ComparisonOp.LOOSE_NE,
    }
    return flip_map.get(op, op)


def _negate_op(op: ComparisonOp) -> ComparisonOp:
    negate_map = {
        ComparisonOp.LT: ComparisonOp.GE,
        ComparisonOp.LE: ComparisonOp.GT,
        ComparisonOp.GT: ComparisonOp.LE,
        ComparisonOp.GE: ComparisonOp.LT,
        ComparisonOp.STRICT_EQ: ComparisonOp.STRICT_NE,
        ComparisonOp.STRICT_NE: ComparisonOp.STRICT_EQ,
        ComparisonOp.LOOSE_EQ: ComparisonOp.LOOSE_NE,
        ComparisonOp.LOOSE_NE: ComparisonOp.LOOSE_EQ,
    }
    return negate_map.get(op, op)


def _op_from_str(s: str) -> ComparisonOp:
    mapping = {
        "===": ComparisonOp.STRICT_EQ,
        "!==": ComparisonOp.STRICT_NE,
        "==": ComparisonOp.LOOSE_EQ,
        "!=": ComparisonOp.LOOSE_NE,
        "<": ComparisonOp.LT,
        "<=": ComparisonOp.LE,
        ">": ComparisonOp.GT,
        ">=": ComparisonOp.GE,
    }
    return mapping.get(s, ComparisonOp.STRICT_EQ)


def _is_strict_op(op: ComparisonOp) -> bool:
    return op in (ComparisonOp.STRICT_EQ, ComparisonOp.STRICT_NE)


def _is_equality_op(op: ComparisonOp) -> bool:
    return op in (
        ComparisonOp.STRICT_EQ, ComparisonOp.STRICT_NE,
        ComparisonOp.LOOSE_EQ, ComparisonOp.LOOSE_NE,
    )


# ---------------------------------------------------------------------------
# AST parser from JSON
# ---------------------------------------------------------------------------

class TSASTParser:
    """Parse a JSON AST into our simplified node representation."""

    def __init__(self, filename: str = "") -> None:
        self.filename = filename

    def parse(self, data: Dict[str, Any]) -> TSNode:
        """Parse a JSON AST node."""
        kind = data.get("kind", data.get("type", ""))
        loc = SourceLocation.from_dict(data.get("loc", {}), self.filename)

        if kind in ("Identifier", "IdentifierExpression"):
            return TSIdentifier(
                kind=kind, loc=loc,
                raw=data.get("text", data.get("name", "")),
                name=data.get("name", data.get("text", "")),
            )

        if kind in ("StringLiteral", "NumericLiteral", "BooleanLiteral",
                     "NullLiteral", "BigIntLiteral"):
            return TSLiteral(
                kind=kind, loc=loc,
                raw=data.get("text", ""),
                value=data.get("value"),
                literal_type=kind,
            )

        if kind == "BinaryExpression":
            left = self.parse(data["left"]) if "left" in data else None
            right = self.parse(data["right"]) if "right" in data else None
            return TSBinaryExpr(
                kind=kind, loc=loc,
                raw=data.get("text", ""),
                left=left,
                operator=data.get("operator", data.get("operatorToken", "")),
                right=right,
            )

        if kind in ("PrefixUnaryExpression", "UnaryExpression"):
            operand = self.parse(data["operand"]) if "operand" in data else None
            if operand is None and "argument" in data:
                operand = self.parse(data["argument"])
            return TSUnaryExpr(
                kind=kind, loc=loc,
                raw=data.get("text", ""),
                operator=data.get("operator", ""),
                operand=operand,
                prefix=data.get("prefix", True),
            )

        if kind in ("CallExpression", "CallExpr"):
            callee = None
            if "callee" in data:
                callee = self.parse(data["callee"])
            elif "expression" in data:
                callee = self.parse(data["expression"])

            arguments = []
            for arg_data in data.get("arguments", []):
                arguments.append(self.parse(arg_data))

            return TSCallExpr(
                kind=kind, loc=loc,
                raw=data.get("text", ""),
                callee=callee,
                arguments=arguments,
            )

        if kind in ("PropertyAccessExpression", "MemberExpression"):
            obj = None
            if "expression" in data:
                obj = self.parse(data["expression"])
            elif "object" in data:
                obj = self.parse(data["object"])

            prop = data.get("name", data.get("property", ""))
            if isinstance(prop, dict):
                prop = prop.get("text", prop.get("name", ""))

            return TSMemberExpr(
                kind=kind, loc=loc,
                raw=data.get("text", ""),
                object_node=obj,
                property_name=prop,
                computed=data.get("computed", False),
                optional=data.get("optional", False),
            )

        if kind == "TypeOfExpression":
            operand = self.parse(data["operand"]) if "operand" in data else None
            if operand is None and "argument" in data:
                operand = self.parse(data["argument"])
            return TSTypeOfExpr(
                kind=kind, loc=loc,
                raw=data.get("text", ""),
                operand=operand,
            )

        if kind == "ConditionalExpression":
            return TSConditionalExpr(
                kind=kind, loc=loc,
                raw=data.get("text", ""),
                test=self.parse(data["test"]) if "test" in data else None,
                consequent=self.parse(data["consequent"]) if "consequent" in data else None,
                alternate=self.parse(data["alternate"]) if "alternate" in data else None,
            )

        return TSNode(kind=kind, loc=loc, raw=data.get("text", data.get("raw", "")))

    def parse_many(self, data_list: List[Dict[str, Any]]) -> List[TSNode]:
        return [self.parse(d) for d in data_list]


# ---------------------------------------------------------------------------
# Regex-based source scanner
# ---------------------------------------------------------------------------

class TSSourceScanner:
    """Regex-based scanner for extracting guard patterns directly from
    TypeScript source text when a JSON AST is unavailable."""

    # Patterns for common guards
    TYPEOF_PATTERN = re.compile(
        r"typeof\s+(\w+(?:\.\w+)*)\s*(===|!==|==|!=)\s*['\"](\w+)['\"]"
    )
    TYPEOF_REVERSE_PATTERN = re.compile(
        r"['\"](\w+)['\"]\s*(===|!==|==|!=)\s*typeof\s+(\w+(?:\.\w+)*)"
    )
    INSTANCEOF_PATTERN = re.compile(
        r"(\w+(?:\.\w+)*)\s+instanceof\s+(\w+(?:\.\w+)*)"
    )
    NULL_CHECK_PATTERN = re.compile(
        r"(\w+(?:\.\w+)*)\s*(===|!==|==|!=)\s*(null|undefined)"
    )
    NULL_CHECK_REVERSE_PATTERN = re.compile(
        r"(null|undefined)\s*(===|!==|==|!=)\s*(\w+(?:\.\w+)*)"
    )
    PROP_IN_PATTERN = re.compile(
        r"['\"](\w+)['\"]\s+in\s+(\w+(?:\.\w+)*)"
    )
    ARRAY_IS_ARRAY_PATTERN = re.compile(
        r"Array\.isArray\s*\(\s*(\w+(?:\.\w+)*)\s*\)"
    )
    OPTIONAL_CHAIN_PATTERN = re.compile(
        r"(\w+)(\?\.\w+)+"
    )
    NULLISH_COALESCE_PATTERN = re.compile(
        r"(\w+(?:\.\w+)*)\s*\?\?\s*(.+?)(?:\s*[;,)\]}]|$)"
    )
    DISCRIMINANT_PATTERN = re.compile(
        r"(\w+)\.(\w+)\s*(===|!==)\s*['\"](\w+)['\"]"
    )
    TYPE_PREDICATE_PATTERN = re.compile(
        r"function\s+(\w+)\s*\([^)]*\)\s*:\s*\w+\s+is\s+(\w+)"
    )
    ASSERTS_PATTERN = re.compile(
        r"function\s+(\w+)\s*\([^)]*\)\s*:\s*asserts\s+(\w+)(?:\s+is\s+(\w+))?"
    )

    def __init__(self, filename: str = "") -> None:
        self.filename = filename

    def scan(self, source: str) -> List[ExtractedTSGuard]:
        """Scan source text for guard patterns."""
        guards: List[ExtractedTSGuard] = []
        lines = source.split("\n")

        for line_num, line in enumerate(lines, 1):
            loc = SourceLocation(
                filename=self.filename,
                line=line_num, col=0,
                end_line=line_num, end_col=len(line),
            )

            # typeof x === 'type'
            for m in self.TYPEOF_PATTERN.finditer(line):
                guards.append(self._make_typeof_guard(
                    m.group(1), m.group(2), m.group(3), loc, m.group(0)
                ))

            # 'type' === typeof x
            for m in self.TYPEOF_REVERSE_PATTERN.finditer(line):
                guards.append(self._make_typeof_guard(
                    m.group(3), m.group(2), m.group(1), loc, m.group(0)
                ))

            # x instanceof Class
            for m in self.INSTANCEOF_PATTERN.finditer(line):
                guards.append(self._make_instanceof_guard(
                    m.group(1), m.group(2), loc, m.group(0)
                ))

            # x === null / x !== undefined
            for m in self.NULL_CHECK_PATTERN.finditer(line):
                guards.append(self._make_null_guard(
                    m.group(1), m.group(2), m.group(3), loc, m.group(0)
                ))

            # null === x / undefined !== x
            for m in self.NULL_CHECK_REVERSE_PATTERN.finditer(line):
                guards.append(self._make_null_guard(
                    m.group(3), m.group(2), m.group(1), loc, m.group(0)
                ))

            # 'prop' in obj
            for m in self.PROP_IN_PATTERN.finditer(line):
                guards.append(self._make_prop_in_guard(
                    m.group(2), m.group(1), loc, m.group(0)
                ))

            # Array.isArray(x)
            for m in self.ARRAY_IS_ARRAY_PATTERN.finditer(line):
                guards.append(self._make_array_guard(
                    m.group(1), loc, m.group(0)
                ))

            # x?.prop (optional chaining)
            for m in self.OPTIONAL_CHAIN_PATTERN.finditer(line):
                guards.append(self._make_optional_chain_guard(
                    m.group(1), m.group(0), loc
                ))

            # x ?? default
            for m in self.NULLISH_COALESCE_PATTERN.finditer(line):
                guards.append(self._make_nullish_coalesce_guard(
                    m.group(1), m.group(2).strip(), loc, m.group(0)
                ))

            # x.kind === 'circle' (discriminated union)
            for m in self.DISCRIMINANT_PATTERN.finditer(line):
                guards.append(self._make_discriminant_guard(
                    m.group(1), m.group(2), m.group(3), m.group(4),
                    loc, m.group(0),
                ))

            # function isFoo(x): x is Foo
            for m in self.TYPE_PREDICATE_PATTERN.finditer(line):
                guards.append(self._make_type_predicate_guard(
                    m.group(1), m.group(2), loc, m.group(0)
                ))

        return guards

    def _make_typeof_guard(
        self, var: str, op: str, type_str: str,
        loc: SourceLocation, raw: str,
    ) -> ExtractedTSGuard:
        is_positive = op in ("===", "==")
        is_strict = op in ("===", "!==")

        if type_str == "function":
            pred: TSPredicateTemplate = TSCallablePredicate(
                target_variable=var, variables=[var],
            )
        else:
            pred = TSTypeTagPredicate(
                target_variable=var,
                type_names=(type_str,),
                tag_source="typeof",
                variables=[var],
            )

        if not is_positive:
            pred = pred.negate()

        return ExtractedTSGuard(
            pattern=TSGuardPattern.TypeOf,
            variables=[var],
            predicate=pred,
            source_location=loc,
            polarity=Polarity.TRUE_BRANCH,
            raw_source=raw,
            is_strict=is_strict,
        )

    def _make_instanceof_guard(
        self, var: str, cls: str, loc: SourceLocation, raw: str,
    ) -> ExtractedTSGuard:
        pred = TSTypeTagPredicate(
            target_variable=var,
            type_names=(cls,),
            tag_source="instanceof",
            variables=[var],
        )
        return ExtractedTSGuard(
            pattern=TSGuardPattern.InstanceOf,
            variables=[var],
            predicate=pred,
            source_location=loc,
            polarity=Polarity.TRUE_BRANCH,
            raw_source=raw,
        )

    def _make_null_guard(
        self, var: str, op: str, null_val: str,
        loc: SourceLocation, raw: str,
    ) -> ExtractedTSGuard:
        is_positive = op in ("===", "==")
        is_strict = op in ("===", "!==")
        is_null = null_val == "null"
        is_undef = null_val == "undefined"
        covers_both = not is_strict  # == null covers both

        pred = TSNullityPredicate(
            target_variable=var,
            is_null=is_null if is_strict else True,
            is_undefined=is_undef if is_strict else True,
            covers_both=covers_both,
            variables=[var],
        )

        polarity = Polarity.TRUE_BRANCH if is_positive else Polarity.TRUE_BRANCH
        if not is_positive:
            pred_final: TSPredicateTemplate = pred.negate()
        else:
            pred_final = pred

        return ExtractedTSGuard(
            pattern=TSGuardPattern.Nullity,
            variables=[var],
            predicate=pred_final,
            source_location=loc,
            polarity=polarity,
            raw_source=raw,
            is_strict=is_strict,
        )

    def _make_prop_in_guard(
        self, obj: str, prop: str, loc: SourceLocation, raw: str,
    ) -> ExtractedTSGuard:
        pred = TSHasAttrPredicate(
            target_variable=obj,
            property_name=prop,
            variables=[obj],
        )
        return ExtractedTSGuard(
            pattern=TSGuardPattern.PropertyIn,
            variables=[obj],
            predicate=pred,
            source_location=loc,
            polarity=Polarity.TRUE_BRANCH,
            raw_source=raw,
        )

    def _make_array_guard(
        self, var: str, loc: SourceLocation, raw: str,
    ) -> ExtractedTSGuard:
        pred = TSArrayCheckPredicate(
            target_variable=var, variables=[var],
        )
        return ExtractedTSGuard(
            pattern=TSGuardPattern.ArrayIsArray,
            variables=[var],
            predicate=pred,
            source_location=loc,
            polarity=Polarity.TRUE_BRANCH,
            raw_source=raw,
        )

    def _make_optional_chain_guard(
        self, base_var: str, full_expr: str, loc: SourceLocation,
    ) -> ExtractedTSGuard:
        parts = full_expr.split("?.")
        guarded = [p.rstrip(".") for p in parts[:-1]]
        pred = TSOptionalChainPredicate(
            target_variable=base_var,
            chain_path=full_expr,
            guarded_segments=guarded,
            variables=[base_var],
        )
        return ExtractedTSGuard(
            pattern=TSGuardPattern.OptionalChaining,
            variables=[base_var],
            predicate=pred,
            source_location=loc,
            polarity=Polarity.TRUE_BRANCH,
            raw_source=full_expr,
            confidence=0.8,
        )

    def _make_nullish_coalesce_guard(
        self, var: str, default: str, loc: SourceLocation, raw: str,
    ) -> ExtractedTSGuard:
        pred = TSNullishCoalescePredicate(
            target_variable=var,
            default_expr=default,
            variables=[var],
        )
        return ExtractedTSGuard(
            pattern=TSGuardPattern.NullishCoalescing,
            variables=[var],
            predicate=pred,
            source_location=loc,
            polarity=Polarity.TRUE_BRANCH,
            raw_source=raw,
            confidence=0.7,
        )

    def _make_discriminant_guard(
        self, obj: str, prop: str, op: str, value: str,
        loc: SourceLocation, raw: str,
    ) -> ExtractedTSGuard:
        pred = TSDiscriminatedUnionPredicate(
            target_variable=obj,
            discriminant_property=prop,
            discriminant_value=value,
            variables=[obj],
        )
        is_positive = op == "==="
        if not is_positive:
            pred_final: TSPredicateTemplate = pred.negate()
        else:
            pred_final = pred

        return ExtractedTSGuard(
            pattern=TSGuardPattern.DiscriminatedUnion,
            variables=[obj],
            predicate=pred_final,
            source_location=loc,
            polarity=Polarity.TRUE_BRANCH,
            raw_source=raw,
        )

    def _make_type_predicate_guard(
        self, func_name: str, pred_type: str,
        loc: SourceLocation, raw: str,
    ) -> ExtractedTSGuard:
        pred = TSTypePredicatePredicate(
            target_variable="<param>",
            predicate_type=pred_type,
            function_name=func_name,
            variables=[],
        )
        return ExtractedTSGuard(
            pattern=TSGuardPattern.TypePredicate,
            variables=[],
            predicate=pred,
            source_location=loc,
            polarity=Polarity.TRUE_BRANCH,
            raw_source=raw,
        )


# ---------------------------------------------------------------------------
# TypeScriptGuardExtractor
# ---------------------------------------------------------------------------

class TypeScriptGuardExtractor:
    """Main extractor that processes TypeScript AST nodes (JSON-based)
    and falls back to source scanning."""

    def __init__(self, filename: str = "", source: str = "") -> None:
        self.filename = filename
        self.source = source
        self._parser = TSASTParser(filename)
        self._scanner = TSSourceScanner(filename)

    # -- public API --

    def extract(self, ast_data: Dict[str, Any]) -> List[ExtractedTSGuard]:
        """Extract guards from a JSON AST."""
        guards: List[ExtractedTSGuard] = []
        self._walk(ast_data, guards)
        return guards

    def extract_from_source(self, source: str) -> List[ExtractedTSGuard]:
        """Extract guards using regex scanning on source text."""
        return self._scanner.scan(source)

    def extract_combined(
        self,
        source: str,
        ast_data: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractedTSGuard]:
        """Extract from both AST (if available) and source scanning,
        then deduplicate."""
        guards: List[ExtractedTSGuard] = []
        if ast_data:
            guards.extend(self.extract(ast_data))
        guards.extend(self.extract_from_source(source))
        return self._deduplicate(guards)

    # -- AST walking --

    def _walk(self, data: Dict[str, Any], guards: List[ExtractedTSGuard]) -> None:
        kind = data.get("kind", data.get("type", ""))

        if kind in ("IfStatement", "ConditionalExpression"):
            test = data.get("test", data.get("expression"))
            if test:
                guards.extend(self._extract_from_condition(test))

        elif kind == "WhileStatement":
            test = data.get("test", data.get("expression"))
            if test:
                guards.extend(self._extract_from_condition(test))

        elif kind == "SwitchStatement":
            discriminant = data.get("discriminant", data.get("expression"))
            cases = data.get("cases", data.get("clauses", []))
            if discriminant and cases:
                guards.extend(self._extract_switch(discriminant, cases))

        # Recurse
        for key, value in data.items():
            if isinstance(value, dict) and ("kind" in value or "type" in value):
                self._walk(value, guards)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and ("kind" in item or "type" in item):
                        self._walk(item, guards)

    def _extract_from_condition(
        self, data: Dict[str, Any]
    ) -> List[ExtractedTSGuard]:
        node = self._parser.parse(data)
        return self._extract_guard_from_node(node, Polarity.TRUE_BRANCH)

    def _extract_guard_from_node(
        self, node: TSNode, polarity: Polarity
    ) -> List[ExtractedTSGuard]:
        guards: List[ExtractedTSGuard] = []
        loc = node.loc or SourceLocation(filename=self.filename)

        # Binary expressions
        if isinstance(node, TSBinaryExpr):
            # && (logical and)
            if node.operator in ("&&", "and"):
                left_guards = self._extract_guard_from_node(node.left, polarity) if node.left else []
                right_guards = self._extract_guard_from_node(node.right, polarity) if node.right else []
                all_guards = left_guards + right_guards
                if len(all_guards) > 1:
                    all_vars = sorted(set(v for g in all_guards for v in g.variables))
                    conj = TSConjunctionPredicate(
                        children=[g.predicate for g in all_guards],
                        variables=all_vars,
                    )
                    combined = ExtractedTSGuard(
                        pattern=TSGuardPattern.Conjunction,
                        variables=all_vars,
                        predicate=conj,
                        source_location=loc,
                        polarity=polarity,
                        raw_source=_node_to_str(node),
                    )
                    guards.append(combined)
                guards.extend(all_guards)
                return guards

            # || (logical or)
            if node.operator in ("||", "or"):
                left_guards = self._extract_guard_from_node(node.left, polarity) if node.left else []
                right_guards = self._extract_guard_from_node(node.right, polarity) if node.right else []
                all_guards = left_guards + right_guards
                if len(all_guards) > 1:
                    all_vars = sorted(set(v for g in all_guards for v in g.variables))
                    disj = TSDisjunctionPredicate(
                        children=[g.predicate for g in all_guards],
                        variables=all_vars,
                    )
                    combined = ExtractedTSGuard(
                        pattern=TSGuardPattern.Disjunction,
                        variables=all_vars,
                        predicate=disj,
                        source_location=loc,
                        polarity=polarity,
                        raw_source=_node_to_str(node),
                    )
                    guards = [combined]
                return guards

            # ?? (nullish coalescing)
            if node.operator == "??":
                if node.left:
                    var = _node_to_str(node.left)
                    default = _node_to_str(node.right) if node.right else "?"
                    pred = TSNullishCoalescePredicate(
                        target_variable=var,
                        default_expr=default,
                        variables=_collect_identifiers(node.left),
                    )
                    guards.append(ExtractedTSGuard(
                        pattern=TSGuardPattern.NullishCoalescing,
                        variables=_collect_identifiers(node.left),
                        predicate=pred,
                        source_location=loc,
                        polarity=polarity,
                        raw_source=_node_to_str(node),
                        confidence=0.7,
                    ))
                return guards

            # instanceof
            if node.operator == "instanceof":
                if node.left and node.right:
                    var = _node_to_str(node.left)
                    cls = _node_to_str(node.right)
                    pred = TSTypeTagPredicate(
                        target_variable=var,
                        type_names=(cls,),
                        tag_source="instanceof",
                        variables=_collect_identifiers(node.left),
                    )
                    guards.append(ExtractedTSGuard(
                        pattern=TSGuardPattern.InstanceOf,
                        variables=_collect_identifiers(node.left),
                        predicate=pred,
                        source_location=loc,
                        polarity=polarity,
                        raw_source=_node_to_str(node),
                    ))
                return guards

            # in operator
            if node.operator == "in":
                if node.left and node.right:
                    prop = _node_to_str(node.left)
                    obj = _node_to_str(node.right)
                    # Strip quotes for string literal property names
                    prop_clean = prop.strip("'\"")
                    pred = TSHasAttrPredicate(
                        target_variable=obj,
                        property_name=prop_clean,
                        variables=_collect_identifiers(node.right),
                    )
                    guards.append(ExtractedTSGuard(
                        pattern=TSGuardPattern.PropertyIn,
                        variables=_collect_identifiers(node.right),
                        predicate=pred,
                        source_location=loc,
                        polarity=polarity,
                        raw_source=_node_to_str(node),
                    ))
                return guards

            # Equality / comparison operators
            if node.operator in ("===", "!==", "==", "!=", "<", "<=", ">", ">="):
                op = _op_from_str(node.operator)
                guard = self._handle_equality(node.left, op, node.right, loc, polarity, node)
                if guard:
                    guards.append(guard)
                return guards

        # Unary not
        if isinstance(node, TSUnaryExpr) and node.operator in ("!", "not"):
            flipped = (
                Polarity.FALSE_BRANCH
                if polarity == Polarity.TRUE_BRANCH
                else Polarity.TRUE_BRANCH
            )
            if node.operand:
                inner = self._extract_guard_from_node(node.operand, flipped)
                for g in inner:
                    guards.append(g.negated())
            return guards

        # typeof expression used as boolean (typeof x)
        if isinstance(node, TSTypeOfExpr):
            if node.operand:
                var = _node_to_str(node.operand)
                pred = TSTruthinessPredicate(
                    target_variable=f"typeof {var}",
                    variables=_collect_identifiers(node.operand),
                )
                guards.append(ExtractedTSGuard(
                    pattern=TSGuardPattern.Truthiness,
                    variables=_collect_identifiers(node.operand),
                    predicate=pred,
                    source_location=loc,
                    polarity=polarity,
                    raw_source=_node_to_str(node),
                    confidence=0.5,
                ))
            return guards

        # Call expression
        if isinstance(node, TSCallExpr):
            guard = self._handle_call(node, loc, polarity)
            if guard:
                guards.append(guard)
                return guards

        # Member expression with optional chaining
        if isinstance(node, TSMemberExpr) and node.optional:
            if node.object_node:
                var = _node_to_str(node.object_node)
                pred = TSOptionalChainPredicate(
                    target_variable=var,
                    chain_path=_node_to_str(node),
                    guarded_segments=[var],
                    variables=_collect_identifiers(node.object_node),
                )
                guards.append(ExtractedTSGuard(
                    pattern=TSGuardPattern.OptionalChaining,
                    variables=_collect_identifiers(node.object_node),
                    predicate=pred,
                    source_location=loc,
                    polarity=polarity,
                    raw_source=_node_to_str(node),
                    confidence=0.8,
                ))
            return guards

        # Identifier or member expression (truthiness)
        if isinstance(node, (TSIdentifier, TSMemberExpr)):
            var = _node_to_str(node)
            ids = _collect_identifiers(node)
            pred = TSTruthinessPredicate(
                target_variable=var,
                variables=ids,
            )
            guards.append(ExtractedTSGuard(
                pattern=TSGuardPattern.Truthiness,
                variables=ids,
                predicate=pred,
                source_location=loc,
                polarity=polarity,
                raw_source=var,
            ))
            return guards

        return guards

    def _handle_equality(
        self,
        left: Optional[TSNode],
        op: ComparisonOp,
        right: Optional[TSNode],
        loc: SourceLocation,
        polarity: Polarity,
        parent: TSNode,
    ) -> Optional[ExtractedTSGuard]:
        if not left or not right:
            return None

        is_strict = _is_strict_op(op)
        is_equality = _is_equality_op(op)

        # typeof x === 'string'
        typeof_operand = _is_typeof_expr(left)
        if typeof_operand and is_equality:
            type_str = _is_string_literal(right)
            if type_str:
                return self._make_typeof_comparison(
                    typeof_operand, op, type_str, loc, polarity, parent, is_strict
                )

        # 'string' === typeof x
        typeof_operand = _is_typeof_expr(right)
        if typeof_operand and is_equality:
            type_str = _is_string_literal(left)
            if type_str:
                return self._make_typeof_comparison(
                    typeof_operand, op, type_str, loc, polarity, parent, is_strict
                )

        # x === null / x !== null / x === undefined / x !== undefined
        is_nullish_r, is_null_r, is_undef_r = _is_null_or_undefined(right)
        if is_nullish_r and is_equality:
            var = _node_to_str(left)
            covers_both = not is_strict
            is_positive = op in (ComparisonOp.STRICT_EQ, ComparisonOp.LOOSE_EQ)
            pred = TSNullityPredicate(
                target_variable=var,
                is_null=is_null_r or covers_both,
                is_undefined=is_undef_r or covers_both,
                covers_both=covers_both,
                variables=_collect_identifiers(left),
            )
            pred_final: TSPredicateTemplate = pred
            if not is_positive:
                pred_final = pred.negate()
            return ExtractedTSGuard(
                pattern=TSGuardPattern.Nullity,
                variables=_collect_identifiers(left),
                predicate=pred_final,
                source_location=loc,
                polarity=polarity,
                raw_source=_node_to_str(parent),
                is_strict=is_strict,
            )

        # null === x / undefined !== x
        is_nullish_l, is_null_l, is_undef_l = _is_null_or_undefined(left)
        if is_nullish_l and is_equality:
            var = _node_to_str(right)
            covers_both = not is_strict
            is_positive = op in (ComparisonOp.STRICT_EQ, ComparisonOp.LOOSE_EQ)
            pred = TSNullityPredicate(
                target_variable=var,
                is_null=is_null_l or covers_both,
                is_undefined=is_undef_l or covers_both,
                covers_both=covers_both,
                variables=_collect_identifiers(right),
            )
            pred_final2: TSPredicateTemplate = pred
            if not is_positive:
                pred_final2 = pred.negate()
            return ExtractedTSGuard(
                pattern=TSGuardPattern.Nullity,
                variables=_collect_identifiers(right),
                predicate=pred_final2,
                source_location=loc,
                polarity=polarity,
                raw_source=_node_to_str(parent),
                is_strict=is_strict,
            )

        # x.kind === 'value' (discriminated union)
        if isinstance(left, TSMemberExpr) and is_equality:
            val_str = _is_string_literal(right)
            if val_str and left.object_node:
                obj_var = _node_to_str(left.object_node)
                is_positive = op in (ComparisonOp.STRICT_EQ, ComparisonOp.LOOSE_EQ)
                pred = TSDiscriminatedUnionPredicate(
                    target_variable=obj_var,
                    discriminant_property=left.property_name,
                    discriminant_value=val_str,
                    variables=_collect_identifiers(left.object_node),
                )
                disc_pred: TSPredicateTemplate = pred
                if not is_positive:
                    disc_pred = pred.negate()
                return ExtractedTSGuard(
                    pattern=TSGuardPattern.DiscriminatedUnion,
                    variables=_collect_identifiers(left.object_node),
                    predicate=disc_pred,
                    source_location=loc,
                    polarity=polarity,
                    raw_source=_node_to_str(parent),
                    is_strict=is_strict,
                )

        # 'value' === x.kind
        if isinstance(right, TSMemberExpr) and is_equality:
            val_str = _is_string_literal(left)
            if val_str and right.object_node:
                obj_var = _node_to_str(right.object_node)
                is_positive = op in (ComparisonOp.STRICT_EQ, ComparisonOp.LOOSE_EQ)
                pred = TSDiscriminatedUnionPredicate(
                    target_variable=obj_var,
                    discriminant_property=right.property_name,
                    discriminant_value=val_str,
                    variables=_collect_identifiers(right.object_node),
                )
                disc_pred2: TSPredicateTemplate = pred
                if not is_positive:
                    disc_pred2 = pred.negate()
                return ExtractedTSGuard(
                    pattern=TSGuardPattern.DiscriminatedUnion,
                    variables=_collect_identifiers(right.object_node),
                    predicate=disc_pred2,
                    source_location=loc,
                    polarity=polarity,
                    raw_source=_node_to_str(parent),
                    is_strict=is_strict,
                )

        # General comparison
        all_vars = sorted(set(
            _collect_identifiers(left) + _collect_identifiers(right)
        ))
        comp_pred = TSComparisonPredicate(
            left_expr=_node_to_str(left),
            op=op,
            right_expr=_node_to_str(right),
            is_strict=is_strict,
            variables=all_vars,
        )
        pattern = (
            TSGuardPattern.StrictEquality if is_strict
            else (TSGuardPattern.LooseEquality if is_equality else TSGuardPattern.Comparison)
        )
        return ExtractedTSGuard(
            pattern=pattern,
            variables=all_vars,
            predicate=comp_pred,
            source_location=loc,
            polarity=polarity,
            raw_source=_node_to_str(parent),
            is_strict=is_strict,
        )

    def _make_typeof_comparison(
        self,
        operand: TSNode,
        op: ComparisonOp,
        type_str: str,
        loc: SourceLocation,
        polarity: Polarity,
        parent: TSNode,
        is_strict: bool,
    ) -> ExtractedTSGuard:
        var = _node_to_str(operand)
        is_positive = op in (ComparisonOp.STRICT_EQ, ComparisonOp.LOOSE_EQ)

        if type_str == "function":
            base_pred: TSPredicateTemplate = TSCallablePredicate(
                target_variable=var,
                variables=_collect_identifiers(operand),
            )
        elif type_str == "undefined":
            base_pred = TSNullityPredicate(
                target_variable=var,
                is_null=False,
                is_undefined=True,
                covers_both=False,
                variables=_collect_identifiers(operand),
            )
        else:
            base_pred = TSTypeTagPredicate(
                target_variable=var,
                type_names=(type_str,),
                tag_source="typeof",
                variables=_collect_identifiers(operand),
            )

        final_pred = base_pred if is_positive else base_pred.negate()

        return ExtractedTSGuard(
            pattern=TSGuardPattern.TypeOf,
            variables=_collect_identifiers(operand),
            predicate=final_pred,
            source_location=loc,
            polarity=polarity,
            raw_source=_node_to_str(parent),
            is_strict=is_strict,
        )

    def _handle_call(
        self, node: TSCallExpr, loc: SourceLocation, polarity: Polarity
    ) -> Optional[ExtractedTSGuard]:
        if not node.callee:
            return None

        # Array.isArray(x)
        if isinstance(node.callee, TSMemberExpr):
            if (
                node.callee.property_name == "isArray"
                and isinstance(node.callee.object_node, TSIdentifier)
                and node.callee.object_node.name == "Array"
                and len(node.arguments) >= 1
            ):
                var = _node_to_str(node.arguments[0])
                pred = TSArrayCheckPredicate(
                    target_variable=var,
                    variables=_collect_identifiers(node.arguments[0]),
                )
                return ExtractedTSGuard(
                    pattern=TSGuardPattern.ArrayIsArray,
                    variables=_collect_identifiers(node.arguments[0]),
                    predicate=pred,
                    source_location=loc,
                    polarity=polarity,
                    raw_source=_node_to_str(node),
                )

        # User-defined type guard call: isFoo(x)
        if isinstance(node.callee, TSIdentifier) and node.arguments:
            func_name = node.callee.name
            if func_name.startswith("is") and len(func_name) > 2 and func_name[2].isupper():
                var = _node_to_str(node.arguments[0])
                pred_type = func_name[2:]
                pred = TSTypePredicatePredicate(
                    target_variable=var,
                    predicate_type=pred_type,
                    function_name=func_name,
                    variables=_collect_identifiers(node.arguments[0]),
                )
                return ExtractedTSGuard(
                    pattern=TSGuardPattern.TypePredicate,
                    variables=_collect_identifiers(node.arguments[0]),
                    predicate=pred,
                    source_location=loc,
                    polarity=polarity,
                    raw_source=_node_to_str(node),
                    confidence=0.7,
                )

        # Truthiness of function call result
        var = _node_to_str(node)
        ids = _collect_identifiers(node)
        pred = TSTruthinessPredicate(
            target_variable=var,
            variables=ids,
        )
        return ExtractedTSGuard(
            pattern=TSGuardPattern.Truthiness,
            variables=ids,
            predicate=pred,
            source_location=loc,
            polarity=polarity,
            raw_source=var,
            confidence=0.5,
        )

    def _extract_switch(
        self,
        discriminant: Dict[str, Any],
        cases: List[Dict[str, Any]],
    ) -> List[ExtractedTSGuard]:
        """Extract guards from switch statements (often discriminated unions)."""
        guards: List[ExtractedTSGuard] = []
        disc_node = self._parser.parse(discriminant)
        disc_str = _node_to_str(disc_node)
        disc_vars = _collect_identifiers(disc_node)

        # Check if discriminant is a member expression (x.kind)
        is_disc_union = isinstance(disc_node, TSMemberExpr)

        for case_data in cases:
            test = case_data.get("test", case_data.get("expression"))
            if test is None:
                continue

            test_node = self._parser.parse(test)
            test_str = _node_to_str(test_node)
            loc = disc_node.loc or SourceLocation(filename=self.filename)

            if is_disc_union and isinstance(disc_node, TSMemberExpr):
                val_str = _is_string_literal(test_node)
                if val_str and disc_node.object_node:
                    obj_var = _node_to_str(disc_node.object_node)
                    pred = TSDiscriminatedUnionPredicate(
                        target_variable=obj_var,
                        discriminant_property=disc_node.property_name,
                        discriminant_value=val_str,
                        variables=_collect_identifiers(disc_node.object_node),
                    )
                    guards.append(ExtractedTSGuard(
                        pattern=TSGuardPattern.DiscriminatedUnion,
                        variables=_collect_identifiers(disc_node.object_node),
                        predicate=pred,
                        source_location=loc,
                        polarity=Polarity.TRUE_BRANCH,
                        raw_source=f"case {val_str}",
                    ))
                    continue

            # General case
            pred = TSComparisonPredicate(
                left_expr=disc_str,
                op=ComparisonOp.STRICT_EQ,
                right_expr=test_str,
                is_strict=True,
                variables=disc_vars,
            )
            guards.append(ExtractedTSGuard(
                pattern=TSGuardPattern.StrictEquality,
                variables=disc_vars,
                predicate=pred,
                source_location=loc,
                polarity=Polarity.TRUE_BRANCH,
                raw_source=f"case {test_str}",
            ))

        return guards

    def _deduplicate(
        self, guards: List[ExtractedTSGuard]
    ) -> List[ExtractedTSGuard]:
        seen: Set[str] = set()
        result: List[ExtractedTSGuard] = []
        for g in guards:
            key = f"{g.pattern.name}|{g.raw_source}|{g.source_location.line}"
            if key not in seen:
                seen.add(key)
                result.append(g)
        return result


# ---------------------------------------------------------------------------
# TypeScriptGuardNormalizer
# ---------------------------------------------------------------------------

class TypeScriptGuardNormalizer:
    """Normalizes TypeScript guards into canonical form."""

    def normalize(self, guard: ExtractedTSGuard) -> ExtractedTSGuard:
        pred = self._normalize_predicate(guard.predicate)
        return ExtractedTSGuard(
            pattern=guard.pattern,
            variables=sorted(set(guard.variables)),
            predicate=pred,
            source_location=guard.source_location,
            polarity=guard.polarity,
            raw_source=guard.raw_source,
            confidence=guard.confidence,
            is_strict=guard.is_strict,
        )

    def normalize_all(
        self, guards: List[ExtractedTSGuard]
    ) -> List[ExtractedTSGuard]:
        seen: Set[str] = set()
        result: List[ExtractedTSGuard] = []
        for g in guards:
            ng = self.normalize(g)
            key = self._guard_key(ng)
            if key not in seen:
                seen.add(key)
                result.append(ng)
        return result

    def _normalize_predicate(
        self, pred: TSPredicateTemplate
    ) -> TSPredicateTemplate:
        pred = copy.deepcopy(pred)
        pred.variables = sorted(set(pred.variables))

        if isinstance(pred, TSConjunctionPredicate):
            pred.children = [self._normalize_predicate(c) for c in pred.children]
            pred.children = self._flatten_conjunction(pred.children)
            pred.children.sort(key=self._pred_sort_key)
        elif isinstance(pred, TSDisjunctionPredicate):
            pred.children = [self._normalize_predicate(c) for c in pred.children]
            pred.children = self._flatten_disjunction(pred.children)
            pred.children.sort(key=self._pred_sort_key)
        elif isinstance(pred, TSNegationPredicate) and pred.child:
            pred.child = self._normalize_predicate(pred.child)
            if isinstance(pred.child, TSNegationPredicate) and pred.child.child:
                return pred.child.child

        # Normalize loose == null to covers_both
        if isinstance(pred, TSNullityPredicate) and pred.covers_both:
            pred.is_null = True
            pred.is_undefined = True

        return pred

    def _flatten_conjunction(
        self, children: List[TSPredicateTemplate]
    ) -> List[TSPredicateTemplate]:
        result: List[TSPredicateTemplate] = []
        for c in children:
            if isinstance(c, TSConjunctionPredicate):
                result.extend(self._flatten_conjunction(c.children))
            else:
                result.append(c)
        return result

    def _flatten_disjunction(
        self, children: List[TSPredicateTemplate]
    ) -> List[TSPredicateTemplate]:
        result: List[TSPredicateTemplate] = []
        for c in children:
            if isinstance(c, TSDisjunctionPredicate):
                result.extend(self._flatten_disjunction(c.children))
            else:
                result.append(c)
        return result

    @staticmethod
    def _pred_sort_key(pred: TSPredicateTemplate) -> str:
        return f"{pred.kind.name}:{','.join(pred.variables)}"

    @staticmethod
    def _guard_key(guard: ExtractedTSGuard) -> str:
        return (
            f"{guard.pattern.name}|{guard.polarity.name}|"
            f"{','.join(guard.variables)}|{guard.predicate.kind.name}"
        )


# ---------------------------------------------------------------------------
# DiscriminatedUnionExtractor
# ---------------------------------------------------------------------------

@dataclass
class DiscriminatedUnionInfo:
    """Information about a discriminated union pattern."""
    object_variable: str
    discriminant_property: str
    variants: List[str]
    source_locations: List[SourceLocation]
    complete: bool = False  # whether all variants are covered


class DiscriminatedUnionExtractor:
    """Detects and extracts discriminated union patterns from guard sets."""

    def extract_unions(
        self, guards: List[ExtractedTSGuard]
    ) -> List[DiscriminatedUnionInfo]:
        """Find discriminated union patterns in guard set."""
        union_map: Dict[Tuple[str, str], DiscriminatedUnionInfo] = {}

        for g in guards:
            if not isinstance(g.predicate, TSDiscriminatedUnionPredicate):
                # Check inside negations
                pred = g.predicate
                if isinstance(pred, TSNegationPredicate) and isinstance(
                    pred.child, TSDiscriminatedUnionPredicate
                ):
                    pred = pred.child
                else:
                    continue

            if isinstance(g.predicate, TSDiscriminatedUnionPredicate):
                dp = g.predicate
            elif isinstance(g.predicate, TSNegationPredicate) and isinstance(
                g.predicate.child, TSDiscriminatedUnionPredicate
            ):
                dp = g.predicate.child
            else:
                continue

            key = (dp.target_variable, dp.discriminant_property)
            if key not in union_map:
                union_map[key] = DiscriminatedUnionInfo(
                    object_variable=dp.target_variable,
                    discriminant_property=dp.discriminant_property,
                    variants=[],
                    source_locations=[],
                )

            info = union_map[key]
            if dp.discriminant_value not in info.variants:
                info.variants.append(dp.discriminant_value)
            info.source_locations.append(g.source_location)

        return list(union_map.values())

    def is_exhaustive(
        self,
        union_info: DiscriminatedUnionInfo,
        known_variants: Optional[Set[str]] = None,
    ) -> bool:
        """Check if the discriminated union pattern is exhaustive."""
        if known_variants is None:
            return False
        return set(union_info.variants) >= known_variants


# ---------------------------------------------------------------------------
# NarrowingTracker
# ---------------------------------------------------------------------------

@dataclass
class NarrowingScope:
    """Type narrowing state within a scope."""
    narrowed_types: Dict[str, List[str]] = field(default_factory=dict)
    eliminated_types: Dict[str, List[str]] = field(default_factory=dict)
    active_guards: List[ExtractedTSGuard] = field(default_factory=list)
    parent: Optional[NarrowingScope] = None

    def get_narrowed(self, variable: str) -> List[str]:
        """Get narrowed types for variable, including parent scopes."""
        types = list(self.narrowed_types.get(variable, []))
        if self.parent:
            types.extend(self.parent.get_narrowed(variable))
        return types

    def get_eliminated(self, variable: str) -> List[str]:
        """Get eliminated types for variable."""
        types = list(self.eliminated_types.get(variable, []))
        if self.parent:
            types.extend(self.parent.get_eliminated(variable))
        return types

    def narrow(self, variable: str, type_name: str) -> None:
        self.narrowed_types.setdefault(variable, []).append(type_name)

    def eliminate(self, variable: str, type_name: str) -> None:
        self.eliminated_types.setdefault(variable, []).append(type_name)

    def child_scope(self) -> NarrowingScope:
        return NarrowingScope(parent=self)


class NarrowingTracker:
    """Tracks type narrowing through control flow based on extracted guards."""

    def __init__(self) -> None:
        self._root_scope = NarrowingScope()
        self._current_scope = self._root_scope

    @property
    def current_scope(self) -> NarrowingScope:
        return self._current_scope

    def enter_scope(self) -> NarrowingScope:
        """Enter a new narrowing scope."""
        child = self._current_scope.child_scope()
        self._current_scope = child
        return child

    def exit_scope(self) -> NarrowingScope:
        """Exit current scope, returning to parent."""
        if self._current_scope.parent:
            self._current_scope = self._current_scope.parent
        return self._current_scope

    def apply_guard(
        self, guard: ExtractedTSGuard, branch: Polarity
    ) -> None:
        """Apply a guard's narrowing effect to the current scope."""
        pred = guard.predicate
        is_true_branch = branch == Polarity.TRUE_BRANCH

        if isinstance(pred, TSNegationPredicate):
            inner_guard = ExtractedTSGuard(
                pattern=guard.pattern,
                variables=guard.variables,
                predicate=pred.child if pred.child else pred,
                source_location=guard.source_location,
                polarity=guard.polarity,
                raw_source=guard.raw_source,
            )
            flipped = (
                Polarity.FALSE_BRANCH
                if branch == Polarity.TRUE_BRANCH
                else Polarity.TRUE_BRANCH
            )
            self.apply_guard(inner_guard, flipped)
            return

        if isinstance(pred, TSTypeTagPredicate):
            for type_name in pred.type_names:
                if is_true_branch:
                    self._current_scope.narrow(pred.target_variable, type_name)
                else:
                    self._current_scope.eliminate(pred.target_variable, type_name)

        elif isinstance(pred, TSNullityPredicate):
            if pred.is_null or pred.covers_both:
                if is_true_branch:
                    self._current_scope.narrow(pred.target_variable, "null")
                else:
                    self._current_scope.eliminate(pred.target_variable, "null")
            if pred.is_undefined or pred.covers_both:
                if is_true_branch:
                    self._current_scope.narrow(pred.target_variable, "undefined")
                else:
                    self._current_scope.eliminate(pred.target_variable, "undefined")

        elif isinstance(pred, TSTruthinessPredicate):
            if is_true_branch:
                self._current_scope.eliminate(pred.target_variable, "null")
                self._current_scope.eliminate(pred.target_variable, "undefined")
                self._current_scope.eliminate(pred.target_variable, "false")
                self._current_scope.eliminate(pred.target_variable, '""')
                self._current_scope.eliminate(pred.target_variable, "0")

        elif isinstance(pred, TSArrayCheckPredicate):
            if is_true_branch:
                self._current_scope.narrow(pred.target_variable, "Array")
            else:
                self._current_scope.eliminate(pred.target_variable, "Array")

        elif isinstance(pred, TSDiscriminatedUnionPredicate):
            variant = f"{pred.discriminant_property}={pred.discriminant_value}"
            if is_true_branch:
                self._current_scope.narrow(pred.target_variable, variant)
            else:
                self._current_scope.eliminate(pred.target_variable, variant)

        elif isinstance(pred, TSCallablePredicate):
            if is_true_branch:
                self._current_scope.narrow(pred.target_variable, "function")
            else:
                self._current_scope.eliminate(pred.target_variable, "function")

        elif isinstance(pred, TSConjunctionPredicate):
            for child in pred.children:
                child_guard = ExtractedTSGuard(
                    pattern=guard.pattern,
                    variables=child.variables,
                    predicate=child,
                    source_location=guard.source_location,
                    polarity=guard.polarity,
                    raw_source="",
                )
                self.apply_guard(child_guard, branch)

        self._current_scope.active_guards.append(guard)

    def get_narrowing_at(self, variable: str) -> Dict[str, List[str]]:
        """Get current narrowing state for a variable."""
        return {
            "narrowed_to": self._current_scope.get_narrowed(variable),
            "eliminated": self._current_scope.get_eliminated(variable),
        }


# ---------------------------------------------------------------------------
# OptionalChainingExtractor
# ---------------------------------------------------------------------------

class OptionalChainingExtractor:
    """Extracts and analyzes optional chaining and nullish coalescing patterns."""

    OPTIONAL_CHAIN_RE = re.compile(r"(\w+(?:\.\w+)*)(\?\.\w+(?:\.\w+)*)+")
    NULLISH_COALESCE_RE = re.compile(r"(\w+(?:\??\.\w+)*)\s*\?\?\s*")

    def extract_from_source(self, source: str, filename: str = "") -> List[ExtractedTSGuard]:
        """Extract optional chaining guards from source."""
        guards: List[ExtractedTSGuard] = []
        lines = source.split("\n")

        for line_num, line in enumerate(lines, 1):
            loc = SourceLocation(
                filename=filename, line=line_num, col=0,
                end_line=line_num, end_col=len(line),
            )

            # Optional chaining
            for m in self.OPTIONAL_CHAIN_RE.finditer(line):
                base = m.group(1)
                full_expr = m.group(0)
                segments = self._parse_chain_segments(full_expr)

                pred = TSOptionalChainPredicate(
                    target_variable=base,
                    chain_path=full_expr,
                    guarded_segments=segments,
                    variables=[base],
                )

                guards.append(ExtractedTSGuard(
                    pattern=TSGuardPattern.OptionalChaining,
                    variables=[base],
                    predicate=pred,
                    source_location=loc,
                    polarity=Polarity.TRUE_BRANCH,
                    raw_source=full_expr,
                    confidence=0.8,
                ))

                # Generate implicit null guards for each segment
                for seg in segments:
                    null_pred = TSNullityPredicate(
                        target_variable=seg,
                        is_null=True,
                        is_undefined=True,
                        covers_both=True,
                        variables=[seg.split(".")[0]],
                    )
                    guards.append(ExtractedTSGuard(
                        pattern=TSGuardPattern.Nullity,
                        variables=[seg.split(".")[0]],
                        predicate=null_pred.negate(),
                        source_location=loc,
                        polarity=Polarity.TRUE_BRANCH,
                        raw_source=f"{seg} != null",
                        confidence=0.6,
                    ))

            # Nullish coalescing
            for m in self.NULLISH_COALESCE_RE.finditer(line):
                var = m.group(1)
                pred = TSNullishCoalescePredicate(
                    target_variable=var,
                    default_expr="<default>",
                    variables=[var.split(".")[0].rstrip("?")],
                )
                guards.append(ExtractedTSGuard(
                    pattern=TSGuardPattern.NullishCoalescing,
                    variables=[var.split(".")[0].rstrip("?")],
                    predicate=pred,
                    source_location=loc,
                    polarity=Polarity.TRUE_BRANCH,
                    raw_source=m.group(0).strip(),
                    confidence=0.7,
                ))

        return guards

    def _parse_chain_segments(self, expr: str) -> List[str]:
        """Parse x?.a?.b into guarded segments: ['x', 'x.a']."""
        segments: List[str] = []
        parts = expr.split("?.")
        current = parts[0]
        segments.append(current)
        for part in parts[1:]:
            current = current + "." + part
            segments.append(current.replace("?.", "."))
        return segments[:-1]

    def analyze_safety(
        self, guards: List[ExtractedTSGuard]
    ) -> Dict[str, bool]:
        """Analyze which variables are safely guarded against null."""
        safety: Dict[str, bool] = {}

        for g in guards:
            if isinstance(g.predicate, TSOptionalChainPredicate):
                safety[g.predicate.target_variable] = True
            elif isinstance(g.predicate, TSNullishCoalescePredicate):
                safety[g.predicate.target_variable] = True
            elif isinstance(g.predicate, TSNullityPredicate):
                if isinstance(g.predicate, TSNegationPredicate):
                    pass
                else:
                    if not g.predicate.is_null and not g.predicate.is_undefined:
                        safety[g.predicate.target_variable] = True

        return safety


# ---------------------------------------------------------------------------
# Guard ranking
# ---------------------------------------------------------------------------

@dataclass
class RankedTSGuard:
    """A guard with rank score."""
    guard: ExtractedTSGuard
    score: float
    frequency: int = 1
    information_content: float = 0.0


class TSGuardRanker:
    """Ranks TypeScript guards for contract discovery seeding."""

    PATTERN_WEIGHTS: Dict[TSGuardPattern, float] = {
        TSGuardPattern.TypeOf: 1.0,
        TSGuardPattern.InstanceOf: 1.0,
        TSGuardPattern.Nullity: 0.9,
        TSGuardPattern.StrictEquality: 0.7,
        TSGuardPattern.LooseEquality: 0.6,
        TSGuardPattern.PropertyIn: 0.85,
        TSGuardPattern.ArrayIsArray: 0.9,
        TSGuardPattern.Truthiness: 0.3,
        TSGuardPattern.DiscriminatedUnion: 0.95,
        TSGuardPattern.TypePredicate: 0.9,
        TSGuardPattern.OptionalChaining: 0.4,
        TSGuardPattern.NullishCoalescing: 0.35,
        TSGuardPattern.Comparison: 0.6,
        TSGuardPattern.Negation: 0.4,
        TSGuardPattern.Conjunction: 0.7,
        TSGuardPattern.Disjunction: 0.5,
    }

    def rank(self, guards: List[ExtractedTSGuard]) -> List[RankedTSGuard]:
        freq_map = self._compute_frequencies(guards)
        ranked: List[RankedTSGuard] = []

        for g in guards:
            info = self._information_content(g)
            key = self._signature(g)
            freq = freq_map.get(key, 1)
            freq_bonus = min(1.0 + 0.1 * (freq - 1), 1.5)
            strictness_bonus = 0.1 if g.is_strict else 0.0
            score = (info + strictness_bonus) * freq_bonus * g.confidence
            ranked.append(RankedTSGuard(
                guard=g, score=score, frequency=freq,
                information_content=info,
            ))

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked

    def top_k(
        self, guards: List[ExtractedTSGuard], k: int = 10
    ) -> List[ExtractedTSGuard]:
        return [r.guard for r in self.rank(guards)[:k]]

    def _information_content(self, guard: ExtractedTSGuard) -> float:
        w = self.PATTERN_WEIGHTS.get(guard.pattern, 0.5)
        var_bonus = min(len(guard.variables) * 0.1, 0.3)
        return w + var_bonus

    def _compute_frequencies(
        self, guards: List[ExtractedTSGuard]
    ) -> Dict[str, int]:
        freq: Dict[str, int] = {}
        for g in guards:
            key = self._signature(g)
            freq[key] = freq.get(key, 0) + 1
        return freq

    @staticmethod
    def _signature(guard: ExtractedTSGuard) -> str:
        return (
            f"{guard.pattern.name}:{guard.predicate.kind.name}:"
            f"{','.join(sorted(guard.variables))}"
        )


# ---------------------------------------------------------------------------
# Guard serialization
# ---------------------------------------------------------------------------

class TSGuardSerializer:
    """Serialize/deserialize TS guards."""

    def to_dict(self, guard: ExtractedTSGuard) -> Dict[str, Any]:
        return {
            "pattern": guard.pattern.name,
            "variables": guard.variables,
            "predicate": self._pred_to_dict(guard.predicate),
            "source_location": {
                "filename": guard.source_location.filename,
                "line": guard.source_location.line,
                "col": guard.source_location.col,
                "end_line": guard.source_location.end_line,
                "end_col": guard.source_location.end_col,
            },
            "polarity": guard.polarity.name,
            "raw_source": guard.raw_source,
            "confidence": guard.confidence,
            "is_strict": guard.is_strict,
        }

    def to_json(self, guards: List[ExtractedTSGuard], indent: int = 2) -> str:
        return json.dumps(
            [self.to_dict(g) for g in guards], indent=indent
        )

    def _pred_to_dict(self, pred: TSPredicateTemplate) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "kind": pred.kind.name,
            "variables": pred.variables,
        }

        if isinstance(pred, TSTypeTagPredicate):
            d["target_variable"] = pred.target_variable
            d["type_names"] = list(pred.type_names)
            d["tag_source"] = pred.tag_source
        elif isinstance(pred, TSNullityPredicate):
            d["target_variable"] = pred.target_variable
            d["is_null"] = pred.is_null
            d["is_undefined"] = pred.is_undefined
            d["covers_both"] = pred.covers_both
        elif isinstance(pred, TSTruthinessPredicate):
            d["target_variable"] = pred.target_variable
            d["narrows_out"] = pred.narrows_out
        elif isinstance(pred, TSHasAttrPredicate):
            d["target_variable"] = pred.target_variable
            d["property_name"] = pred.property_name
        elif isinstance(pred, TSComparisonPredicate):
            d["left_expr"] = pred.left_expr
            d["op"] = pred.op.value
            d["right_expr"] = pred.right_expr
            d["is_strict"] = pred.is_strict
        elif isinstance(pred, TSCallablePredicate):
            d["target_variable"] = pred.target_variable
        elif isinstance(pred, TSDiscriminatedUnionPredicate):
            d["target_variable"] = pred.target_variable
            d["discriminant_property"] = pred.discriminant_property
            d["discriminant_value"] = pred.discriminant_value
        elif isinstance(pred, TSTypePredicatePredicate):
            d["target_variable"] = pred.target_variable
            d["predicate_type"] = pred.predicate_type
            d["function_name"] = pred.function_name
            d["asserts"] = pred.asserts
        elif isinstance(pred, TSOptionalChainPredicate):
            d["target_variable"] = pred.target_variable
            d["chain_path"] = pred.chain_path
            d["guarded_segments"] = pred.guarded_segments
        elif isinstance(pred, TSNullishCoalescePredicate):
            d["target_variable"] = pred.target_variable
            d["default_expr"] = pred.default_expr
        elif isinstance(pred, TSArrayCheckPredicate):
            d["target_variable"] = pred.target_variable
        elif isinstance(pred, TSIdentityPredicate):
            d["left_variable"] = pred.left_variable
            d["right_expr"] = pred.right_expr
            d["is_strict"] = pred.is_strict
            d["is_positive"] = pred.is_positive
        elif isinstance(pred, TSConjunctionPredicate):
            d["children"] = [self._pred_to_dict(c) for c in pred.children]
        elif isinstance(pred, TSDisjunctionPredicate):
            d["children"] = [self._pred_to_dict(c) for c in pred.children]
        elif isinstance(pred, TSNegationPredicate):
            d["child"] = self._pred_to_dict(pred.child) if pred.child else None

        return d


# ---------------------------------------------------------------------------
# Guard statistics
# ---------------------------------------------------------------------------

@dataclass
class TSGuardStatistics:
    """Summary statistics for TypeScript guards."""
    total_count: int = 0
    by_pattern: Dict[str, int] = field(default_factory=dict)
    by_predicate_kind: Dict[str, int] = field(default_factory=dict)
    unique_variables: Set[str] = field(default_factory=set)
    strict_count: int = 0
    loose_count: int = 0
    avg_confidence: float = 0.0
    discriminated_unions: int = 0
    optional_chains: int = 0
    type_predicates: int = 0
    most_guarded_variables: List[Tuple[str, int]] = field(default_factory=list)

    @staticmethod
    def compute(guards: List[ExtractedTSGuard]) -> TSGuardStatistics:
        stats = TSGuardStatistics()
        stats.total_count = len(guards)
        var_counts: Dict[str, int] = {}
        conf_sum = 0.0

        for g in guards:
            pname = g.pattern.name
            stats.by_pattern[pname] = stats.by_pattern.get(pname, 0) + 1

            kname = g.predicate.kind.name
            stats.by_predicate_kind[kname] = stats.by_predicate_kind.get(kname, 0) + 1

            for v in g.variables:
                stats.unique_variables.add(v)
                var_counts[v] = var_counts.get(v, 0) + 1

            if g.is_strict:
                stats.strict_count += 1
            else:
                stats.loose_count += 1

            conf_sum += g.confidence

            if g.pattern == TSGuardPattern.DiscriminatedUnion:
                stats.discriminated_unions += 1
            elif g.pattern == TSGuardPattern.OptionalChaining:
                stats.optional_chains += 1
            elif g.pattern == TSGuardPattern.TypePredicate:
                stats.type_predicates += 1

        if guards:
            stats.avg_confidence = conf_sum / len(guards)

        stats.most_guarded_variables = sorted(
            var_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:20]

        return stats


# ---------------------------------------------------------------------------
# Guard filter
# ---------------------------------------------------------------------------

class TSGuardFilter:
    """Utilities for filtering TS guards."""

    @staticmethod
    def by_pattern(
        guards: List[ExtractedTSGuard], pattern: TSGuardPattern
    ) -> List[ExtractedTSGuard]:
        return [g for g in guards if g.pattern == pattern]

    @staticmethod
    def by_variable(
        guards: List[ExtractedTSGuard], variable: str
    ) -> List[ExtractedTSGuard]:
        return [g for g in guards if variable in g.variables]

    @staticmethod
    def by_strictness(
        guards: List[ExtractedTSGuard], strict: bool = True
    ) -> List[ExtractedTSGuard]:
        return [g for g in guards if g.is_strict == strict]

    @staticmethod
    def type_narrowing_guards(
        guards: List[ExtractedTSGuard],
    ) -> List[ExtractedTSGuard]:
        """Return guards that perform type narrowing."""
        narrowing_patterns = {
            TSGuardPattern.TypeOf,
            TSGuardPattern.InstanceOf,
            TSGuardPattern.ArrayIsArray,
            TSGuardPattern.DiscriminatedUnion,
            TSGuardPattern.TypePredicate,
        }
        return [g for g in guards if g.pattern in narrowing_patterns]

    @staticmethod
    def null_safety_guards(
        guards: List[ExtractedTSGuard],
    ) -> List[ExtractedTSGuard]:
        """Return guards related to null/undefined safety."""
        null_patterns = {
            TSGuardPattern.Nullity,
            TSGuardPattern.OptionalChaining,
            TSGuardPattern.NullishCoalescing,
        }
        return [g for g in guards if g.pattern in null_patterns]

    @staticmethod
    def structural_guards(
        guards: List[ExtractedTSGuard],
    ) -> List[ExtractedTSGuard]:
        return [g for g in guards if g.pattern == TSGuardPattern.PropertyIn]

    @staticmethod
    def by_confidence(
        guards: List[ExtractedTSGuard], min_confidence: float = 0.5
    ) -> List[ExtractedTSGuard]:
        return [g for g in guards if g.confidence >= min_confidence]


# ---------------------------------------------------------------------------
# Guard dependency analysis
# ---------------------------------------------------------------------------

class TSGuardDependencyAnalyzer:
    """Analyze dependencies between TS guards."""

    def compute_dependencies(
        self, guards: List[ExtractedTSGuard]
    ) -> Dict[int, Set[int]]:
        deps: Dict[int, Set[int]] = {i: set() for i in range(len(guards))}
        var_to_indices: Dict[str, List[int]] = {}

        for i, g in enumerate(guards):
            for v in g.variables:
                var_to_indices.setdefault(v, []).append(i)

        for indices in var_to_indices.values():
            for i in indices:
                for j in indices:
                    if i != j:
                        deps[i].add(j)

        return deps

    def connected_components(
        self, guards: List[ExtractedTSGuard]
    ) -> List[List[ExtractedTSGuard]]:
        deps = self.compute_dependencies(guards)
        visited: Set[int] = set()
        components: List[List[ExtractedTSGuard]] = []

        def dfs(idx: int, comp: List[int]) -> None:
            if idx in visited:
                return
            visited.add(idx)
            comp.append(idx)
            for neighbor in deps.get(idx, set()):
                dfs(neighbor, comp)

        for i in range(len(guards)):
            if i not in visited:
                comp: List[int] = []
                dfs(i, comp)
                components.append([guards[j] for j in sorted(comp)])

        return components


# ---------------------------------------------------------------------------
# Predicate pretty-printer
# ---------------------------------------------------------------------------

class TSPredicatePrinter:
    """Pretty-print TS predicate templates."""

    def format(self, pred: TSPredicateTemplate, indent: int = 0) -> str:
        prefix = "  " * indent

        if isinstance(pred, TSTypeTagPredicate):
            src = f" [{pred.tag_source}]" if pred.tag_source else ""
            return f"{prefix}TypeTag({pred.target_variable} :: {' | '.join(pred.type_names)}{src})"

        if isinstance(pred, TSNullityPredicate):
            parts = []
            if pred.is_null:
                parts.append("null")
            if pred.is_undefined:
                parts.append("undefined")
            both = " (covers both)" if pred.covers_both else ""
            return f"{prefix}Nullity({pred.target_variable} == {' | '.join(parts)}{both})"

        if isinstance(pred, TSTruthinessPredicate):
            return f"{prefix}Truthy({pred.target_variable})"

        if isinstance(pred, TSHasAttrPredicate):
            return f"{prefix}HasProp('{pred.property_name}' in {pred.target_variable})"

        if isinstance(pred, TSComparisonPredicate):
            strict = "strict" if pred.is_strict else "loose"
            return f"{prefix}Compare({pred.left_expr} {pred.op.value} {pred.right_expr} [{strict}])"

        if isinstance(pred, TSCallablePredicate):
            return f"{prefix}Callable({pred.target_variable})"

        if isinstance(pred, TSDiscriminatedUnionPredicate):
            return f"{prefix}Discriminant({pred.target_variable}.{pred.discriminant_property} === '{pred.discriminant_value}')"

        if isinstance(pred, TSTypePredicatePredicate):
            asserts = " [asserts]" if pred.asserts else ""
            return f"{prefix}TypePred({pred.function_name}: {pred.target_variable} is {pred.predicate_type}{asserts})"

        if isinstance(pred, TSOptionalChainPredicate):
            return f"{prefix}OptChain({pred.chain_path})"

        if isinstance(pred, TSNullishCoalescePredicate):
            return f"{prefix}Nullish({pred.target_variable} ?? {pred.default_expr})"

        if isinstance(pred, TSArrayCheckPredicate):
            return f"{prefix}ArrayCheck({pred.target_variable})"

        if isinstance(pred, TSConjunctionPredicate):
            parts = [self.format(c, indent + 1) for c in pred.children]
            return f"{prefix}AND(\n" + ",\n".join(parts) + f"\n{prefix})"

        if isinstance(pred, TSDisjunctionPredicate):
            parts = [self.format(c, indent + 1) for c in pred.children]
            return f"{prefix}OR(\n" + ",\n".join(parts) + f"\n{prefix})"

        if isinstance(pred, TSNegationPredicate):
            inner = self.format(pred.child, indent + 1) if pred.child else "?"
            return f"{prefix}NOT(\n{inner}\n{prefix})"

        return f"{prefix}{pred.kind.name}({', '.join(pred.variables)})"


# ---------------------------------------------------------------------------
# Compound / Convenience API
# ---------------------------------------------------------------------------

class FullTSGuardExtractor:
    """Orchestrates all TS sub-extractors."""

    def __init__(self, filename: str = "", source: str = "") -> None:
        self.filename = filename
        self.source = source
        self._main = TypeScriptGuardExtractor(filename, source)
        self._normalizer = TypeScriptGuardNormalizer()
        self._ranker = TSGuardRanker()
        self._disc_extractor = DiscriminatedUnionExtractor()
        self._optional_extractor = OptionalChainingExtractor()
        self._narrowing = NarrowingTracker()

    def extract_all(
        self,
        source: Optional[str] = None,
        ast_data: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractedTSGuard]:
        """Extract, normalize, and deduplicate all guards."""
        src = source or self.source
        guards = self._main.extract_combined(src, ast_data)

        # Also extract optional chaining patterns
        opt_guards = self._optional_extractor.extract_from_source(
            src, self.filename
        )
        guards.extend(opt_guards)

        guards = self._normalizer.normalize_all(guards)
        return guards

    def extract_ranked(
        self,
        source: Optional[str] = None,
        ast_data: Optional[Dict[str, Any]] = None,
        top_k: int = 50,
    ) -> List[RankedTSGuard]:
        guards = self.extract_all(source, ast_data)
        return self._ranker.rank(guards)[:top_k]

    def extract_with_narrowing(
        self,
        source: Optional[str] = None,
        ast_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[ExtractedTSGuard], NarrowingTracker]:
        """Extract guards and set up narrowing tracker."""
        guards = self.extract_all(source, ast_data)
        tracker = NarrowingTracker()
        for g in guards:
            tracker.apply_guard(g, g.polarity)
        return guards, tracker

    def extract_discriminated_unions(
        self,
        source: Optional[str] = None,
        ast_data: Optional[Dict[str, Any]] = None,
    ) -> List[DiscriminatedUnionInfo]:
        """Extract discriminated union patterns."""
        guards = self.extract_all(source, ast_data)
        return self._disc_extractor.extract_unions(guards)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def extract_ts_guards(
    source: str, filename: str = "<string>"
) -> List[ExtractedTSGuard]:
    """Extract all guards from TypeScript source code."""
    extractor = FullTSGuardExtractor(filename=filename, source=source)
    return extractor.extract_all(source)


def extract_and_rank_ts(
    source: str, filename: str = "<string>", top_k: int = 50
) -> List[RankedTSGuard]:
    """Extract and rank guards from TypeScript source."""
    extractor = FullTSGuardExtractor(filename=filename, source=source)
    return extractor.extract_ranked(source, top_k=top_k)


def extract_discriminated_unions(
    source: str, filename: str = "<string>"
) -> List[DiscriminatedUnionInfo]:
    """Extract discriminated union patterns from TypeScript source."""
    extractor = FullTSGuardExtractor(filename=filename, source=source)
    return extractor.extract_discriminated_unions(source)


def print_ts_guards(source: str, filename: str = "<string>") -> None:
    """Pretty-print all extracted TS guards."""
    guards = extract_ts_guards(source, filename)
    printer = TSPredicatePrinter()
    for i, g in enumerate(guards):
        print(f"Guard #{i + 1} [{g.pattern.name}] @ {g.source_location.line}:{g.source_location.col}")
        print(f"  Polarity: {g.polarity.name}")
        print(f"  Variables: {g.variables}")
        print(f"  Strict: {g.is_strict}")
        print(f"  Raw: {g.raw_source}")
        print(f"  Predicate: {printer.format(g.predicate)}")
        print()
