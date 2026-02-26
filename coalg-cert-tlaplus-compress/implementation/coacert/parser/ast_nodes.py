"""Complete typed AST node hierarchy for TLA-lite.

Every syntactic construct in the TLA-lite fragment is represented by a
dataclass node.  Nodes carry a ``source_location`` field so error
messages, pretty-printers, and tooling can reference exact positions.

A :class:`ASTVisitor` base class provides the Visitor pattern for
traversal and transformation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeVar

from .source_map import SourceLocation, UNKNOWN_LOCATION

T = TypeVar("T")


# ============================================================================
# Operator enumeration
# ============================================================================

class Operator(Enum):
    """All operators expressible in TLA-lite."""

    # ── Logical ──────────────────────────────────────────────────
    LAND = auto()
    LOR = auto()
    LNOT = auto()
    IMPLIES = auto()
    EQUIV = auto()

    # ── Equality / comparison ────────────────────────────────────
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LEQ = auto()
    GEQ = auto()

    # ── Arithmetic ───────────────────────────────────────────────
    PLUS = auto()
    MINUS = auto()
    TIMES = auto()
    DIV = auto()
    MOD = auto()
    UMINUS = auto()
    RANGE = auto()       # a..b

    # ── Set operators ────────────────────────────────────────────
    IN = auto()
    NOTIN = auto()
    UNION = auto()
    INTERSECT = auto()
    SETDIFF = auto()
    SUBSETEQ = auto()
    CROSS = auto()       # \X
    POWERSET = auto()    # SUBSET S
    UNION_ALL = auto()   # UNION S

    # ── Function / record ────────────────────────────────────────
    FUNC_APPLY = auto()
    FUNC_EXCEPT = auto()
    DOMAIN_OP = auto()
    COLON_GT = auto()    # :>
    AT_AT = auto()       # @@

    # ── Tuple / sequence ─────────────────────────────────────────
    TUPLE_ACCESS = auto()  # t[i]

    # ── String ───────────────────────────────────────────────────
    STRING_CONCAT = auto()

    # ── Temporal ─────────────────────────────────────────────────
    PRIME = auto()
    ALWAYS = auto()
    EVENTUALLY = auto()
    LEADS_TO = auto()
    ENABLED_OP = auto()
    UNCHANGED_OP = auto()

    # ── Action ───────────────────────────────────────────────────
    STUTTER = auto()     # [A]_v
    NO_STUTTER = auto()  # <A>_v
    WF = auto()
    SF = auto()

    # ── Sequence operators ───────────────────────────────────────
    APPEND = auto()
    HEAD = auto()
    TAIL = auto()
    LEN = auto()
    SUBSEQ = auto()
    SEQ = auto()         # Seq(S)
    CONCAT = auto()      # s \o t

    def __repr__(self) -> str:
        return f"Operator.{self.name}"


# ============================================================================
# Base AST node
# ============================================================================

@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    source_location: SourceLocation = field(
        default_factory=lambda: UNKNOWN_LOCATION, repr=False, compare=False
    )

    def accept(self, visitor: ASTVisitor[T]) -> T:
        method_name = f"visit_{type(self).__name__}"
        method = getattr(visitor, method_name, None)
        if method:
            return method(self)
        return visitor.generic_visit(self)


# ============================================================================
# Type annotations
# ============================================================================

@dataclass
class TypeAnnotation(ASTNode):
    """Base for type annotation nodes."""
    pass


@dataclass
class IntType(TypeAnnotation):
    """The integer type."""
    pass


@dataclass
class BoolType(TypeAnnotation):
    """The boolean type."""
    pass


@dataclass
class StringType(TypeAnnotation):
    """The string type."""
    pass


@dataclass
class SetType(TypeAnnotation):
    """Set type parameterised by element type."""
    element_type: TypeAnnotation = field(default_factory=lambda: AnyType())


@dataclass
class FunctionType(TypeAnnotation):
    """Function type [D -> R]."""
    domain_type: TypeAnnotation = field(default_factory=lambda: AnyType())
    range_type: TypeAnnotation = field(default_factory=lambda: AnyType())


@dataclass
class TupleType(TypeAnnotation):
    """Tuple (Cartesian product) type."""
    element_types: List[TypeAnnotation] = field(default_factory=list)


@dataclass
class RecordType(TypeAnnotation):
    """Record type [field1: T1, field2: T2, …]."""
    field_types: Dict[str, TypeAnnotation] = field(default_factory=dict)


@dataclass
class SequenceType(TypeAnnotation):
    """Sequence type Seq(T)."""
    element_type: TypeAnnotation = field(default_factory=lambda: AnyType())


@dataclass
class AnyType(TypeAnnotation):
    """Wildcard / unknown type used during inference."""
    pass


@dataclass
class OperatorType(TypeAnnotation):
    """Type of an operator: (T1, T2, …) -> R."""
    param_types: List[TypeAnnotation] = field(default_factory=list)
    return_type: TypeAnnotation = field(default_factory=lambda: AnyType())


# ============================================================================
# Expressions
# ============================================================================

@dataclass
class Expression(ASTNode):
    """Abstract base for all expression nodes."""
    inferred_type: Optional[TypeAnnotation] = field(
        default=None, repr=False, compare=False
    )


@dataclass
class IntLiteral(Expression):
    """An integer constant."""
    value: int = 0


@dataclass
class BoolLiteral(Expression):
    """TRUE or FALSE."""
    value: bool = True


@dataclass
class StringLiteral(Expression):
    """A string constant."""
    value: str = ""


@dataclass
class Identifier(Expression):
    """A plain identifier reference."""
    name: str = ""


@dataclass
class PrimedIdentifier(Expression):
    """A primed variable reference (e.g. x')."""
    name: str = ""


@dataclass
class OperatorApplication(Expression):
    """Application of a built-in or user-defined operator."""
    operator: Operator = Operator.PLUS
    operands: List[Expression] = field(default_factory=list)
    operator_name: Optional[str] = None  # for user-defined ops


@dataclass
class SetEnumeration(Expression):
    """Explicit set: {e1, e2, …}."""
    elements: List[Expression] = field(default_factory=list)


@dataclass
class SetComprehension(Expression):
    """{x \\in S : P(x)} or {e : x \\in S}."""
    variable: str = ""
    set_expr: Optional[Expression] = None
    predicate: Optional[Expression] = None
    map_expr: Optional[Expression] = None  # for {e : x \\in S}


@dataclass
class FunctionConstruction(Expression):
    """[x \\in S |-> e(x)]."""
    variable: str = ""
    set_expr: Optional[Expression] = None
    body: Optional[Expression] = None


@dataclass
class FunctionApplication(Expression):
    """f[e]."""
    function: Optional[Expression] = None
    argument: Optional[Expression] = None


@dataclass
class RecordConstruction(Expression):
    """[field1 |-> e1, field2 |-> e2, …]."""
    fields: List[tuple[str, Expression]] = field(default_factory=list)


@dataclass
class RecordAccess(Expression):
    """r.field."""
    record: Optional[Expression] = None
    field_name: str = ""


@dataclass
class TupleLiteral(Expression):
    """<<e1, e2, …>>."""
    elements: List[Expression] = field(default_factory=list)


@dataclass
class SequenceLiteral(Expression):
    """Same as TupleLiteral but semantically used as a sequence."""
    elements: List[Expression] = field(default_factory=list)


@dataclass
class QuantifiedExpr(Expression):
    """\\A x \\in S : P(x)  or  \\E x \\in S : P(x)."""
    quantifier: str = "forall"  # "forall" | "exists"
    variables: List[tuple[str, Expression]] = field(default_factory=list)
    body: Optional[Expression] = None


@dataclass
class IfThenElse(Expression):
    """IF cond THEN e1 ELSE e2."""
    condition: Optional[Expression] = None
    then_expr: Optional[Expression] = None
    else_expr: Optional[Expression] = None


@dataclass
class LetIn(Expression):
    """LET defs IN expr."""
    definitions: List[Definition] = field(default_factory=list)
    body: Optional[Expression] = None


@dataclass
class CaseArm(ASTNode):
    """A single arm of a CASE expression."""
    condition: Optional[Expression] = None
    value: Optional[Expression] = None


@dataclass
class CaseExpr(Expression):
    """CASE p1 -> e1 [] p2 -> e2 … [] OTHER -> eN."""
    arms: List[CaseArm] = field(default_factory=list)
    other: Optional[Expression] = None


@dataclass
class UnchangedExpr(Expression):
    """UNCHANGED <<v1, v2, …>>  or  UNCHANGED v."""
    variables: List[Expression] = field(default_factory=list)


@dataclass
class ExceptExpr(Expression):
    """[f EXCEPT ![a] = e, ![b] = e2, …]."""
    base: Optional[Expression] = None
    substitutions: List[tuple[List[Expression], Expression]] = field(
        default_factory=list
    )


@dataclass
class ChooseExpr(Expression):
    """CHOOSE x \\in S : P(x)."""
    variable: str = ""
    set_expr: Optional[Expression] = None
    predicate: Optional[Expression] = None


@dataclass
class DomainExpr(Expression):
    """DOMAIN f."""
    expr: Optional[Expression] = None


# ============================================================================
# Definitions
# ============================================================================

@dataclass
class Definition(ASTNode):
    """Base for definition nodes."""
    pass


@dataclass
class OperatorDef(Definition):
    """Op(p1, p2, …) == body."""
    name: str = ""
    params: List[str] = field(default_factory=list)
    body: Optional[Expression] = None
    is_local: bool = False


@dataclass
class FunctionDef(Definition):
    """f[x \\in S] == body."""
    name: str = ""
    variable: str = ""
    set_expr: Optional[Expression] = None
    body: Optional[Expression] = None
    is_local: bool = False


@dataclass
class VariableDecl(Definition):
    """VARIABLE(S) v1, v2, …."""
    names: List[str] = field(default_factory=list)


@dataclass
class ConstantDecl(Definition):
    """CONSTANT(S) c1, c2, …."""
    names: List[str] = field(default_factory=list)


@dataclass
class Assumption(Definition):
    """ASSUME expr."""
    expr: Optional[Expression] = None


@dataclass
class Theorem(Definition):
    """THEOREM expr."""
    expr: Optional[Expression] = None


@dataclass
class InstanceDef(Definition):
    """INSTANCE ModName WITH p1 <- e1, …."""
    module_name: str = ""
    substitutions: List[tuple[str, Expression]] = field(default_factory=list)
    is_local: bool = False


# ============================================================================
# Action nodes
# ============================================================================

@dataclass
class ActionExpr(Expression):
    """An action-level expression wrapping a state-level body."""
    body: Optional[Expression] = None


@dataclass
class StutteringAction(Expression):
    """[A]_v  or  <A>_v."""
    action: Optional[Expression] = None
    variables: Optional[Expression] = None
    is_angle: bool = False  # True => <A>_v ; False => [A]_v


@dataclass
class FairnessExpr(Expression):
    """WF_v(A) or SF_v(A)."""
    kind: str = "WF"  # "WF" | "SF"
    variables: Optional[Expression] = None
    action: Optional[Expression] = None


# ============================================================================
# Temporal nodes
# ============================================================================

@dataclass
class AlwaysExpr(Expression):
    """[]P."""
    expr: Optional[Expression] = None


@dataclass
class EventuallyExpr(Expression):
    """<>P."""
    expr: Optional[Expression] = None


@dataclass
class LeadsToExpr(Expression):
    """P ~> Q."""
    left: Optional[Expression] = None
    right: Optional[Expression] = None


@dataclass
class TemporalForallExpr(Expression):
    """\\AA x : P."""
    variable: str = ""
    body: Optional[Expression] = None


@dataclass
class TemporalExistsExpr(Expression):
    """\\EE x : P."""
    variable: str = ""
    body: Optional[Expression] = None


# ============================================================================
# Property nodes
# ============================================================================

@dataclass
class Property(ASTNode):
    """Base class for specification properties."""
    name: str = ""


@dataclass
class InvariantProperty(Property):
    """A state invariant."""
    expr: Optional[Expression] = None


@dataclass
class TemporalProperty(Property):
    """A general temporal property."""
    expr: Optional[Expression] = None


@dataclass
class SafetyProperty(Property):
    """A safety property (something bad never happens)."""
    expr: Optional[Expression] = None


@dataclass
class LivenessProperty(Property):
    """A liveness property (something good eventually happens)."""
    expr: Optional[Expression] = None


# ============================================================================
# Module
# ============================================================================

@dataclass
class Module(ASTNode):
    """Top-level module node."""
    name: str = ""
    extends: List[str] = field(default_factory=list)
    constants: List[ConstantDecl] = field(default_factory=list)
    variables: List[VariableDecl] = field(default_factory=list)
    definitions: List[Definition] = field(default_factory=list)
    assumptions: List[Assumption] = field(default_factory=list)
    theorems: List[Theorem] = field(default_factory=list)
    properties: List[Property] = field(default_factory=list)
    instances: List[InstanceDef] = field(default_factory=list)


# ============================================================================
# Visitor pattern
# ============================================================================

class ASTVisitor(ABC, Generic[T]):
    """Base visitor with a visit method per AST node type.

    Override any ``visit_NodeName`` method you need; unhandled nodes fall
    through to :meth:`generic_visit`.
    """

    def generic_visit(self, node: ASTNode) -> T:
        """Default handler — visits all child nodes."""
        result: Any = None
        for child in self._iter_children(node):
            result = child.accept(self)
        return result

    @staticmethod
    def _iter_children(node: ASTNode):
        """Yield direct AST-node children of *node*."""
        for fld in node.__dataclass_fields__:
            val = getattr(node, fld, None)
            if isinstance(val, ASTNode):
                yield val
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, ASTNode):
                        yield item
                    elif isinstance(item, tuple):
                        for sub in item:
                            if isinstance(sub, ASTNode):
                                yield sub

    # ── Module ──────────────────────────────────────────────────────
    def visit_Module(self, node: Module) -> T:
        return self.generic_visit(node)

    # ── Expressions ─────────────────────────────────────────────────
    def visit_IntLiteral(self, node: IntLiteral) -> T:
        return self.generic_visit(node)

    def visit_BoolLiteral(self, node: BoolLiteral) -> T:
        return self.generic_visit(node)

    def visit_StringLiteral(self, node: StringLiteral) -> T:
        return self.generic_visit(node)

    def visit_Identifier(self, node: Identifier) -> T:
        return self.generic_visit(node)

    def visit_PrimedIdentifier(self, node: PrimedIdentifier) -> T:
        return self.generic_visit(node)

    def visit_OperatorApplication(self, node: OperatorApplication) -> T:
        return self.generic_visit(node)

    def visit_SetEnumeration(self, node: SetEnumeration) -> T:
        return self.generic_visit(node)

    def visit_SetComprehension(self, node: SetComprehension) -> T:
        return self.generic_visit(node)

    def visit_FunctionConstruction(self, node: FunctionConstruction) -> T:
        return self.generic_visit(node)

    def visit_FunctionApplication(self, node: FunctionApplication) -> T:
        return self.generic_visit(node)

    def visit_RecordConstruction(self, node: RecordConstruction) -> T:
        return self.generic_visit(node)

    def visit_RecordAccess(self, node: RecordAccess) -> T:
        return self.generic_visit(node)

    def visit_TupleLiteral(self, node: TupleLiteral) -> T:
        return self.generic_visit(node)

    def visit_SequenceLiteral(self, node: SequenceLiteral) -> T:
        return self.generic_visit(node)

    def visit_QuantifiedExpr(self, node: QuantifiedExpr) -> T:
        return self.generic_visit(node)

    def visit_IfThenElse(self, node: IfThenElse) -> T:
        return self.generic_visit(node)

    def visit_LetIn(self, node: LetIn) -> T:
        return self.generic_visit(node)

    def visit_CaseExpr(self, node: CaseExpr) -> T:
        return self.generic_visit(node)

    def visit_CaseArm(self, node: CaseArm) -> T:
        return self.generic_visit(node)

    def visit_UnchangedExpr(self, node: UnchangedExpr) -> T:
        return self.generic_visit(node)

    def visit_ExceptExpr(self, node: ExceptExpr) -> T:
        return self.generic_visit(node)

    def visit_ChooseExpr(self, node: ChooseExpr) -> T:
        return self.generic_visit(node)

    def visit_DomainExpr(self, node: DomainExpr) -> T:
        return self.generic_visit(node)

    # ── Definitions ─────────────────────────────────────────────────
    def visit_OperatorDef(self, node: OperatorDef) -> T:
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: FunctionDef) -> T:
        return self.generic_visit(node)

    def visit_VariableDecl(self, node: VariableDecl) -> T:
        return self.generic_visit(node)

    def visit_ConstantDecl(self, node: ConstantDecl) -> T:
        return self.generic_visit(node)

    def visit_Assumption(self, node: Assumption) -> T:
        return self.generic_visit(node)

    def visit_Theorem(self, node: Theorem) -> T:
        return self.generic_visit(node)

    def visit_InstanceDef(self, node: InstanceDef) -> T:
        return self.generic_visit(node)

    # ── Action nodes ────────────────────────────────────────────────
    def visit_ActionExpr(self, node: ActionExpr) -> T:
        return self.generic_visit(node)

    def visit_StutteringAction(self, node: StutteringAction) -> T:
        return self.generic_visit(node)

    def visit_FairnessExpr(self, node: FairnessExpr) -> T:
        return self.generic_visit(node)

    # ── Temporal nodes ──────────────────────────────────────────────
    def visit_AlwaysExpr(self, node: AlwaysExpr) -> T:
        return self.generic_visit(node)

    def visit_EventuallyExpr(self, node: EventuallyExpr) -> T:
        return self.generic_visit(node)

    def visit_LeadsToExpr(self, node: LeadsToExpr) -> T:
        return self.generic_visit(node)

    def visit_TemporalForallExpr(self, node: TemporalForallExpr) -> T:
        return self.generic_visit(node)

    def visit_TemporalExistsExpr(self, node: TemporalExistsExpr) -> T:
        return self.generic_visit(node)

    # ── Property nodes ──────────────────────────────────────────────
    def visit_InvariantProperty(self, node: InvariantProperty) -> T:
        return self.generic_visit(node)

    def visit_TemporalProperty(self, node: TemporalProperty) -> T:
        return self.generic_visit(node)

    def visit_SafetyProperty(self, node: SafetyProperty) -> T:
        return self.generic_visit(node)

    def visit_LivenessProperty(self, node: LivenessProperty) -> T:
        return self.generic_visit(node)

    # ── Types ───────────────────────────────────────────────────────
    def visit_IntType(self, node: IntType) -> T:
        return self.generic_visit(node)

    def visit_BoolType(self, node: BoolType) -> T:
        return self.generic_visit(node)

    def visit_StringType(self, node: StringType) -> T:
        return self.generic_visit(node)

    def visit_SetType(self, node: SetType) -> T:
        return self.generic_visit(node)

    def visit_FunctionType(self, node: FunctionType) -> T:
        return self.generic_visit(node)

    def visit_TupleType(self, node: TupleType) -> T:
        return self.generic_visit(node)

    def visit_RecordType(self, node: RecordType) -> T:
        return self.generic_visit(node)

    def visit_SequenceType(self, node: SequenceType) -> T:
        return self.generic_visit(node)

    def visit_AnyType(self, node: AnyType) -> T:
        return self.generic_visit(node)

    def visit_OperatorType(self, node: OperatorType) -> T:
        return self.generic_visit(node)
