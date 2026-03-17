"""
AST node definitions for the RegSynth regulatory compliance DSL.

Provides a complete typed AST for representing regulatory obligations,
jurisdictions, compliance strategies, temporal constraints, and risk levels
aligned with frameworks like the EU AI Act.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RiskLevel(enum.Enum):
    """EU AI Act risk classification tiers."""
    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"

    def __lt__(self, other: RiskLevel) -> bool:
        _order = {
            RiskLevel.MINIMAL: 0,
            RiskLevel.LIMITED: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.UNACCEPTABLE: 3,
        }
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return _order[self] < _order[other]

    def __le__(self, other: RiskLevel) -> bool:
        return self == other or self.__lt__(other)

    def __gt__(self, other: RiskLevel) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return not self.__le__(other)

    def __ge__(self, other: RiskLevel) -> bool:
        return self == other or self.__gt__(other)


class ObligationType(enum.Enum):
    """How binding an obligation is."""
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    CONDITIONAL = "conditional"


class FrameworkType(enum.Enum):
    """Legal enforceability of a jurisdiction's framework."""
    BINDING = "binding"
    VOLUNTARY = "voluntary"
    HYBRID = "hybrid"


class ComposeMode(enum.Enum):
    """Strategy composition modes."""
    UNION = "union"
    INTERSECT = "intersect"
    SEQUENCE = "sequence"
    OVERRIDE = "override"


# ---------------------------------------------------------------------------
# Source location
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Pinpoints a position in source text."""
    line: int
    column: int
    offset: int

    def __repr__(self) -> str:
        return f"SourceLocation(line={self.line}, col={self.column}, offset={self.offset})"

    def __str__(self) -> str:
        return f"{self.line}:{self.column}"


# ---------------------------------------------------------------------------
# Base AST node
# ---------------------------------------------------------------------------

class ASTNode:
    """Abstract base for every node in the regulatory DSL AST."""

    def __init__(self, location: Optional[SourceLocation] = None) -> None:
        self.location = location

    # -- visitor dispatch ----------------------------------------------------
    def accept(self, visitor: ASTVisitor) -> Any:
        method_name = "visit_" + type(self).__name__
        method = getattr(visitor, method_name, visitor.generic_visit)
        return method(self)

    # -- child enumeration ---------------------------------------------------
    def children(self) -> list[ASTNode]:
        """Return immediate child nodes (override in subclasses)."""
        return []

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        assert isinstance(other, ASTNode)
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items()
                           if k != "location")
        return f"{type(self).__name__}({fields})"


# ---------------------------------------------------------------------------
# Top-level program
# ---------------------------------------------------------------------------

class Program(ASTNode):
    """Root node containing all top-level declarations."""

    def __init__(
        self,
        declarations: list[Declaration],
        location: Optional[SourceLocation] = None,
    ) -> None:
        super().__init__(location)
        self.declarations = list(declarations)

    def children(self) -> list[ASTNode]:
        return list(self.declarations)


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

class Declaration(ASTNode):
    """Abstract base for top-level declarations."""


class ObligationDecl(Declaration):
    """
    A regulatory obligation declaration.

    Maps to a concrete compliance requirement from a regulatory text such as
    EU AI Act articles or GDPR provisions.
    """

    def __init__(
        self,
        name: str,
        jurisdiction: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None,
        category: Optional[ObligationType] = None,
        articles: Optional[list[str]] = None,
        temporal: Optional[TemporalExpr] = None,
        requirements: Optional[list[Expression]] = None,
        metadata: Optional[dict[str, Any]] = None,
        location: Optional[SourceLocation] = None,
    ) -> None:
        super().__init__(location)
        self.name = name
        self.jurisdiction = jurisdiction
        self.risk_level = risk_level
        self.category = category
        self.articles = articles or []
        self.temporal = temporal
        self.requirements = requirements or []
        self.metadata = metadata or {}

    def children(self) -> list[ASTNode]:
        kids: list[ASTNode] = []
        if self.temporal is not None:
            kids.append(self.temporal)
        kids.extend(self.requirements)
        return kids


class JurisdictionDecl(Declaration):
    """
    Declares a regulatory jurisdiction and its framework metadata.
    """

    def __init__(
        self,
        name: str,
        framework_type: Optional[FrameworkType] = None,
        region: Optional[str] = None,
        enforcement_date: Optional[str] = None,
        penalties: Optional[list[Expression]] = None,
        location: Optional[SourceLocation] = None,
    ) -> None:
        super().__init__(location)
        self.name = name
        self.framework_type = framework_type
        self.region = region
        self.enforcement_date = enforcement_date
        self.penalties = penalties or []

    def children(self) -> list[ASTNode]:
        return list(self.penalties)


class StrategyDecl(Declaration):
    """
    A compliance strategy that addresses one or more obligations.
    """

    def __init__(
        self,
        name: str,
        obligations: Optional[list[str]] = None,
        cost: Optional[Expression] = None,
        timeline: Optional[TemporalExpr] = None,
        location: Optional[SourceLocation] = None,
    ) -> None:
        super().__init__(location)
        self.name = name
        self.obligations = obligations or []
        self.cost = cost
        self.timeline = timeline

    def children(self) -> list[ASTNode]:
        kids: list[ASTNode] = []
        if self.cost is not None:
            kids.append(self.cost)
        if self.timeline is not None:
            kids.append(self.timeline)
        return kids


class CompositionDecl(Declaration):
    """
    Composes multiple strategies via a chosen mode (union, intersect, etc.).
    """

    def __init__(
        self,
        name: str,
        strategies: Optional[list[str]] = None,
        mode: ComposeMode = ComposeMode.UNION,
        location: Optional[SourceLocation] = None,
    ) -> None:
        super().__init__(location)
        self.name = name
        self.strategies = strategies or []
        self.mode = mode

    def children(self) -> list[ASTNode]:
        return []


class ConstraintDecl(Declaration):
    """
    Declares a named constraint on Pareto-optimal solutions.
    """

    def __init__(
        self,
        name: str,
        constraint_type: Optional[str] = None,
        parameters: Optional[dict[str, Expression]] = None,
        location: Optional[SourceLocation] = None,
    ) -> None:
        super().__init__(location)
        self.name = name
        self.constraint_type = constraint_type
        self.parameters = parameters or {}

    def children(self) -> list[ASTNode]:
        return list(self.parameters.values())


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

class Expression(ASTNode):
    """Abstract base for all expression nodes."""


class BinaryOp(Expression):
    """Binary operation: left <op> right."""

    VALID_OPS = frozenset({
        "AND", "OR", "IMPLIES", "EQ", "NEQ",
        "LT", "GT", "LTE", "GTE",
        "PLUS", "MINUS", "STAR", "SLASH", "PERCENT",
    })

    def __init__(
        self,
        op: str,
        left: Expression,
        right: Expression,
        location: Optional[SourceLocation] = None,
    ) -> None:
        super().__init__(location)
        if op not in self.VALID_OPS:
            raise ValueError(f"Invalid binary operator: {op!r}")
        self.op = op
        self.left = left
        self.right = right

    def children(self) -> list[ASTNode]:
        return [self.left, self.right]


class UnaryOp(Expression):
    """Unary operation: <op> operand."""

    VALID_OPS = frozenset({"NOT", "REQUIRED", "MINUS"})

    def __init__(
        self,
        op: str,
        operand: Expression,
        location: Optional[SourceLocation] = None,
    ) -> None:
        super().__init__(location)
        if op not in self.VALID_OPS:
            raise ValueError(f"Invalid unary operator: {op!r}")
        self.op = op
        self.operand = operand

    def children(self) -> list[ASTNode]:
        return [self.operand]


class Identifier(Expression):
    """A named reference."""

    def __init__(
        self,
        name: str,
        location: Optional[SourceLocation] = None,
    ) -> None:
        super().__init__(location)
        self.name = name

    def children(self) -> list[ASTNode]:
        return []


class Literal(Expression):
    """A literal value (string, int, float, bool, date)."""

    VALID_TYPES = frozenset({"string", "int", "float", "bool", "date"})

    def __init__(
        self,
        value: Any,
        literal_type: str,
        location: Optional[SourceLocation] = None,
    ) -> None:
        super().__init__(location)
        if literal_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid literal type: {literal_type!r}")
        self.value = value
        self.literal_type = literal_type

    def children(self) -> list[ASTNode]:
        return []


class TemporalExpr(Expression):
    """
    Temporal constraint expression.

    Represents deadlines, phased timelines, and recurrence schedules
    that are common in regulatory compliance calendars.
    """

    VALID_OPERATORS = frozenset({"BEFORE", "AFTER", "WITHIN", "EVERY"})

    def __init__(
        self,
        operator: str,
        deadline: Optional[str] = None,
        phase: Optional[str] = None,
        recurrence: Optional[str] = None,
        location: Optional[SourceLocation] = None,
    ) -> None:
        super().__init__(location)
        if operator not in self.VALID_OPERATORS:
            raise ValueError(f"Invalid temporal operator: {operator!r}")
        self.operator = operator
        self.deadline = deadline
        self.phase = phase
        self.recurrence = recurrence

    def children(self) -> list[ASTNode]:
        return []


# ---------------------------------------------------------------------------
# Visitor infrastructure
# ---------------------------------------------------------------------------

class ASTVisitor:
    """
    Base visitor with a ``visit_<ClassName>`` dispatch protocol.

    Subclass and override ``visit_Program``, ``visit_ObligationDecl``, etc.
    Falls back to ``generic_visit`` for unhandled node types.
    """

    def generic_visit(self, node: ASTNode) -> Any:
        """Default handler — visits all children in order."""
        results: list[Any] = []
        for child in node.children():
            results.append(child.accept(self))
        return results

    # -- convenience entry point ---------------------------------------------
    def visit(self, node: ASTNode) -> Any:
        return node.accept(self)

    # -- concrete visit stubs (override as needed) ---------------------------
    def visit_Program(self, node: Program) -> Any:
        return self.generic_visit(node)

    def visit_ObligationDecl(self, node: ObligationDecl) -> Any:
        return self.generic_visit(node)

    def visit_JurisdictionDecl(self, node: JurisdictionDecl) -> Any:
        return self.generic_visit(node)

    def visit_StrategyDecl(self, node: StrategyDecl) -> Any:
        return self.generic_visit(node)

    def visit_CompositionDecl(self, node: CompositionDecl) -> Any:
        return self.generic_visit(node)

    def visit_ConstraintDecl(self, node: ConstraintDecl) -> Any:
        return self.generic_visit(node)

    def visit_BinaryOp(self, node: BinaryOp) -> Any:
        return self.generic_visit(node)

    def visit_UnaryOp(self, node: UnaryOp) -> Any:
        return self.generic_visit(node)

    def visit_Identifier(self, node: Identifier) -> Any:
        return self.generic_visit(node)

    def visit_Literal(self, node: Literal) -> Any:
        return self.generic_visit(node)

    def visit_TemporalExpr(self, node: TemporalExpr) -> Any:
        return self.generic_visit(node)


class ASTTransformer(ASTVisitor):
    """
    Visitor that rebuilds the tree, returning a (possibly modified) copy.

    Override individual ``visit_*`` methods to transform specific node types.
    Unoverridden methods perform a deep identity copy.
    """

    def generic_visit(self, node: ASTNode) -> ASTNode:
        return node

    def visit_Program(self, node: Program) -> Program:
        new_decls = [d.accept(self) for d in node.declarations]
        return Program(new_decls, location=node.location)

    def visit_ObligationDecl(self, node: ObligationDecl) -> ObligationDecl:
        new_temporal = node.temporal.accept(self) if node.temporal else None
        new_reqs = [r.accept(self) for r in node.requirements]
        return ObligationDecl(
            name=node.name,
            jurisdiction=node.jurisdiction,
            risk_level=node.risk_level,
            category=node.category,
            articles=list(node.articles),
            temporal=new_temporal,
            requirements=new_reqs,
            metadata=dict(node.metadata),
            location=node.location,
        )

    def visit_JurisdictionDecl(self, node: JurisdictionDecl) -> JurisdictionDecl:
        new_penalties = [p.accept(self) for p in node.penalties]
        return JurisdictionDecl(
            name=node.name,
            framework_type=node.framework_type,
            region=node.region,
            enforcement_date=node.enforcement_date,
            penalties=new_penalties,
            location=node.location,
        )

    def visit_StrategyDecl(self, node: StrategyDecl) -> StrategyDecl:
        new_cost = node.cost.accept(self) if node.cost else None
        new_timeline = node.timeline.accept(self) if node.timeline else None
        return StrategyDecl(
            name=node.name,
            obligations=list(node.obligations),
            cost=new_cost,
            timeline=new_timeline,
            location=node.location,
        )

    def visit_CompositionDecl(self, node: CompositionDecl) -> CompositionDecl:
        return CompositionDecl(
            name=node.name,
            strategies=list(node.strategies),
            mode=node.mode,
            location=node.location,
        )

    def visit_ConstraintDecl(self, node: ConstraintDecl) -> ConstraintDecl:
        new_params = {k: v.accept(self) for k, v in node.parameters.items()}
        return ConstraintDecl(
            name=node.name,
            constraint_type=node.constraint_type,
            parameters=new_params,
            location=node.location,
        )

    def visit_BinaryOp(self, node: BinaryOp) -> BinaryOp:
        return BinaryOp(
            op=node.op,
            left=node.left.accept(self),
            right=node.right.accept(self),
            location=node.location,
        )

    def visit_UnaryOp(self, node: UnaryOp) -> UnaryOp:
        return UnaryOp(
            op=node.op,
            operand=node.operand.accept(self),
            location=node.location,
        )

    def visit_Identifier(self, node: Identifier) -> Identifier:
        return Identifier(name=node.name, location=node.location)

    def visit_Literal(self, node: Literal) -> Literal:
        return Literal(
            value=node.value,
            literal_type=node.literal_type,
            location=node.location,
        )

    def visit_TemporalExpr(self, node: TemporalExpr) -> TemporalExpr:
        return TemporalExpr(
            operator=node.operator,
            deadline=node.deadline,
            phase=node.phase,
            recurrence=node.recurrence,
            location=node.location,
        )


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def pretty_print(node: ASTNode, indent: int = 0) -> str:
    """
    Return a human-readable indented text representation of *node*.
    """
    pad = "  " * indent
    lines: list[str] = []

    if isinstance(node, Program):
        lines.append(f"{pad}Program")
        for decl in node.declarations:
            lines.append(pretty_print(decl, indent + 1))

    elif isinstance(node, ObligationDecl):
        lines.append(f"{pad}Obligation '{node.name}'")
        if node.jurisdiction:
            lines.append(f"{pad}  jurisdiction: {node.jurisdiction}")
        if node.risk_level is not None:
            lines.append(f"{pad}  risk_level: {node.risk_level.value}")
        if node.category is not None:
            lines.append(f"{pad}  category: {node.category.value}")
        if node.articles:
            lines.append(f"{pad}  articles: {', '.join(node.articles)}")
        if node.temporal:
            lines.append(f"{pad}  temporal:")
            lines.append(pretty_print(node.temporal, indent + 2))
        if node.requirements:
            lines.append(f"{pad}  requirements:")
            for req in node.requirements:
                lines.append(pretty_print(req, indent + 2))
        if node.metadata:
            for k, v in node.metadata.items():
                lines.append(f"{pad}  meta.{k}: {v}")

    elif isinstance(node, JurisdictionDecl):
        lines.append(f"{pad}Jurisdiction '{node.name}'")
        if node.framework_type is not None:
            lines.append(f"{pad}  framework: {node.framework_type.value}")
        if node.region:
            lines.append(f"{pad}  region: {node.region}")
        if node.enforcement_date:
            lines.append(f"{pad}  enforcement_date: {node.enforcement_date}")
        if node.penalties:
            lines.append(f"{pad}  penalties:")
            for pen in node.penalties:
                lines.append(pretty_print(pen, indent + 2))

    elif isinstance(node, StrategyDecl):
        lines.append(f"{pad}Strategy '{node.name}'")
        if node.obligations:
            lines.append(f"{pad}  obligations: {', '.join(node.obligations)}")
        if node.cost is not None:
            lines.append(f"{pad}  cost:")
            lines.append(pretty_print(node.cost, indent + 2))
        if node.timeline is not None:
            lines.append(f"{pad}  timeline:")
            lines.append(pretty_print(node.timeline, indent + 2))

    elif isinstance(node, CompositionDecl):
        lines.append(f"{pad}Compose '{node.name}' mode={node.mode.value}")
        if node.strategies:
            lines.append(f"{pad}  strategies: {', '.join(node.strategies)}")

    elif isinstance(node, ConstraintDecl):
        lines.append(f"{pad}Constraint '{node.name}'")
        if node.constraint_type:
            lines.append(f"{pad}  type: {node.constraint_type}")
        if node.parameters:
            lines.append(f"{pad}  parameters:")
            for k, v in node.parameters.items():
                lines.append(f"{pad}    {k}:")
                lines.append(pretty_print(v, indent + 3))

    elif isinstance(node, BinaryOp):
        lines.append(f"{pad}BinaryOp({node.op})")
        lines.append(pretty_print(node.left, indent + 1))
        lines.append(pretty_print(node.right, indent + 1))

    elif isinstance(node, UnaryOp):
        lines.append(f"{pad}UnaryOp({node.op})")
        lines.append(pretty_print(node.operand, indent + 1))

    elif isinstance(node, Identifier):
        lines.append(f"{pad}Ident({node.name})")

    elif isinstance(node, Literal):
        lines.append(f"{pad}Literal({node.literal_type}: {node.value!r})")

    elif isinstance(node, TemporalExpr):
        parts = [f"op={node.operator}"]
        if node.deadline:
            parts.append(f"deadline={node.deadline}")
        if node.phase:
            parts.append(f"phase={node.phase}")
        if node.recurrence:
            parts.append(f"recurrence={node.recurrence}")
        lines.append(f"{pad}Temporal({', '.join(parts)})")

    else:
        lines.append(f"{pad}{node!r}")

    return "\n".join(lines)
