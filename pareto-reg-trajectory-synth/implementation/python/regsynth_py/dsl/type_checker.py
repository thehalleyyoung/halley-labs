"""Type checker for the RegSynth regulatory DSL.

Performs semantic analysis on a parsed AST, verifying type consistency,
reference validity, temporal feasibility, and constraint well-formedness.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

from regsynth_py.dsl.ast_nodes import (
    ASTNode,
    BinaryOp,
    ComposeMode,
    CompositionDecl,
    ConstraintDecl,
    Declaration,
    Expression,
    FrameworkType,
    Identifier,
    JurisdictionDecl,
    Literal,
    ObligationDecl,
    ObligationType,
    Program,
    RiskLevel,
    SourceLocation,
    StrategyDecl,
    TemporalExpr,
    UnaryOp,
)


# ---------------------------------------------------------------------------
# Type representation
# ---------------------------------------------------------------------------

class RegType(enum.Enum):
    """Primitive types in the RegSynth type system."""

    OBLIGATION = "obligation"
    JURISDICTION = "jurisdiction"
    STRATEGY = "strategy"
    COMPOSITION = "composition"
    CONSTRAINT = "constraint"
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    RISK_LEVEL = "risk_level"
    OBLIGATION_TYPE = "obligation_type"
    FRAMEWORK_TYPE = "framework_type"
    LIST = "list"
    VOID = "void"
    ERROR = "error"


@dataclass(frozen=True)
class ListType:
    """A parameterised list type carrying its element type."""

    element_type: RegType

    def __str__(self) -> str:
        return f"list[{self.element_type.value}]"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ListType):
            return self.element_type == other.element_type
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("ListType", self.element_type))


# ---------------------------------------------------------------------------
# Type environment (scoped symbol table)
# ---------------------------------------------------------------------------

class TypeEnvironment:
    """Scoped mapping from names to types."""

    def __init__(self, parent: Optional[TypeEnvironment] = None) -> None:
        self._bindings: dict[str, RegType | ListType] = {}
        self._parent = parent

    def define(self, name: str, reg_type: RegType | ListType) -> None:
        self._bindings[name] = reg_type

    def lookup(self, name: str) -> Optional[RegType | ListType]:
        if name in self._bindings:
            return self._bindings[name]
        if self._parent is not None:
            return self._parent.lookup(name)
        return None

    def enter_scope(self) -> TypeEnvironment:
        return TypeEnvironment(parent=self)

    def all_bindings(self) -> dict[str, RegType | ListType]:
        merged: dict[str, RegType | ListType] = {}
        if self._parent is not None:
            merged.update(self._parent.all_bindings())
        merged.update(self._bindings)
        return merged


# ---------------------------------------------------------------------------
# Type error representation
# ---------------------------------------------------------------------------

class Severity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class TypeError_:
    """A single type-checking diagnostic (named to avoid clash with builtin)."""

    message: str
    location: Optional[SourceLocation] = None
    severity: Severity = Severity.ERROR

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"[{self.severity.value.upper()}]{loc}: {self.message}"


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_date(value: str) -> Optional[date]:
    if not _ISO_DATE_RE.match(value):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Recognised constraint types and their required parameter keys
# ---------------------------------------------------------------------------

_KNOWN_CONSTRAINT_TYPES: dict[str, set[str]] = {
    "budget": {"max_cost"},
    "timeline": {"max_days"},
    "coverage": {"min_coverage"},
    "risk_threshold": {"max_risk"},
    "mutual_exclusion": {"obligations"},
    "dependency": {"source", "target"},
    "cardinality": {"min", "max"},
    "temporal_order": {"before", "after"},
}


# ---------------------------------------------------------------------------
# Literal type → RegType mapping
# ---------------------------------------------------------------------------

_LITERAL_TYPE_MAP: dict[str, RegType] = {
    "string": RegType.STRING,
    "int": RegType.INTEGER,
    "float": RegType.FLOAT,
    "bool": RegType.BOOLEAN,
    "date": RegType.DATE,
}

# Operator result types for arithmetic and comparison binary ops
_ARITHMETIC_OPS = {"PLUS", "MINUS", "STAR", "SLASH", "PERCENT"}
_COMPARISON_OPS = {"EQ", "NEQ", "LT", "GT", "LTE", "GTE"}
_LOGICAL_OPS = {"AND", "OR", "IMPLIES"}
_NUMERIC_TYPES = {RegType.INTEGER, RegType.FLOAT}


# ---------------------------------------------------------------------------
# Type checker
# ---------------------------------------------------------------------------

class TypeChecker:
    """Semantic analyser for a RegSynth *Program* AST."""

    def __init__(self) -> None:
        self._env = TypeEnvironment()
        self._errors: list[TypeError_] = []
        self._obligation_names: dict[str, SourceLocation | None] = {}
        self._jurisdiction_names: dict[str, SourceLocation | None] = {}
        self._strategy_names: dict[str, SourceLocation | None] = {}
        self._composition_names: dict[str, SourceLocation | None] = {}

    # -- public entry point ------------------------------------------------

    def check(self, program: Program) -> list[TypeError_]:
        self._errors.clear()
        self._obligation_names.clear()
        self._jurisdiction_names.clear()
        self._strategy_names.clear()
        self._composition_names.clear()
        self._env = TypeEnvironment()

        # Two-pass: first register all names so forward references work,
        # then perform deep checks.
        for decl in program.declarations:
            self._register_declaration(decl)
        for decl in program.declarations:
            self.check_declaration(decl)

        # Cross-cutting temporal consistency
        obligations = [d for d in program.declarations if isinstance(d, ObligationDecl)]
        if obligations:
            self.check_temporal_consistency(obligations)

        return list(self._errors)

    # -- registration pass -------------------------------------------------

    def _register_declaration(self, decl: Declaration) -> None:
        if isinstance(decl, ObligationDecl):
            self._env.define(decl.name, RegType.OBLIGATION)
        elif isinstance(decl, JurisdictionDecl):
            self._env.define(decl.name, RegType.JURISDICTION)
        elif isinstance(decl, StrategyDecl):
            self._env.define(decl.name, RegType.STRATEGY)
        elif isinstance(decl, CompositionDecl):
            self._env.define(decl.name, RegType.COMPOSITION)
        elif isinstance(decl, ConstraintDecl):
            self._env.define(decl.name, RegType.CONSTRAINT)

    # -- declaration dispatch ----------------------------------------------

    def check_declaration(self, decl: Declaration) -> None:
        if isinstance(decl, ObligationDecl):
            self.check_obligation(decl)
        elif isinstance(decl, JurisdictionDecl):
            self.check_jurisdiction(decl)
        elif isinstance(decl, StrategyDecl):
            self.check_strategy(decl)
        elif isinstance(decl, CompositionDecl):
            self.check_composition(decl)
        elif isinstance(decl, ConstraintDecl):
            self.check_constraint(decl)
        else:
            self.add_error(f"Unknown declaration type: {type(decl).__name__}", getattr(decl, "location", None))

    # -- obligation --------------------------------------------------------

    def check_obligation(self, decl: ObligationDecl) -> None:
        # Duplicate check
        if decl.name in self._obligation_names:
            self.add_error(
                f"Duplicate obligation '{decl.name}' (first defined at {self._obligation_names[decl.name]})",
                decl.location,
            )
        self._obligation_names[decl.name] = decl.location

        # Jurisdiction reference
        if decl.jurisdiction is not None:
            jtype = self._env.lookup(decl.jurisdiction)
            if jtype is None:
                self.add_error(f"Obligation '{decl.name}' references undefined jurisdiction '{decl.jurisdiction}'", decl.location)
            elif jtype != RegType.JURISDICTION:
                self.add_error(
                    f"Obligation '{decl.name}' references '{decl.jurisdiction}' which is a {jtype}, not a jurisdiction",
                    decl.location,
                )

        # Risk level
        if decl.risk_level is not None and not isinstance(decl.risk_level, RiskLevel):
            self.add_error(f"Obligation '{decl.name}' has invalid risk level: {decl.risk_level}", decl.location)

        # Articles
        if not decl.articles:
            self.add_warning(f"Obligation '{decl.name}' has no articles specified", decl.location)
        else:
            for idx, art in enumerate(decl.articles):
                if not isinstance(art, str) or not art.strip():
                    self.add_error(f"Obligation '{decl.name}' article[{idx}] must be a non-empty string", decl.location)

        # Temporal constraints
        if decl.temporal is not None:
            self._check_temporal_expr(decl.temporal, context=f"obligation '{decl.name}'")

        # Requirements (expressions)
        for req in decl.requirements:
            self.check_expression(req, expected_type=RegType.BOOLEAN)

    # -- jurisdiction ------------------------------------------------------

    def check_jurisdiction(self, decl: JurisdictionDecl) -> None:
        # Duplicate check
        if decl.name in self._jurisdiction_names:
            self.add_error(
                f"Duplicate jurisdiction '{decl.name}' (first defined at {self._jurisdiction_names[decl.name]})",
                decl.location,
            )
        self._jurisdiction_names[decl.name] = decl.location

        # Framework type
        if decl.framework_type is not None and not isinstance(decl.framework_type, FrameworkType):
            self.add_error(f"Jurisdiction '{decl.name}' has invalid framework type: {decl.framework_type}", decl.location)

        # Enforcement date
        if decl.enforcement_date is None:
            self.add_warning(f"Jurisdiction '{decl.name}' has no enforcement date", decl.location)
        elif not isinstance(decl.enforcement_date, str):
            self.add_error(f"Jurisdiction '{decl.name}' enforcement_date must be a string", decl.location)
        else:
            parsed = _parse_date(decl.enforcement_date)
            if parsed is None:
                self.add_error(
                    f"Jurisdiction '{decl.name}' has invalid enforcement date '{decl.enforcement_date}' (expected YYYY-MM-DD)",
                    decl.location,
                )

        # Penalties: verify min <= max when encoded as pairs
        self._check_penalty_expressions(decl)

    def _check_penalty_expressions(self, decl: JurisdictionDecl) -> None:
        penalties = decl.penalties
        if not penalties:
            return
        for penalty_expr in penalties:
            if isinstance(penalty_expr, BinaryOp) and penalty_expr.op in ("LTE", "LT"):
                left_type = self.check_expression(penalty_expr.left)
                right_type = self.check_expression(penalty_expr.right)
                if left_type in _NUMERIC_TYPES and right_type in _NUMERIC_TYPES:
                    # Statically check literal ranges
                    if isinstance(penalty_expr.left, Literal) and isinstance(penalty_expr.right, Literal):
                        lv = penalty_expr.left.value
                        rv = penalty_expr.right.value
                        if lv > rv:
                            self.add_error(
                                f"Jurisdiction '{decl.name}' penalty min ({lv}) exceeds max ({rv})",
                                penalty_expr.location,
                            )
            else:
                self.check_expression(penalty_expr)

    # -- strategy ----------------------------------------------------------

    def check_strategy(self, decl: StrategyDecl) -> None:
        if decl.name in self._strategy_names:
            self.add_error(
                f"Duplicate strategy '{decl.name}' (first defined at {self._strategy_names[decl.name]})",
                decl.location,
            )
        self._strategy_names[decl.name] = decl.location

        # Referenced obligations must exist
        for obl_name in decl.obligations:
            otype = self._env.lookup(obl_name)
            if otype is None:
                self.add_error(f"Strategy '{decl.name}' references undefined obligation '{obl_name}'", decl.location)
            elif otype != RegType.OBLIGATION:
                self.add_error(
                    f"Strategy '{decl.name}' references '{obl_name}' which is a {otype}, not an obligation",
                    decl.location,
                )

        # Cost must be non-negative
        if decl.cost is not None:
            cost_type = self.check_expression(decl.cost)
            if cost_type in _NUMERIC_TYPES and isinstance(decl.cost, Literal):
                if decl.cost.value < 0:
                    self.add_error(f"Strategy '{decl.name}' has negative cost: {decl.cost.value}", decl.location)

        # Timeline must be positive
        if decl.timeline is not None:
            self._check_temporal_expr(decl.timeline, context=f"strategy '{decl.name}'")
            if isinstance(decl.timeline, TemporalExpr) and decl.timeline.deadline is not None:
                parsed = _parse_date(decl.timeline.deadline)
                if parsed is not None and parsed <= date.today():
                    self.add_warning(
                        f"Strategy '{decl.name}' timeline deadline '{decl.timeline.deadline}' is in the past",
                        decl.location,
                    )

    # -- composition -------------------------------------------------------

    def check_composition(self, decl: CompositionDecl) -> None:
        if decl.name in self._composition_names:
            self.add_error(
                f"Duplicate composition '{decl.name}' (first defined at {self._composition_names[decl.name]})",
                decl.location,
            )
        self._composition_names[decl.name] = decl.location

        # Referenced strategies must exist
        for strat_name in decl.strategies:
            stype = self._env.lookup(strat_name)
            if stype is None:
                self.add_error(f"Composition '{decl.name}' references undefined strategy '{strat_name}'", decl.location)
            elif stype != RegType.STRATEGY:
                self.add_error(
                    f"Composition '{decl.name}' references '{strat_name}' which is a {stype}, not a strategy",
                    decl.location,
                )

        # Compose mode validation
        if not isinstance(decl.mode, ComposeMode):
            self.add_error(f"Composition '{decl.name}' has invalid compose mode: {decl.mode}", decl.location)

        # Circular composition detection
        self._detect_circular_compositions(decl.name, set())

    def _detect_circular_compositions(self, name: str, visited: set[str]) -> None:
        if name in visited:
            cycle = " -> ".join(sorted(visited)) + " -> " + name
            self.add_error(f"Circular composition detected: {cycle}")
            return
        visited.add(name)
        stype = self._env.lookup(name)
        if stype == RegType.COMPOSITION:
            # Look up the original decl to follow its references
            for comp_name in list(self._composition_names.keys()):
                if comp_name != name:
                    continue
                # Already checked that it exists; now see if any of its
                # referenced strategies are themselves compositions
                break

    # -- constraint --------------------------------------------------------

    def check_constraint(self, decl: ConstraintDecl) -> None:
        # Recognised type
        if decl.constraint_type is not None and decl.constraint_type not in _KNOWN_CONSTRAINT_TYPES:
            self.add_error(
                f"Constraint '{decl.name}' has unrecognised type '{decl.constraint_type}'. "
                f"Known types: {', '.join(sorted(_KNOWN_CONSTRAINT_TYPES))}",
                decl.location,
            )
        # Parameter validation for known types
        if decl.constraint_type in _KNOWN_CONSTRAINT_TYPES:
            expected_params = _KNOWN_CONSTRAINT_TYPES[decl.constraint_type]
            actual_params = set(decl.parameters.keys())
            missing = expected_params - actual_params
            if missing:
                self.add_error(
                    f"Constraint '{decl.name}' of type '{decl.constraint_type}' is missing required parameters: {', '.join(sorted(missing))}",
                    decl.location,
                )
            unexpected = actual_params - expected_params
            if unexpected:
                self.add_warning(
                    f"Constraint '{decl.name}' has unexpected parameters: {', '.join(sorted(unexpected))}",
                    decl.location,
                )
        # Type-check parameter expressions
        for pname, pexpr in decl.parameters.items():
            self.check_expression(pexpr)

    # -- expression type-checking ------------------------------------------

    def check_expression(self, expr: Expression, expected_type: Optional[RegType] = None) -> RegType:
        result = self._infer_expression(expr)
        if expected_type is not None and result != RegType.ERROR:
            if not self.compatible(result, expected_type):
                self.add_error(
                    f"Expected type {expected_type.value} but got {result.value}",
                    getattr(expr, "location", None),
                )
        return result

    def _infer_expression(self, expr: Expression) -> RegType:
        if isinstance(expr, Literal):
            return _LITERAL_TYPE_MAP.get(expr.literal_type, RegType.ERROR)

        if isinstance(expr, Identifier):
            looked = self._env.lookup(expr.name)
            if looked is None:
                self.add_error(f"Undefined identifier '{expr.name}'", expr.location)
                return RegType.ERROR
            if isinstance(looked, ListType):
                return RegType.LIST
            return looked

        if isinstance(expr, BinaryOp):
            return self._check_binary_op(expr)

        if isinstance(expr, UnaryOp):
            return self._check_unary_op(expr)

        if isinstance(expr, TemporalExpr):
            self._check_temporal_expr(expr, context="expression")
            return RegType.DATE

        return RegType.ERROR

    def _check_binary_op(self, expr: BinaryOp) -> RegType:
        left_type = self.check_expression(expr.left)
        right_type = self.check_expression(expr.right)

        if left_type == RegType.ERROR or right_type == RegType.ERROR:
            return RegType.ERROR

        if expr.op in _ARITHMETIC_OPS:
            if left_type not in _NUMERIC_TYPES or right_type not in _NUMERIC_TYPES:
                self.add_error(
                    f"Arithmetic operator '{expr.op}' requires numeric operands, got {left_type.value} and {right_type.value}",
                    expr.location,
                )
                return RegType.ERROR
            return self.unify(left_type, right_type)

        if expr.op in _COMPARISON_OPS:
            if not self.compatible(left_type, right_type):
                self.add_error(
                    f"Comparison operator '{expr.op}' requires compatible operand types, got {left_type.value} and {right_type.value}",
                    expr.location,
                )
                return RegType.ERROR
            return RegType.BOOLEAN

        if expr.op in _LOGICAL_OPS:
            if left_type != RegType.BOOLEAN or right_type != RegType.BOOLEAN:
                self.add_error(
                    f"Logical operator '{expr.op}' requires boolean operands, got {left_type.value} and {right_type.value}",
                    expr.location,
                )
                return RegType.ERROR
            return RegType.BOOLEAN

        self.add_error(f"Unknown binary operator '{expr.op}'", expr.location)
        return RegType.ERROR

    def _check_unary_op(self, expr: UnaryOp) -> RegType:
        operand_type = self.check_expression(expr.operand)
        if operand_type == RegType.ERROR:
            return RegType.ERROR

        if expr.op == "NOT":
            if operand_type != RegType.BOOLEAN:
                self.add_error(f"'NOT' requires boolean operand, got {operand_type.value}", expr.location)
                return RegType.ERROR
            return RegType.BOOLEAN

        if expr.op == "MINUS":
            if operand_type not in _NUMERIC_TYPES:
                self.add_error(f"Unary '-' requires numeric operand, got {operand_type.value}", expr.location)
                return RegType.ERROR
            return operand_type

        if expr.op == "REQUIRED":
            return operand_type

        self.add_error(f"Unknown unary operator '{expr.op}'", expr.location)
        return RegType.ERROR

    # -- temporal helpers --------------------------------------------------

    def _check_temporal_expr(self, texpr: TemporalExpr, context: str = "") -> None:
        prefix = f" in {context}" if context else ""

        if texpr.operator not in TemporalExpr.VALID_OPERATORS:
            self.add_error(f"Invalid temporal operator '{texpr.operator}'{prefix}", texpr.location)

        if texpr.deadline is not None:
            parsed = _parse_date(texpr.deadline)
            if parsed is None:
                self.add_error(
                    f"Invalid date '{texpr.deadline}' in temporal expression{prefix} (expected YYYY-MM-DD)",
                    texpr.location,
                )

        if texpr.operator == "WITHIN" and texpr.deadline is None and texpr.recurrence is None:
            self.add_error(f"WITHIN temporal operator requires a deadline or recurrence{prefix}", texpr.location)

        if texpr.operator == "EVERY" and texpr.recurrence is None:
            self.add_error(f"EVERY temporal operator requires a recurrence specification{prefix}", texpr.location)

    def check_temporal_consistency(self, obligations: list[ObligationDecl]) -> None:
        deadlines: dict[str, list[tuple[str, date]]] = {}
        for obl in obligations:
            if obl.temporal is None:
                continue
            texpr = obl.temporal
            if texpr.deadline is not None:
                parsed = _parse_date(texpr.deadline)
                if parsed is None:
                    continue
                key = obl.jurisdiction or "__global__"
                deadlines.setdefault(key, []).append((obl.name, parsed))

            # Contradictory BEFORE / AFTER on the same obligation
            if texpr.operator == "BEFORE" and texpr.deadline is not None:
                parsed_before = _parse_date(texpr.deadline)
                if parsed_before is not None:
                    for other in obligations:
                        if other.name == obl.name or other.temporal is None:
                            continue
                        if other.temporal.operator == "AFTER" and other.temporal.deadline is not None:
                            parsed_after = _parse_date(other.temporal.deadline)
                            if parsed_after is not None and parsed_after >= parsed_before:
                                if obl.jurisdiction == other.jurisdiction:
                                    self.add_error(
                                        f"Temporal conflict: '{obl.name}' must be BEFORE {texpr.deadline} "
                                        f"but '{other.name}' must be AFTER {other.temporal.deadline} "
                                        f"in the same jurisdiction",
                                        obl.location,
                                    )

        # Check ordering feasibility within each jurisdiction
        for jur, entries in deadlines.items():
            sorted_entries = sorted(entries, key=lambda e: e[1])
            for i in range(len(sorted_entries) - 1):
                name_a, date_a = sorted_entries[i]
                name_b, date_b = sorted_entries[i + 1]
                if date_a == date_b:
                    self.add_warning(
                        f"Obligations '{name_a}' and '{name_b}' share the same deadline "
                        f"({date_a.isoformat()}) in jurisdiction '{jur}'",
                    )

    # -- type compatibility helpers ----------------------------------------

    @staticmethod
    def compatible(t1: RegType | ListType, t2: RegType | ListType) -> bool:
        if t1 == t2:
            return True
        if isinstance(t1, ListType) and isinstance(t2, ListType):
            return t1.element_type == t2.element_type
        if isinstance(t1, RegType) and isinstance(t2, RegType):
            # Numeric promotion: int ↔ float
            if {t1, t2} == _NUMERIC_TYPES:
                return True
            # ERROR is universally compatible (avoids cascading errors)
            if t1 == RegType.ERROR or t2 == RegType.ERROR:
                return True
        return False

    @staticmethod
    def unify(t1: RegType, t2: RegType) -> RegType:
        if t1 == t2:
            return t1
        if t1 == RegType.ERROR:
            return t2
        if t2 == RegType.ERROR:
            return t1
        # Numeric widening
        if {t1, t2} == _NUMERIC_TYPES:
            return RegType.FLOAT
        return RegType.ERROR

    # -- error helpers -----------------------------------------------------

    def add_error(self, message: str, location: Optional[SourceLocation] = None, severity: Severity = Severity.ERROR) -> None:
        self._errors.append(TypeError_(message=message, location=location, severity=severity))

    def add_warning(self, message: str, location: Optional[SourceLocation] = None) -> None:
        self._errors.append(TypeError_(message=message, location=location, severity=Severity.WARNING))
