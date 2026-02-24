"""
Python guard extraction module.

Extracts runtime guards from Python AST and converts them to predicate
templates suitable for refinement-type inference and CEGAR seeding.
"""

from __future__ import annotations

import ast
import copy
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

class GuardPattern(Enum):
    """Classification of a runtime guard pattern."""
    Comparison = auto()
    TypeTag = auto()
    Nullity = auto()
    Structural = auto()
    Truthiness = auto()
    Membership = auto()
    Identity = auto()


class Polarity(Enum):
    """Which branch of the conditional the guard applies to."""
    TRUE_BRANCH = auto()
    FALSE_BRANCH = auto()


class PredicateKind(Enum):
    """Kind of predicate produced by guard extraction."""
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
    LenComparison = auto()
    RangeBound = auto()
    ExceptionType = auto()
    Assertion = auto()
    PatternMatch = auto()


class ComparisonOp(Enum):
    """Comparison operators in canonical form."""
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="
    IS = "is"
    IS_NOT = "is not"
    IN = "in"
    NOT_IN = "not in"


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
    def from_node(node: ast.AST, filename: str = "") -> SourceLocation:
        return SourceLocation(
            filename=filename,
            line=getattr(node, "lineno", 0),
            col=getattr(node, "col_offset", 0),
            end_line=getattr(node, "end_lineno", 0) or 0,
            end_col=getattr(node, "end_col_offset", 0) or 0,
        )


# ---------------------------------------------------------------------------
# Predicate templates
# ---------------------------------------------------------------------------

@dataclass
class PredicateTemplate:
    """Base predicate template produced by guard extraction."""
    kind: PredicateKind
    variables: List[str] = field(default_factory=list)

    def negate(self) -> PredicateTemplate:
        return NegationPredicate(child=self, variables=list(self.variables))

    def conjoin(self, other: PredicateTemplate) -> ConjunctionPredicate:
        combined = sorted(set(self.variables + other.variables))
        return ConjunctionPredicate(children=[self, other], variables=combined)

    def disjoin(self, other: PredicateTemplate) -> DisjunctionPredicate:
        combined = sorted(set(self.variables + other.variables))
        return DisjunctionPredicate(children=[self, other], variables=combined)


@dataclass
class TypeTagPredicate(PredicateTemplate):
    """Guard that checks the runtime type tag of a variable."""
    kind: PredicateKind = field(default=PredicateKind.TypeTag, init=False)
    target_variable: str = ""
    type_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class NullityPredicate(PredicateTemplate):
    """Guard that checks for None / null."""
    kind: PredicateKind = field(default=PredicateKind.Nullity, init=False)
    target_variable: str = ""
    is_none: bool = True

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class TruthinessPredicate(PredicateTemplate):
    """Guard that checks truthiness (``if x``)."""
    kind: PredicateKind = field(default=PredicateKind.Truthiness, init=False)
    target_variable: str = ""

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class HasAttrPredicate(PredicateTemplate):
    """Guard that checks structural presence of an attribute."""
    kind: PredicateKind = field(default=PredicateKind.HasAttr, init=False)
    target_variable: str = ""
    attr_name: str = ""

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class ComparisonPredicate(PredicateTemplate):
    """Guard that performs a comparison."""
    kind: PredicateKind = field(default=PredicateKind.Comparison, init=False)
    left_expr: str = ""
    op: ComparisonOp = ComparisonOp.LT
    right_expr: str = ""

    def __post_init__(self) -> None:
        if not self.variables:
            self.variables = []


@dataclass
class LenComparisonPredicate(PredicateTemplate):
    """Guard comparing against len(x)."""
    kind: PredicateKind = field(default=PredicateKind.LenComparison, init=False)
    index_variable: str = ""
    collection_variable: str = ""
    op: ComparisonOp = ComparisonOp.LT

    def __post_init__(self) -> None:
        vs: List[str] = []
        if self.index_variable:
            vs.append(self.index_variable)
        if self.collection_variable:
            vs.append(self.collection_variable)
        if not self.variables:
            self.variables = vs


@dataclass
class MembershipPredicate(PredicateTemplate):
    """Guard that checks membership (``x in collection``)."""
    kind: PredicateKind = field(default=PredicateKind.Membership, init=False)
    element_variable: str = ""
    collection_expr: str = ""

    def __post_init__(self) -> None:
        if self.element_variable and self.element_variable not in self.variables:
            self.variables = [self.element_variable] + self.variables


@dataclass
class CallablePredicate(PredicateTemplate):
    """Guard that checks if a variable is callable."""
    kind: PredicateKind = field(default=PredicateKind.Callable, init=False)
    target_variable: str = ""

    def __post_init__(self) -> None:
        if self.target_variable and self.target_variable not in self.variables:
            self.variables = [self.target_variable] + self.variables


@dataclass
class IdentityPredicate(PredicateTemplate):
    """Guard using ``is`` / ``is not`` identity comparison."""
    kind: PredicateKind = field(default=PredicateKind.Identity, init=False)
    left_variable: str = ""
    right_expr: str = ""
    is_positive: bool = True

    def __post_init__(self) -> None:
        if self.left_variable and self.left_variable not in self.variables:
            self.variables = [self.left_variable] + self.variables


@dataclass
class ConjunctionPredicate(PredicateTemplate):
    """Conjunction of multiple predicates."""
    kind: PredicateKind = field(default=PredicateKind.Conjunction, init=False)
    children: List[PredicateTemplate] = field(default_factory=list)


@dataclass
class DisjunctionPredicate(PredicateTemplate):
    """Disjunction of multiple predicates."""
    kind: PredicateKind = field(default=PredicateKind.Disjunction, init=False)
    children: List[PredicateTemplate] = field(default_factory=list)


@dataclass
class NegationPredicate(PredicateTemplate):
    """Negation of a predicate."""
    kind: PredicateKind = field(default=PredicateKind.Negation, init=False)
    child: Optional[PredicateTemplate] = None


@dataclass
class RangeBoundPredicate(PredicateTemplate):
    """Guard for loop-range bounds."""
    kind: PredicateKind = field(default=PredicateKind.RangeBound, init=False)
    loop_variable: str = ""
    lower_bound: Optional[str] = None
    upper_bound: Optional[str] = None
    step: Optional[str] = None

    def __post_init__(self) -> None:
        if self.loop_variable and self.loop_variable not in self.variables:
            self.variables = [self.loop_variable] + self.variables


@dataclass
class ExceptionTypePredicate(PredicateTemplate):
    """Guard derived from except clauses."""
    kind: PredicateKind = field(default=PredicateKind.ExceptionType, init=False)
    exception_variable: str = ""
    exception_types: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.exception_variable and self.exception_variable not in self.variables:
            self.variables = [self.exception_variable] + self.variables


@dataclass
class AssertionPredicate(PredicateTemplate):
    """Predicate derived from an assert statement."""
    kind: PredicateKind = field(default=PredicateKind.Assertion, init=False)
    inner: Optional[PredicateTemplate] = None
    message: str = ""


@dataclass
class PatternMatchPredicate(PredicateTemplate):
    """Guard derived from a match/case statement (Python 3.10+)."""
    kind: PredicateKind = field(default=PredicateKind.PatternMatch, init=False)
    subject_variable: str = ""
    pattern_description: str = ""
    bound_variables: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.subject_variable and self.subject_variable not in self.variables:
            self.variables = [self.subject_variable] + self.variables


# ---------------------------------------------------------------------------
# ExtractedGuard
# ---------------------------------------------------------------------------

@dataclass
class ExtractedGuard:
    """A guard extracted from a Python AST node."""
    pattern: GuardPattern
    variables: List[str]
    predicate: PredicateTemplate
    source_location: SourceLocation
    polarity: Polarity = Polarity.TRUE_BRANCH
    raw_source: str = ""
    confidence: float = 1.0

    def negated(self) -> ExtractedGuard:
        """Return a copy with flipped polarity and negated predicate."""
        new_polarity = (
            Polarity.FALSE_BRANCH
            if self.polarity == Polarity.TRUE_BRANCH
            else Polarity.TRUE_BRANCH
        )
        return ExtractedGuard(
            pattern=self.pattern,
            variables=list(self.variables),
            predicate=self.predicate.negate(),
            source_location=self.source_location,
            polarity=new_polarity,
            raw_source=self.raw_source,
            confidence=self.confidence,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expr_to_str(node: ast.AST) -> str:
    """Best-effort conversion of an AST expression to source text."""
    try:
        return ast.unparse(node)
    except Exception:
        return "<expr>"


def _collect_names(node: ast.AST) -> List[str]:
    """Collect all Name identifiers from an AST expression."""
    names: List[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.append(child.id)
    return sorted(set(names))


def _ast_cmpop_to_enum(op: ast.cmpop) -> ComparisonOp:
    """Map an AST comparison operator to our enum."""
    mapping: Dict[type, ComparisonOp] = {
        ast.Lt: ComparisonOp.LT,
        ast.LtE: ComparisonOp.LE,
        ast.Gt: ComparisonOp.GT,
        ast.GtE: ComparisonOp.GE,
        ast.Eq: ComparisonOp.EQ,
        ast.NotEq: ComparisonOp.NE,
        ast.Is: ComparisonOp.IS,
        ast.IsNot: ComparisonOp.IS_NOT,
        ast.In: ComparisonOp.IN,
        ast.NotIn: ComparisonOp.NOT_IN,
    }
    return mapping.get(type(op), ComparisonOp.EQ)


def _is_none(node: ast.AST) -> bool:
    """Check if a node represents ``None``."""
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.NameConstant) and node.value is None:  # type: ignore[attr-defined]
        return True
    return False


def _is_len_call(node: ast.AST) -> Optional[str]:
    """If *node* is ``len(x)`` return the stringified argument, else None."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
    ):
        return _expr_to_str(node.args[0])
    return None


def _flip_op(op: ComparisonOp) -> ComparisonOp:
    """Flip a comparison operator (swap sides)."""
    flip_map = {
        ComparisonOp.LT: ComparisonOp.GT,
        ComparisonOp.LE: ComparisonOp.GE,
        ComparisonOp.GT: ComparisonOp.LT,
        ComparisonOp.GE: ComparisonOp.LE,
        ComparisonOp.EQ: ComparisonOp.EQ,
        ComparisonOp.NE: ComparisonOp.NE,
        ComparisonOp.IS: ComparisonOp.IS,
        ComparisonOp.IS_NOT: ComparisonOp.IS_NOT,
        ComparisonOp.IN: ComparisonOp.IN,
        ComparisonOp.NOT_IN: ComparisonOp.NOT_IN,
    }
    return flip_map.get(op, op)


def _negate_op(op: ComparisonOp) -> ComparisonOp:
    """Negate a comparison operator."""
    negate_map = {
        ComparisonOp.LT: ComparisonOp.GE,
        ComparisonOp.LE: ComparisonOp.GT,
        ComparisonOp.GT: ComparisonOp.LE,
        ComparisonOp.GE: ComparisonOp.LT,
        ComparisonOp.EQ: ComparisonOp.NE,
        ComparisonOp.NE: ComparisonOp.EQ,
        ComparisonOp.IS: ComparisonOp.IS_NOT,
        ComparisonOp.IS_NOT: ComparisonOp.IS,
        ComparisonOp.IN: ComparisonOp.NOT_IN,
        ComparisonOp.NOT_IN: ComparisonOp.IN,
    }
    return negate_map.get(op, op)


# ---------------------------------------------------------------------------
# PythonGuardExtractor
# ---------------------------------------------------------------------------

class PythonGuardExtractor(ast.NodeVisitor):
    """Walk a Python AST and extract runtime guards from conditionals,
    assertions, except clauses, loops, and match statements."""

    def __init__(self, filename: str = "<unknown>", source: str = "") -> None:
        self.filename = filename
        self.source = source
        self._guards: List[ExtractedGuard] = []
        self._source_lines: List[str] = source.splitlines() if source else []

    # -- public API ---------------------------------------------------------

    def extract(self, tree: ast.AST) -> List[ExtractedGuard]:
        """Extract all guards from *tree*."""
        self._guards = []
        self.visit(tree)
        return list(self._guards)

    @staticmethod
    def extract_from_source(source: str, filename: str = "<string>") -> List[ExtractedGuard]:
        """Convenience: parse source and extract guards."""
        tree = ast.parse(source, filename=filename)
        extractor = PythonGuardExtractor(filename=filename, source=source)
        return extractor.extract(tree)

    # -- visitors -----------------------------------------------------------

    def visit_If(self, node: ast.If) -> None:
        guards = self._extract_guard_from_test(node.test, Polarity.TRUE_BRANCH)
        self._guards.extend(guards)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        guards = self._extract_guard_from_test(node.test, Polarity.TRUE_BRANCH)
        self._guards.extend(guards)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        guards = self._extract_guard_from_test(node.test, Polarity.TRUE_BRANCH)
        self._guards.extend(guards)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        assertion_extractor = AssertionExtractor(self.filename)
        self._guards.extend(assertion_extractor.extract_assertion(node))
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        loop_extractor = LoopGuardExtractor(self.filename)
        self._guards.extend(loop_extractor.extract_for_loop(node))
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        exc_extractor = ExceptionGuardExtractor(self.filename)
        self._guards.extend(exc_extractor.extract_except(node))
        self.generic_visit(node)

    def visit_Match(self, node: ast.AST) -> None:
        """Handle match/case (Python 3.10+)."""
        pm_extractor = PatternMatchExtractor(self.filename)
        self._guards.extend(pm_extractor.extract_match(node))
        self.generic_visit(node)

    # -- core extraction logic ----------------------------------------------

    def _extract_guard_from_test(
        self, node: ast.expr, polarity: Polarity
    ) -> List[ExtractedGuard]:
        """Dispatch on the shape of a test expression."""
        guards: List[ExtractedGuard] = []

        # --- BoolOp (and / or) ---
        if isinstance(node, ast.BoolOp):
            guards.extend(self._handle_boolop(node, polarity))

        # --- UnaryOp (not) ---
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            guards.extend(self._handle_not(node, polarity))

        # --- Compare ---
        elif isinstance(node, ast.Compare):
            guards.extend(self._handle_compare(node, polarity))

        # --- Call ---
        elif isinstance(node, ast.Call):
            guards.extend(self._handle_call(node, polarity))

        # --- bare Name (truthiness) ---
        elif isinstance(node, ast.Name):
            guards.append(self._make_truthiness_guard(node, polarity))

        # --- Attribute (truthiness on attr access) ---
        elif isinstance(node, ast.Attribute):
            guards.append(self._make_truthiness_guard(node, polarity))

        # --- Subscript (truthiness on subscript) ---
        elif isinstance(node, ast.Subscript):
            guards.append(self._make_truthiness_guard(node, polarity))

        # --- NamedExpr (walrus, e.g. if (m := re.match(...))) ---
        elif isinstance(node, ast.NamedExpr):
            inner = self._extract_guard_from_test(node.value, polarity)
            if inner:
                guards.extend(inner)
            else:
                guards.append(self._make_truthiness_guard(node, polarity))

        return guards

    # -- BoolOp handler -----------------------------------------------------

    def _handle_boolop(
        self, node: ast.BoolOp, polarity: Polarity
    ) -> List[ExtractedGuard]:
        child_guards: List[List[ExtractedGuard]] = []
        for value in node.values:
            child_guards.append(self._extract_guard_from_test(value, polarity))

        flat_children = [g for gs in child_guards for g in gs]
        if not flat_children:
            return []

        if isinstance(node.op, ast.And):
            return self._combine_conjunction(flat_children, node, polarity)
        else:
            return self._combine_disjunction(flat_children, node, polarity)

    def _combine_conjunction(
        self,
        guards: List[ExtractedGuard],
        node: ast.BoolOp,
        polarity: Polarity,
    ) -> List[ExtractedGuard]:
        if len(guards) == 1:
            return guards

        all_vars: List[str] = sorted(
            set(v for g in guards for v in g.variables)
        )
        predicates = [g.predicate for g in guards]
        conj = ConjunctionPredicate(
            children=predicates, variables=list(all_vars)
        )
        combined = ExtractedGuard(
            pattern=guards[0].pattern,
            variables=all_vars,
            predicate=conj,
            source_location=SourceLocation.from_node(node, self.filename),
            polarity=polarity,
            raw_source=_expr_to_str(node),
        )
        return [combined] + guards

    def _combine_disjunction(
        self,
        guards: List[ExtractedGuard],
        node: ast.BoolOp,
        polarity: Polarity,
    ) -> List[ExtractedGuard]:
        if len(guards) == 1:
            return guards

        all_vars: List[str] = sorted(
            set(v for g in guards for v in g.variables)
        )
        predicates = [g.predicate for g in guards]
        disj = DisjunctionPredicate(
            children=predicates, variables=list(all_vars)
        )
        combined = ExtractedGuard(
            pattern=guards[0].pattern,
            variables=all_vars,
            predicate=disj,
            source_location=SourceLocation.from_node(node, self.filename),
            polarity=polarity,
            raw_source=_expr_to_str(node),
        )
        return [combined]

    # -- not handler --------------------------------------------------------

    def _handle_not(
        self, node: ast.UnaryOp, polarity: Polarity
    ) -> List[ExtractedGuard]:
        flipped = (
            Polarity.FALSE_BRANCH
            if polarity == Polarity.TRUE_BRANCH
            else Polarity.TRUE_BRANCH
        )
        inner_guards = self._extract_guard_from_test(node.operand, flipped)
        result: List[ExtractedGuard] = []
        for g in inner_guards:
            result.append(g.negated())
        return result

    # -- Compare handler ----------------------------------------------------

    def _handle_compare(
        self, node: ast.Compare, polarity: Polarity
    ) -> List[ExtractedGuard]:
        guards: List[ExtractedGuard] = []

        # Chained comparison: 0 <= i < len(arr)
        if len(node.ops) > 1:
            guards.extend(self._handle_chained_compare(node, polarity))
            return guards

        left = node.left
        op_node = node.ops[0]
        right = node.comparators[0]
        op = _ast_cmpop_to_enum(op_node)

        # x is None / x is not None
        if op in (ComparisonOp.IS, ComparisonOp.IS_NOT):
            if _is_none(right):
                is_none = op == ComparisonOp.IS
                var_name = _expr_to_str(left)
                pred = NullityPredicate(
                    target_variable=var_name,
                    is_none=is_none,
                    variables=[var_name],
                )
                guards.append(
                    ExtractedGuard(
                        pattern=GuardPattern.Nullity,
                        variables=_collect_names(left),
                        predicate=pred,
                        source_location=SourceLocation.from_node(
                            node, self.filename
                        ),
                        polarity=polarity,
                        raw_source=_expr_to_str(node),
                    )
                )
                return guards
            elif _is_none(left):
                is_none = op == ComparisonOp.IS
                var_name = _expr_to_str(right)
                pred = NullityPredicate(
                    target_variable=var_name,
                    is_none=is_none,
                    variables=[var_name],
                )
                guards.append(
                    ExtractedGuard(
                        pattern=GuardPattern.Nullity,
                        variables=_collect_names(right),
                        predicate=pred,
                        source_location=SourceLocation.from_node(
                            node, self.filename
                        ),
                        polarity=polarity,
                        raw_source=_expr_to_str(node),
                    )
                )
                return guards

        # type(x) == T  /  type(x) is T
        if op in (ComparisonOp.EQ, ComparisonOp.IS):
            type_guard = self._try_type_eq_guard(left, right, node, polarity)
            if type_guard:
                guards.append(type_guard)
                return guards
            type_guard = self._try_type_eq_guard(right, left, node, polarity)
            if type_guard:
                guards.append(type_guard)
                return guards

        # x in collection / x not in collection
        if op in (ComparisonOp.IN, ComparisonOp.NOT_IN):
            elem = _expr_to_str(left)
            coll = _expr_to_str(right)
            pred = MembershipPredicate(
                element_variable=elem,
                collection_expr=coll,
                variables=_collect_names(left) + _collect_names(right),
            )
            pattern = GuardPattern.Membership
            if op == ComparisonOp.NOT_IN:
                pred_final: PredicateTemplate = pred.negate()
            else:
                pred_final = pred
            guards.append(
                ExtractedGuard(
                    pattern=pattern,
                    variables=sorted(set(_collect_names(left) + _collect_names(right))),
                    predicate=pred_final,
                    source_location=SourceLocation.from_node(node, self.filename),
                    polarity=polarity,
                    raw_source=_expr_to_str(node),
                )
            )
            return guards

        # len comparisons: x < len(arr), len(arr) > x, etc.
        len_guard = self._try_len_comparison(left, op, right, node, polarity)
        if len_guard:
            guards.append(len_guard)
            return guards

        # General comparison
        all_vars = sorted(set(_collect_names(left) + _collect_names(right)))
        pred = ComparisonPredicate(
            left_expr=_expr_to_str(left),
            op=op,
            right_expr=_expr_to_str(right),
            variables=all_vars,
        )
        guards.append(
            ExtractedGuard(
                pattern=GuardPattern.Comparison,
                variables=all_vars,
                predicate=pred,
                source_location=SourceLocation.from_node(node, self.filename),
                polarity=polarity,
                raw_source=_expr_to_str(node),
            )
        )
        return guards

    def _handle_chained_compare(
        self, node: ast.Compare, polarity: Polarity
    ) -> List[ExtractedGuard]:
        """Handle chained comparison like ``0 <= i < len(arr)``."""
        parts: List[ExtractedGuard] = []
        operands = [node.left] + list(node.comparators)
        for idx, op_node in enumerate(node.ops):
            left = operands[idx]
            right = operands[idx + 1]
            op = _ast_cmpop_to_enum(op_node)

            len_g = self._try_len_comparison(left, op, right, node, polarity)
            if len_g:
                parts.append(len_g)
                continue

            all_vars = sorted(set(_collect_names(left) + _collect_names(right)))
            pred = ComparisonPredicate(
                left_expr=_expr_to_str(left),
                op=op,
                right_expr=_expr_to_str(right),
                variables=all_vars,
            )
            parts.append(
                ExtractedGuard(
                    pattern=GuardPattern.Comparison,
                    variables=all_vars,
                    predicate=pred,
                    source_location=SourceLocation.from_node(node, self.filename),
                    polarity=polarity,
                    raw_source=_expr_to_str(node),
                )
            )

        if len(parts) > 1:
            all_vars = sorted(set(v for g in parts for v in g.variables))
            conj = ConjunctionPredicate(
                children=[g.predicate for g in parts],
                variables=all_vars,
            )
            combined = ExtractedGuard(
                pattern=GuardPattern.Comparison,
                variables=all_vars,
                predicate=conj,
                source_location=SourceLocation.from_node(node, self.filename),
                polarity=polarity,
                raw_source=_expr_to_str(node),
            )
            parts.insert(0, combined)

        return parts

    # -- Call handler -------------------------------------------------------

    def _handle_call(
        self, node: ast.Call, polarity: Polarity
    ) -> List[ExtractedGuard]:
        guards: List[ExtractedGuard] = []

        func = node.func

        # isinstance(x, T) / isinstance(x, (T1, T2, ...))
        if isinstance(func, ast.Name) and func.id == "isinstance":
            guard = self._extract_isinstance(node, polarity)
            if guard:
                guards.append(guard)
                return guards

        # hasattr(x, 'attr')
        if isinstance(func, ast.Name) and func.id == "hasattr":
            guard = self._extract_hasattr(node, polarity)
            if guard:
                guards.append(guard)
                return guards

        # callable(x)
        if isinstance(func, ast.Name) and func.id == "callable":
            guard = self._extract_callable(node, polarity)
            if guard:
                guards.append(guard)
                return guards

        # issubclass(x, T)
        if isinstance(func, ast.Name) and func.id == "issubclass":
            guard = self._extract_issubclass(node, polarity)
            if guard:
                guards.append(guard)
                return guards

        # any()/all() with generators
        if isinstance(func, ast.Name) and func.id in ("any", "all"):
            guard = self._extract_any_all(node, polarity)
            if guard:
                guards.append(guard)
                return guards

        # Fallback: treat call result as truthiness
        guards.append(self._make_truthiness_guard(node, polarity))
        return guards

    # -- specific extractors ------------------------------------------------

    def _extract_isinstance(
        self, node: ast.Call, polarity: Polarity
    ) -> Optional[ExtractedGuard]:
        if len(node.args) < 2:
            return None

        target = node.args[0]
        type_arg = node.args[1]

        var_name = _expr_to_str(target)
        type_names: List[str] = []

        if isinstance(type_arg, ast.Tuple):
            for elt in type_arg.elts:
                type_names.append(_expr_to_str(elt))
        else:
            type_names.append(_expr_to_str(type_arg))

        pred = TypeTagPredicate(
            target_variable=var_name,
            type_names=tuple(type_names),
            variables=_collect_names(target),
        )
        return ExtractedGuard(
            pattern=GuardPattern.TypeTag,
            variables=_collect_names(target),
            predicate=pred,
            source_location=SourceLocation.from_node(node, self.filename),
            polarity=polarity,
            raw_source=_expr_to_str(node),
        )

    def _extract_hasattr(
        self, node: ast.Call, polarity: Polarity
    ) -> Optional[ExtractedGuard]:
        if len(node.args) < 2:
            return None

        target = node.args[0]
        attr_arg = node.args[1]

        var_name = _expr_to_str(target)
        attr_name = ""
        if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
            attr_name = attr_arg.value
        else:
            attr_name = _expr_to_str(attr_arg)

        pred = HasAttrPredicate(
            target_variable=var_name,
            attr_name=attr_name,
            variables=_collect_names(target),
        )
        return ExtractedGuard(
            pattern=GuardPattern.Structural,
            variables=_collect_names(target),
            predicate=pred,
            source_location=SourceLocation.from_node(node, self.filename),
            polarity=polarity,
            raw_source=_expr_to_str(node),
        )

    def _extract_callable(
        self, node: ast.Call, polarity: Polarity
    ) -> Optional[ExtractedGuard]:
        if len(node.args) < 1:
            return None

        target = node.args[0]
        var_name = _expr_to_str(target)
        pred = CallablePredicate(
            target_variable=var_name,
            variables=_collect_names(target),
        )
        return ExtractedGuard(
            pattern=GuardPattern.TypeTag,
            variables=_collect_names(target),
            predicate=pred,
            source_location=SourceLocation.from_node(node, self.filename),
            polarity=polarity,
            raw_source=_expr_to_str(node),
        )

    def _extract_issubclass(
        self, node: ast.Call, polarity: Polarity
    ) -> Optional[ExtractedGuard]:
        if len(node.args) < 2:
            return None

        target = node.args[0]
        type_arg = node.args[1]

        var_name = _expr_to_str(target)
        type_names: List[str] = []
        if isinstance(type_arg, ast.Tuple):
            for elt in type_arg.elts:
                type_names.append(_expr_to_str(elt))
        else:
            type_names.append(_expr_to_str(type_arg))

        pred = TypeTagPredicate(
            target_variable=var_name,
            type_names=tuple(type_names),
            variables=_collect_names(target),
        )
        return ExtractedGuard(
            pattern=GuardPattern.TypeTag,
            variables=_collect_names(target),
            predicate=pred,
            source_location=SourceLocation.from_node(node, self.filename),
            polarity=polarity,
            raw_source=_expr_to_str(node),
            confidence=0.8,
        )

    def _extract_any_all(
        self, node: ast.Call, polarity: Polarity
    ) -> Optional[ExtractedGuard]:
        if len(node.args) != 1:
            return None
        arg = node.args[0]
        if not isinstance(arg, (ast.GeneratorExp, ast.ListComp)):
            return None

        all_vars = _collect_names(arg)
        pred = TruthinessPredicate(
            target_variable=_expr_to_str(node),
            variables=all_vars,
        )
        return ExtractedGuard(
            pattern=GuardPattern.Truthiness,
            variables=all_vars,
            predicate=pred,
            source_location=SourceLocation.from_node(node, self.filename),
            polarity=polarity,
            raw_source=_expr_to_str(node),
            confidence=0.6,
        )

    # -- type(x) == T -------------------------------------------------------

    def _try_type_eq_guard(
        self,
        maybe_type_call: ast.expr,
        maybe_type_name: ast.expr,
        node: ast.Compare,
        polarity: Polarity,
    ) -> Optional[ExtractedGuard]:
        if not (
            isinstance(maybe_type_call, ast.Call)
            and isinstance(maybe_type_call.func, ast.Name)
            and maybe_type_call.func.id == "type"
            and len(maybe_type_call.args) == 1
        ):
            return None

        target = maybe_type_call.args[0]
        var_name = _expr_to_str(target)
        type_name = _expr_to_str(maybe_type_name)

        pred = TypeTagPredicate(
            target_variable=var_name,
            type_names=(type_name,),
            variables=_collect_names(target),
        )
        return ExtractedGuard(
            pattern=GuardPattern.TypeTag,
            variables=_collect_names(target),
            predicate=pred,
            source_location=SourceLocation.from_node(node, self.filename),
            polarity=polarity,
            raw_source=_expr_to_str(node),
        )

    # -- len comparison ------------------------------------------------------

    def _try_len_comparison(
        self,
        left: ast.expr,
        op: ComparisonOp,
        right: ast.expr,
        node: ast.AST,
        polarity: Polarity,
    ) -> Optional[ExtractedGuard]:
        len_arg_right = _is_len_call(right)
        len_arg_left = _is_len_call(left)

        if len_arg_right:
            idx_var = _expr_to_str(left)
            coll_var = len_arg_right
            effective_op = op
        elif len_arg_left:
            idx_var = _expr_to_str(right)
            coll_var = len_arg_left
            effective_op = _flip_op(op)
        else:
            return None

        all_vars = sorted(
            set(_collect_names(left) + _collect_names(right))
        )
        pred = LenComparisonPredicate(
            index_variable=idx_var,
            collection_variable=coll_var,
            op=effective_op,
            variables=all_vars,
        )
        return ExtractedGuard(
            pattern=GuardPattern.Comparison,
            variables=all_vars,
            predicate=pred,
            source_location=SourceLocation.from_node(node, self.filename),
            polarity=polarity,
            raw_source=_expr_to_str(node) if isinstance(node, ast.expr) else "",
        )

    # -- truthiness ----------------------------------------------------------

    def _make_truthiness_guard(
        self, node: ast.expr, polarity: Polarity
    ) -> ExtractedGuard:
        var_name = _expr_to_str(node)
        all_vars = _collect_names(node)
        pred = TruthinessPredicate(
            target_variable=var_name,
            variables=all_vars,
        )
        return ExtractedGuard(
            pattern=GuardPattern.Truthiness,
            variables=all_vars,
            predicate=pred,
            source_location=SourceLocation.from_node(node, self.filename),
            polarity=polarity,
            raw_source=var_name,
        )


# ---------------------------------------------------------------------------
# GuardNormalizer
# ---------------------------------------------------------------------------

class GuardNormalizer:
    """Normalizes extracted guards into a canonical form suitable for
    deduplication and comparison."""

    def normalize(self, guard: ExtractedGuard) -> ExtractedGuard:
        """Return a normalized copy of *guard*."""
        pred = self._normalize_predicate(guard.predicate)
        return ExtractedGuard(
            pattern=guard.pattern,
            variables=sorted(set(guard.variables)),
            predicate=pred,
            source_location=guard.source_location,
            polarity=guard.polarity,
            raw_source=guard.raw_source,
            confidence=guard.confidence,
        )

    def normalize_all(self, guards: List[ExtractedGuard]) -> List[ExtractedGuard]:
        """Normalize and deduplicate a list of guards."""
        seen: Set[str] = set()
        result: List[ExtractedGuard] = []
        for g in guards:
            ng = self.normalize(g)
            key = self._guard_key(ng)
            if key not in seen:
                seen.add(key)
                result.append(ng)
        return result

    def _normalize_predicate(self, pred: PredicateTemplate) -> PredicateTemplate:
        pred = copy.deepcopy(pred)
        pred.variables = sorted(set(pred.variables))

        if isinstance(pred, ConjunctionPredicate):
            pred.children = [self._normalize_predicate(c) for c in pred.children]
            pred.children = self._flatten_conjunction(pred.children)
            pred.children.sort(key=self._predicate_sort_key)
        elif isinstance(pred, DisjunctionPredicate):
            pred.children = [self._normalize_predicate(c) for c in pred.children]
            pred.children = self._flatten_disjunction(pred.children)
            pred.children.sort(key=self._predicate_sort_key)
        elif isinstance(pred, NegationPredicate) and pred.child:
            pred.child = self._normalize_predicate(pred.child)
            # Double negation elimination
            if isinstance(pred.child, NegationPredicate) and pred.child.child:
                return pred.child.child

        # Normalize comparison direction: always put names on left
        if isinstance(pred, ComparisonPredicate):
            pred = self._normalize_comparison(pred)

        return pred

    def _normalize_comparison(self, pred: ComparisonPredicate) -> ComparisonPredicate:
        """Ensure the 'simpler' expression is on the left."""
        if pred.left_expr > pred.right_expr and pred.op in (
            ComparisonOp.LT, ComparisonOp.LE, ComparisonOp.GT, ComparisonOp.GE,
            ComparisonOp.EQ, ComparisonOp.NE,
        ):
            return ComparisonPredicate(
                left_expr=pred.right_expr,
                op=_flip_op(pred.op),
                right_expr=pred.left_expr,
                variables=pred.variables,
            )
        return pred

    def _flatten_conjunction(
        self, children: List[PredicateTemplate]
    ) -> List[PredicateTemplate]:
        result: List[PredicateTemplate] = []
        for c in children:
            if isinstance(c, ConjunctionPredicate):
                result.extend(self._flatten_conjunction(c.children))
            else:
                result.append(c)
        return result

    def _flatten_disjunction(
        self, children: List[PredicateTemplate]
    ) -> List[PredicateTemplate]:
        result: List[PredicateTemplate] = []
        for c in children:
            if isinstance(c, DisjunctionPredicate):
                result.extend(self._flatten_disjunction(c.children))
            else:
                result.append(c)
        return result

    @staticmethod
    def _predicate_sort_key(pred: PredicateTemplate) -> str:
        return f"{pred.kind.name}:{','.join(pred.variables)}"

    @staticmethod
    def _guard_key(guard: ExtractedGuard) -> str:
        parts = [
            guard.pattern.name,
            guard.polarity.name,
            ",".join(guard.variables),
            guard.predicate.kind.name,
        ]
        return "|".join(parts)


# ---------------------------------------------------------------------------
# GuardRanker
# ---------------------------------------------------------------------------

@dataclass
class RankedGuard:
    """A guard with its computed rank score."""
    guard: ExtractedGuard
    score: float
    frequency: int = 1
    information_content: float = 0.0


class GuardRanker:
    """Ranks extracted guards by information content and frequency,
    producing an ordering useful for CEGAR seeding."""

    # Weights for different guard patterns
    PATTERN_WEIGHTS: Dict[GuardPattern, float] = {
        GuardPattern.TypeTag: 1.0,
        GuardPattern.Nullity: 0.9,
        GuardPattern.Comparison: 0.7,
        GuardPattern.Structural: 0.8,
        GuardPattern.Truthiness: 0.3,
        GuardPattern.Membership: 0.6,
        GuardPattern.Identity: 0.5,
    }

    PREDICATE_WEIGHTS: Dict[PredicateKind, float] = {
        PredicateKind.TypeTag: 1.0,
        PredicateKind.Nullity: 0.9,
        PredicateKind.HasAttr: 0.85,
        PredicateKind.Callable: 0.8,
        PredicateKind.Comparison: 0.7,
        PredicateKind.LenComparison: 0.75,
        PredicateKind.Membership: 0.6,
        PredicateKind.Truthiness: 0.3,
        PredicateKind.Conjunction: 0.8,
        PredicateKind.Disjunction: 0.5,
        PredicateKind.Negation: 0.4,
        PredicateKind.RangeBound: 0.65,
        PredicateKind.ExceptionType: 0.7,
        PredicateKind.Assertion: 0.6,
        PredicateKind.PatternMatch: 0.85,
        PredicateKind.Identity: 0.5,
    }

    def rank(self, guards: List[ExtractedGuard]) -> List[RankedGuard]:
        """Rank guards by information content, returning highest first."""
        freq_map: Dict[str, int] = self._compute_frequencies(guards)
        ranked: List[RankedGuard] = []

        for guard in guards:
            info = self._information_content(guard)
            key = self._guard_signature(guard)
            freq = freq_map.get(key, 1)
            # Higher frequency slightly boosts, but with diminishing returns
            freq_bonus = min(1.0 + 0.1 * (freq - 1), 1.5)
            score = info * freq_bonus * guard.confidence
            ranked.append(
                RankedGuard(
                    guard=guard,
                    score=score,
                    frequency=freq,
                    information_content=info,
                )
            )

        ranked.sort(key=lambda rg: rg.score, reverse=True)
        return ranked

    def top_k(self, guards: List[ExtractedGuard], k: int = 10) -> List[ExtractedGuard]:
        """Return top-k guards by ranking."""
        ranked = self.rank(guards)
        return [rg.guard for rg in ranked[:k]]

    def _information_content(self, guard: ExtractedGuard) -> float:
        pattern_w = self.PATTERN_WEIGHTS.get(guard.pattern, 0.5)
        pred_w = self.PREDICATE_WEIGHTS.get(guard.predicate.kind, 0.5)
        var_count_bonus = min(len(guard.variables) * 0.1, 0.3)
        return (pattern_w + pred_w) / 2.0 + var_count_bonus

    def _compute_frequencies(
        self, guards: List[ExtractedGuard]
    ) -> Dict[str, int]:
        freq: Dict[str, int] = {}
        for g in guards:
            key = self._guard_signature(g)
            freq[key] = freq.get(key, 0) + 1
        return freq

    @staticmethod
    def _guard_signature(guard: ExtractedGuard) -> str:
        return f"{guard.pattern.name}:{guard.predicate.kind.name}:{','.join(sorted(guard.variables))}"


# ---------------------------------------------------------------------------
# ConditionalGuardTracker
# ---------------------------------------------------------------------------

@dataclass
class GuardScope:
    """Guards active within a particular scope (branch)."""
    active_guards: List[ExtractedGuard] = field(default_factory=list)
    parent: Optional[GuardScope] = None

    def all_guards(self) -> List[ExtractedGuard]:
        """Return all guards active in this scope, including parent scopes."""
        result = list(self.active_guards)
        if self.parent:
            result.extend(self.parent.all_guards())
        return result

    def add_guard(self, guard: ExtractedGuard) -> None:
        self.active_guards.append(guard)

    def child_scope(self) -> GuardScope:
        return GuardScope(parent=self)


class ConditionalGuardTracker(ast.NodeVisitor):
    """Tracks which guards are active on each branch of conditionals,
    building a map from AST nodes to their guard contexts."""

    def __init__(self, filename: str = "") -> None:
        self.filename = filename
        self._current_scope = GuardScope()
        self._node_guards: Dict[int, List[ExtractedGuard]] = {}
        self._extractor = PythonGuardExtractor(filename=filename)

    def track(self, tree: ast.AST) -> Dict[int, List[ExtractedGuard]]:
        """Walk the tree and build guard-context mapping."""
        self._current_scope = GuardScope()
        self._node_guards = {}
        self.visit(tree)
        return dict(self._node_guards)

    def get_guards_at(self, node: ast.AST) -> List[ExtractedGuard]:
        """Return guards active at a particular node."""
        return self._node_guards.get(id(node), [])

    def visit_If(self, node: ast.If) -> None:
        test_guards = self._extractor._extract_guard_from_test(
            node.test, Polarity.TRUE_BRANCH
        )

        # True branch
        true_scope = self._current_scope.child_scope()
        for g in test_guards:
            true_scope.add_guard(g)

        old_scope = self._current_scope
        self._current_scope = true_scope
        for stmt in node.body:
            self._node_guards[id(stmt)] = true_scope.all_guards()
            self.visit(stmt)

        # False branch (else / elif)
        false_scope = old_scope.child_scope()
        for g in test_guards:
            false_scope.add_guard(g.negated())

        self._current_scope = false_scope
        for stmt in node.orelse:
            self._node_guards[id(stmt)] = false_scope.all_guards()
            self.visit(stmt)

        self._current_scope = old_scope

    def visit_While(self, node: ast.While) -> None:
        test_guards = self._extractor._extract_guard_from_test(
            node.test, Polarity.TRUE_BRANCH
        )

        body_scope = self._current_scope.child_scope()
        for g in test_guards:
            body_scope.add_guard(g)

        old_scope = self._current_scope
        self._current_scope = body_scope
        for stmt in node.body:
            self._node_guards[id(stmt)] = body_scope.all_guards()
            self.visit(stmt)

        if node.orelse:
            else_scope = old_scope.child_scope()
            for g in test_guards:
                else_scope.add_guard(g.negated())
            self._current_scope = else_scope
            for stmt in node.orelse:
                self._node_guards[id(stmt)] = else_scope.all_guards()
                self.visit(stmt)

        self._current_scope = old_scope

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        old_scope = self._current_scope
        self._current_scope = GuardScope()
        self.generic_visit(node)
        self._current_scope = old_scope

    visit_AsyncFunctionDef = visit_FunctionDef

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if id(child) not in self._node_guards:
                self._node_guards[id(child)] = self._current_scope.all_guards()
            self.visit(child)


# ---------------------------------------------------------------------------
# LoopGuardExtractor
# ---------------------------------------------------------------------------

class LoopGuardExtractor:
    """Extracts guards relevant to loop constructs: range bounds,
    enumerate patterns, and iteration-variable constraints."""

    def __init__(self, filename: str = "") -> None:
        self.filename = filename

    def extract_for_loop(self, node: ast.For) -> List[ExtractedGuard]:
        """Extract guards from a for-loop header."""
        guards: List[ExtractedGuard] = []

        iter_node = node.iter
        target = node.target

        # for i in range(...)
        if self._is_range_call(iter_node):
            guards.extend(self._extract_range_guards(target, iter_node, node))

        # for i, x in enumerate(...)
        elif self._is_enumerate_call(iter_node):
            guards.extend(self._extract_enumerate_guards(target, iter_node, node))

        # for x in collection (implicit: x is element type)
        elif isinstance(target, ast.Name):
            coll_expr = _expr_to_str(iter_node)
            coll_vars = _collect_names(iter_node)
            pred = MembershipPredicate(
                element_variable=target.id,
                collection_expr=coll_expr,
                variables=[target.id] + coll_vars,
            )
            guards.append(
                ExtractedGuard(
                    pattern=GuardPattern.Membership,
                    variables=[target.id] + coll_vars,
                    predicate=pred,
                    source_location=SourceLocation.from_node(node, self.filename),
                    polarity=Polarity.TRUE_BRANCH,
                    raw_source=_expr_to_str(node),
                    confidence=0.7,
                )
            )

        # for k, v in dict.items()
        if self._is_dict_items_call(iter_node) and isinstance(target, ast.Tuple):
            guards.extend(self._extract_dict_items_guards(target, iter_node, node))

        # for x in zip(...)
        if self._is_zip_call(iter_node):
            guards.extend(self._extract_zip_guards(target, iter_node, node))

        return guards

    def _is_range_call(self, node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
        )

    def _is_enumerate_call(self, node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "enumerate"
        )

    def _is_dict_items_call(self, node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "items"
        )

    def _is_zip_call(self, node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "zip"
        )

    def _extract_range_guards(
        self, target: ast.AST, call: ast.Call, node: ast.For
    ) -> List[ExtractedGuard]:
        guards: List[ExtractedGuard] = []
        args = call.args

        if not isinstance(target, ast.Name):
            return guards

        loop_var = target.id
        lower: Optional[str] = None
        upper: Optional[str] = None
        step: Optional[str] = None

        if len(args) == 1:
            lower = "0"
            upper = _expr_to_str(args[0])
        elif len(args) >= 2:
            lower = _expr_to_str(args[0])
            upper = _expr_to_str(args[1])
        if len(args) >= 3:
            step = _expr_to_str(args[2])

        all_vars = [loop_var]
        for arg in args:
            all_vars.extend(_collect_names(arg))
        all_vars = sorted(set(all_vars))

        pred = RangeBoundPredicate(
            loop_variable=loop_var,
            lower_bound=lower,
            upper_bound=upper,
            step=step,
            variables=all_vars,
        )
        guards.append(
            ExtractedGuard(
                pattern=GuardPattern.Comparison,
                variables=all_vars,
                predicate=pred,
                source_location=SourceLocation.from_node(node, self.filename),
                polarity=Polarity.TRUE_BRANCH,
                raw_source=_expr_to_str(call),
            )
        )

        # Emit explicit bounds guards
        if lower is not None:
            lb_pred = ComparisonPredicate(
                left_expr=loop_var,
                op=ComparisonOp.GE,
                right_expr=lower,
                variables=[loop_var],
            )
            guards.append(
                ExtractedGuard(
                    pattern=GuardPattern.Comparison,
                    variables=[loop_var],
                    predicate=lb_pred,
                    source_location=SourceLocation.from_node(node, self.filename),
                    polarity=Polarity.TRUE_BRANCH,
                    raw_source=f"{loop_var} >= {lower}",
                )
            )
        if upper is not None:
            ub_pred = ComparisonPredicate(
                left_expr=loop_var,
                op=ComparisonOp.LT,
                right_expr=upper,
                variables=[loop_var],
            )
            guards.append(
                ExtractedGuard(
                    pattern=GuardPattern.Comparison,
                    variables=[loop_var],
                    predicate=ub_pred,
                    source_location=SourceLocation.from_node(node, self.filename),
                    polarity=Polarity.TRUE_BRANCH,
                    raw_source=f"{loop_var} < {upper}",
                )
            )

        return guards

    def _extract_enumerate_guards(
        self, target: ast.AST, call: ast.Call, node: ast.For
    ) -> List[ExtractedGuard]:
        guards: List[ExtractedGuard] = []
        if not isinstance(target, ast.Tuple) or len(target.elts) != 2:
            return guards

        idx_target = target.elts[0]
        val_target = target.elts[1]

        if not isinstance(idx_target, ast.Name):
            return guards

        idx_var = idx_target.id
        start = "0"
        if len(call.args) >= 2:
            start = _expr_to_str(call.args[1])
        elif call.keywords:
            for kw in call.keywords:
                if kw.arg == "start":
                    start = _expr_to_str(kw.value)

        # idx >= start
        lb_pred = ComparisonPredicate(
            left_expr=idx_var,
            op=ComparisonOp.GE,
            right_expr=start,
            variables=[idx_var],
        )
        guards.append(
            ExtractedGuard(
                pattern=GuardPattern.Comparison,
                variables=[idx_var],
                predicate=lb_pred,
                source_location=SourceLocation.from_node(node, self.filename),
                polarity=Polarity.TRUE_BRANCH,
                raw_source=f"{idx_var} >= {start}",
            )
        )

        # val in iterable
        if isinstance(val_target, ast.Name) and call.args:
            coll = _expr_to_str(call.args[0])
            mem_pred = MembershipPredicate(
                element_variable=val_target.id,
                collection_expr=coll,
                variables=[val_target.id] + _collect_names(call.args[0]),
            )
            guards.append(
                ExtractedGuard(
                    pattern=GuardPattern.Membership,
                    variables=[val_target.id],
                    predicate=mem_pred,
                    source_location=SourceLocation.from_node(node, self.filename),
                    polarity=Polarity.TRUE_BRANCH,
                    raw_source=f"{val_target.id} in {coll}",
                    confidence=0.7,
                )
            )

        return guards

    def _extract_dict_items_guards(
        self,
        target: ast.Tuple,
        call: ast.Call,
        node: ast.For,
    ) -> List[ExtractedGuard]:
        guards: List[ExtractedGuard] = []
        if len(target.elts) != 2:
            return guards

        key_target = target.elts[0]
        val_target = target.elts[1]

        if isinstance(key_target, ast.Name) and isinstance(call.func, ast.Attribute):
            dict_expr = _expr_to_str(call.func.value)
            mem_pred = MembershipPredicate(
                element_variable=key_target.id,
                collection_expr=dict_expr,
                variables=[key_target.id] + _collect_names(call.func.value),
            )
            guards.append(
                ExtractedGuard(
                    pattern=GuardPattern.Membership,
                    variables=[key_target.id],
                    predicate=mem_pred,
                    source_location=SourceLocation.from_node(node, self.filename),
                    polarity=Polarity.TRUE_BRANCH,
                    raw_source=f"{key_target.id} in {dict_expr}",
                    confidence=0.8,
                )
            )

        return guards

    def _extract_zip_guards(
        self,
        target: ast.AST,
        call: ast.Call,
        node: ast.For,
    ) -> List[ExtractedGuard]:
        guards: List[ExtractedGuard] = []
        if not isinstance(target, ast.Tuple):
            return guards

        for i, (elt, arg) in enumerate(zip(target.elts, call.args)):
            if isinstance(elt, ast.Name):
                coll = _expr_to_str(arg)
                mem_pred = MembershipPredicate(
                    element_variable=elt.id,
                    collection_expr=coll,
                    variables=[elt.id] + _collect_names(arg),
                )
                guards.append(
                    ExtractedGuard(
                        pattern=GuardPattern.Membership,
                        variables=[elt.id],
                        predicate=mem_pred,
                        source_location=SourceLocation.from_node(node, self.filename),
                        polarity=Polarity.TRUE_BRANCH,
                        raw_source=f"{elt.id} in {coll}",
                        confidence=0.6,
                    )
                )
        return guards


# ---------------------------------------------------------------------------
# ExceptionGuardExtractor
# ---------------------------------------------------------------------------

class ExceptionGuardExtractor:
    """Extracts type guards from except clauses."""

    def __init__(self, filename: str = "") -> None:
        self.filename = filename

    def extract_except(self, node: ast.ExceptHandler) -> List[ExtractedGuard]:
        """Extract guards from an except clause."""
        guards: List[ExtractedGuard] = []
        if node.type is None:
            return guards

        exc_var = node.name or "<exception>"
        type_names: List[str] = []

        if isinstance(node.type, ast.Tuple):
            for elt in node.type.elts:
                type_names.append(_expr_to_str(elt))
        else:
            type_names.append(_expr_to_str(node.type))

        pred = ExceptionTypePredicate(
            exception_variable=exc_var,
            exception_types=tuple(type_names),
            variables=[exc_var] if node.name else [],
        )
        guards.append(
            ExtractedGuard(
                pattern=GuardPattern.TypeTag,
                variables=[exc_var] if node.name else [],
                predicate=pred,
                source_location=SourceLocation.from_node(node, self.filename),
                polarity=Polarity.TRUE_BRANCH,
                raw_source=f"except {', '.join(type_names)}",
            )
        )

        return guards

    def extract_try(self, node: ast.Try) -> List[ExtractedGuard]:
        """Extract all exception guards from a try statement."""
        guards: List[ExtractedGuard] = []
        for handler in node.handlers:
            guards.extend(self.extract_except(handler))
        return guards


# ---------------------------------------------------------------------------
# AssertionExtractor
# ---------------------------------------------------------------------------

class AssertionExtractor:
    """Extracts assertion statements as guard predicates."""

    def __init__(self, filename: str = "") -> None:
        self.filename = filename
        self._guard_extractor = PythonGuardExtractor(filename=filename)

    def extract_assertion(self, node: ast.Assert) -> List[ExtractedGuard]:
        """Extract guards from an assert statement."""
        guards: List[ExtractedGuard] = []

        # Try to extract structured guard from the test
        inner_guards = self._guard_extractor._extract_guard_from_test(
            node.test, Polarity.TRUE_BRANCH
        )

        msg = ""
        if node.msg and isinstance(node.msg, ast.Constant):
            msg = str(node.msg.value)

        if inner_guards:
            for ig in inner_guards:
                assertion_pred = AssertionPredicate(
                    inner=ig.predicate,
                    message=msg,
                    variables=ig.variables,
                )
                guards.append(
                    ExtractedGuard(
                        pattern=ig.pattern,
                        variables=ig.variables,
                        predicate=assertion_pred,
                        source_location=SourceLocation.from_node(
                            node, self.filename
                        ),
                        polarity=Polarity.TRUE_BRANCH,
                        raw_source=_expr_to_str(node.test),
                    )
                )
        else:
            # Fallback: treat as truthiness assertion
            var_name = _expr_to_str(node.test)
            all_vars = _collect_names(node.test)
            inner_pred = TruthinessPredicate(
                target_variable=var_name, variables=all_vars
            )
            assertion_pred = AssertionPredicate(
                inner=inner_pred, message=msg, variables=all_vars
            )
            guards.append(
                ExtractedGuard(
                    pattern=GuardPattern.Truthiness,
                    variables=all_vars,
                    predicate=assertion_pred,
                    source_location=SourceLocation.from_node(node, self.filename),
                    polarity=Polarity.TRUE_BRANCH,
                    raw_source=var_name,
                )
            )

        return guards

    def extract_all_assertions(self, tree: ast.AST) -> List[ExtractedGuard]:
        """Extract guards from all assert statements in a tree."""
        guards: List[ExtractedGuard] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                guards.extend(self.extract_assertion(node))
        return guards


# ---------------------------------------------------------------------------
# PatternMatchExtractor
# ---------------------------------------------------------------------------

class PatternMatchExtractor:
    """Extracts guards from match/case statements (Python 3.10+)."""

    def __init__(self, filename: str = "") -> None:
        self.filename = filename

    def extract_match(self, node: ast.AST) -> List[ExtractedGuard]:
        """Extract guards from a match statement."""
        guards: List[ExtractedGuard] = []

        if not hasattr(node, "subject") or not hasattr(node, "cases"):
            return guards

        subject = getattr(node, "subject")
        cases = getattr(node, "cases")
        subject_var = _expr_to_str(subject)
        subject_names = _collect_names(subject)

        for case_node in cases:
            pattern = getattr(case_node, "pattern", None)
            guard_expr = getattr(case_node, "guard", None)

            if pattern is not None:
                case_guards = self._extract_pattern(
                    pattern, subject_var, subject_names, node
                )
                guards.extend(case_guards)

            # case ... if guard:
            if guard_expr is not None:
                extractor = PythonGuardExtractor(self.filename)
                guard_preds = extractor._extract_guard_from_test(
                    guard_expr, Polarity.TRUE_BRANCH
                )
                guards.extend(guard_preds)

        return guards

    def _extract_pattern(
        self,
        pattern: ast.AST,
        subject_var: str,
        subject_names: List[str],
        match_node: ast.AST,
    ) -> List[ExtractedGuard]:
        guards: List[ExtractedGuard] = []
        loc = SourceLocation.from_node(pattern, self.filename)

        # MatchValue: case 42:
        if hasattr(ast, "MatchValue") and isinstance(pattern, ast.MatchValue):
            val = _expr_to_str(pattern.value)
            pred = ComparisonPredicate(
                left_expr=subject_var,
                op=ComparisonOp.EQ,
                right_expr=val,
                variables=subject_names,
            )
            guards.append(
                ExtractedGuard(
                    pattern=GuardPattern.Comparison,
                    variables=subject_names,
                    predicate=pred,
                    source_location=loc,
                    polarity=Polarity.TRUE_BRANCH,
                    raw_source=f"case {val}",
                )
            )

        # MatchSingleton: case None / case True / case False
        elif hasattr(ast, "MatchSingleton") and isinstance(pattern, ast.MatchSingleton):
            if pattern.value is None:
                pred = NullityPredicate(
                    target_variable=subject_var,
                    is_none=True,
                    variables=subject_names,
                )
                guards.append(
                    ExtractedGuard(
                        pattern=GuardPattern.Nullity,
                        variables=subject_names,
                        predicate=pred,
                        source_location=loc,
                        polarity=Polarity.TRUE_BRANCH,
                        raw_source="case None",
                    )
                )
            else:
                pred = ComparisonPredicate(
                    left_expr=subject_var,
                    op=ComparisonOp.IS,
                    right_expr=repr(pattern.value),
                    variables=subject_names,
                )
                guards.append(
                    ExtractedGuard(
                        pattern=GuardPattern.Identity,
                        variables=subject_names,
                        predicate=pred,
                        source_location=loc,
                        polarity=Polarity.TRUE_BRANCH,
                        raw_source=f"case {pattern.value!r}",
                    )
                )

        # MatchSequence: case [a, b, c]:
        elif hasattr(ast, "MatchSequence") and isinstance(pattern, ast.MatchSequence):
            bound_vars: List[str] = []
            for p in pattern.patterns:
                if hasattr(ast, "MatchStar") and isinstance(p, ast.MatchStar):
                    if p.name:
                        bound_vars.append(p.name)
                elif hasattr(p, "name") and p.name:
                    bound_vars.append(p.name)
                elif hasattr(ast, "MatchAs") and isinstance(p, ast.MatchAs):
                    if p.name:
                        bound_vars.append(p.name)

            pred = PatternMatchPredicate(
                subject_variable=subject_var,
                pattern_description=f"sequence[{len(pattern.patterns)}]",
                bound_variables=bound_vars,
                variables=subject_names + bound_vars,
            )
            guards.append(
                ExtractedGuard(
                    pattern=GuardPattern.Structural,
                    variables=subject_names + bound_vars,
                    predicate=pred,
                    source_location=loc,
                    polarity=Polarity.TRUE_BRANCH,
                    raw_source="case [...]",
                )
            )

        # MatchMapping: case {'key': value}:
        elif hasattr(ast, "MatchMapping") and isinstance(pattern, ast.MatchMapping):
            keys = [_expr_to_str(k) for k in pattern.keys]
            bound_vars = []
            if hasattr(pattern, "rest") and pattern.rest:
                bound_vars.append(pattern.rest)

            for key_str in keys:
                pred = HasAttrPredicate(
                    target_variable=subject_var,
                    attr_name=key_str,
                    variables=subject_names,
                )
                guards.append(
                    ExtractedGuard(
                        pattern=GuardPattern.Structural,
                        variables=subject_names,
                        predicate=pred,
                        source_location=loc,
                        polarity=Polarity.TRUE_BRANCH,
                        raw_source=f"case {{'{key_str}': ...}}",
                    )
                )

        # MatchClass: case Point(x=1, y=2):
        elif hasattr(ast, "MatchClass") and isinstance(pattern, ast.MatchClass):
            cls_name = _expr_to_str(pattern.cls)
            pred = TypeTagPredicate(
                target_variable=subject_var,
                type_names=(cls_name,),
                variables=subject_names,
            )
            guards.append(
                ExtractedGuard(
                    pattern=GuardPattern.TypeTag,
                    variables=subject_names,
                    predicate=pred,
                    source_location=loc,
                    polarity=Polarity.TRUE_BRANCH,
                    raw_source=f"case {cls_name}(...)",
                )
            )

            # Extract attribute patterns from keyword patterns
            for kwd_attr, kwd_pat in zip(pattern.kwd_attrs, pattern.kwd_patterns):
                sub_guards = self._extract_pattern(
                    kwd_pat, f"{subject_var}.{kwd_attr}", subject_names, match_node
                )
                guards.extend(sub_guards)

        # MatchOr: case pattern1 | pattern2:
        elif hasattr(ast, "MatchOr") and isinstance(pattern, ast.MatchOr):
            child_guards: List[ExtractedGuard] = []
            for p in pattern.patterns:
                child_guards.extend(
                    self._extract_pattern(p, subject_var, subject_names, match_node)
                )
            if child_guards:
                all_vars = sorted(
                    set(v for g in child_guards for v in g.variables)
                )
                disj = DisjunctionPredicate(
                    children=[g.predicate for g in child_guards],
                    variables=all_vars,
                )
                guards.append(
                    ExtractedGuard(
                        pattern=GuardPattern.TypeTag,
                        variables=all_vars,
                        predicate=disj,
                        source_location=loc,
                        polarity=Polarity.TRUE_BRANCH,
                        raw_source="case ... | ...",
                    )
                )

        # MatchAs: case _ as name  /  case _
        elif hasattr(ast, "MatchAs") and isinstance(pattern, ast.MatchAs):
            if pattern.pattern is not None:
                guards.extend(
                    self._extract_pattern(
                        pattern.pattern, subject_var, subject_names, match_node
                    )
                )

        return guards


# ---------------------------------------------------------------------------
# Compound / Convenience API
# ---------------------------------------------------------------------------

class FullGuardExtractor:
    """Orchestrates all sub-extractors into a single pass over the AST."""

    def __init__(self, filename: str = "", source: str = "") -> None:
        self.filename = filename
        self.source = source
        self._main = PythonGuardExtractor(filename, source)
        self._loop = LoopGuardExtractor(filename)
        self._exc = ExceptionGuardExtractor(filename)
        self._assert = AssertionExtractor(filename)
        self._pattern = PatternMatchExtractor(filename)
        self._normalizer = GuardNormalizer()
        self._ranker = GuardRanker()

    def extract_all(self, tree: ast.AST) -> List[ExtractedGuard]:
        """Extract, normalize, and rank all guards."""
        guards = self._main.extract(tree)
        guards = self._normalizer.normalize_all(guards)
        return guards

    def extract_ranked(
        self, tree: ast.AST, top_k: int = 50
    ) -> List[RankedGuard]:
        """Extract, normalize, and return ranked guards."""
        guards = self.extract_all(tree)
        return self._ranker.rank(guards)[:top_k]

    @staticmethod
    def from_source(
        source: str, filename: str = "<string>"
    ) -> FullGuardExtractor:
        ext = FullGuardExtractor(filename=filename, source=source)
        return ext

    def extract_and_track(
        self, tree: ast.AST
    ) -> Tuple[List[ExtractedGuard], Dict[int, List[ExtractedGuard]]]:
        """Extract guards and build scope tracking map."""
        guards = self.extract_all(tree)
        tracker = ConditionalGuardTracker(self.filename)
        scope_map = tracker.track(tree)
        return guards, scope_map


# ---------------------------------------------------------------------------
# Utility: predicate pretty-printing
# ---------------------------------------------------------------------------

class PredicatePrinter:
    """Pretty-print predicate templates for debugging."""

    def format(self, pred: PredicateTemplate, indent: int = 0) -> str:
        prefix = "  " * indent

        if isinstance(pred, TypeTagPredicate):
            return f"{prefix}TypeTag({pred.target_variable} :: {' | '.join(pred.type_names)})"

        if isinstance(pred, NullityPredicate):
            op = "is" if pred.is_none else "is not"
            return f"{prefix}Nullity({pred.target_variable} {op} None)"

        if isinstance(pred, TruthinessPredicate):
            return f"{prefix}Truthy({pred.target_variable})"

        if isinstance(pred, HasAttrPredicate):
            return f"{prefix}HasAttr({pred.target_variable}, '{pred.attr_name}')"

        if isinstance(pred, ComparisonPredicate):
            return f"{prefix}Compare({pred.left_expr} {pred.op.value} {pred.right_expr})"

        if isinstance(pred, LenComparisonPredicate):
            return f"{prefix}LenCmp({pred.index_variable} {pred.op.value} len({pred.collection_variable}))"

        if isinstance(pred, MembershipPredicate):
            return f"{prefix}Member({pred.element_variable} in {pred.collection_expr})"

        if isinstance(pred, CallablePredicate):
            return f"{prefix}Callable({pred.target_variable})"

        if isinstance(pred, IdentityPredicate):
            op = "is" if pred.is_positive else "is not"
            return f"{prefix}Identity({pred.left_variable} {op} {pred.right_expr})"

        if isinstance(pred, ConjunctionPredicate):
            parts = [self.format(c, indent + 1) for c in pred.children]
            return f"{prefix}AND(\n" + ",\n".join(parts) + f"\n{prefix})"

        if isinstance(pred, DisjunctionPredicate):
            parts = [self.format(c, indent + 1) for c in pred.children]
            return f"{prefix}OR(\n" + ",\n".join(parts) + f"\n{prefix})"

        if isinstance(pred, NegationPredicate):
            inner = self.format(pred.child, indent + 1) if pred.child else "?"
            return f"{prefix}NOT(\n{inner}\n{prefix})"

        if isinstance(pred, RangeBoundPredicate):
            bounds = f"{pred.lower_bound}..{pred.upper_bound}"
            if pred.step:
                bounds += f" step {pred.step}"
            return f"{prefix}Range({pred.loop_variable} in {bounds})"

        if isinstance(pred, ExceptionTypePredicate):
            types = " | ".join(pred.exception_types)
            return f"{prefix}Except({pred.exception_variable} :: {types})"

        if isinstance(pred, AssertionPredicate):
            inner = self.format(pred.inner, indent + 1) if pred.inner else "?"
            msg = f" [{pred.message}]" if pred.message else ""
            return f"{prefix}Assert{msg}(\n{inner}\n{prefix})"

        if isinstance(pred, PatternMatchPredicate):
            return f"{prefix}Pattern({pred.subject_variable}: {pred.pattern_description})"

        return f"{prefix}{pred.kind.name}({', '.join(pred.variables)})"


# ---------------------------------------------------------------------------
# Guard serialization / deserialization
# ---------------------------------------------------------------------------

class GuardSerializer:
    """Serialize/deserialize guards to/from dictionaries."""

    def to_dict(self, guard: ExtractedGuard) -> Dict[str, Any]:
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
        }

    def from_dict(self, data: Dict[str, Any]) -> ExtractedGuard:
        loc_data = data.get("source_location", {})
        loc = SourceLocation(
            filename=loc_data.get("filename", ""),
            line=loc_data.get("line", 0),
            col=loc_data.get("col", 0),
            end_line=loc_data.get("end_line", 0),
            end_col=loc_data.get("end_col", 0),
        )
        pred = self._pred_from_dict(data.get("predicate", {}))
        return ExtractedGuard(
            pattern=GuardPattern[data["pattern"]],
            variables=data.get("variables", []),
            predicate=pred,
            source_location=loc,
            polarity=Polarity[data.get("polarity", "TRUE_BRANCH")],
            raw_source=data.get("raw_source", ""),
            confidence=data.get("confidence", 1.0),
        )

    def _pred_to_dict(self, pred: PredicateTemplate) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "kind": pred.kind.name,
            "variables": pred.variables,
        }

        if isinstance(pred, TypeTagPredicate):
            d["target_variable"] = pred.target_variable
            d["type_names"] = list(pred.type_names)
        elif isinstance(pred, NullityPredicate):
            d["target_variable"] = pred.target_variable
            d["is_none"] = pred.is_none
        elif isinstance(pred, TruthinessPredicate):
            d["target_variable"] = pred.target_variable
        elif isinstance(pred, HasAttrPredicate):
            d["target_variable"] = pred.target_variable
            d["attr_name"] = pred.attr_name
        elif isinstance(pred, ComparisonPredicate):
            d["left_expr"] = pred.left_expr
            d["op"] = pred.op.value
            d["right_expr"] = pred.right_expr
        elif isinstance(pred, LenComparisonPredicate):
            d["index_variable"] = pred.index_variable
            d["collection_variable"] = pred.collection_variable
            d["op"] = pred.op.value
        elif isinstance(pred, MembershipPredicate):
            d["element_variable"] = pred.element_variable
            d["collection_expr"] = pred.collection_expr
        elif isinstance(pred, CallablePredicate):
            d["target_variable"] = pred.target_variable
        elif isinstance(pred, IdentityPredicate):
            d["left_variable"] = pred.left_variable
            d["right_expr"] = pred.right_expr
            d["is_positive"] = pred.is_positive
        elif isinstance(pred, ConjunctionPredicate):
            d["children"] = [self._pred_to_dict(c) for c in pred.children]
        elif isinstance(pred, DisjunctionPredicate):
            d["children"] = [self._pred_to_dict(c) for c in pred.children]
        elif isinstance(pred, NegationPredicate):
            d["child"] = self._pred_to_dict(pred.child) if pred.child else None
        elif isinstance(pred, RangeBoundPredicate):
            d["loop_variable"] = pred.loop_variable
            d["lower_bound"] = pred.lower_bound
            d["upper_bound"] = pred.upper_bound
            d["step"] = pred.step
        elif isinstance(pred, ExceptionTypePredicate):
            d["exception_variable"] = pred.exception_variable
            d["exception_types"] = list(pred.exception_types)
        elif isinstance(pred, AssertionPredicate):
            d["inner"] = self._pred_to_dict(pred.inner) if pred.inner else None
            d["message"] = pred.message
        elif isinstance(pred, PatternMatchPredicate):
            d["subject_variable"] = pred.subject_variable
            d["pattern_description"] = pred.pattern_description
            d["bound_variables"] = pred.bound_variables

        return d

    def _pred_from_dict(self, data: Dict[str, Any]) -> PredicateTemplate:
        kind_name = data.get("kind", "Truthiness")
        variables = data.get("variables", [])

        if kind_name == "TypeTag":
            return TypeTagPredicate(
                target_variable=data.get("target_variable", ""),
                type_names=tuple(data.get("type_names", [])),
                variables=variables,
            )
        elif kind_name == "Nullity":
            return NullityPredicate(
                target_variable=data.get("target_variable", ""),
                is_none=data.get("is_none", True),
                variables=variables,
            )
        elif kind_name == "Truthiness":
            return TruthinessPredicate(
                target_variable=data.get("target_variable", ""),
                variables=variables,
            )
        elif kind_name == "HasAttr":
            return HasAttrPredicate(
                target_variable=data.get("target_variable", ""),
                attr_name=data.get("attr_name", ""),
                variables=variables,
            )
        elif kind_name == "Comparison":
            op_str = data.get("op", "==")
            op = ComparisonOp(op_str)
            return ComparisonPredicate(
                left_expr=data.get("left_expr", ""),
                op=op,
                right_expr=data.get("right_expr", ""),
                variables=variables,
            )
        elif kind_name == "LenComparison":
            op_str = data.get("op", "<")
            op = ComparisonOp(op_str)
            return LenComparisonPredicate(
                index_variable=data.get("index_variable", ""),
                collection_variable=data.get("collection_variable", ""),
                op=op,
                variables=variables,
            )
        elif kind_name == "Membership":
            return MembershipPredicate(
                element_variable=data.get("element_variable", ""),
                collection_expr=data.get("collection_expr", ""),
                variables=variables,
            )
        elif kind_name == "Callable":
            return CallablePredicate(
                target_variable=data.get("target_variable", ""),
                variables=variables,
            )
        elif kind_name == "Identity":
            return IdentityPredicate(
                left_variable=data.get("left_variable", ""),
                right_expr=data.get("right_expr", ""),
                is_positive=data.get("is_positive", True),
                variables=variables,
            )
        elif kind_name == "Conjunction":
            children = [
                self._pred_from_dict(c) for c in data.get("children", [])
            ]
            return ConjunctionPredicate(children=children, variables=variables)
        elif kind_name == "Disjunction":
            children = [
                self._pred_from_dict(c) for c in data.get("children", [])
            ]
            return DisjunctionPredicate(children=children, variables=variables)
        elif kind_name == "Negation":
            child_data = data.get("child")
            child = self._pred_from_dict(child_data) if child_data else None
            return NegationPredicate(child=child, variables=variables)
        elif kind_name == "RangeBound":
            return RangeBoundPredicate(
                loop_variable=data.get("loop_variable", ""),
                lower_bound=data.get("lower_bound"),
                upper_bound=data.get("upper_bound"),
                step=data.get("step"),
                variables=variables,
            )
        elif kind_name == "ExceptionType":
            return ExceptionTypePredicate(
                exception_variable=data.get("exception_variable", ""),
                exception_types=tuple(data.get("exception_types", [])),
                variables=variables,
            )
        elif kind_name == "Assertion":
            inner_data = data.get("inner")
            inner = self._pred_from_dict(inner_data) if inner_data else None
            return AssertionPredicate(
                inner=inner,
                message=data.get("message", ""),
                variables=variables,
            )
        elif kind_name == "PatternMatch":
            return PatternMatchPredicate(
                subject_variable=data.get("subject_variable", ""),
                pattern_description=data.get("pattern_description", ""),
                bound_variables=data.get("bound_variables", []),
                variables=variables,
            )

        return PredicateTemplate(kind=PredicateKind[kind_name], variables=variables)


# ---------------------------------------------------------------------------
# Guard statistics
# ---------------------------------------------------------------------------

@dataclass
class GuardStatistics:
    """Summary statistics over a set of extracted guards."""
    total_count: int = 0
    by_pattern: Dict[str, int] = field(default_factory=dict)
    by_predicate_kind: Dict[str, int] = field(default_factory=dict)
    unique_variables: Set[str] = field(default_factory=set)
    avg_confidence: float = 0.0
    most_guarded_variables: List[Tuple[str, int]] = field(default_factory=list)

    @staticmethod
    def compute(guards: List[ExtractedGuard]) -> GuardStatistics:
        stats = GuardStatistics()
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

            conf_sum += g.confidence

        if guards:
            stats.avg_confidence = conf_sum / len(guards)

        stats.most_guarded_variables = sorted(
            var_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:20]

        return stats


# ---------------------------------------------------------------------------
# Guard filtering utilities
# ---------------------------------------------------------------------------

class GuardFilter:
    """Utilities for filtering guards by various criteria."""

    @staticmethod
    def by_pattern(
        guards: List[ExtractedGuard], pattern: GuardPattern
    ) -> List[ExtractedGuard]:
        return [g for g in guards if g.pattern == pattern]

    @staticmethod
    def by_variable(
        guards: List[ExtractedGuard], variable: str
    ) -> List[ExtractedGuard]:
        return [g for g in guards if variable in g.variables]

    @staticmethod
    def by_polarity(
        guards: List[ExtractedGuard], polarity: Polarity
    ) -> List[ExtractedGuard]:
        return [g for g in guards if g.polarity == polarity]

    @staticmethod
    def by_predicate_kind(
        guards: List[ExtractedGuard], kind: PredicateKind
    ) -> List[ExtractedGuard]:
        return [g for g in guards if g.predicate.kind == kind]

    @staticmethod
    def by_confidence(
        guards: List[ExtractedGuard], min_confidence: float = 0.5
    ) -> List[ExtractedGuard]:
        return [g for g in guards if g.confidence >= min_confidence]

    @staticmethod
    def type_guards(guards: List[ExtractedGuard]) -> List[ExtractedGuard]:
        return [
            g for g in guards
            if g.pattern == GuardPattern.TypeTag
            or isinstance(g.predicate, TypeTagPredicate)
        ]

    @staticmethod
    def null_guards(guards: List[ExtractedGuard]) -> List[ExtractedGuard]:
        return [
            g for g in guards
            if g.pattern == GuardPattern.Nullity
            or isinstance(g.predicate, NullityPredicate)
        ]

    @staticmethod
    def comparison_guards(guards: List[ExtractedGuard]) -> List[ExtractedGuard]:
        return [
            g for g in guards
            if g.pattern == GuardPattern.Comparison
            or isinstance(g.predicate, (ComparisonPredicate, LenComparisonPredicate))
        ]

    @staticmethod
    def structural_guards(guards: List[ExtractedGuard]) -> List[ExtractedGuard]:
        return [
            g for g in guards
            if g.pattern == GuardPattern.Structural
            or isinstance(g.predicate, HasAttrPredicate)
        ]


# ---------------------------------------------------------------------------
# Guard dependency analysis
# ---------------------------------------------------------------------------

class GuardDependencyAnalyzer:
    """Analyzes dependencies between guards based on shared variables."""

    def compute_dependencies(
        self, guards: List[ExtractedGuard]
    ) -> Dict[int, Set[int]]:
        """Return a map from guard index to indices of guards that share
        variables (and thus may interact)."""
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
        self, guards: List[ExtractedGuard]
    ) -> List[List[ExtractedGuard]]:
        """Group guards into connected components by variable sharing."""
        deps = self.compute_dependencies(guards)
        visited: Set[int] = set()
        components: List[List[ExtractedGuard]] = []

        def dfs(idx: int, component: List[int]) -> None:
            if idx in visited:
                return
            visited.add(idx)
            component.append(idx)
            for neighbor in deps.get(idx, set()):
                dfs(neighbor, component)

        for i in range(len(guards)):
            if i not in visited:
                comp: List[int] = []
                dfs(i, comp)
                components.append([guards[j] for j in sorted(comp)])

        return components

    def independent_guard_sets(
        self, guards: List[ExtractedGuard]
    ) -> List[List[ExtractedGuard]]:
        """Return groups of guards that are completely independent
        (share no variables)."""
        return self.connected_components(guards)


# ---------------------------------------------------------------------------
# Guard template instantiation for CEGAR
# ---------------------------------------------------------------------------

@dataclass
class CEGARSeed:
    """A predicate seed for CEGAR loop initialization."""
    predicate: PredicateTemplate
    source_guard: ExtractedGuard
    priority: float = 0.0
    category: str = ""


class CEGARSeeder:
    """Produces initial predicate seeds for CEGAR from extracted guards."""

    def __init__(self) -> None:
        self._ranker = GuardRanker()
        self._normalizer = GuardNormalizer()

    def generate_seeds(
        self, guards: List[ExtractedGuard], max_seeds: int = 100
    ) -> List[CEGARSeed]:
        """Generate CEGAR predicate seeds from guards."""
        normalized = self._normalizer.normalize_all(guards)
        ranked = self._ranker.rank(normalized)

        seeds: List[CEGARSeed] = []
        seen_sigs: Set[str] = set()

        for rg in ranked[:max_seeds]:
            sig = self._seed_signature(rg.guard)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)

            seeds.append(
                CEGARSeed(
                    predicate=rg.guard.predicate,
                    source_guard=rg.guard,
                    priority=rg.score,
                    category=rg.guard.pattern.name,
                )
            )

            # Also generate implied predicates
            implied = self._generate_implied(rg.guard)
            for imp_pred, imp_cat in implied:
                imp_sig = f"{imp_cat}:{','.join(imp_pred.variables)}"
                if imp_sig not in seen_sigs:
                    seen_sigs.add(imp_sig)
                    seeds.append(
                        CEGARSeed(
                            predicate=imp_pred,
                            source_guard=rg.guard,
                            priority=rg.score * 0.5,
                            category=imp_cat,
                        )
                    )

        seeds.sort(key=lambda s: s.priority, reverse=True)
        return seeds[:max_seeds]

    def _generate_implied(
        self, guard: ExtractedGuard
    ) -> List[Tuple[PredicateTemplate, str]]:
        """Generate implied predicates from a guard."""
        implied: List[Tuple[PredicateTemplate, str]] = []

        # isinstance → also implies not None
        if isinstance(guard.predicate, TypeTagPredicate):
            null_pred = NullityPredicate(
                target_variable=guard.predicate.target_variable,
                is_none=False,
                variables=guard.predicate.variables,
            )
            implied.append((null_pred, "implied_not_none"))

        # x < len(arr) → also implies arr is not None
        if isinstance(guard.predicate, LenComparisonPredicate):
            null_pred = NullityPredicate(
                target_variable=guard.predicate.collection_variable,
                is_none=False,
                variables=[guard.predicate.collection_variable],
            )
            implied.append((null_pred, "implied_collection_not_none"))

        # hasattr(x, 'attr') → x is not None
        if isinstance(guard.predicate, HasAttrPredicate):
            null_pred = NullityPredicate(
                target_variable=guard.predicate.target_variable,
                is_none=False,
                variables=[guard.predicate.target_variable],
            )
            implied.append((null_pred, "implied_not_none"))

        return implied

    @staticmethod
    def _seed_signature(guard: ExtractedGuard) -> str:
        return (
            f"{guard.pattern.name}:{guard.predicate.kind.name}:"
            f"{','.join(sorted(guard.variables))}"
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def extract_guards(source: str, filename: str = "<string>") -> List[ExtractedGuard]:
    """Extract all guards from Python source code."""
    tree = ast.parse(source, filename=filename)
    extractor = FullGuardExtractor(filename=filename, source=source)
    return extractor.extract_all(tree)


def extract_and_rank(
    source: str, filename: str = "<string>", top_k: int = 50
) -> List[RankedGuard]:
    """Extract and rank guards from Python source code."""
    tree = ast.parse(source, filename=filename)
    extractor = FullGuardExtractor(filename=filename, source=source)
    return extractor.extract_ranked(tree, top_k=top_k)


def generate_cegar_seeds(
    source: str, filename: str = "<string>", max_seeds: int = 100
) -> List[CEGARSeed]:
    """Generate CEGAR seeds from Python source code."""
    guards = extract_guards(source, filename)
    seeder = CEGARSeeder()
    return seeder.generate_seeds(guards, max_seeds=max_seeds)


def print_guards(source: str, filename: str = "<string>") -> None:
    """Pretty-print all extracted guards (for debugging)."""
    guards = extract_guards(source, filename)
    printer = PredicatePrinter()
    for i, g in enumerate(guards):
        print(f"Guard #{i + 1} [{g.pattern.name}] @ {g.source_location.line}:{g.source_location.col}")
        print(f"  Polarity: {g.polarity.name}")
        print(f"  Variables: {g.variables}")
        print(f"  Raw: {g.raw_source}")
        print(f"  Predicate: {printer.format(g.predicate)}")
        print()
