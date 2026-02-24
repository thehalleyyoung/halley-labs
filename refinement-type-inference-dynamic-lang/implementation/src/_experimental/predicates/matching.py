"""
Guard Pattern Matching.

Matches IR guard nodes to predicate templates from the predicate language P.
Implements pattern rules for isinstance, type checks, null checks, comparisons,
length checks, attribute checks, truthiness, and compound guards.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from .templates import (
    AtomicPredicate,
    ComparisonOp,
    ComparisonPredicate,
    Conjunction,
    Disjunction,
    FalsePredicate,
    HasAttrPredicate,
    Implication,
    LenTerm,
    LinearExpression,
    Negation,
    NullityPredicate,
    PredicateFactory,
    PredicateNormalizer,
    PredicateSimplifier,
    PredicateSort,
    PredicateTemplate,
    Term,
    TruePredicate,
    TruthinessPredicate,
    TypeTagPredicate,
)


# ---------------------------------------------------------------------------
# Guard IR representation
# ---------------------------------------------------------------------------

class GuardKind(Enum):
    """Kinds of IR guard nodes."""
    ISINSTANCE = auto()
    TYPEOF = auto()
    NULL_CHECK = auto()
    COMPARISON = auto()
    LEN_COMPARISON = auto()
    HASATTR = auto()
    TRUTHINESS = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    CALL = auto()
    ATTRIBUTE_ACCESS = auto()
    SUBSCRIPT = auto()
    IN_CHECK = auto()
    CHAINED_COMPARISON = auto()
    UNKNOWN = auto()


@dataclass
class GuardNode:
    """An IR guard node to be matched against predicate templates."""
    kind: GuardKind
    source_text: str = ""
    children: List["GuardNode"] = field(default_factory=list)
    variable: Optional[str] = None
    type_name: Optional[str] = None
    operator: Optional[str] = None
    value: Optional[Any] = None
    attribute: Optional[str] = None
    function_name: Optional[str] = None
    negated: bool = False
    line: int = 0
    column: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_compound(self) -> bool:
        return self.kind in (GuardKind.AND, GuardKind.OR, GuardKind.NOT)

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def variables(self) -> Set[str]:
        result: Set[str] = set()
        if self.variable:
            result.add(self.variable)
        for c in self.children:
            result |= c.variables()
        return result

    def __repr__(self) -> str:
        return f"GuardNode({self.kind.name}, source={self.source_text!r})"


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Result of matching a guard node to a predicate template."""
    predicate: PredicateTemplate
    confidence: float
    source_guard: GuardNode
    variables: FrozenSet[str]
    pattern_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.9

    def is_medium_confidence(self) -> bool:
        return 0.5 <= self.confidence < 0.9

    def is_low_confidence(self) -> bool:
        return self.confidence < 0.5

    def __repr__(self) -> str:
        return (
            f"MatchResult(pred={self.predicate.pretty_print()!r}, "
            f"conf={self.confidence:.2f}, pattern={self.pattern_name!r})"
        )


# ---------------------------------------------------------------------------
# Pattern Rule ABC
# ---------------------------------------------------------------------------

class PatternRule(ABC):
    """Base class for pattern matching rules."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this pattern rule."""
        ...

    @abstractmethod
    def can_match(self, guard: GuardNode) -> bool:
        """Quick check if this rule might apply to the guard."""
        ...

    @abstractmethod
    def match(self, guard: GuardNode) -> Optional[MatchResult]:
        """Attempt to match the guard, returning a MatchResult or None."""
        ...

    def priority(self) -> int:
        """Higher priority rules are tried first (default 0)."""
        return 0

    def __repr__(self) -> str:
        return f"PatternRule({self.name()!r})"


# ---------------------------------------------------------------------------
# IsInstancePattern
# ---------------------------------------------------------------------------

class IsInstancePattern(PatternRule):
    """Matches isinstance(x, T) calls."""

    def name(self) -> str:
        return "isinstance"

    def can_match(self, guard: GuardNode) -> bool:
        return guard.kind == GuardKind.ISINSTANCE

    def match(self, guard: GuardNode) -> Optional[MatchResult]:
        if not self.can_match(guard):
            return None
        var = guard.variable
        type_name = guard.type_name
        if var is None or type_name is None:
            return None
        positive = not guard.negated
        pred = TypeTagPredicate(variable=var, type_tag=type_name, positive=positive)
        return MatchResult(
            predicate=pred,
            confidence=1.0,
            source_guard=guard,
            variables=frozenset({var}),
            pattern_name=self.name(),
        )

    def priority(self) -> int:
        return 10

    @staticmethod
    def match_from_source(source: str) -> Optional[MatchResult]:
        """Parse isinstance(x, T) directly from source text."""
        pattern = r"isinstance\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)"
        m = re.match(pattern, source.strip())
        if m:
            var, type_name = m.group(1), m.group(2)
            guard = GuardNode(
                kind=GuardKind.ISINSTANCE,
                source_text=source,
                variable=var,
                type_name=type_name,
            )
            pred = TypeTagPredicate(variable=var, type_tag=type_name, positive=True)
            return MatchResult(
                predicate=pred,
                confidence=1.0,
                source_guard=guard,
                variables=frozenset({var}),
                pattern_name="isinstance",
            )
        # not isinstance(x, T)
        neg_pattern = r"not\s+isinstance\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)"
        m = re.match(neg_pattern, source.strip())
        if m:
            var, type_name = m.group(1), m.group(2)
            guard = GuardNode(
                kind=GuardKind.ISINSTANCE,
                source_text=source,
                variable=var,
                type_name=type_name,
                negated=True,
            )
            pred = TypeTagPredicate(variable=var, type_tag=type_name, positive=False)
            return MatchResult(
                predicate=pred,
                confidence=1.0,
                source_guard=guard,
                variables=frozenset({var}),
                pattern_name="isinstance",
            )
        return None


# ---------------------------------------------------------------------------
# TypeOfPattern
# ---------------------------------------------------------------------------

class TypeOfPattern(PatternRule):
    """Matches typeof x === 'type' (TypeScript/JavaScript)."""

    def name(self) -> str:
        return "typeof"

    def can_match(self, guard: GuardNode) -> bool:
        return guard.kind == GuardKind.TYPEOF

    def match(self, guard: GuardNode) -> Optional[MatchResult]:
        if not self.can_match(guard):
            return None
        var = guard.variable
        type_name = guard.type_name
        if var is None or type_name is None:
            return None
        tag_map = {
            "number": "int",
            "string": "str",
            "boolean": "bool",
            "object": "object",
            "undefined": "NoneType",
            "function": "function",
            "symbol": "symbol",
            "bigint": "bigint",
        }
        mapped = tag_map.get(type_name, type_name)
        positive = guard.operator in ("===", "==") if guard.operator else not guard.negated
        pred = TypeTagPredicate(variable=var, type_tag=mapped, positive=positive)
        return MatchResult(
            predicate=pred,
            confidence=0.95,
            source_guard=guard,
            variables=frozenset({var}),
            pattern_name=self.name(),
        )

    def priority(self) -> int:
        return 9

    @staticmethod
    def match_from_source(source: str) -> Optional[MatchResult]:
        """Parse typeof x === 'type' from source text."""
        pattern = r"typeof\s+(\w+)\s*(===|!==|==|!=)\s*['\"](\w+)['\"]"
        m = re.search(pattern, source.strip())
        if m:
            var, op, type_name = m.group(1), m.group(2), m.group(3)
            positive = op in ("===", "==")
            guard = GuardNode(
                kind=GuardKind.TYPEOF,
                source_text=source,
                variable=var,
                type_name=type_name,
                operator=op,
                negated=not positive,
            )
            tag_map = {
                "number": "int",
                "string": "str",
                "boolean": "bool",
                "object": "object",
                "undefined": "NoneType",
            }
            mapped = tag_map.get(type_name, type_name)
            pred = TypeTagPredicate(variable=var, type_tag=mapped, positive=positive)
            return MatchResult(
                predicate=pred,
                confidence=0.95,
                source_guard=guard,
                variables=frozenset({var}),
                pattern_name="typeof",
            )
        return None


# ---------------------------------------------------------------------------
# NullCheckPattern
# ---------------------------------------------------------------------------

class NullCheckPattern(PatternRule):
    """Matches x is None, x === null, x != null, etc."""

    def name(self) -> str:
        return "null_check"

    def can_match(self, guard: GuardNode) -> bool:
        return guard.kind == GuardKind.NULL_CHECK

    def match(self, guard: GuardNode) -> Optional[MatchResult]:
        if not self.can_match(guard):
            return None
        var = guard.variable
        if var is None:
            return None
        is_null = not guard.negated
        pred = NullityPredicate(variable=var, is_null=is_null)
        return MatchResult(
            predicate=pred,
            confidence=1.0,
            source_guard=guard,
            variables=frozenset({var}),
            pattern_name=self.name(),
        )

    def priority(self) -> int:
        return 8

    @staticmethod
    def match_from_source(source: str) -> Optional[MatchResult]:
        """Parse null checks from source text."""
        src = source.strip()
        # Python: x is None / x is not None
        pattern_is_none = r"(\w+)\s+is\s+None"
        pattern_is_not_none = r"(\w+)\s+is\s+not\s+None"
        m = re.match(pattern_is_not_none, src)
        if m:
            var = m.group(1)
            guard = GuardNode(kind=GuardKind.NULL_CHECK, source_text=src, variable=var, negated=True)
            pred = NullityPredicate(variable=var, is_null=False)
            return MatchResult(
                predicate=pred, confidence=1.0, source_guard=guard,
                variables=frozenset({var}), pattern_name="null_check",
            )
        m = re.match(pattern_is_none, src)
        if m:
            var = m.group(1)
            guard = GuardNode(kind=GuardKind.NULL_CHECK, source_text=src, variable=var, negated=False)
            pred = NullityPredicate(variable=var, is_null=True)
            return MatchResult(
                predicate=pred, confidence=1.0, source_guard=guard,
                variables=frozenset({var}), pattern_name="null_check",
            )
        # JS/TS: x === null / x !== null
        pattern_eq_null = r"(\w+)\s*(===|==)\s*null"
        pattern_ne_null = r"(\w+)\s*(!==|!=)\s*null"
        m = re.match(pattern_ne_null, src)
        if m:
            var = m.group(1)
            guard = GuardNode(kind=GuardKind.NULL_CHECK, source_text=src, variable=var, negated=True)
            pred = NullityPredicate(variable=var, is_null=False)
            return MatchResult(
                predicate=pred, confidence=0.95, source_guard=guard,
                variables=frozenset({var}), pattern_name="null_check",
            )
        m = re.match(pattern_eq_null, src)
        if m:
            var = m.group(1)
            guard = GuardNode(kind=GuardKind.NULL_CHECK, source_text=src, variable=var, negated=False)
            pred = NullityPredicate(variable=var, is_null=True)
            return MatchResult(
                predicate=pred, confidence=0.95, source_guard=guard,
                variables=frozenset({var}), pattern_name="null_check",
            )
        # x === undefined / x !== undefined
        pattern_eq_undef = r"(\w+)\s*(===|==)\s*undefined"
        pattern_ne_undef = r"(\w+)\s*(!==|!=)\s*undefined"
        m = re.match(pattern_ne_undef, src)
        if m:
            var = m.group(1)
            guard = GuardNode(kind=GuardKind.NULL_CHECK, source_text=src, variable=var, negated=True)
            pred = NullityPredicate(variable=var, is_null=False)
            return MatchResult(
                predicate=pred, confidence=0.9, source_guard=guard,
                variables=frozenset({var}), pattern_name="null_check",
            )
        m = re.match(pattern_eq_undef, src)
        if m:
            var = m.group(1)
            guard = GuardNode(kind=GuardKind.NULL_CHECK, source_text=src, variable=var, negated=False)
            pred = NullityPredicate(variable=var, is_null=True)
            return MatchResult(
                predicate=pred, confidence=0.9, source_guard=guard,
                variables=frozenset({var}), pattern_name="null_check",
            )
        return None


# ---------------------------------------------------------------------------
# ComparisonPattern
# ---------------------------------------------------------------------------

class ComparisonPattern(PatternRule):
    """Matches x op expr comparisons."""

    _OP_MAP: Dict[str, ComparisonOp] = {
        "<": ComparisonOp.Lt,
        "<=": ComparisonOp.Le,
        "==": ComparisonOp.Eq,
        "!=": ComparisonOp.Ne,
        ">=": ComparisonOp.Ge,
        ">": ComparisonOp.Gt,
        "===": ComparisonOp.Eq,
        "!==": ComparisonOp.Ne,
    }

    def name(self) -> str:
        return "comparison"

    def can_match(self, guard: GuardNode) -> bool:
        return guard.kind == GuardKind.COMPARISON

    def match(self, guard: GuardNode) -> Optional[MatchResult]:
        if not self.can_match(guard):
            return None
        if len(guard.children) < 2:
            return None
        op_str = guard.operator
        if op_str is None or op_str not in self._OP_MAP:
            return None
        op = self._OP_MAP[op_str]
        left = self._guard_to_expr(guard.children[0])
        right = self._guard_to_expr(guard.children[1])
        if left is None or right is None:
            return None
        if guard.negated:
            op = op.negate()
        pred = ComparisonPredicate(left=left, op=op, right=right)
        all_vars = pred.variables()
        return MatchResult(
            predicate=pred,
            confidence=0.95,
            source_guard=guard,
            variables=all_vars,
            pattern_name=self.name(),
        )

    def priority(self) -> int:
        return 7

    @staticmethod
    def _guard_to_expr(node: GuardNode) -> Optional[LinearExpression]:
        """Convert a guard node to a linear expression."""
        if node.variable is not None:
            return LinearExpression.from_variable(node.variable)
        if node.value is not None and isinstance(node.value, (int, float)):
            return LinearExpression.from_int(int(node.value))
        if node.kind == GuardKind.CALL and node.function_name == "len" and node.variable:
            return LinearExpression.from_len(node.variable)
        return None

    @staticmethod
    def match_from_source(source: str) -> Optional[MatchResult]:
        """Parse comparisons from source text."""
        ops = ["<=", ">=", "!=", "!==", "===", "==", "<", ">"]
        src = source.strip()
        for op_str in ops:
            parts = src.split(op_str, 1)
            if len(parts) == 2:
                left_s, right_s = parts[0].strip(), parts[1].strip()
                left = ComparisonPattern._parse_expr(left_s)
                right = ComparisonPattern._parse_expr(right_s)
                if left is not None and right is not None:
                    op = ComparisonPattern._OP_MAP.get(op_str)
                    if op is None:
                        continue
                    pred = ComparisonPredicate(left=left, op=op, right=right)
                    guard = GuardNode(
                        kind=GuardKind.COMPARISON,
                        source_text=src,
                        operator=op_str,
                    )
                    return MatchResult(
                        predicate=pred,
                        confidence=0.9,
                        source_guard=guard,
                        variables=pred.variables(),
                        pattern_name="comparison",
                    )
        return None

    @staticmethod
    def _parse_expr(s: str) -> Optional[LinearExpression]:
        """Parse a simple expression string to LinearExpression."""
        s = s.strip()
        # len(x)
        len_match = re.match(r"len\s*\(\s*(\w+)\s*\)", s)
        if len_match:
            return LinearExpression.from_len(len_match.group(1))
        # integer constant
        try:
            val = int(s)
            return LinearExpression.from_int(val)
        except ValueError:
            pass
        # simple variable
        if re.match(r"^\w+$", s):
            return LinearExpression.from_variable(s)
        # x + c or x - c
        add_match = re.match(r"(\w+)\s*\+\s*(\d+)", s)
        if add_match:
            var = add_match.group(1)
            c = int(add_match.group(2))
            return LinearExpression.from_variable(var).add(LinearExpression.from_int(c))
        sub_match = re.match(r"(\w+)\s*-\s*(\d+)", s)
        if sub_match:
            var = sub_match.group(1)
            c = int(sub_match.group(2))
            return LinearExpression.from_variable(var).subtract(LinearExpression.from_int(c))
        # c * x
        mul_match = re.match(r"(\d+)\s*\*\s*(\w+)", s)
        if mul_match:
            c = int(mul_match.group(1))
            var = mul_match.group(2)
            return LinearExpression.from_variable(var).scale(c)
        return None


# ---------------------------------------------------------------------------
# LenComparisonPattern
# ---------------------------------------------------------------------------

class LenComparisonPattern(PatternRule):
    """Matches comparisons involving len()."""

    def name(self) -> str:
        return "len_comparison"

    def can_match(self, guard: GuardNode) -> bool:
        return guard.kind == GuardKind.LEN_COMPARISON

    def match(self, guard: GuardNode) -> Optional[MatchResult]:
        if not self.can_match(guard):
            return None
        if len(guard.children) < 2:
            return None
        op_str = guard.operator
        if op_str is None:
            return None
        op = ComparisonPattern._OP_MAP.get(op_str)
        if op is None:
            return None
        left_node = guard.children[0]
        right_node = guard.children[1]
        left = self._node_to_expr(left_node)
        right = self._node_to_expr(right_node)
        if left is None or right is None:
            return None
        pred = ComparisonPredicate(left=left, op=op, right=right)
        return MatchResult(
            predicate=pred,
            confidence=0.95,
            source_guard=guard,
            variables=pred.variables(),
            pattern_name=self.name(),
        )

    def _node_to_expr(self, node: GuardNode) -> Optional[LinearExpression]:
        if node.function_name == "len" and node.variable:
            return LinearExpression.from_len(node.variable)
        if node.variable:
            return LinearExpression.from_variable(node.variable)
        if node.value is not None and isinstance(node.value, int):
            return LinearExpression.from_int(node.value)
        return None

    def priority(self) -> int:
        return 8

    @staticmethod
    def match_from_source(source: str) -> Optional[MatchResult]:
        """Parse len comparisons from source text."""
        src = source.strip()
        ops = ["<=", ">=", "!=", "==", "<", ">"]
        for op_str in ops:
            parts = src.split(op_str, 1)
            if len(parts) == 2:
                left_s, right_s = parts[0].strip(), parts[1].strip()
                left_has_len = "len(" in left_s
                right_has_len = "len(" in right_s
                if not left_has_len and not right_has_len:
                    continue
                left = ComparisonPattern._parse_expr(left_s)
                right = ComparisonPattern._parse_expr(right_s)
                if left is not None and right is not None:
                    op = ComparisonPattern._OP_MAP.get(op_str)
                    if op is None:
                        continue
                    pred = ComparisonPredicate(left=left, op=op, right=right)
                    guard = GuardNode(kind=GuardKind.LEN_COMPARISON, source_text=src, operator=op_str)
                    return MatchResult(
                        predicate=pred, confidence=0.95, source_guard=guard,
                        variables=pred.variables(), pattern_name="len_comparison",
                    )
        return None


# ---------------------------------------------------------------------------
# HasAttrPattern
# ---------------------------------------------------------------------------

class HasAttrPattern(PatternRule):
    """Matches hasattr(x, k) and 'k' in obj."""

    def name(self) -> str:
        return "hasattr"

    def can_match(self, guard: GuardNode) -> bool:
        return guard.kind == GuardKind.HASATTR

    def match(self, guard: GuardNode) -> Optional[MatchResult]:
        if not self.can_match(guard):
            return None
        var = guard.variable
        attr = guard.attribute
        if var is None or attr is None:
            return None
        positive = not guard.negated
        pred = HasAttrPredicate(variable=var, attribute=attr, positive=positive)
        return MatchResult(
            predicate=pred,
            confidence=1.0,
            source_guard=guard,
            variables=frozenset({var}),
            pattern_name=self.name(),
        )

    def priority(self) -> int:
        return 7

    @staticmethod
    def match_from_source(source: str) -> Optional[MatchResult]:
        """Parse hasattr from source text."""
        src = source.strip()
        # hasattr(x, 'k')
        pattern = r"hasattr\s*\(\s*(\w+)\s*,\s*['\"](\w+)['\"]\s*\)"
        m = re.match(pattern, src)
        if m:
            var, attr = m.group(1), m.group(2)
            guard = GuardNode(kind=GuardKind.HASATTR, source_text=src, variable=var, attribute=attr)
            pred = HasAttrPredicate(variable=var, attribute=attr, positive=True)
            return MatchResult(
                predicate=pred, confidence=1.0, source_guard=guard,
                variables=frozenset({var}), pattern_name="hasattr",
            )
        # not hasattr(x, 'k')
        neg_pattern = r"not\s+hasattr\s*\(\s*(\w+)\s*,\s*['\"](\w+)['\"]\s*\)"
        m = re.match(neg_pattern, src)
        if m:
            var, attr = m.group(1), m.group(2)
            guard = GuardNode(
                kind=GuardKind.HASATTR, source_text=src, variable=var,
                attribute=attr, negated=True,
            )
            pred = HasAttrPredicate(variable=var, attribute=attr, positive=False)
            return MatchResult(
                predicate=pred, confidence=1.0, source_guard=guard,
                variables=frozenset({var}), pattern_name="hasattr",
            )
        # 'k' in obj
        in_pattern = r"['\"](\w+)['\"]\s+in\s+(\w+)"
        m = re.match(in_pattern, src)
        if m:
            attr, var = m.group(1), m.group(2)
            guard = GuardNode(kind=GuardKind.HASATTR, source_text=src, variable=var, attribute=attr)
            pred = HasAttrPredicate(variable=var, attribute=attr, positive=True)
            return MatchResult(
                predicate=pred, confidence=0.85, source_guard=guard,
                variables=frozenset({var}), pattern_name="hasattr",
            )
        # 'k' not in obj
        not_in_pattern = r"['\"](\w+)['\"]\s+not\s+in\s+(\w+)"
        m = re.match(not_in_pattern, src)
        if m:
            attr, var = m.group(1), m.group(2)
            guard = GuardNode(
                kind=GuardKind.HASATTR, source_text=src, variable=var,
                attribute=attr, negated=True,
            )
            pred = HasAttrPredicate(variable=var, attribute=attr, positive=False)
            return MatchResult(
                predicate=pred, confidence=0.85, source_guard=guard,
                variables=frozenset({var}), pattern_name="hasattr",
            )
        return None


# ---------------------------------------------------------------------------
# TruthinessPattern
# ---------------------------------------------------------------------------

class TruthinessPattern(PatternRule):
    """Matches if x (Python truthiness) guards."""

    def name(self) -> str:
        return "truthiness"

    def can_match(self, guard: GuardNode) -> bool:
        return guard.kind == GuardKind.TRUTHINESS

    def match(self, guard: GuardNode) -> Optional[MatchResult]:
        if not self.can_match(guard):
            return None
        var = guard.variable
        if var is None:
            return None
        is_truthy = not guard.negated
        pred = TruthinessPredicate(variable=var, is_truthy=is_truthy)
        return MatchResult(
            predicate=pred,
            confidence=0.8,
            source_guard=guard,
            variables=frozenset({var}),
            pattern_name=self.name(),
        )

    def priority(self) -> int:
        return 3

    @staticmethod
    def match_from_source(source: str) -> Optional[MatchResult]:
        """Parse truthiness guard from source text."""
        src = source.strip()
        # not x
        neg_pattern = r"not\s+(\w+)$"
        m = re.match(neg_pattern, src)
        if m:
            var = m.group(1)
            if var in ("None", "True", "False", "null", "undefined"):
                return None
            guard = GuardNode(kind=GuardKind.TRUTHINESS, source_text=src, variable=var, negated=True)
            pred = TruthinessPredicate(variable=var, is_truthy=False)
            return MatchResult(
                predicate=pred, confidence=0.7, source_guard=guard,
                variables=frozenset({var}), pattern_name="truthiness",
            )
        # bare variable name as guard
        if re.match(r"^\w+$", src) and src not in ("None", "True", "False", "null", "undefined"):
            guard = GuardNode(kind=GuardKind.TRUTHINESS, source_text=src, variable=src)
            pred = TruthinessPredicate(variable=src, is_truthy=True)
            return MatchResult(
                predicate=pred, confidence=0.6, source_guard=guard,
                variables=frozenset({src}), pattern_name="truthiness",
            )
        return None


# ---------------------------------------------------------------------------
# CombinedPattern
# ---------------------------------------------------------------------------

class CombinedPattern(PatternRule):
    """Matches && / || combinations of guards."""

    def __init__(self, engine: Optional["MatchEngine"] = None) -> None:
        self._engine = engine

    def name(self) -> str:
        return "combined"

    def can_match(self, guard: GuardNode) -> bool:
        return guard.kind in (GuardKind.AND, GuardKind.OR)

    def match(self, guard: GuardNode) -> Optional[MatchResult]:
        if not self.can_match(guard):
            return None
        if len(guard.children) < 2:
            return None
        engine = self._engine or MatchEngine()
        child_results: List[MatchResult] = []
        for child in guard.children:
            matches = engine.match(child)
            if not matches:
                return None
            child_results.append(matches[0])
        child_preds = [r.predicate for r in child_results]
        min_conf = min(r.confidence for r in child_results)
        all_vars: Set[str] = set()
        for r in child_results:
            all_vars |= r.variables
        if guard.kind == GuardKind.AND:
            combined = Conjunction.create(*child_preds)
        else:
            combined = Disjunction.create(*child_preds)
        return MatchResult(
            predicate=combined,
            confidence=min_conf * 0.95,
            source_guard=guard,
            variables=frozenset(all_vars),
            pattern_name=self.name(),
        )

    def priority(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# NegatedPattern
# ---------------------------------------------------------------------------

class NegatedPattern(PatternRule):
    """Matches negated guards (not/!)."""

    def __init__(self, engine: Optional["MatchEngine"] = None) -> None:
        self._engine = engine

    def name(self) -> str:
        return "negated"

    def can_match(self, guard: GuardNode) -> bool:
        return guard.kind == GuardKind.NOT

    def match(self, guard: GuardNode) -> Optional[MatchResult]:
        if not self.can_match(guard):
            return None
        if len(guard.children) != 1:
            return None
        engine = self._engine or MatchEngine()
        inner_matches = engine.match(guard.children[0])
        if not inner_matches:
            return None
        inner = inner_matches[0]
        negated_pred = inner.predicate.negate()
        return MatchResult(
            predicate=negated_pred,
            confidence=inner.confidence * 0.98,
            source_guard=guard,
            variables=inner.variables,
            pattern_name=self.name(),
        )

    def priority(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# ChainedComparisonPattern
# ---------------------------------------------------------------------------

class ChainedComparisonPattern(PatternRule):
    """Matches chained comparisons like 0 <= i < len(arr)."""

    def name(self) -> str:
        return "chained_comparison"

    def can_match(self, guard: GuardNode) -> bool:
        return guard.kind == GuardKind.CHAINED_COMPARISON

    def match(self, guard: GuardNode) -> Optional[MatchResult]:
        if not self.can_match(guard):
            return None
        if len(guard.children) < 3:
            return None
        comparisons: List[ComparisonPredicate] = []
        ops = guard.metadata.get("operators", [])
        if len(ops) != len(guard.children) - 1:
            return None
        for i, op_str in enumerate(ops):
            op = ComparisonPattern._OP_MAP.get(op_str)
            if op is None:
                return None
            left = ComparisonPattern._guard_to_expr(guard.children[i])
            right = ComparisonPattern._guard_to_expr(guard.children[i + 1])
            if left is None or right is None:
                return None
            comparisons.append(ComparisonPredicate(left=left, op=op, right=right))
        combined = Conjunction.create(*comparisons)
        all_vars: Set[str] = set()
        for c in comparisons:
            all_vars |= c.variables()
        return MatchResult(
            predicate=combined,
            confidence=0.95,
            source_guard=guard,
            variables=frozenset(all_vars),
            pattern_name=self.name(),
        )

    def priority(self) -> int:
        return 9

    @staticmethod
    def match_from_source(source: str) -> Optional[MatchResult]:
        """Parse chained comparison from source text."""
        src = source.strip()
        # Pattern: expr1 op1 expr2 op2 expr3
        pattern = r"(.+?)\s*(<=?|>=?|==|!=)\s*(.+?)\s*(<=?|>=?|==|!=)\s*(.+)"
        m = re.match(pattern, src)
        if m:
            e1_s, op1_s, e2_s, op2_s, e3_s = (
                m.group(1).strip(), m.group(2), m.group(3).strip(),
                m.group(4), m.group(5).strip(),
            )
            e1 = ComparisonPattern._parse_expr(e1_s)
            e2 = ComparisonPattern._parse_expr(e2_s)
            e3 = ComparisonPattern._parse_expr(e3_s)
            op1 = ComparisonPattern._OP_MAP.get(op1_s)
            op2 = ComparisonPattern._OP_MAP.get(op2_s)
            if all(x is not None for x in [e1, e2, e3, op1, op2]):
                c1 = ComparisonPredicate(left=e1, op=op1, right=e2)  # type: ignore
                c2 = ComparisonPredicate(left=e2, op=op2, right=e3)  # type: ignore
                combined = Conjunction.create(c1, c2)
                guard = GuardNode(kind=GuardKind.CHAINED_COMPARISON, source_text=src)
                return MatchResult(
                    predicate=combined, confidence=0.9, source_guard=guard,
                    variables=combined.variables(), pattern_name="chained_comparison",
                )
        return None


# ---------------------------------------------------------------------------
# ArrayBoundsPattern
# ---------------------------------------------------------------------------

class ArrayBoundsPattern(PatternRule):
    """Matches array bounds checking idioms like 0 <= i < len(arr)."""

    def name(self) -> str:
        return "array_bounds"

    def can_match(self, guard: GuardNode) -> bool:
        if guard.kind == GuardKind.CHAINED_COMPARISON:
            return True
        if guard.kind == GuardKind.AND and len(guard.children) == 2:
            return True
        return False

    def match(self, guard: GuardNode) -> Optional[MatchResult]:
        if not self.can_match(guard):
            return None
        if guard.kind == GuardKind.CHAINED_COMPARISON:
            return self._match_chained(guard)
        if guard.kind == GuardKind.AND:
            return self._match_and(guard)
        return None

    def _match_chained(self, guard: GuardNode) -> Optional[MatchResult]:
        if len(guard.children) < 3:
            return None
        ops = guard.metadata.get("operators", [])
        if len(ops) < 2:
            return None
        first_op = ComparisonPattern._OP_MAP.get(ops[0])
        second_op = ComparisonPattern._OP_MAP.get(ops[1])
        if first_op not in (ComparisonOp.Le, ComparisonOp.Lt):
            return None
        if second_op not in (ComparisonOp.Le, ComparisonOp.Lt):
            return None
        left = ComparisonPattern._guard_to_expr(guard.children[0])
        mid = ComparisonPattern._guard_to_expr(guard.children[1])
        right = ComparisonPattern._guard_to_expr(guard.children[2])
        if left is None or mid is None or right is None:
            return None
        left_cv = left.constant_value()
        has_len = any(isinstance(t, LenTerm) for _, t in right.terms)
        if left_cv == 0 and has_len:
            pred = Conjunction.create(
                ComparisonPredicate(left=left, op=first_op, right=mid),
                ComparisonPredicate(left=mid, op=second_op, right=right),
            )
            return MatchResult(
                predicate=pred, confidence=1.0, source_guard=guard,
                variables=pred.variables(), pattern_name=self.name(),
                metadata={"pattern_type": "array_bounds"},
            )
        return None

    def _match_and(self, guard: GuardNode) -> Optional[MatchResult]:
        if len(guard.children) != 2:
            return None
        engine = MatchEngine()
        m1 = engine.match(guard.children[0])
        m2 = engine.match(guard.children[1])
        if not m1 or not m2:
            return None
        p1 = m1[0].predicate
        p2 = m2[0].predicate
        if not isinstance(p1, ComparisonPredicate) or not isinstance(p2, ComparisonPredicate):
            return None
        is_lower = (
            p1.op in (ComparisonOp.Le, ComparisonOp.Ge)
            and p1.left.constant_value() == 0
        )
        is_upper = any(isinstance(t, LenTerm) for _, t in p2.right.terms)
        if is_lower and is_upper:
            pred = Conjunction.create(p1, p2)
            return MatchResult(
                predicate=pred, confidence=0.95, source_guard=guard,
                variables=pred.variables(), pattern_name=self.name(),
                metadata={"pattern_type": "array_bounds"},
            )
        return None

    def priority(self) -> int:
        return 10

    @staticmethod
    def match_from_source(source: str) -> Optional[MatchResult]:
        """Match array bounds from source text."""
        src = source.strip()
        pattern = r"0\s*(<=?)\s*(\w+)\s*(<=?|<)\s*len\s*\(\s*(\w+)\s*\)"
        m = re.match(pattern, src)
        if m:
            op1_s, idx_var, op2_s, arr_var = m.group(1), m.group(2), m.group(3), m.group(4)
            op1 = ComparisonPattern._OP_MAP.get(op1_s, ComparisonOp.Le)
            op2 = ComparisonPattern._OP_MAP.get(op2_s, ComparisonOp.Lt)
            pred = Conjunction.create(
                ComparisonPredicate(
                    left=LinearExpression.from_int(0), op=op1,
                    right=LinearExpression.from_variable(idx_var),
                ),
                ComparisonPredicate(
                    left=LinearExpression.from_variable(idx_var), op=op2,
                    right=LinearExpression.from_len(arr_var),
                ),
            )
            guard = GuardNode(kind=GuardKind.CHAINED_COMPARISON, source_text=src)
            return MatchResult(
                predicate=pred, confidence=1.0, source_guard=guard,
                variables=pred.variables(), pattern_name="array_bounds",
                metadata={"pattern_type": "array_bounds"},
            )
        return None


# ---------------------------------------------------------------------------
# RangePattern
# ---------------------------------------------------------------------------

class RangePattern(PatternRule):
    """Matches range() loop variable bounds (e.g., for i in range(n))."""

    def name(self) -> str:
        return "range_bounds"

    def can_match(self, guard: GuardNode) -> bool:
        return guard.kind == GuardKind.COMPARISON and guard.metadata.get("from_range", False)

    def match(self, guard: GuardNode) -> Optional[MatchResult]:
        if not self.can_match(guard):
            return None
        var = guard.variable
        if var is None:
            return None
        start = guard.metadata.get("range_start", 0)
        stop = guard.metadata.get("range_stop")
        if stop is None:
            return None
        v = LinearExpression.from_variable(var)
        start_expr = LinearExpression.from_int(start)
        if isinstance(stop, int):
            stop_expr = LinearExpression.from_int(stop)
        elif isinstance(stop, str):
            stop_expr = LinearExpression.from_variable(stop)
        else:
            return None
        pred = Conjunction.create(
            ComparisonPredicate(left=start_expr, op=ComparisonOp.Le, right=v),
            ComparisonPredicate(left=v, op=ComparisonOp.Lt, right=stop_expr),
        )
        return MatchResult(
            predicate=pred,
            confidence=0.9,
            source_guard=guard,
            variables=pred.variables(),
            pattern_name=self.name(),
            metadata={"range_start": start, "range_stop": stop},
        )

    def priority(self) -> int:
        return 6

    @staticmethod
    def match_from_source(source: str) -> Optional[MatchResult]:
        """Match range patterns from source text like 'for i in range(n)'."""
        src = source.strip()
        # range(stop)
        pattern1 = r"for\s+(\w+)\s+in\s+range\s*\(\s*(\w+|\d+)\s*\)"
        m = re.match(pattern1, src)
        if m:
            var, stop_s = m.group(1), m.group(2)
            try:
                stop: Union[int, str] = int(stop_s)
            except ValueError:
                stop = stop_s
            v = LinearExpression.from_variable(var)
            start_expr = LinearExpression.from_int(0)
            stop_expr = LinearExpression.from_int(stop) if isinstance(stop, int) else LinearExpression.from_variable(stop)
            pred = Conjunction.create(
                ComparisonPredicate(left=start_expr, op=ComparisonOp.Le, right=v),
                ComparisonPredicate(left=v, op=ComparisonOp.Lt, right=stop_expr),
            )
            guard = GuardNode(kind=GuardKind.COMPARISON, source_text=src, variable=var,
                              metadata={"from_range": True, "range_start": 0, "range_stop": stop})
            return MatchResult(
                predicate=pred, confidence=0.9, source_guard=guard,
                variables=pred.variables(), pattern_name="range_bounds",
            )
        # range(start, stop)
        pattern2 = r"for\s+(\w+)\s+in\s+range\s*\(\s*(\w+|\d+)\s*,\s*(\w+|\d+)\s*\)"
        m = re.match(pattern2, src)
        if m:
            var, start_s, stop_s = m.group(1), m.group(2), m.group(3)
            try:
                start_v: Union[int, str] = int(start_s)
            except ValueError:
                start_v = start_s
            try:
                stop_v: Union[int, str] = int(stop_s)
            except ValueError:
                stop_v = stop_s
            v = LinearExpression.from_variable(var)
            start_expr = LinearExpression.from_int(start_v) if isinstance(start_v, int) else LinearExpression.from_variable(start_v)
            stop_expr = LinearExpression.from_int(stop_v) if isinstance(stop_v, int) else LinearExpression.from_variable(stop_v)
            pred = Conjunction.create(
                ComparisonPredicate(left=start_expr, op=ComparisonOp.Le, right=v),
                ComparisonPredicate(left=v, op=ComparisonOp.Lt, right=stop_expr),
            )
            guard = GuardNode(kind=GuardKind.COMPARISON, source_text=src, variable=var,
                              metadata={"from_range": True, "range_start": start_v, "range_stop": stop_v})
            return MatchResult(
                predicate=pred, confidence=0.9, source_guard=guard,
                variables=pred.variables(), pattern_name="range_bounds",
            )
        return None


# ---------------------------------------------------------------------------
# EnumeratePattern
# ---------------------------------------------------------------------------

class EnumeratePattern(PatternRule):
    """Matches enumerate() patterns like 'for i, x in enumerate(lst)'."""

    def name(self) -> str:
        return "enumerate"

    def can_match(self, guard: GuardNode) -> bool:
        return guard.metadata.get("from_enumerate", False)

    def match(self, guard: GuardNode) -> Optional[MatchResult]:
        if not self.can_match(guard):
            return None
        idx_var = guard.metadata.get("index_var")
        elem_var = guard.metadata.get("element_var")
        collection_var = guard.metadata.get("collection_var")
        if idx_var is None or collection_var is None:
            return None
        idx = LinearExpression.from_variable(idx_var)
        pred = Conjunction.create(
            ComparisonPredicate(
                left=LinearExpression.from_int(0), op=ComparisonOp.Le, right=idx,
            ),
            ComparisonPredicate(
                left=idx, op=ComparisonOp.Lt,
                right=LinearExpression.from_len(collection_var),
            ),
        )
        vars_set: Set[str] = {idx_var, collection_var}
        if elem_var:
            vars_set.add(elem_var)
        return MatchResult(
            predicate=pred,
            confidence=0.85,
            source_guard=guard,
            variables=frozenset(vars_set),
            pattern_name=self.name(),
            metadata={"index_var": idx_var, "element_var": elem_var, "collection_var": collection_var},
        )

    def priority(self) -> int:
        return 5

    @staticmethod
    def match_from_source(source: str) -> Optional[MatchResult]:
        """Match enumerate patterns from source text."""
        src = source.strip()
        pattern = r"for\s+(\w+)\s*,\s*(\w+)\s+in\s+enumerate\s*\(\s*(\w+)\s*\)"
        m = re.match(pattern, src)
        if m:
            idx_var, elem_var, collection_var = m.group(1), m.group(2), m.group(3)
            idx = LinearExpression.from_variable(idx_var)
            pred = Conjunction.create(
                ComparisonPredicate(
                    left=LinearExpression.from_int(0), op=ComparisonOp.Le, right=idx,
                ),
                ComparisonPredicate(
                    left=idx, op=ComparisonOp.Lt,
                    right=LinearExpression.from_len(collection_var),
                ),
            )
            guard = GuardNode(
                kind=GuardKind.COMPARISON, source_text=src,
                metadata={
                    "from_enumerate": True,
                    "index_var": idx_var,
                    "element_var": elem_var,
                    "collection_var": collection_var,
                },
            )
            return MatchResult(
                predicate=pred, confidence=0.85, source_guard=guard,
                variables=frozenset({idx_var, elem_var, collection_var}),
                pattern_name="enumerate",
            )
        return None


# ---------------------------------------------------------------------------
# PatternDatabase
# ---------------------------------------------------------------------------

class PatternDatabase:
    """Registry of all pattern rules."""

    def __init__(self) -> None:
        self._rules: List[PatternRule] = []

    def register(self, rule: PatternRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority(), reverse=True)

    def unregister(self, name: str) -> None:
        self._rules = [r for r in self._rules if r.name() != name]

    def get_rules(self) -> List[PatternRule]:
        return list(self._rules)

    def get_rule(self, name: str) -> Optional[PatternRule]:
        for r in self._rules:
            if r.name() == name:
                return r
        return None

    def clear(self) -> None:
        self._rules.clear()

    def __len__(self) -> int:
        return len(self._rules)

    def __iter__(self):
        return iter(self._rules)

    @staticmethod
    def default() -> "PatternDatabase":
        """Create a database with all built-in patterns."""
        db = PatternDatabase()
        engine = MatchEngine.__new__(MatchEngine)
        engine._database = db
        db.register(IsInstancePattern())
        db.register(TypeOfPattern())
        db.register(NullCheckPattern())
        db.register(ComparisonPattern())
        db.register(LenComparisonPattern())
        db.register(HasAttrPattern())
        db.register(TruthinessPattern())
        db.register(ChainedComparisonPattern())
        db.register(ArrayBoundsPattern())
        db.register(RangePattern())
        db.register(EnumeratePattern())
        db.register(CombinedPattern(engine))
        db.register(NegatedPattern(engine))
        return db


# ---------------------------------------------------------------------------
# MatchEngine
# ---------------------------------------------------------------------------

class MatchEngine:
    """Applies all patterns to a guard and returns best matches."""

    def __init__(self, database: Optional[PatternDatabase] = None) -> None:
        if database is not None:
            self._database = database
        else:
            self._database = PatternDatabase()
            self._database.register(IsInstancePattern())
            self._database.register(TypeOfPattern())
            self._database.register(NullCheckPattern())
            self._database.register(ComparisonPattern())
            self._database.register(LenComparisonPattern())
            self._database.register(HasAttrPattern())
            self._database.register(TruthinessPattern())
            self._database.register(ChainedComparisonPattern())
            self._database.register(ArrayBoundsPattern())
            self._database.register(RangePattern())
            self._database.register(EnumeratePattern())
            self._database.register(CombinedPattern(self))
            self._database.register(NegatedPattern(self))

    @property
    def database(self) -> PatternDatabase:
        return self._database

    def match(self, guard: GuardNode) -> List[MatchResult]:
        """Match a guard against all patterns, returning results sorted by confidence."""
        results: List[MatchResult] = []
        for rule in self._database.get_rules():
            if rule.can_match(guard):
                result = rule.match(guard)
                if result is not None:
                    results.append(result)
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def match_best(self, guard: GuardNode) -> Optional[MatchResult]:
        """Return only the highest-confidence match."""
        results = self.match(guard)
        return results[0] if results else None

    def match_all(self, guards: Iterable[GuardNode]) -> List[MatchResult]:
        """Match multiple guards, returning all results."""
        results: List[MatchResult] = []
        for guard in guards:
            results.extend(self.match(guard))
        return results

    def match_from_source(self, source: str) -> List[MatchResult]:
        """Try all source-based matchers on a source text string."""
        matchers = [
            IsInstancePattern.match_from_source,
            TypeOfPattern.match_from_source,
            NullCheckPattern.match_from_source,
            ComparisonPattern.match_from_source,
            LenComparisonPattern.match_from_source,
            HasAttrPattern.match_from_source,
            TruthinessPattern.match_from_source,
            ChainedComparisonPattern.match_from_source,
            ArrayBoundsPattern.match_from_source,
            RangePattern.match_from_source,
            EnumeratePattern.match_from_source,
        ]
        results: List[MatchResult] = []
        for matcher in matchers:
            result = matcher(source)
            if result is not None:
                results.append(result)
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def match_best_from_source(self, source: str) -> Optional[MatchResult]:
        """Return the best match from source text."""
        results = self.match_from_source(source)
        return results[0] if results else None


# ---------------------------------------------------------------------------
# GuardDecomposer
# ---------------------------------------------------------------------------

class GuardDecomposer:
    """Decomposes complex guards into atomic predicates."""

    def __init__(self, engine: Optional[MatchEngine] = None) -> None:
        self._engine = engine or MatchEngine()

    def decompose(self, guard: GuardNode) -> List[AtomicPredicate]:
        """Break a compound guard into its constituent atomic predicates."""
        result: List[AtomicPredicate] = []
        self._decompose_recursive(guard, result)
        return result

    def _decompose_recursive(self, guard: GuardNode, acc: List[AtomicPredicate]) -> None:
        if guard.kind in (GuardKind.AND, GuardKind.OR):
            for child in guard.children:
                self._decompose_recursive(child, acc)
            return
        if guard.kind == GuardKind.NOT and len(guard.children) == 1:
            inner_matches = self._engine.match(guard.children[0])
            if inner_matches:
                pred = inner_matches[0].predicate
                negated = pred.negate()
                if isinstance(negated, AtomicPredicate):
                    acc.append(negated)
                else:
                    atoms = negated.atoms()
                    acc.extend(atoms)
            return
        matches = self._engine.match(guard)
        if matches:
            pred = matches[0].predicate
            if isinstance(pred, AtomicPredicate):
                acc.append(pred)
            else:
                atoms = pred.atoms()
                acc.extend(atoms)

    def decompose_to_predicate(self, guard: GuardNode) -> PredicateTemplate:
        """Decompose a guard into a compound predicate."""
        match = self._engine.match_best(guard)
        if match:
            return match.predicate
        if guard.kind == GuardKind.AND:
            children = [self.decompose_to_predicate(c) for c in guard.children]
            return Conjunction.create(*children)
        if guard.kind == GuardKind.OR:
            children = [self.decompose_to_predicate(c) for c in guard.children]
            return Disjunction.create(*children)
        if guard.kind == GuardKind.NOT and len(guard.children) == 1:
            inner = self.decompose_to_predicate(guard.children[0])
            return inner.negate()
        return TruePredicate()

    def decompose_source(self, source: str) -> List[AtomicPredicate]:
        """Decompose from source text."""
        matches = self._engine.match_from_source(source)
        result: List[AtomicPredicate] = []
        for m in matches:
            atoms = m.predicate.atoms()
            result.extend(atoms)
        return result

    def flatten_conjunction(self, guard: GuardNode) -> List[GuardNode]:
        """Flatten a conjunction guard into its conjuncts."""
        if guard.kind == GuardKind.AND:
            result: List[GuardNode] = []
            for child in guard.children:
                result.extend(self.flatten_conjunction(child))
            return result
        return [guard]

    def flatten_disjunction(self, guard: GuardNode) -> List[GuardNode]:
        """Flatten a disjunction guard into its disjuncts."""
        if guard.kind == GuardKind.OR:
            result: List[GuardNode] = []
            for child in guard.children:
                result.extend(self.flatten_disjunction(child))
            return result
        return [guard]


# ---------------------------------------------------------------------------
# ConflictDetector
# ---------------------------------------------------------------------------

class ConflictDetector:
    """Detects conflicting predicates."""

    @staticmethod
    def find_conflicts(predicates: Sequence[PredicateTemplate]) -> List[Tuple[PredicateTemplate, PredicateTemplate]]:
        """Find pairs of conflicting predicates."""
        conflicts: List[Tuple[PredicateTemplate, PredicateTemplate]] = []
        for i, p1 in enumerate(predicates):
            for j, p2 in enumerate(predicates):
                if i < j and ConflictDetector.are_conflicting(p1, p2):
                    conflicts.append((p1, p2))
        return conflicts

    @staticmethod
    def are_conflicting(p1: PredicateTemplate, p2: PredicateTemplate) -> bool:
        """Check if two predicates are contradictory."""
        if p1.negate() == p2:
            return True
        if isinstance(p1, ComparisonPredicate) and isinstance(p2, ComparisonPredicate):
            return ConflictDetector._comparison_conflict(p1, p2)
        if isinstance(p1, TypeTagPredicate) and isinstance(p2, TypeTagPredicate):
            return ConflictDetector._type_tag_conflict(p1, p2)
        if isinstance(p1, NullityPredicate) and isinstance(p2, NullityPredicate):
            return ConflictDetector._nullity_conflict(p1, p2)
        if isinstance(p1, TruthinessPredicate) and isinstance(p2, TruthinessPredicate):
            return ConflictDetector._truthiness_conflict(p1, p2)
        return False

    @staticmethod
    def _comparison_conflict(p1: ComparisonPredicate, p2: ComparisonPredicate) -> bool:
        if p1.left != p2.left or p1.right != p2.right:
            return False
        if p1.op == p2.op.negate():
            return True
        # x < c1 and x > c2 where c1 <= c2 is not a conflict itself
        # x == c1 and x == c2 where c1 != c2
        if p1.op == ComparisonOp.Eq and p2.op == ComparisonOp.Eq:
            c1 = p1.right.constant_value()
            c2 = p2.right.constant_value()
            if c1 is not None and c2 is not None and c1 != c2:
                return True
        # x < c1 and x > c2 where c2 >= c1
        if (p1.op == ComparisonOp.Lt and p2.op == ComparisonOp.Gt):
            c1 = p1.right.constant_value()
            c2 = p2.right.constant_value()
            if c1 is not None and c2 is not None and c2 >= c1:
                return True
        if (p1.op == ComparisonOp.Gt and p2.op == ComparisonOp.Lt):
            c1 = p1.right.constant_value()
            c2 = p2.right.constant_value()
            if c1 is not None and c2 is not None and c1 >= c2:
                return True
        # x <= c1 and x >= c2 where c2 > c1
        if (p1.op == ComparisonOp.Le and p2.op == ComparisonOp.Ge):
            c1 = p1.right.constant_value()
            c2 = p2.right.constant_value()
            if c1 is not None and c2 is not None and c2 > c1:
                return True
        if (p1.op == ComparisonOp.Ge and p2.op == ComparisonOp.Le):
            c1 = p1.right.constant_value()
            c2 = p2.right.constant_value()
            if c1 is not None and c2 is not None and c1 > c2:
                return True
        return False

    @staticmethod
    def _type_tag_conflict(p1: TypeTagPredicate, p2: TypeTagPredicate) -> bool:
        if p1.variable != p2.variable:
            return False
        if p1.type_tag == p2.type_tag and p1.positive != p2.positive:
            return True
        if p1.positive and p2.positive and p1.type_tag != p2.type_tag:
            return True
        return False

    @staticmethod
    def _nullity_conflict(p1: NullityPredicate, p2: NullityPredicate) -> bool:
        if p1.variable != p2.variable:
            return False
        return p1.is_null != p2.is_null

    @staticmethod
    def _truthiness_conflict(p1: TruthinessPredicate, p2: TruthinessPredicate) -> bool:
        if p1.variable != p2.variable:
            return False
        return p1.is_truthy != p2.is_truthy

    @staticmethod
    def has_conflict(predicates: Sequence[PredicateTemplate]) -> bool:
        """Check if any pair of predicates conflicts."""
        return len(ConflictDetector.find_conflicts(predicates)) > 0

    @staticmethod
    def conflict_groups(predicates: Sequence[PredicateTemplate]) -> List[Set[int]]:
        """Group conflicting predicates by connected component."""
        n = len(predicates)
        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
        for i, p1 in enumerate(predicates):
            for j, p2 in enumerate(predicates):
                if i < j and ConflictDetector.are_conflicting(p1, p2):
                    adj[i].add(j)
                    adj[j].add(i)
        visited: Set[int] = set()
        groups: List[Set[int]] = []
        for i in range(n):
            if i not in visited and adj[i]:
                group: Set[int] = set()
                stack = [i]
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    group.add(node)
                    stack.extend(adj[node] - visited)
                groups.append(group)
        return groups

    @staticmethod
    def maximal_consistent(predicates: Sequence[PredicateTemplate]) -> List[PredicateTemplate]:
        """Find a maximal consistent subset of predicates (greedy)."""
        result: List[PredicateTemplate] = []
        for p in predicates:
            candidate = result + [p]
            if not ConflictDetector.has_conflict(candidate):
                result.append(p)
        return result


# ---------------------------------------------------------------------------
# GuardPatternMatcher (main class)
# ---------------------------------------------------------------------------

class GuardPatternMatcher:
    """Main class that matches IR GuardNodes to PredicateTemplates."""

    def __init__(self, database: Optional[PatternDatabase] = None) -> None:
        self._database = database or PatternDatabase.default()
        self._engine = MatchEngine(self._database)
        self._decomposer = GuardDecomposer(self._engine)
        self._conflict_detector = ConflictDetector()

    @property
    def engine(self) -> MatchEngine:
        return self._engine

    @property
    def decomposer(self) -> GuardDecomposer:
        return self._decomposer

    def match_guard(self, guard: GuardNode) -> List[MatchResult]:
        """Match a single guard node against all patterns."""
        return self._engine.match(guard)

    def match_guard_best(self, guard: GuardNode) -> Optional[MatchResult]:
        """Get the best match for a guard."""
        return self._engine.match_best(guard)

    def match_guards(self, guards: Iterable[GuardNode]) -> Dict[GuardNode, List[MatchResult]]:
        """Match multiple guards."""
        results: Dict[GuardNode, List[MatchResult]] = {}
        for guard in guards:
            results[guard] = self._engine.match(guard)
        return results

    def match_source(self, source: str) -> List[MatchResult]:
        """Match from source code text."""
        return self._engine.match_from_source(source)

    def decompose_guard(self, guard: GuardNode) -> PredicateTemplate:
        """Decompose a guard into a predicate."""
        return self._decomposer.decompose_to_predicate(guard)

    def extract_predicates(self, guards: Iterable[GuardNode]) -> List[PredicateTemplate]:
        """Extract all predicates from a set of guards."""
        predicates: List[PredicateTemplate] = []
        for guard in guards:
            match = self._engine.match_best(guard)
            if match:
                predicates.append(match.predicate)
        return predicates

    def extract_atoms(self, guards: Iterable[GuardNode]) -> List[AtomicPredicate]:
        """Extract all atomic predicates from guards."""
        atoms: List[AtomicPredicate] = []
        for guard in guards:
            atoms.extend(self._decomposer.decompose(guard))
        seen: Set[AtomicPredicate] = set()
        unique: List[AtomicPredicate] = []
        for a in atoms:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique

    def check_consistency(self, guards: Iterable[GuardNode]) -> Tuple[bool, List[Tuple[PredicateTemplate, PredicateTemplate]]]:
        """Check if the predicates from a set of guards are consistent."""
        predicates = self.extract_predicates(guards)
        conflicts = self._conflict_detector.find_conflicts(predicates)
        return len(conflicts) == 0, conflicts

    def simplify_guard_predicate(self, guard: GuardNode) -> PredicateTemplate:
        """Match and simplify a guard's predicate."""
        pred = self.decompose_guard(guard)
        return PredicateSimplifier.full_simplify(pred)

    def normalize_guard_predicate(self, guard: GuardNode) -> PredicateTemplate:
        """Match and normalize a guard's predicate."""
        pred = self.decompose_guard(guard)
        return PredicateNormalizer.normalize(pred)

    def summary(self, guards: Iterable[GuardNode]) -> str:
        """Generate a summary of guard matching results."""
        guard_list = list(guards)
        lines: List[str] = [f"Guard Pattern Matching Summary ({len(guard_list)} guards)"]
        lines.append("=" * 50)
        for i, guard in enumerate(guard_list):
            matches = self._engine.match(guard)
            lines.append(f"\nGuard {i + 1}: {guard.source_text or guard.kind.name}")
            if matches:
                best = matches[0]
                lines.append(f"  Best match: {best.pattern_name} (confidence: {best.confidence:.2f})")
                lines.append(f"  Predicate: {best.predicate.pretty_print()}")
                lines.append(f"  Variables: {sorted(best.variables)}")
            else:
                lines.append("  No match found")
        predicates = self.extract_predicates(guard_list)
        conflicts = self._conflict_detector.find_conflicts(predicates)
        if conflicts:
            lines.append(f"\nConflicts detected: {len(conflicts)}")
            for p1, p2 in conflicts:
                lines.append(f"  {p1.pretty_print()} ⊥ {p2.pretty_print()}")
        else:
            lines.append("\nNo conflicts detected.")
        return "\n".join(lines)
