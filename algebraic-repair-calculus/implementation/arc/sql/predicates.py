"""
Predicate Analysis
==================

Provides tools for analyzing SQL predicates:
- Extract column references from predicates
- Evaluate predicate satisfiability
- Check predicate containment (p1 ⊆ p2)
- Predicate pushdown feasibility
- Three-valued logic handling (NULL)
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
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Three-Valued Logic
# ---------------------------------------------------------------------------

class ThreeValuedBool(Enum):
    """SQL three-valued logic: TRUE, FALSE, UNKNOWN (NULL)."""
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"

    def __and__(self, other: ThreeValuedBool) -> ThreeValuedBool:
        if self == ThreeValuedBool.FALSE or other == ThreeValuedBool.FALSE:
            return ThreeValuedBool.FALSE
        if self == ThreeValuedBool.UNKNOWN or other == ThreeValuedBool.UNKNOWN:
            return ThreeValuedBool.UNKNOWN
        return ThreeValuedBool.TRUE

    def __or__(self, other: ThreeValuedBool) -> ThreeValuedBool:
        if self == ThreeValuedBool.TRUE or other == ThreeValuedBool.TRUE:
            return ThreeValuedBool.TRUE
        if self == ThreeValuedBool.UNKNOWN or other == ThreeValuedBool.UNKNOWN:
            return ThreeValuedBool.UNKNOWN
        return ThreeValuedBool.FALSE

    def __invert__(self) -> ThreeValuedBool:
        if self == ThreeValuedBool.TRUE:
            return ThreeValuedBool.FALSE
        if self == ThreeValuedBool.FALSE:
            return ThreeValuedBool.TRUE
        return ThreeValuedBool.UNKNOWN

    def to_bool(self) -> Optional[bool]:
        if self == ThreeValuedBool.TRUE:
            return True
        if self == ThreeValuedBool.FALSE:
            return False
        return None

    @staticmethod
    def from_bool(b: Optional[bool]) -> ThreeValuedBool:
        if b is True:
            return ThreeValuedBool.TRUE
        if b is False:
            return ThreeValuedBool.FALSE
        return ThreeValuedBool.UNKNOWN


# ---------------------------------------------------------------------------
# Comparison Operators
# ---------------------------------------------------------------------------

class ComparisonOp(Enum):
    """SQL comparison operators."""
    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    IN = "IN"
    NOT_IN = "NOT IN"
    BETWEEN = "BETWEEN"
    NOT_BETWEEN = "NOT BETWEEN"

    def negate(self) -> ComparisonOp:
        negations = {
            ComparisonOp.EQ: ComparisonOp.NE,
            ComparisonOp.NE: ComparisonOp.EQ,
            ComparisonOp.LT: ComparisonOp.GE,
            ComparisonOp.LE: ComparisonOp.GT,
            ComparisonOp.GT: ComparisonOp.LE,
            ComparisonOp.GE: ComparisonOp.LT,
            ComparisonOp.IS_NULL: ComparisonOp.IS_NOT_NULL,
            ComparisonOp.IS_NOT_NULL: ComparisonOp.IS_NULL,
            ComparisonOp.LIKE: ComparisonOp.NOT_LIKE,
            ComparisonOp.NOT_LIKE: ComparisonOp.LIKE,
            ComparisonOp.IN: ComparisonOp.NOT_IN,
            ComparisonOp.NOT_IN: ComparisonOp.IN,
            ComparisonOp.BETWEEN: ComparisonOp.NOT_BETWEEN,
            ComparisonOp.NOT_BETWEEN: ComparisonOp.BETWEEN,
        }
        return negations.get(self, self)

    def evaluate(self, left: Any, right: Any) -> ThreeValuedBool:
        if self == ComparisonOp.IS_NULL:
            return ThreeValuedBool.from_bool(left is None)
        if self == ComparisonOp.IS_NOT_NULL:
            return ThreeValuedBool.from_bool(left is not None)
        if left is None or right is None:
            return ThreeValuedBool.UNKNOWN
        try:
            if self == ComparisonOp.EQ:
                return ThreeValuedBool.from_bool(left == right)
            if self == ComparisonOp.NE:
                return ThreeValuedBool.from_bool(left != right)
            if self == ComparisonOp.LT:
                return ThreeValuedBool.from_bool(left < right)
            if self == ComparisonOp.LE:
                return ThreeValuedBool.from_bool(left <= right)
            if self == ComparisonOp.GT:
                return ThreeValuedBool.from_bool(left > right)
            if self == ComparisonOp.GE:
                return ThreeValuedBool.from_bool(left >= right)
            if self == ComparisonOp.IN:
                return ThreeValuedBool.from_bool(left in right)
            if self == ComparisonOp.NOT_IN:
                return ThreeValuedBool.from_bool(left not in right)
            if self == ComparisonOp.BETWEEN:
                if isinstance(right, (list, tuple)) and len(right) == 2:
                    return ThreeValuedBool.from_bool(right[0] <= left <= right[1])
            if self == ComparisonOp.NOT_BETWEEN:
                if isinstance(right, (list, tuple)) and len(right) == 2:
                    return ThreeValuedBool.from_bool(not (right[0] <= left <= right[1]))
            if self == ComparisonOp.LIKE:
                pattern = str(right).replace("%", ".*").replace("_", ".")
                return ThreeValuedBool.from_bool(bool(re.match(f"^{pattern}$", str(left))))
            if self == ComparisonOp.NOT_LIKE:
                pattern = str(right).replace("%", ".*").replace("_", ".")
                return ThreeValuedBool.from_bool(not re.match(f"^{pattern}$", str(left)))
        except (TypeError, ValueError):
            return ThreeValuedBool.UNKNOWN
        return ThreeValuedBool.UNKNOWN


# ---------------------------------------------------------------------------
# Predicate AST Nodes
# ---------------------------------------------------------------------------

class PredicateNode(ABC):
    """Base class for predicate AST nodes."""

    @abstractmethod
    def evaluate(self, row: Dict[str, Any]) -> ThreeValuedBool:
        """Evaluate this predicate against a row."""

    @abstractmethod
    def columns_referenced(self) -> Set[str]:
        """Return all column names referenced in this predicate."""

    @abstractmethod
    def negate(self) -> PredicateNode:
        """Return the negation of this predicate."""

    @abstractmethod
    def to_sql(self) -> str:
        """Convert to SQL string representation."""

    @abstractmethod
    def simplify(self) -> PredicateNode:
        """Simplify the predicate."""

    @abstractmethod
    def is_tautology(self) -> Optional[bool]:
        """Check if this predicate is always true. Returns None if unknown."""

    @abstractmethod
    def is_contradiction(self) -> Optional[bool]:
        """Check if this predicate is always false. Returns None if unknown."""

    def contains_column(self, column: str) -> bool:
        return column in self.columns_referenced()

    def is_deterministic(self) -> bool:
        return True


@dataclass(frozen=True)
class TruePredicate(PredicateNode):
    """Constant TRUE predicate."""

    def evaluate(self, row: Dict[str, Any]) -> ThreeValuedBool:
        return ThreeValuedBool.TRUE

    def columns_referenced(self) -> Set[str]:
        return set()

    def negate(self) -> PredicateNode:
        return FalsePredicate()

    def to_sql(self) -> str:
        return "TRUE"

    def simplify(self) -> PredicateNode:
        return self

    def is_tautology(self) -> Optional[bool]:
        return True

    def is_contradiction(self) -> Optional[bool]:
        return False

    def __repr__(self) -> str:
        return "TRUE"


@dataclass(frozen=True)
class FalsePredicate(PredicateNode):
    """Constant FALSE predicate."""

    def evaluate(self, row: Dict[str, Any]) -> ThreeValuedBool:
        return ThreeValuedBool.FALSE

    def columns_referenced(self) -> Set[str]:
        return set()

    def negate(self) -> PredicateNode:
        return TruePredicate()

    def to_sql(self) -> str:
        return "FALSE"

    def simplify(self) -> PredicateNode:
        return self

    def is_tautology(self) -> Optional[bool]:
        return False

    def is_contradiction(self) -> Optional[bool]:
        return True

    def __repr__(self) -> str:
        return "FALSE"


@dataclass(frozen=True)
class ColumnRef:
    """Reference to a column in a predicate."""
    name: str
    table: Optional[str] = None

    @property
    def qualified_name(self) -> str:
        if self.table:
            return f"{self.table}.{self.name}"
        return self.name


@dataclass(frozen=True)
class LiteralValue:
    """A literal value in a predicate."""
    value: Any
    type_hint: Optional[str] = None

    def to_sql(self) -> str:
        if self.value is None:
            return "NULL"
        if isinstance(self.value, str):
            escaped = self.value.replace("'", "''")
            return f"'{escaped}'"
        if isinstance(self.value, bool):
            return "TRUE" if self.value else "FALSE"
        return str(self.value)


@dataclass(frozen=True)
class ComparisonPredicate(PredicateNode):
    """A comparison predicate: column op value."""
    column: ColumnRef
    operator: ComparisonOp
    value: Union[LiteralValue, ColumnRef, List[LiteralValue]]

    def evaluate(self, row: Dict[str, Any]) -> ThreeValuedBool:
        left = row.get(self.column.name)
        if left is None and self.column.table:
            left = row.get(self.column.qualified_name)

        if isinstance(self.value, ColumnRef):
            right = row.get(self.value.name)
            if right is None and self.value.table:
                right = row.get(self.value.qualified_name)
        elif isinstance(self.value, LiteralValue):
            right = self.value.value
        elif isinstance(self.value, list):
            right = [v.value if isinstance(v, LiteralValue) else v for v in self.value]
        else:
            right = self.value

        return self.operator.evaluate(left, right)

    def columns_referenced(self) -> Set[str]:
        cols = {self.column.name}
        if isinstance(self.value, ColumnRef):
            cols.add(self.value.name)
        return cols

    def negate(self) -> PredicateNode:
        return ComparisonPredicate(
            column=self.column,
            operator=self.operator.negate(),
            value=self.value,
        )

    def to_sql(self) -> str:
        col = self.column.qualified_name
        if self.operator in (ComparisonOp.IS_NULL, ComparisonOp.IS_NOT_NULL):
            return f"{col} {self.operator.value}"
        if isinstance(self.value, ColumnRef):
            return f"{col} {self.operator.value} {self.value.qualified_name}"
        if isinstance(self.value, LiteralValue):
            return f"{col} {self.operator.value} {self.value.to_sql()}"
        if isinstance(self.value, list):
            vals = ", ".join(
                v.to_sql() if isinstance(v, LiteralValue) else str(v)
                for v in self.value
            )
            if self.operator == ComparisonOp.BETWEEN:
                return f"{col} BETWEEN {vals}"
            return f"{col} {self.operator.value} ({vals})"
        return f"{col} {self.operator.value} {self.value}"

    def simplify(self) -> PredicateNode:
        return self

    def is_tautology(self) -> Optional[bool]:
        return None

    def is_contradiction(self) -> Optional[bool]:
        if (
            isinstance(self.value, LiteralValue)
            and self.value.value is None
            and self.operator not in (ComparisonOp.IS_NULL, ComparisonOp.IS_NOT_NULL)
        ):
            return True
        return None

    def __repr__(self) -> str:
        return self.to_sql()


@dataclass(frozen=True)
class AndPredicate(PredicateNode):
    """Conjunction of predicates."""
    children: Tuple[PredicateNode, ...]

    def evaluate(self, row: Dict[str, Any]) -> ThreeValuedBool:
        result = ThreeValuedBool.TRUE
        for child in self.children:
            result = result & child.evaluate(row)
            if result == ThreeValuedBool.FALSE:
                return ThreeValuedBool.FALSE
        return result

    def columns_referenced(self) -> Set[str]:
        cols: Set[str] = set()
        for child in self.children:
            cols |= child.columns_referenced()
        return cols

    def negate(self) -> PredicateNode:
        return OrPredicate(tuple(c.negate() for c in self.children))

    def to_sql(self) -> str:
        parts = [f"({c.to_sql()})" for c in self.children]
        return " AND ".join(parts)

    def simplify(self) -> PredicateNode:
        simplified = []
        for child in self.children:
            s = child.simplify()
            if isinstance(s, FalsePredicate):
                return FalsePredicate()
            if isinstance(s, TruePredicate):
                continue
            if isinstance(s, AndPredicate):
                simplified.extend(s.children)
            else:
                simplified.append(s)

        if not simplified:
            return TruePredicate()
        if len(simplified) == 1:
            return simplified[0]
        return AndPredicate(tuple(simplified))

    def is_tautology(self) -> Optional[bool]:
        for child in self.children:
            t = child.is_tautology()
            if t is False:
                return False
            if t is None:
                return None
        return True

    def is_contradiction(self) -> Optional[bool]:
        for child in self.children:
            c = child.is_contradiction()
            if c is True:
                return True
        return None

    def __repr__(self) -> str:
        return self.to_sql()


@dataclass(frozen=True)
class OrPredicate(PredicateNode):
    """Disjunction of predicates."""
    children: Tuple[PredicateNode, ...]

    def evaluate(self, row: Dict[str, Any]) -> ThreeValuedBool:
        result = ThreeValuedBool.FALSE
        for child in self.children:
            result = result | child.evaluate(row)
            if result == ThreeValuedBool.TRUE:
                return ThreeValuedBool.TRUE
        return result

    def columns_referenced(self) -> Set[str]:
        cols: Set[str] = set()
        for child in self.children:
            cols |= child.columns_referenced()
        return cols

    def negate(self) -> PredicateNode:
        return AndPredicate(tuple(c.negate() for c in self.children))

    def to_sql(self) -> str:
        parts = [f"({c.to_sql()})" for c in self.children]
        return " OR ".join(parts)

    def simplify(self) -> PredicateNode:
        simplified = []
        for child in self.children:
            s = child.simplify()
            if isinstance(s, TruePredicate):
                return TruePredicate()
            if isinstance(s, FalsePredicate):
                continue
            if isinstance(s, OrPredicate):
                simplified.extend(s.children)
            else:
                simplified.append(s)

        if not simplified:
            return FalsePredicate()
        if len(simplified) == 1:
            return simplified[0]
        return OrPredicate(tuple(simplified))

    def is_tautology(self) -> Optional[bool]:
        for child in self.children:
            t = child.is_tautology()
            if t is True:
                return True
        return None

    def is_contradiction(self) -> Optional[bool]:
        for child in self.children:
            c = child.is_contradiction()
            if c is False:
                return False
            if c is None:
                return None
        return True

    def __repr__(self) -> str:
        return self.to_sql()


@dataclass(frozen=True)
class NotPredicate(PredicateNode):
    """Negation of a predicate."""
    child: PredicateNode

    def evaluate(self, row: Dict[str, Any]) -> ThreeValuedBool:
        return ~self.child.evaluate(row)

    def columns_referenced(self) -> Set[str]:
        return self.child.columns_referenced()

    def negate(self) -> PredicateNode:
        return self.child

    def to_sql(self) -> str:
        return f"NOT ({self.child.to_sql()})"

    def simplify(self) -> PredicateNode:
        s = self.child.simplify()
        if isinstance(s, TruePredicate):
            return FalsePredicate()
        if isinstance(s, FalsePredicate):
            return TruePredicate()
        if isinstance(s, NotPredicate):
            return s.child.simplify()
        return NotPredicate(s)

    def is_tautology(self) -> Optional[bool]:
        c = self.child.is_contradiction()
        return c

    def is_contradiction(self) -> Optional[bool]:
        return self.child.is_tautology()

    def __repr__(self) -> str:
        return self.to_sql()


@dataclass(frozen=True)
class ExistsPredicate(PredicateNode):
    """EXISTS subquery predicate."""
    subquery_sql: str
    correlated_columns: FrozenSet[str] = frozenset()

    def evaluate(self, row: Dict[str, Any]) -> ThreeValuedBool:
        return ThreeValuedBool.UNKNOWN

    def columns_referenced(self) -> Set[str]:
        return set(self.correlated_columns)

    def negate(self) -> PredicateNode:
        return NotPredicate(self)

    def to_sql(self) -> str:
        return f"EXISTS ({self.subquery_sql})"

    def simplify(self) -> PredicateNode:
        return self

    def is_tautology(self) -> Optional[bool]:
        return None

    def is_contradiction(self) -> Optional[bool]:
        return None

    def is_deterministic(self) -> bool:
        return True

    def __repr__(self) -> str:
        return self.to_sql()


@dataclass(frozen=True)
class InSubqueryPredicate(PredicateNode):
    """IN (subquery) predicate."""
    column: ColumnRef
    subquery_sql: str
    negated: bool = False

    def evaluate(self, row: Dict[str, Any]) -> ThreeValuedBool:
        return ThreeValuedBool.UNKNOWN

    def columns_referenced(self) -> Set[str]:
        return {self.column.name}

    def negate(self) -> PredicateNode:
        return InSubqueryPredicate(
            column=self.column,
            subquery_sql=self.subquery_sql,
            negated=not self.negated,
        )

    def to_sql(self) -> str:
        neg = "NOT " if self.negated else ""
        return f"{self.column.qualified_name} {neg}IN ({self.subquery_sql})"

    def simplify(self) -> PredicateNode:
        return self

    def is_tautology(self) -> Optional[bool]:
        return None

    def is_contradiction(self) -> Optional[bool]:
        return None

    def __repr__(self) -> str:
        return self.to_sql()


# ---------------------------------------------------------------------------
# Predicate Analyzer
# ---------------------------------------------------------------------------

class PredicateAnalyzer:
    """
    Analyzes SQL predicates for column references, satisfiability,
    containment, and pushdown feasibility.
    """

    @staticmethod
    def extract_columns(predicate: PredicateNode) -> Set[str]:
        """Extract all column names referenced in a predicate."""
        return predicate.columns_referenced()

    @staticmethod
    def extract_columns_from_sql(sql_predicate: str) -> Set[str]:
        """Extract column names from a raw SQL predicate string."""
        columns: Set[str] = set()
        tokens = re.split(r"[\s,()]+", sql_predicate)
        sql_keywords = {
            "AND", "OR", "NOT", "IN", "BETWEEN", "LIKE", "IS", "NULL",
            "TRUE", "FALSE", "EXISTS", "SELECT", "FROM", "WHERE", "HAVING",
            "GROUP", "BY", "ORDER", "LIMIT", "OFFSET", "UNION", "ALL",
            "CASE", "WHEN", "THEN", "ELSE", "END", "AS", "JOIN", "ON",
            "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS", "NATURAL",
            "ASC", "DESC", "DISTINCT", "SET", "VALUES", "INSERT", "UPDATE",
            "DELETE", "CREATE", "DROP", "ALTER", "TABLE", "INDEX", "VIEW",
        }
        for token in tokens:
            token = token.strip("'\"")
            if not token:
                continue
            if token.upper() in sql_keywords:
                continue
            try:
                float(token)
                continue
            except ValueError:
                pass
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", token):
                parts = token.split(".")
                columns.add(parts[-1])
        return columns

    @staticmethod
    def is_satisfiable(predicate: PredicateNode) -> Optional[bool]:
        """
        Check if a predicate is satisfiable (can be TRUE for some input).
        Returns True if satisfiable, False if contradiction, None if unknown.
        """
        simplified = predicate.simplify()
        if simplified.is_contradiction() is True:
            return False
        if simplified.is_tautology() is True:
            return True
        return None

    @staticmethod
    def is_tautology(predicate: PredicateNode) -> Optional[bool]:
        """Check if a predicate is always true."""
        return predicate.simplify().is_tautology()

    @staticmethod
    def check_containment(p1: PredicateNode, p2: PredicateNode) -> Optional[bool]:
        """
        Check if p1 ⊆ p2 (every row satisfying p1 also satisfies p2).

        Returns True if p1 entails p2, False if not, None if unknown.
        Uses syntactic analysis for common cases.
        """
        s1 = p1.simplify()
        s2 = p2.simplify()

        if s1.is_contradiction() is True:
            return True
        if s2.is_tautology() is True:
            return True
        if s1.is_tautology() is True and s2.is_tautology() is not True:
            return False
        if s1 == s2:
            return True

        if isinstance(s1, ComparisonPredicate) and isinstance(s2, ComparisonPredicate):
            return PredicateAnalyzer._check_comparison_containment(s1, s2)

        if isinstance(s1, AndPredicate):
            for child in s1.children:
                if PredicateAnalyzer.check_containment(child, s2) is True:
                    return True

        if isinstance(s2, OrPredicate):
            for child in s2.children:
                if PredicateAnalyzer.check_containment(s1, child) is True:
                    return True

        return None

    @staticmethod
    def _check_comparison_containment(
        p1: ComparisonPredicate, p2: ComparisonPredicate
    ) -> Optional[bool]:
        """Check containment between two comparison predicates."""
        if p1.column.name != p2.column.name:
            return None

        v1 = p1.value.value if isinstance(p1.value, LiteralValue) else None
        v2 = p2.value.value if isinstance(p2.value, LiteralValue) else None

        if v1 is None or v2 is None:
            if isinstance(p1.value, ColumnRef) and isinstance(p2.value, ColumnRef):
                if p1.value.name == p2.value.name and p1.operator == p2.operator:
                    return True
            return None

        if p1.operator == ComparisonOp.EQ:
            if p2.operator == ComparisonOp.EQ:
                return v1 == v2
            if p2.operator == ComparisonOp.NE:
                return v1 != v2
            if p2.operator == ComparisonOp.LT:
                try:
                    return v1 < v2
                except TypeError:
                    return None
            if p2.operator == ComparisonOp.LE:
                try:
                    return v1 <= v2
                except TypeError:
                    return None
            if p2.operator == ComparisonOp.GT:
                try:
                    return v1 > v2
                except TypeError:
                    return None
            if p2.operator == ComparisonOp.GE:
                try:
                    return v1 >= v2
                except TypeError:
                    return None
            if p2.operator == ComparisonOp.IS_NOT_NULL:
                return True

        if p1.operator == ComparisonOp.LT and p2.operator == ComparisonOp.LE:
            try:
                return v1 <= v2
            except TypeError:
                return None

        if p1.operator == ComparisonOp.LE and p2.operator == ComparisonOp.LE:
            try:
                return v1 <= v2
            except TypeError:
                return None

        if p1.operator == ComparisonOp.GT and p2.operator == ComparisonOp.GE:
            try:
                return v1 >= v2
            except TypeError:
                return None

        if p1.operator == ComparisonOp.IS_NOT_NULL and p2.operator == ComparisonOp.IS_NOT_NULL:
            return True

        return None

    @staticmethod
    def pushdown_feasible(
        predicate: PredicateNode,
        available_columns: Set[str],
    ) -> bool:
        """
        Check if a predicate can be pushed down to a level where
        only available_columns are accessible.
        """
        required = predicate.columns_referenced()
        return required.issubset(available_columns)

    @staticmethod
    def split_conjuncts(predicate: PredicateNode) -> List[PredicateNode]:
        """Split a predicate into its conjunctive components."""
        simplified = predicate.simplify()
        if isinstance(simplified, AndPredicate):
            result = []
            for child in simplified.children:
                result.extend(PredicateAnalyzer.split_conjuncts(child))
            return result
        return [simplified]

    @staticmethod
    def split_disjuncts(predicate: PredicateNode) -> List[PredicateNode]:
        """Split a predicate into its disjunctive components."""
        simplified = predicate.simplify()
        if isinstance(simplified, OrPredicate):
            result = []
            for child in simplified.children:
                result.extend(PredicateAnalyzer.split_disjuncts(child))
            return result
        return [simplified]

    @staticmethod
    def partition_pushable(
        predicate: PredicateNode,
        available_columns: Set[str],
    ) -> Tuple[PredicateNode, PredicateNode]:
        """
        Partition a conjunction into pushable and non-pushable parts.

        Returns (pushable, remainder) where pushable only references
        available_columns and remainder is the rest.
        """
        conjuncts = PredicateAnalyzer.split_conjuncts(predicate)
        pushable: List[PredicateNode] = []
        remainder: List[PredicateNode] = []

        for c in conjuncts:
            if PredicateAnalyzer.pushdown_feasible(c, available_columns):
                pushable.append(c)
            else:
                remainder.append(c)

        push_pred: PredicateNode = TruePredicate()
        if pushable:
            if len(pushable) == 1:
                push_pred = pushable[0]
            else:
                push_pred = AndPredicate(tuple(pushable))

        rem_pred: PredicateNode = TruePredicate()
        if remainder:
            if len(remainder) == 1:
                rem_pred = remainder[0]
            else:
                rem_pred = AndPredicate(tuple(remainder))

        return push_pred, rem_pred

    @staticmethod
    def rename_columns(
        predicate: PredicateNode,
        rename_map: Dict[str, str],
    ) -> PredicateNode:
        """Rename column references in a predicate."""
        if isinstance(predicate, TruePredicate) or isinstance(predicate, FalsePredicate):
            return predicate

        if isinstance(predicate, ComparisonPredicate):
            new_col = ColumnRef(
                name=rename_map.get(predicate.column.name, predicate.column.name),
                table=predicate.column.table,
            )
            new_val = predicate.value
            if isinstance(predicate.value, ColumnRef):
                new_val = ColumnRef(
                    name=rename_map.get(predicate.value.name, predicate.value.name),
                    table=predicate.value.table,
                )
            return ComparisonPredicate(
                column=new_col,
                operator=predicate.operator,
                value=new_val,
            )

        if isinstance(predicate, AndPredicate):
            return AndPredicate(
                tuple(
                    PredicateAnalyzer.rename_columns(c, rename_map)
                    for c in predicate.children
                )
            )

        if isinstance(predicate, OrPredicate):
            return OrPredicate(
                tuple(
                    PredicateAnalyzer.rename_columns(c, rename_map)
                    for c in predicate.children
                )
            )

        if isinstance(predicate, NotPredicate):
            return NotPredicate(
                PredicateAnalyzer.rename_columns(predicate.child, rename_map)
            )

        return predicate

    @staticmethod
    def evaluate_predicate(
        predicate: PredicateNode, row: Dict[str, Any]
    ) -> ThreeValuedBool:
        """Evaluate a predicate against a row dictionary."""
        return predicate.evaluate(row)

    @staticmethod
    def to_callable(
        predicate: PredicateNode,
    ) -> Callable[[Dict[str, Any]], bool]:
        """Convert a predicate to a Python callable (two-valued: UNKNOWN → False)."""
        def pred_fn(row: Dict[str, Any]) -> bool:
            result = predicate.evaluate(row)
            return result == ThreeValuedBool.TRUE
        return pred_fn

    @staticmethod
    def from_sql_fragment(sql: str) -> PredicateNode:
        """
        Parse a simple SQL predicate fragment into a PredicateNode.
        Handles basic comparisons, AND, OR, NOT, IS NULL, IS NOT NULL.
        """
        sql = sql.strip()

        if not sql or sql.upper() == "TRUE":
            return TruePredicate()
        if sql.upper() == "FALSE":
            return FalsePredicate()

        if sql.upper().startswith("NOT "):
            inner = sql[4:].strip()
            if inner.startswith("(") and inner.endswith(")"):
                inner = inner[1:-1]
            return NotPredicate(PredicateAnalyzer.from_sql_fragment(inner))

        and_parts = _split_logical(sql, "AND")
        if len(and_parts) > 1:
            children = tuple(
                PredicateAnalyzer.from_sql_fragment(p) for p in and_parts
            )
            return AndPredicate(children)

        or_parts = _split_logical(sql, "OR")
        if len(or_parts) > 1:
            children = tuple(
                PredicateAnalyzer.from_sql_fragment(p) for p in or_parts
            )
            return OrPredicate(children)

        if sql.startswith("(") and sql.endswith(")"):
            return PredicateAnalyzer.from_sql_fragment(sql[1:-1])

        upper = sql.upper()
        if " IS NOT NULL" in upper:
            col = sql[:upper.index(" IS NOT NULL")].strip()
            return ComparisonPredicate(
                column=ColumnRef(name=col),
                operator=ComparisonOp.IS_NOT_NULL,
                value=LiteralValue(None),
            )
        if " IS NULL" in upper:
            col = sql[:upper.index(" IS NULL")].strip()
            return ComparisonPredicate(
                column=ColumnRef(name=col),
                operator=ComparisonOp.IS_NULL,
                value=LiteralValue(None),
            )

        for op_str, op_enum in [
            ("!=", ComparisonOp.NE),
            ("<>", ComparisonOp.NE),
            (">=", ComparisonOp.GE),
            ("<=", ComparisonOp.LE),
            (">", ComparisonOp.GT),
            ("<", ComparisonOp.LT),
            ("=", ComparisonOp.EQ),
        ]:
            if op_str in sql:
                parts = sql.split(op_str, 1)
                if len(parts) == 2:
                    col_name = parts[0].strip()
                    val_str = parts[1].strip()
                    return ComparisonPredicate(
                        column=ColumnRef(name=col_name),
                        operator=op_enum,
                        value=LiteralValue(_parse_literal(val_str)),
                    )

        return TruePredicate()


def _split_logical(sql: str, keyword: str) -> List[str]:
    """Split SQL by a logical keyword, respecting parentheses."""
    parts: List[str] = []
    depth = 0
    current = []
    tokens = sql.split()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        for ch in token:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
        if token.upper() == keyword and depth == 0:
            parts.append(" ".join(current))
            current = []
        else:
            current.append(token)
        i += 1
    if current:
        parts.append(" ".join(current))
    return [p.strip() for p in parts if p.strip()]


def _parse_literal(s: str) -> Any:
    """Parse a SQL literal value."""
    s = s.strip()
    if s.upper() == "NULL":
        return None
    if s.upper() == "TRUE":
        return True
    if s.upper() == "FALSE":
        return False
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s
