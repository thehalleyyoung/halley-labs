"""
Fragment F Testing (Algorithm A6)
==================================

Checks whether a data pipeline (represented as a query graph) falls within
Fragment F: the deterministic, order-independent SQL subset for which
delta propagation is provably correct.

Fragment F requires:
- No ORDER BY with LIMIT (nondeterministic tie-breaking)
- No TABLESAMPLE
- No nondeterministic functions (random(), now(), etc.)
- No floating-point arithmetic in GROUP BY keys
- All operators in {SELECT, JOIN, GROUP_BY, FILTER, UNION, WINDOW, CTE, SET_OP}
- Exact arithmetic (no float aggregations that lose precision)
- No correlated subqueries with nondeterministic evaluation order
"""

from __future__ import annotations

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

try:
    import sqlglot
    import sqlglot.expressions as exp
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

from arc.sql.parser import ParsedQuery, SQLParser
from arc.sql.operators import SQLOperatorType


# ---------------------------------------------------------------------------
# Fragment Violation Types
# ---------------------------------------------------------------------------

class ViolationCategory(Enum):
    """Categories of Fragment F violations."""
    NONDETERMINISTIC_ORDER = "NONDETERMINISTIC_ORDER"
    NONDETERMINISTIC_FUNCTION = "NONDETERMINISTIC_FUNCTION"
    TABLESAMPLE = "TABLESAMPLE"
    FLOAT_GROUP_BY = "FLOAT_GROUP_BY"
    UNSUPPORTED_OPERATOR = "UNSUPPORTED_OPERATOR"
    FLOAT_AGGREGATION = "FLOAT_AGGREGATION"
    CORRELATED_SUBQUERY = "CORRELATED_SUBQUERY"
    NONDETERMINISTIC_WINDOW = "NONDETERMINISTIC_WINDOW"
    RECURSIVE_CTE_NONDETERMINISTIC = "RECURSIVE_CTE_NONDETERMINISTIC"
    LATERAL_JOIN = "LATERAL_JOIN"
    VOLATILE_FUNCTION = "VOLATILE_FUNCTION"

    @property
    def severity(self) -> str:
        critical = {
            ViolationCategory.NONDETERMINISTIC_ORDER,
            ViolationCategory.NONDETERMINISTIC_FUNCTION,
            ViolationCategory.TABLESAMPLE,
        }
        warning = {
            ViolationCategory.FLOAT_GROUP_BY,
            ViolationCategory.FLOAT_AGGREGATION,
        }
        if self in critical:
            return "critical"
        if self in warning:
            return "warning"
        return "info"


@dataclass(frozen=True)
class FragmentViolation:
    """A single violation of Fragment F requirements."""
    category: ViolationCategory
    message: str
    node_id: Optional[str] = None
    sql_fragment: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None

    @property
    def severity(self) -> str:
        return self.category.severity

    def __repr__(self) -> str:
        loc = f" at node {self.node_id}" if self.node_id else ""
        return f"[{self.severity.upper()}] {self.category.value}{loc}: {self.message}"


# ---------------------------------------------------------------------------
# Fragment Check Result
# ---------------------------------------------------------------------------

@dataclass
class FragmentResult:
    """Result of a Fragment F check on a pipeline."""
    is_in_fragment: bool = True
    violations: List[FragmentViolation] = field(default_factory=list)
    node_results: Dict[str, NodeFragmentResult] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    checked_nodes: int = 0

    def add_violation(self, violation: FragmentViolation) -> None:
        self.violations.append(violation)
        if violation.severity == "critical":
            self.is_in_fragment = False

    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "critical")

    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")

    def info_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "info")

    def violations_by_category(self) -> Dict[ViolationCategory, List[FragmentViolation]]:
        result: Dict[ViolationCategory, List[FragmentViolation]] = {}
        for v in self.violations:
            result.setdefault(v.category, []).append(v)
        return result

    def violations_for_node(self, node_id: str) -> List[FragmentViolation]:
        return [v for v in self.violations if v.node_id == node_id]

    def summary(self) -> Dict[str, Any]:
        return {
            "in_fragment": self.is_in_fragment,
            "total_violations": len(self.violations),
            "critical": self.critical_count(),
            "warnings": self.warning_count(),
            "info": self.info_count(),
            "checked_nodes": self.checked_nodes,
        }

    def __repr__(self) -> str:
        status = "✓ IN Fragment F" if self.is_in_fragment else "✗ NOT in Fragment F"
        return (
            f"FragmentResult({status}, "
            f"{self.critical_count()} critical, "
            f"{self.warning_count()} warnings)"
        )


@dataclass
class NodeFragmentResult:
    """Fragment F check result for a single pipeline node."""
    node_id: str
    is_in_fragment: bool = True
    violations: List[FragmentViolation] = field(default_factory=list)
    query_sql: Optional[str] = None

    def __repr__(self) -> str:
        status = "✓" if self.is_in_fragment else "✗"
        return f"Node({self.node_id}: {status}, {len(self.violations)} violations)"


# ---------------------------------------------------------------------------
# Nondeterministic Function Registry
# ---------------------------------------------------------------------------

NONDETERMINISTIC_FUNCTIONS: Set[str] = {
    "random", "rand", "setseed",
    "now", "current_timestamp", "current_date", "current_time",
    "clock_timestamp", "statement_timestamp", "transaction_timestamp",
    "timeofday",
    "gen_random_uuid", "uuid_generate_v1", "uuid_generate_v4",
    "nextval", "currval", "lastval",
    "pg_backend_pid", "pg_postmaster_start_time",
    "inet_client_addr", "inet_client_port",
    "inet_server_addr", "inet_server_port",
    "version", "current_user", "session_user",
    "txid_current", "txid_current_snapshot",
}

VOLATILE_FUNCTIONS: Set[str] = NONDETERMINISTIC_FUNCTIONS | {
    "pg_sleep", "lo_import", "lo_export",
    "dblink", "dblink_exec",
    "notify", "listen",
}

FLOAT_TYPES: Set[str] = {
    "float", "float4", "float8", "double", "double precision",
    "real", "numeric", "decimal",
}

ALLOWED_OPERATORS: Set[SQLOperatorType] = {
    SQLOperatorType.SELECT,
    SQLOperatorType.JOIN,
    SQLOperatorType.GROUP_BY,
    SQLOperatorType.FILTER,
    SQLOperatorType.UNION,
    SQLOperatorType.WINDOW,
    SQLOperatorType.CTE,
    SQLOperatorType.SET_OP,
    SQLOperatorType.DISTINCT,
}


# ---------------------------------------------------------------------------
# Fragment Checker
# ---------------------------------------------------------------------------

class FragmentChecker:
    """
    Checks whether queries/pipelines fall within Fragment F.

    Algorithm A6: For each node in the pipeline graph, verify:
    1. No ORDER BY with LIMIT (nondeterministic tie-breaking)
    2. No TABLESAMPLE
    3. No nondeterministic functions
    4. No floating-point arithmetic in GROUP BY keys
    5. All operators in allowed set
    6. Exact arithmetic requirements
    7. Deterministic window functions
    """

    def __init__(self, parser: Optional[SQLParser] = None) -> None:
        self._parser = parser or SQLParser()

    def check_query(
        self,
        sql: str,
        node_id: str = "query",
    ) -> FragmentResult:
        """Check a single query against Fragment F."""
        result = FragmentResult()
        parsed = self._parser.parse(sql)
        node_result = self._check_parsed_query(parsed, node_id)
        result.node_results[node_id] = node_result
        result.checked_nodes = 1

        for v in node_result.violations:
            result.add_violation(v)

        return result

    def check_pipeline(
        self,
        nodes: List[Tuple[str, str]],
    ) -> FragmentResult:
        """
        Check a pipeline (list of (node_id, sql) pairs) against Fragment F.
        """
        result = FragmentResult()
        result.checked_nodes = len(nodes)

        for node_id, sql in nodes:
            parsed = self._parser.parse(sql)
            node_result = self._check_parsed_query(parsed, node_id)
            result.node_results[node_id] = node_result

            for v in node_result.violations:
                result.add_violation(v)

        return result

    def check_parsed(
        self,
        parsed: ParsedQuery,
        node_id: str = "query",
    ) -> FragmentResult:
        """Check a pre-parsed query against Fragment F."""
        result = FragmentResult()
        node_result = self._check_parsed_query(parsed, node_id)
        result.node_results[node_id] = node_result
        result.checked_nodes = 1

        for v in node_result.violations:
            result.add_violation(v)

        return result

    def _check_parsed_query(
        self,
        parsed: ParsedQuery,
        node_id: str,
    ) -> NodeFragmentResult:
        """Check all Fragment F requirements for a parsed query."""
        node_result = NodeFragmentResult(
            node_id=node_id,
            query_sql=parsed.raw_sql,
        )

        self._check_order_by_limit(parsed, node_id, node_result)
        self._check_tablesample(parsed, node_id, node_result)
        self._check_nondeterministic_functions(parsed, node_id, node_result)
        self._check_float_group_by(parsed, node_id, node_result)
        self._check_operator_type(parsed, node_id, node_result)
        self._check_float_aggregation(parsed, node_id, node_result)
        self._check_window_determinism(parsed, node_id, node_result)
        self._check_recursive_cte(parsed, node_id, node_result)
        self._check_subqueries(parsed, node_id, node_result)

        node_result.is_in_fragment = not any(
            v.severity == "critical" for v in node_result.violations
        )
        return node_result

    def _check_order_by_limit(
        self,
        parsed: ParsedQuery,
        node_id: str,
        result: NodeFragmentResult,
    ) -> None:
        """Check for ORDER BY with LIMIT (nondeterministic when ties exist)."""
        if parsed.has_ordering() and parsed.has_limit():
            result.violations.append(FragmentViolation(
                category=ViolationCategory.NONDETERMINISTIC_ORDER,
                message=(
                    "ORDER BY with LIMIT is nondeterministic when ties exist. "
                    "Add a tiebreaker column or remove LIMIT."
                ),
                node_id=node_id,
                suggestion="Add a unique column to ORDER BY to break ties deterministically",
            ))

        if parsed.has_ordering() and not parsed.has_limit():
            has_tiebreaker = len(parsed.order_by) > 1
            if not has_tiebreaker:
                result.violations.append(FragmentViolation(
                    category=ViolationCategory.NONDETERMINISTIC_ORDER,
                    message=(
                        "ORDER BY without unique tiebreaker may not be "
                        "deterministic across executions"
                    ),
                    node_id=node_id,
                    suggestion="ORDER BY alone doesn't affect Fragment F if no LIMIT",
                ))

        if HAS_SQLGLOT and parsed.ast:
            for node in parsed.ast.find_all(exp.Order):
                parent = node.parent
                if parent and hasattr(parent, "find"):
                    limit = parent.find(exp.Limit)
                    if limit:
                        result.violations.append(FragmentViolation(
                            category=ViolationCategory.NONDETERMINISTIC_ORDER,
                            message="Subquery has ORDER BY with LIMIT",
                            node_id=node_id,
                            sql_fragment=node.sql(),
                        ))

    def _check_tablesample(
        self,
        parsed: ParsedQuery,
        node_id: str,
        result: NodeFragmentResult,
    ) -> None:
        """Check for TABLESAMPLE (nondeterministic)."""
        sql_upper = parsed.raw_sql.upper()
        if "TABLESAMPLE" in sql_upper:
            result.violations.append(FragmentViolation(
                category=ViolationCategory.TABLESAMPLE,
                message="TABLESAMPLE is nondeterministic and not in Fragment F",
                node_id=node_id,
                suggestion="Remove TABLESAMPLE; use a deterministic WHERE filter instead",
            ))

        if HAS_SQLGLOT and parsed.ast:
            for ts in parsed.ast.find_all(exp.TableSample):
                result.violations.append(FragmentViolation(
                    category=ViolationCategory.TABLESAMPLE,
                    message=f"TABLESAMPLE found: {ts.sql()}",
                    node_id=node_id,
                    sql_fragment=ts.sql(),
                ))

    def _check_nondeterministic_functions(
        self,
        parsed: ParsedQuery,
        node_id: str,
        result: NodeFragmentResult,
    ) -> None:
        """Check for nondeterministic function calls."""
        sql_lower = parsed.raw_sql.lower()

        for func_name in NONDETERMINISTIC_FUNCTIONS:
            pattern = rf"\b{re.escape(func_name)}\s*\("
            if re.search(pattern, sql_lower):
                result.violations.append(FragmentViolation(
                    category=ViolationCategory.NONDETERMINISTIC_FUNCTION,
                    message=f"Nondeterministic function '{func_name}()' found",
                    node_id=node_id,
                    sql_fragment=func_name,
                    suggestion=f"Replace {func_name}() with a deterministic alternative",
                ))

        if HAS_SQLGLOT and parsed.ast:
            for func_node in parsed.ast.find_all(exp.Anonymous):
                func_name_lower = func_node.name.lower() if hasattr(func_node, "name") else ""
                if func_name_lower in NONDETERMINISTIC_FUNCTIONS:
                    result.violations.append(FragmentViolation(
                        category=ViolationCategory.NONDETERMINISTIC_FUNCTION,
                        message=f"Nondeterministic function '{func_name_lower}()' in AST",
                        node_id=node_id,
                        sql_fragment=func_node.sql(),
                    ))

    def _check_float_group_by(
        self,
        parsed: ParsedQuery,
        node_id: str,
        result: NodeFragmentResult,
    ) -> None:
        """Check for floating-point columns in GROUP BY keys."""
        if not parsed.group_by_columns:
            return

        sql_lower = parsed.raw_sql.lower()
        for gb_col in parsed.group_by_columns:
            col_name = gb_col.name.lower()

            for float_type in FLOAT_TYPES:
                cast_pattern = rf"cast\s*\(\s*{re.escape(col_name)}\s+as\s+{re.escape(float_type)}"
                if re.search(cast_pattern, sql_lower):
                    result.violations.append(FragmentViolation(
                        category=ViolationCategory.FLOAT_GROUP_BY,
                        message=(
                            f"GROUP BY column '{gb_col.name}' involves "
                            f"floating-point cast to {float_type}"
                        ),
                        node_id=node_id,
                        suggestion="Use integer or string types for GROUP BY keys",
                    ))

        for gb_col in parsed.group_by_columns:
            col_name = gb_col.name.lower()
            for op in [" / ", " * 1.0", " * 0."]:
                if f"{col_name}{op}" in sql_lower or f"({col_name}){op}" in sql_lower:
                    result.violations.append(FragmentViolation(
                        category=ViolationCategory.FLOAT_GROUP_BY,
                        message=(
                            f"GROUP BY key '{gb_col.name}' may involve "
                            f"floating-point arithmetic"
                        ),
                        node_id=node_id,
                        suggestion="Avoid float arithmetic in GROUP BY keys",
                    ))

    def _check_operator_type(
        self,
        parsed: ParsedQuery,
        node_id: str,
        result: NodeFragmentResult,
    ) -> None:
        """Check that all operators are in the allowed set."""
        if parsed.operator_type not in ALLOWED_OPERATORS:
            result.violations.append(FragmentViolation(
                category=ViolationCategory.UNSUPPORTED_OPERATOR,
                message=(
                    f"Operator type '{parsed.operator_type.value}' "
                    f"is not in Fragment F"
                ),
                node_id=node_id,
                suggestion=(
                    f"Rewrite to use only: "
                    f"{', '.join(o.value for o in ALLOWED_OPERATORS)}"
                ),
            ))

    def _check_float_aggregation(
        self,
        parsed: ParsedQuery,
        node_id: str,
        result: NodeFragmentResult,
    ) -> None:
        """Check for float aggregations that may lose precision."""
        float_sensitive_aggs = {"AVG", "STDDEV", "VARIANCE", "PERCENTILE"}
        sql_upper = parsed.raw_sql.upper()

        for agg_name in float_sensitive_aggs:
            if f"{agg_name}(" in sql_upper:
                result.violations.append(FragmentViolation(
                    category=ViolationCategory.FLOAT_AGGREGATION,
                    message=(
                        f"Aggregate '{agg_name}' may produce "
                        f"non-deterministic floating-point results"
                    ),
                    node_id=node_id,
                    suggestion=(
                        f"Consider using exact arithmetic (NUMERIC type) "
                        f"or rounding to fixed precision"
                    ),
                ))

    def _check_window_determinism(
        self,
        parsed: ParsedQuery,
        node_id: str,
        result: NodeFragmentResult,
    ) -> None:
        """Check that window functions are deterministic."""
        rank_functions = {"ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE"}
        sql_upper = parsed.raw_sql.upper()

        for func in rank_functions:
            if f"{func}()" in sql_upper:
                if "ORDER BY" not in sql_upper:
                    result.violations.append(FragmentViolation(
                        category=ViolationCategory.NONDETERMINISTIC_WINDOW,
                        message=(
                            f"Window function {func}() without ORDER BY "
                            f"is nondeterministic"
                        ),
                        node_id=node_id,
                        suggestion=f"Add ORDER BY to the {func}() window specification",
                    ))

        for ws in parsed.window_specs:
            func_name = ws.function.value if ws.function else ""
            if func_name in {"ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE"}:
                if not ws.order_by:
                    result.violations.append(FragmentViolation(
                        category=ViolationCategory.NONDETERMINISTIC_WINDOW,
                        message=(
                            f"Window function {func_name} in '{ws.output_alias}' "
                            f"has no ORDER BY"
                        ),
                        node_id=node_id,
                        suggestion="Add ORDER BY to ensure deterministic results",
                    ))

    def _check_recursive_cte(
        self,
        parsed: ParsedQuery,
        node_id: str,
        result: NodeFragmentResult,
    ) -> None:
        """Check recursive CTEs for Fragment F compliance."""
        for cte in parsed.ctes:
            if cte.is_recursive:
                cte_sql_upper = cte.query_sql.upper() if cte.query_sql else ""
                if "ORDER BY" in cte_sql_upper or "LIMIT" in cte_sql_upper:
                    result.violations.append(FragmentViolation(
                        category=ViolationCategory.RECURSIVE_CTE_NONDETERMINISTIC,
                        message=(
                            f"Recursive CTE '{cte.name}' uses ORDER BY or LIMIT, "
                            f"which makes iteration order nondeterministic"
                        ),
                        node_id=node_id,
                        suggestion="Remove ORDER BY/LIMIT from recursive CTE body",
                    ))

    def _check_subqueries(
        self,
        parsed: ParsedQuery,
        node_id: str,
        result: NodeFragmentResult,
    ) -> None:
        """Check subqueries for Fragment F compliance."""
        for sub in parsed.subqueries:
            sub_result = self._check_parsed_query(sub, f"{node_id}.subquery")
            result.violations.extend(sub_result.violations)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def check_fragment_f(sql: str) -> FragmentResult:
    """Quick check if a single query is in Fragment F."""
    checker = FragmentChecker()
    return checker.check_query(sql)


def check_pipeline_fragment_f(
    nodes: List[Tuple[str, str]],
) -> FragmentResult:
    """Quick check if a pipeline is in Fragment F."""
    checker = FragmentChecker()
    return checker.check_pipeline(nodes)


def is_deterministic_query(sql: str) -> bool:
    """Quick check if a query is deterministic (subset of Fragment F check)."""
    checker = FragmentChecker()
    result = checker.check_query(sql)
    nondet = [
        v for v in result.violations
        if v.category in (
            ViolationCategory.NONDETERMINISTIC_FUNCTION,
            ViolationCategory.NONDETERMINISTIC_ORDER,
            ViolationCategory.NONDETERMINISTIC_WINDOW,
            ViolationCategory.TABLESAMPLE,
        )
    ]
    return len(nondet) == 0


def fragment_f_violations(sql: str) -> List[str]:
    """Return human-readable violation messages for a query."""
    result = check_fragment_f(sql)
    return [repr(v) for v in result.violations]
