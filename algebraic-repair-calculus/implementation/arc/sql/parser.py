"""
SQL Parser
===========

Parses SQL queries into structured ParsedQuery objects using sqlglot.
Supports PostgreSQL and DuckDB dialects. Extracts:
- AST
- Operator type classification
- Source tables
- Output columns
- Join conditions
- Filter predicates
- Group by columns
- Aggregations
- Window specifications
- CTEs
- Subqueries
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
    from sqlglot.dialects import postgres, duckdb
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

from arc.sql.operators import (
    AggregateFunctionType,
    ColumnReference,
    CTESpec,
    ExpressionRef,
    JoinConditionSpec,
    JoinKind,
    OrderBySpec,
    SQLOperatorType,
    TableReference,
    TransformationType,
    WindowFrameKind,
    WindowFrameSpec,
    WindowFunctionSpec,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Dialect(Enum):
    """Supported SQL dialects."""
    POSTGRES = "postgres"
    DUCKDB = "duckdb"
    GENERIC = ""


# ---------------------------------------------------------------------------
# Parsed Query
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    """
    Structured representation of a parsed SQL query.
    Contains all operator-level metadata extracted from the AST.
    """
    raw_sql: str
    ast: Any = None  # sqlglot.Expression
    dialect: Dialect = Dialect.GENERIC

    # Classification
    operator_type: SQLOperatorType = SQLOperatorType.SELECT
    is_subquery: bool = False
    is_cte_query: bool = False

    # Source tables
    source_tables: List[TableReference] = field(default_factory=list)

    # Output
    output_columns: List[ColumnReference] = field(default_factory=list)
    output_expressions: List[ExpressionRef] = field(default_factory=list)
    has_star: bool = False

    # Join
    join_conditions: List[JoinConditionSpec] = field(default_factory=list)
    join_kind: Optional[JoinKind] = None

    # Filter
    filter_predicates: List[str] = field(default_factory=list)
    filter_columns: Set[str] = field(default_factory=set)
    having_predicates: List[str] = field(default_factory=list)

    # Group by
    group_by_columns: List[ColumnReference] = field(default_factory=list)
    aggregations: List[ExpressionRef] = field(default_factory=list)

    # Window
    window_specs: List[WindowFunctionSpec] = field(default_factory=list)

    # Order / Limit
    order_by: List[OrderBySpec] = field(default_factory=list)
    limit: Optional[int] = None
    offset: Optional[int] = None

    # CTE
    ctes: List[CTESpec] = field(default_factory=list)

    # Set operations
    set_operation: Optional[str] = None
    set_branches: List[ParsedQuery] = field(default_factory=list)

    # Subqueries
    subqueries: List[ParsedQuery] = field(default_factory=list)

    # Distinct
    is_distinct: bool = False

    # Metadata
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def column_names(self) -> List[str]:
        """Return output column names."""
        names = [c.output_name for c in self.output_columns]
        names.extend(e.output_name for e in self.output_expressions)
        return names

    def all_source_columns(self) -> Set[str]:
        """Return all referenced source columns."""
        cols: Set[str] = set()
        for c in self.output_columns:
            cols.add(c.name)
        for e in self.output_expressions:
            cols.update(e.source_columns)
        for c in self.group_by_columns:
            cols.add(c.name)
        cols.update(self.filter_columns)
        for jc in self.join_conditions:
            cols.add(jc.left.name)
            cols.add(jc.right.name)
        for ws in self.window_specs:
            for c in ws.input_columns:
                cols.add(c.name)
            for c in ws.partition_by:
                cols.add(c.name)
        return cols

    def has_aggregation(self) -> bool:
        return bool(self.aggregations) or bool(self.group_by_columns)

    def has_joins(self) -> bool:
        return bool(self.join_conditions) or self.join_kind is not None

    def has_window_functions(self) -> bool:
        return bool(self.window_specs)

    def has_ctes(self) -> bool:
        return bool(self.ctes)

    def has_subqueries(self) -> bool:
        return bool(self.subqueries)

    def has_set_operations(self) -> bool:
        return self.set_operation is not None

    def has_ordering(self) -> bool:
        return bool(self.order_by)

    def has_limit(self) -> bool:
        return self.limit is not None

    def complexity_score(self) -> int:
        """Estimate query complexity (higher = more complex)."""
        score = 1
        score += len(self.source_tables)
        score += len(self.join_conditions) * 2
        score += len(self.group_by_columns)
        score += len(self.aggregations)
        score += len(self.window_specs) * 3
        score += len(self.ctes) * 2
        score += len(self.subqueries) * 3
        if self.set_operation:
            score += 2
        return score

    def __repr__(self) -> str:
        parts = [self.operator_type.value]
        if self.source_tables:
            tables = ", ".join(t.effective_name for t in self.source_tables)
            parts.append(f"from=[{tables}]")
        if self.join_kind:
            parts.append(f"join={self.join_kind.value}")
        if self.group_by_columns:
            parts.append(f"group_by={len(self.group_by_columns)}")
        if self.window_specs:
            parts.append(f"windows={len(self.window_specs)}")
        if self.ctes:
            parts.append(f"ctes={len(self.ctes)}")
        return f"ParsedQuery({', '.join(parts)})"


# ---------------------------------------------------------------------------
# SQL Parser
# ---------------------------------------------------------------------------

class SQLParser:
    """
    Parses SQL queries into ParsedQuery objects.
    Uses sqlglot for AST parsing with support for PostgreSQL and DuckDB dialects.
    """

    def __init__(self, dialect: str = "postgres") -> None:
        self.dialect = Dialect(dialect) if dialect else Dialect.GENERIC

    def parse(self, sql: str, dialect: Optional[str] = None) -> ParsedQuery:
        """Parse a single SQL query into a ParsedQuery."""
        effective_dialect = Dialect(dialect) if dialect else self.dialect
        query = ParsedQuery(raw_sql=sql.strip(), dialect=effective_dialect)

        if not HAS_SQLGLOT:
            query.errors.append("sqlglot not available; falling back to regex parsing")
            return self._fallback_parse(sql, query)

        try:
            read_dialect = effective_dialect.value if effective_dialect != Dialect.GENERIC else None
            parsed = sqlglot.parse(sql, read=read_dialect)
            if not parsed:
                query.errors.append("No statements parsed")
                return query

            ast = parsed[0]
            if ast is None:
                query.errors.append("Parse returned None")
                return query

            query.ast = ast
            self._extract_all(ast, query)

        except Exception as e:
            query.errors.append(f"Parse error: {e}")
            return self._fallback_parse(sql, query)

        return query

    def parse_many(self, sqls: List[str], dialect: Optional[str] = None) -> List[ParsedQuery]:
        """Parse multiple SQL queries."""
        return [self.parse(sql, dialect) for sql in sqls]

    def parse_file(self, sql_text: str, dialect: Optional[str] = None) -> List[ParsedQuery]:
        """Parse a SQL file containing multiple statements."""
        statements = self._split_statements(sql_text)
        return self.parse_many(statements, dialect)

    # -----------------------------------------------------------------------
    # Extraction Methods
    # -----------------------------------------------------------------------

    def _extract_all(self, ast: Any, query: ParsedQuery) -> None:
        """Extract all information from the AST."""
        self._classify_operator(ast, query)
        self._extract_ctes(ast, query)
        self._extract_sources(ast, query)
        self._extract_columns(ast, query)
        self._extract_joins(ast, query)
        self._extract_where(ast, query)
        self._extract_group_by(ast, query)
        self._extract_having(ast, query)
        self._extract_aggregations(ast, query)
        self._extract_windows(ast, query)
        self._extract_order_by(ast, query)
        self._extract_limit(ast, query)
        self._extract_set_operations(ast, query)
        self._extract_subqueries(ast, query)
        self._extract_distinct(ast, query)

    def _classify_operator(self, ast: Any, query: ParsedQuery) -> None:
        """Classify the primary operator type of the query."""
        if isinstance(ast, exp.Union):
            query.operator_type = SQLOperatorType.UNION
        elif isinstance(ast, exp.Intersect):
            query.operator_type = SQLOperatorType.SET_OP
        elif isinstance(ast, exp.Except):
            query.operator_type = SQLOperatorType.SET_OP
        elif isinstance(ast, exp.Select):
            group_by = ast.find(exp.Group)
            joins = list(ast.find_all(exp.Join))
            windows = list(ast.find_all(exp.Window))

            if group_by:
                query.operator_type = SQLOperatorType.GROUP_BY
            elif joins:
                query.operator_type = SQLOperatorType.JOIN
            elif windows:
                query.operator_type = SQLOperatorType.WINDOW
            else:
                query.operator_type = SQLOperatorType.SELECT
        else:
            query.operator_type = SQLOperatorType.SELECT

    def _extract_ctes(self, ast: Any, query: ParsedQuery) -> None:
        """Extract Common Table Expressions."""
        for cte in ast.find_all(exp.CTE):
            cte_name = cte.alias
            if not cte_name:
                continue

            cols = ()
            cte_alias = cte.find(exp.TableAlias)
            if cte_alias:
                col_defs = list(cte_alias.find_all(exp.Column))
                if col_defs:
                    cols = tuple(c.name for c in col_defs)

            cte_sql = ""
            this = cte.this
            if this:
                cte_sql = this.sql()

            is_recursive = False
            with_node = ast.find(exp.With)
            if with_node:
                is_recursive = getattr(with_node, "recursive", False)

            query.ctes.append(CTESpec(
                name=cte_name,
                columns=cols,
                query_sql=cte_sql,
                is_recursive=is_recursive,
            ))
            query.is_cte_query = True

    def _extract_sources(self, ast: Any, query: ParsedQuery) -> None:
        """Extract source tables."""
        for table in ast.find_all(exp.Table):
            name = table.name
            if not name:
                continue
            schema_name = None
            if hasattr(table, "db") and table.db:
                schema_name = table.db

            alias = table.alias if hasattr(table, "alias") else None
            is_cte = any(c.name == name for c in query.ctes)

            query.source_tables.append(TableReference(
                name=name,
                schema=schema_name,
                alias=alias,
                is_cte=is_cte,
            ))

    def _extract_columns(self, ast: Any, query: ParsedQuery) -> None:
        """Extract output columns and expressions."""
        if not isinstance(ast, exp.Select):
            select = ast.find(exp.Select)
            if select is None:
                return
            ast_select = select
        else:
            ast_select = ast

        for expr_node in ast_select.expressions:
            if isinstance(expr_node, exp.Star):
                query.has_star = True
                continue

            alias = None
            if isinstance(expr_node, exp.Alias):
                alias = expr_node.alias
                inner = expr_node.this
            else:
                inner = expr_node

            if isinstance(inner, exp.Column):
                col_name = inner.name
                tbl = inner.table if hasattr(inner, "table") else None
                query.output_columns.append(ColumnReference(
                    name=col_name,
                    table=tbl,
                    alias=alias,
                ))
            else:
                sql_str = inner.sql()
                source_cols = self._extract_column_refs(inner)
                is_agg = bool(inner.find(exp.AggFunc))
                is_window = bool(inner.find(exp.Window))

                if is_agg:
                    trans_type = TransformationType.AGGREGATED
                elif is_window:
                    trans_type = TransformationType.WINDOWED
                else:
                    trans_type = TransformationType.COMPUTED

                _NONDETERMINISTIC = {"random", "rand", "uuid", "gen_random_uuid", "newid", "setseed"}
                has_anon = bool(inner.find(exp.Anonymous))
                has_nondet_func = any(
                    getattr(fn, 'sql_name', lambda: '')().lower() in _NONDETERMINISTIC
                    or fn.key in _NONDETERMINISTIC
                    for fn in inner.find_all(exp.Func)
                )
                is_det = not has_anon and not has_nondet_func

                query.output_expressions.append(ExpressionRef(
                    sql=sql_str,
                    output_alias=alias,
                    source_columns=tuple(source_cols),
                    transformation=trans_type,
                    is_deterministic=is_det,
                ))

    def _extract_joins(self, ast: Any, query: ParsedQuery) -> None:
        """Extract join information."""
        for join_node in ast.find_all(exp.Join):
            kind_str = join_node.side or ""
            kind_upper = kind_str.upper() if kind_str else ""
            join_kind = JoinKind.INNER

            if "LEFT" in kind_upper:
                join_kind = JoinKind.LEFT
            elif "RIGHT" in kind_upper:
                join_kind = JoinKind.RIGHT
            elif "FULL" in kind_upper:
                join_kind = JoinKind.FULL
            elif "CROSS" in kind_upper:
                join_kind = JoinKind.CROSS

            if hasattr(join_node, "kind") and join_node.kind:
                k = join_node.kind.upper()
                if "SEMI" in k:
                    join_kind = JoinKind.SEMI
                elif "ANTI" in k:
                    join_kind = JoinKind.ANTI

            query.join_kind = join_kind

            on_clause = join_node.find(exp.On)
            if on_clause:
                for eq in on_clause.find_all(exp.EQ):
                    left_expr = eq.left
                    right_expr = eq.right

                    left_col = self._expr_to_column_ref(left_expr)
                    right_col = self._expr_to_column_ref(right_expr)

                    if left_col and right_col:
                        query.join_conditions.append(JoinConditionSpec(
                            left=left_col,
                            right=right_col,
                            operator="=",
                        ))

            using_clause = join_node.find(exp.Using) if hasattr(exp, "Using") else None

    def _extract_where(self, ast: Any, query: ParsedQuery) -> None:
        """Extract WHERE clause."""
        where = ast.find(exp.Where)
        if where is None:
            return

        pred_sql = where.this.sql() if where.this else ""
        if pred_sql:
            query.filter_predicates.append(pred_sql)
            query.filter_columns = self._extract_column_refs(where.this)

    def _extract_group_by(self, ast: Any, query: ParsedQuery) -> None:
        """Extract GROUP BY columns."""
        group = ast.find(exp.Group)
        if group is None:
            return

        for g_expr in group.expressions:
            if isinstance(g_expr, exp.Column):
                query.group_by_columns.append(ColumnReference(
                    name=g_expr.name,
                    table=g_expr.table if hasattr(g_expr, "table") else None,
                ))
            else:
                cols = self._extract_column_refs(g_expr)
                if cols:
                    for c in cols:
                        query.group_by_columns.append(ColumnReference(name=c))

    def _extract_having(self, ast: Any, query: ParsedQuery) -> None:
        """Extract HAVING clause."""
        having = ast.find(exp.Having)
        if having is None:
            return

        pred_sql = having.this.sql() if having.this else ""
        if pred_sql:
            query.having_predicates.append(pred_sql)

    def _extract_aggregations(self, ast: Any, query: ParsedQuery) -> None:
        """Extract aggregate function calls."""
        for agg_node in ast.find_all(exp.AggFunc):
            if agg_node.find_ancestor(exp.Window):
                continue

            func_name = type(agg_node).__name__.upper()
            source_cols = self._extract_column_refs(agg_node)

            alias = None
            parent = agg_node.parent
            if isinstance(parent, exp.Alias):
                alias = parent.alias

            agg_sql = agg_node.sql()
            query.aggregations.append(ExpressionRef(
                sql=agg_sql,
                output_alias=alias or agg_sql,
                source_columns=tuple(source_cols),
                transformation=TransformationType.AGGREGATED,
            ))

    def _extract_windows(self, ast: Any, query: ParsedQuery) -> None:
        """Extract window function specifications."""
        for win_node in ast.find_all(exp.Window):
            func_node = win_node.this
            func_name = type(func_node).__name__.upper() if func_node else "UNKNOWN"

            input_cols = tuple(
                ColumnReference(name=c)
                for c in self._extract_column_refs(func_node)
            ) if func_node else ()

            alias = None
            parent = win_node.parent
            if isinstance(parent, exp.Alias):
                alias = parent.alias

            partition_by = ()
            partition = win_node.find(exp.PartitionedByProperty)
            if partition is None:
                partition = win_node.args.get("partition_by")
            if partition:
                if isinstance(partition, list):
                    partition_by = tuple(
                        ColumnReference(name=c)
                        for p in partition
                        for c in self._extract_column_refs(p)
                    )
                elif hasattr(partition, "expressions"):
                    partition_by = tuple(
                        ColumnReference(name=c)
                        for p in partition.expressions
                        for c in self._extract_column_refs(p)
                    )

            order_specs = ()
            order = win_node.args.get("order")
            if order:
                obs = []
                order_exprs = order.expressions if hasattr(order, "expressions") else [order]
                for oe in order_exprs:
                    col_name = None
                    ascending = True
                    if isinstance(oe, exp.Ordered):
                        col_refs = self._extract_column_refs(oe.this)
                        col_name = next(iter(col_refs)) if col_refs else None
                        ascending = not oe.args.get("desc", False)
                    elif isinstance(oe, exp.Column):
                        col_name = oe.name
                    if col_name:
                        obs.append(OrderBySpec(
                            column=ColumnReference(name=col_name),
                            ascending=ascending,
                        ))
                order_specs = tuple(obs)

            func_type = self._map_aggregate_function(func_name)

            query.window_specs.append(WindowFunctionSpec(
                function=func_type,
                input_columns=input_cols,
                output_alias=alias or win_node.sql(),
                partition_by=partition_by,
                order_by=order_specs,
            ))

    def _extract_order_by(self, ast: Any, query: ParsedQuery) -> None:
        """Extract ORDER BY clause."""
        order = ast.find(exp.Order)
        if order is None:
            return

        for oe in order.expressions:
            if isinstance(oe, exp.Ordered):
                col_refs = self._extract_column_refs(oe.this)
                col_name = next(iter(col_refs), None)
                if col_name:
                    query.order_by.append(OrderBySpec(
                        column=ColumnReference(name=col_name),
                        ascending=not oe.args.get("desc", False),
                    ))
            elif isinstance(oe, exp.Column):
                query.order_by.append(OrderBySpec(
                    column=ColumnReference(name=oe.name),
                ))

    def _extract_limit(self, ast: Any, query: ParsedQuery) -> None:
        """Extract LIMIT/OFFSET."""
        limit_node = ast.find(exp.Limit)
        if limit_node and limit_node.this:
            try:
                query.limit = int(limit_node.this.this)
            except (ValueError, TypeError, AttributeError):
                pass

        offset_node = ast.find(exp.Offset)
        if offset_node and offset_node.this:
            try:
                query.offset = int(offset_node.this.this)
            except (ValueError, TypeError, AttributeError):
                pass

    def _extract_set_operations(self, ast: Any, query: ParsedQuery) -> None:
        """Extract UNION/INTERSECT/EXCEPT operations."""
        if isinstance(ast, exp.Union):
            query.set_operation = "UNION ALL" if ast.args.get("distinct") is False else "UNION"
            for branch in [ast.left, ast.right]:
                if branch:
                    sub = ParsedQuery(raw_sql=branch.sql())
                    sub.ast = branch
                    self._extract_all(branch, sub)
                    query.set_branches.append(sub)
        elif isinstance(ast, exp.Intersect):
            query.set_operation = "INTERSECT"
            for branch in [ast.left, ast.right]:
                if branch:
                    sub = ParsedQuery(raw_sql=branch.sql())
                    sub.ast = branch
                    self._extract_all(branch, sub)
                    query.set_branches.append(sub)
        elif isinstance(ast, exp.Except):
            query.set_operation = "EXCEPT"
            for branch in [ast.left, ast.right]:
                if branch:
                    sub = ParsedQuery(raw_sql=branch.sql())
                    sub.ast = branch
                    self._extract_all(branch, sub)
                    query.set_branches.append(sub)

    def _extract_subqueries(self, ast: Any, query: ParsedQuery) -> None:
        """Extract subqueries."""
        for subq in ast.find_all(exp.Subquery):
            inner = subq.this
            if inner:
                sub = ParsedQuery(raw_sql=inner.sql(), is_subquery=True)
                sub.ast = inner
                try:
                    self._extract_all(inner, sub)
                except Exception:
                    pass
                query.subqueries.append(sub)

    def _extract_distinct(self, ast: Any, query: ParsedQuery) -> None:
        """Check for DISTINCT."""
        if isinstance(ast, exp.Select):
            distinct = ast.find(exp.Distinct)
            if distinct:
                query.is_distinct = True

    # -----------------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------------

    def _extract_column_refs(self, node: Any) -> Set[str]:
        """Extract column name references from an expression node."""
        cols: Set[str] = set()
        if node is None:
            return cols
        for col in node.find_all(exp.Column):
            if col.name:
                cols.add(col.name)
        return cols

    def _expr_to_column_ref(self, node: Any) -> Optional[ColumnReference]:
        """Convert an expression to a ColumnReference if possible."""
        if isinstance(node, exp.Column):
            return ColumnReference(
                name=node.name,
                table=node.table if hasattr(node, "table") else None,
            )
        cols = self._extract_column_refs(node)
        if cols:
            return ColumnReference(name=next(iter(cols)))
        return None

    def _map_aggregate_function(self, name: str) -> AggregateFunctionType:
        """Map a function name to an AggregateFunctionType."""
        mapping = {
            "COUNT": AggregateFunctionType.COUNT,
            "SUM": AggregateFunctionType.SUM,
            "AVG": AggregateFunctionType.AVG,
            "MIN": AggregateFunctionType.MIN,
            "MAX": AggregateFunctionType.MAX,
            "COUNTDISTINCT": AggregateFunctionType.COUNT_DISTINCT,
            "ARRAYAGG": AggregateFunctionType.ARRAY_AGG,
            "STRINGAGG": AggregateFunctionType.STRING_AGG,
            "BOOLAND": AggregateFunctionType.BOOL_AND,
            "BOOLOR": AggregateFunctionType.BOOL_OR,
            "STDDEV": AggregateFunctionType.STDDEV,
            "VARIANCE": AggregateFunctionType.VARIANCE,
            "ROWNUMBER": AggregateFunctionType.ROW_NUMBER,
            "RANK": AggregateFunctionType.RANK,
            "DENSERANK": AggregateFunctionType.DENSE_RANK,
            "NTILE": AggregateFunctionType.NTILE,
            "LAG": AggregateFunctionType.LAG,
            "LEAD": AggregateFunctionType.LEAD,
            "FIRSTVALUE": AggregateFunctionType.FIRST_VALUE,
            "LASTVALUE": AggregateFunctionType.LAST_VALUE,
            "NTHVALUE": AggregateFunctionType.NTH_VALUE,
            "CUMEDIST": AggregateFunctionType.CUME_DIST,
            "PERCENTRANK": AggregateFunctionType.PERCENT_RANK,
        }
        clean = re.sub(r"[_\s]", "", name.upper())
        return mapping.get(clean, AggregateFunctionType.COUNT)

    def _split_statements(self, sql_text: str) -> List[str]:
        """Split SQL text into individual statements."""
        stmts: List[str] = []
        current: List[str] = []
        depth = 0
        in_string = False
        quote_char = ""

        for char in sql_text:
            if in_string:
                current.append(char)
                if char == quote_char:
                    in_string = False
                continue

            if char in ("'", '"'):
                in_string = True
                quote_char = char
                current.append(char)
                continue

            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1

            if char == ";" and depth == 0:
                stmt = "".join(current).strip()
                if stmt:
                    stmts.append(stmt)
                current = []
            else:
                current.append(char)

        last = "".join(current).strip()
        if last:
            stmts.append(last)

        return stmts

    # -----------------------------------------------------------------------
    # Fallback Parsing (when sqlglot is unavailable)
    # -----------------------------------------------------------------------

    def _fallback_parse(self, sql: str, query: ParsedQuery) -> ParsedQuery:
        """Regex-based fallback parser."""
        sql_upper = sql.upper().strip()

        if "UNION" in sql_upper:
            query.operator_type = SQLOperatorType.UNION
        elif "INTERSECT" in sql_upper:
            query.operator_type = SQLOperatorType.SET_OP
        elif "EXCEPT" in sql_upper:
            query.operator_type = SQLOperatorType.SET_OP
        elif "GROUP BY" in sql_upper:
            query.operator_type = SQLOperatorType.GROUP_BY
        elif "JOIN" in sql_upper:
            query.operator_type = SQLOperatorType.JOIN
        elif "WINDOW" in sql_upper or "OVER" in sql_upper:
            query.operator_type = SQLOperatorType.WINDOW
        else:
            query.operator_type = SQLOperatorType.SELECT

        table_match = re.findall(r"FROM\s+(\w+)", sql, re.IGNORECASE)
        for t in table_match:
            query.source_tables.append(TableReference(name=t))

        join_tables = re.findall(r"JOIN\s+(\w+)", sql, re.IGNORECASE)
        for t in join_tables:
            query.source_tables.append(TableReference(name=t))

        where_match = re.search(
            r"WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|HAVING|UNION|INTERSECT|EXCEPT|$)",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if where_match:
            pred = where_match.group(1).strip().rstrip(";")
            query.filter_predicates.append(pred)

        group_match = re.search(
            r"GROUP\s+BY\s+(.+?)(?:HAVING|ORDER|LIMIT|$)",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if group_match:
            cols = group_match.group(1).strip().rstrip(";").split(",")
            for c in cols:
                c = c.strip()
                if c:
                    query.group_by_columns.append(ColumnReference(name=c))

        limit_match = re.search(r"LIMIT\s+(\d+)", sql, re.IGNORECASE)
        if limit_match:
            query.limit = int(limit_match.group(1))

        if "DISTINCT" in sql_upper:
            query.is_distinct = True

        cte_matches = re.finditer(
            r"(\w+)\s+AS\s*\(",
            sql[:sql_upper.find("SELECT")] if "SELECT" in sql_upper else "",
            re.IGNORECASE,
        )
        for m in cte_matches:
            query.ctes.append(CTESpec(name=m.group(1)))

        return query
