"""
PySpark-specific lineage extraction via AST analysis.

:class:`PySparkAnalyzer` walks the AST of PySpark-based ETL code and
extracts:

* **Source nodes** – ``spark.read.csv``, ``spark.read.parquet``, ``spark.sql``.
* **Transformations** – ``select``, ``filter``/``where``, ``withColumn``, ``groupBy``.
* **Join operations** – ``join``.
* **Sink nodes** – ``write.csv``, ``write.parquet``, ``write.saveAsTable``.
* **SQL queries** – literal SQL passed to ``spark.sql()``.
* **Column lineage** using PySpark ``col()`` / ``F.col()`` references.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

from arc.python_etl.ast_utils import (
    ast_to_source,
    extract_keyword_arg,
    extract_string_args,
    extract_string_list_arg,
    get_root_variable,
    resolve_method_chain,
    safe_parse,
    track_assignments,
)


def _resolve_chain_with_calls(node: ast.expr) -> list[tuple[str, ast.Call | None]]:
    """Resolve a method chain into (method_name, call_node) pairs.

    Like ``resolve_method_chain`` but also returns the call node so that
    arguments can be inspected.
    """
    chain: list[tuple[str, ast.Call | None]] = []
    current: ast.expr = node

    while True:
        if isinstance(current, ast.Call):
            if isinstance(current.func, ast.Attribute):
                chain.append((current.func.attr, current))
                current = current.func.value
            else:
                break
        elif isinstance(current, ast.Attribute):
            chain.append((current.attr, None))
            current = current.value
        elif isinstance(current, ast.Subscript):
            chain.append(("__getitem__", None))
            current = current.value
        else:
            break

    chain.reverse()
    return chain
from arc.types.base import (
    DataflowEdge,
    DataflowGraph,
    DataflowNode,
    ETLFramework,
    LineageChain,
    LineageStep,
    QualityPattern,
    QualityPatternType,
    SparkLineage,
    Transformation,
    TransformationType,
)

logger = logging.getLogger(__name__)

# ── Method → TransformationType mapping ────────────────────────────────

_SPARK_METHOD_MAP: dict[str, TransformationType] = {
    # Sources
    "csv": TransformationType.SOURCE,
    "parquet": TransformationType.SOURCE,
    "json": TransformationType.SOURCE,
    "orc": TransformationType.SOURCE,
    "avro": TransformationType.SOURCE,
    "jdbc": TransformationType.SOURCE,
    "table": TransformationType.SOURCE,
    "text": TransformationType.SOURCE,
    "load": TransformationType.SOURCE,
    "sql": TransformationType.SOURCE,
    "createDataFrame": TransformationType.SOURCE,
    "range": TransformationType.SOURCE,
    # Selection
    "select": TransformationType.SELECT,
    "selectExpr": TransformationType.SELECT,
    "col": TransformationType.SELECT,
    "columns": TransformationType.SELECT,
    # Filtering
    "filter": TransformationType.FILTER,
    "where": TransformationType.FILTER,
    "limit": TransformationType.FILTER,
    "distinct": TransformationType.FILTER,
    "dropDuplicates": TransformationType.FILTER,
    "drop_duplicates": TransformationType.FILTER,
    "sample": TransformationType.FILTER,
    # Schema mutation
    "withColumn": TransformationType.ASSIGN,
    "withColumnRenamed": TransformationType.RENAME,
    "withColumnsRenamed": TransformationType.RENAME,
    "drop": TransformationType.DROP,
    "toDF": TransformationType.RENAME,
    "alias": TransformationType.RENAME,
    "cast": TransformationType.CAST,
    # Joins
    "join": TransformationType.JOIN,
    "crossJoin": TransformationType.JOIN,
    # Grouping / Aggregation
    "groupBy": TransformationType.GROUP_BY,
    "groupby": TransformationType.GROUP_BY,
    "rollup": TransformationType.GROUP_BY,
    "cube": TransformationType.GROUP_BY,
    "agg": TransformationType.GROUP_BY,
    "count": TransformationType.GROUP_BY,
    "sum": TransformationType.GROUP_BY,
    "avg": TransformationType.GROUP_BY,
    "mean": TransformationType.GROUP_BY,
    "min": TransformationType.GROUP_BY,
    "max": TransformationType.GROUP_BY,
    # Pivoting
    "pivot": TransformationType.PIVOT,
    "unpivot": TransformationType.UNPIVOT,
    # Sorting
    "orderBy": TransformationType.SORT,
    "sort": TransformationType.SORT,
    "sortWithinPartitions": TransformationType.SORT,
    # Null handling
    "na": TransformationType.DROPNA,
    "dropna": TransformationType.DROPNA,
    "fillna": TransformationType.FILLNA,
    # Sinks
    "save": TransformationType.SINK,
    "saveAsTable": TransformationType.SINK,
    "insertInto": TransformationType.SINK,
    # Window
    "over": TransformationType.WINDOW,
    "partitionBy": TransformationType.WINDOW,
    # Set ops
    "union": TransformationType.UNION,
    "unionAll": TransformationType.UNION,
    "unionByName": TransformationType.UNION,
    "intersect": TransformationType.FILTER,
    "intersectAll": TransformationType.FILTER,
    "subtract": TransformationType.FILTER,
    "exceptAll": TransformationType.FILTER,
    # Custom
    "foreach": TransformationType.CUSTOM,
    "foreachBatch": TransformationType.CUSTOM,
    "transform": TransformationType.CUSTOM,
}

# Methods that signify a write chain (write.X)
_WRITE_METHODS = frozenset({
    "save", "saveAsTable", "insertInto", "csv", "parquet",
    "json", "orc", "avro", "text", "jdbc",
})

# Regex for column references in SQL strings
_SQL_TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN|INTO)\s+([a-zA-Z_][a-zA-Z0-9_.]*)",
    re.IGNORECASE,
)
_SQL_COL_RE = re.compile(
    r"(?:SELECT|WHERE|ON|AND|OR|GROUP\s+BY|ORDER\s+BY)\s+([a-zA-Z_*][a-zA-Z0-9_.*,\s]*)",
    re.IGNORECASE,
)


class PySparkAnalyzer:
    """Extract lineage from PySpark code using AST analysis.

    Walks the AST looking for PySpark DataFrame operations, builds
    lineage chains and a dataflow graph.
    """

    def __init__(self) -> None:
        self._source: str = ""
        self._tree: ast.Module | None = None
        self._lines: list[str] = []
        self._assignments: dict[str, list[int]] = {}
        self._df_ops: dict[str, list[Transformation]] = {}
        self._sql_queries: list[tuple[int, str]] = []

    # ── Public API ─────────────────────────────────────────────────────

    def analyze(self, source: str) -> SparkLineage:
        """Analyze PySpark source code and extract lineage.

        Parameters
        ----------
        source:
            Python source code.

        Returns
        -------
        SparkLineage
        """
        self._source = source
        self._lines = source.splitlines()
        self._tree = safe_parse(source)
        if self._tree is None:
            return SparkLineage()

        self._assignments = track_assignments(self._tree)
        self._df_ops = {}
        self._sql_queries = []

        transformations: list[Transformation] = []
        sources: list[str] = []
        sinks: list[str] = []

        # Walk the AST for assignments and expressions
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        ts = self._analyze_assignment(target.id, node.value, node.lineno)
                        for t in ts:
                            transformations.append(t)
                            self._df_ops.setdefault(target.id, []).append(t)
                            if t.transform_type == TransformationType.SOURCE:
                                sources.append(target.id)

            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                ts = self._analyze_expression_chain(node.value, node.lineno)
                for t in ts:
                    transformations.append(t)
                    if t.transform_type == TransformationType.SINK:
                        sinks.append(t.output_var or "")

        # Build lineage chains
        chains: dict[str, LineageChain] = {}
        for var_name, ops in self._df_ops.items():
            steps = []
            for op in ops:
                steps.append(LineageStep(
                    variable_name=var_name,
                    operation=op.transform_type.value,
                    source_line=op.source_line,
                    columns_in=op.columns_read,
                    columns_out=op.columns_written,
                    metadata=op.parameters,
                ))
            chains[var_name] = LineageChain(
                variable_name=var_name,
                steps=tuple(steps),
            )

        graph = self._build_graph(transformations)

        return SparkLineage(
            chains=chains,
            sources=tuple(sources),
            sinks=tuple(sinks),
            transformations=tuple(transformations),
            dataflow_graph=graph,
            sql_queries=tuple(q for _, q in self._sql_queries),
        )

    def trace_dataframe_lineage(
        self,
        source: str,
        df_name: str,
    ) -> LineageChain:
        """Trace lineage for a specific DataFrame variable."""
        lineage = self.analyze(source)
        return lineage.chains.get(df_name, LineageChain(variable_name=df_name))

    def extract_sql_queries(self, source: str) -> list[tuple[int, str]]:
        """Extract SQL queries from spark.sql(...) calls.

        Returns
        -------
        list of (lineno, sql_string) tuples
        """
        tree = safe_parse(source)
        if tree is None:
            return []

        queries: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "sql":
                continue

            root = get_root_variable(node.func.value)
            if root not in ("spark", "sqlContext", "hive_context"):
                continue

            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    queries.append((node.lineno, arg.value))

        return queries

    def detect_quality_patterns(self, source: str) -> list[QualityPattern]:
        """Detect quality-related patterns in PySpark code."""
        tree = safe_parse(source)
        if tree is None:
            return []

        patterns: list[QualityPattern] = []
        lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue

            method = node.func.attr
            src_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""

            if method in ("dropna", "na.drop"):
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.NULL_CHECK,
                    source_line=node.lineno,
                    source_text=src_line,
                    description="PySpark null removal",
                ))

            elif method in ("fillna", "na.fill"):
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.NULL_CHECK,
                    source_line=node.lineno,
                    source_text=src_line,
                    description="PySpark null filling",
                ))

            elif method in ("isNull", "isNotNull"):
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.NULL_CHECK,
                    source_line=node.lineno,
                    source_text=src_line,
                    description=f"PySpark null check: .{method}()",
                ))

            elif method in ("dropDuplicates", "drop_duplicates"):
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.DEDUPLICATION,
                    source_line=node.lineno,
                    source_text=src_line,
                    description="PySpark deduplication",
                ))

            elif method == "cast":
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.TYPE_CAST,
                    source_line=node.lineno,
                    source_text=src_line,
                    description="PySpark type cast",
                ))

            elif method in ("between",):
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.RANGE_VALIDATION,
                    source_line=node.lineno,
                    source_text=src_line,
                    description="PySpark range validation",
                ))

            elif method in ("rlike", "like"):
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.REGEX_VALIDATION,
                    source_line=node.lineno,
                    source_text=src_line,
                    description=f"PySpark pattern validation: .{method}()",
                ))

        return patterns

    # ── Private methods ────────────────────────────────────────────────

    def _analyze_assignment(
        self,
        target_name: str,
        value: ast.expr,
        lineno: int,
    ) -> list[Transformation]:
        """Analyze a single assignment, handling method chains."""
        if isinstance(value, ast.Call):
            chain = _resolve_chain_with_calls(value)
            return self._process_chain(target_name, chain, lineno)
        return []

    def _analyze_expression_chain(
        self,
        call: ast.Call,
        lineno: int,
    ) -> list[Transformation]:
        """Analyze a standalone expression (e.g. df.write.parquet(...))."""
        chain = _resolve_chain_with_calls(call)
        return self._process_chain("", chain, lineno)

    def _process_chain(
        self,
        target_name: str,
        chain: list[tuple[str, ast.Call | None]],
        lineno: int,
    ) -> list[Transformation]:
        """Process a resolved method chain into transformations."""
        if not chain:
            return []

        transformations: list[Transformation] = []
        input_vars: list[str] = []
        is_write_chain = False

        # Identify root DataFrame from the first call
        if chain:
            first_method, first_call = chain[0]
            if first_call is not None and isinstance(first_call.func, ast.Attribute):
                root = get_root_variable(first_call.func.value)
                if root and root in self._assignments:
                    input_vars.append(root)

        for method_name, call_node in chain:
            # Detect write chain
            if method_name == "write":
                is_write_chain = True
                continue
            if method_name == "mode":
                continue
            if method_name in ("format", "option", "options", "schema"):
                continue

            # Determine transformation type
            if is_write_chain and method_name in _WRITE_METHODS:
                tt = TransformationType.SINK
            elif method_name == "read" or method_name == "readStream":
                continue
            else:
                tt = _SPARK_METHOD_MAP.get(method_name)
                if tt is None:
                    continue

            cols_read: list[str] = []
            cols_written: list[str] = []

            if call_node is not None:
                cols_read = self._extract_column_refs(call_node, method_name)

                # Special handling for certain methods
                if method_name in ("withColumn", "withColumnRenamed"):
                    wc_cols = extract_string_args(call_node)
                    if wc_cols:
                        if method_name == "withColumn":
                            cols_written.append(wc_cols[0])
                        elif method_name == "withColumnRenamed" and len(wc_cols) >= 2:
                            cols_read.append(wc_cols[0])
                            cols_written.append(wc_cols[1])

                if method_name == "join":
                    for arg in call_node.args:
                        if isinstance(arg, ast.Name) and arg.id in self._assignments:
                            if arg.id not in input_vars:
                                input_vars.append(arg.id)

                if method_name == "sql":
                    for arg in call_node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            self._sql_queries.append((lineno, arg.value))
                            cols_read.extend(self._extract_columns_from_sql(arg.value))

                if method_name in ("select", "selectExpr"):
                    cols_read = self._extract_select_columns(call_node)

                if method_name == "groupBy":
                    cols_read = self._extract_groupby_columns(call_node)

            src_line = self._lines[lineno - 1].strip() if lineno <= len(self._lines) else ""

            transformations.append(Transformation(
                transform_type=tt,
                input_vars=tuple(input_vars),
                output_var=target_name,
                columns_read=tuple(dict.fromkeys(cols_read)),
                columns_written=tuple(dict.fromkeys(cols_written)),
                source_line=lineno,
                source_text=src_line,
                parameters={"method": method_name},
            ))

        return transformations

    def _build_graph(
        self,
        transformations: list[Transformation],
    ) -> DataflowGraph:
        """Build a DataflowGraph from transformations."""
        graph = DataflowGraph(framework=ETLFramework.PYSPARK)
        var_to_node: dict[str, str] = {}

        for i, t in enumerate(transformations):
            nid = f"spark_{i}_{t.transform_type.value.lower()}"
            graph.add_node(DataflowNode(
                node_id=nid,
                variable_name=t.output_var,
                transform=t,
            ))

            if t.output_var:
                var_to_node[t.output_var] = nid

            for inp in t.input_vars:
                src_node = var_to_node.get(inp)
                if src_node is not None:
                    graph.add_edge(DataflowEdge(
                        source=src_node,
                        target=nid,
                        columns=t.columns_read,
                    ))

        return graph

    def _extract_column_refs(
        self,
        call: ast.Call,
        method_name: str,
    ) -> list[str]:
        """Extract column name references from call arguments."""
        cols: list[str] = []
        for arg in call.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                cols.append(arg.value)
            elif isinstance(arg, ast.Call):
                # col("name") or F.col("name")
                fname = self._get_call_name(arg)
                if fname in ("col", "column"):
                    for a in arg.args:
                        if isinstance(a, ast.Constant) and isinstance(a.value, str):
                            cols.append(a.value)
        return cols

    def _extract_select_columns(self, call: ast.Call) -> list[str]:
        """Extract column names from select() or selectExpr()."""
        cols: list[str] = []
        for arg in call.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                cols.append(arg.value)
            elif isinstance(arg, ast.Call):
                fname = self._get_call_name(arg)
                if fname in ("col", "column"):
                    for a in arg.args:
                        if isinstance(a, ast.Constant) and isinstance(a.value, str):
                            cols.append(a.value)
                elif fname in ("alias", "as"):
                    # Dig into the aliased expression
                    if arg.args:
                        inner = arg.args[0]
                        if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                            cols.append(inner.value)
            elif isinstance(arg, ast.Attribute):
                # df.colName style
                cols.append(arg.attr)
        return cols

    def _extract_groupby_columns(self, call: ast.Call) -> list[str]:
        """Extract column names from groupBy()."""
        cols: list[str] = []
        for arg in call.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                cols.append(arg.value)
            elif isinstance(arg, ast.Call):
                fname = self._get_call_name(arg)
                if fname in ("col", "column"):
                    for a in arg.args:
                        if isinstance(a, ast.Constant) and isinstance(a.value, str):
                            cols.append(a.value)
        return cols

    def _extract_columns_from_sql(self, sql: str) -> list[str]:
        """Extract column names from a SQL query string."""
        cols: list[str] = []
        for match in _SQL_COL_RE.finditer(sql):
            for part in match.group(1).split(","):
                part = part.strip()
                if part == "*":
                    continue
                # Remove table qualifiers
                if "." in part:
                    part = part.split(".")[-1]
                # Remove AS alias
                if " " in part:
                    part = part.split()[0]
                if part and part.isidentifier():
                    cols.append(part)
        return cols

    @staticmethod
    def _get_call_name(call: ast.Call) -> str:
        """Get the function name from a Call node."""
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return ""

    def __repr__(self) -> str:
        return "PySparkAnalyzer()"
