"""
Pandas-specific lineage extraction via AST analysis.

:class:`PandasAnalyzer` walks the AST of pandas-based ETL code and
extracts:

* **Source nodes** – ``read_csv``, ``read_sql``, ``read_parquet``, etc.
* **Join operations** – ``merge``, ``join``.
* **Group-by** – ``groupby().agg()``.
* **Filter** – ``query()``, boolean indexing.
* **Schema operations** – ``assign``, ``drop``, ``rename``.
* **Sink nodes** – ``to_csv``, ``to_sql``, ``to_parquet``.
* **Lineage chains** for each DataFrame variable.
"""

from __future__ import annotations

import ast
import logging
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
from arc.types.base import (
    DataflowEdge,
    DataflowGraph,
    DataflowNode,
    ETLFramework,
    LineageChain,
    LineageStep,
    PandasLineage,
    QualityPattern,
    QualityPatternType,
    Transformation,
    TransformationType,
)

logger = logging.getLogger(__name__)

# Mapping from pandas method names to transformation types
_PANDAS_METHOD_MAP: dict[str, TransformationType] = {
    # Sources
    "read_csv": TransformationType.SOURCE,
    "read_excel": TransformationType.SOURCE,
    "read_sql": TransformationType.SOURCE,
    "read_sql_query": TransformationType.SOURCE,
    "read_sql_table": TransformationType.SOURCE,
    "read_parquet": TransformationType.SOURCE,
    "read_json": TransformationType.SOURCE,
    "read_feather": TransformationType.SOURCE,
    "read_hdf": TransformationType.SOURCE,
    "read_pickle": TransformationType.SOURCE,
    "read_table": TransformationType.SOURCE,
    "read_fwf": TransformationType.SOURCE,
    "read_clipboard": TransformationType.SOURCE,
    "DataFrame": TransformationType.SOURCE,
    # Joins
    "merge": TransformationType.JOIN,
    "join": TransformationType.JOIN,
    "concat": TransformationType.UNION,
    "append": TransformationType.UNION,
    # Grouping
    "groupby": TransformationType.GROUP_BY,
    "pivot_table": TransformationType.PIVOT,
    "pivot": TransformationType.PIVOT,
    "melt": TransformationType.UNPIVOT,
    "stack": TransformationType.UNPIVOT,
    "unstack": TransformationType.PIVOT,
    # Filtering
    "query": TransformationType.FILTER,
    "filter": TransformationType.FILTER,
    "where": TransformationType.FILTER,
    "mask": TransformationType.FILTER,
    "dropna": TransformationType.DROPNA,
    "drop_duplicates": TransformationType.FILTER,
    "head": TransformationType.FILTER,
    "tail": TransformationType.FILTER,
    "sample": TransformationType.FILTER,
    "nlargest": TransformationType.FILTER,
    "nsmallest": TransformationType.FILTER,
    # Schema operations
    "assign": TransformationType.ASSIGN,
    "drop": TransformationType.DROP,
    "rename": TransformationType.RENAME,
    "reindex": TransformationType.RENAME,
    "set_index": TransformationType.RENAME,
    "reset_index": TransformationType.RENAME,
    "astype": TransformationType.CAST,
    "fillna": TransformationType.FILLNA,
    # Selection
    "select_dtypes": TransformationType.SELECT,
    # Sorting
    "sort_values": TransformationType.SORT,
    "sort_index": TransformationType.SORT,
    # Aggregation (often chained after groupby)
    "agg": TransformationType.GROUP_BY,
    "aggregate": TransformationType.GROUP_BY,
    "sum": TransformationType.GROUP_BY,
    "mean": TransformationType.GROUP_BY,
    "count": TransformationType.GROUP_BY,
    "min": TransformationType.GROUP_BY,
    "max": TransformationType.GROUP_BY,
    "std": TransformationType.GROUP_BY,
    "var": TransformationType.GROUP_BY,
    "median": TransformationType.GROUP_BY,
    # Sinks
    "to_csv": TransformationType.SINK,
    "to_sql": TransformationType.SINK,
    "to_parquet": TransformationType.SINK,
    "to_json": TransformationType.SINK,
    "to_excel": TransformationType.SINK,
    "to_feather": TransformationType.SINK,
    "to_hdf": TransformationType.SINK,
    "to_pickle": TransformationType.SINK,
    "to_clipboard": TransformationType.SINK,
    # Window
    "rolling": TransformationType.WINDOW,
    "expanding": TransformationType.WINDOW,
    "ewm": TransformationType.WINDOW,
    # Apply (custom)
    "apply": TransformationType.CUSTOM,
    "applymap": TransformationType.CUSTOM,
    "map": TransformationType.CUSTOM,
    "transform": TransformationType.CUSTOM,
    "pipe": TransformationType.CUSTOM,
}


class PandasAnalyzer:
    """Extract lineage from pandas code using AST analysis.

    Walks the AST looking for pandas DataFrame operations, builds
    lineage chains and a dataflow graph.
    """

    def __init__(self) -> None:
        self._source: str = ""
        self._tree: ast.Module | None = None
        self._lines: list[str] = []
        self._assignments: dict[str, list[int]] = {}
        self._df_ops: dict[str, list[Transformation]] = {}

    # ── Public API ─────────────────────────────────────────────────────

    def analyze(self, source: str) -> PandasLineage:
        """Analyze pandas source code and extract lineage.

        Parameters
        ----------
        source:
            Python source code.

        Returns
        -------
        PandasLineage
        """
        self._source = source
        self._lines = source.splitlines()
        self._tree = safe_parse(source)
        if self._tree is None:
            return PandasLineage()

        self._assignments = track_assignments(self._tree)
        self._df_ops = {}

        transformations: list[Transformation] = []
        sources: list[str] = []
        sinks: list[str] = []

        # Walk the AST for assignments
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        t = self._analyze_assignment(target.id, node.value, node.lineno)
                        if t is not None:
                            transformations.append(t)
                            self._df_ops.setdefault(target.id, []).append(t)
                            if t.transform_type == TransformationType.SOURCE:
                                sources.append(target.id)
                            elif t.transform_type == TransformationType.SINK:
                                sinks.append(target.id)

            # Handle expression statements (e.g. df.to_csv(...))
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                t = self._analyze_expression(node.value, node.lineno)
                if t is not None:
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

        # Build dataflow graph
        graph = self._build_graph(transformations)

        return PandasLineage(
            chains=chains,
            sources=tuple(sources),
            sinks=tuple(sinks),
            transformations=tuple(transformations),
            dataflow_graph=graph,
        )

    def trace_dataframe_lineage(
        self,
        source: str,
        df_name: str,
    ) -> LineageChain:
        """Trace lineage for a specific DataFrame variable.

        Parameters
        ----------
        source:
            Python source code.
        df_name:
            The DataFrame variable name to trace.

        Returns
        -------
        LineageChain
        """
        lineage = self.analyze(source)
        return lineage.chains.get(df_name, LineageChain(variable_name=df_name))

    def extract_schema_from_operations(
        self,
        ops: list[Transformation],
    ) -> list[str]:
        """Extract column names referenced across operations."""
        columns: set[str] = set()
        for op in ops:
            columns.update(op.columns_read)
            columns.update(op.columns_written)
        return sorted(columns)

    def detect_quality_patterns(self, source: str) -> list[QualityPattern]:
        """Detect quality-related patterns in pandas code."""
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

            if method in ("isnull", "isna", "notnull", "notna"):
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.NULL_CHECK,
                    source_line=node.lineno,
                    source_text=src_line,
                    description=f"Null check: .{method}()",
                ))

            elif method == "dropna":
                cols = self._extract_dropna_columns(node)
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.NULL_CHECK,
                    source_line=node.lineno,
                    source_text=src_line,
                    columns=tuple(cols),
                    description="Drop nulls",
                ))

            elif method == "fillna":
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.NULL_CHECK,
                    source_line=node.lineno,
                    source_text=src_line,
                    description="Fill nulls",
                ))

            elif method == "astype":
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.TYPE_CAST,
                    source_line=node.lineno,
                    source_text=src_line,
                    description="Type cast",
                ))

            elif method == "drop_duplicates":
                cols = extract_string_args(node)
                subset_kw = extract_keyword_arg(node, "subset")
                if subset_kw is not None:
                    cols = extract_string_list_arg(subset_kw)
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.DEDUPLICATION,
                    source_line=node.lineno,
                    source_text=src_line,
                    columns=tuple(cols),
                    description="Deduplication",
                ))

            elif method == "between":
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.RANGE_VALIDATION,
                    source_line=node.lineno,
                    source_text=src_line,
                    description="Range validation: .between()",
                ))

            elif method in ("match", "contains", "fullmatch"):
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.REGEX_VALIDATION,
                    source_line=node.lineno,
                    source_text=src_line,
                    description=f"Regex validation: .str.{method}()",
                ))

        return patterns

    # ── Private methods ────────────────────────────────────────────────

    def _analyze_assignment(
        self,
        target_name: str,
        value: ast.expr,
        lineno: int,
    ) -> Transformation | None:
        """Analyze a single assignment: target = value."""
        if isinstance(value, ast.Call):
            return self._analyze_call(target_name, value, lineno)
        elif isinstance(value, ast.Subscript):
            return self._analyze_subscript(target_name, value, lineno)
        return None

    def _analyze_call(
        self,
        target_name: str,
        call: ast.Call,
        lineno: int,
    ) -> Transformation | None:
        """Analyze a function/method call."""
        method_name = ""
        input_vars: list[str] = []

        if isinstance(call.func, ast.Attribute):
            method_name = call.func.attr
            root = get_root_variable(call.func.value)
            if root and root in self._assignments:
                input_vars.append(root)
        elif isinstance(call.func, ast.Name):
            method_name = call.func.id

        if not method_name:
            return None

        tt = _PANDAS_METHOD_MAP.get(method_name)
        if tt is None:
            return None

        # Extract columns referenced
        cols_read = self._extract_columns_from_call(call, method_name)
        cols_written: list[str] = []

        # For merge/join, also get the "right" DataFrame
        if method_name in ("merge", "join"):
            for arg in call.args:
                if isinstance(arg, ast.Name) and arg.id in self._assignments:
                    if arg.id not in input_vars:
                        input_vars.append(arg.id)
            right_kw = extract_keyword_arg(call, "right")
            if right_kw is not None and isinstance(right_kw, ast.Name):
                if right_kw.id not in input_vars:
                    input_vars.append(right_kw.id)
            # Extract on/left_on/right_on
            on_kw = extract_keyword_arg(call, "on")
            if on_kw:
                cols_read.extend(extract_string_list_arg(on_kw))
            left_on = extract_keyword_arg(call, "left_on")
            if left_on:
                cols_read.extend(extract_string_list_arg(left_on))
            right_on = extract_keyword_arg(call, "right_on")
            if right_on:
                cols_read.extend(extract_string_list_arg(right_on))

        # For assign, get new column names
        if method_name == "assign":
            for kw in call.keywords:
                if kw.arg is not None:
                    cols_written.append(kw.arg)

        # For rename, get the mapping
        if method_name == "rename":
            columns_kw = extract_keyword_arg(call, "columns")
            if columns_kw and isinstance(columns_kw, ast.Dict):
                for key in columns_kw.keys:
                    if key is not None:
                        val = self._extract_string(key)
                        if val:
                            cols_read.append(val)
                for val_node in columns_kw.values:
                    val = self._extract_string(val_node)
                    if val:
                        cols_written.append(val)

        # For drop, get dropped columns
        if method_name == "drop":
            cols_read.extend(extract_string_args(call))
            columns_kw = extract_keyword_arg(call, "columns")
            if columns_kw:
                cols_read.extend(extract_string_list_arg(columns_kw))

        # For groupby, get group columns
        if method_name == "groupby":
            cols_read.extend(extract_string_args(call))
            by_kw = extract_keyword_arg(call, "by")
            if by_kw:
                cols_read.extend(extract_string_list_arg(by_kw))

        # For sort_values, get sort columns
        if method_name == "sort_values":
            cols_read.extend(extract_string_args(call))
            by_kw = extract_keyword_arg(call, "by")
            if by_kw:
                cols_read.extend(extract_string_list_arg(by_kw))

        src_line = self._lines[lineno - 1].strip() if lineno <= len(self._lines) else ""

        return Transformation(
            transform_type=tt,
            input_vars=tuple(input_vars),
            output_var=target_name,
            columns_read=tuple(dict.fromkeys(cols_read)),
            columns_written=tuple(dict.fromkeys(cols_written)),
            source_line=lineno,
            source_text=src_line,
            parameters={"method": method_name},
        )

    def _analyze_expression(
        self,
        call: ast.Call,
        lineno: int,
    ) -> Transformation | None:
        """Analyze a standalone expression (typically a sink)."""
        if not isinstance(call.func, ast.Attribute):
            return None

        method = call.func.attr
        tt = _PANDAS_METHOD_MAP.get(method)
        if tt is None:
            return None

        root = get_root_variable(call.func.value)
        input_vars = [root] if root else []

        src_line = self._lines[lineno - 1].strip() if lineno <= len(self._lines) else ""

        return Transformation(
            transform_type=tt,
            input_vars=tuple(input_vars),
            output_var="",
            source_line=lineno,
            source_text=src_line,
            parameters={"method": method},
        )

    def _analyze_subscript(
        self,
        target_name: str,
        subscript: ast.Subscript,
        lineno: int,
    ) -> Transformation | None:
        """Analyze DataFrame column selection: df["col"] or df[["col1","col2"]]."""
        root = get_root_variable(subscript.value)
        if root is None or root not in self._assignments:
            return None

        cols: list[str] = []
        slice_node = subscript.slice
        if isinstance(slice_node, ast.List):
            for elt in slice_node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    cols.append(elt.value)
        elif isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            cols.append(slice_node.value)
        elif isinstance(slice_node, ast.Index):
            # Python 3.8 compat
            idx_val = slice_node.value  # type: ignore[attr-defined]
            if isinstance(idx_val, ast.Constant) and isinstance(idx_val.value, str):
                cols.append(idx_val.value)

        if not cols:
            return None

        src_line = self._lines[lineno - 1].strip() if lineno <= len(self._lines) else ""

        return Transformation(
            transform_type=TransformationType.SELECT,
            input_vars=(root,),
            output_var=target_name,
            columns_read=tuple(cols),
            source_line=lineno,
            source_text=src_line,
            parameters={"method": "__getitem__"},
        )

    def _build_graph(
        self,
        transformations: list[Transformation],
    ) -> DataflowGraph:
        """Build a DataflowGraph from transformations."""
        graph = DataflowGraph(framework=ETLFramework.PANDAS)
        var_to_node: dict[str, str] = {}

        for i, t in enumerate(transformations):
            nid = f"pd_{i}_{t.transform_type.value.lower()}"
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

    def _extract_columns_from_call(
        self,
        call: ast.Call,
        method_name: str,
    ) -> list[str]:
        """Extract column references from a call's arguments."""
        cols: list[str] = []
        cols.extend(extract_string_args(call))
        return cols

    def _extract_dropna_columns(self, call: ast.Call) -> list[str]:
        """Extract columns from a .dropna() call."""
        subset = extract_keyword_arg(call, "subset")
        if subset is not None:
            return extract_string_list_arg(subset)
        return []

    @staticmethod
    def _extract_string(node: ast.expr) -> str:
        """Extract a string constant from an AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return ""

    def __repr__(self) -> str:
        return "PandasAnalyzer()"
