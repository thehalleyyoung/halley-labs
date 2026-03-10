"""
Top-level Python ETL analyzer.

:class:`PythonETLAnalyzer` detects the ETL framework used in a Python
file (pandas, PySpark, dbt) and dispatches to the appropriate
specialised analyzer to extract pipeline structure, lineage, and
quality patterns.
"""

from __future__ import annotations

import logging
from typing import Any

from arc.python_etl.ast_utils import (
    extract_imports,
    safe_parse,
    safe_parse_file,
    track_assignments,
)
from arc.types.base import (
    DataflowEdge,
    DataflowGraph,
    DataflowNode,
    ETLAnalysisResult,
    ETLFramework,
    QualityPattern,
    QualityPatternType,
    Transformation,
    TransformationType,
)

logger = logging.getLogger(__name__)


# ── Framework detection heuristics ─────────────────────────────────────

_PANDAS_INDICATORS = frozenset({
    "pandas",
    "pd.read_csv",
    "pd.read_sql",
    "pd.read_parquet",
    "pd.DataFrame",
    "pd.merge",
    "pd.concat",
})

_PYSPARK_INDICATORS = frozenset({
    "pyspark",
    "pyspark.sql",
    "SparkSession",
    "spark.read",
    "spark.sql",
})

_DBT_INDICATORS = frozenset({
    "dbt",
    "ref(",
    "source(",
    "config(",
})


class PythonETLAnalyzer:
    """Analyze Python ETL code to extract pipeline structure.

    Detects the framework (pandas, PySpark, dbt), then delegates to
    the specialised analyzer.

    Parameters
    ----------
    detect_quality:
        Whether to scan for quality-related patterns.
    """

    def __init__(self, detect_quality: bool = True) -> None:
        self.detect_quality = detect_quality

    # ── Public API ─────────────────────────────────────────────────────

    def analyze_file(self, filepath: str) -> ETLAnalysisResult:
        """Analyze a Python ETL file.

        Parameters
        ----------
        filepath:
            Path to the Python file.

        Returns
        -------
        ETLAnalysisResult
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
        except (OSError, IOError) as exc:
            return ETLAnalysisResult(
                source_file=filepath,
                errors=(f"Cannot read file: {exc}",),
            )

        result = self.analyze_source(source)
        return ETLAnalysisResult(
            source_file=filepath,
            framework=result.framework,
            transformations=result.transformations,
            dataflow_graph=result.dataflow_graph,
            lineage=result.lineage,
            quality_patterns=result.quality_patterns,
            errors=result.errors,
            metadata=result.metadata,
        )

    def analyze_source(self, source: str) -> ETLAnalysisResult:
        """Analyze Python ETL source code.

        Parameters
        ----------
        source:
            Python source code string.

        Returns
        -------
        ETLAnalysisResult
        """
        tree = safe_parse(source)
        if tree is None:
            return ETLAnalysisResult(errors=("Failed to parse source",))

        imports = extract_imports(tree)
        framework = self.detect_framework(source, imports)

        errors: list[str] = []
        transformations: list[Transformation] = []
        dataflow_graph: DataflowGraph | None = None
        lineage = None
        quality_patterns: list[QualityPattern] = []

        try:
            if framework == ETLFramework.PANDAS:
                from arc.python_etl.pandas_analyzer import PandasAnalyzer
                pa = PandasAnalyzer()
                result = pa.analyze(source)
                transformations = list(result.transformations)
                dataflow_graph = result.dataflow_graph
                lineage = result
            elif framework == ETLFramework.PYSPARK:
                from arc.python_etl.pyspark_analyzer import PySparkAnalyzer
                psa = PySparkAnalyzer()
                result = psa.analyze(source)
                transformations = list(result.transformations)
                dataflow_graph = result.dataflow_graph
            else:
                transformations = self._extract_generic_transformations(tree, source)
                dataflow_graph = self.build_dataflow_graph(transformations)

        except Exception as exc:
            errors.append(f"Analysis error: {exc}")
            logger.exception("ETL analysis failed")

        if self.detect_quality:
            try:
                quality_patterns = self._detect_quality_patterns(tree, source)
            except Exception as exc:
                errors.append(f"Quality pattern detection error: {exc}")

        return ETLAnalysisResult(
            framework=framework,
            transformations=tuple(transformations),
            dataflow_graph=dataflow_graph,
            lineage=lineage,
            quality_patterns=tuple(quality_patterns),
            errors=tuple(errors),
            metadata={"imports": imports},
        )

    def detect_framework(
        self,
        source: str,
        imports: dict[str, str] | None = None,
    ) -> ETLFramework:
        """Detect the ETL framework used in the source code.

        Parameters
        ----------
        source:
            Python source code.
        imports:
            Pre-extracted import mappings (optional).

        Returns
        -------
        ETLFramework
        """
        if imports is None:
            tree = safe_parse(source)
            imports = extract_imports(tree) if tree else {}

        all_modules = set(imports.values())
        all_names = set(imports.keys())
        combined = all_modules | all_names

        # Check for PySpark
        pyspark_score = sum(1 for ind in _PYSPARK_INDICATORS if any(ind in s for s in combined))
        if pyspark_score >= 1 or "pyspark" in source:
            return ETLFramework.PYSPARK

        # Check for pandas
        pandas_score = sum(1 for ind in _PANDAS_INDICATORS if any(ind in s for s in combined))
        if pandas_score >= 1 or "pandas" in source or "import pandas" in source:
            return ETLFramework.PANDAS

        # Check for dbt
        dbt_score = sum(1 for ind in _DBT_INDICATORS if ind in source)
        if dbt_score >= 2:
            return ETLFramework.DBT

        # Check for SQLAlchemy
        if "sqlalchemy" in source or "create_engine" in source:
            return ETLFramework.SQLALCHEMY

        return ETLFramework.UNKNOWN

    def extract_transformations(
        self,
        source: str,
    ) -> list[Transformation]:
        """Extract transformations from source (convenience wrapper)."""
        result = self.analyze_source(source)
        return list(result.transformations)

    def build_dataflow_graph(
        self,
        transformations: list[Transformation],
    ) -> DataflowGraph:
        """Build a dataflow graph from a list of transformations.

        Each transformation becomes a node.  Edges are inferred from
        the input/output variable names.
        """
        graph = DataflowGraph()
        var_to_node: dict[str, str] = {}

        for i, t in enumerate(transformations):
            nid = f"t_{i}_{t.transform_type.value.lower()}"
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

    # ── Private methods ────────────────────────────────────────────────

    def _extract_generic_transformations(
        self,
        tree: Any,
        source: str,
    ) -> list[Transformation]:
        """Extract transformations from generic Python code (no framework)."""
        import ast as ast_mod
        transformations: list[Transformation] = []
        assignments = track_assignments(tree)

        for node in ast_mod.walk(tree):
            if isinstance(node, ast_mod.Assign) and isinstance(node.value, ast_mod.Call):
                call = node.value
                names = []
                for target in node.targets:
                    if isinstance(target, ast_mod.Name):
                        names.append(target.id)

                func_name = ""
                if isinstance(call.func, ast_mod.Attribute):
                    func_name = call.func.attr
                elif isinstance(call.func, ast_mod.Name):
                    func_name = call.func.id

                if not func_name:
                    continue

                # Infer transformation type from function name
                tt = TransformationType.CUSTOM
                if "read" in func_name.lower() or "load" in func_name.lower() or "open" in func_name.lower():
                    tt = TransformationType.SOURCE
                elif "write" in func_name.lower() or "save" in func_name.lower() or "to_" in func_name.lower():
                    tt = TransformationType.SINK
                elif "filter" in func_name.lower() or "where" in func_name.lower() or "query" in func_name.lower():
                    tt = TransformationType.FILTER
                elif "merge" in func_name.lower() or "join" in func_name.lower():
                    tt = TransformationType.JOIN
                elif "group" in func_name.lower() or "agg" in func_name.lower():
                    tt = TransformationType.GROUP_BY
                elif "sort" in func_name.lower() or "order" in func_name.lower():
                    tt = TransformationType.SORT
                elif "rename" in func_name.lower():
                    tt = TransformationType.RENAME
                elif "drop" in func_name.lower():
                    tt = TransformationType.DROP
                elif "select" in func_name.lower():
                    tt = TransformationType.SELECT

                # Get source line
                src_line = ""
                lines = source.splitlines()
                if 1 <= node.lineno <= len(lines):
                    src_line = lines[node.lineno - 1].strip()

                transformations.append(Transformation(
                    transform_type=tt,
                    input_vars=tuple(
                        n for n in _get_call_variable_refs(call) if n in assignments
                    ),
                    output_var=names[0] if names else "",
                    source_line=node.lineno,
                    source_text=src_line,
                    parameters={"function": func_name},
                ))

        return transformations

    def _detect_quality_patterns(
        self,
        tree: Any,
        source: str,
    ) -> list[QualityPattern]:
        """Detect quality-related patterns in the source code."""
        import ast as ast_mod
        patterns: list[QualityPattern] = []
        lines = source.splitlines()

        for node in ast_mod.walk(tree):
            # Null checks: isnull(), isna(), notnull(), notna(), dropna(), fillna()
            if isinstance(node, ast_mod.Call) and isinstance(node.func, ast_mod.Attribute):
                method = node.func.attr

                if method in ("isnull", "isna", "notnull", "notna"):
                    src_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                    patterns.append(QualityPattern(
                        pattern_type=QualityPatternType.NULL_CHECK,
                        source_line=node.lineno,
                        source_text=src_line,
                        description=f"Null check using .{method}()",
                    ))

                elif method == "dropna":
                    src_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                    patterns.append(QualityPattern(
                        pattern_type=QualityPatternType.NULL_CHECK,
                        source_line=node.lineno,
                        source_text=src_line,
                        description="Null removal using .dropna()",
                    ))

                elif method == "fillna":
                    src_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                    patterns.append(QualityPattern(
                        pattern_type=QualityPatternType.NULL_CHECK,
                        source_line=node.lineno,
                        source_text=src_line,
                        description="Null filling using .fillna()",
                    ))

                elif method == "astype":
                    src_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                    patterns.append(QualityPattern(
                        pattern_type=QualityPatternType.TYPE_CAST,
                        source_line=node.lineno,
                        source_text=src_line,
                        description="Type casting using .astype()",
                    ))

                elif method == "drop_duplicates":
                    src_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                    patterns.append(QualityPattern(
                        pattern_type=QualityPatternType.DEDUPLICATION,
                        source_line=node.lineno,
                        source_text=src_line,
                        description="Deduplication using .drop_duplicates()",
                    ))

            # Assert statements
            if isinstance(node, ast_mod.Assert):
                src_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.ASSERTION,
                    source_line=node.lineno,
                    source_text=src_line,
                    description="Assertion check",
                ))

            # Try/except for error handling
            if isinstance(node, ast_mod.Try):
                src_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                patterns.append(QualityPattern(
                    pattern_type=QualityPatternType.ERROR_HANDLING,
                    source_line=node.lineno,
                    source_text=src_line,
                    description="Error handling with try/except",
                ))

            # Logging calls
            if isinstance(node, ast_mod.Call) and isinstance(node.func, ast_mod.Attribute):
                if node.func.attr in ("info", "warning", "error", "debug", "critical"):
                    root = _get_root_name(node.func.value)
                    if root in ("logger", "logging", "log"):
                        src_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                        patterns.append(QualityPattern(
                            pattern_type=QualityPatternType.LOGGING,
                            source_line=node.lineno,
                            source_text=src_line,
                            description=f"Logging call: {node.func.attr}",
                        ))

        return patterns

    def __repr__(self) -> str:
        return f"PythonETLAnalyzer(detect_quality={self.detect_quality})"


# ── Private helpers ────────────────────────────────────────────────────

def _get_call_variable_refs(call: Any) -> list[str]:
    """Extract variable references from a function call."""
    import ast as ast_mod
    refs: list[str] = []
    for node in ast_mod.walk(call):
        if isinstance(node, ast_mod.Name):
            refs.append(node.id)
    return refs


def _get_root_name(node: Any) -> str:
    """Get the root name from an attribute chain."""
    import ast as ast_mod
    if isinstance(node, ast_mod.Name):
        return node.id
    if isinstance(node, ast_mod.Attribute):
        return _get_root_name(node.value)
    return ""
