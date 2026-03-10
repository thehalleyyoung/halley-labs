"""
Column-Level Lineage Analysis
==============================

Traces data lineage at the column level through SQL queries.
For each output column, determines which source columns contributed
and what type of transformation was applied.

Handles:
- Direct column references
- Computed expressions
- Aggregate functions
- Window functions
- Subqueries
- UDFs (conservative approximation)
- CTEs (including recursive)
- Set operations
"""

from __future__ import annotations

from collections import defaultdict
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

from arc.sql.operators import (
    ColumnReference,
    TransformationType,
)
from arc.sql.parser import ParsedQuery, SQLParser


# ---------------------------------------------------------------------------
# Source Column
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceColumn:
    """A source column in the lineage graph."""
    column_name: str
    table_name: Optional[str] = None
    schema_name: Optional[str] = None
    is_literal: bool = False
    is_all_columns: bool = False

    @property
    def qualified_name(self) -> str:
        parts = []
        if self.schema_name:
            parts.append(self.schema_name)
        if self.table_name:
            parts.append(self.table_name)
        parts.append(self.column_name)
        return ".".join(parts)

    def matches(self, name: str, table: Optional[str] = None) -> bool:
        if self.column_name != name:
            return False
        if table and self.table_name and self.table_name != table:
            return False
        return True

    def __repr__(self) -> str:
        return self.qualified_name


@dataclass(frozen=True)
class TransformationStep:
    """A single transformation step in a lineage chain."""
    step_type: TransformationType
    expression_sql: str = ""
    function_name: Optional[str] = None
    is_deterministic: bool = True
    is_lossy: bool = False

    def __repr__(self) -> str:
        return f"{self.step_type.value}: {self.expression_sql}"


@dataclass
class ColumnLineageEntry:
    """Lineage information for a single output column."""
    output_column: str
    source_columns: Set[SourceColumn] = field(default_factory=set)
    transformation_type: TransformationType = TransformationType.DIRECT
    transformation_steps: List[TransformationStep] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)
    is_deterministic: bool = True
    confidence: float = 1.0

    def add_source(self, source: SourceColumn) -> None:
        self.source_columns.add(source)

    def add_step(self, step: TransformationStep) -> None:
        self.transformation_steps.append(step)
        if not step.is_deterministic:
            self.is_deterministic = False

    def is_direct_mapping(self) -> bool:
        return (
            self.transformation_type == TransformationType.DIRECT
            and len(self.source_columns) == 1
        )

    def is_computed(self) -> bool:
        return self.transformation_type in (
            TransformationType.COMPUTED,
            TransformationType.AGGREGATED,
            TransformationType.WINDOWED,
        )

    def source_column_names(self) -> Set[str]:
        return {s.column_name for s in self.source_columns}

    def source_table_names(self) -> Set[str]:
        return {s.table_name for s in self.source_columns if s.table_name}


# ---------------------------------------------------------------------------
# Column Lineage (complete for a query)
# ---------------------------------------------------------------------------

@dataclass
class ColumnLineage:
    """
    Complete column-level lineage for a SQL query.
    Maps each output column to its source columns and transformations.
    """
    column_sources: Dict[str, Set[SourceColumn]] = field(default_factory=dict)
    transformation_type: Dict[str, TransformationType] = field(default_factory=dict)
    predicates_on: Dict[str, List[str]] = field(default_factory=dict)
    entries: Dict[str, ColumnLineageEntry] = field(default_factory=dict)

    # Metadata
    query_sql: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_entry(self, entry: ColumnLineageEntry) -> None:
        self.entries[entry.output_column] = entry
        self.column_sources[entry.output_column] = entry.source_columns
        self.transformation_type[entry.output_column] = entry.transformation_type
        if entry.predicates:
            self.predicates_on[entry.output_column] = entry.predicates

    def get_sources(self, output_col: str) -> Set[SourceColumn]:
        return self.column_sources.get(output_col, set())

    def get_transformation(self, output_col: str) -> TransformationType:
        return self.transformation_type.get(
            output_col, TransformationType.DIRECT
        )

    def output_columns(self) -> List[str]:
        return list(self.entries.keys())

    def all_source_columns(self) -> Set[SourceColumn]:
        result: Set[SourceColumn] = set()
        for sources in self.column_sources.values():
            result |= sources
        return result

    def all_source_tables(self) -> Set[str]:
        tables: Set[str] = set()
        for sources in self.column_sources.values():
            for s in sources:
                if s.table_name:
                    tables.add(s.table_name)
        return tables

    def direct_mappings(self) -> Dict[str, SourceColumn]:
        """Return output columns that are direct 1:1 mappings from source."""
        result: Dict[str, SourceColumn] = {}
        for col, entry in self.entries.items():
            if entry.is_direct_mapping():
                result[col] = next(iter(entry.source_columns))
        return result

    def computed_columns(self) -> List[str]:
        return [
            col for col, entry in self.entries.items()
            if entry.is_computed()
        ]

    def columns_from_table(self, table: str) -> Set[str]:
        result: Set[str] = set()
        for col, sources in self.column_sources.items():
            for s in sources:
                if s.table_name == table:
                    result.add(col)
        return result

    def is_deterministic(self) -> bool:
        return all(e.is_deterministic for e in self.entries.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            col: {
                "sources": [s.qualified_name for s in entry.source_columns],
                "transformation": entry.transformation_type.value,
                "deterministic": entry.is_deterministic,
            }
            for col, entry in self.entries.items()
        }

    def __repr__(self) -> str:
        cols = len(self.entries)
        sources = len(self.all_source_columns())
        return f"ColumnLineage({cols} outputs, {sources} sources)"


# ---------------------------------------------------------------------------
# Lineage Analyzer
# ---------------------------------------------------------------------------

class LineageAnalyzer:
    """
    Analyzes SQL queries to determine column-level lineage.
    Uses sqlglot AST analysis for precise lineage tracking.
    """

    def __init__(self, parser: Optional[SQLParser] = None) -> None:
        self._parser = parser or SQLParser()

    def analyze(self, query: Union[str, ParsedQuery]) -> ColumnLineage:
        """Analyze a query and return column-level lineage."""
        if isinstance(query, str):
            parsed = self._parser.parse(query)
        else:
            parsed = query

        lineage = ColumnLineage(query_sql=parsed.raw_sql)

        try:
            self._analyze_parsed(parsed, lineage)
        except Exception as e:
            lineage.errors.append(f"Analysis error: {e}")

        return lineage

    def trace_column(
        self, query: Union[str, ParsedQuery], output_col: str
    ) -> Set[SourceColumn]:
        """Trace a single output column back to its sources."""
        lineage = self.analyze(query)
        return lineage.get_sources(output_col)

    def analyze_many(
        self, queries: List[Union[str, ParsedQuery]]
    ) -> List[ColumnLineage]:
        """Analyze multiple queries."""
        return [self.analyze(q) for q in queries]

    # -----------------------------------------------------------------------
    # Internal Analysis
    # -----------------------------------------------------------------------

    def _analyze_parsed(
        self, parsed: ParsedQuery, lineage: ColumnLineage
    ) -> None:
        """Analyze a ParsedQuery to extract lineage."""
        if parsed.has_set_operations():
            self._analyze_set_operation(parsed, lineage)
            return

        table_schemas = self._build_table_schema_map(parsed)

        for col_ref in parsed.output_columns:
            entry = self._trace_column_ref(col_ref, parsed, table_schemas)
            lineage.add_entry(entry)

        for expr_ref in parsed.output_expressions:
            entry = self._trace_expression(expr_ref, parsed, table_schemas)
            lineage.add_entry(entry)

        if parsed.has_star:
            for table_ref in parsed.source_tables:
                entry = ColumnLineageEntry(
                    output_column=f"{table_ref.effective_name}.*",
                    source_columns={
                        SourceColumn(
                            column_name="*",
                            table_name=table_ref.effective_name,
                            is_all_columns=True,
                        )
                    },
                    transformation_type=TransformationType.DIRECT,
                    confidence=0.8,
                )
                lineage.add_entry(entry)

        if parsed.filter_predicates:
            for pred in parsed.filter_predicates:
                for col, sources in lineage.column_sources.items():
                    for source in sources:
                        if source.column_name in pred:
                            lineage.predicates_on.setdefault(col, []).append(pred)

    def _trace_column_ref(
        self,
        col_ref: ColumnReference,
        parsed: ParsedQuery,
        table_schemas: Dict[str, Set[str]],
    ) -> ColumnLineageEntry:
        """Trace a direct column reference."""
        entry = ColumnLineageEntry(
            output_column=col_ref.output_name,
            transformation_type=TransformationType.DIRECT,
        )

        source_table = col_ref.table
        if source_table:
            entry.add_source(SourceColumn(
                column_name=col_ref.name,
                table_name=source_table,
            ))
        else:
            candidates = self._find_column_table(
                col_ref.name, parsed, table_schemas
            )
            if candidates:
                for tbl in candidates:
                    entry.add_source(SourceColumn(
                        column_name=col_ref.name,
                        table_name=tbl,
                    ))
            else:
                entry.add_source(SourceColumn(column_name=col_ref.name))

        for cte in parsed.ctes:
            if source_table == cte.name:
                entry.add_step(TransformationStep(
                    step_type=TransformationType.COMPUTED,
                    expression_sql=f"CTE: {cte.name}",
                ))
                if cte.is_recursive:
                    entry.is_deterministic = True

        return entry

    def _trace_expression(
        self,
        expr_ref: ExpressionRef,
        parsed: ParsedQuery,
        table_schemas: Dict[str, Set[str]],
    ) -> ColumnLineageEntry:
        """Trace a computed expression."""
        entry = ColumnLineageEntry(
            output_column=expr_ref.output_name,
            transformation_type=expr_ref.transformation,
            is_deterministic=expr_ref.is_deterministic,
        )

        for src_col in expr_ref.source_columns:
            candidates = self._find_column_table(
                src_col, parsed, table_schemas
            )
            if candidates:
                for tbl in candidates:
                    entry.add_source(SourceColumn(
                        column_name=src_col,
                        table_name=tbl,
                    ))
            else:
                entry.add_source(SourceColumn(column_name=src_col))

        entry.add_step(TransformationStep(
            step_type=expr_ref.transformation,
            expression_sql=expr_ref.sql,
            is_deterministic=expr_ref.is_deterministic,
        ))

        if not expr_ref.source_columns:
            entry.add_source(SourceColumn(
                column_name=expr_ref.output_name,
                is_literal=True,
            ))
            entry.transformation_type = TransformationType.CONSTANT

        return entry

    def _analyze_set_operation(
        self, parsed: ParsedQuery, lineage: ColumnLineage
    ) -> None:
        """Analyze lineage through set operations (UNION, INTERSECT, EXCEPT)."""
        branch_lineages: List[ColumnLineage] = []
        for branch in parsed.set_branches:
            bl = ColumnLineage()
            self._analyze_parsed(branch, bl)
            branch_lineages.append(bl)

        if not branch_lineages:
            return

        first = branch_lineages[0]
        for col in first.output_columns():
            entry = ColumnLineageEntry(
                output_column=col,
                transformation_type=TransformationType.COMPUTED,
            )
            for bl in branch_lineages:
                sources = bl.get_sources(col)
                for src in sources:
                    entry.add_source(src)

            entry.add_step(TransformationStep(
                step_type=TransformationType.COMPUTED,
                expression_sql=f"{parsed.set_operation}",
            ))
            lineage.add_entry(entry)

    def _build_table_schema_map(
        self, parsed: ParsedQuery
    ) -> Dict[str, Set[str]]:
        """Build a map from table name/alias to known column names."""
        schema_map: Dict[str, Set[str]] = {}

        for table_ref in parsed.source_tables:
            name = table_ref.effective_name
            cols: Set[str] = set()

            for c in parsed.output_columns:
                if c.table == name or c.table == table_ref.alias:
                    cols.add(c.name)

            for jc in parsed.join_conditions:
                if jc.left.table == name:
                    cols.add(jc.left.name)
                if jc.right.table == name:
                    cols.add(jc.right.name)

            for cte in parsed.ctes:
                if cte.name == name and cte.columns:
                    cols.update(cte.columns)

            schema_map[name] = cols
            if table_ref.alias and table_ref.alias != name:
                schema_map[table_ref.alias] = cols

        return schema_map

    def _find_column_table(
        self,
        column_name: str,
        parsed: ParsedQuery,
        table_schemas: Dict[str, Set[str]],
    ) -> List[str]:
        """Find which table(s) a column might belong to."""
        candidates: List[str] = []
        for tbl_name, cols in table_schemas.items():
            if column_name in cols:
                candidates.append(tbl_name)

        if not candidates and parsed.source_tables:
            if len(parsed.source_tables) == 1:
                candidates.append(parsed.source_tables[0].effective_name)

        return candidates


# ---------------------------------------------------------------------------
# Lineage Graph (multi-query)
# ---------------------------------------------------------------------------

@dataclass
class LineageEdge:
    """An edge in the lineage graph."""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    transformation: TransformationType = TransformationType.DIRECT
    query_index: int = 0

    def __repr__(self) -> str:
        return (
            f"{self.source_table}.{self.source_column} "
            f"-[{self.transformation.value}]-> "
            f"{self.target_table}.{self.target_column}"
        )


@dataclass
class LineageGraph:
    """
    A directed graph representing data lineage across multiple queries.
    """
    edges: List[LineageEdge] = field(default_factory=list)
    tables: Set[str] = field(default_factory=set)
    columns_by_table: Dict[str, Set[str]] = field(default_factory=dict)

    def add_edge(self, edge: LineageEdge) -> None:
        self.edges.append(edge)
        self.tables.add(edge.source_table)
        self.tables.add(edge.target_table)
        self.columns_by_table.setdefault(
            edge.source_table, set()
        ).add(edge.source_column)
        self.columns_by_table.setdefault(
            edge.target_table, set()
        ).add(edge.target_column)

    def upstream_of(
        self, table: str, column: str
    ) -> List[LineageEdge]:
        """Find all edges flowing into a column."""
        return [
            e for e in self.edges
            if e.target_table == table and e.target_column == column
        ]

    def downstream_of(
        self, table: str, column: str
    ) -> List[LineageEdge]:
        """Find all edges flowing out of a column."""
        return [
            e for e in self.edges
            if e.source_table == table and e.source_column == column
        ]

    def all_upstream_tables(self, table: str) -> Set[str]:
        """Find all tables that feed into a given table (transitive)."""
        visited: Set[str] = set()
        queue = [table]
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            for e in self.edges:
                if e.target_table == current:
                    queue.append(e.source_table)
        visited.discard(table)
        return visited

    def all_downstream_tables(self, table: str) -> Set[str]:
        """Find all tables that a given table feeds into (transitive)."""
        visited: Set[str] = set()
        queue = [table]
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            for e in self.edges:
                if e.source_table == current:
                    queue.append(e.target_table)
        visited.discard(table)
        return visited

    def trace_column_upstream(
        self, table: str, column: str, max_depth: int = 10
    ) -> List[List[LineageEdge]]:
        """Trace all paths upstream from a column."""
        paths: List[List[LineageEdge]] = []
        self._trace_upstream_dfs(table, column, [], paths, max_depth)
        return paths

    def _trace_upstream_dfs(
        self,
        table: str,
        column: str,
        current_path: List[LineageEdge],
        all_paths: List[List[LineageEdge]],
        max_depth: int,
    ) -> None:
        if len(current_path) >= max_depth:
            all_paths.append(list(current_path))
            return

        incoming = self.upstream_of(table, column)
        if not incoming:
            if current_path:
                all_paths.append(list(current_path))
            return

        for edge in incoming:
            current_path.append(edge)
            self._trace_upstream_dfs(
                edge.source_table,
                edge.source_column,
                current_path,
                all_paths,
                max_depth,
            )
            current_path.pop()

    def edge_count(self) -> int:
        return len(self.edges)

    def table_count(self) -> int:
        return len(self.tables)

    def __repr__(self) -> str:
        return (
            f"LineageGraph({self.table_count()} tables, "
            f"{self.edge_count()} edges)"
        )


# ---------------------------------------------------------------------------
# Build Lineage Graph from Multiple Queries
# ---------------------------------------------------------------------------

def build_lineage_graph(
    queries: List[Tuple[str, str]],
    analyzer: Optional[LineageAnalyzer] = None,
) -> LineageGraph:
    """
    Build a lineage graph from a list of (target_table, sql) pairs.

    Each pair represents a query that populates target_table.
    """
    if analyzer is None:
        analyzer = LineageAnalyzer()

    graph = LineageGraph()

    for i, (target_table, sql) in enumerate(queries):
        lineage = analyzer.analyze(sql)

        for col, entry in lineage.entries.items():
            for source in entry.source_columns:
                src_table = source.table_name or "unknown"
                graph.add_edge(LineageEdge(
                    source_table=src_table,
                    source_column=source.column_name,
                    target_table=target_table,
                    target_column=col,
                    transformation=entry.transformation_type,
                    query_index=i,
                ))

    return graph


def trace_impact(
    graph: LineageGraph,
    changed_table: str,
    changed_column: str,
) -> Dict[str, Set[str]]:
    """
    Given a change to a column, find all downstream columns affected.

    Returns a dict mapping table names to sets of affected column names.
    """
    affected: Dict[str, Set[str]] = defaultdict(set)
    queue: List[Tuple[str, str]] = [(changed_table, changed_column)]
    visited: Set[Tuple[str, str]] = set()

    while queue:
        tbl, col = queue.pop(0)
        if (tbl, col) in visited:
            continue
        visited.add((tbl, col))

        downstream = graph.downstream_of(tbl, col)
        for edge in downstream:
            affected[edge.target_table].add(edge.target_column)
            queue.append((edge.target_table, edge.target_column))

    return dict(affected)
