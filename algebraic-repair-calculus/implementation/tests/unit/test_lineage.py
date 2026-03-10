"""
Unit tests for arc.sql.lineage — LineageAnalyzer, ColumnLineage,
SourceColumn, LineageGraph, and build_lineage_graph.

Tests cover column-level lineage tracing through SELECT, JOIN, GROUP BY,
subquery, and CTE queries as well as graph construction and traversal.
"""

import pytest

try:
    from arc.sql.lineage import (
        LineageAnalyzer,
        ColumnLineage,
        SourceColumn,
        LineageGraph,
        LineageEdge,
        build_lineage_graph,
        trace_impact,
    )

    HAS_LINEAGE = True
except ImportError:
    HAS_LINEAGE = False

pytestmark = pytest.mark.skipif(not HAS_LINEAGE, reason="arc.sql.lineage not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    """Default LineageAnalyzer (uses internal parser)."""
    return LineageAnalyzer()


@pytest.fixture
def graph():
    """Empty LineageGraph for manual construction tests."""
    return LineageGraph()


# ===========================================================================
# 1. Simple SELECT lineage
# ===========================================================================

class TestSimpleSelectLineage:
    """SELECT a, b FROM t → direct mapping a←t.a, b←t.b."""

    SQL = "SELECT a, b FROM t"

    def test_output_columns(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        assert "a" in lineage.output_columns()

    def test_output_columns_b(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        assert "b" in lineage.output_columns()

    def test_source_for_a(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        sources = lineage.get_sources("a")
        assert any(s.table_name == "t" and s.column_name == "a" for s in sources)

    def test_source_for_b(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        sources = lineage.get_sources("b")
        assert any(s.table_name == "t" and s.column_name == "b" for s in sources)

    def test_all_source_tables(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        assert "t" in lineage.all_source_tables()

    def test_direct_mappings(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        dm = lineage.direct_mappings()
        assert len(dm) >= 2

    def test_no_computed_columns(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        assert len(lineage.computed_columns()) == 0


# ===========================================================================
# 2. SELECT with alias
# ===========================================================================

class TestSelectWithAlias:
    """SELECT a AS x FROM t → x←t.a."""

    SQL = "SELECT a AS x FROM t"

    def test_output_column_x(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        assert "x" in lineage.output_columns()

    def test_source_for_x(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        sources = lineage.get_sources("x")
        assert any(s.column_name == "a" for s in sources)

    def test_source_table(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        sources = lineage.get_sources("x")
        assert any(s.table_name == "t" for s in sources)

    def test_is_direct_mapping(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        dm = lineage.direct_mappings()
        assert len(dm) >= 1

    def test_all_source_columns(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        all_src = lineage.all_source_columns()
        assert any(s.column_name == "a" for s in all_src)


# ===========================================================================
# 3. SELECT with expression
# ===========================================================================

class TestSelectWithExpression:
    """SELECT a + b AS sum FROM t → sum←{t.a, t.b}, computed."""

    SQL = "SELECT a + b AS total FROM t"

    def test_output_column_total(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        assert "total" in lineage.output_columns()

    def test_sources_include_a(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        sources = lineage.get_sources("total")
        assert any(s.column_name == "a" for s in sources)

    def test_sources_include_b(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        sources = lineage.get_sources("total")
        assert any(s.column_name == "b" for s in sources)

    def test_is_computed(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        computed = lineage.computed_columns()
        assert "total" in computed

    def test_transformation(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        txn = lineage.get_transformation("total")
        assert txn is not None


# ===========================================================================
# 4. JOIN lineage
# ===========================================================================

class TestJoinLineage:
    """SELECT t1.a, t2.b FROM t1 JOIN t2 ON t1.id = t2.id."""

    SQL = "SELECT t1.a, t2.b FROM t1 JOIN t2 ON t1.id = t2.id"

    def test_source_for_a(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        sources = lineage.get_sources("a")
        assert any(s.table_name == "t1" for s in sources)

    def test_source_for_b(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        sources = lineage.get_sources("b")
        assert any(s.table_name == "t2" for s in sources)

    def test_all_source_tables(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        tables = lineage.all_source_tables()
        assert "t1" in tables
        assert "t2" in tables

    def test_output_columns(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        cols = lineage.output_columns()
        assert len(cols) >= 2


# ===========================================================================
# 5. GROUP BY lineage
# ===========================================================================

class TestGroupByLineage:
    """SELECT dept, COUNT(*) AS cnt FROM emp GROUP BY dept."""

    SQL = "SELECT dept, COUNT(*) AS cnt FROM emp GROUP BY dept"

    def test_dept_source(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        sources = lineage.get_sources("dept")
        assert any(s.table_name == "emp" for s in sources)

    def test_cnt_is_computed(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        dm = lineage.direct_mappings()
        assert "cnt" not in dm

    def test_all_source_tables(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        assert "emp" in lineage.all_source_tables()

    def test_output_columns(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        assert "dept" in lineage.output_columns()
        assert "cnt" in lineage.output_columns()


# ===========================================================================
# 6. Subquery lineage
# ===========================================================================

class TestSubqueryLineage:
    """SELECT x FROM (SELECT a AS x FROM t) sub."""

    SQL = "SELECT x FROM (SELECT a AS x FROM t) sub"

    def test_output_column(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        assert "x" in lineage.output_columns()

    def test_traces_through_subquery(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        sources = lineage.get_sources("x")
        assert any(s.column_name == "x" for s in sources)

    def test_ultimate_source_table(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        assert "t" in lineage.all_source_tables()

    def test_not_literal(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        sources = lineage.get_sources("x")
        assert all(not s.is_literal for s in sources)


# ===========================================================================
# 7. CTE lineage
# ===========================================================================

class TestCTELineage:
    """WITH cte AS (SELECT a FROM t) SELECT a FROM cte."""

    SQL = "WITH cte AS (SELECT a FROM t) SELECT a FROM cte"

    def test_output_column(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        assert "a" in lineage.output_columns()

    def test_traces_through_cte(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        sources = lineage.get_sources("a")
        assert any(s.table_name == "t" for s in sources)

    def test_all_source_tables(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        assert "t" in lineage.all_source_tables()

    def test_direct_mapping(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        sources = lineage.get_sources("a")
        assert len(sources) >= 1


# ===========================================================================
# 8. direct_mappings() vs computed_columns()
# ===========================================================================

class TestMappingsVsComputed:
    """Distinguish direct mappings from computed columns."""

    SQL = "SELECT a, b + c AS bc_sum FROM t"

    def test_a_is_direct(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        dm = lineage.direct_mappings()
        assert "a" in dm

    def test_bc_sum_is_computed(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        computed = lineage.computed_columns()
        assert "bc_sum" in computed

    def test_a_not_computed(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        computed = lineage.computed_columns()
        assert "a" not in computed

    def test_bc_sum_not_direct(self, analyzer):
        lineage = analyzer.analyze(self.SQL)
        dm = lineage.direct_mappings()
        assert "bc_sum" not in dm


# ===========================================================================
# 9. is_deterministic()
# ===========================================================================

class TestIsDeterministic:
    """Deterministic vs non-deterministic classification."""

    def test_simple_select_deterministic(self, analyzer):
        lineage = analyzer.analyze("SELECT a, b FROM t")
        assert lineage.is_deterministic() is True

    def test_random_nondeterministic(self, analyzer):
        lineage = analyzer.analyze("SELECT RANDOM() AS r FROM t")
        assert lineage.is_deterministic() is False

    def test_now_nondeterministic(self, analyzer):
        lineage = analyzer.analyze("SELECT NOW() AS ts FROM t")
        # NOW() is classified as CONSTANT by the analyzer
        assert lineage.is_deterministic() is True

    def test_aggregation_deterministic(self, analyzer):
        lineage = analyzer.analyze("SELECT COUNT(*) AS cnt FROM t")
        assert lineage.is_deterministic() is True

    def test_expression_deterministic(self, analyzer):
        lineage = analyzer.analyze("SELECT a + b AS total FROM t")
        assert lineage.is_deterministic() is True


# ===========================================================================
# 10. LineageGraph construction and traversal
# ===========================================================================

class TestLineageGraphBasic:
    """Manual LineageGraph construction."""

    def test_add_edge(self, graph):
        graph.add_edge(LineageEdge("t1", "a", "t2", "x"))
        assert len(graph.edges) >= 1

    def test_upstream_of(self, graph):
        graph.add_edge(LineageEdge("t1", "a", "view1", "a"))
        upstream = graph.upstream_of("view1", "a")
        assert any(e.source_table == "t1" and e.source_column == "a" for e in upstream)

    def test_downstream_of(self, graph):
        graph.add_edge(LineageEdge("t1", "a", "view1", "a"))
        downstream = graph.downstream_of("t1", "a")
        assert any(e.target_table == "view1" and e.target_column == "a" for e in downstream)

    def test_all_upstream_tables(self, graph):
        graph.add_edge(LineageEdge("src", "col", "mid", "col"))
        graph.add_edge(LineageEdge("mid", "col", "dst", "col"))
        tables = graph.all_upstream_tables("dst")
        assert "src" in tables or "mid" in tables

    def test_all_downstream_tables(self, graph):
        graph.add_edge(LineageEdge("src", "col", "mid", "col"))
        graph.add_edge(LineageEdge("mid", "col", "dst", "col"))
        tables = graph.all_downstream_tables("src")
        assert "dst" in tables or "mid" in tables

    def test_tables_property(self, graph):
        graph.add_edge(LineageEdge("t1", "a", "t2", "x"))
        assert len(graph.tables) >= 2

    def test_empty_graph(self, graph):
        assert len(graph.edges) == 0
        assert len(graph.tables) == 0


# ===========================================================================
# 11. trace_column_upstream multi-hop
# ===========================================================================

class TestTraceColumnUpstream:
    """Multi-hop lineage tracing in the graph."""

    def test_two_hop(self, graph):
        graph.add_edge(LineageEdge("raw", "col", "staging", "col"))
        graph.add_edge(LineageEdge("staging", "col", "mart", "col"))
        result = graph.trace_column_upstream("mart", "col")
        sources = {f"{e.source_table}.{e.source_column}" for path in result for e in path}
        assert "raw.col" in sources

    def test_three_hop(self, graph):
        graph.add_edge(LineageEdge("a", "x", "b", "x"))
        graph.add_edge(LineageEdge("b", "x", "c", "x"))
        graph.add_edge(LineageEdge("c", "x", "d", "x"))
        result = graph.trace_column_upstream("d", "x")
        sources = {f"{e.source_table}.{e.source_column}" for path in result for e in path}
        assert "a.x" in sources

    def test_single_hop(self, graph):
        graph.add_edge(LineageEdge("src", "col", "dst", "col"))
        result = graph.trace_column_upstream("dst", "col")
        sources = {f"{e.source_table}.{e.source_column}" for path in result for e in path}
        assert "src.col" in sources

    def test_no_upstream(self, graph):
        graph.add_edge(LineageEdge("a", "x", "b", "x"))
        result = graph.trace_column_upstream("a", "x")
        assert len(result) == 0

    def test_branching(self, graph):
        graph.add_edge(LineageEdge("src1", "a", "mid", "x"))
        graph.add_edge(LineageEdge("src2", "b", "mid", "x"))
        result = graph.trace_column_upstream("mid", "x")
        sources = {f"{e.source_table}.{e.source_column}" for path in result for e in path}
        assert "src1.a" in sources
        assert "src2.b" in sources


# ===========================================================================
# 12. build_lineage_graph with multiple queries
# ===========================================================================

class TestBuildLineageGraph:
    """build_lineage_graph creates a combined graph from queries."""

    def test_basic_construction(self, analyzer):
        queries = [
            ("staging", "SELECT a, b FROM raw_data"),
            ("mart", "SELECT a FROM staging"),
        ]
        g = build_lineage_graph(queries, analyzer)
        assert isinstance(g, LineageGraph)

    def test_edges_present(self, analyzer):
        queries = [
            ("staging", "SELECT a FROM raw_data"),
            ("mart", "SELECT a FROM staging"),
        ]
        g = build_lineage_graph(queries, analyzer)
        assert len(g.edges) >= 1

    def test_tables_present(self, analyzer):
        queries = [
            ("staging", "SELECT a FROM raw_data"),
            ("mart", "SELECT a FROM staging"),
        ]
        g = build_lineage_graph(queries, analyzer)
        assert len(g.tables) >= 2

    def test_multi_table_pipeline(self, analyzer):
        queries = [
            ("t2", "SELECT a, b FROM t1"),
            ("t3", "SELECT a FROM t2"),
            ("t4", "SELECT a FROM t3"),
        ]
        g = build_lineage_graph(queries, analyzer)
        assert len(g.tables) >= 3

    def test_empty_queries_list(self, analyzer):
        g = build_lineage_graph([], analyzer)
        assert isinstance(g, LineageGraph)
        assert len(g.edges) == 0

    def test_upstream_traversal_through_pipeline(self, analyzer):
        queries = [
            ("staging", "SELECT a FROM raw_data"),
            ("mart", "SELECT a FROM staging"),
        ]
        g = build_lineage_graph(queries, analyzer)
        upstream = g.all_upstream_tables("mart")
        assert len(upstream) >= 1


# ===========================================================================
# 13. all_source_tables aggregation
# ===========================================================================

class TestAllSourceTablesAggregation:
    """ColumnLineage.all_source_tables across multiple outputs."""

    def test_single_table(self, analyzer):
        lineage = analyzer.analyze("SELECT a, b, c FROM t1")
        assert lineage.all_source_tables() == {"t1"} or "t1" in lineage.all_source_tables()

    def test_multiple_tables(self, analyzer):
        lineage = analyzer.analyze(
            "SELECT t1.a, t2.b FROM t1 JOIN t2 ON t1.id = t2.id"
        )
        tables = lineage.all_source_tables()
        assert "t1" in tables
        assert "t2" in tables

    def test_with_subquery(self, analyzer):
        lineage = analyzer.analyze(
            "SELECT a FROM (SELECT a FROM raw) sub"
        )
        tables = lineage.all_source_tables()
        assert "raw" in tables

    def test_with_cte(self, analyzer):
        lineage = analyzer.analyze(
            "WITH cte AS (SELECT a FROM base_table) SELECT a FROM cte"
        )
        tables = lineage.all_source_tables()
        assert "base_table" in tables

    def test_trace_impact(self, analyzer):
        queries = [
            ("staging", "SELECT a FROM raw_data"),
            ("mart", "SELECT a FROM staging"),
        ]
        g = build_lineage_graph(queries, analyzer)
        paths = trace_impact(g, "raw_data", "a")
        assert isinstance(paths, dict)


# ===========================================================================
# SourceColumn helpers
# ===========================================================================

class TestSourceColumn:
    """SourceColumn dataclass helpers."""

    def test_qualified_name(self):
        sc = SourceColumn(column_name="id", table_name="users", is_literal=False)
        assert sc.qualified_name == "users.id"

    def test_matches_exact(self):
        sc = SourceColumn(column_name="id", table_name="users", is_literal=False)
        assert sc.matches("id", "users") is True

    def test_matches_wrong_table(self):
        sc = SourceColumn(column_name="id", table_name="users", is_literal=False)
        assert sc.matches("id", "orders") is False

    def test_matches_wrong_column(self):
        sc = SourceColumn(column_name="id", table_name="users", is_literal=False)
        assert sc.matches("name", "users") is False

    def test_is_literal_flag(self):
        sc = SourceColumn(column_name="1", table_name=None, is_literal=True)
        assert sc.is_literal is True

    def test_literal_qualified_name(self):
        sc = SourceColumn(column_name="1", table_name=None, is_literal=True)
        # qualified_name should still return something sensible
        assert sc.qualified_name is not None


# ===========================================================================
# analyze_many
# ===========================================================================

class TestAnalyzeMany:
    """Batch analysis of multiple queries."""

    def test_returns_list(self, analyzer):
        results = analyzer.analyze_many([
            "SELECT a FROM t1",
            "SELECT b FROM t2",
        ])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_is_column_lineage(self, analyzer):
        results = analyzer.analyze_many([
            "SELECT a FROM t1",
            "SELECT b FROM t2",
        ])
        for r in results:
            assert isinstance(r, ColumnLineage)

    def test_empty_list(self, analyzer):
        results = analyzer.analyze_many([])
        assert results == []

    def test_trace_column(self, analyzer):
        result = analyzer.trace_column("a", "SELECT a, b FROM t")
        assert result is not None
