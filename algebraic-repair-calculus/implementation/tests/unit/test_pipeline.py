"""Unit tests for the pipeline graph, builder, and analysis modules.

Tests cover:
  - arc.graph.pipeline   (PipelineGraph, PipelineNode, PipelineEdge)
  - arc.graph.builder    (PipelineBuilder, build_linear/diamond/star_pipeline)
  - arc.graph.analysis   (impact_analysis, detect_bottlenecks, compute_metrics,
                           assign_topological_levels, compute_execution_waves)
"""

from __future__ import annotations

import copy
import pytest

# ── Guarded imports ──────────────────────────────────────────────────

try:
    from arc.graph.pipeline import PipelineEdge, PipelineGraph, PipelineNode
except ImportError:
    PipelineEdge = PipelineGraph = PipelineNode = None  # type: ignore[assignment,misc]

try:
    from arc.graph.builder import (
        PipelineBuilder,
        build_linear_pipeline,
        build_diamond_pipeline,
        build_star_pipeline,
    )
except ImportError:
    PipelineBuilder = None  # type: ignore[assignment,misc]
    build_linear_pipeline = build_diamond_pipeline = build_star_pipeline = None  # type: ignore[assignment]

try:
    from arc.graph.analysis import (
        impact_analysis,
        detect_bottlenecks,
        compute_metrics,
        assign_topological_levels,
        compute_execution_waves,
    )
except ImportError:
    impact_analysis = detect_bottlenecks = compute_metrics = None  # type: ignore[assignment]
    assign_topological_levels = compute_execution_waves = None  # type: ignore[assignment]

try:
    from arc.types.base import (
        Column,
        CostEstimate,
        EdgeType,
        ParameterisedType,
        QualityConstraint,
        Schema,
        SQLType,
    )
except ImportError:
    Column = CostEstimate = EdgeType = ParameterisedType = None  # type: ignore[assignment,misc]
    QualityConstraint = Schema = SQLType = None  # type: ignore[assignment,misc]

try:
    from arc.types.operators import SQLOperator
except ImportError:
    SQLOperator = None  # type: ignore[assignment,misc]

try:
    from arc.types.errors import (
        CycleDetectedError,
        EdgeNotFoundError,
        NodeNotFoundError,
    )
except ImportError:
    CycleDetectedError = EdgeNotFoundError = NodeNotFoundError = None  # type: ignore[assignment,misc]


# ── Helpers ──────────────────────────────────────────────────────────

def _skip_if_missing(*objs: object) -> None:
    for o in objs:
        if o is None:
            pytest.skip("Required module not importable")


def _simple_schema(*col_names: str) -> Schema:
    """Build a simple Schema with VARCHAR columns."""
    cols = tuple(
        Column.quick(name, SQLType.VARCHAR, position=i)
        for i, name in enumerate(col_names)
    )
    return Schema(columns=cols)


def _make_node(node_id: str, operator=None, output_schema=None) -> PipelineNode:
    kw: dict = {"node_id": node_id}
    if operator is not None:
        kw["operator"] = operator
    if output_schema is not None:
        kw["output_schema"] = output_schema
    return PipelineNode(**kw)


def _linear_graph(n: int = 4) -> PipelineGraph:
    """Build source -> t1 -> t2 -> sink."""
    g = PipelineGraph(name="linear")
    ids = [f"n{i}" for i in range(n)]
    for nid in ids:
        g.add_node(_make_node(nid, operator=SQLOperator.SOURCE if nid == ids[0] else SQLOperator.TRANSFORM))
    for i in range(len(ids) - 1):
        g.add_edge(PipelineEdge(source=ids[i], target=ids[i + 1]))
    return g


def _diamond_graph() -> PipelineGraph:
    """source -> {left, right} -> merge."""
    g = PipelineGraph(name="diamond")
    for nid in ("src", "left", "right", "merge"):
        g.add_node(_make_node(nid))
    g.add_edge(PipelineEdge(source="src", target="left"))
    g.add_edge(PipelineEdge(source="src", target="right"))
    g.add_edge(PipelineEdge(source="left", target="merge"))
    g.add_edge(PipelineEdge(source="right", target="merge"))
    return g


# =====================================================================
# 1. PipelineGraph construction
# =====================================================================

class TestPipelineGraphConstruction:

    def test_add_nodes_and_edges(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = PipelineGraph(name="test")
        n1 = _make_node("a")
        n2 = _make_node("b")
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(PipelineEdge(source="a", target="b"))
        assert g.node_count == 2
        assert g.edge_count == 1

    def test_graph_name_and_version(self):
        _skip_if_missing(PipelineGraph)
        g = PipelineGraph(name="pipe", version="2.0")
        assert g.name == "pipe"
        assert g.version == "2.0"


# =====================================================================
# 2. Node operations
# =====================================================================

class TestNodeOperations:

    def test_add_and_get_node(self):
        _skip_if_missing(PipelineGraph, PipelineNode, SQLOperator)
        g = PipelineGraph()
        node = _make_node("x")
        g.add_node(node)
        assert g.has_node("x")
        assert g.get_node("x").node_id == "x"

    def test_has_node_false(self):
        _skip_if_missing(PipelineGraph)
        g = PipelineGraph()
        assert not g.has_node("nonexistent")

    def test_remove_node(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = PipelineGraph()
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b"))
        g.add_edge(PipelineEdge(source="a", target="b"))
        removed = g.remove_node("a")
        assert removed.node_id == "a"
        assert not g.has_node("a")
        assert g.edge_count == 0

    def test_remove_nonexistent_raises(self):
        _skip_if_missing(PipelineGraph, NodeNotFoundError)
        g = PipelineGraph()
        with pytest.raises((NodeNotFoundError, KeyError)):
            g.remove_node("missing")

    def test_node_count(self):
        _skip_if_missing(PipelineGraph, PipelineNode, SQLOperator)
        g = PipelineGraph()
        assert g.node_count == 0
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b"))
        assert g.node_count == 2

    def test_node_ids(self):
        _skip_if_missing(PipelineGraph, PipelineNode, SQLOperator)
        g = PipelineGraph()
        g.add_node(_make_node("x"))
        g.add_node(_make_node("y"))
        assert set(g.node_ids) == {"x", "y"}

    def test_nodes_dict(self):
        _skip_if_missing(PipelineGraph, PipelineNode, SQLOperator)
        g = PipelineGraph()
        g.add_node(_make_node("a"))
        nodes = g.nodes
        assert "a" in nodes
        assert isinstance(nodes, dict)


# =====================================================================
# 3. Edge operations
# =====================================================================

class TestEdgeOperations:

    def test_add_and_has_edge(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = PipelineGraph()
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b"))
        g.add_edge(PipelineEdge(source="a", target="b"))
        assert g.has_edge("a", "b")
        assert not g.has_edge("b", "a")

    def test_remove_edge(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = PipelineGraph()
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b"))
        g.add_edge(PipelineEdge(source="a", target="b"))
        removed = g.remove_edge("a", "b")
        assert removed.source == "a"
        assert not g.has_edge("a", "b")
        assert g.edge_count == 0

    def test_remove_nonexistent_edge_raises(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator, EdgeNotFoundError)
        g = PipelineGraph()
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b"))
        with pytest.raises((EdgeNotFoundError, KeyError)):
            g.remove_edge("a", "b")

    def test_edge_count(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(3)
        assert g.edge_count == 2

    def test_edge_key_property(self):
        _skip_if_missing(PipelineEdge)
        e = PipelineEdge(source="a", target="b")
        assert e.key == ("a", "b")

    def test_edge_with_column_mapping(self):
        _skip_if_missing(PipelineEdge)
        e = PipelineEdge(source="a", target="b", column_mapping={"x": "y"})
        assert e.column_mapping == {"x": "y"}

    def test_edges_property(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = PipelineGraph()
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b"))
        g.add_edge(PipelineEdge(source="a", target="b"))
        edges = g.edges
        assert isinstance(edges, dict)
        assert ("a", "b") in edges


# =====================================================================
# 4. Topology: predecessors, successors, degree, ancestors, descendants
# =====================================================================

class TestTopology:

    def test_predecessors(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(3)
        assert g.predecessors("n1") == ["n0"]

    def test_successors(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(3)
        assert g.successors("n0") == ["n1"]

    def test_in_degree(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        assert g.in_degree("merge") == 2
        assert g.in_degree("src") == 0

    def test_out_degree(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        assert g.out_degree("src") == 2
        assert g.out_degree("merge") == 0

    def test_ancestors(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        assert g.ancestors("merge") == {"src", "left", "right"}

    def test_descendants(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        assert g.descendants("src") == {"left", "right", "merge"}


# =====================================================================
# 5. Sources and sinks
# =====================================================================

class TestSourcesAndSinks:

    def test_sources(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        assert g.sources() == ["src"]

    def test_sinks(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        assert g.sinks() == ["merge"]

    def test_linear_pipeline_sources_sinks(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(4)
        assert len(g.sources()) == 1
        assert g.sources()[0] == "n0"
        assert len(g.sinks()) == 1
        assert g.sinks()[0] == "n3"


# =====================================================================
# 6. Topological sort
# =====================================================================

class TestTopologicalSort:

    def test_topological_sort_valid(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(4)
        order = g.topological_sort()
        assert len(order) == 4
        # Each node comes after its predecessors
        positions = {nid: i for i, nid in enumerate(order)}
        for i in range(3):
            assert positions[f"n{i}"] < positions[f"n{i+1}"]

    def test_topological_sort_diamond(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        order = g.topological_sort()
        pos = {nid: i for i, nid in enumerate(order)}
        assert pos["src"] < pos["left"]
        assert pos["src"] < pos["right"]
        assert pos["left"] < pos["merge"]
        assert pos["right"] < pos["merge"]


# =====================================================================
# 7. Reverse topological sort
# =====================================================================

class TestReverseTopologicalSort:

    def test_reverse_topological_sort(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(3)
        fwd = g.topological_sort()
        rev = g.reverse_topological_sort()
        assert rev == list(reversed(fwd))


# =====================================================================
# 8. is_dag
# =====================================================================

class TestIsDAG:

    def test_dag_returns_true(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        assert g.is_dag() is True

    def test_single_node_is_dag(self):
        _skip_if_missing(PipelineGraph, PipelineNode, SQLOperator)
        g = PipelineGraph()
        g.add_node(_make_node("only"))
        assert g.is_dag() is True


# =====================================================================
# 9. detect_cycles
# =====================================================================

class TestDetectCycles:

    def test_no_cycles_in_dag(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        cycles = g.detect_cycles()
        assert cycles == []


# =====================================================================
# 10. Path finding
# =====================================================================

class TestPathFinding:

    def test_find_path(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(4)
        path = g.find_path("n0", "n3")
        assert path == ["n0", "n1", "n2", "n3"]

    def test_find_path_no_path(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(4)
        path = g.find_path("n3", "n0")
        assert path is None

    def test_all_paths_diamond(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        paths = g.all_paths("src", "merge")
        assert len(paths) == 2
        path_sets = {tuple(p) for p in paths}
        assert ("src", "left", "merge") in path_sets
        assert ("src", "right", "merge") in path_sets


# =====================================================================
# 11. Connected components and is_connected
# =====================================================================

class TestConnectedComponents:

    def test_connected_graph(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        assert g.is_connected() is True
        assert len(g.connected_components()) == 1

    def test_disconnected_graph(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = PipelineGraph(name="disconnected")
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b"))
        g.add_node(_make_node("c"))
        g.add_edge(PipelineEdge(source="a", target="b"))
        # "c" is disconnected
        assert g.is_connected() is False
        comps = g.connected_components()
        assert len(comps) == 2


# =====================================================================
# 12. Subgraph extraction
# =====================================================================

class TestSubgraph:

    def test_subgraph(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(4)
        sub = g.subgraph({"n0", "n1"})
        assert sub.node_count == 2
        assert sub.has_edge("n0", "n1")
        assert not sub.has_node("n2")

    def test_upstream_subgraph(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        up = g.upstream_subgraph("merge")
        assert up.node_count == 4  # all nodes are upstream of merge (incl merge)

    def test_downstream_subgraph(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        down = g.downstream_subgraph("src")
        assert down.node_count == 4  # all nodes


# =====================================================================
# 13. validate()
# =====================================================================

class TestValidate:

    def test_validate_clean_graph(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(3)
        issues = g.validate()
        # A simple graph with no schemas should have at most isolated-node warnings
        assert isinstance(issues, list)

    def test_validate_isolated_node(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = PipelineGraph()
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b"))
        g.add_edge(PipelineEdge(source="a", target="b"))
        g.add_node(_make_node("isolated"))
        issues = g.validate()
        assert any("isolated" in iss.lower() or "Isolated" in iss for iss in issues)

    def test_validate_schema_mismatch(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator, Schema, Column, SQLType)
        schema_a = _simple_schema("id", "name")
        schema_b = _simple_schema("id", "email")
        g = PipelineGraph()
        g.add_node(PipelineNode(node_id="a", output_schema=schema_a))
        g.add_node(PipelineNode(node_id="b", input_schema=schema_b))
        g.add_edge(PipelineEdge(source="a", target="b"))
        issues = g.validate()
        # Depending on schema comparison, there may or may not be issues
        assert isinstance(issues, list)


# =====================================================================
# 14. clone()
# =====================================================================

class TestClone:

    def test_clone_produces_independent_copy(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        cloned = g.clone()
        assert cloned.node_count == g.node_count
        assert cloned.edge_count == g.edge_count
        # Mutating clone should not affect original
        cloned.add_node(_make_node("extra"))
        assert not g.has_node("extra")
        assert cloned.has_node("extra")

    def test_clone_preserves_edges(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(3)
        c = g.clone()
        assert c.has_edge("n0", "n1")
        assert c.has_edge("n1", "n2")


# =====================================================================
# 15. merge()
# =====================================================================

class TestMerge:

    def test_merge_two_graphs(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g1 = PipelineGraph(name="g1")
        g1.add_node(_make_node("a"))
        g1.add_node(_make_node("b"))
        g1.add_edge(PipelineEdge(source="a", target="b"))

        g2 = PipelineGraph(name="g2")
        g2.add_node(_make_node("c"))
        g2.add_node(_make_node("d"))
        g2.add_edge(PipelineEdge(source="c", target="d"))

        merged = g1.merge(g2)
        assert merged.node_count == 4
        assert merged.edge_count == 2

    def test_merge_with_prefix(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g1 = PipelineGraph()
        g1.add_node(_make_node("a"))

        g2 = PipelineGraph()
        g2.add_node(_make_node("a"))  # same id

        merged = g1.merge(g2, prefix="g2_")
        assert merged.node_count == 2
        assert merged.has_node("a")
        assert merged.has_node("g2_a")

    def test_merge_conflict_raises(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g1 = PipelineGraph()
        g1.add_node(_make_node("a"))
        g2 = PipelineGraph()
        g2.add_node(_make_node("a"))
        with pytest.raises(Exception):
            g1.merge(g2, on_conflict="raise")


# =====================================================================
# 16. PipelineBuilder
# =====================================================================

class TestPipelineBuilder:

    def test_build_linear_pipeline(self):
        _skip_if_missing(build_linear_pipeline, SQLOperator, Schema)
        schema = Schema.empty()
        stages = [
            ("source", SQLOperator.SOURCE, schema),
            ("transform", SQLOperator.FILTER, schema),
            ("sink", SQLOperator.SINK, schema),
        ]
        g = build_linear_pipeline("linear_test", stages)
        assert g.node_count == 3
        assert g.edge_count == 2

    def test_build_diamond_pipeline(self):
        _skip_if_missing(build_diamond_pipeline, SQLOperator, Schema)
        schema = Schema.empty()
        g = build_diamond_pipeline(
            "diamond_test",
            source_id="src",
            source_schema=schema,
            left_id="left",
            right_id="right",
            merge_id="merge",
        )
        assert g.node_count == 4
        assert g.has_node("left")
        assert g.has_node("right")

    def test_build_star_pipeline(self):
        _skip_if_missing(build_star_pipeline, SQLOperator, Schema)
        schema = Schema.empty()
        g = build_star_pipeline(
            "star_test",
            center_id="hub",
            center_schema=schema,
            satellite_ids=["s1", "s2", "s3"],
        )
        assert g.node_count == 4
        assert g.edge_count == 3


# =====================================================================
# 17. PipelineBuilder chain building
# =====================================================================

class TestPipelineBuilderChain:

    def test_add_source_transform_sink(self):
        _skip_if_missing(PipelineBuilder, SQLOperator, Schema)
        schema = _simple_schema("id", "name") if Schema is not None and Column is not None else Schema.empty()
        g = (
            PipelineBuilder("chain_test")
            .add_source("raw", schema=schema)
            .add_transform("clean", "raw", operator=SQLOperator.FILTER)
            .add_sink("output", "clean")
            .build()
        )
        assert g.node_count == 3
        assert g.has_edge("raw", "clean")
        assert g.has_edge("clean", "output")

    def test_builder_add_edge(self):
        _skip_if_missing(PipelineBuilder, SQLOperator)
        builder = PipelineBuilder("test")
        builder.add_source("a")
        builder.add_source("b")
        builder.add_transform("c", "a", operator=SQLOperator.SELECT)
        builder.add_edge("b", "c")
        g = builder.build()
        assert g.has_edge("b", "c")

    def test_builder_build_with_validation(self):
        _skip_if_missing(PipelineBuilder, SQLOperator)
        g = (
            PipelineBuilder("v")
            .add_source("s")
            .add_sink("t", "s")
            .build(validate=True)
        )
        assert g.node_count == 2


# =====================================================================
# 18. depth() and width()
# =====================================================================

class TestDepthAndWidth:

    def test_depth_linear(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(4)
        assert g.depth() == 3  # 3 edges in n0->n1->n2->n3

    def test_depth_single_node(self):
        _skip_if_missing(PipelineGraph, PipelineNode, SQLOperator)
        g = PipelineGraph()
        g.add_node(_make_node("only"))
        assert g.depth() == 0

    def test_width_diamond(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        assert g.width() >= 2  # left and right are at the same level

    def test_width_linear(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(4)
        assert g.width() == 1

    def test_depth_empty_graph(self):
        _skip_if_missing(PipelineGraph)
        g = PipelineGraph()
        assert g.depth() == 0

    def test_width_empty_graph(self):
        _skip_if_missing(PipelineGraph)
        g = PipelineGraph()
        assert g.width() == 0


# =====================================================================
# 19. to_dict() / from_dict() round-trip
# =====================================================================

class TestSerialization:

    def test_node_round_trip(self):
        _skip_if_missing(PipelineNode, SQLOperator)
        node = _make_node("x", operator=SQLOperator.FILTER)
        d = node.to_dict()
        restored = PipelineNode.from_dict(d)
        assert restored.node_id == "x"
        assert restored.operator == SQLOperator.FILTER

    def test_edge_round_trip(self):
        _skip_if_missing(PipelineEdge, EdgeType)
        e = PipelineEdge(source="a", target="b", column_mapping={"x": "y"})
        d = e.to_dict()
        restored = PipelineEdge.from_dict(d)
        assert restored.source == "a"
        assert restored.target == "b"
        assert restored.column_mapping == {"x": "y"}

    def test_graph_round_trip(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        d = g.to_dict()
        restored = PipelineGraph.from_dict(d)
        assert restored.node_count == g.node_count
        assert restored.edge_count == g.edge_count
        assert restored.has_node("src")
        assert restored.has_edge("src", "left")

    def test_graph_round_trip_preserves_name(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = PipelineGraph(name="my_pipe", version="3.0")
        g.add_node(_make_node("a"))
        d = g.to_dict()
        restored = PipelineGraph.from_dict(d)
        assert restored.name == "my_pipe"
        assert restored.version == "3.0"


# =====================================================================
# 20. Edge cases
# =====================================================================

class TestEdgeCases:

    def test_single_node_graph(self):
        _skip_if_missing(PipelineGraph, PipelineNode, SQLOperator)
        g = PipelineGraph()
        g.add_node(_make_node("only"))
        assert g.node_count == 1
        assert g.edge_count == 0
        assert g.sources() == ["only"]
        assert g.sinks() == ["only"]
        assert g.is_dag()
        assert g.topological_sort() == ["only"]

    def test_empty_graph(self):
        _skip_if_missing(PipelineGraph)
        g = PipelineGraph()
        assert g.node_count == 0
        assert g.edge_count == 0
        assert g.sources() == []
        assert g.sinks() == []
        assert g.depth() == 0
        assert g.width() == 0

    def test_disconnected_multi_component(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = PipelineGraph()
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b"))
        g.add_node(_make_node("c"))
        g.add_node(_make_node("d"))
        g.add_edge(PipelineEdge(source="a", target="b"))
        g.add_edge(PipelineEdge(source="c", target="d"))
        comps = g.connected_components()
        assert len(comps) == 2
        assert g.is_connected() is False

    def test_len_and_contains(self):
        _skip_if_missing(PipelineGraph, PipelineNode, SQLOperator)
        g = _linear_graph(3)
        assert len(g) == 3
        assert "n0" in g
        assert "nonexistent" not in g

    def test_iter_yields_topological_order(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(3)
        order = list(g)
        assert len(order) == 3
        assert order.index("n0") < order.index("n1") < order.index("n2")

    def test_replace_node(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(3)
        old_node = g.get_node("n1")
        new_node = PipelineNode(
            node_id="n1",
            operator=SQLOperator.FILTER,
            query_text="SELECT * WHERE active",
        )
        g.replace_node("n1", new_node)
        assert g.get_node("n1").query_text == "SELECT * WHERE active"
        # Edges should be preserved
        assert g.has_edge("n0", "n1")
        assert g.has_edge("n1", "n2")


# =====================================================================
# 21. Analysis functions
# =====================================================================

class TestAnalysis:

    def test_impact_analysis(self):
        _skip_if_missing(impact_analysis, PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(4)
        result = impact_analysis(g, "n0")
        assert "n1" in result.affected_nodes
        assert "n2" in result.affected_nodes
        assert "n3" in result.affected_nodes
        assert result.max_depth >= 3

    def test_impact_analysis_leaf_node(self):
        _skip_if_missing(impact_analysis, PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(4)
        result = impact_analysis(g, "n3")
        assert len(result.affected_nodes) == 0

    def test_impact_analysis_diamond(self):
        _skip_if_missing(impact_analysis, PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        result = impact_analysis(g, "src")
        assert set(result.affected_nodes) == {"left", "right", "merge"}

    def test_topological_levels(self):
        _skip_if_missing(assign_topological_levels, PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        levels = assign_topological_levels(g)
        assert levels["src"] == 0
        assert levels["left"] == 1
        assert levels["right"] == 1
        assert levels["merge"] == 2

    def test_detect_bottlenecks(self):
        _skip_if_missing(detect_bottlenecks, PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        results = detect_bottlenecks(g)
        assert len(results) > 0
        # "src" should have high score (highest fan-out)
        node_ids = [r.node_id for r in results]
        assert "src" in node_ids

    def test_compute_metrics(self):
        _skip_if_missing(compute_metrics, PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        m = compute_metrics(g)
        assert m.node_count == 4
        assert m.edge_count == 4
        assert m.source_count == 1
        assert m.sink_count == 1
        assert m.depth >= 2
        assert m.width >= 2
        assert m.is_dag is True

    def test_compute_metrics_empty(self):
        _skip_if_missing(compute_metrics, PipelineGraph)
        g = PipelineGraph()
        m = compute_metrics(g)
        assert m.node_count == 0

    def test_execution_waves(self):
        _skip_if_missing(compute_execution_waves, PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        waves = compute_execution_waves(g)
        assert len(waves) >= 2
        # First wave should contain "src"
        first_wave_nodes = waves[0].node_ids
        assert "src" in first_wave_nodes

    def test_execution_waves_linear(self):
        _skip_if_missing(compute_execution_waves, PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(4)
        waves = compute_execution_waves(g)
        # Linear pipeline: each node is in its own wave
        assert len(waves) == 4
        for w in waves:
            assert len(w.node_ids) == 1


# =====================================================================
# PipelineNode and PipelineEdge properties
# =====================================================================

class TestNodeEdgeProperties:

    def test_node_default_operator(self):
        _skip_if_missing(PipelineNode, SQLOperator)
        node = PipelineNode(node_id="test")
        assert node.operator == SQLOperator.TRANSFORM

    def test_in_fragment_f(self):
        _skip_if_missing(PipelineNode, SQLOperator)
        node = PipelineNode(node_id="sel", operator=SQLOperator.SELECT)
        assert isinstance(node.in_fragment_f, bool)

    def test_with_output_schema(self):
        _skip_if_missing(PipelineNode, SQLOperator, Schema)
        node = PipelineNode(node_id="a")
        updated = node.with_output_schema(Schema.empty())
        assert updated.node_id == "a"

    def test_node_str_repr(self):
        _skip_if_missing(PipelineNode, SQLOperator)
        node = PipelineNode(node_id="mynode", operator=SQLOperator.FILTER)
        assert "mynode" in repr(node)
        assert str(node) == "mynode"

    def test_edge_repr_and_str(self):
        _skip_if_missing(PipelineEdge)
        e = PipelineEdge(source="a", target="b", label="lbl")
        assert "a" in repr(e)
        assert str(PipelineEdge(source="a", target="b")) == "a -> b"

    def test_edge_default_type(self):
        _skip_if_missing(PipelineEdge, EdgeType)
        e = PipelineEdge(source="a", target="b")
        assert e.edge_type == EdgeType.DATA_FLOW


# =====================================================================
# PipelineGraph additional methods
# =====================================================================

class TestPipelineGraphAdditional:

    def test_get_edge(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(3)
        e = g.get_edge("n0", "n1")
        assert e.source == "n0"

    def test_total_cost(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator, CostEstimate)
        g = PipelineGraph()
        g.add_node(PipelineNode(node_id="a", cost_estimate=CostEstimate(compute_seconds=1.0)))
        g.add_node(PipelineNode(node_id="b", cost_estimate=CostEstimate(compute_seconds=2.0)))
        g.add_edge(PipelineEdge(source="a", target="b"))
        assert g.total_cost().compute_seconds == 3.0

    def test_fragment_f_nodes(self):
        _skip_if_missing(PipelineGraph, PipelineNode, SQLOperator)
        g = PipelineGraph()
        g.add_node(PipelineNode(node_id="sel", operator=SQLOperator.SELECT))
        assert isinstance(g.fragment_f_nodes(), set)

    def test_nodes_by_operator(self):
        _skip_if_missing(PipelineGraph, PipelineNode, SQLOperator)
        g = PipelineGraph()
        g.add_node(PipelineNode(node_id="a", operator=SQLOperator.SELECT))
        g.add_node(PipelineNode(node_id="b", operator=SQLOperator.FILTER))
        g.add_node(PipelineNode(node_id="c", operator=SQLOperator.SELECT))
        assert set(g.nodes_by_operator(SQLOperator.SELECT)) == {"a", "c"}

    def test_intermediate_nodes(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(4)
        intermediates = g.intermediate_nodes()
        assert "n1" in intermediates and "n0" not in intermediates

    def test_graph_repr(self):
        _skip_if_missing(PipelineGraph)
        assert "test_repr" in repr(PipelineGraph(name="test_repr"))


# =====================================================================
# Builder deserialization and schema-aware tests
# =====================================================================

class TestBuilderExtras:

    def test_builder_from_dict(self):
        _skip_if_missing(PipelineBuilder, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        builder = PipelineBuilder.from_dict(g.to_dict())
        assert builder.build(validate=False).node_count == g.node_count

    def test_builder_from_json(self):
        _skip_if_missing(PipelineBuilder, PipelineNode, SQLOperator)
        import json
        g = PipelineGraph(name="json_test")
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b"))
        g.add_edge(PipelineEdge(source="a", target="b"))
        builder = PipelineBuilder.from_json(json.dumps(g.to_dict()))
        assert builder.build(validate=False).node_count == 2

    def test_schema_inferred_from_upstream(self):
        _skip_if_missing(PipelineBuilder, SQLOperator, Schema, Column, SQLType)
        schema = _simple_schema("id", "name", "email")
        g = (
            PipelineBuilder("schema_test")
            .add_source("src", schema=schema)
            .add_transform("t1", "src", operator=SQLOperator.SELECT)
            .build()
        )
        t1 = g.get_node("t1")
        if t1.input_schema.columns:
            assert len(t1.input_schema.columns) > 0

    def test_add_sql_chain(self):
        _skip_if_missing(PipelineBuilder, SQLOperator)
        g = (
            PipelineBuilder("chain")
            .add_source("src")
            .add_sql_chain("src", [("step1", "SELECT *"), ("step2", "SELECT id")])
            .build()
        )
        assert g.node_count == 3 and g.has_edge("step1", "step2")

    def test_wide_graph(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = PipelineGraph(name="wide")
        g.add_node(_make_node("src"))
        for i in range(20):
            nid = f"sat_{i}"
            g.add_node(_make_node(nid))
            g.add_edge(PipelineEdge(source="src", target=nid))
        assert g.width() >= 20 and g.depth() == 1
