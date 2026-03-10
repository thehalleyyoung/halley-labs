"""
Unit tests for ``arc.io`` — JSON/YAML serialisation, schema constants,
and round-trip validation for pipeline specs, deltas, and repair plans.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

# ─────────────────────────────────────────────────────────────────────
# Graceful imports
# ─────────────────────────────────────────────────────────────────────

try:
    from arc.io.json_format import (
        PipelineSpec,
        DeltaSerializer,
        RepairPlanSerializer,
    )
    HAS_JSON_FORMAT = True
except ImportError:
    HAS_JSON_FORMAT = False

try:
    from arc.io.yaml_format import YAMLPipelineSpec
    HAS_YAML_FORMAT = True
except ImportError:
    HAS_YAML_FORMAT = False

try:
    from arc.io.schema import (
        PIPELINE_SPEC_SCHEMA_V1,
        DELTA_SPEC_SCHEMA_V1,
        REPAIR_PLAN_SCHEMA_V1,
        EXAMPLE_PIPELINE_SPEC,
        EXAMPLE_DELTA_SPEC,
        EXAMPLE_REPAIR_PLAN,
    )
    HAS_SCHEMA = True
except ImportError:
    HAS_SCHEMA = False

try:
    from arc.graph.pipeline import PipelineNode, PipelineEdge, PipelineGraph
    from arc.graph.builder import PipelineBuilder
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False

try:
    from arc.types.base import (
        SQLType, ParameterisedType, Column, Schema,
    )
    from arc.types.operators import SQLOperator
    HAS_TYPES = True
except ImportError:
    HAS_TYPES = False

requires_json = pytest.mark.skipif(not HAS_JSON_FORMAT, reason="arc.io.json_format not available")
requires_yaml = pytest.mark.skipif(not HAS_YAML_FORMAT, reason="arc.io.yaml_format not available")
requires_schema = pytest.mark.skipif(not HAS_SCHEMA, reason="arc.io.schema not available")
requires_graph = pytest.mark.skipif(not HAS_GRAPH, reason="arc.graph not available")
requires_types = pytest.mark.skipif(not HAS_TYPES, reason="arc.types not available")


# =====================================================================
# Helpers
# =====================================================================

def _col(name: str, base=None, nullable: bool = True, pos: int = 0):
    """Build a Column with minimal boilerplate."""
    if base is None:
        base = SQLType.INT
    return Column.quick(name, base, nullable=nullable, position=pos)


def _simple_schema():
    """Three-column schema: id INT, name VARCHAR, active BOOLEAN."""
    return Schema(columns=(
        _col("id", SQLType.INT, nullable=False, pos=0),
        _col("name", SQLType.VARCHAR, pos=1),
        _col("active", SQLType.BOOLEAN, pos=2),
    ))


def _make_node(node_id, operator=None, query="", schema=None):
    kw: dict[str, Any] = {"node_id": node_id}
    if operator is not None:
        kw["operator"] = operator
    if query:
        kw["query_text"] = query
    if schema is not None:
        kw["output_schema"] = schema
    return PipelineNode(**kw)


def _build_linear_graph():
    """Build a linear 3-node pipeline graph for serialisation tests."""
    schema = _simple_schema()
    g = PipelineGraph(name="linear_io_test")
    g.add_node(_make_node("source", SQLOperator.SOURCE, schema=schema))
    g.add_node(_make_node("transform", SQLOperator.SELECT,
                          query="SELECT id, name FROM source", schema=schema))
    g.add_node(_make_node("sink", SQLOperator.SINK, schema=schema))
    g.add_edge(PipelineEdge(source="source", target="transform"))
    g.add_edge(PipelineEdge(source="transform", target="sink"))
    return g


def _build_empty_graph():
    """Build an empty pipeline graph (no nodes, no edges)."""
    return PipelineGraph(name="empty_io_test")


# =====================================================================
# 1. PipelineSpec.to_json / from_json round-trip
# =====================================================================

@requires_json
@requires_graph
@requires_types
class TestPipelineSpecJsonRoundTrip:
    """Serialise a graph to JSON and deserialise it back."""

    def test_round_trip_linear(self):
        graph = _build_linear_graph()
        json_str = PipelineSpec.to_json(graph)
        restored = PipelineSpec.from_json(json_str)
        assert restored.name == graph.name
        assert set(restored.node_ids) == set(graph.node_ids)

    def test_json_is_valid_string(self):
        graph = _build_linear_graph()
        json_str = PipelineSpec.to_json(graph)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "version" in parsed

    def test_round_trip_preserves_edges(self):
        graph = _build_linear_graph()
        json_str = PipelineSpec.to_json(graph)
        restored = PipelineSpec.from_json(json_str)
        orig_edges = {(e.source, e.target) for e in graph.edges.values()}
        rest_edges = {(e.source, e.target) for e in restored.edges.values()}
        assert orig_edges == rest_edges


# =====================================================================
# 2. PipelineSpec.to_dict / from_dict round-trip
# =====================================================================

@requires_json
@requires_graph
@requires_types
class TestPipelineSpecDictRoundTrip:

    def test_to_dict_returns_dict(self):
        graph = _build_linear_graph()
        d = PipelineSpec.to_dict(graph)
        assert isinstance(d, dict)

    def test_round_trip(self):
        graph = _build_linear_graph()
        d = PipelineSpec.to_dict(graph)
        restored = PipelineSpec.from_dict(d)
        assert set(restored.node_ids) == set(graph.node_ids)

    def test_dict_has_version(self):
        graph = _build_linear_graph()
        d = PipelineSpec.to_dict(graph)
        assert "version" in d


# =====================================================================
# 3. PipelineSpec.save / load file round-trip
# =====================================================================

@requires_json
@requires_graph
@requires_types
class TestPipelineSpecFileRoundTrip:

    def test_save_and_load(self, tmp_path):
        graph = _build_linear_graph()
        path = tmp_path / "pipeline.json"
        PipelineSpec.save(graph, str(path))
        assert path.exists()
        restored = PipelineSpec.load(str(path))
        assert set(restored.node_ids) == set(graph.node_ids)

    def test_saved_file_is_valid_json(self, tmp_path):
        graph = _build_linear_graph()
        path = tmp_path / "pipeline.json"
        PipelineSpec.save(graph, str(path))
        with open(path) as f:
            parsed = json.load(f)
        assert isinstance(parsed, dict)

    def test_load_returns_pipeline_graph(self, tmp_path):
        graph = _build_linear_graph()
        path = tmp_path / "pipeline.json"
        PipelineSpec.save(graph, str(path))
        restored = PipelineSpec.load(str(path))
        assert isinstance(restored, PipelineGraph)


# =====================================================================
# 4-5. PipelineSpec.validate_spec: valid / invalid
# =====================================================================

@requires_json
class TestPipelineSpecValidation:

    def test_valid_spec_no_errors(self, sample_pipeline_spec_dict):
        spec = dict(sample_pipeline_spec_dict)
        # Ensure nodes use 'node_id' key as required by the API
        spec["nodes"] = [
            {**{k: v for k, v in n.items() if k != "id"}, "node_id": n.get("node_id", n.get("id"))}
            for n in spec["nodes"]
        ]
        errors = PipelineSpec.validate_spec(spec)
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_invalid_spec_missing_nodes(self):
        bad = {"version": "1.0", "name": "bad"}
        errors = PipelineSpec.validate_spec(bad)
        assert len(errors) > 0

    def test_invalid_spec_missing_version(self):
        bad = {"name": "bad", "nodes": [], "edges": []}
        errors = PipelineSpec.validate_spec(bad)
        assert len(errors) > 0

    def test_invalid_spec_wrong_type(self):
        errors = PipelineSpec.validate_spec({"__invalid__": True})
        assert len(errors) > 0

    def test_empty_dict_is_invalid(self):
        errors = PipelineSpec.validate_spec({})
        assert len(errors) > 0


# =====================================================================
# 6. DeltaSerializer round-trip
# =====================================================================

@requires_json
class TestDeltaSerializerRoundTrip:

    def test_schema_delta_round_trip(self, sample_delta_dict):
        json_str = DeltaSerializer.serialize_delta(sample_delta_dict)
        restored = DeltaSerializer.deserialize_delta(json_str)
        assert restored["type"] == sample_delta_dict["type"]
        assert len(restored["operations"]) == len(sample_delta_dict["operations"])

    def test_serialise_returns_string(self, sample_delta_dict):
        result = DeltaSerializer.serialize_delta(sample_delta_dict)
        assert isinstance(result, str)

    def test_deserialise_returns_dict(self, sample_delta_dict):
        json_str = DeltaSerializer.serialize_delta(sample_delta_dict)
        result = DeltaSerializer.deserialize_delta(json_str)
        assert isinstance(result, dict)

    def test_round_trip_preserves_operations(self, sample_delta_dict):
        json_str = DeltaSerializer.serialize_delta(sample_delta_dict)
        restored = DeltaSerializer.deserialize_delta(json_str)
        orig_op = sample_delta_dict["operations"][0]
        rest_op = restored["operations"][0]
        assert orig_op["op"] == rest_op["op"]
        assert orig_op["name"] == rest_op["name"]


# =====================================================================
# 7. DeltaSerializer.validate_delta
# =====================================================================

@requires_json
class TestDeltaSerializerValidation:

    def test_valid_delta(self, sample_delta_dict):
        delta = dict(sample_delta_dict)
        # API expects 'sort' key with value 'schema' instead of 'type'/'schema_delta'
        if "sort" not in delta:
            delta["sort"] = "schema"
        # API expects operations to have 'type' key instead of 'op'
        if "operations" in delta:
            delta["operations"] = [
                {**{k: v for k, v in op.items() if k != "op"}, "type": op.get("type", op.get("op"))}
                for op in delta["operations"]
            ]
        errors = DeltaSerializer.validate_delta(delta)
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_invalid_delta_no_type(self):
        bad = {"operations": []}
        errors = DeltaSerializer.validate_delta(bad)
        assert len(errors) > 0

    def test_invalid_delta_wrong_type(self):
        errors = DeltaSerializer.validate_delta({"__invalid__": True})
        assert len(errors) > 0

    def test_invalid_delta_no_operations(self):
        bad = {"type": "schema_delta"}
        errors = DeltaSerializer.validate_delta(bad)
        assert len(errors) > 0


# =====================================================================
# 8. RepairPlanSerializer round-trip
# =====================================================================

@requires_json
class TestRepairPlanSerializerRoundTrip:

    def _sample_plan(self):
        return {
            "version": "1.0",
            "actions": [
                {
                    "node_id": "transform",
                    "action_type": "ALTER_TABLE",
                    "sql": "ALTER TABLE transform ADD COLUMN email VARCHAR",
                    "estimated_cost": 1.0,
                },
            ],
            "cost_estimate": 1.0,
        }

    def test_round_trip(self):
        plan = self._sample_plan()
        json_str = RepairPlanSerializer.serialize(plan)
        restored = RepairPlanSerializer.deserialize(json_str)
        assert restored["version"] == plan["version"]
        assert len(restored["actions"]) == len(plan["actions"])

    def test_serialise_returns_string(self):
        result = RepairPlanSerializer.serialize(self._sample_plan())
        assert isinstance(result, str)

    def test_validate_valid_plan(self):
        errors = RepairPlanSerializer.validate(self._sample_plan())
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_invalid_plan(self):
        errors = RepairPlanSerializer.validate({})
        assert len(errors) > 0


# =====================================================================
# 9. YAMLPipelineSpec.to_yaml / from_yaml round-trip
# =====================================================================

@requires_yaml
@requires_graph
@requires_types
class TestYAMLPipelineSpecYamlRoundTrip:

    def test_round_trip(self):
        graph = _build_linear_graph()
        yaml_str = YAMLPipelineSpec.to_yaml(graph)
        restored = YAMLPipelineSpec.from_yaml(yaml_str)
        assert set(restored.node_ids) == set(graph.node_ids)

    def test_to_yaml_returns_string(self):
        graph = _build_linear_graph()
        result = YAMLPipelineSpec.to_yaml(graph)
        assert isinstance(result, str)

    def test_round_trip_preserves_name(self):
        graph = _build_linear_graph()
        yaml_str = YAMLPipelineSpec.to_yaml(graph)
        restored = YAMLPipelineSpec.from_yaml(yaml_str)
        assert restored.name == graph.name


# =====================================================================
# 10. YAMLPipelineSpec.save / load file round-trip
# =====================================================================

@requires_yaml
@requires_graph
@requires_types
class TestYAMLPipelineSpecFileRoundTrip:

    def test_save_and_load(self, tmp_path):
        graph = _build_linear_graph()
        path = tmp_path / "pipeline.yaml"
        YAMLPipelineSpec.save(graph, str(path))
        assert path.exists()
        restored = YAMLPipelineSpec.load(str(path))
        assert set(restored.node_ids) == set(graph.node_ids)

    def test_saved_file_contains_yaml(self, tmp_path):
        graph = _build_linear_graph()
        path = tmp_path / "pipeline.yaml"
        YAMLPipelineSpec.save(graph, str(path))
        content = path.read_text()
        assert len(content) > 0
        # YAML should not start with '{' (that would be JSON)
        assert not content.strip().startswith("{")


# =====================================================================
# 11. EXAMPLE_PIPELINE_SPEC validates against schema
# =====================================================================

@requires_json
@requires_schema
class TestExampleSpecValidation:

    def test_example_pipeline_spec_validates(self):
        errors = PipelineSpec.validate_spec(EXAMPLE_PIPELINE_SPEC)
        assert isinstance(errors, list)
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_example_delta_spec_validates(self):
        errors = DeltaSerializer.validate_delta(EXAMPLE_DELTA_SPEC)
        assert isinstance(errors, list)
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_example_repair_plan_validates(self):
        errors = RepairPlanSerializer.validate(EXAMPLE_REPAIR_PLAN)
        assert isinstance(errors, list)
        assert len(errors) == 0, f"Validation errors: {errors}"


# =====================================================================
# 12. Schema constants are valid dicts
# =====================================================================

@requires_schema
class TestSchemaConstants:

    def test_pipeline_spec_schema_is_dict(self):
        assert isinstance(PIPELINE_SPEC_SCHEMA_V1, dict)

    def test_delta_spec_schema_is_dict(self):
        assert isinstance(DELTA_SPEC_SCHEMA_V1, dict)

    def test_repair_plan_schema_is_dict(self):
        assert isinstance(REPAIR_PLAN_SCHEMA_V1, dict)

    def test_example_pipeline_spec_is_dict(self):
        assert isinstance(EXAMPLE_PIPELINE_SPEC, dict)

    def test_example_delta_spec_is_dict(self):
        assert isinstance(EXAMPLE_DELTA_SPEC, dict)

    def test_example_repair_plan_is_dict(self):
        assert isinstance(EXAMPLE_REPAIR_PLAN, dict)

    def test_pipeline_schema_has_type_field(self):
        # JSON Schema should have a 'type' or 'properties' field
        assert "type" in PIPELINE_SPEC_SCHEMA_V1 or "properties" in PIPELINE_SPEC_SCHEMA_V1

    def test_delta_schema_has_type_field(self):
        assert "type" in DELTA_SPEC_SCHEMA_V1 or "properties" in DELTA_SPEC_SCHEMA_V1

    def test_repair_plan_schema_has_type_field(self):
        assert "type" in REPAIR_PLAN_SCHEMA_V1 or "properties" in REPAIR_PLAN_SCHEMA_V1


# =====================================================================
# 13. Error handling
# =====================================================================

@requires_json
class TestErrorHandling:

    def test_load_nonexistent_file(self):
        with pytest.raises(Exception):
            PipelineSpec.load("/nonexistent/path/file.json")

    def test_from_json_invalid_json(self):
        with pytest.raises(Exception):
            PipelineSpec.from_json("{not valid json!!!")

    def test_deserialise_delta_invalid_json(self):
        with pytest.raises(Exception):
            DeltaSerializer.deserialize_delta("{broken json")

    def test_deserialise_plan_invalid_json(self):
        with pytest.raises(Exception):
            RepairPlanSerializer.deserialize("{broken json")


@requires_yaml
class TestYAMLErrorHandling:

    def test_load_nonexistent_file(self):
        with pytest.raises(Exception):
            YAMLPipelineSpec.load("/nonexistent/path/file.yaml")

    def test_from_yaml_invalid_yaml(self):
        with pytest.raises(Exception):
            YAMLPipelineSpec.from_yaml(":\n  bad:\n    - [unclosed")


# =====================================================================
# 14. Edge cases: empty graph, minimal spec
# =====================================================================

@requires_json
@requires_graph
@requires_types
class TestEdgeCases:

    def test_empty_graph_json_round_trip(self):
        graph = _build_empty_graph()
        json_str = PipelineSpec.to_json(graph)
        restored = PipelineSpec.from_json(json_str)
        assert len(list(restored.node_ids)) == 0

    def test_empty_graph_dict_round_trip(self):
        graph = _build_empty_graph()
        d = PipelineSpec.to_dict(graph)
        restored = PipelineSpec.from_dict(d)
        assert len(list(restored.node_ids)) == 0

    def test_minimal_spec_validates(self):
        minimal = {
            "version": "1.0",
            "name": "minimal",
            "nodes": [],
            "edges": [],
        }
        errors = PipelineSpec.validate_spec(minimal)
        assert isinstance(errors, list)

    def test_single_node_graph(self):
        schema = _simple_schema()
        g = PipelineGraph(name="single")
        g.add_node(_make_node("only", SQLOperator.SOURCE, schema=schema))
        json_str = PipelineSpec.to_json(g)
        restored = PipelineSpec.from_json(json_str)
        assert set(restored.node_ids) == {"only"}


@requires_yaml
@requires_graph
@requires_types
class TestYAMLEdgeCases:

    def test_empty_graph_yaml_round_trip(self):
        graph = _build_empty_graph()
        yaml_str = YAMLPipelineSpec.to_yaml(graph)
        restored = YAMLPipelineSpec.from_yaml(yaml_str)
        assert len(list(restored.node_ids)) == 0

    def test_empty_graph_file_round_trip(self, tmp_path):
        graph = _build_empty_graph()
        path = tmp_path / "empty.yaml"
        YAMLPipelineSpec.save(graph, str(path))
        restored = YAMLPipelineSpec.load(str(path))
        assert len(list(restored.node_ids)) == 0
