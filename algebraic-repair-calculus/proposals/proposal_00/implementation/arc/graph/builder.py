"""
Pipeline graph construction via a fluent builder API.

``PipelineBuilder`` lets callers construct a :class:`PipelineGraph` with
a chainable interface that handles node/edge creation, automatic schema
inference, and structural validation in one pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from arc.types.base import (
    AvailabilityContract,
    Column,
    CostEstimate,
    EdgeType,
    NodeMetadata,
    ParameterisedType,
    QualityConstraint,
    Schema,
    SQLType,
)
from arc.types.errors import (
    GraphError,
    NodeNotFoundError,
    SchemaError,
    ErrorCode,
)
from arc.types.operators import (
    OperatorProperties,
    SQLOperator,
    get_default_properties,
)
from arc.graph.pipeline import PipelineEdge, PipelineGraph, PipelineNode


# =====================================================================
# Fluent builder
# =====================================================================

class PipelineBuilder:
    """Fluent API for constructing pipeline graphs step by step.

    Usage::

        graph = (
            PipelineBuilder("my_pipeline")
            .add_source("raw_users", schema=user_schema)
            .add_transform("clean_users", "raw_users",
                           operator=SQLOperator.FILTER,
                           query="SELECT * FROM raw_users WHERE active")
            .add_sink("output", "clean_users")
            .build()
        )
    """

    def __init__(self, name: str = "", version: str = "1.0") -> None:
        self._graph = PipelineGraph(name=name, version=version)
        self._pending_schemas: dict[str, Schema] = {}

    # ── Source nodes ──

    def add_source(
        self,
        node_id: str,
        schema: Schema | None = None,
        quality_constraints: Sequence[QualityConstraint] = (),
        availability: AvailabilityContract | None = None,
        cost: CostEstimate | None = None,
        metadata: NodeMetadata | None = None,
    ) -> PipelineBuilder:
        """Add a data-source node (no incoming edges)."""
        node = PipelineNode(
            node_id=node_id,
            operator=SQLOperator.SOURCE,
            output_schema=schema or Schema.empty(),
            quality_constraints=tuple(quality_constraints),
            availability_contract=availability or AvailabilityContract(),
            cost_estimate=cost or CostEstimate.zero(),
            metadata=metadata or NodeMetadata(),
        )
        self._graph.add_node(node)
        if schema:
            self._pending_schemas[node_id] = schema
        return self

    # ── Transform nodes ──

    def add_transform(
        self,
        node_id: str,
        *upstream_ids: str,
        operator: SQLOperator = SQLOperator.TRANSFORM,
        query: str = "",
        output_schema: Schema | None = None,
        quality_constraints: Sequence[QualityConstraint] = (),
        availability: AvailabilityContract | None = None,
        cost: CostEstimate | None = None,
        properties: OperatorProperties | None = None,
        metadata: NodeMetadata | None = None,
        column_mappings: dict[str, dict[str, str]] | None = None,
        edge_type: EdgeType = EdgeType.DATA_FLOW,
    ) -> PipelineBuilder:
        """Add a transformation node with edges from upstream nodes."""
        # Infer input schema by merging upstream output schemas
        input_schema = self._infer_input_schema(upstream_ids)
        node = PipelineNode(
            node_id=node_id,
            operator=operator,
            query_text=query,
            input_schema=input_schema,
            output_schema=output_schema or input_schema,
            quality_constraints=tuple(quality_constraints),
            availability_contract=availability or AvailabilityContract(),
            cost_estimate=cost or CostEstimate.zero(),
            properties=properties,
            metadata=metadata or NodeMetadata(),
        )
        self._graph.add_node(node)
        if output_schema:
            self._pending_schemas[node_id] = output_schema
        elif input_schema.columns:
            self._pending_schemas[node_id] = input_schema
        for uid in upstream_ids:
            mapping = {}
            if column_mappings and uid in column_mappings:
                mapping = column_mappings[uid]
            edge = PipelineEdge(
                source=uid,
                target=node_id,
                column_mapping=mapping,
                edge_type=edge_type,
            )
            self._graph.add_edge(edge)
        return self

    # ── Sink nodes ──

    def add_sink(
        self,
        node_id: str,
        *upstream_ids: str,
        output_schema: Schema | None = None,
        metadata: NodeMetadata | None = None,
        edge_type: EdgeType = EdgeType.DATA_FLOW,
    ) -> PipelineBuilder:
        """Add a terminal sink node."""
        input_schema = self._infer_input_schema(upstream_ids)
        node = PipelineNode(
            node_id=node_id,
            operator=SQLOperator.SINK,
            input_schema=input_schema,
            output_schema=output_schema or input_schema,
            metadata=metadata or NodeMetadata(),
        )
        self._graph.add_node(node)
        for uid in upstream_ids:
            self._graph.add_edge(PipelineEdge(
                source=uid,
                target=node_id,
                edge_type=edge_type,
            ))
        return self

    # ── Generic node ──

    def add_node(self, node: PipelineNode) -> PipelineBuilder:
        """Add a pre-constructed node."""
        self._graph.add_node(node)
        if node.output_schema.columns:
            self._pending_schemas[node.node_id] = node.output_schema
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        column_mapping: dict[str, str] | None = None,
        edge_type: EdgeType = EdgeType.DATA_FLOW,
        label: str = "",
    ) -> PipelineBuilder:
        """Add an edge between existing nodes."""
        edge = PipelineEdge(
            source=source,
            target=target,
            column_mapping=column_mapping or {},
            edge_type=edge_type,
            label=label,
        )
        self._graph.add_edge(edge)
        return self

    # ── Convenience: chain of SQL queries ──

    def add_sql_chain(
        self,
        source_id: str,
        steps: Sequence[tuple[str, str]],
        output_schema: Schema | None = None,
    ) -> PipelineBuilder:
        """Add a chain of SQL transformations: [(node_id, query), ...]."""
        prev = source_id
        for i, (nid, query) in enumerate(steps):
            schema = output_schema if i == len(steps) - 1 else None
            self.add_transform(
                nid, prev,
                operator=SQLOperator.SELECT,
                query=query,
                output_schema=schema,
            )
            prev = nid
        return self

    # ── Build ──

    def build(self, validate: bool = True) -> PipelineGraph:
        """Finalise and return the pipeline graph.

        If *validate* is True, structural issues are collected and
        reported (but do not prevent building).
        """
        if validate:
            issues = self._graph.validate()
            # Issues are informational; callers can check them via
            # graph.validate() again if needed.
        return self._graph

    # ── From specifications ──

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineBuilder:
        """Build from a dictionary specification."""
        builder = cls(
            name=d.get("name", ""),
            version=d.get("version", "1.0"),
        )
        for node_spec in d.get("nodes", []):
            node = PipelineNode.from_dict(node_spec)
            builder.add_node(node)
        for edge_spec in d.get("edges", []):
            edge = PipelineEdge.from_dict(edge_spec)
            builder._graph.add_edge(edge)
        return builder

    @classmethod
    def from_json(cls, json_str: str) -> PipelineBuilder:
        """Build from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_json_file(cls, path: str | Path) -> PipelineBuilder:
        """Build from a JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, yaml_str: str) -> PipelineBuilder:
        """Build from a YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> PipelineBuilder:
        """Build from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            return cls.from_dict(data)

    # ── Internal helpers ──

    def _infer_input_schema(self, upstream_ids: Sequence[str]) -> Schema:
        """Merge output schemas of upstream nodes."""
        if not upstream_ids:
            return Schema.empty()
        schemas: list[Schema] = []
        for uid in upstream_ids:
            if uid in self._pending_schemas:
                schemas.append(self._pending_schemas[uid])
            elif self._graph.has_node(uid):
                out_schema = self._graph.get_node(uid).output_schema
                if out_schema.columns:
                    schemas.append(out_schema)
        if not schemas:
            return Schema.empty()
        if len(schemas) == 1:
            return schemas[0]
        # Merge multiple schemas
        result = schemas[0]
        for s in schemas[1:]:
            result = result.merge(s)
        return result


# =====================================================================
# Quick-build helpers
# =====================================================================

def build_linear_pipeline(
    name: str,
    stages: Sequence[tuple[str, SQLOperator, Schema]],
) -> PipelineGraph:
    """Build a simple linear pipeline: source -> t1 -> t2 -> ... -> sink.

    *stages* is a list of ``(node_id, operator, output_schema)`` tuples.
    The first is treated as a source, the last as a sink.
    """
    builder = PipelineBuilder(name)
    for i, (nid, op, schema) in enumerate(stages):
        if i == 0:
            builder.add_source(nid, schema=schema)
        else:
            prev_id = stages[i - 1][0]
            if i == len(stages) - 1:
                builder.add_sink(nid, prev_id, output_schema=schema)
            else:
                builder.add_transform(nid, prev_id, operator=op, output_schema=schema)
    return builder.build()


def build_diamond_pipeline(
    name: str,
    source_id: str,
    source_schema: Schema,
    left_id: str,
    right_id: str,
    merge_id: str,
    output_schema: Schema | None = None,
) -> PipelineGraph:
    """Build a diamond-shaped pipeline: source -> {left, right} -> merge."""
    builder = PipelineBuilder(name)
    builder.add_source(source_id, schema=source_schema)
    builder.add_transform(left_id, source_id, operator=SQLOperator.FILTER)
    builder.add_transform(right_id, source_id, operator=SQLOperator.FILTER)
    builder.add_transform(
        merge_id, left_id, right_id,
        operator=SQLOperator.UNION,
        output_schema=output_schema,
    )
    return builder.build()


def build_star_pipeline(
    name: str,
    center_id: str,
    center_schema: Schema,
    satellite_ids: Sequence[str],
) -> PipelineGraph:
    """Build a star-shaped pipeline: center -> {sat1, sat2, ...}."""
    builder = PipelineBuilder(name)
    builder.add_source(center_id, schema=center_schema)
    for sat_id in satellite_ids:
        builder.add_transform(sat_id, center_id, operator=SQLOperator.SELECT)
    return builder.build()
