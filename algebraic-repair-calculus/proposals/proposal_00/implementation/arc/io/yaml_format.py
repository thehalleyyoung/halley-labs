"""
YAML serialization for ARC pipeline specifications.

Supports YAML anchors and aliases for DRY configurations, environment
variable interpolation (``${VAR}``), and template patterns for common
pipeline structures.
"""

from __future__ import annotations

import attr
import os
import re
from pathlib import Path
from typing import Any

import yaml

from arc.types.errors import (
    ParseError,
    SchemaViolationError,
    SerializationError,
    ErrorCode,
)
from arc.graph.pipeline import PipelineGraph
from arc.io.json_format import PipelineSpec, SUPPORTED_SPEC_VERSIONS


# =====================================================================
# Environment variable interpolation
# =====================================================================

_ENV_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _interpolate_env(value: str) -> str:
    """Replace ``${VAR}`` and ``${VAR:default}`` with environment values."""
    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        if default is not None:
            return default
        return match.group(0)  # leave unreplaced if no default

    return _ENV_PATTERN.sub(_replace, value)


def _deep_interpolate(obj: Any) -> Any:
    """Recursively interpolate environment variables in strings."""
    if isinstance(obj, str):
        return _interpolate_env(obj)
    if isinstance(obj, dict):
        return {k: _deep_interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_interpolate(item) for item in obj]
    return obj


# =====================================================================
# YAML loader with custom tags
# =====================================================================

class _ARCYAMLLoader(yaml.SafeLoader):
    """Extended YAML loader with ARC-specific tags."""
    pass


def _include_constructor(loader: yaml.Loader, node: yaml.Node) -> Any:
    """Handle ``!include path/to/file.yaml`` tags."""
    filepath = loader.construct_scalar(node)
    # Resolve relative to the current file
    if hasattr(loader, "_root_dir"):
        filepath = os.path.join(loader._root_dir, filepath)  # type: ignore[attr-defined]
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def _env_constructor(loader: yaml.Loader, node: yaml.Node) -> str:
    """Handle ``!env VAR_NAME`` tags."""
    var_name = loader.construct_scalar(node)
    return os.environ.get(var_name, "")


def _env_default_constructor(loader: yaml.Loader, node: yaml.Node) -> str:
    """Handle ``!env_default [VAR_NAME, default_value]`` tags."""
    args = loader.construct_sequence(node)
    if len(args) < 2:
        return os.environ.get(args[0], "")
    return os.environ.get(args[0], str(args[1]))


_ARCYAMLLoader.add_constructor("!include", _include_constructor)
_ARCYAMLLoader.add_constructor("!env", _env_constructor)
_ARCYAMLLoader.add_constructor("!env_default", _env_default_constructor)


# =====================================================================
# YAML pipeline spec
# =====================================================================

class YAMLPipelineSpec:
    """YAML pipeline specification with anchors, aliases, environment
    interpolation, and template support.

    A YAML pipeline spec is structured as::

        version: "1.0"
        name: my_pipeline
        metadata:
          owner: data-team
        nodes:
          - node_id: source_users
            operator: SOURCE
            output_schema:
              columns:
                - name: id
                  sql_type: {base: INT}
        edges:
          - source: source_users
            target: clean_users
    """

    @staticmethod
    def load(path: str | Path) -> PipelineGraph:
        """Load a pipeline graph from a YAML file.

        Supports anchors/aliases, ``!include``, ``!env``, and
        ``${VAR:default}`` interpolation.
        """
        p = Path(path)
        if not p.exists():
            raise SerializationError(
                f"File not found: {p}",
                code=ErrorCode.SERIALIZATION_FILE_NOT_FOUND,
            )
        try:
            loader = _ARCYAMLLoader
            loader._root_dir = str(p.parent)  # type: ignore[attr-defined]
            with open(p, "r") as f:
                data = yaml.load(f, Loader=loader)
        except yaml.YAMLError as e:
            line = None
            if hasattr(e, "problem_mark") and e.problem_mark:
                line = e.problem_mark.line + 1
            raise ParseError(str(p), str(e), line=line)

        # Interpolate environment variables
        data = _deep_interpolate(data)
        return PipelineSpec.from_dict(data)

    @staticmethod
    def from_yaml(yaml_str: str) -> PipelineGraph:
        """Load a pipeline graph from a YAML string."""
        try:
            data = yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            line = None
            if hasattr(e, "problem_mark") and e.problem_mark:
                line = e.problem_mark.line + 1
            raise ParseError("<string>", str(e), line=line)
        data = _deep_interpolate(data)
        return PipelineSpec.from_dict(data)

    @staticmethod
    def save(graph: PipelineGraph, path: str | Path) -> None:
        """Save a pipeline graph to a YAML file."""
        data = PipelineSpec.to_dict(graph)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @staticmethod
    def to_yaml(graph: PipelineGraph) -> str:
        """Serialize a pipeline graph to a YAML string."""
        data = PipelineSpec.to_dict(graph)
        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)


# =====================================================================
# Pipeline templates
# =====================================================================

_TEMPLATES: dict[str, str] = {
    "etl_basic": """\
version: "1.0"
name: "{name}"
metadata:
  template: etl_basic
  description: "Basic ETL pipeline: extract -> transform -> load"
nodes:
  - node_id: extract
    operator: SOURCE
    output_schema:
      columns: []
  - node_id: transform
    operator: TRANSFORM
  - node_id: load
    operator: SINK
edges:
  - source: extract
    target: transform
  - source: transform
    target: load
""",
    "star_schema": """\
version: "1.0"
name: "{name}"
metadata:
  template: star_schema
  description: "Star schema pipeline with fact table and dimension tables"
nodes:
  - node_id: raw_fact
    operator: SOURCE
  - node_id: raw_dim_1
    operator: SOURCE
  - node_id: raw_dim_2
    operator: SOURCE
  - node_id: stage_fact
    operator: TRANSFORM
  - node_id: stage_dim_1
    operator: TRANSFORM
  - node_id: stage_dim_2
    operator: TRANSFORM
  - node_id: fact_table
    operator: JOIN
  - node_id: output
    operator: SINK
edges:
  - source: raw_fact
    target: stage_fact
  - source: raw_dim_1
    target: stage_dim_1
  - source: raw_dim_2
    target: stage_dim_2
  - source: stage_fact
    target: fact_table
  - source: stage_dim_1
    target: fact_table
  - source: stage_dim_2
    target: fact_table
  - source: fact_table
    target: output
""",
    "cdc_pipeline": """\
version: "1.0"
name: "{name}"
metadata:
  template: cdc_pipeline
  description: "Change Data Capture pipeline with dedup and merge"
nodes:
  - node_id: cdc_source
    operator: SOURCE
  - node_id: dedup
    operator: DEDUP
  - node_id: transform
    operator: TRANSFORM
  - node_id: merge_target
    operator: MERGE
  - node_id: quality_check
    operator: TRANSFORM
  - node_id: target
    operator: SINK
edges:
  - source: cdc_source
    target: dedup
  - source: dedup
    target: transform
  - source: transform
    target: merge_target
  - source: merge_target
    target: quality_check
  - source: quality_check
    target: target
""",
    "diamond": """\
version: "1.0"
name: "{name}"
metadata:
  template: diamond
  description: "Diamond-shaped pipeline: source -> branches -> merge"
nodes:
  - node_id: source
    operator: SOURCE
  - node_id: branch_a
    operator: FILTER
  - node_id: branch_b
    operator: FILTER
  - node_id: merge
    operator: UNION
  - node_id: output
    operator: SINK
edges:
  - source: source
    target: branch_a
  - source: source
    target: branch_b
  - source: branch_a
    target: merge
  - source: branch_b
    target: merge
  - source: merge
    target: output
""",
}


def from_template(template_name: str, name: str = "") -> PipelineGraph:
    """Create a pipeline graph from a named template.

    Available templates: ``etl_basic``, ``star_schema``,
    ``cdc_pipeline``, ``diamond``.
    """
    if template_name not in _TEMPLATES:
        available = ", ".join(sorted(_TEMPLATES.keys()))
        raise SerializationError(
            f"Unknown template '{template_name}'; available: {available}",
            code=ErrorCode.SERIALIZATION_FAILURE,
        )
    yaml_str = _TEMPLATES[template_name].format(name=name or template_name)
    return YAMLPipelineSpec.from_yaml(yaml_str)


def list_templates() -> list[str]:
    """Return available template names."""
    return sorted(_TEMPLATES.keys())


def get_template_yaml(template_name: str, name: str = "") -> str:
    """Return the raw YAML for a named template."""
    if template_name not in _TEMPLATES:
        raise SerializationError(
            f"Unknown template '{template_name}'",
            code=ErrorCode.SERIALIZATION_FAILURE,
        )
    return _TEMPLATES[template_name].format(name=name or template_name)


# =====================================================================
# YAML document merger
# =====================================================================

class YAMLMerger:
    """Merge multiple YAML pipeline definitions into one.

    Handles node ID collisions by prefixing with the source document
    name.  Edges referencing renamed nodes are updated automatically.
    """

    @staticmethod
    def merge(
        specs: list[tuple[str, str | Path]],
        merged_name: str = "merged",
    ) -> "PipelineGraph":
        """Merge multiple YAML specs into a single graph.

        Parameters
        ----------
        specs : list[(label, path)]
            List of ``(label, path)`` tuples.  ``label`` is used as a
            prefix when resolving node-id collisions.
        merged_name : str
            Name for the merged pipeline.
        """
        from arc.graph.pipeline import PipelineGraph

        graphs: list[tuple[str, "PipelineGraph"]] = []
        for label, path in specs:
            g = YAMLPipelineSpec.load(path)
            graphs.append((label, g))

        if not graphs:
            return PipelineGraph(name=merged_name)

        base = graphs[0][1].clone()
        base._name = merged_name  # type: ignore[attr-defined]

        seen_ids: set[str] = {n.node_id for n in base.nodes}

        for label, g in graphs[1:]:
            rename_map: dict[str, str] = {}
            for node in g.nodes:
                nid = node.node_id
                if nid in seen_ids:
                    new_id = f"{label}_{nid}"
                    rename_map[nid] = new_id
                else:
                    rename_map[nid] = nid
                seen_ids.add(rename_map[nid])

            for node in g.nodes:
                new_id = rename_map[node.node_id]
                new_node = attr.evolve(node, node_id=new_id)
                base.add_node(new_node)

            for edge in g.edges:
                src = rename_map.get(edge.source_id, edge.source_id)
                tgt = rename_map.get(edge.target_id, edge.target_id)
                new_edge = attr.evolve(edge, source_id=src, target_id=tgt)
                base.add_edge(new_edge)

        return base

    @staticmethod
    def merge_yaml_strings(
        docs: list[tuple[str, str]],
        merged_name: str = "merged",
    ) -> "PipelineGraph":
        """Merge YAML pipeline definitions given as raw strings."""
        from arc.graph.pipeline import PipelineGraph

        graphs: list[tuple[str, "PipelineGraph"]] = []
        for label, yaml_str in docs:
            g = YAMLPipelineSpec.from_yaml(yaml_str)
            graphs.append((label, g))

        if not graphs:
            return PipelineGraph(name=merged_name)

        base = graphs[0][1].clone()
        base._name = merged_name  # type: ignore[attr-defined]

        seen_ids = {n.node_id for n in base.nodes}

        for label, g in graphs[1:]:
            rename_map: dict[str, str] = {}
            for node in g.nodes:
                nid = node.node_id
                new_id = f"{label}_{nid}" if nid in seen_ids else nid
                rename_map[nid] = new_id
                seen_ids.add(new_id)

            for node in g.nodes:
                new_node = attr.evolve(node, node_id=rename_map[node.node_id])
                base.add_node(new_node)

            for edge in g.edges:
                src = rename_map.get(edge.source_id, edge.source_id)
                tgt = rename_map.get(edge.target_id, edge.target_id)
                base.add_edge(attr.evolve(edge, source_id=src, target_id=tgt))

        return base


# =====================================================================
# YAML diff
# =====================================================================

class YAMLDiff:
    """Compute a human-readable diff between two YAML pipeline specs."""

    @staticmethod
    def diff(path_a: str | Path, path_b: str | Path) -> dict[str, Any]:
        """Return a structured diff between two YAML pipeline files.

        Returns a dict with keys ``added_nodes``, ``removed_nodes``,
        ``added_edges``, ``removed_edges``, and ``changed_nodes``.
        """
        g_a = YAMLPipelineSpec.load(path_a)
        g_b = YAMLPipelineSpec.load(path_b)

        ids_a = {n.node_id for n in g_a.nodes}
        ids_b = {n.node_id for n in g_b.nodes}

        added_nodes = sorted(ids_b - ids_a)
        removed_nodes = sorted(ids_a - ids_b)
        common_nodes = ids_a & ids_b

        changed_nodes: list[dict[str, Any]] = []
        for nid in sorted(common_nodes):
            na = g_a.get_node(nid)
            nb = g_b.get_node(nid)
            diffs: dict[str, Any] = {}
            if na.operator != nb.operator:
                diffs["operator"] = {
                    "old": na.operator.value if na.operator else None,
                    "new": nb.operator.value if nb.operator else None,
                }
            if na.output_schema != nb.output_schema:
                diffs["output_schema"] = "changed"
            if na.cost != nb.cost:
                diffs["cost"] = "changed"
            if diffs:
                changed_nodes.append({"node_id": nid, **diffs})

        edges_a = {(e.source_id, e.target_id) for e in g_a.edges}
        edges_b = {(e.source_id, e.target_id) for e in g_b.edges}

        return {
            "added_nodes": added_nodes,
            "removed_nodes": removed_nodes,
            "changed_nodes": changed_nodes,
            "added_edges": sorted(edges_b - edges_a),
            "removed_edges": sorted(edges_a - edges_b),
        }

    @staticmethod
    def diff_yaml(yaml_a: str, yaml_b: str) -> dict[str, Any]:
        """Diff two YAML strings directly."""
        g_a = YAMLPipelineSpec.from_yaml(yaml_a)
        g_b = YAMLPipelineSpec.from_yaml(yaml_b)

        ids_a = {n.node_id for n in g_a.nodes}
        ids_b = {n.node_id for n in g_b.nodes}

        added = sorted(ids_b - ids_a)
        removed = sorted(ids_a - ids_b)

        edges_a = {(e.source_id, e.target_id) for e in g_a.edges}
        edges_b = {(e.source_id, e.target_id) for e in g_b.edges}

        return {
            "added_nodes": added,
            "removed_nodes": removed,
            "added_edges": sorted(edges_b - edges_a),
            "removed_edges": sorted(edges_a - edges_b),
        }
