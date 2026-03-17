"""
taintflow.dag – Pipeline Information DAG (PI-DAG) construction and analysis.

This package provides the core data structures for representing ML pipelines
as directed acyclic graphs where nodes are operations and edges carry data-flow
provenance.  The PI-DAG is the central artefact consumed by the abstract
interpretation engine, capacity estimator, and attribution modules.

Public API
----------
Nodes & edges::

    PipelineNode, SourceLocation
    DataSourceNode, PartitionNode, TransformNode, FitNode, PredictNode
    PandasOpNode, AggregationNode, FeatureEngineeringNode, SelectionNode
    CustomNode, SinkNode
    NodeFactory
    PipelineEdge, EdgeSet

DAG::

    PIDAG

Construction::

    DAGBuilder, TraceEvent, OperationLog

Traversal & analysis::

    DAGTraversal, PathEnumerator, DependencyAnalyzer, StageIdentifier, Stage

Serialization::

    serialize_pidag_json, deserialize_pidag_json
    serialize_pidag_msgpack, deserialize_pidag_msgpack

Visualization::

    to_dot, to_mermaid, to_ascii, DAGRenderer
"""

from __future__ import annotations

from taintflow.dag.node import (
    PipelineNode,
    SourceLocation,
    DataSourceNode,
    PartitionNode,
    TransformNode,
    FitNode,
    PredictNode,
    PandasOpNode,
    AggregationNode,
    FeatureEngineeringNode,
    SelectionNode,
    CustomNode,
    SinkNode,
    NodeFactory,
)
from taintflow.dag.edge import (
    PipelineEdge,
    EdgeSet,
)
from taintflow.dag.pidag import PIDAG
from taintflow.dag.builder import DAGBuilder, TraceEvent, OperationLog
from taintflow.dag.traversal import (
    bfs_forward,
    bfs_backward,
    dfs_forward,
    dfs_backward,
    ReachabilityAnalyzer,
    PathFinder,
    DominatorTree,
    DominatorResult,
    SlicingEngine,
    LoopDetector,
    CycleReport,
    SubgraphExtractor,
    LevelAssigner,
    LevelAssignment,
    CriticalPathAnalyzer,
    CriticalPathResult,
)
from taintflow.dag.serialization import (
    serialize_pidag_json,
    deserialize_pidag_json,
    serialize_pidag_msgpack,
    deserialize_pidag_msgpack,
)
from taintflow.dag.visualization import to_dot, to_mermaid, to_ascii, DAGRenderer

__all__ = [
    # nodes
    "PipelineNode",
    "SourceLocation",
    "DataSourceNode",
    "PartitionNode",
    "TransformNode",
    "FitNode",
    "PredictNode",
    "PandasOpNode",
    "AggregationNode",
    "FeatureEngineeringNode",
    "SelectionNode",
    "CustomNode",
    "SinkNode",
    "NodeFactory",
    # edges
    "PipelineEdge",
    "EdgeSet",
    # dag
    "PIDAG",
    # builder
    "DAGBuilder",
    "TraceEvent",
    "OperationLog",
    # traversal
    "DAGTraversal",
    "PathEnumerator",
    "DependencyAnalyzer",
    "StageIdentifier",
    "Stage",
    # serialization
    "serialize_pidag_json",
    "deserialize_pidag_json",
    "serialize_pidag_msgpack",
    "deserialize_pidag_msgpack",
    # visualization
    "to_dot",
    "to_mermaid",
    "to_ascii",
    "DAGRenderer",
]
