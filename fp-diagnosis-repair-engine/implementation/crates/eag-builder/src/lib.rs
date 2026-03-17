//! Error Amplification Graph (EAG) builder for Penumbra.
//!
//! Constructs a directed graph that models how floating-point errors propagate
//! and amplify through a computation pipeline.

use penumbra_types::{FpOperation, OpId, SourceSpan};
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};

/// Node data in the Error Amplification Graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EagNode {
    /// Unique operation identifier.
    pub op_id: OpId,
    /// The floating-point operation at this node.
    pub operation: FpOperation,
    /// The value computed in standard precision.
    pub computed_value: f64,
    /// The value computed in higher (shadow) precision.
    pub shadow_value: f64,
    /// Absolute error: |computed - shadow|.
    pub error: f64,
    /// Relative error: |computed - shadow| / |shadow|.
    pub relative_error: f64,
    /// Optional source location.
    pub source_location: Option<SourceSpan>,
    /// Locally introduced error (not inherited from inputs).
    pub local_error: f64,
}

/// Edge data in the Error Amplification Graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EagEdge {
    /// How much the source node's error is amplified at the target.
    pub amplification_factor: f64,
    /// Absolute error contribution along this edge.
    pub error_contribution: f64,
    /// Which input position of the target this edge feeds (0-indexed).
    pub input_index: usize,
}

/// The Error Amplification Graph tracks error propagation through a computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAmplificationGraph {
    /// The underlying directed graph.
    pub graph: DiGraph<EagNode, EagEdge>,
}

impl ErrorAmplificationGraph {
    /// Create a new empty EAG.
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
        }
    }

    /// Add an operation node and return its index.
    pub fn add_node(&mut self, node: EagNode) -> NodeIndex {
        self.graph.add_node(node)
    }

    /// Add a directed edge (data-flow dependency) between two nodes.
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, edge: EagEdge) {
        self.graph.add_edge(from, to, edge);
    }

    /// Number of operation nodes.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of data-flow edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get the node data for a given index.
    pub fn node_weight(&self, idx: NodeIndex) -> Option<&EagNode> {
        self.graph.node_weight(idx)
    }

    /// Get a mutable reference to the node data.
    pub fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut EagNode> {
        self.graph.node_weight_mut(idx)
    }

    /// Iterate over all node indices.
    pub fn node_indices(&self) -> petgraph::graph::NodeIndices<EagNode> {
        self.graph.node_indices()
    }

    /// Get predecessors (nodes that feed into the given node).
    pub fn predecessors(&self, idx: NodeIndex) -> Vec<NodeIndex> {
        self.graph
            .neighbors_directed(idx, petgraph::Direction::Incoming)
            .collect()
    }

    /// Get successors (nodes that the given node feeds into).
    pub fn successors(&self, idx: NodeIndex) -> Vec<NodeIndex> {
        self.graph
            .neighbors_directed(idx, petgraph::Direction::Outgoing)
            .collect()
    }

    /// Find nodes with no incoming edges (inputs/constants).
    pub fn source_nodes(&self) -> Vec<NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .neighbors_directed(idx, petgraph::Direction::Incoming)
                    .next()
                    .is_none()
            })
            .collect()
    }

    /// Find nodes with no outgoing edges (final outputs).
    pub fn sink_nodes(&self) -> Vec<NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .neighbors_directed(idx, petgraph::Direction::Outgoing)
                    .next()
                    .is_none()
            })
            .collect()
    }

    /// Get all edges from `from` to `to`.
    pub fn edges_between(&self, from: NodeIndex, to: NodeIndex) -> Vec<&EagEdge> {
        self.graph
            .edges_connecting(from, to)
            .map(|e| e.weight())
            .collect()
    }

    /// Topological sort of nodes. Returns `None` if the graph has a cycle.
    pub fn topological_sort(&self) -> Option<Vec<NodeIndex>> {
        petgraph::algo::toposort(&self.graph, None).ok()
    }
}

impl Default for ErrorAmplificationGraph {
    fn default() -> Self {
        Self::new()
    }
}
