//! Error Amplification Graph (EAG) types.
//!
//! The EAG is the central data structure in Penumbra: a weighted directed
//! acyclic graph where nodes represent floating-point operations and edges
//! carry first-order error-flow magnitudes (∂ε̂ⱼ/∂ε̂ᵢ).
//!
//! This module defines the node/edge types, the graph container, and
//! associated query methods.  Construction is handled by `fpdiag-analysis`;
//! this module provides only the types and read-side API.

use crate::diagnosis::DiagnosisCategory;
use crate::expression::FpOp;
use crate::source::SourceSpan;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ─── Identifiers ────────────────────────────────────────────────────────────

/// Opaque identifier for an EAG node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EagNodeId(pub u32);

impl fmt::Display for EagNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "eag_n{}", self.0)
    }
}

/// Opaque identifier for an EAG edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EagEdgeId(pub u32);

impl fmt::Display for EagEdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "eag_e{}", self.0)
    }
}

// ─── EagNode ────────────────────────────────────────────────────────────────

/// A node in the Error Amplification Graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EagNode {
    /// Unique identifier.
    pub id: EagNodeId,
    /// The floating-point operation at this node.
    pub op: FpOp,
    /// Where in the source this operation lives (if known).
    pub source: Option<SourceSpan>,
    /// Computed value in f64.
    pub computed_value: f64,
    /// Shadow (high-precision) value.
    pub shadow_value: f64,
    /// Local rounding error at this node: |computed − shadow|.
    pub local_error: f64,
    /// Relative local error: local_error / |shadow_value|.
    pub relative_error: f64,
    /// Condition number of this operation (if computable).
    pub condition_number: Option<f64>,
    /// Diagnosed root-cause category (filled after diagnosis pass).
    pub diagnosis: Option<DiagnosisCategory>,
    /// Whether this is a Tier-2 black-box node.
    pub is_black_box: bool,
    /// Optional human-readable label.
    pub label: Option<String>,
}

impl EagNode {
    /// Create a new EAG node.
    pub fn new(id: EagNodeId, op: FpOp, computed: f64, shadow: f64) -> Self {
        let local_error = (computed - shadow).abs();
        let relative_error = if shadow.abs() > 0.0 {
            local_error / shadow.abs()
        } else {
            0.0
        };
        Self {
            id,
            op,
            source: None,
            computed_value: computed,
            shadow_value: shadow,
            local_error,
            relative_error,
            condition_number: None,
            diagnosis: None,
            is_black_box: op.is_black_box(),
            label: None,
        }
    }

    /// Whether the local error exceeds the given threshold (in ULPs of f64 machine epsilon).
    pub fn is_high_error(&self, ulp_threshold: f64) -> bool {
        let eps = f64::EPSILON;
        self.local_error > ulp_threshold * eps * self.shadow_value.abs().max(f64::MIN_POSITIVE)
    }
}

// ─── EagEdge ────────────────────────────────────────────────────────────────

/// An edge in the Error Amplification Graph.
///
/// Represents error flow from `source` to `target` with weight
/// |∂ε̂_target / ∂ε̂_source|, the first-order sensitivity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EagEdge {
    /// Unique identifier.
    pub id: EagEdgeId,
    /// Source node (error flows *from* here).
    pub source: EagNodeId,
    /// Target node (error flows *to* here).
    pub target: EagNodeId,
    /// Edge weight: |∂ε̂_target / ∂ε̂_source|.
    pub weight: OrderedFloat<f64>,
    /// How the weight was computed.
    pub weight_method: WeightMethod,
}

/// How an EAG edge weight was computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightMethod {
    /// Central finite differencing of shadow values.
    FiniteDifference,
    /// Analytic derivative (for known operations).
    Analytic,
    /// Black-box input/output comparison.
    BlackBoxComparison,
}

impl EagEdge {
    /// Create a new edge.
    pub fn new(id: EagEdgeId, source: EagNodeId, target: EagNodeId, weight: f64) -> Self {
        Self {
            id,
            source,
            target,
            weight: OrderedFloat(weight.abs()),
            weight_method: WeightMethod::FiniteDifference,
        }
    }

    /// Whether this edge amplifies error (weight > 1).
    pub fn is_amplifying(&self) -> bool {
        self.weight.0 > 1.0
    }

    /// Whether this edge attenuates error (weight < 1).
    pub fn is_attenuating(&self) -> bool {
        self.weight.0 < 1.0
    }
}

// ─── ErrorAmplificationGraph ────────────────────────────────────────────────

/// The Error Amplification Graph: the central representation for causal
/// error-flow analysis in Penumbra.
///
/// A weighted DAG where:
/// - Nodes are FP operations with local error annotations
/// - Edges carry first-order error-flow sensitivities
/// - Path-weight products bound error propagation (Theorem T1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAmplificationGraph {
    nodes: Vec<EagNode>,
    edges: Vec<EagEdge>,
    /// Adjacency list: node → outgoing edge ids.
    adjacency: HashMap<EagNodeId, Vec<EagEdgeId>>,
    /// Reverse adjacency: node → incoming edge ids.
    reverse_adjacency: HashMap<EagNodeId, Vec<EagEdgeId>>,
}

impl ErrorAmplificationGraph {
    /// Create an empty EAG.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
            reverse_adjacency: HashMap::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: EagNode) -> EagNodeId {
        let id = node.id;
        self.adjacency.entry(id).or_default();
        self.reverse_adjacency.entry(id).or_default();
        self.nodes.push(node);
        id
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, edge: EagEdge) -> EagEdgeId {
        let id = edge.id;
        self.adjacency.entry(edge.source).or_default().push(id);
        self.reverse_adjacency
            .entry(edge.target)
            .or_default()
            .push(id);
        self.edges.push(edge);
        id
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get a node by id.
    pub fn node(&self, id: EagNodeId) -> Option<&EagNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Get a mutable reference to a node by id.
    pub fn node_mut(&mut self, id: EagNodeId) -> Option<&mut EagNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// Get an edge by id.
    pub fn edge(&self, id: EagEdgeId) -> Option<&EagEdge> {
        self.edges.iter().find(|e| e.id == id)
    }

    /// Iterate over all nodes.
    pub fn nodes(&self) -> &[EagNode] {
        &self.nodes
    }

    /// Iterate over all edges.
    pub fn edges(&self) -> &[EagEdge] {
        &self.edges
    }

    /// Outgoing edges from a node.
    pub fn outgoing(&self, node: EagNodeId) -> Vec<&EagEdge> {
        self.adjacency
            .get(&node)
            .map(|ids| ids.iter().filter_map(|id| self.edge(*id)).collect())
            .unwrap_or_default()
    }

    /// Incoming edges to a node.
    pub fn incoming(&self, node: EagNodeId) -> Vec<&EagEdge> {
        self.reverse_adjacency
            .get(&node)
            .map(|ids| ids.iter().filter_map(|id| self.edge(*id)).collect())
            .unwrap_or_default()
    }

    /// Source nodes (no incoming edges).
    pub fn sources(&self) -> Vec<&EagNode> {
        self.nodes
            .iter()
            .filter(|n| {
                self.reverse_adjacency
                    .get(&n.id)
                    .map_or(true, |v| v.is_empty())
            })
            .collect()
    }

    /// Sink nodes (no outgoing edges).
    pub fn sinks(&self) -> Vec<&EagNode> {
        self.nodes
            .iter()
            .filter(|n| self.adjacency.get(&n.id).map_or(true, |v| v.is_empty()))
            .collect()
    }

    /// Find nodes with local error above a threshold.
    pub fn high_error_nodes(&self, ulp_threshold: f64) -> Vec<&EagNode> {
        self.nodes
            .iter()
            .filter(|n| n.is_high_error(ulp_threshold))
            .collect()
    }

    /// Compute the maximum path weight product from any source to the given sink.
    /// This provides an upper bound on error amplification (Theorem T1).
    pub fn max_path_weight_to(&self, sink: EagNodeId) -> f64 {
        let mut max_weight: HashMap<EagNodeId, f64> = HashMap::new();
        // Initialize sources with weight 1.0
        for src in self.sources() {
            max_weight.insert(src.id, 1.0);
        }
        // Topological order traversal (simplified: iterate nodes in order)
        for node in &self.nodes {
            let incoming = self.incoming(node.id);
            if !incoming.is_empty() {
                let best = incoming
                    .iter()
                    .filter_map(|e| max_weight.get(&e.source).map(|w| w * e.weight.0))
                    .fold(0.0_f64, f64::max);
                let current = max_weight.entry(node.id).or_insert(0.0);
                if best > *current {
                    *current = best;
                }
            }
        }
        max_weight.get(&sink).copied().unwrap_or(0.0)
    }

    /// Total attributed error at a sink using T1 bound:
    /// Σ over source-to-sink paths of (Π edge weights) × source local error.
    pub fn t1_error_bound(&self, sink: EagNodeId) -> f64 {
        let mut attributed: HashMap<EagNodeId, f64> = HashMap::new();
        for src in self.sources() {
            attributed.insert(src.id, src.local_error);
        }
        for node in &self.nodes {
            let incoming = self.incoming(node.id);
            if !incoming.is_empty() {
                let propagated: f64 = incoming
                    .iter()
                    .filter_map(|e| attributed.get(&e.source).map(|err| err * e.weight.0))
                    .sum();
                let entry = attributed.entry(node.id).or_insert(0.0);
                *entry += propagated + node.local_error;
            }
        }
        attributed.get(&sink).copied().unwrap_or(0.0)
    }
}

impl Default for ErrorAmplificationGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::FpOp;

    #[test]
    fn empty_eag() {
        let eag = ErrorAmplificationGraph::new();
        assert_eq!(eag.node_count(), 0);
        assert_eq!(eag.edge_count(), 0);
    }

    #[test]
    fn linear_chain_t1_bound() {
        let mut eag = ErrorAmplificationGraph::new();
        let n0 = EagNode::new(EagNodeId(0), FpOp::Add, 1.0000001, 1.0);
        let n1 = EagNode::new(EagNodeId(1), FpOp::Mul, 2.0000004, 2.0);
        eag.add_node(n0);
        eag.add_node(n1);
        eag.add_edge(EagEdge::new(EagEdgeId(0), EagNodeId(0), EagNodeId(1), 2.0));
        let bound = eag.t1_error_bound(EagNodeId(1));
        assert!(bound > 0.0);
    }
}
