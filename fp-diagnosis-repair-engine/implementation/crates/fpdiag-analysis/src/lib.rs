//! # fpdiag-analysis
//!
//! Error Amplification Graph (EAG) construction and analysis for Penumbra.
//!
//! This crate is responsible for:
//! - Building EAGs from execution traces
//! - Computing edge weights via finite differencing of shadow values
//! - Sparsifying the graph by removing negligible edges
//! - Path-weight analysis (Theorem T1 bounds)
//! - Treewidth estimation
//! - Adaptive solving: tree-decomposition for low-treewidth EAGs,
//!   SCC-based interval fallback for high-treewidth EAGs

pub mod high_treewidth;

use fpdiag_types::{
    config::EagConfig,
    eag::{EagEdge, EagEdgeId, EagNode, EagNodeId, ErrorAmplificationGraph, WeightMethod},
    expression::FpOp,
    trace::{ExecutionTrace, TraceEvent},
};
use std::collections::HashMap;
use thiserror::Error;

/// Errors from the analysis module.
#[derive(Debug, Error)]
pub enum AnalysisError {
    #[error("empty trace: no events to analyze")]
    EmptyTrace,
    #[error("invalid finite-difference step: {step}")]
    InvalidStep { step: f64 },
    #[error("cycle detected in EAG (trace event {seq})")]
    CycleDetected { seq: u64 },
    #[error("node not found: {0}")]
    NodeNotFound(EagNodeId),
}

/// Builder for constructing an EAG from an execution trace.
///
/// Uses streaming construction: processes trace events one at a time,
/// emitting EAG nodes and edges incrementally.
pub struct EagBuilder {
    config: EagConfig,
    graph: ErrorAmplificationGraph,
    next_node_id: u32,
    next_edge_id: u32,
    /// Map from trace event seq to EAG node id.
    seq_to_node: HashMap<u64, EagNodeId>,
    /// Tracks the last output node to establish data-flow edges.
    last_outputs: HashMap<String, EagNodeId>,
}

impl EagBuilder {
    /// Create a new builder with the given configuration.
    pub fn new(config: EagConfig) -> Self {
        Self {
            config,
            graph: ErrorAmplificationGraph::new(),
            next_node_id: 0,
            next_edge_id: 0,
            seq_to_node: HashMap::new(),
            last_outputs: HashMap::new(),
        }
    }

    /// Create a builder with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(EagConfig::default())
    }

    /// Process a single trace event.
    pub fn process_event(
        &mut self,
        event: &TraceEvent,
    ) -> Result<Option<EagNodeId>, AnalysisError> {
        match event {
            TraceEvent::Operation {
                seq,
                op,
                inputs,
                output,
                shadow_output,
                source,
                ..
            } => {
                let node_id = EagNodeId(self.next_node_id);
                self.next_node_id += 1;

                let mut node = EagNode::new(node_id, *op, *output, *shadow_output);
                node.source = source.clone();

                // Compute condition number for this operation
                node.condition_number = self.estimate_condition_number(*op, inputs, *output);

                self.graph.add_node(node);
                self.seq_to_node.insert(*seq, node_id);

                // Create edges from predecessor nodes based on data flow
                // In a real implementation, this would track per-value provenance.
                // Here we use a simplified heuristic: connect to recent operations.
                if let Some(prev_id) = self.last_outputs.get("__last__") {
                    let weight = self.compute_edge_weight(*output, *shadow_output, *prev_id)?;
                    if weight.abs() >= self.config.min_edge_weight {
                        let edge_id = EagEdgeId(self.next_edge_id);
                        self.next_edge_id += 1;
                        let edge = EagEdge::new(edge_id, *prev_id, node_id, weight);
                        self.graph.add_edge(edge);
                    }
                }

                self.last_outputs.insert("__last__".to_string(), node_id);
                Ok(Some(node_id))
            }
            TraceEvent::LibraryCall {
                seq,
                function,
                output_error,
                amplification,
                source,
                ..
            } => {
                let node_id = EagNodeId(self.next_node_id);
                self.next_node_id += 1;

                let mut node = EagNode::new(node_id, FpOp::BlackBox, *output_error, 0.0);
                node.source = source.clone();
                node.is_black_box = true;
                node.label = Some(function.clone());
                node.condition_number = Some(*amplification);

                self.graph.add_node(node);
                self.seq_to_node.insert(*seq, node_id);

                // Connect to previous node
                if let Some(prev_id) = self.last_outputs.get("__last__") {
                    let edge_id = EagEdgeId(self.next_edge_id);
                    self.next_edge_id += 1;
                    let mut edge = EagEdge::new(edge_id, *prev_id, node_id, *amplification);
                    edge.weight_method = WeightMethod::BlackBoxComparison;
                    self.graph.add_edge(edge);
                }

                self.last_outputs.insert("__last__".to_string(), node_id);
                Ok(Some(node_id))
            }
            _ => Ok(None), // Region enter/exit and annotations don't produce nodes
        }
    }

    /// Estimate the condition number for an operation.
    fn estimate_condition_number(&self, op: FpOp, inputs: &[f64], output: f64) -> Option<f64> {
        match op {
            FpOp::Sub if inputs.len() == 2 => {
                // κ(a - b) = max(|a|, |b|) / |a - b|
                let max_input = inputs[0].abs().max(inputs[1].abs());
                let diff = output.abs();
                if diff > 0.0 {
                    Some(max_input / diff)
                } else {
                    Some(f64::INFINITY)
                }
            }
            FpOp::Add if inputs.len() == 2 => {
                let sum_abs = inputs[0].abs() + inputs[1].abs();
                let result = output.abs();
                if result > 0.0 {
                    Some(sum_abs / result)
                } else {
                    Some(1.0)
                }
            }
            FpOp::Div if inputs.len() == 2 => {
                // Division condition number is 1 for the numerator, 1 for denominator
                Some(1.0)
            }
            FpOp::Sqrt if inputs.len() == 1 => {
                // sqrt has condition number 0.5
                Some(0.5)
            }
            FpOp::Exp if inputs.len() == 1 => {
                // κ(exp(x)) = |x|
                Some(inputs[0].abs())
            }
            FpOp::Log if inputs.len() == 1 => {
                // κ(log(x)) = 1 / |log(x)|
                let lnx = inputs[0].ln().abs();
                if lnx > 0.0 {
                    Some(1.0 / lnx)
                } else {
                    Some(f64::INFINITY)
                }
            }
            _ => None,
        }
    }

    /// Compute edge weight using finite differencing.
    fn compute_edge_weight(
        &self,
        output: f64,
        shadow_output: f64,
        source_node: EagNodeId,
    ) -> Result<f64, AnalysisError> {
        let source = self
            .graph
            .node(source_node)
            .ok_or(AnalysisError::NodeNotFound(source_node))?;

        let source_error = source.local_error;
        if source_error.abs() < 1e-300 {
            return Ok(0.0);
        }

        let output_error = (output - shadow_output).abs();
        Ok(output_error / source_error)
    }

    /// Build the complete EAG from an execution trace.
    pub fn build_from_trace(&mut self, trace: &ExecutionTrace) -> Result<(), AnalysisError> {
        if trace.events.is_empty() {
            return Err(AnalysisError::EmptyTrace);
        }
        for event in &trace.events {
            self.process_event(event)?;
        }
        self.sparsify();
        Ok(())
    }

    /// Remove edges below the minimum weight threshold.
    pub fn sparsify(&mut self) {
        // Sparsification is handled during edge construction
        // This method is a hook for post-hoc cleanup
        log::debug!(
            "EAG has {} nodes and {} edges after sparsification",
            self.graph.node_count(),
            self.graph.edge_count()
        );
    }

    /// Consume the builder and return the constructed EAG.
    pub fn finish(self) -> ErrorAmplificationGraph {
        self.graph
    }
}

/// Compute the T1 error bound for an EAG.
///
/// T1 (EAG Soundness): Total output error ≤ Σ over source-to-sink paths
/// of (Π edge weights × source error).
pub fn t1_bound(eag: &ErrorAmplificationGraph) -> f64 {
    let sinks = eag.sinks();
    sinks
        .iter()
        .map(|sink| eag.t1_error_bound(sink.id))
        .fold(0.0_f64, f64::max)
}

/// Decompose error by path: returns a list of (path, attributed error).
///
/// Each path is a sequence of node ids from source to sink.
pub fn error_path_decomposition(eag: &ErrorAmplificationGraph) -> Vec<(Vec<EagNodeId>, f64)> {
    let mut paths = Vec::new();

    for source in eag.sources() {
        let mut stack: Vec<(EagNodeId, Vec<EagNodeId>, f64)> =
            vec![(source.id, vec![source.id], source.local_error)];

        while let Some((current, path, attributed)) = stack.pop() {
            let outgoing = eag.outgoing(current);
            if outgoing.is_empty() {
                // Reached a sink
                paths.push((path, attributed));
            } else {
                for edge in outgoing {
                    let mut new_path = path.clone();
                    new_path.push(edge.target);
                    let target_node = eag.node(edge.target);
                    let target_local = target_node.map_or(0.0, |n| n.local_error);
                    stack.push((
                        edge.target,
                        new_path,
                        attributed * edge.weight.0 + target_local,
                    ));
                }
            }
        }
    }

    // Sort by attributed error (descending)
    paths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    paths
}

/// Estimate the treewidth of an EAG using min-fill heuristic.
///
/// Returns an upper bound on the treewidth.
pub fn estimate_treewidth(eag: &ErrorAmplificationGraph) -> usize {
    let n = eag.node_count();
    if n <= 2 {
        return n.saturating_sub(1);
    }

    // Build undirected adjacency from the EAG
    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    for node in eag.nodes() {
        adj.entry(node.id.0).or_default();
    }
    for edge in eag.edges() {
        adj.entry(edge.source.0).or_default().push(edge.target.0);
        adj.entry(edge.target.0).or_default().push(edge.source.0);
    }

    // Min-fill elimination ordering heuristic
    let mut remaining: Vec<u32> = adj.keys().copied().collect();
    let mut max_clique_size: usize = 0;

    while !remaining.is_empty() {
        // Find vertex with minimum fill-in
        let mut best_vertex = remaining[0];
        let mut best_fill = usize::MAX;

        for &v in &remaining {
            let neighbors: Vec<u32> = adj.get(&v).map_or(Vec::new(), |ns| {
                ns.iter()
                    .filter(|n| remaining.contains(n))
                    .copied()
                    .collect()
            });
            let mut fill = 0;
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    if !adj
                        .get(&neighbors[i])
                        .map_or(false, |ns| ns.contains(&neighbors[j]))
                    {
                        fill += 1;
                    }
                }
            }
            if fill < best_fill {
                best_fill = fill;
                best_vertex = v;
            }
        }

        // Count the neighborhood size of the eliminated vertex
        let neighbors: Vec<u32> = adj.get(&best_vertex).map_or(Vec::new(), |ns| {
            ns.iter()
                .filter(|n| remaining.contains(n))
                .copied()
                .collect()
        });
        max_clique_size = max_clique_size.max(neighbors.len() + 1);

        // Add fill edges
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                adj.entry(neighbors[i]).or_default().push(neighbors[j]);
                adj.entry(neighbors[j]).or_default().push(neighbors[i]);
            }
        }

        remaining.retain(|&v| v != best_vertex);
    }

    max_clique_size.saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fpdiag_types::precision::Precision;
    use fpdiag_types::trace::TraceEvent;

    #[test]
    fn build_eag_from_simple_trace() {
        let mut builder = EagBuilder::with_defaults();
        let event = TraceEvent::Operation {
            seq: 0,
            op: FpOp::Add,
            inputs: vec![1.0, 1e-16],
            output: 1.0,
            shadow_output: 1.0 + 1e-16,
            precision: Precision::Double,
            source: None,
            expr_node: None,
        };
        let result = builder.process_event(&event);
        assert!(result.is_ok());
        let eag = builder.finish();
        assert_eq!(eag.node_count(), 1);
    }

    #[test]
    fn empty_trace_error() {
        let trace = fpdiag_types::trace::ExecutionTrace::new();
        let mut builder = EagBuilder::with_defaults();
        assert!(builder.build_from_trace(&trace).is_err());
    }
}
