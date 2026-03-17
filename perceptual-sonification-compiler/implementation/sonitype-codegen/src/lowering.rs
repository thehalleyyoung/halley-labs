//! IR lowering — transforms the high-level `CgGraph` into an executable
//! execution plan by resolving abstract parameters, expanding compound nodes,
//! inserting parameter smoothing, and handling sample-rate conversion.

use crate::{
    BufferKind, CgGraph, CgGraphBuilder, CodegenConfig, CodegenError, CodegenResult, EdgeInfo,
    NodeInfo, NodeKind,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Parameter smoothing
// ---------------------------------------------------------------------------

/// Method for smoothing parameter changes to prevent zipper noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmoothingMethod {
    /// One-pole lowpass: y[n] = α·x[n] + (1-α)·y[n-1]
    OnePole,
    /// Linear crossfade over a specified number of samples.
    Crossfade,
    /// No smoothing.
    None,
}

/// Parameter smoothing configuration for a single parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSmoothing {
    /// Name of the parameter being smoothed.
    pub parameter_name: String,
    /// Smoothing method.
    pub method: SmoothingMethod,
    /// Smoothing time in seconds (controls crossfade length or one-pole coeff).
    pub smoothing_time_s: f64,
    /// Computed smoothing coefficient (α for one-pole, or sample count for crossfade).
    pub coefficient: f64,
}

impl ParameterSmoothing {
    /// Create a one-pole smoother.
    pub fn one_pole(name: &str, smoothing_time_s: f64, sample_rate: f64) -> Self {
        let coeff = if smoothing_time_s > 0.0 {
            1.0 - (-1.0 / (smoothing_time_s * sample_rate)).exp()
        } else {
            1.0
        };
        Self {
            parameter_name: name.to_string(),
            method: SmoothingMethod::OnePole,
            smoothing_time_s,
            coefficient: coeff,
        }
    }

    /// Create a crossfade smoother.
    pub fn crossfade(name: &str, smoothing_time_s: f64, sample_rate: f64) -> Self {
        let samples = (smoothing_time_s * sample_rate).ceil().max(1.0);
        Self {
            parameter_name: name.to_string(),
            method: SmoothingMethod::Crossfade,
            smoothing_time_s,
            coefficient: samples,
        }
    }

    /// Create a no-op smoother.
    pub fn none(name: &str) -> Self {
        Self {
            parameter_name: name.to_string(),
            method: SmoothingMethod::None,
            smoothing_time_s: 0.0,
            coefficient: 1.0,
        }
    }

    /// Apply one step of smoothing: given a current value and target, return
    /// the smoothed value.
    pub fn apply(&self, current: f64, target: f64) -> f64 {
        match self.method {
            SmoothingMethod::OnePole => {
                current + self.coefficient * (target - current)
            }
            SmoothingMethod::Crossfade => {
                // For crossfade, coefficient is the number of samples.
                // Each step moves 1/coefficient of the way.
                let step = 1.0 / self.coefficient;
                current + step * (target - current)
            }
            SmoothingMethod::None => target,
        }
    }
}

// ---------------------------------------------------------------------------
// Sample rate conversion
// ---------------------------------------------------------------------------

/// Sample rate conversion parameters for connecting nodes at different rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleRateConversion {
    pub source_rate: f64,
    pub target_rate: f64,
    pub method: SrcMethod,
    /// Ratio = target / source.
    pub ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SrcMethod {
    /// Linear interpolation (cheap, adequate for control signals).
    LinearInterpolation,
    /// Zero-order hold (cheapest, for triggers).
    ZeroOrderHold,
    /// Cubic interpolation (higher quality for audio).
    CubicInterpolation,
}

impl SampleRateConversion {
    pub fn new(source_rate: f64, target_rate: f64) -> Self {
        let ratio = target_rate / source_rate;
        let method = if (ratio - 1.0).abs() < 1e-9 {
            SrcMethod::ZeroOrderHold
        } else if ratio < 0.5 || ratio > 2.0 {
            SrcMethod::CubicInterpolation
        } else {
            SrcMethod::LinearInterpolation
        };
        Self {
            source_rate,
            target_rate,
            method,
            ratio,
        }
    }

    /// Whether conversion is needed.
    pub fn is_identity(&self) -> bool {
        (self.ratio - 1.0).abs() < 1e-9
    }

    /// Perform linear interpolation on a block of samples, returning
    /// `output_len` samples.
    pub fn convert_block(&self, input: &[f64], output_len: usize) -> Vec<f64> {
        if input.is_empty() || output_len == 0 {
            return vec![0.0; output_len];
        }
        if self.is_identity() && input.len() == output_len {
            return input.to_vec();
        }

        let mut output = Vec::with_capacity(output_len);
        let step = input.len() as f64 / output_len as f64;

        for i in 0..output_len {
            let pos = i as f64 * step;
            let idx = pos as usize;
            let frac = pos - idx as f64;

            match self.method {
                SrcMethod::ZeroOrderHold => {
                    output.push(input[idx.min(input.len() - 1)]);
                }
                SrcMethod::LinearInterpolation => {
                    let a = input[idx.min(input.len() - 1)];
                    let b = input[(idx + 1).min(input.len() - 1)];
                    output.push(a + frac * (b - a));
                }
                SrcMethod::CubicInterpolation => {
                    let n = input.len();
                    let i0 = if idx > 0 { idx - 1 } else { 0 };
                    let i1 = idx.min(n - 1);
                    let i2 = (idx + 1).min(n - 1);
                    let i3 = (idx + 2).min(n - 1);
                    let (y0, y1, y2, y3) = (input[i0], input[i1], input[i2], input[i3]);
                    let t = frac;
                    let t2 = t * t;
                    let t3 = t2 * t;
                    // Catmull-Rom
                    let a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
                    let a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                    let a2 = -0.5 * y0 + 0.5 * y2;
                    let a3 = y1;
                    output.push(a0 * t3 + a1 * t2 + a2 * t + a3);
                }
            }
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Lowered node representation
// ---------------------------------------------------------------------------

/// A lowered node with all parameters resolved to concrete values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoweredNode {
    pub id: u64,
    pub name: String,
    pub kind: NodeKind,
    pub resolved_params: HashMap<String, f64>,
    /// Smoothers attached to this node's parameters.
    pub smoothers: Vec<ParameterSmoothing>,
    /// Whether this node was expanded from a compound node.
    pub expanded: bool,
    /// Original compound node ID if expanded.
    pub parent_compound_id: Option<u64>,
}

/// The lowered execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoweredGraph {
    pub nodes: Vec<LoweredNode>,
    pub edges: Vec<EdgeInfo>,
    pub topological_order: Vec<u64>,
    pub sample_rate: f64,
    pub block_size: usize,
    /// Sample rate conversions inserted.
    pub src_conversions: Vec<(u64, u64, SampleRateConversion)>,
    /// Total smoothing nodes inserted.
    pub smoothing_node_count: usize,
}

// ---------------------------------------------------------------------------
// IrLowerer
// ---------------------------------------------------------------------------

/// Transforms a `CgGraph` into a `LoweredGraph`.
#[derive(Debug, Clone)]
pub struct IrLowerer {
    pub config: CodegenConfig,
    /// Default smoothing time for parameters that can cause zipper noise.
    pub default_smoothing_time: f64,
    /// Parameters that should receive smoothing.
    smoothable_params: Vec<String>,
}

impl IrLowerer {
    pub fn new(config: &CodegenConfig) -> Self {
        Self {
            config: config.clone(),
            default_smoothing_time: 0.005, // 5ms
            smoothable_params: vec![
                "frequency".into(),
                "cutoff".into(),
                "level".into(),
                "position".into(),
                "q".into(),
            ],
        }
    }

    /// Full lowering pipeline.
    pub fn lower(&self, graph: &CgGraph) -> CodegenResult<LoweredGraph> {
        let mut lowered_nodes: Vec<LoweredNode> = Vec::new();
        let mut lowered_edges = graph.edges.clone();
        let mut next_id = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
        let mut src_conversions = Vec::new();
        let mut smoothing_count = 0;

        for node in &graph.nodes {
            // Expand compound nodes.
            let expanded = self.expand_compound(node, &mut next_id);
            if expanded.len() > 1 {
                // Compound node was expanded: add internal edges.
                for i in 0..expanded.len() - 1 {
                    lowered_edges.push(EdgeInfo {
                        id: next_id,
                        source_node: expanded[i].id,
                        source_port: 0,
                        dest_node: expanded[i + 1].id,
                        dest_port: 0,
                        buffer_type: BufferKind::Audio,
                    });
                    next_id += 1;
                }
                // Redirect incoming edges to first, outgoing from last.
                let first_id = expanded.first().unwrap().id;
                let last_id = expanded.last().unwrap().id;
                for e in &mut lowered_edges {
                    if e.dest_node == node.id {
                        e.dest_node = first_id;
                    }
                    if e.source_node == node.id {
                        e.source_node = last_id;
                    }
                }
                lowered_nodes.extend(expanded);
            } else {
                // Single node: resolve parameters and add smoothers.
                let mut resolved = self.resolve_parameters(node);
                let smoothers = self.insert_smoothers(node);
                smoothing_count += smoothers.len();
                resolved.smoothers = smoothers;
                lowered_nodes.push(resolved);
            }
        }

        // Insert sample rate conversions where needed.
        let node_rates: HashMap<u64, f64> = lowered_nodes
            .iter()
            .map(|n| {
                let rate = n
                    .resolved_params
                    .get("sample_rate")
                    .copied()
                    .unwrap_or(self.config.sample_rate);
                (n.id, rate)
            })
            .collect();

        let edges_snapshot = lowered_edges.clone();
        for edge in &edges_snapshot {
            let src_rate = node_rates.get(&edge.source_node).copied().unwrap_or(self.config.sample_rate);
            let dst_rate = node_rates.get(&edge.dest_node).copied().unwrap_or(self.config.sample_rate);
            if (src_rate - dst_rate).abs() > 1.0 {
                let conv = SampleRateConversion::new(src_rate, dst_rate);
                src_conversions.push((edge.source_node, edge.dest_node, conv));
            }
        }

        // Compute topological order on lowered graph.
        let topo = self.topological_sort(&lowered_nodes, &lowered_edges)?;

        Ok(LoweredGraph {
            nodes: lowered_nodes,
            edges: lowered_edges,
            topological_order: topo,
            sample_rate: self.config.sample_rate,
            block_size: self.config.block_size,
            src_conversions,
            smoothing_node_count: smoothing_count,
        })
    }

    /// Resolve abstract parameters to concrete numeric values.
    fn resolve_parameters(&self, node: &NodeInfo) -> LoweredNode {
        let mut resolved = node.parameters.clone();

        match node.kind {
            NodeKind::Oscillator => {
                resolved.entry("frequency".into()).or_insert(440.0);
                resolved.entry("amplitude".into()).or_insert(1.0);
                resolved.entry("phase".into()).or_insert(0.0);
            }
            NodeKind::Filter => {
                resolved.entry("cutoff".into()).or_insert(1000.0);
                resolved.entry("q".into()).or_insert(0.707);
            }
            NodeKind::Envelope => {
                resolved.entry("attack".into()).or_insert(0.01);
                resolved.entry("decay".into()).or_insert(0.1);
                resolved.entry("sustain".into()).or_insert(0.7);
                resolved.entry("release".into()).or_insert(0.3);
            }
            NodeKind::Gain => {
                resolved.entry("level".into()).or_insert(1.0);
            }
            NodeKind::Pan => {
                resolved.entry("position".into()).or_insert(0.0);
            }
            NodeKind::Delay => {
                resolved.entry("samples".into()).or_insert(4800.0);
                resolved.entry("feedback".into()).or_insert(0.0);
                resolved.entry("mix".into()).or_insert(0.5);
            }
            NodeKind::Mixer => {
                resolved.entry("channel_count".into()).or_insert(2.0);
            }
            NodeKind::Compressor => {
                resolved.entry("threshold".into()).or_insert(-20.0);
                resolved.entry("ratio".into()).or_insert(4.0);
                resolved.entry("attack".into()).or_insert(0.01);
                resolved.entry("release".into()).or_insert(0.1);
            }
            NodeKind::Limiter => {
                resolved.entry("threshold".into()).or_insert(-1.0);
                resolved.entry("release".into()).or_insert(0.05);
            }
            _ => {}
        }

        resolved.insert("sample_rate".into(), node.sample_rate);
        resolved.insert("block_size".into(), node.block_size as f64);

        LoweredNode {
            id: node.id,
            name: node.name.clone(),
            kind: node.kind,
            resolved_params: resolved,
            smoothers: Vec::new(),
            expanded: false,
            parent_compound_id: None,
        }
    }

    /// Expand compound nodes into primitives.
    /// E.g., a PitchShifter might expand into Delay + Modulator + Mixer.
    fn expand_compound(&self, node: &NodeInfo, next_id: &mut u64) -> Vec<LoweredNode> {
        match node.kind {
            NodeKind::PitchShifter => {
                let delay_id = *next_id;
                *next_id += 1;
                let mod_id = *next_id;
                *next_id += 1;
                let gain_id = *next_id;
                *next_id += 1;

                vec![
                    LoweredNode {
                        id: delay_id,
                        name: format!("{}_delay", node.name),
                        kind: NodeKind::Delay,
                        resolved_params: {
                            let mut p = HashMap::new();
                            p.insert("samples".into(), 512.0);
                            p.insert("feedback".into(), 0.0);
                            p.insert("mix".into(), 1.0);
                            p.insert("sample_rate".into(), node.sample_rate);
                            p.insert("block_size".into(), node.block_size as f64);
                            p
                        },
                        smoothers: Vec::new(),
                        expanded: true,
                        parent_compound_id: Some(node.id),
                    },
                    LoweredNode {
                        id: mod_id,
                        name: format!("{}_mod", node.name),
                        kind: NodeKind::Modulator,
                        resolved_params: {
                            let mut p = HashMap::new();
                            p.insert("rate".into(), 5.0);
                            p.insert("depth".into(), 0.5);
                            p.insert("sample_rate".into(), node.sample_rate);
                            p.insert("block_size".into(), node.block_size as f64);
                            p
                        },
                        smoothers: Vec::new(),
                        expanded: true,
                        parent_compound_id: Some(node.id),
                    },
                    LoweredNode {
                        id: gain_id,
                        name: format!("{}_gain", node.name),
                        kind: NodeKind::Gain,
                        resolved_params: {
                            let mut p = HashMap::new();
                            p.insert("level".into(), 1.0);
                            p.insert("sample_rate".into(), node.sample_rate);
                            p.insert("block_size".into(), node.block_size as f64);
                            p
                        },
                        smoothers: Vec::new(),
                        expanded: true,
                        parent_compound_id: Some(node.id),
                    },
                ]
            }
            NodeKind::TimeStretch => {
                let delay_id = *next_id;
                *next_id += 1;
                let env_id = *next_id;
                *next_id += 1;

                vec![
                    LoweredNode {
                        id: delay_id,
                        name: format!("{}_delay", node.name),
                        kind: NodeKind::Delay,
                        resolved_params: {
                            let mut p = HashMap::new();
                            p.insert("samples".into(), 1024.0);
                            p.insert("feedback".into(), 0.0);
                            p.insert("mix".into(), 1.0);
                            p.insert("sample_rate".into(), node.sample_rate);
                            p.insert("block_size".into(), node.block_size as f64);
                            p
                        },
                        smoothers: Vec::new(),
                        expanded: true,
                        parent_compound_id: Some(node.id),
                    },
                    LoweredNode {
                        id: env_id,
                        name: format!("{}_env", node.name),
                        kind: NodeKind::Envelope,
                        resolved_params: {
                            let mut p = HashMap::new();
                            p.insert("attack".into(), 0.005);
                            p.insert("decay".into(), 0.0);
                            p.insert("sustain".into(), 1.0);
                            p.insert("release".into(), 0.005);
                            p.insert("sample_rate".into(), node.sample_rate);
                            p.insert("block_size".into(), node.block_size as f64);
                            p
                        },
                        smoothers: Vec::new(),
                        expanded: true,
                        parent_compound_id: Some(node.id),
                    },
                ]
            }
            _ => {
                vec![self.resolve_parameters(node)]
            }
        }
    }

    /// Determine which parameters need smoothing and create smoothers.
    fn insert_smoothers(&self, node: &NodeInfo) -> Vec<ParameterSmoothing> {
        let mut smoothers = Vec::new();
        for param_name in &self.smoothable_params {
            if node.parameters.contains_key(param_name) {
                smoothers.push(ParameterSmoothing::one_pole(
                    param_name,
                    self.default_smoothing_time,
                    node.sample_rate,
                ));
            }
        }
        smoothers
    }

    /// Topological sort on the lowered graph (Kahn's algorithm).
    fn topological_sort(
        &self,
        nodes: &[LoweredNode],
        edges: &[EdgeInfo],
    ) -> CodegenResult<Vec<u64>> {
        let node_ids: std::collections::HashSet<u64> = nodes.iter().map(|n| n.id).collect();
        let mut in_degree: HashMap<u64, usize> = node_ids.iter().map(|&id| (id, 0)).collect();

        for e in edges {
            if node_ids.contains(&e.dest_node) && node_ids.contains(&e.source_node) {
                *in_degree.entry(e.dest_node).or_insert(0) += 1;
            }
        }

        let mut queue: std::collections::VecDeque<u64> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut order = Vec::new();
        while let Some(n) = queue.pop_front() {
            order.push(n);
            for e in edges {
                if e.source_node == n && node_ids.contains(&e.dest_node) {
                    let deg = in_degree.get_mut(&e.dest_node).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(e.dest_node);
                    }
                }
            }
        }

        if order.len() != node_ids.len() {
            return Err(CodegenError::InvalidGraph {
                reason: "Cycle detected in lowered graph".into(),
            });
        }

        Ok(order)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferKind, CgGraphBuilder, CodegenConfig, NodeKind};

    fn test_config() -> CodegenConfig {
        CodegenConfig::default()
    }

    fn simple_chain() -> CgGraph {
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let osc = b.add_node("osc", NodeKind::Oscillator);
        let filt = b.add_node("filt", NodeKind::Filter);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, filt, BufferKind::Audio);
        b.connect(filt, out, BufferKind::Audio);
        b.build()
    }

    #[test]
    fn test_lowering_basic() {
        let cfg = test_config();
        let lowerer = IrLowerer::new(&cfg);
        let graph = simple_chain();
        let lowered = lowerer.lower(&graph).unwrap();
        assert_eq!(lowered.nodes.len(), 3);
        assert!(!lowered.topological_order.is_empty());
    }

    #[test]
    fn test_parameter_resolution_defaults() {
        let cfg = test_config();
        let lowerer = IrLowerer::new(&cfg);
        let graph = simple_chain();
        let lowered = lowerer.lower(&graph).unwrap();
        let osc = lowered.nodes.iter().find(|n| n.name == "osc").unwrap();
        assert!(osc.resolved_params.contains_key("frequency"));
        assert!(osc.resolved_params.contains_key("amplitude"));
    }

    #[test]
    fn test_parameter_resolution_preserves_existing() {
        let cfg = test_config();
        let lowerer = IrLowerer::new(&cfg);
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let mut params = HashMap::new();
        params.insert("frequency".into(), 880.0);
        let osc = b.add_node_with_params("osc", NodeKind::Oscillator, params);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, out, BufferKind::Audio);
        let graph = b.build();
        let lowered = lowerer.lower(&graph).unwrap();
        let osc_node = lowered.nodes.iter().find(|n| n.name == "osc").unwrap();
        assert_eq!(osc_node.resolved_params["frequency"], 880.0);
    }

    #[test]
    fn test_smoothing_insertion() {
        let cfg = test_config();
        let lowerer = IrLowerer::new(&cfg);
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let mut params = HashMap::new();
        params.insert("frequency".into(), 440.0);
        let osc = b.add_node_with_params("osc", NodeKind::Oscillator, params);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, out, BufferKind::Audio);
        let graph = b.build();
        let lowered = lowerer.lower(&graph).unwrap();
        let osc_node = lowered.nodes.iter().find(|n| n.name == "osc").unwrap();
        assert!(!osc_node.smoothers.is_empty());
        assert_eq!(osc_node.smoothers[0].parameter_name, "frequency");
    }

    #[test]
    fn test_compound_expansion_pitch_shifter() {
        let cfg = test_config();
        let lowerer = IrLowerer::new(&cfg);
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let ps = b.add_node("ps", NodeKind::PitchShifter);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(ps, out, BufferKind::Audio);
        let graph = b.build();
        let lowered = lowerer.lower(&graph).unwrap();
        // PitchShifter expands into 3 nodes
        let expanded: Vec<_> = lowered.nodes.iter().filter(|n| n.expanded).collect();
        assert_eq!(expanded.len(), 3);
        assert!(expanded.iter().all(|n| n.parent_compound_id == Some(ps)));
    }

    #[test]
    fn test_compound_expansion_time_stretch() {
        let cfg = test_config();
        let lowerer = IrLowerer::new(&cfg);
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let ts = b.add_node("ts", NodeKind::TimeStretch);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(ts, out, BufferKind::Audio);
        let graph = b.build();
        let lowered = lowerer.lower(&graph).unwrap();
        let expanded: Vec<_> = lowered.nodes.iter().filter(|n| n.expanded).collect();
        assert_eq!(expanded.len(), 2);
    }

    #[test]
    fn test_topological_order_preserved() {
        let cfg = test_config();
        let lowerer = IrLowerer::new(&cfg);
        let graph = simple_chain();
        let lowered = lowerer.lower(&graph).unwrap();
        // All node IDs should appear in topological order
        let topo_set: std::collections::HashSet<u64> =
            lowered.topological_order.iter().copied().collect();
        for node in &lowered.nodes {
            assert!(topo_set.contains(&node.id));
        }
    }

    #[test]
    fn test_one_pole_smoothing() {
        let s = ParameterSmoothing::one_pole("freq", 0.005, 48000.0);
        assert_eq!(s.method, SmoothingMethod::OnePole);
        // Apply: should move toward target
        let v = s.apply(0.0, 1.0);
        assert!(v > 0.0 && v < 1.0);
    }

    #[test]
    fn test_crossfade_smoothing() {
        let s = ParameterSmoothing::crossfade("level", 0.01, 48000.0);
        assert_eq!(s.method, SmoothingMethod::Crossfade);
        let v = s.apply(0.0, 1.0);
        assert!(v > 0.0 && v < 1.0);
    }

    #[test]
    fn test_sample_rate_conversion_identity() {
        let src = SampleRateConversion::new(48000.0, 48000.0);
        assert!(src.is_identity());
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = src.convert_block(&input, 4);
        assert_eq!(output, input);
    }

    #[test]
    fn test_sample_rate_conversion_upsample() {
        let src = SampleRateConversion::new(24000.0, 48000.0);
        assert!(!src.is_identity());
        let input = vec![0.0, 1.0, 0.0, -1.0];
        let output = src.convert_block(&input, 8);
        assert_eq!(output.len(), 8);
        // Interpolated values should be between input samples
        assert!(output[1] > 0.0 && output[1] < 1.0);
    }

    #[test]
    fn test_sample_rate_conversion_downsample() {
        let src = SampleRateConversion::new(48000.0, 24000.0);
        let input: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let output = src.convert_block(&input, 4);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_lowering_filter_params() {
        let cfg = test_config();
        let lowerer = IrLowerer::new(&cfg);
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let filt = b.add_node("filt", NodeKind::Filter);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(filt, out, BufferKind::Audio);
        let graph = b.build();
        let lowered = lowerer.lower(&graph).unwrap();
        let filt_node = lowered.nodes.iter().find(|n| n.name == "filt").unwrap();
        assert!(filt_node.resolved_params.contains_key("cutoff"));
        assert!(filt_node.resolved_params.contains_key("q"));
    }

    #[test]
    fn test_smoothing_none() {
        let s = ParameterSmoothing::none("test");
        assert_eq!(s.apply(0.0, 1.0), 1.0);
    }
}
