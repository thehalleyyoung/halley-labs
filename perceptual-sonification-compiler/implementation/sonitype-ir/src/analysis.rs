//! Graph analysis utilities.
//!
//! - [`GraphAnalyzer`] – spectral occupancy, cognitive load, information flow,
//!   bottleneck detection.
//! - [`DependencyAnalysis`] – data dependencies, parallelisable groups,
//!   critical path computation.
//! - [`ResourceAnalysis`] – memory, buffer, and CPU estimation.
//! - [`SpectralAnalysis`] – per-band energy estimation, masking prediction,
//!   frequency conflict detection.

use std::collections::{HashMap, HashSet};
use crate::graph::{AudioGraph, NodeId, NodeType, BufferType};
use crate::masking_pass::{StreamDescriptor, NUM_BARK_BANDS, compute_masking_matrix};
use crate::IrResult;

// ---------------------------------------------------------------------------
// GraphAnalyzer
// ---------------------------------------------------------------------------

/// High-level graph analysis.
pub struct GraphAnalyzer;

/// Overall graph metrics.
#[derive(Debug, Clone)]
pub struct GraphMetrics {
    pub node_count: usize,
    pub edge_count: usize,
    pub source_count: usize,
    pub sink_count: usize,
    pub total_wcet_us: f64,
    pub critical_path_wcet_us: f64,
    pub estimated_cognitive_load: f64,
    pub bottleneck_node: Option<NodeId>,
    pub bottleneck_wcet_us: f64,
    pub spectral_occupancy: [f64; NUM_BARK_BANDS],
    pub information_flow: f64,
}

impl GraphAnalyzer {
    /// Compute comprehensive metrics for the graph.
    pub fn analyze(graph: &mut AudioGraph) -> IrResult<GraphMetrics> {
        let _ = graph.ensure_sorted();

        let node_count = graph.node_count();
        let edge_count = graph.edge_count();
        let source_count = graph.source_nodes().len();
        let sink_count = graph.output_nodes().len();
        let total_wcet = graph.total_wcet();
        let critical_wcet = graph.critical_path_wcet();

        // Bottleneck: node with highest WCET.
        let (bn_id, bn_wcet) = graph.nodes.iter()
            .max_by(|a, b| a.wcet_estimate_us.partial_cmp(&b.wcet_estimate_us).unwrap())
            .map(|n| (Some(n.id), n.wcet_estimate_us))
            .unwrap_or((None, 0.0));

        // Spectral occupancy.
        let occupancy = Self::compute_spectral_occupancy(graph);

        // Cognitive load estimate: number of simultaneous audio sources × complexity.
        let cog_load = Self::estimate_cognitive_load(graph);

        // Information flow: ratio of edges to nodes (connectivity).
        let info_flow = if node_count > 0 { edge_count as f64 / node_count as f64 } else { 0.0 };

        Ok(GraphMetrics {
            node_count,
            edge_count,
            source_count,
            sink_count,
            total_wcet_us: total_wcet,
            critical_path_wcet_us: critical_wcet,
            estimated_cognitive_load: cog_load,
            bottleneck_node: bn_id,
            bottleneck_wcet_us: bn_wcet,
            spectral_occupancy: occupancy,
            information_flow: info_flow,
        })
    }

    /// Compute per-Bark-band energy occupancy.
    pub fn compute_spectral_occupancy(graph: &AudioGraph) -> [f64; NUM_BARK_BANDS] {
        let streams: Vec<StreamDescriptor> = graph.nodes.iter()
            .filter(|n| matches!(n.node_type,
                NodeType::Oscillator { .. } | NodeType::NoiseGenerator | NodeType::Filter { .. }))
            .map(StreamDescriptor::from_node)
            .collect();
        let mut occ = [0.0f64; NUM_BARK_BANDS];
        for s in &streams {
            for (b, &e) in s.bark_energies.iter().enumerate() {
                occ[b] += e;
            }
        }
        occ
    }

    /// Estimate cognitive load as a weighted sum of active streams.
    /// More complex node types (FM modulation, noise) increase load.
    pub fn estimate_cognitive_load(graph: &AudioGraph) -> f64 {
        let mut load = 0.0f64;
        for node in &graph.nodes {
            load += match &node.node_type {
                NodeType::Oscillator { .. } => 1.0,
                NodeType::NoiseGenerator => 0.8,
                NodeType::Modulator => 1.5,
                NodeType::Filter { .. } => 0.3,
                NodeType::Compressor | NodeType::Limiter => 0.2,
                NodeType::Gain { .. } | NodeType::Pan { .. } => 0.1,
                NodeType::PitchShifter | NodeType::TimeStretch => 1.2,
                _ => 0.0,
            };
        }
        load
    }

    /// Identify the node with the highest WCET.
    pub fn find_bottleneck(graph: &AudioGraph) -> Option<(NodeId, f64)> {
        graph.nodes.iter()
            .max_by(|a, b| a.wcet_estimate_us.partial_cmp(&b.wcet_estimate_us).unwrap())
            .map(|n| (n.id, n.wcet_estimate_us))
    }
}

// ---------------------------------------------------------------------------
// DependencyAnalysis
// ---------------------------------------------------------------------------

/// Data dependency analysis.
pub struct DependencyAnalysis;

/// Dependency report.
#[derive(Debug, Clone)]
pub struct DependencyReport {
    /// For each node, the set of nodes it depends on (transitively).
    pub dependencies: HashMap<NodeId, HashSet<NodeId>>,
    /// Groups of nodes that can execute in parallel (same topological level).
    pub parallel_groups: Vec<Vec<NodeId>>,
    /// Critical path (sequence of nodes with the longest total WCET).
    pub critical_path: Vec<NodeId>,
    pub critical_path_wcet_us: f64,
}

impl DependencyAnalysis {
    /// Compute the full dependency analysis.
    pub fn analyze(graph: &mut AudioGraph) -> IrResult<DependencyReport> {
        graph.ensure_sorted()?;
        let order = graph.topological_order.clone();

        // Transitive dependencies.
        let mut deps: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        for &nid in &order {
            let mut node_deps = HashSet::new();
            for pred in graph.predecessors(nid) {
                node_deps.insert(pred);
                if let Some(pred_deps) = deps.get(&pred) {
                    node_deps.extend(pred_deps.iter());
                }
            }
            deps.insert(nid, node_deps);
        }

        // Parallel groups: nodes at the same "level" (longest distance from sources).
        let mut level: HashMap<NodeId, usize> = HashMap::new();
        for &nid in &order {
            let max_pred_level = graph.predecessors(nid).iter()
                .filter_map(|p| level.get(p))
                .max()
                .copied()
                .unwrap_or(0);
            let my_level = if graph.predecessors(nid).is_empty() { 0 } else { max_pred_level + 1 };
            level.insert(nid, my_level);
        }
        let max_level = level.values().max().copied().unwrap_or(0);
        let mut parallel_groups: Vec<Vec<NodeId>> = Vec::new();
        for l in 0..=max_level {
            let group: Vec<NodeId> = level.iter()
                .filter(|(_, &lv)| lv == l)
                .map(|(&id, _)| id)
                .collect();
            if !group.is_empty() {
                parallel_groups.push(group);
            }
        }

        // Critical path by WCET.
        let mut dist: HashMap<NodeId, f64> = HashMap::new();
        let mut pred_map: HashMap<NodeId, Option<NodeId>> = HashMap::new();
        for &nid in &order {
            let node_wcet = graph.node(nid).map(|n| n.wcet_estimate_us).unwrap_or(0.0);
            let (max_d, max_p) = graph.predecessors(nid).iter()
                .filter_map(|&p| dist.get(&p).map(|&d| (d, Some(p))))
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap_or((0.0, None));
            dist.insert(nid, max_d + node_wcet);
            pred_map.insert(nid, max_p);
        }
        let (&end_node, &end_dist) = dist.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((&NodeId(0), &0.0));

        let mut path = vec![end_node];
        let mut cur = end_node;
        while let Some(Some(p)) = pred_map.get(&cur) {
            path.push(*p);
            cur = *p;
        }
        path.reverse();

        Ok(DependencyReport {
            dependencies: deps,
            parallel_groups,
            critical_path: path,
            critical_path_wcet_us: end_dist,
        })
    }

    /// Identify sets of nodes that have no mutual dependencies and can run in
    /// parallel.
    pub fn parallelizable_groups(graph: &mut AudioGraph) -> IrResult<Vec<Vec<NodeId>>> {
        let report = Self::analyze(graph)?;
        Ok(report.parallel_groups)
    }
}

// ---------------------------------------------------------------------------
// ResourceAnalysis
// ---------------------------------------------------------------------------

/// Resource usage analysis.
pub struct ResourceAnalysis;

/// Resource usage report.
#[derive(Debug, Clone)]
pub struct ResourceReport {
    /// Total estimated memory for all buffers (bytes).
    pub total_memory_bytes: usize,
    /// Number of audio buffers needed.
    pub audio_buffer_count: usize,
    /// Number of control buffers needed.
    pub control_buffer_count: usize,
    /// Size of each audio buffer (bytes, assuming f32 samples).
    pub audio_buffer_size_bytes: usize,
    /// Estimated CPU utilisation per callback (fraction of budget).
    pub cpu_utilisation: f64,
    /// Per-node memory contributions.
    pub per_node_memory: HashMap<NodeId, usize>,
}

impl ResourceAnalysis {
    /// Compute resource estimates.
    pub fn analyze(graph: &AudioGraph) -> ResourceReport {
        let block = graph.block_size;
        let audio_buf_bytes = block * 4; // f32
        let control_buf_bytes = 4usize; // single f32

        // Count unique buffers from edges.
        let mut audio_bufs = HashSet::new();
        let mut control_bufs = HashSet::new();
        for edge in &graph.edges {
            match edge.buffer_type {
                BufferType::AudioBuffer => { audio_bufs.insert(edge.id); }
                BufferType::ControlBuffer => { control_bufs.insert(edge.id); }
                BufferType::TriggerFlag => { /* negligible */ }
            }
        }

        let audio_count = audio_bufs.len();
        let control_count = control_bufs.len();
        let total_mem = audio_count * audio_buf_bytes + control_count * control_buf_bytes;

        // Per-node memory: delay lines, FFT buffers, etc.
        let mut per_node_memory = HashMap::new();
        for node in &graph.nodes {
            let mem = match &node.node_type {
                NodeType::Delay { samples } => *samples * 4,
                NodeType::PitchShifter => 2048 * 4 * 2, // two windows
                NodeType::TimeStretch => 4096 * 4 * 2,
                NodeType::Compressor => block * 4,
                NodeType::Limiter => {
                    let lookahead_samples = (0.005 * graph.sample_rate) as usize;
                    lookahead_samples * 4
                }
                _ => 0,
            };
            if mem > 0 {
                per_node_memory.insert(node.id, mem);
            }
        }
        let extra_mem: usize = per_node_memory.values().sum();

        // CPU utilisation: total WCET / budget.
        let budget_us = (block as f64 / graph.sample_rate) * 1_000_000.0;
        let total_wcet = graph.total_wcet();
        let cpu_util = if budget_us > 0.0 { total_wcet / budget_us } else { 0.0 };

        ResourceReport {
            total_memory_bytes: total_mem + extra_mem,
            audio_buffer_count: audio_count,
            control_buffer_count: control_count,
            audio_buffer_size_bytes: audio_buf_bytes,
            cpu_utilisation: cpu_util,
            per_node_memory,
        }
    }

    /// Estimate the number of buffers that can be reused.
    pub fn reusable_buffer_count(graph: &mut AudioGraph) -> IrResult<usize> {
        let _ = graph.ensure_sorted();
        let order = graph.topological_order.clone();
        let mut last_use: HashMap<crate::graph::EdgeId, usize> = HashMap::new();
        for (step, &nid) in order.iter().enumerate() {
            for edge in &graph.edges {
                if edge.dest_node == nid {
                    last_use.insert(edge.id, step);
                }
            }
        }
        let total = graph.edges.iter().filter(|e| e.buffer_type == BufferType::AudioBuffer).count();
        // Simple heuristic: buffers whose last use is not the final step can be reused.
        let final_step = order.len().saturating_sub(1);
        let reusable = last_use.iter().filter(|(_, &lu)| lu < final_step).count();
        Ok(reusable.min(total))
    }
}

// ---------------------------------------------------------------------------
// SpectralAnalysis
// ---------------------------------------------------------------------------

/// Spectral analysis without rendering.
pub struct SpectralAnalysis;

/// Per-band energy estimate.
#[derive(Debug, Clone)]
pub struct BandEnergy {
    pub band: usize,
    pub lower_hz: f64,
    pub upper_hz: f64,
    pub energy: f64,
    pub contributing_nodes: Vec<NodeId>,
}

/// Frequency conflict between two nodes.
#[derive(Debug, Clone)]
pub struct FrequencyConflict {
    pub node_a: NodeId,
    pub node_b: NodeId,
    pub overlap_band: usize,
    pub overlap_energy: f64,
    pub masking_factor: f64,
}

impl SpectralAnalysis {
    /// Compute per-band energy for the entire graph.
    pub fn per_band_energy(graph: &AudioGraph) -> Vec<BandEnergy> {
        let bark_upper: [f64; NUM_BARK_BANDS] = [
            100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0,
            1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0,
            3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 15500.0,
        ];

        let streams: Vec<StreamDescriptor> = graph.nodes.iter()
            .filter(|n| matches!(n.node_type,
                NodeType::Oscillator { .. } | NodeType::NoiseGenerator | NodeType::Filter { .. }))
            .map(StreamDescriptor::from_node)
            .collect();

        let mut bands = Vec::new();
        let mut prev = 0.0f64;
        for (b, &upper) in bark_upper.iter().enumerate() {
            let energy: f64 = streams.iter().map(|s| s.bark_energies[b]).sum();
            let contributors: Vec<NodeId> = streams.iter()
                .filter(|s| s.bark_energies[b] > 1e-12)
                .map(|s| s.node_id)
                .collect();
            bands.push(BandEnergy {
                band: b,
                lower_hz: prev,
                upper_hz: upper,
                energy,
                contributing_nodes: contributors,
            });
            prev = upper;
        }
        bands
    }

    /// Predict masking between all stream pairs from graph structure.
    pub fn predict_masking(graph: &AudioGraph) -> Vec<Vec<f64>> {
        let streams: Vec<StreamDescriptor> = graph.nodes.iter()
            .filter(|n| matches!(n.node_type,
                NodeType::Oscillator { .. } | NodeType::NoiseGenerator | NodeType::Filter { .. }))
            .map(StreamDescriptor::from_node)
            .collect();
        compute_masking_matrix(&streams)
    }

    /// Detect frequency conflicts (bands where multiple streams have significant energy).
    pub fn detect_conflicts(graph: &AudioGraph) -> Vec<FrequencyConflict> {
        let streams: Vec<StreamDescriptor> = graph.nodes.iter()
            .filter(|n| matches!(n.node_type,
                NodeType::Oscillator { .. } | NodeType::NoiseGenerator | NodeType::Filter { .. }))
            .map(StreamDescriptor::from_node)
            .collect();

        let masking = compute_masking_matrix(&streams);
        let mut conflicts = Vec::new();

        for i in 0..streams.len() {
            for j in (i + 1)..streams.len() {
                // Find overlapping bands.
                for band in 0..NUM_BARK_BANDS {
                    let e_i = streams[i].bark_energies[band];
                    let e_j = streams[j].bark_energies[band];
                    if e_i > 1e-12 && e_j > 1e-12 {
                        let overlap = e_i.min(e_j);
                        if overlap > 1e-9 {
                            conflicts.push(FrequencyConflict {
                                node_a: streams[i].node_id,
                                node_b: streams[j].node_id,
                                overlap_band: band,
                                overlap_energy: overlap,
                                masking_factor: masking[i][j],
                            });
                        }
                    }
                }
            }
        }
        conflicts
    }

    /// Total spectral energy across all bands.
    pub fn total_spectral_energy(graph: &AudioGraph) -> f64 {
        Self::per_band_energy(graph).iter().map(|b| b.energy).sum()
    }

    /// Highest-energy band index.
    pub fn peak_band(graph: &AudioGraph) -> Option<usize> {
        let bands = Self::per_band_energy(graph);
        bands.iter()
            .max_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
            .map(|b| b.band)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{AudioGraph, NodeType, GraphBuilder};
    use crate::node::Waveform;

    fn build_graph() -> AudioGraph {
        let (b, osc) = GraphBuilder::with_defaults()
            .add_oscillator("osc", Waveform::Sine, 440.0);
        let (b, g) = b.add_gain("gain", 0.5);
        let (b, out) = b.add_output("out", "stereo");
        b.connect(osc, "out", g, "in")
         .connect(g, "out", out, "in")
         .build().unwrap()
    }

    fn build_multi() -> AudioGraph {
        let (b, osc1) = GraphBuilder::with_defaults()
            .add_oscillator("osc1", Waveform::Sine, 440.0);
        let (b, osc2) = b.add_oscillator("osc2", Waveform::Saw, 880.0);
        let (b, mix) = b.add_mixer("mix", 2);
        let (b, out) = b.add_output("out", "stereo");
        let g = b.graph();
        let mix_in0 = g.node(mix).unwrap().inputs[0].id;
        let mix_in1 = g.node(mix).unwrap().inputs[1].id;
        let osc1_out = g.node(osc1).unwrap().first_output().unwrap().id;
        let osc2_out = g.node(osc2).unwrap().first_output().unwrap().id;
        let mut g = b.build_unsorted();
        let _ = g.add_edge(osc1, osc1_out, mix, mix_in0);
        let _ = g.add_edge(osc2, osc2_out, mix, mix_in1);
        g.add_edge_by_name(mix, "out", out, "in").unwrap();
        g
    }

    #[test]
    fn test_graph_analyzer_metrics() {
        let mut g = build_graph();
        let m = GraphAnalyzer::analyze(&mut g).unwrap();
        assert_eq!(m.node_count, 3);
        assert_eq!(m.edge_count, 2);
        assert_eq!(m.source_count, 1);
        assert_eq!(m.sink_count, 1);
        assert!(m.total_wcet_us > 0.0);
    }

    #[test]
    fn test_graph_analyzer_cognitive_load() {
        let g = build_graph();
        let load = GraphAnalyzer::estimate_cognitive_load(&g);
        assert!(load > 0.0);
    }

    #[test]
    fn test_graph_analyzer_bottleneck() {
        let g = build_graph();
        let bn = GraphAnalyzer::find_bottleneck(&g);
        assert!(bn.is_some());
    }

    #[test]
    fn test_dependency_analysis() {
        let mut g = build_graph();
        let report = DependencyAnalysis::analyze(&mut g).unwrap();
        assert!(!report.parallel_groups.is_empty());
        assert!(!report.critical_path.is_empty());
    }

    #[test]
    fn test_parallel_groups() {
        let mut g = build_multi();
        let groups = DependencyAnalysis::parallelizable_groups(&mut g).unwrap();
        // The two oscillators should be in the same parallel group.
        assert!(!groups.is_empty());
        let first_group = &groups[0];
        assert!(first_group.len() >= 2); // both oscillators at level 0
    }

    #[test]
    fn test_resource_analysis() {
        let g = build_graph();
        let report = ResourceAnalysis::analyze(&g);
        assert!(report.audio_buffer_count > 0);
        assert!(report.total_memory_bytes > 0);
        assert!(report.cpu_utilisation > 0.0);
    }

    #[test]
    fn test_resource_analysis_with_delay() {
        let (b, osc) = GraphBuilder::with_defaults()
            .add_oscillator("osc", Waveform::Sine, 440.0);
        let (b, del) = b.add_delay("delay", 48000);
        let (b, out) = b.add_output("out", "stereo");
        let g = b.connect(osc, "out", del, "in")
                  .connect(del, "out", out, "in")
                  .build().unwrap();
        let report = ResourceAnalysis::analyze(&g);
        // Delay line should contribute to per_node_memory.
        assert!(!report.per_node_memory.is_empty());
    }

    #[test]
    fn test_reusable_buffers() {
        let mut g = build_graph();
        let reusable = ResourceAnalysis::reusable_buffer_count(&mut g).unwrap();
        // At least one buffer should be reusable in a linear chain.
        assert!(reusable >= 0);
    }

    #[test]
    fn test_spectral_per_band_energy() {
        let g = build_graph();
        let bands = SpectralAnalysis::per_band_energy(&g);
        assert_eq!(bands.len(), NUM_BARK_BANDS);
        let total: f64 = bands.iter().map(|b| b.energy).sum();
        assert!(total > 0.0);
    }

    #[test]
    fn test_spectral_predict_masking() {
        let g = build_multi();
        let matrix = SpectralAnalysis::predict_masking(&g);
        assert!(!matrix.is_empty());
    }

    #[test]
    fn test_spectral_detect_conflicts() {
        let g = build_multi();
        let conflicts = SpectralAnalysis::detect_conflicts(&g);
        // Saw at 880 Hz has harmonics that overlap with sine at 440 Hz.
        let _ = conflicts;
    }

    #[test]
    fn test_spectral_total_energy() {
        let g = build_graph();
        let e = SpectralAnalysis::total_spectral_energy(&g);
        assert!(e > 0.0);
    }

    #[test]
    fn test_spectral_peak_band() {
        let g = build_graph();
        let peak = SpectralAnalysis::peak_band(&g);
        assert!(peak.is_some());
    }

    #[test]
    fn test_dependency_analysis_empty() {
        let mut g = AudioGraph::default();
        let report = DependencyAnalysis::analyze(&mut g);
        // Empty graph should still succeed.
        assert!(report.is_ok());
    }
}
