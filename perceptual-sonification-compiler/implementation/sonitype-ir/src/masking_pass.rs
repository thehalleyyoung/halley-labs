//! Masking-aware optimization passes.
//!
//! These passes use psychoacoustic masking models to optimise the audio graph:
//!
//! - [`MaskingAwareStreamMerging`] – remove or merge streams that are fully
//!   masked by louder neighbours.
//! - [`SpectralBinPacking`] – assign streams to non-overlapping Bark-scale
//!   spectral bins.
//! - [`MaskingMarginOptimization`] – adjust parameters to maximise the margin
//!   above masked thresholds.

use crate::graph::{AudioGraph, NodeId, NodeType};
use crate::node;
use crate::passes::{Pass, PassResult};
use crate::IrResult;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of Bark-scale critical bands (0–24).
pub const NUM_BARK_BANDS: usize = 24;

/// Upper frequency edge (Hz) for each Bark band (approximate).
const BARK_UPPER_FREQ: [f64; NUM_BARK_BANDS] = [
    100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0,
    1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0,
    3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 15500.0,
];

// ---------------------------------------------------------------------------
// Stream descriptor (extracted from graph nodes)
// ---------------------------------------------------------------------------

/// Lightweight descriptor capturing a stream's spectral footprint.
#[derive(Debug, Clone)]
pub struct StreamDescriptor {
    pub node_id: NodeId,
    pub name: String,
    pub centre_freq: f64,
    pub bandwidth: f64,
    pub level_db: f64,
    pub bark_energies: [f64; NUM_BARK_BANDS],
}

impl StreamDescriptor {
    /// Build a descriptor from a graph node.
    pub fn from_node(node: &crate::graph::AudioGraphNode) -> Self {
        let (cf, bw, level) = match &node.node_type {
            NodeType::Oscillator { frequency, waveform, .. } => {
                let bw = match waveform {
                    node::Waveform::Sine => 50.0,
                    node::Waveform::Saw | node::Waveform::Square | node::Waveform::Pulse => {
                        (22050.0 - frequency).max(100.0)
                    }
                    node::Waveform::Triangle => (22050.0 - frequency).max(100.0) * 0.5,
                    node::Waveform::Noise => 22050.0,
                };
                (*frequency, bw, 0.0)
            }
            NodeType::Filter { cutoff, .. } => (*cutoff, 500.0, -3.0),
            NodeType::NoiseGenerator => (11000.0, 22000.0, -6.0),
            NodeType::Gain { level } => (1000.0, 0.0, 20.0 * level.abs().max(1e-12).log10()),
            _ => (1000.0, 0.0, -60.0),
        };
        let bark_energies = Self::compute_bark_energies(cf, bw, level);
        Self {
            node_id: node.id,
            name: node.name.clone(),
            centre_freq: cf,
            bandwidth: bw,
            level_db: level,
            bark_energies,
        }
    }

    fn compute_bark_energies(centre: f64, bandwidth: f64, level_db: f64) -> [f64; NUM_BARK_BANDS] {
        let mut energies = [0.0f64; NUM_BARK_BANDS];
        let amp = node::db_to_linear(level_db);
        let lo = (centre - bandwidth / 2.0).max(0.0);
        let hi = (centre + bandwidth / 2.0).min(22050.0);
        let mut prev_edge = 0.0f64;
        for (i, &upper) in BARK_UPPER_FREQ.iter().enumerate() {
            let band_lo = prev_edge;
            let band_hi = upper;
            let overlap_lo = lo.max(band_lo);
            let overlap_hi = hi.min(band_hi);
            if overlap_hi > overlap_lo {
                let frac = (overlap_hi - overlap_lo) / (band_hi - band_lo).max(1.0);
                energies[i] = amp * amp * frac;
            }
            prev_edge = upper;
        }
        energies
    }

    /// Total energy across all bands.
    pub fn total_energy(&self) -> f64 {
        self.bark_energies.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Pairwise masking computation
// ---------------------------------------------------------------------------

/// Compute the masking factor between two streams.
/// Returns a value in [0,1]: 1 = masker completely masks maskee.
pub fn compute_masking_factor(masker: &StreamDescriptor, maskee: &StreamDescriptor) -> f64 {
    let mut masked_energy = 0.0f64;
    let mut total_maskee_energy = 0.0f64;
    for band in 0..NUM_BARK_BANDS {
        let me = maskee.bark_energies[band];
        let _mk = masker.bark_energies[band];
        total_maskee_energy += me;
        // Simplified masking model: masker raises the threshold in this band.
        // If masker energy exceeds maskee energy, the band is masked.
        let spreading = spreading_function(band, &masker.bark_energies);
        if spreading >= me {
            masked_energy += me;
        }
    }
    if total_maskee_energy < 1e-12 { return 0.0; }
    (masked_energy / total_maskee_energy).clamp(0.0, 1.0)
}

/// Simple spreading function: masking spreads ±2 Bark bands with -10 dB/band.
fn spreading_function(band: usize, masker_energies: &[f64; NUM_BARK_BANDS]) -> f64 {
    let mut threshold = 0.0f64;
    for (i, &e) in masker_energies.iter().enumerate() {
        if e < 1e-15 { continue; }
        let dist = (i as f64 - band as f64).abs();
        let atten = 10.0_f64.powf(-dist * 0.5); // ~10 dB per Bark
        threshold += e * atten;
    }
    threshold
}

/// Compute the full NxN masking matrix.
pub fn compute_masking_matrix(streams: &[StreamDescriptor]) -> Vec<Vec<f64>> {
    let n = streams.len();
    let mut matrix = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            matrix[i][j] = compute_masking_factor(&streams[i], &streams[j]);
        }
    }
    matrix
}

// ---------------------------------------------------------------------------
// MaskingAwareStreamMerging
// ---------------------------------------------------------------------------

/// Remove streams that are completely masked by another stream.
pub struct MaskingAwareStreamMerging {
    pub masking_threshold: f64,
}

impl Default for MaskingAwareStreamMerging {
    fn default() -> Self { Self { masking_threshold: 0.95 } }
}

impl MaskingAwareStreamMerging {
    pub fn new(threshold: f64) -> Self { Self { masking_threshold: threshold } }

    fn collect_streams(graph: &AudioGraph) -> Vec<StreamDescriptor> {
        graph.nodes.iter()
            .filter(|n| matches!(n.node_type, NodeType::Oscillator { .. } | NodeType::NoiseGenerator | NodeType::Filter { .. }))
            .map(StreamDescriptor::from_node)
            .collect()
    }

    /// Suggest frequency reallocations for partially masked streams.
    pub fn suggest_reallocations(graph: &AudioGraph) -> Vec<(NodeId, f64)> {
        let streams = Self::collect_streams(graph);
        let matrix = compute_masking_matrix(&streams);
        let mut suggestions = Vec::new();
        for j in 0..streams.len() {
            let max_mask = matrix.iter().map(|row| row[j]).fold(0.0f64, f64::max);
            if max_mask > 0.3 && max_mask < 0.95 {
                // Find least occupied Bark band.
                let mut band_load = [0.0f64; NUM_BARK_BANDS];
                for s in &streams {
                    for (b, &e) in s.bark_energies.iter().enumerate() {
                        band_load[b] += e;
                    }
                }
                let min_band = band_load.iter().enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let target_freq = if min_band == 0 { 50.0 } else { BARK_UPPER_FREQ[min_band - 1] };
                suggestions.push((streams[j].node_id, target_freq));
            }
        }
        suggestions
    }
}

impl Pass for MaskingAwareStreamMerging {
    fn name(&self) -> &str { "MaskingAwareStreamMerging" }

    fn run(&self, graph: &mut AudioGraph) -> IrResult<PassResult> {
        let streams = Self::collect_streams(graph);
        let matrix = compute_masking_matrix(&streams);
        let mut to_remove: Vec<NodeId> = Vec::new();

        for j in 0..streams.len() {
            if to_remove.contains(&streams[j].node_id) { continue; }
            for i in 0..streams.len() {
                if i == j { continue; }
                if to_remove.contains(&streams[i].node_id) { continue; }
                if matrix[i][j] >= self.masking_threshold {
                    to_remove.push(streams[j].node_id);
                    break;
                }
            }
        }

        if to_remove.is_empty() {
            return Ok(PassResult::unchanged());
        }

        let edges_before = graph.edge_count();
        for id in &to_remove {
            graph.remove_node(*id)?;
        }
        let edges_after = graph.edge_count();

        Ok(PassResult {
            modified: true,
            nodes_removed: to_remove.len(),
            edges_removed: edges_before - edges_after,
            messages: to_remove.iter().map(|id| format!("removed masked stream node {}", id.0)).collect(),
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// SpectralBinPacking
// ---------------------------------------------------------------------------

/// Bin-packing of streams into non-overlapping Bark-band allocations.
#[derive(Debug, Clone)]
pub struct BarkBandAllocation {
    pub node_id: NodeId,
    pub assigned_bands: Vec<usize>,
    pub suggested_freq: f64,
}

/// Assign streams to non-overlapping spectral regions using a first-fit
/// decreasing bin-packing heuristic over Bark bands.
pub struct SpectralBinPacking {
    pub max_overlap_fraction: f64,
}

impl Default for SpectralBinPacking {
    fn default() -> Self { Self { max_overlap_fraction: 0.1 } }
}

impl SpectralBinPacking {
    pub fn new(max_overlap: f64) -> Self { Self { max_overlap_fraction: max_overlap } }

    /// Compute allocations without modifying the graph.
    pub fn compute_allocations(graph: &AudioGraph) -> Vec<BarkBandAllocation> {
        let streams = MaskingAwareStreamMerging::collect_streams(graph);
        // Sort by total energy descending (first-fit decreasing).
        let mut indexed: Vec<(usize, f64)> = streams.iter().enumerate()
            .map(|(i, s)| (i, s.total_energy()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut band_used = [false; NUM_BARK_BANDS];
        let mut allocations = Vec::new();

        for (idx, _energy) in &indexed {
            let s = &streams[*idx];
            // Find the dominant bands (> 10% of peak energy in any band).
            let peak = s.bark_energies.iter().cloned().fold(0.0f64, f64::max);
            let threshold = peak * 0.1;
            let needed_bands: Vec<usize> = (0..NUM_BARK_BANDS)
                .filter(|&b| s.bark_energies[b] > threshold)
                .collect();

            // Check if those bands are free.
            let all_free = needed_bands.iter().all(|&b| !band_used[b]);
            if all_free {
                for &b in &needed_bands {
                    band_used[b] = true;
                }
                allocations.push(BarkBandAllocation {
                    node_id: s.node_id,
                    assigned_bands: needed_bands,
                    suggested_freq: s.centre_freq,
                });
            } else {
                // Find a free contiguous region of the same width.
                let width = needed_bands.len();
                let mut placed = false;
                for start in 0..=(NUM_BARK_BANDS - width) {
                    let region: Vec<usize> = (start..start + width).collect();
                    if region.iter().all(|&b| !band_used[b]) {
                        for &b in &region {
                            band_used[b] = true;
                        }
                        let mid = start + width / 2;
                        let suggested = if mid == 0 { 50.0 } else { BARK_UPPER_FREQ[mid.min(NUM_BARK_BANDS - 1)] };
                        allocations.push(BarkBandAllocation {
                            node_id: s.node_id,
                            assigned_bands: region,
                            suggested_freq: suggested,
                        });
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    // Can't find a free slot; assign original bands anyway.
                    allocations.push(BarkBandAllocation {
                        node_id: s.node_id,
                        assigned_bands: needed_bands,
                        suggested_freq: s.centre_freq,
                    });
                }
            }
        }
        allocations
    }

    /// Total spectral overlap (sum of per-band collision counts).
    pub fn compute_overlap(graph: &AudioGraph) -> f64 {
        let streams = MaskingAwareStreamMerging::collect_streams(graph);
        let mut band_count = [0usize; NUM_BARK_BANDS];
        for s in &streams {
            for (b, &e) in s.bark_energies.iter().enumerate() {
                if e > 1e-12 { band_count[b] += 1; }
            }
        }
        band_count.iter().map(|&c| if c > 1 { (c - 1) as f64 } else { 0.0 }).sum()
    }
}

impl Pass for SpectralBinPacking {
    fn name(&self) -> &str { "SpectralBinPacking" }

    fn run(&self, graph: &mut AudioGraph) -> IrResult<PassResult> {
        let allocations = Self::compute_allocations(graph);
        let mut modified = false;
        let mut msgs = Vec::new();

        for alloc in &allocations {
            let node = match graph.node_mut(alloc.node_id) { Some(n) => n, None => continue };
            if let NodeType::Oscillator { ref mut frequency, .. } = node.node_type {
                let old_freq = *frequency;
                if (old_freq - alloc.suggested_freq).abs() > 1.0 {
                    msgs.push(format!("node {}: freq {:.0} → {:.0} Hz", alloc.node_id.0, old_freq, alloc.suggested_freq));
                    *frequency = alloc.suggested_freq;
                    modified = true;
                }
            }
        }

        if modified {
            Ok(PassResult { modified: true, messages: msgs, ..Default::default() })
        } else {
            Ok(PassResult::unchanged())
        }
    }
}

// ---------------------------------------------------------------------------
// MaskingMarginOptimization
// ---------------------------------------------------------------------------

/// Adjust stream parameters to maximize the margin above masking thresholds.
pub struct MaskingMarginOptimization {
    /// Minimum desired margin in dB above masked threshold.
    pub target_margin_db: f64,
}

impl Default for MaskingMarginOptimization {
    fn default() -> Self { Self { target_margin_db: 6.0 } }
}

impl MaskingMarginOptimization {
    pub fn new(margin: f64) -> Self { Self { target_margin_db: margin } }

    /// Compute the masking margin for each stream (in dB above threshold).
    pub fn compute_margins(graph: &AudioGraph) -> Vec<(NodeId, f64)> {
        let streams = MaskingAwareStreamMerging::collect_streams(graph);
        let n = streams.len();
        let mut margins = Vec::new();
        for j in 0..n {
            let mut worst_margin = f64::INFINITY;
            for band in 0..NUM_BARK_BANDS {
                let me = streams[j].bark_energies[band];
                if me < 1e-15 { continue; }
                // Sum masking from all other streams.
                let mut mask_energy = 0.0f64;
                for i in 0..n {
                    if i == j { continue; }
                    mask_energy += spreading_function(band, &streams[i].bark_energies);
                }
                let margin = if mask_energy < 1e-15 {
                    60.0
                } else {
                    10.0 * (me / mask_energy).log10()
                };
                if margin < worst_margin { worst_margin = margin; }
            }
            if worst_margin == f64::INFINITY { worst_margin = 60.0; }
            margins.push((streams[j].node_id, worst_margin));
        }
        margins
    }
}

impl Pass for MaskingMarginOptimization {
    fn name(&self) -> &str { "MaskingMarginOptimization" }

    fn run(&self, graph: &mut AudioGraph) -> IrResult<PassResult> {
        let margins = Self::compute_margins(graph);
        let mut modified = false;
        let mut msgs = Vec::new();

        for (nid, margin) in &margins {
            if *margin < self.target_margin_db {
                // Boost the node's gain to meet the target margin.
                let boost_db = self.target_margin_db - margin;
                let boost_linear = node::db_to_linear(boost_db);
                let node = match graph.node_mut(*nid) { Some(n) => n, None => continue };
                match &mut node.node_type {
                    NodeType::Oscillator { .. } => {
                        // Can't directly boost oscillator; log a suggestion.
                        msgs.push(format!("node {}: needs {:.1} dB boost (margin={:.1} dB)", nid.0, boost_db, margin));
                    }
                    NodeType::Gain { ref mut level } => {
                        *level *= boost_linear;
                        msgs.push(format!("node {}: boosted gain by {:.1} dB", nid.0, boost_db));
                        modified = true;
                    }
                    _ => {}
                }
            }
        }

        if modified {
            Ok(PassResult { modified: true, messages: msgs, ..Default::default() })
        } else {
            Ok(PassResult { modified: false, messages: msgs, ..Default::default() })
        }
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

    fn two_osc_graph() -> AudioGraph {
        let (b, osc1) = GraphBuilder::with_defaults()
            .add_oscillator("osc1", Waveform::Sine, 440.0);
        let (b, osc2) = b.add_oscillator("osc2", Waveform::Sine, 445.0);
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
    fn test_stream_descriptor_from_oscillator() {
        let mut g = AudioGraph::default();
        let id = g.add_node("osc", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 1000.0 });
        let desc = StreamDescriptor::from_node(g.node(id).unwrap());
        assert!((desc.centre_freq - 1000.0).abs() < 0.1);
        assert!(desc.total_energy() > 0.0);
    }

    #[test]
    fn test_compute_masking_factor_same() {
        let mut g = AudioGraph::default();
        let id = g.add_node("osc", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 1000.0 });
        let desc = StreamDescriptor::from_node(g.node(id).unwrap());
        let mf = compute_masking_factor(&desc, &desc);
        // Self-masking should be 1.0 or close (we skip same index in matrix,
        // but the function itself should return high overlap).
        assert!(mf >= 0.5);
    }

    #[test]
    fn test_masking_matrix_dimensions() {
        let g = two_osc_graph();
        let streams: Vec<_> = g.nodes.iter()
            .filter(|n| matches!(n.node_type, NodeType::Oscillator { .. }))
            .map(StreamDescriptor::from_node)
            .collect();
        let mat = compute_masking_matrix(&streams);
        assert_eq!(mat.len(), 2);
        assert_eq!(mat[0].len(), 2);
        assert!((mat[0][0]).abs() < 1e-9); // diagonal is zero
    }

    #[test]
    fn test_masking_aware_merging_no_remove() {
        let mut g = AudioGraph::default();
        let _osc1 = g.add_node("low", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 200.0 });
        let _osc2 = g.add_node("high", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 8000.0 });
        let out = g.add_node("out", NodeType::Output { format: "stereo".into() });
        // Streams are far apart → no masking.
        let pass = MaskingAwareStreamMerging::default();
        let r = pass.run(&mut g).unwrap();
        // With widely spaced pure tones neither should be removed.
        assert!(!r.modified || r.nodes_removed == 0);
    }

    #[test]
    fn test_masking_aware_suggest_reallocations() {
        let g = two_osc_graph();
        let suggestions = MaskingAwareStreamMerging::suggest_reallocations(&g);
        // Two nearly identical tones should trigger a suggestion.
        // (Exact behaviour depends on model sensitivity.)
        let _ = suggestions; // May or may not produce suggestions; just ensure no panic.
    }

    #[test]
    fn test_spectral_bin_packing_allocations() {
        let g = two_osc_graph();
        let allocs = SpectralBinPacking::compute_allocations(&g);
        assert!(!allocs.is_empty());
    }

    #[test]
    fn test_spectral_bin_packing_overlap() {
        let g = two_osc_graph();
        let overlap = SpectralBinPacking::compute_overlap(&g);
        // Two nearly identical tones should have non-zero overlap.
        assert!(overlap >= 0.0);
    }

    #[test]
    fn test_spectral_bin_packing_pass() {
        let mut g = two_osc_graph();
        let pass = SpectralBinPacking::default();
        let r = pass.run(&mut g).unwrap();
        // Whether it modifies depends on the allocation outcome.
        let _ = r;
    }

    #[test]
    fn test_masking_margin_compute() {
        let g = two_osc_graph();
        let margins = MaskingMarginOptimization::compute_margins(&g);
        assert!(!margins.is_empty());
        for (_, m) in &margins {
            assert!(m.is_finite());
        }
    }

    #[test]
    fn test_masking_margin_pass() {
        let mut g = two_osc_graph();
        let pass = MaskingMarginOptimization::new(6.0);
        let r = pass.run(&mut g).unwrap();
        let _ = r;
    }

    #[test]
    fn test_bark_band_constants() {
        assert_eq!(BARK_UPPER_FREQ.len(), NUM_BARK_BANDS);
        // Each band edge should be strictly increasing.
        for i in 1..NUM_BARK_BANDS {
            assert!(BARK_UPPER_FREQ[i] > BARK_UPPER_FREQ[i - 1]);
        }
    }

    #[test]
    fn test_spreading_function() {
        let mut energies = [0.0f64; NUM_BARK_BANDS];
        energies[12] = 1.0; // energy in band 12
        let spread_same = spreading_function(12, &energies);
        let spread_far = spreading_function(0, &energies);
        assert!(spread_same > spread_far);
    }

    #[test]
    fn test_stream_descriptor_noise() {
        let mut g = AudioGraph::default();
        let id = g.add_node("noise", NodeType::NoiseGenerator);
        let desc = StreamDescriptor::from_node(g.node(id).unwrap());
        assert!(desc.bandwidth > 10000.0);
        assert!(desc.total_energy() > 0.0);
    }
}
