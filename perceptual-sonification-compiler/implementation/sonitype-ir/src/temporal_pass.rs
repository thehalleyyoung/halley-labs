//! Temporal scheduling passes.
//!
//! - [`TemporalScheduler`] – schedule stream activations to minimise temporal
//!   overlap and cognitive load.
//! - [`TemporalBinPacking`] – assign streams to non-overlapping time slots
//!   (first-fit decreasing, accounting for attack/release tails).
//! - [`LatencyAnalysis`] – compute end-to-end latency through the audio graph
//!   and identify the critical path.

use std::collections::HashMap;
use crate::graph::{AudioGraph, NodeId, NodeType};
use crate::passes::{Pass, PassResult};
use crate::IrResult;

// ---------------------------------------------------------------------------
// TemporalDescriptor
// ---------------------------------------------------------------------------

/// Temporal occupancy of a single stream.
#[derive(Debug, Clone)]
pub struct TemporalDescriptor {
    pub node_id: NodeId,
    pub onset: f64,
    pub duration: f64,
    pub attack: f64,
    pub release: f64,
}

impl TemporalDescriptor {
    /// Effective end time including release tail.
    pub fn effective_end(&self) -> f64 {
        self.onset + self.duration + self.release
    }

    /// Effective start including anticipatory processing (pre-onset).
    pub fn effective_start(&self) -> f64 {
        (self.onset - self.attack * 0.1).max(0.0)
    }

    /// Does this descriptor overlap with another?
    pub fn overlaps(&self, other: &Self) -> bool {
        self.effective_start() < other.effective_end()
            && other.effective_start() < self.effective_end()
    }

    /// Amount of temporal overlap (seconds).
    pub fn overlap_amount(&self, other: &Self) -> f64 {
        let start = self.effective_start().max(other.effective_start());
        let end = self.effective_end().min(other.effective_end());
        (end - start).max(0.0)
    }
}

/// Extract temporal descriptors from the graph.
pub fn extract_temporal_descriptors(graph: &AudioGraph) -> Vec<TemporalDescriptor> {
    let mut descs = Vec::new();
    for node in &graph.nodes {
        let (onset, dur, atk, rel) = match &node.node_type {
            NodeType::Envelope { attack, decay, sustain: _, release } => {
                (0.0, attack + decay + 1.0 + release, *attack, *release)
            }
            NodeType::Oscillator { .. } => (0.0, 2.0, 0.01, 0.01),
            NodeType::NoiseGenerator => (0.0, 1.0, 0.005, 0.005),
            NodeType::Delay { samples } => {
                let lat = *samples as f64 / graph.sample_rate;
                (lat, 1.0, 0.0, 0.0)
            }
            _ => continue,
        };
        descs.push(TemporalDescriptor {
            node_id: node.id,
            onset,
            duration: dur,
            attack: atk,
            release: rel,
        });
    }
    descs
}

// ---------------------------------------------------------------------------
// TemporalScheduler
// ---------------------------------------------------------------------------

/// Schedule stream activations to minimise simultaneous cognitive load.
pub struct TemporalScheduler {
    /// Maximum number of streams allowed to overlap simultaneously.
    pub max_concurrent_streams: usize,
    /// Minimum gap (seconds) between stream onsets for segregation.
    pub onset_gap: f64,
}

impl Default for TemporalScheduler {
    fn default() -> Self {
        Self { max_concurrent_streams: 3, onset_gap: 0.15 }
    }
}

/// A scheduled activation.
#[derive(Debug, Clone)]
pub struct ScheduledStream {
    pub node_id: NodeId,
    pub scheduled_onset: f64,
    pub original_onset: f64,
    pub duration: f64,
}

impl TemporalScheduler {
    pub fn new(max_concurrent: usize, onset_gap: f64) -> Self {
        Self { max_concurrent_streams: max_concurrent, onset_gap }
    }

    /// Compute a schedule that respects the maximum concurrency constraint.
    pub fn compute_schedule(&self, descriptors: &[TemporalDescriptor]) -> Vec<ScheduledStream> {
        if descriptors.is_empty() { return Vec::new(); }

        // Sort by original onset (stable).
        let mut indexed: Vec<(usize, f64)> = descriptors.iter()
            .enumerate()
            .map(|(i, d)| (i, d.onset))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut schedule: Vec<ScheduledStream> = Vec::new();
        let mut active_ends: Vec<f64> = Vec::new(); // end times of currently active streams

        for (idx, _) in &indexed {
            let desc = &descriptors[*idx];

            // Remove expired streams.
            let candidate_onset = desc.onset;
            active_ends.retain(|&end| end > candidate_onset);

            let mut scheduled_onset = candidate_onset;

            // Enforce onset gap.
            if let Some(last) = schedule.last() {
                let min_start = last.scheduled_onset + self.onset_gap;
                if scheduled_onset < min_start {
                    scheduled_onset = min_start;
                }
            }

            // Enforce max concurrency.
            while active_ends.len() >= self.max_concurrent_streams {
                // Push onset to after the earliest ending stream.
                let earliest_end = active_ends.iter().cloned().fold(f64::INFINITY, f64::min);
                scheduled_onset = scheduled_onset.max(earliest_end + 0.001);
                active_ends.retain(|&end| end > scheduled_onset);
            }

            active_ends.push(scheduled_onset + desc.duration + desc.release);

            schedule.push(ScheduledStream {
                node_id: desc.node_id,
                scheduled_onset,
                original_onset: desc.onset,
                duration: desc.duration,
            });
        }
        schedule
    }

    /// Compute the total cognitive load over time (streams × seconds).
    pub fn cognitive_load_integral(schedule: &[ScheduledStream]) -> f64 {
        if schedule.is_empty() { return 0.0; }
        // Simple approximation: sum of durations (≈ area under the concurrency curve).
        schedule.iter().map(|s| s.duration).sum()
    }

    /// Peak simultaneous stream count.
    pub fn peak_concurrency(schedule: &[ScheduledStream]) -> usize {
        if schedule.is_empty() { return 0; }
        let mut events: Vec<(f64, i32)> = Vec::new();
        for s in schedule {
            events.push((s.scheduled_onset, 1));
            events.push((s.scheduled_onset + s.duration, -1));
        }
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut current = 0i32;
        let mut peak = 0i32;
        for (_, delta) in events {
            current += delta;
            if current > peak { peak = current; }
        }
        peak.max(0) as usize
    }
}

impl Pass for TemporalScheduler {
    fn name(&self) -> &str { "TemporalScheduler" }

    fn run(&self, graph: &mut AudioGraph) -> IrResult<PassResult> {
        let descs = extract_temporal_descriptors(graph);
        let schedule = self.compute_schedule(&descs);
        let peak = Self::peak_concurrency(&schedule);
        let load = Self::cognitive_load_integral(&schedule);

        Ok(PassResult {
            modified: false,
            messages: vec![
                format!("scheduled {} streams", schedule.len()),
                format!("peak concurrency: {}", peak),
                format!("cognitive load integral: {:.2}s", load),
            ],
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// TemporalBinPacking
// ---------------------------------------------------------------------------

/// Assign streams to non-overlapping time slots using a first-fit decreasing
/// algorithm adapted for audio (accounts for attack/release tails).
pub struct TemporalBinPacking {
    /// Minimum silence gap between consecutive assignments in a slot.
    pub min_gap: f64,
}

impl Default for TemporalBinPacking {
    fn default() -> Self { Self { min_gap: 0.05 } }
}

/// A time slot containing a sequence of stream assignments.
#[derive(Debug, Clone)]
pub struct TimeSlot {
    pub slot_index: usize,
    pub assignments: Vec<(NodeId, f64, f64)>, // (node_id, onset, end)
}

impl TimeSlot {
    pub fn end_time(&self) -> f64 {
        self.assignments.iter().map(|(_, _, e)| *e).fold(0.0f64, f64::max)
    }

    pub fn total_duration(&self) -> f64 {
        self.assignments.iter().map(|(_, s, e)| e - s).sum()
    }
}

impl TemporalBinPacking {
    pub fn new(min_gap: f64) -> Self { Self { min_gap } }

    /// Run the bin-packing algorithm over the given descriptors.
    pub fn pack(&self, descriptors: &[TemporalDescriptor]) -> Vec<TimeSlot> {
        if descriptors.is_empty() { return Vec::new(); }

        // Sort by effective duration descending (FFD).
        let mut indexed: Vec<(usize, f64)> = descriptors.iter()
            .enumerate()
            .map(|(i, d)| (i, d.effective_end() - d.effective_start()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut slots: Vec<TimeSlot> = Vec::new();

        for (idx, _) in &indexed {
            let desc = &descriptors[*idx];
            let stream_dur = desc.effective_end() - desc.effective_start();
            let mut placed = false;

            // First-fit: find the first slot where this stream fits.
            for slot in slots.iter_mut() {
                let slot_end = slot.end_time();
                let gap_needed = self.min_gap + desc.attack * 0.1;
                let proposed_onset = slot_end + gap_needed;
                // We simply append at the end of the slot.
                slot.assignments.push((desc.node_id, proposed_onset, proposed_onset + stream_dur));
                placed = true;
                break;
            }
            if !placed {
                let new_slot = TimeSlot {
                    slot_index: slots.len(),
                    assignments: vec![(desc.node_id, 0.0, stream_dur)],
                };
                slots.push(new_slot);
            }
        }
        slots
    }

    /// Number of slots required.
    pub fn slot_count(&self, descriptors: &[TemporalDescriptor]) -> usize {
        self.pack(descriptors).len()
    }
}

impl Pass for TemporalBinPacking {
    fn name(&self) -> &str { "TemporalBinPacking" }

    fn run(&self, graph: &mut AudioGraph) -> IrResult<PassResult> {
        let descs = extract_temporal_descriptors(graph);
        let slots = self.pack(&descs);
        let msgs = vec![
            format!("packed {} streams into {} slots", descs.len(), slots.len()),
        ];
        Ok(PassResult { modified: false, messages: msgs, ..Default::default() })
    }
}

// ---------------------------------------------------------------------------
// LatencyAnalysis
// ---------------------------------------------------------------------------

/// Compute end-to-end latency through the audio graph.
pub struct LatencyAnalysis;

/// Per-node latency contribution.
#[derive(Debug, Clone)]
pub struct NodeLatency {
    pub node_id: NodeId,
    pub name: String,
    pub latency_samples: usize,
    pub latency_seconds: f64,
}

/// Result of latency analysis.
#[derive(Debug, Clone)]
pub struct LatencyReport {
    pub per_node: Vec<NodeLatency>,
    pub critical_path: Vec<NodeId>,
    pub critical_path_latency_samples: usize,
    pub critical_path_latency_seconds: f64,
    pub total_latency_samples: usize,
    pub total_latency_seconds: f64,
}

impl LatencyAnalysis {
    /// Estimate the latency (in samples) introduced by a single node.
    pub fn node_latency_samples(node: &crate::graph::AudioGraphNode) -> usize {
        match &node.node_type {
            NodeType::Delay { samples } => *samples,
            NodeType::Filter { .. } => 2,
            NodeType::PitchShifter => 2048,
            NodeType::TimeStretch => 4096,
            NodeType::Compressor => {
                // Lookahead of compressor contributes.
                64
            }
            NodeType::Limiter => {
                // Lookahead.
                (0.005 * node.sample_rate).round() as usize
            }
            _ => 0,
        }
    }

    /// Run the full latency analysis.
    pub fn analyze(graph: &mut AudioGraph) -> IrResult<LatencyReport> {
        graph.ensure_sorted()?;
        let order = graph.topological_order.clone();
        let sr = graph.sample_rate;

        // Per-node latency.
        let per_node: Vec<NodeLatency> = graph.nodes.iter().map(|n| {
            let samp = Self::node_latency_samples(n);
            NodeLatency {
                node_id: n.id,
                name: n.name.clone(),
                latency_samples: samp,
                latency_seconds: samp as f64 / sr,
            }
        }).collect();

        // Critical path (longest latency from source to sink).
        let mut dist: HashMap<NodeId, usize> = HashMap::new();
        let mut pred: HashMap<NodeId, Option<NodeId>> = HashMap::new();

        for &nid in &order {
            let node_lat = graph.node(nid)
                .map(|n| Self::node_latency_samples(n))
                .unwrap_or(0);
            let (max_pred_dist, max_pred_id) = graph.predecessors(nid).iter()
                .filter_map(|&p| dist.get(&p).map(|&d| (d, Some(p))))
                .max_by_key(|&(d, _)| d)
                .unwrap_or((0, None));
            dist.insert(nid, max_pred_dist + node_lat);
            pred.insert(nid, max_pred_id);
        }

        // Find the output node with the highest latency.
        let output_nodes = graph.output_nodes();
        let (critical_end, critical_lat) = output_nodes.iter()
            .filter_map(|&id| dist.get(&id).map(|&d| (id, d)))
            .max_by_key(|&(_, d)| d)
            .unwrap_or_else(|| {
                dist.iter().max_by_key(|&(_, &d)| d)
                    .map(|(&id, &d)| (id, d))
                    .unwrap_or((NodeId(0), 0))
            });

        // Reconstruct the critical path.
        let mut path = vec![critical_end];
        let mut cur = critical_end;
        while let Some(Some(p)) = pred.get(&cur) {
            path.push(*p);
            cur = *p;
        }
        path.reverse();

        let total_samples: usize = per_node.iter().map(|n| n.latency_samples).sum();

        Ok(LatencyReport {
            per_node,
            critical_path: path,
            critical_path_latency_samples: critical_lat,
            critical_path_latency_seconds: critical_lat as f64 / sr,
            total_latency_samples: total_samples,
            total_latency_seconds: total_samples as f64 / sr,
        })
    }
}

impl Pass for LatencyAnalysis {
    fn name(&self) -> &str { "LatencyAnalysis" }

    fn run(&self, graph: &mut AudioGraph) -> IrResult<PassResult> {
        let report = Self::analyze(graph)?;
        Ok(PassResult {
            modified: false,
            messages: vec![
                format!("critical path latency: {} samples ({:.4}s)",
                    report.critical_path_latency_samples,
                    report.critical_path_latency_seconds),
                format!("critical path: {:?}",
                    report.critical_path.iter().map(|id| id.0).collect::<Vec<_>>()),
            ],
            ..Default::default()
        })
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

    fn osc_delay_out() -> AudioGraph {
        let (b, osc) = GraphBuilder::with_defaults()
            .add_oscillator("osc", Waveform::Sine, 440.0);
        let (b, del) = b.add_delay("delay", 4800);
        let (b, out) = b.add_output("out", "stereo");
        b.connect(osc, "out", del, "in")
         .connect(del, "out", out, "in")
         .build().unwrap()
    }

    #[test]
    fn test_temporal_descriptor_overlap() {
        let a = TemporalDescriptor { node_id: NodeId(1), onset: 0.0, duration: 1.0, attack: 0.01, release: 0.1 };
        let b = TemporalDescriptor { node_id: NodeId(2), onset: 0.5, duration: 1.0, attack: 0.01, release: 0.1 };
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_temporal_descriptor_no_overlap() {
        let a = TemporalDescriptor { node_id: NodeId(1), onset: 0.0, duration: 0.5, attack: 0.01, release: 0.01 };
        let b = TemporalDescriptor { node_id: NodeId(2), onset: 2.0, duration: 0.5, attack: 0.01, release: 0.01 };
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_temporal_descriptor_overlap_amount() {
        let a = TemporalDescriptor { node_id: NodeId(1), onset: 0.0, duration: 1.0, attack: 0.0, release: 0.0 };
        let b = TemporalDescriptor { node_id: NodeId(2), onset: 0.5, duration: 1.0, attack: 0.0, release: 0.0 };
        let overlap = a.overlap_amount(&b);
        assert!((overlap - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_extract_descriptors() {
        let g = osc_delay_out();
        let descs = extract_temporal_descriptors(&g);
        // Should find the oscillator and the delay.
        assert!(descs.len() >= 2);
    }

    #[test]
    fn test_temporal_scheduler_schedule() {
        let descs = vec![
            TemporalDescriptor { node_id: NodeId(1), onset: 0.0, duration: 1.0, attack: 0.01, release: 0.1 },
            TemporalDescriptor { node_id: NodeId(2), onset: 0.0, duration: 1.0, attack: 0.01, release: 0.1 },
            TemporalDescriptor { node_id: NodeId(3), onset: 0.0, duration: 1.0, attack: 0.01, release: 0.1 },
            TemporalDescriptor { node_id: NodeId(4), onset: 0.0, duration: 1.0, attack: 0.01, release: 0.1 },
        ];
        let sched = TemporalScheduler::new(2, 0.15);
        let schedule = sched.compute_schedule(&descs);
        assert_eq!(schedule.len(), 4);
        let peak = TemporalScheduler::peak_concurrency(&schedule);
        assert!(peak <= 2);
    }

    #[test]
    fn test_temporal_scheduler_empty() {
        let sched = TemporalScheduler::default();
        let schedule = sched.compute_schedule(&[]);
        assert!(schedule.is_empty());
    }

    #[test]
    fn test_temporal_scheduler_cognitive_load() {
        let schedule = vec![
            ScheduledStream { node_id: NodeId(1), scheduled_onset: 0.0, original_onset: 0.0, duration: 1.0 },
            ScheduledStream { node_id: NodeId(2), scheduled_onset: 0.5, original_onset: 0.5, duration: 1.0 },
        ];
        let load = TemporalScheduler::cognitive_load_integral(&schedule);
        assert!((load - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_temporal_bin_packing() {
        let descs = vec![
            TemporalDescriptor { node_id: NodeId(1), onset: 0.0, duration: 2.0, attack: 0.01, release: 0.1 },
            TemporalDescriptor { node_id: NodeId(2), onset: 0.0, duration: 1.0, attack: 0.01, release: 0.1 },
            TemporalDescriptor { node_id: NodeId(3), onset: 0.0, duration: 0.5, attack: 0.01, release: 0.05 },
        ];
        let packer = TemporalBinPacking::new(0.05);
        let slots = packer.pack(&descs);
        assert!(!slots.is_empty());
    }

    #[test]
    fn test_temporal_bin_packing_slot_count() {
        let descs = vec![
            TemporalDescriptor { node_id: NodeId(1), onset: 0.0, duration: 1.0, attack: 0.0, release: 0.0 },
        ];
        let packer = TemporalBinPacking::default();
        assert_eq!(packer.slot_count(&descs), 1);
    }

    #[test]
    fn test_latency_analysis() {
        let mut g = osc_delay_out();
        let report = LatencyAnalysis::analyze(&mut g).unwrap();
        assert!(report.critical_path_latency_samples >= 4800);
    }

    #[test]
    fn test_latency_analysis_no_delay() {
        let (b, osc) = GraphBuilder::with_defaults()
            .add_oscillator("osc", Waveform::Sine, 440.0);
        let (b, out) = b.add_output("out", "stereo");
        let mut g = b.connect(osc, "out", out, "in").build().unwrap();
        let report = LatencyAnalysis::analyze(&mut g).unwrap();
        assert_eq!(report.critical_path_latency_samples, 0);
    }

    #[test]
    fn test_latency_analysis_pass() {
        let mut g = osc_delay_out();
        let pass = LatencyAnalysis;
        let r = pass.run(&mut g).unwrap();
        assert!(!r.messages.is_empty());
    }

    #[test]
    fn test_temporal_scheduler_pass() {
        let mut g = osc_delay_out();
        let pass = TemporalScheduler::default();
        let r = pass.run(&mut g).unwrap();
        assert!(!r.messages.is_empty());
    }
}
