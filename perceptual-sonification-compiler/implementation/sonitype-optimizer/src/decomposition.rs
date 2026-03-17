//! Problem decomposition strategies for sonification optimization.
//!
//! Decomposes the global optimization problem into smaller subproblems
//! that can be solved independently and then merged.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    BarkBand, MappingConfig, OptimizerError, OptimizerResult, OptimizationSolution,
    StreamId, StreamMapping,
};
use crate::config::OptimizerConfig;
use crate::constraints::{Constraint, ConstraintSet};
use crate::objective::ObjectiveFn;

// ─────────────────────────────────────────────────────────────────────────────
// BarkBandDecomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Decompose the optimization into 24 independent Bark-band subproblems
/// (Theorem 1). Each band is solved independently, then solutions are merged.
#[derive(Debug, Clone)]
pub struct BarkBandDecomposition {
    /// Number of Bark bands (typically 24).
    pub num_bands: usize,
    /// Minimum number of streams to form a subproblem.
    pub min_streams_per_band: usize,
    /// Whether to handle cross-band interactions.
    pub handle_cross_band: bool,
}

impl Default for BarkBandDecomposition {
    fn default() -> Self {
        BarkBandDecomposition {
            num_bands: BarkBand::NUM_BANDS,
            min_streams_per_band: 1,
            handle_cross_band: true,
        }
    }
}

/// A subproblem for a single Bark band.
#[derive(Debug, Clone)]
pub struct BandSubproblem {
    pub band: BarkBand,
    pub stream_ids: Vec<StreamId>,
    pub constraints: ConstraintSet,
    pub frequency_range: (f64, f64),
}

/// Result of decomposed optimization.
#[derive(Debug, Clone)]
pub struct DecomposedResult {
    pub subproblem_solutions: Vec<SubproblemSolution>,
    pub merged_config: MappingConfig,
    pub total_objective: f64,
    pub solve_time_ms: f64,
}

/// Solution to a single subproblem.
#[derive(Debug, Clone)]
pub struct SubproblemSolution {
    pub label: String,
    pub config: MappingConfig,
    pub objective_value: f64,
    pub stream_ids: Vec<StreamId>,
}

impl BarkBandDecomposition {
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign streams to Bark bands based on their frequency parameters.
    pub fn assign_streams(
        &self,
        config: &MappingConfig,
    ) -> HashMap<BarkBand, Vec<StreamId>> {
        let mut assignment: HashMap<BarkBand, Vec<StreamId>> = HashMap::new();

        for (sid, mapping) in &config.stream_params {
            let band = mapping.bark_band;
            assignment.entry(band).or_default().push(*sid);
        }

        assignment
    }

    /// Decompose the problem into per-band subproblems.
    pub fn decompose(
        &self,
        config: &MappingConfig,
        constraints: &ConstraintSet,
    ) -> Vec<BandSubproblem> {
        let assignment = self.assign_streams(config);
        let mut subproblems = Vec::new();

        for band_idx in 0..self.num_bands {
            let band = BarkBand(band_idx as u8);
            let stream_ids = assignment.get(&band).cloned().unwrap_or_default();

            if stream_ids.len() < self.min_streams_per_band {
                continue;
            }

            // Build per-band constraints
            let mut band_constraints = ConstraintSet::new();

            // Add global constraints that apply to this band
            for (_, constraint) in constraints.iter() {
                match constraint {
                    Constraint::FrequencyRange { min_hz, max_hz } => {
                        let band_low = band.center_frequency() - band.bandwidth() / 2.0;
                        let band_high = band.center_frequency() + band.bandwidth() / 2.0;
                        band_constraints.add(Constraint::FrequencyRange {
                            min_hz: min_hz.max(band_low),
                            max_hz: max_hz.min(band_high),
                        });
                    }
                    Constraint::AmplitudeRange { .. } => {
                        band_constraints.add(constraint.clone());
                    }
                    Constraint::MaskingClearance { stream_id, .. }
                        if stream_ids.contains(stream_id) =>
                    {
                        band_constraints.add(constraint.clone());
                    }
                    Constraint::SegregationRequired { stream1_id, stream2_id, .. }
                        if stream_ids.contains(stream1_id)
                            && stream_ids.contains(stream2_id) =>
                    {
                        band_constraints.add(constraint.clone());
                    }
                    _ => {}
                }
            }

            let freq_range = (
                band.center_frequency() - band.bandwidth() / 2.0,
                band.center_frequency() + band.bandwidth() / 2.0,
            );

            subproblems.push(BandSubproblem {
                band,
                stream_ids,
                constraints: band_constraints,
                frequency_range: freq_range,
            });
        }

        subproblems
    }

    /// Solve each band subproblem independently using the provided solver.
    pub fn solve<F>(
        &self,
        config: &MappingConfig,
        constraints: &ConstraintSet,
        objective: &dyn ObjectiveFn,
        solver: F,
    ) -> OptimizerResult<DecomposedResult>
    where
        F: Fn(&[StreamId], &ConstraintSet, &dyn ObjectiveFn) -> OptimizerResult<SubproblemSolution>,
    {
        let start = std::time::Instant::now();
        let subproblems = self.decompose(config, constraints);
        let mut solutions = Vec::new();

        for sub in &subproblems {
            let sol = solver(&sub.stream_ids, &sub.constraints, objective)?;
            solutions.push(sol);
        }

        let merged = self.merge_solutions(&solutions);
        let total_obj = objective.evaluate(&merged).unwrap_or(0.0);

        Ok(DecomposedResult {
            subproblem_solutions: solutions,
            merged_config: merged,
            total_objective: total_obj,
            solve_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Merge subproblem solutions into a single configuration.
    pub fn merge_solutions(&self, solutions: &[SubproblemSolution]) -> MappingConfig {
        let mut merged = MappingConfig::new();

        for sol in solutions {
            for (sid, mapping) in &sol.config.stream_params {
                merged.stream_params.insert(*sid, mapping.clone());
            }
            for (key, value) in &sol.config.global_params {
                merged.global_params.insert(key.clone(), *value);
            }
        }

        merged
    }

    /// Handle cross-band interactions after initial decomposed solve.
    pub fn resolve_cross_band_interactions(
        &self,
        merged: &mut MappingConfig,
        constraints: &ConstraintSet,
    ) -> Vec<String> {
        let mut adjustments = Vec::new();

        if !self.handle_cross_band {
            return adjustments;
        }

        // Check for cross-band segregation violations
        let violated = constraints.find_violated(merged);
        for (name, eval) in &violated {
            if name.starts_with("segregation_") {
                adjustments.push(format!(
                    "Cross-band adjustment needed: {} (violation: {:.2})",
                    name, eval.violation_amount
                ));
                // Simple fix: spread streams apart
                let stream_ids: Vec<StreamId> = merged.stream_params.keys().cloned().collect();
                for (i, sid) in stream_ids.iter().enumerate() {
                    if let Some(mapping) = merged.stream_params.get_mut(sid) {
                        let offset = (i as f64 - stream_ids.len() as f64 / 2.0) * 50.0;
                        mapping.frequency_hz += offset;
                        mapping.frequency_hz = mapping.frequency_hz.clamp(20.0, 20000.0);
                    }
                }
            }
        }

        adjustments
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TemporalDecomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Decompose into time windows and optimize each independently.
#[derive(Debug, Clone)]
pub struct TemporalDecomposition {
    /// Window size in milliseconds.
    pub window_ms: f64,
    /// Overlap between windows (fraction, 0..1).
    pub overlap: f64,
    /// Crossfade duration for stitching in ms.
    pub crossfade_ms: f64,
}

impl Default for TemporalDecomposition {
    fn default() -> Self {
        TemporalDecomposition {
            window_ms: 100.0,
            overlap: 0.1,
            crossfade_ms: 10.0,
        }
    }
}

/// A temporal subproblem.
#[derive(Debug, Clone)]
pub struct TemporalSubproblem {
    pub window_index: usize,
    pub start_ms: f64,
    pub end_ms: f64,
    pub stream_ids: Vec<StreamId>,
    pub constraints: ConstraintSet,
}

impl TemporalDecomposition {
    pub fn new(window_ms: f64) -> Self {
        TemporalDecomposition {
            window_ms,
            ..Default::default()
        }
    }

    /// Partition the time axis into non-overlapping (or slightly overlapping) windows.
    pub fn partition(
        &self,
        total_duration_ms: f64,
        stream_ids: &[StreamId],
        constraints: &ConstraintSet,
    ) -> Vec<TemporalSubproblem> {
        let step = self.window_ms * (1.0 - self.overlap);
        let mut windows = Vec::new();
        let mut t = 0.0;
        let mut idx = 0;

        while t < total_duration_ms {
            let end = (t + self.window_ms).min(total_duration_ms);
            windows.push(TemporalSubproblem {
                window_index: idx,
                start_ms: t,
                end_ms: end,
                stream_ids: stream_ids.to_vec(),
                constraints: constraints.clone(),
            });
            t += step;
            idx += 1;
        }

        windows
    }

    /// Solve each temporal window independently.
    pub fn solve<F>(
        &self,
        total_duration_ms: f64,
        stream_ids: &[StreamId],
        constraints: &ConstraintSet,
        objective: &dyn ObjectiveFn,
        solver: F,
    ) -> OptimizerResult<DecomposedResult>
    where
        F: Fn(
            &TemporalSubproblem,
            &dyn ObjectiveFn,
        ) -> OptimizerResult<SubproblemSolution>,
    {
        let start = std::time::Instant::now();
        let windows = self.partition(total_duration_ms, stream_ids, constraints);
        let mut solutions = Vec::new();

        for window in &windows {
            let sol = solver(window, objective)?;
            solutions.push(sol);
        }

        let stitched = self.stitch_solutions(&solutions);
        let total_obj = objective.evaluate(&stitched).unwrap_or(0.0);

        Ok(DecomposedResult {
            subproblem_solutions: solutions,
            merged_config: stitched,
            total_objective: total_obj,
            solve_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Stitch temporal solutions at window boundaries.
    pub fn stitch_solutions(&self, solutions: &[SubproblemSolution]) -> MappingConfig {
        if solutions.is_empty() {
            return MappingConfig::new();
        }

        // For static optimization, just take the best window's config
        let best = solutions
            .iter()
            .max_by(|a, b| {
                a.objective_value
                    .partial_cmp(&b.objective_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        best.config.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamGroupDecomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Group interacting streams and optimize each group independently.
#[derive(Debug, Clone)]
pub struct StreamGroupDecomposition {
    /// Frequency proximity threshold for interaction edges (Hz).
    pub frequency_threshold: f64,
    /// Whether to consider constraint-based interactions.
    pub constraint_based: bool,
}

impl Default for StreamGroupDecomposition {
    fn default() -> Self {
        StreamGroupDecomposition {
            frequency_threshold: 500.0,
            constraint_based: true,
        }
    }
}

/// A group of interacting streams to optimize together.
#[derive(Debug, Clone)]
pub struct StreamGroup {
    pub group_id: usize,
    pub stream_ids: Vec<StreamId>,
    pub constraints: ConstraintSet,
}

impl StreamGroupDecomposition {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the interaction graph between streams.
    pub fn build_interaction_graph(
        &self,
        config: &MappingConfig,
        constraints: &ConstraintSet,
    ) -> HashMap<StreamId, HashSet<StreamId>> {
        let mut graph: HashMap<StreamId, HashSet<StreamId>> = HashMap::new();
        let streams: Vec<(&StreamId, &StreamMapping)> = config.stream_params.iter().collect();

        // Initialize graph
        for (sid, _) in &streams {
            graph.insert(**sid, HashSet::new());
        }

        // Add proximity-based edges
        for i in 0..streams.len() {
            for j in (i + 1)..streams.len() {
                let (sid1, m1) = streams[i];
                let (sid2, m2) = streams[j];

                let freq_dist = (m1.frequency_hz - m2.frequency_hz).abs();
                if freq_dist < self.frequency_threshold {
                    graph.entry(*sid1).or_default().insert(*sid2);
                    graph.entry(*sid2).or_default().insert(*sid1);
                }
            }
        }

        // Add constraint-based edges
        if self.constraint_based {
            for (_, constraint) in constraints.iter() {
                match constraint {
                    Constraint::SegregationRequired { stream1_id, stream2_id, .. } => {
                        graph.entry(*stream1_id).or_default().insert(*stream2_id);
                        graph.entry(*stream2_id).or_default().insert(*stream1_id);
                    }
                    Constraint::MaskingClearance { stream_id, .. } => {
                        // Masking constraints connect to all streams in same band
                        if let Some(mapping) = config.stream_params.get(stream_id) {
                            let band = mapping.bark_band;
                            for (other_sid, other_m) in &config.stream_params {
                                if other_sid != stream_id && other_m.bark_band == band {
                                    graph.entry(*stream_id).or_default().insert(*other_sid);
                                    graph.entry(*other_sid).or_default().insert(*stream_id);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        graph
    }

    /// Find connected components in the interaction graph.
    pub fn find_connected_components(
        &self,
        graph: &HashMap<StreamId, HashSet<StreamId>>,
    ) -> Vec<Vec<StreamId>> {
        let mut visited: HashSet<StreamId> = HashSet::new();
        let mut components = Vec::new();

        for &node in graph.keys() {
            if visited.contains(&node) {
                continue;
            }

            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(node);
            visited.insert(node);

            while let Some(current) = queue.pop_front() {
                component.push(current);
                if let Some(neighbors) = graph.get(&current) {
                    for &neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            component.sort();
            components.push(component);
        }

        components
    }

    /// Decompose into stream groups.
    pub fn decompose(
        &self,
        config: &MappingConfig,
        constraints: &ConstraintSet,
    ) -> Vec<StreamGroup> {
        let graph = self.build_interaction_graph(config, constraints);
        let components = self.find_connected_components(&graph);

        components
            .into_iter()
            .enumerate()
            .map(|(i, stream_ids)| {
                // Filter constraints relevant to this group
                let mut group_constraints = ConstraintSet::new();
                let stream_set: HashSet<StreamId> = stream_ids.iter().cloned().collect();

                for (_, constraint) in constraints.iter() {
                    let relevant = match constraint {
                        Constraint::MaskingClearance { stream_id, .. } => {
                            stream_set.contains(stream_id)
                        }
                        Constraint::SegregationRequired { stream1_id, stream2_id, .. } => {
                            stream_set.contains(stream1_id) || stream_set.contains(stream2_id)
                        }
                        Constraint::FrequencyRange { .. }
                        | Constraint::AmplitudeRange { .. }
                        | Constraint::CognitiveLoadBudget { .. }
                        | Constraint::LatencyBound { .. } => true,
                        _ => false,
                    };

                    if relevant {
                        group_constraints.add(constraint.clone());
                    }
                }

                StreamGroup {
                    group_id: i,
                    stream_ids,
                    constraints: group_constraints,
                }
            })
            .collect()
    }

    /// Solve each stream group independently.
    pub fn solve<F>(
        &self,
        config: &MappingConfig,
        constraints: &ConstraintSet,
        objective: &dyn ObjectiveFn,
        solver: F,
    ) -> OptimizerResult<DecomposedResult>
    where
        F: Fn(&StreamGroup, &dyn ObjectiveFn) -> OptimizerResult<SubproblemSolution>,
    {
        let start = std::time::Instant::now();
        let groups = self.decompose(config, constraints);
        let mut solutions = Vec::new();

        for group in &groups {
            let sol = solver(group, objective)?;
            solutions.push(sol);
        }

        let merged = self.merge_solutions(&solutions);
        let total_obj = objective.evaluate(&merged).unwrap_or(0.0);

        Ok(DecomposedResult {
            subproblem_solutions: solutions,
            merged_config: merged,
            total_objective: total_obj,
            solve_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    fn merge_solutions(&self, solutions: &[SubproblemSolution]) -> MappingConfig {
        let mut merged = MappingConfig::new();
        for sol in solutions {
            for (sid, mapping) in &sol.config.stream_params {
                merged.stream_params.insert(*sid, mapping.clone());
            }
        }
        merged
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SegregationPredicate;

    fn make_config(streams: Vec<(u32, f64, f64)>) -> MappingConfig {
        let mut config = MappingConfig::new();
        for (id, freq, amp) in streams {
            config.stream_params.insert(
                StreamId(id),
                StreamMapping::new(StreamId(id), freq, amp),
            );
        }
        config
    }

    fn basic_constraints() -> ConstraintSet {
        let mut cs = ConstraintSet::new();
        cs.add(Constraint::FrequencyRange { min_hz: 100.0, max_hz: 8000.0 });
        cs.add(Constraint::AmplitudeRange { min_db: 20.0, max_db: 90.0 });
        cs
    }

    #[test]
    fn test_bark_band_assign_streams() {
        let decomp = BarkBandDecomposition::new();
        let config = make_config(vec![
            (0, 440.0, 60.0),
            (1, 450.0, 55.0),
            (2, 4000.0, 65.0),
        ]);

        let assignment = decomp.assign_streams(&config);
        // Streams 0 and 1 should be in the same band (both ~440Hz)
        let mut found_pair = false;
        for (_, sids) in &assignment {
            if sids.contains(&StreamId(0)) && sids.contains(&StreamId(1)) {
                found_pair = true;
            }
        }
        assert!(found_pair, "Streams at similar freq should share a band");
    }

    #[test]
    fn test_bark_band_decompose() {
        let decomp = BarkBandDecomposition::new();
        let config = make_config(vec![
            (0, 440.0, 60.0),
            (1, 4000.0, 65.0),
        ]);
        let constraints = basic_constraints();

        let subproblems = decomp.decompose(&config, &constraints);
        assert!(subproblems.len() >= 1);
    }

    #[test]
    fn test_bark_band_merge() {
        let decomp = BarkBandDecomposition::new();
        let sol1 = SubproblemSolution {
            label: "band_0".into(),
            config: make_config(vec![(0, 440.0, 60.0)]),
            objective_value: 1.0,
            stream_ids: vec![StreamId(0)],
        };
        let sol2 = SubproblemSolution {
            label: "band_1".into(),
            config: make_config(vec![(1, 4000.0, 65.0)]),
            objective_value: 0.5,
            stream_ids: vec![StreamId(1)],
        };

        let merged = decomp.merge_solutions(&[sol1, sol2]);
        assert_eq!(merged.stream_params.len(), 2);
    }

    #[test]
    fn test_temporal_partition() {
        let decomp = TemporalDecomposition::new(100.0);
        let streams = vec![StreamId(0), StreamId(1)];
        let constraints = basic_constraints();

        let windows = decomp.partition(500.0, &streams, &constraints);
        assert!(windows.len() >= 5);
        assert_eq!(windows[0].start_ms, 0.0);
    }

    #[test]
    fn test_temporal_partition_overlap() {
        let decomp = TemporalDecomposition {
            window_ms: 100.0,
            overlap: 0.5,
            crossfade_ms: 10.0,
        };
        let streams = vec![StreamId(0)];
        let constraints = basic_constraints();

        let windows = decomp.partition(200.0, &streams, &constraints);
        // With 50% overlap, 100ms windows over 200ms => ~4 windows
        assert!(windows.len() >= 3);
    }

    #[test]
    fn test_temporal_stitch() {
        let decomp = TemporalDecomposition::new(100.0);
        let solutions = vec![
            SubproblemSolution {
                label: "w0".into(),
                config: make_config(vec![(0, 440.0, 60.0)]),
                objective_value: 0.5,
                stream_ids: vec![StreamId(0)],
            },
            SubproblemSolution {
                label: "w1".into(),
                config: make_config(vec![(0, 880.0, 65.0)]),
                objective_value: 0.8,
                stream_ids: vec![StreamId(0)],
            },
        ];

        let stitched = decomp.stitch_solutions(&solutions);
        assert!(!stitched.stream_params.is_empty());
    }

    #[test]
    fn test_stream_group_interaction_graph() {
        let decomp = StreamGroupDecomposition {
            frequency_threshold: 200.0,
            constraint_based: false,
        };
        let config = make_config(vec![
            (0, 440.0, 60.0),
            (1, 500.0, 55.0), // Close to stream 0
            (2, 4000.0, 65.0), // Far from 0 and 1
        ]);
        let constraints = basic_constraints();

        let graph = decomp.build_interaction_graph(&config, &constraints);
        // Streams 0 and 1 should be connected
        assert!(graph.get(&StreamId(0)).unwrap().contains(&StreamId(1)));
        // Stream 2 should be isolated from 0
        assert!(!graph.get(&StreamId(0)).unwrap().contains(&StreamId(2)));
    }

    #[test]
    fn test_stream_group_connected_components() {
        let decomp = StreamGroupDecomposition {
            frequency_threshold: 200.0,
            constraint_based: false,
        };
        let config = make_config(vec![
            (0, 440.0, 60.0),
            (1, 500.0, 55.0),
            (2, 4000.0, 65.0),
        ]);
        let constraints = basic_constraints();

        let graph = decomp.build_interaction_graph(&config, &constraints);
        let components = decomp.find_connected_components(&graph);

        // Should be 2 components: {0,1} and {2}
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_stream_group_decompose() {
        let decomp = StreamGroupDecomposition::new();
        let config = make_config(vec![
            (0, 440.0, 60.0),
            (1, 4000.0, 65.0),
        ]);
        let constraints = basic_constraints();

        let groups = decomp.decompose(&config, &constraints);
        assert!(!groups.is_empty());
    }

    #[test]
    fn test_stream_group_with_constraint_edges() {
        let decomp = StreamGroupDecomposition {
            frequency_threshold: 100.0, // Won't connect by frequency
            constraint_based: true,
        };
        let config = make_config(vec![
            (0, 440.0, 60.0),
            (1, 4000.0, 65.0),
        ]);
        let mut constraints = basic_constraints();
        constraints.add(Constraint::SegregationRequired {
            stream1_id: StreamId(0),
            stream2_id: StreamId(1),
            predicates: vec![SegregationPredicate::DifferentBarkBands],
        });

        let graph = decomp.build_interaction_graph(&config, &constraints);
        // Should be connected via constraint
        assert!(graph.get(&StreamId(0)).unwrap().contains(&StreamId(1)));

        let components = decomp.find_connected_components(&graph);
        assert_eq!(components.len(), 1); // All in one group
    }

    #[test]
    fn test_cross_band_interactions() {
        let decomp = BarkBandDecomposition::new();
        let mut config = make_config(vec![
            (0, 440.0, 60.0),
            (1, 442.0, 55.0),
        ]);
        let mut constraints = basic_constraints();
        constraints.add(Constraint::SegregationRequired {
            stream1_id: StreamId(0),
            stream2_id: StreamId(1),
            predicates: vec![SegregationPredicate::MinFrequencySeparation(100.0)],
        });

        let adjustments = decomp.resolve_cross_band_interactions(&mut config, &constraints);
        // Should detect a violation
        assert!(!adjustments.is_empty());
    }

    #[test]
    fn test_decomposed_result_structure() {
        let result = DecomposedResult {
            subproblem_solutions: vec![
                SubproblemSolution {
                    label: "test".into(),
                    config: MappingConfig::new(),
                    objective_value: 1.0,
                    stream_ids: vec![StreamId(0)],
                },
            ],
            merged_config: MappingConfig::new(),
            total_objective: 1.0,
            solve_time_ms: 5.0,
        };
        assert_eq!(result.subproblem_solutions.len(), 1);
    }
}
