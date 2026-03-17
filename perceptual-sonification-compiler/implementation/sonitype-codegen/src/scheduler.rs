//! Execution scheduling for audio graph processing.
//!
//! Determines the order in which nodes are processed, assigns buffers,
//! and optimizes for cache locality and minimal buffer lifetimes.

use crate::{
    BufferKind, CgGraph, CodegenConfig, CodegenError, CodegenResult, NodeInfo, NodeKind,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Schedule types
// ---------------------------------------------------------------------------

/// A single step in the execution schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleStep {
    pub node_id: u64,
    pub node_name: String,
    pub kind: NodeKind,
    /// Index of the parallel group this step belongs to (0 = sequential only).
    pub parallel_group: usize,
    /// Buffer IDs this step reads from.
    pub input_buffers: Vec<usize>,
    /// Buffer ID this step writes to.
    pub output_buffer: Option<usize>,
    /// Estimated WCET cycles for this step.
    pub wcet_cycles: f64,
}

/// A complete execution schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    pub steps: Vec<ScheduleStep>,
    /// Total number of buffers required.
    pub buffer_count: usize,
    /// Mapping from edge (src, dst) to buffer ID.
    pub buffer_assignments: HashMap<(u64, u64), usize>,
    /// Parallel groups: each group contains steps that can execute concurrently.
    pub parallel_groups: Vec<Vec<usize>>,
    /// Estimated total WCET cycles (sum of sequential groups).
    pub estimated_total_cycles: f64,
    /// Estimated parallel WCET (critical path through groups).
    pub estimated_parallel_cycles: f64,
}

impl Schedule {
    /// Steps in topological execution order.
    pub fn execution_order(&self) -> &[ScheduleStep] {
        &self.steps
    }

    /// Number of distinct parallel groups.
    pub fn group_count(&self) -> usize {
        self.parallel_groups.len()
    }

    /// Whether this is a purely sequential schedule.
    pub fn is_sequential(&self) -> bool {
        self.parallel_groups.iter().all(|g| g.len() <= 1)
    }

    /// Lookup buffer assignment for an edge.
    pub fn buffer_for_edge(&self, src: u64, dst: u64) -> Option<usize> {
        self.buffer_assignments.get(&(src, dst)).copied()
    }
}

// ---------------------------------------------------------------------------
// Buffer lifetime tracking
// ---------------------------------------------------------------------------

/// Tracks the lifetime of a buffer: the range of schedule steps during which
/// it is "alive" (written but not yet fully consumed).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BufferLifetime {
    buffer_id: usize,
    first_write_step: usize,
    last_read_step: usize,
}

// ---------------------------------------------------------------------------
// ExecutionScheduler
// ---------------------------------------------------------------------------

/// Schedules audio graph nodes for processing.
#[derive(Debug, Clone)]
pub struct ExecutionScheduler {
    pub config: CodegenConfig,
}

impl ExecutionScheduler {
    pub fn new(config: &CodegenConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Generate a sequential schedule in topological order.
    pub fn schedule_sequential(&self, graph: &CgGraph) -> CodegenResult<Schedule> {
        if graph.nodes.is_empty() {
            return Ok(Schedule {
                steps: Vec::new(),
                buffer_count: 0,
                buffer_assignments: HashMap::new(),
                parallel_groups: Vec::new(),
                estimated_total_cycles: 0.0,
                estimated_parallel_cycles: 0.0,
            });
        }

        let mut buffer_assignments: HashMap<(u64, u64), usize> = HashMap::new();
        let mut next_buffer = 0usize;

        // Assign buffers to edges.
        for edge in &graph.edges {
            let key = (edge.source_node, edge.dest_node);
            if !buffer_assignments.contains_key(&key) {
                buffer_assignments.insert(key, next_buffer);
                next_buffer += 1;
            }
        }

        let mut steps = Vec::new();
        let mut parallel_groups = Vec::new();

        for (step_idx, &node_id) in graph.topological_order.iter().enumerate() {
            let node = graph.node(node_id).ok_or(CodegenError::NodeNotFound { node_id })?;

            let input_buffers: Vec<usize> = graph
                .incoming_edges(node_id)
                .iter()
                .filter_map(|e| buffer_assignments.get(&(e.source_node, e.dest_node)).copied())
                .collect();

            let output_buffer = graph
                .outgoing_edges(node_id)
                .first()
                .and_then(|e| buffer_assignments.get(&(e.source_node, e.dest_node)).copied());

            steps.push(ScheduleStep {
                node_id,
                node_name: node.name.clone(),
                kind: node.kind,
                parallel_group: step_idx,
                input_buffers,
                output_buffer,
                wcet_cycles: node.wcet_cycles,
            });

            parallel_groups.push(vec![step_idx]);
        }

        let total_cycles: f64 = steps.iter().map(|s| s.wcet_cycles).sum();

        Ok(Schedule {
            steps,
            buffer_count: next_buffer,
            buffer_assignments,
            parallel_groups,
            estimated_total_cycles: total_cycles,
            estimated_parallel_cycles: total_cycles,
        })
    }

    /// Generate a parallel schedule that groups independent nodes.
    pub fn schedule_parallel(&self, graph: &CgGraph) -> CodegenResult<Schedule> {
        if graph.nodes.is_empty() {
            return self.schedule_sequential(graph);
        }

        // Compute the "level" of each node: the length of the longest path from
        // any source to this node. Nodes at the same level are independent.
        let mut level: HashMap<u64, usize> = HashMap::new();
        for &nid in &graph.topological_order {
            let pred_max = graph
                .predecessors(nid)
                .iter()
                .filter_map(|p| level.get(p))
                .copied()
                .max()
                .unwrap_or(0);
            let my_level = if graph.predecessors(nid).is_empty() {
                0
            } else {
                pred_max + 1
            };
            level.insert(nid, my_level);
        }

        let max_level = level.values().copied().max().unwrap_or(0);

        // Group nodes by level.
        let mut level_groups: Vec<Vec<u64>> = vec![Vec::new(); max_level + 1];
        for (&nid, &lev) in &level {
            level_groups[lev].push(nid);
        }

        // Assign buffers to edges.
        let mut buffer_assignments: HashMap<(u64, u64), usize> = HashMap::new();
        let mut next_buffer = 0usize;
        for edge in &graph.edges {
            let key = (edge.source_node, edge.dest_node);
            if !buffer_assignments.contains_key(&key) {
                buffer_assignments.insert(key, next_buffer);
                next_buffer += 1;
            }
        }

        let mut steps = Vec::new();
        let mut parallel_groups = Vec::new();

        for (group_idx, group) in level_groups.iter().enumerate() {
            let mut group_step_indices = Vec::new();
            for &node_id in group {
                let node = graph.node(node_id).ok_or(CodegenError::NodeNotFound { node_id })?;

                let input_buffers: Vec<usize> = graph
                    .incoming_edges(node_id)
                    .iter()
                    .filter_map(|e| {
                        buffer_assignments.get(&(e.source_node, e.dest_node)).copied()
                    })
                    .collect();

                let output_buffer = graph
                    .outgoing_edges(node_id)
                    .first()
                    .and_then(|e| {
                        buffer_assignments.get(&(e.source_node, e.dest_node)).copied()
                    });

                let step_idx = steps.len();
                steps.push(ScheduleStep {
                    node_id,
                    node_name: node.name.clone(),
                    kind: node.kind,
                    parallel_group: group_idx,
                    input_buffers,
                    output_buffer,
                    wcet_cycles: node.wcet_cycles,
                });
                group_step_indices.push(step_idx);
            }
            if !group_step_indices.is_empty() {
                parallel_groups.push(group_step_indices);
            }
        }

        let total_cycles: f64 = steps.iter().map(|s| s.wcet_cycles).sum();

        // Parallel estimate: max of each group.
        let parallel_cycles: f64 = parallel_groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|&idx| steps[idx].wcet_cycles)
                    .fold(0.0_f64, f64::max)
            })
            .sum();

        Ok(Schedule {
            steps,
            buffer_count: next_buffer,
            buffer_assignments,
            parallel_groups,
            estimated_total_cycles: total_cycles,
            estimated_parallel_cycles: parallel_cycles,
        })
    }
}

// ---------------------------------------------------------------------------
// ScheduleOptimizer
// ---------------------------------------------------------------------------

/// Optimizes a schedule for buffer lifetime minimization and cache locality.
#[derive(Debug, Clone)]
pub struct ScheduleOptimizer {
    pub config: CodegenConfig,
}

impl ScheduleOptimizer {
    pub fn new(config: &CodegenConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Minimize the number of buffers through graph coloring of lifetimes.
    /// Buffers whose lifetimes don't overlap can share the same physical buffer.
    pub fn minimize_buffers(&self, schedule: &Schedule, graph: &CgGraph) -> Schedule {
        if schedule.buffer_count <= 1 {
            return schedule.clone();
        }

        // Compute buffer lifetimes.
        let lifetimes = self.compute_lifetimes(schedule);

        // Build an interference graph: two buffers interfere if their lifetimes overlap.
        let n = schedule.buffer_count;
        let mut interferes = vec![vec![false; n]; n];
        for i in 0..lifetimes.len() {
            for j in (i + 1)..lifetimes.len() {
                let a = &lifetimes[i];
                let b = &lifetimes[j];
                if a.first_write_step <= b.last_read_step
                    && b.first_write_step <= a.last_read_step
                {
                    interferes[a.buffer_id][b.buffer_id] = true;
                    interferes[b.buffer_id][a.buffer_id] = true;
                }
            }
        }

        // Greedy graph coloring.
        let mut color_of: Vec<Option<usize>> = vec![None; n];
        let mut max_color = 0;

        for buf_id in 0..n {
            let mut used_colors: HashSet<usize> = HashSet::new();
            for other in 0..n {
                if interferes[buf_id][other] {
                    if let Some(c) = color_of[other] {
                        used_colors.insert(c);
                    }
                }
            }
            let mut c = 0;
            while used_colors.contains(&c) {
                c += 1;
            }
            color_of[buf_id] = Some(c);
            if c > max_color {
                max_color = c;
            }
        }

        // Remap buffer assignments.
        let new_assignments: HashMap<(u64, u64), usize> = schedule
            .buffer_assignments
            .iter()
            .map(|(&key, &buf_id)| {
                let new_id = color_of[buf_id].unwrap_or(buf_id);
                (key, new_id)
            })
            .collect();

        // Remap step buffer references.
        let old_to_new: Vec<usize> = color_of
            .iter()
            .enumerate()
            .map(|(i, c)| c.unwrap_or(i))
            .collect();

        let new_steps: Vec<ScheduleStep> = schedule
            .steps
            .iter()
            .map(|step| {
                let new_inputs: Vec<usize> = step
                    .input_buffers
                    .iter()
                    .map(|&b| if b < old_to_new.len() { old_to_new[b] } else { b })
                    .collect();
                let new_output = step
                    .output_buffer
                    .map(|b| if b < old_to_new.len() { old_to_new[b] } else { b });
                ScheduleStep {
                    input_buffers: new_inputs,
                    output_buffer: new_output,
                    ..step.clone()
                }
            })
            .collect();

        Schedule {
            steps: new_steps,
            buffer_count: max_color + 1,
            buffer_assignments: new_assignments,
            parallel_groups: schedule.parallel_groups.clone(),
            estimated_total_cycles: schedule.estimated_total_cycles,
            estimated_parallel_cycles: schedule.estimated_parallel_cycles,
        }
    }

    /// Reorder steps within each parallel group to maximize cache locality
    /// (process producer then consumer back-to-back when possible).
    pub fn optimize_cache_locality(&self, schedule: &Schedule, graph: &CgGraph) -> Schedule {
        let mut new_steps = schedule.steps.clone();
        let mut new_groups = Vec::new();

        for group in &schedule.parallel_groups {
            if group.len() <= 1 {
                new_groups.push(group.clone());
                continue;
            }

            // Within a parallel group, sort by the buffer ID they write to,
            // so that nearby buffer writes are grouped together.
            let mut indices = group.clone();
            indices.sort_by(|&a, &b| {
                let buf_a = new_steps[a].output_buffer.unwrap_or(usize::MAX);
                let buf_b = new_steps[b].output_buffer.unwrap_or(usize::MAX);
                buf_a.cmp(&buf_b)
            });
            new_groups.push(indices);
        }

        Schedule {
            steps: new_steps,
            parallel_groups: new_groups,
            ..schedule.clone()
        }
    }

    /// Apply both buffer minimization and cache optimization.
    pub fn optimize(&self, schedule: &Schedule, graph: &CgGraph) -> Schedule {
        let minimized = self.minimize_buffers(schedule, graph);
        self.optimize_cache_locality(&minimized, graph)
    }

    fn compute_lifetimes(&self, schedule: &Schedule) -> Vec<BufferLifetime> {
        let mut first_write: HashMap<usize, usize> = HashMap::new();
        let mut last_read: HashMap<usize, usize> = HashMap::new();

        for (step_idx, step) in schedule.steps.iter().enumerate() {
            if let Some(buf) = step.output_buffer {
                first_write.entry(buf).or_insert(step_idx);
            }
            for &buf in &step.input_buffers {
                last_read
                    .entry(buf)
                    .and_modify(|v| *v = (*v).max(step_idx))
                    .or_insert(step_idx);
            }
        }

        let mut lifetimes = Vec::new();
        for buf_id in 0..schedule.buffer_count {
            let fw = first_write.get(&buf_id).copied().unwrap_or(0);
            let lr = last_read.get(&buf_id).copied().unwrap_or(fw);
            lifetimes.push(BufferLifetime {
                buffer_id: buf_id,
                first_write_step: fw,
                last_read_step: lr,
            });
        }
        lifetimes
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferKind, CgGraphBuilder, NodeKind};

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

    fn parallel_graph() -> CgGraph {
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let osc1 = b.add_node("osc1", NodeKind::Oscillator);
        let osc2 = b.add_node("osc2", NodeKind::Oscillator);
        let mix = b.add_node("mix", NodeKind::Mixer);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc1, mix, BufferKind::Audio);
        b.connect(osc2, mix, BufferKind::Audio);
        b.connect(mix, out, BufferKind::Audio);
        b.build()
    }

    fn diamond_graph() -> CgGraph {
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let src = b.add_node("src", NodeKind::DataInput);
        let left = b.add_node("left", NodeKind::Gain);
        let right = b.add_node("right", NodeKind::Pan);
        let mix = b.add_node("mix", NodeKind::Mixer);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(src, left, BufferKind::Audio);
        b.connect(src, right, BufferKind::Audio);
        b.connect(left, mix, BufferKind::Audio);
        b.connect(right, mix, BufferKind::Audio);
        b.connect(mix, out, BufferKind::Audio);
        b.build()
    }

    #[test]
    fn test_sequential_schedule_basic() {
        let cfg = test_config();
        let scheduler = ExecutionScheduler::new(&cfg);
        let graph = simple_chain();
        let sched = scheduler.schedule_sequential(&graph).unwrap();
        assert_eq!(sched.steps.len(), 3);
        assert!(sched.is_sequential());
    }

    #[test]
    fn test_sequential_schedule_buffer_count() {
        let cfg = test_config();
        let scheduler = ExecutionScheduler::new(&cfg);
        let graph = simple_chain();
        let sched = scheduler.schedule_sequential(&graph).unwrap();
        assert_eq!(sched.buffer_count, 2); // osc->filt, filt->out
    }

    #[test]
    fn test_parallel_schedule_groups() {
        let cfg = test_config();
        let scheduler = ExecutionScheduler::new(&cfg);
        let graph = parallel_graph();
        let sched = scheduler.schedule_parallel(&graph).unwrap();
        // osc1 and osc2 should be in the same group
        assert!(!sched.is_sequential());
        // At least one group has 2+ nodes
        assert!(sched.parallel_groups.iter().any(|g| g.len() >= 2));
    }

    #[test]
    fn test_parallel_schedule_diamond() {
        let cfg = test_config();
        let scheduler = ExecutionScheduler::new(&cfg);
        let graph = diamond_graph();
        let sched = scheduler.schedule_parallel(&graph).unwrap();
        // left and right should be in the same parallel group
        let left_group = sched.steps.iter().find(|s| s.node_name == "left").unwrap().parallel_group;
        let right_group = sched.steps.iter().find(|s| s.node_name == "right").unwrap().parallel_group;
        assert_eq!(left_group, right_group);
    }

    #[test]
    fn test_schedule_empty_graph() {
        let cfg = test_config();
        let scheduler = ExecutionScheduler::new(&cfg);
        let graph = CgGraphBuilder::new(48000.0, 256).build();
        let sched = scheduler.schedule_sequential(&graph).unwrap();
        assert_eq!(sched.steps.len(), 0);
        assert_eq!(sched.buffer_count, 0);
    }

    #[test]
    fn test_buffer_minimization() {
        let cfg = test_config();
        let scheduler = ExecutionScheduler::new(&cfg);
        let optimizer = ScheduleOptimizer::new(&cfg);
        let graph = diamond_graph();
        let sched = scheduler.schedule_sequential(&graph).unwrap();
        let optimized = optimizer.minimize_buffers(&sched, &graph);
        // With non-overlapping lifetimes, buffer count should be <= original
        assert!(optimized.buffer_count <= sched.buffer_count);
    }

    #[test]
    fn test_cache_locality_optimization() {
        let cfg = test_config();
        let scheduler = ExecutionScheduler::new(&cfg);
        let optimizer = ScheduleOptimizer::new(&cfg);
        let graph = parallel_graph();
        let sched = scheduler.schedule_parallel(&graph).unwrap();
        let optimized = optimizer.optimize_cache_locality(&sched, &graph);
        assert_eq!(optimized.steps.len(), sched.steps.len());
    }

    #[test]
    fn test_full_optimization_pipeline() {
        let cfg = test_config();
        let scheduler = ExecutionScheduler::new(&cfg);
        let optimizer = ScheduleOptimizer::new(&cfg);
        let graph = diamond_graph();
        let sched = scheduler.schedule_parallel(&graph).unwrap();
        let optimized = optimizer.optimize(&sched, &graph);
        assert!(optimized.buffer_count <= sched.buffer_count);
        assert_eq!(optimized.steps.len(), sched.steps.len());
    }

    #[test]
    fn test_buffer_for_edge() {
        let cfg = test_config();
        let scheduler = ExecutionScheduler::new(&cfg);
        let graph = simple_chain();
        let sched = scheduler.schedule_sequential(&graph).unwrap();
        let buf = sched.buffer_for_edge(0, 1);
        assert!(buf.is_some());
        assert!(sched.buffer_for_edge(99, 100).is_none());
    }

    #[test]
    fn test_parallel_cycle_estimate() {
        let cfg = test_config();
        let scheduler = ExecutionScheduler::new(&cfg);
        let graph = parallel_graph();
        let sched = scheduler.schedule_parallel(&graph).unwrap();
        // Parallel estimate should be <= total
        assert!(sched.estimated_parallel_cycles <= sched.estimated_total_cycles);
    }

    #[test]
    fn test_execution_order_preserves_all_nodes() {
        let cfg = test_config();
        let scheduler = ExecutionScheduler::new(&cfg);
        let graph = diamond_graph();
        let sched = scheduler.schedule_sequential(&graph).unwrap();
        let ids: HashSet<u64> = sched.steps.iter().map(|s| s.node_id).collect();
        for node in &graph.nodes {
            assert!(ids.contains(&node.id));
        }
    }
}
