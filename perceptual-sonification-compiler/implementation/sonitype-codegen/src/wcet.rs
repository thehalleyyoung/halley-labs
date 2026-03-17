//! WCET (Worst-Case Execution Time) analysis for audio graph scheduling.
//!
//! Implements Theorem 7 (Schedulability): verifies that the total WCET of a
//! graph's critical path fits within a single buffer period with sufficient
//! safety margin (50–100× headroom).

use crate::{
    Architecture, BufferKind, CgGraph, CodegenConfig, CodegenError, CodegenResult, NodeInfo,
    NodeKind,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Cost tables
// ---------------------------------------------------------------------------

/// Per-sample cycle cost entry for a single node kind on a specific architecture.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CycleCost {
    /// Base cost in cycles per sample.
    pub base_cycles: f64,
    /// Whether cost depends on a multiplier parameter (e.g. Mixer channel count).
    pub per_input_cycles: f64,
}

impl CycleCost {
    pub const fn fixed(base: f64) -> Self {
        Self {
            base_cycles: base,
            per_input_cycles: 0.0,
        }
    }

    pub const fn variable(base: f64, per_input: f64) -> Self {
        Self {
            base_cycles: base,
            per_input_cycles: per_input,
        }
    }

    /// Effective cost given an input-count multiplier.
    pub fn effective(&self, input_count: usize) -> f64 {
        self.base_cycles + self.per_input_cycles * input_count as f64
    }
}

/// Hardware-specific cost table mapping `NodeKind` → `CycleCost`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    pub architecture: Architecture,
    pub costs: HashMap<NodeKind, CycleCost>,
    /// Overhead per sample for global bookkeeping (context switch, buffer swap).
    pub per_sample_overhead: f64,
    /// Fixed per-block overhead (function call, cache warm-up).
    pub per_block_overhead: f64,
}

impl CostModel {
    /// Build the default cost model for a given architecture.
    pub fn for_architecture(arch: Architecture) -> Self {
        let scale = match arch {
            Architecture::X86_64 => 1.0,
            Architecture::Aarch64 => 1.1,
            Architecture::AppleSilicon => 0.9,
            Architecture::GenericArm => 1.3,
            Architecture::Wasm32 => 2.0,
        };

        let mut costs = HashMap::new();
        costs.insert(NodeKind::Oscillator, CycleCost::fixed(50.0 * scale));
        costs.insert(NodeKind::Filter, CycleCost::fixed(20.0 * scale));
        costs.insert(NodeKind::Envelope, CycleCost::fixed(10.0 * scale));
        costs.insert(NodeKind::Mixer, CycleCost::variable(0.0, 5.0 * scale));
        costs.insert(NodeKind::Gain, CycleCost::fixed(5.0 * scale));
        costs.insert(NodeKind::Pan, CycleCost::fixed(10.0 * scale));
        costs.insert(NodeKind::Delay, CycleCost::fixed(15.0 * scale));
        costs.insert(NodeKind::Modulator, CycleCost::fixed(45.0 * scale));
        costs.insert(NodeKind::Compressor, CycleCost::fixed(35.0 * scale));
        costs.insert(NodeKind::Limiter, CycleCost::fixed(25.0 * scale));
        costs.insert(NodeKind::DataInput, CycleCost::fixed(8.0 * scale));
        costs.insert(NodeKind::Output, CycleCost::fixed(5.0 * scale));
        costs.insert(NodeKind::Splitter, CycleCost::fixed(2.0 * scale));
        costs.insert(NodeKind::Merger, CycleCost::variable(2.0, 3.0 * scale));
        costs.insert(NodeKind::Constant, CycleCost::fixed(1.0 * scale));
        costs.insert(NodeKind::NoiseGenerator, CycleCost::fixed(30.0 * scale));
        costs.insert(NodeKind::PitchShifter, CycleCost::fixed(60.0 * scale));
        costs.insert(NodeKind::TimeStretch, CycleCost::fixed(80.0 * scale));

        Self {
            architecture: arch,
            costs,
            per_sample_overhead: 2.0 * scale,
            per_block_overhead: 200.0 * scale,
        }
    }

    /// Lookup the cycle cost for a node, given its kind and input count.
    pub fn node_cost(&self, kind: NodeKind, input_count: usize) -> f64 {
        self.costs
            .get(&kind)
            .map(|c| c.effective(input_count))
            .unwrap_or(10.0)
    }

    /// Cost of a node for an entire block.
    pub fn node_block_cost(&self, kind: NodeKind, input_count: usize, block_size: usize) -> f64 {
        self.node_cost(kind, input_count) * block_size as f64
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::for_architecture(Architecture::X86_64)
    }
}

// ---------------------------------------------------------------------------
// WCET per-node result
// ---------------------------------------------------------------------------

/// WCET estimate for one processing node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeWcet {
    pub node_id: u64,
    pub node_name: String,
    pub kind: NodeKind,
    /// Cycles per sample.
    pub per_sample_cycles: f64,
    /// Total cycles for one block.
    pub per_block_cycles: f64,
    /// Input count used for cost calculation.
    pub input_count: usize,
}

// ---------------------------------------------------------------------------
// WCET Budget
// ---------------------------------------------------------------------------

/// Per-node budget allocation with safety margin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WcetBudget {
    /// Budget in cycles for the entire block period.
    pub total_budget_cycles: f64,
    /// Budget consumed by the graph's critical path.
    pub critical_path_cycles: f64,
    /// Budget consumed by total graph.
    pub total_graph_cycles: f64,
    /// Safety margin (ratio of budget to critical path).
    pub safety_margin: f64,
    /// Utilization (0.0–1.0) = total_graph_cycles / total_budget_cycles.
    pub utilization: f64,
    /// Per-node budget entries.
    pub node_budgets: Vec<NodeBudgetEntry>,
    /// Whether the budget passes the safety margin requirement.
    pub passes: bool,
}

/// A single node's budget entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeBudgetEntry {
    pub node_id: u64,
    pub node_name: String,
    pub kind: NodeKind,
    pub allocated_cycles: f64,
    pub wcet_cycles: f64,
    /// Local headroom = allocated / wcet.
    pub headroom: f64,
}

// ---------------------------------------------------------------------------
// Budget violation
// ---------------------------------------------------------------------------

/// Report of a budget violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetViolation {
    pub kind: ViolationKind,
    pub message: String,
    pub severity: ViolationSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationKind {
    CriticalPathExceeded,
    TotalUtilizationExceeded,
    InsufficientSafetyMargin,
    SingleNodeOverbudget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Warning,
    Error,
    Fatal,
}

// ---------------------------------------------------------------------------
// WcetAnalyzer
// ---------------------------------------------------------------------------

/// Analyzes an audio graph to compute per-node and aggregate WCET estimates.
#[derive(Debug, Clone)]
pub struct WcetAnalyzer {
    pub cost_model: CostModel,
    pub config: CodegenConfig,
}

impl WcetAnalyzer {
    pub fn new(config: &CodegenConfig) -> Self {
        Self {
            cost_model: CostModel::for_architecture(config.architecture),
            config: config.clone(),
        }
    }

    pub fn with_cost_model(config: &CodegenConfig, cost_model: CostModel) -> Self {
        Self {
            cost_model,
            config: config.clone(),
        }
    }

    /// Compute per-node WCET for every node in the graph.
    pub fn analyze_nodes(&self, graph: &CgGraph) -> Vec<NodeWcet> {
        graph
            .nodes
            .iter()
            .map(|node| {
                let input_count = graph.incoming_edges(node.id).len();
                let per_sample = self.cost_model.node_cost(node.kind, input_count)
                    + self.cost_model.per_sample_overhead;
                let per_block = per_sample * graph.block_size as f64
                    + self.cost_model.per_block_overhead;
                NodeWcet {
                    node_id: node.id,
                    node_name: node.name.clone(),
                    kind: node.kind,
                    per_sample_cycles: per_sample,
                    per_block_cycles: per_block,
                    input_count,
                }
            })
            .collect()
    }

    /// Total WCET across all nodes (assumes sequential execution).
    pub fn total_wcet(&self, graph: &CgGraph) -> f64 {
        self.analyze_nodes(graph)
            .iter()
            .map(|n| n.per_block_cycles)
            .sum()
    }

    /// Critical-path WCET: the longest path from any source to any sink,
    /// weighted by per-block cycle cost.
    pub fn critical_path_wcet(&self, graph: &CgGraph) -> f64 {
        let node_wcets = self.analyze_nodes(graph);
        let cost_map: HashMap<u64, f64> = node_wcets
            .iter()
            .map(|w| (w.node_id, w.per_block_cycles))
            .collect();

        // Longest-path via dynamic programming on topological order.
        let mut dist: HashMap<u64, f64> = HashMap::new();
        for &nid in &graph.topological_order {
            let self_cost = cost_map.get(&nid).copied().unwrap_or(0.0);
            let pred_max = graph
                .predecessors(nid)
                .iter()
                .filter_map(|p| dist.get(p))
                .copied()
                .fold(0.0_f64, f64::max);
            dist.insert(nid, pred_max + self_cost);
        }
        dist.values().copied().fold(0.0_f64, f64::max)
    }

    /// Return the actual critical path (list of node IDs) from source to sink.
    pub fn critical_path_nodes(&self, graph: &CgGraph) -> Vec<u64> {
        let node_wcets = self.analyze_nodes(graph);
        let cost_map: HashMap<u64, f64> = node_wcets
            .iter()
            .map(|w| (w.node_id, w.per_block_cycles))
            .collect();

        let mut dist: HashMap<u64, f64> = HashMap::new();
        let mut pred_on_path: HashMap<u64, Option<u64>> = HashMap::new();

        for &nid in &graph.topological_order {
            let self_cost = cost_map.get(&nid).copied().unwrap_or(0.0);
            let mut best_pred: Option<u64> = None;
            let mut best_dist = 0.0_f64;
            for &p in &graph.predecessors(nid) {
                let d = dist.get(&p).copied().unwrap_or(0.0);
                if d > best_dist {
                    best_dist = d;
                    best_pred = Some(p);
                }
            }
            dist.insert(nid, best_dist + self_cost);
            pred_on_path.insert(nid, best_pred);
        }

        // Find the sink with the maximum distance.
        let sink = dist
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, _)| id);

        let mut path = Vec::new();
        let mut current = sink;
        while let Some(nid) = current {
            path.push(nid);
            current = pred_on_path.get(&nid).copied().flatten();
        }
        path.reverse();
        path
    }

    /// Compute the WCET budget, checking against the configuration's safety
    /// margin and maximum utilization.
    pub fn compute_budget(&self, graph: &CgGraph) -> WcetBudget {
        let total_budget = self.config.buffer_period_cycles();
        let node_wcets = self.analyze_nodes(graph);
        let critical = self.critical_path_wcet(graph);
        let total: f64 = node_wcets.iter().map(|w| w.per_block_cycles).sum();
        let utilization = total / total_budget;
        let safety_margin = if critical > 0.0 {
            total_budget / critical
        } else {
            f64::INFINITY
        };

        let node_budgets: Vec<NodeBudgetEntry> = node_wcets
            .iter()
            .map(|w| {
                let fraction = if total > 0.0 {
                    w.per_block_cycles / total
                } else {
                    1.0 / node_wcets.len() as f64
                };
                let allocated = fraction * total_budget * self.config.max_utilization;
                let headroom = if w.per_block_cycles > 0.0 {
                    allocated / w.per_block_cycles
                } else {
                    f64::INFINITY
                };
                NodeBudgetEntry {
                    node_id: w.node_id,
                    node_name: w.node_name.clone(),
                    kind: w.kind,
                    allocated_cycles: allocated,
                    wcet_cycles: w.per_block_cycles,
                    headroom,
                }
            })
            .collect();

        let passes =
            safety_margin >= self.config.target_safety_margin && utilization <= self.config.max_utilization;

        WcetBudget {
            total_budget_cycles: total_budget,
            critical_path_cycles: critical,
            total_graph_cycles: total,
            safety_margin,
            utilization,
            node_budgets,
            passes,
        }
    }

    /// Detect all budget violations.
    pub fn detect_violations(&self, graph: &CgGraph) -> Vec<BudgetViolation> {
        let budget = self.compute_budget(graph);
        let mut violations = Vec::new();

        if budget.critical_path_cycles > budget.total_budget_cycles {
            violations.push(BudgetViolation {
                kind: ViolationKind::CriticalPathExceeded,
                message: format!(
                    "Critical path {:.0} cycles exceeds budget {:.0} cycles",
                    budget.critical_path_cycles, budget.total_budget_cycles
                ),
                severity: ViolationSeverity::Fatal,
            });
        }

        if budget.utilization > self.config.max_utilization {
            violations.push(BudgetViolation {
                kind: ViolationKind::TotalUtilizationExceeded,
                message: format!(
                    "Utilization {:.1}% exceeds max {:.1}%",
                    budget.utilization * 100.0,
                    self.config.max_utilization * 100.0
                ),
                severity: ViolationSeverity::Error,
            });
        }

        if budget.safety_margin < self.config.target_safety_margin {
            violations.push(BudgetViolation {
                kind: ViolationKind::InsufficientSafetyMargin,
                message: format!(
                    "Safety margin {:.1}x below target {:.1}x",
                    budget.safety_margin, self.config.target_safety_margin
                ),
                severity: ViolationSeverity::Warning,
            });
        }

        for entry in &budget.node_budgets {
            if entry.wcet_cycles > entry.allocated_cycles {
                violations.push(BudgetViolation {
                    kind: ViolationKind::SingleNodeOverbudget,
                    message: format!(
                        "Node '{}' (id={}) WCET {:.0} exceeds allocated {:.0}",
                        entry.node_name, entry.node_id, entry.wcet_cycles, entry.allocated_cycles
                    ),
                    severity: ViolationSeverity::Error,
                });
            }
        }

        violations
    }

    /// Annotate the graph with WCET values on each node.
    pub fn annotate_graph(&self, graph: &mut CgGraph) {
        let wcets = self.analyze_nodes(graph);
        let cost_map: HashMap<u64, f64> = wcets
            .iter()
            .map(|w| (w.node_id, w.per_block_cycles))
            .collect();
        for node in &mut graph.nodes {
            if let Some(&c) = cost_map.get(&node.id) {
                node.wcet_cycles = c;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SchedulabilityChecker — Theorem 7
// ---------------------------------------------------------------------------

/// Verifies that the graph satisfies Theorem 7 (Schedulability):
/// the total WCET fits within the buffer period with the required margin.
#[derive(Debug, Clone)]
pub struct SchedulabilityChecker {
    pub config: CodegenConfig,
}

impl SchedulabilityChecker {
    pub fn new(config: &CodegenConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Check schedulability. Returns `Ok(budget)` or `Err(CodegenError)`.
    pub fn check(&self, graph: &CgGraph) -> CodegenResult<WcetBudget> {
        let analyzer = WcetAnalyzer::new(&self.config);
        let budget = analyzer.compute_budget(graph);

        if budget.critical_path_cycles > budget.total_budget_cycles {
            return Err(CodegenError::WcetExceeded {
                total_wcet_cycles: budget.critical_path_cycles,
                budget_cycles: budget.total_budget_cycles,
            });
        }

        if budget.utilization > self.config.max_utilization {
            return Err(CodegenError::SchedulabilityViolation {
                utilization: budget.utilization,
                max_utilization: self.config.max_utilization,
            });
        }

        Ok(budget)
    }

    /// Check with parallel execution on `num_cores`.
    pub fn check_parallel(&self, graph: &CgGraph) -> CodegenResult<WcetBudget> {
        let analyzer = WcetAnalyzer::new(&self.config);
        let budget = analyzer.compute_budget(graph);

        let effective_total = budget.total_graph_cycles / self.config.num_cores as f64;
        let effective_utilization = effective_total / budget.total_budget_cycles;

        // Critical path cannot be parallelized further.
        if budget.critical_path_cycles > budget.total_budget_cycles {
            return Err(CodegenError::WcetExceeded {
                total_wcet_cycles: budget.critical_path_cycles,
                budget_cycles: budget.total_budget_cycles,
            });
        }

        if effective_utilization > self.config.max_utilization {
            return Err(CodegenError::SchedulabilityViolation {
                utilization: effective_utilization,
                max_utilization: self.config.max_utilization,
            });
        }

        Ok(WcetBudget {
            utilization: effective_utilization,
            passes: budget.safety_margin >= self.config.target_safety_margin
                && effective_utilization <= self.config.max_utilization,
            ..budget
        })
    }

    /// Minimum number of cores required for the graph to be schedulable.
    pub fn min_cores_required(&self, graph: &CgGraph) -> usize {
        let analyzer = WcetAnalyzer::new(&self.config);
        let budget = analyzer.compute_budget(graph);
        let min_from_total =
            (budget.total_graph_cycles / (budget.total_budget_cycles * self.config.max_utilization))
                .ceil() as usize;
        // At least 1 core.
        min_from_total.max(1)
    }

    /// Suggest a block size that would satisfy the schedulability requirement
    /// with the current graph on a single core.
    pub fn suggest_block_size(&self, graph: &CgGraph) -> usize {
        let analyzer = WcetAnalyzer::new(&self.config);
        let node_wcets = analyzer.analyze_nodes(graph);
        let per_sample_total: f64 = node_wcets.iter().map(|w| w.per_sample_cycles).sum();
        let per_block_overhead: f64 = node_wcets.len() as f64 * self.config.target_safety_margin;

        if per_sample_total <= 0.0 {
            return self.config.block_size;
        }

        // We need: per_sample_total * B + overhead < B/SR * cpu_freq * max_util
        // per_sample_total * B < B * cpu_freq * max_util / SR
        // This is always satisfiable if per_sample_total < cpu_freq * max_util / SR
        let budget_per_sample = self.config.cpu_frequency_hz * self.config.max_utilization
            / self.config.sample_rate;
        if per_sample_total >= budget_per_sample {
            // Unsatisfiable at this sample rate
            return self.config.block_size;
        }

        // Find minimum block size that absorbs per-block overhead
        let min_block = (per_block_overhead / (budget_per_sample - per_sample_total)).ceil() as usize;
        // Round up to next power of 2 for alignment.
        let mut b = 1;
        while b < min_block {
            b <<= 1;
        }
        b.max(32).min(4096)
    }
}

// ---------------------------------------------------------------------------
// Summary report
// ---------------------------------------------------------------------------

/// A human-readable WCET analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WcetReport {
    pub architecture: Architecture,
    pub sample_rate: f64,
    pub block_size: usize,
    pub buffer_period_us: f64,
    pub buffer_period_cycles: f64,
    pub node_wcets: Vec<NodeWcet>,
    pub critical_path_wcet: f64,
    pub total_wcet: f64,
    pub utilization: f64,
    pub safety_margin: f64,
    pub critical_path_nodes: Vec<u64>,
    pub violations: Vec<BudgetViolation>,
    pub schedulable: bool,
}

impl WcetAnalyzer {
    /// Produce a full report for the graph.
    pub fn report(&self, graph: &CgGraph) -> WcetReport {
        let node_wcets = self.analyze_nodes(graph);
        let critical = self.critical_path_wcet(graph);
        let total = self.total_wcet(graph);
        let budget_cycles = self.config.buffer_period_cycles();
        let utilization = total / budget_cycles;
        let safety_margin = if critical > 0.0 {
            budget_cycles / critical
        } else {
            f64::INFINITY
        };
        let cp_nodes = self.critical_path_nodes(graph);
        let violations = self.detect_violations(graph);
        let schedulable = violations.iter().all(|v| v.severity < ViolationSeverity::Error);

        WcetReport {
            architecture: self.config.architecture,
            sample_rate: self.config.sample_rate,
            block_size: self.config.block_size,
            buffer_period_us: self.config.buffer_period_us(),
            buffer_period_cycles: budget_cycles,
            node_wcets,
            critical_path_wcet: critical,
            total_wcet: total,
            utilization,
            safety_margin,
            critical_path_nodes: cp_nodes,
            violations,
            schedulable,
        }
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
        CodegenConfig {
            sample_rate: 48000.0,
            block_size: 256,
            cpu_frequency_hz: 3_000_000_000.0,
            architecture: Architecture::X86_64,
            target_safety_margin: 50.0,
            max_utilization: 0.7,
            num_cores: 1,
            ..Default::default()
        }
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

    #[test]
    fn test_cost_model_default() {
        let cm = CostModel::default();
        assert_eq!(cm.architecture, Architecture::X86_64);
        assert!(cm.node_cost(NodeKind::Oscillator, 0) > 0.0);
    }

    #[test]
    fn test_cost_model_architectures() {
        let x86 = CostModel::for_architecture(Architecture::X86_64);
        let arm = CostModel::for_architecture(Architecture::GenericArm);
        // ARM costs should be scaled higher
        assert!(
            arm.node_cost(NodeKind::Filter, 0) > x86.node_cost(NodeKind::Filter, 0)
        );
    }

    #[test]
    fn test_cost_model_mixer_scaling() {
        let cm = CostModel::default();
        let cost2 = cm.node_cost(NodeKind::Mixer, 2);
        let cost4 = cm.node_cost(NodeKind::Mixer, 4);
        assert!(cost4 > cost2);
        // Each additional input adds per_input_cycles
        let delta = cost4 - cost2;
        assert!((delta - 2.0 * 5.0).abs() < 0.01);
    }

    #[test]
    fn test_analyze_nodes() {
        let cfg = test_config();
        let analyzer = WcetAnalyzer::new(&cfg);
        let graph = simple_chain();
        let wcets = analyzer.analyze_nodes(&graph);
        assert_eq!(wcets.len(), 3);
        // Oscillator should have highest per-sample cost in a simple chain
        let osc_wcet = wcets.iter().find(|w| w.kind == NodeKind::Oscillator).unwrap();
        let filt_wcet = wcets.iter().find(|w| w.kind == NodeKind::Filter).unwrap();
        assert!(osc_wcet.per_sample_cycles > filt_wcet.per_sample_cycles);
    }

    #[test]
    fn test_total_wcet() {
        let cfg = test_config();
        let analyzer = WcetAnalyzer::new(&cfg);
        let graph = simple_chain();
        let total = analyzer.total_wcet(&graph);
        assert!(total > 0.0);
    }

    #[test]
    fn test_critical_path_chain() {
        let cfg = test_config();
        let analyzer = WcetAnalyzer::new(&cfg);
        let graph = simple_chain();
        // In a linear chain, critical path = total
        let cp = analyzer.critical_path_wcet(&graph);
        let total = analyzer.total_wcet(&graph);
        assert!((cp - total).abs() < 1e-6);
    }

    #[test]
    fn test_critical_path_parallel() {
        let cfg = test_config();
        let analyzer = WcetAnalyzer::new(&cfg);
        let graph = parallel_graph();
        let cp = analyzer.critical_path_wcet(&graph);
        let total = analyzer.total_wcet(&graph);
        // With parallel branches, critical path < total
        assert!(cp < total);
    }

    #[test]
    fn test_critical_path_nodes() {
        let cfg = test_config();
        let analyzer = WcetAnalyzer::new(&cfg);
        let graph = simple_chain();
        let path = analyzer.critical_path_nodes(&graph);
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_budget_computation() {
        let cfg = test_config();
        let analyzer = WcetAnalyzer::new(&cfg);
        let graph = simple_chain();
        let budget = analyzer.compute_budget(&graph);
        assert!(budget.total_budget_cycles > 0.0);
        assert!(budget.utilization >= 0.0);
        assert!(budget.node_budgets.len() == 3);
        // With 3 GHz and 256/48000 period, budget should be enormous vs. a 3-node graph
        assert!(budget.passes);
    }

    #[test]
    fn test_schedulability_check_passes() {
        let cfg = test_config();
        let checker = SchedulabilityChecker::new(&cfg);
        let graph = simple_chain();
        let result = checker.check(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_schedulability_check_fails() {
        let cfg = CodegenConfig {
            cpu_frequency_hz: 100.0, // absurdly slow CPU
            max_utilization: 0.01,
            ..test_config()
        };
        let checker = SchedulabilityChecker::new(&cfg);
        let graph = simple_chain();
        let result = checker.check(&graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_violation_detection() {
        let cfg = CodegenConfig {
            cpu_frequency_hz: 100.0,
            target_safety_margin: 1000.0,
            max_utilization: 0.001,
            ..test_config()
        };
        let analyzer = WcetAnalyzer::new(&cfg);
        let graph = simple_chain();
        let violations = analyzer.detect_violations(&graph);
        assert!(!violations.is_empty());
    }

    #[test]
    fn test_annotate_graph() {
        let cfg = test_config();
        let analyzer = WcetAnalyzer::new(&cfg);
        let mut graph = simple_chain();
        assert_eq!(graph.nodes[0].wcet_cycles, 0.0);
        analyzer.annotate_graph(&mut graph);
        assert!(graph.nodes[0].wcet_cycles > 0.0);
    }

    #[test]
    fn test_min_cores_required() {
        let cfg = test_config();
        let checker = SchedulabilityChecker::new(&cfg);
        let graph = simple_chain();
        let cores = checker.min_cores_required(&graph);
        assert!(cores >= 1);
    }

    #[test]
    fn test_report_generation() {
        let cfg = test_config();
        let analyzer = WcetAnalyzer::new(&cfg);
        let graph = simple_chain();
        let report = analyzer.report(&graph);
        assert_eq!(report.block_size, 256);
        assert!(report.schedulable);
        assert_eq!(report.node_wcets.len(), 3);
    }

    #[test]
    fn test_cycle_cost_effective() {
        let c = CycleCost::variable(10.0, 5.0);
        assert!((c.effective(0) - 10.0).abs() < 1e-9);
        assert!((c.effective(3) - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_parallel_schedulability() {
        let cfg = CodegenConfig {
            num_cores: 4,
            ..test_config()
        };
        let checker = SchedulabilityChecker::new(&cfg);
        let graph = parallel_graph();
        let result = checker.check_parallel(&graph);
        assert!(result.is_ok());
    }
}
