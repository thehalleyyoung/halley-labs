//! Solver statistics: node count, cut count by type, gap closure, time breakdown,
//! branching statistics, primal/dual bound history, and iteration log.

use crate::{CutType, NodeId, INFINITY_BOUND};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

/// A single entry in the iteration log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationLogEntry {
    pub node_id: NodeId,
    pub depth: u32,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub gap: f64,
    pub open_nodes: u64,
    pub cuts_added: u32,
    pub elapsed_secs: f64,
    pub event: IterationEvent,
}

/// Categorisation of a log entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IterationEvent {
    NodeProcessed,
    NewIncumbent,
    CutRound,
    HeuristicSolution,
    Fathomed,
    Branched,
    Pruned,
}

impl fmt::Display for IterationEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IterationEvent::NodeProcessed => write!(f, "NodeProc"),
            IterationEvent::NewIncumbent => write!(f, "NewInc"),
            IterationEvent::CutRound => write!(f, "CutRnd"),
            IterationEvent::HeuristicSolution => write!(f, "Heur"),
            IterationEvent::Fathomed => write!(f, "Fathom"),
            IterationEvent::Branched => write!(f, "Branch"),
            IterationEvent::Pruned => write!(f, "Prune"),
        }
    }
}

/// Time breakdown across solver phases.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimeBreakdown {
    pub total_secs: f64,
    pub lp_solve_secs: f64,
    pub cut_separation_secs: f64,
    pub branching_secs: f64,
    pub heuristic_secs: f64,
    pub preprocessing_secs: f64,
    pub node_selection_secs: f64,
    pub other_secs: f64,
}

impl TimeBreakdown {
    pub fn finalize(&mut self) {
        self.other_secs = (self.total_secs
            - self.lp_solve_secs
            - self.cut_separation_secs
            - self.branching_secs
            - self.heuristic_secs
            - self.preprocessing_secs
            - self.node_selection_secs)
            .max(0.0);
    }

    pub fn fraction_lp(&self) -> f64 {
        if self.total_secs > 0.0 {
            self.lp_solve_secs / self.total_secs
        } else {
            0.0
        }
    }

    pub fn fraction_cuts(&self) -> f64 {
        if self.total_secs > 0.0 {
            self.cut_separation_secs / self.total_secs
        } else {
            0.0
        }
    }
}

/// Branching statistics: pseudocost tracking per variable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BranchingStats {
    pub total_branches: u64,
    pub strong_branching_calls: u64,
    pub pseudocost_up: HashMap<usize, f64>,
    pub pseudocost_down: HashMap<usize, f64>,
    pub branch_count_up: HashMap<usize, u64>,
    pub branch_count_down: HashMap<usize, u64>,
    pub variables_branched_on: HashMap<usize, u64>,
}

impl BranchingStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the outcome of branching up on a variable.
    pub fn record_up(&mut self, var: usize, delta_obj: f64, delta_frac: f64) {
        let count = self.branch_count_up.entry(var).or_insert(0);
        let pc = self.pseudocost_up.entry(var).or_insert(0.0);
        let safe_delta = if delta_frac.abs() > 1e-12 {
            delta_obj / delta_frac
        } else {
            delta_obj
        };
        *pc = (*pc * (*count as f64) + safe_delta) / (*count as f64 + 1.0);
        *count += 1;
        *self.variables_branched_on.entry(var).or_insert(0) += 1;
        self.total_branches += 1;
    }

    /// Record the outcome of branching down on a variable.
    pub fn record_down(&mut self, var: usize, delta_obj: f64, delta_frac: f64) {
        let count = self.branch_count_down.entry(var).or_insert(0);
        let pc = self.pseudocost_down.entry(var).or_insert(0.0);
        let safe_delta = if delta_frac.abs() > 1e-12 {
            delta_obj / delta_frac
        } else {
            delta_obj
        };
        *pc = (*pc * (*count as f64) + safe_delta) / (*count as f64 + 1.0);
        *count += 1;
        *self.variables_branched_on.entry(var).or_insert(0) += 1;
        self.total_branches += 1;
    }

    /// Get the pseudocost score for a variable (product scoring).
    pub fn pseudocost_score(&self, var: usize, frac: f64) -> f64 {
        let up = self.pseudocost_up.get(&var).copied().unwrap_or(1.0);
        let down = self.pseudocost_down.get(&var).copied().unwrap_or(1.0);
        let score_up = up * (1.0 - frac);
        let score_down = down * frac;
        (1.0 - 1e-6) * score_up.min(score_down) + 1e-6 * score_up.max(score_down)
    }

    /// Check whether we have enough observations for reliable pseudocosts.
    pub fn is_reliable(&self, var: usize, threshold: u64) -> bool {
        let up_count = self.branch_count_up.get(&var).copied().unwrap_or(0);
        let down_count = self.branch_count_down.get(&var).copied().unwrap_or(0);
        up_count >= threshold && down_count >= threshold
    }
}

/// Record of primal and dual bounds over time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BoundHistory {
    pub entries: Vec<BoundHistoryEntry>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoundHistoryEntry {
    pub elapsed_secs: f64,
    pub node_count: u64,
    pub primal_bound: f64,
    pub dual_bound: f64,
    pub gap: f64,
}

impl BoundHistory {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn record(&mut self, elapsed: f64, nodes: u64, primal: f64, dual: f64) {
        let gap = if primal.abs() > 1e-10 {
            ((primal - dual) / primal.abs()).abs()
        } else if dual.abs() > 1e-10 {
            f64::INFINITY
        } else {
            0.0
        };
        self.entries.push(BoundHistoryEntry {
            elapsed_secs: elapsed,
            node_count: nodes,
            primal_bound: primal,
            dual_bound: dual,
            gap,
        });
    }

    pub fn gap_closed_at(&self, fraction: f64) -> Option<f64> {
        if self.entries.is_empty() {
            return None;
        }
        let initial_gap = self.entries[0].gap;
        if initial_gap <= 0.0 || initial_gap.is_infinite() {
            return None;
        }
        let target = initial_gap * (1.0 - fraction);
        self.entries
            .iter()
            .find(|e| e.gap <= target)
            .map(|e| e.elapsed_secs)
    }

    pub fn final_gap(&self) -> f64 {
        self.entries.last().map_or(f64::INFINITY, |e| e.gap)
    }
}

/// Comprehensive solver statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverStatistics {
    pub nodes_processed: u64,
    pub nodes_created: u64,
    pub nodes_fathomed: u64,
    pub nodes_pruned: u64,
    pub max_depth: u32,
    pub lp_solves: u64,
    pub lp_iterations_total: u64,
    pub cuts_generated: IndexMap<String, u64>,
    pub cuts_applied: IndexMap<String, u64>,
    pub cuts_active_at_end: u64,
    pub global_cuts: u64,
    pub local_cuts: u64,
    pub primal_bound: f64,
    pub dual_bound: f64,
    pub final_gap: f64,
    pub incumbent_updates: u32,
    pub heuristic_solutions: u32,
    pub heuristic_improvements: u32,
    pub time_breakdown: TimeBreakdown,
    pub branching_stats: BranchingStats,
    pub bound_history: BoundHistory,
    pub iteration_log: Vec<IterationLogEntry>,
}

impl SolverStatistics {
    pub fn new() -> Self {
        Self {
            nodes_processed: 0,
            nodes_created: 0,
            nodes_fathomed: 0,
            nodes_pruned: 0,
            max_depth: 0,
            lp_solves: 0,
            lp_iterations_total: 0,
            cuts_generated: IndexMap::new(),
            cuts_applied: IndexMap::new(),
            cuts_active_at_end: 0,
            global_cuts: 0,
            local_cuts: 0,
            primal_bound: INFINITY_BOUND,
            dual_bound: -INFINITY_BOUND,
            final_gap: f64::INFINITY,
            incumbent_updates: 0,
            heuristic_solutions: 0,
            heuristic_improvements: 0,
            time_breakdown: TimeBreakdown::default(),
            branching_stats: BranchingStats::new(),
            bound_history: BoundHistory::new(),
            iteration_log: Vec::new(),
        }
    }

    pub fn record_node(&mut self, _node_id: NodeId, depth: u32) {
        self.nodes_processed += 1;
        if depth > self.max_depth {
            self.max_depth = depth;
        }
    }

    pub fn record_lp_solve(&mut self, iterations: u64) {
        self.lp_solves += 1;
        self.lp_iterations_total += iterations;
    }

    pub fn record_cut(&mut self, cut_type: CutType, applied: bool) {
        let key = format!("{}", cut_type);
        *self.cuts_generated.entry(key.clone()).or_insert(0) += 1;
        if applied {
            *self.cuts_applied.entry(key).or_insert(0) += 1;
        }
    }

    pub fn record_global_cut(&mut self) {
        self.global_cuts += 1;
    }
    pub fn record_local_cut(&mut self) {
        self.local_cuts += 1;
    }
    pub fn record_fathom(&mut self) {
        self.nodes_fathomed += 1;
    }
    pub fn record_prune(&mut self) {
        self.nodes_pruned += 1;
    }

    pub fn record_incumbent_update(&mut self, obj: f64) {
        self.primal_bound = obj;
        self.incumbent_updates += 1;
    }

    pub fn record_heuristic_solution(&mut self, improved: bool) {
        self.heuristic_solutions += 1;
        if improved {
            self.heuristic_improvements += 1;
        }
    }

    pub fn update_dual_bound(&mut self, bound: f64) {
        if bound > self.dual_bound {
            self.dual_bound = bound;
        }
    }

    pub fn record_bounds(&mut self, elapsed: f64) {
        self.bound_history.record(
            elapsed,
            self.nodes_processed,
            self.primal_bound,
            self.dual_bound,
        );
        self.final_gap = crate::compute_gap(self.primal_bound, self.dual_bound);
    }

    pub fn log_iteration(
        &mut self,
        node_id: NodeId,
        depth: u32,
        open_nodes: u64,
        cuts_added: u32,
        elapsed: f64,
        event: IterationEvent,
    ) {
        self.iteration_log.push(IterationLogEntry {
            node_id,
            depth,
            lower_bound: self.dual_bound,
            upper_bound: self.primal_bound,
            gap: self.final_gap,
            open_nodes,
            cuts_added,
            elapsed_secs: elapsed,
            event,
        });
    }

    pub fn finalize(&mut self, total_elapsed: f64) {
        self.time_breakdown.total_secs = total_elapsed;
        self.time_breakdown.finalize();
        self.final_gap = crate::compute_gap(self.primal_bound, self.dual_bound);
    }

    pub fn total_cuts_generated(&self) -> u64 {
        self.cuts_generated.values().sum()
    }
    pub fn total_cuts_applied(&self) -> u64 {
        self.cuts_applied.values().sum()
    }

    pub fn avg_lp_iterations(&self) -> f64 {
        if self.lp_solves > 0 {
            self.lp_iterations_total as f64 / self.lp_solves as f64
        } else {
            0.0
        }
    }
}

impl Default for SolverStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SolverStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Solver Statistics ===")?;
        writeln!(f, "Nodes processed: {}", self.nodes_processed)?;
        writeln!(f, "Nodes created:   {}", self.nodes_created)?;
        writeln!(f, "Nodes fathomed:  {}", self.nodes_fathomed)?;
        writeln!(f, "Nodes pruned:    {}", self.nodes_pruned)?;
        writeln!(f, "Max depth:       {}", self.max_depth)?;
        writeln!(f, "LP solves:       {}", self.lp_solves)?;
        writeln!(
            f,
            "LP iterations:   {} (avg {:.1})",
            self.lp_iterations_total,
            self.avg_lp_iterations()
        )?;
        writeln!(f, "Cuts generated:  {}", self.total_cuts_generated())?;
        writeln!(f, "Cuts applied:    {}", self.total_cuts_applied())?;
        for (k, v) in &self.cuts_generated {
            let applied = self.cuts_applied.get(k).copied().unwrap_or(0);
            writeln!(f, "  {}: {} generated, {} applied", k, v, applied)?;
        }
        writeln!(f, "Primal bound:    {:.6}", self.primal_bound)?;
        writeln!(f, "Dual bound:      {:.6}", self.dual_bound)?;
        writeln!(f, "Gap:             {:.4}%", self.final_gap * 100.0)?;
        writeln!(f, "Incumbent upd:   {}", self.incumbent_updates)?;
        writeln!(f, "Heuristic sols:  {}", self.heuristic_solutions)?;
        writeln!(f, "Time total:      {:.2}s", self.time_breakdown.total_secs)?;
        writeln!(
            f,
            "  LP:            {:.2}s ({:.1}%)",
            self.time_breakdown.lp_solve_secs,
            self.time_breakdown.fraction_lp() * 100.0
        )?;
        writeln!(
            f,
            "  Cuts:          {:.2}s ({:.1}%)",
            self.time_breakdown.cut_separation_secs,
            self.time_breakdown.fraction_cuts() * 100.0
        )?;
        Ok(())
    }
}

/// Timer helper for accumulating phase durations.
#[derive(Debug, Clone)]
pub struct PhaseTimer {
    start: Option<Instant>,
    accumulated_secs: f64,
}

impl PhaseTimer {
    pub fn new() -> Self {
        Self {
            start: None,
            accumulated_secs: 0.0,
        }
    }
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }
    pub fn stop(&mut self) {
        if let Some(s) = self.start.take() {
            self.accumulated_secs += s.elapsed().as_secs_f64();
        }
    }
    pub fn elapsed(&self) -> f64 {
        self.accumulated_secs
    }
    pub fn is_running(&self) -> bool {
        self.start.is_some()
    }
}

impl Default for PhaseTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Collection of phase timers for the solver.
#[derive(Debug, Clone)]
pub struct TimerSet {
    pub lp: PhaseTimer,
    pub cuts: PhaseTimer,
    pub branching: PhaseTimer,
    pub heuristics: PhaseTimer,
    pub preprocessing: PhaseTimer,
    pub node_selection: PhaseTimer,
}

impl TimerSet {
    pub fn new() -> Self {
        Self {
            lp: PhaseTimer::new(),
            cuts: PhaseTimer::new(),
            branching: PhaseTimer::new(),
            heuristics: PhaseTimer::new(),
            preprocessing: PhaseTimer::new(),
            node_selection: PhaseTimer::new(),
        }
    }

    pub fn populate_breakdown(&self, breakdown: &mut TimeBreakdown) {
        breakdown.lp_solve_secs = self.lp.elapsed();
        breakdown.cut_separation_secs = self.cuts.elapsed();
        breakdown.branching_secs = self.branching.elapsed();
        breakdown.heuristic_secs = self.heuristics.elapsed();
        breakdown.preprocessing_secs = self.preprocessing.elapsed();
        breakdown.node_selection_secs = self.node_selection.elapsed();
    }
}

impl Default for TimerSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_new() {
        let stats = SolverStatistics::new();
        assert_eq!(stats.nodes_processed, 0);
        assert_eq!(stats.lp_solves, 0);
        assert!(stats.primal_bound > 1e10);
    }

    #[test]
    fn test_record_node() {
        let mut stats = SolverStatistics::new();
        stats.record_node(1, 5);
        stats.record_node(2, 3);
        assert_eq!(stats.nodes_processed, 2);
        assert_eq!(stats.max_depth, 5);
    }

    #[test]
    fn test_record_lp_solve() {
        let mut stats = SolverStatistics::new();
        stats.record_lp_solve(100);
        stats.record_lp_solve(200);
        assert_eq!(stats.lp_solves, 2);
        assert_eq!(stats.lp_iterations_total, 300);
        assert!((stats.avg_lp_iterations() - 150.0).abs() < 1e-10);
    }

    #[test]
    fn test_record_cuts() {
        let mut stats = SolverStatistics::new();
        stats.record_cut(CutType::Gomory, true);
        stats.record_cut(CutType::Gomory, false);
        stats.record_cut(CutType::BilevelIntersection, true);
        assert_eq!(stats.total_cuts_generated(), 3);
        assert_eq!(stats.total_cuts_applied(), 2);
    }

    #[test]
    fn test_bound_history() {
        let mut bh = BoundHistory::new();
        bh.record(0.0, 0, 100.0, 0.0);
        bh.record(1.0, 10, 80.0, 20.0);
        bh.record(2.0, 50, 60.0, 55.0);
        assert_eq!(bh.entries.len(), 3);
        assert!(bh.final_gap() < 0.15);
    }

    #[test]
    fn test_gap_closed_at() {
        let mut bh = BoundHistory::new();
        bh.record(0.0, 0, 100.0, 0.0);
        bh.record(1.0, 10, 100.0, 50.0);
        bh.record(2.0, 20, 100.0, 90.0);
        let t = bh.gap_closed_at(0.5);
        assert!(t.is_some());
        assert!((t.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_branching_stats_pseudocost() {
        let mut bs = BranchingStats::new();
        bs.record_up(0, 2.0, 0.5);
        bs.record_down(0, 3.0, 0.5);
        assert!(bs.pseudocost_score(0, 0.5) > 0.0);
        assert!(!bs.is_reliable(0, 5));
        for _ in 0..10 {
            bs.record_up(0, 2.0, 0.5);
            bs.record_down(0, 3.0, 0.5);
        }
        assert!(bs.is_reliable(0, 5));
    }

    #[test]
    fn test_time_breakdown() {
        let mut tb = TimeBreakdown::default();
        tb.total_secs = 10.0;
        tb.lp_solve_secs = 5.0;
        tb.cut_separation_secs = 2.0;
        tb.finalize();
        assert!((tb.fraction_lp() - 0.5).abs() < 1e-10);
        assert!((tb.other_secs - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_statistics_display() {
        let stats = SolverStatistics::new();
        let display = format!("{}", stats);
        assert!(display.contains("Solver Statistics"));
    }

    #[test]
    fn test_phase_timer() {
        let mut t = PhaseTimer::new();
        assert!(!t.is_running());
        t.start();
        assert!(t.is_running());
        std::thread::sleep(std::time::Duration::from_millis(10));
        t.stop();
        assert!(!t.is_running());
        assert!(t.elapsed() > 0.0);
    }

    #[test]
    fn test_timer_set() {
        let ts = TimerSet::new();
        let mut breakdown = TimeBreakdown::default();
        ts.populate_breakdown(&mut breakdown);
        assert!((breakdown.lp_solve_secs - 0.0).abs() < 1e-10);
    }
}
