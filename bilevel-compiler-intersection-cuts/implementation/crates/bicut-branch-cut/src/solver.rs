//! Main branch-and-cut solver: orchestrate tree search, cut generation, heuristics,
//! logging, time limits, solution reporting, gap tracking.

use crate::bounding::BoundManager;
use crate::branching::{create_branching_strategy, BranchingDecision, BranchingStrategy};
use crate::cut_callback::{CutCallbackConfig, CutCallbackManager};
use crate::heuristics::HeuristicManager;
use crate::node::{BbNode, BranchDirection, NodeStatus};
use crate::preprocess_node::{NodePreprocessor, PreprocessResult};
use crate::statistics::{IterationEvent, SolverStatistics, TimerSet};
use crate::tree::{BranchAndBoundTree, NodeSelectionStrategy};
use crate::{
    compute_gap, BilevelSolution, CompiledBilevelModel, LpSolverInterface, NodeSelectionType,
    SolutionStatus, SolverConfig, INFINITY_BOUND,
};
use bicut_types::*;
use log::{debug, info, warn};
use std::time::Instant;

/// Result of processing a single node.
#[derive(Debug)]
pub enum NodeProcessResult {
    Fathomed,
    Pruned,
    Integral(f64),
    Branched(Vec<BbNode>),
    CutAdded,
    Error(String),
}

/// Main branch-and-cut solver.
pub struct BranchAndCutSolver {
    pub config: SolverConfig,
    pub tree: BranchAndBoundTree,
    pub bound_manager: BoundManager,
    pub cut_manager: CutCallbackManager,
    pub heuristic_manager: HeuristicManager,
    pub preprocessor: NodePreprocessor,
    pub statistics: SolverStatistics,
    pub timers: TimerSet,
    start_time: Option<Instant>,
    branching_strategy: Box<dyn BranchingStrategy>,
}

impl BranchAndCutSolver {
    pub fn new(config: SolverConfig) -> Self {
        let strategy = create_branching_strategy(config.branching_strategy, &config);
        let node_sel = match config.node_selection {
            NodeSelectionType::BestFirst => crate::tree::NodeSelectionStrategy::BestFirst,
            NodeSelectionType::DepthFirst => crate::tree::NodeSelectionStrategy::DepthFirst,
            NodeSelectionType::Hybrid { switch_depth } => {
                crate::tree::NodeSelectionStrategy::Hybrid { switch_depth }
            }
        };
        let tree = BranchAndBoundTree::new(node_sel);
        let cut_config = CutCallbackConfig::from_solver_config(&config);
        Self {
            tree,
            bound_manager: BoundManager::new(),
            cut_manager: CutCallbackManager::new(cut_config),
            heuristic_manager: HeuristicManager::new(&config),
            preprocessor: NodePreprocessor::new(&config),
            statistics: SolverStatistics::new(),
            timers: TimerSet::new(),
            start_time: None,
            branching_strategy: strategy,
            config,
        }
    }

    /// Main solve entry point.
    pub fn solve(
        &mut self,
        model: &CompiledBilevelModel,
        lp_solver: &dyn LpSolverInterface,
    ) -> BilevelSolution {
        self.start_time = Some(Instant::now());
        info!(
            "Starting branch-and-cut solver with {} vars, {} constraints",
            model.num_vars, model.num_constraints
        );

        // Create and add root node
        let mut root = BbNode::root(model);
        root.id = self.tree.next_node_id();
        self.tree.add_node(root);
        self.statistics.nodes_created += 1;

        // Main loop
        while self.tree.has_nodes() && !self.check_limits() {
            // Select next node
            self.timers.node_selection.start();
            let mut node = match self.tree.select_next_node() {
                Some(n) => n,
                None => break,
            };
            self.timers.node_selection.stop();

            // Process the node
            let result = self.process_node(&mut node, model, lp_solver);

            self.statistics.record_node(node.id, node.depth);
            self.tree.record_processed_node(node.clone());

            match result {
                NodeProcessResult::Fathomed => {
                    self.statistics.record_fathom();
                    self.tree.fathom_node(node.clone(), NodeStatus::Fathomed);
                    self.statistics.log_iteration(
                        node.id,
                        node.depth,
                        self.tree.num_open_nodes() as u64,
                        0,
                        self.elapsed_secs(),
                        IterationEvent::Fathomed,
                    );
                }
                NodeProcessResult::Pruned => {
                    self.statistics.record_prune();
                    self.statistics.log_iteration(
                        node.id,
                        node.depth,
                        self.tree.num_open_nodes() as u64,
                        0,
                        self.elapsed_secs(),
                        IterationEvent::Pruned,
                    );
                }
                NodeProcessResult::Integral(obj) => {
                    let sol = BilevelSolution {
                        values: node.lp_solution.clone(),
                        objective: obj,
                        status: SolutionStatus::Feasible,
                        is_bilevel_feasible: node.check_bilevel_feasibility(model, 1e-6),
                        gap: 0.0,
                        node_count: 0,
                        time_secs: 0.0,
                    };
                    if self.tree.update_incumbent(obj, node.lp_solution.clone()) {
                        self.bound_manager
                            .update_upper_bound(obj, node.lp_solution.clone());
                        self.statistics.record_incumbent_update(obj);
                        info!(
                            "New incumbent: {:.6} at node {} depth {}",
                            obj, node.id, node.depth
                        );
                    }
                    self.statistics.log_iteration(
                        node.id,
                        node.depth,
                        self.tree.num_open_nodes() as u64,
                        0,
                        self.elapsed_secs(),
                        IterationEvent::NewIncumbent,
                    );
                }
                NodeProcessResult::Branched(children) => {
                    let num_children = children.len();
                    for child in children {
                        self.statistics.nodes_created += 1;
                        self.tree.add_node(child);
                    }
                    self.statistics.log_iteration(
                        node.id,
                        node.depth,
                        self.tree.num_open_nodes() as u64,
                        0,
                        self.elapsed_secs(),
                        IterationEvent::Branched,
                    );
                    debug!("Branched node {} into {} children", node.id, num_children);
                }
                NodeProcessResult::CutAdded => {
                    // Node goes back to pool for re-processing
                    self.tree.add_node(node);
                }
                NodeProcessResult::Error(msg) => {
                    warn!("Error processing node {}: {}", node.id, msg);
                    self.statistics.record_fathom();
                }
            }

            // Update global bounds
            let glb = self.tree.get_global_lower_bound();
            self.bound_manager.update_lower_bound(glb);
            self.statistics.update_dual_bound(glb);
            self.statistics.record_bounds(self.elapsed_secs());

            // Log progress periodically
            if self.statistics.nodes_processed % 100 == 0 {
                self.log_progress_summary();
            }

            // Check gap
            let gap = self.tree.get_gap();
            if gap <= self.config.gap_tolerance {
                info!("Optimal gap {:.6} reached", gap);
                break;
            }
        }

        self.finalize(model)
    }

    /// Process a single node: preprocess, solve LP, cuts, integrality check, branch.
    pub fn process_node(
        &mut self,
        node: &mut BbNode,
        model: &CompiledBilevelModel,
        lp_solver: &dyn LpSolverInterface,
    ) -> NodeProcessResult {
        // Preprocessing
        if self.config.enable_preprocessing {
            self.timers.preprocessing.start();
            let pp_result = self.preprocessor.preprocess(node, model, lp_solver);
            self.timers.preprocessing.stop();
            if pp_result.infeasible_detected {
                return NodeProcessResult::Fathomed;
            }
        }

        // Check bounds consistency
        if !node.bounds_consistent() {
            return NodeProcessResult::Fathomed;
        }

        // Solve LP relaxation
        self.timers.lp.start();
        let lp_status = node.solve_lp(model, lp_solver);
        self.timers.lp.stop();
        self.statistics.record_lp_solve(node.lp_iterations);

        match lp_status {
            LpStatus::Infeasible => return NodeProcessResult::Fathomed,
            LpStatus::Optimal => {}
            _ => return NodeProcessResult::Fathomed,
        }

        // Check pruning by bound
        if self.bound_manager.can_prune_node(node) {
            node.prune();
            return NodeProcessResult::Pruned;
        }

        // Cut separation
        if node.num_cut_rounds < self.config.cut_rounds_per_node as u32 {
            self.timers.cuts.start();
            let cut_results = self.cut_manager.run_cut_rounds(
                node,
                model,
                lp_solver,
                self.config.cut_rounds_per_node,
            );
            self.timers.cuts.stop();

            for cr in &cut_results {
                for cut in &cr.cuts_added {
                    self.statistics.record_cut(cut.cut_type, true);
                    if cut.is_global {
                        self.statistics.record_global_cut();
                    } else {
                        self.statistics.record_local_cut();
                    }
                }
            }
        }

        // Re-check pruning after cuts
        if self.bound_manager.can_prune_node(node) {
            node.prune();
            return NodeProcessResult::Pruned;
        }

        // Check integrality
        let is_integral = node.check_integrality(model, self.config.int_tolerance);
        if is_integral {
            return NodeProcessResult::Integral(node.lp_objective);
        }

        // Heuristics
        if self.config.enable_heuristics
            && self
                .heuristic_manager
                .should_run(self.statistics.nodes_processed)
        {
            self.timers.heuristics.start();
            let results = self.heuristic_manager.run_heuristics(
                node,
                model,
                self.bound_manager.incumbent_objective,
                lp_solver,
            );
            self.timers.heuristics.stop();

            for result in &results {
                if result.is_success() {
                    let improved = self.heuristic_manager.update_best(result);
                    self.statistics.record_heuristic_solution(improved);
                    if improved {
                        if let Some(ref sol) = result.solution {
                            let obj = result.objective;
                            let bilevel_sol = BilevelSolution {
                                values: sol.clone(),
                                objective: obj,
                                status: SolutionStatus::Feasible,
                                is_bilevel_feasible: result.is_bilevel_feasible,
                                gap: 0.0,
                                node_count: 0,
                                time_secs: 0.0,
                            };
                            if self.tree.update_incumbent(obj, sol.clone()) {
                                self.bound_manager.update_upper_bound(obj, sol.clone());
                                self.statistics.record_incumbent_update(obj);
                                info!(
                                    "Heuristic {} found solution {:.6}",
                                    result.heuristic_name, obj
                                );
                            }
                        }
                    }
                }
            }
        }

        // Branch
        self.timers.branching.start();
        let decision = self.branching_strategy.select_variable(
            node,
            model,
            lp_solver,
            &mut self.statistics.branching_stats,
        );
        self.timers.branching.stop();

        match decision {
            Some(dec) => {
                let val = dec.value;
                let var = dec.variable;
                let down_id = self.tree.next_node_id();
                let up_id = self.tree.next_node_id();

                let child_down =
                    node.create_child(down_id, var, BranchDirection::Down, val.floor());
                let child_up = node.create_child(up_id, var, BranchDirection::Up, val.ceil());

                let mut children = vec![child_down, child_up];

                // Propagate global cuts to children
                self.cut_manager
                    .propagate_cuts_to_children(node, &mut children);

                node.status = NodeStatus::Branched;
                NodeProcessResult::Branched(children)
            }
            None => {
                // No branching candidates found; node is integral (all integer vars fixed)
                NodeProcessResult::Integral(node.lp_objective)
            }
        }
    }

    /// Check whether time or node limits have been reached.
    pub fn check_limits(&self) -> bool {
        if self.statistics.nodes_processed >= self.config.node_limit {
            return true;
        }
        if self.elapsed_secs() >= self.config.time_limit_secs {
            return true;
        }
        false
    }

    /// Elapsed wall-clock time since solve() was called.
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.map_or(0.0, |t| t.elapsed().as_secs_f64())
    }

    fn log_progress_summary(&self) {
        let gap = self.tree.get_gap();
        info!(
            "Nodes: {} processed, {} open | Bounds: [{:.4}, {:.4}] | Gap: {:.4}% | Time: {:.1}s",
            self.statistics.nodes_processed,
            self.tree.num_open_nodes() as u64,
            self.bound_manager.global_lower_bound,
            self.bound_manager.incumbent_objective,
            gap * 100.0,
            self.elapsed_secs(),
        );
    }

    fn finalize(&mut self, _model: &CompiledBilevelModel) -> BilevelSolution {
        let elapsed = self.elapsed_secs();
        self.timers
            .populate_breakdown(&mut self.statistics.time_breakdown);
        self.statistics.finalize(elapsed);

        let status = self.get_solution_status();
        let gap = self.tree.get_gap();

        info!(
            "Solve complete: {} | Gap: {:.4}% | Nodes: {} | Time: {:.2}s",
            status,
            gap * 100.0,
            self.statistics.nodes_processed,
            elapsed
        );

        let (inc_obj, inc_sol) = self.tree.get_incumbent();
        match inc_sol {
            Some(sol) => BilevelSolution {
                values: sol.clone(),
                objective: inc_obj,
                status,
                is_bilevel_feasible: false,
                gap,
                node_count: self.statistics.nodes_processed,
                time_secs: elapsed,
            },
            None => BilevelSolution {
                values: Vec::new(),
                objective: INFINITY_BOUND,
                status,
                is_bilevel_feasible: false,
                gap: f64::INFINITY,
                node_count: self.statistics.nodes_processed,
                time_secs: elapsed,
            },
        }
    }

    pub fn get_statistics(&self) -> &SolverStatistics {
        &self.statistics
    }

    pub fn get_solution_status(&self) -> SolutionStatus {
        let gap = self.tree.get_gap();
        if self.tree.get_incumbent().1.is_none() {
            if !self.tree.has_nodes() {
                SolutionStatus::Infeasible
            } else {
                SolutionStatus::Unknown
            }
        } else if gap <= self.config.gap_tolerance {
            SolutionStatus::Optimal
        } else if self.elapsed_secs() >= self.config.time_limit_secs {
            SolutionStatus::TimeLimit
        } else if self.statistics.nodes_processed >= self.config.node_limit {
            SolutionStatus::NodeLimit
        } else {
            SolutionStatus::Feasible
        }
    }

    pub fn reset(&mut self) {
        self.tree.reset();
        self.bound_manager = BoundManager::new();
        self.statistics = SolverStatistics::new();
        self.timers = TimerSet::new();
        self.start_time = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BuiltinLpSolver;

    fn make_model() -> CompiledBilevelModel {
        let bilevel = BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0],
            lower_obj_c: vec![1.0],
            lower_a: SparseMatrix::new(1, 1),
            lower_b: vec![5.0],
            lower_linking_b: SparseMatrix::new(1, 1),
            upper_constraints_a: SparseMatrix::new(1, 2),
            upper_constraints_b: vec![10.0],
            num_upper_vars: 1,
            num_lower_vars: 1,
            num_lower_constraints: 1,
            num_upper_constraints: 1,
        };
        let mut m = CompiledBilevelModel::new(bilevel);
        m.integer_vars = vec![0];
        m
    }

    #[test]
    fn test_solver_new() {
        let cfg = SolverConfig::default();
        let solver = BranchAndCutSolver::new(cfg);
        assert_eq!(solver.statistics.nodes_processed, 0);
    }

    #[test]
    fn test_solver_solve_trivial() {
        let cfg = SolverConfig {
            node_limit: 50,
            ..SolverConfig::default()
        };
        let mut solver = BranchAndCutSolver::new(cfg);
        let model = make_model();
        let lp = BuiltinLpSolver::new();
        let result = solver.solve(&model, &lp);
        assert!(solver.statistics.nodes_processed > 0);
    }

    #[test]
    fn test_check_limits_node() {
        let cfg = SolverConfig {
            node_limit: 5,
            ..SolverConfig::default()
        };
        let mut solver = BranchAndCutSolver::new(cfg);
        solver.statistics.nodes_processed = 5;
        assert!(solver.check_limits());
    }

    #[test]
    fn test_check_limits_time() {
        let cfg = SolverConfig {
            time_limit_secs: 0.0,
            ..SolverConfig::default()
        };
        let mut solver = BranchAndCutSolver::new(cfg);
        solver.start_time = Some(Instant::now());
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(solver.check_limits());
    }

    #[test]
    fn test_solver_reset() {
        let cfg = SolverConfig::default();
        let mut solver = BranchAndCutSolver::new(cfg);
        solver.statistics.nodes_processed = 100;
        solver.reset();
        assert_eq!(solver.statistics.nodes_processed, 0);
    }

    #[test]
    fn test_solution_status_no_incumbent() {
        let cfg = SolverConfig::default();
        let solver = BranchAndCutSolver::new(cfg);
        let status = solver.get_solution_status();
        // No incumbent, no nodes => infeasible
        assert_eq!(status, SolutionStatus::Infeasible);
    }

    #[test]
    fn test_elapsed_secs() {
        let cfg = SolverConfig::default();
        let mut solver = BranchAndCutSolver::new(cfg);
        assert!((solver.elapsed_secs() - 0.0).abs() < 1e-6);
        solver.start_time = Some(Instant::now());
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(solver.elapsed_secs() > 0.0);
    }

    #[test]
    fn test_solver_with_time_limit() {
        let cfg = SolverConfig {
            node_limit: 1000,
            time_limit_secs: 0.5,
            ..SolverConfig::default()
        };
        let mut solver = BranchAndCutSolver::new(cfg);
        let model = make_model();
        let lp = BuiltinLpSolver::new();
        let result = solver.solve(&model, &lp);
        assert!(solver.elapsed_secs() < 5.0);
    }
}
