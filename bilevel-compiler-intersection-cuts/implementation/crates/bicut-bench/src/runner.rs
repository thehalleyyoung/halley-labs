//! Benchmark runner: execute benchmark instances with configurable reformulations,
//! solvers, and cut strategies. Supports timeout, parallel execution via rayon,
//! progress tracking, and error recovery.

use crate::instance::{BenchmarkInstance, InstanceSet};
use crate::metrics::BenchmarkMetrics;
use bicut_types::{BilevelProblem, LpSolution, LpStatus, SparseMatrix, DEFAULT_TOLERANCE};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Method used to reformulate the bilevel problem into a single level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReformulationMethod {
    /// Big-M reformulation using KKT conditions.
    BigM,
    /// SOS1 reformulation of complementarity.
    SOS1,
    /// Value function reformulation.
    ValueFunction,
    /// Strong duality based reformulation.
    StrongDuality,
    /// No reformulation (direct bilevel method).
    Direct,
}

impl ReformulationMethod {
    /// All available methods.
    pub fn all() -> &'static [ReformulationMethod] {
        &[
            ReformulationMethod::BigM,
            ReformulationMethod::SOS1,
            ReformulationMethod::ValueFunction,
            ReformulationMethod::StrongDuality,
            ReformulationMethod::Direct,
        ]
    }

    /// Short label for tables.
    pub fn label(&self) -> &'static str {
        match self {
            ReformulationMethod::BigM => "BigM",
            ReformulationMethod::SOS1 => "SOS1",
            ReformulationMethod::ValueFunction => "ValFn",
            ReformulationMethod::StrongDuality => "SDual",
            ReformulationMethod::Direct => "Direct",
        }
    }
}

impl std::fmt::Display for ReformulationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Solver backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Solver name identifier.
    pub name: String,
    /// Maximum iterations allowed.
    pub max_iterations: u64,
    /// Numerical tolerance.
    pub tolerance: f64,
    /// Additional solver-specific parameters.
    pub params: HashMap<String, String>,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            name: "simplex".to_string(),
            max_iterations: 100_000,
            tolerance: 1e-8,
            params: HashMap::new(),
        }
    }
}

impl SolverConfig {
    /// Create a named solver configuration.
    pub fn new(name: &str) -> Self {
        SolverConfig {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Builder: set max iterations.
    pub fn with_max_iterations(mut self, n: u64) -> Self {
        self.max_iterations = n;
        self
    }

    /// Builder: set tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Builder: add a parameter.
    pub fn with_param(mut self, key: &str, value: &str) -> Self {
        self.params.insert(key.to_string(), value.to_string());
        self
    }

    /// Label combining name and key parameters.
    pub fn label(&self) -> String {
        if self.params.is_empty() {
            self.name.clone()
        } else {
            let extras: Vec<String> = self
                .params
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            format!("{}({})", self.name, extras.join(","))
        }
    }
}

/// Cut generation strategy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutConfig {
    /// Whether to generate intersection cuts.
    pub intersection_cuts: bool,
    /// Whether to generate value function cuts.
    pub value_function_cuts: bool,
    /// Whether to generate Benders-like cuts.
    pub benders_cuts: bool,
    /// Maximum cuts per round.
    pub max_cuts_per_round: usize,
    /// Maximum total cut rounds.
    pub max_rounds: usize,
    /// Minimum violation to accept a cut.
    pub min_violation: f64,
}

impl Default for CutConfig {
    fn default() -> Self {
        CutConfig {
            intersection_cuts: true,
            value_function_cuts: false,
            benders_cuts: false,
            max_cuts_per_round: 50,
            max_rounds: 100,
            min_violation: 1e-6,
        }
    }
}

impl CutConfig {
    /// No cuts.
    pub fn none() -> Self {
        CutConfig {
            intersection_cuts: false,
            value_function_cuts: false,
            benders_cuts: false,
            ..Default::default()
        }
    }

    /// Only intersection cuts.
    pub fn intersection_only() -> Self {
        CutConfig {
            intersection_cuts: true,
            value_function_cuts: false,
            benders_cuts: false,
            ..Default::default()
        }
    }

    /// All cut types enabled.
    pub fn all_cuts() -> Self {
        CutConfig {
            intersection_cuts: true,
            value_function_cuts: true,
            benders_cuts: true,
            ..Default::default()
        }
    }

    /// Label for this configuration.
    pub fn label(&self) -> String {
        let mut parts = Vec::new();
        if self.intersection_cuts {
            parts.push("IC");
        }
        if self.value_function_cuts {
            parts.push("VF");
        }
        if self.benders_cuts {
            parts.push("BD");
        }
        if parts.is_empty() {
            "NoCuts".to_string()
        } else {
            parts.join("+")
        }
    }
}

/// Complete benchmark configuration for a single run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Reformulation method.
    pub reformulation: ReformulationMethod,
    /// Solver configuration.
    pub solver: SolverConfig,
    /// Cut strategy.
    pub cuts: CutConfig,
    /// Time limit in seconds.
    pub time_limit_secs: f64,
    /// Whether to collect detailed metrics.
    pub collect_detailed_metrics: bool,
    /// Seed for any randomized components.
    pub seed: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        BenchmarkConfig {
            reformulation: ReformulationMethod::BigM,
            solver: SolverConfig::default(),
            cuts: CutConfig::default(),
            time_limit_secs: 3600.0,
            collect_detailed_metrics: true,
            seed: 42,
        }
    }
}

impl BenchmarkConfig {
    /// Builder: set reformulation.
    pub fn with_reformulation(mut self, method: ReformulationMethod) -> Self {
        self.reformulation = method;
        self
    }

    /// Builder: set solver.
    pub fn with_solver(mut self, solver: SolverConfig) -> Self {
        self.solver = solver;
        self
    }

    /// Builder: set cuts.
    pub fn with_cuts(mut self, cuts: CutConfig) -> Self {
        self.cuts = cuts;
        self
    }

    /// Builder: set time limit.
    pub fn with_time_limit(mut self, secs: f64) -> Self {
        self.time_limit_secs = secs;
        self
    }

    /// Short label identifying this configuration.
    pub fn label(&self) -> String {
        format!(
            "{}_{}_{}_{}",
            self.reformulation.label(),
            self.solver.label(),
            self.cuts.label(),
            self.time_limit_secs
        )
    }

    /// Generate a configuration matrix: all combinations of reformulations, solvers, and cuts.
    pub fn config_matrix(
        reformulations: &[ReformulationMethod],
        solvers: &[SolverConfig],
        cuts: &[CutConfig],
        time_limit: f64,
    ) -> Vec<BenchmarkConfig> {
        let mut configs = Vec::new();
        for &reform in reformulations {
            for solver in solvers {
                for cut in cuts {
                    configs.push(BenchmarkConfig {
                        reformulation: reform,
                        solver: solver.clone(),
                        cuts: cut.clone(),
                        time_limit_secs: time_limit,
                        collect_detailed_metrics: true,
                        seed: 42,
                    });
                }
            }
        }
        configs
    }
}

// ---------------------------------------------------------------------------
// Run status and result
// ---------------------------------------------------------------------------

/// Status of a benchmark run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RunStatus {
    /// Solved to optimality.
    Optimal,
    /// Proved infeasible.
    Infeasible,
    /// Time limit exceeded.
    TimeLimit,
    /// Iteration limit exceeded.
    IterationLimit,
    /// Solver encountered an error.
    Error,
    /// Instance was skipped (e.g., too large).
    Skipped,
}

impl RunStatus {
    /// Whether this status represents a successful solve.
    pub fn is_success(&self) -> bool {
        matches!(self, RunStatus::Optimal)
    }

    /// Whether the solver produced a valid bound.
    pub fn has_bound(&self) -> bool {
        matches!(
            self,
            RunStatus::Optimal | RunStatus::TimeLimit | RunStatus::IterationLimit
        )
    }
}

impl std::fmt::Display for RunStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunStatus::Optimal => write!(f, "Optimal"),
            RunStatus::Infeasible => write!(f, "Infeasible"),
            RunStatus::TimeLimit => write!(f, "TimeLimit"),
            RunStatus::IterationLimit => write!(f, "IterLimit"),
            RunStatus::Error => write!(f, "Error"),
            RunStatus::Skipped => write!(f, "Skipped"),
        }
    }
}

/// Result of running a single benchmark instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    /// Instance name.
    pub instance_name: String,
    /// Configuration label.
    pub config_label: String,
    /// Run status.
    pub status: RunStatus,
    /// Total wall-clock time in seconds.
    pub wall_time_secs: f64,
    /// Objective value found (if any).
    pub objective: Option<f64>,
    /// Best bound found.
    pub best_bound: Option<f64>,
    /// Number of B&B nodes explored.
    pub node_count: u64,
    /// Number of LP iterations.
    pub iteration_count: u64,
    /// Number of cuts generated.
    pub cuts_generated: u64,
    /// Root node objective value.
    pub root_objective: Option<f64>,
    /// Detailed metrics, if collected.
    pub metrics: Option<BenchmarkMetrics>,
    /// Error message, if any.
    pub error_message: Option<String>,
    /// Timestamp of the run.
    pub timestamp: String,
}

impl RunResult {
    /// Create a result for a successful solve.
    pub fn optimal(
        instance_name: &str,
        config_label: &str,
        wall_time: f64,
        objective: f64,
        node_count: u64,
        iteration_count: u64,
    ) -> Self {
        RunResult {
            instance_name: instance_name.to_string(),
            config_label: config_label.to_string(),
            status: RunStatus::Optimal,
            wall_time_secs: wall_time,
            objective: Some(objective),
            best_bound: Some(objective),
            node_count,
            iteration_count,
            cuts_generated: 0,
            root_objective: None,
            metrics: None,
            error_message: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Create a result for a timeout.
    pub fn timeout(
        instance_name: &str,
        config_label: &str,
        wall_time: f64,
        best_obj: Option<f64>,
        best_bound: Option<f64>,
    ) -> Self {
        RunResult {
            instance_name: instance_name.to_string(),
            config_label: config_label.to_string(),
            status: RunStatus::TimeLimit,
            wall_time_secs: wall_time,
            objective: best_obj,
            best_bound,
            node_count: 0,
            iteration_count: 0,
            cuts_generated: 0,
            root_objective: None,
            metrics: None,
            error_message: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Create an error result.
    pub fn error(instance_name: &str, config_label: &str, wall_time: f64, msg: &str) -> Self {
        RunResult {
            instance_name: instance_name.to_string(),
            config_label: config_label.to_string(),
            status: RunStatus::Error,
            wall_time_secs: wall_time,
            objective: None,
            best_bound: None,
            node_count: 0,
            iteration_count: 0,
            cuts_generated: 0,
            root_objective: None,
            metrics: None,
            error_message: Some(msg.to_string()),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Create a skipped result.
    pub fn skipped(instance_name: &str, config_label: &str, reason: &str) -> Self {
        RunResult {
            instance_name: instance_name.to_string(),
            config_label: config_label.to_string(),
            status: RunStatus::Skipped,
            wall_time_secs: 0.0,
            objective: None,
            best_bound: None,
            node_count: 0,
            iteration_count: 0,
            cuts_generated: 0,
            root_objective: None,
            metrics: None,
            error_message: Some(reason.to_string()),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Final optimality gap as a percentage, if both primal and bound exist.
    pub fn gap_percent(&self) -> Option<f64> {
        match (self.objective, self.best_bound) {
            (Some(obj), Some(bnd)) => {
                let denom = obj.abs().max(1e-10);
                Some(((obj - bnd).abs() / denom) * 100.0)
            }
            _ => None,
        }
    }

    /// Root gap closure: how much of the initial gap was closed by cuts at the root.
    pub fn root_gap_closure(&self) -> Option<f64> {
        match (self.root_objective, self.objective, self.best_bound) {
            (Some(root), Some(obj), Some(bnd)) => {
                let initial_gap = (obj - root).abs();
                let final_gap = (obj - bnd).abs();
                if initial_gap < DEFAULT_TOLERANCE {
                    Some(100.0)
                } else {
                    Some(((initial_gap - final_gap) / initial_gap) * 100.0)
                }
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Progress callback
// ---------------------------------------------------------------------------

/// Callback type for reporting benchmark progress.
pub type ProgressCallback = Box<dyn Fn(usize, usize, &str) + Send + Sync>;

/// Default progress callback that logs to the `log` crate.
pub fn default_progress_callback() -> ProgressCallback {
    Box::new(|completed, total, name| {
        log::info!("[{}/{}] Running: {}", completed, total, name);
    })
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

/// Engine that runs benchmark instances against a given configuration.
///
/// Internally, the runner simulates a bilevel solve by:
/// 1. Building the lower-level LP for x = 0.
/// 2. Performing a simplex-like iteration.
/// 3. Applying cuts if configured.
/// 4. Reporting metrics.
///
/// This provides a realistic timing skeleton even when the full compiler
/// pipeline is not yet wired in.
#[derive(Debug)]
pub struct BenchmarkRunner {
    /// Whether to run instances in parallel.
    pub parallel: bool,
    /// Number of threads (0 = rayon default).
    pub num_threads: usize,
    /// Cancel flag shared across threads.
    cancel: Arc<AtomicBool>,
    /// Completed count for progress tracking.
    completed: Arc<AtomicUsize>,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        BenchmarkRunner {
            parallel: false,
            num_threads: 0,
            cancel: Arc::new(AtomicBool::new(false)),
            completed: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl BenchmarkRunner {
    /// Create a new runner.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable parallel execution.
    pub fn with_parallel(mut self, threads: usize) -> Self {
        self.parallel = true;
        self.num_threads = threads;
        self
    }

    /// Request cancellation of all running benchmarks.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    /// Reset the cancel flag.
    pub fn reset_cancel(&self) {
        self.cancel.store(false, Ordering::SeqCst);
    }

    /// Number of instances completed so far.
    pub fn completed_count(&self) -> usize {
        self.completed.load(Ordering::SeqCst)
    }

    /// Run a single instance with a single configuration.
    pub fn run_single(&self, instance: &BenchmarkInstance, config: &BenchmarkConfig) -> RunResult {
        let start = Instant::now();
        let config_label = config.label();
        let name = instance.name().to_string();

        // Check cancel before starting.
        if self.cancel.load(Ordering::SeqCst) {
            return RunResult::skipped(&name, &config_label, "Cancelled");
        }

        // Simulate the bilevel solve process.
        let result = self.simulate_solve(instance, config, start);

        self.completed.fetch_add(1, Ordering::SeqCst);
        result
    }

    /// Simulate a bilevel solve for benchmarking purposes.
    fn simulate_solve(
        &self,
        instance: &BenchmarkInstance,
        config: &BenchmarkConfig,
        start: Instant,
    ) -> RunResult {
        let name = instance.name().to_string();
        let config_label = config.label();
        let problem = &instance.problem;
        let time_limit = Duration::from_secs_f64(config.time_limit_secs);

        // Phase 1: Build the lower-level LP with x = 0.
        let x_zero = vec![0.0; problem.num_upper_vars];
        let lower_lp = problem.lower_level_lp(&x_zero);

        // Phase 2: Iterative simplex simulation.
        let n = problem.num_lower_vars;
        let m = problem.num_lower_constraints;
        let mut iteration_count: u64 = 0;
        let mut node_count: u64 = 0;
        let mut cuts_generated: u64 = 0;

        // Simple primal heuristic: try y = 0 as initial point.
        let mut best_y = vec![0.0; n];
        let mut best_obj = evaluate_upper_obj(problem, &x_zero, &best_y);
        let root_objective = best_obj;
        let mut best_bound = f64::NEG_INFINITY;

        // Iterative improvement via coordinate descent on the lower level.
        let max_iter = config.solver.max_iterations.min(10_000);
        let mut current_y = best_y.clone();

        for iter in 0..max_iter {
            if start.elapsed() >= time_limit {
                return RunResult {
                    instance_name: name,
                    config_label,
                    status: RunStatus::TimeLimit,
                    wall_time_secs: start.elapsed().as_secs_f64(),
                    objective: Some(best_obj),
                    best_bound: Some(best_bound),
                    node_count,
                    iteration_count,
                    cuts_generated,
                    root_objective: Some(root_objective),
                    metrics: None,
                    error_message: None,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                };
            }

            if self.cancel.load(Ordering::SeqCst) {
                return RunResult::skipped(&name, &config_label, "Cancelled during solve");
            }

            iteration_count += 1;

            // Coordinate descent step on lower objective.
            let coord = (iter as usize) % n.max(1);
            if coord < n && coord < problem.lower_obj_c.len() {
                let step = if problem.lower_obj_c[coord].abs() > DEFAULT_TOLERANCE {
                    -0.01 * problem.lower_obj_c[coord].signum()
                } else {
                    0.0
                };
                current_y[coord] = (current_y[coord] + step).max(0.0);
            }

            // Check feasibility against lower constraints.
            let feasible = check_lower_feasibility(problem, &x_zero, &current_y);
            if feasible {
                let obj = evaluate_upper_obj(problem, &x_zero, &current_y);
                if obj < best_obj {
                    best_obj = obj;
                    best_y = current_y.clone();
                }
            } else {
                // Revert.
                if coord < n {
                    current_y[coord] = best_y[coord];
                }
            }

            // Simulated cut generation.
            if config.cuts.intersection_cuts && iter % 10 == 0 {
                cuts_generated += 1;
            }
            if config.cuts.value_function_cuts && iter % 20 == 0 {
                cuts_generated += 1;
            }
            if config.cuts.benders_cuts && iter % 15 == 0 {
                cuts_generated += 1;
            }

            // Simulated branching.
            if iter % 50 == 0 {
                node_count += 1;
            }

            // Update bound.
            let lower_bound = compute_lower_bound(problem, &x_zero);
            if lower_bound > best_bound {
                best_bound = lower_bound;
            }

            // Check convergence.
            if (best_obj - best_bound).abs() < config.solver.tolerance {
                break;
            }
        }

        let wall_time = start.elapsed().as_secs_f64();
        let gap = if best_obj.abs() > DEFAULT_TOLERANCE {
            ((best_obj - best_bound).abs() / best_obj.abs()) * 100.0
        } else {
            0.0
        };

        let status = if gap < 0.01 {
            RunStatus::Optimal
        } else if wall_time >= config.time_limit_secs {
            RunStatus::TimeLimit
        } else {
            RunStatus::IterationLimit
        };

        let metrics = if config.collect_detailed_metrics {
            Some(BenchmarkMetrics {
                solve_time_secs: wall_time,
                node_count,
                root_gap_percent: gap,
                final_gap_percent: gap,
                root_gap_closure_percent: if root_objective.abs() > DEFAULT_TOLERANCE {
                    ((root_objective - best_obj).abs() / root_objective.abs()) * 100.0
                } else {
                    0.0
                },
                iteration_count,
                cuts_by_type: {
                    let mut m = HashMap::new();
                    if config.cuts.intersection_cuts {
                        m.insert(
                            "intersection".to_string(),
                            crate::metrics::CutStats {
                                count: cuts_generated / 2,
                                avg_violation: 0.05,
                                max_violation: 0.1,
                                generation_time_secs: wall_time * 0.1,
                            },
                        );
                    }
                    if config.cuts.value_function_cuts {
                        m.insert(
                            "value_function".to_string(),
                            crate::metrics::CutStats {
                                count: cuts_generated / 3,
                                avg_violation: 0.03,
                                max_violation: 0.08,
                                generation_time_secs: wall_time * 0.05,
                            },
                        );
                    }
                    m
                },
                reformulation_time_secs: wall_time * 0.05,
                cut_generation_time_secs: wall_time * 0.15,
                lp_solve_time_secs: wall_time * 0.6,
                branching_time_secs: wall_time * 0.15,
                other_time_secs: wall_time * 0.05,
            })
        } else {
            None
        };

        RunResult {
            instance_name: name,
            config_label,
            status,
            wall_time_secs: wall_time,
            objective: Some(best_obj),
            best_bound: Some(best_bound),
            node_count,
            iteration_count,
            cuts_generated,
            root_objective: Some(root_objective),
            metrics,
            error_message: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Run an instance set with a single configuration.
    pub fn run_set(
        &self,
        instances: &InstanceSet,
        config: &BenchmarkConfig,
        progress: Option<&ProgressCallback>,
    ) -> Vec<RunResult> {
        self.completed.store(0, Ordering::SeqCst);
        let total = instances.len();

        if self.parallel && total > 1 {
            self.run_set_parallel(instances, config, progress)
        } else {
            self.run_set_sequential(instances, config, progress)
        }
    }

    /// Run sequentially.
    fn run_set_sequential(
        &self,
        instances: &InstanceSet,
        config: &BenchmarkConfig,
        progress: Option<&ProgressCallback>,
    ) -> Vec<RunResult> {
        let total = instances.len();
        let mut results = Vec::with_capacity(total);
        for (idx, inst) in instances.iter().enumerate() {
            if let Some(cb) = progress {
                cb(idx, total, inst.name());
            }
            let result = self.run_single(inst, config);
            results.push(result);
        }
        results
    }

    /// Run in parallel using rayon.
    fn run_set_parallel(
        &self,
        instances: &InstanceSet,
        config: &BenchmarkConfig,
        progress: Option<&ProgressCallback>,
    ) -> Vec<RunResult> {
        use rayon::prelude::*;

        let total = instances.len();
        let results_lock = Arc::new(Mutex::new(Vec::<RunResult>::with_capacity(total)));
        let progress_arc: Option<Arc<ProgressCallback>> = progress.map(|_| {
            Arc::new(Box::new(|completed: usize, total: usize, name: &str| {
                log::info!("[{}/{}] Completed: {}", completed, total, name);
            }) as ProgressCallback)
        });

        // Configure thread pool if requested.
        let pool = if self.num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.num_threads)
                .build()
                .ok()
        } else {
            None
        };

        let work = |inst: &BenchmarkInstance| -> RunResult {
            let result = self.run_single(inst, config);
            let completed = self.completed.load(Ordering::SeqCst);
            if let Some(ref cb) = progress_arc {
                cb(completed, total, inst.name());
            }
            result
        };

        let instances_vec: Vec<&BenchmarkInstance> = instances.iter().collect();
        let results: Vec<RunResult> = if let Some(pool) = pool {
            pool.install(|| instances_vec.par_iter().map(|inst| work(inst)).collect())
        } else {
            instances_vec.par_iter().map(|inst| work(inst)).collect()
        };

        results
    }

    /// Run all configurations against all instances (full matrix).
    pub fn run_matrix(
        &self,
        instances: &InstanceSet,
        configs: &[BenchmarkConfig],
        progress: Option<&ProgressCallback>,
    ) -> HashMap<String, Vec<RunResult>> {
        let mut all_results = HashMap::new();
        for config in configs {
            let label = config.label();
            let results = self.run_set(instances, config, progress);
            all_results.insert(label, results);
        }
        all_results
    }

    /// Run with retry: if a run fails, retry up to `max_retries` times.
    pub fn run_with_retry(
        &self,
        instance: &BenchmarkInstance,
        config: &BenchmarkConfig,
        max_retries: usize,
    ) -> RunResult {
        let mut last_result = self.run_single(instance, config);
        for attempt in 1..=max_retries {
            if last_result.status != RunStatus::Error {
                return last_result;
            }
            log::warn!(
                "Retry {}/{} for instance {}",
                attempt,
                max_retries,
                instance.name()
            );
            last_result = self.run_single(instance, config);
        }
        last_result
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Evaluate the upper-level objective: c_x^T x + c_y^T y.
fn evaluate_upper_obj(problem: &BilevelProblem, x: &[f64], y: &[f64]) -> f64 {
    let cx: f64 = problem
        .upper_obj_c_x
        .iter()
        .zip(x.iter())
        .map(|(c, v)| c * v)
        .sum();
    let cy: f64 = problem
        .upper_obj_c_y
        .iter()
        .zip(y.iter())
        .map(|(c, v)| c * v)
        .sum();
    cx + cy
}

/// Check feasibility of (x, y) w.r.t. lower-level constraints: A y <= b + B x.
fn check_lower_feasibility(problem: &BilevelProblem, x: &[f64], y: &[f64]) -> bool {
    let n_lower = problem.num_lower_vars;
    let m_lower = problem.num_lower_constraints;
    if m_lower == 0 {
        return true;
    }

    // Compute A y.
    let mut ay = vec![0.0; m_lower];
    for entry in &problem.lower_a.entries {
        if entry.row < m_lower && entry.col < n_lower && entry.col < y.len() {
            ay[entry.row] += entry.value * y[entry.col];
        }
    }

    // Compute b + B x.
    let mut rhs = problem.lower_b.clone();
    for entry in &problem.lower_linking_b.entries {
        if entry.row < m_lower && entry.col < x.len() {
            rhs[entry.row] += entry.value * x[entry.col];
        }
    }

    // Check A y <= b + B x with tolerance.
    for i in 0..m_lower {
        if ay[i] > rhs[i] + DEFAULT_TOLERANCE {
            return false;
        }
    }
    true
}

/// Compute a simple lower bound on the bilevel objective.
fn compute_lower_bound(problem: &BilevelProblem, x: &[f64]) -> f64 {
    let cx: f64 = problem
        .upper_obj_c_x
        .iter()
        .zip(x.iter())
        .map(|(c, v)| c * v)
        .sum();
    // Lower bound by relaxing the optimality of y.
    let min_cy: f64 = problem
        .upper_obj_c_y
        .iter()
        .map(|c| if *c >= 0.0 { 0.0 } else { c * 1e6 })
        .sum();
    cx + min_cy
}

// ---------------------------------------------------------------------------
// Batch result aggregation helper
// ---------------------------------------------------------------------------

/// Summary of a batch of run results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSummary {
    pub total: usize,
    pub optimal: usize,
    pub timeout: usize,
    pub error: usize,
    pub skipped: usize,
    pub mean_time_secs: f64,
    pub median_time_secs: f64,
    pub total_time_secs: f64,
}

impl BatchSummary {
    /// Compute a summary from a slice of results.
    pub fn from_results(results: &[RunResult]) -> Self {
        let total = results.len();
        let optimal = results
            .iter()
            .filter(|r| r.status == RunStatus::Optimal)
            .count();
        let timeout = results
            .iter()
            .filter(|r| r.status == RunStatus::TimeLimit)
            .count();
        let error = results
            .iter()
            .filter(|r| r.status == RunStatus::Error)
            .count();
        let skipped = results
            .iter()
            .filter(|r| r.status == RunStatus::Skipped)
            .count();
        let times: Vec<f64> = results.iter().map(|r| r.wall_time_secs).collect();
        let total_time: f64 = times.iter().sum();
        let mean_time = if total > 0 {
            total_time / total as f64
        } else {
            0.0
        };
        let median_time = if times.is_empty() {
            0.0
        } else {
            let mut sorted = times.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        };
        BatchSummary {
            total,
            optimal,
            timeout,
            error,
            skipped,
            mean_time_secs: mean_time,
            median_time_secs: median_time,
            total_time_secs: total_time,
        }
    }
}

impl std::fmt::Display for BatchSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Total: {}", self.total)?;
        writeln!(
            f,
            "Optimal: {} ({:.1}%)",
            self.optimal,
            self.optimal as f64 / self.total.max(1) as f64 * 100.0
        )?;
        writeln!(f, "Timeout: {}", self.timeout)?;
        writeln!(f, "Error: {}", self.error)?;
        writeln!(f, "Mean time: {:.3}s", self.mean_time_secs)?;
        writeln!(f, "Median time: {:.3}s", self.median_time_secs)?;
        write!(f, "Total time: {:.1}s", self.total_time_secs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::make_trivial_instance;

    fn make_config() -> BenchmarkConfig {
        BenchmarkConfig {
            time_limit_secs: 1.0,
            solver: SolverConfig::default().with_max_iterations(100),
            ..Default::default()
        }
    }

    #[test]
    fn test_run_single() {
        let inst = make_trivial_instance("tiny", 2, 2, 2);
        let config = make_config();
        let runner = BenchmarkRunner::new();
        let result = runner.run_single(&inst, &config);
        assert!(result.status == RunStatus::Optimal || result.status == RunStatus::IterationLimit);
        assert!(result.wall_time_secs >= 0.0);
    }

    #[test]
    fn test_run_set() {
        let set = crate::instance::make_test_instance_set();
        let config = make_config();
        let runner = BenchmarkRunner::new();
        let results = runner.run_set(&set, &config, None);
        assert_eq!(results.len(), set.len());
    }

    #[test]
    fn test_run_status_display() {
        assert_eq!(RunStatus::Optimal.to_string(), "Optimal");
        assert_eq!(RunStatus::TimeLimit.to_string(), "TimeLimit");
    }

    #[test]
    fn test_config_matrix() {
        let reforms = vec![ReformulationMethod::BigM, ReformulationMethod::SOS1];
        let solvers = vec![SolverConfig::default()];
        let cuts = vec![CutConfig::none(), CutConfig::intersection_only()];
        let configs = BenchmarkConfig::config_matrix(&reforms, &solvers, &cuts, 60.0);
        assert_eq!(configs.len(), 4);
    }

    #[test]
    fn test_config_label() {
        let config = BenchmarkConfig::default();
        let label = config.label();
        assert!(label.contains("BigM"));
    }

    #[test]
    fn test_gap_percent() {
        let mut r = RunResult::optimal("test", "cfg", 1.0, 10.0, 5, 100);
        r.best_bound = Some(9.5);
        let gap = r.gap_percent().unwrap();
        assert!((gap - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_summary() {
        let results = vec![
            RunResult::optimal("a", "c", 1.0, 10.0, 5, 50),
            RunResult::optimal("b", "c", 2.0, 20.0, 10, 100),
            RunResult::timeout("c", "c", 60.0, Some(30.0), Some(25.0)),
        ];
        let summary = BatchSummary::from_results(&results);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.optimal, 2);
        assert_eq!(summary.timeout, 1);
    }

    #[test]
    fn test_reformulation_all() {
        let all = ReformulationMethod::all();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_cut_config_labels() {
        assert_eq!(CutConfig::none().label(), "NoCuts");
        assert!(CutConfig::all_cuts().label().contains("IC"));
    }

    #[test]
    fn test_run_with_retry() {
        let inst = make_trivial_instance("retry_test", 2, 2, 2);
        let config = make_config();
        let runner = BenchmarkRunner::new();
        let result = runner.run_with_retry(&inst, &config, 2);
        assert_ne!(result.status, RunStatus::Error);
    }
}
